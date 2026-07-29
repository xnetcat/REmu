// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Stage interface emission: the vertex and pixel function signatures, pixel
//! input materialization and the compute system register preamble.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use remu_shader::PixelOutputKind;

use crate::compiler::module::contains_register_reference;
use crate::{Compiler, MetalShaderStage};

pub(crate) const fn vertex_clip_distance_mask(control: u32) -> u32 {
    let low = if control & (1 << 22) != 0 {
        control & 0x0f
    } else {
        0
    };
    let high = if control & (1 << 23) != 0 {
        control & 0xf0
    } else {
        0
    };
    low | high
}

impl Compiler<'_> {
    pub(crate) fn emit_compute_system_registers(&self, source: &mut String) {
        let Some(registers) = self.evaluation.compute_system_registers else {
            return;
        };
        for (register, component) in [
            (registers.work_group_x, "x"),
            (registers.work_group_y, "y"),
            (registers.work_group_z, "z"),
        ] {
            if let Some(register) = register
                && contains_register_reference(&self.body, 's', register)
            {
                let _ = writeln!(source, "    s[{register}] = remu_group_id.{component};");
            }
        }
        if let Some(register) = registers.thread_group_size
            && contains_register_reference(&self.body, 's', register)
        {
            let wave_shift = if self.options.wave_lane_count == 64 {
                6
            } else {
                5
            };
            let wave_mask = self.options.wave_lane_count - 1;
            let _ = writeln!(
                source,
                "    uint remu_wave_id = remu_thread_index >> {wave_shift}u;\n\
                 \x20   uint remu_threadgroup_threads = remu_group_size.x * remu_group_size.y * remu_group_size.z;\n\
                 \x20   uint remu_threadgroup_waves = (remu_threadgroup_threads + {wave_mask}u) >> {wave_shift}u;"
            );
            // Ordinary compute dispatches have a zero ordered-append term. The remaining fields
            // match the RDNA2 initial-wave SGPR layout: FIRST_WAVE[31], WAVE_ID[24:20], and the
            // number of architectural waves in this threadgroup in [5:0].
            let _ = writeln!(
                source,
                "    s[{register}] = (remu_wave_id == 0u ? 0x80000000u : 0u) | ((remu_wave_id & 31u) << 20u) | (remu_threadgroup_waves & 63u);"
            );
        }
    }

    pub(crate) fn emit_vertex_interface(&self, source: &mut String) {
        if !self.evaluation.vertex_inputs.is_empty() {
            source.push_str("struct RemuVsIn {\n");
            let mut locations = BTreeSet::new();
            for input in &self.evaluation.vertex_inputs {
                if locations.insert(input.location) {
                    let ty = if input.component_count == 1 {
                        "float".to_owned()
                    } else {
                        format!("float{}", input.component_count)
                    };
                    let _ = writeln!(
                        source,
                        "    {ty} in{} [[attribute({})]];",
                        input.location, input.location
                    );
                }
            }
            source.push_str("};\n\n");
        }
        source.push_str("struct RemuVsOut {\n    float4 position [[position]];\n");
        if vertex_clip_distance_mask(self.options.vertex_output_control) != 0 {
            // Metal has no cull-distance vertex output. Preserve native clip-distance semantics
            // here; treating cull distances as clip distances would incorrectly reject a
            // primitive when only one of its vertices is outside the cull plane.
            source.push_str("    float clip_distances [[clip_distance]] [8];\n");
        }
        if self.options.render_target_array_length > 1
            && self.options.stage != MetalShaderStage::Mesh
        {
            source.push_str("    uint render_target_array_index [[render_target_array_index]];\n");
        }
        for location in &self.vertex_outputs {
            let _ = writeln!(
                source,
                "    float4 param{location} [[user(locn{location})]];"
            );
        }
        source.push_str("};\n\n");
        if self.options.render_target_array_length > 1
            && self.options.stage == MetalShaderStage::Mesh
        {
            // Metal mesh shaders route the target slice per primitive. Declaring this semantic
            // on the mesh vertex type is rejected by Apple's compiler even though the same
            // semantic belongs to an ordinary vertex-stage output.
            source.push_str(
                "struct RemuPrimitiveOut {\n\
                 \x20   uint render_target_array_index [[render_target_array_index]];\n\
                 };\n\n",
            );
        }
    }

    pub(crate) fn emit_pixel_interface(&self, source: &mut String) {
        source.push_str("struct RemuPsIn {\n    float4 frag_coord [[position]];\n");
        if self.options.render_target_array_length > 1 {
            source.push_str("    uint render_target_array_index [[render_target_array_index]];\n");
        }
        for attribute in &self.pixel_attributes {
            let input = self.options.pixel_input_controls[*attribute as usize];
            if input.uses_default() {
                continue;
            }
            let interpolation = self
                .pixel_interpolations
                .get(attribute)
                .copied()
                .unwrap_or(PixelInterpolation::CenterPerspective);
            let _ = writeln!(
                source,
                "    float4 attr{attribute} [[user(locn{}), {}]];",
                input.input_offset,
                interpolation.msl_attribute()
            );
        }
        source.push_str("};\n\nstruct RemuPsOut {\n");
        for output in &self.options.pixel_outputs {
            let ty = match output.kind {
                PixelOutputKind::Float => "float4",
                PixelOutputKind::Uint => "uint4",
                PixelOutputKind::Sint => "int4",
            };
            let _ = writeln!(
                source,
                "    {ty} mrt{} [[color({})]];",
                output.guest_slot, output.host_location
            );
        }
        source.push_str("};\n\n");
    }

    pub(crate) fn emit_pixel_input_state(&self, source: &mut String) {
        let mut vgpr = 0_u32;
        for (bit, count) in [
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 3),
            (4, 2),
            (5, 2),
            (6, 2),
            (7, 1),
        ] {
            if self.options.pixel_input_address & (1 << bit) != 0 {
                vgpr += count;
            }
        }
        for (bit, component) in [(8, "x"), (9, "y"), (10, "z"), (11, "w")] {
            let mask = 1_u32 << bit;
            if self.options.pixel_input_address & mask != 0 {
                if self.options.pixel_input_enable & mask != 0
                    && contains_register_reference(&self.body, 'v', vgpr)
                {
                    let _ = writeln!(
                        source,
                        "    remu_v_{vgpr} = as_type<uint>(remu_in.frag_coord.{component});"
                    );
                }
                vgpr += 1;
            }
        }
        for bit in 12..=15 {
            let mask = 1_u32 << bit;
            if self.options.pixel_input_address & mask == 0 {
                continue;
            }
            if bit == 13
                && self.options.pixel_input_enable & mask != 0
                && self.options.render_target_array_length > 1
                && contains_register_reference(&self.body, 'v', vgpr)
            {
                // GFX10 ANCILLARY packs the render-target array index into bits 28:16. Metal
                // exposes the exact layer selected by the vertex-stage system value to the
                // fragment stage, so preserve the architectural VGPR layout instead of leaving
                // every volume slice at the zero-initialized layer.
                let _ = writeln!(
                    source,
                    "    remu_v_{vgpr} = remu_in.render_target_array_index << 16u;"
                );
            }
            // SPI_PS_INPUT_ADDR controls compact VGPR allocation independently of INPUT_ENA.
            // Reserve every addressed system input even while its host semantic is not enabled.
            vgpr += 1;
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PixelInterpolation {
    CenterPerspective,
    CentroidPerspective,
    SamplePerspective,
    CenterNoPerspective,
    CentroidNoPerspective,
    SampleNoPerspective,
    Flat,
}

impl PixelInterpolation {
    const fn msl_attribute(self) -> &'static str {
        match self {
            Self::CenterPerspective => "center_perspective",
            Self::CentroidPerspective => "centroid_perspective",
            Self::SamplePerspective => "sample_perspective",
            Self::CenterNoPerspective => "center_no_perspective",
            Self::CentroidNoPerspective => "centroid_no_perspective",
            Self::SampleNoPerspective => "sample_no_perspective",
            Self::Flat => "flat",
        }
    }
}

pub(crate) fn interpolation_for_barycentric_vgpr(
    pixel_input_address: u32,
    register: u32,
) -> Result<PixelInterpolation, String> {
    // VGPRs are allocated in SPI_PS_INPUT_ADDR order. Each enabled interpolation mode contributes
    // its I/J pair; pull-model contributes I/W, J/W, and 1/W, and line stipple contributes one.
    // With no barycentric address bits, retain Metal's historical/default center-perspective mode
    // for API compatibility with callers that have not supplied register state yet.
    if pixel_input_address & 0xff == 0 {
        return Ok(PixelInterpolation::CenterPerspective);
    }
    let mut first_register = 0_u32;
    for (bit, count, interpolation) in [
        (0, 2, Some(PixelInterpolation::SamplePerspective)),
        (1, 2, Some(PixelInterpolation::CenterPerspective)),
        (2, 2, Some(PixelInterpolation::CentroidPerspective)),
        (3, 3, None),
        (4, 2, Some(PixelInterpolation::SampleNoPerspective)),
        (5, 2, Some(PixelInterpolation::CenterNoPerspective)),
        (6, 2, Some(PixelInterpolation::CentroidNoPerspective)),
        (7, 1, None),
    ] {
        if pixel_input_address & (1 << bit) == 0 {
            continue;
        }
        if (first_register..first_register + count).contains(&register) {
            return interpolation.ok_or_else(|| {
                format!(
                    "V_INTERP source v{register} selects pull-model or line-stipple state, which \
                     requires explicit interpolation"
                )
            });
        }
        first_register += count;
    }
    Err(format!(
        "V_INTERP source v{register} is not an enabled barycentric VGPR in \
         SPI_PS_INPUT_ADDR={pixel_input_address:#010x}"
    ))
}
