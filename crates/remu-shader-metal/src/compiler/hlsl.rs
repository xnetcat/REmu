// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Direct3D 12 module assembly around the shared Gen5 instruction body.

use std::fmt::Write as _;

use remu_shader::{InstructionControl, PixelOutputKind, image_is_comparison};

use crate::compiler::control_flow::ControlFlowLowering;
use crate::compiler::image::{ImageKind, metal_image_dimension};
use crate::compiler::interface::PixelInterpolation;
use crate::compiler::memory::{is_format_buffer_memory, is_lds_memory_instruction};
use crate::compiler::module::{contains_register_reference, scalarize_register_state};
use crate::compiler::prelude::{
    emit_div_fixup_f32_prelude, emit_format_prelude, emit_vop3p_prelude,
};
use crate::{
    Compiler, HlslBindingLayout, HlslShader, MetalCompileError, MetalRegisterFootprint,
    MetalShaderStage,
};

impl Compiler<'_> {
    pub(crate) fn compile_hlsl(
        mut self,
        threads_per_group: [u32; 3],
    ) -> Result<HlslShader, MetalCompileError> {
        validate_hlsl_interface(&self, threads_per_group)?;
        self.lower_body()?;
        let direct_control_flow = matches!(
            self.control_flow,
            ControlFlowLowering::Direct(_) | ControlFlowLowering::Structured { .. }
        );
        let (source, references) = self.emit_hlsl_module(threads_per_group)?;
        let global_buffer_count = u32::try_from(self.evaluation.global_memory_bindings.len())
            .map_err(|_| MetalCompileError::InvalidInterface("too many global buffers".into()))?;
        let binding_table_words = 1_u32
            .checked_add(global_buffer_count)
            .and_then(|value| value.checked_add(self.image_slots.texture_count))
            .and_then(|value| value.checked_add(self.image_slots.sampler_count))
            .ok_or_else(|| {
                MetalCompileError::InvalidInterface("binding table is too large".into())
            })?;
        let threadgroup_memory_bytes = if self
            .program
            .instructions
            .iter()
            .any(is_lds_memory_instruction)
        {
            self.options.lds_size_bytes
        } else {
            0
        };
        let target_prefix = match self.options.stage {
            MetalShaderStage::Vertex => "vs",
            MetalShaderStage::Pixel => "ps",
            MetalShaderStage::Compute => "cs",
            MetalShaderStage::Mesh => "ms",
        };
        let target_version = if self.options.stage == MetalShaderStage::Pixel {
            "6_7"
        } else {
            "6_6"
        };
        let mut interpolated_attribute_locations = self
            .pixel_attributes
            .iter()
            .filter_map(|attribute| {
                let input = self.options.pixel_input_controls[*attribute as usize];
                (!input.uses_default()).then_some(u32::from(input.input_offset))
            })
            .collect::<Vec<_>>();
        interpolated_attribute_locations.sort_unstable();
        interpolated_attribute_locations.dedup();
        let interpolated_attribute_count = interpolated_attribute_locations
            .last()
            .map_or(0, |location| location + 1);
        Ok(HlslShader {
            source,
            entry_point: "remu_main".to_owned(),
            target_profile: format!("{target_prefix}_{target_version}"),
            stage: self.options.stage,
            bindings: HlslBindingLayout {
                root_constant_words: 7,
                binding_table_words,
                global_buffer_count,
                texture_count: self.image_slots.texture_count,
                sampler_count: self.image_slots.sampler_count,
                threadgroup_memory_bytes,
            },
            vertex_inputs: self.evaluation.vertex_inputs.clone(),
            interpolated_attribute_count,
            interpolated_attribute_locations,
            register_footprint: MetalRegisterFootprint {
                scalar_values: u32::try_from(references.scalar.len()).map_err(|_| {
                    MetalCompileError::InvalidInterface("too many live scalar values".into())
                })?,
                vector_values: u32::try_from(references.vector.len()).map_err(|_| {
                    MetalCompileError::InvalidInterface("too many live vector values".into())
                })?,
                direct_control_flow,
            },
        })
    }

    fn emit_hlsl_module(
        &self,
        threads_per_group: [u32; 3],
    ) -> Result<(String, super::module::RegisterReferences), MetalCompileError> {
        let mut source = String::new();
        let lds_words = self
            .program
            .instructions
            .iter()
            .any(is_lds_memory_instruction)
            .then_some(self.options.lds_size_bytes / 4);
        if let Some(words) = lds_words {
            let _ = writeln!(source, "groupshared uint remu_lds[{words}];\n");
        }
        if self.split_wave64() {
            if self
                .program
                .instructions
                .iter()
                .any(|instruction| instruction.opcode.is("VReadfirstlaneB32"))
            {
                let slots = self.disjoint_readfirstlane_slot_count().unwrap_or(32);
                let _ = writeln!(
                    source,
                    "groupshared uint remu_readfirstlane_values[{slots}];\n\
                     groupshared uint remu_readfirstlane_active[{slots}];"
                );
            }
            if self
                .program
                .instructions
                .iter()
                .any(|instruction| instruction.opcode.is("VReadlaneB32"))
            {
                source.push_str("groupshared uint remu_readlane_values[1024];\n");
            }
            source.push('\n');
        }
        emit_hlsl_prelude(&mut source, lds_words.is_some());
        if self.program.instructions.iter().any(|instruction| {
            matches!(instruction.control, Some(InstructionControl::Vop3p(_)))
                || instruction.opcode.is("VCvtPkrtzF16F32")
        }) {
            let mut prelude = String::new();
            emit_vop3p_prelude(&mut prelude);
            source.push_str(&translate_common_source(prelude));
        }
        if self
            .program
            .instructions
            .iter()
            .any(is_format_buffer_memory)
        {
            let mut prelude = String::new();
            emit_format_prelude(&mut prelude);
            source.push_str(&translate_format_prelude(prelude));
        }
        if self
            .program
            .instructions
            .iter()
            .any(|instruction| instruction.opcode.is("VDivFixupF32"))
        {
            let mut prelude = String::new();
            emit_div_fixup_f32_prelude(&mut prelude);
            source.push_str(&translate_common_source(prelude));
        }

        self.emit_hlsl_interfaces(&mut source);
        let root_space = match self.options.stage {
            MetalShaderStage::Vertex => 0,
            MetalShaderStage::Pixel => 1,
            MetalShaderStage::Compute => 2,
            MetalShaderStage::Mesh => unreachable!(),
        };
        let _ = writeln!(
            source,
            "cbuffer RemuRootConstants : register(b0, space{root_space}) {{\n\
             \x20   uint remu_binding_table_descriptor;\n\
             \x20   uint3 remu_dispatch_limit;\n\
             \x20   uint3 remu_dispatch_base_group;\n\
             }};\n"
        );

        let signature = match self.options.stage {
            MetalShaderStage::Vertex => {
                format!(
                    "[RootSignature(REMU_ROOT_SIGNATURE)]\nRemuVsOut remu_main(uint remu_vertex_id : SV_VertexID, uint remu_instance_id : SV_InstanceID)"
                )
            }
            MetalShaderStage::Pixel if self.options.pixel_outputs.is_empty()
                && !self.options.pixel_depth_export =>
            {
                "[WaveOpsIncludeHelperLanes]\n[RootSignature(REMU_ROOT_SIGNATURE)]\nvoid remu_main(RemuPsIn remu_in)".to_owned()
            }
            MetalShaderStage::Pixel => {
                "[WaveOpsIncludeHelperLanes]\n[RootSignature(REMU_ROOT_SIGNATURE)]\nRemuPsOut remu_main(RemuPsIn remu_in)".to_owned()
            }
            MetalShaderStage::Compute => format!(
                "[RootSignature(REMU_ROOT_SIGNATURE)]\n[numthreads({}, {}, {})]\nvoid remu_main(uint3 remu_local_id : SV_GroupThreadID, uint3 remu_native_group_id : SV_GroupID, uint3 remu_dispatch_id : SV_DispatchThreadID, uint remu_thread_index : SV_GroupIndex)",
                threads_per_group[0], threads_per_group[1], threads_per_group[2]
            ),
            MetalShaderStage::Mesh => unreachable!("mesh shaders are rejected by validation"),
        };
        let _ = writeln!(source, "{signature} {{");
        source.push_str(
            "    StructuredBuffer<uint> remu_binding_table = ResourceDescriptorHeap[remu_binding_table_descriptor];\n\
             \x20   StructuredBuffer<uint> remu_initial_s = ResourceDescriptorHeap[remu_binding_table[0]];\n",
        );
        for index in 0..self.evaluation.global_memory_bindings.len() {
            let _ = writeln!(
                source,
                "    uint b{index} = remu_binding_table[{}u];",
                index + 1
            );
        }
        let texture_base = 1 + self.evaluation.global_memory_bindings.len();
        for (index, kind) in self.image_kinds.iter().copied().enumerate() {
            let _ = writeln!(
                source,
                "    {} tex{index} = ResourceDescriptorHeap[remu_binding_table[{}u]];",
                hlsl_texture_type(kind),
                texture_base + index
            );
        }
        let sampler_base = texture_base + self.image_kinds.len();
        for slot in 0..self.image_slots.sampler_count {
            let ordinary =
                self.evaluation
                    .image_bindings
                    .iter()
                    .enumerate()
                    .any(|(index, binding)| {
                        self.image_slots.sampler_slots[index] == Some(slot)
                            && !image_is_comparison(binding.opcode)
                    });
            let comparison =
                self.evaluation
                    .image_bindings
                    .iter()
                    .enumerate()
                    .any(|(index, binding)| {
                        self.image_slots.sampler_slots[index] == Some(slot)
                            && image_is_comparison(binding.opcode)
                    });
            if ordinary {
                let _ = writeln!(
                    source,
                    "    SamplerState remu_sampler_{slot} = SamplerDescriptorHeap[remu_binding_table[{}u]];",
                    sampler_base + slot as usize
                );
            }
            if comparison {
                let _ = writeln!(
                    source,
                    "    SamplerComparisonState remu_compare_sampler_{slot} = SamplerDescriptorHeap[remu_binding_table[{}u]];",
                    sampler_base + slot as usize
                );
            }
        }
        source.push_str("    /* REMU_REGISTER_STATE */\n");
        source.push_str("    bool exec = true;\n    bool vcc = false;\n    bool scc = false;\n");

        match self.options.stage {
            MetalShaderStage::Vertex => {
                source.push_str("    uint remu_lane = WavePrefixSum(1u);\n");
                self.emit_hlsl_graphics_lane_storage(&mut source);
                source.push_str("    RemuVsOut remu_out = (RemuVsOut)0;\n");
                if self.options.render_target_array_length > 1 {
                    let _ = writeln!(
                        source,
                        "    remu_out.render_target_array_index = min(remu_instance_id, {}u);",
                        self.options.render_target_array_length - 1
                    );
                }
                if contains_register_reference(&self.body, 'v', 5) {
                    source.push_str("    remu_v_5 = remu_vertex_id;\n");
                }
                if contains_register_reference(&self.body, 'v', 8) {
                    source.push_str("    remu_v_8 = remu_instance_id;\n");
                }
            }
            MetalShaderStage::Pixel => {
                source.push_str("    uint remu_lane = WavePrefixSum(1u);\n");
                self.emit_hlsl_graphics_lane_storage(&mut source);
                if !self.options.pixel_outputs.is_empty() || self.options.pixel_depth_export {
                    source.push_str("    RemuPsOut remu_out = (RemuPsOut)0;\n");
                    for output in &self.options.pixel_outputs {
                        let default = match output.kind {
                            PixelOutputKind::Float => "float4(0.0f, 0.0f, 0.0f, 1.0f)",
                            PixelOutputKind::Uint => "uint4(0u, 0u, 0u, 1u)",
                            PixelOutputKind::Sint => "int4(0, 0, 0, 1)",
                        };
                        let _ =
                            writeln!(source, "    remu_out.mrt{} = {default};", output.guest_slot);
                    }
                    if self.options.pixel_depth_export {
                        source.push_str("    remu_out.depth = remu_in.frag_coord.z;\n");
                    }
                }
                let mut input_state = String::new();
                self.emit_pixel_input_state(&mut input_state);
                source.push_str(&translate_common_source(input_state));
            }
            MetalShaderStage::Compute => {
                source.push_str(
                    "    if (all(remu_dispatch_limit != 0u) && any(remu_dispatch_id >= remu_dispatch_limit)) return;\n\
                     \x20   uint3 remu_group_id = remu_native_group_id + remu_dispatch_base_group;\n",
                );
                source.push_str("    uint remu_lane = WaveGetLaneIndex();\n");
                let _ = writeln!(
                    source,
                    "    uint3 remu_group_size = uint3({}u, {}u, {}u);",
                    threads_per_group[0], threads_per_group[1], threads_per_group[2]
                );
                for (register, component) in [(0, "x"), (1, "y"), (2, "z")] {
                    if contains_register_reference(&self.body, 'v', register) {
                        let _ =
                            writeln!(source, "    remu_v_{register} = remu_local_id.{component};");
                    }
                }
                if self.split_wave64() {
                    source.push_str(
                        "    uint remu_wave_index = remu_thread_index >> 6u;\n\
                         \x20   uint remu_wave_half = (remu_thread_index >> 5u) & 1u;\n\
                         \x20   uint remu_wave_lane = remu_lane + (remu_wave_half << 5u);\n",
                    );
                }
            }
            MetalShaderStage::Mesh => unreachable!(),
        }

        match self.control_flow {
            ControlFlowLowering::Forward => {
                source.push_str("    uint pc = 0u;\n    bool active = true;\n");
            }
            ControlFlowLowering::Dispatcher => {
                source
                    .push_str("    uint pc = 0u;\n    bool active = true;\n    uint steps = 0u;\n");
            }
            ControlFlowLowering::Direct(_) | ControlFlowLowering::Structured { .. } => {}
        }
        self.emit_compute_system_registers(&mut source);
        for register in [crate::VCC_LO, crate::VCC_HI] {
            if contains_register_reference(&self.body, 's', register) {
                let _ = writeln!(source, "    remu_s_{register} = 0u;");
            }
        }
        if contains_register_reference(&self.body, 's', crate::EXEC_LO)
            || contains_register_reference(&self.body, 's', crate::EXEC_HI)
        {
            let (low, high) = self.ballot_words("true");
            if contains_register_reference(&self.body, 's', crate::EXEC_LO) {
                let _ = writeln!(
                    source,
                    "    remu_s_{} = {};",
                    crate::EXEC_LO,
                    translate_common_source(low)
                );
            }
            if contains_register_reference(&self.body, 's', crate::EXEC_HI) {
                let _ = writeln!(
                    source,
                    "    remu_s_{} = {};",
                    crate::EXEC_HI,
                    translate_common_source(high)
                );
            }
        }

        let body = translate_hlsl_body(self.body.clone());
        match self.control_flow {
            ControlFlowLowering::Direct(_) | ControlFlowLowering::Structured { .. } => {
                source.push_str(&body);
            }
            ControlFlowLowering::Forward => source.push_str(&body),
            ControlFlowLowering::Dispatcher => {
                let _ = writeln!(
                    source,
                    "    while (active && steps < {}u) {{\n        switch (pc) {{",
                    self.dispatcher_step_limit()
                );
                source.push_str(&body);
                source.push_str("        default: active = false; break;\n        }\n    }\n");
            }
        }

        match self.options.stage {
            MetalShaderStage::Vertex => source.push_str("    return remu_out;\n"),
            MetalShaderStage::Pixel => {
                if !self.options.pixel_outputs.is_empty() || self.options.pixel_depth_export {
                    source.push_str("    if (!exec) discard;\n    return remu_out;\n");
                }
            }
            MetalShaderStage::Compute => {}
            MetalShaderStage::Mesh => unreachable!(),
        }
        source.push_str("}\n");
        scalarize_register_state(&source)
    }

    fn emit_hlsl_interfaces(&self, source: &mut String) {
        source.push_str(
            "#define REMU_ROOT_SIGNATURE \"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | \
             CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED | SAMPLER_HEAP_DIRECTLY_INDEXED), \
             RootConstants(num32BitConstants=7, b0, space=0, visibility=SHADER_VISIBILITY_VERTEX), \
             RootConstants(num32BitConstants=7, b0, space=1, visibility=SHADER_VISIBILITY_PIXEL), \
             RootConstants(num32BitConstants=7, b0, space=2, visibility=SHADER_VISIBILITY_ALL)\"\n\n",
        );
        match self.options.stage {
            MetalShaderStage::Vertex => {
                source.push_str("struct RemuVsOut {\n    float4 position : SV_Position;\n");
                if self.options.render_target_array_length > 1 {
                    source.push_str(
                        "    uint render_target_array_index : SV_RenderTargetArrayIndex;\n",
                    );
                }
                for location in &self.vertex_outputs {
                    let _ = writeln!(source, "    float4 param{location} : TEXCOORD{location};");
                }
                source.push_str("};\n\n");
            }
            MetalShaderStage::Pixel => {
                source.push_str("struct RemuPsIn {\n    float4 frag_coord : SV_Position;\n");
                if self.options.render_target_array_length > 1 {
                    source.push_str(
                        "    uint render_target_array_index : SV_RenderTargetArrayIndex;\n",
                    );
                }
                for attribute in &self.pixel_attributes {
                    let input = self.options.pixel_input_controls[*attribute as usize];
                    if input.uses_default() {
                        continue;
                    }
                    let qualifier = self
                        .pixel_interpolations
                        .get(attribute)
                        .copied()
                        .unwrap_or(PixelInterpolation::CenterPerspective)
                        .hlsl_qualifier();
                    let _ = writeln!(
                        source,
                        "    {qualifier}float4 attr{attribute} : TEXCOORD{};",
                        input.input_offset
                    );
                }
                source.push_str("};\n\n");
                if !self.options.pixel_outputs.is_empty() || self.options.pixel_depth_export {
                    source.push_str("struct RemuPsOut {\n");
                    for output in &self.options.pixel_outputs {
                        let ty = match output.kind {
                            PixelOutputKind::Float => "float4",
                            PixelOutputKind::Uint => "uint4",
                            PixelOutputKind::Sint => "int4",
                        };
                        let _ = writeln!(
                            source,
                            "    {ty} mrt{} : SV_Target{};",
                            output.guest_slot, output.host_location
                        );
                    }
                    if self.options.pixel_depth_export {
                        source.push_str("    float depth : SV_Depth;\n");
                    }
                    source.push_str("};\n\n");
                }
            }
            MetalShaderStage::Compute => {}
            MetalShaderStage::Mesh => unreachable!(),
        }
    }

    fn emit_hlsl_graphics_lane_storage(&self, source: &mut String) {
        let lane_file_width = if self.options.wave_lane_count == 64 {
            64
        } else {
            32
        };
        for vgpr in self.graphics_lane_file_vgprs() {
            let _ = writeln!(source, "    uint remu_v_{vgpr}_lanes[{lane_file_width}];");
            let _ = writeln!(
                source,
                "    [unroll] for (uint remu_v_{vgpr}_init = 0u; remu_v_{vgpr}_init < {lane_file_width}u; ++remu_v_{vgpr}_init) remu_v_{vgpr}_lanes[remu_v_{vgpr}_init] = 0u;"
            );
        }
        let slots = self.graphics_addtid_write_slot_count();
        if slots != 0 {
            let _ = writeln!(source, "    uint remu_addtid_address[{slots}];");
            let _ = writeln!(source, "    uint remu_addtid_value[{slots}];");
            let _ = writeln!(source, "    uint remu_addtid_sequence[{slots}];");
            let _ = writeln!(source, "    bool remu_addtid_valid[{slots}];");
            let _ = writeln!(
                source,
                "    [unroll] for (uint remu_addtid_init = 0u; remu_addtid_init < {slots}u; ++remu_addtid_init) {{ remu_addtid_address[remu_addtid_init] = 0u; remu_addtid_value[remu_addtid_init] = 0u; remu_addtid_sequence[remu_addtid_init] = 0u; remu_addtid_valid[remu_addtid_init] = false; }}"
            );
            source.push_str("    uint remu_addtid_clock = 0u;\n");
        }
    }
}

impl PixelInterpolation {
    const fn hlsl_qualifier(self) -> &'static str {
        match self {
            Self::CenterPerspective => "",
            Self::CentroidPerspective => "centroid ",
            Self::SamplePerspective => "sample ",
            Self::CenterNoPerspective => "noperspective ",
            Self::CentroidNoPerspective => "noperspective centroid ",
            Self::SampleNoPerspective => "noperspective sample ",
            Self::Flat => "nointerpolation ",
        }
    }
}

fn validate_hlsl_interface(
    compiler: &Compiler<'_>,
    threads_per_group: [u32; 3],
) -> Result<(), MetalCompileError> {
    if compiler.options.stage == MetalShaderStage::Mesh {
        return Err(MetalCompileError::InvalidInterface(
            "NGG mesh lowering requires the D3D12 compute/classic-graphics fallback".into(),
        ));
    }
    if compiler.options.stage == MetalShaderStage::Compute {
        let total = threads_per_group
            .into_iter()
            .try_fold(1_u32, u32::checked_mul);
        if total.is_none_or(|total| total == 0 || total > 1_024)
            || threads_per_group
                .into_iter()
                .any(|dimension| dimension == 0)
        {
            return Err(MetalCompileError::InvalidInterface(format!(
                "compute threadgroup {:?} is outside the D3D12 1..=1024-thread limit",
                threads_per_group
            )));
        }
    }
    Ok(())
}

fn hlsl_texture_type(kind: ImageKind) -> String {
    let dimension = match metal_image_dimension(kind) {
        2 => "Texture3D",
        3 => "Texture2DArray",
        _ => "Texture2D",
    };
    let prefix = if kind.writable { "RW" } else { "" };
    let component = if kind.comparison {
        "float"
    } else {
        kind.component
    };
    let element = if kind.comparison {
        component.to_owned()
    } else {
        format!("{component}4")
    };
    format!("{prefix}{dimension}<{element}>")
}

fn translate_common_source(mut source: String) -> String {
    for (from, to) in [
        ("as_type<uint>(", "asuint("),
        ("as_type<int>(", "asint("),
        ("as_type<float>(", "asfloat("),
        ("as_type<half2>(", "remu_as_half2("),
        ("as_type<ushort2>(", "remu_as_ushort2("),
        ("as_type<short2>(", "remu_as_short2("),
        ("as_type<half>(", "remu_as_half("),
        ("as_type<ushort>(", "remu_as_ushort("),
        ("vec<float, 4>", "float4"),
        ("vec<uint, 4>", "uint4"),
        ("vec<int, 4>", "int4"),
        ("metal::precise::fma(", "fma("),
        ("precise::fma(", "fma("),
        ("reverse_bits(", "reversebits("),
        ("popcount(", "countbits("),
        ("clz(", "remu_clz("),
        ("ctz(", "remu_ctz("),
        ("fract(", "frac("),
        ("rint(", "round("),
        ("fmax(", "max("),
        ("fmin(", "min("),
        ("fabs(", "abs("),
        (
            "threadgroup_barrier(mem_flags::mem_threadgroup)",
            "GroupMemoryBarrierWithGroupSync()",
        ),
        ("simd_prefix_exclusive_sum(", "WavePrefixSum("),
        ("simd_shuffle(", "WaveReadLaneAt("),
        ("static constant ", "static const "),
        ("thread uint& ", "out uint "),
        ("ulong", "uint64_t"),
        ("(long)", "(int64_t)"),
        ("(ushort)", "(uint)"),
        ("#pragma clang fp reassociate(off) contract(off)\n", ""),
    ] {
        source = source.replace(from, to);
    }
    source
}

fn translate_hlsl_body(source: String) -> String {
    let mut source = translate_common_source(source);
    for (from, to) in [
        ("atomic_load_explicit(remu_lds + ", "remu_lds_load("),
        ("atomic_store_explicit(remu_lds + ", "remu_lds_store("),
        ("atomic_fetch_add_explicit(remu_lds + ", "remu_lds_add("),
        ("atomic_fetch_sub_explicit(remu_lds + ", "remu_lds_sub("),
        ("atomic_fetch_min_explicit(remu_lds + ", "remu_lds_min("),
        ("atomic_fetch_max_explicit(remu_lds + ", "remu_lds_max("),
        ("atomic_fetch_and_explicit(remu_lds + ", "remu_lds_and("),
        ("atomic_fetch_or_explicit(remu_lds + ", "remu_lds_or("),
        ("atomic_fetch_xor_explicit(remu_lds + ", "remu_lds_xor("),
        ("atomic_exchange_explicit(remu_lds + ", "remu_lds_swap("),
        ("remu_ds_inc(remu_lds + ", "remu_ds_inc("),
        ("remu_ds_dec(remu_lds + ", "remu_ds_dec("),
        ("remu_ds_smin(remu_lds + ", "remu_ds_smin("),
        ("remu_ds_smax(remu_lds + ", "remu_ds_smax("),
        ("remu_ds_cmpst(remu_lds + ", "remu_ds_cmpst("),
    ] {
        source = source.replace(from, to);
    }
    source.replace(", memory_order_relaxed)", ")")
}

fn translate_format_prelude(mut source: String) -> String {
    source = source.replace("const device uchar* data", "uint data");
    source = source.replace("device uchar* data", "uint data");
    source = source.replace(
        "value |= (((uint)data[source_byte] >> (source_bit & 7u)) & 1u) << bit;",
        "value |= ((remu_load_u8(data, source_byte, 0xffffffffu) >> (source_bit & 7u)) & 1u) << bit;",
    );
    source = source.replace(
        "uint byte_value = (uint)data[destination_byte];",
        "uint byte_value = remu_load_u8(data, destination_byte, 0xffffffffu);",
    );
    source = source.replace(
        "data[destination_byte] = (uchar)byte_value;",
        "remu_store_u8(data, destination_byte, byte_value);",
    );
    source = source.replace(
        "remu_srgb_encode(float linear)",
        "remu_srgb_encode(float remu_linear)",
    );
    source = source.replace(
        "    linear = clamp(linear, 0.0f, 1.0f);",
        "    remu_linear = clamp(remu_linear, 0.0f, 1.0f);",
    );
    source = source.replace("return linear <= 0.0031308f ? linear * 12.92f : 1.055f * pow(linear, 1.0f / 2.4f) - 0.055f;", "return remu_linear <= 0.0031308f ? remu_linear * 12.92f : 1.055f * pow(remu_linear, 1.0f / 2.4f) - 0.055f;");
    translate_common_source(source)
}

fn emit_hlsl_prelude(source: &mut String, uses_lds: bool) {
    source.push_str(
        r"uint remu_clz(uint value) {
    return value == 0u ? 32u : 31u - (uint)firstbithigh(value);
}

uint remu_ctz(uint value) {
    return value == 0u ? 32u : (uint)firstbitlow(value);
}

uint mulhi(uint left, uint right) {
    return (uint)(((uint64_t)left * (uint64_t)right) >> 32u);
}

float fma(float left, float right, float addend) {
    return mad(left, right, addend);
}

uint64_t simd_ballot(bool condition) {
    uint4 ballot = WaveActiveBallot(condition);
    return (uint64_t)ballot.x | ((uint64_t)ballot.y << 32u);
}

float remu_as_half(uint value) { return f16tof32(value & 0xffffu); }
float2 remu_as_half2(uint value) {
    return float2(f16tof32(value & 0xffffu), f16tof32(value >> 16u));
}
uint remu_as_ushort(uint value) { return value & 0xffffu; }
uint remu_as_ushort(float value) { return f32tof16(value); }
uint2 remu_as_ushort2(uint value) { return uint2(value & 0xffffu, value >> 16u); }
int2 remu_as_short2(uint value) {
    return int2((int)(value << 16u) >> 16, (int)value >> 16);
}

uint extract_bits(uint value, uint offset, uint count) {
    if (count == 0u || offset >= 32u) return 0u;
    uint width = min(count, 32u - offset);
    uint mask = width == 32u ? 0xffffffffu : ((1u << width) - 1u);
    return (value >> offset) & mask;
}

int extract_bits(int value, uint offset, uint count) {
    if (count == 0u || offset >= 32u) return 0;
    uint width = min(count, 32u - offset);
    uint shifted = asuint(value) >> offset;
    if (width == 32u) return asint(shifted);
    uint mask = (1u << width) - 1u;
    uint sign = 1u << (width - 1u);
    uint bits = shifted & mask;
    return asint((bits ^ sign) - sign);
}

bool remu_range_valid(uint offset, uint size, uint length) {
    return offset <= length && size <= length - offset;
}

uint remu_load_u32(uint descriptor, uint offset, uint length) {
    if (!remu_range_valid(offset, 4u, length)) return 0u;
    RWByteAddressBuffer data = ResourceDescriptorHeap[descriptor];
    if ((offset & 3u) == 0u) return data.Load(offset);
    uint first = data.Load(offset & ~3u);
    uint second = data.Load((offset & ~3u) + 4u);
    uint shift = (offset & 3u) * 8u;
    return (first >> shift) | (second << (32u - shift));
}

uint remu_load_u8(uint descriptor, uint offset, uint length) {
    if (!remu_range_valid(offset, 1u, length)) return 0u;
    RWByteAddressBuffer data = ResourceDescriptorHeap[descriptor];
    return (data.Load(offset & ~3u) >> ((offset & 3u) * 8u)) & 0xffu;
}

uint remu_load_u16(uint descriptor, uint offset, uint length) {
    if (!remu_range_valid(offset, 2u, length)) return 0u;
    return remu_load_u8(descriptor, offset, length)
        | (remu_load_u8(descriptor, offset + 1u, length) << 8u);
}

uint remu_load_u32_ro(uint descriptor, uint offset, uint length) {
    if (!remu_range_valid(offset, 4u, length)) return 0u;
    ByteAddressBuffer data = ResourceDescriptorHeap[descriptor];
    if ((offset & 3u) == 0u) return data.Load(offset);
    uint first = data.Load(offset & ~3u);
    uint second = data.Load((offset & ~3u) + 4u);
    uint shift = (offset & 3u) * 8u;
    return (first >> shift) | (second << (32u - shift));
}

uint remu_load_u8_ro(uint descriptor, uint offset, uint length) {
    if (!remu_range_valid(offset, 1u, length)) return 0u;
    ByteAddressBuffer data = ResourceDescriptorHeap[descriptor];
    return (data.Load(offset & ~3u) >> ((offset & 3u) * 8u)) & 0xffu;
}

uint remu_load_u16_ro(uint descriptor, uint offset, uint length) {
    if (!remu_range_valid(offset, 2u, length)) return 0u;
    return remu_load_u8_ro(descriptor, offset, length)
        | (remu_load_u8_ro(descriptor, offset + 1u, length) << 8u);
}

uint remu_load_format_bits_ro(uint data, uint address, uint byte_off, uint bit_off, uint bits) {
    uint value = 0u;
    for (uint bit = 0u; bit < bits; ++bit) {
        uint source_bit = bit_off + bit;
        uint source_byte = address + byte_off + source_bit / 8u;
        value |= ((remu_load_u8_ro(data, source_byte, 0xffffffffu) >> (source_bit & 7u)) & 1u) << bit;
    }
    return value;
}

void remu_store_masked(uint descriptor, uint offset, uint value, uint byte_count) {
    RWByteAddressBuffer data = ResourceDescriptorHeap[descriptor];
    for (uint byte_index = 0u; byte_index < byte_count; ++byte_index) {
        uint address = offset + byte_index;
        uint aligned = address & ~3u;
        uint shift = (address & 3u) * 8u;
        uint mask = 0xffu << shift;
        uint observed = data.Load(aligned);
        for (;;) {
            uint desired = (observed & ~mask) | (((value >> (byte_index * 8u)) & 0xffu) << shift);
            uint actual;
            data.InterlockedCompareExchange(aligned, observed, desired, actual);
            if (actual == observed) break;
            observed = actual;
        }
    }
}

void remu_store_u8(uint descriptor, uint offset, uint value) {
    remu_store_masked(descriptor, offset, value, 1u);
}
void remu_store_u16(uint descriptor, uint offset, uint value) {
    remu_store_masked(descriptor, offset, value, 2u);
}
void remu_store_u32(uint descriptor, uint offset, uint value) {
    RWByteAddressBuffer data = ResourceDescriptorHeap[descriptor];
    if ((offset & 3u) == 0u) data.Store(offset, value);
    else remu_store_masked(descriptor, offset, value, 4u);
}

uint remu_atomic_swap(uint descriptor, uint offset, uint value) {
    RWByteAddressBuffer data = ResourceDescriptorHeap[descriptor];
    uint original;
    data.InterlockedExchange(offset, value, original);
    return original;
}
uint remu_atomic_add(uint descriptor, uint offset, uint value) {
    RWByteAddressBuffer data = ResourceDescriptorHeap[descriptor];
    uint original;
    data.InterlockedAdd(offset, value, original);
    return original;
}
uint remu_atomic_sub(uint descriptor, uint offset, uint value) {
    return remu_atomic_add(descriptor, offset, 0u - value);
}
uint remu_atomic_and(uint descriptor, uint offset, uint value) {
    RWByteAddressBuffer data = ResourceDescriptorHeap[descriptor];
    uint original;
    data.InterlockedAnd(offset, value, original);
    return original;
}
uint remu_atomic_or(uint descriptor, uint offset, uint value) {
    RWByteAddressBuffer data = ResourceDescriptorHeap[descriptor];
    uint original;
    data.InterlockedOr(offset, value, original);
    return original;
}
uint remu_atomic_xor(uint descriptor, uint offset, uint value) {
    RWByteAddressBuffer data = ResourceDescriptorHeap[descriptor];
    uint original;
    data.InterlockedXor(offset, value, original);
    return original;
}
uint remu_atomic_cmpswap(uint descriptor, uint offset, uint replacement, uint comparator) {
    RWByteAddressBuffer data = ResourceDescriptorHeap[descriptor];
    uint original;
    data.InterlockedCompareExchange(offset, comparator, replacement, original);
    return original;
}
uint remu_atomic_umin(uint descriptor, uint offset, uint value) {
    RWByteAddressBuffer data = ResourceDescriptorHeap[descriptor];
    uint original;
    data.InterlockedMin(offset, value, original);
    return original;
}
uint remu_atomic_umax(uint descriptor, uint offset, uint value) {
    RWByteAddressBuffer data = ResourceDescriptorHeap[descriptor];
    uint original;
    data.InterlockedMax(offset, value, original);
    return original;
}
uint remu_atomic_smin(uint descriptor, uint offset, uint value) {
    RWByteAddressBuffer data = ResourceDescriptorHeap[descriptor];
    uint observed = data.Load(offset);
    for (;;) {
        uint desired = asint(value) < asint(observed) ? value : observed;
        uint actual;
        data.InterlockedCompareExchange(offset, observed, desired, actual);
        if (actual == observed) return observed;
        observed = actual;
    }
}
uint remu_atomic_smax(uint descriptor, uint offset, uint value) {
    RWByteAddressBuffer data = ResourceDescriptorHeap[descriptor];
    uint observed = data.Load(offset);
    for (;;) {
        uint desired = asint(value) > asint(observed) ? value : observed;
        uint actual;
        data.InterlockedCompareExchange(offset, observed, desired, actual);
        if (actual == observed) return observed;
        observed = actual;
    }
}
uint remu_atomic_inc(uint descriptor, uint offset, uint limit) {
    RWByteAddressBuffer data = ResourceDescriptorHeap[descriptor];
    uint observed = data.Load(offset);
    for (;;) {
        uint desired = observed >= limit ? 0u : observed + 1u;
        uint actual;
        data.InterlockedCompareExchange(offset, observed, desired, actual);
        if (actual == observed) return observed;
        observed = actual;
    }
}
uint remu_atomic_dec(uint descriptor, uint offset, uint limit) {
    RWByteAddressBuffer data = ResourceDescriptorHeap[descriptor];
    uint observed = data.Load(offset);
    for (;;) {
        uint desired = (observed == 0u || observed > limit) ? limit : observed - 1u;
        uint actual;
        data.InterlockedCompareExchange(offset, observed, desired, actual);
        if (actual == observed) return observed;
        observed = actual;
    }
}

",
    );
    if uses_lds {
        source.push_str(
            r"uint remu_lds_load(uint index) {
    uint original;
    InterlockedOr(remu_lds[index], 0u, original);
    return original;
}
void remu_lds_store(uint index, uint value) {
    uint ignored;
    InterlockedExchange(remu_lds[index], value, ignored);
}
uint remu_lds_add(uint index, uint value) {
    uint original;
    InterlockedAdd(remu_lds[index], value, original);
    return original;
}
uint remu_lds_sub(uint index, uint value) { return remu_lds_add(index, 0u - value); }
uint remu_lds_min(uint index, uint value) {
    uint original;
    InterlockedMin(remu_lds[index], value, original);
    return original;
}
uint remu_lds_max(uint index, uint value) {
    uint original;
    InterlockedMax(remu_lds[index], value, original);
    return original;
}
uint remu_lds_and(uint index, uint value) {
    uint original;
    InterlockedAnd(remu_lds[index], value, original);
    return original;
}
uint remu_lds_or(uint index, uint value) {
    uint original;
    InterlockedOr(remu_lds[index], value, original);
    return original;
}
uint remu_lds_xor(uint index, uint value) {
    uint original;
    InterlockedXor(remu_lds[index], value, original);
    return original;
}
uint remu_lds_swap(uint index, uint value) {
    uint original;
    InterlockedExchange(remu_lds[index], value, original);
    return original;
}
uint remu_ds_inc(uint index, uint ceiling) {
    uint observed = remu_lds_load(index);
    for (;;) {
        uint desired = observed >= ceiling ? 0u : observed + 1u;
        uint actual;
        InterlockedCompareExchange(remu_lds[index], observed, desired, actual);
        if (actual == observed) return observed;
        observed = actual;
    }
}
uint remu_ds_dec(uint index, uint ceiling) {
    uint observed = remu_lds_load(index);
    for (;;) {
        uint desired = (observed == 0u || observed > ceiling) ? ceiling : observed - 1u;
        uint actual;
        InterlockedCompareExchange(remu_lds[index], observed, desired, actual);
        if (actual == observed) return observed;
        observed = actual;
    }
}
uint remu_ds_smin(uint index, uint value) {
    uint observed = remu_lds_load(index);
    for (;;) {
        uint desired = asint(value) < asint(observed) ? value : observed;
        uint actual;
        InterlockedCompareExchange(remu_lds[index], observed, desired, actual);
        if (actual == observed) return observed;
        observed = actual;
    }
}
uint remu_ds_smax(uint index, uint value) {
    uint observed = remu_lds_load(index);
    for (;;) {
        uint desired = asint(value) > asint(observed) ? value : observed;
        uint actual;
        InterlockedCompareExchange(remu_lds[index], observed, desired, actual);
        if (actual == observed) return observed;
        observed = actual;
    }
}
uint remu_ds_cmpst(uint index, uint comparator, uint replacement) {
    uint observed = remu_lds_load(index);
    while (observed == comparator) {
        uint actual;
        InterlockedCompareExchange(remu_lds[index], observed, replacement, actual);
        if (actual == observed) return observed;
        observed = actual;
    }
    return observed;
}

",
        );
    }
}
