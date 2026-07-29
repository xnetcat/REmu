// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Export lowering: EXP targets for the vertex, mesh and pixel stages, and the
//! parameter interpolation instructions the pixel stage reads them back with.

use remu_shader::{ExportControl, Instruction, InstructionControl, PixelOutputKind};

use crate::compiler::interface::vertex_clip_distance_mask;
use crate::compiler::vector::{destination_vector, vector_operand};
use crate::{Compiler, MetalShaderStage};

impl Compiler<'_> {
    pub(crate) fn emit_export(&mut self, instruction: &Instruction) -> Result<(), String> {
        let Some(InstructionControl::Export(export)) = instruction.control.as_ref() else {
            return Err("export instruction lacks export control".to_owned());
        };
        if instruction.sources.len() < 4 {
            return Err("export requires four source operands".to_owned());
        }
        match self.options.stage {
            MetalShaderStage::Vertex => self.emit_vertex_export(instruction, export),
            MetalShaderStage::Pixel => self.emit_pixel_export(instruction, export),
            MetalShaderStage::Compute => Ok(()),
            MetalShaderStage::Mesh => self.emit_mesh_export(instruction, export),
        }
    }

    fn emit_mesh_export(
        &mut self,
        instruction: &Instruction,
        export: &ExportControl,
    ) -> Result<(), String> {
        if export.target == 20 {
            let primitive = self.temp("uint", &self.raw_source(instruction, 0)?);
            let lane = self.architectural_lane();
            self.line(
                3,
                &format!("if (exec && ({primitive} & 0x80000000u) == 0u) {{"),
            );
            self.line(
                4,
                &format!("remu_mesh.set_index({lane} * 3u + 0u, ({primitive} >> 0u) & 0x1ffu);"),
            );
            self.line(
                4,
                &format!("remu_mesh.set_index({lane} * 3u + 1u, ({primitive} >> 10u) & 0x1ffu);"),
            );
            self.line(
                4,
                &format!("remu_mesh.set_index({lane} * 3u + 2u, ({primitive} >> 20u) & 0x1ffu);"),
            );
            self.line(3, "}");
            return Ok(());
        }
        self.emit_vertex_export(instruction, export)
    }

    fn emit_vertex_export(
        &mut self,
        instruction: &Instruction,
        export: &ExportControl,
    ) -> Result<(), String> {
        if export.target == 13 && self.options.vertex_output_control & (1 << 21) != 0 {
            return self.emit_vertex_misc_export(instruction, export);
        }
        if (13..=15).contains(&export.target) {
            return self.emit_vertex_distance_export(instruction, export);
        }
        let field = if export.target == 12 {
            "position".to_owned()
        } else if (32..64).contains(&export.target)
            && self.vertex_outputs.contains(&(export.target - 32))
        {
            format!("param{}", export.target - 32)
        } else {
            return Ok(());
        };
        let values = self.export_float_values(instruction, export)?;
        self.line(
            3,
            &format!(
                "if (exec) remu_out.{field} = float4({}, {}, {}, {});",
                values[0], values[1], values[2], values[3]
            ),
        );
        Ok(())
    }

    fn emit_vertex_misc_export(
        &mut self,
        instruction: &Instruction,
        export: &ExportControl,
    ) -> Result<(), String> {
        // PA_CL_VS_OUT_CNTL maps POS1.z to the render-target index when the misc vector and
        // USE_VTX_RENDER_TARGET_INDX are enabled. USE_VTX_KILL_FLAG takes precedence over the
        // same component. The register payload is a raw integer carried through an EXP float
        // operand, so preserve its bits rather than numerically converting the float.
        let control = self.options.vertex_output_control;
        let exports_render_target_index = control & (1 << 18) != 0 && control & (1 << 20) == 0;
        if !exports_render_target_index
            || export.enable_mask & (1 << 2) == 0
            || self.options.render_target_array_length <= 1
        {
            return Ok(());
        }
        let values = self.export_float_values(instruction, export)?;
        let maximum_layer = self.options.render_target_array_length - 1;
        let destination = match self.options.stage {
            MetalShaderStage::Vertex => "remu_out.render_target_array_index",
            MetalShaderStage::Mesh => "remu_mesh_render_target_indices[remu_thread_index]",
            _ => return Ok(()),
        };
        self.line(
            3,
            &format!(
                "if (exec) {destination} = min(as_type<uint>({}), {maximum_layer}u);",
                values[2]
            ),
        );
        Ok(())
    }

    fn emit_vertex_distance_export(
        &mut self,
        instruction: &Instruction,
        export: &ExportControl,
    ) -> Result<(), String> {
        let control = self.options.vertex_output_control;
        let mut packed_target = 13_u32;
        // POS1..3 contain only enabled vectors. A miscellaneous output therefore shifts both
        // clip/cull vectors by one even when REmu does not consume the misc components itself.
        if control & (1 << 21) != 0 {
            if export.target == packed_target {
                return Ok(());
            }
            packed_target += 1;
        }
        let distance_base = if control & (1 << 22) != 0 {
            if export.target == packed_target {
                Some(0_usize)
            } else {
                packed_target += 1;
                None
            }
        } else {
            None
        }
        .or_else(|| {
            (control & (1 << 23) != 0 && export.target == packed_target).then_some(4_usize)
        });
        let Some(distance_base) = distance_base else {
            return Ok(());
        };
        let distance_mask = vertex_clip_distance_mask(control);
        let values = self.export_float_values(instruction, export)?;
        for (component, value) in values.iter().enumerate() {
            let distance = distance_base + component;
            if export.enable_mask & (1 << component) != 0 && distance_mask & (1 << distance) != 0 {
                self.line(
                    3,
                    &format!("if (exec) remu_out.clip_distances[{distance}] = {value};"),
                );
            }
        }
        Ok(())
    }

    fn emit_pixel_export(
        &mut self,
        instruction: &Instruction,
        export: &ExportControl,
    ) -> Result<(), String> {
        let Some(output) = self
            .options
            .pixel_outputs
            .iter()
            .find(|output| output.guest_slot == export.target)
            .copied()
        else {
            return Ok(());
        };
        let field = format!("remu_out.mrt{}", output.guest_slot);
        let mut values = Vec::with_capacity(4);
        for component in 0..4 {
            if export.enable_mask & (1 << component) == 0 {
                values.push(format!("{field}[{component}]"));
                continue;
            }
            let raw = if export.compressed {
                let register = vector_operand(instruction.sources[component >> 1])?;
                format!("(float)as_type<half2>(v[{register}])[{}]", component & 1)
            } else {
                let register = vector_operand(instruction.sources[component])?;
                format!("as_type<float>(v[{register}])")
            };
            values.push(match output.kind {
                PixelOutputKind::Float => raw,
                PixelOutputKind::Uint => format!("(uint)({raw})"),
                PixelOutputKind::Sint => format!("(int)({raw})"),
            });
        }
        let ty = match output.kind {
            PixelOutputKind::Float => "float4",
            PixelOutputKind::Uint => "uint4",
            PixelOutputKind::Sint => "int4",
        };
        self.line(
            3,
            &format!(
                "if (exec) {field} = {ty}({}, {}, {}, {});",
                values[0], values[1], values[2], values[3]
            ),
        );
        Ok(())
    }

    #[allow(clippy::unused_self)]
    fn export_float_values(
        &self,
        instruction: &Instruction,
        export: &ExportControl,
    ) -> Result<[String; 4], String> {
        let mut values = [String::new(), String::new(), String::new(), String::new()];
        for (component, value) in values.iter_mut().enumerate() {
            if export.enable_mask & (1 << component) == 0 {
                *value = String::from(if component == 3 { "1.0f" } else { "0.0f" });
            } else if export.compressed {
                let register = vector_operand(instruction.sources[component >> 1])?;
                *value = format!("(float)as_type<half2>(v[{register}])[{}]", component & 1);
            } else {
                let register = vector_operand(instruction.sources[component])?;
                *value = format!("as_type<float>(v[{register}])");
            }
        }
        Ok(values)
    }

    pub(crate) fn emit_interpolation(&mut self, instruction: &Instruction) -> Result<(), String> {
        let Some(InstructionControl::Interpolation(interpolation)) = instruction.control.as_ref()
        else {
            return Err("missing interpolation control".to_owned());
        };
        if self.options.stage != MetalShaderStage::Pixel
            || !self.pixel_attributes.contains(&interpolation.attribute)
        {
            return Err("interpolation is valid only for a declared pixel input".to_owned());
        }
        // Host rasterization performs the complete interpolation. P1 only forms an intermediate
        // hardware partial product; the translated destination becomes architecturally visible
        // when P2 consumes J, matching shadPS4's non-manual interpolation path.
        if instruction.opcode.starts_with("VInterpP1") {
            return Ok(());
        }
        let destination = destination_vector(instruction)?;
        let input = self.options.pixel_input_controls[interpolation.attribute as usize];
        let value = if input.uses_default() {
            let one = match input.default_value {
                0 => false,
                1 => interpolation.channel == 3,
                2 => interpolation.channel != 3,
                3 => true,
                _ => unreachable!("pixel input controls are validated during compiler creation"),
            };
            if one { "0x3F800000u" } else { "0u" }.to_owned()
        } else {
            format!(
                "as_type<uint>(remu_in.attr{}[{}])",
                interpolation.attribute, interpolation.channel
            )
        };
        self.store_vector(destination, &value);
        Ok(())
    }
}
