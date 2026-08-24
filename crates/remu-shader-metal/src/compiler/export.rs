// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Export lowering: EXP targets for the vertex, mesh and pixel stages, and the
//! parameter interpolation instructions the pixel stage reads them back with.

use remu_shader::{
    ExportControl, Instruction, InstructionControl, OperandKind, PixelExportFormat,
    PixelOutputKind, Program,
};

use crate::compiler::interface::vertex_clip_distance_mask;
use crate::compiler::vector::{destination_vector, vector_operand};
use crate::{Compiler, MetalShaderStage};

/// Locate the SGPR that holds packed NGG export counts `(verts << 16) | prims` immediately before
/// the GFX10 empty-workaround / `GS_ALLOC_REQ` sequence (`S_CMPK_LE_U32 sN, 0xFFFF` then later
/// `S_SENDMSG` message 9).
pub(crate) fn mesh_ngg_packed_export_sgpr(program: &Program) -> Option<u32> {
    let has_gs_alloc = program.instructions.iter().any(|instruction| {
        instruction.opcode.is("SSendmsg")
            && instruction.sources.first().is_some_and(|source| {
                source.kind == OperandKind::EncodedConstant && source.value == 9
            })
    });
    if !has_gs_alloc {
        return None;
    }
    program.instructions.iter().find_map(|instruction| {
        if !instruction.opcode.is("SCmpkLeU32") {
            return None;
        }
        let imm = instruction.words.first().copied().unwrap_or(0) & 0xffff;
        if imm != 0xffff {
            return None;
        }
        instruction
            .destinations
            .first()
            .filter(|operand| operand.kind == OperandKind::ScalarRegister)
            .map(|operand| operand.value)
    })
}

/// Locate the VGPR carrying the compacted NGG vertex id used to gate POS export (typically
/// `V_CMPX_* sN, vM` immediately before the first `EXP POS`). PRIM indices address this same
/// compacted domain; the Metal vertex records must therefore use it as well.
pub(crate) fn mesh_export_vertex_vgpr(program: &Program) -> Option<u32> {
    let exp_pos = program.instructions.iter().position(|instruction| {
        matches!(
            instruction.control.as_ref(),
            Some(InstructionControl::Export(export)) if export.target == 12
        )
    })?;
    for instruction in program.instructions[..exp_pos].iter().rev().take(32) {
        if !instruction.opcode.starts_with("VCmpx") {
            continue;
        }
        let mut vgpr = None;
        let mut has_sgpr = false;
        for source in &instruction.sources {
            match source.kind {
                OperandKind::VectorRegister => vgpr = Some(source.value),
                OperandKind::ScalarRegister => has_sgpr = true,
                _ => {}
            }
        }
        if has_sgpr {
            return vgpr;
        }
    }
    None
}

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
            if self.options.ngg_fixed_allocation_fallback {
                let lane = self.architectural_lane();
                self.line(
                    3,
                    &format!("if (exec && ({primitive} & 0x80000000u) == 0u) {{"),
                );
                self.line(
                    4,
                    &format!(
                        "remu_mesh.set_index({lane} * 3u + 0u, ({primitive} >> 0u) & 0x1ffu);"
                    ),
                );
                self.line(
                    4,
                    &format!(
                        "remu_mesh.set_index({lane} * 3u + 1u, ({primitive} >> 10u) & 0x1ffu);"
                    ),
                );
                self.line(
                    4,
                    &format!(
                        "remu_mesh.set_index({lane} * 3u + 2u, ({primitive} >> 20u) & 0x1ffu);"
                    ),
                );
                self.line(3, "}");
                return Ok(());
            }
            self.line(
                3,
                &format!(
                    "if (exec && remu_thread_index < {}u && ({primitive} & 0x80000000u) == 0u) {{",
                    self.options.mesh_max_primitive_count
                ),
            );
            // A GFX10 primitive word carries 9-bit vertex fields (0..511) while the Metal mesh
            // declares only mesh_max_vertex_count vertices. An index beyond that range could
            // never have been exported, and passing it to set_index is undefined behavior in
            // Metal; keep the degenerate prefill for such primitives instead.
            self.line(
                4,
                &format!(
                    "uint remu_prim_i0 = ({primitive} >> 0u) & 0x1ffu;\n\
                     \x20           uint remu_prim_i1 = ({primitive} >> 10u) & 0x1ffu;\n\
                     \x20           uint remu_prim_i2 = ({primitive} >> 20u) & 0x1ffu;\n\
                     \x20           if (max(max(remu_prim_i0, remu_prim_i1), remu_prim_i2) < {}u) {{",
                    self.options.mesh_max_vertex_count
                ),
            );
            self.line(
                5,
                "remu_mesh.set_index(remu_thread_index * 3u + 0u, remu_prim_i0);",
            );
            self.line(
                5,
                "remu_mesh.set_index(remu_thread_index * 3u + 1u, remu_prim_i1);",
            );
            self.line(
                5,
                "remu_mesh.set_index(remu_thread_index * 3u + 2u, remu_prim_i2);",
            );
            if self.options.render_target_array_length > 1 {
                self.line(
                    5,
                    "remu_mesh_primitive_source_vertices[remu_thread_index] = remu_prim_i0;",
                );
            }
            self.line(4, "}");
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
        if field == "position" {
            self.convert_compacted_mesh_position();
        }
        self.publish_compacted_mesh_vertex();
        Ok(())
    }

    /// Convert a compacted mesh position at its POS export. Later PARAM exports publish the same
    /// already-converted position, while the next POS overwrites it before converting again.
    fn convert_compacted_mesh_position(&mut self) {
        if self.options.stage != MetalShaderStage::Mesh || self.mesh_export_vertex_vgpr.is_none() {
            return;
        }
        const VTX_XY_FMT: u32 = 1 << 8;
        const VTX_Z_FMT: u32 = 1 << 9;
        const VTX_W0_FMT: u32 = 1 << 10;
        let control = self.options.viewport_transform_control;
        let xy_predivided = control & VTX_XY_FMT != 0;
        let z_predivided = control & VTX_Z_FMT != 0;
        let w_is_w0 = control & VTX_W0_FMT != 0;
        if !xy_predivided && !z_predivided && w_is_w0 {
            return;
        }
        let w0 = if w_is_w0 {
            "remu_out.position.w".to_owned()
        } else {
            "1.0f / remu_out.position.w".to_owned()
        };
        self.line(3, &format!("if (exec) {{ float remu_position_w0 = {w0};"));
        if xy_predivided {
            self.line(4, "remu_out.position.xy *= remu_position_w0;");
        }
        if z_predivided {
            self.line(4, "remu_out.position.z *= remu_position_w0;");
        }
        self.line(4, "remu_out.position.w = remu_position_w0;");
        self.line(3, "}");
    }

    /// Publish the current output under the compacted vertex id used by the guest PRIM records.
    /// Direct mesh-record publication avoids duplicating a potentially full 32-KiB guest LDS.
    fn publish_compacted_mesh_vertex(&mut self) {
        let Some(vgpr) = self.mesh_export_vertex_vgpr else {
            return;
        };
        if self.options.stage != MetalShaderStage::Mesh {
            return;
        }
        self.line(
            3,
            &format!(
                "if (exec && v[{vgpr}] < {}u) remu_mesh.set_vertex(v[{vgpr}], remu_out);",
                self.options.mesh_max_vertex_count
            ),
        );
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
        let (destination, lane_guard) = match self.options.stage {
            MetalShaderStage::Vertex => (
                "remu_out.render_target_array_index".to_owned(),
                String::new(),
            ),
            MetalShaderStage::Mesh => {
                let max = self.options.mesh_max_vertex_count;
                match self.mesh_export_vertex_vgpr {
                    Some(vgpr) => (
                        format!("remu_mesh_render_target_indices[v[{vgpr}]]"),
                        format!(" && v[{vgpr}] < {max}u"),
                    ),
                    None => (
                        "remu_mesh_render_target_indices[remu_thread_index]".to_owned(),
                        format!(" && remu_thread_index < {max}u"),
                    ),
                }
            }
            _ => return Ok(()),
        };
        self.line(
            3,
            &format!(
                "if (exec{lane_guard}) {destination} = min(as_type<uint>({}), {maximum_layer}u);",
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
        self.publish_compacted_mesh_vertex();
        Ok(())
    }

    fn emit_pixel_export(
        &mut self,
        instruction: &Instruction,
        export: &ExportControl,
    ) -> Result<(), String> {
        // RDNA2 EXP target 8 is MRTZ. Component zero is fragment depth when
        // DB_SHADER_CONTROL.Z_EXPORT_ENABLE is set. Silent Hill's captured gameplay shaders use
        // the canonical uncompressed EN=1 form; dropping it made alpha-tested/depth-replacing
        // materials populate color while retaining unrelated rasterizer depth.
        if export.target == 8 {
            if !self.options.pixel_depth_export || export.enable_mask & 1 == 0 {
                return Ok(());
            }
            if export.compressed {
                return Err(
                    "compressed MRTZ depth export requires SPI_SHADER_Z_FORMAT lowering".to_owned(),
                );
            }
            let register = vector_operand(instruction.sources[0])?;
            self.line(
                3,
                &format!("if (exec) remu_out.depth = as_type<float>(v[{register}]);"),
            );
            return Ok(());
        }
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
            // AMD compressed color exports use EN[0] for the packed R/G pair and EN[2] for the
            // packed B/A pair. EN[1] and EN[3] are not independent component enables. Treating
            // the mask like an uncompressed export silently dropped G and A from the canonical
            // 0b0101 compressed mask.
            let enable_component = if export.compressed {
                export.enable_mask & (1 << (component & !1)) != 0
            } else {
                export.enable_mask & (1 << component) != 0
            };
            if !enable_component {
                values.push(format!("{field}[{component}]"));
                continue;
            }
            let register = vector_operand(
                instruction.sources[if export.compressed {
                    component >> 1
                } else {
                    component
                }],
            )?;
            values.push(match (output.kind, export.compressed, output.export_format) {
                (PixelOutputKind::Float, false, _) => {
                    format!("as_type<float>(v[{register}])")
                }
                (
                    PixelOutputKind::Float,
                    true,
                    PixelExportFormat::AbgrFp16 | PixelExportFormat::Zero,
                ) => {
                    // AGC seeds SPI_SHADER_COL_FORMAT to Zero. Cover/warmup draws still emit a
                    // compressed float EXP; AbgrFp16 matches that payload. Write-masking stays
                    // with the CB (Zero keeps mask 0) so this does not invent a color write.
                    format!("(float)as_type<half2>(v[{register}])[{}]", component & 1)
                }
                (PixelOutputKind::Float, true, PixelExportFormat::AbgrUnorm16) => format!(
                    "(float)as_type<ushort2>(v[{register}])[{}] * (1.0f / 65535.0f)",
                    component & 1
                ),
                (PixelOutputKind::Float, true, PixelExportFormat::AbgrSnorm16) => format!(
                    "fmax(-1.0f, (float)as_type<short2>(v[{register}])[{}] * (1.0f / 32767.0f))",
                    component & 1
                ),
                // Integer color exports carry integer component bits. Reinterpreting those bits
                // as float first and then value-converting made every small ID/culling value a
                // subnormal float which converted to zero. Unreal's GPU-driven passes rely on
                // these integer render targets before the visible scene can be composed.
                (PixelOutputKind::Uint, false, _) => format!("v[{register}]"),
                (PixelOutputKind::Sint, false, _) => format!("as_type<int>(v[{register}])"),
                (PixelOutputKind::Uint, true, PixelExportFormat::AbgrUint16) => {
                    format!("(uint)as_type<ushort2>(v[{register}])[{}]", component & 1)
                }
                (PixelOutputKind::Sint, true, PixelExportFormat::AbgrSint16) => {
                    format!("(int)as_type<short2>(v[{register}])[{}]", component & 1)
                }
                (kind, true, format) => {
                    return Err(format!(
                        "compressed MRT{} export format {format:?} is incompatible with {kind:?} output",
                        output.guest_slot
                    ));
                }
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
