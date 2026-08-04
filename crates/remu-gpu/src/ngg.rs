// SPDX-License-Identifier: GPL-2.0-or-later

//! Backend-neutral GFX10 next-generation geometry (NGG) classification and subgroup sizing.

use std::collections::HashMap;
use std::sync::Arc;

use remu_shader::{InstructionControl, OperandKind, Program};

use crate::GpuRegister;

pub const GE_MAX_OUTPUT_PER_SUBGROUP: u32 = 0x1ff;
pub const VGT_GS_ONCHIP_CNTL: u32 = 0x291;
pub const VGT_GS_OUT_PRIM_TYPE: u32 = 0x29b;
pub const VGT_GS_MAX_VERT_OUT: u32 = 0x2ce;
pub const GE_NGG_SUBGRP_CNTL: u32 = 0x2d3;
pub const VGT_GS_INSTANCE_CNT: u32 = 0x2e4;
pub const NGG_MESHLET_WORDS_PER_GROUP: usize = 514;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedNggMeshlets {
    pub words: Arc<[u32]>,
    pub group_count: u32,
    pub vertex_references: usize,
    pub primitive_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NggOutputPrimitiveType {
    PointList,
    LineStrip,
    TriangleStrip,
    RectangleList,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NggSubgroupLimits {
    pub es_vertices_per_subgroup: u32,
    pub gs_primitives_per_subgroup: u32,
    pub gs_instance_primitives_per_subgroup: u32,
    pub gs_instance_count: u32,
    pub gs_max_vertices_out: u32,
    pub max_output_vertices_per_subgroup: u32,
    pub primitive_amplification_factor: u32,
    pub max_output_primitives_per_subgroup: u32,
    pub output_primitive_type: Option<NggOutputPrimitiveType>,
    pub threadgroup_threads: u32,
}

impl NggSubgroupLimits {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        es_vertices_per_subgroup: u32,
        gs_primitives_per_subgroup: u32,
        gs_instance_primitives_per_subgroup: u32,
        gs_max_vertices_out: u32,
        max_output_vertices_per_subgroup: u32,
        primitive_amplification_factor: u32,
        output_primitive_type: Option<NggOutputPrimitiveType>,
        programmed_instance_count: Option<u32>,
    ) -> Result<Self, String> {
        let derived_instance_count = gs_instance_primitives_per_subgroup
            .checked_div(gs_primitives_per_subgroup)
            .filter(|&count| {
                count != 0
                    && gs_primitives_per_subgroup.checked_mul(count)
                        == Some(gs_instance_primitives_per_subgroup)
            })
            .ok_or_else(|| {
                format!(
                    "GS_INST_PRIMS_IN_SUBGRP({gs_instance_primitives_per_subgroup}) is not an integer multiple of GS_PRIMS_PER_SUBGRP({gs_primitives_per_subgroup})"
                )
            })?;
        let gs_instance_count = programmed_instance_count.unwrap_or(derived_instance_count);
        if gs_instance_count != derived_instance_count {
            return Err(format!(
                "VGT_GS_INSTANCE_CNT={gs_instance_count} disagrees with subgroup primitive ratio {derived_instance_count}"
            ));
        }
        let max_output_primitives_per_subgroup = gs_instance_primitives_per_subgroup
            .checked_mul(primitive_amplification_factor)
            .ok_or("NGG primitive output limit overflow")?;
        let max_input_vertices = if es_vertices_per_subgroup < 256 {
            es_vertices_per_subgroup
        } else {
            gs_instance_primitives_per_subgroup
                .checked_mul(3)
                .ok_or("NGG input vertex workgroup size overflow")?
        };
        let threadgroup_threads = max_input_vertices
            .max(gs_instance_primitives_per_subgroup)
            .max(max_output_vertices_per_subgroup)
            .max(max_output_primitives_per_subgroup);
        let limits = Self {
            es_vertices_per_subgroup,
            gs_primitives_per_subgroup,
            gs_instance_primitives_per_subgroup,
            gs_instance_count,
            gs_max_vertices_out,
            max_output_vertices_per_subgroup,
            primitive_amplification_factor,
            max_output_primitives_per_subgroup,
            output_primitive_type,
            threadgroup_threads,
        };
        if limits.es_vertices_per_subgroup == 0
            || limits.gs_primitives_per_subgroup == 0
            || limits.gs_instance_primitives_per_subgroup == 0
            || limits.gs_instance_count == 0
            || limits.max_output_vertices_per_subgroup == 0
            || limits.primitive_amplification_factor == 0
        {
            return Err(format!("NGG subgroup limits must be non-zero: {limits:?}"));
        }
        // Both D3D12 compute threadgroups and the current Metal mesh ABI are bounded by 256
        // records here. Keep this guest contract independent from either native feature tier.
        if limits.es_vertices_per_subgroup > 256
            || limits.gs_primitives_per_subgroup > 256
            || limits.gs_instance_primitives_per_subgroup > 256
            || limits.max_output_vertices_per_subgroup > 256
            || limits.max_output_primitives_per_subgroup > 256
            || limits.threadgroup_threads > 256
        {
            return Err(format!(
                "NGG subgroup vertices/primitives/threads {}/{}/{} exceed the 256-record backend contract",
                limits.max_output_vertices_per_subgroup,
                limits.max_output_primitives_per_subgroup,
                limits.threadgroup_threads
            ));
        }
        if limits.gs_max_vertices_out != 0 {
            let architectural_max = limits
                .gs_instance_primitives_per_subgroup
                .checked_mul(limits.gs_max_vertices_out)
                .ok_or("NGG subgroup output limit overflow")?;
            if limits.max_output_vertices_per_subgroup > architectural_max {
                return Err(format!(
                    "GE_MAX_OUTPUT_PER_SUBGROUP={} exceeds GS_INST_PRIMS_IN_SUBGRP({}) * VGT_GS_MAX_VERT_OUT({})",
                    limits.max_output_vertices_per_subgroup,
                    limits.gs_instance_primitives_per_subgroup,
                    limits.gs_max_vertices_out
                ));
            }
        }
        Ok(limits)
    }
}

#[must_use]
pub fn is_ngg_export_stage(program: &Program) -> bool {
    let mut allocation = false;
    let mut primitive = false;
    let mut position = false;
    for instruction in &program.instructions {
        allocation |= instruction.opcode.is("SSendmsg")
            && instruction
                .sources
                .first()
                .is_some_and(|source| source.value == 9);
        if let Some(InstructionControl::Export(export)) = instruction.control.as_ref() {
            primitive |= export.target == 20 && export.enable_mask != 0;
            position |= export.target == 12 && export.enable_mask != 0;
        }
    }
    allocation && primitive && position
}

#[must_use]
pub fn ngg_uses_general_lds(program: &Program) -> bool {
    program.instructions.iter().any(|instruction| {
        matches!(instruction.control, Some(InstructionControl::DataShare(_)))
            && !matches!(
                instruction.opcode.known_name(),
                Some("DsSwizzleB32" | "DsWriteAddtidB32" | "DsReadAddtidB32")
            )
    })
}

#[must_use]
pub fn ngg_uses_wave64_exec(program: &Program) -> bool {
    const EXEC_LO: u32 = 126;
    program.instructions.iter().any(|instruction| {
        let Some(name) = instruction.opcode.known_name() else {
            return false;
        };
        (name.ends_with("B64") || name.ends_with("U64") || name.ends_with("I64"))
            && (name.contains("Saveexec")
                || instruction.destinations.first().is_some_and(|destination| {
                    destination.kind == OperandKind::ScalarRegister && destination.value == EXEC_LO
                }))
    })
}

pub fn decode_ngg_subgroup_limits(
    registers: &[GpuRegister],
) -> Result<Option<NggSubgroupLimits>, String> {
    let value = |offset| {
        registers
            .iter()
            .find(|register| (register.offset & !0x7000_0000) == offset)
            .map(|register| register.value)
    };
    let onchip = value(VGT_GS_ONCHIP_CNTL);
    let max_vert_out = value(VGT_GS_MAX_VERT_OUT);
    let max_output = value(GE_MAX_OUTPUT_PER_SUBGROUP);
    let subgroup_control = value(GE_NGG_SUBGRP_CNTL);
    if onchip.is_none() && subgroup_control.is_none() {
        return Ok(None);
    }
    let onchip = onchip.ok_or("NGG state is missing VGT_GS_ONCHIP_CNTL")?;
    let max_vert_out = max_vert_out.ok_or("NGG state is missing VGT_GS_MAX_VERT_OUT")?;
    let max_output = max_output.ok_or("NGG state is missing GE_MAX_OUTPUT_PER_SUBGROUP")?;
    let subgroup_control = subgroup_control.ok_or("NGG state is missing GE_NGG_SUBGRP_CNTL")?;
    if subgroup_control >> 9 != 0 {
        return Err(format!(
            "non-zero GE_NGG_SUBGRP_CNTL.THDS_PER_SUBGRP is not represented: {subgroup_control:#010x}"
        ));
    }
    let output_primitive_type = value(VGT_GS_OUT_PRIM_TYPE)
        .map(|raw| {
            if raw & !0x3f != 0 {
                return Err(format!(
                    "multi-stream VGT_GS_OUT_PRIM_TYPE is not represented: {raw:#010x}"
                ));
            }
            match raw & 0x3f {
                0 => Ok(NggOutputPrimitiveType::PointList),
                1 => Ok(NggOutputPrimitiveType::LineStrip),
                2 => Ok(NggOutputPrimitiveType::TriangleStrip),
                3 | 4 => Ok(NggOutputPrimitiveType::RectangleList),
                kind => Err(format!("unknown VGT_GS_OUT_PRIM_TYPE {kind:#x}")),
            }
        })
        .transpose()?;
    let programmed_instance_count = value(VGT_GS_INSTANCE_CNT)
        .map(|raw| {
            let count = (raw >> 2) & 0x7f;
            if count != 0 {
                Ok(count)
            } else if raw & 1 == 0 {
                Ok(1)
            } else {
                Err(format!(
                    "enabled VGT_GS_INSTANCE_CNT has a zero invocation count: {raw:#010x}"
                ))
            }
        })
        .transpose()?;
    NggSubgroupLimits::new(
        onchip & 0x7ff,
        (onchip >> 11) & 0x7ff,
        (onchip >> 22) & 0x3ff,
        max_vert_out & 0x7ff,
        max_output & 0x3ff,
        subgroup_control & 0x1ff,
        output_primitive_type,
        programmed_instance_count,
    )
    .map(Some)
}

/// Packs ordered triangle input into the merged NGG shader's fixed 514-dword workgroup ABI.
#[must_use]
pub fn prepare_ngg_meshlets(indices: &[u32], limits: NggSubgroupLimits) -> PreparedNggMeshlets {
    const STORAGE_VERTICES: usize = 256;
    const STORAGE_PRIMITIVES: usize = 256;
    let primitive_count = indices.len() / 3;
    let max_vertices = limits.es_vertices_per_subgroup as usize;
    let max_primitives = limits.gs_primitives_per_subgroup as usize;
    let mut words = Vec::new();
    let mut vertices = Vec::<u32>::with_capacity(max_vertices);
    let mut slots = HashMap::<u32, u32>::with_capacity(max_vertices);
    let mut primitives = Vec::<u32>::with_capacity(max_primitives);
    let mut first_primitive = 0_usize;
    let flush = |words: &mut Vec<u32>,
                 vertices: &mut Vec<u32>,
                 slots: &mut HashMap<u32, u32>,
                 primitives: &mut Vec<u32>,
                 first_primitive: usize| {
        if primitives.is_empty() {
            return;
        }
        let vertex_count = vertices.len() as u32;
        let primitive_count = primitives.len() as u32;
        words.push((vertex_count & 0xff) | ((primitive_count & 0xff) << 8));
        words.push(first_primitive as u32);
        words.extend(vertices.iter().copied());
        words.resize(words.len() + STORAGE_VERTICES - vertices.len(), 0);
        words.extend(primitives.iter().copied());
        words.resize(
            words.len() + STORAGE_PRIMITIVES - primitives.len(),
            0x8000_0000,
        );
        vertices.clear();
        slots.clear();
        primitives.clear();
    };
    for (primitive, triangle) in indices.chunks_exact(3).enumerate() {
        let additional = triangle
            .iter()
            .enumerate()
            .filter(|(component, index)| {
                !slots.contains_key(index) && !triangle[..*component].contains(index)
            })
            .count();
        if !primitives.is_empty()
            && (primitives.len() == max_primitives || vertices.len() + additional > max_vertices)
        {
            flush(
                &mut words,
                &mut vertices,
                &mut slots,
                &mut primitives,
                first_primitive,
            );
            first_primitive = primitive;
        }
        let mut local = [0_u32; 3];
        for (component, &index) in triangle.iter().enumerate() {
            local[component] = *slots.entry(index).or_insert_with(|| {
                let slot = vertices.len() as u32;
                vertices.push(index);
                slot
            });
        }
        primitives.push(local[0] | (local[1] << 10) | (local[2] << 20));
    }
    flush(
        &mut words,
        &mut vertices,
        &mut slots,
        &mut primitives,
        first_primitive,
    );
    let vertex_references = words
        .chunks_exact(NGG_MESHLET_WORDS_PER_GROUP)
        .map(|group| {
            let encoded = group[0] & 0xff;
            if encoded == 0 { 256 } else { encoded as usize }
        })
        .sum();
    PreparedNggMeshlets {
        group_count: (words.len() / NGG_MESHLET_WORDS_PER_GROUP) as u32,
        words: words.into(),
        vertex_references,
        primitive_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_subgroup_registers_decode_without_a_native_mesh_dependency() {
        let registers = [
            GpuRegister {
                offset: GE_MAX_OUTPUT_PER_SUBGROUP,
                value: 64,
            },
            GpuRegister {
                offset: VGT_GS_ONCHIP_CNTL,
                value: 64 | (32 << 11) | (32 << 22),
            },
            GpuRegister {
                offset: VGT_GS_OUT_PRIM_TYPE,
                value: 2,
            },
            GpuRegister {
                offset: VGT_GS_MAX_VERT_OUT,
                value: 4,
            },
            GpuRegister {
                offset: GE_NGG_SUBGRP_CNTL,
                value: 2,
            },
        ];
        let limits = decode_ngg_subgroup_limits(&registers)
            .expect("valid registers")
            .expect("NGG state");
        assert_eq!(limits.threadgroup_threads, 64);
        assert_eq!(limits.max_output_primitives_per_subgroup, 64);
        assert_eq!(
            limits.output_primitive_type,
            Some(NggOutputPrimitiveType::TriangleStrip)
        );
    }

    #[test]
    fn meshlets_preserve_reused_vertices_and_fixed_group_offsets() {
        let limits = NggSubgroupLimits::new(
            64,
            32,
            32,
            4,
            64,
            2,
            Some(NggOutputPrimitiveType::TriangleStrip),
            None,
        )
        .expect("limits");
        let packed = prepare_ngg_meshlets(&[10, 11, 12, 12, 11, 13], limits);
        assert_eq!(packed.group_count, 1);
        assert_eq!(packed.words.len(), NGG_MESHLET_WORDS_PER_GROUP);
        assert_eq!(packed.words[0], 4 | (2 << 8));
        assert_eq!(&packed.words[2..6], &[10, 11, 12, 13]);
        assert_eq!(packed.words[258], (1 << 10) | (2 << 20));
        assert_eq!(packed.words[259], 2 | (1 << 10) | (3 << 20));
    }
}
