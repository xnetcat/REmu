// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CB_BLEND0_CONTROL, CB_TARGET_MASK, CX_GRAPHICS_SCRATCH_RING_SIZE, CX_VGT_SHADER_STAGES_EN,
    CX_VGT_SHADER_STAGES_EN_GS_WAVE32, DB_DEPTH_CONTROL, DB_SHADER_CONTROL,
    DEFAULT_CB_BLEND_CONTROL, DEFAULT_PA_SU_SC_MODE_CNTL, DecodedDepthBias, DecodedRasterState,
    DecodedRenderTarget, DepthTargetPipelineState, GE_MAX_OUTPUT_PER_SUBGROUP, GE_NGG_SUBGRP_CNTL,
    GraphicsDiagnosticPipelineState, GraphicsPipelineKey, GraphicsShaderResourceKey,
    GraphicsWaveState, MetalBlendAttachmentState, NggOutputPrimitiveType, NggSubgroupLimits,
    PA_CL_VS_OUT_CNTL, PA_CL_VTE_CNTL, PA_SU_SC_MODE_CNTL, ProgramIdentity,
    RenderTargetPipelineState, SPI_SHADER_COL_FORMAT, UC_GE_CNTL, VGT_ESGS_RING_ITEMSIZE,
    VGT_GS_INSTANCE_CNT, VGT_GS_MAX_VERT_OUT, VGT_GS_ONCHIP_CNTL, VGT_GS_OUT_PRIM_TYPE,
    common_render_target_array_length, decode_depth_target, decode_render_targets,
    diagnostic_capture_pixel_output, diagnostic_capture_vertex_position, diagnostic_dump_gbuffer,
    diagnostic_dump_glyph_draw, diagnostic_dump_shadow_consumer, diagnostic_dump_target_address,
    diagnostic_dump_target_address_range, diagnostic_dump_target_minimum_draw,
    diagnostic_dump_target_mrt_min, diagnostic_dump_target_pixel_instruction_count,
    diagnostic_dump_target_pixel_shader, diagnostic_dump_video_chain_consumer,
    diagnostic_dump_video_draw, diagnostic_dump_video_target_consumer,
    diagnostic_force_msaa_pixel_color, diagnostic_force_msaa_vertex_pattern,
    diagnostic_force_vertex_fetch_id, diagnostic_force_visible_clip_depth,
    diagnostic_note_full_resolution_mrt_draw, diagnostic_target_address_matches,
    diagnostic_video_chain_selected_draw, evaluation_samples_glyph_atlas,
    evaluation_samples_large_linear_video, evaluation_samples_layered_comparison,
    fragmentless_pixel_evaluation, fragmentless_pixel_program,
    graphics_allocated_vector_register_count, image_component_class, is_storage_image_opcode,
    ngg_memoryless_cover_export_stage, ngg_mesh_export_stage, ngg_mesh_stage_for_draw,
    pixel_input_controls, procedural_rectangle_stage_for_draw, program_embeds_global_base,
    register_value, video_target_binding_summary,
};
use metal::{
    MTLBlendFactor, MTLBlendOperation, MTLColorWriteMask, MTLCullMode, MTLPrimitiveTopologyClass,
    MTLPrimitiveType, MTLTriangleFillMode, MTLWinding,
};
use remu_gpu::IndexedDraw;
use remu_shader::{PixelExportFormat, ShaderEvaluation, image_is_comparison};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

pub(crate) const NGG_MESHLET_WORDS_PER_GROUP: usize = 514;
const NGG_MESHLET_STORAGE_VERTICES: usize = 256;
const NGG_MESHLET_STORAGE_PRIMITIVES: usize = 256;
const LEGACY_NGG_MESHLET_WORDS_PER_GROUP: usize = 130;
const LEGACY_NGG_MESHLET_STORAGE_VERTICES: usize = 64;
const LEGACY_NGG_MESHLET_STORAGE_PRIMITIVES: usize = 64;
const METAL_MESH_MAX_OUTPUTS: u32 = 256;
const LEGACY_NGG_FIXED_ALLOCATION: u32 = 64;

/// Cover topologies that memoryless NGG stages emit for fullscreen blits and that currently shard
/// under the meshlet ABI. Vertex-fetching NGG programs with the same topologies are excluded by
/// [`ngg_memoryless_cover_export_stage`].
///
/// Indexed two-triangle quads (`count=6`) are not covers: positioned UI/sprite quads share that
/// topology with memoryless NGG exports, and synthesizing a fullscreen triangle replaces their
/// bounds with solid cells. Keep single-triangle (`count=3`) covers and non-indexed strip covers.
///
/// Resource-fetching Poppy/UE YUV strips carry their visible-width UV extent in a tiny guest vertex
/// record. Replacing those stages with the synthetic cover expands UVs across the padded texture
/// pitch and exposes the NV12 padding as a green band. Only a genuinely memoryless stage may use
/// the synthetic cover; every stage with guest resource traffic retains its translated vertex
/// program and explicit PARAM exports.
pub(crate) fn ngg_fullscreen_cover_draw(
    program: &remu_shader::Program,
    draw: &IndexedDraw,
) -> bool {
    if draw.indirect_arguments_address.is_some() {
        return false;
    }
    let strip_cover = !draw.indexed && draw.primitive_type == 6 && draw.index_count == 4;
    let indexed_triangle_cover = draw.indexed
        && draw.primitive_type == 4
        && draw.index_count == 3
        && draw.instance_count == 1;
    if strip_cover {
        ngg_memoryless_cover_export_stage(program)
    } else if indexed_triangle_cover {
        ngg_memoryless_cover_export_stage(program)
    } else {
        false
    }
}

/// Returns whether a draw is a UE/Octopath memoryless cover placeholder with no attachments.
///
/// Boot and frame-boundary fences occasionally emit ps=0 draws whose `CB_TARGET_MASK` is zero and
/// whose depth block is inactive. They carry no renderable state and exist only to preserve queue
/// ordering until the next real pass.
pub(crate) fn noop_placeholder_cover_draw(draw: &IndexedDraw) -> bool {
    if draw.pixel_shader_address != 0 {
        return false;
    }
    let target_mask = register_value(&draw.context_registers, CB_TARGET_MASK).unwrap_or_default();
    if target_mask != 0 {
        return false;
    }
    if decode_depth_target(draw).ok().flatten().is_some() {
        return false;
    }
    draw.export_shader.as_ref().is_some_and(|shader| {
        procedural_rectangle_stage_for_draw(&shader.program, draw)
            || ngg_memoryless_cover_export_stage(&shader.program)
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PreparedNggMeshlets {
    pub(crate) words: Arc<[u32]>,
    pub(crate) group_count: u64,
    pub(crate) vertex_references: usize,
    pub(crate) primitive_count: usize,
}

/// Expands a non-indexed triangle strip into the triangle-list indices consumed by NGG meshlets.
///
/// GFX10 alternates strip winding for each new vertex. Metal mesh shaders instead publish an
/// explicit index triplet per primitive, so odd primitives swap their first two vertices.
fn expand_nonindexed_triangle_strip(
    first_vertex: u32,
    vertex_count: u32,
) -> Result<Arc<[u32]>, String> {
    let primitive_count = vertex_count.saturating_sub(2);
    if primitive_count != 0 {
        first_vertex
            .checked_add(vertex_count - 1)
            .ok_or("triangle-strip vertex ID overflow")?;
    }
    let index_count = usize::try_from(primitive_count)
        .map_err(|_| "triangle-strip primitive count does not fit usize")?
        .checked_mul(3)
        .ok_or("triangle-strip index count overflow")?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(index_count)
        .map_err(|_| format!("triangle-strip index allocation failed for {index_count} entries"))?;
    for primitive in 0..primitive_count {
        let first = first_vertex
            .checked_add(primitive)
            .ok_or("triangle-strip vertex ID overflow")?;
        let second = first
            .checked_add(1)
            .ok_or("triangle-strip vertex ID overflow")?;
        let third = first
            .checked_add(2)
            .ok_or("triangle-strip vertex ID overflow")?;
        if primitive & 1 == 0 {
            indices.extend_from_slice(&[first, second, third]);
        } else {
            indices.extend_from_slice(&[second, first, third]);
        }
    }
    Ok(indices.into())
}

/// Expands indexed triangle-strip indices into the triangle-list form NGG meshlets consume.
///
/// Odd primitives reverse the first two vertices to preserve GFX10 strip winding. Segments that
/// already went through primitive-restart expansion arrive as triangle lists and must not be
/// expanded again.
pub(crate) fn expand_triangle_strip_indices(indices: &[u32]) -> Arc<[u32]> {
    let primitive_count = indices.len().saturating_sub(2);
    let mut prepared = Vec::with_capacity(primitive_count.saturating_mul(3));
    for (triangle, vertices) in indices.windows(3).enumerate() {
        if triangle & 1 == 0 {
            prepared.extend_from_slice(vertices);
        } else {
            prepared.extend_from_slice(&[vertices[1], vertices[0], vertices[2]]);
        }
    }
    prepared.into()
}

/// Expands a point list into the triangle-shaped meshlet ABI.
///
/// Each point becomes one input primitive whose three packed local slots reference the same
/// vertex. Merged NGG shaders that amplify points into triangles only consume the first vertex of
/// each input primitive; duplicating the index keeps the existing triangle meshlet packer intact.
pub(crate) fn expand_point_list_indices(indices: &[u32]) -> Arc<[u32]> {
    let mut prepared = Vec::with_capacity(indices.len().saturating_mul(3));
    for &index in indices {
        prepared.extend_from_slice(&[index, index, index]);
    }
    prepared.into()
}

/// Prepares source vertex IDs for a non-indexed NGG triangle-strip draw.
///
/// `index_offset` is the guest draw's first vertex. `base_vertex` belongs to indexed draws and
/// does not participate in non-indexed auto-generated vertex IDs.
pub(crate) fn prepare_nonindexed_ngg_indices(draw: &IndexedDraw) -> Result<Arc<[u32]>, String> {
    match (draw.indexed, draw.primitive_type) {
        (false, 6) => expand_nonindexed_triangle_strip(draw.index_offset, draw.index_count),
        (false, 4) => expand_nonindexed_triangle_list(draw.index_offset, draw.index_count),
        (false, 1) => {
            let count = usize::try_from(draw.index_count)
                .map_err(|_| "point-list vertex count does not fit usize")?;
            let mut indices = Vec::new();
            indices
                .try_reserve_exact(count)
                .map_err(|_| format!("point-list index allocation failed for {count} entries"))?;
            for offset in 0..draw.index_count {
                let index = draw
                    .index_offset
                    .checked_add(offset)
                    .ok_or("point-list vertex ID overflow")?;
                indices.push(index);
            }
            Ok(expand_point_list_indices(&indices))
        }
        _ => Err("NGG source is not a non-indexed triangle list/strip or point list".to_owned()),
    }
}

/// Synthesizes consecutive vertex IDs for a non-indexed triangle list.
///
/// Incomplete trailing vertices (count not divisible by 3) are dropped, matching GFX10's
/// ignore-partial-primitive rule.
fn expand_nonindexed_triangle_list(
    first_vertex: u32,
    vertex_count: u32,
) -> Result<Arc<[u32]>, String> {
    let complete = vertex_count - vertex_count % 3;
    if complete != 0 {
        first_vertex
            .checked_add(complete - 1)
            .ok_or("triangle-list vertex ID overflow")?;
    }
    let index_count =
        usize::try_from(complete).map_err(|_| "triangle-list vertex count does not fit usize")?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(index_count)
        .map_err(|_| format!("triangle-list index allocation failed for {index_count} entries"))?;
    for offset in 0..complete {
        let index = first_vertex
            .checked_add(offset)
            .ok_or("triangle-list vertex ID overflow")?;
        indices.push(index);
    }
    Ok(indices.into())
}

/// Prepares triangle-list indices for an indexed NGG mesh draw.
///
/// Indexed triangle lists pass through. Indexed triangle strips expand unless
/// [`prepare_indexed_primitive`] already converted them via primitive restart. Point lists are
/// triplicated so the triangle meshlet packer can feed one input primitive per point.
pub(crate) fn prepare_indexed_ngg_indices(
    draw: &IndexedDraw,
    indices: Arc<[u32]>,
    metal_primitive: MTLPrimitiveType,
) -> Result<Arc<[u32]>, String> {
    match draw.primitive_type {
        4 => Ok(indices),
        1 => Ok(expand_point_list_indices(indices.as_ref())),
        6 if matches!(metal_primitive, MTLPrimitiveType::Triangle) => Ok(indices),
        6 if matches!(metal_primitive, MTLPrimitiveType::TriangleStrip) => {
            Ok(expand_triangle_strip_indices(indices.as_ref()))
        }
        _ => Err(format!(
            "NGG source is not an indexed triangle list/strip/point (topology={:#x}, metal={metal_primitive:?})",
            draw.primitive_type
        )),
    }
}

/// Returns the signed vertex-ID adjustment consumed by the native mesh ABI.
pub(crate) const fn ngg_mesh_base_vertex(draw: &IndexedDraw) -> i32 {
    if draw.indexed { draw.base_vertex } else { 0 }
}

/// The legacy fixed allocation is selected from the actual fused-program shape. Context GS
/// registers persist across draws and therefore cannot prove that an ordinary no-GS primitive
/// shader has a linked API geometry stage.
pub(crate) fn ngg_fixed_allocation_fallback_for_program(program: &remu_shader::Program) -> bool {
    !program
        .instructions
        .iter()
        .any(|instruction| program.linked_transfer_target(instruction.pc).is_some())
}

impl NggSubgroupLimits {
    /// No API geometry shader means `VGT_GS_MAX_VERT_OUT` cannot prove that the guest's
    /// `GS_ALLOC_REQ` payload describes the passthrough primitive shader's live output. REmu's
    /// original, known-good NGG path executed these stages as one fixed wave64 and ignored that
    /// payload. Keep the register-sized path for a real GS, but retain the fixed allocation for
    /// this otherwise ambiguous no-GS ABI.
    pub(crate) const fn uses_fixed_allocation_fallback(self) -> bool {
        self.fixed_allocation_fallback
    }

    pub(crate) const fn with_fixed_allocation_fallback(mut self, enabled: bool) -> Self {
        self.fixed_allocation_fallback = enabled;
        self
    }

    pub(crate) const fn meshlet_vertex_limit(self) -> u32 {
        if self.uses_fixed_allocation_fallback() {
            LEGACY_NGG_FIXED_ALLOCATION
        } else {
            self.es_vertices_per_subgroup
        }
    }

    pub(crate) const fn meshlet_primitive_limit(self) -> u32 {
        if self.uses_fixed_allocation_fallback() {
            LEGACY_NGG_FIXED_ALLOCATION
        } else {
            self.gs_primitives_per_subgroup
        }
    }

    pub(crate) const fn metal_threadgroup_threads(self) -> u32 {
        if self.uses_fixed_allocation_fallback() {
            LEGACY_NGG_FIXED_ALLOCATION
        } else {
            self.threadgroup_threads
        }
    }

    pub(crate) const fn meshlet_storage_vertices(self) -> usize {
        if self.uses_fixed_allocation_fallback() {
            LEGACY_NGG_MESHLET_STORAGE_VERTICES
        } else {
            NGG_MESHLET_STORAGE_VERTICES
        }
    }

    pub(crate) const fn meshlet_storage_primitives(self) -> usize {
        if self.uses_fixed_allocation_fallback() {
            LEGACY_NGG_MESHLET_STORAGE_PRIMITIVES
        } else {
            NGG_MESHLET_STORAGE_PRIMITIVES
        }
    }

    pub(crate) const fn meshlet_words_per_group(self) -> usize {
        if self.uses_fixed_allocation_fallback() {
            LEGACY_NGG_MESHLET_WORDS_PER_GROUP
        } else {
            NGG_MESHLET_WORDS_PER_GROUP
        }
    }

    pub(crate) fn new(
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
            // Standalone register decoding cannot inspect the shader program. Preserve the old
            // zero-register fallback for decoder callers; pipeline-key construction below
            // replaces this proxy with the actual linked-stage classification.
            fixed_allocation_fallback: gs_max_vertices_out == 0,
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
        if limits.es_vertices_per_subgroup > 256
            || limits.gs_primitives_per_subgroup > 256
            || limits.gs_instance_primitives_per_subgroup > 256
        {
            return Err(format!(
                "NGG input subgroup {limits:?} exceeds Metal's 256-thread mesh limit"
            ));
        }
        if limits.max_output_vertices_per_subgroup > METAL_MESH_MAX_OUTPUTS
            || limits.max_output_primitives_per_subgroup > METAL_MESH_MAX_OUTPUTS
            || limits.threadgroup_threads > METAL_MESH_MAX_OUTPUTS
        {
            return Err(format!(
                "NGG subgroup vertices/primitives/threads {}/{}/{} exceed Metal's limit of {METAL_MESH_MAX_OUTPUTS}",
                limits.max_output_vertices_per_subgroup,
                limits.max_output_primitives_per_subgroup,
                limits.threadgroup_threads
            ));
        }
        // VGT_GS_MAX_VERT_OUT is the per-input-primitive clamp for an API geometry shader. A
        // zero value leaves that GS-only clamp inactive in passthrough NGG pipelines; their
        // subgroup output storage remains independently and explicitly bounded by
        // GE_MAX_OUTPUT_PER_SUBGROUP. Apply the derived GS bound only when the guest programs it.
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

#[cfg(test)]
pub(crate) fn decode_ngg_subgroup_limits(
    context_registers: &[remu_gpu::GpuRegister],
) -> Result<Option<NggSubgroupLimits>, String> {
    decode_ngg_subgroup_limits_with_uc(context_registers, &[])
}

/// Decode NGG subgroup limits from Cx state, falling back to SDK `UcGeCntl` sizing when the
/// AGC-invisible Cx offsets (0x291 / 0x2d3 / 0x2e4) were never written.
pub(crate) fn decode_ngg_subgroup_limits_with_uc(
    context_registers: &[remu_gpu::GpuRegister],
    user_config_registers: &[remu_gpu::GpuRegister],
) -> Result<Option<NggSubgroupLimits>, String> {
    let onchip = register_value(context_registers, VGT_GS_ONCHIP_CNTL);
    let max_vert_out = register_value(context_registers, VGT_GS_MAX_VERT_OUT);
    let max_output = register_value(context_registers, GE_MAX_OUTPUT_PER_SUBGROUP);
    let subgroup_control = register_value(context_registers, GE_NGG_SUBGRP_CNTL);
    let output_primitive = register_value(context_registers, VGT_GS_OUT_PRIM_TYPE);
    let instance_control = register_value(context_registers, VGT_GS_INSTANCE_CNT);
    let stages = register_value(context_registers, CX_VGT_SHADER_STAGES_EN).unwrap_or(0x2000);
    let ge_cntl = register_value(user_config_registers, UC_GE_CNTL).unwrap_or(0);
    let ge_prim_group = ge_cntl & 0x1ff;
    let ge_vert_group = (ge_cntl >> 9) & 0x1ff;
    let gs_enabled = stages & 0x20 != 0;
    let passthrough = stages & 0x0200_0000 != 0;

    // Pure AGC statebuffers never program Cx 0x291/0x2d3/0x2e4. Derive sizing from UcGeCntl and
    // CxVgtShaderStagesEn when those legacy Cx knobs are absent.
    let derived_from_ge = onchip.is_none()
        && subgroup_control.is_none()
        && (ge_prim_group != 0 || ge_vert_group != 0 || gs_enabled || passthrough);
    if onchip.is_none() && subgroup_control.is_none() && !derived_from_ge {
        return Ok(None);
    }
    let onchip = onchip.unwrap_or_else(|| {
        // Pack UcGeCntl group sizes into the VGT_GS_ONCHIP_CNTL field layout the limits builder
        // already understands: ES_VERTS | GS_PRIMS<<11 | GS_INST_PRIMS<<22. With no programmed
        // GS instance count, instance primitives equal the prim group (one invocation).
        let prims = ge_prim_group.max(1);
        let verts = ge_vert_group.max(1);
        verts | (prims << 11) | (prims << 22)
    });
    // UcGeCntl contains *input* primitive/vertex group sizes only. It cannot stand in for either
    // the API-GS output count or GE_NGG_SUBGRP_CNTL.PRIM_AMP_FACTOR. With no API GS, NGG has an
    // exact amplification factor of one. With a GS, hardware derives PRIM_AMP_FACTOR from
    // VGT_GS_MAX_VERT_OUT, so that Cx register must be present rather than inventing a value from
    // VERT_GRP_SIZE.
    let max_vert_out = if gs_enabled {
        max_vert_out.ok_or_else(|| {
            "GS-enabled UcGeCntl fallback is missing VGT_GS_MAX_VERT_OUT; input VERT_GRP_SIZE cannot prove primitive amplification"
                .to_owned()
        })?
    } else {
        // VGT_GS_MAX_VERT_OUT is inactive without an API GS. Ignore a stale/default Cx value;
        // carrying it into the common validator can incorrectly cap passthrough output.
        0
    };
    let derived_primitive_amplification = if gs_enabled {
        let vertices = max_vert_out & 0x7ff;
        if vertices == 0 || vertices > 0x1ff {
            return Err(format!(
                "GS-enabled VGT_GS_MAX_VERT_OUT={vertices} cannot encode GE_NGG_SUBGRP_CNTL.PRIM_AMP_FACTOR"
            ));
        }
        vertices
    } else {
        1
    };
    let subgroup_control = subgroup_control.unwrap_or(derived_primitive_amplification);
    let max_output = max_output.unwrap_or_else(|| {
        if gs_enabled {
            ge_prim_group
                .max(1)
                .saturating_mul(derived_primitive_amplification)
        } else {
            ge_vert_group.max(1)
        }
    });
    if subgroup_control >> 9 != 0 {
        return Err(format!(
            "non-zero GE_NGG_SUBGRP_CNTL.THDS_PER_SUBGRP is not represented: {subgroup_control:#010x}"
        ));
    }
    let output_primitive_type = output_primitive
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
                // Prospero `GsOutputPrimitiveType`: k2dRectangle=3, kRectList=4. Both describe
                // the three-vertex AMD rectangle-list contract that host paths expand to a
                // four-corner triangle strip (Dreaming Sarah Gen5 UI covers, fullscreen blits).
                3 | 4 => Ok(NggOutputPrimitiveType::RectangleList),
                kind => Err(format!("unknown VGT_GS_OUT_PRIM_TYPE {kind:#x}")),
            }
        })
        .transpose()?;
    let programmed_instance_count = instance_control
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

/// Decodes the graphics temp-ring allocation and the export stage's architectural wave width.
///
/// SDK 10 `registerstructs.h`:
/// * `CxGraphicsScratchRingSize` (Cx 0x1ba) — `getWaveSize()` reads bits [24:12]; the gnmp interop
///   helper `setGraphicsScratchSize(maxNumWaves, num1KByteChunksPerWave)` gives the unit as 1 KiB
///   per architectural wave, the same encoding `ShComputeScratchRingSize` uses for compute.
/// * `CxVgtShaderStagesEn` (Cx 0x2d5) — `GsWaveSize` (bit 22) and `HsWaveSize` (bit 21) are the only
///   wave-width fields AGC exposes, both documented as `k32`/`k64` with a default of `k64`. There is
///   no pixel-stage counterpart anywhere in the SDK, so the pixel wave width stays unproven.
pub(crate) fn decode_graphics_wave_state(
    context_registers: &[remu_gpu::GpuRegister],
) -> GraphicsWaveState {
    GraphicsWaveState {
        scratch_bytes_per_wave: register_value(context_registers, CX_GRAPHICS_SCRATCH_RING_SIZE)
            .map_or(0, |ring| ((ring >> 12) & 0x1fff) * 1024),
        export_lane_count: register_value(context_registers, CX_VGT_SHADER_STAGES_EN).map(
            |stages| {
                if stages & CX_VGT_SHADER_STAGES_EN_GS_WAVE32 == 0 {
                    64
                } else {
                    32
                }
            },
        ),
    }
}

/// Captures the raw SDK register state required to prove NGG mesh lowering.
///
/// `SPI_SHADER_PGM_RSRC1_*` is two SH-register slots before the merged stage's user-data base.
/// The other values are context registers. In particular, SDK 10
/// `CxGeMaxOutputPerSubgroup::getMaxVertsPerSubgroup()` reads the low ten bits of
/// `GE_MAX_OUTPUT_PER_SUBGROUP`; `GE_NGG_SUBGRP_CNTL` instead controls subgroup threads and must
/// never be substituted for the output capacity.
pub(crate) fn decode_ngg_registers_from_state(
    shader_registers: &[remu_gpu::GpuRegister],
    context_registers: &[remu_gpu::GpuRegister],
    export_user_data_register: u32,
) -> Result<remu_shader_metal::NggRegisters, String> {
    let pgm_rsrc1 = register_value(
        shader_registers,
        export_user_data_register.saturating_sub(2),
    )
    .ok_or("NGG state is missing merged-stage SPI_SHADER_PGM_RSRC1")?;
    // SDK `ShShaderResource2Gs_0 = 0x008b` sits immediately before the merged-stage user-data
    // base (`0x08c`). Same relative slot for the LS/ES user-data bases.
    let pgm_rsrc2 = register_value(
        shader_registers,
        export_user_data_register.saturating_sub(1),
    )
    .ok_or("NGG state is missing merged-stage SPI_SHADER_PGM_RSRC2")?;
    let gs_max_vert_out = register_value(context_registers, VGT_GS_MAX_VERT_OUT)
        .ok_or("NGG state is missing VGT_GS_MAX_VERT_OUT")?;
    let ge_max_output_per_subgroup = register_value(context_registers, GE_MAX_OUTPUT_PER_SUBGROUP)
        .ok_or("NGG state is missing GE_MAX_OUTPUT_PER_SUBGROUP")?;
    let esgs_ring_itemsize =
        register_value(context_registers, VGT_ESGS_RING_ITEMSIZE).unwrap_or_default();
    let primgen_passthrough = register_value(context_registers, CX_VGT_SHADER_STAGES_EN)
        .is_some_and(|stages| stages & 0x0200_0000 != 0);
    Ok(remu_shader_metal::NggRegisters {
        pgm_rsrc1,
        pgm_rsrc2,
        gs_max_vert_out,
        ge_max_output_per_subgroup,
        esgs_ring_itemsize,
        primgen_passthrough,
    })
}

/// Builds the GFX10 NGG workgroups consumed by a merged export/geometry shader.
///
/// The guest registers independently limit unique ES vertices and GS primitives in each subgroup.
/// Preserve index reuse by greedily packing consecutive triangles until either declared input
/// limit is reached. The backing record has the architectural 256-entry maximum so changing guest
/// limits never changes the Metal-side offsets.
pub(crate) fn prepare_ngg_meshlets(
    indices: &[u32],
    limits: NggSubgroupLimits,
) -> PreparedNggMeshlets {
    let primitive_count = indices.len() / 3;
    let max_vertices = usize::try_from(limits.meshlet_vertex_limit())
        .expect("validated NGG vertex count fits usize");
    let max_primitives = usize::try_from(limits.meshlet_primitive_limit())
        .expect("validated NGG primitive count fits usize");
    let storage_vertices = limits.meshlet_storage_vertices();
    let storage_primitives = limits.meshlet_storage_primitives();
    let words_per_group = limits.meshlet_words_per_group();
    let mut words = Vec::new();
    let mut vertices = Vec::<u32>::with_capacity(max_vertices);
    let mut vertex_slots = HashMap::<u32, u32>::with_capacity(max_vertices);
    let mut primitives = Vec::<u32>::with_capacity(max_primitives);
    let mut first_primitive = 0_usize;

    let flush = |words: &mut Vec<u32>,
                 vertices: &mut Vec<u32>,
                 vertex_slots: &mut HashMap<u32, u32>,
                 primitives: &mut Vec<u32>,
                 first_primitive: usize| {
        if primitives.is_empty() {
            return;
        }
        let vertex_count = u32::try_from(vertices.len()).expect("NGG vertex count fits u32");
        let primitive_count =
            u32::try_from(primitives.len()).expect("NGG primitive count fits u32");
        // The merged-shader ABI stores each count in eight bits; 256 is encoded as zero.
        words.push((vertex_count & 0xff) | ((primitive_count & 0xff) << 8));
        words.push(u32::try_from(first_primitive).expect("NGG primitive ID fits u32"));
        words.extend(vertices.iter().copied());
        words.resize(words.len() + storage_vertices - vertices.len(), 0);
        words.extend(primitives.iter().copied());
        words.resize(
            words.len() + storage_primitives - primitives.len(),
            0x8000_0000,
        );
        vertices.clear();
        vertex_slots.clear();
        primitives.clear();
    };

    for (primitive, triangle) in indices.chunks_exact(3).enumerate() {
        let additional_vertices = triangle
            .iter()
            .enumerate()
            .filter(|(component, index)| {
                !vertex_slots.contains_key(index) && !triangle[..*component].contains(index)
            })
            .count();
        if !primitives.is_empty()
            && (primitives.len() == max_primitives
                || vertices.len() + additional_vertices > max_vertices)
        {
            flush(
                &mut words,
                &mut vertices,
                &mut vertex_slots,
                &mut primitives,
                first_primitive,
            );
            first_primitive = primitive;
        }

        let mut local = [0_u32; 3];
        for (component, &index) in triangle.iter().enumerate() {
            local[component] = *vertex_slots.entry(index).or_insert_with(|| {
                let slot = u32::try_from(vertices.len()).expect("NGG vertex slot fits u32");
                vertices.push(index);
                slot
            });
        }
        primitives.push(local[0] | (local[1] << 10) | (local[2] << 20));
    }
    flush(
        &mut words,
        &mut vertices,
        &mut vertex_slots,
        &mut primitives,
        first_primitive,
    );

    // Count all packed vertex records rather than only the final workgroup.
    let vertex_references = words
        .chunks_exact(words_per_group)
        .map(|group| {
            let encoded = group[0] & 0xff;
            usize::try_from(if encoded == 0 { 256 } else { encoded })
                .expect("vertex count fits usize")
        })
        .sum();
    let group_count =
        u64::try_from(words.len() / words_per_group).expect("NGG meshlet group count fits u64");
    PreparedNggMeshlets {
        words: words.into(),
        group_count,
        vertex_references,
        primitive_count,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ColorOperationMode {
    Disable,
    Normal,
    EliminateFastClear,
    Resolve,
    FmaskDecompress,
}

pub(crate) fn decode_color_operation_mode(control: u32) -> Result<ColorOperationMode, String> {
    match (control >> 4) & 0x7 {
        0 => Ok(ColorOperationMode::Disable),
        1 => Ok(ColorOperationMode::Normal),
        2 => Ok(ColorOperationMode::EliminateFastClear),
        3 => Ok(ColorOperationMode::Resolve),
        5 => Ok(ColorOperationMode::FmaskDecompress),
        mode => Err(format!(
            "unsupported CB_COLOR_CONTROL operation mode {mode} in {control:#010x}"
        )),
    }
}

pub(crate) const fn color_operation_is_raster_noop(
    operation: ColorOperationMode,
    has_active_depth: bool,
) -> bool {
    matches!(operation, ColorOperationMode::Disable) && !has_active_depth
}

pub(crate) fn metal_primitive_type(guest: u32) -> Result<MTLPrimitiveType, String> {
    match guest {
        1 => Ok(MTLPrimitiveType::Point),
        2 => Ok(MTLPrimitiveType::Line),
        3 => Ok(MTLPrimitiveType::LineStrip),
        4 => Ok(MTLPrimitiveType::Triangle),
        // GFX10's 2D rectangle and rectangle-list topologies consume three guest vertices but
        // rasterize a complete four-corner rectangle. The observed AGC shaders derive the fourth
        // corner from vertex ID, so invoking the translated vertex function once more preserves
        // the native rectangle while using Metal's triangle-strip topology.
        6 | 7 | 0x11 => Ok(MTLPrimitiveType::TriangleStrip),
        5 => Err("triangle-fan topology requires explicit Metal index expansion".to_owned()),
        _ => Err(format!("unknown guest primitive topology {guest:#x}")),
    }
}

pub(crate) fn prepare_indexed_primitive(
    draw: &IndexedDraw,
    indices: Arc<[u32]>,
) -> Result<(Arc<[u32]>, MTLPrimitiveType), String> {
    let primitive = metal_primitive_type(draw.primitive_type)?;
    let Some(restart) = primitive_restart_index(draw) else {
        return Ok((indices, primitive));
    };
    if !indices.contains(&restart) {
        return Ok((indices, primitive));
    }

    let mut prepared = Vec::with_capacity(indices.len());
    let primitive = match draw.primitive_type {
        1 | 2 | 4 => {
            let vertices_per_primitive = match draw.primitive_type {
                1 => 1,
                2 => 2,
                _ => 3,
            };
            for segment in indices.split(|&index| index == restart) {
                let complete = segment.len() / vertices_per_primitive * vertices_per_primitive;
                prepared.extend_from_slice(&segment[..complete]);
            }
            primitive
        }
        3 => {
            for segment in indices.split(|&index| index == restart) {
                for pair in segment.windows(2) {
                    prepared.extend_from_slice(pair);
                }
            }
            MTLPrimitiveType::Line
        }
        6 => {
            for segment in indices.split(|&index| index == restart) {
                prepared.extend_from_slice(expand_triangle_strip_indices(segment).as_ref());
            }
            MTLPrimitiveType::Triangle
        }
        _ => {
            return Err(format!(
                "primitive restart is unsupported for guest topology {:#x}",
                draw.primitive_type
            ));
        }
    };

    static LOGGED_RESTARTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let sequence = LOGGED_RESTARTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if sequence <= 16 || sequence.is_power_of_two() {
        eprintln!(
            "[REMu][METAL][PRIMITIVE-RESTART] draw={} topology={:#x} restart={restart:#x} raw={} prepared={}",
            draw.sequence,
            draw.primitive_type,
            indices.len(),
            prepared.len(),
        );
    }
    Ok((prepared.into(), primitive))
}

fn primitive_restart_index(draw: &IndexedDraw) -> Option<u32> {
    const VGT_MULTI_PRIM_IB_RESET_EN: u32 = 0x2a5;
    const VGT_MULTI_PRIM_IB_RESET_INDX: u32 = 0x103;

    let enabled = register_value(&draw.context_registers, VGT_MULTI_PRIM_IB_RESET_EN)
        .is_some_and(|value| value & 1 != 0);
    if !enabled {
        return None;
    }
    let programmed =
        register_value(&draw.context_registers, VGT_MULTI_PRIM_IB_RESET_INDX).unwrap_or(u32::MAX);
    match draw.index_size {
        0 => Some(programmed & u32::from(u16::MAX)),
        1 => Some(programmed),
        2 => Some(programmed & u32::from(u8::MAX)),
        _ => None,
    }
}

pub(crate) fn metal_primitive_topology_class(
    guest: u32,
) -> Result<MTLPrimitiveTopologyClass, String> {
    match guest {
        1 => Ok(MTLPrimitiveTopologyClass::Point),
        2 | 3 => Ok(MTLPrimitiveTopologyClass::Line),
        4 | 5 | 6 | 7 | 0x11 => Ok(MTLPrimitiveTopologyClass::Triangle),
        _ => Err(format!("unknown guest primitive topology {guest:#x}")),
    }
}

pub(crate) const fn metal_nonindexed_vertex_count(
    guest_primitive_type: u32,
    guest_count: u32,
) -> u32 {
    if matches!(guest_primitive_type, 7 | 0x11) && guest_count == 3 {
        4
    } else {
        guest_count
    }
}

pub(crate) fn decode_raster_state(draw: &IndexedDraw) -> Result<DecodedRasterState, String> {
    // AGC / SDK default is 0x240 (CCW front face). The old `1 << 2` (CW) fallback inverted
    // winding for every draw whose FilteredStateBuffer suppressed a default write.
    let control = register_value(&draw.context_registers, PA_SU_SC_MODE_CNTL)
        .unwrap_or(DEFAULT_PA_SU_SC_MODE_CNTL);
    let cull_front = control & 1 != 0;
    let cull_back = control & 2 != 0;
    let cull_mode = match (cull_front, cull_back) {
        (false, false) => MTLCullMode::None,
        (true, false) => MTLCullMode::Front,
        (false, true) => MTLCullMode::Back,
        (true, true) => {
            return Err(
                "simultaneous front- and back-face culling has no exact Metal raster state"
                    .to_owned(),
            );
        }
    };
    let front_winding = if control & (1 << 2) != 0 {
        MTLWinding::Clockwise
    } else {
        MTLWinding::CounterClockwise
    };

    let polygon_mode = (control >> 3) & 0x3;
    let fill_mode = match polygon_mode {
        0 => MTLTriangleFillMode::Fill,
        1 => {
            let front_type = (control >> 5) & 0x7;
            let back_type = (control >> 8) & 0x7;
            let visible_type = match cull_mode {
                MTLCullMode::Front => back_type,
                MTLCullMode::Back => front_type,
                MTLCullMode::None if front_type == back_type => front_type,
                MTLCullMode::None => {
                    return Err(format!(
                        "different front/back polygon modes ({front_type}/{back_type}) have no exact Metal raster state"
                    ));
                }
            };
            match visible_type {
                1 => MTLTriangleFillMode::Lines,
                2 => MTLTriangleFillMode::Fill,
                0 => {
                    return Err(
                        "point polygon mode has no exact Metal triangle raster state".to_owned(),
                    );
                }
                value => return Err(format!("reserved GFX10 polygon primitive type {value}")),
            }
        }
        value => return Err(format!("reserved GFX10 polygon mode {value}")),
    };

    let front_scale = register_value(
        &draw.context_registers,
        super::PA_SU_POLY_OFFSET_FRONT_SCALE,
    )
    .map(f32::from_bits)
    .filter(|value| value.is_finite())
    .unwrap_or(0.0);
    let front_offset = register_value(
        &draw.context_registers,
        super::PA_SU_POLY_OFFSET_FRONT_OFFSET,
    )
    .map(f32::from_bits)
    .filter(|value| value.is_finite())
    .unwrap_or(0.0);
    let back_scale = register_value(&draw.context_registers, super::PA_SU_POLY_OFFSET_BACK_SCALE)
        .map(f32::from_bits)
        .filter(|value| value.is_finite())
        .unwrap_or(front_scale);
    let back_offset = register_value(
        &draw.context_registers,
        super::PA_SU_POLY_OFFSET_BACK_OFFSET,
    )
    .map(f32::from_bits)
    .filter(|value| value.is_finite())
    .unwrap_or(front_offset);
    let clamp = register_value(&draw.context_registers, super::PA_SU_POLY_OFFSET_CLAMP)
        .map(f32::from_bits)
        .filter(|value| value.is_finite())
        .unwrap_or(0.0);
    if front_scale != back_scale || front_offset != back_offset {
        return Err(format!(
            "different front/back polygon offsets have no exact Metal depth-bias state \
             (front_scale={front_scale}, front_offset={front_offset}, \
             back_scale={back_scale}, back_offset={back_offset})"
        ));
    }
    // SDK documents scale in 1/16th units (pass 16.0 for a scale of 1.0).
    let depth_bias = DecodedDepthBias {
        slope_scale: front_scale / 16.0,
        constant: front_offset,
        clamp,
    };

    Ok(DecodedRasterState {
        cull_mode,
        front_winding,
        fill_mode,
        depth_bias,
    })
}

pub(crate) fn graphics_pipeline_key(draw: &IndexedDraw) -> Result<GraphicsPipelineKey, String> {
    let targets = decode_render_targets(draw)?;
    graphics_pipeline_key_from_targets(draw, &targets)
}

pub(crate) fn pixel_depth_export_enabled(
    context_registers: &[remu_gpu::GpuRegister],
    has_depth_target: bool,
) -> bool {
    has_depth_target
        && register_value(context_registers, DB_SHADER_CONTROL)
            .is_some_and(|control| control & 1 != 0)
}

fn graphics_diagnostic_pipeline_state(
    draw: &IndexedDraw,
    targets: &[DecodedRenderTarget],
    pixel_evaluation: &ShaderEvaluation,
    pixel_program: &remu_shader::Program,
    sample_count: u32,
) -> GraphicsDiagnosticPipelineState {
    if diagnostic_dump_gbuffer() && draw.pixel_shader_address >= 0x8030_0000_00 {
        eprintln!(
            "[REMu][METAL][GBUFFER-CANDIDATE] draw={} es={:#x} ps={:#x} count={} indexed={} col_format={:#010x} targets={:?}",
            draw.sequence,
            draw.export_shader_address,
            draw.pixel_shader_address,
            draw.index_count,
            draw.indexed,
            register_value(&draw.context_registers, SPI_SHADER_COL_FORMAT).unwrap_or_default(),
            targets
                .iter()
                .map(|target| (
                    target.guest_slot,
                    target.texture.address,
                    target.texture.width,
                    target.texture.height,
                    target.texture.pixel_format,
                ))
                .collect::<Vec<_>>(),
        );
    }
    // Shader instrumentation must be compiled before execution reaches the semantic MRT gate:
    // pipeline workers can evaluate draws out of order and the ordinary shader pair may already
    // be cached by then. Execution metadata keeps the capture branch disabled until the gate.
    let selected_target = draw.sequence >= diagnostic_dump_target_minimum_draw()
        && targets.iter().any(|target| {
            diagnostic_target_address_matches(
                diagnostic_dump_target_address(),
                target.texture.address,
                target.texture.width,
                target.texture.height,
            ) || diagnostic_dump_target_address_range().is_some_and(|(minimum, maximum)| {
                (minimum..=maximum).contains(&target.texture.address)
            })
        });
    let explicit_target_filter = diagnostic_dump_target_address().is_some()
        || diagnostic_dump_target_address_range().is_some();
    let selected_mrt = diagnostic_dump_target_mrt_min().is_some_and(|minimum| {
        targets.len() >= minimum
            && targets
                .iter()
                .all(|target| target.texture.width >= 1_280 && target.texture.height >= 720)
    });
    let selected_shader = diagnostic_dump_target_pixel_shader() == Some(draw.pixel_shader_address)
        || diagnostic_dump_target_pixel_instruction_count()
            == Some(pixel_program.instructions.len())
        || (diagnostic_dump_gbuffer()
            && draw.index_count == 6
            && !targets.is_empty()
            && targets
                .iter()
                .all(|target| target.texture.width == 1_920 && target.texture.height == 1_080)
            && (targets.len() >= 4 || draw.pixel_shader_address >= 0x8030_0000_00))
        || (diagnostic_dump_video_target_consumer()
            && video_target_binding_summary(pixel_evaluation).is_some())
        || (diagnostic_dump_video_chain_consumer()
            && diagnostic_video_chain_selected_draw().load(Ordering::Relaxed) == draw.sequence)
        || (diagnostic_dump_shadow_consumer()
            && evaluation_samples_layered_comparison(pixel_evaluation))
        || (diagnostic_dump_glyph_draw() && evaluation_samples_glyph_atlas(pixel_evaluation));
    let explicit_shader_filter = diagnostic_dump_target_pixel_shader().is_some()
        || diagnostic_dump_target_pixel_instruction_count().is_some();
    // Explicit shader and attachment selectors are conjunctive. Treating ADDRESS=auto as an
    // alternative to PS instruments every full-resolution draw after BasePass, allocating and
    // clearing diagnostic buffers across the whole post chain before the one requested composite
    // can retire.
    let selected_draw = diagnostic_filter_selection(
        explicit_target_filter,
        selected_target,
        explicit_shader_filter,
        selected_shader,
    );
    // Pipeline instrumentation may safely be a superset of the ultimately armed capture. In
    // particular, compile an explicitly named PS even when asynchronous pipeline evaluation sees
    // it before the dynamic target selector has stabilized. `execute_draw` keeps the branch off
    // until the semantic gate and all target constraints match.
    let instrumented_draw = selected_draw || selected_mrt || selected_shader;
    if diagnostic_dump_target_pixel_shader() == Some(draw.pixel_shader_address) {
        static LOGGED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        if LOGGED.fetch_add(1, Ordering::Relaxed) < 8 {
            eprintln!(
                "[REMu][METAL][TARGET-PIPELINE-SELECT] draw={} ps={:#x} target_filter={explicit_target_filter}/{selected_target} shader_filter={explicit_shader_filter}/{selected_shader} selected={selected_draw} instrumented={instrumented_draw} targets={:?}",
                draw.sequence,
                draw.pixel_shader_address,
                targets
                    .iter()
                    .map(|target| (
                        target.texture.address,
                        target.texture.width,
                        target.texture.height,
                        target.direct_scanout,
                    ))
                    .collect::<Vec<_>>(),
            );
        }
    }
    let samples_large_video =
        diagnostic_dump_video_draw() && evaluation_samples_large_linear_video(pixel_evaluation);

    GraphicsDiagnosticPipelineState {
        force_pixel_color: sample_count > 1
            && pixel_program.instructions.len() > 64
            && diagnostic_force_msaa_pixel_color(),
        // The semantic video-chain selector is armed only when the first consumer is dequeued.
        // Pipeline lookahead can compile the ordinary pair before then, so make the chain dump
        // self-contained: once selected, its distinct diagnostic key always carries both probes
        // even when the generic capture environment variables were not set early enough.
        capture_pixel_output: (diagnostic_capture_pixel_output()
            || diagnostic_dump_video_chain_consumer())
            && ((sample_count > 1 && pixel_program.instructions.len() == 370)
                || instrumented_draw
                || samples_large_video),
        force_visible_clip_depth: sample_count > 1 && diagnostic_force_visible_clip_depth(),
        force_vertex_pattern: diagnostic_force_msaa_vertex_pattern()
            && (sample_count > 1 || instrumented_draw),
        force_vertex_fetch_id: diagnostic_force_vertex_fetch_id()
            && (sample_count > 1 || instrumented_draw),
        capture_vertex_position: (diagnostic_capture_vertex_position()
            || diagnostic_dump_video_chain_consumer())
            && (sample_count > 1 || instrumented_draw || samples_large_video),
    }
}

const fn diagnostic_filter_selection(
    target_filter_present: bool,
    target_matches: bool,
    shader_filter_present: bool,
    shader_matches: bool,
) -> bool {
    match (target_filter_present, shader_filter_present) {
        (true, true) => target_matches && shader_matches,
        (true, false) => target_matches,
        (false, true) => shader_matches,
        (false, false) => false,
    }
}

pub(crate) fn graphics_pipeline_key_from_targets(
    draw: &IndexedDraw,
    targets: &[DecodedRenderTarget],
) -> Result<GraphicsPipelineKey, String> {
    let depth = decode_depth_target(draw)?;
    if targets.is_empty() && depth.is_none() {
        // Depth decode returns Ok(None) when STENCIL_ENABLE is latched without Z ops so color
        // MRTs can still build a pipeline. A draw with neither color nor depth, but stencil
        // still enabled, is the unsupported stencil-only case.
        let control = register_value(&draw.context_registers, DB_DEPTH_CONTROL).unwrap_or_default();
        if control & 1 != 0 {
            return Err(
                "stencil is enabled without an active depth/stencil surface; \
                 stencil-only targets are not implemented"
                    .to_owned(),
            );
        }
        return Err(
            "draw has neither an enabled color target nor an active depth target".to_owned(),
        );
    }
    let export_evaluation = draw
        .export_shader_evaluation
        .as_ref()
        .ok_or("draw has no export resource evaluation")?;
    let export_program = &draw
        .export_shader
        .as_ref()
        .ok_or("draw has no decoded export shader")?
        .program;
    let (pixel_program, pixel_evaluation, fragmentless) = match (
        draw.pixel_shader.as_ref(),
        draw.pixel_shader_evaluation.as_ref(),
    ) {
        (Some(pixel), Some(evaluation)) => (&pixel.program, evaluation.as_ref(), false),
        (None, None) if targets.is_empty() && depth.is_some() => (
            fragmentless_pixel_program(),
            fragmentless_pixel_evaluation(),
            true,
        ),
        (None, None) => {
            return Err(
                "fragmentless draw requires an active depth target and no color targets".to_owned(),
            );
        }
        (Some(_), None) => return Err("draw has no pixel resource evaluation".to_owned()),
        (None, Some(_)) => {
            return Err("draw has pixel resources but no decoded pixel shader".to_owned());
        }
    };
    let sample_count = targets
        .first()
        .map(|target| target.sample_count)
        .or_else(|| depth.map(|depth| depth.sample_count))
        .unwrap_or(1);
    let render_target_array_length = common_render_target_array_length(
        targets
            .iter()
            .map(|target| target.layer_count)
            .chain(depth.into_iter().map(|depth| depth.layer_count)),
    );
    let vertex_output_control =
        register_value(&draw.context_registers, PA_CL_VS_OUT_CNTL).unwrap_or_default();
    if let Some(target) = targets
        .iter()
        .find(|target| target.sample_count != sample_count)
    {
        return Err(format!(
            "color target {} uses {} samples while the render pass uses {sample_count}",
            target.guest_slot, target.sample_count
        ));
    }
    if let Some(depth) = depth
        && depth.sample_count != sample_count
    {
        return Err(format!(
            "depth target uses {} samples while color targets use {sample_count}",
            depth.sample_count
        ));
    }
    // Pipeline construction runs ahead of Metal execution. Open the semantic diagnostic gate
    // here, in guest draw order, so the first compositor immediately following a full BasePass
    // receives the instrumented key. Waiting until the MRT has executed is too late for a
    // one-shot title composite whose ordinary pipeline may already be queued or cached.
    diagnostic_note_full_resolution_mrt_draw(
        draw.sequence,
        targets.len(),
        !targets.is_empty()
            && targets
                .iter()
                .all(|target| target.texture.width >= 1_280 && target.texture.height >= 720),
    );
    let export_user_data_register = [0x08c, 0x0cc, 0x04c]
        .into_iter()
        .find(|base| {
            draw.shader_registers
                .iter()
                .any(|register| register.offset == base - 1)
        })
        .unwrap_or(0x0cc);
    let ngg_subgroup_limits =
        decode_ngg_subgroup_limits_with_uc(&draw.context_registers, &draw.user_config_registers)?
            .map(|limits| {
                limits.with_fixed_allocation_fallback(ngg_fixed_allocation_fallback_for_program(
                    export_program,
                ))
            });
    let ngg_registers = ngg_subgroup_limits
        .map(|_| {
            decode_ngg_registers_from_state(
                &draw.shader_registers,
                &draw.context_registers,
                export_user_data_register,
            )
        })
        .transpose()?;
    Ok(GraphicsPipelineKey {
        export_shader_address: draw.export_shader_address,
        pixel_shader_address: draw.pixel_shader_address,
        export_program_identity: ProgramIdentity(Arc::clone(export_program)),
        pixel_program_identity: ProgramIdentity(Arc::clone(pixel_program)),
        export_resources: graphics_shader_resource_key(
            export_evaluation,
            program_embeds_global_base(export_program),
        ),
        pixel_resources: graphics_shader_resource_key(
            pixel_evaluation,
            program_embeds_global_base(pixel_program),
        ),
        export_allocated_vector_register_count: graphics_allocated_vector_register_count(
            &draw.shader_registers,
            export_user_data_register,
        ),
        // Fragmentless pixel programs still go through the ordinary Metal lowering path, which
        // rejects a zero VGPR allocation. Keep one four-register block so depth-only prepasses
        // (common on Unreal title/menu frames) can compile even though they never touch VGPRs.
        pixel_allocated_vector_register_count: if fragmentless {
            4
        } else {
            graphics_allocated_vector_register_count(&draw.shader_registers, 0x00c)
        },
        vertex_output_control,
        // AGC seeds `PA_CL_VTE_CNTL=0x43f` (VTX_W0_FMT set). Missing after a sparse test draw
        // must not invent that default — only seeded/live state may force W=1.
        viewport_transform_control: register_value(&draw.context_registers, PA_CL_VTE_CNTL)
            .unwrap_or_default(),
        pixel_input_controls: if fragmentless {
            [0; 32]
        } else {
            pixel_input_controls(&draw.context_registers)
        },
        ngg_subgroup_limits,
        ngg_registers,
        wave_state: decode_graphics_wave_state(&draw.context_registers),
        pixel_depth_export: !fragmentless
            && pixel_depth_export_enabled(&draw.context_registers, depth.is_some()),
        diagnostics: graphics_diagnostic_pipeline_state(
            draw,
            targets,
            pixel_evaluation,
            pixel_program,
            sample_count,
        ),
        targets: {
            // Seeded AGC default for SPI_SHADER_COL_FORMAT is 0 (no exports). Missing after a
            // sparse test draw also resolves to 0 — never invent a non-zero writemask format.
            let shader_color_formats =
                register_value(&draw.context_registers, SPI_SHADER_COL_FORMAT).unwrap_or(0);
            let mut targets = targets
                .iter()
                .map(|target| {
                    let export_nibble =
                        (shader_color_formats >> (target.guest_slot.saturating_mul(4))) & 0xf;
                    let export_format = PixelExportFormat::from_register_nibble(export_nibble)
                        .ok_or_else(|| {
                            format!(
                                "color target {} has invalid SPI_SHADER_COL_FORMAT nibble {export_nibble:#x} in {shader_color_formats:#010x}",
                                target.guest_slot
                            )
                        })?;
                    let control = register_value(
                        &draw.context_registers,
                        CB_BLEND0_CONTROL + target.guest_slot,
                    )
                    .unwrap_or(DEFAULT_CB_BLEND_CONTROL);
                    Ok(RenderTargetPipelineState {
                        guest_slot: target.guest_slot,
                        pixel_format: target.texture.pixel_format,
                        output_kind: target.output_kind,
                        export_format,
                        // A ZERO export format suppresses the corresponding CB write even if the
                        // shader contains an EXP instruction and CB_TARGET_MASK remains enabled.
                        write_mask: if export_format == PixelExportFormat::Zero {
                            0
                        } else {
                            target.write_mask
                        },
                        blend: if target.blend_bypass {
                            disabled_metal_blend_attachment()
                        } else {
                            decode_metal_blend_attachment(control).map_err(|error| {
                                format!(
                                    "color target {} has invalid CB_BLEND_CONTROL={control:#010x}: {error}",
                                    target.guest_slot
                                )
                            })?
                        },
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let live_pixel_scalars = Some(std::array::from_fn(|index| {
                register_value(
                    &draw.shader_registers,
                    0x00c + u32::try_from(index).expect("scalar index fits"),
                )
                .unwrap_or_default()
            }));
            apply_transparent_premultiplied_fill_clear_with_color(
                &mut targets,
                export_evaluation,
                pixel_evaluation,
                live_pixel_scalars,
            );
            targets
        },
        depth: depth.map(|depth| DepthTargetPipelineState {
            pixel_format: depth.texture.pixel_format,
        }),
        sample_count,
        render_target_array_length,
        uses_mesh_stage: {
            // Memoryless NGG covers (Poppy/UE fullscreen blits) still shard under the meshlet ABI
            // for a few topologies. Only those position-from-VID stages may use the synthesized
            // covering VS — NGG sprite quads that fetch vertex buffers (Dreaming Sarah) must keep
            // the mesh path so per-draw bounds survive.
            if ngg_fullscreen_cover_draw(export_program, draw)
                || procedural_rectangle_stage_for_draw(export_program, draw)
            {
                false
            } else {
                ngg_mesh_stage_for_draw(export_program, draw)?
            }
        },
        uses_procedural_rectangle_stage: {
            let procedural = ngg_fullscreen_cover_draw(export_program, draw)
                || procedural_rectangle_stage_for_draw(export_program, draw);
            if !procedural
                && !draw.indexed
                && draw.indirect_arguments_address.is_none()
                && draw.primitive_type == 6
                && draw.index_count == 4
                && ngg_mesh_export_stage(export_program)
                && !ngg_memoryless_cover_export_stage(export_program)
            {
                static LOGGED_RESOURCE_RECTANGLES: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                let sequence = LOGGED_RESOURCE_RECTANGLES
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    + 1;
                if sequence <= 16 || sequence.is_power_of_two() {
                    eprintln!(
                        "[REMu][METAL][RESOURCE-RECT] sequence={sequence} draw={} es={:#x} guest_count={} guest_primitive={:#x} translated=true",
                        draw.sequence,
                        draw.export_shader_address,
                        draw.index_count,
                        draw.primitive_type,
                    );
                }
            }
            if !procedural
                && !draw.indexed
                && draw.indirect_arguments_address.is_none()
                && draw.index_count == 3
                && matches!(draw.primitive_type, 7 | 0x11)
            {
                static LOGGED: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                    let opcodes = export_program
                        .instructions
                        .iter()
                        .map(|instruction| {
                            instruction
                                .opcode
                                .known_name()
                                .unwrap_or("unknown")
                                .to_owned()
                        })
                        .collect::<Vec<_>>()
                        .join(",");
                    eprintln!(
                        "[REMu][METAL][RECT-MISS] draw={} es={:#x} instructions={} ops=[{opcodes}]",
                        draw.sequence,
                        draw.export_shader_address,
                        export_program.instructions.len(),
                    );
                }
            }
            procedural
        },
        primitive_topology_class: metal_primitive_topology_class(draw.primitive_type)?,
    })
}

pub(crate) fn graphics_shader_resource_key(
    evaluation: &ShaderEvaluation,
    _embeds_global_base: bool,
) -> GraphicsShaderResourceKey {
    let image_slots = evaluation.image_binding_slots();
    GraphicsShaderResourceKey {
        initial_scalar_count: evaluation.initial_scalar_registers.len(),
        // Every lowering embeds the byte extent. Only GLOBAL instructions still embed a full
        // guest base; descriptor-backed buffer instructions address the current dword-aligned
        // Metal binding through the descriptor's dynamic low-bit bias.
        global_bindings: evaluation
            .global_memory_bindings
            .iter()
            .map(|binding| (binding.instruction_pcs.to_vec(), 0, 0, binding.writable))
            .collect(),
        // Image addresses, component selection, exact pixel formats, and sampler state are bound
        // dynamically. Only the resource order, MSL component class, and access/interface shape
        // alter the generated Metal function signature.
        image_bindings: evaluation
            .image_bindings
            .iter()
            .enumerate()
            .map(|(index, binding)| {
                (
                    binding.pc,
                    image_component_class(binding),
                    is_storage_image_opcode(binding.opcode),
                    image_is_comparison(binding.opcode),
                    binding.control.dimension,
                    binding
                        .resource_descriptor
                        .get(3)
                        .map_or(0, |word| (word >> 28) & 0xf),
                    image_slots.texture_slots[index],
                    image_slots.sampler_slots[index],
                )
            })
            .collect(),
        vertex_inputs: evaluation
            .vertex_inputs
            .iter()
            .map(|input| {
                (
                    input.pc,
                    input.location,
                    input.component_count,
                    input.data_format,
                    input.number_format,
                    input.stride,
                    input.offset_bytes,
                )
            })
            .collect(),
    }
}

/// Disabled Metal blend attachment used for both guest blend-off and Chowdren fill clears.
pub(crate) const fn disabled_metal_blend_attachment() -> MetalBlendAttachmentState {
    MetalBlendAttachmentState {
        enabled: false,
        source_rgb: MTLBlendFactor::One,
        destination_rgb: MTLBlendFactor::Zero,
        rgb_operation: MTLBlendOperation::Add,
        source_alpha: MTLBlendFactor::One,
        destination_alpha: MTLBlendFactor::Zero,
        alpha_operation: MTLBlendOperation::Add,
    }
}

fn transparent_premultiplied_fill_clear_enabled() -> bool {
    std::env::var_os("REMU_DISABLE_TRANSPARENT_FILL_CLEAR").is_none_or(|value| value != "1")
}

/// Chowdren resets effect layers with an untextured transparent-black fill using premultiplied
/// blending (`One` / `OneMinusSrcAlpha`). That draw is otherwise a no-op and fog/vignette layers
/// accumulate across frames. Treat precisely that shape as an overwrite.
///
/// Do not widen this to straight `SrcAlpha` / `OneMinusSrcAlpha`: those transparent-black draws
/// are often intentional no-ops, and forcing an overwrite wipes scene layers (e.g. the player).
///
/// Opt out with `REMU_DISABLE_TRANSPARENT_FILL_CLEAR=1`.
#[cfg(test)]
pub(crate) fn apply_transparent_premultiplied_fill_clear(
    targets: &mut [RenderTargetPipelineState],
    export_evaluation: &ShaderEvaluation,
    pixel_evaluation: &ShaderEvaluation,
) {
    apply_transparent_premultiplied_fill_clear_with_color(
        targets,
        export_evaluation,
        pixel_evaluation,
        None,
    );
}

/// Same as [`apply_transparent_premultiplied_fill_clear`], but prefers live pixel user-data words
/// from the draw's SH registers when provided. Evaluation caches key on user data, so the two
/// sources normally agree; the live words are the ground truth for the color the Metal scalar
/// buffer will actually bind.
pub(crate) fn apply_transparent_premultiplied_fill_clear_with_color(
    targets: &mut [RenderTargetPipelineState],
    export_evaluation: &ShaderEvaluation,
    pixel_evaluation: &ShaderEvaluation,
    live_pixel_scalars: Option<[u32; 4]>,
) {
    if !transparent_premultiplied_fill_clear_enabled()
        || targets.is_empty()
        || !export_evaluation.vertex_inputs.is_empty()
        || !pixel_evaluation.image_bindings.is_empty()
        || !targets
            .iter()
            .all(|target| is_transparent_premultiplied_fill_blend(&target.blend))
    {
        return;
    }
    let fill_color = match live_pixel_scalars {
        Some(scalars) => scalars,
        None => {
            if pixel_evaluation.initial_scalar_registers.len() < 4 {
                return;
            }
            std::array::from_fn(|index| pixel_evaluation.initial_scalar_registers[index])
        }
    };
    if fill_color.iter().any(|value| value & 0x7fff_ffff != 0) {
        return;
    }
    for target in targets.iter_mut() {
        target.blend = disabled_metal_blend_attachment();
    }
}

pub(crate) fn is_transparent_premultiplied_fill_blend(blend: &MetalBlendAttachmentState) -> bool {
    blend.enabled
        && blend.source_rgb == MTLBlendFactor::One
        && blend.destination_rgb == MTLBlendFactor::OneMinusSourceAlpha
        && blend.rgb_operation == MTLBlendOperation::Add
}

pub(crate) fn decode_metal_blend_attachment(
    control: u32,
) -> Result<MetalBlendAttachmentState, String> {
    let disabled = disabled_metal_blend_attachment();
    if control & (1 << 30) == 0 {
        return Ok(disabled);
    }

    let source_rgb = metal_blend_factor(control & 0x1f)?;
    let rgb_operation = metal_blend_operation((control >> 5) & 0x7)?;
    let destination_rgb = metal_blend_factor((control >> 8) & 0x1f)?;
    let separate_alpha = control & (1 << 29) != 0;
    let (source_alpha, alpha_operation, destination_alpha) = if separate_alpha {
        (
            metal_blend_factor((control >> 16) & 0x1f)?,
            metal_blend_operation((control >> 21) & 0x7)?,
            metal_blend_factor((control >> 24) & 0x1f)?,
        )
    } else {
        (source_rgb, rgb_operation, destination_rgb)
    };
    Ok(MetalBlendAttachmentState {
        enabled: true,
        source_rgb,
        destination_rgb,
        rgb_operation,
        source_alpha,
        destination_alpha,
        alpha_operation,
    })
}

pub(crate) fn metal_blend_factor(value: u32) -> Result<MTLBlendFactor, String> {
    match value {
        0 => Ok(MTLBlendFactor::Zero),
        1 => Ok(MTLBlendFactor::One),
        2 => Ok(MTLBlendFactor::SourceColor),
        3 => Ok(MTLBlendFactor::OneMinusSourceColor),
        4 => Ok(MTLBlendFactor::SourceAlpha),
        5 => Ok(MTLBlendFactor::OneMinusSourceAlpha),
        6 => Ok(MTLBlendFactor::DestinationAlpha),
        7 => Ok(MTLBlendFactor::OneMinusDestinationAlpha),
        8 => Ok(MTLBlendFactor::DestinationColor),
        9 => Ok(MTLBlendFactor::OneMinusDestinationColor),
        10 => Ok(MTLBlendFactor::SourceAlphaSaturated),
        13 => Ok(MTLBlendFactor::BlendColor),
        14 => Ok(MTLBlendFactor::OneMinusBlendColor),
        19 => Ok(MTLBlendFactor::BlendAlpha),
        20 => Ok(MTLBlendFactor::OneMinusBlendAlpha),
        15..=18 => Err(format!(
            "dual-source blend factor {value} requires a second pixel-shader color output"
        )),
        _ => Err(format!("unsupported blend factor {value}")),
    }
}

pub(crate) fn metal_blend_operation(value: u32) -> Result<MTLBlendOperation, String> {
    match value {
        0 => Ok(MTLBlendOperation::Add),
        1 => Ok(MTLBlendOperation::Subtract),
        2 => Ok(MTLBlendOperation::Min),
        3 => Ok(MTLBlendOperation::Max),
        4 => Ok(MTLBlendOperation::ReverseSubtract),
        _ => Err(format!("unsupported blend operation {value}")),
    }
}

pub(crate) fn metal_color_write_mask(guest_mask: u32) -> MTLColorWriteMask {
    // GFX10 CB_TARGET_MASK is R=1,G=2,B=4,A=8; MTLColorWriteMask uses the reverse bits.
    let metal_mask = u64::from(guest_mask & 1 != 0) << 3
        | u64::from(guest_mask & 2 != 0) << 2
        | u64::from(guest_mask & 4 != 0) << 1
        | u64::from(guest_mask & 8 != 0);
    MTLColorWriteMask::from_bits_retain(metal_mask)
}

#[cfg(test)]
mod tests {
    use super::{DB_SHADER_CONTROL, diagnostic_filter_selection, pixel_depth_export_enabled};
    use remu_gpu::GpuRegister;

    #[test]
    fn gates_mrtz_depth_output_on_db_shader_control_and_an_active_depth_target() {
        let registers = [GpuRegister {
            offset: DB_SHADER_CONTROL,
            value: 1,
        }];

        assert!(pixel_depth_export_enabled(&registers, true));
        assert!(!pixel_depth_export_enabled(&registers, false));
        assert!(!pixel_depth_export_enabled(
            &[GpuRegister {
                offset: DB_SHADER_CONTROL,
                value: 0,
            }],
            true,
        ));
    }

    #[test]
    fn combines_explicit_diagnostic_shader_and_target_filters_conjunctively() {
        assert!(diagnostic_filter_selection(true, true, true, true));
        assert!(!diagnostic_filter_selection(true, true, true, false));
        assert!(!diagnostic_filter_selection(true, false, true, true));
        assert!(diagnostic_filter_selection(true, true, false, false));
        assert!(diagnostic_filter_selection(false, false, true, true));
        assert!(!diagnostic_filter_selection(false, false, false, false));
    }
}
