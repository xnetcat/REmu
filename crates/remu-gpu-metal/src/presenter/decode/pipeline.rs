// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CB_BLEND0_CONTROL, DB_SHADER_CONTROL, DecodedRasterState, DecodedRenderTarget,
    DepthTargetPipelineState,
    GraphicsPipelineKey, GraphicsShaderResourceKey, MetalBlendAttachmentState, PA_CL_VS_OUT_CNTL,
    PA_SU_SC_MODE_CNTL, ProgramIdentity, RenderTargetPipelineState,
    common_render_target_array_length, decode_depth_target, decode_render_targets,
    fragmentless_pixel_evaluation, fragmentless_pixel_program,
    graphics_allocated_vector_register_count, image_component_class, is_storage_image_opcode,
    ngg_mesh_stage_for_draw, pixel_input_controls, procedural_rectangle_stage_for_draw,
    program_embeds_global_base, register_value, video_target_binding_summary,
};
use metal::{
    MTLBlendFactor, MTLBlendOperation, MTLColorWriteMask, MTLCullMode, MTLPrimitiveTopologyClass,
    MTLPrimitiveType, MTLTriangleFillMode, MTLWinding,
};
use remu_gpu::IndexedDraw;
use remu_shader::{ShaderEvaluation, image_is_comparison};
use std::collections::HashMap;
use std::sync::Arc;

pub(crate) const NGG_MESHLET_WORDS_PER_GROUP: usize = 130;
const NGG_MESHLET_MAX_VERTICES: usize = 64;
const NGG_MESHLET_MAX_PRIMITIVES: usize = 64;

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

/// Prepares the source vertex IDs for a non-indexed NGG triangle-strip draw.
///
/// `index_offset` is the guest draw's first vertex. `base_vertex` is an indexed-draw offset and
/// does not participate in non-indexed auto-generated vertex IDs.
pub(crate) fn prepare_nonindexed_ngg_indices(draw: &IndexedDraw) -> Result<Arc<[u32]>, String> {
    if draw.indexed || draw.primitive_type != 6 {
        return Err("NGG source is not a non-indexed triangle strip".to_owned());
    }
    expand_nonindexed_triangle_strip(draw.index_offset, draw.index_count)
}

/// Returns the signed vertex-ID adjustment consumed by the native mesh ABI.
pub(crate) const fn ngg_mesh_base_vertex(draw: &IndexedDraw) -> i32 {
    if draw.indexed { draw.base_vertex } else { 0 }
}

/// Builds the GFX10 NGG workgroups consumed by a merged export/geometry shader.
///
/// The native ABI supplies up to 64 unique vertex IDs and 64 packed triangle records to each
/// wave. Treating every three source indices as three new vertices restricts a wave to only 21
/// triangles, repeats vertex work, and can expand a large Unreal draw into thousands of Metal
/// mesh threadgroups. Preserve index reuse by greedily packing consecutive triangles until
/// either architectural limit is reached.
pub(crate) fn prepare_ngg_meshlets(indices: &[u32]) -> PreparedNggMeshlets {
    let primitive_count = indices.len() / 3;
    let mut words = Vec::new();
    let mut vertices = Vec::<u32>::with_capacity(NGG_MESHLET_MAX_VERTICES);
    let mut vertex_slots = HashMap::<u32, u32>::with_capacity(NGG_MESHLET_MAX_VERTICES);
    let mut primitives = Vec::<u32>::with_capacity(NGG_MESHLET_MAX_PRIMITIVES);
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
        words.push(vertex_count | (primitive_count << 8));
        words.push(u32::try_from(first_primitive).expect("NGG primitive ID fits u32"));
        words.extend(vertices.iter().copied());
        words.resize(words.len() + NGG_MESHLET_MAX_VERTICES - vertices.len(), 0);
        words.extend(primitives.iter().copied());
        words.resize(
            words.len() + NGG_MESHLET_MAX_PRIMITIVES - primitives.len(),
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
            && (primitives.len() == NGG_MESHLET_MAX_PRIMITIVES
                || vertices.len() + additional_vertices > NGG_MESHLET_MAX_VERTICES)
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
        .chunks_exact(NGG_MESHLET_WORDS_PER_GROUP)
        .map(|group| usize::try_from(group[0] & 0xff).expect("vertex count fits usize"))
        .sum();
    let group_count = u64::try_from(words.len() / NGG_MESHLET_WORDS_PER_GROUP)
        .expect("NGG meshlet group count fits u64");
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
                for (triangle, vertices) in segment.windows(3).enumerate() {
                    if triangle & 1 == 0 {
                        prepared.extend_from_slice(vertices);
                    } else {
                        prepared.extend_from_slice(&[vertices[1], vertices[0], vertices[2]]);
                    }
                }
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
    // GFX10's reset value has FACE set and all culling/polygon-mode bits clear.
    let control = register_value(&draw.context_registers, PA_SU_SC_MODE_CNTL).unwrap_or(1 << 2);
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

    Ok(DecodedRasterState {
        cull_mode,
        front_winding,
        fill_mode,
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

pub(crate) fn graphics_pipeline_key_from_targets(
    draw: &IndexedDraw,
    targets: &[DecodedRenderTarget],
) -> Result<GraphicsPipelineKey, String> {
    let depth = decode_depth_target(draw)?;
    if targets.is_empty() && depth.is_none() {
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
    let export_user_data_register = [0x08c, 0x0cc, 0x04c]
        .into_iter()
        .find(|base| {
            draw.shader_registers
                .iter()
                .any(|register| register.offset == base - 1)
        })
        .unwrap_or(0x0cc);
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
        pixel_allocated_vector_register_count: if fragmentless {
            0
        } else {
            graphics_allocated_vector_register_count(&draw.shader_registers, 0x00c)
        },
        vertex_output_control: register_value(&draw.context_registers, PA_CL_VS_OUT_CNTL)
            .unwrap_or_default(),
        pixel_input_controls: if fragmentless {
            [0; 32]
        } else {
            pixel_input_controls(&draw.context_registers)
        },
        pixel_depth_export: !fragmentless
            && pixel_depth_export_enabled(&draw.context_registers, depth.is_some()),
        targets: targets
            .iter()
            .map(|target| {
                let control = register_value(
                    &draw.context_registers,
                    CB_BLEND0_CONTROL + target.guest_slot,
                )
                .unwrap_or(0x2001_0001);
                Ok(RenderTargetPipelineState {
                    guest_slot: target.guest_slot,
                    pixel_format: target.texture.pixel_format,
                    output_kind: target.output_kind,
                    write_mask: target.write_mask,
                    blend: decode_metal_blend_attachment(control).map_err(|error| {
                        format!(
                            "color target {} has invalid CB_BLEND_CONTROL={control:#010x}: {error}",
                            target.guest_slot
                        )
                    })?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
        depth: depth.map(|depth| DepthTargetPipelineState {
            pixel_format: depth.texture.pixel_format,
        }),
        sample_count,
        render_target_array_length,
        uses_mesh_stage: ngg_mesh_stage_for_draw(export_program, draw)?,
        uses_procedural_rectangle_stage: procedural_rectangle_stage_for_draw(export_program, draw),
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

pub(crate) fn decode_metal_blend_attachment(
    control: u32,
) -> Result<MetalBlendAttachmentState, String> {
    let disabled = MetalBlendAttachmentState {
        enabled: false,
        source_rgb: MTLBlendFactor::One,
        destination_rgb: MTLBlendFactor::Zero,
        rgb_operation: MTLBlendOperation::Add,
        source_alpha: MTLBlendFactor::One,
        destination_alpha: MTLBlendFactor::Zero,
        alpha_operation: MTLBlendOperation::Add,
    };
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
    use super::{DB_SHADER_CONTROL, pixel_depth_export_enabled};
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
}
