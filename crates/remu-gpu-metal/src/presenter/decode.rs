// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    DecodedDepthTarget, DecodedImageDescriptor, DecodedRasterState, DecodedRenderTarget,
    DepthTargetPipelineState, DisplayConfiguration, GraphicsPipelineKey, GraphicsShaderResourceKey,
    GraphicsDiagnosticPipelineState, GuestScissor, GuestTextureDescriptor,
    MetalBlendAttachmentState, ProgramIdentity, RenderTargetFormat, RenderTargetPipelineState,
    decode_sampled_texture_format, diagnostic_capture_pixel_output,
    diagnostic_capture_vertex_position, diagnostic_dump_shadow_consumer,
    diagnostic_dump_target_pixel_instruction_count, diagnostic_dump_target_pixel_shader,
    diagnostic_dump_video_chain_consumer, diagnostic_dump_video_draw,
    diagnostic_dump_video_target_consumer, diagnostic_force_msaa_pixel_color,
    diagnostic_force_msaa_vertex_pattern, diagnostic_force_vertex_fetch_id,
    diagnostic_force_visible_clip_depth, diagnostic_video_chain_selected_draw,
    effective_linear_pitch, evaluation_samples_large_linear_video,
    evaluation_samples_layered_comparison, graphics_allocated_vector_register_count,
    guest_texture_byte_count, half_float_to_f32, is_storage_image_opcode, ngg_mesh_draw_supported,
    ngg_mesh_export_stage, program_embeds_global_base, tiled_block_dimensions,
    trace_image_descriptors, trace_target_sampling, unsigned_float_to_f32,
    video_target_binding_summary,
};
use metal::{MTLScissorRect, MTLViewport};
use remu_gpu::{GpuRegister, IndexedDraw};
use std::sync::Arc;
mod image;
mod pipeline;
mod scanout;
mod targets;
pub(crate) use self::image::{
    decode_image_descriptor, fallback_image_descriptor, image_component_class,
    sampled_alias_descriptor,
};
pub(crate) use self::pipeline::{
    ColorOperationMode, color_operation_is_raster_noop, decode_color_operation_mode,
    decode_metal_blend_attachment, decode_raster_state, graphics_pipeline_key,
    metal_color_write_mask, metal_nonindexed_vertex_count, metal_primitive_type,
    prepare_indexed_primitive, prepare_ngg_meshlets, prepare_nonindexed_ngg_indices,
};
#[cfg(test)]
pub(crate) use self::pipeline::{
    NGG_MESHLET_WORDS_PER_GROUP, graphics_shader_resource_key, metal_primitive_topology_class,
};
#[cfg(test)]
pub(crate) use self::scanout::{
    VIDEO_OUT_DISPLAY_TILE_MODE, VIDEO_OUT_TILING_MODE_LINEAR, VIDEO_OUT_TILING_MODE_TILE,
    raw_video_out_layout_is_linear,
};
pub(crate) use self::scanout::{decode_scanout_target, raw_video_out_texture_descriptor};
pub(crate) use self::targets::{
    common_render_target_array_length, decode_color_eliminate_fast_clear_target,
    decode_color_resolve_targets, decode_depth_target, decode_render_target_format,
    decode_render_target_state, decode_render_targets, initial_depth_clear_for_target,
    initial_render_target_clear_color, render_target_cmask_address,
    render_target_fast_clear_color,
};
#[cfg(test)]
pub(crate) use self::targets::{
    decode_fast_clear_color, decode_render_targets_from_registers, initial_depth_clear_value,
};

pub(super) const CB_TARGET_MASK: u32 = 0x08e;

pub(super) const CB_BLEND_RED: u32 = 0x105;

pub(super) const CB_BLEND_GREEN: u32 = 0x106;

pub(super) const CB_BLEND_BLUE: u32 = 0x107;

pub(super) const CB_BLEND_ALPHA: u32 = 0x108;

pub(super) const SPI_PS_INPUT_CNTL_0: u32 = 0x191;

pub(super) const CB_BLEND0_CONTROL: u32 = 0x1e0;

pub(super) const CB_COLOR_CONTROL: u32 = 0x202;

pub(super) const DEFAULT_CB_COLOR_CONTROL: u32 = 0x00cc_0010;

pub(super) const DB_SHADER_CONTROL: u32 = 0x203;

pub(super) const PA_SU_SC_MODE_CNTL: u32 = 0x205;

pub(super) const PA_CL_VS_OUT_CNTL: u32 = 0x207;

pub(super) const CB_COLOR0_BASE: u32 = 0x318;

pub(super) const CB_COLOR_REGISTER_STRIDE: u32 = 15;

pub(super) const CB_COLOR0_VIEW: u32 = 0x31b;

pub(super) const CB_COLOR0_INFO: u32 = 0x31c;

pub(super) const CB_COLOR0_ATTRIB: u32 = 0x31d;

pub(super) const CB_COLOR0_CMASK: u32 = 0x31f;

pub(super) const CB_COLOR0_FMASK: u32 = 0x321;

pub(super) const CB_COLOR0_CLEAR_WORD0: u32 = 0x323;

pub(super) const CB_COLOR0_CLEAR_WORD1: u32 = 0x324;

pub(super) const CB_COLOR0_BASE_EXT: u32 = 0x390;

pub(super) const CB_COLOR0_FMASK_BASE_EXT: u32 = 0x3a0;

pub(super) const CB_COLOR0_ATTRIB2: u32 = 0x3b0;

pub(super) const CB_COLOR0_ATTRIB3: u32 = 0x3b8;

pub(super) const PA_SC_AA_MASK_X0Y0_X1Y0: u32 = 0x30e;

pub(super) const PA_SC_AA_MASK_X0Y1_X1Y1: u32 = 0x30f;

pub(super) const COLOR_TARGET_COUNT: u32 = 8;

pub(super) const DB_RENDER_CONTROL: u32 = 0x000;

pub(super) const DB_DEPTH_VIEW: u32 = 0x002;

pub(super) const DB_HTILE_DATA_BASE: u32 = 0x005;

pub(super) const DB_DEPTH_SIZE_XY: u32 = 0x007;

pub(super) const DB_DEPTH_CLEAR: u32 = 0x00b;

pub(super) const DB_Z_INFO: u32 = 0x010;

pub(super) const DB_Z_READ_BASE: u32 = 0x012;

pub(super) const DB_Z_WRITE_BASE: u32 = 0x014;

pub(super) const DB_DEPTH_SIZE: u32 = 0x016;

pub(super) const DB_DEPTH_SLICE: u32 = 0x017;

pub(super) const DB_Z_READ_BASE_HI: u32 = 0x01a;

pub(super) const DB_Z_WRITE_BASE_HI: u32 = 0x01c;

pub(super) const DB_DEPTH_CONTROL: u32 = 0x200;

pub(super) fn decode_scissor(
    draw: &IndexedDraw,
    target_width: u32,
    target_height: u32,
) -> Result<GuestScissor, String> {
    let mut bounds = [0_i64, 0, i64::from(target_width), i64::from(target_height)];
    let window_scissor_tl = register_value(&draw.context_registers, 0x081);
    let enable_window_offset = window_scissor_tl.is_none_or(|value| value & 0x8000_0000 == 0);
    let (window_offset_x, window_offset_y) = if enable_window_offset {
        register_value(&draw.context_registers, 0x080).map_or((0_i64, 0_i64), |value| {
            (
                i64::from(i16::from_ne_bytes(
                    u16::try_from(value & 0xffff)
                        .expect("masked window X offset fits u16")
                        .to_ne_bytes(),
                )),
                i64::from(i16::from_ne_bytes(
                    u16::try_from(value >> 16)
                        .expect("window Y offset fits u16")
                        .to_ne_bytes(),
                )),
            )
        })
    } else {
        (0, 0)
    };
    intersect_scissor_pair(draw, 0x00c, 0x00d, 0, 0, true, &mut bounds);
    intersect_scissor_pair(
        draw,
        0x081,
        0x082,
        window_offset_x,
        window_offset_y,
        false,
        &mut bounds,
    );
    intersect_scissor_pair(
        draw,
        0x090,
        0x091,
        window_offset_x,
        window_offset_y,
        false,
        &mut bounds,
    );
    if register_value(&draw.context_registers, 0x292).is_none_or(|mode| mode & 0x2 != 0) {
        intersect_scissor_pair(draw, 0x094, 0x095, 0, 0, false, &mut bounds);
    }
    bounds[0] = bounds[0].clamp(0, i64::from(target_width));
    bounds[1] = bounds[1].clamp(0, i64::from(target_height));
    bounds[2] = bounds[2].clamp(bounds[0], i64::from(target_width));
    bounds[3] = bounds[3].clamp(bounds[1], i64::from(target_height));
    if bounds == [0, 0, i64::from(target_width), i64::from(target_height)] {
        return Ok(GuestScissor::Full);
    }
    let width = bounds[2] - bounds[0];
    let height = bounds[3] - bounds[1];
    if width == 0 || height == 0 {
        return Ok(GuestScissor::Empty);
    }
    Ok(GuestScissor::Rect(MTLScissorRect {
        x: u64::try_from(bounds[0]).map_err(|_| "scissor X does not fit Metal")?,
        y: u64::try_from(bounds[1]).map_err(|_| "scissor Y does not fit Metal")?,
        width: u64::try_from(width).map_err(|_| "scissor width does not fit Metal")?,
        height: u64::try_from(height).map_err(|_| "scissor height does not fit Metal")?,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn intersect_scissor_pair(
    draw: &IndexedDraw,
    top_left_register: u32,
    bottom_right_register: u32,
    offset_x: i64,
    offset_y: i64,
    ignore_all_zero: bool,
    bounds: &mut [i64; 4],
) {
    let (Some(top_left), Some(bottom_right)) = (
        register_value(&draw.context_registers, top_left_register),
        register_value(&draw.context_registers, bottom_right_register),
    ) else {
        return;
    };
    if ignore_all_zero && top_left == 0 && bottom_right == 0 {
        return;
    }
    let pair = [
        i64::from(top_left & 0x7fff) + offset_x,
        i64::from((top_left >> 16) & 0x7fff) + offset_y,
        i64::from(bottom_right & 0x7fff) + offset_x,
        i64::from((bottom_right >> 16) & 0x7fff) + offset_y,
    ];
    bounds[0] = bounds[0].max(pair[0]);
    bounds[1] = bounds[1].max(pair[1]);
    bounds[2] = bounds[2].min(pair[2]);
    bounds[3] = bounds[3].min(pair[3]);
}

pub(super) fn decode_viewport(
    draw: &IndexedDraw,
    target_width: u32,
    target_height: u32,
    scissor: GuestScissor,
) -> MTLViewport {
    let mut minimum_depth = 0.0_f32;
    let mut maximum_depth = 1.0_f32;
    if let (Some(minimum), Some(maximum)) = (
        register_value(&draw.context_registers, 0x0b4).map(f32::from_bits),
        register_value(&draw.context_registers, 0x0b5).map(f32::from_bits),
    ) && minimum.is_finite()
        && maximum.is_finite()
        && maximum > minimum
    {
        minimum_depth = minimum;
        maximum_depth = maximum;
    }
    let programmed = [0x10f, 0x110, 0x111, 0x112]
        .map(|register| register_value(&draw.context_registers, register).map(f32::from_bits));
    if let [Some(x_scale), Some(x_offset), Some(y_scale), Some(y_offset)] = programmed
        && [x_scale, x_offset, y_scale, y_offset]
            .into_iter()
            .all(f32::is_finite)
        && x_scale > 0.0
        && y_scale != 0.0
    {
        let mut origin_y = y_offset - y_scale;
        let mut height = y_scale * 2.0;
        if height < 0.0 {
            origin_y += height;
            height = -height;
        }
        return MTLViewport {
            originX: f64::from(x_offset - x_scale),
            originY: f64::from(origin_y),
            width: f64::from(x_scale * 2.0),
            height: f64::from(height),
            znear: f64::from(minimum_depth),
            zfar: f64::from(maximum_depth),
        };
    }
    let (x, y, width, height) = match scissor {
        GuestScissor::Rect(rect) => (rect.x, rect.y, rect.width, rect.height),
        GuestScissor::Full | GuestScissor::Empty => {
            (0, 0, u64::from(target_width), u64::from(target_height))
        }
    };
    MTLViewport {
        #[allow(clippy::cast_precision_loss)]
        originX: x as f64,
        #[allow(clippy::cast_precision_loss)]
        originY: y as f64,
        #[allow(clippy::cast_precision_loss)]
        width: width as f64,
        #[allow(clippy::cast_precision_loss)]
        height: height as f64,
        znear: f64::from(minimum_depth),
        zfar: f64::from(maximum_depth),
    }
}

pub(super) fn register_value(registers: &[GpuRegister], offset: u32) -> Option<u32> {
    registers
        .binary_search_by_key(&offset, |register| register.offset)
        .ok()
        .map(|index| registers[index].value)
}

pub(super) fn pixel_input_controls(registers: &[GpuRegister]) -> [u32; 32] {
    std::array::from_fn(|index| {
        let index = u32::try_from(index).expect("PS input-control index fits u32");
        register_value(registers, SPI_PS_INPUT_CNTL_0 + index).unwrap_or(index)
    })
}

pub(super) fn read_draw_indices(draw: &IndexedDraw) -> Option<Arc<[u32]>> {
    if let Some(indices) = &draw.index_snapshot {
        return Some(indices.clone());
    }
    if draw.index_buffer_address < 0x10_000 || draw.index_count > 65_536 {
        return None;
    }
    let element_size = match draw.index_size {
        0 => 2_u64,
        1 => 4,
        2 => 1,
        _ => return None,
    };
    let first = u64::from(draw.index_offset).checked_mul(element_size)?;
    let base = draw.index_buffer_address.checked_add(first)?;
    Some(
        (0..draw.index_count)
            .map(|index| {
                let address = base + u64::from(index) * element_size;
                let address = usize::try_from(address).expect("validated index address");
                // SAFETY: The draw snapshot retains a live index allocation containing the
                // bounded index range used by this synchronous software draw.
                unsafe {
                    match element_size {
                        1 => u32::from(std::ptr::read_unaligned(address as *const u8)),
                        2 => u32::from(std::ptr::read_unaligned(address as *const u16)),
                        _ => std::ptr::read_unaligned(address as *const u32),
                    }
                }
            })
            .collect::<Vec<_>>()
            .into(),
    )
}
