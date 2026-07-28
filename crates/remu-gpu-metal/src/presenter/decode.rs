// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    DecodedDepthTarget, DecodedImageDescriptor, DecodedRasterState, DecodedRenderTarget,
    DepthTargetPipelineState, DisplayConfiguration, GraphicsPipelineKey, GraphicsShaderResourceKey,
    GuestScissor, GuestTextureDescriptor, MetalBlendAttachmentState, ProgramIdentity,
    RenderTargetFormat, RenderTargetPipelineState, decode_sampled_texture_format,
    effective_linear_pitch, guest_texture_byte_count, half_float_to_f32, is_storage_image_opcode,
    program_embeds_global_base, tiled_block_dimensions, trace_image_descriptors,
    trace_target_sampling, unsigned_float_to_f32,
};
use metal::{
    MTLBlendFactor, MTLBlendOperation, MTLClearColor, MTLColorWriteMask, MTLCompareFunction,
    MTLCullMode, MTLPixelFormat, MTLPrimitiveTopologyClass, MTLPrimitiveType, MTLScissorRect,
    MTLTriangleFillMode, MTLViewport, MTLWinding,
};
use remu_gpu::{GpuRegister, GpuScanoutTarget, IndexedDraw};
use remu_shader::{ImageBinding, PixelOutputKind, ShaderEvaluation, image_is_comparison};
use std::collections::HashSet;
use std::sync::{Arc, Mutex, OnceLock};

pub(super) const CB_TARGET_MASK: u32 = 0x08e;

pub(super) const CB_BLEND_RED: u32 = 0x105;

pub(super) const CB_BLEND_GREEN: u32 = 0x106;

pub(super) const CB_BLEND_BLUE: u32 = 0x107;

pub(super) const CB_BLEND_ALPHA: u32 = 0x108;

pub(super) const SPI_PS_INPUT_CNTL_0: u32 = 0x191;

pub(super) const CB_BLEND0_CONTROL: u32 = 0x1e0;

pub(super) const PA_SU_SC_MODE_CNTL: u32 = 0x205;

pub(super) const CB_COLOR0_BASE: u32 = 0x318;

pub(super) const CB_COLOR_REGISTER_STRIDE: u32 = 15;

pub(super) const CB_COLOR0_VIEW: u32 = 0x31b;

pub(super) const CB_COLOR0_INFO: u32 = 0x31c;

pub(super) const CB_COLOR0_ATTRIB: u32 = 0x31d;

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

pub(super) fn programmed_viewport_has_negative_height(draw: &IndexedDraw) -> bool {
    let programmed = [0x10f, 0x110, 0x111, 0x112]
        .map(|register| register_value(&draw.context_registers, register).map(f32::from_bits));
    matches!(
        programmed,
        [Some(x_scale), Some(x_offset), Some(y_scale), Some(y_offset)]
            if [x_scale, x_offset, y_scale, y_offset]
                .into_iter()
                .all(f32::is_finite)
                && x_scale > 0.0
                && y_scale < 0.0
    )
}

pub(super) fn decode_image_descriptor(words: &[u32]) -> Result<DecodedImageDescriptor, String> {
    if words.len() < 4 {
        return Err(format!(
            "image resource descriptor has {} dwords, expected at least four",
            words.len()
        ));
    }
    let address = ((u64::from(words[1] & 0xff) << 32) | u64::from(words[0])) << 8;
    let width = ((words[1] >> 30) | ((words[2] & 0x3fff) << 2))
        .checked_add(1)
        .ok_or("image width overflows")?;
    let height = ((words[2] >> 14) & 0xffff)
        .checked_add(1)
        .ok_or("image height overflows")?;
    let unified_format = (words[1] >> 20) & 0x1ff;
    let (data_format, number_format) = remu_shader::decode_unified_format(unified_format)
        .ok_or_else(|| format!("unknown GFX10 unified image format {unified_format}"))?;
    let (pixel_format, bytes_per_element) =
        decode_sampled_texture_format(data_format, number_format).map_err(|_| {
        format!(
            "Metal image upload does not yet implement GFX10 format {data_format}/{number_format} (unified {unified_format})"
        )
    })?;
    let tile_mode = (words[3] >> 20) & 0x1f;
    let texture_type = (words[3] >> 28) & 0xf;
    let base_level = (words[3] >> 12) & 0xf;
    let last_level = (words[3] >> 16) & 0xf;
    let destination_select = words[3] & 0xfff;
    let descriptor_pitch_word = words.get(4).copied().unwrap_or_default();
    let has_explicit_descriptor_pitch =
        matches!(texture_type, 8 | 9 | 14) && descriptor_pitch_word != 0;
    let descriptor_pitch = if has_explicit_descriptor_pitch {
        (descriptor_pitch_word & 0x3fff)
            .checked_add(1)
            .ok_or("image pitch overflows")?
    } else {
        width
    };
    if trace_image_descriptors() && (tile_mode == 9 || (width >= 700 && height >= 500)) {
        static LOGGED_IMAGES: OnceLock<Mutex<HashSet<(u64, u32, u32, u32, u32, u32)>>> =
            OnceLock::new();
        let key = (
            address,
            width,
            height,
            unified_format,
            descriptor_pitch,
            tile_mode,
        );
        let mut logged = LOGGED_IMAGES
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if logged.len() < 128 && logged.insert(key) {
            eprintln!(
                "[REMu][METAL][IMAGE] mode={tile_mode} address={address:#x} extent={width}x{height} pitch={descriptor_pitch} format={data_format}/{number_format} unified={unified_format} type={texture_type}"
            );
        }
    }
    let pitch = if tile_mode == 0 && !has_explicit_descriptor_pitch {
        effective_linear_pitch(width, bytes_per_element).ok_or("linear image pitch overflows")?
    } else {
        descriptor_pitch
    };
    if address < 0x10_000 || width > 8_192 || height > 8_192 || pitch < width || pitch > 8_192 {
        return Err(format!(
            "invalid image range address={address:#x} extent={width}x{height} pitch={pitch}"
        ));
    }
    if !matches!(texture_type, 9..=11 | 13) {
        return Err(format!(
            "Metal image upload does not yet implement texture type {texture_type}"
        ));
    }
    let layer_count = if texture_type == 10 {
        (words.get(4).copied().unwrap_or_default() & 0x1fff)
            .checked_add(1)
            .ok_or("3D image depth overflows")?
    } else if texture_type == 11 {
        let last_array_slice = words.get(4).copied().unwrap_or_default() & 0x1fff;
        let base_array_slice = (words.get(4).copied().unwrap_or_default() >> 16) & 0x1fff;
        if base_array_slice > last_array_slice {
            return Err(format!(
                "invalid cube array slice range {base_array_slice}..={last_array_slice}"
            ));
        }
        let layers = last_array_slice - base_array_slice + 1;
        if !layers.is_multiple_of(6) {
            return Err(format!(
                "cube resource exposes {layers} faces instead of a multiple of six"
            ));
        }
        layers
    } else if texture_type == 13 {
        let slices = words.get(4).copied().unwrap_or_default();
        let last_array_slice = slices & 0x1fff;
        let base_array_slice = (slices >> 16) & 0x1fff;
        if base_array_slice > last_array_slice {
            return Err(format!(
                "invalid array slice range {base_array_slice}..={last_array_slice}"
            ));
        }
        last_array_slice - base_array_slice + 1
    } else {
        1
    };
    let maximum_mip_level = 31 - width.max(height).leading_zeros();
    if base_level > last_level || last_level > maximum_mip_level {
        return Err(format!(
            "invalid image mip range {base_level}..={last_level} for {width}x{height}"
        ));
    }
    if !matches!(tile_mode, 0 | 1 | 5 | 9 | 24 | 27) {
        return Err(format!(
            "Metal image upload does not yet implement tile mode {tile_mode}"
        ));
    }
    Ok(DecodedImageDescriptor {
        guest: GuestTextureDescriptor {
            address,
            width,
            height,
            pitch,
            tile_mode,
            pixel_format,
            bytes_per_pixel: bytes_per_element,
        },
        destination_select,
        base_mip_level: base_level,
        mip_level_count: last_level - base_level + 1,
        texture_type,
        layer_count,
    })
}

pub(super) fn sampled_alias_descriptor(binding: &ImageBinding) -> Option<DecodedImageDescriptor> {
    decode_image_descriptor(&binding.resource_descriptor).ok()
}

pub(super) fn fallback_image_descriptor(
    binding: &ImageBinding,
) -> Result<DecodedImageDescriptor, String> {
    let unified_format = binding
        .resource_descriptor
        .get(1)
        .map_or(0, |word| (word >> 20) & 0x1ff);
    let decoded_format = remu_shader::decode_unified_format(unified_format);
    let (pixel_format, bytes_per_pixel) = decoded_format
        .and_then(|(data_format, number_format)| {
            decode_sampled_texture_format(data_format, number_format).ok()
        })
        .unwrap_or_else(
            || match decoded_format.map(|(_, number_format)| number_format) {
                Some(4) => (MTLPixelFormat::RGBA8Uint, 4),
                Some(5) => (MTLPixelFormat::RGBA8Sint, 4),
                _ => (MTLPixelFormat::RGBA8Unorm, 4),
            },
        );
    let storage = is_storage_image_opcode(binding.opcode);
    let encoded_texture_type = binding
        .resource_descriptor
        .get(3)
        .map_or(0, |word| (word >> 28) & 0xf);
    // Match the texture type emitted by remu-shader-metal. Sample instructions select their
    // Metal dimension from the instruction itself, while storage instructions use the descriptor
    // only to distinguish 3D from 2D.
    let texture_type = if storage {
        if encoded_texture_type == 10 { 10 } else { 9 }
    } else {
        match binding.control.dimension {
            2 => 10,
            3 => 11,
            _ => 9,
        }
    };
    let layer_count = if texture_type == 11 { 6 } else { 1 };
    Ok(DecodedImageDescriptor {
        guest: GuestTextureDescriptor {
            address: 0,
            width: 1,
            height: 1,
            pitch: 1,
            tile_mode: 0,
            pixel_format,
            bytes_per_pixel,
        },
        destination_select: 0,
        base_mip_level: 0,
        mip_level_count: 1,
        texture_type,
        layer_count,
    })
}

pub(super) fn metal_primitive_type(guest: u32) -> Result<MTLPrimitiveType, String> {
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

pub(super) fn metal_primitive_topology_class(
    guest: u32,
) -> Result<MTLPrimitiveTopologyClass, String> {
    match guest {
        1 => Ok(MTLPrimitiveTopologyClass::Point),
        2 | 3 => Ok(MTLPrimitiveTopologyClass::Line),
        4 | 5 | 6 | 7 | 0x11 => Ok(MTLPrimitiveTopologyClass::Triangle),
        _ => Err(format!("unknown guest primitive topology {guest:#x}")),
    }
}

pub(super) const fn metal_nonindexed_vertex_count(
    guest_primitive_type: u32,
    guest_count: u32,
) -> u32 {
    if matches!(guest_primitive_type, 7 | 0x11) && guest_count == 3 {
        4
    } else {
        guest_count
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

pub(super) fn decode_render_target_format(
    data_format: u32,
    number_type: u32,
    component_swap: u32,
) -> Result<RenderTargetFormat, String> {
    // Mesa's ac_get_cb_format/ac_get_cb_number_type tables establish the valid GFX10
    // combinations. Metal has direct, bit-compatible render-target formats for this cohort.
    // Scaled number types have no Metal color-attachment equivalent and are rejected.
    let (pixel_format, bytes_per_pixel) = match (data_format, number_type, component_swap) {
        (1, 0, 0) => (MTLPixelFormat::R8Unorm, 1),
        (1, 1, 0) => (MTLPixelFormat::R8Snorm, 1),
        (1, 4, 0) => (MTLPixelFormat::R8Uint, 1),
        (1, 5, 0) => (MTLPixelFormat::R8Sint, 1),
        (1, 6, 0) => (MTLPixelFormat::R8Unorm_sRGB, 1),
        (2, 0, 0) => (MTLPixelFormat::R16Unorm, 2),
        (2, 1, 0) => (MTLPixelFormat::R16Snorm, 2),
        (2, 4, 0) => (MTLPixelFormat::R16Uint, 2),
        (2, 5, 0) => (MTLPixelFormat::R16Sint, 2),
        (2, 7, 0) => (MTLPixelFormat::R16Float, 2),
        (3, 0, 0) => (MTLPixelFormat::RG8Unorm, 2),
        (3, 1, 0) => (MTLPixelFormat::RG8Snorm, 2),
        (3, 4, 0) => (MTLPixelFormat::RG8Uint, 2),
        (3, 5, 0) => (MTLPixelFormat::RG8Sint, 2),
        (3, 6, 0) => (MTLPixelFormat::RG8Unorm_sRGB, 2),
        (4, 4, 0) => (MTLPixelFormat::R32Uint, 4),
        (4, 5, 0) => (MTLPixelFormat::R32Sint, 4),
        (4, 7, 0) => (MTLPixelFormat::R32Float, 4),
        (5, 0, 0) => (MTLPixelFormat::RG16Unorm, 4),
        (5, 1, 0) => (MTLPixelFormat::RG16Snorm, 4),
        (5, 4, 0) => (MTLPixelFormat::RG16Uint, 4),
        (5, 5, 0) => (MTLPixelFormat::RG16Sint, 4),
        (5, 7, 0) => (MTLPixelFormat::RG16Float, 4),
        // Metal's RG11B10Float is bit-compatible only with AMD 10_11_11_FLOAT. The reversed
        // 11_11_10 layout requires repacking and an R/B selector remap; treating it as the same
        // attachment silently exchanges both component widths and positions.
        (6, 7, 0) => (MTLPixelFormat::RG11B10Float, 4),
        // Gen5 2_10_10_10 places a 2-bit red component in the low bits and cannot be represented
        // by either Metal packed-10 format without repacking. Standard 10_10_10_2 puts R in bits
        // 0..9 and is bit-compatible with Metal RGB10A2.
        (9, 0, 0) => (MTLPixelFormat::RGB10A2Unorm, 4),
        (10, 0, 0) => (MTLPixelFormat::RGBA8Unorm, 4),
        (10, 0, 1) => (MTLPixelFormat::BGRA8Unorm, 4),
        (10, 1, 0) => (MTLPixelFormat::RGBA8Snorm, 4),
        (10, 4, 0) => (MTLPixelFormat::RGBA8Uint, 4),
        (10, 5, 0) => (MTLPixelFormat::RGBA8Sint, 4),
        (10, 6, 0) => (MTLPixelFormat::RGBA8Unorm_sRGB, 4),
        (10, 6, 1) => (MTLPixelFormat::BGRA8Unorm_sRGB, 4),
        (11, 4, 0) => (MTLPixelFormat::RG32Uint, 8),
        (11, 5, 0) => (MTLPixelFormat::RG32Sint, 8),
        (11, 7, 0) => (MTLPixelFormat::RG32Float, 8),
        (12, 0, 0) => (MTLPixelFormat::RGBA16Unorm, 8),
        (12, 1, 0) => (MTLPixelFormat::RGBA16Snorm, 8),
        (12, 4, 0) => (MTLPixelFormat::RGBA16Uint, 8),
        (12, 5, 0) => (MTLPixelFormat::RGBA16Sint, 8),
        // Gen5 commonly programs ALT component swap for 16-bit float scanout surfaces. Metal has
        // no BGRA16Float attachment; the shader export remains in logical component order, which
        // is the behavior used by the native Gen5 path for this attachment class.
        (12, 7, 0 | 1) => (MTLPixelFormat::RGBA16Float, 8),
        (14, 4, 0) => (MTLPixelFormat::RGBA32Uint, 16),
        (14, 5, 0) => (MTLPixelFormat::RGBA32Sint, 16),
        (14, 7, 0) => (MTLPixelFormat::RGBA32Float, 16),
        (24, 7, 0) => (MTLPixelFormat::RGB9E5Float, 4),
        _ => {
            return Err(format!(
                "unsupported GFX10 color format={data_format} number_type={number_type} component_swap={component_swap}"
            ));
        }
    };
    let output_kind = match number_type {
        4 => PixelOutputKind::Uint,
        5 => PixelOutputKind::Sint,
        _ => PixelOutputKind::Float,
    };
    Ok(RenderTargetFormat {
        pixel_format,
        output_kind,
        bytes_per_pixel,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ClearChannelEncoding {
    Unorm,
    Snorm,
    Uint,
    Sint,
    Float,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FastClearLayout {
    R8,
    R16,
    R32,
    Rg8,
    Rg16,
    Rg32,
    Rgba8,
    Bgra8,
    Rgba16,
    Rgba32,
    Rgb10A2,
    Rg11B10Float,
    Rgb9E5Float,
}

pub(super) fn sign_extend_clear_lane(value: u32, bits: u32) -> i64 {
    debug_assert!((1..=32).contains(&bits));
    let shift = 64 - bits;
    ((u64::from(value) << shift) as i64) >> shift
}

pub(super) fn decode_clear_lane(value: u32, bits: u32, encoding: ClearChannelEncoding) -> f64 {
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1_u32 << bits) - 1
    };
    let value = value & mask;
    match encoding {
        ClearChannelEncoding::Unorm => f64::from(value) / f64::from(mask),
        ClearChannelEncoding::Snorm => {
            let maximum = (1_i64 << (bits - 1)) - 1;
            ((sign_extend_clear_lane(value, bits) as f64) / (maximum as f64)).max(-1.0)
        }
        ClearChannelEncoding::Uint => f64::from(value),
        ClearChannelEncoding::Sint => sign_extend_clear_lane(value, bits) as f64,
        ClearChannelEncoding::Float if bits == 16 => f64::from(half_float_to_f32(value as u16)),
        ClearChannelEncoding::Float if bits == 32 => f64::from(f32::from_bits(value)),
        ClearChannelEncoding::Float => unreachable!("only IEEE 16- and 32-bit lanes are supported"),
    }
}

pub(super) fn inverse_srgb_clear_component(encoded: f64) -> f64 {
    if encoded <= 0.040_45 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

pub(super) fn decode_fast_clear_color(
    pixel_format: MTLPixelFormat,
    words: [u32; 2],
) -> Option<MTLClearColor> {
    use ClearChannelEncoding::{Float, Sint, Snorm, Uint, Unorm};
    use FastClearLayout::{
        Bgra8, R8, R16, R32, Rg8, Rg11B10Float, Rg16, Rg32, Rgb9E5Float, Rgb10A2, Rgba8, Rgba16,
        Rgba32,
    };

    let (encoding, layout, srgb) = match pixel_format {
        MTLPixelFormat::R8Unorm => (Unorm, R8, false),
        MTLPixelFormat::R8Snorm => (Snorm, R8, false),
        MTLPixelFormat::R8Uint => (Uint, R8, false),
        MTLPixelFormat::R8Sint => (Sint, R8, false),
        MTLPixelFormat::R8Unorm_sRGB => (Unorm, R8, true),
        MTLPixelFormat::R16Unorm => (Unorm, R16, false),
        MTLPixelFormat::R16Snorm => (Snorm, R16, false),
        MTLPixelFormat::R16Uint => (Uint, R16, false),
        MTLPixelFormat::R16Sint => (Sint, R16, false),
        MTLPixelFormat::R16Float => (Float, R16, false),
        MTLPixelFormat::RG8Unorm => (Unorm, Rg8, false),
        MTLPixelFormat::RG8Snorm => (Snorm, Rg8, false),
        MTLPixelFormat::RG8Uint => (Uint, Rg8, false),
        MTLPixelFormat::RG8Sint => (Sint, Rg8, false),
        MTLPixelFormat::RG8Unorm_sRGB => (Unorm, Rg8, true),
        MTLPixelFormat::R32Uint => (Uint, R32, false),
        MTLPixelFormat::R32Sint => (Sint, R32, false),
        MTLPixelFormat::R32Float => (Float, R32, false),
        MTLPixelFormat::RG16Unorm => (Unorm, Rg16, false),
        MTLPixelFormat::RG16Snorm => (Snorm, Rg16, false),
        MTLPixelFormat::RG16Uint => (Uint, Rg16, false),
        MTLPixelFormat::RG16Sint => (Sint, Rg16, false),
        MTLPixelFormat::RG16Float => (Float, Rg16, false),
        MTLPixelFormat::RG11B10Float => (Float, Rg11B10Float, false),
        MTLPixelFormat::RGB10A2Unorm => (Unorm, Rgb10A2, false),
        MTLPixelFormat::RGBA8Unorm => (Unorm, Rgba8, false),
        MTLPixelFormat::BGRA8Unorm => (Unorm, Bgra8, false),
        MTLPixelFormat::RGBA8Snorm => (Snorm, Rgba8, false),
        MTLPixelFormat::RGBA8Uint => (Uint, Rgba8, false),
        MTLPixelFormat::RGBA8Sint => (Sint, Rgba8, false),
        MTLPixelFormat::RGBA8Unorm_sRGB => (Unorm, Rgba8, true),
        MTLPixelFormat::BGRA8Unorm_sRGB => (Unorm, Bgra8, true),
        MTLPixelFormat::RG32Uint => (Uint, Rg32, false),
        MTLPixelFormat::RG32Sint => (Sint, Rg32, false),
        MTLPixelFormat::RG32Float => (Float, Rg32, false),
        MTLPixelFormat::RGBA16Unorm => (Unorm, Rgba16, false),
        MTLPixelFormat::RGBA16Snorm => (Snorm, Rgba16, false),
        MTLPixelFormat::RGBA16Uint => (Uint, Rgba16, false),
        MTLPixelFormat::RGBA16Sint => (Sint, Rgba16, false),
        MTLPixelFormat::RGBA16Float => (Float, Rgba16, false),
        MTLPixelFormat::RGBA32Uint => (Uint, Rgba32, false),
        MTLPixelFormat::RGBA32Sint => (Sint, Rgba32, false),
        MTLPixelFormat::RGBA32Float => (Float, Rgba32, false),
        MTLPixelFormat::RGB9E5Float => (Float, Rgb9E5Float, false),
        _ => return None,
    };

    let packed = u64::from(words[0]) | (u64::from(words[1]) << 32);
    let lane = |shift: u32, bits: u32| decode_clear_lane((packed >> shift) as u32, bits, encoding);
    let (mut red, mut green, mut blue, alpha) = match layout {
        R8 => (lane(0, 8), 0.0, 0.0, 1.0),
        R16 => (lane(0, 16), 0.0, 0.0, 1.0),
        R32 => (lane(0, 32), 0.0, 0.0, 1.0),
        Rg8 => (lane(0, 8), lane(8, 8), 0.0, 1.0),
        Rg16 => (lane(0, 16), lane(16, 16), 0.0, 1.0),
        Rg32 => (lane(0, 32), lane(32, 32), 0.0, 1.0),
        Rgba8 => (lane(0, 8), lane(8, 8), lane(16, 8), lane(24, 8)),
        Bgra8 => (lane(16, 8), lane(8, 8), lane(0, 8), lane(24, 8)),
        Rgba16 => (lane(0, 16), lane(16, 16), lane(32, 16), lane(48, 16)),
        // GFX10 represents a fast clear for a 128-bit target with one shared RGB word and a
        // separate alpha word. Mesa only enables this representation when R, G, and B are equal.
        Rgba32 => {
            let rgb = decode_clear_lane(words[0], 32, encoding);
            (rgb, rgb, rgb, decode_clear_lane(words[1], 32, encoding))
        }
        Rgb10A2 => (lane(0, 10), lane(10, 10), lane(20, 10), lane(30, 2)),
        Rg11B10Float => (
            f64::from(unsigned_float_to_f32(words[0] & 0x7ff, 6)),
            f64::from(unsigned_float_to_f32((words[0] >> 11) & 0x7ff, 6)),
            f64::from(unsigned_float_to_f32((words[0] >> 22) & 0x3ff, 5)),
            1.0,
        ),
        Rgb9E5Float => {
            let exponent = ((words[0] >> 27) & 0x1f) as i32 - 24;
            let scale = 2.0_f64.powi(exponent);
            (
                f64::from(words[0] & 0x1ff) * scale,
                f64::from((words[0] >> 9) & 0x1ff) * scale,
                f64::from((words[0] >> 18) & 0x1ff) * scale,
                1.0,
            )
        }
    };
    if srgb {
        red = inverse_srgb_clear_component(red);
        green = inverse_srgb_clear_component(green);
        blue = inverse_srgb_clear_component(blue);
    }
    Some(MTLClearColor::new(red, green, blue, alpha))
}

pub(super) fn render_target_fast_clear_color(
    target: &DecodedRenderTarget,
    context_registers: &[GpuRegister],
) -> Option<MTLClearColor> {
    if !target.fast_clear_enabled {
        return None;
    }
    let primary = target.guest_slot * CB_COLOR_REGISTER_STRIDE;
    let words = [
        register_value(context_registers, CB_COLOR0_CLEAR_WORD0 + primary).unwrap_or_default(),
        register_value(context_registers, CB_COLOR0_CLEAR_WORD1 + primary).unwrap_or_default(),
    ];
    decode_fast_clear_color(target.texture.pixel_format, words)
}

pub(super) fn initial_render_target_clear_color(
    target: &DecodedRenderTarget,
    context_registers: &[GpuRegister],
    has_resident_storage: bool,
) -> Option<MTLClearColor> {
    if has_resident_storage {
        return None;
    }
    render_target_fast_clear_color(target, context_registers).or_else(|| {
        // DCC backing memory cannot be decoded as the uncompressed image represented by the
        // resident Metal texture. Without a programmed fast clear, discard that compressed
        // first-use content instead of loading its raw bytes through an uncompressed attachment.
        target
            .dcc_enabled
            .then(|| MTLClearColor::new(0.0, 0.0, 0.0, 1.0))
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_render_target_state(
    guest_slot: u32,
    address: u64,
    view: u32,
    info: u32,
    attrib: u32,
    attrib2: u32,
    attrib3: u32,
    write_mask: u32,
) -> Result<DecodedRenderTarget, String> {
    let width = ((attrib2 >> 14) & 0x3fff)
        .checked_add(1)
        .ok_or("render-target width overflows")?;
    let height = (attrib2 & 0x3fff)
        .checked_add(1)
        .ok_or("render-target height overflows")?;
    if address < 0x10_000 || width > 16_384 || height > 16_384 {
        return Err(format!(
            "invalid color target {guest_slot} address={address:#x} extent={width}x{height}"
        ));
    }

    let endian = info & 0x3;
    let fast_clear = (info >> 13) & 1;
    let fmask_compression = (info >> 14) & 1;
    let dcc_enable = (info >> 28) & 1;
    if endian != 0 || fmask_compression != 0 {
        return Err(format!(
            "color target {guest_slot} requires unsupported metadata/endian state endian={endian} fast_clear={fast_clear} fmask={fmask_compression} dcc={dcc_enable}"
        ));
    }
    let sample_log2 = (attrib >> 12) & 0x7;
    let fragment_log2 = (attrib >> 15) & 0x3;
    if sample_log2 != fragment_log2 || sample_log2 > 3 {
        return Err(format!(
            "color target {guest_slot} has unrepresentable sample/fragment counts (samples_log2={sample_log2}, fragments_log2={fragment_log2})"
        ));
    }
    let sample_count = 1_u32 << sample_log2;
    let slice_start = view & 0x1fff;
    let slice_max = (view >> 13) & 0x1fff;
    let mip_level = (view >> 26) & 0xf;
    let max_mip = (attrib2 >> 28) & 0xf;
    let mip0_depth = attrib3 & 0x1fff;
    let tile_mode = (attrib3 >> 14) & 0x1f;
    let maximum_extent_mip = 31 - width.max(height).leading_zeros();
    if mip_level > max_mip || max_mip > maximum_extent_mip {
        return Err(format!(
            "color target {guest_slot} has invalid mip view slices={slice_start}..={slice_max} mip={mip_level}/{max_mip} maximum={maximum_extent_mip} depth={mip0_depth}"
        ));
    }
    if max_mip != 0 && (sample_count != 1 || tile_mode != 27) {
        return Err(format!(
            "color target {guest_slot} mip hierarchy requires single-sample mode-27 storage, got samples={sample_count} tile={tile_mode} mip={mip_level}/{max_mip}"
        ));
    }
    if !matches!(tile_mode, 0 | 27) {
        return Err(format!(
            "color target {guest_slot} uses unsupported GFX10 swizzle mode {tile_mode}"
        ));
    }
    let resource_type = (attrib3 >> 24) & 0x3;
    let resource_level = (attrib3 >> 27) & 0x7;
    let non_block_compressed_view = (attrib3 >> 31) & 1;
    if resource_level != 0 || non_block_compressed_view != 0 {
        return Err(format!(
            "color target {guest_slot} requires unsupported resource state type={resource_type} level={resource_level} nbc={non_block_compressed_view} extent={width}x{height}"
        ));
    }
    let (texture_type, layer_count, base_layer) = match resource_type {
        0 if height == 1 && slice_start == 0 && slice_max == 0 && mip0_depth == 0 => (9, 1, 0),
        1 if slice_start == 0 && slice_max == 0 && mip0_depth == 0 => (9, 1, 0),
        2 if slice_start == 0 && mip0_depth != 0 => {
            let layers = mip0_depth
                .checked_add(1)
                .ok_or("3D color-target depth overflows")?;
            // GFX10 programs MIP0_DEPTH as depth-1. Some Prospero command streams program
            // SLICE_MAX as either the inclusive final layer or the exclusive layer count.
            if slice_max != mip0_depth && slice_max != layers {
                return Err(format!(
                    "color target {guest_slot} has inconsistent 3D slice/depth state slices={slice_start}..={slice_max} depth={mip0_depth}"
                ));
            }
            (10, layers, 0)
        }
        _ => {
            return Err(format!(
                "color target {guest_slot} requires unsupported resource state type={resource_type} slices={slice_start}..={slice_max} depth={mip0_depth} extent={width}x{height}"
            ));
        }
    };
    if max_mip != 0 && texture_type != 9 {
        return Err(format!(
            "color target {guest_slot} mipmapped resource type={resource_type} is not implemented"
        ));
    }
    let format =
        decode_render_target_format((info >> 2) & 0x1f, (info >> 8) & 0x7, (info >> 11) & 0x3)?;
    let pitch = if tile_mode == 0 {
        effective_linear_pitch(width, format.bytes_per_pixel)
            .ok_or("linear render-target pitch overflows")?
    } else {
        width
    };
    Ok(DecodedRenderTarget {
        guest_slot,
        texture: GuestTextureDescriptor {
            address,
            width,
            height,
            pitch,
            tile_mode,
            pixel_format: format.pixel_format,
            bytes_per_pixel: format.bytes_per_pixel,
        },
        output_kind: format.output_kind,
        write_mask,
        direct_scanout: false,
        sample_count,
        texture_type,
        layer_count,
        base_layer,
        mip_level,
        mip_level_count: max_mip + 1,
        dcc_enabled: dcc_enable != 0,
        fast_clear_enabled: fast_clear != 0,
    })
}

pub(super) fn decode_render_targets(
    draw: &IndexedDraw,
) -> Result<Vec<DecodedRenderTarget>, String> {
    let pixel = draw
        .pixel_shader
        .as_ref()
        .ok_or("draw has no decoded pixel shader")?;
    let export_masks = pixel.program.pixel_color_export_masks();
    if let Some(target) = draw.scanout_target {
        let shader_write_mask = (export_masks >> (target.export_slot * 4)) & 0xf;
        if shader_write_mask == 0 || shader_write_mask & target.write_mask == 0 {
            return Err(format!(
                "scanout target slot {} has no matching pixel export ({export_masks:#010x})",
                target.export_slot
            ));
        }
        return decode_scanout_target(target).map(|target| vec![target]);
    }
    decode_render_targets_from_registers(&draw.context_registers, export_masks)
}

pub(super) fn decode_scanout_target(
    target: GpuScanoutTarget,
) -> Result<DecodedRenderTarget, String> {
    if let Some(surface) = target.color_surface {
        let mut decoded = decode_render_target_state(
            target.export_slot,
            surface.address,
            surface.view,
            surface.info,
            surface.attrib,
            surface.attrib2,
            surface.attrib3,
            target.write_mask,
        )?;
        if decoded.texture.address != target.address
            || decoded.texture.width != target.width
            || decoded.texture.height != target.height
        {
            return Err(
                "programmed scanout surface disagrees with VideoOut registration".to_owned(),
            );
        }
        decoded.guest_slot = target.export_slot;
        return Ok(decoded);
    }
    let (pixel_format, bytes_per_pixel) = decode_video_out_pixel_format(target.pixel_format)?;
    if target.address < 0x10_000
        || target.width == 0
        || target.height == 0
        || target.width > 16_384
        || target.height > 16_384
        || target.pitch < target.width
    {
        return Err(format!(
            "invalid direct scanout address={:#x} extent={}x{} pitch={}",
            target.address, target.width, target.height, target.pitch
        ));
    }
    Ok(DecodedRenderTarget {
        guest_slot: target.export_slot,
        texture: GuestTextureDescriptor {
            address: target.address,
            width: target.width,
            height: target.height,
            pitch: target.pitch,
            tile_mode: 0,
            pixel_format,
            bytes_per_pixel,
        },
        output_kind: PixelOutputKind::Float,
        write_mask: target.write_mask,
        direct_scanout: true,
        sample_count: 1,
        texture_type: 9,
        layer_count: 1,
        base_layer: 0,
        mip_level: 0,
        mip_level_count: 1,
        dcc_enabled: false,
        fast_clear_enabled: false,
    })
}

pub(super) fn decode_video_out_pixel_format(
    pixel_format: u64,
) -> Result<(MTLPixelFormat, u32), String> {
    const A8_R8_G8_B8_SRGB: u64 = 0x8000_0000;
    const A8_B8_G8_R8_SRGB: u64 = 0x8000_2200;
    const A2_R10_G10_B10: u64 = 0x8806_0000;
    const A2_R10_G10_B10_SRGB: u64 = 0x8800_0000;
    const A2_R10_G10_B10_BT2020_PQ: u64 = 0x8874_0000;
    const F2_R8_G8_B8_A8_SRGB: u64 = 0x8000_0000_2200_0000;
    const F2_B8_G8_R8_A8_SRGB: u64 = 0x8000_0000_0000_0000;
    const F2_R10_G10_B10_A2: u64 = 0x8100_0006_2200_0000;
    const F2_B10_G10_R10_A2: u64 = 0x8100_0006_0000_0000;
    const F2_R10_G10_B10_A2_SRGB: u64 = 0x8100_0000_2200_0000;
    const F2_B10_G10_R10_A2_SRGB: u64 = 0x8100_0000_0000_0000;
    const F2_R10_G10_B10_A2_BT2100_PQ: u64 = 0x8100_0704_2200_0000;
    const F2_B10_G10_R10_A2_BT2100_PQ: u64 = 0x8100_0704_0000_0000;
    let pixel_format = match pixel_format {
        A8_R8_G8_B8_SRGB
        | A8_B8_G8_R8_SRGB
        | A2_R10_G10_B10
        | A2_R10_G10_B10_SRGB
        | A2_R10_G10_B10_BT2020_PQ
        | F2_R8_G8_B8_A8_SRGB
        | F2_B8_G8_R8_A8_SRGB
        | F2_R10_G10_B10_A2
        | F2_B10_G10_R10_A2
        | F2_R10_G10_B10_A2_SRGB
        | F2_B10_G10_R10_A2_SRGB
        | F2_R10_G10_B10_A2_BT2100_PQ
        | F2_B10_G10_R10_A2_BT2100_PQ => pixel_format,
        _ => {
            return Err(format!(
                "unsupported VideoOut pixel format {pixel_format:#018x}"
            ));
        }
    };
    let metal = match pixel_format {
        A8_B8_G8_R8_SRGB | F2_R8_G8_B8_A8_SRGB => MTLPixelFormat::RGBA8Unorm_sRGB,
        A8_R8_G8_B8_SRGB | F2_B8_G8_R8_A8_SRGB => MTLPixelFormat::BGRA8Unorm_sRGB,
        F2_R10_G10_B10_A2 | F2_R10_G10_B10_A2_SRGB | F2_R10_G10_B10_A2_BT2100_PQ => {
            MTLPixelFormat::RGB10A2Unorm
        }
        A2_R10_G10_B10
        | A2_R10_G10_B10_SRGB
        | A2_R10_G10_B10_BT2020_PQ
        | F2_B10_G10_R10_A2
        | F2_B10_G10_R10_A2_SRGB
        | F2_B10_G10_R10_A2_BT2100_PQ => MTLPixelFormat::BGR10A2Unorm,
        _ => unreachable!("VideoOut format cohort was exhaustively matched"),
    };
    Ok((metal, 4))
}

#[allow(clippy::too_many_lines)]
pub(super) fn decode_render_targets_from_registers(
    context_registers: &[GpuRegister],
    export_masks: u32,
) -> Result<Vec<DecodedRenderTarget>, String> {
    let target_mask = register_value(context_registers, CB_TARGET_MASK)
        .ok_or("draw has no CB_TARGET_MASK register")?;
    let mut targets = Vec::new();
    for guest_slot in 0..COLOR_TARGET_COUNT {
        let target_write_mask = (target_mask >> (guest_slot * 4)) & 0xf;
        let shader_write_mask = (export_masks >> (guest_slot * 4)) & 0xf;
        let write_mask = target_write_mask & shader_write_mask;
        if write_mask == 0 {
            continue;
        }
        let primary = guest_slot * CB_COLOR_REGISTER_STRIDE;
        let required = |offset, name| {
            register_value(context_registers, offset).ok_or_else(|| {
                format!("color target {guest_slot} has no {name} register ({offset:#x})")
            })
        };
        let base = required(CB_COLOR0_BASE + primary, "CB_COLOR_BASE")?;
        let base_ext = required(CB_COLOR0_BASE_EXT + guest_slot, "CB_COLOR_BASE_EXT")?;
        let view = required(CB_COLOR0_VIEW + primary, "CB_COLOR_VIEW")?;
        let info = required(CB_COLOR0_INFO + primary, "CB_COLOR_INFO")?;
        let attrib = required(CB_COLOR0_ATTRIB + primary, "CB_COLOR_ATTRIB")?;
        let attrib2 = required(CB_COLOR0_ATTRIB2 + guest_slot, "CB_COLOR_ATTRIB2")?;
        let attrib3 = required(CB_COLOR0_ATTRIB3 + guest_slot, "CB_COLOR_ATTRIB3")?;

        let address = (u64::from(base_ext & 0xff) << 40) | (u64::from(base) << 8);
        targets.push(decode_render_target_state(
            guest_slot, address, view, info, attrib, attrib2, attrib3, write_mask,
        )?);
    }
    if targets.is_empty() && target_mask != 0 {
        return Err(format!(
            "draw has no color target enabled by both CB_TARGET_MASK={target_mask:#010x} and pixel exports={export_masks:#010x}"
        ));
    }
    Ok(targets)
}

pub(super) fn decode_raster_state(draw: &IndexedDraw) -> Result<DecodedRasterState, String> {
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
    let mut front_winding = if control & (1 << 2) != 0 {
        MTLWinding::Clockwise
    } else {
        MTLWinding::CounterClockwise
    };
    // Gen5 titles commonly program a negative viewport Y scale to account for the framebuffer's
    // top-left origin. `decode_viewport` represents that transform with a positive Metal viewport
    // (Metal rejects a negative height), which mirrors the projected triangle orientation. Keep
    // guest front/back-face classification intact by applying the corresponding winding flip.
    if programmed_viewport_has_negative_height(draw) {
        front_winding = match front_winding {
            MTLWinding::Clockwise => MTLWinding::CounterClockwise,
            MTLWinding::CounterClockwise => MTLWinding::Clockwise,
        };
    }

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

pub(super) fn decode_depth_target(
    draw: &IndexedDraw,
) -> Result<Option<DecodedDepthTarget>, String> {
    let control = register_value(&draw.context_registers, DB_DEPTH_CONTROL).unwrap_or_default();
    let render_control =
        register_value(&draw.context_registers, DB_RENDER_CONTROL).unwrap_or_default();
    let depth_enabled = control & (1 << 1) != 0;
    let write_enabled = control & (1 << 2) != 0;
    let clear_enabled = render_control & 1 != 0;
    if !depth_enabled && !write_enabled && !clear_enabled {
        return Ok(None);
    }

    let required = |offset, name| {
        register_value(&draw.context_registers, offset)
            .ok_or_else(|| format!("active depth target is missing {name}"))
    };
    let z_info = required(DB_Z_INFO, "DB_Z_INFO")?;
    let size = required(DB_DEPTH_SIZE_XY, "DB_DEPTH_SIZE_XY")?;
    let format = match z_info & 0x3 {
        // The AGC register defaults leave the high metadata bit set while the format is zero.
        // Unreal can carry that default into a color-only composite even with Z_ENABLE still
        // latched. SharpEmu treats guest format zero as no depth surface.
        0 => return Ok(None),
        1 => (MTLPixelFormat::Depth16Unorm, 2),
        3 => (MTLPixelFormat::Depth32Float, 4),
        value => {
            return Err(format!(
                "active depth target uses unsupported Z format {value} in DB_Z_INFO={z_info:#010x}"
            ));
        }
    };
    let samples_log2 = (z_info >> 2) & 0x3;
    let sample_count = 1_u32 << samples_log2;
    // DB_Z_INFO does not carry the GFX10 texture SW_MODE in bits 4..8. Prospero's depth-target
    // packet carries a legacy tile-mode index in bits 20..22; index zero selects the fixed
    // 64-KiB depth layout. Shader resource descriptors expose that same allocation as SW mode
    // 0x18 (kDepth). Treating index zero as a linear texture caused every depth-sampling pass to
    // read back, reinterpret, and upload the full surface, and decoded the guest bytes wrongly.
    let depth_tile_mode_index = (z_info >> 20) & 0x7;
    if depth_tile_mode_index != 0 {
        return Err(format!(
            "active depth target uses unsupported tile-mode index {depth_tile_mode_index} in DB_Z_INFO={z_info:#010x}"
        ));
    }
    const DEPTH_TILE_MODE: u32 = 0x18;
    let tile_mode = DEPTH_TILE_MODE;

    let width = (size & 0x3fff)
        .checked_add(1)
        .ok_or("depth-target width overflows")?;
    let height = ((size >> 16) & 0x3fff)
        .checked_add(1)
        .ok_or("depth-target height overflows")?;
    let read_address = (u64::from(
        register_value(&draw.context_registers, DB_Z_READ_BASE_HI).unwrap_or_default() & 0xff,
    ) << 40)
        | (u64::from(register_value(&draw.context_registers, DB_Z_READ_BASE).unwrap_or_default())
            << 8);
    let write_address = (u64::from(
        register_value(&draw.context_registers, DB_Z_WRITE_BASE_HI).unwrap_or_default() & 0xff,
    ) << 40)
        | (u64::from(register_value(&draw.context_registers, DB_Z_WRITE_BASE).unwrap_or_default())
            << 8);
    if read_address != 0 && write_address != 0 && read_address != write_address {
        return Err(format!(
            "separate depth read/write surfaces are unsupported (read={read_address:#x}, write={write_address:#x})"
        ));
    }
    let address = if write_enabled && write_address != 0 {
        write_address
    } else if read_address != 0 {
        read_address
    } else {
        write_address
    };
    if address < 0x10_000 || width > 16_384 || height > 16_384 {
        return Err(format!(
            "invalid depth target address={address:#x} extent={width}x{height}"
        ));
    }
    let (block_width, _) = tiled_block_dimensions(
        65_536,
        usize::try_from(format.1).map_err(|_| "depth element size does not fit usize")?,
    )
    .ok_or("depth target has an unsupported element size")?;
    let pitch = usize::try_from(width)
        .map_err(|_| "depth-target width does not fit usize")?
        .div_ceil(block_width)
        .checked_mul(block_width)
        .and_then(|pitch| u32::try_from(pitch).ok())
        .ok_or("depth-target pitch overflows")?;
    let compare = match (control >> 4) & 0x7 {
        0 => MTLCompareFunction::Never,
        1 => MTLCompareFunction::Less,
        2 => MTLCompareFunction::Equal,
        3 => MTLCompareFunction::LessEqual,
        4 => MTLCompareFunction::Greater,
        5 => MTLCompareFunction::NotEqual,
        6 => MTLCompareFunction::GreaterEqual,
        7 => MTLCompareFunction::Always,
        _ => unreachable!("three-bit compare function"),
    };
    let clear_depth = register_value(&draw.context_registers, DB_DEPTH_CLEAR)
        .map(f32::from_bits)
        .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
        .unwrap_or(1.0);
    let texture = GuestTextureDescriptor {
        address,
        width,
        height,
        pitch,
        tile_mode,
        pixel_format: format.0,
        bytes_per_pixel: format.1,
    };
    if trace_target_sampling() {
        static LOGGED_DEPTH_TARGETS: OnceLock<Mutex<HashSet<(u64, u32, u32, u32, u32, u32)>>> =
            OnceLock::new();
        let key = (address, width, height, z_info, control, render_control);
        let mut logged = LOGGED_DEPTH_TARGETS
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if logged.len() < 256 && logged.insert(key) {
            let encoded_size = register_value(&draw.context_registers, DB_DEPTH_SIZE);
            let encoded_slice = register_value(&draw.context_registers, DB_DEPTH_SLICE);
            let bytes = guest_texture_byte_count(texture);
            eprintln!(
                "[REMu][METAL][DEPTH] address={address:#x} read={read_address:#x} write={write_address:#x} extent={width}x{height} pitch={pitch} format={:?} samples={sample_count} guest_tile={tile_mode} z_info={z_info:#010x} tile_index={depth_tile_mode_index} control={control:#010x} render_control={render_control:#010x} size_xy={size:#010x} depth_size={encoded_size:08x?} depth_slice={encoded_slice:08x?} bytes={bytes:x?}",
                format.0,
            );
        }
    }
    Ok(Some(DecodedDepthTarget {
        texture,
        compare,
        test_enabled: depth_enabled,
        write_enabled,
        clear_enabled,
        clear_depth: f64::from(clear_depth),
        sample_count,
    }))
}

pub(super) fn initial_depth_clear_value(compare: MTLCompareFunction) -> f64 {
    if matches!(
        compare,
        MTLCompareFunction::Greater | MTLCompareFunction::GreaterEqual
    ) {
        0.0
    } else {
        1.0
    }
}

pub(super) fn initial_depth_clear_for_target(depth: DecodedDepthTarget) -> f64 {
    if depth.clear_enabled || !depth.test_enabled || depth.write_enabled {
        depth.clear_depth
    } else {
        // If a depth surface is first encountered as read-only, its guest clear register can be
        // stale state from another target. Pick the comparison-neutral far value so first use
        // cannot discard the whole scene. This is also SharpEmu's proven Silent Hill behavior.
        initial_depth_clear_value(depth.compare)
    }
}

pub(super) fn graphics_pipeline_key(draw: &IndexedDraw) -> Result<GraphicsPipelineKey, String> {
    let targets = decode_render_targets(draw)?;
    graphics_pipeline_key_from_targets(draw, &targets)
}

pub(super) fn graphics_pipeline_key_from_targets(
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
    let pixel_evaluation = draw
        .pixel_shader_evaluation
        .as_ref()
        .ok_or("draw has no pixel resource evaluation")?;
    let export_program = &draw
        .export_shader
        .as_ref()
        .ok_or("draw has no decoded export shader")?
        .program;
    let pixel_program = &draw
        .pixel_shader
        .as_ref()
        .ok_or("draw has no decoded pixel shader")?
        .program;
    let sample_count = targets
        .first()
        .map(|target| target.sample_count)
        .or_else(|| depth.map(|depth| depth.sample_count))
        .unwrap_or(1);
    let render_target_array_length =
        common_render_target_array_length(targets.iter().map(|target| target.layer_count));
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
    if render_target_array_length > 1 && depth.is_some() {
        return Err(
            "layered color rendering with a depth attachment is not implemented".to_owned(),
        );
    }
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
        pixel_input_controls: pixel_input_controls(&draw.context_registers),
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
        primitive_topology_class: metal_primitive_topology_class(draw.primitive_type)?,
    })
}

pub(super) fn common_render_target_array_length(
    layer_counts: impl IntoIterator<Item = u32>,
) -> u32 {
    layer_counts
        .into_iter()
        .map(|count| count.max(1))
        .min()
        .unwrap_or(1)
}

pub(super) fn graphics_shader_resource_key(
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

pub(super) fn image_component_class(binding: &ImageBinding) -> u32 {
    let unified_format = binding
        .resource_descriptor
        .get(1)
        .map_or(0, |word| (word >> 20) & 0x1ff);
    let number_format =
        remu_shader::decode_unified_format(unified_format).map_or(0, |(_, number)| number);
    match number_format {
        4 => 1, // uint
        5 => 2, // int
        _ => 0, // float, normalized, scaled, and sRGB
    }
}

pub(super) fn decode_metal_blend_attachment(
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

pub(super) fn metal_blend_factor(value: u32) -> Result<MTLBlendFactor, String> {
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

pub(super) fn metal_blend_operation(value: u32) -> Result<MTLBlendOperation, String> {
    match value {
        0 => Ok(MTLBlendOperation::Add),
        1 => Ok(MTLBlendOperation::Subtract),
        2 => Ok(MTLBlendOperation::Min),
        3 => Ok(MTLBlendOperation::Max),
        4 => Ok(MTLBlendOperation::ReverseSubtract),
        _ => Err(format!("unsupported blend operation {value}")),
    }
}

pub(super) fn metal_color_write_mask(guest_mask: u32) -> MTLColorWriteMask {
    // GFX10 CB_TARGET_MASK is R=1,G=2,B=4,A=8; MTLColorWriteMask uses the reverse bits.
    let metal_mask = u64::from(guest_mask & 1 != 0) << 3
        | u64::from(guest_mask & 2 != 0) << 2
        | u64::from(guest_mask & 4 != 0) << 1
        | u64::from(guest_mask & 8 != 0);
    MTLColorWriteMask::from_bits_retain(metal_mask)
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
pub(super) const VIDEO_OUT_TILING_MODE_TILE: u32 = 0;

pub(super) const VIDEO_OUT_TILING_MODE_LINEAR: u32 = 1;

pub(super) const VIDEO_OUT_DISPLAY_TILE_MODE: u32 = 27;

#[cfg(test)]
pub(super) const fn raw_video_out_layout_is_linear(
    tiling_mode: u32,
    pitch: u32,
    width: u32,
) -> bool {
    tiling_mode == VIDEO_OUT_TILING_MODE_LINEAR && pitch >= width
}

pub(super) const fn raw_video_out_texture_descriptor(
    address: u64,
    display: DisplayConfiguration,
) -> Option<GuestTextureDescriptor> {
    if display.pitch < display.width {
        return None;
    }
    let tile_mode = match display.tiling_mode {
        VIDEO_OUT_TILING_MODE_TILE => VIDEO_OUT_DISPLAY_TILE_MODE,
        VIDEO_OUT_TILING_MODE_LINEAR => 0,
        _ => return None,
    };
    Some(GuestTextureDescriptor {
        address,
        width: display.width,
        height: display.height,
        pitch: display.pitch,
        tile_mode,
        pixel_format: MTLPixelFormat::BGRA8Unorm,
        bytes_per_pixel: 4,
    })
}
