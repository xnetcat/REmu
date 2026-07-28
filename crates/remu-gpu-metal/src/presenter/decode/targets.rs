// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CB_COLOR_REGISTER_STRIDE, CB_COLOR0_ATTRIB, CB_COLOR0_ATTRIB2, CB_COLOR0_ATTRIB3,
    CB_COLOR0_BASE, CB_COLOR0_BASE_EXT, CB_COLOR0_CLEAR_WORD0, CB_COLOR0_CLEAR_WORD1,
    CB_COLOR0_INFO, CB_COLOR0_VIEW, CB_TARGET_MASK, COLOR_TARGET_COUNT, DB_DEPTH_CLEAR,
    DB_DEPTH_CONTROL, DB_DEPTH_SIZE, DB_DEPTH_SIZE_XY, DB_DEPTH_SLICE, DB_RENDER_CONTROL,
    DB_Z_INFO, DB_Z_READ_BASE, DB_Z_READ_BASE_HI, DB_Z_WRITE_BASE, DB_Z_WRITE_BASE_HI,
    DecodedDepthTarget, DecodedRenderTarget, GuestTextureDescriptor, RenderTargetFormat,
    decode_scanout_target, effective_linear_pitch, guest_texture_byte_count, half_float_to_f32,
    register_value, tiled_block_dimensions, trace_target_sampling, unsigned_float_to_f32,
};
use metal::{MTLClearColor, MTLCompareFunction, MTLPixelFormat};
use remu_gpu::{GpuRegister, IndexedDraw};
use remu_shader::PixelOutputKind;
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

pub(crate) fn decode_render_target_format(
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
pub(crate) enum ClearChannelEncoding {
    Unorm,
    Snorm,
    Uint,
    Sint,
    Float,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FastClearLayout {
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

pub(crate) fn sign_extend_clear_lane(value: u32, bits: u32) -> i64 {
    debug_assert!((1..=32).contains(&bits));
    let shift = 64 - bits;
    ((u64::from(value) << shift) as i64) >> shift
}

pub(crate) fn decode_clear_lane(value: u32, bits: u32, encoding: ClearChannelEncoding) -> f64 {
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

pub(crate) fn inverse_srgb_clear_component(encoded: f64) -> f64 {
    if encoded <= 0.040_45 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

pub(crate) fn decode_fast_clear_color(
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

pub(crate) fn render_target_fast_clear_color(
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

pub(crate) fn initial_render_target_clear_color(
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
pub(crate) fn decode_render_target_state(
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

pub(crate) fn decode_render_targets(
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

#[allow(clippy::too_many_lines)]
pub(crate) fn decode_render_targets_from_registers(
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

pub(crate) fn decode_depth_target(
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

pub(crate) fn initial_depth_clear_value(compare: MTLCompareFunction) -> f64 {
    if matches!(
        compare,
        MTLCompareFunction::Greater | MTLCompareFunction::GreaterEqual
    ) {
        0.0
    } else {
        1.0
    }
}

pub(crate) fn initial_depth_clear_for_target(depth: DecodedDepthTarget) -> f64 {
    if depth.clear_enabled || !depth.test_enabled || depth.write_enabled {
        depth.clear_depth
    } else {
        // If a depth surface is first encountered as read-only, its guest clear register can be
        // stale state from another target. Pick the comparison-neutral far value so first use
        // cannot discard the whole scene. This is also SharpEmu's proven Silent Hill behavior.
        initial_depth_clear_value(depth.compare)
    }
}

pub(crate) fn common_render_target_array_length(
    layer_counts: impl IntoIterator<Item = u32>,
) -> u32 {
    layer_counts
        .into_iter()
        .map(|count| count.max(1))
        .min()
        .unwrap_or(1)
}
