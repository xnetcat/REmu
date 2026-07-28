// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{GuestTextureDescriptor, decode_render_target_format};
use metal::MTLPixelFormat;
use std::sync::OnceLock;

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(super) fn linear_unorm8_to_srgb_unorm8(linear: u8) -> u8 {
    static LUT: OnceLock<[u8; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        std::array::from_fn(|linear| {
            let linear =
                f64::from(u8::try_from(linear).expect("linear table index fits u8")) / 255.0;
            let encoded = if linear <= 0.003_130_8 {
                linear * 12.92
            } else {
                1.055 * linear.powf(1.0 / 2.4) - 0.055
            };
            // Both transfer-function branches map the unorm input into 0..=1, so rounding the
            // scaled result cannot produce a negative value or exceed u8.
            (encoded * 255.0).round() as u8
        })
    })[usize::from(linear)]
}

pub(super) fn half_float_to_srgb_unorm8(bits: u16) -> u8 {
    linear_unorm8_to_srgb_unorm8(half_float_to_unorm8(bits))
}

pub(super) fn unsigned_float_to_f32(bits: u32, mantissa_bits: u32) -> f32 {
    debug_assert!(matches!(mantissa_bits, 5 | 6));
    let mantissa_mask = (1_u32 << mantissa_bits) - 1;
    let exponent = (bits >> mantissa_bits) & 0x1f;
    let mantissa = bits & mantissa_mask;
    let float_bits = match exponent {
        0 if mantissa == 0 => 0,
        0 => {
            let leading_bit = 31 - mantissa.leading_zeros();
            let float_exponent = 113 + leading_bit - mantissa_bits;
            let float_mantissa = (mantissa ^ (1 << leading_bit)) << (23 - leading_bit);
            (float_exponent << 23) | float_mantissa
        }
        0x1f => 0x7f80_0000 | (mantissa << (23 - mantissa_bits)),
        _ => ((exponent + 112) << 23) | (mantissa << (23 - mantissa_bits)),
    };
    f32::from_bits(float_bits)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(super) fn float_to_unorm8(value: f32) -> u8 {
    if value.is_nan() || value <= 0.0 {
        0
    } else if value >= 1.0 {
        255
    } else {
        (value * 255.0).round() as u8
    }
}

pub(super) fn unsigned_float_to_srgb_unorm8(bits: u32, mantissa_bits: u32) -> u8 {
    linear_unorm8_to_srgb_unorm8(float_to_unorm8(unsigned_float_to_f32(bits, mantissa_bits)))
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(super) fn half_float_to_unorm8(bits: u16) -> u8 {
    float_to_unorm8(half_float_to_f32(bits))
}

pub(super) fn half_float_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15);
    let exponent = u32::from((bits >> 10) & 0x1f);
    let significand = u32::from(bits & 0x03ff);
    let float_bits = match exponent {
        0 if significand == 0 => sign << 31,
        0 => {
            let leading_zeros = significand.leading_zeros() - 21;
            let normalized = (significand << leading_zeros) & 0x03ff;
            let exponent = 113_u32.saturating_sub(leading_zeros);
            (sign << 31) | (exponent << 23) | (normalized << 13)
        }
        0x1f => (sign << 31) | 0x7f80_0000 | (significand << 13),
        _ => (sign << 31) | ((exponent + 112) << 23) | (significand << 13),
    };
    f32::from_bits(float_bits)
}

pub(super) fn decode_sampled_texture_format(
    data_format: u32,
    number_format: u32,
) -> Result<(MTLPixelFormat, u32), String> {
    match (data_format, number_format) {
        (10, 9) => Ok((MTLPixelFormat::RGBA8Unorm_sRGB, 4)),
        (169, _) => Ok((MTLPixelFormat::BC1_RGBA, 8)),
        (170, _) => Ok((MTLPixelFormat::BC1_RGBA_sRGB, 8)),
        (171, _) => Ok((MTLPixelFormat::BC2_RGBA, 16)),
        (172, _) => Ok((MTLPixelFormat::BC2_RGBA_sRGB, 16)),
        (173, _) => Ok((MTLPixelFormat::BC3_RGBA, 16)),
        (174, _) => Ok((MTLPixelFormat::BC3_RGBA_sRGB, 16)),
        (175, _) => Ok((MTLPixelFormat::BC4_RUnorm, 8)),
        (176, _) => Ok((MTLPixelFormat::BC4_RSnorm, 8)),
        (177, _) => Ok((MTLPixelFormat::BC5_RGUnorm, 16)),
        (178, _) => Ok((MTLPixelFormat::BC5_RGSnorm, 16)),
        (179, _) => Ok((MTLPixelFormat::BC6H_RGBUfloat, 16)),
        (180, _) => Ok((MTLPixelFormat::BC6H_RGBFloat, 16)),
        (181, _) => Ok((MTLPixelFormat::BC7_RGBAUnorm, 16)),
        (182, _) => Ok((MTLPixelFormat::BC7_RGBAUnorm_sRGB, 16)),
        _ => decode_render_target_format(data_format, number_format, 0)
            .map(|format| (format.pixel_format, format.bytes_per_pixel)),
    }
}

pub(super) fn compressed_texture_block_extent(format: MTLPixelFormat) -> Option<(usize, usize)> {
    match format {
        MTLPixelFormat::BC1_RGBA
        | MTLPixelFormat::BC1_RGBA_sRGB
        | MTLPixelFormat::BC2_RGBA
        | MTLPixelFormat::BC2_RGBA_sRGB
        | MTLPixelFormat::BC3_RGBA
        | MTLPixelFormat::BC3_RGBA_sRGB
        | MTLPixelFormat::BC4_RUnorm
        | MTLPixelFormat::BC4_RSnorm
        | MTLPixelFormat::BC5_RGUnorm
        | MTLPixelFormat::BC5_RGSnorm
        | MTLPixelFormat::BC6H_RGBFloat
        | MTLPixelFormat::BC6H_RGBUfloat
        | MTLPixelFormat::BC7_RGBAUnorm
        | MTLPixelFormat::BC7_RGBAUnorm_sRGB => Some((4, 4)),
        _ => None,
    }
}

pub(super) fn texture_element_layout(
    descriptor: GuestTextureDescriptor,
) -> Result<(usize, usize, usize, usize), String> {
    let width =
        usize::try_from(descriptor.width).map_err(|_| "texture width does not fit usize")?;
    let height =
        usize::try_from(descriptor.height).map_err(|_| "texture height does not fit usize")?;
    let pitch =
        usize::try_from(descriptor.pitch).map_err(|_| "texture pitch does not fit usize")?;
    let bytes_per_element = usize::try_from(descriptor.bytes_per_pixel)
        .map_err(|_| "texture element byte count does not fit usize")?;
    if let Some((block_width, block_height)) =
        compressed_texture_block_extent(descriptor.pixel_format)
    {
        Ok((
            width.div_ceil(block_width),
            height.div_ceil(block_height),
            pitch.div_ceil(block_width),
            bytes_per_element,
        ))
    } else {
        Ok((width, height, pitch, bytes_per_element))
    }
}
