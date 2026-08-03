// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    DecodedImageDescriptor, GuestTextureDescriptor, ResourceSpan, decode_sampled_texture_format,
    effective_linear_pitch, is_storage_image_opcode, trace_image_descriptors,
};
use metal::MTLPixelFormat;
use remu_shader::ImageBinding;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) fn decode_image_descriptor(words: &[u32]) -> Result<DecodedImageDescriptor, String> {
    decode_image_descriptor_impl(words, true)
}

/// Decodes an image SRD without emitting optional descriptor diagnostics.
///
/// Shared-event draw preflight must be observational: it may classify immutable packet state but
/// cannot log or mutate a process-global diagnostic cache before the producer has host-retired.
pub(crate) fn decode_image_descriptor_without_diagnostics(
    words: &[u32],
) -> Result<DecodedImageDescriptor, String> {
    decode_image_descriptor_impl(words, false)
}

fn decode_image_descriptor_impl(
    words: &[u32],
    emit_diagnostics: bool,
) -> Result<DecodedImageDescriptor, String> {
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
    if emit_diagnostics
        && trace_image_descriptors()
        && (tile_mode == 9 || (width >= 700 && height >= 500))
    {
        // K1: lock-free bounded sampler (was a mutex-guarded HashSet dedup).
        static LOGGED_IMAGES: AtomicUsize = AtomicUsize::new(0);
        if LOGGED_IMAGES.fetch_add(1, Ordering::Relaxed) < 128 {
            eprintln!(
                "[REMu][METAL][IMAGE] mode={tile_mode} address={address:#x} extent={width}x{height} pitch={descriptor_pitch} format={data_format}/{number_format} unified={unified_format} type={texture_type} mips={base_level}..={last_level}"
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
    let guest = GuestTextureDescriptor {
        address,
        width,
        height,
        pitch,
        tile_mode,
        pixel_format,
        bytes_per_pixel: bytes_per_element,
    };
    Ok(DecodedImageDescriptor {
        guest,
        // The one place a bound image's physical span is resolved. Every later overlap query
        // against this binding reuses it instead of re-entering the shared-mapping registry.
        span: ResourceSpan::for_texture(guest),
        destination_select,
        base_mip_level: base_level,
        mip_level_count: last_level - base_level + 1,
        texture_type,
        layer_count,
    })
}

pub(crate) fn sampled_alias_descriptor(binding: &ImageBinding) -> Option<DecodedImageDescriptor> {
    decode_image_descriptor(&binding.resource_descriptor).ok()
}

pub(crate) fn fallback_image_descriptor(
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
    // Metal dimension from the instruction itself, while storage instructions preserve known
    // descriptor 3D and array types even when the remaining descriptor fields are invalid.
    let texture_type = if storage {
        match encoded_texture_type {
            10 => 10,
            11 | 12 | 13 | 15 => 13,
            _ => 9,
        }
    } else {
        match binding.control.dimension {
            2 => 10,
            3 => 11,
            _ => 9,
        }
    };
    let layer_count = if texture_type == 11 { 6 } else { 1 };
    let guest = GuestTextureDescriptor {
        address: 0,
        width: 1,
        height: 1,
        pitch: 1,
        tile_mode: 0,
        pixel_format,
        bytes_per_pixel,
    };
    Ok(DecodedImageDescriptor {
        guest,
        span: ResourceSpan::for_texture(guest),
        destination_select: 0,
        base_mip_level: 0,
        mip_level_count: 1,
        texture_type,
        layer_count,
    })
}

pub(crate) fn image_component_class(binding: &ImageBinding) -> u32 {
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
