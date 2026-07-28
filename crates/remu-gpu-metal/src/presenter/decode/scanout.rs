// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    DecodedRenderTarget, DisplayConfiguration, GuestTextureDescriptor, decode_render_target_state,
};
use metal::MTLPixelFormat;
use remu_gpu::GpuScanoutTarget;
use remu_shader::PixelOutputKind;

pub(crate) fn decode_scanout_target(
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

pub(crate) fn decode_video_out_pixel_format(
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

pub(crate) const VIDEO_OUT_TILING_MODE_TILE: u32 = 0;

pub(crate) const VIDEO_OUT_TILING_MODE_LINEAR: u32 = 1;

pub(crate) const VIDEO_OUT_DISPLAY_TILE_MODE: u32 = 27;

#[cfg(test)]
pub(crate) const fn raw_video_out_layout_is_linear(
    tiling_mode: u32,
    pitch: u32,
    width: u32,
) -> bool {
    tiling_mode == VIDEO_OUT_TILING_MODE_LINEAR && pitch >= width
}

pub(crate) const fn raw_video_out_texture_descriptor(
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
