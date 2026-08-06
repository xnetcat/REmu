// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Phase 1G production encode helpers: guest-backed MTLBuffer ↔ texture via GPU tiler.
//!
//! These stay on the open guest command buffer. Encoding never snapshots guest RAM on the
//! CPU, never commits a private readback buffer, and never invents zeros when backing is
//! missing — missing shared identity is a loud refusal.

use super::super::cache::metal_buffer_with_guest_backing_range;
use super::super::tiling::{
    guest_texture_byte_count, tiled_block_bytes, tiled_block_dimensions,
};
use super::super::{GuestTextureDescriptor, ResourceSpan, texture_element_layout};
use super::msl::{mode27_tiler_pipelines, mode5_tiler_pipelines};
use metal::{
    Buffer, CommandBufferRef, ComputePipelineState, Device, MTLBlitOption, MTLOrigin,
    MTLResourceOptions, MTLSize, Texture,
};
use remu_memory::guest_shared_backing_range;
use std::ffi::c_void;

fn align_256(value: u64) -> Option<u64> {
    value.checked_add(255).map(|value| value & !255)
}

/// True when Phase 1G GPU-timeline tiling owns this descriptor's upload/publish path.
pub(crate) fn gpu_tiler_owns_descriptor(descriptor: GuestTextureDescriptor) -> bool {
    matches!(descriptor.tile_mode, 0 | 5 | 24 | 27)
        && super::super::metal_uncompressed_bytes_per_pixel(descriptor.pixel_format)
            == Some(descriptor.bytes_per_pixel)
}

#[derive(Clone, Copy)]
struct DetileSubresource {
    width: u64,
    height: u64,
    linear_offset: u64,
    linear_row_bytes: u64,
    linear_image_bytes: u64,
    level_offset: u64,
    pitch: u64,
    bytes_per_element: u64,
    block_width: u64,
    block_height: u64,
    surface_z_xor: u64,
    tile_mode: u64,
    metal_slice: u64,
    metal_level: u64,
    metal_z: u64,
}

fn single_level_subresource(
    descriptor: GuestTextureDescriptor,
) -> Result<(DetileSubresource, u64, u64), String> {
    let bytes_per_element = u64::from(descriptor.bytes_per_pixel);
    let face_bytes = guest_texture_byte_count(descriptor)
        .ok_or("GPU tiler face size overflows")?;
    let (block_width, block_height) = if descriptor.tile_mode == 0 {
        (1_u64, 1_u64)
    } else {
        let block_bytes = tiled_block_bytes(descriptor.tile_mode)
            .ok_or("GPU tiler block size unavailable for tile mode")?;
        let (w, h) = tiled_block_dimensions(
            block_bytes,
            usize::try_from(bytes_per_element).map_err(|_| "element size does not fit usize")?,
        )
        .ok_or("GPU tiler block dimensions unavailable")?;
        (
            u64::try_from(w).map_err(|_| "block width does not fit u64")?,
            u64::try_from(h).map_err(|_| "block height does not fit u64")?,
        )
    };
    let packed_row = u64::from(descriptor.width)
        .checked_mul(bytes_per_element)
        .ok_or("GPU tiler packed row overflows")?;
    let linear_row_bytes = align_256(packed_row).ok_or("GPU tiler row alignment overflows")?;
    let linear_image_bytes = linear_row_bytes
        .checked_mul(u64::from(descriptor.height))
        .ok_or("GPU tiler image size overflows")?;
    Ok((
        DetileSubresource {
            width: u64::from(descriptor.width),
            height: u64::from(descriptor.height),
            linear_offset: 0,
            linear_row_bytes,
            linear_image_bytes,
            level_offset: 0,
            pitch: u64::from(descriptor.pitch),
            bytes_per_element,
            block_width,
            block_height,
            surface_z_xor: 0,
            tile_mode: u64::from(descriptor.tile_mode),
            metal_slice: 0,
            metal_level: 0,
            metal_z: 0,
        },
        face_bytes,
        linear_image_bytes,
    ))
}

fn mode27_parameters(
    sub: DetileSubresource,
    resource_base: u64,
    target_base: u64,
    target_length: u64,
    run_count: usize,
) -> Result<[u64; 18], String> {
    Ok([
        sub.width,
        sub.height,
        sub.linear_row_bytes,
        sub.linear_offset,
        resource_base,
        target_base,
        sub.level_offset,
        sub.pitch,
        sub.bytes_per_element,
        sub.block_width,
        sub.block_height,
        0, // tail_x
        0, // tail_y
        sub.surface_z_xor,
        0, // in_tail
        target_length,
        u64::try_from(run_count).map_err(|_| "GPU tiler run count does not fit u64")?,
        sub.tile_mode,
    ])
}

fn mode5_parameters(
    sub: DetileSubresource,
    resource_base: u64,
    target_base: u64,
    target_length: u64,
    run_count: usize,
) -> Result<[u64; 16], String> {
    Ok([
        sub.width,
        sub.height,
        sub.linear_row_bytes,
        sub.linear_offset,
        resource_base,
        target_base,
        sub.level_offset,
        sub.pitch,
        sub.bytes_per_element,
        sub.block_width,
        sub.block_height,
        0, // tail_x
        0, // tail_y
        0, // in_tail
        target_length,
        u64::try_from(run_count).map_err(|_| "GPU tiler run count does not fit u64")?,
    ])
}

fn guest_buffer_for_descriptor(
    device: &Device,
    descriptor: GuestTextureDescriptor,
    byte_count: u64,
) -> Result<(Buffer, u64), String> {
    let backing_range = guest_shared_backing_range(descriptor.address, byte_count).ok_or_else(
        || {
            format!(
                "Phase 1G: guest texture {:#x}+{byte_count:#x} has no committed/shared backing for Metal (tile={})",
                descriptor.address, descriptor.tile_mode
            )
        },
    )?;
    let buffer = metal_buffer_with_guest_backing_range(device, &backing_range).ok_or_else(|| {
        format!(
            "Phase 1G: guest texture {:#x} backing cannot be exposed as MTLBuffer",
            descriptor.address
        )
    })?;
    Ok((buffer, backing_range.resource_offset()))
}

/// Encodes guest tiled/linear bytes → `destination` texture on `command_buffer`.
///
/// Mode 0 blit-copies rows from the guest buffer. Modes 5/24/27 detile through the S2
/// compute kernels. Never snapshots guest RAM on the CPU.
pub(crate) fn encode_gpu_upload_from_guest(
    device: &Device,
    command_buffer: &CommandBufferRef,
    descriptor: GuestTextureDescriptor,
    destination: &Texture,
) -> Result<(), String> {
    let _ = texture_element_layout(descriptor)?;
    if !gpu_tiler_owns_descriptor(descriptor) {
        return Err(format!(
            "Phase 1G: GPU upload refuses tile={} format={:?} bpe={} (no tiler ownership)",
            descriptor.tile_mode, descriptor.pixel_format, descriptor.bytes_per_pixel
        ));
    }
    let (sub, face_bytes, linear_length) = single_level_subresource(descriptor)?;
    let (guest, guest_offset) = guest_buffer_for_descriptor(device, descriptor, face_bytes)?;
    let span = ResourceSpan::for_texture(descriptor)
        .ok_or("Phase 1G: guest texture span overflows")?;
    // RangeSet / tiler kernels share ResourceSpan::range_start space (backing offset when
    // resolved, else guest VA) — never mix guest VAs into debt/run comparisons.
    let resource_base = span.range_start();
    let target_base = span.range_start();

    if descriptor.tile_mode == 0 {
        let guest_row_bytes = u64::from(descriptor.pitch)
            .checked_mul(u64::from(descriptor.bytes_per_pixel))
            .ok_or("Phase 1G linear guest row bytes overflow")?;
        let guest_image_bytes = guest_row_bytes
            .checked_mul(u64::from(descriptor.height))
            .ok_or("Phase 1G linear guest image bytes overflow")?;
        let blit = command_buffer.new_blit_command_encoder();
        blit.set_label("REMu Phase 1G linear guest→texture upload");
        blit.copy_from_buffer_to_texture(
            &guest,
            guest_offset,
            guest_row_bytes,
            guest_image_bytes,
            MTLSize::new(sub.width, sub.height, 1),
            destination,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLBlitOption::None,
        );
        blit.end_encoding();
        return Ok(());
    }

    let linear = device.new_buffer(linear_length, MTLResourceOptions::StorageModePrivate);
    linear.set_label("REMu Phase 1G detile linear");
    match descriptor.tile_mode {
        5 => {
            let (detile, _) = mode5_tiler_pipelines(device)?;
            let parameters =
                mode5_parameters(sub, resource_base, target_base, face_bytes, 0)?;
            encode_detile_dispatch(
                device,
                command_buffer,
                &detile,
                &guest,
                guest_offset,
                face_bytes,
                &linear,
                &parameters,
                sub,
            );
        }
        24 | 27 => {
            let (detile, _) = mode27_tiler_pipelines(device)?;
            let parameters =
                mode27_parameters(sub, resource_base, target_base, face_bytes, 0)?;
            encode_detile_dispatch(
                device,
                command_buffer,
                &detile,
                &guest,
                guest_offset,
                face_bytes,
                &linear,
                &parameters,
                sub,
            );
        }
        _ => unreachable!("tile mode gated by gpu_tiler_owns_descriptor"),
    }

    let upload = command_buffer.new_blit_command_encoder();
    upload.set_label("REMu Phase 1G detiled linear→texture");
    upload.copy_from_buffer_to_texture(
        &linear,
        sub.linear_offset,
        sub.linear_row_bytes,
        sub.linear_image_bytes,
        MTLSize::new(sub.width, sub.height, 1),
        destination,
        sub.metal_slice,
        sub.metal_level,
        MTLOrigin {
            x: 0,
            y: 0,
            z: sub.metal_z,
        },
        MTLBlitOption::None,
    );
    upload.end_encoding();
    Ok(())
}

fn encode_detile_dispatch(
    device: &Device,
    command_buffer: &CommandBufferRef,
    detile: &ComputePipelineState,
    guest: &Buffer,
    guest_offset: u64,
    face_bytes: u64,
    linear: &Buffer,
    parameters: &[u64],
    sub: DetileSubresource,
) {
    // Tiled address math indexes `physical - target_base` into buffer 0, so the bound
    // buffer must start at the resource base. Page-aligned views with a nonzero resource
    // offset are copied into a private scratch first.
    let tiled_source = if guest_offset == 0 {
        guest.clone()
    } else {
        let scratch = device.new_buffer(face_bytes, MTLResourceOptions::StorageModePrivate);
        let blit = command_buffer.new_blit_command_encoder();
        blit.set_label("REMu Phase 1G align guest tiled view");
        blit.copy_from_buffer(guest, guest_offset, &scratch, 0, face_bytes);
        blit.end_encoding();
        scratch
    };

    let compute = command_buffer.new_compute_command_encoder();
    compute.set_label("REMu Phase 1G GPU detile");
    compute.set_compute_pipeline_state(detile);
    compute.set_buffer(0, Some(&tiled_source), 0);
    compute.set_buffer(1, Some(linear), 0);
    compute.set_bytes(
        3,
        u64::try_from(std::mem::size_of_val(parameters)).expect("parameter byte count fits u64"),
        parameters.as_ptr().cast::<c_void>(),
    );
    compute.dispatch_threads(
        MTLSize::new(sub.width, sub.height, 1),
        MTLSize::new(16, 16, 1),
    );
    compute.end_encoding();
}

/// Encodes `source` texture → guest tiled/linear bytes for `authored` absolute ranges.
///
/// `authored` is an ephemeral run list for this encode only (not persistent debt tracking).
/// An empty run list is a no-op success.
pub(crate) fn encode_gpu_publish_to_guest(
    device: &Device,
    command_buffer: &CommandBufferRef,
    descriptor: GuestTextureDescriptor,
    source: &Texture,
    authored: &[(u64, u64)],
) -> Result<(), String> {
    if authored.is_empty() {
        return Ok(());
    }
    if !gpu_tiler_owns_descriptor(descriptor) {
        return Err(format!(
            "Phase 1G: GPU publish refuses tile={} format={:?} bpe={}",
            descriptor.tile_mode, descriptor.pixel_format, descriptor.bytes_per_pixel
        ));
    }
    let (sub, face_bytes, linear_length) = single_level_subresource(descriptor)?;
    let (guest, guest_offset) = guest_buffer_for_descriptor(device, descriptor, face_bytes)?;
    let span = ResourceSpan::for_texture(descriptor)
        .ok_or("Phase 1G: guest texture span overflows")?;
    let resource_base = span.range_start();
    let target_base = span.range_start();
    let linear = device.new_buffer(linear_length, MTLResourceOptions::StorageModePrivate);
    linear.set_label("REMu Phase 1G retile linear");

    let blit = command_buffer.new_blit_command_encoder();
    blit.set_label("REMu Phase 1G texture→linear for guest publish");
    blit.copy_from_texture_to_buffer(
        source,
        sub.metal_slice,
        sub.metal_level,
        MTLOrigin {
            x: 0,
            y: 0,
            z: sub.metal_z,
        },
        MTLSize::new(sub.width, sub.height, 1),
        &linear,
        sub.linear_offset,
        sub.linear_row_bytes,
        sub.linear_image_bytes,
        MTLBlitOption::None,
    );
    blit.end_encoding();

    if descriptor.tile_mode == 0 {
        // Publish only authored byte runs into the guest linear buffer.
        for &(start, end) in authored {
            if end <= start || start < target_base {
                continue;
            }
            let offset = start - target_base;
            let len = end - start;
            if offset.saturating_add(len) > face_bytes {
                continue;
            }
            let copy = command_buffer.new_blit_command_encoder();
            copy.set_label("REMu Phase 1G linear guest publish run");
            copy.copy_from_buffer(&linear, offset, &guest, guest_offset + offset, len);
            copy.end_encoding();
        }
        return Ok(());
    }

    let mut run_words = Vec::with_capacity(authored.len() * 2);
    for &(start, end) in authored {
        run_words.extend_from_slice(&[start, end]);
    }
    let run_bytes = u64::try_from(run_words.len())
        .ok()
        .and_then(|words| words.checked_mul(8))
        .ok_or("GPU publish run buffer size overflows")?;
    let runs = device.new_buffer_with_data(
        run_words.as_ptr().cast::<c_void>(),
        run_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    runs.set_label("REMu Phase 1G publish authored runs");

    let tiled_dest = if guest_offset == 0 {
        guest.clone()
    } else {
        // Retile into a private scratch then blit the authored runs into the offset view.
        let scratch = device.new_buffer(face_bytes, MTLResourceOptions::StorageModePrivate);
        scratch.set_label("REMu Phase 1G retile scratch");
        let init = command_buffer.new_blit_command_encoder();
        init.set_label("REMu Phase 1G seed retile scratch from guest");
        init.copy_from_buffer(&guest, guest_offset, &scratch, 0, face_bytes);
        init.end_encoding();
        scratch
    };

    match descriptor.tile_mode {
        5 => {
            let (_, retile) = mode5_tiler_pipelines(device)?;
            let parameters =
                mode5_parameters(sub, resource_base, target_base, face_bytes, authored.len())?;
            let compute = command_buffer.new_compute_command_encoder();
            compute.set_label("REMu Phase 1G GPU retile publish");
            compute.set_compute_pipeline_state(&retile);
            compute.set_buffer(0, Some(&linear), 0);
            compute.set_buffer(1, Some(&tiled_dest), 0);
            compute.set_buffer(2, Some(&runs), 0);
            compute.set_bytes(
                3,
                u64::try_from(std::mem::size_of_val(&parameters))
                    .expect("parameter byte count fits u64"),
                parameters.as_ptr().cast::<c_void>(),
            );
            compute.dispatch_threads(
                MTLSize::new(sub.width, sub.height, 1),
                MTLSize::new(16, 16, 1),
            );
            compute.end_encoding();
        }
        24 | 27 => {
            let (_, retile) = mode27_tiler_pipelines(device)?;
            let parameters =
                mode27_parameters(sub, resource_base, target_base, face_bytes, authored.len())?;
            let compute = command_buffer.new_compute_command_encoder();
            compute.set_label("REMu Phase 1G GPU retile publish");
            compute.set_compute_pipeline_state(&retile);
            compute.set_buffer(0, Some(&linear), 0);
            compute.set_buffer(1, Some(&tiled_dest), 0);
            compute.set_buffer(2, Some(&runs), 0);
            compute.set_bytes(
                3,
                u64::try_from(std::mem::size_of_val(&parameters))
                    .expect("parameter byte count fits u64"),
                parameters.as_ptr().cast::<c_void>(),
            );
            compute.dispatch_threads(
                MTLSize::new(sub.width, sub.height, 1),
                MTLSize::new(16, 16, 1),
            );
            compute.end_encoding();
        }
        _ => unreachable!("tile mode gated by gpu_tiler_owns_descriptor"),
    }

    if guest_offset != 0 {
        // Copy authored runs from the private scratch back into the offset guest view.
        for &(start, end) in authored {
            if end <= start || start < target_base {
                continue;
            }
            let offset = start - target_base;
            let len = end - start;
            if offset.saturating_add(len) > face_bytes {
                continue;
            }
            let copy = command_buffer.new_blit_command_encoder();
            copy.set_label("REMu Phase 1G retile scratch→guest");
            copy.copy_from_buffer(&tiled_dest, offset, &guest, guest_offset + offset, len);
            copy.end_encoding();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::MTLPixelFormat;

    fn descriptor(tile_mode: u32) -> GuestTextureDescriptor {
        GuestTextureDescriptor {
            address: 0x1000_0000,
            width: 64,
            height: 64,
            pitch: 64,
            tile_mode,
            pixel_format: MTLPixelFormat::BGRA8Unorm,
            bytes_per_pixel: 4,
        }
    }

    #[test]
    fn tiler_owns_linear_and_mode27_uncompressed() {
        assert!(gpu_tiler_owns_descriptor(descriptor(0)));
        assert!(gpu_tiler_owns_descriptor(descriptor(27)));
        assert!(gpu_tiler_owns_descriptor(descriptor(24)));
        assert!(gpu_tiler_owns_descriptor(descriptor(5)));
    }
}
