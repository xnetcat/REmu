// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Phase 1G production encode helpers: guest-backed MTLBuffer ↔ texture via GPU tiler.
//!
//! These stay on the open guest command buffer. Encoding never snapshots guest RAM on the
//! CPU, never commits a private readback buffer, and never invents zeros when backing is
//! missing — missing shared identity is a loud refusal.

use super::super::cache::metal_buffer_with_guest_backing_range;
use super::super::tiling::{
    guest_texture_byte_count, mode27_surface_z_xor, mode5_mip_chain_layout,
    mode27_mip_chain_layout, tiled_block_bytes, tiled_block_dimensions,
};
use super::super::{
    GuestTextureDescriptor, ResourceSpan, compressed_texture_block_extent,
    metal_uncompressed_bytes_per_pixel, texture_element_layout,
};
use super::msl::{mode27_tiler_pipelines, mode5_tiler_pipelines, mode9_tiler_pipelines};
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
    if metal_uncompressed_bytes_per_pixel(descriptor.pixel_format)
        == Some(descriptor.bytes_per_pixel)
    {
        return matches!(descriptor.tile_mode, 0 | 1 | 5 | 9 | 24 | 27);
    }
    // Mode-1/5/9 BC family: the CPU tiler already detiles the element grid; Silent Hill's
    // VideoOut compositor samples these, so the GPU path must own them too.
    matches!(descriptor.tile_mode, 1 | 5 | 9)
        && compressed_texture_block_extent(descriptor.pixel_format).is_some()
        && matches!(descriptor.bytes_per_pixel, 8 | 16)
}

/// View shape for GPU upload/publish beyond a single-level 2D face.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuTilerView {
    pub texture_type: u32,
    pub layer_count: u32,
    pub mip_level_count: u32,
}

impl GpuTilerView {
    pub(crate) const SINGLE_LEVEL_2D: Self = Self {
        texture_type: 9,
        layer_count: 1,
        mip_level_count: 1,
    };
}

#[derive(Clone, Copy)]
struct DetileSubresource {
    /// Detile dispatch grid in tiling elements (texels for uncompressed, BC blocks for BC).
    width: u64,
    height: u64,
    /// Metal blit extent in pixels (equals width/height except for block-compressed formats).
    metal_width: u64,
    metal_height: u64,
    linear_offset: u64,
    linear_row_bytes: u64,
    linear_image_bytes: u64,
    level_offset: u64,
    pitch: u64,
    bytes_per_element: u64,
    block_width: u64,
    block_height: u64,
    tail_x: u64,
    tail_y: u64,
    surface_z_xor: u64,
    in_tail: u64,
    tile_mode: u64,
    metal_slice: u64,
    metal_level: u64,
    metal_z: u64,
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
        sub.tail_x,
        sub.tail_y,
        sub.surface_z_xor,
        sub.in_tail,
        target_length,
        u64::try_from(run_count).map_err(|_| "GPU tiler run count does not fit u64")?,
        sub.tile_mode,
    ])
}

fn mode9_parameters(
    sub: DetileSubresource,
    resource_base: u64,
    target_base: u64,
    target_length: u64,
    run_count: usize,
) -> Result<[u64; 17], String> {
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
        sub.tail_x,
        sub.tail_y,
        sub.in_tail,
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
        sub.tail_x,
        sub.tail_y,
        sub.in_tail,
        target_length,
        u64::try_from(run_count).map_err(|_| "GPU tiler run count does not fit u64")?,
    ])
}

fn guest_buffer_for_range(
    device: &Device,
    address: u64,
    byte_count: u64,
    tile_mode: u32,
) -> Result<(Buffer, u64), String> {
    let backing_range = guest_shared_backing_range(address, byte_count).ok_or_else(|| {
        format!(
            "Phase 1G: guest texture {address:#x}+{byte_count:#x} has no committed/shared backing for Metal (tile={tile_mode})"
        )
    })?;
    let buffer = metal_buffer_with_guest_backing_range(device, &backing_range).ok_or_else(|| {
        format!("Phase 1G: guest texture {address:#x} backing cannot be exposed as MTLBuffer")
    })?;
    Ok((buffer, backing_range.resource_offset()))
}

fn plan_upload_subresources(
    descriptor: GuestTextureDescriptor,
    view: GpuTilerView,
) -> Result<(Vec<DetileSubresource>, u64, u64), String> {
    if view.layer_count == 0 || view.mip_level_count == 0 {
        return Err("Phase 1G GPU tiler view requires nonempty layers and mips".to_owned());
    }
    if !matches!(view.texture_type, 9 | 10 | 11 | 13) {
        return Err(format!(
            "Phase 1G GPU tiler cannot represent texture type {}",
            view.texture_type
        ));
    }
    if view.mip_level_count > 1
        && descriptor.tile_mode == 0
        && (view.texture_type != 9 || view.layer_count != 1)
    {
        return Err(
            "Phase 1G linear GPU upload synthesizes mips only for single-layer 2D base uploads"
                .to_owned(),
        );
    }
    if view.mip_level_count > 1
        && matches!(descriptor.tile_mode, 5 | 24 | 27)
        && !(matches!(descriptor.tile_mode, 27) && view.texture_type == 9 && view.layer_count == 1
            || matches!(descriptor.tile_mode, 5)
                && matches!(view.texture_type, 9 | 11 | 13)
                && view.layer_count >= 1)
    {
        return Err(format!(
            "Phase 1G GPU tiler cannot represent mipmapped type={} layers={} tile={}",
            view.texture_type, view.layer_count, descriptor.tile_mode
        ));
    }
    if descriptor.tile_mode == 24 && view.mip_level_count > 1 {
        return Err("Phase 1G GPU tiler refuses mode-24 mip chains (no SDK mip layout)".to_owned());
    }

    let (elements_wide, elements_high, elements_pitch, layout_bpe) =
        texture_element_layout(descriptor)?;
    let bytes_per_element =
        u64::try_from(layout_bpe).map_err(|_| "GPU tiler element size does not fit u64")?;
    let compressed_block = compressed_texture_block_extent(descriptor.pixel_format);
    if compressed_block.is_some() && !matches!(descriptor.tile_mode, 1 | 5 | 9) {
        return Err(format!(
            "Phase 1G GPU tiler refuses compressed format {:?} on tile={}",
            descriptor.pixel_format, descriptor.tile_mode
        ));
    }
    // Mode-5 already has AddrLib-backed mip-chain layout in element space (texels or BC
    // blocks). Mode-1/9 stay single-level until their mip layouts are wired; mode-27 BC
    // remains outside ownership entirely.
    if compressed_block.is_some() && view.mip_level_count > 1 && descriptor.tile_mode != 5 {
        return Err(format!(
            "Phase 1G GPU tiler refuses compressed mip chains on tile={} (mode-5 element-grid mips only)",
            descriptor.tile_mode
        ));
    }
    let mut linear_offset = 0_u64;
    let mut subs = Vec::new();

    let mut push = |width: u64,
                    height: u64,
                    metal_width: u64,
                    metal_height: u64,
                    level_offset: u64,
                    pitch: u64,
                    block_width: u64,
                    block_height: u64,
                    tail_x: u64,
                    tail_y: u64,
                    in_tail: bool,
                    surface_z: u32,
                    metal_level: u64|
     -> Result<(), String> {
        let packed_row = width
            .checked_mul(bytes_per_element)
            .ok_or("GPU tiler packed row overflows")?;
        let linear_row_bytes = align_256(packed_row).ok_or("GPU tiler row alignment overflows")?;
        let linear_image_bytes = linear_row_bytes
            .checked_mul(height)
            .ok_or("GPU tiler image size overflows")?;
        linear_offset = align_256(linear_offset).ok_or("GPU tiler offset overflows")?;
        let metal_slice = if matches!(view.texture_type, 11 | 13) {
            u64::from(surface_z)
        } else {
            0
        };
        let metal_z = if view.texture_type == 10 {
            u64::from(surface_z)
        } else {
            0
        };
        subs.push(DetileSubresource {
            width,
            height,
            metal_width,
            metal_height,
            linear_offset,
            linear_row_bytes,
            linear_image_bytes,
            level_offset,
            pitch,
            bytes_per_element,
            block_width,
            block_height,
            tail_x,
            tail_y,
            surface_z_xor: u64::try_from(mode27_surface_z_xor(
                usize::try_from(surface_z).map_err(|_| "surface index does not fit usize")?,
            ))
            .map_err(|_| "surface xor does not fit u64")?,
            in_tail: u64::from(in_tail),
            tile_mode: u64::from(descriptor.tile_mode),
            metal_slice,
            metal_level,
            metal_z,
        });
        linear_offset = linear_offset
            .checked_add(linear_image_bytes)
            .ok_or("GPU tiler linear buffer size overflows")?;
        Ok(())
    };

    let guest_bytes = if view.mip_level_count > 1 {
        match descriptor.tile_mode {
            27 => {
                let layout = mode27_mip_chain_layout(descriptor, view.mip_level_count)?;
                for (level_index, level) in layout.levels.iter().enumerate() {
                    let level_w =
                        u64::try_from(level.width).map_err(|_| "mip width does not fit u64")?;
                    let level_h =
                        u64::try_from(level.height).map_err(|_| "mip height does not fit u64")?;
                    push(
                        level_w,
                        level_h,
                        level_w,
                        level_h,
                        u64::try_from(level.offset).map_err(|_| "mip offset does not fit u64")?,
                        u64::try_from(level.padded_width)
                            .map_err(|_| "mip pitch does not fit u64")?,
                        u64::try_from(layout.block_width)
                            .map_err(|_| "mip block width does not fit u64")?,
                        u64::try_from(layout.block_height)
                            .map_err(|_| "mip block height does not fit u64")?,
                        u64::try_from(level.tail_x).map_err(|_| "mip tail X does not fit u64")?,
                        u64::try_from(level.tail_y).map_err(|_| "mip tail Y does not fit u64")?,
                        level.in_tail,
                        0,
                        u64::try_from(level_index).map_err(|_| "mip index does not fit u64")?,
                    )?;
                }
                u64::try_from(layout.byte_count).map_err(|_| "mip chain bytes do not fit u64")?
            }
            5 => {
                let layout = mode5_mip_chain_layout(descriptor, view.mip_level_count)?;
                let face_bytes =
                    u64::try_from(layout.byte_count).map_err(|_| "mode-5 face bytes overflow")?;
                for surface_z in 0..view.layer_count {
                    let layer_base = face_bytes
                        .checked_mul(u64::from(surface_z))
                        .ok_or("mode-5 layer base overflows")?;
                    for (level_index, level) in layout.levels.iter().enumerate() {
                        let level_offset = layer_base
                            .checked_add(
                                u64::try_from(level.offset)
                                    .map_err(|_| "mode-5 mip offset does not fit u64")?,
                            )
                            .ok_or("mode-5 layered mip offset overflows")?;
                        let level_w = u64::try_from(level.width)
                            .map_err(|_| "mode-5 mip width does not fit u64")?;
                        let level_h = u64::try_from(level.height)
                            .map_err(|_| "mode-5 mip height does not fit u64")?;
                        // Layout widths are tiling elements (texels or BC blocks). Metal blit
                        // extents stay in pixels — floor-shift the guest base, matching the
                        // CPU exact-mip upload path (not element_count * block_size, which
                        // overshoots sub-block mips).
                        let shift = u32::try_from(level_index)
                            .map_err(|_| "mode-5 mip index does not fit u32")?;
                        let metal_w = if compressed_block.is_some() {
                            u64::from(descriptor.width.checked_shr(shift).unwrap_or(0).max(1))
                        } else {
                            level_w
                        };
                        let metal_h = if compressed_block.is_some() {
                            u64::from(descriptor.height.checked_shr(shift).unwrap_or(0).max(1))
                        } else {
                            level_h
                        };
                        push(
                            level_w,
                            level_h,
                            metal_w,
                            metal_h,
                            level_offset,
                            u64::try_from(level.padded_width)
                                .map_err(|_| "mode-5 mip pitch does not fit u64")?,
                            u64::try_from(layout.block_width)
                                .map_err(|_| "mode-5 block width does not fit u64")?,
                            u64::try_from(layout.block_height)
                                .map_err(|_| "mode-5 block height does not fit u64")?,
                            u64::try_from(level.tail_x)
                                .map_err(|_| "mode-5 mip tail X does not fit u64")?,
                            u64::try_from(level.tail_y)
                                .map_err(|_| "mode-5 mip tail Y does not fit u64")?,
                            level.in_tail,
                            surface_z,
                            u64::try_from(level_index)
                                .map_err(|_| "mode-5 mip index does not fit u64")?,
                        )?;
                    }
                }
                face_bytes
                    .checked_mul(u64::from(view.layer_count))
                    .ok_or("mode-5 layered mip chain overflows")?
            }
            _ => {
                return Err(format!(
                    "Phase 1G GPU tiler refuses mip upload for tile={}",
                    descriptor.tile_mode
                ));
            }
        }
    } else {
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
        let face_bytes =
            guest_texture_byte_count(descriptor).ok_or("GPU tiler face size overflows")?;
        let elem_w =
            u64::try_from(elements_wide).map_err(|_| "element width does not fit u64")?;
        let elem_h =
            u64::try_from(elements_high).map_err(|_| "element height does not fit u64")?;
        let elem_pitch =
            u64::try_from(elements_pitch).map_err(|_| "element pitch does not fit u64")?;
        for surface_z in 0..view.layer_count {
            push(
                elem_w,
                elem_h,
                u64::from(descriptor.width),
                u64::from(descriptor.height),
                face_bytes
                    .checked_mul(u64::from(surface_z))
                    .ok_or("GPU tiler layer offset overflows")?,
                elem_pitch,
                block_width,
                block_height,
                0,
                0,
                false,
                surface_z,
                0,
            )?;
        }
        face_bytes
            .checked_mul(u64::from(view.layer_count))
            .ok_or("GPU tiler layered face size overflows")?
    };

    let linear_length = linear_offset;
    Ok((subs, guest_bytes, linear_length))
}

/// Encodes guest tiled/linear bytes → `destination` texture on `command_buffer`.
///
/// Mode 0 blit-copies rows from the guest buffer. Modes 1/5/9/24/27 detile through the
/// GPU compute kernels. Never snapshots guest RAM on the CPU.
pub(crate) fn encode_gpu_upload_from_guest(
    device: &Device,
    command_buffer: &CommandBufferRef,
    descriptor: GuestTextureDescriptor,
    destination: &Texture,
) -> Result<(), String> {
    encode_gpu_upload_from_guest_view(
        device,
        command_buffer,
        descriptor,
        destination,
        GpuTilerView::SINGLE_LEVEL_2D,
    )
}

/// Guest → texture upload for a layered / mipmapped view owned by the GPU tiler.
pub(crate) fn encode_gpu_upload_from_guest_view(
    device: &Device,
    command_buffer: &CommandBufferRef,
    descriptor: GuestTextureDescriptor,
    destination: &Texture,
    view: GpuTilerView,
) -> Result<(), String> {
    let _ = texture_element_layout(descriptor)?;
    if !gpu_tiler_owns_descriptor(descriptor) {
        return Err(format!(
            "Phase 1G: GPU upload refuses tile={} format={:?} bpe={} (no tiler ownership)",
            descriptor.tile_mode, descriptor.pixel_format, descriptor.bytes_per_pixel
        ));
    }
    // Linear/non-exact sampled mips: upload base level only; caller may generate_mipmaps.
    let plan_view = if descriptor.tile_mode == 0 && view.mip_level_count > 1 {
        GpuTilerView {
            mip_level_count: 1,
            ..view
        }
    } else {
        view
    };
    let (subs, guest_bytes, linear_length) = plan_upload_subresources(descriptor, plan_view)?;
    let (guest, guest_offset) =
        guest_buffer_for_range(device, descriptor.address, guest_bytes, descriptor.tile_mode)?;
    let span = ResourceSpan::for_texture(descriptor)
        .ok_or("Phase 1G: guest texture span overflows")?;
    let resource_base = span.range_start();
    let target_base = span.range_start();

    if descriptor.tile_mode == 0 {
        for sub in &subs {
            let guest_row_bytes = sub
                .pitch
                .checked_mul(sub.bytes_per_element)
                .ok_or("Phase 1G linear guest row bytes overflow")?;
            let guest_image_bytes = guest_row_bytes
                .checked_mul(sub.height)
                .ok_or("Phase 1G linear guest image bytes overflow")?;
            let blit = command_buffer.new_blit_command_encoder();
            blit.set_label("REMu Phase 1G linear guest→texture upload");
            blit.copy_from_buffer_to_texture(
                &guest,
                guest_offset
                    .checked_add(sub.level_offset)
                    .ok_or("Phase 1G linear guest layer offset overflows")?,
                guest_row_bytes,
                guest_image_bytes,
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
            blit.end_encoding();
        }
        return Ok(());
    }

    let linear = device.new_buffer(linear_length, MTLResourceOptions::StorageModePrivate);
    linear.set_label("REMu Phase 1G detile linear");
    let tiled_source = if guest_offset == 0 {
        guest.clone()
    } else {
        let scratch = device.new_buffer(guest_bytes, MTLResourceOptions::StorageModePrivate);
        let blit = command_buffer.new_blit_command_encoder();
        blit.set_label("REMu Phase 1G align guest tiled view");
        blit.copy_from_buffer(&guest, guest_offset, &scratch, 0, guest_bytes);
        blit.end_encoding();
        scratch
    };

    match descriptor.tile_mode {
        1 | 9 => {
            let (detile, _) = mode9_tiler_pipelines(device)?;
            for sub in &subs {
                let parameters =
                    mode9_parameters(*sub, resource_base, target_base, guest_bytes, 0)?;
                encode_detile_dispatch(
                    command_buffer,
                    &detile,
                    &tiled_source,
                    &linear,
                    &parameters,
                    *sub,
                );
            }
        }
        5 => {
            let (detile, _) = mode5_tiler_pipelines(device)?;
            for sub in &subs {
                let parameters =
                    mode5_parameters(*sub, resource_base, target_base, guest_bytes, 0)?;
                encode_detile_dispatch(
                    command_buffer,
                    &detile,
                    &tiled_source,
                    &linear,
                    &parameters,
                    *sub,
                );
            }
        }
        24 | 27 => {
            let (detile, _) = mode27_tiler_pipelines(device)?;
            for sub in &subs {
                let parameters =
                    mode27_parameters(*sub, resource_base, target_base, guest_bytes, 0)?;
                encode_detile_dispatch(
                    command_buffer,
                    &detile,
                    &tiled_source,
                    &linear,
                    &parameters,
                    *sub,
                );
            }
        }
        _ => unreachable!("tile mode gated by gpu_tiler_owns_descriptor"),
    }

    let upload = command_buffer.new_blit_command_encoder();
    upload.set_label("REMu Phase 1G detiled linear→texture");
    for sub in &subs {
        upload.copy_from_buffer_to_texture(
            &linear,
            sub.linear_offset,
            sub.linear_row_bytes,
            sub.linear_image_bytes,
            MTLSize::new(sub.metal_width, sub.metal_height, 1),
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
    }
    upload.end_encoding();
    Ok(())
}

fn encode_detile_dispatch(
    command_buffer: &CommandBufferRef,
    detile: &ComputePipelineState,
    tiled_source: &Buffer,
    linear: &Buffer,
    parameters: &[u64],
    sub: DetileSubresource,
) {
    let compute = command_buffer.new_compute_command_encoder();
    compute.set_label("REMu Phase 1G GPU detile");
    compute.set_compute_pipeline_state(detile);
    compute.set_buffer(0, Some(tiled_source), 0);
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
    encode_gpu_publish_to_guest_view(
        device,
        command_buffer,
        descriptor,
        source,
        authored,
        GpuTilerView::SINGLE_LEVEL_2D,
    )
}

/// Texture → guest publish for a layered / mipmapped view.
pub(crate) fn encode_gpu_publish_to_guest_view(
    device: &Device,
    command_buffer: &CommandBufferRef,
    descriptor: GuestTextureDescriptor,
    source: &Texture,
    authored: &[(u64, u64)],
    view: GpuTilerView,
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
    let (subs, guest_bytes, linear_length) = plan_upload_subresources(descriptor, view)?;
    let (guest, guest_offset) =
        guest_buffer_for_range(device, descriptor.address, guest_bytes, descriptor.tile_mode)?;
    let span = ResourceSpan::for_texture(descriptor)
        .ok_or("Phase 1G: guest texture span overflows")?;
    let resource_base = span.range_start();
    let target_base = span.range_start();
    let linear = device.new_buffer(linear_length, MTLResourceOptions::StorageModePrivate);
    linear.set_label("REMu Phase 1G retile linear");

    let blit = command_buffer.new_blit_command_encoder();
    blit.set_label("REMu Phase 1G texture→linear for guest publish");
    for sub in &subs {
        blit.copy_from_texture_to_buffer(
            source,
            sub.metal_slice,
            sub.metal_level,
            MTLOrigin {
                x: 0,
                y: 0,
                z: sub.metal_z,
            },
            MTLSize::new(sub.metal_width, sub.metal_height, 1),
            &linear,
            sub.linear_offset,
            sub.linear_row_bytes,
            sub.linear_image_bytes,
            MTLBlitOption::None,
        );
    }
    blit.end_encoding();

    if descriptor.tile_mode == 0 {
        for &(start, end) in authored {
            if end <= start || start < target_base {
                continue;
            }
            let offset = start - target_base;
            let len = end - start;
            if offset.saturating_add(len) > guest_bytes {
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
        let scratch = device.new_buffer(guest_bytes, MTLResourceOptions::StorageModePrivate);
        scratch.set_label("REMu Phase 1G retile scratch");
        let init = command_buffer.new_blit_command_encoder();
        init.set_label("REMu Phase 1G seed retile scratch from guest");
        init.copy_from_buffer(&guest, guest_offset, &scratch, 0, guest_bytes);
        init.end_encoding();
        scratch
    };

    match descriptor.tile_mode {
        1 | 9 => {
            let (_, retile) = mode9_tiler_pipelines(device)?;
            for sub in &subs {
                let parameters = mode9_parameters(
                    *sub,
                    resource_base,
                    target_base,
                    guest_bytes,
                    authored.len(),
                )?;
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
        }
        5 => {
            let (_, retile) = mode5_tiler_pipelines(device)?;
            for sub in &subs {
                let parameters = mode5_parameters(
                    *sub,
                    resource_base,
                    target_base,
                    guest_bytes,
                    authored.len(),
                )?;
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
        }
        24 | 27 => {
            let (_, retile) = mode27_tiler_pipelines(device)?;
            for sub in &subs {
                let parameters = mode27_parameters(
                    *sub,
                    resource_base,
                    target_base,
                    guest_bytes,
                    authored.len(),
                )?;
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
        }
        _ => unreachable!("tile mode gated by gpu_tiler_owns_descriptor"),
    }

    if guest_offset != 0 {
        for &(start, end) in authored {
            if end <= start || start < target_base {
                continue;
            }
            let offset = start - target_base;
            let len = end - start;
            if offset.saturating_add(len) > guest_bytes {
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
        assert!(gpu_tiler_owns_descriptor(descriptor(1)));
        assert!(gpu_tiler_owns_descriptor(descriptor(5)));
        assert!(gpu_tiler_owns_descriptor(descriptor(9)));
        assert!(gpu_tiler_owns_descriptor(descriptor(27)));
        assert!(gpu_tiler_owns_descriptor(descriptor(24)));
    }

    #[test]
    fn tiler_owns_mode1_5_9_bc7_element_grid() {
        let mut bc7 = descriptor(5);
        bc7.pixel_format = MTLPixelFormat::BC7_RGBAUnorm_sRGB;
        bc7.bytes_per_pixel = 16;
        assert!(gpu_tiler_owns_descriptor(bc7));
        bc7.tile_mode = 9;
        assert!(gpu_tiler_owns_descriptor(bc7));
        bc7.tile_mode = 1;
        assert!(gpu_tiler_owns_descriptor(bc7));
        // Compressed ownership stays off mode-27 until that element-grid encode is wired.
        bc7.tile_mode = 27;
        assert!(!gpu_tiler_owns_descriptor(bc7));
    }

    #[test]
    fn plans_mode5_bc7_single_level_in_element_space() {
        let mut bc7 = descriptor(5);
        bc7.width = 64;
        bc7.height = 64;
        bc7.pitch = 64;
        bc7.pixel_format = MTLPixelFormat::BC7_RGBAUnorm_sRGB;
        bc7.bytes_per_pixel = 16;
        let (subs, _, _) = plan_upload_subresources(bc7, GpuTilerView::SINGLE_LEVEL_2D)
            .expect("mode-5 BC7 plan");
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].width, 16); // 64/4 BC blocks
        assert_eq!(subs[0].height, 16);
        assert_eq!(subs[0].metal_width, 64);
        assert_eq!(subs[0].metal_height, 64);
    }

    #[test]
    fn plans_mode5_bc7_mip_chain_element_grid_with_pixel_blit_extents() {
        let mut bc7 = descriptor(5);
        bc7.width = 64;
        bc7.height = 64;
        bc7.pitch = 64;
        bc7.pixel_format = MTLPixelFormat::BC7_RGBAUnorm_sRGB;
        bc7.bytes_per_pixel = 16;
        let (subs, guest_bytes, _) = plan_upload_subresources(
            bc7,
            GpuTilerView {
                texture_type: 9,
                layer_count: 1,
                mip_level_count: 7,
            },
        )
        .expect("mode-5 BC7 mip chain plan");
        assert_eq!(subs.len(), 7);
        assert_eq!((subs[0].width, subs[0].height), (16, 16));
        assert_eq!((subs[0].metal_width, subs[0].metal_height), (64, 64));
        assert_eq!((subs[1].width, subs[1].height), (8, 8));
        assert_eq!((subs[1].metal_width, subs[1].metal_height), (32, 32));
        // Sub-block mips: one BC block element, but Metal extent is 1×1 pixel — not 4×4.
        assert_eq!((subs[6].width, subs[6].height), (1, 1));
        assert_eq!((subs[6].metal_width, subs[6].metal_height), (1, 1));
        assert!(guest_bytes > 0);
        for (index, sub) in subs.iter().enumerate() {
            assert_eq!(sub.metal_level, u64::try_from(index).unwrap());
        }
    }

    #[test]
    fn refuses_mode9_bc7_mip_chains_until_layout_exists() {
        let mut bc7 = descriptor(9);
        bc7.width = 64;
        bc7.height = 64;
        bc7.pitch = 64;
        bc7.pixel_format = MTLPixelFormat::BC7_RGBAUnorm_sRGB;
        bc7.bytes_per_pixel = 16;
        let err = match plan_upload_subresources(
            bc7,
            GpuTilerView {
                texture_type: 9,
                layer_count: 1,
                mip_level_count: 4,
            },
        ) {
            Ok(_) => panic!("mode-9 BC mips must stay loud"),
            Err(err) => err,
        };
        assert!(
            err.contains("compressed mip chains on tile=9"),
            "unexpected refuse: {err}"
        );
    }

    #[test]
    fn plans_mode9_bc7_and_mode1_r8_element_space_like_silent_hill_samples() {
        let mut bc7 = descriptor(9);
        bc7.width = 3840;
        bc7.height = 2160;
        bc7.pitch = 3840;
        bc7.pixel_format = MTLPixelFormat::BC7_RGBAUnorm_sRGB;
        bc7.bytes_per_pixel = 16;
        let (subs, _, _) = plan_upload_subresources(bc7, GpuTilerView::SINGLE_LEVEL_2D)
            .expect("mode-9 BC7 plan");
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].width, 960);
        assert_eq!(subs[0].height, 540);
        assert_eq!(subs[0].metal_width, 3840);
        assert_eq!(subs[0].metal_height, 2160);
        assert_eq!(subs[0].pitch, 960);

        let mut atlas = descriptor(9);
        atlas.width = 1024;
        atlas.height = 1024;
        atlas.pitch = 1024;
        atlas.pixel_format = MTLPixelFormat::R8Unorm;
        atlas.bytes_per_pixel = 1;
        let (subs, _, _) = plan_upload_subresources(atlas, GpuTilerView::SINGLE_LEVEL_2D)
            .expect("mode-9 R8 plan");
        assert_eq!(subs[0].width, 1024);
        assert_eq!(subs[0].metal_width, 1024);
        assert_eq!(subs[0].bytes_per_element, 1);

        let mut tiny = descriptor(1);
        tiny.width = 1;
        tiny.height = 1;
        tiny.pitch = 1;
        tiny.pixel_format = MTLPixelFormat::RGBA8Unorm;
        tiny.bytes_per_pixel = 4;
        assert!(gpu_tiler_owns_descriptor(tiny));
        let (subs, _, _) = plan_upload_subresources(tiny, GpuTilerView::SINGLE_LEVEL_2D)
            .expect("mode-1 1x1 plan");
        assert_eq!(subs[0].width, 1);
        assert_eq!(subs[0].metal_width, 1);
    }

    #[test]
    fn plans_mode5_layered_single_level() {
        let (subs, guest_bytes, _) =
            plan_upload_subresources(descriptor(5), GpuTilerView {
                texture_type: 13,
                layer_count: 3,
                mip_level_count: 1,
            })
            .expect("mode-5 layered plan");
        assert_eq!(subs.len(), 3);
        assert_eq!(guest_bytes, guest_texture_byte_count(descriptor(5)).unwrap() * 3);
    }
}
