// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    GuestMipChainLayout, GuestMipLevelLayout, GuestTextureDescriptor, MODE27_DECODE_BYTES,
    MODE27_DECODE_COUNT, MODE27_DECODE_NANOSECONDS, record_texture_work, texture_element_layout,
};
mod blocks;
mod descriptors;
mod mode27;
mod mode5;
#[cfg(test)]
pub(crate) use self::blocks::{
    mode5_within_block_byte_offset, mode9_within_block_byte_offset,
    mode24_within_block_byte_offset_for_format,
};
pub(crate) use self::blocks::{
    tiled_block_byte_offsets, tiled_block_bytes, tiled_block_dimensions,
};
#[cfg(test)]
pub(crate) use self::descriptors::guest_texture_descriptors_share_component_reordered_storage;
pub(crate) use self::descriptors::{
    GuestTextureOverlapQuery, guest_physical_ranges_overlap, guest_ranges_overlap,
    guest_texture_descriptors_have_same_layout, guest_texture_descriptors_overlap,
    guest_texture_descriptors_share_color_view_storage,
    guest_texture_descriptors_share_depth_sample_storage, guest_texture_descriptors_share_storage,
    render_resolve_candidate_matches,
};
#[cfg(test)]
pub(crate) use self::mode5::encode_mode5_mip_chain_into_raw;
pub(crate) use self::mode5::{decode_mode5_mip_chain_from_raw, mode5_mip_chain_layout};
pub(crate) use self::mode27::{
    decode_mode27_mip_chain_from_raw, encode_mode27_mip_chain_into_raw, mode27_block_byte_offsets,
    mode27_mip_chain_layout, mode27_surface_z_xor,
};
#[cfg(test)]
pub(crate) use self::mode27::{
    mode27_block_dimensions, mode27_source_byte_offset, mode27_within_block_byte_offset,
    mode27_within_block_byte_offset_for_format,
};

pub(super) fn copy_tiled_to_linear_at_z(
    source: &[u8],
    descriptor: GuestTextureDescriptor,
    destination: &mut [u8],
    surface_z: usize,
) -> Result<(), String> {
    let (width, height, pitch, bytes_per_element) = texture_element_layout(descriptor)?;
    let block_bytes = tiled_block_bytes(descriptor.tile_mode)
        .ok_or_else(|| format!("unsupported texture tile mode {}", descriptor.tile_mode))?;
    let (block_width, block_height) = tiled_block_dimensions(block_bytes, bytes_per_element)
        .ok_or("tiled texture requires a 1, 2, 4, 8, or 16-byte element format")?;
    let expected_source = usize::try_from(
        guest_texture_byte_count(descriptor).ok_or("tiled texture byte count overflows")?,
    )
    .map_err(|_| "tiled texture byte count does not fit usize")?;
    let expected_destination = width
        .checked_mul(height)
        .and_then(|elements| elements.checked_mul(bytes_per_element))
        .ok_or("tiled texture linear byte count overflows")?;
    if source.len() != expected_source || destination.len() != expected_destination {
        return Err(format!(
            "tiled texture copy has source/destination lengths {}/{}, expected {expected_source}/{expected_destination}",
            source.len(),
            destination.len(),
        ));
    }
    // The guest pitch is the physical row width. Visible width only bounds copied texels;
    // using it as the block stride shifts every later tile row on padded PS5 surfaces.
    let blocks_per_row = pitch.div_ceil(block_width);
    let visible_blocks_per_row = width.div_ceil(block_width);
    for block_y in 0..height.div_ceil(block_height) {
        let rows = (height - block_y * block_height).min(block_height);
        for block_x in 0..visible_blocks_per_row {
            let columns = (width - block_x * block_width).min(block_width);
            let block_base = (block_y * blocks_per_row + block_x) * block_bytes;
            let offsets = tiled_block_byte_offsets(
                descriptor.tile_mode,
                block_x,
                block_y,
                bytes_per_element,
                block_width,
                block_height,
            )?;
            for local_y in 0..rows {
                let destination_row = ((block_y * block_height + local_y) * width
                    + block_x * block_width)
                    * bytes_per_element;
                let offset_row = local_y * block_width;
                for local_x in 0..columns {
                    let within_block = usize::from(offsets[offset_row + local_x])
                        ^ if descriptor.tile_mode == 27 {
                            mode27_surface_z_xor(surface_z)
                        } else {
                            0
                        };
                    let source_offset = block_base + within_block;
                    let destination_offset = destination_row + local_x * bytes_per_element;
                    destination[destination_offset..destination_offset + bytes_per_element]
                        .copy_from_slice(&source[source_offset..source_offset + bytes_per_element]);
                }
            }
        }
    }
    Ok(())
}

pub(super) fn copy_linear_to_tiled_at_z(
    source: &[u8],
    descriptor: GuestTextureDescriptor,
    destination: &mut [u8],
    surface_z: usize,
) -> Result<(), String> {
    let (width, height, pitch, bytes_per_element) = texture_element_layout(descriptor)?;
    let block_bytes = tiled_block_bytes(descriptor.tile_mode)
        .ok_or_else(|| format!("unsupported texture tile mode {}", descriptor.tile_mode))?;
    let (block_width, block_height) = tiled_block_dimensions(block_bytes, bytes_per_element)
        .ok_or("tiled texture requires a 1, 2, 4, 8, or 16-byte element format")?;
    let expected_source = width
        .checked_mul(height)
        .and_then(|elements| elements.checked_mul(bytes_per_element))
        .ok_or("tiled texture linear byte count overflows")?;
    let expected_destination = usize::try_from(
        guest_texture_byte_count(descriptor).ok_or("tiled texture byte count overflows")?,
    )
    .map_err(|_| "tiled texture byte count does not fit usize")?;
    if source.len() != expected_source || destination.len() != expected_destination {
        return Err(format!(
            "tiled texture copy has source/destination lengths {}/{}, expected {expected_source}/{expected_destination}",
            source.len(),
            destination.len(),
        ));
    }
    let blocks_per_row = pitch.div_ceil(block_width);
    let visible_blocks_per_row = width.div_ceil(block_width);
    for block_y in 0..height.div_ceil(block_height) {
        let rows = (height - block_y * block_height).min(block_height);
        for block_x in 0..visible_blocks_per_row {
            let columns = (width - block_x * block_width).min(block_width);
            let block_base = (block_y * blocks_per_row + block_x) * block_bytes;
            let offsets = tiled_block_byte_offsets(
                descriptor.tile_mode,
                block_x,
                block_y,
                bytes_per_element,
                block_width,
                block_height,
            )?;
            for local_y in 0..rows {
                let source_row = ((block_y * block_height + local_y) * width
                    + block_x * block_width)
                    * bytes_per_element;
                let offset_row = local_y * block_width;
                for local_x in 0..columns {
                    let source_offset = source_row + local_x * bytes_per_element;
                    let within_block = usize::from(offsets[offset_row + local_x])
                        ^ if descriptor.tile_mode == 27 {
                            mode27_surface_z_xor(surface_z)
                        } else {
                            0
                        };
                    let destination_offset = block_base + within_block;
                    destination[destination_offset..destination_offset + bytes_per_element]
                        .copy_from_slice(&source[source_offset..source_offset + bytes_per_element]);
                }
            }
        }
    }
    Ok(())
}

pub(super) fn effective_linear_pitch(width: u32, bytes_per_pixel: u32) -> Option<u32> {
    if !matches!(bytes_per_pixel, 1 | 2 | 4 | 8 | 16) {
        return None;
    }
    let alignment = 256 / bytes_per_pixel;
    width
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
}

pub(super) fn guest_texture_byte_count(descriptor: GuestTextureDescriptor) -> Option<u64> {
    let (_width, height, pitch, bytes_per_element) = texture_element_layout(descriptor).ok()?;
    if let Some(block_bytes) = tiled_block_bytes(descriptor.tile_mode) {
        let (block_width, block_height) = tiled_block_dimensions(block_bytes, bytes_per_element)?;
        let blocks_wide = u64::try_from(pitch)
            .ok()?
            .div_ceil(u64::try_from(block_width).ok()?);
        let blocks_high = u64::try_from(height)
            .ok()?
            .div_ceil(u64::try_from(block_height).ok()?);
        blocks_wide
            .checked_mul(blocks_high)?
            .checked_mul(u64::try_from(block_bytes).ok()?)
    } else {
        u64::try_from(pitch)
            .ok()?
            .checked_mul(u64::try_from(height).ok()?)?
            .checked_mul(u64::try_from(bytes_per_element).ok()?)
    }
}
