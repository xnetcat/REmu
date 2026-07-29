// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CachedSampledTexture, DecodedDepthTarget, DecodedImageDescriptor, DecodedRenderTarget,
    DepthArrayKey, DepthArrayResidentTexture, DirectScanoutTexture, GuestTextureDescriptor,
    GuestTextureOverlapQuery, GuestTextureWriteWatch, MIP_UPLOAD_BYTES, MIP_UPLOAD_COUNT,
    MIP_UPLOAD_NANOSECONDS, MetalPresenter, Mode27WorkOrigin, MultisampleResidentTexture,
    MultisampleTextureKey, RESIDENT_DEPTH_SAMPLE_HITS, RESIDENT_TEXTURE_ACQUIRE_INVALIDATIONS,
    ResidentGuestTexture, SAMPLED_ALIAS_WRITEBACK_BYTES, SAMPLED_ALIAS_WRITEBACK_COUNT,
    SAMPLED_ALIAS_WRITEBACK_NANOSECONDS, SAMPLED_TEXTURE_CACHE_ACQUIRE_INVALIDATIONS,
    SAMPLED_TEXTURE_CACHE_CAPACITY, SAMPLED_TEXTURE_CACHE_CAPACITY_EVICTIONS,
    SAMPLED_TEXTURE_CACHE_DIRTY_RELOADS, SAMPLED_TEXTURE_CACHE_HITS, SAMPLED_TEXTURE_CACHE_MISSES,
    SampledTextureKey, ScanoutSnapshotTexture, TEXTURE_SNAPSHOT_BYTES, TEXTURE_SNAPSHOT_COUNT,
    TEXTURE_SNAPSHOT_NANOSECONDS, UploadedImageTexture, WriteWatchRefresh,
    canonical_texture_bindings, compressed_texture_block_extent, copy_linear_to_tiled_at_z,
    copy_mode27_mip_level_to_linear, copy_tiled_to_linear_at_z, decode_image_descriptor,
    decode_mode5_mip_chain_from_raw, decode_mode27_mip_chain_from_raw, fallback_image_descriptor,
    guest_frame_fingerprint, guest_ranges_overlap, guest_texture_byte_count,
    guest_texture_descriptors_have_same_layout, guest_texture_descriptors_overlap,
    guest_texture_descriptors_share_color_view_storage,
    guest_texture_descriptors_share_depth_sample_storage, guest_texture_descriptors_share_storage,
    image_resource_descriptor_is_null, initial_depth_clear_for_target, is_storage_image_opcode,
    is_writable_image_opcode, mode5_mip_chain_layout, mode27_mip_chain_layout,
    record_mode27_work_detail, record_texture_dirty_work_detail,
    record_texture_snapshot_work_detail, record_texture_work, render_resolve_candidate_matches,
    sampled_alias_descriptor, sampler_uses_normalized_coordinates, texture_element_layout,
    trace_image_descriptors, trace_target_sampling, write_resident_texture_contents_to_guest,
};
#[cfg(test)]
use remu_memory::guest_shared_mapping_identity;
use remu_memory::{CpuWriteWatchRetirement, read_guest_shared_backing, write_guest_shared_backing};
mod depth_array;
mod mips;
mod residency;
mod retire;
mod sampled;
mod upload;
mod watch;
pub(crate) use self::depth_array::resident_is_depth_array_face;
#[cfg(test)]
pub(crate) use self::depth_array::{depth_array_face_descriptor, depth_array_face_set_is_complete};
pub(crate) use self::mips::{
    create_exact_mode5_sampled_mip_view, create_exact_mode27_sampled_mip_view,
    create_mipmapped_guest_texture, ensure_mipmapped_resident_texture_in_map,
    promote_resident_mip_hierarchy, storage_image_requires_mip_hierarchy,
};
pub(crate) use self::residency::{create_guest_texture, ensure_resident_texture_in_map};
pub(crate) use self::retire::{
    retire_multisample_textures_overlapping_writeback, retire_overlapping_sampled_textures,
    retire_resident_textures_overlapping_writeback, retire_sampled_texture,
    retire_sampled_textures_matching, retire_sampled_textures_overlapping_writeback,
    trace_resident_texture_removal,
};
#[cfg(test)]
pub(crate) use self::sampled::{
    apply_image_destination_select, image_destination_select_is_native,
    new_color_alias_texture_view, new_red_blue_swizzled_texture_view,
    prepare_sampled_texture_upload,
};
pub(crate) use self::sampled::{
    create_sampled_guest_texture, decode_sampled_texture_linear_from_raw,
    resident_color_view_sample_is_eligible, resident_depth_sample_is_eligible,
    resident_texture_for_sample, sampled_texture_cache_needs_reload, sampled_texture_key,
};
pub(crate) use self::upload::{comparison_depth32_upload, upload_image_textures};
#[cfg(test)]
pub(crate) use self::upload::{
    sampled_texture_cache_accepts, sampled_texture_cache_requires_eviction,
};
#[cfg(test)]
pub(crate) use self::watch::{guest_texture_raw_range, sampled_texture_raw_range};
pub(crate) use self::watch::{
    persistent_sampled_texture_write_watch, refresh_resident_texture_write_watch,
    refresh_sampled_texture_write_watch, sampled_texture_uses_exact_mode5_mips,
    sampled_texture_uses_exact_mode27_mips, snapshot_guest_texture_raw,
    snapshot_watched_guest_range, snapshot_watched_guest_texture_raw,
    snapshot_watched_sampled_texture_raw, watch_guest_texture, watch_resident_texture,
    watch_sampled_texture, write_watched_guest_range,
};

/// Metal exposes this C structure by value in the swizzled texture-view Objective-C method.
/// Its members are `uint8_t` in the macOS SDK; keep the layout explicit because metal-rs 0.33
/// does not yet bind this overload.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct MetalTextureSwizzleChannels {
    red: u8,
    green: u8,
    blue: u8,
    alpha: u8,
}

pub(super) fn decode_guest_texture_linear_from_raw(
    descriptor: GuestTextureDescriptor,
    source: &[u8],
) -> Result<Vec<u8>, String> {
    decode_guest_texture_linear_from_raw_at_z(descriptor, source, 0)
}

pub(super) fn decode_guest_texture_linear_from_raw_at_z(
    descriptor: GuestTextureDescriptor,
    source: &[u8],
    surface_z: usize,
) -> Result<Vec<u8>, String> {
    let (width, height, pitch, bytes_per_element) = texture_element_layout(descriptor)?;
    let output_length = u64::try_from(width)
        .ok()
        .and_then(|width| width.checked_mul(u64::try_from(height).ok()?))
        .and_then(|elements| elements.checked_mul(u64::try_from(bytes_per_element).ok()?))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or("guest texture dimensions overflow")?;
    let expected_source = usize::try_from(
        guest_texture_byte_count(descriptor).ok_or("guest texture byte count overflows")?,
    )
    .map_err(|_| "guest texture byte count does not fit usize")?;
    if source.len() != expected_source {
        return Err(format!(
            "guest texture source has {} bytes, expected {expected_source}",
            source.len()
        ));
    }
    let mut output = vec![0_u8; output_length];
    if descriptor.tile_mode == 0 {
        let source_row_bytes = pitch
            .checked_mul(bytes_per_element)
            .ok_or("source row byte count overflows")?;
        let output_row_bytes = width
            .checked_mul(bytes_per_element)
            .ok_or("output row byte count overflows")?;
        for y in 0..height {
            let source_offset = y
                .checked_mul(source_row_bytes)
                .ok_or("source row offset overflows")?;
            let destination_offset = y
                .checked_mul(output_row_bytes)
                .ok_or("destination row offset overflows")?;
            output[destination_offset..destination_offset + output_row_bytes].copy_from_slice(
                source
                    .get(source_offset..source_offset + output_row_bytes)
                    .ok_or("guest texture row exceeds its decoded allocation")?,
            );
        }
    } else {
        copy_tiled_to_linear_at_z(source, descriptor, &mut output, surface_z)?;
    }
    Ok(output)
}

pub(super) fn write_guest_texture_linear_into_raw(
    descriptor: GuestTextureDescriptor,
    linear: &[u8],
    destination: &mut [u8],
) -> Result<(), String> {
    write_guest_texture_linear_into_raw_at_z(descriptor, linear, destination, 0)
}

pub(super) fn write_guest_texture_linear_into_raw_at_z(
    descriptor: GuestTextureDescriptor,
    linear: &[u8],
    destination: &mut [u8],
    surface_z: usize,
) -> Result<(), String> {
    let (width, height, pitch, bytes_per_element) = texture_element_layout(descriptor)?;
    if descriptor.tile_mode == 0 {
        let destination_row_bytes = pitch
            .checked_mul(bytes_per_element)
            .ok_or("destination row byte count overflows")?;
        let source_row_bytes = width
            .checked_mul(bytes_per_element)
            .ok_or("source row byte count overflows")?;
        for y in 0..height {
            let source_offset = y
                .checked_mul(source_row_bytes)
                .ok_or("source row offset overflows")?;
            let destination_offset = y
                .checked_mul(destination_row_bytes)
                .ok_or("destination row offset overflows")?;
            destination
                .get_mut(destination_offset..destination_offset + source_row_bytes)
                .ok_or("guest texture row exceeds its decoded allocation")?
                .copy_from_slice(&linear[source_offset..source_offset + source_row_bytes]);
        }
    } else {
        copy_linear_to_tiled_at_z(linear, descriptor, destination, surface_z)?;
    }
    Ok(())
}

pub(super) fn write_watched_guest_texture_linear(
    descriptor: GuestTextureDescriptor,
    linear: &[u8],
    write_watch: &GuestTextureWriteWatch,
) -> Result<(), String> {
    if write_watch.retirement() == Some(CpuWriteWatchRetirement::Replaced) {
        return Err(format!(
            "guest texture mapping {:#x} was retired before writeback",
            descriptor.address
        ));
    }
    let byte_count = guest_texture_byte_count(descriptor)
        .ok_or("guest texture byte count overflows the host address space")?;
    let destination_length =
        usize::try_from(byte_count).map_err(|_| "guest texture byte count does not fit usize")?;
    write_watch
        .with_live_backing(|| {
            let mut destination = vec![0; destination_length];
            if !read_guest_shared_backing(descriptor.address, &mut destination) {
                return Err(format!(
                    "guest texture range {:#x}+{byte_count:#x} has no complete live shared backing",
                    descriptor.address
                ));
            }
            write_guest_texture_linear_into_raw(descriptor, linear, &mut destination)?;
            if !write_guest_shared_backing(descriptor.address, &destination) {
                return Err(format!(
                    "guest texture range {:#x}+{byte_count:#x} lost shared backing during writeback",
                    descriptor.address
                ));
            }
            Ok(())
        })
        .and_then(|result| result)
}

#[cfg(test)]
pub(super) fn guest_texture_raw_matches(
    descriptor: GuestTextureDescriptor,
    shadow: &[u8],
) -> Result<bool, String> {
    let (source_address, source_length) = guest_texture_raw_range(descriptor)?;
    // SAFETY: See snapshot_guest_texture_raw; the comparison does not retain the borrowed range.
    let source = unsafe { std::slice::from_raw_parts(source_address as *const u8, source_length) };
    Ok(source == shadow)
}

#[cfg(test)]
pub(super) fn snapshot_guest_texture_linear(
    descriptor: GuestTextureDescriptor,
) -> Result<Vec<u8>, String> {
    let (source_address, source_length) = guest_texture_raw_range(descriptor)?;
    // SAFETY: See snapshot_guest_texture_raw; decoding completes before this function returns.
    let source = unsafe { std::slice::from_raw_parts(source_address as *const u8, source_length) };
    decode_guest_texture_linear_from_raw(descriptor, source)
}

#[cfg(test)]
pub(super) fn write_guest_texture_linear(
    descriptor: GuestTextureDescriptor,
    linear: &[u8],
) -> Result<(), String> {
    let byte_count = guest_texture_byte_count(descriptor)
        .ok_or("guest texture byte count overflows the host address space")?;
    if descriptor.address < 0x10_000 {
        return Err(format!(
            "invalid guest texture range address={:#x} bytes={byte_count}",
            descriptor.address
        ));
    }
    let (width, height, _pitch, bytes_per_element) = texture_element_layout(descriptor)?;
    let required = width
        .checked_mul(height)
        .and_then(|elements| elements.checked_mul(bytes_per_element))
        .ok_or("guest texture dimensions overflow")?;
    if linear.len() != required {
        return Err(format!(
            "tight guest image has {} bytes, expected {required}",
            linear.len()
        ));
    }
    let destination_length =
        usize::try_from(byte_count).map_err(|_| "guest texture byte count does not fit usize")?;
    if guest_shared_mapping_identity(descriptor.address, byte_count).is_some() {
        let destination_address = usize::try_from(descriptor.address)
            .map_err(|_| "guest texture address does not fit usize")?;
        // SAFETY: the registry proved the full writable texture range is one live contiguous
        // mapping, and the caller's write watch has not been retired.
        let destination = unsafe {
            std::slice::from_raw_parts_mut(destination_address as *mut u8, destination_length)
        };
        write_guest_texture_linear_into_raw(descriptor, linear, destination)?;
        return Ok(());
    }
    let mut destination = vec![0; destination_length];
    let has_shared_backing = read_guest_shared_backing(descriptor.address, &mut destination);
    #[cfg(test)]
    if !has_shared_backing {
        let destination_address = usize::try_from(descriptor.address)
            .map_err(|_| "guest texture address does not fit usize")?;
        // SAFETY: process-local unit-test descriptors retain the full destination allocation.
        destination.copy_from_slice(unsafe {
            std::slice::from_raw_parts(destination_address as *const u8, destination_length)
        });
    }
    #[cfg(not(test))]
    if !has_shared_backing {
        return Err(format!(
            "guest texture range {:#x}+{byte_count:#x} has no complete live shared backing",
            descriptor.address
        ));
    }
    write_guest_texture_linear_into_raw(descriptor, linear, &mut destination)?;
    if has_shared_backing {
        if !write_guest_shared_backing(descriptor.address, &destination) {
            return Err(format!(
                "guest texture range {:#x}+{byte_count:#x} lost shared backing during writeback",
                descriptor.address
            ));
        }
    } else {
        #[cfg(test)]
        {
            let destination_address = usize::try_from(descriptor.address)
                .map_err(|_| "guest texture address does not fit usize")?;
            // SAFETY: process-local unit-test descriptors retain the full destination allocation.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    destination.as_ptr(),
                    destination_address as *mut u8,
                    destination.len(),
                );
            }
        }
    }
    Ok(())
}
