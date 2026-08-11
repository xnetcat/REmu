// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CachedSampledTexture, DecodedDepthTarget, DecodedImageDescriptor, DecodedRenderTarget,
    DepthArrayKey, DepthArrayResidentTexture, DirectScanoutTexture, GuestScissor,
    GuestTextureContentState, GuestTextureDescriptor, GuestTextureWatchRange,
    GuestTextureWriteWatch, MIP_UPLOAD_BYTES, MIP_UPLOAD_COUNT, MIP_UPLOAD_NANOSECONDS,
    MetalPresenter, Mode27WorkOrigin, MultisampleResidentTexture, MultisampleTextureKey,
    OWNED_FALLBACK_FULL_RANGE, OWNED_RANGE_SET_COARSENED, OWNED_REMATERIALISE_BLIT,
    OWNED_REMATERIALISE_CPU, OWNED_REMATERIALISE_DETILE, OWNED_REMATERIALISE_FAILED,
    OWNED_REMATERIALISE_FAILED_CPU_GPU, OWNED_REMATERIALISE_FAILED_LAYOUT,
    OWNED_REMATERIALISE_FAILED_MAPPING, OWNED_REMATERIALISE_FAILED_METAL, OWNED_REMATERIALISE_VIEW,
    OWNED_SUBTRACT_TENANTS_SURVIVED, RESIDENT_DEPTH_SAMPLE_HITS,
    RESIDENT_TEXTURE_ACQUIRE_INVALIDATIONS, RESIDENT_WRITE_ATTRIBUTION_MISSES, ResidencyState,
    ResidentGuestTexture, ResidentTextureActivityStamp, ResourceSpan,
    SAMPLED_ALIAS_WRITEBACK_BYTES, SAMPLED_ALIAS_WRITEBACK_COUNT,
    SAMPLED_ALIAS_WRITEBACK_NANOSECONDS, SAMPLED_TEXTURE_CACHE_ACQUIRE_INVALIDATIONS,
    SAMPLED_TEXTURE_CACHE_CAPACITY, SAMPLED_TEXTURE_CACHE_CAPACITY_EVICTIONS,
    SAMPLED_TEXTURE_CACHE_DIRTY_RELOADS, SAMPLED_TEXTURE_CACHE_HITS, SAMPLED_TEXTURE_CACHE_MISSES,
    SampledTextureKey, ScanoutSnapshotTexture, StaleSource, TEXTURE_DIRTY_VALIDATION_BYTES,
    TEXTURE_DIRTY_VALIDATION_COUNT, TEXTURE_DIRTY_VALIDATION_NANOSECONDS,
    TEXTURE_DIRTY_VALIDATION_UNCHANGED, TEXTURE_SNAPSHOT_BYTES, TEXTURE_SNAPSHOT_COUNT,
    TEXTURE_SNAPSHOT_NANOSECONDS, TEXTURE_SNAPSHOT_UNSTABLE_COUNT, UploadedImageTexture,
    WriteWatchRefresh, canonical_texture_bindings, compressed_texture_block_extent,
    copy_linear_to_tiled_at_z, copy_mode27_mip_level_to_linear, copy_tiled_to_linear_at_z,
    decode_image_descriptor, decode_mode5_mip_chain_from_raw, decode_mode27_mip_chain_from_raw,
    depth_sample_view_must_not_steal_depth_authority, fallback_image_descriptor,
    guest_frame_fingerprint, guest_ranges_overlap, guest_texture_byte_count,
    guest_texture_descriptors_have_same_layout, guest_texture_descriptors_overlap,
    guest_texture_descriptors_share_byte_storage,
    guest_texture_descriptors_share_color_view_storage,
    guest_texture_descriptors_share_depth_sample_storage,
    guest_texture_descriptors_share_gpu_sample_storage, guest_texture_descriptors_share_storage,
    image_resource_descriptor_is_null, initial_depth_clear_for_target, is_storage_image_opcode,
    is_writable_image_opcode, mode5_mip_chain_layout, mode27_mip_chain_layout,
    mode27_surface_z_xor, record_mode27_work_detail, record_texture_dirty_work_detail,
    record_texture_snapshot_work_detail, record_texture_work, render_resolve_candidate_matches,
    sampled_alias_descriptor, sampler_uses_normalized_coordinates, spans_overlap,
    texture_element_layout, tiled_block_bytes, tiled_block_dimensions, trace_glyph_atlas,
    trace_image_descriptors, trace_target_sampling, trace_video_target_consumers,
};
use remu_memory::{CpuWriteWatchRetirement, read_guest_shared_backing};
#[cfg(test)]
use remu_memory::{guest_shared_mapping_identity, write_guest_shared_backing};
mod correlation;
mod depth_array;
mod extent;
mod lifetime;
mod mips;
mod ranges;
mod residency;
mod resident_map;
mod retire;
mod sampled;
mod upload;
mod watch;
pub(crate) use self::correlation::{
    texture_alias_trace_match, trace_alias_correlated_acquire,
    trace_alias_correlated_image_bindings, trace_residency_churn,
};
pub(crate) use self::depth_array::resident_is_depth_array_face;
#[cfg(test)]
pub(crate) use self::depth_array::{depth_array_face_descriptor, depth_array_face_set_is_complete};
pub(crate) use self::extent::{WrittenRegion, resident_declared_span, written_byte_ranges};
pub(crate) use self::lifetime::{
    RESIDENT_FLIP_AGE_LABELS, RESIDENT_WORK_AGE_LABELS, resident_texture_activity_stats,
    resident_texture_backing_overlap_stats, resident_texture_eviction_plan,
};
pub(crate) use self::mips::{
    create_exact_mode5_sampled_mip_view, create_exact_mode27_sampled_mip_view,
    create_mipmapped_guest_texture, ensure_mipmapped_resident_texture_in_map,
    promote_resident_mip_hierarchy, storage_image_requires_mip_hierarchy,
};
pub(crate) use self::ranges::RangeSet;
#[cfg(test)]
pub(crate) use self::residency::rematerialisation_blit_source_origin_y;
pub(crate) use self::residency::{
    ensure_resident_texture_in_map, resident_texture_ownership_stats,
};
#[cfg(test)]
pub(crate) use self::resident_map::ResidentSpanIndex;
pub(crate) use self::resident_map::{ResidentTextureMap, SpanQueryOrder};
pub(crate) use self::retire::{
    retire_multisample_textures_overlapping_writeback, retire_overlapping_sampled_textures,
    retire_sampled_texture, retire_sampled_textures_matching,
    retire_sampled_textures_overlapping_writeback, trace_resident_texture_demotion,
    trace_resident_texture_removal,
};
pub(crate) use self::sampled::new_color_alias_texture_view;
#[cfg(test)]
pub(crate) use self::sampled::{
    apply_image_destination_select, image_destination_select_is_native,
    new_red_blue_swizzled_texture_view, prepare_sampled_texture_upload,
};
pub(crate) use self::sampled::{
    create_sampled_guest_texture, create_sampled_guest_texture_from_guest_gpu,
    decode_sampled_texture_linear_from_raw, gpu_written_resident_serving_sampled_key,
    resident_color_view_sample_is_eligible, resident_depth_sample_is_eligible,
    resident_same_image_sample_is_eligible, resident_sample_authority_ok,
    resident_texture_for_sample, sampled_texture_cache_needs_reload, sampled_texture_key,
    sampled_texture_supports_access, sampled_texture_usage, sampled_texture_view_for_binding,
};
pub(crate) use self::upload::{comparison_depth32_upload, upload_image_textures};
#[cfg(test)]
pub(crate) use self::upload::{
    sampled_texture_cache_accepts, sampled_texture_cache_requires_eviction,
};
#[cfg(test)]
pub(crate) use self::watch::guest_texture_raw_range;
pub(crate) use self::watch::{
    TextureWritebackOutcome, persistent_sampled_texture_write_watch,
    refresh_resident_texture_write_watch, refresh_sampled_texture_write_watch,
    resident_texture_raw_byte_count, sampled_texture_raw_range,
    sampled_texture_uses_exact_mode5_mips, sampled_texture_uses_exact_mode27_mips,
    snapshot_guest_texture_raw, snapshot_watched_guest_range, snapshot_watched_guest_texture_raw,
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

/// Publishes a decoded image into its guest bytes.
///
/// `authored` names the runs the GPU actually wrote, as offsets from `descriptor.address`; `None`
/// publishes the whole declared range. Only the tiled path consults it — a linear image already
/// publishes exactly its visible rows and never touches the padding between them.
///
/// Publication goes through [`remu_memory::GuestRangeLease`] (MASTER-PLAN D3). A revoked lease
/// returns [`TextureWritebackOutcome::DiscardedStale`] so the caller can demote the resident
/// (D4-texture) instead of silently publishing into recycled guest memory.
pub(super) fn write_watched_guest_texture_linear(
    descriptor: GuestTextureDescriptor,
    linear: &[u8],
    authored: Option<&[(u64, u64)]>,
    write_watch: &GuestTextureWriteWatch,
) -> Result<TextureWritebackOutcome, String> {
    if write_watch.retirement() == Some(CpuWriteWatchRetirement::Replaced) {
        return Err(format!(
            "guest texture mapping {:#x} was retired before writeback",
            descriptor.address
        ));
    }
    let byte_count = guest_texture_byte_count(descriptor)
        .ok_or("guest texture byte count overflows the host address space")?;
    crate::presenter::diagnostics::trace_diagnostic_guest_write(
        "texture-writeback",
        descriptor.address,
        byte_count,
    );
    let destination_length =
        usize::try_from(byte_count).map_err(|_| "guest texture byte count does not fit usize")?;
    let entry_generations = write_watch.generations();
    let (width, _, pitch, bytes_per_element) = texture_element_layout(descriptor)?;
    if descriptor.tile_mode == 0 {
        // A linear image owns only its visible row bytes; the pitch padding between rows
        // belongs to whoever else lives in the pooled allocation. Publish exactly the rows —
        // no read-modify-write, so a concurrent guest store to the padding can never be
        // reverted (the MallocBinned3 canary corruption mechanism).
        let row_bytes = width
            .checked_mul(bytes_per_element)
            .ok_or("source row byte count overflows")?;
        let destination_row_bytes = pitch
            .checked_mul(bytes_per_element)
            .ok_or("destination row byte count overflows")?;
        if row_bytes == 0 || !linear.len().is_multiple_of(row_bytes) {
            return Err(format!(
                "linear writeback payload of {} bytes is not whole {row_bytes}-byte rows",
                linear.len()
            ));
        }
        let segments = linear
            .chunks_exact(row_bytes)
            .enumerate()
            .map(|(row, bytes)| (row as u64 * destination_row_bytes as u64, bytes))
            .collect::<Vec<_>>();
        if write_watch.generations() != entry_generations {
            return Err(format!(
                "guest wrote texture range {:#x} during writeback; publishing would revert it",
                descriptor.address
            ));
        }
        let mut raw = vec![0_u8; destination_length];
        let mut runs = Vec::with_capacity(segments.len());
        for (offset, bytes) in &segments {
            let start = usize::try_from(*offset)
                .map_err(|_| "linear writeback row offset does not fit usize")?;
            let end = start
                .checked_add(bytes.len())
                .ok_or("linear writeback row range overflows")?;
            raw.get_mut(start..end)
                .ok_or("linear writeback row exceeds its footprint")?
                .copy_from_slice(bytes);
            runs.push((
                *offset,
                offset.saturating_add(u64::try_from(bytes.len()).unwrap_or(0)),
            ));
        }
        return write_watched_guest_range(
            descriptor.address,
            &raw,
            Some(&runs),
            None,
            write_watch,
            None,
        );
    }
    let baseline = write_watch.with_live_backing(|| {
        let mut destination = vec![0; destination_length];
        if !read_guest_shared_backing(descriptor.address, &mut destination) {
            return Err(format!(
                "guest texture range {:#x}+{byte_count:#x} has no complete live shared backing",
                descriptor.address
            ));
        }
        Ok(destination)
    })??;
    let mut destination = baseline.clone();
    write_guest_texture_linear_into_raw(descriptor, linear, &mut destination)?;
    // The tiled footprint is re-encoded through a read-modify-write, so a guest store
    // between the read above and this write would be reverted. Detect that store by
    // its fault generation and abort loudly rather than publish over it.
    if write_watch.generations() != entry_generations {
        return Err(format!(
            "guest wrote texture range {:#x} during writeback; publishing would revert it",
            descriptor.address
        ));
    }
    write_watched_guest_range(
        descriptor.address,
        &destination,
        authored,
        Some(&baseline),
        write_watch,
        None,
    )
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
