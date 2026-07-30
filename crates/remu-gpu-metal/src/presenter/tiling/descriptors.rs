// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{GuestTextureDescriptor, guest_texture_byte_count};
use metal::MTLPixelFormat;
use remu_memory::{GuestSharedMappingIdentity, guest_shared_mapping_identity};

#[derive(Clone, Copy)]
pub(crate) struct GuestTextureOverlapQuery {
    descriptor: GuestTextureDescriptor,
    byte_count: Option<u64>,
    backing: Option<GuestSharedMappingIdentity>,
}

impl GuestTextureOverlapQuery {
    pub(crate) fn new(descriptor: GuestTextureDescriptor) -> Self {
        let byte_count = guest_texture_byte_count(descriptor);
        let backing = byte_count.and_then(|byte_count| {
            remu_memory::guest_shared_backing_identity(descriptor.address, byte_count)
        });
        Self {
            descriptor,
            byte_count,
            backing,
        }
    }

    #[cfg(test)]
    pub(crate) fn overlaps(&self, other: GuestTextureDescriptor) -> bool {
        self.overlaps_query(&Self::new(other))
    }

    /// Reports overlap with a descriptor whose physical identity was resolved previously.
    ///
    /// Sampled textures keep this query for their cache lifetime. Render-target invalidation can
    /// therefore scan the cache without taking the process-global shared-mapping registry lock
    /// once per entry. Keeping the admission-time identity also preserves the physical storage
    /// represented by a cached texture if its original VA is later rebound.
    pub(crate) fn overlaps_query(&self, other: &Self) -> bool {
        let (Some(left_bytes), Some(right_bytes)) = (self.byte_count, other.byte_count) else {
            return false;
        };
        if guest_ranges_overlap(
            self.descriptor.address,
            left_bytes,
            other.descriptor.address,
            right_bytes,
        ) {
            return true;
        }
        let (Some(left_backing), Some(right_backing)) = (self.backing, other.backing) else {
            return remu_memory::guest_shared_backing_ranges_overlap(
                self.descriptor.address,
                left_bytes,
                other.descriptor.address,
                right_bytes,
            );
        };
        left_backing.backing_id == right_backing.backing_id
            && guest_ranges_overlap(
                left_backing.backing_offset,
                left_bytes,
                right_backing.backing_offset,
                right_bytes,
            )
    }
}

pub(crate) fn guest_texture_descriptors_overlap(
    left: GuestTextureDescriptor,
    right: GuestTextureDescriptor,
) -> bool {
    let Some(left_bytes) = guest_texture_byte_count(left) else {
        return false;
    };
    let Some(right_bytes) = guest_texture_byte_count(right) else {
        return false;
    };
    if guest_ranges_overlap(left.address, left_bytes, right.address, right_bytes) {
        return true;
    }
    remu_memory::guest_shared_backing_ranges_overlap(
        left.address,
        left_bytes,
        right.address,
        right_bytes,
    )
}

pub(crate) fn guest_texture_descriptors_share_storage(
    left: GuestTextureDescriptor,
    right: GuestTextureDescriptor,
) -> bool {
    if left == right {
        return true;
    }
    if !guest_texture_descriptors_have_same_layout(left, right) {
        return false;
    }
    guest_texture_descriptors_share_byte_storage(left, right)
}

#[cfg(test)]
pub(crate) fn guest_texture_descriptors_share_component_reordered_storage(
    left: GuestTextureDescriptor,
    right: GuestTextureDescriptor,
) -> bool {
    let formats_are_red_blue_reorders = matches!(
        (left.pixel_format, right.pixel_format),
        (MTLPixelFormat::BGRA8Unorm, MTLPixelFormat::RGBA8Unorm)
            | (MTLPixelFormat::RGBA8Unorm, MTLPixelFormat::BGRA8Unorm)
            | (
                MTLPixelFormat::BGRA8Unorm_sRGB,
                MTLPixelFormat::RGBA8Unorm_sRGB
            )
            | (
                MTLPixelFormat::RGBA8Unorm_sRGB,
                MTLPixelFormat::BGRA8Unorm_sRGB
            )
    );
    formats_are_red_blue_reorders
        && guest_texture_descriptors_have_same_byte_layout(left, right)
        && guest_texture_descriptors_share_byte_storage(left, right)
}

pub(crate) fn guest_texture_descriptors_share_color_view_storage(
    left: GuestTextureDescriptor,
    right: GuestTextureDescriptor,
) -> bool {
    let is_view_compatible_color = |format| {
        matches!(
            format,
            MTLPixelFormat::RGBA8Unorm
                | MTLPixelFormat::RGBA8Unorm_sRGB
                | MTLPixelFormat::BGRA8Unorm
                | MTLPixelFormat::BGRA8Unorm_sRGB
        )
    };
    left.pixel_format != right.pixel_format
        && is_view_compatible_color(left.pixel_format)
        && is_view_compatible_color(right.pixel_format)
        && guest_texture_descriptors_have_same_byte_layout(left, right)
        && guest_texture_descriptors_share_byte_storage(left, right)
}

pub(crate) fn guest_texture_descriptors_share_depth_sample_storage(
    rendered: GuestTextureDescriptor,
    sampled: GuestTextureDescriptor,
    destination_select: u32,
) -> bool {
    let formats_are_depth_sample_views = matches!(
        (rendered.pixel_format, sampled.pixel_format),
        (MTLPixelFormat::Depth16Unorm, MTLPixelFormat::R16Unorm)
            | (MTLPixelFormat::Depth32Float, MTLPixelFormat::R32Float)
    );
    // Prospero depth SRDs expose the native Z plane through R, optionally replicating it or
    // supplying zero/one in the other channels. Metal samples the native depth texture as a
    // scalar float in .x; XYZW identity (0xFAC) is common for R32 views that only read .x/.r.
    let supported_select = matches!(destination_select, 0 | 0x004 | 0x204 | 0x924 | 0xfac);
    formats_are_depth_sample_views
        && supported_select
        && guest_texture_descriptors_have_same_byte_layout(rendered, sampled)
        && guest_texture_descriptors_share_byte_storage(rendered, sampled)
}

pub(crate) fn guest_texture_descriptors_share_byte_storage(
    left: GuestTextureDescriptor,
    right: GuestTextureDescriptor,
) -> bool {
    let Some(byte_count) = guest_texture_byte_count(left) else {
        return false;
    };
    if guest_texture_byte_count(right) != Some(byte_count) {
        return false;
    }
    if left.address == right.address {
        return true;
    }
    let Some(left_mapping) = guest_shared_mapping_identity(left.address, byte_count) else {
        return false;
    };
    let Some(right_mapping) = guest_shared_mapping_identity(right.address, byte_count) else {
        return false;
    };
    left_mapping == right_mapping
}

pub(crate) fn guest_texture_descriptors_have_same_layout(
    left: GuestTextureDescriptor,
    right: GuestTextureDescriptor,
) -> bool {
    guest_texture_descriptors_have_same_byte_layout(left, right)
        && left.pixel_format == right.pixel_format
}

pub(crate) fn guest_texture_descriptors_have_same_byte_layout(
    left: GuestTextureDescriptor,
    right: GuestTextureDescriptor,
) -> bool {
    left.width == right.width
        && left.height == right.height
        && left.pitch == right.pitch
        && left.tile_mode == right.tile_mode
        && left.bytes_per_pixel == right.bytes_per_pixel
}

pub(crate) fn render_resolve_candidate_matches(
    sampled: GuestTextureDescriptor,
    rendered: GuestTextureDescriptor,
    sample_count: u32,
) -> bool {
    if !guest_texture_descriptors_have_same_layout(sampled, rendered) {
        return false;
    }
    if guest_texture_descriptors_overlap(sampled, rendered) {
        return true;
    }
    // Gen5's shader-readable single-sample view commonly follows the multisample allocation in
    // guest memory. GNM allocators round that allocation to a large GPU page, so the SRD base is
    // just beyond the raw sample storage rather than overlapping CB_COLOR_BASE. This relationship
    // is narrow enough to retain allocation identity while rejecting unrelated same-size targets.
    const GPU_LARGE_PAGE_BYTES: u64 = 2 * 1_024 * 1_024;
    let Some(multisample_bytes) = guest_texture_byte_count(rendered)
        .and_then(|bytes| bytes.checked_mul(u64::from(sample_count)))
    else {
        return false;
    };
    let Some(rounded_bytes) = multisample_bytes
        .checked_add(GPU_LARGE_PAGE_BYTES - 1)
        .map(|bytes| bytes & !(GPU_LARGE_PAGE_BYTES - 1))
    else {
        return false;
    };
    let Some(minimum_resolve_address) = rendered.address.checked_add(multisample_bytes) else {
        return false;
    };
    let Some(maximum_resolve_address) = rendered.address.checked_add(rounded_bytes) else {
        return false;
    };
    (minimum_resolve_address..=maximum_resolve_address).contains(&sampled.address)
}

pub(crate) fn guest_ranges_overlap(
    left: u64,
    left_bytes: u64,
    right: u64,
    right_bytes: u64,
) -> bool {
    let left_end = left.saturating_add(left_bytes);
    let right_end = right.saturating_add(right_bytes);
    left < right_end && right < left_end
}

pub(crate) fn guest_physical_ranges_overlap(
    left: u64,
    left_bytes: u64,
    right: u64,
    right_bytes: u64,
) -> bool {
    if guest_ranges_overlap(left, left_bytes, right, right_bytes) {
        return true;
    }
    remu_memory::guest_shared_backing_ranges_overlap(left, left_bytes, right, right_bytes)
}
