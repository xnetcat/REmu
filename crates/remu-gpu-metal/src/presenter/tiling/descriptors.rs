// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::mode5::mode5_mip_chain_layout;
use super::mode27::mode64k_2d_mip_chain_layout;
use super::{GuestTextureDescriptor, guest_texture_byte_count};
use metal::MTLPixelFormat;
use remu_memory::{GuestSharedMappingIdentity, guest_shared_mapping_identity};

/// The guest bytes a decoded resource names, resolved once when its descriptor is decoded.
///
/// `backing` is the GPU-physical identity of `address .. address + len`; it is absent when the
/// range has no complete live shared mapping. Resolving it once keeps the process-global
/// shared-mapping registry lock off every per-draw candidate pair, and the admission-time
/// identity keeps naming the same physical storage if the original VA is later rebound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResourceSpan {
    address: u64,
    len: u64,
    backing: Option<GuestSharedMappingIdentity>,
}

impl ResourceSpan {
    /// Resolves a texture descriptor's span, or `None` when its byte count overflows.
    ///
    /// `None` is the same answer the overlap predicate gives such a descriptor: it can never
    /// overlap anything, so it is never admitted to an interval index either.
    pub(crate) fn for_texture(descriptor: GuestTextureDescriptor) -> Option<Self> {
        Some(Self::new(
            descriptor.address,
            guest_texture_byte_count(descriptor)?,
        ))
    }

    /// Resolves the complete allocation named by a 2D 64K mip hierarchy.
    pub(crate) fn for_mode64k_2d_mip_chain(
        descriptor: GuestTextureDescriptor,
        mip_level_count: u32,
    ) -> Result<Self, String> {
        let layout = mode64k_2d_mip_chain_layout(descriptor, mip_level_count)?;
        let byte_count = u64::try_from(layout.byte_count)
            .map_err(|_| "64 KiB mip-chain span does not fit u64")?;
        Ok(Self::new(descriptor.address, byte_count))
    }

    /// Resolves exactly the guest bytes selected by an image view.
    ///
    /// `descriptor.address` remains the allocation base from the SRD. Array/cube views select an
    /// absolute slice range through `base_array_slice`; their returned span starts at the first
    /// selected face rather than silently naming the allocation prefix.
    pub(crate) fn for_image_view(
        descriptor: GuestTextureDescriptor,
        base_mip_level: u32,
        mip_level_count: u32,
        texture_type: u32,
        base_array_slice: u32,
        layer_count: u32,
    ) -> Result<Option<Self>, String> {
        let underlying_mip_count = base_mip_level
            .checked_add(mip_level_count)
            .ok_or("image underlying mip count overflows")?;
        let face_byte_count = if underlying_mip_count > 1 && descriptor.tile_mode == 5 {
            if !matches!(texture_type, 9 | 11 | 13) {
                return Err(format!(
                    "exact mode-5 image view requires type 9, 11, or 13, got {texture_type}"
                ));
            }
            u64::try_from(mode5_mip_chain_layout(descriptor, underlying_mip_count)?.byte_count)
                .map_err(|_| "mode-5 image face byte count does not fit u64")?
        } else if underlying_mip_count > 1 && matches!(descriptor.tile_mode, 9 | 27) {
            if texture_type != 9 || layer_count != 1 {
                return Err(format!(
                    "mode-{} exact mip layout does not support layered or 3D mip views (type={texture_type} layers={layer_count})",
                    descriptor.tile_mode,
                ));
            }
            if super::super::metal_uncompressed_bytes_per_pixel(descriptor.pixel_format)
                != Some(descriptor.bytes_per_pixel)
            {
                return Err(format!(
                    "mode-{} exact mip layout requires an uncompressed Metal format",
                    descriptor.tile_mode,
                ));
            }
            u64::try_from(mode64k_2d_mip_chain_layout(descriptor, underlying_mip_count)?.byte_count)
                .map_err(|_| "64 KiB image face byte count does not fit u64")?
        } else {
            guest_texture_byte_count(descriptor).ok_or("image face byte count overflows")?
        };
        let selected_base = if matches!(texture_type, 11 | 13) {
            descriptor
                .address
                .checked_add(
                    face_byte_count
                        .checked_mul(u64::from(base_array_slice))
                        .ok_or("image base-array byte offset overflows")?,
                )
                .ok_or("image selected address overflows")?
        } else {
            if base_array_slice != 0 {
                return Err(format!(
                    "texture type {texture_type} cannot select base array slice {base_array_slice}"
                ));
            }
            descriptor.address
        };
        let selected_faces = if matches!(texture_type, 10 | 11 | 13) {
            layer_count
        } else {
            1
        };
        let selected_byte_count = face_byte_count
            .checked_mul(u64::from(selected_faces))
            .ok_or("image selected byte count overflows")?;
        Ok(Some(Self::new(selected_base, selected_byte_count)))
    }

    pub(crate) fn new(address: u64, len: u64) -> Self {
        Self {
            address,
            len,
            backing: remu_memory::guest_shared_backing_identity(address, len),
        }
    }

    pub(crate) fn address(&self) -> u64 {
        self.address
    }

    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    pub(crate) fn end(&self) -> u64 {
        self.address.saturating_add(self.len)
    }

    pub(crate) fn backing(&self) -> Option<GuestSharedMappingIdentity> {
        self.backing
    }

    /// First offset of this span in the space `RangeSet` keeps its runs in.
    ///
    /// A resolved backing gives a physical offset, which is what makes two virtual aliases of one
    /// resource comparable byte for byte. A span with no complete live backing has no physical
    /// identity to share, so it stays in virtual-address space and the range algebra only ever
    /// compares it against itself.
    pub(crate) fn range_start(&self) -> u64 {
        self.backing
            .map_or(self.address, |backing| backing.backing_offset)
    }

    pub(crate) fn range_end(&self) -> u64 {
        self.range_start().saturating_add(self.len)
    }

    /// True when both spans keep their `RangeSet` runs in the same offset space.
    ///
    /// Runs of two resolved spans are only comparable inside one backing object; an unresolved
    /// span stays in guest virtual space, where only another unresolved span can meet it.
    pub(crate) fn shares_offset_space(&self, other: &Self) -> bool {
        match (self.backing, other.backing) {
            (Some(left), Some(right)) => left.backing_id == right.backing_id,
            (None, None) => true,
            _ => false,
        }
    }

    #[cfg(test)]
    pub(crate) fn overlaps_texture(&self, other: GuestTextureDescriptor) -> bool {
        Self::for_texture(other).is_some_and(|other| self.overlaps(&other))
    }

    /// Reports overlap against a span whose physical identity was resolved previously.
    ///
    /// This is the whole overlap predicate `guest_texture_descriptors_overlap` implements, with
    /// both identities supplied instead of resolved per call. The fallback is asymmetric — the
    /// registry only walks aliases of the left operand's backing — so callers must keep the
    /// operand order the linear scan they replace used.
    pub(crate) fn overlaps(&self, other: &Self) -> bool {
        if guest_ranges_overlap(self.address, self.len, other.address, other.len) {
            return true;
        }
        let (Some(left_backing), Some(right_backing)) = (self.backing, other.backing) else {
            return remu_memory::guest_shared_backing_ranges_overlap(
                self.address,
                self.len,
                other.address,
                other.len,
            );
        };
        left_backing.backing_id == right_backing.backing_id
            && guest_ranges_overlap(
                left_backing.backing_offset,
                self.len,
                right_backing.backing_offset,
                other.len,
            )
    }
}

/// Overlap between two spans resolved at decode, in `left.overlaps(right)` operand order.
///
/// A `None` operand names a descriptor whose byte count overflows, which the predicate answers
/// false for against everything.
pub(crate) fn spans_overlap(left: Option<ResourceSpan>, right: Option<ResourceSpan>) -> bool {
    left.zip(right)
        .is_some_and(|(left, right)| left.overlaps(&right))
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
    let formats_are_view_compatible = pixel_format_view_class(left.pixel_format)
        .zip(pixel_format_view_class(right.pixel_format))
        .is_some_and(|(left, right)| left == right);
    let layout_compatible = guest_texture_descriptors_have_same_byte_layout(left, right)
        || (left.address == right.address
            && left.width == right.width
            && left.height == right.height
            && left.tile_mode == right.tile_mode
            && left.bytes_per_pixel == right.bytes_per_pixel
            && left.pitch >= left.width
            && right.pitch >= right.width);
    left.pixel_format != right.pixel_format
        && formats_are_view_compatible
        && layout_compatible
        && guest_texture_descriptors_share_byte_storage(left, right)
}

/// Metal's pixel-format view class for `format`, or `None` when it cannot back a format view.
///
/// `newTextureViewWithPixelFormat:` accepts a different format only within one class, where every
/// member shares the channel count and channel width and differs solely in how the sampler
/// interprets those bits. That is exactly the reinterpretation a Prospero SRD asks for when it
/// describes a render target's bytes with another numeric format, so a view reproduces the CPU
/// decode path's result without reading the target back through guest memory. Octopath renders its
/// occlusion chain to `R8Uint` targets and samples them as `R8Unorm`; routing those six surfaces
/// through the writeback path cost hundreds of megabytes and roughly a second of presenter time
/// per interval.
///
/// Depth and stencil formats are deliberately absent: they are not view-compatible with colour
/// formats, and `guest_texture_descriptors_share_depth_sample_storage` already models the native
/// depth-plane aliases Metal does allow.
fn pixel_format_view_class(format: MTLPixelFormat) -> Option<PixelFormatViewClass> {
    use PixelFormatViewClass as Class;
    Some(match format {
        MTLPixelFormat::R8Unorm
        | MTLPixelFormat::R8Unorm_sRGB
        | MTLPixelFormat::R8Snorm
        | MTLPixelFormat::R8Uint
        | MTLPixelFormat::R8Sint => Class::OneChannel8,
        MTLPixelFormat::R16Unorm
        | MTLPixelFormat::R16Snorm
        | MTLPixelFormat::R16Uint
        | MTLPixelFormat::R16Sint
        | MTLPixelFormat::R16Float => Class::OneChannel16,
        MTLPixelFormat::R32Uint | MTLPixelFormat::R32Sint | MTLPixelFormat::R32Float => {
            Class::OneChannel32
        }
        MTLPixelFormat::RG8Unorm
        | MTLPixelFormat::RG8Unorm_sRGB
        | MTLPixelFormat::RG8Snorm
        | MTLPixelFormat::RG8Uint
        | MTLPixelFormat::RG8Sint => Class::TwoChannel8,
        MTLPixelFormat::RG16Unorm
        | MTLPixelFormat::RG16Snorm
        | MTLPixelFormat::RG16Uint
        | MTLPixelFormat::RG16Sint
        | MTLPixelFormat::RG16Float => Class::TwoChannel16,
        MTLPixelFormat::RG32Uint | MTLPixelFormat::RG32Sint | MTLPixelFormat::RG32Float => {
            Class::TwoChannel32
        }
        MTLPixelFormat::RGBA8Unorm
        | MTLPixelFormat::RGBA8Unorm_sRGB
        | MTLPixelFormat::RGBA8Snorm
        | MTLPixelFormat::RGBA8Uint
        | MTLPixelFormat::RGBA8Sint
        | MTLPixelFormat::BGRA8Unorm
        | MTLPixelFormat::BGRA8Unorm_sRGB => Class::FourChannel8,
        MTLPixelFormat::RGBA16Unorm
        | MTLPixelFormat::RGBA16Snorm
        | MTLPixelFormat::RGBA16Uint
        | MTLPixelFormat::RGBA16Sint
        | MTLPixelFormat::RGBA16Float => Class::FourChannel16,
        MTLPixelFormat::RGBA32Uint | MTLPixelFormat::RGBA32Sint | MTLPixelFormat::RGBA32Float => {
            Class::FourChannel32
        }
        _ => return None,
    })
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum PixelFormatViewClass {
    OneChannel8,
    OneChannel16,
    OneChannel32,
    TwoChannel8,
    TwoChannel16,
    TwoChannel32,
    FourChannel8,
    FourChannel16,
    FourChannel32,
}

/// Prospero depth SRDs reinterpret the native DB as R16/R32. That sample view must never become
/// a competing GPU writer that demotes the Depth* resident — Metal still holds Z in the DB
/// texture, and later lighting/tonemap ImageLoads would otherwise bind an empty R32 shadow
/// (`depth_native=0`, magnitude 0) after a real depth prepass.
pub(crate) fn depth_sample_view_must_not_steal_depth_authority(
    writer: GuestTextureDescriptor,
    depth_candidate: GuestTextureDescriptor,
) -> bool {
    let formats_are_depth_sample_views = matches!(
        (depth_candidate.pixel_format, writer.pixel_format),
        (MTLPixelFormat::Depth16Unorm, MTLPixelFormat::R16Unorm)
            | (MTLPixelFormat::Depth32Float, MTLPixelFormat::R32Float)
    );
    formats_are_depth_sample_views
        && writer.width == depth_candidate.width
        && writer.height == depth_candidate.height
        && writer.tile_mode == depth_candidate.tile_mode
        && writer.bytes_per_pixel == depth_candidate.bytes_per_pixel
        && guest_texture_descriptors_share_byte_storage(writer, depth_candidate)
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
    // Poppy title samples Depth32Float residents through an R32Float SRD whose pitch field is
    // the visible width (2848) while the depth target keeps RDNA2 padded pitch (2944). Same
    // address + extent still identifies the native Z plane; forcing CPU writeback here costs
    // ~75 MiB/frame and still feeds the sampler from a pitch-mismatched decode.
    let layout_compatible = guest_texture_descriptors_have_same_byte_layout(rendered, sampled)
        || (rendered.address == sampled.address
            && rendered.width == sampled.width
            && rendered.height == sampled.height
            && rendered.tile_mode == sampled.tile_mode
            && rendered.bytes_per_pixel == sampled.bytes_per_pixel
            && rendered.pitch >= rendered.width
            && sampled.pitch >= sampled.width);
    formats_are_depth_sample_views
        && supported_select
        && layout_compatible
        && guest_texture_descriptors_share_byte_storage(rendered, sampled)
}

pub(crate) fn guest_texture_descriptors_share_byte_storage(
    left: GuestTextureDescriptor,
    right: GuestTextureDescriptor,
) -> bool {
    // Same virtual address always names one allocation. CB vs SRD pitch disagreements must not
    // force a CPU writeback when a Metal resident already holds the authoritative image.
    if left.address == right.address {
        return true;
    }
    let Some(byte_count) = guest_texture_byte_count(left) else {
        return false;
    };
    if guest_texture_byte_count(right) != Some(byte_count) {
        return false;
    }
    let Some(left_mapping) = guest_shared_mapping_identity(left.address, byte_count) else {
        return false;
    };
    let Some(right_mapping) = guest_shared_mapping_identity(right.address, byte_count) else {
        return false;
    };
    left_mapping == right_mapping
}

/// True when a GPU-authored resident can be sampled through an SRD that may disagree on pitch.
///
/// Pitch only matters for CPU tiling decode. Once the image lives in a Metal resident, same
/// format + extent + physical storage is enough to bind it directly and skip alias writeback.
pub(crate) fn guest_texture_descriptors_share_gpu_sample_storage(
    rendered: GuestTextureDescriptor,
    sampled: GuestTextureDescriptor,
) -> bool {
    if guest_texture_descriptors_share_storage(rendered, sampled) {
        return true;
    }
    rendered.pixel_format == sampled.pixel_format
        && rendered.width == sampled.width
        && rendered.height == sampled.height
        && rendered.tile_mode == sampled.tile_mode
        && rendered.bytes_per_pixel == sampled.bytes_per_pixel
        && rendered.pitch >= rendered.width
        && sampled.pitch >= sampled.width
        && guest_texture_descriptors_share_byte_storage(rendered, sampled)
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
