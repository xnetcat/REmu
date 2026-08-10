// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! J1: `MTLVisibilityResultBuffer` publication for Gen5 occlusion / pixel-pipe dumps.
//!
//! Runtime emits `GpuEvent::OcclusionQuery { destination }` (synthetic STEP counter deleted).
//! This module:
//! 1. Allocates one Shared visibility-result buffer per presenter.
//! 2. Arms `MTLVisibilityResultMode::Counting` on draws inside an open render pass.
//! 3. Publishes `PIXEL_PIPE_STAT_VALID | sample_count` into the 16-pair guest block.
//!
//! Metal reports one sample count for the pass; Gen5 has sixteen render-backend pairs
//! (`OcclusionQueryResults`, `drawcommandbuffer.h:385-392`). Reporting the same total in every
//! even qword (odd = padding) is the honest approximation from hardware-fidelity R-5 — not the
//! old fabricated STEP sequence.
//!
//! If visibility was never armed (dump with no intervening counted draws), publish `VALID | 0`
//! after a loud diagnostic — never invent a non-zero count.

use metal::{Buffer, Device, MTLResourceOptions, MTLVisibilityResultMode, RenderCommandEncoder};
use remu_gpu::{
    GpuMemoryRelease, GpuMemoryReleaseValue, PIXEL_PIPE_STAT_COUNTER_PAIRS, PIXEL_PIPE_STAT_VALID,
};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::super::metal_bytes::MetalBytes;

/// Bytes reserved for the presenter-wide visibility result buffer (one qword is enough for
/// Counting mode offset 0; keep a small pad for future multi-query offsets).
pub(crate) const fn visibility_result_buffer_byte_count() -> u64 {
    256
}

/// Ensures the presenter owns a Shared visibility-result buffer suitable for Counting mode.
pub(crate) fn ensure_visibility_result_buffer(
    device: &Device,
    existing: &mut Option<Buffer>,
) -> Buffer {
    if let Some(buffer) = existing.as_ref() {
        return buffer.clone();
    }
    let buffer = device.new_buffer(
        visibility_result_buffer_byte_count(),
        MTLResourceOptions::StorageModeShared,
    );
    buffer.set_label("REMu occlusion MTLVisibilityResultBuffer");
    // Zero so a dump before any counted draw publishes VALID|0 rather than stale host bytes.
    let bytes = MetalBytes::new(&buffer);
    if let Some(window) = bytes.slice_mut(0, 8) {
        window.fill(0);
    }
    *existing = Some(buffer.clone());
    buffer
}

/// Attaches the visibility buffer to a new render-pass descriptor.
pub(crate) fn attach_visibility_result_buffer(
    pass: &metal::RenderPassDescriptorRef,
    buffer: &Buffer,
) {
    pass.set_visibility_result_buffer(Some(buffer));
}

/// Arms Counting mode at offset 0 for draws that should contribute to the next dump.
pub(crate) fn arm_visibility_counting(encoder: &RenderCommandEncoder) {
    encoder.set_visibility_result_mode(MTLVisibilityResultMode::Counting, 0);
}

/// Reads the Metal sample count (or 0) and builds the 16 guest MemoryRelease publications.
///
/// `armed` is true when at least one draw in the preceding window called
/// [`arm_visibility_counting`]. When false, the count is forced to 0 and a loud diagnostic fires.
pub(crate) fn occlusion_counter_releases(
    visibility_buffer: Option<&Buffer>,
    destination: u64,
    armed: bool,
) -> Vec<GpuMemoryRelease> {
    let sample_count = if armed {
        read_visibility_sample_count(visibility_buffer)
    } else {
        static UNARMED: AtomicUsize = AtomicUsize::new(0);
        let sequence = UNARMED.fetch_add(1, Ordering::Relaxed) + 1;
        if sequence <= 16 || sequence.is_power_of_two() {
            eprintln!(
                "[REMu][METAL][J1] occlusion dump with no armed visibility counting sequence={sequence} destination={destination:#x}; publishing VALID|0 (not a fabricated STEP)"
            );
        }
        0
    };
    let value = PIXEL_PIPE_STAT_VALID | (sample_count & !PIXEL_PIPE_STAT_VALID);
    (0..PIXEL_PIPE_STAT_COUNTER_PAIRS)
        .map(|pair| {
            GpuMemoryRelease::label(
                destination + pair * 16,
                GpuMemoryReleaseValue::Immediate64(value),
            )
        })
        .collect()
}

fn read_visibility_sample_count(buffer: Option<&Buffer>) -> u64 {
    let Some(buffer) = buffer else {
        return 0;
    };
    let bytes = MetalBytes::new(buffer);
    let Some(window) = bytes.slice(0, 8) else {
        return 0;
    };
    let mut qword = [0_u8; 8];
    qword.copy_from_slice(window);
    u64::from_le_bytes(qword)
}

/// Publishes occlusion counter releases through the caller's MemoryRelease executor.
///
/// Separated so submission can close the active pass (flushing visibility into the buffer)
/// before reading, then reuse the ordinary guest-write path for ordering.
#[allow(dead_code)] // Helper kept for tests / alternate call sites; production uses MetalPresenter::publish_occlusion_query.
pub(crate) fn publish_occlusion_query_to_guest<E, F>(
    visibility_buffer: Option<&Buffer>,
    destination: u64,
    armed: bool,
    mut publish_release: F,
) -> Result<(), E>
where
    F: FnMut(GpuMemoryRelease) -> Result<(), E>,
{
    for release in occlusion_counter_releases(visibility_buffer, destination, armed) {
        publish_release(release)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unarmed_dump_publishes_valid_zero_not_synthetic_step() {
        let releases = occlusion_counter_releases(None, 0x1000, false);
        assert_eq!(releases.len(), PIXEL_PIPE_STAT_COUNTER_PAIRS as usize);
        for (pair, release) in releases.into_iter().enumerate() {
            assert_eq!(release.address, 0x1000 + pair as u64 * 16);
            assert_eq!(
                release.value,
                GpuMemoryReleaseValue::Immediate64(PIXEL_PIPE_STAT_VALID)
            );
        }
    }

    #[test]
    fn counter_pair_count_matches_sdk_occlusion_query_results() {
        // sce::Agc::OcclusionQueryResults::m_results[16] (drawcommandbuffer.h:392).
        assert_eq!(PIXEL_PIPE_STAT_COUNTER_PAIRS, 16);
        assert_eq!(visibility_result_buffer_byte_count() % 8, 0);
    }
}
