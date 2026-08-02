// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    GuestTextureDescriptor, Mode27WorkAggregate, Mode27WorkKey, Mode27WorkOrigin,
    PresenterStatistics, TextureDirtyWorkAggregate, TextureSnapshotWorkKey,
    WRITEBACK_TRIGGER_COUNT,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ProcessMemoryStatistics {
    pub(super) resident_bytes: u64,
    pub(super) physical_footprint_bytes: u64,
    pub(super) peak_physical_footprint_bytes: u64,
}

unsafe extern "C" {
    #[link_name = "proc_pid_rusage"]
    fn process_pid_rusage(
        pid: libc::c_int,
        flavor: libc::c_int,
        buffer: *mut libc::c_void,
    ) -> libc::c_int;
}

pub(super) static TEXTURE_SNAPSHOT_COUNT: AtomicU64 = AtomicU64::new(0);

pub(super) static TEXTURE_SNAPSHOT_BYTES: AtomicU64 = AtomicU64::new(0);

pub(super) static TEXTURE_SNAPSHOT_NANOSECONDS: AtomicU64 = AtomicU64::new(0);

/// Snapshots published while a guest thread was still writing the range.
pub(super) static TEXTURE_SNAPSHOT_UNSTABLE_COUNT: AtomicU64 = AtomicU64::new(0);

pub(super) static TEXTURE_DIRTY_VALIDATION_COUNT: AtomicU64 = AtomicU64::new(0);

pub(super) static TEXTURE_DIRTY_VALIDATION_BYTES: AtomicU64 = AtomicU64::new(0);

pub(super) static TEXTURE_DIRTY_VALIDATION_NANOSECONDS: AtomicU64 = AtomicU64::new(0);

pub(super) static TEXTURE_DIRTY_VALIDATION_UNCHANGED: AtomicU64 = AtomicU64::new(0);

pub(super) static MODE27_DECODE_COUNT: AtomicU64 = AtomicU64::new(0);

pub(super) static MODE27_DECODE_BYTES: AtomicU64 = AtomicU64::new(0);

pub(super) static MODE27_DECODE_NANOSECONDS: AtomicU64 = AtomicU64::new(0);

pub(super) static MIP_UPLOAD_COUNT: AtomicU64 = AtomicU64::new(0);

pub(super) static MIP_UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);

pub(super) static MIP_UPLOAD_NANOSECONDS: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_TEXTURE_CACHE_HITS: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_TEXTURE_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_TEXTURE_CACHE_DIRTY_RELOADS: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_TEXTURE_CACHE_ACQUIRE_INVALIDATIONS: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_TEXTURE_CACHE_CAPACITY_EVICTIONS: AtomicU64 = AtomicU64::new(0);

pub(super) static RESIDENT_TEXTURE_ACQUIRE_INVALIDATIONS: AtomicU64 = AtomicU64::new(0);

/// GPU writes that reached the mark path without the resident the ensure path served.
///
/// Resolving a written target by exact key or shared storage requires an identical pitch, while
/// the ensure path also accepts a pitch-mismatched GPU-sample alias. Any count here is a write
/// whose `gpu_written`, `content_version` and activity never reached a resident that exists.
pub(super) static RESIDENT_WRITE_ATTRIBUTION_MISSES: AtomicU64 = AtomicU64::new(0);

pub(super) static RESIDENT_DEPTH_SAMPLE_HITS: AtomicU64 = AtomicU64::new(0);

/// Times a range set widened its claim instead of tracking every run exactly.
///
/// Widening an authority claim can only cost an extra re-materialisation, never stale pixels, so
/// this is a throughput signal rather than a correctness one. Counted for the `RANGE_SET_MAX_RUNS`
/// collapse and for the layout cases whose rect refinement is deliberately not attempted.
pub(super) static OWNED_RANGE_SET_COARSENED: AtomicU64 = AtomicU64::new(0);

/// Residents admitted with no complete live shared backing for their declared range.
///
/// Their ranges stay in virtual-address space and are never compared against a resident that did
/// resolve, so they are excluded from the cross-resident algebra entirely.
pub(super) static OWNED_UNRESOLVED_SPAN: AtomicU64 = AtomicU64::new(0);

/// Written-region computations that fell back to the writer's whole declared span.
///
/// Every failure inside `written_byte_ranges` lands here, which is exactly the descriptor-level
/// range the code claimed before the range algebra existed.
pub(super) static OWNED_FALLBACK_FULL_RANGE: AtomicU64 = AtomicU64::new(0);

/// Residents an intersecting write reached through the index but did not take a byte from.
///
/// This is the direct measure of pooled tenants surviving a neighbour's write: every count is a
/// resident the descriptor-overlap rule would have destroyed.
pub(super) static OWNED_SUBTRACT_TENANTS_SURVIVED: AtomicU64 = AtomicU64::new(0);

/// Residents that lost their last owned byte and moved to `Stale` in place.
pub(super) static OWNED_DEMOTIONS: AtomicU64 = AtomicU64::new(0);

/// Demoted residents re-taken by a bind that overwrites every texel, at no copy cost.
pub(super) static OWNED_RECLAIM_DISCARD: AtomicU64 = AtomicU64::new(0);

/// Demoted allocations restored by reusing a compatible Metal texture view.
pub(super) static OWNED_REMATERIALISE_VIEW: AtomicU64 = AtomicU64::new(0);

/// Demoted allocations restored by an ordered texture-to-texture copy.
pub(super) static OWNED_REMATERIALISE_BLIT: AtomicU64 = AtomicU64::new(0);

/// Demoted allocations reconstructed by ordered GPU retile/de-tile work.
pub(super) static OWNED_REMATERIALISE_DETILE: AtomicU64 = AtomicU64::new(0);

/// Demoted allocations rebuilt from guest memory after proving RAM is authoritative.
pub(super) static OWNED_REMATERIALISE_CPU: AtomicU64 = AtomicU64::new(0);

/// GPU-current stale reads for which no lossless rematerialisation rung was available.
pub(super) static OWNED_REMATERIALISE_FAILED: AtomicU64 = AtomicU64::new(0);

/// Failure reasons are deliberately separate from the total so a run says which rung to build.
pub(super) static OWNED_REMATERIALISE_FAILED_MAPPING: AtomicU64 = AtomicU64::new(0);
pub(super) static OWNED_REMATERIALISE_FAILED_CPU_GPU: AtomicU64 = AtomicU64::new(0);
pub(super) static OWNED_REMATERIALISE_FAILED_LAYOUT: AtomicU64 = AtomicU64::new(0);
pub(super) static OWNED_REMATERIALISE_FAILED_METAL: AtomicU64 = AtomicU64::new(0);

/// Resident writebacks that reached guest memory, indexed by `WritebackTrigger`.
pub(super) static RESIDENT_WRITEBACK_PUBLISHED: [AtomicU64; WRITEBACK_TRIGGER_COUNT] =
    [const { AtomicU64::new(0) }; WRITEBACK_TRIGGER_COUNT];

/// Resident writebacks refused admission, indexed by `WritebackTrigger`.
///
/// Every count is a host→guest copy REmu declined to make into a range it could not prove still
/// belongs to the texture. The published counters above are the remaining exposure.
pub(super) static RESIDENT_WRITEBACK_DISCARDED: [AtomicU64; WRITEBACK_TRIGGER_COUNT] =
    [const { AtomicU64::new(0) }; WRITEBACK_TRIGGER_COUNT];

pub(super) static RESIDENT_WRITEBACK_PUBLISHED_BYTES: AtomicU64 = AtomicU64::new(0);

pub(super) static RESIDENT_WRITEBACK_DISCARDED_BYTES: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_ALIAS_WRITEBACK_COUNT: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_ALIAS_WRITEBACK_BYTES: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_ALIAS_WRITEBACK_NANOSECONDS: AtomicU64 = AtomicU64::new(0);

pub(super) static PIPELINE_COMPILE_ENQUEUED: AtomicU64 = AtomicU64::new(0);

pub(super) static PIPELINE_COMPILE_QUEUE_FULL: AtomicU64 = AtomicU64::new(0);

pub(super) static PIPELINE_COMPILE_COMPLETED: AtomicU64 = AtomicU64::new(0);

pub(super) static PIPELINE_COMPILE_FAILED: AtomicU64 = AtomicU64::new(0);

pub(super) static PIPELINE_DRAW_GATES: AtomicU64 = AtomicU64::new(0);
pub(super) static PIPELINE_COMPUTE_GATES: AtomicU64 = AtomicU64::new(0);

/// Draw packets consumed without rasterizing anything, by cause. Every one of these paths
/// reports success, so no drop counter or once-per-key failure log ever sees them; the periodic
/// `[REMu][PERF-NOOP]` line is their only footprint. Most are correct emulation (trigger
/// packets, empty scissors), which is exactly why a misclassification here — real geometry
/// deleted with zero trace — needs a number to falsify against.
pub(super) static NOOP_DRAWS_COVER_PLACEHOLDER: AtomicU64 = AtomicU64::new(0);
pub(super) static NOOP_DRAWS_RASTER_DISABLED: AtomicU64 = AtomicU64::new(0);
pub(super) static NOOP_DRAWS_COLOR_OPERATION: AtomicU64 = AtomicU64::new(0);
pub(super) static NOOP_DRAWS_EMPTY_SCISSOR: AtomicU64 = AtomicU64::new(0);
pub(super) static NOOP_DRAWS_MESH_DISPATCH_SKIPPED: AtomicU64 = AtomicU64::new(0);
pub(super) static NOOP_DRAWS_FAST_CLEAR_NOT_PENDING: AtomicU64 = AtomicU64::new(0);

pub(super) fn record_texture_work(
    count: &AtomicU64,
    bytes: &AtomicU64,
    nanoseconds: &AtomicU64,
    byte_count: usize,
    elapsed: Duration,
) {
    count.fetch_add(1, Ordering::Relaxed);
    bytes.fetch_add(
        u64::try_from(byte_count).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    nanoseconds.fetch_add(
        u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
}

pub(super) fn process_memory_statistics() -> Option<ProcessMemoryStatistics> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: `usage` names writable storage for the RUSAGE_INFO_V4 flavor. A zero result
    // guarantees that libproc initialized the complete structure before it is read below.
    let result = unsafe {
        process_pid_rusage(
            i32::try_from(std::process::id()).ok()?,
            libc::RUSAGE_INFO_V4,
            usage.as_mut_ptr().cast::<c_void>(),
        )
    };
    if result != 0 {
        return None;
    }
    // SAFETY: A successful RUSAGE_INFO_V4 query initialized every field of `usage`.
    let usage = unsafe { usage.assume_init() };
    Some(ProcessMemoryStatistics {
        resident_bytes: usage.ri_resident_size,
        physical_footprint_bytes: usage.ri_phys_footprint,
        peak_physical_footprint_bytes: usage.ri_lifetime_max_phys_footprint,
    })
}

#[cfg(test)]
pub(super) fn process_resident_bytes() -> Option<u64> {
    process_memory_statistics().map(|statistics| statistics.resident_bytes)
}

pub(super) static MODE27_WORK_DETAILS: OnceLock<
    Mutex<HashMap<Mode27WorkKey, Mode27WorkAggregate>>,
> = OnceLock::new();

pub(super) static TEXTURE_SNAPSHOT_WORK_DETAILS: OnceLock<
    Mutex<HashMap<TextureSnapshotWorkKey, Mode27WorkAggregate>>,
> = OnceLock::new();

pub(super) static TEXTURE_DIRTY_WORK_DETAILS: OnceLock<
    Mutex<HashMap<TextureSnapshotWorkKey, TextureDirtyWorkAggregate>>,
> = OnceLock::new();

pub(super) fn record_mode27_work_detail(
    origin: Mode27WorkOrigin,
    descriptor: GuestTextureDescriptor,
    mip_level_count: u32,
    byte_count: usize,
    elapsed: Duration,
) {
    let key = Mode27WorkKey {
        origin,
        descriptor,
        mip_level_count,
    };
    let mut details = MODE27_WORK_DETAILS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let aggregate = details.entry(key).or_default();
    aggregate.count = aggregate.count.saturating_add(1);
    aggregate.bytes = aggregate
        .bytes
        .saturating_add(u64::try_from(byte_count).unwrap_or(u64::MAX));
    aggregate.nanoseconds = aggregate
        .nanoseconds
        .saturating_add(u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX));
}

pub(super) fn record_texture_snapshot_work_detail(
    address: u64,
    byte_count: usize,
    caller_line: u32,
    elapsed: Duration,
) {
    let key = TextureSnapshotWorkKey {
        address,
        byte_count,
        caller_line,
    };
    let mut details = TEXTURE_SNAPSHOT_WORK_DETAILS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let aggregate = details.entry(key).or_default();
    aggregate.count = aggregate.count.saturating_add(1);
    aggregate.bytes = aggregate
        .bytes
        .saturating_add(u64::try_from(byte_count).unwrap_or(u64::MAX));
    aggregate.nanoseconds = aggregate
        .nanoseconds
        .saturating_add(u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX));
}

pub(super) fn take_texture_snapshot_work_details()
-> Vec<(TextureSnapshotWorkKey, Mode27WorkAggregate)> {
    let Some(details) = TEXTURE_SNAPSHOT_WORK_DETAILS.get() else {
        return Vec::new();
    };
    let mut details = details
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut snapshot = std::mem::take(&mut *details)
        .into_iter()
        .collect::<Vec<_>>();
    snapshot.sort_unstable_by_key(|(_, aggregate)| {
        std::cmp::Reverse((aggregate.nanoseconds, aggregate.bytes, aggregate.count))
    });
    snapshot
}

pub(super) fn record_texture_dirty_work_detail(
    address: u64,
    byte_count: usize,
    caller_line: u32,
    precise: bool,
    dirty_pages: usize,
    dirty_bytes: u64,
) {
    let key = TextureSnapshotWorkKey {
        address,
        byte_count,
        caller_line,
    };
    let mut details = TEXTURE_DIRTY_WORK_DETAILS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let aggregate = details.entry(key).or_default();
    aggregate.snapshots = aggregate.snapshots.saturating_add(1);
    aggregate.full_bytes = aggregate
        .full_bytes
        .saturating_add(u64::try_from(byte_count).unwrap_or(u64::MAX));
    if !precise {
        aggregate.imprecise_snapshots = aggregate.imprecise_snapshots.saturating_add(1);
    } else if dirty_pages == 0 {
        aggregate.clean_snapshots = aggregate.clean_snapshots.saturating_add(1);
    }
    aggregate.dirty_pages = aggregate
        .dirty_pages
        .saturating_add(u64::try_from(dirty_pages).unwrap_or(u64::MAX));
    aggregate.dirty_bytes = aggregate.dirty_bytes.saturating_add(dirty_bytes);
}

pub(super) fn take_texture_dirty_work_details()
-> Vec<(TextureSnapshotWorkKey, TextureDirtyWorkAggregate)> {
    let Some(details) = TEXTURE_DIRTY_WORK_DETAILS.get() else {
        return Vec::new();
    };
    let mut details = details
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut snapshot = std::mem::take(&mut *details)
        .into_iter()
        .collect::<Vec<_>>();
    snapshot.sort_unstable_by_key(|(_, aggregate)| {
        std::cmp::Reverse((
            aggregate.full_bytes,
            aggregate.dirty_bytes,
            aggregate.snapshots,
        ))
    });
    snapshot
}

pub(super) fn take_mode27_work_details() -> Vec<(Mode27WorkKey, Mode27WorkAggregate)> {
    let Some(details) = MODE27_WORK_DETAILS.get() else {
        return Vec::new();
    };
    let mut details = details
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut snapshot = std::mem::take(&mut *details)
        .into_iter()
        .collect::<Vec<_>>();
    snapshot.sort_unstable_by_key(|(_, aggregate)| {
        std::cmp::Reverse((aggregate.nanoseconds, aggregate.bytes, aggregate.count))
    });
    snapshot
}

impl PresenterStatistics {
    pub(super) fn new() -> Self {
        Self {
            interval_started: Instant::now(),
            last_logged: Instant::now(),
            last_flip_reconciliation: Instant::now(),
            guest_flips: 0,
            completed_presents: 0,
            submitted_guest_batches: 0,
            coalesced_pipeline_barriers: 0,
            encoded_draws: 0,
            encoded_compute_dispatches: 0,
            encoded_render_passes: 0,
            render_pass_ends_for_compute: 0,
            render_pass_ends_for_target_change: 0,
            render_pass_ends_for_color_operation: 0,
            render_pass_ends_for_batch_flush: 0,
            render_pass_key_transitions: Default::default(),
            batch_flushes_from_event_loop: 0,
            batch_flushes_for_guest_memory_write: 0,
            batch_flushes_for_blocked_wait: 0,
            batch_flushes_for_shared_event_host_effect: 0,
            batch_flushes_for_flip_snapshot: 0,
            batch_flushes_for_work_capacity: 0,
            batch_flushes_for_command_processor_barrier: 0,
            batch_flushes_for_cpu_synchronization: 0,
            shared_event_signals: 0,
            shared_event_waits: 0,
            shared_event_host_blocks: 0,
            shared_event_host_wakes: 0,
            shared_event_gpu_ordered_compute_dispatches: 0,
            shared_event_error_advances: 0,
            shared_event_fallbacks_no_exact_producer: 0,
            shared_event_fallbacks_timeline_overflow: 0,
            gpu_execution_time: Duration::ZERO,
            presentation_queue_latency: Duration::ZERO,
            last_audio_output_frames: remu_gpu::audio_output_frames(),
            last_audio_underrun_frames: remu_gpu::audio_underrun_frames(),
        }
    }
}
