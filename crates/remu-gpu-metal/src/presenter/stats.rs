// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    GuestTextureDescriptor, Mode27WorkAggregate, Mode27WorkKey, Mode27WorkOrigin,
    PresenterStatistics, TextureDirtyWorkAggregate, TextureSnapshotWorkKey,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

pub(super) static TEXTURE_SNAPSHOT_COUNT: AtomicU64 = AtomicU64::new(0);

pub(super) static TEXTURE_SNAPSHOT_BYTES: AtomicU64 = AtomicU64::new(0);

pub(super) static TEXTURE_SNAPSHOT_NANOSECONDS: AtomicU64 = AtomicU64::new(0);

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

pub(super) static RESIDENT_DEPTH_SAMPLE_HITS: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_ALIAS_WRITEBACK_COUNT: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_ALIAS_WRITEBACK_BYTES: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_ALIAS_WRITEBACK_NANOSECONDS: AtomicU64 = AtomicU64::new(0);

pub(super) static PIPELINE_COMPILE_ENQUEUED: AtomicU64 = AtomicU64::new(0);

pub(super) static PIPELINE_COMPILE_QUEUE_FULL: AtomicU64 = AtomicU64::new(0);

pub(super) static PIPELINE_COMPILE_COMPLETED: AtomicU64 = AtomicU64::new(0);

pub(super) static PIPELINE_COMPILE_FAILED: AtomicU64 = AtomicU64::new(0);

pub(super) static PIPELINE_DRAW_GATES: AtomicU64 = AtomicU64::new(0);

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

pub(super) fn process_resident_bytes() -> Option<u64> {
    let mut task = std::mem::MaybeUninit::<libc::proc_taskinfo>::zeroed();
    let byte_count = std::mem::size_of::<libc::proc_taskinfo>();
    let byte_count = i32::try_from(byte_count).ok()?;
    // SAFETY: `task` names writable storage of exactly the size passed to `proc_pidinfo` for the
    // current process and is read only after libproc reports a complete `PROC_PIDTASKINFO` result.
    let written = unsafe {
        libc::proc_pidinfo(
            i32::try_from(std::process::id()).ok()?,
            libc::PROC_PIDTASKINFO,
            0,
            task.as_mut_ptr().cast::<c_void>(),
            byte_count,
        )
    };
    if written != byte_count {
        return None;
    }
    // SAFETY: A complete result initialized every field of `proc_taskinfo` above.
    Some(unsafe { task.assume_init() }.pti_resident_size)
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
            guest_flips: 0,
            completed_presents: 0,
            submitted_guest_batches: 0,
            coalesced_pipeline_barriers: 0,
            encoded_draws: 0,
            encoded_compute_dispatches: 0,
            encoded_render_passes: 0,
            render_pass_ends_for_compute: 0,
            render_pass_ends_for_target_change: 0,
            render_pass_ends_for_batch_flush: 0,
            batch_flushes_from_event_loop: 0,
            batch_flushes_for_guest_memory_write: 0,
            batch_flushes_for_blocked_wait: 0,
            batch_flushes_for_flip_snapshot: 0,
            batch_flushes_for_work_capacity: 0,
            batch_flushes_for_command_processor_barrier: 0,
            batch_flushes_for_cpu_synchronization: 0,
            gpu_execution_time: Duration::ZERO,
            presentation_queue_latency: Duration::ZERO,
            last_audio_output_frames: remu_gpu::audio_output_frames(),
            last_audio_underrun_frames: remu_gpu::audio_underrun_frames(),
        }
    }
}
