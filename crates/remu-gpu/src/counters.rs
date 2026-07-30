// SPDX-License-Identifier: GPL-2.0-or-later

//! Process-global GPU queue depth, submission completion and audio callback counters.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::GpuFlipCompletion;

pub(crate) static GPU_EVENTS_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

pub(crate) static GPU_WORK_ITEMS_PENDING: AtomicUsize = AtomicUsize::new(0);

pub(crate) static GPU_FLIPS_PENDING: AtomicUsize = AtomicUsize::new(0);

pub(crate) static GPU_COMPLETED_SUBMISSION_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(crate) static GPU_COMPLETED_SUBMISSIONS: OnceLock<Mutex<BTreeSet<u64>>> = OnceLock::new();

pub(crate) static GPU_COMPLETED_FLIPS: OnceLock<Mutex<Vec<GpuFlipCompletion>>> = OnceLock::new();

static GPU_COMPLETION_NOTIFIER: OnceLock<fn()> = OnceLock::new();

pub(crate) static AUDIO_OUTPUT_FRAMES: AtomicU64 = AtomicU64::new(0);

pub(crate) static AUDIO_UNDERRUN_FRAMES: AtomicU64 = AtomicU64::new(0);

static PROGRESS_EPOCH: OnceLock<Instant> = OnceLock::new();

static GPU_LAST_PROGRESS_MILLIS: AtomicU64 = AtomicU64::new(0);

static AUDIO_LAST_PROGRESS_MILLIS: AtomicU64 = AtomicU64::new(0);

fn progress_timestamp_millis() -> u64 {
    u64::try_from(
        PROGRESS_EPOCH
            .get_or_init(Instant::now)
            .elapsed()
            .as_millis(),
    )
    .unwrap_or(u64::MAX)
    .saturating_add(1)
}

fn progress_age_millis(timestamp: &AtomicU64) -> Option<u64> {
    let timestamp = timestamp.load(Ordering::Acquire);
    (timestamp != 0).then(|| progress_timestamp_millis().saturating_sub(timestamp))
}

fn record_gpu_progress() {
    GPU_LAST_PROGRESS_MILLIS.store(progress_timestamp_millis(), Ordering::Release);
}

/// Returns the number of guest GPU events queued or blocked on the bounded presenter channel.
#[must_use]
pub fn gpu_events_in_flight() -> usize {
    GPU_EVENTS_IN_FLIGHT.load(Ordering::Relaxed)
}

/// Records that the presenter removed one event from the bounded guest GPU channel.
pub fn record_gpu_event_consumed() {
    let _ = GPU_EVENTS_IN_FLIGHT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
        Some(value.saturating_sub(1))
    });
    record_gpu_progress();
}

/// Returns the number of individual GPU work items emitted by the guest but not yet processed by
/// the presenter.
///
/// Unlike [`gpu_events_in_flight`], queue submissions contribute their decoded event count and
/// remain pending after the bounded channel has been drained. This is the useful measure of how
/// far guest rendering is running ahead of translation and presentation.
#[must_use]
pub fn gpu_work_items_pending() -> usize {
    GPU_WORK_ITEMS_PENDING.load(Ordering::Relaxed)
}

/// Returns the number of accepted guest flips which have not reached a terminal presentation
/// command buffer yet.
#[must_use]
pub fn gpu_flips_pending() -> usize {
    GPU_FLIPS_PENDING.load(Ordering::Relaxed)
}

/// Records that the presenter finished processing one individual guest GPU work item.
pub fn record_gpu_work_item_processed() {
    cancel_pending_gpu_work_items(1);
    record_gpu_progress();
}

pub(crate) fn cancel_pending_gpu_work_items(count: usize) {
    let _ = GPU_WORK_ITEMS_PENDING.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
        Some(value.saturating_sub(count))
    });
}

pub(crate) fn cancel_pending_gpu_flips(count: usize) {
    let _ = GPU_FLIPS_PENDING.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
        Some(value.saturating_sub(count))
    });
}

/// Adds one host audio callback's output and underrun frame counts.
pub fn report_audio_frames(output_frames: u64, underrun_frames: u64) {
    AUDIO_OUTPUT_FRAMES.fetch_add(output_frames, Ordering::Relaxed);
    AUDIO_UNDERRUN_FRAMES.fetch_add(underrun_frames, Ordering::Relaxed);
    if output_frames != 0 {
        AUDIO_LAST_PROGRESS_MILLIS.store(progress_timestamp_millis(), Ordering::Release);
    }
}

/// Returns the cumulative number of frames requested by the host audio device while a port was
/// open.
#[must_use]
pub fn audio_output_frames() -> u64 {
    AUDIO_OUTPUT_FRAMES.load(Ordering::Relaxed)
}

/// Returns the cumulative number of host audio frames affected by a guest-port underrun.
#[must_use]
pub fn audio_underrun_frames() -> u64 {
    AUDIO_UNDERRUN_FRAMES.load(Ordering::Relaxed)
}

/// Returns the age of the most recent host audio callback that requested output frames.
///
/// `None` means no audio stream has reached the host callback yet.
#[must_use]
pub fn audio_progress_age_millis() -> Option<u64> {
    progress_age_millis(&AUDIO_LAST_PROGRESS_MILLIS)
}

/// Returns the age of the most recent presenter ingress, work-item retirement, submission
/// completion, or flip completion.
///
/// This is diagnostic progress, not a completion fence. It lets bounded runs distinguish an idle
/// guest from a backend which still has outstanding work but has stopped retiring it.
#[must_use]
pub fn gpu_progress_age_millis() -> Option<u64> {
    progress_age_millis(&GPU_LAST_PROGRESS_MILLIS)
}

/// Returns the numerically newest guest submission observed complete by the backend.
///
/// This is telemetry only: hardware queues retire independently, so callers that deliver guest
/// completion notifications must use [`take_gpu_completed_submission_ids`] instead.
#[must_use]
pub fn gpu_completed_submission_sequence() -> u64 {
    GPU_COMPLETED_SUBMISSION_SEQUENCE.load(Ordering::Acquire)
}

/// Installs the process-lifetime callback used to wake the runtime when backend work completes.
///
/// Completion payloads remain in the exact-ID and flip queues below. The callback only announces
/// that a consumer should re-check those queues, so it cannot publish a guest completion early or
/// change ordering between hardware queues.
pub fn install_gpu_completion_notifier(notifier: fn()) {
    let _ = GPU_COMPLETION_NOTIFIER.set(notifier);
}

fn notify_gpu_completion_consumer() {
    if let Some(notifier) = GPU_COMPLETION_NOTIFIER.get() {
        notifier();
    }
}

/// Publishes completion of an ordered guest submission.
///
/// Backends may retire different hardware queues out of submission order. Exact IDs are retained
/// for the runtime while the scalar maximum remains available for telemetry.
pub fn record_gpu_submission_completed(sequence: u64) {
    GPU_COMPLETED_SUBMISSIONS
        .get_or_init(|| Mutex::new(BTreeSet::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(sequence);
    let _ = GPU_COMPLETED_SUBMISSION_SEQUENCE.fetch_max(sequence, Ordering::AcqRel);
    record_gpu_progress();
    notify_gpu_completion_consumer();
}

/// Drains the exact set of guest submission IDs completed since the previous runtime refresh.
///
/// Returning IDs rather than a global prefix prevents a blocked graphics queue from withholding
/// an independent async-compute interrupt, and prevents a newer completion from falsely
/// completing an older queue.
pub fn take_gpu_completed_submission_ids() -> Vec<u64> {
    let mut completed = GPU_COMPLETED_SUBMISSIONS
        .get_or_init(|| Mutex::new(BTreeSet::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    std::mem::take(&mut *completed).into_iter().collect()
}

/// Publishes completion of one display-buffer presentation.
pub fn record_gpu_flip_completed(completion: GpuFlipCompletion) {
    cancel_pending_gpu_flips(1);
    GPU_COMPLETED_FLIPS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(completion);
    record_gpu_progress();
    notify_gpu_completion_consumer();
}

/// Drains display-buffer presentations completed since the previous runtime refresh.
pub fn take_gpu_completed_flips() -> Vec<GpuFlipCompletion> {
    let mut completed = GPU_COMPLETED_FLIPS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    std::mem::take(&mut *completed)
}
