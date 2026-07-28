// SPDX-License-Identifier: GPL-2.0-or-later

//! The bounded, ordered guest GPU event channel and its process-global sink.

use std::collections::BTreeSet;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Mutex, OnceLock};

use crate::counters::{
    AUDIO_OUTPUT_FRAMES, AUDIO_UNDERRUN_FRAMES, GPU_COMPLETED_FLIPS,
    GPU_COMPLETED_SUBMISSION_SEQUENCE, GPU_COMPLETED_SUBMISSIONS, GPU_EVENTS_IN_FLIGHT,
    GPU_FLIPS_PENDING, GPU_WORK_ITEMS_PENDING, cancel_pending_gpu_flips,
    cancel_pending_gpu_work_items,
};
use crate::waits::{GPU_WAIT_CAPACITY, gpu_wait_registry, register_queue_submission_waits};
use crate::{GpuEvent, GpuMemoryAcquire, GpuPipelineBarrier, record_gpu_event_consumed};

/// Maximum number of submission chunks retained in the bounded channel.
///
/// A guest submission can contain thousands of decoded events, so bounding only the number of
/// outer `QueueSubmission` values does not bound retained guest state. [`emit_event`] splits those
/// submissions into [`GPU_EVENT_SUBMISSION_CHUNK_SIZE`] event chunks, making this limit effective
/// while preserving queue-local order and the final submission boundary.
pub const GPU_EVENT_QUEUE_CAPACITY: usize = 8;

/// Maximum decoded events placed in one bounded-channel submission chunk.
pub const GPU_EVENT_SUBMISSION_CHUNK_SIZE: usize = 32;

/// Maximum decoded events retained after the presenter moves chunks out of the channel.
///
/// The ordered receiver stops ingesting at this threshold. That leaves the bounded channel full
/// and applies backpressure to the guest command processor while cold shader compilation or large
/// texture uploads delay the consumer.
pub const GPU_PENDING_EVENT_CAPACITY: usize = 256;

static GPU_EVENT_SINK: OnceLock<Mutex<Option<SyncSender<GpuEvent>>>> = OnceLock::new();

/// Creates the bounded, ordered event channel used by the guest GPU command processor.
#[must_use]
pub fn gpu_event_channel() -> (SyncSender<GpuEvent>, Receiver<GpuEvent>) {
    std::sync::mpsc::sync_channel(GPU_EVENT_QUEUE_CAPACITY)
}

pub fn install_event_sink(sender: SyncSender<GpuEvent>) {
    GPU_EVENTS_IN_FLIGHT.store(0, Ordering::Relaxed);
    GPU_WORK_ITEMS_PENDING.store(0, Ordering::Relaxed);
    GPU_FLIPS_PENDING.store(0, Ordering::Relaxed);
    GPU_COMPLETED_SUBMISSION_SEQUENCE.store(0, Ordering::Release);
    GPU_COMPLETED_SUBMISSIONS
        .get_or_init(|| Mutex::new(BTreeSet::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    GPU_COMPLETED_FLIPS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    let sink = GPU_EVENT_SINK.get_or_init(|| Mutex::new(None));
    let mut sink = sink
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    gpu_wait_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    AUDIO_OUTPUT_FRAMES.store(0, Ordering::Relaxed);
    AUDIO_UNDERRUN_FRAMES.store(0, Ordering::Relaxed);
    *sink = Some(sender);
}

pub fn clear_event_sink() {
    GPU_EVENTS_IN_FLIGHT.store(0, Ordering::Relaxed);
    GPU_WORK_ITEMS_PENDING.store(0, Ordering::Relaxed);
    GPU_FLIPS_PENDING.store(0, Ordering::Relaxed);
    GPU_COMPLETED_SUBMISSION_SEQUENCE.store(0, Ordering::Release);
    GPU_COMPLETED_SUBMISSIONS
        .get_or_init(|| Mutex::new(BTreeSet::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    GPU_COMPLETED_FLIPS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    if let Some(sink) = GPU_EVENT_SINK.get() {
        let mut sink = sink
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *sink = None;
        // A producer may have incremented either counter and blocked in `send` while holding this
        // sink mutex before teardown began. Reset again after taking ownership so its failed-send
        // rollback cannot leave stale work (or underflow an already reset counter).
        GPU_EVENTS_IN_FLIGHT.store(0, Ordering::Relaxed);
        GPU_WORK_ITEMS_PENDING.store(0, Ordering::Relaxed);
        GPU_FLIPS_PENDING.store(0, Ordering::Relaxed);
        gpu_wait_registry()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    } else {
        gpu_wait_registry()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }
}

/// Publishes one decoded GPU event to the installed bounded presenter channel.
///
/// Queue-submission waits are registered before the potentially blocking channel send. Returning
/// `false` means no backend can complete the event; any wait tickets allocated for a failed send
/// have already been cancelled.
pub fn emit_event(mut event: GpuEvent) -> bool {
    let Some(sink) = GPU_EVENT_SINK.get() else {
        return false;
    };
    let sink = sink
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(sender) = sink.as_ref() else {
        return false;
    };
    let registered_waits = match register_queue_submission_waits(&mut event) {
        Ok(ids) => ids,
        Err(()) => {
            eprintln!(
                "[REMu][GPU] rejected queue submission: more than {GPU_WAIT_CAPACITY} memory waits are pending"
            );
            return false;
        }
    };
    let sent = match event {
        GpuEvent::QueueSubmission {
            queue_id,
            mut events,
        } if events.len() > GPU_EVENT_SUBMISSION_CHUNK_SIZE => {
            let mut all_sent = true;
            loop {
                let remaining = if events.len() > GPU_EVENT_SUBMISSION_CHUNK_SIZE {
                    events.split_off(GPU_EVENT_SUBMISSION_CHUNK_SIZE)
                } else {
                    Vec::new()
                };
                let work_item_count = events.len();
                let flip_count = events
                    .iter()
                    .filter(|event| matches!(event, GpuEvent::Flip { .. }))
                    .count();
                GPU_EVENTS_IN_FLIGHT.fetch_add(1, Ordering::Relaxed);
                GPU_WORK_ITEMS_PENDING.fetch_add(work_item_count, Ordering::Relaxed);
                GPU_FLIPS_PENDING.fetch_add(flip_count, Ordering::Relaxed);
                if sender
                    .send(GpuEvent::QueueSubmission { queue_id, events })
                    .is_err()
                {
                    record_gpu_event_consumed();
                    cancel_pending_gpu_work_items(work_item_count);
                    cancel_pending_gpu_flips(flip_count);
                    all_sent = false;
                    break;
                }
                if remaining.is_empty() {
                    break;
                }
                events = remaining;
            }
            all_sent
        }
        event => {
            let work_item_count = match &event {
                GpuEvent::QueueSubmission { events, .. } => events.len(),
                _ => 1,
            };
            let flip_count = match &event {
                GpuEvent::QueueSubmission { events, .. } => events
                    .iter()
                    .filter(|event| matches!(event, GpuEvent::Flip { .. }))
                    .count(),
                GpuEvent::Flip { .. } => 1,
                _ => 0,
            };
            GPU_EVENTS_IN_FLIGHT.fetch_add(1, Ordering::Relaxed);
            GPU_WORK_ITEMS_PENDING.fetch_add(work_item_count, Ordering::Relaxed);
            GPU_FLIPS_PENDING.fetch_add(flip_count, Ordering::Relaxed);
            if sender.send(event).is_err() {
                record_gpu_event_consumed();
                cancel_pending_gpu_work_items(work_item_count);
                cancel_pending_gpu_flips(flip_count);
                false
            } else {
                true
            }
        }
    };
    if !sent {
        gpu_wait_registry()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cancel(&registered_waits);
        false
    } else {
        true
    }
}

/// Emits an explicit guest GPU cache-invalidation boundary. Event ordering keeps the invalidation
/// between preceding and following guest GPU work without stalling the guest command processor.
pub fn acquire_memory(acquire: GpuMemoryAcquire) {
    emit_event(GpuEvent::MemoryAcquire(acquire));
}

/// Emits an ordered guest pipeline-stage completion boundary.
pub fn pipeline_barrier(barrier: GpuPipelineBarrier) {
    emit_event(GpuEvent::PipelineBarrier(barrier));
}
