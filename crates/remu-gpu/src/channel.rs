// SPDX-License-Identifier: GPL-2.0-or-later

//! The bounded, ordered guest GPU event channel and its process-global sink.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::counters::{
    AUDIO_OUTPUT_FRAMES, AUDIO_UNDERRUN_FRAMES, GPU_COMPLETED_FLIPS,
    GPU_COMPLETED_SUBMISSION_SEQUENCE, GPU_COMPLETED_SUBMISSIONS, GPU_EVENTS_IN_FLIGHT,
    GPU_FLIPS_PENDING, GPU_WORK_ITEMS_PENDING, cancel_pending_gpu_flips,
    cancel_pending_gpu_work_items,
};
use crate::waits::{GPU_WAIT_CAPACITY, gpu_wait_registry, register_queue_submission_waits};
use crate::{GpuEvent, GpuMemoryAcquire, GpuPipelineBarrier, GpuWaitId, record_gpu_event_consumed};

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

#[derive(Default)]
struct EventSinkState {
    sender: Option<SyncSender<GpuEvent>>,
    active_publishers: usize,
}

#[derive(Default)]
struct PublicationCoordinator {
    next_id: u64,
    queues: BTreeMap<Option<u64>, VecDeque<u64>>,
    turns: VecDeque<u64>,
}

#[derive(Clone, Copy)]
struct Publication {
    id: u64,
    queue_id: Option<u64>,
}

struct PublicationGuard {
    publication: Publication,
    finished: bool,
}

/// Local wall-time attribution for one bounded-channel publication call.
///
/// Queue submissions can be split into several bounded chunks. Their FIFO waits and sends are
/// summed here without publishing counters per chunk; the runtime records one aggregate sample
/// for the complete guest publication call.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GpuEventPublicationTiming {
    pub setup: Duration,
    pub queue_fifo_wait: Duration,
    pub bounded_send: Duration,
}

static GPU_EVENT_SINK: OnceLock<(Mutex<EventSinkState>, Condvar)> = OnceLock::new();
static GPU_PUBLICATION_COORDINATOR: OnceLock<(Mutex<PublicationCoordinator>, Condvar)> =
    OnceLock::new();
static REPORTED_LINEAR_SUBMISSION_CHUNKING: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static GPU_PUBLICATION_TIMING_SAMPLES: AtomicUsize = AtomicUsize::new(0);

fn publication_timing_now() -> Instant {
    #[cfg(test)]
    GPU_PUBLICATION_TIMING_SAMPLES.fetch_add(1, Ordering::Relaxed);
    Instant::now()
}

struct ActiveEventSender {
    sender: SyncSender<GpuEvent>,
}

impl Drop for ActiveEventSender {
    fn drop(&mut self) {
        let Some((state, idle)) = GPU_EVENT_SINK.get() else {
            return;
        };
        let mut state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.active_publishers = state.active_publishers.saturating_sub(1);
        idle.notify_all();
    }
}

impl PublicationGuard {
    fn new(event: &GpuEvent) -> Self {
        Self {
            publication: register_publication(event),
            finished: false,
        }
    }

    fn wait_turn(&self) {
        wait_for_publication_turn(self.publication);
    }

    fn complete_chunk(&mut self, finished: bool) {
        complete_publication_chunk(self.publication, finished);
        self.finished = finished;
    }
}

impl Drop for PublicationGuard {
    fn drop(&mut self) {
        if !self.finished {
            abort_publication(self.publication);
        }
    }
}

fn reset_event_sink_state() {
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
    gpu_wait_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    AUDIO_OUTPUT_FRAMES.store(0, Ordering::Relaxed);
    AUDIO_UNDERRUN_FRAMES.store(0, Ordering::Relaxed);
}

fn active_event_sender() -> Option<ActiveEventSender> {
    let (state, _) = GPU_EVENT_SINK.get()?;
    let mut state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sender = state.sender.as_ref()?.clone();
    state.active_publishers = state.active_publishers.saturating_add(1);
    Some(ActiveEventSender { sender })
}

fn register_publication(event: &GpuEvent) -> Publication {
    // Non-queue lifecycle events retain their process-total order. Guest command submissions need
    // only FIFO order within their hardware queue; independent queues must be able to publish the
    // release which another queue is already waiting for.
    let queue_id = match event {
        GpuEvent::QueueSubmission { queue_id, .. } => Some(*queue_id),
        _ => None,
    };
    let (coordinator, _) = GPU_PUBLICATION_COORDINATOR.get_or_init(|| {
        (
            Mutex::new(PublicationCoordinator::default()),
            Condvar::new(),
        )
    });
    let mut coordinator = coordinator
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let id = coordinator.next_id;
    coordinator.next_id = coordinator.next_id.saturating_add(1);
    let queue = coordinator.queues.entry(queue_id).or_default();
    let becomes_active = queue.is_empty();
    queue.push_back(id);
    if becomes_active {
        coordinator.turns.push_back(id);
    }
    Publication { id, queue_id }
}

fn wait_for_publication_turn(publication: Publication) {
    let (coordinator, changed) = GPU_PUBLICATION_COORDINATOR.get_or_init(|| {
        (
            Mutex::new(PublicationCoordinator::default()),
            Condvar::new(),
        )
    });
    let mut coordinator = coordinator
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    while coordinator.turns.front().copied() != Some(publication.id) {
        coordinator = changed
            .wait(coordinator)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

fn complete_publication_chunk(publication: Publication, finished: bool) {
    let (coordinator, changed) = GPU_PUBLICATION_COORDINATOR.get_or_init(|| {
        (
            Mutex::new(PublicationCoordinator::default()),
            Condvar::new(),
        )
    });
    let mut coordinator = coordinator
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(
        coordinator.turns.pop_front(),
        Some(publication.id),
        "only the current fair-publication turn can complete a chunk"
    );
    if finished {
        let next = {
            let queue = coordinator
                .queues
                .get_mut(&publication.queue_id)
                .expect("registered publication owns a queue");
            assert_eq!(
                queue.pop_front(),
                Some(publication.id),
                "same-queue publications retain FIFO order"
            );
            queue.front().copied()
        };
        if let Some(next) = next {
            coordinator.turns.push_back(next);
        } else {
            coordinator.queues.remove(&publication.queue_id);
        }
    } else {
        // One chunk per active hardware queue. Moving this publisher to the back ensures a newly
        // admitted producer queue gets a bounded-channel slot even while a huge submission is
        // still being split.
        coordinator.turns.push_back(publication.id);
    }
    changed.notify_all();
}

fn abort_publication(publication: Publication) {
    let (coordinator, changed) = GPU_PUBLICATION_COORDINATOR.get_or_init(|| {
        (
            Mutex::new(PublicationCoordinator::default()),
            Condvar::new(),
        )
    });
    let mut coordinator = coordinator
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (was_active, next, empty) = {
        let Some(queue) = coordinator.queues.get_mut(&publication.queue_id) else {
            return;
        };
        let Some(position) = queue.iter().position(|id| *id == publication.id) else {
            return;
        };
        let was_active = position == 0;
        queue.remove(position);
        (was_active, queue.front().copied(), queue.is_empty())
    };
    if was_active {
        coordinator.turns.retain(|id| *id != publication.id);
        if let Some(next) = next {
            coordinator.turns.push_back(next);
        }
    }
    if empty {
        coordinator.queues.remove(&publication.queue_id);
    }
    changed.notify_all();
}

fn reset_publication_coordinator() {
    if let Some((coordinator, _)) = GPU_PUBLICATION_COORDINATOR.get() {
        *coordinator
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = PublicationCoordinator::default();
    }
}

fn registered_wait_ids(event: &GpuEvent) -> Vec<GpuWaitId> {
    let GpuEvent::QueueSubmission { events, .. } = event else {
        return Vec::new();
    };
    events
        .iter()
        .filter_map(|event| match event {
            GpuEvent::RegisteredMemoryWait { id, .. } => Some(*id),
            _ => None,
        })
        .collect()
}

/// Creates the bounded, ordered event channel used by the guest GPU command processor.
#[must_use]
pub fn gpu_event_channel() -> (SyncSender<GpuEvent>, Receiver<GpuEvent>) {
    std::sync::mpsc::sync_channel(GPU_EVENT_QUEUE_CAPACITY)
}

pub fn install_event_sink(sender: SyncSender<GpuEvent>) {
    let (state, idle) =
        GPU_EVENT_SINK.get_or_init(|| (Mutex::new(EventSinkState::default()), Condvar::new()));
    let mut state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // Stop admitting work to the previous backend before waiting for its cloned senders. Leaving
    // the old sender installed lets new publishers continuously extend the drain and can starve
    // replacement forever.
    state.sender = None;
    while state.active_publishers != 0 {
        state = idle
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
    state.sender = Some(sender);
    reset_publication_coordinator();
    reset_event_sink_state();
}

pub fn clear_event_sink() {
    if let Some((state, idle)) = GPU_EVENT_SINK.get() {
        let mut state = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.sender = None;
        // Publishers clone the sender so independent hardware queues can block concurrently
        // without monopolizing this lifecycle mutex. Wait for those clones to finish before
        // resetting counters; otherwise a failed-send rollback can underflow freshly reset state.
        while state.active_publishers != 0 {
            state = idle
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
    reset_publication_coordinator();
    reset_event_sink_state();
}

/// Publishes one decoded GPU event to the installed bounded presenter channel.
///
/// Queue-submission waits are registered before the potentially blocking channel send. Returning
/// `false` means no backend can complete the event; any wait tickets allocated for a failed send
/// have already been cancelled.
pub fn emit_event(event: GpuEvent) -> bool {
    emit_event_with_admission(event, || {})
}

/// Publishes one event and returns bounded-channel phase timing for its complete call.
pub fn emit_event_with_timing(event: GpuEvent) -> (bool, GpuEventPublicationTiming) {
    emit_event_with_optional_timing(event, || {}, true)
}

/// Publishes one event and reports when it has entered its queue-local FIFO.
///
/// The callback runs after the event has been assigned an order within its hardware queue but
/// before bounded-channel backpressure can block it. A caller which assigned a process-order
/// ticket may therefore admit the next independent queue without allowing a later submission on
/// this queue to overtake.
pub fn emit_event_with_admission(event: GpuEvent, admitted: impl FnOnce()) -> bool {
    emit_event_with_optional_timing(event, admitted, false).0
}

/// Publishes one event, reports queue admission, and attributes its bounded-channel phases.
pub fn emit_event_with_admission_and_timing(
    event: GpuEvent,
    admitted: impl FnOnce(),
) -> (bool, GpuEventPublicationTiming) {
    emit_event_with_optional_timing(event, admitted, true)
}

fn emit_event_with_optional_timing(
    mut event: GpuEvent,
    admitted: impl FnOnce(),
    collect_timing: bool,
) -> (bool, GpuEventPublicationTiming) {
    let setup_started = collect_timing.then(publication_timing_now);
    let mut timing = collect_timing.then(GpuEventPublicationTiming::default);
    let Some(sender) = active_event_sender() else {
        admitted();
        if let (Some(timing), Some(started)) = (timing.as_mut(), setup_started) {
            timing.setup = started.elapsed();
        }
        return (false, timing.unwrap_or_default());
    };
    let mut unpublished_waits = match register_queue_submission_waits(&mut event) {
        Ok(ids) => ids,
        Err(()) => {
            admitted();
            if let (Some(timing), Some(started)) = (timing.as_mut(), setup_started) {
                timing.setup = started.elapsed();
            }
            eprintln!(
                "[REMu][GPU] rejected queue submission: more than {GPU_WAIT_CAPACITY} memory waits are pending"
            );
            return (false, timing.unwrap_or_default());
        }
    };
    let mut publication = PublicationGuard::new(&event);
    admitted();
    if let (Some(timing), Some(started)) = (timing.as_mut(), setup_started) {
        timing.setup = started.elapsed();
    }
    let sent = match event {
        GpuEvent::QueueSubmission { queue_id, events }
            if events.len() > GPU_EVENT_SUBMISSION_CHUNK_SIZE =>
        {
            if !REPORTED_LINEAR_SUBMISSION_CHUNKING.load(Ordering::Relaxed)
                && REPORTED_LINEAR_SUBMISSION_CHUNKING
                    .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            {
                eprintln!(
                    "[REMu][GPU] linear submission chunking enabled chunk_size={GPU_EVENT_SUBMISSION_CHUNK_SIZE}"
                );
            }
            // Splitting the leading chunk from a `Vec` with `split_off` copies the complete
            // remaining tail on every iteration and leaves each published prefix retaining the
            // previous tail's capacity. Consume the source allocation once instead, then build
            // only the fixed-capacity chunks retained by the bounded presenter channel.
            let mut remaining = events.into_iter();
            let mut all_sent = true;
            while !remaining.as_slice().is_empty() {
                let chunk_size = remaining.len().min(GPU_EVENT_SUBMISSION_CHUNK_SIZE);
                let mut events = Vec::with_capacity(chunk_size);
                events.extend(remaining.by_ref().take(chunk_size));
                let work_item_count = events.len();
                let flip_count = events
                    .iter()
                    .filter(|event| matches!(event, GpuEvent::Flip { .. }))
                    .count();
                let chunk = GpuEvent::QueueSubmission { queue_id, events };
                let chunk_waits = registered_wait_ids(&chunk);
                let fifo_started = collect_timing.then(publication_timing_now);
                publication.wait_turn();
                if let (Some(timing), Some(started)) = (timing.as_mut(), fifo_started) {
                    timing.queue_fifo_wait =
                        timing.queue_fifo_wait.saturating_add(started.elapsed());
                }
                GPU_EVENTS_IN_FLIGHT.fetch_add(1, Ordering::Relaxed);
                GPU_WORK_ITEMS_PENDING.fetch_add(work_item_count, Ordering::Relaxed);
                GPU_FLIPS_PENDING.fetch_add(flip_count, Ordering::Relaxed);
                let send_started = collect_timing.then(publication_timing_now);
                let chunk_sent = sender.sender.send(chunk).is_ok();
                if let (Some(timing), Some(started)) = (timing.as_mut(), send_started) {
                    timing.bounded_send = timing.bounded_send.saturating_add(started.elapsed());
                }
                if !chunk_sent {
                    record_gpu_event_consumed();
                    cancel_pending_gpu_work_items(work_item_count);
                    cancel_pending_gpu_flips(flip_count);
                    publication.complete_chunk(true);
                    all_sent = false;
                    break;
                }
                unpublished_waits.retain(|id| !chunk_waits.contains(id));
                let finished = remaining.as_slice().is_empty();
                publication.complete_chunk(finished);
                if finished {
                    break;
                }
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
            let event_waits = registered_wait_ids(&event);
            let fifo_started = collect_timing.then(publication_timing_now);
            publication.wait_turn();
            if let (Some(timing), Some(started)) = (timing.as_mut(), fifo_started) {
                timing.queue_fifo_wait = timing.queue_fifo_wait.saturating_add(started.elapsed());
            }
            GPU_EVENTS_IN_FLIGHT.fetch_add(1, Ordering::Relaxed);
            GPU_WORK_ITEMS_PENDING.fetch_add(work_item_count, Ordering::Relaxed);
            GPU_FLIPS_PENDING.fetch_add(flip_count, Ordering::Relaxed);
            let send_started = collect_timing.then(publication_timing_now);
            let sent = sender.sender.send(event).is_ok();
            if let (Some(timing), Some(started)) = (timing.as_mut(), send_started) {
                timing.bounded_send = timing.bounded_send.saturating_add(started.elapsed());
            }
            if !sent {
                record_gpu_event_consumed();
                cancel_pending_gpu_work_items(work_item_count);
                cancel_pending_gpu_flips(flip_count);
            } else {
                unpublished_waits.retain(|id| !event_waits.contains(id));
            }
            publication.complete_chunk(true);
            sent
        }
    };
    if !sent {
        gpu_wait_registry()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .cancel(&unpublished_waits);
        (false, timing.unwrap_or_default())
    } else {
        (true, timing.unwrap_or_default())
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

#[cfg(test)]
mod timing_tests {
    use super::{
        GPU_PUBLICATION_TIMING_SAMPLES, clear_event_sink, emit_event, emit_event_with_timing,
        gpu_event_channel, install_event_sink,
    };
    use crate::GpuEvent;
    use std::sync::atomic::Ordering;

    #[test]
    fn ordinary_emit_path_takes_no_publication_timestamps() {
        let (sender, receiver) = gpu_event_channel();
        install_event_sink(sender);
        let before = GPU_PUBLICATION_TIMING_SAMPLES.load(Ordering::Relaxed);
        assert!(emit_event(GpuEvent::HideHostDialog { sequence: 1 }));
        assert_eq!(
            GPU_PUBLICATION_TIMING_SAMPLES.load(Ordering::Relaxed),
            before,
            "the default event path must not sample the monotonic clock"
        );
        receiver.recv().expect("untimed event reaches receiver");

        let before = GPU_PUBLICATION_TIMING_SAMPLES.load(Ordering::Relaxed);
        assert!(emit_event_with_timing(GpuEvent::HideHostDialog { sequence: 2 }).0);
        assert!(GPU_PUBLICATION_TIMING_SAMPLES.load(Ordering::Relaxed) > before);
        receiver.recv().expect("timed event reaches receiver");
        clear_event_sink();
    }
}
