// SPDX-License-Identifier: GPL-2.0-or-later

//! The bounded, ordered guest GPU event channel and its process-global sink.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
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
    lanes: BTreeMap<Option<u64>, Arc<PublicationLane>>,
}

#[derive(Default)]
struct PublicationLane {
    queue: Mutex<VecDeque<u64>>,
    changed: Condvar,
}

#[derive(Clone)]
struct Publication {
    id: u64,
    lane: Arc<PublicationLane>,
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
static GPU_PUBLICATION_COORDINATOR: OnceLock<Mutex<PublicationCoordinator>> = OnceLock::new();
static REPORTED_LINEAR_SUBMISSION_CHUNKING: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static GPU_PUBLICATION_TIMING_SAMPLES: AtomicUsize = AtomicUsize::new(0);

fn publication_timing_now() -> Instant {
    #[cfg(test)]
    GPU_PUBLICATION_TIMING_SAMPLES.fetch_add(1, Ordering::Relaxed);
    Instant::now()
}

// A per-thread event sink override. When a thread installs one, every event it emits is
// delivered exclusively to that thread's receiver and never to the process-global sink. This is
// the isolation the test suite needs (plan E2): the emitting thread — a test — cannot deliver
// into a receiver another test installed, which is the process-global cross-talk that made the
// suite order-dependent and required `--test-threads=1`. Production never installs one, so the
// global sink path is unchanged.
thread_local! {
    static THREAD_LOCAL_EVENT_SINK: std::cell::RefCell<Option<SyncSender<GpuEvent>>> =
        const { std::cell::RefCell::new(None) };
}

struct ActiveEventSender {
    sender: SyncSender<GpuEvent>,
    /// A thread-local sink owns no share of the global `active_publishers` count, so its drop
    /// must not touch that lifecycle counter.
    thread_local: bool,
}

impl Drop for ActiveEventSender {
    fn drop(&mut self) {
        if self.thread_local {
            return;
        }
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
        wait_for_publication_turn(&self.publication);
    }

    fn complete_chunk(&mut self, finished: bool) {
        complete_publication_chunk(&self.publication, finished);
        self.finished = finished;
    }
}

impl Drop for PublicationGuard {
    fn drop(&mut self) {
        if !self.finished {
            abort_publication(&self.publication);
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
    if let Some(sender) = THREAD_LOCAL_EVENT_SINK.with(|cell| cell.borrow().clone()) {
        return Some(ActiveEventSender {
            sender,
            thread_local: true,
        });
    }
    let (state, _) = GPU_EVENT_SINK.get()?;
    let mut state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sender = state.sender.as_ref()?.clone();
    state.active_publishers = state.active_publishers.saturating_add(1);
    Some(ActiveEventSender {
        sender,
        thread_local: false,
    })
}

/// Installs an event sink scoped to the calling thread (plan E2).
///
/// Events emitted on this thread are delivered only to `sender` and never touch the
/// process-global sink, so concurrent callers on other threads cannot deliver into this
/// receiver. Tests use this instead of [`install_event_sink`] so the suite runs in parallel.
/// The counters and the publication coordinator remain process-global and are reset here for a
/// clean per-test baseline; a test that also asserts on those must still not run concurrently
/// with another that mutates them.
pub fn install_thread_local_event_sink(sender: SyncSender<GpuEvent>) {
    THREAD_LOCAL_EVENT_SINK.with(|cell| *cell.borrow_mut() = Some(sender));
    reset_publication_coordinator();
    reset_event_sink_state();
}

/// Removes the calling thread's event sink override.
pub fn clear_thread_local_event_sink() {
    THREAD_LOCAL_EVENT_SINK.with(|cell| *cell.borrow_mut() = None);
    reset_publication_coordinator();
    reset_event_sink_state();
}

fn register_publication(event: &GpuEvent) -> Publication {
    // Non-queue lifecycle events retain their process-total order. Guest command submissions need
    // only FIFO order within their hardware queue; independent queues must be able to publish the
    // release which another queue is already waiting for.
    let queue_id = match event {
        GpuEvent::QueueSubmission { queue_id, .. } => Some(*queue_id),
        _ => None,
    };
    let coordinator =
        GPU_PUBLICATION_COORDINATOR.get_or_init(|| Mutex::new(PublicationCoordinator::default()));
    let mut coordinator = coordinator
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let id = coordinator.next_id;
    coordinator.next_id = coordinator.next_id.saturating_add(1);
    let lane = Arc::clone(
        coordinator
            .lanes
            .entry(queue_id)
            .or_insert_with(|| Arc::new(PublicationLane::default())),
    );
    // Register while the lane is still reached through the coordinator lock. No completion path
    // takes the coordinator, so the fixed coordinator -> lane order cannot invert, and two first
    // publications cannot create separate lanes for one hardware queue.
    lane.queue
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push_back(id);
    Publication { id, lane }
}

fn wait_for_publication_turn(publication: &Publication) {
    let mut queue = publication
        .lane
        .queue
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    while queue.front().copied() != Some(publication.id) {
        queue = publication
            .lane
            .changed
            .wait(queue)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

fn complete_publication_chunk(publication: &Publication, finished: bool) {
    // The publication owns its hardware lane until its complete submission has entered the
    // bounded channel. Independent lanes already make progress concurrently in `SyncSender`, so a
    // continuation neither rotates through a global turnstile nor wakes any parked publisher.
    if !finished {
        return;
    }
    let mut queue = publication
        .lane
        .queue
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(
        queue.pop_front(),
        Some(publication.id),
        "same-queue publications retain FIFO order"
    );
    drop(queue);
    // A Condvar does not promise which waiter `notify_one` selects. Waking the small queue-local
    // set lets only its new head pass the predicate without reviving publishers on other queues.
    publication.lane.changed.notify_all();
}

fn abort_publication(publication: &Publication) {
    let mut queue = publication
        .lane
        .queue
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(position) = queue.iter().position(|id| *id == publication.id) else {
        return;
    };
    let was_active = position == 0;
    queue.remove(position);
    drop(queue);
    if was_active {
        publication.lane.changed.notify_all();
    }
}

fn reset_publication_coordinator() {
    if let Some(coordinator) = GPU_PUBLICATION_COORDINATOR.get() {
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
mod isolation_tests {
    use super::{
        clear_thread_local_event_sink, emit_event, gpu_event_channel, install_event_sink,
        install_thread_local_event_sink,
    };
    use crate::GpuEvent;

    #[test]
    fn a_thread_local_sink_isolates_emission_from_the_process_global_sink() {
        // A global sink stands in for a receiver another test installed on another thread.
        let (global_sender, global_receiver) = gpu_event_channel();
        install_event_sink(global_sender);

        let (local_sender, local_receiver) = gpu_event_channel();
        install_thread_local_event_sink(local_sender);
        assert!(emit_event(GpuEvent::HideHostDialog { sequence: 7 }));

        assert!(
            matches!(
                local_receiver.try_recv(),
                Ok(GpuEvent::HideHostDialog { sequence: 7 })
            ),
            "the event reaches this thread's own receiver"
        );
        assert!(
            global_receiver.try_recv().is_err(),
            "the event must never leak to the process-global sink"
        );

        clear_thread_local_event_sink();
        assert!(emit_event(GpuEvent::HideHostDialog { sequence: 8 }));
        assert!(
            matches!(
                global_receiver.try_recv(),
                Ok(GpuEvent::HideHostDialog { sequence: 8 })
            ),
            "clearing the override restores the process-global path"
        );
        super::clear_event_sink();
    }
}

#[cfg(test)]
mod publication_lane_tests {
    use super::{
        complete_publication_chunk, register_publication, reset_publication_coordinator,
        wait_for_publication_turn,
    };
    use crate::GpuEvent;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn chunks_retain_one_lane_without_waking_other_hardware_queues() {
        reset_publication_coordinator();
        let graphics = register_publication(&GpuEvent::QueueSubmission {
            queue_id: 0,
            events: Vec::new(),
        });
        let later_graphics = register_publication(&GpuEvent::QueueSubmission {
            queue_id: 0,
            events: Vec::new(),
        });
        let compute = register_publication(&GpuEvent::QueueSubmission {
            queue_id: 33,
            events: Vec::new(),
        });
        assert!(Arc::ptr_eq(&graphics.lane, &later_graphics.lane));
        assert!(!Arc::ptr_eq(&graphics.lane, &compute.lane));
        wait_for_publication_turn(&graphics);
        wait_for_publication_turn(&compute);

        let (entered, entered_rx) = std::sync::mpsc::channel();
        let (advanced, advanced_rx) = std::sync::mpsc::channel();
        let waiting = later_graphics.clone();
        let waiter = std::thread::spawn(move || {
            entered.send(()).expect("test receiver remains alive");
            wait_for_publication_turn(&waiting);
            advanced.send(()).expect("test receiver remains alive");
        });
        entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("same-queue waiter starts");

        complete_publication_chunk(&graphics, false);
        complete_publication_chunk(&compute, true);
        assert_eq!(
            advanced_rx.recv_timeout(Duration::from_millis(30)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout),
            "a continuation and an independent lane must not advance the same-queue successor"
        );

        complete_publication_chunk(&graphics, true);
        advanced_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("finishing the lane advances its successor");
        waiter.join().expect("same-queue waiter advances");
        complete_publication_chunk(&later_graphics, true);
        reset_publication_coordinator();
    }

    #[test]
    fn non_submission_events_share_one_lifecycle_lane() {
        reset_publication_coordinator();
        let first = register_publication(&GpuEvent::HideHostDialog { sequence: 1 });
        let second = register_publication(&GpuEvent::HideHostDialog { sequence: 2 });
        let queue = register_publication(&GpuEvent::QueueSubmission {
            queue_id: 0,
            events: Vec::new(),
        });
        assert!(Arc::ptr_eq(&first.lane, &second.lane));
        assert!(!Arc::ptr_eq(&first.lane, &queue.lane));
        complete_publication_chunk(&first, true);
        complete_publication_chunk(&second, true);
        complete_publication_chunk(&queue, true);
        reset_publication_coordinator();
    }
}

#[cfg(test)]
mod timing_tests {
    use super::{
        GPU_PUBLICATION_TIMING_SAMPLES, clear_thread_local_event_sink, emit_event,
        emit_event_with_timing, gpu_event_channel, install_thread_local_event_sink,
    };
    use crate::GpuEvent;
    use std::sync::atomic::Ordering;

    #[test]
    fn ordinary_emit_path_takes_no_publication_timestamps() {
        let (sender, receiver) = gpu_event_channel();
        install_thread_local_event_sink(sender);
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
        clear_thread_local_event_sink();
    }
}
