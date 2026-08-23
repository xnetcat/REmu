// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::{
    GPU_EVENTS_PER_TICK, GPU_EVENTS_PER_TICK_BACKLOG_MULTIPLIER,
    GPU_EVENTS_PER_TICK_BACKLOG_THRESHOLD,
};
use super::{
    OrderedEventPoll, OrderedEventSource, OrderedGpuEventReceiver, PipelineCompileCandidate,
    QueueBlockReason, ScheduledGpuSubmission, ScheduledHostEvent,
    SharedEventHostEffectPendingStatistics,
};
use remu_gpu::{GpuEvent, GpuMemoryReleaseValue, GpuMemoryWait, GpuWaitId};
use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

fn dependency_ingress_budget() -> usize {
    let pending = remu_gpu::gpu_work_items_pending();
    let multiplier = if pending > super::super::GPU_EVENTS_PER_TICK_EXTREME_BACKLOG_THRESHOLD {
        super::super::GPU_EVENTS_PER_TICK_EXTREME_BACKLOG_MULTIPLIER
    } else if pending > super::super::GPU_EVENTS_PER_TICK_HEAVY_BACKLOG_THRESHOLD {
        super::super::GPU_EVENTS_PER_TICK_HEAVY_BACKLOG_MULTIPLIER
    } else if pending > GPU_EVENTS_PER_TICK_BACKLOG_THRESHOLD {
        GPU_EVENTS_PER_TICK_BACKLOG_MULTIPLIER
    } else {
        1
    };
    GPU_EVENTS_PER_TICK.saturating_mul(multiplier)
}

/// Largest producer write, in bytes, indexed per 4-byte cell in `producer_labels`.
///
/// Fence labels and release/write payloads are a few dwords; indexing a multi-megabyte DMA or
/// storage binding per cell would be as expensive as the scan this index replaces. Skipping
/// wide producers undercounts, which only widens dependency ingress — the safe direction.
const PRODUCER_LABEL_MAX_BYTES: u64 = 512;

/// Visits the 4-byte label cells of one producer write, or nothing if it is too wide.
fn for_each_producer_label_cell(address: u64, byte_count: u64, mut visit: impl FnMut(u64)) {
    if byte_count == 0 || byte_count > PRODUCER_LABEL_MAX_BYTES {
        return;
    }
    let first = address >> 2;
    let last = address.saturating_add(byte_count - 1) >> 2;
    for cell in first..=last {
        visit(cell);
    }
}

/// Visits every small producer write in `event`: the release/write/copy destinations and the
/// writable global bindings the exact diagnostic scan also recognises as producers.
fn for_each_event_producer_cell(event: &GpuEvent, mut visit: impl FnMut(u64)) {
    match event {
        GpuEvent::MemoryRelease(release) => {
            let byte_count = match release.value {
                GpuMemoryReleaseValue::Immediate32(_) => 4,
                GpuMemoryReleaseValue::Immediate64(_) | GpuMemoryReleaseValue::Timestamp => 8,
            };
            for_each_producer_label_cell(release.address, byte_count, &mut visit);
        }
        GpuEvent::MemoryWrite(write) => {
            let byte_count = if write.increment_address {
                u64::try_from(write.values.len())
                    .unwrap_or(u64::MAX / 4)
                    .saturating_mul(4)
            } else {
                4
            };
            for_each_producer_label_cell(write.address, byte_count, &mut visit);
        }
        GpuEvent::MemoryCopy(copy) => {
            for_each_producer_label_cell(copy.destination_address, copy.byte_count, &mut visit);
        }
        GpuEvent::ComputeDispatch(dispatch) => {
            for binding in dispatch
                .shader_evaluation
                .iter()
                .flat_map(|evaluation| &evaluation.global_memory_bindings)
                .filter(|binding| binding.writable)
            {
                for_each_producer_label_cell(
                    binding.base_address,
                    u64::try_from(binding.byte_count).unwrap_or(u64::MAX),
                    &mut visit,
                );
            }
        }
        GpuEvent::IndexedDraw(draw) => {
            for binding in draw
                .export_shader_evaluation
                .iter()
                .chain(draw.pixel_shader_evaluation.iter())
                .flat_map(|evaluation| &evaluation.global_memory_bindings)
                .filter(|binding| binding.writable)
            {
                for_each_producer_label_cell(
                    binding.base_address,
                    u64::try_from(binding.byte_count).unwrap_or(u64::MAX),
                    &mut visit,
                );
            }
        }
        _ => {}
    }
}

impl OrderedGpuEventReceiver {
    pub(crate) fn new(receiver: Receiver<GpuEvent>) -> Self {
        Self {
            receiver,
            submissions: HashMap::new(),
            queue_order: Vec::new(),
            next_queue_index: 0,
            next_prefetch_source_index: 0,
            active_queue: None,
            blocked_queues: HashMap::new(),
            host_events: VecDeque::new(),
            host_block: None,
            next_ingress_sequence: 0,
            disconnected: false,
            event_queue_timing: false,
            last_event_ingested_at: None,
            last_event_queue_wait: Duration::ZERO,
            producer_labels: HashMap::new(),
        }
    }

    pub(crate) fn set_event_queue_timing(&mut self, enabled: bool) {
        self.event_queue_timing = enabled;
    }

    pub(crate) fn last_event_queue_wait(&self) -> Duration {
        self.last_event_queue_wait
    }

    fn index_queued_event_producers(&mut self, queue_id: u64, event: &GpuEvent) {
        for_each_event_producer_cell(event, |cell| {
            *self.producer_labels.entry((cell, queue_id)).or_insert(0) += 1;
        });
    }

    fn unindex_queued_event_producers(&mut self, queue_id: u64, event: &GpuEvent) {
        for_each_event_producer_cell(event, |cell| {
            let key = (cell, queue_id);
            if let Some(count) = self.producer_labels.get_mut(&key) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.producer_labels.remove(&key);
                }
            }
        });
    }

    pub(crate) fn ingest(&mut self, receive_limit: usize) {
        if self.disconnected {
            return;
        }
        for _ in 0..receive_limit {
            // Elevated ingress must stand down the moment a just-ingested chunk makes the
            // producer visible, so the decision is per received event. The producer index
            // keeps this evaluation O(blocked queues), not O(queued events).
            let pending_capacity = if self.needs_dependency_ingress() {
                // A producer queue can be arbitrarily far behind an already-decoded tail from
                // the blocked queue. A fixed lookahead merely moves the deadlock boundary: Silent
                // Hill's level load needed 290 more events than the old two-channel reserve and
                // parked both hardware queues until Unreal's 120-second RenderThread watchdog.
                //
                // The publishing thread already owns the complete decoded submission and is
                // synchronously splitting it into bounded chunks. Draining those chunks here
                // transfers that existing state to its per-queue FIFO; it does not allow any tail
                // event to overtake the front wait. Stop as soon as the producer index sees a
                // matching write, at which point ordinary decoded-event backpressure resumes.
                usize::MAX
            } else {
                remu_gpu::GPU_PENDING_EVENT_CAPACITY
            };
            if self.pending_event_count()
                > pending_capacity.saturating_sub(remu_gpu::GPU_EVENT_SUBMISSION_CHUNK_SIZE)
            {
                break;
            }
            let event = match self.receiver.try_recv() {
                Ok(event) => {
                    remu_gpu::record_gpu_event_consumed();
                    event
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.disconnected = true;
                    break;
                }
            };
            let ingress_sequence = self.next_ingress_sequence;
            self.next_ingress_sequence = self.next_ingress_sequence.saturating_add(1);
            let ingested_at = self.event_queue_timing.then(Instant::now);
            match event {
                GpuEvent::QueueSubmission { queue_id, events } => {
                    remu_gpu::record_startup_milestone(
                        remu_gpu::StartupMilestone::FirstQueueSubmission,
                    );
                    if events.is_empty() {
                        continue;
                    }
                    if !self.submissions.contains_key(&queue_id) {
                        self.queue_order.push(queue_id);
                    }
                    for event in &events {
                        self.index_queued_event_producers(queue_id, event);
                    }
                    self.submissions.entry(queue_id).or_default().push_back(
                        ScheduledGpuSubmission {
                            ingress_sequence,
                            ingested_at,
                            events: events.into(),
                        },
                    );
                }
                event => self.host_events.push_back(ScheduledHostEvent {
                    ingress_sequence,
                    ingested_at,
                    event,
                }),
            }
        }
    }

    /// Returns whether the normal decoded-event cap is withholding the only queue which can
    /// satisfy an already-ingested guest-memory wait.
    ///
    /// Large submissions are split into bounded channel chunks. A wait near the beginning of one
    /// chunk can therefore park every queue currently known to the receiver while the producer's
    /// queue is still behind later chunks in the channel. Stopping ingress at the ordinary
    /// backlog cap in that state creates a host-only deadlock: the blocked queue cannot consume
    /// its tail, and the producer cannot become visible.
    ///
    /// Move already-decoded chunks into queue FIFOs only until a matching producer becomes
    /// visible. As soon as one newly discovered queue is runnable, or the producer index records
    /// the awaited write, the next loop iteration restores the normal cap. No event executes out
    /// of order, and a blocked queue's tail remains behind its wait.
    fn needs_dependency_ingress(&self) -> bool {
        let has_memory_wait = self
            .blocked_queues
            .values()
            .any(|reason| matches!(reason, QueueBlockReason::GuestMemoryWait(_)))
            || matches!(self.host_block, Some(QueueBlockReason::GuestMemoryWait(_)));
        if !has_memory_wait {
            return false;
        }
        // Usual case: every known front is a memory wait, so pull ahead for a producer queue.
        if !self.has_dependency_runnable_events() {
            return true;
        }
        // Title-menu case: one queue parks on a label while another still looks runnable. The
        // ordinary pending cap then withholds a cross-queue RELEASE still sitting in the channel
        // (`producers=[none-ingested]`). Keep elevated ingress until an ingested producer appears.
        self.blocked_memory_wait_missing_ingested_producer()
    }

    fn blocked_memory_wait_missing_ingested_producer(&self) -> bool {
        self.blocked_queues
            .iter()
            .any(|(queue_id, reason)| match reason {
                QueueBlockReason::GuestMemoryWait(wait) => {
                    !self.queued_wait_has_indexed_producer(*wait, Some(*queue_id))
                }
                QueueBlockReason::GuestMemoryWaitPendingTerminal(_) => false,
                _ => false,
            })
            || self.host_block.as_ref().is_some_and(|reason| match reason {
                QueueBlockReason::GuestMemoryWait(wait) => {
                    !self.queued_wait_has_indexed_producer(*wait, None)
                }
                QueueBlockReason::GuestMemoryWaitPendingTerminal(_) => false,
                _ => false,
            })
    }

    /// O(1) form of "does any ingested queue still hold a producer for this wait".
    ///
    /// Wide producers are not indexed, so a miss can be an undercount — that only elevates
    /// dependency ingress, the direction the elevated path exists to guarantee. A hit can be
    /// indexed at the guest's minimum 32-bit label granularity. Per-queue keys also keep a
    /// producer behind its own queue's wait from masquerading as runnable cross-queue work.
    fn queued_wait_has_indexed_producer(
        &self,
        wait: GpuMemoryWait,
        excluded_queue: Option<u64>,
    ) -> bool {
        let wait_bytes = if wait.is_64_bit { 8 } else { 4 };
        let first = wait.address >> 2;
        let last = wait.address.saturating_add(wait_bytes - 1) >> 2;
        self.queue_order.iter().any(|queue_id| {
            Some(*queue_id) != excluded_queue
                && (first..=last).any(|cell| self.producer_labels.contains_key(&(cell, *queue_id)))
        })
    }

    fn queue_front_event(&self, queue_id: u64) -> Option<&GpuEvent> {
        self.submissions
            .get(&queue_id)?
            .iter()
            .find_map(|submission| submission.events.front())
    }

    fn queue_front_memory_wait(&self, queue_id: u64) -> Option<GpuMemoryWait> {
        match self.queue_front_event(queue_id)? {
            GpuEvent::MemoryWait(wait) | GpuEvent::RegisteredMemoryWait { wait, .. } => Some(*wait),
            _ => None,
        }
    }

    /// Chooses a runnable queue which already contains a write overlapping another queue's
    /// front wait. Executing that queue first avoids an unnecessary CPU poll, batch flush, and
    /// block/retry cycle while preserving FIFO order inside both guest queues.
    fn queued_producer_before_front_wait(&self) -> Option<(u64, u64)> {
        for consumer_queue in self.queue_order.iter().copied() {
            if self.blocked_queues.contains_key(&consumer_queue) {
                continue;
            }
            let Some(wait) = self.queue_front_memory_wait(consumer_queue) else {
                continue;
            };
            let wait_bytes = if wait.is_64_bit { 8 } else { 4 };
            let first = wait.address >> 2;
            let last = wait.address.saturating_add(wait_bytes - 1) >> 2;
            let producer_queue = self.queue_order.iter().copied().find(|queue_id| {
                *queue_id != consumer_queue
                    && !self.blocked_queues.contains_key(queue_id)
                    && self
                        .queue_front_event(*queue_id)
                        .is_some_and(|event| !event_is_memory_wait(event))
                    && (first..=last)
                        .any(|cell| self.producer_labels.contains_key(&(cell, *queue_id)))
            });
            if let Some(producer_queue) = producer_queue {
                return Some((consumer_queue, producer_queue));
            }
        }
        None
    }

    /// Returns whether one current queue front can execute without first polling a memory wait.
    ///
    /// A wait is popped before the presenter can prove it unsatisfied and reinsert it with a
    /// block reason. Because processing stops at that point, two queues whose fronts are waits
    /// alternate between "blocked" and "not yet polled" on successive AppKit ticks. Treating the
    /// latter as runnable prevents dependency ingress forever once the ordinary pending cap is
    /// full, even when a third queue containing the producer is already waiting in the channel.
    ///
    /// Only the front event of each queue is relevant. Work in later submissions cannot overtake
    /// a front wait and therefore cannot make progress or satisfy another queue.
    fn has_dependency_runnable_events(&self) -> bool {
        if self.host_block.is_none()
            && self
                .host_events
                .front()
                .is_some_and(|host| !event_is_memory_wait(&host.event))
        {
            return true;
        }
        self.queue_order.iter().any(|queue_id| {
            !self.blocked_queues.contains_key(queue_id)
                && self
                    .submissions
                    .get(queue_id)
                    .and_then(|submissions| {
                        submissions
                            .iter()
                            .find_map(|submission| submission.events.front())
                    })
                    .is_some_and(|event| !event_is_memory_wait(event))
        })
    }

    /// Pulls bounded channel chunks while the current block graph can still identify them as
    /// possible cross-queue dependencies.
    ///
    /// Periodic wait polling temporarily makes each suspended queue runnable so its label can be
    /// tested again. Dependency ingress must happen before that reset: otherwise the ordinary
    /// pending-event cap sees the requeued wait as runnable, leaves the channel full, and can
    /// indefinitely withhold a producer from another queue.
    fn ingest_guest_memory_wait_dependencies(&mut self) {
        if self.needs_dependency_ingress() {
            self.ingest(dependency_ingress_budget());
        }
    }

    pub(crate) fn try_recv(
        &mut self,
        mut block_resolved: impl FnMut(&mut QueueBlockReason) -> bool,
    ) -> OrderedEventPoll {
        self.ingest(dependency_ingress_budget());
        self.blocked_queues.retain(|_, reason| match reason {
            QueueBlockReason::GuestMemoryWait(_)
            | QueueBlockReason::GuestMemoryWaitPendingTerminal(_) => true,
            reason @ (QueueBlockReason::FlipSafetyWait(_)
            | QueueBlockReason::GuestSharedEventHostEffect { .. }
            | QueueBlockReason::GraphicsPipeline(_)
            | QueueBlockReason::ComputePipeline(_)) => !block_resolved(reason),
        });
        if self.host_block.as_mut().is_some_and(|reason| match reason {
            QueueBlockReason::GuestMemoryWait(_)
            | QueueBlockReason::GuestMemoryWaitPendingTerminal(_) => false,
            reason @ (QueueBlockReason::FlipSafetyWait(_)
            | QueueBlockReason::GuestSharedEventHostEffect { .. }
            | QueueBlockReason::GraphicsPipeline(_)
            | QueueBlockReason::ComputePipeline(_)) => block_resolved(reason),
        }) {
            self.host_block = None;
        }

        let display_path_event = self
            .host_events
            .front()
            .is_some_and(|host| host_event_bypasses_blocked_guest_queues(&host.event));
        let oldest_submission = self
            .submissions
            .iter()
            .filter(|(queue_id, _)| {
                !display_path_event
                    || !matches!(
                        self.blocked_queues.get(queue_id),
                        Some(
                            QueueBlockReason::GuestMemoryWait(_)
                                | QueueBlockReason::GuestMemoryWaitPendingTerminal(_)
                                | QueueBlockReason::GuestSharedEventHostEffect { .. }
                                | QueueBlockReason::FlipSafetyWait(_)
                        )
                    )
            })
            .filter_map(|(_, queue)| queue.front())
            .map(|submission| submission.ingress_sequence)
            .min();
        if self.host_block.is_none()
            && self.host_events.front().is_some_and(|host| {
                oldest_submission.is_none_or(|oldest| host.ingress_sequence < oldest)
            })
        {
            let host = self
                .host_events
                .pop_front()
                .expect("host event exists after readiness check");
            self.last_event_ingested_at = host.ingested_at;
            self.last_event_queue_wait = host.ingested_at.map_or(Duration::ZERO, |at| {
                Instant::now().saturating_duration_since(at)
            });
            return OrderedEventPoll::Event {
                source: OrderedEventSource::Host {
                    ingress_sequence: host.ingress_sequence,
                },
                event: host.event,
            };
        }

        if let Some((consumer_queue, producer_queue)) = self.queued_producer_before_front_wait()
            && let Some(event) = self.pop_queue_event(producer_queue)
        {
            self.active_queue = Some(consumer_queue);
            return OrderedEventPoll::Event {
                source: OrderedEventSource::GuestQueue(producer_queue),
                event,
            };
        }

        if let Some(queue_id) = self.active_queue {
            if !self.blocked_queues.contains_key(&queue_id)
                && let Some(event) = self.pop_queue_event(queue_id)
            {
                return OrderedEventPoll::Event {
                    source: OrderedEventSource::GuestQueue(queue_id),
                    event,
                };
            }
            self.active_queue = None;
        }

        for offset in 0..self.queue_order.len() {
            let index = (self.next_queue_index + offset) % self.queue_order.len();
            let queue_id = self.queue_order[index];
            if self.blocked_queues.contains_key(&queue_id) {
                continue;
            }
            if let Some(event) = self.pop_queue_event(queue_id) {
                self.active_queue = Some(queue_id);
                return OrderedEventPoll::Event {
                    source: OrderedEventSource::GuestQueue(queue_id),
                    event,
                };
            }
        }

        if self.pending_event_count() != 0 {
            OrderedEventPoll::Blocked
        } else if self.disconnected {
            OrderedEventPoll::Disconnected
        } else {
            OrderedEventPoll::Empty
        }
    }

    pub(crate) fn shared_event_host_effect_pending_statistics(
        &self,
        now: std::time::Instant,
    ) -> SharedEventHostEffectPendingStatistics {
        let mut statistics = SharedEventHostEffectPendingStatistics::default();
        for reason in self.blocked_queues.values().chain(self.host_block.iter()) {
            if let QueueBlockReason::GuestSharedEventHostEffect { latency, .. } = reason {
                statistics.record(
                    latency.phase,
                    now.saturating_duration_since(latency.phase_started_at),
                );
            }
        }
        statistics
    }

    pub(crate) fn pop_queue_event(&mut self, queue_id: u64) -> Option<GpuEvent> {
        let queue = self.submissions.get_mut(&queue_id)?;
        while queue
            .front()
            .is_some_and(|submission| submission.events.is_empty())
        {
            queue.pop_front();
        }
        let ingested_at = queue.front()?.ingested_at;
        let event = queue.front_mut()?.events.pop_front()?;
        self.last_event_ingested_at = ingested_at;
        self.last_event_queue_wait = ingested_at.map_or(Duration::ZERO, |at| {
            Instant::now().saturating_duration_since(at)
        });
        self.unindex_queued_event_producers(queue_id, &event);
        Some(event)
    }

    pub(crate) fn finish_event(&mut self, source: OrderedEventSource) {
        remu_gpu::record_gpu_work_item_processed();
        let OrderedEventSource::GuestQueue(queue_id) = source else {
            return;
        };
        let completed_submission = self.submissions.get_mut(&queue_id).is_some_and(|queue| {
            if queue
                .front()
                .is_some_and(|submission| submission.events.is_empty())
            {
                queue.pop_front();
                true
            } else {
                false
            }
        });
        if completed_submission {
            self.blocked_queues.remove(&queue_id);
            self.active_queue = None;
            self.advance_after(queue_id);
        }
    }

    pub(crate) fn block_event(
        &mut self,
        source: OrderedEventSource,
        event: GpuEvent,
        reason: QueueBlockReason,
    ) {
        match source {
            OrderedEventSource::Host { ingress_sequence } => {
                self.host_events.push_front(ScheduledHostEvent {
                    ingress_sequence,
                    ingested_at: self.last_event_ingested_at,
                    event,
                });
                self.host_block = Some(reason);
            }
            OrderedEventSource::GuestQueue(queue_id) => {
                self.index_queued_event_producers(queue_id, &event);
                self.submissions
                    .get_mut(&queue_id)
                    .and_then(|queue| queue.front_mut())
                    .expect("blocked guest event retains its owning submission")
                    .events
                    .push_front(event);
                self.blocked_queues.insert(queue_id, reason);
                self.active_queue = None;
                self.advance_after(queue_id);
            }
        }
    }

    pub(crate) fn advance_after(&mut self, queue_id: u64) {
        if let Some(index) = self.queue_order.iter().position(|&id| id == queue_id) {
            self.next_queue_index = (index + 1) % self.queue_order.len().max(1);
        }
    }

    pub(crate) fn has_pending_terminal_memory_wait(&self) -> bool {
        self.blocked_queues.values().any(|reason| {
            matches!(reason, QueueBlockReason::GuestMemoryWaitPendingTerminal(_))
        }) || matches!(
            self.host_block,
            Some(QueueBlockReason::GuestMemoryWaitPendingTerminal(_))
        )
    }

    pub(crate) fn retry_pending_terminal_memory_waits(&mut self) {
        self.blocked_queues.retain(|_, reason| {
            !matches!(
                reason,
                QueueBlockReason::GuestMemoryWaitPendingTerminal(_)
            )
        });
        if matches!(
            self.host_block,
            Some(QueueBlockReason::GuestMemoryWaitPendingTerminal(_))
        ) {
            self.host_block = None;
        }
    }

    pub(crate) fn retry_guest_memory_waits(&mut self) {
        self.ingest_guest_memory_wait_dependencies();
        // Producerless waits may be satisfied by an external CPU store, so retain their periodic
        // polling. A wait whose satisfying value is already in a Metal batch is instead woken by
        // exact-range terminal publication; re-polling it here only spins on the same future.
        self.blocked_queues
            .retain(|_, reason| !matches!(reason, QueueBlockReason::GuestMemoryWait(_)));
        if matches!(self.host_block, Some(QueueBlockReason::GuestMemoryWait(_))) {
            self.host_block = None;
        }
    }

    pub(crate) fn retry_guest_memory_waits_overlapping(&mut self, address: u64, byte_count: u64) {
        self.blocked_queues.retain(|_, reason| {
            !matches!(
                reason,
                QueueBlockReason::GuestMemoryWait(wait)
                    | QueueBlockReason::GuestMemoryWaitPendingTerminal(wait)
                    if guest_wait_overlaps_range(*wait, address, byte_count)
            )
        });
        if self.host_block.as_ref().is_some_and(|reason| {
            matches!(
                reason,
                QueueBlockReason::GuestMemoryWait(wait)
                    | QueueBlockReason::GuestMemoryWaitPendingTerminal(wait)
                    if guest_wait_overlaps_range(*wait, address, byte_count)
            )
        }) {
            self.host_block = None;
        }
    }

    /// Wakes only registered waits which just received an exact SharedEvent producer binding.
    ///
    /// The retained queue-front event carries the process-unique ticket generation. Matching that
    /// identity avoids both the periodic-poll delay and the stale-address bug an address-only wake
    /// would introduce when Unreal recycles its label pool.
    pub(crate) fn retry_registered_guest_memory_waits(&mut self, wait_ids: &[GpuWaitId]) {
        self.blocked_queues.retain(|queue_id, reason| {
            if !matches!(reason, QueueBlockReason::GuestMemoryWait(_)) {
                return true;
            }
            let registered_id = self
                .submissions
                .get(queue_id)
                .and_then(|queue| queue.front())
                .and_then(|submission| submission.events.front())
                .and_then(|event| match event {
                    GpuEvent::RegisteredMemoryWait { id, .. } => Some(*id),
                    _ => None,
                });
            !registered_id.is_some_and(|id| wait_ids.contains(&id))
        });
    }

    pub(crate) fn prefetch_pipeline_work(
        &mut self,
        receive_limit: usize,
        work_limit: usize,
        mut include: impl FnMut(&GpuEvent) -> bool,
    ) -> Vec<PipelineCompileCandidate> {
        self.ingest(receive_limit);
        if work_limit == 0 {
            return Vec::new();
        }

        // Treat host events and each guest queue as independent lookahead sources. A per-source
        // quota prevents one queue with hundreds of future draws from hiding another queue's
        // first pipeline, while the rotating start keeps fairness when there are more sources
        // than the bounded output can cover in one tick.
        let mut sources = Vec::with_capacity(self.queue_order.len().saturating_add(1));
        if self
            .host_events
            .iter()
            .any(|host| event_has_pipeline_work(&host.event))
        {
            sources.push(None);
        }
        sources.extend(self.queue_order.iter().copied().filter_map(|queue_id| {
            self.submissions
                .get(&queue_id)
                .is_some_and(|submissions| {
                    submissions
                        .iter()
                        .flat_map(|submission| submission.events.iter())
                        .any(event_has_pipeline_work)
                })
                .then_some(Some(queue_id))
        }));
        if sources.is_empty() {
            return Vec::new();
        }

        let start = self.next_prefetch_source_index % sources.len();
        let per_source_limit = work_limit.div_ceil(sources.len()).max(1);
        let mut candidates = Vec::with_capacity(work_limit.min(16));
        let mut visited_sources = 0;
        for offset in 0..sources.len() {
            if candidates.len() >= work_limit {
                break;
            }
            visited_sources = offset + 1;
            let source = sources[(start + offset) % sources.len()];
            let events: Box<dyn Iterator<Item = &GpuEvent> + '_> = match source {
                None => Box::new(self.host_events.iter().map(|host| &host.event)),
                Some(queue_id) => Box::new(
                    self.submissions
                        .get(&queue_id)
                        .into_iter()
                        .flatten()
                        .flat_map(|submission| submission.events.iter()),
                ),
            };
            candidates.extend(
                events
                    // Cache and within-pass duplicate filtering happens before cloning a large
                    // immutable draw or dispatch snapshot.
                    .filter(|event| include(event))
                    .filter_map(pipeline_compile_candidate)
                    .take(per_source_limit)
                    .take(work_limit - candidates.len()),
            );
        }
        self.next_prefetch_source_index = (start + visited_sources.max(1)) % sources.len();
        candidates
    }

    pub(crate) fn pending_event_count(&self) -> usize {
        self.host_events.len().saturating_add(
            self.submissions
                .values()
                .flatten()
                .map(|submission| submission.events.len())
                .sum::<usize>(),
        )
    }

    /// Returns whether an already-ingested event can make progress without an external producer.
    ///
    /// The windowed presenter uses this to keep AppKit's event loop active while guest GPU work
    /// is runnable. A short `WaitUntil` deadline is heavily coalesced by macOS and otherwise turns
    /// `GPU_EVENTS_PER_TICK` into an unintended throughput ceiling.
    ///
    /// A queue blocked on shader compilation, or on a memory wait whose producer is already
    /// ingested, does not count as runnable: the application can sleep until the producer runs.
    /// A memory wait with no ingested producer does keep polling, because only a CPU store or a
    /// release still sitting in the channel can satisfy it.
    pub(crate) fn has_runnable_events(&self) -> bool {
        if self.host_block.is_none() && !self.host_events.is_empty() {
            return true;
        }
        if self.queue_order.iter().any(|queue_id| {
            !self.blocked_queues.contains_key(queue_id)
                && self.submissions.get(queue_id).is_some_and(|submissions| {
                    submissions
                        .iter()
                        .any(|submission| !submission.events.is_empty())
                })
        }) {
            return true;
        }
        // Keep AppKit polling while a producerless memory wait may still be satisfied by a
        // CPU store or a not-yet-ingested cross-queue release. Sleeping on WaitUntil here
        // turned title-menu fences into multi-second FPS collapses.
        self.blocked_memory_wait_missing_ingested_producer()
    }

    /// Returns whether a draw writing `address` is parked at a blocked queue front while its
    /// Metal pipeline compiles.
    ///
    /// A cold pipeline gate preserves the draw, but it also holds back the submission boundary
    /// that would normally order the frame. Another queue reaching its own boundary must not
    /// publish the VideoOut allocation in that state: Octopath's compositor clears the target,
    /// parks its fullscreen cover on a cold pipeline, and would otherwise present the bare clear
    /// and then discard the cover's output as belonging to the next frame.
    pub(crate) fn pipeline_blocked_draw_writes(&self, address: u64) -> bool {
        self.queue_order.iter().any(|queue_id| {
            matches!(
                self.blocked_queues.get(queue_id),
                Some(QueueBlockReason::GraphicsPipeline(_))
            ) && self
                .submissions
                .get(queue_id)
                .and_then(|submissions| {
                    submissions
                        .iter()
                        .find_map(|submission| submission.events.front())
                })
                .is_some_and(|event| match event {
                    GpuEvent::IndexedDraw(draw) => draw.color_target_address == address,
                    _ => false,
                })
        })
    }

    pub(crate) fn block_reason_counts(&self) -> (usize, usize, usize) {
        let mut pipeline = 0;
        let mut memory = 0;
        let mut flip = 0;
        for reason in self.blocked_queues.values().chain(self.host_block.iter()) {
            match reason {
                QueueBlockReason::GraphicsPipeline(_) | QueueBlockReason::ComputePipeline(_) => {
                    pipeline += 1;
                }
                QueueBlockReason::GuestMemoryWait(_)
                | QueueBlockReason::GuestMemoryWaitPendingTerminal(_)
                | QueueBlockReason::GuestSharedEventHostEffect { .. } => memory += 1,
                QueueBlockReason::FlipSafetyWait(_) => flip += 1,
            }
        }
        (pipeline, memory, flip)
    }

    pub(crate) fn queue_diagnostics(&self) -> String {
        // The host-source queue can block (for example on a flip-safety wait) while every guest
        // queue prints empty/none; the 2026-08-12 RenderThread-starve wedge was invisible here
        // for exactly that reason. Always append the host queue's state.
        let host_blocked = self
            .host_block
            .as_ref()
            .map_or("none".to_owned(), |reason| match reason {
                QueueBlockReason::GuestMemoryWait(wait) => format!("memory:{:#x}", wait.address),
                QueueBlockReason::GuestMemoryWaitPendingTerminal(wait) => {
                    format!("memory-terminal:{:#x}", wait.address)
                }
                QueueBlockReason::GuestSharedEventHostEffect { .. } => {
                    "shared-event-host".to_owned()
                }
                QueueBlockReason::FlipSafetyWait(wait) => format!(
                    "flip-safety:{:#x}/index:{}/required:{:?}",
                    wait.address, wait.buffer_index, wait.required_flip_sequence
                ),
                QueueBlockReason::GraphicsPipeline(key) => format!(
                    "pipeline:{:#x}/{:#x}",
                    key.export_shader_address, key.pixel_shader_address
                ),
                QueueBlockReason::ComputePipeline(key) => {
                    format!("compute-pipeline:{:#x}", key.program_identity.0.address)
                }
            });
        let host = format!(
            "host:pending:{}/blocked:{host_blocked}",
            self.host_events.len()
        );
        self.queue_order
            .iter()
            .take(8)
            .filter_map(|queue_id| {
                let submissions = self.submissions.get(queue_id)?;
                let pending = submissions
                    .iter()
                    .map(|submission| submission.events.len())
                    .sum::<usize>();
                let front = submissions
                    .front()
                    .and_then(|submission| submission.events.front())
                    .map_or_else(
                        || "empty".to_owned(),
                        |event| match event {
                            GpuEvent::RegisteredMemoryWait { id, wait } => format!(
                                "wait:{id:?}@{:#x}/ref:{:#x}/mask:{:#x}/cmp:{}",
                                wait.address, wait.reference, wait.mask, wait.compare_function
                            ),
                            GpuEvent::MemoryWait(wait) => format!(
                                "wait:unregistered@{:#x}/ref:{:#x}/mask:{:#x}/cmp:{}",
                                wait.address, wait.reference, wait.mask, wait.compare_function
                            ),
                            GpuEvent::IndexedDraw(draw) => format!(
                                "draw:{}/es:{:#x}/ps:{:#x}",
                                draw.sequence,
                                draw.export_shader_address,
                                draw.pixel_shader_address
                            ),
                            GpuEvent::ComputeDispatch(dispatch) => {
                                format!(
                                    "compute:{}/cs:{:#x}",
                                    dispatch.sequence, dispatch.shader_address
                                )
                            }
                            GpuEvent::Flip { address, .. } => format!("flip:{address:#x}"),
                            GpuEvent::WaitFlipDone(wait) => format!(
                                "wait-flip:{:#x}/index:{}/required:{:?}",
                                wait.address, wait.buffer_index, wait.required_flip_sequence
                            ),
                            GpuEvent::SubmissionBoundary(sequence) => {
                                format!("boundary:{sequence}")
                            }
                            _ => "other".to_owned(),
                        },
                    );
                let blocked =
                    self.blocked_queues
                        .get(queue_id)
                        .map_or("none".to_owned(), |reason| match reason {
                            QueueBlockReason::GuestMemoryWait(wait) => {
                                format!("memory:{:#x}", wait.address)
                            }
                            QueueBlockReason::GuestMemoryWaitPendingTerminal(wait) => {
                                format!("memory-terminal:{:#x}", wait.address)
                            }
                            QueueBlockReason::GuestSharedEventHostEffect {
                                consumer_queue_id,
                                consumer_ack,
                                terminal_dependencies,
                                ..
                            } => {
                                format!(
                                    "shared-event-host:q{consumer_queue_id}/ack:{}/terminal:{}",
                                    consumer_ack.is_some(),
                                    terminal_dependencies.len()
                                )
                            }
                            QueueBlockReason::FlipSafetyWait(wait) => format!(
                                "flip-safety:{:#x}/index:{}/required:{:?}",
                                wait.address, wait.buffer_index, wait.required_flip_sequence
                            ),
                            QueueBlockReason::GraphicsPipeline(key) => format!(
                                "pipeline:{:#x}/{:#x}",
                                key.export_shader_address, key.pixel_shader_address
                            ),
                            QueueBlockReason::ComputePipeline(key) => {
                                format!("compute-pipeline:{:#x}", key.program_identity.0.address)
                            }
                        });
                Some(format!(
                    "q{queue_id}:pending:{pending}/submissions:{}/blocked:{blocked}/front:{front}",
                    submissions.len()
                ))
            })
            .chain(std::iter::once(host))
            .collect::<Vec<_>>()
            .join("; ")
    }

    pub(crate) fn blocked_wait_dependency_diagnostics(&self) -> String {
        self.queue_order
            .iter()
            .filter_map(|queue_id| {
                let wait = match self.blocked_queues.get(queue_id)? {
                    QueueBlockReason::GuestMemoryWait(wait)
                    | QueueBlockReason::GuestMemoryWaitPendingTerminal(wait) => wait,
                    _ => return None,
                };
                Some(format!(
                    "q{queue_id}:wait@{:#x}/ref:{:#x}/mask:{:#x}/cmp:{}/producers:[{}]",
                    wait.address,
                    wait.reference,
                    wait.mask,
                    wait.compare_function,
                    self.describe_queued_wait_producers(*wait),
                ))
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    /// Describes already-ingested packets which can write the label watched by `wait`.
    ///
    /// This is diagnostic only: queue-local ordering must still be preserved. It distinguishes a
    /// missing producer from a producer trapped behind another blocked queue front, which look
    /// identical when only the live label value is logged.
    pub(crate) fn describe_queued_wait_producers(&self, wait: GpuMemoryWait) -> String {
        let wait_bytes = if wait.is_64_bit { 8 } else { 4 };
        let wait_end = wait.address.saturating_add(wait_bytes);
        let overlaps_wait = |address: u64, byte_count: u64| {
            address < wait_end && wait.address < address.saturating_add(byte_count)
        };
        let binding_overlaps = |base_address: u64, byte_count: usize, writable: bool| {
            writable
                && u64::try_from(byte_count)
                    .is_ok_and(|byte_count| overlaps_wait(base_address, byte_count))
        };
        let mut producers = Vec::new();

        for queue_id in &self.queue_order {
            let Some(submissions) = self.submissions.get(queue_id) else {
                continue;
            };
            for (submission_index, submission) in submissions.iter().enumerate() {
                for (event_index, event) in submission.events.iter().enumerate() {
                    let description = match event {
                        GpuEvent::MemoryRelease(release) => {
                            let (byte_count, value) = match release.value {
                                GpuMemoryReleaseValue::Immediate32(value) => {
                                    (4, format!("{value:#x}"))
                                }
                                GpuMemoryReleaseValue::Immediate64(value) => {
                                    (8, format!("{value:#x}"))
                                }
                                GpuMemoryReleaseValue::Timestamp => (8, "timestamp".to_owned()),
                            };
                            overlaps_wait(release.address, byte_count)
                                .then(|| format!("release@{:#x}={value}", release.address))
                        }
                        GpuEvent::MemoryWrite(write) => {
                            let byte_count = if write.increment_address {
                                u64::try_from(write.values.len())
                                    .unwrap_or(u64::MAX / 4)
                                    .saturating_mul(4)
                            } else {
                                4
                            };
                            overlaps_wait(write.address, byte_count).then(|| {
                                format!(
                                    "write@{:#x}/dwords:{}/increment:{}",
                                    write.address,
                                    write.values.len(),
                                    write.increment_address
                                )
                            })
                        }
                        GpuEvent::MemoryCopy(copy) => {
                            overlaps_wait(copy.destination_address, copy.byte_count).then(|| {
                                format!(
                                    "dma@{:#x}/bytes:{:#x}",
                                    copy.destination_address, copy.byte_count
                                )
                            })
                        }
                        GpuEvent::ComputeDispatch(dispatch) => dispatch
                            .shader_evaluation
                            .as_ref()
                            .and_then(|evaluation| {
                                evaluation.global_memory_bindings.iter().find(|binding| {
                                    binding_overlaps(
                                        binding.base_address,
                                        binding.byte_count,
                                        binding.writable,
                                    )
                                })
                            })
                            .map(|binding| {
                                format!(
                                    "compute:{}/cs:{:#x}/global:{:#x}+{:#x}",
                                    dispatch.sequence,
                                    dispatch.shader_address,
                                    binding.base_address,
                                    binding.byte_count
                                )
                            }),
                        GpuEvent::IndexedDraw(draw) => draw
                            .export_shader_evaluation
                            .iter()
                            .chain(draw.pixel_shader_evaluation.iter())
                            .flat_map(|evaluation| &evaluation.global_memory_bindings)
                            .find(|binding| {
                                binding_overlaps(
                                    binding.base_address,
                                    binding.byte_count,
                                    binding.writable,
                                )
                            })
                            .map(|binding| {
                                format!(
                                    "draw:{}/es:{:#x}/ps:{:#x}/global:{:#x}+{:#x}",
                                    draw.sequence,
                                    draw.export_shader_address,
                                    draw.pixel_shader_address,
                                    binding.base_address,
                                    binding.byte_count
                                )
                            }),
                        _ => None,
                    };
                    if let Some(description) = description {
                        producers.push(format!(
                            "q{queue_id}/s{submission_index}/e{event_index}:{description}"
                        ));
                        if producers.len() == 16 {
                            return producers.join(",");
                        }
                    }
                }
            }
        }

        if producers.is_empty() {
            "none-ingested".to_owned()
        } else {
            producers.join(",")
        }
    }
}

fn event_has_pipeline_work(event: &GpuEvent) -> bool {
    matches!(
        event,
        GpuEvent::IndexedDraw(_) | GpuEvent::ComputeDispatch(_)
    )
}

fn event_is_memory_wait(event: &GpuEvent) -> bool {
    matches!(
        event,
        GpuEvent::MemoryWait(_) | GpuEvent::RegisteredMemoryWait { .. }
    )
}

fn pipeline_compile_candidate(event: &GpuEvent) -> Option<PipelineCompileCandidate> {
    match event {
        GpuEvent::IndexedDraw(draw) => {
            Some(PipelineCompileCandidate::Graphics(draw.as_ref().clone()))
        }
        GpuEvent::ComputeDispatch(dispatch) => {
            Some(PipelineCompileCandidate::Compute(dispatch.as_ref().clone()))
        }
        _ => None,
    }
}

/// Host display management and VideoOut presentation do not execute on a guest AGC ring.
///
/// They retain ingress order against runnable or pipeline-gated guest queues, but an unrelated
/// queue parked on a WAIT_REG_MEM cannot prevent a newer display registration, system splash,
/// direct VideoOut flip, or shutdown from reaching the window thread.
fn host_event_bypasses_blocked_guest_queues(event: &GpuEvent) -> bool {
    matches!(
        event,
        GpuEvent::DisplayRegistered { .. }
            | GpuEvent::ShowHostSplash(_)
            | GpuEvent::HideHostSplash
            | GpuEvent::ShowHostDialog(_)
            | GpuEvent::HideHostDialog { .. }
            | GpuEvent::ShowHostToast(_)
            | GpuEvent::Flip { .. }
            | GpuEvent::Shutdown
    )
}

fn guest_wait_overlaps_range(wait: GpuMemoryWait, address: u64, byte_count: u64) -> bool {
    if byte_count == 0 {
        return false;
    }
    let wait_byte_count = if wait.is_64_bit { 8 } else { 4 };
    address < wait.address.saturating_add(wait_byte_count)
        && wait.address < address.saturating_add(byte_count)
}
