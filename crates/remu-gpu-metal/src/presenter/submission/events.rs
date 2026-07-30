// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    GPU_EVENTS_PER_TICK, OrderedEventPoll, OrderedEventSource, OrderedGpuEventReceiver,
    PipelineCompileCandidate, QueueBlockReason, ScheduledGpuSubmission, ScheduledHostEvent,
};
use remu_gpu::{GpuEvent, GpuMemoryReleaseValue, GpuMemoryWait};
use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{Receiver, TryRecvError};

/// Extra decoded-event storage used only to discover an independent queue behind blocked tails.
///
/// Two complete bounded-channel windows cover a producer which was blocked while the receiver
/// filled its ordinary pending budget, plus the next window made available by draining it. This is
/// deliberately finite: a malformed or genuinely cyclic stream containing only more work for
/// blocked queues must retain backpressure instead of moving an unbounded command stream into the
/// presenter.
const BLOCKED_DEPENDENCY_LOOKAHEAD_CAPACITY: usize = remu_gpu::GPU_EVENT_QUEUE_CAPACITY
    * remu_gpu::GPU_EVENT_SUBMISSION_CHUNK_SIZE
    * 2;

impl OrderedGpuEventReceiver {
    pub(crate) fn new(receiver: Receiver<GpuEvent>) -> Self {
        Self {
            receiver,
            submissions: HashMap::new(),
            queue_order: Vec::new(),
            next_queue_index: 0,
            active_queue: None,
            blocked_queues: HashMap::new(),
            host_events: VecDeque::new(),
            host_block: None,
            next_ingress_sequence: 0,
            disconnected: false,
        }
    }

    pub(crate) fn ingest(&mut self, receive_limit: usize) {
        if self.disconnected {
            return;
        }
        for _ in 0..receive_limit {
            let pending_capacity = if self.needs_dependency_ingress() {
                remu_gpu::GPU_PENDING_EVENT_CAPACITY
                    .saturating_add(BLOCKED_DEPENDENCY_LOOKAHEAD_CAPACITY)
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
            match event {
                GpuEvent::QueueSubmission { queue_id, events } => {
                    if events.is_empty() {
                        continue;
                    }
                    if !self.submissions.contains_key(&queue_id) {
                        self.queue_order.push(queue_id);
                    }
                    self.submissions.entry(queue_id).or_default().push_back(
                        ScheduledGpuSubmission {
                            ingress_sequence,
                            events: events.into(),
                        },
                    );
                }
                event => self.host_events.push_back(ScheduledHostEvent {
                    ingress_sequence,
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
    /// Move a finite reserve of already-decoded chunks into queue FIFOs only while every known
    /// guest queue is memory-wait blocked. As soon as one newly discovered queue is runnable, the
    /// next loop iteration restores the normal cap. If no runnable queue is found within the
    /// reserve, ingress stops with backpressure intact. No event executes out of order, and a
    /// blocked queue's tail remains behind its wait.
    fn needs_dependency_ingress(&self) -> bool {
        let has_memory_wait = self
            .blocked_queues
            .values()
            .any(|reason| matches!(reason, QueueBlockReason::GuestMemoryWait(_)))
            || matches!(self.host_block, Some(QueueBlockReason::GuestMemoryWait(_)));
        has_memory_wait && !self.has_runnable_events()
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
            self.ingest(GPU_EVENTS_PER_TICK);
        }
    }

    pub(crate) fn try_recv(
        &mut self,
        pipeline_resolved: impl Fn(&QueueBlockReason) -> bool,
    ) -> OrderedEventPoll {
        self.ingest(GPU_EVENTS_PER_TICK);
        self.blocked_queues.retain(|_, reason| match reason {
            QueueBlockReason::GuestMemoryWait(_) => true,
            reason @ (QueueBlockReason::GraphicsPipeline(_)
            | QueueBlockReason::ComputePipeline(_)) => !pipeline_resolved(reason),
        });
        if self.host_block.as_ref().is_some_and(|reason| match reason {
            QueueBlockReason::GuestMemoryWait(_) => false,
            reason @ (QueueBlockReason::GraphicsPipeline(_)
            | QueueBlockReason::ComputePipeline(_)) => pipeline_resolved(reason),
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
                        Some(QueueBlockReason::GuestMemoryWait(_))
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
            return OrderedEventPoll::Event {
                source: OrderedEventSource::Host {
                    ingress_sequence: host.ingress_sequence,
                },
                event: host.event,
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
            return OrderedEventPoll::Disconnected;
        } else {
            OrderedEventPoll::Empty
        }
    }

    pub(crate) fn pop_queue_event(&mut self, queue_id: u64) -> Option<GpuEvent> {
        let queue = self.submissions.get_mut(&queue_id)?;
        while queue
            .front()
            .is_some_and(|submission| submission.events.is_empty())
        {
            queue.pop_front();
        }
        queue.front_mut()?.events.pop_front()
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
                    event,
                });
                self.host_block = Some(reason);
            }
            OrderedEventSource::GuestQueue(queue_id) => {
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

    pub(crate) fn retry_guest_memory_waits(&mut self) {
        self.ingest_guest_memory_wait_dependencies();
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
                    if guest_wait_overlaps_range(*wait, address, byte_count)
            )
        });
        if self.host_block.as_ref().is_some_and(|reason| {
            matches!(
                reason,
                QueueBlockReason::GuestMemoryWait(wait)
                    if guest_wait_overlaps_range(*wait, address, byte_count)
            )
        }) {
            self.host_block = None;
        }
    }

    pub(crate) fn prefetch_pipeline_work(
        &mut self,
        receive_limit: usize,
        work_limit: usize,
    ) -> Vec<PipelineCompileCandidate> {
        self.ingest(receive_limit);
        let host_work = self.host_events.iter().filter_map(|host| match &host.event {
            GpuEvent::IndexedDraw(draw) => {
                Some(PipelineCompileCandidate::Graphics(draw.as_ref().clone()))
            }
            GpuEvent::ComputeDispatch(dispatch) => Some(PipelineCompileCandidate::Compute(
                dispatch.as_ref().clone(),
            )),
            _ => None,
        });
        let queue_work = self.queue_order.iter().flat_map(|queue_id| {
            self.submissions
                .get(queue_id)
                .into_iter()
                .flatten()
                .flat_map(|submission| submission.events.iter())
                .filter_map(|event| match event {
                    GpuEvent::IndexedDraw(draw) => {
                        Some(PipelineCompileCandidate::Graphics(draw.as_ref().clone()))
                    }
                    GpuEvent::ComputeDispatch(dispatch) => Some(PipelineCompileCandidate::Compute(
                        dispatch.as_ref().clone(),
                    )),
                    _ => None,
                })
        });
        host_work.chain(queue_work).take(work_limit).collect()
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
    /// `GPU_EVENTS_PER_TICK` into an unintended throughput ceiling. Dependency-blocked queues do
    /// not count as runnable, so a guest waiting on a CPU write or shader compilation still lets
    /// the application sleep.
    pub(crate) fn has_runnable_events(&self) -> bool {
        if self.host_block.is_none() && !self.host_events.is_empty() {
            return true;
        }
        self.queue_order.iter().any(|queue_id| {
            !self.blocked_queues.contains_key(queue_id)
                && self.submissions.get(queue_id).is_some_and(|submissions| {
                    submissions
                        .iter()
                        .any(|submission| !submission.events.is_empty())
                })
        })
    }

    pub(crate) fn block_reason_counts(&self) -> (usize, usize) {
        let mut pipeline = 0;
        let mut memory = 0;
        for reason in self.blocked_queues.values().chain(self.host_block.iter()) {
            match reason {
                QueueBlockReason::GraphicsPipeline(_) | QueueBlockReason::ComputePipeline(_) => {
                    pipeline += 1;
                }
                QueueBlockReason::GuestMemoryWait(_) => memory += 1,
            }
        }
        (pipeline, memory)
    }

    pub(crate) fn queue_diagnostics(&self) -> String {
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
            .collect::<Vec<_>>()
            .join("; ")
    }

    pub(crate) fn blocked_wait_dependency_diagnostics(&self) -> String {
        self.queue_order
            .iter()
            .filter_map(|queue_id| {
                let QueueBlockReason::GuestMemoryWait(wait) = self.blocked_queues.get(queue_id)?
                else {
                    return None;
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
