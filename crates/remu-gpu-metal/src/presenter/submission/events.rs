// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    GPU_EVENTS_PER_TICK, GraphicsPipelineKey, OrderedEventPoll, OrderedEventSource,
    OrderedGpuEventReceiver, QueueBlockReason, ScheduledGpuSubmission, ScheduledHostEvent,
};
use remu_gpu::{GpuEvent, IndexedDraw};
use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{Receiver, TryRecvError};

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
            if self.pending_event_count()
                > remu_gpu::GPU_PENDING_EVENT_CAPACITY
                    .saturating_sub(remu_gpu::GPU_EVENT_SUBMISSION_CHUNK_SIZE)
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

    pub(crate) fn try_recv(
        &mut self,
        pipeline_resolved: impl Fn(&GraphicsPipelineKey) -> bool,
    ) -> OrderedEventPoll {
        self.ingest(GPU_EVENTS_PER_TICK);
        self.blocked_queues.retain(|_, reason| match reason {
            QueueBlockReason::GuestMemoryWait => true,
            QueueBlockReason::GraphicsPipeline(key) => !pipeline_resolved(key),
        });
        if self.host_block.as_ref().is_some_and(|reason| match reason {
            QueueBlockReason::GuestMemoryWait => false,
            QueueBlockReason::GraphicsPipeline(key) => pipeline_resolved(key),
        }) {
            self.host_block = None;
        }

        let oldest_submission = self
            .submissions
            .values()
            .filter_map(|queue| queue.front())
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
        self.blocked_queues
            .retain(|_, reason| !matches!(reason, QueueBlockReason::GuestMemoryWait));
        if matches!(self.host_block, Some(QueueBlockReason::GuestMemoryWait)) {
            self.host_block = None;
        }
    }

    pub(crate) fn prefetch_draws(
        &mut self,
        receive_limit: usize,
        draw_limit: usize,
    ) -> Vec<IndexedDraw> {
        self.ingest(receive_limit);
        let host_draws = self
            .host_events
            .iter()
            .filter_map(|host| match &host.event {
                GpuEvent::IndexedDraw(draw) => Some(draw.as_ref().clone()),
                _ => None,
            });
        let queue_draws = self.queue_order.iter().flat_map(|queue_id| {
            self.submissions
                .get(queue_id)
                .into_iter()
                .flatten()
                .flat_map(|submission| submission.events.iter())
                .filter_map(|event| match event {
                    GpuEvent::IndexedDraw(draw) => Some(draw.as_ref().clone()),
                    _ => None,
                })
        });
        host_draws.chain(queue_draws).take(draw_limit).collect()
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

    pub(crate) fn block_reason_counts(&self) -> (usize, usize) {
        let mut pipeline = 0;
        let mut memory = 0;
        for reason in self.blocked_queues.values().chain(self.host_block.iter()) {
            match reason {
                QueueBlockReason::GraphicsPipeline(_) => pipeline += 1,
                QueueBlockReason::GuestMemoryWait => memory += 1,
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
                            QueueBlockReason::GuestMemoryWait => "memory".to_owned(),
                            QueueBlockReason::GraphicsPipeline(key) => format!(
                                "pipeline:{:#x}/{:#x}",
                                key.export_shader_address, key.pixel_shader_address
                            ),
                        });
                Some(format!(
                    "q{queue_id}:pending:{pending}/submissions:{}/blocked:{blocked}/front:{front}",
                    submissions.len()
                ))
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
}
