// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Queue-typed Metal `SharedEvent` primitives for guest GPU synchronization.
//!
//! This module intentionally owns no presenter queue state. B17 can compose these primitives with
//! per-guest-queue command batches without making a guest label address into fence identity.

use metal::{DeviceRef, SharedEvent, SharedEventRef};
use remu_gpu::GpuWaitId;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Stable identity of the guest queue which owns a `SharedEvent` timeline.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct SharedEventProducerId(u64);

impl SharedEventProducerId {
    #[must_use]
    pub(crate) const fn new(queue_id: u64) -> Self {
        Self(queue_id)
    }

    #[must_use]
    pub(crate) const fn queue_id(self) -> u64 {
        self.0
    }
}

/// One exact point on a producer queue's monotonic `SharedEvent` timeline.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct SharedEventFence {
    producer: SharedEventProducerId,
    value: u64,
}

impl SharedEventFence {
    #[must_use]
    pub(crate) const fn new(producer: SharedEventProducerId, value: u64) -> Self {
        Self { producer, value }
    }

    #[must_use]
    pub(crate) const fn producer(self) -> SharedEventProducerId {
        self.producer
    }

    #[must_use]
    pub(crate) const fn value(self) -> u64 {
        self.value
    }
}

/// A producer queue has allocated every nonzero value in its current event timeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SharedEventTimelineExhausted {
    producer: SharedEventProducerId,
}

impl SharedEventTimelineExhausted {
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn producer(self) -> SharedEventProducerId {
        self.producer
    }
}

impl Display for SharedEventTimelineExhausted {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "Metal SharedEvent timeline for guest queue {} is exhausted",
            self.producer.queue_id()
        )
    }
}

impl Error for SharedEventTimelineExhausted {}

/// A Metal `SharedEvent` and checked monotonic value allocator owned by one producer queue.
pub(crate) struct SharedEventTimeline {
    producer: SharedEventProducerId,
    event: SharedEvent,
    last_allocated_value: u64,
}

impl SharedEventTimeline {
    #[must_use]
    pub(crate) fn new(device: &DeviceRef, producer: SharedEventProducerId) -> Self {
        Self {
            producer,
            event: device.new_shared_event(),
            last_allocated_value: 0,
        }
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn producer(&self) -> SharedEventProducerId {
        self.producer
    }

    #[must_use]
    pub(crate) fn event(&self) -> &SharedEventRef {
        &self.event
    }

    #[must_use]
    pub(crate) fn event_owned(&self) -> SharedEvent {
        self.event.clone()
    }

    /// Allocates the next nonzero value without ever wrapping a live event timeline.
    pub(crate) fn allocate_fence(
        &mut self,
    ) -> Result<SharedEventFence, SharedEventTimelineExhausted> {
        let Some(value) = self.last_allocated_value.checked_add(1) else {
            return Err(SharedEventTimelineExhausted {
                producer: self.producer,
            });
        };
        self.last_allocated_value = value;
        Ok(SharedEventFence {
            producer: self.producer,
            value,
        })
    }

    #[cfg(test)]
    fn set_last_allocated_value_for_test(&mut self, value: u64) {
        self.last_allocated_value = value;
    }
}

/// Result of binding one submitted wait ticket to a concrete producer fence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuWaitFenceBindResult {
    Bound,
    AlreadyBound(SharedEventFence),
}

/// Exact generation-safe bindings from submitted wait tickets to Metal producer fences.
///
/// There is intentionally no label-address index. Guest allocators can recycle an address while an
/// older frame is still in flight; `GpuWaitId` is the existing process-unique generation fence.
#[derive(Default)]
pub(crate) struct GpuWaitFenceMap {
    by_wait: HashMap<GpuWaitId, SharedEventFence>,
    missing_once: HashSet<GpuWaitId>,
}

impl GpuWaitFenceMap {
    pub(crate) fn bind_if_unbound(
        &mut self,
        wait_id: GpuWaitId,
        fence: SharedEventFence,
    ) -> GpuWaitFenceBindResult {
        use std::collections::hash_map::Entry;

        match self.by_wait.entry(wait_id) {
            Entry::Vacant(entry) => {
                self.missing_once.remove(&wait_id);
                entry.insert(fence);
                GpuWaitFenceBindResult::Bound
            }
            Entry::Occupied(entry) => GpuWaitFenceBindResult::AlreadyBound(*entry.get()),
        }
    }

    #[must_use]
    pub(crate) fn get(&self, wait_id: GpuWaitId) -> Option<SharedEventFence> {
        self.by_wait.get(&wait_id).copied()
    }

    pub(crate) fn take(&mut self, wait_id: GpuWaitId) -> Option<SharedEventFence> {
        self.missing_once.remove(&wait_id);
        self.by_wait.remove(&wait_id)
    }

    /// Records the first execution of one exact submitted wait before a producer fence is bound.
    ///
    /// Periodic producerless polling can execute the same queue-front ticket hundreds of times per
    /// second. Counting those attempts as distinct producer misses made the B17 telemetry look like
    /// hundreds of unrelated generations were unbound. The process-unique `GpuWaitId`, not its
    /// recyclable guest label address, is the identity for this diagnostic.
    pub(crate) fn note_missing_once(&mut self, wait_id: GpuWaitId) -> bool {
        if self.by_wait.contains_key(&wait_id) {
            return false;
        }
        self.missing_once.insert(wait_id)
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.by_wait.len()
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.by_wait.is_empty()
    }
}

/// Terminal state relevant to the failed-batch `SharedEvent` retirement policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum SharedEventBatchTerminalState {
    Completed,
    Error,
}

/// A batch tried to retire a fence allocated by a different producer timeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SharedEventProducerMismatch {
    expected: SharedEventProducerId,
    actual: SharedEventProducerId,
}

impl SharedEventProducerMismatch {
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn expected(self) -> SharedEventProducerId {
        self.expected
    }

    #[cfg(test)]
    #[must_use]
    pub(crate) const fn actual(self) -> SharedEventProducerId {
        self.actual
    }
}

impl Display for SharedEventProducerMismatch {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "guest queue {} cannot retire a SharedEvent fence owned by guest queue {}",
            self.expected.queue_id(),
            self.actual.queue_id()
        )
    }
}

impl Error for SharedEventProducerMismatch {}

/// Computes the bounded CPU event advance required by the existing failed-batch retirement policy.
///
/// Successful command buffers must prove ordering through their encoded GPU signal. A failed
/// command buffer can skip an encoded signal, while `REmu` deliberately still retires that batch's
/// command-processor labels and completions. Advancing only through this batch's maximum fence keeps
/// already-submitted consumers from hanging without overtaking values owned by later batches.
pub(crate) fn terminal_shared_event_advance(
    terminal_state: SharedEventBatchTerminalState,
    producer: SharedEventProducerId,
    current_signaled_value: u64,
    batch_maximum_fence: Option<SharedEventFence>,
) -> Result<Option<u64>, SharedEventProducerMismatch> {
    if terminal_state != SharedEventBatchTerminalState::Error {
        return Ok(None);
    }
    let Some(batch_maximum_fence) = batch_maximum_fence else {
        return Ok(None);
    };
    if batch_maximum_fence.producer() != producer {
        return Err(SharedEventProducerMismatch {
            expected: producer,
            actual: batch_maximum_fence.producer(),
        });
    }
    Ok((current_signaled_value < batch_maximum_fence.value())
        .then_some(batch_maximum_fence.value()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presenter::completion::{COMMAND_BUFFER_SETTLE_TIMEOUT, wait_until_settled_within};
    use metal::{Device, MTLCommandBufferStatus, MTLResourceOptions, NSRange};
    use remu_gpu::{GpuEvent, GpuMemoryWait};
    use std::time::Duration;

    fn two_registered_wait_ids_for_same_address() -> [GpuWaitId; 2] {
        remu_gpu::clear_event_sink();
        let (sender, receiver) = remu_gpu::gpu_event_channel();
        remu_gpu::install_event_sink(sender);
        let wait = GpuMemoryWait {
            address: 0x7000_1000,
            reference: 1,
            mask: u64::from(u32::MAX),
            compare_function: 3,
            is_64_bit: false,
        };
        assert!(remu_gpu::emit_event(GpuEvent::QueueSubmission {
            queue_id: 0,
            events: vec![GpuEvent::MemoryWait(wait), GpuEvent::MemoryWait(wait)],
        }));
        let GpuEvent::QueueSubmission { events, .. } = receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("registered wait submission")
        else {
            panic!("expected a registered queue submission");
        };
        let ids = events
            .iter()
            .filter_map(|event| match event {
                GpuEvent::RegisteredMemoryWait { id, .. } => Some(*id),
                _ => None,
            })
            .collect::<Vec<_>>();
        remu_gpu::clear_event_sink();
        ids.try_into().expect("two registered wait IDs")
    }

    #[test]
    fn producer_timelines_allocate_independent_monotonic_values() {
        let device = Device::system_default().expect("Metal device");
        let first_producer = SharedEventProducerId::new(0);
        let second_producer = SharedEventProducerId::new(41);
        let mut first = SharedEventTimeline::new(&device, first_producer);
        let mut second = SharedEventTimeline::new(&device, second_producer);

        let first_one = first.allocate_fence().expect("first fence");
        let first_two = first.allocate_fence().expect("second fence");
        let second_one = second.allocate_fence().expect("independent fence");

        assert_eq!(first.producer(), first_producer);
        assert_eq!(second.producer(), second_producer);
        assert_eq!(first_one.value(), 1);
        assert_eq!(first_two.value(), 2);
        assert_eq!(second_one.value(), 1);
        assert_eq!(first_one.producer(), first_producer);
        assert_eq!(second_one.producer(), second_producer);
        assert!(!std::ptr::eq(first.event(), second.event()));
    }

    #[test]
    fn producer_timeline_stops_at_checked_overflow_without_wrapping() {
        let device = Device::system_default().expect("Metal device");
        let producer = SharedEventProducerId::new(17);
        let mut timeline = SharedEventTimeline::new(&device, producer);
        timeline.set_last_allocated_value_for_test(u64::MAX - 1);

        let last = timeline.allocate_fence().expect("last representable fence");
        assert_eq!(last.value(), u64::MAX);
        let error = timeline
            .allocate_fence()
            .expect_err("a live timeline must never wrap");
        assert_eq!(error.producer(), producer);
        assert_eq!(
            timeline
                .allocate_fence()
                .expect_err("overflow remains terminal for this timeline"),
            error
        );
    }

    #[test]
    fn wait_fence_map_keys_exact_ticket_generations_not_recycled_addresses() {
        let [old_generation, new_generation] = two_registered_wait_ids_for_same_address();
        assert_ne!(old_generation, new_generation);

        let device = Device::system_default().expect("Metal device");
        let mut first_timeline = SharedEventTimeline::new(&device, SharedEventProducerId::new(41));
        let mut second_timeline = SharedEventTimeline::new(&device, SharedEventProducerId::new(42));
        let old_fence = first_timeline.allocate_fence().expect("old fence");
        let new_fence = second_timeline.allocate_fence().expect("new fence");
        let replacement = first_timeline.allocate_fence().expect("replacement fence");
        let mut bindings = GpuWaitFenceMap::default();

        assert!(bindings.note_missing_once(old_generation));
        assert!(!bindings.note_missing_once(old_generation));
        assert!(bindings.note_missing_once(new_generation));
        assert_eq!(
            bindings.bind_if_unbound(old_generation, old_fence),
            GpuWaitFenceBindResult::Bound
        );
        assert_eq!(
            bindings.bind_if_unbound(new_generation, new_fence),
            GpuWaitFenceBindResult::Bound
        );
        assert_eq!(
            bindings.bind_if_unbound(old_generation, replacement),
            GpuWaitFenceBindResult::AlreadyBound(old_fence),
            "the first concrete producer wins for one submitted ticket"
        );
        assert_eq!(bindings.len(), 2);
        assert_eq!(bindings.get(old_generation), Some(old_fence));
        assert_eq!(bindings.get(new_generation), Some(new_fence));
        assert!(!bindings.note_missing_once(old_generation));
        assert_eq!(bindings.take(old_generation), Some(old_fence));
        assert_eq!(bindings.get(old_generation), None);
        assert_eq!(bindings.get(new_generation), Some(new_fence));
        assert!(!bindings.is_empty());
        assert_eq!(bindings.take(new_generation), Some(new_fence));
        assert!(bindings.is_empty());
    }

    #[test]
    fn terminal_error_advance_stops_at_this_batchs_owned_maximum() {
        let device = Device::system_default().expect("Metal device");
        let producer = SharedEventProducerId::new(41);
        let mut timeline = SharedEventTimeline::new(&device, producer);
        let first = timeline.allocate_fence().expect("first fence");
        let batch_maximum = timeline.allocate_fence().expect("batch maximum");
        let later_batch = timeline.allocate_fence().expect("later batch fence");

        assert_eq!(
            terminal_shared_event_advance(
                SharedEventBatchTerminalState::Error,
                producer,
                first.value(),
                Some(batch_maximum),
            ),
            Ok(Some(batch_maximum.value()))
        );
        assert_ne!(batch_maximum.value(), later_batch.value());
        assert_eq!(
            terminal_shared_event_advance(
                SharedEventBatchTerminalState::Error,
                producer,
                batch_maximum.value(),
                Some(batch_maximum),
            ),
            Ok(None),
            "an already-reached value needs no CPU event write"
        );
        assert_eq!(
            terminal_shared_event_advance(
                SharedEventBatchTerminalState::Error,
                producer,
                later_batch.value(),
                Some(batch_maximum),
            ),
            Ok(None),
            "retirement never moves an event backwards"
        );
        assert_eq!(
            terminal_shared_event_advance(SharedEventBatchTerminalState::Error, producer, 0, None,),
            Ok(None)
        );
        assert_eq!(
            terminal_shared_event_advance(
                SharedEventBatchTerminalState::Completed,
                producer,
                0,
                Some(batch_maximum),
            ),
            Ok(None),
            "successful batches must use their encoded GPU signal"
        );
    }

    #[test]
    fn terminal_error_advance_rejects_a_different_producers_fence() {
        let device = Device::system_default().expect("Metal device");
        let expected = SharedEventProducerId::new(0);
        let actual = SharedEventProducerId::new(41);
        let mut other_timeline = SharedEventTimeline::new(&device, actual);
        let other_fence = other_timeline.allocate_fence().expect("other fence");

        let mismatch = terminal_shared_event_advance(
            SharedEventBatchTerminalState::Error,
            expected,
            0,
            Some(other_fence),
        )
        .expect_err("a queue cannot advance another queue's event");
        assert_eq!(mismatch.expected(), expected);
        assert_eq!(mismatch.actual(), actual);
    }

    #[test]
    fn consumer_committed_before_producer_orders_across_distinct_command_queues() {
        const BYTE_COUNT: usize = 64;
        const PRODUCED_BYTE: u8 = 0xa5;

        let device = Device::system_default().expect("Metal device");
        let byte_count = u64::try_from(BYTE_COUNT).expect("test byte count fits Metal NSUInteger");
        let producer_queue = device.new_command_queue();
        let consumer_queue = device.new_command_queue();
        let mut timeline = SharedEventTimeline::new(&device, SharedEventProducerId::new(41));
        let fence = timeline.allocate_fence().expect("producer fence");
        let source_buffer = device.new_buffer(byte_count, MTLResourceOptions::StorageModeShared);
        let observed = device.new_buffer(byte_count, MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::write_bytes(source_buffer.contents().cast::<u8>(), 0, BYTE_COUNT);
            std::ptr::write_bytes(observed.contents().cast::<u8>(), 0, BYTE_COUNT);
        }

        let consumer = consumer_queue.new_command_buffer().to_owned();
        consumer.encode_wait_for_event(timeline.event(), fence.value());
        let copy = consumer.new_blit_command_encoder();
        copy.copy_from_buffer(&source_buffer, 0, &observed, 0, byte_count);
        copy.end_encoding();

        let producer_commands = producer_queue.new_command_buffer().to_owned();
        let fill = producer_commands.new_blit_command_encoder();
        fill.fill_buffer(&source_buffer, NSRange::new(0, byte_count), PRODUCED_BYTE);
        fill.end_encoding();
        producer_commands.encode_signal_event(timeline.event(), fence.value());

        // This order deadlocks if both logical queues are incorrectly routed through one serial
        // MTLCommandQueue: that queue cannot reach the producer signal behind the consumer wait.
        consumer.commit();
        assert!(
            !wait_until_settled_within(&consumer, Duration::ZERO),
            "consumer unexpectedly completed before its producer was submitted"
        );
        producer_commands.commit();

        let consumer_settled = wait_until_settled_within(&consumer, COMMAND_BUFFER_SETTLE_TIMEOUT);
        if !consumer_settled {
            // Never leave a GPU wait outstanding when this regression test fails.
            timeline.event().set_signaled_value(fence.value());
            let _ = wait_until_settled_within(&consumer, COMMAND_BUFFER_SETTLE_TIMEOUT);
        }
        assert!(consumer_settled, "cross-queue consumer did not settle");
        assert!(wait_until_settled_within(
            &producer_commands,
            COMMAND_BUFFER_SETTLE_TIMEOUT
        ));
        assert_eq!(consumer.status(), MTLCommandBufferStatus::Completed);
        assert_eq!(
            producer_commands.status(),
            MTLCommandBufferStatus::Completed
        );
        let observed =
            unsafe { std::slice::from_raw_parts(observed.contents().cast::<u8>(), BYTE_COUNT) };
        assert!(
            observed.iter().all(|&byte| byte == PRODUCED_BYTE),
            "consumer did not observe producer bytes through the SharedEvent edge"
        );
    }
}
