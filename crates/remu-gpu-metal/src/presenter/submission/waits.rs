// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    ActiveGuestMemoryWait, FlushPolicy, GuestBatchFlushReason, GuestMemoryWaitKey,
    GuestMemoryWaitRecovery, GuestMemoryWaitStatus, GuestMemoryWaitTracker, MetalPresenter,
    OrderedEventSource,
};
use remu_gpu::{GpuMemoryWait, GpuWaitId};
use std::sync::atomic::{AtomicUsize, Ordering};

impl MetalPresenter {
    pub(crate) fn execute_guest_memory_wait(
        &mut self,
        source: OrderedEventSource,
        ticket_id: Option<GpuWaitId>,
        wait: GpuMemoryWait,
    ) -> Result<GuestMemoryWaitStatus, String> {
        self.set_active_guest_queue(match source {
            OrderedEventSource::GuestQueue(queue_id) => Some(queue_id),
            OrderedEventSource::Host { .. } => None,
        });
        let key = GuestMemoryWaitKey::new(source, wait);
        if let Some(ticket_id) = ticket_id
            && let Some(value) = remu_gpu::take_gpu_wait_latch(ticket_id)
        {
            let _ = self.gpu_wait_fences.take(ticket_id);
            self.guest_memory_waits.complete(key);
            static RESUMED_TICKET_WAITS: AtomicUsize = AtomicUsize::new(0);
            let sequence = RESUMED_TICKET_WAITS.fetch_add(1, Ordering::Relaxed) + 1;
            if sequence <= 16 || sequence.is_power_of_two() {
                eprintln!(
                    "[REMu][METAL] resumed WAIT_REG_MEM from submitted producer ticket address={:#x} value={value:#x}",
                    wait.address
                );
            }
            return Ok(GuestMemoryWaitStatus::Satisfied);
        }
        if wait.address == 0 {
            self.guest_memory_waits.complete(key);
            if let Some(ticket_id) = ticket_id {
                let _ = self.gpu_wait_fences.take(ticket_id);
                remu_gpu::retire_gpu_wait(ticket_id);
            }
            return Ok(GuestMemoryWaitStatus::Satisfied);
        }
        let required_alignment = if wait.is_64_bit { 8 } else { 4 };
        if !wait.address.is_multiple_of(required_alignment) {
            if let Some(ticket_id) = ticket_id {
                let _ = self.gpu_wait_fences.take(ticket_id);
                remu_gpu::retire_gpu_wait(ticket_id);
            }
            return Err(format!(
                "{}-bit wait address {:#x} is unaligned",
                required_alignment * 8,
                wait.address
            ));
        }
        if self.per_queue_guest_batches
            && let Some(ticket_id) = ticket_id
            && let Some(fence) = self.gpu_wait_fences.get(ticket_id)
        {
            let OrderedEventSource::GuestQueue(consumer_queue_id) = source else {
                let _ = self.gpu_wait_fences.take(ticket_id);
                remu_gpu::retire_gpu_wait(ticket_id);
                return Err("registered guest wait has no logical queue identity".to_owned());
            };
            let producer = self
                .guest_queue_batches
                .get(&fence.producer().queue_id())
                .ok_or_else(|| {
                    format!(
                        "registered wait references missing producer queue {}",
                        fence.producer().queue_id()
                    )
                })?;
            let resource_checkpoint = producer
                .committed_batches
                .iter()
                .chain(producer.open_batch.iter())
                .flat_map(|batch| &batch.signal_checkpoints)
                .find(|checkpoint| checkpoint.fence == fence)
                .cloned()
                .ok_or_else(|| {
                    format!(
                        "registered wait references missing resource checkpoint queue={} fence={}",
                        fence.producer().queue_id(),
                        fence.value()
                    )
                })?;
            // SDK WaitOperation: PFP waits stall the parser (must close open encoders); ME waits
            // leave PFP free to run ahead (sync.cpp waitOnAddress / waitOnLabel).
            let stall_parser = wait
                .wait_operation()
                .is_none_or(|operation| operation.stalls_parser());
            if fence.producer().queue_id() == consumer_queue_id {
                // A3: same-queue WAIT_REG_MEM elision — the default disposition.
                //
                // The O(1) producer index (`gpu_wait_fences`) located the producer's RELEASE_MEM
                // signal checkpoint on THIS very queue, which proves it is encoded ahead of this
                // wait. In the SDK, `gpuSyncPreConsumer` emits a PFP `WAIT_REG_MEM`
                // (`waitOnAddress`, sync.cpp:759) against a label whose producer
                // `gpuSyncPostProducer` `RELEASE_MEM` (`queueEndOfPipeAction`, sync.cpp:749) is
                // queued earlier in the same command stream. A single `MTLCommandQueue` already
                // reproduces that command-processor FIFO order, so a GPU wait for the producer's
                // own event — and the render-pass split `close_active_pass_encoders` would force —
                // is pure overhead. Keep the resource dependency so same-queue DMA fills
                // (`prepare_same_queue_immediate_fill_inner`) and direct draws still consult the
                // producer footprint; skip the event wait and the host dependency (host
                // retirement of one queue is already FIFO-ordered).
                let consumer = self
                    .guest_queue_batches
                    .get_mut(&consumer_queue_id)
                    .ok_or_else(|| format!("missing consumer guest queue {consumer_queue_id}"))?;
                consumer
                    .resource_dependencies
                    .entry(fence.producer())
                    .and_modify(|checkpoint| {
                        if checkpoint.fence.value() < resource_checkpoint.fence.value() {
                            *checkpoint = resource_checkpoint.clone();
                        }
                    })
                    .or_insert(resource_checkpoint);
                let taken = self.gpu_wait_fences.take(ticket_id);
                debug_assert_eq!(taken, Some(fence));
                remu_gpu::retire_gpu_wait(ticket_id);
                self.guest_memory_waits.complete(key);
                self.statistics.same_queue_wait_elisions =
                    self.statistics.same_queue_wait_elisions.saturating_add(1);
                return Ok(GuestMemoryWaitStatus::Satisfied);
            }
            let producer_event = self
                .guest_queue_batches
                .get(&fence.producer().queue_id())
                .expect("producer queue verified above")
                .shared_event_timeline
                .event_owned();
            self.ensure_active_open_guest_batch()?;
            let consumer = self
                .guest_queue_batches
                .get_mut(&consumer_queue_id)
                .ok_or_else(|| format!("missing consumer guest queue {consumer_queue_id}"))?;
            let batch = consumer
                .open_batch
                .as_mut()
                .ok_or("shared-event wait has no open consumer batch")?;
            if stall_parser {
                batch.close_active_pass_encoders();
            }
            batch
                .command_buffer
                .encode_wait_for_event(&producer_event, fence.value());
            consumer
                .host_dependencies
                .entry(fence.producer())
                .and_modify(|value| *value = (*value).max(fence.value()))
                .or_insert(fence.value());
            consumer
                .resource_dependencies
                .entry(fence.producer())
                .and_modify(|checkpoint| {
                    if checkpoint.fence.value() < resource_checkpoint.fence.value() {
                        *checkpoint = resource_checkpoint.clone();
                    }
                })
                .or_insert(resource_checkpoint);
            let taken = self.gpu_wait_fences.take(ticket_id);
            debug_assert_eq!(taken, Some(fence));
            remu_gpu::retire_gpu_wait(ticket_id);
            self.guest_memory_waits.complete(key);
            self.statistics.shared_event_waits =
                self.statistics.shared_event_waits.saturating_add(1);
            return Ok(GuestMemoryWaitStatus::Satisfied);
        }
        // A queued write to this label is the value the GPU will eventually observe, but it is
        // not completion evidence yet. The old fast path returned `Satisfied` from that future
        // value and let the presenter execute following host-side commands immediately. Metal
        // encodes happened to remain ordered on the one serial command queue, but CP DMA copies,
        // cache mutations, and other CPU-emulated effects could overtake the producer. A later
        // submission boundary would still join the producer batch, hiding the lie from the
        // completion counters while those early effects touched allocations the guest considered
        // fenced.
        //
        // Park only this logical guest queue until the producer batch reaches a terminal state
        // and publishes the real label. Other queues remain runnable and no host thread waits.
        // The producer publication also latches registered waits, so the normal ticket path above
        // resumes without polling a recycled label generation.
        let pending_value = self.pending_guest_memory_value(wait.address, wait.is_64_bit);
        let pending_satisfying_producer =
            pending_value.is_some_and(|value| wait.is_satisfied_by(value));
        match pending_value {
            Some(value) => {
                if wait.is_satisfied_by(value) {
                    static DEFERRED_FUTURE_VALUE_WAITS: AtomicUsize = AtomicUsize::new(0);
                    let sequence = DEFERRED_FUTURE_VALUE_WAITS.fetch_add(1, Ordering::Relaxed) + 1;
                    if sequence <= 16 || sequence.is_power_of_two() {
                        eprintln!(
                            "[REMu][LIFETIME-ORDER] phase=defer-future-wait sequence={sequence} address={:#x} value={value:#x} open_batch={} committed_batches={}",
                            wait.address,
                            self.active_open_guest_batch().is_some(),
                            self.active_committed_guest_batches().len()
                        );
                    }
                }
            }
            None if wait.is_satisfied_by(read_guest_memory_wait_value(wait)) => {
                self.guest_memory_waits.complete(key);
                if let Some(ticket_id) = ticket_id {
                    let _ = self.gpu_wait_fences.take(ticket_id);
                    remu_gpu::retire_gpu_wait(ticket_id);
                }
                return Ok(GuestMemoryWaitStatus::Satisfied);
            }
            None => {}
        }
        self.retire_completed_guest_batches();
        match self.pending_guest_memory_value(wait.address, wait.is_64_bit) {
            Some(_) => {}
            None if wait.is_satisfied_by(read_guest_memory_wait_value(wait)) => {
                self.guest_memory_waits.complete(key);
                if let Some(ticket_id) = ticket_id {
                    let _ = self.gpu_wait_fences.take(ticket_id);
                    remu_gpu::retire_gpu_wait(ticket_id);
                }
                return Ok(GuestMemoryWaitStatus::Satisfied);
            }
            None => {}
        }

        // Guest graphics and async-compute queues can wait on labels released by one another.
        // Closing this queue's current Metal batch preserves its work before the wait. The queue
        // scheduler retains this packet at the submission front and runs another guest queue;
        // later commands from this queue are never allowed to overtake the unsatisfied condition.
        if let Err(error) = self
            .flush_guest_command_batch(FlushPolicy::Required, GuestBatchFlushReason::BlockedWait)
        {
            if let Some(ticket_id) = ticket_id {
                let _ = self.gpu_wait_fences.take(ticket_id);
                remu_gpu::retire_gpu_wait(ticket_id);
            }
            return Err(error);
        }
        if self.per_queue_guest_batches
            && let Some(ticket_id) = ticket_id
            && self.gpu_wait_fences.note_missing_once(ticket_id)
        {
            self.statistics.shared_event_fallbacks_no_exact_producer = self
                .statistics
                .shared_event_fallbacks_no_exact_producer
                .saturating_add(1);
        }
        if ticket_id.is_none() {
            match self.guest_memory_waits.observe_unsatisfied(key) {
                GuestMemoryWaitRecovery::ProducerLatched(value) => {
                    static RESUMED_PRODUCER_WAITS: AtomicUsize = AtomicUsize::new(0);
                    let sequence = RESUMED_PRODUCER_WAITS.fetch_add(1, Ordering::Relaxed) + 1;
                    if sequence <= 16 || sequence.is_power_of_two() {
                        eprintln!(
                            "[REMu][METAL] resumed WAIT_REG_MEM from a real producer latch address={:#x} value={value:#x}",
                            wait.address
                        );
                    }
                    return Ok(GuestMemoryWaitStatus::Satisfied);
                }
                GuestMemoryWaitRecovery::None => {}
            }
        }
        static SUSPENDED_WAITS: AtomicUsize = AtomicUsize::new(0);
        let sequence = SUSPENDED_WAITS.fetch_add(1, Ordering::Relaxed) + 1;
        if sequence <= 16 || sequence.is_power_of_two() {
            let value = read_guest_memory_wait_value(wait);
            let pending_value = self.pending_guest_memory_value(wait.address, wait.is_64_bit);
            let queued_producers = self.receiver.describe_queued_wait_producers(wait);
            eprintln!(
                "[REMu][METAL] suspended WAIT_REG_MEM sequence={sequence} address={:#x} value={value:#x} pending={pending_value:?} mask={:#x} reference={:#x} compare={} producers=[{queued_producers}]",
                wait.address, wait.mask, wait.reference, wait.compare_function
            );
        }
        if pending_satisfying_producer
            && self
                .pending_guest_memory_value(wait.address, wait.is_64_bit)
                .is_some_and(|value| wait.is_satisfied_by(value))
        {
            Ok(GuestMemoryWaitStatus::BlockedOnPendingTerminal)
        } else {
            Ok(GuestMemoryWaitStatus::Blocked)
        }
    }

    pub(crate) fn record_submitted_gpu_waits_in_range(&mut self, address: u64, byte_count: u64) {
        let waits = remu_gpu::pending_gpu_waits_overlapping(address, byte_count);
        for (_, wait) in waits {
            let value = read_guest_memory_wait_value(wait);
            self.record_guest_memory_producer(wait.address, wait.is_64_bit, value);
        }
    }

    pub(crate) fn pending_guest_memory_value(&self, address: u64, is_64_bit: bool) -> Option<u64> {
        self.active_open_guest_batch()
            .iter()
            .flat_map(|batch| batch.memory_writes.iter().rev())
            .chain(
                self.active_committed_guest_batches()
                    .iter()
                    .rev()
                    .flat_map(|batch| batch.memory_writes.iter().rev()),
            )
            // The open batch is newer than every committed batch; within each batch, the last
            // write wins. This mirrors guest command-stream order without publishing the label
            // before its preceding Metal work completes.
            .find_map(|write| write.value_at(address, is_64_bit))
    }
}

impl GuestMemoryWaitKey {
    pub(crate) const fn new(source: OrderedEventSource, wait: GpuMemoryWait) -> Self {
        Self {
            source,
            address: wait.address,
            reference: wait.reference,
            mask: wait.mask,
            compare_function: wait.compare_function,
            is_64_bit: wait.is_64_bit,
            operation: wait.operation,
            cache_policy: wait.cache_policy,
        }
    }

    pub(crate) const fn wait(self) -> GpuMemoryWait {
        GpuMemoryWait {
            address: self.address,
            reference: self.reference,
            mask: self.mask,
            compare_function: self.compare_function,
            is_64_bit: self.is_64_bit,
            operation: self.operation,
            cache_policy: self.cache_policy,
        }
    }
}

impl GuestMemoryWaitTracker {
    pub(crate) fn observe_unsatisfied(
        &mut self,
        key: GuestMemoryWaitKey,
    ) -> GuestMemoryWaitRecovery {
        let active = self.active.entry(key).or_insert(ActiveGuestMemoryWait {
            latched_value: None,
        });
        if let Some(value) = active.latched_value {
            self.active.remove(&key);
            return GuestMemoryWaitRecovery::ProducerLatched(value);
        }
        GuestMemoryWaitRecovery::None
    }

    pub(crate) fn complete(&mut self, key: GuestMemoryWaitKey) {
        self.active.remove(&key);
    }

    pub(crate) fn record_produced(&mut self, address: u64, is_64_bit: bool, value: u64) {
        for (key, active) in &mut self.active {
            if key.address == address
                && key.is_64_bit == is_64_bit
                && key.wait().is_satisfied_by(value)
            {
                active.latched_value = Some(value);
            }
        }
    }
}

pub(crate) fn read_guest_memory_wait_value(wait: GpuMemoryWait) -> u64 {
    // execute_guest_memory_wait validates the non-null address and required alignment before
    // calling this helper. Acquire loads pair with release label publication so a satisfied wait
    // observes the payload that the label was ordered after.
    if wait.is_64_bit {
        remu_memory::load_guest_visible_u64(wait.address)
    } else {
        u64::from(remu_memory::load_guest_visible_u32(wait.address))
    }
}
