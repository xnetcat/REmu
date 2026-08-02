// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    ActiveGuestMemoryWait, FlushPolicy, GuestBatchFlushReason, GuestMemoryWaitKey,
    GuestMemoryWaitRecovery,
    GuestMemoryWaitStatus, GuestMemoryWaitTracker, MetalPresenter, OrderedEventSource,
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
        let key = GuestMemoryWaitKey::new(source, wait);
        if let Some(ticket_id) = ticket_id
            && let Some(value) = remu_gpu::take_gpu_wait_latch(ticket_id)
        {
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
                remu_gpu::retire_gpu_wait(ticket_id);
            }
            return Ok(GuestMemoryWaitStatus::Satisfied);
        }
        let required_alignment = if wait.is_64_bit { 8 } else { 4 };
        if !wait.address.is_multiple_of(required_alignment) {
            if let Some(ticket_id) = ticket_id {
                remu_gpu::retire_gpu_wait(ticket_id);
            }
            return Err(format!(
                "{}-bit wait address {:#x} is unaligned",
                required_alignment * 8,
                wait.address
            ));
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
                    let sequence =
                        DEFERRED_FUTURE_VALUE_WAITS.fetch_add(1, Ordering::Relaxed) + 1;
                    if sequence <= 16 || sequence.is_power_of_two() {
                        eprintln!(
                            "[REMu][LIFETIME-ORDER] phase=defer-future-wait sequence={sequence} address={:#x} value={value:#x} open_batch={} committed_batches={}",
                            wait.address,
                            self.open_guest_batch.is_some(),
                            self.committed_guest_batches.len()
                        );
                    }
                }
            }
            None if wait.is_satisfied_by(read_guest_memory_wait_value(wait)) => {
                self.guest_memory_waits.complete(key);
                if let Some(ticket_id) = ticket_id {
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
        if let Err(error) =
            self.flush_guest_command_batch(FlushPolicy::Required, GuestBatchFlushReason::BlockedWait)
        {
            if let Some(ticket_id) = ticket_id {
                remu_gpu::retire_gpu_wait(ticket_id);
            }
            return Err(error);
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
        self.open_guest_batch
            .iter()
            .flat_map(|batch| batch.memory_writes.iter().rev())
            .chain(
                self.committed_guest_batches
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
        }
    }

    pub(crate) const fn wait(self) -> GpuMemoryWait {
        GpuMemoryWait {
            address: self.address,
            reference: self.reference,
            mask: self.mask,
            compare_function: self.compare_function,
            is_64_bit: self.is_64_bit,
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
    // SAFETY: execute_guest_memory_wait validates the non-null address and required alignment
    // before calling this helper. The runtime retains the guest mapping while its ordered GPU
    // events are pending. Volatile access observes label writes from guest CPU threads.
    unsafe {
        if wait.is_64_bit {
            std::ptr::read_volatile(wait.address as *const u64)
        } else {
            u64::from(std::ptr::read_volatile(wait.address as *const u32))
        }
    }
}
