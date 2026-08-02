// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::completion::{
    COMMAND_BUFFER_SETTLE_TIMEOUT, command_buffer_failure_message, settle_timeout_message,
    wait_until_settled_within,
};
use super::super::FlushPolicy;
use crate::presenter::diagnostics::{journal_guest_write, trace_diagnostic_guest_write};
use super::{
    GLOBAL_UPLOAD_ALIGNMENT, GLOBAL_UPLOAD_CHUNK_BYTE_COUNT, GlobalUploadChunk,
    GuestBatchFlushReason, GuestBatchWork, GuestCommandBatch, MetalPresenter,
    PendingGlobalWriteback, PendingGuestWriteRange, UploadedGlobalBuffer,
    guest_physical_ranges_overlap, guest_ranges_overlap,
};
use metal::foreign_types::ForeignType;
use metal::{Buffer, CommandBuffer, ComputeCommandEncoder, Device, MTLCommandBufferStatus, MTLResourceOptions};
use objc::{msg_send, sel, sel_impl};
use remu_memory::{
    CpuWriteWatch, GuestSharedMappingIdentity, GuestWriteSite, guest_shared_backing_identity,
    read_guest_shared_backing, write_guest_shared_backing_at_identity,
    write_guest_shared_backing_if_identity,
};
use remu_shader::ShaderEvaluation;
use std::io::ErrorKind;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

static DISCARDED_STALE_GLOBAL_WRITEBACKS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, PartialEq, Eq)]
enum GuestBatchSettlement {
    Completed,
    Failed(String),
    Pending(String),
}

/// Outcome of settling the oldest committed batch. `Retired` carries that batch's own completion
/// result; `Pending` means it is still queued and no slot was reclaimed.
enum FrontBatchSettlement {
    Empty,
    Retired(Result<(), String>),
    Pending(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GlobalWritebackOutcome {
    Applied,
    DiscardedStale,
}

impl MetalPresenter {
    pub(crate) fn ensure_open_guest_batch(&mut self, queue: &metal::CommandQueue) {
        if self.open_guest_batch.is_some() {
            return;
        }
        let command_buffer = queue.new_command_buffer().to_owned();
        command_buffer.set_label("REMu translated guest batch");
        self.open_guest_batch = Some(GuestCommandBatch {
            command_buffer,
            work: Vec::new(),
            committed_at: None,
            global_upload_chunks: Vec::new(),
            global_writebacks: Vec::new(),
            guest_write_ranges: Vec::new(),
            memory_writes: Vec::new(),
            render_pass: None,
            compute_encoder: None,
            completion_sequences: Vec::new(),
        });
    }

    pub(crate) fn retire_completed_guest_batches(&mut self) {
        loop {
            let Some(batch) = self.committed_guest_batches.front() else {
                return;
            };
            match batch.command_buffer.status() {
                MTLCommandBufferStatus::Completed => {
                    let batch = self
                        .committed_guest_batches
                        .pop_front()
                        .expect("front batch exists");
                    let result = self.finish_terminal_guest_batch(&batch, None);
                    self.statistics.gpu_execution_time +=
                        completed_command_buffer_gpu_duration(&batch.command_buffer);
                    publish_guest_batch_completion(&batch);
                    if let Err(error) = result {
                        eprintln!(
                            "[REMu][METAL] completed guest batch finalization failed: {error}"
                        );
                    }
                }
                MTLCommandBufferStatus::Error => {
                    let batch = self
                        .committed_guest_batches
                        .pop_front()
                        .expect("front batch exists");
                    let execution_error = command_buffer_failure_message(
                        &format!(
                            "Metal guest command batch [{}]",
                            describe_guest_batch(&batch.work)
                        ),
                        &batch.command_buffer,
                    );
                    let result = self.finish_terminal_guest_batch(&batch, Some(execution_error));
                    self.statistics.gpu_execution_time +=
                        completed_command_buffer_gpu_duration(&batch.command_buffer);
                    // A terminal native failure cannot be retried in-place. Publish both its
                    // ordered CPU labels and submission ID after reporting the error so the guest
                    // observes one failed queue item instead of a producerless 120-second wait.
                    publish_guest_batch_completion(&batch);
                    if let Err(error) = result {
                        eprintln!("[REMu][METAL] {error}");
                    }
                }
                _ => return,
            }
        }
    }

    pub(crate) fn finish_terminal_guest_batch(
        &mut self,
        batch: &GuestCommandBatch,
        execution_error: Option<String>,
    ) -> Result<(), String> {
        let execution_succeeded = execution_error.is_none();
        let mut result = execution_error.map_or(Ok(()), Err);
        if execution_succeeded {
            let writeback = write_back_pending_globals(&batch.global_writebacks);
            if let Ok(applied) = &writeback {
                self.record_produced_global_writebacks(applied);
            }
            result = merge_guest_batch_result(result, "global writeback", writeback.map(|_| ()));
        }
        // RELEASE_MEM, WRITE_DATA, and DMA fills are command-processor effects ordered after the
        // native work, not shader results. Once the Metal command buffer reaches either terminal
        // state, publish them even when the draw/dispatch or an unrelated global readback failed.
        // This matches the existing policy of retiring the submission after a native error while
        // retaining the error as the primary result.
        self.publish_terminal_guest_memory_writes(&batch.memory_writes, result)
    }

    pub(super) fn gpu_events_per_tick_budget(&self) -> usize {
        if remu_gpu::gpu_work_items_pending() > super::super::GPU_EVENTS_PER_TICK_BACKLOG_THRESHOLD
        {
            super::super::GPU_EVENTS_PER_TICK
                .saturating_mul(super::super::GPU_EVENTS_PER_TICK_BACKLOG_MULTIPLIER)
        } else {
            super::super::GPU_EVENTS_PER_TICK
        }
    }

    pub(super) fn guest_submission_backpressure_full(&self) -> bool {
        self.committed_guest_batches.len() >= super::super::MAX_COMMITTED_GUEST_BATCHES
            && self
                .open_guest_batch
                .as_ref()
                .is_none_or(|batch| batch.work.len() >= super::super::GUEST_BATCH_WORK_CAPACITY)
    }

    pub(crate) fn flush_guest_command_batch(
        &mut self,
        policy: FlushPolicy,
        reason: GuestBatchFlushReason,
    ) -> Result<(), String> {
        self.flush_guest_command_batch_by(
            policy,
            reason,
            Instant::now() + COMMAND_BUFFER_SETTLE_TIMEOUT,
        )
    }

    /// Flushes the open batch, bounding any budget-reclaiming settle by `settle_deadline`.
    fn flush_guest_command_batch_by(
        &mut self,
        policy: FlushPolicy,
        reason: GuestBatchFlushReason,
        settle_deadline: Instant,
    ) -> Result<(), String> {
        if self.open_guest_batch.is_none() {
            return Ok(());
        }
        // A batch that outlives the settle budget stays queued, so the in-flight count can still be
        // at the cap below. Committing one batch past it is bounded and temporary; declining to
        // commit would hand a `Required` caller the uncommitted results it is about to read.
        let mut settlement = Ok(());
        if self.committed_guest_batches.len() >= super::super::MAX_COMMITTED_GUEST_BATCHES {
            match policy {
                FlushPolicy::Opportunistic => return Ok(()),
                FlushPolicy::Required => {
                    settlement = match self.settle_front_guest_batch(settle_deadline) {
                        FrontBatchSettlement::Empty => Ok(()),
                        FrontBatchSettlement::Retired(result) => result,
                        FrontBatchSettlement::Pending(error) => Err(error),
                    };
                }
            }
        }
        let Some(mut batch) = self.open_guest_batch.take() else {
            return Ok(());
        };
        match reason {
            GuestBatchFlushReason::EventLoop => {
                self.statistics.batch_flushes_from_event_loop = self
                    .statistics
                    .batch_flushes_from_event_loop
                    .saturating_add(1);
            }
            GuestBatchFlushReason::BlockedWait => {
                self.statistics.batch_flushes_for_blocked_wait = self
                    .statistics
                    .batch_flushes_for_blocked_wait
                    .saturating_add(1);
            }
            GuestBatchFlushReason::FlipSnapshot => {
                self.statistics.batch_flushes_for_flip_snapshot = self
                    .statistics
                    .batch_flushes_for_flip_snapshot
                    .saturating_add(1);
            }
            GuestBatchFlushReason::WorkCapacity => {
                self.statistics.batch_flushes_for_work_capacity = self
                    .statistics
                    .batch_flushes_for_work_capacity
                    .saturating_add(1);
            }
            GuestBatchFlushReason::CommandProcessorBarrier => {
                self.statistics.batch_flushes_for_command_processor_barrier = self
                    .statistics
                    .batch_flushes_for_command_processor_barrier
                    .saturating_add(1);
            }
            GuestBatchFlushReason::CpuSynchronization => {
                self.statistics.batch_flushes_for_cpu_synchronization = self
                    .statistics
                    .batch_flushes_for_cpu_synchronization
                    .saturating_add(1);
            }
        }
        batch.end_compute_pass();
        if batch.end_render_pass() {
            self.statistics.render_pass_ends_for_batch_flush = self
                .statistics
                .render_pass_ends_for_batch_flush
                .saturating_add(1);
        }
        // Writable and oversized bindings are shared only within this ordered batch. Rotating
        // them here keeps completed writebacks immutable while reducing per-draw Metal allocation
        // to at most one buffer per unchanged guest address range per batch.
        self.transient_global_buffer_cache.clear();
        batch.committed_at = Some(Instant::now());
        batch.command_buffer.commit();
        self.statistics.submitted_guest_batches =
            self.statistics.submitted_guest_batches.saturating_add(1);
        self.committed_guest_batches.push_back(batch);
        settlement
    }

    /// Settles the oldest committed batch and publishes its completion effects.
    ///
    /// A batch that outlives `deadline` is put back at the front: later buffers from the same queue
    /// cannot complete before it, so none of their CPU-side completion effects may overtake it.
    fn settle_front_guest_batch(&mut self, deadline: Instant) -> FrontBatchSettlement {
        let Some(batch) = self.committed_guest_batches.pop_front() else {
            return FrontBatchSettlement::Empty;
        };
        let execution_error = match settle_guest_batch_by(&batch, deadline) {
            GuestBatchSettlement::Completed => None,
            GuestBatchSettlement::Failed(error) => Some(error),
            GuestBatchSettlement::Pending(error) => {
                self.committed_guest_batches.push_front(batch);
                return FrontBatchSettlement::Pending(error);
            }
        };
        let result = self.finish_terminal_guest_batch(&batch, execution_error);
        self.statistics.gpu_execution_time +=
            completed_command_buffer_gpu_duration(&batch.command_buffer);
        publish_guest_batch_completion(&batch);
        FrontBatchSettlement::Retired(result)
    }

    pub(crate) fn wait_for_guest_command_batches(&mut self) -> Result<(), String> {
        self.wait_for_guest_command_batches_within(COMMAND_BUFFER_SETTLE_TIMEOUT)
    }

    pub(crate) fn wait_for_guest_command_batches_within(
        &mut self,
        timeout: Duration,
    ) -> Result<(), String> {
        // One budget covers the whole drain, including the flush's own slot reclaim. Charging every
        // queued batch its own timeout would sum to far past the host watchdog window this bound
        // exists to stay inside, and a device that stalled on the first batch will not retire the
        // rest either.
        let deadline = Instant::now() + timeout;
        self.flush_guest_command_batch_by(
            FlushPolicy::Required,
            GuestBatchFlushReason::CpuSynchronization,
            deadline,
        )?;
        debug_assert!(self.open_guest_batch.is_none());
        let mut result = Ok(());
        loop {
            match self.settle_front_guest_batch(deadline) {
                FrontBatchSettlement::Empty => return result,
                // Drain every already-committed batch even after one fails. Returning early here
                // used to abandon later RELEASE_MEM labels and completion IDs during shutdown or a
                // synchronous readback, converting a preserved Metal error into an unrelated guest
                // RenderThread timeout.
                FrontBatchSettlement::Retired(batch_result) => {
                    result =
                        merge_guest_batch_result(result, "later ordered guest batch", batch_result);
                }
                FrontBatchSettlement::Pending(error) => {
                    return merge_guest_batch_result(
                        result,
                        "pending ordered guest batch",
                        Err(error),
                    );
                }
            }
        }
    }
}

/// Waits for one committed guest batch until `deadline` and describes why it failed, when it did.
///
/// A batch that outlives the budget is worded differently from a driver-reported failure: its work
/// is still outstanding, so callers must retain it rather than treat it as having run.
fn settle_guest_batch_by(batch: &GuestCommandBatch, deadline: Instant) -> GuestBatchSettlement {
    let budget = deadline.saturating_duration_since(Instant::now());
    let settled = wait_until_settled_within(&batch.command_buffer, budget);
    let operation = format!(
        "Metal guest command batch [{}]",
        describe_guest_batch(&batch.work)
    );
    if !settled {
        return GuestBatchSettlement::Pending(settle_timeout_message(
            &operation,
            &batch.command_buffer,
        ));
    }
    let status = batch.command_buffer.status();
    if status == MTLCommandBufferStatus::Completed {
        GuestBatchSettlement::Completed
    } else {
        GuestBatchSettlement::Failed(command_buffer_failure_message(
            &operation,
            &batch.command_buffer,
        ))
    }
}

pub(crate) fn describe_guest_batch(work: &[GuestBatchWork]) -> String {
    work.iter()
        .map(|item| match item {
            GuestBatchWork::Draw {
                sequence,
                indexed,
                indirect_arguments_address,
                export,
                pixel,
                index_count,
                instance_count,
                target,
                extent,
                export_msl_bytes,
                pixel_msl_bytes,
                export_dispatcher,
                pixel_dispatcher,
            } => {
                let source = indirect_arguments_address.map_or_else(
                    || "direct".to_owned(),
                    |address| format!("indirect@{address:#x}"),
                );
                format!(
                    "draw(seq={sequence},indexed={indexed},source={source},es={export:#x}/{export_msl_bytes}B/dispatcher={export_dispatcher},ps={pixel:#x}/{pixel_msl_bytes}B/dispatcher={pixel_dispatcher},indices={index_count},instances={instance_count},target={target:#x}/{extent:?})"
                )
            }
            GuestBatchWork::Compute {
                sequence,
                shader,
                source,
                grid,
                threads_per_group,
                msl_bytes,
                dispatcher,
            } => format!(
                "compute(seq={sequence},cs={shader:#x}/{msl_bytes}B/dispatcher={dispatcher},source={source:?},grid={grid:?},local={threads_per_group:?})"
            ),
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::{GuestBatchWork, describe_guest_batch};

    #[test]
    fn batch_diagnostics_retain_indirect_geometry_and_dispatcher_cost() {
        let description = describe_guest_batch(&[
            GuestBatchWork::Compute {
                sequence: 17,
                shader: 0x8008_2200_00,
                source: remu_gpu::ComputeDispatchSource::Indirect {
                    dimensions_address: 0x807c_4b00_00,
                },
                grid: remu_gpu::ComputeGrid::Threadgroups([65_535, 4, 1]),
                threads_per_group: [8, 8, 1],
                msl_bytes: 108_553,
                dispatcher: true,
            },
            GuestBatchWork::Draw {
                sequence: 23,
                indexed: true,
                indirect_arguments_address: Some(0x807d_3600_28),
                export: 0x80dc_3b00_00,
                pixel: 0x80dc_4000_00,
                index_count: 36,
                instance_count: 1,
                target: 0x8020_0000_00,
                extent: [1_920, 1_080],
                export_msl_bytes: 83_048,
                pixel_msl_bytes: 447_366,
                export_dispatcher: false,
                pixel_dispatcher: true,
            },
        ]);

        assert!(description.contains(
            "source=Indirect { dimensions_address: 551841103872 },grid=Threadgroups([65535, 4, 1]),local=[8, 8, 1]"
        ));
        assert!(description.contains("cs=0x8008220000/108553B/dispatcher=true"));
        assert!(description.contains("draw(seq=23,indexed=true,source=indirect@0x807d360028"));
        assert!(description.contains(
            "ps=0x80dc400000/447366B/dispatcher=true,indices=36,instances=1,target=0x8020000000/[1920, 1080]"
        ));
    }
}

pub(crate) fn merge_guest_batch_result(
    current: Result<(), String>,
    operation: &str,
    next: Result<(), String>,
) -> Result<(), String> {
    match (current, next) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(format!("{operation}: {error}")),
        (Err(primary), Err(secondary)) => {
            Err(format!("{primary}; {operation} also failed: {secondary}"))
        }
    }
}

impl GuestCommandBatch {
    fn reserve_global_upload(
        &mut self,
        device: &Device,
        byte_count: usize,
    ) -> Result<(Buffer, usize), String> {
        let byte_count = byte_count.max(1);
        let aligned_byte_count = byte_count
            .checked_add(GLOBAL_UPLOAD_ALIGNMENT - 1)
            .map(|value| value & !(GLOBAL_UPLOAD_ALIGNMENT - 1))
            .ok_or("global upload size overflows")?;
        let needs_chunk = self.global_upload_chunks.last().is_none_or(|chunk| {
            chunk
                .used
                .checked_add(aligned_byte_count)
                .is_none_or(|end| end > chunk.buffer.length() as usize)
        });
        if needs_chunk {
            let chunk_byte_count = GLOBAL_UPLOAD_CHUNK_BYTE_COUNT.max(aligned_byte_count);
            let length = u64::try_from(chunk_byte_count)
                .map_err(|_| "global upload chunk size does not fit Metal")?;
            let buffer = device.new_buffer(length, MTLResourceOptions::StorageModeShared);
            buffer.set_label("REMu guest-batch global upload arena");
            self.global_upload_chunks
                .push(GlobalUploadChunk { buffer, used: 0 });
        }
        let chunk = self
            .global_upload_chunks
            .last_mut()
            .expect("global upload chunk was created");
        let offset = chunk.used;
        chunk.used = offset + aligned_byte_count;
        Ok((chunk.buffer.clone(), offset))
    }

    pub(crate) fn upload_global_bytes(
        &mut self,
        device: &Device,
        bytes: &[u8],
    ) -> Result<(Buffer, u64), String> {
        let (buffer, offset) = self.reserve_global_upload(device, bytes.len())?;
        if !bytes.is_empty() {
            // SAFETY: the shared Metal buffer is at least `aligned_byte_count` bytes beyond
            // `offset`, and the source slice remains live for the exact non-overlapping copy.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    buffer.contents().cast::<u8>().add(offset),
                    bytes.len(),
                );
            }
        }
        Ok((
            buffer,
            u64::try_from(offset).map_err(|_| "global upload offset does not fit Metal")?,
        ))
    }

    /// Copies one live guest-physical range directly into this batch's shared upload arena.
    ///
    /// The fallback global-buffer path previously copied the same bytes twice: backing mmap to a
    /// temporary `Vec`, then that `Vec` into Metal. Reading into the final arena keeps the same
    /// validated backing semantics while removing the second full-range copy and allocation.
    pub(crate) fn upload_guest_global_bytes(
        &mut self,
        device: &Device,
        guest_address: u64,
        byte_count: usize,
    ) -> Result<(Buffer, u64), String> {
        if byte_count == 0 {
            return Err("guest global upload is empty".to_owned());
        }
        let (buffer, offset) = self.reserve_global_upload(device, byte_count)?;
        // SAFETY: `reserve_global_upload` retained a shared Metal buffer with at least
        // `byte_count` bytes at `offset`. The temporary slice is used only for this synchronous
        // backing copy and cannot outlive or alias a second mutable arena reservation.
        let destination = unsafe {
            std::slice::from_raw_parts_mut(buffer.contents().cast::<u8>().add(offset), byte_count)
        };
        if !read_guest_shared_backing(guest_address, destination) {
            return Err(format!(
                "synchronized global binding {guest_address:#x}/{byte_count} has no readable guest backing"
            ));
        }
        Ok((
            buffer,
            u64::try_from(offset).map_err(|_| "global upload offset does not fit Metal")?,
        ))
    }

    pub(crate) fn end_render_pass(&mut self) -> bool {
        if let Some(pass) = self.render_pass.take() {
            pass.encoder.end_encoding();
            true
        } else {
            false
        }
    }

    pub(crate) fn close_active_pass_encoders(&mut self) {
        self.end_compute_pass();
        self.end_render_pass();
    }

    pub(crate) fn end_compute_pass(&mut self) -> bool {
        if let Some(encoder) = self.compute_encoder.take() {
            encoder.end_encoding();
            true
        } else {
            false
        }
    }

    pub(crate) fn ensure_compute_encoder(&mut self) -> &ComputeCommandEncoder {
        if self.compute_encoder.is_none() {
            self.compute_encoder = Some(
                self.command_buffer
                    .new_compute_command_encoder()
                    .to_owned(),
            );
        }
        self.compute_encoder
            .as_ref()
            .expect("compute encoder was just opened")
    }
}

pub(crate) fn guest_batch_writes_overlap(
    batch: &GuestCommandBatch,
    address: u64,
    byte_count: u64,
) -> bool {
    batch
        .guest_write_ranges
        .iter()
        .any(|range| pending_guest_write_range_overlaps(range, address, byte_count))
        || batch
            .global_writebacks
            .iter()
            .any(|writeback| pending_global_writeback_overlaps(writeback, address, byte_count))
        || batch.memory_writes.iter().any(|write| {
            let (write_address, write_bytes) = write.range();
            guest_physical_ranges_overlap(write_address, write_bytes, address, byte_count)
        })
}

fn pending_global_writeback_overlaps(
    writeback: &PendingGlobalWriteback,
    address: u64,
    byte_count: u64,
) -> bool {
    let Ok(writeback_byte_count) = u64::try_from(writeback.byte_count) else {
        return true;
    };
    if guest_ranges_overlap(
        writeback.guest_address,
        writeback_byte_count,
        address,
        byte_count,
    ) {
        return true;
    }
    let Some(producer_identity) = writeback.mapping_identity else {
        return guest_physical_ranges_overlap(
            writeback.guest_address,
            writeback_byte_count,
            address,
            byte_count,
        );
    };
    let Some(read_identity) = guest_shared_backing_identity(address, byte_count) else {
        return false;
    };
    producer_identity.backing_id == read_identity.backing_id
        && guest_ranges_overlap(
            producer_identity.backing_offset,
            writeback_byte_count,
            read_identity.backing_offset,
            byte_count,
        )
}

fn pending_guest_write_range_overlaps(
    range: &PendingGuestWriteRange,
    address: u64,
    byte_count: u64,
) -> bool {
    if guest_ranges_overlap(range.address, range.byte_count, address, byte_count) {
        return true;
    }
    let Some(read_identity) = guest_shared_backing_identity(address, byte_count) else {
        return false;
    };
    if range.physical_identity.backing_id != read_identity.backing_id {
        return false;
    }
    guest_ranges_overlap(
        range.physical_identity.backing_offset,
        range.byte_count,
        read_identity.backing_offset,
        byte_count,
    )
}

pub(crate) fn publish_guest_batch_completion(batch: &GuestCommandBatch) {
    for &sequence in &batch.completion_sequences {
        remu_gpu::record_gpu_submission_completed(sequence);
    }
}

pub(crate) fn completed_command_buffer_gpu_duration(command_buffer: &CommandBuffer) -> Duration {
    // Metal exposes GPUStartTime/GPUEndTime only after a command buffer completes. The metal-rs
    // version used here does not wrap them yet, so query the public MTLCommandBuffer selectors
    // directly. Unlike wall time from commit to retirement, this excludes time waiting behind
    // older command buffers and is therefore suitable for the title bar's GPU utilization.
    let (start, end): (f64, f64) = unsafe {
        (
            msg_send![command_buffer.as_ref(), GPUStartTime],
            msg_send![command_buffer.as_ref(), GPUEndTime],
        )
    };
    if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
        return Duration::ZERO;
    }
    Duration::from_secs_f64(end - start)
}

pub(crate) fn collect_pending_global_writebacks(
    buffers: &[UploadedGlobalBuffer],
    evaluation: &ShaderEvaluation,
    pending: &mut Vec<PendingGlobalWriteback>,
) -> Result<(), String> {
    if buffers.len() != evaluation.global_memory_bindings.len() {
        return Err("Metal/global buffer binding count mismatch".to_owned());
    }
    for (buffer, binding) in buffers
        .iter()
        .zip(&evaluation.global_memory_bindings)
        .filter(|(buffer, binding)| {
            !buffer.guest_memory && binding.writable && binding.write_back_to_guest
        })
    {
        if binding.base_address < 0x10_000 || binding.byte_count > 128 * 1024 * 1024 {
            return Err(format!(
                "invalid writable global buffer address={:#x} bytes={}",
                binding.base_address, binding.byte_count
            ));
        }
        if pending.iter().any(|writeback| {
            writeback.buffer.as_ptr() == buffer.buffer.as_ptr()
                && writeback.offset == buffer.offset
                && writeback.guest_address == binding.base_address
                && writeback.byte_count == binding.byte_count
        }) {
            continue;
        }
        let byte_count = u64::try_from(binding.byte_count)
            .map_err(|_| "writable global buffer byte count does not fit u64")?;
        let mapping_identity = buffer
            .physical_identity
            .or_else(|| guest_shared_backing_identity(binding.base_address, byte_count));
        let write_watch = if mapping_identity.is_some() {
            None
        } else {
            // SAFETY: shader evaluation accepted this exact range as a live readable/writable
            // guest binding. Guest mappings are 16 KiB aligned, so its outward-aligned host pages
            // cannot straddle an unrelated reservation. Address-space replacement and protection
            // retire this watch before mutating those pages.
            Some(unsafe { CpuWriteWatch::new(binding.base_address, byte_count) }.map_err(
                |error| {
                    format!(
                        "could not pin copied writable global buffer address={:#x} bytes={}: {error}",
                        binding.base_address, binding.byte_count
                    )
                },
            )?)
        };
        let offset = usize::try_from(buffer.offset)
            .map_err(|_| "writable global buffer offset does not fit usize")?;
        let end = offset
            .checked_add(binding.byte_count)
            .ok_or("writable global buffer range overflow")?;
        if u64::try_from(end).map_or(true, |end| end > buffer.buffer.length()) {
            return Err(format!(
                "writable global buffer range exceeds Metal buffer offset={} bytes={} length={}",
                buffer.offset,
                binding.byte_count,
                buffer.buffer.length()
            ));
        }
        // This shared upload still contains the packet-time input because the owning command
        // buffer has not been committed. Preserve it so completion can publish only bytes that
        // the GPU actually changed, instead of restoring an entire stale CPU snapshot.
        let original_bytes = unsafe {
            std::slice::from_raw_parts(
                buffer.buffer.contents().cast::<u8>().add(offset),
                binding.byte_count,
            )
        }
        .to_vec();
        pending.push(PendingGlobalWriteback {
            buffer: buffer.buffer.clone(),
            offset: buffer.offset,
            guest_address: binding.base_address,
            byte_count: binding.byte_count,
            mapping_identity,
            write_watch,
            original_bytes,
        });
    }
    Ok(())
}

pub(crate) fn collect_pending_guest_write_ranges(
    buffers: &[UploadedGlobalBuffer],
    evaluation: &ShaderEvaluation,
    pending: &mut Vec<PendingGuestWriteRange>,
) -> Result<(), String> {
    if buffers.len() != evaluation.global_memory_bindings.len() {
        return Err("Metal/global buffer binding count mismatch".to_owned());
    }
    for (buffer, binding) in buffers
        .iter()
        .zip(&evaluation.global_memory_bindings)
        .filter(|(buffer, binding)| binding.writable && buffer.guest_memory)
    {
        let byte_count = u64::try_from(binding.byte_count)
            .map_err(|_| "writable guest buffer byte count does not fit u64")?;
        let physical_identity = buffer.physical_identity.ok_or_else(|| {
                format!(
                    "zero-copy writable guest buffer has no captured physical identity address={:#x} bytes={byte_count}",
                    binding.base_address
                )
            })?;
        let range = PendingGuestWriteRange {
            address: binding.base_address,
            byte_count,
            buffer: buffer.buffer.clone(),
            offset: buffer.offset,
            physical_identity,
        };
        append_or_replace_guest_write_range(pending, range);
    }
    Ok(())
}

pub(crate) fn append_unique_global_writebacks(
    destination: &mut Vec<PendingGlobalWriteback>,
    source: Vec<PendingGlobalWriteback>,
) {
    for writeback in source {
        let duplicate = destination.iter().any(|existing| {
            existing.buffer.as_ptr() == writeback.buffer.as_ptr()
                && existing.offset == writeback.offset
                && existing.guest_address == writeback.guest_address
                && existing.byte_count == writeback.byte_count
        });
        if !duplicate {
            destination.push(writeback);
        }
    }
}

pub(crate) fn append_unique_guest_write_ranges(
    destination: &mut Vec<PendingGuestWriteRange>,
    source: Vec<PendingGuestWriteRange>,
) {
    for range in source {
        append_or_replace_guest_write_range(destination, range);
    }
}

fn append_or_replace_guest_write_range(
    destination: &mut Vec<PendingGuestWriteRange>,
    range: PendingGuestWriteRange,
) {
    if let Some(existing) = destination.iter_mut().find(|existing| {
        existing.physical_identity == range.physical_identity
            && existing.byte_count == range.byte_count
    }) {
        *existing = range;
    } else {
        destination.push(range);
    }
}

pub(crate) fn write_back_pending_globals(
    pending: &[PendingGlobalWriteback],
) -> Result<Vec<&PendingGlobalWriteback>, String> {
    let mut applied = Vec::with_capacity(pending.len());
    for writeback in pending {
        if writeback.original_bytes.len() != writeback.byte_count {
            return Err(format!(
                "writable global buffer snapshot length mismatch bytes={} snapshot={}",
                writeback.byte_count,
                writeback.original_bytes.len()
            ));
        }
        let offset = usize::try_from(writeback.offset)
            .map_err(|_| "writable global buffer offset does not fit usize")?;
        let end = offset
            .checked_add(writeback.byte_count)
            .ok_or("writable global buffer range overflow")?;
        if u64::try_from(end).map_or(true, |end| end > writeback.buffer.length()) {
            return Err(format!(
                "writable global buffer range exceeds Metal buffer offset={} bytes={} length={}",
                writeback.offset,
                writeback.byte_count,
                writeback.buffer.length()
            ));
        }
        let current_bytes = unsafe {
            std::slice::from_raw_parts(
                writeback.buffer.contents().cast::<u8>().add(offset),
                writeback.byte_count,
            )
        };
        let outcome = if let Some(expected) = writeback.mapping_identity {
            let byte_count = u64::try_from(writeback.byte_count)
                .map_err(|_| "writable global buffer byte count does not fit u64")?;
            let current = guest_shared_backing_identity(writeback.guest_address, byte_count);
            if current != Some(expected) {
                // Prospero recycles direct-memory VAs while GPU work is still draining. Publishing
                // through the captured producer identity keeps GPU results on those physical pages
                // (Poppy Slate glyph prep at 0x9010960000). Discarding left every later reader of
                // that backing with the pre-dispatch snapshot and solid tofu glyphs.
                trace_remapped_global_writeback_published_by_identity(
                    writeback.guest_address,
                    writeback.byte_count,
                    expected,
                    current,
                );
                write_changed_global_ranges_at_identity(writeback, current_bytes, expected)?
            } else {
                write_changed_global_ranges(writeback, current_bytes)?
            }
        } else if let Some(write_watch) = writeback.write_watch.as_ref() {
            match write_watch
                .with_live_mapping(|| write_changed_global_ranges_direct(writeback, current_bytes))
            {
                Ok(result) => {
                    result?;
                    GlobalWritebackOutcome::Applied
                }
                Err(error) if error.kind() == ErrorKind::NotFound => {
                    trace_discarded_stale_anonymous_global_writeback(
                        writeback.guest_address,
                        writeback.byte_count,
                        &error,
                    );
                    GlobalWritebackOutcome::DiscardedStale
                }
                Err(error) => {
                    return Err(format!(
                        "copied writable global mapping could not be pinned for completion address={:#x} bytes={}: {error}",
                        writeback.guest_address, writeback.byte_count
                    ));
                }
            }
        } else {
            #[cfg(test)]
            {
                write_changed_global_ranges_direct(writeback, current_bytes)?;
                GlobalWritebackOutcome::Applied
            }
            #[cfg(not(test))]
            {
                return Err(format!(
                    "writable global buffer has neither backing identity nor mapping watch address={:#x} bytes={}",
                    writeback.guest_address, writeback.byte_count
                ));
            }
        };
        if outcome == GlobalWritebackOutcome::Applied {
            applied.push(writeback);
        }
    }
    Ok(applied)
}

fn write_changed_global_ranges(
    writeback: &PendingGlobalWriteback,
    current_bytes: &[u8],
) -> Result<GlobalWritebackOutcome, String> {
    let expected = writeback
        .mapping_identity
        .expect("shared writeback path has a mapping identity");
    write_changed_global_ranges_with(
        writeback,
        current_bytes,
        expected,
        GlobalWritePublishMode::RequireLiveVa,
    )
}

fn write_changed_global_ranges_at_identity(
    writeback: &PendingGlobalWriteback,
    current_bytes: &[u8],
    expected: GuestSharedMappingIdentity,
) -> Result<GlobalWritebackOutcome, String> {
    write_changed_global_ranges_with(
        writeback,
        current_bytes,
        expected,
        GlobalWritePublishMode::ProducerIdentity,
    )
}

#[derive(Clone, Copy)]
enum GlobalWritePublishMode {
    RequireLiveVa,
    ProducerIdentity,
}

fn write_changed_global_ranges_with(
    writeback: &PendingGlobalWriteback,
    current_bytes: &[u8],
    expected: GuestSharedMappingIdentity,
    mode: GlobalWritePublishMode,
) -> Result<GlobalWritebackOutcome, String> {
    let mut cursor = 0;
    while cursor < current_bytes.len() {
        while cursor < current_bytes.len()
            && current_bytes[cursor] == writeback.original_bytes[cursor]
        {
            cursor += 1;
        }
        let start = cursor;
        while cursor < current_bytes.len()
            && current_bytes[cursor] != writeback.original_bytes[cursor]
        {
            cursor += 1;
        }
        if start == cursor {
            continue;
        }
        let address = writeback
            .guest_address
            .checked_add(
                u64::try_from(start)
                    .map_err(|_| "writable global buffer changed offset does not fit u64")?,
            )
            .ok_or("writable global buffer changed address overflow")?;
        let backing_offset = expected
            .backing_offset
            .checked_add(
                u64::try_from(start)
                    .map_err(|_| "writable global buffer backing offset does not fit u64")?,
            )
            .ok_or("writable global buffer backing offset overflow")?;
        let range_identity = GuestSharedMappingIdentity {
            backing_id: expected.backing_id,
            backing_offset,
        };
        let byte_count = u64::try_from(cursor - start)
            .map_err(|_| "writable global buffer changed byte count does not fit u64")?;
        let published = match mode {
            GlobalWritePublishMode::RequireLiveVa => {
                write_guest_shared_backing_if_identity(
                    address,
                    range_identity,
                    &current_bytes[start..cursor],
                )
            }
            GlobalWritePublishMode::ProducerIdentity => {
                trace_diagnostic_guest_write("global-producer-identity", address, byte_count);
                write_guest_shared_backing_at_identity(range_identity, &current_bytes[start..cursor])
            }
        };
        if published {
            journal_guest_write(
                match mode {
                    GlobalWritePublishMode::RequireLiveVa => GuestWriteSite::GlobalWriteback,
                    GlobalWritePublishMode::ProducerIdentity => {
                        GuestWriteSite::GlobalWritebackIdentity
                    }
                },
                address,
                byte_count,
                &current_bytes[start..cursor],
            );
        }
        if !published {
            let current = guest_shared_backing_identity(address, byte_count);
            if matches!(mode, GlobalWritePublishMode::RequireLiveVa) && current != Some(range_identity)
            {
                // VA moved mid-publish; fall back to the captured producer pages.
                trace_diagnostic_guest_write("global-producer-identity", address, byte_count);
                if write_guest_shared_backing_at_identity(
                    range_identity,
                    &current_bytes[start..cursor],
                ) {
                    journal_guest_write(
                        GuestWriteSite::GlobalWritebackIdentity,
                        address,
                        byte_count,
                        &current_bytes[start..cursor],
                    );
                    continue;
                }
                trace_discarded_stale_global_writeback(
                    address,
                    cursor - start,
                    "during completion",
                    range_identity,
                    current,
                );
                return Ok(GlobalWritebackOutcome::DiscardedStale);
            }
            if matches!(mode, GlobalWritePublishMode::ProducerIdentity) {
                trace_discarded_stale_global_writeback(
                    address,
                    cursor - start,
                    "producer identity unavailable",
                    range_identity,
                    current,
                );
                return Ok(GlobalWritebackOutcome::DiscardedStale);
            }
            return Err(format!(
                "writable global buffer backing write failed address={address:#x} bytes={} identity={range_identity:?}",
                cursor - start
            ));
        }
    }
    Ok(GlobalWritebackOutcome::Applied)
}

fn trace_discarded_stale_global_writeback(
    address: u64,
    byte_count: usize,
    stage: &str,
    expected: GuestSharedMappingIdentity,
    current: Option<GuestSharedMappingIdentity>,
) {
    let sequence = DISCARDED_STALE_GLOBAL_WRITEBACKS.fetch_add(1, Ordering::Relaxed) + 1;
    if sequence <= 16 || sequence.is_power_of_two() {
        eprintln!(
            "[REMu][METAL] discarded stale writable global writeback sequence={sequence} mapping changed {stage} address={address:#x} bytes={byte_count} expected={expected:?} current={current:?}"
        );
    }
}

fn trace_remapped_global_writeback_published_by_identity(
    address: u64,
    byte_count: usize,
    expected: GuestSharedMappingIdentity,
    current: Option<GuestSharedMappingIdentity>,
) {
    static PUBLISHED: AtomicUsize = AtomicUsize::new(0);
    let sequence = PUBLISHED.fetch_add(1, Ordering::Relaxed) + 1;
    if sequence <= 16 || sequence.is_power_of_two() {
        eprintln!(
            "[REMu][METAL] published remapped writable global writeback sequence={sequence} by producer identity address={address:#x} bytes={byte_count} expected={expected:?} current={current:?}"
        );
    }
}

fn trace_discarded_stale_anonymous_global_writeback(
    address: u64,
    byte_count: usize,
    error: &std::io::Error,
) {
    let sequence = DISCARDED_STALE_GLOBAL_WRITEBACKS.fetch_add(1, Ordering::Relaxed) + 1;
    if sequence <= 16 || sequence.is_power_of_two() {
        eprintln!(
            "[REMu][METAL] discarded stale writable global writeback sequence={sequence} anonymous mapping changed before completion address={address:#x} bytes={byte_count}: {error}"
        );
    }
}

fn write_changed_global_ranges_direct(
    writeback: &PendingGlobalWriteback,
    current_bytes: &[u8],
) -> Result<(), String> {
    let destination = usize::try_from(writeback.guest_address)
        .map_err(|_| "writable global buffer address does not fit usize")?;
    let mut cursor = 0;
    while cursor < current_bytes.len() {
        while cursor < current_bytes.len()
            && current_bytes[cursor] == writeback.original_bytes[cursor]
        {
            cursor += 1;
        }
        let start = cursor;
        while cursor < current_bytes.len()
            && current_bytes[cursor] != writeback.original_bytes[cursor]
        {
            cursor += 1;
        }
        if start == cursor {
            continue;
        }
        // SAFETY: Production holds the mapping's CpuWriteWatch registry lock across this copy.
        // Unit tests keep their local destination allocation alive for the duration of the call.
        unsafe {
            std::ptr::copy_nonoverlapping(
                current_bytes[start..cursor].as_ptr(),
                (destination as *mut u8).add(start),
                cursor - start,
            );
        }
        journal_guest_write(
            GuestWriteSite::GlobalWritebackDirect,
            writeback.guest_address.wrapping_add(start as u64),
            (cursor - start) as u64,
            &current_bytes[start..cursor],
        );
    }
    Ok(())
}
