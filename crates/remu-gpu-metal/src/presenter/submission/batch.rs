// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::completion::{
    COMMAND_BUFFER_SETTLE_TIMEOUT, settle_timeout_message, wait_until_settled_within,
};
use super::{
    GLOBAL_UPLOAD_ALIGNMENT, GLOBAL_UPLOAD_CHUNK_BYTE_COUNT, GlobalUploadChunk,
    GuestBatchFlushReason, GuestBatchWork, GuestCommandBatch, MetalPresenter,
    PendingGlobalWriteback, PendingGuestWriteRange, UploadedGlobalBuffer,
    guest_physical_ranges_overlap,
};
use metal::foreign_types::ForeignType;
use metal::{Buffer, CommandBuffer, Device, MTLCommandBufferStatus, MTLResourceOptions};
use objc::{msg_send, sel, sel_impl};
use remu_shader::ShaderEvaluation;
use std::time::{Duration, Instant};

#[derive(Debug, PartialEq, Eq)]
enum GuestBatchSettlement {
    Completed,
    Failed(String),
    Pending(String),
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
            global_upload_chunks: Vec::new(),
            global_writebacks: Vec::new(),
            guest_write_ranges: Vec::new(),
            memory_writes: Vec::new(),
            render_pass: None,
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
                    let execution_error = format!(
                        "Metal guest command batch [{}] ended in {:?}",
                        describe_guest_batch(&batch.work),
                        batch.command_buffer.status()
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
            if writeback.is_ok() {
                self.record_produced_global_writebacks(&batch.global_writebacks);
            }
            result = merge_guest_batch_result(result, "global writeback", writeback);
        }
        // RELEASE_MEM, WRITE_DATA, and DMA fills are command-processor effects ordered after the
        // native work, not shader results. Once the Metal command buffer reaches either terminal
        // state, publish them even when the draw/dispatch or an unrelated global readback failed.
        // This matches the existing policy of retiring the submission after a native error while
        // retaining the error as the primary result.
        self.publish_terminal_guest_memory_writes(&batch.memory_writes, result)
    }

    pub(crate) fn flush_guest_command_batch(
        &mut self,
        wait: bool,
        reason: GuestBatchFlushReason,
    ) -> Result<(), String> {
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
            GuestBatchFlushReason::CpuSynchronization => {
                self.statistics.batch_flushes_for_cpu_synchronization = self
                    .statistics
                    .batch_flushes_for_cpu_synchronization
                    .saturating_add(1);
            }
        }
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
        batch.command_buffer.commit();
        self.statistics.submitted_guest_batches =
            self.statistics.submitted_guest_batches.saturating_add(1);
        if wait {
            match settle_guest_batch_by(&batch, Instant::now() + COMMAND_BUFFER_SETTLE_TIMEOUT) {
                GuestBatchSettlement::Completed => {
                    let result = self.finish_terminal_guest_batch(&batch, None);
                    self.statistics.gpu_execution_time +=
                        completed_command_buffer_gpu_duration(&batch.command_buffer);
                    publish_guest_batch_completion(&batch);
                    result?;
                }
                GuestBatchSettlement::Failed(error) => {
                    let result = self.finish_terminal_guest_batch(&batch, Some(error));
                    self.statistics.gpu_execution_time +=
                        completed_command_buffer_gpu_duration(&batch.command_buffer);
                    publish_guest_batch_completion(&batch);
                    result?;
                }
                GuestBatchSettlement::Pending(error) => {
                    // A timeout is an observation about the host wait, not a terminal Metal
                    // status. Keep the command buffer and every resource/completion effect alive;
                    // the normal retirement path will finish it once the driver does.
                    self.committed_guest_batches.push_back(batch);
                    return Err(error);
                }
            }
        } else {
            self.committed_guest_batches.push_back(batch);
        }
        Ok(())
    }

    pub(crate) fn wait_for_guest_command_batches(&mut self) -> Result<(), String> {
        self.wait_for_guest_command_batches_within(COMMAND_BUFFER_SETTLE_TIMEOUT)
    }

    pub(crate) fn wait_for_guest_command_batches_within(
        &mut self,
        timeout: Duration,
    ) -> Result<(), String> {
        self.flush_guest_command_batch(false, GuestBatchFlushReason::CpuSynchronization)?;
        let mut result = Ok(());
        // One budget covers the whole drain. Charging every queued batch its own timeout would sum
        // to far past the host watchdog window this bound exists to stay inside, and a device that
        // stalled on the first batch will not retire the rest either.
        let deadline = Instant::now() + timeout;
        while let Some(batch) = self.committed_guest_batches.pop_front() {
            let execution_error = match settle_guest_batch_by(&batch, deadline) {
                GuestBatchSettlement::Completed => None,
                GuestBatchSettlement::Failed(error) => Some(error),
                GuestBatchSettlement::Pending(error) => {
                    // Preserve ordering: later buffers from the same queue cannot complete before
                    // this one, and none of their CPU-side completion effects may overtake it.
                    self.committed_guest_batches.push_front(batch);
                    return merge_guest_batch_result(
                        result,
                        "pending ordered guest batch",
                        Err(error),
                    );
                }
            };
            let batch_result = self.finish_terminal_guest_batch(&batch, execution_error);
            self.statistics.gpu_execution_time +=
                completed_command_buffer_gpu_duration(&batch.command_buffer);
            publish_guest_batch_completion(&batch);
            // Drain every already-committed batch even after one fails. Returning early here used
            // to abandon later RELEASE_MEM labels and completion IDs during shutdown or a
            // synchronous readback, converting a preserved Metal error into an unrelated guest
            // RenderThread timeout.
            result = merge_guest_batch_result(result, "later ordered guest batch", batch_result);
        }
        result
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
        GuestBatchSettlement::Failed(format!("{operation} ended in {status:?}"))
    }
}

pub(crate) fn describe_guest_batch(work: &[GuestBatchWork]) -> String {
    work.iter()
        .map(|item| match item {
            GuestBatchWork::Draw { export, pixel } => format!("draw({export:#x},{pixel:#x})"),
            GuestBatchWork::Compute { shader } => format!("compute({shader:#x})"),
        })
        .collect::<Vec<_>>()
        .join(",")
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
    pub(crate) fn upload_global_bytes(
        &mut self,
        device: &Device,
        bytes: &[u8],
    ) -> Result<(Buffer, u64), String> {
        let byte_count = bytes.len().max(1);
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
        if !bytes.is_empty() {
            // SAFETY: the shared Metal buffer is at least `aligned_byte_count` bytes beyond
            // `offset`, and the source slice remains live for the exact non-overlapping copy.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    chunk.buffer.contents().cast::<u8>().add(offset),
                    bytes.len(),
                );
            }
        }
        chunk.used = offset + aligned_byte_count;
        Ok((
            chunk.buffer.clone(),
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
}

pub(crate) fn guest_batch_writes_overlap(
    batch: &GuestCommandBatch,
    address: u64,
    byte_count: u64,
) -> bool {
    batch.guest_write_ranges.iter().any(|range| {
        guest_physical_ranges_overlap(range.address, range.byte_count, address, byte_count)
    }) || batch.global_writebacks.iter().any(|writeback| {
        u64::try_from(writeback.byte_count).is_ok_and(|length| {
            guest_physical_ranges_overlap(writeback.guest_address, length, address, byte_count)
        })
    }) || batch.memory_writes.iter().any(|write| {
        let (write_address, write_bytes) = write.range();
        guest_physical_ranges_overlap(write_address, write_bytes, address, byte_count)
    })
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
        pending.push(PendingGlobalWriteback {
            buffer: buffer.buffer.clone(),
            offset: buffer.offset,
            guest_address: binding.base_address,
            byte_count: binding.byte_count,
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
    for (_, binding) in buffers
        .iter()
        .zip(&evaluation.global_memory_bindings)
        .filter(|(buffer, binding)| {
            binding.writable && (buffer.guest_memory || binding.write_back_to_guest)
        })
    {
        let byte_count = u64::try_from(binding.byte_count)
            .map_err(|_| "writable guest buffer byte count does not fit u64")?;
        let range = PendingGuestWriteRange {
            address: binding.base_address,
            byte_count,
        };
        if !pending.contains(&range) {
            pending.push(range);
        }
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
        if !destination.contains(&range) {
            destination.push(range);
        }
    }
}

pub(crate) fn write_back_pending_globals(pending: &[PendingGlobalWriteback]) -> Result<(), String> {
    for writeback in pending {
        let destination = usize::try_from(writeback.guest_address)
            .map_err(|_| "writable global buffer address does not fit usize")?;
        // SAFETY: Shader evaluation captured this live writable guest binding, and the completed
        // command buffer owns a Metal buffer of the same exact byte length.
        unsafe {
            std::ptr::copy_nonoverlapping(
                writeback.buffer.contents().cast::<u8>().add(
                    usize::try_from(writeback.offset)
                        .map_err(|_| "writable global buffer offset does not fit usize")?,
                ),
                destination as *mut u8,
                writeback.byte_count,
            );
        }
    }
    Ok(())
}
