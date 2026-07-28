// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    ActiveGuestMemoryWait, DisplayConfiguration, GLOBAL_UPLOAD_ALIGNMENT,
    GLOBAL_UPLOAD_CHUNK_BYTE_COUNT, GPU_EVENTS_PER_TICK, GlobalUploadChunk, GraphicsPipelineKey,
    GuestBatchFlushReason, GuestBatchWork, GuestCommandBatch, GuestMemoryWaitKey,
    GuestMemoryWaitRecovery, GuestMemoryWaitStatus, GuestMemoryWaitTracker, MetalPresenter,
    OrderedEventPoll, OrderedEventSource, OrderedGpuEventReceiver, PIPELINE_DRAW_GATES,
    PendingFlip, PendingGlobalWriteback, PendingGuestMemoryWrite, PendingGuestWriteRange,
    PresenterFlow, QueueBlockReason, ScheduledGpuSubmission, ScheduledHostEvent,
    UploadedGlobalBuffer, WriteWatchRefresh, graphics_pipeline_key, guest_physical_ranges_overlap,
    guest_ranges_overlap, guest_texture_byte_count, pixel_scalars, record_diagnostic_memory_copy,
    refresh_resident_texture_write_watch, register_value,
    retire_multisample_textures_overlapping_writeback,
    retire_resident_textures_overlapping_writeback, retire_sampled_textures_matching,
    retire_sampled_textures_overlapping_writeback, skip_incomplete_rectangle_draw,
    trace_video_target_compute_consumer, trace_video_target_draw_consumer,
    write_resident_texture_contents_to_guest,
};
use metal::foreign_types::ForeignType;
use metal::{
    Buffer, CommandBuffer, Device, MTLCommandBufferStatus, MTLRenderStages, MTLResourceOptions,
    NSUInteger,
};
use objc::{msg_send, sel, sel_impl};
use remu_gpu::{
    GpuEvent, GpuMemoryCopy, GpuMemoryRelease, GpuMemoryReleaseValue, GpuMemoryWait,
    GpuMemoryWrite, GpuWaitId, IndexedDraw,
};
use remu_shader::ShaderEvaluation;
use std::collections::{HashMap, VecDeque};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

impl MetalPresenter {
    #[allow(clippy::too_many_lines)]
    pub(super) fn receive_events(&mut self) -> PresenterFlow {
        self.receive_shader_compilations();
        self.retire_completed_guest_batches();
        self.retire_completed_presentations();
        // A guest label may have been published by a Metal batch that retired since the previous
        // AppKit tick. Retry suspended queue fronts once; an unsatisfied wait is re-suspended
        // without permitting any later event from that queue to overtake it.
        self.receiver.retry_guest_memory_waits();
        // Feed the existing compile workers from bounded ordered lookahead even while the current
        // queue front is runnable. Waiting until a cold draw becomes the blocked front serialized
        // multi-second first-scene Metal compilation behind the warning/movie UI. This only
        // snapshots immutable draw descriptions and schedules deduplicated pipeline keys; event
        // execution and every queue-local dependency remain in their original order.
        for draw in self.receiver.prefetch_draws(512, 128) {
            if let Ok(key) = graphics_pipeline_key(&draw) {
                self.compile_draw_shaders(&draw, &key);
            }
        }
        // A bounded batch prevents a continuously producing guest from starving redraw and input
        // delivery. Queue-local shader and memory waits leave independent guest queues runnable.
        for _ in 0..GPU_EVENTS_PER_TICK {
            let (source, event) = match self.receiver.try_recv(|key| {
                self.compiled_shader_pairs.contains_key(key)
                    || self.failed_shader_pairs.contains(key)
            }) {
                OrderedEventPoll::Event { source, event } => (source, event),
                OrderedEventPoll::Blocked => {
                    // Look ahead far enough to feed all compiler workers while memory-wait queues
                    // remain suspended. Prefetching never executes or reorders an event.
                    for draw in self.receiver.prefetch_draws(512, 128) {
                        if let Ok(key) = graphics_pipeline_key(&draw) {
                            self.compile_draw_shaders(&draw, &key);
                        }
                    }
                    return PresenterFlow::Continue;
                }
                OrderedEventPoll::Empty => {
                    return PresenterFlow::Continue;
                }
                OrderedEventPoll::Disconnected => {
                    eprintln!("[REMu][METAL] GPU event channel disconnected; closing presenter");
                    if let Err(error) = self.wait_for_guest_command_batches() {
                        eprintln!("[REMu][METAL] shutdown synchronization failed: {error}");
                    }
                    self.persist_render_pipeline_archive();
                    return PresenterFlow::Shutdown;
                }
            };
            match event {
                GpuEvent::QueueSubmission { .. } => {
                    eprintln!("[REMu][METAL] ignored invalid nested GPU queue submission");
                }
                GpuEvent::DisplayRegistered {
                    width,
                    height,
                    pitch,
                    pixel_format,
                    tiling_mode,
                    ..
                } => {
                    self.display = Some(DisplayConfiguration {
                        width,
                        height,
                        pitch,
                        pixel_format,
                        tiling_mode,
                    });
                    self.frame_texture = None;
                    self.active_translated_scanout = None;
                    eprintln!(
                        "[REMu][METAL] display registered {width}x{height} pitch={pitch} format={pixel_format:#x} tiling={tiling_mode}"
                    );
                }
                GpuEvent::ShowHostSplash(image) => {
                    let dimensions = (image.width(), image.height());
                    match self.show_host_splash(image) {
                        Ok(()) => eprintln!(
                            "[REMu][METAL] installed-title system splash ready {}x{}",
                            dimensions.0, dimensions.1
                        ),
                        Err(error) => {
                            eprintln!("[REMu][METAL] could not upload system splash: {error}");
                        }
                    }
                }
                GpuEvent::HideHostSplash => {
                    if self.host_splash.take().is_some() {
                        eprintln!("[REMu][METAL] hiding installed-title system splash");
                    }
                    self.redraw_pending = true;
                }
                GpuEvent::Flip {
                    address,
                    buffer_index,
                    flip_argument,
                    width,
                    height,
                    pitch,
                    pixel_format,
                    tiling_mode,
                    ..
                } => {
                    self.statistics.guest_flips = self.statistics.guest_flips.saturating_add(1);
                    self.display = Some(DisplayConfiguration {
                        width,
                        height,
                        pitch,
                        pixel_format,
                        tiling_mode,
                    });
                    self.next_flip_sequence = self.next_flip_sequence.saturating_add(1);
                    self.pending_flip = Some(PendingFlip {
                        address,
                        buffer_index,
                        flip_argument,
                        sequence: self.next_flip_sequence,
                    });
                    self.pending_flip_target_written_since_boundary = false;
                    self.active_translated_scanout = None;
                    self.pending_flip_queue = match source {
                        OrderedEventSource::GuestQueue(queue_id) => {
                            // The flip packet is an order-visible point inside the guest command
                            // stream. Preserve the target here, after all preceding draws but before
                            // following work can reuse the same allocation for the next frame.
                            self.finalize_pending_flip();
                            (self.uploaded_flip_sequence != self.next_flip_sequence)
                                .then_some(queue_id)
                        }
                        OrderedEventSource::Host { .. } => {
                            self.finalize_pending_flip();
                            None
                        }
                    };
                }
                GpuEvent::PipelineBarrier(barrier) => self.end_guest_pipeline_barrier(barrier),
                GpuEvent::MemoryAcquire(acquire) => {
                    if let Err(error) = self.synchronize_cpu_written_guest_resources(acquire) {
                        eprintln!("[REMu][METAL] ACQUIRE_MEM synchronization failed: {error}");
                    }
                }
                GpuEvent::MemoryCopy(copy) => {
                    if let Err(error) = self.execute_guest_memory_copy(copy) {
                        eprintln!("[REMu][METAL] DMA_DATA synchronization failed: {error}");
                    }
                }
                GpuEvent::MemoryWrite(write) => {
                    if let Err(error) = self.execute_guest_memory_write(&write) {
                        eprintln!("[REMu][METAL] WRITE_DATA synchronization failed: {error}");
                    }
                    self.receiver.retry_guest_memory_waits();
                }
                GpuEvent::MemoryRelease(release) => {
                    if let Err(error) = self.execute_guest_memory_release(release) {
                        eprintln!("[REMu][METAL] RELEASE_MEM synchronization failed: {error}");
                    }
                    self.receiver.retry_guest_memory_waits();
                }
                GpuEvent::MemoryWait(wait) => {
                    match self.execute_guest_memory_wait(source, None, wait) {
                        Ok(GuestMemoryWaitStatus::Satisfied) => {}
                        Ok(GuestMemoryWaitStatus::Blocked) => {
                            self.receiver.block_event(
                                source,
                                GpuEvent::MemoryWait(wait),
                                QueueBlockReason::GuestMemoryWait,
                            );
                            return PresenterFlow::Continue;
                        }
                        Err(error) => {
                            eprintln!("[REMu][METAL] WAIT_REG_MEM synchronization failed: {error}");
                        }
                    }
                }
                GpuEvent::RegisteredMemoryWait { id, wait } => {
                    match self.execute_guest_memory_wait(source, Some(id), wait) {
                        Ok(GuestMemoryWaitStatus::Satisfied) => {}
                        Ok(GuestMemoryWaitStatus::Blocked) => {
                            self.receiver.block_event(
                                source,
                                GpuEvent::RegisteredMemoryWait { id, wait },
                                QueueBlockReason::GuestMemoryWait,
                            );
                            return PresenterFlow::Continue;
                        }
                        Err(error) => {
                            eprintln!(
                                "[REMu][METAL] registered WAIT_REG_MEM synchronization failed: {error}"
                            );
                        }
                    }
                }
                GpuEvent::ComputeDispatch(dispatch) => {
                    trace_video_target_compute_consumer(&dispatch);
                    match self.execute_compute_dispatch(&dispatch) {
                        Ok(()) => {
                            self.statistics.encoded_compute_dispatches =
                                self.statistics.encoded_compute_dispatches.saturating_add(1);
                        }
                        Err(error)
                            if self
                                .failed_compute_execution
                                .insert(dispatch.shader_address) =>
                        {
                            eprintln!(
                                "[REMu][METAL] Gen5 compute cs={:#x} could not execute: {error}",
                                dispatch.shader_address
                            );
                        }
                        Err(_) => {}
                    }
                }
                GpuEvent::IndexedDraw(draw) => {
                    trace_video_target_draw_consumer(&draw);
                    if skip_incomplete_rectangle_draw(&draw) {
                        static SKIPPED_INCOMPLETE_RECTANGLES: AtomicUsize = AtomicUsize::new(0);
                        let sequence =
                            SKIPPED_INCOMPLETE_RECTANGLES.fetch_add(1, Ordering::Relaxed) + 1;
                        if sequence <= 32 || sequence.is_power_of_two() {
                            eprintln!(
                                "[REMu][METAL][RECT-INTERFACE-SKIP] sequence={sequence} draw={} es={:#x} ps={:#x} preserves prior target contents",
                                draw.sequence,
                                draw.export_shader_address,
                                draw.pixel_shader_address,
                            );
                        }
                        self.receiver.finish_event(source);
                        continue;
                    }
                    let key = graphics_pipeline_key(&draw);
                    if let Ok(key) = key.as_ref() {
                        self.compile_draw_shaders(&draw, key);
                        if !self.compiled_shader_pairs.contains_key(key)
                            && !self.failed_shader_pairs.contains(key)
                        {
                            PIPELINE_DRAW_GATES.fetch_add(1, Ordering::Relaxed);
                            static LOGGED_PIPELINE_GATES: AtomicUsize = AtomicUsize::new(0);
                            if LOGGED_PIPELINE_GATES.fetch_add(1, Ordering::Relaxed) < 16 {
                                eprintln!(
                                    "[REMu][METAL] preserving cold draw={} es={:#x} ps={:#x} while its pipeline compiles",
                                    draw.sequence,
                                    draw.export_shader_address,
                                    draw.pixel_shader_address,
                                );
                            }
                            // A draw may be the only producer of a render target or synchronization
                            // value. Keep it at the front of its guest queue until the worker
                            // returns its pipeline; dropping it lets later barriers and submission
                            // boundaries retire work that Metal never encoded. Other guest queues
                            // remain runnable, and the blocked poll path looks ahead to compile
                            // subsequent shader pairs in parallel without executing them.
                            self.receiver.block_event(
                                source,
                                GpuEvent::IndexedDraw(draw),
                                QueueBlockReason::GraphicsPipeline(key.clone()),
                            );
                            return PresenterFlow::Continue;
                        }
                    }
                    let result = match key.as_ref() {
                        Ok(key) => self.execute_compiled_draw(&draw, key),
                        Err(error) => Err(error.clone()),
                    };
                    if let Err(error) = result {
                        if key
                            .as_ref()
                            .is_ok_and(|key| self.failed_draw_execution.insert(key.clone()))
                        {
                            eprintln!(
                                "[REMu][METAL] Gen5 draw es={:#x} ps={:#x} could not execute: {error}",
                                draw.export_shader_address, draw.pixel_shader_address
                            );
                        }
                        if self.logged_unsupported_guest_draws.len() < 16
                            && self
                                .logged_unsupported_guest_draws
                                .insert((draw.export_shader_prefix, draw.pixel_shader_prefix))
                        {
                            let pixel_scalars = pixel_scalars(&draw);
                            let pipeline_key_error =
                                key.as_ref().err().map_or("none", |error| error.as_str());
                            eprintln!(
                                "[REMu][METAL] no translated Metal path for es={:#x}/{:x?} ps={:#x}/{:x?} indexed={} count={} instances={} target={:#x} cb_info={:#x} cb_attrib2={:#x} cb_attrib3={:#x} target_mask={:#x} ps_scalars={pixel_scalars:x?} pipeline_key_error={pipeline_key_error}",
                                draw.export_shader_address,
                                draw.export_shader_prefix,
                                draw.pixel_shader_address,
                                draw.pixel_shader_prefix,
                                draw.indexed,
                                draw.index_count,
                                draw.instance_count,
                                draw.color_target_address,
                                register_value(&draw.context_registers, 0x31c).unwrap_or_default(),
                                register_value(&draw.context_registers, 0x3b0).unwrap_or_default(),
                                register_value(&draw.context_registers, 0x3b8).unwrap_or_default(),
                                register_value(&draw.context_registers, 0x08e).unwrap_or_default(),
                            );
                        }
                    } else {
                        self.statistics.encoded_draws =
                            self.statistics.encoded_draws.saturating_add(1);
                    }
                }
                GpuEvent::SubmissionBoundary(sequence) => {
                    // Batch across guest submissions without losing their individual completion
                    // identities. Graphics and compute queues can retire out of global sequence
                    // order, and the shared completion tracker exposes only the contiguous prefix.
                    if let Some(batch) = self.open_guest_batch.as_mut() {
                        batch.completion_sequences.push(sequence);
                    } else if let Some(batch) = self.committed_guest_batches.back_mut() {
                        // Empty submissions and boundaries following a forced flush still depend
                        // on all earlier ordered work. Attaching them to the newest committed
                        // batch avoids publishing completion before that work reaches Metal.
                        batch.completion_sequences.push(sequence);
                    } else {
                        // No open or committed work remains, so this submission is genuinely
                        // empty and can complete immediately.
                        remu_gpu::record_gpu_submission_completed(sequence);
                    }
                    self.finalize_pending_flip_at_submission_boundary(source);
                }
                GpuEvent::Shutdown => {
                    eprintln!("[REMu][METAL] guest runtime requested presenter shutdown");
                    if let Err(error) = self.wait_for_guest_command_batches() {
                        eprintln!("[REMu][METAL] shutdown synchronization failed: {error}");
                    }
                    self.persist_render_pipeline_archive();
                    return PresenterFlow::Shutdown;
                }
            }
            self.receiver.finish_event(source);
            // Do not let a large guest GPU backlog delay an already-requested presentation.
            // In particular, the first flip can precede creation of its translated scanout
            // texture; once a later draw produces that texture, return to Winit immediately so
            // `about_to_wait` can request the redraw instead of processing hundreds more events.
            if self.redraw_pending {
                return PresenterFlow::Continue;
            }
        }
        PresenterFlow::Continue
    }

    pub(super) fn ensure_open_guest_batch(&mut self, queue: &metal::CommandQueue) {
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

    pub(super) fn retire_completed_guest_batches(&mut self) {
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

    pub(super) fn publish_terminal_guest_memory_writes(
        &mut self,
        pending: &[PendingGuestMemoryWrite],
        result: Result<(), String>,
    ) -> Result<(), String> {
        let publication = apply_pending_guest_memory_writes(pending);
        if publication.is_ok() {
            self.record_produced_guest_memory_writes(pending);
        }
        merge_guest_batch_result(result, "ordered guest memory publication", publication)
    }

    pub(super) fn finish_terminal_guest_batch(
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

    pub(super) fn end_guest_pipeline_barrier(&mut self, _barrier: remu_gpu::GpuPipelineBarrier) {
        let Some(batch) = self.open_guest_batch.as_mut() else {
            return;
        };
        // EVENT_WRITE partial flushes are memory dependencies, not submission boundaries. Kyty
        // emits a Vulkan memory barrier in the existing command buffer and shadPS4 likewise does
        // not submit at these events. Keeping a Metal render pass open is particularly important
        // on a tile GPU: ending it here stores every attachment, and the next draw immediately
        // reloads the same attachments.
        //
        // Compute encoders are already ended immediately after each dispatch, so Metal's serial
        // encoder order and tracked resource hazards cover inter-pass dependencies. When the
        // current work is a render pass, use Metal's in-pass all-resource barrier to make earlier
        // vertex/fragment writes visible to following draws without spilling the attachments.
        if let Some(pass) = batch.render_pass.as_ref() {
            const ALL_RESOURCE_SCOPES: NSUInteger = (1 << 0) | (1 << 1) | (1 << 2);
            let render_stages = MTLRenderStages::Vertex | MTLRenderStages::Fragment;
            // SAFETY: `pass.encoder` is a live MTLRenderCommandEncoder and the selector has been
            // available on macOS since 10.14. The three NSUInteger arguments exactly match
            // MTLBarrierScope, `afterStages`, and `beforeStages`.
            unsafe {
                let _: () = msg_send![
                    pass.encoder.as_ref(),
                    memoryBarrierWithScope: ALL_RESOURCE_SCOPES
                    afterStages: render_stages.bits()
                    beforeStages: render_stages.bits()
                ];
            }
        }
        self.statistics.coalesced_pipeline_barriers = self
            .statistics
            .coalesced_pipeline_barriers
            .saturating_add(1);
    }

    pub(super) fn flush_guest_command_batch(
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
            batch.command_buffer.wait_until_completed();
            let execution_error =
                (batch.command_buffer.status() != MTLCommandBufferStatus::Completed).then(|| {
                    format!(
                        "Metal guest command batch [{}] ended in {:?}",
                        describe_guest_batch(&batch.work),
                        batch.command_buffer.status()
                    )
                });
            let result = self.finish_terminal_guest_batch(&batch, execution_error);
            self.statistics.gpu_execution_time +=
                completed_command_buffer_gpu_duration(&batch.command_buffer);
            publish_guest_batch_completion(&batch);
            result?;
        } else {
            self.committed_guest_batches.push_back(batch);
        }
        Ok(())
    }

    pub(super) fn wait_for_guest_command_batches(&mut self) -> Result<(), String> {
        self.flush_guest_command_batch(false, GuestBatchFlushReason::CpuSynchronization)?;
        let mut result = Ok(());
        while let Some(batch) = self.committed_guest_batches.pop_front() {
            batch.command_buffer.wait_until_completed();
            let execution_error =
                (batch.command_buffer.status() != MTLCommandBufferStatus::Completed).then(|| {
                    format!(
                        "Metal guest command batch [{}] ended in {:?}",
                        describe_guest_batch(&batch.work),
                        batch.command_buffer.status()
                    )
                });
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

    pub(super) fn execute_guest_memory_release(
        &mut self,
        release: GpuMemoryRelease,
    ) -> Result<(), String> {
        if release.address == 0 {
            return Err("release destination is null".to_owned());
        }
        validate_guest_memory_release(release)?;
        self.queue_pending_guest_memory_write(PendingGuestMemoryWrite::Release(release))
    }

    pub(super) fn execute_guest_memory_copy(&mut self, copy: GpuMemoryCopy) -> Result<(), String> {
        // GFX10 selector 3 is the regular address path with L2 participation; selector 0 is the
        // uncached address path. Both name guest virtual memory. Source selector 2 repeats the
        // packet's inline low dword across the destination; selector 1 names unsupported GDS.
        if !matches!(copy.source_selection, 0 | 2 | 3)
            || !matches!(copy.destination_selection, 0 | 3)
        {
            return Err(format!(
                "unsupported selectors source={} destination={}",
                copy.source_selection, copy.destination_selection
            ));
        }
        let byte_count = usize::try_from(copy.byte_count)
            .map_err(|_| "DMA byte count does not fit the host address space")?;
        if byte_count == 0 || copy.byte_count > 512 * 1024 * 1024 {
            return Err(format!("invalid DMA byte count {:#x}", copy.byte_count));
        }
        copy.destination_address
            .checked_add(copy.byte_count)
            .ok_or("DMA destination range overflows")?;
        record_diagnostic_memory_copy(copy);

        if copy.source_selection == 2 {
            if !copy.destination_address.is_multiple_of(4) || !copy.byte_count.is_multiple_of(4) {
                return Err(format!(
                    "immediate DMA fill destination={:#x} bytes={:#x} is not dword aligned",
                    copy.destination_address, copy.byte_count
                ));
            }
            self.queue_pending_guest_memory_write(PendingGuestMemoryWrite::Fill {
                address: copy.destination_address,
                value: copy.source_address as u32,
                dword_count: copy.byte_count / 4,
            })?;
        } else {
            copy.source_address
                .checked_add(copy.byte_count)
                .ok_or("DMA source range overflows")?;

            // DMA observes preceding GPU writes to its source, but it does not need to drain
            // unrelated rendering. Most Unity DMA_DATA packets move CPU-populated upload pages;
            // synchronously waiting for the complete Metal queue for every one of those packets
            // cuts frame rate in half. Track actual writable guest ranges so only a real
            // read-after-write dependency pays for retirement/readback.
            let mut source_descriptors = self
                .resident_guest_textures
                .values()
                .filter(|resident| resident.gpu_written)
                .filter(|resident| {
                    resident
                        .write_watch
                        .overlaps_alias_range(copy.source_address, copy.byte_count)
                })
                .map(|resident| resident.descriptor)
                .collect::<Vec<_>>();
            source_descriptors.sort_by_key(|descriptor| {
                (
                    self.resident_guest_textures
                        .get(descriptor)
                        .map_or(0, |resident| resident.content_version),
                    descriptor.address,
                )
            });
            let pending_source_write = self.open_guest_batch.as_ref().is_some_and(|batch| {
                guest_batch_writes_overlap(batch, copy.source_address, copy.byte_count)
            }) || self.committed_guest_batches.iter().any(|batch| {
                guest_batch_writes_overlap(batch, copy.source_address, copy.byte_count)
            });
            let source_was_gpu_written = !source_descriptors.is_empty();
            if !source_descriptors.is_empty() || pending_source_write {
                self.wait_for_guest_command_batches()?;
            }
            if !source_descriptors.is_empty() {
                let device = self
                    .device
                    .as_ref()
                    .ok_or("Metal device is unavailable")?
                    .clone();
                let queue = self
                    .command_queue
                    .as_ref()
                    .ok_or("Metal command queue is unavailable")?
                    .clone();
                for descriptor in source_descriptors {
                    let refresh = refresh_resident_texture_write_watch(
                        self.resident_guest_textures
                            .get_mut(&descriptor)
                            .expect("collected DMA source texture exists"),
                    )?;
                    if refresh == WriteWatchRefresh::Replaced {
                        self.resident_guest_textures.remove(&descriptor);
                        continue;
                    }
                    if refresh == WriteWatchRefresh::Reprotecting {
                        return Err(format!(
                            "DMA source mapping {:#x} is being reprotected",
                            descriptor.address
                        ));
                    }
                    let (alias_ranges, content_version) = {
                        let resident = self
                            .resident_guest_textures
                            .get(&descriptor)
                            .expect("collected DMA source texture exists");
                        (
                            resident.write_watch.alias_ranges(),
                            resident.content_version,
                        )
                    };
                    retire_sampled_textures_overlapping_writeback(
                        &mut self.sampled_texture_cache,
                        &alias_ranges,
                    );
                    retire_resident_textures_overlapping_writeback(
                        &mut self.resident_guest_textures,
                        &alias_ranges,
                        descriptor,
                        content_version,
                    );
                    retire_multisample_textures_overlapping_writeback(
                        &mut self.multisample_textures,
                        &alias_ranges,
                    );
                    let resident = self
                        .resident_guest_textures
                        .get_mut(&descriptor)
                        .expect("collected DMA source texture exists");
                    write_resident_texture_contents_to_guest(&device, &queue, resident)?;
                    resident.gpu_written = false;
                }
            }

            if !copy.cpu_applied || source_was_gpu_written || pending_source_write {
                // SAFETY: the runtime decodes DMA only for bounded non-null guest virtual
                // addresses. The guest owns these mappings for the lifetime of the ordered GPU
                // event stream. `copy` intentionally provides memmove semantics for legal
                // overlapping packets.
                unsafe {
                    std::ptr::copy(
                        copy.source_address as *const u8,
                        copy.destination_address as *mut u8,
                        byte_count,
                    );
                }
            }
            self.record_submitted_gpu_waits_in_range(copy.destination_address, copy.byte_count);
        }

        let destination_descriptors = self
            .resident_guest_textures
            .keys()
            .copied()
            .filter(|descriptor| {
                guest_texture_byte_count(*descriptor).is_some_and(|length| {
                    guest_ranges_overlap(
                        descriptor.address,
                        length,
                        copy.destination_address,
                        copy.byte_count,
                    )
                })
            })
            .collect::<Vec<_>>();
        for descriptor in destination_descriptors {
            self.resident_guest_textures.remove(&descriptor);
        }
        self.multisample_textures.retain(|key, _| {
            guest_texture_byte_count(key.descriptor).is_none_or(|length| {
                !guest_ranges_overlap(
                    key.descriptor.address,
                    length,
                    copy.destination_address,
                    copy.byte_count,
                )
            })
        });
        retire_sampled_textures_matching(&mut self.sampled_texture_cache, |key| {
            guest_texture_byte_count(key.descriptor).is_some_and(|length| {
                guest_ranges_overlap(
                    key.descriptor.address,
                    length,
                    copy.destination_address,
                    copy.byte_count,
                )
            })
        });
        Ok(())
    }

    pub(super) fn execute_guest_memory_write(
        &mut self,
        write: &GpuMemoryWrite,
    ) -> Result<(), String> {
        if write.address == 0 || write.values.is_empty() {
            return Ok(());
        }
        if !write.address.is_multiple_of(4) {
            return Err(format!(
                "WRITE_DATA destination {:#x} is unaligned",
                write.address
            ));
        }
        validate_guest_memory_write(write)?;
        self.queue_pending_guest_memory_write(PendingGuestMemoryWrite::Write(write.clone()))
    }

    pub(super) fn execute_guest_memory_wait(
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
            eprintln!(
                "[REMu][METAL] resumed WAIT_REG_MEM from submitted producer ticket address={:#x} value={value:#x}",
                wait.address
            );
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
        // A release/write queued after the preceding Metal batch is the value the ordered GPU
        // wait will observe. When it already satisfies the comparison, the serial Metal queue
        // provides the required ordering without a CPU round trip. The actual guest-memory write
        // is published as that batch retires, before its completion sequence becomes visible.
        // A WAIT whose condition is already true is only an ordering point. Later Metal command
        // buffers are submitted to the same serial queue, so completing every earlier batch on
        // the CPU before reading the label adds no ordering that the queue does not already
        // provide. This is the overwhelmingly common Unity fence pattern (for example 0 < a
        // monotonically increasing reference) and synchronously draining the queue here turns
        // hundreds of otherwise asynchronous draws into one CPU/GPU round trip per frame.
        //
        // A queued write to this exact label takes precedence over the current memory contents:
        // the guest WAIT is ordered after that write. If no such write exists, the live value can
        // be tested immediately. If the queued value does not satisfy the comparison, retirement
        // below publishes it before the next live-memory test.
        match self.pending_guest_memory_value(wait.address, wait.is_64_bit) {
            Some(value) if wait.is_satisfied_by(value) => {
                self.guest_memory_waits.complete(key);
                if let Some(ticket_id) = ticket_id {
                    remu_gpu::retire_gpu_wait(ticket_id);
                }
                return Ok(GuestMemoryWaitStatus::Satisfied);
            }
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
        self.retire_completed_guest_batches();
        match self.pending_guest_memory_value(wait.address, wait.is_64_bit) {
            Some(value) if wait.is_satisfied_by(value) => {
                self.guest_memory_waits.complete(key);
                if let Some(ticket_id) = ticket_id {
                    remu_gpu::retire_gpu_wait(ticket_id);
                }
                return Ok(GuestMemoryWaitStatus::Satisfied);
            }
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
            self.flush_guest_command_batch(false, GuestBatchFlushReason::BlockedWait)
        {
            if let Some(ticket_id) = ticket_id {
                remu_gpu::retire_gpu_wait(ticket_id);
            }
            return Err(error);
        }
        if ticket_id.is_none() {
            match self.guest_memory_waits.observe_unsatisfied(key) {
                GuestMemoryWaitRecovery::ProducerLatched(value) => {
                    eprintln!(
                        "[REMu][METAL] resumed WAIT_REG_MEM from a real producer latch address={:#x} value={value:#x}",
                        wait.address
                    );
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
            eprintln!(
                "[REMu][METAL] suspended WAIT_REG_MEM sequence={sequence} address={:#x} value={value:#x} pending={pending_value:?} mask={:#x} reference={:#x} compare={}",
                wait.address, wait.mask, wait.reference, wait.compare_function
            );
        }
        Ok(GuestMemoryWaitStatus::Blocked)
    }

    pub(super) fn queue_pending_guest_memory_write(
        &mut self,
        write: PendingGuestMemoryWrite,
    ) -> Result<(), String> {
        // RELEASE_MEM/WRITE_DATA label updates are ordered after the Metal work already encoded
        // in this batch, but do not themselves require a new Metal submission. Publish them when
        // the owning command buffer retires. A later wait can inspect the deferred value, while
        // a genuinely blocked wait and the event-loop boundary still commit the batch.
        if let Some(batch) = self.open_guest_batch.as_mut() {
            batch.memory_writes.push(write);
            return Ok(());
        }
        if let Some(batch) = self.committed_guest_batches.back_mut() {
            batch.memory_writes.push(write);
        } else {
            apply_pending_guest_memory_writes(std::slice::from_ref(&write))?;
            self.record_produced_guest_memory_writes(std::slice::from_ref(&write));
        }
        Ok(())
    }

    pub(super) fn record_guest_memory_producer(
        &mut self,
        address: u64,
        is_64_bit: bool,
        value: u64,
    ) {
        self.guest_memory_waits
            .record_produced(address, is_64_bit, value);
        remu_gpu::record_gpu_wait_producer(address, is_64_bit, value);
    }

    pub(super) fn record_submitted_gpu_waits_in_range(&mut self, address: u64, byte_count: u64) {
        let waits = remu_gpu::pending_gpu_waits_overlapping(address, byte_count);
        for (_, wait) in waits {
            let value = read_guest_memory_wait_value(wait);
            self.record_guest_memory_producer(wait.address, wait.is_64_bit, value);
        }
    }

    pub(super) fn record_produced_global_writebacks(&mut self, pending: &[PendingGlobalWriteback]) {
        for writeback in pending {
            if let Ok(byte_count) = u64::try_from(writeback.byte_count) {
                self.record_submitted_gpu_waits_in_range(writeback.guest_address, byte_count);
            }
        }
    }

    pub(super) fn record_produced_guest_memory_writes(
        &mut self,
        pending: &[PendingGuestMemoryWrite],
    ) {
        for write in pending {
            match write {
                PendingGuestMemoryWrite::Release(release) => match release.value {
                    GpuMemoryReleaseValue::Immediate32(value) => {
                        self.record_guest_memory_producer(release.address, false, u64::from(value));
                    }
                    GpuMemoryReleaseValue::Immediate64(value) => {
                        self.record_guest_memory_producer(release.address, true, value);
                        self.record_guest_memory_producer(
                            release.address,
                            false,
                            value & u64::from(u32::MAX),
                        );
                        self.record_guest_memory_producer(
                            release.address.saturating_add(4),
                            false,
                            value >> 32,
                        );
                    }
                    GpuMemoryReleaseValue::Timestamp => {
                        let value = read_guest_memory_wait_value(GpuMemoryWait {
                            address: release.address,
                            reference: 0,
                            mask: u64::MAX,
                            compare_function: 0,
                            is_64_bit: true,
                        });
                        self.record_guest_memory_producer(release.address, true, value);
                        self.record_guest_memory_producer(
                            release.address,
                            false,
                            value & u64::from(u32::MAX),
                        );
                        self.record_guest_memory_producer(
                            release.address.saturating_add(4),
                            false,
                            value >> 32,
                        );
                    }
                },
                PendingGuestMemoryWrite::Write(write) => {
                    for (index, value) in write.values.iter().copied().enumerate() {
                        let offset = if write.increment_address {
                            u64::try_from(index).unwrap_or(u64::MAX).saturating_mul(4)
                        } else {
                            0
                        };
                        self.record_guest_memory_producer(
                            write.address.saturating_add(offset),
                            false,
                            u64::from(value),
                        );
                    }
                    if write.increment_address {
                        for (index, pair) in write.values.windows(2).enumerate() {
                            let address = write.address.saturating_add(
                                u64::try_from(index).unwrap_or(u64::MAX).saturating_mul(4),
                            );
                            if address.is_multiple_of(8) {
                                let value = u64::from(pair[0]) | (u64::from(pair[1]) << 32);
                                self.record_guest_memory_producer(address, true, value);
                            }
                        }
                    }
                }
                PendingGuestMemoryWrite::Fill { address, value, .. } => {
                    self.record_guest_memory_producer(*address, false, u64::from(*value));
                    if address.is_multiple_of(8) {
                        self.record_guest_memory_producer(
                            *address,
                            true,
                            u64::from(*value) | (u64::from(*value) << 32),
                        );
                    }
                }
            }
        }

        // Large DMA fills are not expanded into the bounded produced-label table. Active waits
        // still latch the exact value they observed at producer completion, including waits on
        // an interior dword or qword of a fill.
        let active = self
            .guest_memory_waits
            .active
            .keys()
            .copied()
            .collect::<Vec<_>>();
        for key in active {
            let width = if key.is_64_bit { 8 } else { 4 };
            if pending
                .iter()
                .any(|write| write.overlaps(key.address, width))
            {
                let value = read_guest_memory_wait_value(key.wait());
                self.record_guest_memory_producer(key.address, key.is_64_bit, value);
            }
        }
        for write in pending {
            let (address, byte_count) = write.range();
            self.record_submitted_gpu_waits_in_range(address, byte_count);
        }
    }

    pub(super) fn pending_guest_memory_value(&self, address: u64, is_64_bit: bool) -> Option<u64> {
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
impl OrderedGpuEventReceiver {
    pub(super) fn new(receiver: Receiver<GpuEvent>) -> Self {
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

    pub(super) fn ingest(&mut self, receive_limit: usize) {
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

    pub(super) fn try_recv(
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

    pub(super) fn pop_queue_event(&mut self, queue_id: u64) -> Option<GpuEvent> {
        let queue = self.submissions.get_mut(&queue_id)?;
        while queue
            .front()
            .is_some_and(|submission| submission.events.is_empty())
        {
            queue.pop_front();
        }
        queue.front_mut()?.events.pop_front()
    }

    pub(super) fn finish_event(&mut self, source: OrderedEventSource) {
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

    pub(super) fn block_event(
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

    pub(super) fn advance_after(&mut self, queue_id: u64) {
        if let Some(index) = self.queue_order.iter().position(|&id| id == queue_id) {
            self.next_queue_index = (index + 1) % self.queue_order.len().max(1);
        }
    }

    pub(super) fn retry_guest_memory_waits(&mut self) {
        self.blocked_queues
            .retain(|_, reason| !matches!(reason, QueueBlockReason::GuestMemoryWait));
        if matches!(self.host_block, Some(QueueBlockReason::GuestMemoryWait)) {
            self.host_block = None;
        }
    }

    pub(super) fn prefetch_draws(
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

    pub(super) fn pending_event_count(&self) -> usize {
        self.host_events.len().saturating_add(
            self.submissions
                .values()
                .flatten()
                .map(|submission| submission.events.len())
                .sum::<usize>(),
        )
    }

    pub(super) fn block_reason_counts(&self) -> (usize, usize) {
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

    pub(super) fn queue_diagnostics(&self) -> String {
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

pub(super) fn describe_guest_batch(work: &[GuestBatchWork]) -> String {
    work.iter()
        .map(|item| match item {
            GuestBatchWork::Draw { export, pixel } => format!("draw({export:#x},{pixel:#x})"),
            GuestBatchWork::Compute { shader } => format!("compute({shader:#x})"),
        })
        .collect::<Vec<_>>()
        .join(",")
}

pub(super) fn merge_guest_batch_result(
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

impl GuestMemoryWaitKey {
    pub(super) const fn new(source: OrderedEventSource, wait: GpuMemoryWait) -> Self {
        Self {
            source,
            address: wait.address,
            reference: wait.reference,
            mask: wait.mask,
            compare_function: wait.compare_function,
            is_64_bit: wait.is_64_bit,
        }
    }

    pub(super) const fn wait(self) -> GpuMemoryWait {
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
    pub(super) fn observe_unsatisfied(
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

    pub(super) fn complete(&mut self, key: GuestMemoryWaitKey) {
        self.active.remove(&key);
    }

    pub(super) fn record_produced(&mut self, address: u64, is_64_bit: bool, value: u64) {
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

impl PendingGuestMemoryWrite {
    pub(super) fn range(&self) -> (u64, u64) {
        match self {
            Self::Release(release) => (
                release.address,
                match release.value {
                    GpuMemoryReleaseValue::Immediate32(_) => 4,
                    GpuMemoryReleaseValue::Immediate64(_) | GpuMemoryReleaseValue::Timestamp => 8,
                },
            ),
            Self::Write(write) => (
                write.address,
                if write.increment_address {
                    u64::try_from(write.values.len())
                        .unwrap_or(u64::MAX / 4)
                        .saturating_mul(4)
                } else {
                    4
                },
            ),
            Self::Fill {
                address,
                dword_count,
                ..
            } => (*address, dword_count.saturating_mul(4)),
        }
    }

    pub(super) fn value_at(&self, address: u64, is_64_bit: bool) -> Option<u64> {
        match self {
            Self::Release(release) if release.address == address => match release.value {
                GpuMemoryReleaseValue::Immediate32(value) if !is_64_bit => Some(u64::from(value)),
                GpuMemoryReleaseValue::Immediate64(value) => Some(value),
                GpuMemoryReleaseValue::Timestamp | GpuMemoryReleaseValue::Immediate32(_) => None,
            },
            Self::Write(write) if !is_64_bit => write
                .values
                .iter()
                .copied()
                .enumerate()
                .rev()
                .find_map(|(index, value)| {
                    let offset = if write.increment_address {
                        u64::try_from(index).ok()?.checked_mul(4)?
                    } else {
                        0
                    };
                    (write.address.checked_add(offset)? == address).then_some(u64::from(value))
                }),
            Self::Fill {
                address: fill_address,
                value,
                dword_count,
            } => {
                let offset = address.checked_sub(*fill_address)?;
                if !offset.is_multiple_of(4) {
                    return None;
                }
                let index = offset / 4;
                let required_dwords = if is_64_bit { 2 } else { 1 };
                (index.checked_add(required_dwords)? <= *dword_count).then_some(if is_64_bit {
                    u64::from(*value) | (u64::from(*value) << 32)
                } else {
                    u64::from(*value)
                })
            }
            Self::Write(_) | Self::Release(_) => None,
        }
    }

    pub(super) fn overlaps(&self, address: u64, byte_count: u64) -> bool {
        let (write_address, write_byte_count) = self.range();
        guest_ranges_overlap(write_address, write_byte_count, address, byte_count)
    }
}

impl GuestCommandBatch {
    pub(super) fn upload_global_bytes(
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

    pub(super) fn end_render_pass(&mut self) -> bool {
        if let Some(pass) = self.render_pass.take() {
            pass.encoder.end_encoding();
            true
        } else {
            false
        }
    }
}

pub(super) fn guest_batch_writes_overlap(
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

pub(super) fn publish_guest_batch_completion(batch: &GuestCommandBatch) {
    for &sequence in &batch.completion_sequences {
        remu_gpu::record_gpu_submission_completed(sequence);
    }
}

pub(super) fn completed_command_buffer_gpu_duration(command_buffer: &CommandBuffer) -> Duration {
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

pub(super) fn collect_pending_global_writebacks(
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

pub(super) fn collect_pending_guest_write_ranges(
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

pub(super) fn append_unique_global_writebacks(
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

pub(super) fn append_unique_guest_write_ranges(
    destination: &mut Vec<PendingGuestWriteRange>,
    source: Vec<PendingGuestWriteRange>,
) {
    for range in source {
        if !destination.contains(&range) {
            destination.push(range);
        }
    }
}

pub(super) fn write_back_pending_globals(pending: &[PendingGlobalWriteback]) -> Result<(), String> {
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

pub(super) fn read_guest_memory_wait_value(wait: GpuMemoryWait) -> u64 {
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

pub(super) fn validate_guest_memory_release(release: GpuMemoryRelease) -> Result<(), String> {
    let alignment = match release.value {
        GpuMemoryReleaseValue::Immediate32(_) => 4,
        GpuMemoryReleaseValue::Immediate64(_) | GpuMemoryReleaseValue::Timestamp => 8,
    };
    if release.address.is_multiple_of(alignment) {
        Ok(())
    } else {
        Err(format!(
            "{}-bit release destination {:#x} is unaligned",
            alignment * 8,
            release.address
        ))
    }
}

pub(super) fn validate_guest_memory_write(write: &GpuMemoryWrite) -> Result<(), String> {
    if !write.address.is_multiple_of(4) {
        return Err(format!(
            "WRITE_DATA destination {:#x} is unaligned",
            write.address
        ));
    }
    if write.increment_address && !write.values.is_empty() {
        let final_index = u64::try_from(write.values.len() - 1)
            .map_err(|_| "WRITE_DATA value count does not fit u64")?;
        write
            .address
            .checked_add(
                final_index
                    .checked_mul(4)
                    .ok_or("WRITE_DATA destination offset overflowed")?,
            )
            .ok_or_else(|| "WRITE_DATA destination range overflowed".to_owned())?;
    }
    Ok(())
}

pub(super) fn apply_pending_guest_memory_writes(
    pending: &[PendingGuestMemoryWrite],
) -> Result<(), String> {
    static GPU_CLOCK_START: OnceLock<Instant> = OnceLock::new();
    for write in pending {
        match write {
            PendingGuestMemoryWrite::Release(release) => {
                validate_guest_memory_release(*release)?;
                // SAFETY: packet decoding admits only aligned, live process-local guest
                // destinations. The owning Metal batch has completed, and volatile writes publish
                // the ordered label update before submission completion becomes visible.
                unsafe {
                    match release.value {
                        GpuMemoryReleaseValue::Immediate32(value) => {
                            std::ptr::write_volatile(release.address as *mut u32, value);
                        }
                        GpuMemoryReleaseValue::Immediate64(value) => {
                            std::ptr::write_volatile(release.address as *mut u64, value);
                        }
                        GpuMemoryReleaseValue::Timestamp => {
                            let timestamp = u64::try_from(
                                GPU_CLOCK_START
                                    .get_or_init(Instant::now)
                                    .elapsed()
                                    .as_nanos(),
                            )
                            .unwrap_or(u64::MAX)
                            .max(1);
                            std::ptr::write_volatile(release.address as *mut u64, timestamp);
                        }
                    }
                }
            }
            PendingGuestMemoryWrite::Write(write) => {
                validate_guest_memory_write(write)?;
                // SAFETY: validation above proves aligned, non-overflowing destinations in the
                // guest mapping supplied by the runtime.
                unsafe {
                    for (index, value) in write.values.iter().copied().enumerate() {
                        let offset = if write.increment_address {
                            u64::try_from(index)
                                .map_err(|_| "WRITE_DATA index does not fit u64")?
                                .checked_mul(4)
                                .ok_or("WRITE_DATA destination offset overflowed")?
                        } else {
                            0
                        };
                        let address = write
                            .address
                            .checked_add(offset)
                            .ok_or("WRITE_DATA destination range overflowed")?;
                        std::ptr::write_volatile(address as *mut u32, value);
                    }
                }
            }
            PendingGuestMemoryWrite::Fill {
                address,
                value,
                dword_count,
            } => {
                let byte_count = dword_count
                    .checked_mul(4)
                    .ok_or("DMA fill byte count overflowed")?;
                if !address.is_multiple_of(4) || address.checked_add(byte_count).is_none() {
                    return Err(format!(
                        "DMA fill destination {address:#x} dwords={dword_count:#x} is invalid"
                    ));
                }
                // SAFETY: packet decoding admitted an aligned, non-overflowing guest range. The
                // owning Metal batch has retired, so publishing the repeated dwords here preserves
                // DMA ordering for following waits and CPU readers.
                unsafe {
                    for index in 0..*dword_count {
                        let offset = index
                            .checked_mul(4)
                            .ok_or("DMA fill destination offset overflowed")?;
                        std::ptr::write_volatile((*address + offset) as *mut u32, *value);
                    }
                }
            }
        }
    }
    Ok(())
}
