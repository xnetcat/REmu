// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    ActiveGuestMemoryWait, DisplayConfiguration, GLOBAL_UPLOAD_ALIGNMENT,
    GLOBAL_UPLOAD_CHUNK_BYTE_COUNT, GPU_EVENTS_PER_TICK, GlobalUploadChunk, GraphicsPipelineKey,
    GuestBatchFlushReason, GuestBatchWork, GuestCommandBatch, GuestMemoryWaitKey,
    GuestMemoryWaitRecovery, GuestMemoryWaitStatus, GuestMemoryWaitTracker,
    MAX_COMMITTED_GUEST_BATCHES, MetalPresenter, OrderedEventPoll, OrderedEventSource,
    OrderedGpuEventReceiver, PIPELINE_DRAW_GATES, PendingFlip, PendingGlobalWriteback,
    PendingGuestMemoryWrite, PendingGuestWriteRange, PresenterFlow, QueueBlockReason,
    ScheduledGpuSubmission, ScheduledHostEvent, UploadedGlobalBuffer, WriteWatchRefresh,
    graphics_pipeline_key, guest_physical_ranges_overlap, guest_ranges_overlap,
    guest_texture_byte_count, pixel_scalars, rearm_depth_metadata_clear,
    record_diagnostic_memory_copy, refresh_resident_texture_write_watch, register_value,
    retire_multisample_textures_overlapping_writeback,
    retire_resident_textures_overlapping_writeback, retire_sampled_textures_matching,
    retire_sampled_textures_overlapping_writeback, skip_incomplete_rectangle_draw,
    trace_resident_texture_removal, trace_video_target_compute_consumer,
    trace_video_target_draw_consumer, write_resident_texture_contents_to_guest,
};
use metal::{MTLRenderStages, NSUInteger};
use objc::{msg_send, sel, sel_impl};
use remu_gpu::GpuEvent;
use std::sync::atomic::{AtomicUsize, Ordering};
mod batch;
mod events;
mod memory;
mod waits;
#[cfg(test)]
pub(crate) use self::batch::write_back_pending_globals;
pub(crate) use self::batch::{
    append_unique_global_writebacks, append_unique_guest_write_ranges,
    collect_pending_global_writebacks, collect_pending_guest_write_ranges,
    completed_command_buffer_gpu_duration, describe_guest_batch, guest_batch_writes_overlap,
    merge_guest_batch_result,
};
#[cfg(test)]
pub(crate) use self::memory::apply_pending_guest_memory_writes;
pub(crate) use self::waits::read_guest_memory_wait_value;

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
        // Do not pull more guest commands while Metal already owns the bounded in-flight budget.
        // The event loop remains responsive and retries after every retirement tick, while the
        // channel's existing capacity propagates backpressure to the guest producer.
        if self.committed_guest_batches.len() >= MAX_COMMITTED_GUEST_BATCHES {
            return PresenterFlow::Continue;
        }
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
                        Err(error) => {
                            // Every dropped dispatch is counted so a pass failing each frame
                            // stays visible in the perf trail; the message logs once per key.
                            *self
                                .dropped_compute_executions
                                .entry(dispatch.shader_address)
                                .or_default() += 1;
                            if self
                                .failed_compute_execution
                                .insert(dispatch.shader_address)
                            {
                                eprintln!(
                                    "[REMu][METAL] Gen5 compute cs={:#x} could not execute: {error}",
                                    dispatch.shader_address
                                );
                            }
                        }
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
                        // Every dropped draw is counted so a scene pass failing each frame stays
                        // visible in the perf trail; the message still logs once per key.
                        *self
                            .dropped_draw_executions
                            .entry((draw.export_shader_address, draw.pixel_shader_address))
                            .or_default() += 1;
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
            if self.committed_guest_batches.len() >= MAX_COMMITTED_GUEST_BATCHES {
                return PresenterFlow::Continue;
            }
        }
        PresenterFlow::Continue
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
}
