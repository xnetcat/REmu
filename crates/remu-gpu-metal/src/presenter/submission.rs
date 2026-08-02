// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    ActiveGuestMemoryWait, DisplayConfiguration, FlipTerminalSource, FlushPolicy,
    GLOBAL_UPLOAD_ALIGNMENT, GLOBAL_UPLOAD_CHUNK_BYTE_COUNT, GlobalUploadChunk,
    GuestBatchFlushReason, GuestBatchWork, GuestCommandBatch, GuestMemoryWaitKey,
    GuestMemoryWaitRecovery, GuestMemoryWaitStatus, GuestMemoryWaitTracker,
    GuestMemoryWriteOperation, MetalPresenter, OrderedEventPoll, OrderedEventSource,
    OrderedGpuEventReceiver, PIPELINE_COMPUTE_GATES, PIPELINE_DRAW_GATES, PendingFlip,
    PendingGlobalWriteback, PendingGuestMemoryWrite, PendingGuestWriteRange,
    PipelineCompileCandidate, PresenterFlow, QueueBlockReason, RangeSet, ResourceSpan,
    ScheduledGpuSubmission, ScheduledHostEvent, UploadedGlobalBuffer, WriteWatchRefresh,
    compute_pipeline_key, diagnostic_skip_export_shader, diagnostic_skip_pixel_shader,
    graphics_pipeline_key, guest_physical_ranges_overlap, guest_ranges_overlap,
    guest_texture_byte_count, noop_placeholder_cover_draw, pixel_scalars,
    rearm_color_metadata_clear, rearm_depth_metadata_clear, record_diagnostic_memory_copy,
    refresh_resident_texture_write_watch, register_value, retire_sampled_textures_matching,
    trace_resident_texture_removal, trace_video_target_compute_consumer,
    trace_video_target_draw_consumer,
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
pub(crate) use self::memory::{
    apply_pending_guest_memory_writes, pin_guest_memory_write, release_value_for_wait,
};
pub(crate) use self::waits::read_guest_memory_wait_value;

impl MetalPresenter {
    #[allow(clippy::too_many_lines)]
    pub(super) fn receive_events(&mut self) -> PresenterFlow {
        self.receive_shader_compilations();
        self.retire_completed_guest_batches();
        self.retire_completed_flip_snapshots();
        self.retire_completed_presentations();
        // A guest label may have been published by a Metal batch that retired since the previous
        // AppKit tick. Before clearing the block graph, bounded dependency ingress discovers any
        // cross-queue producer waiting in later channel chunks. Then retry suspended queue fronts
        // once; an unsatisfied wait is re-suspended without permitting later work to overtake it.
        self.receiver.retry_guest_memory_waits();
        // Do not commit more Metal batches while the in-flight budget is exhausted. Keep draining
        // decoded guest work into the open batch so queue depth (Q W) does not grow just because
        // the GPU is still executing earlier submissions.
        if self.guest_submission_backpressure_full() {
            return PresenterFlow::Continue;
        }
        // Feed the existing compile workers from bounded ordered lookahead even while the current
        // queue front is runnable. Waiting until a cold draw or compute dispatch becomes the
        // blocked front serialized multi-second first-scene Metal compilation behind the
        // warning/movie UI. This only snapshots immutable work descriptions and schedules
        // deduplicated pipeline keys; event execution and every queue-local dependency remain in
        // their original order.
        self.prefetch_pipeline_compilations();
        // A bounded batch prevents a continuously producing guest from starving redraw and input
        // delivery. Queue-local shader and memory waits leave independent guest queues runnable.
        for _ in 0..self.gpu_events_per_tick_budget() {
            // A bounded drain can still spend several seconds preparing hundreds of large shaders.
            // Compiler workers publish into an independent result channel during that time.
            // Refresh before every readiness decision so a pipeline that completed during this
            // callback immediately releases its queue-local gate instead of remaining falsely
            // "active" until the next AppKit callback.
            self.receive_shader_compilations();
            let shared_event_values = self
                .guest_queue_batches
                .iter()
                .map(|(&queue_id, state)| {
                    (
                        super::SharedEventProducerId::new(queue_id),
                        state.shared_event_timeline.event().signaled_value(),
                    )
                })
                .collect::<std::collections::HashMap<_, _>>();
            let guest_queue_outstanding = self
                .guest_queue_batches
                .iter()
                .map(|(&queue_id, state)| {
                    (
                        queue_id,
                        state.open_batch.is_some() || !state.committed_batches.is_empty(),
                    )
                })
                .collect::<std::collections::HashMap<_, _>>();
            let host_retired_values = self
                .guest_queue_batches
                .iter()
                .map(|(&queue_id, state)| {
                    (
                        super::SharedEventProducerId::new(queue_id),
                        state.host_retired_fence_value,
                    )
                })
                .collect::<std::collections::HashMap<_, _>>();
            let (source, event) = match self.receiver.try_recv(|reason| match reason {
                QueueBlockReason::GraphicsPipeline(key) => {
                    self.compiled_shader_pairs.contains_key(key)
                        || self.failed_shader_pairs.contains(key)
                }
                QueueBlockReason::ComputePipeline(key) => {
                    self.compiled_compute_pipelines.contains_key(key)
                        || self.failed_compute_pipelines.contains(key)
                }
                QueueBlockReason::GuestMemoryWait(_)
                | QueueBlockReason::GuestMemoryWaitPendingTerminal(_) => false,
                QueueBlockReason::GuestSharedEventHostEffect {
                    consumer_queue_id,
                    consumer_ack,
                    terminal_dependencies,
                } => {
                    let consumer_reached_wait = consumer_ack.as_ref().map_or_else(
                        || {
                            !guest_queue_outstanding
                                .get(consumer_queue_id)
                                .copied()
                                .unwrap_or(false)
                        },
                        |fence| {
                            shared_event_values
                                .get(&fence.producer())
                                .is_some_and(|value| *value >= fence.value())
                        },
                    );
                    consumer_reached_wait
                        && terminal_dependencies.iter().all(|fence| {
                            host_retired_values
                                .get(&fence.producer())
                                .is_some_and(|value| *value >= fence.value())
                        })
                }
                QueueBlockReason::FlipSafetyWait(wait) => {
                    wait.required_flip_sequence.is_none_or(|sequence| {
                        sequence <= self.safe_flip_sequence_prefix
                            || self
                                .flip_buffer_safety
                                .get(&sequence)
                                .is_some_and(|safety| safety.safe)
                    })
                }
            }) {
                OrderedEventPoll::Event { source, event } => (source, event),
                OrderedEventPoll::Blocked => {
                    // Look ahead far enough to feed all compiler workers while memory-wait queues
                    // remain suspended. Prefetching never executes or reorders an event.
                    self.prefetch_pipeline_compilations();
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
            self.set_active_guest_queue(match source {
                OrderedEventSource::GuestQueue(queue_id) => Some(queue_id),
                OrderedEventSource::Host { .. } => None,
            });
            let mut shared_event_gpu_ordered_compute = false;
            if let OrderedEventSource::GuestQueue(queue_id) = source {
                let dependencies = self.unsignaled_host_dependencies(queue_id);
                let gpu_orderable = dependencies.is_empty()
                    || self.event_is_shared_event_gpu_orderable(source, &event);
                if !gpu_orderable {
                    let acknowledgement = match self.encode_active_guest_queue_host_ack() {
                        Ok(acknowledgement) => acknowledgement,
                        Err(error) => {
                            eprintln!(
                                "[REMu][METAL] shared-event host acknowledgement failed queue={queue_id}: {error}"
                            );
                            None
                        }
                    };
                    if let Err(error) = self.flush_guest_command_batch(
                        FlushPolicy::Required,
                        GuestBatchFlushReason::SharedEventHostEffect,
                    ) {
                        eprintln!(
                            "[REMu][METAL] shared-event host-effect flush failed queue={queue_id}: {error}"
                        );
                    }
                    self.statistics.shared_event_host_blocks =
                        self.statistics.shared_event_host_blocks.saturating_add(1);
                    self.receiver.block_event(
                        source,
                        event,
                        QueueBlockReason::GuestSharedEventHostEffect {
                            consumer_queue_id: queue_id,
                            consumer_ack: acknowledgement,
                            terminal_dependencies: dependencies,
                        },
                    );
                    continue;
                }
                if !dependencies.is_empty() && matches!(&event, GpuEvent::ComputeDispatch(_)) {
                    shared_event_gpu_ordered_compute = true;
                }
            }
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
                    buffers,
                } => {
                    self.display = Some(DisplayConfiguration {
                        width,
                        height,
                        pitch,
                        pixel_format,
                        tiling_mode,
                    });
                    self.registered_video_out_buffers = buffers;
                    self.frame_texture = None;
                    self.active_translated_scanout = None;
                    eprintln!(
                        "[REMu][METAL] display registered {width}x{height} pitch={pitch} format={pixel_format:#x} tiling={tiling_mode} buffers={:x?}",
                        self.registered_video_out_buffers
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
                GpuEvent::ShowHostDialog(request) => {
                    crate::presenter::host_ui::present_host_dialog(&request);
                    self.redraw_pending = true;
                }
                GpuEvent::HideHostDialog { sequence } => {
                    crate::presenter::host_ui::hide_host_dialog(sequence);
                    self.redraw_pending = true;
                }
                GpuEvent::ShowHostToast(request) => {
                    crate::presenter::host_ui::present_host_toast(&request);
                }
                GpuEvent::Flip {
                    address,
                    buffer_index,
                    flip_argument,
                    submission_sequence,
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
                    // A flip the guest replaces before presentation is never shown, but it was
                    // counted against flip pacing on acceptance and the guest still holds it in its
                    // own pending queue, so it still owes exactly one retirement. Publishing that
                    // retirement HERE — at event-dequeue time — told Unreal the frame was done
                    // while its ordered work was still queued or executing, and Unreal returned
                    // the frame's transient memory to MallocBinned3 with its own render thread
                    // still writing through pointers into it. Defer the completion behind the
                    // ordered work exactly as SubmissionBoundary defers empty submissions.
                    if let Some(superseded) = self.pending_flip {
                        self.release_flip_buffer_without_snapshot(
                            superseded.sequence,
                            "superseded-before-snapshot",
                        );
                        if let Some(sequence) = self.claim_flip_terminal(superseded.sequence) {
                            self.mark_flip_terminal(sequence, FlipTerminalSource::Superseded);
                        }
                    }
                    self.next_flip_sequence = self.next_flip_sequence.saturating_add(1);
                    let flip = PendingFlip {
                        address,
                        buffer_index,
                        flip_argument,
                        submission_sequence,
                        sequence: self.next_flip_sequence,
                    };
                    self.register_flip_completion_join(flip);
                    self.pending_flip = Some(flip);
                    self.pending_flip_target_written_since_boundary = false;
                    self.pending_scanout_write_address = None;
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
                GpuEvent::CommandProcessorBarrier => {
                    if let Err(error) = self.end_guest_command_processor_barrier() {
                        eprintln!(
                            "[REMu][METAL] PFP_SYNC_ME command-processor barrier failed: {error}"
                        );
                    }
                }
                GpuEvent::WaitFlipDone(wait) => {
                    let (wait, safe) = self.bind_flip_safety_wait(wait);
                    static WAIT_FLIP_DONE_COUNT: AtomicUsize = AtomicUsize::new(0);
                    let occurrence = WAIT_FLIP_DONE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                    if safe {
                        if occurrence <= 32 || occurrence.is_power_of_two() {
                            eprintln!(
                                "[REMu][WAIT-FLIP-DONE] phase=pass occurrence={occurrence} address={:#x} index={} required={:?}",
                                wait.address, wait.buffer_index, wait.required_flip_sequence,
                            );
                        }
                    } else {
                        if occurrence <= 32 || occurrence.is_power_of_two() {
                            eprintln!(
                                "[REMu][WAIT-FLIP-DONE] phase=block occurrence={occurrence} address={:#x} index={} required={:?}",
                                wait.address, wait.buffer_index, wait.required_flip_sequence,
                            );
                        }
                        self.receiver.block_event(
                            source,
                            GpuEvent::WaitFlipDone(wait),
                            QueueBlockReason::FlipSafetyWait(wait),
                        );
                        continue;
                    }
                }
                GpuEvent::PipelineBarrier(barrier) => self.end_guest_pipeline_barrier(barrier),
                GpuEvent::MemoryAcquire(acquire) => {
                    if let Err(error) = self.synchronize_cpu_written_guest_resources(acquire) {
                        eprintln!("[REMu][METAL] ACQUIRE_MEM synchronization failed: {error}");
                    }
                }
                GpuEvent::MemoryCopy(copy) => {
                    let destination_address = copy.destination_address;
                    let byte_count = copy.byte_count;
                    match self.execute_guest_memory_copy(copy) {
                        Ok(()) => self
                            .receiver
                            .retry_guest_memory_waits_overlapping(destination_address, byte_count),
                        Err(error) => {
                            eprintln!("[REMu][METAL] DMA_DATA synchronization failed: {error}");
                        }
                    }
                }
                GpuEvent::MemoryWrite(write) => match self.execute_guest_memory_write(&write) {
                    Ok(()) => {
                        let byte_count = if write.increment_address {
                            u64::try_from(write.values.len())
                                .unwrap_or(u64::MAX)
                                .saturating_mul(4)
                        } else if write.values.is_empty() {
                            0
                        } else {
                            4
                        };
                        self.receiver
                            .retry_guest_memory_waits_overlapping(write.address, byte_count);
                    }
                    Err(error) => {
                        eprintln!("[REMu][METAL] WRITE_DATA synchronization failed: {error}");
                    }
                },
                GpuEvent::MemoryRelease(release) => {
                    match self.execute_guest_memory_release(source, release) {
                        Ok(()) => {
                            let byte_count = match release.value {
                                remu_gpu::GpuMemoryReleaseValue::Immediate32(_) => 4,
                                remu_gpu::GpuMemoryReleaseValue::Immediate64(_)
                                | remu_gpu::GpuMemoryReleaseValue::Timestamp => 8,
                            };
                            self.receiver
                                .retry_guest_memory_waits_overlapping(release.address, byte_count);
                        }
                        Err(error) => {
                            eprintln!("[REMu][METAL] RELEASE_MEM synchronization failed: {error}");
                        }
                    }
                }
                GpuEvent::MemoryWait(wait) => {
                    match self.execute_guest_memory_wait(source, None, wait) {
                        Ok(GuestMemoryWaitStatus::Satisfied) => {}
                        Ok(GuestMemoryWaitStatus::Blocked) => {
                            self.receiver.block_event(
                                source,
                                GpuEvent::MemoryWait(wait),
                                QueueBlockReason::GuestMemoryWait(wait),
                            );
                            continue;
                        }
                        Ok(GuestMemoryWaitStatus::BlockedOnPendingTerminal) => {
                            self.receiver.block_event(
                                source,
                                GpuEvent::MemoryWait(wait),
                                QueueBlockReason::GuestMemoryWaitPendingTerminal(wait),
                            );
                            continue;
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
                                QueueBlockReason::GuestMemoryWait(wait),
                            );
                            continue;
                        }
                        Ok(GuestMemoryWaitStatus::BlockedOnPendingTerminal) => {
                            self.receiver.block_event(
                                source,
                                GpuEvent::RegisteredMemoryWait { id, wait },
                                QueueBlockReason::GuestMemoryWaitPendingTerminal(wait),
                            );
                            continue;
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
                    if let (Some(snapshot), Some(evaluation)) = (
                        dispatch.shader.as_ref(),
                        dispatch.shader_evaluation.as_ref(),
                    ) {
                        let key = compute_pipeline_key(&dispatch, evaluation, &snapshot.program);
                        self.compile_compute_shader(&dispatch, &key);
                        if !self.compiled_compute_pipelines.contains_key(&key)
                            && !self.failed_compute_pipelines.contains(&key)
                        {
                            PIPELINE_COMPUTE_GATES.fetch_add(1, Ordering::Relaxed);
                            static LOGGED_COMPUTE_PIPELINE_GATES: AtomicUsize = AtomicUsize::new(0);
                            if LOGGED_COMPUTE_PIPELINE_GATES.fetch_add(1, Ordering::Relaxed) < 16 {
                                eprintln!(
                                    "[REMu][METAL] preserving cold compute={} cs={:#x} while its pipeline compiles",
                                    dispatch.sequence, dispatch.shader_address,
                                );
                            }
                            self.receiver.block_event(
                                source,
                                GpuEvent::ComputeDispatch(dispatch),
                                QueueBlockReason::ComputePipeline(key),
                            );
                            continue;
                        }
                    }
                    match self.execute_compute_dispatch(&dispatch) {
                        Ok(()) => {
                            self.statistics.encoded_compute_dispatches =
                                self.statistics.encoded_compute_dispatches.saturating_add(1);
                            if shared_event_gpu_ordered_compute {
                                self.statistics.shared_event_gpu_ordered_compute_dispatches = self
                                    .statistics
                                    .shared_event_gpu_ordered_compute_dispatches
                                    .saturating_add(1);
                            }
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
                    if noop_placeholder_cover_draw(&draw) {
                        super::stats::NOOP_DRAWS_COVER_PLACEHOLDER.fetch_add(1, Ordering::Relaxed);
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
                            continue;
                        }
                    }
                    if diagnostic_skip_pixel_shader(draw.pixel_shader_address)
                        || diagnostic_skip_export_shader(draw.export_shader_address)
                    {
                        static LOGGED_SKIPPED_PS: AtomicUsize = AtomicUsize::new(0);
                        if LOGGED_SKIPPED_PS.fetch_add(1, Ordering::Relaxed) < 8 {
                            eprintln!(
                                "[REMu][METAL] skipping draw={} es={:#x} ps={:#x} (REMU_SKIP_PS/ES)",
                                draw.sequence,
                                draw.export_shader_address,
                                draw.pixel_shader_address,
                            );
                        }
                        continue;
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
                    self.record_guest_submission_boundary(source, sequence);
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
            if self.guest_submission_backpressure_full() {
                return PresenterFlow::Continue;
            }
        }
        PresenterFlow::Continue
    }

    pub(super) fn end_guest_pipeline_barrier(&mut self, _barrier: remu_gpu::GpuPipelineBarrier) {
        let Some(batch) = self.active_open_guest_batch_mut() else {
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

    fn end_guest_command_processor_barrier(&mut self) -> Result<(), String> {
        // Metal command encoders execute serially, but PFP_SYNC_ME has a stronger queue-visible
        // role than an EVENT_WRITE partial flush: parser-side indirect fetches must wait for every
        // preceding ME producer. Commit the preceding work without waiting on the CPU. Metal
        // command queues preserve command-buffer order, while later indirect bindings can still
        // forward the exact pending writer buffer across the committed-batch boundary.
        self.flush_guest_command_batch(
            FlushPolicy::Required,
            GuestBatchFlushReason::CommandProcessorBarrier,
        )
    }
}

impl MetalPresenter {
    /// Conservative host-effect gate for commands following an encoded Metal wait.
    pub(super) fn event_is_shared_event_gpu_orderable(
        &self,
        source: OrderedEventSource,
        event: &GpuEvent,
    ) -> bool {
        match event {
            GpuEvent::CommandProcessorBarrier | GpuEvent::PipelineBarrier(_) => true,
            GpuEvent::ComputeDispatch(dispatch) => {
                matches!(dispatch.source, remu_gpu::ComputeDispatchSource::Direct)
                    && dispatch.shader.as_ref().is_some_and(|snapshot| {
                        dispatch
                            .shader_evaluation
                            .as_ref()
                            .is_some_and(|evaluation| {
                                compute_program_has_only_covered_readonly_buffers(
                                    snapshot, evaluation,
                                ) && self
                                    .shared_event_dependencies_supply_compute_globals(evaluation)
                            })
                    })
            }
            GpuEvent::MemoryWrite(_) | GpuEvent::MemoryRelease(_) => {
                self.active_open_guest_batch().is_some()
                    || !self.active_committed_guest_batches().is_empty()
            }
            GpuEvent::SubmissionBoundary(_) => {
                let initial_flip_boundary = matches!(
                    (source, self.pending_flip_queue),
                    (OrderedEventSource::GuestQueue(queue_id), Some(pending_queue))
                        if queue_id == pending_queue
                );
                let write_completes_pending_flip = self.pending_flip_target_written_since_boundary
                    && self.pending_flip.is_some_and(|flip| {
                        self.pending_scanout_write_address == Some(flip.address)
                            && flip.sequence != self.uploaded_flip_sequence
                    });
                !initial_flip_boundary && !write_completes_pending_flip
            }
            // Draw and image-backed/indirect/writable compute preparation snapshots guest bytes
            // and mutates shared residency/cache state on the presenter thread. DMA, acquire,
            // waits, flips and host UI have still more direct CPU-visible effects. They must
            // observe the producer event first.
            _ => false,
        }
    }
}

fn compute_program_has_only_covered_readonly_buffers(
    snapshot: &remu_gpu::ShaderSnapshot,
    evaluation: &remu_shader::ShaderEvaluation,
) -> bool {
    if !snapshot.supplemental_programs.is_empty()
        || evaluation.global_memory_bindings.is_empty()
        || !evaluation.image_bindings.is_empty()
    {
        return false;
    }
    snapshot
        .program
        .instructions
        .iter()
        .all(|instruction| match instruction.control {
            Some(remu_shader::InstructionControl::Image(_)) => false,
            Some(
                remu_shader::InstructionControl::GlobalMemory(_)
                | remu_shader::InstructionControl::BufferMemory(_)
                | remu_shader::InstructionControl::ScalarMemory(_),
            ) => {
                !instruction.opcode.contains("Store")
                    && !instruction.opcode.contains("Atomic")
                    && evaluation
                        .global_memory_bindings
                        .iter()
                        .any(|binding| binding.instruction_pcs.contains(&instruction.pc))
            }
            _ => true,
        })
}
