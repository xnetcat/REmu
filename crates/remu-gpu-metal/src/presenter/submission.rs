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
    PhysicalFootprintRange, PipelineCompileCandidate, PresenterFlow, QueueBlockReason, RangeSet,
    ResourceSpan, SHARED_HOST_KIND_ACQUIRE, SHARED_HOST_KIND_COMPUTE, SHARED_HOST_KIND_DMA,
    SHARED_HOST_KIND_DRAW, SHARED_HOST_KIND_FLIP, SHARED_HOST_KIND_OTHER, SHARED_HOST_KIND_UI,
    SHARED_HOST_KIND_WAIT, ScheduledGpuSubmission, ScheduledHostEvent,
    SharedEventDirectComputePlan, SharedEventHostEffectLatencyPhase,
    SharedEventHostEffectLatencyState, SharedEventHostEffectPendingStatistics,
    SharedEventWriteKind, UploadedGlobalBuffer, WriteWatchRefresh, compute_pipeline_key,
    diagnostic_skip_export_shader, diagnostic_skip_pixel_shader, graphics_pipeline_key,
    guest_physical_ranges_overlap, guest_ranges_overlap, guest_texture_byte_count,
    noop_placeholder_cover_draw, pixel_scalars, rearm_color_metadata_clear,
    rearm_depth_metadata_clear, record_diagnostic_memory_copy,
    refresh_resident_texture_write_watch, register_value, retire_sampled_textures_matching,
    trace_resident_texture_removal, trace_video_target_compute_consumer,
    trace_video_target_draw_consumer,
};
use metal::{MTLRenderStages, NSUInteger};
use objc::{msg_send, sel, sel_impl};
use remu_gpu::{GpuEvent, GpuMemoryAcquire};
use remu_memory::guest_shared_backing_identity;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
mod batch;
mod draw_footprint;
mod events;
mod memory;
mod waits;
#[cfg(test)]
pub(crate) use self::batch::write_back_pending_globals;
pub(crate) use self::batch::{
    append_unique_global_writebacks, append_unique_guest_write_ranges,
    collect_pending_global_writebacks, collect_pending_guest_write_ranges,
    collect_shared_event_global_versions, completed_command_buffer_gpu_duration,
    describe_guest_batch, guest_batch_writes_overlap, merge_guest_batch_result,
    slow_command_buffer_threshold,
};
pub(crate) use self::draw_footprint::{
    SHARED_EVENT_DRAW_REJECT_COUNT, SharedEventDrawReject, program_memory_is_completely_covered,
};
pub(crate) use self::memory::{SAME_QUEUE_DMA_FILL_REJECT_COUNT, SameQueueDmaFillReject};
#[cfg(test)]
pub(crate) use self::memory::{
    apply_pending_guest_memory_writes, pin_guest_memory_write, release_value_for_wait,
};
pub(crate) use self::waits::read_guest_memory_wait_value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SharedEventEventDisposition {
    GpuOrderable,
    DropSemanticNoOp,
    HostEffect,
}

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
            // Read only the exact queue/fences named by a blocked host effect. Building three
            // complete queue snapshots here made every dequeued GPU event allocate and poll every
            // Metal SharedEvent, even when no SharedEvent host block existed. Queue state cannot
            // mutate while `try_recv` runs on the presenter thread; native event values are
            // monotonic, so each direct lookup preserves the same threshold proof without the
            // command-stream-frequency full-map scan.
            let guest_queue_batches = &self.guest_queue_batches;
            let shared_event_host_effect_latency =
                &mut self.statistics.shared_event_host_effect_latency;
            let mut observed_consumer_acks = Vec::new();
            let poll = self.receiver.try_recv(|reason| match reason {
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
                    latency,
                } => {
                    let now = Instant::now();
                    let waiting_for_ack =
                        latency.phase == SharedEventHostEffectLatencyPhase::ConsumerAck;
                    let resolved = shared_event_host_effect_is_resolved(
                        latency,
                        now,
                        shared_event_host_effect_latency,
                        *consumer_queue_id,
                        *consumer_ack,
                        terminal_dependencies,
                        |queue_id| {
                            guest_queue_batches.get(&queue_id).is_some_and(|state| {
                                state.open_batch.is_some() || !state.committed_batches.is_empty()
                            })
                        },
                        |producer| {
                            guest_queue_batches
                                .get(&producer.queue_id())
                                .map(|state| state.shared_event_timeline.event().signaled_value())
                        },
                        |producer| {
                            guest_queue_batches
                                .get(&producer.queue_id())
                                .map(|state| state.shared_event_timeline.event().signaled_value())
                        },
                        |producer| {
                            guest_queue_batches
                                .get(&producer.queue_id())
                                .map(|state| state.host_retired_fence_value)
                        },
                    );
                    if waiting_for_ack
                        && latency.phase != SharedEventHostEffectLatencyPhase::ConsumerAck
                        && let Some(fence) = *consumer_ack
                    {
                        observed_consumer_acks.push((fence, now));
                    }
                    resolved
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
            });
            for (fence, observed_at) in observed_consumer_acks {
                self.record_shared_event_ack_observed(fence, observed_at);
            }
            let (source, event) = match poll {
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
            let event_processing_started = Instant::now();
            let event_queue_wait = self.receiver.last_event_queue_wait();
            self.set_active_guest_queue(match source {
                OrderedEventSource::GuestQueue(queue_id) => Some(queue_id),
                OrderedEventSource::Host { .. } => None,
            });
            let mut shared_event_gpu_ordered_compute = false;
            let mut shared_event_gpu_ordered_registered_wait = false;
            let mut shared_event_gpu_ordered_draw = false;
            let mut shared_event_direct_compute_plan = None;
            let mut shared_event_direct_draw_plan = None;
            let mut prepared_same_queue_dma_fill = None;
            if let OrderedEventSource::GuestQueue(queue_id) = source {
                let dependencies = self.unsignaled_host_dependencies(queue_id);
                shared_event_direct_draw_plan = if dependencies.is_empty() {
                    None
                } else if let GpuEvent::IndexedDraw(draw) = &event {
                    self.observe_shared_event_direct_draw_preflight(queue_id, &dependencies, draw)
                } else {
                    None
                };
                shared_event_direct_compute_plan = if dependencies.is_empty() {
                    None
                } else if let GpuEvent::ComputeDispatch(dispatch) = &event {
                    self.prepare_shared_event_direct_compute_plan(queue_id, &dependencies, dispatch)
                } else {
                    None
                };
                let direct_draw_gpu_orderable = shared_event_direct_draw_plan.is_some()
                    && self.per_queue_guest_batches
                    && self.shared_event_direct_draw_admission;
                let same_queue_dma_fill = if dependencies.is_empty() {
                    None
                } else if let GpuEvent::MemoryCopy(copy) = &event {
                    self.prepare_same_queue_immediate_fill(queue_id, &dependencies, copy)
                } else {
                    None
                };
                let disposition = if dependencies.is_empty()
                    || direct_draw_gpu_orderable
                    || shared_event_direct_compute_plan.is_some()
                    || same_queue_dma_fill.is_some()
                {
                    SharedEventEventDisposition::GpuOrderable
                } else {
                    self.shared_event_event_disposition(queue_id, source, &dependencies, &event)
                };
                if disposition == SharedEventEventDisposition::DropSemanticNoOp {
                    self.statistics.shared_event_acquire_noop_drops = self
                        .statistics
                        .shared_event_acquire_noop_drops
                        .saturating_add(1);
                    static STABLE_ACQUIRE_NOOP_COUNT: AtomicUsize = AtomicUsize::new(0);
                    let occurrence = STABLE_ACQUIRE_NOOP_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                    if occurrence <= 16 || occurrence.is_power_of_two() {
                        let GpuEvent::MemoryAcquire(acquire) = &event else {
                            unreachable!("only ACQUIRE_MEM has a semantic-no-op disposition")
                        };
                        eprintln!(
                            "[REMu][METAL][ACQUIRE-NOOP] occurrence={occurrence} base={:#x} bytes={:#x} gcr={:#x} watched-overlap=stable-clean action=drop",
                            acquire.base_address, acquire.size_bytes, acquire.gcr_control,
                        );
                    }
                    // The proof linearizes this cache action as empty. Never invoke the mutating
                    // acquire executor after it: a later CPU store belongs to later synchronization
                    // and must remain dirty for the next real resource use.
                    self.receiver.finish_event(source);
                    continue;
                }
                let gpu_orderable = disposition == SharedEventEventDisposition::GpuOrderable;
                if !gpu_orderable {
                    if let GpuEvent::ComputeDispatch(dispatch) = &event {
                        self.trace_shared_event_compute_gate(queue_id, &dependencies, dispatch);
                    }
                    let acknowledgement_started = Instant::now();
                    let acknowledgement = match self.encode_active_guest_queue_host_ack() {
                        Ok(acknowledgement) => acknowledgement,
                        Err(error) => {
                            eprintln!(
                                "[REMu][METAL] shared-event host acknowledgement failed queue={queue_id}: {error}"
                            );
                            None
                        }
                    };
                    let acknowledgement_finished = Instant::now();
                    self.prepare_shared_event_ack_phase_trace(
                        acknowledgement,
                        event_queue_wait,
                        acknowledgement_started.saturating_duration_since(event_processing_started),
                        acknowledgement_finished.saturating_duration_since(acknowledgement_started),
                    );
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
                    let host_kind = shared_event_host_effect_kind(&event);
                    self.statistics.shared_event_host_block_kinds[host_kind] =
                        self.statistics.shared_event_host_block_kinds[host_kind].saturating_add(1);
                    self.receiver.block_event(
                        source,
                        event,
                        QueueBlockReason::GuestSharedEventHostEffect {
                            consumer_queue_id: queue_id,
                            consumer_ack: acknowledgement,
                            terminal_dependencies: dependencies,
                            latency: SharedEventHostEffectLatencyState::new(Instant::now()),
                        },
                    );
                    continue;
                }
                shared_event_gpu_ordered_compute = shared_event_direct_compute_plan.is_some();
                if !dependencies.is_empty()
                    && matches!(&event, GpuEvent::RegisteredMemoryWait { .. })
                {
                    shared_event_gpu_ordered_registered_wait = true;
                }
                shared_event_gpu_ordered_draw = direct_draw_gpu_orderable;
                prepared_same_queue_dma_fill = same_queue_dma_fill;
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
                    remu_gpu::record_startup_milestone(
                        remu_gpu::StartupMilestone::FirstFlipAccepted,
                    );
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
                GpuEvent::RenderBackendFlush(flush) => {
                    self.flush_guest_render_backend(flush);
                }
                GpuEvent::DmaSyncBarrier => {
                    if let Err(error) = self.end_guest_dma_sync_barrier() {
                        eprintln!("[REMu][METAL] gpuSyncDma barrier failed: {error}");
                    }
                }
                GpuEvent::MemoryAcquire(acquire) => {
                    if let Err(error) = self.synchronize_cpu_written_guest_resources(acquire) {
                        eprintln!("[REMu][METAL] ACQUIRE_MEM synchronization failed: {error}");
                    }
                }
                GpuEvent::MemoryCopy(copy) => {
                    let destination_address = copy.destination_address;
                    let byte_count = copy.byte_count;
                    let retry_waits_on_dequeue = prepared_same_queue_dma_fill.is_none();
                    let result = if let Some(prepared) = prepared_same_queue_dma_fill {
                        let OrderedEventSource::GuestQueue(queue_id) = source else {
                            unreachable!("same-queue DMA admission requires a guest queue")
                        };
                        self.queue_prepared_same_queue_immediate_fill(queue_id, &copy, prepared)
                    } else {
                        self.execute_guest_memory_copy(copy)
                    };
                    match result {
                        Ok(()) if retry_waits_on_dequeue => {
                            self.receiver.retry_guest_memory_waits_overlapping(
                                destination_address,
                                byte_count,
                            );
                        }
                        Ok(()) => {}
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
                GpuEvent::OcclusionQuery { destination } => {
                    // J1: close the active pass so Counting-mode results land in the visibility
                    // buffer, then publish VALID|count into the 16-pair guest block.
                    if let Err(error) = self.publish_occlusion_query(source, destination) {
                        eprintln!(
                            "[REMu][METAL][J1] occlusion query publication failed destination={destination:#x}: {error}"
                        );
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
                        Ok(GuestMemoryWaitStatus::Satisfied) => {
                            if shared_event_gpu_ordered_registered_wait {
                                self.statistics.shared_event_gpu_ordered_registered_waits = self
                                    .statistics
                                    .shared_event_gpu_ordered_registered_waits
                                    .saturating_add(1);
                            }
                        }
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
                    match self.execute_compute_dispatch(
                        &dispatch,
                        shared_event_direct_compute_plan.as_ref(),
                    ) {
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
                        super::stats::note_noop_draw("gpu:noop-cover-placeholder");
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
                    let cpu_synchronization_epoch = self.guest_batch_cpu_synchronization_epoch;
                    let texture_overrides = if shared_event_gpu_ordered_draw {
                        shared_event_direct_draw_plan
                            .as_ref()
                            .map_or(&[][..], |plan| plan.texture_overrides.as_slice())
                    } else {
                        &[]
                    };
                    let result = match key.as_ref() {
                        Ok(key) => self.execute_compiled_draw(&draw, key, texture_overrides),
                        Err(error) => Err(error.clone()),
                    };
                    self.record_shared_event_direct_draw_outcome(
                        shared_event_gpu_ordered_draw,
                        cpu_synchronization_epoch,
                        result.is_ok(),
                    );
                    if let Err(error) = result {
                        // Every dropped draw is counted so a scene pass failing each frame stays
                        // visible in the perf trail; log the first failure for every concrete
                        // guest shader pair because multiple addresses can share one pipeline key.
                        let dropped = self
                            .dropped_draw_executions
                            .entry((draw.export_shader_address, draw.pixel_shader_address))
                            .or_default();
                        *dropped += 1;
                        if *dropped == 1 {
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
                        remu_gpu::record_startup_milestone(
                            remu_gpu::StartupMilestone::FirstEncodedDraw,
                        );
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

    /// SDK CB/DB writeback-invalidate EVENT_WRITE: store attachments by ending the render pass.
    pub(super) fn flush_guest_render_backend(&mut self, flush: remu_gpu::GpuRenderBackendFlush) {
        let Some(batch) = self.active_open_guest_batch_mut() else {
            return;
        };
        // Partial CS/PS flushes keep the pass open (TBDR). CB/DB writeback events are the SDK's
        // only render-backend flush: ending the pass stores color/depth to guest-visible memory.
        batch.close_active_pass_encoders();
        self.statistics.render_backend_flushes =
            self.statistics.render_backend_flushes.saturating_add(1);
        let _ = flush;
    }

    /// SDK `gpuSyncDma`: wait for prior CP DMAs with write-confirm and block the CP.
    pub(super) fn end_guest_dma_sync_barrier(&mut self) -> Result<(), String> {
        if let Some(batch) = self.active_open_guest_batch_mut() {
            batch.close_active_pass_encoders();
        }
        // Prior DMA fills/copies are already ordered as terminal memory_writes on this batch.
        // Flushing commits them so following CP work cannot overtake unpublished DMA effects.
        self.flush_guest_command_batch(
            FlushPolicy::Required,
            GuestBatchFlushReason::CommandProcessorBarrier,
        )
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

fn shared_event_host_effect_kind(event: &GpuEvent) -> usize {
    match event {
        GpuEvent::MemoryAcquire(_) => SHARED_HOST_KIND_ACQUIRE,
        GpuEvent::IndexedDraw(_) => SHARED_HOST_KIND_DRAW,
        GpuEvent::ComputeDispatch(_) => SHARED_HOST_KIND_COMPUTE,
        GpuEvent::MemoryCopy(_) | GpuEvent::MemoryWrite(_) | GpuEvent::MemoryRelease(_) => {
            SHARED_HOST_KIND_DMA
        }
        GpuEvent::OcclusionQuery { .. } => SHARED_HOST_KIND_DMA,
        GpuEvent::MemoryWait(_)
        | GpuEvent::RegisteredMemoryWait { .. }
        | GpuEvent::WaitFlipDone(_) => SHARED_HOST_KIND_WAIT,
        GpuEvent::Flip { .. } => SHARED_HOST_KIND_FLIP,
        GpuEvent::DisplayRegistered { .. }
        | GpuEvent::ShowHostSplash(_)
        | GpuEvent::HideHostSplash
        | GpuEvent::ShowHostDialog(_)
        | GpuEvent::HideHostDialog { .. }
        | GpuEvent::ShowHostToast(_) => SHARED_HOST_KIND_UI,
        _ => SHARED_HOST_KIND_OTHER,
    }
}

impl MetalPresenter {
    /// Returns whether an invalidating acquire has any live cached guest resource to inspect.
    ///
    /// This is intentionally only an overlap test. Proving that an overlapping write watch is
    /// clean requires a stable generation snapshot; in contrast, an empty overlap set makes the
    /// existing acquire executor a strict no-op because it visits no resident or sampled entry.
    fn invalidating_acquire_has_watched_resource(&self, acquire: GpuMemoryAcquire) -> bool {
        self.resident_guest_textures.values().any(|resident| {
            let (address, byte_count) = resident.write_watch.range();
            acquire.overlaps(address, byte_count)
        }) || self.sampled_texture_cache.values().any(|cached| {
            let (address, byte_count) = cached.write_watch.range();
            acquire.overlaps(address, byte_count)
        })
    }

    pub(super) fn shared_event_event_disposition(
        &self,
        consumer_queue_id: u64,
        source: OrderedEventSource,
        dependencies: &[super::SharedEventFence],
        event: &GpuEvent,
    ) -> SharedEventEventDisposition {
        if let GpuEvent::MemoryAcquire(acquire) = event
            && acquire.invalidates_guest_resources()
            && self.invalidating_acquire_has_watched_resource(*acquire)
            && self.invalidating_acquire_is_stable_noop(consumer_queue_id, dependencies, *acquire)
        {
            return SharedEventEventDisposition::DropSemanticNoOp;
        }
        if self.event_is_shared_event_gpu_orderable(source, event) {
            SharedEventEventDisposition::GpuOrderable
        } else {
            SharedEventEventDisposition::HostEffect
        }
    }

    /// Proves that the normal ACQUIRE_MEM executor would leave every overlapping cached resource
    /// byte-for-byte unchanged at one stable point.
    ///
    /// Presenter-owned map membership cannot change during this method. Write-watch and mapping
    /// state can change on guest threads, so every complete token is sampled twice. Producer
    /// checkpoint effects which eventually publish into guest RAM are also rejected when they
    /// overlap a watched resource; a clean watch before that deferred publication is not a no-op.
    pub(super) fn invalidating_acquire_is_stable_noop(
        &self,
        consumer_queue_id: u64,
        dependencies: &[super::SharedEventFence],
        acquire: GpuMemoryAcquire,
    ) -> bool {
        struct WatchedResource<'a> {
            watch: &'a super::GuestTextureWriteWatch,
            footprint: PhysicalFootprintRange,
            /// GPU-authored residents are never ACQUIRE no-ops: Metal still owns unpublished
            /// bytes, so an invalidating acquire must run the ordinary path.
            gpu_authored: bool,
        }

        fn watched_resource<'a>(
            watch: &'a super::GuestTextureWriteWatch,
            gpu_authored: bool,
        ) -> Option<WatchedResource<'a>> {
            let (address, byte_count) = watch.range();
            let span = ResourceSpan::new(address, byte_count);
            let footprint = PhysicalFootprintRange::new(
                SharedEventWriteKind::ResidentDestination,
                address,
                byte_count,
                span.backing(),
            )?;
            Some(WatchedResource {
                watch,
                footprint,
                gpu_authored,
            })
        }

        let mut watched = Vec::new();
        for resident in self.resident_guest_textures.values() {
            let (address, byte_count) = resident.write_watch.range();
            if !acquire.overlaps(address, byte_count) {
                continue;
            }
            let Some(resource) = watched_resource(&resident.write_watch, resident.gpu_written())
            else {
                return false;
            };
            watched.push(resource);
        }
        for cached in self.sampled_texture_cache.values() {
            let (address, byte_count) = cached.write_watch.range();
            if !acquire.overlaps(address, byte_count) {
                continue;
            }
            let Some(resource) = watched_resource(&cached.write_watch, false) else {
                return false;
            };
            watched.push(resource);
        }
        if watched.is_empty() {
            return false;
        }

        let first = watched
            .iter()
            .map(|resource| resource.watch.acquire_noop_token())
            .collect::<Vec<_>>();
        if first.iter().any(|token| !token.is_clean()) {
            return false;
        }

        let Some(consumer) = self.guest_queue_batches.get(&consumer_queue_id) else {
            return false;
        };
        for dependency in dependencies {
            let Some(checkpoint) = consumer
                .resource_dependencies
                .get(&dependency.producer())
                .filter(|checkpoint| checkpoint.fence == *dependency)
            else {
                return false;
            };
            if !checkpoint.footprint_complete {
                return false;
            }
            if checkpoint.pending_write_footprint.iter().any(|producer| {
                !matches!(
                    producer.kind,
                    SharedEventWriteKind::ResidentDestination
                        | SharedEventWriteKind::InternalResidentDestination
                ) && watched
                    .iter()
                    .any(|resource| resource.footprint.overlaps(*producer))
            }) {
                return false;
            }
        }

        let second = watched
            .iter()
            .map(|resource| resource.watch.acquire_noop_token())
            .collect::<Vec<_>>();
        first == second
            && second.iter().all(|token| token.is_clean())
            && watched.iter().all(|resource| !resource.gpu_authored)
    }

    pub(super) fn record_shared_event_direct_draw_outcome(
        &mut self,
        gpu_ordered: bool,
        cpu_synchronization_epoch_before: u64,
        encoded: bool,
    ) {
        if !gpu_ordered {
            return;
        }
        if self.guest_batch_cpu_synchronization_epoch != cpu_synchronization_epoch_before {
            self.statistics.shared_event_draw_internal_drains = self
                .statistics
                .shared_event_draw_internal_drains
                .saturating_add(1);
        }
        if encoded {
            self.statistics.shared_event_gpu_ordered_draws = self
                .statistics
                .shared_event_gpu_ordered_draws
                .saturating_add(1);
        }
    }

    /// Conservative host-effect gate for commands following an encoded Metal wait.
    pub(super) fn event_is_shared_event_gpu_orderable(
        &self,
        source: OrderedEventSource,
        event: &GpuEvent,
    ) -> bool {
        match event {
            GpuEvent::CommandProcessorBarrier
            | GpuEvent::PipelineBarrier(_)
            | GpuEvent::RenderBackendFlush(_)
            | GpuEvent::DmaSyncBarrier => true,
            // With no invalidation bits, or with no live cached resource in its effective range,
            // ACQUIRE_MEM has no presenter-side resource effect. It is safe to consume after
            // encoding the queue's Metal wait: the normal handler either returns immediately or
            // visits empty overlap sets, so no cache or guest-visible state can overtake the
            // producer. Invalidating acquires which touch a live watch remain host-gated.
            GpuEvent::MemoryAcquire(acquire) => {
                if !acquire.invalidates_guest_resources() {
                    true
                } else if !self.invalidating_acquire_has_watched_resource(*acquire) {
                    static EMPTY_INVALIDATING_ACQUIRE_COUNT: AtomicUsize = AtomicUsize::new(0);
                    let occurrence =
                        EMPTY_INVALIDATING_ACQUIRE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                    if occurrence <= 16 || occurrence.is_power_of_two() {
                        eprintln!(
                            "[REMu][METAL][ACQUIRE-NOOP] occurrence={occurrence} base={:#x} bytes={:#x} gcr={:#x} watched-overlap=none",
                            acquire.base_address, acquire.size_bytes, acquire.gcr_control,
                        );
                    }
                    true
                } else {
                    false
                }
            }
            // A registered wait can extend an already GPU-ordered consumer segment without
            // making the presenter observe label RAM. Admission is generation-safe only when
            // this exact submitted wait ticket is still bound to a retained producer checkpoint;
            // the normal wait handler then encodes that Metal event and transfers its host and
            // resource dependencies into the consumer batch. Missing or retired checkpoints
            // remain behind the host gate and use the conservative wait path.
            GpuEvent::RegisteredMemoryWait { id, .. } => {
                self.registered_wait_has_exact_shared_event_checkpoint(source, *id)
            }
            // Compute admission owns retained Metal buffer versions and is dependency-specific,
            // so it is decided by `prepare_shared_event_direct_compute_plan` in the event loop.
            GpuEvent::ComputeDispatch(_) => false,
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
            // and mutates shared residency/cache state on the presenter thread. DMA, invalidating
            // acquires, waits, flips and host UI have still more direct CPU-visible effects. They
            // must observe the producer event first.
            _ => false,
        }
    }

    fn registered_wait_has_exact_shared_event_checkpoint(
        &self,
        source: OrderedEventSource,
        wait_id: remu_gpu::GpuWaitId,
    ) -> bool {
        if !self.per_queue_guest_batches || !matches!(source, OrderedEventSource::GuestQueue(_)) {
            return false;
        }
        let Some(fence) = self.gpu_wait_fences.get(wait_id) else {
            return false;
        };
        self.guest_queue_batches
            .get(&fence.producer().queue_id())
            .is_some_and(|producer| {
                producer
                    .committed_batches
                    .iter()
                    .chain(producer.open_batch.iter())
                    .flat_map(|batch| &batch.signal_checkpoints)
                    .any(|checkpoint| checkpoint.fence == fence)
            })
    }

    fn shared_event_compute_reject(
        &self,
        dispatch: &remu_gpu::ComputeDispatch,
    ) -> Option<SharedEventComputeReject> {
        let snapshot = match dispatch.shader.as_ref() {
            Some(snapshot) => snapshot,
            None => return Some(SharedEventComputeReject::MissingShaderSnapshot),
        };
        let evaluation = match dispatch.shader_evaluation.as_ref() {
            Some(evaluation) => evaluation,
            None => return Some(SharedEventComputeReject::MissingShaderEvaluation),
        };
        compute_program_shared_event_reject(dispatch, snapshot, evaluation).or_else(|| {
            (self
                .shared_event_compute_global_binding_plan(evaluation, &[])
                .is_none())
            .then_some(SharedEventComputeReject::UnretainedGlobalVersion)
        })
    }

    pub(super) fn prepare_shared_event_direct_compute_plan(
        &self,
        consumer_queue_id: u64,
        dependencies: &[super::SharedEventFence],
        dispatch: &remu_gpu::ComputeDispatch,
    ) -> Option<SharedEventDirectComputePlan> {
        if !self.per_queue_guest_batches
            || dependencies.is_empty()
            || dependencies
                .iter()
                .any(|fence| fence.producer().queue_id() != consumer_queue_id)
        {
            return None;
        }
        let snapshot = dispatch.shader.as_ref()?;
        let evaluation = dispatch.shader_evaluation.as_ref()?;
        if compute_program_shared_event_reject(dispatch, snapshot, evaluation).is_some()
            || !program_memory_is_completely_covered(snapshot, evaluation, false)
        {
            return None;
        }
        let consumer = self.guest_queue_batches.get(&consumer_queue_id)?;
        let mut dependency_write_footprint = Vec::new();
        for dependency in dependencies {
            let checkpoint = consumer
                .resource_dependencies
                .get(&dependency.producer())
                .filter(|checkpoint| checkpoint.fence == *dependency)?;
            if !checkpoint.footprint_complete {
                return None;
            }
            dependency_write_footprint.extend_from_slice(&checkpoint.pending_write_footprint);
            for &(address, _) in &evaluation.scalar_memory_reads {
                let scalar = PhysicalFootprintRange::new(
                    SharedEventWriteKind::CopiedGlobal,
                    address,
                    4,
                    guest_shared_backing_identity(address, 4),
                )?;
                if checkpoint
                    .pending_write_footprint
                    .iter()
                    .any(|&producer| scalar.overlaps(producer))
                {
                    return None;
                }
            }
        }
        Some(SharedEventDirectComputePlan {
            global_bindings: self.shared_event_compute_global_binding_plan(
                evaluation,
                &dependency_write_footprint,
            )?,
        })
    }

    fn trace_shared_event_compute_gate(
        &self,
        consumer_queue_id: u64,
        dependencies: &[super::SharedEventFence],
        dispatch: &remu_gpu::ComputeDispatch,
    ) {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        if !*ENABLED.get_or_init(|| {
            std::env::var_os("REMU_TRACE_SHARED_EVENT_COMPUTE").is_some_and(|value| value == "1")
        }) {
            return;
        }

        static OCCURRENCES: AtomicUsize = AtomicUsize::new(0);
        let occurrence = OCCURRENCES.fetch_add(1, Ordering::Relaxed) + 1;
        if occurrence > 64 && !occurrence.is_power_of_two() {
            return;
        }

        let reason = self
            .shared_event_compute_reject(dispatch)
            .map_or("non-compute-disposition", SharedEventComputeReject::label);
        let (header_address, instruction_count, supplemental_count) =
            dispatch.shader.as_ref().map_or((0, 0, 0), |snapshot| {
                (
                    snapshot.header_address,
                    snapshot.program.instructions.len(),
                    snapshot.supplemental_programs.len(),
                )
            });
        let (global_count, writable_global_count, image_count, image_load_count, image_write_count) =
            dispatch
                .shader_evaluation
                .as_ref()
                .map_or((0, 0, 0, 0, 0), |evaluation| {
                    (
                        evaluation.global_memory_bindings.len(),
                        evaluation
                            .global_memory_bindings
                            .iter()
                            .filter(|binding| binding.writable)
                            .count(),
                        evaluation.image_bindings.len(),
                        evaluation
                            .image_bindings
                            .iter()
                            .filter(|binding| binding.opcode.starts_with("ImageLoad"))
                            .count(),
                        evaluation
                            .image_bindings
                            .iter()
                            .filter(|binding| {
                                binding.opcode.starts_with("ImageStore")
                                    || binding.opcode.starts_with("ImageAtomic")
                            })
                            .count(),
                    )
                });
        let dependency_text = dependencies
            .iter()
            .map(|fence| format!("q{}:{}", fence.producer().queue_id(), fence.value()))
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "[REMu][METAL][SHARED-COMPUTE-GATE] occurrence={occurrence} dispatch={} shader={:#x} header={header_address:#x} source={:?} consumer=q{consumer_queue_id} dependencies=[{dependency_text}] reason={reason} instructions={instruction_count} supplemental={supplemental_count} globals={global_count} writable-globals={writable_global_count} images={image_count} image-loads={image_load_count} image-writes={image_write_count}",
            dispatch.sequence, dispatch.shader_address, dispatch.source,
        );
    }
}

pub(crate) fn shared_event_host_effect_is_resolved(
    latency: &mut super::SharedEventHostEffectLatencyState,
    now: Instant,
    statistics: &mut super::SharedEventHostEffectLatencyStatistics,
    consumer_queue_id: u64,
    consumer_ack: Option<super::SharedEventFence>,
    terminal_dependencies: &[super::SharedEventFence],
    mut queue_has_outstanding_batches: impl FnMut(u64) -> bool,
    mut consumer_shared_event_value: impl FnMut(super::SharedEventProducerId) -> Option<u64>,
    mut producer_shared_event_value: impl FnMut(super::SharedEventProducerId) -> Option<u64>,
    mut host_retired_value: impl FnMut(super::SharedEventProducerId) -> Option<u64>,
) -> bool {
    if latency.phase == super::SharedEventHostEffectLatencyPhase::ConsumerAck {
        let consumer_reached_wait = consumer_ack.map_or_else(
            || !queue_has_outstanding_batches(consumer_queue_id),
            |fence| {
                consumer_shared_event_value(fence.producer())
                    .is_some_and(|value| value >= fence.value())
            },
        );
        if !consumer_reached_wait {
            return false;
        }
        latency.complete_phase(now, statistics);
    }

    if latency.phase == super::SharedEventHostEffectLatencyPhase::ImmutablePrefix {
        let immutable_prefix_reached = terminal_dependencies.iter().all(|fence| {
            producer_shared_event_value(fence.producer())
                .is_some_and(|value| value >= fence.value())
        });
        if !immutable_prefix_reached {
            return false;
        }
        latency.complete_phase(now, statistics);
    }

    let terminal_reached = terminal_dependencies.iter().all(|fence| {
        host_retired_value(fence.producer()).is_some_and(|value| value >= fence.value())
    });
    if terminal_reached {
        latency.complete_phase(now, statistics);
    }
    terminal_reached
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SharedEventComputeReject {
    IndirectDispatch,
    MissingShaderSnapshot,
    MissingShaderEvaluation,
    SupplementalProgram,
    NoGlobalBindings,
    ImageBinding,
    ImageInstruction,
    GlobalStore,
    GlobalAtomic,
    UncoveredGlobalInstruction,
    UnretainedGlobalVersion,
}

impl SharedEventComputeReject {
    const fn label(self) -> &'static str {
        match self {
            Self::IndirectDispatch => "indirect-dispatch",
            Self::MissingShaderSnapshot => "missing-shader-snapshot",
            Self::MissingShaderEvaluation => "missing-shader-evaluation",
            Self::SupplementalProgram => "supplemental-program",
            Self::NoGlobalBindings => "no-global-bindings",
            Self::ImageBinding => "image-binding",
            Self::ImageInstruction => "image-instruction",
            Self::GlobalStore => "global-store",
            Self::GlobalAtomic => "global-atomic",
            Self::UncoveredGlobalInstruction => "uncovered-global-instruction",
            Self::UnretainedGlobalVersion => "unretained-global-version",
        }
    }
}

fn compute_program_shared_event_reject(
    dispatch: &remu_gpu::ComputeDispatch,
    snapshot: &remu_gpu::ShaderSnapshot,
    evaluation: &remu_shader::ShaderEvaluation,
) -> Option<SharedEventComputeReject> {
    if !matches!(dispatch.source, remu_gpu::ComputeDispatchSource::Direct) {
        return Some(SharedEventComputeReject::IndirectDispatch);
    }
    if !snapshot.supplemental_programs.is_empty() {
        return Some(SharedEventComputeReject::SupplementalProgram);
    }
    if evaluation.global_memory_bindings.is_empty() {
        return Some(SharedEventComputeReject::NoGlobalBindings);
    }
    if !evaluation.image_bindings.is_empty() {
        return Some(SharedEventComputeReject::ImageBinding);
    }
    if !program_memory_is_completely_covered(snapshot, evaluation, false) {
        return Some(SharedEventComputeReject::UncoveredGlobalInstruction);
    }

    snapshot
        .program
        .instructions
        .iter()
        .find_map(|instruction| match instruction.control {
            Some(remu_shader::InstructionControl::Image(_)) => {
                Some(SharedEventComputeReject::ImageInstruction)
            }
            Some(
                remu_shader::InstructionControl::GlobalMemory(_)
                | remu_shader::InstructionControl::BufferMemory(_)
                | remu_shader::InstructionControl::ScalarMemory(_),
            ) => {
                let binding = evaluation
                    .global_memory_bindings
                    .iter()
                    .find(|binding| binding.instruction_pcs.contains(&instruction.pc));
                let Some(binding) = binding else {
                    return Some(SharedEventComputeReject::UncoveredGlobalInstruction);
                };
                if instruction.opcode.contains("Store") && !binding.writable {
                    Some(SharedEventComputeReject::GlobalStore)
                } else if instruction.opcode.contains("Atomic") && !binding.writable {
                    Some(SharedEventComputeReject::GlobalAtomic)
                } else if !binding.instruction_pcs.contains(&instruction.pc) {
                    Some(SharedEventComputeReject::UncoveredGlobalInstruction)
                } else {
                    None
                }
            }
            _ => None,
        })
}
