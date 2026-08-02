// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::WritebackTrigger;
use super::{
    ActiveTranslatedScanout, FlushPolicy, GuestBatchFlushReason, GuestTextureDescriptor,
    HostSplashTexture, MetalPresenter, OrderedEventSource, WriteWatchRefresh, capture_interval,
    completed_command_buffer_gpu_duration, decode_guest_texture_linear_from_raw,
    diagnostic_scanout_writer, diagnostic_texture_writer, raw_video_out_texture_descriptor,
    read_texture_linear, refresh_resident_texture_write_watch, snapshot_guest_texture_raw,
    trace_resident_texture_removal, trace_scanout_writes,
};
use crate::presenter::completion::command_buffer_failure_message;
use metal::{
    MTLCommandBufferStatus, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize, MTLStorageMode,
    MTLTextureType, MTLTextureUsage, Texture, TextureDescriptor,
};
use remu_gpu::GpuHostImage;
use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

impl MetalPresenter {
    fn matches_registered_video_out_scanout(&self, descriptor: GuestTextureDescriptor) -> bool {
        let Some(display) = self.display else {
            return false;
        };
        if descriptor.width != display.width || descriptor.height != display.height {
            return false;
        }
        if self
            .registered_video_out_buffers
            .iter()
            .any(|address| *address == descriptor.address)
        {
            return true;
        }
        self.pending_flip
            .is_some_and(|flip| flip.address == descriptor.address)
    }

    fn scanout_snapshot_address(&self) -> Option<u64> {
        // An accepted flip names the buffer to present. A recorded write only selects the snapshot
        // when no flip is waiting for one, so a write to the other registered buffer cannot divert
        // the pending flip's own snapshot to the frame the guest has not flipped yet.
        if let Some(flip) = self
            .pending_flip
            .filter(|flip| flip.sequence != self.uploaded_flip_sequence)
        {
            return Some(flip.address);
        }
        self.pending_scanout_write_address
            .or_else(|| self.pending_flip.map(|flip| flip.address))
    }

    fn pipeline_blocked_scanout_address(&self) -> Option<u64> {
        self.scanout_snapshot_address()
            .filter(|address| self.receiver.pipeline_blocked_draw_writes(*address))
    }

    pub(crate) fn finalize_pending_flip(&mut self) {
        let Some(display) = self.display else {
            return;
        };
        let Some(address) = self.scanout_snapshot_address() else {
            return;
        };
        let flip = self.pending_flip;
        let activation =
            match self.activate_translated_scanout(address, display.width, display.height) {
                Ok(true) => {
                    if let Some(flip) = flip {
                        self.uploaded_flip_sequence = flip.sequence;
                    }
                    "translated"
                }
                Ok(false) => {
                    if let Err(error) = self.synchronize_flip_target(address) {
                        eprintln!("[REMu][METAL] flip synchronization failed: {error}");
                    }
                    if let Some(flip) = flip {
                        self.uploaded_flip_sequence = flip.sequence;
                    }
                    "raw-pending"
                }
                Err(error) => {
                    eprintln!("[REMu][METAL] translated scanout synchronization failed: {error}");
                    "translated-error"
                }
            };
        if let Some(flip) = flip {
            self.finalized_flip_sequence = flip.sequence;
            self.garbage_collect_resident_textures(flip.sequence);
            // Without a drawable, queue a tail command buffer behind the snapshot and guest work.
            // The old synchronous completion let the guest recycle the scanout allocation while
            // Metal still held it in flight. A tail remains asynchronous without depending on
            // windowing state and gives this half of the completion join a truthful terminal point.
            if self.layer.is_none() {
                if let Some(command_queue) = self.command_queue.clone() {
                    if let Some(sequence) = self.claim_flip_terminal(flip.sequence) {
                        let command_buffer = command_queue.new_command_buffer().to_owned();
                        command_buffer.set_label("REMu surface-less flip retirement tail");
                        command_buffer.commit();
                        self.pending_presentations.push_back(super::super::PendingPresentation {
                            command_buffer,
                            submitted_at: std::time::Instant::now(),
                            flip_sequence: Some(sequence),
                            flip_terminal_source: super::super::FlipTerminalSource::SurfaceLess,
                        });
                    }
                } else {
                    eprintln!(
                        "[REMu][COMPLETION-ORDER] phase=surface-less-blocked flip={} reason=command-queue-unavailable",
                        flip.sequence,
                    );
                }
            }
        }
        self.pending_scanout_write_address = None;
        self.redraw_pending = true;
        if trace_scanout_writes() {
            eprintln!(
                "[REMu][METAL][SCANOUT-FLIP] phase=finalize sequence={} buffer={} address={address:#x} activation={activation} finalized={} uploaded={} submitted={} pending_queue={:?} target_written={}",
                flip.map_or(0, |flip| flip.sequence),
                flip.map_or(0, |flip| flip.buffer_index),
                self.finalized_flip_sequence,
                self.uploaded_flip_sequence,
                self.submitted_flip_sequence,
                self.pending_flip_queue,
                self.pending_flip_target_written_since_boundary,
            );
        }
    }

    /// Claims the terminal side of `sequence` for exactly one retirement path.
    ///
    /// Every accepted flip increments the guest's pacing counter and only a completion decrements
    /// it, so exactly one caller may retire a given sequence: presentation, a surface-less
    /// finalization, or replacement by a newer flip.
    pub(crate) fn claim_flip_terminal(&mut self, sequence: u64) -> Option<u64> {
        let join = self.flip_completion_joins.get_mut(&sequence)?;
        if join.terminal_claimed {
            return None;
        }
        join.terminal_claimed = true;
        self.submitted_flip_sequence = self.submitted_flip_sequence.max(sequence);
        Some(sequence)
    }

    pub(crate) fn register_flip_completion_join(&mut self, flip: super::super::PendingFlip) {
        let submission_terminal = flip.submission_sequence.is_none();
        let replaced = self.flip_completion_joins.insert(
            flip.sequence,
            super::super::FlipCompletionJoin {
                completion: remu_gpu::GpuFlipCompletion {
                    address: flip.address,
                    buffer_index: flip.buffer_index,
                    flip_argument: flip.flip_argument,
                },
                submission_sequence: flip.submission_sequence,
                submission_terminal,
                terminal_claimed: false,
                terminal_source: None,
                accepted_at: std::time::Instant::now(),
            },
        );
        debug_assert!(replaced.is_none(), "flip sequence must be unique");
        self.flip_completion_join_statistics.accepted = self
            .flip_completion_join_statistics
            .accepted
            .saturating_add(1);
        if completion_order_trace_sequence(flip.sequence) {
            eprintln!(
                "[REMu][COMPLETION-ORDER] phase=accept flip={} submission={:?} submission_terminal={submission_terminal}",
                flip.sequence, flip.submission_sequence,
            );
        }
    }

    pub(crate) fn mark_flip_terminal(
        &mut self,
        sequence: u64,
        source: super::super::FlipTerminalSource,
    ) {
        let Some(join) = self.flip_completion_joins.get_mut(&sequence) else {
            return;
        };
        if join.terminal_source.is_some() {
            return;
        }
        join.terminal_source = Some(source);
        if !join.submission_terminal {
            self.flip_completion_join_statistics.deferred = self
                .flip_completion_join_statistics
                .deferred
                .saturating_add(1);
        }
        self.publish_ready_flip_completions();
    }

    pub(crate) fn publish_guest_submission_completion(&mut self, submission_sequence: u64) {
        // Submission visibility is published only after finish_terminal_guest_batch applied every
        // ordered CPU-visible effect. Only then may flips carrying this exact sequence join it.
        remu_gpu::record_gpu_submission_completed(submission_sequence);
        let mut marked = 0_u64;
        for join in self.flip_completion_joins.values_mut().filter(|join| {
            join.submission_sequence == Some(submission_sequence) && !join.submission_terminal
        }) {
            join.submission_terminal = true;
            if join.terminal_source.is_none() {
                marked = marked.saturating_add(1);
            }
        }
        self.flip_completion_join_statistics.deferred = self
            .flip_completion_join_statistics
            .deferred
            .saturating_add(marked);
        self.publish_ready_flip_completions();
    }

    fn publish_ready_flip_completions(&mut self) {
        loop {
            let Some((&sequence, join)) = self.flip_completion_joins.first_key_value() else {
                return;
            };
            let Some(source) = join.terminal_source else {
                return;
            };
            if !join.submission_terminal {
                return;
            }
            let join = self
                .flip_completion_joins
                .remove(&sequence)
                .expect("front flip completion join exists");
            remu_gpu::record_gpu_flip_completed(join.completion);
            self.flip_completion_join_statistics.published = self
                .flip_completion_join_statistics
                .published
                .saturating_add(1);
            if completion_order_trace_sequence(sequence) {
                eprintln!(
                    "[REMu][COMPLETION-ORDER] phase=publish flip={sequence} submission={:?} terminal={} submission_terminal=true age_ms={:.3}",
                    join.submission_sequence,
                    source.as_str(),
                    join.accepted_at.elapsed().as_secs_f64() * 1000.0,
                );
            }
        }
    }

    /// Flips this presenter still owns: accepted but neither presented nor already published.
    fn outstanding_flip_count(&self) -> usize {
        self.flip_completion_joins.len()
    }

    pub(crate) fn flip_completion_join_state_counts(&self) -> (usize, usize, usize) {
        let waiting_submission = self
            .flip_completion_joins
            .values()
            .filter(|join| !join.submission_terminal)
            .count();
        let waiting_terminal = self
            .flip_completion_joins
            .values()
            .filter(|join| join.terminal_source.is_none())
            .count();
        let joined_ordered = self
            .flip_completion_joins
            .values()
            .filter(|join| join.submission_terminal && join.terminal_source.is_some())
            .count();
        (waiting_submission, waiting_terminal, joined_ordered)
    }

    /// Re-syncs the guest's flip pacing counter against this presenter's flip accounting.
    ///
    /// A lost flip is otherwise unrecoverable: the counter only falls on completion, and three
    /// leaked flips put the guest into permanent multi-second pacing stalls. Reconciliation is
    /// restricted to the point where no accepted event is still undelivered, because a flip in the
    /// channel counts against the guest before this presenter has sequenced it.
    pub(crate) fn reconcile_pending_flips(&mut self) {
        const RECONCILE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);
        if self.statistics.last_flip_reconciliation.elapsed() < RECONCILE_INTERVAL {
            return;
        }
        self.statistics.last_flip_reconciliation = std::time::Instant::now();
        if remu_gpu::gpu_work_items_pending() != 0 {
            return;
        }
        let outstanding = self.outstanding_flip_count();
        let Some(observed) = remu_gpu::reconcile_gpu_flips_pending(outstanding) else {
            return;
        };
        let (waiting_submission, waiting_terminal, joined_ordered) =
            self.flip_completion_join_state_counts();
        eprintln!(
            "[REMu][METAL] flip accounting diverged: guest pending={observed} presenter outstanding={outstanding} joins=wait-submission:{waiting_submission}/wait-terminal:{waiting_terminal}/joined-ordered:{joined_ordered} next={} finalized={} uploaded={} submitted={} presentations={}; corrected to {outstanding}",
            self.next_flip_sequence,
            self.finalized_flip_sequence,
            self.uploaded_flip_sequence,
            self.submitted_flip_sequence,
            self.pending_presentations.len(),
        );
    }

    pub(crate) fn show_host_splash(&mut self, image: GpuHostImage) -> Result<(), String> {
        let device = self.device.as_ref().ok_or("Metal device is unavailable")?;
        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm_sRGB);
        descriptor.set_width(u64::from(image.width()));
        descriptor.set_height(u64::from(image.height()));
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_usage(MTLTextureUsage::ShaderRead);
        let texture = device.new_texture(&descriptor);
        texture.set_label("REMu installed-title system splash");
        let bytes_per_row = u64::from(image.width())
            .checked_mul(4)
            .ok_or("system splash row size overflows")?;
        texture.replace_region(
            MTLRegion::new_2d(0, 0, u64::from(image.width()), u64::from(image.height())),
            0,
            image.rgba8().as_ptr().cast::<c_void>(),
            bytes_per_row,
        );
        self.host_splash = Some(HostSplashTexture {
            texture,
            width: image.width(),
            height: image.height(),
        });
        self.redraw_pending = true;
        Ok(())
    }

    pub(crate) fn note_pending_flip_target_write(&mut self, descriptor: GuestTextureDescriptor) {
        if !self.matches_registered_video_out_scanout(descriptor) {
            return;
        }
        // Writes to the buffer already snapshotted for this flip belong to the next frame.
        if self.pending_flip.is_some_and(|flip| {
            flip.sequence == self.uploaded_flip_sequence && flip.address == descriptor.address
        }) {
            return;
        }
        // A guest submission can clear, composite, and decorate one VideoOut target through
        // several render passes. Presenting the first write exposes the clear pass and can
        // retire the flip before the interface/game image is complete. Coalesce all matching
        // writes and activate the final resident at the ordered submission boundary.
        self.pending_flip_target_written_since_boundary = true;
        self.pending_scanout_write_address = Some(descriptor.address);
    }

    pub(crate) fn finalize_pending_flip_at_submission_boundary(
        &mut self,
        source: OrderedEventSource,
    ) {
        let initial_flip_boundary = matches!(
            (source, self.pending_flip_queue),
            (OrderedEventSource::GuestQueue(queue_id), Some(pending_queue))
                if queue_id == pending_queue
        );
        // Only a write to the pending flip's own target completes that flip's frame. A write to the
        // other registered buffer, or another write once this flip is already snapshotted, is state
        // for the next flip: presenting it publishes a mid-render frame and costs a second present
        // per guest frame without ever retiring a flip.
        let write_completes_pending_flip = self.pending_flip_target_written_since_boundary
            && self.pending_flip.is_some_and(|flip| {
                self.pending_scanout_write_address == Some(flip.address)
                    && flip.sequence != self.uploaded_flip_sequence
            });
        if !initial_flip_boundary && !write_completes_pending_flip {
            return;
        }
        // A foreign queue's boundary orders a compute or copy producer that completed this frame.
        // It must not publish the allocation while a draw writing the same target is still parked
        // on a cold pipeline: that presents the compositor clear and then treats the cover's
        // output as the next frame. The parked draw's own boundary finalizes it instead.
        if !initial_flip_boundary
            && self.pipeline_blocked_scanout_address().is_some()
        {
            return;
        }
        if initial_flip_boundary {
            self.pending_flip_queue = None;
        }
        self.pending_flip_target_written_since_boundary = false;
        self.finalize_pending_flip();
    }

    pub(crate) fn retire_completed_presentations(&mut self) {
        // A presented frame's flip completion tells the guest to retire that frame and recycle
        // its memory, so every guest batch that already completed must publish its ordered
        // effects first. The standalone call sites in the event loop otherwise let a flip lead
        // its own frame's RELEASE_MEM/writeback application by a full tick.
        self.retire_completed_guest_batches();
        self.reconcile_pending_flips();
        loop {
            let Some(presentation) = self.pending_presentations.front() else {
                return;
            };
            match presentation.command_buffer.status() {
                MTLCommandBufferStatus::Completed => {
                    let presentation = self
                        .pending_presentations
                        .pop_front()
                        .expect("front presentation exists");
                    if presentation.flip_terminal_source
                        == super::super::FlipTerminalSource::Presentation
                    {
                        self.statistics.completed_presents =
                            self.statistics.completed_presents.saturating_add(1);
                    }
                    self.statistics.gpu_execution_time +=
                        completed_command_buffer_gpu_duration(&presentation.command_buffer);
                    self.statistics.presentation_queue_latency +=
                        presentation.submitted_at.elapsed();
                    if let Some(sequence) = presentation.flip_sequence {
                        self.mark_flip_terminal(sequence, presentation.flip_terminal_source);
                    }
                }
                MTLCommandBufferStatus::Error => {
                    let presentation = self
                        .pending_presentations
                        .pop_front()
                        .expect("front presentation exists");
                    eprintln!(
                        "[REMu][METAL] {}",
                        command_buffer_failure_message(
                            "presentation command buffer",
                            &presentation.command_buffer,
                        )
                    );
                    // A terminal presentation failure cannot be retried in place. Complete the
                    // flip after reporting it so the guest does not deadlock on an unavailable
                    // drawable while later frames may still recover.
                    if let Some(sequence) = presentation.flip_sequence {
                        self.mark_flip_terminal(sequence, presentation.flip_terminal_source);
                    }
                }
                _ => return,
            }
        }
    }

    pub(crate) fn synchronize_flip_target(&mut self, address: u64) -> Result<(), String> {
        let display = self.display;
        let descriptor = self
            .resident_guest_textures
            .values()
            .filter(|resident| {
                resident.descriptor.address == address
                    && resident.gpu_written()
                    && resident.serves_reads()
            })
            .filter(|resident| {
                display.is_none_or(|display| {
                    resident.descriptor.width == display.width
                        && resident.descriptor.height == display.height
                })
            })
            .max_by_key(|resident| resident.content_version)
            .map(|resident| resident.descriptor);
        let Some(descriptor) = descriptor else {
            return Ok(());
        };
        self.wait_for_guest_command_batches()?;
        let resident = self
            .resident_guest_textures
            .get_mut(&descriptor)
            .expect("selected resident texture exists");
        let refresh = refresh_resident_texture_write_watch(resident)?;
        match refresh {
            WriteWatchRefresh::Replaced => {
                if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
                    trace_resident_texture_removal("scanout-mapping-replaced", &resident);
                }
                return Ok(());
            }
            WriteWatchRefresh::Reprotecting => return Ok(()),
            WriteWatchRefresh::Active | WriteWatchRefresh::Renewed { .. } => {}
        }
        if self
            .resident_guest_textures
            .get_mut(&descriptor)
            .expect("refreshed scanout resident exists")
            .cpu_write_obsoletes_gpu_writeback(refresh)?
        {
            if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
                trace_resident_texture_removal("cpu-reused-scanout", &resident);
            }
            return Ok(());
        }
        // The flip buffer is read out of guest memory by presentation and by anything the guest
        // points at the same address, so this range is one the guest itself is about to observe.
        self.settle_resident_texture_writeback(descriptor, WritebackTrigger::FlipTarget)?;
        Ok(())
    }

    pub(crate) fn translated_scanout_source(
        &self,
        address: u64,
        width: u32,
        height: u32,
    ) -> Option<(GuestTextureDescriptor, Texture, &'static str)> {
        // A CB-backed resident represents the architectural render target and carries a content
        // version. A synthetic direct target is only a fallback for streams whose CB state was
        // absent, so it must never outrank newer resident compositor output at the same VideoOut
        // address.
        let resident = self
            .resident_guest_textures
            .values()
            .filter(|resident| {
                resident.serves_reads()
                    && resident.descriptor.address == address
                    && resident.descriptor.width == width
                    && resident.descriptor.height == height
            })
            .max_by_key(|resident| resident.content_version)
            .map(|resident| (resident.descriptor, resident.texture.clone(), "resident"));
        let direct = self
            .direct_scanout_textures
            .get(&address)
            .filter(|scanout| {
                scanout.initialized
                    && scanout.descriptor.width == width
                    && scanout.descriptor.height == height
            })
            .map(|scanout| (scanout.descriptor, scanout.texture.clone(), "direct"));
        resident.or(direct)
    }

    pub(crate) fn trace_translated_scanout_candidates(
        &self,
        address: u64,
        width: u32,
        height: u32,
        selected: GuestTextureDescriptor,
    ) {
        if !trace_scanout_writes() {
            return;
        }
        let candidates = self
            .resident_guest_textures
            .values()
            .filter(|resident| {
                resident.descriptor.address == address
                    && resident.descriptor.width == width
                    && resident.descriptor.height == height
            })
            .map(|resident| {
                (
                    resident.descriptor,
                    resident.content_version,
                    resident.gpu_written(),
                )
            })
            .collect::<Vec<_>>();
        eprintln!(
            "[REMu][METAL][SCANOUT-SOURCE] address={address:#x} {}x{} selected={selected:?} candidates={candidates:?} last_writer={:?} last_texture_writer={:?}",
            width,
            height,
            diagnostic_scanout_writer(address),
            diagnostic_texture_writer(selected),
        );
    }

    pub(crate) fn activate_translated_scanout(
        &mut self,
        address: u64,
        width: u32,
        height: u32,
    ) -> Result<bool, String> {
        let Some((descriptor, source_texture, source)) =
            self.translated_scanout_source(address, width, height)
        else {
            return Ok(false);
        };
        self.trace_translated_scanout_candidates(address, width, height, descriptor);
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
        let snapshot = self.ensure_scanout_snapshot_texture(&device, descriptor);
        // End and commit every encoder before the flip. The single Metal queue then orders this
        // private-to-private copy between the frame's producer work and any later command stream
        // that clears or reuses the same VideoOut allocation.
        self.flush_guest_command_batch(FlushPolicy::Required, GuestBatchFlushReason::FlipSnapshot)?;
        let command_buffer = queue.new_command_buffer().to_owned();
        command_buffer.set_label("REMu ordered VideoOut snapshot");
        let blit = command_buffer.new_blit_command_encoder();
        blit.set_label("REMu translated scanout snapshot");
        blit.copy_from_texture(
            &source_texture,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize::new(u64::from(descriptor.width), u64::from(descriptor.height), 1),
            &snapshot,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
        );
        blit.end_encoding();
        command_buffer.commit();
        self.frame_texture = Some(snapshot);
        self.frame_texture_size = (descriptor.width, descriptor.height);
        let sequence = self
            .pending_flip
            .filter(|flip| flip.address == address)
            .map_or(self.next_flip_sequence, |flip| flip.sequence);
        let snapshotted_content_version = self
            .resident_guest_textures
            .get(&descriptor)
            .map(|resident| resident.content_version)
            .unwrap_or(0);
        self.active_translated_scanout = Some(ActiveTranslatedScanout {
            address,
            width,
            height,
            sequence,
            snapshotted_content_version,
        });
        if trace_scanout_writes() {
            let selected_state = self
                .resident_guest_textures
                .get(&descriptor)
                .map(|resident| (resident.content_version, resident.gpu_written()));
            eprintln!(
                "[REMu][METAL][SCANOUT-SNAPSHOT] sequence={sequence} address={address:#x} source={source} descriptor={descriptor:?} selected_state={selected_state:?} source_format={:?} snapshot_format={:?} pending={} finalized={} uploaded={} submitted={}",
                source_texture.pixel_format(),
                self.frame_texture
                    .as_ref()
                    .map(|texture| texture.pixel_format()),
                self.pending_flip.map_or(0, |flip| flip.sequence),
                self.finalized_flip_sequence,
                self.uploaded_flip_sequence,
                self.submitted_flip_sequence,
            );
        }
        static LOGGED_TRANSLATED_SCANOUTS: OnceLock<Mutex<HashSet<(u64, &'static str)>>> =
            OnceLock::new();
        let first_for_source = LOGGED_TRANSLATED_SCANOUTS
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert((address, source));
        if first_for_source {
            eprintln!(
                "[REMu][METAL] snapshotted translated {source} scanout sequence={} at {address:#x} {}x{} format={:?} tile={}",
                sequence,
                descriptor.width,
                descriptor.height,
                descriptor.pixel_format,
                descriptor.tile_mode,
            );
        }
        Ok(true)
    }

    pub(crate) fn refresh_translated_scanout_before_present(&mut self) {
        let Some(display) = self.display else {
            return;
        };
        let Some(active) = self.active_translated_scanout else {
            return;
        };
        let Some((descriptor, _, _)) =
            self.translated_scanout_source(active.address, active.width, active.height)
        else {
            return;
        };
        let current_version = self
            .resident_guest_textures
            .get(&descriptor)
            .map(|resident| resident.content_version)
            .unwrap_or(0);
        // Re-snapshot only when resident compositor output advanced past the last private copy.
        // An in-flight batch check on committed work was re-serializing every present because
        // unrelated guest writebacks overlap the VideoOut allocation.
        if current_version <= active.snapshotted_content_version {
            return;
        }
        if let Err(error) = self.activate_translated_scanout(
            active.address,
            display.width,
            display.height,
        ) {
            eprintln!("[REMu][METAL] translated scanout refresh failed: {error}");
        }
    }

    pub(crate) fn upload_pending_frame(&mut self) {
        let Some(display) = self.display else {
            return;
        };
        let Some(address) = self.scanout_snapshot_address() else {
            return;
        };
        let flip = self.pending_flip;
        if flip.is_some_and(|flip| flip.sequence == self.uploaded_flip_sequence)
            && self.pending_scanout_write_address.is_none()
        {
            return;
        }
        match self.activate_translated_scanout(address, display.width, display.height) {
            Ok(true) => {
                if let Some(flip) = flip {
                    self.uploaded_flip_sequence = flip.sequence;
                }
                self.pending_scanout_write_address = None;
                return;
            }
            Ok(false) => {}
            Err(error) => {
                eprintln!("[REMu][METAL] translated scanout retry failed: {error}");
            }
        }
        let Some(flip) = flip else {
            return;
        };
        if flip.sequence == self.uploaded_flip_sequence {
            return;
        }
        // A translated render target is authoritative when one exists. Before the first GPU
        // render pass, however, splash screens and decoded videos can be written directly into a
        // registered VideoOut allocation. Preserve both ABI layouts here: zero is the console's
        // display-tiled mode (the same mode-27 layout exposed by its color target), while one is
        // row-linear.
        let Some(guest_descriptor) = raw_video_out_texture_descriptor(flip.address, display) else {
            if !self.logged_unsupported_layout {
                eprintln!(
                    "[REMu][METAL] cannot upload display layout tiling={} pitch={} width={} yet",
                    display.tiling_mode, display.pitch, display.width
                );
                self.logged_unsupported_layout = true;
            }
            return;
        };
        let Some(device) = self.device.as_ref() else {
            return;
        };
        let Some(row_bytes) = u64::from(display.width).checked_mul(4) else {
            return;
        };
        if flip.address == 0 {
            return;
        }
        let frame = match snapshot_guest_texture_raw(guest_descriptor)
            .and_then(|raw| decode_guest_texture_linear_from_raw(guest_descriptor, &raw))
        {
            Ok(frame) => frame,
            Err(error) => {
                if !self.logged_unsupported_layout {
                    eprintln!(
                        "[REMu][METAL] could not upload raw display buffer at {:#x}: {error}",
                        flip.address
                    );
                    self.logged_unsupported_layout = true;
                }
                return;
            }
        };
        if self.frame_texture_size != (display.width, display.height) {
            let descriptor = TextureDescriptor::new();
            descriptor.set_texture_type(MTLTextureType::D2);
            descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            descriptor.set_width(u64::from(display.width));
            descriptor.set_height(u64::from(display.height));
            descriptor.set_storage_mode(MTLStorageMode::Shared);
            descriptor.set_usage(MTLTextureUsage::ShaderRead);
            self.frame_texture = Some(device.new_texture(&descriptor));
            self.frame_texture_size = (display.width, display.height);
        }
        let Some(texture) = self.frame_texture.as_ref() else {
            return;
        };
        let Ok(bytes_per_row) = usize::try_from(row_bytes) else {
            return;
        };
        texture.replace_region(
            MTLRegion::new_2d(0, 0, u64::from(display.width), u64::from(display.height)),
            0,
            frame.as_ptr().cast::<c_void>(),
            bytes_per_row as u64,
        );
        self.active_translated_scanout = None;
        let capture = (self.capture_directory.is_some()
            && self
                .last_guest_capture
                .is_none_or(|last| last.elapsed() >= capture_interval()))
        .then(|| {
            let descriptor = GuestTextureDescriptor {
                address: flip.address,
                width: display.width,
                height: display.height,
                pitch: display.width,
                tile_mode: 0,
                pixel_format: MTLPixelFormat::BGRA8Unorm,
                bytes_per_pixel: 4,
            };
            self.command_queue
                .as_ref()
                .ok_or_else(|| "Metal command queue is unavailable".to_owned())
                .and_then(|queue| read_texture_linear(device, queue, texture, descriptor))
                .map(|bytes| (descriptor, bytes))
        });
        self.uploaded_flip_sequence = flip.sequence;
        if flip.sequence == 1 {
            eprintln!(
                "[REMu][METAL] uploaded first guest display buffer format={:#x}",
                display.pixel_format
            );
        }
        if let Some(capture) = capture {
            match capture {
                Ok((descriptor, bgra)) => {
                    self.guest_bgra_bytes = bgra;
                    self.capture_guest_frame_if_changed(descriptor);
                }
                Err(error) => eprintln!("[REMu][METAL] could not read back flipped frame: {error}"),
            }
        }
    }
}

fn completion_order_trace_sequence(sequence: u64) -> bool {
    sequence <= 32 || sequence.is_power_of_two()
}
