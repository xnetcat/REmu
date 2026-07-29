// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    ActiveTranslatedScanout, GuestBatchFlushReason, GuestTextureDescriptor, HostSplashTexture,
    MetalPresenter, OrderedEventSource, WriteWatchRefresh, capture_interval,
    completed_command_buffer_gpu_duration, decode_guest_texture_linear_from_raw,
    diagnostic_scanout_writer, diagnostic_texture_writer, raw_video_out_texture_descriptor,
    read_texture_linear, refresh_resident_texture_write_watch,
    retire_multisample_textures_overlapping_writeback,
    retire_resident_textures_overlapping_writeback, retire_sampled_textures_overlapping_writeback,
    snapshot_guest_texture_raw, trace_resident_texture_removal, trace_scanout_writes,
    write_resident_texture_contents_to_guest,
};
use metal::{
    MTLCommandBufferStatus, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize, MTLStorageMode,
    MTLTextureType, MTLTextureUsage, Texture, TextureDescriptor,
};
use remu_gpu::GpuHostImage;
use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

impl MetalPresenter {
    pub(crate) fn finalize_pending_flip(&mut self) {
        let (Some(display), Some(flip)) = (self.display, self.pending_flip) else {
            return;
        };
        match self.activate_translated_scanout(flip.address, display.width, display.height) {
            Ok(true) => {
                self.uploaded_flip_sequence = flip.sequence;
            }
            Ok(false) => {
                if let Err(error) = self.synchronize_flip_target(flip.address) {
                    eprintln!("[REMu][METAL] flip synchronization failed: {error}");
                }
            }
            Err(error) => {
                eprintln!("[REMu][METAL] translated scanout synchronization failed: {error}");
            }
        }
        self.finalized_flip_sequence = flip.sequence;
        self.redraw_pending = true;
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
        let (Some(display), Some(flip)) = (self.display, self.pending_flip) else {
            return;
        };
        if flip.sequence == self.uploaded_flip_sequence {
            // This flip already owns an immutable GPU snapshot. Later writes target a future
            // frame, even when the title reuses the same VideoOut allocation.
            return;
        }
        if flip.address == descriptor.address
            && display.width == descriptor.width
            && display.height == descriptor.height
        {
            // A guest submission can clear, composite, and decorate one VideoOut target through
            // several render passes. Presenting the first write exposes the clear pass and can
            // retire the flip before the interface/game image is complete. Coalesce all matching
            // writes and activate the final resident at the ordered submission boundary.
            self.pending_flip_target_written_since_boundary = true;
        }
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
        if !initial_flip_boundary && !self.pending_flip_target_written_since_boundary {
            return;
        }
        if initial_flip_boundary {
            self.pending_flip_queue = None;
        }
        self.pending_flip_target_written_since_boundary = false;
        self.finalize_pending_flip();
    }

    pub(crate) fn retire_completed_presentations(&mut self) {
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
                    self.statistics.completed_presents =
                        self.statistics.completed_presents.saturating_add(1);
                    self.statistics.gpu_execution_time +=
                        completed_command_buffer_gpu_duration(&presentation.command_buffer);
                    self.statistics.presentation_queue_latency +=
                        presentation.submitted_at.elapsed();
                    if let Some(completion) = presentation.flip_completion {
                        remu_gpu::record_gpu_flip_completed(completion);
                    }
                }
                MTLCommandBufferStatus::Error => {
                    let presentation = self
                        .pending_presentations
                        .pop_front()
                        .expect("front presentation exists");
                    eprintln!("[REMu][METAL] presentation command buffer failed");
                    // A terminal presentation failure cannot be retried in place. Complete the
                    // flip after reporting it so the guest does not deadlock on an unavailable
                    // drawable while later frames may still recover.
                    if let Some(completion) = presentation.flip_completion {
                        remu_gpu::record_gpu_flip_completed(completion);
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
            .filter(|resident| resident.descriptor.address == address && resident.gpu_written)
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
        let resident = self
            .resident_guest_textures
            .get_mut(&descriptor)
            .expect("selected resident texture exists");
        let refresh = refresh_resident_texture_write_watch(resident)?;
        match refresh {
            WriteWatchRefresh::Replaced => {
                self.resident_guest_textures.remove(&descriptor);
                return Ok(());
            }
            WriteWatchRefresh::Reprotecting => return Ok(()),
            WriteWatchRefresh::Active | WriteWatchRefresh::Renewed { .. } => {}
        }
        if self
            .resident_guest_textures
            .get(&descriptor)
            .is_some_and(|resident| resident.cpu_write_obsoletes_gpu_writeback(refresh))
        {
            if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
                trace_resident_texture_removal("cpu-reused-scanout", &resident);
            }
            return Ok(());
        }
        let (alias_ranges, content_version) = {
            let resident = self
                .resident_guest_textures
                .get(&descriptor)
                .expect("refreshed resident texture exists");
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
            .expect("refreshed resident texture exists");
        write_resident_texture_contents_to_guest(&device, &queue, resident)?;
        resident.gpu_written = false;
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
                resident.descriptor.address == address
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
                    resident.gpu_written,
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
        self.flush_guest_command_batch(false, GuestBatchFlushReason::FlipSnapshot)?;
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
        self.active_translated_scanout = Some(ActiveTranslatedScanout {
            address,
            width,
            height,
            sequence: self
                .pending_flip
                .filter(|flip| flip.address == address)
                .map_or(self.next_flip_sequence, |flip| flip.sequence),
        });
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
                self.next_flip_sequence,
                descriptor.width,
                descriptor.height,
                descriptor.pixel_format,
                descriptor.tile_mode,
            );
        }
        Ok(true)
    }

    pub(crate) fn upload_pending_frame(&mut self) {
        let (Some(display), Some(flip)) = (self.display, self.pending_flip) else {
            return;
        };
        if flip.sequence == self.uploaded_flip_sequence {
            return;
        }
        match self.activate_translated_scanout(flip.address, display.width, display.height) {
            Ok(true) => {
                self.uploaded_flip_sequence = flip.sequence;
                return;
            }
            Ok(false) => {}
            Err(error) => {
                eprintln!("[REMu][METAL] translated scanout retry failed: {error}");
            }
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
