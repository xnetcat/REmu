// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::completion::{commit_and_settle, settle};
use super::{
    DecodedRenderTarget, GuestTextureDescriptor, METAL_TEXTURE_BUFFER_ROW_ALIGNMENT,
    MetalPresenter, PendingTextureReadback, RESIDENT_WRITEBACK_DISCARDED,
    RESIDENT_WRITEBACK_DISCARDED_BYTES, RESIDENT_WRITEBACK_PUBLISHED,
    RESIDENT_WRITEBACK_PUBLISHED_BYTES, ResidentGuestTexture, SpanQueryOrder,
    TextureReadbackLayout, capture_interval, diagnostic_dump_glyph_draw,
    diagnostic_dump_target_before_pixel_shader, diagnostic_dump_target_index_count,
    diagnostic_dump_target_minimum_draw, diagnostic_scanout_writer,
    encode_mode27_mip_chain_into_raw, evaluation_samples_glyph_atlas, half_float_to_f32,
    half_float_to_srgb_unorm8, half_float_to_unorm8, linear_unorm8_to_srgb_unorm8,
    mode27_mip_chain_layout, retire_multisample_textures_overlapping_writeback,
    retire_sampled_textures_overlapping_writeback, snapshot_watched_guest_range,
    texture_element_layout, unsigned_float_to_f32, unsigned_float_to_srgb_unorm8,
    write_guest_texture_linear_into_raw_at_z, write_watched_guest_range,
    write_watched_guest_texture_linear, StaleSource, TextureWritebackOutcome,
};
use metal::{
    Device, MTLBlitOption, MTLOrigin, MTLPixelFormat, MTLResourceOptions, MTLSize, MTLStorageMode,
    MTLTextureType, Texture,
};
use remu_gpu::IndexedDraw;
use std::collections::HashSet;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

fn begin_capture_attempt_if_due(
    last_capture: &mut Option<Instant>,
    now: Instant,
    interval: Duration,
) -> bool {
    if last_capture.is_some_and(|last| now.saturating_duration_since(last) < interval) {
        return false;
    }
    *last_capture = Some(now);
    true
}

impl MetalPresenter {
    pub(super) fn capture_scanout_writer_after_draw(
        &mut self,
        slot: u64,
        draw: &IndexedDraw,
        target: DecodedRenderTarget,
    ) -> Result<(), String> {
        let Some(capture_root) = self.capture_directory.clone() else {
            eprintln!(
                "[REMu][METAL][SCANOUT-AFTER] slot={slot} draw={} ps={:#x} skipped: REMU_CAPTURE_DIR is unset",
                draw.sequence, draw.pixel_shader_address,
            );
            return Ok(());
        };

        // This is intentionally host-effectful and exists only behind the explicit writer
        // selector. Settling here makes each artifact an ordered post-draw observation instead of
        // a later flip snapshot that may already include several subsequent UI writers.
        self.wait_for_guest_command_batches()?;
        let descriptor = target.texture;
        let texture = if target.direct_scanout {
            self.direct_scanout_textures
                .get(&descriptor.address)
                .ok_or("diagnostic scanout writer has no direct scanout texture")?
                .texture
                .clone()
        } else if let Some(resident) = self
            .resident_guest_textures
            .get(&descriptor)
            .filter(|resident| resident.serves_reads())
        {
            resident.texture.clone()
        } else if let Some(snapshot) = self.scanout_snapshot_textures.get(&descriptor.address) {
            snapshot.texture.clone()
        } else {
            self.resident_guest_textures
                .values()
                .find(|resident| {
                    resident.serves_reads()
                        && super::guest_texture_descriptors_share_storage(
                            resident.descriptor,
                            descriptor,
                        )
                })
                .ok_or("diagnostic scanout writer has no readable Metal texture")?
                .texture
                .clone()
        };
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
        let native = read_texture_linear(&device, &queue, &texture, descriptor)?;
        let fingerprint = guest_frame_fingerprint(&native);
        let bgra = native_texture_capture_to_bgra(&native, descriptor).ok_or_else(|| {
            format!(
                "cannot convert diagnostic scanout format {:?} to BGRA",
                descriptor.pixel_format
            )
        })?;
        let pixels = bgra.chunks_exact(4);
        let pixel_count = pixels.len();
        let black = pixels
            .clone()
            .filter(|pixel| pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0)
            .count();
        let green = pixels
            .filter(|pixel| {
                let blue = u16::from(pixel[0]);
                let green = u16::from(pixel[1]);
                let red = u16::from(pixel[2]);
                green >= 128 && green >= red.saturating_add(48) && green >= blue.saturating_add(48)
            })
            .count();
        let directory = capture_root.join("scanout-writers").join(format!(
            "{slot:03}-draw-{}-ps-{:x}",
            draw.sequence, draw.pixel_shader_address
        ));
        let path = capture_guest_frame(&directory, &bgra, descriptor, fingerprint)?;
        eprintln!(
            "[REMu][METAL][SCANOUT-AFTER] slot={slot} draw={} es={:#x} ps={:#x} target={:#x} {}x{} format={:?} fingerprint={fingerprint:016x} black={black}/{pixel_count} green={green}/{pixel_count} captured={}",
            draw.sequence,
            draw.export_shader_address,
            draw.pixel_shader_address,
            descriptor.address,
            descriptor.width,
            descriptor.height,
            descriptor.pixel_format,
            path.display(),
        );
        Ok(())
    }

    /// Settles a resident's publication debt for `trigger`, and reports whether guest memory
    /// was written.
    ///
    /// A refused trigger keeps the debt: the bytes still exist only in Metal, and saying otherwise
    /// would let a later guest-observable read believe guest RAM already agrees. Authority is not
    /// handed to guest RAM either, so the resident keeps serving reads from its own texels.
    pub(super) fn settle_resident_texture_writeback(
        &mut self,
        descriptor: GuestTextureDescriptor,
        trigger: WritebackTrigger,
    ) -> Result<bool, String> {
        let Some(resident) = self.resident_guest_textures.get(&descriptor) else {
            return Ok(false);
        };
        if !resident.gpu_written() {
            return Ok(false);
        }
        if let Some(refusal) = trigger.refusal() {
            RESIDENT_WRITEBACK_DISCARDED[trigger.index()].fetch_add(1, Ordering::Relaxed);
            RESIDENT_WRITEBACK_DISCARDED_BYTES
                .fetch_add(resident.gpu_authored.byte_count(), Ordering::Relaxed);
            trace_discarded_resident_writeback(resident, trigger, refusal);
            return Ok(false);
        }
        let alias_ranges = resident.write_watch.alias_ranges();
        // A sampled or multisample view of the same storage would shadow the bytes about to be
        // published, so both are retired only on the path that really publishes them.
        retire_sampled_textures_overlapping_writeback(
            &mut self.sampled_texture_cache,
            &alias_ranges,
        );
        self.note_resident_writeback_published(descriptor);
        retire_multisample_textures_overlapping_writeback(
            &mut self.multisample_textures,
            &alias_ranges,
        );
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
            .ok_or("resident texture disappeared before its writeback")?;
        let published_bytes = resident.gpu_authored.byte_count();
        let outcome = write_resident_texture_contents_to_guest(&device, &queue, resident)?;
        match outcome {
            TextureWritebackOutcome::Applied => {
                // Guest RAM and the Metal allocation are identical now: the allocation stays in
                // place, still the authority for its bytes and owing none of them.
                resident.clear_gpu_authored();
                RESIDENT_WRITEBACK_PUBLISHED[trigger.index()].fetch_add(1, Ordering::Relaxed);
                RESIDENT_WRITEBACK_PUBLISHED_BYTES.fetch_add(published_bytes, Ordering::Relaxed);
                Ok(true)
            }
            TextureWritebackOutcome::DiscardedStale => {
                // D4-texture: a revoked lease must demote loudly, never silently rebuild.
                RESIDENT_WRITEBACK_DISCARDED[trigger.index()].fetch_add(1, Ordering::Relaxed);
                RESIDENT_WRITEBACK_DISCARDED_BYTES.fetch_add(published_bytes, Ordering::Relaxed);
                eprintln!(
                    "[REMu][METAL] demoting resident after revoked texture writeback lease address={:#x} trigger={}",
                    resident.descriptor.address,
                    trigger.label(),
                );
                resident.demote(StaleSource::GuestMemory);
                Ok(false)
            }
        }
    }

    pub(super) fn direct_scanout_needs_initial_clear(&self, address: u64) -> bool {
        self.direct_scanout_textures
            .get(&address)
            .is_none_or(|scanout| !scanout.initialized)
    }

    pub(super) fn is_diagnostic_scanout_target(&self, target: &DecodedRenderTarget) -> bool {
        // A render target decoded from ordinary CB registers does not carry `direct_scanout`
        // even when its address is an already registered VideoOut buffer. Diagnostics need the
        // address identity as well so they can follow writers that arrive without the synthetic
        // scanout target metadata. This predicate is intentionally not used for texture routing
        // or initialization, and therefore cannot alter guest rendering.
        let overlaps_translated_scanout = |address: u64, width: u32, height: u32| {
            target.span.is_some_and(|span| {
                self.resident_guest_textures
                    .overlapping(&span, SpanQueryOrder::ResidentFirst)
                    .into_iter()
                    .filter_map(|alias| self.resident_guest_textures.get(&alias))
                    .filter(|resident| resident.serves_reads())
                    .any(|resident| {
                        resident.descriptor.address == address
                            && resident.descriptor.width == width
                            && resident.descriptor.height == height
                    })
            })
        };
        target.direct_scanout
            || self
                .direct_scanout_textures
                .contains_key(&target.texture.address)
            || self
                .pending_flip
                .zip(self.display)
                .is_some_and(|(flip, display)| {
                    flip.address == target.texture.address
                        || overlaps_translated_scanout(flip.address, display.width, display.height)
                })
            || self.active_translated_scanout.is_some_and(|scanout| {
                scanout.address == target.texture.address
                    || overlaps_translated_scanout(scanout.address, scanout.width, scanout.height)
            })
    }

    pub(super) fn diagnostic_capture_texture(
        &self,
        target: &DecodedRenderTarget,
    ) -> Option<Texture> {
        self.direct_scanout_textures
            .get(&target.texture.address)
            .filter(|scanout| scanout.descriptor == target.texture)
            .map(|scanout| scanout.texture.clone())
            .or_else(|| {
                self.resident_guest_textures
                    .get(&target.texture)
                    .filter(|resident| resident.serves_reads())
                    .map(|resident| resident.texture.clone())
            })
    }

    pub(super) fn mark_direct_scanout_initialized(&mut self, address: u64) {
        if let Some(scanout) = self.direct_scanout_textures.get_mut(&address) {
            scanout.initialized = true;
        }
    }

    pub(super) fn capture_scanout_before_pixel_shader(
        &mut self,
        draw: &IndexedDraw,
        targets: &[DecodedRenderTarget],
    ) -> Result<(), String> {
        let shader_selected = diagnostic_dump_target_before_pixel_shader()
            .is_some_and(|pixel_shader| draw.pixel_shader_address == pixel_shader);
        let glyph_selected = diagnostic_dump_glyph_draw()
            && draw
                .pixel_shader_evaluation
                .as_ref()
                .is_some_and(|evaluation| evaluation_samples_glyph_atlas(evaluation));
        if (!shader_selected && !glyph_selected)
            || draw.sequence < diagnostic_dump_target_minimum_draw()
            || diagnostic_dump_target_index_count().is_some_and(|count| count != draw.index_count)
        {
            return Ok(());
        }
        let Some((descriptor, previous_writer, texture)) = targets.iter().find_map(|target| {
            let previous_writer = diagnostic_scanout_writer(target.texture.address)?;
            let texture = self.diagnostic_capture_texture(target)?;
            Some((target.texture, previous_writer, texture))
        }) else {
            return Ok(());
        };
        let Some(directory) = self.capture_directory.clone() else {
            eprintln!(
                "[REMu][METAL][PRE-PS-TARGET] draw={} ps={:#x} skipped: REMU_CAPTURE_DIR is unset",
                draw.sequence, draw.pixel_shader_address
            );
            return Ok(());
        };
        static DUMPED: AtomicBool = AtomicBool::new(false);
        if DUMPED
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return Ok(());
        }

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
        let bytes = read_texture_linear(&device, &queue, &texture, descriptor)?;
        let fingerprint = guest_frame_fingerprint(&bytes);
        let nonzero = bytes.iter().filter(|&&value| value != 0).count();
        let Some(bgra) = native_texture_capture_to_bgra(&bytes, descriptor) else {
            return Err(format!(
                "cannot convert pre-pixel-shader target format {:?} to BGRA",
                descriptor.pixel_format
            ));
        };
        let path = capture_guest_frame(&directory, &bgra, descriptor, fingerprint)?;
        eprintln!(
            "[REMu][METAL][PRE-PS-TARGET] draw={} es={:#x} ps={:#x} address={:#x} {}x{} format={:?} bytes={} nonzero={} fingerprint={fingerprint:016x} previous={previous_writer:?} captured={}",
            draw.sequence,
            draw.export_shader_address,
            draw.pixel_shader_address,
            descriptor.address,
            descriptor.width,
            descriptor.height,
            descriptor.pixel_format,
            bytes.len(),
            nonzero,
            path.display(),
        );
        Ok(())
    }

    pub(super) fn capture_guest_frame_if_changed(&mut self, descriptor: GuestTextureDescriptor) {
        if !self.periodic_frame_capture {
            return;
        }
        let Some(directory) = self.capture_directory.as_deref() else {
            return;
        };
        if guest_frame_is_fully_black(&self.guest_bgra_bytes) {
            // A compositor clear can land in the private scanout snapshot before the cover draw
            // finishes. Defer the capture interval until a later present produces real pixels.
            return;
        }
        if !begin_capture_attempt_if_due(
            &mut self.last_guest_capture,
            Instant::now(),
            capture_interval(),
        ) {
            return;
        }
        let fingerprint = guest_frame_fingerprint(&self.guest_bgra_bytes);
        let captured_frame = (
            descriptor.address,
            descriptor.width,
            descriptor.height,
            descriptor.tile_mode,
            fingerprint,
        );
        if self.captured_guest_frames.len() >= 128
            || !self.captured_guest_frames.insert(captured_frame)
        {
            return;
        }
        match capture_guest_frame(directory, &self.guest_bgra_bytes, descriptor, fingerprint) {
            Ok(path) => eprintln!("[REMu][METAL] captured frame {}", path.display()),
            Err(error) => eprintln!("[REMu][METAL] could not capture frame: {error}"),
        }
    }

    pub(super) fn capture_translated_frame_if_changed(
        &mut self,
        descriptor: GuestTextureDescriptor,
        native_bytes: &[u8],
    ) {
        if !self.periodic_frame_capture
            || self.capture_directory.is_none()
            || self
                .last_guest_capture
                .is_some_and(|last| last.elapsed() < capture_interval())
        {
            return;
        }
        let pixel_count = u64::from(descriptor.width)
            .checked_mul(u64::from(descriptor.height))
            .and_then(|pixels| usize::try_from(pixels).ok());
        let Some(bgra_length) = pixel_count.and_then(|pixels| pixels.checked_mul(4)) else {
            return;
        };
        let Some(bgra) = native_texture_capture_to_bgra(native_bytes, descriptor) else {
            return;
        };
        debug_assert_eq!(bgra.len(), bgra_length);
        if guest_frame_is_fully_black(&bgra) {
            return;
        }
        self.guest_bgra_bytes = bgra;
        self.capture_guest_frame_if_changed(descriptor);
    }
}
impl TextureReadbackLayout {
    pub(super) fn new(width: u32, height: u32, bytes_per_pixel: u32) -> Result<Self, String> {
        if width == 0 || height == 0 || bytes_per_pixel == 0 {
            return Err(format!(
                "invalid Metal texture readback extent {width}x{height} at {bytes_per_pixel} bytes per pixel"
            ));
        }
        let packed_bytes_per_row = u64::from(width)
            .checked_mul(u64::from(bytes_per_pixel))
            .ok_or("Metal texture readback row size overflows")?;
        let staging_bytes_per_row = packed_bytes_per_row
            .checked_add(METAL_TEXTURE_BUFFER_ROW_ALIGNMENT - 1)
            .map(|value| value & !(METAL_TEXTURE_BUFFER_ROW_ALIGNMENT - 1))
            .ok_or("Metal texture readback aligned row size overflows")?;
        let staging_length = staging_bytes_per_row
            .checked_mul(u64::from(height))
            .ok_or("Metal texture readback staging size overflows")?;
        Ok(Self {
            packed_bytes_per_row,
            staging_bytes_per_row,
            staging_length,
        })
    }

    pub(super) fn packed_length(self, height: u32) -> Result<usize, String> {
        self.packed_bytes_per_row
            .checked_mul(u64::from(height))
            .and_then(|length| usize::try_from(length).ok())
            .ok_or_else(|| "Metal texture readback packed size overflows".to_owned())
    }
}

impl PendingTextureReadback {
    // Every source writer must already be committed. The presenter either completes guest batches
    // first or relies on this queue's serial order, so the blit cannot overtake its producer.
    pub(super) fn encode(
        device: &Device,
        queue: &metal::CommandQueue,
        texture: &Texture,
        descriptor: GuestTextureDescriptor,
    ) -> Result<Self, String> {
        Self::encode_mip(device, queue, texture, descriptor, 0)
    }

    pub(super) fn encode_mip(
        device: &Device,
        queue: &metal::CommandQueue,
        texture: &Texture,
        descriptor: GuestTextureDescriptor,
        mip_level: u32,
    ) -> Result<Self, String> {
        validate_texture_readback_source(texture, descriptor)?;
        Self::encode_mip_slice(device, queue, texture, descriptor, mip_level, 0)
    }

    pub(super) fn encode_array_slice(
        device: &Device,
        queue: &metal::CommandQueue,
        texture: &Texture,
        descriptor: GuestTextureDescriptor,
        slice: u32,
    ) -> Result<Self, String> {
        validate_texture_array_readback_source(texture, descriptor, slice)?;
        Self::encode_mip_slice(device, queue, texture, descriptor, 0, slice)
    }

    fn encode_mip_slice(
        device: &Device,
        queue: &metal::CommandQueue,
        texture: &Texture,
        descriptor: GuestTextureDescriptor,
        mip_level: u32,
        slice: u32,
    ) -> Result<Self, String> {
        if u64::from(mip_level) >= texture.mipmap_level_count() {
            return Err(format!(
                "Metal texture has {} mip levels, cannot read level {mip_level}",
                texture.mipmap_level_count()
            ));
        }
        let width = (descriptor.width >> mip_level).max(1);
        let height = (descriptor.height >> mip_level).max(1);
        let layout = TextureReadbackLayout::new(width, height, descriptor.bytes_per_pixel)?;
        let staging = device.new_buffer(
            layout.staging_length,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        staging.set_label("REMu texture readback staging");
        let command_buffer = queue.new_command_buffer().to_owned();
        command_buffer.set_label("REMu texture readback");
        let encoder = command_buffer.new_blit_command_encoder();
        encoder.set_label("REMu texture-to-buffer readback blit");
        encoder.copy_from_texture_to_buffer(
            texture,
            u64::from(slice),
            u64::from(mip_level),
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize::new(u64::from(width), u64::from(height), 1),
            &staging,
            0,
            layout.staging_bytes_per_row,
            layout.staging_length,
            MTLBlitOption::None,
        );
        encoder.end_encoding();
        command_buffer.commit();
        Ok(Self {
            command_buffer,
            staging,
            _source: texture.clone(),
            layout,
            height,
        })
    }

    pub(super) fn finish(self) -> Result<Vec<u8>, String> {
        settle(&self.command_buffer, "Metal texture readback")?;
        let packed_row = usize::try_from(self.layout.packed_bytes_per_row)
            .map_err(|_| "Metal texture readback packed row does not fit usize")?;
        let staging_row = usize::try_from(self.layout.staging_bytes_per_row)
            .map_err(|_| "Metal texture readback staging row does not fit usize")?;
        let staging_length = usize::try_from(self.layout.staging_length)
            .map_err(|_| "Metal texture readback staging size does not fit usize")?;
        let packed_length = self.layout.packed_length(self.height)?;
        // The shared buffer remains owned by `self` and its completed blit wrote exactly
        // `staging_length` bytes; a non-CPU-visible mapping yields a zero-length window.
        let staging_bytes = super::MetalBytes::new(&self.staging);
        let staged = staging_bytes.slice(0, staging_length).ok_or(
            "Metal shared texture readback buffer has no CPU mapping or is smaller than declared",
        )?;
        let mut packed = vec![0_u8; packed_length];
        for row in 0..usize::try_from(self.height).map_err(|_| "height does not fit usize")? {
            let source = row * staging_row;
            let destination = row * packed_row;
            packed[destination..destination + packed_row]
                .copy_from_slice(&staged[source..source + packed_row]);
        }
        Ok(packed)
    }
}

pub(super) fn native_texture_capture_to_bgra(
    native_bytes: &[u8],
    descriptor: GuestTextureDescriptor,
) -> Option<Vec<u8>> {
    let pixel_count = u64::from(descriptor.width)
        .checked_mul(u64::from(descriptor.height))
        .and_then(|pixels| usize::try_from(pixels).ok())?;
    let mut bgra = vec![0; pixel_count.checked_mul(4)?];
    match descriptor.pixel_format {
        MTLPixelFormat::RGBA8Unorm => {
            if native_bytes.len() < bgra.len() {
                return None;
            }
            for (source, destination) in native_bytes.chunks_exact(4).zip(bgra.chunks_exact_mut(4))
            {
                destination.copy_from_slice(&[
                    linear_unorm8_to_srgb_unorm8(source[2]),
                    linear_unorm8_to_srgb_unorm8(source[1]),
                    linear_unorm8_to_srgb_unorm8(source[0]),
                    source[3],
                ]);
            }
        }
        MTLPixelFormat::RGBA8Unorm_sRGB => {
            if native_bytes.len() < bgra.len() {
                return None;
            }
            for (source, destination) in native_bytes.chunks_exact(4).zip(bgra.chunks_exact_mut(4))
            {
                destination.copy_from_slice(&[source[2], source[1], source[0], source[3]]);
            }
        }
        MTLPixelFormat::BGRA8Unorm => {
            if native_bytes.len() < bgra.len() {
                return None;
            }
            for (source, destination) in native_bytes.chunks_exact(4).zip(bgra.chunks_exact_mut(4))
            {
                destination.copy_from_slice(&[
                    linear_unorm8_to_srgb_unorm8(source[0]),
                    linear_unorm8_to_srgb_unorm8(source[1]),
                    linear_unorm8_to_srgb_unorm8(source[2]),
                    source[3],
                ]);
            }
        }
        MTLPixelFormat::BGRA8Unorm_sRGB => {
            if native_bytes.len() < bgra.len() {
                return None;
            }
            let byte_count = bgra.len();
            bgra.copy_from_slice(&native_bytes[..byte_count]);
        }
        MTLPixelFormat::R16Unorm | MTLPixelFormat::Depth16Unorm => {
            if native_bytes.len() < pixel_count.checked_mul(2)? {
                return None;
            }
            for (source, destination) in native_bytes.chunks_exact(2).zip(bgra.chunks_exact_mut(4))
            {
                let value = linear_unorm8_to_srgb_unorm8(source[1]);
                destination.copy_from_slice(&[value, value, value, 255]);
            }
        }
        MTLPixelFormat::RGBA16Float => {
            if native_bytes.len() < pixel_count.checked_mul(8)? {
                return None;
            }
            for (source, destination) in native_bytes.chunks_exact(8).zip(bgra.chunks_exact_mut(4))
            {
                let red = u16::from_le_bytes([source[0], source[1]]);
                let green = u16::from_le_bytes([source[2], source[3]]);
                let blue = u16::from_le_bytes([source[4], source[5]]);
                let alpha = u16::from_le_bytes([source[6], source[7]]);
                destination.copy_from_slice(&[
                    half_float_to_srgb_unorm8(blue),
                    half_float_to_srgb_unorm8(green),
                    half_float_to_srgb_unorm8(red),
                    half_float_to_unorm8(alpha),
                ]);
            }
        }
        MTLPixelFormat::RG11B10Float => {
            if native_bytes.len() < bgra.len() {
                return None;
            }
            for (source, destination) in native_bytes.chunks_exact(4).zip(bgra.chunks_exact_mut(4))
            {
                let packed = u32::from_le_bytes(source.try_into().expect("four-byte pixel"));
                let red = packed & 0x7ff;
                let green = (packed >> 11) & 0x7ff;
                let blue = (packed >> 22) & 0x3ff;
                destination.copy_from_slice(&[
                    unsigned_float_to_srgb_unorm8(blue, 5),
                    unsigned_float_to_srgb_unorm8(green, 6),
                    unsigned_float_to_srgb_unorm8(red, 6),
                    255,
                ]);
            }
        }
        MTLPixelFormat::RGB10A2Unorm => {
            if native_bytes.len() < bgra.len() {
                return None;
            }
            // RGB10A2 VideoOut is display-referred Unorm (same as the host Unorm present path).
            // Do not apply a second linear→sRGB encode here.
            for (source, destination) in native_bytes.chunks_exact(4).zip(bgra.chunks_exact_mut(4))
            {
                let packed = u32::from_le_bytes(source.try_into().expect("four-byte pixel"));
                let red = ((packed & 0x3ff) * 255 + 511) / 1023;
                let green = (((packed >> 10) & 0x3ff) * 255 + 511) / 1023;
                let blue = (((packed >> 20) & 0x3ff) * 255 + 511) / 1023;
                let alpha = ((packed >> 30) & 0x3) * 85;
                destination.copy_from_slice(&[
                    u8::try_from(blue).expect("10-bit blue scales"),
                    u8::try_from(green).expect("10-bit green scales"),
                    u8::try_from(red).expect("10-bit red scales"),
                    u8::try_from(alpha).expect("2-bit alpha scales"),
                ]);
            }
        }
        _ => return None,
    }
    Some(bgra)
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct DiagnosticTextureNumericRange {
    pub(super) component_count: usize,
    pub(super) minimum: [Option<f64>; 4],
    pub(super) maximum: [Option<f64>; 4],
    pub(super) non_finite: [usize; 4],
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct DiagnosticTextureDepthPlaneSummary {
    pub(super) depth_plane: usize,
    pub(super) byte_count: usize,
    pub(super) nonzero_bytes: usize,
    pub(super) fingerprint: u64,
    pub(super) numeric: Option<DiagnosticTextureNumericRange>,
}

fn diagnostic_texture_numeric_components(
    pixel: &[u8],
    pixel_format: MTLPixelFormat,
) -> Option<([f64; 4], usize)> {
    let unorm8 = |value: u8| f64::from(value) / f64::from(u8::MAX);
    let unorm16 = |value: u16| f64::from(value) / f64::from(u16::MAX);
    match pixel_format {
        MTLPixelFormat::RGBA8Unorm | MTLPixelFormat::RGBA8Unorm_sRGB => Some((
            [
                unorm8(*pixel.first()?),
                unorm8(*pixel.get(1)?),
                unorm8(*pixel.get(2)?),
                unorm8(*pixel.get(3)?),
            ],
            4,
        )),
        MTLPixelFormat::BGRA8Unorm | MTLPixelFormat::BGRA8Unorm_sRGB => Some((
            [
                unorm8(*pixel.get(2)?),
                unorm8(*pixel.get(1)?),
                unorm8(*pixel.first()?),
                unorm8(*pixel.get(3)?),
            ],
            4,
        )),
        MTLPixelFormat::R16Unorm | MTLPixelFormat::Depth16Unorm => {
            let value = u16::from_le_bytes([*pixel.first()?, *pixel.get(1)?]);
            Some(([unorm16(value), 0.0, 0.0, 0.0], 1))
        }
        MTLPixelFormat::RGBA16Float => {
            let component = |offset: usize| {
                pixel
                    .get(offset..offset + 2)
                    .and_then(|bytes| bytes.try_into().ok())
                    .map(u16::from_le_bytes)
                    .map(half_float_to_f32)
                    .map(f64::from)
            };
            Some((
                [component(0)?, component(2)?, component(4)?, component(6)?],
                4,
            ))
        }
        MTLPixelFormat::RG11B10Float => {
            let packed = u32::from_le_bytes(pixel.get(..4)?.try_into().ok()?);
            Some((
                [
                    f64::from(unsigned_float_to_f32(packed & 0x7ff, 6)),
                    f64::from(unsigned_float_to_f32((packed >> 11) & 0x7ff, 6)),
                    f64::from(unsigned_float_to_f32((packed >> 22) & 0x3ff, 5)),
                    0.0,
                ],
                3,
            ))
        }
        MTLPixelFormat::RGB10A2Unorm => {
            let packed = u32::from_le_bytes(pixel.get(..4)?.try_into().ok()?);
            Some((
                [
                    f64::from(packed & 0x3ff) / 1023.0,
                    f64::from((packed >> 10) & 0x3ff) / 1023.0,
                    f64::from((packed >> 20) & 0x3ff) / 1023.0,
                    f64::from((packed >> 30) & 0x3) / 3.0,
                ],
                4,
            ))
        }
        _ => None,
    }
}

fn diagnostic_texture_numeric_range(
    plane: &[u8],
    descriptor: GuestTextureDescriptor,
) -> Option<DiagnosticTextureNumericRange> {
    let bytes_per_pixel = usize::try_from(descriptor.bytes_per_pixel).ok()?;
    if bytes_per_pixel == 0 || plane.len() % bytes_per_pixel != 0 {
        return None;
    }
    let mut minimum = [None; 4];
    let mut maximum = [None; 4];
    let mut non_finite = [0_usize; 4];
    let mut component_count: Option<usize> = None;
    for pixel in plane.chunks_exact(bytes_per_pixel) {
        let (values, decoded_component_count) =
            diagnostic_texture_numeric_components(pixel, descriptor.pixel_format)?;
        if component_count
            .replace(decoded_component_count)
            .is_some_and(|count| count != decoded_component_count)
        {
            return None;
        }
        for component in 0..decoded_component_count {
            let value = values[component];
            if value.is_finite() {
                minimum[component] =
                    Some(minimum[component].map_or(value, |current: f64| current.min(value)));
                maximum[component] =
                    Some(maximum[component].map_or(value, |current: f64| current.max(value)));
            } else {
                non_finite[component] = non_finite[component].saturating_add(1);
            }
        }
    }
    Some(DiagnosticTextureNumericRange {
        component_count: component_count?,
        minimum,
        maximum,
        non_finite,
    })
}

pub(super) fn diagnostic_texture_depth_plane_summaries(
    native_bytes: &[u8],
    descriptor: GuestTextureDescriptor,
    depth: u64,
) -> Result<Vec<DiagnosticTextureDepthPlaneSummary>, String> {
    let depth =
        usize::try_from(depth).map_err(|_| "diagnostic texture depth does not fit usize")?;
    if depth == 0 {
        return Err("diagnostic texture depth is zero".to_owned());
    }
    let plane_byte_count = u64::from(descriptor.width)
        .checked_mul(u64::from(descriptor.height))
        .and_then(|pixels| pixels.checked_mul(u64::from(descriptor.bytes_per_pixel)))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or("diagnostic texture plane size overflows")?;
    if plane_byte_count == 0 {
        return Err("diagnostic texture plane size is zero".to_owned());
    }
    let expected_byte_count = plane_byte_count
        .checked_mul(depth)
        .ok_or("diagnostic texture volume size overflows")?;
    if native_bytes.len() != expected_byte_count {
        return Err(format!(
            "diagnostic texture volume has {} bytes, expected {expected_byte_count} for {}x{}x{depth} at {} bytes per pixel",
            native_bytes.len(),
            descriptor.width,
            descriptor.height,
            descriptor.bytes_per_pixel,
        ));
    }
    Ok(native_bytes
        .chunks_exact(plane_byte_count)
        .enumerate()
        .map(|(depth_plane, plane)| DiagnosticTextureDepthPlaneSummary {
            depth_plane,
            byte_count: plane.len(),
            nonzero_bytes: plane.iter().filter(|&&value| value != 0).count(),
            fingerprint: guest_frame_fingerprint(plane),
            numeric: diagnostic_texture_numeric_range(plane, descriptor),
        })
        .collect())
}

pub(super) fn validate_texture_readback_source(
    texture: &Texture,
    descriptor: GuestTextureDescriptor,
) -> Result<(), String> {
    validate_texture_readback_storage(texture, descriptor)?;
    if texture.texture_type() != MTLTextureType::D2 {
        return Err(format!(
            "Metal texture readback requires a 2D single-slice source, got {:?}",
            texture.texture_type()
        ));
    }
    Ok(())
}

pub(super) fn validate_texture_array_readback_source(
    texture: &Texture,
    descriptor: GuestTextureDescriptor,
    slice: u32,
) -> Result<(), String> {
    validate_texture_readback_storage(texture, descriptor)?;
    if texture.texture_type() != MTLTextureType::D2Array {
        return Err(format!(
            "Metal texture array readback requires a 2D-array source, got {:?}",
            texture.texture_type()
        ));
    }
    if u64::from(slice) >= texture.array_length() {
        return Err(format!(
            "Metal texture array has {} slices, cannot read slice {slice}",
            texture.array_length()
        ));
    }
    Ok(())
}

fn validate_texture_readback_storage(
    texture: &Texture,
    descriptor: GuestTextureDescriptor,
) -> Result<(), String> {
    if texture.framebuffer_only() {
        return Err("Metal framebuffer-only textures cannot be copied for readback".to_owned());
    }
    if texture.storage_mode() == MTLStorageMode::Memoryless {
        return Err("Metal memoryless textures have no stored contents to read back".to_owned());
    }
    if texture.sample_count() != 1 {
        return Err(format!(
            "Metal texture readback requires the single-sample resolve target, got {} samples",
            texture.sample_count()
        ));
    }
    if texture.width() < u64::from(descriptor.width)
        || texture.height() < u64::from(descriptor.height)
    {
        return Err(format!(
            "Metal texture readback source is smaller than its guest region: texture={}x{} descriptor={}x{}",
            texture.width(),
            texture.height(),
            descriptor.width,
            descriptor.height
        ));
    }
    if texture.pixel_format() != descriptor.pixel_format {
        return Err(format!(
            "Metal texture readback format mismatch: texture={:?} descriptor={:?}",
            texture.pixel_format(),
            descriptor.pixel_format
        ));
    }
    let expected_bytes =
        metal_uncompressed_bytes_per_pixel(descriptor.pixel_format).ok_or_else(|| {
            format!(
                "Metal texture readback has no exact uncompressed layout for {:?}",
                descriptor.pixel_format
            )
        })?;
    if descriptor.bytes_per_pixel != expected_bytes {
        return Err(format!(
            "Metal texture readback format {:?} requires {expected_bytes} bytes per pixel, got {}",
            descriptor.pixel_format, descriptor.bytes_per_pixel
        ));
    }
    Ok(())
}

pub(super) fn metal_uncompressed_bytes_per_pixel(format: MTLPixelFormat) -> Option<u32> {
    match format {
        MTLPixelFormat::R8Unorm
        | MTLPixelFormat::R8Snorm
        | MTLPixelFormat::R8Uint
        | MTLPixelFormat::R8Sint
        | MTLPixelFormat::R8Unorm_sRGB
        | MTLPixelFormat::Stencil8 => Some(1),
        MTLPixelFormat::R16Unorm
        | MTLPixelFormat::R16Snorm
        | MTLPixelFormat::R16Uint
        | MTLPixelFormat::R16Sint
        | MTLPixelFormat::R16Float
        | MTLPixelFormat::RG8Unorm
        | MTLPixelFormat::RG8Snorm
        | MTLPixelFormat::RG8Uint
        | MTLPixelFormat::RG8Sint
        | MTLPixelFormat::RG8Unorm_sRGB
        | MTLPixelFormat::Depth16Unorm => Some(2),
        MTLPixelFormat::R32Uint
        | MTLPixelFormat::R32Sint
        | MTLPixelFormat::R32Float
        | MTLPixelFormat::RG16Unorm
        | MTLPixelFormat::RG16Snorm
        | MTLPixelFormat::RG16Uint
        | MTLPixelFormat::RG16Sint
        | MTLPixelFormat::RG16Float
        | MTLPixelFormat::RG11B10Float
        | MTLPixelFormat::RGB10A2Unorm
        | MTLPixelFormat::BGR10A2Unorm
        | MTLPixelFormat::RGBA8Unorm
        | MTLPixelFormat::RGBA8Snorm
        | MTLPixelFormat::RGBA8Uint
        | MTLPixelFormat::RGBA8Sint
        | MTLPixelFormat::RGBA8Unorm_sRGB
        | MTLPixelFormat::BGRA8Unorm
        | MTLPixelFormat::BGRA8Unorm_sRGB
        | MTLPixelFormat::RGB9E5Float
        | MTLPixelFormat::Depth32Float => Some(4),
        MTLPixelFormat::RG32Uint
        | MTLPixelFormat::RG32Sint
        | MTLPixelFormat::RG32Float
        | MTLPixelFormat::RGBA16Unorm
        | MTLPixelFormat::RGBA16Snorm
        | MTLPixelFormat::RGBA16Uint
        | MTLPixelFormat::RGBA16Sint
        | MTLPixelFormat::RGBA16Float => Some(8),
        MTLPixelFormat::RGBA32Uint | MTLPixelFormat::RGBA32Sint | MTLPixelFormat::RGBA32Float => {
            Some(16)
        }
        _ => None,
    }
}

pub(super) fn read_texture_linear(
    device: &Device,
    queue: &metal::CommandQueue,
    texture: &Texture,
    descriptor: GuestTextureDescriptor,
) -> Result<Vec<u8>, String> {
    PendingTextureReadback::encode(device, queue, texture, descriptor)?.finish()
}

pub(super) fn read_texture_mip_linear(
    device: &Device,
    queue: &metal::CommandQueue,
    texture: &Texture,
    descriptor: GuestTextureDescriptor,
    mip_level: u32,
) -> Result<Vec<u8>, String> {
    PendingTextureReadback::encode_mip(device, queue, texture, descriptor, mip_level)?.finish()
}

pub(super) fn read_texture_array_linear(
    device: &Device,
    queue: &metal::CommandQueue,
    texture: &Texture,
    descriptor: GuestTextureDescriptor,
) -> Result<Vec<u8>, String> {
    if texture.texture_type() != MTLTextureType::D2Array {
        return Err(format!(
            "Metal array readback requires a 2D-array texture, got {:?}",
            texture.texture_type()
        ));
    }
    let slice_count = u32::try_from(texture.array_length())
        .map_err(|_| "Metal texture array length does not fit u32")?;
    let slice_length = TextureReadbackLayout::new(
        descriptor.width,
        descriptor.height,
        descriptor.bytes_per_pixel,
    )?
    .packed_length(descriptor.height)?;
    let mut packed = Vec::with_capacity(
        slice_length
            .checked_mul(
                usize::try_from(slice_count)
                    .map_err(|_| "Metal texture array length does not fit usize")?,
            )
            .ok_or("Metal texture array packed size overflows")?,
    );
    for slice in 0..slice_count {
        packed.extend(
            PendingTextureReadback::encode_array_slice(device, queue, texture, descriptor, slice)?
                .finish()?,
        );
    }
    Ok(packed)
}

/// Suppresses every resident-texture writeback into guest memory.
///
/// A writeback republishes host-captured bytes into a guest range whose tenancy may have
/// changed since capture — `free()` never touches a mapping, so no liveness check can see the
/// change. Bounding publication to `gpu_authored` removes most of that exposure; this switch
/// removes all of it, so a soak can still attribute any residual corruption to this family.
fn texture_writeback_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var_os("REMU_DISABLE_TEXTURE_WRITEBACK").is_some_and(|value| value == "1")
    })
}

/// Restores publication for the triggers only REmu's own residency bookkeeping raises.
///
/// Those publications are off by default because nothing in the guest asked for them. The switch
/// exists so a run can be compared against the old behaviour without rebuilding.
fn internal_writeback_publication_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_PUBLISH_INTERNAL_WRITEBACKS").is_some_and(|value| value == "1")
    })
}

/// What made REmu consider copying a resident's Metal pixels into guest memory.
///
/// Hardware never performs this copy: a GPU write lands in unified memory at draw time, and memory
/// the guest recycles simply loses its old contents. Every publication here is REmu re-staging
/// bytes on its own timeline into a range whose tenancy it cannot verify, because a guest `free()`
/// touches no mapping and raises no GPU event. A trigger therefore admits a publication only when
/// it names a guest-observable read of those exact bytes; a trigger raised by residency
/// bookkeeping has no such reader and drops the pixels instead.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WritebackTrigger {
    /// A guest DMA_DATA packet reads this range, so the packet itself proves it is live.
    DmaSource,
    /// The flip target VideoOut presents from, which is read out of guest memory.
    FlipTarget,
    /// The host texture budget evicted the allocation. Hardware has no such event.
    PressureEviction,
    /// An overlapping resource is being rebuilt and would decode guest RAM.
    AliasRebuild,
    /// A layered depth array is being assembled over an overlapping resident.
    LayeredDepthAlias,
    /// A sampled SRD reinterprets a range some resident rendered into.
    SampledAlias,
}

pub(crate) const WRITEBACK_TRIGGER_COUNT: usize = 6;

/// The triggers in discriminant order, for statistics and diagnostics.
pub(crate) const WRITEBACK_TRIGGERS: [WritebackTrigger; WRITEBACK_TRIGGER_COUNT] = [
    WritebackTrigger::DmaSource,
    WritebackTrigger::FlipTarget,
    WritebackTrigger::PressureEviction,
    WritebackTrigger::AliasRebuild,
    WritebackTrigger::LayeredDepthAlias,
    WritebackTrigger::SampledAlias,
];

impl WritebackTrigger {
    pub(crate) const fn index(self) -> usize {
        self as usize
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::DmaSource => "dma-source",
            Self::FlipTarget => "flip-target",
            Self::PressureEviction => "pressure-eviction",
            Self::AliasRebuild => "alias-rebuild",
            Self::LayeredDepthAlias => "layered-depth-alias",
            Self::SampledAlias => "sampled-alias",
        }
    }

    /// Why this trigger may not write guest memory, or `None` when it may.
    fn refusal(self) -> Option<&'static str> {
        if texture_writeback_disabled() {
            return Some("REMU_DISABLE_TEXTURE_WRITEBACK=1");
        }
        match self {
            Self::DmaSource | Self::FlipTarget => None,
            Self::PressureEviction
            | Self::AliasRebuild
            | Self::LayeredDepthAlias
            | Self::SampledAlias => (!internal_writeback_publication_enabled())
                .then_some("residency bookkeeping, no guest read of these bytes"),
        }
    }

    /// Whether a publication for this trigger would reach guest memory at all.
    ///
    /// Call sites use this to skip the pipeline flush and readback ordering a publication needs.
    pub(crate) fn publishes_to_guest(self) -> bool {
        self.refusal().is_none()
    }
}

/// Reports a resident whose GPU pixels were dropped rather than published.
///
/// Rate limited per destination and trigger: a dropped writeback repeats every frame for a live
/// render target, and the interesting information is which ranges are affected, not how often.
fn trace_discarded_resident_writeback(
    resident: &ResidentGuestTexture,
    trigger: WritebackTrigger,
    refusal: &str,
) {
    static LOGGED: OnceLock<Mutex<HashSet<(u64, &'static str)>>> = OnceLock::new();
    let mut logged = LOGGED
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if logged.len() >= 256 || !logged.insert((resident.descriptor.address, trigger.label())) {
        return;
    }
    let (watch_address, watch_bytes) = resident.write_watch.range();
    eprintln!(
        "[REMu][METAL] dropped resident writeback trigger={} refused={refusal} address={:#x}+{:#x} debt={:.1}MiB/{} runs image={}x{} pitch={} tile={} format={:?} type={} layers={} mips={}",
        trigger.label(),
        watch_address,
        watch_bytes,
        resident.gpu_authored.byte_count() as f64 / (1024.0 * 1024.0),
        resident.gpu_authored.runs().len(),
        resident.descriptor.width,
        resident.descriptor.height,
        resident.descriptor.pitch,
        resident.descriptor.tile_mode,
        resident.descriptor.pixel_format,
        resident.texture_type,
        resident.layer_count,
        resident.mip_level_count,
    );
}

/// Publishes a resident's Metal pixels back into the guest bytes it owes them for.
///
/// This is the mechanism only; [`MetalPresenter::settle_resident_texture_writeback`] decides
/// whether a publication is admissible at all. Only `gpu_authored` is published: everything else
/// in the declared range is either padding the GPU never wrote or bytes another allocation — or
/// the guest heap, once the render target was freed — answers for, and a whole-span
/// read-modify-write republishes all of it.
pub(super) fn write_resident_texture_contents_to_guest(
    device: &Device,
    queue: &metal::CommandQueue,
    resident: &ResidentGuestTexture,
) -> Result<TextureWritebackOutcome, String> {
    if !resident.gpu_written() {
        return Ok(TextureWritebackOutcome::Applied);
    }
    let authored = resident.gpu_authored_offsets();
    if resident.mip_level_count == 1 {
        match resident.texture.texture_type() {
            MTLTextureType::D2 => {
                let linear =
                    read_texture_linear(device, queue, &resident.texture, resident.descriptor)?;
                return write_watched_guest_texture_linear(
                    resident.descriptor,
                    &linear,
                    authored.as_deref(),
                    &resident.write_watch,
                );
            }
            MTLTextureType::D2Array => {
                let linear = read_texture_array_linear(
                    device,
                    queue,
                    &resident.texture,
                    resident.descriptor,
                )?;
                return write_layered_resident_texture_contents_to_guest(
                    resident,
                    &linear,
                    authored.as_deref(),
                );
            }
            MTLTextureType::D3 => {
                let linear = read_texture_volume_linear(
                    device,
                    queue,
                    &resident.texture,
                    resident.descriptor,
                )?;
                return write_layered_resident_texture_contents_to_guest(
                    resident,
                    &linear,
                    authored.as_deref(),
                );
            }
            texture_type => {
                return Err(format!(
                    "resident texture writeback does not support Metal type {texture_type:?}"
                ));
            }
        }
    }
    if resident.texture_type != 9 || resident.layer_count != 1 {
        return Err(format!(
            "mip writeback requires a 2D single-layer resident, got type={} layers={}",
            resident.texture_type, resident.layer_count
        ));
    }
    let layout = mode27_mip_chain_layout(resident.descriptor, resident.mip_level_count)?;
    // The chain snapshot below seeds the published bytes; a guest store between it and the
    // publish would be reverted. Capture the fault generations here so the publish can abort.
    let entry_generations = resident.write_watch.generations();
    let baseline = snapshot_watched_guest_range(
        resident.descriptor.address,
        layout.byte_count,
        &resident.write_watch,
    )?;
    let mut raw = baseline.clone();
    let mut levels = Vec::with_capacity(layout.levels.len());
    for mip_level in 0..resident.mip_level_count {
        levels.push(read_texture_mip_linear(
            device,
            queue,
            &resident.texture,
            resident.descriptor,
            mip_level,
        )?);
    }
    encode_mode27_mip_chain_into_raw(&layout, &levels, &mut raw)?;
    write_watched_guest_range(
        resident.descriptor.address,
        &raw,
        authored.as_deref(),
        Some(&baseline),
        &resident.write_watch,
        Some(&entry_generations),
    )
}

fn write_layered_resident_texture_contents_to_guest(
    resident: &ResidentGuestTexture,
    linear: &[u8],
    authored: Option<&[(u64, u64)]>,
) -> Result<TextureWritebackOutcome, String> {
    let face_raw_bytes = usize::try_from(
        super::guest_texture_byte_count(resident.descriptor)
            .ok_or("layered guest texture face byte count overflows")?,
    )
    .map_err(|_| "layered guest texture face byte count does not fit usize")?;
    let (width, height, _, bytes_per_element) = texture_element_layout(resident.descriptor)?;
    let face_linear_bytes = width
        .checked_mul(height)
        .and_then(|elements| elements.checked_mul(bytes_per_element))
        .ok_or("layered guest texture linear face size overflows")?;
    let layer_count = usize::try_from(resident.layer_count)
        .map_err(|_| "layered guest texture layer count does not fit usize")?;
    let expected_linear_bytes = face_linear_bytes
        .checked_mul(layer_count)
        .ok_or("layered guest texture linear size overflows")?;
    if linear.len() != expected_linear_bytes {
        return Err(format!(
            "layered guest texture readback has {} bytes, expected {expected_linear_bytes}",
            linear.len()
        ));
    }
    let raw_bytes = face_raw_bytes
        .checked_mul(layer_count)
        .ok_or("layered guest texture raw size overflows")?;
    let entry_generations = resident.write_watch.generations();
    let baseline = snapshot_watched_guest_range(
        resident.descriptor.address,
        raw_bytes,
        &resident.write_watch,
    )?;
    let mut raw = baseline.clone();
    for layer in 0..layer_count {
        let linear_start = layer * face_linear_bytes;
        let raw_start = layer * face_raw_bytes;
        write_guest_texture_linear_into_raw_at_z(
            resident.descriptor,
            &linear[linear_start..linear_start + face_linear_bytes],
            &mut raw[raw_start..raw_start + face_raw_bytes],
            layer,
        )?;
    }
    write_watched_guest_range(
        resident.descriptor.address,
        &raw,
        authored,
        Some(&baseline),
        &resident.write_watch,
        Some(&entry_generations),
    )
}

pub(super) fn read_texture_volume_linear(
    device: &Device,
    queue: &metal::CommandQueue,
    texture: &Texture,
    descriptor: GuestTextureDescriptor,
) -> Result<Vec<u8>, String> {
    if texture.texture_type() != MTLTextureType::D3 {
        return Err(format!(
            "Metal volume readback requires a 3D texture, got {:?}",
            texture.texture_type()
        ));
    }
    if texture.framebuffer_only() || texture.storage_mode() == MTLStorageMode::Memoryless {
        return Err("Metal 3D texture has no readable stored contents".to_owned());
    }
    if texture.sample_count() != 1
        || texture.width() != u64::from(descriptor.width)
        || texture.height() != u64::from(descriptor.height)
        || texture.pixel_format() != descriptor.pixel_format
    {
        return Err(format!(
            "Metal volume readback mismatch: texture={}x{}x{} {:?} descriptor={}x{} {:?}",
            texture.width(),
            texture.height(),
            texture.depth(),
            texture.pixel_format(),
            descriptor.width,
            descriptor.height,
            descriptor.pixel_format,
        ));
    }
    let expected_bytes =
        metal_uncompressed_bytes_per_pixel(descriptor.pixel_format).ok_or_else(|| {
            format!(
                "Metal volume readback has no exact uncompressed layout for {:?}",
                descriptor.pixel_format
            )
        })?;
    if descriptor.bytes_per_pixel != expected_bytes {
        return Err(format!(
            "Metal volume readback format {:?} requires {expected_bytes} bytes per pixel, got {}",
            descriptor.pixel_format, descriptor.bytes_per_pixel
        ));
    }
    let layout = TextureReadbackLayout::new(
        descriptor.width,
        descriptor.height,
        descriptor.bytes_per_pixel,
    )?;
    let depth = texture.depth();
    let staging_bytes_per_image = layout.staging_length;
    let staging_length = staging_bytes_per_image
        .checked_mul(depth)
        .ok_or("Metal volume staging size overflows")?;
    let staging = device.new_buffer(
        staging_length,
        MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache,
    );
    staging.set_label("REMu 3D texture readback staging");
    let command_buffer = queue.new_command_buffer().to_owned();
    command_buffer.set_label("REMu 3D texture readback");
    let encoder = command_buffer.new_blit_command_encoder();
    encoder.set_label("REMu 3D texture-to-buffer readback blit");
    encoder.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        MTLOrigin { x: 0, y: 0, z: 0 },
        MTLSize::new(
            u64::from(descriptor.width),
            u64::from(descriptor.height),
            depth,
        ),
        &staging,
        0,
        layout.staging_bytes_per_row,
        staging_bytes_per_image,
        MTLBlitOption::None,
    );
    encoder.end_encoding();
    commit_and_settle(&command_buffer, "Metal volume readback")?;
    let packed_row = usize::try_from(layout.packed_bytes_per_row)
        .map_err(|_| "Metal volume packed row does not fit usize")?;
    let staging_row = usize::try_from(layout.staging_bytes_per_row)
        .map_err(|_| "Metal volume staging row does not fit usize")?;
    let staging_image = usize::try_from(staging_bytes_per_image)
        .map_err(|_| "Metal volume staging image does not fit usize")?;
    let staging_length = usize::try_from(staging_length)
        .map_err(|_| "Metal volume staging size does not fit usize")?;
    let height =
        usize::try_from(descriptor.height).map_err(|_| "Metal volume height does not fit usize")?;
    let depth = usize::try_from(depth).map_err(|_| "Metal volume depth does not fit usize")?;
    let packed_image = packed_row
        .checked_mul(height)
        .ok_or("Metal volume packed image size overflows")?;
    let packed_length = packed_image
        .checked_mul(depth)
        .ok_or("Metal volume packed size overflows")?;
    // The shared staging buffer remains live and its completed blit initialized the full declared
    // extent; a non-CPU-visible mapping yields a zero-length window.
    let staging_bytes = super::MetalBytes::new(&staging);
    let staged = staging_bytes.slice(0, staging_length).ok_or(
        "Metal shared volume readback buffer has no CPU mapping or is smaller than declared",
    )?;
    let mut packed = vec![0_u8; packed_length];
    for z in 0..depth {
        for y in 0..height {
            let source = z * staging_image + y * staging_row;
            let destination = z * packed_image + y * packed_row;
            packed[destination..destination + packed_row]
                .copy_from_slice(&staged[source..source + packed_row]);
        }
    }
    Ok(packed)
}

pub(super) fn guest_frame_fingerprint(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, &value| {
            (hash ^ u64::from(value)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

fn guest_frame_is_fully_black(bgra: &[u8]) -> bool {
    bgra.chunks_exact(4)
        .all(|pixel| pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0)
}

pub(super) fn capture_guest_frame(
    directory: &Path,
    bgra: &[u8],
    descriptor: GuestTextureDescriptor,
    fingerprint: u64,
) -> Result<PathBuf, String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
    let path = directory.join(format!(
        "remu-{:x}-{}x{}-tile{}-{fingerprint:016x}.ppm",
        descriptor.address, descriptor.width, descriptor.height, descriptor.tile_mode
    ));
    let file = std::fs::File::create(&path)
        .map_err(|error| format!("could not create {}: {error}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "P6\n{} {}\n255",
        descriptor.width, descriptor.height
    )
    .map_err(|error| format!("could not write {}: {error}", path.display()))?;
    let pixel_count = u64::from(descriptor.width)
        .checked_mul(u64::from(descriptor.height))
        .and_then(|pixels| usize::try_from(pixels).ok())
        .ok_or("captured frame dimensions overflow")?;
    if bgra.len() < pixel_count.saturating_mul(4) {
        return Err("captured frame buffer is truncated".to_owned());
    }
    let mut rgb = Vec::with_capacity(pixel_count.saturating_mul(3));
    for pixel in bgra.chunks_exact(4).take(pixel_count) {
        rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
    }
    writer
        .write_all(&rgb)
        .map_err(|error| format!("could not write {}: {error}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::begin_capture_attempt_if_due;
    use std::time::{Duration, Instant};

    #[test]
    fn capture_attempts_remain_throttled_after_each_due_attempt() {
        let interval = Duration::from_secs(10);
        let start = Instant::now();
        let mut last_capture = None;

        assert!(begin_capture_attempt_if_due(
            &mut last_capture,
            start,
            interval
        ));
        assert!(!begin_capture_attempt_if_due(
            &mut last_capture,
            start + Duration::from_secs(9),
            interval
        ));
        assert!(begin_capture_attempt_if_due(
            &mut last_capture,
            start + interval,
            interval
        ));
        assert!(!begin_capture_attempt_if_due(
            &mut last_capture,
            start + interval + Duration::from_secs(1),
            interval
        ));
    }
}
