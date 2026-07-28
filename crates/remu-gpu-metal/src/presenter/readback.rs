// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    DecodedRenderTarget, GuestTextureDescriptor, METAL_TEXTURE_BUFFER_ROW_ALIGNMENT,
    MetalPresenter, PendingTextureReadback, ResidentGuestTexture, TextureReadbackLayout,
    capture_interval, diagnostic_dump_target_before_pixel_shader,
    diagnostic_dump_target_minimum_draw, diagnostic_scanout_writer,
    encode_mode27_mip_chain_into_raw, guest_texture_descriptors_overlap, half_float_to_srgb_unorm8,
    half_float_to_unorm8, linear_unorm8_to_srgb_unorm8, mode27_mip_chain_layout,
    snapshot_watched_guest_range, unsigned_float_to_srgb_unorm8, write_watched_guest_range,
    write_watched_guest_texture_linear,
};
use metal::{
    Device, MTLBlitOption, MTLCommandBufferStatus, MTLOrigin, MTLPixelFormat, MTLResourceOptions,
    MTLSize, MTLStorageMode, MTLTextureType, Texture,
};
use remu_gpu::IndexedDraw;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

impl MetalPresenter {
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
            self.resident_guest_textures
                .values()
                .filter(|resident| {
                    resident.descriptor.address == address
                        && resident.descriptor.width == width
                        && resident.descriptor.height == height
                })
                .any(|resident| {
                    guest_texture_descriptors_overlap(resident.descriptor, target.texture)
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
        let Some(pixel_shader) = diagnostic_dump_target_before_pixel_shader() else {
            return Ok(());
        };
        if draw.pixel_shader_address != pixel_shader
            || draw.sequence < diagnostic_dump_target_minimum_draw()
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
        let Some(directory) = self.capture_directory.as_deref() else {
            return;
        };
        if self
            .last_guest_capture
            .is_some_and(|last| last.elapsed() < capture_interval())
        {
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
        self.last_guest_capture = Some(Instant::now());
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
        if self.capture_directory.is_none()
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
        self.command_buffer.wait_until_completed();
        if self.command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(format!(
                "Metal texture readback ended in {:?}",
                self.command_buffer.status()
            ));
        }
        let packed_row = usize::try_from(self.layout.packed_bytes_per_row)
            .map_err(|_| "Metal texture readback packed row does not fit usize")?;
        let staging_row = usize::try_from(self.layout.staging_bytes_per_row)
            .map_err(|_| "Metal texture readback staging row does not fit usize")?;
        let staging_length = usize::try_from(self.layout.staging_length)
            .map_err(|_| "Metal texture readback staging size does not fit usize")?;
        let packed_length = self.layout.packed_length(self.height)?;
        let contents = self.staging.contents().cast::<u8>();
        if contents.is_null() {
            return Err("Metal shared texture readback buffer has no CPU mapping".to_owned());
        }
        // SAFETY: The shared buffer remains owned by `self`, its completed blit wrote exactly
        // `staging_length` bytes, and `contents` is non-null for this CPU-visible storage mode.
        let staged = unsafe { std::slice::from_raw_parts(contents, staging_length) };
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
        _ => return None,
    }
    Some(bgra)
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

pub(super) fn write_resident_texture_contents_to_guest(
    device: &Device,
    queue: &metal::CommandQueue,
    resident: &ResidentGuestTexture,
) -> Result<(), String> {
    if resident.mip_level_count == 1 {
        let linear = read_texture_linear(device, queue, &resident.texture, resident.descriptor)?;
        return write_watched_guest_texture_linear(
            resident.descriptor,
            &linear,
            &resident.write_watch,
        );
    }
    if resident.texture_type != 9 || resident.layer_count != 1 {
        return Err(format!(
            "mip writeback requires a 2D single-layer resident, got type={} layers={}",
            resident.texture_type, resident.layer_count
        ));
    }
    let layout = mode27_mip_chain_layout(resident.descriptor, resident.mip_level_count)?;
    let mut raw = snapshot_watched_guest_range(
        resident.descriptor.address,
        layout.byte_count,
        &resident.write_watch,
    )?;
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
    write_watched_guest_range(resident.descriptor.address, &raw, &resident.write_watch)
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
    command_buffer.commit();
    command_buffer.wait_until_completed();
    if command_buffer.status() != MTLCommandBufferStatus::Completed {
        return Err(format!(
            "Metal volume readback ended in {:?}",
            command_buffer.status()
        ));
    }
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
    let contents = staging.contents().cast::<u8>();
    if contents.is_null() {
        return Err("Metal shared volume readback buffer has no CPU mapping".to_owned());
    }
    // SAFETY: the shared staging buffer remains live, and the completed blit initialized its
    // complete declared extent.
    let staged = unsafe { std::slice::from_raw_parts(contents, staging_length) };
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
