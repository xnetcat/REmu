// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CachedSampledTexture, DecodedDepthTarget, DecodedImageDescriptor, DecodedRenderTarget,
    DirectScanoutTexture, GuestTextureDescriptor, GuestTextureWriteWatch, MIP_UPLOAD_BYTES,
    MIP_UPLOAD_COUNT, MIP_UPLOAD_NANOSECONDS, MetalPresenter, Mode27WorkOrigin,
    MultisampleResidentTexture, MultisampleTextureKey, RESIDENT_DEPTH_SAMPLE_HITS,
    RESIDENT_TEXTURE_ACQUIRE_INVALIDATIONS, ResidentGuestTexture, SAMPLED_ALIAS_WRITEBACK_BYTES,
    SAMPLED_ALIAS_WRITEBACK_COUNT, SAMPLED_ALIAS_WRITEBACK_NANOSECONDS,
    SAMPLED_TEXTURE_CACHE_ACQUIRE_INVALIDATIONS, SAMPLED_TEXTURE_CACHE_CAPACITY,
    SAMPLED_TEXTURE_CACHE_CAPACITY_EVICTIONS, SAMPLED_TEXTURE_CACHE_DIRTY_RELOADS,
    SAMPLED_TEXTURE_CACHE_HITS, SAMPLED_TEXTURE_CACHE_MISSES, SampledTextureKey,
    ScanoutSnapshotTexture, TEXTURE_SNAPSHOT_BYTES, TEXTURE_SNAPSHOT_COUNT,
    TEXTURE_SNAPSHOT_NANOSECONDS, UploadedImageTexture, WriteWatchRefresh,
    canonical_texture_bindings, compressed_texture_block_extent, copy_linear_to_tiled,
    copy_tiled_to_linear_at_z, decode_image_descriptor, decode_mode27_mip_chain_from_raw,
    fallback_image_descriptor, guest_frame_fingerprint, guest_ranges_overlap,
    guest_texture_byte_count, guest_texture_descriptors_have_same_layout,
    guest_texture_descriptors_overlap, guest_texture_descriptors_share_component_reordered_storage,
    guest_texture_descriptors_share_depth_sample_storage, guest_texture_descriptors_share_storage,
    image_resource_descriptor_is_null, initial_depth_clear_for_target, is_storage_image_opcode,
    is_writable_image_opcode, mode27_mip_chain_layout, record_mode27_work_detail,
    record_texture_dirty_work_detail, record_texture_snapshot_work_detail, record_texture_work,
    render_resolve_candidate_matches, sampled_alias_descriptor,
    sampler_uses_normalized_coordinates, texture_element_layout, trace_image_descriptors,
    trace_target_sampling, write_resident_texture_contents_to_guest,
};
use metal::foreign_types::ForeignType;
use metal::{
    Device, MTLCommandBufferStatus, MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize,
    MTLStorageMode, MTLStoreAction, MTLTextureType, MTLTextureUsage, NSRange, RenderPassDescriptor,
    Texture, TextureDescriptor,
};
use objc::runtime::Object;
use objc::{msg_send, sel, sel_impl};
use remu_gpu::GpuMemoryAcquire;
use remu_memory::{
    CpuWriteWatch, CpuWriteWatchRetirement, guest_shared_backing_alias_segments,
    guest_shared_mapping_identity, read_guest_shared_backing, write_guest_shared_backing,
};
use remu_shader::{ImageBinding, ShaderEvaluation, image_is_comparison};
use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

impl MetalPresenter {
    pub(super) fn ensure_resident_texture(
        &mut self,
        device: &Device,
        descriptor: GuestTextureDescriptor,
    ) -> Result<Texture, String> {
        self.ensure_resident_texture_with_initial_contents(device, descriptor, true)
    }

    pub(super) fn ensure_resident_texture_with_initial_contents(
        &mut self,
        device: &Device,
        descriptor: GuestTextureDescriptor,
        upload_guest_contents: bool,
    ) -> Result<Texture, String> {
        if let Some(resident) =
            self.resident_guest_textures
                .get_mut(&descriptor)
                .filter(|resident| {
                    resident.texture_type == 9
                        && resident.layer_count == 1
                        && resident.mip_level_count >= 1
                })
        {
            match refresh_resident_texture_write_watch(resident)? {
                WriteWatchRefresh::Active
                | WriteWatchRefresh::Reprotecting
                | WriteWatchRefresh::Renewed { .. } => return Ok(resident.texture.clone()),
                WriteWatchRefresh::Replaced => {}
            }
        }
        self.resident_guest_textures.remove(&descriptor);
        let shared_resident = self
            .resident_guest_textures
            .values()
            .find(|resident| {
                resident.texture_type == 9
                    && resident.layer_count == 1
                    && resident.mip_level_count >= 1
                    && guest_texture_descriptors_share_storage(resident.descriptor, descriptor)
            })
            .map(|resident| resident.descriptor);
        if let Some(shared_resident) = shared_resident {
            let resident = self
                .resident_guest_textures
                .get_mut(&shared_resident)
                .expect("selected shared resident texture exists");
            match refresh_resident_texture_write_watch(resident)? {
                WriteWatchRefresh::Active
                | WriteWatchRefresh::Reprotecting
                | WriteWatchRefresh::Renewed { .. } => return Ok(resident.texture.clone()),
                WriteWatchRefresh::Replaced => {
                    self.resident_guest_textures.remove(&shared_resident);
                }
            }
        }
        let mut aliases = self
            .resident_guest_textures
            .values()
            .filter(|resident| guest_texture_descriptors_overlap(resident.descriptor, descriptor))
            .map(|resident| (resident.content_version, resident.descriptor))
            .collect::<Vec<_>>();
        if !aliases.is_empty() {
            self.wait_for_guest_command_batches()?;
            self.multisample_textures
                .retain(|key, _| !guest_texture_descriptors_overlap(key.descriptor, descriptor));
            let queue = self
                .command_queue
                .as_ref()
                .ok_or("Metal command queue is unavailable")?
                .clone();
            aliases.sort_by_key(|(version, _)| *version);
            for (_, alias) in aliases {
                let refresh = refresh_resident_texture_write_watch(
                    self.resident_guest_textures
                        .get_mut(&alias)
                        .expect("collected resident alias exists"),
                )?;
                if refresh == WriteWatchRefresh::Reprotecting {
                    return Err(format!(
                        "resident alias mapping {:#x} is being reprotected",
                        alias.address
                    ));
                }
                let resident = self
                    .resident_guest_textures
                    .remove(&alias)
                    .expect("collected resident alias exists");
                if refresh != WriteWatchRefresh::Replaced && resident.gpu_written {
                    let alias_ranges = resident.write_watch.alias_ranges();
                    retire_sampled_textures_overlapping_writeback(
                        &mut self.sampled_texture_cache,
                        &alias_ranges,
                    );
                    retire_resident_textures_overlapping_writeback(
                        &mut self.resident_guest_textures,
                        &alias_ranges,
                        resident.descriptor,
                        resident.content_version,
                    );
                    retire_multisample_textures_overlapping_writeback(
                        &mut self.multisample_textures,
                        &alias_ranges,
                    );
                    write_resident_texture_contents_to_guest(device, &queue, &resident)?;
                }
            }
            retire_overlapping_sampled_textures(&mut self.sampled_texture_cache, descriptor);
        }
        ensure_resident_texture_in_map(
            device,
            descriptor,
            &mut self.resident_guest_textures,
            upload_guest_contents,
        )
    }

    pub(super) fn ensure_mipmapped_render_target_texture(
        &mut self,
        device: &Device,
        target: DecodedRenderTarget,
    ) -> Result<Texture, String> {
        if target.texture_type != 9
            || target.layer_count != 1
            || target.mip_level_count <= 1
            || target.sample_count != 1
        {
            return Err(format!(
                "mipmapped render target requires a single-sample 2D resource, got type={} layers={} levels={} samples={}",
                target.texture_type,
                target.layer_count,
                target.mip_level_count,
                target.sample_count
            ));
        }
        if let Some(resident) = self
            .resident_guest_textures
            .get_mut(&target.texture)
            .filter(|resident| {
                resident.texture_type == 9
                    && resident.layer_count == 1
                    && resident.mip_level_count >= target.mip_level_count
            })
        {
            match refresh_resident_texture_write_watch(resident)? {
                WriteWatchRefresh::Active
                | WriteWatchRefresh::Reprotecting
                | WriteWatchRefresh::Renewed { .. } => return Ok(resident.texture.clone()),
                WriteWatchRefresh::Replaced => {}
            }
        }
        if self
            .resident_guest_textures
            .get(&target.texture)
            .is_some_and(|resident| resident.gpu_written)
        {
            return Err(format!(
                "cannot reinterpret GPU-written target {:#x} as a {}-level hierarchy",
                target.texture.address, target.mip_level_count
            ));
        }
        self.resident_guest_textures.remove(&target.texture);
        retire_overlapping_sampled_textures(&mut self.sampled_texture_cache, target.texture);
        let started = Instant::now();
        let layout = mode27_mip_chain_layout(target.texture, target.mip_level_count)?;
        let write_watch = watch_resident_texture(
            target.texture,
            target.texture_type,
            target.layer_count,
            target.mip_level_count,
        )?;
        let raw =
            snapshot_watched_guest_range(target.texture.address, layout.byte_count, &write_watch)?;
        let (_, linear_levels) =
            decode_mode27_mip_chain_from_raw(target.texture, target.mip_level_count, &raw)?;
        let texture = create_mipmapped_guest_texture(
            device,
            target.texture,
            target.mip_level_count,
            &linear_levels,
        )?;
        record_mode27_work_detail(
            Mode27WorkOrigin::RenderTargetCreate,
            target.texture,
            target.mip_level_count,
            layout.byte_count,
            started.elapsed(),
        );
        texture.set_label(&format!(
            "REMu {}-level guest image {:#x}",
            target.mip_level_count, target.texture.address
        ));
        self.resident_guest_textures.insert(
            target.texture,
            ResidentGuestTexture {
                texture: texture.clone(),
                component_reorder_view: None,
                descriptor: target.texture,
                texture_type: target.texture_type,
                layer_count: target.layer_count,
                mip_level_count: target.mip_level_count,
                write_watch,
                gpu_written: false,
                content_version: 0,
            },
        );
        Ok(texture)
    }

    pub(super) fn ensure_layered_render_target_texture(
        &mut self,
        device: &Device,
        target: DecodedRenderTarget,
        discard_guest_contents: bool,
    ) -> Result<Texture, String> {
        if target.texture_type == 9 && target.layer_count == 1 {
            return if target.mip_level_count == 1 {
                self.ensure_resident_texture_with_initial_contents(
                    device,
                    target.texture,
                    !discard_guest_contents,
                )
            } else {
                self.ensure_mipmapped_render_target_texture(device, target)
            };
        }
        if target.texture_type != 10 || target.layer_count <= 1 {
            return Err(format!(
                "unsupported layered target texture type={} layers={}",
                target.texture_type, target.layer_count
            ));
        }
        if target.sample_count != 1 {
            return Err("multisampled 3D color targets are not implemented".to_owned());
        }
        if target.mip_level_count != 1 || target.mip_level != 0 {
            return Err(format!(
                "mipmapped 3D color targets are not implemented (mip={}/{})",
                target.mip_level,
                target.mip_level_count - 1
            ));
        }
        if let Some(resident) = self
            .resident_guest_textures
            .get_mut(&target.texture)
            .filter(|resident| {
                resident.texture_type == target.texture_type
                    && resident.layer_count == target.layer_count
                    && resident.mip_level_count == 1
            })
        {
            match refresh_resident_texture_write_watch(resident)? {
                WriteWatchRefresh::Active
                | WriteWatchRefresh::Reprotecting
                | WriteWatchRefresh::Renewed { .. } => return Ok(resident.texture.clone()),
                WriteWatchRefresh::Replaced => {}
            }
        }
        if self
            .resident_guest_textures
            .get(&target.texture)
            .is_some_and(|resident| resident.gpu_written)
        {
            return Err(format!(
                "cannot replace GPU-written target {:#x} with a {}-layer 3D view",
                target.texture.address, target.layer_count
            ));
        }
        self.resident_guest_textures.remove(&target.texture);
        retire_overlapping_sampled_textures(&mut self.sampled_texture_cache, target.texture);
        let descriptor = DecodedImageDescriptor {
            guest: target.texture,
            destination_select: 0,
            base_mip_level: 0,
            mip_level_count: 1,
            texture_type: target.texture_type,
            layer_count: target.layer_count,
        };
        let write_watch = watch_sampled_texture(descriptor)?;
        let raw = snapshot_watched_sampled_texture_raw(descriptor, &write_watch)?;
        let linear = decode_sampled_texture_linear_from_raw(descriptor, &raw)?;
        let texture = create_layered_render_target_texture(device, descriptor, &linear)?;
        texture.set_label(&format!(
            "REMu {}-layer guest volume {:#x}",
            target.layer_count, target.texture.address
        ));
        self.resident_guest_textures.insert(
            target.texture,
            ResidentGuestTexture {
                texture: texture.clone(),
                component_reorder_view: None,
                descriptor: target.texture,
                texture_type: target.texture_type,
                layer_count: target.layer_count,
                mip_level_count: 1,
                write_watch,
                gpu_written: false,
                content_version: 0,
            },
        );
        Ok(texture)
    }

    pub(super) fn ensure_depth_render_target_texture(
        &mut self,
        device: &Device,
        depth: DecodedDepthTarget,
        minimum_width: u32,
        minimum_height: u32,
    ) -> Result<Texture, String> {
        let texture = self.ensure_resident_texture(device, depth.texture)?;
        let host_width = depth.texture.width.max(minimum_width);
        let host_height = depth.texture.height.max(minimum_height);
        if texture.width() >= u64::from(host_width) && texture.height() >= u64::from(host_height) {
            return Ok(texture);
        }

        // Unreal can bind a hierarchically sized DB surface beside a larger color target. The
        // console expands the fast-cleared depth allocation to the active framebuffer extent;
        // Metal instead requires every attachment to cover the explicit render-pass extent.
        // Promote the host depth image, initialize the extra area to a comparison-neutral value,
        // and preserve the guest-declared portion. This is the same strategy used by SharpEmu's
        // working Metal backend and prevents menus/scenes from being clipped to the small DB
        // metadata extent.
        self.wait_for_guest_command_batches()?;
        let queue = self
            .command_queue
            .as_ref()
            .ok_or("Metal command queue is unavailable")?
            .clone();
        let metal_descriptor = TextureDescriptor::new();
        metal_descriptor.set_texture_type(MTLTextureType::D2);
        metal_descriptor.set_pixel_format(depth.texture.pixel_format);
        metal_descriptor.set_width(u64::from(host_width));
        metal_descriptor.set_height(u64::from(host_height));
        metal_descriptor.set_storage_mode(MTLStorageMode::Shared);
        metal_descriptor.set_usage(
            MTLTextureUsage::RenderTarget
                | MTLTextureUsage::ShaderRead
                | MTLTextureUsage::ShaderWrite,
        );
        let promoted = device.new_texture(&metal_descriptor);
        promoted.set_label(&format!(
            "REMu promoted depth image {:#x} {}x{}",
            depth.texture.address, host_width, host_height
        ));

        let command_buffer = queue.new_command_buffer().to_owned();
        command_buffer.set_label("REMu depth-target promotion");
        let pass = RenderPassDescriptor::new();
        pass.set_render_target_width(u64::from(host_width));
        pass.set_render_target_height(u64::from(host_height));
        let attachment = pass
            .depth_attachment()
            .ok_or("Metal render pass has no depth attachment")?;
        attachment.set_texture(Some(&promoted));
        attachment.set_load_action(MTLLoadAction::Clear);
        attachment.set_store_action(MTLStoreAction::Store);
        attachment.set_clear_depth(initial_depth_clear_for_target(depth));
        let encoder = command_buffer.new_render_command_encoder(pass);
        encoder.end_encoding();

        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_texture(
            &texture,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize::new(
                texture.width().min(u64::from(host_width)),
                texture.height().min(u64::from(host_height)),
                1,
            ),
            &promoted,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
        );
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(format!(
                "depth-target promotion ended in {:?}",
                command_buffer.status()
            ));
        }

        let resident_descriptor = self
            .resident_guest_textures
            .contains_key(&depth.texture)
            .then_some(depth.texture)
            .or_else(|| {
                self.resident_guest_textures
                    .values()
                    .find(|resident| {
                        guest_texture_descriptors_share_storage(resident.descriptor, depth.texture)
                    })
                    .map(|resident| resident.descriptor)
            })
            .ok_or("promoted depth target lost its resident identity")?;
        let resident = self
            .resident_guest_textures
            .get_mut(&resident_descriptor)
            .expect("selected depth resident exists");
        resident.texture = promoted.clone();
        resident.component_reorder_view = None;
        Ok(promoted)
    }

    pub(super) fn ensure_direct_scanout_texture(
        &mut self,
        device: &Device,
        descriptor: GuestTextureDescriptor,
    ) -> Texture {
        if let Some(scanout) = self.direct_scanout_textures.get(&descriptor.address)
            && scanout.descriptor == descriptor
        {
            return scanout.texture.clone();
        }
        let metal_descriptor = TextureDescriptor::new();
        metal_descriptor.set_texture_type(MTLTextureType::D2);
        metal_descriptor.set_pixel_format(descriptor.pixel_format);
        metal_descriptor.set_width(u64::from(descriptor.width));
        metal_descriptor.set_height(u64::from(descriptor.height));
        metal_descriptor.set_storage_mode(MTLStorageMode::Private);
        metal_descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        let texture = device.new_texture(&metal_descriptor);
        self.direct_scanout_textures.insert(
            descriptor.address,
            DirectScanoutTexture {
                descriptor,
                texture: texture.clone(),
                initialized: false,
            },
        );
        texture
    }

    pub(super) fn ensure_scanout_snapshot_texture(
        &mut self,
        device: &Device,
        descriptor: GuestTextureDescriptor,
    ) -> Texture {
        if let Some(snapshot) = self.scanout_snapshot_textures.get(&descriptor.address)
            && snapshot.descriptor == descriptor
        {
            return snapshot.texture.clone();
        }
        let metal_descriptor = TextureDescriptor::new();
        metal_descriptor.set_texture_type(MTLTextureType::D2);
        metal_descriptor.set_pixel_format(descriptor.pixel_format);
        metal_descriptor.set_width(u64::from(descriptor.width));
        metal_descriptor.set_height(u64::from(descriptor.height));
        metal_descriptor.set_storage_mode(MTLStorageMode::Private);
        metal_descriptor.set_usage(MTLTextureUsage::ShaderRead);
        let texture = device.new_texture(&metal_descriptor);
        texture.set_label(&format!("REMu scanout snapshot {:#x}", descriptor.address));
        self.scanout_snapshot_textures.insert(
            descriptor.address,
            ScanoutSnapshotTexture {
                descriptor,
                texture: texture.clone(),
            },
        );
        texture
    }

    pub(super) fn ensure_multisample_texture(
        &mut self,
        device: &Device,
        descriptor: GuestTextureDescriptor,
        sample_count: u32,
        minimum_width: u32,
        minimum_height: u32,
    ) -> Result<(Texture, bool), String> {
        if sample_count <= 1 {
            return Err("multisample texture requires at least two samples".to_owned());
        }
        if !device.supports_texture_sample_count(u64::from(sample_count)) {
            return Err(format!(
                "Metal device does not support {sample_count} samples for {:?}",
                descriptor.pixel_format
            ));
        }
        let key = MultisampleTextureKey {
            descriptor,
            sample_count,
        };
        let host_width = descriptor.width.max(minimum_width);
        let host_height = descriptor.height.max(minimum_height);
        if let Some(resident) = self.multisample_textures.get(&key)
            && resident.texture.width() >= u64::from(host_width)
            && resident.texture.height() >= u64::from(host_height)
        {
            return Ok((resident.texture.clone(), !resident.initialized));
        }
        self.multisample_textures.remove(&key);
        let metal_descriptor = TextureDescriptor::new();
        metal_descriptor.set_texture_type(MTLTextureType::D2Multisample);
        metal_descriptor.set_pixel_format(descriptor.pixel_format);
        metal_descriptor.set_width(u64::from(host_width));
        metal_descriptor.set_height(u64::from(host_height));
        metal_descriptor.set_sample_count(u64::from(sample_count));
        metal_descriptor.set_storage_mode(MTLStorageMode::Private);
        metal_descriptor.set_usage(MTLTextureUsage::RenderTarget);
        let texture = device.new_texture(&metal_descriptor);
        texture.set_label(&format!(
            "REMu {}x multisample guest image {:#x}",
            sample_count, descriptor.address
        ));
        self.multisample_textures.insert(
            key,
            MultisampleResidentTexture {
                texture: texture.clone(),
                initialized: false,
            },
        );
        Ok((texture, true))
    }

    pub(super) fn mark_multisample_texture_initialized(
        &mut self,
        descriptor: GuestTextureDescriptor,
        sample_count: u32,
    ) {
        if let Some(resident) = self.multisample_textures.get_mut(&MultisampleTextureKey {
            descriptor,
            sample_count,
        }) {
            resident.initialized = true;
        }
    }

    pub(super) fn synchronize_cpu_written_guest_resources(
        &mut self,
        acquire: GpuMemoryAcquire,
    ) -> Result<(), String> {
        if !acquire.invalidates_guest_resources() {
            return Ok(());
        }
        let device = self
            .device
            .as_ref()
            .ok_or("Metal device is unavailable")?
            .clone();
        let mut invalidated_multisample_descriptors = Vec::new();
        let mut dirty_reprotected_residents = HashSet::new();
        let resident_descriptors = self
            .resident_guest_textures
            .keys()
            .copied()
            .collect::<Vec<_>>();
        for descriptor in resident_descriptors {
            let refresh = refresh_resident_texture_write_watch(
                self.resident_guest_textures
                    .get_mut(&descriptor)
                    .expect("collected resident texture exists"),
            )?;
            match refresh {
                WriteWatchRefresh::Renewed { was_dirty: true } => {
                    dirty_reprotected_residents.insert(descriptor);
                }
                WriteWatchRefresh::Replaced => {
                    self.resident_guest_textures.remove(&descriptor);
                }
                WriteWatchRefresh::Active
                | WriteWatchRefresh::Reprotecting
                | WriteWatchRefresh::Renewed { was_dirty: false } => {}
            }
        }
        let mut dirty_reprotected_sampled = HashSet::new();
        let sampled_keys = self
            .sampled_texture_cache
            .keys()
            .copied()
            .collect::<Vec<_>>();
        for key in sampled_keys {
            let refresh = refresh_sampled_texture_write_watch(
                key,
                self.sampled_texture_cache
                    .get_mut(&key)
                    .expect("collected sampled texture exists"),
                &mut self.sampled_texture_write_watches,
            )?;
            match refresh {
                WriteWatchRefresh::Renewed { was_dirty: true } => {
                    dirty_reprotected_sampled.insert(key);
                }
                WriteWatchRefresh::Replaced => {
                    if let Some(retired) = self.sampled_texture_cache.remove(&key) {
                        retire_sampled_texture(retired);
                    }
                }
                WriteWatchRefresh::Active
                | WriteWatchRefresh::Reprotecting
                | WriteWatchRefresh::Renewed { was_dirty: false } => {}
            }
        }
        let mut invalidated_resident_descriptors = Vec::new();
        for resident in self.resident_guest_textures.values_mut() {
            let (_, byte_count) = resident.write_watch.range();
            if !acquire.overlaps(resident.descriptor.address, byte_count)
                || (!resident.write_watch.is_dirty()
                    && !dirty_reprotected_residents.contains(&resident.descriptor))
            {
                continue;
            }
            // When the CPU is already authoritative for the complete resident, an acquire only
            // invalidates its old Metal copy. Rebuilding it here eagerly decodes large Unreal
            // render allocations even when several more acquires arrive before the image is used.
            // Drop the stale resident and let the ordinary render/sample path materialize one
            // stable snapshot on demand. GPU-authored residents cannot take this path because
            // clean bytes may still exist only in Metal and must be merged by the writeback path.
            if !resident.gpu_written {
                invalidated_resident_descriptors.push(resident.descriptor);
                invalidated_multisample_descriptors.push(resident.descriptor);
                continue;
            }
            let started = Instant::now();
            let texture = if resident.mip_level_count == 1 {
                let raw =
                    snapshot_watched_guest_texture_raw(resident.descriptor, &resident.write_watch)?;
                let linear = decode_guest_texture_linear_from_raw(resident.descriptor, &raw)?;
                create_guest_texture(
                    &device,
                    resident.descriptor,
                    MTLTextureUsage::RenderTarget
                        | MTLTextureUsage::ShaderRead
                        | MTLTextureUsage::ShaderWrite,
                    &linear,
                )?
            } else {
                let layout =
                    mode27_mip_chain_layout(resident.descriptor, resident.mip_level_count)?;
                let raw = snapshot_watched_guest_range(
                    resident.descriptor.address,
                    layout.byte_count,
                    &resident.write_watch,
                )?;
                let (_, linear_levels) = decode_mode27_mip_chain_from_raw(
                    resident.descriptor,
                    resident.mip_level_count,
                    &raw,
                )?;
                create_mipmapped_guest_texture(
                    &device,
                    resident.descriptor,
                    resident.mip_level_count,
                    &linear_levels,
                )?
            };
            if resident.mip_level_count > 1 {
                let layout =
                    mode27_mip_chain_layout(resident.descriptor, resident.mip_level_count)?;
                record_mode27_work_detail(
                    Mode27WorkOrigin::AcquireReload,
                    resident.descriptor,
                    resident.mip_level_count,
                    layout.byte_count,
                    started.elapsed(),
                );
            }
            texture.set_label(&format!(
                "REMu guest image {:#x}",
                resident.descriptor.address
            ));
            // Replace instead of mutating the old texture. Already encoded Metal work retains the
            // old allocation, while following draws observe the CPU update without a GPU wait.
            resident.texture = texture;
            resident.component_reorder_view = None;
            resident.gpu_written = false;
            invalidated_multisample_descriptors.push(resident.descriptor);
        }
        for descriptor in invalidated_resident_descriptors {
            if self.resident_guest_textures.remove(&descriptor).is_some() {
                RESIDENT_TEXTURE_ACQUIRE_INVALIDATIONS.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.multisample_textures.retain(|key, _| {
            !invalidated_multisample_descriptors
                .iter()
                .any(|descriptor| guest_texture_descriptors_overlap(key.descriptor, *descriptor))
        });

        let invalidated_sampled_keys = self
            .sampled_texture_cache
            .iter()
            .filter_map(|(key, cached)| {
                let (_, byte_count) = cached.write_watch.range();
                if !acquire.overlaps(key.descriptor.address, byte_count)
                    || (!cached.write_watch.is_dirty() && !dirty_reprotected_sampled.contains(key))
                {
                    return None;
                }
                Some(*key)
            })
            .collect::<Vec<_>>();
        // ACQUIRE_MEM makes CPU-authored bytes authoritative, but it does not require eagerly
        // materializing a new Metal texture. Evict the stale cache entry and let the ordinary
        // upload path rebuild it from a stable watched snapshot only if a later shader actually
        // samples it. This also coalesces repeated acquires before first use instead of decoding
        // and uploading the same full texture after every packet.
        for key in invalidated_sampled_keys {
            if let Some(retired) = self.sampled_texture_cache.remove(&key) {
                SAMPLED_TEXTURE_CACHE_ACQUIRE_INVALIDATIONS.fetch_add(1, Ordering::Relaxed);
                retire_sampled_texture(retired);
            }
        }
        Ok(())
    }

    pub(super) fn mark_resident_texture_written(&mut self, descriptor: GuestTextureDescriptor) {
        self.mark_resident_texture_written_preserving_multisample(descriptor);
        self.multisample_textures
            .retain(|key, _| !guest_texture_descriptors_overlap(key.descriptor, descriptor));
    }

    pub(super) fn mark_resident_texture_written_preserving_multisample(
        &mut self,
        descriptor: GuestTextureDescriptor,
    ) {
        self.resident_content_version = self.resident_content_version.saturating_add(1);
        let resident_descriptor = self
            .resident_guest_textures
            .contains_key(&descriptor)
            .then_some(descriptor)
            .or_else(|| {
                self.resident_guest_textures
                    .values()
                    .find(|resident| {
                        guest_texture_descriptors_share_storage(resident.descriptor, descriptor)
                    })
                    .map(|resident| resident.descriptor)
            });
        if let Some(resident) = resident_descriptor
            .and_then(|descriptor| self.resident_guest_textures.get_mut(&descriptor))
        {
            resident.gpu_written = true;
            resident.content_version = self.resident_content_version;
        }
        retire_overlapping_sampled_textures(&mut self.sampled_texture_cache, descriptor);
    }

    pub(super) fn synchronize_sampled_aliases(
        &mut self,
        evaluations: &[&ShaderEvaluation],
    ) -> Result<(), String> {
        let mut aliases = Vec::new();
        for &evaluation in evaluations {
            for binding in evaluation
                .image_bindings
                .iter()
                .filter(|binding| !is_storage_image_opcode(binding.opcode))
                .filter(|binding| !image_resource_descriptor_is_null(&binding.resource_descriptor))
            {
                let Some(sampled) = sampled_alias_descriptor(binding) else {
                    // The upload path binds a dimension-compatible zero texture for malformed,
                    // sentinel, or otherwise unreadable sampled descriptors. Alias
                    // synchronization has no guest range to publish in that case and must not
                    // reject the entire draw before the fallback can be installed.
                    continue;
                };
                let comparison = image_is_comparison(binding.opcode);
                let sampled_bytes = guest_texture_byte_count(sampled.guest).ok_or_else(|| {
                    format!(
                        "image at PC {:#x} has an invalid guest byte extent",
                        binding.pc
                    )
                })?;
                // A Gen5 MSAA allocation can expose its shader-readable resolved image through
                // an overlapping SRD whose base follows the allocation's mask metadata. Keep the
                // Metal resolve GPU-current so `upload_image_textures` can bind it directly. A
                // CPU writeback here both discards that provenance and decodes from the shifted
                // SRD base, producing an empty post-process input.
                //
                // Each resident write watch already records every coherent CPU alias of the
                // underlying physical range. Consult those immutable ranges directly instead of
                // resolving both descriptors through the global mapping table for every
                // resident/sample pair. Silent Hill has roughly one hundred residents and
                // performs this query for nearly every draw; the old nested mapping-lock lookup
                // consumed about one quarter of presenter CPU time in a sampled title workload.
                aliases.extend(
                    self.resident_guest_textures
                        .values()
                        .filter(|resident| resident.gpu_written)
                        .filter(|resident| {
                            let overlaps_resident = resident
                                .write_watch
                                .overlaps_alias_range(sampled.guest.address, sampled_bytes);
                            let aliases_sample = overlaps_resident
                                && resident.descriptor != sampled.guest
                                && !guest_texture_descriptors_share_storage(
                                    resident.descriptor,
                                    sampled.guest,
                                )
                                && !resident_component_reordered_sample_is_eligible(
                                    resident, sampled, comparison,
                                )
                                && !resident_depth_sample_is_eligible(resident, sampled)
                                && !self.multisample_textures.iter().any(|(key, multisample)| {
                                    multisample.initialized
                                        && key.descriptor == resident.descriptor
                                        && render_resolve_candidate_matches(
                                            sampled.guest,
                                            key.descriptor,
                                            key.sample_count,
                                        )
                                });
                            if aliases_sample && trace_target_sampling() {
                                static LOGGED_ALIAS_FLUSHES: OnceLock<
                                    Mutex<HashSet<(u64, u64, u32, u32)>>,
                                > = OnceLock::new();
                                let key = (
                                    resident.descriptor.address,
                                    sampled.guest.address,
                                    sampled.base_mip_level,
                                    sampled.mip_level_count,
                                );
                                let mut logged = LOGGED_ALIAS_FLUSHES
                                    .get_or_init(|| Mutex::new(HashSet::new()))
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                                if logged.len() < 256 && logged.insert(key) {
                                    eprintln!(
                                        "[REMu][METAL][ALIAS-FLUSH] resident={:#x}/{}x{}/p{}/{:?}/tile{}/mips{} sampled={:#x}/{}x{}/p{}/{:?}/tile{}/base{}/mips{} pc={:#x} declared_overlap={}",
                                        resident.descriptor.address,
                                        resident.descriptor.width,
                                        resident.descriptor.height,
                                        resident.descriptor.pitch,
                                        resident.descriptor.pixel_format,
                                        resident.descriptor.tile_mode,
                                        resident.mip_level_count,
                                        sampled.guest.address,
                                        sampled.guest.width,
                                        sampled.guest.height,
                                        sampled.guest.pitch,
                                        sampled.guest.pixel_format,
                                        sampled.guest.tile_mode,
                                        sampled.base_mip_level,
                                        sampled.mip_level_count,
                                        binding.pc,
                                        guest_texture_descriptors_overlap(
                                            resident.descriptor,
                                            sampled.guest,
                                        ),
                                    );
                                }
                            }
                            aliases_sample
                        })
                        .map(|resident| resident.descriptor),
                );
            }
        }
        aliases.sort_unstable_by_key(|descriptor| descriptor.address);
        aliases.dedup();
        aliases.sort_by_key(|descriptor| {
            (
                self.resident_guest_textures
                    .get(descriptor)
                    .map_or(0, |resident| resident.content_version),
                descriptor.address,
            )
        });
        if aliases.is_empty() {
            return Ok(());
        }
        let alias_started = Instant::now();
        let alias_bytes = aliases.iter().fold(0_u64, |total, &descriptor| {
            total.saturating_add(guest_texture_byte_count(descriptor).unwrap_or_default())
        });
        let alias_count = aliases.len();
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
        for descriptor in aliases {
            let refresh = refresh_resident_texture_write_watch(
                self.resident_guest_textures
                    .get_mut(&descriptor)
                    .expect("collected sampled alias exists"),
            )?;
            if refresh == WriteWatchRefresh::Replaced {
                self.resident_guest_textures.remove(&descriptor);
                continue;
            }
            if refresh == WriteWatchRefresh::Reprotecting {
                return Err(format!(
                    "sampled alias mapping {:#x} is being reprotected",
                    descriptor.address
                ));
            }
            let (alias_ranges, content_version) = {
                let resident = self
                    .resident_guest_textures
                    .get(&descriptor)
                    .expect("collected sampled alias exists");
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
                .expect("collected sampled alias exists");
            write_resident_texture_contents_to_guest(&device, &queue, resident)?;
            resident.gpu_written = false;
        }
        SAMPLED_ALIAS_WRITEBACK_COUNT.fetch_add(
            u64::try_from(alias_count).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        SAMPLED_ALIAS_WRITEBACK_BYTES.fetch_add(alias_bytes, Ordering::Relaxed);
        SAMPLED_ALIAS_WRITEBACK_NANOSECONDS.fetch_add(
            u64::try_from(alias_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        Ok(())
    }
}
pub(super) fn sampled_texture_key(
    descriptor: DecodedImageDescriptor,
    comparison: bool,
) -> SampledTextureKey {
    // GFX10 destination selection is applied by generated MSL at sample/load time. It neither
    // changes the decoded backing nor the Metal texture allocation, so including it here creates
    // duplicate 4–64 MiB uploads for views that differ only by channel selection.
    SampledTextureKey {
        descriptor: descriptor.guest,
        base_mip_level: descriptor.base_mip_level,
        mip_level_count: descriptor.mip_level_count,
        texture_type: descriptor.texture_type,
        layer_count: descriptor.layer_count,
        comparison,
    }
}

impl GuestTextureWriteWatch {
    pub(super) fn new(address: u64, byte_count: u64) -> Result<Self, String> {
        let segments = guest_shared_backing_alias_segments(address, byte_count).ok_or_else(|| {
            format!(
                "guest texture range {address:#x}+{byte_count:#x} has no complete live shared backing"
            )
        })?;
        let ranges = segments
            .into_iter()
            .map(|segment| {
                // SAFETY: the backing resolver returned live readable/writable CPU aliases that
                // cover this exact GPU-physical resource span.
                unsafe { CpuWriteWatch::new(segment.address, segment.byte_count) }.map_err(
                    |error| {
                        format!(
                            "could not watch guest texture CPU writes at {:#x}+{:#x}: {error}",
                            segment.address, segment.byte_count
                        )
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            address,
            byte_count,
            ranges,
        })
    }

    pub(super) fn retirement(&self) -> Option<CpuWriteWatchRetirement> {
        let mut retirement = None;
        for watch in &self.ranges {
            match watch.retirement() {
                Some(CpuWriteWatchRetirement::Replaced) => {
                    return Some(CpuWriteWatchRetirement::Replaced);
                }
                Some(CpuWriteWatchRetirement::Reprotecting) => {
                    retirement = Some(CpuWriteWatchRetirement::Reprotecting);
                }
                Some(CpuWriteWatchRetirement::Reprotected)
                    if retirement != Some(CpuWriteWatchRetirement::Reprotecting) =>
                {
                    retirement = Some(CpuWriteWatchRetirement::Reprotected);
                }
                None | Some(CpuWriteWatchRetirement::Reprotected) => {}
            }
        }
        retirement
    }

    pub(super) fn consume_dirty(&self) -> bool {
        self.ranges
            .iter()
            .fold(false, |dirty, watch| watch.consume_dirty() || dirty)
    }

    pub(super) fn is_dirty(&self) -> bool {
        self.ranges.iter().any(CpuWriteWatch::is_dirty)
    }

    pub(super) fn precise_tracking(&self) -> bool {
        self.ranges.iter().all(CpuWriteWatch::precise_tracking)
    }

    pub(super) fn generations(&self) -> Vec<u64> {
        self.ranges.iter().map(CpuWriteWatch::generation).collect()
    }

    pub(super) fn dirty_page_ranges(&self) -> Vec<(u64, u64)> {
        self.ranges
            .iter()
            .flat_map(CpuWriteWatch::dirty_page_ranges)
            .collect()
    }

    pub(super) fn rearm(&self) -> Result<(), String> {
        for watch in &self.ranges {
            watch
                .rearm()
                .map_err(|error| format!("could not rearm guest texture CPU writes: {error}"))?;
        }
        Ok(())
    }

    pub(super) fn range(&self) -> (u64, u64) {
        (self.address, self.byte_count)
    }

    pub(super) fn with_contiguous_mapping<T>(
        &self,
        address: u64,
        byte_count: u64,
        operation: impl FnOnce() -> T,
    ) -> Result<T, String> {
        if let Some(retirement) = self.retirement() {
            return Err(format!(
                "guest texture mapping {address:#x}+{byte_count:#x} is retired ({retirement:?})"
            ));
        }
        let end = address
            .checked_add(byte_count)
            .ok_or("guest texture mapping range overflows")?;
        let watch = self
            .ranges
            .iter()
            .find(|watch| {
                let (watch_address, watch_bytes) = watch.range();
                watch_address <= address
                    && watch_address
                        .checked_add(watch_bytes)
                        .is_some_and(|watch_end| end <= watch_end)
            })
            .ok_or_else(|| {
                format!(
                    "guest texture mapping {address:#x}+{byte_count:#x} has no single covering write watch"
                )
            })?;
        watch.with_live_mapping(operation).map_err(|error| {
            format!(
                "guest texture mapping {address:#x}+{byte_count:#x} was retired during access: {error}"
            )
        })
    }

    pub(super) fn with_live_backing<T>(&self, operation: impl FnOnce() -> T) -> Result<T, String> {
        if let Some(retirement) = self.retirement() {
            return Err(format!(
                "guest texture backing {:#x}+{:#x} is retired ({retirement:?})",
                self.address, self.byte_count
            ));
        }
        let watch = self
            .ranges
            .first()
            .ok_or("guest texture backing has no live write-watch range")?;
        // Mapping mutation retires watches under the same process-wide lock held by this guard.
        // Pinning any one live alias therefore pins the shared backing transaction even when the
        // physical resource is exposed through several disjoint guest virtual mappings.
        watch
            .with_live_mapping(|| {
                if let Some(retirement) = self.retirement() {
                    Err(format!(
                        "guest texture backing {:#x}+{:#x} is retired ({retirement:?})",
                        self.address, self.byte_count
                    ))
                } else {
                    Ok(operation())
                }
            })
            .map_err(|error| {
                format!(
                    "guest texture backing {:#x}+{:#x} was retired during access: {error}",
                    self.address, self.byte_count
                )
            })
            .and_then(|result| result)
    }

    pub(super) fn has_single_covering_mapping(&self, address: u64, byte_count: u64) -> bool {
        let Some(end) = address.checked_add(byte_count) else {
            return false;
        };
        self.ranges.iter().any(|watch| {
            let (watch_address, watch_bytes) = watch.range();
            watch_address <= address
                && watch_address
                    .checked_add(watch_bytes)
                    .is_some_and(|watch_end| end <= watch_end)
        })
    }

    pub(super) fn overlaps_alias_range(&self, address: u64, byte_count: u64) -> bool {
        Self::ranges_overlap_alias(&self.ranges, address, byte_count, |watch| watch.range())
    }

    pub(super) fn alias_ranges(&self) -> Vec<(u64, u64)> {
        self.ranges.iter().map(CpuWriteWatch::range).collect()
    }

    pub(super) fn overlaps_alias_ranges(&self, ranges: &[(u64, u64)]) -> bool {
        ranges
            .iter()
            .any(|&(address, byte_count)| self.overlaps_alias_range(address, byte_count))
    }

    pub(super) fn ranges_overlap_alias<T>(
        ranges: &[T],
        address: u64,
        byte_count: u64,
        range: impl Fn(&T) -> (u64, u64),
    ) -> bool {
        let Some(end) = address.checked_add(byte_count) else {
            return false;
        };
        byte_count != 0
            && ranges.iter().any(|entry| {
                let (watch_address, watch_bytes) = range(entry);
                watch_address < end
                    && watch_address
                        .checked_add(watch_bytes)
                        .is_some_and(|watch_end| address < watch_end)
            })
    }
}

pub(super) fn new_red_blue_swizzled_texture_view(
    texture: &Texture,
    base_mip_level: u32,
    mip_level_count: u32,
) -> Option<Texture> {
    let underlying_mip_count = base_mip_level.checked_add(mip_level_count)?;
    if mip_level_count == 0 || underlying_mip_count > texture.mipmap_level_count() as u32 {
        return None;
    }
    const RED: u8 = 2;
    const GREEN: u8 = 3;
    const BLUE: u8 = 4;
    const ALPHA: u8 = 5;
    let swizzle = MetalTextureSwizzleChannels {
        red: BLUE,
        green: GREEN,
        blue: RED,
        alpha: ALPHA,
    };
    // SAFETY: `texture` is a live MTLTexture. The selector is available on every supported macOS
    // version (10.15+), the enum arguments use Metal's NSUInteger ABI, NSRange is repr(C), and the
    // swizzle structure above exactly matches MTLTextureSwizzleChannels. A `new...` selector
    // returns an owned (+1) object, which `Texture::from_ptr` assumes.
    let view: *mut Object = unsafe {
        msg_send![
            texture.as_ref(),
            newTextureViewWithPixelFormat: texture.pixel_format()
            textureType: texture.texture_type()
            levels: NSRange::new(u64::from(base_mip_level), u64::from(mip_level_count))
            slices: NSRange::new(0, texture.array_length())
            swizzle: swizzle
        ]
    };
    if view.is_null() {
        None
    } else {
        // SAFETY: The non-null Objective-C result conforms to MTLTexture by method contract and
        // carries the retained ownership documented above.
        Some(unsafe { Texture::from_ptr(view.cast()) })
    }
}

pub(super) fn resident_texture_for_sample(
    resident: &mut ResidentGuestTexture,
    sampled: DecodedImageDescriptor,
) -> Option<Texture> {
    let underlying_mip_count = sampled
        .base_mip_level
        .checked_add(sampled.mip_level_count.max(1))?;
    if resident.mip_level_count < underlying_mip_count {
        return None;
    }
    if guest_texture_descriptors_share_depth_sample_storage(
        resident.descriptor,
        sampled.guest,
        sampled.destination_select,
    ) {
        RESIDENT_DEPTH_SAMPLE_HITS.fetch_add(1, Ordering::Relaxed);
        if sampled.base_mip_level == 0 && sampled.mip_level_count.max(1) == resident.mip_level_count
        {
            return Some(resident.texture.clone());
        }
        return Some(resident.texture.new_texture_view_from_slice(
            resident.texture.pixel_format(),
            MTLTextureType::D2,
            NSRange::new(
                u64::from(sampled.base_mip_level),
                u64::from(sampled.mip_level_count.max(1)),
            ),
            NSRange::new(0, 1),
        ));
    }
    if guest_texture_descriptors_share_storage(resident.descriptor, sampled.guest) {
        if sampled.base_mip_level == 0 && sampled.mip_level_count.max(1) == resident.mip_level_count
        {
            return Some(resident.texture.clone());
        }
        return Some(resident.texture.new_texture_view_from_slice(
            resident.texture.pixel_format(),
            MTLTextureType::D2,
            NSRange::new(
                u64::from(sampled.base_mip_level),
                u64::from(sampled.mip_level_count.max(1)),
            ),
            NSRange::new(0, 1),
        ));
    }
    if !guest_texture_descriptors_share_component_reordered_storage(
        resident.descriptor,
        sampled.guest,
    ) {
        return None;
    }
    let sampled_mip_level_count = sampled.mip_level_count.max(1);
    let uses_complete_hierarchy =
        sampled.base_mip_level == 0 && sampled_mip_level_count == resident.mip_level_count;
    if uses_complete_hierarchy && resident.component_reorder_view.is_none() {
        let view = new_red_blue_swizzled_texture_view(
            &resident.texture,
            sampled.base_mip_level,
            sampled_mip_level_count,
        )?;
        view.set_label(&format!(
            "REMu component-reordered guest image {:#x}",
            resident.descriptor.address
        ));
        resident.component_reorder_view = Some(view);
        if trace_target_sampling() {
            static CREATED_COMPONENT_REORDER_VIEWS: AtomicUsize = AtomicUsize::new(0);
            let sequence = CREATED_COMPONENT_REORDER_VIEWS.fetch_add(1, Ordering::Relaxed) + 1;
            if sequence <= 16 || sequence.is_power_of_two() {
                eprintln!(
                    "[REMu][METAL][ALIAS-VIEW] sequence={sequence} address={:#x} rendered={:?} sampled={:?} extent={}x{} tile={}",
                    resident.descriptor.address,
                    resident.descriptor.pixel_format,
                    sampled.guest.pixel_format,
                    sampled.guest.width,
                    sampled.guest.height,
                    sampled.guest.tile_mode,
                );
            }
        }
    }
    if uses_complete_hierarchy {
        resident.component_reorder_view.clone()
    } else {
        let view = new_red_blue_swizzled_texture_view(
            &resident.texture,
            sampled.base_mip_level,
            sampled_mip_level_count,
        )?;
        view.set_label(&format!(
            "REMu component-reordered guest image {:#x} mips {}..{}",
            resident.descriptor.address,
            sampled.base_mip_level,
            sampled.base_mip_level + sampled_mip_level_count,
        ));
        Some(view)
    }
}

pub(super) fn resident_component_reordered_sample_is_eligible(
    resident: &ResidentGuestTexture,
    sampled: DecodedImageDescriptor,
    comparison: bool,
) -> bool {
    let underlying_mip_count = sampled
        .base_mip_level
        .checked_add(sampled.mip_level_count.max(1));
    !comparison
        && resident.gpu_written
        && resident.texture_type == 9
        && resident.layer_count == 1
        && sampled.texture_type == 9
        && sampled.layer_count == 1
        && underlying_mip_count.is_some_and(|count| resident.mip_level_count >= count)
        && guest_texture_descriptors_share_component_reordered_storage(
            resident.descriptor,
            sampled.guest,
        )
}

pub(super) fn resident_depth_sample_is_eligible(
    resident: &ResidentGuestTexture,
    sampled: DecodedImageDescriptor,
) -> bool {
    let underlying_mip_count = sampled
        .base_mip_level
        .checked_add(sampled.mip_level_count.max(1));
    resident.gpu_written
        && resident.texture_type == 9
        && resident.layer_count == 1
        && sampled.texture_type == 9
        && sampled.layer_count == 1
        && underlying_mip_count.is_some_and(|count| resident.mip_level_count >= count)
        && guest_texture_descriptors_share_depth_sample_storage(
            resident.descriptor,
            sampled.guest,
            sampled.destination_select,
        )
}

pub(super) fn sampled_texture_cache_needs_reload(
    refresh: WriteWatchRefresh,
    active_watch_is_dirty: bool,
) -> bool {
    match refresh {
        WriteWatchRefresh::Active => active_watch_is_dirty,
        WriteWatchRefresh::Renewed { was_dirty: true } | WriteWatchRefresh::Replaced => true,
        WriteWatchRefresh::Reprotecting | WriteWatchRefresh::Renewed { was_dirty: false } => false,
    }
}

pub(super) fn refresh_resident_texture_write_watch(
    resident: &mut ResidentGuestTexture,
) -> Result<WriteWatchRefresh, String> {
    match resident.write_watch.retirement() {
        None => Ok(WriteWatchRefresh::Active),
        Some(CpuWriteWatchRetirement::Reprotecting) => Ok(WriteWatchRefresh::Reprotecting),
        Some(CpuWriteWatchRetirement::Reprotected) => {
            let was_dirty = resident.write_watch.is_dirty();
            resident.write_watch = watch_resident_texture(
                resident.descriptor,
                resident.texture_type,
                resident.layer_count,
                resident.mip_level_count,
            )?;
            Ok(WriteWatchRefresh::Renewed { was_dirty })
        }
        Some(CpuWriteWatchRetirement::Replaced) => Ok(WriteWatchRefresh::Replaced),
    }
}

pub(super) fn watch_resident_texture(
    descriptor: GuestTextureDescriptor,
    texture_type: u32,
    layer_count: u32,
    mip_level_count: u32,
) -> Result<GuestTextureWriteWatch, String> {
    if texture_type == 9 && layer_count == 1 {
        if mip_level_count == 1 {
            watch_guest_texture(descriptor)
        } else {
            let layout = mode27_mip_chain_layout(descriptor, mip_level_count)?;
            GuestTextureWriteWatch::new(
                descriptor.address,
                u64::try_from(layout.byte_count)
                    .map_err(|_| "mip chain byte count does not fit u64")?,
            )
        }
    } else {
        if mip_level_count != 1 {
            return Err(format!(
                "mipmapped resident texture type={texture_type} layers={layer_count} is unsupported"
            ));
        }
        watch_sampled_texture(DecodedImageDescriptor {
            guest: descriptor,
            destination_select: 0,
            base_mip_level: 0,
            mip_level_count: 1,
            texture_type,
            layer_count,
        })
    }
}

pub(super) fn refresh_sampled_texture_write_watch(
    key: SampledTextureKey,
    cached: &mut CachedSampledTexture,
    write_watches: &mut HashMap<SampledTextureKey, Arc<GuestTextureWriteWatch>>,
) -> Result<WriteWatchRefresh, String> {
    match cached.write_watch.retirement() {
        None => Ok(WriteWatchRefresh::Active),
        Some(CpuWriteWatchRetirement::Reprotecting) => Ok(WriteWatchRefresh::Reprotecting),
        Some(CpuWriteWatchRetirement::Reprotected) => {
            let was_dirty = cached.write_watch.is_dirty();
            let write_watch = Arc::new(watch_sampled_texture(DecodedImageDescriptor {
                guest: key.descriptor,
                // Component selection is lowered into each shader and does not change the host
                // texture allocation or the guest bytes covered by its write watch.
                destination_select: 0,
                base_mip_level: key.base_mip_level,
                mip_level_count: key.mip_level_count,
                texture_type: key.texture_type,
                layer_count: key.layer_count,
            })?);
            cached.write_watch = Arc::clone(&write_watch);
            write_watches.insert(key, write_watch);
            Ok(WriteWatchRefresh::Renewed { was_dirty })
        }
        Some(CpuWriteWatchRetirement::Replaced) => Ok(WriteWatchRefresh::Replaced),
    }
}

pub(super) fn persistent_sampled_texture_write_watch(
    key: SampledTextureKey,
    descriptor: DecodedImageDescriptor,
    write_watches: &mut HashMap<SampledTextureKey, Arc<GuestTextureWriteWatch>>,
) -> Result<Arc<GuestTextureWriteWatch>, String> {
    if let Some(write_watch) = write_watches.get(&key)
        && !matches!(
            write_watch.retirement(),
            Some(CpuWriteWatchRetirement::Reprotected | CpuWriteWatchRetirement::Replaced)
        )
    {
        return Ok(Arc::clone(write_watch));
    }
    write_watches.remove(&key);
    let write_watch = Arc::new(watch_sampled_texture(descriptor)?);
    // The texture cache has the same capacity. Keeping one coherent watch per admitted key lets
    // render-target invalidation discard stale Metal pixels without repeatedly rebuilding the
    // process-wide signal-handler registry. Rare overflow keys retain their watch only as long as
    // their sampled texture entry.
    if write_watches.len() < SAMPLED_TEXTURE_CACHE_CAPACITY {
        write_watches.insert(key, Arc::clone(&write_watch));
    }
    Ok(write_watch)
}

pub(super) fn retire_sampled_texture(texture: CachedSampledTexture) {
    static RETIRER: OnceLock<mpsc::Sender<CachedSampledTexture>> = OnceLock::new();
    let sender = RETIRER.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<CachedSampledTexture>();
        if let Err(error) = std::thread::Builder::new()
            .name("REMu texture reclaimer".into())
            .spawn(move || {
                while let Ok(texture) = receiver.recv() {
                    drop(texture);
                }
            })
        {
            eprintln!("[REMu][METAL] could not start texture reclaimer: {error}");
        }
        sender
    });
    if let Err(error) = sender.send(texture) {
        drop(error.0);
    }
}

pub(super) fn retire_sampled_textures_matching(
    cache: &mut HashMap<SampledTextureKey, CachedSampledTexture>,
    mut should_retire: impl FnMut(SampledTextureKey) -> bool,
) {
    // Dropping a write watch republishes the signal-handler's immutable page-owner snapshot.
    // Invalidating a render graph can evict hundreds of sampled aliases at once, so doing those
    // drops inside HashMap::retain stalls the presenter for an entire guest frame. Remove the
    // entries immediately for correctness, then let the existing reclaimer perform their costly
    // destruction away from the event-loop thread.
    let retired = cache
        .keys()
        .copied()
        .filter(|key| should_retire(*key))
        .collect::<Vec<_>>();
    for key in retired {
        if let Some(texture) = cache.remove(&key) {
            retire_sampled_texture(texture);
        }
    }
}

pub(super) fn retire_overlapping_sampled_textures(
    cache: &mut HashMap<SampledTextureKey, CachedSampledTexture>,
    descriptor: GuestTextureDescriptor,
) {
    retire_sampled_textures_matching(cache, |key| {
        guest_texture_descriptors_overlap(key.descriptor, descriptor)
    });
}

pub(super) fn retire_sampled_textures_overlapping_writeback(
    cache: &mut HashMap<SampledTextureKey, CachedSampledTexture>,
    alias_ranges: &[(u64, u64)],
) {
    let retired = cache
        .iter()
        .filter_map(|(key, cached)| {
            cached
                .write_watch
                .overlaps_alias_ranges(alias_ranges)
                .then_some(*key)
        })
        .collect::<Vec<_>>();
    for key in retired {
        if let Some(texture) = cache.remove(&key) {
            retire_sampled_texture(texture);
        }
    }
}

pub(super) fn retire_resident_textures_overlapping_writeback(
    cache: &mut HashMap<GuestTextureDescriptor, ResidentGuestTexture>,
    alias_ranges: &[(u64, u64)],
    source_descriptor: GuestTextureDescriptor,
    source_content_version: u64,
) {
    let retired = cache
        .iter()
        .filter_map(|(descriptor, resident)| {
            (*descriptor != source_descriptor
                && resident.content_version <= source_content_version
                && resident.write_watch.overlaps_alias_ranges(alias_ranges))
            .then_some(*descriptor)
        })
        .collect::<Vec<_>>();
    for descriptor in retired {
        cache.remove(&descriptor);
    }
}

pub(super) fn retire_multisample_textures_overlapping_writeback(
    cache: &mut HashMap<MultisampleTextureKey, MultisampleResidentTexture>,
    alias_ranges: &[(u64, u64)],
) {
    cache.retain(|key, _| {
        guest_texture_byte_count(key.descriptor).is_none_or(|byte_count| {
            !alias_ranges.iter().any(|&(address, alias_bytes)| {
                guest_ranges_overlap(address, alias_bytes, key.descriptor.address, byte_count)
            })
        })
    });
}

pub(super) fn ensure_resident_texture_in_map(
    device: &Device,
    descriptor: GuestTextureDescriptor,
    resident_guest_textures: &mut HashMap<GuestTextureDescriptor, ResidentGuestTexture>,
    upload_guest_contents: bool,
) -> Result<Texture, String> {
    if let Some(resident) = resident_guest_textures
        .get_mut(&descriptor)
        .filter(|resident| {
            resident.texture_type == 9 && resident.layer_count == 1 && resident.mip_level_count >= 1
        })
    {
        match refresh_resident_texture_write_watch(resident)? {
            WriteWatchRefresh::Active
            | WriteWatchRefresh::Reprotecting
            | WriteWatchRefresh::Renewed { .. } => return Ok(resident.texture.clone()),
            WriteWatchRefresh::Replaced => {}
        }
    }
    resident_guest_textures.remove(&descriptor);
    let write_watch = watch_guest_texture(descriptor)?;
    let usage =
        MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite;
    let texture = if upload_guest_contents {
        let raw = snapshot_watched_guest_texture_raw(descriptor, &write_watch)?;
        let linear = decode_guest_texture_linear_from_raw(descriptor, &raw)?;
        create_guest_texture(device, descriptor, usage, &linear)?
    } else {
        create_uninitialized_guest_texture(device, descriptor, usage)?
    };
    texture.set_label(&format!("REMu guest image {:#x}", descriptor.address));
    resident_guest_textures.insert(
        descriptor,
        ResidentGuestTexture {
            texture: texture.clone(),
            component_reorder_view: None,
            descriptor,
            texture_type: 9,
            layer_count: 1,
            mip_level_count: 1,
            write_watch,
            gpu_written: false,
            content_version: 0,
        },
    );
    Ok(texture)
}

pub(super) fn ensure_mipmapped_resident_texture_in_map(
    device: &Device,
    descriptor: DecodedImageDescriptor,
    resident_guest_textures: &mut HashMap<GuestTextureDescriptor, ResidentGuestTexture>,
    origin: Mode27WorkOrigin,
) -> Result<Texture, String> {
    if descriptor.texture_type != 9
        || descriptor.layer_count != 1
        || descriptor.guest.tile_mode != 27
        || descriptor.mip_level_count == 0
    {
        return Err(format!(
            "mipmapped storage image requires a nonempty 2D single-layer mode-27 resource, got type={} layers={} tile={} base={} count={}",
            descriptor.texture_type,
            descriptor.layer_count,
            descriptor.guest.tile_mode,
            descriptor.base_mip_level,
            descriptor.mip_level_count
        ));
    }
    let underlying_mip_count = descriptor
        .base_mip_level
        .checked_add(descriptor.mip_level_count)
        .ok_or("storage-image mip range overflows")?;
    if let Some(resident) = resident_guest_textures
        .get_mut(&descriptor.guest)
        .filter(|resident| {
            resident.texture_type == 9
                && resident.layer_count == 1
                && resident.mip_level_count >= underlying_mip_count
        })
    {
        match refresh_resident_texture_write_watch(resident)? {
            WriteWatchRefresh::Active
            | WriteWatchRefresh::Reprotecting
            | WriteWatchRefresh::Renewed { .. } => {
                return resident_texture_for_sample(resident, descriptor).ok_or_else(|| {
                    "resident mip hierarchy cannot represent storage view".to_owned()
                });
            }
            WriteWatchRefresh::Replaced => {}
        }
    }
    if resident_guest_textures
        .get(&descriptor.guest)
        .is_some_and(|resident| resident.gpu_written)
    {
        return Err(format!(
            "cannot reinterpret GPU-written image {:#x} as a {underlying_mip_count}-level hierarchy",
            descriptor.guest.address
        ));
    }
    resident_guest_textures.remove(&descriptor.guest);
    let started = Instant::now();
    let layout = mode27_mip_chain_layout(descriptor.guest, underlying_mip_count)?;
    let write_watch = watch_resident_texture(
        descriptor.guest,
        descriptor.texture_type,
        descriptor.layer_count,
        underlying_mip_count,
    )?;
    let raw =
        snapshot_watched_guest_range(descriptor.guest.address, layout.byte_count, &write_watch)?;
    let (_, linear_levels) =
        decode_mode27_mip_chain_from_raw(descriptor.guest, underlying_mip_count, &raw)?;
    let texture = create_mipmapped_guest_texture(
        device,
        descriptor.guest,
        underlying_mip_count,
        &linear_levels,
    )?;
    record_mode27_work_detail(
        origin,
        descriptor.guest,
        underlying_mip_count,
        layout.byte_count,
        started.elapsed(),
    );
    texture.set_label(&format!(
        "REMu {}-level storage image {:#x}",
        underlying_mip_count, descriptor.guest.address
    ));
    resident_guest_textures.insert(
        descriptor.guest,
        ResidentGuestTexture {
            texture,
            component_reorder_view: None,
            descriptor: descriptor.guest,
            texture_type: descriptor.texture_type,
            layer_count: descriptor.layer_count,
            mip_level_count: underlying_mip_count,
            write_watch,
            gpu_written: false,
            content_version: 0,
        },
    );
    resident_texture_for_sample(
        resident_guest_textures
            .get_mut(&descriptor.guest)
            .expect("inserted mipmapped resident exists"),
        descriptor,
    )
    .ok_or_else(|| "created mip hierarchy cannot represent storage view".to_owned())
}
/// Metal exposes this C structure by value in the swizzled texture-view Objective-C method.
/// Its members are `uint8_t` in the macOS SDK; keep the layout explicit because metal-rs 0.33
/// does not yet bind this overload.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct MetalTextureSwizzleChannels {
    red: u8,
    green: u8,
    blue: u8,
    alpha: u8,
}

pub(super) fn guest_texture_raw_range(
    descriptor: GuestTextureDescriptor,
) -> Result<(usize, usize), String> {
    let byte_count = guest_texture_byte_count(descriptor)
        .ok_or("guest texture byte count overflows the host address space")?;
    if descriptor.address < 0x10_000 || byte_count > 128 * 1024 * 1024 {
        return Err(format!(
            "invalid guest texture range address={:#x} bytes={byte_count}",
            descriptor.address
        ));
    }
    let source_address = usize::try_from(descriptor.address)
        .map_err(|_| "guest texture address does not fit usize")?;
    let source_length =
        usize::try_from(byte_count).map_err(|_| "guest texture byte count does not fit usize")?;
    Ok((source_address, source_length))
}

pub(super) fn snapshot_guest_texture_raw(
    descriptor: GuestTextureDescriptor,
) -> Result<Vec<u8>, String> {
    let (_source_address, source_length) = guest_texture_raw_range(descriptor)?;
    let mut raw = vec![0; source_length];
    if read_guest_shared_backing(descriptor.address, &mut raw) {
        return Ok(raw);
    }
    #[cfg(test)]
    {
        // Tests also exercise the tiler with ordinary process-local allocations.
        // SAFETY: those test descriptors retain their complete source allocation.
        return Ok(unsafe {
            std::slice::from_raw_parts(_source_address as *const u8, source_length)
        }
        .to_vec());
    }
    #[cfg(not(test))]
    Err(format!(
        "guest texture range {:#x}+{source_length:#x} has no complete live shared backing",
        descriptor.address
    ))
}

pub(super) fn watch_guest_texture(
    descriptor: GuestTextureDescriptor,
) -> Result<GuestTextureWriteWatch, String> {
    let (_, byte_count) = guest_texture_raw_range(descriptor)?;
    let byte_count =
        u64::try_from(byte_count).map_err(|_| "guest texture byte count does not fit u64")?;
    GuestTextureWriteWatch::new(descriptor.address, byte_count)
}

pub(super) fn sampled_texture_raw_range(
    descriptor: DecodedImageDescriptor,
) -> Result<(usize, usize), String> {
    let exact_mip_chain = sampled_texture_uses_exact_mode27_mips(descriptor);
    let byte_count = if exact_mip_chain {
        if descriptor.texture_type != 9 || descriptor.layer_count != 1 {
            return Err(format!(
                "exact sampled mip-chain upload requires a 2D single-layer mode-27 resource, got type={} layers={} tile={} base={} count={}",
                descriptor.texture_type,
                descriptor.layer_count,
                descriptor.guest.tile_mode,
                descriptor.base_mip_level,
                descriptor.mip_level_count
            ));
        }
        let underlying_mip_count = descriptor
            .base_mip_level
            .checked_add(descriptor.mip_level_count)
            .ok_or("sampled mip range overflows")?;
        u64::try_from(mode27_mip_chain_layout(descriptor.guest, underlying_mip_count)?.byte_count)
            .map_err(|_| "sampled mip-chain byte count does not fit u64")?
    } else {
        guest_texture_byte_count(descriptor.guest)
            .ok_or("sampled texture face byte count overflows")?
            .checked_mul(u64::from(descriptor.layer_count))
            .ok_or("sampled texture byte count overflows")?
    };
    if descriptor.guest.address < 0x10_000 || byte_count > 512 * 1024 * 1024 {
        return Err(format!(
            "invalid sampled texture range address={:#x} bytes={byte_count}",
            descriptor.guest.address
        ));
    }
    Ok((
        usize::try_from(descriptor.guest.address)
            .map_err(|_| "sampled texture address does not fit usize")?,
        usize::try_from(byte_count).map_err(|_| "sampled texture byte count does not fit usize")?,
    ))
}

pub(super) fn sampled_texture_uses_exact_mode27_mips(descriptor: DecodedImageDescriptor) -> bool {
    descriptor.guest.tile_mode == 27
        && (descriptor.base_mip_level != 0 || descriptor.mip_level_count != 1)
}

pub(super) fn watch_sampled_texture(
    descriptor: DecodedImageDescriptor,
) -> Result<GuestTextureWriteWatch, String> {
    let (_, byte_count) = sampled_texture_raw_range(descriptor)?;
    GuestTextureWriteWatch::new(
        descriptor.guest.address,
        u64::try_from(byte_count).map_err(|_| "sampled byte count does not fit u64")?,
    )
}

#[track_caller]
pub(super) fn snapshot_watched_sampled_texture_raw(
    descriptor: DecodedImageDescriptor,
    write_watch: &GuestTextureWriteWatch,
) -> Result<Vec<u8>, String> {
    let (_, source_length) = sampled_texture_raw_range(descriptor)?;
    snapshot_watched_guest_range(descriptor.guest.address, source_length, write_watch)
}

pub(super) fn decode_sampled_texture_linear_from_raw(
    descriptor: DecodedImageDescriptor,
    raw: &[u8],
) -> Result<Vec<u8>, String> {
    let face_bytes = usize::try_from(
        guest_texture_byte_count(descriptor.guest)
            .ok_or("sampled texture face byte count overflows")?,
    )
    .map_err(|_| "sampled texture face byte count does not fit usize")?;
    let expected = face_bytes
        .checked_mul(
            usize::try_from(descriptor.layer_count)
                .map_err(|_| "sampled layer count does not fit usize")?,
        )
        .ok_or("sampled texture byte count overflows")?;
    if raw.len() != expected {
        return Err(format!(
            "sampled texture source has {} bytes, expected {expected}",
            raw.len()
        ));
    }
    let mut linear = Vec::new();
    for (surface_z, face) in raw.chunks_exact(face_bytes).enumerate() {
        linear.extend(decode_guest_texture_linear_from_raw_at_z(
            descriptor.guest,
            face,
            surface_z,
        )?);
    }
    Ok(linear)
}

#[track_caller]
pub(super) fn snapshot_watched_guest_range(
    address: u64,
    byte_count: usize,
    write_watch: &GuestTextureWriteWatch,
) -> Result<Vec<u8>, String> {
    let started = Instant::now();
    let caller_line = std::panic::Location::caller().line();
    let precise_tracking = write_watch.precise_tracking();
    let dirty_page_ranges = precise_tracking
        .then(|| write_watch.dirty_page_ranges())
        .unwrap_or_default();
    let dirty_bytes = dirty_page_ranges
        .iter()
        .fold(0_u64, |total, (_, range_byte_count)| {
            total.saturating_add(*range_byte_count)
        });
    record_texture_dirty_work_detail(
        address,
        byte_count,
        caller_line,
        precise_tracking,
        dirty_page_ranges.len(),
        dirty_bytes,
    );
    let byte_count_u64 =
        u64::try_from(byte_count).map_err(|_| "guest range length does not fit u64")?;
    let source_address =
        usize::try_from(address).map_err(|_| "guest range address does not fit usize")?;
    let direct_snapshot = || {
        // SAFETY: with_contiguous_mapping pins the complete live mapping around this copy.
        unsafe { std::slice::from_raw_parts(source_address as *const u8, byte_count) }.to_vec()
    };
    let snapshot = || {
        if guest_shared_mapping_identity(address, byte_count_u64).is_some()
            && write_watch.has_single_covering_mapping(address, byte_count_u64)
        {
            write_watch.with_contiguous_mapping(address, byte_count_u64, direct_snapshot)
        } else {
            let mut raw = vec![0; byte_count];
            if read_guest_shared_backing(address, &mut raw) {
                Ok(raw)
            } else {
                Err(format!(
                    "guest range {address:#x}+{byte_count:#x} has no complete live shared backing"
                ))
            }
        }
    };
    let result = if !write_watch.precise_tracking() {
        snapshot()
    } else {
        loop {
            let _ = write_watch.consume_dirty();
            write_watch.rearm()?;
            let generations = write_watch.generations();
            let raw = snapshot()?;
            if write_watch.generations() == generations && !write_watch.is_dirty() {
                break Ok(raw);
            }
        }
    };
    if result.is_ok() {
        let elapsed = started.elapsed();
        record_texture_work(
            &TEXTURE_SNAPSHOT_COUNT,
            &TEXTURE_SNAPSHOT_BYTES,
            &TEXTURE_SNAPSHOT_NANOSECONDS,
            byte_count,
            elapsed,
        );
        record_texture_snapshot_work_detail(address, byte_count, caller_line, elapsed);
    }
    result
}

#[track_caller]
pub(super) fn snapshot_watched_guest_texture_raw(
    descriptor: GuestTextureDescriptor,
    write_watch: &GuestTextureWriteWatch,
) -> Result<Vec<u8>, String> {
    let (_, source_length) = guest_texture_raw_range(descriptor)?;
    snapshot_watched_guest_range(descriptor.address, source_length, write_watch)
}

pub(super) fn write_watched_guest_range(
    address: u64,
    raw: &[u8],
    write_watch: &GuestTextureWriteWatch,
) -> Result<(), String> {
    if write_watch.retirement() == Some(CpuWriteWatchRetirement::Replaced) {
        return Err(format!(
            "guest range mapping {address:#x} was retired before writeback"
        ));
    }
    let byte_count = u64::try_from(raw.len()).map_err(|_| "guest range length does not fit u64")?;
    let wrote = write_watch.with_live_backing(|| write_guest_shared_backing(address, raw))?;
    if wrote {
        Ok(())
    } else {
        Err(format!(
            "guest range {address:#x}+{byte_count:#x} lost shared backing during writeback"
        ))
    }
}

pub(super) fn decode_guest_texture_linear_from_raw(
    descriptor: GuestTextureDescriptor,
    source: &[u8],
) -> Result<Vec<u8>, String> {
    decode_guest_texture_linear_from_raw_at_z(descriptor, source, 0)
}

pub(super) fn decode_guest_texture_linear_from_raw_at_z(
    descriptor: GuestTextureDescriptor,
    source: &[u8],
    surface_z: usize,
) -> Result<Vec<u8>, String> {
    let (width, height, pitch, bytes_per_element) = texture_element_layout(descriptor)?;
    let output_length = u64::try_from(width)
        .ok()
        .and_then(|width| width.checked_mul(u64::try_from(height).ok()?))
        .and_then(|elements| elements.checked_mul(u64::try_from(bytes_per_element).ok()?))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or("guest texture dimensions overflow")?;
    let expected_source = usize::try_from(
        guest_texture_byte_count(descriptor).ok_or("guest texture byte count overflows")?,
    )
    .map_err(|_| "guest texture byte count does not fit usize")?;
    if source.len() != expected_source {
        return Err(format!(
            "guest texture source has {} bytes, expected {expected_source}",
            source.len()
        ));
    }
    let mut output = vec![0_u8; output_length];
    if descriptor.tile_mode == 0 {
        let source_row_bytes = pitch
            .checked_mul(bytes_per_element)
            .ok_or("source row byte count overflows")?;
        let output_row_bytes = width
            .checked_mul(bytes_per_element)
            .ok_or("output row byte count overflows")?;
        for y in 0..height {
            let source_offset = y
                .checked_mul(source_row_bytes)
                .ok_or("source row offset overflows")?;
            let destination_offset = y
                .checked_mul(output_row_bytes)
                .ok_or("destination row offset overflows")?;
            output[destination_offset..destination_offset + output_row_bytes].copy_from_slice(
                source
                    .get(source_offset..source_offset + output_row_bytes)
                    .ok_or("guest texture row exceeds its decoded allocation")?,
            );
        }
    } else {
        copy_tiled_to_linear_at_z(source, descriptor, &mut output, surface_z)?;
    }
    Ok(output)
}

pub(super) fn write_guest_texture_linear_into_raw(
    descriptor: GuestTextureDescriptor,
    linear: &[u8],
    destination: &mut [u8],
) -> Result<(), String> {
    let (width, height, pitch, bytes_per_element) = texture_element_layout(descriptor)?;
    if descriptor.tile_mode == 0 {
        let destination_row_bytes = pitch
            .checked_mul(bytes_per_element)
            .ok_or("destination row byte count overflows")?;
        let source_row_bytes = width
            .checked_mul(bytes_per_element)
            .ok_or("source row byte count overflows")?;
        for y in 0..height {
            let source_offset = y
                .checked_mul(source_row_bytes)
                .ok_or("source row offset overflows")?;
            let destination_offset = y
                .checked_mul(destination_row_bytes)
                .ok_or("destination row offset overflows")?;
            destination
                .get_mut(destination_offset..destination_offset + source_row_bytes)
                .ok_or("guest texture row exceeds its decoded allocation")?
                .copy_from_slice(&linear[source_offset..source_offset + source_row_bytes]);
        }
    } else {
        copy_linear_to_tiled(linear, descriptor, destination)?;
    }
    Ok(())
}

pub(super) fn write_watched_guest_texture_linear(
    descriptor: GuestTextureDescriptor,
    linear: &[u8],
    write_watch: &GuestTextureWriteWatch,
) -> Result<(), String> {
    if write_watch.retirement() == Some(CpuWriteWatchRetirement::Replaced) {
        return Err(format!(
            "guest texture mapping {:#x} was retired before writeback",
            descriptor.address
        ));
    }
    let byte_count = guest_texture_byte_count(descriptor)
        .ok_or("guest texture byte count overflows the host address space")?;
    let destination_length =
        usize::try_from(byte_count).map_err(|_| "guest texture byte count does not fit usize")?;
    write_watch
        .with_live_backing(|| {
            let mut destination = vec![0; destination_length];
            if !read_guest_shared_backing(descriptor.address, &mut destination) {
                return Err(format!(
                    "guest texture range {:#x}+{byte_count:#x} has no complete live shared backing",
                    descriptor.address
                ));
            }
            write_guest_texture_linear_into_raw(descriptor, linear, &mut destination)?;
            if !write_guest_shared_backing(descriptor.address, &destination) {
                return Err(format!(
                    "guest texture range {:#x}+{byte_count:#x} lost shared backing during writeback",
                    descriptor.address
                ));
            }
            Ok(())
        })
        .and_then(|result| result)
}

pub(super) fn create_guest_texture(
    device: &Device,
    guest: GuestTextureDescriptor,
    usage: MTLTextureUsage,
    linear: &[u8],
) -> Result<Texture, String> {
    let (elements_wide, elements_high, _, bytes_per_element) = texture_element_layout(guest)?;
    let required = u64::try_from(elements_wide)
        .ok()
        .and_then(|width| width.checked_mul(u64::try_from(elements_high).ok()?))
        .and_then(|elements| elements.checked_mul(u64::try_from(bytes_per_element).ok()?))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or("Metal texture dimensions overflow")?;
    if guest.width == 0 || guest.height == 0 || linear.len() != required {
        return Err(format!(
            "invalid guest texture upload {}x{} format {:?} with {} bytes (expected {required})",
            guest.width,
            guest.height,
            guest.pixel_format,
            linear.len()
        ));
    }
    let texture = create_uninitialized_guest_texture(device, guest, usage)?;
    texture.replace_region(
        MTLRegion::new_2d(0, 0, u64::from(guest.width), u64::from(guest.height)),
        0,
        linear.as_ptr().cast::<c_void>(),
        u64::try_from(elements_wide).expect("validated texture element width fits u64")
            * u64::try_from(bytes_per_element)
                .expect("validated texture element byte count fits u64"),
    );
    Ok(texture)
}

pub(super) fn create_uninitialized_guest_texture(
    device: &Device,
    guest: GuestTextureDescriptor,
    usage: MTLTextureUsage,
) -> Result<Texture, String> {
    if guest.width == 0 || guest.height == 0 {
        return Err(format!(
            "invalid zero-sized guest texture {}x{} format {:?}",
            guest.width, guest.height, guest.pixel_format,
        ));
    }
    // Validate the guest/Metal format relationship even though a following render-pass clear,
    // rather than a CPU upload, supplies the initial contents.
    let _ = texture_element_layout(guest)?;
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(guest.pixel_format);
    descriptor.set_width(u64::from(guest.width));
    descriptor.set_height(u64::from(guest.height));
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(usage);
    Ok(device.new_texture(&descriptor))
}

pub(super) fn create_mipmapped_guest_texture(
    device: &Device,
    guest: GuestTextureDescriptor,
    mip_level_count: u32,
    linear_levels: &[Vec<u8>],
) -> Result<Texture, String> {
    let started = Instant::now();
    let layout = mode27_mip_chain_layout(guest, mip_level_count)?;
    if linear_levels.len() != layout.levels.len() {
        return Err(format!(
            "Metal mip upload has {} levels, expected {}",
            linear_levels.len(),
            layout.levels.len()
        ));
    }
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(guest.pixel_format);
    descriptor.set_width(u64::from(guest.width));
    descriptor.set_height(u64::from(guest.height));
    descriptor.set_mipmap_level_count(u64::from(mip_level_count));
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(
        MTLTextureUsage::RenderTarget
            | MTLTextureUsage::ShaderRead
            | MTLTextureUsage::ShaderWrite
            | MTLTextureUsage::PixelFormatView,
    );
    let texture = device.new_texture(&descriptor);
    for (level_index, (&level, linear)) in layout.levels.iter().zip(linear_levels).enumerate() {
        let expected = level
            .width
            .checked_mul(level.height)
            .and_then(|elements| elements.checked_mul(layout.bytes_per_element))
            .ok_or("Metal mip upload size overflows")?;
        if linear.len() != expected {
            return Err(format!(
                "Metal mip {level_index} upload has {} bytes, expected {expected}",
                linear.len()
            ));
        }
        texture.replace_region(
            MTLRegion::new_2d(
                0,
                0,
                u64::try_from(level.width).expect("validated mip width fits u64"),
                u64::try_from(level.height).expect("validated mip height fits u64"),
            ),
            u64::try_from(level_index).expect("validated mip index fits u64"),
            linear.as_ptr().cast::<c_void>(),
            u64::try_from(level.width * layout.bytes_per_element)
                .expect("validated mip row size fits u64"),
        );
    }
    record_texture_work(
        &MIP_UPLOAD_COUNT,
        &MIP_UPLOAD_BYTES,
        &MIP_UPLOAD_NANOSECONDS,
        linear_levels.iter().map(Vec::len).sum(),
        started.elapsed(),
    );
    Ok(texture)
}

pub(super) fn create_exact_mode27_sampled_mip_view(
    device: &Device,
    descriptor: DecodedImageDescriptor,
    raw: &[u8],
    comparison: bool,
) -> Result<Texture, String> {
    if comparison {
        return Err(format!(
            "exact comparison-sampled mip view base={} count={} is not implemented",
            descriptor.base_mip_level, descriptor.mip_level_count
        ));
    }
    if descriptor.texture_type != 9
        || descriptor.layer_count != 1
        || descriptor.guest.tile_mode != 27
        || descriptor.mip_level_count == 0
    {
        return Err(format!(
            "exact sampled mip view requires a nonempty 2D single-layer mode-27 resource, got type={} layers={} tile={} base={} count={}",
            descriptor.texture_type,
            descriptor.layer_count,
            descriptor.guest.tile_mode,
            descriptor.base_mip_level,
            descriptor.mip_level_count
        ));
    }
    let underlying_mip_count = descriptor
        .base_mip_level
        .checked_add(descriptor.mip_level_count)
        .ok_or("sampled mip range overflows")?;
    let started = Instant::now();
    let (_, linear_levels) =
        decode_mode27_mip_chain_from_raw(descriptor.guest, underlying_mip_count, raw)?;
    let texture = create_mipmapped_guest_texture(
        device,
        descriptor.guest,
        underlying_mip_count,
        &linear_levels,
    )?;
    record_mode27_work_detail(
        Mode27WorkOrigin::SampledCacheCreate,
        descriptor.guest,
        underlying_mip_count,
        raw.len(),
        started.elapsed(),
    );
    if descriptor.base_mip_level == 0 && descriptor.mip_level_count == underlying_mip_count {
        return Ok(texture);
    }
    Ok(texture.new_texture_view_from_slice(
        descriptor.guest.pixel_format,
        MTLTextureType::D2,
        NSRange::new(
            u64::from(descriptor.base_mip_level),
            u64::from(descriptor.mip_level_count),
        ),
        NSRange::new(0, 1),
    ))
}

pub(super) fn create_layered_render_target_texture(
    device: &Device,
    descriptor: DecodedImageDescriptor,
    linear: &[u8],
) -> Result<Texture, String> {
    if descriptor.texture_type != 10 || descriptor.layer_count <= 1 {
        return Err(format!(
            "layered render target requires a 3D texture, got type={} layers={}",
            descriptor.texture_type, descriptor.layer_count
        ));
    }
    let (elements_wide, elements_high, _, bytes_per_element) =
        texture_element_layout(descriptor.guest)?;
    let bytes_per_row = elements_wide
        .checked_mul(bytes_per_element)
        .ok_or("layered Metal texture row size overflows")?;
    let bytes_per_image = bytes_per_row
        .checked_mul(elements_high)
        .ok_or("layered Metal texture image size overflows")?;
    let required = bytes_per_image
        .checked_mul(
            usize::try_from(descriptor.layer_count)
                .map_err(|_| "layered Metal texture depth does not fit usize")?,
        )
        .ok_or("layered Metal texture byte count overflows")?;
    if linear.len() != required {
        return Err(format!(
            "invalid layered guest texture upload {}x{}x{} format {:?} with {} bytes (expected {required})",
            descriptor.guest.width,
            descriptor.guest.height,
            descriptor.layer_count,
            descriptor.guest.pixel_format,
            linear.len()
        ));
    }
    let metal_descriptor = TextureDescriptor::new();
    metal_descriptor.set_texture_type(MTLTextureType::D3);
    metal_descriptor.set_pixel_format(descriptor.guest.pixel_format);
    metal_descriptor.set_width(u64::from(descriptor.guest.width));
    metal_descriptor.set_height(u64::from(descriptor.guest.height));
    metal_descriptor.set_depth(u64::from(descriptor.layer_count));
    metal_descriptor.set_storage_mode(MTLStorageMode::Shared);
    metal_descriptor.set_usage(
        MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
    );
    let texture = device.new_texture(&metal_descriptor);
    texture.replace_region_in_slice(
        MTLRegion::new_3d(
            0,
            0,
            0,
            u64::from(descriptor.guest.width),
            u64::from(descriptor.guest.height),
            u64::from(descriptor.layer_count),
        ),
        0,
        0,
        linear.as_ptr().cast::<c_void>(),
        u64::try_from(bytes_per_row).expect("validated row size fits u64"),
        u64::try_from(bytes_per_image).expect("validated image size fits u64"),
    );
    Ok(texture)
}

pub(super) fn create_sampled_guest_texture(
    device: &Device,
    queue: &metal::CommandQueue,
    guest: GuestTextureDescriptor,
    linear: &[u8],
    base_mip_level: u32,
    requested_mip_level_count: u32,
    texture_type: u32,
    layer_count: u32,
    comparison: bool,
) -> Result<Texture, String> {
    let (elements_wide, elements_high, _, bytes_per_element) = texture_element_layout(guest)?;
    let source_face_bytes = elements_wide
        .checked_mul(elements_high)
        .and_then(|elements| elements.checked_mul(bytes_per_element))
        .ok_or("Metal sampled texture dimensions overflow")?;
    let required = source_face_bytes
        .checked_mul(
            usize::try_from(layer_count).map_err(|_| "sampled layer count does not fit usize")?,
        )
        .ok_or("Metal sampled texture layer dimensions overflow")?;
    if guest.width == 0 || guest.height == 0 || linear.len() != required {
        return Err(format!(
            "invalid sampled guest texture upload {}x{} format {:?} with {} bytes (expected {required})",
            guest.width,
            guest.height,
            guest.pixel_format,
            linear.len()
        ));
    }
    if trace_image_descriptors()
        && guest.tile_mode == 0
        && guest.width >= 700
        && guest.height >= 500
        && matches!(
            guest.pixel_format,
            MTLPixelFormat::R8Unorm | MTLPixelFormat::RG8Unorm
        )
    {
        static LOGGED_VIDEO_UPLOADS: OnceLock<Mutex<HashSet<(u64, u64)>>> = OnceLock::new();
        let fingerprint = guest_frame_fingerprint(linear);
        let mut logged = LOGGED_VIDEO_UPLOADS
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if logged.len() < 32 && logged.insert((guest.address, fingerprint)) {
            let (minimum, maximum, total) = linear.iter().fold(
                (u8::MAX, u8::MIN, 0_u64),
                |(minimum, maximum, total), &value| {
                    (
                        minimum.min(value),
                        maximum.max(value),
                        total + u64::from(value),
                    )
                },
            );
            let average = total as f64 / linear.len().max(1) as f64;
            eprintln!(
                "[REMu][METAL][IMAGE] uploaded linear video plane address={:#x} extent={}x{} format={:?} bytes={} hash={fingerprint:016x} range={minimum}..={maximum} average={average:.1}",
                guest.address,
                guest.width,
                guest.height,
                guest.pixel_format,
                linear.len(),
            );
        }
    }
    let comparison_linear = comparison
        .then(|| comparison_depth32_upload(guest.pixel_format, linear))
        .transpose()?
        .flatten();
    let upload = comparison_linear.as_deref().unwrap_or(linear);
    let upload_bytes_per_element = if comparison_linear.is_some() {
        std::mem::size_of::<f32>()
    } else {
        bytes_per_element
    };
    let face_bytes = elements_wide
        .checked_mul(elements_high)
        .and_then(|elements| elements.checked_mul(upload_bytes_per_element))
        .ok_or("Metal sampled texture upload dimensions overflow")?;
    // GFX10 packs small mip levels into a tiled mip tail. Until that guest layout is decoded,
    // generate the same filtered hierarchy from the exact base image on Metal. Compressed BC
    // formats cannot be filtered by Metal's blit mip generator, so they remain base-level views.
    let underlying_mip_level_count = base_mip_level
        .checked_add(requested_mip_level_count.max(1))
        .ok_or("sampled Metal mip range overflows")?;
    let mip_level_count = if compressed_texture_block_extent(guest.pixel_format).is_some() {
        if base_mip_level != 0 {
            return Err(format!(
                "compressed sampled Metal texture cannot synthesize base mip {base_mip_level}"
            ));
        }
        1
    } else {
        underlying_mip_level_count
    };
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(match texture_type {
        9 => MTLTextureType::D2,
        10 => MTLTextureType::D3,
        11 if layer_count == 6 => MTLTextureType::Cube,
        11 => MTLTextureType::CubeArray,
        13 => MTLTextureType::D2Array,
        _ => {
            return Err(format!(
                "unsupported sampled Metal texture type {texture_type}"
            ));
        }
    });
    let pixel_format = if comparison {
        match guest.pixel_format {
            MTLPixelFormat::R16Unorm => MTLPixelFormat::Depth16Unorm,
            MTLPixelFormat::R32Float => MTLPixelFormat::Depth32Float,
            _ => MTLPixelFormat::Depth32Float,
        }
    } else {
        guest.pixel_format
    };
    descriptor.set_pixel_format(pixel_format);
    descriptor.set_width(u64::from(guest.width));
    descriptor.set_height(u64::from(guest.height));
    if texture_type == 10 {
        descriptor.set_depth(u64::from(layer_count));
    }
    descriptor.set_mipmap_level_count(u64::from(mip_level_count));
    if texture_type == 13 {
        descriptor.set_array_length(u64::from(layer_count));
    }
    if texture_type == 11 {
        descriptor.set_array_length(u64::from(layer_count / 6));
    }
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(if texture_type == 10 && !comparison {
        MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite
    } else {
        MTLTextureUsage::ShaderRead
    });
    let texture = device.new_texture(&descriptor);
    let bytes_per_row = u64::try_from(elements_wide)
        .expect("validated texture element width fits u64")
        * u64::try_from(upload_bytes_per_element)
            .expect("validated texture element byte count fits u64");
    if texture_type == 10 {
        texture.replace_region_in_slice(
            MTLRegion::new_3d(
                0,
                0,
                0,
                u64::from(guest.width),
                u64::from(guest.height),
                u64::from(layer_count),
            ),
            0,
            0,
            upload.as_ptr().cast::<c_void>(),
            bytes_per_row,
            u64::try_from(face_bytes).expect("validated image byte count fits u64"),
        );
    } else if matches!(texture_type, 11 | 13) {
        for layer in 0..layer_count {
            let offset = usize::try_from(layer)
                .expect("validated layer fits usize")
                .checked_mul(face_bytes)
                .expect("validated layer offset fits usize");
            texture.replace_region_in_slice(
                MTLRegion::new_2d(0, 0, u64::from(guest.width), u64::from(guest.height)),
                0,
                u64::from(layer),
                upload[offset..].as_ptr().cast::<c_void>(),
                bytes_per_row,
                u64::try_from(face_bytes).expect("validated face byte count fits u64"),
            );
        }
    } else {
        texture.replace_region(
            MTLRegion::new_2d(0, 0, u64::from(guest.width), u64::from(guest.height)),
            0,
            upload.as_ptr().cast::<c_void>(),
            bytes_per_row,
        );
    }
    if mip_level_count > 1 {
        let command_buffer = queue.new_command_buffer().to_owned();
        command_buffer.set_label("REMu sampled image mip generation");
        let encoder = command_buffer.new_blit_command_encoder();
        encoder.set_label("REMu sampled image generated mip hierarchy");
        encoder.generate_mipmaps(&texture);
        encoder.end_encoding();
        command_buffer.commit();
    }
    if base_mip_level == 0 {
        return Ok(texture);
    }
    if texture_type != 9 {
        return Err(format!(
            "nonzero sampled base mip {base_mip_level} is not implemented for texture type {texture_type}"
        ));
    }
    Ok(texture.new_texture_view_from_slice(
        pixel_format,
        MTLTextureType::D2,
        NSRange::new(
            u64::from(base_mip_level),
            u64::from(requested_mip_level_count.max(1)),
        ),
        NSRange::new(0, 1),
    ))
}

pub(super) fn comparison_depth32_upload(
    format: MTLPixelFormat,
    linear: &[u8],
) -> Result<Option<Vec<u8>>, String> {
    let (stride, red_offset, srgb) = match format {
        MTLPixelFormat::R16Unorm | MTLPixelFormat::R32Float => return Ok(None),
        MTLPixelFormat::R8Unorm => (1, 0, false),
        MTLPixelFormat::R8Unorm_sRGB => (1, 0, true),
        MTLPixelFormat::RG8Unorm => (2, 0, false),
        MTLPixelFormat::RG8Unorm_sRGB => (2, 0, true),
        MTLPixelFormat::RGBA8Unorm => (4, 0, false),
        MTLPixelFormat::RGBA8Unorm_sRGB => (4, 0, true),
        MTLPixelFormat::BGRA8Unorm => (4, 2, false),
        MTLPixelFormat::BGRA8Unorm_sRGB => (4, 2, true),
        _ => {
            return Err(format!(
                "comparison sampling cannot convert Metal image format {format:?} to depth"
            ));
        }
    };
    if !linear.len().is_multiple_of(stride) {
        return Err(format!(
            "comparison image byte count {} is not aligned to its {stride}-byte texel",
            linear.len()
        ));
    }
    let mut depth = Vec::with_capacity(linear.len() / stride * std::mem::size_of::<f32>());
    for texel in linear.chunks_exact(stride) {
        let encoded = f32::from(texel[red_offset]) / 255.0;
        let value = if srgb {
            if encoded <= 0.040_45 {
                encoded / 12.92
            } else {
                ((encoded + 0.055) / 1.055).powf(2.4)
            }
        } else {
            encoded
        };
        depth.extend_from_slice(&value.to_ne_bytes());
    }
    Ok(Some(depth))
}

pub(super) fn upload_image_textures(
    device: &Device,
    queue: &metal::CommandQueue,
    evaluation: &ShaderEvaluation,
    resident_guest_textures: &mut HashMap<GuestTextureDescriptor, ResidentGuestTexture>,
    multisample_textures: &HashMap<MultisampleTextureKey, MultisampleResidentTexture>,
    sampled_texture_cache: &mut HashMap<SampledTextureKey, CachedSampledTexture>,
    sampled_texture_write_watches: &mut HashMap<SampledTextureKey, Arc<GuestTextureWriteWatch>>,
    fallback_texture_cache: &mut HashMap<SampledTextureKey, Texture>,
    sampled_texture_admission_order: &mut VecDeque<(SampledTextureKey, u64)>,
    next_sampled_texture_admission_id: &mut u64,
) -> Result<Vec<UploadedImageTexture>, String> {
    canonical_texture_bindings(evaluation)
        .into_iter()
        .map(|binding| {
            let writable = is_writable_image_opcode(binding.opcode);
            let storage = is_storage_image_opcode(binding.opcode);
            let comparison = image_is_comparison(binding.opcode);
            let decoded = decode_image_descriptor(&binding.resource_descriptor);
            let fallback = decoded.is_err();
            let descriptor = decoded
                .or_else(|_| fallback_image_descriptor(binding))
                .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
            if trace_target_sampling()
                && descriptor.guest.width == 1_920
                && descriptor.guest.height == 1_080
            {
                static LOGGED: OnceLock<Mutex<HashSet<(u64, u32, u32, u32, bool, bool)>>> =
                    OnceLock::new();
                let normalized_coordinates =
                    sampler_uses_normalized_coordinates(&binding.sampler_descriptor);
                let key = (
                    descriptor.guest.address,
                    descriptor.guest.tile_mode,
                    descriptor.destination_select,
                    descriptor.mip_level_count,
                    comparison,
                    normalized_coordinates,
                );
                let mut logged = LOGGED
                    .get_or_init(|| Mutex::new(HashSet::new()))
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if logged.len() < 128 && logged.insert(key) {
                    let byte_count = guest_texture_byte_count(descriptor.guest);
                    let mapping = byte_count.and_then(|byte_count| {
                        guest_shared_mapping_identity(descriptor.guest.address, byte_count)
                    });
                    let mapping_start =
                        guest_shared_mapping_identity(descriptor.guest.address, 1);
                    let mapping_end = byte_count
                        .and_then(|byte_count| descriptor.guest.address.checked_add(byte_count - 1))
                        .and_then(|address| guest_shared_mapping_identity(address, 1));
                    let exact = resident_guest_textures
                        .get(&descriptor.guest)
                        .map(|resident| (resident.gpu_written, resident.content_version));
                    let aliases = resident_guest_textures
                        .values()
                        .filter(|resident| {
                            guest_texture_descriptors_overlap(
                                resident.descriptor,
                                descriptor.guest,
                            )
                        })
                        .map(|resident| {
                            (
                                resident.descriptor.address,
                                resident.descriptor.width,
                                resident.descriptor.height,
                                resident.descriptor.pixel_format,
                                resident.descriptor.tile_mode,
                                resident.gpu_written,
                                resident.content_version,
                            )
                        })
                        .collect::<Vec<_>>();
                    let multisample_candidates = multisample_textures
                        .iter()
                        .filter(|(key, _)| {
                            guest_texture_descriptors_have_same_layout(
                                descriptor.guest,
                                key.descriptor,
                            )
                        })
                        .map(|(key, multisample)| {
                            (
                                key.descriptor.address,
                                key.sample_count,
                                multisample.initialized,
                                guest_texture_descriptors_overlap(
                                    descriptor.guest,
                                    key.descriptor,
                                ),
                                resident_guest_textures
                                    .get(&key.descriptor)
                                    .map(|resident| {
                                        (resident.gpu_written, resident.content_version)
                                    }),
                            )
                        })
                        .collect::<Vec<_>>();
                    eprintln!(
                        "[REMu][METAL][SAMPLE] pc={:#x} op={} guest={:#x} 1920x1080 format={:?} tile={} select={:#x} mips={} type={} comparison={} normalized={} sampler={:08x?} bytes={byte_count:x?} mapping={mapping:?} endpoints={mapping_start:?}/{mapping_end:?} exact={exact:?} aliases={aliases:?} msaa={multisample_candidates:?}",
                        binding.pc,
                        binding.opcode,
                        descriptor.guest.address,
                        descriptor.guest.pixel_format,
                        descriptor.guest.tile_mode,
                        descriptor.destination_select,
                        descriptor.mip_level_count,
                        descriptor.texture_type,
                        comparison,
                        normalized_coordinates,
                        binding.sampler_descriptor,
                    );
                }
            }
            let underlying_mip_count = descriptor
                .base_mip_level
                .checked_add(descriptor.mip_level_count.max(1))
                .ok_or("sampled resident mip range overflows")?;
            if matches!(descriptor.texture_type, 9 | 10) {
                let replaced = if let Some(resident) = resident_guest_textures
                .get_mut(&descriptor.guest)
                .filter(|resident| {
                    resident.texture_type == descriptor.texture_type
                        && resident.layer_count == descriptor.layer_count
                        && resident.mip_level_count >= underlying_mip_count
                })
                {
                    refresh_resident_texture_write_watch(resident)?
                        == WriteWatchRefresh::Replaced
                } else {
                    false
                };
                if replaced {
                    resident_guest_textures.remove(&descriptor.guest);
                }
            }
            let multisample_resolve = (!comparison && descriptor.texture_type == 9)
                .then(|| {
                    multisample_textures
                        .iter()
                        .filter(|(_, multisample)| multisample.initialized)
                        .filter(|(key, _)| {
                            render_resolve_candidate_matches(
                                descriptor.guest,
                                key.descriptor,
                                key.sample_count,
                            )
                        })
                        .filter_map(|(key, _)| {
                            let resident = resident_guest_textures.get(&key.descriptor)?;
                            resident.gpu_written.then_some((
                                resident.content_version,
                                key.descriptor,
                                resident.texture.clone(),
                            ))
                        })
                        .max_by_key(|(version, _, _)| *version)
                })
                .flatten();
            let resident_descriptor = matches!(descriptor.texture_type, 9 | 10)
                .then(|| {
                    (!comparison)
                        .then(|| {
                            resident_guest_textures
                                .get(&descriptor.guest)
                                .filter(|resident| {
                                    resident.texture_type == descriptor.texture_type
                                        && resident.layer_count == descriptor.layer_count
                                        && resident.mip_level_count >= underlying_mip_count
                                })
                                .map(|resident| resident.descriptor)
                        })
                        .flatten()
                        .or_else(|| {
                            (descriptor.texture_type == 9).then(|| {
                                resident_guest_textures
                                    .values()
                                    .filter(|resident| {
                                        resident.texture_type == 9
                                            && resident.layer_count == 1
                                            && resident.mip_level_count >= underlying_mip_count
                                            && ((!comparison
                                                && (guest_texture_descriptors_share_storage(
                                                    resident.descriptor,
                                                    descriptor.guest,
                                                ) || resident_component_reordered_sample_is_eligible(
                                                    resident,
                                                    descriptor,
                                                    comparison,
                                                )))
                                                || resident_depth_sample_is_eligible(
                                                    resident, descriptor,
                                                ))
                                    })
                                    .max_by_key(|resident| resident.content_version)
                                    .map(|resident| resident.descriptor)
                            })?
                        })
                })
                .flatten();
            let resident_resolve = resident_descriptor.and_then(|resident_descriptor| {
                let resident = resident_guest_textures.get_mut(&resident_descriptor)?;
                let version = resident.content_version;
                resident_texture_for_sample(resident, descriptor)
                    .map(|texture| (version, texture))
            });
            // A guest descriptor can retain an exact single-sample resident from an earlier
            // render phase while the current frame is produced through a distinct MSAA target
            // with the same layout. Address equality alone is therefore not sufficient: choose
            // whichever compatible GPU image was written most recently. This also naturally
            // retains an exact post-process result when it is newer than the MSAA resolve.
            let select_multisample = match (&resident_resolve, &multisample_resolve) {
                (Some((resident_version, _)), Some((multisample_version, _, _))) => {
                    multisample_version > resident_version
                }
                (None, Some(_)) => true,
                _ => false,
            };
            if trace_target_sampling()
                && descriptor.guest.width == 1_920
                && descriptor.guest.height == 1_080
                && select_multisample
                && let Some((version, source, _)) = multisample_resolve.as_ref()
            {
                static LOGGED_RESOLVES: OnceLock<Mutex<HashSet<(u64, u64)>>> = OnceLock::new();
                let key = (descriptor.guest.address, source.address);
                let mut logged = LOGGED_RESOLVES
                    .get_or_init(|| Mutex::new(HashSet::new()))
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if logged.len() < 256 && logged.insert(key) {
                    eprintln!(
                        "[REMu][METAL][MSAA-SAMPLE] guest={:#x} source={:#x} version={} format={:?} tile={}",
                        descriptor.guest.address,
                        source.address,
                        version,
                        source.pixel_format,
                        source.tile_mode,
                    );
                }
            }
            let render_resolve = if select_multisample {
                multisample_resolve
                    .as_ref()
                    .map(|(_, _, texture)| texture.clone())
            } else {
                resident_resolve
                    .as_ref()
                    .map(|(_, texture)| texture.clone())
            };
            let texture = if fallback {
                let key = sampled_texture_key(descriptor, comparison);
                if let Some(texture) = fallback_texture_cache.get(&key) {
                    texture.clone()
                } else {
                    let (_, _, _, bytes_per_element) =
                        texture_element_layout(descriptor.guest)?;
                    let byte_count = bytes_per_element
                        .checked_mul(
                            usize::try_from(descriptor.layer_count)
                                .map_err(|_| "fallback image layer count does not fit usize")?,
                        )
                        .ok_or("fallback image byte count overflows")?;
                    let mut linear = vec![0; byte_count];
                    if matches!(
                        descriptor.guest.pixel_format,
                        MTLPixelFormat::RGBA8Unorm
                            | MTLPixelFormat::RGBA8Unorm_sRGB
                            | MTLPixelFormat::RGBA8Uint
                            | MTLPixelFormat::RGBA8Sint
                    ) {
                        for texel in linear.chunks_exact_mut(4) {
                            texel[3] = u8::MAX;
                        }
                    }
                    let texture = create_sampled_guest_texture(
                        device,
                        queue,
                        descriptor.guest,
                        &linear,
                        0,
                        1,
                        descriptor.texture_type,
                        descriptor.layer_count,
                        comparison,
                    )?;
                    texture.set_label("REMu fallback guest image");
                    fallback_texture_cache.insert(key, texture.clone());
                    texture
                }
            } else if storage && descriptor.texture_type == 9 {
                if storage_image_requires_mip_hierarchy(binding, descriptor) {
                    ensure_mipmapped_resident_texture_in_map(
                        device,
                        descriptor,
                        resident_guest_textures,
                        Mode27WorkOrigin::StorageCreate,
                    )?
                } else {
                    ensure_resident_texture_in_map(
                        device,
                        descriptor.guest,
                        resident_guest_textures,
                        true,
                    )?
                }
            } else if let Some(texture) = render_resolve {
                texture
            } else {
                let key = sampled_texture_key(descriptor, comparison);
                let reload = if let Some(cached) = sampled_texture_cache.get_mut(&key) {
                    let refresh = refresh_sampled_texture_write_watch(
                        key,
                        cached,
                        sampled_texture_write_watches,
                    )?;
                    sampled_texture_cache_needs_reload(refresh, cached.write_watch.is_dirty())
                } else {
                    false
                };
                if reload {
                    // CPU-produced images such as AvPlayer's rotating NV12 frames can be updated
                    // immediately before a draw without an intervening ACQUIRE_MEM packet. The
                    // write watch has already observed that store, so retaining the Metal cache
                    // here would sample the zero-filled/predecessor frame forever. Rebuild from a
                    // stable watched snapshot; the persistent watch is shared with the new entry
                    // and is rearmed by snapshot_watched_sampled_texture_raw.
                    if let Some(retired) = sampled_texture_cache.remove(&key) {
                        SAMPLED_TEXTURE_CACHE_DIRTY_RELOADS.fetch_add(1, Ordering::Relaxed);
                        retire_sampled_texture(retired);
                    }
                }
                if let Some(cached) = sampled_texture_cache.get(&key) {
                    SAMPLED_TEXTURE_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                    cached.texture.clone()
                } else {
                    SAMPLED_TEXTURE_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
                    let write_watch = persistent_sampled_texture_write_watch(
                        key,
                        descriptor,
                        sampled_texture_write_watches,
                    )
                        .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
                    let raw = snapshot_watched_sampled_texture_raw(descriptor, &write_watch)
                        .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
                    let texture =
                        if sampled_texture_uses_exact_mode27_mips(descriptor) {
                            create_exact_mode27_sampled_mip_view(
                                device,
                                descriptor,
                                &raw,
                                comparison,
                            )
                            .map_err(|error| {
                                format!("image at PC {:#x}: {error}", binding.pc)
                            })?
                        } else {
                            let linear =
                                decode_sampled_texture_linear_from_raw(descriptor, &raw).map_err(
                                    |error| format!("image at PC {:#x}: {error}", binding.pc),
                                )?;
                            create_sampled_guest_texture(
                                device,
                                queue,
                                descriptor.guest,
                                &linear,
                                descriptor.base_mip_level,
                                descriptor.mip_level_count,
                                descriptor.texture_type,
                                descriptor.layer_count,
                                comparison,
                            )?
                        };
                    texture.set_label(&format!(
                        "REMu sampled guest image {:#x}",
                        descriptor.guest.address
                    ));
                    while sampled_texture_cache.len() >= SAMPLED_TEXTURE_CACHE_CAPACITY {
                        let Some((old_key, admission_id)) =
                            sampled_texture_admission_order.pop_front()
                        else {
                            break;
                        };
                        let is_current = sampled_texture_cache
                            .get(&old_key)
                            .is_some_and(|cached| cached.admission_id == admission_id);
                        if is_current {
                            if let Some(retired) = sampled_texture_cache.remove(&old_key) {
                                SAMPLED_TEXTURE_CACHE_CAPACITY_EVICTIONS
                                    .fetch_add(1, Ordering::Relaxed);
                                retire_sampled_texture(retired);
                            }
                        }
                    }
                    let admission_id = *next_sampled_texture_admission_id;
                    *next_sampled_texture_admission_id =
                        next_sampled_texture_admission_id.wrapping_add(1);
                    sampled_texture_cache.insert(
                        key,
                        CachedSampledTexture {
                            texture: texture.clone(),
                            write_watch,
                            admission_id,
                        },
                    );
                    sampled_texture_admission_order.push_back((key, admission_id));
                    texture
                }
            };
            Ok(UploadedImageTexture {
                texture,
                descriptor,
                // The resident 2D path has exact guest writeback today. A writable 3D image stays
                // resident in the shared sampled-image cache so later GPU reads see its contents;
                // it must not invalidate that cache through the 2D writeback path.
                writable: !fallback && writable && descriptor.texture_type == 9,
            })
        })
        .collect()
}

pub(super) fn storage_image_requires_mip_hierarchy(
    binding: &ImageBinding,
    descriptor: DecodedImageDescriptor,
) -> bool {
    // IMAGE_LOAD/STORE address the resource view's BASE_LEVEL. LAST_LEVEL only bounds the view
    // and does not make an ordinary base-zero access touch every exposed mip. Only the _MIP
    // variants consume a per-invocation mip coordinate. Avoid promoting a normal full-view UAV
    // to a complete CPU-decoded hierarchy; besides doing needless work, that incorrectly rejects
    // valid SW_64KB_S (mode 9) base-level resources.
    descriptor.base_mip_level != 0
        || binding.opcode.is("ImageLoadMip")
        || binding.opcode.is("ImageStoreMip")
}
#[cfg(test)]
pub(super) fn guest_texture_raw_matches(
    descriptor: GuestTextureDescriptor,
    shadow: &[u8],
) -> Result<bool, String> {
    let (source_address, source_length) = guest_texture_raw_range(descriptor)?;
    // SAFETY: See snapshot_guest_texture_raw; the comparison does not retain the borrowed range.
    let source = unsafe { std::slice::from_raw_parts(source_address as *const u8, source_length) };
    Ok(source == shadow)
}

#[cfg(test)]
pub(super) fn snapshot_guest_texture_linear(
    descriptor: GuestTextureDescriptor,
) -> Result<Vec<u8>, String> {
    let (source_address, source_length) = guest_texture_raw_range(descriptor)?;
    // SAFETY: See snapshot_guest_texture_raw; decoding completes before this function returns.
    let source = unsafe { std::slice::from_raw_parts(source_address as *const u8, source_length) };
    decode_guest_texture_linear_from_raw(descriptor, source)
}

#[cfg(test)]
pub(super) fn write_guest_texture_linear(
    descriptor: GuestTextureDescriptor,
    linear: &[u8],
) -> Result<(), String> {
    let byte_count = guest_texture_byte_count(descriptor)
        .ok_or("guest texture byte count overflows the host address space")?;
    if descriptor.address < 0x10_000 || byte_count > 128 * 1024 * 1024 {
        return Err(format!(
            "invalid guest texture range address={:#x} bytes={byte_count}",
            descriptor.address
        ));
    }
    let (width, height, _pitch, bytes_per_element) = texture_element_layout(descriptor)?;
    let required = width
        .checked_mul(height)
        .and_then(|elements| elements.checked_mul(bytes_per_element))
        .ok_or("guest texture dimensions overflow")?;
    if linear.len() != required {
        return Err(format!(
            "tight guest image has {} bytes, expected {required}",
            linear.len()
        ));
    }
    let destination_length =
        usize::try_from(byte_count).map_err(|_| "guest texture byte count does not fit usize")?;
    if guest_shared_mapping_identity(descriptor.address, byte_count).is_some() {
        let destination_address = usize::try_from(descriptor.address)
            .map_err(|_| "guest texture address does not fit usize")?;
        // SAFETY: the registry proved the full writable texture range is one live contiguous
        // mapping, and the caller's write watch has not been retired.
        let destination = unsafe {
            std::slice::from_raw_parts_mut(destination_address as *mut u8, destination_length)
        };
        write_guest_texture_linear_into_raw(descriptor, linear, destination)?;
        return Ok(());
    }
    let mut destination = vec![0; destination_length];
    let has_shared_backing = read_guest_shared_backing(descriptor.address, &mut destination);
    #[cfg(test)]
    if !has_shared_backing {
        let destination_address = usize::try_from(descriptor.address)
            .map_err(|_| "guest texture address does not fit usize")?;
        // SAFETY: process-local unit-test descriptors retain the full destination allocation.
        destination.copy_from_slice(unsafe {
            std::slice::from_raw_parts(destination_address as *const u8, destination_length)
        });
    }
    #[cfg(not(test))]
    if !has_shared_backing {
        return Err(format!(
            "guest texture range {:#x}+{byte_count:#x} has no complete live shared backing",
            descriptor.address
        ));
    }
    write_guest_texture_linear_into_raw(descriptor, linear, &mut destination)?;
    if has_shared_backing {
        if !write_guest_shared_backing(descriptor.address, &destination) {
            return Err(format!(
                "guest texture range {:#x}+{byte_count:#x} lost shared backing during writeback",
                descriptor.address
            ));
        }
    } else {
        #[cfg(test)]
        {
            let destination_address = usize::try_from(descriptor.address)
                .map_err(|_| "guest texture address does not fit usize")?;
            // SAFETY: process-local unit-test descriptors retain the full destination allocation.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    destination.as_ptr(),
                    destination_address as *mut u8,
                    destination.len(),
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn apply_image_destination_select(
    rgba: &mut [u8],
    destination_select: u32,
) -> Result<(), String> {
    if destination_select == 0 || destination_select == 0xfac {
        return Ok(());
    }
    let selectors =
        std::array::from_fn::<_, 4, _>(|channel| (destination_select >> (channel * 3)) & 0x7);
    if selectors
        .iter()
        .any(|selector| !matches!(selector, 0 | 1 | 4..=7))
    {
        return Err(format!(
            "invalid image destination select {destination_select:#x}"
        ));
    }
    for pixel in rgba.chunks_exact_mut(4) {
        let source: [u8; 4] = pixel.try_into().expect("RGBA pixel has four channels");
        for (channel, selector) in selectors.iter().copied().enumerate() {
            pixel[channel] = match selector {
                0 => 0,
                1 => 255,
                4..=7 => source[usize::try_from(selector - 4).expect("channel fits usize")],
                _ => unreachable!("validated destination selector"),
            };
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn prepare_sampled_texture_upload(
    descriptor: GuestTextureDescriptor,
    mut linear: Vec<u8>,
    destination_select: u32,
) -> Result<(GuestTextureDescriptor, Vec<u8>), String> {
    if image_destination_select_is_native(descriptor.pixel_format, destination_select) {
        return Ok((descriptor, linear));
    }
    if descriptor.bytes_per_pixel == 4
        && matches!(
            descriptor.pixel_format,
            MTLPixelFormat::RGBA8Unorm
                | MTLPixelFormat::RGBA8Unorm_sRGB
                | MTLPixelFormat::RGBA8Snorm
                | MTLPixelFormat::RGBA8Uint
                | MTLPixelFormat::RGBA8Sint
                | MTLPixelFormat::BGRA8Unorm
                | MTLPixelFormat::BGRA8Unorm_sRGB
        )
    {
        apply_image_destination_select(&mut linear, destination_select)?;
        return Ok((descriptor, linear));
    }
    if matches!(
        descriptor.pixel_format,
        MTLPixelFormat::R8Unorm | MTLPixelFormat::R8Unorm_sRGB
    ) {
        let selectors =
            std::array::from_fn::<_, 4, _>(|channel| (destination_select >> (channel * 3)) & 0x7);
        if selectors
            .iter()
            .any(|selector| !matches!(selector, 0 | 1 | 4..=7))
        {
            return Err(format!(
                "invalid image destination select {destination_select:#x}"
            ));
        }
        let mut rgba = Vec::with_capacity(linear.len().saturating_mul(4));
        for red in linear {
            for &selector in &selectors {
                rgba.push(match selector {
                    0 | 5 | 6 => 0,
                    1 | 7 => u8::MAX,
                    4 => red,
                    _ => unreachable!("validated destination selector"),
                });
            }
        }
        return Ok((
            GuestTextureDescriptor {
                pixel_format: if descriptor.pixel_format == MTLPixelFormat::R8Unorm_sRGB {
                    MTLPixelFormat::RGBA8Unorm_sRGB
                } else {
                    MTLPixelFormat::RGBA8Unorm
                },
                bytes_per_pixel: 4,
                ..descriptor
            },
            rgba,
        ));
    }
    Err(format!(
        "destination select {destination_select:#x} is not representable for {:?}",
        descriptor.pixel_format
    ))
}

#[cfg(test)]
pub(super) fn image_destination_select_is_native(
    pixel_format: MTLPixelFormat,
    destination_select: u32,
) -> bool {
    if destination_select == 0 {
        return true;
    }
    let native_select = match pixel_format {
        MTLPixelFormat::R8Unorm
        | MTLPixelFormat::R8Snorm
        | MTLPixelFormat::R8Uint
        | MTLPixelFormat::R8Sint
        | MTLPixelFormat::R8Unorm_sRGB
        | MTLPixelFormat::R16Unorm
        | MTLPixelFormat::R16Snorm
        | MTLPixelFormat::R16Uint
        | MTLPixelFormat::R16Sint
        | MTLPixelFormat::R16Float
        | MTLPixelFormat::R32Uint
        | MTLPixelFormat::R32Sint
        | MTLPixelFormat::R32Float => 0x204,
        MTLPixelFormat::RG8Unorm
        | MTLPixelFormat::RG8Snorm
        | MTLPixelFormat::RG8Uint
        | MTLPixelFormat::RG8Sint
        | MTLPixelFormat::RG8Unorm_sRGB
        | MTLPixelFormat::RG16Unorm
        | MTLPixelFormat::RG16Snorm
        | MTLPixelFormat::RG16Uint
        | MTLPixelFormat::RG16Sint
        | MTLPixelFormat::RG16Float
        | MTLPixelFormat::RG32Uint
        | MTLPixelFormat::RG32Sint
        | MTLPixelFormat::RG32Float => 0x22c,
        MTLPixelFormat::RG11B10Float | MTLPixelFormat::RGB9E5Float => 0x3ac,
        _ => 0xfac,
    };
    destination_select == native_select
}
