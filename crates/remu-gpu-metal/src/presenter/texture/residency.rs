// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::completion::commit_and_settle;
use super::{
    DecodedDepthTarget, DecodedImageDescriptor, DecodedRenderTarget, DirectScanoutTexture,
    GuestTextureDescriptor, MetalPresenter, Mode27WorkOrigin, MultisampleResidentTexture,
    MultisampleTextureKey, ResidentGuestTexture, ResidentTextureActivityStamp,
    ScanoutSnapshotTexture, WriteWatchRefresh, create_mipmapped_guest_texture,
    decode_guest_texture_linear_from_raw, decode_mode27_mip_chain_from_raw,
    decode_sampled_texture_linear_from_raw, guest_texture_byte_count,
    guest_texture_descriptors_overlap, guest_texture_descriptors_share_storage,
    initial_depth_clear_for_target, mode27_mip_chain_layout, promote_resident_mip_hierarchy,
    record_mode27_work_detail, refresh_resident_texture_write_watch,
    resident_texture_eviction_plan, resident_texture_for_sample,
    retire_multisample_textures_overlapping_writeback, retire_overlapping_sampled_textures,
    retire_resident_textures_overlapping_writeback, retire_sampled_textures_overlapping_writeback,
    snapshot_watched_guest_range, snapshot_watched_guest_texture_raw,
    snapshot_watched_sampled_texture_raw, texture_element_layout, trace_residency_churn,
    trace_resident_texture_removal, watch_guest_texture, watch_resident_texture,
    watch_sampled_texture, write_resident_texture_contents_to_guest,
};
use metal::{
    Device, MTLLoadAction, MTLOrigin, MTLRegion, MTLSize, MTLStorageMode, MTLStoreAction,
    MTLTextureType, MTLTextureUsage, RenderPassDescriptor, Texture, TextureDescriptor,
};
use remu_memory::{CpuWriteWatchRetirement, guest_shared_backing_identity};
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::time::Instant;

/// Read-only attribution for persistent Metal render/storage targets.
///
/// GPU-owned residents may contain the only current copy of partially rendered pixels and cannot
/// be dropped merely to satisfy a host-memory budget. Mapping retirement and repeated physical
/// backing identities are the semantics-bearing signals needed to distinguish stale residents
/// from live render-graph allocations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ResidentTextureOwnershipStats {
    pub(crate) gpu_owned_count: usize,
    pub(crate) gpu_owned_bytes: u64,
    pub(crate) cpu_owned_count: usize,
    pub(crate) cpu_owned_bytes: u64,
    pub(crate) active_mapping_count: usize,
    pub(crate) active_mapping_bytes: u64,
    pub(crate) reprotected_mapping_count: usize,
    pub(crate) reprotected_mapping_bytes: u64,
    pub(crate) reprotecting_mapping_count: usize,
    pub(crate) reprotecting_mapping_bytes: u64,
    pub(crate) replaced_mapping_count: usize,
    pub(crate) replaced_mapping_bytes: u64,
    pub(crate) unique_backing_bases: usize,
    pub(crate) duplicate_backing_bases: usize,
    pub(crate) unique_backing_spans: usize,
    pub(crate) duplicate_backing_spans: usize,
    pub(crate) unresolved_backing_count: usize,
    pub(crate) unresolved_backing_bytes: u64,
}

pub(crate) fn resident_texture_ownership_stats(
    residents: &HashMap<GuestTextureDescriptor, ResidentGuestTexture>,
) -> ResidentTextureOwnershipStats {
    let mut stats = ResidentTextureOwnershipStats::default();
    let mut backing_bases = HashSet::new();
    let mut backing_spans = HashSet::new();
    for resident in residents.values() {
        let allocated_bytes = u64::try_from(resident.texture.allocated_size()).unwrap_or(u64::MAX);
        if resident.gpu_written {
            stats.gpu_owned_count += 1;
            stats.gpu_owned_bytes = stats.gpu_owned_bytes.saturating_add(allocated_bytes);
        } else {
            stats.cpu_owned_count += 1;
            stats.cpu_owned_bytes = stats.cpu_owned_bytes.saturating_add(allocated_bytes);
        }

        let mapping_is_live = match resident.write_watch.retirement() {
            None => {
                stats.active_mapping_count += 1;
                stats.active_mapping_bytes =
                    stats.active_mapping_bytes.saturating_add(allocated_bytes);
                true
            }
            Some(CpuWriteWatchRetirement::Reprotected) => {
                stats.reprotected_mapping_count += 1;
                stats.reprotected_mapping_bytes = stats
                    .reprotected_mapping_bytes
                    .saturating_add(allocated_bytes);
                true
            }
            Some(CpuWriteWatchRetirement::Reprotecting) => {
                stats.reprotecting_mapping_count += 1;
                stats.reprotecting_mapping_bytes = stats
                    .reprotecting_mapping_bytes
                    .saturating_add(allocated_bytes);
                true
            }
            Some(CpuWriteWatchRetirement::Replaced) => {
                stats.replaced_mapping_count += 1;
                stats.replaced_mapping_bytes =
                    stats.replaced_mapping_bytes.saturating_add(allocated_bytes);
                false
            }
        };
        if !mapping_is_live {
            continue;
        }
        let (address, byte_count) = resident.write_watch.range();
        let Some(identity) = guest_shared_backing_identity(address, byte_count) else {
            stats.unresolved_backing_count += 1;
            stats.unresolved_backing_bytes = stats
                .unresolved_backing_bytes
                .saturating_add(allocated_bytes);
            continue;
        };
        if !backing_bases.insert((identity.backing_id, identity.backing_offset)) {
            stats.duplicate_backing_bases += 1;
        }
        if !backing_spans.insert((identity.backing_id, identity.backing_offset, byte_count)) {
            stats.duplicate_backing_spans += 1;
        }
    }
    stats.unique_backing_bases = backing_bases.len();
    stats.unique_backing_spans = backing_spans.len();
    stats
}

impl MetalPresenter {
    pub(crate) fn garbage_collect_resident_textures_to_budget(
        &mut self,
        byte_capacity: u64,
        current_flip: u64,
        maximum_evictions: usize,
    ) -> Result<(usize, u64), String> {
        let plan = resident_texture_eviction_plan(
            &self.resident_guest_textures,
            byte_capacity,
            current_flip,
            maximum_evictions,
        );
        if plan.descriptors.is_empty() {
            return Ok((0, 0));
        }

        // End every encoder before observing or replacing resident allocations. Command buffers
        // retain their bound textures until completion, and the single Metal queue orders the
        // following readbacks after all producer work.
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

        let mut eviction_count = 0_usize;
        let mut eviction_bytes = 0_u64;
        for descriptor in plan.descriptors {
            let Some(resident) = self.resident_guest_textures.get_mut(&descriptor) else {
                // Writing back an older overlapping version may already have retired this alias.
                continue;
            };
            let allocated_bytes =
                u64::try_from(resident.texture.allocated_size()).unwrap_or(u64::MAX);
            let refresh = refresh_resident_texture_write_watch(resident)?;
            if refresh == WriteWatchRefresh::Reprotecting {
                // A concurrent CPU write is changing watch ownership. Leave this resident in
                // place and retry at a later flip rather than publishing pixels into that range.
                continue;
            }
            if refresh == WriteWatchRefresh::Replaced {
                if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
                    trace_resident_texture_removal("pressure-mapping-replaced", &resident);
                }
                eviction_count += 1;
                eviction_bytes = eviction_bytes.saturating_add(allocated_bytes);
                continue;
            }

            let gpu_written = resident.gpu_written;
            if gpu_written && resident.cpu_write_obsoletes_gpu_writeback(refresh)? {
                if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
                    trace_resident_texture_removal("pressure-cpu-reused", &resident);
                }
                eviction_count += 1;
                eviction_bytes = eviction_bytes.saturating_add(allocated_bytes);
                continue;
            }

            if gpu_written {
                let (alias_ranges, content_version) = {
                    let resident = self
                        .resident_guest_textures
                        .get(&descriptor)
                        .expect("pressure candidate survived watch refresh");
                    (
                        resident.write_watch.alias_ranges(),
                        resident.content_version,
                    )
                };
                // A sampled alias can retain a texture view of the same allocation. Retire every
                // overlapping interpretation before publishing the authoritative pixels to RAM,
                // and discard older resident/MSAA versions that the writeback supersedes.
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
                    .get(&descriptor)
                    .expect("pressure source survives overlapping-version retirement");
                write_resident_texture_contents_to_guest(&device, &queue, resident)?;
            }

            if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
                trace_resident_texture_removal("pressure-lru", &resident);
            }
            eviction_count += 1;
            eviction_bytes = eviction_bytes.saturating_add(allocated_bytes);
        }
        if eviction_count != 0 {
            eprintln!(
                "[REMu][METAL] resident texture pressure GC evicted {eviction_count} images/{:.1} MiB (resident {:.1} MiB, budget {:.1} MiB)",
                eviction_bytes as f64 / (1024.0 * 1024.0),
                plan.resident_bytes as f64 / (1024.0 * 1024.0),
                byte_capacity as f64 / (1024.0 * 1024.0),
            );
        }
        Ok((eviction_count, eviction_bytes))
    }

    pub(crate) fn garbage_collect_resident_textures(&mut self, current_flip: u64) {
        let Some(byte_capacity) = super::super::resident_texture_cache_byte_capacity() else {
            return;
        };
        if let Err(error) = self.garbage_collect_resident_textures_to_budget(
            byte_capacity,
            current_flip,
            super::super::MAX_RESIDENT_TEXTURE_EVICTIONS_PER_FLIP,
        ) {
            eprintln!("[REMu][METAL] resident texture pressure GC failed: {error}");
        }
    }

    pub(crate) fn ensure_mipmapped_storage_texture(
        &mut self,
        device: &Device,
        descriptor: DecodedImageDescriptor,
    ) -> Result<Texture, String> {
        let underlying_mip_count = descriptor
            .base_mip_level
            .checked_add(descriptor.mip_level_count)
            .ok_or("storage-image mip range overflows")?;
        let needs_gpu_promotion = self
            .resident_guest_textures
            .get(&descriptor.guest)
            .is_some_and(|resident| {
                resident.gpu_written
                    && resident.texture_type == 9
                    && resident.layer_count == 1
                    && resident.mip_level_count < underlying_mip_count
            });
        if needs_gpu_promotion {
            let queue = self
                .command_queue
                .as_ref()
                .ok_or("Metal command queue is unavailable")?
                .clone();
            self.ensure_open_guest_batch(&queue);
            let command_buffer = {
                let batch = self
                    .open_guest_batch
                    .as_mut()
                    .expect("guest batch was opened for mip promotion");
                batch.end_render_pass();
                batch.command_buffer.clone()
            };
            let refresh = refresh_resident_texture_write_watch(
                self.resident_guest_textures
                    .get_mut(&descriptor.guest)
                    .expect("selected GPU-written resident exists"),
            )?;
            if refresh != WriteWatchRefresh::Replaced {
                let resident = self
                    .resident_guest_textures
                    .remove(&descriptor.guest)
                    .expect("selected GPU-written resident exists");
                let promoted = promote_resident_mip_hierarchy(
                    device,
                    &command_buffer,
                    descriptor,
                    resident,
                    Mode27WorkOrigin::StorageCreate,
                )?;
                self.resident_guest_textures
                    .insert(descriptor.guest, promoted);
                return resident_texture_for_sample(
                    self.resident_guest_textures
                        .get_mut(&descriptor.guest)
                        .expect("promoted storage hierarchy exists"),
                    descriptor,
                )
                .ok_or_else(|| "promoted storage hierarchy cannot represent view".to_owned());
            }
            if let Some(resident) = self.resident_guest_textures.remove(&descriptor.guest) {
                trace_resident_texture_removal("mip-promotion-mapping-replaced", &resident);
            }
        }
        let needs_clean_promotion = self
            .resident_guest_textures
            .get(&descriptor.guest)
            .is_some_and(|resident| {
                !resident.gpu_written
                    && resident.texture_type == 9
                    && resident.layer_count == 1
                    && resident.mip_level_count < underlying_mip_count
            });
        let promotion_command_buffer = if needs_clean_promotion {
            let queue = self
                .command_queue
                .as_ref()
                .ok_or("Metal command queue is unavailable")?
                .clone();
            self.ensure_open_guest_batch(&queue);
            let batch = self
                .open_guest_batch
                .as_mut()
                .expect("guest batch was opened for mip promotion");
            batch.end_render_pass();
            Some(batch.command_buffer.clone())
        } else {
            None
        };
        super::ensure_mipmapped_resident_texture_in_map(
            device,
            promotion_command_buffer.as_deref(),
            descriptor,
            &mut self.resident_guest_textures,
            Mode27WorkOrigin::StorageCreate,
        )
    }

    pub(crate) fn ensure_resident_texture(
        &mut self,
        device: &Device,
        descriptor: GuestTextureDescriptor,
    ) -> Result<Texture, String> {
        self.ensure_resident_texture_with_initial_contents(device, descriptor, true)
    }

    pub(crate) fn ensure_resident_texture_with_initial_contents(
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
        if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
            trace_resident_texture_removal("ensure-exact-resident-replaced", &resident);
        }
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
                    if let Some(resident) = self.resident_guest_textures.remove(&shared_resident) {
                        trace_resident_texture_removal(
                            "ensure-shared-resident-mapping-replaced",
                            &resident,
                        );
                    }
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
            trace_residency_churn(
                descriptor,
                upload_guest_contents,
                aliases.iter().filter_map(|(_, alias)| {
                    self.resident_guest_textures.get(alias).map(|resident| {
                        (
                            resident.descriptor,
                            resident.gpu_written,
                            resident.content_version,
                        )
                    })
                }),
            );
            let has_gpu_written_alias = aliases.iter().any(|(_, alias)| {
                self.resident_guest_textures
                    .get(alias)
                    .is_some_and(|resident| resident.gpu_written)
            });
            let queue = if has_gpu_written_alias {
                self.wait_for_guest_command_batches()?;
                self.multisample_textures.retain(|key, _| {
                    !guest_texture_descriptors_overlap(key.descriptor, descriptor)
                });
                Some(
                    self.command_queue
                        .as_ref()
                        .ok_or("Metal command queue is unavailable")?
                        .clone(),
                )
            } else {
                None
            };
            aliases.sort_by_key(|(version, _)| *version);
            for (_, alias) in aliases {
                let refresh = refresh_resident_texture_write_watch(
                    self.resident_guest_textures
                        .get_mut(&alias)
                        .expect("collected resident alias exists"),
                )?;
                let gpu_written = self
                    .resident_guest_textures
                    .get(&alias)
                    .is_some_and(|resident| resident.gpu_written);
                let cpu_write_obsoletes_writeback = self
                    .resident_guest_textures
                    .get_mut(&alias)
                    .expect("refreshed resident alias exists")
                    .cpu_write_obsoletes_gpu_writeback(refresh)?;
                if refresh == WriteWatchRefresh::Reprotecting && gpu_written {
                    return Err(format!(
                        "resident alias mapping {:#x} is being reprotected",
                        alias.address
                    ));
                }
                if refresh == WriteWatchRefresh::Replaced {
                    if let Some(resident) = self.resident_guest_textures.remove(&alias) {
                        trace_resident_texture_removal("alias-mapping-replaced", &resident);
                    }
                } else if gpu_written {
                    let resident = self
                        .resident_guest_textures
                        .remove(&alias)
                        .expect("collected GPU-written resident alias exists");
                    if cpu_write_obsoletes_writeback {
                        // The guest reused this heap range after the last render. Publishing the
                        // old Metal image now would overwrite the new indirect-command bytes.
                        trace_resident_texture_removal("cpu-reused-gpu-alias", &resident);
                        continue;
                    }
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
                    write_resident_texture_contents_to_guest(
                        device,
                        queue
                            .as_ref()
                            .expect("GPU-written aliases acquired the command queue"),
                        &resident,
                    )?;
                }
            }
        }
        ensure_resident_texture_in_map(
            device,
            descriptor,
            &mut self.resident_guest_textures,
            upload_guest_contents,
        )
    }

    pub(crate) fn ensure_mipmapped_render_target_texture(
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
            let queue = self
                .command_queue
                .as_ref()
                .ok_or("Metal command queue is unavailable")?
                .clone();
            self.ensure_open_guest_batch(&queue);
            let command_buffer = {
                let batch = self
                    .open_guest_batch
                    .as_mut()
                    .expect("guest batch was opened for mip promotion");
                batch.end_render_pass();
                batch.command_buffer.clone()
            };
            retire_overlapping_sampled_textures(&mut self.sampled_texture_cache, target.texture);
            let resident = self
                .resident_guest_textures
                .remove(&target.texture)
                .expect("selected GPU-written target exists");
            let descriptor = DecodedImageDescriptor {
                guest: target.texture,
                destination_select: 0,
                base_mip_level: 0,
                mip_level_count: target.mip_level_count,
                texture_type: target.texture_type,
                layer_count: target.layer_count,
            };
            let promoted = promote_resident_mip_hierarchy(
                device,
                &command_buffer,
                descriptor,
                resident,
                Mode27WorkOrigin::RenderTargetCreate,
            )?;
            let texture = promoted.texture.clone();
            self.resident_guest_textures
                .insert(target.texture, promoted);
            return Ok(texture);
        }
        if let Some(resident) = self.resident_guest_textures.remove(&target.texture) {
            trace_resident_texture_removal("mipmapped-target-resident-replaced", &resident);
        }
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
                gpu_write_watch_generations: Vec::new(),
                last_use: ResidentTextureActivityStamp::default(),
                activity: super::super::ResidentTextureActivity::enabled(),
            },
        );
        Ok(texture)
    }

    pub(crate) fn ensure_layered_render_target_texture(
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
        if let Some(resident) = self.resident_guest_textures.remove(&target.texture) {
            trace_resident_texture_removal("layered-target-resident-replaced", &resident);
        }
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
                gpu_write_watch_generations: Vec::new(),
                last_use: ResidentTextureActivityStamp::default(),
                activity: super::super::ResidentTextureActivity::enabled(),
            },
        );
        Ok(texture)
    }

    pub(crate) fn ensure_depth_render_target_texture(
        &mut self,
        device: &Device,
        depth: DecodedDepthTarget,
        minimum_width: u32,
        minimum_height: u32,
    ) -> Result<Texture, String> {
        let texture = if depth.layer_count == 1 {
            self.ensure_resident_texture(device, depth.texture)?
        } else {
            self.ensure_layered_depth_render_target_texture(device, depth)?
        };
        let host_width = depth.texture.width.max(minimum_width);
        let host_height = depth.texture.height.max(minimum_height);
        if texture.width() >= u64::from(host_width) && texture.height() >= u64::from(host_height) {
            return Ok(texture);
        }

        // Unreal can bind a hierarchically sized DB surface beside a larger color target. The
        // console expands the fast-cleared depth allocation to the active framebuffer extent;
        // Metal instead requires every attachment to cover the explicit render-pass extent.
        // Promote the host depth image, initialize the extra area to a comparison-neutral value,
        // and preserve the guest-declared portion. This prevents menus and scenes from being
        // clipped to the small DB metadata extent.
        self.wait_for_guest_command_batches()?;
        let queue = self
            .command_queue
            .as_ref()
            .ok_or("Metal command queue is unavailable")?
            .clone();
        let metal_descriptor = TextureDescriptor::new();
        metal_descriptor.set_texture_type(if depth.layer_count == 1 {
            MTLTextureType::D2
        } else {
            MTLTextureType::D2Array
        });
        metal_descriptor.set_pixel_format(depth.texture.pixel_format);
        metal_descriptor.set_width(u64::from(host_width));
        metal_descriptor.set_height(u64::from(host_height));
        if depth.layer_count > 1 {
            metal_descriptor.set_array_length(u64::from(depth.layer_count));
        }
        metal_descriptor.set_storage_mode(MTLStorageMode::Shared);
        metal_descriptor.set_usage(
            MTLTextureUsage::RenderTarget
                | MTLTextureUsage::ShaderRead
                | MTLTextureUsage::ShaderWrite,
        );
        let promoted = device.new_texture(&metal_descriptor);
        promoted.set_label(&format!(
            "REMu promoted {}-layer depth image {:#x} {}x{}",
            depth.layer_count, depth.texture.address, host_width, host_height
        ));

        let command_buffer = queue.new_command_buffer().to_owned();
        command_buffer.set_label("REMu depth-target promotion");
        let pass = RenderPassDescriptor::new();
        pass.set_render_target_width(u64::from(host_width));
        pass.set_render_target_height(u64::from(host_height));
        pass.set_render_target_array_length(u64::from(depth.layer_count));
        let attachment = pass
            .depth_attachment()
            .ok_or("Metal render pass has no depth attachment")?;
        attachment.set_texture(Some(&promoted));
        if depth.layer_count > 1 {
            attachment.set_slice(0);
        }
        attachment.set_load_action(MTLLoadAction::Clear);
        attachment.set_store_action(MTLStoreAction::Store);
        attachment.set_clear_depth(initial_depth_clear_for_target(depth));
        let encoder = command_buffer.new_render_command_encoder(pass);
        encoder.end_encoding();

        let blit = command_buffer.new_blit_command_encoder();
        for layer in 0..depth.layer_count {
            blit.copy_from_texture(
                &texture,
                u64::from(layer),
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize::new(
                    texture.width().min(u64::from(host_width)),
                    texture.height().min(u64::from(host_height)),
                    1,
                ),
                &promoted,
                u64::from(layer),
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
            );
        }
        blit.end_encoding();
        commit_and_settle(&command_buffer, "depth-target promotion")?;

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

    fn ensure_layered_depth_render_target_texture(
        &mut self,
        device: &Device,
        depth: DecodedDepthTarget,
    ) -> Result<Texture, String> {
        if depth.layer_count <= 1 || depth.sample_count != 1 {
            return Err(format!(
                "layered depth target requires multiple single-sample slices, got layers={} samples={}",
                depth.layer_count, depth.sample_count
            ));
        }
        const DEPTH_ARRAY_TEXTURE_TYPE: u32 = 13;
        if let Some(resident) =
            self.resident_guest_textures
                .get_mut(&depth.texture)
                .filter(|resident| {
                    resident.texture_type == DEPTH_ARRAY_TEXTURE_TYPE
                        && resident.layer_count == depth.layer_count
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

        let face_bytes = guest_texture_byte_count(depth.texture)
            .ok_or("layered depth target face byte count overflows")?;
        let array_bytes = face_bytes
            .checked_mul(u64::from(depth.layer_count))
            .ok_or("layered depth target byte count overflows")?;
        let aliases = self
            .resident_guest_textures
            .values()
            .filter(|resident| {
                resident.gpu_written
                    && resident
                        .write_watch
                        .overlaps_alias_range(depth.texture.address, array_bytes)
            })
            .map(|resident| resident.descriptor)
            .collect::<Vec<_>>();
        if !aliases.is_empty() {
            self.wait_for_guest_command_batches()?;
            let queue = self
                .command_queue
                .as_ref()
                .ok_or("Metal command queue is unavailable")?
                .clone();
            for alias in aliases {
                let refresh = refresh_resident_texture_write_watch(
                    self.resident_guest_textures
                        .get_mut(&alias)
                        .expect("collected layered-depth alias exists"),
                )?;
                let mut resident = self
                    .resident_guest_textures
                    .remove(&alias)
                    .expect("collected layered-depth alias exists");
                if refresh != WriteWatchRefresh::Replaced
                    && !resident.cpu_write_obsoletes_gpu_writeback(refresh)?
                {
                    let alias_ranges = resident.write_watch.alias_ranges();
                    retire_sampled_textures_overlapping_writeback(
                        &mut self.sampled_texture_cache,
                        &alias_ranges,
                    );
                    write_resident_texture_contents_to_guest(device, &queue, &resident)?;
                } else if refresh != WriteWatchRefresh::Replaced {
                    trace_resident_texture_removal("cpu-reused-layered-depth-alias", &resident);
                } else {
                    trace_resident_texture_removal(
                        "layered-depth-alias-mapping-replaced",
                        &resident,
                    );
                }
            }
        }
        if let Some(resident) = self.resident_guest_textures.remove(&depth.texture) {
            trace_resident_texture_removal("layered-depth-resident-replaced", &resident);
        }

        let descriptor = DecodedImageDescriptor {
            guest: depth.texture,
            destination_select: 0,
            base_mip_level: 0,
            mip_level_count: 1,
            texture_type: DEPTH_ARRAY_TEXTURE_TYPE,
            layer_count: depth.layer_count,
        };
        let write_watch = watch_sampled_texture(descriptor)?;
        retire_sampled_textures_overlapping_writeback(
            &mut self.sampled_texture_cache,
            &write_watch.alias_ranges(),
        );
        let raw = snapshot_watched_sampled_texture_raw(descriptor, &write_watch)?;
        let linear = decode_sampled_texture_linear_from_raw(descriptor, &raw)?;
        let texture = create_layered_depth_render_target_texture(device, descriptor, &linear)?;
        texture.set_label(&format!(
            "REMu {}-layer guest depth image {:#x}",
            depth.layer_count, depth.texture.address
        ));
        self.resident_guest_textures.insert(
            depth.texture,
            ResidentGuestTexture {
                texture: texture.clone(),
                component_reorder_view: None,
                descriptor: depth.texture,
                texture_type: DEPTH_ARRAY_TEXTURE_TYPE,
                layer_count: depth.layer_count,
                mip_level_count: 1,
                write_watch,
                gpu_written: false,
                content_version: 0,
                gpu_write_watch_generations: Vec::new(),
                last_use: ResidentTextureActivityStamp::default(),
                activity: super::super::ResidentTextureActivity::enabled(),
            },
        );
        Ok(texture)
    }

    pub(crate) fn ensure_direct_scanout_texture(
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

    pub(crate) fn ensure_scanout_snapshot_texture(
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

    pub(crate) fn ensure_multisample_texture(
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

    pub(crate) fn mark_multisample_texture_initialized(
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

    #[cfg(test)]
    pub(crate) fn mark_resident_texture_written(&mut self, descriptor: GuestTextureDescriptor) {
        self.mark_resident_texture_written_at(descriptor, ResidentTextureActivityStamp::default());
    }

    pub(crate) fn mark_resident_texture_written_at(
        &mut self,
        descriptor: GuestTextureDescriptor,
        activity_stamp: ResidentTextureActivityStamp,
    ) {
        self.mark_resident_texture_written_preserving_multisample_at(descriptor, activity_stamp);
        self.multisample_textures
            .retain(|key, _| !guest_texture_descriptors_overlap(key.descriptor, descriptor));
    }

    pub(crate) fn mark_resident_texture_written_preserving_multisample_at(
        &mut self,
        descriptor: GuestTextureDescriptor,
        activity_stamp: ResidentTextureActivityStamp,
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
        if let Some(resident_descriptor) = resident_descriptor {
            // Multiple read-only interpretations of the same guest bytes may coexist. Creating a
            // second clean view must not evict and immediately reconstruct the first one. Once a
            // view is actually written, however, every other clean overlapping interpretation is
            // stale and can be retired without a writeback.
            let stale_aliases = self
                .resident_guest_textures
                .iter()
                .filter_map(|(candidate, resident)| {
                    (*candidate != resident_descriptor
                        && !resident.gpu_written
                        && guest_texture_descriptors_overlap(*candidate, descriptor))
                    .then_some(*candidate)
                })
                .collect::<Vec<_>>();
            for stale_alias in stale_aliases {
                if let Some(resident) = self.resident_guest_textures.remove(&stale_alias) {
                    trace_resident_texture_removal("gpu-write-invalidates-clean-alias", &resident);
                }
            }
            let resident = self
                .resident_guest_textures
                .get_mut(&resident_descriptor)
                .expect("selected resident survived clean-alias retirement");
            resident.note_write(activity_stamp);
            resident.gpu_written = true;
            resident.content_version = self.resident_content_version;
            resident.note_gpu_write();
        }
        retire_overlapping_sampled_textures(&mut self.sampled_texture_cache, descriptor);
    }

    pub(crate) fn note_resident_texture_read_at(
        &mut self,
        descriptor: GuestTextureDescriptor,
        activity_stamp: ResidentTextureActivityStamp,
    ) {
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
        if let Some(resident) = resident_descriptor.and_then(|resident_descriptor| {
            self.resident_guest_textures.get_mut(&resident_descriptor)
        }) {
            resident.note_read(activity_stamp);
        }
    }
}

pub(crate) fn ensure_resident_texture_in_map(
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
    if let Some(resident) = resident_guest_textures.remove(&descriptor) {
        let reason = if resident.write_watch.retirement() == Some(CpuWriteWatchRetirement::Replaced)
        {
            "ensure-map-mapping-replaced"
        } else {
            "ensure-map-incompatible-resident"
        };
        trace_resident_texture_removal(reason, &resident);
    }
    let write_watch = watch_guest_texture(descriptor)?;
    let usage = MTLTextureUsage::RenderTarget
        | MTLTextureUsage::ShaderRead
        | MTLTextureUsage::ShaderWrite
        | MTLTextureUsage::PixelFormatView;
    let texture = if upload_guest_contents {
        let raw = snapshot_watched_guest_texture_raw(descriptor, &write_watch)?;
        let linear = decode_guest_texture_linear_from_raw(descriptor, &raw)?;
        create_guest_texture(device, descriptor, usage, &linear)?
    } else {
        // A discarded render target still needs a CPU-content baseline. Mapping a later coherent
        // alias retires this watch and requests validation; without initialized fingerprints an
        // unchanged guest allocation would look CPU-modified and could discard newer GPU output.
        let _ = snapshot_watched_guest_texture_raw(descriptor, &write_watch)?;
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
            gpu_write_watch_generations: Vec::new(),
            last_use: ResidentTextureActivityStamp::default(),
            activity: super::super::ResidentTextureActivity::enabled(),
        },
    );
    Ok(texture)
}

pub(crate) fn create_guest_texture(
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

pub(crate) fn create_uninitialized_guest_texture(
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

pub(crate) fn create_layered_render_target_texture(
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

fn create_layered_depth_render_target_texture(
    device: &Device,
    descriptor: DecodedImageDescriptor,
    linear: &[u8],
) -> Result<Texture, String> {
    if descriptor.texture_type != 13 || descriptor.layer_count <= 1 {
        return Err(format!(
            "layered depth target requires a 2D array, got type={} layers={}",
            descriptor.texture_type, descriptor.layer_count
        ));
    }
    let (elements_wide, elements_high, _, bytes_per_element) =
        texture_element_layout(descriptor.guest)?;
    let bytes_per_row = elements_wide
        .checked_mul(bytes_per_element)
        .ok_or("layered depth row size overflows")?;
    let bytes_per_image = bytes_per_row
        .checked_mul(elements_high)
        .ok_or("layered depth image size overflows")?;
    let required = bytes_per_image
        .checked_mul(
            usize::try_from(descriptor.layer_count)
                .map_err(|_| "layered depth layer count does not fit usize")?,
        )
        .ok_or("layered depth byte count overflows")?;
    if linear.len() != required {
        return Err(format!(
            "invalid layered depth upload {}x{}x{} format {:?} with {} bytes (expected {required})",
            descriptor.guest.width,
            descriptor.guest.height,
            descriptor.layer_count,
            descriptor.guest.pixel_format,
            linear.len()
        ));
    }
    let metal_descriptor = TextureDescriptor::new();
    metal_descriptor.set_texture_type(MTLTextureType::D2Array);
    metal_descriptor.set_pixel_format(descriptor.guest.pixel_format);
    metal_descriptor.set_width(u64::from(descriptor.guest.width));
    metal_descriptor.set_height(u64::from(descriptor.guest.height));
    metal_descriptor.set_array_length(u64::from(descriptor.layer_count));
    metal_descriptor.set_storage_mode(MTLStorageMode::Shared);
    metal_descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    let texture = device.new_texture(&metal_descriptor);
    for layer in 0..descriptor.layer_count {
        let offset = usize::try_from(layer)
            .expect("validated depth layer fits usize")
            .checked_mul(bytes_per_image)
            .expect("validated depth layer offset fits usize");
        texture.replace_region_in_slice(
            MTLRegion::new_2d(
                0,
                0,
                u64::from(descriptor.guest.width),
                u64::from(descriptor.guest.height),
            ),
            0,
            u64::from(layer),
            linear[offset..].as_ptr().cast::<c_void>(),
            u64::try_from(bytes_per_row).expect("validated depth row size fits u64"),
            u64::try_from(bytes_per_image).expect("validated depth image size fits u64"),
        );
    }
    Ok(texture)
}
