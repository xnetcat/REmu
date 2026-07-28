// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CachedSampledTexture, DecodedImageDescriptor, GuestTextureDescriptor, GuestTextureWriteWatch,
    MetalPresenter, Mode27WorkOrigin, RESIDENT_TEXTURE_ACQUIRE_INVALIDATIONS, ResidentGuestTexture,
    SAMPLED_TEXTURE_CACHE_ACQUIRE_INVALIDATIONS, SAMPLED_TEXTURE_CACHE_CAPACITY, SampledTextureKey,
    TEXTURE_SNAPSHOT_BYTES, TEXTURE_SNAPSHOT_COUNT, TEXTURE_SNAPSHOT_NANOSECONDS,
    WriteWatchRefresh, create_guest_texture, create_mipmapped_guest_texture,
    decode_guest_texture_linear_from_raw, decode_mode27_mip_chain_from_raw,
    guest_texture_byte_count, guest_texture_descriptors_overlap, mode5_mip_chain_layout,
    mode27_mip_chain_layout, record_mode27_work_detail, record_texture_dirty_work_detail,
    record_texture_snapshot_work_detail, record_texture_work, retire_sampled_texture,
};
use metal::MTLTextureUsage;
use remu_gpu::GpuMemoryAcquire;
use remu_memory::{
    CpuWriteWatch, CpuWriteWatchRetirement, guest_shared_backing_alias_segments,
    guest_shared_mapping_identity, read_guest_shared_backing, write_guest_shared_backing,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

impl MetalPresenter {
    pub(crate) fn synchronize_cpu_written_guest_resources(
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
}

impl GuestTextureWriteWatch {
    pub(crate) fn new(address: u64, byte_count: u64) -> Result<Self, String> {
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

    pub(crate) fn retirement(&self) -> Option<CpuWriteWatchRetirement> {
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

    pub(crate) fn consume_dirty(&self) -> bool {
        self.ranges
            .iter()
            .fold(false, |dirty, watch| watch.consume_dirty() || dirty)
    }

    pub(crate) fn is_dirty(&self) -> bool {
        self.ranges.iter().any(CpuWriteWatch::is_dirty)
    }

    pub(crate) fn precise_tracking(&self) -> bool {
        self.ranges.iter().all(CpuWriteWatch::precise_tracking)
    }

    pub(crate) fn generations(&self) -> Vec<u64> {
        self.ranges.iter().map(CpuWriteWatch::generation).collect()
    }

    pub(crate) fn dirty_page_ranges(&self) -> Vec<(u64, u64)> {
        self.ranges
            .iter()
            .flat_map(CpuWriteWatch::dirty_page_ranges)
            .collect()
    }

    pub(crate) fn rearm(&self) -> Result<(), String> {
        for watch in &self.ranges {
            watch
                .rearm()
                .map_err(|error| format!("could not rearm guest texture CPU writes: {error}"))?;
        }
        Ok(())
    }

    pub(crate) fn range(&self) -> (u64, u64) {
        (self.address, self.byte_count)
    }

    pub(crate) fn with_contiguous_mapping<T>(
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

    pub(crate) fn with_live_backing<T>(&self, operation: impl FnOnce() -> T) -> Result<T, String> {
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

    pub(crate) fn has_single_covering_mapping(&self, address: u64, byte_count: u64) -> bool {
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

    pub(crate) fn overlaps_alias_range(&self, address: u64, byte_count: u64) -> bool {
        Self::ranges_overlap_alias(&self.ranges, address, byte_count, |watch| watch.range())
    }

    pub(crate) fn alias_ranges(&self) -> Vec<(u64, u64)> {
        self.ranges.iter().map(CpuWriteWatch::range).collect()
    }

    pub(crate) fn overlaps_alias_ranges(&self, ranges: &[(u64, u64)]) -> bool {
        ranges
            .iter()
            .any(|&(address, byte_count)| self.overlaps_alias_range(address, byte_count))
    }

    pub(crate) fn ranges_overlap_alias<T>(
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

pub(crate) fn refresh_resident_texture_write_watch(
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

pub(crate) fn watch_resident_texture(
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

pub(crate) fn refresh_sampled_texture_write_watch(
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

pub(crate) fn persistent_sampled_texture_write_watch(
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

pub(crate) fn guest_texture_raw_range(
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

pub(crate) fn snapshot_guest_texture_raw(
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

pub(crate) fn watch_guest_texture(
    descriptor: GuestTextureDescriptor,
) -> Result<GuestTextureWriteWatch, String> {
    let (_, byte_count) = guest_texture_raw_range(descriptor)?;
    let byte_count =
        u64::try_from(byte_count).map_err(|_| "guest texture byte count does not fit u64")?;
    GuestTextureWriteWatch::new(descriptor.address, byte_count)
}

pub(crate) fn sampled_texture_raw_range(
    descriptor: DecodedImageDescriptor,
) -> Result<(usize, usize), String> {
    let exact_mode5_mip_chain = sampled_texture_uses_exact_mode5_mips(descriptor);
    let exact_mode27_mip_chain = sampled_texture_uses_exact_mode27_mips(descriptor);
    let byte_count = if exact_mode5_mip_chain || exact_mode27_mip_chain {
        if exact_mode27_mip_chain && (descriptor.texture_type != 9 || descriptor.layer_count != 1) {
            return Err(format!(
                "exact mode-27 sampled mip-chain upload requires a 2D single-layer resource, got type={} layers={} tile={} base={} count={}",
                descriptor.texture_type,
                descriptor.layer_count,
                descriptor.guest.tile_mode,
                descriptor.base_mip_level,
                descriptor.mip_level_count
            ));
        }
        if exact_mode5_mip_chain && !matches!(descriptor.texture_type, 9 | 11 | 13) {
            return Err(format!(
                "exact mode-5 sampled mip-chain upload requires a 2D or array resource, got type={} layers={} tile={} base={} count={}",
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
        let byte_count = if exact_mode5_mip_chain {
            mode5_mip_chain_layout(descriptor.guest, underlying_mip_count)?
                .byte_count
                .checked_mul(
                    usize::try_from(descriptor.layer_count)
                        .map_err(|_| "sampled layer count does not fit usize")?,
                )
                .ok_or("mode-5 sampled mip-chain byte count overflows")?
        } else {
            mode27_mip_chain_layout(descriptor.guest, underlying_mip_count)?.byte_count
        };
        u64::try_from(byte_count).map_err(|_| "sampled mip-chain byte count does not fit u64")?
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

pub(crate) fn sampled_texture_uses_exact_mode5_mips(descriptor: DecodedImageDescriptor) -> bool {
    descriptor.guest.tile_mode == 5
        && (descriptor.base_mip_level != 0 || descriptor.mip_level_count != 1)
}

pub(crate) fn sampled_texture_uses_exact_mode27_mips(descriptor: DecodedImageDescriptor) -> bool {
    descriptor.guest.tile_mode == 27
        && (descriptor.base_mip_level != 0 || descriptor.mip_level_count != 1)
}

pub(crate) fn watch_sampled_texture(
    descriptor: DecodedImageDescriptor,
) -> Result<GuestTextureWriteWatch, String> {
    let (_, byte_count) = sampled_texture_raw_range(descriptor)?;
    GuestTextureWriteWatch::new(
        descriptor.guest.address,
        u64::try_from(byte_count).map_err(|_| "sampled byte count does not fit u64")?,
    )
}

#[track_caller]
pub(crate) fn snapshot_watched_sampled_texture_raw(
    descriptor: DecodedImageDescriptor,
    write_watch: &GuestTextureWriteWatch,
) -> Result<Vec<u8>, String> {
    let (_, source_length) = sampled_texture_raw_range(descriptor)?;
    snapshot_watched_guest_range(descriptor.guest.address, source_length, write_watch)
}

#[track_caller]
pub(crate) fn snapshot_watched_guest_range(
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
pub(crate) fn snapshot_watched_guest_texture_raw(
    descriptor: GuestTextureDescriptor,
    write_watch: &GuestTextureWriteWatch,
) -> Result<Vec<u8>, String> {
    let (_, source_length) = guest_texture_raw_range(descriptor)?;
    snapshot_watched_guest_range(descriptor.address, source_length, write_watch)
}

pub(crate) fn write_watched_guest_range(
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
