// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CachedSampledTexture, DecodedImageDescriptor, GuestTextureContentState, GuestTextureDescriptor,
    GuestTextureWatchRange, GuestTextureWriteWatch, MetalPresenter,
    RESIDENT_TEXTURE_ACQUIRE_INVALIDATIONS, ResidentGuestTexture,
    SAMPLED_TEXTURE_CACHE_ACQUIRE_INVALIDATIONS, SAMPLED_TEXTURE_CACHE_CAPACITY, SampledTextureKey,
    TEXTURE_DIRTY_VALIDATION_BYTES, TEXTURE_DIRTY_VALIDATION_COUNT,
    TEXTURE_DIRTY_VALIDATION_NANOSECONDS, TEXTURE_DIRTY_VALIDATION_UNCHANGED,
    TEXTURE_SNAPSHOT_BYTES, TEXTURE_SNAPSHOT_COUNT, TEXTURE_SNAPSHOT_NANOSECONDS,
    TEXTURE_SNAPSHOT_UNSTABLE_COUNT, WriteWatchRefresh, guest_texture_byte_count,
    guest_texture_descriptors_overlap, mode5_mip_chain_layout, mode27_mip_chain_layout,
    record_texture_dirty_work_detail, record_texture_snapshot_work_detail, record_texture_work,
    retire_sampled_texture, texture_alias_trace_match, trace_alias_correlated_acquire,
    trace_resident_texture_removal,
};
use remu_gpu::GpuMemoryAcquire;
use remu_memory::{
    CpuWriteWatch, CpuWriteWatchRetirement, guest_shared_backing_alias_segments,
    guest_shared_mapping_epoch, guest_shared_mapping_identity, read_guest_shared_backing,
    write_guest_shared_backing,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// Validation granularity for page-dirty guest image contents.
///
/// This is deliberately smaller than the 16 KiB guest page and independent of the host page
/// size. A false-positive protection fault therefore hashes only the touched neighborhood rather
/// than copying and decoding a complete multi-megabyte image.
const CONTENT_FINGERPRINT_CHUNK_BYTES: u64 = 4 * 1024;

/// Attempts allowed while trying to copy a guest resource nobody else is writing.
///
/// A store that lands during the copy invalidates it, and retrying keeps an ordinary
/// false-positive protection fault from publishing torn pixels. A surface the guest rewrites every
/// frame never converges, though, and each attempt re-copies the entire resource: an unbounded
/// retry turned a single 64 MiB Unreal render target into seconds of presenter time, which starved
/// submission while the GPU sat idle. Publishing the last copy is the same outcome the hardware
/// gives an unsynchronized guest, and the range stays dirty so ordinary content validation
/// re-examines it on the next use.
const MAX_STABLE_SNAPSHOT_ATTEMPTS: u32 = 4;

impl MetalPresenter {
    pub(crate) fn synchronize_cpu_written_guest_resources(
        &mut self,
        acquire: GpuMemoryAcquire,
    ) -> Result<(), String> {
        if !acquire.invalidates_guest_resources() {
            return Ok(());
        }
        let mut invalidated_multisample_descriptors = Vec::new();
        let resident_descriptors = self
            .resident_guest_textures
            .keys()
            .copied()
            .collect::<Vec<_>>();
        for descriptor in resident_descriptors {
            let Some((address, byte_count)) = self
                .resident_guest_textures
                .get(&descriptor)
                .map(|resident| resident.write_watch.range())
            else {
                continue;
            };
            // ACQUIRE_MEM carries an exact cache-invalidation scope. Refreshing unrelated
            // watches here caused every packet to walk the entire texture set, and could turn
            // unrelated mapping reprotection into a whole-texture reload. A later use of an
            // unrelated resource still refreshes its watch through the ordinary residency path.
            if !acquire.overlaps(address, byte_count) {
                continue;
            }
            let alias_trace = texture_alias_trace_match(descriptor);
            let alias_trace_state = alias_trace.as_ref().map(|_| {
                let resident = self
                    .resident_guest_textures
                    .get(&descriptor)
                    .expect("collected resident texture exists");
                (
                    resident.gpu_written,
                    resident.write_watch.is_dirty(),
                    resident.write_watch.precise_tracking(),
                    resident.write_watch.dirty_page_ranges(),
                    resident.write_watch.generations(),
                    resident.gpu_write_watch_generations.clone(),
                )
            });
            let trace_acquire = |resident: &ResidentGuestTexture,
                                 refresh: WriteWatchRefresh,
                                 after_owner: &str,
                                 content_changed: Option<bool>,
                                 outcome: &str| {
                let (
                    Some(trace),
                    Some((
                        before_gpu_owned,
                        dirty,
                        precise,
                        dirty_pages,
                        watch_generations_before,
                        gpu_write_generations_before,
                    )),
                ) = (alias_trace.as_ref(), alias_trace_state.as_ref())
                else {
                    return;
                };
                trace_alias_correlated_acquire(
                    trace,
                    acquire,
                    *before_gpu_owned,
                    after_owner,
                    refresh,
                    *dirty,
                    *precise,
                    dirty_pages,
                    watch_generations_before,
                    &resident.write_watch.generations(),
                    gpu_write_generations_before,
                    &resident.gpu_write_watch_generations,
                    content_changed,
                    outcome,
                );
            };
            let refresh = refresh_resident_texture_write_watch(
                self.resident_guest_textures
                    .get_mut(&descriptor)
                    .expect("collected resident texture exists"),
            )?;
            match refresh {
                WriteWatchRefresh::Replaced => {
                    if let Some(resident) = self.resident_guest_textures.get(&descriptor) {
                        trace_acquire(
                            resident,
                            refresh,
                            "removed",
                            None,
                            "removed-mapping-replaced",
                        );
                    }
                    if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
                        trace_resident_texture_removal("acquire-mapping-replaced", &resident);
                    }
                    continue;
                }
                WriteWatchRefresh::Reprotecting => {
                    let resident = self
                        .resident_guest_textures
                        .get(&descriptor)
                        .expect("reprotecting resident texture exists");
                    trace_acquire(
                        resident,
                        refresh,
                        if resident.gpu_written { "gpu" } else { "cpu" },
                        None,
                        "retained-reprotecting",
                    );
                    continue;
                }
                WriteWatchRefresh::Active | WriteWatchRefresh::Renewed { .. } => {}
            }
            let resident = self
                .resident_guest_textures
                .get_mut(&descriptor)
                .expect("overlapping resident survived write-watch refresh");
            let gpu_written = resident.gpu_written;
            let content_changed = resident.write_watch.content_has_changed()?;
            if !content_changed {
                if resident.gpu_written {
                    // A permission-only renewal resets watch generations. Once sparse content
                    // validation proves that guest bytes did not change, move the GPU ownership
                    // boundary into the new generation domain instead of retiring it later.
                    resident.note_gpu_write();
                }
                trace_acquire(
                    resident,
                    refresh,
                    if resident.gpu_written { "gpu" } else { "cpu" },
                    Some(false),
                    if resident.gpu_written {
                        "retained-gpu"
                    } else {
                        "retained-cpu"
                    },
                );
                continue;
            }
            // An acquire only invalidates the old Metal copy; it does not require materializing a
            // replacement here. Rebuilding eagerly decoded whole multi-megabyte Unreal render
            // allocations even when several more acquires arrived before the image was used, and
            // even when the next render pass discards those guest bytes anyway. Drop the stale
            // resident and let the ordinary render/sample path materialize one stable snapshot on
            // demand. A GPU-authored resident is safe to drop as well: the reload it replaces
            // overwrote every pixel with guest bytes, so no Metal-only content survived it either.
            invalidated_multisample_descriptors.push(descriptor);
            trace_acquire(
                resident,
                refresh,
                "removed",
                Some(true),
                if gpu_written {
                    "removed-gpu-dirty"
                } else {
                    "removed-cpu-dirty"
                },
            );
            if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
                trace_resident_texture_removal(
                    if gpu_written {
                        "acquire-gpu-dirty"
                    } else {
                        "acquire-cpu-dirty"
                    },
                    &resident,
                );
            }
            RESIDENT_TEXTURE_ACQUIRE_INVALIDATIONS.fetch_add(1, Ordering::Relaxed);
        }
        self.multisample_textures.retain(|key, _| {
            !invalidated_multisample_descriptors
                .iter()
                .any(|descriptor| guest_texture_descriptors_overlap(key.descriptor, *descriptor))
        });

        let sampled_keys = self
            .sampled_texture_cache
            .keys()
            .copied()
            .collect::<Vec<_>>();
        for key in sampled_keys {
            let Some((address, byte_count)) = self
                .sampled_texture_cache
                .get(&key)
                .map(|cached| cached.write_watch.range())
            else {
                continue;
            };
            if !acquire.overlaps(address, byte_count) {
                continue;
            }
            let refresh = refresh_sampled_texture_write_watch(
                key,
                self.sampled_texture_cache
                    .get_mut(&key)
                    .expect("overlapping sampled texture exists"),
                &mut self.sampled_texture_write_watches,
            )?;
            match refresh {
                WriteWatchRefresh::Replaced => {
                    if let Some(retired) = self.sampled_texture_cache.remove(&key) {
                        retire_sampled_texture(retired);
                    }
                    continue;
                }
                WriteWatchRefresh::Reprotecting => continue,
                WriteWatchRefresh::Active | WriteWatchRefresh::Renewed { .. } => {}
            }
            if !self
                .sampled_texture_cache
                .get(&key)
                .expect("overlapping sampled texture survived write-watch refresh")
                .write_watch
                .content_has_changed()?
            {
                continue;
            }
            // ACQUIRE_MEM makes CPU-authored bytes authoritative, but it does not require eagerly
            // materializing a new Metal texture. Evict the stale cache entry and let the ordinary
            // upload path rebuild it from a stable watched snapshot only if a later shader
            // actually samples it. This also coalesces repeated acquires before first use.
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
        loop {
            let epoch = guest_shared_mapping_epoch();
            if epoch & 1 != 0 {
                std::thread::yield_now();
                continue;
            }
            match Self::new_at_stable_epoch(address, byte_count) {
                Ok(watch) if guest_shared_mapping_epoch() == epoch => return Ok(watch),
                Ok(watch) => {
                    // A mapping transition raced alias discovery or watch installation. Dropping
                    // these partial watches restores their pages before retrying against the
                    // complete alias set.
                    drop(watch);
                }
                Err(error) if guest_shared_mapping_epoch() == epoch => return Err(error),
                Err(_) => {}
            }
        }
    }

    fn new_at_stable_epoch(address: u64, byte_count: u64) -> Result<Self, String> {
        let segments = guest_shared_backing_alias_segments(address, byte_count).ok_or_else(|| {
            format!(
                "guest texture range {address:#x}+{byte_count:#x} has no complete live shared backing"
            )
        })?;
        let canonical_identity = guest_shared_mapping_identity(address, 1).ok_or_else(|| {
            format!("guest texture base {address:#x} has no live shared-backing identity")
        })?;
        let ranges = segments
            .into_iter()
            .map(|segment| {
                let alias_identity =
                    guest_shared_mapping_identity(segment.address, 1).ok_or_else(|| {
                        format!(
                            "guest texture alias {:#x} has no live shared-backing identity",
                            segment.address
                        )
                    })?;
                if alias_identity.backing_id != canonical_identity.backing_id
                    || alias_identity.backing_offset < canonical_identity.backing_offset
                {
                    return Err(format!(
                        "guest texture alias {:#x} does not belong to canonical backing at {address:#x}",
                        segment.address
                    ));
                }
                let canonical_address = address
                    .checked_add(alias_identity.backing_offset - canonical_identity.backing_offset)
                    .ok_or("guest texture canonical alias address overflows")?;
                // SAFETY: the backing resolver returned live readable/writable CPU aliases that
                // cover this exact GPU-physical resource span.
                unsafe { CpuWriteWatch::new(segment.address, segment.byte_count) }
                    .map(|watch| GuestTextureWatchRange {
                        watch,
                        canonical_address,
                    })
                    .map_err(|error| {
                        format!(
                            "could not watch guest texture CPU writes at {:#x}+{:#x}: {error}",
                            segment.address, segment.byte_count
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let chunk_count = usize::try_from(byte_count.div_ceil(CONTENT_FINGERPRINT_CHUNK_BYTES))
            .map_err(|_| "guest texture fingerprint chunk count does not fit usize")?;
        Ok(Self {
            address,
            byte_count,
            ranges,
            content_state: Arc::new(std::sync::Mutex::new(GuestTextureContentState {
                chunk_fingerprints: vec![0; chunk_count],
                initialized_chunks: vec![false; chunk_count],
                pending_dirty_ranges: Vec::new(),
            })),
        })
    }

    pub(crate) fn retirement(&self) -> Option<CpuWriteWatchRetirement> {
        let mut retirement = None;
        for range in &self.ranges {
            match range.watch.retirement() {
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
            .fold(false, |dirty, range| range.watch.consume_dirty() || dirty)
    }

    pub(crate) fn is_dirty(&self) -> bool {
        self.ranges.iter().any(|range| range.watch.is_dirty())
    }

    pub(crate) fn precise_tracking(&self) -> bool {
        self.ranges
            .iter()
            .all(|range| range.watch.precise_tracking())
    }

    pub(crate) fn generations(&self) -> Vec<u64> {
        self.ranges
            .iter()
            .map(|range| range.watch.generation())
            .collect()
    }

    pub(crate) fn dirty_page_ranges(&self) -> Vec<(u64, u64)> {
        self.ranges
            .iter()
            .flat_map(|range| {
                let (alias_address, _) = range.watch.range();
                range.watch.dirty_page_ranges().into_iter().filter_map(
                    move |(dirty_address, dirty_bytes)| {
                        let offset = dirty_address.checked_sub(alias_address)?;
                        let canonical_address = range.canonical_address.checked_add(offset)?;
                        Some((canonical_address, dirty_bytes))
                    },
                )
            })
            .collect()
    }

    pub(crate) fn rearm(&self) -> Result<(), String> {
        for range in &self.ranges {
            range
                .watch
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
            .find(|range| {
                let (watch_address, watch_bytes) = range.watch.range();
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
        watch.watch.with_live_mapping(operation).map_err(|error| {
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
            .watch
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
        self.ranges.iter().any(|range| {
            let (watch_address, watch_bytes) = range.watch.range();
            watch_address <= address
                && watch_address
                    .checked_add(watch_bytes)
                    .is_some_and(|watch_end| end <= watch_end)
        })
    }

    fn alias_address_for_canonical_range(&self, address: u64, byte_count: u64) -> Option<u64> {
        self.ranges.iter().find_map(|range| {
            let offset = address.checked_sub(range.canonical_address)?;
            let (alias_address, alias_byte_count) = range.watch.range();
            (offset.checked_add(byte_count)? <= alias_byte_count)
                .then(|| alias_address.checked_add(offset))
                .flatten()
        })
    }

    /// Copies a canonical physical range through whichever live CPU aliases cover it.
    ///
    /// # Safety
    ///
    /// The caller must hold [`Self::with_live_backing`] for the duration of this operation.
    unsafe fn copy_from_live_aliases(
        &self,
        address: u64,
        destination: &mut [u8],
    ) -> Result<(), String> {
        let byte_count = u64::try_from(destination.len())
            .map_err(|_| "canonical texture copy length does not fit u64")?;
        let end = address
            .checked_add(byte_count)
            .ok_or("canonical texture copy range overflows")?;
        let mut cursor = address;
        while cursor < end {
            let (range, canonical_end) = self
                .ranges
                .iter()
                .filter_map(|range| {
                    let (_, alias_byte_count) = range.watch.range();
                    let range_end = range.canonical_address.checked_add(alias_byte_count)?;
                    (range.canonical_address <= cursor && cursor < range_end)
                        .then_some((range, range_end))
                })
                .max_by_key(|(_, range_end)| *range_end)
                .ok_or_else(|| {
                    format!("canonical texture range {cursor:#x}..{end:#x} has no live CPU alias")
                })?;
            let copy_end = canonical_end.min(end);
            let copy_bytes = usize::try_from(copy_end - cursor)
                .map_err(|_| "canonical texture copy segment is too large")?;
            let destination_offset = usize::try_from(cursor - address)
                .map_err(|_| "canonical texture copy offset is too large")?;
            let (alias_address, _) = range.watch.range();
            let source_address = alias_address
                .checked_add(cursor - range.canonical_address)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or("canonical texture alias address does not fit usize")?;
            // SAFETY: the caller pins every write-watch mapping against replacement, and the
            // selected alias range covers this exact source segment.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source_address as *const u8,
                    destination.as_mut_ptr().add(destination_offset),
                    copy_bytes,
                );
            }
            cursor = copy_end;
        }
        Ok(())
    }

    pub(crate) fn overlaps_alias_range(&self, address: u64, byte_count: u64) -> bool {
        Self::ranges_overlap_alias(&self.ranges, address, byte_count, |range| {
            range.watch.range()
        })
    }

    pub(crate) fn alias_ranges(&self) -> Vec<(u64, u64)> {
        self.ranges
            .iter()
            .map(|range| range.watch.range())
            .collect()
    }

    pub(crate) fn inherit_content_state_from(&mut self, previous: &Self) -> Result<(), String> {
        if self.range() != previous.range() {
            return Err(format!(
                "cannot inherit texture content state across ranges {:#x}+{:#x} and {:#x}+{:#x}",
                previous.address, previous.byte_count, self.address, self.byte_count
            ));
        }
        let was_dirty = previous.is_dirty();
        let mut dirty_ranges = previous.dirty_page_ranges();
        if was_dirty && dirty_ranges.is_empty() {
            // Lifetime-only tracking cannot provide page detail. Preserve correctness by making
            // the whole resource a pending validation range instead of losing the old dirty bit.
            dirty_ranges.push(previous.range());
        }
        let content_state = Arc::clone(&previous.content_state);
        if !dirty_ranges.is_empty() {
            content_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .pending_dirty_ranges
                .extend(dirty_ranges);
        }
        self.content_state = content_state;
        Ok(())
    }

    pub(crate) fn content_has_changed(&self) -> Result<bool, String> {
        let mut dirty_ranges = self
            .content_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending_dirty_ranges
            .clone();
        dirty_ranges.extend(self.dirty_page_ranges());
        if self.is_dirty() && dirty_ranges.is_empty() {
            dirty_ranges.push(self.range());
        }
        let dirty_ranges = normalize_content_ranges(self, dirty_ranges);
        if dirty_ranges.is_empty() {
            return Ok(false);
        }
        if !self.precise_tracking() {
            return Ok(true);
        }

        let started = Instant::now();
        let chunks = snapshot_dirty_content_fingerprints(self, dirty_ranges)?;
        let validated_bytes = chunks
            .iter()
            .fold(0_u64, |total, chunk| total.saturating_add(chunk.byte_count));
        let mut state = self
            .content_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let changed = chunks.iter().any(|chunk| {
            !state
                .initialized_chunks
                .get(chunk.index)
                .copied()
                .unwrap_or(false)
                || state
                    .chunk_fingerprints
                    .get(chunk.index)
                    .copied()
                    .unwrap_or_default()
                    != chunk.fingerprint
        });
        state.pending_dirty_ranges.clear();
        drop(state);

        TEXTURE_DIRTY_VALIDATION_COUNT.fetch_add(1, Ordering::Relaxed);
        TEXTURE_DIRTY_VALIDATION_BYTES.fetch_add(validated_bytes, Ordering::Relaxed);
        TEXTURE_DIRTY_VALIDATION_NANOSECONDS.fetch_add(
            u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        if !changed {
            TEXTURE_DIRTY_VALIDATION_UNCHANGED.fetch_add(1, Ordering::Relaxed);
        }
        Ok(changed)
    }

    pub(crate) fn content_validation_pending(&self) -> bool {
        !self
            .content_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending_dirty_ranges
            .is_empty()
    }

    pub(crate) fn record_content_snapshot(&self, address: u64, bytes: &[u8]) {
        let Ok(byte_count) = u64::try_from(bytes.len()) else {
            return;
        };
        let Some(snapshot_end) = address.checked_add(byte_count) else {
            return;
        };
        let watch_end = self.address.saturating_add(self.byte_count);
        if address < self.address || snapshot_end > watch_end {
            return;
        }
        let mut state = self
            .content_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for index in 0..state.chunk_fingerprints.len() {
            let Some((chunk_address, chunk_bytes)) = content_chunk_range(self, index) else {
                continue;
            };
            let chunk_end = chunk_address.saturating_add(chunk_bytes);
            if address > chunk_address || chunk_end > snapshot_end {
                continue;
            }
            let Ok(start) = usize::try_from(chunk_address - address) else {
                continue;
            };
            let Ok(length) = usize::try_from(chunk_bytes) else {
                continue;
            };
            let Some(chunk) = bytes.get(start..start.saturating_add(length)) else {
                continue;
            };
            state.chunk_fingerprints[index] = content_fingerprint(chunk);
            state.initialized_chunks[index] = true;
        }
        state.pending_dirty_ranges.clear();
    }

    /// Fingerprint the live guest mapping in place without allocating a full host snapshot.
    ///
    /// Discarded render targets need an initialized CPU-content baseline so a later coherent
    /// alias renewal does not look like a CPU rewrite. Copying multi-megabyte Unreal surfaces
    /// into a throwaway `Vec` dominated texture snapshot time on title screens.
    pub(crate) fn record_live_content_baseline(&self) -> Result<(), String> {
        let chunks = snapshot_dirty_content_fingerprints(self, vec![self.range()])?;
        let mut state = self
            .content_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for chunk in chunks {
            if let Some(slot) = state.chunk_fingerprints.get_mut(chunk.index) {
                *slot = chunk.fingerprint;
            }
            if let Some(initialized) = state.initialized_chunks.get_mut(chunk.index) {
                *initialized = true;
            }
        }
        state.pending_dirty_ranges.clear();
        Ok(())
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DirtyContentFingerprint {
    index: usize,
    byte_count: u64,
    fingerprint: u64,
}

fn normalize_content_ranges(
    write_watch: &GuestTextureWriteWatch,
    ranges: Vec<(u64, u64)>,
) -> Vec<(u64, u64)> {
    let watch_end = write_watch.address.saturating_add(write_watch.byte_count);
    let mut ranges = ranges
        .into_iter()
        .filter_map(|(address, byte_count)| {
            let end = address.checked_add(byte_count)?;
            let clipped_start = address.max(write_watch.address);
            let clipped_end = end.min(watch_end);
            (clipped_start < clipped_end).then_some((clipped_start, clipped_end - clipped_start))
        })
        .collect::<Vec<_>>();
    ranges.sort_unstable_by_key(|&(address, _)| address);
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (address, byte_count) in ranges {
        if let Some((last_address, last_bytes)) = merged.last_mut() {
            let last_end = last_address.saturating_add(*last_bytes);
            if address <= last_end {
                *last_bytes = last_end
                    .max(address.saturating_add(byte_count))
                    .saturating_sub(*last_address);
                continue;
            }
        }
        merged.push((address, byte_count));
    }
    merged
}

fn content_chunk_range(write_watch: &GuestTextureWriteWatch, index: usize) -> Option<(u64, u64)> {
    let offset = u64::try_from(index)
        .ok()?
        .checked_mul(CONTENT_FINGERPRINT_CHUNK_BYTES)?;
    if offset >= write_watch.byte_count {
        return None;
    }
    Some((
        write_watch.address.checked_add(offset)?,
        CONTENT_FINGERPRINT_CHUNK_BYTES.min(write_watch.byte_count - offset),
    ))
}

fn dirty_content_chunk_indices(
    write_watch: &GuestTextureWriteWatch,
    ranges: &[(u64, u64)],
) -> Vec<usize> {
    let mut indices = Vec::new();
    for &(address, byte_count) in ranges {
        let start = (address - write_watch.address) / CONTENT_FINGERPRINT_CHUNK_BYTES;
        let end = (address - write_watch.address)
            .saturating_add(byte_count)
            .div_ceil(CONTENT_FINGERPRINT_CHUNK_BYTES);
        for index in start..end {
            if let Ok(index) = usize::try_from(index) {
                indices.push(index);
            }
        }
    }
    indices.sort_unstable();
    indices.dedup();
    indices
}

fn contiguous_dirty_chunk_runs(indices: &[usize]) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &index in indices {
        if let Some((start, count)) = runs.last_mut()
            && start
                .checked_add(*count)
                .is_some_and(|run_end| run_end == index)
        {
            *count += 1;
        } else {
            runs.push((index, 1));
        }
    }
    runs
}

fn fingerprint_dirty_chunk_run(
    write_watch: &GuestTextureWriteWatch,
    start_index: usize,
    chunk_count: usize,
) -> Result<Vec<DirtyContentFingerprint>, String> {
    let (address, _) = content_chunk_range(write_watch, start_index)
        .ok_or("dirty texture fingerprint run starts out of range")?;
    let last_index = start_index
        .checked_add(chunk_count)
        .and_then(|end| end.checked_sub(1))
        .ok_or("dirty texture fingerprint run is empty or overflows")?;
    let (last_address, last_byte_count) = content_chunk_range(write_watch, last_index)
        .ok_or("dirty texture fingerprint run ends out of range")?;
    let byte_count = last_address
        .checked_add(last_byte_count)
        .and_then(|end| end.checked_sub(address))
        .ok_or("dirty texture fingerprint run byte count overflows")?;
    let length =
        usize::try_from(byte_count).map_err(|_| "dirty texture fingerprint run is too large")?;

    let fingerprint = |bytes: &[u8]| {
        let mut chunks = Vec::with_capacity(chunk_count);
        for index in start_index..start_index + chunk_count {
            let (chunk_address, chunk_byte_count) = content_chunk_range(write_watch, index)
                .ok_or("dirty texture fingerprint chunk is out of range")?;
            let offset = usize::try_from(chunk_address - address)
                .map_err(|_| "dirty texture fingerprint chunk offset is too large")?;
            let chunk_length = usize::try_from(chunk_byte_count)
                .map_err(|_| "dirty texture fingerprint chunk is too large")?;
            let chunk = bytes
                .get(offset..offset.saturating_add(chunk_length))
                .ok_or("dirty texture fingerprint chunk exceeds its stable run")?;
            chunks.push(DirtyContentFingerprint {
                index,
                byte_count: chunk_byte_count,
                fingerprint: content_fingerprint(chunk),
            });
        }
        Ok(chunks)
    };

    if let Some(alias_address) = write_watch.alias_address_for_canonical_range(address, byte_count)
    {
        let source_address = usize::try_from(alias_address)
            .map_err(|_| "dirty texture fingerprint address does not fit usize")?;
        // SAFETY: the caller holds `with_live_backing`, and this exact canonical range has one
        // covering live alias. A racing CPU store faults after rearm and is rejected by the
        // generation/dirty check before these fingerprints can become authoritative.
        let bytes = unsafe { std::slice::from_raw_parts(source_address as *const u8, length) };
        fingerprint(bytes)
    } else if chunk_count > 1 {
        // A canonical run can cross a seam between unrelated CPU virtual aliases. Split only at
        // that seam; each half normally becomes a zero-copy direct hash, while preserving physical
        // chunk order in the returned fingerprints.
        let left_count = chunk_count / 2;
        let mut chunks = fingerprint_dirty_chunk_run(write_watch, start_index, left_count)?;
        chunks.extend(fingerprint_dirty_chunk_run(
            write_watch,
            start_index + left_count,
            chunk_count - left_count,
        )?);
        Ok(chunks)
    } else {
        // A single fingerprint chunk can straddle an alias seam when the texture base is not
        // chunk-aligned. Assemble only that 4 KiB neighborhood through the pinned aliases.
        let mut bytes = vec![0; length];
        // SAFETY: snapshot_dirty_content_fingerprints holds with_live_backing around this call.
        unsafe {
            write_watch.copy_from_live_aliases(address, &mut bytes)?;
        }
        fingerprint(&bytes)
    }
}

fn snapshot_dirty_content_fingerprints(
    write_watch: &GuestTextureWriteWatch,
    mut dirty_ranges: Vec<(u64, u64)>,
) -> Result<Vec<DirtyContentFingerprint>, String> {
    loop {
        dirty_ranges.extend(write_watch.dirty_page_ranges());
        dirty_ranges = normalize_content_ranges(write_watch, dirty_ranges);
        let chunk_indices = dirty_content_chunk_indices(write_watch, &dirty_ranges);
        let chunk_runs = contiguous_dirty_chunk_runs(&chunk_indices);
        let _ = write_watch.consume_dirty();
        write_watch.rearm()?;
        let generations = write_watch.generations();
        let chunks = write_watch
            .with_live_backing(|| {
                chunk_runs.iter().try_fold(
                    Vec::with_capacity(chunk_indices.len()),
                    |mut chunks, run| {
                        chunks.extend(fingerprint_dirty_chunk_run(write_watch, run.0, run.1)?);
                        Ok::<_, String>(chunks)
                    },
                )
            })
            .and_then(|result| result)?;
        if write_watch.generations() == generations && !write_watch.is_dirty() {
            return Ok(chunks);
        }
    }
}

fn content_fingerprint(bytes: &[u8]) -> u64 {
    let mut hash = 0x9e37_79b1_85eb_ca87_u64
        ^ u64::try_from(bytes.len())
            .unwrap_or(u64::MAX)
            .rotate_left(17);
    let mut words = bytes.chunks_exact(8);
    for word in &mut words {
        let value = u64::from_le_bytes(word.try_into().expect("eight-byte content word"));
        hash ^= value.wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
        hash = hash.rotate_left(31).wrapping_mul(0x9e37_79b1_85eb_ca87);
    }
    for &byte in words.remainder() {
        hash ^= u64::from(byte).wrapping_mul(0x1656_67b1_9e37_79f9);
        hash = hash.rotate_left(11).wrapping_mul(0x9e37_79b1_85eb_ca87);
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^= hash >> 33;
    hash.wrapping_mul(0xc4ce_b9fe_1a85_ec53) ^ (hash >> 33)
}

pub(crate) fn refresh_resident_texture_write_watch(
    resident: &mut ResidentGuestTexture,
) -> Result<WriteWatchRefresh, String> {
    match resident.write_watch.retirement() {
        None => Ok(WriteWatchRefresh::Active),
        Some(CpuWriteWatchRetirement::Reprotecting) => Ok(WriteWatchRefresh::Reprotecting),
        Some(CpuWriteWatchRetirement::Reprotected) => {
            let was_dirty = resident.write_watch.is_dirty();
            let mut renewed = watch_resident_texture(
                resident.descriptor,
                resident.texture_type,
                resident.layer_count,
                resident.mip_level_count,
            )?;
            renewed.inherit_content_state_from(&resident.write_watch)?;
            resident.write_watch = renewed;
            if resident.gpu_written && !was_dirty {
                // Reprotection replaced the watch objects but not the shared backing. Preserve
                // the ownership boundary in the new generation domain when no CPU store was
                // observed during the transition.
                resident.note_gpu_write();
            }
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
            let mut renewed = watch_sampled_texture(DecodedImageDescriptor {
                guest: key.descriptor,
                // Component selection is lowered into each shader and does not change the host
                // texture allocation or the guest bytes covered by its write watch.
                destination_select: 0,
                base_mip_level: key.base_mip_level,
                mip_level_count: key.mip_level_count,
                texture_type: key.texture_type,
                layer_count: key.layer_count,
            })?;
            renewed.inherit_content_state_from(&cached.write_watch)?;
            let write_watch = Arc::new(renewed);
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
    // Size is constrained by the decoded image dimensions and checked arithmetic above. Do not
    // impose an unrelated host staging limit here: RDNA2 explicitly permits pitch to exceed the
    // visible width, and large HDR/storage surfaces can legitimately exceed 128 MiB. Production
    // callers validate complete live shared-backing coverage before dereferencing or allocating
    // for this range (through GuestTextureWriteWatch or read_guest_shared_backing).
    if descriptor.address < 0x10_000 {
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
    let mut stable = true;
    let result = if !write_watch.precise_tracking() {
        snapshot()
    } else {
        let mut attempts = 0_u32;
        loop {
            let _ = write_watch.consume_dirty();
            write_watch.rearm()?;
            let generations = write_watch.generations();
            let raw = snapshot()?;
            if write_watch.generations() == generations && !write_watch.is_dirty() {
                break Ok(raw);
            }
            attempts += 1;
            if attempts >= MAX_STABLE_SNAPSHOT_ATTEMPTS {
                stable = false;
                TEXTURE_SNAPSHOT_UNSTABLE_COUNT.fetch_add(1, Ordering::Relaxed);
                break Ok(raw);
            }
        }
    };
    if let Ok(raw) = &result {
        if stable {
            write_watch.record_content_snapshot(address, raw);
        }
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
        write_watch.record_content_snapshot(address, raw);
        Ok(())
    } else {
        Err(format!(
            "guest range {address:#x}+{byte_count:#x} lost shared backing during writeback"
        ))
    }
}

#[cfg(test)]
mod content_validation_tests {
    use super::contiguous_dirty_chunk_runs;

    #[test]
    fn coalesces_only_adjacent_dirty_fingerprint_chunks() {
        assert_eq!(contiguous_dirty_chunk_runs(&[]), Vec::new());
        assert_eq!(
            contiguous_dirty_chunk_runs(&[1, 2, 3, 7, 9, 10]),
            vec![(1, 3), (7, 1), (9, 2)]
        );
    }
}
