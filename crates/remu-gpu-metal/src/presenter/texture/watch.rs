// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CachedSampledTexture, DecodedImageDescriptor, GuestTextureContentState, GuestTextureDescriptor,
    GuestTextureWatchRange, GuestTextureWriteWatch, MetalPresenter,
    RESIDENT_TEXTURE_ACQUIRE_INVALIDATIONS, RangeSet, ResidentGuestTexture, ResourceSpan,
    SAMPLED_TEXTURE_CACHE_ACQUIRE_INVALIDATIONS, SAMPLED_TEXTURE_CACHE_CAPACITY, SampledTextureKey,
    TEXTURE_SNAPSHOT_BYTES, TEXTURE_SNAPSHOT_COUNT, TEXTURE_SNAPSHOT_NANOSECONDS,
    TEXTURE_SNAPSHOT_UNSTABLE_COUNT, WriteWatchRefresh, gpu_written_resident_serving_sampled_key,
    guest_texture_byte_count, mode64k_2d_mip_chain_layout, record_texture_dirty_work_detail,
    record_texture_snapshot_work_detail, record_texture_work, retire_sampled_texture,
    texture_alias_trace_match, trace_alias_correlated_acquire, trace_resident_texture_removal,
};
use remu_gpu::GpuMemoryAcquire;
use remu_memory::{
    CpuWriteWatch, CpuWriteWatchRetirement, guest_shared_backing_alias_segments,
    guest_shared_mapping_epoch, guest_shared_mapping_identity, read_guest_shared_backing,
};
#[cfg(test)]
use remu_memory::{
    GuestRangeLease, GuestWriteSite, LeasePublishOutcome, guest_shared_backing_identity,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// Outcome of a leased texture publication into guest memory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg(test)]
pub(crate) enum TextureWritebackOutcome {
    /// At least one authored run landed (live VA or producer identity).
    Applied,
    /// The lease was revoked (mapping gone / identity refused); caller must demote loudly.
    DiscardedStale,
}

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

/// Read-only write-watch state used to prove that an ACQUIRE_MEM cache action is empty.
///
/// Every field is observed twice before an acquire can be dropped. The token deliberately contains
/// both coarse and page-range dirty state: a protection fault or mapping transition which straddles
/// one read must change the second complete token and conservatively reject the proof.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AcquireNoOpWatchToken {
    precise_tracking: bool,
    retirement: Option<CpuWriteWatchRetirement>,
    dirty: bool,
    dirty_page_ranges: Vec<(u64, u64)>,
    content_validation_pending: bool,
    generations: Vec<u64>,
}

impl AcquireNoOpWatchToken {
    pub(crate) fn is_clean(&self) -> bool {
        self.precise_tracking
            && self.retirement.is_none()
            && !self.dirty
            && self.dirty_page_ranges.is_empty()
            && !self.content_validation_pending
    }
}

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
                    resident.gpu_written(),
                    resident.write_watch.is_dirty(),
                    resident.write_watch.precise_tracking(),
                    resident.write_watch.dirty_page_ranges(),
                    resident.write_watch.generations(),
                )
            });
            let trace_acquire = |resident: &ResidentGuestTexture,
                                 refresh: WriteWatchRefresh,
                                 after_owner: &str,
                                 content_changed: Option<bool>,
                                 outcome: &str| {
                let (
                    Some(trace),
                    Some((before_gpu_owned, dirty, precise, dirty_pages, watch_generations_before)),
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
                    &[],
                    &[],
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
                        if resident.gpu_written() { "gpu" } else { "cpu" },
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
            let gpu_written = resident.gpu_written();
            let content_changed = resident.write_watch.content_has_changed()?;
            if !content_changed {
                if resident.gpu_written() {
                    // A permission-only renewal resets watch generations. Once sparse content
                    // validation proves that guest bytes did not change, move the GPU ownership
                    // boundary into the new generation domain instead of retiring it later.
                }
                trace_acquire(
                    resident,
                    refresh,
                    if resident.gpu_written() { "gpu" } else { "cpu" },
                    Some(false),
                    if resident.gpu_written() {
                        "retained-gpu"
                    } else {
                        "retained-cpu"
                    },
                );
                continue;
            }
            // Phase 1H consumer-aware demotion: ACQUIRE_MEM proves the Metal copy is stale, but
            // demoting here walked residents that no later bind rebuilt (REPORT fingerprint waste
            // loop). Mark invalidate and demote at the first consumer reclaim/bind instead.
            // Sampled-cache entries below still evict — a sample is itself a consumer.
            invalidated_multisample_descriptors.push(descriptor);
            trace_acquire(
                resident,
                refresh,
                if gpu_written { "gpu" } else { "cpu" },
                Some(true),
                "acquire-invalidated-deferred",
            );
            resident.acquire_invalidated = true;
            if gpu_written {
                // CPU overwrote GPU-authored bytes: drop publication debt now; demotion still waits
                // for a consumer bind so unused targets pay nothing.
                resident.gpu_authored = RangeSet::default();
            }
            RESIDENT_TEXTURE_ACQUIRE_INVALIDATIONS.fetch_add(1, Ordering::Relaxed);
        }
        // Resolve each invalidated descriptor once instead of once per multisample entry.
        let invalidated_multisample_spans = invalidated_multisample_descriptors
            .iter()
            .filter_map(|descriptor| ResourceSpan::for_texture(*descriptor))
            .collect::<Vec<_>>();
        self.multisample_textures.retain(|key, _| {
            !ResourceSpan::for_texture(key.descriptor).is_some_and(|candidate| {
                invalidated_multisample_spans
                    .iter()
                    .any(|invalidated| candidate.overlaps(invalidated))
            })
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
            // Permission-only acquire renewals can mark the sampled watch dirty even when a
            // GPU-authored resident still owns the pixels (Octopath 0x80f27e0000 snapshots).
            // Prefer the resident's sparse content validation over the sampled watch dirty bit:
            // pitch-mismatched SRDs and component-reordered views share the same GPU image.
            let sampled_span = self
                .sampled_texture_cache
                .get(&key)
                .and_then(|cached| cached.overlap);
            if let Some(resident_descriptor) = gpu_written_resident_serving_sampled_key(
                &self.resident_guest_textures,
                sampled_span,
            ) && let Some(resident) = self.resident_guest_textures.get_mut(&resident_descriptor)
                && !resident.write_watch.content_has_changed()?
            {
                continue;
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
    pub(crate) fn acquire_noop_token(&self) -> AcquireNoOpWatchToken {
        AcquireNoOpWatchToken {
            precise_tracking: self.precise_tracking(),
            retirement: self.retirement(),
            dirty: self.is_dirty(),
            dirty_page_ranges: self.dirty_page_ranges(),
            content_validation_pending: self.content_validation_pending(),
            generations: self.generations(),
        }
    }

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
        Ok(Self {
            address,
            byte_count,
            ranges,
            content_state: Arc::new(std::sync::Mutex::new(GuestTextureContentState {
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

    #[cfg(test)]
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

    /// The CPU watches backing this texture watch, for self-exclusion in write notifications.
    #[cfg(test)]
    pub(crate) fn cpu_watches(&self) -> Vec<&remu_memory::CpuWriteWatch> {
        self.ranges.iter().map(|range| &range.watch).collect()
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

    /// Coarse write-watch invalidation: any observed guest store retires derived Metal objects.
    ///
    /// Phase 1G deletes content fingerprints / dirty chunk compares. Self-readback generations
    /// are gone with CPU writeback, so a fault means a real guest write.
    pub(crate) fn content_has_changed(&self) -> Result<bool, String> {
        let mut state = self
            .content_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let pending = !state.pending_dirty_ranges.is_empty();
        state.pending_dirty_ranges.clear();
        drop(state);
        Ok(pending || self.is_dirty() || !self.dirty_page_ranges().is_empty())
    }

    pub(crate) fn content_validation_pending(&self) -> bool {
        !self
            .content_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending_dirty_ranges
            .is_empty()
    }

    /// Phase 1G: fingerprints deleted. Clears pending dirty after a GPU publish settled.
    pub(crate) fn record_content_snapshot(&self, _address: u64, _bytes: &[u8]) {
        self.content_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending_dirty_ranges
            .clear();
    }

    /// Phase 1G: no content fingerprint baseline — coarse dirty invalidation only.
    #[cfg(test)]
    pub(crate) fn record_live_content_baseline(&self) -> Result<(), String> {
        self.content_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending_dirty_ranges
            .clear();
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
                resident.base_array_slice,
                resident.layer_count,
                resident.mip_level_count,
            )?;
            renewed.inherit_content_state_from(&resident.write_watch)?;
            resident.write_watch = renewed;
            if resident.gpu_written() && !was_dirty {
                // Reprotection replaced the watch objects but not the shared backing. Preserve
                // the ownership boundary in the new generation domain when no CPU store was
                // observed during the transition.
            }
            Ok(WriteWatchRefresh::Renewed { was_dirty })
        }
        Some(CpuWriteWatchRetirement::Replaced) => Ok(WriteWatchRefresh::Replaced),
    }
}

pub(crate) fn watch_resident_texture(
    descriptor: GuestTextureDescriptor,
    texture_type: u32,
    base_array_slice: u32,
    layer_count: u32,
    mip_level_count: u32,
) -> Result<GuestTextureWriteWatch, String> {
    let span = ResourceSpan::for_image_view(
        descriptor,
        0,
        mip_level_count,
        texture_type,
        base_array_slice,
        layer_count,
    )?
    .ok_or("resident texture selected span overflows")?;
    GuestTextureWriteWatch::new(span.address(), span.len())
}

/// Returns the complete guest allocation extent used by a resident texture.
///
/// Residency watches and shared-event draw preflight deliberately share this calculation. A
/// second attachment-footprint formula could silently omit mip tails, array faces, or tiled
/// padding and admit a draw against an in-flight producer write.
pub(crate) fn resident_texture_raw_byte_count(
    descriptor: GuestTextureDescriptor,
    texture_type: u32,
    layer_count: u32,
    mip_level_count: u32,
) -> Result<u64, String> {
    if texture_type == 9 && layer_count == 1 {
        if mip_level_count == 1 {
            return guest_texture_byte_count(descriptor)
                .ok_or_else(|| "resident texture byte count overflows".to_owned());
        }
        if !matches!(descriptor.tile_mode, 9 | 27)
            || super::super::metal_uncompressed_bytes_per_pixel(descriptor.pixel_format)
                != Some(descriptor.bytes_per_pixel)
        {
            return Err(format!(
                "mipmapped resident byte count requires uncompressed tile 9/27, got tile={} format={:?} bpe={}",
                descriptor.tile_mode, descriptor.pixel_format, descriptor.bytes_per_pixel
            ));
        }
        let layout = mode64k_2d_mip_chain_layout(descriptor, mip_level_count)?;
        return u64::try_from(layout.byte_count)
            .map_err(|_| "mip chain byte count does not fit u64".to_owned());
    }
    if mip_level_count != 1 {
        return Err(format!(
            "mipmapped resident texture type={texture_type} layers={layer_count} is unsupported"
        ));
    }
    let descriptor = DecodedImageDescriptor {
        guest: descriptor,
        span: ResourceSpan::for_texture(descriptor),
        destination_select: 0,
        base_mip_level: 0,
        mip_level_count: 1,
        texture_type,
        base_array_slice: 0,
        layer_count,
    };
    let (_, byte_count) = sampled_texture_raw_range(descriptor)?;
    u64::try_from(byte_count).map_err(|_| "resident texture byte count does not fit u64".to_owned())
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
                span: ResourceSpan::for_image_view(
                    key.descriptor,
                    key.base_mip_level,
                    key.mip_level_count,
                    key.texture_type,
                    key.base_array_slice,
                    key.layer_count,
                )?,
                // Component selection is lowered into each shader and does not change the host
                // texture allocation or the guest bytes covered by its write watch.
                destination_select: 0,
                base_mip_level: key.base_mip_level,
                mip_level_count: key.mip_level_count,
                texture_type: key.texture_type,
                base_array_slice: key.base_array_slice,
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
        Ok(
            unsafe { std::slice::from_raw_parts(_source_address as *const u8, source_length) }
                .to_vec(),
        )
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
    let span = ResourceSpan::for_image_view(
        descriptor.guest,
        descriptor.base_mip_level,
        descriptor.mip_level_count,
        descriptor.texture_type,
        descriptor.base_array_slice,
        descriptor.layer_count,
    )?
    .ok_or("sampled texture selected span overflows")?;
    if span.address() < 0x10_000 || span.len() > 512 * 1024 * 1024 {
        return Err(format!(
            "invalid sampled texture range address={:#x} bytes={}",
            span.address(),
            span.len()
        ));
    }
    Ok((
        usize::try_from(span.address())
            .map_err(|_| "sampled texture address does not fit usize")?,
        usize::try_from(span.len()).map_err(|_| "sampled texture byte count does not fit usize")?,
    ))
}

pub(crate) fn sampled_texture_uses_exact_mode5_mips(descriptor: DecodedImageDescriptor) -> bool {
    descriptor.guest.tile_mode == 5
        && (descriptor.base_mip_level != 0 || descriptor.mip_level_count != 1)
}

pub(crate) fn sampled_texture_uses_exact_mode64k_2d_mips(
    descriptor: DecodedImageDescriptor,
) -> bool {
    matches!(descriptor.guest.tile_mode, 9 | 27)
        && (descriptor.base_mip_level != 0 || descriptor.mip_level_count != 1)
}

pub(crate) fn watch_sampled_texture(
    descriptor: DecodedImageDescriptor,
) -> Result<GuestTextureWriteWatch, String> {
    let (address, byte_count) = sampled_texture_raw_range(descriptor)?;
    GuestTextureWriteWatch::new(
        u64::try_from(address).map_err(|_| "sampled address does not fit u64")?,
        u64::try_from(byte_count).map_err(|_| "sampled byte count does not fit u64")?,
    )
}

#[track_caller]
pub(crate) fn snapshot_watched_sampled_texture_raw(
    descriptor: DecodedImageDescriptor,
    write_watch: &GuestTextureWriteWatch,
) -> Result<Vec<u8>, String> {
    let (source_address, source_length) = sampled_texture_raw_range(descriptor)?;
    snapshot_watched_guest_range(
        u64::try_from(source_address).map_err(|_| "sampled address does not fit u64")?,
        source_length,
        write_watch,
    )
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
    let dirty_page_ranges = if precise_tracking {
        write_watch.dirty_page_ranges()
    } else {
        Default::default()
    };
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
#[cfg(test)]
pub(crate) fn snapshot_watched_guest_texture_raw(
    descriptor: GuestTextureDescriptor,
    write_watch: &GuestTextureWriteWatch,
) -> Result<Vec<u8>, String> {
    let (_, source_length) = guest_texture_raw_range(descriptor)?;
    snapshot_watched_guest_range(descriptor.address, source_length, write_watch)
}

/// Publishes a whole-range image assembled from a guest snapshot back into guest memory.
///
/// `authored` names, as byte offsets from `address`, the runs the GPU actually wrote; every other
/// byte of `raw` came from the snapshot and is republished only because `None` asks for it. That
/// republication is how a read-modify-write writeback overwrites bytes this allocation does not
/// answer for — pitch and chain padding, a neighbouring tenant's surface, or, once the guest has
/// freed the render target, whatever its heap put there since.
///
/// `baseline_raw`, when present, is the pre-patch guest snapshot for the same span as `raw`. RMW
/// callers pass it so [`GuestRangeLease::publish_run_if_baseline`] can refuse a run the guest has
/// since written (MASTER-PLAN D3). When absent, each run's baseline is read live immediately
/// before publish (linear row path).
#[cfg(test)]
pub(crate) fn write_watched_guest_range(
    address: u64,
    raw: &[u8],
    authored: Option<&[(u64, u64)]>,
    baseline_raw: Option<&[u8]>,
    write_watch: &GuestTextureWriteWatch,
    expected_generations: Option<&[u64]>,
) -> Result<TextureWritebackOutcome, String> {
    if write_watch.retirement() == Some(CpuWriteWatchRetirement::Replaced) {
        return Err(format!(
            "guest range mapping {address:#x} was retired before writeback"
        ));
    }
    // `raw` was assembled from a guest snapshot taken before a multi-millisecond GPU
    // readback. A guest store in that window would be reverted by publishing; the caller's
    // pre-snapshot fault generations detect it, and aborting loudly beats silent corruption.
    if let Some(expected) = expected_generations
        && write_watch.generations() != expected
    {
        return Err(format!(
            "guest wrote range {address:#x} during writeback; publishing would revert it"
        ));
    }
    let byte_count = u64::try_from(raw.len()).map_err(|_| "guest range length does not fit u64")?;
    let segments = match authored {
        Some(runs) => authored_segments(raw, runs),
        None => vec![(0, raw)],
    };
    if segments.is_empty() {
        return Ok(TextureWritebackOutcome::Applied);
    }
    let captured_baselines;
    let baseline_source = if let Some(baseline) = baseline_raw {
        if baseline.len() != raw.len() {
            return Err(format!(
                "texture writeback baseline length {} != payload {}",
                baseline.len(),
                raw.len()
            ));
        }
        captured_baselines = match authored {
            Some(runs) => authored_segments(baseline, runs),
            None => vec![(0, baseline)],
        };
        BaselineSource::Captured {
            baselines: &captured_baselines,
        }
    } else {
        BaselineSource::LiveGuest
    };
    publish_segments_leased(
        address,
        byte_count,
        &segments,
        baseline_source,
        write_watch,
        GuestWriteSite::TextureWritebackRange,
    )
}

/// How a leased texture publish obtains the capture-time baseline for each run.
#[cfg(test)]
enum BaselineSource<'a> {
    /// Read the current guest bytes immediately before each publish (linear row path).
    LiveGuest,
    /// Use bytes captured before the GPU retile patched `raw` (RMW tiled/mip/layered path).
    Captured { baselines: &'a [(u64, &'a [u8])] },
}

/// Publishes authored texture runs through a short-lived [`GuestRangeLease`].
///
/// A revoked lease becomes [`TextureWritebackOutcome::DiscardedStale`] (D4-texture): the caller
/// demotes the resident to `Stale` rather than silently rebuilding or publishing by a dead identity.
#[cfg(test)]
fn publish_segments_leased(
    address: u64,
    byte_count: u64,
    segments: &[(u64, &[u8])],
    baselines: BaselineSource<'_>,
    write_watch: &GuestTextureWriteWatch,
    site: GuestWriteSite,
) -> Result<TextureWritebackOutcome, String> {
    let lease = GuestRangeLease::acquire(address, byte_count).ok_or_else(|| {
        format!("guest texture range {address:#x}+{byte_count:#x} has no leasable shared backing")
    })?;
    let mut applied_any = false;
    let mut journal: Vec<(u64, Vec<u8>)> = Vec::with_capacity(segments.len());
    for (offset, source) in segments {
        let live_baseline = match baselines {
            BaselineSource::LiveGuest => {
                let mut buffer = vec![0_u8; source.len()];
                let run_address = address.checked_add(*offset).ok_or_else(|| {
                    format!("texture writeback run address overflows offset={offset:#x}")
                })?;
                if !read_guest_shared_backing(run_address, &mut buffer) {
                    return Err(format!(
                        "guest texture range {run_address:#x}+{:#x} lost shared backing while capturing lease baseline",
                        source.len()
                    ));
                }
                Some(buffer)
            }
            BaselineSource::Captured { .. } => None,
        };
        let baseline: &[u8] = match (&baselines, live_baseline.as_deref()) {
            (BaselineSource::LiveGuest, Some(bytes)) => bytes,
            (
                BaselineSource::Captured {
                    baselines: captured,
                },
                None,
            ) => {
                let Some((_, bytes)) = captured
                    .iter()
                    .find(|(run_offset, _)| *run_offset == *offset)
                else {
                    return Err(format!(
                        "texture writeback missing capture baseline for offset={offset:#x}"
                    ));
                };
                if bytes.len() != source.len() {
                    return Err(format!(
                        "texture writeback baseline length {} != source {} at offset={offset:#x}",
                        bytes.len(),
                        source.len()
                    ));
                }
                bytes
            }
            _ => unreachable!("baseline source pairing"),
        };
        match lease.publish_run_if_baseline(*offset, baseline, source) {
            LeasePublishOutcome::AppliedAtLiveVa | LeasePublishOutcome::AppliedByIdentity => {
                applied_any = true;
                journal.push((*offset, source.to_vec()));
            }
            LeasePublishOutcome::GuestModified => {
                // Guest bytes win — same contract as global leased writebacks.
            }
            LeasePublishOutcome::Revoked => {
                trace_discarded_stale_texture_writeback(
                    address.saturating_add(*offset),
                    source.len(),
                    lease.identity(),
                    guest_shared_backing_identity(
                        address.saturating_add(*offset),
                        u64::try_from(source.len()).unwrap_or(0),
                    ),
                );
                return Ok(TextureWritebackOutcome::DiscardedStale);
            }
            LeasePublishOutcome::WriteFailed => {
                return Err(format!(
                    "guest texture backing write failed address={:#x} bytes={} identity={:?}",
                    address.saturating_add(*offset),
                    source.len(),
                    lease.identity(),
                ));
            }
        }
    }
    if !applied_any {
        // Every run was GuestModified: nothing published, but the lease was live.
        return Ok(TextureWritebackOutcome::Applied);
    }
    let journal_refs: Vec<(u64, &[u8])> = journal
        .iter()
        .map(|(offset, bytes)| (*offset, bytes.as_slice()))
        .collect();
    crate::presenter::diagnostics::journal_guest_write_segments(site, address, &journal_refs);
    // A backing write bypasses every armed page, so watches on other aliases of these
    // bytes would keep validating as clean against content that just changed. Publish the
    // write to every intersecting watch except the writer's own — a self-writeback must
    // leave the writer quiescent — then re-baseline its fingerprints.
    let _ = remu_memory::notify_guest_memory_written_excluding(
        address,
        byte_count,
        &write_watch.cpu_watches(),
    );
    for (offset, bytes) in &journal {
        write_watch.record_content_snapshot(address.saturating_add(*offset), bytes);
    }
    Ok(TextureWritebackOutcome::Applied)
}

#[cfg(test)]
fn trace_discarded_stale_texture_writeback(
    address: u64,
    byte_count: usize,
    expected: remu_memory::GuestSharedMappingIdentity,
    current: Option<remu_memory::GuestSharedMappingIdentity>,
) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static DISCARDED: AtomicUsize = AtomicUsize::new(0);
    let sequence = DISCARDED.fetch_add(1, Ordering::Relaxed) + 1;
    if sequence <= 16 || sequence.is_power_of_two() {
        eprintln!(
            "[REMu][METAL] discarded stale texture writeback sequence={sequence} lease revoked address={address:#x} bytes={byte_count} expected={expected:?} current={current:?}"
        );
    }
}

/// Clamps `runs` to `raw` and slices it, dropping anything that names no byte of the image.
#[cfg(test)]
pub(crate) fn authored_segments<'a>(raw: &'a [u8], runs: &[(u64, u64)]) -> Vec<(u64, &'a [u8])> {
    runs.iter()
        .filter_map(|&(start, end)| {
            let start = usize::try_from(start).ok()?;
            let end = usize::try_from(end).ok()?.min(raw.len());
            (start < end).then(|| (start as u64, &raw[start..end]))
        })
        .collect()
}
