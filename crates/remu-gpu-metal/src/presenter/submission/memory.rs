// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::WritebackTrigger;
use super::{
    GuestMemoryWriteOperation, MetalPresenter, OrderedEventSource, PendingGlobalWriteback,
    PendingGuestMemoryWrite, RangeSet, ResourceSpan, WriteWatchRefresh, guest_batch_writes_overlap,
    guest_physical_ranges_overlap, guest_ranges_overlap, guest_texture_byte_count,
    merge_guest_batch_result, read_guest_memory_wait_value, rearm_color_metadata_clear,
    rearm_depth_metadata_clear, record_diagnostic_memory_copy,
    refresh_resident_texture_write_watch, retire_sampled_textures_matching,
    trace_resident_texture_removal,
};
use crate::presenter::diagnostics::{journal_guest_write, trace_diagnostic_guest_write};
use remu_gpu::{
    GpuMemoryCopy, GpuMemoryRelease, GpuMemoryReleaseValue, GpuMemoryWait, GpuMemoryWrite,
    GpuWaitId,
};
use remu_memory::{
    CpuWriteWatch, GuestSharedMappingIdentity, GuestWriteSite, guest_publish_release_fence,
    guest_shared_mapping_identity, load_guest_shared_u64, notify_guest_memory_written,
    publish_guest_shared_u32_if_identity, publish_guest_shared_u64_if_identity,
    store_guest_visible_u32, store_guest_visible_u64, write_guest_shared_backing_if_identity,
};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

pub(crate) const SAME_QUEUE_DMA_FILL_REJECT_COUNT: usize = 11;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum SameQueueDmaFillReject {
    NonImmediateSource,
    DestinationSelector,
    SourceSnapshot,
    UnsupportedWidth,
    NonZeroImmediate,
    InvalidRange,
    MissingIdentity,
    MetadataOverlap,
    CacheOverlap,
    IncompleteDependency,
    MissingOpenBatch,
}

impl SameQueueDmaFillReject {
    pub(crate) const fn index(self) -> usize {
        self as usize
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::NonImmediateSource => "non-immediate-source",
            Self::DestinationSelector => "destination-selector",
            Self::SourceSnapshot => "source-snapshot",
            Self::UnsupportedWidth => "unsupported-width",
            Self::NonZeroImmediate => "nonzero-immediate",
            Self::InvalidRange => "invalid-range",
            Self::MissingIdentity => "missing-identity",
            Self::MetadataOverlap => "metadata-overlap",
            Self::CacheOverlap => "cache-overlap",
            Self::IncompleteDependency => "incomplete-dependency",
            Self::MissingOpenBatch => "missing-open-batch",
        }
    }
}

pub(crate) struct PreparedSameQueueDmaFill {
    pending: PendingGuestMemoryWrite,
}

impl MetalPresenter {
    pub(crate) fn publish_terminal_guest_memory_writes(
        &mut self,
        pending: &[PendingGuestMemoryWrite],
        mut result: Result<(), String>,
    ) -> Result<(), String> {
        for write in pending {
            let publication = apply_pending_guest_memory_writes(std::slice::from_ref(write));
            let published = publication.is_ok();
            if published {
                self.publish_deferred_dma_destination_authority(std::slice::from_ref(write));
                self.record_produced_guest_memory_writes(std::slice::from_ref(write));
                let (address, byte_count) = write.range();
                self.receiver
                    .retry_guest_memory_waits_overlapping(address, byte_count);
            }
            result =
                merge_guest_batch_result(result, "ordered guest memory publication", publication);
            if !published {
                break;
            }
        }
        result
    }

    pub(crate) fn execute_guest_memory_release(
        &mut self,
        source: OrderedEventSource,
        release: GpuMemoryRelease,
    ) -> Result<(), String> {
        if release.address == 0 {
            return Err("release destination is null".to_owned());
        }
        validate_guest_memory_release(release)?;
        // SDK queueEndOfPipeAction: non-kNone CB/DB actions flush render-backend caches before the
        // label write (sync.cpp). On Metal, ending the open pass stores attachments.
        if release.flushes_render_backend() {
            if let Some(batch) = self.active_open_guest_batch_mut() {
                batch.close_active_pass_encoders();
            }
            self.statistics.render_backend_flushes =
                self.statistics.render_backend_flushes.saturating_add(1);
        }
        trace_diagnostic_guest_write(
            "release",
            release.address,
            if matches!(release.value, GpuMemoryReleaseValue::Immediate32(_)) {
                4
            } else {
                8
            },
        );
        let pending = pin_guest_memory_write(GuestMemoryWriteOperation::Release(release))?;
        let matching_waits = release_shared_event_waits(release);
        if self.per_queue_guest_batches && !matching_waits.is_empty() {
            let OrderedEventSource::GuestQueue(queue_id) = source else {
                return Err("guest memory release has no logical queue identity".to_owned());
            };
            // A release-only producer is still an exact GPU queue point. Declining the binding
            // merely because preceding work already committed forced every matching consumer back
            // through host polling. Open a queue-local batch for the signal; Metal FIFO order keeps
            // it behind earlier committed work, and host retirement remains FIFO as well.
            self.ensure_active_open_guest_batch()?;
            // The release label is a terminal command-processor write owned by this exact queue
            // point. Attach it before freezing the signal checkpoint so a consumer touching the
            // label cannot be misclassified as disjoint while the store remains unpublished.
            self.queue_pinned_guest_memory_write(pending)?;
            self.encode_guest_release_shared_event(queue_id, &matching_waits)?;
            return Ok(());
        }
        self.queue_pinned_guest_memory_write(pending)
    }

    pub(crate) fn prepare_same_queue_immediate_fill(
        &mut self,
        queue_id: u64,
        dependencies: &[super::super::SharedEventFence],
        copy: &GpuMemoryCopy,
    ) -> Option<PreparedSameQueueDmaFill> {
        self.statistics.shared_event_dma_fill_attempts = self
            .statistics
            .shared_event_dma_fill_attempts
            .saturating_add(1);
        match self.prepare_same_queue_immediate_fill_inner(queue_id, dependencies, copy) {
            Ok(prepared) => {
                self.statistics.shared_event_gpu_ordered_dma_fills = self
                    .statistics
                    .shared_event_gpu_ordered_dma_fills
                    .saturating_add(1);
                static ACCEPTED: AtomicUsize = AtomicUsize::new(0);
                let occurrence = ACCEPTED.fetch_add(1, Ordering::Relaxed) + 1;
                if occurrence <= 16 || occurrence.is_power_of_two() {
                    eprintln!(
                        "[REMu][METAL][SHARED-DMA-FILL] occurrence={occurrence} queue=q{queue_id} destination={:#x} bytes={:#x} value={:#010x} action=attach-terminal",
                        copy.destination_address, copy.byte_count, copy.source_address as u32,
                    );
                }
                Some(prepared)
            }
            Err(reason) => {
                let rejects = &mut self.statistics.shared_event_dma_fill_rejections;
                rejects[reason.index()] = rejects[reason.index()].saturating_add(1);
                static REJECTED: AtomicUsize = AtomicUsize::new(0);
                let occurrence = REJECTED.fetch_add(1, Ordering::Relaxed) + 1;
                if occurrence <= 16 || occurrence.is_power_of_two() {
                    eprintln!(
                        "[REMu][METAL][SHARED-DMA-FILL] occurrence={occurrence} queue=q{queue_id} destination={:#x} bytes={:#x} src-sel={} dst-sel={} action=host-gate reason={}",
                        copy.destination_address,
                        copy.byte_count,
                        copy.source_selection,
                        copy.destination_selection,
                        reason.label(),
                    );
                }
                None
            }
        }
    }

    fn prepare_same_queue_immediate_fill_inner(
        &self,
        queue_id: u64,
        dependencies: &[super::super::SharedEventFence],
        copy: &GpuMemoryCopy,
    ) -> Result<PreparedSameQueueDmaFill, SameQueueDmaFillReject> {
        if dependencies.is_empty() {
            return Err(SameQueueDmaFillReject::IncompleteDependency);
        }
        // Phase 1H / W712: cross-queue fence DMA fills stay on the GPU timeline. Look up the
        // producer's signal checkpoint on its own queue (same as WAIT_REG_MEM) instead of
        // rejecting foreign producers as a 1ms host effect.
        if copy.source_selection != 2 {
            return Err(SameQueueDmaFillReject::NonImmediateSource);
        }
        if copy.destination_selection != 3 {
            return Err(SameQueueDmaFillReject::DestinationSelector);
        }
        if copy.source_snapshot.is_some() {
            return Err(SameQueueDmaFillReject::SourceSnapshot);
        }
        if copy.byte_count != 4 {
            return Err(SameQueueDmaFillReject::UnsupportedWidth);
        }
        if copy.source_address as u32 != 0 {
            return Err(SameQueueDmaFillReject::NonZeroImmediate);
        }
        if copy.destination_address == 0
            || !copy.destination_address.is_multiple_of(4)
            || copy
                .destination_address
                .checked_add(copy.byte_count)
                .is_none()
        {
            return Err(SameQueueDmaFillReject::InvalidRange);
        }

        let Some(consumer) = self.guest_queue_batches.get(&queue_id) else {
            return Err(SameQueueDmaFillReject::MissingOpenBatch);
        };
        if consumer.open_batch.is_none() {
            return Err(SameQueueDmaFillReject::MissingOpenBatch);
        }
        for dependency in dependencies {
            let producer_queue = dependency.producer().queue_id();
            let checkpoint = if producer_queue == queue_id {
                consumer
                    .resource_dependencies
                    .get(&dependency.producer())
                    .filter(|checkpoint| checkpoint.fence == *dependency)
            } else {
                self.guest_queue_batches
                    .get(&producer_queue)
                    .and_then(|producer| {
                        producer
                            .committed_batches
                            .iter()
                            .chain(producer.open_batch.iter())
                            .flat_map(|batch| &batch.signal_checkpoints)
                            .find(|checkpoint| checkpoint.fence == *dependency)
                    })
            };
            let Some(checkpoint) = checkpoint else {
                return Err(SameQueueDmaFillReject::IncompleteDependency);
            };
            if !checkpoint.footprint_complete {
                return Err(SameQueueDmaFillReject::IncompleteDependency);
            }
        }

        let Some(identity) =
            guest_shared_mapping_identity(copy.destination_address, copy.byte_count)
        else {
            return Err(SameQueueDmaFillReject::MissingIdentity);
        };
        let metadata_overlap = self
            .color_metadata_clears
            .keys()
            .chain(self.depth_metadata_clears.keys())
            .any(|&address| {
                guest_physical_ranges_overlap(copy.destination_address, copy.byte_count, address, 4)
            });
        if metadata_overlap {
            return Err(SameQueueDmaFillReject::MetadataOverlap);
        }
        let resident_overlap = self.resident_guest_textures.values().any(|resident| {
            let (address, byte_count) = resident.write_watch.range();
            guest_physical_ranges_overlap(
                copy.destination_address,
                copy.byte_count,
                address,
                byte_count,
            )
        });
        let sampled_overlap = self.sampled_texture_cache.values().any(|cached| {
            let (address, byte_count) = cached.write_watch.range();
            guest_physical_ranges_overlap(
                copy.destination_address,
                copy.byte_count,
                address,
                byte_count,
            )
        });
        let multisample_overlap = self.multisample_textures.keys().any(|key| {
            guest_texture_byte_count(key.descriptor).is_none_or(|byte_count| {
                guest_physical_ranges_overlap(
                    copy.destination_address,
                    copy.byte_count,
                    key.descriptor.address,
                    byte_count,
                )
            })
        });
        let direct_scanout_overlap = self.direct_scanout_textures.values().any(|cached| {
            guest_texture_byte_count(cached.descriptor).is_none_or(|byte_count| {
                guest_physical_ranges_overlap(
                    copy.destination_address,
                    copy.byte_count,
                    cached.descriptor.address,
                    byte_count,
                )
            })
        });
        let scanout_snapshot_overlap = self.scanout_snapshot_textures.values().any(|cached| {
            guest_texture_byte_count(cached.descriptor).is_none_or(|byte_count| {
                guest_physical_ranges_overlap(
                    copy.destination_address,
                    copy.byte_count,
                    cached.descriptor.address,
                    byte_count,
                )
            })
        });
        if resident_overlap
            || sampled_overlap
            || multisample_overlap
            || direct_scanout_overlap
            || scanout_snapshot_overlap
        {
            return Err(SameQueueDmaFillReject::CacheOverlap);
        }

        let pending = pin_guest_memory_write(GuestMemoryWriteOperation::Fill {
            address: copy.destination_address,
            value: copy.source_address as u32,
            dword_count: copy.byte_count / 4,
        })
        .map_err(|_| SameQueueDmaFillReject::MissingIdentity)?;
        if pending.mapping_identity != Some(identity) {
            return Err(SameQueueDmaFillReject::MissingIdentity);
        }
        Ok(PreparedSameQueueDmaFill { pending })
    }

    pub(crate) fn queue_prepared_same_queue_immediate_fill(
        &mut self,
        queue_id: u64,
        copy: &GpuMemoryCopy,
        prepared: PreparedSameQueueDmaFill,
    ) -> Result<(), String> {
        if self.active_guest_queue != Some(queue_id) {
            return Err(format!(
                "same-queue DMA fill lost active queue expected={queue_id} actual={:?}",
                self.active_guest_queue,
            ));
        }
        record_diagnostic_memory_copy(copy);
        // Encode waits on any foreign producer fences already recorded as host/resource deps so
        // the fill cannot become guest-visible ahead of those producers (W712 convoy).
        self.encode_pending_cross_queue_waits(queue_id)?;
        let batch = self
            .guest_queue_batches
            .get_mut(&queue_id)
            .and_then(|state| state.open_batch.as_mut())
            .ok_or_else(|| format!("same-queue DMA fill has no open q{queue_id} batch"))?;
        batch.memory_writes.push(prepared.pending);
        // Unreal fence_label_dma: 4-byte zero fills are RELEASE_MEM equivalents for wait wakeups.
        if copy.source_selection == 2 && copy.byte_count == 4 && copy.source_address as u32 == 0 {
            let release = GpuMemoryRelease::label(
                copy.destination_address,
                GpuMemoryReleaseValue::Immediate32(0),
            );
            let matching_waits = release_shared_event_waits(release);
            if !matching_waits.is_empty() {
                self.encode_guest_release_shared_event(queue_id, &matching_waits)?;
            }
        }
        Ok(())
    }

    fn signal_dma_fence_label(&mut self, address: u64, value: u32) -> Result<(), String> {
        if !self.per_queue_guest_batches {
            return Ok(());
        }
        let Some(queue_id) = self.active_guest_queue else {
            return Ok(());
        };
        let release = GpuMemoryRelease::label(address, GpuMemoryReleaseValue::Immediate32(value));
        let matching_waits = release_shared_event_waits(release);
        if matching_waits.is_empty() {
            return Ok(());
        }
        self.encode_guest_release_shared_event(queue_id, &matching_waits)
    }

    fn signal_dma_fence_label_u64(&mut self, address: u64, value: u64) -> Result<(), String> {
        if !self.per_queue_guest_batches {
            return Ok(());
        }
        let Some(queue_id) = self.active_guest_queue else {
            return Ok(());
        };
        let release = GpuMemoryRelease::label(address, GpuMemoryReleaseValue::Immediate64(value));
        let matching_waits = release_shared_event_waits(release);
        if matching_waits.is_empty() {
            return Ok(());
        }
        self.encode_guest_release_shared_event(queue_id, &matching_waits)
    }

    fn encode_pending_cross_queue_waits(&mut self, consumer_queue_id: u64) -> Result<(), String> {
        let host_deps: Vec<_> = self
            .guest_queue_batches
            .get(&consumer_queue_id)
            .map(|state| {
                state
                    .host_dependencies
                    .iter()
                    .map(|(&producer, &value)| (producer, value))
                    .collect()
            })
            .unwrap_or_default();
        for (producer, value) in host_deps {
            if producer.queue_id() == consumer_queue_id {
                continue;
            }
            let producer_event = self
                .guest_queue_batches
                .get(&producer.queue_id())
                .ok_or_else(|| {
                    format!(
                        "cross-queue DMA fill missing producer queue {}",
                        producer.queue_id()
                    )
                })?
                .shared_event_timeline
                .event_owned();
            let consumer = self
                .guest_queue_batches
                .get_mut(&consumer_queue_id)
                .ok_or_else(|| format!("missing consumer guest queue {consumer_queue_id}"))?;
            let batch = consumer
                .open_batch
                .as_mut()
                .ok_or("cross-queue DMA fill has no open consumer batch")?;
            batch.close_active_pass_encoders();
            batch
                .command_buffer
                .encode_wait_for_event(&producer_event, value);
            batch.encoded_event_waits = batch.encoded_event_waits.saturating_add(1);
        }
        Ok(())
    }

    pub(crate) fn execute_guest_memory_copy(&mut self, copy: GpuMemoryCopy) -> Result<(), String> {
        // GFX10 selector 3 is the regular address path with L2 participation; selector 0 is the
        // uncached address path. Both name guest virtual memory. Source selector 2 repeats the
        // packet's inline low dword across the destination; selector 1 names unsupported GDS.
        if !matches!(copy.source_selection, 0 | 2 | 3)
            || !matches!(copy.destination_selection, 0 | 3)
        {
            return Err(format!(
                "unsupported selectors source={} destination={}",
                copy.source_selection, copy.destination_selection
            ));
        }
        let byte_count = usize::try_from(copy.byte_count)
            .map_err(|_| "DMA byte count does not fit the host address space")?;
        if byte_count == 0 || copy.byte_count > 512 * 1024 * 1024 {
            return Err(format!("invalid DMA byte count {:#x}", copy.byte_count));
        }
        copy.destination_address
            .checked_add(copy.byte_count)
            .ok_or("DMA destination range overflows")?;
        if let Some(snapshot) = copy.source_snapshot.as_ref()
            && snapshot.len() != byte_count
        {
            return Err(format!(
                "DMA source snapshot length mismatch bytes={} snapshot={}",
                byte_count,
                snapshot.len()
            ));
        }
        if copy.source_selection == 2 && copy.source_snapshot.is_some() {
            return Err("immediate DMA fill unexpectedly retained a source snapshot".to_owned());
        }
        record_diagnostic_memory_copy(&copy);
        remu_memory::debug_log!(
            "dma",
            "copy src={:#x} dst={:#x} bytes={:#x} selectors={}/{}",
            copy.source_address,
            copy.destination_address,
            copy.byte_count,
            copy.source_selection,
            copy.destination_selection
        );
        trace_diagnostic_guest_write("dma-copy", copy.destination_address, copy.byte_count);
        if copy.source_selection != 2 {
            trace_diagnostic_guest_write("dma-copy-source", copy.source_address, copy.byte_count);
        }

        if copy.source_selection == 2 {
            if !copy.destination_address.is_multiple_of(4) || !copy.byte_count.is_multiple_of(4) {
                return Err(format!(
                    "immediate DMA fill destination={:#x} bytes={:#x} is not dword aligned",
                    copy.destination_address, copy.byte_count
                ));
            }
            self.queue_pending_guest_memory_write(GuestMemoryWriteOperation::Fill {
                address: copy.destination_address,
                value: copy.source_address as u32,
                dword_count: copy.byte_count / 4,
            })?;
            // Fence-shaped immediate fills are RELEASE_MEM equivalents for wait wakeups (W712).
            if copy.byte_count == 4 && copy.source_address as u32 == 0 {
                self.signal_dma_fence_label(copy.destination_address, 0)?;
            }
            // The fill itself is a terminal command-processor publication. Cache and metadata
            // authority must move at that same truthful point, not while the owning Metal batch
            // is still in flight.
            return Ok(());
        } else {
            copy.source_address
                .checked_add(copy.byte_count)
                .ok_or("DMA source range overflows")?;

            // DMA observes preceding GPU writes to its source, but it does not need to drain
            // unrelated rendering. Most Unity DMA_DATA packets move CPU-populated upload pages;
            // synchronously waiting for the complete Metal queue for every one of those packets
            // cuts frame rate in half. Track actual writable guest ranges so only a real
            // read-after-write dependency pays for retirement/readback.
            let mut source_descriptors = self
                .resident_guest_textures
                .values()
                .filter(|resident| resident.gpu_written())
                .filter(|resident| {
                    resident
                        .write_watch
                        .overlaps_alias_range(copy.source_address, copy.byte_count)
                })
                .map(|resident| resident.descriptor)
                .collect::<Vec<_>>();
            source_descriptors.sort_by_key(|descriptor| {
                (
                    self.resident_guest_textures
                        .get(descriptor)
                        .map_or(0, |resident| resident.content_version),
                    descriptor.address,
                )
            });
            let pending_source_write = self.any_guest_batch(|batch| {
                guest_batch_writes_overlap(batch, copy.source_address, copy.byte_count)
            });
            let mut authoritative_gpu_source = pending_source_write;
            if !source_descriptors.is_empty() || pending_source_write {
                self.wait_for_guest_command_batches()?;
            }
            if !source_descriptors.is_empty() {
                for descriptor in source_descriptors {
                    let refresh = refresh_resident_texture_write_watch(
                        self.resident_guest_textures
                            .get_mut(&descriptor)
                            .expect("collected DMA source texture exists"),
                    )?;
                    if refresh == WriteWatchRefresh::Replaced {
                        if let Some(resident) = self
                            .resident_guest_textures
                            .retire_clean(&descriptor, "dma-source-mapping-replaced")?
                        {
                            trace_resident_texture_removal(
                                "dma-source-mapping-replaced",
                                &resident,
                            );
                        }
                        continue;
                    }
                    if refresh == WriteWatchRefresh::Reprotecting {
                        return Err(format!(
                            "DMA source mapping {:#x} is being reprotected",
                            descriptor.address
                        ));
                    }
                    if self
                        .resident_guest_textures
                        .get_mut(&descriptor)
                        .expect("refreshed DMA source texture exists")
                        .cpu_write_obsoletes_gpu_writeback(refresh)?
                    {
                        if let Some(resident) = self
                            .resident_guest_textures
                            .retire_clean(&descriptor, "cpu-reused-dma-source")?
                        {
                            trace_resident_texture_removal("cpu-reused-dma-source", &resident);
                        }
                        continue;
                    }
                    authoritative_gpu_source = true;
                    // The packet names this source range, so the guest is reading these exact
                    // bytes now and the allocation is provably still its own.
                    self.settle_resident_texture_writeback(
                        descriptor,
                        WritebackTrigger::DmaSource,
                    )?;
                }
            }

            // A fence-shaped copy must not become guest-visible at event-dequeue time: Unreal
            // writes GPU fences through DMA_DATA as well as RELEASE_MEM (the runtime traces the
            // pattern as fence_label_dma), and an immediately-applied fence signals frame
            // completion ahead of the frame's in-flight Metal work — the guest then recycles
            // frame memory its own threads are still writing, which is the MallocBinned3
            // corruption. Bulk copies stay immediate: they stage upload pages that later
            // dequeue-time decode legitimately reads back out of guest RAM.
            let fence_shaped = copy.byte_count <= 32
                || (0x7e00_0000_0000..0x8000_0000_0000).contains(&copy.destination_address);
            if fence_shaped {
                // Resolve the source bytes now — every ordered producer that could feed this
                // source was drained above, exactly as the immediate path required — and queue
                // the destination write behind the batch like every other ordered CP effect.
                let mut bytes = Vec::new();
                bytes
                    .try_reserve_exact(byte_count)
                    .map_err(|error| format!("could not stage DMA copy source: {error}"))?;
                if !authoritative_gpu_source && let Some(snapshot) = copy.source_snapshot.as_ref() {
                    bytes.extend_from_slice(snapshot);
                } else {
                    // SAFETY: the runtime decodes DMA only for bounded non-null guest virtual
                    // addresses, and the guest owns the source mapping for the ordered stream.
                    let source = unsafe {
                        std::slice::from_raw_parts(copy.source_address as *const u8, byte_count)
                    };
                    bytes.extend_from_slice(source);
                }
                let fence32 = (bytes.len() == 4)
                    .then(|| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
                let fence64 = (bytes.len() == 8).then(|| {
                    u64::from_le_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                        bytes[7],
                    ])
                });
                self.queue_pending_guest_memory_write(GuestMemoryWriteOperation::Copy {
                    address: copy.destination_address,
                    bytes,
                })?;
                if let Some(value) = fence32 {
                    self.signal_dma_fence_label(copy.destination_address, value)?;
                } else if let Some(value) = fence64 {
                    self.signal_dma_fence_label_u64(copy.destination_address, value)?;
                }
            } else {
                if !authoritative_gpu_source && let Some(snapshot) = copy.source_snapshot.as_ref() {
                    // SAFETY: snapshot validation above proves a complete immutable packet-time
                    // source, and the runtime validated the destination guest mapping.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            snapshot.as_ptr(),
                            copy.destination_address as *mut u8,
                            byte_count,
                        );
                    }
                } else {
                    // SAFETY: the runtime decodes DMA only for bounded non-null guest virtual
                    // addresses. The guest owns these mappings for the ordered event stream. The
                    // live path intentionally provides memmove semantics for overlapping packets.
                    unsafe {
                        std::ptr::copy(
                            copy.source_address as *const u8,
                            copy.destination_address as *mut u8,
                            byte_count,
                        );
                    }
                }
                if remu_memory::guest_write_journal_enabled() {
                    // SAFETY: the copy above just wrote exactly these bytes, so the destination
                    // is readable for the journal's uniformity scan.
                    let written = unsafe {
                        std::slice::from_raw_parts(
                            copy.destination_address as *const u8,
                            byte_count,
                        )
                    };
                    journal_guest_write(
                        GuestWriteSite::DmaCopy,
                        copy.destination_address,
                        copy.byte_count,
                        written,
                    );
                }
                let _ = notify_guest_memory_written(copy.destination_address, copy.byte_count);
                self.record_submitted_gpu_waits_in_range(copy.destination_address, copy.byte_count);
            }
        }

        // The copy made guest RAM authoritative for its destination. Resolving that range
        // physically also reaches residents mapped at an unrelated virtual address, which the
        // virtual-address scan this replaces could not see.
        self.publish_dma_destination_authority(copy.destination_address, copy.byte_count);
        Ok(())
    }

    fn publish_dma_destination_authority(&mut self, address: u64, byte_count: u64) {
        let destination_span = ResourceSpan::new(address, byte_count);
        self.note_guest_memory_authoritative(
            &destination_span,
            &RangeSet::for_span(&destination_span),
            None,
            "dma-destination-write",
        );
        self.multisample_textures.retain(|key, _| {
            guest_texture_byte_count(key.descriptor).is_none_or(|length| {
                !guest_physical_ranges_overlap(key.descriptor.address, length, address, byte_count)
            })
        });
        retire_sampled_textures_matching(&mut self.sampled_texture_cache, |key, _| {
            guest_texture_byte_count(key.descriptor).is_some_and(|length| {
                guest_physical_ranges_overlap(key.descriptor.address, length, address, byte_count)
            })
        });
        // Direct scanout and scanout snapshots are private Metal copies without write watches.
        // Once DMA publishes guest RAM they cannot remain readable aliases of those bytes.
        self.direct_scanout_textures.retain(|_, cached| {
            guest_texture_byte_count(cached.descriptor).is_some_and(|length| {
                !guest_physical_ranges_overlap(
                    address,
                    byte_count,
                    cached.descriptor.address,
                    length,
                )
            })
        });
        self.scanout_snapshot_textures.retain(|_, cached| {
            guest_texture_byte_count(cached.descriptor).is_some_and(|length| {
                !guest_physical_ranges_overlap(
                    address,
                    byte_count,
                    cached.descriptor.address,
                    length,
                )
            })
        });
    }

    fn publish_deferred_dma_destination_authority(&mut self, pending: &[PendingGuestMemoryWrite]) {
        for write in pending
            .iter()
            .filter(|write| write.publishes_guest_memory_authority)
        {
            let (address, byte_count) = write.range();
            let color_metadata = self
                .color_metadata_clears
                .keys()
                .copied()
                .filter(|&metadata| guest_physical_ranges_overlap(address, byte_count, metadata, 4))
                .collect::<Vec<_>>();
            for metadata in color_metadata {
                rearm_color_metadata_clear(&mut self.color_metadata_clears, metadata);
            }
            let depth_metadata = self
                .depth_metadata_clears
                .keys()
                .copied()
                .filter(|&metadata| guest_physical_ranges_overlap(address, byte_count, metadata, 4))
                .collect::<Vec<_>>();
            for metadata in depth_metadata {
                rearm_depth_metadata_clear(&mut self.depth_metadata_clears, metadata);
            }
            self.publish_dma_destination_authority(address, byte_count);
        }
    }

    pub(crate) fn execute_guest_memory_write(
        &mut self,
        write: &GpuMemoryWrite,
    ) -> Result<(), String> {
        if write.address == 0 || write.values.is_empty() {
            return Ok(());
        }
        if !write.address.is_multiple_of(4) {
            return Err(format!(
                "WRITE_DATA destination {:#x} is unaligned",
                write.address
            ));
        }
        validate_guest_memory_write(write)?;
        let byte_count = if write.increment_address {
            u64::try_from(write.values.len())
                .unwrap_or(u64::MAX)
                .saturating_mul(4)
        } else {
            4
        };
        trace_diagnostic_guest_write("write-data", write.address, byte_count);
        self.queue_pending_guest_memory_write(GuestMemoryWriteOperation::Write(write.clone()))
    }

    pub(crate) fn queue_pending_guest_memory_write(
        &mut self,
        write: GuestMemoryWriteOperation,
    ) -> Result<(), String> {
        let write = pin_guest_memory_write(write)?;
        self.queue_pinned_guest_memory_write(write)
    }

    fn queue_pinned_guest_memory_write(
        &mut self,
        write: PendingGuestMemoryWrite,
    ) -> Result<(), String> {
        // RELEASE_MEM/WRITE_DATA label updates are ordered after the Metal work already encoded
        // in this batch, but do not themselves require a new Metal submission. Publish them when
        // the owning command buffer retires. A later wait can inspect the deferred value, while
        // a genuinely blocked wait and the event-loop boundary still commit the batch.
        if let Some(batch) = self.active_open_guest_batch_mut() {
            batch.memory_writes.push(write);
            return Ok(());
        }
        if let Some(batch) = self.active_committed_guest_batches_mut().back_mut() {
            batch.memory_writes.push(write);
        } else {
            apply_pending_guest_memory_writes(std::slice::from_ref(&write))?;
            self.publish_deferred_dma_destination_authority(std::slice::from_ref(&write));
            self.record_produced_guest_memory_writes(std::slice::from_ref(&write));
        }
        Ok(())
    }

    fn encode_guest_release_shared_event(
        &mut self,
        queue_id: u64,
        matching_waits: &[GpuWaitId],
    ) -> Result<(), String> {
        let state = self
            .guest_queue_batches
            .get_mut(&queue_id)
            .ok_or_else(|| format!("missing guest queue {queue_id} for RELEASE_MEM"))?;
        let fence = match state.shared_event_timeline.allocate_fence() {
            Ok(fence) => fence,
            Err(_) => {
                self.statistics.shared_event_fallbacks_timeline_overflow = self
                    .statistics
                    .shared_event_fallbacks_timeline_overflow
                    .saturating_add(1);
                return Ok(());
            }
        };
        {
            let batch = state
                .open_batch
                .as_mut()
                .ok_or("shared-event release has no open producer batch")?;
            batch.close_active_pass_encoders();
            batch
                .command_buffer
                .encode_signal_event(state.shared_event_timeline.event(), fence.value());
            batch.max_encoded_signal = Some(fence);
        }
        let checkpoint =
            super::super::SharedEventSignalCheckpoint::capture_queue_prefix(state, fence);
        if !checkpoint.footprint_complete {
            self.statistics.shared_event_incomplete_signal_footprints = self
                .statistics
                .shared_event_incomplete_signal_footprints
                .saturating_add(1);
        }
        debug_assert!(
            checkpoint.footprint_complete,
            "shared-event RELEASE_MEM has an incomplete producer footprint"
        );
        state
            .open_batch
            .as_mut()
            .expect("shared-event release batch remains open")
            .signal_checkpoints
            .push(checkpoint);
        for &wait_id in matching_waits {
            let _ = self.gpu_wait_fences.bind_if_unbound(wait_id, fence);
        }
        self.receiver
            .retry_registered_guest_memory_waits(matching_waits);
        self.statistics.shared_event_signals =
            self.statistics.shared_event_signals.saturating_add(1);
        Ok(())
    }

    pub(crate) fn record_guest_memory_producer(
        &mut self,
        address: u64,
        is_64_bit: bool,
        value: u64,
    ) {
        self.guest_memory_waits
            .record_produced(address, is_64_bit, value);
        remu_gpu::record_gpu_wait_producer(address, is_64_bit, value);
    }

    pub(crate) fn record_produced_global_writebacks(
        &mut self,
        pending: &[&PendingGlobalWriteback],
    ) {
        for writeback in pending {
            if let Ok(byte_count) = u64::try_from(writeback.byte_count) {
                self.record_submitted_gpu_waits_in_range(writeback.guest_address, byte_count);
            }
        }
    }

    pub(crate) fn record_produced_guest_memory_writes(
        &mut self,
        pending: &[PendingGuestMemoryWrite],
    ) {
        for write in pending {
            match &write.operation {
                GuestMemoryWriteOperation::Release(release) => match release.value {
                    GpuMemoryReleaseValue::Immediate32(value) => {
                        self.record_guest_memory_producer(release.address, false, u64::from(value));
                    }
                    GpuMemoryReleaseValue::Immediate64(value) => {
                        self.record_guest_memory_producer(release.address, true, value);
                        self.record_guest_memory_producer(
                            release.address,
                            false,
                            value & u64::from(u32::MAX),
                        );
                        self.record_guest_memory_producer(
                            release.address.saturating_add(4),
                            false,
                            value >> 32,
                        );
                    }
                    GpuMemoryReleaseValue::Timestamp => {
                        let value = read_guest_memory_wait_value(GpuMemoryWait::memory(
                            release.address,
                            0,
                            u64::MAX,
                            0,
                            true,
                        ));
                        self.record_guest_memory_producer(release.address, true, value);
                        self.record_guest_memory_producer(
                            release.address,
                            false,
                            value & u64::from(u32::MAX),
                        );
                        self.record_guest_memory_producer(
                            release.address.saturating_add(4),
                            false,
                            value >> 32,
                        );
                    }
                },
                GuestMemoryWriteOperation::Write(write) => {
                    for (index, value) in write.values.iter().copied().enumerate() {
                        let offset = if write.increment_address {
                            u64::try_from(index).unwrap_or(u64::MAX).saturating_mul(4)
                        } else {
                            0
                        };
                        self.record_guest_memory_producer(
                            write.address.saturating_add(offset),
                            false,
                            u64::from(value),
                        );
                    }
                    if write.increment_address {
                        for (index, pair) in write.values.windows(2).enumerate() {
                            let address = write.address.saturating_add(
                                u64::try_from(index).unwrap_or(u64::MAX).saturating_mul(4),
                            );
                            if address.is_multiple_of(8) {
                                let value = u64::from(pair[0]) | (u64::from(pair[1]) << 32);
                                self.record_guest_memory_producer(address, true, value);
                            }
                        }
                    }
                }
                GuestMemoryWriteOperation::Fill { address, value, .. } => {
                    self.record_guest_memory_producer(*address, false, u64::from(*value));
                    if address.is_multiple_of(8) {
                        self.record_guest_memory_producer(
                            *address,
                            true,
                            u64::from(*value) | (u64::from(*value) << 32),
                        );
                    }
                }
                GuestMemoryWriteOperation::Copy { address, bytes } => {
                    // Fence labels written through DMA_DATA are dword or qword sized; larger
                    // copies rely on the active-wait latch below, like large fills.
                    if bytes.len() >= 4 {
                        let mut value = [0_u8; 4];
                        value.copy_from_slice(&bytes[..4]);
                        self.record_guest_memory_producer(
                            *address,
                            false,
                            u64::from(u32::from_le_bytes(value)),
                        );
                    }
                    if bytes.len() >= 8 {
                        let mut value = [0_u8; 8];
                        value.copy_from_slice(&bytes[..8]);
                        self.record_guest_memory_producer(
                            *address,
                            true,
                            u64::from_le_bytes(value),
                        );
                    }
                }
            }
        }

        // Large DMA fills are not expanded into the bounded produced-label table. Active waits
        // still latch the exact value they observed at producer completion, including waits on
        // an interior dword or qword of a fill.
        let active = self
            .guest_memory_waits
            .active
            .keys()
            .copied()
            .collect::<Vec<_>>();
        for key in active {
            let width = if key.is_64_bit { 8 } else { 4 };
            if pending
                .iter()
                .any(|write| write.overlaps(key.address, width))
            {
                let value = read_guest_memory_wait_value(key.wait());
                self.record_guest_memory_producer(key.address, key.is_64_bit, value);
            }
        }
        for write in pending {
            let (address, byte_count) = write.range();
            self.record_submitted_gpu_waits_in_range(address, byte_count);
        }
    }

    /// Publishes a Gen5 occlusion / pixel-pipe dump through MTLVisibilityResultBuffer counts.
    ///
    /// Closes the active render pass first so Counting-mode results are flushed into the Shared
    /// visibility buffer, then writes `PIXEL_PIPE_STAT_VALID | count` into each of the sixteen
    /// guest counter pairs via the ordinary MemoryRelease path (preserves wait wakeups / ordering).
    pub(crate) fn publish_occlusion_query(
        &mut self,
        source: OrderedEventSource,
        destination: u64,
    ) -> Result<(), String> {
        if destination == 0 || !destination.is_multiple_of(8) {
            return Err(format!(
                "occlusion query destination {destination:#x} is not a qword-aligned guest address"
            ));
        }
        if let Some(batch) = self.active_open_guest_batch_mut() {
            batch.close_active_pass_encoders();
        }
        let armed = self.visibility_counting_armed;
        let buffer = self.visibility_result_buffer.clone();
        let releases = super::super::attachment::occlusion_counter_releases(
            buffer.as_ref(),
            destination,
            armed,
        );
        self.visibility_counting_armed = false;
        for release in releases {
            self.execute_guest_memory_release(source, release)?;
            self.receiver
                .retry_guest_memory_waits_overlapping(release.address, 8);
        }
        Ok(())
    }
}

impl GuestMemoryWriteOperation {
    pub(crate) fn range(&self) -> (u64, u64) {
        match self {
            Self::Release(release) => (
                release.address,
                match release.value {
                    GpuMemoryReleaseValue::Immediate32(_) => 4,
                    GpuMemoryReleaseValue::Immediate64(_) | GpuMemoryReleaseValue::Timestamp => 8,
                },
            ),
            Self::Write(write) => (
                write.address,
                if write.increment_address {
                    u64::try_from(write.values.len())
                        .unwrap_or(u64::MAX / 4)
                        .saturating_mul(4)
                } else {
                    4
                },
            ),
            Self::Copy { address, bytes } => {
                (*address, u64::try_from(bytes.len()).unwrap_or(u64::MAX))
            }
            Self::Fill {
                address,
                dword_count,
                ..
            } => (*address, dword_count.saturating_mul(4)),
        }
    }

    pub(crate) fn value_at(&self, address: u64, is_64_bit: bool) -> Option<u64> {
        match self {
            Self::Release(release) if release.address == address => match release.value {
                GpuMemoryReleaseValue::Immediate32(value) if !is_64_bit => Some(u64::from(value)),
                GpuMemoryReleaseValue::Immediate64(value) => Some(value),
                GpuMemoryReleaseValue::Timestamp | GpuMemoryReleaseValue::Immediate32(_) => None,
            },
            Self::Write(write) if !is_64_bit => write
                .values
                .iter()
                .copied()
                .enumerate()
                .rev()
                .find_map(|(index, value)| {
                    let offset = if write.increment_address {
                        u64::try_from(index).ok()?.checked_mul(4)?
                    } else {
                        0
                    };
                    (write.address.checked_add(offset)? == address).then_some(u64::from(value))
                }),
            Self::Fill {
                address: fill_address,
                value,
                dword_count,
            } => {
                let offset = address.checked_sub(*fill_address)?;
                if !offset.is_multiple_of(4) {
                    return None;
                }
                let index = offset / 4;
                let required_dwords = if is_64_bit { 2 } else { 1 };
                (index.checked_add(required_dwords)? <= *dword_count).then_some(if is_64_bit {
                    u64::from(*value) | (u64::from(*value) << 32)
                } else {
                    u64::from(*value)
                })
            }
            Self::Copy {
                address: copy_address,
                bytes,
            } => {
                let offset = usize::try_from(address.checked_sub(*copy_address)?).ok()?;
                let width = if is_64_bit { 8 } else { 4 };
                let end = offset.checked_add(width)?;
                if end > bytes.len() {
                    return None;
                }
                let mut value = [0_u8; 8];
                value[..width].copy_from_slice(&bytes[offset..end]);
                Some(u64::from_le_bytes(value))
            }
            Self::Write(_) | Self::Release(_) => None,
        }
    }

    pub(crate) fn overlaps(&self, address: u64, byte_count: u64) -> bool {
        let (write_address, write_byte_count) = self.range();
        guest_ranges_overlap(write_address, write_byte_count, address, byte_count)
    }

    fn journal_site(&self) -> GuestWriteSite {
        match self {
            Self::Release(_) => GuestWriteSite::ReleaseMem,
            Self::Write(_) => GuestWriteSite::WriteData,
            Self::Fill { .. } => GuestWriteSite::DmaFill,
            Self::Copy { .. } => GuestWriteSite::DmaCopy,
        }
    }

    /// Returns the leading published bytes, or an empty slice when the value is only known to the
    /// applying store, so the journal reports the verdict as unknown instead of inventing zeros.
    fn journal_payload(&self) -> ([u8; 8], usize) {
        let mut payload = [0_u8; 8];
        match self {
            Self::Release(GpuMemoryRelease {
                value: GpuMemoryReleaseValue::Immediate32(value),
                ..
            }) => {
                payload[..4].copy_from_slice(&value.to_le_bytes());
                (payload, 4)
            }
            Self::Release(GpuMemoryRelease {
                value: GpuMemoryReleaseValue::Immediate64(value),
                ..
            }) => {
                payload.copy_from_slice(&value.to_le_bytes());
                (payload, 8)
            }
            Self::Release(_) => (payload, 0),
            Self::Write(write) => {
                let mut length = 0;
                for value in write.values.iter().take(2) {
                    payload[length..length + 4].copy_from_slice(&value.to_le_bytes());
                    length += 4;
                }
                (payload, length)
            }
            Self::Fill { value, .. } => {
                payload[..4].copy_from_slice(&value.to_le_bytes());
                payload[4..].copy_from_slice(&value.to_le_bytes());
                (payload, 8)
            }
            Self::Copy { bytes, .. } => {
                let length = bytes.len().min(8);
                payload[..length].copy_from_slice(&bytes[..length]);
                (payload, length)
            }
        }
    }
}

fn release_shared_event_waits(release: GpuMemoryRelease) -> Vec<GpuWaitId> {
    let byte_count = match release.value {
        GpuMemoryReleaseValue::Immediate32(_) => 4,
        GpuMemoryReleaseValue::Immediate64(_) => 8,
        GpuMemoryReleaseValue::Timestamp => return Vec::new(),
    };
    remu_gpu::pending_gpu_waits_overlapping(release.address, byte_count)
        .into_iter()
        .filter_map(|(id, wait)| {
            release_value_for_wait(release, wait)
                .is_some_and(|value| wait.is_satisfied_by(value))
                .then_some(id)
        })
        .collect()
}

pub(crate) fn release_value_for_wait(
    release: GpuMemoryRelease,
    wait: GpuMemoryWait,
) -> Option<u64> {
    match release.value {
        GpuMemoryReleaseValue::Immediate32(value)
            if !wait.is_64_bit && wait.address == release.address =>
        {
            Some(u64::from(value))
        }
        GpuMemoryReleaseValue::Immediate64(value) if wait.address == release.address => {
            Some(if wait.is_64_bit {
                value
            } else {
                value & u64::from(u32::MAX)
            })
        }
        GpuMemoryReleaseValue::Immediate64(value)
            if !wait.is_64_bit && wait.address == release.address.checked_add(4)? =>
        {
            Some(value >> 32)
        }
        GpuMemoryReleaseValue::Immediate32(_)
        | GpuMemoryReleaseValue::Immediate64(_)
        | GpuMemoryReleaseValue::Timestamp => None,
    }
}

impl PendingGuestMemoryWrite {
    pub(crate) fn range(&self) -> (u64, u64) {
        self.operation.range()
    }

    pub(crate) fn value_at(&self, address: u64, is_64_bit: bool) -> Option<u64> {
        self.operation.value_at(address, is_64_bit)
    }

    pub(crate) fn overlaps(&self, address: u64, byte_count: u64) -> bool {
        self.operation.overlaps(address, byte_count)
    }
}

pub(crate) fn validate_guest_memory_release(release: GpuMemoryRelease) -> Result<(), String> {
    let alignment = match release.value {
        GpuMemoryReleaseValue::Immediate32(_) => 4,
        GpuMemoryReleaseValue::Immediate64(_) | GpuMemoryReleaseValue::Timestamp => 8,
    };
    if release.address.is_multiple_of(alignment) {
        Ok(())
    } else {
        Err(format!(
            "{}-bit release destination {:#x} is unaligned",
            alignment * 8,
            release.address
        ))
    }
}

pub(crate) fn validate_guest_memory_write(write: &GpuMemoryWrite) -> Result<(), String> {
    if !write.address.is_multiple_of(4) {
        return Err(format!(
            "WRITE_DATA destination {:#x} is unaligned",
            write.address
        ));
    }
    if write.increment_address && !write.values.is_empty() {
        let final_index = u64::try_from(write.values.len() - 1)
            .map_err(|_| "WRITE_DATA value count does not fit u64")?;
        write
            .address
            .checked_add(
                final_index
                    .checked_mul(4)
                    .ok_or("WRITE_DATA destination offset overflowed")?,
            )
            .ok_or_else(|| "WRITE_DATA destination range overflowed".to_owned())?;
    }
    Ok(())
}

fn validate_guest_memory_operation(write: &GuestMemoryWriteOperation) -> Result<(), String> {
    match write {
        GuestMemoryWriteOperation::Release(release) => validate_guest_memory_release(*release),
        GuestMemoryWriteOperation::Write(write) => validate_guest_memory_write(write),
        GuestMemoryWriteOperation::Fill {
            address,
            dword_count,
            ..
        } => {
            let byte_count = dword_count
                .checked_mul(4)
                .ok_or("DMA fill byte count overflowed")?;
            if !address.is_multiple_of(4) || *address == 0 || byte_count == 0 {
                return Err(format!(
                    "DMA fill destination {address:#x} dwords={dword_count:#x} is invalid"
                ));
            }
            address
                .checked_add(byte_count)
                .ok_or("DMA fill destination range overflowed")?;
            Ok(())
        }
        GuestMemoryWriteOperation::Copy { address, bytes } => {
            if *address == 0 || bytes.is_empty() {
                return Err(format!(
                    "DMA copy destination {address:#x} bytes={} is invalid",
                    bytes.len()
                ));
            }
            address
                .checked_add(u64::try_from(bytes.len()).map_err(|_| "DMA copy length overflow")?)
                .ok_or("DMA copy destination range overflowed")?;
            Ok(())
        }
    }
}

pub(crate) fn pin_guest_memory_write(
    operation: GuestMemoryWriteOperation,
) -> Result<PendingGuestMemoryWrite, String> {
    validate_guest_memory_operation(&operation)?;
    let publishes_guest_memory_authority =
        matches!(operation, GuestMemoryWriteOperation::Fill { .. });
    let (address, byte_count) = operation.range();
    let mapping_identity = guest_shared_mapping_identity(address, byte_count);
    let write_watch = if mapping_identity.is_some() {
        None
    } else {
        // SAFETY: command decoding admitted this exact writable guest range. The watch is kept
        // with the deferred command-processor effect and pins its mapping during publication.
        Some(unsafe { CpuWriteWatch::new(address, byte_count) }.map_err(|error| {
            format!(
                "could not pin deferred guest memory destination address={address:#x} bytes={byte_count}: {error}"
            )
        })?)
    };
    Ok(PendingGuestMemoryWrite {
        operation,
        mapping_identity,
        write_watch,
        publishes_guest_memory_authority,
    })
}

fn gpu_timestamp() -> u64 {
    static GPU_CLOCK_START: OnceLock<Instant> = OnceLock::new();
    u64::try_from(
        GPU_CLOCK_START
            .get_or_init(Instant::now)
            .elapsed()
            .as_nanos(),
    )
    .unwrap_or(u64::MAX)
    .max(1)
}

fn apply_guest_memory_operation_direct(
    operation: &GuestMemoryWriteOperation,
) -> Result<(), String> {
    match operation {
        GuestMemoryWriteOperation::Release(release) => match release.value {
            GpuMemoryReleaseValue::Immediate32(value) => {
                store_guest_visible_u32(release.address, value);
            }
            GpuMemoryReleaseValue::Immediate64(value) => {
                store_guest_visible_u64(release.address, value);
            }
            GpuMemoryReleaseValue::Timestamp => {
                store_guest_visible_u64(release.address, gpu_timestamp());
            }
        },
        GuestMemoryWriteOperation::Write(write) => {
            for (index, value) in write.values.iter().copied().enumerate() {
                let offset = if write.increment_address {
                    u64::try_from(index)
                        .map_err(|_| "WRITE_DATA index does not fit u64")?
                        .checked_mul(4)
                        .ok_or("WRITE_DATA destination offset overflowed")?
                } else {
                    0
                };
                let address = write
                    .address
                    .checked_add(offset)
                    .ok_or("WRITE_DATA destination range overflowed")?;
                store_guest_visible_u32(address, value);
            }
        }
        GuestMemoryWriteOperation::Fill {
            address,
            value,
            dword_count,
        } => {
            // A DMA fill is the one publication path that wrote raw to the guest VA. Every other
            // path proves the destination is a live mapping first, and a fill is the most
            // damaging thing to get wrong: DCC clear keys are hundreds of kilobytes of one
            // repeated dword, so a stale or short destination sprays that pattern across whatever
            // now occupies the range. Prove the whole range resolves to one live shared mapping
            // before writing a single dword, and refuse loudly rather than scribble.
            let byte_count = dword_count
                .checked_mul(4)
                .ok_or("DMA fill byte count overflowed")?;
            if guest_shared_mapping_identity(*address, byte_count).is_none() {
                static REFUSED: AtomicU64 = AtomicU64::new(0);
                let sequence = REFUSED.fetch_add(1, Ordering::Relaxed) + 1;
                if sequence <= 16 || sequence.is_power_of_two() {
                    eprintln!(
                        "[REMu][VM] refused DMA fill sequence={sequence} address={address:#x} bytes={byte_count:#x} value={value:#010x}: no single live mapping covers the range"
                    );
                }
                return Err(format!(
                    "DMA fill destination {address:#x}+{byte_count:#x} is not one live mapping"
                ));
            }
            for index in 0..*dword_count {
                let offset = index
                    .checked_mul(4)
                    .ok_or("DMA fill destination offset overflowed")?;
                store_guest_visible_u32(*address + offset, *value);
            }
        }
        GuestMemoryWriteOperation::Copy { address, bytes } => {
            if let Some(value) = label_word_from_bytes(*address, bytes) {
                match value {
                    LabelWord::U32(value) => store_guest_visible_u32(*address, value),
                    LabelWord::U64(value) => store_guest_visible_u64(*address, value),
                }
            } else {
                // SAFETY: the caller pins the original mapping against replacement for the duration
                // of this operation, and validation bounded the destination range.
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), *address as *mut u8, bytes.len());
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum LabelWord {
    U32(u32),
    U64(u64),
}

fn label_word_from_bytes(address: u64, bytes: &[u8]) -> Option<LabelWord> {
    match bytes.len() {
        4 if address.is_multiple_of(4) => {
            Some(LabelWord::U32(u32::from_le_bytes(bytes.try_into().ok()?)))
        }
        8 if address.is_multiple_of(8) => {
            Some(LabelWord::U64(u64::from_le_bytes(bytes.try_into().ok()?)))
        }
        _ => None,
    }
}

fn is_ordered_label_publication(operation: &GuestMemoryWriteOperation) -> bool {
    match operation {
        GuestMemoryWriteOperation::Release(_) | GuestMemoryWriteOperation::Write(_) => true,
        GuestMemoryWriteOperation::Copy { address, bytes } => {
            label_word_from_bytes(*address, bytes).is_some()
        }
        GuestMemoryWriteOperation::Fill { .. } => false,
    }
}

fn write_shared_identity_bytes(
    address: u64,
    identity: GuestSharedMappingIdentity,
    offset: u64,
    bytes: &[u8],
) -> Result<(), String> {
    let destination = address
        .checked_add(offset)
        .ok_or("deferred guest memory destination overflowed")?;
    let expected = GuestSharedMappingIdentity {
        backing_id: identity.backing_id,
        backing_offset: identity
            .backing_offset
            .checked_add(offset)
            .ok_or("deferred guest memory backing offset overflowed")?,
    };
    if let Some(value) = label_word_from_bytes(destination, bytes) {
        let published = match value {
            LabelWord::U32(value) => {
                publish_guest_shared_u32_if_identity(destination, expected, value)
            }
            LabelWord::U64(value) => {
                publish_guest_shared_u64_if_identity(destination, expected, value)
            }
        };
        if published {
            return Ok(());
        }
        return Err(format!(
            "deferred guest memory mapping changed before completion address={destination:#x} bytes={}",
            bytes.len()
        ));
    }
    if write_guest_shared_backing_if_identity(destination, expected, bytes) {
        Ok(())
    } else {
        Err(format!(
            "deferred guest memory mapping changed before completion address={destination:#x} bytes={}",
            bytes.len()
        ))
    }
}

fn write_shared_identity_u32(
    address: u64,
    identity: GuestSharedMappingIdentity,
    offset: u64,
    value: u32,
) -> Result<(), String> {
    let destination = address
        .checked_add(offset)
        .ok_or("deferred guest memory destination overflowed")?;
    let expected = GuestSharedMappingIdentity {
        backing_id: identity.backing_id,
        backing_offset: identity
            .backing_offset
            .checked_add(offset)
            .ok_or("deferred guest memory backing offset overflowed")?,
    };
    if publish_guest_shared_u32_if_identity(destination, expected, value) {
        Ok(())
    } else {
        Err(format!(
            "deferred guest memory mapping changed before completion address={destination:#x} bytes=4"
        ))
    }
}

fn write_shared_identity_u64(
    address: u64,
    identity: GuestSharedMappingIdentity,
    value: u64,
) -> Result<(), String> {
    if publish_guest_shared_u64_if_identity(address, identity, value) {
        Ok(())
    } else {
        Err(format!(
            "deferred guest memory mapping changed before completion address={address:#x} bytes=8"
        ))
    }
}

fn apply_guest_memory_operation_shared(
    operation: &GuestMemoryWriteOperation,
    identity: GuestSharedMappingIdentity,
) -> Result<(), String> {
    let (address, _) = operation.range();
    match operation {
        GuestMemoryWriteOperation::Release(release) => match release.value {
            GpuMemoryReleaseValue::Immediate32(value) => {
                write_shared_identity_u32(address, identity, 0, value)
            }
            GpuMemoryReleaseValue::Immediate64(value) => {
                write_shared_identity_u64(address, identity, value)
            }
            GpuMemoryReleaseValue::Timestamp => {
                write_shared_identity_u64(address, identity, gpu_timestamp())
            }
        },
        GuestMemoryWriteOperation::Write(write) => {
            for (index, value) in write.values.iter().copied().enumerate() {
                let offset = if write.increment_address {
                    u64::try_from(index)
                        .map_err(|_| "WRITE_DATA index does not fit u64")?
                        .checked_mul(4)
                        .ok_or("WRITE_DATA destination offset overflowed")?
                } else {
                    0
                };
                write_shared_identity_u32(address, identity, offset, value)?;
            }
            Ok(())
        }
        GuestMemoryWriteOperation::Fill {
            value, dword_count, ..
        } => {
            const CHUNK_DWORDS: u64 = 16 * 1024;
            let pattern = value.to_le_bytes();
            let mut chunk = vec![0_u8; usize::try_from(CHUNK_DWORDS * 4).unwrap_or(64 * 1024)];
            for bytes in chunk.chunks_exact_mut(4) {
                bytes.copy_from_slice(&pattern);
            }
            let mut written_dwords = 0;
            while written_dwords < *dword_count {
                let current_dwords = (*dword_count - written_dwords).min(CHUNK_DWORDS);
                let current_bytes = usize::try_from(current_dwords * 4)
                    .map_err(|_| "DMA fill chunk does not fit usize")?;
                write_shared_identity_bytes(
                    address,
                    identity,
                    written_dwords
                        .checked_mul(4)
                        .ok_or("DMA fill destination offset overflowed")?,
                    &chunk[..current_bytes],
                )?;
                written_dwords += current_dwords;
            }
            Ok(())
        }
        GuestMemoryWriteOperation::Copy { bytes, .. } => {
            write_shared_identity_bytes(address, identity, 0, bytes)
        }
    }
}

pub(crate) fn apply_pending_guest_memory_writes(
    pending: &[PendingGuestMemoryWrite],
) -> Result<(), String> {
    let mut published_unordered_payload = false;
    for pending_write in pending {
        validate_guest_memory_operation(&pending_write.operation)?;
        let (pixel_pipe_trace, query_completion_trace) = match pending_write.operation {
            GuestMemoryWriteOperation::Release(release) => (
                remu_gpu::take_pixel_pipe_stat_release_trace(release),
                remu_gpu::take_pixel_pipe_query_completion_release_trace(release),
            ),
            GuestMemoryWriteOperation::Write(_)
            | GuestMemoryWriteOperation::Fill { .. }
            | GuestMemoryWriteOperation::Copy { .. } => (None, None),
        };
        let observed_before = (pixel_pipe_trace.is_some() || query_completion_trace.is_some())
            .then(|| observe_pending_guest_memory_release(pending_write))
            .flatten();
        let label_publication = is_ordered_label_publication(&pending_write.operation);
        if label_publication && published_unordered_payload {
            // Payload bytes become guest-visible only when the following label publishes. The
            // large memcpy path cannot carry a per-store release, so order it here explicitly.
            guest_publish_release_fence();
            published_unordered_payload = false;
        }
        let result = if let Some(identity) = pending_write.mapping_identity {
            apply_guest_memory_operation_shared(&pending_write.operation, identity)
        } else if let Some(write_watch) = pending_write.write_watch.as_ref() {
            write_watch
                .with_live_mapping(|| {
                    apply_guest_memory_operation_direct(&pending_write.operation)
                })
                .map_err(|error| {
                    let (address, byte_count) = pending_write.range();
                    format!(
                        "deferred guest memory mapping changed before completion address={address:#x} bytes={byte_count}: {error}"
                    )
                })
                .and_then(|result| result)
        } else {
            let (address, byte_count) = pending_write.range();
            Err(format!(
                "deferred guest memory write has no mapping lifetime guard address={address:#x} bytes={byte_count}"
            ))
        };
        if result.is_ok() && !label_publication {
            published_unordered_payload = true;
        }
        if result.is_ok() {
            let (address, byte_count) = pending_write.range();
            let (payload, payload_length) = pending_write.operation.journal_payload();
            journal_guest_write(
                pending_write.operation.journal_site(),
                address,
                byte_count,
                &payload[..payload_length],
            );
            if matches!(
                pending_write.operation,
                GuestMemoryWriteOperation::Fill { .. } | GuestMemoryWriteOperation::Copy { .. }
            ) {
                // Backing-identity writes bypass protected guest VAs. Publish the same dirtying
                // that an address-space write would so every persistent alias watch revalidates.
                let _ = notify_guest_memory_written(address, byte_count);
            }
        }
        if let Some(trace) = pixel_pipe_trace {
            let GuestMemoryWriteOperation::Release(release) = pending_write.operation else {
                unreachable!("only a release can carry pixel-pipe trace provenance");
            };
            let expected = match release.value {
                GpuMemoryReleaseValue::Immediate64(value) => value,
                GpuMemoryReleaseValue::Immediate32(_) | GpuMemoryReleaseValue::Timestamp => {
                    unreachable!("pixel-pipe trace provenance names an immediate 64-bit release")
                }
            };
            let observed_after = observe_pending_guest_memory_release(pending_write);
            eprintln!(
                "[REMu][AGC][PIXEL-PIPE][PUBLISH] sequence={} pair={} address={:#x} expected={expected:#018x} observed_before={observed_before:#018x?} observed_after={observed_after:#018x?} success={} error={:?}",
                trace.dump_sequence,
                trace.pair_index,
                release.address,
                result.is_ok(),
                result.as_ref().err(),
            );
        }
        if let Some(trace) = query_completion_trace {
            let GuestMemoryWriteOperation::Release(release) = pending_write.operation else {
                unreachable!("only a release can carry query-completion trace provenance");
            };
            let observed_after = observe_pending_guest_memory_release(pending_write);
            eprintln!(
                "[REMu][AGC][PIXEL-PIPE][COMPLETE][PUBLISH] first_sequence={} second_sequence={} payload={:#x} address={:#x} expected=0x0000000000000001 observed_before={observed_before:#018x?} observed_after={observed_after:#018x?} success={} error={:?}",
                trace.first_dump_sequence,
                trace.second_dump_sequence,
                trace.payload_address,
                release.address,
                result.is_ok(),
                result.as_ref().err(),
            );
        }
        result?;
    }
    Ok(())
}

fn observe_pending_guest_memory_release(pending: &PendingGuestMemoryWrite) -> Option<u64> {
    let GuestMemoryWriteOperation::Release(GpuMemoryRelease {
        address,
        value: GpuMemoryReleaseValue::Immediate64(_),
        ..
    }) = pending.operation
    else {
        return None;
    };
    if pending.mapping_identity.is_some() {
        return load_guest_shared_u64(address);
    }
    let write_watch = pending.write_watch.as_ref()?;
    write_watch
        .with_live_mapping(|| remu_memory::load_guest_visible_u64(address))
        .ok()
}
