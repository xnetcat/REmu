// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    MetalPresenter, PendingGlobalWriteback, PendingGuestMemoryWrite, WriteWatchRefresh,
    guest_batch_writes_overlap, guest_ranges_overlap, guest_texture_byte_count,
    merge_guest_batch_result, read_guest_memory_wait_value, rearm_depth_metadata_clear,
    record_diagnostic_memory_copy, refresh_resident_texture_write_watch,
    retire_multisample_textures_overlapping_writeback,
    retire_resident_textures_overlapping_writeback, retire_sampled_textures_matching,
    retire_sampled_textures_overlapping_writeback, trace_resident_texture_removal,
    write_resident_texture_contents_to_guest,
};
use remu_gpu::{
    GpuMemoryCopy, GpuMemoryRelease, GpuMemoryReleaseValue, GpuMemoryWait, GpuMemoryWrite,
};
use std::sync::OnceLock;
use std::time::Instant;

impl MetalPresenter {
    pub(crate) fn publish_terminal_guest_memory_writes(
        &mut self,
        pending: &[PendingGuestMemoryWrite],
        result: Result<(), String>,
    ) -> Result<(), String> {
        let publication = apply_pending_guest_memory_writes(pending);
        if publication.is_ok() {
            self.record_produced_guest_memory_writes(pending);
        }
        merge_guest_batch_result(result, "ordered guest memory publication", publication)
    }

    pub(crate) fn execute_guest_memory_release(
        &mut self,
        release: GpuMemoryRelease,
    ) -> Result<(), String> {
        if release.address == 0 {
            return Err("release destination is null".to_owned());
        }
        validate_guest_memory_release(release)?;
        self.queue_pending_guest_memory_write(PendingGuestMemoryWrite::Release(release))
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
        record_diagnostic_memory_copy(copy);

        if copy.source_selection == 2 {
            if !copy.destination_address.is_multiple_of(4) || !copy.byte_count.is_multiple_of(4) {
                return Err(format!(
                    "immediate DMA fill destination={:#x} bytes={:#x} is not dword aligned",
                    copy.destination_address, copy.byte_count
                ));
            }
            self.queue_pending_guest_memory_write(PendingGuestMemoryWrite::Fill {
                address: copy.destination_address,
                value: copy.source_address as u32,
                dword_count: copy.byte_count / 4,
            })?;
            rearm_depth_metadata_clear(&mut self.depth_metadata_clears, copy.destination_address);
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
                .filter(|resident| resident.gpu_written)
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
            let pending_source_write = self.open_guest_batch.as_ref().is_some_and(|batch| {
                guest_batch_writes_overlap(batch, copy.source_address, copy.byte_count)
            }) || self.committed_guest_batches.iter().any(|batch| {
                guest_batch_writes_overlap(batch, copy.source_address, copy.byte_count)
            });
            let source_was_gpu_written = !source_descriptors.is_empty();
            if !source_descriptors.is_empty() || pending_source_write {
                self.wait_for_guest_command_batches()?;
            }
            if !source_descriptors.is_empty() {
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
                for descriptor in source_descriptors {
                    let refresh = refresh_resident_texture_write_watch(
                        self.resident_guest_textures
                            .get_mut(&descriptor)
                            .expect("collected DMA source texture exists"),
                    )?;
                    if refresh == WriteWatchRefresh::Replaced {
                        if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
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
                    let (alias_ranges, content_version) = {
                        let resident = self
                            .resident_guest_textures
                            .get(&descriptor)
                            .expect("collected DMA source texture exists");
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
                        .expect("collected DMA source texture exists");
                    write_resident_texture_contents_to_guest(&device, &queue, resident)?;
                    resident.gpu_written = false;
                }
            }

            if !copy.cpu_applied || source_was_gpu_written || pending_source_write {
                // SAFETY: the runtime decodes DMA only for bounded non-null guest virtual
                // addresses. The guest owns these mappings for the lifetime of the ordered GPU
                // event stream. `copy` intentionally provides memmove semantics for legal
                // overlapping packets.
                unsafe {
                    std::ptr::copy(
                        copy.source_address as *const u8,
                        copy.destination_address as *mut u8,
                        byte_count,
                    );
                }
            }
            self.record_submitted_gpu_waits_in_range(copy.destination_address, copy.byte_count);
        }

        let destination_descriptors = self
            .resident_guest_textures
            .keys()
            .copied()
            .filter(|descriptor| {
                guest_texture_byte_count(*descriptor).is_some_and(|length| {
                    guest_ranges_overlap(
                        descriptor.address,
                        length,
                        copy.destination_address,
                        copy.byte_count,
                    )
                })
            })
            .collect::<Vec<_>>();
        for descriptor in destination_descriptors {
            if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
                trace_resident_texture_removal("dma-destination-write", &resident);
            }
        }
        self.multisample_textures.retain(|key, _| {
            guest_texture_byte_count(key.descriptor).is_none_or(|length| {
                !guest_ranges_overlap(
                    key.descriptor.address,
                    length,
                    copy.destination_address,
                    copy.byte_count,
                )
            })
        });
        retire_sampled_textures_matching(&mut self.sampled_texture_cache, |key| {
            guest_texture_byte_count(key.descriptor).is_some_and(|length| {
                guest_ranges_overlap(
                    key.descriptor.address,
                    length,
                    copy.destination_address,
                    copy.byte_count,
                )
            })
        });
        Ok(())
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
        self.queue_pending_guest_memory_write(PendingGuestMemoryWrite::Write(write.clone()))
    }

    pub(crate) fn queue_pending_guest_memory_write(
        &mut self,
        write: PendingGuestMemoryWrite,
    ) -> Result<(), String> {
        // RELEASE_MEM/WRITE_DATA label updates are ordered after the Metal work already encoded
        // in this batch, but do not themselves require a new Metal submission. Publish them when
        // the owning command buffer retires. A later wait can inspect the deferred value, while
        // a genuinely blocked wait and the event-loop boundary still commit the batch.
        if let Some(batch) = self.open_guest_batch.as_mut() {
            batch.memory_writes.push(write);
            return Ok(());
        }
        if let Some(batch) = self.committed_guest_batches.back_mut() {
            batch.memory_writes.push(write);
        } else {
            apply_pending_guest_memory_writes(std::slice::from_ref(&write))?;
            self.record_produced_guest_memory_writes(std::slice::from_ref(&write));
        }
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

    pub(crate) fn record_produced_global_writebacks(&mut self, pending: &[PendingGlobalWriteback]) {
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
            match write {
                PendingGuestMemoryWrite::Release(release) => match release.value {
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
                        let value = read_guest_memory_wait_value(GpuMemoryWait {
                            address: release.address,
                            reference: 0,
                            mask: u64::MAX,
                            compare_function: 0,
                            is_64_bit: true,
                        });
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
                PendingGuestMemoryWrite::Write(write) => {
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
                PendingGuestMemoryWrite::Fill { address, value, .. } => {
                    self.record_guest_memory_producer(*address, false, u64::from(*value));
                    if address.is_multiple_of(8) {
                        self.record_guest_memory_producer(
                            *address,
                            true,
                            u64::from(*value) | (u64::from(*value) << 32),
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
}

impl PendingGuestMemoryWrite {
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
            Self::Write(_) | Self::Release(_) => None,
        }
    }

    pub(crate) fn overlaps(&self, address: u64, byte_count: u64) -> bool {
        let (write_address, write_byte_count) = self.range();
        guest_ranges_overlap(write_address, write_byte_count, address, byte_count)
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

pub(crate) fn apply_pending_guest_memory_writes(
    pending: &[PendingGuestMemoryWrite],
) -> Result<(), String> {
    static GPU_CLOCK_START: OnceLock<Instant> = OnceLock::new();
    for write in pending {
        match write {
            PendingGuestMemoryWrite::Release(release) => {
                validate_guest_memory_release(*release)?;
                // SAFETY: packet decoding admits only aligned, live process-local guest
                // destinations. The owning Metal batch has completed, and volatile writes publish
                // the ordered label update before submission completion becomes visible.
                unsafe {
                    match release.value {
                        GpuMemoryReleaseValue::Immediate32(value) => {
                            std::ptr::write_volatile(release.address as *mut u32, value);
                        }
                        GpuMemoryReleaseValue::Immediate64(value) => {
                            std::ptr::write_volatile(release.address as *mut u64, value);
                        }
                        GpuMemoryReleaseValue::Timestamp => {
                            let timestamp = u64::try_from(
                                GPU_CLOCK_START
                                    .get_or_init(Instant::now)
                                    .elapsed()
                                    .as_nanos(),
                            )
                            .unwrap_or(u64::MAX)
                            .max(1);
                            std::ptr::write_volatile(release.address as *mut u64, timestamp);
                        }
                    }
                }
            }
            PendingGuestMemoryWrite::Write(write) => {
                validate_guest_memory_write(write)?;
                // SAFETY: validation above proves aligned, non-overflowing destinations in the
                // guest mapping supplied by the runtime.
                unsafe {
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
                        std::ptr::write_volatile(address as *mut u32, value);
                    }
                }
            }
            PendingGuestMemoryWrite::Fill {
                address,
                value,
                dword_count,
            } => {
                let byte_count = dword_count
                    .checked_mul(4)
                    .ok_or("DMA fill byte count overflowed")?;
                if !address.is_multiple_of(4) || address.checked_add(byte_count).is_none() {
                    return Err(format!(
                        "DMA fill destination {address:#x} dwords={dword_count:#x} is invalid"
                    ));
                }
                // SAFETY: packet decoding admitted an aligned, non-overflowing guest range. The
                // owning Metal batch has retired, so publishing the repeated dwords here preserves
                // DMA ordering for following waits and CPU readers.
                unsafe {
                    for index in 0..*dword_count {
                        let offset = index
                            .checked_mul(4)
                            .ok_or("DMA fill destination offset overflowed")?;
                        std::ptr::write_volatile((*address + offset) as *mut u32, *value);
                    }
                }
            }
        }
    }
    Ok(())
}
