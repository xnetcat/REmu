// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    GuestMemoryWriteOperation, MetalPresenter, PendingGlobalWriteback, PendingGuestMemoryWrite,
    WriteWatchRefresh, guest_batch_writes_overlap, guest_ranges_overlap, guest_texture_byte_count,
    merge_guest_batch_result, read_guest_memory_wait_value, rearm_color_metadata_clear,
    rearm_depth_metadata_clear, record_diagnostic_memory_copy,
    refresh_resident_texture_write_watch, retire_multisample_textures_overlapping_writeback,
    retire_resident_textures_overlapping_writeback, retire_sampled_textures_matching,
    retire_sampled_textures_overlapping_writeback, trace_resident_texture_removal,
    write_resident_texture_contents_to_guest,
};
use crate::presenter::diagnostics::trace_diagnostic_guest_write;
use remu_gpu::{
    GpuMemoryCopy, GpuMemoryRelease, GpuMemoryReleaseValue, GpuMemoryWait, GpuMemoryWrite,
};
use remu_memory::{
    CpuWriteWatch, GuestSharedMappingIdentity, guest_shared_mapping_identity,
    notify_guest_memory_written, read_guest_shared_backing, write_guest_shared_backing_if_identity,
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
        trace_diagnostic_guest_write(
            "release",
            release.address,
            if matches!(release.value, GpuMemoryReleaseValue::Immediate32(_)) {
                4
            } else {
                8
            },
        );
        self.queue_pending_guest_memory_write(GuestMemoryWriteOperation::Release(release))
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
            rearm_color_metadata_clear(&mut self.color_metadata_clears, copy.destination_address);
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
            let pending_source_write = self.open_guest_batch.as_ref().is_some_and(|batch| {
                guest_batch_writes_overlap(batch, copy.source_address, copy.byte_count)
            }) || self.committed_guest_batches.iter().any(|batch| {
                guest_batch_writes_overlap(batch, copy.source_address, copy.byte_count)
            });
            let mut authoritative_gpu_source = pending_source_write;
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
                    if self
                        .resident_guest_textures
                        .get_mut(&descriptor)
                        .expect("refreshed DMA source texture exists")
                        .cpu_write_obsoletes_gpu_writeback(refresh)?
                    {
                        if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
                            trace_resident_texture_removal("cpu-reused-dma-source", &resident);
                        }
                        continue;
                    }
                    authoritative_gpu_source = true;
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
                        &mut self.retired_texture_quarantine,
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
                    resident.clear_gpu_authored();
                }
            }

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
                // addresses. The guest owns these mappings for the ordered event stream. The live
                // path intentionally provides memmove semantics for legal overlapping packets.
                unsafe {
                    std::ptr::copy(
                        copy.source_address as *const u8,
                        copy.destination_address as *mut u8,
                        byte_count,
                    );
                }
            }
            let _ = notify_guest_memory_written(copy.destination_address, copy.byte_count);
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
        retire_sampled_textures_matching(&mut self.sampled_texture_cache, |key, _| {
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
    }
}

pub(crate) fn pin_guest_memory_write(
    operation: GuestMemoryWriteOperation,
) -> Result<PendingGuestMemoryWrite, String> {
    validate_guest_memory_operation(&operation)?;
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
        GuestMemoryWriteOperation::Release(release) => {
            // SAFETY: the caller pins the original mapping against replacement for the duration
            // of this operation. Validation proves the destination alignment.
            unsafe {
                match release.value {
                    GpuMemoryReleaseValue::Immediate32(value) => {
                        std::ptr::write_volatile(release.address as *mut u32, value);
                    }
                    GpuMemoryReleaseValue::Immediate64(value) => {
                        std::ptr::write_volatile(release.address as *mut u64, value);
                    }
                    GpuMemoryReleaseValue::Timestamp => {
                        std::ptr::write_volatile(release.address as *mut u64, gpu_timestamp());
                    }
                }
            }
        }
        GuestMemoryWriteOperation::Write(write) => {
            // SAFETY: the caller pins the original mapping against replacement for the duration
            // of this operation. Validation proves every destination is aligned and bounded.
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
        GuestMemoryWriteOperation::Fill {
            address,
            value,
            dword_count,
        } => {
            // SAFETY: the caller pins the complete original mapping against replacement.
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
    Ok(())
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
    if write_guest_shared_backing_if_identity(destination, expected, bytes) {
        Ok(())
    } else {
        Err(format!(
            "deferred guest memory mapping changed before completion address={destination:#x} bytes={}",
            bytes.len()
        ))
    }
}

fn apply_guest_memory_operation_shared(
    operation: &GuestMemoryWriteOperation,
    identity: GuestSharedMappingIdentity,
) -> Result<(), String> {
    let (address, _) = operation.range();
    match operation {
        GuestMemoryWriteOperation::Release(release) => {
            let bytes = match release.value {
                GpuMemoryReleaseValue::Immediate32(value) => value.to_le_bytes().to_vec(),
                GpuMemoryReleaseValue::Immediate64(value) => value.to_le_bytes().to_vec(),
                GpuMemoryReleaseValue::Timestamp => gpu_timestamp().to_le_bytes().to_vec(),
            };
            write_shared_identity_bytes(address, identity, 0, &bytes)
        }
        GuestMemoryWriteOperation::Write(write) if write.increment_address => {
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(write.values.len().saturating_mul(4))
                .map_err(|error| format!("could not stage WRITE_DATA values: {error}"))?;
            for value in &write.values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            write_shared_identity_bytes(address, identity, 0, &bytes)
        }
        GuestMemoryWriteOperation::Write(write) => {
            for value in &write.values {
                write_shared_identity_bytes(address, identity, 0, &value.to_le_bytes())?;
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
    }
}

pub(crate) fn apply_pending_guest_memory_writes(
    pending: &[PendingGuestMemoryWrite],
) -> Result<(), String> {
    for pending_write in pending {
        validate_guest_memory_operation(&pending_write.operation)?;
        let (pixel_pipe_trace, query_completion_trace) = match pending_write.operation {
            GuestMemoryWriteOperation::Release(release) => (
                remu_gpu::take_pixel_pipe_stat_release_trace(release),
                remu_gpu::take_pixel_pipe_query_completion_release_trace(release),
            ),
            GuestMemoryWriteOperation::Write(_) | GuestMemoryWriteOperation::Fill { .. } => {
                (None, None)
            }
        };
        let observed_before = (pixel_pipe_trace.is_some() || query_completion_trace.is_some())
            .then(|| observe_pending_guest_memory_release(pending_write))
            .flatten();
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
    }) = pending.operation
    else {
        return None;
    };
    if pending.mapping_identity.is_some() {
        let mut bytes = [0_u8; 8];
        return read_guest_shared_backing(address, &mut bytes).then(|| u64::from_le_bytes(bytes));
    }
    let write_watch = pending.write_watch.as_ref()?;
    write_watch
        .with_live_mapping(|| {
            // SAFETY: the write watch pins this aligned eight-byte guest range against replacement
            // for the complete diagnostic read.
            unsafe { std::ptr::read_volatile(address as *const u64) }
        })
        .ok()
}
