// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{
    AGC_R_WAIT_MEM_64, AgcSubmissionError, low_u32, read_guest_u32, read_guest_u64,
    trace_guest_sync,
};

/// GPU label / fence addresses used by UE WAIT_REG_MEM handshakes (Poppy title menu).
/// Always trace producers that touch this window — rate-limited SYNC logs previously hid them.
fn is_gpu_sync_label_address(address: u64) -> bool {
    (0x7e00_0000_0000..0x8000_0000_0000).contains(&address)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AgcEventWrite {
    PipelineBarrier(remu_gpu::GpuPipelineBarrier),
    RenderBackendFlush(remu_gpu::GpuRenderBackendFlush),
    PixelPipeStatDump { destination: u64 },
}

pub(crate) fn decode_agc_event_write(
    packet_address: u64,
    length: u32,
) -> Result<Option<AgcEventWrite>, AgcSubmissionError> {
    if length < 2 {
        return Err(AgcSubmissionError::EventWritePacketLength { actual: length });
    }
    let control = read_guest_u32(packet_address + 4);
    let event_type = control & 0x3f;
    let event_index = (control >> 8) & 0xf;
    if matches!(event_type, 0x07 | 0x0f | 0x10) {
        return decode_agc_pipeline_barrier(packet_address, length)
            .map(AgcEventWrite::PipelineBarrier)
            .map(Some);
    }
    // SDK EventType CB/DB writeback-invalidate forms (drawcommandbuffer.h:77-84). sync.cpp
    // emits these before end-of-pipe / acquire; silently dropping them deletes the only
    // render-backend flush mechanism.
    if let Some(flush) = remu_gpu::GpuRenderBackendFlush::from_event_type(event_type) {
        if control & !0x3f != 0 {
            return Err(AgcSubmissionError::UnsupportedEventWriteControl(control));
        }
        return Ok(Some(AgcEventWrite::RenderBackendFlush(flush)));
    }
    // EVENT_TYPE=0x39 is PIXEL_PIPE_STAT_DUMP / sce::Agc::EventType::kOcclusionCounterDump
    // (sdk/target/include/agc/drawcommandbuffer.h:86). AMD assigns it EVENT_INDEX=1 (the
    // ZPASS_DONE/PIXEL_PIPE_STAT group), and the two remaining packet dwords name the qword-aligned
    // occlusion-counter destination. The attached RDNA2 ISA manual covers shader instructions,
    // not PM4, but AMD's packet documentation and Linux's GFX10 `nvd.h` agree on this four-dword
    // form. The command processor must publish a validity bit for every render-backend counter:
    // Unreal's blocking render-query path yields until those bits appear. J1: decode only names the
    // destination; `GpuEvent::OcclusionQuery` carries it to the Metal visibility-result publisher.
    if event_type == 0x39 {
        if event_index != 1 || length != 4 || control & !0x0000_0f3f != 0 {
            return Err(AgcSubmissionError::UnsupportedEventWrite {
                event_type,
                event_index,
                length,
            });
        }
        let destination = read_guest_u64(packet_address + 8);
        let result_block_bytes = super::PIXEL_PIPE_STAT_COUNTER_PAIRS * 16;
        if destination == 0
            || !destination.is_multiple_of(8)
            || destination.checked_add(result_block_bytes - 1).is_none()
        {
            return Err(AgcSubmissionError::InvalidPixelPipeStatAddress(destination));
        }
        return Ok(Some(AgcEventWrite::PixelPipeStatDump { destination }));
    }
    // Non-addressed Gen5 events use only EVENT_TYPE. EVENT_INDEX is reserved for the three
    // partial-flush forms above and the addressed pixel-pipe dump handled above; these packets
    // perform GPU cache actions and do not themselves signal a kernel graphics queue. The packet
    // processor consumes the header-declared length: some native command streams pad these events
    // to four dwords, and the unused payload does not change the event semantics.
    if control & !0x3f != 0 {
        return if event_index != 0 {
            Err(AgcSubmissionError::UnsupportedEventWrite {
                event_type,
                event_index,
                length,
            })
        } else {
            Err(AgcSubmissionError::UnsupportedEventWriteControl(control))
        };
    }
    Ok(None)
}

pub(super) fn decode_agc_memory_write(
    packet_address: u64,
    length: u32,
    standard_packet: bool,
) -> Option<remu_gpu::GpuMemoryWrite> {
    if length < 5 {
        return None;
    }
    let control = read_guest_u32(packet_address + 4);
    let (destination, increment_address) = if standard_packet {
        // GFX10 PKT3_WRITE_DATA: DST_SEL is 11:8 and ADDR_INCR=0 advances the address.
        ((control >> 8) & 0x0f, control & (1 << 16) == 0)
    } else {
        // sceAgc's NOP wrapper stores four byte-sized fields: destination, cache policy,
        // increment suppression and write confirmation.
        (control & 0xff, (control >> 16) & 0xff == 0)
    };
    if !matches!(destination, 1 | 2 | 4 | 5) {
        return None;
    }
    let address = read_guest_u64(packet_address + 8);
    if address == 0 || !address.is_multiple_of(4) {
        return None;
    }
    let values = (0..length - 4)
        .map(|index| read_guest_u32(packet_address + 16 + u64::from(index) * 4))
        .collect::<Vec<_>>();
    static WRITE_DATA_TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);
    let sequence = WRITE_DATA_TRACE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if is_gpu_sync_label_address(address) || sequence <= 32 || sequence.is_power_of_two() {
        eprintln!(
            "[REMu][AGC][SYNC] write sequence={sequence} packet={packet_address:#x} standard={standard_packet} destination={destination} address={address:#x} dwords={} increment={increment_address} values={values:?}",
            values.len()
        );
    }
    Some(remu_gpu::GpuMemoryWrite {
        address,
        values,
        increment_address,
    })
}

/// Decoded Gen5 `DMA_DATA`: either a memory copy/fill or SDK `gpuSyncDma` (`DmaDataDst::kNowhere`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum AgcDmaData {
    Copy(remu_gpu::GpuMemoryCopy),
    /// `sync.cpp` `gpuSyncDma`: dst=kNowhere, size=0, wait-previous + write-confirm + block CP.
    SyncBarrier,
}

pub(crate) fn decode_agc_dma_data(
    packet_address: u64,
    length: u32,
    standard_packet: bool,
) -> Option<AgcDmaData> {
    let (source_address, destination_address, byte_count, source_selection, destination_selection) =
        if standard_packet {
            if length < 7 {
                return None;
            }
            let control = read_guest_u32(packet_address + 4);
            let command = read_guest_u32(packet_address + 24);
            (
                read_guest_u64(packet_address + 8),
                read_guest_u64(packet_address + 16),
                u64::from(command & 0x03ff_ffff),
                ((control >> 29) & 0x3) | ((command >> 24) & 0x4) | ((command >> 25) & 0x8),
                ((control >> 20) & 0x3) | ((command >> 25) & 0x4) | ((command >> 26) & 0x8),
            )
        } else {
            if length < 8 {
                return None;
            }
            let control = read_guest_u32(packet_address + 4);
            (
                read_guest_u64(packet_address + 24),
                read_guest_u64(packet_address + 16),
                u64::from(read_guest_u32(packet_address + 12)),
                (control >> 16) & 0xff,
                control & 0xff,
            )
        };
    // SDK `gpuSyncDma` (sync.cpp:694-708): DmaDataDst::kNowhere (2), addresses 0, size 0.
    if destination_selection == 2 && destination_address == 0 && byte_count == 0 {
        return Some(AgcDmaData::SyncBarrier);
    }
    let source_is_memory = matches!(source_selection, 0 | 3);
    let source_is_immediate = source_selection == 2;
    if (!source_is_memory && !source_is_immediate)
        || (source_is_memory && source_address < 0x10_000)
        || destination_address < 0x10_000
        || byte_count == 0
        || byte_count > 512 * 1024 * 1024
    {
        return None;
    }
    Some(AgcDmaData::Copy(remu_gpu::GpuMemoryCopy {
        source_address,
        destination_address,
        byte_count,
        source_selection,
        destination_selection,
        source_snapshot: None,
    }))
}


pub(crate) fn decode_agc_memory_acquire(packet_address: u64) -> remu_gpu::GpuMemoryAcquire {
    let coher_control = read_guest_u32(packet_address + 4);
    let size_units = u64::from(read_guest_u32(packet_address + 8))
        | (u64::from(read_guest_u32(packet_address + 12) & 0xff) << 32);
    let base_units = u64::from(read_guest_u32(packet_address + 16))
        | (u64::from(read_guest_u32(packet_address + 20) & 0x00ff_ffff) << 32);
    remu_gpu::GpuMemoryAcquire {
        engine: coher_control >> 31,
        cb_db_control: coher_control & 0x7fff_ffff,
        base_address: base_units << 8,
        size_bytes: size_units << 8,
        poll_interval: read_guest_u32(packet_address + 24) & 0xffff,
        gcr_control: read_guest_u32(packet_address + 28) & 0x7_ffff,
    }
}

pub(super) fn decode_agc_memory_release(
    packet_address: u64,
    standard_packet: bool,
) -> Option<remu_gpu::GpuMemoryRelease> {
    static TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);
    // NOP wrapper layout (agc_command_abi::cb_release_mem / sync.cpp queueEndOfPipeAction):
    //   +4: action | (cache_policy << 8)
    //   +8: gcr_control | (data_selection << 16) | (interrupt << 24)
    let action_control = read_guest_u32(packet_address + 4);
    let control = read_guest_u32(packet_address + 8);
    let address = read_guest_u64(packet_address + 12);
    let data = read_guest_u64(packet_address + 20);
    let (action, cache_policy, gcr_control, data_selection, interrupt) = if standard_packet {
        (
            action_control & 0xff,
            (action_control >> 25) & 0x3,
            control & 0xffff,
            (control >> 29) & 0x7,
            (control >> 24) & 0x7,
        )
    } else {
        (
            action_control & 0xff,
            (action_control >> 8) & 0xff,
            control & 0xffff,
            (control >> 16) & 0xff,
            (control >> 24) & 0xff,
        )
    };
    let value = match data_selection {
        1 if address.is_multiple_of(4) => {
            Some(remu_gpu::GpuMemoryReleaseValue::Immediate32(low_u32(data)))
        }
        2 if address.is_multiple_of(8) => Some(remu_gpu::GpuMemoryReleaseValue::Immediate64(data)),
        3 | 4 if address.is_multiple_of(8) => Some(remu_gpu::GpuMemoryReleaseValue::Timestamp),
        _ => None,
    };
    let trace = TRACE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if is_gpu_sync_label_address(address)
        || trace_guest_sync()
        || trace <= 96
        || trace.is_power_of_two()
    {
        eprintln!(
            "[REMu][AGC][SYNC] release packet={packet_address:#x} standard={standard_packet} action={action:#x} gcr={gcr_control:#x} cache_policy={cache_policy} interrupt={interrupt} selection={data_selection} address={address:#x} data={data:#x} emitted={}",
            value.is_some() && address != 0
        );
    }
    Some(remu_gpu::GpuMemoryRelease {
        address,
        value: value?,
        action,
        gcr_control,
        cache_policy,
        interrupt,
    })
    .filter(|_| address != 0)
}


pub(crate) fn decode_agc_memory_wait(
    packet_address: u64,
    register: u32,
    standard_packet: bool,
) -> Option<remu_gpu::GpuMemoryWait> {
    static TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);
    let is_64_bit = !standard_packet && register == AGC_R_WAIT_MEM_64;
    let (address, reference, mask, control) = if standard_packet {
        (
            read_guest_u64(packet_address + 8),
            u64::from(read_guest_u32(packet_address + 16)),
            u64::from(read_guest_u32(packet_address + 20)),
            read_guest_u32(packet_address + 4),
        )
    } else if is_64_bit {
        (
            read_guest_u64(packet_address + 4),
            read_guest_u64(packet_address + 20),
            read_guest_u64(packet_address + 12),
            read_guest_u32(packet_address + 28),
        )
    } else {
        (
            read_guest_u64(packet_address + 4),
            u64::from(read_guest_u32(packet_address + 16)),
            u64::from(read_guest_u32(packet_address + 12)),
            read_guest_u32(packet_address + 20),
        )
    };
    let compare_function = control & 0x7;
    // SDK WaitOperation packing (agc_command_abi::wait_reg_mem / drawcommandbuffer.h):
    // engine select is the low bit (kWaitMe=0, kWaitPfp=1).
    let (operation, cache_policy) = if standard_packet {
        ((control >> 8) & 0x1, (control >> 25) & 0x3)
    } else if is_64_bit {
        let engine = (control >> 8) & 0x1;
        let cond = (control >> 6) & 0x2;
        (engine | cond, (control >> 25) & 0x3)
    } else {
        let low = (control >> 8) & 0x3;
        let high = (control >> 4) & 0x0c;
        (low | high, (control >> 25) & 0x3)
    };
    let alignment = if is_64_bit { 8 } else { 4 };
    let decoded = if address == 0 || !address.is_multiple_of(alignment) || compare_function == 7 {
        None
    } else {
        Some(remu_gpu::GpuMemoryWait {
            address,
            reference,
            mask,
            compare_function,
            is_64_bit,
            operation,
            cache_policy,
        })
    };
    let trace = TRACE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if trace_guest_sync() || trace <= 64 || trace.is_power_of_two() {
        let current = decoded.map(|wait| {
            if wait.is_64_bit {
                read_guest_u64(wait.address)
            } else {
                u64::from(read_guest_u32(wait.address))
            }
        });
        eprintln!(
            "[REMu][AGC][SYNC] wait packet={packet_address:#x} standard={standard_packet} bits={} control={control:#010x} compare={compare_function} operation={operation} cache_policy={cache_policy} address={address:#x} mask={mask:#x} reference={reference:#x} current={current:?} satisfied={:?}",
            if is_64_bit { 64 } else { 32 },
            current
                .zip(decoded)
                .map(|(value, wait)| wait.is_satisfied_by(value))
        );
    }
    decoded
}

pub(crate) fn decode_agc_pipeline_barrier(
    packet_address: u64,
    length: u32,
) -> Result<remu_gpu::GpuPipelineBarrier, AgcSubmissionError> {
    if length < 2 {
        return Err(AgcSubmissionError::EventWritePacketLength { actual: length });
    }
    let control = read_guest_u32(packet_address + 4);
    let event_type = control & 0x3f;
    let event_index = (control >> 8) & 0x7;
    let wait_for = match event_type {
        0x07 => remu_gpu::GpuPipelineStage::ComputeShader,
        0x0f => remu_gpu::GpuPipelineStage::VertexShader,
        0x10 => remu_gpu::GpuPipelineStage::PixelShader,
        _ => {
            return Err(AgcSubmissionError::UnsupportedEventWrite {
                event_type,
                event_index,
                length,
            });
        }
    };
    if control & !0x0000_073f != 0 {
        return Err(AgcSubmissionError::UnsupportedEventWriteControl(control));
    }
    if event_index != 4 {
        return Err(AgcSubmissionError::PartialFlushEventIndex {
            event_type,
            actual: event_index,
        });
    }
    Ok(remu_gpu::GpuPipelineBarrier { wait_for })
}
