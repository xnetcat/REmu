// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use remu_shader::GuestMemoryReader;

use crate::{
    AgcGraphicsState, AgcSubmissionError, ProcessGuestMemory, process_guest_range_is_readable,
    read_guest_u32, read_guest_u64, trace_agc_programs, trace_agc_registers,
};

use super::AgcSubmissionPacket;

pub(super) fn agc_read_mapped_u32(address: u64) -> Option<u32> {
    let mut bytes = [0_u8; 4];
    ProcessGuestMemory
        .read(address, &mut bytes)
        .then(|| u32::from_le_bytes(bytes))
}

pub(crate) fn agc_read_mapped_u64(address: u64) -> Option<u64> {
    let mut bytes = [0_u8; 8];
    ProcessGuestMemory
        .read(address, &mut bytes)
        .then(|| u64::from_le_bytes(bytes))
}

pub(super) fn agc_apply_predication(
    packet: AgcSubmissionPacket,
    graphics: &mut AgcGraphicsState,
) -> Result<(), AgcSubmissionError> {
    if packet.length < 3 {
        return Err(AgcSubmissionError::PredicationPacketLength {
            actual: packet.length,
        });
    }
    let payload_dwords = packet.length - 1;
    let first = read_guest_u32(packet.address + 4);
    let second = read_guest_u32(packet.address + 8);
    let third = (payload_dwords >= 3).then(|| read_guest_u32(packet.address + 12));
    const PREDICATION_FLAGS_MASK: u32 = 0x0007_1100;
    let (flags, value_address) =
        if third.is_some_and(|third| first & !PREDICATION_FLAGS_MASK == 0 && third <= 0xffff) {
            (
                first,
                u64::from(second & 0xffff_fff0) | (u64::from(third.unwrap_or_default()) << 32),
            )
        } else {
            (
                second,
                u64::from(first & 0xffff_fff0) | (u64::from(second & 0xff) << 32),
            )
        };
    let condition = (flags >> 8) & 1;
    let operation = (flags >> 16) & 7;
    match operation {
        0 => graphics.predicate_skip = false,
        3 => {
            let value = agc_read_mapped_u64(value_address).ok_or(
                AgcSubmissionError::UnreadablePredicationValue(value_address),
            )?;
            graphics.predicate_skip = match condition {
                0 => value != 0,
                _ => value == 0,
            };
            if trace_agc_registers() || trace_agc_programs() {
                eprintln!(
                    "[REMu][AGC][PRED] address={value_address:#x} value={value:#x} condition={condition} skip={}",
                    graphics.predicate_skip
                );
            }
        }
        _ => {
            return Err(AgcSubmissionError::UnsupportedPredicationOperation(
                operation,
            ));
        }
    }
    Ok(())
}

pub(super) fn agc_cond_exec_skip_end(
    packet: AgcSubmissionPacket,
    command_end: u64,
) -> Result<Option<u64>, AgcSubmissionError> {
    if packet.length != 5 {
        return Err(AgcSubmissionError::CondExecPacketLength {
            actual: packet.length,
        });
    }
    let low = read_guest_u32(packet.address + 4);
    let high = read_guest_u32(packet.address + 8);
    let reserved = read_guest_u32(packet.address + 12);
    let control = low & 3;
    if control != 0 || reserved != 0 {
        return Err(AgcSubmissionError::UnsupportedCondExecControl(
            control | reserved,
        ));
    }
    let value_address = u64::from(low & !3) | (u64::from(high) << 32);
    let value = agc_read_mapped_u32(value_address)
        .ok_or(AgcSubmissionError::UnreadableCondExecValue(value_address))?;
    if value != 0 {
        return Ok(None);
    }
    let skip_dwords = read_guest_u32(packet.address + 16) & 0x3fff;
    let packet_end = packet.address + u64::from(packet.length) * 4;
    let Some(skip_end) = packet_end.checked_add(u64::from(skip_dwords) * 4) else {
        return Err(AgcSubmissionError::InvalidCondExecRange {
            packet_address: packet.address,
            skip_dwords,
        });
    };
    if skip_end > command_end {
        return Err(AgcSubmissionError::InvalidCondExecRange {
            packet_address: packet.address,
            skip_dwords,
        });
    }
    if trace_agc_registers() || trace_agc_programs() {
        eprintln!(
            "[REMu][AGC][COND] address={value_address:#x} value=0 skip_dwords={skip_dwords} end={skip_end:#x}"
        );
    }
    Ok(Some(skip_end))
}

pub(super) fn agc_branch_is_satisfied(
    value: u64,
    reference: u64,
    mask: u64,
    comparison: u32,
) -> bool {
    match comparison {
        0 => true,
        1 => value & mask < reference,
        2 => value & mask <= reference,
        3 => value & mask == reference,
        4 => value & mask != reference,
        5 => value & mask >= reference,
        6 => value & mask > reference,
        _ => false,
    }
}

pub(super) fn agc_branch_target(
    packet: AgcSubmissionPacket,
) -> Result<Option<(u64, u32)>, AgcSubmissionError> {
    let control = read_guest_u32(packet.address + 4);
    let mode = control & 3;
    let comparison = (control >> 8) & 7;
    if !matches!(mode, 1 | 2) {
        return Err(AgcSubmissionError::UnsupportedBranchMode(mode));
    }
    if comparison > 6 {
        return Err(AgcSubmissionError::UnsupportedBranchComparison(comparison));
    }
    let compare_address = u64::from(read_guest_u32(packet.address + 8) & !7)
        | (u64::from(read_guest_u32(packet.address + 12)) << 32);
    // Comparison zero is the native unconditional form. The address still acts as a queue label
    // for command-buffer stitching, but the command processor does not need its contents to pick
    // a target.
    let value = if comparison == 0 {
        0
    } else {
        agc_read_mapped_u64(compare_address)
            .ok_or(AgcSubmissionError::UnreadableBranchValue(compare_address))?
    };
    let mask = read_guest_u64(packet.address + 16);
    let reference = read_guest_u64(packet.address + 24);
    let take_then = agc_branch_is_satisfied(value, reference, mask, comparison);
    let (address_offset, count_offset) = if take_then {
        (32, 40)
    } else if mode == 2 {
        (44, 52)
    } else {
        return Ok(None);
    };
    let address = u64::from(read_guest_u32(packet.address + address_offset) & !3)
        | (u64::from(read_guest_u32(packet.address + address_offset + 4)) << 32);
    let dword_count = read_guest_u32(packet.address + count_offset) & 0x000f_ffff;
    if trace_agc_registers() || trace_agc_programs() {
        eprintln!(
            "[REMu][AGC][BRANCH] compare={compare_address:#x} value={value:#x} reference={reference:#x} mask={mask:#x} function={comparison} then={take_then} target={address:#x}/{dword_count}"
        );
    }
    (address != 0 && dword_count != 0)
        .then_some((address, dword_count))
        .map_or(Ok(None), |(address, dword_count)| {
            let byte_count = usize::try_from(u64::from(dword_count) * 4)
                .expect("the 20-bit branch dword count fits usize");
            if process_guest_range_is_readable(address, byte_count) {
                Ok(Some((address, dword_count)))
            } else {
                Err(AgcSubmissionError::InvalidIndirectBuffer {
                    address,
                    dword_count,
                })
            }
        })
}
