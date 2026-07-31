// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    AGC_IT_CLEAR_STATE, AGC_IT_COND_EXEC, AGC_IT_EVENT_WRITE, AGC_IT_INDIRECT_BUFFER, AGC_IT_NOP,
    AGC_IT_SET_PREDICATION, AGC_R_ACB_RESET, AGC_R_DRAW_RESET, AGC_R_FLIP, AgcGraphicsState,
    AgcShaderRegistry, AgcSubmissionError, AgcVideoOutTarget, VideoOutState, agc_command_abi,
    agc_packet_dword_count, agc_video_out_target, process_guest_range_is_readable, read_guest_u32,
    trace_agc_programs, trace_agc_registers, trace_logarithmic_sequence,
};

use super::{
    AgcSubmissionPacket, agc_apply_predication, agc_branch_target, agc_cond_exec_skip_end,
    agc_handle_submission_packet, decode_agc_event_write,
};

pub(crate) struct AgcSubmissionContext<'a> {
    pub(crate) graphics: &'a mut AgcGraphicsState,
    pub(crate) shaders: &'a mut AgcShaderRegistry,
    pub(crate) video_out: &'a VideoOutState,
    pub(crate) submission_count: &'a mut u64,
    pub(crate) flips: &'a mut Vec<(u32, u32, u64)>,
    pub(crate) event_writes: &'a mut Vec<u32>,
}

pub(super) fn agc_packet_video_out_target(
    packet: AgcSubmissionPacket,
    video_out: &VideoOutState,
) -> Option<AgcVideoOutTarget> {
    if packet.opcode != AGC_IT_NOP || packet.register != AGC_R_FLIP || packet.length < 6 {
        return None;
    }
    agc_video_out_target(
        video_out,
        read_guest_u32(packet.address + 4),
        read_guest_u32(packet.address + 8),
    )
}

pub(super) fn agc_scanout_inference_boundary(packet: AgcSubmissionPacket) -> bool {
    packet.predicated
        || matches!(
            packet.opcode,
            AGC_IT_SET_PREDICATION | AGC_IT_COND_EXEC | AGC_IT_INDIRECT_BUFFER | AGC_IT_CLEAR_STATE
        )
        || (packet.opcode == AGC_IT_NOP
            && matches!(packet.register, AGC_R_DRAW_RESET | AGC_R_ACB_RESET))
}

pub(crate) fn agc_parse_submission(
    command_address: u64,
    dword_count: u32,
    context: &mut AgcSubmissionContext<'_>,
) -> Result<(), AgcSubmissionError> {
    let next_submission = context.submission_count.saturating_add(1);
    // Packet counts are diagnostic only. Building a BTreeMap for every ordinary command buffer,
    // and another discarded map for every nested IB, added allocator traffic directly to the
    // guest's submission path. Collect the same top-level summary only when it will be printed.
    let collect_packet_counts =
        trace_agc_programs() || trace_logarithmic_sequence(next_submission, 8);
    let (offset, packet_counts) = agc_parse_command_stream(
        command_address,
        dword_count,
        context,
        0,
        collect_packet_counts,
    )?;

    *context.submission_count = next_submission;
    if collect_packet_counts {
        let packet_summary = packet_counts
            .iter()
            .map(|(&(opcode, register), &count)| {
                format!("op={opcode:#04x}/reg={register:#04x}:{count}")
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "[REMu][AGC] submit={} address={command_address:#x} dwords={dword_count} parsed={offset} packets=[{packet_summary}] flips={}",
            context.submission_count,
            context.flips.len()
        );
    }
    Ok(())
}

pub(crate) fn agc_parse_graphics_submission(
    command_address: u64,
    dword_count: u32,
    context: &mut AgcSubmissionContext<'_>,
) -> Result<(), AgcSubmissionError> {
    let continuation_packets = agc_command_abi::take_graphics_continuations_before(command_address);
    let has_explicit_continuation = !continuation_packets.is_empty();
    for packet_address in continuation_packets {
        let packet = AgcSubmissionPacket {
            address: packet_address,
            length: 14,
            opcode: AGC_IT_INDIRECT_BUFFER,
            register: 0,
            predicated: false,
        };
        if let Some((target_address, target_dword_count)) = agc_branch_target(packet)? {
            eprintln!(
                "[REMu][AGC] applying graphics continuation packet={packet_address:#x} target={target_address:#x}/{target_dword_count} before={command_address:#x}"
            );
            agc_parse_command_stream(target_address, target_dword_count, context, 1, false)?;
            agc_command_abi::consume_recorded_graphics_range(target_address, target_dword_count);
        }
    }
    // Continuations cover the IB14 grow-split prefix. A separate HLE RELEASE can still sit
    // immediately before the submitted tail (outside the continuation target). Suppressing prefix
    // recovery whenever any continuation matched is what leaves Poppy title waits with
    // producers=[none-ingested] after the IB path already consumed a different range.
    if let Some((prefix_address, prefix_dword_count)) =
        agc_command_abi::take_recorded_graphics_prefix(command_address)
    {
        eprintln!(
            "[REMu][AGC] applying recorded graphics prefix address={prefix_address:#x} dwords={prefix_dword_count} before={command_address:#x} after_continuation={has_explicit_continuation}"
        );
        agc_parse_command_stream(prefix_address, prefix_dword_count, context, 1, false)?;
    }
    agc_parse_submission(command_address, dword_count, context)
}

pub(super) fn agc_parse_command_stream(
    command_address: u64,
    dword_count: u32,
    context: &mut AgcSubmissionContext<'_>,
    nesting: u32,
    collect_packet_counts: bool,
) -> Result<(u32, BTreeMap<(u32, u32), u32>), AgcSubmissionError> {
    const MAX_INDIRECT_BUFFER_NESTING: u32 = 32;
    if nesting > MAX_INDIRECT_BUFFER_NESTING {
        return Err(AgcSubmissionError::IndirectBufferNesting {
            maximum: MAX_INDIRECT_BUFFER_NESTING,
        });
    }
    let mut offset = 0_u32;
    let mut packet_counts = BTreeMap::<(u32, u32), u32>::new();
    let mut packets = Vec::new();
    while offset < dword_count {
        let packet_address = command_address + u64::from(offset) * 4;
        let header = read_guest_u32(packet_address);
        let packet_type = header >> 30;
        let Some(length) = agc_packet_dword_count(header) else {
            return Err(AgcSubmissionError::UnsupportedPacketType {
                dword: offset,
                packet_type,
                header,
            });
        };
        if length == 1 {
            offset += 1;
            continue;
        }
        let remaining = dword_count - offset;
        if length > remaining {
            return Err(AgcSubmissionError::InvalidPacketLength {
                dword: offset,
                address: packet_address,
                header,
                length,
                remaining,
            });
        }
        let opcode = (header >> 8) & 0xff;
        let register = (header >> 2) & 0x3f;
        if collect_packet_counts {
            *packet_counts.entry((opcode, register)).or_default() += 1;
        }
        packets.push(AgcSubmissionPacket {
            address: packet_address,
            length,
            opcode,
            register,
            predicated: header & 1 != 0,
        });
        offset += length;
    }

    let future_scanout_targets = agc_future_scanout_targets(&packets, context.video_out);
    let command_end = command_address + u64::from(dword_count) * 4;
    let mut conditional_skip_end = None;
    for (packet, future_scanout_target) in packets
        .iter()
        .copied()
        .zip(future_scanout_targets.into_iter())
    {
        if conditional_skip_end.is_some_and(|skip_end| packet.address < skip_end) {
            let packet_end = packet.address + u64::from(packet.length) * 4;
            if packet_end > conditional_skip_end.unwrap_or_default() {
                return Err(AgcSubmissionError::InvalidCondExecRange {
                    packet_address: packet.address,
                    skip_dwords: packet.length,
                });
            }
            continue;
        }
        conditional_skip_end = None;
        if packet.predicated && context.graphics.predicate_skip {
            if trace_agc_registers() {
                eprintln!(
                    "[REMu][AGC][PRED] skipped packet={:#x} opcode={:#x} register={:#x} dwords={}",
                    packet.address, packet.opcode, packet.register, packet.length
                );
            }
            continue;
        }
        // The command processor decodes order-visible packet payloads only after predication and
        // COND_EXEC have selected the executable stream. Validating all packet forms in a
        // preliminary pass rejected intentionally skipped debug/query packets and therefore the
        // whole DCB, which Unreal treats as fatal.
        if packet.opcode == AGC_IT_EVENT_WRITE {
            decode_agc_event_write(packet.address, packet.length)?;
        }
        if packet.opcode == AGC_IT_INDIRECT_BUFFER && !matches!(packet.length, 4 | 14) {
            return Err(AgcSubmissionError::IndirectBufferPacketLength {
                actual: packet.length,
            });
        }
        if packet.opcode == AGC_IT_SET_PREDICATION {
            agc_apply_predication(packet, context.graphics)?;
            continue;
        }
        if packet.opcode == AGC_IT_COND_EXEC {
            conditional_skip_end = agc_cond_exec_skip_end(packet, command_end)?;
            continue;
        }
        if packet.opcode == AGC_IT_INDIRECT_BUFFER {
            if packet.length == 14 {
                if agc_command_abi::graphics_continuation_was_consumed(packet.address) {
                    if trace_agc_registers() || trace_agc_programs() {
                        eprintln!(
                            "[REMu][AGC][BRANCH] skipped already-applied continuation packet={:#x}",
                            packet.address
                        );
                    }
                    continue;
                }
                if let Some((branch_address, branch_dword_count)) = agc_branch_target(packet)? {
                    agc_parse_command_stream(
                        branch_address,
                        branch_dword_count,
                        context,
                        nesting + 1,
                        false,
                    )?;
                    // This target has now executed as part of the native branch. Its constructor
                    // records exist only so a target separated from its submitted tail can be
                    // recovered when the branch packet itself is unavailable. Leaving them live
                    // lets the adjacent tail reconstruct and execute this prefix a second time,
                    // after completion labels in it may already have been returned to the guest
                    // allocator. That stale replay creates producerless WAIT_REG_MEM deadlocks.
                    agc_command_abi::consume_recorded_graphics_range(
                        branch_address,
                        branch_dword_count,
                    );
                }
                agc_command_abi::retire_pending_graphics_continuation(packet.address);
                continue;
            }
            let indirect_address = u64::from(read_guest_u32(packet.address + 4) & !3)
                | (u64::from(read_guest_u32(packet.address + 8)) << 32);
            let indirect_dword_count = read_guest_u32(packet.address + 12) & 0x000f_ffff;
            if indirect_dword_count == 0 {
                continue;
            }
            let byte_count = usize::try_from(u64::from(indirect_dword_count) * 4)
                .expect("the 20-bit indirect dword count fits usize");
            if indirect_address == 0
                || !process_guest_range_is_readable(indirect_address, byte_count)
            {
                return Err(AgcSubmissionError::InvalidIndirectBuffer {
                    address: indirect_address,
                    dword_count: indirect_dword_count,
                });
            }
            static INDIRECT_TRACE_COUNT: AtomicU64 = AtomicU64::new(0);
            let sequence = INDIRECT_TRACE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
            if trace_agc_programs() || (nesting < 8 && trace_logarithmic_sequence(sequence, 32)) {
                eprintln!(
                    "[REMu][AGC] indirect sequence={sequence} depth={} packet={:#x} address={indirect_address:#x} dwords={indirect_dword_count}",
                    nesting + 1,
                    packet.address,
                );
            }
            agc_parse_command_stream(
                indirect_address,
                indirect_dword_count,
                context,
                nesting + 1,
                false,
            )?;
            continue;
        }
        agc_handle_submission_packet(
            packet,
            context.graphics,
            context.shaders,
            context.video_out,
            context.flips,
            context.event_writes,
            future_scanout_target,
        )?;
    }

    Ok((offset, packet_counts))
}

pub(crate) fn agc_future_scanout_targets(
    packets: &[AgcSubmissionPacket],
    video_out: &VideoOutState,
) -> Vec<Option<AgcVideoOutTarget>> {
    let mut future = vec![None; packets.len()];
    let mut next = None;
    for (index, packet) in packets.iter().copied().enumerate().rev() {
        if agc_scanout_inference_boundary(packet) {
            next = None;
            continue;
        }
        future[index] = next;
        if let Some(target) = agc_packet_video_out_target(packet, video_out) {
            next = Some(target);
        }
    }
    future
}
