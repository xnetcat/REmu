// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::{collections::BTreeSet, sync::Mutex};

use super::{
    COMMAND_BUFFER_CALLBACK_OFFSET, COMMAND_BUFFER_CURSOR_DOWN_OFFSET,
    COMMAND_BUFFER_CURSOR_UP_OFFSET, COMMAND_BUFFER_RESERVED_DWORDS_OFFSET,
    COMMAND_BUFFER_USER_DATA_OFFSET, GrowCallback,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PendingGraphicsContinuation {
    packet: u64,
    then_start: u64,
    then_end: u64,
    else_start: u64,
    else_end: u64,
}

pub(super) static RECENT_COMMAND_ALLOCATIONS: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

pub(super) static PENDING_GRAPHICS_CONTINUATIONS: Mutex<Vec<PendingGraphicsContinuation>> =
    Mutex::new(Vec::new());

pub(super) static CONSUMED_GRAPHICS_CONTINUATIONS: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// Command-buffer descriptors most recently used through an ACB-only constructor.
///
/// The Gen5 common helpers accept both DCB and ACB descriptors, so the descriptor is the stable
/// piece of information which tells `allocate_dwords` whether a neighboring allocation can extend
/// a submitted graphics stream. Without this distinction, an ACB built immediately after a DCB in
/// the same backing ring is reconstructed as a post-submit graphics tail and then executed again
/// through the compute submission.
pub(super) static ASYNC_COMMAND_BUFFERS: Mutex<BTreeSet<u64>> = Mutex::new(BTreeSet::new());
pub(super) fn mark_async_command_buffer(command_buffer: u64) {
    if command_buffer != 0 {
        ASYNC_COMMAND_BUFFERS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(command_buffer);
    }
}

pub(super) fn mark_graphics_command_buffer(command_buffer: u64) {
    if command_buffer != 0 {
        ASYNC_COMMAND_BUFFERS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&command_buffer);
    }
}

pub(super) fn is_async_command_buffer(command_buffer: u64) -> bool {
    ASYNC_COMMAND_BUFFERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .contains(&command_buffer)
}

pub(super) fn trace_failed_sync_command(
    name: &str,
    arguments: [u64; 6],
    stack_arguments: [u64; 6],
    command: u64,
) -> u64 {
    if command == 0 {
        eprintln!(
            "[REMu][AGC][SYNC][BUILD] {name} rejected args={arguments:#x?} stack={stack_arguments:#x?}"
        );
    }
    command
}

pub(super) fn low_u32(value: u64) -> u32 {
    u32::try_from(value & u64::from(u32::MAX)).expect("value is masked to 32 bits")
}

pub(super) fn low_u16(value: u64) -> u32 {
    low_u32(value) & 0xffff
}

pub(super) fn low_u8(value: u64) -> u32 {
    low_u32(value) & 0xff
}

pub(super) fn pm4(length_dwords: u32, opcode: u32, register: u32) -> u32 {
    0xc000_0000
        | ((length_dwords.wrapping_sub(2) & 0x3fff) << 16)
        | ((opcode & 0xff) << 8)
        | ((register & 0x3f) << 2)
}

pub(super) fn packet_opcode(command: u64) -> Option<u32> {
    (command != 0 && command.is_multiple_of(4)).then(|| (read_u32(command) >> 8) & 0xff)
}

pub(super) fn packet_register(command: u64) -> u32 {
    (read_u32(command) >> 2) & 0x3f
}

pub(super) fn is_packet(command: u64, opcode: u32, register: Option<u32>) -> bool {
    packet_opcode(command).is_some_and(|actual| actual == opcode)
        && register.is_none_or(|expected| packet_register(command) == expected)
}

pub(super) fn read_u8(address: u64) -> u8 {
    // SAFETY: import arguments are guest pointers into REmu's live in-process mapping.
    unsafe { std::ptr::read_unaligned(address as *const u8) }
}

pub(super) fn read_u32(address: u64) -> u32 {
    // SAFETY: import arguments are guest pointers into REmu's live in-process mapping.
    unsafe { std::ptr::read_unaligned(address as *const u32) }
}

pub(super) fn read_u64(address: u64) -> u64 {
    // SAFETY: import arguments are guest pointers into REmu's live in-process mapping.
    unsafe { std::ptr::read_unaligned(address as *const u64) }
}

pub(super) fn write_u32(address: u64, value: u32) {
    // SAFETY: validated command-buffer addresses point into writable guest memory.
    unsafe { std::ptr::write_unaligned(address as *mut u32, value) }
}

pub(super) fn write_u64(address: u64, value: u64) {
    // SAFETY: validated command-buffer addresses point into writable guest memory.
    unsafe { std::ptr::write_unaligned(address as *mut u64, value) }
}

pub(super) fn write_words(address: u64, words: &[u32]) {
    for (index, &word) in words.iter().enumerate() {
        write_u32(
            address + u64::try_from(index).expect("word index fits") * 4,
            word,
        );
    }
}

pub(super) fn allocate_dwords(command_buffer: u64, dword_count: u32) -> Option<u64> {
    if command_buffer == 0 || command_buffer & 7 != 0 || dword_count == 0 {
        return None;
    }
    let mut cursor_up = read_u64(command_buffer + COMMAND_BUFFER_CURSOR_UP_OFFSET);
    let mut cursor_down = read_u64(command_buffer + COMMAND_BUFFER_CURSOR_DOWN_OFFSET);
    let reserved = read_u32(command_buffer + COMMAND_BUFFER_RESERVED_DWORDS_OFFSET);
    if !has_capacity(cursor_up, cursor_down, reserved, dword_count) {
        let previous_cursor_up = cursor_up;
        let previous_cursor_down = cursor_down;
        let callback = read_u64(command_buffer + COMMAND_BUFFER_CALLBACK_OFFSET);
        if callback == 0 {
            return None;
        }
        let requested = dword_count.checked_add(reserved)?;
        let user_data = read_u64(command_buffer + COMMAND_BUFFER_USER_DATA_OFFSET);
        // SAFETY: this field is the guest-provided SysV command-buffer growth callback.
        let callback_address = usize::try_from(callback).ok()?;
        // SAFETY: the validated address is the callback pointer stored in the guest descriptor.
        let callback: GrowCallback = unsafe { std::mem::transmute(callback_address) };
        // SAFETY: the callback ABI and arguments are defined by the Gen5 CommandBuffer layout.
        if !unsafe { callback(command_buffer, requested, user_data) } {
            return None;
        }
        cursor_up = read_u64(command_buffer + COMMAND_BUFFER_CURSOR_UP_OFFSET);
        cursor_down = read_u64(command_buffer + COMMAND_BUFFER_CURSOR_DOWN_OFFSET);
        eprintln!(
            "[REMu][AGC] command-buffer grow descriptor={command_buffer:#x} requested={dword_count} reserved={reserved} old={previous_cursor_up:#x}..{previous_cursor_down:#x} new={cursor_up:#x}..{cursor_down:#x}"
        );
        if !has_capacity(cursor_up, cursor_down, reserved, dword_count) {
            return None;
        }
    }
    let byte_count = u64::from(dword_count).checked_mul(4)?;
    let next = cursor_up.checked_add(byte_count)?;
    write_u64(command_buffer + COMMAND_BUFFER_CURSOR_UP_OFFSET, next);
    if !is_async_command_buffer(command_buffer) {
        track_graphics_allocation(cursor_up, dword_count);
    }
    Some(cursor_up)
}

pub(super) fn register_graphics_continuation(
    packet: u64,
    then_address: u64,
    then_count: u32,
    else_address: u64,
    else_count: u32,
) {
    let target_end = |address: u64, count: u32| {
        (address != 0 && count != 0)
            .then(|| address.checked_add(u64::from(count) * 4))
            .flatten()
            .unwrap_or(0)
    };
    let then_end = target_end(then_address, then_count);
    let else_end = target_end(else_address, else_count);
    if packet == 0 || (then_end == 0 && else_end == 0) {
        return;
    }

    CONSUMED_GRAPHICS_CONTINUATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .retain(|&consumed| consumed != packet);
    let mut pending = PENDING_GRAPHICS_CONTINUATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    pending.retain(|continuation| continuation.packet != packet);
    pending.push(PendingGraphicsContinuation {
        packet,
        then_start: then_address,
        then_end,
        else_start: else_address,
        else_end,
    });
    if pending.len() > 4_096 {
        let drain_count = pending.len() - 2_048;
        pending.drain(..drain_count);
    }
}

/// Takes branch packets whose selected indirect-buffer range is immediately before a submitted
/// DCB. Gen5 uses this form when a command buffer grows: the branch executes the prefix written
/// into the replacement allocation, then execution continues with the explicitly submitted tail.
pub(crate) fn take_graphics_continuations_before(command_address: u64) -> Vec<u64> {
    let mut pending = PENDING_GRAPHICS_CONTINUATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut matched = Vec::new();
    let mut matched_ranges = Vec::new();
    pending.retain(|continuation| {
        if continuation.then_end == command_address || continuation.else_end == command_address {
            matched.push(continuation.packet);
            matched_ranges.extend([
                (continuation.then_start, continuation.then_end),
                (continuation.else_start, continuation.else_end),
            ]);
            false
        } else {
            true
        }
    });
    drop(pending);
    for (start, end) in matched_ranges {
        if let Ok(dword_count) = u32::try_from(end.saturating_sub(start) / 4)
            && start != 0
            && dword_count != 0
        {
            consume_recorded_graphics_range(start, dword_count);
        }
    }

    if !matched.is_empty() {
        let mut consumed = CONSUMED_GRAPHICS_CONTINUATIONS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        consumed.extend(matched.iter().copied());
        if consumed.len() > 4_096 {
            let drain_count = consumed.len() - 2_048;
            consumed.drain(..drain_count);
        }
    }
    matched
}

pub(crate) fn graphics_continuation_was_consumed(packet: u64) -> bool {
    CONSUMED_GRAPHICS_CONTINUATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .contains(&packet)
}

/// Retires constructor-side recovery metadata after the native branch packet itself executes.
///
/// The pending record exists only to recover a branch target which was separated from the
/// explicitly submitted tail by command-buffer growth. If the packet is already present in a
/// parsed DCB, normal branch execution has performed that work and the recovery record must not
/// replay it before the next DCB. This deliberately does not mark the packet globally consumed:
/// a native command stream may legally execute the same indirect branch more than once.
pub(crate) fn retire_pending_graphics_continuation(packet: u64) {
    let mut pending = PENDING_GRAPHICS_CONTINUATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut retired_ranges = Vec::new();
    pending.retain(|continuation| {
        if continuation.packet == packet {
            retired_ranges.extend([
                (continuation.then_start, continuation.then_end),
                (continuation.else_start, continuation.else_end),
            ]);
            false
        } else {
            true
        }
    });
    drop(pending);
    // Once the native branch packet is present in a parsed DCB, both alternatives belong to that
    // branch's control-flow graph. The selected one executes now and the unselected one does not;
    // neither may later be guessed as a missing prefix for an adjacent submission.
    for (start, end) in retired_ranges {
        if let Ok(dword_count) = u32::try_from(end.saturating_sub(start) / 4)
            && start != 0
            && dword_count != 0
        {
            consume_recorded_graphics_range(start, dword_count);
        }
    }
}

/// Recovers command packets allocated before an explicitly submitted DCB tail.
///
/// A grow callback can move the DCB while another guest thread is preparing the stitching branch.
/// The command constructors have still recorded the replacement buffer's prefix. Only a complete,
/// readable PM4 stream ending exactly at the submitted address is accepted, so unrelated recent
/// allocations cannot be replayed as graphics work.
pub(crate) fn take_recorded_graphics_prefix(command_address: u64) -> Option<(u64, u32)> {
    const MAX_PREFIX_DWORDS: u64 = 1_000_000;
    let lower_bound = command_address.saturating_sub(MAX_PREFIX_DWORDS * 4);
    let mut allocations = RECENT_COMMAND_ALLOCATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut candidates = allocations
        .iter()
        .filter_map(|&(start, end)| {
            (start >= lower_bound && start < command_address && end <= command_address)
                .then_some(start)
        })
        .collect::<Vec<_>>();
    candidates.sort_unstable();
    candidates.dedup();
    // Some Gen5 command helpers reserve only an anchor packet; the guest writes neighboring PM4
    // directly. Accept those guest-authored packets only inside a complete readable stream that
    // begins at a constructor-recorded address and ends exactly at the submitted tail. Ranges
    // parsed by an explicit branch or earlier submission are retired separately, preventing a
    // valid-looking old stream from being selected a second time.
    let prefix_start = candidates
        .into_iter()
        .find(|&start| scan_valid_command_stream_end(start, command_address) == command_address)?;
    let dword_count = u32::try_from((command_address - prefix_start) / 4).ok()?;
    if dword_count == 0 {
        return None;
    }
    allocations.retain(|&(start, end)| !(start >= prefix_start && end <= command_address));
    Some((prefix_start, dword_count))
}

/// Retires command-constructor records covered by work that has already been parsed.
///
/// A native branch target and the submitted DCB that follows it are distinct allocations. Both
/// must be retired after use; otherwise a later tail at a higher address can reconstruct and
/// execute the branch target for a second time.
pub(crate) fn consume_recorded_graphics_range(command_address: u64, dword_count: u32) {
    let Some(command_end) = command_address.checked_add(u64::from(dword_count) * 4) else {
        return;
    };
    RECENT_COMMAND_ALLOCATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .retain(|&(allocation_start, allocation_end)| {
            !(allocation_start >= command_address && allocation_end <= command_end)
        });
}

pub(crate) fn track_graphics_allocation(command: u64, dword_count: u32) {
    if command == 0 || dword_count == 0 {
        return;
    }
    let Some(command_end) = command.checked_add(u64::from(dword_count) * 4) else {
        return;
    };
    let mut allocations = RECENT_COMMAND_ALLOCATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    allocations.push((command, command_end));
    if allocations.len() > 16_384 {
        let drain_count = allocations.len() - 8_192;
        allocations.drain(..drain_count);
    }
}

pub(crate) fn track_graphics_submission(command_address: u64, dword_count: u32) {
    let Some(submitted_end) = command_address.checked_add(u64::from(dword_count) * 4) else {
        return;
    };
    {
        let mut allocations = RECENT_COMMAND_ALLOCATIONS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Command-buffer storage is a ring and may wrap to a numerically lower address. Retire only
        // allocations covered by this submission; dropping every lower address loses a freshly
        // built prefix in a recycled allocation. Allocations beyond `submitted_end` belong to a
        // future DCB: the guest can build it before submitting an intervening ACB, but it must not
        // execute until that future DCB is explicitly submitted.
        allocations.retain(|&(allocation_start, allocation_end)| {
            !(allocation_start >= command_address && allocation_end <= submitted_end)
        });
    }

    // A grow/stitch branch can be constructed before its replacement allocation is submitted
    // directly. Once every non-empty target of that branch has been covered by a successful
    // explicit submission, retaining the branch would replay the same target when the adjacent
    // tail arrives. Besides duplicating draws, release-memory packets can then write queue labels
    // into storage which the guest has already returned to its allocator.
    let target_was_submitted =
        |start: u64, end: u64| end == 0 || (start >= command_address && end <= submitted_end);
    let mut retired_packets = Vec::new();
    {
        let mut pending = PENDING_GRAPHICS_CONTINUATIONS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        pending.retain(|continuation| {
            let retire = target_was_submitted(continuation.then_start, continuation.then_end)
                && target_was_submitted(continuation.else_start, continuation.else_end);
            if retire {
                retired_packets.push(continuation.packet);
            }
            !retire
        });
    }
    if !retired_packets.is_empty() {
        let mut consumed = CONSUMED_GRAPHICS_CONTINUATIONS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        consumed.extend(retired_packets);
        if consumed.len() > 4_096 {
            let drain_count = consumed.len() - 2_048;
            consumed.drain(..drain_count);
        }
    }
}

pub(super) fn scan_valid_command_stream_end(start: u64, recorded_end: u64) -> u64 {
    if start == 0 || recorded_end <= start {
        return start;
    }
    let Ok(byte_count) = usize::try_from(recorded_end - start) else {
        return start;
    };
    // Prove readability from the mapped-range snapshot without materializing the span: this runs
    // once per recorded candidate per submission, and copying multi-megabyte prefixes here
    // dominated whole frames on command-heavy titles while starving other import threads of the
    // dispatch lock.
    if !crate::process_guest_range_has_read_access(start, byte_count) {
        return start;
    }
    let mut command = start;
    while command < recorded_end {
        let header = read_u32(command);
        let dword_count = match header >> 30 {
            2 => 1,
            3 => ((header >> 16) & 0x3fff) + 2,
            _ => break,
        };
        let Some(next) = command.checked_add(u64::from(dword_count) * 4) else {
            break;
        };
        if next > recorded_end {
            break;
        }
        command = next;
    }
    command
}

pub(super) fn has_capacity(cursor_up: u64, cursor_down: u64, reserved: u32, needed: u32) -> bool {
    cursor_up != 0
        && cursor_up.is_multiple_of(4)
        && cursor_down.is_multiple_of(4)
        && cursor_down > cursor_up
        && cursor_down
            .checked_sub(cursor_up)
            .map(|bytes| bytes / 4)
            .and_then(|available| available.checked_sub(u64::from(reserved)))
            .is_some_and(|available| available >= u64::from(needed))
}
