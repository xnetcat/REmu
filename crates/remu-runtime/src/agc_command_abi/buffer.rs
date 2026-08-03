// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use super::{
    COMMAND_BUFFER_BASE_DOWN_OFFSET, COMMAND_BUFFER_BASE_UP_OFFSET, COMMAND_BUFFER_CALLBACK_OFFSET,
    COMMAND_BUFFER_CURSOR_DOWN_OFFSET, COMMAND_BUFFER_CURSOR_UP_OFFSET,
    COMMAND_BUFFER_RESERVED_DWORDS_OFFSET, COMMAND_BUFFER_USER_DATA_OFFSET, GrowCallback,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PendingGraphicsContinuation {
    packet: u64,
    then_start: u64,
    then_end: u64,
    else_start: u64,
    else_end: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RecentGraphicsAllocation {
    descriptor: u64,
    generation: u64,
    buffer_start: u64,
    buffer_end: u64,
    start: u64,
    end: u64,
    /// This exact allocation is an HLE-constructed release which has not yet appeared in a
    /// successful submission. Such a release can be the synchronization prefix immediately
    /// before a guest-authored DCB tail even when no grow callback ran.
    recoverable_release: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GraphicsCommandBufferGeneration {
    generation: u64,
    buffer_start: u64,
    buffer_end: u64,
    last_allocation_end: u64,
    /// Start of a replacement allocation observed directly after the guest grow callback.
    ///
    /// Only this boundary proves that commands preceding an explicitly submitted tail were split
    /// away by growth. Cursor regression is ordinary ring reuse and does not prove a missing
    /// prefix, even when the recycled bytes still form valid PM4.
    recovery_start: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SubmittedGraphicsRange {
    start: u64,
    end: u64,
}

static NEXT_GRAPHICS_GENERATION: AtomicU64 = AtomicU64::new(0);
static GRAPHICS_COMMAND_BUFFER_GENERATIONS: Mutex<BTreeMap<u64, GraphicsCommandBufferGeneration>> =
    Mutex::new(BTreeMap::new());
static RECENT_COMMAND_ALLOCATIONS: Mutex<Vec<RecentGraphicsAllocation>> = Mutex::new(Vec::new());
static RECENT_GRAPHICS_SUBMISSIONS: Mutex<Vec<SubmittedGraphicsRange>> = Mutex::new(Vec::new());
#[cfg(test)]
static GRAPHICS_TRACKING_TEST_LOCK: Mutex<()> = Mutex::new(());

const MAX_RECENT_GRAPHICS_ALLOCATIONS: usize = 16_384;
const TARGET_RECENT_GRAPHICS_ALLOCATIONS: usize = 8_192;

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

#[cfg(test)]
pub(crate) fn graphics_tracking_test_guard() -> std::sync::MutexGuard<'static, ()> {
    let guard = GRAPHICS_TRACKING_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    NEXT_GRAPHICS_GENERATION.store(0, Ordering::Relaxed);
    GRAPHICS_COMMAND_BUFFER_GENERATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    RECENT_COMMAND_ALLOCATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    RECENT_GRAPHICS_SUBMISSIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    PENDING_GRAPHICS_CONTINUATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    CONSUMED_GRAPHICS_CONTINUATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    ASYNC_COMMAND_BUFFERS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
    guard
}

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

/// The command-buffer descriptor fields this module reads, through the reserved dword count.
const COMMAND_BUFFER_DESCRIPTOR_BYTES: u64 = COMMAND_BUFFER_RESERVED_DWORDS_OFFSET + 4;

/// Reports whether a guest range is live in this process before any raw pointer access.
///
/// The command-ABI entry points take pointers straight from guest registers, and titles do call
/// them with pointers the guest itself would reject. Octopath's RHIThread reaches the register
/// indirect-patch imports with a null descriptor; the surrounding validation returns
/// `GRAPHICS5_ERROR_INVALID_PACKET` for that, but a non-null pointer into unmapped memory used to
/// reach `read_u32` unchecked and fault the emulator instead. The other AGC command modules
/// already gate every read this way.
///
/// This runs for every packet the guest builds, so it consults the cached kernel mapping table
/// first and only falls back to a probing read for addresses that table does not describe.
fn guest_range_is_readable(address: u64, length: u64) -> bool {
    if address == 0 || address.checked_add(length).is_none() {
        return false;
    }
    usize::try_from(length)
        .is_ok_and(|length| crate::process_guest_range_has_read_access(address, length))
}

pub(super) fn packet_opcode(command: u64) -> Option<u32> {
    (command != 0 && command.is_multiple_of(4) && guest_range_is_readable(command, 4))
        .then(|| (read_u32(command) >> 8) & 0xff)
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

/// Names why a packet reservation was refused. A refused reservation makes the builder return a
/// null packet address, which the import-failure predicate cannot see — the guest simply loses
/// the packet and with it a draw, dispatch, or fence — so the reason is named here or nowhere.
pub(crate) fn decline_reservation(
    reason: &str,
    command_buffer: u64,
    dword_count: u32,
) -> Option<u64> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static DECLINED: AtomicU64 = AtomicU64::new(0);
    let sequence = DECLINED.fetch_add(1, Ordering::Relaxed) + 1;
    if crate::trace_logarithmic_sequence(sequence, 16) {
        eprintln!(
            "[REMu][AGC] declined packet reservation sequence={sequence} \
             descriptor={command_buffer:#x} dwords={dword_count} reason={reason}"
        );
    }
    None
}

pub(super) fn allocate_dwords(command_buffer: u64, dword_count: u32) -> Option<u64> {
    if command_buffer == 0
        || command_buffer & 7 != 0
        || dword_count == 0
        || !guest_range_is_readable(command_buffer, COMMAND_BUFFER_DESCRIPTOR_BYTES)
    {
        return decline_reservation("unusable-descriptor", command_buffer, dword_count);
    }
    let mut cursor_up = read_u64(command_buffer + COMMAND_BUFFER_CURSOR_UP_OFFSET);
    let mut cursor_down = read_u64(command_buffer + COMMAND_BUFFER_CURSOR_DOWN_OFFSET);
    let reserved = read_u32(command_buffer + COMMAND_BUFFER_RESERVED_DWORDS_OFFSET);
    let mut grew = false;
    if !has_capacity(cursor_up, cursor_down, reserved, dword_count) {
        let previous_cursor_up = cursor_up;
        let previous_cursor_down = cursor_down;
        let callback = read_u64(command_buffer + COMMAND_BUFFER_CALLBACK_OFFSET);
        if callback == 0 {
            return decline_reservation("full-without-grow-callback", command_buffer, dword_count);
        }
        let requested = dword_count.checked_add(reserved)?;
        let user_data = read_u64(command_buffer + COMMAND_BUFFER_USER_DATA_OFFSET);
        // SAFETY: this field is the guest-provided SysV command-buffer growth callback.
        let callback_address = usize::try_from(callback).ok()?;
        // SAFETY: the validated address is the callback pointer stored in the guest descriptor.
        let callback: GrowCallback = unsafe { std::mem::transmute(callback_address) };
        // SAFETY: the callback ABI and arguments are defined by the Gen5 CommandBuffer layout.
        if !unsafe { callback(command_buffer, requested, user_data) } {
            return decline_reservation("grow-refused", command_buffer, dword_count);
        }
        grew = true;
        cursor_up = read_u64(command_buffer + COMMAND_BUFFER_CURSOR_UP_OFFSET);
        cursor_down = read_u64(command_buffer + COMMAND_BUFFER_CURSOR_DOWN_OFFSET);
        eprintln!(
            "[REMu][AGC] command-buffer grow descriptor={command_buffer:#x} requested={dword_count} reserved={reserved} old={previous_cursor_up:#x}..{previous_cursor_down:#x} new={cursor_up:#x}..{cursor_down:#x}"
        );
        if !has_capacity(cursor_up, cursor_down, reserved, dword_count) {
            return decline_reservation("short-after-grow", command_buffer, dword_count);
        }
    }
    let byte_count = u64::from(dword_count).checked_mul(4)?;
    let next = cursor_up.checked_add(byte_count)?;
    write_u64(command_buffer + COMMAND_BUFFER_CURSOR_UP_OFFSET, next);
    if !is_async_command_buffer(command_buffer) {
        track_graphics_command_allocation(command_buffer, cursor_up, dword_count, grew);
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

    for (name, start, end) in [
        ("then", then_address, then_end),
        ("else", else_address, else_end),
    ] {
        if start == 0 || end == 0 {
            continue;
        }
        let valid_end = scan_valid_command_stream_end(start, end);
        if valid_end != end {
            let offset = valid_end.saturating_sub(start) / 4;
            let header = (valid_end < end).then(|| read_u32(valid_end));
            eprintln!(
                "[REMu][AGC][BRANCH-BUILD] invalid {name} target packet={packet:#x} range={start:#x}..{end:#x} valid_end={valid_end:#x} dword={offset} header={header:#010x?}"
            );
        }
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
    // One-shot: the adjacent-tail skip must suppress only the immediate reappearance of a
    // continuation already applied via take_graphics_continuations_before. Keeping the address
    // forever poisons recycled command-buffer slots — later legitimate 14-dword INDIRECT_BUFFER
    // packets at the same address are skipped and their RELEASE producers never emit, which
    // deadlocks WAIT_REG_MEM with producers=[none-ingested] (Poppy title menu).
    let mut consumed = CONSUMED_GRAPHICS_CONTINUATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(index) = consumed.iter().rposition(|&candidate| candidate == packet) {
        consumed.remove(index);
        true
    } else {
        false
    }
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
/// Recoverable HLE releases that can form a readable PM4 prefix ending at `command_address`.
///
/// Ordered earliest-start first (then newest generation) so a run of marked releases before the
/// submitted tail is recovered together. Callers must skip candidates that cross an already
/// submitted range — a stale terminal RELEASE from a prior recovered prefix can still scan as
/// valid PM4 after ring reuse and would otherwise poison every later submit.
fn adjacent_recoverable_release_candidates(
    command_address: u64,
    lower_bound: u64,
) -> Vec<(u64, u64, u64)> {
    let mut candidates: Vec<_> = RECENT_COMMAND_ALLOCATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .filter(|allocation| {
            allocation.recoverable_release
                && allocation.end <= command_address
                && allocation.start >= lower_bound
                && scan_valid_command_stream_end(allocation.start, command_address)
                    == command_address
                && !prefix_crosses_submitted_range(allocation.start, command_address)
        })
        .map(|allocation| {
            (
                allocation.descriptor,
                allocation.generation,
                allocation.start,
            )
        })
        .collect();
    candidates.sort_by_key(|&(_, generation, prefix_start)| {
        (prefix_start, std::cmp::Reverse(generation))
    });
    candidates
}

/// Retires every recoverable release covered by a recovered prefix, regardless of generation.
///
/// Matching only the recovered generation leaves a later-generation mark at the terminal RELEASE
/// of that prefix live. The next submit then selects that stale tip, fails submission-overlap,
/// and never tries a newer adjacent producer — Poppy title q41 waits stay `none-ingested`.
fn retire_recoverable_releases_in_prefix(prefix_start: u64, command_address: u64) {
    RECENT_COMMAND_ALLOCATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .retain(|allocation| {
            !(allocation.recoverable_release
                && allocation.start >= prefix_start
                && allocation.end <= command_address)
        });
}

/// Drops marked releases whose bytes already executed in a recorded submission.
///
/// Without this, a later-generation re-mark at a prior prefix's terminal RELEASE keeps winning
/// earliest-start selection and only fails later on overlap — forever blocking newer producers if
/// the caller returns on the first failure.
fn prune_submitted_recoverable_releases() {
    let submitted = RECENT_GRAPHICS_SUBMISSIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    if submitted.is_empty() {
        return;
    }
    RECENT_COMMAND_ALLOCATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .retain(|allocation| {
            if !allocation.recoverable_release {
                return true;
            }
            !submitted
                .iter()
                .any(|range| allocation.start >= range.start && allocation.end <= range.end)
        });
}

fn prefix_crosses_submitted_range(prefix_start: u64, command_address: u64) -> bool {
    RECENT_GRAPHICS_SUBMISSIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .any(|range| range.start < command_address && prefix_start < range.end)
}

pub(crate) fn take_recorded_graphics_prefix(command_address: u64) -> Option<(u64, u32)> {
    const MAX_PREFIX_DWORDS: u64 = 1_000_000;
    let lower_bound = command_address.saturating_sub(MAX_PREFIX_DWORDS * 4);
    prune_submitted_recoverable_releases();
    let recoveries = GRAPHICS_COMMAND_BUFFER_GENERATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // The callback-observed replacement start is the only authoritative evidence that an
    // explicitly submitted tail lost a prefix to command-buffer growth. A cursor regression can
    // mean ordinary ring reuse; scanning that recycled memory for PM4 shape replays stale waits.
    let grown_prefix = recoveries
        .iter()
        .filter_map(|(&descriptor, recovery)| {
            recovery
                .recovery_start
                .filter(|&prefix_start| {
                    recovery.buffer_start <= prefix_start
                        && prefix_start < command_address
                        && command_address <= recovery.buffer_end
                        && prefix_start >= lower_bound
                })
                .map(|prefix_start| (descriptor, recovery.generation, prefix_start))
        })
        .max_by_key(|&(_, generation, _)| generation);
    drop(recoveries);

    // Prefer a validated grow split, but never let a stale/invalid grow boundary permanently hide
    // a still-pending HLE release that sits immediately before the submitted tail.
    if let Some((descriptor, generation, prefix_start)) = grown_prefix {
        let scan_ok =
            scan_valid_command_stream_end(prefix_start, command_address) == command_address;
        let overlap = prefix_crosses_submitted_range(prefix_start, command_address);
        if scan_ok && !overlap {
            let dword_count = u32::try_from((command_address - prefix_start) / 4).ok()?;
            if dword_count != 0 {
                let mut recoveries = GRAPHICS_COMMAND_BUFFER_GENERATIONS
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(recovery) = recoveries.get_mut(&descriptor)
                    && recovery.generation == generation
                    && recovery.recovery_start == Some(prefix_start)
                {
                    recovery.recovery_start = None;
                    drop(recoveries);
                    eprintln!(
                        "[REMu][AGC][PREFIX] recovered grow split={prefix_start:#x}..{command_address:#x}/{dword_count} descriptor={descriptor:#x} generation={generation}"
                    );
                    retire_recoverable_releases_in_prefix(prefix_start, command_address);
                    let mut allocations = RECENT_COMMAND_ALLOCATIONS
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    allocations.retain(|allocation| {
                        allocation.descriptor != descriptor
                            || allocation.generation != generation
                            || !(allocation.start >= prefix_start
                                && allocation.end <= command_address)
                    });
                    return Some((prefix_start, dword_count));
                }
                eprintln!(
                    "[REMu][AGC][PREFIX] grow split rejected submit={command_address:#x} prefix={prefix_start:#x} reason=generation-mismatch; trying adjacent release"
                );
            }
        } else {
            eprintln!(
                "[REMu][AGC][PREFIX] grow split rejected submit={command_address:#x} prefix={prefix_start:#x} reason={} ; trying adjacent release",
                if !scan_ok {
                    "no-valid-scan"
                } else {
                    "submission-overlap"
                }
            );
        }
    }

    // Gen5 also emits a release through an HLE constructor immediately before a guest-authored
    // DCB tail without invoking the descriptor's grow callback. Exact `end == submit` is too
    // strict: UE often inserts NOP padding between the HLE release and the submitted tail.
    let candidates = adjacent_recoverable_release_candidates(command_address, lower_bound);
    if candidates.is_empty() {
        let marked = RECENT_COMMAND_ALLOCATIONS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .filter(|allocation| {
                allocation.recoverable_release
                    && allocation.end <= command_address
                    && allocation.start >= lower_bound
            })
            .count();
        if marked != 0 {
            eprintln!(
                "[REMu][AGC][PREFIX] adjacent release rejected submit={command_address:#x} marked={marked} reason=no-valid-unsubmitted-candidate"
            );
        }
        return None;
    }

    for (descriptor, generation, prefix_start) in candidates {
        let dword_count = match u32::try_from((command_address - prefix_start) / 4) {
            Ok(count) if count != 0 => count,
            _ => continue,
        };
        // Re-check under the current submission set: a concurrent submit can land between
        // candidate collection and recovery.
        if prefix_crosses_submitted_range(prefix_start, command_address)
            || scan_valid_command_stream_end(prefix_start, command_address) != command_address
        {
            eprintln!(
                "[REMu][AGC][PREFIX] adjacent release skipped submit={command_address:#x} prefix={prefix_start:#x} reason=submission-overlap-or-scan"
            );
            continue;
        }
        eprintln!(
            "[REMu][AGC][PREFIX] recovered adjacent release={prefix_start:#x}..{command_address:#x}/{dword_count} descriptor={descriptor:#x} generation={generation}"
        );
        retire_recoverable_releases_in_prefix(prefix_start, command_address);
        return Some((prefix_start, dword_count));
    }
    None
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
        .retain(|allocation| {
            !(allocation.start >= command_address && allocation.end <= command_end)
        });
    for recovery in GRAPHICS_COMMAND_BUFFER_GENERATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .values_mut()
    {
        if recovery
            .recovery_start
            .is_some_and(|start| command_address <= start && start < command_end)
        {
            recovery.recovery_start = None;
        }
    }
}

/// Marks a graphics-DCB HLE release as a recoverable adjacent-tail prefix.
///
/// Async command buffers allocate without graphics tracking; their RELEASE packets execute in the
/// ACB submit path and must not be treated as DCB prefix candidates.
pub(crate) fn mark_graphics_release_for_command_buffer(command_buffer: u64, command: u64) {
    if is_async_command_buffer(command_buffer) {
        return;
    }
    mark_graphics_release_allocation(command);
}

/// Marks one constructor allocation as a synchronization prefix candidate.
///
/// The guest can submit a directly authored DCB tail beginning immediately after this release.
/// Keeping the semantic marker on the allocation lets submission recovery include that release
/// without treating unrelated PM4 bytes in the recycled command ring as executable.
pub(crate) fn mark_graphics_release_allocation(command: u64) {
    const RELEASE_DWORDS: u64 = 8;
    let Some(command_end) = command.checked_add(RELEASE_DWORDS * 4) else {
        return;
    };
    // The HLE constructor just wrote these words; still refuse to probe unmapped command memory.
    if command == 0 || !command.is_multiple_of(4) {
        eprintln!(
            "[REMu][AGC][PREFIX] mark release missed command={command:#x} reason=bad-address"
        );
        return;
    }
    if !is_packet(command, super::IT_NOP, Some(super::R_RELEASE_MEM))
        && !is_packet(command, super::IT_RELEASE_MEM, None)
    {
        eprintln!(
            "[REMu][AGC][PREFIX] mark release missed command={command:#x} reason=not-release-packet"
        );
        return;
    }
    let mut allocations = RECENT_COMMAND_ALLOCATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // Prefer an exact 8-dword constructor record. When the release sits inside a larger tracked
    // span, publish a dedicated 8-dword recoverable record so adjacent-tail recovery can require
    // `end <= submit` without dragging unrelated PM4 into the prefix.
    if let Some(allocation) = allocations
        .iter_mut()
        .rev()
        .find(|allocation| allocation.start == command && allocation.end == command_end)
    {
        allocation.recoverable_release = true;
        return;
    }
    if let Some(template) = allocations
        .iter()
        .rev()
        .find(|allocation| allocation.start <= command && allocation.end >= command_end)
        .copied()
    {
        allocations.push(RecentGraphicsAllocation {
            descriptor: template.descriptor,
            generation: template.generation,
            buffer_start: template.buffer_start,
            buffer_end: template.buffer_end,
            start: command,
            end: command_end,
            recoverable_release: true,
        });
        return;
    }
    eprintln!(
        "[REMu][AGC][PREFIX] mark release missed command={command:#x}..{command_end:#x} reason=no-allocation"
    );
}

/// Bounds ordinary constructor history without discarding pending queue synchronization.
///
/// A recoverable release remains live until a submission consumes it, prefix recovery takes it,
/// or its command-buffer generation is replaced. Evicting one merely because unrelated draw
/// constructors crossed the history budget loses a real release/wait handshake and can leave both
/// guest queues waiting forever. Ordinary allocation records are only reconstruction aids, so
/// those are the safe records to prune under pressure.
fn prune_recent_graphics_allocations(allocations: &mut Vec<RecentGraphicsAllocation>) {
    if allocations.len() <= MAX_RECENT_GRAPHICS_ALLOCATIONS {
        return;
    }

    let mut ordinary_to_remove = allocations
        .len()
        .saturating_sub(TARGET_RECENT_GRAPHICS_ALLOCATIONS);
    allocations.retain(|allocation| {
        if ordinary_to_remove != 0 && !allocation.recoverable_release {
            ordinary_to_remove -= 1;
            false
        } else {
            true
        }
    });
}

#[cfg(test)]
pub(crate) fn track_graphics_allocation(command: u64, dword_count: u32) {
    track_graphics_allocation_for_generation(0, 0, 0, u64::MAX, command, dword_count);
}

/// Records a constructor allocation against the concrete command-buffer generation that owns it.
///
/// A successful growth callback is authoritative even if it recycles the same virtual extent.
/// Cursor regression covers guest-side ring resets which do not invoke the callback.
pub(crate) fn track_graphics_command_allocation(
    command_buffer: u64,
    command: u64,
    dword_count: u32,
    grew: bool,
) {
    if command == 0 || dword_count == 0 {
        return;
    }
    let Some(command_end) = command.checked_add(u64::from(dword_count) * 4) else {
        return;
    };
    let buffer_start = read_u64(command_buffer + COMMAND_BUFFER_BASE_UP_OFFSET);
    let buffer_end = read_u64(command_buffer + COMMAND_BUFFER_BASE_DOWN_OFFSET);
    if buffer_start == 0
        || buffer_end <= buffer_start
        || command < buffer_start
        || command_end > buffer_end
    {
        eprintln!(
            "[REMu][AGC] ignored allocation outside descriptor extent descriptor={command_buffer:#x} buffer={buffer_start:#x}..{buffer_end:#x} allocation={command:#x}..{command_end:#x}"
        );
        return;
    }

    let (generation, started_new_generation) = {
        let mut generations = GRAPHICS_COMMAND_BUFFER_GENERATIONS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let needs_new_generation = generations.get(&command_buffer).is_none_or(|current| {
            grew || current.buffer_start != buffer_start
                || current.buffer_end != buffer_end
                || command < current.last_allocation_end
        });
        if needs_new_generation {
            let generation = NEXT_GRAPHICS_GENERATION.fetch_add(1, Ordering::Relaxed) + 1;
            generations.insert(
                command_buffer,
                GraphicsCommandBufferGeneration {
                    generation,
                    buffer_start,
                    buffer_end,
                    last_allocation_end: command_end,
                    recovery_start: grew.then_some(command),
                },
            );
            (generation, true)
        } else {
            let current = generations
                .get_mut(&command_buffer)
                .expect("an existing command-buffer generation was just observed");
            current.last_allocation_end = current.last_allocation_end.max(command_end);
            (current.generation, false)
        }
    };

    let mut allocations = RECENT_COMMAND_ALLOCATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if started_new_generation {
        // Ordinary constructor history for this descriptor is generation-scoped. An unsubmitted
        // HLE release is queue synchronization, not reconstruction aid: keep it across grow so the
        // adjacent-tail submit can still recover the producer. Only drop a marked release when the
        // new allocation recycles its bytes (cursor wrap / in-place reuse), which makes the old
        // packet contents invalid.
        allocations.retain(|allocation| {
            if allocation.descriptor != command_buffer {
                return true;
            }
            if allocation.recoverable_release {
                let overlaps_new = allocation.start < command_end && command < allocation.end;
                !overlaps_new
            } else {
                false
            }
        });
    }
    // Prune before appending so a release constructor's fresh allocation remains present for
    // `mark_graphics_release_allocation`, which runs after its words have been written.
    prune_recent_graphics_allocations(&mut allocations);
    allocations.push(RecentGraphicsAllocation {
        descriptor: command_buffer,
        generation,
        buffer_start,
        buffer_end,
        start: command,
        end: command_end,
        recoverable_release: false,
    });
    drop(allocations);
    if started_new_generation {
        RECENT_GRAPHICS_SUBMISSIONS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|submission| submission.start < buffer_start || submission.end > buffer_end);
    }
}

#[cfg(test)]
fn track_graphics_allocation_for_generation(
    descriptor: u64,
    generation: u64,
    buffer_start: u64,
    buffer_end: u64,
    command: u64,
    dword_count: u32,
) {
    if command == 0 || dword_count == 0 {
        return;
    }
    let Some(command_end) = command.checked_add(u64::from(dword_count) * 4) else {
        return;
    };
    let mut generations = GRAPHICS_COMMAND_BUFFER_GENERATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match generations.get_mut(&descriptor) {
        Some(current) if current.generation == generation => {
            current.last_allocation_end = current.last_allocation_end.max(command_end);
        }
        _ => {
            generations.insert(
                descriptor,
                GraphicsCommandBufferGeneration {
                    generation,
                    buffer_start,
                    buffer_end,
                    last_allocation_end: command_end,
                    recovery_start: Some(command),
                },
            );
        }
    }
    drop(generations);
    let mut allocations = RECENT_COMMAND_ALLOCATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    prune_recent_graphics_allocations(&mut allocations);
    allocations.push(RecentGraphicsAllocation {
        descriptor,
        generation,
        buffer_start,
        buffer_end,
        start: command,
        end: command_end,
        recoverable_release: false,
    });
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
        allocations.retain(|allocation| {
            !(allocation.start >= command_address && allocation.end <= submitted_end)
        });
    }
    for recovery in GRAPHICS_COMMAND_BUFFER_GENERATIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .values_mut()
    {
        if recovery
            .recovery_start
            .is_some_and(|start| command_address <= start && start < submitted_end)
        {
            recovery.recovery_start = None;
        }
    }
    let mut submissions = RECENT_GRAPHICS_SUBMISSIONS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    submissions.push(SubmittedGraphicsRange {
        start: command_address,
        end: submitted_end,
    });
    if submissions.len() > 16_384 {
        let drain_count = submissions.len() - 8_192;
        submissions.drain(..drain_count);
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
        let Some(dword_count) = crate::agc_packet_dword_count(header) else {
            break;
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
