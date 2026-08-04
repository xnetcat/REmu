// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Agc asynchronous compute command-buffer builders.
//!
//! This module implements sceAgcAcb* entry points, which append packets to a compute queue's command buffer.
//!
//! A builder reports the guest address of the packet it wrote, and zero when the request is
//! malformed or the buffer could not make room for it; none of the entry points here patch an
//! existing packet, so none of them report an SCE error code instead.
//!
//! A size query and the builder it names are two halves of one contract: the guest reserves
//! command-buffer space with the query and then fills that space with the builder, so a size
//! reported here has to match the packet the runtime actually emits for that call. Several size
//! queries have no builder of their own in this module, because the typed operation they size is
//! issued through a primitive that the shared command ABI already emits; those queries report that
//! primitive's packet.
//!
//! [`dispatch`] is consulted by the runtime's import dispatcher only after the main table
//! misses, and returns `None` for any NID it does not own so the caller can keep looking.

use crate::{ProbeDispatchState, process_guest_range_is_readable};

const IT_NOP: u32 = 0x10;
const IT_INDIRECT_BUFFER: u32 = 0x3f;
const IT_REWIND: u32 = 0x59;

const R_ZERO: u32 = 0x00;
const R_WAIT_FLIP_DONE: u32 = 0x06;
const R_PUSH_MARKER: u32 = 0x0b;
const R_FLIP: u32 = 0x17;

const COMMAND_BUFFER_CURSOR_UP_OFFSET: u64 = 0x10;
const COMMAND_BUFFER_CURSOR_DOWN_OFFSET: u64 = 0x18;
const COMMAND_BUFFER_CALLBACK_OFFSET: u64 = 0x20;
const COMMAND_BUFFER_USER_DATA_OFFSET: u64 = 0x28;
const COMMAND_BUFFER_RESERVED_DWORDS_OFFSET: u64 = 0x30;
const COMMAND_BUFFER_DESCRIPTOR_BYTES: u64 = 0x34;

const MAX_PACKET_DWORDS: u32 = 0x4001;

const ATOMIC_MEM_PACKET_BYTES: u64 = 36;
const COPY_DATA_PACKET_BYTES: u64 = 24;
const DISPATCH_INDIRECT_PACKET_BYTES: u64 = 16;
const EVENT_WRITE_PACKET_BYTES: u64 = 8;
const JUMP_PACKET_BYTES: u64 = 16;
const REWIND_PACKET_BYTES: u64 = 8;
const END_OF_SHADER_ACTION_PACKET_BYTES: u64 = 32;
const WAIT_ON_ADDRESS_32_PACKET_BYTES: u64 = 28;
const WAIT_ON_ADDRESS_64_PACKET_BYTES: u64 = 36;

const JUMP_PACKET_DWORDS: u32 = 4;
const REWIND_PACKET_DWORDS: u32 = 2;
const FLIP_PACKET_DWORDS: u32 = 6;
const WAIT_FLIP_DONE_PACKET_DWORDS: u32 = 7;
const WORKLOADS_ACTIVE_PACKET_DWORDS: u32 = 18;
const WORKLOAD_COMPLETE_PACKET_DWORDS: u32 = 12;

/// Virtual-memory id, chain selector and least-recently-used cache policy of a compute jump.
///
/// A compute queue can only chain, never call, and its jump target is always read through the
/// least-recently-used policy, so every field of this control word except the target length is
/// fixed for the whole entry point.
const JUMP_CHAIN_CONTROL: u32 = 0x0f30_0000;
const MAX_JUMP_DWORDS: u32 = 0x000f_ffff;

const REWIND_POLLING_OFFLOAD_SHIFT: u32 = 30;
const REWIND_VALID_SHIFT: u32 = 31;

const LABEL_SIZE_32: u32 = 0;
const LABEL_SIZE_64: u32 = 1;

const MIN_WORKLOAD_STREAM_ID: u32 = 1;
const MAX_WORKLOAD_STREAM_ID: u32 = 31;
const MAX_WORKLOAD_ID: u32 = 63;
const MAX_WORKLOAD_IDS: u64 = 64;
const WORKLOAD_ID_BYTES: u64 = 4;

#[cfg(all(windows, target_arch = "x86_64"))]
type GrowCallback = unsafe extern "sysv64" fn(u64, u32, u64) -> bool;
#[cfg(not(all(windows, target_arch = "x86_64")))]
type GrowCallback = unsafe extern "C" fn(u64, u32, u64) -> bool;

/// Handles a NID belonging to this module, or returns `None` if it owns no such NID.
pub(crate) fn dispatch(
    nid: &str,
    arguments: [u64; 6],
    _state: &mut ProbeDispatchState,
) -> Option<u64> {
    dispatch_builder(nid, arguments).or_else(|| dispatch_packet_size(nid, arguments))
}

fn dispatch_builder(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "e1DFTg+Sd8U" => jump(arguments),
        "DwICrVxerkY" => rewind(arguments),
        "ebixW91gpPw" => set_flip(arguments),
        "GPbUp9jXQa8" => wait_until_safe_for_rendering(arguments),
        // Both marker entry points annotate a profiler capture and are consumed by the same
        // reader, so they have to agree on a payload layout with the builders that take a
        // nul-terminated string.
        "8Kly1JrJUlw" | "pxx-GoOSdw4" => marker_span(arguments),
        "rVOmPz2RBlg" => set_workloads_active(arguments),
        "opR1JeJZCBU" => set_workload_complete(arguments),
        _ => return None,
    })
}

fn dispatch_packet_size(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "da1Sm8-QDoU" => ATOMIC_MEM_PACKET_BYTES,
        "CbQh3DKMSno" => COPY_DATA_PACKET_BYTES,
        "PxKWV2fVAps" => DISPATCH_INDIRECT_PACKET_BYTES,
        // Every event a compute queue can signal is one of the four that carry no address, so this
        // query has no argument and never reports the larger addressed form.
        "Y-5vneiBtzk" => EVENT_WRITE_PACKET_BYTES,
        "b-oySn+G2tE" => JUMP_PACKET_BYTES,
        "0ZOG0jc9nRg" => REWIND_PACKET_BYTES,
        // A shader-stage end action is queued through the same release primitive that ends a
        // pipeline stage, which selects its event index from the action code alone.
        "F8NLhWvFemI" => END_OF_SHADER_ACTION_PACKET_BYTES,
        "idlaArvdXEs" => wait_on_address_size(arguments[0]),
        _ => return None,
    })
}

/// Returns the byte size of the packet that polls a label of the requested width, or zero if no
/// label has that width.
fn wait_on_address_size(size: u64) -> u64 {
    match low_u8(size) {
        LABEL_SIZE_32 => WAIT_ON_ADDRESS_32_PACKET_BYTES,
        LABEL_SIZE_64 => WAIT_ON_ADDRESS_64_PACKET_BYTES,
        _ => 0,
    }
}

fn jump(arguments: [u64; 6]) -> u64 {
    let target = arguments[2];
    let dword_count = low_u32(arguments[3]);
    // The command processor faults on a zero-length chain target, so an empty stream is refused
    // here rather than written into the queue.
    if target == 0 || target & 3 != 0 || dword_count == 0 || dword_count > MAX_JUMP_DWORDS {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], JUMP_PACKET_DWORDS) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(JUMP_PACKET_DWORDS, IT_INDIRECT_BUFFER, R_ZERO),
            low_u32(target) & !3,
            low_u32(target >> 32),
            JUMP_CHAIN_CONTROL | dword_count,
        ],
    );
    command
}

fn rewind(arguments: [u64; 6]) -> u64 {
    let initial_state = low_u8(arguments[1]);
    let offload = low_u8(arguments[2]);
    if initial_state > 1 || offload > 1 {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], REWIND_PACKET_DWORDS) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(REWIND_PACKET_DWORDS, IT_REWIND, R_ZERO),
            (initial_state << REWIND_VALID_SHIFT) | (offload << REWIND_POLLING_OFFLOAD_SHIFT),
        ],
    );
    command
}

fn set_flip(arguments: [u64; 6]) -> u64 {
    let Some(command) = allocate_dwords(arguments[0], FLIP_PACKET_DWORDS) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(FLIP_PACKET_DWORDS, IT_NOP, R_FLIP),
            low_u32(arguments[1]),
            low_u32(arguments[2]),
            low_u32(arguments[3]),
            low_u32(arguments[4]),
            low_u32(arguments[4] >> 32),
        ],
    );
    command
}

fn wait_until_safe_for_rendering(arguments: [u64; 6]) -> u64 {
    let Some(command) = allocate_dwords(arguments[0], WAIT_FLIP_DONE_PACKET_DWORDS) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(WAIT_FLIP_DONE_PACKET_DWORDS, IT_NOP, R_WAIT_FLIP_DONE),
            low_u32(arguments[1]),
            low_u32(arguments[2]),
            0,
            0,
            0,
            0,
        ],
    );
    command
}

fn marker_span(arguments: [u64; 6]) -> u64 {
    let text = arguments[1];
    let byte_count = arguments[2];
    // The payload reserves a byte past the copied span. A span whose length is an exact multiple
    // of four would otherwise leave the marker text running into the next packet's header when a
    // capture tool reads it as a string.
    let Some(payload_bytes) = byte_count.checked_add(1) else {
        return 0;
    };
    if payload_bytes > u64::from(MAX_PACKET_DWORDS - 1) * 4 {
        return 0;
    }
    if byte_count != 0 && (text == 0 || !guest_range_is_readable(text, byte_count)) {
        return 0;
    }
    let Ok(payload_dwords) = u32::try_from(payload_bytes.div_ceil(4)) else {
        return 0;
    };
    let count = payload_dwords + 1;
    let Some(command) = allocate_dwords(arguments[0], count) else {
        return 0;
    };
    write_u32(command, pm4(count, IT_NOP, R_PUSH_MARKER));
    for index in 1..u64::from(count) {
        write_u32(command + index * 4, 0);
    }
    for index in 0..byte_count {
        write_u8(command + 4 + index, read_u8(text + index));
    }
    command
}

/// Reports whether a workload stream id is one the debug layer is able to register.
fn is_workload_stream(stream_id: u32) -> bool {
    (MIN_WORKLOAD_STREAM_ID..=MAX_WORKLOAD_STREAM_ID).contains(&stream_id)
}

fn set_workloads_active(arguments: [u64; 6]) -> u64 {
    let stream_id = low_u32(arguments[1]);
    let workload_ids = arguments[2];
    let id_count = arguments[3];
    if !is_workload_stream(stream_id) || id_count == 0 || id_count > MAX_WORKLOAD_IDS {
        return 0;
    }
    let byte_count = id_count * WORKLOAD_ID_BYTES;
    if workload_ids == 0
        || workload_ids & 3 != 0
        || !guest_range_is_readable(workload_ids, byte_count)
    {
        return 0;
    }
    let mut workload_mask = 0_u64;
    for index in 0..id_count {
        let workload_id = read_u32(workload_ids + index * WORKLOAD_ID_BYTES);
        if workload_id > MAX_WORKLOAD_ID {
            return 0;
        }
        let bit = 1_u64 << workload_id;
        // A repeated id would be marked complete by the first matching completion packet, leaving
        // the rest of the stream's work attributed to a workload that already ended.
        if workload_mask & bit != 0 {
            return 0;
        }
        workload_mask |= bit;
    }
    let Some(command) = allocate_dwords(arguments[0], WORKLOADS_ACTIVE_PACKET_DWORDS) else {
        return 0;
    };
    write_padded_words(
        command,
        WORKLOADS_ACTIVE_PACKET_DWORDS,
        &[
            pm4(WORKLOADS_ACTIVE_PACKET_DWORDS, IT_NOP, R_ZERO),
            stream_id,
            low_u32(workload_mask),
            low_u32(workload_mask >> 32),
        ],
    );
    command
}

fn set_workload_complete(arguments: [u64; 6]) -> u64 {
    let stream_id = low_u32(arguments[1]);
    let workload_id = low_u32(arguments[2]);
    if !is_workload_stream(stream_id) || workload_id > MAX_WORKLOAD_ID {
        return 0;
    }
    let clear_mask = !(1_u64 << workload_id);
    let Some(command) = allocate_dwords(arguments[0], WORKLOAD_COMPLETE_PACKET_DWORDS) else {
        return 0;
    };
    write_padded_words(
        command,
        WORKLOAD_COMPLETE_PACKET_DWORDS,
        &[
            pm4(WORKLOAD_COMPLETE_PACKET_DWORDS, IT_NOP, R_ZERO),
            stream_id,
            workload_id,
            low_u32(clear_mask),
            low_u32(clear_mask >> 32),
        ],
    );
    command
}

fn low_u32(value: u64) -> u32 {
    u32::try_from(value & u64::from(u32::MAX)).expect("value is masked to 32 bits")
}

fn low_u8(value: u64) -> u32 {
    low_u32(value) & 0xff
}

fn pm4(length_dwords: u32, opcode: u32, register: u32) -> u32 {
    0xc000_0000
        | ((length_dwords.wrapping_sub(2) & 0x3fff) << 16)
        | ((opcode & 0xff) << 8)
        | ((register & 0x3f) << 2)
}

fn guest_range_is_readable(address: u64, length: u64) -> bool {
    if address == 0 || address.checked_add(length).is_none() {
        return false;
    }
    usize::try_from(length).is_ok_and(|length| process_guest_range_is_readable(address, length))
}

fn read_u8(address: u64) -> u8 {
    // SAFETY: the caller proved this range readable before any access.
    unsafe { std::ptr::read_unaligned(address as *const u8) }
}

fn read_u32(address: u64) -> u32 {
    // SAFETY: the caller proved this range readable before any access.
    unsafe { std::ptr::read_unaligned(address as *const u32) }
}

fn read_u64(address: u64) -> u64 {
    // SAFETY: the caller proved this range readable before any access.
    unsafe { std::ptr::read_unaligned(address as *const u64) }
}

fn write_u8(address: u64, value: u8) {
    // SAFETY: the packet range was proved mapped when the command buffer allocated it.
    unsafe { std::ptr::write_unaligned(address as *mut u8, value) }
}

fn write_u32(address: u64, value: u32) {
    // SAFETY: the packet range was proved mapped when the command buffer allocated it.
    unsafe { std::ptr::write_unaligned(address as *mut u32, value) }
}

fn write_u64(address: u64, value: u64) {
    // SAFETY: the descriptor was proved mapped before the cursor was read.
    unsafe { std::ptr::write_unaligned(address as *mut u64, value) }
}

fn write_words(address: u64, words: &[u32]) {
    for (index, &word) in words.iter().enumerate() {
        let index = u64::try_from(index).expect("word index fits");
        write_u32(address + index * 4, word);
    }
}

/// Writes `words` at the head of a packet and clears the rest of its `dword_count` dwords.
fn write_padded_words(address: u64, dword_count: u32, words: &[u32]) {
    write_words(address, words);
    for index in u64::try_from(words.len()).expect("word count fits")..u64::from(dword_count) {
        write_u32(address + index * 4, 0);
    }
}

/// Reserves `dword_count` dwords at the head of a command buffer, growing it if the guest supplied
/// a callback, and returns the address the packet must be written to.
fn allocate_dwords(command_buffer: u64, dword_count: u32) -> Option<u64> {
    use crate::agc_command_abi::decline_reservation;
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
    if !has_capacity(cursor_up, cursor_down, reserved, dword_count) {
        let callback = read_u64(command_buffer + COMMAND_BUFFER_CALLBACK_OFFSET);
        if callback == 0 {
            return decline_reservation("full-without-grow-callback", command_buffer, dword_count);
        }
        let requested = dword_count.checked_add(reserved)?;
        let user_data = read_u64(command_buffer + COMMAND_BUFFER_USER_DATA_OFFSET);
        let callback_address = usize::try_from(callback).ok()?;
        // SAFETY: the descriptor field this address came from is the guest's own SysV growth
        // callback, whose signature the Gen5 command-buffer layout fixes.
        let callback: GrowCallback = unsafe { std::mem::transmute(callback_address) };
        // SAFETY: the callback owns the cursors it rewrites and reports whether it grew the buffer.
        if !unsafe { callback(command_buffer, requested, user_data) } {
            return decline_reservation("grow-refused", command_buffer, dword_count);
        }
        cursor_up = read_u64(command_buffer + COMMAND_BUFFER_CURSOR_UP_OFFSET);
        cursor_down = read_u64(command_buffer + COMMAND_BUFFER_CURSOR_DOWN_OFFSET);
        if !has_capacity(cursor_up, cursor_down, reserved, dword_count) {
            return decline_reservation("short-after-grow", command_buffer, dword_count);
        }
    }
    let byte_count = u64::from(dword_count).checked_mul(4)?;
    let next = cursor_up.checked_add(byte_count)?;
    if !guest_range_is_readable(cursor_up, byte_count) {
        return None;
    }
    write_u64(command_buffer + COMMAND_BUFFER_CURSOR_UP_OFFSET, next);
    Some(cursor_up)
}

fn has_capacity(cursor_up: u64, cursor_down: u64, reserved: u32, needed: u32) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    struct CommandFixture {
        descriptor: Box<[u64; 7]>,
        words: Box<[u32; 128]>,
    }

    impl CommandFixture {
        fn new() -> Self {
            let mut fixture = Self {
                descriptor: Box::new([0_u64; 7]),
                words: Box::new([0_u32; 128]),
            };
            let start = fixture.words.as_mut_ptr().addr() as u64;
            let end = start + fixture.words.len() as u64 * 4;
            fixture.descriptor[0] = start;
            fixture.descriptor[1] = end;
            fixture.descriptor[2] = start;
            fixture.descriptor[3] = end;
            fixture
        }

        fn address(&mut self) -> u64 {
            self.descriptor.as_mut_ptr().addr() as u64
        }

        fn cursor(&self) -> u64 {
            self.descriptor[2]
        }

        fn start(&self) -> u64 {
            self.words.as_ptr().addr() as u64
        }
    }

    fn args(values: &[u64]) -> [u64; 6] {
        let mut result = [0_u64; 6];
        result[..values.len()].copy_from_slice(values);
        result
    }

    fn call(nid: &str, arguments: [u64; 6]) -> u64 {
        let mut state = ProbeDispatchState::default();
        dispatch(nid, arguments, &mut state).expect("known test NID")
    }

    fn payload_bytes(command: u64, length: usize) -> Vec<u8> {
        (0..length)
            .map(|index| read_u8(command + 4 + index as u64))
            .collect()
    }

    #[test]
    fn reports_the_byte_size_of_every_fixed_size_compute_packet() {
        assert_eq!(call("da1Sm8-QDoU", [0; 6]), 36);
        assert_eq!(call("CbQh3DKMSno", [0; 6]), 24);
        assert_eq!(call("PxKWV2fVAps", [0; 6]), 16);
        assert_eq!(call("Y-5vneiBtzk", [0; 6]), 8);
        assert_eq!(call("b-oySn+G2tE", [0; 6]), 16);
        assert_eq!(call("0ZOG0jc9nRg", [0; 6]), 8);
        assert_eq!(call("F8NLhWvFemI", [0; 6]), 32);
    }

    #[test]
    fn a_label_wait_reports_the_size_of_the_packet_that_polls_that_label_width() {
        assert_eq!(call("idlaArvdXEs", args(&[0])), 28);
        assert_eq!(call("idlaArvdXEs", args(&[1])), 36);
        assert_eq!(call("idlaArvdXEs", args(&[2])), 0);
        assert_eq!(call("idlaArvdXEs", args(&[0xff])), 0);
    }

    #[test]
    fn every_reported_size_matches_the_packet_the_matching_builder_emits() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();
        let start = fixture.start();

        assert_eq!(
            call("e1DFTg+Sd8U", args(&[buffer, 0, start + 64, 4])),
            start
        );
        assert_eq!(fixture.cursor() - start, call("b-oySn+G2tE", [0; 6]));

        let before = fixture.cursor();
        assert_eq!(call("DwICrVxerkY", args(&[buffer, 0, 0])), before);
        assert_eq!(fixture.cursor() - before, call("0ZOG0jc9nRg", [0; 6]));
    }

    #[test]
    fn builds_a_chaining_jump_packet_that_carries_its_target_and_length() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();
        let target = 0x0000_1234_5678_9ab0_u64;

        let command = call("e1DFTg+Sd8U", args(&[buffer, 0, target, 0x1234]));
        assert_eq!(command, fixture.start());
        assert_eq!(
            &fixture.words[..4],
            &[
                pm4(4, IT_INDIRECT_BUFFER, R_ZERO),
                0x5678_9ab0,
                0x0000_1234,
                0x0f30_1234,
            ]
        );
        assert_eq!(fixture.cursor(), fixture.start() + 16);
    }

    #[test]
    fn a_jump_to_a_null_misaligned_or_empty_target_is_rejected_rather_than_faulting() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();
        let target = 0x0000_1234_5678_9ab0_u64;

        assert_eq!(call("e1DFTg+Sd8U", args(&[buffer, 0, 0, 4])), 0);
        assert_eq!(call("e1DFTg+Sd8U", args(&[buffer, 0, target + 1, 4])), 0);
        assert_eq!(call("e1DFTg+Sd8U", args(&[buffer, 0, target, 0])), 0);
        assert_eq!(
            call("e1DFTg+Sd8U", args(&[buffer, 0, target, 0x0010_0000])),
            0
        );
        assert_eq!(call("e1DFTg+Sd8U", args(&[0, 0, target, 4])), 0);
        assert_eq!(fixture.cursor(), fixture.start());
    }

    #[test]
    fn builds_a_rewind_packet_whose_state_and_polling_offload_occupy_distinct_bits() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();

        assert_eq!(call("DwICrVxerkY", args(&[buffer, 0, 0])), fixture.start());
        assert_eq!(&fixture.words[..2], &[pm4(2, IT_REWIND, R_ZERO), 0]);

        assert_eq!(
            call("DwICrVxerkY", args(&[buffer, 1, 1])),
            fixture.start() + 8
        );
        assert_eq!(fixture.words[3], 0xc000_0000);

        assert_eq!(
            call("DwICrVxerkY", args(&[buffer, 0, 1])),
            fixture.start() + 16
        );
        assert_eq!(fixture.words[5], 0x4000_0000);
        assert_eq!(fixture.cursor(), fixture.start() + 24);
    }

    #[test]
    fn a_rewind_state_or_polling_offload_outside_its_two_values_is_rejected() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();

        assert_eq!(call("DwICrVxerkY", args(&[buffer, 2, 0])), 0);
        assert_eq!(call("DwICrVxerkY", args(&[buffer, 0, 2])), 0);
        assert_eq!(call("DwICrVxerkY", args(&[0, 0, 0])), 0);
        assert_eq!(fixture.cursor(), fixture.start());
    }

    #[test]
    fn builds_the_flip_and_flip_fence_packets_the_submission_walker_reads() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();

        let flip = call(
            "ebixW91gpPw",
            args(&[buffer, 1, 2, 3, 0x1122_3344_5566_7788]),
        );
        assert_eq!(flip, fixture.start());
        assert_eq!(
            &fixture.words[..6],
            &[pm4(6, IT_NOP, R_FLIP), 1, 2, 3, 0x5566_7788, 0x1122_3344,]
        );

        let fence = call("GPbUp9jXQa8", args(&[buffer, 1, 2]));
        assert_eq!(fence, fixture.start() + 24);
        assert_eq!(
            &fixture.words[6..13],
            &[pm4(7, IT_NOP, R_WAIT_FLIP_DONE), 1, 2, 0, 0, 0, 0]
        );
        assert_eq!(fixture.cursor(), fixture.start() + 52);
    }

    #[test]
    fn a_flip_or_flip_fence_without_a_command_buffer_is_rejected_rather_than_faulting() {
        assert_eq!(call("ebixW91gpPw", args(&[0, 1, 2, 3, 4])), 0);
        assert_eq!(call("GPbUp9jXQa8", args(&[0, 1, 2])), 0);
    }

    #[test]
    fn builds_a_marker_packet_that_always_terminates_its_copied_span() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();
        let text = b"pass";

        let command = call(
            "8Kly1JrJUlw",
            args(&[buffer, text.as_ptr().addr() as u64, 4, 0x00ff_8000]),
        );
        assert_eq!(command, fixture.start());
        assert_eq!(fixture.words[0], pm4(3, IT_NOP, R_PUSH_MARKER));
        assert_eq!(payload_bytes(command, 8), b"pass\0\0\0\0");
        assert_eq!(fixture.cursor(), fixture.start() + 12);

        let embedded = b"a\0b";
        let second = call(
            "pxx-GoOSdw4",
            args(&[buffer, embedded.as_ptr().addr() as u64, 3]),
        );
        assert_eq!(second, fixture.start() + 12);
        assert_eq!(fixture.words[3], pm4(2, IT_NOP, R_PUSH_MARKER));
        assert_eq!(payload_bytes(second, 4), b"a\0b\0");
    }

    #[test]
    fn an_empty_marker_span_still_emits_a_two_dword_packet() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();

        let command = call("8Kly1JrJUlw", args(&[buffer, 0, 0]));
        assert_eq!(command, fixture.start());
        assert_eq!(&fixture.words[..2], &[pm4(2, IT_NOP, R_PUSH_MARKER), 0]);
        assert_eq!(fixture.cursor(), fixture.start() + 8);
    }

    #[test]
    fn a_marker_span_that_cannot_be_read_or_cannot_fit_is_rejected() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();
        let text = b"marker";
        let address = text.as_ptr().addr() as u64;

        assert_eq!(call("8Kly1JrJUlw", args(&[buffer, 0, 6])), 0);
        assert_eq!(call("8Kly1JrJUlw", args(&[buffer, address, 0x1_0000])), 0);
        assert_eq!(call("8Kly1JrJUlw", args(&[buffer, address, u64::MAX])), 0);
        assert_eq!(call("8Kly1JrJUlw", args(&[0, address, 6])), 0);
        assert_eq!(fixture.cursor(), fixture.start());
    }

    #[test]
    fn builds_workload_annotations_that_carry_the_stream_and_its_workload_mask() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();
        let workload_ids = [1_u32, 63, 32];

        let active = call(
            "rVOmPz2RBlg",
            args(&[buffer, 7, workload_ids.as_ptr().addr() as u64, 3]),
        );
        assert_eq!(active, fixture.start());
        assert_eq!(
            &fixture.words[..4],
            &[pm4(18, IT_NOP, R_ZERO), 7, 0x0000_0002, 0x8000_0001]
        );
        assert_eq!(&fixture.words[4..18], &[0; 14]);
        assert_eq!(fixture.cursor(), fixture.start() + 72);

        let complete = call("opR1JeJZCBU", args(&[buffer, 7, 63]));
        assert_eq!(complete, fixture.start() + 72);
        assert_eq!(
            &fixture.words[18..23],
            &[pm4(12, IT_NOP, R_ZERO), 7, 63, 0xffff_ffff, 0x7fff_ffff]
        );
        assert_eq!(&fixture.words[23..30], &[0; 7]);
        assert_eq!(fixture.cursor(), fixture.start() + 120);
    }

    #[test]
    fn a_workload_annotation_outside_the_registrable_stream_or_id_range_is_rejected() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();
        let workload_ids = [1_u32, 64];
        let address = workload_ids.as_ptr().addr() as u64;

        assert_eq!(call("rVOmPz2RBlg", args(&[buffer, 0, address, 1])), 0);
        assert_eq!(call("rVOmPz2RBlg", args(&[buffer, 32, address, 1])), 0);
        assert_eq!(call("rVOmPz2RBlg", args(&[buffer, 1, address, 2])), 0);
        assert_eq!(call("rVOmPz2RBlg", args(&[buffer, 1, address, 0])), 0);
        assert_eq!(call("rVOmPz2RBlg", args(&[buffer, 1, address, 65])), 0);
        assert_eq!(call("rVOmPz2RBlg", args(&[buffer, 1, 0, 1])), 0);
        assert_eq!(call("rVOmPz2RBlg", args(&[buffer, 1, address + 1, 1])), 0);
        assert_eq!(call("opR1JeJZCBU", args(&[buffer, 0, 1])), 0);
        assert_eq!(call("opR1JeJZCBU", args(&[buffer, 32, 1])), 0);
        assert_eq!(call("opR1JeJZCBU", args(&[buffer, 1, 64])), 0);
        assert_eq!(call("opR1JeJZCBU", args(&[0, 1, 1])), 0);
        assert_eq!(fixture.cursor(), fixture.start());
    }

    #[test]
    fn a_repeated_workload_id_is_rejected_rather_than_folded_into_one_bit() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();
        let workload_ids = [5_u32, 5];

        assert_eq!(
            call(
                "rVOmPz2RBlg",
                args(&[buffer, 3, workload_ids.as_ptr().addr() as u64, 2])
            ),
            0
        );
        assert_eq!(fixture.cursor(), fixture.start());
    }

    #[test]
    fn a_command_buffer_without_room_or_a_growth_callback_builds_nothing() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();
        let start = fixture.start();

        fixture.descriptor[3] = start + 8;
        assert_eq!(call("ebixW91gpPw", args(&[buffer, 1, 2, 3, 4])), 0);
        assert_eq!(fixture.cursor(), start);

        fixture.descriptor[3] = start + 512;
        fixture.descriptor[6] = 127;
        assert_eq!(call("ebixW91gpPw", args(&[buffer, 1, 2, 3, 4])), 0);
        assert_eq!(fixture.cursor(), start);

        assert_eq!(call("ebixW91gpPw", args(&[buffer + 4, 1, 2, 3, 4])), 0);
    }

    #[test]
    fn a_full_command_buffer_is_grown_through_the_guest_callback() {
        guest_abi_function! {
            unsafe fn grow(command_buffer: u64, requested: u32, user_data: u64) -> bool {
                if requested != 2 {
                    return false;
                }
                write_u64(
                    command_buffer + COMMAND_BUFFER_CURSOR_UP_OFFSET,
                    read_u64(user_data),
                );
                write_u64(
                    command_buffer + COMMAND_BUFFER_CURSOR_DOWN_OFFSET,
                    read_u64(user_data + 8),
                );
                true
            }
        }

        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();
        let start = fixture.start();
        let extent = Box::new([start, start + 32]);
        fixture.descriptor[3] = start + 4;
        fixture.descriptor[4] = grow as *const () as usize as u64;
        fixture.descriptor[5] = extent.as_ptr().addr() as u64;

        assert_eq!(call("DwICrVxerkY", args(&[buffer, 1, 0])), start);
        assert_eq!(&fixture.words[..2], &[pm4(2, IT_REWIND, R_ZERO), 1 << 31]);
        assert_eq!(fixture.cursor(), start + 8);
    }

    #[test]
    fn nids_this_module_does_not_own_are_left_for_the_other_modules() {
        let mut state = ProbeDispatchState::default();
        for nid in [
            "not-an-agc-nid",
            // sceAgcAcbAtomicMem, whose comparison data and poll interval arrive past the six
            // registers this dispatcher receives.
            "XKKuA6VkSRc",
            "cduV1f0dcGQ",
            "szG7hz2yEhA",
            "q4VuU-QsLOE",
            "FcgdDM3MB+k",
            "N7M3+N-sBCw",
            "MDLD5Ly94Xk",
        ] {
            assert_eq!(dispatch(nid, [0; 6], &mut state), None);
        }
    }
}
