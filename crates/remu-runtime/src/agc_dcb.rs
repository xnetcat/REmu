// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Agc draw and copy command-buffer builders.
//!
//! This module implements sceAgcDcb* and sceAgcCb* entry points, which append PM4 packets to a guest-owned draw command buffer.
//!
//! A size query and the builder it names are two halves of one contract: the guest reserves
//! command-buffer space with the query and then fills that space with the builder, so a size
//! reported here has to match the packet the runtime actually emits for that call. Several size
//! queries have no builder export of their own, because the typed operation they size is issued
//! through a more general primitive; those queries report that primitive's packet.
//!
//! [`dispatch`] is consulted by the runtime's import dispatcher only after the main table
//! misses, and returns `None` for any NID it does not own so the caller can keep looking.

use crate::{ProbeDispatchState, process_guest_range_is_readable};

const IT_NOP: u32 = 0x10;
const IT_SET_UCONFIG_REG: u32 = 0x79;

const R_PUSH_MARKER: u32 = 0x0b;

const COMMAND_BUFFER_CURSOR_UP_OFFSET: u64 = 0x10;
const COMMAND_BUFFER_CURSOR_DOWN_OFFSET: u64 = 0x18;
const COMMAND_BUFFER_CALLBACK_OFFSET: u64 = 0x20;
const COMMAND_BUFFER_USER_DATA_OFFSET: u64 = 0x28;
const COMMAND_BUFFER_RESERVED_DWORDS_OFFSET: u64 = 0x30;
const COMMAND_BUFFER_DESCRIPTOR_BYTES: u64 = 0x34;

const MAX_PACKET_DWORDS: u32 = 0x4001;
const MAX_REGISTER_OFFSET: u32 = 0xffff;
const MAX_REGISTER_RANGE_VALUES: u32 = 0x3fff;
const MAX_REGISTER_ARRAY_ENTRIES: u32 = 4_096;
const REGISTER_ARRAY_ENTRY_BYTES: u64 = 8;

const MAX_EVENT_TYPE: u32 = 0x3f;
/// The two occlusion-counter events carry a destination address; every other event type does not.
const ADDRESSED_EVENT_TYPE_GROUP: u32 = 0x38;

const BRANCH_PACKET_BYTES: u64 = 56;
const RELEASE_MEM_PACKET_BYTES: u64 = 32;
const COPY_DATA_PACKET_BYTES: u64 = 24;
const DRAW_INDEX_INDIRECT_MULTI_PACKET_BYTES: u64 = 40;
const SET_BASE_PACKET_BYTES: u64 = 16;
const SET_PREDICATION_PACKET_BYTES: u64 = 16;
const REGISTERS_INDIRECT_PACKET_BYTES: u64 = 20;
const REGISTER_DIRECT_PACKET_BYTES: u64 = 12;
const SET_INDEX_SIZE_PACKET_BYTES: u64 = 12;
const EVENT_WRITE_PACKET_BYTES: u64 = 8;
const ADDRESSED_EVENT_WRITE_PACKET_BYTES: u64 = 16;
/// A register array is grouped into runs of consecutive offsets, so its worst case is one
/// three-dword packet per entry.
const REGISTER_ARRAY_WORST_CASE_BYTES: u64 = 12;
/// A range packet holds an opcode dword and a first-offset dword ahead of its values.
const REGISTER_RANGE_HEADER_BYTES: u64 = 8;

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
        "MDLD5Ly94Xk" => set_uc_register_range_direct(arguments),
        "03RZmELWWzw" => set_uc_registers_direct(arguments),
        // Both marker entry points annotate a profiler capture and are consumed by the same
        // reader, so they have to agree on a payload layout with the builders that take a
        // nul-terminated string.
        "N7M3+N-sBCw" | "BYcSvEsINWU" => marker_span(arguments),
        _ => return None,
    })
}

fn dispatch_packet_size(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "uZW-mqsxkrM" => BRANCH_PACKET_BYTES,
        // An end-of-shader action is queued through the same end-of-pipe primitive, which selects
        // the shader-stage event index from the action code alone.
        "zg6u-N6Otxs" => RELEASE_MEM_PACKET_BYTES,
        "b5u0Jzm8TF8" => COPY_DATA_PACKET_BYTES,
        "r98I08t+LOg" => DRAW_INDEX_INDIRECT_MULTI_PACKET_BYTES,
        "9S4noWrUI0s" | "MMlmJAL7N5w" => SET_BASE_PACKET_BYTES,
        "yheJGN-ay+A" | "vLrBL8DQiz8" | "XN+Iuu7XsM8" => SET_PREDICATION_PACKET_BYTES,
        "GBCh3zCihoU" | "nNlUtdDDvZ0" | "UQGTw4xRlcM" => REGISTERS_INDIRECT_PACKET_BYTES,
        "QhPDD513V0w" | "aP1Ki9G3++4" => REGISTER_DIRECT_PACKET_BYTES,
        "ca4KPvp0qLQ" => SET_INDEX_SIZE_PACKET_BYTES,
        "JOWmDrl+j20" => register_range_direct_size(arguments[0]),
        "yUBESvCCJ4I" | "TGEZzUWLbrc" => registers_direct_size(arguments[0]),
        "C4l9fB17t8w" => event_write_size(arguments[0]),
        _ => return None,
    })
}

/// Returns the byte size of the single range packet that carries `count` register values.
fn register_range_direct_size(count: u64) -> u64 {
    u64::from(low_u32(count)) * 4 + REGISTER_RANGE_HEADER_BYTES
}

/// Returns the byte size that `count` individually addressed registers can never exceed.
fn registers_direct_size(count: u64) -> u64 {
    u64::from(low_u32(count)) * REGISTER_ARRAY_WORST_CASE_BYTES
}

/// Returns the byte size of the packet that signals `event_type`, or zero if no such event exists.
fn event_write_size(event_type: u64) -> u64 {
    let event_type = low_u8(event_type);
    if event_type > MAX_EVENT_TYPE {
        return 0;
    }
    if event_type & !1 == ADDRESSED_EVENT_TYPE_GROUP {
        ADDRESSED_EVENT_WRITE_PACKET_BYTES
    } else {
        EVENT_WRITE_PACKET_BYTES
    }
}

fn set_uc_register_range_direct(arguments: [u64; 6]) -> u64 {
    let first_register = low_u32(arguments[1]);
    let values = arguments[2];
    let value_count = low_u32(arguments[3]);
    if value_count == 0
        || value_count > MAX_REGISTER_RANGE_VALUES
        || first_register > MAX_REGISTER_OFFSET
        || (values != 0
            && (values & 3 != 0 || !guest_range_is_readable(values, u64::from(value_count) * 4)))
    {
        return 0;
    }
    let packet_count = value_count + 2;
    let Some(command) = allocate_dwords(arguments[0], packet_count) else {
        return 0;
    };
    write_u32(command, pm4(packet_count, IT_SET_UCONFIG_REG, 0));
    write_u32(command + 4, first_register);
    // A null values pointer reserves an inline data packet whose payload the caller fills after
    // `sceAgcGetDataPacketPayloadAddress`. This is a first-class ABI form, not an error.
    if values != 0 {
        for index in 0..u64::from(value_count) {
            write_u32(command + 8 + index * 4, read_u32(values + index * 4));
        }
    }
    command
}

fn set_uc_registers_direct(arguments: [u64; 6]) -> u64 {
    let command_buffer = arguments[0];
    let registers = arguments[1];
    let register_count = low_u32(arguments[2]);
    if registers == 0
        || registers & 3 != 0
        || register_count == 0
        || register_count > MAX_REGISTER_ARRAY_ENTRIES
        || !guest_range_is_readable(
            registers,
            u64::from(register_count) * REGISTER_ARRAY_ENTRY_BYTES,
        )
    {
        return 0;
    }

    let mut first_command = 0;
    let mut run_start = 0;
    while run_start < register_count {
        let first_offset = read_u32(registers + u64::from(run_start) * REGISTER_ARRAY_ENTRY_BYTES);
        if first_offset > MAX_REGISTER_OFFSET {
            return first_command;
        }
        let mut run_end = run_start + 1;
        let mut previous = first_offset;
        while run_end < register_count {
            let next = read_u32(registers + u64::from(run_end) * REGISTER_ARRAY_ENTRY_BYTES);
            if next != previous.wrapping_add(1) {
                break;
            }
            previous = next;
            run_end += 1;
        }
        let value_count = run_end - run_start;
        let packet_count = value_count + 2;
        let Some(command) = allocate_dwords(command_buffer, packet_count) else {
            return first_command;
        };
        if first_command == 0 {
            first_command = command;
        }
        write_u32(command, pm4(packet_count, IT_SET_UCONFIG_REG, 0));
        write_u32(command + 4, first_offset);
        for index in 0..u64::from(value_count) {
            write_u32(
                command + 8 + index * 4,
                read_u32(
                    registers + (u64::from(run_start) + index) * REGISTER_ARRAY_ENTRY_BYTES + 4,
                ),
            );
        }
        run_start = run_end;
    }
    first_command
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
        return decline_reservation("unreadable-packet-range", command_buffer, dword_count);
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
    fn reports_the_byte_size_of_every_fixed_size_packet() {
        assert_eq!(call("uZW-mqsxkrM", [0; 6]), 56);
        assert_eq!(call("zg6u-N6Otxs", [0; 6]), 32);
        assert_eq!(call("b5u0Jzm8TF8", [0; 6]), 24);
        assert_eq!(call("r98I08t+LOg", [0; 6]), 40);
        assert_eq!(call("9S4noWrUI0s", [0; 6]), 16);
        assert_eq!(call("MMlmJAL7N5w", [0; 6]), 16);
        assert_eq!(call("yheJGN-ay+A", [0; 6]), 16);
        assert_eq!(call("vLrBL8DQiz8", [0; 6]), 16);
        assert_eq!(call("XN+Iuu7XsM8", [0; 6]), 16);
        assert_eq!(call("GBCh3zCihoU", [0; 6]), 20);
        assert_eq!(call("nNlUtdDDvZ0", [0; 6]), 20);
        assert_eq!(call("UQGTw4xRlcM", [0; 6]), 20);
        assert_eq!(call("QhPDD513V0w", [0; 6]), 12);
        assert_eq!(call("aP1Ki9G3++4", [0; 6]), 12);
        assert_eq!(call("ca4KPvp0qLQ", [0; 6]), 12);
    }

    #[test]
    fn register_packet_sizes_scale_with_the_requested_register_count() {
        assert_eq!(call("JOWmDrl+j20", args(&[0])), 8);
        assert_eq!(call("JOWmDrl+j20", args(&[7])), 36);
        assert_eq!(call("yUBESvCCJ4I", args(&[0])), 0);
        assert_eq!(call("yUBESvCCJ4I", args(&[5])), 60);
        assert_eq!(call("TGEZzUWLbrc", args(&[5])), 60);
        assert_eq!(call("TGEZzUWLbrc", args(&[0xffff_ffff])), 0x0b_ffff_fff4);
    }

    #[test]
    fn an_event_write_is_larger_only_for_the_events_that_carry_an_address() {
        assert_eq!(call("C4l9fB17t8w", args(&[0])), 8);
        assert_eq!(call("C4l9fB17t8w", args(&[0x37])), 8);
        assert_eq!(call("C4l9fB17t8w", args(&[0x38])), 16);
        assert_eq!(call("C4l9fB17t8w", args(&[0x39])), 16);
        assert_eq!(call("C4l9fB17t8w", args(&[0x3a])), 8);
        assert_eq!(call("C4l9fB17t8w", args(&[0x40])), 0);
    }

    #[test]
    fn builds_a_user_config_register_range_packet_from_a_value_array() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();
        let values = [0x1111_2222_u32, 0x3333_4444, 0x5555_6666];
        let command = call(
            "MDLD5Ly94Xk",
            args(&[buffer, 0x2a1, values.as_ptr().addr() as u64, 3]),
        );

        assert_eq!(command, fixture.start());
        assert_eq!(
            &fixture.words[..5],
            &[
                pm4(5, IT_SET_UCONFIG_REG, 0),
                0x2a1,
                0x1111_2222,
                0x3333_4444,
                0x5555_6666,
            ]
        );
        assert_eq!(fixture.cursor(), fixture.start() + 20);
    }

    #[test]
    fn a_null_value_pointer_reserves_the_range_payload_without_writing_it() {
        let mut fixture = CommandFixture::new();
        fixture.words[2..4].copy_from_slice(&[0xaaaa_aaaa, 0xbbbb_bbbb]);
        let buffer = fixture.address();

        assert_eq!(
            call("MDLD5Ly94Xk", args(&[buffer, 0x100, 0, 2])),
            fixture.start()
        );
        assert_eq!(fixture.words[0], pm4(4, IT_SET_UCONFIG_REG, 0));
        assert_eq!(fixture.words[1], 0x100);
        assert_eq!(&fixture.words[2..4], &[0xaaaa_aaaa, 0xbbbb_bbbb]);
        assert_eq!(fixture.cursor(), fixture.start() + 16);
    }

    #[test]
    fn a_malformed_register_range_request_is_rejected_rather_than_faulting() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();
        let values = [0_u32; 4];
        let address = values.as_ptr().addr() as u64;

        assert_eq!(call("MDLD5Ly94Xk", args(&[buffer, 0x100, address, 0])), 0);
        assert_eq!(
            call("MDLD5Ly94Xk", args(&[buffer, 0x100, address, 0x4000])),
            0
        );
        assert_eq!(
            call("MDLD5Ly94Xk", args(&[buffer, 0x1_0000, address, 1])),
            0
        );
        assert_eq!(
            call("MDLD5Ly94Xk", args(&[buffer, 0x100, address + 1, 1])),
            0
        );
        assert_eq!(call("MDLD5Ly94Xk", args(&[0, 0x100, address, 1])), 0);
        assert_eq!(
            call("MDLD5Ly94Xk", args(&[buffer + 4, 0x100, address, 1])),
            0
        );
        assert_eq!(fixture.cursor(), fixture.start());
    }

    #[test]
    fn groups_a_user_config_register_array_into_runs_of_consecutive_offsets() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();
        let registers = [0x30_u32, 0xaaaa, 0x31, 0xbbbb, 0x40, 0xcccc];
        let command = call(
            "03RZmELWWzw",
            args(&[buffer, registers.as_ptr().addr() as u64, 3]),
        );

        assert_eq!(command, fixture.start());
        assert_eq!(
            &fixture.words[..7],
            &[
                pm4(4, IT_SET_UCONFIG_REG, 0),
                0x30,
                0xaaaa,
                0xbbbb,
                pm4(3, IT_SET_UCONFIG_REG, 0),
                0x40,
                0xcccc,
            ]
        );
        assert_eq!(fixture.cursor(), fixture.start() + 28);
    }

    #[test]
    fn a_null_or_misaligned_register_array_is_rejected_rather_than_faulting() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();
        let registers = [0_u32; 4];
        let address = registers.as_ptr().addr() as u64;

        assert_eq!(call("03RZmELWWzw", args(&[buffer, 0, 2])), 0);
        assert_eq!(call("03RZmELWWzw", args(&[buffer, address + 1, 2])), 0);
        assert_eq!(call("03RZmELWWzw", args(&[buffer, address, 0])), 0);
        assert_eq!(call("03RZmELWWzw", args(&[buffer, address, 4_097])), 0);
        assert_eq!(fixture.cursor(), fixture.start());
    }

    #[test]
    fn builds_a_marker_packet_that_always_terminates_its_copied_span() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();
        let text = b"pass";
        let command = call(
            "N7M3+N-sBCw",
            args(&[buffer, text.as_ptr().addr() as u64, 4, 0x00ff_8000]),
        );

        assert_eq!(command, fixture.start());
        assert_eq!(fixture.words[0], pm4(3, IT_NOP, R_PUSH_MARKER));
        assert_eq!(payload_bytes(command, 8), b"pass\0\0\0\0");
        assert_eq!(fixture.cursor(), fixture.start() + 12);

        let embedded = b"a\0b";
        let second = call(
            "BYcSvEsINWU",
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

        let command = call("N7M3+N-sBCw", args(&[buffer, 0, 0]));
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

        assert_eq!(call("N7M3+N-sBCw", args(&[buffer, 0, 6])), 0);
        assert_eq!(call("N7M3+N-sBCw", args(&[buffer, address, 0x1_0000])), 0);
        assert_eq!(call("N7M3+N-sBCw", args(&[buffer, address, u64::MAX])), 0);
        assert_eq!(call("N7M3+N-sBCw", args(&[0, address, 6])), 0);
        assert_eq!(fixture.cursor(), fixture.start());
    }

    #[test]
    fn a_command_buffer_without_room_or_a_growth_callback_builds_nothing() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();
        let start = fixture.start();
        let values = [0_u32; 4];
        let address = values.as_ptr().addr() as u64;

        fixture.descriptor[3] = start + 8;
        assert_eq!(call("MDLD5Ly94Xk", args(&[buffer, 0x100, address, 4])), 0);
        assert_eq!(fixture.cursor(), start);

        fixture.descriptor[6] = 124;
        fixture.descriptor[3] = start + 512;
        assert_eq!(call("MDLD5Ly94Xk", args(&[buffer, 0x100, address, 4])), 0);
        assert_eq!(fixture.cursor(), start);
    }

    #[test]
    fn a_full_command_buffer_is_grown_through_the_guest_callback() {
        unsafe extern "C" fn grow(command_buffer: u64, requested: u32, user_data: u64) -> bool {
            if requested != 4 {
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

        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();
        let start = fixture.start();
        let extent = Box::new([start, start + 32]);
        fixture.descriptor[3] = start + 4;
        fixture.descriptor[4] = grow as *const () as usize as u64;
        fixture.descriptor[5] = extent.as_ptr().addr() as u64;

        let values = [0x1234_5678_u32, 0x9abc_def0];
        assert_eq!(
            call(
                "MDLD5Ly94Xk",
                args(&[buffer, 0x11, values.as_ptr().addr() as u64, 2])
            ),
            start
        );
        assert_eq!(
            &fixture.words[..4],
            &[
                pm4(4, IT_SET_UCONFIG_REG, 0),
                0x11,
                0x1234_5678,
                0x9abc_def0,
            ]
        );
        assert_eq!(fixture.cursor(), start + 16);
    }

    #[test]
    fn nids_this_module_does_not_own_are_left_for_the_other_modules() {
        let mut state = ProbeDispatchState::default();
        for nid in [
            "not-an-agc-nid",
            "1-gUn1PI4Sw",
            "7toV+elXqNM",
            "G0jrLdvEqDw",
            "jt3pl7EN17o",
            "0o3VDdtA6nM",
            "ms1xVoZ-Vwc",
            "P1CugZ99Uzc",
            "oz6zQq1JwCE",
        ] {
            assert_eq!(dispatch(nid, [0; 6], &mut state), None);
        }
    }
}
