// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Agc register, resource and device state helpers.
//!
//! The entry points here rewrite fields of a packet a command-buffer builder has already emitted,
//! report the shape of such a packet, or answer questions about process-wide Agc state. None of
//! them append a packet of their own, so every offset in this module has to agree with the builder
//! in `agc_command_abi` that produced the packet being patched: a patch that lands on a different
//! dword than the builder wrote silently corrupts an otherwise valid command stream.
//!
//! [`dispatch`] is consulted by the runtime's import dispatcher only after the main table
//! misses, and returns `None` for any NID it does not own so the caller can keep looking.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{ProbeDispatchState, process_guest_range_is_readable, try_read_process_guest_u32};

const AGC_ERROR_ALIGNMENT: u64 = 0x8a6c_0002;
const AGC_ERROR_PARAMETER_NULL: u64 = 0x8a6c_000a;
const AGC_ERROR_PACKET_MISMATCH: u64 = 0x8a6c_000c;
const AGC_ERROR_UNEXPECTED_PARAMETER: u64 = 0x8a6c_003c;
const AGC_ERROR_SEMAPHORE_MEMORY_ALREADY_SET: u64 = 0x8a6c_0048;

const IT_NOP: u32 = 0x10;
const IT_WRITE_DATA: u32 = 0x37;
const IT_INDIRECT_BUFFER: u32 = 0x3f;
const IT_REWIND: u32 = 0x59;
const IT_SET_CONTEXT_REG: u32 = 0x69;
const IT_SET_SH_REG: u32 = 0x76;
const IT_SET_UCONFIG_REG: u32 = 0x79;

const R_ZERO: u32 = 0x00;
const R_ACQUIRE_MEM: u32 = 0x14;

/// A type-3 header stores `dwords - 2`, and the whole field being set marks the one-dword filler
/// the command processor skips without reading a payload.
const PACKET_COUNT_MASK: u32 = 0x3fff;
const UNCOUNTED_PACKET_SELECTOR: u32 = 0x3fff_ff00;
const UNCOUNTED_PACKET_HEADER: u32 = 0x3fff_1000;
const UNCOUNTED_PACKET_DWORDS: u32 = 1;
const PACKET_HEADER_DWORDS: u32 = 2;

const REWIND_PACKET_DWORDS: u32 = 2;
const ACQUIRE_MEM_PACKET_DWORDS: u32 = 8;
const BRANCH_PACKET_DWORDS: u32 = 14;
/// Both write-data patches reach no further than the destination address in dwords 2 and 3.
const WRITE_DATA_PATCHED_DWORDS: u32 = 4;
/// A register-range packet carries an opcode dword and a first-offset dword ahead of its values.
const REGISTER_RANGE_HEADER_DWORDS: u32 = 2;

/// Both rewind builders and the acquire-mem builder place a single-bit selector in the top bit of
/// their first payload dword and pack unrelated fields into the bits below it.
const PAYLOAD_TOP_BIT: u32 = 0x8000_0000;

const BRANCH_THEN_TARGET_OFFSET: u64 = 32;
const BRANCH_ELSE_TARGET_OFFSET: u64 = 44;
const BRANCH_COMPARE_ADDRESS_OFFSET: u64 = 8;
const WRITE_DATA_CONTROL_OFFSET: u64 = 4;
const WRITE_DATA_ADDRESS_OFFSET: u64 = 8;

const WRITE_DATA_CACHE_POLICY_SHIFT: u32 = 25;
const WRITE_DATA_CACHE_POLICY_MASK: u32 = 0x0600_0000;
/// The destination selector is split: its engine bit sits at bit 30 and its space bits at 8..11.
const WRITE_DATA_DST_MASK: u32 = 0x4000_0f00;
const MAX_WRITE_DATA_DST: u32 = 0x1f;

const MAX_CACHE_POLICY: u32 = 3;
const MAX_REWIND_STATE: u32 = 1;
const MAX_ACQUIRE_MEM_ENGINE: u32 = 1;
const MAX_INDIRECT_BUFFER_DWORDS: u32 = 0x000f_ffff;
const INDIRECT_BUFFER_CACHE_POLICY_SHIFT: u32 = 28;

/// A semaphore block must cover whole pages of the address-mapping granularity the AMM allocator
/// hands out, in both its base and its length.
const AMM_SEMAPHORE_MEMORY_ALIGNMENT: u64 = 16_384;

/// Offset of the stage tag inside a compiled shader header, as the Gen5 toolchain emits it.
const SHADER_HEADER_TYPE_OFFSET: u64 = 0x5a;
const SHADER_BINARY_TYPE_GS: u8 = 2;
/// `CxDrawPayloadControl` encodes an enabled primitive payload as bit 3 of the register.
const PRIMITIVE_PAYLOAD_DISABLE: u32 = 0x0;

/// Base address of the process-wide semaphore block, or zero while none has been registered.
static AMM_SEMAPHORE_BASE: AtomicU64 = AtomicU64::new(0);
/// Length in bytes of the registered semaphore block, meaningful only once a base is claimed.
static AMM_SEMAPHORE_SIZE: AtomicU64 = AtomicU64::new(0);

/// Handles a NID belonging to this module, or returns `None` if it owns no such NID.
pub(crate) fn dispatch(
    nid: &str,
    arguments: [u64; 6],
    _state: &mut ProbeDispatchState,
) -> Option<u64> {
    dispatch_packet_patch(nid, arguments).or_else(|| dispatch_device_state(nid, arguments))
}

fn dispatch_packet_patch(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        // The draw and async rewind packets are the same two-dword encoding, so one patch serves
        // both queues.
        "ziVA3whp3p4" | "eWaWyFegzgQ" => rewind_patch_state(arguments),
        "W0WEyog0f74" => acquire_mem_set_engine(arguments),
        "d4NZIlguzv0" => write_data_patch_address(arguments),
        "eAy8eGNsCuU" | "y5K5tPktiL8" => write_data_patch_cache_policy(arguments),
        "tmy-+rBpspY" | "EJBA4dbmvfg" => write_data_patch_dst(arguments),
        "xb8VgcXQhvI" => branch_patch_target(arguments, BRANCH_THEN_TARGET_OFFSET),
        "QmfvaYpsOcI" => branch_patch_target(arguments, BRANCH_ELSE_TARGET_OFFSET),
        "GXBlM-ekzrI" => branch_patch_compare_address(arguments),
        "Lkf86B98qPc" => packet_size(arguments[0]),
        "K2mciNVxUCE" => set_nop(arguments[0]),
        "s+VGAMDQ0AQ" => data_packet_payload_range(arguments),
        _ => return None,
    })
}

fn dispatch_device_state(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "BfBDZGbti7A" => get_is_trinity_mode(arguments[0]),
        "OQTgEXyihvA" => set_amm_semaphore_memory(arguments[0], arguments[1]),
        "ICkECTBxrMw" => get_gs_prim_payload(arguments),
        _ => return None,
    })
}

/// Rewrites the release state of a rewind packet without disturbing its polling-offload bit.
fn rewind_patch_state(arguments: [u64; 6]) -> u64 {
    let packet = arguments[0];
    let state = low_u8(arguments[1]);
    if state > MAX_REWIND_STATE
        || packet_prefix(packet, IT_REWIND, None, REWIND_PACKET_DWORDS).is_none()
    {
        return AGC_ERROR_PACKET_MISMATCH;
    }
    patch_payload_top_bit(packet, state);
    0
}

/// Rewrites which command-processor stage an acquire-mem packet stalls.
fn acquire_mem_set_engine(arguments: [u64; 6]) -> u64 {
    let packet = arguments[0];
    let engine = low_u8(arguments[1]);
    if engine > MAX_ACQUIRE_MEM_ENGINE
        || packet_prefix(
            packet,
            IT_NOP,
            Some(R_ACQUIRE_MEM),
            ACQUIRE_MEM_PACKET_DWORDS,
        )
        .is_none()
    {
        return AGC_ERROR_PACKET_MISMATCH;
    }
    patch_payload_top_bit(packet, engine);
    0
}

fn patch_payload_top_bit(packet: u64, value: u32) {
    let control = read_u32(packet + 4);
    write_u32(packet + 4, (control & !PAYLOAD_TOP_BIT) | (value << 31));
}

/// Rewrites the destination address or offset of a write-data packet.
fn write_data_patch_address(arguments: [u64; 6]) -> u64 {
    let packet = arguments[0];
    let address = arguments[1];
    if address & 3 != 0
        || packet_prefix(packet, IT_WRITE_DATA, None, WRITE_DATA_PATCHED_DWORDS).is_none()
    {
        return AGC_ERROR_PACKET_MISMATCH;
    }
    write_u64(packet + WRITE_DATA_ADDRESS_OFFSET, address);
    0
}

/// Rewrites the cache policy a write-data packet uses for its destination.
fn write_data_patch_cache_policy(arguments: [u64; 6]) -> u64 {
    let packet = arguments[0];
    let cache_policy = low_u8(arguments[1]);
    let Some(control) = write_data_control(packet) else {
        return AGC_ERROR_PACKET_MISMATCH;
    };
    if cache_policy > MAX_CACHE_POLICY {
        return AGC_ERROR_PACKET_MISMATCH;
    }
    write_u32(
        packet + WRITE_DATA_CONTROL_OFFSET,
        (control & !WRITE_DATA_CACHE_POLICY_MASK) | (cache_policy << WRITE_DATA_CACHE_POLICY_SHIFT),
    );
    0
}

/// Rewrites the memory space and engine a write-data packet targets.
fn write_data_patch_dst(arguments: [u64; 6]) -> u64 {
    let packet = arguments[0];
    let destination = low_u8(arguments[1]);
    let Some(control) = write_data_control(packet) else {
        return AGC_ERROR_PACKET_MISMATCH;
    };
    if destination > MAX_WRITE_DATA_DST {
        return AGC_ERROR_PACKET_MISMATCH;
    }
    write_u32(
        packet + WRITE_DATA_CONTROL_OFFSET,
        (control & !WRITE_DATA_DST_MASK) | ((destination & 1) << 30) | ((destination & 0x1e) << 7),
    );
    0
}

fn write_data_control(packet: u64) -> Option<u32> {
    packet_prefix(packet, IT_WRITE_DATA, None, PACKET_HEADER_DWORDS)
        .map(|_| read_u32(packet + WRITE_DATA_CONTROL_OFFSET))
}

/// Rewrites one of the two command buffers a branch packet can hand control to.
fn branch_patch_target(arguments: [u64; 6], offset: u64) -> u64 {
    let packet = arguments[0];
    let cache_policy = low_u8(arguments[1]);
    let buffer = arguments[2];
    let dword_count = low_u32(arguments[3]);
    if cache_policy > MAX_CACHE_POLICY
        || buffer & 3 != 0
        || dword_count > MAX_INDIRECT_BUFFER_DWORDS
        || branch_packet(packet).is_none()
    {
        return AGC_ERROR_PACKET_MISMATCH;
    }
    write_u32(packet + offset, low_u32(buffer) & !3);
    write_u32(packet + offset + 4, low_u32(buffer >> 32));
    write_u32(
        packet + offset + 8,
        dword_count | (cache_policy << INDIRECT_BUFFER_CACHE_POLICY_SHIFT),
    );
    0
}

/// Rewrites the address whose contents a branch packet tests before choosing a target.
fn branch_patch_compare_address(arguments: [u64; 6]) -> u64 {
    let packet = arguments[0];
    let address = arguments[1];
    if address == 0 || address & 7 != 0 || branch_packet(packet).is_none() {
        return AGC_ERROR_PACKET_MISMATCH;
    }
    write_u32(
        packet + BRANCH_COMPARE_ADDRESS_OFFSET,
        low_u32(address) & !7,
    );
    write_u32(
        packet + BRANCH_COMPARE_ADDRESS_OFFSET + 4,
        low_u32(address >> 32),
    );
    0
}

/// Reports the length of a packet in dwords, header included, or zero if it cannot be read.
///
/// The Agc size accessors that count dwords return `uint32_t`; the ones that count bytes return
/// `uint64_t`. This is one of the former, so a caller advances a dword pointer by the result.
fn packet_size(packet: u64) -> u64 {
    let Some(header) = packet_header(packet) else {
        return 0;
    };
    u64::from(packet_dword_count(header))
}

/// Turns an already-written packet into a no-operation of exactly the same length.
fn set_nop(packet: u64) -> u64 {
    let Some(header) = packet_header(packet) else {
        return AGC_ERROR_PACKET_MISMATCH;
    };
    let dword_count = packet_dword_count(header);
    if !packet_is_mapped(packet, dword_count) {
        return AGC_ERROR_PACKET_MISMATCH;
    }
    // The payload of a no-operation is never fetched, so the original dwords are left in place:
    // rewriting them would cost a pass over the packet and erase what a capture would show.
    write_u32(packet, pm4(dword_count, IT_NOP, R_ZERO));
    0
}

/// Writes the base and byte length of the inline payload of a data packet to a memory range.
fn data_packet_payload_range(arguments: [u64; 6]) -> u64 {
    let output = arguments[0];
    let packet = arguments[1];
    let kind = low_u32(arguments[2]);
    if output == 0 || !guest_range_is_readable(output, 16) {
        return AGC_ERROR_PARAMETER_NULL;
    }
    let Some(header) = packet_header(packet) else {
        return AGC_ERROR_PACKET_MISMATCH;
    };
    let opcode = packet_opcode(header);
    let payload_offset = if kind == 0 {
        if opcode != IT_NOP {
            return AGC_ERROR_PACKET_MISMATCH;
        }
        UNCOUNTED_PACKET_DWORDS
    } else {
        if !matches!(
            opcode,
            IT_SET_CONTEXT_REG | IT_SET_SH_REG | IT_SET_UCONFIG_REG
        ) {
            return AGC_ERROR_PACKET_MISMATCH;
        }
        REGISTER_RANGE_HEADER_DWORDS
    };
    // A filler packet declares no payload at all, and the caller distinguishes that from an empty
    // one by the null base rather than by the length alone.
    let (base, size) = match packet_dword_count(header).checked_sub(payload_offset) {
        Some(payload) if payload > 0 => (
            packet + u64::from(payload_offset) * 4,
            u64::from(payload) * 4,
        ),
        _ => (0, 0),
    };
    write_u64(output, base);
    write_u64(output + 8, size);
    0
}

/// Reports whether the Trinity-only feature set is available, which on this runtime it is not.
fn get_is_trinity_mode(output: u64) -> u64 {
    if output == 0 || !guest_range_is_readable(output, 1) {
        return AGC_ERROR_PARAMETER_NULL;
    }
    write_u8(output, 0);
    0
}

/// Registers the caller-owned memory block that backs semaphores addressed by slot index.
fn set_amm_semaphore_memory(base: u64, size_in_bytes: u64) -> u64 {
    if base == 0 {
        return AGC_ERROR_PARAMETER_NULL;
    }
    if !base.is_multiple_of(AMM_SEMAPHORE_MEMORY_ALIGNMENT)
        || !size_in_bytes.is_multiple_of(AMM_SEMAPHORE_MEMORY_ALIGNMENT)
    {
        return AGC_ERROR_ALIGNMENT;
    }
    // The block can never be replaced or resized, so the first caller to claim it wins and every
    // later caller has to be told the registration already happened.
    if AMM_SEMAPHORE_BASE
        .compare_exchange(0, base, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return AGC_ERROR_SEMAPHORE_MEMORY_ALREADY_SET;
    }
    AMM_SEMAPHORE_SIZE.store(size_in_bytes, Ordering::Release);
    0
}

/// Reports the primitive-payload setting a geometry shader needs in `CxDrawPayloadControl`.
///
/// Payload propagation is one of the Trinity-only features, and this runtime reports that feature
/// set as unavailable, so the disabled setting is the only one a caller may install.
fn get_gs_prim_payload(arguments: [u64; 6]) -> u64 {
    let output = arguments[0];
    let shader = arguments[1];
    if shader == 0 {
        return AGC_ERROR_PARAMETER_NULL;
    }
    let Some(stage) = read_guest_u8(shader + SHADER_HEADER_TYPE_OFFSET) else {
        return AGC_ERROR_UNEXPECTED_PARAMETER;
    };
    if stage != SHADER_BINARY_TYPE_GS {
        return AGC_ERROR_UNEXPECTED_PARAMETER;
    }
    if output == 0 || !guest_range_is_readable(output, 4) {
        return AGC_ERROR_PARAMETER_NULL;
    }
    write_u32(output, PRIMITIVE_PAYLOAD_DISABLE);
    0
}

/// Returns the header of a branch packet, rejecting the jump packet that shares its opcode.
fn branch_packet(packet: u64) -> Option<u32> {
    let header = packet_prefix(packet, IT_INDIRECT_BUFFER, None, BRANCH_PACKET_DWORDS)?;
    (packet_dword_count(header) == BRANCH_PACKET_DWORDS).then_some(header)
}

/// Returns the header of `packet` once its kind matches and its first `dword_count` dwords are
/// mapped, so a patch that follows can write them without probing again.
fn packet_prefix(packet: u64, opcode: u32, register: Option<u32>, dword_count: u32) -> Option<u32> {
    let header = packet_header(packet)?;
    if packet_opcode(header) != opcode
        || register.is_some_and(|expected| packet_register(header) != expected)
    {
        return None;
    }
    packet_is_mapped(packet, dword_count).then_some(header)
}

fn packet_header(packet: u64) -> Option<u32> {
    (packet != 0 && packet.is_multiple_of(4))
        .then(|| try_read_process_guest_u32(packet))
        .flatten()
}

fn packet_is_mapped(packet: u64, dword_count: u32) -> bool {
    guest_range_is_readable(packet, u64::from(dword_count) * 4)
}

fn packet_dword_count(header: u32) -> u32 {
    if header & UNCOUNTED_PACKET_SELECTOR == UNCOUNTED_PACKET_HEADER {
        UNCOUNTED_PACKET_DWORDS
    } else {
        ((header >> 16) & PACKET_COUNT_MASK) + PACKET_HEADER_DWORDS
    }
}

fn packet_opcode(header: u32) -> u32 {
    (header >> 8) & 0xff
}

fn packet_register(header: u32) -> u32 {
    (header >> 2) & 0x3f
}

fn pm4(length_dwords: u32, opcode: u32, register: u32) -> u32 {
    0xc000_0000
        | ((length_dwords.wrapping_sub(PACKET_HEADER_DWORDS) & PACKET_COUNT_MASK) << 16)
        | ((opcode & 0xff) << 8)
        | ((register & 0x3f) << 2)
}

fn low_u32(value: u64) -> u32 {
    u32::try_from(value & u64::from(u32::MAX)).expect("value is masked to 32 bits")
}

fn low_u8(value: u64) -> u32 {
    low_u32(value) & 0xff
}

fn guest_range_is_readable(address: u64, length: u64) -> bool {
    if address == 0 || address.checked_add(length).is_none() {
        return false;
    }
    usize::try_from(length).is_ok_and(|length| process_guest_range_is_readable(address, length))
}

fn read_guest_u8(address: u64) -> Option<u8> {
    guest_range_is_readable(address, 1).then(|| read_u8(address))
}

fn read_u8(address: u64) -> u8 {
    // SAFETY: the caller proved this byte readable before any access.
    unsafe { std::ptr::read_unaligned(address as *const u8) }
}

fn read_u32(address: u64) -> u32 {
    // SAFETY: the caller proved this dword readable before any access.
    unsafe { std::ptr::read_unaligned(address as *const u32) }
}

fn write_u8(address: u64, value: u8) {
    // SAFETY: the destination was proved mapped before the patch was accepted.
    unsafe { std::ptr::write_unaligned(address as *mut u8, value) }
}

fn write_u32(address: u64, value: u32) {
    // SAFETY: the destination was proved mapped before the patch was accepted.
    unsafe { std::ptr::write_unaligned(address as *mut u32, value) }
}

fn write_u64(address: u64, value: u64) {
    // SAFETY: the destination was proved mapped before the patch was accepted.
    unsafe { std::ptr::write_unaligned(address as *mut u64, value) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[u64]) -> [u64; 6] {
        let mut result = [0_u64; 6];
        result[..values.len()].copy_from_slice(values);
        result
    }

    fn call(nid: &str, arguments: [u64; 6]) -> u64 {
        let mut state = ProbeDispatchState::default();
        dispatch(nid, arguments, &mut state).expect("known test NID")
    }

    fn address_of(words: &mut [u32]) -> u64 {
        words.as_mut_ptr().addr() as u64
    }

    fn branch_words() -> Vec<u32> {
        let mut words = vec![0_u32; BRANCH_PACKET_DWORDS as usize];
        words[0] = pm4(BRANCH_PACKET_DWORDS, IT_INDIRECT_BUFFER, 0);
        words
    }

    #[test]
    fn a_nid_owned_by_no_handler_here_is_left_for_the_next_module() {
        let mut state = ProbeDispatchState::default();
        assert_eq!(dispatch("not-an-agc-state-nid", [0; 6], &mut state), None);
        // The oversubscription query passes a float in XMM0, which this register-only dispatch
        // path cannot observe, so it must stay unclaimed rather than read a stale integer.
        assert_eq!(dispatch("NKIzURsgV7I", [0; 6], &mut state), None);
        assert_eq!(dispatch("Y3ymLfZ1384", [0; 6], &mut state), None);
        assert_eq!(dispatch("AAeX-U5-P3M", [0; 6], &mut state), None);
    }

    #[test]
    fn patching_a_rewind_state_keeps_the_polling_offload_bit_of_the_async_packet() {
        let mut words = [pm4(REWIND_PACKET_DWORDS, IT_REWIND, 0), 0x4000_0000];
        let packet = address_of(&mut words);

        assert_eq!(call("ziVA3whp3p4", args(&[packet, 1])), 0);
        assert_eq!(words[1], 0xc000_0000);
        assert_eq!(call("eWaWyFegzgQ", args(&[packet, 0])), 0);
        assert_eq!(words[1], 0x4000_0000);
        assert_eq!(words[0], pm4(REWIND_PACKET_DWORDS, IT_REWIND, 0));
    }

    #[test]
    fn a_rewind_patch_rejects_a_foreign_packet_a_bad_state_and_a_null_pointer() {
        let mut words = [pm4(2, IT_NOP, R_ZERO), 0];
        let packet = address_of(&mut words);

        assert_eq!(
            call("ziVA3whp3p4", args(&[packet, 1])),
            AGC_ERROR_PACKET_MISMATCH
        );
        words[0] = pm4(REWIND_PACKET_DWORDS, IT_REWIND, 0);
        assert_eq!(
            call("ziVA3whp3p4", args(&[packet, 2])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(
            call("ziVA3whp3p4", args(&[0, 1])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(
            call("ziVA3whp3p4", args(&[packet + 1, 1])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(words[1], 0);
    }

    #[test]
    fn patching_the_acquire_mem_engine_preserves_the_cache_operation_field() {
        let mut words = [0_u32; ACQUIRE_MEM_PACKET_DWORDS as usize];
        words[0] = pm4(ACQUIRE_MEM_PACKET_DWORDS, IT_NOP, R_ACQUIRE_MEM);
        words[1] = 0x0000_00c3;
        let packet = address_of(&mut words);

        assert_eq!(call("W0WEyog0f74", args(&[packet, 1])), 0);
        assert_eq!(words[1], 0x8000_00c3);
        assert_eq!(call("W0WEyog0f74", args(&[packet, 0])), 0);
        assert_eq!(words[1], 0x0000_00c3);
        assert_eq!(
            call("W0WEyog0f74", args(&[packet, 2])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(
            call("W0WEyog0f74", args(&[0, 1])),
            AGC_ERROR_PACKET_MISMATCH
        );
    }

    #[test]
    fn an_acquire_mem_patch_rejects_a_release_mem_packet_that_shares_its_opcode() {
        let mut words = [0_u32; ACQUIRE_MEM_PACKET_DWORDS as usize];
        words[0] = pm4(ACQUIRE_MEM_PACKET_DWORDS, IT_NOP, 0x18);
        let packet = address_of(&mut words);

        assert_eq!(
            call("W0WEyog0f74", args(&[packet, 1])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(words[1], 0);
    }

    #[test]
    fn write_data_patches_touch_only_the_field_each_one_names() {
        let mut words = [
            pm4(6, IT_WRITE_DATA, 0),
            (1 << 16) | (1 << 20),
            0,
            0,
            0x1111_2222,
            0x3333_4444,
        ];
        let packet = address_of(&mut words);

        assert_eq!(
            call("d4NZIlguzv0", args(&[packet, 0x1234_5678_9abc_def0])),
            0
        );
        assert_eq!(call("eAy8eGNsCuU", args(&[packet, 3])), 0);
        assert_eq!(call("tmy-+rBpspY", args(&[packet, 7])), 0);
        assert_eq!(
            words,
            [
                pm4(6, IT_WRITE_DATA, 0),
                (1 << 16) | (1 << 20) | (3 << 25) | (1 << 30) | (6 << 7),
                0x9abc_def0,
                0x1234_5678,
                0x1111_2222,
                0x3333_4444,
            ]
        );

        assert_eq!(call("y5K5tPktiL8", args(&[packet, 0])), 0);
        assert_eq!(call("EJBA4dbmvfg", args(&[packet, 2])), 0);
        assert_eq!(words[1], (1 << 16) | (1 << 20) | (2 << 7));
    }

    #[test]
    fn a_write_data_patch_rejects_misaligned_addresses_and_out_of_range_selectors() {
        let mut words = [pm4(5, IT_WRITE_DATA, 0), 0, 0, 0, 0];
        let packet = address_of(&mut words);

        assert_eq!(
            call("d4NZIlguzv0", args(&[packet, 0x1002])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(
            call("eAy8eGNsCuU", args(&[packet, 4])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(
            call("tmy-+rBpspY", args(&[packet, 0x20])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(
            call("d4NZIlguzv0", args(&[0, 0x1000])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(&words[1..], &[0, 0, 0, 0]);
    }

    #[test]
    fn branch_targets_and_compare_address_land_on_the_dwords_the_builder_wrote() {
        let mut words = branch_words();
        let packet = address_of(&mut words);

        assert_eq!(
            call(
                "xb8VgcXQhvI",
                args(&[packet, 2, 0x1122_3344_5566_7780, 0x1234])
            ),
            0
        );
        assert_eq!(
            call(
                "QmfvaYpsOcI",
                args(&[packet, 1, 0x00aa_bbcc_ddee_ff00, 0x5678])
            ),
            0
        );
        assert_eq!(
            call("GXBlM-ekzrI", args(&[packet, 0x0000_7fff_0000_1230])),
            0
        );

        assert_eq!(&words[2..4], &[0x0000_1230, 0x0000_7fff]);
        assert_eq!(
            &words[8..11],
            &[0x5566_7780, 0x1122_3344, 0x1234 | (2 << 28)]
        );
        assert_eq!(
            &words[11..14],
            &[0xddee_ff00, 0x00aa_bbcc, 0x5678 | (1 << 28)]
        );
        assert_eq!(&words[4..8], &[0, 0, 0, 0]);
    }

    #[test]
    fn a_branch_patch_rejects_the_shorter_jump_packet_that_shares_its_opcode() {
        let mut jump = [pm4(4, IT_INDIRECT_BUFFER, 0), 0, 0, 0];
        let packet = address_of(&mut jump);

        assert_eq!(
            call("xb8VgcXQhvI", args(&[packet, 0, 0x1000, 4])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(
            call("GXBlM-ekzrI", args(&[packet, 0x1000])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(&jump[1..], &[0, 0, 0]);
    }

    #[test]
    fn a_branch_patch_rejects_bad_policies_sizes_and_compare_addresses() {
        let mut words = branch_words();
        let packet = address_of(&mut words);

        assert_eq!(
            call("xb8VgcXQhvI", args(&[packet, 4, 0x1000, 4])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(
            call("xb8VgcXQhvI", args(&[packet, 0, 0x1002, 4])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(
            call("QmfvaYpsOcI", args(&[packet, 0, 0x1000, 0x0010_0000])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(
            call("GXBlM-ekzrI", args(&[packet, 0])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(
            call("GXBlM-ekzrI", args(&[packet, 0x1004])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(&words[1..], &[0; BRANCH_PACKET_DWORDS as usize - 1]);
    }

    #[test]
    fn packet_size_counts_the_header_dword_and_collapses_the_filler_packet() {
        let mut counted = [pm4(7, IT_WRITE_DATA, 0), 0, 0, 0, 0, 0, 0];
        let mut filler = [pm4(UNCOUNTED_PACKET_DWORDS, IT_NOP, R_ZERO)];

        assert_eq!(call("Lkf86B98qPc", args(&[address_of(&mut counted)])), 7);
        assert_eq!(call("Lkf86B98qPc", args(&[address_of(&mut filler)])), 1);
        assert_eq!(call("Lkf86B98qPc", args(&[0])), 0);
    }

    #[test]
    fn converting_a_packet_to_a_nop_keeps_its_length_and_its_payload() {
        let mut words = [pm4(6, IT_WRITE_DATA, 0), 0xaaaa, 0xbbbb, 0xcccc, 1, 2];
        let packet = address_of(&mut words);

        assert_eq!(call("K2mciNVxUCE", args(&[packet])), 0);
        assert_eq!(words[0], pm4(6, IT_NOP, R_ZERO));
        assert_eq!(&words[1..], &[0xaaaa, 0xbbbb, 0xcccc, 1, 2]);
        assert_eq!(call("Lkf86B98qPc", args(&[packet])), 6);

        let mut filler = [pm4(UNCOUNTED_PACKET_DWORDS, IT_NOP, R_ZERO)];
        let filler_packet = address_of(&mut filler);
        assert_eq!(call("K2mciNVxUCE", args(&[filler_packet])), 0);
        assert_eq!(filler[0], pm4(UNCOUNTED_PACKET_DWORDS, IT_NOP, R_ZERO));
        assert_eq!(call("Lkf86B98qPc", args(&[filler_packet])), 1);

        assert_eq!(call("K2mciNVxUCE", args(&[0])), AGC_ERROR_PACKET_MISMATCH);
        assert_eq!(call("K2mciNVxUCE", args(&[2])), AGC_ERROR_PACKET_MISMATCH);
    }

    #[test]
    fn a_nop_payload_range_covers_every_dword_behind_the_header() {
        let mut words = [pm4(4, IT_NOP, R_ZERO), 1, 2, 3];
        let packet = address_of(&mut words);
        let mut range = [0_u64; 2];
        let output = range.as_mut_ptr().addr() as u64;

        assert_eq!(call("s+VGAMDQ0AQ", args(&[output, packet, 0])), 0);
        assert_eq!(range, [packet + 4, 12]);

        let mut registers = [pm4(5, IT_SET_SH_REG, 0), 0x8c, 1, 2, 3];
        let register_packet = address_of(&mut registers);
        assert_eq!(call("s+VGAMDQ0AQ", args(&[output, register_packet, 1])), 0);
        assert_eq!(range, [register_packet + 8, 12]);
    }

    #[test]
    fn a_filler_packet_reports_an_empty_payload_range() {
        let mut filler = [pm4(UNCOUNTED_PACKET_DWORDS, IT_NOP, R_ZERO)];
        let packet = address_of(&mut filler);
        let mut range = [0xdead_u64; 2];
        let output = range.as_mut_ptr().addr() as u64;

        assert_eq!(call("s+VGAMDQ0AQ", args(&[output, packet, 0])), 0);
        assert_eq!(range, [0, 0]);
    }

    #[test]
    fn a_payload_range_rejects_a_null_output_and_a_packet_of_the_wrong_kind() {
        let mut words = [pm4(4, IT_WRITE_DATA, 0), 0, 0, 0];
        let packet = address_of(&mut words);
        let mut range = [0_u64; 2];
        let output = range.as_mut_ptr().addr() as u64;

        assert_eq!(
            call("s+VGAMDQ0AQ", args(&[0, packet, 0])),
            AGC_ERROR_PARAMETER_NULL
        );
        assert_eq!(
            call("s+VGAMDQ0AQ", args(&[output, packet, 0])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(
            call("s+VGAMDQ0AQ", args(&[output, packet, 1])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(
            call("s+VGAMDQ0AQ", args(&[output, 0, 0])),
            AGC_ERROR_PACKET_MISMATCH
        );
        assert_eq!(range, [0, 0]);
    }

    #[test]
    fn trinity_mode_is_reported_as_unavailable_rather_than_left_uninitialized() {
        let mut flag = [0xff_u8; 4];
        let output = flag.as_mut_ptr().addr() as u64;

        assert_eq!(call("BfBDZGbti7A", args(&[output])), 0);
        assert_eq!(flag, [0, 0xff, 0xff, 0xff]);
        assert_eq!(call("BfBDZGbti7A", args(&[0])), AGC_ERROR_PARAMETER_NULL);
    }

    #[test]
    fn a_geometry_shader_is_told_to_leave_the_primitive_payload_disabled() {
        let stage = usize::try_from(SHADER_HEADER_TYPE_OFFSET).expect("header offset fits");
        let mut geometry = [0_u8; 0x60];
        let mut pixel = [0_u8; 0x60];
        geometry[stage] = SHADER_BINARY_TYPE_GS;
        pixel[stage] = 1;
        let geometry = geometry.as_mut_ptr().addr() as u64;
        let pixel = pixel.as_mut_ptr().addr() as u64;
        let mut payload = [0xffff_ffff_u32];
        let output = payload.as_mut_ptr().addr() as u64;

        assert_eq!(call("ICkECTBxrMw", args(&[output, geometry])), 0);
        assert_eq!(payload, [PRIMITIVE_PAYLOAD_DISABLE]);
        assert_eq!(
            call("ICkECTBxrMw", args(&[0, geometry])),
            AGC_ERROR_PARAMETER_NULL
        );
        assert_eq!(
            call("ICkECTBxrMw", args(&[output, 0])),
            AGC_ERROR_PARAMETER_NULL
        );
        assert_eq!(
            call("ICkECTBxrMw", args(&[output, pixel])),
            AGC_ERROR_UNEXPECTED_PARAMETER
        );
    }

    #[test]
    fn semaphore_memory_is_validated_then_accepted_exactly_once_per_process() {
        let alignment = AMM_SEMAPHORE_MEMORY_ALIGNMENT;

        assert_eq!(
            call("OQTgEXyihvA", args(&[0, alignment])),
            AGC_ERROR_PARAMETER_NULL
        );
        assert_eq!(
            call("OQTgEXyihvA", args(&[alignment + 32, alignment])),
            AGC_ERROR_ALIGNMENT
        );
        assert_eq!(
            call("OQTgEXyihvA", args(&[alignment, alignment + 32])),
            AGC_ERROR_ALIGNMENT
        );

        assert_eq!(call("OQTgEXyihvA", args(&[alignment, alignment * 2])), 0);
        assert_eq!(AMM_SEMAPHORE_BASE.load(Ordering::Acquire), alignment);
        assert_eq!(AMM_SEMAPHORE_SIZE.load(Ordering::Acquire), alignment * 2);

        assert_eq!(
            call("OQTgEXyihvA", args(&[alignment * 4, alignment])),
            AGC_ERROR_SEMAPHORE_MEMORY_ALREADY_SET
        );
        assert_eq!(AMM_SEMAPHORE_BASE.load(Ordering::Acquire), alignment);
    }
}
