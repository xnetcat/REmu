// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    IT_DISPATCH_DIRECT, IT_INDIRECT_BUFFER, IT_NOP, IT_SET_SH_REG, MAX_PACKET_DWORDS,
    R_RELEASE_MEM, R_ZERO, allocate_dwords, low_u8, low_u16, low_u32,
    mark_graphics_release_for_command_buffer, pm4, read_u32, register_graphics_continuation,
    write_u32,
    write_words,
};

pub(super) fn type_two_packet(arguments: [u64; 6]) -> u64 {
    let Some(command) = allocate_dwords(arguments[0], 1) else {
        return 0;
    };
    write_u32(command, 0x8000_0000);
    command
}

pub(super) fn cb_nop(arguments: [u64; 6]) -> u64 {
    let count = low_u32(arguments[1]);
    if !(2..=MAX_PACKET_DWORDS).contains(&count) {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], count) else {
        return 0;
    };
    write_u32(command, pm4(count, IT_NOP, R_ZERO));
    for index in 1..count {
        write_u32(command + u64::from(index) * 4, 0);
    }
    command
}

pub(super) fn cb_dispatch(arguments: [u64; 6]) -> u64 {
    let Some(command) = allocate_dwords(arguments[0], 5) else {
        return 0;
    };
    let modifier = low_u32(arguments[4]);
    write_words(
        command,
        &[
            pm4(5, IT_DISPATCH_DIRECT, 0),
            low_u32(arguments[1]),
            low_u32(arguments[2]),
            low_u32(arguments[3]),
            (modifier & 0xa038) | 0x41,
        ],
    );
    command
}

pub(super) fn cb_branch(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
    let mode = low_u8(arguments[1]);
    let compare = low_u8(arguments[2]);
    let compare_address = arguments[3];
    let then_address = stack[1];
    let then_count = low_u32(stack[2]);
    let else_address = stack[4];
    let else_count = low_u32(stack[5]);
    if mode > 3
        || compare > 7
        || compare_address == 0
        || compare_address & 7 != 0
        || then_address & 3 != 0
        || else_address & 3 != 0
        || then_count > 0x000f_ffff
        || else_count > 0x000f_ffff
        || low_u8(stack[0]) > 3
        || low_u8(stack[3]) > 3
    {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 14) else {
        return 0;
    };
    eprintln!(
        "[REMu][AGC] build branch descriptor={:#x} command={command:#x} mode={mode} compare={compare} address={compare_address:#x} mask={:#x} reference={:#x} then={then_address:#x}/{then_count} cache={} else={else_address:#x}/{else_count} cache={}",
        arguments[0],
        arguments[4],
        arguments[5],
        low_u8(stack[0]),
        low_u8(stack[3]),
    );
    write_words(
        command,
        &[
            pm4(14, IT_INDIRECT_BUFFER, 0),
            mode | (compare << 8),
            low_u32(compare_address) & !7,
            low_u32(compare_address >> 32),
            low_u32(arguments[4]),
            low_u32(arguments[4] >> 32),
            low_u32(arguments[5]),
            low_u32(arguments[5] >> 32),
            low_u32(then_address) & !3,
            low_u32(then_address >> 32),
            then_count | (low_u8(stack[0]) << 28),
            low_u32(else_address) & !3,
            low_u32(else_address >> 32),
            else_count | (low_u8(stack[3]) << 28),
        ],
    );
    register_graphics_continuation(command, then_address, then_count, else_address, else_count);
    command
}

pub(super) fn cb_set_register_range(arguments: [u64; 6], opcode: u32) -> u64 {
    let first_register = low_u32(arguments[1]);
    let values = arguments[2];
    let value_count = low_u32(arguments[3]);
    let Some(packet_count) = value_count.checked_add(2) else {
        return 0;
    };
    if value_count == 0
        || value_count > 0x3fff
        || first_register > 0xffff
        || (values != 0 && values & 3 != 0)
    {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], packet_count) else {
        return 0;
    };
    write_u32(command, pm4(packet_count, opcode, 0));
    write_u32(command + 4, first_register);
    // A null values pointer reserves an inline data packet whose payload the caller fills after
    // `sceAgcGetDataPacketPayloadAddress`. This is a first-class ABI form, not an error.
    if values != 0 {
        for index in 0..value_count {
            write_u32(
                command + 8 + u64::from(index) * 4,
                read_u32(values + u64::from(index) * 4),
            );
        }
    }
    command
}

pub(super) fn cb_set_sh_registers(arguments: [u64; 6]) -> u64 {
    let command_buffer = arguments[0];
    let registers = arguments[1];
    let register_count = low_u32(arguments[2]);
    if registers == 0 || registers & 3 != 0 || register_count == 0 || register_count > 4_096 {
        return 0;
    }

    let mut first_command = 0;
    let mut run_start = 0;
    while run_start < register_count {
        let first_offset = read_u32(registers + u64::from(run_start) * 8);
        if first_offset > 0xffff {
            return first_command;
        }
        let mut run_end = run_start + 1;
        let mut previous = first_offset;
        while run_end < register_count {
            let next = read_u32(registers + u64::from(run_end) * 8);
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
        write_u32(command, pm4(packet_count, IT_SET_SH_REG, 0));
        write_u32(command + 4, first_offset);
        for index in 0..value_count {
            write_u32(
                command + 8 + u64::from(index) * 4,
                read_u32(registers + u64::from(run_start + index) * 8 + 4),
            );
        }
        run_start = run_end;
    }
    first_command
}

pub(super) fn cb_release_mem(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
    let action = low_u8(arguments[1]);
    let gcr_control = low_u16(arguments[2]);
    let destination = low_u8(arguments[3]);
    let cache_policy = low_u8(arguments[4]);
    let address = arguments[5];
    let data_selection = low_u8(stack[0]);
    let data = stack[1];
    let gds_offset = low_u16(stack[2]);
    let gds_size = low_u16(stack[3]);
    let interrupt = low_u8(stack[4]);
    let interrupt_context = low_u32(stack[5]);
    if destination > 1 || data_selection > 3 || gds_offset != 0 || gds_size > 2 || interrupt > 3 {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 8) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(8, IT_NOP, R_RELEASE_MEM),
            action | (cache_policy << 8),
            gcr_control | (data_selection << 16) | (interrupt << 24),
            low_u32(address),
            low_u32(address >> 32),
            low_u32(data),
            low_u32(data >> 32),
            interrupt_context,
        ],
    );
    mark_graphics_release_for_command_buffer(arguments[0], command);
    command
}
