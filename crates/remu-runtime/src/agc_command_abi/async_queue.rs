// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    IT_COND_EXEC, IT_DISPATCH_INDIRECT, IT_EVENT_WRITE, IT_NOP, R_DISPATCH_RESET,
    WaitRegMemArguments, acquire_mem, allocate_dwords, dma_data, low_u8, low_u32, pm4,
    wait_reg_mem, write_u32, write_words,
};

pub(super) fn acb_reset_queue(arguments: [u64; 6]) -> u64 {
    if low_u32(arguments[1]) & !0x1c2 != 0 {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 2) else {
        return 0;
    };
    write_words(command, &[pm4(2, IT_NOP, R_DISPATCH_RESET), 0]);
    command
}

pub(super) fn event_write(arguments: [u64; 6]) -> u64 {
    let event_type = low_u8(arguments[1]);
    let address = arguments[2];
    if event_type > 0x3f {
        return 0;
    }
    let addressed = event_type & 0xfe == 0x38;
    if addressed && (address == 0 || address & 7 != 0) {
        return 0;
    }
    let count = if addressed { 4 } else { 2 };
    let Some(command) = allocate_dwords(arguments[0], count) else {
        return 0;
    };
    write_u32(command, pm4(count, IT_EVENT_WRITE, 0));
    if matches!(event_type, 7 | 15 | 16) {
        write_u32(command + 4, 0x400 | event_type);
    } else if addressed {
        write_words(
            command + 4,
            &[
                0x100 | event_type,
                low_u32(address) & !7,
                low_u32(address >> 32),
            ],
        );
    } else {
        write_u32(command + 4, event_type);
    }
    command
}

pub(super) fn acb_acquire_mem(arguments: [u64; 6]) -> u64 {
    acquire_mem(
        arguments[0],
        1,
        0,
        low_u32(arguments[1]),
        arguments[2],
        arguments[3],
        low_u32(arguments[4]),
    )
}

pub(super) fn cond_exec(arguments: [u64; 6]) -> u64 {
    let address = arguments[1];
    let count = low_u32(arguments[2]);
    if address == 0 || address & 3 != 0 || count > 0x3fff {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 5) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(5, IT_COND_EXEC, 0),
            low_u32(address) & !3,
            low_u32(address >> 32),
            0,
            count,
        ],
    );
    command
}

pub(super) fn acb_wait_reg_mem(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
    wait_reg_mem(WaitRegMemArguments {
        command_buffer: arguments[0],
        size: low_u8(arguments[1]),
        compare: low_u8(arguments[2]),
        operation: 0,
        cache_policy: low_u8(arguments[3]),
        address: arguments[4],
        reference: arguments[5],
        mask: stack[0],
        poll_cycles: low_u32(stack[1]),
    })
}

pub(super) fn acb_dma_data(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
    // Current Gen5 ACB callers omit the engine and block-engine parameters because the queue
    // selects the asynchronous engine itself. Shift the remaining ten arguments into the common
    // DCB encoder instead of interpreting the destination selector as an invalid engine value.
    dma_data(
        [
            arguments[0],
            1,
            arguments[1],
            arguments[2],
            arguments[3],
            arguments[4],
        ],
        [arguments[5], stack[0], stack[1], stack[2], stack[3], 1],
    )
}

pub(super) fn acb_dispatch_indirect(arguments: [u64; 6]) -> u64 {
    let address = arguments[1];
    if address == 0 || address & 3 != 0 {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 4) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(4, IT_DISPATCH_INDIRECT, 0),
            low_u32(address),
            low_u32(address >> 32),
            (low_u32(arguments[2]) & 0xa038) | 0x41,
        ],
    );
    command
}
