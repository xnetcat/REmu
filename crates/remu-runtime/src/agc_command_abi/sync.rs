// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    GRAPHICS5_ERROR_INVALID_PACKET, IT_COPY_DATA, IT_DMA_DATA, IT_GET_LOD_STATS,
    IT_INDIRECT_BUFFER, IT_NOP, IT_PFP_SYNC_ME, IT_REWIND, IT_SET_PREDICATION, IT_WRITE_DATA,
    MAX_PACKET_DWORDS, R_ACQUIRE_MEM, R_FLIP, R_POP_MARKER, R_PUSH_MARKER, R_WAIT_MEM_32,
    R_WAIT_MEM_64, WaitRegMemArguments, allocate_dwords, low_u8, low_u32, pm4, read_u8, read_u32,
    write_u32, write_words,
};

pub(super) fn predication_bit(value: u64) -> u32 {
    u32::from(low_u8(value) == 1)
}

pub(super) fn set_packet_predication(arguments: [u64; 6]) -> u64 {
    let packet = arguments[0];
    if packet == 0 || !packet.is_multiple_of(4) {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    let header = read_u32(packet);
    write_u32(packet, (header & !1) | predication_bit(arguments[1]));
    0
}

pub(super) fn set_range_predication(arguments: [u64; 6]) -> u64 {
    let mut packet = arguments[0];
    let end = arguments[1];
    if packet >= end {
        return 0;
    }
    if packet == 0 || end == 0 || !packet.is_multiple_of(4) || !end.is_multiple_of(4) {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    let predication = predication_bit(arguments[2]);
    while packet < end {
        let header = read_u32(packet);
        write_u32(packet, (header & !1) | predication);
        let dword_count = if header >> 30 == 2 || header & 0x3fff_ff00 == 0x3fff_1000 {
            1
        } else {
            ((header >> 16) & 0x3fff) + 2
        };
        let Some(next) = packet.checked_add(u64::from(dword_count) * 4) else {
            return GRAPHICS5_ERROR_INVALID_PACKET;
        };
        packet = next;
    }
    0
}

pub(super) fn dcb_acquire_mem(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
    acquire_mem(
        arguments[0],
        low_u8(arguments[1]),
        low_u32(arguments[2]),
        low_u32(arguments[3]),
        arguments[4],
        arguments[5],
        low_u32(stack[0]),
    )
}

pub(super) fn dcb_wait_reg_mem(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
    wait_reg_mem(WaitRegMemArguments {
        command_buffer: arguments[0],
        size: low_u8(arguments[1]),
        compare: low_u8(arguments[2]),
        operation: low_u8(arguments[3]),
        cache_policy: low_u8(arguments[4]),
        address: arguments[5],
        reference: stack[0],
        mask: stack[1],
        poll_cycles: low_u32(stack[2]),
    })
}

pub(super) fn acquire_mem(
    command_buffer: u64,
    engine: u32,
    cb_db_operation: u32,
    gcr_control: u32,
    base: u64,
    size_bytes: u64,
    poll_cycles: u32,
) -> u64 {
    let no_size = size_bytes == u64::MAX;
    if engine > 1
        || base & 0xff != 0
        || base >> 40 != 0
        || (!no_size && (size_bytes & 0xff != 0 || size_bytes >> 40 != 0))
    {
        return 0;
    }
    let Some(command) = allocate_dwords(command_buffer, 8) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(8, IT_NOP, R_ACQUIRE_MEM),
            (engine << 31) | cb_db_operation,
            if no_size { 0 } else { low_u32(size_bytes >> 8) },
            0,
            low_u32(base >> 8),
            0,
            poll_cycles / 40,
            gcr_control,
        ],
    );
    command
}

pub(super) fn wait_reg_mem(arguments: WaitRegMemArguments) -> u64 {
    let WaitRegMemArguments {
        command_buffer,
        size,
        compare,
        operation,
        cache_policy,
        address,
        reference,
        mask,
        poll_cycles,
    } = arguments;
    if size > 1
        || compare > 7
        || operation > 0x0f
        || cache_policy > 3
        || address == 0
        || address & if size == 0 { 3 } else { 7 } != 0
    {
        return 0;
    }
    let count = if size == 0 { 7 } else { 9 };
    let Some(command) = allocate_dwords(command_buffer, count) else {
        return 0;
    };
    let poll = (poll_cycles >> 4).min(0xffff);
    if size == 0 {
        let control = 0x10
            | compare
            | ((operation & 3) << 8)
            | ((operation & 0x0c) << 4)
            | (cache_policy << 25);
        write_words(
            command,
            &[
                pm4(7, IT_NOP, R_WAIT_MEM_32),
                low_u32(address) & !3,
                low_u32(address >> 32) & 0x3_ffff,
                low_u32(mask),
                low_u32(reference),
                control,
                poll,
            ],
        );
    } else {
        let control =
            0x10 | compare | ((operation & 1) << 8) | ((operation & 6) << 5) | (cache_policy << 25);
        write_words(
            command,
            &[
                pm4(9, IT_NOP, R_WAIT_MEM_64),
                low_u32(address) & !7,
                low_u32(address >> 32) & 0x3_ffff,
                low_u32(mask),
                low_u32(mask >> 32),
                low_u32(reference),
                low_u32(reference >> 32),
                control,
                poll,
            ],
        );
    }
    command
}

pub(super) fn dma_data(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
    let engine = low_u8(arguments[1]);
    let destination = low_u8(arguments[2]);
    let destination_cache = low_u8(arguments[3]);
    let destination_address = arguments[4];
    let source = low_u8(arguments[5]);
    let source_cache = low_u8(stack[0]);
    let mut source_address = stack[1];
    let byte_count = low_u32(stack[2]);
    let wait = low_u8(stack[3]);
    let confirm = low_u8(stack[4]);
    let block = low_u8(stack[5]);
    if engine > 1
        || destination_cache > 3
        || source_cache > 3
        || byte_count > 0x03ff_ffff
        || wait > 1
        || confirm > 1
        || block > 1
    {
        return 0;
    }
    source_address = match source {
        0x14 => {
            if engine == 1 {
                0x30148
            } else {
                0x30174
            }
        }
        0x24 => {
            if engine == 1 {
                0x30150
            } else {
                0x3017c
            }
        }
        0x64 => {
            if engine == 1 {
                0x30158
            } else {
                0x30184
            }
        }
        _ => source_address,
    };
    let Some(command) = allocate_dwords(arguments[0], 7) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(7, IT_DMA_DATA, 0),
            (engine & 1)
                | ((source_cache & 3) << 13)
                | ((destination & 3) << 20)
                | ((destination_cache & 3) << 25)
                | ((source & 3) << 29)
                | ((block & 1) << 31),
            low_u32(source_address),
            low_u32(source_address >> 32),
            low_u32(destination_address),
            low_u32(destination_address >> 32),
            byte_count
                | ((source & 4) << 24)
                | ((destination & 4) << 25)
                | ((source & 8) << 25)
                | ((destination & 8) << 26)
                | ((wait & 1) << 30)
                | ((confirm & 1) << 31),
        ],
    );
    command
}

pub(super) fn copy_data(arguments: [u64; 6], stack: [u64; 6], acb: bool) -> u64 {
    let destination = low_u8(arguments[1]);
    let destination_cache = low_u8(arguments[2]);
    let destination_address = arguments[3];
    let mut source = low_u8(arguments[4]);
    let source_cache = low_u8(arguments[5]);
    let source_value = stack[0];
    let item_size = low_u8(stack[1]);
    let confirm = low_u8(stack[2]);
    if acb && source == 5 {
        source = 10;
    }
    if destination_cache > 3 || source_cache > 3 || item_size > 1 || confirm > 1 {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 6) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(6, IT_COPY_DATA, 0),
            ((source >> 1) & 0xf)
                | (((destination >> 1) & 0xf) << 8)
                | ((source_cache & 3) << 13)
                | ((item_size & 1) << 16)
                | ((confirm & 1) << 20)
                | ((destination_cache & 3) << 25)
                | ((source & 1) << 30),
            low_u32(source_value),
            low_u32(source_value >> 32),
            low_u32(destination_address),
            low_u32(destination_address >> 32),
        ],
    );
    command
}

pub(super) fn dcb_jump(arguments: [u64; 6]) -> u64 {
    let mode = low_u8(arguments[1]);
    let cache_policy = low_u8(arguments[2]);
    let target = arguments[3];
    let count = low_u32(arguments[4]);
    if mode > 1 || cache_policy > 3 || target == 0 || target & 3 != 0 || count > 0x000f_ffff {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 4) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(4, IT_INDIRECT_BUFFER, 0),
            low_u32(target) & !3,
            low_u32(target >> 32),
            0x0f20_0000 | (cache_policy << 28) | (mode << 20) | count,
        ],
    );
    command
}

pub(super) fn dcb_rewind(arguments: [u64; 6]) -> u64 {
    if low_u32(arguments[1]) > 1 {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 2) else {
        return 0;
    };
    write_words(
        command,
        &[pm4(2, IT_REWIND, 0), (low_u32(arguments[1]) & 1) << 31],
    );
    command
}

pub(super) fn set_predication(arguments: [u64; 6]) -> u64 {
    let condition = low_u8(arguments[1]);
    let operation = low_u8(arguments[2]);
    let wait = low_u8(arguments[3]);
    let address = arguments[4];
    if condition > 1 || operation > 7 || wait > 1 || address == 0 || address & 0x0f != 0 {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 4) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(4, IT_SET_PREDICATION, 0),
            (condition << 8) | (wait << 12) | (operation << 16),
            low_u32(address) & !0x0f,
            low_u32(address >> 32),
        ],
    );
    command
}

pub(super) fn stall_command_buffer_parser(arguments: [u64; 6]) -> u64 {
    let Some(command) = allocate_dwords(arguments[0], 2) else {
        return 0;
    };
    write_words(command, &[pm4(2, IT_PFP_SYNC_ME, 0), 0]);
    command
}

pub(super) fn write_data(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
    let destination = low_u8(arguments[1]);
    let cache_policy = low_u8(arguments[2]);
    let address = arguments[3];
    let data = arguments[4];
    let count = low_u32(arguments[5]);
    let increment = low_u8(stack[0]);
    let confirm = low_u8(stack[1]);
    let Some(packet_count) = count.checked_add(4) else {
        return 0;
    };
    if data == 0
        || data & 3 != 0
        || cache_policy > 3
        || increment > 1
        || confirm > 1
        || packet_count > MAX_PACKET_DWORDS
        || address & 3 != 0
    {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], packet_count) else {
        return 0;
    };
    let write_confirm = if destination == 0 { 0 } else { confirm };
    write_words(
        command,
        &[
            pm4(packet_count, IT_WRITE_DATA, 0),
            ((destination & 1) << 30)
                | ((destination & 0x1e) << 7)
                | (increment << 16)
                | (write_confirm << 20)
                | (cache_policy << 25),
            low_u32(address) & !3,
            low_u32(address >> 32),
        ],
    );
    for index in 0..count {
        write_u32(
            command + 16 + u64::from(index) * 4,
            read_u32(data + u64::from(index) * 4),
        );
    }
    command
}

pub(super) fn dcb_get_lod_stats(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
    let cache_policy = low_u8(arguments[1]);
    let address = arguments[2];
    if cache_policy > 3 || address == 0 || address & 0x3f != 0 {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 5) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(5, IT_GET_LOD_STATS, 0),
            low_u32(arguments[3]),
            low_u32(address) & !0x3f,
            low_u32(address >> 32),
            (cache_policy << 28)
                | ((low_u8(stack[0]) & 1) << 19)
                | ((low_u8(arguments[5]) & 1) << 18)
                | ((low_u32(arguments[4]) & 0xff) << 10)
                | ((low_u32(stack[1]) & 0xff) << 2),
        ],
    );
    command
}

pub(super) fn push_marker(arguments: [u64; 6]) -> u64 {
    let string = arguments[1];
    let max_bytes = usize::try_from((MAX_PACKET_DWORDS - 1) * 4).expect("marker bound fits");
    let length = if string == 0 {
        1
    } else {
        let mut length = 0_usize;
        while length < max_bytes
            && read_u8(string + u64::try_from(length).expect("length fits")) != 0
        {
            length += 1;
        }
        if length == max_bytes {
            return 0;
        }
        length + 1
    };
    let payload = u32::try_from(length.div_ceil(4)).expect("bounded marker payload");
    let count = 1 + payload.max(1);
    let Some(command) = allocate_dwords(arguments[0], count) else {
        return 0;
    };
    write_u32(command, pm4(count, IT_NOP, R_PUSH_MARKER));
    for index in 1..count {
        write_u32(command + u64::from(index) * 4, 0);
    }
    if string != 0 {
        for index in 0..length {
            let value = read_u8(string + u64::try_from(index).expect("length fits"));
            // SAFETY: the allocated payload contains `length` bytes.
            unsafe {
                std::ptr::write(((command + 4) as *mut u8).add(index), value);
            }
        }
    }
    command
}

pub(super) fn pop_marker(arguments: [u64; 6]) -> u64 {
    let Some(command) = allocate_dwords(arguments[0], 2) else {
        return 0;
    };
    write_words(command, &[pm4(2, IT_NOP, R_POP_MARKER), 0]);
    command
}

pub(super) fn set_flip(arguments: [u64; 6]) -> u64 {
    let Some(command) = allocate_dwords(arguments[0], 6) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(6, IT_NOP, R_FLIP),
            low_u32(arguments[1]),
            low_u32(arguments[2]),
            low_u32(arguments[3]),
            low_u32(arguments[4]),
            low_u32(arguments[4] >> 32),
        ],
    );
    command
}
