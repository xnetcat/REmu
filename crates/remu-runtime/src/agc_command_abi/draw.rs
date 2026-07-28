// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    IT_CLEAR_STATE, IT_DISPATCH_DRAW_PREAMBLE, IT_DISPATCH_INDIRECT, IT_DRAW_INDEX_2,
    IT_DRAW_INDEX_AUTO, IT_DRAW_INDEX_INDIRECT, IT_DRAW_INDEX_INDIRECT_MULTI,
    IT_DRAW_INDEX_OFFSET_2, IT_DRAW_INDIRECT, IT_INDEX_BUFFER_SIZE, IT_NOP, IT_NUM_INSTANCES,
    IT_SET_BASE, IT_SET_UCONFIG_REG_INDEX, IndirectRegisterKind, R_WAIT_FLIP_DONE, VGT_INDEX_TYPE,
    allocate_dwords, low_u8, low_u32, pm4, write_words,
};

pub(super) fn dcb_reset_queue(arguments: [u64; 6]) -> u64 {
    if low_u32(arguments[1]) & !0x0fff != 0 || low_u32(arguments[2]) > 0x0f {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 2) else {
        return 0;
    };
    write_words(
        command,
        &[pm4(2, IT_CLEAR_STATE, 0), low_u32(arguments[2]) & 0x0f],
    );
    command
}

pub(super) fn dcb_wait_until_safe(arguments: [u64; 6]) -> u64 {
    let Some(command) = allocate_dwords(arguments[0], 7) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(7, IT_NOP, R_WAIT_FLIP_DONE),
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

pub(super) fn set_register_direct(arguments: [u64; 6], opcode: u32) -> u64 {
    let register = low_u32(arguments[1]);
    let value = low_u32(arguments[1] >> 32);
    if register > 0xffff {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 3) else {
        return 0;
    };
    write_words(command, &[pm4(3, opcode, 0), register, value]);
    command
}

pub(super) fn set_registers_indirect(arguments: [u64; 6], kind: IndirectRegisterKind) -> u64 {
    let registers = arguments[1];
    let register_count = low_u32(arguments[2]);
    if registers == 0 || registers & 3 != 0 || register_count > 0x3fff {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 5) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(5, kind.opcode(), 0),
            low_u32(registers) & !3,
            low_u32(registers >> 32),
            0x8000_0000,
            register_count,
        ],
    );
    command
}

pub(super) fn set_index_size(arguments: [u64; 6]) -> u64 {
    let index_size = low_u8(arguments[1]);
    let cache_policy = low_u8(arguments[2]);
    if index_size > 3 || cache_policy > 3 {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 3) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(3, IT_SET_UCONFIG_REG_INDEX, 0),
            0x2000_0000 | VGT_INDEX_TYPE,
            0x400 | index_size | (cache_policy << 6),
        ],
    );
    command
}

pub(super) fn set_index_count(arguments: [u64; 6]) -> u64 {
    let Some(command) = allocate_dwords(arguments[0], 2) else {
        return 0;
    };
    write_words(
        command,
        &[pm4(2, IT_INDEX_BUFFER_SIZE, 0), low_u32(arguments[1])],
    );
    command
}

pub(super) fn set_num_instances(arguments: [u64; 6]) -> u64 {
    let Some(command) = allocate_dwords(arguments[0], 2) else {
        return 0;
    };
    write_words(
        command,
        &[pm4(2, IT_NUM_INSTANCES, 0), low_u32(arguments[1])],
    );
    command
}

pub(super) fn draw_index_initiator(modifier: u64) -> u32 {
    if modifier & (1_u64 << 32) == 0 {
        (low_u32(modifier) >> 3) & 0x20
    } else {
        0
    }
}

pub(super) fn indirect_modifier_sgpr_base(modifier: u32) -> u32 {
    if matches!(modifier >> 29, 3 | 5) {
        0x10c
    } else {
        0x8c
    }
}

pub(super) fn indirect_patch_offsets(modifier: u64, indexed: bool) -> u64 {
    let low = low_u32(modifier);
    let sgpr_base = indirect_modifier_sgpr_base(low);
    let mut base_vertex = 0x280_u64;
    if low & 1 != 0 {
        base_vertex = u64::from(sgpr_base + ((low >> 9) & 0x1f));
    }
    let mut start_instance = 0x280_u64;
    if low & 4 != 0 {
        start_instance = u64::from(sgpr_base + ((low >> 19) & 0x1f));
    }
    if indexed && low & 2 != 0 {
        base_vertex |= u64::from(sgpr_base + ((low >> 14) & 0x1f)) << 16;
        base_vertex |= 1_u64 << 59;
    }
    base_vertex | (start_instance << 32)
}

pub(super) fn indirect_draw_initiator(modifier: u64) -> u32 {
    if modifier & (1_u64 << 32) == 0 {
        ((low_u32(modifier) >> 3) & 0x20) | 2
    } else {
        0
    }
}

pub(super) fn draw_index(arguments: [u64; 6]) -> u64 {
    let count = low_u32(arguments[1]);
    let address = arguments[2];
    if address == 0 || address & 1 != 0 {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 6) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(6, IT_DRAW_INDEX_2, 0),
            count.max(1),
            low_u32(address),
            low_u32(address >> 32),
            count,
            draw_index_initiator(arguments[3]),
        ],
    );
    command
}

pub(super) fn draw_index_multi_instanced(arguments: [u64; 6]) -> u64 {
    let index_count = low_u32(arguments[1]);
    let index_address = arguments[2];
    let object_ids = arguments[3];
    let instance_count = low_u32(arguments[4]);
    if index_address == 0 || index_address & 1 != 0 || object_ids == 0 || object_ids & 3 != 0 {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 9) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(9, IT_DISPATCH_DRAW_PREAMBLE, 0),
            index_count,
            low_u32(index_address),
            low_u32(index_address >> 32),
            instance_count.max(1),
            low_u32(object_ids),
            low_u32(object_ids >> 32),
            instance_count,
            draw_index_initiator(arguments[5]) | 0x80,
        ],
    );
    command
}

pub(super) fn draw_index_auto(arguments: [u64; 6]) -> u64 {
    let Some(command) = allocate_dwords(arguments[0], 3) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(3, IT_DRAW_INDEX_AUTO, 0),
            low_u32(arguments[1]),
            draw_index_initiator(arguments[2]) | 2,
        ],
    );
    command
}

pub(super) fn draw_index_offset(arguments: [u64; 6]) -> u64 {
    let count = low_u32(arguments[2]);
    let Some(command) = allocate_dwords(arguments[0], 5) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(5, IT_DRAW_INDEX_OFFSET_2, 0),
            count.max(1),
            low_u32(arguments[1]),
            count,
            draw_index_initiator(arguments[3]),
        ],
    );
    command
}

pub(super) fn set_base_indirect_args(arguments: [u64; 6]) -> u64 {
    let shader_type = low_u32(arguments[1]);
    let address = arguments[2];
    if shader_type > 1 || address == 0 || address & 7 != 0 {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 4) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(4, IT_SET_BASE, 0) | (shader_type << 1),
            1,
            low_u32(address) & !7,
            low_u32(address >> 32),
        ],
    );
    command
}

pub(super) fn draw_indirect(arguments: [u64; 6], indexed: bool) -> u64 {
    if low_u32(arguments[1]) & 3 != 0 {
        return 0;
    }
    let modifier = arguments[2];
    let offsets = indirect_patch_offsets(modifier, indexed);
    let Some(command) = allocate_dwords(arguments[0], 5) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(
                5,
                if indexed {
                    IT_DRAW_INDEX_INDIRECT
                } else {
                    IT_DRAW_INDIRECT
                },
                0,
            ),
            low_u32(arguments[1]),
            low_u32(offsets),
            low_u32(offsets >> 32),
            indirect_draw_initiator(modifier),
        ],
    );
    command
}

pub(super) fn draw_index_indirect_multi(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
    let data_offset = low_u32(arguments[1]);
    let count_indirect = low_u32(arguments[2]);
    let count_address = arguments[4];
    let stride = low_u32(arguments[5]);
    let modifier = stack[0];
    if data_offset & 3 != 0
        || count_indirect > 1
        || (count_indirect != 0 && (count_address == 0 || count_address & 3 != 0))
        || stride & 3 != 0
    {
        return 0;
    }
    let offsets = indirect_patch_offsets(modifier, true);
    let Some(command) = allocate_dwords(arguments[0], 10) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(10, IT_DRAW_INDEX_INDIRECT_MULTI, 0),
            data_offset,
            low_u32(offsets),
            low_u32(offsets >> 32),
            count_indirect << 30,
            low_u32(arguments[3]),
            low_u32(count_address) & !3,
            low_u32(count_address >> 32),
            stride,
            indirect_draw_initiator(modifier),
        ],
    );
    command
}

pub(super) fn dcb_dispatch_indirect(arguments: [u64; 6]) -> u64 {
    if low_u32(arguments[1]) & 3 != 0 {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 3) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(3, IT_DISPATCH_INDIRECT, 0),
            low_u32(arguments[1]),
            (low_u32(arguments[2]) & 0xa038) | 0x41,
        ],
    );
    command
}
