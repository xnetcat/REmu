//! Guest-facing Gen5 AGC command-buffer construction ABI.
//!
//! The packet layouts in this module are the native Gen5 encodings used by the
//! open-source `KytyPS5` implementation.  Keeping construction and patching in one
//! place is important: patch functions validate the packet opcode they mutate,
//! and every `GetSize` result is tied to the corresponding encoder.

const GRAPHICS5_ERROR_INVALID_PACKET: u64 = 0x8a6c_000c;

const COMMAND_BUFFER_CURSOR_UP_OFFSET: u64 = 0x10;
const COMMAND_BUFFER_CURSOR_DOWN_OFFSET: u64 = 0x18;
const COMMAND_BUFFER_CALLBACK_OFFSET: u64 = 0x20;
const COMMAND_BUFFER_USER_DATA_OFFSET: u64 = 0x28;
const COMMAND_BUFFER_RESERVED_DWORDS_OFFSET: u64 = 0x30;

const IT_NOP: u32 = 0x10;
const IT_SET_BASE: u32 = 0x11;
const IT_CLEAR_STATE: u32 = 0x12;
const IT_INDEX_BUFFER_SIZE: u32 = 0x13;
const IT_DISPATCH_DIRECT: u32 = 0x15;
const IT_DISPATCH_INDIRECT: u32 = 0x16;
const IT_SET_PREDICATION: u32 = 0x20;
const IT_COND_EXEC: u32 = 0x22;
const IT_DRAW_INDIRECT: u32 = 0x24;
const IT_DRAW_INDEX_INDIRECT: u32 = 0x25;
const IT_DRAW_INDEX_2: u32 = 0x27;
const IT_DRAW_INDEX_AUTO: u32 = 0x2d;
const IT_NUM_INSTANCES: u32 = 0x2f;
const IT_DRAW_INDEX_OFFSET_2: u32 = 0x35;
const IT_WRITE_DATA: u32 = 0x37;
const IT_DRAW_INDEX_INDIRECT_MULTI: u32 = 0x38;
const IT_DISPATCH_DRAW_PREAMBLE: u32 = 0x3a;
const IT_INDIRECT_BUFFER: u32 = 0x3f;
const IT_COPY_DATA: u32 = 0x40;
const IT_PFP_SYNC_ME: u32 = 0x42;
const IT_EVENT_WRITE: u32 = 0x46;
const IT_EVENT_WRITE_EOP: u32 = 0x47;
const IT_RELEASE_MEM: u32 = 0x49;
const IT_DMA_DATA: u32 = 0x50;
const IT_REWIND: u32 = 0x59;
const IT_SET_SH_REG_INDIRECT: u32 = 0x63;
const IT_SET_UCONFIG_REG_INDIRECT: u32 = 0x64;
const IT_SET_CONTEXT_REG: u32 = 0x69;
const IT_SET_SH_REG: u32 = 0x76;
const IT_SET_UCONFIG_REG: u32 = 0x79;
const IT_SET_UCONFIG_REG_INDEX: u32 = 0x7a;
const IT_GET_LOD_STATS: u32 = 0x8e;
const IT_SET_CONTEXT_REG_INDIRECT: u32 = 0x9f;

const R_ZERO: u32 = 0x00;
const R_WAIT_FLIP_DONE: u32 = 0x06;
const R_DISPATCH_RESET: u32 = 0x09;
const R_WAIT_MEM_32: u32 = 0x0a;
const R_PUSH_MARKER: u32 = 0x0b;
const R_POP_MARKER: u32 = 0x0c;
const R_ACQUIRE_MEM: u32 = 0x14;
const R_WAIT_MEM_64: u32 = 0x16;
const R_FLIP: u32 = 0x17;
const R_RELEASE_MEM: u32 = 0x18;

const VGT_INDEX_TYPE: u32 = 0x243;
const MAX_PACKET_DWORDS: u32 = 0x4001;

type GrowCallback = unsafe extern "C" fn(u64, u32, u64) -> bool;

#[derive(Clone, Copy)]
enum IndirectRegisterKind {
    Context,
    Shader,
    UserConfig,
}

#[derive(Clone, Copy)]
struct WaitRegMemArguments {
    command_buffer: u64,
    size: u32,
    compare: u32,
    operation: u32,
    cache_policy: u32,
    address: u64,
    reference: u64,
    mask: u64,
    poll_cycles: u32,
}

impl IndirectRegisterKind {
    const fn opcode(self) -> u32 {
        match self {
            Self::Context => IT_SET_CONTEXT_REG_INDIRECT,
            Self::Shader => IT_SET_SH_REG_INDIRECT,
            Self::UserConfig => IT_SET_UCONFIG_REG_INDIRECT,
        }
    }
}

/// Dispatch one stateless Gen5 AGC command-construction import.
///
/// `arguments` contains the six `SysV` integer-register arguments and
/// `stack_arguments` starts with the seventh argument. The return is `None`
/// only when `nid` is outside this coherent ABI cohort.
pub(crate) fn dispatch_agc_command_abi(
    nid: &str,
    arguments: [u64; 6],
    stack_arguments: [u64; 6],
) -> Option<u64> {
    dispatch_indirect_register_patch(nid, arguments)
        .or_else(|| dispatch_common_command(nid, arguments, stack_arguments))
        .or_else(|| dispatch_async_command(nid, arguments, stack_arguments))
        .or_else(|| dispatch_draw_command(nid, arguments, stack_arguments))
        .or_else(|| dispatch_draw_sync_command(nid, arguments, stack_arguments))
        .or_else(|| dispatch_packet_patch(nid, arguments))
}

fn dispatch_indirect_register_patch(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "vcmNN+AAXnY" => patch_indirect_address(arguments, IndirectRegisterKind::Context),
        "Qrj4c+61z4A" => patch_indirect_address(arguments, IndirectRegisterKind::Shader),
        "6lNcCp+fxi4" => patch_indirect_address(arguments, IndirectRegisterKind::UserConfig),
        "whb1RL7K4Ss" => patch_indirect_count(arguments, IndirectRegisterKind::Context, false),
        "nCUgItdN2ms" => patch_indirect_count(arguments, IndirectRegisterKind::Shader, false),
        "fRG-JOH5+sI" => patch_indirect_count(arguments, IndirectRegisterKind::UserConfig, false),
        "d-6uF9sZDIU" => patch_indirect_count(arguments, IndirectRegisterKind::Context, true),
        "z2duB-hHQSM" => patch_indirect_count(arguments, IndirectRegisterKind::Shader, true),
        "vRoArM9zaIk" => patch_indirect_count(arguments, IndirectRegisterKind::UserConfig, true),
        _ => return None,
    })
}

fn dispatch_common_command(
    nid: &str,
    arguments: [u64; 6],
    stack_arguments: [u64; 6],
) -> Option<u64> {
    Some(match nid {
        "qj7QZpgr9Uw" => type_two_packet(arguments),
        "LtTouSCZjHM" => cb_nop(arguments),
        "t7PlZ9nt5Lc" => low_u32(arguments[0]).wrapping_mul(4).into(),
        "k3GhuSNmBLU" => cb_dispatch(arguments),
        "Abendgtz+3o" => 20,
        "w1KFAHVqpaU" => cb_branch(arguments, stack_arguments),
        "n2fD4A+pb+g" => cb_set_register_range(arguments, IT_SET_SH_REG),
        "bxGoVxpdSPQ" => low_u32(arguments[0]).wrapping_mul(4).wrapping_add(8).into(),
        "UZbQjYAwwXM" => cb_set_sh_registers(arguments),
        "wr23dPKyWc0" => cb_release_mem(arguments, stack_arguments),
        "hL7C0IRpWZI" => 32,
        _ => return None,
    })
}

fn type_two_packet(arguments: [u64; 6]) -> u64 {
    let Some(command) = allocate_dwords(arguments[0], 1) else {
        return 0;
    };
    write_u32(command, 0x8000_0000);
    command
}

fn dispatch_async_command(
    nid: &str,
    arguments: [u64; 6],
    stack_arguments: [u64; 6],
) -> Option<u64> {
    Some(match nid {
        "JrtiDtKeS38" => acb_reset_queue(arguments),
        "cFazmnXpJOE" => event_write(arguments),
        "KT-hTp-Ch14" => acb_acquire_mem(arguments),
        "ewobAQeMo5k" | "M0ttm8h7SKA" => 32,
        "qyM2bxYFPAk" => cond_exec(arguments),
        "ozKzBP4aki4" => 20,
        "htn36gPnBk4" => acb_wait_reg_mem(arguments, stack_arguments),
        "-RnpfpxIhec" => dma_data(arguments, stack_arguments),
        "qzMN2XKGA4k" => copy_data(arguments, stack_arguments, true),
        "j3EtxFkSIhQ" => acb_dispatch_indirect(arguments),
        "eZ4+17OQz4Q" => write_data(arguments, stack_arguments),
        "xAeBOa0A3kk" | "cpCILPya5Zk" => push_marker(arguments),
        "6mFxkVqdmbQ" => pop_marker(arguments),
        _ => return None,
    })
}

fn dispatch_draw_command(nid: &str, arguments: [u64; 6], stack_arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "TRO721eVt4g" => dcb_reset_queue(arguments),
        "MWiElSNE8j8" => dcb_wait_until_safe(arguments),
        "pFLArOT53+w" => set_register_direct(arguments, IT_SET_SH_REG),
        "LHFXRrlTPD8" => set_register_direct(arguments, IT_SET_CONTEXT_REG),
        "w4-d0n60hdo" => set_register_direct(arguments, IT_SET_UCONFIG_REG),
        "ZvwO9euwYzc" => set_registers_indirect(arguments, IndirectRegisterKind::Context),
        "-HOOCn0JY48" => set_registers_indirect(arguments, IndirectRegisterKind::Shader),
        "hvUfkUIQcOE" => set_registers_indirect(arguments, IndirectRegisterKind::UserConfig),
        "GIIW2J37e70" => set_index_size(arguments),
        // DcbSetIndexBuffer stays in the parent runtime because its four-argument ABI associates
        // a CPU index pointer with a GPU resource address.
        "8N2tmT3jmC8" => set_index_count(arguments),
        "mljzuGDZRQ4" => 28,
        "tSBxhAPyytQ" => set_num_instances(arguments),
        "q88lQ+GP5Yk" => draw_index(arguments),
        "6ee9Hd3EWXQ" => 24,
        "Rlx+bykm0r0" => draw_index_multi_instanced(arguments),
        "mR9j7+SfM34" => 36,
        "Yw0jKSqop+E" => draw_index_auto(arguments),
        "B+aG9DUnTKA" => draw_index_offset(arguments),
        "RmaJwLtc8rY" => set_base_indirect_args(arguments),
        "1q1titRBL6o" => draw_indirect(arguments, false),
        "t1vNu082-jM" => draw_indirect(arguments, true),
        "ypVBz4uPKcQ" => draw_index_indirect_multi(arguments, stack_arguments),
        "CtB+A9-VxO0" => dcb_dispatch_indirect(arguments),
        "1DeUNpRIDDA" | "WrdP9Zxx3lQ" | "w8HVkEeXPv8" => 12,
        "qMlfB1ZhMDc" | "cxPZ4Wgvdj8" | "mStuvI0zOtc" => 20,
        "6DFuRKT4C9w" => 8,
        _ => return None,
    })
}

fn dispatch_draw_sync_command(
    nid: &str,
    arguments: [u64; 6],
    stack_arguments: [u64; 6],
) -> Option<u64> {
    Some(match nid {
        "aJf+j5yntiU" => event_write(arguments),
        "57labkp+rSQ" => dcb_acquire_mem(arguments, stack_arguments),
        "-vnlTPPXPrw" | "2ccJz9LQI+w" => 32,
        "1rZSWUv1IRc" => copy_data(arguments, stack_arguments, false),
        "WmAc2MEj6Io" => dma_data(arguments, stack_arguments),
        "xSAR0LTcRKM" => dcb_jump(arguments),
        "VEGu4dixjUg" => 16,
        "zfcxg-ewMK8" => dcb_rewind(arguments),
        "BIPexNBSGog" => cond_exec(arguments),
        "bbFueFP+J4k" => set_predication(arguments),
        "u2T2DiA5hRI" => stall_command_buffer_parser(arguments),
        "i1jyy49AjXU" => write_data(arguments, stack_arguments),
        "p9tI+yTvx68" | "rUuVjyR+Rd4" => low_u32(arguments[0])
            .wrapping_mul(4)
            .wrapping_add(16)
            .into(),
        "vuSXe69VILM" => dcb_get_lod_stats(arguments, stack_arguments),
        "VmW0Tdpy420" => dcb_wait_reg_mem(arguments, stack_arguments),
        "43WJ08sSugE" => match low_u32(arguments[0]) {
            0 => 56,
            1 => 64,
            _ => 0,
        },
        "+kSrjIVxKFE" | "QhCbS4X9Rl8" => push_marker(arguments),
        "H7uZqCoNuWk" => pop_marker(arguments),
        "QIXCsbipds0" | "+u6dKSLWM2o" => 8,
        "ou16V5hh5sg" => 20,
        "YUeqkyT7mEQ" => set_flip(arguments),
        _ => return None,
    })
}

fn dispatch_packet_patch(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "w6Dj1VJt5qY" => set_packet_predication(arguments),
        "n8vgpaQg6dA" => set_range_predication(arguments),
        "IxYiarKlXxM" => patch_dma_address(arguments, false),
        "cdDRpqcFGbU" => patch_dma_address(arguments, true),
        "2BS4EtAaF28" => patch_jump_target(arguments),
        "YWTKOju587o" | "3ZWa3AoyWZQ" => patch_cond_exec_address(arguments),
        "ORWsxIbk4TE" | "k-JpyR2dYAM" => patch_cond_exec_end(arguments),
        "3KDcnM3lrcU" => patch_wait_address(arguments),
        "n485EBnIWmk" => patch_wait_compare(arguments),
        "7nOoijNPvEU" => patch_wait_reference(arguments),
        "hXAnLgDHCoI" => patch_wait_mask(arguments),
        "0fWWK5uG9rQ" => patch_release_address(arguments),
        "J8YCgfKAMQs" => patch_release_gcr(arguments),
        "MlEw1feXcjg" => patch_release_data(arguments),
        "T9fjQIINoeE" => patch_release_type(arguments),
        "fPSCdQxgpSw" => patch_write_data_address(arguments),
        _ => return None,
    })
}

fn predication_bit(value: u64) -> u32 {
    u32::from(low_u8(value) == 1)
}

fn set_packet_predication(arguments: [u64; 6]) -> u64 {
    let packet = arguments[0];
    if packet == 0 || !packet.is_multiple_of(4) {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    let header = read_u32(packet);
    write_u32(packet, (header & !1) | predication_bit(arguments[1]));
    0
}

fn set_range_predication(arguments: [u64; 6]) -> u64 {
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
        let dword_count = if header & 0x3fff_ff00 == 0x3fff_1000 {
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

fn low_u32(value: u64) -> u32 {
    u32::try_from(value & u64::from(u32::MAX)).expect("value is masked to 32 bits")
}

fn low_u16(value: u64) -> u32 {
    low_u32(value) & 0xffff
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

fn packet_opcode(command: u64) -> Option<u32> {
    (command != 0 && command.is_multiple_of(4)).then(|| (read_u32(command) >> 8) & 0xff)
}

fn packet_register(command: u64) -> u32 {
    (read_u32(command) >> 2) & 0x3f
}

fn is_packet(command: u64, opcode: u32, register: Option<u32>) -> bool {
    packet_opcode(command).is_some_and(|actual| actual == opcode)
        && register.is_none_or(|expected| packet_register(command) == expected)
}

fn read_u8(address: u64) -> u8 {
    // SAFETY: import arguments are guest pointers into REmu's live in-process mapping.
    unsafe { std::ptr::read_unaligned(address as *const u8) }
}

fn read_u32(address: u64) -> u32 {
    // SAFETY: import arguments are guest pointers into REmu's live in-process mapping.
    unsafe { std::ptr::read_unaligned(address as *const u32) }
}

fn read_u64(address: u64) -> u64 {
    // SAFETY: import arguments are guest pointers into REmu's live in-process mapping.
    unsafe { std::ptr::read_unaligned(address as *const u64) }
}

fn write_u32(address: u64, value: u32) {
    // SAFETY: validated command-buffer addresses point into writable guest memory.
    unsafe { std::ptr::write_unaligned(address as *mut u32, value) }
}

fn write_u64(address: u64, value: u64) {
    // SAFETY: validated command-buffer addresses point into writable guest memory.
    unsafe { std::ptr::write_unaligned(address as *mut u64, value) }
}

fn write_words(address: u64, words: &[u32]) {
    for (index, &word) in words.iter().enumerate() {
        write_u32(
            address + u64::try_from(index).expect("word index fits") * 4,
            word,
        );
    }
}

fn allocate_dwords(command_buffer: u64, dword_count: u32) -> Option<u64> {
    if command_buffer == 0 || command_buffer & 7 != 0 || dword_count == 0 {
        return None;
    }
    let mut cursor_up = read_u64(command_buffer + COMMAND_BUFFER_CURSOR_UP_OFFSET);
    let mut cursor_down = read_u64(command_buffer + COMMAND_BUFFER_CURSOR_DOWN_OFFSET);
    let reserved = read_u32(command_buffer + COMMAND_BUFFER_RESERVED_DWORDS_OFFSET);
    if !has_capacity(cursor_up, cursor_down, reserved, dword_count) {
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
        if !has_capacity(cursor_up, cursor_down, reserved, dword_count) {
            return None;
        }
    }
    let byte_count = u64::from(dword_count).checked_mul(4)?;
    let next = cursor_up.checked_add(byte_count)?;
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

fn cb_nop(arguments: [u64; 6]) -> u64 {
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

fn cb_dispatch(arguments: [u64; 6]) -> u64 {
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

fn cb_branch(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
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
    command
}

fn cb_set_register_range(arguments: [u64; 6], opcode: u32) -> u64 {
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

fn cb_set_sh_registers(arguments: [u64; 6]) -> u64 {
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

fn cb_release_mem(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
    let action = low_u8(arguments[1]);
    let mut gcr_control = low_u16(arguments[2]);
    let destination = low_u8(arguments[3]);
    let cache_policy = low_u8(arguments[4]);
    let mut address = arguments[5];
    let data_selection = low_u8(stack[0]);
    let mut data = stack[1];
    let gds_offset = low_u16(stack[2]);
    let gds_size = low_u16(stack[3]);
    let interrupt = low_u8(stack[4]);
    let interrupt_context = low_u32(stack[5]);
    if destination > 1
        || cache_policy > 3
        || !matches!(data_selection, 0 | 1 | 2 | 3 | 5)
        || interrupt > 4
        || address & 3 != 0
    {
        return 0;
    }
    if gcr_control & 0x300 == 0x100 {
        gcr_control |= 0x200;
    }
    if interrupt == 4 {
        address = 0;
        data = 0;
    } else if data_selection == 5 {
        data = u64::from(gds_offset & 0xffff) | (u64::from(gds_size & 0xffff) << 16);
    }
    let Some(command) = allocate_dwords(arguments[0], 8) else {
        return 0;
    };
    let event_index = if action >= 0x2f { 6 } else { 5 };
    write_words(
        command,
        &[
            pm4(8, IT_NOP, R_RELEASE_MEM),
            (action & 0x3f)
                | (event_index << 8)
                | ((gcr_control & 0xfff) << 12)
                | (cache_policy << 25),
            (destination << 16) | (interrupt << 24) | (data_selection << 29),
            low_u32(address) & !3,
            low_u32(address >> 32),
            low_u32(data),
            low_u32(data >> 32),
            interrupt_context & 0x07ff_ffff,
        ],
    );
    command
}

fn acb_reset_queue(arguments: [u64; 6]) -> u64 {
    if low_u32(arguments[1]) & !0x1c2 != 0 {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], 2) else {
        return 0;
    };
    write_words(command, &[pm4(2, IT_NOP, R_DISPATCH_RESET), 0]);
    command
}

fn event_write(arguments: [u64; 6]) -> u64 {
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

fn acb_acquire_mem(arguments: [u64; 6]) -> u64 {
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

fn cond_exec(arguments: [u64; 6]) -> u64 {
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

fn acb_wait_reg_mem(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
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

fn acb_dispatch_indirect(arguments: [u64; 6]) -> u64 {
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

fn dcb_reset_queue(arguments: [u64; 6]) -> u64 {
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

fn dcb_wait_until_safe(arguments: [u64; 6]) -> u64 {
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

fn set_register_direct(arguments: [u64; 6], opcode: u32) -> u64 {
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

fn set_registers_indirect(arguments: [u64; 6], kind: IndirectRegisterKind) -> u64 {
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

fn set_index_size(arguments: [u64; 6]) -> u64 {
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

fn set_index_count(arguments: [u64; 6]) -> u64 {
    let Some(command) = allocate_dwords(arguments[0], 2) else {
        return 0;
    };
    write_words(
        command,
        &[pm4(2, IT_INDEX_BUFFER_SIZE, 0), low_u32(arguments[1])],
    );
    command
}

fn set_num_instances(arguments: [u64; 6]) -> u64 {
    let Some(command) = allocate_dwords(arguments[0], 2) else {
        return 0;
    };
    write_words(
        command,
        &[pm4(2, IT_NUM_INSTANCES, 0), low_u32(arguments[1])],
    );
    command
}

fn draw_index_initiator(modifier: u64) -> u32 {
    if modifier & (1_u64 << 32) == 0 {
        (low_u32(modifier) >> 3) & 0x20
    } else {
        0
    }
}

fn indirect_modifier_sgpr_base(modifier: u32) -> u32 {
    if matches!(modifier >> 29, 3 | 5) {
        0x10c
    } else {
        0x8c
    }
}

fn indirect_patch_offsets(modifier: u64, indexed: bool) -> u64 {
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

fn indirect_draw_initiator(modifier: u64) -> u32 {
    if modifier & (1_u64 << 32) == 0 {
        ((low_u32(modifier) >> 3) & 0x20) | 2
    } else {
        0
    }
}

fn draw_index(arguments: [u64; 6]) -> u64 {
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

fn draw_index_multi_instanced(arguments: [u64; 6]) -> u64 {
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

fn draw_index_auto(arguments: [u64; 6]) -> u64 {
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

fn draw_index_offset(arguments: [u64; 6]) -> u64 {
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

fn set_base_indirect_args(arguments: [u64; 6]) -> u64 {
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

fn draw_indirect(arguments: [u64; 6], indexed: bool) -> u64 {
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

fn draw_index_indirect_multi(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
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

fn dcb_dispatch_indirect(arguments: [u64; 6]) -> u64 {
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

fn dcb_acquire_mem(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
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

fn dcb_wait_reg_mem(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
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

fn acquire_mem(
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

fn wait_reg_mem(arguments: WaitRegMemArguments) -> u64 {
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

fn dma_data(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
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

fn copy_data(arguments: [u64; 6], stack: [u64; 6], acb: bool) -> u64 {
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

fn dcb_jump(arguments: [u64; 6]) -> u64 {
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

fn dcb_rewind(arguments: [u64; 6]) -> u64 {
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

fn set_predication(arguments: [u64; 6]) -> u64 {
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

fn stall_command_buffer_parser(arguments: [u64; 6]) -> u64 {
    let Some(command) = allocate_dwords(arguments[0], 2) else {
        return 0;
    };
    write_words(command, &[pm4(2, IT_PFP_SYNC_ME, 0), 0]);
    command
}

fn write_data(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
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

fn dcb_get_lod_stats(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
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

fn push_marker(arguments: [u64; 6]) -> u64 {
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

fn pop_marker(arguments: [u64; 6]) -> u64 {
    let Some(command) = allocate_dwords(arguments[0], 2) else {
        return 0;
    };
    write_words(command, &[pm4(2, IT_NOP, R_POP_MARKER), 0]);
    command
}

fn set_flip(arguments: [u64; 6]) -> u64 {
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

fn patch_indirect_address(arguments: [u64; 6], kind: IndirectRegisterKind) -> u64 {
    let command = arguments[0];
    let address = arguments[1];
    if address == 0 || address & 3 != 0 || !is_packet(command, kind.opcode(), None) {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    write_u32(
        command + 4,
        (read_u32(command + 4) & 3) | (low_u32(address) & !3),
    );
    write_u32(command + 8, low_u32(address >> 32));
    0
}

fn patch_indirect_count(arguments: [u64; 6], kind: IndirectRegisterKind, add: bool) -> u64 {
    let command = arguments[0];
    if !is_packet(command, kind.opcode(), None) {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    let count = low_u32(arguments[1]);
    if count > 0x3fff {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    let previous = read_u32(command + 16);
    let next = if add {
        (previous & 0x3fff).wrapping_add(count) & 0x3fff
    } else {
        count
    };
    write_u32(command + 16, (previous & !0x3fff) | next);
    0
}

fn patch_dma_address(arguments: [u64; 6], source: bool) -> u64 {
    let command = arguments[0];
    if !is_packet(command, IT_DMA_DATA, None) {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    write_u64(command + if source { 8 } else { 16 }, arguments[1]);
    0
}

fn patch_jump_target(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let target = arguments[1];
    let count = low_u32(arguments[2]);
    if target == 0
        || target & 3 != 0
        || count > 0x000f_ffff
        || !is_packet(command, IT_INDIRECT_BUFFER, None)
    {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    write_u32(command + 4, low_u32(target));
    write_u32(
        command + 8,
        (read_u32(command + 8) & 0xffff_0000) | (low_u32(target >> 32) & 0xffff),
    );
    write_u32(command + 12, (read_u32(command + 12) & 0xfff0_0000) | count);
    0
}

fn patch_cond_exec_address(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let address = arguments[1];
    if address == 0 || address & 3 != 0 || !is_packet(command, IT_COND_EXEC, None) {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    write_u32(
        command + 4,
        (read_u32(command + 4) & 3) | (low_u32(address) & !3),
    );
    write_u32(command + 8, low_u32(address >> 32));
    0
}

fn patch_cond_exec_end(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let end = arguments[1];
    if end == 0 || end & 3 != 0 || !is_packet(command, IT_COND_EXEC, None) {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    let Some(packet_end) = command.checked_add(20) else {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    };
    let Some(bytes) = end.checked_sub(packet_end) else {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    };
    let count = bytes / 4;
    if count > 0x3fff {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    write_u32(
        command + 16,
        (read_u32(command + 16) & !0x3fff) | low_u32(count),
    );
    0
}

fn wait_kind(command: u64) -> Option<(u64, u64, u64)> {
    if is_packet(command, IT_NOP, Some(R_WAIT_MEM_32)) {
        Some((20, 16, 12))
    } else if is_packet(command, IT_NOP, Some(R_WAIT_MEM_64)) {
        Some((28, 20, 12))
    } else {
        None
    }
}

fn patch_wait_address(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let address = arguments[1];
    let header = super::try_read_process_guest_u32(command);
    let is_wait_32 = header.is_some_and(|header| {
        (header >> 8) & 0xff == IT_NOP && (header >> 2) & 0x3f == R_WAIT_MEM_32
    });
    let is_wait_64 = header.is_some_and(|header| {
        (header >> 8) & 0xff == IT_NOP && (header >> 2) & 0x3f == R_WAIT_MEM_64
    });
    let alignment = if is_wait_32 {
        4
    } else if is_wait_64 {
        8
    } else {
        // Newer Gen5 SDKs pass (address-or-offset, pointer-to-address-field). The first argument
        // may be a GPU-only virtual address in the host's PROT_NONE aperture, so never probe it as
        // a packet. The second argument is the CPU-writable field returned by the packet builder.
        if command == 0
            || command & 3 != 0
            || address == 0
            || !address.is_multiple_of(4)
            || !super::process_guest_range_is_readable(address, 8)
        {
            return GRAPHICS5_ERROR_INVALID_PACKET;
        }
        write_u64(address, command);
        return 0;
    };
    if address == 0 || address & (alignment - 1) != 0 || address >> 50 != 0 {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    let alignment = u32::try_from(alignment).expect("wait alignment fits in u32");
    write_u32(command + 4, low_u32(address) & !(alignment - 1));
    write_u32(command + 8, low_u32(address >> 32) & 0x3_ffff);
    0
}

fn patch_wait_compare(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let compare = low_u8(arguments[1]);
    let Some((control_offset, _, _)) = wait_kind(command) else {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    };
    if compare > 7 {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    write_u32(
        command + control_offset,
        (read_u32(command + control_offset) & !7) | compare,
    );
    0
}

fn patch_wait_reference(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let Some((_, reference_offset, _)) = wait_kind(command) else {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    };
    if is_packet(command, IT_NOP, Some(R_WAIT_MEM_32)) {
        write_u32(command + reference_offset, low_u32(arguments[1]));
    } else {
        write_u64(command + reference_offset, arguments[1]);
    }
    0
}

fn patch_wait_mask(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let Some((_, _, mask_offset)) = wait_kind(command) else {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    };
    if is_packet(command, IT_NOP, Some(R_WAIT_MEM_32)) {
        write_u32(command + mask_offset, low_u32(arguments[1]));
    } else {
        write_u64(command + mask_offset, arguments[1]);
    }
    0
}

fn release_packet_kind(command: u64) -> Option<u32> {
    let opcode = packet_opcode(command)?;
    ((opcode == IT_NOP && packet_register(command) == R_RELEASE_MEM)
        || matches!(opcode, IT_RELEASE_MEM | IT_EVENT_WRITE_EOP))
    .then_some(opcode)
}

fn patch_release_address(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let address = arguments[1];
    let Some(opcode) = release_packet_kind(command) else {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    };
    if address & 3 != 0 {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    if opcode == IT_EVENT_WRITE_EOP {
        write_u32(command + 8, low_u32(address));
        write_u32(
            command + 12,
            (read_u32(command + 12) & 0xffff_0000) | (low_u32(address >> 32) & 0xffff),
        );
    } else {
        write_u64(command + 12, address);
    }
    0
}

fn patch_release_gcr(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let gcr = low_u16(arguments[1]);
    if gcr > 0xfff || !is_packet(command, IT_NOP, Some(R_RELEASE_MEM)) {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    write_u32(
        command + 4,
        (read_u32(command + 4) & !0x00ff_f000) | (gcr << 12),
    );
    0
}

fn patch_release_data(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let Some(opcode) = release_packet_kind(command) else {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    };
    let mut data = arguments[3];
    if opcode == IT_NOP
        && packet_register(command) == R_RELEASE_MEM
        && low_u32(arguments[1]) > 1
        && low_u32(arguments[2]) == 1
    {
        data = (u64::from(low_u32(arguments[1]) - 2) << 24) | (data & 0x00ff_ffff);
    }
    write_u64(
        command + if opcode == IT_EVENT_WRITE_EOP { 16 } else { 20 },
        data,
    );
    0
}

fn patch_release_type(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let data_selection = low_u8(arguments[1]);
    if data_selection > 7 || !is_packet(command, IT_NOP, Some(R_RELEASE_MEM)) {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    write_u32(
        command + 8,
        (read_u32(command + 8) & !(7 << 29)) | (data_selection << 29),
    );
    0
}

fn patch_write_data_address(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    if !is_packet(command, IT_WRITE_DATA, None) || arguments[1] & 3 != 0 {
        return GRAPHICS5_ERROR_INVALID_PACKET;
    }
    write_u64(command + 8, arguments[1]);
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CommandFixture {
        descriptor: Box<[u64; 7]>,
        words: Box<[u32; 128]>,
    }

    impl CommandFixture {
        fn new(reserved: u32) -> Self {
            let descriptor = Box::new([0_u64; 7]);
            let words = Box::new([0_u32; 128]);
            let mut fixture = Self { descriptor, words };
            let start = fixture.words.as_mut_ptr().addr() as u64;
            fixture.descriptor[0] = start;
            fixture.descriptor[1] = start + fixture.words.len() as u64 * 4;
            fixture.descriptor[2] = start;
            fixture.descriptor[3] = start + fixture.words.len() as u64 * 4;
            fixture.descriptor[6] = u64::from(reserved);
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

    fn call(nid: &str, arguments: [u64; 6], stack: [u64; 6]) -> u64 {
        dispatch_agc_command_abi(nid, arguments, stack).expect("known test NID")
    }

    #[test]
    fn returns_exact_fixed_and_variable_packet_sizes_in_bytes() {
        assert_eq!(call("Abendgtz+3o", [0; 6], [0; 6]), 20);
        assert_eq!(call("-vnlTPPXPrw", [0; 6], [0; 6]), 32);
        assert_eq!(call("6ee9Hd3EWXQ", [0; 6], [0; 6]), 24);
        assert_eq!(call("t7PlZ9nt5Lc", args(&[13]), [0; 6]), 52);
        assert_eq!(call("bxGoVxpdSPQ", args(&[7]), [0; 6]), 36);
        assert_eq!(call("p9tI+yTvx68", args(&[9]), [0; 6]), 52);
        assert_eq!(
            dispatch_agc_command_abi("not-an-agc-nid", [0; 6], [0; 6]),
            None
        );
    }

    #[test]
    fn builds_native_register_and_draw_packets() {
        let mut fixture = CommandFixture::new(2);
        let buffer = fixture.address();
        let register = u64::from(0xdead_beef_u32) << 32 | 0x191;

        assert_eq!(
            call("LHFXRrlTPD8", args(&[buffer, register]), [0; 6]),
            fixture.start()
        );
        assert_eq!(
            &fixture.words[..3],
            &[pm4(3, IT_SET_CONTEXT_REG, 0), 0x191, 0xdead_beef]
        );

        let registers = [0x123_u32, 0x456, 0x124, 0x789];
        let register_address = registers.as_ptr().addr() as u64;
        assert_eq!(
            call("-HOOCn0JY48", args(&[buffer, register_address, 2]), [0; 6]),
            fixture.start() + 12
        );
        assert_eq!(
            &fixture.words[3..8],
            &[
                pm4(5, IT_SET_SH_REG_INDIRECT, 0),
                low_u32(register_address) & !3,
                low_u32(register_address >> 32),
                0x8000_0000,
                2,
            ]
        );

        assert_eq!(
            call("Yw0jKSqop+E", args(&[buffer, 6, 0]), [0; 6]),
            fixture.start() + 32
        );
        assert_eq!(
            &fixture.words[8..11],
            &[pm4(3, IT_DRAW_INDEX_AUTO, 0), 6, 2]
        );
        assert_eq!(fixture.cursor(), fixture.start() + 44);
    }

    #[test]
    fn builds_native_type_two_packet() {
        let mut fixture = CommandFixture::new(0);
        let buffer = fixture.address();

        assert_eq!(
            call("qj7QZpgr9Uw", args(&[buffer]), [0; 6]),
            fixture.start()
        );
        assert_eq!(fixture.words[0], 0x8000_0000);
        assert_eq!(fixture.cursor(), fixture.start() + 4);
    }

    #[test]
    fn patches_single_and_range_packet_predication_bits() {
        let mut packet = [pm4(3, IT_SET_SH_REG, 0), 0x123, 0x456];
        let address = packet.as_mut_ptr().addr() as u64;

        assert_eq!(call("w6Dj1VJt5qY", args(&[address, 0x101]), [0; 6]), 0);
        assert_eq!(packet[0], pm4(3, IT_SET_SH_REG, 0) | 1);
        assert_eq!(call("w6Dj1VJt5qY", args(&[address, 2]), [0; 6]), 0);
        assert_eq!(packet[0], pm4(3, IT_SET_SH_REG, 0));
        assert_eq!(
            call("w6Dj1VJt5qY", args(&[0, 1]), [0; 6]),
            GRAPHICS5_ERROR_INVALID_PACKET
        );

        let mut range = [
            pm4(2, IT_CLEAR_STATE, 0),
            0x0f,
            pm4(3, IT_SET_SH_REG, 0),
            0x123,
            0x456,
        ];
        let start = range.as_mut_ptr().addr() as u64;
        let end = start + u64::try_from(range.len() * 4).expect("range byte count fits u64");
        assert_eq!(call("n8vgpaQg6dA", args(&[start, end, 1]), [0; 6]), 0);
        assert_eq!(range[0], pm4(2, IT_CLEAR_STATE, 0) | 1);
        assert_eq!(range[2], pm4(3, IT_SET_SH_REG, 0) | 1);
        assert_eq!(&range[1..2], &[0x0f]);
        assert_eq!(&range[3..], &[0x123, 0x456]);
    }

    #[test]
    fn reserves_a_register_range_payload_when_values_are_null() {
        let mut fixture = CommandFixture::new(0);
        fixture.words[2..6].copy_from_slice(&[0x1111, 0x2222, 0x3333, 0x4444]);
        let buffer = fixture.address();

        assert_eq!(
            call("n2fD4A+pb+g", args(&[buffer, 0x8c, 0, 4]), [0; 6]),
            fixture.start()
        );
        assert_eq!(fixture.words[0], pm4(6, IT_SET_SH_REG, 0));
        assert_eq!(fixture.words[1], 0x8c);
        assert_eq!(&fixture.words[2..6], &[0x1111, 0x2222, 0x3333, 0x4444]);
        assert_eq!(fixture.cursor(), fixture.start() + 24);
    }

    #[test]
    fn preserves_indirect_register_packet_fields_while_patching() {
        let mut fixture = CommandFixture::new(0);
        let buffer = fixture.address();
        let registers = [0_u64; 4];
        let first = registers.as_ptr().addr() as u64;
        let command = call("ZvwO9euwYzc", args(&[buffer, first, 7]), [0; 6]);
        fixture.words[4] |= 0xc000_0000;
        let second = first + 0x20;

        assert_eq!(call("vcmNN+AAXnY", args(&[command, second]), [0; 6]), 0);
        assert_eq!(call("d-6uF9sZDIU", args(&[command, 9]), [0; 6]), 0);
        assert_eq!(fixture.words[1], low_u32(second) & !3);
        assert_eq!(fixture.words[2], low_u32(second >> 32));
        assert_eq!(fixture.words[4], 0xc000_0010);
        assert_eq!(
            call("Qrj4c+61z4A", args(&[command, second]), [0; 6]),
            GRAPHICS5_ERROR_INVALID_PACKET
        );
    }

    #[test]
    fn builds_and_patches_native_dma_data() {
        let mut fixture = CommandFixture::new(0);
        let buffer = fixture.address();
        let command = call(
            "WmAc2MEj6Io",
            args(&[buffer, 1, 2, 3, 0x1122_3344_5566_7788, 1]),
            args(&[2, 0x8877_6655_4433_2211, 0x3456, 1, 1, 1]),
        );
        assert_eq!(command, fixture.start());
        assert_eq!(fixture.words[0], pm4(7, IT_DMA_DATA, 0));
        assert_eq!(
            &fixture.words[2..6],
            &[0x4433_2211, 0x8877_6655, 0x5566_7788, 0x1122_3344]
        );
        assert_eq!(
            call(
                "IxYiarKlXxM",
                args(&[command, 0xaabb_ccdd_eeff_0010]),
                [0; 6]
            ),
            0
        );
        assert_eq!(
            call(
                "cdDRpqcFGbU",
                args(&[command, 0x1234_5678_9abc_def0]),
                [0; 6]
            ),
            0
        );
        assert_eq!(
            &fixture.words[2..6],
            &[0x9abc_def0, 0x1234_5678, 0xeeff_0010, 0xaabb_ccdd]
        );
    }

    #[test]
    fn builds_and_patches_64_bit_wait_packet() {
        let mut fixture = CommandFixture::new(0);
        let buffer = fixture.address();
        let wait_address = 0x1234_5678_9abc_def0;
        let command = call(
            "VmW0Tdpy420",
            args(&[buffer, 1, 2, 5, 3, wait_address]),
            args(&[0x1111_2222_3333_4444, 0xffff_ffff_0000_0000, 0x12345]),
        );
        assert_eq!(command, fixture.start());
        assert_eq!(fixture.words[0], pm4(9, IT_NOP, R_WAIT_MEM_64));
        assert_eq!(fixture.words[8], 0x1234);

        assert_eq!(call("n485EBnIWmk", args(&[command, 7]), [0; 6]), 0);
        assert_eq!(
            call(
                "7nOoijNPvEU",
                args(&[command, 0xaaaa_bbbb_cccc_dddd]),
                [0; 6]
            ),
            0
        );
        assert_eq!(
            call(
                "hXAnLgDHCoI",
                args(&[command, 0x0123_4567_89ab_cdef]),
                [0; 6]
            ),
            0
        );
        assert_eq!(fixture.words[7] & 7, 7);
        assert_eq!(
            &fixture.words[3..7],
            &[0x89ab_cdef, 0x0123_4567, 0xcccc_dddd, 0xaaaa_bbbb]
        );
        assert_eq!(
            call("3KDcnM3lrcU", args(&[command, wait_address | 4]), [0; 6]),
            GRAPHICS5_ERROR_INVALID_PACKET
        );
    }

    #[test]
    fn release_mem_and_patchers_share_the_native_layout() {
        let mut fixture = CommandFixture::new(0);
        let buffer = fixture.address();
        let command = call(
            "wr23dPKyWc0",
            args(&[buffer, 0x2f, 0x100, 1, 2, 0x1000]),
            args(&[1, 0x1122_3344_5566_7788, 0, 0, 2, 0x123]),
        );
        assert_eq!(command, fixture.start());
        assert_eq!(fixture.words[0], pm4(8, IT_NOP, R_RELEASE_MEM));
        assert_eq!(fixture.words[1] & 0x00ff_f000, 0x0030_0000);

        assert_eq!(call("J8YCgfKAMQs", args(&[command, 0x456]), [0; 6]), 0);
        assert_eq!(call("T9fjQIINoeE", args(&[command, 3]), [0; 6]), 0);
        assert_eq!(call("0fWWK5uG9rQ", args(&[command, 0x2000]), [0; 6]), 0);
        assert_eq!(
            call("MlEw1feXcjg", args(&[command, 2, 1, 0x00ab_cdef]), [0; 6]),
            0
        );
        assert_eq!(fixture.words[1] & 0x00ff_f000, 0x0045_6000);
        assert_eq!(fixture.words[2] >> 29, 3);
        assert_eq!(&fixture.words[3..7], &[0x2000, 0, 0x00ab_cdef, 0]);
    }

    #[test]
    fn command_buffer_reservation_and_alignment_are_enforced() {
        let mut fixture = CommandFixture::new(127);
        let buffer = fixture.address();
        let start = fixture.start();
        assert_eq!(call("Yw0jKSqop+E", args(&[buffer, 3, 0]), [0; 6]), 0);
        assert_eq!(fixture.cursor(), start);

        fixture.descriptor[6] = 0;
        fixture.descriptor[3] = start + 8;
        assert_eq!(call("Yw0jKSqop+E", args(&[buffer, 3, 0]), [0; 6]), 0);
        assert_eq!(fixture.cursor(), start);

        fixture.descriptor[3] = start + 12;
        assert_eq!(call("Yw0jKSqop+E", args(&[buffer, 3, 0]), [0; 6]), start);
        assert_eq!(fixture.cursor(), start + 12);
        assert_eq!(call("Yw0jKSqop+E", args(&[buffer + 4, 3, 0]), [0; 6]), 0);
    }

    #[test]
    fn invokes_the_gen5_growth_callback_before_allocating() {
        unsafe extern "C" fn grow(command_buffer: u64, requested: u32, user_data: u64) -> bool {
            if requested != 7 {
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

        let mut fixture = CommandFixture::new(2);
        let buffer = fixture.address();
        let start = fixture.start();
        let extent = Box::new([start, start + 28]);
        fixture.descriptor[3] = start + 4;
        fixture.descriptor[4] = grow as *const () as usize as u64;
        fixture.descriptor[5] = extent.as_ptr().addr() as u64;

        assert_eq!(
            call("k3GhuSNmBLU", args(&[buffer, 2, 3, 4, 0]), [0; 6]),
            start
        );
        assert_eq!(fixture.cursor(), start + 20);
        assert_eq!(fixture.words[0], pm4(5, IT_DISPATCH_DIRECT, 0));
    }

    #[test]
    fn four_argument_index_buffer_builder_remains_stateful() {
        assert_eq!(
            dispatch_agc_command_abi("l4fM9K-Lyks", [0; 6], [0; 6]),
            None
        );
        assert_eq!(
            dispatch_agc_command_abi("j4emHHndCPY", [0; 6], [0; 6]),
            None
        );
    }

    #[test]
    fn writes_inline_data_and_marker_payloads() {
        let mut fixture = CommandFixture::new(0);
        let buffer = fixture.address();
        let data = [0x1122_3344_u32, 0xaabb_ccdd];
        let command = call(
            "i1jyy49AjXU",
            args(&[buffer, 2, 1, 0x4000, data.as_ptr().addr() as u64, 2]),
            args(&[1, 1]),
        );
        assert_eq!(command, fixture.start());
        assert_eq!(fixture.words[0], pm4(6, IT_WRITE_DATA, 0));
        assert_eq!(&fixture.words[4..6], &data);

        let marker = b"frame-start\0";
        let marker_command = call(
            "+kSrjIVxKFE",
            args(&[buffer, marker.as_ptr().addr() as u64]),
            [0; 6],
        );
        assert_eq!(marker_command, fixture.start() + 24);
        let payload =
            unsafe { std::slice::from_raw_parts((marker_command + 4) as *const u8, marker.len()) };
        assert_eq!(payload, marker);
    }
}
