//! Guest-facing Gen5 AGC command-buffer construction ABI.
//!
//! The packet layouts in this module are the native Gen5 encodings used by the
//! open-source `KytyPS5` implementation.  Keeping construction and patching in one
//! place is important: patch functions validate the packet opcode they mutate,
//! and every `GetSize` result is tied to the corresponding encoder.

mod async_queue;
mod buffer;
mod common;
mod draw;
mod patch;
mod sync;
use self::async_queue::{
    acb_acquire_mem, acb_dispatch_indirect, acb_dma_data, acb_reset_queue, acb_wait_reg_mem,
    cond_exec, event_write,
};
#[cfg(test)]
use self::buffer::read_u64;
use self::buffer::{
    allocate_dwords, is_packet, low_u8, low_u16, low_u32, mark_async_command_buffer,
    mark_graphics_command_buffer, packet_opcode, packet_register, pm4, read_u8, read_u32,
    register_graphics_continuation, trace_failed_sync_command, write_u32, write_u64, write_words,
};
pub(crate) use self::buffer::{
    consume_recorded_graphics_range, graphics_continuation_was_consumed,
    retire_pending_graphics_continuation, take_graphics_continuations_before,
    take_recorded_graphics_prefix, track_graphics_command_allocation, track_graphics_submission,
};
#[cfg(test)]
pub(crate) use self::buffer::{graphics_tracking_test_guard, track_graphics_allocation};
use self::common::{
    cb_branch, cb_dispatch, cb_nop, cb_release_mem, cb_set_register_range, cb_set_sh_registers,
    type_two_packet,
};
use self::draw::{
    dcb_dispatch_indirect, dcb_reset_queue, dcb_wait_until_safe, draw_index, draw_index_auto,
    draw_index_indirect_multi, draw_index_multi_instanced, draw_index_offset, draw_indirect,
    set_base_indirect_args, set_index_count, set_index_size, set_num_instances,
    set_register_direct, set_registers_indirect,
};
use self::patch::{
    patch_cond_exec_address, patch_cond_exec_end, patch_dma_address, patch_indirect_address,
    patch_indirect_count, patch_jump_target, patch_release_address, patch_release_data,
    patch_release_gcr, patch_release_type, patch_wait_address, patch_wait_compare, patch_wait_mask,
    patch_wait_reference, patch_write_data_address,
};
use self::sync::{
    acquire_mem, copy_data, dcb_acquire_mem, dcb_get_lod_stats, dcb_jump, dcb_rewind,
    dcb_wait_reg_mem, dma_data, pop_marker, push_marker, set_flip, set_packet_predication,
    set_predication, set_range_predication, stall_command_buffer_parser, wait_reg_mem, write_data,
};

const GRAPHICS5_ERROR_INVALID_PACKET: u64 = 0x8a6c_000c;

const COMMAND_BUFFER_BASE_UP_OFFSET: u64 = 0x00;
const COMMAND_BUFFER_BASE_DOWN_OFFSET: u64 = 0x08;
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
        "wr23dPKyWc0" => trace_failed_sync_command(
            "sceAgcCbReleaseMem",
            arguments,
            stack_arguments,
            cb_release_mem(arguments, stack_arguments),
        ),
        "hL7C0IRpWZI" => 32,
        _ => return None,
    })
}

fn dispatch_async_command(
    nid: &str,
    arguments: [u64; 6],
    stack_arguments: [u64; 6],
) -> Option<u64> {
    if !matches!(
        nid,
        "JrtiDtKeS38"
            | "cFazmnXpJOE"
            | "KT-hTp-Ch14"
            | "ewobAQeMo5k"
            | "M0ttm8h7SKA"
            | "qyM2bxYFPAk"
            | "ozKzBP4aki4"
            | "htn36gPnBk4"
            | "-RnpfpxIhec"
            | "qzMN2XKGA4k"
            | "j3EtxFkSIhQ"
            | "eZ4+17OQz4Q"
            | "xAeBOa0A3kk"
            | "cpCILPya5Zk"
            | "6mFxkVqdmbQ"
    ) {
        return None;
    }
    mark_async_command_buffer(arguments[0]);
    Some(match nid {
        "JrtiDtKeS38" => acb_reset_queue(arguments),
        "cFazmnXpJOE" => event_write(arguments),
        "KT-hTp-Ch14" => acb_acquire_mem(arguments),
        "ewobAQeMo5k" | "M0ttm8h7SKA" => 32,
        "qyM2bxYFPAk" => cond_exec(arguments),
        "ozKzBP4aki4" => 20,
        "htn36gPnBk4" => trace_failed_sync_command(
            "sceAgcAcbWaitRegMem",
            arguments,
            stack_arguments,
            acb_wait_reg_mem(arguments, stack_arguments),
        ),
        "-RnpfpxIhec" => trace_failed_sync_command(
            "sceAgcAcbDmaData",
            arguments,
            stack_arguments,
            acb_dma_data(arguments, stack_arguments),
        ),
        "qzMN2XKGA4k" => trace_failed_sync_command(
            "sceAgcAcbCopyData",
            arguments,
            stack_arguments,
            copy_data(arguments, stack_arguments, true),
        ),
        "j3EtxFkSIhQ" => acb_dispatch_indirect(arguments),
        "eZ4+17OQz4Q" => trace_failed_sync_command(
            "sceAgcAcbWriteData",
            arguments,
            stack_arguments,
            write_data(arguments, stack_arguments),
        ),
        "xAeBOa0A3kk" | "cpCILPya5Zk" => push_marker(arguments),
        "6mFxkVqdmbQ" => pop_marker(arguments),
        _ => unreachable!("the ACB NID set was checked above"),
    })
}

fn dispatch_draw_command(nid: &str, arguments: [u64; 6], stack_arguments: [u64; 6]) -> Option<u64> {
    if !matches!(
        nid,
        "TRO721eVt4g"
            | "MWiElSNE8j8"
            | "pFLArOT53+w"
            | "LHFXRrlTPD8"
            | "w4-d0n60hdo"
            | "ZvwO9euwYzc"
            | "-HOOCn0JY48"
            | "hvUfkUIQcOE"
            | "GIIW2J37e70"
            | "8N2tmT3jmC8"
            | "mljzuGDZRQ4"
            | "tSBxhAPyytQ"
            | "q88lQ+GP5Yk"
            | "6ee9Hd3EWXQ"
            | "Rlx+bykm0r0"
            | "mR9j7+SfM34"
            | "Yw0jKSqop+E"
            | "B+aG9DUnTKA"
            | "RmaJwLtc8rY"
            | "1q1titRBL6o"
            | "t1vNu082-jM"
            | "ypVBz4uPKcQ"
            | "CtB+A9-VxO0"
            | "1DeUNpRIDDA"
            | "WrdP9Zxx3lQ"
            | "w8HVkEeXPv8"
            | "qMlfB1ZhMDc"
            | "cxPZ4Wgvdj8"
            | "mStuvI0zOtc"
            | "6DFuRKT4C9w"
    ) {
        return None;
    }
    mark_graphics_command_buffer(arguments[0]);
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
        _ => unreachable!("the DCB NID set was checked above"),
    })
}

fn dispatch_draw_sync_command(
    nid: &str,
    arguments: [u64; 6],
    stack_arguments: [u64; 6],
) -> Option<u64> {
    if !matches!(
        nid,
        "aJf+j5yntiU"
            | "57labkp+rSQ"
            | "-vnlTPPXPrw"
            | "2ccJz9LQI+w"
            | "1rZSWUv1IRc"
            | "WmAc2MEj6Io"
            | "xSAR0LTcRKM"
            | "VEGu4dixjUg"
            | "zfcxg-ewMK8"
            | "BIPexNBSGog"
            | "bbFueFP+J4k"
            | "u2T2DiA5hRI"
            | "i1jyy49AjXU"
            | "p9tI+yTvx68"
            | "rUuVjyR+Rd4"
            | "vuSXe69VILM"
            | "VmW0Tdpy420"
            | "43WJ08sSugE"
            | "+kSrjIVxKFE"
            | "QhCbS4X9Rl8"
            | "H7uZqCoNuWk"
            | "QIXCsbipds0"
            | "+u6dKSLWM2o"
            | "ou16V5hh5sg"
            | "YUeqkyT7mEQ"
    ) {
        return None;
    }
    mark_graphics_command_buffer(arguments[0]);
    Some(match nid {
        "aJf+j5yntiU" => event_write(arguments),
        "57labkp+rSQ" => dcb_acquire_mem(arguments, stack_arguments),
        "-vnlTPPXPrw" | "2ccJz9LQI+w" => 32,
        "1rZSWUv1IRc" => trace_failed_sync_command(
            "sceAgcDcbCopyData",
            arguments,
            stack_arguments,
            copy_data(arguments, stack_arguments, false),
        ),
        "WmAc2MEj6Io" => trace_failed_sync_command(
            "sceAgcDcbDmaData",
            arguments,
            stack_arguments,
            dma_data(arguments, stack_arguments),
        ),
        "xSAR0LTcRKM" => dcb_jump(arguments),
        "VEGu4dixjUg" => 16,
        "zfcxg-ewMK8" => dcb_rewind(arguments),
        "BIPexNBSGog" => cond_exec(arguments),
        "bbFueFP+J4k" => set_predication(arguments),
        "u2T2DiA5hRI" => stall_command_buffer_parser(arguments),
        "i1jyy49AjXU" => trace_failed_sync_command(
            "sceAgcDcbWriteData",
            arguments,
            stack_arguments,
            write_data(arguments, stack_arguments),
        ),
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
        _ => unreachable!("the DCB synchronization NID set was checked above"),
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

#[cfg(test)]
mod tests;
