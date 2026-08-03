// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! `libSceVideodec2` compute-queue and decoder entry points.
//!
//! Titles size a shared CPU/GPU workspace, allocate a compute queue from it, and attach that
//! queue to a decoder. Decode itself is in-process libavcodec (plan P5) for AVC/HEVC elementary
//! streams. Struct layouts and error codes come from SDK `videodec2.h` / `videodec2/error.h`
//! (cite file:line in NOTES). Offline compute-queue behaviour is REmu's: query reports a fixed
//! workspace size, allocate returns the caller-supplied workspace pointer as the queue opaque.

use crate::av_player_libav::LibavCodecSession;
use crate::{
    ProbeDispatchState, process_guest_range_is_readable, read_guest_u32, read_guest_u64,
    sce_kernel_error, write_guest_u32, write_guest_u64, write_guest_u8,
};

fn low_u8(value: u64) -> u8 {
    value as u8
}

fn low_u16(value: u64) -> u16 {
    value as u16
}

const VIDEODEC2_ERROR_STRUCT_SIZE: u32 = 0x811d_0101;
const VIDEODEC2_ERROR_ARGUMENT_POINTER: u32 = 0x811d_0102;
const VIDEODEC2_ERROR_DECODER_INSTANCE: u32 = 0x811d_0103;
const VIDEODEC2_ERROR_MEMORY_POINTER: u32 = 0x811d_0105;
const VIDEODEC2_ERROR_FRAME_BUFFER_SIZE: u32 = 0x811d_0106;
const VIDEODEC2_ERROR_FRAME_BUFFER_POINTER: u32 = 0x811d_0107;
const VIDEODEC2_ERROR_COMPUTE_QUEUE: u32 = 0x811d_0110;
const VIDEODEC2_ERROR_CONFIG_INFO: u32 = 0x811d_0200;
const VIDEODEC2_ERROR_COMPUTE_PIPE_ID: u32 = 0x811d_0201;
const VIDEODEC2_ERROR_COMPUTE_QUEUE_ID: u32 = 0x811d_0202;
const VIDEODEC2_ERROR_CODEC_TYPE: u32 = 0x811d_0204;

/// `sizeof(OrbisVideodec2ComputeMemoryInfo)`.
const COMPUTE_MEMORY_INFO_SIZE: u64 = 0x18;
/// `sizeof(OrbisVideodec2ComputeConfigInfo)`.
const COMPUTE_CONFIG_INFO_SIZE: u64 = 0x10;
/// `sizeof(SceVideodec2DecoderConfigInfo)` — sdk `videodec2.h:72-90`.
const DECODER_CONFIG_INFO_SIZE: u64 = 0x48;
/// `sizeof(SceVideodec2DecoderMemoryInfo)` — sdk `videodec2.h:92-103` (8×size_t/ptr + u32 + u32).
const DECODER_MEMORY_INFO_SIZE: u64 = 0x48;
/// `sizeof(SceVideodec2InputData)` — sdk `videodec2.h:115-122`.
const INPUT_DATA_SIZE: u64 = 0x30;
/// `sizeof(SceVideodec2OutputInfo)` — sdk `videodec2.h:124-138`.
const OUTPUT_INFO_SIZE: u64 = 0x38;
/// `sizeof(SceVideodec2FrameBuffer)` — sdk `videodec2.h:140-145` (with trailing pad).
const FRAME_BUFFER_SIZE: u64 = 0x20;

/// Workspace size reported when a title has not yet attached decoder configuration.
const MINIMUM_COMPUTE_MEMORY_SIZE: u64 = 16 * 1024 * 1024;
const MINIMUM_DECODER_CPU_MEMORY: u64 = 4 * 1024 * 1024;
const MINIMUM_FRAME_BUFFER_SIZE: u64 = 8 * 1024 * 1024;

/// SDK `videodec2/avc.h:10`.
const CODEC_TYPE_AVC: u32 = 1;
/// SDK `videodec2/hevc.h:16`.
const CODEC_TYPE_HEVC: u32 = 974_921;

#[derive(Debug)]
pub(crate) struct Videodec2Decoder {
    codec_type: u32,
    session: LibavCodecSession,
    next_handle_stamp: u64,
}

/// Handles a Videodec2 NID, or returns `None` so the caller can keep looking.
pub(crate) fn dispatch(
    nid: &str,
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> Option<u64> {
    Some(match nid {
        "RnDibcGCPKw" => query_compute_memory_info(arguments[0]),
        "eD+X2SmxUt4" => allocate_compute_queue(arguments, state),
        "UvtA3FAiF4Y" => release_compute_queue(arguments[0], state),
        "qqMCwlULR+E" => query_decoder_memory_info(arguments),
        "CNNRoRYd8XI" => create_decoder(arguments, state),
        "jwImxXRGSKA" => delete_decoder(arguments[0], state),
        "852F5+q6+iM" => decode(arguments, state),
        "l1hXwscLuCY" => flush(arguments, state),
        "wJXikG6QFN8" => reset(arguments[0], state),
        _ => return None,
    })
}

/// `sceVideodec2QueryComputeMemoryInfo(OrbisVideodec2ComputeMemoryInfo*)`.
fn query_compute_memory_info(info: u64) -> u64 {
    if info == 0 || !process_guest_range_is_readable(info, COMPUTE_MEMORY_INFO_SIZE as usize) {
        return sce_kernel_error(VIDEODEC2_ERROR_ARGUMENT_POINTER);
    }
    let this_size = read_guest_u64(info);
    if this_size != COMPUTE_MEMORY_INFO_SIZE {
        return sce_kernel_error(VIDEODEC2_ERROR_STRUCT_SIZE);
    }
    write_guest_u64(info + 0x08, MINIMUM_COMPUTE_MEMORY_SIZE);
    write_guest_u64(info + 0x10, 0);
    0
}

/// `sceVideodec2AllocateComputeQueue(config*, memInfo*, queueOut*)`.
fn allocate_compute_queue(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (config, mem_info, queue_out) = (arguments[0], arguments[1], arguments[2]);
    if config == 0
        || mem_info == 0
        || queue_out == 0
        || !process_guest_range_is_readable(config, COMPUTE_CONFIG_INFO_SIZE as usize)
        || !process_guest_range_is_readable(mem_info, COMPUTE_MEMORY_INFO_SIZE as usize)
        || !process_guest_range_is_readable(queue_out, 8)
    {
        return sce_kernel_error(VIDEODEC2_ERROR_ARGUMENT_POINTER);
    }
    if read_guest_u64(config) != COMPUTE_CONFIG_INFO_SIZE
        || read_guest_u64(mem_info) != COMPUTE_MEMORY_INFO_SIZE
    {
        return sce_kernel_error(VIDEODEC2_ERROR_STRUCT_SIZE);
    }
    let reserved0 = low_u8(read_guest_u64(config + 0x08) >> 40);
    let reserved1 = low_u16(read_guest_u64(config + 0x08) >> 48);
    if reserved0 != 0 || reserved1 != 0 {
        return sce_kernel_error(VIDEODEC2_ERROR_CONFIG_INFO);
    }
    let pipe_id = low_u16(read_guest_u64(config + 0x08));
    let queue_id = low_u16(read_guest_u64(config + 0x08) >> 16);
    if pipe_id > 4 {
        return sce_kernel_error(VIDEODEC2_ERROR_COMPUTE_PIPE_ID);
    }
    if queue_id > 7 {
        return sce_kernel_error(VIDEODEC2_ERROR_COMPUTE_QUEUE_ID);
    }
    let memory = read_guest_u64(mem_info + 0x10);
    let memory_size = read_guest_u64(mem_info + 0x08);
    if memory == 0
        || memory_size < MINIMUM_COMPUTE_MEMORY_SIZE
        || !process_guest_range_is_readable(memory, memory_size as usize)
    {
        return sce_kernel_error(VIDEODEC2_ERROR_MEMORY_POINTER);
    }
    state.videodec2_compute_queues.insert(memory);
    write_guest_u64(queue_out, memory);
    eprintln!(
        "[REMu][VIDEODEC2] allocate compute queue pipe={pipe_id} queue={queue_id} memory={memory:#x} size={memory_size:#x}"
    );
    0
}

/// `sceVideodec2ReleaseComputeQueue(queue)`.
fn release_compute_queue(queue: u64, state: &mut ProbeDispatchState) -> u64 {
    if queue == 0 || !state.videodec2_compute_queues.remove(&queue) {
        return sce_kernel_error(VIDEODEC2_ERROR_COMPUTE_QUEUE);
    }
    eprintln!("[REMu][VIDEODEC2] release compute queue={queue:#x}");
    0
}

/// `sceVideodec2QueryDecoderMemoryInfo(config*, memInfo*)`.
fn query_decoder_memory_info(arguments: [u64; 6]) -> u64 {
    let (config, mem_info) = (arguments[0], arguments[1]);
    if config == 0
        || mem_info == 0
        || !process_guest_range_is_readable(config, DECODER_CONFIG_INFO_SIZE as usize)
        || !process_guest_range_is_readable(mem_info, DECODER_MEMORY_INFO_SIZE as usize)
    {
        return sce_kernel_error(VIDEODEC2_ERROR_ARGUMENT_POINTER);
    }
    if read_guest_u64(config) != DECODER_CONFIG_INFO_SIZE
        || read_guest_u64(mem_info) != DECODER_MEMORY_INFO_SIZE
    {
        return sce_kernel_error(VIDEODEC2_ERROR_STRUCT_SIZE);
    }
    let codec_type = read_guest_u32(config + 0x0c);
    if codec_type != CODEC_TYPE_AVC && codec_type != CODEC_TYPE_HEVC {
        eprintln!("[REMu][VIDEODEC2] QueryDecoderMemoryInfo unsupported codecType={codec_type}");
        return sce_kernel_error(VIDEODEC2_ERROR_CODEC_TYPE);
    }
    // Fill memory requirements; pointers stay caller-owned zeros until Create.
    write_guest_u64(mem_info + 0x08, MINIMUM_DECODER_CPU_MEMORY);
    write_guest_u64(mem_info + 0x10, 0);
    write_guest_u64(mem_info + 0x18, 0);
    write_guest_u64(mem_info + 0x20, 0);
    write_guest_u64(mem_info + 0x28, 0);
    write_guest_u64(mem_info + 0x30, 0);
    write_guest_u64(mem_info + 0x38, MINIMUM_FRAME_BUFFER_SIZE);
    write_guest_u32(mem_info + 0x40, 0x100);
    0
}

/// `sceVideodec2CreateDecoder(config*, memInfo*, decoderOut*)`.
fn create_decoder(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (config, mem_info, decoder_out) = (arguments[0], arguments[1], arguments[2]);
    if config == 0
        || mem_info == 0
        || decoder_out == 0
        || !process_guest_range_is_readable(config, DECODER_CONFIG_INFO_SIZE as usize)
        || !process_guest_range_is_readable(mem_info, DECODER_MEMORY_INFO_SIZE as usize)
        || !process_guest_range_is_readable(decoder_out, 8)
    {
        return sce_kernel_error(VIDEODEC2_ERROR_ARGUMENT_POINTER);
    }
    if read_guest_u64(config) != DECODER_CONFIG_INFO_SIZE
        || read_guest_u64(mem_info) != DECODER_MEMORY_INFO_SIZE
    {
        return sce_kernel_error(VIDEODEC2_ERROR_STRUCT_SIZE);
    }
    let codec_type = read_guest_u32(config + 0x0c);
    let session = match LibavCodecSession::open(codec_type) {
        Ok(session) => session,
        Err(error) => {
            eprintln!("[REMu][VIDEODEC2] CreateDecoder libav open failed: {error}");
            return sce_kernel_error(VIDEODEC2_ERROR_CODEC_TYPE);
        }
    };
    let handle = next_decoder_handle(state);
    state.videodec2_decoders.insert(
        handle,
        Videodec2Decoder {
            codec_type,
            session,
            next_handle_stamp: 0,
        },
    );
    write_guest_u64(decoder_out, handle);
    eprintln!("[REMu][VIDEODEC2] CreateDecoder handle={handle:#x} codecType={codec_type}");
    0
}

fn next_decoder_handle(state: &mut ProbeDispatchState) -> u64 {
    let mut candidate = state
        .videodec2_decoders
        .keys()
        .next_back()
        .copied()
        .unwrap_or(0)
        .saturating_add(1)
        .max(1);
    while state.videodec2_decoders.contains_key(&candidate) {
        candidate = candidate.saturating_add(1).max(1);
    }
    candidate
}

/// `sceVideodec2DeleteDecoder(decoder)`.
fn delete_decoder(decoder: u64, state: &mut ProbeDispatchState) -> u64 {
    if decoder == 0 || state.videodec2_decoders.remove(&decoder).is_none() {
        return sce_kernel_error(VIDEODEC2_ERROR_DECODER_INSTANCE);
    }
    eprintln!("[REMu][VIDEODEC2] DeleteDecoder handle={decoder:#x}");
    0
}

/// `sceVideodec2Decode(decoder, input*, frameBuffer*, output*)`.
fn decode(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (decoder, input, frame_buffer, output) =
        (arguments[0], arguments[1], arguments[2], arguments[3]);
    if decoder == 0
        || input == 0
        || frame_buffer == 0
        || output == 0
        || !process_guest_range_is_readable(input, INPUT_DATA_SIZE as usize)
        || !process_guest_range_is_readable(frame_buffer, FRAME_BUFFER_SIZE as usize)
        || !process_guest_range_is_readable(output, OUTPUT_INFO_SIZE as usize)
    {
        return sce_kernel_error(VIDEODEC2_ERROR_ARGUMENT_POINTER);
    }
    if read_guest_u64(input) != INPUT_DATA_SIZE
        || read_guest_u64(frame_buffer) != FRAME_BUFFER_SIZE
        || read_guest_u64(output) != OUTPUT_INFO_SIZE
    {
        return sce_kernel_error(VIDEODEC2_ERROR_STRUCT_SIZE);
    }
    let Some(instance) = state.videodec2_decoders.get_mut(&decoder) else {
        return sce_kernel_error(VIDEODEC2_ERROR_DECODER_INSTANCE);
    };
    let au = read_guest_u64(input + 0x08);
    let au_size = read_guest_u64(input + 0x10);
    let pts = read_guest_u64(input + 0x18) as i64;
    if au == 0
        || au_size == 0
        || !process_guest_range_is_readable(au, au_size as usize)
    {
        return sce_kernel_error(VIDEODEC2_ERROR_ARGUMENT_POINTER);
    }
    let frame_ptr = read_guest_u64(frame_buffer + 0x08);
    let frame_size = read_guest_u64(frame_buffer + 0x10);
    if frame_ptr == 0 || frame_size < MINIMUM_FRAME_BUFFER_SIZE {
        return sce_kernel_error(VIDEODEC2_ERROR_FRAME_BUFFER_POINTER);
    }
    if !process_guest_range_is_readable(frame_ptr, frame_size as usize) {
        return sce_kernel_error(VIDEODEC2_ERROR_FRAME_BUFFER_POINTER);
    }
    // SAFETY: guest AU range was validated readable above.
    let au_bytes =
        unsafe { std::slice::from_raw_parts(au as *const u8, au_size as usize) }.to_vec();
    if instance.session.send(&au_bytes, pts).is_err() {
        eprintln!("[REMu][VIDEODEC2] Decode send failed handle={decoder:#x}");
        return sce_kernel_error(VIDEODEC2_ERROR_CONFIG_INFO);
    }
    // SAFETY: guest frame buffer was validated writable/readable above.
    let frame_out =
        unsafe { std::slice::from_raw_parts_mut(frame_ptr as *mut u8, frame_size as usize) };
    match instance.session.receive_nv12(frame_out) {
        Ok(Some((width, height, pitch))) => {
            let need = u64::from(pitch)
                .saturating_mul(u64::from(height))
                .saturating_mul(3)
                / 2;
            if frame_size < need {
                return sce_kernel_error(VIDEODEC2_ERROR_FRAME_BUFFER_SIZE);
            }
            write_guest_u8(frame_buffer + 0x18, 1); // isAccepted
            write_output_info(output, instance.codec_type, width, height, pitch, frame_ptr, need);
            instance.next_handle_stamp = instance.next_handle_stamp.saturating_add(1);
            0
        }
        Ok(None) => {
            // Need more input — publish an empty but valid output record.
            write_guest_u8(frame_buffer + 0x18, 0);
            write_guest_u8(output + 0x08, 0); // isValid = false
            0
        }
        Err(()) => {
            eprintln!("[REMu][VIDEODEC2] Decode receive failed handle={decoder:#x}");
            sce_kernel_error(VIDEODEC2_ERROR_CONFIG_INFO)
        }
    }
}

fn write_output_info(
    output: u64,
    codec_type: u32,
    width: u32,
    height: u32,
    pitch: u32,
    frame_ptr: u64,
    frame_bytes: u64,
) {
    write_guest_u8(output + 0x08, 1); // isValid
    write_guest_u8(output + 0x09, 0); // isErrorFrame
    write_guest_u8(output + 0x0a, 1); // pictureCount
    write_guest_u8(output + 0x0b, 0); // isDiscardedFrame
    write_guest_u32(output + 0x0c, codec_type);
    write_guest_u32(output + 0x10, width);
    write_guest_u32(output + 0x14, pitch);
    write_guest_u32(output + 0x18, height);
    write_guest_u64(output + 0x20, frame_ptr);
    write_guest_u64(output + 0x28, frame_bytes);
    write_guest_u32(output + 0x30, 0); // FRAME_FORMAT_DEFAULT
    write_guest_u32(output + 0x34, pitch);
}

/// `sceVideodec2Flush(decoder, frameBuffer*, output*)`.
fn flush(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (decoder, frame_buffer, output) = (arguments[0], arguments[1], arguments[2]);
    if decoder == 0
        || frame_buffer == 0
        || output == 0
        || !process_guest_range_is_readable(frame_buffer, FRAME_BUFFER_SIZE as usize)
        || !process_guest_range_is_readable(output, OUTPUT_INFO_SIZE as usize)
    {
        return sce_kernel_error(VIDEODEC2_ERROR_ARGUMENT_POINTER);
    }
    let Some(instance) = state.videodec2_decoders.get_mut(&decoder) else {
        return sce_kernel_error(VIDEODEC2_ERROR_DECODER_INSTANCE);
    };
    if instance.session.flush().is_err() {
        eprintln!("[REMu][VIDEODEC2] Flush failed handle={decoder:#x}");
        return sce_kernel_error(VIDEODEC2_ERROR_CONFIG_INFO);
    }
    let frame_ptr = read_guest_u64(frame_buffer + 0x08);
    let frame_size = read_guest_u64(frame_buffer + 0x10);
    if frame_ptr == 0 || !process_guest_range_is_readable(frame_ptr, frame_size as usize) {
        write_guest_u8(output + 0x08, 0);
        return 0;
    }
    let frame_out =
        unsafe { std::slice::from_raw_parts_mut(frame_ptr as *mut u8, frame_size as usize) };
    match instance.session.receive_nv12(frame_out) {
        Ok(Some((width, height, pitch))) => {
            let need = u64::from(pitch)
                .saturating_mul(u64::from(height))
                .saturating_mul(3)
                / 2;
            write_guest_u8(frame_buffer + 0x18, 1);
            write_output_info(output, instance.codec_type, width, height, pitch, frame_ptr, need);
            0
        }
        Ok(None) | Err(()) => {
            write_guest_u8(frame_buffer + 0x18, 0);
            write_guest_u8(output + 0x08, 0);
            0
        }
    }
}

/// `sceVideodec2Reset(decoder)`.
fn reset(decoder: u64, state: &mut ProbeDispatchState) -> u64 {
    let Some(instance) = state.videodec2_decoders.get_mut(&decoder) else {
        return sce_kernel_error(VIDEODEC2_ERROR_DECODER_INSTANCE);
    };
    if instance.session.flush().is_err() {
        eprintln!("[REMu][VIDEODEC2] Reset failed handle={decoder:#x}");
        return sce_kernel_error(VIDEODEC2_ERROR_CONFIG_INFO);
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProbeDispatchState;

    #[test]
    fn query_reports_a_fixed_workspace_and_clears_the_pointer() {
        let mut info = [0_u64; 3];
        info[0] = COMPUTE_MEMORY_INFO_SIZE;
        info[1] = 0xdead;
        info[2] = 0xbeef;
        assert_eq!(
            query_compute_memory_info(info.as_mut_ptr().addr() as u64),
            0
        );
        assert_eq!(info[1], MINIMUM_COMPUTE_MEMORY_SIZE);
        assert_eq!(info[2], 0);
    }

    #[test]
    fn allocate_returns_the_workspace_pointer_as_the_queue() {
        let mut config = [0_u8; COMPUTE_CONFIG_INFO_SIZE as usize];
        config[..8].copy_from_slice(&COMPUTE_CONFIG_INFO_SIZE.to_le_bytes());
        config[8] = 1;
        config[9] = 0;
        config[10] = 2;
        config[11] = 0;
        let mut workspace = vec![0_u8; MINIMUM_COMPUTE_MEMORY_SIZE as usize];
        let mut mem_info = [0_u64; 3];
        mem_info[0] = COMPUTE_MEMORY_INFO_SIZE;
        mem_info[1] = MINIMUM_COMPUTE_MEMORY_SIZE;
        mem_info[2] = workspace.as_mut_ptr().addr() as u64;
        let mut queue = 0_u64;
        let mut state = ProbeDispatchState::default();
        assert_eq!(
            allocate_compute_queue(
                [
                    config.as_ptr().addr() as u64,
                    mem_info.as_ptr().addr() as u64,
                    (&raw mut queue).addr() as u64,
                    0,
                    0,
                    0,
                ],
                &mut state,
            ),
            0
        );
        assert_eq!(queue, mem_info[2]);
        assert!(state.videodec2_compute_queues.contains(&queue));
        assert_eq!(release_compute_queue(queue, &mut state), 0);
        assert!(state.videodec2_compute_queues.is_empty());
    }

    #[test]
    fn query_rejects_a_mismatched_struct_size() {
        let mut info = [0_u64; 3];
        info[0] = 0x10;
        assert_eq!(
            query_compute_memory_info(info.as_mut_ptr().addr() as u64),
            sce_kernel_error(VIDEODEC2_ERROR_STRUCT_SIZE)
        );
    }

    #[test]
    fn dispatch_leaves_unrelated_nids_alone() {
        let mut state = ProbeDispatchState::default();
        assert_eq!(dispatch("not-a-videodec2-nid", [0; 6], &mut state), None);
    }

    #[test]
    fn create_decoder_rejects_unsupported_codec_loudly() {
        let mut config = vec![0_u8; DECODER_CONFIG_INFO_SIZE as usize];
        config[..8].copy_from_slice(&DECODER_CONFIG_INFO_SIZE.to_le_bytes());
        config[0x0c..0x10].copy_from_slice(&2382845_u32.to_le_bytes()); // VP9
        let mut mem = vec![0_u8; DECODER_MEMORY_INFO_SIZE as usize];
        mem[..8].copy_from_slice(&DECODER_MEMORY_INFO_SIZE.to_le_bytes());
        let mut out = 0_u64;
        let mut state = ProbeDispatchState::default();
        let rc = create_decoder(
            [
                config.as_ptr().addr() as u64,
                mem.as_ptr().addr() as u64,
                (&raw mut out).addr() as u64,
                0,
                0,
                0,
            ],
            &mut state,
        );
        assert_eq!(rc, sce_kernel_error(VIDEODEC2_ERROR_CODEC_TYPE));
        assert!(state.videodec2_decoders.is_empty());
    }

    #[test]
    fn decode_rejects_null_pointers_loudly() {
        let mut state = ProbeDispatchState::default();
        assert_eq!(
            decode([0, 0, 0, 0, 0, 0], &mut state),
            sce_kernel_error(VIDEODEC2_ERROR_ARGUMENT_POINTER)
        );
    }
}
