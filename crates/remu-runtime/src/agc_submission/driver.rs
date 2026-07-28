// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::{
    AgcSubmissionError, ProbeDispatchState, agc_command_abi, capture_agc_gpu_events, low_u32,
    queue_agc_event_write, queue_video_out_flip, read_guest_u32, read_guest_u64,
    request_agc_compute_submission_completion, request_agc_submission_completion,
    trace_agc_programs, trace_logarithmic_sequence,
};

use super::{AgcSubmissionContext, agc_parse_graphics_submission, agc_parse_submission};

pub(crate) fn agc_driver_submit_dcb(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let packet = arguments[0];
    if packet == 0 {
        eprintln!("[REMu][AGC] rejected DCB submission: null packet");
        return u64::MAX;
    }
    let command_address = read_guest_u64(packet);
    let dword_count = read_guest_u32(packet + 8);
    if command_address == 0 || dword_count == 0 || dword_count > 1_000_000 {
        // The guest treats a refused submission as fatal, so name the packet it rejected.
        eprintln!(
            "[REMu][AGC] rejected DCB submission: packet={packet:#x} commands={command_address:#x} dwords={dword_count} args={:#x},{:#x},{:#x}",
            arguments[1], arguments[2], arguments[3]
        );
        return u64::MAX;
    }
    let next_submission = state.agc_submission_count.saturating_add(1);
    if trace_agc_programs() || trace_logarithmic_sequence(next_submission, 8) {
        eprintln!(
            "[REMu][AGC] queue=graphics submit={next_submission} address={command_address:#x} dwords={dword_count}"
        );
    }
    let mut flips = Vec::new();
    let mut event_writes = Vec::new();
    let (result, gpu_events) = capture_agc_gpu_events(|| {
        agc_parse_graphics_submission(
            command_address,
            dword_count,
            &mut AgcSubmissionContext {
                graphics: &mut state.agc_graphics,
                shaders: &mut state.agc_shader_registry,
                index_resources: &state.agc_index_resources,
                video_out: &state.video_out,
                submission_count: &mut state.agc_submission_count,
                flips: &mut flips,
                event_writes: &mut event_writes,
            },
        )
    });
    match result {
        Ok(()) => {
            agc_command_abi::track_graphics_submission(command_address, dword_count);
            if !request_agc_submission_completion(state, 0, gpu_events) {
                eprintln!("[REMu][AGC] rejected DCB submission: presenter is unavailable");
                return u64::MAX;
            }
            for (handle, buffer_index, flip_argument) in flips {
                queue_video_out_flip(state, handle, buffer_index, flip_argument);
            }
            for event_type in event_writes {
                queue_agc_event_write(state, event_type);
            }
            0
        }
        Err(error) => {
            eprintln!("[REMu][AGC] rejected DCB submission: {error}");
            u64::MAX
        }
    }
}

pub(crate) fn agc_driver_submit_multi_dcbs(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let addresses = arguments[0];
    let sizes = arguments[1];
    let count = low_u32(arguments[2]);
    if addresses == 0 || sizes == 0 || count == 0 || count > 4096 {
        return u64::MAX;
    }
    for index in 0..count {
        let command_address = read_guest_u64(addresses + u64::from(index) * 8);
        let dword_count = read_guest_u32(sizes + u64::from(index) * 4);
        if command_address != 0 && dword_count != 0 && dword_count <= 1_000_000 {
            let mut flips = Vec::new();
            let mut event_writes = Vec::new();
            let (result, gpu_events) = capture_agc_gpu_events(|| {
                agc_parse_graphics_submission(
                    command_address,
                    dword_count,
                    &mut AgcSubmissionContext {
                        graphics: &mut state.agc_graphics,
                        shaders: &mut state.agc_shader_registry,
                        index_resources: &state.agc_index_resources,
                        video_out: &state.video_out,
                        submission_count: &mut state.agc_submission_count,
                        flips: &mut flips,
                        event_writes: &mut event_writes,
                    },
                )
            });
            match result {
                Ok(()) => {
                    agc_command_abi::track_graphics_submission(command_address, dword_count);
                    if !request_agc_submission_completion(state, 0, gpu_events) {
                        eprintln!(
                            "[REMu][AGC] rejected DCB {index} in multi-submission: presenter is unavailable"
                        );
                        return u64::MAX;
                    }
                    for (handle, buffer_index, flip_argument) in flips {
                        queue_video_out_flip(state, handle, buffer_index, flip_argument);
                    }
                    for event_type in event_writes {
                        queue_agc_event_write(state, event_type);
                    }
                }
                Err(error) => {
                    eprintln!("[REMu][AGC] rejected DCB {index} in multi-submission: {error}");
                    return u64::MAX;
                }
            }
        }
    }
    0
}

pub(super) fn agc_resolve_acb_descriptor(
    command_address: u64,
    dword_count: u32,
) -> Result<(u64, u32), AgcSubmissionError> {
    const ACB_DESCRIPTOR_MAGIC: u32 = 0x5533_ccaa;
    if dword_count < 5
        || read_guest_u32(command_address + 16) != ACB_DESCRIPTOR_MAGIC
        || read_guest_u32(command_address + 12) != 0
    {
        return Ok((command_address, dword_count));
    }
    let descriptor_address = read_guest_u64(command_address);
    let descriptor_dword_count = read_guest_u32(command_address + 8);
    if descriptor_address == 0 || descriptor_dword_count == 0 || descriptor_dword_count > 1_000_000
    {
        return Err(AgcSubmissionError::InvalidAcbDescriptor {
            address: descriptor_address,
            dword_count: descriptor_dword_count,
        });
    }
    Ok((descriptor_address, descriptor_dword_count))
}

pub(super) fn agc_submit_compute_commands(
    owner: u32,
    command_address: u64,
    dword_count: u32,
    state: &mut ProbeDispatchState,
    flips: &mut Vec<(u32, u32, u64)>,
    event_writes: &mut Vec<u32>,
) -> Result<(), AgcSubmissionError> {
    let (command_address, dword_count) = agc_resolve_acb_descriptor(command_address, dword_count)?;
    let graphics = state.agc_compute_queues.entry(owner).or_default();
    agc_parse_submission(
        command_address,
        dword_count,
        &mut AgcSubmissionContext {
            graphics,
            shaders: &mut state.agc_shader_registry,
            index_resources: &state.agc_index_resources,
            video_out: &state.video_out,
            submission_count: &mut state.agc_submission_count,
            flips,
            event_writes,
        },
    )
}

pub(crate) fn agc_driver_submit_acb(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let owner = low_u32(arguments[0]);
    let packet = arguments[1];
    if packet == 0 {
        eprintln!("[REMu][AGC] rejected ACB submission for owner {owner}: null packet");
        return u64::MAX;
    }
    let command_address = read_guest_u64(packet);
    let dword_count = read_guest_u32(packet + 8);
    if command_address == 0 || dword_count == 0 || dword_count > 1_000_000 {
        // A submission the driver refuses is fatal to the guest, so describe the packet it
        // rejected rather than returning a bare error the title can only assert on.
        eprintln!(
            "[REMu][AGC] rejected ACB submission for owner {owner}: packet={packet:#x} commands={command_address:#x} dwords={dword_count} args={:#x},{:#x},{:#x},{:#x}",
            arguments[2], arguments[3], arguments[4], arguments[5]
        );
        return u64::MAX;
    }
    let next_submission = state.agc_submission_count.saturating_add(1);
    if trace_agc_programs() || trace_logarithmic_sequence(next_submission, 8) {
        eprintln!(
            "[REMu][AGC] queue=async-compute owner={owner} submit={next_submission} address={command_address:#x} dwords={dword_count}"
        );
    }
    let mut flips = Vec::new();
    let mut event_writes = Vec::new();
    let (result, gpu_events) = capture_agc_gpu_events(|| {
        agc_submit_compute_commands(
            owner,
            command_address,
            dword_count,
            state,
            &mut flips,
            &mut event_writes,
        )
    });
    match result {
        Ok(()) => {
            // The native driver asks the compute backend to trigger an interrupt when the
            // command buffer does not already contain one. SubmissionBoundary is our backend
            // flush/completion point, so every accepted ACB needs one just like a DCB does.
            // Without it, the last compute batch remains open and the guest waits forever for
            // the queue completion before it can submit the next frame.
            if !request_agc_compute_submission_completion(state, u64::from(owner), gpu_events) {
                eprintln!(
                    "[REMu][AGC] rejected ACB submission for owner {owner}: presenter is unavailable"
                );
                return u64::MAX;
            }
            for (handle, buffer_index, flip_argument) in flips {
                queue_video_out_flip(state, handle, buffer_index, flip_argument);
            }
            for event_type in event_writes {
                queue_agc_event_write(state, event_type);
            }
            0
        }
        Err(error) => {
            eprintln!("[REMu][AGC] rejected ACB submission for owner {owner}: {error}");
            u64::MAX
        }
    }
}

pub(crate) fn agc_driver_submit_multi_acbs(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let owner = low_u32(arguments[0]);
    let addresses = arguments[1];
    let sizes = arguments[2];
    let count = low_u32(arguments[3]);
    if addresses == 0 || sizes == 0 || count == 0 || count > 4096 {
        return u64::MAX;
    }
    for index in 0..count {
        let command_address = read_guest_u64(addresses + u64::from(index) * 8);
        let dword_count = read_guest_u32(sizes + u64::from(index) * 4);
        if command_address == 0 || dword_count == 0 || dword_count > 1_000_000 {
            return u64::MAX;
        }
        let mut flips = Vec::new();
        let mut event_writes = Vec::new();
        let (result, gpu_events) = capture_agc_gpu_events(|| {
            agc_submit_compute_commands(
                owner,
                command_address,
                dword_count,
                state,
                &mut flips,
                &mut event_writes,
            )
        });
        match result {
            Ok(()) => {
                if !request_agc_compute_submission_completion(state, u64::from(owner), gpu_events) {
                    eprintln!(
                        "[REMu][AGC] rejected ACB {index} for owner {owner}: presenter is unavailable"
                    );
                    return u64::MAX;
                }
                for (handle, buffer_index, flip_argument) in flips {
                    queue_video_out_flip(state, handle, buffer_index, flip_argument);
                }
                for event_type in event_writes {
                    queue_agc_event_write(state, event_type);
                }
            }
            Err(error) => {
                eprintln!(
                    "[REMu][AGC] rejected ACB {index} for owner {owner} in multi-submission: {error}"
                );
                return u64::MAX;
            }
        }
    }
    0
}
