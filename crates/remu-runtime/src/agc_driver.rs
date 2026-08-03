// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Agc driver submission and event plumbing.
//!
//! This module implements sceAgcDriver* entry points covering submission, event queues and capture
//! control.
//!
//! Three cohorts here answer for a capability REmu does not host, and each reports the console's
//! own unavailable-capability error rather than a success the guest could not observe: Razor GPU
//! capture and submit validation both live in a tooling module that is never loaded, and resource
//! registration is refused outright on any target whose PA Debug setting is off, which is every
//! target REmu presents.
//!
//! The packet writers share their encodings with the public command-buffer builders in
//! `agc_command_abi`. Both routes reach the same command processor and the same capture readers,
//! so an operation that encoded differently depending on which export the title imported would
//! decode as two unrelated commands in one stream.
//!
//! [`dispatch`] is consulted by the runtime's import dispatcher only after the main table
//! misses, and returns `None` for any NID it does not own so the caller can keep looking.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::{
    AgcSubmissionContext, KERNEL_EVENT_FILTER_GRAPHICS, ProbeDispatchState,
    SCE_KERNEL_ERROR_NOT_FOUND, agc_command_abi, agc_parse_graphics_submission,
    capture_agc_gpu_events, process_guest_range_is_readable, queue_agc_event_write,
    queue_video_out_flip, request_agc_submission_completion, sce_kernel_error,
};

const AGC_ERROR_ALIGNMENT: u64 = 0x8a6c_0002;
const AGC_ERROR_INVALID_SUBMIT_QUEUE: u64 = 0x8a6c_0009;
const AGC_ERROR_PARAMETER_NULL: u64 = 0x8a6c_000a;
const AGC_ERROR_OUT_OF_BOUNDS: u64 = 0x8a6c_000b;
const AGC_ERROR_SUBMIT_SIZE_TOO_LARGE: u64 = 0x8a6c_0020;
const AGC_ERROR_INVALID_WORKLOAD_STREAM_ID: u64 = 0x8a6c_0033;
const AGC_ERROR_INVALID_WORKLOAD_STREAM_NAME_PTR: u64 = 0x8a6c_0035;
const AGC_ERROR_WORKLOAD_STREAM_NOT_ALLOCATED: u64 = 0x8a6c_003a;
const AGC_ERROR_UNEXPECTED_PARAMETER: u64 = 0x8a6c_003c;
const AGC_ERROR_GPU_CAPTURE_NOT_AVAILABLE: u64 = 0x8a6c_1000;
const AGC_ERROR_RESOURCE_REGISTRATION_NO_PA_DEBUG: u64 = 0x8a6c_9018;

const IT_NOP: u32 = 0x10;

const R_WAIT_FLIP_DONE: u32 = 0x06;
const R_PUSH_MARKER: u32 = 0x0b;
const R_POP_MARKER: u32 = 0x0c;
const R_FLIP: u32 = 0x17;

const SET_FLIP_PACKET_DWORDS: u32 = 6;
const WAIT_RENDERING_PACKET_DWORDS: u32 = 7;
const POP_MARKER_PACKET_DWORDS: u32 = 2;
/// A type-3 header stores its length as `dwords - 2` in fourteen bits.
const MAX_PACKET_DWORDS: u32 = 0x4001;

const COMMAND_BUFFER_CURSOR_UP_OFFSET: u64 = 0x10;
const COMMAND_BUFFER_CURSOR_DOWN_OFFSET: u64 = 0x18;
const COMMAND_BUFFER_CALLBACK_OFFSET: u64 = 0x20;
const COMMAND_BUFFER_USER_DATA_OFFSET: u64 = 0x28;
const COMMAND_BUFFER_RESERVED_DWORDS_OFFSET: u64 = 0x30;
const COMMAND_BUFFER_DESCRIPTOR_BYTES: u64 = 0x34;

/// Only the normal graphics ring carries work; the interrupt ring exists in the selector but the
/// console does not run submissions on it.
const MAX_GRAPHICS_QUEUE: u32 = 1;
/// A command buffer may not exceed four megabytes less one dword, which bounds a submission's
/// dword count.
const MAX_SUBMIT_DWORDS: u32 = 0x000f_ffff;

/// Byte stride of one entry in a `SceKernelEvent` array as `sceKernelWaitEqueue` fills it.
const KERNEL_EVENT_IDENT_OFFSET: u64 = 0x00;
const KERNEL_EVENT_DATA_OFFSET: u64 = 0x10;
/// An interrupt context identifier is a 27-bit constant carried in the event payload.
const INTERRUPT_CONTEXT_ID_MASK: u64 = 0x07ff_ffff;

const MIN_WORKLOAD_STREAM_ID: u32 = 1;
const MAX_WORKLOAD_STREAM_ID: u32 = 31;
/// A stream name occupies at most this many bytes including its terminator.
const MAX_WORKLOAD_STREAM_NAME_BYTES: usize = 32;

/// Largest span probed at once while scanning a marker string for its terminator.
const MARKER_PROBE_BYTES: u64 = 64;

/// Wavefront debugging is refused on a target whose PA Debug setting is off.
const SHADER_DEBUGGING_NOT_AVAILABLE: u64 = 1;

type GrowCallback = unsafe extern "C" fn(u64, u32, u64) -> bool;

/// Names registered for the workload stream identifiers Razor GPU can select between.
static WORKLOAD_STREAMS: Mutex<BTreeMap<u32, String>> = Mutex::new(BTreeMap::new());

/// Handles a NID belonging to this module, or returns `None` if it owns no such NID.
pub(crate) fn dispatch(
    nid: &str,
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> Option<u64> {
    dispatch_submission(nid, arguments, state)
        .or_else(|| dispatch_event_queue(nid, arguments, state))
        .or_else(|| dispatch_command(nid, arguments))
        .or_else(|| dispatch_workload_stream(nid, arguments))
        .or_else(|| dispatch_tooling(nid))
        .or_else(|| dispatch_resource_registration(nid))
}

fn dispatch_submission(
    nid: &str,
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> Option<u64> {
    Some(match nid {
        "AhGvpITrf4M" => agr_submit_dcb(arguments, state),
        "+T8Xo6LtFJI" => agr_submit_multi_dcbs(arguments, state),
        _ => return None,
    })
}

fn dispatch_event_queue(
    nid: &str,
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> Option<u64> {
    Some(match nid {
        "DL2RXaXOy88" => delete_eq_event(arguments[0], arguments[1], state),
        "5CdQTZIQPxM" => event_type(arguments[0]),
        "Zw7uUVPulbw" => event_context_id(arguments[0]),
        _ => return None,
    })
}

fn dispatch_command(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "cwbxjPSJ7WQ" => set_flip(arguments),
        "u8BkdHb1+Po" => wait_until_safe_for_rendering(arguments),
        "+b34-CLWc0s" => write_push_marker(arguments),
        "LEnn-4ARRJM" => write_pop_marker(arguments[0]),
        "2PrsbRYyZi4" => u64::from(SET_FLIP_PACKET_DWORDS),
        "0MtUJ3BpGhE" => u64::from(WAIT_RENDERING_PACKET_DWORDS),
        _ => return None,
    })
}

fn dispatch_workload_stream(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "3AyTaWcF-H8" => register_workload_stream(low_u32(arguments[0]), arguments[1]),
        "n5ElQVYsU1A" => unregister_workload_stream(low_u32(arguments[0])),
        _ => return None,
    })
}

/// Answers the capture, trace and submit-validation queries, none of which REmu can service.
///
/// Every one of these is a Razor GPU or Target Manager facility provided by a tooling module the
/// console loads separately. REmu never loads it, so the queries report inactive and the control
/// entry points report the module's absence.
fn dispatch_tooling(nid: &str) -> Option<u64> {
    Some(match nid {
        "Ddwk4gLT5j0" | "+TN0oRTBxJQ" | "qspAL8bgcBY" | "HMnVBVUyajk" => 0,
        "rJUyMrDdxJg" => SHADER_DEBUGGING_NOT_AVAILABLE,
        "Xq5WmbwPTnQ" | "SAfhzJPcjuk" | "FOwvmNlFLjM" | "+iAOE3jCnkk" | "WHIOMbb+iIU"
        | "chJWZcNSzjk" | "ICaGtkEIXTk" | "8F5j7U-Bq6s" => AGC_ERROR_GPU_CAPTURE_NOT_AVAILABLE,
        _ => return None,
    })
}

/// Refuses the resource-registration cluster the way a target without PA Debug does.
///
/// Registration exists to label memory for Razor GPU, and the console gates the whole cluster —
/// initialisation, registration, lookup and teardown alike — on the PA Debug target setting. REmu
/// presents no debug target, which `sceAgcDriverIsPaDebug` also reports, so accepting a
/// registration here would hand back handles that no later query could resolve.
fn dispatch_resource_registration(nid: &str) -> Option<u64> {
    Some(match nid {
        "F0Y42t-3e18" | "AOLcoIkQDgM" | "X-Nm5KLREeg" | "U9ueyEhSkF4" | "W5z4eZrjEas"
        | "emP3ckeS2uo" | "5l3IfCFJxBs" | "LepGrgk77sM" | "M9yBzRKkjPc" | "NghWEUXp1qM"
        | "mXn+K9E-wOA" | "rI9lNAXPMIw" | "ls4jfY576lw" | "VOMSpd9+vxU" | "SCoAN5fYlUM"
        | "ZLJk9r2+2Aw" | "pWLG7WOpVcw" | "F0ZXt5q0ZTA" | "uJziRsODk1c" => {
            // The refusal is policy, not a defect, but its error code surfaces in [REMu][FAIL]
            // lines that otherwise read as a broken import. Say so once.
            static EXPLAINED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !EXPLAINED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[REMu][AGC] declining resource registration (first: nid={nid}): REmu \
                     presents no PA Debug target, so the whole cluster reports \
                     {AGC_ERROR_RESOURCE_REGISTRATION_NO_PA_DEBUG:#x}; this is expected, \
                     registration only labels memory for Razor GPU"
                );
            }
            AGC_ERROR_RESOURCE_REGISTRATION_NO_PA_DEBUG
        }
        _ => return None,
    })
}

/// Submits one command buffer to a graphics ring and requests a completion event for it.
fn agr_submit_dcb(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let queue = low_u32(arguments[0]);
    let command_address = arguments[1];
    let dword_count = low_u32(arguments[2]);
    if let Err(error) = check_submission(queue, command_address, dword_count) {
        eprintln!(
            "[REMu][AGC] rejected graphics submission: queue={queue} commands={command_address:#x} dwords={dword_count} error={error:#x}"
        );
        return error;
    }
    // A caller that asked for no dwords asked the driver for no work, and the console has nothing
    // to reject; skipping the parse also keeps a zero-length range out of the readability check.
    if dword_count == 0 {
        return 0;
    }
    submit_graphics(queue, command_address, dword_count, state)
}

/// Submits a batch of command buffers to a graphics ring in the order the guest listed them.
fn agr_submit_multi_dcbs(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let queue = low_u32(arguments[0]);
    let count = low_u32(arguments[1]);
    let addresses = arguments[2];
    let sizes = arguments[3];
    if queue > MAX_GRAPHICS_QUEUE {
        return AGC_ERROR_INVALID_SUBMIT_QUEUE;
    }
    if count == 0 {
        return 0;
    }
    if addresses == 0 || sizes == 0 {
        return AGC_ERROR_PARAMETER_NULL;
    }
    if !guest_range_is_readable(addresses, u64::from(count) * 8)
        || !guest_range_is_readable(sizes, u64::from(count) * 4)
    {
        return AGC_ERROR_OUT_OF_BOUNDS;
    }
    // Every reference in the batch is checked before any of it runs. A malformed reference found
    // halfway through would otherwise leave the earlier buffers' flips and interrupts already
    // queued against a submission the guest is told failed.
    for index in 0..u64::from(count) {
        let command_address = read_u64(addresses + index * 8);
        let dword_count = read_u32(sizes + index * 4);
        if let Err(error) = check_submission(queue, command_address, dword_count) {
            eprintln!(
                "[REMu][AGC] rejected graphics submission {index} of {count}: commands={command_address:#x} dwords={dword_count} error={error:#x}"
            );
            return error;
        }
    }
    for index in 0..u64::from(count) {
        let command_address = read_u64(addresses + index * 8);
        let dword_count = read_u32(sizes + index * 4);
        if dword_count == 0 {
            continue;
        }
        let result = submit_graphics(queue, command_address, dword_count, state);
        if result != 0 {
            return result;
        }
    }
    0
}

/// Validates the ring selector and the shape of one command-buffer reference.
fn check_submission(queue: u32, command_address: u64, dword_count: u32) -> Result<(), u64> {
    if queue > MAX_GRAPHICS_QUEUE {
        return Err(AGC_ERROR_INVALID_SUBMIT_QUEUE);
    }
    if dword_count > MAX_SUBMIT_DWORDS {
        return Err(AGC_ERROR_SUBMIT_SIZE_TOO_LARGE);
    }
    if dword_count == 0 {
        return Ok(());
    }
    if command_address == 0 {
        return Err(AGC_ERROR_PARAMETER_NULL);
    }
    if !command_address.is_multiple_of(4) {
        return Err(AGC_ERROR_ALIGNMENT);
    }
    if !guest_range_is_readable(command_address, u64::from(dword_count) * 4) {
        return Err(AGC_ERROR_OUT_OF_BOUNDS);
    }
    Ok(())
}

/// Walks a validated packet stream and applies the flips and interrupts it carries.
fn submit_graphics(
    queue: u32,
    command_address: u64,
    dword_count: u32,
    state: &mut ProbeDispatchState,
) -> u64 {
    let mut flips = Vec::new();
    let mut event_writes = Vec::new();
    let (result, gpu_events) = capture_agc_gpu_events(|| {
        agc_parse_graphics_submission(
            command_address,
            dword_count,
            &mut AgcSubmissionContext {
                graphics: &mut state.agc_graphics,
                shaders: &mut state.agc_shader_registry,
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
            if !request_agc_submission_completion(state, u64::from(queue), gpu_events) {
                eprintln!("[REMu][AGC] rejected graphics submission: presenter is unavailable");
                return AGC_ERROR_UNEXPECTED_PARAMETER;
            }
            for (handle, buffer_index, flip_argument) in flips {
                queue_video_out_flip(state, handle, buffer_index, flip_argument);
            }
            for event in event_writes {
                queue_agc_event_write(state, event);
            }
            0
        }
        Err(error) => {
            eprintln!("[REMu][AGC] rejected graphics submission: {error}");
            AGC_ERROR_UNEXPECTED_PARAMETER
        }
    }
}

/// Stops an event queue listening for the interrupts a matching `sceAgcDriverAddEqEvent` selected.
///
/// The event type is a four-byte structure the caller passes in a register, so the upper half of
/// that register is not part of the value: a registration is matched on the low word alone, or a
/// deletion issued from a different call site than the registration would never find it.
fn delete_eq_event(equeue: u64, requested: u64, state: &mut ProbeDispatchState) -> u64 {
    if !state.kernel_equeues.contains(&equeue) {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    let requested = low_u32(requested);
    let matched = state
        .kernel_event_registrations
        .keys()
        .filter(|&&(registered_queue, filter, ident)| {
            registered_queue == equeue
                && filter == KERNEL_EVENT_FILTER_GRAPHICS
                && low_u32(ident) == requested
        })
        .map(|&(_, _, ident)| ident)
        .collect::<Vec<_>>();
    if matched.is_empty() {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    for ident in matched {
        state
            .kernel_event_registrations
            .remove(&(equeue, KERNEL_EVENT_FILTER_GRAPHICS, ident));
        // An interrupt already queued for a listener the guest just cancelled must not surface on
        // the next wait, where it would resolve user data the guest considers released.
        crate::discard_kernel_events(state, |event| {
            event.equeue == equeue
                && event.filter == KERNEL_EVENT_FILTER_GRAPHICS
                && event.ident == ident
        });
    }
    0
}

/// Extracts the packed interrupt-source descriptor from a delivered kernel event.
///
/// The ABI has no failure return, so an event pointer the guest never mapped reports the
/// all-zero descriptor rather than faulting inside the accessor.
fn event_type(event: u64) -> u64 {
    let Some(ident) = read_event_field(event, KERNEL_EVENT_IDENT_OFFSET) else {
        return 0;
    };
    u64::from(low_u32(ident))
}

/// Extracts the interrupt context identifier a delivered kernel event carries.
fn event_context_id(event: u64) -> u64 {
    let Some(data) = read_event_field(event, KERNEL_EVENT_DATA_OFFSET) else {
        return 0;
    };
    data & INTERRUPT_CONTEXT_ID_MASK
}

fn read_event_field(event: u64, offset: u64) -> Option<u64> {
    let address = event.checked_add(offset)?;
    guest_range_is_readable(address, 8).then(|| read_u64(address))
}

/// Associates a display name with a workload stream identifier for capture tooling.
fn register_workload_stream(stream_id: u32, name_address: u64) -> u64 {
    if !(MIN_WORKLOAD_STREAM_ID..=MAX_WORKLOAD_STREAM_ID).contains(&stream_id) {
        return AGC_ERROR_INVALID_WORKLOAD_STREAM_ID;
    }
    let Some(name) = read_guest_name(name_address, MAX_WORKLOAD_STREAM_NAME_BYTES - 1) else {
        return AGC_ERROR_INVALID_WORKLOAD_STREAM_NAME_PTR;
    };
    workload_streams().insert(stream_id, name);
    0
}

/// Releases a workload stream identifier so it can be registered again under a new name.
fn unregister_workload_stream(stream_id: u32) -> u64 {
    if !(MIN_WORKLOAD_STREAM_ID..=MAX_WORKLOAD_STREAM_ID).contains(&stream_id) {
        return AGC_ERROR_INVALID_WORKLOAD_STREAM_ID;
    }
    if workload_streams().remove(&stream_id).is_none() {
        return AGC_ERROR_WORKLOAD_STREAM_NOT_ALLOCATED;
    }
    0
}

fn workload_streams() -> std::sync::MutexGuard<'static, BTreeMap<u32, String>> {
    WORKLOAD_STREAMS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Appends the packet that presents a display buffer once the preceding work has retired.
fn set_flip(arguments: [u64; 6]) -> u64 {
    let Some(command) = allocate_dwords(arguments[0], SET_FLIP_PACKET_DWORDS) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(SET_FLIP_PACKET_DWORDS, IT_NOP, R_FLIP),
            low_u32(arguments[1]),
            low_u32(arguments[2]),
            low_u32(arguments[3]),
            low_u32(arguments[4]),
            low_u32(arguments[4] >> 32),
        ],
    );
    command
}

/// Appends the packet that stalls the command processor until a display buffer leaves scanout.
fn wait_until_safe_for_rendering(arguments: [u64; 6]) -> u64 {
    let Some(command) = allocate_dwords(arguments[0], WAIT_RENDERING_PACKET_DWORDS) else {
        return 0;
    };
    write_words(
        command,
        &[
            pm4(WAIT_RENDERING_PACKET_DWORDS, IT_NOP, R_WAIT_FLIP_DONE),
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

/// Appends the packet that opens a named marker region in a capture.
///
/// The colour the public marker API accepts has no located field in this encoding, so it is not
/// written: guessing a field would overwrite payload the capture reader takes as marker text.
fn write_push_marker(arguments: [u64; 6]) -> u64 {
    let text = arguments[1];
    let Some(name) = read_marker_text(text) else {
        return 0;
    };
    // The payload always keeps the terminator, so a name whose length is a multiple of four does
    // not run into the following packet's header when a reader scans it as a string.
    let payload_dwords = u32::try_from(name.len().div_ceil(4))
        .unwrap_or(MAX_PACKET_DWORDS)
        .max(1);
    let count = payload_dwords + 1;
    if count > MAX_PACKET_DWORDS {
        return 0;
    }
    let Some(command) = allocate_dwords(arguments[0], count) else {
        return 0;
    };
    write_u32(command, pm4(count, IT_NOP, R_PUSH_MARKER));
    for index in 1..u64::from(count) {
        write_u32(command + index * 4, 0);
    }
    for (index, &byte) in name.iter().enumerate() {
        write_u8(
            command + 4 + u64::try_from(index).expect("the marker payload is bounded"),
            byte,
        );
    }
    command
}

/// Appends the packet that closes the innermost open marker region in a capture.
fn write_pop_marker(command_buffer: u64) -> u64 {
    let Some(command) = allocate_dwords(command_buffer, POP_MARKER_PACKET_DWORDS) else {
        return 0;
    };
    write_words(
        command,
        &[pm4(POP_MARKER_PACKET_DWORDS, IT_NOP, R_POP_MARKER), 0],
    );
    command
}

/// Reads a marker string with its terminator, or `None` if the guest range cannot be read.
fn read_marker_text(text: u64) -> Option<Vec<u8>> {
    if text == 0 {
        return Some(vec![0]);
    }
    let limit = u64::from((MAX_PACKET_DWORDS - 1) * 4);
    let mut bytes = Vec::new();
    let mut offset = 0;
    while offset < limit {
        let base = text.checked_add(offset)?;
        let mut span = MARKER_PROBE_BYTES.min(limit - offset);
        // Markers are written per draw region, so the span is proved mapped in one probe rather
        // than one probe per byte. A name that ends close to the end of its mapping is legal, so
        // the probe narrows instead of refusing the whole span.
        while span > 0 && !guest_range_is_readable(base, span) {
            span /= 2;
        }
        if span == 0 {
            return None;
        }
        for index in 0..span {
            let byte = read_u8(base + index);
            bytes.push(byte);
            if byte == 0 {
                return Some(bytes);
            }
        }
        offset += span;
    }
    None
}

/// Reads a NUL-terminated name of at most `limit` bytes, truncating anything longer.
///
/// Each byte is proved mapped before it is read: a name that runs to the end of its mapping is a
/// guest bug the driver reports, not a fault inside the accessor.
fn read_guest_name(address: u64, limit: usize) -> Option<String> {
    if address == 0 {
        return None;
    }
    let mut bytes = Vec::new();
    for offset in 0..limit {
        let byte_address = address.checked_add(u64::try_from(offset).ok()?)?;
        if !guest_range_is_readable(byte_address, 1) {
            return None;
        }
        let byte = read_u8(byte_address);
        if byte == 0 {
            break;
        }
        bytes.push(byte);
    }
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

fn low_u32(value: u64) -> u32 {
    u32::try_from(value & u64::from(u32::MAX)).expect("value is masked to 32 bits")
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

fn write_words(address: u64, words: &[u32]) {
    for (index, &word) in words.iter().enumerate() {
        write_u32(
            address + u64::try_from(index).expect("word index fits a u64") * 4,
            word,
        );
    }
}

/// Reserves `dword_count` dwords at the head of a command buffer, growing it if the guest supplied
/// a callback, and returns the address the packet must be written to.
fn allocate_dwords(command_buffer: u64, dword_count: u32) -> Option<u64> {
    if command_buffer == 0
        || command_buffer & 7 != 0
        || dword_count == 0
        || !guest_range_is_readable(command_buffer, COMMAND_BUFFER_DESCRIPTOR_BYTES)
    {
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
        let callback_address = usize::try_from(callback).ok()?;
        // SAFETY: the descriptor field this address came from is the guest's own SysV growth
        // callback, whose signature the Gen5 command-buffer layout fixes.
        let callback: GrowCallback = unsafe { std::mem::transmute(callback_address) };
        // SAFETY: the callback owns the cursors it rewrites and reports whether it grew the buffer.
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
    if !guest_range_is_readable(cursor_up, byte_count) {
        return None;
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
        words: Box<[u32; 64]>,
    }

    impl CommandFixture {
        fn new() -> Self {
            let mut fixture = Self {
                descriptor: Box::new([0_u64; 7]),
                words: Box::new([0_u32; 64]),
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

    fn call_with(nid: &str, arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
        dispatch(nid, arguments, state).expect("known test NID")
    }

    /// A stream the submission parser accepts: one type-2 filler dword and one empty type-3 NOP.
    fn command_stream() -> Box<[u32; 4]> {
        Box::new([0x8000_0000, pm4(3, IT_NOP, 0), 0, 0])
    }

    #[test]
    fn a_graphics_submission_parses_its_stream_and_requests_a_completion_event() {
        let (sender, receiver) = remu_gpu::gpu_event_channel();
        remu_gpu::install_thread_local_event_sink(sender);
        let stream = command_stream();
        let address = stream.as_ptr().addr() as u64;
        let mut state = ProbeDispatchState::default();

        assert_eq!(
            call_with("AhGvpITrf4M", args(&[0, address, 4]), &mut state),
            0
        );
        assert_eq!(state.agc_submission_count, 1);
        assert_eq!(state.pending_agc_completions.len(), 1);
        assert!(matches!(
            receiver.recv(),
            Ok(remu_gpu::GpuEvent::QueueSubmission {
                queue_id: 0,
                events,
            }) if matches!(
                events.last(),
                Some(remu_gpu::GpuEvent::SubmissionBoundary(1))
            )
        ));
        remu_gpu::clear_thread_local_event_sink();
    }

    #[test]
    fn a_graphics_submission_to_an_unknown_ring_is_rejected() {
        let stream = command_stream();
        let address = stream.as_ptr().addr() as u64;

        assert_eq!(
            call("AhGvpITrf4M", args(&[2, address, 4])),
            AGC_ERROR_INVALID_SUBMIT_QUEUE
        );
    }

    #[test]
    fn a_graphics_submission_larger_than_a_command_buffer_is_rejected() {
        let stream = command_stream();
        let address = stream.as_ptr().addr() as u64;

        assert_eq!(
            call(
                "AhGvpITrf4M",
                args(&[0, address, u64::from(MAX_SUBMIT_DWORDS) + 1])
            ),
            AGC_ERROR_SUBMIT_SIZE_TOO_LARGE
        );
    }

    #[test]
    fn a_null_or_misaligned_submission_pointer_is_rejected_rather_than_faulting() {
        let stream = command_stream();
        let address = stream.as_ptr().addr() as u64;

        assert_eq!(
            call("AhGvpITrf4M", args(&[0, 0, 4])),
            AGC_ERROR_PARAMETER_NULL
        );
        assert_eq!(
            call("AhGvpITrf4M", args(&[0, address + 1, 4])),
            AGC_ERROR_ALIGNMENT
        );
        assert_eq!(
            call("AhGvpITrf4M", args(&[0, u64::MAX - 15, 4])),
            AGC_ERROR_OUT_OF_BOUNDS
        );
        assert_eq!(call("AhGvpITrf4M", args(&[0, 0, 0])), 0);
    }

    #[test]
    fn a_multi_buffer_submission_runs_every_listed_command_buffer() {
        let (sender, receiver) = remu_gpu::gpu_event_channel();
        remu_gpu::install_thread_local_event_sink(sender);
        let stream = command_stream();
        let address = stream.as_ptr().addr() as u64;
        let addresses = [address, address];
        let sizes = [4_u32, 4];
        let mut state = ProbeDispatchState::default();

        assert_eq!(
            call_with(
                "+T8Xo6LtFJI",
                args(&[
                    0,
                    2,
                    addresses.as_ptr().addr() as u64,
                    sizes.as_ptr().addr() as u64,
                ]),
                &mut state,
            ),
            0
        );
        assert_eq!(state.agc_submission_count, 2);
        assert_eq!(state.pending_agc_completions.len(), 2);
        for sequence in 1..=2 {
            assert!(matches!(
                receiver.recv(),
                Ok(remu_gpu::GpuEvent::QueueSubmission {
                    queue_id: 0,
                    events,
                }) if matches!(
                    events.last(),
                    Some(remu_gpu::GpuEvent::SubmissionBoundary(actual)) if *actual == sequence
                )
            ));
        }
        remu_gpu::clear_thread_local_event_sink();
    }

    #[test]
    fn a_multi_buffer_submission_runs_nothing_when_one_entry_is_invalid() {
        let stream = command_stream();
        let address = stream.as_ptr().addr() as u64;
        let addresses = [address, 0];
        let sizes = [4_u32, 4];
        let mut state = ProbeDispatchState::default();

        assert_eq!(
            call_with(
                "+T8Xo6LtFJI",
                args(&[
                    0,
                    2,
                    addresses.as_ptr().addr() as u64,
                    sizes.as_ptr().addr() as u64,
                ]),
                &mut state,
            ),
            AGC_ERROR_PARAMETER_NULL
        );
        assert_eq!(state.agc_submission_count, 0);
        assert!(state.pending_agc_completions.is_empty());
    }

    #[test]
    fn a_multi_buffer_submission_rejects_null_and_unreadable_argument_arrays() {
        let sizes = [4_u32];

        assert_eq!(
            call(
                "+T8Xo6LtFJI",
                args(&[0, 1, 0, sizes.as_ptr().addr() as u64])
            ),
            AGC_ERROR_PARAMETER_NULL
        );
        assert_eq!(
            call(
                "+T8Xo6LtFJI",
                args(&[0, 0x1000, 0x40, sizes.as_ptr().addr() as u64])
            ),
            AGC_ERROR_OUT_OF_BOUNDS
        );
        assert_eq!(call("+T8Xo6LtFJI", args(&[0, 0, 0, 0])), 0);
    }

    #[test]
    fn deleting_an_event_drops_its_registration_and_undelivered_interrupts() {
        let mut state = ProbeDispatchState::default();
        let mut equeue = 0_u64;
        assert_eq!(
            crate::kernel_create_equeue((&raw mut equeue).addr() as u64, &mut state),
            0
        );
        assert_eq!(
            crate::agc_driver_add_eq_event(args(&[equeue, 0x0000_0002, 0xfeed]), &mut state),
            0
        );
        queue_agc_event_write(&mut state, 7);
        assert_eq!(state.pending_events.len(), 1);
        state
            .pending_events
            .front_mut()
            .expect("queued event")
            .agc_graphics_completions = 2;

        // The event type is a four-byte value in a register whose upper half the caller never
        // sets, so a stale upper half must not hide the registration.
        assert_eq!(
            call_with(
                "DL2RXaXOy88",
                args(&[equeue, 0xdead_0000_0000_0002]),
                &mut state
            ),
            0
        );
        assert!(state.kernel_event_registrations.is_empty());
        assert!(state.pending_events.is_empty());
        assert_eq!(
            state
                .agc_completion_telemetry
                .cancelled_graphics_notifications,
            2
        );
    }

    #[test]
    fn deleting_an_event_no_queue_ever_registered_reports_not_found() {
        let mut state = ProbeDispatchState::default();
        let mut equeue = 0_u64;
        assert_eq!(
            crate::kernel_create_equeue((&raw mut equeue).addr() as u64, &mut state),
            0
        );
        let not_found = sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);

        assert_eq!(
            call_with("DL2RXaXOy88", args(&[equeue, 3]), &mut state),
            not_found
        );
        assert_eq!(
            call_with("DL2RXaXOy88", args(&[equeue + 100, 3]), &mut state),
            not_found
        );
    }

    #[test]
    fn the_event_accessors_read_the_delivered_kernel_event_fields() {
        // The identifier occupies the first quadword and the payload the third, so a delivered
        // event is built here exactly as `sceKernelWaitEqueue` fills one.
        let event = Box::new([0xdead_beef_0000_0103_u64, 0, 0xffff_ffff, 0]);
        let address = event.as_ptr().addr() as u64;

        assert_eq!(call("5CdQTZIQPxM", args(&[address])), 0x0000_0103);
        assert_eq!(call("Zw7uUVPulbw", args(&[address])), 0x07ff_ffff);
    }

    #[test]
    fn a_null_event_pointer_reports_an_empty_event_rather_than_faulting() {
        assert_eq!(call("5CdQTZIQPxM", args(&[0])), 0);
        assert_eq!(call("Zw7uUVPulbw", args(&[0])), 0);
    }

    #[test]
    fn a_workload_stream_is_named_once_and_released_once() {
        let name = c"physics";
        let address = name.as_ptr().addr() as u64;

        assert_eq!(call("3AyTaWcF-H8", args(&[9, address])), 0);
        assert_eq!(
            workload_streams().get(&9).map(String::as_str),
            Some("physics")
        );
        assert_eq!(call("n5ElQVYsU1A", args(&[9])), 0);
        assert_eq!(
            call("n5ElQVYsU1A", args(&[9])),
            AGC_ERROR_WORKLOAD_STREAM_NOT_ALLOCATED
        );
    }

    #[test]
    fn a_workload_stream_outside_the_selectable_range_or_without_a_name_is_rejected() {
        let name = c"audio";
        let address = name.as_ptr().addr() as u64;

        assert_eq!(
            call("3AyTaWcF-H8", args(&[0, address])),
            AGC_ERROR_INVALID_WORKLOAD_STREAM_ID
        );
        assert_eq!(
            call("3AyTaWcF-H8", args(&[32, address])),
            AGC_ERROR_INVALID_WORKLOAD_STREAM_ID
        );
        assert_eq!(
            call("3AyTaWcF-H8", args(&[4, 0])),
            AGC_ERROR_INVALID_WORKLOAD_STREAM_NAME_PTR
        );
        assert_eq!(
            call("n5ElQVYsU1A", args(&[64])),
            AGC_ERROR_INVALID_WORKLOAD_STREAM_ID
        );
    }

    #[test]
    fn the_flip_and_wait_packets_match_the_sizes_the_driver_reports() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();

        assert_eq!(
            call(
                "cwbxjPSJ7WQ",
                args(&[buffer, 1, 2, 3, 0x1122_3344_5566_7788])
            ),
            fixture.start()
        );
        assert_eq!(
            &fixture.words[..6],
            &[pm4(6, IT_NOP, R_FLIP), 1, 2, 3, 0x5566_7788, 0x1122_3344]
        );

        assert_eq!(
            call("u8BkdHb1+Po", args(&[buffer, 1, 5])),
            fixture.start() + 24
        );
        assert_eq!(
            &fixture.words[6..13],
            &[pm4(7, IT_NOP, R_WAIT_FLIP_DONE), 1, 5, 0, 0, 0, 0]
        );
        assert_eq!(fixture.cursor(), fixture.start() + 52);

        assert_eq!(call("2PrsbRYyZi4", [0; 6]), 6);
        assert_eq!(call("0MtUJ3BpGhE", [0; 6]), 7);
    }

    #[test]
    fn a_marker_region_carries_its_name_with_the_terminator_inside_the_packet() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();
        let text = c"shadow";

        assert_eq!(
            call(
                "+b34-CLWc0s",
                args(&[buffer, text.as_ptr().addr() as u64, 0xff])
            ),
            fixture.start()
        );
        assert_eq!(fixture.words[0], pm4(3, IT_NOP, R_PUSH_MARKER));
        assert_eq!(&fixture.words[1..3], &[0x6461_6873, 0x0000_776f]);

        assert_eq!(call("LEnn-4ARRJM", args(&[buffer])), fixture.start() + 12);
        assert_eq!(&fixture.words[3..5], &[pm4(2, IT_NOP, R_POP_MARKER), 0]);
        assert_eq!(fixture.cursor(), fixture.start() + 20);
    }

    #[test]
    fn a_marker_with_no_name_still_writes_a_terminated_payload() {
        let mut fixture = CommandFixture::new();
        let buffer = fixture.address();

        assert_eq!(call("+b34-CLWc0s", args(&[buffer, 0, 0])), fixture.start());
        assert_eq!(fixture.words[0], pm4(2, IT_NOP, R_PUSH_MARKER));
        assert_eq!(fixture.words[1], 0);
    }

    #[test]
    fn a_packet_writer_without_command_buffer_space_reports_no_packet() {
        let mut descriptor = Box::new([0_u64; 7]);
        let buffer = descriptor.as_mut_ptr().addr() as u64;

        assert_eq!(call("cwbxjPSJ7WQ", args(&[buffer, 1, 2, 3, 4])), 0);
        assert_eq!(call("u8BkdHb1+Po", args(&[buffer, 1, 2])), 0);
        assert_eq!(call("LEnn-4ARRJM", args(&[buffer])), 0);
        assert_eq!(call("+b34-CLWc0s", args(&[0, 0, 0])), 0);
    }

    #[test]
    fn capture_and_validation_control_reports_the_absent_tooling_module() {
        for nid in [
            "Xq5WmbwPTnQ",
            "SAfhzJPcjuk",
            "FOwvmNlFLjM",
            "+iAOE3jCnkk",
            "WHIOMbb+iIU",
            "chJWZcNSzjk",
            "ICaGtkEIXTk",
            "8F5j7U-Bq6s",
        ] {
            assert_eq!(
                call(nid, args(&[0, 0])),
                AGC_ERROR_GPU_CAPTURE_NOT_AVAILABLE,
                "{nid}"
            );
        }
        for nid in ["Ddwk4gLT5j0", "+TN0oRTBxJQ", "qspAL8bgcBY", "HMnVBVUyajk"] {
            assert_eq!(call(nid, [0; 6]), 0, "{nid}");
        }
        assert_eq!(call("rJUyMrDdxJg", [0; 6]), SHADER_DEBUGGING_NOT_AVAILABLE);
    }

    #[test]
    fn resource_registration_reports_the_disabled_pa_debug_target_setting() {
        for nid in [
            "F0Y42t-3e18",
            "AOLcoIkQDgM",
            "X-Nm5KLREeg",
            "U9ueyEhSkF4",
            "W5z4eZrjEas",
            "emP3ckeS2uo",
            "5l3IfCFJxBs",
            "LepGrgk77sM",
            "M9yBzRKkjPc",
            "NghWEUXp1qM",
            "mXn+K9E-wOA",
            "rI9lNAXPMIw",
            "ls4jfY576lw",
            "VOMSpd9+vxU",
            "SCoAN5fYlUM",
            "ZLJk9r2+2Aw",
            "pWLG7WOpVcw",
            "F0ZXt5q0ZTA",
            "uJziRsODk1c",
        ] {
            assert_eq!(
                call(nid, [0; 6]),
                AGC_ERROR_RESOURCE_REGISTRATION_NO_PA_DEBUG,
                "{nid}"
            );
        }
    }

    #[test]
    fn an_unowned_nid_is_left_for_the_next_module() {
        let mut state = ProbeDispatchState::default();

        for nid in [
            "zmw2uVSEj94",
            "UM9b9NunSrE",
            "i6bfTi13ApA",
            "not-an-agc-nid",
        ] {
            assert_eq!(dispatch(nid, [0; 6], &mut state), None, "{nid}");
        }
    }
}
