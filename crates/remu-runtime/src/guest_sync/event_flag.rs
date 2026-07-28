// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::sync::atomic::Ordering;

use std::time::{Duration, Instant};

use crate::{
    SCE_KERNEL_ERROR_BUSY, SCE_KERNEL_ERROR_INTERRUPTED, SCE_KERNEL_ERROR_INVALID_ARGUMENT,
    SCE_KERNEL_ERROR_MEMORY_FAULT, SCE_KERNEL_ERROR_NOT_FOUND, SCE_KERNEL_ERROR_TIMED_OUT,
    current_guest_thread_handle, current_guest_thread_name, guest_exception_is_pending, low_u32,
    read_guest_cstring, read_guest_u32, sce_kernel_error, trace_guest_sync, write_guest_u32,
    write_guest_u64,
};

use super::{EVENT_FLAG_SET_SEQUENCE, EVENT_FLAG_WAIT_SEQUENCE, GuestEventFlagState, guest_sync};

pub(super) const EVENT_FLAG_WAIT_AND: u32 = 0x01;

pub(crate) const EVENT_FLAG_WAIT_OR: u32 = 0x02;

pub(super) const EVENT_FLAG_CLEAR_ALL: u32 = 0x10;

pub(super) const EVENT_FLAG_CLEAR_PATTERN: u32 = 0x20;

pub(super) fn event_flag_attributes_valid(attributes: u32) -> bool {
    matches!(attributes & 0x0f, 0 | 0x01 | 0x02)
        && matches!(attributes & 0xf0, 0 | 0x10 | 0x20)
        && attributes & !0x33 == 0
}

pub(super) fn event_flag_wait_mode_valid(wait_mode: u32) -> bool {
    matches!(wait_mode & 0x0f, EVENT_FLAG_WAIT_AND | EVENT_FLAG_WAIT_OR)
        && matches!(
            wait_mode & 0xf0,
            0 | EVENT_FLAG_CLEAR_ALL | EVENT_FLAG_CLEAR_PATTERN
        )
        && wait_mode & !0x33 == 0
}

pub(super) fn event_flag_satisfied(bits: u64, pattern: u64, wait_mode: u32) -> bool {
    if wait_mode & 0x0f == EVENT_FLAG_WAIT_AND {
        bits & pattern == pattern
    } else {
        bits & pattern != 0
    }
}

pub(super) fn event_flag_apply_clear(bits: &mut u64, pattern: u64, wait_mode: u32) {
    match wait_mode & 0xf0 {
        EVENT_FLAG_CLEAR_ALL => *bits = 0,
        EVENT_FLAG_CLEAR_PATTERN => *bits &= !pattern,
        _ => {}
    }
}

pub(super) fn guest_event_flag_create(arguments: [u64; 6]) -> u64 {
    let output = arguments[0];
    let Some(name) = read_guest_cstring(arguments[1]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    let attributes = low_u32(arguments[2]);
    if output == 0
        || arguments[1] == 0
        || arguments[4] != 0
        || name.len() > 31
        || !event_flag_attributes_valid(attributes)
    {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let handle = loop {
        state.next_event_flag_handle = state.next_event_flag_handle.wrapping_add(1).max(1);
        if !state
            .event_flags
            .contains_key(&state.next_event_flag_handle)
        {
            break state.next_event_flag_handle;
        }
    };
    state.event_flags.insert(
        handle,
        GuestEventFlagState {
            name: name.clone(),
            attributes,
            bits: arguments[3],
            waiters: 0,
            cancellation_generation: 0,
        },
    );
    write_guest_u64(output, handle);
    if trace_guest_sync() {
        eprintln!(
            "[REMu][EVENT_FLAG] create handle={handle} name={name:?} attr={attributes:#x} bits={:#x} thread={:?}/{:#x}",
            arguments[3],
            current_guest_thread_name(),
            current_guest_thread_handle(),
        );
    }
    0
}

pub(super) fn guest_event_flag_delete(handle: u64) -> u64 {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(flag) = state.event_flags.remove(&handle) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if trace_guest_sync() {
        eprintln!(
            "[REMu][EVENT_FLAG] delete handle={handle} name={:?} attr={:#x} waiters={} thread={:?}/{:#x}",
            flag.name,
            flag.attributes,
            flag.waiters,
            current_guest_thread_name(),
            current_guest_thread_handle(),
        );
    }
    sync.1.notify_all();
    0
}

pub(super) fn guest_event_flag_set(handle: u64, pattern: u64) -> u64 {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(flag) = state.event_flags.get_mut(&handle) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    flag.bits |= pattern;
    let bits = flag.bits;
    let name = flag.name.clone();
    let waiters = flag.waiters;
    if trace_guest_sync() {
        let sequence = EVENT_FLAG_SET_SEQUENCE
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);
        if sequence <= 32 || sequence.is_power_of_two() {
            eprintln!(
                "[REMu][EVENT_FLAG] set sequence={sequence} handle={handle} name={name:?} pattern={pattern:#x} bits={bits:#x} waiters={waiters} thread={:?}/{:#x}",
                current_guest_thread_name(),
                current_guest_thread_handle(),
            );
        }
    }
    sync.1.notify_all();
    0
}

pub(super) fn guest_event_flag_clear(handle: u64, mask: u64) -> u64 {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(flag) = state.event_flags.get_mut(&handle) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    flag.bits &= mask;
    0
}

pub(super) fn guest_event_flag_poll(arguments: [u64; 6]) -> u64 {
    let pattern = arguments[1];
    let wait_mode = low_u32(arguments[2]);
    if pattern == 0 || !event_flag_wait_mode_valid(wait_mode) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(flag) = state.event_flags.get_mut(&arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if arguments[3] != 0 {
        write_guest_u64(arguments[3], flag.bits);
    }
    if !event_flag_satisfied(flag.bits, pattern, wait_mode) {
        return sce_kernel_error(SCE_KERNEL_ERROR_BUSY);
    }
    event_flag_apply_clear(&mut flag.bits, pattern, wait_mode);
    0
}

pub(crate) fn guest_event_flag_wait(arguments: [u64; 6]) -> u64 {
    let handle = arguments[0];
    let pattern = arguments[1];
    let wait_mode = low_u32(arguments[2]);
    if pattern == 0 || !event_flag_wait_mode_valid(wait_mode) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let timeout = (arguments[4] != 0).then(|| u64::from(read_guest_u32(arguments[4])));
    let deadline = timeout
        .and_then(|microseconds| Instant::now().checked_add(Duration::from_micros(microseconds)));
    let sync = guest_sync();
    let guest_thread = current_guest_thread_handle();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(flag) = state.event_flags.get_mut(&handle) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let wait_sequence = EVENT_FLAG_WAIT_SEQUENCE
        .fetch_add(1, Ordering::Relaxed)
        .saturating_add(1);
    let trace_wait = trace_guest_sync() && (wait_sequence <= 32 || wait_sequence.is_power_of_two());
    if trace_wait {
        eprintln!(
            "[REMu][EVENT_FLAG] wait sequence={wait_sequence} handle={handle} name={:?} pattern={pattern:#x} mode={wait_mode:#x} bits={:#x} timeout_us={timeout:?} thread={:?}/{guest_thread:#x}",
            flag.name,
            flag.bits,
            current_guest_thread_name(),
        );
    }
    let cancellation_generation = flag.cancellation_generation;
    if event_flag_satisfied(flag.bits, pattern, wait_mode) {
        if arguments[3] != 0 {
            write_guest_u64(arguments[3], flag.bits);
        }
        event_flag_apply_clear(&mut flag.bits, pattern, wait_mode);
        if trace_wait {
            eprintln!(
                "[REMu][EVENT_FLAG] wait satisfied sequence={wait_sequence} handle={handle} immediately=true remaining={:#x}",
                flag.bits
            );
        }
        return 0;
    }
    flag.waiters = flag.waiters.saturating_add(1);
    loop {
        let Some(flag) = state.event_flags.get_mut(&handle) else {
            return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
        };
        if guest_exception_is_pending(guest_thread) {
            if arguments[3] != 0 {
                write_guest_u64(arguments[3], flag.bits);
            }
            flag.waiters = flag.waiters.saturating_sub(1);
            return sce_kernel_error(SCE_KERNEL_ERROR_INTERRUPTED);
        }
        if flag.cancellation_generation != cancellation_generation {
            flag.waiters = flag.waiters.saturating_sub(1);
            return sce_kernel_error(SCE_KERNEL_ERROR_INTERRUPTED);
        }
        if event_flag_satisfied(flag.bits, pattern, wait_mode) {
            if arguments[3] != 0 {
                write_guest_u64(arguments[3], flag.bits);
            }
            event_flag_apply_clear(&mut flag.bits, pattern, wait_mode);
            flag.waiters = flag.waiters.saturating_sub(1);
            if trace_wait {
                eprintln!(
                    "[REMu][EVENT_FLAG] wait satisfied sequence={wait_sequence} handle={handle} immediately=false remaining={:#x}",
                    flag.bits
                );
            }
            return 0;
        }
        if let Some(deadline) = deadline {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                if arguments[3] != 0 {
                    write_guest_u64(arguments[3], flag.bits);
                }
                if arguments[4] != 0 {
                    write_guest_u32(arguments[4], 0);
                }
                flag.waiters = flag.waiters.saturating_sub(1);
                return sce_kernel_error(SCE_KERNEL_ERROR_TIMED_OUT);
            };
            let (next_state, wait_result) = sync
                .1
                .wait_timeout(state, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next_state;
            if wait_result.timed_out() {
                continue;
            }
        } else {
            state = sync
                .1
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

pub(super) fn guest_event_flag_cancel(arguments: [u64; 6]) -> u64 {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(flag) = state.event_flags.get_mut(&arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if arguments[2] != 0 {
        write_guest_u32(arguments[2], flag.waiters);
    }
    flag.bits = arguments[1];
    flag.cancellation_generation = flag.cancellation_generation.wrapping_add(1);
    sync.1.notify_all();
    0
}
