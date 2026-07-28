// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::{
    SCE_KERNEL_ERROR_INTERRUPTED, SCE_KERNEL_ERROR_NOT_FOUND, current_guest_thread_handle,
    guest_exception_is_pending, low_i32, low_u32, read_guest_cstring, sce_kernel_error,
    trace_guest_sync, trace_logarithmic_sequence,
};

use super::{GuestSemaphoreState, guest_sync};

pub(super) fn guest_semaphore_create(arguments: [u64; 6]) -> u64 {
    let output = arguments[0];
    let initial = low_i32(arguments[3]);
    let maximum = low_i32(arguments[4]);
    if output == 0 || initial < 0 || maximum <= 0 || initial > maximum {
        return u64::MAX;
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.next_semaphore_handle = state.next_semaphore_handle.wrapping_add(1).max(1);
    let handle = state.next_semaphore_handle;
    state.semaphores.insert(
        handle,
        GuestSemaphoreState {
            count: initial,
            maximum,
        },
    );
    // SAFETY: sceKernelCreateSema supplies a writable 32-bit handle output.
    unsafe {
        std::ptr::write_unaligned(output as *mut u32, handle);
    }
    if trace_guest_sync() {
        eprintln!(
            "[REMu][SEMA] create handle={handle} name={:?} initial={initial} maximum={maximum}",
            read_guest_cstring(arguments[1]).unwrap_or_default()
        );
    }
    0
}

pub(crate) fn guest_semaphore_wait(arguments: [u64; 6]) -> u64 {
    let handle = low_u32(arguments[0]);
    let needed = low_i32(arguments[1]);
    if needed <= 0 {
        return u64::MAX;
    }
    let sync = guest_sync();
    let guest_thread = current_guest_thread_handle();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.semaphore_wait_count = state.semaphore_wait_count.saturating_add(1);
    let wait_sequence = state.semaphore_wait_count;
    let trace_call = trace_guest_sync() && trace_logarithmic_sequence(wait_sequence, 8);
    let mut logged_block = false;
    loop {
        let Some(semaphore) = state.semaphores.get_mut(&handle) else {
            return u64::MAX;
        };
        if semaphore.count >= needed {
            semaphore.count -= needed;
            if trace_call {
                eprintln!(
                    "[REMu][SEMA] acquire sequence={wait_sequence} handle={handle} needed={needed} remaining={}",
                    semaphore.count
                );
            }
            return 0;
        }
        if guest_exception_is_pending(guest_thread) {
            return sce_kernel_error(SCE_KERNEL_ERROR_INTERRUPTED);
        }
        if trace_call && !logged_block {
            eprintln!(
                "[REMu][SEMA] wait sequence={wait_sequence} handle={handle} needed={needed} available={}",
                semaphore.count
            );
            logged_block = true;
        }
        state = sync
            .1
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

pub(super) fn guest_semaphore_delete(handle: u64) -> u64 {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.semaphores.remove(&low_u32(handle)).is_none() {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    sync.1.notify_all();
    0
}

pub(super) fn guest_semaphore_signal(arguments: [u64; 6]) -> u64 {
    let handle = low_u32(arguments[0]);
    let amount = low_i32(arguments[1]);
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.semaphore_signal_count = state.semaphore_signal_count.saturating_add(1);
    let signal_sequence = state.semaphore_signal_count;
    let Some(semaphore) = state.semaphores.get_mut(&handle) else {
        return u64::MAX;
    };
    semaphore.count = semaphore
        .count
        .saturating_add(amount)
        .min(semaphore.maximum);
    if trace_guest_sync() && trace_logarithmic_sequence(signal_sequence, 8) {
        eprintln!(
            "[REMu][SEMA] signal sequence={signal_sequence} handle={handle} amount={amount} available={}",
            semaphore.count
        );
    }
    sync.1.notify_all();
    0
}
