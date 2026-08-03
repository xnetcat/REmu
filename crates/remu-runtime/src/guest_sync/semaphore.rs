// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::{
    SCE_KERNEL_ERROR_BUSY, SCE_KERNEL_ERROR_INTERRUPTED, SCE_KERNEL_ERROR_INVALID_ARGUMENT,
    SCE_KERNEL_ERROR_NOT_FOUND, SCE_KERNEL_ERROR_TIMED_OUT, assert_wait_is_off_the_dispatch_lock,
    current_guest_thread_handle, guest_exception_is_pending, low_i32, low_u32, read_guest_cstring,
    read_guest_u32, sce_kernel_error, trace_guest_sync, trace_logarithmic_sequence,
    write_guest_u32,
};

use super::state::{GuestSemaphoreEntry, GuestSemaphoreState};
use super::guest_sync;

/// Test/helper constructor for a per-object semaphore entry.
#[cfg(test)]
pub(crate) fn guest_semaphore_entry(count: i32, maximum: i32) -> Arc<GuestSemaphoreEntry> {
    Arc::new(GuestSemaphoreEntry {
        state: Mutex::new(GuestSemaphoreState { count, maximum }),
        wake: Condvar::new(),
    })
}

fn lookup_semaphore(handle: u32) -> Option<Arc<GuestSemaphoreEntry>> {
    let sync = guest_sync();
    let state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.semaphores.get(&handle).cloned()
}

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
        Arc::new(GuestSemaphoreEntry {
            state: Mutex::new(GuestSemaphoreState {
                count: initial,
                maximum,
            }),
            wake: Condvar::new(),
        }),
    );
    write_guest_u32(output, handle);
    if trace_guest_sync() {
        eprintln!(
            "[REMu][SEMA] create handle={handle} name={:?} initial={initial} maximum={maximum}",
            read_guest_cstring(arguments[1]).unwrap_or_default()
        );
    }
    0
}

/// Takes a count from a semaphore when one is already available, and never waits.
pub(crate) fn guest_semaphore_poll(arguments: [u64; 6]) -> u64 {
    let handle = low_u32(arguments[0]);
    let needed = low_i32(arguments[1]);
    if needed <= 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(entry) = lookup_semaphore(handle) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let mut semaphore = entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if semaphore.count < needed {
        return sce_kernel_error(SCE_KERNEL_ERROR_BUSY);
    }
    semaphore.count -= needed;
    0
}

pub(crate) fn guest_semaphore_wait(arguments: [u64; 6]) -> u64 {
    let handle = low_u32(arguments[0]);
    let needed = low_i32(arguments[1]);
    if needed <= 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let timeout_pointer = arguments[2];
    let timeout = (timeout_pointer != 0)
        .then(|| Duration::from_micros(u64::from(read_guest_u32(timeout_pointer))));
    let deadline = timeout.and_then(|timeout| Instant::now().checked_add(timeout));
    let guest_thread = current_guest_thread_handle();
    let wait_sequence = {
        let sync = guest_sync();
        let mut registry = sync
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        registry.semaphore_wait_count = registry.semaphore_wait_count.saturating_add(1);
        registry.semaphore_wait_count
    };
    let Some(entry) = lookup_semaphore(handle) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let trace_call = trace_guest_sync() && trace_logarithmic_sequence(wait_sequence, 8);
    let mut logged_block = false;
    let mut semaphore = entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        if needed > semaphore.maximum {
            return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
        }
        if semaphore.count >= needed {
            semaphore.count -= needed;
            write_semaphore_timeout_remaining(timeout_pointer, deadline);
            if trace_call {
                eprintln!(
                    "[REMu][SEMA] acquire sequence={wait_sequence} handle={handle} needed={needed} remaining={}",
                    semaphore.count
                );
            }
            return 0;
        }
        if guest_exception_is_pending(guest_thread) {
            write_semaphore_timeout_remaining(timeout_pointer, deadline);
            return sce_kernel_error(SCE_KERNEL_ERROR_INTERRUPTED);
        }
        if trace_call && !logged_block {
            eprintln!(
                "[REMu][SEMA] wait sequence={wait_sequence} handle={handle} needed={needed} available={}",
                semaphore.count
            );
            logged_block = true;
        }
        assert_wait_is_off_the_dispatch_lock("sceKernelWaitSema");
        if let Some(deadline) = deadline {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                write_guest_u32(timeout_pointer, 0);
                return sce_kernel_error(SCE_KERNEL_ERROR_TIMED_OUT);
            };
            let (next, _) = entry
                .wake
                .wait_timeout(semaphore, remaining)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            semaphore = next;
        } else {
            semaphore = entry
                .wake
                .wait(semaphore)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        // The registry may have deleted this handle while we were parked.
        if lookup_semaphore(handle).is_none() {
            return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
        }
    }
}

fn write_semaphore_timeout_remaining(timeout_pointer: u64, deadline: Option<Instant>) {
    if timeout_pointer == 0 {
        return;
    }
    let remaining = deadline
        .and_then(|deadline| deadline.checked_duration_since(Instant::now()))
        .unwrap_or(Duration::ZERO)
        .as_micros()
        .min(u128::from(u32::MAX)) as u32;
    write_guest_u32(timeout_pointer, remaining);
}

pub(super) fn guest_semaphore_delete(handle: u64) -> u64 {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(entry) = state.semaphores.remove(&low_u32(handle)) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    drop(state);
    entry.wake.notify_all();
    0
}

pub(super) fn guest_semaphore_signal(arguments: [u64; 6]) -> u64 {
    let handle = low_u32(arguments[0]);
    let amount = low_i32(arguments[1]);
    if amount <= 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let signal_sequence = {
        let sync = guest_sync();
        let mut registry = sync
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        registry.semaphore_signal_count = registry.semaphore_signal_count.saturating_add(1);
        registry.semaphore_signal_count
    };
    let Some(entry) = lookup_semaphore(handle) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let mut semaphore = entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(next) = semaphore.count.checked_add(amount) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if next > semaphore.maximum {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    semaphore.count = next;
    if trace_guest_sync() && trace_logarithmic_sequence(signal_sequence, 8) {
        eprintln!(
            "[REMu][SEMA] signal sequence={signal_sequence} handle={handle} amount={amount} available={}",
            semaphore.count
        );
    }
    entry.wake.notify_all();
    0
}
