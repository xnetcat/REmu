// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::{
    DEFAULT_THREAD_AFFINITY_MASK, DEFAULT_THREAD_PRIORITY, ProbeDispatchState,
    SCE_KERNEL_ERROR_BUSY, SCE_KERNEL_ERROR_INVALID_ARGUMENT, SCE_KERNEL_ERROR_NOT_FOUND,
    current_guest_thread_handle, guest_sync, low_i32, low_u32, pthread_attr_state,
    sce_kernel_error, signed_i32_return, update_pthread_attr, write_guest_bytes, write_guest_u32,
    write_guest_u64,
};

use super::set_guest_errno;

/// Scheduling policy identifiers accepted by the priority-range queries.
pub(super) const SCHED_FIFO: i32 = 1;

pub(super) const SCHED_OTHER: i32 = 2;

pub(super) const SCHED_RR: i32 = 3;

/// Most and least urgent thread priorities the console's realtime class accepts.
///
/// The scale is inverted with respect to POSIX: a smaller number is more urgent.
pub(super) const KERNEL_PRIORITY_HIGHEST: i32 = 256;

pub(super) const KERNEL_PRIORITY_LOWEST: i32 = 767;

/// Bytes copied for a thread name, terminator included.
///
/// The name accessor takes no capacity argument, so the ABI fixes one maximum every caller must
/// have room for. Names longer than this are truncated rather than overrunning that buffer.
pub(super) const PTHREAD_NAME_CAPACITY: usize = 32;

/// Reports the CPU affinity mask of a running guest thread.
pub(super) fn pthread_get_affinity(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (thread, output) = (arguments[0], arguments[1]);
    if thread == 0 || output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let mask = state
        .thread_attributes
        .get(&thread)
        .map_or(DEFAULT_THREAD_AFFINITY_MASK, |attributes| {
            attributes.affinity_mask
        });
    write_guest_u64(output, mask);
    0
}

/// Reports the scheduling priority of a running guest thread.
pub(super) fn pthread_get_priority(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (thread, output) = (arguments[0], arguments[1]);
    if thread == 0 || output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let priority = state
        .thread_attributes
        .get(&thread)
        .map_or(DEFAULT_THREAD_PRIORITY, |attributes| {
            attributes.schedule_priority
        });
    write_guest_u32(output, priority.cast_unsigned());
    0
}

/// Copies a running guest thread's name into the caller's buffer.
pub(super) fn pthread_get_name(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (thread, output) = (arguments[0], arguments[1]);
    if thread == 0 || output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(object) = state.pthread_threads.get(&thread) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let name = object.name.as_bytes();
    let mut buffer = Vec::with_capacity(PTHREAD_NAME_CAPACITY);
    buffer.extend_from_slice(&name[..name.len().min(PTHREAD_NAME_CAPACITY - 1)]);
    buffer.push(0);
    write_guest_bytes(output, &buffer);
    0
}

/// Reports whether an attribute template creates joinable or detached threads.
pub(super) fn pthread_attr_get_detach_state(
    arguments: [u64; 6],
    state: &ProbeDispatchState,
) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(attributes) = pthread_attr_state(arguments[0], state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    write_guest_u32(arguments[1], attributes.detach_state.cast_unsigned());
    0
}

/// Accepts the dedicated-scheduling-slot request on an attribute template.
///
/// Guest threads run on host threads whose placement the host scheduler owns, so no attribute
/// value can reserve a core for one. The template is still resolved, because a caller that names
/// an attribute the runtime never created has made an error worth reporting.
pub(super) fn pthread_attr_set_solo_schedule(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    update_pthread_attr(arguments[0], state, |_| {})
}

/// Takes a count from a semaphore when one is already available, and never waits.
pub(super) fn kernel_poll_semaphore(arguments: [u64; 6]) -> u64 {
    let handle = low_u32(arguments[0]);
    let needed = low_i32(arguments[1]);
    if needed <= 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(semaphore) = state.semaphores.get_mut(&handle) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if semaphore.count < needed {
        return sce_kernel_error(SCE_KERNEL_ERROR_BUSY);
    }
    semaphore.count -= needed;
    0
}

/// Reports one end of the priority range a scheduling policy accepts.
///
/// The realtime and time-sharing classes share one range here, because guest threads are created
/// with the same default priority whichever policy names them.
pub(super) fn scheduler_priority_bound(policy: i32, bound: i32) -> u64 {
    if matches!(policy, SCHED_FIFO | SCHED_OTHER | SCHED_RR) {
        signed_i32_return(bound)
    } else {
        set_guest_errno(libc::EINVAL);
        signed_i32_return(-1)
    }
}

/// Reports which logical CPU the calling thread is running on.
///
/// Guest threads are not pinned, so the answer is derived from the affinity the thread itself
/// asked for: the first core it allows. A thread that has never constrained its affinity reports
/// the first core of the default set.
pub(super) fn kernel_current_cpu(state: &ProbeDispatchState) -> u64 {
    let mask = state
        .thread_attributes
        .get(&current_guest_thread_handle())
        .map_or(DEFAULT_THREAD_AFFINITY_MASK, |attributes| {
            attributes.affinity_mask
        });
    if mask == 0 {
        0
    } else {
        u64::from(mask.trailing_zeros())
    }
}
