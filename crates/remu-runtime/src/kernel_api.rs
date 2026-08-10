// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Remaining libkernel and POSIX entry points.
//!
//! This module implements the libkernel and libScePosix entry points not already served by the main dispatcher.
//!
//! Three return conventions meet here and each entry point below commits to exactly one of them.
//! A `sceKernelXxx` name reports the packed kernel code directly, a plain POSIX name reports `-1`
//! and leaves the plain errno in the calling thread's cell, and a `pthread_*` name reports the
//! small errno as its value. The kernel spelling of a call and its POSIX twin therefore share one
//! implementation and differ only in the wrapper applied to its result.
//!
//! Everything here runs while the caller holds the process-wide dispatch lock, so nothing in this
//! module may park: an entry point that would have to wait for another guest thread is left
//! unclaimed rather than implemented, because that thread cannot reach its own release call
//! without the lock this one would be holding.
//!
//! [`dispatch`] is consulted by the runtime's import dispatcher only after the main table
//! misses, and returns `None` for any NID it does not own so the caller can keep looking.

use std::sync::{Mutex, MutexGuard};

use crate::{
    KERNEL_EVENT_FILTER_USER, KERNEL_FILES, KernelFileState, ProbeDispatchState,
    SCE_KERNEL_ERROR_INVALID_ARGUMENT, SCE_KERNEL_ERROR_MEMORY_FAULT, SCE_KERNEL_ERROR_NOT_FOUND,
    current_guest_thread_handle, guest_errno_address, guest_kernel_rmdir, guest_posix_file_result,
    kernel_process_time, kernel_register_event, low_i32, posix_pthread_result,
    process_guest_range_is_readable, pthread_attr_set_inherit_schedule,
    pthread_attr_set_schedule_param, pthread_get_schedule_param, pthread_rename, read_guest_u16,
    read_guest_u64, sce_kernel_error, signed_i32_return, terminate_probe, write_guest_u32,
};
mod aio;
mod filesystem;
mod net;
mod thread;
#[cfg(test)]
use self::aio::AIO_STATE_COMPLETED;
use self::aio::{
    kernel_aio_delete_request, kernel_aio_poll_request, kernel_aio_poll_requests,
    kernel_aio_submit, kernel_aio_wait_request,
};
use self::filesystem::{
    kernel_check_reachability, kernel_chmod, kernel_fsync, kernel_positional_transfer,
    kernel_rename, kernel_truncate, kernel_unlink, kernel_utimes,
};
#[cfg(test)]
use self::net::{AF_INET, AF_INET6};
use self::net::{inet_ntop, inet_pton};
use self::thread::{
    KERNEL_PRIORITY_HIGHEST, KERNEL_PRIORITY_LOWEST, kernel_current_cpu,
    pthread_attr_get_detach_state, pthread_attr_set_solo_schedule, pthread_get_affinity,
    pthread_get_name, pthread_get_priority, scheduler_priority_bound,
};
#[cfg(test)]
use self::thread::{PTHREAD_NAME_CAPACITY, SCHED_FIFO, SCHED_OTHER, SCHED_RR};

/// Exit status reported when a guest frame's stack canary is found overwritten.
const STACK_CHECK_EXIT_STATUS: i32 = 89;

/// Bytes one delivered kernel event occupies.
const KERNEL_EVENT_SIZE: usize = 0x20;
/// Byte offset of a delivered event's filter, a signed 16-bit field.
const KERNEL_EVENT_FILTER_OFFSET: u64 = 0x08;
/// Byte offset of a delivered event's user data.
const KERNEL_EVENT_USER_DATA_OFFSET: u64 = 0x18;

/// Permission bits a guest may set through `sceKernelChmod`.
const KERNEL_MODE_BITS: u32 = 0o7777;

/// Microseconds in one second, the exclusive upper bound of a `timeval`'s fractional field.
const MICROSECONDS_PER_SECOND: i64 = 1_000_000;
/// Bytes one guest `timeval` occupies: two 64-bit fields on this target.
const TIMEVAL_SIZE: u64 = 0x10;
/// Byte offset of a guest `timeval`'s fractional field.
const TIMEVAL_MICROSECONDS_OFFSET: u64 = 0x08;

/// Ticks per second reported for the guest-visible timestamp counter.
///
/// No console constant fixes this rate, so the runtime picks one and derives `sceKernelReadTsc`
/// from the same monotonic clock at exactly that scale. Only the ratio of a tick difference to
/// this frequency is observable, and that ratio is real elapsed time.
const KERNEL_TSC_FREQUENCY: u64 = 1_000_000_000;

/// Devkit input lines the runtime reports as asserted: none, since no such switches exist here.
const KERNEL_DEVKIT_INPUT_BITS: u64 = 0;
/// Hardware revision the runtime reports itself as. It models one profile, never the later one.
const KERNEL_TRINITY_MODE: u64 = 0;

/// Handles a NID belonging to this module, or returns `None` if it owns no such NID.
pub(crate) fn dispatch(
    nid: &str,
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> Option<u64> {
    if let Some(result) = dispatch_filesystem(nid, arguments, state) {
        return Some(result);
    }
    if let Some(result) = dispatch_aio(nid, arguments) {
        return Some(result);
    }
    if let Some(result) = dispatch_event(nid, arguments, state) {
        return Some(result);
    }
    if let Some(result) = dispatch_thread(nid, arguments, state) {
        return Some(result);
    }
    if let Some(result) = dispatch_process(nid, arguments, state) {
        return Some(result);
    }
    dispatch_inet_address(nid, arguments)
}

/// Handles the path and descriptor operations the main file table does not already serve.
fn dispatch_filesystem(nid: &str, arguments: [u64; 6], state: &ProbeDispatchState) -> Option<u64> {
    Some(match nid {
        "AUXVxWeJU-A" => kernel_unlink(arguments, state),
        "VAzswvTOCzI" => guest_posix_file_result(kernel_unlink(arguments, state)),
        "52NcYU9+lEo" => kernel_rename(arguments, state),
        "NN01qLRhiqU" => guest_posix_file_result(kernel_rename(arguments, state)),
        "c7ZnT7V1B98" => guest_posix_file_result(guest_kernel_rmdir(arguments, state)),
        "fgIsQ10xYVA" => kernel_chmod(arguments, state),
        "WlyEA-sLDf0" => kernel_truncate(arguments, state),
        "0Cq8ipKr9n0" => kernel_utimes(arguments, state),
        "uWyW3v98sU4" => kernel_check_reachability(arguments, state),
        "fTx66l5iWIA" => kernel_fsync(arguments),
        "+r3rMFwItV4" => kernel_positional_transfer(arguments, false),
        "nKWi-N2HBV4" => kernel_positional_transfer(arguments, true),
        _ => return None,
    })
}

/// Handles the asynchronous file-I/O submission and completion queries.
fn dispatch_aio(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "HgX7+AORI58" => kernel_aio_submit(arguments, false),
        "XQ8C8y+de+E" => kernel_aio_submit(arguments, true),
        "2pOuoWoCxdk" => kernel_aio_poll_request(arguments),
        "o7O4z3jwKzo" => kernel_aio_poll_requests(arguments),
        "KOF-oJbQVvc" => kernel_aio_wait_request(arguments),
        "5TgME6AYty4" => kernel_aio_delete_request(arguments),
        _ => return None,
    })
}

/// Handles the user-event registrations and the accessors on a delivered event.
fn dispatch_event(nid: &str, arguments: [u64; 6], state: &mut ProbeDispatchState) -> Option<u64> {
    Some(match nid {
        "4R6-OvI2cEA" => kernel_register_event(
            state,
            arguments[0],
            arguments[1],
            KERNEL_EVENT_FILTER_USER,
            0,
        ),
        "LJDwdSNTnDg" => kernel_delete_user_event(arguments, state),
        "23CPPI1tyBY" => kernel_event_filter(arguments[0]),
        "vz+pg2zdopI" => kernel_event_user_data(arguments[0]),
        _ => return None,
    })
}

/// Handles the thread and thread-attribute entry points.
///
/// The reader/writer-lock family is deliberately absent: its blocking acquires park, so it is
/// served by [`crate::dispatch_guest_sync_import_at_import`] before the dispatch lock is taken.
fn dispatch_thread(nid: &str, arguments: [u64; 6], state: &mut ProbeDispatchState) -> Option<u64> {
    Some(match nid {
        "3PtV6p3QNX4" => u64::from(arguments[0] == arguments[1] && arguments[0] != 0),
        "rcrVFJsQWRY" => pthread_get_affinity(arguments, state),
        "1tKyG7RlMJo" => pthread_get_priority(arguments, state),
        "How7B8Oet6k" => pthread_get_name(arguments, state),
        "JaRMy+QcpeU" => pthread_attr_get_detach_state(arguments, state),
        // POSIX spellings of calls the runtime already serves under their console names. Only the
        // reported error shape differs.
        "FIs3-UQT9sg" => posix_pthread_result(pthread_get_schedule_param(arguments, state)),
        "7ZlAakEf0Qg" => posix_pthread_result(pthread_attr_set_inherit_schedule(arguments, state)),
        "euKRgm0Vn2M" => posix_pthread_result(pthread_attr_set_schedule_param(arguments, state)),
        "9vyP6Z7bqzc" => posix_pthread_result(pthread_rename(arguments, state)),
        "2+pVfgiEd7A" => posix_pthread_result(pthread_attr_set_solo_schedule(arguments, state)),
        _ => return None,
    })
}

/// Handles the process-wide queries, the timestamp counter, and the devkit indicator lines.
fn dispatch_process(nid: &str, arguments: [u64; 6], state: &mut ProbeDispatchState) -> Option<u64> {
    Some(match nid {
        "Ou3iL1abvng" => stack_check_failed(),
        "HoLVWNanBBc" => u64::from(std::process::id()),
        "6XG4B33N09g" => {
            std::thread::yield_now();
            0
        }
        "CBNtXOoef-E" => scheduler_priority_bound(low_i32(arguments[0]), KERNEL_PRIORITY_HIGHEST),
        "m0iS6jNsXds" => scheduler_priority_bound(low_i32(arguments[0]), KERNEL_PRIORITY_LOWEST),
        "1j3S3n-tTW4" => KERNEL_TSC_FREQUENCY,
        "-2IRUCO--PM" => kernel_process_time(true),
        "g0VTBxfJyu0" => kernel_current_cpu(state),
        // Devkit indicator lines have no emulated hardware behind them: nothing observes the
        // outputs and nothing drives the inputs.
        "ca7v6Cxulzs" => {
            remu_gpu::note_inert("ca7v6Cxulzs");
            0
        }
        "4oXYe9Xmk0Q" => KERNEL_DEVKIT_INPUT_BITS,
        "tU5e3f9gSiU" => KERNEL_TRINITY_MODE,
        "IWIBBdTHit4" => kernel_map_flexible_memory(arguments, state),
        _ => return None,
    })
}

/// Handles the textual conversions of binary internet addresses.
fn dispatch_inet_address(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "5jRCs2axtr4" => inet_ntop(arguments),
        "4n51s0zEf0c" => inet_pton(arguments),
        _ => return None,
    })
}

/// Reports the caller's own thread as the frame whose stack canary was overwritten, then ends the
/// process.
///
/// The corrupted frame's return address has already been clobbered, so there is no caller left to
/// return a status to.
fn stack_check_failed() -> ! {
    eprintln!(
        "[REMu][KERNEL] guest stack canary overwritten thread={:?} guest={:#x}",
        std::thread::current().name().unwrap_or("unnamed"),
        current_guest_thread_handle()
    );
    terminate_probe(STACK_CHECK_EXIT_STATUS);
}

/// Borrows the descriptor table shared by every file entry point.
fn kernel_files() -> MutexGuard<'static, KernelFileState> {
    KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Reports whether a guest range of `length` bytes can be read in full.
///
/// A null base and an arithmetic wrap are rejected before the range is probed, so a malformed
/// argument never reaches the host memory query.
fn guest_range_is_readable(address: u64, length: usize) -> bool {
    let Ok(span) = u64::try_from(length) else {
        return false;
    };
    address != 0
        && address.checked_add(span).is_some()
        && process_guest_range_is_readable(address, length)
}

/// Stores a plain errno in the calling thread's cell for a POSIX-shaped failure.
pub(crate) fn set_guest_errno(errno: i32) {
    let address = guest_errno_address();
    if address != 0 {
        write_guest_u32(address, errno.cast_unsigned());
    }
}

/// Removes a user-event registration and any of its events still awaiting delivery.
fn kernel_delete_user_event(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (equeue, ident) = (arguments[0], arguments[1]);
    let (_h, mut events) = crate::event_state::lock_event_state_for_write();
    if !events.kernel_equeues.contains(&equeue) {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    if events
        .kernel_event_registrations
        .remove(&(equeue, KERNEL_EVENT_FILTER_USER, ident))
        .is_none()
    {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    // An event queued before the interest was withdrawn must not still arrive after it.
    crate::discard_kernel_events_into(&mut events, state, |event| {
        event.equeue == equeue && event.filter == KERNEL_EVENT_FILTER_USER && event.ident == ident
    });
    0
}

/// Reads the filter of a delivered kernel event.
fn kernel_event_filter(event: u64) -> u64 {
    if !guest_range_is_readable(event, KERNEL_EVENT_SIZE) {
        return 0;
    }
    signed_i32_return(i32::from(
        read_guest_u16(event + KERNEL_EVENT_FILTER_OFFSET).cast_signed(),
    ))
}

/// Reads the user data of a delivered kernel event.
fn kernel_event_user_data(event: u64) -> u64 {
    if !guest_range_is_readable(event, KERNEL_EVENT_SIZE) {
        return 0;
    }
    read_guest_u64(event + KERNEL_EVENT_USER_DATA_OFFSET)
}

/// Maps flexible memory into the guest address space without naming the range.
#[cfg(target_os = "macos")]
fn kernel_map_flexible_memory(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (output, length) = (arguments[0], arguments[1]);
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    if length == 0 || !length.is_multiple_of(remu_memory::GUEST_PAGE_SIZE) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    crate::map_flexible_memory_core(
        output,
        length,
        low_i32(arguments[2]),
        arguments[3],
        "anon",
        state,
    )
}

/// Maps flexible memory into the guest address space without naming the range.
#[cfg(not(target_os = "macos"))]
fn kernel_map_flexible_memory(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
}

#[cfg(test)]
mod tests;
