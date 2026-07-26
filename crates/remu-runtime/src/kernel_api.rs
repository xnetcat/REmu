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

use std::collections::BTreeMap;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileExt, PermissionsExt};
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::{
    DEFAULT_THREAD_AFFINITY_MASK, DEFAULT_THREAD_PRIORITY, KERNEL_EVENT_FILTER_USER, KERNEL_FILES,
    KernelFileState, ProbeDispatchState, SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR,
    SCE_KERNEL_ERROR_BUSY, SCE_KERNEL_ERROR_INVALID_ARGUMENT, SCE_KERNEL_ERROR_IS_DIRECTORY,
    SCE_KERNEL_ERROR_MEMORY_FAULT, SCE_KERNEL_ERROR_NOT_FOUND, SCE_KERNEL_ERROR_PERMISSION_DENIED,
    current_guest_thread_handle, guest_errno_address, guest_kernel_rmdir, guest_posix_file_result,
    guest_rwlock_destroy, guest_rwlock_init, guest_rwlock_lock, guest_rwlock_unlock, guest_sync,
    kernel_process_time, kernel_register_event, low_i32, low_u32, posix_pthread_result,
    process_guest_range_is_readable, pthread_attr_set_inherit_schedule,
    pthread_attr_set_schedule_param, pthread_attr_state, pthread_get_schedule_param,
    pthread_rename, read_guest_bytes, read_guest_cstring, read_guest_u16, read_guest_u32,
    read_guest_u64, resolve_probe_guest_path, sce_kernel_error, sce_kernel_io_error,
    signed_i32_return, terminate_probe, update_pthread_attr, write_guest_bytes, write_guest_u32,
    write_guest_u64,
};

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

/// Scheduling policy identifiers accepted by the priority-range queries.
const SCHED_FIFO: i32 = 1;
const SCHED_OTHER: i32 = 2;
const SCHED_RR: i32 = 3;

/// Most and least urgent thread priorities the console's realtime class accepts.
///
/// The scale is inverted with respect to POSIX: a smaller number is more urgent.
const KERNEL_PRIORITY_HIGHEST: i32 = 256;
const KERNEL_PRIORITY_LOWEST: i32 = 767;

/// Bytes copied for a thread name, terminator included.
///
/// The name accessor takes no capacity argument, so the ABI fixes one maximum every caller must
/// have room for. Names longer than this are truncated rather than overrunning that buffer.
const PTHREAD_NAME_CAPACITY: usize = 32;

/// Address families the textual address conversions understand.
const AF_INET: i32 = 2;
const AF_INET6: i32 = 28;
/// Bytes an address of each family occupies in its binary form.
const IPV4_ADDRESS_BYTES: usize = 4;
const IPV6_ADDRESS_BYTES: usize = 16;

/// Bytes one asynchronous read or write request occupies.
const AIO_REQUEST_SIZE: u64 = 0x28;
const AIO_REQUEST_LENGTH_OFFSET: u64 = 0x08;
const AIO_REQUEST_BUFFER_OFFSET: u64 = 0x10;
const AIO_REQUEST_RESULT_OFFSET: u64 = 0x18;
const AIO_REQUEST_FD_OFFSET: u64 = 0x20;

/// Byte offset of a request result's state word; the transferred count precedes it.
const AIO_RESULT_STATE_OFFSET: u64 = 0x08;

/// Terminal state of a request whose transfer has finished.
const AIO_STATE_COMPLETED: u32 = 3;
/// Requests one submission may carry.
const AIO_REQUEST_LIMIT: i32 = 128;
/// Lowest and highest scheduling priorities a submission may name.
const AIO_PRIORITY_LOW: i32 = 1;
const AIO_PRIORITY_HIGH: i32 = 3;

/// Outcome of one completed asynchronous submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AioSubmission {
    state: u32,
    result: i32,
}

/// Submissions the guest has not deleted yet, keyed by the identifier it was given.
#[derive(Debug)]
struct AioState {
    next_id: i32,
    submissions: BTreeMap<i32, AioSubmission>,
}

impl Default for AioState {
    fn default() -> Self {
        Self {
            next_id: 1,
            submissions: BTreeMap::new(),
        }
    }
}

static AIO_SUBMISSIONS: OnceLock<Mutex<AioState>> = OnceLock::new();

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

/// Handles the thread, thread-attribute, and reader/writer-lock entry points.
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
        // The console's reader/writer lock is the POSIX object under another name, so these share
        // the guest-synchronisation table with the entry points already exported for it.
        "6ULAa0fq4jA" => guest_rwlock_init(arguments[0]),
        "BB+kb08Tl9A" => guest_rwlock_destroy(arguments[0]),
        "XD3mDeybCnk" => guest_rwlock_lock(arguments[0], false, true),
        "bIHoZCTomsI" => guest_rwlock_lock(arguments[0], true, true),
        "+L98PIbGttk" => guest_rwlock_unlock(arguments[0]),
        "12wOHk8ywb0" => kernel_poll_semaphore(arguments),
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
        "ca7v6Cxulzs" => 0,
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
fn set_guest_errno(errno: i32) {
    let address = guest_errno_address();
    if address != 0 {
        write_guest_u32(address, errno.cast_unsigned());
    }
}

/// Resolves a guest path that a mutating call is about to act on.
///
/// Mounts are read-only as a whole, so a call that would change anything under one is refused
/// before it reaches the host filesystem.
fn resolve_writable_path(
    address: u64,
    state: &ProbeDispatchState,
) -> Result<std::path::PathBuf, u64> {
    let Some(guest_path) = read_guest_cstring(address) else {
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT));
    };
    let Some((host_path, writable)) = resolve_probe_guest_path(state, &guest_path) else {
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND));
    };
    if !writable {
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_PERMISSION_DENIED));
    }
    Ok(host_path)
}

/// Removes a directory entry that is not itself a directory.
fn kernel_unlink(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let host_path = match resolve_writable_path(arguments[0], state) {
        Ok(host_path) => host_path,
        Err(error) => return error,
    };
    let Ok(metadata) = std::fs::symlink_metadata(&host_path) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if metadata.is_dir() {
        return sce_kernel_error(SCE_KERNEL_ERROR_IS_DIRECTORY);
    }
    std::fs::remove_file(&host_path).map_or_else(|error| sce_kernel_io_error(&error), |()| 0)
}

/// Moves a file or directory to another guest path.
fn kernel_rename(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let source = match resolve_writable_path(arguments[0], state) {
        Ok(source) => source,
        Err(error) => return error,
    };
    let destination = match resolve_writable_path(arguments[1], state) {
        Ok(destination) => destination,
        Err(error) => return error,
    };
    if !source.exists() {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    std::fs::rename(&source, &destination).map_or_else(|error| sce_kernel_io_error(&error), |()| 0)
}

/// Replaces the permission bits of a guest path.
fn kernel_chmod(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let host_path = match resolve_writable_path(arguments[0], state) {
        Ok(host_path) => host_path,
        Err(error) => return error,
    };
    if !host_path.exists() {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    let mode = low_u32(arguments[1]) & KERNEL_MODE_BITS;
    std::fs::set_permissions(&host_path, PermissionsExt::from_mode(mode))
        .map_or_else(|error| sce_kernel_io_error(&error), |()| 0)
}

/// Sets the length of the file at a guest path, extending it with zeroes when it grows.
fn kernel_truncate(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let length = i64::from_ne_bytes(arguments[1].to_ne_bytes());
    if length < 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let host_path = match resolve_writable_path(arguments[0], state) {
        Ok(host_path) => host_path,
        Err(error) => return error,
    };
    match std::fs::OpenOptions::new().write(true).open(&host_path) {
        Ok(file) => file
            .set_len(length.cast_unsigned())
            .map_or_else(|error| sce_kernel_io_error(&error), |()| 0),
        Err(error) => sce_kernel_io_error(&error),
    }
}

/// Reads one guest `timeval` into the host's own representation of the same instant.
///
/// Both fields are 64 bits wide in the guest ABI while the host narrows the fractional one, so
/// the pair is decoded field by field instead of copied.
fn read_guest_timeval(address: u64) -> Option<libc::timeval> {
    let seconds = i64::from_ne_bytes(read_guest_u64(address).to_ne_bytes());
    let microseconds =
        i64::from_ne_bytes(read_guest_u64(address + TIMEVAL_MICROSECONDS_OFFSET).to_ne_bytes());
    if !(0..MICROSECONDS_PER_SECOND).contains(&microseconds) {
        return None;
    }
    Some(libc::timeval {
        tv_sec: libc::time_t::try_from(seconds).ok()?,
        tv_usec: libc::suseconds_t::try_from(microseconds).ok()?,
    })
}

/// Sets the access and modification times of a guest path.
///
/// A null time array asks for the current time on both.
fn kernel_utimes(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let host_path = match resolve_writable_path(arguments[0], state) {
        Ok(host_path) => host_path,
        Err(error) => return error,
    };
    let Ok(host_path) = std::ffi::CString::new(host_path.as_os_str().as_bytes()) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let times = arguments[1];
    let result = if times == 0 {
        // SAFETY: the path is a live NUL-terminated host string and a null time array is the
        // documented request for the current time.
        unsafe { libc::utimes(host_path.as_ptr(), std::ptr::null()) }
    } else {
        let length = usize::try_from(TIMEVAL_SIZE * 2).expect("two timevals fit usize");
        if !guest_range_is_readable(times, length) {
            return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
        }
        let (Some(accessed), Some(modified)) = (
            read_guest_timeval(times),
            read_guest_timeval(times + TIMEVAL_SIZE),
        ) else {
            return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
        };
        let values = [accessed, modified];
        // SAFETY: the path is a live NUL-terminated host string and `values` is a complete
        // two-element host time array owned by this frame.
        unsafe { libc::utimes(host_path.as_ptr(), values.as_ptr()) }
    };
    if result == 0 {
        0
    } else {
        sce_kernel_io_error(&std::io::Error::last_os_error())
    }
}

/// Reports whether a guest path names something the filesystem layer can reach.
fn kernel_check_reachability(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let Some(guest_path) = read_guest_cstring(arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    let Some((host_path, _)) = resolve_probe_guest_path(state, &guest_path) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if host_path.exists() {
        0
    } else {
        sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
    }
}

/// Flushes an open descriptor's contents and metadata to the host filesystem.
fn kernel_fsync(arguments: [u64; 6]) -> u64 {
    let fd = low_i32(arguments[0]);
    let files = kernel_files();
    let Some(file) = files.files.get(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    };
    file.sync_all()
        .map_or_else(|error| sce_kernel_io_error(&error), |()| 0)
}

/// Transfers bytes at an explicit file offset without disturbing the descriptor's shared cursor.
fn kernel_positional_transfer(arguments: [u64; 6], write: bool) -> u64 {
    let fd = low_i32(arguments[0]);
    let buffer = arguments[1];
    let offset = i64::from_ne_bytes(arguments[3].to_ne_bytes());
    if offset < 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Ok(length) = usize::try_from(arguments[2]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if length != 0 && buffer == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let files = kernel_files();
    let Some(file) = files.files.get(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    };
    if length == 0 {
        return 0;
    }
    let transferred = if write {
        // SAFETY: the positional write ABI supplies a readable guest range of exactly `length`
        // bytes, and the descriptor table lock serialises access to the file.
        let source = unsafe { std::slice::from_raw_parts(buffer as *const u8, length) };
        file.write_at(source, offset.cast_unsigned())
    } else {
        // SAFETY: the positional read ABI supplies a writable guest range of exactly `length`
        // bytes, and the descriptor table lock serialises access to the file.
        let destination = unsafe { std::slice::from_raw_parts_mut(buffer as *mut u8, length) };
        file.read_at(destination, offset.cast_unsigned())
    };
    match transferred {
        Ok(transferred) => u64::try_from(transferred).expect("transferred byte count fits u64"),
        Err(error) => sce_kernel_io_error(&error),
    }
}

/// Performs one batch of positional transfers and records its terminal state.
///
/// The transfers run before the call returns, so a submission is already complete by the time the
/// guest first asks about it. That is a legal schedule for the ABI: every later query reports the
/// terminal state, and the guest's buffers hold their data as soon as it can observe them.
fn kernel_aio_submit(arguments: [u64; 6], write: bool) -> u64 {
    let (requests, identifier) = (arguments[0], arguments[3]);
    let count = low_i32(arguments[1]);
    let priority = low_i32(arguments[2]);
    if requests == 0 || identifier == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    if count <= 0 || count > AIO_REQUEST_LIMIT {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if !(AIO_PRIORITY_LOW..=AIO_PRIORITY_HIGH).contains(&priority) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let count = u64::from(count.cast_unsigned());
    let Ok(span) = usize::try_from(count * AIO_REQUEST_SIZE) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if !guest_range_is_readable(requests, span) {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    // A batch naming an unusable descriptor is rejected whole, so no request in it is half applied.
    let files = kernel_files();
    for index in 0..count {
        let request = requests + index * AIO_REQUEST_SIZE;
        if !files
            .files
            .contains_key(&read_guest_u32(request + AIO_REQUEST_FD_OFFSET).cast_signed())
        {
            return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
        }
    }
    drop(files);
    let mut outcome = 0;
    for index in 0..count {
        let request = requests + index * AIO_REQUEST_SIZE;
        let transfer = [
            u64::from(read_guest_u32(request + AIO_REQUEST_FD_OFFSET)),
            read_guest_u64(request + AIO_REQUEST_BUFFER_OFFSET),
            read_guest_u64(request + AIO_REQUEST_LENGTH_OFFSET),
            read_guest_u64(request),
            0,
            0,
        ];
        let transferred = kernel_positional_transfer(transfer, write);
        let result = read_guest_u64(request + AIO_REQUEST_RESULT_OFFSET);
        if result != 0 {
            write_guest_u64(result, transferred);
            write_guest_u32(result + AIO_RESULT_STATE_OFFSET, AIO_STATE_COMPLETED);
        }
        if low_u32(transferred) & 0x8000_0000 != 0 {
            outcome = low_i32(transferred);
        }
    }
    let mut aio = aio_submissions();
    let id = aio.next_id;
    aio.next_id = aio.next_id.checked_add(1).unwrap_or(1);
    aio.submissions.insert(
        id,
        AioSubmission {
            state: AIO_STATE_COMPLETED,
            result: outcome,
        },
    );
    drop(aio);
    write_guest_u32(identifier, id.cast_unsigned());
    0
}

/// Borrows the table of submissions the guest has not deleted yet.
fn aio_submissions() -> MutexGuard<'static, AioState> {
    AIO_SUBMISSIONS
        .get_or_init(|| Mutex::new(AioState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Reports the state of one submission without waiting.
fn kernel_aio_poll_request(arguments: [u64; 6]) -> u64 {
    let output = arguments[1];
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let aio = aio_submissions();
    let Some(submission) = aio.submissions.get(&low_i32(arguments[0])) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let state = submission.state;
    drop(aio);
    write_guest_u32(output, state);
    0
}

/// Reports the state of several submissions without waiting.
fn kernel_aio_poll_requests(arguments: [u64; 6]) -> u64 {
    let (identifiers, states) = (arguments[0], arguments[2]);
    let count = low_i32(arguments[1]);
    if identifiers == 0 || states == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    if count <= 0 || count > AIO_REQUEST_LIMIT {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let count = u64::from(count.cast_unsigned());
    let Ok(span) = usize::try_from(count * 4) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if !guest_range_is_readable(identifiers, span) {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let aio = aio_submissions();
    let mut resolved = Vec::with_capacity(span / 4);
    for index in 0..count {
        let id = read_guest_u32(identifiers + index * 4).cast_signed();
        let Some(submission) = aio.submissions.get(&id) else {
            return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
        };
        resolved.push(submission.state);
    }
    drop(aio);
    for (index, state) in resolved.into_iter().enumerate() {
        let offset = u64::try_from(index * 4).expect("the request index fits u64");
        write_guest_u32(states + offset, state);
    }
    0
}

/// Reports the terminal state of one submission.
///
/// Every submission has already finished when it is created, so no time is consumed and the
/// caller's remaining-timeout cell is left as it was.
fn kernel_aio_wait_request(arguments: [u64; 6]) -> u64 {
    kernel_aio_poll_request(arguments)
}

/// Discards the bookkeeping for one submission and reports its outcome.
fn kernel_aio_delete_request(arguments: [u64; 6]) -> u64 {
    let mut aio = aio_submissions();
    let Some(submission) = aio.submissions.remove(&low_i32(arguments[0])) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    drop(aio);
    if arguments[1] != 0 {
        write_guest_u32(arguments[1], submission.result.cast_unsigned());
    }
    0
}

/// Removes a user-event registration and any of its events still awaiting delivery.
fn kernel_delete_user_event(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (equeue, ident) = (arguments[0], arguments[1]);
    if !state.kernel_equeues.contains(&equeue) {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    if state
        .kernel_event_registrations
        .remove(&(equeue, KERNEL_EVENT_FILTER_USER, ident))
        .is_none()
    {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    // An event queued before the interest was withdrawn must not still arrive after it.
    state.pending_events.retain(|event| {
        event.equeue != equeue || event.filter != KERNEL_EVENT_FILTER_USER || event.ident != ident
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

/// Reports the CPU affinity mask of a running guest thread.
fn pthread_get_affinity(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
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
fn pthread_get_priority(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
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
fn pthread_get_name(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
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
fn pthread_attr_get_detach_state(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
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
fn pthread_attr_set_solo_schedule(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    update_pthread_attr(arguments[0], state, |_| {})
}

/// Takes a count from a semaphore when one is already available, and never waits.
fn kernel_poll_semaphore(arguments: [u64; 6]) -> u64 {
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
fn scheduler_priority_bound(policy: i32, bound: i32) -> u64 {
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
fn kernel_current_cpu(state: &ProbeDispatchState) -> u64 {
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

/// Formats a binary internet address into the caller's text buffer.
///
/// Returns the buffer it wrote on success and a null pointer otherwise.
fn inet_ntop(arguments: [u64; 6]) -> u64 {
    let (source, destination) = (arguments[1], arguments[2]);
    let Ok(capacity) = usize::try_from(low_u32(arguments[3])) else {
        set_guest_errno(libc::ENOSPC);
        return 0;
    };
    let family = low_i32(arguments[0]);
    if source == 0 || destination == 0 {
        set_guest_errno(libc::EFAULT);
        return 0;
    }
    let text = match family {
        AF_INET => {
            if !guest_range_is_readable(source, IPV4_ADDRESS_BYTES) {
                set_guest_errno(libc::EFAULT);
                return 0;
            }
            let mut octets = [0_u8; IPV4_ADDRESS_BYTES];
            read_guest_bytes(source, &mut octets);
            Ipv4Addr::from(octets).to_string()
        }
        AF_INET6 => {
            if !guest_range_is_readable(source, IPV6_ADDRESS_BYTES) {
                set_guest_errno(libc::EFAULT);
                return 0;
            }
            let mut octets = [0_u8; IPV6_ADDRESS_BYTES];
            read_guest_bytes(source, &mut octets);
            Ipv6Addr::from(octets).to_string()
        }
        _ => {
            set_guest_errno(libc::EAFNOSUPPORT);
            return 0;
        }
    };
    if text.len() + 1 > capacity {
        set_guest_errno(libc::ENOSPC);
        return 0;
    }
    let mut bytes = text.into_bytes();
    bytes.push(0);
    write_guest_bytes(destination, &bytes);
    destination
}

/// Parses a textual internet address into its binary form.
///
/// Reports one for an address it accepted, zero for text that is not an address of that family,
/// and minus one for a family it does not know.
fn inet_pton(arguments: [u64; 6]) -> u64 {
    let (source, destination) = (arguments[1], arguments[2]);
    let family = low_i32(arguments[0]);
    if !matches!(family, AF_INET | AF_INET6) {
        set_guest_errno(libc::EAFNOSUPPORT);
        return signed_i32_return(-1);
    }
    if source == 0 || destination == 0 {
        set_guest_errno(libc::EFAULT);
        return signed_i32_return(-1);
    }
    let Some(text) = read_guest_cstring(source) else {
        set_guest_errno(libc::EFAULT);
        return signed_i32_return(-1);
    };
    if family == AF_INET {
        let Ok(address) = text.parse::<Ipv4Addr>() else {
            return 0;
        };
        write_guest_bytes(destination, &address.octets());
    } else {
        let Ok(address) = text.parse::<Ipv6Addr>() else {
            return 0;
        };
        write_guest_bytes(destination, &address.octets());
    }
    1
}

#[cfg(test)]
mod tests {
    use std::io::Seek;

    use super::*;
    use crate::{
        GuestSemaphoreState, KernelQueuedEvent, PthreadAttrState, PthreadThreadObject,
        SaveDataMount,
    };

    const SUCCESS: u64 = 0;
    const KERNEL_NOT_FOUND: u64 = 0xffff_ffff_8002_0002;
    const KERNEL_BAD_FILE_DESCRIPTOR: u64 = 0xffff_ffff_8002_0009;
    const KERNEL_PERMISSION_DENIED: u64 = 0xffff_ffff_8002_000d;
    const KERNEL_MEMORY_FAULT: u64 = 0xffff_ffff_8002_000e;
    const KERNEL_BUSY: u64 = 0xffff_ffff_8002_0010;
    const KERNEL_IS_DIRECTORY: u64 = 0xffff_ffff_8002_0015;
    const KERNEL_INVALID_ARGUMENT: u64 = 0xffff_ffff_8002_0016;
    const POSIX_FAILURE: u64 = u64::MAX;

    /// A directory removed when the test that made it ends.
    struct TemporaryDirectory {
        path: std::path::PathBuf,
    }

    impl TemporaryDirectory {
        fn new(label: &str) -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
            let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "remu-kernel-api-{label}-{}-{id}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("the test directory can be created");
            Self { path }
        }
    }

    impl Drop for TemporaryDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// A writable guest mount backed by a temporary host directory.
    struct MountFixture {
        directory: TemporaryDirectory,
        state: ProbeDispatchState,
    }

    impl MountFixture {
        fn new(label: &str) -> Self {
            Self::with_read_only(label, false)
        }

        fn with_read_only(label: &str, read_only: bool) -> Self {
            let directory = TemporaryDirectory::new(label);
            let mut state = ProbeDispatchState::default();
            state.save_mounts.insert(
                "/savedata0".to_owned(),
                SaveDataMount {
                    path: directory.path.clone(),
                    committed_path: directory.path.clone(),
                    dir_name: "test".to_owned(),
                    user_id: 1,
                    resource: 1,
                    read_only,
                },
            );
            Self { directory, state }
        }

        fn write(&self, name: &str, contents: &[u8]) -> std::path::PathBuf {
            let path = self.directory.path.join(name);
            std::fs::write(&path, contents).expect("the test file can be written");
            path
        }

        fn call(&mut self, nid: &str, arguments: [u64; 6]) -> u64 {
            dispatch(nid, arguments, &mut self.state).expect("the test names an owned NID")
        }
    }

    fn args(values: &[u64]) -> [u64; 6] {
        let mut arguments = [0_u64; 6];
        arguments[..values.len()].copy_from_slice(values);
        arguments
    }

    fn guest_string(text: &str) -> Vec<u8> {
        let mut bytes = text.as_bytes().to_vec();
        bytes.push(0);
        bytes
    }

    fn address(bytes: &[u8]) -> u64 {
        bytes.as_ptr().addr() as u64
    }

    /// Registers an open host file in the shared descriptor table and reports its guest fd.
    fn register_descriptor(file: std::fs::File) -> i32 {
        let mut files = kernel_files();
        let fd = files.next_fd;
        files.next_fd = fd.wrapping_add(1);
        files.files.insert(fd, file);
        fd
    }

    fn release_descriptor(fd: i32) {
        kernel_files().files.remove(&fd);
    }

    #[test]
    fn unlinking_removes_a_file_and_refuses_a_directory_or_a_missing_name() {
        let mut fixture = MountFixture::new("unlink");
        fixture.write("save.bin", b"payload");
        std::fs::create_dir(fixture.directory.path.join("folder")).expect("the directory is made");
        let file = guest_string("/savedata0/save.bin");
        let folder = guest_string("/savedata0/folder");
        let missing = guest_string("/savedata0/absent.bin");

        assert_eq!(
            fixture.call("AUXVxWeJU-A", args(&[address(&file)])),
            SUCCESS
        );
        assert!(!fixture.directory.path.join("save.bin").exists());
        assert_eq!(
            fixture.call("AUXVxWeJU-A", args(&[address(&folder)])),
            KERNEL_IS_DIRECTORY
        );
        assert_eq!(
            fixture.call("AUXVxWeJU-A", args(&[address(&missing)])),
            KERNEL_NOT_FOUND
        );
        assert_eq!(fixture.call("AUXVxWeJU-A", [0; 6]), KERNEL_MEMORY_FAULT);
    }

    #[test]
    fn the_posix_spelling_of_a_path_call_reports_minus_one_rather_than_a_kernel_code() {
        let mut fixture = MountFixture::new("posix-path");
        fixture.write("save.bin", b"payload");
        let file = guest_string("/savedata0/save.bin");
        let missing = guest_string("/savedata0/absent.bin");

        assert_eq!(
            fixture.call("VAzswvTOCzI", args(&[address(&missing)])),
            POSIX_FAILURE
        );
        assert_eq!(
            fixture.call("VAzswvTOCzI", args(&[address(&file)])),
            SUCCESS
        );
        assert!(!fixture.directory.path.join("save.bin").exists());
    }

    #[test]
    fn a_read_only_mount_refuses_every_call_that_would_change_it() {
        let mut fixture = MountFixture::with_read_only("read-only", true);
        fixture.write("save.bin", b"payload");
        let file = guest_string("/savedata0/save.bin");
        let renamed = guest_string("/savedata0/other.bin");
        let times = [0_u64; 4];

        for nid in ["AUXVxWeJU-A", "fgIsQ10xYVA", "WlyEA-sLDf0"] {
            assert_eq!(
                fixture.call(nid, args(&[address(&file), 0])),
                KERNEL_PERMISSION_DENIED,
                "{nid} accepted a read-only mount"
            );
        }
        assert_eq!(
            fixture.call("52NcYU9+lEo", args(&[address(&file), address(&renamed)])),
            KERNEL_PERMISSION_DENIED
        );
        assert_eq!(
            fixture.call(
                "0Cq8ipKr9n0",
                args(&[address(&file), times.as_ptr().addr() as u64])
            ),
            KERNEL_PERMISSION_DENIED
        );
        // Reading the mount is still allowed.
        assert_eq!(
            fixture.call("uWyW3v98sU4", args(&[address(&file)])),
            SUCCESS
        );
        assert!(fixture.directory.path.join("save.bin").exists());
    }

    #[test]
    fn renaming_moves_a_file_and_reports_a_missing_source() {
        let mut fixture = MountFixture::new("rename");
        fixture.write("from.bin", b"payload");
        let from = guest_string("/savedata0/from.bin");
        let to = guest_string("/savedata0/to.bin");
        let missing = guest_string("/savedata0/absent.bin");

        assert_eq!(
            fixture.call("52NcYU9+lEo", args(&[address(&from), address(&to)])),
            SUCCESS
        );
        assert_eq!(
            std::fs::read(fixture.directory.path.join("to.bin")).expect("the moved file is there"),
            b"payload"
        );
        assert_eq!(
            fixture.call("52NcYU9+lEo", args(&[address(&missing), address(&to)])),
            KERNEL_NOT_FOUND
        );
        assert_eq!(
            fixture.call("NN01qLRhiqU", args(&[address(&missing), address(&to)])),
            POSIX_FAILURE
        );
    }

    #[test]
    fn removing_a_directory_through_the_posix_name_reports_minus_one_when_it_is_absent() {
        let mut fixture = MountFixture::new("rmdir");
        std::fs::create_dir(fixture.directory.path.join("folder")).expect("the directory is made");
        let folder = guest_string("/savedata0/folder");
        let missing = guest_string("/savedata0/absent");

        assert_eq!(
            fixture.call("c7ZnT7V1B98", args(&[address(&missing)])),
            POSIX_FAILURE
        );
        assert_eq!(
            fixture.call("c7ZnT7V1B98", args(&[address(&folder)])),
            SUCCESS
        );
        assert!(!fixture.directory.path.join("folder").exists());
    }

    #[test]
    fn changing_permissions_keeps_only_the_mode_bits_the_guest_may_set() {
        let mut fixture = MountFixture::new("chmod");
        let path = fixture.write("save.bin", b"payload");
        let file = guest_string("/savedata0/save.bin");

        // The high bits above the permission field belong to the file type and must not survive.
        let mode = 0xffff_0000 | u64::from(0o640_u32);
        assert_eq!(
            fixture.call("fgIsQ10xYVA", args(&[address(&file), mode])),
            SUCCESS
        );
        let mode = std::fs::metadata(&path)
            .expect("the file is still there")
            .permissions()
            .mode()
            & KERNEL_MODE_BITS;
        assert_eq!(mode, 0o640);
    }

    #[test]
    fn truncating_a_path_sets_its_length_and_rejects_a_negative_one() {
        let mut fixture = MountFixture::new("truncate");
        let path = fixture.write("save.bin", b"0123456789");
        let file = guest_string("/savedata0/save.bin");

        assert_eq!(
            fixture.call("WlyEA-sLDf0", args(&[address(&file), 4])),
            SUCCESS
        );
        assert_eq!(
            std::fs::read(&path).expect("the file is still there"),
            b"0123"
        );
        assert_eq!(
            fixture.call("WlyEA-sLDf0", args(&[address(&file), u64::MAX])),
            KERNEL_INVALID_ARGUMENT
        );
    }

    #[test]
    fn setting_times_decodes_both_eight_byte_fields_of_each_entry() {
        let mut fixture = MountFixture::new("utimes");
        let path = fixture.write("save.bin", b"payload");
        let file = guest_string("/savedata0/save.bin");
        // Access seconds and microseconds, then modification seconds and microseconds.
        let times: [u64; 4] = [1_000_000, 250_000, 2_000_000, 500_000];

        assert_eq!(
            fixture.call(
                "0Cq8ipKr9n0",
                args(&[address(&file), times.as_ptr().addr() as u64])
            ),
            SUCCESS
        );
        let modified = std::fs::metadata(&path)
            .expect("the file is still there")
            .modified()
            .expect("the host reports a modification time")
            .duration_since(std::time::UNIX_EPOCH)
            .expect("the modification time is after the epoch");
        assert_eq!(modified.as_secs(), 2_000_000);
        assert_eq!(modified.subsec_micros(), 500_000);
    }

    #[test]
    fn a_fractional_second_outside_its_range_is_rejected_before_the_host_is_asked() {
        let mut fixture = MountFixture::new("utimes-range");
        fixture.write("save.bin", b"payload");
        let file = guest_string("/savedata0/save.bin");
        let times: [u64; 4] = [1, 1_000_000, 2, 0];

        assert_eq!(
            fixture.call(
                "0Cq8ipKr9n0",
                args(&[address(&file), times.as_ptr().addr() as u64])
            ),
            KERNEL_INVALID_ARGUMENT
        );
    }

    #[test]
    fn reachability_separates_a_present_path_from_an_absent_one() {
        let mut fixture = MountFixture::new("reachability");
        fixture.write("save.bin", b"payload");
        let file = guest_string("/savedata0/save.bin");
        let missing = guest_string("/savedata0/absent.bin");

        assert_eq!(
            fixture.call("uWyW3v98sU4", args(&[address(&file)])),
            SUCCESS
        );
        assert_eq!(
            fixture.call("uWyW3v98sU4", args(&[address(&missing)])),
            KERNEL_NOT_FOUND
        );
        assert_eq!(fixture.call("uWyW3v98sU4", [0; 6]), KERNEL_MEMORY_FAULT);
    }

    #[test]
    fn positional_transfers_leave_the_descriptor_cursor_where_they_found_it() {
        let directory = TemporaryDirectory::new("positional");
        let path = directory.path.join("data.bin");
        std::fs::write(&path, b"0123456789").expect("the test file can be written");
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("the test file can be opened");
        let fd = register_descriptor(file);
        let mut state = ProbeDispatchState::default();
        let mut buffer = [0_u8; 4];

        let fd_argument = u64::from(fd.cast_unsigned());
        let read = dispatch(
            "+r3rMFwItV4",
            args(&[fd_argument, buffer.as_mut_ptr().addr() as u64, 4, 6]),
            &mut state,
        );
        assert_eq!(read, Some(4));
        assert_eq!(&buffer, b"6789");

        let written = guest_string("AB");
        let wrote = dispatch(
            "nKWi-N2HBV4",
            args(&[fd_argument, address(&written), 2, 0]),
            &mut state,
        );
        assert_eq!(wrote, Some(2));
        assert_eq!(
            std::fs::read(&path).expect("the file is still there"),
            b"AB23456789"
        );

        // A positional transfer never moves the shared cursor, so the descriptor still reads from
        // the beginning of the file.
        let position = kernel_files()
            .files
            .get_mut(&fd)
            .expect("the descriptor is open")
            .stream_position()
            .expect("the host reports the descriptor cursor");
        assert_eq!(position, 0);
        release_descriptor(fd);
    }

    #[test]
    fn a_positional_transfer_rejects_an_unopened_descriptor_and_a_null_buffer() {
        let mut state = ProbeDispatchState::default();

        assert_eq!(
            dispatch("+r3rMFwItV4", args(&[9_999, 0x1000, 4, 0]), &mut state),
            Some(KERNEL_BAD_FILE_DESCRIPTOR)
        );
        assert_eq!(
            dispatch("nKWi-N2HBV4", args(&[9_999, 0, 4, 0]), &mut state),
            Some(KERNEL_MEMORY_FAULT)
        );
        assert_eq!(
            dispatch(
                "+r3rMFwItV4",
                args(&[9_999, 0x1000, 4, u64::MAX]),
                &mut state
            ),
            Some(KERNEL_INVALID_ARGUMENT)
        );
        assert_eq!(
            dispatch("fTx66l5iWIA", args(&[9_999]), &mut state),
            Some(KERNEL_BAD_FILE_DESCRIPTOR)
        );
    }

    #[test]
    fn flushing_an_open_descriptor_commits_its_contents() {
        let directory = TemporaryDirectory::new("fsync");
        let path = directory.path.join("data.bin");
        let file = std::fs::File::create(&path).expect("the test file can be created");
        let fd = register_descriptor(file);
        let mut state = ProbeDispatchState::default();

        assert_eq!(
            dispatch(
                "fTx66l5iWIA",
                args(&[u64::from(fd.cast_unsigned())]),
                &mut state
            ),
            Some(SUCCESS)
        );
        release_descriptor(fd);
    }

    #[test]
    fn a_user_event_registration_can_be_added_and_withdrawn_again() {
        let mut state = ProbeDispatchState::default();
        state.kernel_equeues.insert(7);

        assert_eq!(
            dispatch("4R6-OvI2cEA", args(&[7, 42]), &mut state),
            Some(SUCCESS)
        );
        assert!(
            state
                .kernel_event_registrations
                .contains_key(&(7, KERNEL_EVENT_FILTER_USER, 42))
        );
        state.pending_events.push_back(KernelQueuedEvent {
            equeue: 7,
            ident: 42,
            filter: KERNEL_EVENT_FILTER_USER,
            flags: 0,
            fflags: 1,
            data: 0,
            user_data: 0,
        });

        assert_eq!(
            dispatch("LJDwdSNTnDg", args(&[7, 42]), &mut state),
            Some(SUCCESS)
        );
        assert!(state.kernel_event_registrations.is_empty());
        assert!(state.pending_events.is_empty());
        assert_eq!(
            dispatch("LJDwdSNTnDg", args(&[7, 42]), &mut state),
            Some(KERNEL_NOT_FOUND)
        );
        assert_eq!(
            dispatch("4R6-OvI2cEA", args(&[8, 42]), &mut state),
            Some(KERNEL_NOT_FOUND)
        );
    }

    #[test]
    fn the_event_accessors_read_the_fields_at_their_wire_offsets() {
        let mut event = [0_u8; KERNEL_EVENT_SIZE];
        event[0x00..0x08].copy_from_slice(&0x1234_u64.to_le_bytes());
        // A filter is signed and every console filter is negative.
        event[0x08..0x0a].copy_from_slice(&(-11_i16).to_le_bytes());
        event[0x18..0x20].copy_from_slice(&0xdead_beef_u64.to_le_bytes());
        let mut state = ProbeDispatchState::default();
        let event_address = address(&event);

        assert_eq!(
            dispatch("23CPPI1tyBY", args(&[event_address]), &mut state),
            Some(signed_i32_return(-11))
        );
        assert_eq!(
            dispatch("vz+pg2zdopI", args(&[event_address]), &mut state),
            Some(0xdead_beef)
        );
        assert_eq!(dispatch("23CPPI1tyBY", [0; 6], &mut state), Some(0));
        assert_eq!(dispatch("vz+pg2zdopI", [0; 6], &mut state), Some(0));
    }

    #[test]
    fn thread_accessors_report_what_the_matching_setters_recorded() {
        let mut state = ProbeDispatchState::default();
        state.thread_attributes.insert(
            0x2000,
            PthreadAttrState {
                affinity_mask: 0x18,
                schedule_priority: 300,
                ..PthreadAttrState::default()
            },
        );
        state.pthread_threads.insert(
            0x2000,
            PthreadThreadObject {
                handle: 0x3000,
                unique_id: 1,
                name: "AudioMixer".to_owned(),
            },
        );
        let mut mask = 0_u64;
        let mut priority = 0_u32;
        let mut name = [0_u8; PTHREAD_NAME_CAPACITY];

        assert_eq!(
            dispatch(
                "rcrVFJsQWRY",
                args(&[0x2000, std::ptr::addr_of_mut!(mask).addr() as u64]),
                &mut state
            ),
            Some(SUCCESS)
        );
        assert_eq!(mask, 0x18);
        assert_eq!(
            dispatch(
                "1tKyG7RlMJo",
                args(&[0x2000, std::ptr::addr_of_mut!(priority).addr() as u64]),
                &mut state
            ),
            Some(SUCCESS)
        );
        assert_eq!(priority, 300);
        assert_eq!(
            dispatch(
                "How7B8Oet6k",
                args(&[0x2000, name.as_mut_ptr().addr() as u64]),
                &mut state
            ),
            Some(SUCCESS)
        );
        assert_eq!(&name[..11], b"AudioMixer\0");
    }

    #[test]
    fn thread_accessors_reject_a_null_handle_or_a_null_result() {
        let mut state = ProbeDispatchState::default();

        for nid in ["rcrVFJsQWRY", "1tKyG7RlMJo", "How7B8Oet6k"] {
            assert_eq!(
                dispatch(nid, args(&[0, 0x1000]), &mut state),
                Some(KERNEL_INVALID_ARGUMENT),
                "{nid} accepted a null thread"
            );
            assert_eq!(
                dispatch(nid, args(&[0x2000, 0]), &mut state),
                Some(KERNEL_INVALID_ARGUMENT),
                "{nid} accepted a null result"
            );
        }
        assert_eq!(
            dispatch("How7B8Oet6k", args(&[0x2000, 0x1000]), &mut state),
            Some(KERNEL_NOT_FOUND)
        );
    }

    #[test]
    fn a_thread_the_runtime_never_tracked_reports_the_defaults_it_would_have_been_created_with() {
        let mut state = ProbeDispatchState::default();
        let mut mask = 0_u64;
        let mut priority = 0_u32;

        assert_eq!(
            dispatch(
                "rcrVFJsQWRY",
                args(&[0x4000, std::ptr::addr_of_mut!(mask).addr() as u64]),
                &mut state
            ),
            Some(SUCCESS)
        );
        assert_eq!(mask, DEFAULT_THREAD_AFFINITY_MASK);
        assert_eq!(
            dispatch(
                "1tKyG7RlMJo",
                args(&[0x4000, std::ptr::addr_of_mut!(priority).addr() as u64]),
                &mut state
            ),
            Some(SUCCESS)
        );
        assert_eq!(priority, DEFAULT_THREAD_PRIORITY.cast_unsigned());
    }

    #[test]
    fn comparing_thread_handles_matches_only_the_same_live_thread() {
        let mut state = ProbeDispatchState::default();

        assert_eq!(
            dispatch("3PtV6p3QNX4", args(&[0x2000, 0x2000]), &mut state),
            Some(1)
        );
        assert_eq!(
            dispatch("3PtV6p3QNX4", args(&[0x2000, 0x3000]), &mut state),
            Some(0)
        );
        assert_eq!(dispatch("3PtV6p3QNX4", [0; 6], &mut state), Some(0));
    }

    #[test]
    fn an_attribute_template_the_runtime_never_created_is_refused_by_every_accessor() {
        let mut state = ProbeDispatchState::default();
        let mut detach_state = 0_u32;

        assert_eq!(
            dispatch(
                "JaRMy+QcpeU",
                args(&[0x5000, std::ptr::addr_of_mut!(detach_state).addr() as u64]),
                &mut state
            ),
            Some(KERNEL_INVALID_ARGUMENT)
        );
        assert_eq!(
            dispatch("JaRMy+QcpeU", args(&[0x5000, 0]), &mut state),
            Some(KERNEL_INVALID_ARGUMENT)
        );
        // The POSIX spellings report the same rejection as a small errno instead.
        assert_eq!(
            dispatch("2+pVfgiEd7A", args(&[0x5000, 1]), &mut state),
            Some(u64::from(libc::EINVAL.cast_unsigned()))
        );
        assert_eq!(
            dispatch("FIs3-UQT9sg", args(&[0, 0, 0]), &mut state),
            Some(u64::from(libc::EINVAL.cast_unsigned()))
        );
    }

    #[test]
    fn a_reader_writer_lock_is_tracked_under_its_console_names_too() {
        let lock = Box::new([0_u64; 8]);
        let address = lock.as_ptr().addr() as u64;
        let mut state = ProbeDispatchState::default();

        assert_eq!(
            dispatch("6ULAa0fq4jA", args(&[address, 0, 0]), &mut state),
            Some(SUCCESS)
        );
        assert_eq!(
            dispatch("XD3mDeybCnk", args(&[address]), &mut state),
            Some(SUCCESS)
        );
        // A reader holds the lock, so an exclusive attempt cannot take it.
        assert_eq!(
            dispatch("bIHoZCTomsI", args(&[address]), &mut state),
            Some(u64::from(libc::EBUSY.cast_unsigned()))
        );
        assert_eq!(
            dispatch("+L98PIbGttk", args(&[address]), &mut state),
            Some(SUCCESS)
        );
        assert_eq!(
            dispatch("bIHoZCTomsI", args(&[address]), &mut state),
            Some(SUCCESS)
        );
        assert_eq!(
            dispatch("+L98PIbGttk", args(&[address]), &mut state),
            Some(SUCCESS)
        );
        assert_eq!(
            dispatch("BB+kb08Tl9A", args(&[address]), &mut state),
            Some(SUCCESS)
        );
    }

    #[test]
    fn polling_a_semaphore_takes_a_count_only_when_one_is_already_available() {
        let handle = {
            let sync = guest_sync();
            let mut sync_state = sync
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            sync_state.next_semaphore_handle =
                sync_state.next_semaphore_handle.wrapping_add(1).max(1);
            let handle = sync_state.next_semaphore_handle;
            sync_state.semaphores.insert(
                handle,
                GuestSemaphoreState {
                    count: 2,
                    maximum: 4,
                },
            );
            handle
        };
        let mut state = ProbeDispatchState::default();
        let handle_argument = u64::from(handle);

        assert_eq!(
            dispatch("12wOHk8ywb0", args(&[handle_argument, 2]), &mut state),
            Some(SUCCESS)
        );
        assert_eq!(
            dispatch("12wOHk8ywb0", args(&[handle_argument, 1]), &mut state),
            Some(KERNEL_BUSY)
        );
        assert_eq!(
            dispatch("12wOHk8ywb0", args(&[handle_argument, 0]), &mut state),
            Some(KERNEL_INVALID_ARGUMENT)
        );
        assert_eq!(
            dispatch("12wOHk8ywb0", args(&[0, 1]), &mut state),
            Some(KERNEL_NOT_FOUND)
        );

        guest_sync()
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .semaphores
            .remove(&handle);
    }

    #[test]
    fn the_timestamp_counter_advances_at_the_frequency_it_reports() {
        let mut state = ProbeDispatchState::default();

        assert_eq!(
            dispatch("1j3S3n-tTW4", [0; 6], &mut state),
            Some(KERNEL_TSC_FREQUENCY)
        );
        let first = dispatch("-2IRUCO--PM", [0; 6], &mut state).expect("the counter is owned");
        std::thread::sleep(std::time::Duration::from_millis(2));
        let second = dispatch("-2IRUCO--PM", [0; 6], &mut state).expect("the counter is owned");
        assert!(second > first);
        // Two milliseconds of real time cannot read back as a whole second at this scale.
        assert!(second - first < KERNEL_TSC_FREQUENCY);
    }

    #[test]
    fn the_priority_range_covers_every_policy_the_console_defines() {
        let mut state = ProbeDispatchState::default();

        for policy in [SCHED_FIFO, SCHED_OTHER, SCHED_RR] {
            let policy = u64::from(policy.cast_unsigned());
            assert_eq!(
                dispatch("CBNtXOoef-E", args(&[policy]), &mut state),
                Some(signed_i32_return(KERNEL_PRIORITY_HIGHEST))
            );
            assert_eq!(
                dispatch("m0iS6jNsXds", args(&[policy]), &mut state),
                Some(signed_i32_return(KERNEL_PRIORITY_LOWEST))
            );
        }
        assert_eq!(
            dispatch("CBNtXOoef-E", args(&[99]), &mut state),
            Some(POSIX_FAILURE)
        );
        assert_eq!(
            dispatch("m0iS6jNsXds", args(&[99]), &mut state),
            Some(POSIX_FAILURE)
        );
    }

    #[test]
    fn the_process_queries_report_stable_values_the_guest_can_compare() {
        let mut state = ProbeDispatchState::default();

        assert_eq!(
            dispatch("HoLVWNanBBc", [0; 6], &mut state),
            Some(u64::from(std::process::id()))
        );
        assert_eq!(dispatch("6XG4B33N09g", [0; 6], &mut state), Some(0));
        assert_eq!(dispatch("4oXYe9Xmk0Q", [0; 6], &mut state), Some(0));
        assert_eq!(dispatch("ca7v6Cxulzs", args(&[0xff]), &mut state), Some(0));
        assert_eq!(dispatch("tU5e3f9gSiU", [0; 6], &mut state), Some(0));
        // With the default affinity the first allowed core is core zero.
        assert_eq!(dispatch("g0VTBxfJyu0", [0; 6], &mut state), Some(0));
    }

    #[test]
    fn a_thread_pinned_away_from_the_first_core_reports_the_first_core_it_allows() {
        let mut state = ProbeDispatchState::default();
        state.thread_attributes.insert(
            current_guest_thread_handle(),
            PthreadAttrState {
                affinity_mask: 0x30,
                ..PthreadAttrState::default()
            },
        );

        assert_eq!(dispatch("g0VTBxfJyu0", [0; 6], &mut state), Some(4));
    }

    #[test]
    fn an_address_is_formatted_and_parsed_back_to_the_bytes_it_started_from() {
        let mut state = ProbeDispatchState::default();
        let source = [192_u8, 168, 1, 200];
        let mut text = [0_u8; 46];
        let text_address = text.as_mut_ptr().addr() as u64;

        assert_eq!(
            dispatch(
                "5jRCs2axtr4",
                args(&[
                    u64::from(AF_INET.cast_unsigned()),
                    address(&source),
                    text_address,
                    46
                ]),
                &mut state
            ),
            Some(text_address)
        );
        assert_eq!(&text[..16], b"192.168.1.200\0\0\0");

        let mut parsed = [0_u8; 4];
        assert_eq!(
            dispatch(
                "4n51s0zEf0c",
                args(&[
                    u64::from(AF_INET.cast_unsigned()),
                    text_address,
                    parsed.as_mut_ptr().addr() as u64
                ]),
                &mut state
            ),
            Some(1)
        );
        assert_eq!(parsed, source);
    }

    #[test]
    fn a_sixteen_byte_address_round_trips_through_its_compressed_text_form() {
        let mut state = ProbeDispatchState::default();
        let source = [
            0x20_u8, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01,
        ];
        let mut text = [0_u8; 46];
        let text_address = text.as_mut_ptr().addr() as u64;

        assert_eq!(
            dispatch(
                "5jRCs2axtr4",
                args(&[
                    u64::from(AF_INET6.cast_unsigned()),
                    address(&source),
                    text_address,
                    46
                ]),
                &mut state
            ),
            Some(text_address)
        );
        assert_eq!(&text[..11], b"2001:db8::1");
        assert_eq!(text[11], 0);

        let mut parsed = [0_u8; 16];
        assert_eq!(
            dispatch(
                "4n51s0zEf0c",
                args(&[
                    u64::from(AF_INET6.cast_unsigned()),
                    text_address,
                    parsed.as_mut_ptr().addr() as u64
                ]),
                &mut state
            ),
            Some(1)
        );
        assert_eq!(parsed, source);
    }

    #[test]
    fn address_conversion_separates_an_unknown_family_from_text_that_is_not_an_address() {
        let mut state = ProbeDispatchState::default();
        let source = [1_u8, 2, 3, 4];
        let mut text = [0_u8; 8];
        let malformed = guest_string("192.168.1");

        assert_eq!(
            dispatch(
                "5jRCs2axtr4",
                args(&[
                    99,
                    address(&source),
                    text.as_mut_ptr().addr() as u64,
                    text.len() as u64
                ]),
                &mut state
            ),
            Some(0)
        );
        // A dotted quad needs more room than this buffer offers.
        let source = [192_u8, 168, 1, 200];
        assert_eq!(
            dispatch(
                "5jRCs2axtr4",
                args(&[
                    u64::from(AF_INET.cast_unsigned()),
                    address(&source),
                    text.as_mut_ptr().addr() as u64,
                    8
                ]),
                &mut state
            ),
            Some(0)
        );
        let mut parsed = [0_u8; 4];
        assert_eq!(
            dispatch(
                "4n51s0zEf0c",
                args(&[
                    u64::from(AF_INET.cast_unsigned()),
                    address(&malformed),
                    parsed.as_mut_ptr().addr() as u64
                ]),
                &mut state
            ),
            Some(0)
        );
        assert_eq!(
            dispatch(
                "4n51s0zEf0c",
                args(&[99, address(&malformed), parsed.as_mut_ptr().addr() as u64]),
                &mut state
            ),
            Some(POSIX_FAILURE)
        );
    }

    #[test]
    fn a_submitted_read_batch_fills_every_buffer_and_result_before_it_reports_an_identifier() {
        let directory = TemporaryDirectory::new("aio-read");
        let path = directory.path.join("data.bin");
        std::fs::write(&path, b"0123456789").expect("the test file can be written");
        let file = std::fs::File::open(&path).expect("the test file can be opened");
        let fd = register_descriptor(file);
        let mut state = ProbeDispatchState::default();
        let mut first = [0_u8; 4];
        let mut second = [0_u8; 3];
        let mut results = [0_u64; 4];
        let results_address = results.as_mut_ptr().addr() as u64;
        let mut requests = [0_u64; 10];
        requests[0] = 0;
        requests[1] = 4;
        requests[2] = first.as_mut_ptr().addr() as u64;
        requests[3] = results_address;
        requests[4] = u64::from(fd.cast_unsigned());
        requests[5] = 6;
        requests[6] = 3;
        requests[7] = second.as_mut_ptr().addr() as u64;
        requests[8] = results_address + 0x10;
        requests[9] = u64::from(fd.cast_unsigned());
        let mut identifier = 0_u32;

        let submitted = dispatch(
            "HgX7+AORI58",
            args(&[
                requests.as_ptr().addr() as u64,
                2,
                2,
                std::ptr::addr_of_mut!(identifier).addr() as u64,
            ]),
            &mut state,
        );

        assert_eq!(submitted, Some(SUCCESS));
        assert_eq!(&first, b"0123");
        assert_eq!(&second, b"678");
        // Each result carries the transferred count and then the terminal state.
        assert_eq!(results[0], 4);
        assert_eq!(
            results[1] & u64::from(u32::MAX),
            u64::from(AIO_STATE_COMPLETED)
        );
        assert_eq!(results[2], 3);
        assert_eq!(
            results[3] & u64::from(u32::MAX),
            u64::from(AIO_STATE_COMPLETED)
        );
        assert!(identifier > 0);

        let mut reported = 0_u32;
        let identifier_argument = u64::from(identifier);
        assert_eq!(
            dispatch(
                "2pOuoWoCxdk",
                args(&[
                    identifier_argument,
                    std::ptr::addr_of_mut!(reported).addr() as u64
                ]),
                &mut state
            ),
            Some(SUCCESS)
        );
        assert_eq!(reported, AIO_STATE_COMPLETED);
        assert_eq!(
            dispatch(
                "KOF-oJbQVvc",
                args(&[
                    identifier_argument,
                    std::ptr::addr_of_mut!(reported).addr() as u64
                ]),
                &mut state
            ),
            Some(SUCCESS)
        );
        let mut outcome = 0xffff_ffff_u32;
        assert_eq!(
            dispatch(
                "5TgME6AYty4",
                args(&[
                    identifier_argument,
                    std::ptr::addr_of_mut!(outcome).addr() as u64
                ]),
                &mut state
            ),
            Some(SUCCESS)
        );
        assert_eq!(outcome, 0);
        assert_eq!(
            dispatch(
                "5TgME6AYty4",
                args(&[
                    identifier_argument,
                    std::ptr::addr_of_mut!(outcome).addr() as u64
                ]),
                &mut state
            ),
            Some(KERNEL_NOT_FOUND)
        );
        release_descriptor(fd);
    }

    #[test]
    fn a_submitted_write_batch_reaches_the_host_file_at_the_offset_it_named() {
        let directory = TemporaryDirectory::new("aio-write");
        let path = directory.path.join("data.bin");
        std::fs::write(&path, b"0123456789").expect("the test file can be written");
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .expect("the test file can be opened");
        let fd = register_descriptor(file);
        let mut state = ProbeDispatchState::default();
        let payload = *b"XY";
        let mut requests = [0_u64; 5];
        requests[0] = 4;
        requests[1] = 2;
        requests[2] = address(&payload);
        requests[3] = 0;
        requests[4] = u64::from(fd.cast_unsigned());
        let mut identifier = 0_u32;

        assert_eq!(
            dispatch(
                "XQ8C8y+de+E",
                args(&[
                    requests.as_ptr().addr() as u64,
                    1,
                    3,
                    std::ptr::addr_of_mut!(identifier).addr() as u64
                ]),
                &mut state
            ),
            Some(SUCCESS)
        );
        assert_eq!(
            std::fs::read(&path).expect("the file is still there"),
            b"0123XY6789"
        );
        release_descriptor(fd);
    }

    #[test]
    fn a_submission_naming_a_bad_descriptor_or_count_is_refused_whole() {
        let mut state = ProbeDispatchState::default();
        let mut requests = [0_u64; 5];
        requests[1] = 4;
        requests[2] = 0x1000;
        requests[4] = 9_999;
        let mut identifier = 0_u32;
        let identifier_address = std::ptr::addr_of_mut!(identifier).addr() as u64;
        let requests_address = requests.as_ptr().addr() as u64;

        assert_eq!(
            dispatch(
                "HgX7+AORI58",
                args(&[requests_address, 1, 2, identifier_address]),
                &mut state
            ),
            Some(KERNEL_BAD_FILE_DESCRIPTOR)
        );
        assert_eq!(
            dispatch(
                "HgX7+AORI58",
                args(&[requests_address, 0, 2, identifier_address]),
                &mut state
            ),
            Some(KERNEL_INVALID_ARGUMENT)
        );
        assert_eq!(
            dispatch(
                "HgX7+AORI58",
                args(&[requests_address, 129, 2, identifier_address]),
                &mut state
            ),
            Some(KERNEL_INVALID_ARGUMENT)
        );
        assert_eq!(
            dispatch(
                "HgX7+AORI58",
                args(&[requests_address, 1, 9, identifier_address]),
                &mut state
            ),
            Some(KERNEL_INVALID_ARGUMENT)
        );
        assert_eq!(
            dispatch(
                "HgX7+AORI58",
                args(&[0, 1, 2, identifier_address]),
                &mut state
            ),
            Some(KERNEL_MEMORY_FAULT)
        );
        assert_eq!(
            dispatch(
                "HgX7+AORI58",
                args(&[requests_address, 1, 2, 0]),
                &mut state
            ),
            Some(KERNEL_MEMORY_FAULT)
        );
        assert_eq!(identifier, 0);
    }

    #[test]
    fn polling_several_identifiers_reports_nothing_unless_the_runtime_knows_them_all() {
        let mut state = ProbeDispatchState::default();
        let identifiers = [0x7fff_0001_u32, 0x7fff_0002];
        let mut states = [0xffff_ffff_u32; 2];

        assert_eq!(
            dispatch(
                "o7O4z3jwKzo",
                args(&[
                    identifiers.as_ptr().addr() as u64,
                    2,
                    states.as_mut_ptr().addr() as u64
                ]),
                &mut state
            ),
            Some(KERNEL_NOT_FOUND)
        );
        assert_eq!(states, [0xffff_ffff; 2]);
        assert_eq!(
            dispatch("o7O4z3jwKzo", args(&[0, 2, 0x1000]), &mut state),
            Some(KERNEL_MEMORY_FAULT)
        );
        assert_eq!(
            dispatch(
                "o7O4z3jwKzo",
                args(&[
                    identifiers.as_ptr().addr() as u64,
                    0,
                    states.as_mut_ptr().addr() as u64
                ]),
                &mut state
            ),
            Some(KERNEL_INVALID_ARGUMENT)
        );
    }

    #[test]
    fn a_nid_this_module_does_not_own_is_left_for_the_next_one() {
        let mut state = ProbeDispatchState::default();

        assert_eq!(dispatch("AAAAAAAAAAA", [0; 6], &mut state), None);
        assert_eq!(dispatch("", [0; 6], &mut state), None);
    }
}
