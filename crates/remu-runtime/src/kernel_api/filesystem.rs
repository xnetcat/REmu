// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::{
    ProbeDispatchState, SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR, SCE_KERNEL_ERROR_INVALID_ARGUMENT,
    SCE_KERNEL_ERROR_IS_DIRECTORY, SCE_KERNEL_ERROR_MEMORY_FAULT, SCE_KERNEL_ERROR_NOT_FOUND,
    SCE_KERNEL_ERROR_OUT_OF_MEMORY, SCE_KERNEL_ERROR_PERMISSION_DENIED, low_i32, low_u32,
    process_guest_range_has_read_access, read_guest_cstring, read_guest_u64,
    resolve_probe_guest_path, sce_kernel_error, sce_kernel_io_error,
};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{FileExt, PermissionsExt};

use super::{
    KERNEL_MODE_BITS, MICROSECONDS_PER_SECOND, TIMEVAL_MICROSECONDS_OFFSET, TIMEVAL_SIZE,
    guest_range_is_readable, kernel_files,
};

/// Resolves a guest path that a mutating call is about to act on.
///
/// Mounts are read-only as a whole, so a call that would change anything under one is refused
/// before it reaches the host filesystem.
pub(super) fn resolve_writable_path(
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
pub(super) fn kernel_unlink(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
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
pub(super) fn kernel_rename(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
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
pub(super) fn kernel_chmod(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
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
pub(super) fn kernel_truncate(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
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
pub(super) fn read_guest_timeval(address: u64) -> Option<libc::timeval> {
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
pub(super) fn kernel_utimes(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
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
pub(super) fn kernel_check_reachability(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
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
pub(super) fn kernel_fsync(arguments: [u64; 6]) -> u64 {
    let fd = low_i32(arguments[0]);
    let files = kernel_files();
    let Some(file) = files.files.get(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    };
    file.sync_all()
        .map_or_else(|error| sce_kernel_io_error(&error), |()| 0)
}

/// Transfers bytes at an explicit file offset without disturbing the descriptor's shared cursor.
pub(super) fn kernel_positional_transfer(arguments: [u64; 6], write: bool) -> u64 {
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
        if !process_guest_range_has_read_access(buffer, length) {
            return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
        }
        let mut staging = Vec::new();
        if staging.try_reserve_exact(length).is_err() {
            return sce_kernel_error(SCE_KERNEL_ERROR_OUT_OF_MEMORY);
        }
        staging.resize(length, 0_u8);
        file.read_at(&mut staging, offset.cast_unsigned())
            .inspect(|&transferred| {
                // SAFETY: the complete guest destination range was validated above. Copying with
                // CPU stores lets CpuWriteWatch synchronously disarm and dirty protected pages.
                unsafe {
                    std::ptr::copy_nonoverlapping(staging.as_ptr(), buffer as *mut u8, transferred);
                }
            })
    };
    match transferred {
        Ok(transferred) => u64::try_from(transferred).expect("transferred byte count fits u64"),
        Err(error) => sce_kernel_io_error(&error),
    }
}
