// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Host page geometry and the mprotect transitions the watch primitive relies on.

use std::ffi::c_void;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) static HOST_PAGE_SIZE: AtomicU64 = AtomicU64::new(0);

pub(super) fn protect_page(page_address: u64, read_only: bool) -> io::Result<()> {
    protect_range(page_address, host_page_size_infallible(), read_only)
}

fn protect_range(address: u64, byte_count: u64, read_only: bool) -> io::Result<()> {
    let address = usize::try_from(address)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "page address does not fit"))?
        as *mut c_void;
    let byte_count = usize::try_from(byte_count)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "byte count does not fit"))?;
    let protection = libc::PROT_READ | if read_only { 0 } else { libc::PROT_WRITE };
    // SAFETY: CpuWriteWatch's contract keeps every registered page live and permits exactly these
    // read-only/read-write transitions.
    if unsafe { libc::mprotect(address, byte_count, protection) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

pub(super) fn host_page_size() -> io::Result<u64> {
    // SAFETY: sysconf has no pointer preconditions.
    let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page_size =
        u64::try_from(value).map_err(|_| io::Error::other("host page size is unavailable"))?;
    if !page_size.is_power_of_two() {
        return Err(io::Error::other("host page size is invalid"));
    }
    Ok(page_size)
}

pub(super) fn host_page_size_infallible() -> u64 {
    HOST_PAGE_SIZE.load(Ordering::Acquire)
}

pub(super) const fn align_down(value: u64, alignment: u64) -> u64 {
    value & !(alignment - 1)
}

pub(super) const fn align_up(value: u64, alignment: u64) -> Option<u64> {
    match value.checked_add(alignment - 1) {
        Some(sum) => Some(sum & !(alignment - 1)),
        None => None,
    }
}
