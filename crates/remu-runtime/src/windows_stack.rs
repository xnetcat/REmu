// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Windows stack preparation for native SysV guest execution.
//!
//! Windows grows a thread stack one guard page at a time. PS5 machine code has no Windows
//! `__chkstk` probes, so a large frame can skip the guard page and touch a reserved page directly,
//! producing `STATUS_ACCESS_VIOLATION` despite ample stack reservation. Guest-capable threads
//! commit their existing reservation before entering console code; the reservation boundaries and
//! overflow behavior remain fixed by Windows.

use std::io;

use windows::Win32::System::Memory::{MEM_COMMIT, PAGE_READWRITE, VirtualAlloc};
use windows::Win32::System::Threading::GetCurrentThreadStackLimits;

/// Commits the complete stack reservation Windows assigned to the current thread.
///
/// Calling `VirtualAlloc(MEM_COMMIT)` over already committed pages is explicitly idempotent. The
/// call only makes the still-reserved portion resident; it neither reserves a larger stack nor
/// changes the bounds returned by `GetCurrentThreadStackLimits`.
pub(crate) fn precommit_current_thread_stack() -> io::Result<usize> {
    let mut low = 0_usize;
    let mut high = 0_usize;
    // SAFETY: both out-pointers remain valid for this call.
    unsafe { GetCurrentThreadStackLimits(&raw mut low, &raw mut high) };
    let byte_count = high
        .checked_sub(low)
        .filter(|size| *size != 0)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid current-thread stack limits {low:#x}..{high:#x}"),
            )
        })?;
    // SAFETY: the exact range was reserved by Windows for this current thread. MEM_COMMIT may be
    // repeated for pages already resident and leaves the reservation ownership unchanged.
    let committed = unsafe {
        VirtualAlloc(
            Some(low as *const core::ffi::c_void),
            byte_count,
            MEM_COMMIT,
            PAGE_READWRITE,
        )
    };
    if committed.is_null() {
        return Err(io::Error::last_os_error());
    }
    if committed.addr() != low {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Windows committed stack at {:#x}, expected {low:#x}",
                committed.addr()
            ),
        ));
    }
    Ok(byte_count)
}

#[cfg(test)]
mod tests {
    use super::precommit_current_thread_stack;

    #[test]
    fn commits_a_dedicated_guest_thread_stack_reservation() {
        const REQUESTED_STACK_BYTES: usize = 2 * 1024 * 1024;
        let committed = std::thread::Builder::new()
            .stack_size(REQUESTED_STACK_BYTES)
            .spawn(precommit_current_thread_stack)
            .expect("dedicated stack test thread starts")
            .join()
            .expect("dedicated stack test thread does not panic")
            .expect("the complete Windows stack reservation commits");

        assert!(committed >= REQUESTED_STACK_BYTES);
    }
}
