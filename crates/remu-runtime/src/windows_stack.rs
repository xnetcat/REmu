// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Windows stack preparation for native SysV guest execution.
//!
//! Windows grows a thread stack one guard page at a time. PS5 machine code has no Windows
//! `__chkstk` probes, so a large frame can skip the guard page and touch a reserved page directly,
//! producing `STATUS_ACCESS_VIOLATION` despite ample stack reservation. Guest-capable threads
//! commit their existing reservation before entering console code. Committing reserved pages does
//! not update the TEB's `StackLimit`, so this module also lowers that metadata to the reservation
//! boundary. Native Windows code otherwise rejects an otherwise-valid guest RSP with
//! `FAST_FAIL_STACK_COOKIE_CHECK_FAILURE` after the guest crosses the old guard-page boundary.

use std::io;

use windows::Win32::System::Kernel::NT_TIB;
use windows::Win32::System::Memory::{MEM_COMMIT, PAGE_READWRITE, VirtualAlloc};
use windows::Win32::System::Threading::GetCurrentThreadStackLimits;

#[derive(Clone, Copy, Debug)]
struct StackMetadata {
    low: usize,
    high: usize,
    tib: *mut NT_TIB,
    teb_limit: usize,
}

fn current_thread_information_block() -> io::Result<*mut NT_TIB> {
    #[cfg(target_arch = "x86_64")]
    {
        let tib: *mut NT_TIB;
        // NtCurrentTeb is a Windows header intrinsic rather than an exported function. On x64,
        // the documented NT_TIB Self field is available through GS:[0x30].
        // SAFETY: this only reads the current thread's segment-relative NT_TIB pointer.
        unsafe {
            core::arch::asm!(
                "mov {}, gs:[0x30]",
                out(reg) tib,
                options(nostack, preserves_flags, readonly)
            );
        }
        if tib.is_null() {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Windows returned a null current-thread TIB",
            ))
        } else {
            Ok(tib)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "native PS5 stack preparation requires Windows x86-64",
        ))
    }
}

fn current_stack_metadata() -> io::Result<StackMetadata> {
    let mut low = 0_usize;
    let mut high = 0_usize;
    // SAFETY: both out-pointers remain valid for this call.
    unsafe { GetCurrentThreadStackLimits(&raw mut low, &raw mut high) };
    if high <= low {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid current-thread stack limits {low:#x}..{high:#x}"),
        ));
    }

    let tib = current_thread_information_block()?;
    // SAFETY: `tib` is the current thread's NT_TIB and remains valid for the thread's lifetime.
    let (self_pointer, teb_base, teb_limit) = unsafe {
        (
            (*tib).Self_,
            (*tib).StackBase.addr(),
            (*tib).StackLimit.addr(),
        )
    };
    if self_pointer != tib || teb_base != high || !(low..high).contains(&teb_limit) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "inconsistent Windows stack metadata api={low:#x}..{high:#x} teb={teb_limit:#x}..{teb_base:#x} self={self_pointer:p}"
            ),
        ));
    }

    Ok(StackMetadata {
        low,
        high,
        tib,
        teb_limit,
    })
}

/// Commits the complete stack reservation Windows assigned to the current thread.
///
/// Calling `VirtualAlloc(MEM_COMMIT)` over already committed pages is explicitly idempotent. The
/// call only makes the still-reserved portion resident and does not reserve a larger stack. The
/// TEB limit is then lowered to the same allocation boundary so Windows-native callees accept a
/// guest stack pointer anywhere inside those now-committed pages.
pub(crate) fn precommit_current_thread_stack() -> io::Result<usize> {
    let metadata = current_stack_metadata()?;
    let byte_count = metadata.high - metadata.low;
    // SAFETY: the exact range was reserved by Windows for this current thread. MEM_COMMIT may be
    // repeated for pages already resident and leaves the reservation ownership unchanged.
    let committed = unsafe {
        VirtualAlloc(
            Some(metadata.low as *const core::ffi::c_void),
            byte_count,
            MEM_COMMIT,
            PAGE_READWRITE,
        )
    };
    if committed.is_null() {
        return Err(io::Error::last_os_error());
    }
    if committed.addr() != metadata.low {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Windows committed stack at {:#x}, expected {:#x}",
                committed.addr(),
                metadata.low
            ),
        ));
    }
    // VirtualAlloc changes page state but deliberately does not rewrite per-thread stack
    // metadata. The current thread exclusively owns its NT_TIB, and every address down to `low`
    // is now committed as writable stack memory.
    // SAFETY: `metadata.tib` was validated as the current NT_TIB above. This updates only its
    // StackLimit field after the corresponding reservation was successfully committed.
    if metadata.teb_limit != metadata.low {
        unsafe {
            (*metadata.tib).StackLimit = metadata.low as *mut core::ffi::c_void;
        }
    }
    Ok(byte_count)
}

#[cfg(test)]
mod tests {
    use super::{current_stack_metadata, precommit_current_thread_stack};

    #[test]
    fn commits_a_dedicated_guest_thread_stack_reservation() {
        const REQUESTED_STACK_BYTES: usize = 2 * 1024 * 1024;
        let (committed, reservation_size, old_limit, new_limit, low) = std::thread::Builder::new()
            .stack_size(REQUESTED_STACK_BYTES)
            .spawn(|| {
                let before = current_stack_metadata()?;
                let committed = precommit_current_thread_stack()?;
                let after = current_stack_metadata()?;
                Ok::<_, std::io::Error>((
                    committed,
                    before.high - before.low,
                    before.teb_limit,
                    after.teb_limit,
                    before.low,
                ))
            })
            .expect("dedicated stack test thread starts")
            .join()
            .expect("dedicated stack test thread does not panic")
            .expect("the complete Windows stack reservation commits");

        assert!(committed >= REQUESTED_STACK_BYTES);
        assert_eq!(committed, reservation_size);
        assert!(
            old_limit > low,
            "Windows begins with only the stack top committed"
        );
        assert_eq!(new_limit, low, "the TEB must describe every committed page");
    }
}
