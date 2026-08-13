// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Asynchronous stderr: every `eprintln!` in the process stops paying for file I/O.
//!
//! The emulator narrates through stderr from hot paths — draw diagnostics, shader compiles,
//! perf reports — and each line is a synchronous `write(2)` under `std::io::Stderr`'s lock.
//! [`install_async_stderr`] replaces file descriptor 2 with a pipe drained by one pump thread
//! that forwards bytes to the original stderr target, so authoring threads only copy into the
//! in-kernel pipe buffer. Ordering across threads is preserved: fd 2 is a single pipe and the
//! pump writes bytes in arrival order.
//!
//! Crash safety: the original stderr is kept on a private descriptor. Signal-context dumps
//! (the guest-write journal) write to that descriptor directly via [`crash_output_fd`] —
//! bytes sitting in the pipe when the process dies are lost, so the last-chance path must not
//! go through it. Orderly exits call [`flush_async_stderr`], which waits (bounded) for the
//! pipe to drain.

use std::sync::atomic::{AtomicI32, Ordering};

/// Set to `1` to keep stderr synchronous (diagnosing the pump itself, or CI byte-exactness).
pub const SYNC_STDERR_VARIABLE: &str = "REMU_SYNC_STDERR";

/// The preserved real stderr while the pump owns fd 2; `-1` until installed.
static REAL_STDERR: AtomicI32 = AtomicI32::new(-1);
/// The pipe read end, kept for [`flush_async_stderr`]'s drain probe; `-1` until installed.
static PIPE_READ: AtomicI32 = AtomicI32::new(-1);

/// Descriptor that async-signal-safe crash dumps must write to.
///
/// Before installation (or when the pump is disabled) this is plain stderr; afterwards it is
/// the preserved real target, bypassing the pipe a dying process would never drain.
#[must_use]
pub fn crash_output_fd() -> i32 {
    match REAL_STDERR.load(Ordering::Acquire) {
        -1 => libc::STDERR_FILENO,
        fd => fd,
    }
}

/// Replaces fd 2 with a pump-drained pipe. Call once, early, from the process entry point.
///
/// Returns whether the pump now owns stderr. `REMU_SYNC_STDERR=1` and any installation
/// failure leave stderr untouched — logging stays synchronous but correct.
pub fn install_async_stderr() -> bool {
    if std::env::var_os(SYNC_STDERR_VARIABLE).is_some_and(|value| value == "1") {
        return false;
    }
    if REAL_STDERR.load(Ordering::Acquire) != -1 {
        return true;
    }
    // SAFETY: plain descriptor plumbing on startup-owned descriptors; every call is checked.
    unsafe {
        let real = libc::dup(libc::STDERR_FILENO);
        if real < 0 {
            return false;
        }
        // The preserved target must survive accidental closes of "high" fds by the guest; it
        // must also not leak into spawned children as their stderr.
        libc::fcntl(real, libc::F_SETFD, libc::FD_CLOEXEC);
        let mut ends = [0_i32; 2];
        if libc::pipe(ends.as_mut_ptr()) != 0 {
            libc::close(real);
            return false;
        }
        let (read_end, write_end) = (ends[0], ends[1]);
        libc::fcntl(read_end, libc::F_SETFD, libc::FD_CLOEXEC);
        if libc::dup2(write_end, libc::STDERR_FILENO) < 0 {
            libc::close(read_end);
            libc::close(write_end);
            libc::close(real);
            return false;
        }
        libc::close(write_end);
        let spawned = std::thread::Builder::new()
            .name("remu-stderr-pump".into())
            .spawn(move || pump(read_end, real));
        if spawned.is_err() {
            // Restore synchronous stderr; the pipe would otherwise fill and block writers.
            libc::dup2(real, libc::STDERR_FILENO);
            libc::close(read_end);
            libc::close(real);
            return false;
        }
        PIPE_READ.store(read_end, Ordering::Release);
        REAL_STDERR.store(real, Ordering::Release);
    }
    true
}

fn pump(read_end: i32, real_stderr: i32) {
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        // SAFETY: the pump exclusively owns reads on its pipe end; the buffer outlives the call.
        let received = unsafe {
            libc::read(
                read_end,
                buffer.as_mut_ptr().cast::<std::ffi::c_void>(),
                buffer.len(),
            )
        };
        if received == 0 {
            return;
        }
        if received < 0 {
            if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return;
        }
        let mut written = 0_isize;
        while written < received {
            // SAFETY: forwards exactly the received prefix; short writes advance the cursor.
            let sent = unsafe {
                libc::write(
                    real_stderr,
                    buffer.as_ptr().offset(written).cast::<std::ffi::c_void>(),
                    (received - written) as usize,
                )
            };
            if sent <= 0 {
                return;
            }
            written += sent;
        }
    }
}

/// Waits (bounded) until the stderr pipe is empty so exit-path reports reach the log file.
pub fn flush_async_stderr() {
    let read_end = PIPE_READ.load(Ordering::Acquire);
    if read_end < 0 {
        return;
    }
    for _ in 0..200 {
        let mut pending: libc::c_int = 0;
        // SAFETY: FIONREAD on an owned pipe read end writes one c_int.
        let ok = unsafe { libc::ioctl(read_end, libc::FIONREAD, &raw mut pending) };
        if ok != 0 || pending == 0 {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}
