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
//!
//! Pump death is the hazard this design has to answer for: fd 2 is a pipe with exactly one
//! reader, so a pump that stops reading turns every `eprintln!` in the process into a permanent
//! block once 64 KiB has accumulated — while holding `std::io::Stderr`'s lock. The pump
//! therefore treats `EINTR` as routine on both sides, and gives up only by first putting the
//! real stderr back on fd 2 (see [`restore_real_stderr`]), so the failure mode is a reported
//! write error at the call site rather than a silently wedged process.

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

/// What the pump does with one `write(2)` result.
#[derive(Debug, PartialEq, Eq)]
enum WriteStep {
    /// Bytes went out; advance the cursor by this many.
    Advanced(usize),
    /// The call was interrupted before writing anything; issue it again unchanged.
    Retry,
    /// The target will not accept these bytes, now or later.
    Fatal,
}

/// Classifies a `write(2)` return so the pump's policy is decided in one testable place.
///
/// `EINTR` is routine rather than exceptional here: the process installs its fault handlers
/// without `SA_RESTART` (`write_watch/signal.rs`), so any signal delivered while the pump is
/// mid-write surfaces as `-1`/`EINTR`. Treating that as fatal is what killed the pump, and a
/// dead pump makes every later `eprintln!` block forever on a pipe nothing drains. The read side
/// already retried it; the write side now matches.
fn classify_write(sent: isize, errno: i32) -> WriteStep {
    if sent > 0 {
        return WriteStep::Advanced(sent as usize);
    }
    if sent < 0 && errno == libc::EINTR {
        return WriteStep::Retry;
    }
    // A zero return for a non-empty request makes no progress; looping on it would spin.
    WriteStep::Fatal
}

/// Puts the real stderr back on fd 2 so writers stop queueing into a pipe with no reader.
///
/// Without this the pump's death is silent and terminal: 942 `eprintln!` sites fill the 64 KiB
/// pipe, then block inside `write(2)` while holding `std::io::Stderr`'s lock, and the process
/// wedges. Restoring fd 2 returns the process to the behaviour it would have had if the pump had
/// never been installed — a write error on stderr, reported at the call site.
fn restore_real_stderr(real_stderr: i32) {
    // Publish the pipe as gone before rewiring, so a concurrent `flush_async_stderr` stops
    // waiting on a queue that will never drain.
    PIPE_READ.store(-1, Ordering::Release);
    // SAFETY: `real_stderr` is the pump-owned duplicate of the original stderr, live for the
    // pump's whole lifetime; `dup2` onto fd 2 closes the pipe write end it replaces.
    unsafe {
        libc::dup2(real_stderr, libc::STDERR_FILENO);
    }
}

fn pump(read_end: i32, real_stderr: i32) {
    let mut buffer = [0_u8; 64 * 1024];
    // Set once the forwarding target is unusable. The pump keeps reading anyway: writers already
    // blocked on the full pipe when the target died can only finish if somebody drains it, and
    // discarding is the only remaining option once the bytes have nowhere to go. Everything
    // authored after `restore_real_stderr` bypasses the pipe entirely.
    let mut discarding = false;
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
            // The pipe is unreadable, so nothing will drain it either — same deadlock as a dead
            // write side, and the same remedy.
            if !discarding {
                restore_real_stderr(real_stderr);
            }
            return;
        }
        if discarding {
            continue;
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
            let errno = if sent < 0 {
                std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
            } else {
                0
            };
            match classify_write(sent, errno) {
                WriteStep::Advanced(bytes) => written += bytes as isize,
                WriteStep::Retry => {}
                WriteStep::Fatal => {
                    restore_real_stderr(real_stderr);
                    discarding = true;
                    break;
                }
            }
        }
    }
}

/// Waits (bounded) until the stderr pipe is empty so exit-path reports reach the log file.
///
/// Returns immediately when the pump never owned stderr, and also once the pump has given up and
/// restored fd 2 — after that the pipe holds only bytes nobody will ever forward, and waiting a
/// second for them would only slow down an exit.
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

#[cfg(test)]
mod tests {
    use super::{WriteStep, classify_write};

    #[test]
    fn an_interrupted_write_is_retried_rather_than_killing_the_pump() {
        assert_eq!(classify_write(-1, libc::EINTR), WriteStep::Retry);
    }

    #[test]
    fn a_short_write_advances_the_cursor_by_what_went_out() {
        assert_eq!(classify_write(4096, 0), WriteStep::Advanced(4096));
        assert_eq!(classify_write(1, 0), WriteStep::Advanced(1));
    }

    #[test]
    fn a_closed_or_stalled_target_is_fatal() {
        // `remu run … 2>&1 | head`: the reader exits and stderr answers EPIPE from then on.
        assert_eq!(classify_write(-1, libc::EPIPE), WriteStep::Fatal);
        assert_eq!(classify_write(-1, libc::EBADF), WriteStep::Fatal);
        // Zero bytes for a non-empty request is no progress; retrying would spin the pump.
        assert_eq!(classify_write(0, 0), WriteStep::Fatal);
    }
}
