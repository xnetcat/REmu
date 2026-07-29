// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use metal::{CommandBufferRef, MTLCommandBufferStatus};
use std::thread;
use std::time::{Duration, Instant};

/// Longest a host thread blocks on one Metal command buffer before reporting it as stalled.
///
/// macOS kills `WindowServer` once it misses a 40-second watchdog checkin and panics the kernel at
/// 120 seconds. The presenter waits on guest batches from the `AppKit` main thread, so a submission
/// that never reaches a terminal state used to stall the host compositor instead of only the
/// emulator: `waitUntilCompleted` has no upper bound, and a batch the driver never schedules never
/// releases it. Silent Hill legitimately submits transition batches that remain scheduled for more
/// than five seconds, so keep a generous margin while still returning before the host watchdog.
///
/// Timing out does not make the command buffer terminal. Callers must retain its resources and
/// ordered guest completion effects until Metal later reports `Completed` or `Error`.
pub(crate) const COMMAND_BUFFER_SETTLE_TIMEOUT: Duration = Duration::from_secs(30);

/// Yield rather than sleep for this long first. Batches almost always finish inside it, so the
/// common path keeps `waitUntilCompleted`'s latency instead of paying a sleep quantum per wait.
const SETTLE_YIELD_PERIOD: Duration = Duration::from_millis(2);

/// Poll granularity once a batch has outlived [`SETTLE_YIELD_PERIOD`] and is worth sleeping on.
const SETTLE_POLL_INTERVAL: Duration = Duration::from_micros(250);

/// Blocks until `command_buffer` reaches a terminal state, giving up after
/// [`COMMAND_BUFFER_SETTLE_TIMEOUT`]. Returns whether it settled.
pub(crate) fn wait_until_settled(command_buffer: &CommandBufferRef) -> bool {
    wait_until_settled_within(command_buffer, COMMAND_BUFFER_SETTLE_TIMEOUT)
}

/// [`wait_until_settled`] with an explicit budget, so tests can exercise the timeout directly.
pub(crate) fn wait_until_settled_within(
    command_buffer: &CommandBufferRef,
    timeout: Duration,
) -> bool {
    let started = Instant::now();
    loop {
        if command_buffer_is_settled(command_buffer) {
            return true;
        }
        let waited = started.elapsed();
        let Some(remaining) = timeout.checked_sub(waited) else {
            // Re-read once more: the batch may have settled inside the final interval.
            return command_buffer_is_settled(command_buffer);
        };
        if waited < SETTLE_YIELD_PERIOD {
            thread::yield_now();
        } else {
            thread::sleep(SETTLE_POLL_INTERVAL.min(remaining));
        }
    }
}

/// Commits `command_buffer` and reports whether it completed, naming `operation` on failure.
pub(crate) fn commit_and_settle(
    command_buffer: &CommandBufferRef,
    operation: &str,
) -> Result<(), String> {
    command_buffer.commit();
    settle(command_buffer, operation)
}

/// Waits for an already-committed `command_buffer` and reports whether it completed.
pub(crate) fn settle(command_buffer: &CommandBufferRef, operation: &str) -> Result<(), String> {
    if !wait_until_settled(command_buffer) {
        return Err(settle_timeout_message(operation, command_buffer));
    }
    let status = command_buffer.status();
    if status != MTLCommandBufferStatus::Completed {
        return Err(format!("{operation} ended in {status:?}"));
    }
    Ok(())
}

/// Shared wording so a timed-out batch is never mistaken for a driver-reported failure.
pub(crate) fn settle_timeout_message(operation: &str, command_buffer: &CommandBufferRef) -> String {
    format!(
        "{operation} did not complete within {COMMAND_BUFFER_SETTLE_TIMEOUT:?} (status {:?})",
        command_buffer.status()
    )
}

fn command_buffer_is_settled(command_buffer: &CommandBufferRef) -> bool {
    matches!(
        command_buffer.status(),
        MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
    )
}
