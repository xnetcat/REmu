// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::completion::{COMMAND_BUFFER_SETTLE_TIMEOUT, wait_until_settled_within};
use metal::{Device, MTLCommandBufferStatus};
use std::time::{Duration, Instant};

/// A command buffer that waits on an event nobody signals models the wedged submission that took
/// the host down: `waitUntilCompleted` on it never returns.
#[test]
fn never_completing_command_buffer_gives_up_instead_of_blocking_forever() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let event = device.new_shared_event();
    let command_buffer = queue.new_command_buffer().to_owned();
    command_buffer.encode_wait_for_event(&event, 1);
    command_buffer.commit();

    let timeout = Duration::from_millis(200);
    let started = Instant::now();
    let settled = wait_until_settled_within(&command_buffer, timeout);
    let waited = started.elapsed();

    assert!(
        !settled,
        "an unsignaled event must not report a settled batch"
    );
    assert!(
        waited >= timeout,
        "the wait returned after {waited:?}, before its {timeout:?} budget"
    );
    assert!(
        waited < timeout * 10,
        "the wait ran {waited:?}, far past its {timeout:?} budget"
    );
    assert_ne!(command_buffer.status(), MTLCommandBufferStatus::Completed);

    // Release the submission so the test never leaves host GPU work outstanding.
    event.set_signaled_value(1);
    assert!(wait_until_settled_within(
        &command_buffer,
        COMMAND_BUFFER_SETTLE_TIMEOUT
    ));
    assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
}

#[test]
fn completing_command_buffer_settles_within_its_budget() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let command_buffer = queue.new_command_buffer().to_owned();
    command_buffer.commit();

    assert!(wait_until_settled_within(
        &command_buffer,
        COMMAND_BUFFER_SETTLE_TIMEOUT
    ));
    assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
}

/// Draining a queue of stalled batches shares one deadline, so every batch after the first waits
/// on an exhausted budget. That has to return at once, or N batches would cost N whole timeouts.
#[test]
fn exhausted_budget_reports_without_waiting() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let event = device.new_shared_event();
    let command_buffer = queue.new_command_buffer().to_owned();
    command_buffer.encode_wait_for_event(&event, 1);
    command_buffer.commit();

    let started = Instant::now();
    let settled = wait_until_settled_within(&command_buffer, Duration::ZERO);
    let waited = started.elapsed();

    assert!(
        !settled,
        "an unsignaled event must not report a settled batch"
    );
    assert!(
        waited < Duration::from_millis(50),
        "an exhausted budget waited {waited:?} instead of returning at once"
    );

    event.set_signaled_value(1);
    assert!(wait_until_settled_within(
        &command_buffer,
        COMMAND_BUFFER_SETTLE_TIMEOUT
    ));
}

/// The budget has to stay far enough inside the 40-second `WindowServer` watchdog checkin that a
/// failing batch is reported as a `REmu` error rather than escalating into a host watchdog kill.
#[test]
fn settle_timeout_stays_inside_the_host_watchdog_budget() {
    assert!(COMMAND_BUFFER_SETTLE_TIMEOUT < Duration::from_secs(40));
}
