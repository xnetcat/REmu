// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::supervise_diagnostic_timeout;
use std::cell::{Cell, RefCell};
use std::time::Duration;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TimeoutAction {
    Wait(Duration),
    Expire,
    RequestPresenter,
    HardFallback,
}

#[test]
fn diagnostic_timeout_requests_owner_cleanup_before_its_bounded_hard_fallback() {
    let limit = Duration::from_secs(7);
    let grace = Duration::from_secs(3);
    let actions = RefCell::new(Vec::new());

    supervise_diagnostic_timeout(
        limit,
        grace,
        |duration| actions.borrow_mut().push(TimeoutAction::Wait(duration)),
        || false,
        || actions.borrow_mut().push(TimeoutAction::Expire),
        || {
            actions.borrow_mut().push(TimeoutAction::RequestPresenter);
            true
        },
        || actions.borrow_mut().push(TimeoutAction::HardFallback),
    );

    assert_eq!(
        actions.into_inner(),
        [
            TimeoutAction::Wait(limit),
            TimeoutAction::Expire,
            TimeoutAction::RequestPresenter,
            TimeoutAction::Wait(grace),
            TimeoutAction::HardFallback,
        ],
        "Metal cleanup is requested from its owning loop before the process-wide backstop"
    );
}

#[test]
fn diagnostic_timeout_does_not_fire_the_fallback_after_graceful_shutdown() {
    let limit = Duration::from_secs(7);
    let grace = Duration::from_secs(3);
    let cancelled = Cell::new(false);
    let hard_fallbacks = Cell::new(0);

    supervise_diagnostic_timeout(
        limit,
        grace,
        |duration| {
            if duration == grace {
                cancelled.set(true);
            }
        },
        || cancelled.get(),
        || {},
        || true,
        || hard_fallbacks.set(hard_fallbacks.get() + 1),
    );

    assert_eq!(hard_fallbacks.get(), 0);
}

#[test]
fn diagnostic_probe_without_a_presenter_falls_back_at_the_original_limit() {
    let limit = Duration::from_secs(7);
    let waited = Cell::new(Duration::ZERO);
    let hard_fallbacks = Cell::new(0);

    supervise_diagnostic_timeout(
        limit,
        Duration::from_secs(35),
        |duration| waited.set(waited.get() + duration),
        || false,
        || {},
        || false,
        || hard_fallbacks.set(hard_fallbacks.get() + 1),
    );

    assert_eq!(waited.get(), limit);
    assert_eq!(hard_fallbacks.get(), 1);
}
