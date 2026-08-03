//! Documented lock hierarchy for the split dispatch state (plan C2 / B20 guard rails).
//!
//! Order (lowest → highest; never acquire a lower level while holding a higher one):
//! 1. [`LockLevel::PathRoots`] — snapshot publish only (readers are lock-free via `Arc`)
//! 2. [`LockLevel::GuestSync`] — per-object sync registry / shards
//! 3. [`LockLevel::EventState`] — equeues, registrations, pending events, flips
//! 4. [`LockLevel::AgcState`] — graphics/compute/shader registry
//! 5. [`LockLevel::ProbeDispatch`] — remaining process service graph
//!
//! Parking under [`LockLevel::ProbeDispatch`] (or any level above `GuestSync` while waiting on a
//! guest sync object) is rejected by [`crate::assert_wait_is_off_the_dispatch_lock`].

#![allow(dead_code)]

use std::cell::Cell;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub(crate) enum LockLevel {
    PathRoots = 1,
    GuestSync = 2,
    EventState = 3,
    AgcState = 4,
    ProbeDispatch = 5,
}

thread_local! {
    static HELD_LEVEL: Cell<u8> = const { Cell::new(0) };
}

/// RAII record that this thread holds `level`. Drop restores the previous held level.
pub(crate) struct LockHierarchyGuard {
    previous: u8,
}

impl LockHierarchyGuard {
    /// Records acquisition of `level`. Panics under `debug_assertions` / test builds when the
    /// thread already holds a *higher* level (lower-then-higher violation).
    pub(crate) fn enter(level: LockLevel) -> Self {
        let previous = HELD_LEVEL.get();
        let next = level as u8;
        #[cfg(any(test, debug_assertions))]
        {
            assert!(
                previous == 0 || previous <= next,
                "lock-hierarchy violation: acquiring {level:?} ({next}) while holding level {previous}"
            );
        }
        HELD_LEVEL.set(next);
        Self { previous }
    }
}

impl Drop for LockHierarchyGuard {
    fn drop(&mut self) {
        HELD_LEVEL.set(self.previous);
    }
}

/// Highest lock level currently held on this thread, or `None`.
#[must_use]
pub(crate) fn held_lock_level() -> Option<LockLevel> {
    match HELD_LEVEL.get() {
        0 => None,
        1 => Some(LockLevel::PathRoots),
        2 => Some(LockLevel::GuestSync),
        3 => Some(LockLevel::EventState),
        4 => Some(LockLevel::AgcState),
        5 => Some(LockLevel::ProbeDispatch),
        other => panic!("corrupt lock-hierarchy depth {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{LockHierarchyGuard, LockLevel, held_lock_level};

    #[test]
    fn acquiring_in_ascending_order_is_allowed() {
        let _path = LockHierarchyGuard::enter(LockLevel::PathRoots);
        let _sync = LockHierarchyGuard::enter(LockLevel::GuestSync);
        let _events = LockHierarchyGuard::enter(LockLevel::EventState);
        assert_eq!(held_lock_level(), Some(LockLevel::EventState));
    }

    #[test]
    #[should_panic(expected = "lock-hierarchy violation")]
    fn acquiring_a_lower_level_while_holding_a_higher_one_panics() {
        let _dispatch = LockHierarchyGuard::enter(LockLevel::ProbeDispatch);
        let _events = LockHierarchyGuard::enter(LockLevel::EventState);
    }
}
