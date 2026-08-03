//! Event queues and VideoOut flip retirement behind their own lock (plan C2 step 2 / B20).
//!
//! Extracting this state lets an equeue wake stop taking the process-wide dispatch write lock and
//! lets per-import flip refresh move to the flip/equeue paths only.
//!
//! Migration note: [`ProbeDispatchState`] still mirrors these fields during Solo S1 so existing
//! call sites compile; new and migrated paths take [`lock_event_state_for_write`] /
//! [`lock_event_state_for_read`]. Full field deletion from `ProbeDispatchState` is the follow-up
//! seam once every equeue/flip site is retargeted.

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{OnceLock, RwLock};

use crate::dispatch_locks::{LockHierarchyGuard, LockLevel};
use crate::{KernelQueuedEvent, PendingVideoOutFlip};

#[derive(Debug, Default)]
pub(crate) struct EventState {
    pub(crate) next_equeue: u64,
    pub(crate) kernel_equeues: BTreeSet<u64>,
    pub(crate) kernel_event_registrations: BTreeMap<(u64, i16, u64), u64>,
    pub(crate) pending_events: VecDeque<KernelQueuedEvent>,
    pub(crate) pending_video_out_flips: VecDeque<PendingVideoOutFlip>,
    pub(crate) pending_agc_completions: VecDeque<(u64, u64)>,
    pub(crate) pending_agc_user_interrupts: VecDeque<u64>,
}

static EVENT_STATE: OnceLock<RwLock<EventState>> = OnceLock::new();

fn cell() -> &'static RwLock<EventState> {
    EVENT_STATE.get_or_init(|| RwLock::new(EventState::default()))
}

pub(crate) fn lock_event_state_for_write()
-> (
    LockHierarchyGuard,
    std::sync::RwLockWriteGuard<'static, EventState>,
) {
    let hierarchy = LockHierarchyGuard::enter(LockLevel::EventState);
    let guard = cell()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    (hierarchy, guard)
}

pub(crate) fn lock_event_state_for_read()
-> (
    LockHierarchyGuard,
    std::sync::RwLockReadGuard<'static, EventState>,
) {
    let hierarchy = LockHierarchyGuard::enter(LockLevel::EventState);
    let guard = cell()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    (hierarchy, guard)
}

/// Copies mirrored event fields from dispatch state into the dedicated event lock (bootstrap).
pub(crate) fn sync_from_dispatch(state: &crate::ProbeDispatchState) {
    let (_h, mut events) = lock_event_state_for_write();
    events.next_equeue = state.next_equeue;
    events.kernel_equeues.clone_from(&state.kernel_equeues);
    events
        .kernel_event_registrations
        .clone_from(&state.kernel_event_registrations);
    events.pending_events.clone_from(&state.pending_events);
    events
        .pending_video_out_flips
        .clone_from(&state.pending_video_out_flips);
    events
        .pending_agc_completions
        .clone_from(&state.pending_agc_completions);
    events
        .pending_agc_user_interrupts
        .clone_from(&state.pending_agc_user_interrupts);
}

/// Pushes a pending event under the event lock and mirrors into `state` during the dual-write window.
pub(crate) fn queue_event_dual(
    state: &mut crate::ProbeDispatchState,
    event: KernelQueuedEvent,
) {
    state.pending_events.push_back(event.clone());
    let (_h, mut events) = lock_event_state_for_write();
    events.pending_events.push_back(event);
}

pub(crate) fn reset_event_state_for_tests() {
    let mut state = cell()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *state = EventState::default();
}

#[cfg(test)]
mod tests {
    use super::{lock_event_state_for_write, reset_event_state_for_tests};

    #[test]
    fn event_state_lock_is_usable_without_probe_dispatch() {
        reset_event_state_for_tests();
        let (_h, mut state) = lock_event_state_for_write();
        state.next_equeue = 7;
        state.kernel_equeues.insert(7);
        assert!(state.kernel_equeues.contains(&7));
    }
}
