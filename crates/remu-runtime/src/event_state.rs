//! Event queues and VideoOut flip retirement behind their own lock (plan C2 step 2 / B20).
//!
//! Sole owner of equeue / registration / pending-event / flip / AGC-completion queues. Call sites
//! that already hold [`LockLevel::ProbeDispatch`] may nest this lock (see
//! [`lock_event_state_for_write`]); EventState critical sections must never acquire ProbeDispatch.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::dispatch_locks::{LockHierarchyGuard, LockLevel, held_lock_level};
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
    EVENT_STATE.get_or_init(|| RwLock::new(EventState {
        next_equeue: 1,
        ..EventState::default()
    }))
}

/// Write lock for EventState. When ProbeDispatch is already held, nests without re-entering the
/// hierarchy (producers under the dispatch lock still publish into EventState as sole store).
pub(crate) fn lock_event_state_for_write()
-> (
    Option<LockHierarchyGuard>,
    RwLockWriteGuard<'static, EventState>,
) {
    let hierarchy = match held_lock_level() {
        Some(LockLevel::ProbeDispatch) | Some(LockLevel::AgcState) => None,
        _ => Some(LockHierarchyGuard::enter(LockLevel::EventState)),
    };
    let guard = cell()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    (hierarchy, guard)
}

pub(crate) fn lock_event_state_for_read()
-> (
    Option<LockHierarchyGuard>,
    RwLockReadGuard<'static, EventState>,
) {
    let hierarchy = match held_lock_level() {
        Some(LockLevel::ProbeDispatch) | Some(LockLevel::AgcState) => None,
        _ => Some(LockHierarchyGuard::enter(LockLevel::EventState)),
    };
    let guard = cell()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    (hierarchy, guard)
}

pub(crate) fn reset_event_state() {
    let mut state = cell()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *state = EventState {
        next_equeue: 1,
        ..EventState::default()
    };
}

#[cfg(test)]
pub(crate) fn reset_event_state_for_tests() {
    reset_event_state();
}

/// Queues a kernel event under the EventState lock (coalesce on equeue/ident/filter).
pub(crate) fn queue_kernel_event(mut event: KernelQueuedEvent) {
    let (_h, mut events) = lock_event_state_for_write();
    queue_kernel_event_into(&mut events, event);
}

pub(crate) fn queue_kernel_event_into(events: &mut EventState, mut event: KernelQueuedEvent) {
    if let Some(pending) = events.pending_events.iter_mut().find(|pending| {
        pending.equeue == event.equeue
            && pending.ident == event.ident
            && pending.filter == event.filter
    }) {
        event.fflags = pending.fflags.saturating_add(1).max(event.fflags);
        event.agc_graphics_completions = pending
            .agc_graphics_completions
            .saturating_add(event.agc_graphics_completions);
        event.agc_user_interrupts = pending
            .agc_user_interrupts
            .saturating_add(event.agc_user_interrupts);
        *pending = event;
    } else {
        events.pending_events.push_back(event);
    }
}


pub(crate) fn with_event_state_mut<R>(f: impl FnOnce(&mut EventState) -> R) -> R {
    let (_h, mut events) = lock_event_state_for_write();
    f(&mut events)
}

pub(crate) fn with_event_state<R>(f: impl FnOnce(&EventState) -> R) -> R {
    let (_h, events) = lock_event_state_for_read();
    f(&events)
}

#[cfg(test)]
mod tests {
    use super::{lock_event_state_for_write, reset_event_state_for_tests};
    use crate::dispatch_locks::{LockHierarchyGuard, LockLevel};

    #[test]
    fn event_state_lock_is_usable_without_probe_dispatch() {
        reset_event_state_for_tests();
        let (_h, mut state) = lock_event_state_for_write();
        state.next_equeue = 7;
        crate::event_state::with_event_state_mut(|e| { e.kernel_equeues.insert(7); });
        assert!(state.kernel_equeues.contains(&7));
    }

    #[test]
    fn event_state_may_nest_under_probe_dispatch() {
        reset_event_state_for_tests();
        let _dispatch = LockHierarchyGuard::enter(LockLevel::ProbeDispatch);
        let (_h, mut state) = lock_event_state_for_write();
        crate::event_state::with_event_state_mut(|e| { e.kernel_equeues.insert(3); });
        assert!(state.kernel_equeues.contains(&3));
    }
}
