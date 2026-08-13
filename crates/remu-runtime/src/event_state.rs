//! Event queues and VideoOut flip retirement behind their own lock (plan C2 step 2 / B20).
//!
//! Sole owner of equeue / registration / pending-event / flip / AGC-completion queues. Call sites
//! that already hold [`LockLevel::ProbeDispatch`] may nest this lock (see
//! [`lock_event_state_for_write`]); EventState critical sections must never acquire ProbeDispatch.

#![allow(dead_code)] // with_/reset helpers exercised in tests; production still uses lock_* directly.
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
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
    /// Backend completions the runtime has drained but could not yet turn into a guest event.
    ///
    /// The backend's completion set is drained destructively, so a completion observed before its
    /// arm is recorded — or before the guest registers the equeue event it belongs to — used to be
    /// counted and thrown away, leaving the arm to wait forever. Carrying it here makes delivery
    /// level-triggered: every refresh reconsiders it until it is claimed or evicted.
    pub(crate) unclaimed_agc_completions: BTreeSet<u64>,
}

static EVENT_STATE: OnceLock<RwLock<EventState>> = OnceLock::new();

fn cell() -> &'static RwLock<EventState> {
    EVENT_STATE.get_or_init(|| {
        RwLock::new(EventState {
            next_equeue: 1,
            ..EventState::default()
        })
    })
}

/// Write lock for EventState. When ProbeDispatch is already held, nests without re-entering the
/// hierarchy (producers under the dispatch lock still publish into EventState as sole store).
pub(crate) fn lock_event_state_for_write() -> (
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

pub(crate) fn lock_event_state_for_read() -> (
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

/// Carried-completion depth, or `None` when the lock is busy.
///
/// A diagnostic peek for the import-trace reporter. Deliberately bare `try_read`: it takes no
/// hierarchy guard because it acquires nothing else and never waits, and a reporter that blocked
/// here would go silent during the wedge it exists to describe.
pub(crate) fn try_carried_agc_completion_depth() -> Option<usize> {
    cell()
        .try_read()
        .ok()
        .map(|events| events.unclaimed_agc_completions.len())
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
pub(crate) fn queue_kernel_event(event: KernelQueuedEvent) {
    let (_h, mut events) = lock_event_state_for_write();
    queue_kernel_event_into(&mut events, event);
}

/// Kernel events pushed as new entries, and those merged into an already-pending entry.
///
/// Coalescing is correct by BSD `kevent` semantics — `fflags` accumulates — but it means N backend
/// completions can surface to the guest as one returned event. A high coalesce count beside a
/// starving waiter is evidence the guest counts returned events rather than reading `fflags`.
static EQUEUE_EVENTS_INSERTED: AtomicU64 = AtomicU64::new(0);
static EQUEUE_EVENTS_COALESCED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn equeue_insertion_counts() -> (u64, u64) {
    (
        EQUEUE_EVENTS_INSERTED.load(Ordering::Relaxed),
        EQUEUE_EVENTS_COALESCED.load(Ordering::Relaxed),
    )
}

/// Pending-event total, per-equeue count for `equeue`, and outstanding AGC arms.
///
/// `None` when the lock is busy. A bare `try_read` for the same reason as
/// [`try_carried_agc_completion_depth`]: the reporter must never block on the state it describes.
pub(crate) fn try_equeue_snapshot(equeue: u64) -> Option<(usize, usize, usize)> {
    cell().try_read().ok().map(|events| {
        (
            events.pending_events.len(),
            events
                .pending_events
                .iter()
                .filter(|event| event.equeue == equeue)
                .count(),
            events.pending_agc_completions.len() + events.pending_agc_user_interrupts.len(),
        )
    })
}

pub(crate) fn queue_kernel_event_into(events: &mut EventState, mut event: KernelQueuedEvent) {
    if let Some(pending) = events.pending_events.iter_mut().find(|pending| {
        pending.equeue == event.equeue
            && pending.ident == event.ident
            && pending.filter == event.filter
    }) {
        EQUEUE_EVENTS_COALESCED.fetch_add(1, Ordering::Relaxed);
        event.fflags = pending.fflags.saturating_add(1).max(event.fflags);
        event.agc_graphics_completions = pending
            .agc_graphics_completions
            .saturating_add(event.agc_graphics_completions);
        event.agc_user_interrupts = pending
            .agc_user_interrupts
            .saturating_add(event.agc_user_interrupts);
        *pending = event;
    } else {
        EQUEUE_EVENTS_INSERTED.fetch_add(1, Ordering::Relaxed);
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
        {
            let (_h, mut state) = lock_event_state_for_write();
            state.next_equeue = 7;
        }
        crate::event_state::with_event_state_mut(|e| {
            e.kernel_equeues.insert(7);
        });
        assert!(crate::event_state::with_event_state(|state| state
            .kernel_equeues
            .contains(&7)));
    }

    #[test]
    fn event_state_may_nest_under_probe_dispatch() {
        reset_event_state_for_tests();
        let _dispatch = LockHierarchyGuard::enter(LockLevel::ProbeDispatch);
        {
            let (_h, mut state) = lock_event_state_for_write();
            state.next_equeue = 3;
        }
        crate::event_state::with_event_state_mut(|e| {
            e.kernel_equeues.insert(3);
        });
        assert!(crate::event_state::with_event_state(|state| state
            .kernel_equeues
            .contains(&3)));
    }
}
