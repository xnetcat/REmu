//! Event queues and VideoOut flip retirement behind their own lock (plan C2 step 2 / B20).
//!
//! Sole owner of equeue / registration / pending-event / flip / AGC-completion queues. Call sites
//! that already hold [`LockLevel::ProbeDispatch`] may nest this lock (see
//! [`lock_event_state_for_write`]); EventState critical sections must never acquire ProbeDispatch.

#![allow(dead_code)] // with_/reset helpers exercised in tests; production still uses lock_* directly.
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Duration;

use crate::dispatch_locks::{LockHierarchyGuard, LockLevel, held_lock_level};
use crate::{
    AgcCompletionTelemetry, CompletedVideoOutFlip, KernelQueuedEvent, PendingVideoOutFlip,
};

#[derive(Debug, Default)]
pub(crate) struct EqueueWake {
    generation: Mutex<u64>,
    changed: Condvar,
}

impl EqueueWake {
    pub(crate) fn snapshot(&self) -> u64 {
        *self
            .generation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(crate) fn signal(&self) {
        let mut generation = self
            .generation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *generation = generation.wrapping_add(1);
        drop(generation);
        self.changed.notify_all();
    }

    /// Waits until a producer advances this equeue's generation. `None` parks indefinitely;
    /// `Some` returns false only when that exact deadline elapses without a signal.
    pub(crate) fn wait_for_change(&self, observed: u64, timeout: Option<Duration>) -> bool {
        let generation = self
            .generation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *generation != observed {
            return true;
        }
        if let Some(timeout) = timeout {
            let (generation, result) = self
                .changed
                .wait_timeout_while(generation, timeout, |generation| *generation == observed)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            !result.timed_out() || *generation != observed
        } else {
            drop(
                self.changed
                    .wait_while(generation, |generation| *generation == observed)
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            );
            true
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct EventState {
    pub(crate) next_equeue: u64,
    pub(crate) kernel_equeues: BTreeMap<u64, Arc<EqueueWake>>,
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
    pub(crate) agc_completion_telemetry: AgcCompletionTelemetry,
    pub(crate) video_out_flip_events: BTreeMap<u64, u64>,
    pub(crate) video_out_vblank_events: BTreeMap<u64, u64>,
    pub(crate) completed_video_out_flip_count: u64,
    pub(crate) last_completed_video_out_flip: Option<CompletedVideoOutFlip>,
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
    let wakes = state.kernel_equeues.values().cloned().collect::<Vec<_>>();
    *state = EventState {
        next_equeue: 1,
        ..EventState::default()
    };
    drop(state);
    for wake in wakes {
        crate::signal_kernel_equeue_wake(&wake);
    }
}

#[cfg(test)]
pub(crate) fn reset_event_state_for_tests() {
    reset_event_state();
}

/// Queues a kernel event under the EventState lock (coalesce on equeue/ident/filter).
pub(crate) fn queue_kernel_event(event: KernelQueuedEvent) {
    let equeue = event.equeue;
    let (_h, mut events) = lock_event_state_for_write();
    queue_kernel_event_into(&mut events, event);
    let wake = events.kernel_equeues.get(&equeue).cloned();
    drop(events);
    if let Some(wake) = wake {
        crate::signal_kernel_equeue_wake(&wake);
    }
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

/// What a blocked waiter's equeue currently holds.
pub(crate) struct EqueueSnapshot {
    pub(crate) events_total: usize,
    pub(crate) events_for_equeue: usize,
    pub(crate) arms: usize,
    /// Event filters registered on this equeue. An equeue with no registrations is a listener
    /// parked on a source that was never armed — idle by construction, not starving.
    pub(crate) filters: Vec<i16>,
}

/// Snapshot of `equeue`, or `None` when the lock is busy.
///
/// A bare `try_read` for the same reason as [`try_carried_agc_completion_depth`]: the reporter must
/// never block on the state it describes.
pub(crate) fn try_equeue_snapshot(equeue: u64) -> Option<EqueueSnapshot> {
    cell().try_read().ok().map(|events| EqueueSnapshot {
        events_total: events.pending_events.len(),
        events_for_equeue: events
            .pending_events
            .iter()
            .filter(|event| event.equeue == equeue)
            .count(),
        arms: events.pending_agc_completions.len() + events.pending_agc_user_interrupts.len(),
        filters: events
            .kernel_event_registrations
            .keys()
            .filter_map(|&(registered, filter, _)| (registered == equeue).then_some(filter))
            .collect(),
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
    use super::{EqueueWake, lock_event_state_for_write, reset_event_state_for_tests};
    use crate::KernelQueuedEvent;
    use crate::dispatch_locks::{LockHierarchyGuard, LockLevel};
    use std::sync::Arc;

    #[test]
    fn event_state_lock_is_usable_without_probe_dispatch() {
        reset_event_state_for_tests();
        {
            let (_h, mut state) = lock_event_state_for_write();
            state.next_equeue = 7;
        }
        crate::event_state::with_event_state_mut(|e| {
            e.kernel_equeues.insert(7, Arc::new(EqueueWake::default()));
        });
        assert!(crate::event_state::with_event_state(|state| state
            .kernel_equeues
            .contains_key(&7)));
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
            e.kernel_equeues.insert(3, Arc::new(EqueueWake::default()));
        });
        assert!(crate::event_state::with_event_state(|state| state
            .kernel_equeues
            .contains_key(&3)));
    }

    #[test]
    fn equeue_generations_are_isolated_and_pre_wait_signals_are_observed() {
        let first = EqueueWake::default();
        let second = EqueueWake::default();
        let observed_first = first.snapshot();
        let observed_second = second.snapshot();
        first.signal();
        assert!(first.wait_for_change(observed_first, Some(std::time::Duration::ZERO)));
        assert!(!second.wait_for_change(observed_second, Some(std::time::Duration::ZERO)));
    }

    #[test]
    fn event_publication_signals_only_its_target_equeue() {
        reset_event_state_for_tests();
        let first = Arc::new(EqueueWake::default());
        let second = Arc::new(EqueueWake::default());
        let observed_first = first.snapshot();
        let observed_second = second.snapshot();
        crate::event_state::with_event_state_mut(|state| {
            state.kernel_equeues.insert(1, Arc::clone(&first));
            state.kernel_equeues.insert(2, Arc::clone(&second));
        });

        crate::event_state::queue_kernel_event(KernelQueuedEvent {
            equeue: 1,
            ident: 7,
            filter: 3,
            flags: 0,
            fflags: 0,
            data: 0,
            user_data: 0,
            agc_graphics_completions: 0,
            agc_user_interrupts: 0,
        });

        assert!(first.wait_for_change(observed_first, Some(std::time::Duration::ZERO)));
        assert!(!second.wait_for_change(observed_second, Some(std::time::Duration::ZERO)));
        reset_event_state_for_tests();
    }
}
