// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::ThreadId;

use crate::PthreadMutexAttrState;

#[derive(Debug, Default)]
pub(crate) struct GuestSyncState {
    pub(crate) conditions: HashMap<u64, Arc<GuestConditionEntry>>,
    pub(crate) condition_waiters: HashMap<u64, GuestConditionWaiterState>,
    pub(crate) next_condition_waiter: u64,
    pub(crate) condition_notify_count: u64,
    pub(crate) semaphores: HashMap<u32, Arc<GuestSemaphoreEntry>>,
    pub(crate) next_semaphore_handle: u32,
    pub(crate) semaphore_wait_count: u64,
    pub(crate) semaphore_signal_count: u64,
    pub(crate) event_flags: HashMap<u64, Arc<GuestEventFlagEntry>>,
    pub(crate) next_event_flag_handle: u64,
}

#[derive(Debug, Default)]
pub(crate) struct GuestConditionState {
    pub(crate) waiters: VecDeque<u64>,
    pub(crate) bound_mutex: Option<u64>,
    /// Signals delivered to this condition. A stalled waiter on a condition that has never been
    /// signaled is a missing producer; one on a signaled condition is a lost wakeup.
    pub(crate) notifies: u64,
    /// Timed waits that expired without being signaled. A large count with no signals is a poll
    /// whose predicate never becomes true.
    pub(crate) timeouts: u64,
    /// Tickets selected by signal/broadcast; checked under this object's mutex (plan C3 wake).
    pub(crate) released: HashSet<u64>,
}

impl GuestConditionState {
    pub(crate) fn register_waiter(&mut self, waiter: u64, mutex: u64) -> bool {
        if self.bound_mutex.is_some_and(|bound| bound != mutex) {
            return false;
        }
        self.bound_mutex = Some(mutex);
        self.waiters.push_back(waiter);
        true
    }

    pub(crate) fn release_one(&mut self) -> Option<u64> {
        let waiter = self.waiters.pop_front()?;
        self.released.insert(waiter);
        self.clear_binding_if_idle();
        Some(waiter)
    }

    pub(crate) fn release_all(&mut self) -> VecDeque<u64> {
        self.bound_mutex = None;
        let waiters = std::mem::take(&mut self.waiters);
        self.released.extend(waiters.iter().copied());
        waiters
    }

    pub(crate) fn waiter_released(&self, waiter: u64) -> bool {
        self.released.contains(&waiter)
    }

    pub(crate) fn take_release(&mut self, waiter: u64) -> bool {
        self.released.remove(&waiter)
    }

    pub(crate) fn cancel_wait(&mut self, waiter: u64) {
        self.waiters.retain(|candidate| *candidate != waiter);
        self.clear_binding_if_idle();
    }

    pub(crate) fn is_idle(&self) -> bool {
        self.waiters.is_empty()
    }

    pub(crate) fn clear_binding_if_idle(&mut self) {
        if self.waiters.is_empty() {
            self.bound_mutex = None;
        }
    }
}

#[derive(Debug)]
pub(crate) struct GuestConditionWaiterState {
    pub(crate) released: bool,
}

impl GuestSyncState {
    pub(crate) fn condition_by_handle(&self, handle: u64) -> Option<Arc<GuestConditionEntry>> {
        self.conditions.get(&handle).cloned()
    }

    pub(crate) fn insert_condition(
        &mut self,
        handle: u64,
        storage_address: u64,
    ) -> Arc<GuestConditionEntry> {
        let entry = GuestConditionEntry::new(handle, storage_address);
        self.conditions.insert(handle, entry.clone());
        entry
    }

    pub(crate) fn register_condition_waiter(
        &mut self,
        condition_handle: u64,
        mutex_identity: u64,
    ) -> Option<u64> {
        let entry = self.condition_by_handle(condition_handle)?;
        let mut condition_state = entry
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if condition_state
            .bound_mutex
            .is_some_and(|bound| bound != mutex_identity)
        {
            return None;
        }
        let waiter = loop {
            self.next_condition_waiter = self.next_condition_waiter.wrapping_add(1).max(1);
            if !self
                .condition_waiters
                .contains_key(&self.next_condition_waiter)
            {
                break self.next_condition_waiter;
            }
        };
        assert!(
            condition_state.register_waiter(waiter, mutex_identity),
            "condition mutex binding was validated under the synchronization lock"
        );
        drop(condition_state);
        self.condition_waiters
            .insert(waiter, GuestConditionWaiterState { released: false });
        Some(waiter)
    }

    #[allow(dead_code)] // Query helper for diagnostics / tests beside finish_condition_wait.
    pub(crate) fn condition_waiter_released(&self, waiter: u64) -> bool {
        self.condition_waiters
            .get(&waiter)
            .is_some_and(|waiter| waiter.released)
    }

    pub(crate) fn finish_condition_wait(&mut self, condition: u64, waiter: u64, timed_out: bool) {
        let released = if let Some(entry) = self.conditions.get(&condition) {
            let mut condition_state = entry
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if timed_out {
                condition_state.cancel_wait(waiter);
            }
            condition_state.take_release(waiter)
        } else {
            false
        };
        let waiter_state = self
            .condition_waiters
            .remove(&waiter)
            .expect("registered condition waiter survives condition destruction");
        assert_eq!(
            released || waiter_state.released,
            !timed_out,
            "timeout and notification have exactly one winner"
        );
        let _ = waiter_state;
    }
}

#[derive(Debug)]
pub(crate) struct GuestMutexState {
    pub(crate) owner: Option<ThreadId>,
    pub(crate) owner_guest_thread: u64,
    pub(crate) recursion: u32,
    pub(crate) mutex_type: i32,
    pub(crate) waiters: u32,
}

#[derive(Debug)]
pub(crate) struct GuestMutexEntry {
    pub(crate) state: Mutex<GuestMutexState>,
    pub(crate) wake: Condvar,
    /// Cleared while holding `state` before this generation is removed from its shard.
    ///
    /// Thread-local lookup caches may keep the `Arc` alive after destruction, so every operation
    /// rechecks this bit under `state` before observing or changing ownership.
    pub(crate) active: AtomicBool,
    /// Slot that originally published this control block (zeroed on destroy).
    pub(crate) storage_address: u64,
    /// Control-block pointer written into every `ScePthreadMutex` slot for this object.
    ///
    /// This is the SDK identity (`pthread_mutex_t` is a pointer typedef). Registry shards are
    /// keyed by this value. Libc may write bookkeeping fields into the object after init.
    pub(crate) guest_object: u64,
}

#[derive(Debug, Default)]
pub(crate) struct GuestRwlockState {
    pub(crate) readers: HashMap<ThreadId, u32>,
    pub(crate) reader_total: u32,
    pub(crate) writer: Option<ThreadId>,
    pub(crate) waiting_readers: u32,
    pub(crate) waiting_writers: u32,
}

#[derive(Debug, Default)]
pub(crate) struct GuestRwlockShard {
    pub(crate) locks: Mutex<HashMap<u64, Arc<GuestRwlockEntry>>>,
}

impl Default for GuestMutexState {
    fn default() -> Self {
        let attributes = PthreadMutexAttrState::default();
        Self {
            owner: None,
            owner_guest_thread: 0,
            recursion: 0,
            mutex_type: attributes.mutex_type,
            waiters: 0,
        }
    }
}

#[derive(Debug)]
pub(crate) struct GuestSemaphoreState {
    pub(crate) count: i32,
    pub(crate) maximum: i32,
}

/// Per-object semaphore entry (plan C3): own mutex + condvar, registry only for lookup.
#[derive(Debug)]
pub(crate) struct GuestSemaphoreEntry {
    pub(crate) state: Mutex<GuestSemaphoreState>,
    pub(crate) wake: Condvar,
}

#[derive(Debug)]
pub(crate) struct GuestEventFlagState {
    pub(crate) name: String,
    pub(crate) attributes: u32,
    pub(crate) bits: u64,
    pub(crate) waiters: u32,
    pub(crate) cancellation_generation: u64,
}

/// Per-object event-flag entry (plan C3 / B16): own mutex + condvar; registry is lookup only.
#[derive(Debug)]
pub(crate) struct GuestEventFlagEntry {
    pub(crate) state: Mutex<GuestEventFlagState>,
    pub(crate) wake: Condvar,
}

/// Per-object condition entry (plan C3 / B16): FIFO-ticket logic stays in [`GuestConditionState`];
/// only the host wake is per-object so unrelated waiters are not broadcast-woken.
#[derive(Debug)]
pub(crate) struct GuestConditionEntry {
    pub(crate) state: Mutex<GuestConditionState>,
    pub(crate) wake: Condvar,
    pub(crate) guest_object: u64,
    pub(crate) storage_address: u64,
}

impl GuestConditionEntry {
    pub(crate) fn new(guest_object: u64, storage_address: u64) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(GuestConditionState::default()),
            wake: Condvar::new(),
            guest_object,
            storage_address,
        })
    }
}

/// Per-object rwlock entry (plan C3 / B16): shard map is a registry keyed by control-block handle.
#[derive(Debug)]
pub(crate) struct GuestRwlockEntry {
    pub(crate) state: Mutex<GuestRwlockState>,
    pub(crate) wake: Condvar,
    pub(crate) guest_object: u64,
    pub(crate) storage_address: u64,
}

impl GuestRwlockEntry {
    pub(crate) fn new(guest_object: u64, storage_address: u64) -> Self {
        Self {
            state: Mutex::new(GuestRwlockState::default()),
            wake: Condvar::new(),
            guest_object,
            storage_address,
        }
    }
}

pub(crate) static GUEST_SYNC: OnceLock<(Mutex<GuestSyncState>, Condvar)> = OnceLock::new();

pub(super) static EVENT_FLAG_WAIT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(super) static EVENT_FLAG_SET_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(super) static MUTEX_CONTENTION_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(super) const GUEST_MUTEX_SHARD_COUNT: usize = 64;

pub(super) const GUEST_RWLOCK_SHARD_COUNT: usize = 64;

pub(super) static GUEST_MUTEX_SHARDS: OnceLock<
    [Mutex<HashMap<u64, Arc<GuestMutexEntry>>>; GUEST_MUTEX_SHARD_COUNT],
> = OnceLock::new();

pub(super) static GUEST_RWLOCK_SHARDS: OnceLock<[GuestRwlockShard; GUEST_RWLOCK_SHARD_COUNT]> =
    OnceLock::new();

pub(crate) fn guest_sync() -> &'static (Mutex<GuestSyncState>, Condvar) {
    GUEST_SYNC.get_or_init(|| (Mutex::new(GuestSyncState::default()), Condvar::new()))
}

pub(super) fn guest_mutex_shards()
-> &'static [Mutex<HashMap<u64, Arc<GuestMutexEntry>>>; GUEST_MUTEX_SHARD_COUNT] {
    GUEST_MUTEX_SHARDS.get_or_init(|| std::array::from_fn(|_| Mutex::new(HashMap::new())))
}

pub(super) fn guest_mutex_shard(handle: u64) -> usize {
    // Multiplicative hash (Knuth/Fibonacci) over the control-block handle — SDK identity.
    let hash = handle.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    usize::try_from(hash >> (64 - 6)).unwrap_or_default()
}

pub(crate) fn guest_rwlock_shards() -> &'static [GuestRwlockShard; GUEST_RWLOCK_SHARD_COUNT] {
    GUEST_RWLOCK_SHARDS.get_or_init(|| std::array::from_fn(|_| GuestRwlockShard::default()))
}

pub(super) fn guest_rwlock_shard(address: u64) -> usize {
    let hash = address.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    usize::try_from(hash >> (64 - 6)).unwrap_or_default()
}
