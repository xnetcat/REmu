// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::ThreadId;

use crate::PthreadMutexAttrState;

#[derive(Debug, Default)]
pub(crate) struct GuestSyncState {
    pub(crate) rwlocks: HashMap<u64, GuestRwlockState>,
    pub(crate) conditions: HashMap<u64, GuestConditionState>,
    pub(crate) condition_waiters: HashMap<u64, GuestConditionWaiterState>,
    pub(crate) next_condition_waiter: u64,
    pub(crate) condition_notify_count: u64,
    pub(crate) semaphores: HashMap<u32, GuestSemaphoreState>,
    pub(crate) next_semaphore_handle: u32,
    pub(crate) semaphore_wait_count: u64,
    pub(crate) semaphore_signal_count: u64,
    pub(crate) event_flags: HashMap<u64, GuestEventFlagState>,
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
        let waiter = self.waiters.pop_front();
        self.clear_binding_if_idle();
        waiter
    }

    pub(crate) fn release_all(&mut self) -> VecDeque<u64> {
        self.bound_mutex = None;
        std::mem::take(&mut self.waiters)
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
    pub(crate) fn register_condition_waiter(&mut self, condition: u64, mutex: u64) -> Option<u64> {
        if self
            .conditions
            .get(&condition)
            .and_then(|condition| condition.bound_mutex)
            .is_some_and(|bound| bound != mutex)
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
            self.conditions
                .entry(condition)
                .or_default()
                .register_waiter(waiter, mutex),
            "condition mutex binding was validated under the synchronization lock"
        );
        self.condition_waiters
            .insert(waiter, GuestConditionWaiterState { released: false });
        Some(waiter)
    }

    pub(crate) fn condition_waiter_released(&self, waiter: u64) -> bool {
        self.condition_waiters
            .get(&waiter)
            .is_some_and(|waiter| waiter.released)
    }

    pub(crate) fn finish_condition_wait(&mut self, condition: u64, waiter: u64, timed_out: bool) {
        if timed_out {
            self.conditions
                .get_mut(&condition)
                .expect("a pending condition waiter prevents destruction")
                .cancel_wait(waiter);
        }
        let waiter_state = self
            .condition_waiters
            .remove(&waiter)
            .expect("registered condition waiter survives condition destruction");
        assert_eq!(
            waiter_state.released, !timed_out,
            "timeout and notification have exactly one winner"
        );
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
    pub(crate) storage_address: u64,
    /// Opaque object installed in the guest's `ScePthreadMutex` storage by the init ABI.
    ///
    /// The bundled libc writes bookkeeping fields in this object immediately after
    /// `scePthreadMutexInit` succeeds, even though lock ownership itself is modeled by REmu.
    /// Synthetic unit-test addresses are deliberately allowed to leave this zero.
    pub(crate) guest_object: u64,
}

#[derive(Debug)]
pub(crate) struct GuestRwlockReaderState {
    pub(crate) guest_thread: u64,
    pub(crate) recursion: u32,
}

#[derive(Debug, Default)]
pub(crate) struct GuestRwlockState {
    pub(crate) readers: HashMap<ThreadId, GuestRwlockReaderState>,
    pub(crate) reader_total: u32,
    pub(crate) writer: Option<ThreadId>,
    pub(crate) writer_guest_thread: u64,
    pub(crate) waiting_writers: u32,
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

#[derive(Debug)]
pub(crate) struct GuestEventFlagState {
    pub(crate) name: String,
    pub(crate) attributes: u32,
    pub(crate) bits: u64,
    pub(crate) waiters: u32,
    pub(crate) cancellation_generation: u64,
}

pub(crate) static GUEST_SYNC: OnceLock<(Mutex<GuestSyncState>, Condvar)> = OnceLock::new();

pub(super) static EVENT_FLAG_WAIT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(super) static EVENT_FLAG_SET_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(super) static MUTEX_CONTENTION_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(super) const GUEST_MUTEX_SHARD_COUNT: usize = 64;

pub(super) static GUEST_MUTEX_SHARDS: OnceLock<
    [Mutex<HashMap<u64, Arc<GuestMutexEntry>>>; GUEST_MUTEX_SHARD_COUNT],
> = OnceLock::new();

pub(crate) fn guest_sync() -> &'static (Mutex<GuestSyncState>, Condvar) {
    GUEST_SYNC.get_or_init(|| (Mutex::new(GuestSyncState::default()), Condvar::new()))
}

pub(super) fn guest_mutex_shards()
-> &'static [Mutex<HashMap<u64, Arc<GuestMutexEntry>>>; GUEST_MUTEX_SHARD_COUNT] {
    GUEST_MUTEX_SHARDS.get_or_init(|| std::array::from_fn(|_| Mutex::new(HashMap::new())))
}

pub(super) fn guest_mutex_shard(address: u64) -> usize {
    usize::try_from(address >> 4).unwrap_or_default() & (GUEST_MUTEX_SHARD_COUNT - 1)
}
