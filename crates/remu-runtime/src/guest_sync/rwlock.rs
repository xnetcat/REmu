// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{GuestRwlockState, guest_sync};
use crate::assert_wait_is_off_the_dispatch_lock;

pub(crate) fn guest_rwlock_init(address: u64) -> u64 {
    if address == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.rwlocks.contains_key(&address) {
        return u64::from(libc::EBUSY.cast_unsigned());
    }
    state.rwlocks.insert(address, GuestRwlockState::default());
    0
}

pub(crate) fn guest_rwlock_destroy(address: u64) -> u64 {
    if address == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match state.rwlocks.get(&address) {
        Some(lock)
            if lock.writer.is_some() || lock.reader_total != 0 || lock.waiting_writers != 0 =>
        {
            u64::from(libc::EBUSY.cast_unsigned())
        }
        Some(_) => {
            state.rwlocks.remove(&address);
            0
        }
        None => u64::from(libc::EINVAL.cast_unsigned()),
    }
}

pub(crate) fn guest_rwlock_lock(address: u64, write: bool, try_only: bool) -> u64 {
    if address == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    let sync = guest_sync();
    let current = std::thread::current().id();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.rwlocks.entry(address).or_default();

    if write {
        if state
            .rwlocks
            .get(&address)
            .is_some_and(|lock| lock.writer.as_ref() == Some(&current))
        {
            return u64::from(libc::EDEADLK.cast_unsigned());
        }
        let acquired = state
            .rwlocks
            .get(&address)
            .is_some_and(|lock| lock.writer.is_none() && lock.reader_total == 0);
        if acquired {
            state
                .rwlocks
                .get_mut(&address)
                .expect("rwlock was initialized")
                .writer = Some(current);
            return 0;
        }
        if try_only {
            return u64::from(libc::EBUSY.cast_unsigned());
        }
        state
            .rwlocks
            .get_mut(&address)
            .expect("rwlock was initialized")
            .waiting_writers += 1;
        loop {
            let acquired = state
                .rwlocks
                .get(&address)
                .is_some_and(|lock| lock.writer.is_none() && lock.reader_total == 0);
            if acquired {
                let lock = state
                    .rwlocks
                    .get_mut(&address)
                    .expect("a waiting writer prevents rwlock destruction");
                lock.waiting_writers -= 1;
                lock.writer = Some(current);
                return 0;
            }
            assert_wait_is_off_the_dispatch_lock("scePthreadRwlockWrlock");
            state = sync
                .1
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    loop {
        // Gen5 rwlocks are reader-first: only an active writer excludes a reader. Making queued
        // writers exclusive here can deadlock a reader that recursively acquires the same lock.
        let acquired = state
            .rwlocks
            .get(&address)
            .is_some_and(|lock| lock.writer.is_none());
        if acquired {
            let lock = state
                .rwlocks
                .get_mut(&address)
                .expect("rwlock was initialized");
            *lock.readers.entry(current).or_default() += 1;
            lock.reader_total += 1;
            return 0;
        }
        if try_only {
            return u64::from(libc::EBUSY.cast_unsigned());
        }
        assert_wait_is_off_the_dispatch_lock("scePthreadRwlockRdlock");
        state = sync
            .1
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

pub(crate) fn guest_rwlock_unlock(address: u64) -> u64 {
    let sync = guest_sync();
    let current = std::thread::current().id();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(lock) = state.rwlocks.get_mut(&address) else {
        return u64::from(libc::EINVAL.cast_unsigned());
    };
    if lock.writer.as_ref() == Some(&current) {
        lock.writer = None;
        sync.1.notify_all();
        return 0;
    }
    let Some(recursion) = lock.readers.get_mut(&current) else {
        return u64::from(libc::EPERM.cast_unsigned());
    };
    *recursion -= 1;
    lock.reader_total -= 1;
    if *recursion == 0 {
        lock.readers.remove(&current);
    }
    if lock.reader_total == 0 {
        sync.1.notify_all();
    }
    0
}
