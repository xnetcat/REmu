// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::{current_guest_thread_handle, current_guest_thread_name, stall_report_interval};

use super::{GuestRwlockReaderState, GuestRwlockState, guest_sync};

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
    let guest_thread = current_guest_thread_handle();
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
            state
                .rwlocks
                .get_mut(&address)
                .expect("rwlock was initialized")
                .writer_guest_thread = guest_thread;
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
        let mut reported_stall = false;
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
                lock.writer_guest_thread = guest_thread;
                return 0;
            }
            if let Some(report_after) = stall_report_interval() {
                let (next_state, wait_result) = sync
                    .1
                    .wait_timeout(state, report_after)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                state = next_state;
                if wait_result.timed_out() && !reported_stall {
                    reported_stall = true;
                    let lock = state
                        .rwlocks
                        .get(&address)
                        .expect("a waiting writer prevents rwlock destruction");
                    eprintln!(
                        "[REMu][STALL] rwlock write lock exceeded {}s thread={:?} \
                         handle={guest_thread:#x} rwlock={address:#x} writer={:?} \
                         writer_handle={:#x} readers={:?} reader_total={} waiting_writers={}",
                        report_after.as_secs(),
                        current_guest_thread_name(),
                        lock.writer,
                        lock.writer_guest_thread,
                        lock.readers,
                        lock.reader_total,
                        lock.waiting_writers,
                    );
                }
            } else {
                state = sync
                    .1
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        }
    }

    let mut reported_stall = false;
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
            let reader = lock
                .readers
                .entry(current)
                .or_insert(GuestRwlockReaderState {
                    guest_thread,
                    recursion: 0,
                });
            reader.guest_thread = guest_thread;
            reader.recursion += 1;
            lock.reader_total += 1;
            return 0;
        }
        if try_only {
            return u64::from(libc::EBUSY.cast_unsigned());
        }
        if let Some(report_after) = stall_report_interval() {
            let (next_state, wait_result) = sync
                .1
                .wait_timeout(state, report_after)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state = next_state;
            if wait_result.timed_out() && !reported_stall {
                reported_stall = true;
                let lock = state
                    .rwlocks
                    .get(&address)
                    .expect("an active writer prevents rwlock destruction");
                eprintln!(
                    "[REMu][STALL] rwlock read lock exceeded {}s thread={:?} \
                     handle={guest_thread:#x} rwlock={address:#x} writer={:?} \
                     writer_handle={:#x} readers={:?} reader_total={} waiting_writers={}",
                    report_after.as_secs(),
                    current_guest_thread_name(),
                    lock.writer,
                    lock.writer_guest_thread,
                    lock.readers,
                    lock.reader_total,
                    lock.waiting_writers,
                );
            }
        } else {
            state = sync
                .1
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
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
        lock.writer_guest_thread = 0;
        sync.1.notify_all();
        return 0;
    }
    let Some(reader) = lock.readers.get_mut(&current) else {
        return u64::from(libc::EPERM.cast_unsigned());
    };
    reader.recursion -= 1;
    lock.reader_total -= 1;
    if reader.recursion == 0 {
        lock.readers.remove(&current);
    }
    if lock.reader_total == 0 {
        sync.1.notify_all();
    }
    0
}
