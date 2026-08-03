// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::sync::Arc;

use super::state::{GuestRwlockEntry, GuestRwlockShard, guest_rwlock_shard, guest_rwlock_shards};
use crate::{
    assert_wait_is_off_the_dispatch_lock, current_guest_thread_handle, current_guest_thread_name,
    stall_report_interval,
};

fn rwlock_shard(address: u64) -> &'static GuestRwlockShard {
    &guest_rwlock_shards()[guest_rwlock_shard(address)]
}

fn lookup_or_insert_rwlock(address: u64) -> Arc<GuestRwlockEntry> {
    let shard = rwlock_shard(address);
    let mut locks = shard
        .locks
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    locks
        .entry(address)
        .or_insert_with(|| Arc::new(GuestRwlockEntry::default()))
        .clone()
}

fn lookup_rwlock(address: u64) -> Option<Arc<GuestRwlockEntry>> {
    let shard = rwlock_shard(address);
    let locks = shard
        .locks
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    locks.get(&address).cloned()
}

pub(crate) fn guest_rwlock_init(address: u64) -> u64 {
    if address == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    let shard = rwlock_shard(address);
    let mut locks = shard
        .locks
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if locks.contains_key(&address) {
        return u64::from(libc::EBUSY.cast_unsigned());
    }
    locks.insert(address, Arc::new(GuestRwlockEntry::default()));
    0
}

pub(crate) fn guest_rwlock_destroy(address: u64) -> u64 {
    if address == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    let shard = rwlock_shard(address);
    let mut locks = shard
        .locks
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(entry) = locks.get(&address).cloned() else {
        return u64::from(libc::EINVAL.cast_unsigned());
    };
    let lock = entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if lock.writer.is_some()
        || lock.reader_total != 0
        || lock.waiting_readers != 0
        || lock.waiting_writers != 0
    {
        return u64::from(libc::EBUSY.cast_unsigned());
    }
    drop(lock);
    locks.remove(&address);
    0
}

pub(crate) fn guest_rwlock_lock(address: u64, write: bool, try_only: bool) -> u64 {
    if address == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    let entry = lookup_or_insert_rwlock(address);
    let current = std::thread::current().id();
    let mut lock = entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    if write {
        if lock.writer.as_ref() == Some(&current) {
            return u64::from(libc::EDEADLK.cast_unsigned());
        }
        if lock.writer.is_none() && lock.reader_total == 0 {
            lock.writer = Some(current);
            return 0;
        }
        if try_only {
            return u64::from(libc::EBUSY.cast_unsigned());
        }
        lock.waiting_writers += 1;

        let mut stall_reported = false;
        loop {
            if lock.writer.is_none() && lock.reader_total == 0 {
                lock.waiting_writers -= 1;
                lock.writer = Some(current);
                return 0;
            }
            assert_wait_is_off_the_dispatch_lock("scePthreadRwlockWrlock");
            if let Some(report_after) = stall_report_interval() {
                let (next_lock, wait_result) = entry
                    .wake
                    .wait_timeout(lock, report_after)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                lock = next_lock;
                if wait_result.timed_out() && !stall_reported {
                    stall_reported = true;
                    eprintln!(
                        "[REMu][STALL] rwlock write wait exceeded {}s thread={:?} handle={:#x} rwlock={address:#x} writer={:?} readers={} reader_threads={} waiting_readers={} waiting_writers={}",
                        report_after.as_secs(),
                        current_guest_thread_name(),
                        current_guest_thread_handle(),
                        lock.writer,
                        lock.reader_total,
                        lock.readers.len(),
                        lock.waiting_readers,
                        lock.waiting_writers,
                    );
                }
            } else {
                lock = entry
                    .wake
                    .wait(lock)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        }
    }

    if lock.writer.is_none() {
        *lock.readers.entry(current).or_default() += 1;
        lock.reader_total += 1;
        return 0;
    }
    if try_only {
        return u64::from(libc::EBUSY.cast_unsigned());
    }
    lock.waiting_readers += 1;

    let mut stall_reported = false;
    loop {
        // Gen5 rwlocks are reader-first: only an active writer excludes a reader. Making queued
        // writers exclusive here can deadlock a reader that recursively acquires the same lock.
        if lock.writer.is_none() {
            lock.waiting_readers -= 1;
            *lock.readers.entry(current).or_default() += 1;
            lock.reader_total += 1;
            return 0;
        }
        assert_wait_is_off_the_dispatch_lock("scePthreadRwlockRdlock");
        if let Some(report_after) = stall_report_interval() {
            let (next_lock, wait_result) = entry
                .wake
                .wait_timeout(lock, report_after)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            lock = next_lock;
            if wait_result.timed_out() && !stall_reported {
                stall_reported = true;
                eprintln!(
                    "[REMu][STALL] rwlock read wait exceeded {}s thread={:?} handle={:#x} rwlock={address:#x} writer={:?} readers={} reader_threads={} waiting_readers={} waiting_writers={}",
                    report_after.as_secs(),
                    current_guest_thread_name(),
                    current_guest_thread_handle(),
                    lock.writer,
                    lock.reader_total,
                    lock.readers.len(),
                    lock.waiting_readers,
                    lock.waiting_writers,
                );
            }
        } else {
            lock = entry
                .wake
                .wait(lock)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

pub(crate) fn guest_rwlock_unlock(address: u64) -> u64 {
    let Some(entry) = lookup_rwlock(address) else {
        return u64::from(libc::EINVAL.cast_unsigned());
    };
    let current = std::thread::current().id();
    let mut lock = entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if lock.writer.as_ref() == Some(&current) {
        lock.writer = None;
        if lock.waiting_readers != 0 || lock.waiting_writers != 0 {
            entry.wake.notify_all();
        }
        return 0;
    }
    let Some(recursion) = lock.readers.get_mut(&current) else {
        return u64::from(libc::EPERM.cast_unsigned());
    };
    *recursion -= 1;
    let drop_reader = *recursion == 0;
    lock.reader_total -= 1;
    if drop_reader {
        lock.readers.remove(&current);
    }
    if lock.reader_total == 0 && lock.waiting_writers != 0 {
        entry.wake.notify_all();
    }
    0
}

#[cfg(test)]
pub(crate) fn guest_rwlock_waiter_counts(address: u64) -> Option<(u32, u32)> {
    let entry = lookup_rwlock(address)?;
    let lock = entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Some((lock.waiting_readers, lock.waiting_writers))
}
