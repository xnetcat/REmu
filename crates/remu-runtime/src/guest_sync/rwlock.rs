// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::sync::Arc;

use super::identity::{
    GUEST_PTHREAD_OBJECT_SIZE, PthreadSlotValue, classify_pthread_slot_value, read_pthread_slot,
};
use super::state::{GuestRwlockEntry, GuestRwlockShard, guest_rwlock_shard, guest_rwlock_shards};
use crate::{
    SCE_KERNEL_ERROR_TRY_AGAIN, assert_wait_is_off_the_dispatch_lock, current_guest_thread_handle,
    current_guest_thread_name, guest_calloc, guest_free, stall_report_interval, write_guest_u64,
};

fn rwlock_shard(handle: u64) -> &'static GuestRwlockShard {
    &guest_rwlock_shards()[guest_rwlock_shard(handle)]
}

fn lookup_rwlock_by_handle(handle: u64) -> Option<Arc<GuestRwlockEntry>> {
    if handle == 0 {
        return None;
    }
    let shard = rwlock_shard(handle);
    let locks = shard
        .locks
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    locks.get(&handle).cloned()
}

fn resolve_rwlock(address: u64, allow_static_init: bool) -> Option<Arc<GuestRwlockEntry>> {
    if address == 0 {
        return None;
    }
    if let Some(entry) = lookup_rwlock_by_handle(address) {
        return Some(entry);
    }
    let slot_value = read_pthread_slot(address)?;
    match classify_pthread_slot_value(slot_value) {
        PthreadSlotValue::ControlBlock(handle) => lookup_rwlock_by_handle(handle),
        PthreadSlotValue::StaticDefault if allow_static_init => materialize_static_rwlock(address),
        PthreadSlotValue::StaticDefault | PthreadSlotValue::StaticAdaptiveMutex => None,
    }
}

fn materialize_static_rwlock(slot: u64) -> Option<Arc<GuestRwlockEntry>> {
    // Re-check under a process-wide gate so two first-lock threads share one object.
    let sync = super::guest_sync();
    let _init_gate = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(entry) = lookup_rwlock_by_handle(slot) {
        return Some(entry);
    }
    let slot_value = read_pthread_slot(slot)?;
    match classify_pthread_slot_value(slot_value) {
        PthreadSlotValue::ControlBlock(handle) => lookup_rwlock_by_handle(handle),
        PthreadSlotValue::StaticDefault => {
            let guest_object = guest_calloc(1, GUEST_PTHREAD_OBJECT_SIZE);
            if guest_object == 0 {
                return None;
            }
            let entry = Arc::new(GuestRwlockEntry::new(guest_object, slot));
            let shard = rwlock_shard(guest_object);
            let mut locks = shard
                .locks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            locks.insert(guest_object, entry.clone());
            drop(locks);
            write_guest_u64(slot, guest_object);
            Some(entry)
        }
        PthreadSlotValue::StaticAdaptiveMutex => {
            unreachable!("rwlock slots do not use adaptive init")
        }
    }
}

pub(crate) fn guest_rwlock_init(address: u64) -> u64 {
    if address == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    // Init takes a slot pointer, never a bare control-block handle.
    if lookup_rwlock_by_handle(address).is_some() {
        return u64::from(libc::EBUSY.cast_unsigned());
    }
    let Some(slot_value) = read_pthread_slot(address) else {
        return u64::from(libc::EINVAL.cast_unsigned());
    };
    // Same pointer-typedef rule as mutexes: only canonical publishing storage is busy.
    if let PthreadSlotValue::ControlBlock(handle) = classify_pthread_slot_value(slot_value)
        && let Some(entry) = lookup_rwlock_by_handle(handle)
        && entry.storage_address == address
    {
        return u64::from(libc::EBUSY.cast_unsigned());
    }
    let guest_object = guest_calloc(1, GUEST_PTHREAD_OBJECT_SIZE);
    if guest_object == 0 {
        return sce_kernel_try_again();
    }
    let shard = rwlock_shard(guest_object);
    let mut locks = shard
        .locks
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    locks.insert(
        guest_object,
        Arc::new(GuestRwlockEntry::new(guest_object, address)),
    );
    drop(locks);
    write_guest_u64(address, guest_object);
    0
}

fn sce_kernel_try_again() -> u64 {
    // Keep rwlock errno style (raw errno) consistent with the rest of this module.
    let _ = SCE_KERNEL_ERROR_TRY_AGAIN;
    u64::from(libc::EAGAIN.cast_unsigned())
}

fn destroy_resolved_rwlock(entry: Arc<GuestRwlockEntry>, passed_arg: u64) -> u64 {
    let guest_object = entry.guest_object;
    let storage_address = entry.storage_address;
    let shard = rwlock_shard(guest_object);
    let mut locks = shard
        .locks
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(stored) = locks.get(&guest_object).cloned() else {
        return u64::from(libc::EINVAL.cast_unsigned());
    };
    if !Arc::ptr_eq(&entry, &stored) {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    let lock = stored
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
    locks.remove(&guest_object);
    drop(locks);
    if storage_address != 0 {
        write_guest_u64(storage_address, 0);
    }
    if passed_arg != 0
        && passed_arg != guest_object
        && passed_arg != storage_address
        && read_pthread_slot(passed_arg) == Some(guest_object)
    {
        write_guest_u64(passed_arg, 0);
    }
    guest_free(guest_object);
    0
}

pub(crate) fn guest_rwlock_destroy(address: u64) -> u64 {
    if address == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    let Some(entry) = resolve_rwlock(address, false) else {
        return u64::from(libc::EINVAL.cast_unsigned());
    };
    destroy_resolved_rwlock(entry, address)
}

pub(crate) fn guest_rwlock_lock(address: u64, write: bool, try_only: bool) -> u64 {
    if address == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    let Some(entry) = resolve_rwlock(address, true) else {
        return u64::from(libc::EINVAL.cast_unsigned());
    };
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
                        "[REMu][STALL] rwlock write wait exceeded {}s thread={:?} handle={:#x} rwlock={address:#x} object={:#x} writer={:?} readers={} reader_threads={} waiting_readers={} waiting_writers={}",
                        report_after.as_secs(),
                        current_guest_thread_name(),
                        current_guest_thread_handle(),
                        entry.guest_object,
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
                    "[REMu][STALL] rwlock read wait exceeded {}s thread={:?} handle={:#x} rwlock={address:#x} object={:#x} writer={:?} readers={} reader_threads={} waiting_readers={} waiting_writers={}",
                    report_after.as_secs(),
                    current_guest_thread_name(),
                    current_guest_thread_handle(),
                    entry.guest_object,
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
    let Some(entry) = resolve_rwlock(address, false) else {
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
    let entry = resolve_rwlock(address, false)?;
    let lock = entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Some((lock.waiting_readers, lock.waiting_writers))
}
