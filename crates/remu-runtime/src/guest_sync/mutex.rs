// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::sync::atomic::Ordering;
use std::sync::{Arc, Condvar, Mutex};

use remu_shader::GuestMemoryReader;

use crate::{
    ProcessGuestMemory, PthreadMutexAttrState, SCE_KERNEL_ERROR_INVALID_ARGUMENT,
    SCE_KERNEL_ERROR_NOT_FOUND, SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED,
    SCE_KERNEL_ERROR_TRY_AGAIN, current_guest_thread_handle, current_guest_thread_name,
    guest_calloc, guest_free, read_guest_u32, sce_kernel_error, stall_report_interval,
    trace_guest_sync, trace_logarithmic_sequence, write_guest_u32, write_guest_u64,
};

use super::{
    GuestMutexEntry, GuestMutexState, MUTEX_CONTENTION_SEQUENCE, guest_mutex_shard,
    guest_mutex_shards, guest_sync,
};

pub(crate) fn guest_mutex_entry(address: u64, create: bool) -> Option<Arc<GuestMutexEntry>> {
    let shard = guest_mutex_shards()[guest_mutex_shard(address)]
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(entry) = shard.get(&address) {
        return Some(entry.clone());
    }
    drop(shard);
    for candidate in guest_mutex_shards() {
        let candidate = candidate
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(entry) = candidate
            .values()
            .find(|entry| entry.guest_object != 0 && entry.guest_object == address)
        {
            return Some(entry.clone());
        }
    }
    if !create {
        return None;
    }
    let mut shard = guest_mutex_shards()[guest_mutex_shard(address)]
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(entry) = shard.get(&address) {
        return Some(entry.clone());
    }
    create.then(|| {
        let entry = Arc::new(GuestMutexEntry {
            state: Mutex::new(GuestMutexState::default()),
            wake: Condvar::new(),
            storage_address: address,
            guest_object: 0,
        });
        shard.insert(address, entry.clone());
        entry
    })
}

pub(crate) fn guest_pthread_once(control: u64, routine: u64) -> u64 {
    const NEVER_DONE: u32 = 0;
    const DONE: u32 = 1;
    const IN_PROGRESS: u32 = 2;
    const WAIT: u32 = 3;
    if control == 0 || routine == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        match read_guest_u32(control) {
            DONE => return 0,
            NEVER_DONE => {
                write_guest_u32(control, IN_PROGRESS);
                break;
            }
            IN_PROGRESS | WAIT => {
                write_guest_u32(control, WAIT);
                state = sync
                    .1
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            _ => return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT),
        }
    }
    drop(state);

    type OnceRoutine = unsafe extern "C" fn();
    // SAFETY: scePthreadOnce's ABI requires a live no-argument guest function pointer. No REmu
    // state lock is held while it runs, so initialization may call back through other imports.
    let routine: OnceRoutine = unsafe { std::mem::transmute(routine) };
    unsafe { routine() };

    let _state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    write_guest_u32(control, DONE);
    sync.1.notify_all();
    0
}

pub(crate) fn guest_mutex_init(address: u64, attributes: PthreadMutexAttrState) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let guest_object = match allocate_guest_mutex_object(address, attributes) {
        Ok(object) => object,
        Err(error) => return error,
    };
    let mut shard = guest_mutex_shards()[guest_mutex_shard(address)]
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(existing) = shard.get(&address).cloned() {
        let mutex = existing
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // A prior lock/trylock can auto-create the host entry before scePthreadMutexInit runs.
        // Replace that placeholder when it is idle; reject a live mutex the way Orbis does.
        if mutex.owner.is_some() || mutex.waiters != 0 {
            if guest_object != 0 {
                guest_free(guest_object);
            }
            return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
        }
        drop(mutex);
        if let Some(removed) = shard.remove(&address) {
            if removed.guest_object != 0 {
                guest_free(removed.guest_object);
            }
        }
    }
    shard.insert(
        address,
        Arc::new(GuestMutexEntry {
            state: Mutex::new(GuestMutexState {
                mutex_type: attributes.mutex_type,
                ..GuestMutexState::default()
            }),
            wake: Condvar::new(),
            storage_address: address,
            guest_object,
        }),
    );
    if guest_object != 0 {
        write_guest_u64(address, guest_object);
    }
    0
}

pub(super) fn allocate_guest_mutex_object(
    storage: u64,
    attributes: PthreadMutexAttrState,
) -> Result<u64, u64> {
    let mut existing = [0_u8; 8];
    if !ProcessGuestMemory.read(storage, &mut existing) {
        // Synchronization unit tests also exercise deliberately synthetic, non-dereferenceable
        // addresses. They only require the host-side ownership model; a real mapped guest mutex
        // must always take the object-backed path above.
        return Ok(0);
    }
    const GUEST_MUTEX_OBJECT_SIZE: u64 = 0x100;
    const GUEST_MUTEX_TYPE_OFFSET: u64 = 0x20;
    const GUEST_MUTEX_PROTOCOL_OFFSET: u64 = 0x3c;
    let object = guest_calloc(1, GUEST_MUTEX_OBJECT_SIZE);
    if object == 0 {
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_TRY_AGAIN));
    }
    write_guest_u32(
        object + GUEST_MUTEX_TYPE_OFFSET,
        attributes.mutex_type.cast_unsigned(),
    );
    write_guest_u32(
        object + GUEST_MUTEX_PROTOCOL_OFFSET,
        attributes.protocol.cast_unsigned(),
    );
    Ok(object)
}

pub(crate) fn guest_mutex_destroy(address: u64) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(entry) = guest_mutex_entry(address, false) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let storage_address = entry.storage_address;
    let mut shard = guest_mutex_shards()[guest_mutex_shard(storage_address)]
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(stored_entry) = shard.get(&storage_address).cloned() else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if !Arc::ptr_eq(&entry, &stored_entry) {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    let mutex = stored_entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if mutex.owner.is_some() || mutex.waiters != 0 {
        return u64::from(libc::EBUSY.cast_unsigned());
    }
    drop(mutex);
    let entry = shard
        .remove(&storage_address)
        .expect("the mutex entry checked above remains in its shard");
    let guest_object = entry.guest_object;
    drop(shard);
    if guest_object != 0 {
        write_guest_u64(storage_address, 0);
        guest_free(guest_object);
    }
    0
}

pub(crate) fn guest_mutex_lock(address: u64, try_only: bool) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    let Some(entry) = guest_mutex_entry(address, true) else {
        return u64::MAX;
    };
    let current = std::thread::current().id();
    let mut mutex = entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let guest_thread = current_guest_thread_handle();
    let mut reported_stall = false;
    loop {
        match mutex.owner.as_ref() {
            None => {
                mutex.owner = Some(current);
                mutex.owner_guest_thread = guest_thread;
                mutex.recursion = 1;
                return 0;
            }
            Some(owner) if *owner == current && mutex.mutex_type == 2 => {
                mutex.recursion = mutex.recursion.saturating_add(1);
                return 0;
            }
            Some(owner) if *owner == current && try_only => {
                return u64::from(libc::EBUSY.cast_unsigned());
            }
            // Error-checking and adaptive mutexes diagnose a same-thread relock. A normal mutex
            // deliberately falls through to the blocking path below and deadlocks, matching the
            // Gen5 pthread contract.
            Some(owner) if *owner == current && matches!(mutex.mutex_type, 1 | 4) => {
                return u64::from(libc::EDEADLK.cast_unsigned());
            }
            Some(_) if try_only => return u64::from(libc::EBUSY as u32),
            Some(_) => {
                mutex.waiters = mutex.waiters.saturating_add(1);
                let contention_sequence =
                    MUTEX_CONTENTION_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1;
                if trace_guest_sync() && trace_logarithmic_sequence(contention_sequence, 8) {
                    eprintln!(
                        "[REMu][MUTEX] contention sequence={contention_sequence} thread={:?} \
                         handle={guest_thread:#x} mutex={address:#x} type={} owner={:#x} \
                         recursion={} waiters={}",
                        current_guest_thread_name(),
                        mutex.mutex_type,
                        mutex.owner_guest_thread,
                        mutex.recursion,
                        mutex.waiters,
                    );
                }
                if let Some(report_after) = stall_report_interval() {
                    let (next_mutex, wait_result) = entry
                        .wake
                        .wait_timeout(mutex, report_after)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    mutex = next_mutex;
                    if wait_result.timed_out() && !reported_stall {
                        reported_stall = true;
                        eprintln!(
                            "[REMu][STALL] mutex lock exceeded {}s thread={:?} \
                             handle={guest_thread:#x} mutex={address:#x} \
                             owner={:#x} recursion={} waiters={}",
                            report_after.as_secs(),
                            current_guest_thread_name(),
                            mutex.owner_guest_thread,
                            mutex.recursion,
                            mutex.waiters,
                        );
                    }
                } else {
                    mutex = entry
                        .wake
                        .wait(mutex)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                mutex.waiters = mutex.waiters.saturating_sub(1);
            }
        }
    }
}

pub(crate) fn guest_mutex_unlock(address: u64) -> u64 {
    let Some(entry) = guest_mutex_entry(address, false) else {
        eprintln!(
            "[REMu][MUTEX] rejected unlock of unknown mutex thread={:?} handle={:#x} mutex={address:#x}",
            current_guest_thread_name(),
            current_guest_thread_handle(),
        );
        return sce_kernel_error(SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED);
    };
    let current = std::thread::current().id();
    let mut mutex = entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !mutex.owner.as_ref().is_some_and(|owner| *owner == current) {
        eprintln!(
            "[REMu][MUTEX] rejected non-owner unlock thread={:?} handle={:#x} mutex={address:#x} owner={:#x}",
            current_guest_thread_name(),
            current_guest_thread_handle(),
            mutex.owner_guest_thread,
        );
        return sce_kernel_error(SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED);
    }
    if mutex.recursion > 1 {
        mutex.recursion -= 1;
    } else {
        mutex.owner = None;
        mutex.owner_guest_thread = 0;
        mutex.recursion = 0;
        entry.wake.notify_one();
    }
    0
}
