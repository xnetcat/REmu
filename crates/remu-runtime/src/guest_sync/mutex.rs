// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, Once};

use crate::{
    PthreadMutexAttrState, SCE_KERNEL_ERROR_INVALID_ARGUMENT, SCE_KERNEL_ERROR_NOT_FOUND,
    SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED, SCE_KERNEL_ERROR_TRY_AGAIN,
    assert_wait_is_off_the_dispatch_lock, current_guest_thread_handle, current_guest_thread_name,
    guest_calloc, guest_free, read_guest_u32, sce_kernel_error, stall_report_interval,
    trace_guest_sync, trace_logarithmic_sequence, write_guest_u32, write_guest_u64,
};

use super::identity::{
    GUEST_PTHREAD_OBJECT_SIZE, PTHREAD_ADAPTIVE_MUTEX_INITIALIZER, PthreadSlotValue,
    classify_mutex_slot_value, read_pthread_slot,
};
use super::state::{
    GuestMutexEntry, GuestMutexState, MUTEX_CONTENTION_SEQUENCE, guest_mutex_shard,
    guest_mutex_shards, guest_sync,
};

const GUEST_MUTEX_ENTRY_CACHE_SIZE: usize = 16;
const GUEST_MUTEX_TYPE_OFFSET: u64 = 0x20;
const GUEST_MUTEX_PROTOCOL_OFFSET: u64 = 0x3c;

#[cfg(test)]
pub(crate) static GUEST_MUTEX_WAKE_COUNT: AtomicU64 = AtomicU64::new(0);

/// First-N mutex ABI trace when `REMU_TRACE_MUTEX_ABI=1` (Demon's Souls / Scream bring-up).
static MUTEX_ABI_TRACE_REMAINING: AtomicU64 = AtomicU64::new(0);

fn mutex_abi_trace_enabled() -> bool {
    static ONCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONCE.get_or_init(|| {
        let enabled = std::env::var_os("REMU_TRACE_MUTEX_ABI").is_some_and(|value| value != "0");
        if enabled {
            MUTEX_ABI_TRACE_REMAINING.store(64, Ordering::Relaxed);
        }
        enabled
    })
}

fn trace_mutex_abi(op: &str, address: u64, result: u64) {
    if !mutex_abi_trace_enabled() {
        return;
    }
    if MUTEX_ABI_TRACE_REMAINING
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_sub(1)
        })
        .is_err()
    {
        return;
    }
    let slot = read_pthread_slot(address);
    let object = lookup_mutex_by_handle(address)
        .or_else(|| slot.and_then(|value| lookup_mutex_by_handle(value)))
        .map(|entry| entry.guest_object)
        .unwrap_or(0);
    eprintln!(
        "[REMu][MUTEX][ABI] op={op} arg={address:#x} slot={slot:?} object={object:#x} result={result:#x}"
    );
}

thread_local! {
    /// Per-host-thread cache keyed by control-block handle (SDK identity), not by slot address.
    static GUEST_MUTEX_ENTRY_CACHE: RefCell<
        [Option<(u64, Arc<GuestMutexEntry>)>; GUEST_MUTEX_ENTRY_CACHE_SIZE]
    > = const { RefCell::new([const { None }; GUEST_MUTEX_ENTRY_CACHE_SIZE]) };
}

fn guest_mutex_cache_index(handle: u64) -> usize {
    usize::try_from(handle >> 4).unwrap_or_default() & (GUEST_MUTEX_ENTRY_CACHE_SIZE - 1)
}

fn cached_guest_mutex_entry(handle: u64) -> Option<Arc<GuestMutexEntry>> {
    GUEST_MUTEX_ENTRY_CACHE.with_borrow(|cache| {
        cache[guest_mutex_cache_index(handle)]
            .as_ref()
            .filter(|(cached_handle, entry)| {
                *cached_handle == handle && entry.active.load(Ordering::Acquire)
            })
            .map(|(_, entry)| entry.clone())
    })
}

fn cache_guest_mutex_entry(handle: u64, entry: &Arc<GuestMutexEntry>) {
    static LOGGED: Once = Once::new();
    LOGGED.call_once(|| {
        eprintln!(
            "[REMu][PTHREAD] mutex lookup cache enabled slots={GUEST_MUTEX_ENTRY_CACHE_SIZE} \
             identity=control-block-handle uncontended-wake-elision=true"
        );
    });
    GUEST_MUTEX_ENTRY_CACHE.with_borrow_mut(|cache| {
        cache[guest_mutex_cache_index(handle)] = Some((handle, entry.clone()));
    });
}

fn lookup_mutex_by_handle(handle: u64) -> Option<Arc<GuestMutexEntry>> {
    if handle == 0 || handle == PTHREAD_ADAPTIVE_MUTEX_INITIALIZER {
        return None;
    }
    if let Some(entry) = cached_guest_mutex_entry(handle) {
        return Some(entry);
    }
    let shard = guest_mutex_shards()[guest_mutex_shard(handle)]
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let entry = shard.get(&handle)?.clone();
    cache_guest_mutex_entry(handle, &entry);
    Some(entry)
}

/// Resolve a `ScePthreadMutex *` argument to its host entry via the SDK pointer-typedef ABI.
///
/// Order (matches libc / stack-copy titles):
/// 1. `arg` is itself a published control-block handle (bare-handle call sites).
/// 2. `*arg` is a published control-block handle (normal slot or stack copy of the handle).
/// 3. `*arg` is a static initializer → materialize a control block into the slot.
///
/// Unknown non-initializer values refuse; there is no create-on-first-use fabrication.
pub(crate) fn guest_mutex_entry(
    address: u64,
    allow_static_init: bool,
) -> Option<Arc<GuestMutexEntry>> {
    if address == 0 {
        return None;
    }
    if let Some(entry) = lookup_mutex_by_handle(address) {
        return Some(entry);
    }
    let slot_value = read_pthread_slot(address)?;
    match classify_mutex_slot_value(slot_value) {
        PthreadSlotValue::ControlBlock(handle) => lookup_mutex_by_handle(handle),
        PthreadSlotValue::StaticDefault if allow_static_init => {
            materialize_static_mutex(address, PthreadMutexAttrState::default())
        }
        PthreadSlotValue::StaticAdaptiveMutex if allow_static_init => materialize_static_mutex(
            address,
            PthreadMutexAttrState {
                mutex_type: 4,
                ..PthreadMutexAttrState::default()
            },
        ),
        PthreadSlotValue::StaticDefault | PthreadSlotValue::StaticAdaptiveMutex => None,
    }
}

/// Publish a control block for a static-initializer slot. Races re-read the slot and adopt the
/// winner's handle rather than fabricating a second object for the same guest mutex.
fn materialize_static_mutex(
    slot: u64,
    attributes: PthreadMutexAttrState,
) -> Option<Arc<GuestMutexEntry>> {
    // Serialize static materialization so two first-lock threads cannot both publish.
    let sync = guest_sync();
    let _init_gate = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(entry) = lookup_mutex_by_handle(slot) {
        return Some(entry);
    }
    let slot_value = read_pthread_slot(slot)?;
    match classify_mutex_slot_value(slot_value) {
        PthreadSlotValue::ControlBlock(handle) => lookup_mutex_by_handle(handle),
        PthreadSlotValue::StaticDefault | PthreadSlotValue::StaticAdaptiveMutex => {
            let guest_object = allocate_guest_mutex_object(attributes).ok()?;
            let entry = insert_mutex_entry(slot, guest_object, attributes);
            write_guest_u64(slot, guest_object);
            Some(entry)
        }
    }
}

fn insert_mutex_entry(
    storage_address: u64,
    guest_object: u64,
    attributes: PthreadMutexAttrState,
) -> Arc<GuestMutexEntry> {
    let entry = Arc::new(GuestMutexEntry {
        state: Mutex::new(GuestMutexState {
            mutex_type: attributes.mutex_type,
            ..GuestMutexState::default()
        }),
        wake: Condvar::new(),
        active: AtomicBool::new(true),
        storage_address,
        guest_object,
    });
    let mut shard = guest_mutex_shards()[guest_mutex_shard(guest_object)]
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    shard.insert(guest_object, entry.clone());
    cache_guest_mutex_entry(guest_object, &entry);
    entry
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
                assert_wait_is_off_the_dispatch_lock("scePthreadOnce");
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
    // Init takes a pointer to a slot (`ScePthreadMutex *`), never a bare control-block handle.
    if lookup_mutex_by_handle(address).is_some() {
        eprintln!(
            "[REMu][MUTEX] refused init of bare control-block handle={address:#x} thread={:?}",
            current_guest_thread_name(),
        );
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(previous_handle) = read_pthread_slot(address) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    // Overwriting below is deliberate (see the comment there), but it forks identity: threads that
    // resolved before this point keep using the old object while later ones use the new one. That is
    // harmless for a dead object and fatal for a held one — mutual exclusion is gone, and the loser
    // can never be unlocked because its owner's unlock resolves to the new object. Nothing else in
    // the process reports this, so report it here rather than let it stay invisible.
    if let Some(previous) = lookup_mutex_by_handle(previous_handle) {
        let state = previous
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.owner.is_some() || state.waiters != 0 {
            eprintln!(
                "[REMu][MUTEX][FORK] init over live mutex slot={address:#x} previous_handle={previous_handle:#x} owner_thread={:#x} recursion={} waiters={} thread={:?}",
                state.owner_guest_thread,
                state.recursion,
                state.waiters,
                current_guest_thread_name(),
            );
        }
    }
    // Always publish a fresh control block into the slot. Identity is the handle value
    // (`pthread_mutex_t` is a pointer typedef), not durable ownership of the slot address.
    // Scream creates recursive mutexes in stack locals, returns the handle by value into a
    // global, then reuses the same stack offset for a later CreateMutex while the leftover
    // bits still name the prior live object (and our recorded `storage_address` still points
    // here). Refusing that as "already initialized" returns failure, leaves the title with a
    // NULL handle, and unlock later hits `slot=Some(0)`. POSIX re-init without destroy is UB;
    // console libc overwrites, so we overwrite. The prior object stays reachable via any handle
    // the guest still holds.
    let guest_object = match allocate_guest_mutex_object(attributes) {
        Ok(object) => object,
        Err(error) => return error,
    };
    insert_mutex_entry(address, guest_object, attributes);
    write_guest_u64(address, guest_object);
    trace_mutex_abi("init", address, 0);
    0
}

fn allocate_guest_mutex_object(attributes: PthreadMutexAttrState) -> Result<u64, u64> {
    let object = guest_calloc(1, GUEST_PTHREAD_OBJECT_SIZE);
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

fn destroy_resolved_mutex(entry: Arc<GuestMutexEntry>, passed_arg: u64) -> u64 {
    let guest_object = entry.guest_object;
    let storage_address = entry.storage_address;
    let mut shard = guest_mutex_shards()[guest_mutex_shard(guest_object)]
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(stored_entry) = shard.get(&guest_object).cloned() else {
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
    stored_entry.active.store(false, Ordering::Release);
    drop(mutex);
    let entry = shard
        .remove(&guest_object)
        .expect("the mutex entry checked above remains in its shard");
    drop(shard);
    // Only clear publishing storage when it still holds this object's handle. Stack-init
    // then return-by-value leaves `storage_address` pointing at a recycled frame; blanking
    // unconditionally would corrupt whatever reused that slot.
    if storage_address != 0 && read_pthread_slot(storage_address) == Some(guest_object) {
        write_guest_u64(storage_address, 0);
    }
    // Zero the slot the guest passed when it is a distinct handle-copy of the same object.
    if passed_arg != 0
        && passed_arg != guest_object
        && passed_arg != storage_address
        && read_pthread_slot(passed_arg) == Some(guest_object)
    {
        write_guest_u64(passed_arg, 0);
    }
    guest_free(entry.guest_object);
    0
}

pub(crate) fn guest_mutex_destroy(address: u64) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(entry) = guest_mutex_entry(address, false) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let result = destroy_resolved_mutex(entry, address);
    trace_mutex_abi("destroy", address, result);
    result
}

pub(crate) fn guest_mutex_lock(address: u64, try_only: bool) -> u64 {
    let result = guest_mutex_lock_inner(address, try_only);
    trace_mutex_abi(if try_only { "trylock" } else { "lock" }, address, result);
    result
}

fn guest_mutex_lock_inner(address: u64, try_only: bool) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    let current = std::thread::current().id();
    let guest_thread = current_guest_thread_handle();
    'resolve: loop {
        // Static initializers materialize here; unknown handles refuse (no phantom entries).
        let Some(entry) = guest_mutex_entry(address, true) else {
            eprintln!(
                "[REMu][MUTEX] refused lock of unknown mutex thread={:?} handle={guest_thread:#x} \
                 mutex={address:#x} slot={:?}",
                current_guest_thread_name(),
                read_pthread_slot(address),
            );
            return sce_kernel_error(SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED);
        };
        let mut mutex = entry
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !entry.active.load(Ordering::Acquire) {
            continue 'resolve;
        }
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
                    assert_wait_is_off_the_dispatch_lock("scePthreadMutexLock");
                    mutex.waiters = mutex.waiters.saturating_add(1);
                    let contention_sequence =
                        MUTEX_CONTENTION_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1;
                    if trace_guest_sync() && trace_logarithmic_sequence(contention_sequence, 8) {
                        eprintln!(
                            "[REMu][MUTEX] contention sequence={contention_sequence} thread={:?} \
                         handle={guest_thread:#x} mutex={address:#x} object={:#x} type={} owner={:#x} \
                         recursion={} waiters={}",
                            current_guest_thread_name(),
                            entry.guest_object,
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
                             handle={guest_thread:#x} mutex={address:#x} object={:#x} \
                             owner={:#x} recursion={} waiters={}",
                                report_after.as_secs(),
                                current_guest_thread_name(),
                                entry.guest_object,
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
}

pub(crate) fn guest_mutex_unlock(address: u64) -> u64 {
    let result = guest_mutex_unlock_inner(address);
    trace_mutex_abi("unlock", address, result);
    result
}

fn guest_mutex_unlock_inner(address: u64) -> u64 {
    let current = std::thread::current().id();
    loop {
        let Some(entry) = guest_mutex_entry(address, false) else {
            eprintln!(
                "[REMu][MUTEX] rejected unlock of unknown mutex thread={:?} handle={:#x} mutex={address:#x} slot={:?}",
                current_guest_thread_name(),
                current_guest_thread_handle(),
                read_pthread_slot(address),
            );
            return sce_kernel_error(SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED);
        };
        let mut mutex = entry
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !entry.active.load(Ordering::Acquire) {
            continue;
        }
        if !mutex.owner.as_ref().is_some_and(|owner| *owner == current) {
            eprintln!(
                "[REMu][MUTEX] rejected non-owner unlock thread={:?} handle={:#x} mutex={address:#x} object={:#x} owner={:#x}",
                current_guest_thread_name(),
                current_guest_thread_handle(),
                entry.guest_object,
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
            // `waiters` is protected by the same state mutex used by the condition wait. A
            // contender increments it before atomically releasing that mutex in `Condvar::wait`,
            // so observing zero here proves that there is no sleeper (or waiter transitioning to
            // sleep) to notify. Avoiding the unconditional host condition signal matters for
            // engines that take tens of thousands of uncontended guest locks per second.
            if mutex.waiters != 0 {
                #[cfg(test)]
                GUEST_MUTEX_WAKE_COUNT.fetch_add(1, Ordering::Relaxed);
                entry.wake.notify_one();
            }
        }
        return 0;
    }
}
