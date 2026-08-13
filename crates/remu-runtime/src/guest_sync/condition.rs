// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use crate::{
    SCE_KERNEL_ERROR_BUSY, SCE_KERNEL_ERROR_INVALID_ARGUMENT,
    SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED, SCE_KERNEL_ERROR_TIMED_OUT,
    SCE_KERNEL_ERROR_TRY_AGAIN, STALL_SNAPSHOT_EMITTED, assert_wait_is_off_the_dispatch_lock,
    current_guest_thread_handle, current_guest_thread_name, guest_calloc,
    guest_exception_is_pending, guest_free, report_kernel_event_registrations, sce_kernel_error,
    stall_break_enabled, stall_report_interval, trace_guest_sync, trace_logarithmic_sequence,
    write_guest_u64,
};

use super::identity::{
    GUEST_PTHREAD_OBJECT_SIZE, PthreadSlotValue, classify_pthread_slot_value, read_pthread_slot,
};
use super::state::GuestConditionEntry;
use super::{guest_mutex_entry, guest_sync};

fn lookup_condition_by_handle(handle: u64) -> Option<Arc<GuestConditionEntry>> {
    if handle == 0 {
        return None;
    }
    let sync = guest_sync();
    let state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.condition_by_handle(handle)
}

/// Resolve a `ScePthreadCond *` argument to its host entry (SDK pointer-typedef identity).
fn resolve_condition(address: u64, allow_static_init: bool) -> Option<Arc<GuestConditionEntry>> {
    if address == 0 {
        return None;
    }
    if let Some(entry) = lookup_condition_by_handle(address) {
        return Some(entry);
    }
    let slot_value = read_pthread_slot(address)?;
    match classify_pthread_slot_value(slot_value) {
        PthreadSlotValue::ControlBlock(handle) => lookup_condition_by_handle(handle),
        PthreadSlotValue::StaticDefault if allow_static_init => {
            materialize_static_condition(address)
        }
        PthreadSlotValue::StaticDefault | PthreadSlotValue::StaticAdaptiveMutex => None,
    }
}

fn materialize_static_condition(slot: u64) -> Option<Arc<GuestConditionEntry>> {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(entry) = state.condition_by_handle(slot) {
        return Some(entry);
    }
    let slot_value = read_pthread_slot(slot)?;
    match classify_pthread_slot_value(slot_value) {
        PthreadSlotValue::ControlBlock(handle) => state.condition_by_handle(handle),
        PthreadSlotValue::StaticDefault => {
            let guest_object = guest_calloc(1, GUEST_PTHREAD_OBJECT_SIZE);
            if guest_object == 0 {
                return None;
            }
            let entry = state.insert_condition(guest_object, slot);
            write_guest_u64(slot, guest_object);
            Some(entry)
        }
        PthreadSlotValue::StaticAdaptiveMutex => {
            unreachable!("cond slots do not use adaptive init")
        }
    }
}

pub(crate) fn guest_condition_init(address: u64) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    // Init takes a slot pointer, never a bare control-block handle.
    if lookup_condition_by_handle(address).is_some() {
        return sce_kernel_error(SCE_KERNEL_ERROR_BUSY);
    }
    let Some(previous_handle) = read_pthread_slot(address) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    // Forking a condition that still has parked waiters strands them permanently: every later
    // signal resolves to the new object, so nothing can ever release their tickets. Silent today.
    if let Some(previous) = lookup_condition_by_handle(previous_handle) {
        let state = previous
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !state.waiters.is_empty() {
            eprintln!(
                "[REMu][COND][FORK] init over live condition slot={address:#x} previous_handle={previous_handle:#x} stranded_waiters={} notifies={} thread={:?}",
                state.waiters.len(),
                state.notifies,
                current_guest_thread_name(),
            );
        }
    }
    // Always overwrite the slot with a fresh control block (same pointer-typedef rule as mutexes).
    let guest_object = guest_calloc(1, GUEST_PTHREAD_OBJECT_SIZE);
    if guest_object == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_TRY_AGAIN);
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.insert_condition(guest_object, address);
    write_guest_u64(address, guest_object);
    0
}

fn destroy_resolved_condition(entry: Arc<GuestConditionEntry>, passed_arg: u64) -> u64 {
    let guest_object = entry.guest_object;
    let storage_address = entry.storage_address;
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(stored) = state.conditions.get(&guest_object).cloned() else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if !Arc::ptr_eq(&entry, &stored) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let idle = stored
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .is_idle();
    if !idle {
        return sce_kernel_error(SCE_KERNEL_ERROR_BUSY);
    }
    state.conditions.remove(&guest_object);
    drop(state);
    if storage_address != 0 && read_pthread_slot(storage_address) == Some(guest_object) {
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

pub(crate) fn guest_condition_destroy(address: u64) -> u64 {
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(entry) = resolve_condition(address, false) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    destroy_resolved_condition(entry, address)
}

#[cfg(test)]
pub(crate) fn guest_condition_wait(
    condition: u64,
    mutex_address: u64,
    timeout_usec: Option<u64>,
) -> u64 {
    guest_condition_wait_with_interrupt(condition, mutex_address, timeout_usec, || {})
}

pub(crate) fn guest_condition_wait_with_interrupt<F>(
    condition: u64,
    mutex_address: u64,
    timeout_usec: Option<u64>,
    mut on_interrupt: F,
) -> u64
where
    F: FnMut(),
{
    if condition == 0 || mutex_address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sync = guest_sync();
    let current = std::thread::current().id();
    let Some(mutex_entry) = guest_mutex_entry(mutex_address, false) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED);
    };
    let mutex_identity = mutex_entry.guest_object;
    let mut mutex = mutex_entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let released_recursion = if mutex.owner.as_ref().is_some_and(|owner| *owner == current) {
        mutex.recursion.max(1)
    } else {
        return sce_kernel_error(SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED);
    };
    let Some(cond_entry) = resolve_condition(condition, true) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let condition_handle = cond_entry.guest_object;
    let (waiter, entry) = {
        let mut state = sync
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(waiter) = state.register_condition_waiter(condition_handle, mutex_identity) else {
            return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
        };
        let entry = state
            .condition_by_handle(condition_handle)
            .expect("register_condition_waiter requires a live condition entry");
        if trace_guest_sync() && trace_logarithmic_sequence(waiter, 8) {
            eprintln!(
                "[REMu][COND] wait ticket={waiter} thread={} condition={condition:#x} object={condition_handle:#x} mutex={mutex_address:#x} mutex_object={mutex_identity:#x} recursion={released_recursion} timeout_usec={timeout_usec:?}",
                current_guest_thread_handle(),
            );
        }
        (waiter, entry)
    };
    mutex.owner = None;
    mutex.owner_guest_thread = 0;
    mutex.recursion = 0;
    mutex_entry.wake.notify_one();
    drop(mutex);
    let guest_thread = current_guest_thread_handle();

    // Set when a diagnostic spurious wake retires an otherwise unbounded wait.
    let mut released_spuriously = false;
    assert_wait_is_off_the_dispatch_lock("scePthreadCondWait");
    let mut cond = entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // `notifies` is cumulative over the condition object's whole life and is never reset, so on a
    // hot condition it says nothing about the wait in front of you: 7688-and-stalled is equally
    // consistent with "signalled constantly, our wake was lost" and "not signalled once since we
    // parked". The delta against this snapshot is the number that separates them.
    let notifies_at_wait = cond.notifies;
    let (timed_out, interrupted) = if let Some(timeout) = timeout_usec {
        let duration = Duration::from_micros(timeout);
        let mut expired = false;
        if let Some(report_after) = stall_report_interval().filter(|report| duration > *report) {
            let started = Instant::now();
            let mut reported = false;
            loop {
                let elapsed = started.elapsed();
                let remaining = duration.saturating_sub(elapsed);
                if remaining.is_zero() {
                    expired = true;
                    break;
                }
                let (next_cond, wait_result) = entry
                    .wake
                    .wait_timeout_while(cond, remaining.min(report_after), |cond| {
                        !cond.waiter_released(waiter) && !guest_exception_is_pending(guest_thread)
                    })
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                cond = next_cond;
                if cond.waiter_released(waiter) || guest_exception_is_pending(guest_thread) {
                    break;
                }
                if wait_result.timed_out() && started.elapsed() >= duration {
                    expired = true;
                    break;
                }
                if !reported {
                    reported = true;
                    eprintln!(
                        "[REMu][STALL] timed condition wait exceeded {}s thread={:?} handle={guest_thread:#x} ticket={waiter} condition={condition:#x} mutex={mutex_address:#x} timeout_usec={timeout} notifies={} since_wait={}",
                        report_after.as_secs(),
                        current_guest_thread_name(),
                        cond.notifies,
                        cond.notifies.saturating_sub(notifies_at_wait),
                    );
                }
                if stall_break_enabled() {
                    eprintln!(
                        "[REMu][STALL] releasing stalled timed wait thread={:?} condition={condition:#x} ticket={waiter}",
                        current_guest_thread_name(),
                    );
                    released_spuriously = true;
                    break;
                }
            }
        } else {
            let (next_cond, wait_result) = entry
                .wake
                .wait_timeout_while(cond, duration, |cond| {
                    !cond.waiter_released(waiter) && !guest_exception_is_pending(guest_thread)
                })
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            cond = next_cond;
            expired = wait_result.timed_out();
        }
        let interrupted = !cond.waiter_released(waiter) && guest_exception_is_pending(guest_thread);
        expired = expired && !cond.waiter_released(waiter) && !interrupted;
        if expired {
            cond.timeouts = cond.timeouts.saturating_add(1);
        }
        (expired, interrupted)
    } else if let Some(report_after) = stall_report_interval() {
        // An unbounded wait that never completes is the hardest stall class to attribute, because
        // the thread is parked with no record of what it wanted. Wake periodically to name the
        // wait once; the predicate is unchanged, so this cannot alter wait semantics.
        let mut reported = false;
        let mut report_ticks = 0_u32;
        loop {
            let (next_cond, wait_result) = entry
                .wake
                .wait_timeout_while(cond, report_after, |cond| {
                    !cond.waiter_released(waiter) && !guest_exception_is_pending(guest_thread)
                })
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            cond = next_cond;
            if !wait_result.timed_out() {
                break;
            }
            if stall_break_enabled() {
                // POSIX permits spurious wakeups, so releasing a stalled waiter cannot corrupt a
                // correct guest: it re-tests its predicate and waits again. Whether it proceeds
                // distinguishes a lost wakeup from a predicate that was never satisfied.
                eprintln!(
                    "[REMu][STALL] releasing stalled wait thread={:?} condition={condition:#x} ticket={waiter}",
                    current_guest_thread_name()
                );
                released_spuriously = true;
                break;
            }
            report_ticks += 1;
            // A healthy run parks dozens of idle pool threads in unbounded waits; per-tick
            // reports from all of them bury the engine-thread signal a wedge run needs (746
            // bystander lines in run 20260813T225147Z). Frame-pipeline threads keep every
            // tick; everything else backs off to power-of-two ticks, so an idle thread costs
            // a handful of lines per run while a wedged engine thread never goes quiet.
            let thread_name = current_guest_thread_name();
            let frame_pipeline_thread = ["Render", "RHI", "Agc", "GameThread"]
                .iter()
                .any(|marker| thread_name.contains(marker));
            if frame_pipeline_thread || report_ticks.is_power_of_two() {
                reported = true;
                let notifies = cond.notifies;
                // The guarded guest state beside the condvar and mutex objects discriminates
                // the two stall classes across ticks: static bytes while the producer cycles
                // means the enqueue is gated; changing bytes (queued work) beside a parked
                // waiter means a wake was missed. Checked reads only — a diagnostic that can
                // fault is worse than none.
                let peek = |address: u64| -> String {
                    let mut bytes = [0_u8; 32];
                    if remu_memory::read_guest_shared_backing(address, &mut bytes) {
                        let words: [u32; 8] = std::array::from_fn(|index| {
                            u32::from_le_bytes(
                                bytes[index * 4..index * 4 + 4].try_into().expect("dword"),
                            )
                        });
                        format!("{words:08x?}")
                    } else {
                        "unmapped".to_owned()
                    }
                };
                eprintln!(
                    "[REMu][STALL] condition wait exceeded {}s thread={thread_name:?} handle={guest_thread:#x} ticket={waiter} condition={condition:#x} mutex={mutex_address:#x} notifies={notifies} since_wait={} cond_obj={} mutex_obj={}",
                    report_after.as_secs(),
                    notifies.saturating_sub(notifies_at_wait),
                    peek(condition),
                    peek(mutex_address),
                );
                if !STALL_SNAPSHOT_EMITTED.swap(true, Ordering::Relaxed) {
                    drop(cond);
                    emit_condition_stall_snapshot();
                    cond = entry
                        .state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                }
            }
        }
        let interrupted = !cond.waiter_released(waiter) && guest_exception_is_pending(guest_thread);
        (false, interrupted)
    } else {
        cond = entry
            .wake
            .wait_while(cond, |cond| {
                !cond.waiter_released(waiter) && !guest_exception_is_pending(guest_thread)
            })
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let interrupted = !cond.waiter_released(waiter) && guest_exception_is_pending(guest_thread);
        (false, interrupted)
    };
    if timed_out && trace_guest_sync() {
        let timeouts = cond.timeouts;
        if trace_logarithmic_sequence(timeouts, 8) {
            eprintln!(
                "[REMu][COND] timeout sequence={timeouts} thread={guest_thread:#x} condition={condition:#x} mutex={mutex_address:#x} requested_usec={timeout_usec:?}"
            );
        }
    }
    if timed_out || interrupted || released_spuriously {
        // Timeout/interrupt was decided while holding this condition's state lock. Retire the
        // ticket before releasing that lock so a concurrent notify cannot select it in the gap
        // before `finish_condition_wait` acquires the registry lock. Selecting an already-expired
        // ticket both loses the signal and used to trip the exactly-one-winner assertion.
        debug_assert!(!cond.waiter_released(waiter));
        cond.cancel_wait(waiter);
    }
    drop(cond);

    {
        let mut state = sync
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.finish_condition_wait(
            condition_handle,
            waiter,
            timed_out || interrupted || released_spuriously,
        );
    }

    if interrupted {
        // A cooperative guest exception is analogous to a signal interrupting pthread_cond_wait.
        // Run the handler before trying to reacquire the guest mutex: a stop-the-world handler can
        // suspend the mutex owner, so reacquiring first creates a circular suspension deadlock.
        // Once the handler returns (the guest has resumed), normal cond-wait semantics still
        // require this thread to reacquire the mutex before returning to guest code.
        on_interrupt();
    }

    let mut mutex = mutex_entry
        .state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        if mutex.owner.is_none() {
            mutex.owner = Some(current);
            mutex.owner_guest_thread = guest_thread;
            mutex.recursion = released_recursion;
            return if timed_out {
                sce_kernel_error(SCE_KERNEL_ERROR_TIMED_OUT)
            } else {
                0
            };
        }
        assert_wait_is_off_the_dispatch_lock("scePthreadCondWait mutex reacquire");
        mutex.waiters = mutex.waiters.saturating_add(1);
        mutex = mutex_entry
            .wake
            .wait(mutex)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        mutex.waiters = mutex.waiters.saturating_sub(1);
    }
}

fn emit_condition_stall_snapshot() {
    let sync = guest_sync();
    let state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut summary: Vec<_> = state
        .conditions
        .iter()
        .map(|(handle, entry)| {
            let guest_condition = entry
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (
                *handle,
                guest_condition.waiters.len(),
                guest_condition.notifies,
                guest_condition.timeouts,
            )
        })
        .collect();
    summary.sort_by_key(|(_, waiters, notifies, timeouts)| {
        (
            std::cmp::Reverse(*timeouts),
            std::cmp::Reverse(*waiters),
            *notifies,
        )
    });
    for (handle, waiters, notifies, timeouts) in summary.iter().take(24) {
        eprintln!(
            "[REMu][STALL] condition_object={handle:#x} waiters={waiters} notifies={notifies} timeouts={timeouts}"
        );
    }
    drop(state);
    report_kernel_event_registrations();
}

pub(crate) fn guest_condition_notify(condition: u64, broadcast: bool) -> u64 {
    if condition == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(entry) = resolve_condition(condition, true) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let sync = guest_sync();
    let (selected, notify_sequence, pending_before) = {
        let mut state = sync
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.condition_notify_count = state.condition_notify_count.saturating_add(1);
        let notify_sequence = state.condition_notify_count;
        let mut guest_condition = entry
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guest_condition.notifies = guest_condition.notifies.saturating_add(1);
        let pending_before = guest_condition.waiters.len();
        let selected = if broadcast {
            guest_condition.release_all()
        } else {
            guest_condition.release_one().into_iter().collect()
        };
        for waiter in &selected {
            if let Some(waiter_state) = state.condition_waiters.get_mut(waiter) {
                waiter_state.released = true;
            }
        }
        drop(guest_condition);
        (selected, notify_sequence, pending_before)
    };
    if trace_guest_sync() && trace_logarithmic_sequence(notify_sequence, 8) {
        eprintln!(
            "[REMu][COND] notify sequence={notify_sequence} condition={condition:#x} object={:#x} broadcast={broadcast} pending={pending_before} selected={selected:?}",
            entry.guest_object
        );
    }
    if !selected.is_empty() {
        // Per-object wake (plan C3 / B16): only waiters on this condition are notified. FIFO
        // ticket selection above is unchanged — only the host Condvar is per-object.
        entry.wake.notify_all();
    }
    0
}

/// Wakes every live guest-sync waiter so cooperative exception delivery cannot lose a parked
/// thread after per-object Condvars replace the process-wide `GUEST_SYNC` Condvar.
pub(crate) fn wake_all_guest_sync_for_exception() {
    let sync = guest_sync();
    let state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let event_flags: Vec<Arc<_>> = state.event_flags.values().cloned().collect();
    let semaphores: Vec<Arc<_>> = state.semaphores.values().cloned().collect();
    let conditions: Vec<Arc<GuestConditionEntry>> = state.conditions.values().cloned().collect();
    drop(state);
    // pthread_once still parks on the registry Condvar.
    sync.1.notify_all();
    for entry in event_flags {
        entry.wake.notify_all();
    }
    for entry in semaphores {
        entry.wake.notify_all();
    }
    for entry in conditions {
        entry.wake.notify_all();
    }
    for shard in super::state::guest_rwlock_shards() {
        let locks = shard
            .locks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for entry in locks.values() {
            entry.wake.notify_all();
        }
    }
}
