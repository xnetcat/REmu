// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::{
    DEFAULT_THREAD_AFFINITY_MASK, DEFAULT_THREAD_PRIORITY, ProbeDispatchState,
    SCE_KERNEL_ERROR_INVALID_ARGUMENT, SCE_KERNEL_ERROR_NOT_FOUND, current_guest_thread_handle,
    pthread_attr_state, sce_kernel_error, signed_i32_return, trace_guest_affinity,
    update_pthread_attr, write_guest_bytes, write_guest_u32, write_guest_u64,
};

use super::set_guest_errno;

/// Scheduling policy identifiers accepted by the priority-range queries.
pub(super) const SCHED_FIFO: i32 = 1;

pub(super) const SCHED_OTHER: i32 = 2;

pub(super) const SCHED_RR: i32 = 3;

/// Most and least urgent thread priorities the console's realtime class accepts.
///
/// The scale is inverted with respect to POSIX: a smaller number is more urgent.
pub(super) const KERNEL_PRIORITY_HIGHEST: i32 = 256;

pub(super) const KERNEL_PRIORITY_LOWEST: i32 = 767;

/// Bytes copied for a thread name, terminator included.
///
/// The name accessor takes no capacity argument, so the ABI fixes one maximum every caller must
/// have room for. Names longer than this are truncated rather than overrunning that buffer.
pub(super) const PTHREAD_NAME_CAPACITY: usize = 32;

/// Reports the CPU affinity mask of a running guest thread.
pub(super) fn pthread_get_affinity(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (thread, output) = (arguments[0], arguments[1]);
    if thread == 0 || output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let mask = state
        .thread_attributes
        .get(&thread)
        .map_or(DEFAULT_THREAD_AFFINITY_MASK, |attributes| {
            attributes.affinity_mask
        });
    write_guest_u64(output, mask);
    0
}

/// Reports the scheduling priority of a running guest thread.
pub(super) fn pthread_get_priority(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (thread, output) = (arguments[0], arguments[1]);
    if thread == 0 || output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let priority = state
        .thread_attributes
        .get(&thread)
        .map_or(DEFAULT_THREAD_PRIORITY, |attributes| {
            attributes.schedule_priority
        });
    write_guest_u32(output, priority.cast_unsigned());
    0
}

/// Copies a running guest thread's name into the caller's buffer.
pub(super) fn pthread_get_name(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (thread, output) = (arguments[0], arguments[1]);
    if thread == 0 || output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(object) = state.pthread_threads.get(&thread) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let name = object.name.as_bytes();
    let mut buffer = Vec::with_capacity(PTHREAD_NAME_CAPACITY);
    buffer.extend_from_slice(&name[..name.len().min(PTHREAD_NAME_CAPACITY - 1)]);
    buffer.push(0);
    write_guest_bytes(output, &buffer);
    0
}

/// Reports whether an attribute template creates joinable or detached threads.
pub(super) fn pthread_attr_get_detach_state(
    arguments: [u64; 6],
    state: &ProbeDispatchState,
) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(attributes) = pthread_attr_state(arguments[0], state) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    write_guest_u32(arguments[1], attributes.detach_state.cast_unsigned());
    0
}

/// Accepts the dedicated-scheduling-slot request on an attribute template.
///
/// Guest threads run on host threads whose placement the host scheduler owns, so no attribute
/// value can reserve a core for one. The template is still resolved, because a caller that names
/// an attribute the runtime never created has made an error worth reporting.
pub(super) fn pthread_attr_set_solo_schedule(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    update_pthread_attr(arguments[0], state, |_| {})
}

/// Reports one end of the priority range a scheduling policy accepts.
///
/// The realtime and time-sharing classes share one range here, because guest threads are created
/// with the same default priority whichever policy names them.
pub(super) fn scheduler_priority_bound(policy: i32, bound: i32) -> u64 {
    if matches!(policy, SCHED_FIFO | SCHED_OTHER | SCHED_RR) {
        signed_i32_return(bound)
    } else {
        set_guest_errno(libc::EINVAL);
        signed_i32_return(-1)
    }
}

/// Logical CPU count of the Prospero SYSTEM/NATIVE mask (`SCE_KERNEL_CPUMASK_13CPU` = `0x1fff`).
///
/// Game-process defaults use the 7-CPU mask (`DEFAULT_THREAD_AFFINITY_MASK` = `0x7f`), but a title
/// may still call `scePthreadSetaffinity(…, SCE_KERNEL_CPUMASK_13CPU)`, so placement must accept
/// cores 0–12 whenever the thread's mask permits them.
const PS5_LOGICAL_CPU_COUNT: u32 = 13;
const _: () = assert!(crate::SYSTEM_THREAD_AFFINITY_MASK == 0x1fff);
const _: () = assert!(crate::DEFAULT_THREAD_AFFINITY_MASK == 0x7f);

/// Distributes freshly placed threads across the cores their affinity permits.
///
/// A round-robin seed, not a per-call value: once a thread is placed it keeps its core, so this
/// only advances when a thread is placed for the first time or migrated after an affinity change.
static NEXT_CORE_PLACEMENT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Reports which logical CPU the calling thread is running on.
///
/// REmu runs guest threads on host threads and does not pin them to emulated cores, and macOS does
/// not expose the host core a thread is executing on. Deriving the answer from the affinity mask —
/// as this used to — is a fabrication: the affinity mask is what a thread *permits*, not where it
/// runs, and recomputing it every call pretends a placement that never happened.
///
/// Instead REmu makes a real placement decision it owns and reports consistently: the first time a
/// thread asks, it is assigned a core round-robin among those its affinity permits, and it keeps
/// that core on every later call. The placement is only revisited when an affinity change stops
/// permitting the current core, which the console models as a migration. A thread whose affinity
/// permits no core in the console's range reports core 0, the value the console returns when no
/// affinity applies.
pub(super) fn kernel_current_cpu(state: &mut ProbeDispatchState) -> u64 {
    let handle = current_guest_thread_handle();
    let mask = state
        .thread_attributes
        .get(&handle)
        .map_or(DEFAULT_THREAD_AFFINITY_MASK, |attributes| {
            attributes.affinity_mask
        });
    u64::from(placed_core(state, handle, mask))
}

/// Returns the emulated core REmu is running `handle` on, placing it if it has none yet.
fn placed_core(state: &mut ProbeDispatchState, handle: u64, affinity_mask: u64) -> u32 {
    if let Some(&core) = state.thread_running_cpu.get(&handle)
        && core_is_permitted(affinity_mask, core)
    {
        return core;
    }
    let permitted: Vec<u32> = (0..PS5_LOGICAL_CPU_COUNT)
        .filter(|&core| core_is_permitted(affinity_mask, core))
        .collect();
    if permitted.is_empty() {
        // No console core is permitted, so there is no honest placement to report or store.
        return 0;
    }
    let placement = NEXT_CORE_PLACEMENT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let core = permitted[placement as usize % permitted.len()];
    state.thread_running_cpu.insert(handle, core);
    if trace_guest_affinity() {
        eprintln!(
            "[REMu][AFFINITY] place thread={handle:#x} name={:?} mask={affinity_mask:#x} core={core}",
            state
                .pthread_threads
                .get(&handle)
                .map(|thread| thread.name.as_str())
                .unwrap_or("unknown"),
        );
    }
    core
}

fn core_is_permitted(affinity_mask: u64, core: u32) -> bool {
    affinity_mask & (1_u64 << core) != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SYSTEM_THREAD_AFFINITY_MASK;

    const THREAD: u64 = 0x4000_1000;

    #[test]
    fn a_thread_is_placed_on_a_core_its_affinity_permits() {
        let mut state = ProbeDispatchState::default();
        // Affinity permitting only core 3 leaves exactly one honest placement.
        assert_eq!(placed_core(&mut state, THREAD, 1 << 3), 3);
        assert_eq!(state.thread_running_cpu.get(&THREAD), Some(&3));
    }

    #[test]
    fn a_placement_is_stable_across_calls_rather_than_recomputed() {
        let mut state = ProbeDispatchState::default();
        let first = placed_core(&mut state, THREAD, 1 << 5);
        // Later calls report the placement REmu committed to, not a fresh derivation. The mask
        // still permits core 5, so the same core comes back even though other threads have since
        // advanced the round-robin seed.
        let _ = placed_core(&mut state, THREAD + 0x100, 0x7f);
        assert_eq!(placed_core(&mut state, THREAD, 1 << 5), first);
        assert_eq!(first, 5);
    }

    #[test]
    fn losing_affinity_for_the_current_core_migrates_the_thread() {
        let mut state = ProbeDispatchState::default();
        assert_eq!(placed_core(&mut state, THREAD, 1 << 2), 2);
        // The thread re-pins to core 0 only, so its placement must migrate off core 2.
        assert_eq!(placed_core(&mut state, THREAD, 1 << 0), 0);
        assert_eq!(state.thread_running_cpu.get(&THREAD), Some(&0));
    }

    #[test]
    fn an_affinity_permitting_no_console_core_reports_core_zero_without_a_placement() {
        let mut state = ProbeDispatchState::default();
        // Bits set only outside the console's thirteen logical CPUs: no honest placement exists.
        assert_eq!(placed_core(&mut state, THREAD, 0xe000), 0);
        assert_eq!(placed_core(&mut state, THREAD, 0), 0);
        assert_eq!(state.thread_running_cpu.get(&THREAD), None);
    }

    #[test]
    fn every_placement_falls_inside_the_default_affinity_set() {
        let mut state = ProbeDispatchState::default();
        // Game-process default (0x7f) permits cores 0–6 only.
        for index in 0..28 {
            let core = placed_core(&mut state, THREAD + index, DEFAULT_THREAD_AFFINITY_MASK);
            assert!(core_is_permitted(DEFAULT_THREAD_AFFINITY_MASK, core));
            assert!(core < 7);
        }
    }

    #[test]
    fn sdk_system_affinity_can_place_a_thread_on_the_thirteenth_cpu() {
        let mut state = ProbeDispatchState::default();

        assert_eq!(placed_core(&mut state, THREAD, SYSTEM_THREAD_AFFINITY_MASK & (1 << 12)), 12);
        assert_eq!(state.thread_running_cpu.get(&THREAD), Some(&12));
    }
}
