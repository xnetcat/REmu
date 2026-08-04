// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;

#[test]
fn thread_accessors_report_what_the_matching_setters_recorded() {
    let mut state = ProbeDispatchState::default();
    state.thread_attributes.insert(
        0x2000,
        PthreadAttrState {
            affinity_mask: 0x18,
            schedule_priority: 300,
            ..PthreadAttrState::default()
        },
    );
    state.pthread_threads.insert(
        0x2000,
        PthreadThreadObject {
            handle: 0x3000,
            unique_id: 1,
            name: "AudioMixer".to_owned(),
        },
    );
    let mut mask = 0_u64;
    let mut priority = 0_u32;
    let mut name = [0_u8; PTHREAD_NAME_CAPACITY];

    assert_eq!(
        dispatch(
            "rcrVFJsQWRY",
            args(&[0x2000, std::ptr::addr_of_mut!(mask).addr() as u64]),
            &mut state
        ),
        Some(SUCCESS)
    );
    assert_eq!(mask, 0x18);
    assert_eq!(
        dispatch(
            "1tKyG7RlMJo",
            args(&[0x2000, std::ptr::addr_of_mut!(priority).addr() as u64]),
            &mut state
        ),
        Some(SUCCESS)
    );
    assert_eq!(priority, 300);
    assert_eq!(
        dispatch(
            "How7B8Oet6k",
            args(&[0x2000, name.as_mut_ptr().addr() as u64]),
            &mut state
        ),
        Some(SUCCESS)
    );
    assert_eq!(&name[..11], b"AudioMixer\0");
}

#[test]
fn thread_accessors_reject_a_null_handle_or_a_null_result() {
    let mut state = ProbeDispatchState::default();

    for nid in ["rcrVFJsQWRY", "1tKyG7RlMJo", "How7B8Oet6k"] {
        assert_eq!(
            dispatch(nid, args(&[0, 0x1000]), &mut state),
            Some(KERNEL_INVALID_ARGUMENT),
            "{nid} accepted a null thread"
        );
        assert_eq!(
            dispatch(nid, args(&[0x2000, 0]), &mut state),
            Some(KERNEL_INVALID_ARGUMENT),
            "{nid} accepted a null result"
        );
    }
    assert_eq!(
        dispatch("How7B8Oet6k", args(&[0x2000, 0x1000]), &mut state),
        Some(KERNEL_NOT_FOUND)
    );
}

#[test]
fn a_thread_the_runtime_never_tracked_reports_the_defaults_it_would_have_been_created_with() {
    let mut state = ProbeDispatchState::default();
    let mut mask = 0_u64;
    let mut priority = 0_u32;

    assert_eq!(
        dispatch(
            "rcrVFJsQWRY",
            args(&[0x4000, std::ptr::addr_of_mut!(mask).addr() as u64]),
            &mut state
        ),
        Some(SUCCESS)
    );
    assert_eq!(mask, DEFAULT_THREAD_AFFINITY_MASK);
    assert_eq!(
        dispatch(
            "1tKyG7RlMJo",
            args(&[0x4000, std::ptr::addr_of_mut!(priority).addr() as u64]),
            &mut state
        ),
        Some(SUCCESS)
    );
    assert_eq!(priority, DEFAULT_THREAD_PRIORITY.cast_unsigned());
}

#[test]
fn comparing_thread_handles_matches_only_the_same_live_thread() {
    let mut state = ProbeDispatchState::default();

    assert_eq!(
        dispatch("3PtV6p3QNX4", args(&[0x2000, 0x2000]), &mut state),
        Some(1)
    );
    assert_eq!(
        dispatch("3PtV6p3QNX4", args(&[0x2000, 0x3000]), &mut state),
        Some(0)
    );
    assert_eq!(dispatch("3PtV6p3QNX4", [0; 6], &mut state), Some(0));
}

#[test]
fn an_attribute_template_the_runtime_never_created_is_refused_by_every_accessor() {
    let mut state = ProbeDispatchState::default();
    let mut detach_state = 0_u32;

    assert_eq!(
        dispatch(
            "JaRMy+QcpeU",
            args(&[0x5000, std::ptr::addr_of_mut!(detach_state).addr() as u64]),
            &mut state
        ),
        Some(KERNEL_INVALID_ARGUMENT)
    );
    assert_eq!(
        dispatch("JaRMy+QcpeU", args(&[0x5000, 0]), &mut state),
        Some(KERNEL_INVALID_ARGUMENT)
    );
    // The POSIX spellings report the same rejection as a small errno instead.
    assert_eq!(
        dispatch("2+pVfgiEd7A", args(&[0x5000, 1]), &mut state),
        Some(u64::from(libc::EINVAL.cast_unsigned()))
    );
    assert_eq!(
        dispatch("FIs3-UQT9sg", args(&[0, 0, 0]), &mut state),
        Some(u64::from(libc::EINVAL.cast_unsigned()))
    );
}

#[test]
fn the_timestamp_counter_advances_at_the_frequency_it_reports() {
    let mut state = ProbeDispatchState::default();

    assert_eq!(
        dispatch("1j3S3n-tTW4", [0; 6], &mut state),
        Some(KERNEL_TSC_FREQUENCY)
    );
    let first = dispatch("-2IRUCO--PM", [0; 6], &mut state).expect("the counter is owned");
    std::thread::sleep(std::time::Duration::from_millis(2));
    let second = dispatch("-2IRUCO--PM", [0; 6], &mut state).expect("the counter is owned");
    assert!(second > first);
    // Two milliseconds of real time cannot read back as a whole second at this scale.
    assert!(second - first < KERNEL_TSC_FREQUENCY);
}

#[test]
fn the_priority_range_covers_every_policy_the_console_defines() {
    let mut state = ProbeDispatchState::default();

    for policy in [SCHED_FIFO, SCHED_OTHER, SCHED_RR] {
        let policy = u64::from(policy.cast_unsigned());
        assert_eq!(
            dispatch("CBNtXOoef-E", args(&[policy]), &mut state),
            Some(signed_i32_return(KERNEL_PRIORITY_HIGHEST))
        );
        assert_eq!(
            dispatch("m0iS6jNsXds", args(&[policy]), &mut state),
            Some(signed_i32_return(KERNEL_PRIORITY_LOWEST))
        );
    }
    assert_eq!(
        dispatch("CBNtXOoef-E", args(&[99]), &mut state),
        Some(POSIX_FAILURE)
    );
    assert_eq!(
        dispatch("m0iS6jNsXds", args(&[99]), &mut state),
        Some(POSIX_FAILURE)
    );
}

#[test]
fn the_process_queries_report_stable_values_the_guest_can_compare() {
    crate::reset_kernel_process_time();
    super::super::thread::reset_core_placement_for_tests();
    let mut state = ProbeDispatchState::default();

    assert_eq!(
        dispatch("HoLVWNanBBc", [0; 6], &mut state),
        Some(u64::from(std::process::id()))
    );
    assert_eq!(dispatch("6XG4B33N09g", [0; 6], &mut state), Some(0));
    assert_eq!(dispatch("4oXYe9Xmk0Q", [0; 6], &mut state), Some(0));
    assert_eq!(dispatch("ca7v6Cxulzs", args(&[0xff]), &mut state), Some(0));
    assert_eq!(dispatch("tU5e3f9gSiU", [0; 6], &mut state), Some(0));
    // With the default affinity the first allowed core is core zero.
    assert_eq!(dispatch("g0VTBxfJyu0", [0; 6], &mut state), Some(0));
}

#[test]
fn a_thread_pinned_away_from_the_first_core_reports_the_first_core_it_allows() {
    let mut state = ProbeDispatchState::default();
    state.thread_attributes.insert(
        current_guest_thread_handle(),
        PthreadAttrState {
            affinity_mask: 0x30,
            ..PthreadAttrState::default()
        },
    );

    assert_eq!(dispatch("g0VTBxfJyu0", [0; 6], &mut state), Some(4));
}
