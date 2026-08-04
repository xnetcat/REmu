// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;

#[test]
fn a_user_event_registration_can_be_added_and_withdrawn_again() {
    let mut state = ProbeDispatchState::default();
    crate::event_state::with_event_state_mut(|e| {
        e.kernel_equeues.insert(7);
    });

    assert_eq!(
        dispatch("4R6-OvI2cEA", args(&[7, 42]), &mut state),
        Some(SUCCESS)
    );
    assert!(crate::event_state::with_event_state(|e| {
        e.kernel_event_registrations
            .contains_key(&(7, KERNEL_EVENT_FILTER_USER, 42))
    }));
    crate::event_state::with_event_state_mut(|e| {
        e.pending_events.push_back(KernelQueuedEvent {
            equeue: 7,
            ident: 42,
            filter: KERNEL_EVENT_FILTER_USER,
            flags: 0,
            fflags: 1,
            data: 0,
            user_data: 0,
            agc_graphics_completions: 0,
            agc_user_interrupts: 0,
        });
    });

    assert_eq!(
        dispatch("LJDwdSNTnDg", args(&[7, 42]), &mut state),
        Some(SUCCESS)
    );
    assert!(crate::event_state::with_event_state(|e| e
        .kernel_event_registrations
        .is_empty()));
    assert!(crate::event_state::with_event_state(|e| e
        .pending_events
        .is_empty()));
    assert_eq!(
        dispatch("LJDwdSNTnDg", args(&[7, 42]), &mut state),
        Some(KERNEL_NOT_FOUND)
    );
    assert_eq!(
        dispatch("4R6-OvI2cEA", args(&[8, 42]), &mut state),
        Some(KERNEL_NOT_FOUND)
    );
}

#[test]
fn the_event_accessors_read_the_fields_at_their_wire_offsets() {
    let mut event = [0_u8; KERNEL_EVENT_SIZE];
    event[0x00..0x08].copy_from_slice(&0x1234_u64.to_le_bytes());
    // A filter is signed and every console filter is negative.
    event[0x08..0x0a].copy_from_slice(&(-11_i16).to_le_bytes());
    event[0x18..0x20].copy_from_slice(&0xdead_beef_u64.to_le_bytes());
    let mut state = ProbeDispatchState::default();
    let event_address = address(&event);

    assert_eq!(
        dispatch("23CPPI1tyBY", args(&[event_address]), &mut state),
        Some(signed_i32_return(-11))
    );
    assert_eq!(
        dispatch("vz+pg2zdopI", args(&[event_address]), &mut state),
        Some(0xdead_beef)
    );
    assert_eq!(dispatch("23CPPI1tyBY", [0; 6], &mut state), Some(0));
    assert_eq!(dispatch("vz+pg2zdopI", [0; 6], &mut state), Some(0));
}
