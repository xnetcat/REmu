// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;
use crate::ajm_engine::SCE_AJM_ERROR_INVALID_PARAMETER;

#[test]
fn an_error_code_is_described_at_one_address_the_guest_can_read() {
    let mut state = ProbeDispatchState::default();
    let Some(address) = dispatch(
        "AxhcqVv5AYU",
        args(&[u64::from(SCE_AJM_ERROR_INVALID_PARAMETER)]),
        &mut state,
    ) else {
        panic!("the module owns this NID");
    };
    assert_ne!(address, 0);
    assert_eq!(
        crate::read_guest_cstring(address).as_deref(),
        Some("invalid parameter")
    );
    assert_eq!(
        dispatch(
            "AxhcqVv5AYU",
            args(&[u64::from(SCE_AJM_ERROR_INVALID_PARAMETER)]),
            &mut state
        ),
        Some(address),
        "one code must always describe itself at one address"
    );
    let Some(unknown) = dispatch("AxhcqVv5AYU", args(&[0x8093_7fff]), &mut state) else {
        panic!("the module owns this NID");
    };
    assert_eq!(
        crate::read_guest_cstring(unknown).as_deref(),
        Some("unrecognized error code")
    );
}
