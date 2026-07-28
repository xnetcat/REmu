// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;

#[test]
fn published_coprocessor_memory_must_be_page_aligned_and_owned_by_a_live_context() {
    let mut state = ProbeDispatchState::default();
    state
        .ajm_contexts
        .insert(AJM_TEST_CONTEXT, AjmContextState::default());
    let region = 0x1_0000_0000_u64;
    assert_eq!(
        dispatch(
            "bkRHEYG6lEM",
            args(&[u64::from(AJM_TEST_CONTEXT), region + 1, 1]),
            &mut state
        ),
        Some(error(SCE_AJM_ERROR_INVALID_PARAMETER))
    );
    assert_eq!(
        dispatch(
            "bkRHEYG6lEM",
            args(&[u64::from(AJM_TEST_CONTEXT), region, 0]),
            &mut state
        ),
        Some(error(SCE_AJM_ERROR_INVALID_PARAMETER))
    );
    assert_eq!(
        dispatch("bkRHEYG6lEM", args(&[0, region, 1]), &mut state),
        Some(error(SCE_AJM_ERROR_INVALID_CONTEXT))
    );
    assert_eq!(
        dispatch(
            "bkRHEYG6lEM",
            args(&[u64::from(AJM_TEST_CONTEXT), region, 2]),
            &mut state
        ),
        Some(SUCCESS)
    );
    assert_eq!(
        dispatch(
            "bkRHEYG6lEM",
            args(&[
                u64::from(AJM_TEST_CONTEXT),
                region + AJM_MEMORY_PAGE_SIZE,
                1
            ]),
            &mut state
        ),
        Some(error(SCE_AJM_ERROR_INVALID_PARAMETER)),
        "a region inside one already published must be refused"
    );
    assert_eq!(
        dispatch(
            "bkRHEYG6lEM",
            args(&[u64::from(AJM_TEST_CONTEXT), region, AJM_MEMORY_PAGE_LIMIT]),
            &mut state
        ),
        Some(error(SCE_AJM_ERROR_INVALID_PARAMETER))
    );
    assert_eq!(
        dispatch(
            "pIpGiaYkHkM",
            args(&[u64::from(AJM_TEST_CONTEXT), region + AJM_MEMORY_PAGE_SIZE]),
            &mut state
        ),
        Some(error(SCE_AJM_ERROR_INVALID_PARAMETER))
    );
    assert_eq!(
        dispatch(
            "pIpGiaYkHkM",
            args(&[u64::from(AJM_TEST_CONTEXT), region]),
            &mut state
        ),
        Some(SUCCESS)
    );
    assert_eq!(
        dispatch(
            "pIpGiaYkHkM",
            args(&[u64::from(AJM_TEST_CONTEXT), region]),
            &mut state
        ),
        Some(error(SCE_AJM_ERROR_INVALID_PARAMETER))
    );
}

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
