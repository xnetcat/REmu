// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Character encoding service (CES) profile and context initialization.
//!
//! Titles use CES to convert between UCS and MBCS encodings for save titles and UI strings.
//! This module owns the profile/context bookkeeping those conversions require. The conversion
//! entry points remain unserved until a title reaches them with a concrete buffer contract.

use crate::{
    ProbeDispatchState, SCE_KERNEL_ERROR_INVALID_ARGUMENT, guest_memset, sce_kernel_error,
};

/// Bytes one UCS profile occupies for the SJIS / CP932 initializer Void Terrarium calls.
const CES_UCS_PROFILE_SIZE: u64 = 0x100;
/// Bytes one MBCS↔UCS conversion context occupies.
const CES_MBCS_UCS_CONTEXT_SIZE: u64 = 0x100;

/// `sceCesUcsProfileInitSJis1997Cp932(profile)`.
pub(crate) fn ces_ucs_profile_init_sjis1997_cp932(arguments: [u64; 6]) -> u64 {
    let profile = arguments[0];
    if profile == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    guest_memset(profile, 0, CES_UCS_PROFILE_SIZE);
    // Profile magic / version markers observed from the console initializer.
    crate::write_guest_u32(profile, 0x4353_0001);
    0
}

/// `sceCesMbcsUcsContextInit(context, profile)`.
///
/// Some titles pass a null profile after a successful profile initializer and expect the context
/// to bind the last initialized / default SJIS profile rather than reject the call.
pub(crate) fn ces_mbcs_ucs_context_init(arguments: [u64; 6]) -> u64 {
    let (context, profile) = (arguments[0], arguments[1]);
    if context == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    guest_memset(context, 0, CES_MBCS_UCS_CONTEXT_SIZE);
    crate::write_guest_u64(context, profile);
    0
}

/// Routes the CES entry points reached during Void Terrarium startup.
pub(crate) fn dispatch(nid: &str, arguments: [u64; 6], _state: &mut ProbeDispatchState) -> Option<u64> {
    Some(match nid {
        "ZiDCxUUGbec" => ces_ucs_profile_init_sjis1997_cp932(arguments),
        "538bRGc6Zo8" => ces_mbcs_ucs_context_init(arguments),
        _ => return None,
    })
}
