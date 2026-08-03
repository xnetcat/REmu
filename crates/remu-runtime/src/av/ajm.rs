// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Stateless `libSceAjm` helpers. The engine itself — contexts, instances, batches, memory
//! registration — lives in `crate::ajm_engine`; this module keeps the error-string surface
//! that needs no engine state.

use crate::{guest_malloc, low_i32, write_guest_bytes};
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

/// Returns the description a code carries.
pub(super) fn ajm_error_text(code: i32) -> &'static str {
    match code.cast_unsigned() {
        0 => "no error",
        0x8093_0001 => "unknown error",
        0x8093_0002 => "invalid context",
        0x8093_0003 => "invalid instance",
        0x8093_0004 => "invalid batch",
        0x8093_0005 => "invalid parameter",
        0x8093_0006 => "out of memory",
        0x8093_0007 => "out of resources",
        0x8093_0008 => "codec not supported",
        0x8093_0009 => "codec already registered",
        0x8093_000a => "codec not registered",
        0x8093_000b => "wrong revision flag",
        0x8093_000c => "flag not supported",
        0x8093_000d => "busy",
        0x8093_000e => "bad priority",
        0x8093_000f => "in progress",
        0x8093_0010 => "retry",
        0x8093_0011 => "malformed batch",
        0x8093_0012 => "job creation failed",
        0x8093_0013 => "invalid opcode",
        0x8093_0014 => "priority violation",
        0x8093_0015 => "buffer too big",
        0x8093_0016 => "invalid address",
        0x8093_0017 => "cancelled",
        0x8093_0018 => "memory already registered",
        0x8093_0019 => "memory not registered",
        0x8093_001a => "instance memory already extended",
        _ => "unrecognized error code",
    }
}

/// Descriptions already published to the guest, so one code always reports one address.
pub(super) static AJM_ERROR_STRINGS: OnceLock<Mutex<BTreeMap<i32, u64>>> = OnceLock::new();

/// Reports a readable description of an error code.
pub(super) fn ajm_str_error(code: u64) -> u64 {
    let code = low_i32(code);
    let mut published = AJM_ERROR_STRINGS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(&address) = published.get(&code) {
        return address;
    }
    let text = ajm_error_text(code);
    let length = u64::try_from(text.len()).unwrap_or(0) + 1;
    let address = guest_malloc(length);
    if address == 0 {
        return 0;
    }
    let mut bytes = text.as_bytes().to_vec();
    bytes.push(0);
    write_guest_bytes(address, &bytes);
    published.insert(code, address);
    address
}
