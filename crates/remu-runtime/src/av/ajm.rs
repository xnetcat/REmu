// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::{
    ProbeDispatchState, SCE_AJM_ERROR_INVALID_CONTEXT, SCE_AJM_ERROR_INVALID_PARAMETER,
    guest_malloc, low_i32, low_u32, sce_kernel_error, write_guest_bytes,
};
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

/// Error the audio-job-manager entry points here add to the ones the main table already reports.
pub(super) const SCE_AJM_ERROR_OUT_OF_RESOURCES: u32 = 0x8093_0007;

/// Bytes one page of memory published to the audio coprocessor covers.
pub(super) const AJM_MEMORY_PAGE_SIZE: u64 = 2 * 1024 * 1024;
/// Pages one context may publish.
///
/// No console ceiling is published for this. The bound keeps a runaway caller from growing the
/// region table without limit, and reports the same exhaustion error the hardware would.
pub(super) const AJM_MEMORY_PAGE_LIMIT: u64 = 512;

/// Regions each context has published, keyed by context and starting address.
pub(super) static AJM_REGIONS: OnceLock<Mutex<BTreeMap<(u32, u64), u64>>> = OnceLock::new();

/// Returns the process-wide table of published regions.
pub(super) fn ajm_regions() -> std::sync::MutexGuard<'static, BTreeMap<(u32, u64), u64>> {
    AJM_REGIONS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Declares a region of guest memory available to the audio coprocessor.
pub(super) fn ajm_memory_register(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (context, region, pages) = (low_u32(arguments[0]), arguments[1], arguments[2]);
    if !state.ajm_contexts.contains_key(&context) {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
    }
    let Some(length) = pages.checked_mul(AJM_MEMORY_PAGE_SIZE) else {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    };
    if region == 0
        || pages == 0
        || !region.is_multiple_of(AJM_MEMORY_PAGE_SIZE)
        || region.checked_add(length).is_none()
    {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    let mut regions = ajm_regions();
    // A finalized context takes its regions with it; the table is only consulted here.
    regions.retain(|&(owner, _), _| state.ajm_contexts.contains_key(&owner));
    let mut published = 0_u64;
    for (&(owner, start), &count) in regions.iter() {
        if owner != context {
            continue;
        }
        published = published.saturating_add(count);
        let end = start.saturating_add(count.saturating_mul(AJM_MEMORY_PAGE_SIZE));
        if region < end && start < region.saturating_add(length) {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
        }
    }
    if published.saturating_add(pages) > AJM_MEMORY_PAGE_LIMIT {
        return sce_kernel_error(SCE_AJM_ERROR_OUT_OF_RESOURCES);
    }
    regions.insert((context, region), pages);
    0
}

/// Withdraws a region previously published to the audio coprocessor.
pub(super) fn ajm_memory_unregister(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (context, region) = (low_u32(arguments[0]), arguments[1]);
    if !state.ajm_contexts.contains_key(&context) {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
    }
    if region == 0 {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    if ajm_regions().remove(&(context, region)).is_none() {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    0
}

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
