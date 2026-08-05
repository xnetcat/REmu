// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Sync-object identity from the Gen5 SDK ABI.
//!
//! `sys/_pthreadtypes.h` defines:
//! ```c
//! typedef struct pthread_mutex  *pthread_mutex_t;
//! typedef struct pthread_cond   *pthread_cond_t;
//! typedef struct pthread_rwlock *pthread_rwlock_t;
//! ```
//! and `kernel/semaphore.h` defines `typedef struct _SceKernelSema* SceKernelSema`.
//!
//! Each HLE entry point therefore receives a pointer to an opaque control-block handle.
//! Identity is the handle value written by init (or by static-initializer materialization),
//! never the address of the slot the guest happened to pass. Stack copies of the same handle
//! (Demon's Souls / Scream) must resolve to one host object.
//!
//! Static initializers from `_pthread.h`:
//! - `SCE_PTHREAD_MUTEX_INITIALIZER` / `COND` / `RWLOCK` = `NULL` (0)
//! - `SCE_PTHREAD_ADAPTIVE_MUTEX_INITIALIZER` = `(pthread_mutex_t)1`

use crate::try_read_process_guest_u64;

/// `SCE_PTHREAD_*_INITIALIZER` / `NULL`.
pub(super) const PTHREAD_STATIC_INITIALIZER: u64 = 0;

/// `SCE_PTHREAD_ADAPTIVE_MUTEX_INITIALIZER`.
pub(super) const PTHREAD_ADAPTIVE_MUTEX_INITIALIZER: u64 = 1;

/// Opaque control-block size installed for mutex/cond/rwlock init. Libc may write bookkeeping
/// into the mutex object after `scePthreadMutexInit`; cond/rwlock keep the same allocation
/// contract so the written handle is always a real, unique guest pointer.
pub(super) const GUEST_PTHREAD_OBJECT_SIZE: u64 = 0x100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PthreadSlotValue {
    /// `*_INITIALIZER` — first use must allocate and publish a control block.
    StaticDefault,
    /// `SCE_PTHREAD_ADAPTIVE_MUTEX_INITIALIZER` only (mutex slots).
    StaticAdaptiveMutex,
    /// Already-published control-block pointer.
    ControlBlock(u64),
}

/// Classify a mutex slot. Adaptive static initializer (`1`) is mutex-only.
pub(super) fn classify_mutex_slot_value(value: u64) -> PthreadSlotValue {
    match value {
        PTHREAD_STATIC_INITIALIZER => PthreadSlotValue::StaticDefault,
        PTHREAD_ADAPTIVE_MUTEX_INITIALIZER => PthreadSlotValue::StaticAdaptiveMutex,
        handle => PthreadSlotValue::ControlBlock(handle),
    }
}

/// Classify a cond/rwlock slot. Only `NULL` is a static initializer; any other value is a handle.
pub(super) fn classify_pthread_slot_value(value: u64) -> PthreadSlotValue {
    match value {
        PTHREAD_STATIC_INITIALIZER => PthreadSlotValue::StaticDefault,
        handle => PthreadSlotValue::ControlBlock(handle),
    }
}

/// Read the handle stored at a `pthread_*_t *` argument when the slot is mapped guest memory.
pub(super) fn read_pthread_slot(slot: u64) -> Option<u64> {
    if slot == 0 {
        return None;
    }
    try_read_process_guest_u64(slot)
}
