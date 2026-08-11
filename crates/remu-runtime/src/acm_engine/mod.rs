// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! PlayStation 5 Audio Compute Manager service.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::{guest_output_pointer, low_u32, write_guest_u32};

use self::abi::{AcmError, return_error};

mod abi;
mod batch;
mod builders;

pub(crate) const SDK10_ACM_NIDS: [&str; 13] = [
    "ZIXln2K3XMk",
    "jBgBjAj02R8",
    "tW9W+CAG4FE",
    "8fe55ktlNVo",
    "RLN3gRlXJLE",
    "r7z5YQFZo+U",
    "KovqaFbmtsM",
    "DR-ZCmvVR9Q",
    "LA4RCNKnFjg",
    "u70oWo92SYQ",
    "9nLbWmRDpa8",
    "wM41Cc+ieog",
    "PVx9oQUaDdk",
];

pub(crate) fn owns_nid(nid: &str) -> bool {
    SDK10_ACM_NIDS.contains(&nid)
}

#[derive(Default)]
struct ContextState;

#[derive(Default)]
struct EngineState {
    next_context: u32,
    contexts: BTreeMap<u32, ContextState>,
}

pub(crate) struct AcmEngine {
    state: Mutex<EngineState>,
}

static ENGINE: OnceLock<AcmEngine> = OnceLock::new();

impl AcmEngine {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(EngineState::default()),
        }
    }

    fn global() -> &'static Self {
        ENGINE.get_or_init(Self::new)
    }

    fn lock(&self) -> MutexGuard<'_, EngineState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn context_create(&self, arguments: [u64; 6]) -> u64 {
        let output = arguments[0];
        if guest_output_pointer(output, std::mem::size_of::<u32>()).is_none() {
            return return_error(AcmError::InvalidParameter);
        }

        let mut state = self.lock();
        let context = state.next_context.max(1);
        if context == u32::MAX {
            return return_error(AcmError::OutOfResources);
        }
        state.next_context = context + 1;
        state.contexts.insert(context, ContextState);
        drop(state);
        write_guest_u32(output, context);
        0
    }

    fn context_destroy(&self, arguments: [u64; 6]) -> u64 {
        let context = low_u32(arguments[0]);
        if self.lock().contexts.remove(&context).is_none() {
            return return_error(AcmError::InvalidContext);
        }
        0
    }

    fn reset(&self) {
        *self.lock() = EngineState::default();
    }
}

pub(crate) fn dispatch(nid: &str, arguments: [u64; 6], _stack_arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "ZIXln2K3XMk" => AcmEngine::global().context_create(arguments),
        "jBgBjAj02R8" => AcmEngine::global().context_destroy(arguments),
        "tW9W+CAG4FE" | "8fe55ktlNVo" | "RLN3gRlXJLE" => {
            let context = low_u32(arguments[0]);
            if !AcmEngine::global().lock().contexts.contains_key(&context) {
                return Some(return_error(AcmError::InvalidContext));
            }
            return Some(return_error(AcmError::InvalidParameter));
        }
        "r7z5YQFZo+U" | "KovqaFbmtsM" | "DR-ZCmvVR9Q" | "LA4RCNKnFjg" | "u70oWo92SYQ"
        | "9nLbWmRDpa8" | "wM41Cc+ieog" | "PVx9oQUaDdk" => {
            return Some(return_error(AcmError::InvalidParameter));
        }
        _ => return None,
    })
}

#[cfg(any(test, target_arch = "x86_64"))]
pub(crate) fn reset_for_new_process() {
    if let Some(engine) = ENGINE.get() {
        engine.reset();
    }
}

#[cfg(test)]
mod tests;
