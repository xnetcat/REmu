// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Runtime-linker TLS destructor reference counts.
//!
//! `_sceKernelRtldThreadAtexitIncrement` / `Decrement` pin the module that owns a pending
//! `__cxa_thread_atexit` destructor so `dlclose` cannot unload it while a destructor can still
//! run. The argument is a code/data address inside the owning module (typically `__dso_handle`),
//! matching fpPS4's `AcqureFileByCodeAdr` behaviour and glibc's link-map TLS dtor count.

use crate::ProbeDispatchState;

/// Acquire a TLS-destructor reference for the module containing `dso_address`.
///
/// Returns `0` on success, or `-1` when no loaded module covers the address.
pub(crate) fn rtld_thread_atexit_increment(
    dso_address: u64,
    state: &mut ProbeDispatchState,
) -> u64 {
    let Some(module_base) = module_base_containing(dso_address, state) else {
        eprintln!(
            "[REMu][RTLD] thread-atexit increment: no module contains address={dso_address:#x}"
        );
        return (-1_i64) as u64;
    };
    let count = state
        .module_tls_destructor_refs
        .entry(module_base)
        .or_insert(0);
    *count = count.saturating_add(1);
    0
}

/// Release a TLS-destructor reference previously taken for the module containing `dso_address`.
///
/// Mirrors fpPS4: resolve the module, then drop one reference. Missing modules are a no-op.
pub(crate) fn rtld_thread_atexit_decrement(dso_address: u64, state: &mut ProbeDispatchState) -> u64 {
    let Some(module_base) = module_base_containing(dso_address, state) else {
        eprintln!(
            "[REMu][RTLD] thread-atexit decrement: no module contains address={dso_address:#x}"
        );
        return 0;
    };
    match state.module_tls_destructor_refs.get_mut(&module_base) {
        Some(count) if *count > 0 => {
            *count -= 1;
            if *count == 0 {
                state.module_tls_destructor_refs.remove(&module_base);
            }
        }
        _ => {
            eprintln!(
                "[REMu][RTLD] thread-atexit decrement underflow module_base={module_base:#x} address={dso_address:#x}"
            );
        }
    }
    0
}

fn module_base_containing(address: u64, state: &ProbeDispatchState) -> Option<u64> {
    state
        .unwind_modules
        .iter()
        .find(|module| (module.base..module.end).contains(&address))
        .map(|module| module.base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unwind::GuestUnwindModule;

    fn state_with_module(base: u64, end: u64) -> ProbeDispatchState {
        let mut state = ProbeDispatchState::default();
        state.unwind_modules.push(GuestUnwindModule {
            name: "eboot.bin".to_owned(),
            base,
            end,
            eh_frame_header: 0,
            eh_frame: 0,
            eh_frame_size: 0,
            segments: Vec::new(),
        });
        state
    }

    #[test]
    fn increment_pins_module_and_decrement_releases() {
        let mut state = state_with_module(0x1000_0000, 0x1001_0000);
        assert_eq!(rtld_thread_atexit_increment(0x1000_0100, &mut state), 0);
        assert_eq!(
            state.module_tls_destructor_refs.get(&0x1000_0000),
            Some(&1)
        );
        assert_eq!(rtld_thread_atexit_increment(0x1000_0200, &mut state), 0);
        assert_eq!(
            state.module_tls_destructor_refs.get(&0x1000_0000),
            Some(&2)
        );
        assert_eq!(rtld_thread_atexit_decrement(0x1000_0100, &mut state), 0);
        assert_eq!(
            state.module_tls_destructor_refs.get(&0x1000_0000),
            Some(&1)
        );
        assert_eq!(rtld_thread_atexit_decrement(0x1000_0200, &mut state), 0);
        assert!(state.module_tls_destructor_refs.is_empty());
    }

    #[test]
    fn unknown_address_returns_minus_one_on_increment() {
        let mut state = state_with_module(0x1000_0000, 0x1001_0000);
        assert_eq!(
            rtld_thread_atexit_increment(0x2000_0000, &mut state) as i32,
            -1
        );
        assert!(state.module_tls_destructor_refs.is_empty());
    }
}
