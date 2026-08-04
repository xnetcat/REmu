// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Guest module lifecycle outside `ProbeDispatchState` (F4).
//!
//! `sceKernelLoadStartModule` must not publish a module as started before its constructors
//! run: a second thread that `dlsym`s during that window would resolve into a half-constructed
//! image. Lifecycle phases and the waitable registry live here so export visibility is gated on
//! `Started`, independent of the process-wide dispatch lock.
//!
//! SDK: `SceKernelModule sceKernelLoadStartModule(...)` / `sceKernelDlsym` (`_kernel.h:79-85`);
//! `module_start` failure is `SCE_KERNEL_ERROR_ESTART` (`sys/sce_errno.h` / `ESTART` in
//! `sys/errno.h:194`).

use remu_core::GuestAddress;
use std::collections::BTreeMap;
use std::sync::{Condvar, Mutex, OnceLock};

/// Per-module lifecycle phases. Only `Started` modules expose exports to `dlsym`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GuestModulePhase {
    #[allow(dead_code)] // Matched in load/start; construction lands with finer load staging.
    Mapped,
    Relocated,
    Starting,
    Started,
    #[allow(dead_code)] // Matched in dlsym gate; unload path constructs later.
    Stopping,
    #[allow(dead_code)] // Matched in dlsym gate; unload path constructs later.
    Stopped,
}

/// One guest module tracked by the process-wide registry.
#[derive(Clone, Debug)]
pub(crate) struct GuestModule {
    pub(crate) name: String,
    pub(crate) handle: u32,
    pub(crate) exports_by_nid: BTreeMap<String, u64>,
    pub(crate) initializers: Vec<GuestAddress>,
    pub(crate) phase: GuestModulePhase,
    pub(crate) refcount: u32,
}

#[derive(Default)]
struct GuestModuleRegistry {
    by_handle: BTreeMap<u32, GuestModule>,
}

struct GuestModuleRegistryState {
    registry: GuestModuleRegistry,
}

static GUEST_MODULE_REGISTRY: OnceLock<(Mutex<GuestModuleRegistryState>, Condvar)> =
    OnceLock::new();

fn registry_pair() -> &'static (Mutex<GuestModuleRegistryState>, Condvar) {
    GUEST_MODULE_REGISTRY.get_or_init(|| {
        (
            Mutex::new(GuestModuleRegistryState {
                registry: GuestModuleRegistry::default(),
            }),
            Condvar::new(),
        )
    })
}

/// Clears the registry and installs relocated (or already-started) modules from a fresh link.
#[cfg_attr(not(any(test, target_arch = "x86_64")), allow(dead_code))]
pub(crate) fn reset_guest_module_registry(modules: impl IntoIterator<Item = GuestModule>) {
    let (lock, condvar) = registry_pair();
    let mut state = lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.registry.by_handle.clear();
    for module in modules {
        state.registry.by_handle.insert(module.handle, module);
    }
    condvar.notify_all();
}

/// Looks up a bundled module by path or basename. Returns handle and whether constructors must run.
pub(crate) fn begin_load_start_module(
    requested_path: &str,
    requested_name: &str,
) -> Option<LoadStartClaim> {
    let (lock, condvar) = registry_pair();
    let mut state = lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        #[allow(clippy::question_mark)] // `?` would exit the wait-loop; keep explicit None return.
        let Some(handle) = state
            .registry
            .by_handle
            .values()
            .find(|module| {
                module.name.eq_ignore_ascii_case(requested_path)
                    || std::path::Path::new(&module.name)
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.eq_ignore_ascii_case(requested_name))
            })
            .map(|module| module.handle)
        else {
            return None;
        };
        let module = state
            .registry
            .by_handle
            .get_mut(&handle)
            .expect("handle was just resolved from the registry");
        match module.phase {
            GuestModulePhase::Starting => {
                state = condvar
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                continue;
            }
            GuestModulePhase::Started => {
                module.refcount = module.refcount.saturating_add(1);
                return Some(LoadStartClaim {
                    handle,
                    name: module.name.clone(),
                    initializers: Vec::new(),
                });
            }
            GuestModulePhase::Mapped | GuestModulePhase::Relocated => {
                module.phase = GuestModulePhase::Starting;
                module.refcount = module.refcount.saturating_add(1);
                return Some(LoadStartClaim {
                    handle,
                    name: module.name.clone(),
                    initializers: module.initializers.clone(),
                });
            }
            GuestModulePhase::Stopping | GuestModulePhase::Stopped => {
                return None;
            }
        }
    }
}

/// Completes a start attempt: success → `Started`, failure → back to `Relocated`.
pub(crate) fn finish_load_start_module(handle: u32, succeeded: bool) {
    let (lock, condvar) = registry_pair();
    let mut state = lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(module) = state.registry.by_handle.get_mut(&handle)
        && module.phase == GuestModulePhase::Starting
    {
        module.phase = if succeeded {
            GuestModulePhase::Started
        } else {
            // Constructors failed (`SCE_KERNEL_ERROR_ESTART`). Keep the module relocated so a
            // later LoadStartModule can retry; exports stay invisible.
            GuestModulePhase::Relocated
        };
    }
    condvar.notify_all();
}

/// Resolves an export only once the module has reached `Started`.
///
/// Callers blocked while the module is `Starting` wait on the registry condvar so they never
/// observe a half-constructed export table as live.
pub(crate) fn resolve_started_export(handle: u32, nid: &str) -> Option<u64> {
    let (lock, condvar) = registry_pair();
    let mut state = lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        let module = state.registry.by_handle.get(&handle)?;
        match module.phase {
            GuestModulePhase::Starting => {
                state = condvar
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                continue;
            }
            GuestModulePhase::Started => {
                return module.exports_by_nid.get(nid).copied();
            }
            GuestModulePhase::Mapped
            | GuestModulePhase::Relocated
            | GuestModulePhase::Stopping
            | GuestModulePhase::Stopped => {
                return None;
            }
        }
    }
}

/// Snapshot of a module's current phase (test/diagnostics helper).
#[cfg(test)]
pub(crate) fn guest_module_phase(handle: u32) -> Option<GuestModulePhase> {
    let (lock, _) = registry_pair();
    let state = lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state
        .registry
        .by_handle
        .get(&handle)
        .map(|module| module.phase)
}

/// Non-blocking export probe used by tests to assert Starting hides exports.
#[cfg(test)]
fn exports_visible_without_waiting(handle: u32, nid: &str) -> bool {
    let (lock, _) = registry_pair();
    let state = lock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.registry.by_handle.get(&handle).is_some_and(|module| {
        module.phase == GuestModulePhase::Started && module.exports_by_nid.contains_key(nid)
    })
}

pub(crate) struct LoadStartClaim {
    pub(crate) handle: u32,
    pub(crate) name: String,
    pub(crate) initializers: Vec<GuestAddress>,
}

#[cfg(test)]
mod tests {
    use super::{
        GuestModule, GuestModulePhase, begin_load_start_module, finish_load_start_module,
        guest_module_phase, reset_guest_module_registry, resolve_started_export,
    };
    use remu_core::GuestAddress;
    use std::collections::BTreeMap;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    fn sample_module(phase: GuestModulePhase) -> GuestModule {
        let mut exports = BTreeMap::new();
        exports.insert("TestNid00001".to_owned(), 0x1000);
        GuestModule {
            name: "Media/Plugins/test.prx".to_owned(),
            handle: 2,
            exports_by_nid: exports,
            initializers: vec![GuestAddress::new(0x2000)],
            phase,
            refcount: 0,
        }
    }

    #[test]
    fn load_start_keeps_exports_invisible_until_constructors_finish() {
        reset_guest_module_registry([sample_module(GuestModulePhase::Relocated)]);
        let claim = begin_load_start_module("Media/Plugins/test.prx", "test.prx")
            .expect("relocated module can start");
        assert_eq!(claim.handle, 2);
        assert_eq!(claim.initializers.len(), 1);
        // Same-thread resolve would wait on Starting; the phase gate is the contract under test.
        assert_eq!(guest_module_phase(2), Some(GuestModulePhase::Starting));
        assert!(!super::exports_visible_without_waiting(2, "TestNid00001"));
        finish_load_start_module(2, true);
        assert_eq!(guest_module_phase(2), Some(GuestModulePhase::Started));
        assert_eq!(resolve_started_export(2, "TestNid00001"), Some(0x1000));
    }

    #[test]
    fn failed_start_returns_module_to_relocated_without_export_visibility() {
        reset_guest_module_registry([sample_module(GuestModulePhase::Relocated)]);
        let _ = begin_load_start_module("test.prx", "test.prx").expect("start");
        finish_load_start_module(2, false);
        assert_eq!(guest_module_phase(2), Some(GuestModulePhase::Relocated));
        assert_eq!(resolve_started_export(2, "TestNid00001"), None);
    }

    #[test]
    fn dlsym_waits_out_starting_phase_instead_of_seeing_half_constructed_exports() {
        reset_guest_module_registry([sample_module(GuestModulePhase::Relocated)]);
        let barrier = Arc::new(Barrier::new(2));
        let claim = begin_load_start_module("test.prx", "test.prx").expect("start");
        assert_eq!(claim.handle, 2);

        let barrier_for_waiter = Arc::clone(&barrier);
        let waiter = thread::spawn(move || {
            barrier_for_waiter.wait();
            resolve_started_export(2, "TestNid00001")
        });

        barrier.wait();
        thread::sleep(Duration::from_millis(20));
        assert_eq!(guest_module_phase(2), Some(GuestModulePhase::Starting));
        finish_load_start_module(2, true);
        assert_eq!(waiter.join().expect("waiter thread"), Some(0x1000));
    }

    #[test]
    fn second_load_start_after_started_skips_constructors() {
        reset_guest_module_registry([sample_module(GuestModulePhase::Relocated)]);
        let first = begin_load_start_module("test.prx", "test.prx").expect("first");
        finish_load_start_module(first.handle, true);
        let second = begin_load_start_module("test.prx", "test.prx").expect("second");
        assert!(second.initializers.is_empty());
        assert_eq!(guest_module_phase(2), Some(GuestModulePhase::Started));
    }
}
