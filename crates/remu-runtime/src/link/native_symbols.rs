// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
use crate::{
    current_guest_thread_handle, current_guest_thread_unique_id, guest_mutex_lock,
    guest_mutex_unlock, guest_rwlock_lock, guest_rwlock_unlock, kernel_process_time, low_u32,
    posix_pthread_result, pthread_get_specific, pthread_set_specific,
};
use std::collections::BTreeMap;
use std::sync::OnceLock;

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct NativeCxxRuntime {
    pub(super) cxxabi_handle: usize,
    pub(super) unwind_handle: usize,
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(crate) fn native_cxx_runtime() -> Option<NativeCxxRuntime> {
    static RUNTIME: OnceLock<Option<NativeCxxRuntime>> = OnceLock::new();
    *RUNTIME.get_or_init(|| {
        // SAFETY: Both paths are NUL-terminated constants naming macOS shared-cache libraries.
        let cxxabi_handle = unsafe {
            libc::dlopen(
                c"/usr/lib/libc++abi.dylib".as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            )
        };
        // SAFETY: Same as above; the handle is intentionally retained for the process lifetime.
        let unwind_handle = unsafe {
            libc::dlopen(
                c"/usr/lib/system/libunwind.dylib".as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            )
        };
        if cxxabi_handle.is_null() || unwind_handle.is_null() {
            None
        } else {
            Some(NativeCxxRuntime {
                cxxabi_handle: cxxabi_handle as usize,
                unwind_handle: unwind_handle as usize,
            })
        }
    })
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
impl NativeCxxRuntime {
    pub(super) fn cxxabi_symbol(self, name: &std::ffi::CStr) -> Option<u64> {
        // SAFETY: `cxxabi_handle` is a retained live dlopen handle and `name` is NUL-terminated.
        let address =
            unsafe { libc::dlsym(self.cxxabi_handle as *mut libc::c_void, name.as_ptr()) };
        (!address.is_null()).then_some(address as usize as u64)
    }

    pub(crate) fn unwind_symbol(self, name: &std::ffi::CStr) -> Option<u64> {
        // SAFETY: `unwind_handle` is a retained live dlopen handle and `name` is NUL-terminated.
        let address =
            unsafe { libc::dlsym(self.unwind_handle as *mut libc::c_void, name.as_ptr()) };
        (!address.is_null()).then_some(address as usize as u64)
    }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(super) fn native_cxx_symbol_name(nid: &str) -> Option<&'static std::ffi::CStr> {
    Some(match nid {
        "XwLA5cTHjt4" => c"__gxx_personality_v0",
        "IJKVjsmxxWI" => c"__cxa_allocate_dependent_exception",
        "cfAXurvfl5o" => c"__cxa_allocate_exception",
        "pBxafllkvt0" => c"__cxa_bad_cast",
        "xcc6DTcL8QA" => c"__cxa_bad_typeid",
        "3cUUypQzMiI" => c"__cxa_begin_catch",
        "BxPeH9TTcs4" => c"__cxa_current_exception_type",
        "lX+4FNUklF0" => c"__cxa_end_catch",
        "kBxt5LwtLA4" => c"__cxa_free_dependent_exception",
        "nOIEswYD4Ig" => c"__cxa_free_exception",
        "Y6Sl4Xw7gfA" => c"__cxa_get_exception_ptr",
        "3GPpjQdAMTw" => c"__cxa_guard_acquire",
        "9rAeANT2tyE" => c"__cxa_guard_release",
        "2emaaluWzUw" => c"__cxa_guard_abort",
        "zr094EQ39Ww" => c"__cxa_pure_virtual",
        "ZL9FV4mJXxo" => c"__cxa_rethrow",
        "vkuuLfhnSZI" => c"__cxa_throw",
        "XbAd9Ach9mo" => c"__cxa_throw_bad_array_new_length",
        "cmBRQsLPqwA" => c"__cxa_uncaught_exception",
        "qYhnoevd9bI" => c"_ZSt9terminatev",
        "ozMAr28BwSY" => c"_ZSt14_Xout_of_rangePKc",
        "eT2UsmTewbU" => c"_ZSt11_Xbad_allocv",
        "pZ9WXcClPO8" => c"_ZTVN10__cxxabiv120__si_class_type_infoE",
        "byV+FWlAnB4" => c"_ZTVN10__cxxabiv117__class_type_infoE",
        "tQIo+GIPklo" => c"_ZSt14_Xlength_errorPKc",
        "9ByRMdo7ywg" => c"_ZTVN10__cxxabiv121__vmi_class_type_infoE",
        "V23qt24VPVs" => c"_ZSt17iostream_categoryv",
        "tyHd3P7oDrU" => c"_ZNKSt9exception8_DoraiseEv",
        "Q1BL70XVV0o" => c"_ZSt18uncaught_exceptionv",
        "MELi-cKqWq0" => c"_ZSt19_Xbad_function_callv",
        "NU-T4QowTNA" => c"_ZSt18_Xinvalid_argumentPKc",
        "hMAe+TWS9mQ" => c"__dynamic_cast",
        "apPZ6HKZWaQ" => c"_ZNKSt9exception4whatEv",
        "7P1Wm-5KgAY" => c"_ZNSt12domain_errorD1Ev",
        "MOBxtefPZUg" => c"_ZNSt9exceptionD2Ev",
        "5BIbzIuDxTQ" => c"_ZTISt12domain_error",
        "oAidKrxuUv0" => c"_ZTVSt12domain_error",
        "udTM6Nxx-Ng" => c"_ZTVSt11logic_error",
        "N2f485TmJms" => c"_ZNSt8ios_base7failureD1Ev",
        "oe9tS0VztYk" => c"_ZNSt13runtime_errorD2Ev",
        "47RvLSo2HN8" => c"_ZNSt8bad_castD1Ev",
        "oQDS9nX05Qg" => c"_ZNSt16invalid_argumentD1Ev",
        "Kcb+MNSzZcc" => c"_ZNSt12out_of_rangeD1Ev",
        "BJCgW9-OxLA" => c"_ZTISt8ios_base",
        "Bq8m04PN1zw" => c"_ZTVSt12system_error",
        "-L+-8F0+gBc" => c"_ZTVSt13runtime_error",
        "bLPn1gfqSW8" => c"_ZTISt13runtime_error",
        "qOD-ksTkE08" => c"_ZTISt8bad_cast",
        "tVHE+C8vGXk" => c"_ZTVSt8bad_cast",
        "XZzWt0ygWdw" => c"_ZTISt16invalid_argument",
        "keXoyW-rV-0" => c"_ZTVSt16invalid_argument",
        "dKjhNUf9FBc" => c"_ZTISt12out_of_range",
        "n+aUKkC-3sI" => c"_ZTVSt12out_of_range",
        "n2kx+OmFUis" => c"_ZTISt9exception",
        "BgZcGDh7o9g" => c"_ZNSt9exceptionD1Ev",
        "dCzeFfg9WWI" => c"_ZTVSt9exception",
        _ => return None,
    })
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(super) fn native_unwind_symbol_name(nid: &str) -> Option<&'static std::ffi::CStr> {
    Some(match nid {
        "s62MgBhosjU" => c"_Unwind_Backtrace",
        "GslDM6l8E7U" => c"_Unwind_DeleteException",
        "go1LaO7f15k" => c"_Unwind_FindEnclosingFunction",
        "Jhba7poZWZ0" => c"_Unwind_Find_FDE",
        "zs817D8+hPA" => c"_Unwind_ForcedUnwind",
        "uKwYpqeUkzo" => c"_Unwind_GetBSP",
        "MmTshn7gIgU" => c"_Unwind_GetCFA",
        "nqD9bBu-W6U" => c"_Unwind_GetDataRelBase",
        "oQpl5qCT014" => c"_Unwind_GetGR",
        "sETNbyWsEHs" => c"_Unwind_GetIP",
        "5-3wdp3FVtc" => c"_Unwind_GetIPInfo",
        "gpc4YXZffkQ" => c"_Unwind_GetLanguageSpecificData",
        "p3BVovMPEE4" => c"_Unwind_GetRegionStart",
        "HPlmfRhMWwo" => c"_Unwind_GetTextRelBase",
        "wpNJwmDDtxw" => c"_Unwind_RaiseException",
        "f1zwJ3jAI2k" => c"_Unwind_Resume",
        "xUsJSLsdv9I" => c"_Unwind_Resume_or_Rethrow",
        "Xr4pio56GGQ" => c"_Unwind_SetGR",
        "BWAtsvaXToA" => c"_Unwind_SetIP",
        _ => return None,
    })
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(super) fn native_libc_symbol_name(nid: &str) -> Option<&'static std::ffi::CStr> {
    Some(match nid {
        "8zTFvBIAIN8" => c"memset",
        "Q3VBxCXhUHs" => c"memcpy",
        "+P6FRGH4LfA" => c"memmove",
        "DfivPArhucg" => c"memcmp",
        "8u8lPzUEq+U" => c"memchr",
        "j4ViWNHEgww" => c"strlen",
        "Ovb2dSJOAuE" => c"strcmp",
        "aesyjrHVWy4" => c"strncmp",
        "kiZSXIWd9vg" => c"strcpy",
        "6sJWiWSRuqk" => c"strncpy",
        "Ls4tzzhimqQ" => c"strcat",
        "AV6ipCNa4Rw" => c"strcasecmp",
        "pXvbDfchu6k" => c"strncasecmp",
        "viiwFMaNamA" => c"strstr",
        "g7zzzLDYGw0" => c"strdup",
        "SfQIZcqvvms" => c"strlcpy",
        "5OqszGpy7Mg" => c"strtoull",
        "VOBg+iNwB-4" => c"strtoll",
        "2vDqwBlpF-o" => c"strtod",
        "dhK16CKwhQg" => c"__isfinited",
        "GfxAp9Xyiqs" => c"__isnand",
        "eLdDw6l0-bU" => c"snprintf",
        "Q2V+iqvjgC0" => c"vsnprintf",
        "tcVi5SivF7Q" => c"sprintf",
        "1Pk0qZQGeWo" => c"sscanf",
        "hcuQgD53UxM" => c"printf",
        "YQ0navp+YIc" => c"puts",
        "HUbZmOnT-Dg" => c"atan2",
        "5ZkEP3Rq7As" => c"cbrt",
        "JBcgYuW8lPU" => c"acos",
        "7Ly52zaL44Q" => c"asin",
        "OXmauLdQ8kY" => c"atan",
        "T7uyNqP7vQA" => c"tan",
        "kn0yiYeExgA" => c"ldexpf",
        "weDug8QD-lE" => c"atanf",
        "wuAQt-j+p4o" => c"exp2f",
        "hsi9drzHR2k" => c"log2f",
        "1D0H2KNjshE" => c"powf",
        "nlaojL9hDtA" => c"round",
        "DDHG1a6+3q0" => c"roundf",
        "9LCjpWyQ5Zc" => c"pow",
        "jMB7EFyu30Y" => c"sincos",
        "pKwslsMUmSk" => c"fmod",
        "RQXLbdT2lc4" => c"logf",
        "pztV4AF18iI" => c"sincosf",
        "H8ya2H00jbI" => c"sin",
        "Q4rRL34CEeE" => c"sinf",
        "rtV7-jWC6Yg" => c"log",
        "WuMbPBKN1TU" => c"log10",
        "2WE3BTYVwKM" => c"cos",
        "Y5DhuDKGlnQ" => c"log2",
        "NVadfnzQhHQ" => c"exp",
        "8zsu04XNsZ4" => c"expf",
        "-P6FNMzk2Kc" => c"cosf",
        _ => return None,
    })
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn record_direct_runtime_import(import: remu_gpu::DirectDispatchImport) {
    remu_gpu::record_direct_dispatch_import(import);
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_sce_pthread_self() -> u64 {
    record_direct_runtime_import(remu_gpu::DirectDispatchImport::ScePthreadSelf);
    current_guest_thread_handle()
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_posix_pthread_self() -> u64 {
    record_direct_runtime_import(remu_gpu::DirectDispatchImport::PosixPthreadSelf);
    current_guest_thread_handle()
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_sce_pthread_get_thread_id() -> u64 {
    record_direct_runtime_import(remu_gpu::DirectDispatchImport::ScePthreadGetThreadId);
    current_guest_thread_unique_id()
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_posix_pthread_get_specific(key: u64) -> u64 {
    record_direct_runtime_import(remu_gpu::DirectDispatchImport::PosixPthreadGetSpecific);
    pthread_get_specific(low_u32(key))
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_sce_pthread_get_specific(key: u64) -> u64 {
    record_direct_runtime_import(remu_gpu::DirectDispatchImport::ScePthreadGetSpecific);
    pthread_get_specific(low_u32(key))
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_posix_pthread_set_specific(key: u64, value: u64) -> u64 {
    record_direct_runtime_import(remu_gpu::DirectDispatchImport::PosixPthreadSetSpecific);
    posix_pthread_result(pthread_set_specific([key, value, 0, 0, 0, 0]))
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_sce_pthread_set_specific(key: u64, value: u64) -> u64 {
    record_direct_runtime_import(remu_gpu::DirectDispatchImport::ScePthreadSetSpecific);
    pthread_set_specific([key, value, 0, 0, 0, 0])
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_sce_guest_mutex_lock(address: u64) -> u64 {
    record_direct_runtime_import(remu_gpu::DirectDispatchImport::ScePthreadMutexLock);
    guest_mutex_lock(address, false)
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_posix_guest_mutex_lock(address: u64) -> u64 {
    record_direct_runtime_import(remu_gpu::DirectDispatchImport::PosixPthreadMutexLock);
    guest_mutex_lock(address, false)
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_sce_guest_mutex_unlock(address: u64) -> u64 {
    record_direct_runtime_import(remu_gpu::DirectDispatchImport::ScePthreadMutexUnlock);
    guest_mutex_unlock(address)
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_posix_guest_mutex_unlock(address: u64) -> u64 {
    record_direct_runtime_import(remu_gpu::DirectDispatchImport::PosixPthreadMutexUnlock);
    guest_mutex_unlock(address)
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_guest_rwlock_write_lock(address: u64) -> u64 {
    record_direct_runtime_import(remu_gpu::DirectDispatchImport::PosixPthreadRwlockWriteLock);
    guest_rwlock_lock(address, true, false)
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_guest_rwlock_unlock(address: u64) -> u64 {
    record_direct_runtime_import(remu_gpu::DirectDispatchImport::PosixPthreadRwlockUnlock);
    guest_rwlock_unlock(address)
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_process_time_counter() -> u64 {
    record_direct_runtime_import(remu_gpu::DirectDispatchImport::KernelGetProcessTimeCounter);
    kernel_process_time(true)
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(super) fn direct_runtime_import_target(nid: &str) -> Option<u64> {
    Some(match nid {
        // Unreal's archive/name parser calls these leaf TLS and synchronization operations
        // millions of times during startup. They already use dedicated sharded/thread-local
        // state and need neither the process dispatch lock nor a captured eight-XMM import frame.
        "aI+OeCz8xrQ" => direct_sce_pthread_self as *const () as usize as u64,
        "EotR8a3ASf4" => direct_posix_pthread_self as *const () as usize as u64,
        "0-KXaS70xy4" => direct_posix_pthread_get_specific as *const () as usize as u64,
        "eoht7mQOCmo" => direct_sce_pthread_get_specific as *const () as usize as u64,
        "WrOLvHU0yQM" => direct_posix_pthread_set_specific as *const () as usize as u64,
        "+BzXYkqYeLE" => direct_sce_pthread_set_specific as *const () as usize as u64,
        "9UK1vLZQft4" => direct_sce_guest_mutex_lock as *const () as usize as u64,
        "7H0iTOciTLo" => direct_posix_guest_mutex_lock as *const () as usize as u64,
        "tn3VlD0hG60" => direct_sce_guest_mutex_unlock as *const () as usize as u64,
        "2Z+PpY6CaJg" => direct_posix_guest_mutex_unlock as *const () as usize as u64,
        "sIlRvQqsN2Y" => direct_guest_rwlock_write_lock as *const () as usize as u64,
        "EgmLo6EWgso" => direct_guest_rwlock_unlock as *const () as usize as u64,
        "EI-5-jlq2dE" => direct_sce_pthread_get_thread_id as *const () as usize as u64,
        "fgxnMeTNUtY" => direct_process_time_counter as *const () as usize as u64,
        _ => return None,
    })
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(super) fn native_import_target(nid: &str) -> Option<u64> {
    if let Some(target) = direct_runtime_import_target(nid) {
        return Some(target);
    }
    let runtime = native_cxx_runtime()?;
    if let Some(name) = native_cxx_symbol_name(nid) {
        return runtime.cxxabi_symbol(name);
    }
    if let Some(name) = native_unwind_symbol_name(nid) {
        return runtime.unwind_symbol(name);
    }
    runtime.cxxabi_symbol(native_libc_symbol_name(nid)?)
}

#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
pub(super) const fn native_import_target(_nid: &str) -> Option<u64> {
    None
}

extern "C" fn synthetic_data_noop(
    _argument0: u64,
    _argument1: u64,
    _argument2: u64,
    _argument3: u64,
    _argument4: u64,
    _argument5: u64,
) -> u64 {
    0
}

pub(super) struct SyntheticDataImports {
    pub(super) _classic_vtable: Box<[u64; 32]>,
    pub(super) _classic_object: Box<[u64; 8]>,
    pub(super) classic_symbol: Box<u64>,
    pub(super) cells: BTreeMap<&'static str, Box<[u64; 32]>>,
}

pub(super) fn synthetic_data_imports() -> &'static SyntheticDataImports {
    static IMPORTS: OnceLock<SyntheticDataImports> = OnceLock::new();
    IMPORTS.get_or_init(|| {
        let noop = synthetic_data_noop as *const () as usize as u64;
        let mut classic_vtable = Box::new([0_u64; 32]);
        classic_vtable[2..].fill(noop);
        let mut classic_object = Box::new([0_u64; 8]);
        classic_object[0] = classic_vtable.as_ptr() as usize as u64 + 16;
        let classic_symbol = Box::new(classic_object.as_ptr() as usize as u64);

        let mut cells = BTreeMap::new();
        for nid in [
            "H8AprKeZtNg",
            "sBCTjFk7Gi4",
            "yLE5H3058Ao",
            "Cv+zC4EjGMA",
            "H4fcpQOpc08",
            "FQ9NFbBHb5Y",
            "wiR+rIcbnlc",
            "E14mW8pVpoE",
            "1kZFcktOm+s",
            "VmqsS6auJzo",
            "irGo1yaJ-vM",
            "2sWzhYqFH4E",
            "f7uOxY9mM1U",
        ] {
            let mut cell = Box::new([0_u64; 32]);
            if nid == "FQ9NFbBHb5Y" {
                cell[0] = u64::MAX;
            } else if nid == "f7uOxY9mM1U" {
                cell[0] = 0x9e37_79b9_7f4a_7c15;
            } else if matches!(nid, "yLE5H3058Ao" | "1kZFcktOm+s") {
                cell[2..].fill(noop);
            }
            cells.insert(nid, cell);
        }
        SyntheticDataImports {
            _classic_vtable: classic_vtable,
            _classic_object: classic_object,
            classic_symbol,
            cells,
        }
    })
}

pub(super) fn synthetic_data_import_target(nid: &str) -> Option<u64> {
    let imports = synthetic_data_imports();
    if nid == "Qoo175Ig+-k" {
        return Some((&raw const *imports.classic_symbol) as usize as u64);
    }
    imports
        .cells
        .get(nid)
        .map(|cell| cell.as_ptr() as usize as u64)
}

#[cfg(all(test, target_os = "macos", target_arch = "x86_64"))]
mod tests {
    use super::{
        direct_posix_pthread_set_specific, direct_runtime_import_target,
        direct_sce_pthread_get_thread_id, direct_sce_pthread_self,
        direct_sce_pthread_set_specific, native_libc_symbol_name,
    };

    #[test]
    fn guest_timeval_does_not_resolve_to_the_darwin_abi() {
        // Darwin declares tv_usec as a 32-bit integer, while the console ABI reads the field as
        // 64 bits. Calling the host export directly leaves the upper half of the guest field
        // untouched and can turn a millisecond timeout into a multi-hour wait.
        assert!(native_libc_symbol_name("n88vx3C5nW8").is_none());
    }

    #[test]
    fn pthread_set_specific_resolves_to_lock_free_leaf_targets() {
        assert_eq!(
            direct_runtime_import_target("WrOLvHU0yQM"),
            Some(direct_posix_pthread_set_specific as *const () as usize as u64)
        );
        assert_eq!(
            direct_runtime_import_target("+BzXYkqYeLE"),
            Some(direct_sce_pthread_set_specific as *const () as usize as u64)
        );
    }

    #[test]
    fn pthread_handle_and_integer_thread_id_remain_distinct() {
        let _ = remu_gpu::take_dispatch_telemetry_sample();
        crate::set_current_guest_thread_handle(0x9000_1234_0000);
        crate::set_current_guest_thread_unique_id(37);
        assert_eq!(direct_sce_pthread_self(), 0x9000_1234_0000);
        assert_eq!(direct_sce_pthread_get_thread_id(), 37);
        crate::set_current_guest_thread_handle(0);
        crate::set_current_guest_thread_unique_id(0);

        let sample = remu_gpu::take_dispatch_telemetry_sample();
        assert_eq!(
            sample.direct_import_calls_for(remu_gpu::DirectDispatchImport::ScePthreadSelf),
            1
        );
        assert_eq!(
            sample.direct_import_calls_for(remu_gpu::DirectDispatchImport::ScePthreadGetThreadId),
            1
        );
    }
}
