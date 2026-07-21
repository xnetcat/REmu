// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

mod agc_register_defaults;

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ffi::CString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Condvar, Mutex, OnceLock, RwLock};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use remu_core::GuestAddress;
use remu_loader::{DynamicInfo, Elf64Rela, Elf64Symbol, LoadedExecutable};
use remu_memory::{GUEST_EXECUTE, GUEST_READ, GUEST_WRITE, GuestImage, MappingError};
use thiserror::Error;

const R_X86_64_64: u32 = 1;
const R_X86_64_GLOB_DAT: u32 = 6;
const R_X86_64_JUMP_SLOT: u32 = 7;
const R_X86_64_RELATIVE: u32 = 8;
const IMPORT_STUB_SIZE: u64 = 32;
const IMPORT_STUB_SIZE_USIZE: usize = 32;
const PT_TLS: u32 = 7;
const PT_GNU_EH_FRAME: u32 = 0x6474_e550;
const DW_EH_PE_PCREL_SDATA4: u8 = 0x1b;
const GUEST_TLS_PREFIX_SIZE: u64 = 0x20_000;
const GUEST_TLS_TCB_SIZE: u64 = 0x10_000;
const TLS_HELPER_SIZE: u64 = 96;
const TLS_HELPER_SIZE_USIZE: usize = 96;
const TLS_HELPER_COUNT: u64 = 16;
const TLS_HELPER_REGION_SIZE: u64 = TLS_HELPER_SIZE * TLS_HELPER_COUNT;
const TLS_HELPER_REGION_SIZE_USIZE: usize = 16 * TLS_HELPER_SIZE_USIZE;
const MACOS_GUEST_PAGE_ALIGNMENT: u64 = 0x4000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportStub {
    pub nid: String,
    pub address: GuestAddress,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RelocationStats {
    pub absolute: usize,
    pub global_data: usize,
    pub jump_slots: usize,
    pub relative: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuestTls {
    pub thread_pointer: GuestAddress,
    pub static_block_size: u64,
    pub patched_loads: usize,
}

impl RelocationStats {
    #[must_use]
    pub const fn total(self) -> usize {
        self.absolute + self.global_data + self.jump_slots + self.relative
    }
}

#[derive(Debug)]
pub struct LinkedImage {
    pub memory: GuestImage,
    pub import_stubs: Vec<ImportStub>,
    pub relocation_stats: RelocationStats,
    pub tls: GuestTls,
    unwind_fdes: Vec<GuestAddress>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProbeOutcome {
    pub return_value: i32,
    pub import_hits: usize,
}

#[derive(Debug, Error)]
pub enum LinkError {
    #[error(transparent)]
    Mapping(#[from] MappingError),
    #[error("executable does not contain an ELF image")]
    MissingElf,
    #[error("executable does not contain dynamic-linker metadata")]
    MissingDynamicMetadata,
    #[error("import stub allocation size overflow")]
    StubAllocationOverflow,
    #[error("relocation references missing symbol {index}")]
    MissingSymbol { index: u32 },
    #[error("unresolved non-weak symbol {index} has no import stub")]
    MissingImportStub { index: u32 },
    #[error("unsupported x86-64 relocation type {relocation_type} at {offset}")]
    UnsupportedRelocation {
        relocation_type: u32,
        offset: GuestAddress,
    },
    #[error("guest address arithmetic overflow")]
    AddressOverflow,
    #[error("guest TLS alignment 0x{0:x} is not a power of two")]
    InvalidTlsAlignment(u64),
    #[error(
        "guest static TLS requires 0x{required:x} bytes, exceeding the 0x{available:x} startup reservation"
    )]
    StaticTlsTooLarge { required: u64, available: u64 },
    #[error("executable region size cannot be represented on this host")]
    ExecutableRegionTooLarge,
    #[error("TLS helper at {helper} is outside rel32 range of guest instruction {instruction}")]
    TlsHelperOutOfRange {
        instruction: GuestAddress,
        helper: GuestAddress,
    },
    #[error("native guest probing requires an x86-64 process")]
    ProbeRequiresX86_64,
    #[error("process image name contains an interior NUL byte")]
    InvalidProcessImageName,
    #[error("unsupported guest .eh_frame pointer encoding 0x{0:02x}")]
    UnsupportedEhFrameEncoding(u8),
    #[error("malformed guest .eh_frame data: {0}")]
    MalformedEhFrame(&'static str),
    #[error("native macOS C++ unwind runtime is unavailable")]
    NativeCxxRuntimeUnavailable,
    #[error("native macOS unwind symbol {0} is unavailable")]
    MissingNativeUnwindSymbol(&'static str),
}

impl LinkedImage {
    pub fn link_main(executable: &LoadedExecutable) -> Result<Self, LinkError> {
        let elf = executable.info.elf.as_ref().ok_or(LinkError::MissingElf)?;
        let dynamic = elf
            .dynamic
            .as_ref()
            .ok_or(LinkError::MissingDynamicMetadata)?;
        let mut memory = GuestImage::map_main(executable)?;
        let (import_stubs, symbol_stubs) = create_import_stubs(&mut memory, dynamic)?;
        let relocation_stats = apply_relocations(&mut memory, dynamic, &symbol_stubs)?;
        let unwind_fdes = collect_unwind_fdes(&memory, executable)?;
        let tls = install_tls_support(&mut memory, executable)?;
        memory.finalize_permissions()?;
        Ok(Self {
            memory,
            import_stubs,
            relocation_stats,
            tls,
            unwind_fdes,
        })
    }

    pub fn probe_entry(
        &self,
        process_image_name: &str,
        game_root: &Path,
        maximum_import_hits: usize,
        timeout_seconds: u32,
    ) -> Result<ProbeOutcome, LinkError> {
        probe_entry(
            self,
            process_image_name,
            game_root,
            maximum_import_hits,
            timeout_seconds,
        )
    }

    #[must_use]
    pub fn unwind_record_count(&self) -> usize {
        self.unwind_fdes.len()
    }
}

/// Returns the number of guest pthread workers that are still executing.
///
/// A PS5 process can return from its entry routine after handing long-lived work to pthreads. The
/// graphical launcher uses this count to keep the mapped image and GPU event sink alive until those
/// workers finish or the launch timeout expires.
#[must_use]
pub fn active_guest_thread_count() -> usize {
    PROBE_DISPATCH.get().map_or(0, |state| {
        state
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .thread_handles
            .values()
            .filter(|thread| !thread.is_finished())
            .count()
    })
}

fn install_tls_support(
    memory: &mut GuestImage,
    executable: &LoadedExecutable,
) -> Result<GuestTls, LinkError> {
    // Keep helpers beside the image so every rewritten load can use a five-byte rel32 call.
    let helper_base =
        memory.allocate_staging(TLS_HELPER_REGION_SIZE, 16, GUEST_READ | GUEST_EXECUTE)?;
    // Keep writable TLS on a separate host page from executable import/TLS helpers.
    let allocation_base = memory.allocate_staging(
        GUEST_TLS_PREFIX_SIZE + GUEST_TLS_TCB_SIZE,
        MACOS_GUEST_PAGE_ALIGNMENT,
        GUEST_READ | GUEST_WRITE,
    )?;
    let thread_pointer = GuestAddress::new(
        allocation_base
            .get()
            .checked_add(GUEST_TLS_PREFIX_SIZE)
            .ok_or(LinkError::AddressOverflow)?,
    );

    let mut static_block_size = 0;
    if let Some(tls_header) = executable
        .info
        .elf
        .as_ref()
        .and_then(|elf| {
            elf.program_headers
                .iter()
                .find(|header| header.program_type == PT_TLS)
        })
        .filter(|header| header.memory_size != 0)
    {
        let alignment = if tls_header.alignment == 0 {
            1
        } else {
            tls_header.alignment
        };
        if !alignment.is_power_of_two() {
            return Err(LinkError::InvalidTlsAlignment(alignment));
        }
        static_block_size = calculate_static_tls_offset(
            0,
            tls_header.memory_size,
            alignment,
            tls_header.virtual_address.get(),
        )?;
        if static_block_size > GUEST_TLS_PREFIX_SIZE {
            return Err(LinkError::StaticTlsTooLarge {
                required: static_block_size,
                available: GUEST_TLS_PREFIX_SIZE,
            });
        }
        let block = GuestAddress::new(
            thread_pointer
                .get()
                .checked_sub(static_block_size)
                .ok_or(LinkError::AddressOverflow)?,
        );
        let initialized_size = tls_header.file_size.min(tls_header.memory_size);
        if initialized_size != 0 {
            let initialized_size = usize::try_from(initialized_size)
                .map_err(|_| LinkError::ExecutableRegionTooLarge)?;
            let source = runtime_address(memory, tls_header.virtual_address, initialized_size)?;
            let mut template = vec![0_u8; initialized_size];
            memory.read(source, &mut template)?;
            memory.write(block, &template)?;
        }
    }

    seed_tls_thread_control(memory, thread_pointer, static_block_size)?;

    write_tls_helpers(memory, helper_base)?;
    let patched_loads = patch_tls_loads(memory, helper_base)?;
    Ok(GuestTls {
        thread_pointer,
        static_block_size,
        patched_loads,
    })
}

fn seed_tls_thread_control(
    memory: &mut GuestImage,
    thread_pointer: GuestAddress,
    static_block_size: u64,
) -> Result<(), LinkError> {
    let dtv = add_address(thread_pointer, 0x100)?;
    write_u64(memory, thread_pointer, thread_pointer.get())?;
    write_u64(memory, add_address(thread_pointer, 0x08)?, dtv.get())?;
    write_u64(
        memory,
        add_address(thread_pointer, 0x10)?,
        thread_pointer.get(),
    )?;
    write_u64(
        memory,
        add_address(thread_pointer, 0x28)?,
        0xc0de_c0de_cafe_ba00,
    )?;
    write_u64(
        memory,
        add_address(thread_pointer, 0x60)?,
        thread_pointer.get(),
    )?;
    write_u64(memory, subtract_address(thread_pointer, 0xf0)?, 0)?;
    // One startup module: generation, maximum module ID, and module-1 block address.
    write_u64(memory, dtv, 1)?;
    write_u64(
        memory,
        add_address(dtv, 8)?,
        usize::from(static_block_size != 0) as u64,
    )?;
    if static_block_size != 0 {
        write_u64(
            memory,
            add_address(dtv, 16)?,
            thread_pointer.get() - static_block_size,
        )?;
    }
    Ok(())
}

fn calculate_static_tls_offset(
    previous_offset: u64,
    size: u64,
    alignment: u64,
    alignment_bias: u64,
) -> Result<u64, LinkError> {
    let result = previous_offset
        .checked_add(size)
        .and_then(|value| value.checked_add(alignment - 1))
        .ok_or(LinkError::AddressOverflow)?;
    let biased = result
        .checked_add(alignment_bias)
        .ok_or(LinkError::AddressOverflow)?;
    Ok(result - (biased & (alignment - 1)))
}

fn write_tls_helpers(memory: &mut GuestImage, helper_base: GuestAddress) -> Result<(), LinkError> {
    let mut helpers = vec![0x90_u8; TLS_HELPER_REGION_SIZE_USIZE];
    let saved_registers = [0_u8, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let dispatcher = current_guest_thread_pointer as *const () as usize as u64;
    for destination in 0_u8..16 {
        let offset = usize::from(destination) * TLS_HELPER_SIZE_USIZE;
        let slot = &mut helpers[offset..offset + TLS_HELPER_SIZE_USIZE];
        let mut cursor = 0;
        slot[cursor] = 0x9c; // pushfq
        cursor += 1;
        for register in saved_registers {
            emit_push_register(slot, &mut cursor, register);
        }
        slot[cursor..cursor + 4].copy_from_slice(&[0x48, 0x83, 0xec, 0x08]);
        cursor += 4;
        slot[cursor..cursor + 2].copy_from_slice(&[0x48, 0xb8]);
        slot[cursor + 2..cursor + 10].copy_from_slice(&dispatcher.to_le_bytes());
        cursor += 10;
        slot[cursor..cursor + 2].copy_from_slice(&[0xff, 0xd0]);
        cursor += 2;
        slot[cursor..cursor + 4].copy_from_slice(&[0x48, 0x83, 0xc4, 0x08]);
        cursor += 4;
        let saved_index = saved_registers
            .iter()
            .position(|register| *register == destination)
            .unwrap_or(0);
        let saved_offset = (saved_registers.len() - 1 - saved_index) * 8;
        slot[cursor..cursor + 5].copy_from_slice(&[
            0x48,
            0x89,
            0x44,
            0x24,
            u8::try_from(saved_offset).map_err(|_| LinkError::AddressOverflow)?,
        ]);
        cursor += 5;
        for register in saved_registers.into_iter().rev() {
            emit_pop_register(slot, &mut cursor, register);
        }
        slot[cursor] = 0x9d; // popfq
        cursor += 1;
        slot[cursor] = 0xc3;
    }
    memory.write(helper_base, &helpers)?;
    Ok(())
}

fn emit_push_register(slot: &mut [u8], cursor: &mut usize, register: u8) {
    if register < 8 {
        slot[*cursor] = 0x50 + register;
        *cursor += 1;
    } else {
        slot[*cursor..*cursor + 2].copy_from_slice(&[0x41, 0x50 + (register - 8)]);
        *cursor += 2;
    }
}

fn emit_pop_register(slot: &mut [u8], cursor: &mut usize, register: u8) {
    if register < 8 {
        slot[*cursor] = 0x58 + register;
        *cursor += 1;
    } else {
        slot[*cursor..*cursor + 2].copy_from_slice(&[0x41, 0x58 + (register - 8)]);
        *cursor += 2;
    }
}

fn patch_tls_loads(memory: &mut GuestImage, helper_base: GuestAddress) -> Result<usize, LinkError> {
    let executable_regions = memory
        .regions()
        .iter()
        .filter(|region| region.flags & GUEST_EXECUTE != 0 && region.file_size >= 9)
        .copied()
        .collect::<Vec<_>>();
    let mut patches = Vec::new();
    for region in executable_regions {
        let length =
            usize::try_from(region.file_size).map_err(|_| LinkError::ExecutableRegionTooLarge)?;
        let mut bytes = vec![0_u8; length];
        memory.read(region.start, &mut bytes)?;
        for offset in 0..=bytes.len() - 9 {
            let source = &bytes[offset..];
            if source[0] != 0x64 || !(0x48..=0x4f).contains(&source[1]) {
                continue;
            }
            let rex = source[1];
            if rex & 0x08 == 0 || source[2] != 0x8b {
                continue;
            }
            let mod_rm = source[3];
            if mod_rm >> 6 != 0 || mod_rm & 7 != 4 || source[4] != 0x25 {
                continue;
            }
            if source[5..9] != [0, 0, 0, 0] {
                continue;
            }
            let destination = ((mod_rm >> 3) & 7) | ((rex & 4) << 1);
            // Replacing an RSP destination would invalidate the call's return frame.
            if destination == 4 {
                continue;
            }
            let address = add_address(
                region.start,
                u64::try_from(offset).map_err(|_| LinkError::AddressOverflow)?,
            )?;
            patches.push((address, destination));
        }
    }

    for (address, destination) in &patches {
        let helper = add_address(helper_base, u64::from(*destination) * TLS_HELPER_SIZE)?;
        let next = address
            .get()
            .checked_add(5)
            .ok_or(LinkError::AddressOverflow)?;
        let displacement = i128::from(helper.get()) - i128::from(next);
        let displacement =
            i32::try_from(displacement).map_err(|_| LinkError::TlsHelperOutOfRange {
                instruction: *address,
                helper,
            })?;
        let mut replacement = [0x90_u8; 9];
        replacement[0] = 0xe8;
        replacement[1..5].copy_from_slice(&displacement.to_le_bytes());
        memory.write(*address, &replacement)?;
    }
    Ok(patches.len())
}

fn add_address(address: GuestAddress, offset: u64) -> Result<GuestAddress, LinkError> {
    Ok(GuestAddress::new(
        address
            .get()
            .checked_add(offset)
            .ok_or(LinkError::AddressOverflow)?,
    ))
}

fn subtract_address(address: GuestAddress, offset: u64) -> Result<GuestAddress, LinkError> {
    Ok(GuestAddress::new(
        address
            .get()
            .checked_sub(offset)
            .ok_or(LinkError::AddressOverflow)?,
    ))
}

fn write_u64(memory: &mut GuestImage, address: GuestAddress, value: u64) -> Result<(), LinkError> {
    memory.write(address, &value.to_le_bytes())?;
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug)]
struct NativeCxxRuntime {
    cxxabi_handle: usize,
    unwind_handle: usize,
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn native_cxx_runtime() -> Option<NativeCxxRuntime> {
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
    fn cxxabi_symbol(self, name: &std::ffi::CStr) -> Option<u64> {
        // SAFETY: `cxxabi_handle` is a retained live dlopen handle and `name` is NUL-terminated.
        let address =
            unsafe { libc::dlsym(self.cxxabi_handle as *mut libc::c_void, name.as_ptr()) };
        (!address.is_null()).then_some(address as usize as u64)
    }

    fn unwind_symbol(self, name: &std::ffi::CStr) -> Option<u64> {
        // SAFETY: `unwind_handle` is a retained live dlopen handle and `name` is NUL-terminated.
        let address =
            unsafe { libc::dlsym(self.unwind_handle as *mut libc::c_void, name.as_ptr()) };
        (!address.is_null()).then_some(address as usize as u64)
    }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn native_cxx_symbol_name(nid: &str) -> Option<&'static std::ffi::CStr> {
    Some(match nid {
        "XwLA5cTHjt4" => c"__gxx_personality_v0",
        "IJKVjsmxxWI" => c"__cxa_allocate_dependent_exception",
        "cfAXurvfl5o" => c"__cxa_allocate_exception",
        "tsvEmnenz48" => c"__cxa_atexit",
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
fn native_unwind_symbol_name(nid: &str) -> Option<&'static std::ffi::CStr> {
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
fn native_libc_symbol_name(nid: &str) -> Option<&'static std::ffi::CStr> {
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
        "fL3O02ypZFE" => c"wmemcpy",
        "Noj9PsJrsa8" => c"wmemmove",
        "fnUEjBCNRVU" => c"wmemchr",
        "QJ5xVfKkni0" => c"wmemcmp",
        "n88vx3C5nW8" => c"gettimeofday",
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
fn native_import_target(nid: &str) -> Option<u64> {
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
const fn native_import_target(_nid: &str) -> Option<u64> {
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

struct SyntheticDataImports {
    _classic_vtable: Box<[u64; 32]>,
    _classic_object: Box<[u64; 8]>,
    classic_symbol: Box<u64>,
    cells: BTreeMap<&'static str, Box<[u64; 32]>>,
}

fn synthetic_data_imports() -> &'static SyntheticDataImports {
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

fn synthetic_data_import_target(nid: &str) -> Option<u64> {
    let imports = synthetic_data_imports();
    if nid == "Qoo175Ig+-k" {
        return Some((&raw const *imports.classic_symbol) as usize as u64);
    }
    imports
        .cells
        .get(nid)
        .map(|cell| cell.as_ptr() as usize as u64)
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn register_guest_unwind_frames(image: &LinkedImage) -> Result<usize, LinkError> {
    if image.unwind_fdes.is_empty() {
        return Ok(0);
    }
    let runtime = native_cxx_runtime().ok_or(LinkError::NativeCxxRuntimeUnavailable)?;
    let register_address = runtime
        .unwind_symbol(c"__register_frame")
        .ok_or(LinkError::MissingNativeUnwindSymbol("__register_frame"))?;
    type RegisterFrame = unsafe extern "C" fn(*const libc::c_void);
    // SAFETY: dlsym returned the address of libunwind's documented single-pointer function.
    let register: RegisterFrame = unsafe { std::mem::transmute(register_address as usize) };
    for fde in &image.unwind_fdes {
        // SAFETY: Every address was decoded and bounds-checked from the mapped guest .eh_frame.
        unsafe {
            register(fde.get() as usize as *const libc::c_void);
        }
    }
    Ok(image.unwind_fdes.len())
}

fn create_import_stubs(
    memory: &mut GuestImage,
    dynamic: &DynamicInfo,
) -> Result<(Vec<ImportStub>, BTreeMap<u32, GuestAddress>), LinkError> {
    let nids = dynamic
        .imports
        .iter()
        .filter(|import| !import.weak)
        .map(|import| extract_nid(&import.name).to_owned())
        .collect::<BTreeSet<_>>();
    if nids.is_empty() {
        return Ok((Vec::new(), BTreeMap::new()));
    }

    let stub_bytes = u64::try_from(nids.len())
        .map_err(|_| LinkError::StubAllocationOverflow)?
        .checked_mul(IMPORT_STUB_SIZE)
        .ok_or(LinkError::StubAllocationOverflow)?;
    let stub_base =
        memory.allocate_staging(stub_bytes, IMPORT_STUB_SIZE, GUEST_READ | GUEST_EXECUTE)?;
    let bytes_len = usize::try_from(stub_bytes).map_err(|_| LinkError::StubAllocationOverflow)?;
    let mut bytes = vec![0_u8; bytes_len];
    let mut import_stubs = Vec::with_capacity(nids.len());
    let mut by_nid = BTreeMap::new();
    let dispatcher = import_dispatch as *const () as usize as u64;
    for (index, nid) in nids.into_iter().enumerate() {
        let byte_offset = index * IMPORT_STUB_SIZE_USIZE;
        if let Some(target) = native_import_target(&nid) {
            write_native_import_stub(&mut bytes[byte_offset..], target);
        } else {
            write_import_stub(&mut bytes[byte_offset..], index, dispatcher)?;
        }
        let address = GuestAddress::new(
            stub_base
                .get()
                .checked_add(byte_offset as u64)
                .ok_or(LinkError::AddressOverflow)?,
        );
        by_nid.insert(nid.clone(), address);
        import_stubs.push(ImportStub { nid, address });
    }
    memory.write(stub_base, &bytes)?;

    let mut symbol_stubs = BTreeMap::new();
    for import in &dynamic.imports {
        if import.weak {
            continue;
        }
        let nid = extract_nid(&import.name);
        let address = by_nid
            .get(nid)
            .copied()
            .ok_or(LinkError::MissingImportStub {
                index: import.symbol_index,
            })?;
        symbol_stubs.insert(import.symbol_index, address);
    }
    Ok((import_stubs, symbol_stubs))
}

fn apply_relocations(
    memory: &mut GuestImage,
    dynamic: &DynamicInfo,
    symbol_stubs: &BTreeMap<u32, GuestAddress>,
) -> Result<RelocationStats, LinkError> {
    let mut stats = RelocationStats::default();
    for relocation in dynamic.relocations.iter().chain(&dynamic.jump_relocations) {
        let target = runtime_address(memory, relocation.offset, 8)?;
        let value = match relocation.relocation_type {
            R_X86_64_RELATIVE => {
                stats.relative += 1;
                memory
                    .image_base()
                    .get()
                    .wrapping_add_signed(relocation.addend)
            }
            R_X86_64_64 => {
                stats.absolute += 1;
                resolve_symbol(memory, dynamic, symbol_stubs, relocation)?
                    .wrapping_add_signed(relocation.addend)
            }
            R_X86_64_GLOB_DAT => {
                stats.global_data += 1;
                resolve_symbol(memory, dynamic, symbol_stubs, relocation)?
            }
            R_X86_64_JUMP_SLOT => {
                stats.jump_slots += 1;
                resolve_symbol(memory, dynamic, symbol_stubs, relocation)?
            }
            relocation_type => {
                return Err(LinkError::UnsupportedRelocation {
                    relocation_type,
                    offset: relocation.offset,
                });
            }
        };
        memory.write(target, &value.to_le_bytes())?;
    }
    Ok(stats)
}

fn resolve_symbol(
    memory: &GuestImage,
    dynamic: &DynamicInfo,
    symbol_stubs: &BTreeMap<u32, GuestAddress>,
    relocation: &Elf64Rela,
) -> Result<u64, LinkError> {
    let index = usize::try_from(relocation.symbol_index).map_err(|_| LinkError::MissingSymbol {
        index: relocation.symbol_index,
    })?;
    let symbol = dynamic.symbols.get(index).ok_or(LinkError::MissingSymbol {
        index: relocation.symbol_index,
    })?;
    if symbol.is_undefined() {
        if symbol.is_weak() {
            return Ok(0);
        }
        if let Some(address) = native_import_target(extract_nid(&symbol.name)) {
            return Ok(address);
        }
        if let Some(address) = synthetic_data_import_target(extract_nid(&symbol.name)) {
            return Ok(address);
        }
        return symbol_stubs
            .get(&relocation.symbol_index)
            .map(|address| address.get())
            .ok_or(LinkError::MissingImportStub {
                index: relocation.symbol_index,
            });
    }
    defined_symbol_address(memory, symbol)
}

fn defined_symbol_address(memory: &GuestImage, symbol: &Elf64Symbol) -> Result<u64, LinkError> {
    if memory.contains(symbol.value, 1) {
        return Ok(symbol.value.get());
    }
    memory
        .image_base()
        .get()
        .checked_add(symbol.value.get())
        .ok_or(LinkError::AddressOverflow)
}

fn runtime_address(
    memory: &GuestImage,
    value: GuestAddress,
    length: usize,
) -> Result<GuestAddress, LinkError> {
    if memory.contains(value, length) {
        return Ok(value);
    }
    Ok(GuestAddress::new(
        memory
            .image_base()
            .get()
            .checked_add(value.get())
            .ok_or(LinkError::AddressOverflow)?,
    ))
}

fn collect_unwind_fdes(
    memory: &GuestImage,
    executable: &LoadedExecutable,
) -> Result<Vec<GuestAddress>, LinkError> {
    let Some(header) = executable
        .info
        .elf
        .as_ref()
        .and_then(|elf| {
            elf.program_headers
                .iter()
                .find(|header| header.program_type == PT_GNU_EH_FRAME)
        })
        .filter(|header| header.file_size >= 12)
    else {
        return Ok(Vec::new());
    };
    let header_address = runtime_address(memory, header.virtual_address, 12)?;
    let mut header_bytes = [0_u8; 12];
    memory.read(header_address, &mut header_bytes)?;
    if header_bytes[0] != 1 {
        return Err(LinkError::MalformedEhFrame(
            "unsupported .eh_frame_hdr version",
        ));
    }
    if header_bytes[1] != DW_EH_PE_PCREL_SDATA4 {
        return Err(LinkError::UnsupportedEhFrameEncoding(header_bytes[1]));
    }
    if header_bytes[2] != 0x03 {
        return Err(LinkError::MalformedEhFrame(
            "unsupported FDE count encoding",
        ));
    }

    let displacement = i32::from_le_bytes(
        header_bytes[4..8]
            .try_into()
            .expect("fixed-width .eh_frame displacement"),
    );
    let frame_address = header_address
        .get()
        .checked_add(4)
        .and_then(|address| address.checked_add_signed(i64::from(displacement)))
        .ok_or(LinkError::MalformedEhFrame(".eh_frame address overflow"))?;
    if frame_address >= header_address.get() {
        return Err(LinkError::MalformedEhFrame(
            ".eh_frame does not precede its header",
        ));
    }
    let expected_fdes = usize::try_from(u32::from_le_bytes(
        header_bytes[8..12]
            .try_into()
            .expect("fixed-width FDE count"),
    ))
    .map_err(|_| LinkError::MalformedEhFrame("FDE count is too large"))?;
    let mut fdes = Vec::with_capacity(expected_fdes);
    let mut cursor = frame_address;
    while cursor < header_address.get() {
        let mut length_bytes = [0_u8; 4];
        memory.read(GuestAddress::new(cursor), &mut length_bytes)?;
        let short_length = u32::from_le_bytes(length_bytes);
        if short_length == 0 {
            break;
        }

        let (length_field_size, entry_length, cie_pointer_size) = if short_length == u32::MAX {
            let mut extended = [0_u8; 8];
            memory.read(GuestAddress::new(cursor + 4), &mut extended)?;
            (12_u64, u64::from_le_bytes(extended), 8_usize)
        } else {
            (4_u64, u64::from(short_length), 4_usize)
        };
        if entry_length < cie_pointer_size as u64 {
            return Err(LinkError::MalformedEhFrame(
                "entry is shorter than its CIE pointer",
            ));
        }
        let entry_end = cursor
            .checked_add(length_field_size)
            .and_then(|address| address.checked_add(entry_length))
            .ok_or(LinkError::MalformedEhFrame("entry address overflow"))?;
        if entry_end > header_address.get() {
            return Err(LinkError::MalformedEhFrame(
                "entry extends beyond .eh_frame",
            ));
        }
        let mut cie_pointer = [0_u8; 8];
        memory.read(
            GuestAddress::new(cursor + length_field_size),
            &mut cie_pointer[..cie_pointer_size],
        )?;
        if u64::from_le_bytes(cie_pointer) != 0 {
            fdes.push(GuestAddress::new(cursor));
        }
        cursor = entry_end;
    }
    if fdes.len() != expected_fdes {
        return Err(LinkError::MalformedEhFrame(
            "decoded FDE count differs from .eh_frame_hdr",
        ));
    }
    Ok(fdes)
}

fn extract_nid(symbol_name: &str) -> &str {
    symbol_name
        .split_once('#')
        .map_or(symbol_name, |(nid, _)| nid)
}

/// Returns the service name for NIDs whose ABI is currently implemented or under active bring-up.
#[must_use]
pub fn known_nid_name(nid: &str) -> Option<&'static str> {
    Some(match nid {
        "pO96TwzOm5E" => "sceKernelGetDirectMemorySize",
        "rTXw65xmLIA" => "sceKernelAllocateDirectMemory",
        "L-Q3LEjIbgA" => "sceKernelMapDirectMemory",
        "D0OdFMjp46I" => "sceKernelCreateEqueue",
        "fzyMKs9kim0" => "sceKernelWaitEqueue",
        "lLMT9vJAck0" => "clock_gettime",
        "n88vx3C5nW8" => "gettimeofday",
        "9BcDykPmo1I" => "__error",
        "5OqszGpy7Mg" => "strtoull",
        "VOBg+iNwB-4" => "strtoll",
        "2vDqwBlpF-o" => "strtod",
        "8u8lPzUEq+U" => "memchr",
        "dhK16CKwhQg" => "__isfinite",
        "cfAXurvfl5o" => "__cxa_allocate_exception",
        "nOIEswYD4Ig" => "__cxa_free_exception",
        "vkuuLfhnSZI" => "__cxa_throw",
        "CdWp0oHWGr0" => "sceUserServiceGetInitialUser",
        "fPhymKNvK-A" => "sceUserServiceGetLoginUserIdList",
        "xk0AcarP3V4" => "scePadOpen",
        "clVvL4ZDntw" => "scePadSetMotionSensorState",
        "gjP9-KQzoUk" => "scePadGetControllerInformation",
        "YndgXqQVV7c" => "scePadReadState",
        "rPo6tV8D9bM" => "sceSystemServiceGetStatus",
        "TRO721eVt4g" => "sceAgcDcbResetQueue",
        "ZvwO9euwYzc" => "sceAgcDcbSetCxRegistersIndirect",
        "-HOOCn0JY48" => "sceAgcDcbSetShRegistersIndirect",
        "hvUfkUIQcOE" => "sceAgcDcbSetUcRegistersIndirect",
        "D9sr1xGUriE" => "sceAgcCreatePrimitiveState",
        "HV4j+E0MBHE" => "sceAgcCreateInterpolantMapping",
        "l4fM9K-Lyks" => "sceAgcDcbSetIndexBuffer",
        "GIIW2J37e70" => "sceAgcDcbSetIndexSize",
        "B+aG9DUnTKA" => "sceAgcDcbDrawIndexOffset",
        "tSBxhAPyytQ" => "sceAgcDcbSetNumInstances",
        "Yw0jKSqop+E" => "sceAgcDcbDrawIndexAuto",
        "n2fD4A+pb+g" => "sceAgcCbSetShRegisterRangeDirect",
        "V++UgBtQhn0" => "sceAgcGetDataPacketPayloadAddress",
        "aJf+j5yntiU" => "sceAgcDcbEventWrite",
        "LtTouSCZjHM" => "sceAgcCbNop",
        "WmAc2MEj6Io" => "sceAgcDcbDmaData",
        "wr23dPKyWc0" => "sceAgcCbReleaseMem",
        "VmW0Tdpy420" => "sceAgcDcbWaitRegMem",
        "IxYiarKlXxM" => "sceAgcDmaDataPatchSetDstAddressOrOffset",
        "3KDcnM3lrcU" => "sceAgcWaitRegMemPatchAddress",
        "0fWWK5uG9rQ" => "sceAgcQueueEndOfPipeActionPatchAddress",
        "57labkp+rSQ" => "sceAgcDcbAcquireMem",
        "HXzjK9yI30k" => "sceVideoOutAddFlipEvent",
        "PjS5uASwcV8" => "sceVideoOutSetBufferAttribute2",
        "rKBUtgRrtbk" => "sceVideoOutRegisterBuffers2",
        "UglJIZjGssM" => "sceAgcDriverSubmitDcb",
        "6UzEidRZwkg" => "sceAgcDriverSubmitMultiDcbs",
        "YUeqkyT7mEQ" => "sceAgcDcbSetFlip",
        "h9z6+0hEydk" => "sceAgcSuspendPoint",
        "1G3lF1Gg1k8" => "sceKernelOpen",
        "eV9wAD2riIA" => "sceKernelStat",
        "Cg4srZ6TKbU" => "sceKernelRead",
        "UK2Tl2DWUns" => "sceKernelClose",
        "TywrFKCoLGY" => "sceSaveDataInitialize3",
        "dyIhnXq-0SM" => "sceSaveDataDirNameSearch",
        "ZP4e7rlzOUk" => "sceSaveDataMount3",
        "gjRZNnw0JPE" => "sceSaveDataCreateTransactionResource",
        "uW4vfTwMQVo" => "sceSaveDataUmount2",
        "sDCBrmc61XU" => "sceSaveDataPrepare",
        "85zul--eGXs" => "sceSaveDataSetParam",
        "c88Yy54Mx0w" => "sceSaveDataSaveIcon",
        "ie7qhZ4X0Cc" => "sceSaveDataCommit",
        "pztV4AF18iI" => "sincosf",
        "AEJdIVZTEmo" => "qsort",
        "sUP1hBaouOw" => "_Getpctype",
        "rcQCUr0EaRU" => "_Getptoupper",
        _ => return None,
    })
}

fn write_import_stub(slot: &mut [u8], index: usize, dispatcher: u64) -> Result<(), LinkError> {
    let index = u32::try_from(index).map_err(|_| LinkError::StubAllocationOverflow)?;
    // push imm32; movabs rax, dispatcher; call rax; add rsp, 8; ret
    slot[0] = 0x68;
    slot[1..5].copy_from_slice(&index.to_le_bytes());
    slot[5] = 0x48;
    slot[6] = 0xb8;
    slot[7..15].copy_from_slice(&dispatcher.to_le_bytes());
    slot[15] = 0xff;
    slot[16] = 0xd0;
    slot[17..21].copy_from_slice(&[0x48, 0x83, 0xc4, 0x08]);
    slot[21] = 0xc3;
    Ok(())
}

fn write_native_import_stub(slot: &mut [u8], target: u64) {
    // movabs rax, target; jmp rax. The tail jump leaves no synthetic frame between the guest and
    // libc++abi, which lets the host unwinder return directly to the guest call site.
    slot[0] = 0x48;
    slot[1] = 0xb8;
    slot[2..10].copy_from_slice(&target.to_le_bytes());
    slot[10] = 0xff;
    slot[11] = 0xe0;
}

#[derive(Debug, Default)]
struct ProbeDispatchState {
    nids: Vec<String>,
    seen_nids: BTreeSet<String>,
    unimplemented_nids: BTreeSet<String>,
    import_counts: BTreeMap<String, usize>,
    hits: usize,
    maximum_hits: usize,
    next_direct_offset: u64,
    next_virtual_hint: u64,
    mapped_allocations: Vec<(u64, u64)>,
    next_thread_handle: u64,
    thread_handles: BTreeMap<u64, std::thread::JoinHandle<u64>>,
    agc_primary_register_defaults: u64,
    agc_internal_register_defaults: u64,
    ctype_table: u64,
    toupper_table: u64,
    tls_template: Vec<u8>,
    tls_static_block_size: u64,
    video_out: VideoOutState,
    game_root: PathBuf,
    next_np_handle: i32,
    agc_submission_count: u64,
    agc_graphics: AgcGraphicsState,
    agc_index_resources: BTreeMap<u64, u64>,
    pending_events: VecDeque<KernelQueuedEvent>,
    next_kernel_fd: i32,
    kernel_files: BTreeMap<i32, std::fs::File>,
    save_root: PathBuf,
    save_mounts: BTreeMap<String, PathBuf>,
    next_save_transaction_resource: u32,
}

#[derive(Debug, Default)]
struct VideoOutState {
    pixel_format: u64,
    tiling_mode: u32,
    width: u32,
    height: u32,
    display_buffers: Vec<u64>,
    flip_events: BTreeMap<u64, u64>,
}

#[derive(Debug, Default)]
struct AgcGraphicsState {
    context_registers: BTreeMap<u32, u32>,
    shader_registers: BTreeMap<u32, u32>,
    user_config_registers: BTreeMap<u32, u32>,
    index_buffer_address: u64,
    index_resource_address: u64,
    index_buffer_count: u32,
    index_size: u32,
    instance_count: u32,
    draw_index_offset: u32,
    emitted_draws: u64,
    traced_shader_pairs: BTreeSet<(u64, u64)>,
}

impl AgcGraphicsState {
    fn reset_pipeline(&mut self) {
        self.context_registers.clear();
        self.shader_registers.clear();
        self.user_config_registers.clear();
        self.index_buffer_address = 0;
        self.index_resource_address = 0;
        self.index_buffer_count = 0;
        self.index_size = 0;
        self.instance_count = 1;
        self.draw_index_offset = 0;
    }
}

#[derive(Clone, Copy, Debug)]
struct KernelQueuedEvent {
    equeue: u64,
    ident: u64,
    filter: i16,
    flags: u16,
    fflags: u32,
    data: u64,
    user_data: u64,
}

static PROBE_DISPATCH: OnceLock<RwLock<ProbeDispatchState>> = OnceLock::new();

#[derive(Debug, Default)]
struct GuestSyncState {
    mutexes: HashMap<u64, GuestMutexState>,
    conditions: HashMap<u64, u64>,
    semaphores: HashMap<u32, GuestSemaphoreState>,
    next_semaphore_handle: u32,
}

#[derive(Debug, Default)]
struct GuestMutexState {
    owner: Option<ThreadId>,
    recursion: u32,
}

#[derive(Debug)]
struct GuestSemaphoreState {
    count: i32,
    maximum: i32,
}

static GUEST_SYNC: OnceLock<(Mutex<GuestSyncState>, Condvar)> = OnceLock::new();

thread_local! {
    static CURRENT_GUEST_THREAD_POINTER: Cell<u64> = const { Cell::new(0) };
}

extern "C" fn current_guest_thread_pointer() -> u64 {
    CURRENT_GUEST_THREAD_POINTER.get()
}

fn set_current_guest_thread_pointer(thread_pointer: u64) {
    CURRENT_GUEST_THREAD_POINTER.set(thread_pointer);
}

#[cfg(target_arch = "x86_64")]
fn configure_probe_dispatch(image: &LinkedImage, game_root: &Path, maximum_hits: usize) {
    let state = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
    let mut state = state
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.nids = image
        .import_stubs
        .iter()
        .map(|stub| stub.nid.clone())
        .collect();
    state.seen_nids.clear();
    state.unimplemented_nids.clear();
    state.import_counts.clear();
    state.hits = 0;
    state.maximum_hits = maximum_hits;
    state.next_direct_offset = 0;
    state.next_virtual_hint = 0x20_0000_0000;
    state.mapped_allocations.clear();
    state.next_thread_handle = 1;
    state.thread_handles.clear();
    state.agc_primary_register_defaults = 0;
    state.agc_internal_register_defaults = 0;
    state.ctype_table = 0;
    state.toupper_table = 0;
    let tls_template_address = image.tls.thread_pointer.get() - GUEST_TLS_PREFIX_SIZE;
    let tls_template_size = usize::try_from(GUEST_TLS_PREFIX_SIZE + GUEST_TLS_TCB_SIZE)
        .expect("guest TLS template size fits usize");
    state.tls_template.resize(tls_template_size, 0);
    // SAFETY: The linked image owns this live TLS reservation for the entire probe. Snapshot it
    // before guest entry so later workers never inherit the main thread's mutated allocator caches.
    unsafe {
        std::ptr::copy_nonoverlapping(
            tls_template_address as *const u8,
            state.tls_template.as_mut_ptr(),
            tls_template_size,
        );
    }
    state.tls_static_block_size = image.tls.static_block_size;
    state.video_out = VideoOutState::default();
    game_root.clone_into(&mut state.game_root);
    state.next_np_handle = 1;
    state.agc_submission_count = 0;
    state.agc_graphics = AgcGraphicsState::default();
    state.agc_index_resources.clear();
    state.pending_events.clear();
    state.next_kernel_fd = 3;
    state.kernel_files.clear();
    state.save_root = save_data_root(game_root);
    state.save_mounts.clear();
    state.next_save_transaction_resource = 1;
    let sync = GUEST_SYNC.get_or_init(|| (Mutex::new(GuestSyncState::default()), Condvar::new()));
    *sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = GuestSyncState::default();
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
extern "C" fn import_dispatch(
    argument0: u64,
    argument1: u64,
    argument2: u64,
    argument3: u64,
    argument4: u64,
    argument5: u64,
    float_argument0: f64,
    index: u64,
    guest_return_address: u64,
    stack_argument0: u64,
    stack_argument1: u64,
    stack_argument2: u64,
    stack_argument3: u64,
    stack_argument4: u64,
    stack_argument5: u64,
) -> ProbeImportReturn {
    let call_args = [
        argument0, argument1, argument2, argument3, argument4, argument5,
    ];
    let stack_args = [
        stack_argument0,
        stack_argument1,
        stack_argument2,
        stack_argument3,
        stack_argument4,
        stack_argument5,
    ];
    let state = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
    let mut state = state
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(nid) = usize::try_from(index)
        .ok()
        .and_then(|index| state.nids.get(index))
        .cloned()
    else {
        eprintln!("[REMu][IMPORT] invalid stub index {index}");
        terminate_probe(87);
    };
    state.hits += 1;
    *state.import_counts.entry(nid.clone()).or_default() += 1;
    if state.seen_nids.insert(nid.clone()) {
        let name = known_nid_name(&nid).unwrap_or("unknown");
        eprintln!(
            "[REMu][IMPORT] first_hit={} nid={} name={} return={guest_return_address:#x} args={argument0:#x},{argument1:#x},{argument2:#x},{argument3:#x},{argument4:#x},{argument5:#x}",
            state.hits, nid, name,
        );
    }
    if state.maximum_hits != 0 && state.hits >= state.maximum_hits {
        eprintln!("[REMu][PROBE] import hit limit reached at nid={nid}");
        let mut counts = state
            .import_counts
            .iter()
            .map(|(nid, count)| (nid, *count))
            .collect::<Vec<_>>();
        counts.sort_unstable_by_key(|(_, count)| std::cmp::Reverse(*count));
        for (nid, count) in counts.into_iter().take(8) {
            eprintln!("[REMu][PROBE] hot_import nid={nid} hits={count}");
        }
        terminate_probe(86);
    }
    if nid == "6UgtwV+0zb4" {
        let handle = state.next_thread_handle;
        state.next_thread_handle = state.next_thread_handle.saturating_add(1);
        if call_args[0] != 0 {
            write_guest_u64(call_args[0], handle);
        }
        let entry = call_args[2];
        let thread_argument = call_args[3];
        let guest_tls = clone_guest_tls(&state.tls_template, state.tls_static_block_size);
        let thread = std::thread::spawn(move || {
            set_current_guest_thread_pointer(guest_tls.thread_pointer);
            let result = run_guest_thread(entry, thread_argument);
            drop(guest_tls);
            result
        });
        state.thread_handles.insert(handle, thread);
        return ProbeImportReturn::integer(0);
    }
    if nid == "onNY9Byn-W8" {
        let handle = state.thread_handles.remove(&call_args[0]);
        drop(state);
        let return_value = handle.map_or(0, |thread| thread.join().unwrap_or(0));
        if call_args[1] != 0 {
            write_guest_u64(call_args[1], return_value);
        }
        return ProbeImportReturn::integer(0);
    }
    if is_guest_sync_import(&nid) {
        drop(state);
        return ProbeImportReturn::integer(dispatch_guest_sync_import(&nid, call_args));
    }
    // A real AudioOut submission blocks for approximately one 256-frame buffer. Without pacing,
    // the emulated audio worker monopolizes the import boundary and starves the render worker.
    if nid == "QOQtbeDqsT4" {
        drop(state);
        std::thread::sleep(Duration::from_micros(5_333));
        return ProbeImportReturn::integer(0);
    }
    if matches!(nid.as_str(), "5OqszGpy7Mg" | "VOBg+iNwB-4" | "2vDqwBlpF-o") {
        drop(state);
        return dispatch_guest_numeric_parser(&nid, call_args);
    }
    if nid == "pztV4AF18iI" {
        drop(state);
        return ProbeImportReturn::integer(guest_sincosf(
            low_f32(float_argument0),
            call_args[0],
            call_args[1],
        ));
    }
    if nid == "AEJdIVZTEmo" {
        drop(state);
        return ProbeImportReturn::integer(guest_qsort(call_args));
    }
    if nid == "dhK16CKwhQg" {
        return ProbeImportReturn::integer(u64::from(float_argument0.is_finite()));
    }
    ProbeImportReturn::integer(dispatch_probe_import(
        &nid, call_args, stack_args, &mut state,
    ))
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ProbeImportReturn {
    integer: u64,
    float: f64,
}

impl ProbeImportReturn {
    const fn integer(integer: u64) -> Self {
        Self {
            integer,
            float: 0.0,
        }
    }

    const fn float(float: f64) -> Self {
        Self { integer: 0, float }
    }
}

struct GuestThreadTls {
    _storage: Vec<u8>,
    thread_pointer: u64,
}

fn clone_guest_tls(template: &[u8], static_block_size: u64) -> GuestThreadTls {
    let size = usize::try_from(GUEST_TLS_PREFIX_SIZE + GUEST_TLS_TCB_SIZE)
        .expect("guest TLS allocation size fits usize");
    assert_eq!(
        template.len(),
        size,
        "guest TLS template has the linked size"
    );
    let mut storage = template.to_vec();
    let thread_pointer = storage.as_mut_ptr() as usize as u64 + GUEST_TLS_PREFIX_SIZE;
    write_guest_u64(thread_pointer, thread_pointer);
    write_guest_u64(thread_pointer + 0x08, thread_pointer + 0x100);
    write_guest_u64(thread_pointer + 0x10, thread_pointer);
    write_guest_u64(thread_pointer + 0x60, thread_pointer);
    write_guest_u64(thread_pointer + 0x100, 1);
    write_guest_u64(
        thread_pointer + 0x108,
        usize::from(static_block_size != 0) as u64,
    );
    if static_block_size != 0 {
        write_guest_u64(thread_pointer + 0x110, thread_pointer - static_block_size);
    }
    GuestThreadTls {
        _storage: storage,
        thread_pointer,
    }
}

fn is_guest_sync_import(nid: &str) -> bool {
    matches!(
        nid,
        "cmo1RIYva9o"
            | "2Of0f+3mhhE"
            | "9UK1vLZQft4"
            | "upoVrzMHFeE"
            | "tn3VlD0hG60"
            | "ttHNfU+qDBU"
            | "ltCfaGr2JGE"
            | "7H0iTOciTLo"
            | "K-jXhbt2gn4"
            | "2Z+PpY6CaJg"
            | "2Tb92quprl0"
            | "0TyVk4MSLt0"
            | "g+PZd2hiacg"
            | "RXXqi4CtF8w"
            | "WKAXJ4XBPQ4"
            | "BmMjYxmew1w"
            | "kDh-NfxgMtE"
            | "JGgj7Uvrl+A"
            | "188x57JYp0g"
            | "Zxa0VhQVTsk"
            | "4czppHBiriw"
    )
}

fn dispatch_guest_sync_import(nid: &str, arguments: [u64; 6]) -> u64 {
    match nid {
        "cmo1RIYva9o" | "ttHNfU+qDBU" => guest_mutex_init(arguments[0]),
        "2Of0f+3mhhE" | "ltCfaGr2JGE" => guest_mutex_destroy(arguments[0]),
        "9UK1vLZQft4" | "7H0iTOciTLo" => guest_mutex_lock(arguments[0], false),
        "upoVrzMHFeE" | "K-jXhbt2gn4" => guest_mutex_lock(arguments[0], true),
        "tn3VlD0hG60" | "2Z+PpY6CaJg" => guest_mutex_unlock(arguments[0]),
        "2Tb92quprl0" | "0TyVk4MSLt0" => guest_condition_init(arguments[0]),
        "g+PZd2hiacg" | "RXXqi4CtF8w" => guest_condition_destroy(arguments[0]),
        "WKAXJ4XBPQ4" => guest_condition_wait(arguments[0], arguments[1], None),
        "BmMjYxmew1w" => guest_condition_wait(
            arguments[0],
            arguments[1],
            Some(u64::from(low_u32(arguments[2]))),
        ),
        "kDh-NfxgMtE" | "JGgj7Uvrl+A" => guest_condition_signal(arguments[0]),
        "188x57JYp0g" => guest_semaphore_create(arguments),
        "Zxa0VhQVTsk" => guest_semaphore_wait(arguments),
        "4czppHBiriw" => guest_semaphore_signal(arguments),
        _ => 0,
    }
}

fn guest_sync() -> &'static (Mutex<GuestSyncState>, Condvar) {
    GUEST_SYNC.get_or_init(|| (Mutex::new(GuestSyncState::default()), Condvar::new()))
}

fn guest_mutex_init(address: u64) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    let sync = guest_sync();
    sync.0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .mutexes
        .entry(address)
        .or_default();
    0
}

fn guest_mutex_destroy(address: u64) -> u64 {
    let sync = guest_sync();
    sync.0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .mutexes
        .remove(&address);
    0
}

fn guest_mutex_lock(address: u64, try_only: bool) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    let sync = guest_sync();
    let current = std::thread::current().id();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        let mutex = state.mutexes.entry(address).or_default();
        match mutex.owner.as_ref() {
            None => {
                mutex.owner = Some(current);
                mutex.recursion = 1;
                return 0;
            }
            Some(owner) if *owner == current => {
                mutex.recursion = mutex.recursion.saturating_add(1);
                return 0;
            }
            Some(_) if try_only => return u64::from(libc::EBUSY as u32),
            Some(_) => {
                state = sync
                    .1
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        }
    }
}

fn guest_mutex_unlock(address: u64) -> u64 {
    let sync = guest_sync();
    let current = std::thread::current().id();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(mutex) = state.mutexes.get_mut(&address)
        && mutex.owner.as_ref().is_some_and(|owner| *owner == current)
    {
        if mutex.recursion > 1 {
            mutex.recursion -= 1;
        } else {
            mutex.owner = None;
            mutex.recursion = 0;
            sync.1.notify_all();
        }
    }
    0
}

fn guest_condition_init(address: u64) -> u64 {
    let sync = guest_sync();
    sync.0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .conditions
        .entry(address)
        .or_default();
    0
}

fn guest_condition_destroy(address: u64) -> u64 {
    let sync = guest_sync();
    sync.0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .conditions
        .remove(&address);
    0
}

fn guest_condition_wait(condition: u64, mutex_address: u64, timeout_usec: Option<u64>) -> u64 {
    let sync = guest_sync();
    let current = std::thread::current().id();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let generation = *state.conditions.entry(condition).or_default();
    if let Some(mutex) = state.mutexes.get_mut(&mutex_address)
        && mutex.owner.as_ref().is_some_and(|owner| *owner == current)
    {
        mutex.owner = None;
        mutex.recursion = 0;
    }
    sync.1.notify_all();

    if let Some(timeout) = timeout_usec {
        let duration = Duration::from_micros(timeout);
        let (next_state, _) = sync
            .1
            .wait_timeout_while(state, duration, |state| {
                state.conditions.get(&condition).copied().unwrap_or(0) == generation
            })
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state = next_state;
    } else {
        state = sync
            .1
            .wait_while(state, |state| {
                state.conditions.get(&condition).copied().unwrap_or(0) == generation
            })
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }

    loop {
        let mutex = state.mutexes.entry(mutex_address).or_default();
        if mutex.owner.is_none() {
            mutex.owner = Some(current);
            mutex.recursion = 1;
            return 0;
        }
        state = sync
            .1
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

fn guest_condition_signal(condition: u64) -> u64 {
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let generation = state.conditions.entry(condition).or_default();
    *generation = generation.wrapping_add(1);
    sync.1.notify_all();
    0
}

fn guest_semaphore_create(arguments: [u64; 6]) -> u64 {
    let output = arguments[0];
    let initial = low_i32(arguments[3]);
    let maximum = low_i32(arguments[4]);
    if output == 0 || initial < 0 || maximum <= 0 || initial > maximum {
        return u64::MAX;
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.next_semaphore_handle = state.next_semaphore_handle.wrapping_add(1).max(1);
    let handle = state.next_semaphore_handle;
    state.semaphores.insert(
        handle,
        GuestSemaphoreState {
            count: initial,
            maximum,
        },
    );
    // SAFETY: sceKernelCreateSema supplies a writable 32-bit handle output.
    unsafe {
        std::ptr::write_unaligned(output as *mut u32, handle);
    }
    0
}

fn guest_semaphore_wait(arguments: [u64; 6]) -> u64 {
    let handle = low_u32(arguments[0]);
    let needed = low_i32(arguments[1]);
    if needed <= 0 {
        return u64::MAX;
    }
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        let Some(semaphore) = state.semaphores.get_mut(&handle) else {
            return u64::MAX;
        };
        if semaphore.count >= needed {
            semaphore.count -= needed;
            return 0;
        }
        state = sync
            .1
            .wait(state)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}

fn guest_semaphore_signal(arguments: [u64; 6]) -> u64 {
    let handle = low_u32(arguments[0]);
    let amount = low_i32(arguments[1]);
    let sync = guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(semaphore) = state.semaphores.get_mut(&handle) else {
        return u64::MAX;
    };
    semaphore.count = semaphore
        .count
        .saturating_add(amount)
        .min(semaphore.maximum);
    sync.1.notify_all();
    0
}

fn low_u32(value: u64) -> u32 {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn low_i32(value: u64) -> i32 {
    let bytes = value.to_le_bytes();
    i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn low_f32(value: f64) -> f32 {
    let bytes = value.to_bits().to_le_bytes();
    f32::from_bits(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn guest_atomic_fetch_4(address: u64, value: u64, subtract: bool) -> u64 {
    if address == 0 || !address.is_multiple_of(std::mem::align_of::<AtomicU32>() as u64) {
        return u64::MAX;
    }
    let Ok(host_address) = usize::try_from(address) else {
        return u64::MAX;
    };
    let pointer = host_address as *mut u32;
    // SAFETY: Guest atomic imports provide a live, naturally aligned four-byte object in the
    // shared process address space. SeqCst is at least as strong as every guest memory-order mode.
    let atomic = unsafe { AtomicU32::from_ptr(pointer) };
    let value = low_u32(value);
    let previous = if subtract {
        atomic.fetch_sub(value, Ordering::SeqCst)
    } else {
        atomic.fetch_add(value, Ordering::SeqCst)
    };
    u64::from(previous)
}

fn dispatch_kernel_file_import(
    nid: &str,
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> Option<u64> {
    // These raw libKernel calls use Gen5 kernel errors, not POSIX -1/errno semantics.
    Some(match nid {
        "1G3lF1Gg1k8" => guest_kernel_open(arguments, state),
        "eV9wAD2riIA" => guest_kernel_stat(arguments, state),
        "Cg4srZ6TKbU" => guest_kernel_read(arguments, state),
        "UK2Tl2DWUns" => guest_kernel_close(arguments, state),
        _ => return None,
    })
}

#[allow(clippy::too_many_lines)]
fn dispatch_probe_import(
    nid: &str,
    arguments: [u64; 6],
    stack_arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    if let Some(result) = dispatch_kernel_file_import(nid, arguments, state) {
        return result;
    }
    match nid {
        // sceKernelGetDirectMemorySize
        "pO96TwzOm5E" => 16 * 1024 * 1024 * 1024,
        // sceKernelAllocateDirectMemory
        "rTXw65xmLIA" => allocate_direct_memory(arguments, state),
        // sceKernelMapDirectMemory
        "L-Q3LEjIbgA" => map_direct_memory(arguments, state),
        // libc memset
        "8zTFvBIAIN8" => guest_memset(arguments[0], arguments[1], arguments[2]),
        // libc memcpy
        "Q3VBxCXhUHs" => guest_memcpy(arguments[0], arguments[1], arguments[2]),
        // libc memmove
        "+P6FRGH4LfA" => guest_memmove(arguments[0], arguments[1], arguments[2]),
        // libc memcmp
        "DfivPArhucg" => guest_memcmp(arguments[0], arguments[1], arguments[2]),
        // libc memchr
        "8u8lPzUEq+U" => guest_memchr(arguments[0], arguments[1], arguments[2]),
        // The guest ABI returns the previous four-byte value; the third argument is memory order.
        "2HnmKiLmV6s" => guest_atomic_fetch_4(arguments[0], arguments[1], true),
        "iPBqs+YUUFw" => guest_atomic_fetch_4(arguments[0], arguments[1], false),
        // The native x86-64 process and guest share the libc allocation ABI.
        "gQX+4GDQjpM" => guest_malloc(arguments[0]),
        "tIhsqj0qsFE" => guest_free(arguments[0]),
        "2X5agFjKxMc" => guest_calloc(arguments[0], arguments[1]),
        "Y7aJ1uydPMo" => guest_realloc(arguments[0], arguments[1]),
        "Ujf3KzMvRmI" | "2Btkg8k24Zg" => guest_aligned_alloc(arguments[0], arguments[1]),
        "cVSk9y8URbc" => guest_posix_memalign(arguments[0], arguments[1], arguments[2]),
        // libc strlen
        "j4ViWNHEgww" => guest_strlen(arguments[0]),
        "Ovb2dSJOAuE" => guest_strcmp(arguments[0], arguments[1]),
        "aesyjrHVWy4" => guest_strncmp(arguments[0], arguments[1], arguments[2]),
        "kiZSXIWd9vg" => guest_strcpy(arguments[0], arguments[1]),
        "6sJWiWSRuqk" => guest_strncpy(arguments[0], arguments[1], arguments[2]),
        "Ls4tzzhimqQ" => guest_strcat(arguments[0], arguments[1]),
        "AV6ipCNa4Rw" => guest_strcasecmp(arguments[0], arguments[1]),
        "viiwFMaNamA" => guest_strstr(arguments[0], arguments[1]),
        // libc stdio uses host FILE objects after resolving console paths inside the game dump.
        "xeYO4u7uyJ0" => guest_fopen(arguments[0], arguments[1], state),
        "rQFVBXp-Cxg" => guest_fseek(arguments[0], arguments[1], arguments[2]),
        "Qazy8LmXTvw" => guest_ftell(arguments[0]),
        "uodLYyUip20" => guest_fclose(arguments[0]),
        "lbB+UlZqVG0" => guest_fread(arguments[0], arguments[1], arguments[2], arguments[3]),
        // Dinkumware's locale helpers are used by the save-error formatting path. The character
        // table accessor must return a live table; its lock and lifetime helpers are process-local
        // bookkeeping in the single-process HLE runtime.
        "sUP1hBaouOw" | "1uJgoVq3bQU" => guest_ctype_table(state),
        "rcQCUr0EaRU" => guest_toupper_table(state),
        "kALvdgEv5ME" | "9nf8joUTSaQ" | "Q1BL70XVV0o" | "sDCBrmc61XU" | "85zul--eGXs"
        | "c88Yy54Mx0w" | "ie7qhZ4X0Cc" => 0,
        "hqi8yMOCmG0" | "p6LrHjIQMdk" | "QW2jL1J5rwY" | "P8F2oavZXtY" => arguments[0],
        // Preserve the SysV variadic register layout for the integer/pointer formats used by the
        // game's asset path builder. Floating-point varargs require a wider import frame later.
        "tcVi5SivF7Q" => guest_sprintf(arguments),
        // libc fputs
        "QrZZdJ8XsX0" => guest_fputs(arguments[0]),
        // Itanium C++ one-time initialization guards.
        "3GPpjQdAMTw" => guest_guard_acquire(arguments[0]),
        "9rAeANT2tyE" => guest_guard_release(arguments[0]),
        "2emaaluWzUw" => guest_guard_abort(arguments[0]),
        // libc++abi exception storage. Throwing currently ends the probe with a decoded diagnostic
        // until guest unwind metadata is registered with the host unwinder.
        "cfAXurvfl5o" => guest_cxa_allocate_exception(arguments[0]),
        "nOIEswYD4Ig" => guest_cxa_free_exception(arguments[0]),
        "vkuuLfhnSZI" => guest_cxa_throw(arguments),
        // libc __error returns a stable, writable errno cell owned by the current guest thread.
        "9BcDykPmo1I" => guest_errno_address(),
        // The single-thread bootstrap uses these attributes only to configure the first worker.
        "F8bUHwAG284" | "nsYoNRywwNg" => initialize_synthetic_handle(arguments[0]),
        "smWEktiyyG0" | "62KCwEMmzcM" => destroy_synthetic_handle(arguments[0]),
        // scePthreadAttrGetstack returns both values through caller-owned outputs.
        "-quPa4SEJUw" => pthread_attr_get_stack(arguments),
        // Initial single-device handles for input, video, and audio.
        "xk0AcarP3V4" | "Up36PTk687E" | "ekNvsT22rsY" => 1,
        // AGC shader construction and VideoOut mutate caller-owned descriptors. Games consume the
        // resulting pointers and command data immediately, so successful no-ops are insufficient.
        "f3dg2CSgRKY" => agc_create_shader(arguments),
        "TRO721eVt4g" => agc_dcb_reset_queue(arguments),
        "ZvwO9euwYzc" => agc_dcb_set_registers_indirect(arguments, AGC_R_CX_REGS_INDIRECT),
        "-HOOCn0JY48" => agc_dcb_set_registers_indirect(arguments, AGC_R_SH_REGS_INDIRECT),
        "hvUfkUIQcOE" => agc_dcb_set_registers_indirect(arguments, AGC_R_UC_REGS_INDIRECT),
        "vcmNN+AAXnY" | "Qrj4c+61z4A" | "6lNcCp+fxi4" => agc_set_indirect_patch_address(arguments),
        "d-6uF9sZDIU" | "z2duB-hHQSM" | "vRoArM9zaIk" => {
            agc_add_indirect_patch_registers(arguments)
        }
        "D9sr1xGUriE" => agc_create_primitive_state(arguments),
        "HV4j+E0MBHE" => agc_create_interpolant_mapping(arguments),
        "n2fD4A+pb+g" => agc_cb_set_sh_register_range_direct(arguments),
        "V++UgBtQhn0" => agc_get_data_packet_payload_address(arguments),
        "qj7QZpgr9Uw" => agc_type_two_packet(arguments),
        "aJf+j5yntiU" => agc_dcb_event_write(arguments),
        "LtTouSCZjHM" => agc_cb_nop(arguments),
        "WmAc2MEj6Io" => agc_dcb_dma_data(arguments, stack_arguments),
        "wr23dPKyWc0" => agc_cb_release_mem(arguments, stack_arguments),
        "VmW0Tdpy420" => agc_dcb_wait_reg_mem(arguments, stack_arguments),
        "IxYiarKlXxM" => agc_dma_data_patch_destination(arguments),
        "3KDcnM3lrcU" => agc_wait_reg_mem_patch_address(arguments),
        "0fWWK5uG9rQ" => agc_release_mem_patch_address(arguments),
        "57labkp+rSQ" => agc_dcb_acquire_mem(arguments, stack_arguments),
        "l4fM9K-Lyks" => agc_dcb_set_index_buffer(arguments, state),
        "GIIW2J37e70" => agc_dcb_set_index_size(arguments),
        "B+aG9DUnTKA" => agc_dcb_draw_index_offset(arguments),
        "tSBxhAPyytQ" => agc_dcb_set_num_instances(arguments),
        "Yw0jKSqop+E" => agc_dcb_draw_index_auto(arguments),
        "YUeqkyT7mEQ" => agc_dcb_set_flip(arguments),
        "UglJIZjGssM" => agc_driver_submit_dcb(arguments, state),
        "6UzEidRZwkg" => agc_driver_submit_multi_dcbs(arguments, state),
        "PjS5uASwcV8" => video_out_set_buffer_attribute2(arguments, state),
        "rKBUtgRrtbk" => video_out_register_buffers2(arguments, state),
        "HXzjK9yI30k" => video_out_add_flip_event(arguments, state),
        // libScePad. Dreaming Sarah consumes these structures during controller discovery and
        // subsequently on every frame, so leaving them as successful no-ops exposes stale stack
        // contents as controller capabilities and button state.
        "clVvL4ZDntw" => pad_set_motion_sensor_state(arguments),
        "gjP9-KQzoUk" => pad_get_controller_information(arguments),
        "YndgXqQVV7c" => pad_read_state(arguments),
        // libSceSystemService status used by the foreground/background state machine.
        "rPo6tV8D9bM" => system_service_get_status(arguments),
        // libSceSaveData. Returning success without filling the mount result leaves the game's
        // mount point empty, which turns `/savedata0/-saveindex` into the invalid `/-saveindex`.
        "TywrFKCoLGY" => save_data_initialize(state),
        "dyIhnXq-0SM" => save_data_dir_name_search(arguments),
        "ZP4e7rlzOUk" => save_data_mount3(arguments, state),
        "gjRZNnw0JPE" => save_data_create_transaction_resource(arguments, state),
        "uW4vfTwMQVo" => save_data_umount2(arguments, state),
        // User-service IDs use the retail slot encoding expected by the game's controller layer.
        "CdWp0oHWGr0" => write_output_i32(arguments[0], 0x1000_0000),
        "fPhymKNvK-A" => user_service_get_login_user_id_list(arguments),
        // Initial NP and trophy context IDs.
        "5zBnau1uIEo" | "Bagshr7OQ6Q" => write_output_i32(arguments[0], 1),
        // sceKernelCreateEqueue
        "D0OdFMjp46I" => write_output_u64(arguments[0], 1),
        // sceKernelWaitEqueue. Until GPU completion events are connected, report an empty wait
        // deterministically. Not writing outCount made the game interpret old stack bytes as user
        // login events and overflow its four-controller table.
        "fzyMKs9kim0" => kernel_wait_equeue(arguments, state),
        // Minimal offline NP identities. These services only require stable nonzero IDs at boot.
        "hT0IAEvN+M0" => np_create_handle(arguments, state),
        // Exact Gen5 register-default descriptors consumed by indirect AGC state packets.
        "2JtWUUiYBXs" => agc_register_defaults(arguments[0], false, state),
        "wRbq6ZjNop4" => agc_register_defaults(arguments[0], true, state),
        // sceKernelUsleep
        "1jfXLRVzisc" => guest_usleep(arguments[0]),
        // POSIX clock_gettime. Darwin's timespec layout matches the x86-64 guest ABI.
        "lLMT9vJAck0" => guest_clock_gettime(arguments[0], arguments[1]),
        // libKernel exit
        "uMei1W9uyNo" => {
            let bytes = arguments[0].to_le_bytes();
            let status = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            eprintln!("[REMu][PROBE] guest requested exit({status})");
            terminate_probe(status);
        }
        // Keep discovery moving, but identify the exact calls that reached the fallback. The
        // general first-hit log also includes known imports whose implementation lives above this
        // dispatcher, so it cannot distinguish a missing ABI from an incomplete name table.
        _ => {
            if state.unimplemented_nids.insert(nid.to_owned()) {
                eprintln!(
                    "[REMu][IMPORT] unimplemented nid={nid} args={:#x},{:#x},{:#x},{:#x},{:#x},{:#x}",
                    arguments[0],
                    arguments[1],
                    arguments[2],
                    arguments[3],
                    arguments[4],
                    arguments[5]
                );
            }
            0
        }
    }
}

fn kernel_wait_equeue(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let handle = arguments[0];
    let events = arguments[1];
    let event_capacity = low_u32(arguments[2]);
    let out_count = arguments[3];
    if events == 0 || event_capacity == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    if let Some(index) = state
        .pending_events
        .iter()
        .position(|event| event.equeue == handle)
    {
        let event = state
            .pending_events
            .remove(index)
            .expect("the queued event index was just found");
        write_guest_u64(events, event.ident);
        write_guest_u16(events + 0x08, event.filter.cast_unsigned());
        write_guest_u16(events + 0x0a, event.flags);
        write_guest_u32(events + 0x0c, event.fflags);
        write_guest_u64(events + 0x10, event.data);
        write_guest_u64(events + 0x18, event.user_data);
        if out_count != 0 {
            write_guest_u32(out_count, 1);
        }
        return 0;
    }
    if out_count != 0 {
        write_guest_u32(out_count, 0);
    }
    // Avoid letting the temporary empty-queue implementation turn the render worker into a hot
    // spin. A real wait will be woken by the Metal submission/completion path.
    std::thread::sleep(Duration::from_micros(100));
    0
}

fn user_service_get_login_user_id_list(arguments: [u64; 6]) -> u64 {
    let output = arguments[0];
    if output == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    write_guest_u32(output, 0x1000_0000);
    for offset in [0x04, 0x08, 0x0c] {
        write_guest_u32(output + offset, u32::MAX);
    }
    0
}

fn pad_set_motion_sensor_state(arguments: [u64; 6]) -> u64 {
    usize::from(!matches!(low_i32(arguments[0]), 0 | 1)) as u64
}

fn pad_get_controller_information(arguments: [u64; 6]) -> u64 {
    let output = arguments[1];
    if output == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    guest_memset(output, 0, 0x1c);
    write_guest_u32(output, 44.86_f32.to_bits());
    write_guest_u16(output + 0x04, 1920);
    write_guest_u16(output + 0x06, 943);
    write_guest_u8(output + 0x08, 30);
    write_guest_u8(output + 0x09, 30);
    write_guest_u8(output + 0x0b, 1);
    write_guest_u8(output + 0x0c, 1);
    0
}

fn pad_read_state(arguments: [u64; 6]) -> u64 {
    let output = arguments[1];
    if output == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    guest_memset(output, 0, 0x78);
    let host = remu_gpu::host_pad_state();
    let buttons = if auto_cross_active() {
        host.buttons | remu_gpu::PAD_BUTTON_CROSS
    } else {
        host.buttons
    };
    write_guest_u32(output, buttons);
    write_guest_u8(output + 0x04, host.left_x);
    write_guest_u8(output + 0x05, host.left_y);
    write_guest_u8(output + 0x06, host.right_x);
    write_guest_u8(output + 0x07, host.right_y);
    write_guest_u8(output + 0x08, host.l2);
    write_guest_u8(output + 0x09, host.r2);
    write_guest_u32(output + 0x18, 1.0_f32.to_bits());
    write_guest_u8(output + 0x4c, 1);
    let mut timestamp = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: timestamp is a valid writable host timespec.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &raw mut timestamp) } == 0 {
        let seconds = u64::try_from(timestamp.tv_sec).unwrap_or(0);
        let nanoseconds = u64::try_from(timestamp.tv_nsec).unwrap_or(0);
        write_guest_u64(
            output + 0x50,
            seconds
                .saturating_mul(1_000_000)
                .saturating_add(nanoseconds / 1_000),
        );
    }
    write_guest_u8(output + 0x68, 1);
    0
}

fn auto_cross_active() -> bool {
    static START: OnceLock<Instant> = OnceLock::new();
    static TIMES: OnceLock<Vec<Duration>> = OnceLock::new();
    static HOLD: OnceLock<Duration> = OnceLock::new();
    let times = TIMES.get_or_init(|| {
        std::env::var("REMU_AUTO_CROSS")
            .unwrap_or_default()
            .split(',')
            .filter_map(|value| value.trim().parse::<f64>().ok())
            .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
            .map(Duration::from_secs_f64)
            .collect()
    });
    if times.is_empty() {
        return false;
    }
    let hold = *HOLD.get_or_init(|| {
        std::env::var("REMU_AUTO_CROSS_HOLD_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .map_or(Duration::from_millis(400), Duration::from_millis)
    });
    let elapsed = START.get_or_init(Instant::now).elapsed();
    times
        .iter()
        .any(|&time| elapsed >= time && elapsed < time.saturating_add(hold))
}

fn system_service_get_status(arguments: [u64; 6]) -> u64 {
    let output = arguments[0];
    if output == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    guest_memset(output, 0, 0x0c);
    write_guest_u8(output + 0x06, 1);
    0
}

fn agc_register_defaults(version: u64, internal: bool, state: &mut ProbeDispatchState) -> u64 {
    if !matches!(low_u32(version), 7 | 8 | 10 | 13) {
        return 0;
    }

    let cached = if internal {
        &mut state.agc_internal_register_defaults
    } else {
        &mut state.agc_primary_register_defaults
    };
    if *cached != 0 {
        return *cached;
    }

    let layout = if internal {
        agc_register_defaults::RegisterDefaultsLayout::internal()
    } else {
        agc_register_defaults::RegisterDefaultsLayout::primary()
    };
    let blob_length = layout.blob_length();
    let Ok(blob_length_u64) = u64::try_from(blob_length) else {
        return 0;
    };
    let address = guest_calloc(1, blob_length_u64);
    if address == 0 {
        return 0;
    }
    let Some(blob) = layout.build(address) else {
        guest_free(address);
        return 0;
    };
    debug_assert_eq!(blob.len(), blob_length);
    // SAFETY: `address` owns a live allocation of exactly `blob_length` bytes and the source is a
    // separate Vec allocation. The descriptor remains cached for the lifetime of the probe.
    unsafe {
        std::ptr::copy_nonoverlapping(blob.as_ptr(), address as *mut u8, blob.len());
    }
    *cached = address;
    eprintln!(
        "[REMu][AGC] register defaults internal={internal} version={} groups={} address=0x{address:016x}",
        low_u32(version),
        layout.group_count()
    );
    address
}

const AGC_SHADER_FILE_HEADER: u32 = 0x3433_3231;
const AGC_SHADER_VERSION: u32 = 0x18;
const AGC_SHADER_USER_DATA_OFFSET: u64 = 0x08;
const AGC_SHADER_CODE_OFFSET: u64 = 0x10;
const AGC_SHADER_CX_REGISTERS_OFFSET: u64 = 0x18;
const AGC_SHADER_SH_REGISTERS_OFFSET: u64 = 0x20;
const AGC_SHADER_SPECIALS_OFFSET: u64 = 0x28;
const AGC_SHADER_INPUT_SEMANTICS_OFFSET: u64 = 0x30;
const AGC_SHADER_OUTPUT_SEMANTICS_OFFSET: u64 = 0x38;
const AGC_SHADER_TYPE_OFFSET: u64 = 0x5a;
const AGC_SHADER_SH_REGISTER_COUNT_OFFSET: u64 = 0x5c;
const AGC_SHADER_INPUT_SEMANTICS_COUNT_OFFSET: u64 = 0x50;
const AGC_SHADER_OUTPUT_SEMANTICS_COUNT_OFFSET: u64 = 0x56;

const AGC_COMMAND_BUFFER_CURSOR_UP_OFFSET: u64 = 0x10;
const AGC_COMMAND_BUFFER_CURSOR_DOWN_OFFSET: u64 = 0x18;
const AGC_COMMAND_BUFFER_RESERVED_DWORDS_OFFSET: u64 = 0x30;

const AGC_IT_NOP: u32 = 0x10;
const AGC_IT_INDEX_BUFFER_SIZE: u32 = 0x13;
const AGC_IT_INDEX_BASE: u32 = 0x26;
const AGC_IT_INDEX_TYPE: u32 = 0x2a;
const AGC_IT_DRAW_INDEX_AUTO: u32 = 0x2d;
const AGC_IT_NUM_INSTANCES: u32 = 0x2f;
const AGC_IT_DRAW_INDEX_OFFSET: u32 = 0x35;
const AGC_IT_WAIT_REG_MEM: u32 = 0x3c;
const AGC_IT_EVENT_WRITE: u32 = 0x46;
const AGC_IT_SET_CONTEXT_REG: u32 = 0x69;
const AGC_IT_SET_SH_REG: u32 = 0x76;
const AGC_IT_SET_UCONFIG_REG: u32 = 0x79;
const AGC_R_ZERO: u32 = 0x00;
const AGC_R_DRAW_INDEX_AUTO: u32 = 0x04;
const AGC_R_DRAW_RESET: u32 = 0x05;
const AGC_R_WAIT_MEM_32: u32 = 0x0a;
const AGC_R_SH_REGS_INDIRECT: u32 = 0x11;
const AGC_R_CX_REGS_INDIRECT: u32 = 0x12;
const AGC_R_UC_REGS_INDIRECT: u32 = 0x13;
const AGC_R_ACQUIRE_MEM: u32 = 0x14;
const AGC_R_WAIT_MEM_64: u32 = 0x16;
const AGC_R_FLIP: u32 = 0x17;
const AGC_R_RELEASE_MEM: u32 = 0x18;
const AGC_R_DMA_DATA: u32 = 0x19;
const AGC_CB_SET_SH_REGISTER_RANGE_MARKER: u32 = 0x6875_000d;

const AGC_SPI_PS_INPUT_CNTL_0: u32 = 0x191;
const AGC_VGT_PRIMITIVE_TYPE: u32 = 0x242;
const AGC_SPI_SHADER_PGM_LO_PS: u32 = 0x008;
const AGC_SPI_SHADER_PGM_HI_PS: u32 = 0x009;
const AGC_SPI_SHADER_PGM_LO_ES: u32 = 0x0c8;
const AGC_SPI_SHADER_PGM_HI_ES: u32 = 0x0c9;
const AGC_CB_COLOR0_BASE: u32 = 0x318;
const AGC_CB_COLOR0_BASE_EXT: u32 = 0x390;

const AGC_SHADER_SPECIAL_GE_CNTL_OFFSET: u64 = 0x00;
const AGC_SHADER_SPECIAL_VGT_SHADER_STAGES_EN_OFFSET: u64 = 0x08;
const AGC_SHADER_SPECIAL_VGT_GS_OUT_PRIM_TYPE_OFFSET: u64 = 0x20;
const AGC_SHADER_SPECIAL_GE_USER_VGPR_EN_OFFSET: u64 = 0x28;

fn agc_create_shader(arguments: [u64; 6]) -> u64 {
    let destination = arguments[0];
    let header = arguments[1];
    let code = arguments[2];
    if header == 0 || code == 0 {
        return u64::MAX;
    }
    if read_guest_u32(header) != AGC_SHADER_FILE_HEADER
        || read_guest_u32(header + 4) != AGC_SHADER_VERSION
    {
        eprintln!(
            "[REMu][AGC] rejected shader header at {header:#x}: magic={:#x} version={:#x}",
            read_guest_u32(header),
            read_guest_u32(header + 4)
        );
        return u64::MAX;
    }

    for offset in [
        AGC_SHADER_CX_REGISTERS_OFFSET,
        AGC_SHADER_SH_REGISTERS_OFFSET,
        AGC_SHADER_USER_DATA_OFFSET,
        AGC_SHADER_SPECIALS_OFFSET,
        AGC_SHADER_INPUT_SEMANTICS_OFFSET,
        AGC_SHADER_OUTPUT_SEMANTICS_OFFSET,
    ] {
        relocate_guest_pointer(header + offset);
    }
    write_guest_u64(header + AGC_SHADER_CODE_OFFSET, code);

    let user_data = read_guest_u64(header + AGC_SHADER_USER_DATA_OFFSET);
    if user_data != 0 {
        for offset in [0, 8, 0x10, 0x18, 0x20] {
            relocate_guest_pointer(user_data + offset);
        }
    }

    if !patch_agc_shader_program_registers(header, code) {
        return u64::MAX;
    }
    if destination != 0 {
        write_guest_u64(destination, header);
    }
    0
}

fn agc_pm4(length_dwords: u32, opcode: u32, register: u32) -> u32 {
    0xc000_0000
        | ((length_dwords.wrapping_sub(2) & 0x3fff) << 16)
        | ((opcode & 0xff) << 8)
        | ((register & 0x3f) << 2)
}

fn agc_allocate_command_dwords(command_buffer: u64, size_dwords: u32) -> Option<u64> {
    if command_buffer == 0 || size_dwords == 0 {
        return None;
    }
    let cursor_up = read_guest_u64(command_buffer + AGC_COMMAND_BUFFER_CURSOR_UP_OFFSET);
    let cursor_down = read_guest_u64(command_buffer + AGC_COMMAND_BUFFER_CURSOR_DOWN_OFFSET);
    let reserved_dwords =
        read_guest_u32(command_buffer + AGC_COMMAND_BUFFER_RESERVED_DWORDS_OFFSET);
    let available_dwords = cursor_down
        .checked_sub(cursor_up)
        .map_or(0, |bytes| bytes / 4);
    let usable_dwords = available_dwords.saturating_sub(u64::from(reserved_dwords));
    if u64::from(size_dwords) > usable_dwords {
        eprintln!(
            "[REMu][AGC] command buffer full at {command_buffer:#x}: need={size_dwords} available={usable_dwords}"
        );
        return None;
    }
    let size_bytes = u64::from(size_dwords).checked_mul(4)?;
    let next_cursor = cursor_up.checked_add(size_bytes)?;
    write_guest_u64(
        command_buffer + AGC_COMMAND_BUFFER_CURSOR_UP_OFFSET,
        next_cursor,
    );
    Some(cursor_up)
}

fn agc_dcb_reset_queue(arguments: [u64; 6]) -> u64 {
    let command_buffer = arguments[0];
    if low_u32(arguments[1]) != 0x3ff || low_u32(arguments[2]) != 0 {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(command_buffer, 2) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(2, AGC_IT_NOP, AGC_R_DRAW_RESET));
    write_guest_u32(command + 4, 0);
    command
}

fn agc_dcb_set_registers_indirect(arguments: [u64; 6], packet_register: u32) -> u64 {
    let command_buffer = arguments[0];
    let registers = arguments[1];
    let register_count = low_u32(arguments[2]);
    let Some(command) = agc_allocate_command_dwords(command_buffer, 4) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(4, AGC_IT_NOP, packet_register));
    write_guest_u32(command + 4, register_count);
    write_guest_u32(command + 8, low_u32(registers));
    write_guest_u32(command + 12, (registers >> 32) as u32);
    command
}

fn agc_set_indirect_patch_address(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let registers = arguments[1];
    if command == 0 || registers == 0 {
        return u64::MAX;
    }
    write_guest_u32(command + 8, low_u32(registers));
    write_guest_u32(command + 12, (registers >> 32) as u32);
    0
}

fn agc_add_indirect_patch_registers(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    if command == 0 {
        return u64::MAX;
    }
    let register_count = low_u32(arguments[1]);
    let current_count = read_guest_u32(command + 4);
    write_guest_u32(command + 4, current_count.wrapping_add(register_count));
    0
}

fn agc_copy_shader_register(source: u64, destination: u64) {
    write_guest_u32(destination, read_guest_u32(source));
    write_guest_u32(destination + 4, read_guest_u32(source + 4));
}

fn agc_create_primitive_state(arguments: [u64; 6]) -> u64 {
    let cx_registers = arguments[0];
    let uc_registers = arguments[1];
    let hull_shader = arguments[2];
    let geometry_shader = arguments[3];
    let primitive_type = low_u32(arguments[4]);
    if cx_registers == 0 || uc_registers == 0 || hull_shader != 0 || geometry_shader == 0 {
        return u64::MAX;
    }
    let shader_type = read_guest_u8(geometry_shader + AGC_SHADER_TYPE_OFFSET);
    if !matches!(shader_type, 2 | 6) {
        return u64::MAX;
    }
    let specials = read_guest_u64(geometry_shader + AGC_SHADER_SPECIALS_OFFSET);
    if specials == 0 {
        return u64::MAX;
    }
    agc_copy_shader_register(
        specials + AGC_SHADER_SPECIAL_VGT_SHADER_STAGES_EN_OFFSET,
        cx_registers,
    );
    agc_copy_shader_register(
        specials + AGC_SHADER_SPECIAL_VGT_GS_OUT_PRIM_TYPE_OFFSET,
        cx_registers + 8,
    );
    agc_copy_shader_register(specials + AGC_SHADER_SPECIAL_GE_CNTL_OFFSET, uc_registers);
    agc_copy_shader_register(
        specials + AGC_SHADER_SPECIAL_GE_USER_VGPR_EN_OFFSET,
        uc_registers + 8,
    );
    write_guest_u32(uc_registers + 16, AGC_VGT_PRIMITIVE_TYPE);
    write_guest_u32(uc_registers + 20, primitive_type);
    0
}

fn agc_create_interpolant_mapping(arguments: [u64; 6]) -> u64 {
    let registers = arguments[0];
    let geometry_shader = arguments[1];
    let pixel_shader = arguments[2];
    if registers == 0 || geometry_shader == 0 {
        return u64::MAX;
    }
    let output_semantics = read_guest_u64(geometry_shader + AGC_SHADER_OUTPUT_SEMANTICS_OFFSET);
    let output_count = read_guest_u32(geometry_shader + AGC_SHADER_OUTPUT_SEMANTICS_COUNT_OFFSET);
    let input_semantics = if pixel_shader == 0 {
        0
    } else {
        // Reading the count matches the API's descriptor validation even though only the pointer
        // is needed to construct the fixed 32-register mapping.
        let _ = read_guest_u32(pixel_shader + AGC_SHADER_INPUT_SEMANTICS_COUNT_OFFSET);
        read_guest_u64(pixel_shader + AGC_SHADER_INPUT_SEMANTICS_OFFSET)
    };
    for index in 0_u32..32 {
        let mut value = 0;
        if index < output_count && output_semantics != 0 {
            let flat = input_semantics != 0
                && (read_guest_u32(input_semantics + u64::from(index) * 4) >> 22) & 1 != 0;
            value = index | if flat { 0x400 } else { 0 };
        }
        let destination = registers + u64::from(index) * 8;
        write_guest_u32(destination, AGC_SPI_PS_INPUT_CNTL_0 + index);
        write_guest_u32(destination + 4, value);
    }
    0
}

fn agc_cb_set_sh_register_range_direct(arguments: [u64; 6]) -> u64 {
    let command_buffer = arguments[0];
    let offset = low_u32(arguments[1]);
    let values = arguments[2];
    let value_count = low_u32(arguments[3]);
    if command_buffer == 0 || offset == 0 || offset > 0x3ff || value_count == 0 {
        return 0;
    }
    let Some(marker) = agc_allocate_command_dwords(command_buffer, 2) else {
        return 0;
    };
    write_guest_u32(marker, agc_pm4(2, AGC_IT_NOP, AGC_R_ZERO));
    write_guest_u32(marker + 4, AGC_CB_SET_SH_REGISTER_RANGE_MARKER);
    let Some(command) = agc_allocate_command_dwords(command_buffer, value_count.saturating_add(2))
    else {
        return 0;
    };
    write_guest_u32(
        command,
        agc_pm4(value_count.saturating_add(2), AGC_IT_SET_SH_REG, 0),
    );
    write_guest_u32(command + 4, offset);
    for index in 0..value_count {
        let value = if values == 0 {
            0
        } else {
            read_guest_u32(values + u64::from(index) * 4)
        };
        write_guest_u32(command + 8 + u64::from(index) * 4, value);
    }
    command
}

fn agc_get_data_packet_payload_address(arguments: [u64; 6]) -> u64 {
    let output = arguments[0];
    let command = arguments[1];
    if output == 0 || command == 0 {
        return u64::MAX;
    }
    let payload = if low_i32(arguments[2]) == 0 {
        let header = read_guest_u32(command);
        if header & 0x3fff_0000 == 0x3fff_0000 {
            0
        } else {
            command + 4
        }
    } else {
        command + 8
    };
    write_guest_u64(output, payload);
    0
}

fn agc_type_two_packet(arguments: [u64; 6]) -> u64 {
    let Some(command) = agc_allocate_command_dwords(arguments[0], 1) else {
        return 0;
    };
    write_guest_u32(command, 0x8000_0000);
    command
}

fn agc_dcb_event_write(arguments: [u64; 6]) -> u64 {
    let event_type = low_u32(arguments[1]) & 0xff;
    if event_type > 0x3f || arguments[2] != 0 {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(arguments[0], 2) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(2, AGC_IT_EVENT_WRITE, 0));
    write_guest_u32(command + 4, event_type);
    command
}

fn agc_cb_nop(arguments: [u64; 6]) -> u64 {
    let dword_count = low_u32(arguments[1]);
    if !(2..=0x4001).contains(&dword_count) {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(arguments[0], dword_count) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(dword_count, AGC_IT_NOP, AGC_R_ZERO));
    for index in 1..dword_count {
        write_guest_u32(command + u64::from(index) * 4, 0);
    }
    command
}

fn agc_dcb_dma_data(arguments: [u64; 6], stack_arguments: [u64; 6]) -> u64 {
    let byte_count = low_u32(stack_arguments[2]);
    if arguments[0] == 0 || byte_count == 0 || byte_count & 3 != 0 {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(arguments[0], 8) else {
        return 0;
    };
    let destination = low_u32(arguments[1]) & 0xff;
    let destination_cache_policy = low_u32(arguments[2]) & 0xff;
    let source = low_u32(arguments[3]) & 0xff;
    let source_cache_policy = low_u32(arguments[5]) & 0xff;
    let control4 = low_u32(stack_arguments[0]) & 0xff;
    let source_address = stack_arguments[1];
    let control7 = low_u32(stack_arguments[3]) & 0xff;
    let control8 = low_u32(stack_arguments[4]) & 0xff;
    let control9 = low_u32(stack_arguments[5]) & 0xff;
    write_guest_u32(command, agc_pm4(8, AGC_IT_NOP, AGC_R_DMA_DATA));
    write_guest_u32(
        command + 4,
        destination
            | (destination_cache_policy << 8)
            | (source << 16)
            | (source_cache_policy << 24),
    );
    write_guest_u32(
        command + 8,
        control4 | (control7 << 8) | (control8 << 16) | (control9 << 24),
    );
    write_guest_u32(command + 12, byte_count);
    write_guest_u64(command + 16, arguments[4]);
    write_guest_u64(command + 24, source_address);
    command
}

fn agc_cb_release_mem(arguments: [u64; 6], stack_arguments: [u64; 6]) -> u64 {
    let destination = low_u32(arguments[3]) & 0xff;
    let data_selection = low_u32(stack_arguments[0]) & 0xff;
    let gds_offset = low_u32(stack_arguments[2]) & 0xffff;
    let gds_size = low_u32(stack_arguments[3]) & 0xffff;
    let interrupt = low_u32(stack_arguments[4]) & 0xff;
    if arguments[0] == 0
        || destination > 1
        || data_selection > 3
        || gds_offset != 0
        || gds_size > 2
        || interrupt > 3
    {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(arguments[0], 8) else {
        return 0;
    };
    let action = low_u32(arguments[1]) & 0xff;
    let gcr_control = low_u32(arguments[2]) & 0xffff;
    let cache_policy = low_u32(arguments[4]) & 0xff;
    let destination_address = arguments[5];
    let data = stack_arguments[1];
    write_guest_u32(command, agc_pm4(8, AGC_IT_NOP, AGC_R_RELEASE_MEM));
    write_guest_u32(command + 4, action | (cache_policy << 8));
    write_guest_u32(
        command + 8,
        gcr_control | (data_selection << 16) | (interrupt << 24),
    );
    write_guest_u32(command + 12, low_u32(destination_address));
    write_guest_u32(command + 16, (destination_address >> 32) as u32);
    write_guest_u32(command + 20, low_u32(data));
    write_guest_u32(command + 24, (data >> 32) as u32);
    write_guest_u32(command + 28, low_u32(stack_arguments[5]));
    command
}

fn agc_dcb_wait_reg_mem(arguments: [u64; 6], stack_arguments: [u64; 6]) -> u64 {
    let size = low_u32(arguments[1]) & 0xff;
    let compare_function = low_u32(arguments[2]) & 0xff;
    let operation = low_u32(arguments[3]) & 0xff;
    let cache_policy = low_u32(arguments[4]) & 0xff;
    if arguments[0] == 0 || size > 1 || compare_function > 7 || operation > 4 || cache_policy > 3 {
        return 0;
    }
    let standard_wait = matches!(operation, 2 | 3);
    let packet_dwords = if standard_wait {
        7
    } else if size == 0 {
        6
    } else {
        9
    };
    let Some(command) = agc_allocate_command_dwords(arguments[0], packet_dwords) else {
        return 0;
    };
    let address = arguments[5];
    let reference = stack_arguments[0];
    let mask = stack_arguments[1];
    let poll_cycles = low_u32(stack_arguments[2]);
    if standard_wait {
        write_guest_u32(command, agc_pm4(packet_dwords, AGC_IT_WAIT_REG_MEM, 0));
        write_guest_u32(command + 4, compare_function | ((operation & 1) << 8));
        write_guest_u32(command + 8, low_u32(address));
        write_guest_u32(command + 12, (address >> 32) as u32);
        write_guest_u32(command + 16, low_u32(reference));
        write_guest_u32(command + 20, low_u32(mask));
        write_guest_u32(command + 24, poll_cycles / 40);
    } else {
        let packet_register = if size == 0 {
            AGC_R_WAIT_MEM_32
        } else {
            AGC_R_WAIT_MEM_64
        };
        write_guest_u32(command, agc_pm4(packet_dwords, AGC_IT_NOP, packet_register));
        write_guest_u32(command + 4, low_u32(address));
        write_guest_u32(command + 8, (address >> 32) as u32);
        write_guest_u32(command + 12, low_u32(mask));
        if size == 0 {
            write_guest_u32(command + 16, compare_function | (operation << 8));
            write_guest_u32(command + 20, low_u32(reference));
        } else {
            write_guest_u32(command + 16, (mask >> 32) as u32);
            write_guest_u32(command + 20, low_u32(reference));
            write_guest_u32(command + 24, (reference >> 32) as u32);
            write_guest_u32(command + 28, compare_function | (operation << 8));
            write_guest_u32(command + 32, poll_cycles / 40);
        }
    }
    command
}

fn agc_packet_identity(command: u64) -> Option<(u32, u32, u32)> {
    if command == 0 {
        return None;
    }
    let header = read_guest_u32(command);
    if header >> 30 != 3 {
        return None;
    }
    Some((
        ((header >> 8) & 0xff),
        (header >> 2) & 0x3f,
        ((header >> 16) & 0x3fff) + 2,
    ))
}

fn agc_dma_data_patch_destination(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let destination = arguments[1];
    let Some((opcode, register, length)) = agc_packet_identity(command) else {
        return u64::MAX;
    };
    if opcode != AGC_IT_NOP || register != AGC_R_DMA_DATA {
        return u64::MAX;
    }
    let destination_offset = if length == 7 { 4 } else { 16 };
    write_guest_u64(command + destination_offset, destination);
    0
}

fn agc_wait_reg_mem_patch_address(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let address = arguments[1];
    let Some((opcode, register, _)) = agc_packet_identity(command) else {
        return u64::MAX;
    };
    let offset = if opcode == AGC_IT_WAIT_REG_MEM {
        8
    } else if opcode == AGC_IT_NOP && matches!(register, AGC_R_WAIT_MEM_32 | AGC_R_WAIT_MEM_64) {
        4
    } else {
        return u64::MAX;
    };
    write_guest_u64(command + offset, address);
    0
}

fn agc_release_mem_patch_address(arguments: [u64; 6]) -> u64 {
    let command = arguments[0];
    let Some((opcode, register, _)) = agc_packet_identity(command) else {
        return u64::MAX;
    };
    if opcode != AGC_IT_NOP || register != AGC_R_RELEASE_MEM {
        return u64::MAX;
    }
    write_guest_u64(command + 12, arguments[1]);
    0
}

fn agc_dcb_acquire_mem(arguments: [u64; 6], stack_arguments: [u64; 6]) -> u64 {
    let command_buffer = arguments[0];
    let engine = low_u32(arguments[1]) & 0xff;
    let cb_db_operation = low_u32(arguments[2]);
    let gcr_control = low_u32(arguments[3]);
    let base_address = arguments[4];
    let size_bytes = arguments[5];
    let no_size = size_bytes == u64::MAX;
    if command_buffer == 0
        || engine > 1
        || (!no_size && (size_bytes & 0xff != 0 || size_bytes >> 40 != 0))
        || base_address & 0xff != 0
        || base_address >> 40 != 0
    {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(command_buffer, 8) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(8, AGC_IT_NOP, AGC_R_ACQUIRE_MEM));
    write_guest_u32(command + 4, (engine << 31) | cb_db_operation);
    write_guest_u32(
        command + 8,
        if no_size { 0 } else { low_u32(size_bytes >> 8) },
    );
    write_guest_u32(command + 12, 0);
    write_guest_u32(command + 16, low_u32(base_address >> 8));
    write_guest_u32(command + 20, 0);
    write_guest_u32(command + 24, low_u32(stack_arguments[0]) / 40);
    write_guest_u32(command + 28, gcr_control);
    command
}

fn relocate_guest_pointer(field: u64) {
    let relative = read_guest_u64(field);
    if relative != 0 {
        write_guest_u64(field, field.wrapping_add(relative));
    }
}

fn patch_agc_shader_program_registers(header: u64, code: u64) -> bool {
    let registers = read_guest_u64(header + AGC_SHADER_SH_REGISTERS_OFFSET);
    let shader_type = read_guest_u8(header + AGC_SHADER_TYPE_OFFSET);
    let register_count = read_guest_u8(header + AGC_SHADER_SH_REGISTER_COUNT_OFFSET);
    if registers == 0 || register_count < 2 {
        return false;
    }
    let expected = match shader_type {
        0 => (0x20c, 0x20d),
        1 => (0x008, 0x009),
        2 | 6 => (0x0c8, 0x0c9),
        4 => (0x08a, 0x08b),
        7 => (0x148, 0x149),
        _ => return false,
    };
    if (read_guest_u32(registers), read_guest_u32(registers + 8)) != expected {
        return false;
    }
    write_guest_u32(registers + 4, ((code >> 8) & 0xffff_ffff) as u32);
    write_guest_u32(registers + 12, ((code >> 40) & 0xff) as u32);
    true
}

fn video_out_set_buffer_attribute2(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let attribute = arguments[0];
    if attribute == 0 {
        return u64::MAX;
    }
    guest_memset(attribute, 0, 0x50);
    write_guest_u32(attribute + 0x04, low_u32(arguments[2]));
    write_guest_u32(attribute + 0x0c, low_u32(arguments[3]));
    write_guest_u32(attribute + 0x10, low_u32(arguments[4]));
    write_guest_u64(attribute + 0x18, arguments[5]);
    write_guest_u64(attribute + 0x20, arguments[1]);

    state.video_out.pixel_format = arguments[1];
    state.video_out.tiling_mode = low_u32(arguments[2]);
    state.video_out.width = low_u32(arguments[3]);
    state.video_out.height = low_u32(arguments[4]);
    0
}

fn video_out_register_buffers2(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let set_index = low_u32(arguments[1]);
    let buffer_index_start = usize::try_from(low_u32(arguments[2])).unwrap_or(usize::MAX);
    let entries = arguments[3];
    let buffer_count = usize::try_from(low_u32(arguments[4])).unwrap_or(usize::MAX);
    let attribute = arguments[5];
    if arguments[0] == 0
        || entries == 0
        || attribute == 0
        || buffer_index_start > 15
        || buffer_count == 0
        || buffer_count > 16
        || buffer_index_start.saturating_add(buffer_count) > 16
    {
        return u64::MAX;
    }

    state.video_out.pixel_format = read_guest_u64(attribute + 0x20);
    state.video_out.tiling_mode = read_guest_u32(attribute + 0x04);
    state.video_out.width = read_guest_u32(attribute + 0x0c);
    state.video_out.height = read_guest_u32(attribute + 0x10);
    state
        .video_out
        .display_buffers
        .resize(buffer_index_start + buffer_count, 0);
    for index in 0..buffer_count {
        let entry = entries + u64::try_from(index).unwrap_or(u64::MAX) * 0x20;
        state.video_out.display_buffers[buffer_index_start + index] = read_guest_u64(entry);
    }
    eprintln!(
        "[REMu][VIDEO] registered {} buffers at {}x{} format={:#x} tiling={} addresses={:x?}",
        buffer_count,
        state.video_out.width,
        state.video_out.height,
        state.video_out.pixel_format,
        state.video_out.tiling_mode,
        state.video_out.display_buffers
    );
    remu_gpu::emit_event(remu_gpu::GpuEvent::DisplayRegistered {
        width: state.video_out.width,
        height: state.video_out.height,
        pitch: state.video_out.width,
        pixel_format: state.video_out.pixel_format,
        tiling_mode: state.video_out.tiling_mode,
        buffers: state.video_out.display_buffers.clone(),
    });
    u64::from(set_index)
}

fn video_out_add_flip_event(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let equeue = arguments[0];
    let handle = low_i32(arguments[1]);
    if equeue == 0 || handle != 1 {
        return u64::MAX;
    }
    state.video_out.flip_events.insert(equeue, arguments[2]);
    0
}

fn agc_dcb_set_flip(arguments: [u64; 6]) -> u64 {
    let command_buffer = arguments[0];
    if command_buffer == 0 {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(command_buffer, 6) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(6, AGC_IT_NOP, AGC_R_FLIP));
    write_guest_u32(command + 4, low_u32(arguments[1]));
    write_guest_u32(command + 8, low_u32(arguments[2]));
    write_guest_u32(command + 12, low_u32(arguments[3]));
    write_guest_u32(command + 16, low_u32(arguments[4]));
    write_guest_u32(command + 20, low_u32(arguments[4] >> 32));
    command
}

fn agc_dcb_set_index_buffer(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(command) = agc_allocate_command_dwords(arguments[0], 5) else {
        return 0;
    };
    // Gen5's four-argument form carries a CPU-readable index pointer, its GPU resource address,
    // and the element capacity. Keep the association so the host renderer can read the indices
    // while preserving the resource address needed for later descriptor translation.
    let index_buffer_address = arguments[1];
    let index_resource_address = arguments[2];
    let index_count = low_u32(arguments[3]);
    state
        .agc_index_resources
        .insert(index_buffer_address, index_resource_address);
    write_guest_u32(command, agc_pm4(3, AGC_IT_INDEX_BASE, 0));
    write_guest_u32(command + 4, low_u32(index_buffer_address));
    write_guest_u32(command + 8, low_u32(index_buffer_address >> 32));
    write_guest_u32(command + 12, agc_pm4(2, AGC_IT_INDEX_BUFFER_SIZE, 0));
    write_guest_u32(command + 16, index_count);
    command
}

fn agc_dcb_set_index_size(arguments: [u64; 6]) -> u64 {
    if low_u32(arguments[2]) != 0 {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(arguments[0], 2) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(2, AGC_IT_INDEX_TYPE, 0));
    write_guest_u32(command + 4, low_u32(arguments[1]) & 0xff);
    command
}

fn agc_dcb_set_num_instances(arguments: [u64; 6]) -> u64 {
    let Some(command) = agc_allocate_command_dwords(arguments[0], 2) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(2, AGC_IT_NUM_INSTANCES, 0));
    write_guest_u32(command + 4, low_u32(arguments[1]));
    command
}

fn agc_dcb_draw_index_auto(arguments: [u64; 6]) -> u64 {
    if low_u32(arguments[2]) != 0x4000_0000 {
        return 0;
    }
    let Some(command) = agc_allocate_command_dwords(arguments[0], 7) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(7, AGC_IT_NOP, AGC_R_DRAW_INDEX_AUTO));
    write_guest_u32(command + 4, low_u32(arguments[1]));
    for index in 2..7_u64 {
        write_guest_u32(command + index * 4, 0);
    }
    command
}

fn agc_dcb_draw_index_offset(arguments: [u64; 6]) -> u64 {
    let Some(command) = agc_allocate_command_dwords(arguments[0], 5) else {
        return 0;
    };
    write_guest_u32(command, agc_pm4(5, AGC_IT_DRAW_INDEX_OFFSET, 0));
    write_guest_u32(command + 4, low_u32(arguments[2]));
    write_guest_u32(command + 8, low_u32(arguments[1]));
    write_guest_u32(command + 12, low_u32(arguments[2]));
    write_guest_u32(command + 16, low_u32(arguments[3]) & 0xe000_0001);
    command
}

fn queue_video_out_flip(
    state: &mut ProbeDispatchState,
    handle: u32,
    buffer_index: u32,
    flip_argument: u64,
) {
    let Some(&display_address) = state.video_out.display_buffers.get(buffer_index as usize) else {
        eprintln!("[REMu][VIDEO] ignored flip for unregistered buffer index {buffer_index}");
        return;
    };
    let event_data = 6 | ((flip_argument & 0x0000_ffff_ffff_ffff) << 16);
    remu_gpu::emit_event(remu_gpu::GpuEvent::Flip {
        address: display_address,
        buffer_index,
        flip_argument,
    });
    for (&equeue, &user_data) in &state.video_out.flip_events {
        state.pending_events.push_back(KernelQueuedEvent {
            equeue,
            ident: 6,
            filter: -13,
            flags: 0,
            fflags: 0,
            data: event_data,
            user_data,
        });
        if state.agc_submission_count <= 8 {
            eprintln!(
                "[REMu][VIDEO] queued flip handle={handle} index={buffer_index} address={display_address:#x} equeue={equeue:#x}"
            );
        }
    }
}

fn agc_apply_submission_registers(
    packet_address: u64,
    length: u32,
    opcode: u32,
    register: u32,
    graphics: &mut AgcGraphicsState,
) {
    if matches!(
        opcode,
        AGC_IT_SET_CONTEXT_REG | AGC_IT_SET_SH_REG | AGC_IT_SET_UCONFIG_REG
    ) {
        if length < 3 {
            return;
        }
        let first_register = read_guest_u32(packet_address + 4);
        let destination = match opcode {
            AGC_IT_SET_CONTEXT_REG => &mut graphics.context_registers,
            AGC_IT_SET_SH_REG => &mut graphics.shader_registers,
            _ => &mut graphics.user_config_registers,
        };
        for index in 0..length - 2 {
            destination.insert(
                first_register + index,
                read_guest_u32(packet_address + u64::from(index + 2) * 4),
            );
        }
        return;
    }

    if opcode != AGC_IT_NOP
        || !matches!(
            register,
            AGC_R_SH_REGS_INDIRECT | AGC_R_CX_REGS_INDIRECT | AGC_R_UC_REGS_INDIRECT
        )
        || length < 4
    {
        return;
    }
    let register_count = read_guest_u32(packet_address + 4).min(4_096);
    let registers_address = read_guest_u64(packet_address + 8);
    if registers_address == 0 {
        return;
    }
    let destination = match register {
        AGC_R_CX_REGS_INDIRECT => &mut graphics.context_registers,
        AGC_R_SH_REGS_INDIRECT => &mut graphics.shader_registers,
        _ => &mut graphics.user_config_registers,
    };
    for index in 0..register_count {
        let entry = registers_address + u64::from(index) * 8;
        destination.insert(read_guest_u32(entry), read_guest_u32(entry + 4));
    }
}

fn agc_program_address(registers: &BTreeMap<u32, u32>, low: u32, high: u32) -> u64 {
    let Some((&low, &high)) = registers.get(&low).zip(registers.get(&high)) else {
        return 0;
    };
    (u64::from(high) << 40) | (u64::from(low) << 8)
}

fn agc_color_target_address(registers: &BTreeMap<u32, u32>) -> u64 {
    let Some((&low, &high)) = registers
        .get(&AGC_CB_COLOR0_BASE)
        .zip(registers.get(&AGC_CB_COLOR0_BASE_EXT))
    else {
        return 0;
    };
    (u64::from(high & 0xff) << 40) | (u64::from(low) << 8)
}

fn agc_register_snapshot(registers: &BTreeMap<u32, u32>) -> Vec<remu_gpu::GpuRegister> {
    registers
        .iter()
        .map(|(&offset, &value)| remu_gpu::GpuRegister { offset, value })
        .collect()
}

fn agc_shader_prefix_snapshot(address: u64) -> [u32; 4] {
    if address == 0 {
        return [0; 4];
    }
    std::array::from_fn(|index| {
        read_guest_u32(address + u64::try_from(index).expect("shader prefix index fits") * 4)
    })
}

fn agc_emit_draw(
    graphics: &mut AgcGraphicsState,
    indexed: bool,
    element_offset: u32,
    element_count: u32,
    draw_flags: u32,
) {
    if element_count == 0 {
        return;
    }
    graphics.draw_index_offset = element_offset;
    graphics.emitted_draws = graphics.emitted_draws.saturating_add(1);

    let export_shader_address = agc_program_address(
        &graphics.shader_registers,
        AGC_SPI_SHADER_PGM_LO_ES,
        AGC_SPI_SHADER_PGM_HI_ES,
    );
    let pixel_shader_address = agc_program_address(
        &graphics.shader_registers,
        AGC_SPI_SHADER_PGM_LO_PS,
        AGC_SPI_SHADER_PGM_HI_PS,
    );
    let color_target_address = agc_color_target_address(&graphics.context_registers);
    let primitive_type = graphics
        .user_config_registers
        .get(&AGC_VGT_PRIMITIVE_TYPE)
        .copied()
        .unwrap_or_default();
    let draw = remu_gpu::IndexedDraw {
        sequence: graphics.emitted_draws,
        indexed,
        instance_count: graphics.instance_count.max(1),
        index_buffer_address: graphics.index_buffer_address,
        index_resource_address: graphics.index_resource_address,
        index_buffer_count: graphics.index_buffer_count,
        index_size: graphics.index_size,
        index_offset: element_offset,
        index_count: element_count,
        draw_flags,
        primitive_type,
        export_shader_address,
        pixel_shader_address,
        export_shader_prefix: agc_shader_prefix_snapshot(export_shader_address),
        pixel_shader_prefix: agc_shader_prefix_snapshot(pixel_shader_address),
        color_target_address,
        context_registers: agc_register_snapshot(&graphics.context_registers),
        shader_registers: agc_register_snapshot(&graphics.shader_registers),
        user_config_registers: agc_register_snapshot(&graphics.user_config_registers),
    };
    if draw.sequence <= 8 {
        eprintln!(
            "[REMu][AGC] draw={} indexed={} count={} instances={} offset={} index_buffer={:#x}/{} resource={:#x} index_size={} primitive={:#x} es={:#x} ps={:#x} target={:#x}",
            draw.sequence,
            draw.indexed,
            draw.index_count,
            draw.instance_count,
            draw.index_offset,
            draw.index_buffer_address,
            draw.index_buffer_count,
            draw.index_resource_address,
            draw.index_size,
            draw.primitive_type,
            draw.export_shader_address,
            draw.pixel_shader_address,
            draw.color_target_address,
        );
    }
    if graphics
        .traced_shader_pairs
        .insert((draw.export_shader_address, draw.pixel_shader_address))
        && graphics.traced_shader_pairs.len() <= 16
    {
        eprintln!(
            "[REMu][AGC][SHADER] es={:#x} ps={:#x} indexed={} count={} instances={} target={:#x} es_prefix={:x?} ps_prefix={:x?} cx={:x?} sh={:x?} uc={:x?}",
            draw.export_shader_address,
            draw.pixel_shader_address,
            draw.indexed,
            draw.index_count,
            draw.instance_count,
            draw.color_target_address,
            draw.export_shader_prefix,
            draw.pixel_shader_prefix,
            draw.context_registers,
            draw.shader_registers,
            draw.user_config_registers,
        );
    }
    remu_gpu::emit_event(remu_gpu::GpuEvent::IndexedDraw(draw));
}

fn agc_emit_indexed_draw(packet_address: u64, graphics: &mut AgcGraphicsState) {
    agc_emit_draw(
        graphics,
        true,
        read_guest_u32(packet_address + 8),
        read_guest_u32(packet_address + 12),
        read_guest_u32(packet_address + 16),
    );
}

fn agc_handle_submission_packet(
    packet_address: u64,
    length: u32,
    opcode: u32,
    register: u32,
    state: &mut ProbeDispatchState,
    flips: &mut Vec<(u32, u32, u64)>,
) {
    if opcode == AGC_IT_NOP && register == AGC_R_DRAW_RESET && length >= 2 {
        state.agc_graphics.reset_pipeline();
    }
    agc_apply_submission_registers(
        packet_address,
        length,
        opcode,
        register,
        &mut state.agc_graphics,
    );
    if opcode == AGC_IT_INDEX_BASE && length >= 3 {
        state.agc_graphics.index_buffer_address = u64::from(read_guest_u32(packet_address + 4))
            | (u64::from(read_guest_u32(packet_address + 8)) << 32);
        state.agc_graphics.index_resource_address = state
            .agc_index_resources
            .get(&state.agc_graphics.index_buffer_address)
            .copied()
            .unwrap_or_default();
    }
    if opcode == AGC_IT_INDEX_BUFFER_SIZE && length >= 2 {
        state.agc_graphics.index_buffer_count = read_guest_u32(packet_address + 4);
    }
    if opcode == AGC_IT_INDEX_TYPE && length >= 2 {
        state.agc_graphics.index_size = read_guest_u32(packet_address + 4) & 0x3;
    }
    if opcode == AGC_IT_NUM_INSTANCES && length >= 2 {
        state.agc_graphics.instance_count = read_guest_u32(packet_address + 4).max(1);
    }
    if opcode == AGC_IT_DRAW_INDEX_OFFSET && length >= 5 {
        agc_emit_indexed_draw(packet_address, &mut state.agc_graphics);
    }
    if opcode == AGC_IT_DRAW_INDEX_AUTO && length >= 3 {
        agc_emit_draw(
            &mut state.agc_graphics,
            false,
            0,
            read_guest_u32(packet_address + 4),
            read_guest_u32(packet_address + 8),
        );
    }
    if opcode == AGC_IT_NOP && register == AGC_R_DRAW_INDEX_AUTO && length >= 2 {
        agc_emit_draw(
            &mut state.agc_graphics,
            false,
            0,
            read_guest_u32(packet_address + 4),
            0,
        );
    }
    if opcode == AGC_IT_NOP && register == AGC_R_FLIP && length >= 6 {
        let handle = read_guest_u32(packet_address + 4);
        let buffer_index = read_guest_u32(packet_address + 8);
        let flip_argument = u64::from(read_guest_u32(packet_address + 16))
            | (u64::from(read_guest_u32(packet_address + 20)) << 32);
        flips.push((handle, buffer_index, flip_argument));
    }
}

fn agc_parse_submission(command_address: u64, dword_count: u32, state: &mut ProbeDispatchState) {
    let mut offset = 0_u32;
    let mut packets = BTreeMap::<(u32, u32), u32>::new();
    let mut flips = Vec::new();
    while offset < dword_count {
        let packet_address = command_address + u64::from(offset) * 4;
        let header = read_guest_u32(packet_address);
        let packet_type = header >> 30;
        if packet_type == 2 {
            offset += 1;
            continue;
        }
        if packet_type != 3 {
            eprintln!(
                "[REMu][AGC] stopped submission parse at dw={offset}: unsupported packet type {packet_type} header={header:#010x}"
            );
            break;
        }
        let length = ((header >> 16) & 0x3fff) + 2;
        if length == 0 || length > dword_count - offset {
            eprintln!(
                "[REMu][AGC] stopped submission parse at dw={offset}: length={length} remaining={}",
                dword_count - offset
            );
            break;
        }
        let opcode = (header >> 8) & 0xff;
        let register = (header >> 2) & 0x3f;
        *packets.entry((opcode, register)).or_default() += 1;
        agc_handle_submission_packet(packet_address, length, opcode, register, state, &mut flips);
        offset += length;
    }

    state.agc_submission_count = state.agc_submission_count.saturating_add(1);
    if state.agc_submission_count <= 8 {
        let packet_summary = packets
            .iter()
            .map(|(&(opcode, register), &count)| {
                format!("op={opcode:#04x}/reg={register:#04x}:{count}")
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "[REMu][AGC] submit={} address={command_address:#x} dwords={dword_count} parsed={offset} packets=[{packet_summary}] flips={}",
            state.agc_submission_count,
            flips.len()
        );
    }
    for (handle, buffer_index, flip_argument) in flips {
        queue_video_out_flip(state, handle, buffer_index, flip_argument);
    }
}

fn agc_driver_submit_dcb(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let packet = arguments[0];
    if packet == 0 {
        return u64::MAX;
    }
    let command_address = read_guest_u64(packet);
    let dword_count = read_guest_u32(packet + 8);
    if command_address == 0 || dword_count == 0 || dword_count > 1_000_000 {
        return u64::MAX;
    }
    agc_parse_submission(command_address, dword_count, state);
    0
}

fn agc_driver_submit_multi_dcbs(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let addresses = arguments[0];
    let sizes = arguments[1];
    let count = low_u32(arguments[2]);
    if addresses == 0 || sizes == 0 || count == 0 || count > 4096 {
        return u64::MAX;
    }
    for index in 0..count {
        let command_address = read_guest_u64(addresses + u64::from(index) * 8);
        let dword_count = read_guest_u32(sizes + u64::from(index) * 4);
        if command_address != 0 && dword_count != 0 && dword_count <= 1_000_000 {
            agc_parse_submission(command_address, dword_count, state);
        }
    }
    0
}

fn guest_ctype_table(state: &mut ProbeDispatchState) -> u64 {
    if state.ctype_table == 0 {
        // Sony's libc follows the Dinkumware mask layout and indexes this table over
        // [-128, 255]. Return the c == 0 entry, leaving 128 valid entries before it.
        let storage = guest_calloc(384, 2);
        if storage == 0 {
            return 0;
        }
        state.ctype_table = storage + 128 * 2;
        for character in 0_u16..=0x7f {
            write_guest_u16(
                state.ctype_table + u64::from(character) * 2,
                guest_ctype_flags(character),
            );
        }
    }
    state.ctype_table
}

fn guest_toupper_table(state: &mut ProbeDispatchState) -> u64 {
    if state.toupper_table == 0 {
        // Dinkumware indexes the signed-char range as well as 0..=255. Each entry is a 32-bit
        // character value and the exported pointer addresses the c == 0 element.
        let storage = guest_calloc(384, 4);
        if storage == 0 {
            return 0;
        }
        state.toupper_table = storage + 128 * 4;
        for index in 0_u32..384 {
            let character = i32::try_from(index).expect("ctype index fits i32") - 128;
            let mapped = if (i32::from(b'a')..=i32::from(b'z')).contains(&character) {
                character - i32::from(b'a') + i32::from(b'A')
            } else {
                character
            };
            write_guest_u32(storage + u64::from(index) * 4, mapped.cast_unsigned());
        }
    }
    state.toupper_table
}

fn guest_ctype_flags(character: u16) -> u16 {
    const XDIGIT: u16 = 0x001;
    const UPPER: u16 = 0x002;
    const SPACE: u16 = 0x004;
    const PUNCT: u16 = 0x008;
    const LOWER: u16 = 0x010;
    const DIGIT: u16 = 0x020;
    const CONTROL_SPACE: u16 = 0x040;
    const CONTROL: u16 = 0x080;
    const BLANK: u16 = 0x400;

    let upper = (u16::from(b'A')..=u16::from(b'Z')).contains(&character);
    let lower = (u16::from(b'a')..=u16::from(b'z')).contains(&character);
    let digit = (u16::from(b'0')..=u16::from(b'9')).contains(&character);
    let hexadecimal = digit
        || (u16::from(b'A')..=u16::from(b'F')).contains(&character)
        || (u16::from(b'a')..=u16::from(b'f')).contains(&character);
    let alphanumeric = upper || lower || digit;
    let control_space = (0x09..=0x0d).contains(&character);
    let control = character <= 0x08 || (0x0e..=0x1f).contains(&character) || character == 0x7f;
    let printable = (0x20..=0x7e).contains(&character);

    let mut flags = 0;
    if upper {
        flags |= UPPER;
    }
    if lower {
        flags |= LOWER;
    }
    if digit {
        flags |= DIGIT;
    }
    if hexadecimal {
        flags |= XDIGIT;
    }
    if character == u16::from(b' ') {
        flags |= SPACE | BLANK;
    }
    if character == u16::from(b'\t') {
        flags |= BLANK;
    }
    if control_space {
        flags |= CONTROL_SPACE;
    }
    if printable && !alphanumeric && character != u16::from(b' ') {
        flags |= PUNCT;
    }
    if control {
        flags |= CONTROL;
    }
    flags
}

fn run_guest_thread(entry: u64, argument: u64) -> u64 {
    type GuestThreadEntry = unsafe extern "C" fn(u64) -> u64;

    if entry == 0 {
        return 0;
    }
    // SAFETY: The guest supplied a mapped executable entry point through scePthreadCreate. Rosetta
    // executes it on a native host thread while the import boundary provides HLE synchronization.
    let entry: GuestThreadEntry = unsafe { std::mem::transmute(entry) };
    // SAFETY: The guest owns the argument and execution contract for its thread entry.
    unsafe { entry(argument) }
}

fn initialize_synthetic_handle(address: u64) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    write_guest_u64(address, address);
    0
}

fn destroy_synthetic_handle(address: u64) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    write_guest_u64(address, 0);
    0
}

fn pthread_attr_get_stack(arguments: [u64; 6]) -> u64 {
    if arguments[0] == 0 || arguments[1] == 0 || arguments[2] == 0 {
        return u64::MAX;
    }
    write_guest_u64(arguments[1], 0);
    write_guest_u64(arguments[2], 0x80000);
    0
}

fn np_create_handle(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let output = if arguments[0] != 0 {
        arguments[0]
    } else {
        arguments[1]
    };
    if output == 0 {
        return u64::MAX;
    }
    let handle = state.next_np_handle;
    state.next_np_handle = state.next_np_handle.saturating_add(1);
    write_output_i32(output, handle)
}

fn guest_guard_acquire(address: u64) -> u64 {
    if address == 0 {
        return 0;
    }
    // SAFETY: C++ emits a writable eight-byte guard object and passes its live address.
    let value = unsafe { std::ptr::read_unaligned(address as *const u64) };
    if value & 1 != 0 || value & 0xff00 != 0 {
        return 0;
    }
    write_guest_u64(address, 0x100);
    1
}

fn guest_guard_release(address: u64) -> u64 {
    if address != 0 {
        write_guest_u64(address, 1);
    }
    0
}

fn guest_guard_abort(address: u64) -> u64 {
    if address != 0 {
        write_guest_u64(address, 0);
    }
    0
}

fn guest_cxa_allocate_exception(size: u64) -> u64 {
    let Some(allocation_size) = size.checked_add(0x80) else {
        return 0;
    };
    let allocation = guest_calloc(1, allocation_size);
    if allocation == 0 {
        0
    } else {
        allocation.checked_add(0x80).unwrap_or(0)
    }
}

fn guest_cxa_free_exception(exception: u64) -> u64 {
    if let Some(allocation) = exception.checked_sub(0x80) {
        guest_free(allocation);
    }
    0
}

fn guest_cxa_throw(arguments: [u64; 6]) -> u64 {
    let exception = arguments[0];
    let type_info = arguments[1];
    eprintln!(
        "[REMu][CXX] guest exception object={exception:#x} type_info={type_info:#x} destructor={:#x} object_word0={:#x} object_word1={:#x}",
        arguments[2],
        read_guest_u64(exception),
        read_guest_u64(exception + 8),
    );
    terminate_probe(88);
}

fn guest_errno_address() -> u64 {
    let thread_pointer = current_guest_thread_pointer();
    if thread_pointer == 0 {
        0
    } else {
        thread_pointer.checked_add(0x40).unwrap_or(0)
    }
}

fn dispatch_guest_numeric_parser(nid: &str, arguments: [u64; 6]) -> ProbeImportReturn {
    if arguments[0] == 0 {
        return ProbeImportReturn::integer(0);
    }
    set_host_errno(0);
    if guest_errno_address() != 0 {
        write_guest_u32(guest_errno_address(), 0);
    }
    let end_pointer = arguments[1] as *mut *mut libc::c_char;
    let result = match nid {
        // libc strtoull
        "5OqszGpy7Mg" => {
            // SAFETY: The guest supplies a NUL-terminated string and optional writable end pointer.
            ProbeImportReturn::integer(unsafe {
                libc::strtoull(
                    arguments[0] as *const libc::c_char,
                    end_pointer,
                    low_i32(arguments[2]),
                )
            })
        }
        // libc strtoll
        "VOBg+iNwB-4" => {
            // SAFETY: The guest supplies a NUL-terminated string and optional writable end pointer.
            ProbeImportReturn::integer(unsafe {
                libc::strtoll(
                    arguments[0] as *const libc::c_char,
                    end_pointer,
                    low_i32(arguments[2]),
                )
                .cast_unsigned()
            })
        }
        // libc strtod. ProbeImportReturn's mixed INTEGER/SSE ABI returns this through XMM0 while
        // preserving RAX for every ordinary import.
        "2vDqwBlpF-o" => {
            // SAFETY: The guest supplies a NUL-terminated string and optional writable end pointer.
            ProbeImportReturn::float(unsafe {
                libc::strtod(arguments[0] as *const libc::c_char, end_pointer)
            })
        }
        _ => ProbeImportReturn::integer(0),
    };
    let guest_errno = guest_errno_address();
    if guest_errno != 0 {
        write_guest_u32(guest_errno, host_errno().cast_unsigned());
    }
    result
}

#[cfg(target_os = "macos")]
fn set_host_errno(value: i32) {
    // SAFETY: __error returns this host thread's writable errno cell.
    unsafe {
        *libc::__error() = value;
    }
}

#[cfg(target_os = "macos")]
fn host_errno() -> i32 {
    // SAFETY: __error returns this host thread's live errno cell.
    unsafe { *libc::__error() }
}

#[cfg(target_os = "linux")]
fn set_host_errno(value: i32) {
    // SAFETY: __errno_location returns this host thread's writable errno cell.
    unsafe {
        *libc::__errno_location() = value;
    }
}

#[cfg(target_os = "linux")]
fn host_errno() -> i32 {
    // SAFETY: __errno_location returns this host thread's live errno cell.
    unsafe { *libc::__errno_location() }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn set_host_errno(_value: i32) {}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn host_errno() -> i32 {
    0
}

fn write_guest_u64(address: u64, value: u64) {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe {
        std::ptr::write_unaligned(address as *mut u64, value);
    }
}

fn read_guest_u8(address: u64) -> u8 {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe { std::ptr::read_unaligned(address as *const u8) }
}

fn read_guest_u32(address: u64) -> u32 {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe { std::ptr::read_unaligned(address as *const u32) }
}

fn read_guest_u64(address: u64) -> u64 {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe { std::ptr::read_unaligned(address as *const u64) }
}

fn write_guest_u8(address: u64, value: u8) {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe {
        std::ptr::write_unaligned(address as *mut u8, value);
    }
}

fn write_guest_u32(address: u64, value: u32) {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe {
        std::ptr::write_unaligned(address as *mut u32, value);
    }
}

fn write_guest_u16(address: u64, value: u16) {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe {
        std::ptr::write_unaligned(address as *mut u16, value);
    }
}

fn write_output_u64(address: u64, value: u64) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    write_guest_u64(address, value);
    0
}

fn write_output_i32(address: u64, value: i32) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    // SAFETY: The HLE ABI identifies `address` as a writable 32-bit output field.
    unsafe {
        std::ptr::write_unaligned(address as *mut i32, value);
    }
    0
}

fn allocate_direct_memory(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let length = arguments[2];
    let alignment = arguments[3].max(0x4000);
    let output = arguments[5];
    if length == 0 || output == 0 || !alignment.is_power_of_two() {
        return u64::MAX;
    }
    let Some(offset) = align_up_u64(state.next_direct_offset, alignment) else {
        return u64::MAX;
    };
    let Some(next) = offset.checked_add(length) else {
        return u64::MAX;
    };
    if next > 16 * 1024 * 1024 * 1024 {
        return u64::MAX;
    }
    // SAFETY: The guest supplied an ABI output pointer in its live stack/image mapping.
    unsafe {
        std::ptr::write_unaligned(output as *mut u64, offset);
    }
    state.next_direct_offset = next;
    0
}

#[cfg(unix)]
fn map_direct_memory(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let output = arguments[0];
    let length = arguments[1];
    let alignment = arguments[5].max(0x4000);
    if output == 0 || length == 0 || !alignment.is_power_of_two() {
        return u64::MAX;
    }
    let Ok(length_usize) = usize::try_from(length) else {
        return u64::MAX;
    };
    let Ok(alignment_usize) = usize::try_from(alignment) else {
        return u64::MAX;
    };
    let Some(reservation_size) = length_usize.checked_add(alignment_usize) else {
        return u64::MAX;
    };
    let hint = align_up_u64(state.next_virtual_hint, alignment).unwrap_or(0);
    // SAFETY: Anonymous mmap creates a new process-local reservation. The hint is non-destructive
    // because MAP_FIXED is deliberately absent.
    let reservation = unsafe {
        libc::mmap(
            hint as *mut std::ffi::c_void,
            reservation_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    if reservation == libc::MAP_FAILED {
        return u64::MAX;
    }
    let reservation_address = reservation as usize;
    let Some(aligned_address) = reservation_address
        .checked_add(alignment_usize - 1)
        .map(|value| value & !(alignment_usize - 1))
    else {
        // SAFETY: Releases the reservation returned by mmap above.
        unsafe {
            libc::munmap(reservation, reservation_size);
        }
        return u64::MAX;
    };
    let prefix = aligned_address - reservation_address;
    let suffix = reservation_size - prefix - length_usize;
    // SAFETY: Prefix/suffix are page-aligned subranges of the live anonymous reservation.
    unsafe {
        if prefix != 0 {
            libc::munmap(reservation, prefix);
        }
        if suffix != 0 {
            libc::munmap(
                (aligned_address + length_usize) as *mut std::ffi::c_void,
                suffix,
            );
        }
        std::ptr::write_unaligned(output as *mut u64, aligned_address as u64);
    }
    state.next_virtual_hint = (aligned_address as u64).saturating_add(length);
    state
        .mapped_allocations
        .push((aligned_address as u64, length));
    0
}

#[cfg(not(unix))]
fn map_direct_memory(_arguments: [u64; 6], _state: &mut ProbeDispatchState) -> u64 {
    u64::MAX
}

fn guest_memset(destination: u64, value: u64, length: u64) -> u64 {
    if destination == 0 || length == 0 {
        return destination;
    }
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    // SAFETY: The call originates in guest code and supplies a live destination mapping.
    unsafe {
        std::ptr::write_bytes(destination as *mut u8, value.to_le_bytes()[0], length);
    }
    destination
}

fn guest_memcpy(destination: u64, source: u64, length: u64) -> u64 {
    if length == 0 {
        return destination;
    }
    if destination == 0 || source == 0 {
        return 0;
    }
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    // SAFETY: memcpy's guest ABI contract requires two valid non-overlapping ranges.
    unsafe {
        std::ptr::copy_nonoverlapping(source as *const u8, destination as *mut u8, length);
    }
    destination
}

fn guest_memmove(destination: u64, source: u64, length: u64) -> u64 {
    if length == 0 {
        return destination;
    }
    if destination == 0 || source == 0 {
        return 0;
    }
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    // SAFETY: memmove's guest ABI contract supplies two valid ranges; `copy` permits overlap.
    unsafe {
        std::ptr::copy(source as *const u8, destination as *mut u8, length);
    }
    destination
}

fn guest_malloc(size: u64) -> u64 {
    let Ok(size) = usize::try_from(size) else {
        return 0;
    };
    // SAFETY: The returned allocation is process-local and uses the same x86-64 pointer ABI.
    unsafe { libc::malloc(size) as usize as u64 }
}

fn guest_free(address: u64) -> u64 {
    if address != 0 {
        // SAFETY: The guest passes an allocation previously returned by this libc HLE family.
        unsafe {
            libc::free(address as *mut std::ffi::c_void);
        }
    }
    0
}

fn guest_calloc(count: u64, size: u64) -> u64 {
    let (Ok(count), Ok(size)) = (usize::try_from(count), usize::try_from(size)) else {
        return 0;
    };
    // SAFETY: libc validates the allocation-size multiplication and returns process-local memory.
    unsafe { libc::calloc(count, size) as usize as u64 }
}

fn guest_realloc(address: u64, size: u64) -> u64 {
    let Ok(size) = usize::try_from(size) else {
        return 0;
    };
    // SAFETY: `address` is null or belongs to the same libc allocation family.
    unsafe { libc::realloc(address as *mut std::ffi::c_void, size) as usize as u64 }
}

fn guest_aligned_alloc(alignment: u64, size: u64) -> u64 {
    let (Ok(mut alignment), Ok(size)) = (usize::try_from(alignment), usize::try_from(size)) else {
        return 0;
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return 0;
    }
    // Guest memalign accepts sub-pointer alignments; the host allocator requires at least
    // sizeof(void*), which still satisfies the guest's weaker request.
    alignment = alignment.max(std::mem::size_of::<usize>());
    let mut output = std::ptr::null_mut();
    // SAFETY: `output` is writable and alignment/size are represented in the host ABI.
    let result = unsafe { libc::posix_memalign(&raw mut output, alignment, size) };
    if result == 0 {
        output as usize as u64
    } else {
        0
    }
}

fn guest_posix_memalign(output: u64, alignment: u64, size: u64) -> u64 {
    if output == 0 {
        return libc::EINVAL as u64;
    }
    let (Ok(alignment), Ok(size)) = (usize::try_from(alignment), usize::try_from(size)) else {
        return libc::ENOMEM as u64;
    };
    let mut allocation = std::ptr::null_mut();
    // SAFETY: `allocation` is writable and libc validates the requested alignment.
    let result = unsafe { libc::posix_memalign(&raw mut allocation, alignment, size) };
    if result == 0 {
        write_guest_u64(output, allocation as usize as u64);
    }
    u64::try_from(result).unwrap_or(u64::MAX)
}

fn guest_strlen(address: u64) -> u64 {
    if address == 0 {
        return 0;
    }
    // SAFETY: The guest libc call contract supplies a valid NUL-terminated string.
    unsafe { libc::strlen(address as *const libc::c_char) as u64 }
}

fn guest_strcmp(left: u64, right: u64) -> u64 {
    if left == 0 || right == 0 {
        return u64::MAX;
    }
    // SAFETY: strcmp's guest ABI contract supplies two NUL-terminated strings.
    let result = unsafe { libc::strcmp(left as *const libc::c_char, right as *const libc::c_char) };
    signed_i32_return(result)
}

fn guest_strncmp(left: u64, right: u64, length: u64) -> u64 {
    if left == 0 || right == 0 {
        return u64::MAX;
    }
    let Ok(length) = usize::try_from(length) else {
        return u64::MAX;
    };
    // SAFETY: strncmp's guest ABI contract supplies readable ranges through `length`.
    let result = unsafe {
        libc::strncmp(
            left as *const libc::c_char,
            right as *const libc::c_char,
            length,
        )
    };
    signed_i32_return(result)
}

fn guest_strcasecmp(left: u64, right: u64) -> u64 {
    if left == 0 || right == 0 {
        return u64::from(left != right);
    }
    // SAFETY: strcasecmp's guest ABI contract supplies two NUL-terminated strings.
    let result =
        unsafe { libc::strcasecmp(left as *const libc::c_char, right as *const libc::c_char) };
    signed_i32_return(result)
}

fn guest_strstr(haystack: u64, needle: u64) -> u64 {
    if haystack == 0 || needle == 0 {
        return 0;
    }
    // SAFETY: strstr's guest ABI contract supplies two NUL-terminated strings. Its result points
    // into the unchanged haystack, which is already process-addressable guest memory.
    unsafe {
        libc::strstr(
            haystack as *const libc::c_char,
            needle as *const libc::c_char,
        ) as usize as u64
    }
}

#[cfg(target_arch = "x86_64")]
fn save_data_root(game_root: &Path) -> PathBuf {
    let base = std::env::var_os("REMU_SAVEDATA_DIR")
        .map_or_else(|| std::env::temp_dir().join("remu-savedata"), PathBuf::from);
    let title = game_root
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_suffix("-app0").or(Some(name)))
        .unwrap_or("unknown-title");
    base.join(sanitize_save_segment(title))
}

fn sanitize_save_segment(value: &str) -> String {
    let value = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if value.trim_matches('_').is_empty() {
        "default".to_owned()
    } else {
        value
    }
}

fn save_data_initialize(state: &ProbeDispatchState) -> u64 {
    match std::fs::create_dir_all(&state.save_root) {
        Ok(()) => {
            eprintln!(
                "[REMu][SAVEDATA] initialized root={:?}",
                state.save_root.display()
            );
            0
        }
        Err(error) => {
            eprintln!("[REMu][SAVEDATA] initialize failed: {error}");
            sce_kernel_error(0x809f_000b)
        }
    }
}

fn save_data_dir_name_search(arguments: [u64; 6]) -> u64 {
    let result = arguments[1];
    if arguments[0] == 0 || result == 0 {
        return sce_kernel_error(0x809f_0000);
    }
    // The launch path only needs an empty initial search result. These are the hitNum and setNum
    // fields of SceSaveDataDirNameSearchResult; all caller-owned output buffers remain untouched.
    write_guest_u32(result, 0);
    write_guest_u32(result + 0x14, 0);
    0
}

fn save_data_create_transaction_resource(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let id = state.next_save_transaction_resource.max(1);
    state.next_save_transaction_resource = id.saturating_add(1);
    let output = if arguments[2] == 0 {
        0
    } else if u16::try_from(arguments[2]).is_ok() {
        arguments[3]
    } else {
        arguments[2]
    };
    if output != 0 {
        write_guest_u32(output, id);
    }
    eprintln!("[REMu][SAVEDATA] transaction resource={id} output={output:#x}");
    0
}

fn save_data_mount3(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    const MOUNT_RESULT_SIZE: usize = 0x40;
    const MOUNT_POINT: &str = "/savedata0";
    let (mount, result) = (arguments[0], arguments[1]);
    if mount == 0 || result == 0 {
        return sce_kernel_error(0x809f_0000);
    }
    let user_id = low_i32(u64::from(read_guest_u32(mount)));
    let dir_name_address = read_guest_u64(mount + 0x08);
    let blocks = read_guest_u64(mount + 0x10);
    let system_blocks = read_guest_u64(mount + 0x18);
    let mount_mode = read_guest_u32(mount + 0x20);
    let resource = read_guest_u32(mount + 0x24);
    let mode = read_guest_u32(mount + 0x28);
    let Some(dir_name) = read_guest_cstring(dir_name_address) else {
        return sce_kernel_error(0x809f_0000);
    };
    if user_id < 0 || dir_name.is_empty() {
        return sce_kernel_error(0x809f_0000);
    }

    let slot = state.save_root.join(sanitize_save_segment(&dir_name));
    let existed = slot.is_dir();
    if let Err(error) = std::fs::create_dir_all(&slot) {
        eprintln!(
            "[REMu][SAVEDATA] mount failed path={:?}: {error}",
            slot.display()
        );
        return sce_kernel_error(0x809f_000b);
    }

    // SAFETY: sceSaveDataMount3 supplies a writable 0x40-byte result structure.
    unsafe {
        std::ptr::write_bytes(result as *mut u8, 0, MOUNT_RESULT_SIZE);
        std::ptr::copy_nonoverlapping(MOUNT_POINT.as_ptr(), result as *mut u8, MOUNT_POINT.len());
    }
    write_guest_u32(result + 0x1c, u32::from(!existed));
    state
        .save_mounts
        .insert(MOUNT_POINT.to_owned(), slot.clone());
    eprintln!(
        "[REMu][SAVEDATA] mount3 user={user_id} dir={dir_name:?} blocks={blocks:#x} system_blocks={system_blocks:#x} mount_mode={mount_mode:#x} resource={resource} mode={mode:#x} mount={MOUNT_POINT:?} host={:?}",
        slot.display()
    );
    0
}

fn save_data_umount2(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(mount_point) = read_guest_cstring(arguments[0]) else {
        return 0;
    };
    state.save_mounts.remove(&mount_point);
    eprintln!("[REMu][SAVEDATA] unmounted {mount_point:?}");
    0
}

fn guest_sincosf(value: f32, sin_address: u64, cos_address: u64) -> u64 {
    let (sin, cos) = value.sin_cos();
    if sin_address != 0 {
        write_guest_u32(sin_address, sin.to_bits());
    }
    if cos_address != 0 {
        write_guest_u32(cos_address, cos.to_bits());
    }
    0
}

fn guest_qsort(arguments: [u64; 6]) -> u64 {
    let (base, count, size, comparator) = (arguments[0], arguments[1], arguments[2], arguments[3]);
    let (Ok(count), Ok(size), Ok(comparator)) = (
        usize::try_from(count),
        usize::try_from(size),
        usize::try_from(comparator),
    ) else {
        return 0;
    };
    if base == 0
        || comparator == 0
        || count < 2
        || size == 0
        || count > 1_000_000
        || size > 1_048_576
    {
        return 0;
    }
    let Ok(base) = usize::try_from(base) else {
        return 0;
    };
    // The guest comparator is native x86-64 code in this Rosetta process. Dropping the dispatcher
    // lock before entering this function permits a comparator that itself crosses an import stub.
    let compare: unsafe extern "C" fn(*const u8, *const u8) -> i32 =
        unsafe { std::mem::transmute(comparator) };
    let mut key = vec![0_u8; size];
    for index in 1..count {
        // SAFETY: qsort's ABI guarantees a writable `count * size` byte range at `base`.
        unsafe {
            let current = (base + index * size) as *mut u8;
            std::ptr::copy_nonoverlapping(current, key.as_mut_ptr(), size);
            let mut insertion = index;
            while insertion > 0 {
                let previous = (base + (insertion - 1) * size) as *mut u8;
                if compare(previous, key.as_ptr()) <= 0 {
                    break;
                }
                std::ptr::copy(previous, previous.add(size), size);
                insertion -= 1;
            }
            std::ptr::copy_nonoverlapping(key.as_ptr(), (base + insertion * size) as *mut u8, size);
        }
    }
    0
}

fn guest_fopen(path_address: u64, mode_address: u64, state: &ProbeDispatchState) -> u64 {
    let (Some(guest_path), Some(mode)) = (
        read_guest_cstring(path_address),
        read_guest_cstring(mode_address),
    ) else {
        return 0;
    };
    let Some((host_path, writable)) = resolve_probe_guest_path(state, &guest_path) else {
        eprintln!("[REMu][VFS] rejected path {guest_path:?}");
        return 0;
    };
    if !mode.starts_with('r') && !writable {
        eprintln!("[REMu][VFS] rejected write mode {mode:?} for {guest_path:?}");
        return 0;
    }
    if writable
        && !mode.starts_with('r')
        && let Some(parent) = host_path.parent()
        && std::fs::create_dir_all(parent).is_err()
    {
        return 0;
    }
    let (Ok(host_path), Ok(mode)) = (
        CString::new(host_path.to_string_lossy().as_bytes()),
        CString::new(mode),
    ) else {
        return 0;
    };
    // SAFETY: Both strings are NUL-terminated and host libc owns the returned FILE object.
    let file = unsafe { libc::fopen(host_path.as_ptr(), mode.as_ptr()) };
    eprintln!(
        "[REMu][VFS] fopen guest={guest_path:?} host={:?} result={file:p}",
        host_path.to_string_lossy()
    );
    file as usize as u64
}

fn resolve_guest_path(game_root: &Path, guest_path: &str) -> Option<PathBuf> {
    let normalized = guest_path.replace('\\', "/");
    let relative = if let Some(path) = normalized.strip_prefix("/app0/") {
        path
    } else if normalized == "/app0" {
        ""
    } else if let Some((_, path)) = normalized.split_once("/app0/") {
        path
    } else if normalized.starts_with('/') {
        return None;
    } else {
        normalized.as_str()
    };
    resolve_path_beneath(game_root, relative)
}

fn resolve_probe_guest_path(
    state: &ProbeDispatchState,
    guest_path: &str,
) -> Option<(PathBuf, bool)> {
    let normalized = guest_path.replace('\\', "/");
    for (mount_point, host_root) in &state.save_mounts {
        if normalized == *mount_point {
            return Some((host_root.clone(), true));
        }
        if let Some(relative) = normalized.strip_prefix(&format!("{mount_point}/")) {
            return resolve_path_beneath(host_root, relative).map(|path| (path, true));
        }
    }
    resolve_guest_path(&state.game_root, &normalized).map(|path| (path, false))
}

fn resolve_path_beneath(root: &Path, relative: &str) -> Option<PathBuf> {
    let mut host = root.to_owned();
    // Console paths routinely contain duplicate separators (for example
    // `/app0//shaders/presenter.vert`). Treat them like POSIX does while keeping parent traversal
    // outside the read-only game root impossible.
    for component in relative.split('/') {
        match component {
            "" | "." => {}
            ".." => return None,
            component => host.push(component),
        }
    }
    Some(host)
}

fn read_guest_cstring(address: u64) -> Option<String> {
    if address == 0 {
        return None;
    }
    // SAFETY: Guest libc string arguments are NUL-terminated within live guest memory.
    let string = unsafe { std::ffi::CStr::from_ptr(address as *const libc::c_char) };
    string.to_str().ok().map(ToOwned::to_owned)
}

fn guest_fseek(handle: u64, offset: u64, whence: u64) -> u64 {
    if handle == 0 {
        return u64::MAX;
    }
    // SAFETY: fopen returned the handle, and libc validates offset/whence.
    signed_i32_return(unsafe {
        libc::fseek(
            handle as *mut libc::FILE,
            i64::from_ne_bytes(offset.to_ne_bytes()),
            low_i32(whence),
        )
    })
}

fn guest_ftell(handle: u64) -> u64 {
    if handle == 0 {
        return u64::MAX;
    }
    // SAFETY: fopen returned the live FILE handle.
    unsafe { libc::ftell(handle as *mut libc::FILE).cast_unsigned() }
}

fn guest_fclose(handle: u64) -> u64 {
    if handle == 0 {
        return u64::MAX;
    }
    // SAFETY: fopen returned the live FILE handle and ownership ends here.
    signed_i32_return(unsafe { libc::fclose(handle as *mut libc::FILE) })
}

fn guest_fread(destination: u64, size: u64, count: u64, handle: u64) -> u64 {
    if destination == 0 || handle == 0 || size == 0 || count == 0 {
        return 0;
    }
    let (Ok(size), Ok(count)) = (usize::try_from(size), usize::try_from(count)) else {
        return 0;
    };
    // SAFETY: The guest supplies a destination sized for size*count and fopen returned the handle.
    unsafe {
        libc::fread(
            destination as *mut libc::c_void,
            size,
            count,
            handle as *mut libc::FILE,
        ) as u64
    }
}

const SCE_KERNEL_ERROR_PERMISSION_DENIED: u32 = 0x8002_0001;
const SCE_KERNEL_ERROR_NOT_FOUND: u32 = 0x8002_0002;
const SCE_KERNEL_ERROR_INVALID_ARGUMENT: u32 = 0x8002_0003;
const SCE_KERNEL_ERROR_MEMORY_FAULT: u32 = 0x8002_0101;
const SCE_KERNEL_STAT_SIZE: usize = 120;

fn sce_kernel_error(code: u32) -> u64 {
    signed_i32_return(i32::from_ne_bytes(code.to_ne_bytes()))
}

fn guest_kernel_open(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(guest_path) = read_guest_cstring(arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    let flags = low_i32(arguments[1]);
    let Some((host_path, writable)) = resolve_probe_guest_path(state, &guest_path) else {
        eprintln!("[REMu][VFS] sceKernelOpen rejected guest={guest_path:?}");
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if flags & 0x3 != 0 && !writable {
        eprintln!(
            "[REMu][VFS] sceKernelOpen rejected write access guest={guest_path:?} flags={flags:#x}"
        );
        return sce_kernel_error(SCE_KERNEL_ERROR_PERMISSION_DENIED);
    }
    let access = flags & 0x3;
    let mut options = std::fs::OpenOptions::new();
    options
        .read(access != 1)
        .write(access != 0)
        .append(flags & 0x8 != 0)
        .create(writable && flags & 0x200 != 0)
        .truncate(writable && flags & 0x400 != 0);
    if writable
        && flags & 0x200 != 0
        && let Some(parent) = host_path.parent()
        && std::fs::create_dir_all(parent).is_err()
    {
        return sce_kernel_error(SCE_KERNEL_ERROR_PERMISSION_DENIED);
    }
    let Ok(file) = options.open(&host_path) else {
        eprintln!(
            "[REMu][VFS] sceKernelOpen missing guest={guest_path:?} host={:?}",
            host_path.display()
        );
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let fd = state.next_kernel_fd.max(3);
    state.next_kernel_fd = fd.saturating_add(1);
    state.kernel_files.insert(fd, file);
    eprintln!(
        "[REMu][VFS] sceKernelOpen guest={guest_path:?} host={:?} flags={flags:#x} fd={fd}",
        host_path.display()
    );
    signed_i32_return(fd)
}

fn guest_kernel_stat(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(guest_path) = read_guest_cstring(arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    let Some((host_path, _)) = resolve_probe_guest_path(state, &guest_path) else {
        eprintln!("[REMu][VFS] sceKernelStat rejected guest={guest_path:?}");
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let Ok(metadata) = std::fs::metadata(&host_path) else {
        eprintln!(
            "[REMu][VFS] sceKernelStat missing guest={guest_path:?} host={:?}",
            host_path.display()
        );
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let payload = kernel_stat_payload(metadata.len(), metadata.is_dir(), guest_path.as_bytes());
    // SAFETY: sceKernelStat supplies a writable 120-byte Gen5 stat object in live guest memory.
    unsafe {
        std::ptr::copy_nonoverlapping(payload.as_ptr(), arguments[1] as *mut u8, payload.len());
    }
    eprintln!(
        "[REMu][VFS] sceKernelStat guest={guest_path:?} host={:?} size={}",
        host_path.display(),
        metadata.len()
    );
    0
}

fn guest_kernel_read(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let fd = low_i32(arguments[0]);
    let destination = arguments[1];
    let requested = arguments[2];
    if requested == 0 {
        return 0;
    }
    if destination == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Ok(requested) = usize::try_from(requested) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if requested > 128 * 1024 * 1024 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(file) = state.kernel_files.get_mut(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    // SAFETY: sceKernelRead supplies a writable range of exactly `requested` bytes in live guest
    // memory, and the global import dispatcher serializes access to the associated File.
    let destination = unsafe { std::slice::from_raw_parts_mut(destination as *mut u8, requested) };
    let Ok(read) = file.read(destination) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    eprintln!("[REMu][VFS] sceKernelRead fd={fd} requested={requested} read={read}");
    u64::try_from(read).expect("read byte count fits u64")
}

fn guest_kernel_close(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let fd = low_i32(arguments[0]);
    if state.kernel_files.remove(&fd).is_none() {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    eprintln!("[REMu][VFS] sceKernelClose fd={fd}");
    0
}

fn kernel_stat_payload(size: u64, is_directory: bool, inode_seed: &[u8]) -> [u8; 120] {
    let mut payload = [0_u8; SCE_KERNEL_STAT_SIZE];
    let inode = inode_seed.iter().fold(0x811c_9dc5_u32, |hash, &byte| {
        (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
    });
    payload[4..8].copy_from_slice(&inode.to_le_bytes());
    payload[8..10]
        .copy_from_slice(&(if is_directory { 0x41ff_u16 } else { 0x81ff_u16 }).to_le_bytes());
    payload[10..12].copy_from_slice(&1_u16.to_le_bytes());
    let size = i64::try_from(size).unwrap_or(i64::MAX);
    payload[72..80].copy_from_slice(&size.to_le_bytes());
    let blocks = if is_directory {
        128
    } else {
        (size + 511) / 512
    };
    payload[80..88].copy_from_slice(&blocks.to_le_bytes());
    payload[88..92]
        .copy_from_slice(&(if is_directory { 65_536_u32 } else { 512_u32 }).to_le_bytes());
    payload
}

fn guest_memcmp(left: u64, right: u64, length: u64) -> u64 {
    if length == 0 {
        return 0;
    }
    if left == 0 || right == 0 {
        return u64::MAX;
    }
    let Ok(length) = usize::try_from(length) else {
        return u64::MAX;
    };
    // SAFETY: memcmp's guest ABI contract supplies readable ranges through `length`.
    signed_i32_return(unsafe {
        libc::memcmp(
            left as *const libc::c_void,
            right as *const libc::c_void,
            length,
        )
    })
}

fn guest_memchr(address: u64, character: u64, length: u64) -> u64 {
    if address == 0 || length == 0 {
        return 0;
    }
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    // SAFETY: memchr's guest ABI contract supplies a readable range through `length`.
    unsafe {
        libc::memchr(address as *const libc::c_void, low_i32(character), length) as usize as u64
    }
}

fn guest_sprintf(arguments: [u64; 6]) -> u64 {
    if arguments[0] == 0 || arguments[1] == 0 {
        return u64::MAX;
    }
    // SAFETY: The guest supplies a writable destination, a valid format string, and SysV integer
    // or pointer varargs. Passing all four available argument registers preserves their bit layout.
    signed_i32_return(unsafe {
        libc::sprintf(
            arguments[0] as *mut libc::c_char,
            arguments[1] as *const libc::c_char,
            arguments[2],
            arguments[3],
            arguments[4],
            arguments[5],
        )
    })
}

fn guest_strcpy(destination: u64, source: u64) -> u64 {
    if destination == 0 || source == 0 {
        return 0;
    }
    // SAFETY: strcpy's guest ABI contract supplies a large-enough destination and C string source.
    unsafe {
        libc::strcpy(
            destination as *mut libc::c_char,
            source as *const libc::c_char,
        );
    }
    destination
}

fn guest_strncpy(destination: u64, source: u64, length: u64) -> u64 {
    if destination == 0 || source == 0 {
        return 0;
    }
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    // SAFETY: strncpy's guest ABI contract supplies readable/writable ranges through `length`.
    unsafe {
        libc::strncpy(
            destination as *mut libc::c_char,
            source as *const libc::c_char,
            length,
        );
    }
    destination
}

fn guest_strcat(destination: u64, source: u64) -> u64 {
    if destination == 0 || source == 0 {
        return 0;
    }
    // SAFETY: strcat's guest ABI contract supplies a large-enough destination and C string source.
    unsafe {
        libc::strcat(
            destination as *mut libc::c_char,
            source as *const libc::c_char,
        );
    }
    destination
}

fn signed_i32_return(value: i32) -> u64 {
    u64::from_ne_bytes(i64::from(value).to_ne_bytes())
}

fn guest_usleep(microseconds: u64) -> u64 {
    let bounded = microseconds.min(u64::from(u32::MAX));
    let Ok(bounded) = u32::try_from(bounded) else {
        return u64::MAX;
    };
    // SAFETY: usleep has no pointer arguments and the duration is bounded to its host ABI width.
    signed_i32_return(unsafe { libc::usleep(bounded) })
}

fn guest_clock_gettime(clock_id: u64, output: u64) -> u64 {
    if output == 0 {
        return u64::MAX;
    }
    // SAFETY: Darwin and the PS5 x86-64 ABI both represent timespec as two signed 64-bit fields.
    signed_i32_return(unsafe {
        libc::clock_gettime(low_u32(clock_id), output as *mut libc::timespec)
    })
}

fn guest_fputs(address: u64) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    // SAFETY: The guest libc call contract supplies a valid NUL-terminated string.
    let text = unsafe { std::ffi::CStr::from_ptr(address as *const libc::c_char) };
    eprint!("{}", text.to_string_lossy());
    0
}

const fn align_up_u64(value: u64, alignment: u64) -> Option<u64> {
    let mask = alignment - 1;
    match value.checked_add(mask) {
        Some(value) => Some(value & !mask),
        None => None,
    }
}

#[cfg(target_arch = "x86_64")]
fn probe_entry(
    image: &LinkedImage,
    process_image_name: &str,
    game_root: &Path,
    maximum_import_hits: usize,
    timeout_seconds: u32,
) -> Result<ProbeOutcome, LinkError> {
    #[repr(C)]
    struct EntryParameters {
        argument_count: u32,
        reserved: u32,
        argument0: u64,
        argument1: u64,
        argument2: u64,
    }
    type GuestEntry = unsafe extern "C" fn(*const EntryParameters, extern "C" fn(i32)) -> i32;

    let image_name =
        CString::new(process_image_name).map_err(|_| LinkError::InvalidProcessImageName)?;
    let parameters = EntryParameters {
        argument_count: 1,
        reserved: 0,
        argument0: image_name.as_ptr() as usize as u64,
        argument1: 0,
        argument2: 0,
    };
    let registered_fdes = register_guest_unwind_frames(image)?;
    if registered_fdes != 0 {
        eprintln!("[REMu][CXX] registered {registered_fdes} guest unwind records");
    }
    configure_probe_dispatch(image, game_root, maximum_import_hits);
    set_current_guest_thread_pointer(image.tls.thread_pointer.get());
    if timeout_seconds != 0 {
        // SAFETY: Alarm only bounds this diagnostic process; its default action terminates it.
        unsafe {
            libc::alarm(timeout_seconds);
        }
    }
    // SAFETY: The linker validated and mapped the ELF entry in executable guest memory. The process
    // entry ABI is the same System V x86-64 convention used by this Rosetta process.
    let entry: GuestEntry = unsafe { std::mem::transmute(image.memory.entry_address().get()) };
    // SAFETY: `parameters` and its image-name pointer stay live for the duration of the call.
    let return_value = unsafe { entry(&raw const parameters, guest_exit) };
    set_current_guest_thread_pointer(0);
    // SAFETY: Cancels the diagnostic alarm after a normal return.
    unsafe {
        libc::alarm(0);
    }
    let import_hits = PROBE_DISPATCH.get().map_or(0, |state| {
        state
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .hits
    });
    Ok(ProbeOutcome {
        return_value,
        import_hits,
    })
}

#[cfg(not(target_arch = "x86_64"))]
const fn probe_entry(
    _image: &LinkedImage,
    _process_image_name: &str,
    _game_root: &Path,
    _maximum_import_hits: usize,
    _timeout_seconds: u32,
) -> Result<ProbeOutcome, LinkError> {
    Err(LinkError::ProbeRequiresX86_64)
}

#[cfg(target_arch = "x86_64")]
extern "C" fn guest_exit(code: i32) {
    eprintln!("[REMu][PROBE] guest exit handler called with code {code}");
    terminate_probe(code);
}

fn terminate_probe(code: i32) -> ! {
    #[cfg(unix)]
    // SAFETY: `_exit` terminates the isolated CLI probe without unwinding across guest frames.
    unsafe {
        libc::_exit(code);
    }

    #[cfg(not(unix))]
    std::process::exit(code)
}

#[cfg(test)]
mod platform_independent_tests {
    use std::path::Path;

    use super::{
        AGC_IT_INDEX_BASE, AGC_IT_INDEX_BUFFER_SIZE, AGC_IT_NOP, AGC_IT_NUM_INSTANCES,
        AGC_R_DRAW_INDEX_AUTO, ProbeDispatchState, agc_dcb_draw_index_auto,
        agc_dcb_set_index_buffer, agc_dcb_set_num_instances, agc_pm4, guest_qsort,
        kernel_stat_payload, low_f32, low_u32, read_guest_u64, resolve_guest_path,
        sanitize_save_segment, sce_kernel_error, write_guest_u32, write_guest_u64,
    };

    unsafe extern "C" fn compare_i32(left: *const u8, right: *const u8) -> i32 {
        // SAFETY: guest_qsort passes pointers to complete i32 elements in the test array.
        let left = unsafe { std::ptr::read_unaligned(left.cast::<i32>()) };
        // SAFETY: guest_qsort passes pointers to complete i32 elements in the test array.
        let right = unsafe { std::ptr::read_unaligned(right.cast::<i32>()) };
        match left.cmp(&right) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        }
    }

    #[test]
    fn resolves_duplicate_console_path_separators() {
        assert_eq!(
            resolve_guest_path(Path::new("/games/title"), "/app0//shaders/presenter.vert"),
            Some(Path::new("/games/title/shaders/presenter.vert").to_owned())
        );
    }

    #[test]
    fn rejects_game_root_parent_traversal() {
        assert_eq!(
            resolve_guest_path(Path::new("/games/title"), "/app0/../outside.bin"),
            None
        );
    }

    #[test]
    fn builds_the_gen5_kernel_stat_layout() {
        let payload = kernel_stat_payload(1_025, false, b"/app0/data.bin");

        assert_eq!(
            u16::from_le_bytes(payload[8..10].try_into().unwrap()),
            0x81ff
        );
        assert_eq!(u16::from_le_bytes(payload[10..12].try_into().unwrap()), 1);
        assert_eq!(
            i64::from_le_bytes(payload[72..80].try_into().unwrap()),
            1_025
        );
        assert_eq!(i64::from_le_bytes(payload[80..88].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(payload[88..92].try_into().unwrap()), 512);
    }

    #[test]
    fn sign_extends_gen5_kernel_errors() {
        assert_eq!(sce_kernel_error(0x8002_0002), 0xffff_ffff_8002_0002);
    }

    #[test]
    fn decodes_float_argument_from_low_xmm_bits() {
        let expected = -1.25_f32;
        assert_eq!(
            low_f32(f64::from_bits(u64::from(expected.to_bits()))).to_bits(),
            expected.to_bits()
        );
    }

    #[test]
    fn sorts_guest_elements_with_the_guest_comparator() {
        let mut values = [7_i32, -2, 11, 0, 4];
        guest_qsort([
            u64::try_from(values.as_mut_ptr().addr()).unwrap(),
            u64::try_from(values.len()).unwrap(),
            u64::try_from(std::mem::size_of::<i32>()).unwrap(),
            u64::try_from(compare_i32 as *const () as usize).unwrap(),
            0,
            0,
        ]);
        assert_eq!(values, [-2, 0, 4, 7, 11]);
    }

    #[test]
    fn sanitizes_save_slot_path_segments() {
        assert_eq!(sanitize_save_segment("SAVEDATA00"), "SAVEDATA00");
        assert_eq!(sanitize_save_segment("../outside"), "___outside");
        assert_eq!(sanitize_save_segment("///"), "default");
    }

    #[test]
    fn builds_the_four_argument_gen5_index_buffer_packets() {
        let mut command_buffer = vec![0_u8; 0x40];
        let mut commands = vec![0_u32; 16];
        let indices = [0_u32, 1, 2, 1, 2, 3];
        let command_buffer_address =
            u64::try_from(command_buffer.as_mut_ptr().addr()).expect("command buffer address");
        let commands_address =
            u64::try_from(commands.as_mut_ptr().addr()).expect("command storage address");
        let index_buffer_address =
            u64::try_from(indices.as_ptr().addr()).expect("index buffer address");
        let index_resource_address = 0x70_0007_3030;
        write_guest_u64(command_buffer_address + 0x10, commands_address);
        write_guest_u64(command_buffer_address + 0x18, commands_address + 64);
        write_guest_u32(command_buffer_address + 0x30, 0);

        let mut state = ProbeDispatchState::default();
        let command = agc_dcb_set_index_buffer(
            [
                command_buffer_address,
                index_buffer_address,
                index_resource_address,
                16,
                0,
                0,
            ],
            &mut state,
        );

        assert_eq!(command, commands_address);
        assert_eq!(commands[0], agc_pm4(3, AGC_IT_INDEX_BASE, 0));
        assert_eq!(commands[1], low_u32(index_buffer_address));
        assert_eq!(commands[2], low_u32(index_buffer_address >> 32));
        assert_eq!(commands[3], agc_pm4(2, AGC_IT_INDEX_BUFFER_SIZE, 0));
        assert_eq!(commands[4], 16);
        assert_eq!(
            read_guest_u64(command_buffer_address + 0x10),
            commands_address + 20
        );
        assert_eq!(
            state.agc_index_resources.get(&index_buffer_address),
            Some(&index_resource_address)
        );
    }

    #[test]
    fn builds_the_gen5_instanced_auto_draw_packets() {
        let mut command_buffer = vec![0_u8; 0x40];
        let mut commands = vec![0_u32; 16];
        let command_buffer_address =
            u64::try_from(command_buffer.as_mut_ptr().addr()).expect("command buffer address");
        let commands_address =
            u64::try_from(commands.as_mut_ptr().addr()).expect("command storage address");
        write_guest_u64(command_buffer_address + 0x10, commands_address);
        write_guest_u64(command_buffer_address + 0x18, commands_address + 64);
        write_guest_u32(command_buffer_address + 0x30, 0);

        let instances = agc_dcb_set_num_instances([command_buffer_address, 3, 0, 0, 0, 0]);
        let draw = agc_dcb_draw_index_auto([command_buffer_address, 3, 0x4000_0000, 0, 0, 0]);

        assert_eq!(instances, commands_address);
        assert_eq!(commands[0], agc_pm4(2, AGC_IT_NUM_INSTANCES, 0));
        assert_eq!(commands[1], 3);
        assert_eq!(draw, commands_address + 8);
        assert_eq!(commands[2], agc_pm4(7, AGC_IT_NOP, AGC_R_DRAW_INDEX_AUTO));
        assert_eq!(commands[3], 3);
        assert_eq!(&commands[4..9], &[0; 5]);
        assert_eq!(
            read_guest_u64(command_buffer_address + 0x10),
            commands_address + 36
        );
    }
}

#[cfg(all(test, target_os = "macos", target_arch = "x86_64"))]
mod tests {
    use std::io::Cursor;

    use remu_core::GuestAddress;
    use remu_loader::load_reader;

    use super::{LinkedImage, set_current_guest_thread_pointer};

    type TlsHelper = unsafe extern "C" fn() -> u64;

    #[test]
    fn links_relative_and_import_relocations_into_fixed_memory() {
        let mut bytes = vec![0_u8; 0x400];
        write_elf_header(&mut bytes);
        write_program_header(&mut bytes, 0, 1, 7, 0x100, 0x4000, 0x200, 0x300);
        write_program_header(&mut bytes, 1, 2, 6, 0x100, 0x4000, 0x70, 0x70);
        write_program_header(&mut bytes, 2, 7, 4, 0x260, 0x4160, 2, 0x20);
        bytes[0xe0..0xe8].copy_from_slice(&0x10_u64.to_le_bytes());
        write_dynamic(&mut bytes[0x100..]);
        bytes[0x180..0x18d].copy_from_slice(b"\0testNid#A#A\0");
        bytes[0x1d8..0x1dc].copy_from_slice(&1_u32.to_le_bytes());
        bytes[0x1dc] = 0x12;
        write_relocation(&mut bytes[0x1f0..], 0, 0x4180, 8, 0x1234);
        write_relocation(&mut bytes[0x1f0..], 1, 0x4188, (1_u64 << 32) | 7, 0);
        bytes[0x240..0x249].copy_from_slice(&[0x64, 0x48, 0x8b, 0x04, 0x25, 0, 0, 0, 0]);
        bytes[0x260..0x262].copy_from_slice(&[0xaa, 0xbb]);
        let executable = load_reader(&mut Cursor::new(bytes)).unwrap();

        let linked = LinkedImage::link_main(&executable).unwrap();
        assert_eq!(linked.relocation_stats.relative, 1);
        assert_eq!(linked.relocation_stats.jump_slots, 1);
        assert_eq!(linked.import_stubs.len(), 1);
        assert_eq!(linked.tls.static_block_size, 0x20);
        assert_eq!(linked.tls.patched_loads, 1);
        let base = linked.memory.image_base().get();
        let mut relative = [0_u8; 8];
        linked
            .memory
            .read(GuestAddress::new(base + 0x4180), &mut relative)
            .unwrap();
        assert_eq!(u64::from_le_bytes(relative), base + 0x1234);
        let mut import = [0_u8; 8];
        linked
            .memory
            .read(GuestAddress::new(base + 0x4188), &mut import)
            .unwrap();
        assert_eq!(
            u64::from_le_bytes(import),
            linked.import_stubs[0].address.get()
        );
        let mut patched_tls_load = [0_u8; 9];
        linked
            .memory
            .read(GuestAddress::new(base + 0x4140), &mut patched_tls_load)
            .unwrap();
        assert_eq!(patched_tls_load[0], 0xe8);
        assert_eq!(&patched_tls_load[5..], &[0x90; 4]);
        let displacement = i32::from_le_bytes(
            patched_tls_load[1..5]
                .try_into()
                .expect("fixed-width call displacement"),
        );
        let helper_address = (base + 0x4145).wrapping_add_signed(i64::from(displacement));
        set_current_guest_thread_pointer(0x1234_5678_9abc_def0);
        // SAFETY: The linker emitted and finalized the tested helper as executable x86-64 code.
        let helper: TlsHelper = unsafe { std::mem::transmute(helper_address) };
        // SAFETY: The helper takes no arguments and returns the configured pointer in RAX.
        assert_eq!(unsafe { helper() }, 0x1234_5678_9abc_def0);
        set_current_guest_thread_pointer(0);
        let mut tls_template = [0_u8; 2];
        linked
            .memory
            .read(
                GuestAddress::new(linked.tls.thread_pointer.get() - 0x20),
                &mut tls_template,
            )
            .unwrap();
        assert_eq!(tls_template, [0xaa, 0xbb]);
    }

    fn write_elf_header(bytes: &mut [u8]) {
        bytes[..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[0x12..0x14].copy_from_slice(&62_u16.to_le_bytes());
        bytes[0x18..0x20].copy_from_slice(&0x4000_u64.to_le_bytes());
        bytes[0x20..0x28].copy_from_slice(&64_u64.to_le_bytes());
        bytes[0x34..0x36].copy_from_slice(&64_u16.to_le_bytes());
        bytes[0x36..0x38].copy_from_slice(&56_u16.to_le_bytes());
        bytes[0x38..0x3a].copy_from_slice(&3_u16.to_le_bytes());
    }

    #[allow(clippy::too_many_arguments)]
    fn write_program_header(
        bytes: &mut [u8],
        index: usize,
        program_type: u32,
        flags: u32,
        file_offset: u64,
        virtual_address: u64,
        file_size: u64,
        memory_size: u64,
    ) {
        let offset = 64 + index * 56;
        bytes[offset..offset + 4].copy_from_slice(&program_type.to_le_bytes());
        bytes[offset + 4..offset + 8].copy_from_slice(&flags.to_le_bytes());
        bytes[offset + 8..offset + 16].copy_from_slice(&file_offset.to_le_bytes());
        bytes[offset + 16..offset + 24].copy_from_slice(&virtual_address.to_le_bytes());
        bytes[offset + 32..offset + 40].copy_from_slice(&file_size.to_le_bytes());
        bytes[offset + 40..offset + 48].copy_from_slice(&memory_size.to_le_bytes());
        bytes[offset + 48..offset + 56].copy_from_slice(&0x4000_u64.to_le_bytes());
    }

    fn write_dynamic(bytes: &mut [u8]) {
        let entries = [
            (5_i64, 0x4080_u64),
            (10, 0x40),
            (6, 0x40c0),
            (0x6100_003f, 0x30),
            (7, 0x40f0),
            (8, 0x30),
            (0, 0),
        ];
        for (index, (tag, value)) in entries.into_iter().enumerate() {
            let offset = index * 16;
            bytes[offset..offset + 8].copy_from_slice(&tag.to_le_bytes());
            bytes[offset + 8..offset + 16].copy_from_slice(&value.to_le_bytes());
        }
    }

    fn write_relocation(bytes: &mut [u8], index: usize, offset: u64, info: u64, addend: i64) {
        let entry = index * 24;
        bytes[entry..entry + 8].copy_from_slice(&offset.to_le_bytes());
        bytes[entry + 8..entry + 16].copy_from_slice(&info.to_le_bytes());
        bytes[entry + 16..entry + 24].copy_from_slice(&addend.to_le_bytes());
    }
}
