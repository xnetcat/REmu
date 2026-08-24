// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Recovers Silent Hill's two FMallocBinned3 pointer-width faults and resumes the load.
//!
//! The TLS-cache pop dies on a 32-bit head. Silent Hill stores `0x70xxxxxxxx` blocks,
//! `0x80xxxxxxxx` blocks shifted down one byte (`0x80015f00` for `0x80015f0000`), and
//! also a slot whose head is `0x1` with count `0xffffffff`. Reconstruct an `0x8xxxxxxx`
//! candidate only when that page is already a usable FreeBlock. A resident `0x80` page
//! can still be user data, and `0x1 | 0x70` is Unreal's "unrecognized block 7000000001"
//! fatal. In those cases empty the TLS slot and resume on the existing miss path so
//! GMalloc allocates a real block.
//!
//! The next death after a 1 TiB pin is `mov edx, [r12+rbx*4]` with `r12 = 0x8076ee04800`:
//! an 8 TiB-class PoolInfo page (`0x08 << 40 | offset`). GetOrCreate initializes a zero
//! page when the load succeeds, so committing the faulting page (mprotect inside the
//! reservation, or a MAP_FIXED island when the 8 TiB pin could not be placed) lets the
//! lookup continue instead of dying as `space=host`.
//!
//! A mapped unaligned head (`0x7000000001`) does not fault. The pop returns it as malloc and
//! Unreal later fatals on realloc. After link, patch that load to reject unaligned heads and
//! resume on the existing OS-alloc tail.
//!
//! New PoolInfo slabs pre-fill unused dwords with `0x20003` (bin canary 3). Growing the
//! `0x70` heap later OS-maps a page whose slot is still that sentinel, and
//! `CheckCanary(1)` fatals. Promote only `0x20003` to `0x20001`.
//!
//! A later walk can still load `0x7100000000` / `0x7100000001` as a PoolInfo* and die on
//! `[rax+0x98]` (`pc=0x8011363c4`) or `[rax+0x178]` (`pc=0x801136a17`). That is leftover
//! packed state, not a real heap page. Skip only that table slot and keep walking — the
//! function remaps the remaining objects, and aborting at the epilogue lets RenderThread
//! stop submitting. A fabricated zero FirstFreeBlock immediately null-derefs at
//! `pc=0x80113647f`.
//!
//! GetOrCreate can also treat a leftover qword at the 8 TiB pin base (`0x7000000000` →
//! `0x1e701000200`) as a live PoolInfo page because it is non-null. `mov eax, [rbx]` at
//! `pc=0x801237ddb` then SIGBUS. Clear that directory slot and resume on the existing
//! allocate-fresh-page path — do not invent an 8 TiB-class address for the leftover.
//!
//! A later FMallocBinned3 small-block return (`0x70d461f580` / `0x7027e1dfc0`) can sit
//! inside the pin but never have been `MapDirectMemory`'d. The realloc memcpy at
//! `pc=0x80112d8ec` and the TArray sentinel store at `pc=0x802586aa4` then SIGBUS.
//! Do not `mprotect` those pin holes — that stalls RenderThread. Null the stored dest
//! and skip the faulting store. A later memcpy helper at `pc=0x8011aa0e9` (caller
//! `0x80123a57d`) does the same. Unknown store encodings are skipped by length-decode.

#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
use std::sync::atomic::AtomicU64;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
use std::sync::atomic::{AtomicU64, Ordering};

/// Counts truncated-heap recoveries that rewrote at least one register and resumed.
#[allow(dead_code)]
pub(crate) static TRUNCATED_HEAP_RECOVERED: AtomicU64 = AtomicU64::new(0);
/// Counts sparse pages committed so a Binned3 PoolInfo load can initialize the slot.
#[allow(dead_code)]
pub(crate) static SPARSE_HEAP_PAGES_COMMITTED: AtomicU64 = AtomicU64::new(0);
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
static LAST_RECOVERED: AtomicU64 = AtomicU64::new(0);
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
static LAST_SPARSE_PAGE: AtomicU64 = AtomicU64::new(0);

const UNREAL_HEAP_PREFIX: u64 = 0x0000_0070_0000_0000;
/// Distance between adjacent 0x7X allocation banks (`0x76a0…` − `0x75a0…`).
const HEAP_BANK_STRIDE: u64 = 0x0000_0001_0000_0000;
/// `0x8076ee04800 >> 40`. FMallocBinned3's shifted PoolInfo pages live in this 8 TiB class.
const EIGHT_TIB_CLASS_PREFIX: u64 = 0x8;
const POOLINFO_TABLE_SIZE: u64 = 64 * 1024;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const GUEST_PAGE_SIZE: u64 = 16 * 1024;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const MAX_SPARSE_COMMITS: u64 = 16_384;

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
use std::ffi::c_void;

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(crate) unsafe extern "C" fn deliver_or_recover_truncated_heap_pointer(
    signal: i32,
    fault_address: u64,
    context: *mut c_void,
) -> bool {
    // SAFETY: the caller is the SIGSEGV/SIGBUS reporter with a live SA_SIGINFO context.
    if unsafe { try_recover_truncated_heap_pointer(signal, fault_address, context) } {
        return true;
    }
    if unsafe { try_commit_sparse_heap_page(fault_address, context) } {
        return true;
    }
    if unsafe { try_retarget_unaligned_reservation_load(fault_address, context) } {
        return true;
    }
    if unsafe { try_skip_unmapped_canary_trampoline(fault_address, context) } {
        return true;
    }
    if unsafe { try_divert_garbage_poolinfo_directory(fault_address, context) } {
        return true;
    }
    if unsafe { try_retarget_adjacent_heap_bank(fault_address, context) } {
        return true;
    }
    if unsafe { try_skip_unmapped_realloc_memcpy(fault_address, context) } {
        return true;
    }
    if unsafe { try_skip_unmapped_memcpy_helper(fault_address, context) } {
        return true;
    }
    if unsafe { try_skip_unmapped_tarray_sentinel_store(fault_address, context) } {
        return true;
    }
    if unsafe { try_skip_unmapped_heap_store(fault_address, context) } {
        return true;
    }
    // SAFETY: same live context; this is the previously registered guest-exception path.
    unsafe { crate::guest_signal::try_deliver_guest_fault(signal, fault_address, context) }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
unsafe extern "C" fn try_recover_truncated_heap_pointer(
    _signal: i32,
    fault_address: u64,
    context: *mut c_void,
) -> bool {
    if context.is_null() {
        return false;
    }
    // SAFETY: SA_SIGINFO supplies a live Darwin ucontext for the duration of this callback.
    let Some(machine) = (unsafe { crate::tls_fault::machine_context_mut(context) }) else {
        return false;
    };
    if LAST_RECOVERED.load(Ordering::Relaxed) == fault_address {
        return false;
    }
    if is_truncated_unreal_pointer(fault_address)
        && let Some(reconstructed) = reconstruct_truncated_pointer(fault_address)
        && reconstructed != fault_address
        && rewrite_registers_to(machine, fault_address, reconstructed) != 0
    {
        rewrite_tls_slot_if_truncated(machine.rsi, fault_address, reconstructed);
        rewrite_tls_slot_if_truncated(machine.rcx, fault_address, reconstructed);
        LAST_RECOVERED.store(fault_address, Ordering::Relaxed);
        TRUNCATED_HEAP_RECOVERED.fetch_add(1, Ordering::Relaxed);
        write_pointer_diagnostic(
            b"[REMu][VM] recovered truncated heap pointer ",
            fault_address,
            reconstructed,
            machine.instruction_pointer,
        );
        return true;
    }
    // Silent Hill's TLS slot can hold `0x1` with count `0xffffffff`, not only an `0x8xxxxxxx`
    // truncation. Reconstructing `0x1 | 0x70` is the Unreal "unrecognized block 7000000001"
    // fatal. Empty the slot and send any bad pop at the known PCs to the miss path.
    if !divert_truncated_pop_to_slow_path(machine, fault_address)
        && !divert_truncated_freelist_walk(machine, fault_address)
    {
        return false;
    }
    LAST_RECOVERED.store(fault_address, Ordering::Relaxed);
    let recovered = TRUNCATED_HEAP_RECOVERED.fetch_add(1, Ordering::Relaxed);
    if recovered < 8 {
        write_pointer_diagnostic(
            b"[REMu][VM] diverted truncated heap pop to slow path ",
            fault_address,
            machine.instruction_pointer,
            machine.instruction_pointer,
        );
    }
    true
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
unsafe extern "C" fn try_commit_sparse_heap_page(fault_address: u64, context: *mut c_void) -> bool {
    if fault_address == 0 || context.is_null() {
        return false;
    }
    let page = align_down(fault_address, GUEST_PAGE_SIZE);
    if LAST_SPARSE_PAGE.load(Ordering::Relaxed) == page {
        return false;
    }
    if SPARSE_HEAP_PAGES_COMMITTED.load(Ordering::Relaxed) >= MAX_SPARSE_COMMITS {
        return false;
    }
    // SAFETY: SA_SIGINFO supplies a live Darwin ucontext for the duration of this callback.
    let Some(machine) = (unsafe { crate::tls_fault::machine_context_mut(context) }) else {
        return false;
    };
    let table_base = poolinfo_table_base(fault_address, machine.r12);
    // Only PoolInfo pages (the 8 TiB class, `0x08 << 40`) may be committed from a fault.
    // The same 8 TiB reservation also covers the `0x70` user heap. Committing those
    // pages on a stray touch lets Unreal treat them as live bin pools; the later
    // official `MapDirectMemory` then dies on `CheckCanary(1)` against a leftover 3.
    // Never `mprotect` the 8 TiB pin from this handler — that stalls the faulting thread
    // for Unreal's 120s watchdog. Punch a `MAP_FIXED` island even when the page sits
    // inside the published reservation.
    let committed = if is_eight_tib_class_pointer(fault_address)
        || is_eight_tib_class_pointer(table_base)
    {
        map_isolated_poolinfo_pages(table_base, fault_address)
    } else {
        0
    };
    if committed == 0 {
        return false;
    }
    LAST_SPARSE_PAGE.store(page, Ordering::Relaxed);
    SPARSE_HEAP_PAGES_COMMITTED.fetch_add(committed, Ordering::Relaxed);
    write_pointer_diagnostic(
        b"[REMu][VM] committed sparse heap page ",
        fault_address,
        table_base,
        machine.instruction_pointer,
    );
    true
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
unsafe extern "C" fn try_retarget_unaligned_reservation_load(
    fault_address: u64,
    context: *mut c_void,
) -> bool {
    if context.is_null() {
        return false;
    }
    // SAFETY: SA_SIGINFO supplies a live Darwin ucontext for the duration of this callback.
    let Some(machine) = (unsafe { crate::tls_fault::machine_context_mut(context) }) else {
        return false;
    };
    let registers = [
        machine.rax,
        machine.rcx,
        machine.rdx,
        machine.rbx,
        machine.rsi,
        machine.rdi,
        machine.r8,
        machine.r9,
        machine.r10,
        machine.r11,
        machine.r12,
        machine.r13,
        machine.r14,
        machine.r15,
    ];
    let Some(base) = registers.into_iter().find(|&register| {
        poison_poolinfo_load_base(fault_address, register)
            && remu_memory::guest_reservation_contains(register)
    }) else {
        return false;
    };
    let resume = if machine.instruction_pointer == BINNED3_HASH_WALK_POOLINFO_LOAD {
        let limit_address = machine.rbp.wrapping_add_signed(BINNED3_HASH_WALK_LIMIT_SLOT);
        let limit = if page_is_resident(limit_address) {
            read_guest_u64(limit_address)
        } else {
            0
        };
        let (next_index, resume) = hash_walk_skip_slot(machine.r12, limit);
        machine.r12 = next_index;
        resume
    } else if machine.instruction_pointer == BINNED3_OBJECT_WALK_POOLINFO_LOAD {
        let (next_index, resume) = object_walk_skip_slot(machine.r13, machine.rbx);
        machine.r13 = next_index;
        resume
    } else {
        return false;
    };
    machine.instruction_pointer = resume;
    LAST_RECOVERED.store(fault_address, Ordering::Relaxed);
    TRUNCATED_HEAP_RECOVERED.fetch_add(1, Ordering::Relaxed);
    write_pointer_diagnostic(
        b"[REMu][VM] skipped poison PoolInfo slot ",
        fault_address,
        base,
        resume,
    );
    true
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
unsafe extern "C" fn try_skip_unmapped_canary_trampoline(
    fault_address: u64,
    context: *mut c_void,
) -> bool {
    if context.is_null() {
        return false;
    }
    // SAFETY: SA_SIGINFO supplies a live Darwin ucontext for the duration of this callback.
    let Some(machine) = (unsafe { crate::tls_fault::machine_context_mut(context) }) else {
        return false;
    };
    let pc = machine.instruction_pointer;
    let dummy = BINNED3_CANARY_DUMMY_TABLE;
    let canary2 = (BINNED3_CANARY2_TRAMPOLINE..BINNED3_CANARY2_TRAMPOLINE + 0x60).contains(&pc)
        || pc == BINNED3_CANARY2_PACKED_LOAD
        || pc == BINNED3_CANARY2_PACKED_RELOAD
        || pc == BINNED3_CANARY2_LOAD;
    let canary1 = (BINNED3_CANARY1_TRAMPOLINE..BINNED3_CANARY1_TRAMPOLINE + 0x40).contains(&pc)
        || pc == BINNED3_CANARY1_LOAD;
    if !canary2 && !canary1 {
        return false;
    }
    if page_is_resident(fault_address) && is_eight_tib_class_pointer(fault_address) {
        return false;
    }
    if canary2 {
        machine.r12 = dummy;
        machine.rdx = 0x0002_0001;
        machine.rcx = 1;
        // The OK path reloads `[r12+rbx*4]` twice. Changing only r12 and retrying the
        // trampoline load storms SIGBUS when the next FreeBlock arrives with the same
        // leftover table. Skip the faulting load so one recover finishes the check.
        if pc == BINNED3_CANARY2_PACKED_LOAD {
            machine.rax = 0x0002_0001;
            machine.instruction_pointer = BINNED3_CANARY2_AFTER_PACKED_LOAD;
        } else if pc == BINNED3_CANARY2_PACKED_RELOAD {
            machine.rax = 0x0002_0001;
            machine.instruction_pointer = BINNED3_CANARY2_AFTER_PACKED_RELOAD;
        } else {
            machine.instruction_pointer = BINNED3_CANARY2_OK;
        }
    } else {
        machine.rax = dummy;
    }
    LAST_RECOVERED.store(fault_address, Ordering::Relaxed);
    TRUNCATED_HEAP_RECOVERED.fetch_add(1, Ordering::Relaxed);
    write_pointer_diagnostic(
        b"[REMu][VM] retargeted unmapped Binned3 canary table ",
        fault_address,
        dummy,
        pc,
    );
    true
}

/// GetOrCreate loaded a non-null leftover qword as a PoolInfo page and died on `[rbx]`.
///
/// PPSA10112 keeps the first pool's page directory at `[allocator+0x78]`. After the 8 TiB
/// pin that slot can hold `0x1e701000200` (packed bytes at `0x7000000000`) instead of
/// NULL. The null check then skips the 64 KiB alloc, and `lea rbx, [rcx+rdx*4]` /
/// `mov eax, [rbx]` SIGBUS. Both the first-use load (`0x801237ddb`) and the grow-count
/// load (`0x801237e0f`) use that rbx. Zero the directory entry and resume at the
/// existing `test rcx` fallthrough so the guest allocates a real page.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
unsafe extern "C" fn try_divert_garbage_poolinfo_directory(
    fault_address: u64,
    context: *mut c_void,
) -> bool {
    if context.is_null() {
        return false;
    }
    // SAFETY: SA_SIGINFO supplies a live Darwin ucontext for the duration of this callback.
    let Some(machine) = (unsafe { crate::tls_fault::machine_context_mut(context) }) else {
        return false;
    };
    if !is_getorcreate_slot_load_pc(machine.instruction_pointer) {
        return false;
    }
    let page = machine.rcx;
    if !looks_like_garbage_poolinfo_page(page) || page_is_resident(page) {
        return false;
    }
    if fault_address < page || fault_address.saturating_sub(page) >= POOLINFO_TABLE_SIZE {
        return false;
    }
    let directory = machine.r8.wrapping_add(machine.r15.wrapping_mul(8));
    if page_is_resident(directory) && read_guest_u64(directory) == page {
        write_guest_u64(directory, 0);
    }
    machine.rcx = 0;
    machine.rbx = 0;
    machine.instruction_pointer = BINNED3_GETORCREATE_ALLOC;
    LAST_RECOVERED.store(fault_address, Ordering::Relaxed);
    TRUNCATED_HEAP_RECOVERED.fetch_add(1, Ordering::Relaxed);
    write_pointer_diagnostic(
        b"[REMu][VM] diverted garbage PoolInfo directory to alloc ",
        fault_address,
        page,
        BINNED3_GETORCREATE_ALLOC,
    );
    true
}

/// Silent Hill's OS-backed bins sometimes store `0x76a0xxxx` while the matching
/// `MapDirectMemory` landed at `0x75a0xxxx` (bit 32 of the 0x7X prefix). Host copies then
/// SIGBUS on the unmapped sibling. If the 4 GiB neighbour is already resident, retarget
/// every GPR that still holds the dead page.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
unsafe extern "C" fn try_retarget_adjacent_heap_bank(
    fault_address: u64,
    context: *mut c_void,
) -> bool {
    if context.is_null() || fault_address == 0 {
        return false;
    }
    let bank = fault_address >> 32;
    if !(0x70..=0x7f).contains(&bank) {
        return false;
    }
    if page_is_resident(fault_address) {
        return false;
    }
    let lower = fault_address.wrapping_sub(HEAP_BANK_STRIDE);
    let upper = fault_address.wrapping_add(HEAP_BANK_STRIDE);
    let sibling = if (0x70..=0x7f).contains(&(lower >> 32)) && page_is_resident(lower) {
        lower
    } else if (0x70..=0x7f).contains(&(upper >> 32)) && page_is_resident(upper) {
        upper
    } else {
        return false;
    };
    // SAFETY: SA_SIGINFO supplies a live Darwin ucontext for the duration of this callback.
    let Some(machine) = (unsafe { crate::tls_fault::machine_context_mut(context) }) else {
        return false;
    };
    let page = align_down(fault_address, GUEST_PAGE_SIZE);
    let sibling_page = align_down(sibling, GUEST_PAGE_SIZE);
    let rewritten = rewrite_registers_to(machine, page, sibling_page)
        + rewrite_registers_to(machine, fault_address, sibling);
    if rewritten == 0 {
        return false;
    }
    LAST_RECOVERED.store(fault_address, Ordering::Relaxed);
    TRUNCATED_HEAP_RECOVERED.fetch_add(1, Ordering::Relaxed);
    write_pointer_diagnostic(
        b"[REMu][VM] retargeted adjacent 0x7X heap bank ",
        fault_address,
        sibling,
        machine.instruction_pointer,
    );
    true
}

/// Realloc stored an unmapped Binned3 block at `object+0x48` and died copying into it.
///
/// `mprotect` of that pin hole from this handler stalls RenderThread (`Q W0`, 120s
/// watchdog). Clear the stored dest and resume after the copy so the object keeps a
/// null buffer instead of a poisoned pointer.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
unsafe extern "C" fn try_skip_unmapped_realloc_memcpy(
    fault_address: u64,
    context: *mut c_void,
) -> bool {
    if context.is_null() || fault_address == 0 {
        return false;
    }
    // SAFETY: SA_SIGINFO supplies a live Darwin ucontext for the duration of this callback.
    let Some(machine) = (unsafe { crate::tls_fault::machine_context_mut(context) }) else {
        return false;
    };
    if !is_committable_heap_store(fault_address) || page_is_resident(fault_address) {
        return false;
    }
    let dest_in_regs = machine.r13 == fault_address || machine.rdi == fault_address;
    if !dest_in_regs && !is_binned3_memcpy_store_pc(machine.instruction_pointer) {
        return false;
    }
    let Some(object) = realloc_object_holding_dest(machine, fault_address) else {
        return false;
    };
    write_guest_u64(object.wrapping_add(BINNED3_REALLOC_BUFFER_SLOT), 0);
    machine.r14 = object;
    machine.r13 = 0;
    machine.rax = 0;
    machine.instruction_pointer = BINNED3_MEMCPY_AFTER_COPY;
    LAST_RECOVERED.store(fault_address, Ordering::Relaxed);
    TRUNCATED_HEAP_RECOVERED.fetch_add(1, Ordering::Relaxed);
    write_pointer_diagnostic(
        b"[REMu][VM] skipped unmapped realloc memcpy dest ",
        fault_address,
        object,
        BINNED3_MEMCPY_AFTER_COPY,
    );
    true
}

/// The hang at `20260823T124624Z` was `call` `0x8011aa0e9` from `0x80123a57d` into an
/// unmapped 0x7X dest. That helper is a memcpy-class store; returning to the caller
/// and nulling the stored dest avoids `mprotect` of the pin.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
unsafe extern "C" fn try_skip_unmapped_memcpy_helper(
    fault_address: u64,
    context: *mut c_void,
) -> bool {
    if context.is_null() || fault_address == 0 {
        return false;
    }
    // SAFETY: SA_SIGINFO supplies a live Darwin ucontext for the duration of this callback.
    let Some(machine) = (unsafe { crate::tls_fault::machine_context_mut(context) }) else {
        return false;
    };
    if !is_committable_heap_store(fault_address) || page_is_resident(fault_address) {
        return false;
    }
    if machine.rdi != fault_address && machine.r13 != fault_address {
        return false;
    }
    if !is_binned3_memcpy_helper_pc(machine.instruction_pointer) {
        return false;
    }
    if !page_is_resident(machine.rsp) {
        return false;
    }
    let return_address = read_guest_u64(machine.rsp);
    if !is_guest_image_return(return_address) {
        return false;
    }
    if let Some(object) = realloc_object_holding_dest(machine, fault_address) {
        write_guest_u64(object.wrapping_add(BINNED3_REALLOC_BUFFER_SLOT), 0);
        machine.r14 = object;
    }
    machine.r13 = 0;
    machine.rdi = 0;
    machine.rax = 0;
    machine.rsp = machine.rsp.wrapping_add(8);
    machine.instruction_pointer = return_address;
    LAST_RECOVERED.store(fault_address, Ordering::Relaxed);
    TRUNCATED_HEAP_RECOVERED.fetch_add(1, Ordering::Relaxed);
    write_pointer_diagnostic(
        b"[REMu][VM] skipped unmapped memcpy helper dest ",
        fault_address,
        return_address,
        return_address,
    );
    true
}

/// TArray grow stored an unmapped Binned3 block at `[r14]` and died writing `0x80000000`.
///
/// `mov dword ptr [rax+r15*4], 80000000h` at `pc=0x802586aa4` with `rax = 0x704c3d6200`.
/// Null the data pointer and skip the sentinel store so the array stays empty.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
unsafe extern "C" fn try_skip_unmapped_tarray_sentinel_store(
    fault_address: u64,
    context: *mut c_void,
) -> bool {
    if context.is_null() || fault_address == 0 {
        return false;
    }
    // SAFETY: SA_SIGINFO supplies a live Darwin ucontext for the duration of this callback.
    let Some(machine) = (unsafe { crate::tls_fault::machine_context_mut(context) }) else {
        return false;
    };
    if machine.instruction_pointer != BINNED3_TARRAY_SENTINEL_STORE {
        return false;
    }
    if !is_committable_heap_store(fault_address) || page_is_resident(fault_address) {
        return false;
    }
    if machine.rax != fault_address || !page_is_resident(machine.r14) {
        return false;
    }
    if read_guest_u64(machine.r14) != machine.rax {
        return false;
    }
    write_guest_u64(machine.r14, 0);
    machine.rax = 0;
    machine.instruction_pointer = BINNED3_TARRAY_SENTINEL_NEXT;
    LAST_RECOVERED.store(fault_address, Ordering::Relaxed);
    TRUNCATED_HEAP_RECOVERED.fetch_add(1, Ordering::Relaxed);
    write_pointer_diagnostic(
        b"[REMu][VM] skipped unmapped TArray sentinel store ",
        fault_address,
        machine.r14,
        BINNED3_TARRAY_SENTINEL_NEXT,
    );
    true
}

/// Skips an unknown store into an unmapped 0x7X pin hole.
///
/// Specific memcpy / TArray recoveries run first. This is the remaining store PCs
/// Binned3 invents after those — never `mprotect` the pin.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
unsafe extern "C" fn try_skip_unmapped_heap_store(
    fault_address: u64,
    context: *mut c_void,
) -> bool {
    if context.is_null() || fault_address == 0 {
        return false;
    }
    // SAFETY: SA_SIGINFO supplies a live Darwin ucontext for the duration of this callback.
    let Some(machine) = (unsafe { crate::tls_fault::machine_context_mut(context) }) else {
        return false;
    };
    if !is_committable_heap_store(fault_address) || page_is_resident(fault_address) {
        return false;
    }
    if machine.instruction_pointer == 0 || !page_is_resident(machine.instruction_pointer) {
        return false;
    }
    let mut insn = [0_u8; 15];
    let Ok(base) = usize::try_from(machine.instruction_pointer) else {
        return false;
    };
    for (index, slot) in insn.iter_mut().enumerate() {
        let Some(pointer) = base.checked_add(index) else {
            return false;
        };
        // SAFETY: RIP is guest code in the identity-mapped image.
        *slot = unsafe { std::ptr::read_volatile(pointer as *const u8) };
    }
    let Some(length) = unmapped_heap_store_length(&insn) else {
        return false;
    };
    if let Some(object) = realloc_object_holding_dest(machine, fault_address) {
        write_guest_u64(object.wrapping_add(BINNED3_REALLOC_BUFFER_SLOT), 0);
        machine.r14 = object;
    } else if page_is_resident(machine.r14) && read_guest_u64(machine.r14) == fault_address {
        write_guest_u64(machine.r14, 0);
    }
    if machine.rax == fault_address {
        machine.rax = 0;
    }
    if machine.rdi == fault_address {
        machine.rdi = 0;
    }
    if machine.r13 == fault_address {
        machine.r13 = 0;
    }
    machine.instruction_pointer = machine.instruction_pointer.wrapping_add(length);
    LAST_RECOVERED.store(fault_address, Ordering::Relaxed);
    TRUNCATED_HEAP_RECOVERED.fetch_add(1, Ordering::Relaxed);
    write_pointer_diagnostic(
        b"[REMu][VM] skipped unmapped heap store ",
        fault_address,
        length,
        machine.instruction_pointer,
    );
    true
}

#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
#[allow(dead_code)]
pub(crate) unsafe extern "C" fn try_recover_truncated_heap_pointer(
    _signal: i32,
    _fault_address: u64,
    _context: *mut std::ffi::c_void,
) -> bool {
    false
}

fn is_truncated_unreal_pointer(address: u64) -> bool {
    address != 0 && address < (1 << 32) && address & 0x8000_0000 != 0
}

/// Rebuilds a 32-bit TLS-cache head.
///
/// The same 32-bit value is either a `0x70xxxxxxxx` block (`0x80015f00` → `0x7080015f00`) or a
/// `0x80xxxxxxxx` page stored down one byte (`0x80015f00` → `0x80015f0000`). The 0x80 page is
/// mapped in Silent Hill but holds packed integers, not a FreeBlock — using it writes garbage
/// back into the TLS slot. Accept a resident candidate only when its first qword looks like a
/// heap pointer. Otherwise return `None` so the fault handler empties the TLS slot and
/// resumes on the miss path.
fn reconstruct_truncated_pointer(fault_address: u64) -> Option<u64> {
    if !is_truncated_unreal_pointer(fault_address) {
        return None;
    }
    let shifted = fault_address << 8;
    let prefixed = fault_address | UNREAL_HEAP_PREFIX;
    if is_usable_freelist_block(shifted) {
        return Some(shifted);
    }
    if is_usable_freelist_block(prefixed) {
        return Some(prefixed);
    }
    None
}

/// True when `register` is a leftover `0x71xxxxxxx` PoolInfo* whose displacement faulted.
///
/// Silent Hill has walked `0x7100000001`, `0x7100000000`, and `0x7100808080` at
/// `pc=0x8011363c4` (`[rax+0x98]`). Those addresses sit in the 8 TiB pin but were never
/// `MapDirectMemory`'d. Committing them would recreate the virgin-canary fatal.
fn poison_poolinfo_load_base(fault_address: u64, register: u64) -> bool {
    if register == 0
        || fault_address < register
        || fault_address.saturating_sub(register) >= 0x200
    {
        return false;
    }
    (register & 0xf) != 0 || (register >> 32) == 0x71
}

/// Advances the hash-walk table index past a poison PoolInfo* (`add r12, 8`).
///
/// `[rbp-0x80]` holds `count << 3`. When the slot was the last entry, resume at the
/// outer-object tail; otherwise retry `table[r12]`.
fn hash_walk_skip_slot(r12: u64, limit: u64) -> (u64, u64) {
    let next = r12.wrapping_add(8);
    if limit == 0 || next >= limit {
        (next, BINNED3_HASH_WALK_INNER_DONE)
    } else {
        (next, BINNED3_HASH_WALK_INNER_NEXT)
    }
}

/// Advances the later object walk past a poison PoolInfo* (`add r13, 8`).
///
/// `rbx` is the byte length (`count << 3`). The body is `mov esi, [rax+0x178]` at
/// `pc=0x801136a17`; skipping the call keeps the remap walking the remaining slots.
fn object_walk_skip_slot(r13: u64, limit: u64) -> (u64, u64) {
    let next = r13.wrapping_add(8);
    if limit == 0 || next >= limit {
        (next, BINNED3_OBJECT_WALK_DONE)
    } else {
        (next, BINNED3_OBJECT_WALK_NEXT)
    }
}

fn looks_like_heap_pointer(value: u64) -> bool {
    let high = value >> 32;
    value == 0
        || high == 0x70
        || high == 0x80
        || (value >> 40) == EIGHT_TIB_CLASS_PREFIX
}

/// Leftover packed qword used as a PoolInfo page — not an 8 TiB-class table and not a 0x7X bank.
fn looks_like_garbage_poolinfo_page(page: u64) -> bool {
    if page == 0 || is_eight_tib_class_pointer(page) {
        return false;
    }
    !((0x70..=0x7f).contains(&(page >> 32)))
}

fn is_getorcreate_slot_load_pc(pc: u64) -> bool {
    pc == BINNED3_GETORCREATE_SLOT_LOAD || pc == BINNED3_GETORCREATE_SLOT_GROW
}

fn is_binned3_memcpy_store_pc(pc: u64) -> bool {
    (BINNED3_MEMCPY_STORE_START..BINNED3_MEMCPY_STORE_END).contains(&pc)
}

fn is_binned3_memcpy_helper_pc(pc: u64) -> bool {
    pc == BINNED3_MEMCPY_HELPER
        || (BINNED3_MEMCPY_HELPER_START..BINNED3_MEMCPY_HELPER_END).contains(&pc)
}

fn is_guest_image_return(address: u64) -> bool {
    (GUEST_IMAGE_RETURN_START..GUEST_IMAGE_RETURN_END).contains(&address)
}

fn unmapped_heap_store_length(bytes: &[u8]) -> Option<u64> {
    if bytes.is_empty() {
        return None;
    }
    let mut decoder =
        iced_x86::Decoder::with_ip(64, bytes, 0, iced_x86::DecoderOptions::NONE);
    let instruction = decoder.decode();
    if instruction.is_invalid() || instruction.len() == 0 {
        return None;
    }
    if instruction.op_count() == 0
        || instruction.op0_kind() != iced_x86::OpKind::Memory
        || !matches!(
            instruction.mnemonic(),
            iced_x86::Mnemonic::Mov
                | iced_x86::Mnemonic::Movd
                | iced_x86::Mnemonic::Movq
                | iced_x86::Mnemonic::Movaps
                | iced_x86::Mnemonic::Movups
                | iced_x86::Mnemonic::Movdqa
                | iced_x86::Mnemonic::Movdqu
                | iced_x86::Mnemonic::Movntps
                | iced_x86::Mnemonic::Movnti
                | iced_x86::Mnemonic::Vmovaps
                | iced_x86::Mnemonic::Vmovups
                | iced_x86::Mnemonic::Vmovdqa
                | iced_x86::Mnemonic::Vmovdqu
        )
    {
        return None;
    }
    u64::try_from(instruction.len()).ok()
}

/// Live 0x7X malloc dest. `0x71` leftover PoolInfo pages must stay uncommitted.
fn is_committable_heap_store(address: u64) -> bool {
    let bank = address >> 32;
    (0x70..=0x7f).contains(&bank) && bank != 0x71
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn realloc_object_holding_dest(
    machine: &crate::tls_fault::MachineContext,
    dest: u64,
) -> Option<u64> {
    let saved = machine.rbp.wrapping_add_signed(BINNED3_REALLOC_OBJECT_SLOT);
    let candidates = [
        machine.r14,
        if page_is_resident(saved) {
            read_guest_u64(saved)
        } else {
            0
        },
    ];
    candidates.into_iter().find(|&object| {
        object != 0
            && page_is_resident(object.wrapping_add(BINNED3_REALLOC_BUFFER_SLOT))
            && read_guest_u64(object.wrapping_add(BINNED3_REALLOC_BUFFER_SLOT)) == dest
    })
}

fn is_usable_freelist_block(address: u64) -> bool {
    address.is_multiple_of(16)
        && page_is_resident(address)
        && looks_like_heap_pointer(read_guest_u64(address))
}

fn read_guest_u64(address: u64) -> u64 {
    if address == 0 || !address.is_multiple_of(8) {
        return 0;
    }
    // SAFETY: the caller checked the page is resident in the identity-mapped guest space.
    unsafe { std::ptr::read_volatile(address as usize as *const u64) }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn write_guest_u64(address: u64, value: u64) {
    if address == 0 || !address.is_multiple_of(8) {
        return;
    }
    // SAFETY: the caller checked the page is resident in the identity-mapped guest space.
    unsafe { std::ptr::write_volatile(address as usize as *mut u64, value) }
}

fn is_eight_tib_class_pointer(address: u64) -> bool {
    address != 0 && (address >> 40) == EIGHT_TIB_CLASS_PREFIX
}

fn poolinfo_table_base(fault_address: u64, r12: u64) -> u64 {
    if r12 != 0 && fault_address >= r12 && fault_address.saturating_sub(r12) < POOLINFO_TABLE_SIZE {
        r12
    } else {
        fault_address
    }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn align_down(address: u64, alignment: u64) -> u64 {
    address & !(alignment - 1)
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn page_is_resident(address: u64) -> bool {
    if address == 0 {
        return false;
    }
    let page = align_down(address, GUEST_PAGE_SIZE);
    let Ok(length) = usize::try_from(GUEST_PAGE_SIZE) else {
        return false;
    };
    let mut status = 0_i8;
    // SAFETY: `mincore` only inspects the page that contains `address`. It is async-signal-safe
    // and does not take a lock.
    unsafe {
        libc::mincore(page as usize as *mut c_void, length, &raw mut status) == 0 && status & 1 != 0
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
fn page_is_resident(_address: u64) -> bool {
    false
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn map_isolated_poolinfo_pages(table_base: u64, fault_address: u64) -> u64 {
    let start = align_down(table_base.min(fault_address), GUEST_PAGE_SIZE);
    let end = match table_base
        .max(fault_address)
        .checked_add(POOLINFO_TABLE_SIZE)
        .map(|limit| align_down(limit + GUEST_PAGE_SIZE - 1, GUEST_PAGE_SIZE))
    {
        Some(end) if end > start => end,
        _ => return 0,
    };
    let Some(span) = end.checked_sub(start) else {
        return 0;
    };
    if span == 0 || span > POOLINFO_TABLE_SIZE + GUEST_PAGE_SIZE {
        return 0;
    }
    let Ok(length) = usize::try_from(span) else {
        return 0;
    };
    // SAFETY: anonymous MAP_FIXED island for an 8 TiB-class PoolInfo page. Prefer this over
    // `mprotect` of the 8 TiB pin — one-page protection of that VMA can stall the handler.
    let mapped = unsafe {
        libc::mmap(
            start as usize as *mut c_void,
            length,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED,
            -1,
            0,
        )
    };
    if mapped == libc::MAP_FAILED || mapped as usize as u64 != start {
        return 0;
    }
    remu_memory::publish_guest_committed_range(start, span);
    span / GUEST_PAGE_SIZE
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
/// PPSA10112 FMallocBinned3 TLS-cache pop. `test rcx` already passed; this is the OS-alloc tail.
const BINNED3_POP_SLOW_PATH: u64 = 0x0000_0008_0124_101A;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
/// Overflow/realloc pop: `test rax, rax` / `je empty` after promoting the overflow head.
const BINNED3_OVERFLOW_POP_TEST: u64 = 0x0000_0008_0124_117B;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_POP_LOAD: u64 = 0x0000_0008_0124_1006;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_POP_LOAD_BYTES: [u8; 6] = [0x48, 0x8b, 0x01, 0x48, 0x89, 0x06];
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_OVERFLOW_LOAD: u64 = 0x0000_0008_0124_1189;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_OVERFLOW_LOAD_BYTES: [u8; 6] = [0x48, 0x8b, 0x10, 0x48, 0x89, 0x11];
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_OVERFLOW_CONTINUE: u64 = 0x0000_0008_0124_118F;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_POP_CAVE: u64 = 0x0000_0008_0124_1032;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_POP_RETURN_BLOCK: u64 = 0x0000_0008_0124_100C;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_TRAMPOLINE_PAGE: u64 = 0x0000_0007_FF00_0000;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_OVERFLOW_TRAMPOLINE: u64 = BINNED3_TRAMPOLINE_PAGE + 64;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_CANARY1_TRAMPOLINE: u64 = BINNED3_TRAMPOLINE_PAGE + 0x80;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_CANARY2_TRAMPOLINE: u64 = BINNED3_TRAMPOLINE_PAGE + 0xC0;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
/// Dummy packed-canary table (`0x20001` dwords) for an unmapped `r12`/`rax` PoolInfo page.
const BINNED3_CANARY_DUMMY_TABLE: u64 = BINNED3_TRAMPOLINE_PAGE + 0x400;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
/// `mov r14, [rax+0x98]` inside the PPSA10112 hash-walk that loads a poison PoolInfo*.
const BINNED3_HASH_WALK_POOLINFO_LOAD: u64 = 0x0000_0008_0113_63C4;
/// Inner-loop body after `mov rax, [table+r12]`.
const BINNED3_HASH_WALK_INNER_NEXT: u64 = 0x0000_0008_0113_63AF;
/// Tail of one outer object after the PoolInfo table is exhausted.
const BINNED3_HASH_WALK_INNER_DONE: u64 = 0x0000_0008_0113_64E0;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
/// `mov esi, [rax+0x178]` in the later remap that loads a poison PoolInfo*.
const BINNED3_OBJECT_WALK_POOLINFO_LOAD: u64 = 0x0000_0008_0113_6A17;
/// Inner-loop body after `mov rax, [r14+r13]`.
const BINNED3_OBJECT_WALK_NEXT: u64 = 0x0000_0008_0113_6A10;
/// Restore saved r13/r15 after that object table is exhausted.
const BINNED3_OBJECT_WALK_DONE: u64 = 0x0000_0008_0113_6A31;
/// `[rbp-0x80]` = PoolInfo table byte length (`count << 3`).
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_HASH_WALK_LIMIT_SLOT: i64 = -0x80;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
/// Function epilogue for that walk (`edx == 0` already uses this path).
#[allow(dead_code)]
const BINNED3_HASH_WALK_EPILOGUE: u64 = 0x0000_0008_0113_680A;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
/// `mov rax, [r14]` while walking a truncated `0x80015f00` FreeBlock list.
const BINNED3_FREELIST_WALK_LOAD: u64 = 0x0000_0008_0123_A097;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
/// `mov rax, [r14+8]` predecessor of the same walk.
const BINNED3_FREELIST_WALK_NEXT: u64 = 0x0000_0008_0123_A06D;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
/// Empty-list target (`test r14` / `je`) for that walk.
const BINNED3_FREELIST_WALK_EMPTY: u64 = 0x0000_0008_0123_A288;
/// `mov eax, [rbx]` in GetOrCreate after `lea rbx, [rcx+rdx*4]`.
const BINNED3_GETORCREATE_SLOT_LOAD: u64 = 0x0000_0008_0123_7DDB;
/// Grow-count path also loads `[rbx]` after incrementing the live hash.
const BINNED3_GETORCREATE_SLOT_GROW: u64 = 0x0000_0008_0123_7E0F;
/// Fallthrough after `test rcx, rcx` / `jne used` — allocate a fresh 64 KiB PoolInfo page.
const BINNED3_GETORCREATE_ALLOC: u64 = 0x0000_0008_0123_7CE1;
/// PPSA10112 realloc memcpy stores (`vmovups [r13], …`) after a Binned3 allocate.
const BINNED3_MEMCPY_STORE_START: u64 = 0x0000_0008_0112_D800;
const BINNED3_MEMCPY_STORE_END: u64 = 0x0000_0008_0112_D960;
/// After the copy helpers (`movsxd rax, [r15+18h]`).
const BINNED3_MEMCPY_AFTER_COPY: u64 = 0x0000_0008_0112_D957;
/// Hang site `20260823T124624Z`: memcpy-class helper entered from `0x80123a57d`.
const BINNED3_MEMCPY_HELPER: u64 = 0x0000_0008_011A_A0E9;
const BINNED3_MEMCPY_HELPER_START: u64 = 0x0000_0008_011A_A000;
const BINNED3_MEMCPY_HELPER_END: u64 = 0x0000_0008_011A_C000;
const GUEST_IMAGE_RETURN_START: u64 = 0x0000_0008_0000_0000;
const GUEST_IMAGE_RETURN_END: u64 = 0x0000_0008_0C00_0000;
/// `[r14+48h]` holds the freshly allocated dest.
const BINNED3_REALLOC_BUFFER_SLOT: u64 = 0x48;
/// `[rbp-70h]` saved `this` before the allocate/copy.
const BINNED3_REALLOC_OBJECT_SLOT: i64 = -0x70;
/// `mov dword ptr [rax+r15*4], 80000000h` after TArray grow.
const BINNED3_TARRAY_SENTINEL_STORE: u64 = 0x0000_0008_0258_6AA4;
const BINNED3_TARRAY_SENTINEL_NEXT: u64 = 0x0000_0008_0258_6AAC;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_CANARY1_LOAD: u64 = 0x0000_0008_0123_7BF8;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_CANARY1_OK: u64 = 0x0000_0008_0123_7C34;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_CANARY1_FATAL: u64 = 0x0000_0008_0123_7C0A;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_CANARY1_BYTES: [u8; 18] = [
    0x8b, 0x08, 0x48, 0x89, 0xc3, 0xb8, 0x01, 0x00, 0x00, 0x00, 0x83, 0xe1, 0x03, 0x83, 0xf9, 0x01,
    0x74, 0x2a,
];
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_CANARY2_LOAD: u64 = 0x0000_0008_0123_A148;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_CANARY2_OK: u64 = 0x0000_0008_0123_A17F;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
/// `mov eax, [r12+rbx*4]` immediately after the canary2 OK store.
const BINNED3_CANARY2_PACKED_LOAD: u64 = 0x0000_0008_0123_A187;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
/// `mov ecx, eax` after the first OK-path canary reload.
const BINNED3_CANARY2_AFTER_PACKED_LOAD: u64 = 0x0000_0008_0123_A18B;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
/// Second packed-dword reload on the same OK path.
const BINNED3_CANARY2_PACKED_RELOAD: u64 = 0x0000_0008_0123_A1BF;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
/// Instruction after the second OK-path canary reload (`mov eax, [r12+rbx*4]` is 4 bytes).
const BINNED3_CANARY2_AFTER_PACKED_RELOAD: u64 = 0x0000_0008_0123_A1C3;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_CANARY2_FATAL: u64 = 0x0000_0008_0123_A156;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_CANARY2_BYTES: [u8; 14] = [
    0x41, 0x8b, 0x14, 0x9c, 0x89, 0xd1, 0x83, 0xe1, 0x03, 0x83, 0xf9, 0x01, 0x74, 0x29,
];
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_TLS_SLOT_SIZE: u64 = 32;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINNED3_TLS_SLOT_COUNT: u64 = 64;

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn rel32(from: u64, instruction_size: u64, to: u64) -> Option<i32> {
    i32::try_from(to as i64 - (from as i64 + instruction_size as i64)).ok()
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn jmp_rel32(from: u64, to: u64) -> Option<[u8; 5]> {
    let displacement = rel32(from, 5, to)?;
    let bytes = displacement.to_le_bytes();
    Some([0xe9, bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn je_rel32(from: u64, to: u64) -> Option<[u8; 6]> {
    let displacement = rel32(from, 6, to)?;
    let bytes = displacement.to_le_bytes();
    Some([0x0f, 0x84, bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn make_guest_page_rwx(address: u64) -> bool {
    let page = align_down(address, GUEST_PAGE_SIZE);
    let Ok(length) = usize::try_from(GUEST_PAGE_SIZE) else {
        return false;
    };
    unsafe {
        libc::mprotect(
            page as usize as *mut c_void,
            length,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
        ) == 0
    }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn ensure_binned3_trampoline_page() -> bool {
    let page = BINNED3_TRAMPOLINE_PAGE as usize as *mut c_void;
    let mut status = 0_i8;
    // SAFETY: mincore is async-signal-safe and only inspects one page.
    let resident = unsafe {
        libc::mincore(page, GUEST_PAGE_SIZE as usize, &raw mut status) == 0 && status & 1 != 0
    };
    if resident {
        return true;
    }
    let mapped = unsafe {
        libc::mmap(
            page,
            0x1000,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED,
            -1,
            0,
        )
    };
    if mapped == libc::MAP_FAILED || mapped as usize as u64 != BINNED3_TRAMPOLINE_PAGE {
        eprintln!(
            "[REMu][VM] cannot map Binned3 canary trampoline page: {}",
            std::io::Error::last_os_error()
        );
        return false;
    }
    true
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn restore_guest_page_rx(address: u64) {
    let page = align_down(address, GUEST_PAGE_SIZE);
    let Ok(length) = usize::try_from(GUEST_PAGE_SIZE) else {
        return;
    };
    let _ = unsafe {
        libc::mprotect(
            page as usize as *mut c_void,
            length,
            libc::PROT_READ | libc::PROT_EXEC,
        )
    };
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(crate) fn patch_silent_hill_binned3_tls_pop() {
    let load = BINNED3_POP_LOAD as usize as *mut u8;
    let overflow = BINNED3_OVERFLOW_LOAD as usize as *mut u8;
    // SAFETY: the eboot is identity-mapped. These addresses were fingerprinted from PPSA10112.
    let load_ok = unsafe { std::slice::from_raw_parts(load, BINNED3_POP_LOAD_BYTES.len()) }
        == BINNED3_POP_LOAD_BYTES;
    let overflow_ok =
        unsafe { std::slice::from_raw_parts(overflow, BINNED3_OVERFLOW_LOAD_BYTES.len()) }
            == BINNED3_OVERFLOW_LOAD_BYTES;
    if !load_ok || !overflow_ok {
        eprintln!(
            "[REMu][VM] skipped Binned3 TLS-pop patch load_ok={load_ok} overflow_ok={overflow_ok}"
        );
        patch_silent_hill_binned3_virgin_os_canary();
        return;
    }
    let trampoline_page = unsafe {
        libc::mmap(
            BINNED3_TRAMPOLINE_PAGE as usize as *mut c_void,
            0x1000,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED,
            -1,
            0,
        )
    };
    if trampoline_page == libc::MAP_FAILED
        || trampoline_page as usize as u64 != BINNED3_TRAMPOLINE_PAGE
    {
        eprintln!(
            "[REMu][VM] cannot map Binned3 trampoline page: {}",
            std::io::Error::last_os_error()
        );
        return;
    }
    let Some(primary_back) = jmp_rel32(BINNED3_TRAMPOLINE_PAGE + 20, BINNED3_POP_RETURN_BLOCK)
    else {
        return;
    };
    let Some(primary_bad) = jmp_rel32(BINNED3_TRAMPOLINE_PAGE + 39, BINNED3_POP_SLOW_PATH) else {
        return;
    };
    let Some(overflow_back) =
        jmp_rel32(BINNED3_OVERFLOW_TRAMPOLINE + 19, BINNED3_OVERFLOW_CONTINUE)
    else {
        return;
    };
    let Some(overflow_bad) =
        jmp_rel32(BINNED3_OVERFLOW_TRAMPOLINE + 40, BINNED3_OVERFLOW_POP_TEST)
    else {
        return;
    };
    let Some(hook) = jmp_rel32(BINNED3_POP_LOAD, BINNED3_TRAMPOLINE_PAGE) else {
        return;
    };
    let Some(overflow_hook) = jmp_rel32(BINNED3_OVERFLOW_LOAD, BINNED3_OVERFLOW_TRAMPOLINE) else {
        return;
    };
    // Reject 32-bit heads (`0x80015f00`, `0x1`) and unaligned 64-bit heads (`0x7000000001`).
    let primary = [
        0x49, 0x89, 0xcb, // mov r11, rcx
        0x49, 0xc1, 0xeb, 0x20, // shr r11, 32
        0x74, 0x10, // jz primary_bad
        0xf6, 0xc1, 0x0f, // test cl, 0x0f
        0x75, 0x0b, // jnz primary_bad
        0x48, 0x8b, 0x01, // mov rax, [rcx]
        0x48, 0x89, 0x06, // mov [rsi], rax
        primary_back[0],
        primary_back[1],
        primary_back[2],
        primary_back[3],
        primary_back[4],
        0x48, 0xc7, 0x06, 0x00, 0x00, 0x00, 0x00, // mov qword [rsi], 0
        0xc7, 0x46, 0x08, 0x00, 0x00, 0x00, 0x00, // mov dword [rsi+8], 0
        primary_bad[0],
        primary_bad[1],
        primary_bad[2],
        primary_bad[3],
        primary_bad[4],
    ];
    let overflow_code = [
        0x49, 0x89, 0xc3, // mov r11, rax
        0x49, 0xc1, 0xeb, 0x20, // shr r11, 32
        0x74, 0x0f, // jz overflow_bad
        0xa8, 0x0f, // test al, 0x0f
        0x75, 0x0b, // jnz overflow_bad
        0x48, 0x8b, 0x10, // mov rdx, [rax]
        0x48, 0x89, 0x11, // mov [rcx], rdx
        overflow_back[0],
        overflow_back[1],
        overflow_back[2],
        overflow_back[3],
        overflow_back[4],
        0x48, 0xc7, 0x01, 0x00, 0x00, 0x00, 0x00, // mov qword [rcx], 0
        0xc7, 0x41, 0x08, 0x00, 0x00, 0x00, 0x00, // mov dword [rcx+8], 0
        0x31, 0xc0, // xor eax, eax
        overflow_bad[0],
        overflow_bad[1],
        overflow_bad[2],
        overflow_bad[3],
        overflow_bad[4],
    ];
    let page = align_down(BINNED3_POP_LOAD, GUEST_PAGE_SIZE);
    let Ok(length) = usize::try_from(GUEST_PAGE_SIZE) else {
        return;
    };
    let protected = unsafe {
        libc::mprotect(
            page as usize as *mut c_void,
            length,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
        )
    };
    if protected != 0 {
        eprintln!(
            "[REMu][VM] cannot mprotect Binned3 TLS-pop page: {}",
            std::io::Error::last_os_error()
        );
        return;
    }
    let mut hook_bytes = [0x90_u8; 6];
    hook_bytes[..5].copy_from_slice(&hook);
    let mut overflow_hook_bytes = [0x90_u8; 6];
    overflow_hook_bytes[..5].copy_from_slice(&overflow_hook);
    // SAFETY: fingerprint matched; the eboot page is RWX and the trampoline page is ours.
    unsafe {
        std::ptr::copy_nonoverlapping(
            primary.as_ptr(),
            BINNED3_TRAMPOLINE_PAGE as usize as *mut u8,
            primary.len(),
        );
        std::ptr::copy_nonoverlapping(
            overflow_code.as_ptr(),
            BINNED3_OVERFLOW_TRAMPOLINE as usize as *mut u8,
            overflow_code.len(),
        );
        std::ptr::copy_nonoverlapping(hook_bytes.as_ptr(), load, hook_bytes.len());
        std::ptr::copy_nonoverlapping(
            overflow_hook_bytes.as_ptr(),
            overflow,
            overflow_hook_bytes.len(),
        );
    }
    let _ = unsafe {
        libc::mprotect(
            page as usize as *mut c_void,
            length,
            libc::PROT_READ | libc::PROT_EXEC,
        )
    };
    eprintln!(
        "[REMu][VM] patched Binned3 TLS pops to reject 32-bit heads primary={BINNED3_POP_LOAD:#x} overflow={BINNED3_OVERFLOW_LOAD:#x}"
    );
    patch_silent_hill_binned3_virgin_os_canary();
}

/// PPSA10112 pre-fills unused PoolInfo dwords with `0x20003` (canary 3 = bin).
///
/// A later OS-backed page then hits `CheckCanary(FirstFreeBlockIsOSAllocSize=1)` and Unreal
/// fatals `MallocBinned3 Corruption Canary was 0x3, should be 0x1`. Promote only that
/// virgin sentinel to `0x20001` so a live bin word is left alone.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn patch_silent_hill_binned3_virgin_os_canary() {
    let site1 = BINNED3_CANARY1_LOAD as usize as *mut u8;
    let site2 = BINNED3_CANARY2_LOAD as usize as *mut u8;
    let site1_ok = unsafe { std::slice::from_raw_parts(site1, BINNED3_CANARY1_BYTES.len()) }
        == BINNED3_CANARY1_BYTES;
    let site2_ok = unsafe { std::slice::from_raw_parts(site2, BINNED3_CANARY2_BYTES.len()) }
        == BINNED3_CANARY2_BYTES;
    if !site1_ok || !site2_ok {
        eprintln!(
            "[REMu][VM] skipped Binned3 virgin-canary patch site1_ok={site1_ok} site2_ok={site2_ok}"
        );
        return;
    }
    if !ensure_binned3_trampoline_page() {
        return;
    }
    let Some(site1_ok_jmp) = je_rel32(BINNED3_CANARY1_TRAMPOLINE + 0x23, BINNED3_CANARY1_OK) else {
        return;
    };
    let Some(site1_fatal_jmp) = jmp_rel32(BINNED3_CANARY1_TRAMPOLINE + 0x29, BINNED3_CANARY1_FATAL)
    else {
        return;
    };
    let Some(site2_ok_jmp) = je_rel32(BINNED3_CANARY2_TRAMPOLINE + 0x21, BINNED3_CANARY2_OK) else {
        return;
    };
    let Some(site2_fatal_jmp) = jmp_rel32(BINNED3_CANARY2_TRAMPOLINE + 0x27, BINNED3_CANARY2_FATAL)
    else {
        return;
    };
    let Some(hook1) = jmp_rel32(BINNED3_CANARY1_LOAD, BINNED3_CANARY1_TRAMPOLINE) else {
        return;
    };
    let Some(hook2) = jmp_rel32(BINNED3_CANARY2_LOAD, BINNED3_CANARY2_TRAMPOLINE) else {
        return;
    };
    let canary1 = [
        0x8b, 0x08, // mov ecx, [rax]
        0x48, 0x89, 0xc3, // mov rbx, rax
        0x81, 0xf9, 0x03, 0x00, 0x02, 0x00, // cmp ecx, 0x20003
        0x75, 0x0b, // jne skip
        0xc7, 0x00, 0x01, 0x00, 0x02, 0x00, // mov dword [rax], 0x20001
        0xb9, 0x01, 0x00, 0x02, 0x00, // mov ecx, 0x20001
        0xb8, 0x01, 0x00, 0x00, 0x00, // skip: mov eax, 1
        0x83, 0xe1, 0x03, // and ecx, 3
        0x83, 0xf9, 0x01, // cmp ecx, 1
        site1_ok_jmp[0],
        site1_ok_jmp[1],
        site1_ok_jmp[2],
        site1_ok_jmp[3],
        site1_ok_jmp[4],
        site1_ok_jmp[5],
        site1_fatal_jmp[0],
        site1_fatal_jmp[1],
        site1_fatal_jmp[2],
        site1_fatal_jmp[3],
        site1_fatal_jmp[4],
    ];
    // Only promote the virgin 0x20003 sentinel. A leftover table pointer faults here; the
    // handler retargets r12 to the dummy and resumes at the OK path so the packed reloads
    // do not SIGBUS-storm.
    let canary2 = [
        0x41, 0x8b, 0x14, 0x9c, // mov edx, [r12+rbx*4]
        0x81, 0xfa, 0x03, 0x00, 0x02, 0x00, // cmp edx, 0x20003
        0x75, 0x0d, // jne skip
        0x41, 0xc7, 0x04, 0x9c, 0x01, 0x00, 0x02, 0x00, // mov dword [r12+rbx*4], 0x20001
        0xba, 0x01, 0x00, 0x02, 0x00, // mov edx, 0x20001
        0x89, 0xd1, // skip: mov ecx, edx
        0x83, 0xe1, 0x03, // and ecx, 3
        0x83, 0xf9, 0x01, // cmp ecx, 1
        site2_ok_jmp[0],
        site2_ok_jmp[1],
        site2_ok_jmp[2],
        site2_ok_jmp[3],
        site2_ok_jmp[4],
        site2_ok_jmp[5],
        site2_fatal_jmp[0],
        site2_fatal_jmp[1],
        site2_fatal_jmp[2],
        site2_fatal_jmp[3],
        site2_fatal_jmp[4],
    ];
    if !make_guest_page_rwx(BINNED3_CANARY1_LOAD) || !make_guest_page_rwx(BINNED3_CANARY2_LOAD) {
        eprintln!(
            "[REMu][VM] cannot mprotect Binned3 canary pages: {}",
            std::io::Error::last_os_error()
        );
        return;
    }
    let mut hook1_bytes = [0x90_u8; BINNED3_CANARY1_BYTES.len()];
    hook1_bytes[..5].copy_from_slice(&hook1);
    let mut hook2_bytes = [0x90_u8; BINNED3_CANARY2_BYTES.len()];
    hook2_bytes[..5].copy_from_slice(&hook2);
    unsafe {
        std::ptr::copy_nonoverlapping(
            canary1.as_ptr(),
            BINNED3_CANARY1_TRAMPOLINE as usize as *mut u8,
            canary1.len(),
        );
        std::ptr::copy_nonoverlapping(
            canary2.as_ptr(),
            BINNED3_CANARY2_TRAMPOLINE as usize as *mut u8,
            canary2.len(),
        );
        std::ptr::copy_nonoverlapping(hook1_bytes.as_ptr(), site1, hook1_bytes.len());
        std::ptr::copy_nonoverlapping(hook2_bytes.as_ptr(), site2, hook2_bytes.len());
        let dummy = BINNED3_CANARY_DUMMY_TABLE as usize as *mut u32;
        let dwords = (0x1000 - 0x400) / 4;
        for index in 0..dwords {
            dummy.add(index).write(0x0002_0001);
        }
    }
    restore_guest_page_rx(BINNED3_CANARY1_LOAD);
    restore_guest_page_rx(BINNED3_CANARY2_LOAD);
    eprintln!(
        "[REMu][VM] patched Binned3 virgin 0x20003 PoolInfo canary to OS at {BINNED3_CANARY1_LOAD:#x} and {BINNED3_CANARY2_LOAD:#x}"
    );
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn rewrite_registers_to(
    machine: &mut crate::tls_fault::MachineContext,
    from: u64,
    to: u64,
) -> u32 {
    let mut rewritten = 0_u32;
    for register in gprs_mut(machine) {
        if *register == from {
            *register = to;
            rewritten += 1;
        }
    }
    rewritten
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn divert_truncated_freelist_walk(
    machine: &mut crate::tls_fault::MachineContext,
    truncated: u64,
) -> bool {
    let pc = machine.instruction_pointer;
    if pc != BINNED3_FREELIST_WALK_LOAD && pc != BINNED3_FREELIST_WALK_NEXT {
        return false;
    }
    if !is_truncated_unreal_pointer(truncated) {
        return false;
    }
    let head_slot = machine.rbp.wrapping_sub(0x60);
    if head_slot != 0 && head_slot.is_multiple_of(8) && page_is_resident(head_slot) {
        // SAFETY: `rbp-0x60` is the walk's list head. Clearing it matches the `je empty` path.
        unsafe {
            std::ptr::write_volatile(head_slot as *mut u64, 0);
        }
    }
    machine.r14 = 0;
    machine.instruction_pointer = BINNED3_FREELIST_WALK_EMPTY;
    true
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn divert_truncated_pop_to_slow_path(
    machine: &mut crate::tls_fault::MachineContext,
    truncated: u64,
) -> bool {
    let pc = machine.instruction_pointer;
    let is_primary_pop = (0x0000_0008_0124_1000..0x0000_0008_0124_101A).contains(&pc)
        || (BINNED3_POP_CAVE..BINNED3_POP_CAVE + 14).contains(&pc)
        || (BINNED3_TRAMPOLINE_PAGE..BINNED3_TRAMPOLINE_PAGE + 64).contains(&pc);
    let is_overflow_pop = (0x0000_0008_0124_1184..0x0000_0008_0124_1190).contains(&pc)
        || (BINNED3_OVERFLOW_TRAMPOLINE..BINNED3_OVERFLOW_TRAMPOLINE + 64).contains(&pc);
    if !is_primary_pop && !is_overflow_pop {
        return false;
    }
    if is_overflow_pop
        && !is_truncated_unreal_pointer(truncated)
        && truncated >= 0x10000
    {
        return false;
    }
    empty_tls_cache_slot(machine.rsi, truncated);
    empty_tls_cache_slot(machine.rcx, truncated);
    if is_primary_pop {
        sanitize_tls_cache(machine.rax);
        machine.rcx = 0;
        machine.instruction_pointer = BINNED3_POP_SLOW_PATH;
        return true;
    }
    sanitize_tls_cache(machine.r14);
    machine.rax = 0;
    machine.instruction_pointer = BINNED3_OVERFLOW_POP_TEST;
    true
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn sanitize_tls_cache(base: u64) {
    if base == 0 || !base.is_multiple_of(BINNED3_TLS_SLOT_SIZE) || !page_is_resident(base) {
        return;
    }
    for index in 0..BINNED3_TLS_SLOT_COUNT {
        let slot = base + index * BINNED3_TLS_SLOT_SIZE;
        if !page_is_resident(slot) {
            break;
        }
        let head = read_guest_u64(slot);
        let count = unsafe { std::ptr::read_volatile((slot + 8) as *const u32) };
        if head == 0 {
            continue;
        }
        if (head & 0xf) != 0 || count == u32::MAX || !looks_like_heap_pointer(head) {
            unsafe {
                std::ptr::write_bytes(slot as *mut u8, 0, BINNED3_TLS_SLOT_SIZE as usize);
            }
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn empty_tls_cache_slot(slot: u64, truncated: u64) {
    if slot == 0 || !slot.is_multiple_of(8) || !page_is_resident(slot) {
        return;
    }
    // SAFETY: `slot` is an 8-byte-aligned resident TLS-cache entry. Clearing head and count
    // makes the next pop take the miss path instead of chasing a truncated block.
    let pointer = slot as *mut u64;
    if unsafe { std::ptr::read_volatile(pointer) } == truncated {
        unsafe {
            std::ptr::write_bytes(slot as *mut u8, 0, BINNED3_TLS_SLOT_SIZE as usize);
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn gprs_mut(machine: &mut crate::tls_fault::MachineContext) -> [&mut u64; 16] {
    [
        &mut machine.rax,
        &mut machine.rcx,
        &mut machine.rdx,
        &mut machine.rbx,
        &mut machine.rsp,
        &mut machine.rbp,
        &mut machine.rsi,
        &mut machine.rdi,
        &mut machine.r8,
        &mut machine.r9,
        &mut machine.r10,
        &mut machine.r11,
        &mut machine.r12,
        &mut machine.r13,
        &mut machine.r14,
        &mut machine.r15,
    ]
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn rewrite_tls_slot_if_truncated(slot: u64, truncated: u64, reconstructed: u64) {
    if slot == 0 || !slot.is_multiple_of(8) || !remu_memory::guest_reservation_contains(slot) {
        return;
    }
    // SAFETY: `slot` is an 8-byte-aligned address inside the published guest reservation. The
    // Binned3 TLS cache lives there; rewriting the truncated head prevents the next pop from
    // repeating the same fault.
    let pointer = slot as *mut u64;
    if unsafe { std::ptr::read_volatile(pointer) } == truncated {
        unsafe { std::ptr::write_volatile(pointer, reconstructed) };
    }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn write_pointer_diagnostic(prefix: &[u8], fault_address: u64, other: u64, program_counter: u64) {
    let mut message = [0_u8; 196];
    let mut written = copy_bytes(&mut message, 0, prefix);
    written = write_hex(&mut message, written, fault_address);
    written = copy_bytes(&mut message, written, b" -> ");
    written = write_hex(&mut message, written, other);
    written = copy_bytes(&mut message, written, b" pc=");
    written = write_hex(&mut message, written, program_counter);
    written = copy_bytes(&mut message, written, b"\n");
    // SAFETY: `write` is async-signal-safe. Use the preserved stderr so a full async pipe
    // cannot wedge the faulting guest thread.
    unsafe {
        libc::write(
            remu_memory::crash_output_fd(),
            message.as_ptr().cast::<c_void>(),
            written,
        );
    }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn copy_bytes(buffer: &mut [u8], start: usize, bytes: &[u8]) -> usize {
    let available = buffer.len().saturating_sub(start);
    let count = bytes.len().min(available);
    buffer[start..start + count].copy_from_slice(&bytes[..count]);
    start + count
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn write_hex(buffer: &mut [u8], start: usize, value: u64) -> usize {
    let mut written = copy_bytes(buffer, start, b"0x");
    let mut started = false;
    for shift in (0..16).rev() {
        let nibble = ((value >> (shift * 4)) & 0xf) as u8;
        if !started && nibble == 0 && shift != 0 {
            continue;
        }
        started = true;
        let digit = if nibble < 10 {
            b'0' + nibble
        } else {
            b'a' + (nibble - 10)
        };
        written = copy_bytes(buffer, written, &[digit]);
    }
    written
}

#[cfg(test)]
mod tests {
    use super::{
        BINNED3_GETORCREATE_ALLOC, BINNED3_GETORCREATE_SLOT_GROW, BINNED3_GETORCREATE_SLOT_LOAD,
        BINNED3_HASH_WALK_INNER_DONE, BINNED3_HASH_WALK_INNER_NEXT, BINNED3_MEMCPY_AFTER_COPY,
        BINNED3_MEMCPY_HELPER, BINNED3_MEMCPY_STORE_END, BINNED3_MEMCPY_STORE_START,
        BINNED3_OBJECT_WALK_DONE,
        BINNED3_OBJECT_WALK_NEXT, BINNED3_REALLOC_BUFFER_SLOT, BINNED3_REALLOC_OBJECT_SLOT,
        BINNED3_TARRAY_SENTINEL_NEXT, BINNED3_TARRAY_SENTINEL_STORE,
        EIGHT_TIB_CLASS_PREFIX, HEAP_BANK_STRIDE, UNREAL_HEAP_PREFIX, hash_walk_skip_slot,
        is_binned3_memcpy_helper_pc, is_binned3_memcpy_store_pc, is_committable_heap_store,
        is_eight_tib_class_pointer, is_getorcreate_slot_load_pc, is_guest_image_return,
        is_truncated_unreal_pointer, looks_like_garbage_poolinfo_page,
        unmapped_heap_store_length,
        looks_like_heap_pointer, object_walk_skip_slot, poison_poolinfo_load_base,
        poolinfo_table_base, reconstruct_truncated_pointer,
    };

    #[test]
    fn silent_hill_freelist_head_is_a_truncated_unreal_pointer() {
        assert!(is_truncated_unreal_pointer(0x8001_5f00));
        assert_eq!(0x8001_5f00_u64 << 8, 0x0000_0080_015f_0000);
        assert_eq!(0x8001_5f00_u64 | UNREAL_HEAP_PREFIX, 0x70_8001_5f00);
        assert!(!is_truncated_unreal_pointer(0));
        assert!(!is_truncated_unreal_pointer(1));
        assert!(!is_truncated_unreal_pointer(0x7000_e9cc_e0));
        assert!(!is_truncated_unreal_pointer(0x1_0000));
        assert_eq!(1_u64 | UNREAL_HEAP_PREFIX, 0x70_0000_0001);
        assert!(looks_like_heap_pointer(0x7000_e9c7_c0));
        assert!(looks_like_heap_pointer(0x0000_0080_015f_0000));
        assert!(!looks_like_heap_pointer(0x0002_4001_0002_4001));
        // Unit tests have no guest mapping, so neither candidate is usable.
        assert_eq!(reconstruct_truncated_pointer(0x8001_5f00), None);
        assert!(!0x70_0000_0001_u64.is_multiple_of(16));
    }

    #[test]
    fn silent_hill_binned3_pop_cave_displacements_fit() {
        let trampoline = 0x0000_0007_ff00_0000_i64;
        let load = 0x0000_0008_0124_1006_i64;
        let overflow = 0x0000_0008_0124_1189_i64;
        assert!(i32::try_from(trampoline - (load + 5)).is_ok());
        assert!(i32::try_from((trampoline + 64) - (overflow + 5)).is_ok());
        assert!(i32::try_from(0x0000_0008_0124_100C - (trampoline + 20 + 5)).is_ok());
        assert!(i32::try_from(0x0000_0008_0124_101A - (trampoline + 25 + 5)).is_ok());
        let canary1 = trampoline + 0x80;
        let canary2 = trampoline + 0xC0;
        assert!(i32::try_from(canary1 - (0x0000_0008_0123_7BF8 + 5)).is_ok());
        assert!(i32::try_from(canary2 - (0x0000_0008_0123_A148 + 5)).is_ok());
        assert!(i32::try_from(0x0000_0008_0123_7C34 - (canary1 + 0x23 + 6)).is_ok());
        assert!(i32::try_from(0x0000_0008_0123_7C0A - (canary1 + 0x29 + 5)).is_ok());
        assert!(i32::try_from(0x0000_0008_0123_A17F - (canary2 + 0x21 + 6)).is_ok());
        assert!(i32::try_from(0x0000_0008_0123_A156 - (canary2 + 0x27 + 5)).is_ok());
        assert_eq!(0x0001_e701_0002_00_u64 >> 40, 1);
        assert_eq!(0x0000_0807_6ee0_4800_u64 >> 40, 8);
        assert_eq!(
            0x0076_a0fa_c028_u64.wrapping_sub(HEAP_BANK_STRIDE),
            0x0075_a0fa_c028
        );
        assert_eq!(
            0x0076_a0fa_c000_u64.wrapping_sub(HEAP_BANK_STRIDE),
            0x0075_a0fa_c000
        );
        assert_eq!(
            0x0000_0008_0113_680A_u64 - 0x0000_0008_0113_63C4,
            0x446
        );
        assert_eq!(
            0x0000_0008_0123_A288_u64 - 0x0000_0008_0123_A097,
            0x1F1
        );
        assert!(canary2 + 0x40 <= trampoline + 0x400);
        assert_eq!(trampoline + 0x400, 0x0000_0007_ff00_0400);
    }

    #[test]
    fn silent_hill_unaligned_poolinfo_plus_98_is_a_poison_load() {
        assert!(poison_poolinfo_load_base(0x7100_0000_99, 0x7100_0000_01));
        assert!(poison_poolinfo_load_base(0x7100_0000_98, 0x7100_0000_00));
        assert!(poison_poolinfo_load_base(0x7100_8080_80 + 0x98, 0x7100_8080_80));
        assert!(!poison_poolinfo_load_base(0x70a0_ee00_00, 0x70a0_ee00_00));
        assert!(!poison_poolinfo_load_base(0x7100_0000_99, 0));
        assert_eq!(
            hash_walk_skip_slot(0, 8),
            (8, BINNED3_HASH_WALK_INNER_DONE)
        );
        assert_eq!(
            hash_walk_skip_slot(0, 16),
            (8, BINNED3_HASH_WALK_INNER_NEXT)
        );
        assert_eq!(
            hash_walk_skip_slot(8, 16),
            (16, BINNED3_HASH_WALK_INNER_DONE)
        );
        assert!(poison_poolinfo_load_base(0x7100_0001_79, 0x7100_0000_01));
        assert_eq!(
            object_walk_skip_slot(0, 8),
            (8, BINNED3_OBJECT_WALK_DONE)
        );
        assert_eq!(
            object_walk_skip_slot(0, 16),
            (8, BINNED3_OBJECT_WALK_NEXT)
        );
    }

    #[test]
    fn silent_hill_poolinfo_page_is_an_eight_tib_class_pointer() {
        assert!(is_eight_tib_class_pointer(0x0000_0807_6ee0_4800));
        assert_eq!(0x0000_0807_6ee0_4800 >> 40, EIGHT_TIB_CLASS_PREFIX);
        assert!(!is_eight_tib_class_pointer(0x0000_0080_00df_0000));
        assert!(!is_eight_tib_class_pointer(0x0000_0070_00e9_0000));
        assert!(!is_eight_tib_class_pointer(0x0000_0070_a0ee_0000));
        assert!(!is_eight_tib_class_pointer(0));
        assert_eq!(
            poolinfo_table_base(0x0000_0807_6ee0_4828, 0x0000_0807_6ee0_4800),
            0x0000_0807_6ee0_4800
        );
    }

    #[test]
    fn silent_hill_getorcreate_pin_base_qword_is_garbage_poolinfo() {
        assert!(looks_like_garbage_poolinfo_page(0x0001_e701_0002_00));
        assert!(!looks_like_garbage_poolinfo_page(0));
        assert!(!looks_like_garbage_poolinfo_page(0x0000_0807_6ee0_4800));
        assert!(!looks_like_garbage_poolinfo_page(0x7100_0000_01));
        assert!(!looks_like_garbage_poolinfo_page(0x7000_e900_00));
        assert!(is_getorcreate_slot_load_pc(BINNED3_GETORCREATE_SLOT_LOAD));
        assert!(is_getorcreate_slot_load_pc(BINNED3_GETORCREATE_SLOT_GROW));
        assert!(!is_getorcreate_slot_load_pc(BINNED3_GETORCREATE_ALLOC));
        assert_eq!(BINNED3_GETORCREATE_SLOT_LOAD, 0x0000_0008_0123_7DDB);
        assert_eq!(BINNED3_GETORCREATE_SLOT_GROW, 0x0000_0008_0123_7E0F);
        assert_eq!(BINNED3_GETORCREATE_ALLOC, 0x0000_0008_0123_7CE1);
    }

    #[test]
    fn silent_hill_realloc_memcpy_dest_is_a_committable_heap_store() {
        assert!(is_committable_heap_store(0x70d4_61f5_80));
        assert!(is_committable_heap_store(0x75e0_e440_00));
        assert!(!is_committable_heap_store(0x7100_0000_01));
        assert!(!is_committable_heap_store(0x0001_e701_0002_00));
        assert!(!is_committable_heap_store(0));
        assert!(is_binned3_memcpy_store_pc(0x0000_0008_0112_D8EC));
        assert!(is_binned3_memcpy_store_pc(BINNED3_MEMCPY_STORE_START));
        assert!(!is_binned3_memcpy_store_pc(BINNED3_MEMCPY_STORE_END));
        assert!(!is_binned3_memcpy_store_pc(BINNED3_GETORCREATE_SLOT_LOAD));
        assert_eq!(BINNED3_MEMCPY_AFTER_COPY, 0x0000_0008_0112_D957);
        assert_eq!(BINNED3_REALLOC_BUFFER_SLOT, 0x48);
        assert_eq!(BINNED3_REALLOC_OBJECT_SLOT, -0x70);
        assert!(is_committable_heap_store(0x704c_3d62_00));
        assert_eq!(BINNED3_TARRAY_SENTINEL_STORE, 0x0000_0008_0258_6AA4);
        assert_eq!(
            BINNED3_TARRAY_SENTINEL_NEXT - BINNED3_TARRAY_SENTINEL_STORE,
            8
        );
        assert!(is_binned3_memcpy_helper_pc(BINNED3_MEMCPY_HELPER));
        assert!(is_binned3_memcpy_helper_pc(0x0000_0008_011A_A100));
        assert!(!is_binned3_memcpy_helper_pc(BINNED3_TARRAY_SENTINEL_STORE));
        assert!(is_guest_image_return(0x0000_0008_0123_A582));
        assert!(!is_guest_image_return(0x70d4_61f5_80));
        assert_eq!(
            unmapped_heap_store_length(&[0x42, 0xc7, 0x04, 0xb8, 0x00, 0x00, 0x00, 0x80]),
            Some(8),
            "TArray sentinel `mov dword [rax+r15*4], 80000000h`"
        );
        assert_eq!(
            unmapped_heap_store_length(&[0x41, 0x0f, 0x11, 0x45, 0x00]),
            Some(5),
            "movups [r13], xmm0"
        );
        assert_eq!(
            unmapped_heap_store_length(&[0x8b, 0x00]),
            None,
            "loads must not be skipped"
        );
        assert_eq!(unmapped_heap_store_length(&[0xe8, 0, 0, 0, 0]), None);
    }
}
