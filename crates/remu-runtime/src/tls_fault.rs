// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Fault-driven backstop for guest `fs:` accesses the AOT rewriter could not patch in place.
//!
//! Under Rosetta the FS base is zero, so an unpatched `mov rax, fs:[0x28]` dies at address
//! `0x28` with no TLS attribution. This handler decodes the faulting instruction, emulates the
//! absolute/indexed FS forms it understands against [`current_guest_thread_pointer`], and advances
//! RIP. Unsupported FS shapes log a loud refusal before the fault is forwarded.

use std::sync::atomic::AtomicU64;
#[cfg(any(test, all(target_os = "macos", target_arch = "x86_64")))]
use std::sync::atomic::Ordering;

/// Counts FS faults the backstop successfully emulated.
#[allow(dead_code)] // read by diagnostics/tests; written only on the x86_64 fault path
pub(crate) static FS_FAULT_EMULATED: AtomicU64 = AtomicU64::new(0);

/// Counts FS faults the backstop recognized but refused to emulate.
#[allow(dead_code)] // read by diagnostics/tests; written only on the x86_64 fault path
pub(crate) static FS_FAULT_REFUSED: AtomicU64 = AtomicU64::new(0);

/// Registers the process-global FS fault backstop. Safe to call once at process entry.
///
/// Only the x86_64 guest under Rosetta emits `fs:` memory operands; native arm64 builds keep a
/// no-op installer so the call site in `probe_entry` stays unconditional under `macos`.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(crate) fn install_fs_fault_backstop() {
    let _ = remu_memory::set_fs_fault_handler(handle_fs_fault);
}

#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
#[allow(dead_code)] // `probe_entry` (the only caller) is itself x86_64-only.
pub(crate) fn install_fs_fault_backstop() {}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
use crate::current_guest_thread_pointer;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
use crate::link::fs_decode::{FsAccessKind, classify_fs_instruction_bytes};
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
use std::ffi::c_void;

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
unsafe extern "C" fn handle_fs_fault(fault_address: u64, context: *mut c_void) -> bool {
    if context.is_null() {
        return false;
    }
    // SAFETY: SA_SIGINFO supplies a live Darwin ucontext for the duration of this callback.
    let Some(machine) = (unsafe { machine_context_mut(context) }) else {
        return false;
    };
    let rip = machine.instruction_pointer;
    if rip == 0 {
        return false;
    }
    let mut insn = [0_u8; 15];
    // SAFETY: guest code is identity-mapped; RIP names the faulting instruction bytes.
    if !unsafe { read_instruction_bytes(rip, &mut insn) } {
        return false;
    }
    let Some(site) = classify_fs_instruction_bytes(&insn) else {
        return false;
    };
    let thread_pointer = current_guest_thread_pointer();
    if thread_pointer == 0 {
        FS_FAULT_REFUSED.fetch_add(1, Ordering::Relaxed);
        write_fs_fault_diagnostic(
            b"[REMu][TLS][FS] fault-backstop refused: guest thread pointer is null\n",
        );
        return false;
    }

    let Some(effective_address) = fs_effective_address(thread_pointer, &site.kind) else {
        FS_FAULT_REFUSED.fetch_add(1, Ordering::Relaxed);
        write_fs_fault_diagnostic(
            b"[REMu][TLS][FS] fault-backstop refused: unsupported fs: effective address\n",
        );
        return false;
    };

    match site.kind {
        FsAccessKind::LeaGpr { destination, .. } => {
            // LEA does not touch memory and should not fault; if we are here the AOT pass missed it.
            write_gpr_width(machine, destination, 8, effective_address);
        }
        FsAccessKind::LoadGpr {
            destination,
            width_bytes,
            ..
        } => {
            if fault_address != effective_address {
                return false;
            }
            let Some(value) = read_guest_width(effective_address, width_bytes) else {
                FS_FAULT_REFUSED.fetch_add(1, Ordering::Relaxed);
                write_fs_fault_diagnostic(
                    b"[REMu][TLS][FS] fault-backstop refused: load target unreadable\n",
                );
                return false;
            };
            write_gpr_width(machine, destination, width_bytes, value);
        }
        FsAccessKind::StoreGpr {
            source,
            width_bytes,
            ..
        } => {
            if fault_address != effective_address {
                return false;
            }
            let value = read_gpr_width(machine, source, width_bytes);
            if !write_guest_width(effective_address, width_bytes, value) {
                FS_FAULT_REFUSED.fetch_add(1, Ordering::Relaxed);
                write_fs_fault_diagnostic(
                    b"[REMu][TLS][FS] fault-backstop refused: store target unwritable\n",
                );
                return false;
            }
        }
        FsAccessKind::StoreImm {
            width_bytes,
            immediate,
            ..
        } => {
            if fault_address != effective_address {
                return false;
            }
            if !write_guest_width(effective_address, width_bytes, immediate as u64) {
                FS_FAULT_REFUSED.fetch_add(1, Ordering::Relaxed);
                write_fs_fault_diagnostic(
                    b"[REMu][TLS][FS] fault-backstop refused: imm store target unwritable\n",
                );
                return false;
            }
        }
        FsAccessKind::Other => {
            FS_FAULT_REFUSED.fetch_add(1, Ordering::Relaxed);
            write_fs_fault_diagnostic(
                b"[REMu][TLS][FS] fault-backstop refused: non-mov/lea fs: op (loud; not silent at 0)\n",
            );
            return false;
        }
    }

    machine.instruction_pointer = rip.saturating_add(site.length as u64);
    FS_FAULT_EMULATED.fetch_add(1, Ordering::Relaxed);
    true
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn fs_effective_address(thread_pointer: u64, kind: &FsAccessKind) -> Option<u64> {
    let displacement = match *kind {
        FsAccessKind::LoadGpr { displacement, .. }
        | FsAccessKind::StoreGpr { displacement, .. }
        | FsAccessKind::StoreImm { displacement, .. }
        | FsAccessKind::LeaGpr { displacement, .. } => displacement,
        FsAccessKind::Other => return None,
    };
    // Absolute FS forms only in the classifier's AOT shapes; indexed forms remain Other and refuse
    // loudly above.
    Some(thread_pointer.wrapping_add(displacement as i64 as u64))
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn gpr_mut(machine: &mut MachineContext, register: u8) -> Option<&mut u64> {
    // Darwin's mcontext order is not the AMD64 register-number encoding; map explicitly.
    Some(match register {
        0 => &mut machine.rax,
        1 => &mut machine.rcx,
        2 => &mut machine.rdx,
        3 => &mut machine.rbx,
        4 => &mut machine.rsp,
        5 => &mut machine.rbp,
        6 => &mut machine.rsi,
        7 => &mut machine.rdi,
        8 => &mut machine.r8,
        9 => &mut machine.r9,
        10 => &mut machine.r10,
        11 => &mut machine.r11,
        12 => &mut machine.r12,
        13 => &mut machine.r13,
        14 => &mut machine.r14,
        15 => &mut machine.r15,
        _ => return None,
    })
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn write_gpr_width(machine: &mut MachineContext, register: u8, width_bytes: u8, value: u64) {
    let Some(slot) = gpr_mut(machine, register) else {
        return;
    };
    match width_bytes {
        1 => *slot = (*slot & !0xff) | (value & 0xff),
        2 => *slot = (*slot & !0xffff) | (value & 0xffff),
        4 => *slot = value as u32 as u64,
        _ => *slot = value,
    }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn read_gpr_width(machine: &MachineContext, register: u8, width_bytes: u8) -> u64 {
    let value = match register {
        0 => machine.rax,
        1 => machine.rcx,
        2 => machine.rdx,
        3 => machine.rbx,
        4 => machine.rsp,
        5 => machine.rbp,
        6 => machine.rsi,
        7 => machine.rdi,
        8 => machine.r8,
        9 => machine.r9,
        10 => machine.r10,
        11 => machine.r11,
        12 => machine.r12,
        13 => machine.r13,
        14 => machine.r14,
        15 => machine.r15,
        _ => 0,
    };
    match width_bytes {
        1 => value & 0xff,
        2 => value & 0xffff,
        4 => value as u32 as u64,
        _ => value,
    }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn read_guest_width(address: u64, width_bytes: u8) -> Option<u64> {
    let width = usize::from(width_bytes);
    if width == 0 || width > 8 {
        return None;
    }
    let mut bytes = [0_u8; 8];
    // SAFETY: the effective address is a guest TLS location inside the identity-mapped image.
    unsafe {
        for (index, slot) in bytes[..width].iter_mut().enumerate() {
            let pointer = (address as usize).checked_add(index)?;
            *slot = std::ptr::read_volatile(pointer as *const u8);
        }
    }
    Some(u64::from_le_bytes(bytes))
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn write_guest_width(address: u64, width_bytes: u8, value: u64) -> bool {
    let width = usize::from(width_bytes);
    if width == 0 || width > 8 {
        return false;
    }
    let bytes = value.to_le_bytes();
    // SAFETY: the effective address is a guest TLS location inside the identity-mapped image.
    unsafe {
        for (index, byte) in bytes[..width].iter().enumerate() {
            let Some(pointer) = (address as usize).checked_add(index) else {
                return false;
            };
            std::ptr::write_volatile(pointer as *mut u8, *byte);
        }
    }
    true
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
unsafe fn read_instruction_bytes(rip: u64, insn: &mut [u8; 15]) -> bool {
    let Ok(base) = usize::try_from(rip) else {
        return false;
    };
    // SAFETY: RIP is the interrupted guest PC; instruction length is at most 15 bytes.
    for (index, slot) in insn.iter_mut().enumerate() {
        let pointer = match base.checked_add(index) {
            Some(pointer) => pointer,
            None => return false,
        };
        *slot = unsafe { std::ptr::read_volatile(pointer as *const u8) };
    }
    true
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn write_fs_fault_diagnostic(message: &[u8]) {
    // SAFETY: `write` is async-signal-safe.
    unsafe {
        libc::write(
            libc::STDERR_FILENO,
            message.as_ptr().cast::<c_void>(),
            message.len(),
        );
    }
}

/// Darwin `__darwin_ucontext` truncated after the machine-context pointer.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
#[repr(C)]
pub(crate) struct UserContext {
    _on_stack: libc::c_int,
    _signal_mask: u32,
    _stack: libc::stack_t,
    _link: *mut UserContext,
    _machine_context_size: usize,
    machine_context: *mut MachineContext,
}

/// Darwin `_STRUCT_MCONTEXT64` integer state used by the FS emulator.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
#[repr(C)]
pub(crate) struct MachineContext {
    _trap_number: u16,
    _cpu: u16,
    _error: u32,
    _fault_address: u64,
    pub(crate) rax: u64,
    pub(crate) rbx: u64,
    pub(crate) rcx: u64,
    pub(crate) rdx: u64,
    pub(crate) rdi: u64,
    pub(crate) rsi: u64,
    pub(crate) rbp: u64,
    pub(crate) rsp: u64,
    pub(crate) r8: u64,
    pub(crate) r9: u64,
    pub(crate) r10: u64,
    pub(crate) r11: u64,
    pub(crate) r12: u64,
    pub(crate) r13: u64,
    pub(crate) r14: u64,
    pub(crate) r15: u64,
    pub(crate) instruction_pointer: u64,
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(crate) unsafe fn machine_context_mut(
    context: *mut c_void,
) -> Option<&'static mut MachineContext> {
    // SAFETY: caller guarantees a live ucontext from SA_SIGINFO.
    let user = unsafe { &mut *context.cast::<UserContext>() };
    if user.machine_context.is_null() {
        return None;
    }
    // SAFETY: the kernel initializes the machine context for every SIGSEGV/SIGBUS delivery.
    Some(unsafe { &mut *user.machine_context })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::link::fs_decode::{FsAccessKind, classify_fs_instruction_bytes};

    #[test]
    fn stack_guard_load_classifies_for_fault_emulation() {
        let encoding: [u8; 9] = [0x64, 0x48, 0x8b, 0x04, 0x25, 0x28, 0, 0, 0];
        let site = classify_fs_instruction_bytes(&encoding).expect("fs site");
        assert_eq!(
            site.kind,
            FsAccessKind::LoadGpr {
                destination: 0,
                width_bytes: 8,
                displacement: 0x28,
            }
        );
    }

    #[test]
    fn fault_counters_are_readable() {
        let _ = FS_FAULT_EMULATED.load(Ordering::Relaxed);
        let _ = FS_FAULT_REFUSED.load(Ordering::Relaxed);
    }
}
