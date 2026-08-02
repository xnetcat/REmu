// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! x86-64 context switching for the guest fiber ABI.
//!
//! A fiber is a stack plus a saved register set. Switching between two of them is the classic
//! `setjmp`/`longjmp` pair over a context owned here, plus one entry trampoline that starts a fiber
//! on the guest-supplied stack for the first time. The same shape already exists in this crate for
//! guest thread exit (`invoke_guest_entry_with_exit_boundary` / `escape_guest_entry`), which is
//! what establishes that switching `rsp` under Rosetta translation is sound.
//!
//! The register-save layout is owned here and never reaches the guest, so both floating-point
//! control words sit at naturally aligned offsets rather than at the packed offsets the console
//! library happens to use.

use super::{FiberContext, FiberData};

/// Saves the caller's resumable state into `context`.
///
/// Returns 0 when it saves and 1 when execution resumes here through [`fiber_long_jmp`], exactly
/// as `setjmp` reports the two arrivals apart.
///
/// # Safety
///
/// `context` must point to a live, writable [`FiberContext`] that outlives every resume through
/// it. Resuming a context whose stack frame has returned jumps into freed stack.
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
// A stable export prevents ThinLTO from internalizing the naked body into a different codegen
// unit than its callers, which intermittently left the mangled symbol undefined at link time as
// unrelated code in this crate shifted partitioning.
#[unsafe(export_name = "remu_fiber_set_jmp")]
pub(super) unsafe extern "C" fn fiber_set_jmp(_context: *mut FiberContext) -> u32 {
    core::arch::naked_asm!(
        "mov [rdi + 0x00], rax",
        // The return address is the resume point: a later long jump puts it back on the restored
        // stack and returns to it.
        "mov rdx, [rsp]",
        "mov [rdi + 0x10], rdx",
        "mov [rdi + 0x08], rcx",
        "mov [rdi + 0x18], rbx",
        "mov [rdi + 0x20], rsp",
        "mov [rdi + 0x28], rbp",
        "mov [rdi + 0x30], r8",
        "mov [rdi + 0x38], r9",
        "mov [rdi + 0x40], r10",
        "mov [rdi + 0x48], r11",
        "mov [rdi + 0x50], r12",
        "mov [rdi + 0x58], r13",
        "mov [rdi + 0x60], r14",
        "mov [rdi + 0x68], r15",
        "stmxcsr [rdi + 0x70]",
        "fnstcw [rdi + 0x74]",
        "xor eax, eax",
        "ret",
    );
}

/// Resumes the state saved in `context`, arriving as a return of 1 from [`fiber_set_jmp`].
///
/// # Safety
///
/// `context` must hold state saved by [`fiber_set_jmp`] on a stack that is still live.
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
#[unsafe(export_name = "remu_fiber_long_jmp")]
pub(super) unsafe extern "C" fn fiber_long_jmp(_context: *mut FiberContext) -> ! {
    core::arch::naked_asm!(
        // Restore the saved rounding and masking bits while keeping the live exception flags,
        // which are sticky status the resumed code may still be reading.
        "stmxcsr [rsp - 8]",
        "mov eax, [rdi + 0x70]",
        "and eax, 0xffffffc0",
        "mov ecx, [rsp - 8]",
        "and ecx, 0x3f",
        "xor ecx, eax",
        "mov [rsp - 8], ecx",
        "ldmxcsr [rsp - 8]",
        "mov rax, [rdi + 0x00]",
        "mov rcx, [rdi + 0x08]",
        "mov rdx, [rdi + 0x10]",
        "mov rbx, [rdi + 0x18]",
        "mov rsp, [rdi + 0x20]",
        "mov rbp, [rdi + 0x28]",
        "mov r8,  [rdi + 0x30]",
        "mov r9,  [rdi + 0x38]",
        "mov r10, [rdi + 0x40]",
        "mov r11, [rdi + 0x48]",
        "mov r12, [rdi + 0x50]",
        "mov r13, [rdi + 0x58]",
        "mov r14, [rdi + 0x60]",
        "mov r15, [rdi + 0x68]",
        "fldcw [rdi + 0x74]",
        // Put the saved resume address back where `ret` expects it, then report the second arrival.
        "mov [rsp], rdx",
        "mov eax, 1",
        "ret",
    );
}

/// Enters a fiber for the first time on its own stack.
///
/// The guest entry receives `(arg_on_initialize, arg_on_run_to)` and is not expected to return;
/// a return is a guest defect and is routed to `force_quit` rather than off the end of the stack.
///
/// # Safety
///
/// `data` must describe a live fiber whose stack top is sixteen-byte aligned and writable, and
/// `force_quit` must not return.
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
#[unsafe(export_name = "remu_fiber_switch_entry")]
pub(super) unsafe extern "C" fn fiber_switch_entry(
    _data: *const FiberData,
    _set_floating_point: u32,
) -> ! {
    core::arch::naked_asm!(
        "mov r11, rdi",
        // Move onto the fiber's own stack before touching anything else.
        "mov rsp, [r11 + 0x18]",
        "xor ebp, ebp",
        // The fiber this one displaced becomes idle only once its registers can no longer be
        // reached, which is here: its stack is no longer current.
        "mov r10, [r11 + 0x20]",
        "test r10, r10",
        "jz 2f",
        "mov dword ptr [r10], 2",
        "2:",
        "test esi, esi",
        "jz 3f",
        "ldmxcsr [r11 + 0x28]",
        "fldcw [r11 + 0x2c]",
        "3:",
        "mov rdi, [r11 + 0x08]",
        "mov rsi, [r11 + 0x10]",
        "mov r11, [r11 + 0x00]",
        // A fresh fiber must not observe register contents from whatever ran before it.
        "xor eax, eax",
        "xor ebx, ebx",
        "xor ecx, ecx",
        "xor edx, edx",
        "xor r8d, r8d",
        "xor r9d, r9d",
        "xor r10d, r10d",
        "xor r12d, r12d",
        "xor r13d, r13d",
        "xor r14d, r14d",
        "xor r15d, r15d",
        "call r11",
        // The entry returned. There is no frame beneath it to return into.
        "mov edi, 1",
        "call {force_quit}",
        "ud2",
        force_quit = sym super::fiber_force_quit,
    );
}
