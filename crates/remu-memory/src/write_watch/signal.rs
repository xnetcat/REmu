// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! The process SIGBUS/SIGSEGV handlers that turn protection faults into dirty pages.

use std::ffi::c_void;
use std::io;
use std::mem::MaybeUninit;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::page::{align_down, host_page_size_infallible};
use super::snapshot::{ACTIVE_HANDLERS, WATCH_SNAPSHOT};

static RECOVERED_DISARMED_FAULTS: AtomicU64 = AtomicU64::new(0);

static HANDLER_RESULT: OnceLock<Result<(), i32>> = OnceLock::new();

struct PreviousAction(std::cell::UnsafeCell<MaybeUninit<libc::sigaction>>);

// SAFETY: installation initializes the slot before a watched page is published and the stored
// action is immutable afterwards. Only the signal handler reads it.
unsafe impl Sync for PreviousAction {}

impl PreviousAction {
    const fn new() -> Self {
        Self(std::cell::UnsafeCell::new(MaybeUninit::uninit()))
    }

    const fn as_mut_ptr(&self) -> *mut libc::sigaction {
        self.0.get().cast::<libc::sigaction>()
    }

    unsafe fn get(&self) -> &libc::sigaction {
        // SAFETY: successful handler installation initializes both previous-action slots.
        unsafe { &*self.as_mut_ptr() }
    }
}

static PREVIOUS_SIGBUS: PreviousAction = PreviousAction::new();

static PREVIOUS_SIGSEGV: PreviousAction = PreviousAction::new();

pub(super) fn install_fault_handlers() -> io::Result<()> {
    let result = HANDLER_RESULT.get_or_init(|| {
        // SAFETY: the static previous-action slots outlive the process and the installed callback
        // has the SA_SIGINFO ABI required by sigaction.
        unsafe { install_fault_handlers_once() }
    });
    result.map_err(io::Error::from_raw_os_error)
}

unsafe fn install_fault_handlers_once() -> Result<(), i32> {
    // SAFETY: a zeroed sigaction is a valid base before its mask, callback and flags are set.
    let mut action = unsafe { std::mem::zeroed::<libc::sigaction>() };
    action.sa_sigaction = write_fault_signal_handler as *const () as usize;
    action.sa_flags = libc::SA_SIGINFO;
    // SAFETY: action.sa_mask is live writable storage.
    unsafe { libc::sigemptyset(&raw mut action.sa_mask) };
    // SAFETY: both actions are valid and PREVIOUS_SIGBUS is uninitialized writable storage.
    if unsafe {
        libc::sigaction(
            libc::SIGBUS,
            &raw const action,
            PREVIOUS_SIGBUS.as_mut_ptr(),
        )
    } != 0
    {
        return Err(last_errno());
    }
    // SAFETY: both actions are valid and PREVIOUS_SIGSEGV is uninitialized writable storage.
    if unsafe {
        libc::sigaction(
            libc::SIGSEGV,
            &raw const action,
            PREVIOUS_SIGSEGV.as_mut_ptr(),
        )
    } != 0
    {
        let error = last_errno();
        // SAFETY: restore the successfully captured SIGBUS action on partial installation.
        unsafe {
            libc::sigaction(
                libc::SIGBUS,
                PREVIOUS_SIGBUS.as_mut_ptr(),
                std::ptr::null_mut(),
            );
        }
        return Err(error);
    }
    Ok(())
}

fn last_errno() -> i32 {
    io::Error::last_os_error().raw_os_error().unwrap_or(1)
}

extern "C" fn write_fault_signal_handler(
    signal: libc::c_int,
    information: *mut libc::siginfo_t,
    context: *mut c_void,
) {
    ACTIVE_HANDLERS.fetch_add(1, Ordering::SeqCst);
    let fault_address = if information.is_null() {
        0
    } else {
        // SAFETY: SA_SIGINFO supplies a live siginfo_t for the duration of this callback.
        unsafe { (*information).si_addr().addr() as u64 }
    };
    let handled = try_handle_write_fault(fault_address);
    ACTIVE_HANDLERS.fetch_sub(1, Ordering::SeqCst);
    if handled {
        return;
    }
    // SAFETY: the previous action was captured before this handler was published.
    unsafe { chain_previous_action(signal, information, context) };
}

fn try_handle_write_fault(fault_address: u64) -> bool {
    if fault_address == 0 {
        write_signal_diagnostic(b"[REMu][WATCH] unhandled fault: null address\n");
        return false;
    }
    let snapshot_pointer = WATCH_SNAPSHOT.load(Ordering::SeqCst);
    if snapshot_pointer.is_null() {
        write_signal_diagnostic(b"[REMu][WATCH] unhandled fault: no watch snapshot\n");
        return false;
    }
    // SAFETY: ACTIVE_HANDLERS pins the immutable snapshot until this handler leaves.
    let snapshot = unsafe { &*snapshot_pointer };
    let page_size = host_page_size_infallible();
    let page_address = align_down(fault_address, page_size);
    let has_page = snapshot.contains_page(page_address);
    if !has_page && !snapshot.contains_writable_page(page_address) {
        // This handler receives every process SIGSEGV/SIGBUS while CPU write watching is active.
        // A fault outside the immutable watch snapshot is not a write-watch failure; forward it
        // unchanged so the system crash report retains the guest RIP and register state.
        write_signal_diagnostic(
            b"[REMu][WATCH] unrelated fault outside watched memory; forwarding\n",
        );
        return false;
    }
    let has_armed_owner = has_page
        && snapshot.owners(page_address).any(|owner| {
            // SAFETY: the snapshot pins all range allocations referenced by its owners.
            let range = unsafe { &*owner.range };
            range.pages[owner.page_index].armed.load(Ordering::Acquire)
        });

    let Ok(address) = usize::try_from(page_address) else {
        write_signal_diagnostic(b"[REMu][WATCH] unhandled fault: address conversion failed\n");
        return false;
    };
    let Ok(length) = usize::try_from(page_size) else {
        write_signal_diagnostic(b"[REMu][WATCH] unhandled fault: length conversion failed\n");
        return false;
    };
    // SAFETY: all page owners registered this live mapping. This is the exact protected host page.
    if unsafe {
        libc::mprotect(
            address as *mut c_void,
            length,
            libc::PROT_READ | libc::PROT_WRITE,
        )
    } != 0
    {
        write_signal_diagnostic(b"[REMu][WATCH] unhandled fault: mprotect failed\n");
        return false;
    }

    // A watched page with no armed owner is required to be read/write: a previous handled store
    // disarmed it. Guest process-image protection and overlapping resource lifetimes can race that
    // invariant under Rosetta. Because the immutable snapshot still proves this is a live unified-
    // memory resource, recover its expected write access and mark every owner dirty instead of
    // forwarding a fatal SIGBUS. Legitimate guest protection changes retire the watches before
    // reaching mprotect, so they do not enter this path.
    if !has_armed_owner {
        // Multiple Unity writers can already be trapped on the same page when the first handler
        // disarms it. The later handlers are expected recovery, not actionable diagnostics;
        // writing one line per collision from signal context caused severe stderr and audio
        // stalls in update-heavy scenes.
        RECOVERED_DISARMED_FAULTS.fetch_add(1, Ordering::Relaxed);
    }

    for owner in snapshot.owners(page_address) {
        // SAFETY: the active snapshot pins the range, and page_index was produced from that range.
        let range = unsafe { &*owner.range };
        let watched_page = &range.pages[owner.page_index];
        if watched_page.armed.swap(false, Ordering::AcqRel) {
            range.generation.fetch_add(1, Ordering::AcqRel);
        }
        watched_page.dirty.store(true, Ordering::Release);
        range.dirty.store(true, Ordering::Release);
    }
    true
}

fn write_signal_diagnostic(message: &'static [u8]) {
    // SAFETY: `write` is async-signal-safe, and the static byte slice remains live throughout the
    // call. Ignore short writes because this is a last-chance crash diagnostic.
    unsafe {
        libc::write(
            libc::STDERR_FILENO,
            message.as_ptr().cast::<c_void>(),
            message.len(),
        );
    }
}

unsafe fn chain_previous_action(
    signal: libc::c_int,
    information: *mut libc::siginfo_t,
    context: *mut c_void,
) {
    let previous = if signal == libc::SIGBUS {
        // SAFETY: installation initialized this slot before the handler became reachable.
        unsafe { PREVIOUS_SIGBUS.get() }
    } else {
        // SAFETY: installation initialized this slot before the handler became reachable.
        unsafe { PREVIOUS_SIGSEGV.get() }
    };
    let callback = previous.sa_sigaction;
    if callback == libc::SIG_DFL {
        // Restoring the default and returning causes the instruction to fault again, preserving
        // the original signal and crash context.
        // SAFETY: previous is the complete action captured by sigaction.
        unsafe { libc::sigaction(signal, previous, std::ptr::null_mut()) };
    } else if callback != libc::SIG_IGN {
        if previous.sa_flags & libc::SA_SIGINFO != 0 {
            // SAFETY: SA_SIGINFO identifies the stored callback's three-argument ABI.
            let callback: unsafe extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut c_void) =
                unsafe { std::mem::transmute(callback) };
            // SAFETY: forward the exact signal arguments supplied by the kernel.
            unsafe { callback(signal, information, context) };
        } else {
            // SAFETY: without SA_SIGINFO the stored callback has the one-argument signal ABI.
            let callback: unsafe extern "C" fn(libc::c_int) =
                unsafe { std::mem::transmute(callback) };
            // SAFETY: forward the signal number supplied by the kernel.
            unsafe { callback(signal) };
        }
    }
}
