// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! The process SIGBUS/SIGSEGV handlers that turn protection faults into dirty pages.

use std::ffi::c_void;
use std::io;
use std::mem::MaybeUninit;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::diagnostic::{
    DiagnosticBuffer, FaultReason, FaultRegister, FaultReport, FaultSpace, format_fault_registers,
    format_fault_report, format_fault_stack_words,
};
use super::guest_space::contains_guest_address;
use super::page::{align_down, host_page_size_infallible};
use super::snapshot::{ACTIVE_HANDLERS, WATCH_SNAPSHOT};

static RECOVERED_DISARMED_FAULTS: AtomicU64 = AtomicU64::new(0);

/// Fills a guest page with a producer's (GPU) bytes so a faulting guest read observes them.
///
/// It is invoked directly from the SIGSEGV/SIGBUS handler with the page made temporarily writable,
/// and therefore **must be async-signal-safe**: no heap allocation, no lock any non-handler code
/// can hold, and no re-entry into the watch registry — only a bounded copy into
/// `[page_address, page_address + page_size)`. Returning `false` leaves the fault unhandled and
/// forwards it, so the failure is loud rather than a silent wrong-bytes read.
///
/// This is the D5 integration seam: the gpu-metal consumer registers a materializer that copies
/// the resident texture's bytes (from a host-visible staging buffer it already owns) into the page.
pub type GuestReadMaterializer = unsafe extern "C" fn(page_address: u64, page_size: u64) -> bool;

/// The process-global read-watch materializer, stored as a raw function-pointer address (0 = none).
static READ_MATERIALIZER: AtomicUsize = AtomicUsize::new(0);

/// Emulates an unpatched guest `fs:` access against the guest thread pointer.
///
/// Invoked from the SIGSEGV/SIGBUS handler before write-watch recovery (and before the null-address
/// short-circuit). Must be async-signal-safe. Returning `true` means the fault was emulated and
/// the interrupted context was advanced past the instruction; `false` leaves the fault unhandled.
pub type GuestFsFaultHandler =
    unsafe extern "C" fn(fault_address: u64, context: *mut c_void) -> bool;

/// The process-global FS fault backstop, stored as a raw function-pointer address (0 = none).
static FS_FAULT_HANDLER: AtomicUsize = AtomicUsize::new(0);

/// Registers the read-watch materializer, returning the previously registered one if any.
pub(super) fn set_read_watch_materializer(
    materializer: GuestReadMaterializer,
) -> Option<GuestReadMaterializer> {
    let previous = READ_MATERIALIZER.swap(materializer as usize, Ordering::AcqRel);
    // SAFETY: a nonzero slot only ever holds a value stored from a live `GuestReadMaterializer`.
    (previous != 0)
        .then(|| unsafe { std::mem::transmute::<usize, GuestReadMaterializer>(previous) })
}

/// Returns the currently registered read-watch materializer, if any.
pub(super) fn read_watch_materializer() -> Option<GuestReadMaterializer> {
    let raw = READ_MATERIALIZER.load(Ordering::Acquire);
    // SAFETY: a nonzero slot only ever holds a value stored from a live `GuestReadMaterializer`.
    (raw != 0).then(|| unsafe { std::mem::transmute::<usize, GuestReadMaterializer>(raw) })
}

/// Registers the FS fault backstop, returning the previously registered handler if any.
pub(super) fn set_fs_fault_handler(handler: GuestFsFaultHandler) -> Option<GuestFsFaultHandler> {
    let previous = FS_FAULT_HANDLER.swap(handler as usize, Ordering::AcqRel);
    // SAFETY: a nonzero slot only ever holds a value stored from a live `GuestFsFaultHandler`.
    (previous != 0).then(|| unsafe { std::mem::transmute::<usize, GuestFsFaultHandler>(previous) })
}

fn fs_fault_handler() -> Option<GuestFsFaultHandler> {
    let raw = FS_FAULT_HANDLER.load(Ordering::Acquire);
    // SAFETY: a nonzero slot only ever holds a value stored from a live `GuestFsFaultHandler`.
    (raw != 0).then(|| unsafe { std::mem::transmute::<usize, GuestFsFaultHandler>(raw) })
}

/// Whether a delivered fault was a read, a write, or could not be classified from the context.
#[derive(Clone, Copy, Eq, PartialEq)]
enum FaultAccess {
    Read,
    Write,
    Unknown,
}

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

    /// # Safety
    ///
    /// The slot must have been initialized by a successful `install_fault_handlers_once`, which is
    /// the sole writer and runs exactly once before any handler can read it.
    unsafe fn get(&self) -> &libc::sigaction {
        // SAFETY: successful handler installation initializes both previous-action slots.
        unsafe { &*self.as_mut_ptr() }
    }
}

static PREVIOUS_SIGBUS: PreviousAction = PreviousAction::new();

static PREVIOUS_SIGSEGV: PreviousAction = PreviousAction::new();

/// Installs the process SIGSEGV/SIGBUS reporter (and write-watch recovery) once.
///
/// Safe to call before any CPU write-watch is armed: unwatched faults are diagnosed and
/// forwarded. Must also be paired with [`install_thread_signal_stack`] on every thread that
/// may take a guest fault, because the handler runs with `SA_ONSTACK`.
pub fn install_fault_handlers() -> io::Result<()> {
    // Resolve the journal's enable flag here so the handler never reads the environment.
    let _ = crate::guest_write_journal_enabled();
    let result = HANDLER_RESULT.get_or_init(|| {
        // SAFETY: the static previous-action slots outlive the process and the installed callback
        // has the SA_SIGINFO ABI required by sigaction.
        unsafe { install_fault_handlers_once() }
    });
    result.map_err(io::Error::from_raw_os_error)
}

/// Bytes reserved for the alternate signal stack on each host thread.
///
/// Large enough for the allocation-free fault reporter plus a deep Metal/Rosetta frame; the
/// kernel delivers SIGSEGV/SIGBUS here when the main stack is exhausted or corrupted.
const SIGNAL_STACK_BYTES: usize = 256 * 1024;

/// Installs an alternate signal stack for the calling thread so `SA_ONSTACK` handlers can run.
pub fn install_thread_signal_stack() -> io::Result<()> {
    thread_local! {
        static SIGNAL_STACK: std::cell::RefCell<Option<Box<[u8]>>> =
            const { std::cell::RefCell::new(None) };
    }
    SIGNAL_STACK.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Ok(());
        }
        let mut memory = vec![0_u8; SIGNAL_STACK_BYTES].into_boxed_slice();
        // SAFETY: `memory` is uniquely owned for the life of the thread; the kernel only uses it
        // while a signal is delivered on this thread.
        let stack = libc::stack_t {
            ss_sp: memory.as_mut_ptr().cast(),
            ss_size: SIGNAL_STACK_BYTES,
            ss_flags: 0,
        };
        if unsafe { libc::sigaltstack(&raw const stack, std::ptr::null_mut()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        *slot = Some(memory);
        Ok(())
    })
}

/// # Safety
///
/// Must run exactly once (enforced by the `HANDLER_RESULT` `OnceLock` in `install_fault_handlers`)
/// so the previous-action slots are written before any delivered signal can read them.
unsafe fn install_fault_handlers_once() -> Result<(), i32> {
    // SAFETY: a zeroed sigaction is a valid base before its mask, callback and flags are set.
    let mut action = unsafe { std::mem::zeroed::<libc::sigaction>() };
    action.sa_sigaction = write_fault_signal_handler as *const () as usize;
    // SA_ONSTACK is mandatory: a stack-overflow SIGSEGV cannot be reported from the exhausted
    // stack, and the audit found no sigaltstack anywhere in the tree before Phase 0.
    action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
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
    // SIGILL is deliberately NOT installed here: the runtime's SHA-NI recovery owns it
    // (`remu-runtime` sha_ni.rs), emulating the SHA extension instructions Rosetta refuses and
    // resuming the guest. Claiming it here would silently disable that recovery.
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
    let (fault_address, fault_code) = if information.is_null() {
        (0, 0)
    } else {
        // SAFETY: SA_SIGINFO supplies a live siginfo_t for the duration of this callback.
        unsafe {
            (
                (*information).si_addr().addr() as u64,
                (*information).si_code,
            )
        }
    };
    // Unpatched `fs:` accesses fault at absolute low addresses (Rosetta FS base is 0), including
    // `fs:[0]`. Try the TLS backstop before write-watch recovery and before the null-address
    // diagnostic so a missed AOT rewrite is loud-and-emulated rather than a silent death at 0.
    if let Some(handler) = fs_fault_handler() {
        // SAFETY: the registered handler is async-signal-safe and receives the live fault context.
        if unsafe { handler(fault_address, context) } {
            ACTIVE_HANDLERS.fetch_sub(1, Ordering::SeqCst);
            return;
        }
    }
    // SAFETY: SA_SIGINFO supplies a live ucontext_t (or null) for the duration of this callback.
    let access = unsafe { fault_access(context) };
    let handled = try_handle_write_fault(fault_address, access, || {
        // SAFETY: SA_SIGINFO supplies a live ucontext_t for the duration of this callback.
        let (program_counter, registers) = unsafe { fault_machine_state(context) };
        (
            FaultReport {
                signal,
                code: fault_code,
                address: fault_address,
                program_counter,
                space: if contains_guest_address(fault_address) {
                    FaultSpace::Guest
                } else {
                    FaultSpace::Host
                },
            },
            registers,
        )
    });
    ACTIVE_HANDLERS.fetch_sub(1, Ordering::SeqCst);
    if handled {
        return;
    }
    // SAFETY: the previous action was captured before this handler was published.
    unsafe { chain_previous_action(signal, information, context) };
}

type FaultState = (FaultReport, Option<[FaultRegister; REGISTER_COUNT]>);

fn try_handle_write_fault(
    fault_address: u64,
    access: FaultAccess,
    state: impl Fn() -> FaultState,
) -> bool {
    if fault_address == 0 {
        write_fault_diagnostic(state(), FaultReason::NullAddress);
        return false;
    }
    let snapshot_pointer = WATCH_SNAPSHOT.load(Ordering::SeqCst);
    if snapshot_pointer.is_null() {
        write_fault_diagnostic(state(), FaultReason::NoSnapshot);
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
        write_fault_diagnostic(state(), FaultReason::Unwatched);
        return false;
    }
    let has_armed_owner = has_page
        && snapshot.owners(page_address).any(|owner| {
            // SAFETY: the snapshot pins all range allocations referenced by its owners.
            let range = unsafe { &*owner.range };
            range.pages[owner.page_index].armed.load(Ordering::Acquire)
        });

    let Ok(address) = usize::try_from(page_address) else {
        write_fault_diagnostic(state(), FaultReason::AddressConversion);
        return false;
    };
    let Ok(length) = usize::try_from(page_size) else {
        write_fault_diagnostic(state(), FaultReason::LengthConversion);
        return false;
    };

    // D5 read-watch: a guest READ (or an access we cannot classify as a write) of an armed, not
    // yet materialized read-watch page must materialize the producer's bytes before the read
    // resumes. A definite WRITE falls through: the guest overwrites the stale bytes, so it only
    // needs the ordinary make-writable-and-dirty path (and its owners are marked materialized so a
    // later read never overwrites those guest bytes).
    if access != FaultAccess::Write
        && has_page
        && snapshot.owners(page_address).any(|owner| {
            // SAFETY: the snapshot pins the range allocation referenced by its owner.
            let range = unsafe { &*owner.range };
            let page = &range.pages[owner.page_index];
            range.read_watch
                && page.armed.load(Ordering::Acquire)
                && !page.materialized.load(Ordering::Acquire)
        })
    {
        return materialize_read_watch_page(snapshot, page_address, address, length, &state);
    }

    // SAFETY: all page owners registered this live mapping. This is the exact protected host page.
    if unsafe {
        libc::mprotect(
            address as *mut c_void,
            length,
            libc::PROT_READ | libc::PROT_WRITE,
        )
    } != 0
    {
        write_fault_diagnostic(state(), FaultReason::ProtectFailed);
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
        // A guest write to a read-watch page overwrites the stale GPU bytes, so it is materialized
        // by definition; never copy the producer's bytes over what the guest just wrote.
        if range.read_watch {
            watched_page.materialized.store(true, Ordering::Release);
        }
        watched_page.dirty.store(true, Ordering::Release);
        range.dirty.store(true, Ordering::Release);
    }
    // The page is about to resume at the faulting guest store and remain writable until every
    // owner rearms it. Preserve the exact opening PC in the env-gated crash journal: if a later
    // allocator failure names this page, one crash can identify which guest code first took CPU
    // write ownership after the last GPU snapshot. Recording the whole page is intentional — the
    // signal layer cannot observe subsequent stores while it is disarmed.
    if crate::guest_write_journal::guest_cpu_page_journal_armed() {
        let (report, _) = state();
        crate::guest_write_journal::record_guest_cpu_page_open(
            page_address,
            page_size,
            fault_address,
            report.program_counter,
        );
    }
    true
}

/// Materializes a read-watch page from the registered producer, then leaves it readable.
///
/// The page is made temporarily writable so the materializer can fill it, then dropped to
/// read-only so a later guest write still faults and is tracked. The read-watch owners are marked
/// materialized so the copy happens exactly once per armed window. Any failure — no materializer,
/// a failing producer, or an `mprotect` error — is reported and the fault is forwarded rather than
/// resumed over wrong bytes.
fn materialize_read_watch_page(
    snapshot: &super::snapshot::WatchSnapshot,
    page_address: u64,
    address: usize,
    length: usize,
    state: &impl Fn() -> FaultState,
) -> bool {
    let Some(materializer) = read_watch_materializer() else {
        // A read-watch cannot be armed without a materializer, so this is unreachable in practice;
        // forward rather than fabricate a materialized page.
        write_fault_diagnostic(state(), FaultReason::MaterializeFailed);
        return false;
    };
    // SAFETY: every owner registered this live mapping; this is the exact protected host page.
    if unsafe {
        libc::mprotect(
            address as *mut c_void,
            length,
            libc::PROT_READ | libc::PROT_WRITE,
        )
    } != 0
    {
        write_fault_diagnostic(state(), FaultReason::ProtectFailed);
        return false;
    }
    // SAFETY: the page is now writable and spans `[page_address, page_address + length)`; the
    // materializer's async-signal-safety is its registration contract.
    if !unsafe { materializer(page_address, length as u64) } {
        write_fault_diagnostic(state(), FaultReason::MaterializeFailed);
        return false;
    }
    // Leave the page readable but not writable: the read may resume now, and a subsequent guest
    // write faults again and is handled on the ordinary write path.
    // SAFETY: same live protected host page as above.
    if unsafe { libc::mprotect(address as *mut c_void, length, libc::PROT_READ) } != 0 {
        write_fault_diagnostic(state(), FaultReason::ProtectFailed);
        return false;
    }
    for owner in snapshot.owners(page_address) {
        // SAFETY: the active snapshot pins the range, and page_index was produced from that range.
        let range = unsafe { &*owner.range };
        if range.read_watch {
            let page = &range.pages[owner.page_index];
            page.materialized.store(true, Ordering::Release);
            // Disarm the page: it is now readable, so it will not fault again until the consumer
            // rearms. Only a rearm (which re-arms this page) re-applies no-access and re-stales it.
            // A materializing read is not a guest write, so it neither bumps the generation nor
            // dirties the watch.
            page.armed.store(false, Ordering::Release);
        }
    }
    true
}

/// Reports a fault the watch layer declined, with everything needed to identify its real cause.
///
/// Every process SIGSEGV/SIGBUS reaches this handler while CPU write watching is active, so this
/// line is normally the last output before the fault refaults under the previous action and
/// terminates the process. A report that only says the watch layer declined the fault cannot
/// distinguish a wild guest store from an emulator access to memory whose mapping went away.
fn write_fault_diagnostic((report, registers): FaultState, reason: FaultReason) {
    let mut buffer = DiagnosticBuffer::new();
    format_fault_report(report, reason, &mut buffer);
    if let Some(registers) = registers.as_ref() {
        format_fault_registers(registers, &mut buffer);
    }
    write_signal_diagnostic_bytes(buffer.as_slice());
    if let Some(registers) = registers.as_ref() {
        write_interrupted_stack_diagnostic(registers);
    }
    // A declined fault is normally the crash itself. When the journal is armed, name every
    // emulator publication that covers the faulting bytes before the process dies. The register
    // values anchor the dump: a poisoned dereference faults on the derived garbage pointer, while
    // the corrupted slot it was loaded from is still in the interrupted register file.
    let mut anchors = [0_u64; REGISTER_COUNT];
    let mut anchor_count = 0;
    if let Some(registers) = registers.as_ref() {
        for register in registers {
            // The stack pointers anchor thousands of legitimate HLE out-parameter writes near
            // the faulting thread's own frames; they can never name a heap victim.
            if register.name == b"rsp" || register.name == b"rbp" {
                continue;
            }
            anchors[anchor_count] = register.value;
            anchor_count += 1;
        }
    }
    crate::dump_guest_write_journal(report.address, &anchors[..anchor_count]);
}

/// Emits the mapped remainder of the interrupted stack page for offline symbolication.
///
/// The signal frame itself proves that bytes below the interrupted stack pointer are mapped, but
/// a pointer exactly at a page boundary may legally name an unmapped next page. Requiring a
/// nonzero in-page offset and never crossing that page keeps this last-chance read bounded. The
/// handler deliberately uses unaligned volatile reads because the corrupted frame may no longer
/// satisfy the ABI alignment rule.
fn write_interrupted_stack_diagnostic(registers: &[FaultRegister]) {
    const MAX_STACK_WORDS: usize = 48;
    let Some(stack_pointer) = registers
        .iter()
        .find(|register| register.name == b"rsp" || register.name == b"sp")
        .map(|register| register.value)
    else {
        return;
    };
    let Ok(stack_pointer_usize) = usize::try_from(stack_pointer) else {
        return;
    };
    let page_size = host_page_size_infallible() as usize;
    let in_page_offset = stack_pointer_usize & (page_size - 1);
    if in_page_offset == 0 {
        return;
    }
    let available_words = (page_size - in_page_offset) / std::mem::size_of::<u64>();
    let word_count = available_words.min(MAX_STACK_WORDS);
    if word_count == 0 {
        return;
    }
    let mut words = [0_u64; MAX_STACK_WORDS];
    for (index, word) in words[..word_count].iter_mut().enumerate() {
        // SAFETY: the interrupted stack pointer lies in the mapped stack page used to enter this
        // handler, and `word_count` is bounded to the remainder of that same page.
        *word = unsafe { std::ptr::read_volatile((stack_pointer_usize + index * 8) as *const u64) };
    }
    let mut buffer = DiagnosticBuffer::new();
    format_fault_stack_words(stack_pointer, &words[..word_count], &mut buffer);
    write_signal_diagnostic_bytes(buffer.as_slice());
}

/// Reads the interrupted instruction pointer and integer registers out of the machine context.
///
/// # Safety
///
/// `context` must be the `ucontext_t` pointer an `SA_SIGINFO` handler received, or null.
unsafe fn fault_machine_state(
    context: *mut c_void,
) -> (u64, Option<[FaultRegister; REGISTER_COUNT]>) {
    if context.is_null() {
        return (0, None);
    }
    // SAFETY: the caller guarantees this is the live signal ucontext_t.
    let machine_context = unsafe { (*context.cast::<UserContext>()).machine_context };
    if machine_context.is_null() {
        return (0, None);
    }
    // SAFETY: the kernel initializes the machine context for every SIGSEGV/SIGBUS delivery.
    let machine_context = unsafe { &*machine_context };
    (
        machine_context.program_counter(),
        Some(machine_context.registers()),
    )
}

/// Classifies a delivered fault as a read or a write from the machine context's exception state.
///
/// # Safety
///
/// `context` must be the `ucontext_t` pointer an `SA_SIGINFO` handler received, or null.
unsafe fn fault_access(context: *mut c_void) -> FaultAccess {
    if context.is_null() {
        return FaultAccess::Unknown;
    }
    // SAFETY: the caller guarantees this is the live signal ucontext_t.
    let machine_context = unsafe { (*context.cast::<UserContext>()).machine_context };
    if machine_context.is_null() {
        return FaultAccess::Unknown;
    }
    // SAFETY: the kernel initializes the machine context for every SIGSEGV/SIGBUS delivery.
    unsafe { &*machine_context }.access()
}

/// Darwin's `__darwin_ucontext`, truncated after the machine-context pointer this handler needs.
#[repr(C)]
struct UserContext {
    _on_stack: libc::c_int,
    _signal_mask: u32,
    _stack: libc::stack_t,
    _link: *mut UserContext,
    _machine_context_size: usize,
    machine_context: *const MachineContext,
}

/// Darwin's `_STRUCT_MCONTEXT64`, truncated after the integer thread state.
///
/// The trailing vector state is never read, so leaving it out of the declaration keeps the layout
/// dependency limited to the exception and integer register blocks.
#[cfg(target_arch = "x86_64")]
#[repr(C)]
struct MachineContext {
    _trap_number: u16,
    _cpu: u16,
    _error: u32,
    _fault_address: u64,
    rax: u64,
    rbx: u64,
    rcx: u64,
    rdx: u64,
    rdi: u64,
    rsi: u64,
    rbp: u64,
    rsp: u64,
    r8: u64,
    r9: u64,
    r10: u64,
    r11: u64,
    r12: u64,
    r13: u64,
    r14: u64,
    r15: u64,
    instruction_pointer: u64,
}

#[cfg(target_arch = "aarch64")]
#[repr(C)]
struct MachineContext {
    _fault_address: u64,
    _syndrome: u32,
    _exception: u32,
    general: [u64; 29],
    frame_pointer: u64,
    link_register: u64,
    stack_pointer: u64,
    instruction_pointer: u64,
}

/// Fixed register-file width, sized for the widest supported architecture.
#[cfg(target_arch = "x86_64")]
const REGISTER_COUNT: usize = 16;

#[cfg(target_arch = "aarch64")]
const REGISTER_COUNT: usize = 32;

impl MachineContext {
    const fn program_counter(&self) -> u64 {
        self.instruction_pointer
    }

    /// Whether the fault was a store, a load, or could not be classified.
    ///
    /// x86-64 encodes the direction in bit 1 of the page-fault error code; AArch64 encodes it in
    /// the `WnR` bit (6) of a data-abort syndrome. A syndrome that is not a data abort — or any
    /// state the direction cannot be read from — is `Unknown`, which the read-watch path treats
    /// conservatively as a read (materialize) so it never skips a materialization it owed.
    #[cfg(target_arch = "x86_64")]
    const fn access(&self) -> FaultAccess {
        if self._error & 0x2 != 0 {
            FaultAccess::Write
        } else {
            FaultAccess::Read
        }
    }

    #[cfg(target_arch = "aarch64")]
    const fn access(&self) -> FaultAccess {
        // ESR_EL1.EC == 0b100100 (lower EL) or 0b100101 (same EL) is a data abort; WnR is ISS bit 6.
        let exception_class = (self._syndrome >> 26) & 0x3f;
        if exception_class == 0b10_0100 || exception_class == 0b10_0101 {
            if self._syndrome & (1 << 6) != 0 {
                FaultAccess::Write
            } else {
                FaultAccess::Read
            }
        } else {
            FaultAccess::Unknown
        }
    }

    #[cfg(target_arch = "x86_64")]
    const fn registers(&self) -> [FaultRegister; REGISTER_COUNT] {
        const fn register(name: &'static [u8], value: u64) -> FaultRegister {
            FaultRegister { name, value }
        }
        [
            register(b"rax", self.rax),
            register(b"rbx", self.rbx),
            register(b"rcx", self.rcx),
            register(b"rdx", self.rdx),
            register(b"rdi", self.rdi),
            register(b"rsi", self.rsi),
            register(b"rbp", self.rbp),
            register(b"rsp", self.rsp),
            register(b"r8", self.r8),
            register(b"r9", self.r9),
            register(b"r10", self.r10),
            register(b"r11", self.r11),
            register(b"r12", self.r12),
            register(b"r13", self.r13),
            register(b"r14", self.r14),
            register(b"r15", self.r15),
        ]
    }

    #[cfg(target_arch = "aarch64")]
    fn registers(&self) -> [FaultRegister; REGISTER_COUNT] {
        const NAMES: [&[u8]; 29] = [
            b"x0", b"x1", b"x2", b"x3", b"x4", b"x5", b"x6", b"x7", b"x8", b"x9", b"x10", b"x11",
            b"x12", b"x13", b"x14", b"x15", b"x16", b"x17", b"x18", b"x19", b"x20", b"x21", b"x22",
            b"x23", b"x24", b"x25", b"x26", b"x27", b"x28",
        ];
        let mut registers = [FaultRegister {
            name: b"",
            value: 0,
        }; REGISTER_COUNT];
        for (index, name) in NAMES.iter().enumerate() {
            registers[index] = FaultRegister {
                name,
                value: self.general[index],
            };
        }
        registers[29] = FaultRegister {
            name: b"fp",
            value: self.frame_pointer,
        };
        registers[30] = FaultRegister {
            name: b"lr",
            value: self.link_register,
        };
        registers[31] = FaultRegister {
            name: b"sp",
            value: self.stack_pointer,
        };
        registers
    }
}

fn write_signal_diagnostic_bytes(message: &[u8]) {
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

/// # Safety
///
/// Callable only from the installed signal handler, forwarding the exact `signal`, `information`
/// and `context` the kernel delivered; the previous-action slots it reads were initialized by
/// `install_fault_handlers_once` before the handler became reachable.
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
