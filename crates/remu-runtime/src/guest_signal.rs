//! Guest signal / exception delivery substrate.
//!
//! Orbis/Prospero expose `sceKernelInstallExceptionHandler` /
//! `sceKernelRaiseException` rather than raw POSIX `sigaction`/`pthread_kill` to
//! titles. Unity's Boehm GC stop-the-world path installs SIGUSR1 (30) and raises
//! it against every worker; the handler must observe a full Orbis `ucontext_t`
//! (mcontext at `+0x40`) so register roots are collected. Delivery must interrupt
//! spinning guest threads, not only poll at HLE import boundaries.
//!
//! On Darwin under Rosetta the host delivers a real `ucontext_t` whose
//! `__darwin_mcontext64` integer state is translated into that Orbis layout
//! (SharpEmu / KytyPS5 reference shape). Host `SIGUSR1` is the interrupt vehicle
//! for `sceKernelRaiseException`; guest exception type 30 is carried in the
//! pending queue, not as the host signal number.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::{
    ImportCallFrame, SCE_KERNEL_ERROR_INVALID_ARGUMENT, current_guest_thread_handle,
    current_guest_thread_pointer, low_i32, read_guest_u64, sce_kernel_error, write_guest_u64,
};

/// Orbis `ucontext_t` size Unity / libScePs5Util walk for GC roots.
pub(crate) const ORBIS_UCONTEXT_BYTES: usize = 0x500;
/// Bytes before the amd64 `mcontext_t` (sigmask + private header).
pub(crate) const ORBIS_MCONTEXT_OFFSET: usize = 0x40;
/// `sizeof(Orbis mcontext_t)` written at `mcontext + 0xc8`.
const ORBIS_MCONTEXT_LEN: u64 = 0x480;

#[repr(C, align(16))]
pub(crate) struct GuestExceptionContext(pub(crate) [u8; ORBIS_UCONTEXT_BYTES]);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PendingGuestException {
    pub(crate) handler: u64,
    pub(crate) exception: i32,
}

#[derive(Debug, Default)]
pub(crate) struct GuestExceptionDeliveryState {
    pub(crate) pending: BTreeMap<u64, PendingGuestException>,
    pub(crate) active: std::collections::BTreeSet<u64>,
}

static GUEST_EXCEPTION_DELIVERY: OnceLock<Mutex<GuestExceptionDeliveryState>> = OnceLock::new();
static GUEST_EXCEPTION_PENDING: AtomicBool = AtomicBool::new(false);

/// Process-wide guest exception handlers, indexed by POSIX signal number.
///
/// Mirrored out of `ProbeDispatchState` so the fault / delivery signal path can
/// resolve a handler without taking the dispatch lock (not async-signal-safe).
static INSTALLED_GUEST_HANDLERS: [AtomicU64; 128] = [const { AtomicU64::new(0) }; 128];

/// Host `pthread_t` for each live guest thread handle (truncated to `u64`).
static HOST_PTHREADS: OnceLock<Mutex<BTreeMap<u64, u64>>> = OnceLock::new();

/// Per-thread guest signal mask bits (Orbis `_sigprocmask`).
static GUEST_SIGNAL_MASK: AtomicU64 = AtomicU64::new(0);

fn host_pthreads() -> &'static Mutex<BTreeMap<u64, u64>> {
    HOST_PTHREADS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

pub(crate) fn guest_exception_delivery() -> &'static Mutex<GuestExceptionDeliveryState> {
    GUEST_EXCEPTION_DELIVERY.get_or_init(|| Mutex::new(GuestExceptionDeliveryState::default()))
}

pub(crate) fn guest_exception_is_pending(thread: u64) -> bool {
    thread != 0
        && guest_exception_delivery()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending
            .contains_key(&thread)
}

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) fn guest_exception_pending_flag() -> &'static AtomicBool {
    &GUEST_EXCEPTION_PENDING
}

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) fn clear_installed_guest_handlers() {
    for slot in &INSTALLED_GUEST_HANDLERS {
        slot.store(0, Ordering::Release);
    }
}

pub(crate) fn publish_installed_guest_handler(exception: i32, handler: u64) {
    if (0..128).contains(&exception) {
        INSTALLED_GUEST_HANDLERS[exception as usize].store(handler, Ordering::Release);
    }
}

pub(crate) fn clear_installed_guest_handler(exception: i32) {
    if (0..128).contains(&exception) {
        INSTALLED_GUEST_HANDLERS[exception as usize].store(0, Ordering::Release);
    }
}

#[cfg_attr(
    not(all(target_os = "macos", target_arch = "x86_64")),
    allow(dead_code)
)]
fn installed_guest_handler(exception: i32) -> u64 {
    if (0..128).contains(&exception) {
        INSTALLED_GUEST_HANDLERS[exception as usize].load(Ordering::Acquire)
    } else {
        0
    }
}

/// Records the calling host pthread as the native identity of `guest_handle`.
pub(crate) fn register_current_host_pthread(guest_handle: u64) {
    if guest_handle == 0 {
        return;
    }
    // SAFETY: `pthread_self` returns the calling thread's valid pthread id.
    let host = unsafe { libc::pthread_self() } as u64;
    host_pthreads()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(guest_handle, host);
}

pub(crate) fn unregister_host_pthread(guest_handle: u64) {
    host_pthreads()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&guest_handle);
}

fn host_pthread_for(guest_handle: u64) -> Option<libc::pthread_t> {
    host_pthreads()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&guest_handle)
        .copied()
        .map(|value| value as libc::pthread_t)
}

pub(crate) fn write_exception_context_word(
    context: &mut GuestExceptionContext,
    offset: usize,
    value: u64,
) {
    context.0[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg_attr(
    not(all(target_os = "macos", target_arch = "x86_64")),
    allow(dead_code)
)]
fn read_exception_context_word(context: &GuestExceptionContext, offset: usize) -> u64 {
    u64::from_le_bytes(context.0[offset..offset + 8].try_into().unwrap())
}

/// Fills an Orbis `ucontext_t` from an import-boundary call frame (cooperative path).
pub(crate) fn fill_orbis_ucontext_from_import(
    context: &mut GuestExceptionContext,
    frame: &ImportCallFrame,
) {
    const M: usize = ORBIS_MCONTEXT_OFFSET;
    for (offset, value) in [
        (0x08, frame.integer_arguments[0]),
        (0x10, frame.integer_arguments[1]),
        (0x18, frame.integer_arguments[2]),
        (0x20, frame.integer_arguments[3]),
        (0x28, frame.integer_arguments[4]),
        (0x30, frame.integer_arguments[5]),
        (0x38, frame.rax),
        (0x40, frame.guest_rbx),
    ] {
        write_exception_context_word(context, M + offset, value);
    }
    write_exception_context_word(context, M + 0xa0, frame.guest_return_address);
    write_exception_context_word(context, M + 0xb8, frame.guest_rsp);
    write_exception_context_word(context, M + 0xc8, ORBIS_MCONTEXT_LEN);
    write_exception_context_word(context, M + 0x440, current_guest_thread_pointer());
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
mod darwin_x86 {
    use super::{
        GuestExceptionContext, ORBIS_MCONTEXT_LEN, ORBIS_MCONTEXT_OFFSET,
        write_exception_context_word,
    };
    use std::ffi::c_void;

    /// Darwin `__darwin_ucontext` truncated after the machine-context pointer.
    #[repr(C)]
    pub(super) struct UserContext {
        _on_stack: libc::c_int,
        _signal_mask: u32,
        _stack: libc::stack_t,
        _link: *mut UserContext,
        _machine_context_size: usize,
        machine_context: *mut MachineContext,
    }

    /// Darwin `_STRUCT_MCONTEXT64` integer thread state (Rosetta x86-64).
    #[repr(C)]
    pub(super) struct MachineContext {
        trap_number: u16,
        cpu: u16,
        error: u32,
        fault_address: u64,
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
        rip: u64,
        rflags: u64,
        cs: u64,
        fs: u64,
        gs: u64,
    }

    pub(super) unsafe fn fill_orbis_from_darwin(
        context: &mut GuestExceptionContext,
        darwin: *mut c_void,
        fs_base: u64,
    ) -> bool {
        if darwin.is_null() {
            return false;
        }
        // SAFETY: caller supplies the live SA_SIGINFO ucontext.
        let user = unsafe { &*darwin.cast::<UserContext>() };
        if user.machine_context.is_null() {
            return false;
        }
        // SAFETY: kernel-initialized mcontext for this delivery.
        let machine = unsafe { &*user.machine_context };
        const M: usize = ORBIS_MCONTEXT_OFFSET;
        // Orbis mcontext register order matches FreeBSD amd64 / SharpEmu.
        write_exception_context_word(context, M + 0x08, machine.rdi);
        write_exception_context_word(context, M + 0x10, machine.rsi);
        write_exception_context_word(context, M + 0x18, machine.rdx);
        write_exception_context_word(context, M + 0x20, machine.rcx);
        write_exception_context_word(context, M + 0x28, machine.r8);
        write_exception_context_word(context, M + 0x30, machine.r9);
        write_exception_context_word(context, M + 0x38, machine.rax);
        write_exception_context_word(context, M + 0x40, machine.rbx);
        write_exception_context_word(context, M + 0x48, machine.rbp);
        write_exception_context_word(context, M + 0x50, machine.r10);
        write_exception_context_word(context, M + 0x58, machine.r11);
        write_exception_context_word(context, M + 0x60, machine.r12);
        write_exception_context_word(context, M + 0x68, machine.r13);
        write_exception_context_word(context, M + 0x70, machine.r14);
        write_exception_context_word(context, M + 0x78, machine.r15);
        write_exception_context_word(context, M + 0x80, u64::from(machine.trap_number));
        write_exception_context_word(context, M + 0x88, machine.fault_address);
        write_exception_context_word(context, M + 0x98, u64::from(machine.error));
        write_exception_context_word(context, M + 0xa0, machine.rip);
        write_exception_context_word(context, M + 0xa8, machine.cs);
        write_exception_context_word(context, M + 0xb0, machine.rflags);
        write_exception_context_word(context, M + 0xb8, machine.rsp);
        write_exception_context_word(context, M + 0xc8, ORBIS_MCONTEXT_LEN);
        write_exception_context_word(context, M + 0x440, fs_base);
        let _ = (machine.cpu, machine.fs, machine.gs);
        true
    }

    pub(super) unsafe fn apply_orbis_to_darwin(
        context: &GuestExceptionContext,
        darwin: *mut c_void,
    ) {
        use super::read_exception_context_word;
        if darwin.is_null() {
            return;
        }
        // SAFETY: same live ucontext the handler received.
        let user = unsafe { &mut *darwin.cast::<UserContext>() };
        if user.machine_context.is_null() {
            return;
        }
        // SAFETY: writable mcontext for resume.
        let machine = unsafe { &mut *user.machine_context };
        const M: usize = ORBIS_MCONTEXT_OFFSET;
        machine.rdi = read_exception_context_word(context, M + 0x08);
        machine.rsi = read_exception_context_word(context, M + 0x10);
        machine.rdx = read_exception_context_word(context, M + 0x18);
        machine.rcx = read_exception_context_word(context, M + 0x20);
        machine.r8 = read_exception_context_word(context, M + 0x28);
        machine.r9 = read_exception_context_word(context, M + 0x30);
        machine.rax = read_exception_context_word(context, M + 0x38);
        machine.rbx = read_exception_context_word(context, M + 0x40);
        machine.rbp = read_exception_context_word(context, M + 0x48);
        machine.r10 = read_exception_context_word(context, M + 0x50);
        machine.r11 = read_exception_context_word(context, M + 0x58);
        machine.r12 = read_exception_context_word(context, M + 0x60);
        machine.r13 = read_exception_context_word(context, M + 0x68);
        machine.r14 = read_exception_context_word(context, M + 0x70);
        machine.r15 = read_exception_context_word(context, M + 0x78);
        machine.rip = read_exception_context_word(context, M + 0xa0);
        machine.rflags = read_exception_context_word(context, M + 0xb0);
        machine.rsp = read_exception_context_word(context, M + 0xb8);
    }
}

fn invoke_guest_exception_handler(
    exception: i32,
    handler: u64,
    context: &mut GuestExceptionContext,
) {
    type GuestExceptionHandler = unsafe extern "C" fn(i32, *mut u8);
    // SAFETY: handler is a mapped guest executable installed via InstallExceptionHandler.
    let handler: GuestExceptionHandler = unsafe { std::mem::transmute(handler) };
    // SAFETY: context is live for the synchronous invocation.
    unsafe { handler(exception, context.0.as_mut_ptr()) };
}

/// Delivers a pending cooperative exception at an HLE import boundary.
pub(crate) fn deliver_pending_guest_exception_at_import(frame: &ImportCallFrame) {
    let thread = current_guest_thread_handle();
    if thread == 0 || (!cfg!(test) && !GUEST_EXCEPTION_PENDING.load(Ordering::Acquire)) {
        return;
    }
    let pending = {
        let mut delivery = guest_exception_delivery()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if delivery.active.contains(&thread) {
            return;
        }
        let Some(pending) = delivery.pending.remove(&thread) else {
            return;
        };
        GUEST_EXCEPTION_PENDING.store(!delivery.pending.is_empty(), Ordering::Release);
        delivery.active.insert(thread);
        pending
    };

    let mut context = GuestExceptionContext([0; ORBIS_UCONTEXT_BYTES]);
    fill_orbis_ucontext_from_import(&mut context, frame);
    eprintln!(
        "[REMu][KERNEL] delivering cooperative exception target={thread:#x} type={} handler={:#x} rip={:#x} rsp={:#x}",
        pending.exception, pending.handler, frame.guest_return_address, frame.guest_rsp,
    );
    invoke_guest_exception_handler(pending.exception, pending.handler, &mut context);

    guest_exception_delivery()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .active
        .remove(&thread);
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn take_pending_for_current_thread() -> Option<PendingGuestException> {
    let thread = current_guest_thread_handle();
    if thread == 0 {
        return None;
    }
    let Ok(mut delivery) = guest_exception_delivery().try_lock() else {
        return None;
    };
    if delivery.active.contains(&thread) {
        return None;
    }
    let pending = delivery.pending.remove(&thread)?;
    GUEST_EXCEPTION_PENDING.store(!delivery.pending.is_empty(), Ordering::Release);
    delivery.active.insert(thread);
    Some(pending)
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn finish_active_delivery() {
    let thread = current_guest_thread_handle();
    if thread == 0 {
        return;
    }
    if let Ok(mut delivery) = guest_exception_delivery().try_lock() {
        delivery.active.remove(&thread);
    }
}

#[cfg_attr(
    not(all(target_os = "macos", target_arch = "x86_64")),
    allow(dead_code)
)]
fn guest_signal_is_blocked(exception: i32) -> bool {
    if !(0..64).contains(&exception) {
        return false;
    }
    GUEST_SIGNAL_MASK.load(Ordering::Acquire) & (1_u64 << exception) != 0
}

/// Host SIGUSR1 handler: interruptive delivery for `sceKernelRaiseException`.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn host_delivery_signal_handler(
    _signal: libc::c_int,
    _information: *mut libc::siginfo_t,
    context: *mut c_void,
) {
    let Some(pending) = take_pending_for_current_thread() else {
        return;
    };
    if guest_signal_is_blocked(pending.exception) {
        // Re-queue: the guest asked to block this signal.
        let thread = current_guest_thread_handle();
        if let Ok(mut delivery) = guest_exception_delivery().try_lock() {
            delivery.pending.entry(thread).or_insert(pending);
            delivery.active.remove(&thread);
            GUEST_EXCEPTION_PENDING.store(true, Ordering::Release);
        }
        return;
    }

    let mut guest_context = GuestExceptionContext([0; ORBIS_UCONTEXT_BYTES]);
    // SAFETY: SA_SIGINFO supplies a live Darwin ucontext for this delivery.
    let translated = unsafe {
        darwin_x86::fill_orbis_from_darwin(
            &mut guest_context,
            context,
            current_guest_thread_pointer(),
        )
    };
    if !translated {
        finish_active_delivery();
        return;
    }

    eprintln!(
        "[REMu][KERNEL] delivering interruptive exception type={} handler={:#x}",
        pending.exception, pending.handler
    );
    invoke_guest_exception_handler(pending.exception, pending.handler, &mut guest_context);

    // SAFETY: same live Darwin ucontext; guest may have rewritten registers.
    unsafe { darwin_x86::apply_orbis_to_darwin(&guest_context, context) };
    finish_active_delivery();
}

/// Native arm64 builds never execute guest code; keep the delivery vehicle as a loud no-op.
#[cfg(all(target_os = "macos", not(target_arch = "x86_64")))]
#[allow(dead_code)] // installed from x86_64-only probe entry; kept for arm64 check completeness
extern "C" fn host_delivery_signal_handler(
    _signal: libc::c_int,
    _information: *mut libc::siginfo_t,
    _context: *mut c_void,
) {
    let Some(pending) = take_pending_for_current_thread() else {
        return;
    };
    eprintln!(
        "[REMu][KERNEL] REFUSAL: interruptive exception type={} requires Rosetta x86_64 host context",
        pending.exception
    );
    finish_active_delivery();
}

/// Installs the host SIGUSR1 delivery vehicle once.
#[cfg(target_os = "macos")]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) fn install_guest_signal_delivery() {
    static INSTALLED: AtomicBool = AtomicBool::new(false);
    if INSTALLED.swap(true, Ordering::AcqRel) {
        return;
    }
    // SAFETY: zeroed sigaction is a valid base before fields are filled.
    let mut action = unsafe { std::mem::zeroed::<libc::sigaction>() };
    action.sa_sigaction = host_delivery_signal_handler as *const () as usize;
    action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK | libc::SA_RESTART;
    // SAFETY: sa_mask is live writable storage.
    unsafe { libc::sigemptyset(&raw mut action.sa_mask) };
    // SAFETY: install process-wide delivery handler; previous action discarded (default).
    if unsafe { libc::sigaction(libc::SIGUSR1, &raw const action, std::ptr::null_mut()) } != 0 {
        eprintln!(
            "[REMu][KERNEL] REFUSAL: host SIGUSR1 delivery handler unavailable: {}",
            std::io::Error::last_os_error()
        );
        INSTALLED.store(false, Ordering::Release);
    }
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn install_guest_signal_delivery() {}

/// Tries to deliver a real host fault to an installed guest exception handler.
///
/// Called from the write-watch SIGSEGV/SIGBUS path after recovery declines. Returns
/// true when the guest handler ran (fault consumed).
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(crate) unsafe extern "C" fn try_deliver_guest_fault(
    signal: i32,
    _fault_address: u64,
    context: *mut c_void,
) -> bool {
    let handler = installed_guest_handler(signal);
    if handler == 0 || guest_signal_is_blocked(signal) {
        return false;
    }
    let thread = current_guest_thread_handle();
    if thread == 0 {
        return false;
    }
    let mut guest_context = GuestExceptionContext([0; ORBIS_UCONTEXT_BYTES]);
    // SAFETY: live Darwin fault ucontext from SA_SIGINFO.
    if !unsafe {
        darwin_x86::fill_orbis_from_darwin(
            &mut guest_context,
            context,
            current_guest_thread_pointer(),
        )
    } {
        return false;
    }
    eprintln!(
        "[REMu][KERNEL] delivering fault exception type={signal} handler={handler:#x} to guest"
    );
    invoke_guest_exception_handler(signal, handler, &mut guest_context);
    // SAFETY: write guest-updated registers back for resume.
    unsafe { darwin_x86::apply_orbis_to_darwin(&guest_context, context) };
    true
}

/// Queues a guest exception and interrupts the target host thread when possible.
pub(crate) fn queue_and_interrupt_guest_exception(
    target: u64,
    exception: i32,
    handler: u64,
    target_name: Option<&str>,
) -> u64 {
    if target == 0 || !(0..128).contains(&exception) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let source = current_guest_thread_handle();
    guest_exception_delivery()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .pending
        .entry(target)
        .or_insert(PendingGuestException { handler, exception });
    let sync = crate::guest_sync();
    let _sync_guard = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    GUEST_EXCEPTION_PENDING.store(true, Ordering::Release);
    drop(_sync_guard);
    crate::guest_sync::wake_all_guest_sync_for_exception();
    crate::notify_kernel_equeue_waiters();

    let self_target = target == source && source != 0;
    #[cfg(target_os = "macos")]
    {
        if self_target {
            eprintln!(
                "[REMu][KERNEL] queued self exception source={source:#x} type={exception} handler={handler:#x} (cooperative at return)"
            );
        } else if let Some(host) = host_pthread_for(target) {
            // SAFETY: host pthread was registered at guest thread start and remains live while
            // the guest handle is in `pthread_threads`.
            let result = unsafe { libc::pthread_kill(host, libc::SIGUSR1) };
            if result != 0 {
                eprintln!(
                    "[REMu][KERNEL] REFUSAL: pthread_kill failed target={target:#x} errno={result}; falling back to cooperative delivery"
                );
            } else {
                eprintln!(
                    "[REMu][KERNEL] interruptive exception source={source:#x} target={target:#x} target_name={:?} type={exception} handler={handler:#x}",
                    target_name.unwrap_or("unknown")
                );
            }
        } else {
            eprintln!(
                "[REMu][KERNEL] queued cooperative exception source={source:#x} target={target:#x} target_name={:?} type={exception} handler={handler:#x} (no host pthread)",
                target_name.unwrap_or("unknown")
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (self_target, target_name);
        eprintln!(
            "[REMu][KERNEL] queued cooperative exception source={source:#x} target={target:#x} type={exception} handler={handler:#x}"
        );
    }
    0
}

/// `_sigprocmask(how, set, oldset)` — guest-visible mask with honest oldset writeback.
pub(crate) fn guest_sigprocmask(arguments: [u64; 6]) -> u64 {
    let how = low_i32(arguments[0]);
    let set = arguments[1];
    let oldset = arguments[2];
    let previous = GUEST_SIGNAL_MASK.load(Ordering::Acquire);
    if oldset != 0 {
        // Orbis sigset_t is a single u64 bitset for the signals titles use.
        write_guest_u64(oldset, previous);
    }
    if set == 0 {
        return 0;
    }
    let bits = read_guest_u64(set);
    let next = match how {
        1 => previous | bits,  // SIG_BLOCK
        2 => previous & !bits, // SIG_UNBLOCK
        3 => bits,             // SIG_SETMASK
        _ => {
            return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
        }
    };
    GUEST_SIGNAL_MASK.store(next, Ordering::Release);
    0
}

/// Host-priority mapping for console FIFO priorities (256 urgent … 767 idle).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SchedulingBand {
    Realtime,
    Interactive,
    Normal,
    Background,
}

impl SchedulingBand {
    pub(crate) fn from_console_priority(priority: i32) -> Self {
        match priority {
            ..=399 => Self::Realtime,
            400..=599 => Self::Interactive,
            600..=699 => Self::Normal,
            _ => Self::Background,
        }
    }
}

/// Applies a console priority to the calling host thread, or logs a loud refusal.
pub(crate) fn apply_host_priority(priority: i32) {
    let band = SchedulingBand::from_console_priority(priority);
    #[cfg(target_os = "macos")]
    {
        // Darwin QoS classes approximate the console bands. Realtime FIFO 256–399 maps to
        // user-interactive; there is no unprivileged true realtime class on macOS.
        let qos = match band {
            SchedulingBand::Realtime | SchedulingBand::Interactive => {
                libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE
            }
            SchedulingBand::Normal => libc::qos_class_t::QOS_CLASS_USER_INITIATED,
            SchedulingBand::Background => libc::qos_class_t::QOS_CLASS_UTILITY,
        };
        // SAFETY: pthread_set_qos_class_self_np adjusts the calling thread only.
        let result = unsafe { libc::pthread_set_qos_class_self_np(qos, 0) };
        if result != 0 {
            static REFUSED: AtomicBool = AtomicBool::new(false);
            if !REFUSED.swap(true, Ordering::AcqRel) {
                eprintln!(
                    "[REMu][KERNEL] REFUSAL: pthread_set_qos_class_self_np failed errno={result}; guest priorities recorded but not enforced on host"
                );
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        static REFUSED: AtomicBool = AtomicBool::new(false);
        if !REFUSED.swap(true, Ordering::AcqRel) {
            eprintln!(
                "[REMu][KERNEL] REFUSAL: host priority band {band:?} (console prio={priority}) is not enforced on this OS"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbis_ucontext_import_layout_places_rsp_at_0xf8() {
        let mut context = GuestExceptionContext([0; ORBIS_UCONTEXT_BYTES]);
        let frame = ImportCallFrame {
            guest_rsp: 0x7000_0000_1000,
            guest_return_address: 0x8000_0000_2000,
            rax: 0x11,
            guest_rbx: 0x22,
            integer_arguments: [1, 2, 3, 4, 5, 6],
            ..ImportCallFrame::default()
        };
        fill_orbis_ucontext_from_import(&mut context, &frame);
        assert_eq!(
            read_exception_context_word(&context, 0xf8),
            0x7000_0000_1000
        );
        assert_eq!(
            read_exception_context_word(&context, 0xe0),
            0x8000_0000_2000
        );
        assert_eq!(
            read_exception_context_word(&context, ORBIS_MCONTEXT_OFFSET + 0xc8),
            ORBIS_MCONTEXT_LEN
        );
    }

    #[test]
    fn sigprocmask_writes_oldset_and_updates_mask() {
        GUEST_SIGNAL_MASK.store(0, Ordering::SeqCst);
        let mut old = 0_u64;
        let mut set = 1_u64 << 30;
        assert_eq!(
            guest_sigprocmask([
                3,
                (&raw mut set).addr() as u64,
                (&raw mut old).addr() as u64,
                0,
                0,
                0
            ]),
            0
        );
        assert_eq!(old, 0);
        assert_eq!(GUEST_SIGNAL_MASK.load(Ordering::SeqCst), 1_u64 << 30);
        let mut old2 = 0_u64;
        assert_eq!(
            guest_sigprocmask([3, 0, (&raw mut old2).addr() as u64, 0, 0, 0]),
            0
        );
        assert_eq!(old2, 1_u64 << 30);
        GUEST_SIGNAL_MASK.store(0, Ordering::SeqCst);
    }

    #[test]
    fn scheduling_bands_follow_console_priority_ranges() {
        assert_eq!(
            SchedulingBand::from_console_priority(256),
            SchedulingBand::Realtime
        );
        assert_eq!(
            SchedulingBand::from_console_priority(500),
            SchedulingBand::Interactive
        );
        assert_eq!(
            SchedulingBand::from_console_priority(650),
            SchedulingBand::Normal
        );
        assert_eq!(
            SchedulingBand::from_console_priority(700),
            SchedulingBand::Background
        );
    }
}
