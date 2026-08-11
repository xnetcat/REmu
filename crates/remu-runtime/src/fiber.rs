// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! The guest fiber ABI (`libSceFiber`).
//!
//! A fiber is cooperative: it runs on the thread that started it, keeps its own stack, and yields
//! only by naming who runs next. Engines build job systems on it, so the entry points here run
//! guest code directly and must not be reached while the process dispatch lock is held — a fiber
//! calls further imports as soon as it starts. They are therefore dispatched on the stateless path.
//!
//! The fiber object itself is guest storage: the guest allocates it and passes its address, so its
//! fields are read and written through the guest accessors rather than held host-side. The only
//! host-side state is the per-thread record of which fiber is currently running.
//!
//! A fiber does *not* stay on one thread. An observed job system parks a fiber on one worker and
//! lets whichever worker picks the job up next resume it, so a suspended frame routinely continues
//! on a thread that did not suspend it. Everything a resumed frame needs must therefore be read
//! fresh after the resume rather than carried across it — see [`fiber_context`], where getting
//! this wrong left the resuming thread reading the suspending thread's already-cleared slot.

mod switch;

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::{
    current_guest_thread_name, guest_memset, read_guest_cstring, read_guest_u32, read_guest_u64,
    trace_fiber, write_guest_u32, write_guest_u64,
};

/// Logs one fiber transition when `REMU_TRACE_FIBER=1`.
///
/// Fiber calls repeat far too often for the general first-hit import log to show what a job
/// system is actually doing, and a refused call is invisible there because it still returns.
fn trace(call: &str, fiber: u64, result: u64) {
    if trace_fiber() {
        eprintln!(
            "[REMu][FIBER] {call} fiber={fiber:#x} result={result:#x} current={:#x} context={:?} slot={:?} host={:?} thread={:?}",
            current_fiber_identity(),
            fiber_context(),
            THREAD_FIBER_CONTEXT.with(std::ptr::from_ref),
            host_thread_id(),
            current_guest_thread_name()
        );
    }
}

fn host_thread_id() -> u64 {
    #[cfg(unix)]
    {
        // SAFETY: `pthread_self` is always valid on the calling thread.
        unsafe { libc::pthread_self() as usize as u64 }
    }
    #[cfg(windows)]
    {
        // SAFETY: GetCurrentThreadId has no preconditions.
        u64::from(unsafe { windows_get_current_thread_id() })
    }
    #[cfg(not(any(unix, windows)))]
    {
        0
    }
}

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    #[link_name = "GetCurrentThreadId"]
    fn windows_get_current_thread_id() -> u32;
}

const SCE_FIBER_ERROR_NULL: u32 = 0x8059_0001;
const SCE_FIBER_ERROR_ALIGNMENT: u32 = 0x8059_0002;
const SCE_FIBER_ERROR_RANGE: u32 = 0x8059_0003;
const SCE_FIBER_ERROR_INVALID: u32 = 0x8059_0004;
const SCE_FIBER_ERROR_PERMISSION: u32 = 0x8059_0005;
const SCE_FIBER_ERROR_STATE: u32 = 0x8059_0006;

/// Paired guards the guest object carries so an uninitialized or overwritten fiber is rejected
/// rather than jumped into.
const FIBER_MAGIC_START: u32 = 0xdef1_649c;
const FIBER_MAGIC_END: u32 = 0xb375_92a0;
const FIBER_OPT_PARAM_MAGIC: u32 = 0xbb40_e64d;
const SCE_FIBER_OPT_PARAM_SIZE: usize = 128;
const SCE_FIBER_OPT_PARAM_DEBUG_OPTION_CHECK_CONTEXT_SIZE: u32 = 1;
/// Written to the lowest word of a fiber's stack so an overflow that walks past the limit is
/// detectable at the next switch instead of silently corrupting neighbouring guest storage.
const FIBER_STACK_GUARD: u64 = 0x7149_f2ca_7149_f2ca;

const FIBER_STATE_RUN: u32 = 1;
const FIBER_STATE_IDLE: u32 = 2;
const FIBER_STATE_TERMINATED: u32 = 3;

const FIBER_MAX_NAME_LENGTH: usize = 31;
const FIBER_CONTEXT_MINIMUM_SIZE: u64 = 512;
const FIBER_INFO_SIZE: u64 = 128;

/// Guest fiber object field offsets. The guest allocates the storage and only this module
/// interprets it, but the layout follows the console library so a title that inspects or copies
/// the object sees what it expects.
mod offset {
    pub(super) const MAGIC_START: u64 = 0x00;
    pub(super) const STATE: u64 = 0x04;
    pub(super) const ENTRY: u64 = 0x08;
    pub(super) const ARG_ON_INITIALIZE: u64 = 0x10;
    pub(super) const ADDR_CONTEXT: u64 = 0x18;
    pub(super) const SIZE_CONTEXT: u64 = 0x20;
    pub(super) const NAME: u64 = 0x28;
    pub(super) const CONTEXT: u64 = 0x48;
    pub(super) const FLAGS: u64 = 0x50;
    pub(super) const MAGIC_END: u64 = 0x68;
    /// Host-owned resumable register storage inside the SDK's opaque 256-byte object.
    pub(super) const SAVED_CONTEXT: u64 = 0x70;
}

/// The register state required to resume a suspended SysV call.
///
/// Only callee-saved registers belong here. Caller-saved registers are allowed to change across
/// the `fiber_set_jmp` call, and keeping the record compact lets every suspended fiber own stable
/// storage inside its opaque 256-byte SDK object instead of leaving the record on a stack that a
/// migrating job system can reuse on another worker.
#[repr(C)]
#[derive(Default)]
pub(crate) struct FiberJumpContext {
    rbx: u64,
    rsp: u64,
    rbp: u64,
    r12: u64,
    r13: u64,
    r14: u64,
    r15: u64,
    return_address: u64,
    mxcsr: u32,
    floating_point_control: u16,
    _padding: u16,
}

/// The saved `sceFiberRun` frame plus bookkeeping shared by every fiber in that run chain.
#[repr(C)]
#[derive(Default)]
pub(crate) struct FiberContext {
    jump: FiberJumpContext,
    current_fiber: u64,
    previous_fiber: u64,
    argument_on_run_to: u64,
    argument_on_return: u64,
    entry_returned: u64,
}

/// The handoff record [`switch::fiber_switch_entry`] consumes when starting a fiber.
#[repr(C)]
#[derive(Default)]
pub(crate) struct FiberData {
    entry: u64,
    argument_on_initialize: u64,
    argument_on_run_to: u64,
    stack_address: u64,
    previous_state: u64,
    mxcsr: u32,
    floating_point_control: u16,
    _padding: u16,
}

const _: () = {
    assert!(std::mem::offset_of!(FiberJumpContext, mxcsr) == 0x40);
    assert!(std::mem::offset_of!(FiberJumpContext, floating_point_control) == 0x44);
    assert!(std::mem::size_of::<FiberJumpContext>() == 0x48);
    assert!(std::mem::offset_of!(FiberContext, current_fiber) == 0x48);
    assert!(std::mem::size_of::<FiberContext>() == 0x70);
    assert!(offset::SAVED_CONTEXT as usize + std::mem::size_of::<FiberJumpContext>() <= 256);
    assert!(std::mem::offset_of!(FiberData, stack_address) == 0x18);
    assert!(std::mem::offset_of!(FiberData, previous_state) == 0x20);
    assert!(std::mem::offset_of!(FiberData, mxcsr) == 0x28);
    assert!(std::mem::offset_of!(FiberData, floating_point_control) == 0x2c);
};

/// Returns the stable register record owned by `fiber`'s opaque SDK object.
fn saved_context(fiber: u64) -> *mut FiberJumpContext {
    (fiber + offset::SAVED_CONTEXT) as *mut FiberJumpContext
}

/// The guest-visible publication word for a suspended fiber's host-owned register record.
fn saved_context_publication(fiber: u64) -> &'static AtomicU64 {
    // SAFETY: every validated fiber is eight-byte aligned, and CONTEXT is an aligned u64 field in
    // its live 256-byte object. The object outlives every access while its state is not terminated.
    unsafe { AtomicU64::from_ptr((fiber + offset::CONTEXT) as *mut u64) }
}

/// Validates and atomically takes the opaque resume record before handing it to naked assembly.
///
/// A copied or overwritten fiber object must fail as an SDK error, not turn an arbitrary guest
/// pointer into the host stack pointer. The saved call frame itself must remain inside the fiber's
/// declared context allocation. Taking the publication here also closes the validation/use gap:
/// the transfer must jump through this exact validated pointer, never a second guest-memory read.
fn take_saved_context(fiber: u64) -> Result<Option<*mut FiberJumpContext>, u64> {
    let publication = saved_context_publication(fiber);
    let published = publication.load(Ordering::Acquire);
    if published == 0 {
        return Ok(None);
    }
    let expected = saved_context(fiber);
    if published != expected as u64 {
        return Err(fiber_error(SCE_FIBER_ERROR_INVALID));
    }
    // SAFETY: `expected` lies within the validated, live 256-byte fiber object.
    let (rsp, return_address) = unsafe { ((*expected).rsp, (*expected).return_address) };
    let stack_start = read_guest_u64(fiber + offset::ADDR_CONTEXT);
    let stack_size = read_guest_u64(fiber + offset::SIZE_CONTEXT);
    let Some(stack_end) = stack_start.checked_add(stack_size) else {
        return Err(fiber_error(SCE_FIBER_ERROR_INVALID));
    };
    if return_address == 0
        || rsp & 7 != 0
        || rsp < stack_start
        || rsp.checked_add(8).is_none_or(|end| end > stack_end)
    {
        return Err(fiber_error(SCE_FIBER_ERROR_INVALID));
    }
    publication
        .compare_exchange(published, 0, Ordering::AcqRel, Ordering::Acquire)
        .map(|_| Some(expected))
        .map_err(|_| fiber_error(SCE_FIBER_ERROR_STATE))
}

thread_local! {
    /// The context of the `sceFiberRun` frame on this thread, or null when no fiber is running.
    ///
    /// This is per host thread because that is the ABI: a fiber runs on the thread that ran it and
    /// yields back into that same frame, so guest TLS and thread identity stay the caller's own.
    static THREAD_FIBER_CONTEXT: Cell<*mut FiberContext> = const { Cell::new(std::ptr::null_mut()) };
}

/// Reads this thread's current fiber context.
///
/// Deliberately never inlined. A fiber suspended on one thread can be resumed on another — job
/// systems hand a parked fiber to whichever worker picks it up next — so a frame that resumes
/// after [`switch::fiber_set_jmp`] may be running on a different thread than the one that
/// suspended it. Thread-local access normally lets the compiler hoist and reuse the resolved slot
/// address, which is sound only because the thread cannot change mid-function; here it can, and a
/// hoisted address would read the *suspending* thread's slot, which by then reads null. Keeping
/// the resolution behind a call that cannot be inlined forces it to happen after the resume, on
/// the thread that actually continues.
#[inline(never)]
fn fiber_context() -> *mut FiberContext {
    THREAD_FIBER_CONTEXT.with(Cell::get)
}

/// Returns the running SceFiber object for diagnostics, or zero outside a fiber chain.
fn current_fiber_identity() -> u64 {
    let context = fiber_context();
    if context.is_null() {
        0
    } else {
        // SAFETY: a non-null TLS context belongs to the live run frame on this host thread.
        unsafe { (*context).current_fiber }
    }
}

/// Publishes this thread's current fiber context. Never inlined, for the reason above.
#[inline(never)]
fn set_fiber_context(context: *mut FiberContext) {
    THREAD_FIBER_CONTEXT.with(|slot| slot.set(context));
}

fn fiber_error(code: u32) -> u64 {
    u64::from(code)
}

/// Reads a fiber's state word as the shared atomic the guest may also observe.
fn fiber_state(fiber: u64) -> &'static AtomicU32 {
    // SAFETY: callers validate `fiber` as a live, aligned guest fiber object first, and the state
    // word is a naturally aligned `u32` inside it.
    unsafe { AtomicU32::from_ptr((fiber + offset::STATE) as *mut u32) }
}

/// Rejects anything that is not a live, correctly aligned fiber object.
fn validate_fiber(fiber: u64) -> Result<(), u64> {
    if fiber == 0 {
        return Err(fiber_error(SCE_FIBER_ERROR_NULL));
    }
    if fiber & 7 != 0 {
        return Err(fiber_error(SCE_FIBER_ERROR_ALIGNMENT));
    }
    if read_guest_u32(fiber + offset::MAGIC_START) != FIBER_MAGIC_START
        || read_guest_u32(fiber + offset::MAGIC_END) != FIBER_MAGIC_END
    {
        return Err(fiber_error(SCE_FIBER_ERROR_INVALID));
    }
    Ok(())
}

/// Claims a fiber for execution, which only an idle fiber permits.
fn claim_idle(fiber: u64) -> Result<(), u64> {
    fiber_state(fiber)
        .compare_exchange(
            FIBER_STATE_IDLE,
            FIBER_STATE_RUN,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .map(|_| ())
        .map_err(|_| fiber_error(SCE_FIBER_ERROR_STATE))
}

/// Reports whether a running fiber has written past the low end of its stack.
fn stack_intact(fiber: u64) -> bool {
    if read_guest_u32(fiber + offset::FLAGS) & SCE_FIBER_OPT_PARAM_DEBUG_OPTION_CHECK_CONTEXT_SIZE
        == 0
    {
        return true;
    }
    let address = read_guest_u64(fiber + offset::ADDR_CONTEXT);
    address == 0 || read_guest_u64(address) == FIBER_STACK_GUARD
}

/// Builds the handoff record that starts `fiber` on `stack_top`.
fn entry_data(
    fiber: u64,
    argument_on_run_to: u64,
    stack_top: u64,
    previous_state: u64,
) -> FiberData {
    FiberData {
        entry: read_guest_u64(fiber + offset::ENTRY),
        argument_on_initialize: read_guest_u64(fiber + offset::ARG_ON_INITIALIZE),
        argument_on_run_to,
        stack_address: stack_top,
        previous_state,
        // The console sets a defined floating-point environment for fibers built by modern SDKs,
        // which is every PS5 title.
        mxcsr: 0x9fc0,
        floating_point_control: 0x037f,
        _padding: 0,
    }
}

/// Starts `fiber`, or resumes it if it already has a saved context.
///
/// # Safety
///
/// `context` must be the live thread context, and `fiber` a validated fiber claimed for running.
unsafe fn switch_to_fiber(
    fiber: u64,
    argument_on_run_to: u64,
    context: *mut FiberContext,
    saved: Option<*mut FiberJumpContext>,
) -> ! {
    if trace_fiber() {
        let jump = saved_context(fiber);
        let (saved_rsp, saved_return) = if saved.is_none() {
            (0, 0)
        } else {
            // SAFETY: validation and the state claim precede every call to this function.
            unsafe { ((*jump).rsp, (*jump).return_address) }
        };
        // SAFETY: the caller supplies the live thread context for the complete transfer.
        let previous = unsafe { (*context).previous_fiber };
        eprintln!(
            "[REMu][FIBER] transfer from={previous:#x} to={fiber:#x} mode={} entry={:#x} arg_init={:#x} arg_run={argument_on_run_to:#x} stack={:#x} size={:#x} saved_context={:#x} saved_rsp={saved_rsp:#x} saved_return={saved_return:#x} host={:?} thread={:?}",
            if saved.is_none() { "fresh" } else { "resume" },
            read_guest_u64(fiber + offset::ENTRY),
            read_guest_u64(fiber + offset::ARG_ON_INITIALIZE),
            read_guest_u64(fiber + offset::ADDR_CONTEXT),
            read_guest_u64(fiber + offset::SIZE_CONTEXT),
            saved.map_or(0, |saved| saved as u64),
            host_thread_id(),
            current_guest_thread_name(),
        );
    }
    if let Some(saved) = saved {
        // SAFETY: the pointer was stored by this module from a frame that is still suspended.
        unsafe {
            (*context).argument_on_run_to = argument_on_run_to;
            switch::fiber_long_jmp(saved, switch::JUMP_SOURCE_FIBER_RESUME)
        }
    }
    // SAFETY: `context` is live for the duration of the switch.
    let previous_state = unsafe {
        let previous = (*context).previous_fiber;
        (*context).previous_fiber = 0;
        if previous == 0 {
            0
        } else {
            previous + offset::STATE
        }
    };
    let stack_top = read_guest_u64(fiber + offset::ADDR_CONTEXT)
        .wrapping_add(read_guest_u64(fiber + offset::SIZE_CONTEXT));
    let data = entry_data(fiber, argument_on_run_to, stack_top, previous_state);
    // SAFETY: the stack top is sixteen-byte aligned by the size and address checks applied when
    // the context was attached, and the entry never returns.
    unsafe { switch::fiber_switch_entry(&raw const data, 1) }
}

/// Called from the entry trampoline when a fiber's entry function returns, which it must not.
///
/// # Safety
///
/// Called only from [`switch::fiber_switch_entry`] with a live thread context.
pub(crate) unsafe extern "C" fn fiber_force_quit(value: u64) -> ! {
    let context = fiber_context();
    assert!(
        !context.is_null(),
        "a fiber entry returned with no thread context to return into"
    );
    // SAFETY: the thread context frame is suspended, not returned.
    unsafe {
        (*context).entry_returned = value;
        switch::fiber_long_jmp(&raw mut (*context).jump, switch::JUMP_SOURCE_ENTRY_RETURN)
    }
}

/// `sceFiberOptParamInitialize(optParam)`.
fn opt_param_initialize(opt_param: u64) -> u64 {
    if opt_param == 0 {
        return fiber_error(SCE_FIBER_ERROR_NULL);
    }
    if opt_param & 7 != 0 {
        return fiber_error(SCE_FIBER_ERROR_ALIGNMENT);
    }
    guest_memset(opt_param, 0, SCE_FIBER_OPT_PARAM_SIZE as u64);
    write_guest_u32(opt_param, FIBER_OPT_PARAM_MAGIC);
    0
}

/// `_sceFiberInitializeImpl(fiber, name, entry, argOnInitialize, addrContext, sizeContext,
/// optParam, buildVersion)`.
fn initialize(arguments: [u64; 6], stack_arguments: [u64; 6]) -> u64 {
    let fiber = arguments[0];
    let name = arguments[1];
    let entry = arguments[2];
    let argument_on_initialize = arguments[3];
    let address = arguments[4];
    let size = arguments[5];
    let opt_param = stack_arguments[0];

    if fiber == 0 || name == 0 || entry == 0 {
        return fiber_error(SCE_FIBER_ERROR_NULL);
    }
    if fiber & 7 != 0 || address & 15 != 0 {
        return fiber_error(SCE_FIBER_ERROR_ALIGNMENT);
    }
    if opt_param != 0 && opt_param & 7 != 0 {
        return fiber_error(SCE_FIBER_ERROR_ALIGNMENT);
    }
    if size != 0 && size < FIBER_CONTEXT_MINIMUM_SIZE {
        return fiber_error(SCE_FIBER_ERROR_RANGE);
    }
    if size & 15 != 0 || (address == 0) != (size == 0) {
        return fiber_error(SCE_FIBER_ERROR_INVALID);
    }
    if opt_param != 0 && read_guest_u32(opt_param) != FIBER_OPT_PARAM_MAGIC {
        return fiber_error(SCE_FIBER_ERROR_INVALID);
    }
    let debug_options = if opt_param == 0 {
        0
    } else {
        read_guest_u32(opt_param + 4)
    };
    if debug_options & !SCE_FIBER_OPT_PARAM_DEBUG_OPTION_CHECK_CONTEXT_SIZE != 0 {
        return fiber_error(SCE_FIBER_ERROR_INVALID);
    }

    write_guest_u64(fiber + offset::ENTRY, entry);
    write_guest_u64(fiber + offset::ARG_ON_INITIALIZE, argument_on_initialize);
    write_guest_u64(fiber + offset::ADDR_CONTEXT, address);
    write_guest_u64(fiber + offset::SIZE_CONTEXT, size);
    write_guest_u64(fiber + offset::CONTEXT, 0);
    guest_memset(
        fiber + offset::SAVED_CONTEXT,
        0,
        std::mem::size_of::<FiberJumpContext>() as u64,
    );
    write_guest_u32(fiber + offset::FLAGS, debug_options);
    write_fiber_name(fiber, name);
    if address != 0 && debug_options & SCE_FIBER_OPT_PARAM_DEBUG_OPTION_CHECK_CONTEXT_SIZE != 0 {
        write_guest_u64(address, FIBER_STACK_GUARD);
    }
    write_guest_u32(fiber + offset::MAGIC_START, FIBER_MAGIC_START);
    write_guest_u32(fiber + offset::MAGIC_END, FIBER_MAGIC_END);
    // The state guard is published last so a fiber is never claimable before it is fully described.
    fiber_state(fiber).store(FIBER_STATE_IDLE, Ordering::Release);
    0
}

/// Copies a NUL-terminated name into the fiber object, truncated to the ABI limit.
fn write_fiber_name(fiber: u64, name: u64) {
    let text = read_guest_cstring(name).unwrap_or_default();
    let mut bytes = text.into_bytes();
    bytes.truncate(FIBER_MAX_NAME_LENGTH);
    bytes.push(0);
    for (index, byte) in bytes.iter().enumerate() {
        crate::write_guest_u8(fiber + offset::NAME + index as u64, *byte);
    }
}

/// `sceFiberFinalize(fiber)`.
fn finalize(fiber: u64) -> u64 {
    if let Err(error) = validate_fiber(fiber) {
        return error;
    }
    match fiber_state(fiber).compare_exchange(
        FIBER_STATE_IDLE,
        FIBER_STATE_TERMINATED,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => 0,
        Err(_) => fiber_error(SCE_FIBER_ERROR_STATE),
    }
}

/// `sceFiberRun(fiber, argOnRunTo, argOnReturn)`.
///
/// Returns only when the fiber, or one it handed off to, returns to this thread.
fn run(arguments: [u64; 6]) -> u64 {
    let fiber = arguments[0];
    let argument_on_run_to = arguments[1];
    let argument_on_return = arguments[2];
    if let Err(error) = validate_fiber(fiber) {
        return error;
    }
    // A thread already inside a fiber must use `sceFiberSwitch`; nesting `run` would strand the
    // outer frame with no way back.
    if !fiber_context().is_null() {
        return fiber_error(SCE_FIBER_ERROR_PERMISSION);
    }
    if read_guest_u64(fiber + offset::ADDR_CONTEXT) == 0 {
        return fiber_error(SCE_FIBER_ERROR_INVALID);
    }
    if let Err(error) = claim_idle(fiber) {
        return error;
    }
    let saved = match take_saved_context(fiber) {
        Ok(saved) => saved,
        Err(error) => {
            fiber_state(fiber).store(FIBER_STATE_IDLE, Ordering::Release);
            return error;
        }
    };

    // A run chain can migrate through a fiber whose context allocation is also reused as a
    // worker stack. Keeping this long-lived return record in the caller's stack lets another
    // worker zero that stack while `fiber_long_jmp` is restoring it, producing a torn context
    // (a valid RSP followed by a zero return target). Give the chain stable storage independent
    // of every guest/native stack it may traverse.
    let context = Box::into_raw(Box::new(FiberContext {
        current_fiber: fiber,
        ..FiberContext::default()
    }));
    set_fiber_context(context);

    // SAFETY: `context` lives in this frame, which stays suspended until the fiber returns here.
    let resumed = unsafe { switch::fiber_set_jmp(&raw mut (*context).jump) };
    if resumed == 0 {
        // SAFETY: the fiber is claimed and its context validated above.
        unsafe { switch_to_fiber(fiber, argument_on_run_to, context, saved) }
    }

    // SAFETY: reached only by a long jump back into this frame.
    let context = fiber_context();
    assert!(
        !context.is_null(),
        "a resumed fiber run chain must retain its stable thread context"
    );
    let (current, returned, argument) = unsafe {
        let current = (*context).current_fiber;
        (*context).current_fiber = 0;
        (
            current,
            (*context).entry_returned,
            (*context).argument_on_return,
        )
    };
    if current != 0 {
        fiber_state(current).store(FIBER_STATE_IDLE, Ordering::Release);
    }
    set_fiber_context(std::ptr::null_mut());
    // SAFETY: this run invocation created the allocation and has cleared the only published TLS
    // pointer after the chain returned. No fiber can retain the thread context past this point.
    drop(unsafe { Box::from_raw(context) });
    if returned != 0 {
        // The entry ran off its own end. Report it rather than pretending the fiber yielded.
        return fiber_error(SCE_FIBER_ERROR_STATE);
    }
    if argument_on_return != 0 {
        write_guest_u64(argument_on_return, argument);
    }
    0
}

/// `sceFiberSwitch(fiber, argOnRunTo, argOnRun)`.
fn switch_fiber(arguments: [u64; 6]) -> u64 {
    let fiber = arguments[0];
    let argument_on_run_to = arguments[1];
    let argument_on_run = arguments[2];
    if let Err(error) = validate_fiber(fiber) {
        return error;
    }
    let thread_context = fiber_context();
    if thread_context.is_null() {
        return fiber_error(SCE_FIBER_ERROR_PERMISSION);
    }
    if read_guest_u64(fiber + offset::ADDR_CONTEXT) == 0 {
        return fiber_error(SCE_FIBER_ERROR_INVALID);
    }
    if let Err(error) = claim_idle(fiber) {
        return error;
    }
    let saved_target = match take_saved_context(fiber) {
        Ok(saved) => saved,
        Err(error) => {
            fiber_state(fiber).store(FIBER_STATE_IDLE, Ordering::Release);
            return error;
        }
    };

    // SAFETY: the thread context frame is suspended below this one.
    let current = unsafe { (*thread_context).current_fiber };
    let saved = saved_context(current);
    // SAFETY: the opaque fiber object stays live while suspended and owns this register record.
    let resumed = unsafe { switch::fiber_set_jmp(saved) };
    if resumed == 0 {
        assert!(
            stack_intact(current),
            "a fiber overflowed its context before switching away"
        );
        saved_context_publication(current).store(saved as u64, Ordering::Release);
        // SAFETY: both fibers are validated and the target is claimed.
        unsafe {
            (*thread_context).previous_fiber = current;
            (*thread_context).current_fiber = fiber;
            switch_to_fiber(fiber, argument_on_run_to, thread_context, saved_target)
        }
    }
    resume_here(argument_on_run)
}

/// `sceFiberReturnToThread(argOnReturn, argOnRun)`.
fn return_to_thread(arguments: [u64; 6]) -> u64 {
    let argument_on_return = arguments[0];
    let argument_on_run = arguments[1];
    let thread_context = fiber_context();
    if thread_context.is_null() {
        return fiber_error(SCE_FIBER_ERROR_PERMISSION);
    }
    // SAFETY: the thread context frame is suspended below this one.
    let current = unsafe { (*thread_context).current_fiber };
    let saved = saved_context(current);
    // SAFETY: the opaque fiber object stays live while suspended and owns this register record.
    let resumed = unsafe { switch::fiber_set_jmp(saved) };
    if resumed == 0 {
        assert!(
            stack_intact(current),
            "a fiber overflowed its context before returning to its thread"
        );
        saved_context_publication(current).store(saved as u64, Ordering::Release);
        // SAFETY: the thread context frame is live and suspended.
        unsafe {
            (*thread_context).argument_on_return = argument_on_return;
            switch::fiber_long_jmp(
                &raw mut (*thread_context).jump,
                switch::JUMP_SOURCE_RETURN_TO_THREAD,
            )
        }
    }
    resume_here(argument_on_run)
}

/// Shared tail for the two entry points a fiber can be resumed through.
///
/// Never inlined: every arrival here is a resume, possibly on a different thread than the one that
/// suspended the frame, so the thread context has to be resolved fresh rather than reusing
/// anything the suspending frame had already computed.
#[inline(never)]
fn resume_here(argument_on_run: u64) -> u64 {
    // The thread context is re-read rather than reused: this arrival came through a long jump, and
    // the running fiber may have moved on since the value was last loaded.
    let thread_context = fiber_context();
    if thread_context.is_null() {
        return fiber_error(SCE_FIBER_ERROR_PERMISSION);
    }
    // SAFETY: the thread context frame is suspended below this one.
    let (previous, argument) = unsafe {
        let previous = (*thread_context).previous_fiber;
        (*thread_context).previous_fiber = 0;
        (previous, (*thread_context).argument_on_run_to)
    };
    if previous != 0 {
        fiber_state(previous).store(FIBER_STATE_IDLE, Ordering::Release);
    }
    if argument_on_run != 0 {
        write_guest_u64(argument_on_run, argument);
    }
    0
}

/// `sceFiberGetSelf(fiber)`.
fn get_self(output: u64) -> u64 {
    if output == 0 {
        return fiber_error(SCE_FIBER_ERROR_NULL);
    }
    let thread_context = fiber_context();
    if thread_context.is_null() {
        return fiber_error(SCE_FIBER_ERROR_PERMISSION);
    }
    // SAFETY: the thread context frame is suspended below this one.
    write_guest_u64(output, unsafe { (*thread_context).current_fiber });
    0
}

/// `sceFiberGetInfo(fiber, info)`.
fn get_info(fiber: u64, info: u64) -> u64 {
    if fiber == 0 || info == 0 {
        return fiber_error(SCE_FIBER_ERROR_NULL);
    }
    if let Err(error) = validate_fiber(fiber) {
        return error;
    }
    write_guest_u64(info, FIBER_INFO_SIZE);
    write_guest_u64(info + 0x08, read_guest_u64(fiber + offset::ENTRY));
    write_guest_u64(
        info + 0x10,
        read_guest_u64(fiber + offset::ARG_ON_INITIALIZE),
    );
    write_guest_u64(info + 0x18, read_guest_u64(fiber + offset::ADDR_CONTEXT));
    write_guest_u64(info + 0x20, read_guest_u64(fiber + offset::SIZE_CONTEXT));
    for index in 0..=FIBER_MAX_NAME_LENGTH as u64 {
        let byte = crate::read_guest_u8(fiber + offset::NAME + index);
        crate::write_guest_u8(info + 0x28 + index, byte);
    }
    // Unused-stack measurement needs the fill-and-scan mode this build does not arm, so the margin
    // is reported as unknown rather than as a fabricated figure.
    write_guest_u64(info + 0x48, 0);
    0
}

/// `sceFiberRename(fiber, name)`.
fn rename(fiber: u64, name: u64) -> u64 {
    if fiber == 0 || name == 0 {
        return fiber_error(SCE_FIBER_ERROR_NULL);
    }
    if let Err(error) = validate_fiber(fiber) {
        return error;
    }
    write_fiber_name(fiber, name);
    0
}

/// Routes the fiber entry points.
///
/// This is reached from the stateless dispatch path deliberately: `run` and `switch` execute guest
/// code that calls further imports, so they must not hold the process dispatch lock.
pub(crate) fn dispatch(nid: &str, arguments: [u64; 6], stack_arguments: [u64; 6]) -> Option<u64> {
    // A successful switch or return never comes back through this frame, so entry has to be
    // logged separately from the result or the whole successful path is invisible.
    if trace_fiber()
        && matches!(
            nid,
            "a0LLrZWac0M" | "PFT2S-tJ7Uk" | "B0ZX2hx9DMw" | "p+zLIOg27zU"
        )
    {
        eprintln!(
            "[REMu][FIBER] enter nid={nid} arg0={:#x} current={:#x} context={:?} slot={:?} host={:?} thread={:?}",
            arguments[0],
            current_fiber_identity(),
            fiber_context(),
            THREAD_FIBER_CONTEXT.with(std::ptr::from_ref),
            host_thread_id(),
            current_guest_thread_name()
        );
    }
    let (call, result) = match nid {
        "asjUJJ+aa8s" => ("opt-param-init", opt_param_initialize(arguments[0])),
        "hVYD7Ou2pCQ" => ("initialize", initialize(arguments, stack_arguments)),
        "JeNX5F-NzQU" => ("finalize", finalize(arguments[0])),
        "a0LLrZWac0M" => ("run", run(arguments)),
        "PFT2S-tJ7Uk" => ("switch", switch_fiber(arguments)),
        "B0ZX2hx9DMw" => ("return-to-thread", return_to_thread(arguments)),
        "p+zLIOg27zU" => ("get-self", get_self(arguments[0])),
        "uq2Y5BFz0PE" => ("get-info", get_info(arguments[0], arguments[1])),
        "JzyT91ucGDc" => ("rename", rename(arguments[0], arguments[1])),
        _ => return None,
    };
    // A refused fiber call still returns, so it is invisible in the first-hit import log. Trace
    // every transition and its result to make a rejected switch or run legible.
    trace(call, arguments[0], result);
    Some(result)
}

#[cfg(test)]
mod tests;
