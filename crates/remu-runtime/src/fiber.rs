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
//! host-side state is the per-thread record of which fiber is currently running, which is exactly
//! the console's model — a fiber belongs to one thread for its whole life.

mod switch;

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::{read_guest_cstring, read_guest_u32, read_guest_u64, write_guest_u32, write_guest_u64};

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
}

/// A saved register set. This layout is owned here; it never reaches the guest.
#[repr(C)]
#[derive(Default)]
pub(crate) struct FiberContext {
    registers: [u64; 14],
    mxcsr: u32,
    floating_point_control: u16,
    _padding: u16,
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
    assert!(std::mem::offset_of!(FiberContext, mxcsr) == 0x70);
    assert!(std::mem::offset_of!(FiberContext, floating_point_control) == 0x74);
    assert!(std::mem::offset_of!(FiberData, stack_address) == 0x18);
    assert!(std::mem::offset_of!(FiberData, previous_state) == 0x20);
    assert!(std::mem::offset_of!(FiberData, mxcsr) == 0x28);
    assert!(std::mem::offset_of!(FiberData, floating_point_control) == 0x2c);
};

thread_local! {
    /// The context of the `sceFiberRun` frame on this thread, or null when no fiber is running.
    ///
    /// This is per host thread because that is the ABI: a fiber runs on the thread that ran it and
    /// yields back into that same frame, so guest TLS and thread identity stay the caller's own.
    static THREAD_FIBER_CONTEXT: Cell<*mut FiberContext> = const { Cell::new(std::ptr::null_mut()) };
}

fn fiber_context() -> *mut FiberContext {
    THREAD_FIBER_CONTEXT.with(Cell::get)
}

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
unsafe fn switch_to_fiber(fiber: u64, argument_on_run_to: u64, context: *mut FiberContext) -> ! {
    let saved = read_guest_u64(fiber + offset::CONTEXT);
    if saved != 0 {
        // SAFETY: the pointer was stored by this module from a frame that is still suspended.
        unsafe {
            (*context).argument_on_run_to = argument_on_run_to;
            write_guest_u64(fiber + offset::CONTEXT, 0);
            switch::fiber_long_jmp(saved as *mut FiberContext)
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
        switch::fiber_long_jmp(context)
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

    write_guest_u64(fiber + offset::ENTRY, entry);
    write_guest_u64(fiber + offset::ARG_ON_INITIALIZE, argument_on_initialize);
    write_guest_u64(fiber + offset::ADDR_CONTEXT, address);
    write_guest_u64(fiber + offset::SIZE_CONTEXT, size);
    write_guest_u64(fiber + offset::CONTEXT, 0);
    write_guest_u32(fiber + offset::FLAGS, 0);
    write_fiber_name(fiber, name);
    if address != 0 {
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

    let mut context = FiberContext {
        current_fiber: fiber,
        ..FiberContext::default()
    };
    let context = &raw mut context;
    set_fiber_context(context);

    // SAFETY: `context` lives in this frame, which stays suspended until the fiber returns here.
    let resumed = unsafe { switch::fiber_set_jmp(context) };
    if resumed == 0 {
        // SAFETY: the fiber is claimed and its context validated above.
        unsafe { switch_to_fiber(fiber, argument_on_run_to, context) }
    }

    // SAFETY: reached only by a long jump back into this frame.
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

    // SAFETY: the thread context frame is suspended below this one.
    let current = unsafe { (*thread_context).current_fiber };
    let mut saved = FiberContext::default();
    let saved = &raw mut saved;
    // SAFETY: `saved` lives in this frame, which stays suspended until something switches back.
    let resumed = unsafe { switch::fiber_set_jmp(saved) };
    if resumed == 0 {
        assert!(
            stack_intact(current),
            "a fiber overflowed its context before switching away"
        );
        write_guest_u64(current + offset::CONTEXT, saved as u64);
        // SAFETY: both fibers are validated and the target is claimed.
        unsafe {
            (*thread_context).previous_fiber = current;
            (*thread_context).current_fiber = fiber;
            switch_to_fiber(fiber, argument_on_run_to, thread_context)
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
    let mut saved = FiberContext::default();
    let saved = &raw mut saved;
    // SAFETY: `saved` lives in this frame, which stays suspended until this fiber runs again.
    let resumed = unsafe { switch::fiber_set_jmp(saved) };
    if resumed == 0 {
        assert!(
            stack_intact(current),
            "a fiber overflowed its context before returning to its thread"
        );
        write_guest_u64(current + offset::CONTEXT, saved as u64);
        // SAFETY: the thread context frame is live and suspended.
        unsafe {
            (*thread_context).argument_on_return = argument_on_return;
            switch::fiber_long_jmp(thread_context)
        }
    }
    resume_here(argument_on_run)
}

/// Shared tail for the two entry points a fiber can be resumed through.
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
    Some(match nid {
        "asjUJJ+aa8s" => opt_param_initialize(arguments[0]),
        "hVYD7Ou2pCQ" => initialize(arguments, stack_arguments),
        "JeNX5F-NzQU" => finalize(arguments[0]),
        "a0LLrZWac0M" => run(arguments),
        "PFT2S-tJ7Uk" => switch_fiber(arguments),
        "B0ZX2hx9DMw" => return_to_thread(arguments),
        "p+zLIOg27zU" => get_self(arguments[0]),
        "uq2Y5BFz0PE" => get_info(arguments[0], arguments[1]),
        "JzyT91ucGDc" => rename(arguments[0], arguments[1]),
        _ => return None,
    })
}

#[cfg(test)]
mod tests;
