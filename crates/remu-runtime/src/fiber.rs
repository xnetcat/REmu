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
//! ABI fields are read and written through the guest accessors. Resumable host register records
//! are different: a title may place an SceFiber object inside another fiber's reusable stack, so
//! they live in a host registry for the complete initialize/finalize generation.
//!
//! A fiber does *not* stay on one thread. An observed job system parks a fiber on one worker and
//! lets whichever worker picks the job up next resume it, so a suspended frame routinely continues
//! on a thread that did not suspend it. Everything a resumed frame needs must therefore be read
//! fresh after the resume rather than carried across it — see [`fiber_context`], where getting
//! this wrong left the resuming thread reading the suspending thread's already-cleared slot.

mod switch;

use std::cell::{Cell, UnsafeCell};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};
use std::sync::{atomic::AtomicBool, atomic::AtomicU32, atomic::AtomicU64, atomic::Ordering};

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
    /// Non-authoritative guest mirror of the host-owned resumable register storage.
    pub(super) const SAVED_CONTEXT: u64 = 0x70;
}

/// The register state required to resume a suspended SysV call.
///
/// Only callee-saved registers belong here. Caller-saved registers are allowed to change across
/// the `fiber_set_jmp` call. The record lives in the host fiber registry rather than the guest
/// object or a stack that a migrating job system can reuse on another worker.
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
    /// Stable host-generation pointer for `current_fiber`, carried with the migrating run chain.
    current_host_fiber: u64,
    /// Host lifecycle word released with `previous_fiber` after a direct handoff.
    previous_host_state: u64,
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
    previous_host_state: u64,
}

const _: () = {
    assert!(std::mem::offset_of!(FiberJumpContext, mxcsr) == 0x40);
    assert!(std::mem::offset_of!(FiberJumpContext, floating_point_control) == 0x44);
    assert!(std::mem::size_of::<FiberJumpContext>() == 0x48);
    assert!(std::mem::offset_of!(FiberContext, current_fiber) == 0x48);
    assert!(std::mem::size_of::<FiberContext>() == 0x80);
    assert!(offset::SAVED_CONTEXT as usize + std::mem::size_of::<FiberJumpContext>() <= 256);
    assert!(std::mem::offset_of!(FiberData, stack_address) == 0x18);
    assert!(std::mem::offset_of!(FiberData, previous_state) == 0x20);
    assert!(std::mem::offset_of!(FiberData, mxcsr) == 0x28);
    assert!(std::mem::offset_of!(FiberData, floating_point_control) == 0x2c);
    assert!(std::mem::offset_of!(FiberData, previous_host_state) == 0x30);
    assert!(std::mem::size_of::<FiberData>() == 0x38);
};

const HOST_FIBER_IDLE: u32 = 1;
const HOST_FIBER_RUN: u32 = 2;
const HOST_FIBER_RETIRED: u32 = 3;

/// One stable initialize/finalize generation of a guest fiber.
///
/// The registry owns this allocation while the fiber is initialized. A successful claim changes
/// its host lifecycle to RUN under the registry read lock before exposing its raw pointer, which
/// excludes initialize/finalize writers until the last pointer use is complete. Guest state is an
/// ABI mirror and is never trusted to bound this allocation's lifetime.
struct HostFiberState {
    generation: u64,
    entry: u64,
    argument_on_initialize: u64,
    stack_start: u64,
    stack_size: u64,
    debug_options: u32,
    lifecycle: AtomicU32,
    published: AtomicBool,
    argument_on_run: AtomicU64,
    jump: UnsafeCell<FiberJumpContext>,
}

impl HostFiberState {
    fn new(
        generation: u64,
        entry: u64,
        argument_on_initialize: u64,
        stack_start: u64,
        stack_size: u64,
        debug_options: u32,
    ) -> Self {
        #[cfg(test)]
        LIVE_HOST_FIBER_STATES.fetch_add(1, Ordering::Relaxed);
        Self {
            generation,
            entry,
            argument_on_initialize,
            stack_start,
            stack_size,
            debug_options,
            lifecycle: AtomicU32::new(HOST_FIBER_IDLE),
            published: AtomicBool::new(false),
            argument_on_run: AtomicU64::new(0),
            jump: UnsafeCell::new(FiberJumpContext::default()),
        }
    }

    fn jump(&self) -> *mut FiberJumpContext {
        self.jump.get()
    }
}

#[cfg(test)]
impl Drop for HostFiberState {
    fn drop(&mut self) {
        LIVE_HOST_FIBER_STATES.fetch_sub(1, Ordering::Relaxed);
    }
}

// SAFETY: the jump record has one writer while host lifecycle is RUN. Release publication and the
// next successful host claim transfer it to the sole successful Acquire taker before any read.
unsafe impl Send for HostFiberState {}
// SAFETY: the protocol above prevents concurrent jump-record reads/writes; all other fields are
// immutable or atomic, and registry lifecycle serialization bounds the allocation lifetime.
unsafe impl Sync for HostFiberState {}

#[derive(Clone, Copy)]
struct ActiveHostFiber {
    identity: u64,
    generation: u64,
    state: *const HostFiberState,
}

impl ActiveHostFiber {
    fn state(&self) -> &HostFiberState {
        // SAFETY: an ActiveHostFiber exists only after host IDLE -> RUN while holding the registry
        // read lock. Initialize/finalize require its write lock and cannot remove this allocation
        // until host lifecycle returns to IDLE, after the last use of this handle.
        unsafe { &*self.state }
    }

    fn jump(self) -> *mut FiberJumpContext {
        self.state().jump()
    }

    fn stack_bounds(self) -> (u64, u64) {
        (self.state().stack_start, self.state().stack_size)
    }
}

#[derive(Default)]
struct HostFiberRegistry {
    next_generation: u64,
    active: HashMap<u64, Box<HostFiberState>>,
}

fn host_fiber_registry() -> &'static RwLock<HostFiberRegistry> {
    static REGISTRY: OnceLock<RwLock<HostFiberRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(HostFiberRegistry::default()))
}

#[cfg(test)]
static LIVE_HOST_FIBER_STATES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
static INITIALIZE_LIFECYCLE_PAUSE: std::sync::Mutex<Option<std::sync::Arc<std::sync::Barrier>>> =
    std::sync::Mutex::new(None);

#[cfg(test)]
static RUN_AFTER_CLAIM_PAUSE: std::sync::Mutex<Option<(u64, std::sync::Arc<std::sync::Barrier>)>> =
    std::sync::Mutex::new(None);

#[cfg(test)]
fn pause_initialize_lifecycle_for_test() {
    let barrier = INITIALIZE_LIFECYCLE_PAUSE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    if let Some(barrier) = barrier {
        barrier.wait();
        barrier.wait();
    }
}

#[cfg(test)]
fn pause_run_after_claim_for_test(fiber: u64) {
    let barrier = RUN_AFTER_CLAIM_PAUSE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .filter(|(target, _)| *target == fiber)
        .map(|(_, barrier)| std::sync::Arc::clone(barrier));
    if let Some(barrier) = barrier {
        barrier.wait();
        barrier.wait();
    }
}

fn register_host_fiber(
    registry: &mut HostFiberRegistry,
    fiber: u64,
    entry: u64,
    argument_on_initialize: u64,
    stack_start: u64,
    stack_size: u64,
    debug_options: u32,
) {
    if let Some(previous) = registry.active.remove(&fiber) {
        assert_eq!(previous.lifecycle.load(Ordering::Acquire), HOST_FIBER_IDLE);
        previous
            .lifecycle
            .store(HOST_FIBER_RETIRED, Ordering::Release);
    }
    registry.next_generation = registry
        .next_generation
        .checked_add(1)
        .expect("fiber generation counter exhausted");
    let generation = registry.next_generation;
    let state = Box::new(HostFiberState::new(
        generation,
        entry,
        argument_on_initialize,
        stack_start,
        stack_size,
        debug_options,
    ));
    registry.active.insert(fiber, state);
}

fn retire_host_fiber(registry: &mut HostFiberRegistry, fiber: u64) {
    if let Some(handle) = registry.active.remove(&fiber) {
        assert_eq!(handle.lifecycle.load(Ordering::Acquire), HOST_FIBER_IDLE);
        handle
            .lifecycle
            .store(HOST_FIBER_RETIRED, Ordering::Release);
    }
}

#[cfg(test)]
fn active_host_fiber(fiber: u64) -> Option<ActiveHostFiber> {
    let registry = host_fiber_registry()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let state = registry.active.get(&fiber)?;
    if state.lifecycle.load(Ordering::Acquire) == HOST_FIBER_RETIRED {
        return None;
    }
    Some(ActiveHostFiber {
        identity: fiber,
        generation: state.generation,
        state: std::ptr::from_ref(state.as_ref()),
    })
}

#[cfg(test)]
fn saved_context(fiber: u64) -> *mut FiberJumpContext {
    active_host_fiber(fiber)
        .expect("a live fiber must have a host generation")
        .jump()
}

/// Publishes a completed host record and mirrors it into the opaque guest ABI fields.
fn publish_saved_context(fiber: ActiveHostFiber, saved: *mut FiberJumpContext) {
    assert_eq!(
        fiber.state().lifecycle.load(Ordering::Acquire),
        HOST_FIBER_RUN
    );
    assert_eq!(fiber.state().generation, fiber.generation);
    assert_eq!(fiber.jump(), saved);
    // SAFETY: the fiber is RUN, so it exclusively owns both its host record and opaque guest ABI
    // mirror. The latter is diagnostic compatibility only and is never used as a jump source.
    unsafe {
        std::ptr::copy_nonoverlapping(
            saved,
            (fiber.identity + offset::SAVED_CONTEXT) as *mut FiberJumpContext,
            1,
        );
    }
    write_guest_u64(
        fiber.identity + offset::CONTEXT,
        fiber.identity + offset::SAVED_CONTEXT,
    );
    fiber.state().published.store(true, Ordering::Release);
}

/// Validates and atomically takes the opaque resume record before handing it to naked assembly.
///
/// Guest object storage is never consulted for the resume pointer or saved registers. Taking the
/// host publication closes the validation/use gap: the transfer jumps through the exact stable
/// generation record this function checked, even if the guest identity is later reinitialized.
#[cfg(test)]
fn take_saved_context(fiber: u64) -> Result<Option<*mut FiberJumpContext>, u64> {
    let fiber = active_host_fiber(fiber).ok_or_else(|| fiber_error(SCE_FIBER_ERROR_INVALID))?;
    take_saved_context_from(fiber)
}

fn take_saved_context_from(fiber: ActiveHostFiber) -> Result<Option<*mut FiberJumpContext>, u64> {
    if fiber
        .state()
        .published
        .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Ok(None);
    }
    let expected = fiber.jump();
    // SAFETY: successful publication transfers the fully written record to this sole taker.
    let (rsp, return_address) = unsafe { ((*expected).rsp, (*expected).return_address) };
    let (stack_start, stack_size) = fiber.stack_bounds();
    let Some(stack_end) = stack_start.checked_add(stack_size) else {
        fiber.state().published.store(true, Ordering::Release);
        return Err(fiber_error(SCE_FIBER_ERROR_INVALID));
    };
    if return_address == 0
        || rsp & 7 != 0
        || rsp < stack_start
        || rsp.checked_add(8).is_none_or(|end| end > stack_end)
    {
        fiber.state().published.store(true, Ordering::Release);
        return Err(fiber_error(SCE_FIBER_ERROR_INVALID));
    }
    write_guest_u64(fiber.identity + offset::CONTEXT, 0);
    Ok(Some(expected))
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

/// Returns the stable generation carried by the live run chain without another registry lookup.
///
/// # Safety
///
/// `context` must be the non-null live context for the currently running fiber.
unsafe fn running_host_fiber(context: *mut FiberContext) -> ActiveHostFiber {
    // SAFETY: guaranteed by the caller; its current fiber remains RUN for every pointer use here.
    let (identity, state) = unsafe {
        (
            (*context).current_fiber,
            (*context).current_host_fiber as *const HostFiberState,
        )
    };
    assert_ne!(
        identity, 0,
        "a running chain must identify its current fiber"
    );
    assert!(
        !state.is_null(),
        "a running chain must carry its host generation"
    );
    // SAFETY: the current fiber is RUN, excluding the registry writers that can remove it.
    let state = unsafe { &*state };
    assert_eq!(state.lifecycle.load(Ordering::Acquire), HOST_FIBER_RUN);
    ActiveHostFiber {
        identity,
        generation: state.generation,
        state: std::ptr::from_ref(state),
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

fn validate_fiber_lifecycle(fiber: u64) -> Result<(), u64> {
    let _registry = host_fiber_registry()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    validate_fiber(fiber)
}

/// Resolves one stable host generation and claims its guest fiber for execution.
///
/// The registry lookup is intentionally paid once per target transition. All subsequent saved
/// record, stack-bound, and publication operations use this handle without taking the global map.
fn claim_idle(fiber: u64) -> Result<ActiveHostFiber, u64> {
    let registry = host_fiber_registry()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    validate_fiber(fiber)?;
    if fiber_state(fiber).load(Ordering::Acquire) != FIBER_STATE_IDLE {
        return Err(fiber_error(SCE_FIBER_ERROR_STATE));
    }
    let state = registry
        .active
        .get(&fiber)
        .ok_or_else(|| fiber_error(SCE_FIBER_ERROR_INVALID))?;
    if state.lifecycle.load(Ordering::Acquire) != HOST_FIBER_IDLE {
        return Err(fiber_error(SCE_FIBER_ERROR_STATE));
    }
    // Holding the read lock through the state claim prevents initialize/finalize from replacing
    // or removing the allocation before RUN makes it ineligible for either operation.
    let host = ActiveHostFiber {
        identity: fiber,
        generation: state.generation,
        state: std::ptr::from_ref(state.as_ref()),
    };
    if host.state().stack_start == 0 {
        return Err(fiber_error(SCE_FIBER_ERROR_INVALID));
    }
    state
        .lifecycle
        .compare_exchange(
            HOST_FIBER_IDLE,
            HOST_FIBER_RUN,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .map_err(|_| fiber_error(SCE_FIBER_ERROR_STATE))?;
    if fiber_state(fiber)
        .compare_exchange(
            FIBER_STATE_IDLE,
            FIBER_STATE_RUN,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_err()
    {
        state.lifecycle.store(HOST_FIBER_IDLE, Ordering::Release);
        return Err(fiber_error(SCE_FIBER_ERROR_STATE));
    }
    Ok(host)
}

/// Releases one claimed generation after its final pointer use.
fn release_claimed_fiber(fiber: ActiveHostFiber) {
    fiber_state(fiber.identity).store(FIBER_STATE_IDLE, Ordering::Release);
    fiber
        .state()
        .lifecycle
        .compare_exchange(
            HOST_FIBER_RUN,
            HOST_FIBER_IDLE,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .expect("a released fiber must own the host RUN generation");
}

fn host_lifecycle_address(fiber: ActiveHostFiber) -> u64 {
    std::ptr::from_ref(&fiber.state().lifecycle).addr() as u64
}

/// Releases a host lifecycle pointer captured while its generation was RUN.
///
/// # Safety
///
/// `lifecycle` must name the live lifecycle word of the RUN generation being handed off. No code
/// may access its HostFiberState after this call because a registry writer may immediately drop it.
unsafe fn release_host_lifecycle(lifecycle: u64) {
    assert_ne!(lifecycle, 0);
    // SAFETY: guaranteed by the caller; AtomicU32 has natural four-byte alignment.
    let lifecycle = unsafe { AtomicU32::from_ptr(lifecycle as *mut u32) };
    lifecycle
        .compare_exchange(
            HOST_FIBER_RUN,
            HOST_FIBER_IDLE,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .expect("a handed-off fiber must own the host RUN generation");
}

/// Reports whether a running fiber has written past the low end of its stack.
fn host_stack_intact(fiber: ActiveHostFiber) -> bool {
    if fiber.state().debug_options & SCE_FIBER_OPT_PARAM_DEBUG_OPTION_CHECK_CONTEXT_SIZE == 0 {
        return true;
    }
    let (address, _) = fiber.stack_bounds();
    address == 0 || read_guest_u64(address) == FIBER_STACK_GUARD
}

#[cfg(test)]
fn stack_intact(fiber: u64) -> bool {
    active_host_fiber(fiber).is_some_and(host_stack_intact)
}

/// Builds the handoff record that starts `fiber` on `stack_top`.
fn entry_data(
    fiber: ActiveHostFiber,
    argument_on_run_to: u64,
    stack_top: u64,
    previous_state: u64,
    previous_host_state: u64,
) -> FiberData {
    FiberData {
        entry: fiber.state().entry,
        argument_on_initialize: fiber.state().argument_on_initialize,
        argument_on_run_to,
        stack_address: stack_top,
        previous_state,
        // The console sets a defined floating-point environment for fibers built by modern SDKs,
        // which is every PS5 title.
        mxcsr: 0x9fc0,
        floating_point_control: 0x037f,
        _padding: 0,
        previous_host_state,
    }
}

/// Starts `fiber`, or resumes it if it already has a saved context.
///
/// # Safety
///
/// `context` must be the live thread context, and `fiber` a validated fiber claimed for running.
unsafe fn switch_to_fiber(
    fiber: ActiveHostFiber,
    argument_on_run_to: u64,
    context: *mut FiberContext,
    saved: Option<*mut FiberJumpContext>,
) -> ! {
    let (stack_start, stack_size) = fiber.stack_bounds();
    if trace_fiber() {
        let (saved_rsp, saved_return) = saved.map_or((0, 0), |jump| {
            // SAFETY: validation and the state claim precede every call to this function.
            unsafe { ((*jump).rsp, (*jump).return_address) }
        });
        // SAFETY: the caller supplies the live thread context for the complete transfer.
        let previous = unsafe { (*context).previous_fiber };
        eprintln!(
            "[REMu][FIBER] transfer from={previous:#x} to={:#x} mode={} entry={:#x} arg_init={:#x} arg_run={argument_on_run_to:#x} stack={:#x} size={:#x} saved_context={:#x} saved_rsp={saved_rsp:#x} saved_return={saved_return:#x} host={:?} thread={:?}",
            fiber.identity,
            if saved.is_none() { "fresh" } else { "resume" },
            fiber.state().entry,
            fiber.state().argument_on_initialize,
            stack_start,
            stack_size,
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
    let (previous_state, previous_host_state) = unsafe {
        let previous = (*context).previous_fiber;
        let previous_host_state = (*context).previous_host_state;
        (*context).previous_fiber = 0;
        (*context).previous_host_state = 0;
        if previous == 0 {
            assert_eq!(previous_host_state, 0);
            (0, 0)
        } else {
            assert_ne!(previous_host_state, 0);
            (previous + offset::STATE, previous_host_state)
        }
    };
    let stack_top = stack_start
        .checked_add(stack_size)
        .expect("fiber stack bounds were validated at initialization");
    let data = entry_data(
        fiber,
        argument_on_run_to,
        stack_top,
        previous_state,
        previous_host_state,
    );
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
    if address.checked_add(size).is_none() {
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

    let mut registry = host_fiber_registry()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(active) = registry.active.get(&fiber)
        && active.lifecycle.load(Ordering::Acquire) != HOST_FIBER_IDLE
    {
        return fiber_error(SCE_FIBER_ERROR_STATE);
    }
    #[cfg(test)]
    pause_initialize_lifecycle_for_test();
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
    register_host_fiber(
        &mut registry,
        fiber,
        entry,
        argument_on_initialize,
        address,
        size,
        debug_options,
    );
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
    let mut registry = host_fiber_registry()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if registry
        .active
        .get(&fiber)
        .is_some_and(|active| active.lifecycle.load(Ordering::Acquire) != HOST_FIBER_IDLE)
    {
        return fiber_error(SCE_FIBER_ERROR_STATE);
    }
    if let Err(error) = validate_fiber(fiber) {
        return error;
    }
    match fiber_state(fiber).compare_exchange(
        FIBER_STATE_IDLE,
        FIBER_STATE_TERMINATED,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => {
            retire_host_fiber(&mut registry, fiber);
            write_guest_u64(fiber + offset::CONTEXT, 0);
            0
        }
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
    // A thread already inside a fiber must use `sceFiberSwitch`; nesting `run` would strand the
    // outer frame with no way back.
    if !fiber_context().is_null() {
        if let Err(error) = validate_fiber_lifecycle(fiber) {
            return error;
        }
        return fiber_error(SCE_FIBER_ERROR_PERMISSION);
    }
    let fiber_host = match claim_idle(fiber) {
        Ok(host) => host,
        Err(error) => return error,
    };
    #[cfg(test)]
    pause_run_after_claim_for_test(fiber);
    let saved = match take_saved_context_from(fiber_host) {
        Ok(saved) => saved,
        Err(error) => {
            release_claimed_fiber(fiber_host);
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
        current_host_fiber: fiber_host.state.addr() as u64,
        ..FiberContext::default()
    }));
    set_fiber_context(context);

    // SAFETY: `context` lives in this frame, which stays suspended until the fiber returns here.
    let resumed = unsafe { switch::fiber_set_jmp(&raw mut (*context).jump) };
    if resumed == 0 {
        // SAFETY: the fiber is claimed and its context validated above.
        unsafe { switch_to_fiber(fiber_host, argument_on_run_to, context, saved) }
    }

    // SAFETY: reached only by a long jump back into this frame.
    let context = fiber_context();
    assert!(
        !context.is_null(),
        "a resumed fiber run chain must retain its stable thread context"
    );
    let (current, current_host_fiber, returned, argument) = unsafe {
        let current = (*context).current_fiber;
        let current_host_fiber = (*context).current_host_fiber;
        (*context).current_fiber = 0;
        (*context).current_host_fiber = 0;
        (
            current,
            current_host_fiber,
            (*context).entry_returned,
            (*context).argument_on_return,
        )
    };
    if current != 0 {
        assert_ne!(current_host_fiber, 0);
        // SAFETY: the current generation remains host-RUN until this final release.
        let lifecycle = unsafe {
            std::ptr::from_ref(&(*(current_host_fiber as *const HostFiberState)).lifecycle).addr()
                as u64
        };
        fiber_state(current).store(FIBER_STATE_IDLE, Ordering::Release);
        // SAFETY: no HostFiberState access occurs after releasing its last ownership word.
        unsafe { release_host_lifecycle(lifecycle) };
    } else {
        assert_eq!(current_host_fiber, 0);
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
    let thread_context = fiber_context();
    if thread_context.is_null() {
        if let Err(error) = validate_fiber_lifecycle(fiber) {
            return error;
        }
        return fiber_error(SCE_FIBER_ERROR_PERMISSION);
    }
    let target_host = match claim_idle(fiber) {
        Ok(host) => host,
        Err(error) => return error,
    };
    let saved_target = match take_saved_context_from(target_host) {
        Ok(saved) => saved,
        Err(error) => {
            release_claimed_fiber(target_host);
            return error;
        }
    };

    // SAFETY: the thread context frame is suspended below this one.
    let current = unsafe { (*thread_context).current_fiber };
    // SAFETY: `thread_context` is the non-null live run chain and identifies `current`.
    let current_host = unsafe { running_host_fiber(thread_context) };
    assert_eq!(current_host.identity, current);
    let saved = current_host.jump();
    current_host
        .state()
        .argument_on_run
        .store(argument_on_run, Ordering::Release);
    // SAFETY: this fiber generation owns the stable host record while its guest state is RUN.
    let resumed = unsafe { switch::fiber_set_jmp(saved) };
    if resumed == 0 {
        assert!(
            host_stack_intact(current_host),
            "a fiber overflowed its context before switching away"
        );
        publish_saved_context(current_host, saved);
        // SAFETY: both fibers are validated and the target is claimed.
        unsafe {
            (*thread_context).previous_fiber = current;
            (*thread_context).previous_host_state = host_lifecycle_address(current_host);
            (*thread_context).current_fiber = fiber;
            (*thread_context).current_host_fiber = target_host.state.addr() as u64;
            switch_to_fiber(
                target_host,
                argument_on_run_to,
                thread_context,
                saved_target,
            )
        }
    }
    resume_here()
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
    // SAFETY: `thread_context` is the non-null live run chain and identifies `current`.
    let current_host = unsafe { running_host_fiber(thread_context) };
    assert_eq!(current_host.identity, current);
    let saved = current_host.jump();
    current_host
        .state()
        .argument_on_run
        .store(argument_on_run, Ordering::Release);
    // SAFETY: this fiber generation owns the stable host record while its guest state is RUN.
    let resumed = unsafe { switch::fiber_set_jmp(saved) };
    if resumed == 0 {
        assert!(
            host_stack_intact(current_host),
            "a fiber overflowed its context before returning to its thread"
        );
        publish_saved_context(current_host, saved);
        // SAFETY: the thread context frame is live and suspended.
        unsafe {
            (*thread_context).argument_on_return = argument_on_return;
            switch::fiber_long_jmp(
                &raw mut (*thread_context).jump,
                switch::JUMP_SOURCE_RETURN_TO_THREAD,
            )
        }
    }
    resume_here()
}

/// Shared tail for the two entry points a fiber can be resumed through.
///
/// Never inlined: every arrival here is a resume, possibly on a different thread than the one that
/// suspended the frame, so the thread context has to be resolved fresh rather than reusing
/// anything the suspending frame had already computed.
#[inline(never)]
fn resume_here() -> u64 {
    // The thread context is re-read rather than reused: this arrival came through a long jump, and
    // the running fiber may have moved on since the value was last loaded.
    let thread_context = fiber_context();
    if thread_context.is_null() {
        return fiber_error(SCE_FIBER_ERROR_PERMISSION);
    }
    // SAFETY: the thread context frame is suspended below this one.
    let (previous, previous_host_state, argument) = unsafe {
        let previous = (*thread_context).previous_fiber;
        let previous_host_state = (*thread_context).previous_host_state;
        (*thread_context).previous_fiber = 0;
        (*thread_context).previous_host_state = 0;
        (
            previous,
            previous_host_state,
            (*thread_context).argument_on_run_to,
        )
    };
    // A Rust local is not safe across the returns-twice assembly boundary: the optimizer may keep
    // it in a caller-saved register. Resolve the per-fiber output pointer from stable host state.
    // SAFETY: `thread_context` identifies the RUN fiber that has just resumed.
    let argument_on_run = unsafe { running_host_fiber(thread_context) }
        .state()
        .argument_on_run
        .load(Ordering::Acquire);
    if previous != 0 {
        assert_ne!(previous_host_state, 0);
        fiber_state(previous).store(FIBER_STATE_IDLE, Ordering::Release);
        // SAFETY: the long jump has installed the target stack, and this is the final access to
        // the outgoing generation whose lifecycle pointer the switch captured.
        unsafe { release_host_lifecycle(previous_host_state) };
    } else {
        assert_eq!(previous_host_state, 0);
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
