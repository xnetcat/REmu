// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::sync::atomic::Ordering;

use super::*;

/// A sixteen-byte-aligned stack standing in for the guest's fiber context memory.
#[repr(C, align(16))]
struct FiberStack([u8; 0x8000]);

impl FiberStack {
    fn new() -> Box<Self> {
        Box::new(Self([0; 0x8000]))
    }

    fn address(&self) -> u64 {
        std::ptr::from_ref(self).addr() as u64
    }

    fn size() -> u64 {
        0x8000
    }
}

/// An eight-byte-aligned fiber object, as the guest would allocate.
#[repr(C, align(16))]
struct FiberObject([u8; 256]);

impl Default for FiberObject {
    fn default() -> Self {
        Self([0; 256])
    }
}

impl FiberObject {
    fn address(&self) -> u64 {
        std::ptr::from_ref(self).addr() as u64
    }
}

fn initialize_fiber(
    fiber: &FiberObject,
    stack: &FiberStack,
    entry: unsafe extern "C" fn(u64, u64),
    argument_on_initialize: u64,
) -> u64 {
    let name = c"test-fiber";
    initialize(
        [
            fiber.address(),
            name.as_ptr().addr() as u64,
            entry as *const () as usize as u64,
            argument_on_initialize,
            stack.address(),
            FiberStack::size(),
        ],
        [0; 6],
    )
}

/// Records what a fiber entry observed, so the test can prove the switch carried real values.
static OBSERVED: std::sync::Mutex<Vec<(u64, u64)>> = std::sync::Mutex::new(Vec::new());

unsafe extern "C" fn yields_once(argument_on_initialize: u64, argument_on_run_to: u64) {
    OBSERVED
        .lock()
        .unwrap()
        .push((argument_on_initialize, argument_on_run_to));
    // Hand control back to the thread with a value it can check, then never resume.
    return_to_thread([0xfeed_face, 0, 0, 0, 0, 0]);
    // Reached only if the thread runs this fiber again; yield again so the test can drive a
    // second round trip through the same saved context.
    return_to_thread([0xdead_beef, 0, 0, 0, 0, 0]);
    unreachable!("the test never resumes this fiber a third time");
}

#[test]
fn a_fiber_runs_on_its_own_stack_and_yields_back_with_both_arguments() {
    OBSERVED.lock().unwrap().clear();

    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    assert_eq!(initialize_fiber(&fiber, &stack, yields_once, 0x1111), 0);
    assert_eq!(
        fiber_state(fiber.address()).load(Ordering::Acquire),
        FIBER_STATE_IDLE
    );

    let mut returned = 0_u64;
    assert_eq!(
        run([
            fiber.address(),
            0x2222,
            (&raw mut returned).addr() as u64,
            0,
            0,
            0,
        ]),
        0
    );
    // The entry saw the value bound at initialize and the value passed at run.
    assert_eq!(OBSERVED.lock().unwrap().as_slice(), &[(0x1111, 0x2222)]);
    assert_eq!(returned, 0xfeed_face);
    assert_eq!(
        read_guest_u64(fiber.address() + offset::CONTEXT),
        fiber.address() + offset::SAVED_CONTEXT,
        "a suspended register record must live in the opaque fiber object, not on its stack"
    );
    // A yielded fiber is idle again and the thread is no longer inside a fiber.
    assert_eq!(
        fiber_state(fiber.address()).load(Ordering::Acquire),
        FIBER_STATE_IDLE
    );
    assert!(fiber_context().is_null());

    // Running it again resumes the saved context rather than re-entering the entry function.
    assert_eq!(
        run([
            fiber.address(),
            0x3333,
            (&raw mut returned).addr() as u64,
            0,
            0,
            0,
        ]),
        0
    );
    assert_eq!(OBSERVED.lock().unwrap().len(), 1, "entry must not re-run");
    assert_eq!(returned, 0xdead_beef);
}

#[test]
fn a_corrupt_resume_pointer_is_rejected_before_restoring_the_host_stack() {
    OBSERVED.lock().unwrap().clear();
    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    assert_eq!(initialize_fiber(&fiber, &stack, yields_once, 0), 0);
    assert_eq!(run([fiber.address(), 0, 0, 0, 0, 0]), 0);
    write_guest_u64(
        fiber.address() + offset::CONTEXT,
        fiber.address() + offset::SAVED_CONTEXT + 8,
    );

    assert_eq!(
        run([fiber.address(), 0, 0, 0, 0, 0]),
        fiber_error(SCE_FIBER_ERROR_INVALID)
    );
    assert_eq!(
        fiber_state(fiber.address()).load(Ordering::Acquire),
        FIBER_STATE_IDLE,
        "a refused resume must not claim the fiber"
    );
}

/// The two peers of the switch test, published before either fiber starts. A fiber entry is
/// `extern "C"` and cannot unwind, so an unexpected arrival is recorded and yielded rather than
/// asserted on: a panic here would abort the whole test binary instead of failing one test.
static FIRST_FIBER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static SECOND_FIBER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Marks an arrival the test expects never to happen.
const UNEXPECTED: u64 = 0xffff;

unsafe extern "C" fn switches_to_peer(_argument_on_initialize: u64, argument_on_run_to: u64) {
    OBSERVED.lock().unwrap().push((1, argument_on_run_to));
    let mut argument_on_run = 0_u64;
    let result = switch_fiber([
        SECOND_FIBER.load(Ordering::Acquire),
        0x5555,
        (&raw mut argument_on_run).addr() as u64,
        0,
        0,
        0,
    ]);
    // Reached only when the peer switches back, which proves a two-way fiber-to-fiber handoff.
    OBSERVED.lock().unwrap().push((3, result));
    OBSERVED.lock().unwrap().push((4, argument_on_run));
    return_to_thread([0xabcd, 0, 0, 0, 0, 0]);
    OBSERVED.lock().unwrap().push((UNEXPECTED, 0));
    return_to_thread([0, 0, 0, 0, 0, 0]);
}

unsafe extern "C" fn switches_back(_argument_on_initialize: u64, argument_on_run_to: u64) {
    OBSERVED.lock().unwrap().push((2, argument_on_run_to));
    let result = switch_fiber([FIRST_FIBER.load(Ordering::Acquire), 0x6666, 0, 0, 0, 0]);
    // A switch away must not return here; record it so the test fails rather than aborts.
    OBSERVED.lock().unwrap().push((UNEXPECTED, result));
    return_to_thread([0, 0, 0, 0, 0, 0]);
}

#[test]
fn one_fiber_switches_directly_to_another_and_back() {
    OBSERVED.lock().unwrap().clear();
    let first = FiberObject::default();
    let first_stack = FiberStack::new();
    let second = FiberObject::default();
    let second_stack = FiberStack::new();
    assert_eq!(
        initialize_fiber(&first, &first_stack, switches_to_peer, 0),
        0
    );
    assert_eq!(
        initialize_fiber(&second, &second_stack, switches_back, 0),
        0
    );
    // The first fiber switches to the second; the second switches back into the first's saved
    // context, which resumes inside `sceFiberSwitch` rather than at the entry function.
    FIRST_FIBER.store(first.address(), Ordering::Release);
    SECOND_FIBER.store(second.address(), Ordering::Release);

    let mut returned = 0_u64;
    assert_eq!(
        run([
            first.address(),
            0x4444,
            (&raw mut returned).addr() as u64,
            0,
            0,
            0,
        ]),
        0
    );
    assert_eq!(
        OBSERVED.lock().unwrap().as_slice(),
        &[
            // first entered with the run argument
            (1, 0x4444),
            // second entered with the argument the switch carried
            (2, 0x5555),
            // first resumed from its switch, reporting success
            (3, 0),
            // and received the argument the peer switched back with
            (4, 0x6666),
        ]
    );
    assert_eq!(returned, 0xabcd);
    // Both fibers are idle again once control left them.
    assert_eq!(
        fiber_state(first.address()).load(Ordering::Acquire),
        FIBER_STATE_IDLE
    );
    assert_eq!(
        fiber_state(second.address()).load(Ordering::Acquire),
        FIBER_STATE_IDLE
    );
}

unsafe extern "C" fn reports_self(_argument_on_initialize: u64, _argument_on_run_to: u64) {
    let mut self_fiber = 0_u64;
    let result = get_self((&raw mut self_fiber).addr() as u64);
    OBSERVED.lock().unwrap().push((result, self_fiber));
    return_to_thread([0, 0, 0, 0, 0, 0]);
    unreachable!("the test never resumes this fiber");
}

#[test]
fn a_running_fiber_can_identify_itself() {
    OBSERVED.lock().unwrap().clear();
    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    assert_eq!(initialize_fiber(&fiber, &stack, reports_self, 0), 0);
    assert_eq!(run([fiber.address(), 0, 0, 0, 0, 0]), 0);
    assert_eq!(
        OBSERVED.lock().unwrap().as_slice(),
        &[(0, fiber.address())],
        "sceFiberGetSelf must name the running fiber"
    );
}

#[test]
fn fiber_entry_points_reject_calls_from_outside_a_fiber() {
    assert!(fiber_context().is_null());
    let mut value = 0_u64;
    assert_eq!(
        return_to_thread([0, (&raw mut value).addr() as u64, 0, 0, 0, 0]),
        fiber_error(SCE_FIBER_ERROR_PERMISSION)
    );
    assert_eq!(
        get_self((&raw mut value).addr() as u64),
        fiber_error(SCE_FIBER_ERROR_PERMISSION)
    );
    assert_eq!(get_self(0), fiber_error(SCE_FIBER_ERROR_NULL));
}

#[test]
fn initialize_validates_its_arguments_before_publishing_the_object() {
    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    let name = c"n";
    let entry = yields_once as *const () as usize as u64;
    let base = [
        fiber.address(),
        name.as_ptr().addr() as u64,
        entry,
        0,
        stack.address(),
        FiberStack::size(),
    ];

    let mut null_entry = base;
    null_entry[2] = 0;
    assert_eq!(
        initialize(null_entry, [0; 6]),
        fiber_error(SCE_FIBER_ERROR_NULL)
    );
    let mut misaligned = base;
    misaligned[0] = fiber.address() + 1;
    assert_eq!(
        initialize(misaligned, [0; 6]),
        fiber_error(SCE_FIBER_ERROR_ALIGNMENT)
    );
    let mut tiny = base;
    tiny[5] = 128;
    assert_eq!(initialize(tiny, [0; 6]), fiber_error(SCE_FIBER_ERROR_RANGE));
    let mut ragged = base;
    ragged[5] = FiberStack::size() + 8;
    assert_eq!(
        initialize(ragged, [0; 6]),
        fiber_error(SCE_FIBER_ERROR_INVALID)
    );
    // A context address without a size, or the reverse, describes no usable stack.
    let mut sizeless = base;
    sizeless[5] = 0;
    assert_eq!(
        initialize(sizeless, [0; 6]),
        fiber_error(SCE_FIBER_ERROR_INVALID)
    );
    // A rejected call must leave the object unpublished, so a later run cannot claim it.
    assert_eq!(
        run([fiber.address(), 0, 0, 0, 0, 0]),
        fiber_error(SCE_FIBER_ERROR_INVALID)
    );
}

#[test]
fn an_uninitialized_or_finalized_fiber_cannot_be_run() {
    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    assert_eq!(
        run([fiber.address(), 0, 0, 0, 0, 0]),
        fiber_error(SCE_FIBER_ERROR_INVALID),
        "an object without the paired magics is not a fiber"
    );
    assert_eq!(initialize_fiber(&fiber, &stack, yields_once, 0), 0);
    assert_eq!(finalize(fiber.address()), 0);
    assert_eq!(
        run([fiber.address(), 0, 0, 0, 0, 0]),
        fiber_error(SCE_FIBER_ERROR_STATE),
        "a terminated fiber is not runnable"
    );
    assert_eq!(
        finalize(fiber.address()),
        fiber_error(SCE_FIBER_ERROR_STATE),
        "finalizing twice is a state error"
    );
}

#[test]
fn opt_param_initialize_publishes_the_magic_initialize_checks_for() {
    let mut opt_param = [u64::MAX; SCE_FIBER_OPT_PARAM_SIZE / std::mem::size_of::<u64>()];
    let address = opt_param.as_mut_ptr().addr() as u64;
    assert_eq!(opt_param_initialize(0), fiber_error(SCE_FIBER_ERROR_NULL));
    assert_eq!(
        opt_param_initialize(address + 1),
        fiber_error(SCE_FIBER_ERROR_ALIGNMENT)
    );
    assert_eq!(opt_param_initialize(address), 0);
    assert_eq!(read_guest_u32(address), FIBER_OPT_PARAM_MAGIC);
    assert_eq!(
        read_guest_u32(address + 4),
        0,
        "the initialized debug-option field must default to disabled"
    );
    assert!(
        opt_param[1..].iter().all(|word| *word == 0),
        "initialization must not leave caller garbage in the public option object"
    );

    // An opt param the guest never initialized is rejected rather than trusted.
    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    let name = c"n";
    let mut unset = [0_u64; 2];
    let arguments = [
        fiber.address(),
        name.as_ptr().addr() as u64,
        yields_once as *const () as usize as u64,
        0,
        stack.address(),
        FiberStack::size(),
    ];
    let mut stack_arguments = [0_u64; 6];
    stack_arguments[0] = unset.as_mut_ptr().addr() as u64;
    assert_eq!(
        initialize(arguments, stack_arguments),
        fiber_error(SCE_FIBER_ERROR_INVALID)
    );
    stack_arguments[0] = address;
    assert_eq!(initialize(arguments, stack_arguments), 0);
}

#[test]
fn context_memory_is_untouched_unless_the_sdk_debug_option_requests_a_size_check() {
    let fiber = FiberObject::default();
    let mut stack = FiberStack::new();
    stack.0.fill(0xa5);
    let name = c"n";
    let arguments = [
        fiber.address(),
        name.as_ptr().addr() as u64,
        yields_once as *const () as usize as u64,
        0,
        stack.address(),
        FiberStack::size(),
    ];

    assert_eq!(initialize(arguments, [0; 6]), 0);
    assert_eq!(
        &stack.0[..8],
        &[0xa5; 8],
        "a default fiber owns every byte of the context range it supplied"
    );
    assert!(stack_intact(fiber.address()));

    let mut opt_param = [0_u64; SCE_FIBER_OPT_PARAM_SIZE / std::mem::size_of::<u64>()];
    let opt_param_address = opt_param.as_mut_ptr().addr() as u64;
    assert_eq!(opt_param_initialize(opt_param_address), 0);
    write_guest_u32(
        opt_param_address + 4,
        SCE_FIBER_OPT_PARAM_DEBUG_OPTION_CHECK_CONTEXT_SIZE,
    );
    let mut stack_arguments = [0_u64; 6];
    stack_arguments[0] = opt_param_address;
    assert_eq!(initialize(arguments, stack_arguments), 0);
    assert_eq!(read_guest_u64(stack.address()), FIBER_STACK_GUARD);
    assert!(stack_intact(fiber.address()));
    write_guest_u64(stack.address(), 0);
    assert!(!stack_intact(fiber.address()));
}

#[test]
fn get_info_and_rename_report_the_object_the_guest_supplied() {
    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    assert_eq!(initialize_fiber(&fiber, &stack, yields_once, 0x77), 0);

    let mut info = [0_u8; 128];
    let info_address = info.as_mut_ptr().addr() as u64;
    assert_eq!(get_info(fiber.address(), info_address), 0);
    assert_eq!(u64::from_le_bytes(info[0..8].try_into().unwrap()), 128);
    assert_eq!(
        u64::from_le_bytes(info[8..16].try_into().unwrap()),
        yields_once as *const () as usize as u64
    );
    assert_eq!(u64::from_le_bytes(info[16..24].try_into().unwrap()), 0x77);
    assert_eq!(
        u64::from_le_bytes(info[24..32].try_into().unwrap()),
        stack.address()
    );
    assert_eq!(
        u64::from_le_bytes(info[32..40].try_into().unwrap()),
        FiberStack::size()
    );
    assert_eq!(&info[40..50], b"test-fiber");

    let renamed = c"renamed";
    assert_eq!(rename(fiber.address(), renamed.as_ptr().addr() as u64), 0);
    assert_eq!(get_info(fiber.address(), info_address), 0);
    assert_eq!(&info[40..47], b"renamed");
    assert_eq!(info[47], 0, "the shorter name must terminate");

    assert_eq!(get_info(0, info_address), fiber_error(SCE_FIBER_ERROR_NULL));
    assert_eq!(
        get_info(fiber.address(), 0),
        fiber_error(SCE_FIBER_ERROR_NULL)
    );
}

#[test]
fn a_long_name_is_truncated_to_the_abi_limit() {
    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    let long = c"0123456789abcdef0123456789abcdef0123456789";
    assert_eq!(
        initialize(
            [
                fiber.address(),
                long.as_ptr().addr() as u64,
                yields_once as *const () as usize as u64,
                0,
                stack.address(),
                FiberStack::size(),
            ],
            [0; 6],
        ),
        0
    );
    let mut info = [0_u8; 128];
    assert_eq!(
        get_info(fiber.address(), info.as_mut_ptr().addr() as u64),
        0
    );
    assert_eq!(
        &info[40..40 + FIBER_MAX_NAME_LENGTH],
        &long.to_bytes()[..FIBER_MAX_NAME_LENGTH]
    );
    assert_eq!(
        info[40 + FIBER_MAX_NAME_LENGTH],
        0,
        "the name must stay NUL terminated at the limit"
    );
}

/// The fiber a nested `run` will be attempted on, prepared by the test before the outer fiber
/// starts. A fiber entry must not build one itself: a fiber's stack is only as large as the
/// context the guest supplied, and a 32 KiB local on a 32 KiB stack overflows it.
static NESTED_TARGET: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[test]
fn nested_run_is_refused_rather_than_stranding_the_outer_frame() {
    OBSERVED.lock().unwrap().clear();
    let inner = FiberObject::default();
    let inner_stack = FiberStack::new();
    assert_eq!(initialize_fiber(&inner, &inner_stack, yields_once, 0), 0);
    NESTED_TARGET.store(inner.address(), Ordering::Release);

    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    assert_eq!(initialize_fiber(&fiber, &stack, attempts_nested_run, 0), 0);
    assert_eq!(run([fiber.address(), 0, 0, 0, 0, 0]), 0);
    assert_eq!(
        OBSERVED.lock().unwrap().first().map(|entry| entry.0),
        Some(u64::from(SCE_FIBER_ERROR_PERMISSION))
    );
    // The refused target was never claimed, so it stays runnable.
    assert_eq!(
        fiber_state(inner.address()).load(Ordering::Acquire),
        FIBER_STATE_IDLE
    );
}

unsafe extern "C" fn attempts_nested_run(_initialize: u64, _run_to: u64) {
    // A second `run` on a thread already inside a fiber has no way back and must be refused.
    let result = run([NESTED_TARGET.load(Ordering::Acquire), 0, 0, 0, 0, 0]);
    OBSERVED.lock().unwrap().push((result, 0));
    return_to_thread([0, 0, 0, 0, 0, 0]);
    unreachable!("the test never resumes this fiber");
}

static CONCURRENT_FIBER_ACTIVE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static RELEASE_CONCURRENT_FIBER: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

unsafe extern "C" fn waits_while_another_thread_attempts_run(_initialize: u64, _run_to: u64) {
    CONCURRENT_FIBER_ACTIVE.store(true, Ordering::Release);
    while !RELEASE_CONCURRENT_FIBER.load(Ordering::Acquire) {
        std::thread::yield_now();
    }
    return_to_thread([0, 0, 0, 0, 0, 0]);
    unreachable!("the test never resumes this fiber");
}

#[test]
fn a_fiber_cannot_be_claimed_by_two_threads_at_once() {
    CONCURRENT_FIBER_ACTIVE.store(false, Ordering::Release);
    RELEASE_CONCURRENT_FIBER.store(false, Ordering::Release);
    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    assert_eq!(
        initialize_fiber(&fiber, &stack, waits_while_another_thread_attempts_run, 0),
        0
    );
    let fiber_address = fiber.address();
    let running = std::thread::spawn(move || run([fiber_address, 0, 0, 0, 0, 0]));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !CONCURRENT_FIBER_ACTIVE.load(Ordering::Acquire) {
        assert!(
            std::time::Instant::now() < deadline,
            "the first worker did not enter the fiber"
        );
        std::thread::yield_now();
    }

    assert_eq!(
        run([fiber.address(), 0, 0, 0, 0, 0]),
        fiber_error(SCE_FIBER_ERROR_STATE),
        "the state claim must exclude a second worker while the fiber stack is active"
    );
    RELEASE_CONCURRENT_FIBER.store(true, Ordering::Release);
    assert_eq!(running.join().expect("the first worker must return"), 0);
}

/// The fiber a second thread will resume, and the value it reports on that resume.
static MIGRATED_FIBER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MIGRATED_RESULT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

unsafe extern "C" fn yields_then_reports_resume(
    _argument_on_initialize: u64,
    _argument_on_run_to: u64,
) {
    // Suspend on the first thread. Whatever resumes this frame may be a different thread.
    let mut argument_on_run = 0_u64;
    let result = return_to_thread([0x11, (&raw mut argument_on_run).addr() as u64, 0, 0, 0, 0]);
    MIGRATED_RESULT.store(result, Ordering::Release);
    OBSERVED.lock().unwrap().push((result, argument_on_run));
    return_to_thread([0x22, 0, 0, 0, 0, 0]);
    OBSERVED.lock().unwrap().push((UNEXPECTED, 0));
    return_to_thread([0, 0, 0, 0, 0, 0]);
}

/// A job system parks a fiber on one worker and lets another pick it up, so a suspended frame must
/// resume correctly on a thread that did not suspend it. This is the case that fails when the
/// resumed frame reuses a thread-local slot address resolved before the switch: the reader sees the
/// suspending thread's slot, which by then is null, and the resume reports a permission error.
#[test]
fn a_suspended_fiber_resumes_on_a_different_thread() {
    OBSERVED.lock().unwrap().clear();
    MIGRATED_RESULT.store(u64::MAX, Ordering::Release);
    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    assert_eq!(
        initialize_fiber(&fiber, &stack, yields_then_reports_resume, 0),
        0
    );
    MIGRATED_FIBER.store(fiber.address(), Ordering::Release);

    let mut returned = 0_u64;
    assert_eq!(
        run([
            fiber.address(),
            0,
            (&raw mut returned).addr() as u64,
            0,
            0,
            0,
        ]),
        0
    );
    assert_eq!(returned, 0x11);
    assert!(fiber_context().is_null());

    // Resume the same fiber from a second thread, which is where the stale-slot defect appears.
    let resumed = std::thread::spawn(|| {
        let mut returned = 0_u64;
        let status = run([
            MIGRATED_FIBER.load(Ordering::Acquire),
            0x99,
            (&raw mut returned).addr() as u64,
            0,
            0,
            0,
        ]);
        (status, returned)
    })
    .join()
    .expect("the resuming thread must not panic");

    assert_eq!(resumed.0, 0, "resuming on another thread must succeed");
    assert_eq!(
        resumed.1, 0x22,
        "the resumed fiber must yield again normally"
    );
    assert_eq!(
        MIGRATED_RESULT.load(Ordering::Acquire),
        0,
        "the resumed frame must observe the resuming thread's context, not the suspending one"
    );
    assert_eq!(
        OBSERVED.lock().unwrap().as_slice(),
        &[(0, 0x99)],
        "the resume must deliver the argument the second thread passed"
    );
}
