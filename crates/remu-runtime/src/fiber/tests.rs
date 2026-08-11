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

/// The 16 KiB context size used by the repeated migration regression. Keeping the stack small
/// makes a bad saved RSP or gradually drifting call frame fail quickly instead of hiding inside
/// the roomier default fixture.
#[repr(C, align(16))]
struct SmallFiberStack([u8; 0x4000]);

impl SmallFiberStack {
    fn new() -> Box<Self> {
        Box::new(Self([0; 0x4000]))
    }

    fn address(&self) -> u64 {
        std::ptr::from_ref(self).addr() as u64
    }

    fn size() -> u64 {
        0x4000
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

unsafe extern "C" fn returns_immediately(_argument_on_initialize: u64, _argument_on_run_to: u64) {}

#[cfg(target_os = "macos")]
static THREAD_CONTEXT_WAS_ON_NATIVE_STACK: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(target_os = "macos")]
unsafe extern "C" fn records_thread_context_storage(
    _argument_on_initialize: u64,
    _argument_on_run_to: u64,
) {
    let context = fiber_context().addr();
    // SAFETY: these Darwin queries only inspect the current live pthread's immutable stack range.
    let thread = unsafe { libc::pthread_self() };
    // SAFETY: `thread` is the current live pthread.
    let top = unsafe { libc::pthread_get_stackaddr_np(thread) }.addr();
    // SAFETY: `thread` is the current live pthread.
    let size = unsafe { libc::pthread_get_stacksize_np(thread) };
    let base = top
        .checked_sub(size)
        .expect("a native stack range cannot underflow");
    THREAD_CONTEXT_WAS_ON_NATIVE_STACK.store(
        (base..top).contains(&context),
        std::sync::atomic::Ordering::Release,
    );
}

#[test]
fn a_returning_entry_is_caught_by_the_fiber_boundary_instead_of_jumping_to_zero() {
    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    assert_eq!(initialize_fiber(&fiber, &stack, returns_immediately, 0), 0);
    assert_eq!(
        run([fiber.address(), 0, 0, 0, 0, 0]),
        fiber_error(SCE_FIBER_ERROR_STATE)
    );
    assert_eq!(
        fiber_state(fiber.address()).load(Ordering::Acquire),
        FIBER_STATE_IDLE,
        "the force-quit boundary must release a returning fiber"
    );
    assert_eq!(
        active_host_fiber(fiber.address())
            .expect("returning fiber must retain its quiescent generation")
            .state()
            .lifecycle
            .load(Ordering::Acquire),
        HOST_FIBER_IDLE
    );
}

#[cfg(target_os = "macos")]
#[test]
fn a_migrating_thread_context_is_not_owned_by_a_reusable_native_stack() {
    THREAD_CONTEXT_WAS_ON_NATIVE_STACK.store(false, Ordering::Release);
    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    assert_eq!(
        initialize_fiber(&fiber, &stack, records_thread_context_storage, 0),
        0
    );
    assert_eq!(
        run([fiber.address(), 0, 0, 0, 0, 0]),
        fiber_error(SCE_FIBER_ERROR_STATE)
    );
    assert!(
        !THREAD_CONTEXT_WAS_ON_NATIVE_STACK.load(Ordering::Acquire),
        "the run-chain context outlives stack migration and must use stable storage"
    );
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
        "the guest ABI mirror must retain its opaque in-object context address"
    );
    // A yielded fiber is idle again and the thread is no longer inside a fiber.
    assert_eq!(
        fiber_state(fiber.address()).load(Ordering::Acquire),
        FIBER_STATE_IDLE
    );
    assert_eq!(
        active_host_fiber(fiber.address())
            .expect("yielded fiber must retain its host generation")
            .state()
            .lifecycle
            .load(Ordering::Acquire),
        HOST_FIBER_IDLE
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
fn a_corrupt_guest_resume_mirror_cannot_redirect_the_host_context() {
    OBSERVED.lock().unwrap().clear();
    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    assert_eq!(initialize_fiber(&fiber, &stack, yields_once, 0), 0);
    assert_eq!(run([fiber.address(), 0, 0, 0, 0, 0]), 0);
    write_guest_u64(
        fiber.address() + offset::CONTEXT,
        fiber.address() + offset::SAVED_CONTEXT + 8,
    );

    assert_eq!(run([fiber.address(), 0, 0, 0, 0, 0]), 0);
    assert_eq!(
        fiber_state(fiber.address()).load(Ordering::Acquire),
        FIBER_STATE_IDLE,
        "the non-authoritative guest mirror must not redirect the resume"
    );
    assert_eq!(OBSERVED.lock().unwrap().len(), 1, "entry must not re-run");
}

#[test]
fn a_validated_resume_pointer_is_consumed_exactly_once() {
    OBSERVED.lock().unwrap().clear();
    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    assert_eq!(initialize_fiber(&fiber, &stack, yields_once, 0), 0);
    assert_eq!(run([fiber.address(), 0, 0, 0, 0, 0]), 0);

    assert_eq!(
        take_saved_context(fiber.address()).unwrap(),
        Some(saved_context(fiber.address())),
        "validation must return the exact host-generation context it checked"
    );
    assert_eq!(
        take_saved_context(fiber.address()).unwrap(),
        None,
        "the host publication cannot be taken as a second jump target"
    );
}

#[test]
fn a_stack_nested_fiber_uses_a_resume_record_outside_guest_stack_memory() {
    OBSERVED.lock().unwrap().clear();
    let containing_stack = SmallFiberStack::new();
    let fiber = containing_stack.address() + 0x1000;
    let context_stack = FiberStack::new();
    let name = c"stack-nested-fiber";
    assert_eq!(
        initialize(
            [
                fiber,
                name.as_ptr().addr() as u64,
                yields_once as *const () as usize as u64,
                0,
                context_stack.address(),
                FiberStack::size(),
            ],
            [0; 6],
        ),
        0
    );
    assert_eq!(run([fiber, 0, 0, 0, 0, 0]), 0);

    let saved = take_saved_context(fiber)
        .unwrap()
        .expect("the yielded fiber must publish a resume record")
        .addr() as u64;
    let containing_end = containing_stack.address() + SmallFiberStack::size();
    assert!(
        !(containing_stack.address()..containing_end).contains(&saved),
        "a guest stack can be recycled while the nested fiber remains suspended"
    );
    assert!(
        !(fiber..fiber + 256).contains(&saved),
        "the opaque guest object is not stable host resume storage"
    );
}

#[test]
fn a_taken_resume_record_survives_clearing_the_guest_object_storage() {
    OBSERVED.lock().unwrap().clear();
    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    assert_eq!(initialize_fiber(&fiber, &stack, yields_once, 0), 0);
    assert_eq!(run([fiber.address(), 0, 0, 0, 0, 0]), 0);

    let saved = take_saved_context(fiber.address())
        .unwrap()
        .expect("the yielded fiber must publish a resume record");
    guest_memset(
        fiber.address() + offset::SAVED_CONTEXT,
        0,
        std::mem::size_of::<FiberJumpContext>() as u64,
    );
    // SAFETY: a taken record must stay owned by the host until the non-returning jump consumes it.
    let (rsp, return_address) = unsafe { ((*saved).rsp, (*saved).return_address) };
    assert_ne!(
        rsp, 0,
        "guest object reuse must not clear the saved stack pointer"
    );
    assert_ne!(
        return_address, 0,
        "guest object reuse must not clear the saved return target"
    );
}

#[test]
fn concurrent_resume_takes_have_exactly_one_winner() {
    OBSERVED.lock().unwrap().clear();
    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    assert_eq!(initialize_fiber(&fiber, &stack, yields_once, 0), 0);
    assert_eq!(run([fiber.address(), 0, 0, 0, 0, 0]), 0);

    let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
    let mut workers = Vec::new();
    for _ in 0..2 {
        let barrier = std::sync::Arc::clone(&barrier);
        let fiber = fiber.address();
        workers.push(std::thread::spawn(move || {
            barrier.wait();
            take_saved_context(fiber)
                .unwrap()
                .map_or(0, |saved| saved.addr() as u64)
        }));
    }
    barrier.wait();
    let mut results = workers
        .into_iter()
        .map(|worker| worker.join().expect("resume taker must not panic"))
        .collect::<Vec<_>>();
    results.sort_unstable();
    assert_eq!(results[0], 0, "one racing take must observe consumed state");
    assert_ne!(results[1], 0, "one racing take must own the resume record");
}

#[test]
fn an_invalid_host_resume_record_remains_published_for_diagnosis() {
    OBSERVED.lock().unwrap().clear();
    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    assert_eq!(initialize_fiber(&fiber, &stack, yields_once, 0), 0);
    assert_eq!(run([fiber.address(), 0, 0, 0, 0, 0]), 0);

    let saved = saved_context(fiber.address());
    // SAFETY: the fiber is suspended and this test exclusively owns the fixture.
    let valid_return = unsafe { std::mem::replace(&mut (*saved).return_address, 0) };
    assert_eq!(
        take_saved_context(fiber.address()),
        Err(fiber_error(SCE_FIBER_ERROR_INVALID))
    );
    assert_eq!(
        take_saved_context(fiber.address()),
        Err(fiber_error(SCE_FIBER_ERROR_INVALID)),
        "failed validation must not silently turn a suspended fiber into a fresh entry"
    );
    // SAFETY: the invalid record stayed suspended and exclusively owned by this fixture.
    unsafe { (*saved).return_address = valid_return };
    assert_eq!(take_saved_context(fiber.address()).unwrap(), Some(saved));
}

#[test]
fn reinitializing_one_guest_address_does_not_reuse_a_retired_resume_generation() {
    OBSERVED.lock().unwrap().clear();
    let fiber = FiberObject::default();
    let first_stack = FiberStack::new();
    assert_eq!(initialize_fiber(&fiber, &first_stack, yields_once, 0), 0);
    assert_eq!(run([fiber.address(), 0, 0, 0, 0, 0]), 0);
    let retired = active_host_fiber(fiber.address())
        .expect("the first generation must stay registered while suspended")
        .generation;
    assert_eq!(finalize(fiber.address()), 0);

    let second_stack = FiberStack::new();
    assert_eq!(initialize_fiber(&fiber, &second_stack, yields_once, 0), 0);
    assert_eq!(run([fiber.address(), 0, 0, 0, 0, 0]), 0);
    let replacement = active_host_fiber(fiber.address())
        .expect("the replacement generation must stay registered while suspended")
        .generation;
    assert_ne!(
        replacement, retired,
        "a stale raw pointer must never name a replacement fiber generation"
    );
}

#[test]
fn initialize_serializes_guest_publication_against_a_claim() {
    let fiber = FiberObject::default();
    let first_stack = FiberStack::new();
    let second_stack = FiberStack::new();
    assert_eq!(initialize_fiber(&fiber, &first_stack, yields_once, 0), 0);

    let initialize_pause = std::sync::Arc::new(std::sync::Barrier::new(2));
    *INITIALIZE_LIFECYCLE_PAUSE.lock().unwrap() = Some(std::sync::Arc::clone(&initialize_pause));
    let fiber_address = fiber.address();
    let second_stack_address = second_stack.address();
    let initializer = std::thread::spawn(move || {
        let name = c"claim-race";
        initialize(
            [
                fiber_address,
                name.as_ptr().addr() as u64,
                yields_once as *const () as usize as u64,
                0,
                second_stack_address,
                FiberStack::size(),
            ],
            [0; 6],
        )
    });
    initialize_pause.wait();

    let claim_started = std::sync::Arc::new(std::sync::Barrier::new(2));
    let (claimed_tx, claimed_rx) = std::sync::mpsc::channel();
    let worker_started = std::sync::Arc::clone(&claim_started);
    let claimant = std::thread::spawn(move || {
        worker_started.wait();
        claimed_tx
            .send(claim_idle(fiber_address).map(|host| host.generation))
            .unwrap();
    });
    claim_started.wait();
    let early_claim = claimed_rx.recv_timeout(std::time::Duration::from_millis(200));

    initialize_pause.wait();
    *INITIALIZE_LIFECYCLE_PAUSE.lock().unwrap() = None;
    let initialize_result = initializer.join().expect("initializer must not panic");
    let claim_completed_while_initialize_was_paused = early_claim.is_ok();
    let claimed_generation = match early_claim {
        Ok(result) => result,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => claimed_rx.recv().unwrap(),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            panic!("claimant disconnected")
        }
    }
    .expect("the published generation must be claimable");
    claimant.join().expect("claimant must not panic");
    let active_generation = active_host_fiber(fiber.address())
        .expect("a successful initialize must retain an active generation")
        .generation;
    let state_after_race = fiber_state(fiber.address()).load(Ordering::Acquire);
    if state_after_race == FIBER_STATE_RUN {
        release_claimed_fiber(
            active_host_fiber(fiber.address())
                .expect("the claimed generation must remain registered"),
        );
    }
    assert_eq!(finalize(fiber.address()), 0);

    assert_eq!(initialize_result, 0);
    assert!(
        !claim_completed_while_initialize_was_paused,
        "claim must not observe guest fields before their host generation is published"
    );
    assert_eq!(claimed_generation, active_generation);
    assert_eq!(state_after_race, FIBER_STATE_RUN);
}

#[test]
fn initialize_serializes_guest_publication_against_finalize() {
    let fiber = FiberObject::default();
    let first_stack = FiberStack::new();
    let second_stack = FiberStack::new();
    assert_eq!(initialize_fiber(&fiber, &first_stack, yields_once, 0), 0);

    let initialize_pause = std::sync::Arc::new(std::sync::Barrier::new(2));
    *INITIALIZE_LIFECYCLE_PAUSE.lock().unwrap() = Some(std::sync::Arc::clone(&initialize_pause));
    let fiber_address = fiber.address();
    let second_stack_address = second_stack.address();
    let initializer = std::thread::spawn(move || {
        let name = c"finalize-race";
        initialize(
            [
                fiber_address,
                name.as_ptr().addr() as u64,
                yields_once as *const () as usize as u64,
                0,
                second_stack_address,
                FiberStack::size(),
            ],
            [0; 6],
        )
    });
    initialize_pause.wait();

    let finalize_started = std::sync::Arc::new(std::sync::Barrier::new(2));
    let (finalized_tx, finalized_rx) = std::sync::mpsc::channel();
    let worker_started = std::sync::Arc::clone(&finalize_started);
    let finalizer = std::thread::spawn(move || {
        worker_started.wait();
        finalized_tx.send(finalize(fiber_address)).unwrap();
    });
    finalize_started.wait();
    let early_finalize = finalized_rx.recv_timeout(std::time::Duration::from_millis(200));

    initialize_pause.wait();
    *INITIALIZE_LIFECYCLE_PAUSE.lock().unwrap() = None;
    let initialize_result = initializer.join().expect("initializer must not panic");
    let finalize_completed_while_initialize_was_paused = early_finalize.is_ok();
    let finalize_result = match early_finalize {
        Ok(result) => result,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => finalized_rx.recv().unwrap(),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            panic!("finalizer disconnected")
        }
    };
    finalizer.join().expect("finalizer must not panic");
    let state_after_race = fiber_state(fiber.address()).load(Ordering::Acquire);
    let has_active_generation = active_host_fiber(fiber.address()).is_some();
    if state_after_race == FIBER_STATE_IDLE {
        assert_eq!(finalize(fiber.address()), 0);
    }

    assert_eq!(initialize_result, 0);
    assert_eq!(finalize_result, 0);
    assert!(
        !finalize_completed_while_initialize_was_paused,
        "finalize must not retire a different generation than the guest state it terminates"
    );
    assert_eq!(state_after_race, FIBER_STATE_TERMINATED);
    assert!(!has_active_generation);
}

#[test]
fn guest_object_corruption_cannot_retire_a_claimed_host_generation() {
    let fiber = FiberObject::default();
    let first_stack = FiberStack::new();
    let second_stack = FiberStack::new();
    assert_eq!(initialize_fiber(&fiber, &first_stack, yields_once, 0), 0);
    let claimed = claim_idle(fiber.address()).expect("idle fiber must be claimable");

    guest_memset(fiber.address(), 0, 256);
    let name = c"replacement";
    let replacement_arguments = [
        fiber.address(),
        name.as_ptr().addr() as u64,
        yields_once as *const () as usize as u64,
        0,
        second_stack.address(),
        FiberStack::size(),
    ];
    let replacement_while_claimed = initialize(replacement_arguments, [0; 6]);

    let (preserved_generation, replacement_after_release) =
        if replacement_while_claimed == fiber_error(SCE_FIBER_ERROR_STATE) {
            let preserved = active_host_fiber(fiber.address())
                .is_some_and(|active| active.generation == claimed.generation);
            release_claimed_fiber(claimed);
            (preserved, Some(initialize(replacement_arguments, [0; 6])))
        } else {
            (false, None)
        };
    if fiber_state(fiber.address()).load(Ordering::Acquire) == FIBER_STATE_IDLE {
        assert_eq!(finalize(fiber.address()), 0);
    }

    assert_eq!(
        replacement_while_claimed,
        fiber_error(SCE_FIBER_ERROR_STATE),
        "host RUN ownership must survive cleared guest lifecycle bytes"
    );
    assert!(preserved_generation);
    assert_eq!(replacement_after_release, Some(0));
}

#[test]
fn a_fresh_claim_uses_its_host_snapshot_after_guest_object_recycling() {
    OBSERVED.lock().unwrap().clear();
    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    assert_eq!(initialize_fiber(&fiber, &stack, yields_once, 0x1357), 0);

    let pause = std::sync::Arc::new(std::sync::Barrier::new(2));
    *RUN_AFTER_CLAIM_PAUSE.lock().unwrap() = Some((fiber.address(), std::sync::Arc::clone(&pause)));
    let fiber_address = fiber.address();
    let runner = std::thread::spawn(move || run([fiber_address, 0x2468, 0, 0, 0, 0]));
    pause.wait();
    guest_memset(fiber.address(), 0, 256);
    pause.wait();
    *RUN_AFTER_CLAIM_PAUSE.lock().unwrap() = None;
    let result = runner.join().expect("fresh fiber runner must not panic");

    assert_eq!(result, 0);
    assert_eq!(OBSERVED.lock().unwrap().as_slice(), &[(0x1357, 0x2468)]);
    assert_eq!(
        active_host_fiber(fiber.address())
            .expect("recycled guest bytes must not retire the host generation")
            .state()
            .lifecycle
            .load(Ordering::Acquire),
        HOST_FIBER_IDLE
    );
    // Restore the guest ABI object before finalizing the fixture. The quiescent old generation
    // may be replaced because no raw host pointer survives the run.
    assert_eq!(initialize_fiber(&fiber, &stack, yields_once, 0), 0);
    assert_eq!(finalize(fiber.address()), 0);
}

#[test]
fn repeated_initialize_finalize_keeps_host_generation_storage_bounded() {
    let fiber = FiberObject::default();
    let stack = FiberStack::new();
    // Normalize any stale registry entry left by a fixture whose heap address was recycled before
    // taking the baseline for this exact identity.
    assert_eq!(initialize_fiber(&fiber, &stack, yields_once, 0), 0);
    assert_eq!(finalize(fiber.address()), 0);
    let baseline = LIVE_HOST_FIBER_STATES.load(Ordering::Relaxed);
    let mut previous_generation = 0;

    for _ in 0..128 {
        assert_eq!(initialize_fiber(&fiber, &stack, yields_once, 0), 0);
        let generation = active_host_fiber(fiber.address())
            .expect("initialized fiber must have a host generation")
            .generation;
        assert!(generation > previous_generation);
        previous_generation = generation;
        assert_eq!(finalize(fiber.address()), 0);
    }

    assert_eq!(
        LIVE_HOST_FIBER_STATES.load(Ordering::Relaxed),
        baseline,
        "quiescent finalized generations must release their host record"
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
    for fiber in [first.address(), second.address()] {
        assert_eq!(
            active_host_fiber(fiber)
                .expect("handed-off fiber must retain its quiescent generation")
                .state()
                .lifecycle
                .load(Ordering::Acquire),
            HOST_FIBER_IDLE
        );
    }
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

const REPEATED_MIGRATIONS: u64 = 64;
const REPEATED_MIGRATION_DONE: u64 = 0xd00d_f1be;
static REPEATED_MIGRATION_ERROR: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Keeps a real nested return address and local payload live across each thread migration.
#[inline(never)]
fn yield_from_nested_migration_frame(step: u64, argument_on_run: &mut u64) -> u64 {
    let marker = step ^ 0xa5a5_5a5a_1234_5678;
    let payload = [marker; 32];
    let status = return_to_thread([
        marker,
        (argument_on_run as *mut u64).addr() as u64,
        0,
        0,
        0,
        0,
    ]);
    if payload.iter().any(|word| *word != marker) {
        REPEATED_MIGRATION_ERROR.store(step + 1, Ordering::Release);
    }
    std::hint::black_box(&payload);
    status
}

unsafe extern "C" fn repeatedly_yields_across_threads(
    _argument_on_initialize: u64,
    argument_on_run_to: u64,
) {
    let mut received = argument_on_run_to;
    for step in 0..REPEATED_MIGRATIONS {
        if received != 0x1000 + step {
            REPEATED_MIGRATION_ERROR.store(0x1000 + step, Ordering::Release);
        }
        let mut next = 0;
        let status = yield_from_nested_migration_frame(step, &mut next);
        if status != 0 {
            REPEATED_MIGRATION_ERROR.store(0x2000 + step, Ordering::Release);
        }
        received = next;
    }
    return_to_thread([REPEATED_MIGRATION_DONE, 0, 0, 0, 0, 0]);
    unreachable!("the migration stress fiber is not resumed after its final yield");
}

#[test]
fn a_small_fiber_preserves_nested_return_frames_across_repeated_thread_migration() {
    REPEATED_MIGRATION_ERROR.store(0, Ordering::Release);
    let fiber = FiberObject::default();
    let stack = SmallFiberStack::new();
    let name = c"migrating-fiber";
    assert_eq!(
        initialize(
            [
                fiber.address(),
                name.as_ptr().addr() as u64,
                repeatedly_yields_across_threads as *const () as usize as u64,
                0,
                stack.address(),
                SmallFiberStack::size(),
            ],
            [0; 6],
        ),
        0
    );

    let mut returned = 0_u64;
    assert_eq!(
        run([
            fiber.address(),
            0x1000,
            (&raw mut returned).addr() as u64,
            0,
            0,
            0,
        ]),
        0
    );
    assert_eq!(returned, 0xa5a5_5a5a_1234_5678);

    for step in 1..=REPEATED_MIGRATIONS {
        let fiber_address = fiber.address();
        let (status, yielded) = std::thread::spawn(move || {
            let mut yielded = 0;
            let status = run([
                fiber_address,
                0x1000 + step,
                (&raw mut yielded).addr() as u64,
                0,
                0,
                0,
            ]);
            (status, yielded)
        })
        .join()
        .expect("every migration worker must return normally");
        assert_eq!(status, 0, "migration {step} must resume normally");
        let expected = if step == REPEATED_MIGRATIONS {
            REPEATED_MIGRATION_DONE
        } else {
            step ^ 0xa5a5_5a5a_1234_5678
        };
        assert_eq!(yielded, expected, "migration {step} yielded wrong marker");
    }
    assert_eq!(
        REPEATED_MIGRATION_ERROR.load(Ordering::Acquire),
        0,
        "no resumed nested frame may lose its arguments, return address, or local payload"
    );
}

#[cfg(all(unix, target_arch = "x86_64"))]
unsafe extern "C" fn yields_through_generated_import_stub(
    stub_address: u64,
    _argument_on_run_to: u64,
) {
    // SAFETY: the test maps a complete executable import stub with this two-argument SysV shape.
    let stub = unsafe {
        std::mem::transmute::<usize, unsafe extern "C" fn(u64, u64) -> u64>(stub_address as usize)
    };
    let mut argument_on_run = 0_u64;
    // This call does not return until another host thread resumes the suspended import frame.
    let result = unsafe { stub(0x1111, (&raw mut argument_on_run).addr() as u64) };
    OBSERVED.lock().unwrap().push((result, argument_on_run));
    // Yield once more so the resuming thread can return normally without letting the entry end.
    let _ = unsafe { stub(0x2222, 0) };
    OBSERVED.lock().unwrap().push((UNEXPECTED, 0));
    return_to_thread([0, 0, 0, 0, 0, 0]);
}

#[cfg(all(unix, target_arch = "x86_64"))]
#[test]
fn a_generated_import_stub_preserves_its_return_frame_across_fiber_migration() {
    const PAGE_SIZE: usize = 0x4000;
    OBSERVED.lock().unwrap().clear();
    let mut slot = [0_u8; crate::IMPORT_STUB_SIZE_USIZE];
    crate::write_import_stub(
        &mut slot,
        usize::try_from(crate::TEST_FIBER_RETURN_DISPATCH_INDEX).unwrap(),
        crate::import_dispatch_trampoline as *const () as usize as u64,
    )
    .unwrap();
    // SAFETY: the test owns this anonymous page until both fiber legs complete.
    let mapping = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            PAGE_SIZE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    assert_ne!(mapping, libc::MAP_FAILED);
    // SAFETY: the mapping is writable and the fixed stub fits at its start.
    unsafe { std::ptr::copy_nonoverlapping(slot.as_ptr(), mapping.cast::<u8>(), slot.len()) };
    // SAFETY: mapping and size are the exact values returned by mmap above.
    assert_eq!(
        unsafe { libc::mprotect(mapping, PAGE_SIZE, libc::PROT_READ | libc::PROT_EXEC) },
        0
    );

    let fiber = FiberObject::default();
    let stack = SmallFiberStack::new();
    let name = c"stub-migration";
    assert_eq!(
        initialize(
            [
                fiber.address(),
                name.as_ptr().addr() as u64,
                yields_through_generated_import_stub as *const () as usize as u64,
                mapping.addr() as u64,
                stack.address(),
                SmallFiberStack::size(),
            ],
            [0; 6],
        ),
        0
    );
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
    assert_eq!(returned, 0x1111);

    let fiber_address = fiber.address();
    let (status, returned) = std::thread::spawn(move || {
        let mut returned = 0_u64;
        let status = run([
            fiber_address,
            0x9999,
            (&raw mut returned).addr() as u64,
            0,
            0,
            0,
        ]);
        (status, returned)
    })
    .join()
    .expect("the import-frame resume must not fault");
    assert_eq!(status, 0);
    assert_eq!(returned, 0x2222);
    assert_eq!(OBSERVED.lock().unwrap().as_slice(), &[(0, 0x9999)]);

    // SAFETY: no fiber can execute the stub after the second yield has returned to its thread.
    assert_eq!(unsafe { libc::munmap(mapping, PAGE_SIZE) }, 0);
}
