// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{GUEST_CONDITION_TEST, sce_kernel_error, wait_for_guest_condition_waiters};

#[test]
fn semaphore_wait_honors_timeouts_and_rejects_impossible_counts() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    const SEMAPHORE: u32 = 0x7fff_5100;
    let sync = crate::guest_sync();
    sync.0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .semaphores
        .insert(
            SEMAPHORE,
            crate::GuestSemaphoreState {
                count: 0,
                maximum: 1,
            },
        );

    let mut timeout_microseconds = 2_000_u32;
    assert_eq!(
        crate::guest_semaphore_wait([
            u64::from(SEMAPHORE),
            1,
            (&raw mut timeout_microseconds).addr() as u64,
            0,
            0,
            0,
        ]),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_TIMED_OUT)
    );
    assert_eq!(timeout_microseconds, 0);
    assert_eq!(
        crate::guest_semaphore_wait([u64::from(SEMAPHORE), 2, 0, 0, 0, 0]),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT),
        "a request larger than the semaphore maximum must not park forever"
    );

    {
        let mut state = sync
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.semaphores.get_mut(&SEMAPHORE).unwrap().count = 1;
    }
    let mut available_timeout = 50_000_u32;
    assert_eq!(
        crate::guest_semaphore_wait([
            u64::from(SEMAPHORE),
            1,
            (&raw mut available_timeout).addr() as u64,
            0,
            0,
            0,
        ]),
        0
    );
    assert!(available_timeout <= 50_000);
    sync.0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .semaphores
        .remove(&SEMAPHORE);
}

#[test]
fn semaphore_signal_reports_overflow_without_losing_the_count() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    const SEMAPHORE: u32 = 0x7fff_5200;
    let sync = crate::guest_sync();
    sync.0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .semaphores
        .insert(
            SEMAPHORE,
            crate::GuestSemaphoreState {
                count: 1,
                maximum: 1,
            },
        );

    assert_eq!(
        crate::dispatch_guest_sync_import(
            "4czppHBiriw",
            [u64::from(SEMAPHORE), 1, 0, 0, 0, 0],
            None,
        ),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
    assert_eq!(
        sync.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .semaphores[&SEMAPHORE]
            .count,
        1
    );
    sync.0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .semaphores
        .remove(&SEMAPHORE);
}

#[test]
fn condition_signal_selects_one_waiter_and_broadcast_selects_the_rest() {
    let mut condition = crate::GuestConditionState::default();
    let mutex = 0x1234;
    let first = 1;
    let second = 2;
    let third = 3;
    assert!(condition.register_waiter(first, mutex));
    assert!(condition.register_waiter(second, mutex));
    assert!(condition.register_waiter(third, mutex));
    assert!(!condition.register_waiter(4, mutex + 1));

    assert_eq!(condition.release_one(), Some(first));
    assert_eq!(
        condition.waiters.iter().copied().collect::<Vec<_>>(),
        [second, third]
    );
    assert_eq!(condition.bound_mutex, Some(mutex));

    assert_eq!(
        condition.release_all().into_iter().collect::<Vec<_>>(),
        [second, third]
    );
    assert!(condition.waiters.is_empty());
    assert_eq!(condition.bound_mutex, None);
    assert!(condition.is_idle());

    assert_eq!(condition.release_one(), None, "signals are not retained");
    let future = 5;
    assert!(condition.register_waiter(future, mutex + 1));
    condition.cancel_wait(future);
    assert!(condition.is_idle());
    assert_eq!(condition.bound_mutex, None);
}

#[test]
fn rwlock_tracks_recursive_read_ownership_and_excludes_writers() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let address = 0x7fff_0000_0080;
    let guest_thread = 0x1234_5678;
    crate::set_current_guest_thread_handle(guest_thread);
    assert_eq!(crate::guest_rwlock_init(address), 0);
    assert_eq!(crate::guest_rwlock_lock(address, false, false), 0);
    assert_eq!(crate::guest_rwlock_lock(address, false, false), 0);
    {
        let state = crate::guest_sync()
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let lock = state.rwlocks.get(&address).expect("initialized rwlock");
        let reader = lock
            .readers
            .get(&std::thread::current().id())
            .expect("current reader");
        assert_eq!(reader.guest_thread, guest_thread);
        assert_eq!(reader.recursion, 2);
    }
    assert_eq!(
        crate::guest_rwlock_lock(address, true, true),
        u64::from(libc::EBUSY.cast_unsigned())
    );
    assert_eq!(
        crate::guest_rwlock_destroy(address),
        u64::from(libc::EBUSY.cast_unsigned())
    );
    assert_eq!(crate::guest_rwlock_unlock(address), 0);
    assert_eq!(crate::guest_rwlock_unlock(address), 0);
    assert_eq!(crate::guest_rwlock_lock(address, true, false), 0);
    {
        let state = crate::guest_sync()
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let lock = state.rwlocks.get(&address).expect("initialized rwlock");
        assert_eq!(lock.writer_guest_thread, guest_thread);
    }
    assert_eq!(
        crate::guest_rwlock_lock(address, false, true),
        u64::from(libc::EBUSY.cast_unsigned())
    );
    assert_eq!(crate::guest_rwlock_unlock(address), 0);
    assert_eq!(crate::guest_rwlock_destroy(address), 0);
    crate::set_current_guest_thread_handle(0);
}

#[test]
fn rwlock_keeps_reader_priority_while_a_writer_is_waiting() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let address = 0x7fff_0000_0180;
    assert_eq!(crate::guest_rwlock_init(address), 0);
    assert_eq!(crate::guest_rwlock_lock(address, false, false), 0);

    let (writer_result, receiver) = std::sync::mpsc::channel();
    let writer = std::thread::spawn(move || {
        let result = crate::guest_rwlock_lock(address, true, false);
        writer_result
            .send(result)
            .expect("test receiver remains live");
        if result == 0 {
            assert_eq!(crate::guest_rwlock_unlock(address), 0);
        }
    });

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        let waiting = crate::guest_sync()
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .rwlocks
            .get(&address)
            .map_or(0, |lock| lock.waiting_writers);
        if waiting == 1 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "guest rwlock writer did not block"
        );
        std::thread::yield_now();
    }

    assert_eq!(
        crate::guest_rwlock_lock(address, false, true),
        0,
        "a queued writer must not block a recursive reader"
    );
    assert_eq!(crate::guest_rwlock_unlock(address), 0);

    let reader = std::thread::spawn(move || {
        let result = crate::guest_rwlock_lock(address, false, true);
        if result == 0 {
            assert_eq!(crate::guest_rwlock_unlock(address), 0);
        }
        result
    });
    assert_eq!(
        reader.join().expect("guest reader thread"),
        0,
        "a queued writer must not block a new reader"
    );
    assert!(
        receiver
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err(),
        "the writer acquired before the active reader released"
    );

    assert_eq!(crate::guest_rwlock_unlock(address), 0);
    assert_eq!(
        receiver
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("writer acquired after readers released"),
        0
    );
    writer.join().expect("guest writer thread");
    assert_eq!(crate::guest_rwlock_destroy(address), 0);
}

#[test]
fn mutex_types_apply_gen5_same_thread_relock_semantics() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let cases = [
        (0x7fff_0000_1100, 1, libc::EDEADLK),
        (0x7fff_0000_1200, 3, libc::EBUSY),
        (0x7fff_0000_1300, 4, libc::EDEADLK),
    ];
    for (address, mutex_type, expected) in cases {
        assert_eq!(
            crate::guest_mutex_init(
                address,
                crate::PthreadMutexAttrState {
                    mutex_type,
                    ..crate::PthreadMutexAttrState::default()
                }
            ),
            0
        );
        assert_eq!(crate::guest_mutex_lock(address, false), 0);
        assert_eq!(
            crate::guest_mutex_lock(address, mutex_type == 3),
            u64::from(expected.cast_unsigned())
        );
        assert_eq!(crate::guest_mutex_unlock(address), 0);
        assert_eq!(crate::guest_mutex_destroy(address), 0);
    }

    let recursive = 0x7fff_0000_1400;
    assert_eq!(
        crate::guest_mutex_init(
            recursive,
            crate::PthreadMutexAttrState {
                mutex_type: 2,
                ..crate::PthreadMutexAttrState::default()
            }
        ),
        0
    );
    assert_eq!(crate::guest_mutex_lock(recursive, false), 0);
    assert_eq!(crate::guest_mutex_lock(recursive, false), 0);
    assert_eq!(crate::guest_mutex_unlock(recursive), 0);
    assert_eq!(
        crate::guest_mutex_destroy(recursive),
        u64::from(libc::EBUSY.cast_unsigned())
    );
    assert_eq!(crate::guest_mutex_unlock(recursive), 0);
    assert_eq!(crate::guest_mutex_destroy(recursive), 0);
}

#[test]
fn mutex_unlock_rejects_missing_and_non_owner_mutexes() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let missing = 0x7fff_0000_1500;
    assert_eq!(
        crate::guest_mutex_unlock(missing),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED)
    );

    let mutex = 0x7fff_0000_1600;
    assert_eq!(
        crate::guest_mutex_init(mutex, crate::PthreadMutexAttrState::default()),
        0
    );
    assert_eq!(crate::guest_mutex_lock(mutex, false), 0);
    assert_eq!(
        std::thread::spawn(move || crate::guest_mutex_unlock(mutex))
            .join()
            .expect("non-owner unlock thread"),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED)
    );
    assert_eq!(crate::guest_mutex_unlock(mutex), 0);
    assert_eq!(crate::guest_mutex_destroy(mutex), 0);
}

#[test]
fn mutex_init_installs_the_guest_visible_object_libc_updates() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut storage = 0_u64;
    let address = (&raw mut storage).addr() as u64;
    assert_eq!(
        crate::guest_mutex_init(address, crate::PthreadMutexAttrState::default()),
        0
    );
    assert_ne!(storage, 0);
    let handle = storage;
    assert_eq!(
        crate::read_guest_u32(handle + 0x20) & 0xff,
        crate::PthreadMutexAttrState::default()
            .mutex_type
            .cast_unsigned()
    );
    assert_eq!(
        crate::read_guest_u32(handle + 0x3c),
        crate::PthreadMutexAttrState::default()
            .protocol
            .cast_unsigned()
    );

    // PS5 libc performs this exact update after scePthreadMutexInit succeeds. The returned
    // opaque object therefore has to be real writable guest-visible storage.
    let flags = crate::read_guest_u32(handle + 0x20);
    crate::write_guest_u32(handle + 0x20, flags | 0x100);
    assert_eq!(crate::read_guest_u32(handle + 0x20) & 0x100, 0x100);

    // Both the public storage address and libc's resolved opaque pointer identify the same
    // mutex in different generations of the SDK.
    assert_eq!(crate::guest_mutex_lock(handle, false), 0);
    assert_eq!(crate::guest_mutex_unlock(address), 0);
    assert_eq!(crate::guest_mutex_destroy(handle), 0);
    assert_eq!(storage, 0);
}

#[test]
fn timed_condition_wait_reacquires_the_guest_mutex_before_returning() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let condition = 0x7fff_0000_0100;
    let mutex = 0x7fff_0000_0200;
    assert_eq!(crate::guest_condition_init(condition), 0);
    assert_eq!(
        crate::guest_mutex_init(
            mutex,
            crate::PthreadMutexAttrState {
                mutex_type: 2,
                ..crate::PthreadMutexAttrState::default()
            }
        ),
        0
    );
    assert_eq!(crate::guest_mutex_lock(mutex, false), 0);
    assert_eq!(crate::guest_mutex_lock(mutex, false), 0);

    assert_eq!(
        crate::guest_condition_wait(condition, mutex, Some(0)),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_TIMED_OUT)
    );
    {
        let entry = crate::guest_mutex_entry(mutex, false).expect("test mutex");
        let guest_mutex = entry
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(guest_mutex.owner, Some(std::thread::current().id()));
        assert_eq!(guest_mutex.recursion, 2);
        let state = crate::guest_sync()
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(
            state
                .conditions
                .get(&condition)
                .is_some_and(crate::GuestConditionState::is_idle)
        );
    }

    assert_eq!(crate::guest_mutex_unlock(mutex), 0);
    assert_eq!(crate::guest_mutex_unlock(mutex), 0);
    assert_eq!(crate::guest_mutex_destroy(mutex), 0);
    assert_eq!(crate::guest_condition_destroy(condition), 0);
}

#[test]
fn condition_wait_delivers_interruption_before_reacquiring_guest_mutex() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    const THREAD: u64 = 0x7fff_3000;
    let condition = 0x7fff_0000_0500;
    let mutex = 0x7fff_0000_0600;
    assert_eq!(crate::guest_condition_init(condition), 0);
    assert_eq!(
        crate::guest_mutex_init(mutex, crate::PthreadMutexAttrState::default()),
        0
    );

    let (handler_entered_tx, handler_entered_rx) = std::sync::mpsc::channel();
    let (resume_handler_tx, resume_handler_rx) = std::sync::mpsc::channel();
    let (finished_tx, finished_rx) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        crate::set_current_guest_thread_handle(THREAD);
        assert_eq!(crate::guest_mutex_lock(mutex, false), 0);
        assert_eq!(
            crate::guest_condition_wait_with_interrupt(condition, mutex, None, || {
                handler_entered_tx
                    .send(())
                    .expect("test receiver remains live");
                resume_handler_rx
                    .recv()
                    .expect("test suspension controller remains live");
            }),
            0
        );
        assert_eq!(crate::guest_mutex_unlock(mutex), 0);
        crate::set_current_guest_thread_handle(0);
        finished_tx.send(()).expect("test receiver remains live");
    });
    wait_for_guest_condition_waiters(condition, 1);

    // Model stop-the-world with another guest thread holding this condition's mutex. The
    // interrupted waiter must enter its suspension handler even though mutex reacquisition
    // cannot yet succeed.
    assert_eq!(crate::guest_mutex_lock(mutex, false), 0);
    crate::guest_exception_delivery()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .pending
        .insert(
            THREAD,
            crate::PendingGuestException {
                handler: 1,
                exception: 30,
            },
        );
    crate::guest_sync().1.notify_all();
    handler_entered_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("interruption handler ran before mutex reacquisition");

    resume_handler_tx
        .send(())
        .expect("waiter remains in its handler");
    assert!(
        finished_rx
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err(),
        "condition wait returned without reacquiring its mutex"
    );
    assert_eq!(crate::guest_mutex_unlock(mutex), 0);
    finished_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("resumed waiter reacquired the mutex");
    waiter.join().expect("guest waiter thread");

    crate::guest_exception_delivery()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .pending
        .remove(&THREAD);
    assert_eq!(crate::guest_mutex_destroy(mutex), 0);
    assert_eq!(crate::guest_condition_destroy(condition), 0);
}

#[test]
fn condition_wait_errors_use_the_selected_pthread_abi() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let condition = 0x7fff_0000_0900;
    let mutex = 0x7fff_0000_0a00;
    assert_eq!(crate::guest_condition_init(condition), 0);
    assert_eq!(
        crate::guest_mutex_init(mutex, crate::PthreadMutexAttrState::default()),
        0
    );
    assert_eq!(
        crate::dispatch_guest_sync_import("WKAXJ4XBPQ4", [condition, mutex, 0, 0, 0, 0], None),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED)
    );
    assert_eq!(
        crate::dispatch_guest_sync_import("Op8TBGY5KHg", [condition, mutex, 0, 0, 0, 0], None),
        u64::from(libc::EPERM.cast_unsigned())
    );
    assert_eq!(crate::guest_mutex_destroy(mutex), 0);
    assert_eq!(crate::guest_condition_destroy(condition), 0);
}

#[test]
fn condition_signal_wakes_exactly_one_blocked_guest_thread() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let condition = 0x7fff_0000_1100;
    let mutex = 0x7fff_0000_1200;
    assert_eq!(crate::guest_condition_init(condition), 0);
    assert_eq!(
        crate::guest_mutex_init(mutex, crate::PthreadMutexAttrState::default()),
        0
    );
    let (finished, receiver) = std::sync::mpsc::channel();
    let waiters = (0..2)
        .map(|waiter| {
            let finished = finished.clone();
            std::thread::spawn(move || {
                assert_eq!(crate::guest_mutex_lock(mutex, false), 0);
                assert_eq!(crate::guest_condition_wait(condition, mutex, None), 0);
                assert_eq!(crate::guest_mutex_unlock(mutex), 0);
                finished.send(waiter).expect("test receiver remains live");
            })
        })
        .collect::<Vec<_>>();
    drop(finished);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        let waiting = crate::guest_sync()
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .conditions
            .get(&condition)
            .map_or(0, |condition| condition.waiters.len());
        if waiting == 2 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "guest waiters did not block"
        );
        std::thread::yield_now();
    }

    assert_eq!(
        crate::dispatch_guest_sync_import("kDh-NfxgMtE", [condition, 0, 0, 0, 0, 0], None),
        0
    );
    let first = receiver
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("one signaled waiter");
    let early_second = receiver
        .recv_timeout(std::time::Duration::from_millis(50))
        .ok();
    assert_eq!(
        crate::dispatch_guest_sync_import("JGgj7Uvrl+A", [condition, 0, 0, 0, 0, 0], None),
        0
    );
    let second = early_second.unwrap_or_else(|| {
        receiver
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("broadcast waiter")
    });
    assert_ne!(first, second);
    assert!(
        early_second.is_none(),
        "signal acquired broadcast semantics"
    );
    for waiter in waiters {
        waiter.join().expect("guest waiter thread");
    }
    assert_eq!(crate::guest_mutex_destroy(mutex), 0);
    assert_eq!(crate::guest_condition_destroy(condition), 0);
}

#[test]
fn condition_signal_allows_exactly_one_predicate_guarded_task_consumption() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let condition = 0x7fff_0000_1900;
    let mutex = 0x7fff_0000_1a00;
    assert_eq!(crate::guest_condition_init(condition), 0);
    assert_eq!(
        crate::guest_mutex_init(mutex, crate::PthreadMutexAttrState::default()),
        0
    );
    let jobs = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let stopping = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (consumed, receiver) = std::sync::mpsc::channel();
    let workers = (0..3)
        .map(|worker| {
            let jobs = std::sync::Arc::clone(&jobs);
            let stopping = std::sync::Arc::clone(&stopping);
            let consumed = consumed.clone();
            std::thread::spawn(move || {
                loop {
                    assert_eq!(crate::guest_mutex_lock(mutex, false), 0);
                    while jobs.load(std::sync::atomic::Ordering::SeqCst) == 0
                        && !stopping.load(std::sync::atomic::Ordering::SeqCst)
                    {
                        assert_eq!(crate::guest_condition_wait(condition, mutex, None), 0);
                    }
                    if stopping.load(std::sync::atomic::Ordering::SeqCst) {
                        assert_eq!(crate::guest_mutex_unlock(mutex), 0);
                        break;
                    }
                    assert_eq!(jobs.fetch_sub(1, std::sync::atomic::Ordering::SeqCst), 1);
                    assert_eq!(crate::guest_mutex_unlock(mutex), 0);
                    consumed.send(worker).expect("test receiver remains live");
                }
            })
        })
        .collect::<Vec<_>>();
    drop(consumed);
    wait_for_guest_condition_waiters(condition, 3);

    assert_eq!(crate::guest_mutex_lock(mutex, false), 0);
    jobs.store(1, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(crate::guest_condition_notify(condition, false), 0);
    assert_eq!(crate::guest_mutex_unlock(mutex), 0);
    receiver
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("one worker consumed the queued task");
    assert!(
        receiver
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err(),
        "one signal cannot produce a second task consumption"
    );
    wait_for_guest_condition_waiters(condition, 3);

    assert_eq!(crate::guest_mutex_lock(mutex, false), 0);
    stopping.store(true, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(crate::guest_condition_notify(condition, true), 0);
    assert_eq!(crate::guest_mutex_unlock(mutex), 0);
    for worker in workers {
        worker.join().expect("guest worker thread");
    }
    assert_eq!(jobs.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(crate::guest_mutex_destroy(mutex), 0);
    assert_eq!(crate::guest_condition_destroy(condition), 0);
}

#[test]
fn selected_condition_waiter_survives_legal_condition_destruction() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let condition = 0x7fff_0000_2100;
    let mutex = 0x7fff_0000_2200;
    assert_eq!(crate::guest_condition_init(condition), 0);
    assert_eq!(
        crate::guest_mutex_init(mutex, crate::PthreadMutexAttrState::default()),
        0
    );
    let (events, receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        assert_eq!(crate::guest_mutex_lock(mutex, false), 0);
        events.send("locked").expect("test receiver remains live");
        assert_eq!(crate::guest_condition_wait(condition, mutex, None), 0);
        events.send("returned").expect("test receiver remains live");
        assert_eq!(crate::guest_mutex_unlock(mutex), 0);
    });
    assert_eq!(
        receiver
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("waiter acquired guest mutex"),
        "locked"
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        let waiting = crate::guest_sync()
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .conditions
            .get(&condition)
            .map_or(0, |condition| condition.waiters.len());
        if waiting == 1 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "guest waiter did not block"
        );
        std::thread::yield_now();
    }

    assert_eq!(crate::guest_mutex_lock(mutex, false), 0);
    assert_eq!(
        crate::guest_condition_destroy(condition),
        sce_kernel_error(crate::SCE_KERNEL_ERROR_BUSY)
    );
    assert_eq!(crate::guest_condition_notify(condition, false), 0);
    assert_eq!(crate::guest_condition_destroy(condition), 0);
    assert!(
        receiver
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err(),
        "selected waiter must reacquire its mutex before returning"
    );
    assert_eq!(crate::guest_mutex_unlock(mutex), 0);
    assert_eq!(
        receiver
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("selected waiter reacquired its mutex"),
        "returned"
    );
    waiter.join().expect("guest waiter thread");
    assert_eq!(crate::guest_mutex_destroy(mutex), 0);
}
