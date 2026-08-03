// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::sync::atomic::Ordering;

use std::time::{Duration, Instant};

use super::{
    DEFAULT_FLIP_INTERVAL_MICROSECONDS, GPU_EVENT_SINK_TEST, ProbeDispatchState,
    VIDEO_OUT_FLIP_INTERVAL, flip_interval_microseconds, pace_video_out_flip, read_guest_u16,
    read_guest_u32, read_guest_u64, targetless_scanout_test_video, video_out_submit_flip,
};

#[test]
fn gpu_publication_waits_until_the_dispatch_state_can_be_released() {
    let _event_guard = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    remu_gpu::clear_event_sink();
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender);
    let _ = remu_gpu::take_dispatch_telemetry_sample();

    crate::begin_deferred_gpu_publication();
    assert!(crate::publish_gpu_event(remu_gpu::GpuEvent::Shutdown));
    assert!(
        receiver.try_recv().is_err(),
        "event must not enter the bounded backend channel while dispatch state is locked"
    );

    let publication = crate::take_deferred_gpu_publication();
    assert!(crate::flush_deferred_gpu_publication(publication));
    let telemetry = remu_gpu::take_dispatch_telemetry_sample();
    assert_eq!(telemetry.gpu_publication_wait.count, 1);
    for phase in [
        remu_gpu::GpuPublicationPhase::TicketOrderWait,
        remu_gpu::GpuPublicationPhase::BarrierAcquire,
        remu_gpu::GpuPublicationPhase::ChannelSetup,
        remu_gpu::GpuPublicationPhase::QueueFifoWait,
        remu_gpu::GpuPublicationPhase::BoundedSend,
        remu_gpu::GpuPublicationPhase::Residual,
    ] {
        assert_eq!(
            telemetry
                .gpu_publication_phase(remu_gpu::GpuPublicationRoute::Mixed, phase)
                .count,
            1,
            "each exact phase receives one sample for the mixed publication"
        );
        assert_eq!(
            telemetry
                .gpu_publication_phase(remu_gpu::GpuPublicationRoute::QueueOnly, phase)
                .count,
            0
        );
    }
    assert!(matches!(receiver.recv(), Ok(remu_gpu::GpuEvent::Shutdown)));
    remu_gpu::record_gpu_event_consumed();
    remu_gpu::clear_event_sink();
}

#[test]
fn gpu_publication_summary_separates_queue_work_from_outer_events() {
    let publication = crate::DeferredGpuPublication {
        ticket: Some(7),
        events: vec![
            remu_gpu::GpuEvent::QueueSubmission {
                queue_id: 0,
                events: vec![
                    remu_gpu::GpuEvent::SubmissionBoundary(1),
                    remu_gpu::GpuEvent::SubmissionBoundary(2),
                ],
            },
            remu_gpu::GpuEvent::QueueSubmission {
                queue_id: 33,
                events: vec![remu_gpu::GpuEvent::SubmissionBoundary(3)],
            },
            remu_gpu::GpuEvent::Shutdown,
        ],
    };

    let summary = publication.summary();
    assert_eq!(summary.outer_events, 3);
    assert_eq!(summary.work_items, 4);
    assert_eq!(summary.queue_work_items.get(&0), Some(&2));
    assert_eq!(summary.queue_work_items.get(&33), Some(&1));
}

#[test]
fn gpu_publication_backpressure_releases_dispatch_state_and_preserves_ticket_order() {
    let _event_guard = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    remu_gpu::clear_event_sink();
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender.clone());
    for _ in 0..remu_gpu::GPU_EVENT_QUEUE_CAPACITY {
        sender
            .try_send(remu_gpu::GpuEvent::Shutdown)
            .expect("filler event fits the bounded backend channel");
    }

    crate::begin_deferred_gpu_publication();
    assert!(crate::publish_gpu_event(
        remu_gpu::GpuEvent::SubmissionBoundary(0xa11)
    ));
    let first = crate::take_deferred_gpu_publication();
    crate::begin_deferred_gpu_publication();
    assert!(crate::publish_gpu_event(
        remu_gpu::GpuEvent::SubmissionBoundary(0xb22)
    ));
    let second = crate::take_deferred_gpu_publication();

    let (first_started, first_started_rx) = std::sync::mpsc::channel();
    let (first_done, first_done_rx) = std::sync::mpsc::channel();
    let first_thread = std::thread::spawn(move || {
        first_started.send(()).expect("test receiver remains alive");
        first_done
            .send(crate::flush_deferred_gpu_publication(first))
            .expect("test receiver remains alive");
    });
    first_started_rx
        .recv()
        .expect("first publisher reaches the flush");
    assert_eq!(
        first_done_rx.recv_timeout(Duration::from_millis(20)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout),
        "a full backend channel must retain bounded producer backpressure"
    );

    let dispatch =
        crate::PROBE_DISPATCH.get_or_init(|| std::sync::RwLock::new(ProbeDispatchState::default()));
    assert!(
        dispatch.try_write().is_ok(),
        "a publisher blocked on backend capacity must not retain the process-wide dispatch state"
    );

    let (second_done, second_done_rx) = std::sync::mpsc::channel();
    let second_thread = std::thread::spawn(move || {
        second_done
            .send(crate::flush_deferred_gpu_publication(second))
            .expect("test receiver remains alive");
    });
    assert_eq!(
        second_done_rx.recv_timeout(Duration::from_millis(20)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout),
        "a later ticket must not overtake the blocked publisher"
    );

    assert_eq!(receiver.recv(), Ok(remu_gpu::GpuEvent::Shutdown));
    assert_eq!(
        first_done_rx.recv_timeout(Duration::from_secs(1)),
        Ok(true),
        "receiver progress must release the first publisher"
    );
    assert_eq!(
        second_done_rx.recv_timeout(Duration::from_millis(20)),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout),
        "the second publisher remains backpressured until another slot opens"
    );
    for _ in 1..remu_gpu::GPU_EVENT_QUEUE_CAPACITY {
        assert_eq!(receiver.recv(), Ok(remu_gpu::GpuEvent::Shutdown));
    }
    assert_eq!(
        receiver.recv(),
        Ok(remu_gpu::GpuEvent::SubmissionBoundary(0xa11)),
        "the first publication must retain its ticket order"
    );
    remu_gpu::record_gpu_event_consumed();
    remu_gpu::record_gpu_work_item_processed();
    assert_eq!(
        second_done_rx.recv_timeout(Duration::from_secs(1)),
        Ok(true),
        "receiver progress must release the second publisher"
    );
    assert_eq!(
        receiver.recv(),
        Ok(remu_gpu::GpuEvent::SubmissionBoundary(0xb22)),
        "the later publication must follow the earlier ticket"
    );
    remu_gpu::record_gpu_event_consumed();
    remu_gpu::record_gpu_work_item_processed();

    first_thread.join().expect("first publisher exits cleanly");
    second_thread
        .join()
        .expect("second publisher exits cleanly");
    remu_gpu::clear_event_sink();
}

#[test]
fn gpu_publication_tickets_admit_an_independent_queue_before_graphics_drains() {
    let _event_guard = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    remu_gpu::clear_event_sink();
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender.clone());
    for _ in 0..remu_gpu::GPU_EVENT_QUEUE_CAPACITY {
        sender
            .try_send(remu_gpu::GpuEvent::Shutdown)
            .expect("filler event fits the bounded backend channel");
    }

    let q0_chunk_count = remu_gpu::GPU_EVENT_QUEUE_CAPACITY + 2;
    crate::begin_deferred_gpu_publication();
    assert!(crate::publish_gpu_event(
        remu_gpu::GpuEvent::QueueSubmission {
            queue_id: 0,
            events: vec![
                remu_gpu::GpuEvent::SubmissionBoundary(0xa11);
                q0_chunk_count * remu_gpu::GPU_EVENT_SUBMISSION_CHUNK_SIZE
            ],
        }
    ));
    let graphics = crate::take_deferred_gpu_publication();
    crate::begin_deferred_gpu_publication();
    assert!(crate::publish_gpu_event(
        remu_gpu::GpuEvent::QueueSubmission {
            queue_id: 33,
            events: vec![remu_gpu::GpuEvent::SubmissionBoundary(0xb22)],
        }
    ));
    let compute = crate::take_deferred_gpu_publication();

    let (publication_done, publication_done_rx) = std::sync::mpsc::channel();
    let (graphics_admitted, graphics_admitted_rx) = std::sync::mpsc::channel();
    let graphics_done = publication_done.clone();
    let graphics_thread = std::thread::spawn(move || {
        graphics_done
            .send((
                0,
                crate::flush_deferred_gpu_publication_with_admission(graphics, || {
                    graphics_admitted
                        .send(())
                        .expect("test receiver remains alive");
                }),
            ))
            .expect("test receiver remains alive");
    });
    graphics_admitted_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("graphics receives the first process and queue turn");

    let compute_done = publication_done;
    let (compute_admitted, compute_admitted_rx) = std::sync::mpsc::channel();
    let compute_thread = std::thread::spawn(move || {
        compute_done
            .send((
                33,
                crate::flush_deferred_gpu_publication_with_admission(compute, || {
                    compute_admitted
                        .send(())
                        .expect("test receiver remains alive");
                }),
            ))
            .expect("test receiver remains alive");
    });
    compute_admitted_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("compute is admitted while graphics remains backpressured");

    // The fair coordinator owns the send turn explicitly: q0 takes the first released slot, then
    // yields before attempting another chunk, so q33 deterministically takes the second.
    for _ in 0..2 {
        assert_eq!(receiver.recv(), Ok(remu_gpu::GpuEvent::Shutdown));
    }
    assert_eq!(
        publication_done_rx.recv_timeout(Duration::from_secs(1)),
        Ok((33, true)),
        "the later compute ticket must publish before the graphics ticket fully drains"
    );

    let expected_events = remu_gpu::GPU_EVENT_QUEUE_CAPACITY + q0_chunk_count + 1;
    let mut received_events = 2;
    while received_events != expected_events {
        receiver.recv().expect("remaining bounded event");
        received_events += 1;
    }
    assert_eq!(
        publication_done_rx.recv_timeout(Duration::from_secs(1)),
        Ok((0, true))
    );
    graphics_thread
        .join()
        .expect("graphics publisher exits cleanly");
    compute_thread
        .join()
        .expect("compute publisher exits cleanly");
    remu_gpu::clear_event_sink();
}

#[test]
fn mixed_gpu_publication_is_a_full_barrier_around_queue_only_work() {
    let _event_guard = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    remu_gpu::clear_event_sink();
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender.clone());
    for _ in 0..remu_gpu::GPU_EVENT_QUEUE_CAPACITY {
        sender
            .try_send(remu_gpu::GpuEvent::HideHostSplash)
            .expect("filler event fits the bounded backend channel");
    }

    crate::begin_deferred_gpu_publication();
    assert!(crate::publish_gpu_event(
        remu_gpu::GpuEvent::QueueSubmission {
            queue_id: 0,
            events: vec![
                remu_gpu::GpuEvent::SubmissionBoundary(1);
                remu_gpu::GPU_EVENT_SUBMISSION_CHUNK_SIZE * 3
            ],
        }
    ));
    let graphics = crate::take_deferred_gpu_publication();
    crate::begin_deferred_gpu_publication();
    assert!(crate::publish_gpu_event(
        remu_gpu::GpuEvent::QueueSubmission {
            queue_id: 33,
            events: vec![remu_gpu::GpuEvent::SubmissionBoundary(2)],
        }
    ));
    assert!(crate::publish_gpu_event(remu_gpu::GpuEvent::Shutdown));
    let mixed = crate::take_deferred_gpu_publication();

    let (graphics_admitted, graphics_admitted_rx) = std::sync::mpsc::channel();
    let graphics_thread = std::thread::spawn(move || {
        crate::flush_deferred_gpu_publication_with_admission(graphics, || {
            graphics_admitted
                .send(())
                .expect("test receiver remains alive");
        })
    });
    graphics_admitted_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("graphics is admitted before the full barrier");

    let (mixed_started, mixed_started_rx) = std::sync::mpsc::channel();
    let (mixed_admitted, mixed_admitted_rx) = std::sync::mpsc::channel();
    let mixed_thread = std::thread::spawn(move || {
        mixed_started.send(()).expect("test receiver remains alive");
        crate::flush_deferred_gpu_publication_with_admission(mixed, || {
            mixed_admitted
                .send(())
                .expect("test receiver remains alive");
        })
    });
    mixed_started_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("mixed publisher reaches its flush");

    for _ in 0..remu_gpu::GPU_EVENT_QUEUE_CAPACITY {
        assert_eq!(receiver.recv(), Ok(remu_gpu::GpuEvent::HideHostSplash));
    }
    for _ in 0..3 {
        let remu_gpu::GpuEvent::QueueSubmission { queue_id, events } =
            receiver.recv().expect("graphics chunk before the barrier")
        else {
            panic!("mixed publication overtook active graphics");
        };
        assert_eq!(queue_id, 0);
        assert!(
            events
                .iter()
                .all(|event| event == &remu_gpu::GpuEvent::SubmissionBoundary(1))
        );
    }
    mixed_admitted_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("mixed publication enters only after graphics releases its read guard");
    assert!(matches!(
        receiver.recv(),
        Ok(remu_gpu::GpuEvent::QueueSubmission { queue_id: 33, .. })
    ));
    assert_eq!(receiver.recv(), Ok(remu_gpu::GpuEvent::Shutdown));
    assert!(
        graphics_thread
            .join()
            .expect("graphics publisher exits cleanly")
    );
    assert!(mixed_thread.join().expect("mixed publisher exits cleanly"));
    remu_gpu::clear_event_sink();
}

#[test]
fn agc_completion_tracking_requires_a_published_queue_submission() {
    let _event_guard = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    remu_gpu::clear_event_sink();
    let mut state = ProbeDispatchState {
        agc_submission_count: 41,
        ..ProbeDispatchState::default()
    };

    crate::request_agc_submission_completion(&mut state, 0, Vec::new());
    crate::request_agc_compute_submission_completion(&mut state, 32, Vec::new());

    assert!(state.pending_agc_completions.is_empty());
    assert!(state.pending_agc_user_interrupts.is_empty());

    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender);
    crate::request_agc_submission_completion(&mut state, 0, Vec::new());
    state.agc_submission_count = 42;
    crate::request_agc_compute_submission_completion(&mut state, 32, Vec::new());
    assert_eq!(
        state
            .pending_agc_completions
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [(41, 0)]
    );
    assert_eq!(
        state
            .pending_agc_user_interrupts
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [42]
    );
    assert!(matches!(
        receiver.recv(),
        Ok(remu_gpu::GpuEvent::QueueSubmission {
            queue_id: 0,
            events,
        }) if events == [remu_gpu::GpuEvent::SubmissionBoundary(41)]
    ));
    assert!(matches!(
        receiver.recv(),
        Ok(remu_gpu::GpuEvent::QueueSubmission {
            queue_id: 33,
            events,
        }) if events == [remu_gpu::GpuEvent::SubmissionBoundary(42)]
    ));
    remu_gpu::clear_event_sink();
}

#[test]
fn containing_submission_sequence_is_attached_only_to_captured_flips() {
    let _event_guard = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    remu_gpu::clear_event_sink();
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender);
    let address = 0x0070_1234_5600;
    let mut state = ProbeDispatchState {
        agc_submission_count: 73,
        ..ProbeDispatchState::default()
    };
    let flip = remu_gpu::GpuEvent::Flip {
        address,
        buffer_index: 1,
        flip_argument: 0x1234,
        submission_sequence: None,
        width: 1_280,
        height: 720,
        pitch: 1_280,
        pixel_format: 0x8000_0000_2200_0000,
        tiling_mode: 0,
    };

    assert!(crate::request_agc_submission_completion(
        &mut state,
        7,
        vec![flip],
    ));
    let remu_gpu::GpuEvent::QueueSubmission { queue_id, events } =
        receiver.recv().expect("captured submission reaches presenter")
    else {
        panic!("captured work must retain its queue envelope");
    };
    assert_eq!(queue_id, 7);
    assert!(matches!(
        events.as_slice(),
        [
            remu_gpu::GpuEvent::Flip {
                submission_sequence: Some(73),
                ..
            },
            remu_gpu::GpuEvent::SubmissionBoundary(73),
        ]
    ));
    remu_gpu::record_gpu_event_consumed();
    remu_gpu::record_gpu_work_item_processed();
    remu_gpu::record_gpu_work_item_processed();
    remu_gpu::record_gpu_flip_completed(remu_gpu::GpuFlipCompletion {
        address,
        buffer_index: 1,
        flip_argument: 0x1234,
    });
    let _ = remu_gpu::take_gpu_completed_flips();
    remu_gpu::clear_event_sink();
}

#[test]
fn delivers_completed_agc_submissions_through_the_registered_graphics_equeue() {
    let _serial = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let completed = remu_gpu::gpu_completed_submission_sequence();
    let first = completed.saturating_add(1);
    let second = first.saturating_add(1);
    let mut state = ProbeDispatchState::default();
    let mut equeue = 0_u64;
    assert_eq!(
        crate::dispatch_stateless_import(
            "D0OdFMjp46I",
            [(&raw mut equeue).addr() as u64, 0, 0, 0, 0, 0],
            [0; 6],
        ),
        None,
        "the stateful equeue creator must not be shadowed by a placeholder"
    );
    assert_eq!(
        crate::kernel_create_equeue((&raw mut equeue).addr() as u64, &mut state),
        0
    );
    assert_eq!(equeue, 1);
    assert_eq!(
        crate::agc_driver_add_eq_event([equeue, 0x1234, 0xfeed_face, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        crate::agc_driver_add_eq_event([equeue, 0x20, 0xcafe_babe, 0, 0, 0], &mut state),
        0
    );
    state
        .pending_agc_completions
        .extend([(first, 0x1234), (second, 0x1234)]);
    remu_gpu::record_gpu_submission_completed(second);
    remu_gpu::record_gpu_submission_completed(first);

    let mut event = [0_u8; 0x40];
    let mut count = 0_u32;
    assert_eq!(
        crate::kernel_poll_equeue(
            [
                equeue,
                event.as_mut_ptr().addr() as u64,
                2,
                (&raw mut count).addr() as u64,
                0,
                0,
            ],
            &mut state,
        ),
        Ok(1)
    );
    assert_eq!(count, 1);
    assert_eq!(read_guest_u64(event.as_ptr().addr() as u64), 0x1234);
    assert_eq!(
        read_guest_u16(event.as_ptr().addr() as u64 + 8),
        crate::KERNEL_EVENT_FILTER_GRAPHICS.cast_unsigned()
    );
    assert_eq!(read_guest_u32(event.as_ptr().addr() as u64 + 0x0c), 2);
    assert_eq!(read_guest_u64(event.as_ptr().addr() as u64 + 0x10), 0x1234);
    assert_eq!(
        read_guest_u64(event.as_ptr().addr() as u64 + 0x18),
        0xfeed_face
    );
    assert!(
        state
            .pending_events
            .iter()
            .all(|pending| pending.ident != 0x20),
        "graphics completion must not wake an unrelated compute queue"
    );
    assert!(state.pending_agc_completions.is_empty());
    assert_eq!(state.agc_completion_telemetry.refreshed_graphics, 2);
    assert_eq!(
        state.agc_completion_telemetry.queued_graphics_notifications,
        2
    );
    assert_eq!(
        state
            .agc_completion_telemetry
            .delivered_graphics_notifications,
        2,
        "coalescing must preserve both logical completions in the diagnostic ledger"
    );
    assert_eq!(state.agc_completion_telemetry.unmatched_backend_ids, 0);
}

#[test]
fn routes_completed_compute_submissions_only_to_the_user_interrupt() {
    let _serial = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sequence = remu_gpu::gpu_completed_submission_sequence().saturating_add(1);
    let mut state = ProbeDispatchState::default();
    state.kernel_equeues.extend([1, 2]);
    assert_eq!(
        crate::kernel_register_event(
            &mut state,
            1,
            0x1800,
            crate::KERNEL_EVENT_FILTER_USER,
            0xfeed_face,
        ),
        0
    );
    assert_eq!(
        crate::kernel_register_event(
            &mut state,
            2,
            0x20,
            crate::KERNEL_EVENT_FILTER_GRAPHICS,
            0xcafe_babe,
        ),
        0
    );
    state.pending_agc_user_interrupts.push_back(sequence);
    remu_gpu::record_gpu_submission_completed(sequence);

    crate::refresh_completed_agc_submissions(&mut state);

    assert!(state.pending_agc_completions.is_empty());
    assert!(state.pending_agc_user_interrupts.is_empty());
    assert_eq!(state.pending_events.len(), 1);
    let user_event = state.pending_events.front().expect("AGC user interrupt");
    assert_eq!(user_event.equeue, 1);
    assert_eq!(user_event.ident, 0x1800);
    assert_eq!(user_event.fflags, 1);
    assert_eq!(user_event.user_data, 0xfeed_face);
    assert!(
        state
            .pending_events
            .iter()
            .all(|event| event.filter != crate::KERNEL_EVENT_FILTER_GRAPHICS),
        "compute completion must not also wake the owner's graphics event"
    );
    assert_eq!(state.agc_completion_telemetry.refreshed_compute, 1);
    assert_eq!(state.agc_completion_telemetry.queued_user_interrupts, 1);
    assert_eq!(state.agc_completion_telemetry.delivered_user_interrupts, 0);

    let mut delivered = [0_u8; 0x20];
    assert_eq!(
        crate::kernel_poll_equeue(
            [1, delivered.as_mut_ptr().addr() as u64, 1, 0, 0, 0],
            &mut state,
        ),
        Ok(1)
    );
    assert_eq!(state.agc_completion_telemetry.delivered_user_interrupts, 1);
}

#[test]
fn completion_ledger_identifies_a_backend_id_without_a_runtime_owner() {
    let _serial = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sequence = remu_gpu::gpu_completed_submission_sequence().saturating_add(1);
    let mut state = ProbeDispatchState::default();
    remu_gpu::record_gpu_submission_completed(sequence);

    crate::refresh_completed_agc_submissions(&mut state);

    assert_eq!(state.agc_completion_telemetry.unmatched_backend_ids, 1);
    assert_eq!(
        state.agc_completion_telemetry.last_unmatched_backend_id,
        Some(sequence)
    );
    assert!(state.pending_events.is_empty());
}

#[test]
fn direct_video_out_submit_flip_reaches_the_presenter() {
    let _event_guard = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let address = 0x0070_1234_5600_u64;
    let mut state = ProbeDispatchState {
        video_out: targetless_scanout_test_video(address),
        ..ProbeDispatchState::default()
    };
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender);

    let result = video_out_submit_flip([1, 0, 0, 0x1234, 0, 0], &mut state);
    crate::QUEUED_FLIPS.with(|queued| queued.set(0));

    assert_eq!(result, 0);
    assert_eq!(remu_gpu::gpu_flips_pending(), 1);
    let remu_gpu::GpuEvent::Flip {
        address: flipped_address,
        buffer_index,
        flip_argument,
        submission_sequence,
        width,
        height,
        ..
    } = receiver.recv().expect("direct VideoOut flip event")
    else {
        panic!("direct VideoOut submission must emit a presenter flip");
    };
    assert_eq!(flipped_address, address);
    assert_eq!(buffer_index, 0);
    assert_eq!(flip_argument, 0x1234);
    assert_eq!(submission_sequence, None);
    assert_eq!((width, height), (1_280, 720));
    assert_eq!(
        crate::video_out_is_flip_pending([1, 0, 0, 0, 0, 0], &mut state),
        1
    );
    remu_gpu::record_gpu_flip_completed(remu_gpu::GpuFlipCompletion {
        address,
        buffer_index,
        flip_argument,
    });
    assert_eq!(remu_gpu::gpu_flips_pending(), 0);
    crate::refresh_completed_video_out_flips(&mut state);
    assert_eq!(
        crate::video_out_is_flip_pending([1, 0, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(state.video_out.completed_flip_count, 1);
    assert!(receiver.try_recv().is_err());
    remu_gpu::clear_event_sink();
}

#[test]
fn gpu_progress_telemetry_keeps_a_prefix_and_logarithmic_checkpoints() {
    for sequence in 0..=8 {
        assert!(crate::trace_logarithmic_sequence(sequence, 8));
    }
    assert!(!crate::trace_logarithmic_sequence(9, 8));
    assert!(crate::trace_logarithmic_sequence(16, 8));
    assert!(!crate::trace_logarithmic_sequence(17, 8));
    assert!(crate::trace_logarithmic_sequence(1_u64 << 63, 8));
}

/// The selector chooses the display's hold time; an unrecognized value keeps the 60 Hz
/// default rather than disabling pacing.
#[test]
fn flip_rate_selectors_map_to_display_intervals() {
    assert_eq!(flip_interval_microseconds(0), 16_666);
    assert_eq!(flip_interval_microseconds(1), 33_333);
    assert_eq!(flip_interval_microseconds(2), 50_000);
    assert_eq!(flip_interval_microseconds(7), 16_666);
}

/// Pacing must gate the guest's frame loop: submitting flips in a tight loop has to take at
/// least the display interval per flip.
#[test]
fn paced_flips_are_held_for_the_display_interval() {
    VIDEO_OUT_FLIP_INTERVAL.store(4_000, Ordering::Relaxed);
    pace_video_out_flip();
    let start = Instant::now();
    for _ in 0..5 {
        pace_video_out_flip();
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed >= Duration::from_micros(4_000 * 4),
        "five paced flips returned in {elapsed:?}"
    );
    VIDEO_OUT_FLIP_INTERVAL.store(DEFAULT_FLIP_INTERVAL_MICROSECONDS, Ordering::Relaxed);
}
