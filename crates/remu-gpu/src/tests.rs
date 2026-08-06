// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    COMPUTE_NUM_THREAD_X_REGISTER, COMPUTE_NUM_THREAD_Y_REGISTER, COMPUTE_NUM_THREAD_Z_REGISTER,
    COMPUTE_PGM_HI_REGISTER, COMPUTE_PGM_LO_REGISTER, COMPUTE_PGM_RSRC2_REGISTER,
    COMPUTE_START_X_REGISTER, COMPUTE_START_Y_REGISTER, COMPUTE_START_Z_REGISTER,
    COMPUTE_TMPRING_SIZE_REGISTER, ComputeDispatchDecodeError, ComputeGrid,
    GPU_EVENT_QUEUE_CAPACITY, GPU_EVENT_SUBMISSION_CHUNK_SIZE, GpuCachePolicy,
    GpuEndOfPipeCbDbAction, GpuEvent, GpuFlipCompletion, GpuHostImage, GpuMemoryAcquire,
    GpuMemoryRelease, GpuMemoryReleaseValue, GpuMemoryWait, GpuRegister, GpuRenderBackendFlush,
    GpuWaitOperation, HostPadState, audio_output_frames,
    audio_progress_age_millis, audio_underrun_frames, clear_event_sink, compute_program_address,
    decode_compute_dispatch_direct, decode_compute_dispatch_indirect, decode_compute_launch,
    decode_compute_system_registers, emit_event, emit_event_with_admission,
    gpu_completed_submission_ids_snapshot, gpu_completed_submission_sequence, gpu_event_channel,
    gpu_events_in_flight, gpu_progress_age_millis, gpu_work_items_pending, host_pad_state,
    install_event_sink, install_gpu_completion_notifier, progress_telemetry_enabled,
    record_gpu_event_consumed, record_gpu_flip_completed, record_gpu_submission_completed,
    record_gpu_wait_producer, record_gpu_work_item_processed, report_audio_frames,
    set_host_pad_state, set_progress_telemetry_enabled, take_gpu_completed_flips,
    take_gpu_completed_submission_ids, take_gpu_wait_latch,
};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

static EVENT_SINK_TEST: Mutex<()> = Mutex::new(());
static GPU_COMPLETION_NOTIFICATIONS: AtomicUsize = AtomicUsize::new(0);

struct ProgressTelemetryReset;

impl Drop for ProgressTelemetryReset {
    fn drop(&mut self) {
        set_progress_telemetry_enabled(false);
    }
}

fn record_gpu_completion_notification() {
    GPU_COMPLETION_NOTIFICATIONS.fetch_add(1, Ordering::Relaxed);
}

#[test]
fn host_image_requires_exact_nonempty_rgba8_storage() {
    let image = GpuHostImage::new(2, 1, vec![0_u8; 8]).expect("complete RGBA image");
    assert_eq!((image.width(), image.height()), (2, 1));
    assert_eq!(image.rgba8().len(), 8);
    assert!(GpuHostImage::new(0, 1, Vec::<u8>::new()).is_none());
    assert!(GpuHostImage::new(2, 1, vec![0_u8; 7]).is_none());
    assert!(GpuHostImage::new(u32::MAX, u32::MAX, Vec::<u8>::new()).is_none());
}

#[test]
fn gpu_event_channel_applies_bounded_backpressure() {
    let (sender, receiver) = gpu_event_channel();
    for _ in 0..GPU_EVENT_QUEUE_CAPACITY {
        sender
            .try_send(GpuEvent::Shutdown)
            .expect("event fits bounded queue");
    }
    assert!(matches!(
        sender.try_send(GpuEvent::Shutdown),
        Err(std::sync::mpsc::TrySendError::Full(GpuEvent::Shutdown))
    ));
    assert_eq!(receiver.try_recv(), Ok(GpuEvent::Shutdown));
    sender
        .try_send(GpuEvent::Shutdown)
        .expect("receiver progress releases producer backpressure");
}

#[test]
fn runtime_telemetry_tracks_gpu_queue_and_audio_callback_progress() {
    let _event_guard = EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let _progress_reset = ProgressTelemetryReset;
    set_progress_telemetry_enabled(false);
    assert!(!progress_telemetry_enabled());
    assert_eq!(gpu_progress_age_millis(), None);
    assert_eq!(audio_progress_age_millis(), None);

    let disabled_output_before = audio_output_frames();
    let disabled_underruns_before = audio_underrun_frames();
    record_gpu_event_consumed();
    report_audio_frames(48, 3);
    assert_eq!(audio_output_frames(), disabled_output_before + 48);
    assert_eq!(audio_underrun_frames(), disabled_underruns_before + 3);
    assert_eq!(
        gpu_progress_age_millis(),
        None,
        "ordinary presenter progress must not sample its diagnostic clock"
    );
    assert_eq!(
        audio_progress_age_millis(),
        None,
        "ordinary callbacks must retain counters without sampling their diagnostic clock"
    );

    set_progress_telemetry_enabled(true);
    assert!(progress_telemetry_enabled());
    assert_eq!(gpu_progress_age_millis(), None);
    assert_eq!(audio_progress_age_millis(), None);

    let (sender, receiver) = gpu_event_channel();
    install_event_sink(sender);
    assert!(emit_event(GpuEvent::Shutdown));
    assert_eq!(gpu_events_in_flight(), 1);
    assert_eq!(gpu_work_items_pending(), 1);
    assert_eq!(receiver.recv(), Ok(GpuEvent::Shutdown));
    record_gpu_event_consumed();
    assert_eq!(gpu_events_in_flight(), 0);
    assert_eq!(gpu_work_items_pending(), 1);
    record_gpu_work_item_processed();
    assert_eq!(gpu_work_items_pending(), 0);
    assert!(
        gpu_progress_age_millis().is_some_and(|age| age < 1_000),
        "queue retirement must update the bounded-run GPU progress clock"
    );

    install_gpu_completion_notifier(record_gpu_completion_notification);
    let notifications_before = GPU_COMPLETION_NOTIFICATIONS.load(Ordering::Relaxed);
    record_gpu_submission_completed(2);
    record_gpu_submission_completed(1);
    assert_eq!(gpu_completed_submission_sequence(), 2);
    assert_eq!(gpu_completed_submission_ids_snapshot(), [1, 2]);
    assert_eq!(take_gpu_completed_submission_ids(), [1, 2]);
    assert!(gpu_completed_submission_ids_snapshot().is_empty());
    assert_eq!(
        GPU_COMPLETION_NOTIFICATIONS.load(Ordering::Relaxed),
        notifications_before + 2
    );

    let flip = GpuFlipCompletion {
        address: 0x7000_0000,
        buffer_index: 1,
        flip_argument: 42,
    };
    record_gpu_flip_completed(flip);
    record_gpu_flip_completed(flip);
    assert_eq!(take_gpu_completed_flips(), [flip, flip]);
    assert_eq!(
        GPU_COMPLETION_NOTIFICATIONS.load(Ordering::Relaxed),
        notifications_before + 4
    );

    let output_before = audio_output_frames();
    let underruns_before = audio_underrun_frames();
    report_audio_frames(480, 12);
    assert_eq!(audio_output_frames(), output_before + 480);
    assert_eq!(audio_underrun_frames(), underruns_before + 12);
    assert!(
        audio_progress_age_millis().is_some_and(|age| age < 1_000),
        "host callbacks must update the bounded-run audio progress clock"
    );
    clear_event_sink();
}

#[test]
fn queued_submission_telemetry_counts_decoded_work_until_processed() {
    let _event_guard = EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (sender, receiver) = gpu_event_channel();
    install_event_sink(sender);
    assert!(emit_event(GpuEvent::QueueSubmission {
        queue_id: 7,
        events: vec![GpuEvent::Shutdown, GpuEvent::Shutdown, GpuEvent::Shutdown,],
    }));
    assert_eq!(gpu_events_in_flight(), 1);
    assert_eq!(gpu_work_items_pending(), 3);
    let GpuEvent::QueueSubmission { events, .. } = receiver.recv().expect("queued submission")
    else {
        panic!("emitted submission must retain its wrapper");
    };
    record_gpu_event_consumed();
    assert_eq!(gpu_events_in_flight(), 0);
    assert_eq!(gpu_work_items_pending(), 3);
    for _ in events {
        record_gpu_work_item_processed();
    }
    assert_eq!(gpu_work_items_pending(), 0);
    clear_event_sink();
}

#[test]
fn queue_submission_registers_wait_before_presenter_ingress() {
    let _event_guard = EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let wait = GpuMemoryWait::memory(0x7000_2000, 1, u64::from(u32::MAX), 3, false);
    let (sender, receiver) = gpu_event_channel();
    install_event_sink(sender);

    assert!(emit_event(GpuEvent::QueueSubmission {
        queue_id: 7,
        events: vec![GpuEvent::MemoryWait(wait)],
    }));
    record_gpu_wait_producer(wait.address, false, 1);

    let GpuEvent::QueueSubmission { queue_id, events } =
        receiver.recv().expect("registered queue submission")
    else {
        panic!("queue submission wrapper was lost");
    };
    let [GpuEvent::RegisteredMemoryWait { id, wait: actual }] = events.as_slice() else {
        panic!("submitted wait was not assigned an identity");
    };
    assert_eq!(queue_id, 7);
    assert_eq!(*actual, wait);
    assert_eq!(take_gpu_wait_latch(*id), Some(1));
    assert_eq!(take_gpu_wait_latch(*id), None);
    clear_event_sink();
}

#[test]
fn large_queue_submission_is_chunked_before_bounded_ingress() {
    let _event_guard = EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (sender, receiver) = gpu_event_channel();
    install_event_sink(sender);
    let event_count = GPU_EVENT_SUBMISSION_CHUNK_SIZE * 2 + 3;
    let events = (0..event_count)
        .map(|sequence| GpuEvent::SubmissionBoundary(sequence as u64))
        .collect();

    assert!(emit_event(GpuEvent::QueueSubmission {
        queue_id: 17,
        events,
    }));

    let chunks = (0..3)
        .map(|_| {
            let GpuEvent::QueueSubmission { queue_id, events } =
                receiver.recv().expect("chunked queue submission")
            else {
                panic!("queue submission wrapper was lost");
            };
            assert_eq!(queue_id, 17);
            let sequences = events
                .iter()
                .map(|event| match event {
                    GpuEvent::SubmissionBoundary(sequence) => *sequence,
                    other => panic!("unexpected chunk event {other:?}"),
                })
                .collect::<Vec<_>>();
            ((events.len(), events.capacity()), sequences)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        chunks
            .iter()
            .map(|(storage, _)| *storage)
            .collect::<Vec<_>>(),
        [
            (
                GPU_EVENT_SUBMISSION_CHUNK_SIZE,
                GPU_EVENT_SUBMISSION_CHUNK_SIZE
            ),
            (
                GPU_EVENT_SUBMISSION_CHUNK_SIZE,
                GPU_EVENT_SUBMISSION_CHUNK_SIZE
            ),
            (3, 3)
        ]
    );
    assert_eq!(
        chunks
            .into_iter()
            .flat_map(|(_, sequences)| sequences)
            .collect::<Vec<_>>(),
        (0..event_count as u64).collect::<Vec<_>>(),
        "chunking must preserve the exact queue-local event order"
    );
    assert_eq!(gpu_work_items_pending(), event_count);
    clear_event_sink();
}

#[test]
fn independent_queue_is_admitted_while_large_submission_is_backpressured() {
    let _event_guard = EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (sender, receiver) = gpu_event_channel();
    install_event_sink(sender.clone());
    for _ in 0..GPU_EVENT_QUEUE_CAPACITY {
        sender
            .try_send(GpuEvent::Shutdown)
            .expect("filler event fits the bounded channel");
    }

    let q0_chunk_count = GPU_EVENT_QUEUE_CAPACITY + 2;
    let q0_events =
        vec![GpuEvent::SubmissionBoundary(1); q0_chunk_count * GPU_EVENT_SUBMISSION_CHUNK_SIZE];
    let (q0_admitted, q0_admitted_rx) = std::sync::mpsc::channel();
    let (q0_done, q0_done_rx) = std::sync::mpsc::channel();
    let q0_thread = std::thread::spawn(move || {
        let sent = emit_event_with_admission(
            GpuEvent::QueueSubmission {
                queue_id: 0,
                events: q0_events,
            },
            || q0_admitted.send(()).expect("test receiver remains alive"),
        );
        q0_done.send(sent).expect("test receiver remains alive");
    });
    q0_admitted_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("graphics submission reaches queue admission");
    assert_eq!(
        q0_done_rx.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty),
        "the large graphics submission must retain bounded backpressure"
    );

    let (q33_admitted, q33_admitted_rx) = std::sync::mpsc::channel();
    let (q33_done, q33_done_rx) = std::sync::mpsc::channel();
    let q33_thread = std::thread::spawn(move || {
        let sent = emit_event_with_admission(
            GpuEvent::QueueSubmission {
                queue_id: 33,
                events: vec![GpuEvent::SubmissionBoundary(2)],
            },
            || q33_admitted.send(()).expect("test receiver remains alive"),
        );
        q33_done.send(sent).expect("test receiver remains alive");
    });
    q33_admitted_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("independent compute queue must be admitted before graphics drains");
    assert_eq!(
        q33_done_rx.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty),
        "the full backend channel still applies bounded backpressure"
    );

    // The first released slot goes to the publisher already blocked in `send`; the independent
    // queue was waiting before graphics attempts its next chunk, so the second slot admits q33.
    for _ in 0..2 {
        assert_eq!(receiver.recv(), Ok(GpuEvent::Shutdown));
    }
    assert_eq!(
        q33_done_rx.recv_timeout(Duration::from_secs(1)),
        Ok(true),
        "compute must enter the channel before the large graphics publication completes"
    );
    assert_eq!(
        q0_done_rx.try_recv(),
        Err(std::sync::mpsc::TryRecvError::Empty),
        "graphics still has chunks waiting behind the compute producer"
    );
    for _ in 2..GPU_EVENT_QUEUE_CAPACITY {
        assert_eq!(receiver.recv(), Ok(GpuEvent::Shutdown));
    }

    let expected_submissions = q0_chunk_count + 1;
    let mut received_submissions = 0;
    let mut saw_q33 = false;
    while received_submissions != expected_submissions {
        let GpuEvent::QueueSubmission { queue_id, events } =
            receiver.recv().expect("published queue chunk")
        else {
            panic!("only queue submissions remain after the fillers");
        };
        if queue_id == 33 {
            assert_eq!(events, [GpuEvent::SubmissionBoundary(2)]);
            saw_q33 = true;
        } else {
            assert_eq!(queue_id, 0);
            assert_eq!(events.len(), GPU_EVENT_SUBMISSION_CHUNK_SIZE);
        }
        received_submissions += 1;
    }
    assert!(saw_q33, "the independent producer must not be lost");
    assert_eq!(
        q0_done_rx.recv_timeout(Duration::from_secs(1)),
        Ok(true),
        "draining all chunks releases graphics publication"
    );
    q0_thread.join().expect("graphics publisher exits cleanly");
    q33_thread.join().expect("compute publisher exits cleanly");
    clear_event_sink();
}

#[test]
fn same_queue_publications_remain_fifo_across_chunk_yields() {
    let _event_guard = EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (sender, receiver) = gpu_event_channel();
    install_event_sink(sender.clone());
    for _ in 0..GPU_EVENT_QUEUE_CAPACITY {
        sender
            .try_send(GpuEvent::Shutdown)
            .expect("filler event fits the bounded channel");
    }

    let (first_admitted, first_admitted_rx) = std::sync::mpsc::channel();
    let first = std::thread::spawn(move || {
        emit_event_with_admission(
            GpuEvent::QueueSubmission {
                queue_id: 0,
                events: vec![GpuEvent::SubmissionBoundary(1); GPU_EVENT_SUBMISSION_CHUNK_SIZE * 3],
            },
            || {
                first_admitted
                    .send(())
                    .expect("test receiver remains alive");
            },
        )
    });
    first_admitted_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("first graphics publication is admitted");

    let (second_admitted, second_admitted_rx) = std::sync::mpsc::channel();
    let second = std::thread::spawn(move || {
        emit_event_with_admission(
            GpuEvent::QueueSubmission {
                queue_id: 0,
                events: vec![GpuEvent::SubmissionBoundary(2)],
            },
            || {
                second_admitted
                    .send(())
                    .expect("test receiver remains alive");
            },
        )
    });
    second_admitted_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("later graphics publication enters the same-queue FIFO");

    for _ in 0..GPU_EVENT_QUEUE_CAPACITY {
        assert_eq!(receiver.recv(), Ok(GpuEvent::Shutdown));
    }
    let mut boundaries = Vec::new();
    for _ in 0..4 {
        let GpuEvent::QueueSubmission { queue_id, events } =
            receiver.recv().expect("graphics queue chunk")
        else {
            panic!("only graphics submissions remain");
        };
        assert_eq!(queue_id, 0);
        boundaries.push(match events.as_slice() {
            [GpuEvent::SubmissionBoundary(sequence)] => *sequence,
            events
                if events
                    .iter()
                    .all(|event| event == &GpuEvent::SubmissionBoundary(1)) =>
            {
                1
            }
            _ => panic!("unexpected same-queue test payload"),
        });
    }
    assert_eq!(
        boundaries,
        [1, 1, 1, 2],
        "a later submission must not enter between chunks of its same-queue predecessor"
    );
    assert!(first.join().expect("first publisher exits cleanly"));
    assert!(second.join().expect("second publisher exits cleanly"));
    clear_event_sink();
}

#[test]
fn old_producer_does_not_satisfy_a_later_label_generation() {
    let _event_guard = EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let wait = GpuMemoryWait::memory(0x7000_2020, 1, u64::from(u32::MAX), 3, false);
    let (sender, receiver) = gpu_event_channel();
    install_event_sink(sender);
    record_gpu_wait_producer(wait.address, false, 1);

    assert!(emit_event(GpuEvent::QueueSubmission {
        queue_id: 8,
        events: vec![GpuEvent::MemoryWait(wait)],
    }));
    let GpuEvent::QueueSubmission { events, .. } =
        receiver.recv().expect("registered queue submission")
    else {
        panic!("queue submission wrapper was lost");
    };
    let [GpuEvent::RegisteredMemoryWait { id, .. }] = events.as_slice() else {
        panic!("submitted wait was not assigned an identity");
    };
    assert_eq!(take_gpu_wait_latch(*id), None);
    super::retire_gpu_wait(*id);
    clear_event_sink();
}

#[test]
fn failed_queue_send_cancels_all_registered_waits() {
    let _event_guard = EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let wait = GpuMemoryWait::memory(0x7000_2040, 1, u64::from(u32::MAX), 3, false);
    let (sender, receiver) = gpu_event_channel();
    drop(receiver);
    install_event_sink(sender);

    assert!(!emit_event(GpuEvent::QueueSubmission {
        queue_id: 9,
        events: vec![GpuEvent::MemoryWait(wait)],
    }));
    assert!(
        super::gpu_wait_registry()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .waits
            .is_empty()
    );
    assert_eq!(gpu_work_items_pending(), 0);
    clear_event_sink();
}

#[test]
fn partial_queue_send_cancels_only_waits_that_never_reached_the_presenter() {
    let _event_guard = EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let first_wait = GpuMemoryWait::memory(0x7000_2060, 1, u64::from(u32::MAX), 3, false);
    let last_wait = GpuMemoryWait {
        address: 0x7000_2080,
        ..first_wait
    };
    let chunk_count = GPU_EVENT_QUEUE_CAPACITY + 2;
    let mut events =
        vec![GpuEvent::SubmissionBoundary(1); chunk_count * GPU_EVENT_SUBMISSION_CHUNK_SIZE];
    events[0] = GpuEvent::MemoryWait(first_wait);
    events[(chunk_count - 1) * GPU_EVENT_SUBMISSION_CHUNK_SIZE] = GpuEvent::MemoryWait(last_wait);

    let (sender, receiver) = gpu_event_channel();
    install_event_sink(sender.clone());
    for _ in 0..GPU_EVENT_QUEUE_CAPACITY {
        sender
            .try_send(GpuEvent::Shutdown)
            .expect("filler event fits the bounded channel");
    }
    let (admitted, admitted_rx) = std::sync::mpsc::channel();
    let publisher = std::thread::spawn(move || {
        emit_event_with_admission(
            GpuEvent::QueueSubmission {
                queue_id: 0,
                events,
            },
            || admitted.send(()).expect("test receiver remains alive"),
        )
    });
    admitted_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("submission registers both waits before ingress");

    // Each removed filler lets q0 place one chunk. After all fillers and the first q0 chunk are
    // removed, at least that first wait is presenter-owned while the final chunk is still blocked.
    for _ in 0..GPU_EVENT_QUEUE_CAPACITY {
        assert_eq!(receiver.recv(), Ok(GpuEvent::Shutdown));
    }
    let GpuEvent::QueueSubmission {
        events: first_chunk,
        ..
    } = receiver.recv().expect("first published queue chunk")
    else {
        panic!("first published event must be a queue chunk");
    };
    let first_id = first_chunk
        .iter()
        .find_map(|event| match event {
            GpuEvent::RegisteredMemoryWait { id, wait } if *wait == first_wait => Some(*id),
            _ => None,
        })
        .expect("first chunk carries its registered wait");
    drop(receiver);
    assert!(
        !publisher.join().expect("publisher exits after disconnect"),
        "disconnecting before the final chunk must fail the submission"
    );

    let live_waits = super::gpu_wait_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .waits
        .keys()
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(
        live_waits,
        [first_id],
        "a failed later chunk must not cancel a wait already handed to the presenter"
    );
    super::retire_gpu_wait(first_id);
    clear_event_sink();
}

#[test]
fn shares_a_host_pad_snapshot_across_the_gpu_runtime_seam() {
    let state = HostPadState {
        buttons: 0x4030,
        left_x: 1,
        left_y: 2,
        right_x: 3,
        right_y: 4,
        l2: 5,
        r2: 6,
    };
    set_host_pad_state(state);
    assert_eq!(host_pad_state(), state);
    set_host_pad_state(HostPadState::default());
    assert_eq!(host_pad_state(), HostPadState::default());

    set_host_pad_state(state);
    set_host_pad_state(HostPadState::default());
    assert_eq!(host_pad_state().buttons, state.buttons);
    assert_eq!(host_pad_state(), HostPadState::default());
}

#[test]
fn decodes_acquire_invalidation_scope_without_backend_assumptions() {
    let range = GpuMemoryAcquire {
        engine: 1,
        cb_db_control: 0x7fc0,
        base_address: 0x1234_0000,
        size_bytes: 0x8000,
        poll_interval: 10,
        gcr_control: 0x1_0388,
    };
    assert!(range.invalidates_guest_resources());
    assert!(!range.covers_all_guest_memory());
    assert!(range.overlaps(0x1234_7000, 0x2000));
    assert!(!range.overlaps(0x1235_0000, 0x1000));

    let all = GpuMemoryAcquire {
        size_bytes: 0,
        gcr_control: 0x280,
        ..range
    };
    assert!(all.invalidates_guest_resources());
    assert!(all.covers_all_guest_memory());
    assert!(all.overlaps(0x7000_0000_0000, 0x1000));
}

#[test]
fn memory_wait_comparisons_apply_the_packet_mask() {
    let wait = GpuMemoryWait::memory(0x1000, 0x34, 0xff, 3, false);
    assert!(wait.is_satisfied_by(0xabcd_1234));
    assert!(!wait.is_satisfied_by(0xabcd_1235));
    assert!(
        GpuMemoryWait {
            compare_function: 4,
            ..wait
        }
        .is_satisfied_by(0xabcd_1235)
    );
}

#[test]
fn release_cache_policy_distinguishes_cpu_and_gpu_label_visibility() {
    let mut release = GpuMemoryRelease::label(0x1000, GpuMemoryReleaseValue::Immediate64(1));
    assert!(!release.cpu_visible());
    assert!(!release.flushes_render_backend());
    release.cache_policy = GpuCachePolicy::Bypass as u32;
    assert!(release.cpu_visible());
    release.action = GpuEndOfPipeCbDbAction::WritebackInvalidateCbData as u32;
    assert!(release.flushes_render_backend());
}

#[test]
fn wait_operation_engine_select_matches_sdk_wait_operation() {
    assert_eq!(
        GpuWaitOperation::from_u32(0),
        Some(GpuWaitOperation::WaitMe)
    );
    assert_eq!(
        GpuWaitOperation::from_u32(1),
        Some(GpuWaitOperation::WaitPfp)
    );
    assert!(GpuWaitOperation::WaitPfp.stalls_parser());
    assert!(!GpuWaitOperation::WaitMe.stalls_parser());
}

#[test]
fn render_backend_flush_decodes_sdk_cb_db_event_types() {
    assert_eq!(
        GpuRenderBackendFlush::from_event_type(0x2E),
        Some(GpuRenderBackendFlush::CbMetadata)
    );
    assert_eq!(
        GpuRenderBackendFlush::from_event_type(0x2C),
        Some(GpuRenderBackendFlush::DbMetadata)
    );
    assert_eq!(
        GpuRenderBackendFlush::from_event_type(0x31),
        Some(GpuRenderBackendFlush::CbData)
    );
    assert_eq!(
        GpuRenderBackendFlush::from_event_type(0x2A),
        Some(GpuRenderBackendFlush::DbData)
    );
    assert_eq!(
        GpuRenderBackendFlush::from_event_type(0x16),
        Some(GpuRenderBackendFlush::CbDbData)
    );
    assert_eq!(GpuRenderBackendFlush::from_event_type(0x07), None);
}

fn compute_registers(thread_words: [u32; 3], starts: [u32; 3]) -> Vec<GpuRegister> {
    [
        (COMPUTE_NUM_THREAD_X_REGISTER, thread_words[0]),
        (COMPUTE_NUM_THREAD_Y_REGISTER, thread_words[1]),
        (COMPUTE_NUM_THREAD_Z_REGISTER, thread_words[2]),
        (COMPUTE_START_X_REGISTER, starts[0]),
        (COMPUTE_START_Y_REGISTER, starts[1]),
        (COMPUTE_START_Z_REGISTER, starts[2]),
        (COMPUTE_PGM_RSRC2_REGISTER, 64 << 15),
        // Two 1-KiB blocks per wave: 64 bytes per lane for this wave32 launch.
        (COMPUTE_TMPRING_SIZE_REGISTER, 2 << 12),
    ]
    .map(|(offset, value)| GpuRegister { offset, value })
    .to_vec()
}

#[test]
fn decodes_direct_group_dimensions_wave_size_and_lds() {
    let registers = compute_registers([64, 2, 1], [99, 99, 99]);
    let packet = [0xc003_1500, 10, 4, 2, 0x8045];
    let launch =
        decode_compute_dispatch_direct(&packet, &registers).expect("valid direct dispatch");
    assert_eq!(launch.grid, ComputeGrid::Threadgroups([10, 4, 2]));
    assert_eq!(launch.group_count, [10, 4, 2]);
    assert_eq!(launch.base_group, [0; 3]);
    assert_eq!(launch.threads_per_group, [64, 2, 1]);
    assert_eq!(launch.threads_per_group_total(), 128);
    assert_eq!(launch.wave_lane_count, 32);
    assert_eq!(launch.lds_size_bytes, 32 * 1024);
    assert_eq!(launch.scratch_size_bytes, 64);
}

#[test]
fn decodes_compute_num_thread_registers_with_selector_high_bits() {
    let registers = [
        GpuRegister {
            offset: 0x2000_0000 | COMPUTE_NUM_THREAD_X_REGISTER,
            value: 64,
        },
        GpuRegister {
            offset: 0x2000_0000 | COMPUTE_NUM_THREAD_Y_REGISTER,
            value: 1,
        },
        GpuRegister {
            offset: 0x2000_0000 | COMPUTE_NUM_THREAD_Z_REGISTER,
            value: 1,
        },
        GpuRegister {
            offset: COMPUTE_PGM_RSRC2_REGISTER,
            value: 64 << 15,
        },
    ];
    let launch = decode_compute_launch([16, 1, 1], 0x8005, &registers)
        .expect("NUM_THREAD with UE selector high bits");
    assert_eq!(launch.threads_per_group, [64, 1, 1]);
    assert_eq!(launch.grid, ComputeGrid::Threadgroups([16, 1, 1]));
}

#[test]
fn decodes_thread_dimensions_as_an_exact_unpadded_grid() {
    let registers = compute_registers([64, 1, 1], [0; 3]);
    let launch = decode_compute_launch([130, 3, 1], 0x8025, &registers)
        .expect("valid thread-dimension dispatch");
    assert_eq!(launch.grid, ComputeGrid::Threads([130, 3, 1]));
    assert_eq!(launch.group_count, [3, 3, 1]);
}

#[test]
fn decodes_zero_extent_dispatches_as_empty_work() {
    let registers = compute_registers([64, 1, 1], [0; 3]);
    let groups = decode_compute_launch([0, 8, 1], 0x8005, &registers)
        .expect("zero group extent is an empty dispatch");
    assert_eq!(groups.grid, ComputeGrid::Threadgroups([0, 8, 1]));
    assert!(groups.is_empty());

    let threads = decode_compute_launch([0, 8, 1], 0x8025, &registers)
        .expect("zero thread extent is an empty dispatch");
    assert_eq!(threads.grid, ComputeGrid::Threads([0, 8, 1]));
    assert!(threads.is_empty());
}

#[test]
fn decodes_partial_final_workgroups_without_padded_invocations() {
    let registers = compute_registers([(32 << 16) | 64, (1 << 16) | 4, (1 << 16) | 1], [0; 3]);
    let launch = decode_compute_launch([3, 2, 1], 0x8007, &registers)
        .expect("valid partial-threadgroup dispatch");
    assert_eq!(launch.grid, ComputeGrid::Threads([160, 5, 1]));
    assert_eq!(launch.group_count, [3, 2, 1]);
}

#[test]
fn preserves_nonzero_compute_start_for_backend_validation() {
    let registers = compute_registers([8, 4, 2], [4, 3, 2]);
    let launch = decode_compute_launch([10, 8, 5], 0x8001, &registers)
        .expect("valid based dispatch geometry");
    assert_eq!(launch.base_group, [4, 3, 2]);
    assert_eq!(launch.group_count, [6, 5, 3]);
    assert_eq!(launch.grid, ComputeGrid::Threadgroups([6, 5, 3]));
}

#[test]
fn rejects_disabled_and_malformed_compute_dispatches() {
    let registers = compute_registers([64, 1, 1], [0; 3]);
    assert_eq!(
        decode_compute_launch([1; 3], 0x8004, &registers),
        Err(ComputeDispatchDecodeError::Disabled(0x8004))
    );
    assert!(matches!(
        decode_compute_dispatch_direct(&[0xc002_1500, 1, 1, 1], &registers),
        Err(ComputeDispatchDecodeError::DeclaredPacketLength(4))
    ));
    assert!(matches!(
        decode_compute_launch([1; 3], 0x8003, &compute_registers([64, 1, 1], [0; 3])),
        Err(ComputeDispatchDecodeError::InvalidPartialSize { axis: 'x', .. })
    ));
}

#[test]
fn decodes_compute_program_and_system_sgpr_state() {
    let registers = [
        GpuRegister {
            offset: COMPUTE_PGM_LO_REGISTER,
            value: 0x1234_5678,
        },
        GpuRegister {
            offset: COMPUTE_PGM_HI_REGISTER,
            value: 0xdead_be5a,
        },
        GpuRegister {
            offset: COMPUTE_PGM_RSRC2_REGISTER,
            value: (4 << 1) | (1 << 7) | (1 << 9) | (1 << 10),
        },
    ];
    assert_eq!(compute_program_address(&registers), Some(0x5a12_3456_7800));
    let system = decode_compute_system_registers(&registers);
    assert_eq!(system.work_group_x, Some(4));
    assert_eq!(system.work_group_y, None);
    assert_eq!(system.work_group_z, Some(5));
    assert_eq!(system.thread_group_size, Some(6));
}

#[test]
fn decodes_base_relative_and_absolute_indirect_dispatch_packets() {
    let relative = decode_compute_dispatch_indirect(&[0xc001_1600, 0x280, 0x8041])
        .expect("relative indirect packet");
    assert_eq!(
        relative.dimensions,
        super::ComputeIndirectDimensions::BaseOffset(0x280)
    );
    assert_eq!(relative.initiator, 0x8041);

    let absolute = decode_compute_dispatch_indirect(&[0xc002_1600, 0x7654_3200, 0x1234, 0x8001])
        .expect("absolute indirect packet");
    assert_eq!(
        absolute.dimensions,
        super::ComputeIndirectDimensions::Absolute(0x1234_7654_3200)
    );
    assert_eq!(absolute.initiator, 0x8001);
}
