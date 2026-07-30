// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    COMPUTE_NUM_THREAD_X_REGISTER, COMPUTE_NUM_THREAD_Y_REGISTER, COMPUTE_NUM_THREAD_Z_REGISTER,
    COMPUTE_PGM_HI_REGISTER, COMPUTE_PGM_LO_REGISTER, COMPUTE_PGM_RSRC2_REGISTER,
    COMPUTE_START_X_REGISTER, COMPUTE_START_Y_REGISTER, COMPUTE_START_Z_REGISTER,
    ComputeDispatchDecodeError, ComputeGrid, GPU_EVENT_QUEUE_CAPACITY,
    GPU_EVENT_SUBMISSION_CHUNK_SIZE, GpuEvent, GpuFlipCompletion, GpuHostImage, GpuMemoryAcquire,
    GpuMemoryWait, GpuRegister, HostPadState, audio_output_frames, audio_underrun_frames,
    clear_event_sink, compute_program_address, decode_compute_dispatch_direct,
    decode_compute_dispatch_indirect, decode_compute_launch, decode_compute_system_registers,
    emit_event, gpu_completed_submission_sequence, gpu_event_channel, gpu_events_in_flight,
    gpu_work_items_pending, host_pad_state, install_event_sink, record_gpu_event_consumed,
    record_gpu_flip_completed, record_gpu_submission_completed, record_gpu_wait_producer,
    record_gpu_work_item_processed, report_audio_frames, request_presenter_shutdown,
    set_host_pad_state, take_gpu_completed_flips, take_gpu_completed_submission_ids,
    take_gpu_wait_latch, take_presenter_shutdown_request,
};
use std::sync::Mutex;
use std::time::{Duration, Instant};

static EVENT_SINK_TEST: Mutex<()> = Mutex::new(());

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
fn out_of_band_presenter_shutdown_coalesces_without_entering_the_full_queue() {
    let _event_guard = EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_event_sink();
    assert!(
        !request_presenter_shutdown(),
        "a probe without a presenter must retain the hard timeout path"
    );

    let (sender, receiver) = gpu_event_channel();
    install_event_sink(sender.clone());
    for _ in 0..GPU_EVENT_QUEUE_CAPACITY {
        sender
            .try_send(GpuEvent::Shutdown)
            .expect("fill bounded queue");
    }

    let blocked_producer = std::thread::spawn(|| emit_event(GpuEvent::Shutdown));
    let producer_deadline = Instant::now() + Duration::from_secs(1);
    while gpu_events_in_flight() == 0 {
        assert!(
            Instant::now() < producer_deadline,
            "real event producer did not enter its blocking send"
        );
        std::thread::yield_now();
    }

    let (request_result_sender, request_result_receiver) = std::sync::mpsc::channel();
    let requester = std::thread::spawn(move || {
        let started = Instant::now();
        let installed = request_presenter_shutdown();
        request_result_sender
            .send((installed, started.elapsed()))
            .expect("request result receiver remains alive");
    });
    let request_result = request_result_receiver.recv_timeout(Duration::from_millis(500));

    // Always release the real producer before asserting the timeout result. That makes a
    // regression fail cleanly instead of leaving both test threads deadlocked on the sink mutex.
    assert_eq!(receiver.recv(), Ok(GpuEvent::Shutdown));
    assert!(
        blocked_producer
            .join()
            .expect("blocked event producer completes after queue progress")
    );
    requester.join().expect("shutdown requester completes");
    let (installed, request_duration) =
        request_result.expect("out-of-band request must not wait for the event-sink mutex");
    assert!(installed);
    assert!(
        request_duration < Duration::from_millis(100),
        "out-of-band request took {request_duration:?} while the event-sink mutex was held"
    );
    assert!(request_presenter_shutdown());
    assert!(
        take_presenter_shutdown_request(),
        "the owning presenter observes one pending publication"
    );
    assert!(
        !take_presenter_shutdown_request(),
        "repeated requests must coalesce instead of publishing twice"
    );
    clear_event_sink();
}

#[test]
fn runtime_telemetry_tracks_gpu_queue_and_audio_callback_progress() {
    let _event_guard = EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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

    record_gpu_submission_completed(2);
    record_gpu_submission_completed(1);
    assert_eq!(gpu_completed_submission_sequence(), 2);
    assert_eq!(take_gpu_completed_submission_ids(), [1, 2]);

    let flip = GpuFlipCompletion {
        address: 0x7000_0000,
        buffer_index: 1,
        flip_argument: 42,
    };
    record_gpu_flip_completed(flip);
    record_gpu_flip_completed(flip);
    assert_eq!(take_gpu_completed_flips(), [flip, flip]);

    report_audio_frames(480, 12);
    assert_eq!(audio_output_frames(), 480);
    assert_eq!(audio_underrun_frames(), 12);
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
    let wait = GpuMemoryWait {
        address: 0x7000_2000,
        reference: 1,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
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
    let events = vec![
        GpuEvent::PipelineBarrier(super::GpuPipelineBarrier {
            wait_for: super::GpuPipelineStage::PixelShader,
        });
        event_count
    ];

    assert!(emit_event(GpuEvent::QueueSubmission {
        queue_id: 17,
        events,
    }));

    let chunk_lengths = (0..3)
        .map(|_| {
            let GpuEvent::QueueSubmission { queue_id, events } =
                receiver.recv().expect("chunked queue submission")
            else {
                panic!("queue submission wrapper was lost");
            };
            assert_eq!(queue_id, 17);
            events.len()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        chunk_lengths,
        [
            GPU_EVENT_SUBMISSION_CHUNK_SIZE,
            GPU_EVENT_SUBMISSION_CHUNK_SIZE,
            3
        ]
    );
    assert_eq!(gpu_work_items_pending(), event_count);
    clear_event_sink();
}

#[test]
fn old_producer_does_not_satisfy_a_later_label_generation() {
    let _event_guard = EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let wait = GpuMemoryWait {
        address: 0x7000_2020,
        reference: 1,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
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
    let wait = GpuMemoryWait {
        address: 0x7000_2040,
        reference: 1,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
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
    let wait = GpuMemoryWait {
        address: 0x1000,
        reference: 0x34,
        mask: 0xff,
        compare_function: 3,
        is_64_bit: false,
    };
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

fn compute_registers(thread_words: [u32; 3], starts: [u32; 3]) -> Vec<GpuRegister> {
    [
        (COMPUTE_NUM_THREAD_X_REGISTER, thread_words[0]),
        (COMPUTE_NUM_THREAD_Y_REGISTER, thread_words[1]),
        (COMPUTE_NUM_THREAD_Z_REGISTER, thread_words[2]),
        (COMPUTE_START_X_REGISTER, starts[0]),
        (COMPUTE_START_Y_REGISTER, starts[1]),
        (COMPUTE_START_Z_REGISTER, starts[2]),
        (COMPUTE_PGM_RSRC2_REGISTER, 64 << 15),
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
