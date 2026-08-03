// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;

#[test]
fn closing_the_display_port_releases_its_buffers_and_refuses_a_second_close() {
    let mut state = open_display();
    assert_eq!(
        dispatch("uquVH4-Du78", args(&[VIDEO_OUT_HANDLE]), &mut state),
        Some(SUCCESS)
    );
    assert!(!state.video_out.is_open);
    assert!(state.video_out.display_buffers.is_empty());
    assert!(state.video_out.display_buffer_attributes.is_empty());
    assert_eq!(
        dispatch("uquVH4-Du78", args(&[VIDEO_OUT_HANDLE]), &mut state),
        Some(error(VIDEO_OUT_ERROR_INVALID_HANDLE))
    );
}

#[test]
fn a_display_call_on_a_handle_no_port_owns_is_rejected() {
    let mut state = open_display();
    for nid in ["uquVH4-Du78", "N5KDtkIjjJ4", "1FZBKy8HeNU", "T4ucGB8CsnM"] {
        assert_eq!(
            dispatch(nid, args(&[2, 0]), &mut state),
            Some(error(VIDEO_OUT_ERROR_INVALID_HANDLE)),
            "{nid} accepted a handle no port owns"
        );
    }
}

/// Sets the process display port's open flag, returning the previous value for restoration.
fn set_process_display_open(open: bool) -> bool {
    let dispatch_lock =
        crate::PROBE_DISPATCH.get_or_init(|| std::sync::RwLock::new(ProbeDispatchState::default()));
    let mut state = crate::lock_probe_dispatch_for_write(dispatch_lock);
    std::mem::replace(&mut state.video_out.is_open, open)
}

#[test]
fn wait_vblank_validates_its_port_briefly_and_sleeps_off_the_dispatch_lock() {
    // Dispatched before the process-wide dispatch lock (plan C5): the port check takes a short
    // read lock and the vblank sleep holds nothing another import needs.
    let was_open = set_process_display_open(false);
    assert_eq!(
        video_out_wait_vblank(args(&[2])),
        error(VIDEO_OUT_ERROR_INVALID_HANDLE)
    );
    set_process_display_open(true);
    let started = std::time::Instant::now();
    assert_eq!(video_out_wait_vblank(args(&[VIDEO_OUT_HANDLE])), SUCCESS);
    assert!(
        started.elapsed() < std::time::Duration::from_millis(250),
        "a vblank wait must resolve within a frame, not a scheduler pathology"
    );
    set_process_display_open(was_open);
}

#[test]
fn wait_vblank_refuses_to_run_under_the_dispatch_write_lock() {
    let dispatch_lock =
        crate::PROBE_DISPATCH.get_or_init(|| std::sync::RwLock::new(ProbeDispatchState::default()));
    let guard = crate::lock_probe_dispatch_for_write(dispatch_lock);
    // Catch the refusal inside the guard's lifetime: letting the panic unwind out of the test
    // would drop the guard on a panicking thread and poison the process-wide lock for every
    // later test in this binary.
    let refusal = std::panic::catch_unwind(|| video_out_wait_vblank(args(&[VIDEO_OUT_HANDLE])))
        .expect_err("a vblank wait under the dispatch write lock must refuse");
    drop(guard);
    let message = refusal
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(
        message.contains("process-wide dispatch write lock"),
        "unexpected refusal: {message}"
    );
}

#[test]
fn unregistering_the_buffer_set_clears_it_once_and_refuses_an_unknown_set() {
    let mut state = open_display();
    assert_eq!(
        dispatch("N5KDtkIjjJ4", args(&[VIDEO_OUT_HANDLE, 1]), &mut state),
        Some(error(VIDEO_OUT_ERROR_INVALID_INDEX))
    );
    assert_eq!(
        dispatch("N5KDtkIjjJ4", args(&[VIDEO_OUT_HANDLE, 0]), &mut state),
        Some(SUCCESS)
    );
    assert!(state.video_out.display_buffers.is_empty());
    assert_eq!(
        dispatch("N5KDtkIjjJ4", args(&[VIDEO_OUT_HANDLE, 0]), &mut state),
        Some(error(VIDEO_OUT_ERROR_INVALID_INDEX))
    );
}

#[test]
fn changing_the_buffer_attributes_retargets_every_registered_buffer() {
    let mut state = open_display();
    let attribute = buffer_attribute(VIDEO_OUT_TILING_MODE_LINEAR, 1920, 1080);
    assert_eq!(
        dispatch(
            "HuViW4HnrOw",
            args(&[VIDEO_OUT_HANDLE, 0, address(&attribute), 0]),
            &mut state
        ),
        Some(SUCCESS)
    );
    assert_eq!(state.video_out.width, 1920);
    assert_eq!(state.video_out.height, 1080);
    assert_eq!(state.video_out.tiling_mode, VIDEO_OUT_TILING_MODE_LINEAR);
    assert_eq!(state.video_out.pixel_format, 0x8000_0000_2200_0000);
    assert_eq!(
        state.video_out.display_buffer_attributes,
        vec![
            Some(VideoOutBufferAttributes {
                width: 1920,
                height: 1080,
                pitch: 1920,
                pixel_format: 0x8000_0000_2200_0000,
                tiling_mode: VIDEO_OUT_TILING_MODE_LINEAR,
            });
            2
        ]
    );
}

#[test]
fn a_malformed_buffer_attribute_change_is_rejected_rather_than_applied() {
    let mut state = open_display();
    let unusable_tiling = buffer_attribute(2, 1920, 1080);
    let no_resolution = buffer_attribute(VIDEO_OUT_TILING_MODE_TILE, 0, 1080);
    for (attribute, expected) in [
        (
            address(&unusable_tiling),
            VIDEO_OUT_ERROR_INVALID_TILING_MODE,
        ),
        (address(&no_resolution), VIDEO_OUT_ERROR_INVALID_RESOLUTION),
        (0, VIDEO_OUT_ERROR_INVALID_ADDRESS),
    ] {
        assert_eq!(
            dispatch(
                "HuViW4HnrOw",
                args(&[VIDEO_OUT_HANDLE, 0, attribute, 0]),
                &mut state
            ),
            Some(error(expected))
        );
    }
    assert_eq!(state.video_out.width, 1280);
    assert_eq!(
        dispatch(
            "HuViW4HnrOw",
            args(&[VIDEO_OUT_HANDLE, 1, address(&no_resolution), 0]),
            &mut state
        ),
        Some(error(VIDEO_OUT_ERROR_INVALID_INDEX))
    );
}

#[test]
fn the_vblank_status_reports_a_free_running_counter_in_its_documented_layout() {
    let mut state = open_display();
    let mut status = [0xff_u8; 0x28];
    let status_address = address_mut(&mut status);
    assert_eq!(
        dispatch(
            "1FZBKy8HeNU",
            args(&[VIDEO_OUT_HANDLE, status_address]),
            &mut state
        ),
        Some(SUCCESS)
    );
    let count = u64::from_le_bytes(status[..8].try_into().expect("the count is eight bytes"));
    let process_time =
        u64::from_le_bytes(status[8..16].try_into().expect("the time is eight bytes"));
    assert!(
        process_time <= kernel_process_time(false),
        "the reported vblank must already have happened"
    );
    assert_eq!(status[16..24], [0; 8], "the reserved word must be cleared");
    assert_eq!(
        status[32..40],
        [0; 8],
        "flags, phase and padding must be zero"
    );

    // The wait is dispatched before the process lock and checks the process display port.
    let was_open = set_process_display_open(true);
    assert_eq!(video_out_wait_vblank(args(&[VIDEO_OUT_HANDLE])), SUCCESS);
    set_process_display_open(was_open);
    assert_eq!(
        dispatch(
            "1FZBKy8HeNU",
            args(&[VIDEO_OUT_HANDLE, status_address]),
            &mut state
        ),
        Some(SUCCESS)
    );
    let waited = u64::from_le_bytes(status[..8].try_into().expect("the count is eight bytes"));
    assert!(
        waited > count,
        "a completed vblank wait must advance the counter: {count} then {waited}"
    );
    assert_eq!(
        dispatch("1FZBKy8HeNU", args(&[VIDEO_OUT_HANDLE, 0]), &mut state),
        Some(error(VIDEO_OUT_ERROR_INVALID_ADDRESS))
    );
}

#[test]
fn the_flip_status_reports_pending_depth_and_the_last_completed_flip() {
    let mut state = open_display();
    let mut status = [0xff_u8; 0x80];
    let status_address = address_mut(&mut status);
    assert_eq!(
        dispatch(
            "SbU3dwp80lQ",
            args(&[VIDEO_OUT_HANDLE, status_address]),
            &mut state
        ),
        Some(SUCCESS)
    );
    assert_eq!(
        status, [0_u8; 0x80],
        "before any flip every field must read zero"
    );

    state
        .pending_video_out_flips
        .push_back(PendingVideoOutFlip {
            completion: remu_gpu::GpuFlipCompletion {
                address: 0x2000,
                buffer_index: 1,
                flip_argument: 7,
            },
            handle: 1,
            submit_process_time_counter: 210_000,
        });
    state.video_out.completed_flip_count = 3;
    state.video_out.last_completed_flip = Some(CompletedVideoOutFlip {
        process_time: 111,
        process_time_counter: 222_000,
        flip_argument: 0x4242,
        buffer_index: 1,
        submit_process_time_counter: 210_000,
    });
    assert_eq!(
        dispatch(
            "SbU3dwp80lQ",
            args(&[VIDEO_OUT_HANDLE, status_address]),
            &mut state
        ),
        Some(SUCCESS)
    );
    let field_u64 = |offset: usize| {
        u64::from_le_bytes(
            status[offset..offset + 8]
                .try_into()
                .expect("an eight-byte field"),
        )
    };
    let field_u32 = |offset: usize| {
        u32::from_le_bytes(
            status[offset..offset + 4]
                .try_into()
                .expect("a four-byte field"),
        )
    };
    assert_eq!(field_u64(0x00), 3, "count");
    assert_eq!(field_u64(0x08), 111, "processTime");
    assert_eq!(field_u64(0x18), 0x4242, "flipArg");
    assert_eq!(field_u64(0x28), 222_000, "processTimeCounter");
    assert_eq!(field_u32(0x30), 0, "gcQueueNum");
    assert_eq!(field_u32(0x34), 1, "flipPendingNum");
    assert_eq!(field_u32(0x38), 1, "currentBuffer");
    assert_eq!(field_u64(0x40), 210_000, "submitProcessTimeCounter");

    assert_eq!(
        dispatch("SbU3dwp80lQ", args(&[VIDEO_OUT_HANDLE, 0]), &mut state),
        Some(error(VIDEO_OUT_ERROR_INVALID_ADDRESS))
    );
    assert_eq!(
        dispatch("SbU3dwp80lQ", args(&[2, status_address]), &mut state),
        Some(error(VIDEO_OUT_ERROR_INVALID_HANDLE))
    );
}

#[test]
fn a_delivered_event_reports_the_display_event_it_carries() {
    let mut state = open_display();
    let mut event = [0_u8; KERNEL_EVENT_SIZE];
    event[12..16].copy_from_slice(&1_u32.to_le_bytes());
    assert_eq!(
        dispatch("U2JJtSqNKZI", args(&[address(&event)]), &mut state),
        Some(1)
    );
    event[12..16].copy_from_slice(&0_u32.to_le_bytes());
    assert_eq!(
        dispatch("U2JJtSqNKZI", args(&[address(&event)]), &mut state),
        Some(SUCCESS)
    );
    assert_eq!(
        dispatch("U2JJtSqNKZI", args(&[0]), &mut state),
        Some(error(VIDEO_OUT_ERROR_INVALID_ADDRESS))
    );
}

#[test]
fn output_options_are_cleared_and_a_null_options_block_is_rejected() {
    let mut state = open_display();
    let mut options = [0xff_u8; 0x40];
    assert_eq!(
        dispatch(
            "+I4K03i3EL0",
            args(&[address_mut(&mut options)]),
            &mut state
        ),
        Some(SUCCESS)
    );
    assert_eq!(options, [0; 0x40]);
    assert_eq!(
        dispatch("+I4K03i3EL0", args(&[0]), &mut state),
        Some(error(VIDEO_OUT_ERROR_INVALID_ADDRESS))
    );
}

#[test]
fn only_the_default_output_mode_is_reported_as_available() {
    let mut state = open_display();
    assert_eq!(
        dispatch(
            "Nv8c-Kb+DUM",
            args(&[VIDEO_OUT_HANDLE, 1, 0, 0, 0]),
            &mut state
        ),
        Some(VIDEO_OUT_TRUE)
    );
    assert_eq!(
        dispatch(
            "Nv8c-Kb+DUM",
            args(&[VIDEO_OUT_HANDLE, 0xd000_000a, 0, 0, 0]),
            &mut state
        ),
        Some(VIDEO_OUT_FALSE)
    );
    assert_eq!(
        dispatch(
            "Nv8c-Kb+DUM",
            args(&[VIDEO_OUT_HANDLE, 1, 0, 8, 0]),
            &mut state
        ),
        Some(error(VIDEO_OUT_ERROR_INVALID_VALUE))
    );
    assert_eq!(
        dispatch(
            "Nv8c-Kb+DUM",
            args(&[VIDEO_OUT_HANDLE, 1, 0, 0, 1]),
            &mut state
        ),
        Some(error(VIDEO_OUT_ERROR_INVALID_VALUE))
    );
}

#[test]
fn colour_adjustment_requires_the_declared_size_and_a_usable_gamma() {
    let mut state = open_display();
    let settings = [1.0_f32, 0.5, 2.0, f32::from_bits(0)];
    let broken = [f32::NAN, 1.0, 1.0, 0.0];
    assert_eq!(
        dispatch(
            "pv9CI5VC+R0",
            args(&[VIDEO_OUT_HANDLE, address(&settings), 16]),
            &mut state
        ),
        Some(SUCCESS)
    );
    assert_eq!(
        dispatch(
            "pv9CI5VC+R0",
            args(&[VIDEO_OUT_HANDLE, address(&settings), 8]),
            &mut state
        ),
        Some(error(VIDEO_OUT_ERROR_INVALID_VALUE))
    );
    assert_eq!(
        dispatch(
            "pv9CI5VC+R0",
            args(&[VIDEO_OUT_HANDLE, address(&broken), 16]),
            &mut state
        ),
        Some(error(VIDEO_OUT_ERROR_INVALID_VALUE))
    );
    assert_eq!(
        dispatch("pv9CI5VC+R0", args(&[VIDEO_OUT_HANDLE, 0, 16]), &mut state),
        Some(error(VIDEO_OUT_ERROR_INVALID_ADDRESS))
    );
}

#[test]
fn releasing_a_pegged_refresh_rate_needs_the_open_port() {
    let mut state = open_display();
    assert_eq!(
        dispatch("T4ucGB8CsnM", args(&[VIDEO_OUT_HANDLE]), &mut state),
        Some(SUCCESS)
    );
    state.video_out.is_open = false;
    assert_eq!(
        dispatch("T4ucGB8CsnM", args(&[VIDEO_OUT_HANDLE]), &mut state),
        Some(error(VIDEO_OUT_ERROR_INVALID_HANDLE))
    );
}
