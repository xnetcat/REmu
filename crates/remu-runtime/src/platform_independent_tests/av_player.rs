// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::time::{Duration, Instant};

use super::{ProbeDispatchState, sce_kernel_error};

#[test]
fn av_player_decoder_backpressure_trace_schedule_is_bounded() {
    assert_eq!(
        crate::av_player::av_player_next_backpressure_report(Some(Duration::from_secs(1))),
        Some(Duration::from_secs(2))
    );
    assert_eq!(
        crate::av_player::av_player_next_backpressure_report(Some(Duration::from_secs(32))),
        Some(Duration::from_secs(64))
    );
    assert_eq!(
        crate::av_player::av_player_next_backpressure_report(Some(Duration::from_secs(64))),
        None
    );
    assert_eq!(
        crate::av_player::av_player_next_backpressure_report(None),
        None
    );
}

#[test]
fn av_player_content_summary_distinguishes_live_and_constant_frames() {
    let constant = crate::av_player::av_player_byte_content_summary(&[16; 8]);
    assert_eq!(constant.minimum, 16);
    assert_eq!(constant.maximum, 16);
    assert_eq!(constant.nonzero_bytes, 8);

    let live = crate::av_player::av_player_byte_content_summary(&[0, 16, 64, 128, 255]);
    assert_eq!(live.minimum, 0);
    assert_eq!(live.maximum, 255);
    assert_eq!(live.nonzero_bytes, 4);
    assert_ne!(live.hash, constant.hash);

    let empty = crate::av_player::av_player_byte_content_summary(&[]);
    assert_eq!(empty.minimum, 0);
    assert_eq!(empty.maximum, 0);
    assert_eq!(empty.nonzero_bytes, 0);
}

#[test]
fn av_player_resolves_relative_file_urls_inside_app0_case_insensitively() {
    let root = std::env::temp_dir().join(format!("remu-avplayer-path-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let movie = root.join("shpc/content/movies/ui_titlemovie01.mp4");
    std::fs::create_dir_all(movie.parent().unwrap()).unwrap();
    std::fs::write(&movie, b"movie").unwrap();

    let resolved = crate::av_player_resolve_source(
        &root,
        "file://../../../shpc/Content/Movies/UI_TitleMovie01.mp4",
    )
    .unwrap();
    assert!(resolved.is_file(), "unresolved: {}", resolved.display());
    assert_eq!(std::fs::read(resolved).unwrap(), b"movie");

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn av_player_resolves_paths_without_dot_segments_unchanged() {
    // The dot-segment clamp runs for every source, not only relative ones. A path that has no
    // leading dot segments must reach the resolver exactly as the guest supplied it.
    let root = std::env::temp_dir().join(format!("remu-avplayer-plain-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let movie = root.join("content/movies/intro.mp4");
    std::fs::create_dir_all(movie.parent().unwrap()).unwrap();
    std::fs::write(&movie, b"movie").unwrap();

    for source in [
        "/app0/Content/Movies/Intro.mp4",
        "Content/Movies/Intro.mp4",
        "file:///app0/Content/Movies/Intro.mp4",
    ] {
        let resolved = crate::av_player_resolve_source(&root, source)
            .unwrap_or_else(|| panic!("unresolved: {source}"));
        assert_eq!(
            std::fs::read(resolved).unwrap(),
            b"movie",
            "source: {source}"
        );
    }

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn av_player_relative_paths_cannot_collapse_to_a_root_filename() {
    let root = std::env::temp_dir().join(format!("remu-avplayer-escape-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("outside.mp4"), b"inside decoy").unwrap();

    assert_eq!(
        crate::av_player_resolve_source(&root, "file://../../../outside.mp4"),
        None
    );
    assert_eq!(
        crate::av_player_resolve_source(&root, "../../../outside.mp4"),
        None
    );

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn av_player_initialization_accepts_the_gen5_layout_and_owns_handles() {
    let mut state = ProbeDispatchState::default();
    let mut initialization = [0_u8; 0x230];
    let initialization_size = initialization.len() as u64;
    initialization[..8].copy_from_slice(&initialization_size.to_ne_bytes());
    for offset in [16_usize, 24, 32, 40] {
        initialization[offset..offset + 8].copy_from_slice(&(0x1000 + offset as u64).to_ne_bytes());
    }
    let mut handle = 0_u64;
    assert_eq!(
        crate::av_player_set_log_callback([0x1234, 0x5678, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        crate::av_player_init_ex(
            [
                initialization.as_ptr().addr() as u64,
                (&raw mut handle).addr() as u64,
                0,
                0,
                0,
                0,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(handle, 1);
    assert_eq!(
        state.av_player.players[&handle]._initialization.len(),
        initialization.len()
    );
    let post_initialization = 1_u64;
    assert_eq!(
        crate::av_player_post_init(
            [
                handle,
                (&raw const post_initialization).addr() as u64,
                0,
                0,
                0,
                0,
            ],
            &state,
        ),
        0
    );
    let source = b"/app0/Media/intro.mp4\0";
    assert_eq!(
        crate::av_player_add_source(
            [handle, source.as_ptr().addr() as u64, 0, 0, 0, 0],
            &mut state,
        ),
        sce_kernel_error(crate::AV_PLAYER_ERROR_OPERATION_FAILED)
    );
    assert_eq!(crate::av_player_is_active(handle, &state), 0);
    assert_eq!(crate::av_player_close(handle, &mut state), 0);
    assert_eq!(
        crate::known_nid_name("o9eWRkSL+M4"),
        Some("sceAvPlayerInitEx")
    );
    assert_eq!(
        crate::known_nid_name("HD1YKVU26-M"),
        Some("sceAvPlayerPostInit")
    );
}

#[test]
fn av_player_source_is_active_while_ready_before_start_and_inactive_after_stop() {
    let Some(root) = av_player_generate_test_clip("ready-active") else {
        return;
    };
    let mut state = ProbeDispatchState::default();
    state.game_root = root.clone();
    let handle = av_player_test_player(&mut state);
    {
        let player = state.av_player.players.get_mut(&handle).unwrap();
        player.event_callback = 0x1234;
        player.event_object = 0x5678;
    }
    let source = b"/app0/media/clip.mp4\0";
    assert_eq!(
        crate::av_player_add_source(
            [handle, source.as_ptr().addr() as u64, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
    assert_eq!(state.av_player.pending_callbacks.len(), 1);
    assert_eq!(state.av_player.pending_callbacks[0].event, 2);
    assert_eq!(
        state.av_player.pending_callbacks[0].delivery,
        crate::AvPlayerEventDelivery::Controller,
        "Ready is an asynchronous controller-thread notification"
    );
    state.av_player.pending_callbacks.clear();
    assert_eq!(
        crate::av_player_is_active(handle, &state),
        1,
        "a source in the native Ready state is active before Start"
    );
    assert!(
        state.av_player.players[&handle].playing_since.is_none(),
        "Ready activity must not start the media clock"
    );

    assert_eq!(crate::av_player_start(handle, &mut state), 0);
    assert_eq!(crate::av_player_stop(handle, &mut state), 0);
    assert_eq!(
        crate::av_player_is_active(handle, &state),
        0,
        "Stop ends the active session even though the source remains attached"
    );
    assert_eq!(crate::av_player_close(handle, &mut state), 0);
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn av_player_playback_events_stay_on_the_initiating_guest_thread() {
    let mut state = ProbeDispatchState::default();
    let handle = av_player_test_player(&mut state);
    {
        let player = state.av_player.players.get_mut(&handle).unwrap();
        player.event_callback = 0x1234;
        player.event_object = 0x5678;
        player.source = Some("test-source".to_owned());
    }

    for (operation, expected_event) in [
        (
            crate::av_player_start as fn(u64, &mut ProbeDispatchState) -> u64,
            3,
        ),
        (crate::av_player_pause, 4),
        (crate::av_player_resume, 3),
        (crate::av_player_stop, 1),
    ] {
        assert_eq!(operation(handle, &mut state), 0);
        let callbacks = crate::take_av_player_calling_thread_callbacks(&mut state);
        assert_eq!(callbacks.len(), 1);
        assert_eq!(callbacks[0].event, expected_event);
        assert_eq!(callbacks[0].function, 0x1234);
        assert_eq!(callbacks[0].object, 0x5678);
        assert_eq!(
            callbacks[0].delivery,
            crate::AvPlayerEventDelivery::CallingThread
        );
        assert!(
            state.av_player.pending_callbacks.is_empty(),
            "a synchronous state notification must not be stealable by a later importer"
        );
    }

    assert_eq!(crate::av_player_close(handle, &mut state), 0);
}

#[test]
fn av_player_gen5_stream_info_is_one_based_and_exactly_32_bytes() {
    let mut stream_info = [0xab_u8; 40];
    crate::av_player_write_stream_info(
        stream_info.as_mut_ptr().addr() as u64,
        0,
        378,
        150,
        std::time::Duration::from_millis(9_109),
    );

    assert_eq!(
        u32::from_ne_bytes(stream_info[0..4].try_into().unwrap()),
        crate::AV_PLAYER_STREAM_TYPE_VIDEO
    );
    assert_eq!(crate::AV_PLAYER_STREAM_TYPE_VIDEO, 1);
    assert_eq!(
        u32::from_ne_bytes(stream_info[8..12].try_into().unwrap()),
        384
    );
    assert_eq!(
        u32::from_ne_bytes(stream_info[12..16].try_into().unwrap()),
        160
    );
    assert_eq!(
        u64::from_ne_bytes(stream_info[24..32].try_into().unwrap()),
        9_109
    );
    assert_eq!(&stream_info[32..], &[0xab; 8]);

    crate::av_player_write_stream_info(
        stream_info.as_mut_ptr().addr() as u64,
        1,
        378,
        150,
        std::time::Duration::from_millis(9_109),
    );
    assert_eq!(
        u32::from_ne_bytes(stream_info[0..4].try_into().unwrap()),
        crate::AV_PLAYER_STREAM_TYPE_AUDIO
    );
    assert_eq!(crate::AV_PLAYER_STREAM_TYPE_AUDIO, 2);
}

#[test]
fn av_player_gen5_extended_stream_info_preserves_size_and_reports_frame_rate() {
    let mut stream_info = [0xab_u8; 112];
    stream_info[..8].copy_from_slice(&104_u64.to_ne_bytes());
    crate::av_player_write_stream_info_ex(
        stream_info.as_mut_ptr().addr() as u64,
        0,
        1_440,
        1_080,
        60.0,
        std::time::Duration::from_millis(43_166),
    );

    assert_eq!(
        u64::from_ne_bytes(stream_info[0..8].try_into().unwrap()),
        104
    );
    assert_eq!(
        u32::from_ne_bytes(stream_info[8..12].try_into().unwrap()),
        crate::AV_PLAYER_STREAM_TYPE_VIDEO
    );
    assert_eq!(
        u32::from_ne_bytes(stream_info[16..20].try_into().unwrap()),
        1_440
    );
    assert_eq!(
        u32::from_ne_bytes(stream_info[20..24].try_into().unwrap()),
        1_080
    );
    assert_eq!(
        f64::from_ne_bytes(stream_info[64..72].try_into().unwrap()),
        60.0
    );
    assert_eq!(
        u64::from_ne_bytes(stream_info[96..104].try_into().unwrap()),
        43_166
    );
    assert_eq!(&stream_info[104..], &[0xab; 8]);
}

#[test]
fn av_player_gen5_extended_frame_info_uses_right_and_bottom_crop_fields() {
    assert_eq!(
        crate::av_player::av_player_nv12_layout(378, 150, true),
        Some((512, 122_880)),
        "the extended ABI uses its explicit pitch and the GPU-aligned storage height"
    );
    assert_eq!(
        crate::av_player::av_player_nv12_layout(1_920, 1_080, true),
        Some((2_048, 3_342_336)),
        "1080-line NV12 must reserve the 1088-line texture extent consumed by the guest"
    );
    let mut frame_info = [0xab_u8; 112];
    crate::av_player_write_video_frame_info(
        frame_info.as_mut_ptr().addr() as u64,
        0x1234_5600,
        4_238,
        378,
        150,
        512,
        29.97,
        true,
    );

    assert_eq!(
        u64::from_ne_bytes(frame_info[0..8].try_into().unwrap()),
        0x1234_5600
    );
    assert_eq!(
        u64::from_ne_bytes(frame_info[16..24].try_into().unwrap()),
        4_238
    );
    assert_eq!(
        u32::from_ne_bytes(frame_info[24..28].try_into().unwrap()),
        378,
        "the width field carries the displayable extent, not the storage pitch"
    );
    assert_eq!(
        u32::from_ne_bytes(frame_info[28..32].try_into().unwrap()),
        150
    );
    assert_eq!(
        u32::from_ne_bytes(frame_info[44..48].try_into().unwrap()),
        0
    );
    assert_eq!(
        u32::from_ne_bytes(frame_info[48..52].try_into().unwrap()),
        0,
        "a displayable width leaves nothing for the crop fields to trim"
    );
    assert_eq!(
        u32::from_ne_bytes(frame_info[52..56].try_into().unwrap()),
        0
    );
    assert_eq!(
        u32::from_ne_bytes(frame_info[56..60].try_into().unwrap()),
        0
    );
    assert_eq!(
        u32::from_ne_bytes(frame_info[60..64].try_into().unwrap()),
        512
    );
    assert_eq!(&frame_info[64..67], &[8, 8, 0]);
    assert_eq!(
        f64::from_bits(u64::from_ne_bytes(
            frame_info[0x48..0x50].try_into().unwrap()
        )),
        29.97
    );
    assert_eq!(&frame_info[104..], &[0xab; 8]);
}

/// Builds a fresh player owned by `state` and returns its handle. The initialization block
/// mirrors the Gen5 InitDataEx layout the live titles supply.
fn av_player_test_player(state: &mut ProbeDispatchState) -> u64 {
    let mut initialization = [0_u8; 0x230];
    let initialization_size = initialization.len() as u64;
    initialization[..8].copy_from_slice(&initialization_size.to_ne_bytes());
    for offset in [16_usize, 24, 32, 40] {
        initialization[offset..offset + 8].copy_from_slice(&(0x1000 + offset as u64).to_ne_bytes());
    }
    let mut handle = 0_u64;
    assert_eq!(
        crate::av_player_init_ex(
            [
                initialization.as_ptr().addr() as u64,
                (&raw mut handle).addr() as u64,
                0,
                0,
                0,
                0,
            ],
            state,
        ),
        0
    );
    // The initialization block above carries a dummy texture allocator address the tests cannot
    // call; clearing it routes frame delivery through the fallback buffer allocation.
    state
        .av_player
        .players
        .get_mut(&handle)
        .expect("initialized player exists")
        .allocate_texture_callback = 0;
    handle
}

#[test]
fn av_player_stall_snapshot_distinguishes_decoder_and_guest_progress() {
    use std::sync::atomic::Ordering;

    let mut state = ProbeDispatchState::default();
    let handle = av_player_test_player(&mut state);
    let (sender, frames) = std::sync::mpsc::sync_channel(1);
    sender.try_send(vec![0x80; 16]).unwrap();
    let progress = std::sync::Arc::new(crate::AvPlayerDecoderProgress::default());
    progress.produced_frames.store(17, Ordering::Release);
    progress.last_progress_millis.store(1, Ordering::Release);
    progress.queue_blocked.store(true, Ordering::Release);
    {
        let player = state.av_player.players.get_mut(&handle).unwrap();
        player.source = Some("diagnostic.mp4".to_owned());
        player.active_session = true;
        player.playing_since = Some(Instant::now());
        player.next_video_frame = 4;
        player.last_video_delivery = Some(Instant::now());
        player.video_decoder = Some(crate::AvPlayerDecoder {
            child: None,
            frames: std::sync::Mutex::new(frames),
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            worker: None,
            progress: Some(progress),
        });
    }

    let snapshots = crate::av_player::av_player_stall_snapshots(&state);
    assert_eq!(snapshots.len(), 1);
    let snapshot = &snapshots[0];
    assert_eq!(snapshot.handle, handle);
    assert_eq!(snapshot.source.as_deref(), Some("diagnostic.mp4"));
    assert!(snapshot.active);
    assert!(snapshot.playing);
    assert_eq!(snapshot.delivered_video_frames, 4);
    assert!(
        snapshot
            .video_delivery_age_millis
            .is_some_and(|age| age < 1_000)
    );
    assert!(snapshot.decoder_present);
    assert_eq!(snapshot.decoder_produced_frames, 17);
    assert!(
        snapshot
            .decoder_progress_age_millis
            .is_some_and(|age| age < 1_000)
    );
    assert!(snapshot.decoder_queue_blocked);
}

/// Generates a two-second 128x96 test clip (mpeg4 video + AAC sine audio) with the same ffmpeg
/// binary the runtime decoder requires. Returns None — after logging loudly — when the tools
/// are unavailable, mirroring the accepted environmental-skip pattern.
fn av_player_generate_test_clip(name: &str) -> Option<std::path::PathBuf> {
    let (Some(ffmpeg), Some(_ffprobe)) = (
        crate::av_player_tool("ffmpeg", "REMU_FFMPEG"),
        crate::av_player_tool("ffprobe", "REMU_FFPROBE"),
    ) else {
        eprintln!("[REMu][TEST-SKIP] {name}: ffmpeg/ffprobe unavailable, media test skipped");
        return None;
    };
    let root =
        std::env::temp_dir().join(format!("remu-avplayer-clip-{}-{name}", std::process::id()));
    std::fs::create_dir_all(root.join("media")).ok()?;
    let clip = root.join("media/clip.mp4");
    if !clip.is_file() {
        let status = std::process::Command::new(ffmpeg)
            .args([
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=2:size=128x96:rate=30",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2",
                "-c:v",
                "mpeg4",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
            ])
            .arg(&clip)
            .status()
            .ok()?;
        if !status.success() {
            eprintln!("[REMu][TEST-SKIP] {name}: ffmpeg could not generate the test clip");
            return None;
        }
    }
    Some(root)
}

fn av_player_frame_timestamp(frame_info: &[u8; 112]) -> u64 {
    u64::from_ne_bytes(frame_info[16..24].try_into().unwrap())
}

fn av_player_wait_for_video_frame(handle: u64, output: u64, state: &mut ProbeDispatchState) -> u64 {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let result = crate::av_player_get_video_data([handle, output, 0, 0, 0, 0], true, state);
        if result != 0 || Instant::now() >= deadline {
            return result;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn av_player_wait_for_audio_frame(handle: u64, output: u64, state: &mut ProbeDispatchState) -> u64 {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let result = crate::av_player_get_audio_data([handle, output, 0, 0, 0, 0], state);
        if result != 0 || Instant::now() >= deadline {
            return result;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn av_player_pause_freezes_data_clock_and_resumes_monotonically() {
    let Some(root) = av_player_generate_test_clip("prepare-pattern") else {
        return;
    };
    let mut state = ProbeDispatchState::default();
    state.game_root = root.clone();
    let handle = av_player_test_player(&mut state);
    let source = b"/app0/media/clip.mp4\0";
    assert_eq!(
        crate::av_player_add_source(
            [handle, source.as_ptr().addr() as u64, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
    assert_eq!(crate::av_player_start(handle, &mut state), 0);

    let mut frame_info = [0_u8; 112];
    let output = frame_info.as_mut_ptr().addr() as u64;
    // The first frame is always due once the asynchronous decoder has produced it.
    assert_eq!(
        av_player_wait_for_video_frame(handle, output, &mut state),
        1,
        "first video frame must be delivered"
    );
    let time_before_pause = crate::av_player_current_time(handle, &mut state);
    let frames_before_pause = {
        let player = &state.av_player.players[&handle];
        (player.next_video_frame, player.next_audio_frame)
    };

    // Native Pause freezes both public data delivery and CurrentTime. Decoder workers may retain
    // their private queues, but GetVideoData/GetAudioData release nothing until Resume.
    assert_eq!(crate::av_player_pause(handle, &mut state), 0);
    let mut audio_info = [0_u8; 40];
    let audio_output = audio_info.as_mut_ptr().addr() as u64;
    for _ in 0..16 {
        state.video_out.completed_flip_count += 4;
        assert_eq!(
            crate::av_player_get_video_data([handle, output, 0, 0, 0, 0], true, &mut state),
            0,
            "a paused player must not release video"
        );
        assert_eq!(
            crate::av_player_get_audio_data([handle, audio_output, 0, 0, 0, 0], &mut state),
            0,
            "a paused player must not release audio"
        );
    }
    let paused_player = &state.av_player.players[&handle];
    assert!(
        paused_player.playing_since.is_none(),
        "Pause must stop the running wall-clock segment"
    );
    assert_eq!(
        (
            paused_player.next_video_frame,
            paused_player.next_audio_frame
        ),
        frames_before_pause,
        "Pause must not advance public stream positions"
    );
    assert_eq!(
        crate::av_player_is_active(handle, &state),
        1,
        "a paused player is still an active playback session"
    );
    assert_eq!(
        crate::av_player_current_time(handle, &mut state),
        time_before_pause,
        "CurrentTime must remain frozen throughout Pause"
    );

    // Resume: delivery continues, frames stay monotonic, and the player clock advances.
    assert_eq!(crate::av_player_resume(handle, &mut state), 0);
    assert_eq!(
        av_player_wait_for_audio_frame(handle, audio_output, &mut state),
        1,
        "audio must establish the clock master after Resume"
    );
    std::thread::sleep(Duration::from_millis(40));
    let mut last_timestamp = av_player_frame_timestamp(&frame_info);
    let mut resumed_frames = 0_u64;
    for _ in 0..16 {
        if crate::av_player_get_video_data([handle, output, 0, 0, 0, 0], true, &mut state) == 1 {
            let timestamp = av_player_frame_timestamp(&frame_info);
            assert!(
                timestamp > last_timestamp,
                "resumed timestamps must stay monotonic: {timestamp} after {last_timestamp}"
            );
            last_timestamp = timestamp;
            resumed_frames += 1;
        }
    }
    assert!(
        resumed_frames >= 1,
        "resumed playback must deliver more video frames"
    );
    let resumed_time = crate::av_player_current_time(handle, &mut state);
    assert!(
        resumed_time >= time_before_pause,
        "CurrentTime must be monotonic"
    );
    assert_eq!(crate::av_player_close(handle, &mut state), 0);
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn av_player_video_only_clock_advances_before_dequeue_but_stays_bounded() {
    let mut state = ProbeDispatchState::default();
    let handle = av_player_test_player(&mut state);
    {
        let player = state.av_player.players.get_mut(&handle).unwrap();
        player.source = Some("video-only.mp4".to_owned());
        player.has_audio = false;
        player.duration = Duration::from_secs(9);
    }
    assert_eq!(crate::av_player_start(handle, &mut state), 0);
    {
        let player = state.av_player.players.get_mut(&handle).unwrap();
        player.playing_since = Some(Instant::now() - Duration::from_millis(4_250));
        player.presentation_anchor_host = Some(Instant::now() - Duration::from_millis(4_250));
    }

    let player = &state.av_player.players[&handle];
    assert_eq!(
        player.next_video_frame, 0,
        "the regression requires that no video frame has been dequeued"
    );
    assert!(
        (1..=67).contains(&crate::av_player_current_time(handle, &mut state)),
        "the start anchor must advance enough to wake the poller but by no more than two frames"
    );

    {
        let player = state.av_player.players.get_mut(&handle).unwrap();
        player.presentation_anchor_media = Duration::from_millis(4_238);
        player.presentation_anchor_host = Some(Instant::now() - Duration::from_secs(30));
        player.presentation_clock_reported = Duration::from_millis(4_238);
        player.presentation_master_delivered = true;
        player.next_video_frame = 128;
    }
    let current = crate::av_player_current_time(handle, &mut state);
    assert!(
        (4_238..=4_305).contains(&current),
        "a 30 second host stall at frame 128 must remain near its 4238 ms media timestamp, got {current}"
    );
    assert_ne!(
        current, 9_000,
        "host time alone must not complete a slow video-only stream"
    );
}

#[test]
fn av_player_video_only_clock_advances_between_frames_without_escaping_media_progress() {
    let mut state = ProbeDispatchState::default();
    let handle = av_player_test_player(&mut state);
    {
        let player = state.av_player.players.get_mut(&handle).unwrap();
        player.source = Some("video-only.mp4".to_owned());
        player.has_audio = false;
        player.duration = Duration::from_millis(9_109);
    }
    assert_eq!(crate::av_player_start(handle, &mut state), 0);
    {
        let player = state.av_player.players.get_mut(&handle).unwrap();
        player.presentation_anchor_media = Duration::from_millis(4_238);
        player.presentation_anchor_host = Some(Instant::now() - Duration::from_millis(20));
        player.presentation_clock_reported = Duration::from_millis(4_238);
        player.presentation_master_delivered = true;
    }
    let between_frames = crate::av_player_current_time(handle, &mut state);
    assert!(
        (4_238..=4_305).contains(&between_frames) && between_frames > 4_238,
        "host elapsed time must advance the clock between video deliveries, got {between_frames}"
    );

    // Delivering the next 30 fps timestamp re-anchors the host segment. The public result remains
    // monotonic, while an arbitrarily long subsequent stall is still bounded to two frame times.
    {
        let player = state.av_player.players.get_mut(&handle).unwrap();
        player.presentation_anchor_media = Duration::from_millis(4_271);
        player.presentation_anchor_host = Some(Instant::now() - Duration::from_millis(10));
    }
    let reanchored = crate::av_player_current_time(handle, &mut state);
    assert!(reanchored >= between_frames);
    {
        state
            .av_player
            .players
            .get_mut(&handle)
            .unwrap()
            .presentation_anchor_host = Some(Instant::now() - Duration::from_secs(30));
    }
    assert!(
        crate::av_player_current_time(handle, &mut state) <= 4_338,
        "video-only clock may lead the 4271 ms frame by at most two frame periods"
    );
}

#[test]
fn av_player_sync_none_video_only_clock_uses_start_timeline() {
    let mut state = ProbeDispatchState::default();
    let handle = av_player_test_player(&mut state);
    {
        let player = state.av_player.players.get_mut(&handle).unwrap();
        player.source = Some("video-only.mp4".to_owned());
        player.has_audio = false;
        player.duration = Duration::from_secs(9);
    }
    assert_eq!(crate::av_player_start(handle, &mut state), 0);
    {
        let player = state.av_player.players.get_mut(&handle).unwrap();
        let four_seconds_ago = Instant::now() - Duration::from_secs(4);
        player.playing_since = Some(four_seconds_ago);
        player.presentation_anchor_host = Some(four_seconds_ago);
    }

    let default_time = crate::av_player_current_time(handle, &mut state);
    assert!(
        default_time <= 67,
        "Default sync retains REmu's bounded video clock, got {default_time}"
    );
    assert_eq!(
        crate::av_player_set_av_sync_mode([handle, 1, 0, 0, 0, 0], &mut state),
        0
    );
    let caller_controlled_time = crate::av_player_current_time(handle, &mut state);
    assert!(
        (3_900..4_500).contains(&caller_controlled_time),
        "None sync must expose the Start wall-clock timeline independently of dequeue, got {caller_controlled_time}"
    );
}

#[test]
fn av_player_audio_master_waits_for_audio_then_tracks_its_timestamp() {
    let mut state = ProbeDispatchState::default();
    let handle = av_player_test_player(&mut state);
    {
        let player = state.av_player.players.get_mut(&handle).unwrap();
        player.source = Some("audio-video.mp4".to_owned());
        player.has_audio = true;
        player.duration = Duration::from_secs(9);
    }
    assert_eq!(crate::av_player_start(handle, &mut state), 0);
    {
        let player = state.av_player.players.get_mut(&handle).unwrap();
        player.presentation_anchor_host = Some(Instant::now() - Duration::from_secs(4));
    }
    assert_eq!(
        crate::av_player_current_time(handle, &mut state),
        0,
        "an audio source has no running master clock before its first audio sample"
    );

    {
        let player = state.av_player.players.get_mut(&handle).unwrap();
        player.presentation_anchor_media = Duration::from_millis(2_000);
        player.presentation_anchor_host = Some(Instant::now() - Duration::from_millis(100));
        player.presentation_master_delivered = true;
        player.next_video_frame = 240;
    }
    let audio_time = crate::av_player_current_time(handle, &mut state);
    assert!(
        (2_000..3_000).contains(&audio_time),
        "audio PTS plus host elapsed must master CurrentTime, got {audio_time}"
    );
}

#[test]
fn av_player_actual_eof_exposes_duration_and_queues_one_controller_stop() {
    let mut state = ProbeDispatchState::default();
    let handle = av_player_test_player(&mut state);
    let (sender, frames) = std::sync::mpsc::sync_channel(1);
    drop(sender);
    {
        let player = state.av_player.players.get_mut(&handle).unwrap();
        player.source = Some("video-only.mp4".to_owned());
        player.duration = Duration::from_millis(9_109);
        player.active_session = true;
        player.playing_since = Some(Instant::now());
        player.presentation_anchor_media = Duration::from_millis(9_000);
        player.presentation_anchor_host = Some(Instant::now());
        player.presentation_clock_reported = Duration::from_millis(9_000);
        player.event_callback = 0x1234;
        player.event_object = 0x5678;
        player.video_decoder = Some(crate::AvPlayerDecoder {
            child: None,
            frames: std::sync::Mutex::new(frames),
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            worker: None,
            progress: Some(std::sync::Arc::new(
                crate::AvPlayerDecoderProgress::default(),
            )),
        });
    }
    let mut frame_info = [0_u8; 112];
    assert_eq!(
        crate::av_player_get_video_data(
            [handle, frame_info.as_mut_ptr().addr() as u64, 0, 0, 0, 0],
            true,
            &mut state,
        ),
        0,
        "a disconnected decoder reports EOF rather than a frame"
    );
    assert!(state.av_player.players[&handle].end_of_stream);
    assert_eq!(state.av_player.pending_callbacks.len(), 1);
    assert_eq!(state.av_player.pending_callbacks[0].event, 1);
    assert_eq!(state.av_player.pending_callbacks[0].function, 0x1234);
    assert_eq!(state.av_player.pending_callbacks[0].object, 0x5678);
    assert_eq!(
        state.av_player.pending_callbacks[0].delivery,
        crate::AvPlayerEventDelivery::Controller,
        "natural EOF is reported asynchronously by the controller"
    );
    assert_eq!(
        crate::av_player_current_time(handle, &mut state),
        9_109,
        "decoder EOF, unlike a running clock, must expose the exact duration"
    );
    assert_eq!(
        crate::av_player_get_video_data(
            [handle, frame_info.as_mut_ptr().addr() as u64, 0, 0, 0, 0],
            true,
            &mut state,
        ),
        0
    );
    assert_eq!(
        state.av_player.pending_callbacks.len(),
        1,
        "later polls must not queue duplicate EOF Stop events"
    );
}

#[test]
fn av_player_multiplexed_source_waits_for_video_and_audio_to_drain() {
    let mut state = ProbeDispatchState::default();
    let handle = av_player_test_player(&mut state);
    let (video_sender, video_frames) = std::sync::mpsc::sync_channel(1);
    let (audio_sender, audio_frames) = std::sync::mpsc::sync_channel(1);
    drop(video_sender);
    drop(audio_sender);
    {
        let player = state.av_player.players.get_mut(&handle).unwrap();
        player.source = Some("audio-video.mp4".to_owned());
        player.has_audio = true;
        player.duration = Duration::from_secs(4);
        player.active_session = true;
        player.playing_since = Some(Instant::now());
        player.event_callback = 0x1234;
        player.event_object = 0x5678;
        player.video_decoder = Some(crate::AvPlayerDecoder {
            child: None,
            frames: std::sync::Mutex::new(video_frames),
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            worker: None,
            progress: Some(std::sync::Arc::new(
                crate::AvPlayerDecoderProgress::default(),
            )),
        });
        player.audio_decoder = Some(crate::AvPlayerDecoder {
            child: None,
            frames: std::sync::Mutex::new(audio_frames),
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            worker: None,
            progress: Some(std::sync::Arc::new(
                crate::AvPlayerDecoderProgress::default(),
            )),
        });
    }

    let mut video_info = [0_u8; 112];
    assert_eq!(
        crate::av_player_get_video_data(
            [handle, video_info.as_mut_ptr().addr() as u64, 0, 0, 0, 0,],
            true,
            &mut state,
        ),
        0
    );
    {
        let player = &state.av_player.players[&handle];
        assert!(player.video_decoder_drained);
        assert!(!player.audio_decoder_drained);
        assert!(player.active_session);
        assert!(!player.end_of_stream);
    }
    assert!(
        state.av_player.pending_callbacks.is_empty(),
        "video EOF alone must not advance a multiplexed cutscene"
    );

    let mut audio_info = [0_u8; 40];
    assert_eq!(
        crate::av_player_get_audio_data(
            [handle, audio_info.as_mut_ptr().addr() as u64, 0, 0, 0, 0,],
            &mut state,
        ),
        0
    );
    let player = &state.av_player.players[&handle];
    assert!(player.video_decoder_drained);
    assert!(player.audio_decoder_drained);
    assert!(!player.active_session);
    assert!(player.end_of_stream);
    assert_eq!(state.av_player.pending_callbacks.len(), 1);
    assert_eq!(
        state.av_player.pending_callbacks[0].delivery,
        crate::AvPlayerEventDelivery::Controller
    );
}

#[test]
fn av_player_loop_restarts_only_after_every_enabled_stream_drains() {
    let mut state = ProbeDispatchState::default();
    let handle = av_player_test_player(&mut state);
    let (video_sender, video_frames) = std::sync::mpsc::sync_channel(1);
    let (audio_sender, audio_frames) = std::sync::mpsc::sync_channel(1);
    drop(video_sender);
    drop(audio_sender);
    {
        let player = state.av_player.players.get_mut(&handle).unwrap();
        player.source = Some("audio-video.mp4".to_owned());
        player.has_audio = true;
        player.active_session = true;
        player.playing_since = Some(Instant::now());
        player.looping = true;
        player.video_decoder = Some(crate::AvPlayerDecoder {
            child: None,
            frames: std::sync::Mutex::new(video_frames),
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            worker: None,
            progress: Some(std::sync::Arc::new(
                crate::AvPlayerDecoderProgress::default(),
            )),
        });
        player.audio_decoder = Some(crate::AvPlayerDecoder {
            child: None,
            frames: std::sync::Mutex::new(audio_frames),
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            worker: None,
            progress: Some(std::sync::Arc::new(
                crate::AvPlayerDecoderProgress::default(),
            )),
        });
    }

    let mut video_info = [0_u8; 112];
    assert_eq!(
        crate::av_player_get_video_data(
            [handle, video_info.as_mut_ptr().addr() as u64, 0, 0, 0, 0,],
            true,
            &mut state,
        ),
        0
    );
    assert!(
        state.av_player.players[&handle].video_decoder_drained,
        "video EOF must remain visible while audio drains"
    );

    let mut audio_info = [0_u8; 40];
    assert_eq!(
        crate::av_player_get_audio_data(
            [handle, audio_info.as_mut_ptr().addr() as u64, 0, 0, 0, 0,],
            &mut state,
        ),
        0
    );
    let player = &state.av_player.players[&handle];
    assert!(player.active_session);
    assert!(player.playing_since.is_some());
    assert!(!player.end_of_stream);
    assert!(!player.video_decoder_drained);
    assert!(!player.audio_decoder_drained);
    assert!(state.av_player.pending_callbacks.is_empty());
}

#[test]
fn av_player_jump_to_time_seeks_the_decoder_instead_of_replaying_from_zero() {
    let Some(root) = av_player_generate_test_clip("jump-to-time") else {
        return;
    };
    let mut state = ProbeDispatchState::default();
    state.game_root = root.clone();
    let handle = av_player_test_player(&mut state);
    let source = b"/app0/media/clip.mp4\0";
    assert_eq!(
        crate::av_player_add_source(
            [handle, source.as_ptr().addr() as u64, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
    assert_eq!(crate::av_player_start(handle, &mut state), 0);
    assert_eq!(
        crate::av_player_jump_to_time([handle, 1_000, 0, 0, 0, 0], &mut state),
        0
    );
    {
        let player = &state.av_player.players[&handle];
        assert_eq!(player.decoder_seek, Duration::from_millis(1_000));
        assert!(
            player.playing_since.is_some(),
            "a jump during playback must keep playing"
        );
    }
    let mut frame_info = [0_u8; 112];
    let output = frame_info.as_mut_ptr().addr() as u64;
    assert_eq!(
        av_player_wait_for_video_frame(handle, output, &mut state),
        1,
        "the frame at the seek target must be delivered"
    );
    let timestamp = av_player_frame_timestamp(&frame_info);
    // 30 fps content: the first delivered frame after a 1000 ms jump is within one frame.
    assert!(
        (967..=1_034).contains(&timestamp),
        "expected the first frame near 1000 ms, got {timestamp}"
    );
    assert!(crate::av_player_current_time(handle, &mut state) >= 1_000);
    assert_eq!(crate::av_player_close(handle, &mut state), 0);
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn av_player_change_stream_accepts_active_streams_and_rejects_unknown_ones() {
    let mut state = ProbeDispatchState::default();
    let handle = av_player_test_player(&mut state);
    // A player without an audio track exposes exactly one single-representation stream.
    assert_eq!(
        crate::av_player_change_stream([handle, 0, 0, 0, 0, 0], &state),
        0
    );
    assert_eq!(
        crate::av_player_change_stream([handle, 1, 0, 0, 0, 0], &state),
        sce_kernel_error(crate::AV_PLAYER_ERROR_INVALID_PARAMS)
    );
    assert_eq!(
        crate::av_player_change_stream([0xdead, 0, 0, 0, 0, 0], &state),
        sce_kernel_error(crate::AV_PLAYER_ERROR_INVALID_PARAMS)
    );
    assert_eq!(
        crate::known_nid_name("buMCiJftcfw"),
        Some("sceAvPlayerChangeStream")
    );
}

#[test]
fn av_player_sync_none_hands_video_timing_to_the_caller() {
    let mut state = ProbeDispatchState::default();
    let handle = av_player_test_player(&mut state);
    let (sender, receiver) = std::sync::mpsc::sync_channel(4);
    sender.try_send(vec![0x80; 16 * 16 * 3 / 2]).unwrap();
    {
        let player = state.av_player.players.get_mut(&handle).unwrap();
        player.source = Some("video-only.mp4".to_owned());
        player.width = 16;
        player.height = 16;
        player.frames_per_second = 30.0;
        player.active_session = true;
        player.playing_since = Some(Instant::now());
        player.next_video_frame = 240;
        player.video_decoder = Some(crate::AvPlayerDecoder {
            child: None,
            frames: std::sync::Mutex::new(receiver),
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            worker: None,
            progress: Some(std::sync::Arc::new(
                crate::AvPlayerDecoderProgress::default(),
            )),
        });
    }
    let mut frame_info = [0_u8; 112];
    let output = frame_info.as_mut_ptr().addr() as u64;
    assert_eq!(
        crate::av_player_get_video_data([handle, output, 0, 0, 0, 0], true, &mut state),
        0,
        "Default sync must withhold frame 240 before its 8000 ms timestamp"
    );
    assert_eq!(
        crate::av_player_set_av_sync_mode([handle, 1, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(state.av_player.players[&handle].av_sync_mode, 1);
    assert_eq!(
        crate::av_player_get_video_data([handle, output, 0, 0, 0, 0], true, &mut state),
        1,
        "None sync must release an available frame without timestamp gating"
    );
    let delivered_timestamp = av_player_frame_timestamp(&frame_info);
    assert_eq!(delivered_timestamp, 8_000);
    let current_time = crate::av_player_current_time(handle, &mut state);
    assert!(
        current_time < delivered_timestamp,
        "dequeueing an 8000 ms frame in None mode must not jump the independent public clock, got {current_time}"
    );
    assert_eq!(
        crate::av_player_set_av_sync_mode([handle, 2, 0, 0, 0, 0], &mut state),
        sce_kernel_error(crate::AV_PLAYER_ERROR_INVALID_PARAMS)
    );
    assert_eq!(
        state.av_player.players[&handle].av_sync_mode, 1,
        "an invalid mode must not replace the last accepted mode"
    );
    drop(sender);
}

#[test]
fn av_player_trick_speed_scales_the_media_clock_and_rejects_reverse_rates() {
    let mut state = ProbeDispatchState::default();
    let handle = av_player_test_player(&mut state);
    assert_eq!(
        crate::av_player_set_trick_speed([handle, 200, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(state.av_player.players[&handle].trick_speed_percent, 200);
    assert_eq!(
        crate::av_player_set_trick_speed([handle, 0, 0, 0, 0, 0], &mut state),
        sce_kernel_error(crate::AV_PLAYER_ERROR_INVALID_PARAMS)
    );
    let reverse = u64::from((-100_i32) as u32);
    assert_eq!(
        crate::av_player_set_trick_speed([handle, reverse, 0, 0, 0, 0], &mut state),
        sce_kernel_error(crate::AV_PLAYER_ERROR_OPERATION_FAILED)
    );
    assert_eq!(state.av_player.players[&handle].trick_speed_percent, 200);
    // A doubled rate reaches the second 30 fps frame after roughly 17 ms of wall time.
    assert!(crate::av_player_video_frame_is_due(
        0,
        30.0,
        Duration::from_millis(0)
    ));
    assert!(crate::av_player_video_frame_is_due(
        1,
        30.0,
        Duration::from_millis(17).mul_f64(2.0)
    ));
    assert_eq!(
        crate::known_nid_name("av8Z++94rs0"),
        Some("sceAvPlayerSetTrickSpeed")
    );
}

#[test]
fn av_player_video_clock_releases_frames_from_media_time() {
    assert!(crate::av_player_video_frame_is_due(
        0,
        29.97,
        Duration::ZERO
    ));
    assert!(!crate::av_player_video_frame_is_due(
        1,
        29.97,
        Duration::from_millis(16)
    ));
    assert!(crate::av_player_video_frame_is_due(
        1,
        29.97,
        Duration::from_millis(34)
    ));
    assert!(!crate::av_player_video_frame_is_due(
        2,
        29.97,
        Duration::from_millis(34)
    ));
    assert!(crate::av_player_video_frame_is_due(
        2,
        29.97,
        Duration::from_millis(67)
    ));
}

#[test]
fn av_player_video_catch_up_preserves_first_frame_then_tracks_wall_clock() {
    let now = Instant::now();
    assert_eq!(
        crate::av_player_video_catch_up_target(
            None,
            0,
            29.97,
            Duration::from_millis(9_109),
            378 * 150 * 3 / 2,
            now + Duration::from_secs(30),
        ),
        0
    );

    let origin = Some((now, 1));
    assert_eq!(
        crate::av_player_video_catch_up_target(
            origin,
            1,
            29.97,
            Duration::from_millis(9_109),
            378 * 150 * 3 / 2,
            now + Duration::from_millis(550),
        ),
        17
    );
    assert_eq!(
        crate::av_player_video_catch_up_target(
            origin,
            200,
            29.97,
            Duration::from_millis(9_109),
            378 * 150 * 3 / 2,
            now + Duration::from_secs(30),
        ),
        272
    );
}

#[test]
fn av_player_video_catch_up_bounds_large_frame_drains_by_bytes() {
    let now = Instant::now();
    let twelve_mibibyte_frame = 12 * 1_024 * 1_024;
    assert_eq!(
        crate::av_player_video_catch_up_target(
            Some((now, 1)),
            1,
            60.0,
            Duration::from_secs(60),
            twelve_mibibyte_frame,
            now + Duration::from_secs(10),
        ),
        6
    );
}
