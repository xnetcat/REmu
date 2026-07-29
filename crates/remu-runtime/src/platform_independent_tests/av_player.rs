// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::time::{Duration, Instant};

use super::{ProbeDispatchState, sce_kernel_error};

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
    let source = b"/app0/media/clip.mp4\0";
    assert_eq!(
        crate::av_player_add_source(
            [handle, source.as_ptr().addr() as u64, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
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
        512
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
        134
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
    let time_before_pause = crate::av_player_current_time(handle, &state);
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
        crate::av_player_current_time(handle, &state),
        time_before_pause,
        "CurrentTime must remain frozen throughout Pause"
    );

    // Resume: delivery continues, frames stay monotonic, and the player clock advances.
    assert_eq!(crate::av_player_resume(handle, &mut state), 0);
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
    let resumed_time = crate::av_player_current_time(handle, &state);
    assert!(
        resumed_time >= time_before_pause,
        "CurrentTime must be monotonic"
    );
    assert_eq!(crate::av_player_close(handle, &mut state), 0);
    std::fs::remove_dir_all(&root).ok();
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
    assert!(crate::av_player_current_time(handle, &state) >= 1_000);
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
