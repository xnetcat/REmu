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
fn av_player_gen5_stream_info_is_zero_based_and_exactly_32_bytes() {
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

#[test]
fn av_player_video_clock_releases_frames_from_completed_flips() {
    let mut origin = None;
    let interval = 1_000_000 / 60;

    assert!(crate::av_player_video_frame_is_due(
        &mut origin,
        0,
        29.97,
        100,
        interval
    ));
    assert!(!crate::av_player_video_frame_is_due(
        &mut origin,
        1,
        29.97,
        101,
        interval
    ));
    assert!(crate::av_player_video_frame_is_due(
        &mut origin,
        1,
        29.97,
        102,
        interval
    ));
    assert!(!crate::av_player_video_frame_is_due(
        &mut origin,
        2,
        29.97,
        102,
        interval
    ));
    assert!(crate::av_player_video_frame_is_due(
        &mut origin,
        2,
        29.97,
        104,
        interval
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
