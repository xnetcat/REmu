// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    ProbeDispatchState, play_go_get_chunk_id, play_go_get_install_speed, play_go_get_locus,
    play_go_initialize, play_go_open, sce_kernel_error,
};

fn open_playgo_state() -> ProbeDispatchState {
    let mut state = ProbeDispatchState::default();
    state.services.play_go_initialized = true;
    state.services.play_go_open = true;
    state
}

#[test]
fn playgo_locus_rejects_more_than_the_sdk_max_without_writing() {
    let state = open_playgo_state();
    let chunk_ids = vec![0_u16; 1_001];
    let mut loci = vec![0xcc_u8; 1_003];

    assert_eq!(
        play_go_get_locus(
            [
                1,
                chunk_ids.as_ptr().addr() as u64,
                1_001,
                loci.as_mut_ptr().addr() as u64,
                0,
                0,
            ],
            &state,
        ),
        sce_kernel_error(0x80b2_000b)
    );
    assert!(loci.iter().all(|byte| *byte == 0xcc));
}

#[test]
fn playgo_locus_rejects_an_unknown_chunk_without_writing() {
    let state = open_playgo_state();
    let chunk_id = 1_000_u16;
    let mut locus = [0xcc_u8; 3];

    assert_eq!(
        play_go_get_locus(
            [
                1,
                (&raw const chunk_id).addr() as u64,
                1,
                locus[1..].as_mut_ptr().addr() as u64,
                0,
                0,
            ],
            &state,
        ),
        sce_kernel_error(0x80b2_000c)
    );
    assert_eq!(locus, [0xcc; 3]);
}

#[test]
fn playgo_locus_writes_one_byte_per_valid_chunk() {
    let state = open_playgo_state();
    let chunk_id = 0_u16;
    let mut locus = [0xcc_u8; 3];

    assert_eq!(
        play_go_get_locus(
            [
                1,
                (&raw const chunk_id).addr() as u64,
                1,
                locus[1..].as_mut_ptr().addr() as u64,
                0,
                0,
            ],
            &state,
        ),
        0
    );
    assert_eq!(locus, [0xcc, 3, 0xcc]);
}

#[test]
fn playgo_chunk_id_uses_the_sdk_uint16_element_width() {
    let state = open_playgo_state();
    let mut chunk_bytes = [0xcc_u8; 4];
    let mut count = 0_u32;

    assert_eq!(
        play_go_get_chunk_id(
            [
                1,
                chunk_bytes[1..].as_mut_ptr().addr() as u64,
                1,
                (&raw mut count).addr() as u64,
                0,
                0,
            ],
            &state,
        ),
        0
    );
    assert_eq!(chunk_bytes, [0xcc, 0, 0, 0xcc]);
    assert_eq!(count, 1);
}

#[test]
fn playgo_reports_the_sdk_full_install_speed_value() {
    let state = open_playgo_state();
    let mut speed = u32::MAX;

    assert_eq!(
        play_go_get_install_speed([1, (&raw mut speed).addr() as u64, 0, 0, 0, 0], &state,),
        0
    );
    assert_eq!(speed, 2);
}

#[test]
fn playgo_exposes_only_enabled_chunks_from_the_unpacked_package_manifest() {
    let temporary = super::TemporaryTestDirectory::new("playgo-chunks");
    std::fs::write(
        temporary.path.join("playgo-chunkdefs.xml"),
        r#"<bp_playgo_chunks><chunks>
            <chunk id="2" label="area" enabled="true" />
            <chunk id="1" label="placeholder" enabled="false" />
            <chunk id="0" label="launch" enabled="true" />
            <chunk id="2" label="duplicate" enabled="true" />
        </chunks><chunksets><chunkset name="all" chunks="0,1,2" /></chunksets></bp_playgo_chunks>"#,
    )
    .unwrap();
    let mut state = ProbeDispatchState {
        game_root: temporary.path.clone(),
        ..ProbeDispatchState::default()
    };
    let mut handle = 0_u32;

    assert_eq!(play_go_initialize(&mut state), 0);
    assert_eq!(
        play_go_open([(&raw mut handle).addr() as u64, 0, 0, 0, 0, 0], &mut state,),
        0
    );
    assert_eq!(handle, 1);

    let mut chunks = [u16::MAX; 4];
    let mut count = 0_u32;
    assert_eq!(
        play_go_get_chunk_id(
            [
                1,
                chunks.as_mut_ptr().addr() as u64,
                4,
                (&raw mut count).addr() as u64,
                0,
                0,
            ],
            &state,
        ),
        0
    );
    assert_eq!(count, 2);
    assert_eq!(chunks, [0, 2, u16::MAX, u16::MAX]);

    let disabled = 1_u16;
    let mut locus = 0xcc_u8;
    assert_eq!(
        play_go_get_locus(
            [
                1,
                (&raw const disabled).addr() as u64,
                1,
                (&raw mut locus).addr() as u64,
                0,
                0,
            ],
            &state,
        ),
        sce_kernel_error(0x80b2_000c)
    );
    assert_eq!(locus, 0xcc);
}
