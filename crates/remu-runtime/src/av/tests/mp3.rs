// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;

#[test]
fn a_frame_header_reports_the_geometry_the_decoder_needs() {
    let mut state = ProbeDispatchState::default();
    let frame = mp3_frame();
    let mut parsed = [0xff_u8; MP3_PARSE_FRAME_SIZE];
    assert_eq!(
        dispatch(
            "eDFeTyi+G3Y",
            args(&[
                frame.as_ptr() as u64,
                frame.len() as u64,
                0,
                address_mut(&mut parsed)
            ]),
            &mut state
        ),
        Some(SUCCESS)
    );
    let mut expected = [0_u8; MP3_PARSE_FRAME_SIZE];
    expected[..8].copy_from_slice(&417_u64.to_le_bytes());
    expected[8..12].copy_from_slice(&2_u32.to_le_bytes());
    expected[12..16].copy_from_slice(&1152_u32.to_le_bytes());
    expected[16..20].copy_from_slice(&128_000_u32.to_le_bytes());
    expected[20..24].copy_from_slice(&44_100_u32.to_le_bytes());
    assert_eq!(parsed, expected);
}

#[test]
fn an_original_file_length_header_is_read_only_when_it_is_asked_for() {
    let mut state = ProbeDispatchState::default();
    let mut frame = mp3_frame();
    frame[36..40].copy_from_slice(b"Xing");
    frame[40..44].copy_from_slice(&1_u32.to_be_bytes());
    frame[44..48].copy_from_slice(&1000_u32.to_be_bytes());
    frame[48..52].copy_from_slice(b"LAME");
    frame[69] = 0x02;
    frame[70] = 0x10;
    let mut parsed = [0_u8; MP3_PARSE_FRAME_SIZE];
    let output = address_mut(&mut parsed);
    assert_eq!(
        dispatch(
            "eDFeTyi+G3Y",
            args(&[frame.as_ptr() as u64, frame.len() as u64, 1, output]),
            &mut state
        ),
        Some(SUCCESS)
    );
    let field = |offset: usize| {
        u32::from_le_bytes(
            parsed[offset..offset + 4]
                .try_into()
                .expect("a field is four bytes"),
        )
    };
    assert_eq!(
        field(24),
        33,
        "the encoder delay comes from the encoder tag"
    );
    assert_eq!(field(28), 1000);
    assert_eq!(field(32), 1000 * 1152);
    assert_eq!(field(36), MP3_OFL_TYPE_LAME);

    assert_eq!(
        dispatch(
            "eDFeTyi+G3Y",
            args(&[frame.as_ptr() as u64, frame.len() as u64, 0, output]),
            &mut state
        ),
        Some(SUCCESS)
    );
    assert_eq!(parsed[24..40], [0; 16], "an unrequested header is not read");
}

#[test]
fn a_bitstream_carrying_no_frame_is_rejected_rather_than_guessed_at() {
    let mut state = ProbeDispatchState::default();
    let silence = [0_u8; 64];
    let mut parsed = [0_u8; MP3_PARSE_FRAME_SIZE];
    let output = address_mut(&mut parsed);
    for arguments in [
        args(&[address(&silence), silence.len() as u64, 0, output]),
        args(&[0, 64, 0, output]),
        args(&[address(&silence), 0, 0, output]),
        args(&[address(&silence), silence.len() as u64, 0, 0]),
    ] {
        assert_eq!(
            dispatch("eDFeTyi+G3Y", arguments, &mut state),
            Some(error(SCE_AJM_ERROR_INVALID_PARAMETER))
        );
    }
    assert_eq!(parsed, [0; MP3_PARSE_FRAME_SIZE]);
}
