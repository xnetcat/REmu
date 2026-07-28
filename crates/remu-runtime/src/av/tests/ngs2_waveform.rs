// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;

#[test]
fn a_sample_range_becomes_the_block_that_addresses_it() {
    let mut state = ProbeDispatchState::default();
    let format = waveform_format(NGS2_WAVEFORM_TYPE_PCM_I16L, 2);
    let mut block = [0xff_u8; NGS2_WAVEFORM_BLOCK_SIZE];
    assert_eq!(
        dispatch(
            "3pCNbVM11UA",
            args(&[address(&format), 100, 50, address_mut(&mut block)]),
            &mut state
        ),
        Some(SUCCESS)
    );
    let mut expected = [0_u8; NGS2_WAVEFORM_BLOCK_SIZE];
    expected[..8].copy_from_slice(&400_u64.to_le_bytes());
    expected[8..16].copy_from_slice(&200_u64.to_le_bytes());
    expected[16..20].copy_from_slice(&1_u32.to_le_bytes());
    expected[24..28].copy_from_slice(&50_u32.to_le_bytes());
    assert_eq!(block, expected);
}

#[test]
fn a_sample_range_inside_a_compressed_unit_reports_the_samples_to_skip() {
    let mut state = ProbeDispatchState::default();
    let format = waveform_format(NGS2_WAVEFORM_TYPE_VAG, 1);
    let mut block = [0_u8; NGS2_WAVEFORM_BLOCK_SIZE];
    assert_eq!(
        dispatch(
            "3pCNbVM11UA",
            args(&[address(&format), 30, 28, address_mut(&mut block)]),
            &mut state
        ),
        Some(SUCCESS)
    );
    assert_eq!(
        u64::from_le_bytes(block[..8].try_into().expect("the offset is eight bytes")),
        u64::from(NGS2_VAG_UNIT_BYTES)
    );
    assert_eq!(
        u64::from_le_bytes(block[8..16].try_into().expect("the size is eight bytes")),
        u64::from(NGS2_VAG_UNIT_BYTES) * 2
    );
    assert_eq!(
        u32::from_le_bytes(block[20..24].try_into().expect("the skip is four bytes")),
        2
    );
}

#[test]
fn a_waveform_this_build_cannot_address_by_sample_is_refused() {
    let mut state = ProbeDispatchState::default();
    let codec = waveform_format(0x40, 2);
    let no_channels = waveform_format(NGS2_WAVEFORM_TYPE_PCM_I16L, 0);
    let mut block = [0_u8; NGS2_WAVEFORM_BLOCK_SIZE];
    let output = address_mut(&mut block);
    assert_eq!(
        dispatch(
            "3pCNbVM11UA",
            args(&[address(&codec), 0, 1, output]),
            &mut state
        ),
        Some(error(NGS2_ERROR_INVALID_WAVEFORM_TYPE))
    );
    assert_eq!(
        dispatch(
            "3pCNbVM11UA",
            args(&[address(&no_channels), 0, 1, output]),
            &mut state
        ),
        Some(error(NGS2_ERROR_INVALID_NUM_CHANNELS))
    );
    assert_eq!(
        dispatch("3pCNbVM11UA", args(&[address(&codec), 0, 1, 0]), &mut state),
        Some(error(NGS2_ERROR_INVALID_OUT_ADDRESS))
    );
    assert_eq!(
        dispatch("3pCNbVM11UA", args(&[0, 0, 1, output]), &mut state),
        Some(error(NGS2_ERROR_INVALID_WAVEFORM_ADDRESS))
    );
}

#[test]
fn a_recognized_container_reports_its_format_data_and_first_block() {
    let mut state = ProbeDispatchState::default();
    let container = riff_container(2);
    let mut info = [0xff_u8; NGS2_WAVEFORM_INFO_SIZE];
    assert_eq!(
        dispatch(
            "hyVLT2VlOYk",
            args(&[
                container.as_ptr() as u64,
                container.len() as u64,
                address_mut(&mut info)
            ]),
            &mut state
        ),
        Some(SUCCESS)
    );
    let field = |offset: usize| {
        u32::from_le_bytes(
            info[offset..offset + 4]
                .try_into()
                .expect("a field is four bytes"),
        )
    };
    assert_eq!(field(0), NGS2_WAVEFORM_TYPE_PCM_I16L);
    assert_eq!(field(4), 2);
    assert_eq!(field(8), 48_000);
    assert_eq!(info[12..24], [0; 12], "no codec configuration is reported");
    assert_eq!(field(24), 44);
    assert_eq!(field(28), 8);
    assert_eq!(info[32..40], [0; 8], "an unlooped waveform reports no loop");
    assert_eq!(field(40), 2);
    assert_eq!(field(44), 4);
    assert_eq!(field(48), 1);
    assert_eq!(field(52), 1);
    assert_eq!(field(56), 4);
    assert_eq!(field(60), 1);
    assert_eq!(field(64), 0);
    assert_eq!(field(68), 1);
    let mut block = [0_u8; NGS2_WAVEFORM_BLOCK_SIZE];
    block[8..16].copy_from_slice(&8_u64.to_le_bytes());
    block[16..20].copy_from_slice(&1_u32.to_le_bytes());
    block[24..28].copy_from_slice(&2_u32.to_le_bytes());
    assert_eq!(info[72..112], block);
    assert_eq!(info[112..], [0; 120], "only one block is described");
}

#[test]
fn a_container_this_build_does_not_recognize_is_reported_as_unknown() {
    let mut state = ProbeDispatchState::default();
    let mut info = [0_u8; NGS2_WAVEFORM_INFO_SIZE];
    let output = address_mut(&mut info);
    let unknown = *b"OggS____________";
    assert_eq!(
        dispatch(
            "hyVLT2VlOYk",
            args(&[address(&unknown), unknown.len() as u64, output]),
            &mut state
        ),
        Some(error(NGS2_ERROR_UNKNOWN_WAVEFORM_FORMAT))
    );
    let container = riff_container(2);
    assert_eq!(
        dispatch(
            "hyVLT2VlOYk",
            args(&[container.as_ptr() as u64, container.len() as u64, 0]),
            &mut state
        ),
        Some(error(NGS2_ERROR_INVALID_OUT_ADDRESS))
    );
    assert_eq!(
        dispatch("hyVLT2VlOYk", args(&[0, 64, output]), &mut state),
        Some(error(NGS2_ERROR_INVALID_WAVEFORM_DATA))
    );
    assert_eq!(
        dispatch(
            "hyVLT2VlOYk",
            args(&[container.as_ptr() as u64, 4, output]),
            &mut state
        ),
        Some(error(NGS2_ERROR_INVALID_WAVEFORM_DATA))
    );
}
