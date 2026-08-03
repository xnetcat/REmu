// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;

#[test]
fn a_synchronized_output_bounds_its_port_array_before_it_touches_a_port() {
    let mut entries = [0_u8; 32];
    entries[..4].copy_from_slice(&0x7fff_u32.to_le_bytes());
    entries[16..20].copy_from_slice(&0x7ffe_u32.to_le_bytes());
    assert_eq!(
        audio_out_outputs(args(&[0, 1])),
        audio_out_error(audio::AudioOutError::InvalidPointer)
    );
    assert_eq!(
        audio_out_outputs(args(&[address(&entries), 0])),
        error(AUDIO_OUT_ERROR_INVALID_ARG)
    );
    assert_eq!(
        audio_out_outputs(args(&[address(&entries), AUDIO_OUT_OUTPUTS_MAX + 1])),
        error(AUDIO_OUT_ERROR_INVALID_ARG)
    );
    assert_eq!(
        audio_out_outputs(args(&[address(&entries), 2])),
        audio_out_error(audio::AudioOutError::InvalidPort)
    );
}

#[test]
fn a_pad_speaker_mix_level_outside_its_range_is_rejected_before_the_port_is_looked_up() {
    let mut state = ProbeDispatchState::default();
    for level in [
        u64::MAX,
        u64::try_from(AUDIO_OUT_MIXLEVEL_MAXIMUM + 1).unwrap_or(0),
    ] {
        assert_eq!(
            dispatch("wVwPU50pS1c", args(&[0x7fff, level]), &mut state),
            Some(error(AUDIO_OUT_ERROR_INVALID_MIXLEVEL))
        );
    }
    assert_eq!(
        dispatch(
            "wVwPU50pS1c",
            args(&[
                0x7fff,
                u64::try_from(AUDIO_OUT_MIXLEVEL_MAXIMUM).unwrap_or(0)
            ]),
            &mut state
        ),
        Some(audio_out_error(audio::AudioOutError::InvalidPort))
    );
}

#[test]
fn a_nid_no_module_here_owns_is_left_for_the_next_dispatcher() {
    let mut state = open_display();
    assert_eq!(dispatch("AAAAAAAAAAA", args(&[0]), &mut state), None);
}
