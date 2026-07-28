// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;

#[test]
fn a_created_rack_hands_out_one_stable_handle_per_voice_it_was_sized_for() {
    let mut fixture = VoiceFixture::new(4, 2, 2);
    let mut again = 0_u64;
    assert_eq!(
        dispatch(
            "MwmHz8pAdAo",
            args(&[fixture.rack, 0, address_mut(&mut again)]),
            &mut fixture.state
        ),
        Some(SUCCESS)
    );
    assert_eq!(again, fixture.voice);
    let mut last = 0_u64;
    assert_eq!(
        dispatch(
            "MwmHz8pAdAo",
            args(&[fixture.rack, 3, address_mut(&mut last)]),
            &mut fixture.state
        ),
        Some(SUCCESS)
    );
    assert_ne!(last, fixture.voice);
    assert_eq!(
        dispatch(
            "MwmHz8pAdAo",
            args(&[fixture.rack, 4, address_mut(&mut last)]),
            &mut fixture.state
        ),
        Some(error(NGS2_ERROR_INVALID_VOICE_INDEX))
    );
    assert_eq!(
        dispatch(
            "MwmHz8pAdAo",
            args(&[fixture.rack, 0, 0]),
            &mut fixture.state
        ),
        Some(error(NGS2_ERROR_INVALID_OUT_ADDRESS))
    );
}

#[test]
fn destroying_a_rack_reports_its_allocator_data_and_retires_its_voices() {
    let mut state = ProbeDispatchState::default();
    let allocator = buffer_allocator(0xabcd);
    let option = system_option(48_000, 256);
    let mut system = 0_u64;
    assert_eq!(
        dispatch(
            "mPYgU4oYpuY",
            args(&[
                address(&option),
                address(&allocator),
                address_mut(&mut system)
            ]),
            &mut state
        ),
        Some(SUCCESS)
    );
    let option = rack_option(2, 1, 1);
    let mut rack = 0_u64;
    assert_eq!(
        dispatch(
            "U546k6orxQo",
            args(&[
                system,
                0x2000,
                address(&option),
                address(&allocator),
                address_mut(&mut rack)
            ]),
            &mut state
        ),
        Some(SUCCESS)
    );
    let mut voice = 0_u64;
    assert_eq!(
        dispatch(
            "MwmHz8pAdAo",
            args(&[rack, 1, address_mut(&mut voice)]),
            &mut state
        ),
        Some(SUCCESS)
    );
    let mut buffer_info = [0xff_u8; 64];
    assert_eq!(
        dispatch(
            "lCqD7oycmIM",
            args(&[rack, address_mut(&mut buffer_info)]),
            &mut state
        ),
        Some(SUCCESS)
    );
    assert_eq!(
        buffer_info[..56],
        [0; 56],
        "the buffer record must be cleared"
    );
    assert_eq!(
        u64::from_le_bytes(
            buffer_info[56..]
                .try_into()
                .expect("user data is eight bytes")
        ),
        0xabcd
    );
    let mut flags = 0_u32;
    assert_eq!(
        dispatch(
            "rEh728kXk3w",
            args(&[voice, address_mut(&mut flags)]),
            &mut state
        ),
        Some(error(NGS2_ERROR_INVALID_VOICE_HANDLE))
    );
    assert_eq!(
        dispatch("lCqD7oycmIM", args(&[rack, 0]), &mut state),
        Some(error(NGS2_ERROR_INVALID_RACK_HANDLE))
    );
    assert_eq!(
        dispatch("u-WrYDaJA3k", args(&[system, 0]), &mut state),
        Some(SUCCESS)
    );
    assert_eq!(
        dispatch("u-WrYDaJA3k", args(&[system, 0]), &mut state),
        Some(error(NGS2_ERROR_INVALID_SYSTEM_HANDLE))
    );
}

#[test]
fn creating_a_system_rejects_a_missing_allocator_and_an_option_it_cannot_honour() {
    let mut state = ProbeDispatchState::default();
    let allocator = buffer_allocator(0);
    let option = system_option(48_000, 256);
    let mut handle = 0_u64;
    assert_eq!(
        dispatch(
            "mPYgU4oYpuY",
            args(&[address(&option), address(&allocator), 0]),
            &mut state
        ),
        Some(error(NGS2_ERROR_INVALID_OUT_ADDRESS))
    );
    assert_eq!(
        dispatch(
            "mPYgU4oYpuY",
            args(&[address(&option), 0, address_mut(&mut handle)]),
            &mut state
        ),
        Some(error(NGS2_ERROR_INVALID_CONTEXT_ALLOCATOR))
    );
    let no_handlers = buffer_allocator(0).map(|_| 0);
    assert_eq!(
        dispatch(
            "mPYgU4oYpuY",
            args(&[
                address(&option),
                address(&no_handlers),
                address_mut(&mut handle)
            ]),
            &mut state
        ),
        Some(error(NGS2_ERROR_INVALID_CONTEXT_ALLOCATOR))
    );
    let unusable_rate = system_option(1_000, 256);
    assert_eq!(
        dispatch(
            "mPYgU4oYpuY",
            args(&[
                address(&unusable_rate),
                address(&allocator),
                address_mut(&mut handle)
            ]),
            &mut state
        ),
        Some(error(NGS2_ERROR_INVALID_SAMPLE_RATE))
    );
    let unusable_grain = system_option(48_000, 100);
    assert_eq!(
        dispatch(
            "mPYgU4oYpuY",
            args(&[
                address(&unusable_grain),
                address(&allocator),
                address_mut(&mut handle)
            ]),
            &mut state
        ),
        Some(error(NGS2_ERROR_INVALID_NUM_GRAIN_SAMPLES))
    );
    let mut short_option = system_option(48_000, 256);
    short_option[..8].copy_from_slice(&8_u64.to_le_bytes());
    assert_eq!(
        dispatch(
            "mPYgU4oYpuY",
            args(&[
                address(&short_option),
                address(&allocator),
                address_mut(&mut handle)
            ]),
            &mut state
        ),
        Some(error(NGS2_ERROR_INVALID_OPTION_SIZE))
    );
}

#[test]
fn creating_a_rack_rejects_an_unknown_identifier_system_or_voice_count() {
    let mut fixture = VoiceFixture::new(1, 1, 1);
    let allocator = buffer_allocator(0);
    let option = rack_option(1, 1, 1);
    let mut rack = 0_u64;
    assert_eq!(
        dispatch(
            "U546k6orxQo",
            args(&[
                fixture.system,
                0x1234,
                address(&option),
                address(&allocator),
                address_mut(&mut rack)
            ]),
            &mut fixture.state
        ),
        Some(error(NGS2_ERROR_INVALID_RACK_ID))
    );
    assert_eq!(
        dispatch(
            "U546k6orxQo",
            args(&[
                0,
                0x1000,
                address(&option),
                address(&allocator),
                address_mut(&mut rack)
            ]),
            &mut fixture.state
        ),
        Some(error(NGS2_ERROR_INVALID_SYSTEM_HANDLE))
    );
    let no_voices = rack_option(0, 1, 1);
    assert_eq!(
        dispatch(
            "U546k6orxQo",
            args(&[
                fixture.system,
                0x1000,
                address(&no_voices),
                address(&allocator),
                address_mut(&mut rack)
            ]),
            &mut fixture.state
        ),
        Some(error(NGS2_ERROR_INVALID_NUM_VOICES))
    );
}

#[test]
fn the_system_option_defaults_to_zero_with_its_own_size() {
    let mut state = ProbeDispatchState::default();
    let mut option = [0xff_u8; 0x90];
    assert_eq!(
        dispatch("AQkj7C0f3PY", args(&[address_mut(&mut option)]), &mut state),
        Some(SUCCESS)
    );
    assert_eq!(
        u64::from_le_bytes(option[..8].try_into().expect("the size is eight bytes")),
        NGS2_SYSTEM_OPTION_SIZE
    );
    assert_eq!(option[8..], [0; 0x88]);
    assert_eq!(
        dispatch("AQkj7C0f3PY", args(&[0]), &mut state),
        Some(error(NGS2_ERROR_INVALID_OUT_ADDRESS))
    );
}

#[test]
fn a_control_chain_kicks_every_entry_it_links_and_leaves_the_voice_playing() {
    let mut fixture = VoiceFixture::new(2, 2, 2);
    let mut chain = [0_u8; 28];
    control_entry(&mut chain, 0, 12, 12, NGS2_VOICE_KICK_EVENT);
    chain[8..12].copy_from_slice(&NGS2_VOICE_EVENT_PLAY.to_le_bytes());
    control_entry(&mut chain, 12, 16, 0, NGS2_VOICE_SET_PORT_VOLUME);
    chain[24..28].copy_from_slice(&1.0_f32.to_bits().to_le_bytes());
    let voice = fixture.voice;
    assert_eq!(
        dispatch(
            "uu94irFOGpA",
            args(&[voice, address(&chain)]),
            &mut fixture.state
        ),
        Some(SUCCESS)
    );
    assert_eq!(
        fixture.state_flags(),
        NGS2_VOICE_STATE_FLAG_IN_USE | NGS2_VOICE_STATE_FLAG_PLAYING
    );

    let mut stop = [0_u8; 12];
    control_entry(&mut stop, 0, 12, 0, NGS2_VOICE_KICK_EVENT);
    stop[8..12].copy_from_slice(&NGS2_VOICE_EVENT_STOP.to_le_bytes());
    assert_eq!(
        dispatch(
            "uu94irFOGpA",
            args(&[voice, address(&stop)]),
            &mut fixture.state
        ),
        Some(SUCCESS)
    );
    assert_eq!(
        fixture.state_flags(),
        NGS2_VOICE_STATE_FLAG_IN_USE | NGS2_VOICE_STATE_FLAG_STOPPED
    );
}

#[test]
fn a_malformed_control_chain_is_rejected_rather_than_walked() {
    let mut fixture = VoiceFixture::new(2, 2, 2);
    let voice = fixture.voice;
    assert_eq!(
        dispatch("uu94irFOGpA", args(&[voice, 0]), &mut fixture.state),
        Some(error(NGS2_ERROR_INVALID_VOICE_CONTROL_ADDRESS))
    );

    let mut unknown = [0_u8; 12];
    control_entry(&mut unknown, 0, 12, 0, 0x99);
    assert_eq!(
        dispatch(
            "uu94irFOGpA",
            args(&[voice, address(&unknown)]),
            &mut fixture.state
        ),
        Some(error(NGS2_ERROR_INVALID_VOICE_CONTROL_ID))
    );

    let mut truncated = [0_u8; 12];
    control_entry(&mut truncated, 0, 8, 0, NGS2_VOICE_KICK_EVENT);
    assert_eq!(
        dispatch(
            "uu94irFOGpA",
            args(&[voice, address(&truncated)]),
            &mut fixture.state
        ),
        Some(error(NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE))
    );

    let mut unknown_port = [0_u8; 16];
    control_entry(&mut unknown_port, 0, 16, 0, NGS2_VOICE_SET_PORT_VOLUME);
    unknown_port[8..12].copy_from_slice(&5_u32.to_le_bytes());
    assert_eq!(
        dispatch(
            "uu94irFOGpA",
            args(&[voice, address(&unknown_port)]),
            &mut fixture.state
        ),
        Some(error(NGS2_ERROR_INVALID_VOICE_OUTPUT_INDEX))
    );

    let mut unknown_event = [0_u8; 12];
    control_entry(&mut unknown_event, 0, 12, 0, NGS2_VOICE_KICK_EVENT);
    unknown_event[8..12].copy_from_slice(&0x40_u32.to_le_bytes());
    assert_eq!(
        dispatch(
            "uu94irFOGpA",
            args(&[voice, address(&unknown_event)]),
            &mut fixture.state
        ),
        Some(error(NGS2_ERROR_INVALID_VOICE_EVENT_TYPE))
    );
    assert_eq!(fixture.state_flags(), 0);
}

#[test]
fn a_control_chain_that_links_back_to_itself_is_refused_rather_than_walked_forever() {
    let mut fixture = VoiceFixture::new(2, 2, 2);
    let mut chain = [0_u8; 24];
    control_entry(&mut chain, 0, 12, 12, NGS2_VOICE_KICK_EVENT);
    chain[8..12].copy_from_slice(&NGS2_VOICE_EVENT_PLAY.to_le_bytes());
    control_entry(&mut chain, 12, 12, -12, NGS2_VOICE_KICK_EVENT);
    chain[20..24].copy_from_slice(&NGS2_VOICE_EVENT_PLAY.to_le_bytes());
    let voice = fixture.voice;
    assert_eq!(
        dispatch(
            "uu94irFOGpA",
            args(&[voice, address(&chain)]),
            &mut fixture.state
        ),
        Some(error(NGS2_ERROR_INVALID_VOICE_CONTROL_ADDRESS))
    );
}

#[test]
fn a_command_array_sets_a_voice_up_and_kicks_it_in_order() {
    let mut fixture = VoiceFixture::new(2, 2, 2);
    let setup = [2_u32, 3];
    let mut commands = [0_u8; 32];
    command_entry(&mut commands, 0, NGS2_VOICE_COMMAND_SETUP, address(&setup));
    command_entry(
        &mut commands,
        1,
        NGS2_VOICE_COMMAND_KICK_EVENT,
        u64::from(NGS2_VOICE_EVENT_PLAY),
    );
    let voice = fixture.voice;
    assert_eq!(
        dispatch(
            "AbYvTOZ8Pts",
            args(&[voice, address(&commands), 2]),
            &mut fixture.state
        ),
        Some(SUCCESS)
    );
    assert_eq!(
        fixture.state_flags(),
        NGS2_VOICE_STATE_FLAG_IN_USE | NGS2_VOICE_STATE_FLAG_PLAYING
    );

    let mut state = [0_u8; 8];
    assert_eq!(
        dispatch(
            "-TOuuAQ-buE",
            args(&[voice, address_mut(&mut state), 8]),
            &mut fixture.state
        ),
        Some(SUCCESS)
    );
    let mut expected = [0_u8; 8];
    expected[..4].copy_from_slice(
        &(NGS2_VOICE_STATE_FLAG_IN_USE | NGS2_VOICE_STATE_FLAG_PLAYING).to_le_bytes(),
    );
    assert_eq!(
        state, expected,
        "the flags lead the state and no error follows"
    );
    assert_eq!(
        dispatch(
            "-TOuuAQ-buE",
            args(&[voice, address_mut(&mut state), 4]),
            &mut fixture.state
        ),
        Some(error(NGS2_ERROR_INVALID_STATE_SIZE))
    );
    assert_eq!(
        dispatch("-TOuuAQ-buE", args(&[voice, 0, 8]), &mut fixture.state),
        Some(error(NGS2_ERROR_INVALID_OUT_ADDRESS))
    );
}

#[test]
fn a_command_array_refuses_a_parameter_that_does_not_address_this_voice() {
    let mut fixture = VoiceFixture::new(2, 2, 2);
    let voice = fixture.voice;
    let mut commands = [0_u8; 16];
    command_entry(&mut commands, 0, 0x00f0_0000, 0);
    assert_eq!(
        dispatch(
            "AbYvTOZ8Pts",
            args(&[voice, address(&commands), 1]),
            &mut fixture.state
        ),
        Some(error(NGS2_ERROR_INVALID_VOICE_PARAM_ID))
    );
    command_entry(&mut commands, 0, 0x4000, 0);
    assert_eq!(
        dispatch(
            "AbYvTOZ8Pts",
            args(&[voice, address(&commands), 1]),
            &mut fixture.state
        ),
        Some(error(NGS2_ERROR_INVALID_VOICE_PARAM_ID))
    );
    command_entry(&mut commands, 0, 0x000e, 0);
    assert_eq!(
        dispatch(
            "AbYvTOZ8Pts",
            args(&[voice, address(&commands), 1]),
            &mut fixture.state
        ),
        Some(error(NGS2_ERROR_INVALID_VOICE_PARAM_ID))
    );
    // A module component carries a parameter for a module the rack would own.
    command_entry(&mut commands, 0, 0x0001_0002, 0);
    assert_eq!(
        dispatch(
            "AbYvTOZ8Pts",
            args(&[voice, address(&commands), 1]),
            &mut fixture.state
        ),
        Some(SUCCESS)
    );
    assert_eq!(
        dispatch("AbYvTOZ8Pts", args(&[voice, 0, 1]), &mut fixture.state),
        Some(error(NGS2_ERROR_INVALID_PARAM_ADDRESS))
    );
    let too_many = [0_u8; 16];
    assert_eq!(
        dispatch(
            "AbYvTOZ8Pts",
            args(&[voice, address(&too_many), NGS2_COMMAND_PARAM_LIMIT + 1]),
            &mut fixture.state
        ),
        Some(error(NGS2_ERROR_INVALID_PARAM_ADDRESS))
    );
}

#[test]
fn a_voice_setup_command_bounds_the_channel_count_it_accepts() {
    let mut fixture = VoiceFixture::new(2, 2, 2);
    let voice = fixture.voice;
    let setup = [64_u32, 3];
    let mut commands = [0_u8; 16];
    command_entry(&mut commands, 0, NGS2_VOICE_COMMAND_SETUP, address(&setup));
    assert_eq!(
        dispatch(
            "AbYvTOZ8Pts",
            args(&[voice, address(&commands), 1]),
            &mut fixture.state
        ),
        Some(error(NGS2_ERROR_INVALID_NUM_CHANNELS))
    );
    let setup = [2_u32, 9];
    command_entry(&mut commands, 0, NGS2_VOICE_COMMAND_SETUP, address(&setup));
    assert_eq!(
        dispatch(
            "AbYvTOZ8Pts",
            args(&[voice, address(&commands), 1]),
            &mut fixture.state
        ),
        Some(error(NGS2_ERROR_INVALID_VOICE_SETUP_FLAG))
    );
    command_entry(&mut commands, 0, NGS2_VOICE_COMMAND_SETUP, 0);
    assert_eq!(
        dispatch(
            "AbYvTOZ8Pts",
            args(&[voice, address(&commands), 1]),
            &mut fixture.state
        ),
        Some(error(NGS2_ERROR_INVALID_PARAM_ADDRESS))
    );
}
