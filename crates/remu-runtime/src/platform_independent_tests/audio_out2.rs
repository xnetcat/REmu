// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::time::Instant;

use super::{
    AUDIO_OUT2_CONTEXT_MEMORY_BASE_SIZE, AUDIO_OUT2_CONTEXT_MEMORY_QUEUE_STRIDE,
    AUDIO_OUT2_CONTEXT_PARAM_SIZE, AUDIO_OUT2_DEFAULT_GRAIN_FRAMES, AUDIO_OUT2_DEFAULT_QUEUE_DEPTH,
    AUDIO_OUT2_PORT_TYPE_VIBRATION, AUDIO_OUT2_TEST, SCE_AUDIO_OUT2_ERROR_NOT_READY,
    SceAudioOut2Attribute, SceAudioOut2ContextParam, SceAudioOut2Pcm, SceAudioOut2PortParam,
    SceAudioOut2PortState, create_audio_out2_test_context, create_audio_out2_test_port,
    decode_audio_out2_attribute, decode_audio_out2_channel_format, dispatch_audio_out2_import,
    reset_audio_out2_test_state, sce_kernel_error,
};

#[test]
fn audio_out2_mastering_initialization_is_an_accepted_mixer_stage() {
    assert_eq!(
        dispatch_audio_out2_import("XHl38ZNknbs", [0, 0x1000, 0x2000, 0, 0x3000, 0x4000]),
        0
    );
    assert_eq!(
        dispatch_audio_out2_import("v8iOE+j8a5o", [0x5000, 0, 0, 0x6000, 2, 0xa]),
        0
    );
}

#[test]
fn audio_out2_channel_formats_map_to_matching_host_layouts() {
    assert_eq!(decode_audio_out2_channel_format(0x100), Some((1, 3)));
    assert_eq!(decode_audio_out2_channel_format(0x200), Some((2, 4)));
    assert_eq!(decode_audio_out2_channel_format(0x880), Some((8, 5)));
    assert_eq!(decode_audio_out2_channel_format(0), None);
    assert_eq!(decode_audio_out2_channel_format(0x480), None);
}

#[test]
fn audio_out2_attribute_id_does_not_include_its_reserved_word() {
    let attribute = SceAudioOut2Attribute {
        attribute_id: 0,
        reserved: 0x80,
        value: 0x1122_3344_5566_7788,
        value_size: 8,
    };
    let decoded = decode_audio_out2_attribute(std::ptr::addr_of!(attribute) as u64);
    assert_eq!(decoded.attribute_id, 0);
    assert_eq!(decoded.reserved, 0x80);
    assert_eq!(decoded.value, 0x1122_3344_5566_7788);
    assert_eq!(decoded.value_size, 8);
}

#[test]
fn audio_out2_abi_layouts_match_silent_hill_objects() {
    assert_eq!(std::mem::size_of::<SceAudioOut2ContextParam>(), 0x40);
    assert_eq!(std::mem::size_of::<SceAudioOut2PortParam>(), 0x40);
    assert_eq!(std::mem::size_of::<SceAudioOut2Attribute>(), 0x18);
    assert_eq!(std::mem::size_of::<SceAudioOut2Pcm>(), 0x08);
    assert_eq!(std::mem::size_of::<SceAudioOut2PortState>(), 0x40);
    assert_eq!(
        std::mem::offset_of!(SceAudioOut2PortParam, user_handle),
        0x10
    );
    assert_eq!(
        std::mem::offset_of!(SceAudioOut2Attribute, value_size),
        0x10
    );
    assert_eq!(std::mem::offset_of!(SceAudioOut2PortState, reserved), 0x10);
}

#[test]
fn audio_out2_gain_only_attribute_does_not_fabricate_pcm() {
    let _serial = AUDIO_OUT2_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_audio_out2_test_state();
    let context = create_audio_out2_test_context(AUDIO_OUT2_DEFAULT_GRAIN_FRAMES);
    let port = create_audio_out2_test_port(context, 0, 0x880);
    let gains = [0.125_f32, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0];
    let attribute = SceAudioOut2Attribute {
        attribute_id: 1,
        value: gains.as_ptr() as u64,
        value_size: std::mem::size_of_val(&gains) as u64,
        ..SceAudioOut2Attribute::default()
    };

    assert_eq!(
        dispatch_audio_out2_import(
            "8XTArSPyWHk",
            [port, std::ptr::addr_of!(attribute) as u64, 1, 0, 0, 0,],
        ),
        0
    );
    assert_eq!(
        dispatch_audio_out2_import("aII9h5nli9U", [context, 0, 0, 0, 0, 0]),
        0
    );

    let state = crate::AUDIO_OUT2
        .get()
        .expect("the AudioOut2 test state is initialized")
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(state.ports.get(&port).unwrap().gains, gains);
    assert_eq!(state.ports.get(&port).unwrap().buffer_address, 0);
    assert!(state.submitted_pcm.is_empty());
}

#[test]
fn audio_out2_pcm_attribute_snapshots_each_512_by_8_float_grain() {
    let _serial = AUDIO_OUT2_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_audio_out2_test_state();
    let context = create_audio_out2_test_context(512);
    let port = create_audio_out2_test_port(context, 0, 0x880);
    let first = (0..512 * 8)
        .map(|sample| (sample as f32 - 2048.0) / 4096.0)
        .collect::<Vec<_>>();
    let second = (0..512 * 8)
        .map(|sample| 0.75 - sample as f32 / 8192.0)
        .collect::<Vec<_>>();
    let pcm = SceAudioOut2Pcm {
        data: first.as_ptr() as u64,
    };
    let attribute = SceAudioOut2Attribute {
        attribute_id: 0,
        value: std::ptr::addr_of!(pcm) as u64,
        value_size: std::mem::size_of::<SceAudioOut2Pcm>() as u64,
        ..SceAudioOut2Attribute::default()
    };

    assert_eq!(
        dispatch_audio_out2_import(
            "8XTArSPyWHk",
            [port, std::ptr::addr_of!(attribute) as u64, 1, 0, 0, 0,],
        ),
        0
    );
    assert_eq!(
        dispatch_audio_out2_import("aII9h5nli9U", [context, 0, 0, 0, 0, 0]),
        0
    );

    let second_pcm = SceAudioOut2Pcm {
        data: second.as_ptr() as u64,
    };
    let second_attribute = SceAudioOut2Attribute {
        attribute_id: 0,
        value: std::ptr::addr_of!(second_pcm) as u64,
        value_size: std::mem::size_of::<SceAudioOut2Pcm>() as u64,
        ..SceAudioOut2Attribute::default()
    };
    assert_eq!(
        dispatch_audio_out2_import(
            "8XTArSPyWHk",
            [
                port,
                std::ptr::addr_of!(second_attribute) as u64,
                1,
                0,
                0,
                0,
            ],
        ),
        0
    );
    assert_eq!(
        dispatch_audio_out2_import("aII9h5nli9U", [context, 0, 0, 0, 0, 0]),
        0
    );

    let expected_first = first
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let expected_second = second
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let state = crate::AUDIO_OUT2
        .get()
        .expect("the AudioOut2 test state is initialized")
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(state.submitted_pcm.len(), 2);
    assert_eq!(state.submitted_pcm[0], (port, expected_first));
    assert_eq!(state.submitted_pcm[1], (port, expected_second));
}

#[test]
fn audio_out2_vibration_pcm_never_enters_the_speaker_backend() {
    let _serial = AUDIO_OUT2_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_audio_out2_test_state();
    let context = create_audio_out2_test_context(512);
    let port = create_audio_out2_test_port(context, AUDIO_OUT2_PORT_TYPE_VIBRATION, 0x200);
    let samples = vec![0.5_f32; 512 * 2];
    let pcm = SceAudioOut2Pcm {
        data: samples.as_ptr() as u64,
    };
    let attribute = SceAudioOut2Attribute {
        attribute_id: 0,
        value: std::ptr::addr_of!(pcm) as u64,
        value_size: std::mem::size_of::<SceAudioOut2Pcm>() as u64,
        ..SceAudioOut2Attribute::default()
    };
    assert_eq!(
        dispatch_audio_out2_import(
            "8XTArSPyWHk",
            [port, std::ptr::addr_of!(attribute) as u64, 1, 0, 0, 0,],
        ),
        0
    );
    assert_eq!(
        dispatch_audio_out2_import("aII9h5nli9U", [context, 0, 0, 0, 0, 0]),
        0
    );

    let state = crate::AUDIO_OUT2
        .get()
        .expect("the AudioOut2 test state is initialized")
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert!(state.ports.get(&port).unwrap().backend_handle.is_none());
    assert!(state.submitted_pcm.is_empty());
}

#[test]
fn audio_out2_queue_level_accepts_each_optional_output_independently() {
    let _serial = AUDIO_OUT2_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_audio_out2_test_state();
    let context = create_audio_out2_test_context(AUDIO_OUT2_DEFAULT_GRAIN_FRAMES);
    let mut queued = u32::MAX;
    let mut available = u32::MAX;
    assert_eq!(
        dispatch_audio_out2_import(
            "R7d0F1g2qsU",
            [
                context,
                std::ptr::from_mut(&mut queued) as u64,
                std::ptr::from_mut(&mut available) as u64,
                0,
                0,
                0,
            ],
        ),
        0
    );
    assert_eq!(queued, 0);
    assert_eq!(available, AUDIO_OUT2_DEFAULT_QUEUE_DEPTH);

    queued = u32::MAX;
    assert_eq!(
        dispatch_audio_out2_import(
            "R7d0F1g2qsU",
            [context, std::ptr::from_mut(&mut queued) as u64, 0, 0, 0, 0],
        ),
        0
    );
    assert_eq!(queued, 0);

    available = u32::MAX;
    assert_eq!(
        dispatch_audio_out2_import(
            "R7d0F1g2qsU",
            [
                context,
                0,
                std::ptr::from_mut(&mut available) as u64,
                0,
                0,
                0,
            ],
        ),
        0
    );
    assert_eq!(available, AUDIO_OUT2_DEFAULT_QUEUE_DEPTH);
    assert_eq!(
        dispatch_audio_out2_import("R7d0F1g2qsU", [context, 0, 0, 0, 0, 0]),
        0
    );
}

#[test]
fn audio_out2_context_reset_writes_the_complete_hardware_defaults() {
    let mut storage = [0xa5_u8; 0x60];
    let parameter = &mut storage[0x10..0x50];
    assert_eq!(
        dispatch_audio_out2_import(
            "t5YrizufpQc",
            [parameter.as_mut_ptr() as u64, 0, 0, 0, 0, 0],
        ),
        0
    );
    assert_eq!(u32::from_le_bytes(parameter[0..4].try_into().unwrap()), 256);
    assert_eq!(u32::from_le_bytes(parameter[4..8].try_into().unwrap()), 256);
    assert_eq!(u32::from_le_bytes(parameter[8..12].try_into().unwrap()), 0);
    assert_eq!(
        u32::from_le_bytes(parameter[12..16].try_into().unwrap()),
        AUDIO_OUT2_DEFAULT_QUEUE_DEPTH
    );
    assert_eq!(
        u32::from_le_bytes(parameter[16..20].try_into().unwrap()),
        AUDIO_OUT2_DEFAULT_GRAIN_FRAMES
    );
    assert_eq!(u32::from_le_bytes(parameter[20..24].try_into().unwrap()), 1);
    assert!(parameter[24..].iter().all(|byte| *byte == 0));
    assert_eq!(&storage[..0x10], &[0xa5; 0x10]);
    assert_eq!(&storage[0x50..], &[0xa5; 0x10]);
}

#[test]
fn audio_out2_context_memory_query_writes_only_the_size_qword() {
    let parameter = [0_u8; AUDIO_OUT2_CONTEXT_PARAM_SIZE as usize];
    let mut storage = [0xa5_u8; 0x28];
    let memory_size = &mut storage[0x10..0x18];
    assert_eq!(
        dispatch_audio_out2_import(
            "pDmme7Bgm6E",
            [
                parameter.as_ptr() as u64,
                memory_size.as_mut_ptr() as u64,
                0,
                0,
                0,
                0,
            ],
        ),
        0
    );
    assert_eq!(
        u64::from_le_bytes(memory_size.try_into().unwrap()),
        AUDIO_OUT2_CONTEXT_MEMORY_BASE_SIZE
            + u64::from(AUDIO_OUT2_DEFAULT_QUEUE_DEPTH) * AUDIO_OUT2_CONTEXT_MEMORY_QUEUE_STRIDE
    );
    assert_eq!(&storage[..0x10], &[0xa5; 0x10]);
    assert_eq!(&storage[0x18..], &[0xa5; 0x10]);
}

#[test]
fn audio_out2_context_memory_query_scales_with_queue_depth() {
    let mut parameter = [0_u8; AUDIO_OUT2_CONTEXT_PARAM_SIZE as usize];
    parameter[12..16].copy_from_slice(&8_u32.to_le_bytes());
    let mut memory_size = 0_u64;
    assert_eq!(
        dispatch_audio_out2_import(
            "pDmme7Bgm6E",
            [
                parameter.as_ptr() as u64,
                std::ptr::from_mut(&mut memory_size) as u64,
                0,
                0,
                0,
                0,
            ],
        ),
        0
    );
    assert_eq!(
        memory_size,
        AUDIO_OUT2_CONTEXT_MEMORY_BASE_SIZE + 8 * AUDIO_OUT2_CONTEXT_MEMORY_QUEUE_STRIDE
    );
}

#[test]
fn audio_out2_context_push_reserves_and_drains_hardware_queue_slots() {
    let _serial = AUDIO_OUT2_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_audio_out2_test_state();
    let context = create_audio_out2_test_context(8192);
    for _ in 0..AUDIO_OUT2_DEFAULT_QUEUE_DEPTH {
        assert_eq!(
            dispatch_audio_out2_import("aII9h5nli9U", [context, 0, 0, 0, 0, 0]),
            0
        );
    }
    let mut queued = u32::MAX;
    let mut available = u32::MAX;
    assert_eq!(
        dispatch_audio_out2_import(
            "R7d0F1g2qsU",
            [
                context,
                std::ptr::from_mut(&mut queued) as u64,
                std::ptr::from_mut(&mut available) as u64,
                0,
                0,
                0,
            ],
        ),
        0
    );
    assert_eq!(queued, AUDIO_OUT2_DEFAULT_QUEUE_DEPTH);
    assert_eq!(available, 0);
    assert_eq!(
        dispatch_audio_out2_import("aII9h5nli9U", [context, 0, 0, 0, 0, 0]),
        sce_kernel_error(SCE_AUDIO_OUT2_ERROR_NOT_READY)
    );

    let mut state = crate::AUDIO_OUT2
        .get()
        .expect("the AudioOut2 test state is initialized")
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let context_state = state
        .contexts
        .get_mut(&context)
        .expect("the AudioOut2 test context remains live");
    context_state.last_queue_update -= crate::audio_out2_grain_duration(context_state.grain_frames);
    crate::audio_out2_update_context_queue(context_state, Instant::now());
    assert_eq!(context_state.queued, AUDIO_OUT2_DEFAULT_QUEUE_DEPTH - 1);
}

#[test]
fn audio_out2_port_state_reports_the_stable_default_route() {
    let _serial = AUDIO_OUT2_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_audio_out2_test_state();
    let mut storage = [0xa5_u8; 0x60];
    let state = &mut storage[0x10..0x50];
    assert_eq!(
        dispatch_audio_out2_import("gatEUKG+Ea4", [3, state.as_mut_ptr() as u64, 0, 0, 0, 0],),
        0
    );
    assert_eq!(u16::from_le_bytes([state[0], state[1]]), 1);
    assert_eq!(state[2], 2);
    assert_eq!(i16::from_le_bytes([state[4], state[5]]), 127);
    assert_eq!(u16::from_le_bytes([state[6], state[7]]), 0);
    assert!(state[8..].iter().all(|byte| *byte == 0));
    assert_eq!(&storage[..0x10], &[0xa5; 0x10]);
    assert_eq!(&storage[0x50..], &[0xa5; 0x10]);
}

#[test]
fn audio_out2_speaker_info_reports_the_complete_stereo_topology() {
    let mut storage = [0xa5_u8; 0x60];
    let info = &mut storage[0x10..0x50];
    assert_eq!(
        dispatch_audio_out2_import("DImz2Ft9E2g", [info.as_mut_ptr() as u64, 1, 0, 0, 0, 0],),
        0
    );
    assert_eq!(info[0], 0);
    assert_eq!(u32::from_le_bytes(info[4..8].try_into().unwrap()), 3);
    assert_eq!(u32::from_le_bytes(info[8..12].try_into().unwrap()), 0);
    assert_eq!(
        i16::from_le_bytes(info[0x10..0x12].try_into().unwrap()),
        -30
    );
    assert_eq!(i16::from_le_bytes(info[0x12..0x14].try_into().unwrap()), 0);
    assert_eq!(i16::from_le_bytes(info[0x14..0x16].try_into().unwrap()), 30);
    assert_eq!(i16::from_le_bytes(info[0x16..0x18].try_into().unwrap()), 0);
    assert!(info[0x18..].iter().all(|byte| *byte == 0));
    assert_eq!(&storage[..0x10], &[0xa5; 0x10]);
    assert_eq!(&storage[0x50..], &[0xa5; 0x10]);
}
