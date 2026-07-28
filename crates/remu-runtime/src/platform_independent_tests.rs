use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, mpsc};
use std::time::{Duration, Instant};

use super::AprResolveOutputs;
use super::{
    AGC_IT_DISPATCH_DIRECT, AGC_IT_DISPATCH_INDIRECT, AGC_IT_INDEX_BASE, AGC_IT_INDEX_BUFFER_SIZE,
    AGC_IT_NOP, AGC_IT_NUM_INSTANCES, AGC_IT_SET_BASE, AGC_IT_SET_CONTEXT_REG_INDIRECT,
    AGC_IT_SET_SH_REG_INDIRECT, AGC_IT_SET_UCONFIG_REG_INDEX, AGC_IT_SET_UCONFIG_REG_INDIRECT,
    AGC_R_ACB_RESET, AGC_R_DRAW_INDEX_AUTO, AUDIO_OUT2_CONTEXT_MEMORY_BASE_SIZE,
    AUDIO_OUT2_CONTEXT_MEMORY_QUEUE_STRIDE, AUDIO_OUT2_CONTEXT_PARAM_SIZE,
    AUDIO_OUT2_DEFAULT_GRAIN_FRAMES, AUDIO_OUT2_DEFAULT_QUEUE_DEPTH,
    AUDIO_OUT2_PORT_TYPE_VIBRATION, AgcColorTargetState, AgcComputeDispatchError, AgcGraphicsState,
    AgcShaderRegistry, BATCH_MAP_OPERATION_MAP_DIRECT, DEFAULT_FLIP_INTERVAL_MICROSECONDS,
    GPU_EVENT_SINK_TEST, GuestExitCallback, KERNEL_FILES, KernelFileState, ProbeDispatchState,
    PthreadAttrState, SAVE_DATA_ERROR_EXISTS, SAVE_DATA_ERROR_NOT_FOUND, SAVE_DATA_PARAM_SIZE,
    SCE_AJM_ERROR_INVALID_PARAMETER, SCE_AUDIO_OUT2_ERROR_NOT_READY,
    SCE_KERNEL_ERROR_INVALID_ARGUMENT, SCE_KERNEL_ERROR_NOT_FOUND,
    SCE_KERNEL_ERROR_NOT_IMPLEMENTED, SceAjmBatchInfo, SceAudioOut2Attribute,
    SceAudioOut2ContextParam, SceAudioOut2Pcm, SceAudioOut2PortParam, SceAudioOut2PortState,
    VIDEO_OUT_FLIP_INTERVAL, VideoOutBufferAttributes, VideoOutState, agc_apply_set_base,
    agc_apply_submission_registers, agc_color_target_state, agc_dcb_draw_index_auto,
    agc_dcb_set_index_buffer, agc_dcb_set_num_instances, agc_decode_compute_dispatch,
    agc_driver_submit_acb, agc_pm4, ajm_batch_initialize, ajm_finalize, ajm_initialize,
    ajm_instance_create, ajm_instance_destroy, ajm_module_register, ajm_module_unregister,
    allocate_direct_memory, allocate_pthread_thread_object,
    ampr_command_buffer_write_kernel_event_queue, ampr_submit_command_buffer,
    civil_from_unix_seconds, clone_guest_tls, create_pthread_attr, decode_agc_dma_data,
    decode_agc_memory_acquire, decode_agc_memory_wait, decode_audio_out2_attribute,
    decode_audio_out2_channel_format, destroy_pthread_attr, discover_runtime_module_paths,
    dispatch_audio_out2_import, dispatch_guest_numeric_parser, dispatch_probe_import,
    flip_interval_microseconds, guest_clock_gettime, guest_free, guest_global_locale,
    guest_kernel_convert_localtime_to_utc, guest_kernel_convert_utc_to_localtime,
    guest_kernel_fstat, guest_kernel_ftruncate, guest_kernel_gettimeofday, guest_kernel_lseek,
    guest_kernel_read, guest_posix_file_result, guest_pthread_once, guest_qsort,
    guest_tolower_table, guest_toupper_table, guest_wmemchr, guest_wmemcmp, guest_wmemcpy,
    guest_wmemmove, kernel_add_ampr_event, kernel_batch_map, kernel_release_direct_memory,
    kernel_stat_payload, kernel_virtual_query, low_f32, low_u32, pace_video_out_flip,
    pthread_attr_get_affinity, pthread_attr_get_stack, pthread_attr_set_affinity,
    pthread_attr_set_detach_state, pthread_attr_set_schedule_param, pthread_attr_set_stack_size,
    read_guest_u16, read_guest_u32, read_guest_u64, resolve_guest_path,
    rtc_get_current_clock_local_time, rtc_get_current_tick, rtc_get_tick, rtc_set_tick,
    sanitize_save_segment, save_data_commit, save_data_create_transaction_resource,
    save_data_dir_name_search, save_data_mount3, save_data_pattern_matches, save_data_prepare,
    save_data_save_icon, save_data_set_param, save_data_umount2, sce_kernel_error,
    set_current_guest_thread_pointer, system_service_param_get_int, take_exit_callbacks,
    tls_dtv_entry_offset, unix_days_from_civil, video_out_open, video_out_submit_flip,
    write_guest_u8, write_guest_u16, write_guest_u32, write_guest_u64, write_import_stub,
};
use super::{
    AGC_SHADER_INPUT_SEMANTICS_COUNT_OFFSET, AGC_SHADER_INPUT_SEMANTICS_OFFSET,
    AGC_SHADER_OUTPUT_SEMANTICS_COUNT_OFFSET, AGC_SHADER_OUTPUT_SEMANTICS_OFFSET,
    AGC_SPI_PS_INPUT_CNTL_0, AMPR_INVALID_FILE_ID, AmprCommandBufferState, DIRENT_BLOCK_SIZE,
    DIRENT_HEADER_SIZE, KernelDirectory, SHADER_BINARY_TYPE_GS_BACK, SHADER_BINARY_TYPE_GS_FRONT,
    SHADER_HEADER_CODE_OFFSET, SHADER_HEADER_NUM_SH_REGISTERS_OFFSET,
    SHADER_HEADER_SH_REGISTERS_OFFSET, SHADER_HEADER_SIZE, SHADER_HEADER_TYPE_OFFSET,
    SPI_SHADER_PGM_LO_ES, agc_create_interpolant_mapping, agc_fuse_shader_halves,
    agc_get_fused_shader_size, ampr_resolve_file_paths, ampr_resolve_paths_core, join_guest_path,
};

fn reset_audio_out2_test_state() {
    *super::AUDIO_OUT2
        .get_or_init(|| Mutex::new(super::AudioOut2State::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = super::AudioOut2State::default();
    let mut audio = super::AUDIO_OUT
        .get_or_init(|| Mutex::new(super::audio::AudioOutState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *audio = super::audio::AudioOutState::default();
    audio.use_timed_sink_for_test();
}

fn create_audio_out2_test_context(grain_frames: u32) -> u64 {
    let mut parameter = [0_u8; 0x40];
    assert_eq!(
        dispatch_audio_out2_import(
            "t5YrizufpQc",
            [parameter.as_mut_ptr() as u64, 0, 0, 0, 0, 0],
        ),
        0
    );
    parameter[16..20].copy_from_slice(&grain_frames.to_le_bytes());
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
    let mut memory =
        vec![0_u8; usize::try_from(memory_size).expect("the test AudioOut2 allocation fits usize")];
    let mut context = 0_u64;
    assert_eq!(
        dispatch_audio_out2_import(
            "0x6o1VVAYSY",
            [
                parameter.as_ptr() as u64,
                memory.as_mut_ptr() as u64,
                memory_size,
                std::ptr::from_mut(&mut context) as u64,
                0,
                0,
            ],
        ),
        0
    );
    context
}

fn create_ampr_test_file(contents: &[u8]) -> PathBuf {
    static NEXT_FILE: AtomicUsize = AtomicUsize::new(0);
    let sequence = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "remu-ampr-read-{}-{sequence}.bin",
        std::process::id()
    ));
    std::fs::write(&path, contents).expect("create AMPR test file");
    path
}

fn create_audio_out2_test_port(context: u64, port_type: u16, data_format: u32) -> u64 {
    let parameter = SceAudioOut2PortParam {
        port_type,
        data_format,
        sampling_frequency: 48_000,
        ..SceAudioOut2PortParam::default()
    };
    let mut port = 0_u64;
    assert_eq!(
        dispatch_audio_out2_import(
            "JK2wamZPzwM",
            [
                context,
                std::ptr::addr_of!(parameter) as u64,
                std::ptr::from_mut(&mut port) as u64,
                0,
                0,
                0,
            ],
        ),
        0
    );
    port
}

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

    let state = super::AUDIO_OUT2
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
    let state = super::AUDIO_OUT2
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

    let state = super::AUDIO_OUT2
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

    let mut state = super::AUDIO_OUT2
        .get()
        .expect("the AudioOut2 test state is initialized")
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let context_state = state
        .contexts
        .get_mut(&context)
        .expect("the AudioOut2 test context remains live");
    context_state.last_queue_update -= super::audio_out2_grain_duration(context_state.grain_frames);
    super::audio_out2_update_context_queue(context_state, Instant::now());
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

#[repr(C, align(256))]
struct AlignedComputeProgram([u32; 4]);

#[repr(C, align(8))]
struct AlignedIndirectArguments([u32; 8]);

#[repr(C)]
struct AgcSubmitPacket {
    address: u64,
    dword_count: u32,
    flags: [u8; 4],
}

static AUDIO_OUT2_TEST: Mutex<()> = Mutex::new(());
static GUEST_CONDITION_TEST: Mutex<()> = Mutex::new(());
static GUEST_EXCEPTION_TEST: Mutex<()> = Mutex::new(());
static DELIVERED_EXCEPTION: AtomicU32 = AtomicU32::new(0);
static ONCE_CALLBACKS: AtomicUsize = AtomicUsize::new(0);

#[test]
fn agc_completion_tracking_requires_a_published_queue_submission() {
    let _event_guard = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    remu_gpu::clear_event_sink();
    let mut state = ProbeDispatchState {
        agc_submission_count: 41,
        ..ProbeDispatchState::default()
    };

    super::request_agc_submission_completion(&mut state, 0, Vec::new());
    super::request_agc_compute_submission_completion(&mut state, 32, Vec::new());

    assert!(state.pending_agc_completions.is_empty());
    assert!(state.pending_agc_user_interrupts.is_empty());

    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender);
    super::request_agc_submission_completion(&mut state, 0, Vec::new());
    state.agc_submission_count = 42;
    super::request_agc_compute_submission_completion(&mut state, 32, Vec::new());
    assert_eq!(
        state
            .pending_agc_completions
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [(41, 0)]
    );
    assert_eq!(
        state
            .pending_agc_user_interrupts
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        [42]
    );
    assert!(matches!(
        receiver.recv(),
        Ok(remu_gpu::GpuEvent::QueueSubmission {
            queue_id: 0,
            events,
        }) if events == [remu_gpu::GpuEvent::SubmissionBoundary(41)]
    ));
    assert!(matches!(
        receiver.recv(),
        Ok(remu_gpu::GpuEvent::QueueSubmission {
            queue_id: 33,
            events,
        }) if events == [remu_gpu::GpuEvent::SubmissionBoundary(42)]
    ));
    remu_gpu::clear_event_sink();
}

unsafe extern "C" fn once_test_callback() {
    ONCE_CALLBACKS.fetch_add(1, Ordering::Relaxed);
}

#[cfg(target_arch = "x86_64")]
unsafe extern "C" fn pthread_exit_boundary_test_entry(argument0: u64, argument1: u64) -> u64 {
    super::guest_pthread_exit(argument0 + argument1);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn pthread_exit_returns_to_the_host_thread_boundary() {
    assert_eq!(
        super::invoke_guest_entry_with_exit(
            pthread_exit_boundary_test_entry as *const () as usize as u64,
            20,
            22,
        ),
        42
    );
}

static DELIVERED_EXCEPTION_RSP: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn capture_guest_exception(exception: i32, context: *mut u8) {
    DELIVERED_EXCEPTION.store(exception.cast_unsigned(), Ordering::SeqCst);
    // SAFETY: the delivery test passes a complete 0x500-byte synthesized Orbis ucontext.
    let rsp = unsafe { std::ptr::read_unaligned(context.add(0xf8).cast::<u64>()) };
    DELIVERED_EXCEPTION_RSP.store(usize::try_from(rsp).unwrap(), Ordering::SeqCst);
}

#[test]
fn delivers_completed_agc_submissions_through_the_registered_graphics_equeue() {
    let _serial = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let completed = remu_gpu::gpu_completed_submission_sequence();
    let first = completed.saturating_add(1);
    let second = first.saturating_add(1);
    let mut state = ProbeDispatchState::default();
    let mut equeue = 0_u64;
    assert_eq!(
        super::dispatch_stateless_import(
            "D0OdFMjp46I",
            [(&raw mut equeue).addr() as u64, 0, 0, 0, 0, 0],
            [0; 6],
        ),
        None,
        "the stateful equeue creator must not be shadowed by a placeholder"
    );
    assert_eq!(
        super::kernel_create_equeue((&raw mut equeue).addr() as u64, &mut state),
        0
    );
    assert_eq!(equeue, 1);
    assert_eq!(
        super::agc_driver_add_eq_event([equeue, 0x1234, 0xfeed_face, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        super::agc_driver_add_eq_event([equeue, 0x20, 0xcafe_babe, 0, 0, 0], &mut state),
        0
    );
    state
        .pending_agc_completions
        .extend([(first, 0x1234), (second, 0x1234)]);
    remu_gpu::record_gpu_submission_completed(second);
    remu_gpu::record_gpu_submission_completed(first);

    let mut event = [0_u8; 0x40];
    let mut count = 0_u32;
    assert_eq!(
        super::kernel_poll_equeue(
            [
                equeue,
                event.as_mut_ptr().addr() as u64,
                2,
                (&raw mut count).addr() as u64,
                0,
                0,
            ],
            &mut state,
        ),
        Ok(1)
    );
    assert_eq!(count, 1);
    assert_eq!(read_guest_u64(event.as_ptr().addr() as u64), 0x1234);
    assert_eq!(
        read_guest_u16(event.as_ptr().addr() as u64 + 8),
        super::KERNEL_EVENT_FILTER_GRAPHICS.cast_unsigned()
    );
    assert_eq!(read_guest_u32(event.as_ptr().addr() as u64 + 0x0c), 2);
    assert_eq!(read_guest_u64(event.as_ptr().addr() as u64 + 0x10), 0x1234);
    assert_eq!(
        read_guest_u64(event.as_ptr().addr() as u64 + 0x18),
        0xfeed_face
    );
    assert!(
        state
            .pending_events
            .iter()
            .all(|pending| pending.ident != 0x20),
        "graphics completion must not wake an unrelated compute queue"
    );
    assert!(state.pending_agc_completions.is_empty());
}

#[test]
fn routes_completed_compute_submissions_only_to_the_user_interrupt() {
    let _serial = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sequence = remu_gpu::gpu_completed_submission_sequence().saturating_add(1);
    let mut state = ProbeDispatchState::default();
    state.kernel_equeues.extend([1, 2]);
    assert_eq!(
        super::kernel_register_event(
            &mut state,
            1,
            0x1800,
            super::KERNEL_EVENT_FILTER_USER,
            0xfeed_face,
        ),
        0
    );
    assert_eq!(
        super::kernel_register_event(
            &mut state,
            2,
            0x20,
            super::KERNEL_EVENT_FILTER_GRAPHICS,
            0xcafe_babe,
        ),
        0
    );
    state.pending_agc_user_interrupts.push_back(sequence);
    remu_gpu::record_gpu_submission_completed(sequence);

    super::refresh_completed_agc_submissions(&mut state);

    assert!(state.pending_agc_completions.is_empty());
    assert!(state.pending_agc_user_interrupts.is_empty());
    assert_eq!(state.pending_events.len(), 1);
    let user_event = state.pending_events.front().expect("AGC user interrupt");
    assert_eq!(user_event.equeue, 1);
    assert_eq!(user_event.ident, 0x1800);
    assert_eq!(user_event.fflags, 1);
    assert_eq!(user_event.user_data, 0xfeed_face);
    assert!(
        state
            .pending_events
            .iter()
            .all(|event| event.filter != super::KERNEL_EVENT_FILTER_GRAPHICS),
        "compute completion must not also wake the owner's graphics event"
    );
}

fn compute_graphics_state(program_address: u64) -> (AgcGraphicsState, AgcShaderRegistry) {
    let mut graphics = AgcGraphicsState::default();
    for (offset, value) in [
        (
            remu_gpu::COMPUTE_PGM_LO_REGISTER,
            low_u32(program_address >> 8),
        ),
        (
            remu_gpu::COMPUTE_PGM_HI_REGISTER,
            ((program_address >> 40) & 0xff) as u32,
        ),
        (
            remu_gpu::COMPUTE_PGM_RSRC2_REGISTER,
            (2 << 1) | (1 << 7) | (1 << 9) | (1 << 10),
        ),
        (remu_gpu::COMPUTE_NUM_THREAD_X_REGISTER, 64),
        (remu_gpu::COMPUTE_NUM_THREAD_Y_REGISTER, 2),
        (remu_gpu::COMPUTE_NUM_THREAD_Z_REGISTER, 1),
        (0x240, 0x1234_5678),
        (0x241, 0x9abc_def0),
    ] {
        graphics.shader_registers.insert(offset, value);
    }
    let mut shaders = AgcShaderRegistry::default();
    shaders.snapshots.insert(
        program_address,
        remu_gpu::ShaderSnapshot {
            header_address: 0,
            program: Arc::new(remu_shader::Program::new(program_address, Vec::new())),
            supplemental_programs: Arc::from([]),
            metadata: None,
        },
    );
    (graphics, shaders)
}

fn targetless_scanout_test_draw() -> remu_gpu::IndexedDraw {
    let pixel_program = remu_shader::Program::new(
        0x2000,
        vec![remu_shader::Instruction {
            pc: 0,
            encoding: remu_shader::ShaderEncoding::Exp,
            opcode: remu_shader::Opcode::EXPORT,
            words: vec![0, 0],
            sources: Vec::new(),
            destinations: Vec::new(),
            control: Some(remu_shader::InstructionControl::Export(
                remu_shader::ExportControl {
                    target: 0,
                    enable_mask: 0xf,
                    compressed: false,
                    done: true,
                    valid_mask: false,
                },
            )),
        }],
    );
    remu_gpu::IndexedDraw {
        sequence: 1,
        indexed: false,
        instance_count: 1,
        index_buffer_address: 0,
        index_resource_address: 0,
        index_buffer_count: 0,
        index_size: 0,
        index_offset: 0,
        index_count: 3,
        index_snapshot: None,
        draw_flags: 0,
        primitive_type: 4,
        export_shader_address: 0,
        pixel_shader_address: 0x2000,
        export_shader_prefix: [0; 4],
        pixel_shader_prefix: [0; 4],
        export_shader: None,
        pixel_shader: Some(remu_gpu::ShaderSnapshot {
            header_address: 0,
            program: Arc::new(pixel_program),
            supplemental_programs: Arc::from([]),
            metadata: None,
        }),
        export_shader_evaluation: None,
        pixel_shader_evaluation: None,
        color_target_address: 0,
        scanout_target: None,
        context_registers: vec![remu_gpu::GpuRegister {
            offset: super::AGC_CB_SHADER_MASK,
            value: 0xf,
        }],
        shader_registers: Vec::new(),
        user_config_registers: Vec::new(),
    }
}

fn targetless_scanout_test_video(address: u64) -> VideoOutState {
    VideoOutState {
        is_open: true,
        width: 1_280,
        height: 720,
        display_buffers: vec![address],
        display_buffer_attributes: vec![Some(VideoOutBufferAttributes {
            width: 1_280,
            height: 720,
            pitch: 1_280,
            pixel_format: 0x8000_0000_2200_0000,
            tiling_mode: 0,
        })],
        ..VideoOutState::default()
    }
}

#[test]
fn direct_video_out_submit_flip_reaches_the_presenter() {
    let _event_guard = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let address = 0x0070_1234_5600_u64;
    let mut state = ProbeDispatchState {
        video_out: targetless_scanout_test_video(address),
        ..ProbeDispatchState::default()
    };
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender);

    let result = video_out_submit_flip([1, 0, 0, 0x1234, 0, 0], &mut state);
    super::QUEUED_FLIPS.with(|queued| queued.set(0));

    assert_eq!(result, 0);
    assert_eq!(remu_gpu::gpu_flips_pending(), 1);
    let remu_gpu::GpuEvent::Flip {
        address: flipped_address,
        buffer_index,
        flip_argument,
        width,
        height,
        ..
    } = receiver.recv().expect("direct VideoOut flip event")
    else {
        panic!("direct VideoOut submission must emit a presenter flip");
    };
    assert_eq!(flipped_address, address);
    assert_eq!(buffer_index, 0);
    assert_eq!(flip_argument, 0x1234);
    assert_eq!((width, height), (1_280, 720));
    assert_eq!(
        super::video_out_is_flip_pending([1, 0, 0, 0, 0, 0], &mut state),
        1
    );
    remu_gpu::record_gpu_flip_completed(remu_gpu::GpuFlipCompletion {
        address,
        buffer_index,
        flip_argument,
    });
    assert_eq!(remu_gpu::gpu_flips_pending(), 0);
    super::refresh_completed_video_out_flips(&mut state);
    assert_eq!(
        super::video_out_is_flip_pending([1, 0, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(state.video_out.completed_flip_count, 1);
    assert!(receiver.try_recv().is_err());
    remu_gpu::clear_event_sink();
}

fn targetless_scanout_test_shader_state(
    pixel_address: u64,
    target: remu_gpu::GpuColorTarget,
) -> (AgcGraphicsState, AgcShaderRegistry) {
    let mut graphics = AgcGraphicsState::default();
    graphics.known_color_targets.insert(target.address, target);
    let mut template = targetless_scanout_test_draw();
    template.pixel_shader_address = pixel_address;
    let mut pixel_snapshot = template.pixel_shader.take().expect("test pixel shader");
    pixel_snapshot.program = Arc::new(remu_shader::Program::new(
        pixel_address,
        pixel_snapshot.program.instructions.clone(),
    ));
    graphics.context_registers.extend(
        template
            .context_registers
            .iter()
            .map(|register| (register.offset, register.value)),
    );
    graphics.shader_registers.extend([
        (super::AGC_SPI_SHADER_PGM_LO_PS, low_u32(pixel_address >> 8)),
        (
            super::AGC_SPI_SHADER_PGM_HI_PS,
            low_u32(pixel_address >> 40),
        ),
        (0x00b, 0),
    ]);
    let mut shaders = AgcShaderRegistry::default();
    shaders.snapshots.insert(pixel_address, pixel_snapshot);
    (graphics, shaders)
}

fn targetless_barrier_acquire_flip_commands() -> (Vec<u32>, remu_gpu::GpuMemoryAcquire) {
    let acquire = [
        agc_pm4(8, AGC_IT_NOP, super::AGC_R_ACQUIRE_MEM),
        0x8000_1234,
        0x3456_789a,
        0x0000_00ab,
        0x7654_3210,
        0x0000_cdef,
        0xabcd_2468,
        0xfff5_4321,
    ];
    let expected_acquire = decode_agc_memory_acquire(acquire.as_ptr().addr() as u64);
    let mut commands = vec![
        agc_pm4(3, super::AGC_IT_DRAW_INDEX_AUTO, 0),
        3,
        0,
        agc_pm4(2, super::AGC_IT_EVENT_WRITE, 0),
        0x410,
    ];
    commands.extend(acquire);
    commands.extend([
        agc_pm4(6, AGC_IT_NOP, super::AGC_R_FLIP),
        1,
        0,
        0,
        0x1234,
        0,
    ]);
    (commands, expected_acquire)
}

#[test]
fn gpu_progress_telemetry_keeps_a_prefix_and_logarithmic_checkpoints() {
    for sequence in 0..=8 {
        assert!(super::trace_logarithmic_sequence(sequence, 8));
    }
    assert!(!super::trace_logarithmic_sequence(9, 8));
    assert!(super::trace_logarithmic_sequence(16, 8));
    assert!(!super::trace_logarithmic_sequence(17, 8));
    assert!(super::trace_logarithmic_sequence(1_u64 << 63, 8));
}

#[test]
fn condition_signal_selects_one_waiter_and_broadcast_selects_the_rest() {
    let mut condition = super::GuestConditionState::default();
    let mutex = 0x1234;
    let first = 1;
    let second = 2;
    let third = 3;
    assert!(condition.register_waiter(first, mutex));
    assert!(condition.register_waiter(second, mutex));
    assert!(condition.register_waiter(third, mutex));
    assert!(!condition.register_waiter(4, mutex + 1));

    assert_eq!(condition.release_one(), Some(first));
    assert_eq!(
        condition.waiters.iter().copied().collect::<Vec<_>>(),
        [second, third]
    );
    assert_eq!(condition.bound_mutex, Some(mutex));

    assert_eq!(
        condition.release_all().into_iter().collect::<Vec<_>>(),
        [second, third]
    );
    assert!(condition.waiters.is_empty());
    assert_eq!(condition.bound_mutex, None);
    assert!(condition.is_idle());

    assert_eq!(condition.release_one(), None, "signals are not retained");
    let future = 5;
    assert!(condition.register_waiter(future, mutex + 1));
    condition.cancel_wait(future);
    assert!(condition.is_idle());
    assert_eq!(condition.bound_mutex, None);
}

#[test]
fn rwlock_tracks_recursive_read_ownership_and_excludes_writers() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let address = 0x7fff_0000_0080;
    assert_eq!(super::guest_rwlock_init(address), 0);
    assert_eq!(super::guest_rwlock_lock(address, false, false), 0);
    assert_eq!(super::guest_rwlock_lock(address, false, false), 0);
    assert_eq!(
        super::guest_rwlock_lock(address, true, true),
        u64::from(libc::EBUSY.cast_unsigned())
    );
    assert_eq!(
        super::guest_rwlock_destroy(address),
        u64::from(libc::EBUSY.cast_unsigned())
    );
    assert_eq!(super::guest_rwlock_unlock(address), 0);
    assert_eq!(super::guest_rwlock_unlock(address), 0);
    assert_eq!(super::guest_rwlock_lock(address, true, false), 0);
    assert_eq!(
        super::guest_rwlock_lock(address, false, true),
        u64::from(libc::EBUSY.cast_unsigned())
    );
    assert_eq!(super::guest_rwlock_unlock(address), 0);
    assert_eq!(super::guest_rwlock_destroy(address), 0);
}

#[test]
fn mutex_types_apply_gen5_same_thread_relock_semantics() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let cases = [
        (0x7fff_0000_1100, 1, libc::EDEADLK),
        (0x7fff_0000_1200, 3, libc::EBUSY),
        (0x7fff_0000_1300, 4, libc::EDEADLK),
    ];
    for (address, mutex_type, expected) in cases {
        assert_eq!(
            super::guest_mutex_init(
                address,
                super::PthreadMutexAttrState {
                    mutex_type,
                    ..super::PthreadMutexAttrState::default()
                }
            ),
            0
        );
        assert_eq!(super::guest_mutex_lock(address, false), 0);
        assert_eq!(
            super::guest_mutex_lock(address, mutex_type == 3),
            u64::from(expected.cast_unsigned())
        );
        assert_eq!(super::guest_mutex_unlock(address), 0);
        assert_eq!(super::guest_mutex_destroy(address), 0);
    }

    let recursive = 0x7fff_0000_1400;
    assert_eq!(
        super::guest_mutex_init(
            recursive,
            super::PthreadMutexAttrState {
                mutex_type: 2,
                ..super::PthreadMutexAttrState::default()
            }
        ),
        0
    );
    assert_eq!(super::guest_mutex_lock(recursive, false), 0);
    assert_eq!(super::guest_mutex_lock(recursive, false), 0);
    assert_eq!(super::guest_mutex_unlock(recursive), 0);
    assert_eq!(
        super::guest_mutex_destroy(recursive),
        u64::from(libc::EBUSY.cast_unsigned())
    );
    assert_eq!(super::guest_mutex_unlock(recursive), 0);
    assert_eq!(super::guest_mutex_destroy(recursive), 0);
}

#[test]
fn mutex_unlock_rejects_missing_and_non_owner_mutexes() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let missing = 0x7fff_0000_1500;
    assert_eq!(
        super::guest_mutex_unlock(missing),
        sce_kernel_error(super::SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED)
    );

    let mutex = 0x7fff_0000_1600;
    assert_eq!(
        super::guest_mutex_init(mutex, super::PthreadMutexAttrState::default()),
        0
    );
    assert_eq!(super::guest_mutex_lock(mutex, false), 0);
    assert_eq!(
        std::thread::spawn(move || super::guest_mutex_unlock(mutex))
            .join()
            .expect("non-owner unlock thread"),
        sce_kernel_error(super::SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED)
    );
    assert_eq!(super::guest_mutex_unlock(mutex), 0);
    assert_eq!(super::guest_mutex_destroy(mutex), 0);
}

#[test]
fn mutex_init_installs_the_guest_visible_object_libc_updates() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut storage = 0_u64;
    let address = (&raw mut storage).addr() as u64;
    assert_eq!(
        super::guest_mutex_init(address, super::PthreadMutexAttrState::default()),
        0
    );
    assert_ne!(storage, 0);
    let handle = storage;
    assert_eq!(
        super::read_guest_u32(handle + 0x20) & 0xff,
        super::PthreadMutexAttrState::default()
            .mutex_type
            .cast_unsigned()
    );
    assert_eq!(
        super::read_guest_u32(handle + 0x3c),
        super::PthreadMutexAttrState::default()
            .protocol
            .cast_unsigned()
    );

    // PS5 libc performs this exact update after scePthreadMutexInit succeeds. The returned
    // opaque object therefore has to be real writable guest-visible storage.
    let flags = super::read_guest_u32(handle + 0x20);
    super::write_guest_u32(handle + 0x20, flags | 0x100);
    assert_eq!(super::read_guest_u32(handle + 0x20) & 0x100, 0x100);

    // Both the public storage address and libc's resolved opaque pointer identify the same
    // mutex in different generations of the SDK.
    assert_eq!(super::guest_mutex_lock(handle, false), 0);
    assert_eq!(super::guest_mutex_unlock(address), 0);
    assert_eq!(super::guest_mutex_destroy(handle), 0);
    assert_eq!(storage, 0);
}

#[test]
fn module_unwind_query_writes_the_gen5_metadata_layout() {
    let mut state = ProbeDispatchState::default();
    state.unwind_modules.push(super::GuestUnwindModule {
        name: "eboot.bin".to_owned(),
        base: 0x1000,
        end: 0x9000,
        eh_frame_header: 0x7000,
        eh_frame: 0x5000,
        eh_frame_size: 0x2000,
    });
    let mut output = vec![0xcc_u8; 0x130];
    let output_address = output.as_mut_ptr().addr() as u64;
    super::write_guest_u64(output_address, 0x130);
    assert_eq!(
        super::kernel_module_info_for_unwind([0x1234, 1, output_address, 0, 0, 0], &state),
        0
    );
    assert_eq!(super::read_guest_u64(output_address), 0x130);
    assert_eq!(&output[8..17], b"eboot.bin");
    assert_eq!(super::read_guest_u64(output_address + 0x108), 0x7000);
    assert_eq!(super::read_guest_u64(output_address + 0x110), 0x5000);
    assert_eq!(super::read_guest_u64(output_address + 0x118), 0x2000);
    assert_eq!(super::read_guest_u64(output_address + 0x120), 0x1000);
    assert_eq!(super::read_guest_u64(output_address + 0x128), 0x8000);
}

#[test]
fn timed_condition_wait_reacquires_the_guest_mutex_before_returning() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let condition = 0x7fff_0000_0100;
    let mutex = 0x7fff_0000_0200;
    assert_eq!(super::guest_condition_init(condition), 0);
    assert_eq!(
        super::guest_mutex_init(
            mutex,
            super::PthreadMutexAttrState {
                mutex_type: 2,
                ..super::PthreadMutexAttrState::default()
            }
        ),
        0
    );
    assert_eq!(super::guest_mutex_lock(mutex, false), 0);
    assert_eq!(super::guest_mutex_lock(mutex, false), 0);

    assert_eq!(
        super::guest_condition_wait(condition, mutex, Some(0)),
        sce_kernel_error(super::SCE_KERNEL_ERROR_TIMED_OUT)
    );
    {
        let entry = super::guest_mutex_entry(mutex, false).expect("test mutex");
        let guest_mutex = entry
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(guest_mutex.owner, Some(std::thread::current().id()));
        assert_eq!(guest_mutex.recursion, 2);
        let state = super::guest_sync()
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(
            state
                .conditions
                .get(&condition)
                .is_some_and(super::GuestConditionState::is_idle)
        );
    }

    assert_eq!(super::guest_mutex_unlock(mutex), 0);
    assert_eq!(super::guest_mutex_unlock(mutex), 0);
    assert_eq!(super::guest_mutex_destroy(mutex), 0);
    assert_eq!(super::guest_condition_destroy(condition), 0);
}

#[test]
fn condition_wait_delivers_interruption_before_reacquiring_guest_mutex() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    const THREAD: u64 = 0x7fff_3000;
    let condition = 0x7fff_0000_0500;
    let mutex = 0x7fff_0000_0600;
    assert_eq!(super::guest_condition_init(condition), 0);
    assert_eq!(
        super::guest_mutex_init(mutex, super::PthreadMutexAttrState::default()),
        0
    );

    let (handler_entered_tx, handler_entered_rx) = std::sync::mpsc::channel();
    let (resume_handler_tx, resume_handler_rx) = std::sync::mpsc::channel();
    let (finished_tx, finished_rx) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        super::set_current_guest_thread_handle(THREAD);
        assert_eq!(super::guest_mutex_lock(mutex, false), 0);
        assert_eq!(
            super::guest_condition_wait_with_interrupt(condition, mutex, None, || {
                handler_entered_tx
                    .send(())
                    .expect("test receiver remains live");
                resume_handler_rx
                    .recv()
                    .expect("test suspension controller remains live");
            }),
            0
        );
        assert_eq!(super::guest_mutex_unlock(mutex), 0);
        super::set_current_guest_thread_handle(0);
        finished_tx.send(()).expect("test receiver remains live");
    });
    wait_for_guest_condition_waiters(condition, 1);

    // Model stop-the-world with another guest thread holding this condition's mutex. The
    // interrupted waiter must enter its suspension handler even though mutex reacquisition
    // cannot yet succeed.
    assert_eq!(super::guest_mutex_lock(mutex, false), 0);
    super::guest_exception_delivery()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .pending
        .insert(
            THREAD,
            super::PendingGuestException {
                handler: 1,
                exception: 30,
            },
        );
    super::guest_sync().1.notify_all();
    handler_entered_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("interruption handler ran before mutex reacquisition");

    resume_handler_tx
        .send(())
        .expect("waiter remains in its handler");
    assert!(
        finished_rx
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err(),
        "condition wait returned without reacquiring its mutex"
    );
    assert_eq!(super::guest_mutex_unlock(mutex), 0);
    finished_rx
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("resumed waiter reacquired the mutex");
    waiter.join().expect("guest waiter thread");

    super::guest_exception_delivery()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .pending
        .remove(&THREAD);
    assert_eq!(super::guest_mutex_destroy(mutex), 0);
    assert_eq!(super::guest_condition_destroy(condition), 0);
}

#[test]
fn condition_wait_errors_use_the_selected_pthread_abi() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let condition = 0x7fff_0000_0900;
    let mutex = 0x7fff_0000_0a00;
    assert_eq!(super::guest_condition_init(condition), 0);
    assert_eq!(
        super::guest_mutex_init(mutex, super::PthreadMutexAttrState::default()),
        0
    );
    assert_eq!(
        super::dispatch_guest_sync_import("WKAXJ4XBPQ4", [condition, mutex, 0, 0, 0, 0], None),
        sce_kernel_error(super::SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED)
    );
    assert_eq!(
        super::dispatch_guest_sync_import("Op8TBGY5KHg", [condition, mutex, 0, 0, 0, 0], None),
        u64::from(libc::EPERM.cast_unsigned())
    );
    assert_eq!(super::guest_mutex_destroy(mutex), 0);
    assert_eq!(super::guest_condition_destroy(condition), 0);
}

#[test]
fn condition_signal_wakes_exactly_one_blocked_guest_thread() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let condition = 0x7fff_0000_1100;
    let mutex = 0x7fff_0000_1200;
    assert_eq!(super::guest_condition_init(condition), 0);
    assert_eq!(
        super::guest_mutex_init(mutex, super::PthreadMutexAttrState::default()),
        0
    );
    let (finished, receiver) = std::sync::mpsc::channel();
    let waiters = (0..2)
        .map(|waiter| {
            let finished = finished.clone();
            std::thread::spawn(move || {
                assert_eq!(super::guest_mutex_lock(mutex, false), 0);
                assert_eq!(super::guest_condition_wait(condition, mutex, None), 0);
                assert_eq!(super::guest_mutex_unlock(mutex), 0);
                finished.send(waiter).expect("test receiver remains live");
            })
        })
        .collect::<Vec<_>>();
    drop(finished);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        let waiting = super::guest_sync()
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .conditions
            .get(&condition)
            .map_or(0, |condition| condition.waiters.len());
        if waiting == 2 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "guest waiters did not block"
        );
        std::thread::yield_now();
    }

    assert_eq!(
        super::dispatch_guest_sync_import("kDh-NfxgMtE", [condition, 0, 0, 0, 0, 0], None),
        0
    );
    let first = receiver
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("one signaled waiter");
    let early_second = receiver
        .recv_timeout(std::time::Duration::from_millis(50))
        .ok();
    assert_eq!(
        super::dispatch_guest_sync_import("JGgj7Uvrl+A", [condition, 0, 0, 0, 0, 0], None),
        0
    );
    let second = early_second.unwrap_or_else(|| {
        receiver
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("broadcast waiter")
    });
    assert_ne!(first, second);
    assert!(
        early_second.is_none(),
        "signal acquired broadcast semantics"
    );
    for waiter in waiters {
        waiter.join().expect("guest waiter thread");
    }
    assert_eq!(super::guest_mutex_destroy(mutex), 0);
    assert_eq!(super::guest_condition_destroy(condition), 0);
}

#[test]
fn condition_signal_allows_exactly_one_predicate_guarded_task_consumption() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let condition = 0x7fff_0000_1900;
    let mutex = 0x7fff_0000_1a00;
    assert_eq!(super::guest_condition_init(condition), 0);
    assert_eq!(
        super::guest_mutex_init(mutex, super::PthreadMutexAttrState::default()),
        0
    );
    let jobs = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let stopping = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (consumed, receiver) = std::sync::mpsc::channel();
    let workers = (0..3)
        .map(|worker| {
            let jobs = std::sync::Arc::clone(&jobs);
            let stopping = std::sync::Arc::clone(&stopping);
            let consumed = consumed.clone();
            std::thread::spawn(move || {
                loop {
                    assert_eq!(super::guest_mutex_lock(mutex, false), 0);
                    while jobs.load(std::sync::atomic::Ordering::SeqCst) == 0
                        && !stopping.load(std::sync::atomic::Ordering::SeqCst)
                    {
                        assert_eq!(super::guest_condition_wait(condition, mutex, None), 0);
                    }
                    if stopping.load(std::sync::atomic::Ordering::SeqCst) {
                        assert_eq!(super::guest_mutex_unlock(mutex), 0);
                        break;
                    }
                    assert_eq!(jobs.fetch_sub(1, std::sync::atomic::Ordering::SeqCst), 1);
                    assert_eq!(super::guest_mutex_unlock(mutex), 0);
                    consumed.send(worker).expect("test receiver remains live");
                }
            })
        })
        .collect::<Vec<_>>();
    drop(consumed);
    wait_for_guest_condition_waiters(condition, 3);

    assert_eq!(super::guest_mutex_lock(mutex, false), 0);
    jobs.store(1, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(super::guest_condition_notify(condition, false), 0);
    assert_eq!(super::guest_mutex_unlock(mutex), 0);
    receiver
        .recv_timeout(std::time::Duration::from_secs(2))
        .expect("one worker consumed the queued task");
    assert!(
        receiver
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err(),
        "one signal cannot produce a second task consumption"
    );
    wait_for_guest_condition_waiters(condition, 3);

    assert_eq!(super::guest_mutex_lock(mutex, false), 0);
    stopping.store(true, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(super::guest_condition_notify(condition, true), 0);
    assert_eq!(super::guest_mutex_unlock(mutex), 0);
    for worker in workers {
        worker.join().expect("guest worker thread");
    }
    assert_eq!(jobs.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(super::guest_mutex_destroy(mutex), 0);
    assert_eq!(super::guest_condition_destroy(condition), 0);
}

fn wait_for_guest_condition_waiters(condition: u64, expected: usize) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        let waiting = super::guest_sync()
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .conditions
            .get(&condition)
            .map_or(0, |condition| condition.waiters.len());
        if waiting == expected {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "guest condition waiter count did not reach {expected}"
        );
        std::thread::yield_now();
    }
}

#[test]
fn selected_condition_waiter_survives_legal_condition_destruction() {
    let _guard = GUEST_CONDITION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let condition = 0x7fff_0000_2100;
    let mutex = 0x7fff_0000_2200;
    assert_eq!(super::guest_condition_init(condition), 0);
    assert_eq!(
        super::guest_mutex_init(mutex, super::PthreadMutexAttrState::default()),
        0
    );
    let (events, receiver) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || {
        assert_eq!(super::guest_mutex_lock(mutex, false), 0);
        events.send("locked").expect("test receiver remains live");
        assert_eq!(super::guest_condition_wait(condition, mutex, None), 0);
        events.send("returned").expect("test receiver remains live");
        assert_eq!(super::guest_mutex_unlock(mutex), 0);
    });
    assert_eq!(
        receiver
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("waiter acquired guest mutex"),
        "locked"
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        let waiting = super::guest_sync()
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .conditions
            .get(&condition)
            .map_or(0, |condition| condition.waiters.len());
        if waiting == 1 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "guest waiter did not block"
        );
        std::thread::yield_now();
    }

    assert_eq!(super::guest_mutex_lock(mutex, false), 0);
    assert_eq!(
        super::guest_condition_destroy(condition),
        sce_kernel_error(super::SCE_KERNEL_ERROR_BUSY)
    );
    assert_eq!(super::guest_condition_notify(condition, false), 0);
    assert_eq!(super::guest_condition_destroy(condition), 0);
    assert!(
        receiver
            .recv_timeout(std::time::Duration::from_millis(50))
            .is_err(),
        "selected waiter must reacquire its mutex before returning"
    );
    assert_eq!(super::guest_mutex_unlock(mutex), 0);
    assert_eq!(
        receiver
            .recv_timeout(std::time::Duration::from_secs(2))
            .expect("selected waiter reacquired its mutex"),
        "returned"
    );
    waiter.join().expect("guest waiter thread");
    assert_eq!(super::guest_mutex_destroy(mutex), 0);
}

unsafe extern "C" fn compare_i32(left: *const u8, right: *const u8) -> i32 {
    // SAFETY: guest_qsort passes pointers to complete i32 elements in the test array.
    let left = unsafe { std::ptr::read_unaligned(left.cast::<i32>()) };
    // SAFETY: guest_qsort passes pointers to complete i32 elements in the test array.
    let right = unsafe { std::ptr::read_unaligned(right.cast::<i32>()) };
    match left.cmp(&right) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

#[test]
fn posix_file_errors_return_minus_one_and_store_guest_errno() {
    let mut tls = [0_u64; 16];
    let thread_pointer = u64::try_from(tls.as_mut_ptr().addr()).unwrap();
    set_current_guest_thread_pointer(thread_pointer);
    assert_eq!(guest_posix_file_result(7), 7);
    assert_eq!(read_guest_u32(thread_pointer + 0x40), 0);
    assert_eq!(
        guest_posix_file_result(super::sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)),
        u64::MAX
    );
    assert_eq!(
        read_guest_u32(thread_pointer + 0x40),
        libc::ENOENT.cast_unsigned()
    );
    assert_eq!(
        guest_posix_file_result(super::sce_kernel_error(
            super::SCE_KERNEL_ERROR_INVALID_ARGUMENT
        )),
        u64::MAX
    );
    assert_eq!(
        read_guest_u32(thread_pointer + 0x40),
        libc::EINVAL.cast_unsigned()
    );
    set_current_guest_thread_pointer(0);
}

#[test]
fn tls_get_addr_uses_compact_per_thread_module_slots() {
    fn write_template_word(template: &mut [u8], offset: usize, value: u64) {
        template[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    let template_size = usize::try_from(super::GUEST_TLS_PREFIX_SIZE + super::GUEST_TLS_TCB_SIZE)
        .expect("TLS template size fits usize");
    let thread_pointer_offset =
        usize::try_from(super::GUEST_TLS_PREFIX_SIZE).expect("TLS prefix fits usize");
    let dtv_offset = thread_pointer_offset + 0x100;
    let original_thread_pointer = 0x7000_1000_0000_u64;
    let module1_block = original_thread_pointer - 0x40;
    let module2_block = original_thread_pointer - 0x600;
    let mut template = vec![0_u8; template_size];
    write_template_word(
        &mut template,
        thread_pointer_offset,
        original_thread_pointer,
    );
    write_template_word(&mut template, dtv_offset, 1);
    write_template_word(&mut template, dtv_offset + 8, 2);
    assert_eq!(tls_dtv_entry_offset(1), Some(16));
    assert_eq!(tls_dtv_entry_offset(2), Some(24));
    write_template_word(&mut template, dtv_offset + 16, module1_block);
    write_template_word(&mut template, dtv_offset + 24, module2_block);

    let first = clone_guest_tls(&template);
    let second = clone_guest_tls(&template);
    let first_module2 = first.thread_pointer - 0x600;
    let second_module2 = second.thread_pointer - 0x600;
    assert_ne!(first_module2, second_module2);
    assert_eq!(
        read_guest_u64(first.thread_pointer + 0x100 + 24),
        first_module2
    );
    assert_eq!(
        read_guest_u64(second.thread_pointer + 0x100 + 24),
        second_module2
    );
    assert_eq!(read_guest_u64(first.thread_pointer + 0x100 + 32), 0);

    let mut descriptor = [2_u64, 0x280];
    let descriptor_address = descriptor.as_mut_ptr().addr() as u64;
    let mut state = ProbeDispatchState::default();
    set_current_guest_thread_pointer(first.thread_pointer);
    assert_eq!(
        dispatch_probe_import(
            "vNe1w4diLCs",
            [descriptor_address, 0, 0, 0, 0, 0],
            [0; 6],
            &mut state,
        ),
        first_module2 + 0x280
    );
    assert!(state.unimplemented_nids.is_empty());
    assert_eq!(super::known_nid_name("vNe1w4diLCs"), Some("__tls_get_addr"));

    write_guest_u32(first_module2 + 0x280, 0x1122_3344);
    assert_eq!(read_guest_u32(first_module2 + 0x280), 0x1122_3344);
    assert_eq!(read_guest_u32(second_module2 + 0x280), 0);
    set_current_guest_thread_pointer(second.thread_pointer);
    assert_eq!(
        dispatch_probe_import(
            "vNe1w4diLCs",
            [descriptor_address, 0, 0, 0, 0, 0],
            [0; 6],
            &mut state,
        ),
        second_module2 + 0x280
    );

    write_guest_u64(second.thread_pointer + 0x100 + 24, 0);
    assert_eq!(super::guest_tls_get_addr(descriptor_address), 0);
    write_guest_u64(second.thread_pointer + 0x100 + 24, second_module2);
    write_guest_u64(second.thread_pointer + 8, 0);
    assert_eq!(super::guest_tls_get_addr(descriptor_address), 0);
    write_guest_u64(second.thread_pointer + 8, second.thread_pointer + 0x100);

    descriptor[0] = 0;
    assert_eq!(super::guest_tls_get_addr(descriptor_address), 0);
    descriptor[0] = 3;
    assert_eq!(super::guest_tls_get_addr(descriptor_address), 0);
    descriptor[0] = 2;
    descriptor[1] = u64::MAX;
    std::hint::black_box(descriptor[1]);
    assert_eq!(super::guest_tls_get_addr(descriptor_address), 0);
    assert_eq!(super::guest_tls_get_addr(0), 0);
    set_current_guest_thread_pointer(0);
    assert_eq!(super::guest_tls_get_addr(descriptor_address), 0);
}

#[test]
fn video_out_open_owns_the_single_main_display_port() {
    let mut state = ProbeDispatchState::default();
    assert_eq!(video_out_open([0xff, 0, 0, 0, 0, 0], &mut state), 1);
    assert_eq!(state.video_out.user_id, 0xff);
    assert_eq!(state.video_out.bus_type, 0);
    assert_eq!(state.video_out.index, 0);
    assert_eq!(
        video_out_open([0xff, 0, 0, 0, 0, 0], &mut state),
        sce_kernel_error(super::VIDEO_OUT_ERROR_RESOURCE_BUSY)
    );

    let mut other = ProbeDispatchState::default();
    assert_eq!(
        video_out_open([0xff, 1, 0, 0, 0, 0], &mut other),
        sce_kernel_error(super::VIDEO_OUT_ERROR_INVALID_VALUE)
    );
}

#[test]
fn copies_gen5_wide_characters_as_two_byte_elements() {
    let mut words = [
        0x41_u16, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0xaaaa, 0xbbbb, 0xcccc, 0xdddd, 0xeeee,
    ];
    let base = u64::try_from(words.as_mut_ptr().addr()).unwrap();
    let destination = base + 4 * 2;

    assert_eq!(guest_wmemcpy(destination, base, 8), destination);
    assert_eq!(
        words,
        [
            0x41, 0x42, 0x43, 0x44, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0xeeee,
        ]
    );

    assert_eq!(guest_wmemmove(base, destination, 4), base);
    assert_eq!(&words[..4], &[0x41, 0x42, 0x43, 0x44]);
}

#[test]
fn searches_and_compares_gen5_wide_characters() {
    let left = [0x0061_u16, 0x0062, 0x20ac, 0x0064];
    let right = [0x0061_u16, 0x0062, 0x20ad, 0x0064];
    let left_address = u64::try_from(left.as_ptr().addr()).unwrap();
    let right_address = u64::try_from(right.as_ptr().addr()).unwrap();

    assert_eq!(guest_wmemchr(left_address, 0x20ac, 4), left_address + 4);
    assert_eq!(guest_wmemchr(left_address, 0x9999, 4), 0);
    assert_eq!(guest_wmemcmp(left_address, left_address, 4), 0);
    assert_eq!(guest_wmemcmp(left_address, right_address, 4), u64::MAX);
    assert_eq!(guest_wmemcmp(right_address, left_address, 4), 1);
}

#[test]
fn exposes_a_stable_dinkumware_global_locale() {
    let mut state = ProbeDispatchState::default();
    let locale = guest_global_locale(&mut state);
    let vtable = read_guest_u64(locale);
    let retain = read_guest_u64(vtable + 0x10);
    // SAFETY: guest_global_locale installs this exact process-local C ABI function pointer.
    let retain: extern "C" fn(u64) -> u64 = unsafe {
        std::mem::transmute(usize::try_from(retain).expect("function pointer fits usize"))
    };

    assert_ne!(locale, 0);
    assert_ne!(vtable, 0);
    assert_eq!(retain(locale), locale);
    assert_eq!(guest_global_locale(&mut state), locale);

    guest_free(state.locale_object);
    guest_free(state.locale_vtable);
}

#[test]
fn resolves_duplicate_console_path_separators() {
    assert_eq!(
        resolve_guest_path(Path::new("/games/title"), "/app0//shaders/presenter.vert"),
        Some(Path::new("/games/title/shaders/presenter.vert").to_owned())
    );
}

#[test]
fn immutable_app_path_cache_reuses_borrowed_keys_without_growing_on_replacement() {
    let mut cache = super::ImmutableAppPathCache::default();
    let root = Path::new("/games/title");
    cache.insert(
        root,
        "Content/Paks/game.pak",
        root.join("content/paks/game.pak"),
    );

    assert_eq!(
        cache.get(root, "Content/Paks/game.pak"),
        Some(root.join("content/paks/game.pak"))
    );
    assert_eq!(cache.entries, 1);

    cache.insert(
        root,
        "Content/Paks/game.pak",
        root.join("CONTENT/PAKS/game.pak"),
    );
    assert_eq!(
        cache.get(root, "Content/Paks/game.pak"),
        Some(root.join("CONTENT/PAKS/game.pak"))
    );
    assert_eq!(cache.entries, 1);
}

/// Engines address content relative to their binaries directory, so parent components inside
/// the game root must resolve rather than be refused.
/// Dumps are frequently stored with normalized names while titles request the original
/// casing, and console lookups do not distinguish case.
/// A merged shader's fused stage keeps the back half's register block, so the reported
/// scratch size must cover exactly those registers; mismatched halves are refused.
#[test]
fn fused_shader_size_covers_the_back_half_registers() {
    let mut front = [0_u8; 0x60];
    let mut back = [0_u8; 0x60];
    front[0x5a] = 4; // GsFront
    back[0x5a] = 6; // GsBack
    back[0x5c] = 7; // seven shader registers
    let mut output = [0_u64; 2];
    assert_eq!(
        agc_get_fused_shader_size([
            output.as_mut_ptr().addr() as u64,
            front.as_ptr().addr() as u64,
            back.as_ptr().addr() as u64,
            0,
            0,
            0,
        ]),
        0
    );
    assert_eq!(output, [7 * 8, 4]);

    // A tessellation back half does not fuse with a geometry front half.
    back[0x5a] = 7; // HsBack
    assert_ne!(
        agc_get_fused_shader_size([
            output.as_mut_ptr().addr() as u64,
            front.as_ptr().addr() as u64,
            back.as_ptr().addr() as u64,
            0,
            0,
            0,
        ]),
        0
    );
}

#[test]
fn create_shader_registers_the_authoritative_agc_code_extent() {
    let mut header = [0_u8; 0x60];
    let mut shader_registers = [0x20c_u32, 0, 0x20d, 0];
    let code = [0xbf81_0000_u32; 0x1001];
    let mut destination = 0_u64;
    let header_address = header.as_mut_ptr().addr() as u64;
    let registers_address = shader_registers.as_mut_ptr().addr() as u64;
    let code_address = code.as_ptr().addr() as u64;
    write_guest_u32(header_address, super::AGC_SHADER_FILE_HEADER);
    write_guest_u32(header_address + 4, super::AGC_SHADER_VERSION);
    write_guest_u64(
        header_address + super::AGC_SHADER_SH_REGISTERS_OFFSET,
        registers_address.wrapping_sub(header_address + super::AGC_SHADER_SH_REGISTERS_OFFSET),
    );
    write_guest_u32(
        header_address + super::AGC_SHADER_SIZE_OFFSET,
        u32::try_from(std::mem::size_of_val(&code)).expect("test shader size fits u32"),
    );
    write_guest_u8(header_address + super::AGC_SHADER_TYPE_OFFSET, 0);
    write_guest_u8(
        header_address + super::AGC_SHADER_SH_REGISTER_COUNT_OFFSET,
        2,
    );
    let mut state = ProbeDispatchState::default();

    assert_eq!(
        super::agc_create_shader(
            [
                (&raw mut destination).addr() as u64,
                header_address,
                code_address,
                0,
                0,
                0,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(destination, header_address);
    assert_eq!(
        state.agc_shader_registry.code_sizes.get(&code_address),
        Some(&u32::try_from(std::mem::size_of_val(&code)).expect("test shader size fits u32"))
    );
    assert_eq!(shader_registers[1], low_u32(code_address >> 8));
    assert_eq!(shader_registers[3], low_u32(code_address >> 40));
}

#[test]
fn interpolant_mapping_matches_semantics_and_keeps_identity_tail() {
    let mut geometry = [0_u8; 0x60];
    let mut pixel = [0_u8; 0x60];
    let geometry_semantics = [7 | (5 << 8)];
    let pixel_semantics = [7 | (1 << 22) | (2 << 28)];
    let mut registers = [0_u32; 64];
    let geometry_address = geometry.as_mut_ptr().addr() as u64;
    let pixel_address = pixel.as_mut_ptr().addr() as u64;
    write_guest_u64(
        geometry_address + AGC_SHADER_OUTPUT_SEMANTICS_OFFSET,
        geometry_semantics.as_ptr().addr() as u64,
    );
    write_guest_u16(
        geometry_address + AGC_SHADER_OUTPUT_SEMANTICS_COUNT_OFFSET,
        1,
    );
    // This adjacent packed field must not become part of the 16-bit output count.
    write_guest_u16(
        geometry_address + AGC_SHADER_OUTPUT_SEMANTICS_COUNT_OFFSET + 2,
        u16::MAX,
    );
    write_guest_u64(
        pixel_address + AGC_SHADER_INPUT_SEMANTICS_OFFSET,
        pixel_semantics.as_ptr().addr() as u64,
    );
    write_guest_u32(pixel_address + AGC_SHADER_INPUT_SEMANTICS_COUNT_OFFSET, 1);

    assert_eq!(
        agc_create_interpolant_mapping([
            registers.as_mut_ptr().addr() as u64,
            geometry_address,
            pixel_address,
            0,
            0,
            0,
        ]),
        0
    );
    assert_eq!(registers[0], AGC_SPI_PS_INPUT_CNTL_0);
    assert_eq!(registers[1], 0x605);
    assert_eq!(registers[2], AGC_SPI_PS_INPUT_CNTL_0 + 1);
    assert_eq!(registers[3], 1);
    assert_eq!(registers[62], AGC_SPI_PS_INPUT_CNTL_0 + 31);
    assert_eq!(registers[63], 31);

    registers.fill(u32::MAX);
    assert_eq!(
        agc_create_interpolant_mapping([registers.as_mut_ptr().addr() as u64, 0, 0, 0, 0, 0,]),
        0
    );
    assert_eq!(registers[0], AGC_SPI_PS_INPUT_CNTL_0);
    assert_eq!(registers[1], 0);
    assert_eq!(registers[63], 31);
}

#[test]
fn fused_shader_preserves_header_data_and_high_register_flags() {
    let mut front = [0_u8; SHADER_HEADER_SIZE as usize];
    let mut back = [0_u8; SHADER_HEADER_SIZE as usize];
    let mut fused = [0_u8; SHADER_HEADER_SIZE as usize];
    let mut back_registers = [
        SPI_SHADER_PGM_LO_ES,
        0,
        SPI_SHADER_PGM_LO_ES + 1,
        0xa5a5_5a00,
    ];
    let mut scratch = [0_u32; 4];
    let front_address = front.as_mut_ptr().addr() as u64;
    let back_address = back.as_mut_ptr().addr() as u64;
    let code = 0x00ab_cdef_1234_5600_u64;
    let user_data = 0x1234_5678_9abc_def0_u64;
    write_guest_u8(
        front_address + SHADER_HEADER_TYPE_OFFSET,
        SHADER_BINARY_TYPE_GS_FRONT,
    );
    write_guest_u8(
        back_address + SHADER_HEADER_TYPE_OFFSET,
        SHADER_BINARY_TYPE_GS_BACK,
    );
    write_guest_u8(back_address + SHADER_HEADER_NUM_SH_REGISTERS_OFFSET, 2);
    write_guest_u64(front_address + SHADER_HEADER_CODE_OFFSET, code);
    write_guest_u64(back_address + 8, user_data);
    write_guest_u64(
        back_address + SHADER_HEADER_SH_REGISTERS_OFFSET,
        back_registers.as_mut_ptr().addr() as u64,
    );

    assert_eq!(
        agc_fuse_shader_halves([
            fused.as_mut_ptr().addr() as u64,
            front_address,
            back_address,
            scratch.as_mut_ptr().addr() as u64,
            0,
            0,
        ]),
        0
    );
    let fused_address = fused.as_ptr().addr() as u64;
    assert_eq!(read_guest_u64(fused_address + 8), user_data);
    assert_eq!(
        read_guest_u64(fused_address + SHADER_HEADER_SH_REGISTERS_OFFSET),
        scratch.as_ptr().addr() as u64
    );
    assert_eq!(scratch[1], low_u32(code >> 8));
    assert_eq!(scratch[3], 0xa5a5_5a00 | (low_u32(code >> 40) & 0xff));
}

#[test]
fn resolves_guest_paths_that_differ_only_by_case() {
    let root = std::env::temp_dir().join(format!("remu-case-{}", std::process::id()));
    std::fs::create_dir_all(root.join("prx")).unwrap();
    std::fs::write(root.join("prx/akroomverb.prx"), b"module").unwrap();
    // The resolved path must name a file that exists. On a case-insensitive host the request
    // already resolves; on a case-sensitive one only the fallback finds the stored name.
    let resolved = resolve_guest_path(&root, "/app0/prx/AkRoomVerb.prx").unwrap();
    assert!(resolved.is_file(), "unresolved: {}", resolved.display());
    // A genuine miss still reports the name the guest asked for.
    assert_eq!(
        resolve_guest_path(&root, "/app0/prx/Missing.prx"),
        Some(root.join("prx/Missing.prx"))
    );
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn av_player_resolves_relative_file_urls_inside_app0_case_insensitively() {
    let root = std::env::temp_dir().join(format!("remu-avplayer-path-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let movie = root.join("shpc/content/movies/ui_titlemovie01.mp4");
    std::fs::create_dir_all(movie.parent().unwrap()).unwrap();
    std::fs::write(&movie, b"movie").unwrap();

    let resolved = super::av_player_resolve_source(
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
        let resolved = super::av_player_resolve_source(&root, source)
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
        super::av_player_resolve_source(&root, "file://../../../outside.mp4"),
        None
    );
    assert_eq!(
        super::av_player_resolve_source(&root, "../../../outside.mp4"),
        None
    );

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn discovers_runtime_modules_in_media_and_app0_prx_layouts() {
    let root = std::env::temp_dir().join(format!("remu-runtime-modules-{}", std::process::id()));
    std::fs::remove_dir_all(&root).ok();
    let media_plugins = root.join("decrypted/Media/Plugins/Runtime");
    let app_prx = root.join("prx");
    std::fs::create_dir_all(&media_plugins).unwrap();
    std::fs::create_dir_all(&app_prx).unwrap();
    std::fs::write(media_plugins.join("media_plugin.sprx"), b"module").unwrap();
    std::fs::write(app_prx.join("akroomverb.prx"), b"module").unwrap();
    std::fs::write(app_prx.join("akroomverb.prx.esbak"), b"backup").unwrap();

    let modules = discover_runtime_module_paths(&root).unwrap();
    assert_eq!(modules.len(), 2);
    assert!(
        modules
            .iter()
            .any(|path| path.ends_with("media_plugin.sprx"))
    );
    assert!(modules.iter().any(|path| path.ends_with("akroomverb.prx")));

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn resolves_parent_components_inside_the_game_root() {
    assert_eq!(
        resolve_guest_path(
            Path::new("/games/title"),
            "/app0/engine/binaries/ps5/../../../shpc/config/binaryconfig.ini",
        ),
        Some(Path::new("/games/title/shpc/config/binaryconfig.ini").to_owned())
    );
    assert_eq!(
        resolve_guest_path(Path::new("/games/title"), "/app0/shpc/./content/paks"),
        Some(Path::new("/games/title/shpc/content/paks").to_owned())
    );
}

#[test]
fn rejects_game_root_parent_traversal() {
    assert_eq!(
        resolve_guest_path(Path::new("/games/title"), "/app0/../outside.bin"),
        None
    );
}

#[test]
fn builds_the_gen5_kernel_stat_layout() {
    let payload = kernel_stat_payload(1_025, false, b"/app0/data.bin");

    assert_eq!(
        u16::from_le_bytes(payload[8..10].try_into().unwrap()),
        0x81ff
    );
    assert_eq!(u16::from_le_bytes(payload[10..12].try_into().unwrap()), 1);
    assert_eq!(
        i64::from_le_bytes(payload[72..80].try_into().unwrap()),
        1_025
    );
    assert_eq!(i64::from_le_bytes(payload[80..88].try_into().unwrap()), 3);
    assert_eq!(u32::from_le_bytes(payload[88..92].try_into().unwrap()), 512);
}

#[test]
fn sign_extends_gen5_kernel_errors() {
    assert_eq!(sce_kernel_error(0x8002_0002), 0xffff_ffff_8002_0002);
}

#[test]
fn translates_console_clock_ids_to_realtime() {
    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let mut output = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    assert_eq!(guest_clock_gettime(4, (&raw mut output).addr() as u64), 0);
    let seconds = u64::try_from(output.tv_sec).unwrap();
    assert!(seconds >= before);
    assert!(seconds <= before + 2);
    assert!((0..1_000_000_000).contains(&output.tv_nsec));
}

#[test]
fn pthread_once_runs_guest_initializer_exactly_once() {
    ONCE_CALLBACKS.store(0, Ordering::Relaxed);
    let mut control = 0_u32;
    let control_address = (&raw mut control).addr() as u64;
    let routine_address = once_test_callback as *const () as usize as u64;
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(move || {
                assert_eq!(guest_pthread_once(control_address, routine_address), 0);
            });
        }
    });
    assert_eq!(control, 1);
    assert_eq!(ONCE_CALLBACKS.load(Ordering::Relaxed), 1);
}

#[test]
fn fills_gen5_calendar_time_and_round_trips_timezone_conversions() {
    let before = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let mut timeval = [0_u64; 2];
    assert_eq!(
        guest_kernel_gettimeofday((&raw mut timeval).addr() as u64),
        0
    );
    assert!((before..=before + 2).contains(&timeval[0]));
    assert!(timeval[1] < 1_000_000);

    let utc = 1_700_000_000_i64;
    let mut local = 0_i64;
    let mut timesec = [0_u8; 16];
    let mut dst = 0_u64;
    assert_eq!(
        guest_kernel_convert_utc_to_localtime([
            u64::from_ne_bytes(utc.to_ne_bytes()),
            (&raw mut local).addr() as u64,
            (&raw mut timesec).addr() as u64,
            (&raw mut dst).addr() as u64,
            0,
            0,
        ]),
        0
    );
    assert_eq!(i64::from_le_bytes(timesec[..8].try_into().unwrap()), utc);
    assert_eq!(
        u64::from(u32::from_le_bytes(timesec[12..16].try_into().unwrap())),
        dst
    );

    let mut round_trip = 0_i64;
    let mut timezone = [0_i32; 2];
    let mut dst_seconds = 0_i32;
    assert_eq!(
        guest_kernel_convert_localtime_to_utc([
            u64::from_ne_bytes(local.to_ne_bytes()),
            0,
            (&raw mut round_trip).addr() as u64,
            (&raw mut timezone).addr() as u64,
            (&raw mut dst_seconds).addr() as u64,
            0,
        ]),
        0
    );
    assert_eq!(round_trip, utc);
    assert_eq!(dst_seconds, i32::try_from(dst).unwrap());
    assert_eq!(timezone[1], dst_seconds / 60);
}

#[test]
fn builds_distinct_dinkumware_case_mapping_tables() {
    let mut state = ProbeDispatchState::default();
    let lower = guest_tolower_table(&mut state);
    let upper = guest_toupper_table(&mut state);

    assert_ne!(lower, 0);
    assert_ne!(upper, 0);
    assert_ne!(lower, upper);
    assert_eq!(read_guest_u16(lower + u64::from(b'A') * 2), u16::from(b'a'));
    assert_eq!(read_guest_u16(lower + u64::from(b'a') * 2), u16::from(b'a'));
    assert_eq!(read_guest_u16(upper + u64::from(b'a') * 2), u16::from(b'A'));
    assert_eq!(read_guest_u16(upper + u64::from(b'A') * 2), u16::from(b'A'));
    assert_eq!(read_guest_u16(lower - 2), u16::MAX);
    assert_eq!(read_guest_u16(upper - 2), u16::MAX);

    guest_free(lower - 128 * 2);
    guest_free(upper - 128 * 2);
}

#[test]
fn returns_stable_integer_system_parameters() {
    let mut value = 0_i32;
    let output = (&raw mut value).addr() as u64;

    assert_eq!(system_service_param_get_int([1, output, 0, 0, 0, 0]), 0);
    assert_eq!(value, 1);
    assert_eq!(system_service_param_get_int([4, output, 0, 0, 0, 0]), 0);
    assert_eq!(value, 180);
    assert_eq!(
        system_service_param_get_int([1, 0, 0, 0, 0, 0]),
        sce_kernel_error(0x80a1_0003)
    );
}

#[test]
fn fresh_boot_does_not_skip_the_platform_notice_screen() {
    let mut skip = 0xff_u8;
    let mut state = ProbeDispatchState::default();

    assert_eq!(
        dispatch_probe_import(
            "3RQ5aQfnstU",
            [(&raw mut skip).addr() as u64, 0, 0, 0, 0, 0],
            [0; 6],
            &mut state,
        ),
        0
    );
    assert_eq!(skip, 0);
    assert!(state.unimplemented_nids.is_empty());
}

#[test]
fn pad_get_handle_observes_the_primary_controller_lifecycle() {
    let mut state = ProbeDispatchState::default();
    let primary = [0x1000_0000, 0, 0, 0, 0, 0];

    assert_eq!(
        dispatch_probe_import("u1GRHp+oWoY", primary, [0; 6], &mut state),
        sce_kernel_error(super::SCE_PAD_ERROR_NOT_INITIALIZED)
    );
    assert_eq!(
        dispatch_probe_import("hv1luiJrqQM", [0; 6], [0; 6], &mut state),
        0
    );
    assert_eq!(
        dispatch_probe_import("u1GRHp+oWoY", primary, [0; 6], &mut state),
        sce_kernel_error(super::SCE_PAD_ERROR_DEVICE_NO_HANDLE)
    );
    assert_eq!(
        dispatch_probe_import("xk0AcarP3V4", primary, [0; 6], &mut state),
        1
    );
    assert_eq!(
        dispatch_probe_import("u1GRHp+oWoY", primary, [0; 6], &mut state),
        1
    );
    assert_eq!(
        dispatch_probe_import("xk0AcarP3V4", primary, [0; 6], &mut state),
        sce_kernel_error(super::SCE_PAD_ERROR_ALREADY_OPENED)
    );
    assert_eq!(
        dispatch_probe_import("6ncge5+l5Qs", [1, 0, 0, 0, 0, 0], [0; 6], &mut state),
        0
    );
    assert_eq!(
        dispatch_probe_import("u1GRHp+oWoY", primary, [0; 6], &mut state),
        sce_kernel_error(super::SCE_PAD_ERROR_DEVICE_NO_HANDLE)
    );
    assert_eq!(
        dispatch_probe_import("xk0AcarP3V4", primary, [0; 6], &mut state),
        2
    );
    assert_eq!(
        dispatch_probe_import("u1GRHp+oWoY", primary, [0; 6], &mut state),
        2
    );
    assert_eq!(
        dispatch_probe_import("6ncge5+l5Qs", [1, 0, 0, 0, 0, 0], [0; 6], &mut state),
        sce_kernel_error(super::SCE_PAD_ERROR_INVALID_HANDLE)
    );
    assert_eq!(
        dispatch_probe_import("u1GRHp+oWoY", primary, [0; 6], &mut state),
        2,
        "rejecting a stale handle must not close the new controller"
    );
    assert_eq!(
        super::known_nid_name("u1GRHp+oWoY"),
        Some("scePadGetHandle")
    );
    assert!(state.unimplemented_nids.is_empty());
}

#[test]
fn save_data_dialog_initialization_requires_and_owns_common_dialog_state() {
    let mut state = ProbeDispatchState::default();
    assert_eq!(
        dispatch_probe_import("s9e3+YpRnzw", [0; 6], [0; 6], &mut state),
        sce_kernel_error(super::COMMON_DIALOG_ERROR_NOT_SYSTEM_INITIALIZED)
    );
    assert_eq!(
        dispatch_probe_import("uoUpLGNkygk", [0; 6], [0; 6], &mut state),
        0
    );
    assert_eq!(
        dispatch_probe_import("s9e3+YpRnzw", [0; 6], [0; 6], &mut state),
        0
    );
    assert_eq!(
        dispatch_probe_import("ERKzksauAJA", [0; 6], [0; 6], &mut state),
        u64::from(super::COMMON_DIALOG_STATUS_INITIALIZED)
    );
    assert_eq!(
        dispatch_probe_import("s9e3+YpRnzw", [0; 6], [0; 6], &mut state),
        sce_kernel_error(super::COMMON_DIALOG_ERROR_DIALOG_ALREADY_INITIALIZED)
    );
    assert_eq!(
        dispatch_probe_import("lDqxaY1UbEo", [0; 6], [0; 6], &mut state),
        sce_kernel_error(super::COMMON_DIALOG_ERROR_BUSY),
        "save-data dialog must own the shared common-dialog service"
    );
    assert_eq!(
        super::known_nid_name("s9e3+YpRnzw"),
        Some("sceSaveDataDialogInitialize")
    );
    assert!(state.unimplemented_nids.is_empty());

    let mut busy = ProbeDispatchState::default();
    assert_eq!(
        dispatch_probe_import("uoUpLGNkygk", [0; 6], [0; 6], &mut busy),
        0
    );
    assert_eq!(
        dispatch_probe_import("lDqxaY1UbEo", [0; 6], [0; 6], &mut busy),
        0
    );
    assert_eq!(
        dispatch_probe_import("s9e3+YpRnzw", [0; 6], [0; 6], &mut busy),
        sce_kernel_error(super::COMMON_DIALOG_ERROR_BUSY)
    );
    assert_eq!(
        dispatch_probe_import("ePw-kqZmelo", [0; 6], [0; 6], &mut busy),
        0
    );
    assert_eq!(
        dispatch_probe_import("s9e3+YpRnzw", [0; 6], [0; 6], &mut busy),
        0,
        "terminating the message dialog must release common-dialog ownership"
    );
}

#[test]
fn share_and_np_title_services_validate_and_retain_caller_data() {
    let mut state = ProbeDispatchState::default();
    assert_eq!(
        dispatch_probe_import("7QZtURYnXG4", [0; 6], [0; 6], &mut state),
        sce_kernel_error(super::SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
    assert_eq!(
        dispatch_probe_import("Ec63y59l9tw", [0; 6], [0; 6], &mut state),
        sce_kernel_error(super::SCE_NP_ERROR_INVALID_ARGUMENT)
    );

    let content = std::ffi::CString::new("silent-hill-capture").expect("valid C string");
    assert_eq!(
        dispatch_probe_import(
            "7QZtURYnXG4",
            [content.as_ptr() as u64, 0, 0, 0, 0, 0],
            [0; 6],
            &mut state
        ),
        0
    );
    assert_eq!(
        state.services.share_content_parameter.as_deref(),
        Some("silent-hill-capture")
    );
    assert_eq!(
        dispatch_probe_import("YBiIdcDPrxs", [0; 6], [0; 6], &mut state),
        0
    );
    assert!(state.services.share_feature_permitted);

    let mut title_id = *b"PPSA10112_00\0\0\0\0";
    let title_secret = [0x5a_u8; 128];
    assert_eq!(
        dispatch_probe_import("7QZtURYnXG4", [1, 0, 0, 0, 0, 0], [0; 6], &mut state),
        sce_kernel_error(super::SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
    assert_eq!(
        dispatch_probe_import(
            "Ec63y59l9tw",
            [1, title_secret.as_ptr() as u64, 0, 0, 0, 0],
            [0; 6],
            &mut state
        ),
        sce_kernel_error(super::SCE_NP_ERROR_INVALID_ARGUMENT)
    );
    assert_eq!(
        dispatch_probe_import(
            "Ec63y59l9tw",
            [
                title_id.as_mut_ptr() as u64,
                title_secret.as_ptr() as u64,
                0,
                0,
                0,
                0
            ],
            [0; 6],
            &mut state
        ),
        0
    );
    assert_eq!(state.services.np_title_id, Some(title_id));
    assert_eq!(
        super::known_nid_name("YBiIdcDPrxs"),
        Some("sceShareFeaturePermit")
    );
    assert_eq!(
        super::known_nid_name("7QZtURYnXG4"),
        Some("sceShareSetContentParam")
    );
    assert_eq!(
        super::known_nid_name("Ec63y59l9tw"),
        Some("sceNpSetNpTitleId")
    );
    assert!(state.unimplemented_nids.is_empty());
}

#[test]
fn address_sanitizer_query_reports_the_uninstrumented_guest_runtime() {
    let mut state = ProbeDispatchState::default();
    assert_eq!(
        dispatch_probe_import("jh+8XiK4LeE", [0; 6], [0; 6], &mut state),
        0
    );
    assert_eq!(
        super::known_nid_name("jh+8XiK4LeE"),
        Some("sceKernelIsAddressSanitizerEnabled")
    );
    assert!(state.unimplemented_nids.is_empty());
}

#[test]
fn unknown_imports_return_the_gen5_not_implemented_error() {
    let mut state = ProbeDispatchState::default();
    assert_eq!(
        dispatch_probe_import("not-a-real-nid", [0; 6], [0; 6], &mut state),
        sce_kernel_error(SCE_KERNEL_ERROR_NOT_IMPLEMENTED)
    );
    assert_eq!(state.unimplemented_nids.len(), 1);
}

#[test]
fn ajm_batch_initialize_writes_only_the_five_qword_info_object() {
    let batch = [0x5a_u8; 0x708];
    let original_batch = batch;
    let mut info_storage = [0xa5_u8; 0x48];
    let info_address = info_storage[0x10..0x38].as_mut_ptr() as u64;

    assert_eq!(
        ajm_batch_initialize([
            batch.as_ptr() as u64,
            batch.len() as u64,
            info_address,
            0,
            0,
            0,
        ]),
        0
    );

    assert_eq!(
        batch, original_batch,
        "the caller-owned batch arena is preserved"
    );
    assert_eq!(&info_storage[..0x10], &[0xa5; 0x10]);
    assert_eq!(&info_storage[0x38..], &[0xa5; 0x10]);
    // SAFETY: info_address points at the initialized 0x28-byte subrange above.
    let info = unsafe { std::ptr::read_unaligned(info_address as *const SceAjmBatchInfo) };
    assert_eq!(
        info,
        SceAjmBatchInfo {
            buffer: batch.as_ptr() as u64,
            offset: 0,
            size: batch.len() as u64,
            last_good_job: 0,
            last_good_job_return_address: 0,
        }
    );
}

#[test]
fn ajm_batch_initialize_requires_the_ps5_info_output() {
    let batch = [0x5a_u8; 0x708];
    assert_eq!(
        ajm_batch_initialize([batch.as_ptr() as u64, batch.len() as u64, 0, 0, 0, 0,]),
        sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER)
    );
    assert!(batch.iter().all(|byte| *byte == 0x5a));
}

/// Gen5 tags the reserved argument with a library revision; only the low dword must be zero.
#[test]
fn ajm_initialize_accepts_a_revision_tagged_reserved_argument() {
    let mut state = ProbeDispatchState::default();
    let mut context = 0_u32;
    assert_eq!(
        ajm_initialize(
            [0x3_0000_0000, (&raw mut context).addr() as u64, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
    assert_ne!(context, 0);
    // A non-zero low dword is still a malformed reserved value.
    assert_ne!(
        ajm_initialize(
            [0x3_0000_0001, (&raw mut context).addr() as u64, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
}

#[test]
fn ajm_initialize_allocates_stable_nonzero_context_ids() {
    let mut state = ProbeDispatchState::default();
    let mut first = 0_u32;
    let mut second = 0_u32;
    assert_eq!(
        dispatch_probe_import(
            "dl+4eHSzUu4",
            [0, (&raw mut first).addr() as u64, 0, 0, 0, 0],
            [0; 6],
            &mut state,
        ),
        0
    );
    assert_eq!(
        dispatch_probe_import(
            "dl+4eHSzUu4",
            [0, (&raw mut second).addr() as u64, 0, 0, 0, 0],
            [0; 6],
            &mut state,
        ),
        0
    );
    assert_eq!((first, second), (1, 2));
    assert!(state.ajm_contexts.contains_key(&first));
    assert!(state.ajm_contexts.contains_key(&second));
    assert_eq!(
        super::known_nid_name("dl+4eHSzUu4"),
        Some("sceAjmInitialize")
    );
    assert!(state.unimplemented_nids.is_empty());
}

#[test]
fn np_web_api2_initialize_allocates_bounded_library_contexts() {
    let first = super::dispatch_stateless_import("+o9816YQhqQ", [1, 0x8_0000, 0, 0, 0, 0], [0; 6])
        .expect("NP Web API2 initialize is implemented");
    let second = super::dispatch_stateless_import("+o9816YQhqQ", [1, 0x8_0000, 0, 0, 0, 0], [0; 6])
        .expect("second NP Web API2 context");

    assert!((1..0x8000).contains(&first));
    assert_eq!(second, first + 1);
    assert_eq!(
        super::known_nid_name("+o9816YQhqQ"),
        Some("sceNpWebApi2Initialize")
    );
}

#[test]
fn ajm_lifecycle_registers_codec_and_instances() {
    let mut state = ProbeDispatchState::default();
    let mut context = 0_u32;
    let mut instance = 0_u32;
    let context_output = (&raw mut context).addr() as u64;
    let instance_output = (&raw mut instance).addr() as u64;

    assert_eq!(
        ajm_initialize([0, context_output, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        ajm_module_register([u64::from(context), 1, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        ajm_instance_create(
            [u64::from(context), 1, 1, instance_output, 0, 0],
            &mut state,
        ),
        0
    );
    assert_eq!(instance, 0x4001);
    assert_eq!(
        ajm_instance_destroy(
            [u64::from(context), u64::from(instance), 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
    assert_eq!(
        ajm_module_unregister([u64::from(context), 1, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        ajm_finalize([u64::from(context), 0, 0, 0, 0, 0], &mut state),
        0
    );
    assert!(state.ajm_contexts.is_empty());
}

#[test]
fn import_stubs_preserve_guest_rax_for_frame_capture() {
    let mut slot = [0_u8; super::IMPORT_STUB_SIZE_USIZE];
    write_import_stub(&mut slot, 0x1234, 0x1122_3344_5566_7788).unwrap();

    assert_eq!(slot[0], 0x68);
    assert_eq!(i32::from_le_bytes(slot[1..5].try_into().unwrap()), 0x1234);
    assert_eq!(&slot[5..7], &[0x49, 0xbb]);
    assert_eq!(
        u64::from_le_bytes(slot[7..15].try_into().unwrap()),
        0x1122_3344_5566_7788
    );
    assert_eq!(&slot[15..18], &[0x41, 0xff, 0xe3]);
    assert!(slot[18..].iter().all(|byte| *byte == 0));
}

#[test]
fn import_stack_arguments_are_read_from_the_captured_guest_frame() {
    let stack = [
        0xfeed_face_cafe_beef_u64,
        0x11,
        0x22,
        0x33,
        0x44,
        0x55,
        0x66,
    ];
    let frame = super::ImportCallFrame {
        guest_rsp: stack.as_ptr().addr() as u64,
        ..super::ImportCallFrame::default()
    };

    assert_eq!(
        super::read_import_stack_arguments(&frame),
        [0x11, 0x22, 0x33, 0x44, 0x55, 0x66]
    );
}

#[cfg(all(unix, target_arch = "x86_64"))]
#[test]
fn import_trampoline_captures_the_complete_sysv_register_frame() {
    const PAGE_SIZE: usize = 0x4000;
    let mut slot = [0_u8; super::IMPORT_STUB_SIZE_USIZE];
    write_import_stub(
        &mut slot,
        usize::try_from(super::TEST_CAPTURE_DISPATCH_INDEX).unwrap(),
        super::import_dispatch_trampoline as *const () as usize as u64,
    )
    .unwrap();
    // SAFETY: This test owns one anonymous page and changes it from writable to executable
    // before calling the complete import stub copied into it.
    let mapping = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            PAGE_SIZE,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    assert_ne!(mapping, libc::MAP_FAILED);
    // SAFETY: mapping owns PAGE_SIZE writable bytes and slot fits at its start.
    unsafe { std::ptr::copy_nonoverlapping(slot.as_ptr(), mapping.cast::<u8>(), slot.len()) };
    // SAFETY: mapping and its exact size came from mmap above.
    assert_eq!(
        unsafe { libc::mprotect(mapping, PAGE_SIZE, libc::PROT_READ | libc::PROT_EXEC) },
        0
    );
    *super::TEST_CAPTURED_IMPORT_FRAME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;

    // SAFETY: the page now contains a complete executable import stub targeting the real
    // trampoline, and invoke_test_import_stub establishes the guest SysV register state.
    let result = unsafe { super::invoke_test_import_stub(mapping.addr() as u64) };
    let frame = super::TEST_CAPTURED_IMPORT_FRAME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
        .expect("test dispatch captured its frame");
    // SAFETY: mapping and its exact size came from mmap and are no longer in use.
    assert_eq!(unsafe { libc::munmap(mapping, PAGE_SIZE) }, 0);

    assert_eq!(
        frame.integer_arguments,
        [0x11, 0x22, 0x33, 0x44, 0x55, 0x66]
    );
    assert_eq!(frame.rax, 0x0102_0304_0506_0708);
    assert_eq!(frame.dispatch_index, super::TEST_CAPTURE_DISPATCH_INDEX);
    assert_ne!(frame.guest_return_address, 0);
    assert_eq!(frame.guest_rsp & 0xf, 8);
    for (index, xmm) in frame.xmm_arguments.iter().enumerate() {
        let bytes = xmm.to_le_bytes();
        let low = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let high = u64::from_le_bytes(bytes[8..].try_into().unwrap());
        assert_eq!(low, u64::try_from(index + 1).unwrap() * 0x101);
        assert_eq!(high, 0);
    }
    assert_eq!(result.integer, frame.rax);
    assert_eq!(result.float.to_bits(), 0x808);
}

#[test]
fn pending_guest_exception_runs_on_target_import_with_orbis_stack_context() {
    let _guard = GUEST_EXCEPTION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    const THREAD: u64 = 0x1234;
    const GUEST_RSP: u64 = 0x7000_1234_5678;
    DELIVERED_EXCEPTION.store(0, Ordering::SeqCst);
    DELIVERED_EXCEPTION_RSP.store(0, Ordering::SeqCst);
    super::set_current_guest_thread_handle(THREAD);
    let handler = capture_guest_exception as *const () as usize as u64;
    let mut delivery = super::guest_exception_delivery()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *delivery = super::GuestExceptionDeliveryState::default();
    delivery.pending.insert(
        THREAD,
        super::PendingGuestException {
            handler,
            exception: 30,
        },
    );
    drop(delivery);

    super::deliver_pending_guest_exception_at_import(&super::ImportCallFrame {
        guest_rsp: GUEST_RSP,
        guest_return_address: 0x8000_1234,
        ..super::ImportCallFrame::default()
    });

    assert_eq!(DELIVERED_EXCEPTION.load(Ordering::SeqCst), 30);
    assert_eq!(
        DELIVERED_EXCEPTION_RSP.load(Ordering::SeqCst),
        usize::try_from(GUEST_RSP).unwrap()
    );
    let delivery = super::guest_exception_delivery()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert!(delivery.pending.is_empty());
    assert!(delivery.active.is_empty());
    drop(delivery);
    super::set_current_guest_thread_handle(0);
}

#[test]
fn pending_guest_exception_interrupts_event_flag_and_semaphore_waits() {
    let _guard = GUEST_EXCEPTION_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    const THREAD: u64 = 0x5678;
    const EVENT_FLAG: u64 = 0x7fff_0000_4000;
    const SEMAPHORE: u32 = 0x7fff_4000;
    super::set_current_guest_thread_handle(THREAD);
    super::guest_sync()
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .event_flags
        .insert(
            EVENT_FLAG,
            super::GuestEventFlagState {
                name: "exception-test".to_owned(),
                attributes: 0,
                bits: 0,
                waiters: 0,
                cancellation_generation: 0,
            },
        );
    super::guest_sync()
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .semaphores
        .insert(
            SEMAPHORE,
            super::GuestSemaphoreState {
                count: 0,
                maximum: 1,
            },
        );
    super::guest_exception_delivery()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .pending
        .insert(
            THREAD,
            super::PendingGuestException {
                handler: 1,
                exception: 30,
            },
        );

    assert_eq!(
        super::guest_event_flag_wait([EVENT_FLAG, 1, super::EVENT_FLAG_WAIT_OR as u64, 0, 0, 0]),
        sce_kernel_error(super::SCE_KERNEL_ERROR_INTERRUPTED)
    );
    assert_eq!(
        super::guest_semaphore_wait([u64::from(SEMAPHORE), 1, 0, 0, 0, 0]),
        sce_kernel_error(super::SCE_KERNEL_ERROR_INTERRUPTED)
    );

    let sync = super::guest_sync();
    let mut state = sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(state.event_flags[&EVENT_FLAG].waiters, 0);
    state.event_flags.remove(&EVENT_FLAG);
    state.semaphores.remove(&SEMAPHORE);
    drop(state);
    super::guest_exception_delivery()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .pending
        .remove(&THREAD);
    super::set_current_guest_thread_handle(0);
}

#[test]
fn pthread_attributes_preserve_observable_thread_creation_values() {
    let mut state = ProbeDispatchState::default();
    let mut opaque_attribute = 0_u64;
    let attribute_address = (&raw mut opaque_attribute).addr() as u64;
    assert_eq!(
        create_pthread_attr(attribute_address, PthreadAttrState::default(), &mut state),
        0
    );
    assert_ne!(opaque_attribute, 0);
    assert_eq!(
        pthread_attr_set_detach_state([attribute_address, 1, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        pthread_attr_set_stack_size([attribute_address, 0x20_0000, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        pthread_attr_set_affinity([attribute_address, 0x7f, 0, 0, 0, 0], &mut state),
        0
    );
    let priority = 512_i32;
    assert_eq!(
        pthread_attr_set_schedule_param(
            [
                attribute_address,
                (&raw const priority).addr() as u64,
                0,
                0,
                0,
                0,
            ],
            &mut state,
        ),
        0
    );
    let mut stack_address = u64::MAX;
    let mut stack_size = 0_u64;
    let mut affinity_mask = 0_u64;
    assert_eq!(
        pthread_attr_get_stack(
            [
                attribute_address,
                (&raw mut stack_address).addr() as u64,
                (&raw mut stack_size).addr() as u64,
                0,
                0,
                0,
            ],
            &state,
        ),
        0
    );
    let attributes = state.pthread_attributes[&attribute_address].attributes;
    assert_eq!(attributes.detach_state, 1);
    assert_eq!(attributes.stack_size, 0x20_0000);
    assert_eq!(attributes.schedule_priority, priority);
    assert_eq!(stack_address, 0);
    assert_eq!(stack_size, 0x20_0000);
    assert_eq!(
        pthread_attr_get_affinity(
            [
                attribute_address,
                (&raw mut affinity_mask).addr() as u64,
                0,
                0,
                0,
                0,
            ],
            &state,
        ),
        0x7f
    );
    assert_eq!(affinity_mask, 0x7f);
    assert_eq!(destroy_pthread_attr(attribute_address, &mut state), 0);
    assert_eq!(opaque_attribute, 0);
}

#[test]
fn pthread_threads_are_distinct_guest_pointer_objects() {
    let mut state = ProbeDispatchState {
        next_thread_unique_id: 1,
        ..ProbeDispatchState::default()
    };
    let first = allocate_pthread_thread_object(&mut state, "worker").unwrap();
    let second = allocate_pthread_thread_object(&mut state, "").unwrap();

    assert_ne!(first, 0);
    assert_ne!(first, second);
    assert_eq!(read_guest_u64(first), 0);
    assert_eq!(read_guest_u64(second + 0xff8), 0);
    assert_eq!(state.pthread_threads[&first].unique_id, 1);
    assert_eq!(state.pthread_threads[&first].name, "worker");
    assert_eq!(state.pthread_threads[&second].unique_id, 2);
    assert_eq!(state.pthread_threads[&second].name, "Thread-2");

    for object in state.pthread_threads.into_values() {
        guest_free(object.handle);
    }
}

#[test]
fn pthread_keys_reuse_deleted_slots_and_clear_values_before_destructors() {
    let mut state = ProbeDispatchState::default();
    let mut first_key = u32::MAX;
    let first_key_address = (&raw mut first_key).addr() as u64;
    assert_eq!(
        super::pthread_key_create(
            [first_key_address, 0x1234_5678_9abc_def0, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
    assert_eq!(first_key, 0);
    assert_eq!(
        super::pthread_set_specific([u64::from(first_key), 0xcafe_babe, 0, 0, 0, 0], &state),
        0
    );

    assert_eq!(
        super::take_current_thread_specific_destructors(&state),
        [(0x1234_5678_9abc_def0, 0xcafe_babe)]
    );
    assert_eq!(super::pthread_get_specific(first_key), 0);
    assert_eq!(super::pthread_key_delete(first_key, &mut state), 0);

    let mut replacement_key = u32::MAX;
    assert_eq!(
        super::pthread_key_create(
            [(&raw mut replacement_key).addr() as u64, 0, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
    assert_eq!(replacement_key, first_key);
    assert!(
        super::take_current_thread_specific_destructors(&state).is_empty(),
        "a reused key must not inherit the deleted key's thread-specific value"
    );
    super::GUEST_PTHREAD_SPECIFIC.with_borrow_mut(BTreeMap::clear);
}

#[test]
fn finalization_selects_callbacks_in_reverse_registration_order() {
    let mut state = ProbeDispatchState {
        exit_callbacks: vec![
            GuestExitCallback {
                function: 1,
                argument: None,
                module: 0,
            },
            GuestExitCallback {
                function: 2,
                argument: Some(20),
                module: 7,
            },
            GuestExitCallback {
                function: 3,
                argument: Some(30),
                module: 8,
            },
            GuestExitCallback {
                function: 4,
                argument: Some(40),
                module: 7,
            },
        ],
        ..ProbeDispatchState::default()
    };
    let callbacks = take_exit_callbacks(&mut state, Some(7));
    assert_eq!(
        callbacks
            .iter()
            .map(|callback| callback.function)
            .collect::<Vec<_>>(),
        [4, 2]
    );
    assert_eq!(
        state
            .exit_callbacks
            .iter()
            .map(|callback| callback.function)
            .collect::<Vec<_>>(),
        [1, 3]
    );
}

#[test]
fn decodes_float_argument_from_low_xmm_bits() {
    let expected = -1.25_f32;
    assert_eq!(
        low_f32(f64::from_bits(u64::from(expected.to_bits()))).to_bits(),
        expected.to_bits()
    );
}

#[test]
fn parses_dinkumware_unsigned_longs_with_end_pointer() {
    let text = c"0x2a tail";
    let mut end = std::ptr::null_mut::<libc::c_char>();
    let result = dispatch_guest_numeric_parser(
        "zlfEH8FmyUA",
        [
            text.as_ptr().addr() as u64,
            (&raw mut end).addr() as u64,
            0,
            0,
            0,
            0,
        ],
    );

    assert_eq!(result.integer, 42);
    assert_eq!(end.addr() - text.as_ptr().addr(), 4);
}

#[test]
fn sorts_guest_elements_with_the_guest_comparator() {
    let mut values = [7_i32, -2, 11, 0, 4];
    guest_qsort([
        u64::try_from(values.as_mut_ptr().addr()).unwrap(),
        u64::try_from(values.len()).unwrap(),
        u64::try_from(std::mem::size_of::<i32>()).unwrap(),
        u64::try_from(compare_i32 as *const () as usize).unwrap(),
        0,
        0,
    ]);
    assert_eq!(values, [-2, 0, 4, 7, 11]);
}

#[test]
fn sanitizes_save_slot_path_segments() {
    assert_eq!(sanitize_save_segment("SAVEDATA00"), "SAVEDATA00");
    assert_eq!(sanitize_save_segment("../outside"), "___outside");
    assert_eq!(sanitize_save_segment("///"), "default");
}

#[test]
fn matches_save_data_wildcards_case_insensitively() {
    assert!(save_data_pattern_matches("SAVEDATA00", "save%"));
    assert!(save_data_pattern_matches("slot-A", "SLOT__"));
    assert!(save_data_pattern_matches("alpha-beta", "%ha_b%"));
    assert!(!save_data_pattern_matches("slot", "slot_"));
    assert!(!save_data_pattern_matches("other", "save%"));
}

#[test]
fn temporary_app_content_mount_is_not_a_save_data_slot() {
    let temporary = TemporaryTestDirectory::new("savedata-temp-mount");
    let state = ProbeDispatchState {
        save_root: temporary.path.clone(),
        ..ProbeDispatchState::default()
    };
    let mut mount_point = [0_u8; 16];

    assert_eq!(
        super::app_content_temporary_data_mount(
            [0, mount_point.as_mut_ptr().addr() as u64, 0, 0, 0, 0],
            &state,
        ),
        0
    );
    assert_eq!(&mount_point[..7], b"/temp0\0");
    assert!(temporary.path.join("temp0").is_dir());
    assert!(
        super::enumerate_save_data_entries(&temporary.path, "")
            .unwrap()
            .is_empty()
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn persists_and_searches_transactional_save_data() {
    let temporary = TemporaryTestDirectory::new("savedata-transaction");
    let mut state = ProbeDispatchState {
        save_root: temporary.path.clone(),
        save_data_initialized: true,
        next_save_transaction_resource: 1,
        ..ProbeDispatchState::default()
    };
    std::fs::create_dir_all(&state.save_root).unwrap();

    let resource = save_data_create_transaction_resource([0xc_0000, 0, 0, 0, 0, 0], &mut state);
    assert_eq!(resource, 1);
    let resource = u32::try_from(resource).unwrap();
    let mut dir_name = [0_u8; 32];
    dir_name[..10].copy_from_slice(b"SAVEDATA00");
    let mut mount = [0_u8; 0x30];
    let mut mount_result = [0_u8; 0x40];
    let mount_address = mount.as_mut_ptr().addr() as u64;
    write_guest_u32(mount_address, 0x1000_0000);
    write_guest_u64(mount_address + 0x08, dir_name.as_ptr().addr() as u64);
    write_guest_u64(mount_address + 0x10, 56);
    write_guest_u32(mount_address + 0x20, 1 << 5);
    write_guest_u32(mount_address + 0x28, resource);
    let result_address = mount_result.as_mut_ptr().addr() as u64;
    assert_eq!(
        save_data_mount3([mount_address, result_address, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(&mount_result[..11], b"/savedata0\0");
    assert_eq!(
        u32::from_le_bytes(mount_result[0x1c..0x20].try_into().unwrap()),
        1
    );

    let mut prepare = [0_u8; 0x28];
    prepare[..4].copy_from_slice(&resource.to_le_bytes());
    assert_eq!(
        save_data_prepare(
            [result_address, prepare.as_ptr().addr() as u64, 0, 0, 0, 0,],
            &mut state,
        ),
        0
    );

    let mut param = [0_u8; SAVE_DATA_PARAM_SIZE];
    param[..14].copy_from_slice(b"Save Metadata\0");
    param[0x500..0x504].copy_from_slice(&42_u32.to_le_bytes());
    assert_eq!(
        save_data_set_param(
            [
                result_address,
                0,
                param.as_ptr().addr() as u64,
                SAVE_DATA_PARAM_SIZE as u64,
                0,
                0,
            ],
            &mut state,
        ),
        0
    );
    let icon_bytes = b"\x89PNG\r\n\x1a\ntransaction-icon";
    let icon = [
        icon_bytes.as_ptr().addr() as u64,
        icon_bytes.len() as u64,
        icon_bytes.len() as u64,
    ];
    assert_eq!(
        save_data_save_icon(
            [result_address, icon.as_ptr().addr() as u64, 0, 0, 0, 0],
            &state,
        ),
        0
    );

    let committed_param = state.save_root.join("SAVEDATA00/sce_sys/param.bin");
    assert_ne!(
        &std::fs::read(&committed_param).unwrap()[..14],
        b"Save Metadata\0"
    );
    let mut commit = [0_u8; 0x28];
    commit[..4].copy_from_slice(&resource.to_le_bytes());
    assert_eq!(
        save_data_commit([commit.as_ptr().addr() as u64, 0, 0, 0, 0, 0], &mut state,),
        0
    );
    assert_eq!(
        &std::fs::read(&committed_param).unwrap()[..14],
        b"Save Metadata\0"
    );
    assert_eq!(
        std::fs::read(state.save_root.join("SAVEDATA00/sce_sys/icon0.png")).unwrap(),
        icon_bytes
    );

    let mut search_condition = [0_u8; 0x40];
    let mut search_result = [0_u8; 0x30];
    let pattern = b"save%\0";
    let mut found_names = [0_u8; 64];
    let mut found_params = vec![0_u8; SAVE_DATA_PARAM_SIZE * 2];
    let mut found_infos = [0_u8; 0x60];
    let condition_address = search_condition.as_mut_ptr().addr() as u64;
    let search_result_address = search_result.as_mut_ptr().addr() as u64;
    write_guest_u32(condition_address, 0x1000_0000);
    write_guest_u64(condition_address + 0x10, pattern.as_ptr().addr() as u64);
    write_guest_u32(condition_address + 0x18, 1);
    write_guest_u64(
        search_result_address + 0x08,
        found_names.as_mut_ptr().addr() as u64,
    );
    write_guest_u32(search_result_address + 0x10, 2);
    write_guest_u64(
        search_result_address + 0x18,
        found_params.as_mut_ptr().addr() as u64,
    );
    write_guest_u64(
        search_result_address + 0x20,
        found_infos.as_mut_ptr().addr() as u64,
    );
    assert_eq!(
        save_data_dir_name_search(
            [condition_address, search_result_address, 0, 0, 0, 0],
            &state,
        ),
        0
    );
    assert_eq!(
        u32::from_le_bytes(search_result[..4].try_into().unwrap()),
        1
    );
    assert_eq!(
        u32::from_le_bytes(search_result[0x14..0x18].try_into().unwrap()),
        1
    );
    assert_eq!(&found_names[..11], b"SAVEDATA00\0");
    assert_eq!(&found_params[..14], b"Save Metadata\0");
    assert_eq!(u64::from_le_bytes(found_infos[..8].try_into().unwrap()), 56);

    assert_eq!(
        save_data_umount2([0, result_address, 0, 0, 0, 0], &mut state),
        0
    );
    mount_result.fill(0);
    assert_eq!(
        save_data_mount3([mount_address, result_address, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        u32::from_le_bytes(mount_result[0x1c..0x20].try_into().unwrap()),
        0
    );
    assert_eq!(
        save_data_umount2([0, result_address, 0, 0, 0, 0], &mut state),
        0
    );
    write_guest_u32(mount_address + 0x20, 1 << 2);
    assert_eq!(
        save_data_mount3([mount_address, result_address, 0, 0, 0, 0], &mut state),
        sce_kernel_error(SAVE_DATA_ERROR_EXISTS)
    );
    dir_name.fill(0);
    dir_name[..7].copy_from_slice(b"MISSING");
    write_guest_u32(mount_address + 0x20, 1 << 1);
    assert_eq!(
        save_data_mount3([mount_address, result_address, 0, 0, 0, 0], &mut state),
        sce_kernel_error(SAVE_DATA_ERROR_NOT_FOUND)
    );
}

struct TemporaryTestDirectory {
    path: std::path::PathBuf,
}

impl TemporaryTestDirectory {
    fn new(label: &str) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("remu-{label}-{}-{id}", std::process::id()));
        if path.exists() {
            std::fs::remove_dir_all(&path).unwrap();
        }
        std::fs::create_dir(&path).unwrap();
        Self { path }
    }
}

impl Drop for TemporaryTestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[test]
fn immutable_app_directory_index_resolves_case_once_for_sibling_paths() {
    let temporary = TemporaryTestDirectory::new("app-directory-index");
    let content = temporary.path.join("Content");
    std::fs::create_dir(&content).unwrap();
    std::fs::write(content.join("First.PAK"), []).unwrap();
    std::fs::write(content.join("Second.UCAS"), []).unwrap();

    let cache = std::sync::RwLock::new(super::ImmutableAppPathCache::default());
    assert_eq!(
        super::resolve_immutable_app_path(&temporary.path, "content/first.pak", &cache),
        Some(content.join("First.PAK"))
    );
    assert_eq!(
        super::resolve_immutable_app_path(&temporary.path, "CONTENT/second.ucas", &cache),
        Some(content.join("Second.UCAS"))
    );

    let cache = cache
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert!(cache.directories.contains_key(&temporary.path));
    assert!(cache.directories.contains_key(&content));
    assert_eq!(cache.directories.len(), 2);
}

#[test]
fn implements_seek_read_stat_and_truncate_on_one_guest_descriptor() {
    let temporary = TemporaryTestDirectory::new("file-abi");
    let path = temporary.path.join("payload.bin");
    std::fs::write(&path, b"abcdef").unwrap();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *files = KernelFileState::default();
    files.files.insert(3, file);
    drop(files);

    assert_eq!(guest_kernel_lseek([3, 2, 0, 0, 0, 0]), 2);
    let mut bytes = [0_u8; 2];
    assert_eq!(
        guest_kernel_read([3, bytes.as_mut_ptr().addr() as u64, 2, 0, 0, 0]),
        2
    );
    assert_eq!(&bytes, b"cd");

    let mut stat = [0_u8; super::SCE_KERNEL_STAT_SIZE];
    assert_eq!(
        guest_kernel_fstat([3, stat.as_mut_ptr().addr() as u64, 0, 0, 0, 0]),
        0
    );
    assert_eq!(i64::from_le_bytes(stat[72..80].try_into().unwrap()), 6);
    assert_ne!(u32::from_le_bytes(stat[4..8].try_into().unwrap()), 0);
    assert_ne!(u16::from_le_bytes(stat[8..10].try_into().unwrap()), 0);
    assert_ne!(u32::from_le_bytes(stat[88..92].try_into().unwrap()), 0);

    assert_eq!(guest_kernel_ftruncate([3, 4, 0, 0, 0, 0]), 0);
    assert_eq!(
        KERNEL_FILES
            .get()
            .unwrap()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .files
            .get(&3)
            .unwrap()
            .metadata()
            .unwrap()
            .len(),
        4
    );
}

/// The per-entry form attempts every path and reports how many resolved, instead of stopping
/// at the first rejection the way the error-index form does.
#[cfg(target_os = "macos")]
#[test]
fn per_entry_apr_resolution_continues_past_a_rejected_path() {
    let mut state = ProbeDispatchState::default();
    let missing = b"/app0/missing-one.bin\0";
    let also_missing = b"/app0/missing-two.bin\0";
    let list = [
        missing.as_ptr().addr() as u64,
        also_missing.as_ptr().addr() as u64,
    ];
    let mut ids = [0x1234_5678_u32; 2];
    let mut results = [0_u32; 2];
    // Every entry is attempted, so the reported count is the number that resolved: none.
    assert_eq!(
        ampr_resolve_paths_core(
            list.as_ptr().addr() as u64,
            2,
            None,
            AprResolveOutputs {
                ids: ids.as_mut_ptr().addr() as u64,
                results: results.as_mut_ptr().addr() as u64,
                ..AprResolveOutputs::default()
            },
            &mut state,
        ),
        0
    );
    assert_eq!(ids, [AMPR_INVALID_FILE_ID; 2]);
    assert!(results.iter().all(|result| *result != 0));
}

#[test]
fn resolver_prefix_keeps_one_separator() {
    assert_eq!(join_guest_path("/app0", "config.ini"), "/app0/config.ini");
    assert_eq!(join_guest_path("/app0/", "/config.ini"), "/app0/config.ini");
    assert_eq!(join_guest_path("", "/config.ini"), "/config.ini");
}

/// A rejected path must leave an identifier the guest recognizes as invalid. Leaving the
/// caller's buffer untouched lets uninitialized stack data pass as a usable file id, and the
/// guest then issues a read for a file it never resolved.
#[cfg(target_os = "macos")]
#[test]
fn unresolved_apr_paths_report_an_invalid_identifier() {
    let mut state = ProbeDispatchState::default();
    let path = b"/app0/does-not-exist.res\0";
    let path_list = [path.as_ptr().addr() as u64];
    // Pre-fill with values that would look like plausible identifiers if left untouched.
    let mut ids = [0x006e_0069_u32; 1];
    let mut sizes = [0xdead_beef_u64; 1];
    let mut error_index = 0_u32;
    assert_ne!(
        ampr_resolve_file_paths(
            [
                path_list.as_ptr().addr() as u64,
                1,
                ids.as_mut_ptr().addr() as u64,
                sizes.as_mut_ptr().addr() as u64,
                (&raw mut error_index).addr() as u64,
                0,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(ids[0], AMPR_INVALID_FILE_ID);
    assert_eq!(sizes[0], 0);
    assert_eq!(error_index, 0);
}

#[test]
fn ampr_read_releases_process_dispatch_lock_while_copying_and_records_result() {
    let path = create_ampr_test_file(b"host payload");
    let file_id = 0x1234_5678;
    let command_buffer = 0x5000;
    let mut command_records = [0_u8; 0x60];
    let mut destination = [0_u8; 4];
    let mut state = ProbeDispatchState::default();
    state.ampr_files.insert(file_id, path.clone());
    state.ampr_command_buffers.insert(
        command_buffer,
        AmprCommandBufferState {
            buffer: command_records.as_mut_ptr().addr() as u64,
            size: command_records.len() as u64,
            ..AmprCommandBufferState::default()
        },
    );
    let dispatch = RwLock::new(state);
    let order = Mutex::new(());
    let destination_address = destination.as_mut_ptr().addr() as u64;

    assert_eq!(
        super::ampr_command_buffer_read_file_two_phase_with(
            &order,
            &dispatch,
            [command_buffer, 0, 0, file_id.into(), destination_address, 4],
            [7, 0, 0, 0, 0, 0],
            |_, observed_file_id, observed_destination, size, offset, capacity| {
                let unrelated_state = dispatch
                    .try_write()
                    .expect("process dispatch lock must be released during AMPR I/O");
                drop(unrelated_state);
                assert_eq!(observed_file_id, file_id);
                assert_eq!(observed_destination, destination_address);
                assert_eq!((size, offset, capacity), (4, 7, 4));
                // SAFETY: `destination` remains live for the complete synchronous call.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        b"data".as_ptr(),
                        observed_destination as *mut u8,
                        4,
                    );
                }
                Ok(4)
            },
        ),
        0
    );

    assert_eq!(&destination, b"data");
    let state = dispatch.read().unwrap();
    let command = state.ampr_command_buffers.get(&command_buffer).unwrap();
    assert_eq!(command.write_offset, super::AMPR_COMMAND_RECORD_SIZE);
    assert_eq!(command.command_count, 1);
    assert!(state.ampr_file_handles.contains_key(&file_id));
    drop(state);
    assert_eq!(
        u32::from_le_bytes(command_records[0..4].try_into().unwrap()),
        1
    );
    assert_eq!(
        u32::from_le_bytes(command_records[4..8].try_into().unwrap()),
        file_id
    );
    assert_eq!(
        u64::from_le_bytes(command_records[0x18..0x20].try_into().unwrap()),
        7
    );
    assert_eq!(
        u64::from_le_bytes(command_records[0x20..0x28].try_into().unwrap()),
        4
    );
    std::fs::remove_file(path).ok();
}

#[test]
fn ampr_host_read_honors_offset_and_short_end_of_file() {
    let path = create_ampr_test_file(b"012345");
    let file = std::fs::File::open(&path).unwrap();
    let mut destination = [0xcc_u8; 8];
    assert_eq!(
        super::ampr_read_file_to_guest(
            &file,
            0x1234,
            destination.as_mut_ptr().addr() as u64,
            destination.len() as u64,
            2,
            destination.len(),
        ),
        Ok(4)
    );
    assert_eq!(&destination[..4], b"2345");
    assert_eq!(&destination[4..], &[0xcc; 4]);
    std::fs::remove_file(path).ok();
}

#[test]
fn failed_ampr_read_keeps_command_buffer_unmodified() {
    let path = create_ampr_test_file(b"host payload");
    let file_id = 0x2468_1357;
    let command_buffer = 0x5500;
    let mut command_records = [0xa5_u8; 0x60];
    let mut destination = [0_u8; 4];
    let mut state = ProbeDispatchState::default();
    state.ampr_files.insert(file_id, path.clone());
    state.ampr_command_buffers.insert(
        command_buffer,
        AmprCommandBufferState {
            buffer: command_records.as_mut_ptr().addr() as u64,
            size: command_records.len() as u64,
            ..AmprCommandBufferState::default()
        },
    );
    let dispatch = RwLock::new(state);
    let error = sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);

    assert_eq!(
        super::ampr_command_buffer_read_file_two_phase_with(
            &Mutex::new(()),
            &dispatch,
            [
                command_buffer,
                0,
                0,
                file_id.into(),
                destination.as_mut_ptr().addr() as u64,
                destination.len() as u64,
            ],
            [0; 6],
            |_, _, _, _, _, _| Err(error),
        ),
        error
    );

    let state = dispatch.read().unwrap();
    let command = state.ampr_command_buffers.get(&command_buffer).unwrap();
    assert_eq!(command.write_offset, 0);
    assert_eq!(command.command_count, 0);
    assert!(state.ampr_file_handles.contains_key(&file_id));
    assert_eq!(command_records, [0xa5; 0x60]);
    drop(state);
    std::fs::remove_file(path).ok();
}

#[test]
fn ampr_submit_cannot_overtake_an_in_flight_read() {
    let path = create_ampr_test_file(b"host payload");
    let file_id = 0x8765_4321;
    let command_buffer = 0x6000;
    let mut command_records = [0_u8; 0x60];
    let mut destination = [0_u8; 4];
    let mut state = ProbeDispatchState::default();
    state.ampr_files.insert(file_id, path.clone());
    state.ampr_command_buffers.insert(
        command_buffer,
        AmprCommandBufferState {
            buffer: command_records.as_mut_ptr().addr() as u64,
            size: command_records.len() as u64,
            ..AmprCommandBufferState::default()
        },
    );
    let dispatch = Arc::new(RwLock::new(state));
    let order = Arc::new(Mutex::new(()));
    let destination_address = destination.as_mut_ptr().addr() as u64;
    let (read_entered_tx, read_entered_rx) = mpsc::channel();
    let (release_read_tx, release_read_rx) = mpsc::channel();
    let read_dispatch = Arc::clone(&dispatch);
    let read_order = Arc::clone(&order);
    let read = std::thread::spawn(move || {
        super::ampr_command_buffer_read_file_two_phase_with(
            &read_order,
            &read_dispatch,
            [command_buffer, 0, 0, file_id.into(), destination_address, 4],
            [0; 6],
            move |_, _, observed_destination, _, _, _| {
                read_entered_tx.send(()).unwrap();
                release_read_rx.recv().unwrap();
                // SAFETY: the owning test thread retains this destination until both workers
                // join, and the AMPR ordering lock prevents submission from observing it early.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        b"done".as_ptr(),
                        observed_destination as *mut u8,
                        4,
                    );
                }
                Ok(4)
            },
        )
    });
    read_entered_rx
        .recv_timeout(Duration::from_secs(1))
        .expect("read reached unlocked I/O phase");

    let submit_dispatch = Arc::clone(&dispatch);
    let submit_order = Arc::clone(&order);
    let (submit_started_tx, submit_started_rx) = mpsc::channel();
    let (submit_done_tx, submit_done_rx) = mpsc::channel();
    let submit = std::thread::spawn(move || {
        submit_started_tx.send(()).unwrap();
        let _order = submit_order.lock().unwrap();
        let mut state = submit_dispatch.write().unwrap();
        let result = ampr_submit_command_buffer([command_buffer, 0, 0, 0, 0, 0], &mut state);
        submit_done_tx.send(result).unwrap();
    });
    submit_started_rx.recv().unwrap();
    assert!(
        submit_done_rx
            .recv_timeout(Duration::from_millis(50))
            .is_err(),
        "submission overtook the blocked file read"
    );

    release_read_tx.send(()).unwrap();
    assert_eq!(read.join().unwrap(), 0);
    assert_eq!(
        submit_done_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        0
    );
    submit.join().unwrap();
    assert_eq!(&destination, b"done");
    let state = dispatch.read().unwrap();
    assert_eq!(state.next_apr_submission_id, 1);
    assert_eq!(
        state
            .ampr_command_buffers
            .get(&command_buffer)
            .unwrap()
            .write_offset,
        super::AMPR_COMMAND_RECORD_SIZE
    );
    drop(state);
    std::fs::remove_file(path).ok();
}

#[test]
fn ampr_ordering_classifier_covers_every_stateful_operation() {
    for nid in [
        "gEpBkcwxUjw",
        "WT-5NKy42fw",
        "eYAh2vlCY-U",
        "QzB4O+bJQyA",
        "i3HWvW35jao",
        "w5fcCG+t31g",
        "VB-BtuIW8Xc",
        "C+Khtbbx2g8",
        "WvEu7yl3Ivg",
        "ApkYaHb8Sek",
        "eE4Szl8sil8",
        "qvMUCyyaCSI",
        "ASoW5WE-UPo",
        "rqwFKI4PAiM",
        "bBfz7kMF2Ho",
        "8aI7R7WaOlc",
        "a8uLzYY--tM",
        "N-FSPA4S3nI",
        "baQO9ez2gL4",
        "ULvXMDz56po",
        "tZDDEo2tE5k",
        "GnxKOHEawhk",
        "H896Pt-yB4I",
        "GuchCTefuZw",
        "Qs1xtplKo0U",
        "mQ16-QdKv7k",
    ] {
        assert!(
            super::is_serialized_ampr_import(nid),
            "{nid} bypassed AMPR ordering"
        );
    }
}

/// Registering interest must not announce a completion: the guest's listener resolves the
/// event against state the submission creates, so an early wake dereferences a context that
/// does not exist.
#[test]
fn ampr_completions_are_announced_by_submission_not_registration() {
    let mut state = ProbeDispatchState::default();
    let equeue = 0x4000;
    state.kernel_equeues.insert(equeue);
    assert_eq!(
        kernel_add_ampr_event([equeue, 0x7502, 0xabc, 0, 0, 0], &mut state),
        0
    );
    assert!(
        state.pending_events.is_empty(),
        "registration announced a completion"
    );

    let mut buffer = [0_u8; 0x100];
    let command_buffer = 0x5000_u64;
    state.ampr_command_buffers.insert(
        command_buffer,
        AmprCommandBufferState {
            buffer: buffer.as_mut_ptr().addr() as u64,
            size: buffer.len() as u64,
            ..AmprCommandBufferState::default()
        },
    );
    assert_eq!(
        ampr_command_buffer_write_kernel_event_queue(
            [command_buffer, equeue, 0x7502, 0xabc, 0, 0],
            &mut state,
        ),
        0
    );
    assert!(
        state.pending_events.is_empty(),
        "recording announced a completion before submission"
    );

    assert_eq!(
        ampr_submit_command_buffer([command_buffer, 0, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(state.pending_events.len(), 1);
    let event = state.pending_events.front().unwrap();
    assert_eq!(event.equeue, equeue);
    assert_eq!(event.ident, 0x7502);
    assert_eq!(event.user_data, 0xabc);
}

#[test]
fn directory_records_pack_console_dirents_without_crossing_blocks() {
    let entries: Vec<(String, bool)> = (0..40)
        .map(|index| (format!("pakchunk{index}-ps5.pak"), index % 7 == 0))
        .collect();
    let mut next_file_number = 1;
    let directory = KernelDirectory::new(&entries, &mut next_file_number);
    assert_eq!(directory.records.len() % DIRENT_BLOCK_SIZE, 0);

    // Walk the records exactly as the guest does and recover every name in order.
    let mut names = Vec::new();
    let mut offset = 0;
    while offset < directory.records.len() {
        let record = &directory.records[offset..];
        let reclen = usize::from(u16::from_le_bytes([record[4], record[5]]));
        let namlen = usize::from(record[7]);
        if namlen == 0 {
            offset += DIRENT_BLOCK_SIZE - offset % DIRENT_BLOCK_SIZE;
            continue;
        }
        assert!(reclen >= DIRENT_HEADER_SIZE + namlen + 1);
        // A record must lie entirely inside one 512-byte block.
        assert_eq!(
            offset / DIRENT_BLOCK_SIZE,
            (offset + DIRENT_HEADER_SIZE + namlen) / DIRENT_BLOCK_SIZE,
            "record at {offset} crossed a block boundary"
        );
        names.push(String::from_utf8_lossy(&record[8..8 + namlen]).into_owned());
        assert_eq!(record[6], if names.len() % 7 == 1 { 4 } else { 8 });
        offset += reclen;
    }
    assert_eq!(
        names,
        entries
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(next_file_number, 41);
}

/// Engines store timestamps as ticks and decompose them again for display. A tick that does
/// not survive the round trip produces an implausible date and fails their validation.
#[test]
fn rtc_tick_round_trips_through_a_date() {
    let mut date = [0_u16; 8];
    date[0] = 2026;
    date[1] = 7;
    date[2] = 25;
    date[3] = 18;
    date[4] = 43;
    date[5] = 7;
    let mut tick = 0_u64;
    assert_eq!(
        rtc_get_tick([
            date.as_mut_ptr().addr() as u64,
            (&raw mut tick).addr() as u64,
            0,
            0,
            0,
            0,
        ]),
        0
    );
    assert_ne!(tick, 0);

    let mut recovered = [0_u16; 8];
    assert_eq!(
        rtc_set_tick([
            recovered.as_mut_ptr().addr() as u64,
            (&raw const tick).addr() as u64,
            0,
            0,
            0,
            0,
        ]),
        0
    );
    assert_eq!(recovered[..6], date[..6]);

    // The current clock must also decompose to the same instant it reports as a tick.
    let mut now_tick = 0_u64;
    assert_eq!(rtc_get_current_tick((&raw mut now_tick).addr() as u64), 0);
    let mut now_date = [0_u16; 8];
    assert_eq!(
        rtc_set_tick([
            now_date.as_mut_ptr().addr() as u64,
            (&raw const now_tick).addr() as u64,
            0,
            0,
            0,
            0,
        ]),
        0
    );
    assert!(now_date[0] >= 2020, "year was {}", now_date[0]);
    assert!((1..=12).contains(&now_date[1]));
}

#[test]
fn civil_days_round_trip_through_civil_decomposition() {
    for seconds in [
        0_i64,
        1_709_164_800,
        951_782_400,
        -2_203_891_201,
        1_800_000_000,
    ] {
        let (year, month, day, hour, minute, second) = civil_from_unix_seconds(seconds);
        let days = unix_days_from_civil(i64::from(year), i64::from(month), i64::from(day));
        let rebuilt =
            days * 86_400 + i64::from(hour) * 3_600 + i64::from(minute) * 60 + i64::from(second);
        assert_eq!(rebuilt, seconds, "round trip failed for {seconds}");
    }
}

/// The selector chooses the display's hold time; an unrecognized value keeps the 60 Hz
/// default rather than disabling pacing.
#[test]
fn flip_rate_selectors_map_to_display_intervals() {
    assert_eq!(flip_interval_microseconds(0), 16_666);
    assert_eq!(flip_interval_microseconds(1), 33_333);
    assert_eq!(flip_interval_microseconds(2), 50_000);
    assert_eq!(flip_interval_microseconds(7), 16_666);
}

/// Pacing must gate the guest's frame loop: submitting flips in a tight loop has to take at
/// least the display interval per flip.
#[test]
fn paced_flips_are_held_for_the_display_interval() {
    VIDEO_OUT_FLIP_INTERVAL.store(4_000, Ordering::Relaxed);
    pace_video_out_flip();
    let start = Instant::now();
    for _ in 0..5 {
        pace_video_out_flip();
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed >= Duration::from_micros(4_000 * 4),
        "five paced flips returned in {elapsed:?}"
    );
    VIDEO_OUT_FLIP_INTERVAL.store(DEFAULT_FLIP_INTERVAL_MICROSECONDS, Ordering::Relaxed);
}

#[test]
fn civil_decomposition_matches_known_timestamps() {
    // Epoch, a leap day, the end of a non-leap century year, and a pre-epoch date.
    assert_eq!(civil_from_unix_seconds(0), (1970, 1, 1, 0, 0, 0));
    assert_eq!(
        civil_from_unix_seconds(1_709_164_800),
        (2024, 2, 29, 0, 0, 0)
    );
    // Year 2000 is a leap year despite being a century, so the day after February 28 exists.
    assert_eq!(
        civil_from_unix_seconds(951_782_399),
        (2000, 2, 28, 23, 59, 59)
    );
    assert_eq!(civil_from_unix_seconds(951_782_400), (2000, 2, 29, 0, 0, 0));
    // Year 1900 is not a leap year, so February ends after the 28th.
    assert_eq!(
        civil_from_unix_seconds(-2_203_891_201),
        (1900, 2, 28, 23, 59, 59)
    );
    assert_eq!(civil_from_unix_seconds(-1), (1969, 12, 31, 23, 59, 59));
}

#[test]
fn current_local_time_is_a_valid_date_and_rejects_a_null_output() {
    let mut time = [0_u16; 8];
    assert_eq!(
        rtc_get_current_clock_local_time(time.as_mut_ptr().addr() as u64),
        0
    );
    // Engines abort on an all-zero date, so every field must be in its calendar range.
    assert!(time[0] >= 2020, "year was {}", time[0]);
    assert!((1..=12).contains(&time[1]), "month was {}", time[1]);
    assert!((1..=31).contains(&time[2]), "day was {}", time[2]);
    assert!(time[3] < 24 && time[4] < 60 && time[5] < 60);
    assert_ne!(rtc_get_current_clock_local_time(0), 0);
}

/// Builds one `sceKernelBatchMap` entry: start, direct offset, length, protection, type, and
/// operation at the console's fixed field offsets.
#[cfg(target_os = "macos")]
fn batch_map_entry(
    start: u64,
    offset: u64,
    length: u64,
    protection: u8,
    memory_type: u8,
    operation: u32,
) -> [u8; 32] {
    let mut entry = [0_u8; 32];
    entry[0..8].copy_from_slice(&start.to_le_bytes());
    entry[8..16].copy_from_slice(&offset.to_le_bytes());
    entry[16..24].copy_from_slice(&length.to_le_bytes());
    entry[24] = protection;
    entry[25] = memory_type;
    entry[28..32].copy_from_slice(&operation.to_le_bytes());
    entry
}

#[cfg(target_os = "macos")]
#[test]
fn batch_map_applies_entries_in_order_and_reports_the_completed_count() {
    let mut state = ProbeDispatchState::default();
    let mut physical = 0_u64;
    // A non-positive search end means the whole direct-memory range.
    assert_eq!(
        allocate_direct_memory(
            [0, 0, 0x8000, 0x4000, 12, (&raw mut physical).addr() as u64,],
            &mut state,
        ),
        0
    );
    let mut entries = [
        batch_map_entry(0, 0, 0x4000, 0x03, 12, BATCH_MAP_OPERATION_MAP_DIRECT),
        batch_map_entry(0, 0x4000, 0x4000, 0x03, 12, BATCH_MAP_OPERATION_MAP_DIRECT),
    ];
    let mut completed = 0_u32;
    assert_eq!(
        kernel_batch_map(
            [
                entries.as_mut_ptr().addr() as u64,
                2,
                (&raw mut completed).addr() as u64,
                0,
                0,
                0,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(completed, 2);
    // Each entry's own start field receives the address the kernel chose for it.
    let first = u64::from_le_bytes(entries[0][0..8].try_into().unwrap());
    let second = u64::from_le_bytes(entries[1][0..8].try_into().unwrap());
    assert_ne!(first, 0);
    assert_ne!(second, 0);
    assert_ne!(first, second);
}

#[cfg(target_os = "macos")]
#[test]
fn batch_map_stops_at_the_first_invalid_entry_and_still_reports_progress() {
    let mut state = ProbeDispatchState::default();
    let mut physical = 0_u64;
    assert_eq!(
        allocate_direct_memory(
            [0, 0, 0x8000, 0x4000, 12, (&raw mut physical).addr() as u64,],
            &mut state,
        ),
        0
    );
    let mut entries = [
        batch_map_entry(0, 0, 0x4000, 0x03, 12, BATCH_MAP_OPERATION_MAP_DIRECT),
        // An operation outside the defined set is rejected before anything is applied.
        batch_map_entry(0, 0x4000, 0x4000, 0x03, 12, 5),
        batch_map_entry(0, 0x4000, 0x4000, 0x03, 12, BATCH_MAP_OPERATION_MAP_DIRECT),
    ];
    let mut completed = 0_u32;
    assert_eq!(
        kernel_batch_map(
            [
                entries.as_mut_ptr().addr() as u64,
                3,
                (&raw mut completed).addr() as u64,
                0,
                0,
                0,
            ],
            &mut state,
        ),
        sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
    assert_eq!(completed, 1);
    // The entry after the rejected one is never applied.
    assert_eq!(u64::from_le_bytes(entries[2][0..8].try_into().unwrap()), 0);
}

#[cfg(target_os = "macos")]
#[test]
fn maps_named_flexible_memory_with_cpu_read_write_protection() {
    let mut state = ProbeDispatchState::default();
    let mut mapped = 0_u64;
    let name = b"CppWebApi\0";
    assert_eq!(
        super::map_named_flexible_memory(
            [
                (&raw mut mapped).addr() as u64,
                0x8_0000,
                0x02,
                0,
                name.as_ptr().addr() as u64,
                0,
            ],
            &mut state,
        ),
        0
    );
    assert_ne!(mapped, 0);
    // SAFETY: protection 2 is the guest ABI's CPU read/write mode.
    unsafe {
        (mapped as *mut u8).write_volatile(0x5a);
        assert_eq!((mapped as *const u8).read_volatile(), 0x5a);
    }

    let mut output = [0xa5_u8; 72];
    assert_eq!(
        kernel_virtual_query(
            [
                mapped,
                0,
                output.as_mut_ptr().addr() as u64,
                output.len() as u64,
                0,
                0,
            ],
            &state,
        ),
        0
    );
    assert_eq!(u32::from_le_bytes(output[24..28].try_into().unwrap()), 2);
    assert_eq!(u32::from_le_bytes(output[32..36].try_into().unwrap()), 0x11);
    assert_eq!(&output[36..44], b"flexible");
    assert_eq!(
        super::known_nid_name("mL8NDH86iQI"),
        Some("sceKernelMapNamedFlexibleMemory")
    );
}

#[cfg(target_os = "macos")]
#[test]
fn rejects_invalid_named_flexible_memory_arguments() {
    let mut state = ProbeDispatchState::default();
    let mut mapped = 0_u64;
    let short_name = b"flex\0";
    let long_name = [b'x'; 33];
    let mut terminated_long_name = [0_u8; 34];
    terminated_long_name[..33].copy_from_slice(&long_name);

    assert_eq!(
        super::map_named_flexible_memory(
            [
                (&raw mut mapped).addr() as u64,
                1,
                2,
                0,
                short_name.as_ptr().addr() as u64,
                0,
            ],
            &mut state,
        ),
        sce_kernel_error(super::SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
    assert_eq!(
        super::map_named_flexible_memory(
            [
                (&raw mut mapped).addr() as u64,
                0x4000,
                2,
                0,
                terminated_long_name.as_ptr().addr() as u64,
                0,
            ],
            &mut state,
        ),
        sce_kernel_error(super::SCE_KERNEL_ERROR_NAME_TOO_LONG)
    );
    assert_eq!(
        super::map_named_flexible_memory(
            [
                (&raw mut mapped).addr() as u64,
                0x4000,
                2,
                0x10,
                short_name.as_ptr().addr() as u64,
                0,
            ],
            &mut state,
        ),
        sce_kernel_error(super::SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    );
}

#[cfg(target_os = "macos")]
#[test]
fn mspace_allocates_inside_named_flexible_memory_and_reuses_freed_blocks() {
    let mut state = ProbeDispatchState::default();
    let mut mapped = 0_u64;
    let mapping_name = b"WebApiBacking\0";
    assert_eq!(
        super::map_named_flexible_memory(
            [
                (&raw mut mapped).addr() as u64,
                0x8000,
                2,
                0,
                mapping_name.as_ptr().addr() as u64,
                0,
            ],
            &mut state,
        ),
        0
    );
    let mspace_name = b"WebApiMspace\0";
    let handle = super::guest_mspace_create(
        [mspace_name.as_ptr().addr() as u64, mapped, 0x8000, 0, 0, 0],
        &mut state,
    );
    assert_eq!(handle, mapped);

    let first = super::guest_mspace_malloc([handle, 0x58, 0, 0, 0, 0], &mut state);
    let second = super::guest_mspace_malloc([handle, 0x30, 0, 0, 0, 0], &mut state);
    assert!(first >= mapped + 0x100);
    assert_eq!(second, first + 0x60);
    // SAFETY: `first` is a live mspace allocation of 0x60 usable bytes.
    unsafe { std::ptr::write_bytes(first as *mut u8, 0x5a, 0x58) };
    let grown = super::guest_mspace_realloc([handle, first, 0x100, 0, 0, 0], &mut state);
    assert_ne!(grown, 0);
    // SAFETY: realloc preserved the first 0x58 bytes in the returned live allocation.
    assert!(
        unsafe { std::slice::from_raw_parts(grown as *const u8, 0x58) }
            .iter()
            .all(|byte| *byte == 0x5a)
    );
    assert_eq!(
        super::guest_mspace_usable_size([handle, grown, 0, 0, 0, 0], &state),
        0x100
    );

    super::guest_mspace_free([handle, second, 0, 0, 0, 0], &mut state);
    super::guest_mspace_free([handle, grown, 0, 0, 0, 0], &mut state);
    assert_eq!(
        super::guest_mspace_is_heap_empty([handle, 0, 0, 0, 0, 0], &state),
        1
    );
    assert_eq!(
        super::guest_mspace_malloc([handle, 0x58, 0, 0, 0, 0], &mut state),
        first
    );
    assert_eq!(
        super::known_nid_name("OJjm-QOIHlI"),
        Some("sceLibcMspaceMalloc")
    );
}

#[test]
fn json_initializer_tracks_allocator_and_file_buffer_configuration() {
    let mut state = ProbeDispatchState::default();
    let mut initializer = [0_u8; 8];
    let mut parameter = [0_u8; 0x28];
    let mut allocator = [0_u8; 8];
    let initializer_address = initializer.as_mut_ptr().addr() as u64;
    let parameter_address = parameter.as_mut_ptr().addr() as u64;
    let allocator_address = allocator.as_mut_ptr().addr() as u64;

    assert_eq!(
        super::json_init_parameter_set_allocator(
            [parameter_address, allocator_address, 0x1234, 0, 0, 0,],
            &mut state,
        ),
        0
    );
    assert_eq!(
        super::json_init_parameter_set_file_buffer_size(
            [parameter_address, 0x200, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
    assert_eq!(
        super::json_initializer_initialize(
            [initializer_address, parameter_address, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
    assert_eq!(
        state.json_runtime.allocators.get(&parameter_address),
        Some(&(allocator_address, 0x1234))
    );
    assert_eq!(
        state.json_runtime.file_buffer_sizes.get(&parameter_address),
        Some(&0x200)
    );
    assert_eq!(
        state
            .json_runtime
            .initialized_objects
            .get(&initializer_address),
        Some(&parameter_address)
    );
    assert_eq!(
        super::known_nid_name("IXW-z8pggfg"),
        Some("sce::Json::Initializer::initialize(InitParameter2)")
    );
    assert_eq!(
        super::json_initializer_terminate([initializer_address, 0, 0, 0, 0, 0], &mut state,),
        0
    );
}

#[test]
fn offline_np_service_initializers_retain_context_and_property_ownership() {
    let mut state = ProbeDispatchState::default();
    assert_ne!(super::sysmodule_is_loaded(0xcc, &state), 0);
    assert_eq!(super::sysmodule_load(0xcc, &mut state), 0);
    assert_eq!(super::sysmodule_is_loaded(0xcc, &state), 0);
    assert_eq!(super::game_update_initialize(&mut state), 0);

    let entitlement_parameters = [0_u8; 0x20];
    let mut entitlement_context = [0_u8; 0x20];
    let entitlement_context_address = entitlement_context.as_mut_ptr().addr() as u64;
    let entitlement_parameter_address = entitlement_parameters.as_ptr().addr() as u64;
    assert_eq!(
        super::np_entitlement_access_initialize(
            [
                entitlement_context_address,
                entitlement_parameter_address,
                0,
                0,
                0,
                0,
            ],
            &mut state,
        ),
        0
    );
    let signaling_parameters = [0x5a_u8; 0x28];
    assert_eq!(
        super::np_session_signaling_initialize(
            [signaling_parameters.as_ptr().addr() as u64, 0, 0, 0, 0, 0],
            &mut state,
        ),
        0
    );
    let user_context =
        super::np_web_api2_create_user_context([1, 0x1000_0000, 0, 0, 0, 0], &mut state);
    assert_eq!(user_context, 1);
    assert_eq!(super::np_check_callback(&mut state), 0);
    assert_eq!(state.services.np_callback_polls, 1);

    let mut universal_parameters = [0_u8; 16];
    universal_parameters[..8].copy_from_slice(&16_u64.to_ne_bytes());
    assert_eq!(
        super::np_universal_data_initialize(
            universal_parameters.as_ptr().addr() as u64,
            &mut state,
        ),
        0
    );
    let mut object = 0_u64;
    assert_eq!(
        super::np_universal_data_create_property_object(
            (&raw mut object).addr() as u64,
            &mut state,
        ),
        0
    );
    assert_ne!(object, 0);
    assert_eq!(
        super::known_nid_name("s6W4Zl4Slgk"),
        Some("sceNpUniversalDataSystemCreateEventPropertyObject")
    );
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
        super::av_player_set_log_callback([0x1234, 0x5678, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        super::av_player_init_ex(
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
        super::av_player_post_init(
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
        super::av_player_add_source(
            [handle, source.as_ptr().addr() as u64, 0, 0, 0, 0],
            &mut state,
        ),
        sce_kernel_error(super::AV_PLAYER_ERROR_OPERATION_FAILED)
    );
    assert_eq!(super::av_player_is_active(handle, &state), 0);
    assert_eq!(super::av_player_close(handle, &mut state), 0);
    assert_eq!(
        super::known_nid_name("o9eWRkSL+M4"),
        Some("sceAvPlayerInitEx")
    );
    assert_eq!(
        super::known_nid_name("HD1YKVU26-M"),
        Some("sceAvPlayerPostInit")
    );
}

#[test]
fn av_player_gen5_stream_info_is_zero_based_and_exactly_32_bytes() {
    let mut stream_info = [0xab_u8; 40];
    super::av_player_write_stream_info(
        stream_info.as_mut_ptr().addr() as u64,
        0,
        378,
        150,
        std::time::Duration::from_millis(9_109),
    );

    assert_eq!(
        u32::from_ne_bytes(stream_info[0..4].try_into().unwrap()),
        super::AV_PLAYER_STREAM_TYPE_VIDEO
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

    super::av_player_write_stream_info(
        stream_info.as_mut_ptr().addr() as u64,
        1,
        378,
        150,
        std::time::Duration::from_millis(9_109),
    );
    assert_eq!(
        u32::from_ne_bytes(stream_info[0..4].try_into().unwrap()),
        super::AV_PLAYER_STREAM_TYPE_AUDIO
    );
}

#[test]
fn av_player_gen5_extended_frame_info_uses_right_and_bottom_crop_fields() {
    let mut frame_info = [0xab_u8; 112];
    super::av_player_write_video_frame_info(
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

    assert!(super::av_player_video_frame_is_due(
        &mut origin,
        0,
        29.97,
        100,
        interval
    ));
    assert!(!super::av_player_video_frame_is_due(
        &mut origin,
        1,
        29.97,
        101,
        interval
    ));
    assert!(super::av_player_video_frame_is_due(
        &mut origin,
        1,
        29.97,
        102,
        interval
    ));
    assert!(!super::av_player_video_frame_is_due(
        &mut origin,
        2,
        29.97,
        102,
        interval
    ));
    assert!(super::av_player_video_frame_is_due(
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
        super::av_player_video_catch_up_target(
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
        super::av_player_video_catch_up_target(
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
        super::av_player_video_catch_up_target(
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
        super::av_player_video_catch_up_target(
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

#[cfg(target_os = "macos")]
#[test]
fn describes_a_tracked_direct_mapping_to_the_guest() {
    let mut state = ProbeDispatchState::default();
    let mut direct = u64::MAX;
    assert_eq!(
        allocate_direct_memory(
            [
                0x20_0000,
                0x21_0000,
                0x1_0000,
                0x4000,
                12,
                (&raw mut direct).addr() as u64,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(direct, 0x20_0000);
    let mut mapped = 0_u64;
    assert_eq!(
        super::map_direct_memory(
            [
                (&raw mut mapped).addr() as u64,
                0x1_0000,
                0x33,
                0,
                direct,
                0x4000,
            ],
            &mut state,
        ),
        0
    );
    let mut output = [0xa5_u8; 72];
    let result = kernel_virtual_query(
        [
            mapped,
            0,
            output.as_mut_ptr().addr() as u64,
            output.len() as u64,
            0,
            0,
        ],
        &state,
    );
    assert_eq!(result, 0);
    assert_eq!(u64::from_le_bytes(output[0..8].try_into().unwrap()), mapped);
    assert_eq!(
        u64::from_le_bytes(output[8..16].try_into().unwrap()),
        mapped + 0x1_0000
    );
    assert_eq!(
        u64::from_le_bytes(output[16..24].try_into().unwrap()),
        0x20_0000
    );
    assert_eq!(u32::from_le_bytes(output[24..28].try_into().unwrap()), 0x33);
    assert_eq!(u32::from_le_bytes(output[28..32].try_into().unwrap()), 12);
    assert_eq!(u32::from_le_bytes(output[32..36].try_into().unwrap()), 0x12);
    assert_eq!(&output[36..42], b"direct");
    assert_eq!(&output[42..], &[0; 30]);
}

#[cfg(target_os = "macos")]
#[test]
fn allocates_direct_memory_from_real_free_ranges() {
    let mut state = ProbeDispatchState::default();
    let mut occupied = u64::MAX;
    assert_eq!(
        allocate_direct_memory(
            [
                0x4000,
                0x8000,
                0x4000,
                0x4000,
                3,
                (&raw mut occupied).addr() as u64,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(occupied, 0x4000);
    let mut output = u64::MAX;
    let result = allocate_direct_memory(
        [
            0,
            0x1_0000,
            0x4000,
            0x4000,
            12,
            (&raw mut output).addr() as u64,
        ],
        &mut state,
    );
    assert_eq!(result, 0);
    assert_eq!(output, 0);
}

#[cfg(target_os = "macos")]
#[test]
fn release_direct_memory_splits_the_owned_physical_range() {
    let mut state = ProbeDispatchState::default();
    let mut allocated = u64::MAX;
    assert_eq!(
        allocate_direct_memory(
            [
                0,
                0xc000,
                0xc000,
                0x4000,
                7,
                (&raw mut allocated).addr() as u64,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(allocated, 0);
    assert_eq!(
        kernel_release_direct_memory([0x4000, 0x4000, 0, 0, 0, 0], &mut state),
        0
    );
    let mut replacement = u64::MAX;
    assert_eq!(
        allocate_direct_memory(
            [
                0x4000,
                0x8000,
                0x4000,
                0x4000,
                9,
                (&raw mut replacement).addr() as u64,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(replacement, 0x4000);
}

#[cfg(target_os = "macos")]
#[test]
fn direct_memory_mappings_alias_the_same_physical_bytes() {
    use super::{kernel_munmap, map_direct_memory};

    let mut state = ProbeDispatchState::default();
    let mut direct = u64::MAX;
    assert_eq!(
        allocate_direct_memory(
            [
                0,
                0x4000,
                0x4000,
                0x4000,
                12,
                (&raw mut direct).addr() as u64,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(direct, 0);
    let mut first = 0_u64;
    let mut second = 0_u64;
    assert_eq!(
        map_direct_memory(
            [(&raw mut first).addr() as u64, 0x4000, 0x33, 0, 0, 0x4000,],
            &mut state,
        ),
        0
    );
    assert_eq!(
        map_direct_memory(
            [(&raw mut second).addr() as u64, 0x4000, 0x33, 0, 0, 0x4000,],
            &mut state,
        ),
        0
    );
    // SAFETY: both pointers are live read/write mappings created immediately above.
    unsafe {
        (first as *mut u8).write_volatile(0x5a);
        assert_eq!((second as *const u8).read_volatile(), 0x5a);
    }
    assert_eq!(kernel_munmap([first, 0x4000, 0, 0, 0, 0], &mut state), 0);
    assert_eq!(kernel_munmap([second, 0x4000, 0, 0, 0, 0], &mut state), 0);
}

#[cfg(target_os = "macos")]
#[test]
fn direct_memory_abi_honors_fixed_and_hinted_placement_without_overwrite() {
    let mut state = ProbeDispatchState::default();
    let mut direct = u64::MAX;
    assert_eq!(
        allocate_direct_memory(
            [
                0,
                0xc000,
                0xc000,
                0x4000,
                5,
                (&raw mut direct).addr() as u64,
            ],
            &mut state,
        ),
        0
    );
    let base = state.kernel_memory.as_ref().unwrap().virtual_base();
    let mut fixed = base.get() + 4 * 0x4000;
    assert_eq!(
        super::map_direct_memory(
            [
                (&raw mut fixed).addr() as u64,
                0x4000,
                0x33,
                0x10,
                direct,
                0x4000,
            ],
            &mut state,
        ),
        0
    );
    let mut collision = fixed;
    assert_eq!(
        super::map_direct_memory(
            [
                (&raw mut collision).addr() as u64,
                0x4000,
                0x33,
                0x10,
                direct + 0x4000,
                0x4000,
            ],
            &mut state,
        ),
        sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
    );
    let mut hinted = base.get() + 8 * 0x4000;
    let requested_hint = hinted;
    assert_eq!(
        super::map_direct_memory(
            [
                (&raw mut hinted).addr() as u64,
                0x4000,
                0x33,
                0,
                direct + 0x8000,
                0x4000,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(hinted, requested_hint);
    assert_eq!(
        super::kernel_munmap([fixed, 0x4000, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        super::kernel_munmap([hinted, 0x4000, 0, 0, 0, 0], &mut state),
        0
    );
}

#[cfg(target_os = "macos")]
#[test]
fn gen5_direct_memory_v2_and_checked_release_preserve_their_distinct_abi() {
    let mut state = ProbeDispatchState::default();
    assert_eq!(
        super::kernel_checked_release_direct_memory([0, 0x4000, 0, 0, 0, 0], &mut state),
        sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
    );
    let mut direct = u64::MAX;
    assert_eq!(
        allocate_direct_memory(
            [
                0,
                0x4000,
                0x4000,
                0x4000,
                3,
                (&raw mut direct).addr() as u64,
            ],
            &mut state,
        ),
        0
    );
    let mut mapped = 0_u64;
    assert_eq!(
        super::map_direct_memory_2(
            [(&raw mut mapped).addr() as u64, 0x4000, 13, 0x33, 0, direct,],
            0x4000,
            &mut state,
        ),
        0
    );
    let mapping = state
        .kernel_memory
        .as_ref()
        .unwrap()
        .query(remu_core::GuestAddress::new(mapped), false)
        .unwrap();
    assert_eq!(mapping.memory_type, 13);
    assert_eq!(
        super::kernel_checked_release_direct_memory([direct, 0x4000, 0, 0, 0, 0], &mut state),
        0
    );
    assert!(
        state
            .kernel_memory
            .as_ref()
            .unwrap()
            .query(remu_core::GuestAddress::new(mapped), false)
            .is_some(),
        "direct-memory release preserves aliases until explicit munmap"
    );
    assert_eq!(
        super::kernel_checked_release_direct_memory([direct, 0x4000, 0, 0, 0, 0], &mut state),
        sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND)
    );
    assert_eq!(
        super::kernel_release_direct_memory([direct, 0x4000, 0, 0, 0, 0], &mut state),
        0
    );
    assert_eq!(
        super::kernel_munmap([mapped, 0x4000, 0, 0, 0, 0], &mut state),
        0
    );
}

#[test]
fn decodes_gen5_memory_acquire_ranges_and_controls() {
    let packet = [
        agc_pm4(8, AGC_IT_NOP, super::AGC_R_ACQUIRE_MEM),
        0x8000_1234,
        0x3456_789a,
        0x0000_00ab,
        0x7654_3210,
        0x0000_cdef,
        0xabcd_2468,
        0xfff5_4321,
    ];
    let decoded = decode_agc_memory_acquire(packet.as_ptr().addr() as u64);

    assert_eq!(decoded.engine, 1);
    assert_eq!(decoded.cb_db_control, 0x1234);
    assert_eq!(decoded.base_address, 0x00cd_ef76_5432_1000);
    assert_eq!(decoded.size_bytes, 0xab34_5678_9a00);
    assert_eq!(decoded.poll_interval, 0x2468);
    assert_eq!(decoded.gcr_control, 0x5_4321);
}

#[test]
fn decodes_native_32_bit_wait_reference_before_its_control_word() {
    let mut label = 0_u32;
    let label_address = (&raw mut label).addr() as u64;
    let packet = [
        agc_pm4(7, AGC_IT_NOP, super::AGC_R_WAIT_MEM_32),
        low_u32(label_address),
        low_u32(label_address >> 32),
        u32::MAX,
        1,
        0x113,
        0,
    ];

    assert_eq!(
        decode_agc_memory_wait(
            packet.as_ptr().addr() as u64,
            super::AGC_R_WAIT_MEM_32,
            false
        ),
        Some(remu_gpu::GpuMemoryWait {
            address: label_address,
            reference: 1,
            mask: u64::from(u32::MAX),
            compare_function: 3,
            is_64_bit: false,
        })
    );
}

#[test]
fn decodes_native_dma_immediate_fill_without_treating_the_value_as_an_address() {
    let mut destination = [0_u32; 2];
    let destination_address = destination.as_mut_ptr().addr() as u64;
    let control = 1 | (3 << 13) | (3 << 20) | (3 << 25) | (2 << 29) | (1 << 31);
    let packet = [
        agc_pm4(7, super::AGC_IT_DMA_DATA, 0),
        control,
        1,
        0,
        low_u32(destination_address),
        low_u32(destination_address >> 32),
        8,
    ];

    assert_eq!(
        decode_agc_dma_data(packet.as_ptr().addr() as u64, 7, true),
        Some(remu_gpu::GpuMemoryCopy {
            source_address: 1,
            destination_address,
            byte_count: 8,
            source_selection: 2,
            destination_selection: 3,
            cpu_applied: false,
        })
    );
}

#[test]
fn emits_ordered_gen5_release_and_wait_label_operations() {
    let _event_guard = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut label = 0_u64;
    let label_address = (&raw mut label).addr() as u64;
    let commands = [
        agc_pm4(8, AGC_IT_NOP, super::AGC_R_RELEASE_MEM),
        0,
        2 << 16,
        low_u32(label_address),
        low_u32(label_address >> 32),
        0x89ab_cdef,
        0x0123_4567,
        0,
        agc_pm4(9, AGC_IT_NOP, super::AGC_R_WAIT_MEM_64),
        low_u32(label_address),
        low_u32(label_address >> 32),
        u32::MAX,
        u32::MAX,
        0x89ab_cdef,
        0x0123_4567,
        3,
        0,
    ];
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender);
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let mut submission_count = 0;
    super::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut super::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            index_resources: &BTreeMap::new(),
            video_out: &VideoOutState::default(),
            submission_count: &mut submission_count,
            flips: &mut Vec::new(),
            event_writes: &mut Vec::new(),
        },
    )
    .expect("release/wait submission");
    remu_gpu::clear_event_sink();

    assert_eq!(
        receiver.recv().expect("release event"),
        remu_gpu::GpuEvent::MemoryRelease(remu_gpu::GpuMemoryRelease {
            address: label_address,
            value: remu_gpu::GpuMemoryReleaseValue::Immediate64(0x0123_4567_89ab_cdef),
        })
    );
    assert_eq!(
        receiver.recv().expect("wait event"),
        remu_gpu::GpuEvent::MemoryWait(remu_gpu::GpuMemoryWait {
            address: label_address,
            reference: 0x0123_4567_89ab_cdef,
            mask: u64::MAX,
            compare_function: 3,
            is_64_bit: true,
        })
    );
    assert_eq!(submission_count, 1);
    assert!(receiver.try_recv().is_err());
}

#[test]
fn executes_indirect_command_buffer_events_before_following_waits() {
    let _event_guard = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut label = 0_u32;
    let label_address = (&raw mut label).addr() as u64;
    let indirect = [
        agc_pm4(8, AGC_IT_NOP, super::AGC_R_RELEASE_MEM),
        0,
        1 << 16,
        low_u32(label_address),
        low_u32(label_address >> 32),
        1,
        0,
        0,
    ];
    let indirect_address = indirect.as_ptr().addr() as u64;
    let commands = [
        agc_pm4(4, super::AGC_IT_INDIRECT_BUFFER, 0),
        low_u32(indirect_address),
        low_u32(indirect_address >> 32),
        0x0f20_0000 | u32::try_from(indirect.len()).expect("indirect dword count"),
        agc_pm4(7, AGC_IT_NOP, super::AGC_R_WAIT_MEM_32),
        low_u32(label_address),
        low_u32(label_address >> 32),
        u32::MAX,
        1,
        0x13,
        0,
    ];
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender);
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let mut submission_count = 0;
    super::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut super::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            index_resources: &BTreeMap::new(),
            video_out: &VideoOutState::default(),
            submission_count: &mut submission_count,
            flips: &mut Vec::new(),
            event_writes: &mut Vec::new(),
        },
    )
    .expect("indirect release and direct wait submission");
    remu_gpu::clear_event_sink();

    assert_eq!(
        receiver.recv().expect("indirect release event"),
        remu_gpu::GpuEvent::MemoryRelease(remu_gpu::GpuMemoryRelease {
            address: label_address,
            value: remu_gpu::GpuMemoryReleaseValue::Immediate32(1),
        })
    );
    assert_eq!(
        receiver.recv().expect("following wait event"),
        remu_gpu::GpuEvent::MemoryWait(remu_gpu::GpuMemoryWait {
            address: label_address,
            reference: 1,
            mask: u64::from(u32::MAX),
            compare_function: 3,
            is_64_bit: false,
        })
    );
    assert_eq!(submission_count, 1);
    assert!(receiver.try_recv().is_err());
}

#[test]
fn emits_ordered_standard_and_agc_write_data_operations() {
    let _event_guard = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut labels = [0_u32; 4];
    let label_address = labels.as_mut_ptr().addr() as u64;
    let commands = [
        agc_pm4(5, super::AGC_IT_WRITE_DATA, 0),
        5 << 8,
        low_u32(label_address),
        low_u32(label_address >> 32),
        0x1122_3344,
        agc_pm4(6, AGC_IT_NOP, super::AGC_R_WRITE_DATA),
        4 | (1 << 16),
        low_u32(label_address + 8),
        low_u32((label_address + 8) >> 32),
        0x5566_7788,
        0x99aa_bbcc,
    ];
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender);
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let mut submission_count = 0;
    super::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut super::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            index_resources: &BTreeMap::new(),
            video_out: &VideoOutState::default(),
            submission_count: &mut submission_count,
            flips: &mut Vec::new(),
            event_writes: &mut Vec::new(),
        },
    )
    .expect("write-data submission");
    remu_gpu::clear_event_sink();

    assert_eq!(
        receiver.recv().expect("standard write event"),
        remu_gpu::GpuEvent::MemoryWrite(remu_gpu::GpuMemoryWrite {
            address: label_address,
            values: vec![0x1122_3344],
            increment_address: true,
        })
    );
    assert_eq!(
        receiver.recv().expect("AGC write event"),
        remu_gpu::GpuEvent::MemoryWrite(remu_gpu::GpuMemoryWrite {
            address: label_address + 8,
            values: vec![0x5566_7788, 0x99aa_bbcc],
            increment_address: false,
        })
    );
    assert_eq!(submission_count, 1);
    assert!(receiver.try_recv().is_err());
}

#[test]
fn decodes_partial_flush_events_without_inventing_completion_notifications() {
    for (event_type, wait_for) in [
        (0x07, remu_gpu::GpuPipelineStage::ComputeShader),
        (0x0f, remu_gpu::GpuPipelineStage::VertexShader),
        (0x10, remu_gpu::GpuPipelineStage::PixelShader),
    ] {
        let packet = [agc_pm4(2, super::AGC_IT_EVENT_WRITE, 0), 0x400 | event_type];
        assert_eq!(
            super::decode_agc_pipeline_barrier(packet.as_ptr().addr() as u64, 2)
                .expect("supported partial flush"),
            remu_gpu::GpuPipelineBarrier { wait_for }
        );
    }

    // The hardware packet processor uses the Type-3 header length to advance over the
    // packet, while EVENT_INDEX=4 consumes only EVENT_CONTROL. Native Gen5 streams may leave
    // two padding dwords in that packet; they must not turn a valid partial flush into a
    // fatal submission error.
    let padded_partial_flush = [
        agc_pm4(4, super::AGC_IT_EVENT_WRITE, 0),
        0x410,
        0xfeed_face,
        0xcafe_babe,
    ];
    assert_eq!(
        super::decode_agc_pipeline_barrier(padded_partial_flush.as_ptr().addr() as u64, 4)
            .expect("padded partial flush"),
        remu_gpu::GpuPipelineBarrier {
            wait_for: remu_gpu::GpuPipelineStage::PixelShader,
        }
    );

    let wrong_index = [agc_pm4(2, super::AGC_IT_EVENT_WRITE, 0), 0x10];
    assert!(matches!(
        super::decode_agc_pipeline_barrier(wrong_index.as_ptr().addr() as u64, 2),
        Err(super::AgcSubmissionError::PartialFlushEventIndex {
            event_type: 0x10,
            actual: 0,
        })
    ));
    let unsupported = [agc_pm4(2, super::AGC_IT_EVENT_WRITE, 0), 0x416];
    assert!(matches!(
        super::decode_agc_pipeline_barrier(unsupported.as_ptr().addr() as u64, 2),
        Err(super::AgcSubmissionError::UnsupportedEventWrite {
            event_type: 0x16,
            event_index: 4,
            length: 2,
        })
    ));
    let reserved = [agc_pm4(2, super::AGC_IT_EVENT_WRITE, 0), 0x8000_0410];
    assert!(matches!(
        super::decode_agc_pipeline_barrier(reserved.as_ptr().addr() as u64, 2),
        Err(super::AgcSubmissionError::UnsupportedEventWriteControl(
            0x8000_0410
        ))
    ));

    let ordinary_event = [agc_pm4(2, super::AGC_IT_EVENT_WRITE, 0), 0x02];
    let mut graphics = AgcGraphicsState::default();
    let mut notifications = Vec::new();
    super::agc_handle_submission_packet(
        super::AgcSubmissionPacket {
            address: ordinary_event.as_ptr().addr() as u64,
            length: 2,
            opcode: super::AGC_IT_EVENT_WRITE,
            register: 0,
            predicated: false,
        },
        &mut graphics,
        &mut AgcShaderRegistry::default(),
        &BTreeMap::new(),
        &VideoOutState::default(),
        &mut Vec::new(),
        &mut notifications,
        None,
    )
    .expect("ordinary cache event");
    assert!(
        notifications.is_empty(),
        "ordinary EVENT_WRITE must not fabricate a kernel interrupt"
    );

    let query_results = [0_u64; 16];
    let query_address = query_results.as_ptr().addr() as u64;
    let pixel_pipe_stat_dump = [
        agc_pm4(4, super::AGC_IT_EVENT_WRITE, 0),
        0x139,
        low_u32(query_address),
        (query_address >> 32) as u32,
    ];
    assert_eq!(
        super::decode_agc_event_write(pixel_pipe_stat_dump.as_ptr().addr() as u64, 4)
            .expect("addressed pixel-pipe statistic dump"),
        None
    );
    super::agc_handle_submission_packet(
        super::AgcSubmissionPacket {
            address: pixel_pipe_stat_dump.as_ptr().addr() as u64,
            length: 4,
            opcode: super::AGC_IT_EVENT_WRITE,
            register: 0,
            predicated: false,
        },
        &mut graphics,
        &mut AgcShaderRegistry::default(),
        &BTreeMap::new(),
        &VideoOutState::default(),
        &mut Vec::new(),
        &mut notifications,
        None,
    )
    .expect("pixel-pipe statistic dump");
    assert!(
        notifications.is_empty(),
        "occlusion query dumps are not kernel completion interrupts"
    );

    let malformed_query = [
        agc_pm4(4, super::AGC_IT_EVENT_WRITE, 0),
        0x239,
        low_u32(query_address),
        (query_address >> 32) as u32,
    ];
    assert!(matches!(
        super::decode_agc_event_write(malformed_query.as_ptr().addr() as u64, 4),
        Err(super::AgcSubmissionError::UnsupportedEventWrite {
            event_type: 0x39,
            event_index: 2,
            length: 4,
        })
    ));
}

#[test]
fn applies_native_gen5_indirect_register_packets_to_each_register_space() {
    let context_registers = [0x318, 0x1234_5678, 0x390, 0x5a, u32::MAX, 0xfeed_face];
    let shader_registers = [
        0x2000_020c,
        0x7654_3210,
        0x2000_020d,
        0x44,
        0x2000_0213,
        0x388,
    ];
    let user_config_registers = [0x4000_0242, 4];
    let packet = |opcode, registers: &[u32]| {
        let address = registers.as_ptr().addr() as u64;
        [
            agc_pm4(5, opcode, 0),
            low_u32(address),
            (address >> 32) as u32,
            0x8000_0000,
            u32::try_from(registers.len() / 2).expect("register pair count"),
        ]
    };
    let mut graphics = AgcGraphicsState::default();

    for (opcode, registers) in [
        (AGC_IT_SET_CONTEXT_REG_INDIRECT, &context_registers[..]),
        (AGC_IT_SET_SH_REG_INDIRECT, &shader_registers[..]),
        (AGC_IT_SET_UCONFIG_REG_INDIRECT, &user_config_registers[..]),
    ] {
        let command = packet(opcode, registers);
        agc_apply_submission_registers(command.as_ptr().addr() as u64, 5, opcode, 0, &mut graphics)
            .expect("native indirect register packet");
    }

    assert_eq!(graphics.context_registers.get(&0x318), Some(&0x1234_5678));
    assert_eq!(graphics.context_registers.get(&0x390), Some(&0x5a));
    assert!(!graphics.context_registers.contains_key(&u32::MAX));
    assert_eq!(
        graphics
            .shader_registers
            .get(&remu_gpu::COMPUTE_PGM_LO_REGISTER),
        Some(&0x7654_3210)
    );
    assert_eq!(
        graphics
            .shader_registers
            .get(&remu_gpu::COMPUTE_PGM_HI_REGISTER),
        Some(&0x44)
    );
    assert_eq!(graphics.user_config_registers.get(&0x242), Some(&4));
}

#[test]
fn skips_predicated_packets_when_boolean_predication_is_false() {
    let predicate = 1_u64;
    let predicate_address = std::ptr::from_ref(&predicate).addr() as u64;
    let commands = [
        agc_pm4(4, super::AGC_IT_SET_PREDICATION, 0),
        3 << 16,
        low_u32(predicate_address) & !0x0f,
        low_u32(predicate_address >> 32),
        agc_pm4(3, super::AGC_IT_SET_SH_REG, 0) | 1,
        0x00c,
        0xdead_beef,
        agc_pm4(3, super::AGC_IT_SET_SH_REG, 0),
        0x00d,
        0x1234_5678,
    ];
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let mut submission_count = 0;

    super::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut super::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            index_resources: &BTreeMap::new(),
            video_out: &VideoOutState::default(),
            submission_count: &mut submission_count,
            flips: &mut Vec::new(),
            event_writes: &mut Vec::new(),
        },
    )
    .expect("predicated submission");

    assert!(graphics.predicate_skip);
    assert!(!graphics.shader_registers.contains_key(&0x00c));
    assert_eq!(graphics.shader_registers.get(&0x00d), Some(&0x1234_5678));
}

#[test]
fn cond_exec_skips_exactly_its_zero_guarded_dword_range() {
    let predicate = 0_u32;
    let predicate_address = std::ptr::from_ref(&predicate).addr() as u64;
    let commands = [
        agc_pm4(5, super::AGC_IT_COND_EXEC, 0),
        low_u32(predicate_address) & !3,
        low_u32(predicate_address >> 32),
        0,
        3,
        agc_pm4(3, super::AGC_IT_SET_SH_REG, 0),
        0x00c,
        0xdead_beef,
        agc_pm4(3, super::AGC_IT_SET_SH_REG, 0),
        0x00d,
        0x1234_5678,
    ];
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let mut submission_count = 0;

    super::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut super::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            index_resources: &BTreeMap::new(),
            video_out: &VideoOutState::default(),
            submission_count: &mut submission_count,
            flips: &mut Vec::new(),
            event_writes: &mut Vec::new(),
        },
    )
    .expect("conditional submission");

    assert!(!graphics.shader_registers.contains_key(&0x00c));
    assert_eq!(graphics.shader_registers.get(&0x00d), Some(&0x1234_5678));
}

#[test]
fn native_gen5_branch_executes_only_the_selected_indirect_buffer() {
    let comparison_value = 5_u64;
    let comparison_address = std::ptr::from_ref(&comparison_value).addr() as u64;
    let then_commands = [agc_pm4(3, super::AGC_IT_SET_SH_REG, 0), 0x00c, 0x1111_1111];
    let else_commands = [agc_pm4(3, super::AGC_IT_SET_SH_REG, 0), 0x00d, 0x2222_2222];
    let then_address = then_commands.as_ptr().addr() as u64;
    let else_address = else_commands.as_ptr().addr() as u64;
    super::agc_command_abi::track_graphics_allocation(then_address, 3);
    let commands = [
        agc_pm4(14, super::AGC_IT_INDIRECT_BUFFER, 0),
        2 | (3 << 8),
        low_u32(comparison_address) & !7,
        low_u32(comparison_address >> 32),
        u32::MAX,
        u32::MAX,
        5,
        0,
        low_u32(then_address) & !3,
        low_u32(then_address >> 32),
        3,
        low_u32(else_address) & !3,
        low_u32(else_address >> 32),
        3,
    ];
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let mut submission_count = 0;

    super::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut super::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            index_resources: &BTreeMap::new(),
            video_out: &VideoOutState::default(),
            submission_count: &mut submission_count,
            flips: &mut Vec::new(),
            event_writes: &mut Vec::new(),
        },
    )
    .expect("branched submission");

    assert_eq!(graphics.shader_registers.get(&0x00c), Some(&0x1111_1111));
    assert!(!graphics.shader_registers.contains_key(&0x00d));
    assert_eq!(
        super::agc_command_abi::take_recorded_graphics_prefix(then_address + 12),
        None,
        "an already executed native branch target must not be reconstructed before an adjacent tail"
    );
}

#[test]
fn unconditional_gen5_branch_does_not_read_its_queue_label() {
    let then_commands = [agc_pm4(3, super::AGC_IT_SET_SH_REG, 0), 0x00c, 0x8765_4321];
    let then_address = then_commands.as_ptr().addr() as u64;
    let commands = [
        agc_pm4(14, super::AGC_IT_INDIRECT_BUFFER, 0),
        1,
        0x0000_1000,
        0,
        0,
        0,
        0,
        0,
        low_u32(then_address) & !3,
        low_u32(then_address >> 32),
        3,
        0,
        0,
        0,
    ];
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let mut submission_count = 0;

    super::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut super::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            index_resources: &BTreeMap::new(),
            video_out: &VideoOutState::default(),
            submission_count: &mut submission_count,
            flips: &mut Vec::new(),
            event_writes: &mut Vec::new(),
        },
    )
    .expect("unconditional branch");

    assert_eq!(graphics.shader_registers.get(&0x00c), Some(&0x8765_4321));
}

#[test]
fn applies_native_gen5_indexed_uconfig_packet() {
    let packet = [
        agc_pm4(3, AGC_IT_SET_UCONFIG_REG_INDEX, 0),
        0x2000_0243,
        0x400 | 1 | (2 << 6),
    ];
    let mut graphics = AgcGraphicsState::default();

    agc_apply_submission_registers(
        packet.as_ptr().addr() as u64,
        3,
        AGC_IT_SET_UCONFIG_REG_INDEX,
        0,
        &mut graphics,
    )
    .expect("native indexed user-config register packet");

    assert_eq!(graphics.user_config_registers.get(&0x243), Some(&0x481));
    assert_eq!(graphics.index_size, 1);
}

#[test]
fn clear_state_resets_complete_queue_state_without_discarding_residency() {
    let address = 0x0070_1234_5600_u64;
    let target = remu_gpu::GpuColorTarget {
        slot: 0,
        address,
        view: 0,
        info: 10 << 2,
        attrib: 0,
        attrib2: ((1_280 - 1) << 14) | (720 - 1),
        attrib3: (27 << 14) | (1 << 24),
    };
    let packet = [agc_pm4(2, super::AGC_IT_CLEAR_STATE, 0), 0x0f];
    let mut graphics = AgcGraphicsState::default();
    graphics.context_registers.insert(0x318, 1);
    graphics.shader_registers.insert(0x20c, 2);
    graphics.user_config_registers.insert(0x242, 3);
    graphics.index_buffer_address = 0x1000;
    graphics.index_resource_address = 0x2000;
    graphics.index_buffer_count = 9;
    graphics.index_size = 1;
    graphics.instance_count = 7;
    graphics.draw_index_offset = 5;
    graphics.compute_indirect_args_address = 0x3000;
    graphics.emitted_draws = 11;
    graphics.known_color_targets.insert(address, target);

    super::agc_handle_submission_packet(
        super::AgcSubmissionPacket {
            address: packet.as_ptr().addr() as u64,
            length: 2,
            opcode: super::AGC_IT_CLEAR_STATE,
            register: 0,
            predicated: false,
        },
        &mut graphics,
        &mut AgcShaderRegistry::default(),
        &BTreeMap::new(),
        &VideoOutState::default(),
        &mut Vec::new(),
        &mut Vec::new(),
        None,
    )
    .expect("valid CLEAR_STATE packet");

    assert!(graphics.context_registers.is_empty());
    assert!(graphics.shader_registers.is_empty());
    assert!(graphics.user_config_registers.is_empty());
    assert_eq!(graphics.index_buffer_address, 0);
    assert_eq!(graphics.index_resource_address, 0);
    assert_eq!(graphics.index_buffer_count, 0);
    assert_eq!(graphics.index_size, 0);
    assert_eq!(graphics.instance_count, 1);
    assert_eq!(graphics.draw_index_offset, 0);
    assert_eq!(graphics.compute_indirect_args_address, 0);
    assert_eq!(graphics.emitted_draws, 11);
    assert_eq!(graphics.known_color_targets.get(&address), Some(&target));
}

#[test]
fn clear_state_rejects_non_native_layouts() {
    let invalid_control = [agc_pm4(2, super::AGC_IT_CLEAR_STATE, 0), 0x10];
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let index_resources = BTreeMap::new();
    let video_out = VideoOutState::default();
    let mut flips = Vec::new();
    let mut event_writes = Vec::new();

    let mut invoke = |address, length, graphics: &mut AgcGraphicsState| {
        super::agc_handle_submission_packet(
            super::AgcSubmissionPacket {
                address,
                length,
                opcode: super::AGC_IT_CLEAR_STATE,
                register: 0,
                predicated: false,
            },
            graphics,
            &mut shaders,
            &index_resources,
            &video_out,
            &mut flips,
            &mut event_writes,
            None,
        )
    };
    assert!(matches!(
        invoke(invalid_control.as_ptr().addr() as u64, 2, &mut graphics),
        Err(super::AgcSubmissionError::UnsupportedClearStateControl(
            0x10
        ))
    ));
    assert!(matches!(
        invoke(invalid_control.as_ptr().addr() as u64, 3, &mut graphics),
        Err(super::AgcSubmissionError::ClearStatePacketLength { actual: 3 })
    ));
}

#[test]
fn distinguishes_absent_disabled_programmed_and_malformed_color_target_state() {
    let address = 0x0070_1234_5600_u64;
    let mut registers = BTreeMap::from([
        (super::AGC_CB_COLOR0_BASE, low_u32(address >> 8)),
        (super::AGC_CB_COLOR0_BASE_EXT, low_u32(address >> 40)),
        (super::AGC_CB_COLOR0_VIEW, 0),
        (super::AGC_CB_COLOR0_INFO, 10 << 2),
        (super::AGC_CB_COLOR0_ATTRIB, 0),
        (
            super::AGC_CB_COLOR0_ATTRIB2,
            ((1_280 - 1) << 14) | (720 - 1),
        ),
        (super::AGC_CB_COLOR0_ATTRIB3, (27 << 14) | (1 << 24)),
    ]);

    let AgcColorTargetState::Programmed(targets) = agc_color_target_state(&registers) else {
        panic!("a complete target remains enabled when CB_TARGET_MASK is absent");
    };
    assert_eq!(targets.len(), 1);
    assert_eq!(targets[0].address, address);

    registers.remove(&super::AGC_CB_COLOR0_VIEW);
    assert!(matches!(
        agc_color_target_state(&registers),
        AgcColorTargetState::Malformed { slot: 0 }
    ));

    registers.insert(super::AGC_CB_TARGET_MASK, 0);
    assert!(matches!(
        agc_color_target_state(&registers),
        AgcColorTargetState::Disabled
    ));
}

#[test]
fn explicit_zero_target_mask_does_not_enable_targetless_scanout_fallback() {
    let target = super::AgcVideoOutTarget {
        address: 0x0070_1234_5600,
        attributes: VideoOutBufferAttributes {
            width: 1_280,
            height: 720,
            pitch: 1_280,
            pixel_format: 0x8000_0000_2200_0000,
            tiling_mode: 0,
        },
    };
    let mut graphics = AgcGraphicsState::default();
    graphics
        .context_registers
        .insert(super::AGC_CB_TARGET_MASK, 0);
    let (_, color_target_absent) = super::agc_prepare_draw_color_targets(&mut graphics);
    assert!(!color_target_absent);

    let mut draw = targetless_scanout_test_draw();
    draw.pixel_shader_evaluation = Some(Arc::new(remu_shader::ShaderEvaluation {
        initial_scalar_registers: Vec::new().into(),
        scalar_registers: Vec::new().into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: Vec::new(),
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    }));
    super::agc_resolve_targetless_scanout(
        &mut graphics,
        &mut draw,
        color_target_absent,
        Some(target),
    );
    assert_eq!(draw.scanout_target, None);
}

#[test]
fn scanout_lookahead_is_submission_bounded_and_not_limited_by_channel_capacity() {
    let address = 0x0070_1234_5600_u64;
    let video_out = VideoOutState {
        is_open: true,
        display_buffers: vec![address],
        display_buffer_attributes: vec![Some(VideoOutBufferAttributes {
            width: 1_280,
            height: 720,
            pitch: 1_280,
            pixel_format: 0x8000_0000_2200_0000,
            tiling_mode: 0,
        })],
        ..VideoOutState::default()
    };
    let flip = [agc_pm4(6, AGC_IT_NOP, super::AGC_R_FLIP), 1, 0, 0, 0, 0];
    let draw_packet = || super::AgcSubmissionPacket {
        address: 0,
        length: 3,
        opcode: super::AGC_IT_DRAW_INDEX_AUTO,
        register: 0,
        predicated: false,
    };
    let mut packets = vec![draw_packet()];
    packets.extend((0..(remu_gpu::GPU_EVENT_QUEUE_CAPACITY + 17)).map(|_| {
        super::AgcSubmissionPacket {
            address: 0,
            length: 2,
            opcode: AGC_IT_NOP,
            register: 0,
            predicated: false,
        }
    }));
    let second_draw = packets.len();
    packets.push(draw_packet());
    packets.push(super::AgcSubmissionPacket {
        address: flip.as_ptr().addr() as u64,
        length: 6,
        opcode: AGC_IT_NOP,
        register: super::AGC_R_FLIP,
        predicated: false,
    });

    let future = super::agc_future_scanout_targets(&packets, &video_out);
    assert_eq!(future[0].map(|target| target.address), Some(address));
    assert_eq!(
        future[second_draw].map(|target| target.address),
        Some(address)
    );

    let boundary_packets = [
        draw_packet(),
        super::AgcSubmissionPacket {
            address: 0,
            length: 2,
            opcode: super::AGC_IT_CLEAR_STATE,
            register: 0,
            predicated: false,
        },
        draw_packet(),
        *packets.last().expect("flip packet"),
    ];
    let boundary_future = super::agc_future_scanout_targets(&boundary_packets, &video_out);
    assert_eq!(boundary_future[0], None);
    assert_eq!(
        boundary_future[2].map(|target| target.address),
        Some(address)
    );
}

#[test]
fn executes_native_draw_index_2_with_embedded_index_base_and_count() {
    let _event_guard = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut indices = [0_u16, 1, 2, 2, 3, 0];
    let index_address = indices.as_ptr().addr() as u64;
    let index_resource = 0x0070_1234_5600_u64;
    let commands = [
        agc_pm4(6, super::AGC_IT_DRAW_INDEX_2, 0),
        6,
        low_u32(index_address),
        low_u32(index_address >> 32),
        6,
        2,
    ];
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender);
    let mut graphics = AgcGraphicsState::default();
    let mut shaders = AgcShaderRegistry::default();
    let mut submission_count = 0;

    super::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut super::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            index_resources: &BTreeMap::from([(index_address, index_resource)]),
            video_out: &VideoOutState::default(),
            submission_count: &mut submission_count,
            flips: &mut Vec::new(),
            event_writes: &mut Vec::new(),
        },
    )
    .expect("native DRAW_INDEX_2 submission");
    remu_gpu::clear_event_sink();

    let remu_gpu::GpuEvent::IndexedDraw(draw) = receiver.recv().expect("indexed draw event") else {
        panic!("DRAW_INDEX_2 emitted a non-draw event");
    };
    assert!(draw.indexed);
    assert_eq!(draw.index_buffer_address, index_address);
    assert_eq!(draw.index_resource_address, index_resource);
    assert_eq!(draw.index_buffer_count, 6);
    assert_eq!(draw.index_offset, 0);
    assert_eq!(draw.index_count, 6);
    indices.fill(u16::MAX);
    assert_eq!(
        draw.index_snapshot.as_deref(),
        Some([0, 1, 2, 2, 3, 0].as_slice())
    );
    assert_eq!(draw.draw_flags, 2);
    assert_eq!(submission_count, 1);
    assert!(receiver.try_recv().is_err());
}

#[test]
fn targetless_scanout_resolution_requires_complete_unambiguous_draw_state() {
    let target = super::AgcVideoOutTarget {
        address: 0x0070_1234_5600,
        attributes: VideoOutBufferAttributes {
            width: 1_280,
            height: 720,
            pitch: 1_280,
            pixel_format: 0x8000_0000_2200_0000,
            tiling_mode: 0,
        },
    };
    let mut graphics = AgcGraphicsState::default();
    let mut draw = targetless_scanout_test_draw();

    super::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, Some(target));
    assert_eq!(
        draw.scanout_target, None,
        "failed resource evaluation is not guessed"
    );

    draw.pixel_shader_evaluation = Some(Arc::new(remu_shader::ShaderEvaluation {
        initial_scalar_registers: Vec::new().into(),
        scalar_registers: Vec::new().into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: Vec::new(),
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    }));
    super::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, Some(target));
    assert_eq!(
        draw.scanout_target, None,
        "a constant export with no sampled source is not a display composite"
    );

    draw.pixel_shader_evaluation = Some(Arc::new(remu_shader::ShaderEvaluation {
        initial_scalar_registers: Vec::new().into(),
        scalar_registers: Vec::new().into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: vec![remu_shader::ImageBinding {
            pc: 0,
            opcode: remu_shader::Opcode::Known("ImageSample"),
            control: remu_shader::ImageControl {
                dmask: 0xf,
                vector_address: 0,
                address_registers: Vec::new(),
                vector_data: 0,
                scalar_resource: 0,
                scalar_sampler: 0,
                dimension: 2,
                is_array: false,
                glc: false,
                slc: false,
                a16: false,
                d16: false,
            },
            resource_descriptor: vec![1, 0, 0, 0].into(),
            sampler_descriptor: vec![0; 4].into(),
            mip_level: None,
        }],
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    }));
    super::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, Some(target));
    assert_eq!(
        draw.scanout_target.map(|scanout| scanout.address),
        Some(target.address),
        "a side-effect-free sampled draw may be the display composite"
    );

    draw.scanout_target = None;
    graphics
        .context_registers
        .insert(super::AGC_DB_DEPTH_CONTROL, 0x2);
    super::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, Some(target));
    assert_eq!(
        draw.scanout_target.map(|scanout| scanout.address),
        Some(target.address),
        "stale depth control without a bound depth surface does not suppress scanout"
    );

    draw.scanout_target = None;
    graphics.context_registers.insert(super::AGC_DB_Z_INFO, 3);
    graphics
        .context_registers
        .insert(super::AGC_DB_DEPTH_SIZE_XY, 1_279 | (719 << 16));
    graphics
        .context_registers
        .insert(super::AGC_DB_Z_READ_BASE, 0x0012_3456);
    super::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, Some(target));
    assert_eq!(
        draw.scanout_target, None,
        "a complete active depth target is not inferred as scanout"
    );

    graphics.context_registers.clear();
    super::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, None);
    assert_eq!(
        draw.scanout_target, None,
        "association never crosses a missing flip"
    );

    let pixel = draw.pixel_shader.as_mut().expect("test pixel shader");
    let mut instructions = pixel.program.instructions.clone();
    instructions.push(remu_shader::Instruction {
        pc: 8,
        encoding: remu_shader::ShaderEncoding::Exp,
        opcode: remu_shader::Opcode::EXPORT,
        words: vec![0, 0],
        sources: Vec::new(),
        destinations: Vec::new(),
        control: Some(remu_shader::InstructionControl::Export(
            remu_shader::ExportControl {
                target: 1,
                enable_mask: 0xf,
                compressed: false,
                done: true,
                valid_mask: false,
            },
        )),
    });
    pixel.program = Arc::new(remu_shader::Program::new(
        pixel.program.address,
        instructions,
    ));
    draw.context_registers
        .iter_mut()
        .find(|register| register.offset == super::AGC_CB_SHADER_MASK)
        .expect("test shader mask")
        .value = 0xff;
    super::agc_resolve_targetless_scanout(&mut graphics, &mut draw, true, Some(target));
    assert_eq!(
        draw.scanout_target, None,
        "multiple color exports are ambiguous"
    );
}

#[test]
fn constant_targetless_draw_does_not_overwrite_ordered_scanout() {
    let _event_guard = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let address = 0x0070_1234_5600_u64;
    let target = remu_gpu::GpuColorTarget {
        slot: 0,
        address,
        view: 0,
        info: 10 << 2,
        attrib: 0,
        attrib2: ((1_280 - 1) << 14) | (720 - 1),
        attrib3: (27 << 14) | (1 << 24),
    };
    let pixel_code = AlignedComputeProgram([0; 4]);
    let pixel_address = pixel_code.0.as_ptr().addr() as u64;
    let (mut graphics, mut shaders) = targetless_scanout_test_shader_state(pixel_address, target);
    let video_out = targetless_scanout_test_video(address);
    let (commands, expected_acquire) = targetless_barrier_acquire_flip_commands();
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender);

    let mut submission_count = 0;
    let mut flips = Vec::new();
    let mut event_writes = Vec::new();
    super::agc_parse_submission(
        commands.as_ptr().addr() as u64,
        u32::try_from(commands.len()).expect("command dword count"),
        &mut super::AgcSubmissionContext {
            graphics: &mut graphics,
            shaders: &mut shaders,
            index_resources: &BTreeMap::new(),
            video_out: &video_out,
            submission_count: &mut submission_count,
            flips: &mut flips,
            event_writes: &mut event_writes,
        },
    )
    .expect("ordered targetless submission");
    remu_gpu::clear_event_sink();

    let remu_gpu::GpuEvent::IndexedDraw(draw) = receiver.recv().expect("constant draw event")
    else {
        panic!("draw must precede flip");
    };
    assert_eq!(
        draw.scanout_target, None,
        "an evaluated constant export is not redirected into VideoOut"
    );
    assert_eq!(
        receiver.recv().expect("pixel partial flush"),
        remu_gpu::GpuEvent::PipelineBarrier(remu_gpu::GpuPipelineBarrier {
            wait_for: remu_gpu::GpuPipelineStage::PixelShader,
        })
    );
    assert_eq!(
        receiver.recv().expect("memory acquire"),
        remu_gpu::GpuEvent::MemoryAcquire(expected_acquire)
    );
    let remu_gpu::GpuEvent::Flip {
        address: flipped_address,
        width,
        height,
        ..
    } = receiver.recv().expect("ordered flip event")
    else {
        panic!("flip must follow draw");
    };
    assert_eq!(flipped_address, address);
    assert_eq!((width, height), (1_280, 720));
    assert_eq!(flips, vec![(1, 0, 0x1234)]);
    assert!(receiver.try_recv().is_err());
}

#[test]
fn submits_descriptor_wrapped_acb_work_on_an_owner_scoped_compute_queue() {
    let _event_guard = GPU_EVENT_SINK_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let program = AlignedComputeProgram([0, 0, 0, 0]);
    let program_address = program.0.as_ptr().addr() as u64;
    let (_, shaders) = compute_graphics_state(program_address);
    let register_pairs = [
        remu_gpu::COMPUTE_PGM_LO_REGISTER,
        low_u32(program_address >> 8),
        remu_gpu::COMPUTE_PGM_HI_REGISTER,
        ((program_address >> 40) & 0xff) as u32,
        remu_gpu::COMPUTE_PGM_RSRC2_REGISTER,
        0,
        remu_gpu::COMPUTE_NUM_THREAD_X_REGISTER,
        64,
        remu_gpu::COMPUTE_NUM_THREAD_Y_REGISTER,
        1,
        remu_gpu::COMPUTE_NUM_THREAD_Z_REGISTER,
        1,
    ];
    let registers_address = register_pairs.as_ptr().addr() as u64;
    let commands = [
        agc_pm4(2, AGC_IT_NOP, AGC_R_ACB_RESET),
        0,
        agc_pm4(5, AGC_IT_SET_SH_REG_INDIRECT, 0),
        low_u32(registers_address),
        (registers_address >> 32) as u32,
        0x8000_0000,
        6,
        agc_pm4(5, AGC_IT_DISPATCH_DIRECT, 0),
        3,
        2,
        1,
        0x8045,
    ];
    let command_address = commands.as_ptr().addr() as u64;
    let descriptor = [
        low_u32(command_address),
        (command_address >> 32) as u32,
        u32::try_from(commands.len()).expect("command dword count"),
        0,
        0x5533_ccaa,
    ];
    let packet = AgcSubmitPacket {
        address: descriptor.as_ptr().addr() as u64,
        dword_count: 5,
        flags: [0; 4],
    };
    let (sender, receiver) = remu_gpu::gpu_event_channel();
    remu_gpu::install_event_sink(sender);
    let mut state = ProbeDispatchState {
        agc_shader_registry: shaders,
        ..ProbeDispatchState::default()
    };

    let result = agc_driver_submit_acb(
        [7, (&raw const packet).addr() as u64, 0, 0, 0, 0],
        &mut state,
    );
    let submission_event = receiver.recv_timeout(std::time::Duration::from_secs(1));
    remu_gpu::clear_event_sink();

    assert_eq!(result, 0);
    let remu_gpu::GpuEvent::QueueSubmission { queue_id, events } =
        submission_event.expect("compute queue submission")
    else {
        panic!("ACB did not preserve its compute queue");
    };
    assert_eq!(queue_id, 8);
    let [
        remu_gpu::GpuEvent::ComputeDispatch(dispatch),
        remu_gpu::GpuEvent::SubmissionBoundary(1),
    ] = events.as_slice()
    else {
        panic!("ACB submission did not retain compute dispatch and completion order");
    };
    assert_eq!(dispatch.source, remu_gpu::ComputeDispatchSource::Direct);
    assert_eq!(
        dispatch.launch.grid,
        remu_gpu::ComputeGrid::Threadgroups([3, 2, 1])
    );
    assert_eq!(dispatch.shader_address, program_address);
    assert!(state.agc_graphics.shader_registers.is_empty());
    assert_eq!(
        state.agc_compute_queues.get(&7).and_then(|queue| queue
            .shader_registers
            .get(&remu_gpu::COMPUTE_PGM_LO_REGISTER)),
        Some(&low_u32(program_address >> 8))
    );
    assert_eq!(
        state
            .pending_agc_completions
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![]
    );
    assert_eq!(
        state
            .pending_agc_user_interrupts
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![1]
    );
}

#[test]
fn accepts_zero_extent_indirect_compute_without_emitting_gpu_work() {
    let program = AlignedComputeProgram([0; 4]);
    let program_address = program.0.as_ptr().addr() as u64;
    let (mut graphics, mut shaders) = compute_graphics_state(program_address);
    let dimensions = AlignedIndirectArguments([0, 8, 1, 0, 0, 0, 0, 0]);
    graphics.compute_indirect_args_address = dimensions.0.as_ptr().addr() as u64;
    let packet = [agc_pm4(3, AGC_IT_DISPATCH_INDIRECT, 0), 0, 0x8041];
    let mut flips = Vec::new();
    let mut event_writes = Vec::new();

    let (result, gpu_events) = super::capture_agc_gpu_events(|| {
        super::agc_handle_submission_packet(
            super::AgcSubmissionPacket {
                address: packet.as_ptr().addr() as u64,
                length: 3,
                opcode: AGC_IT_DISPATCH_INDIRECT,
                register: 0,
                predicated: false,
            },
            &mut graphics,
            &mut shaders,
            &BTreeMap::new(),
            &VideoOutState::default(),
            &mut flips,
            &mut event_writes,
            None,
        )
    });

    result.expect("zero-extent indirect dispatch is valid empty work");
    assert!(
        gpu_events.is_empty(),
        "empty compute must not reach the backend"
    );
    assert_eq!(graphics.emitted_compute_dispatches, 1);
}

#[test]
fn decodes_and_evaluates_a_complete_direct_compute_dispatch_snapshot() {
    let program = AlignedComputeProgram([0xbeef_cafe, 0x1234_5678, 0, 0]);
    let program_address = program.0.as_ptr().addr() as u64;
    let (mut graphics, mut shaders) = compute_graphics_state(program_address);
    let packet = [agc_pm4(5, AGC_IT_DISPATCH_DIRECT, 0), 10, 4, 2, 0x8045];

    let dispatch = agc_decode_compute_dispatch(
        packet.as_ptr().addr() as u64,
        5,
        AGC_IT_DISPATCH_DIRECT,
        &mut graphics,
        &mut shaders,
    )
    .expect("complete direct compute dispatch");

    assert_eq!(dispatch.sequence, 1);
    assert_eq!(dispatch.source, remu_gpu::ComputeDispatchSource::Direct);
    assert_eq!(
        dispatch.launch.grid,
        remu_gpu::ComputeGrid::Threadgroups([10, 4, 2])
    );
    assert_eq!(dispatch.launch.threads_per_group, [64, 2, 1]);
    assert_eq!(dispatch.launch.wave_lane_count, 32);
    assert_eq!(dispatch.shader_address, program_address);
    assert_eq!(dispatch.shader_prefix, program.0);
    let evaluation = dispatch
        .shader_evaluation
        .as_ref()
        .expect("compute evaluation");
    assert_eq!(evaluation.initial_scalar_registers[0], 0x1234_5678);
    assert_eq!(evaluation.initial_scalar_registers[1], 0x9abc_def0);
    let system = evaluation
        .compute_system_registers
        .expect("compute system registers");
    assert_eq!(system.work_group_x, Some(2));
    assert_eq!(system.work_group_y, None);
    assert_eq!(system.work_group_z, Some(3));
    assert_eq!(system.thread_group_size, Some(4));
}

#[test]
fn resolves_base_relative_and_absolute_compute_indirect_dimensions() {
    let program = AlignedComputeProgram([0, 0, 0, 0]);
    let program_address = program.0.as_ptr().addr() as u64;
    let (mut graphics, mut shaders) = compute_graphics_state(program_address);
    let arguments = AlignedIndirectArguments([0, 0, 7, 3, 1, 11, 5, 2]);
    let arguments_address = arguments.0.as_ptr().addr() as u64;
    let set_base = [
        agc_pm4(4, AGC_IT_SET_BASE, 0) | (1 << 1),
        1,
        low_u32(arguments_address),
        (arguments_address >> 32) as u32,
    ];
    agc_apply_set_base(set_base.as_ptr().addr() as u64, 4, &mut graphics)
        .expect("compute indirect base");
    assert_eq!(graphics.compute_indirect_args_address, arguments_address);

    let relative = [agc_pm4(3, AGC_IT_DISPATCH_INDIRECT, 0), 8, 0x8041];
    let relative_dispatch = agc_decode_compute_dispatch(
        relative.as_ptr().addr() as u64,
        3,
        AGC_IT_DISPATCH_INDIRECT,
        &mut graphics,
        &mut shaders,
    )
    .expect("base-relative compute dispatch");
    assert_eq!(
        relative_dispatch.source,
        remu_gpu::ComputeDispatchSource::Indirect {
            dimensions_address: arguments_address + 8,
        }
    );
    assert_eq!(
        relative_dispatch.launch.grid,
        remu_gpu::ComputeGrid::Threadgroups([7, 3, 1])
    );

    let absolute_address = arguments_address + 20;
    let absolute = [
        agc_pm4(4, AGC_IT_DISPATCH_INDIRECT, 0),
        low_u32(absolute_address),
        (absolute_address >> 32) as u32,
        0x8041,
    ];
    let absolute_dispatch = agc_decode_compute_dispatch(
        absolute.as_ptr().addr() as u64,
        4,
        AGC_IT_DISPATCH_INDIRECT,
        &mut graphics,
        &mut shaders,
    )
    .expect("absolute compute dispatch");
    assert_eq!(
        absolute_dispatch.source,
        remu_gpu::ComputeDispatchSource::Indirect {
            dimensions_address: absolute_address,
        }
    );
    assert_eq!(
        absolute_dispatch.launch.grid,
        remu_gpu::ComputeGrid::Threadgroups([11, 5, 2])
    );
}

#[test]
fn rejects_indirect_compute_without_a_compute_set_base_packet() {
    let program = AlignedComputeProgram([0; 4]);
    let (mut graphics, mut shaders) = compute_graphics_state(program.0.as_ptr().addr() as u64);
    let packet = [agc_pm4(3, AGC_IT_DISPATCH_INDIRECT, 0), 0x80, 0x8041];
    assert!(matches!(
        agc_decode_compute_dispatch(
            packet.as_ptr().addr() as u64,
            3,
            AGC_IT_DISPATCH_INDIRECT,
            &mut graphics,
            &mut shaders,
        ),
        Err(AgcComputeDispatchError::MissingIndirectBase)
    ));
}

#[test]
fn builds_the_three_argument_gen5_index_buffer_packets() {
    let mut command_buffer = vec![0_u8; 0x40];
    let mut commands = vec![0_u32; 16];
    let indices = [0_u32, 1, 2, 1, 2, 3];
    let command_buffer_address =
        u64::try_from(command_buffer.as_mut_ptr().addr()).expect("command buffer address");
    let commands_address =
        u64::try_from(commands.as_mut_ptr().addr()).expect("command storage address");
    let index_buffer_address =
        u64::try_from(indices.as_ptr().addr()).expect("index buffer address");
    write_guest_u64(command_buffer_address + 0x10, commands_address);
    write_guest_u64(command_buffer_address + 0x18, commands_address + 64);
    write_guest_u32(command_buffer_address + 0x30, 0);

    let mut state = ProbeDispatchState::default();
    let command = agc_dcb_set_index_buffer(
        [command_buffer_address, index_buffer_address, 16, 0, 0, 0],
        &mut state,
    );

    assert_eq!(command, commands_address);
    assert_eq!(commands[0], agc_pm4(3, AGC_IT_INDEX_BASE, 0));
    assert_eq!(commands[1], low_u32(index_buffer_address));
    assert_eq!(commands[2], low_u32(index_buffer_address >> 32));
    assert_eq!(commands[3], agc_pm4(2, AGC_IT_INDEX_BUFFER_SIZE, 0));
    assert_eq!(commands[4], 16);
    assert_eq!(
        read_guest_u64(command_buffer_address + 0x10),
        commands_address + 20
    );
    assert_eq!(
        state.agc_index_resources.get(&index_buffer_address),
        Some(&index_buffer_address)
    );
}

#[test]
fn builds_the_gen5_instanced_auto_draw_packets() {
    let mut command_buffer = vec![0_u8; 0x40];
    let mut commands = vec![0_u32; 16];
    let command_buffer_address =
        u64::try_from(command_buffer.as_mut_ptr().addr()).expect("command buffer address");
    let commands_address =
        u64::try_from(commands.as_mut_ptr().addr()).expect("command storage address");
    write_guest_u64(command_buffer_address + 0x10, commands_address);
    write_guest_u64(command_buffer_address + 0x18, commands_address + 64);
    write_guest_u32(command_buffer_address + 0x30, 0);

    let instances = agc_dcb_set_num_instances([command_buffer_address, 3, 0, 0, 0, 0]);
    let draw = agc_dcb_draw_index_auto([command_buffer_address, 3, 0x4000_0000, 0, 0, 0]);

    assert_eq!(instances, commands_address);
    assert_eq!(commands[0], agc_pm4(2, AGC_IT_NUM_INSTANCES, 0));
    assert_eq!(commands[1], 3);
    assert_eq!(draw, commands_address + 8);
    assert_eq!(commands[2], agc_pm4(7, AGC_IT_NOP, AGC_R_DRAW_INDEX_AUTO));
    assert_eq!(commands[3], 3);
    assert_eq!(&commands[4..9], &[0; 5]);
    assert_eq!(
        read_guest_u64(command_buffer_address + 0x10),
        commands_address + 36
    );
}
