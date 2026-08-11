use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::{
    AGC_IT_DISPATCH_DIRECT, AGC_IT_DISPATCH_INDIRECT, AGC_IT_DRAW_INDEX_INDIRECT,
    AGC_IT_DRAW_INDIRECT, AGC_IT_INDEX_BASE, AGC_IT_INDEX_BUFFER_SIZE, AGC_IT_NOP,
    AGC_IT_NUM_INSTANCES, AGC_IT_SET_BASE, AGC_IT_SET_CONTEXT_REG_INDIRECT, AGC_IT_SET_SH_REG,
    AGC_IT_SET_SH_REG_INDIRECT, AGC_IT_SET_UCONFIG_REG, AGC_IT_SET_UCONFIG_REG_INDEX,
    AGC_IT_SET_UCONFIG_REG_INDIRECT, AGC_R_ACB_RESET, AGC_R_DRAW_INDEX_AUTO,
    AGC_SHADER_INPUT_SEMANTICS_COUNT_OFFSET, AGC_SHADER_INPUT_SEMANTICS_OFFSET,
    AGC_SHADER_OUTPUT_SEMANTICS_COUNT_OFFSET, AGC_SHADER_OUTPUT_SEMANTICS_OFFSET,
    AGC_SPI_PS_INPUT_CNTL_0, AGC_VGT_INDEX_TYPE, AMPR_INVALID_FILE_ID,
    AUDIO_OUT2_CONTEXT_MEMORY_BASE_SIZE, AUDIO_OUT2_CONTEXT_MEMORY_QUEUE_STRIDE,
    AUDIO_OUT2_CONTEXT_PARAM_SIZE, AUDIO_OUT2_DEFAULT_GRAIN_FRAMES, AUDIO_OUT2_DEFAULT_QUEUE_DEPTH,
    AUDIO_OUT2_PORT_TYPE_VIBRATION, AgcColorTargetState, AgcComputeDispatchError, AgcDmaData,
    AgcGraphicsState, AgcShaderRegistry, AmprCommandBufferState, AprResolveOutputs,
    BATCH_MAP_OPERATION_MAP_DIRECT, DEFAULT_FLIP_INTERVAL_MICROSECONDS, DIRENT_BLOCK_SIZE,
    DIRENT_HEADER_SIZE, GPU_EVENT_SINK_TEST, GuestExitCallback, KERNEL_FILES, KernelDirectory,
    KernelFileState, ProbeDispatchState, PthreadAttrState, SAVE_DATA_ERROR_BUSY,
    SAVE_DATA_ERROR_EXISTS, SAVE_DATA_ERROR_NOT_FOUND, SAVE_DATA_ERROR_RESOURCE_BUSY,
    SAVE_DATA_ERROR_RESOURCE_FULL, SAVE_DATA_ERROR_RESOURCE_INVALID, SAVE_DATA_PARAM_SIZE,
    SCE_AUDIO_OUT2_ERROR_NOT_READY, SCE_KERNEL_ERROR_INVALID_ARGUMENT, SCE_KERNEL_ERROR_NOT_FOUND,
    SCE_KERNEL_ERROR_NOT_IMPLEMENTED, SCE_KERNEL_ERROR_OUT_OF_MEMORY, SHADER_BINARY_TYPE_GS_BACK,
    SHADER_BINARY_TYPE_GS_FRONT, SHADER_HEADER_CODE_OFFSET, SHADER_HEADER_NUM_SH_REGISTERS_OFFSET,
    SHADER_HEADER_SH_REGISTERS_OFFSET, SHADER_HEADER_SIZE, SHADER_HEADER_TYPE_OFFSET,
    SPI_SHADER_PGM_LO_ES, SaveDataMount, SceAudioOut2Attribute, SceAudioOut2ContextParam,
    SceAudioOut2Pcm, SceAudioOut2PortParam, SceAudioOut2PortState, VIDEO_OUT_FLIP_INTERVAL,
    VideoOutBufferAttributes, VideoOutState, agc_apply_set_base, agc_apply_submission_registers,
    agc_color_target_state, agc_create_interpolant_mapping, agc_dcb_draw_index_auto,
    agc_dcb_set_index_buffer, agc_dcb_set_num_instances, agc_decode_compute_dispatch,
    agc_driver_submit_acb, agc_fuse_shader_halves, agc_get_fused_shader_size, agc_pm4,
    allocate_direct_memory, allocate_pthread_thread_object,
    ampr_command_buffer_write_address_on_completion, ampr_command_buffer_write_kernel_event_queue,
    ampr_resolve_file_paths, ampr_resolve_paths_core, ampr_submit_command_buffer,
    available_direct_memory_size, available_flexible_memory_size, civil_from_unix_seconds,
    clone_guest_tls, create_pthread_attr, decode_agc_dma_data, decode_agc_event_write,
    decode_agc_memory_acquire, decode_agc_memory_wait, decode_audio_out2_attribute,
    decode_audio_out2_channel_format, destroy_pthread_attr, discover_runtime_module_paths,
    dispatch_audio_out2_import, dispatch_guest_numeric_parser, dispatch_probe_import,
    draw_required_vertex_record_count, flip_interval_microseconds, guest_clock_gettime, guest_free,
    guest_global_locale, guest_kernel_convert_localtime_to_utc,
    guest_kernel_convert_utc_to_localtime, guest_kernel_fstat, guest_kernel_ftruncate,
    guest_kernel_gettimeofday, guest_kernel_lseek, guest_kernel_read, guest_posix_file_result,
    guest_posix_gettimeofday, guest_pthread_once, guest_qsort, guest_tolower_table,
    guest_toupper_table, guest_wmemchr, guest_wmemcmp, guest_wmemcpy, guest_wmemmove,
    join_guest_path, kernel_add_ampr_event, kernel_batch_map, kernel_release_direct_memory,
    kernel_stat_payload, kernel_virtual_query, low_f32, low_u32, pace_video_out_flip,
    play_go_get_chunk_id, play_go_get_install_speed, play_go_get_locus, play_go_initialize,
    play_go_open, process_clock_nanos, process_time_since_nanos, pthread_attr_get_affinity,
    pthread_attr_get_stack, pthread_attr_set_affinity, pthread_attr_set_detach_state,
    pthread_attr_set_schedule_param, pthread_attr_set_stack_size, read_guest_u16, read_guest_u32,
    read_guest_u64, resolve_guest_path, resolve_probe_guest_path, rtc_get_current_clock_local_time,
    rtc_get_current_tick, rtc_get_tick, rtc_set_tick, sanitize_save_segment, save_data_commit,
    save_data_create_transaction_resource, save_data_delete, save_data_delete_transaction_resource,
    save_data_dir_name_search, save_data_mount3, save_data_pattern_matches, save_data_prepare,
    save_data_save_icon, save_data_set_param, save_data_terminate, save_data_umount2,
    sce_kernel_error, set_current_guest_thread_pointer, system_service_param_get_int,
    take_exit_callbacks, tls_dtv_entry_offset, unix_days_from_civil, video_out_open,
    video_out_submit_flip, write_guest_u8, write_guest_u16, write_guest_u32, write_guest_u64,
    write_import_stub,
};
mod agc_draw;
mod agc_packets;
mod agc_submission;
mod ampr;
mod audio_out2;
mod av_player;
mod filesystem;
mod guest_libc;
mod imports;
mod memory;
mod playgo;
mod services;
mod shader;
mod synchronization;
mod time;

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

unsafe extern "C" fn once_test_callback() {
    ONCE_CALLBACKS.fetch_add(1, Ordering::Relaxed);
}

#[cfg(target_arch = "x86_64")]
unsafe extern "C" fn pthread_exit_boundary_test_entry(argument0: u64, argument1: u64) -> u64 {
    super::guest_pthread_exit(argument0 + argument1);
}

static DELIVERED_EXCEPTION_RSP: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" fn capture_guest_exception(exception: i32, context: *mut u8) {
    DELIVERED_EXCEPTION.store(exception.cast_unsigned(), Ordering::SeqCst);
    // SAFETY: the delivery test passes a complete 0x500-byte synthesized Orbis ucontext.
    let rsp = unsafe { std::ptr::read_unaligned(context.add(0xf8).cast::<u64>()) };
    DELIVERED_EXCEPTION_RSP.store(usize::try_from(rsp).unwrap(), Ordering::SeqCst);
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
        indirect_arguments_address: None,
        instance_count: 1,
        index_buffer_address: 0,
        index_resource_address: 0,
        index_buffer_count: 0,
        index_size: 0,
        index_offset: 0,
        base_vertex: 0,
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
        }]
        .into(),
        shader_registers: Arc::from([]),
        user_config_registers: Arc::from([]),
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

fn wait_for_guest_condition_waiters(condition: u64, expected: usize) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        let waiting = {
            let state = super::guest_sync()
                .0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // Registry is keyed by control-block handle; `condition` may be the slot address.
            let handle = super::try_read_process_guest_u64(condition).unwrap_or(condition);
            state
                .conditions
                .get(&handle)
                .or_else(|| state.conditions.get(&condition))
                .map_or(0, |entry| {
                    entry
                        .state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .waiters
                        .len()
                })
        };
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
