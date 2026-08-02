// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

mod agc_acb;
mod agc_api;
mod agc_command_abi;
mod agc_dcb;
mod agc_draw;
mod agc_driver;
mod agc_flip;
#[allow(dead_code)]
mod agc_frontend;
mod agc_register_defaults;
mod agc_registers;
mod agc_state;
mod agc_submission;
mod ampr;
mod ampr_api;
mod audio;
mod audio_out2;
mod audio_out2_abi;
mod av;
mod av_player;
mod ces;
mod dialog;
mod guest_files;
mod guest_libc;
// The fiber ABI switches x86-64 register state, so it exists only on that architecture. On any
// other host the NIDs fall through to the unimplemented path rather than to a stub that lies.
#[cfg(target_arch = "x86_64")]
mod fiber;
mod guest_sync;
mod kernel_api;
#[cfg(target_os = "macos")]
mod kernel_memory;
mod link;
mod memory;
mod nid_names;
mod nid_table;
mod np;
mod np_api;
mod pad;
mod playgo;
mod pthread;
mod random;
mod rtld_atexit;
mod sha_ni;
mod save_data;
mod shader_cache;
mod shader_eval;
mod tls;
mod unwind;
mod video_out;
mod videodec2;

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
#[cfg(target_arch = "x86_64")]
use std::ffi::CString;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::Child;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use remu_core::GuestAddress;
#[cfg(target_arch = "x86_64")]
use remu_memory::GUEST_EXECUTE;

#[cfg(test)]
static GPU_EVENT_SINK_TEST: Mutex<()> = Mutex::new(());
pub(crate) use agc_api::{
    AGC_CB_COLOR_REGISTER_STRIDE, AGC_CB_COLOR0_ATTRIB, AGC_CB_COLOR0_ATTRIB2,
    AGC_CB_COLOR0_ATTRIB3, AGC_CB_COLOR0_BASE, AGC_CB_COLOR0_BASE_EXT, AGC_CB_COLOR0_INFO,
    AGC_CB_COLOR0_VIEW, AGC_CB_SHADER_MASK, AGC_CB_TARGET_MASK, AGC_COLOR_TARGET_COUNT,
    AGC_COMPUTE_USER_DATA_0, AGC_DB_DEPTH_CONTROL, AGC_DB_DEPTH_SIZE_XY, AGC_DB_RENDER_CONTROL,
    AGC_DB_Z_INFO, AGC_DB_Z_READ_BASE, AGC_DB_Z_READ_BASE_HI, AGC_DB_Z_WRITE_BASE,
    AGC_DB_Z_WRITE_BASE_HI, AGC_IT_CLEAR_STATE, AGC_IT_COND_EXEC, AGC_IT_DISPATCH_DIRECT,
    AGC_IT_DISPATCH_INDIRECT, AGC_IT_DMA_DATA, AGC_IT_DRAW_INDEX_2, AGC_IT_DRAW_INDEX_AUTO,
    AGC_IT_DRAW_INDEX_INDIRECT, AGC_IT_DRAW_INDEX_INDIRECT_MULTI, AGC_IT_DRAW_INDEX_OFFSET,
    AGC_IT_DRAW_INDIRECT, AGC_IT_EVENT_WRITE, AGC_IT_INDEX_BASE, AGC_IT_INDEX_BUFFER_SIZE,
    AGC_IT_INDEX_TYPE, AGC_IT_INDIRECT_BUFFER, AGC_IT_NOP, AGC_IT_NUM_INSTANCES,
    AGC_IT_PFP_SYNC_ME, AGC_IT_RELEASE_MEM, AGC_IT_SET_BASE, AGC_IT_SET_CONTEXT_REG,
    AGC_IT_SET_CONTEXT_REG_INDIRECT, AGC_IT_SET_PREDICATION, AGC_IT_SET_SH_REG,
    AGC_IT_SET_SH_REG_INDIRECT, AGC_IT_SET_UCONFIG_REG, AGC_IT_SET_UCONFIG_REG_INDEX,
    AGC_IT_SET_UCONFIG_REG_INDIRECT, AGC_IT_WAIT_REG_MEM, AGC_IT_WRITE_DATA, AGC_R_ACB_RESET,
    AGC_R_ACQUIRE_MEM, AGC_R_CX_REGS_INDIRECT, AGC_R_DMA_DATA, AGC_R_DRAW_INDEX_AUTO,
    AGC_R_DRAW_RESET, AGC_R_FLIP, AGC_R_RELEASE_MEM, AGC_R_SH_REGS_INDIRECT,
    AGC_R_UC_REGS_INDIRECT, AGC_R_WAIT_MEM_32, AGC_R_WAIT_MEM_64, AGC_R_WRITE_DATA,
    AGC_SHADER_SH_REGISTER_COUNT_OFFSET, AGC_SHADER_SH_REGISTERS_OFFSET,
    AGC_SHADER_SPECIALS_OFFSET, AGC_SHADER_TYPE_OFFSET, AGC_SPI_PS_INPUT_CNTL_0,
    AGC_SPI_SHADER_PGM_HI_ES, AGC_SPI_SHADER_PGM_HI_PS, AGC_SPI_SHADER_PGM_LO_ES,
    AGC_SPI_SHADER_PGM_LO_PS, AGC_VGT_INDEX_TYPE, AGC_VGT_PRIMITIVE_TYPE,
    agc_add_indirect_patch_registers, agc_allocate_command_dwords, agc_cb_nop, agc_cb_release_mem,
    agc_cb_set_sh_register_range_direct, agc_create_interpolant_mapping,
    agc_create_primitive_state, agc_create_shader, agc_dcb_acquire_mem, agc_dcb_dma_data,
    agc_dcb_event_write, agc_dcb_reset_queue, agc_dcb_set_registers_indirect, agc_dcb_wait_reg_mem,
    agc_dma_data_patch_destination, agc_fuse_shader_halves, agc_get_data_packet_payload_address,
    agc_get_fused_shader_size, agc_packet_dword_count, agc_pm4, agc_register_defaults,
    agc_release_mem_patch_address, agc_set_indirect_patch_address, agc_wait_reg_mem_patch_address,
};
#[cfg(test)]
pub(crate) use agc_api::{
    AGC_SHADER_FILE_HEADER, AGC_SHADER_INPUT_SEMANTICS_COUNT_OFFSET,
    AGC_SHADER_INPUT_SEMANTICS_OFFSET, AGC_SHADER_OUTPUT_SEMANTICS_COUNT_OFFSET,
    AGC_SHADER_OUTPUT_SEMANTICS_OFFSET, AGC_SHADER_SIZE_OFFSET, AGC_SHADER_VERSION,
    SHADER_BINARY_TYPE_GS_BACK, SHADER_BINARY_TYPE_GS_FRONT, SHADER_HEADER_CODE_OFFSET,
    SHADER_HEADER_NUM_SH_REGISTERS_OFFSET, SHADER_HEADER_SH_REGISTERS_OFFSET, SHADER_HEADER_SIZE,
    SHADER_HEADER_TYPE_OFFSET, SPI_SHADER_PGM_LO_ES,
};
pub(crate) use agc_draw::{
    agc_apply_set_base, agc_decode_compute_dispatch, agc_emit_draw,
    agc_emit_draw_with_indirect_arguments, agc_emit_indexed_draw, agc_image_binding_address,
};
#[cfg(test)]
pub(crate) use agc_draw::{agc_prepare_draw_color_targets, agc_resolve_targetless_scanout};
pub(crate) use agc_flip::{
    AgcVideoOutTarget, agc_dcb_draw_index_auto, agc_dcb_draw_index_offset, agc_dcb_set_flip,
    agc_dcb_set_index_buffer, agc_dcb_set_index_size, agc_dcb_set_num_instances,
    agc_submit_ordered_video_out_flip, agc_video_out_target, agc_video_out_target_for_address,
    queue_agc_event_write, queue_video_out_flip, refresh_completed_video_out_flips,
    request_agc_compute_submission_completion, request_agc_submission_completion,
    video_out_add_flip_event, video_out_add_vblank_event, video_out_register_buffers2,
};
pub(crate) use agc_registers::{
    AgcColorTargetState, AgcComputeDispatchError, AgcSubmissionError,
    agc_apply_submission_registers, agc_color_target_state, agc_has_active_depth_state,
    agc_program_address, agc_register_snapshot, agc_shader_code_fingerprint,
    agc_shader_prefix_snapshot, agc_shader_snapshot,
};
#[cfg(test)]
pub(crate) use agc_submission::{
    AgcEventWrite, AgcSubmissionPacket, agc_handle_submission_packet, agc_parse_submission,
    decode_agc_dma_data, decode_agc_event_write, decode_agc_memory_acquire, decode_agc_memory_wait,
};
pub(crate) use agc_submission::{
    AgcSubmissionContext, agc_driver_submit_acb, agc_driver_submit_dcb,
    agc_driver_submit_multi_acbs, agc_driver_submit_multi_dcbs, agc_memory_shadow_overlaps,
    agc_parse_graphics_submission, agc_read_mapped_u64, apply_agc_memory_shadow,
    begin_agc_memory_shadow,
};
#[cfg(test)]
pub(crate) use agc_submission::{agc_future_scanout_targets, decode_agc_pipeline_barrier};
pub(crate) use ampr_api::{
    AMPR_COMMAND_RECORD_SIZE, AprResolveOutputs, ampr_apr_command_buffer_constructor,
    ampr_command_buffer_clear_buffer, ampr_command_buffer_constructor,
    ampr_command_buffer_destructor, ampr_command_buffer_get_current_offset,
    ampr_command_buffer_get_size, ampr_command_buffer_read_file_ordered, ampr_command_buffer_reset,
    ampr_command_buffer_set_buffer, ampr_command_buffer_write_address_on_completion,
    ampr_command_buffer_write_kernel_event_queue, ampr_command_order, ampr_get_file_size,
    ampr_get_file_stat, ampr_measure_command_size, ampr_resolve_file_paths,
    ampr_resolve_paths_core, ampr_submit_command_buffer, ampr_submit_command_buffer_ordered,
    ampr_wait_command_buffer,
};
#[cfg(test)]
pub(crate) use ampr_api::{
    AMPR_INVALID_FILE_ID, ampr_command_buffer_record_read, ampr_read_file_to_guest,
    ampr_submit_command_buffer_two_phase_with, join_guest_path,
};
pub(crate) use audio_out2::dispatch_audio_out2_import;
#[cfg(test)]
pub(crate) use audio_out2_abi::audio_out2_grain_duration;
pub(crate) use audio_out2_abi::{
    AUDIO_OUT2_CONTEXT_MEMORY_BASE_SIZE, AUDIO_OUT2_CONTEXT_MEMORY_QUEUE_STRIDE,
    AUDIO_OUT2_CONTEXT_PARAM_SIZE, AUDIO_OUT2_DEFAULT_GRAIN_FRAMES, AUDIO_OUT2_DEFAULT_QUEUE_DEPTH,
    AUDIO_OUT2_MAX_QUEUE_DEPTH, AUDIO_OUT2_PORT_STATE_SIZE, AUDIO_OUT2_PORT_TYPE_VIBRATION,
    AUDIO_OUT2_SPEAKER_INFO_SIZE, AudioOut2ContextState, AudioOut2PortRuntimeState, AudioOut2State,
    SCE_AUDIO_OUT2_ERROR_NOT_READY, SceAudioOut2Attribute, SceAudioOut2ContextParam,
    SceAudioOut2Pcm, SceAudioOut2PortParam, SceAudioOut2PortState, audio_out2_queue_wait_duration,
    audio_out2_update_context_queue, decode_audio_out2_attribute, decode_audio_out2_channel_format,
    read_audio_out2_context_param, read_audio_out2_port_param, trace_audio_out2_context_memory,
    trace_audio_out2_mastering_memory, trace_audio_out2_memory, trace_audio_out2_pcm,
    write_audio_out2_context_param, write_audio_out2_port_state,
};
#[cfg(all(test, target_os = "macos", target_arch = "x86_64"))]
pub(crate) use av_player::av_player_nv12_layout;
#[cfg(test)]
pub(crate) use av_player::{
    AV_PLAYER_ERROR_INVALID_PARAMS, AV_PLAYER_ERROR_OPERATION_FAILED, AV_PLAYER_STREAM_TYPE_AUDIO,
    AV_PLAYER_STREAM_TYPE_VIDEO, av_player_resolve_source, av_player_tool,
    av_player_video_catch_up_target, av_player_video_frame_is_due, av_player_write_stream_info,
    av_player_write_stream_info_ex, av_player_write_video_frame_info,
};
pub(crate) use av_player::{
    av_player_add_source, av_player_add_source_ex, av_player_change_stream, av_player_close,
    av_player_current_time, av_player_get_audio_data, av_player_get_stream_info,
    av_player_get_video_data, av_player_init, av_player_init_ex, av_player_is_active,
    av_player_jump_to_time, av_player_pause, av_player_post_init, av_player_resume,
    av_player_set_av_sync_mode, av_player_set_log_callback, av_player_set_looping,
    av_player_set_trick_speed, av_player_start, av_player_stop, av_player_stream_count,
    av_player_validate_handle, invoke_av_player_event_callback, invoke_av_player_event_callbacks,
    invoke_av_player_texture_deallocations, prepare_av_player_video_buffers,
    take_av_player_calling_thread_callbacks, take_av_player_controller_callbacks,
};
#[cfg(test)]
pub(crate) use dialog::{
    COMMON_DIALOG_ERROR_BUSY, COMMON_DIALOG_ERROR_DIALOG_ALREADY_INITIALIZED,
    COMMON_DIALOG_ERROR_NOT_SYSTEM_INITIALIZED, COMMON_DIALOG_STATUS_INITIALIZED,
};
pub(crate) use dialog::{
    common_dialog_initialize, error_dialog_close, error_dialog_get_result, error_dialog_get_status,
    error_dialog_initialize, error_dialog_open, error_dialog_terminate, error_dialog_update_status,
    message_dialog_close, message_dialog_get_result, message_dialog_get_status,
    message_dialog_initialize, message_dialog_open, message_dialog_terminate,
    message_dialog_update_status, publish_host_toast, save_data_dialog_close,
    save_data_dialog_get_result, save_data_dialog_get_status, save_data_dialog_initialize,
    save_data_dialog_is_ready_to_display, save_data_dialog_open, save_data_dialog_terminate,
    save_data_dialog_update_status,
};
#[cfg(any(test, not(unix)))]
pub(crate) use guest_files::kernel_stat_payload;
#[cfg(test)]
pub(crate) use guest_files::{
    DIRENT_BLOCK_SIZE, DIRENT_HEADER_SIZE, ImmutableAppPathCache, KernelDirectory,
    resolve_immutable_app_path,
};
pub(crate) use guest_files::{
    KERNEL_FILES, KernelFileState, guest_fclose, guest_fopen, guest_fread, guest_fseek,
    guest_ftell, guest_kernel_close, guest_kernel_fstat, guest_kernel_ftruncate,
    guest_kernel_getdents, guest_kernel_lseek, guest_kernel_mkdir, guest_kernel_open,
    guest_kernel_read, guest_kernel_rmdir, guest_kernel_stat, guest_kernel_write,
    kernel_stat_payload_from_metadata, resolve_guest_path, resolve_path_beneath,
    resolve_probe_guest_path,
};
pub(crate) use guest_libc::{
    dispatch_guest_numeric_parser, guest_ctype_table, guest_errno_address, guest_memcpy,
    guest_memmove, guest_memset, guest_mspace_calloc, guest_mspace_create, guest_mspace_destroy,
    guest_mspace_free, guest_mspace_is_heap_empty, guest_mspace_malloc, guest_mspace_memalign,
    guest_mspace_posix_memalign, guest_mspace_realloc, guest_mspace_usable_size,
    guest_net_errno_address, guest_tolower_table, guest_toupper_table,
};
#[cfg(test)]
pub(crate) use guest_sync::{
    EVENT_FLAG_WAIT_OR, GuestConditionState, GuestEventFlagState, GuestSemaphoreState,
    dispatch_guest_sync_import, guest_condition_destroy, guest_condition_init,
    guest_condition_notify, guest_condition_wait, guest_condition_wait_with_interrupt,
    guest_event_flag_wait, guest_mutex_destroy, guest_mutex_entry, guest_mutex_init,
    guest_pthread_once, guest_rwlock_destroy, guest_rwlock_init, guest_semaphore_wait,
};
#[cfg(any(test, target_arch = "x86_64"))]
pub(crate) use guest_sync::{
    GUEST_SYNC, GuestSyncState, guest_mutex_lock, guest_mutex_unlock, guest_rwlock_lock,
    guest_rwlock_unlock,
};
pub(crate) use guest_sync::{dispatch_guest_sync_import_at_import, guest_sync, is_guest_sync_import};
#[cfg(test)]
pub(crate) use memory::BATCH_MAP_OPERATION_MAP_DIRECT;
pub(crate) use memory::{
    allocate_direct_memory, allocate_main_direct_memory, available_direct_memory_size,
    available_flexible_memory_size, direct_memory_query, kernel_batch_map, kernel_batch_map_2,
    kernel_checked_release_direct_memory, kernel_mprotect, kernel_mtypeprotect, kernel_munmap,
    kernel_release_direct_memory, kernel_reserve_virtual_range, kernel_set_virtual_range_name,
    kernel_virtual_query, map_direct_memory, map_direct_memory_2, map_named_flexible_memory,
};
#[cfg(target_os = "macos")]
pub(crate) use memory::map_flexible_memory_core;
#[cfg(all(test, target_os = "macos", target_arch = "x86_64"))]
pub(crate) use memory::{host_protection_from_guest, process_mapping_contains};
pub use nid_table::{known_nid_name, nid_display_name};
pub(crate) use np_api::{
    np_game_intent_get_property_string, np_game_intent_initialize, np_game_intent_receive,
    np_trophy_abort_handle, np_trophy_create_context, np_trophy_create_handle,
    np_trophy_destroy_context, np_trophy_destroy_handle, np_trophy_get_game_info,
    np_trophy_get_trophy_info_array, np_trophy_register_context,
    np_trophy_register_unlock_callback, np_trophy_unregister_unlock_callback,
    np_universal_data_abort_handle, np_universal_data_array_set_string,
    np_universal_data_create_context, np_universal_data_create_event,
    np_universal_data_create_handle, np_universal_data_create_property_array,
    np_universal_data_create_property_object, np_universal_data_destroy_context,
    np_universal_data_destroy_event, np_universal_data_destroy_handle,
    np_universal_data_destroy_property_array, np_universal_data_initialize,
    np_universal_data_object_set_array, np_universal_data_object_set_i32,
    np_universal_data_object_set_string, np_universal_data_post_event,
    np_universal_data_register_context, np_universal_data_terminate,
};
#[cfg(target_arch = "x86_64")]
pub(crate) use pad::reset_pad_runtime;
#[cfg(test)]
pub(crate) use pad::{
    SCE_PAD_ERROR_ALREADY_OPENED, SCE_PAD_ERROR_DEVICE_NO_HANDLE, SCE_PAD_ERROR_INVALID_HANDLE,
    SCE_PAD_ERROR_NOT_INITIALIZED,
};
pub(crate) use pad::{
    dispatch_pad_import, pad_boolean_operation, pad_close, pad_device_class_parse_data,
    pad_get_controller_information, pad_get_extended_information, pad_get_handle,
    pad_get_trigger_effect_state, pad_handle_operation, pad_initialize, pad_open,
    pad_pointer_operation, pad_read, pad_read_state, pad_set_motion_sensor_state,
    pad_set_vibration_mode,
};
pub(crate) use playgo::{
    play_go_accept_request, play_go_close, play_go_get_chunk_id, play_go_get_eta,
    play_go_get_install_speed, play_go_get_language_mask, play_go_get_locus, play_go_get_progress,
    play_go_initialize, play_go_open, play_go_terminate,
};
#[cfg(test)]
pub(crate) use pthread::take_current_thread_specific_destructors;
pub(crate) use pthread::{
    create_pthread_attr, create_pthread_mutex_attr, destroy_pthread_attr,
    destroy_pthread_mutex_attr, pthread_attr_get, pthread_attr_get_affinity,
    pthread_attr_get_schedule_param, pthread_attr_get_stack, pthread_attr_get_stack_address,
    pthread_attr_get_stack_size, pthread_attr_set_affinity, pthread_attr_set_detach_state,
    pthread_attr_set_inherit_schedule, pthread_attr_set_schedule_param,
    pthread_attr_set_schedule_policy, pthread_attr_set_stack_size, pthread_attr_state,
    pthread_condition_attr_destroy, pthread_condition_attr_init, pthread_get_schedule_param,
    pthread_get_specific, pthread_key_create, pthread_key_delete, pthread_mutex_attr_set_protocol,
    pthread_mutex_attr_set_type, pthread_mutex_attr_state, pthread_rename, pthread_set_affinity,
    pthread_set_priority, pthread_set_schedule_param, pthread_set_specific,
    run_guest_thread_destructors, update_pthread_attr,
};
use remu_shader::GuestMemoryReader;
#[cfg(target_arch = "x86_64")]
pub(crate) use save_data::save_data_root;
#[cfg(test)]
pub(crate) use save_data::{
    SAVE_DATA_ERROR_EXISTS, SAVE_DATA_ERROR_NOT_FOUND, SAVE_DATA_PARAM_SIZE,
    enumerate_save_data_entries, sanitize_save_segment, save_data_pattern_matches,
};
pub(crate) use save_data::{
    current_unix_seconds, read_guest_bytes, save_data_commit,
    save_data_create_transaction_resource, save_data_delete_transaction_resource,
    save_data_dir_name_search, save_data_initialize,
    save_data_mount3, save_data_prepare, save_data_save_icon, save_data_set_param,
    save_data_umount2, write_guest_bytes,
};
use sha1::{Digest, Sha1};
pub(crate) use shader_cache::{
    ShaderEvaluationCache, ShaderEvaluationWorkerRequest, ShaderEvaluationWorkers,
};
#[cfg(all(test, target_os = "macos", target_arch = "x86_64"))]
pub(crate) use shader_cache::{canonical_vertex_record_count, shader_evaluation_user_data_key};
pub(crate) use shader_eval::{
    draw_required_vertex_record_count, dump_draw_shader_ir, evaluate_draw_shader_requests_parallel,
    evaluate_shader_register_snapshot_cached, export_user_data_register,
    shader_control_flow_summary, shader_evaluation_worker_request, snapshot_draw_indices,
    snapshot_small_readonly_global_bindings, trace_agc_shader_draw,
};

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(crate) use link::native_cxx_runtime;
pub use link::{
    GuestTls, ImportResolutionKind, ImportResolutionRecord, ImportResolutionStats, ImportStub,
    LinkError, LinkedImage, ModuleLinkReport, ProbeOutcome, RelocationStats,
};
#[cfg(test)]
pub(crate) use link::{IMPORT_STUB_SIZE_USIZE, discover_runtime_module_paths, write_import_stub};
pub(crate) use link::{
    LinkedGuestModule, MACOS_GUEST_PAGE_ALIGNMENT, PT_TLS, add_address, runtime_address_at,
    subtract_address, write_u64,
};
use tls::{GUEST_TLS_PREFIX_SIZE, GUEST_TLS_TCB_SIZE, tls_dtv_entry_offset};
use unwind::{GuestModuleSegment, GuestUnwindModule};
#[cfg(target_arch = "x86_64")]
use unwind::register_guest_unwind_frames;
#[cfg(test)]
pub(crate) use video_out::{
    DEFAULT_FLIP_INTERVAL_MICROSECONDS, VIDEO_OUT_ERROR_RESOURCE_BUSY, VIDEO_OUT_FLIP_INTERVAL,
    flip_interval_microseconds,
};
pub(crate) use video_out::{
    VIDEO_OUT_ERROR_INVALID_ADDRESS, VIDEO_OUT_ERROR_INVALID_HANDLE, VIDEO_OUT_ERROR_INVALID_INDEX,
    VIDEO_OUT_ERROR_INVALID_VALUE, pace_video_out_flip, video_out_get_output_status,
    video_out_is_flip_pending, video_out_open, video_out_set_buffer_attribute2,
    video_out_set_flip_rate, video_out_submit_flip,
};

#[cfg(target_os = "macos")]
#[link(name = "System")]
unsafe extern "C" {
    fn mach_task_self() -> u32;
    fn mach_vm_read_overwrite(
        task: u32,
        address: u64,
        size: u64,
        output_address: u64,
        output_size: *mut u64,
    ) -> i32;
}

#[cfg(windows)]
#[link(name = "ucrt")]
unsafe extern "C" {
    fn _localtime64_s(tm_dest: *mut libc::tm, source_time: *const i64) -> i32;
}

/// Returns the number of guest pthread workers that are still executing.
///
/// A PS5 process can return from its entry routine after handing long-lived work to pthreads. The
/// graphical launcher uses this count to keep the mapped image and GPU event sink alive until those
/// workers finish or the launch timeout expires.
#[must_use]
pub fn active_guest_thread_count() -> usize {
    PROBE_DISPATCH.get().map_or(0, |state| {
        state
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .thread_handles
            .values()
            .filter(|thread| !thread.is_finished())
            .count()
    })
}

#[derive(Debug, Default)]
struct ProbeDispatchState {
    unimplemented_nids: BTreeSet<String>,
    #[cfg(target_os = "macos")]
    kernel_memory: Option<kernel_memory::KernelMemory>,
    #[cfg(target_os = "macos")]
    process_mapping_start: u64,
    #[cfg(target_os = "macos")]
    process_mapping_end: u64,
    next_thread_unique_id: u64,
    thread_handles: BTreeMap<u64, std::thread::JoinHandle<u64>>,
    thread_attributes: BTreeMap<u64, PthreadAttrState>,
    detached_threads: BTreeSet<u64>,
    pthread_threads: BTreeMap<u64, PthreadThreadObject>,
    pthread_attributes: BTreeMap<u64, PthreadAttrObject>,
    pthread_mutex_attributes: BTreeMap<u64, PthreadMutexAttrObject>,
    pthread_condition_attributes: BTreeSet<u64>,
    pthread_keys: BTreeMap<u32, PthreadKeyState>,
    ampr_files: BTreeMap<u32, PathBuf>,
    ampr_file_handles: BTreeMap<u32, Arc<std::fs::File>>,
    ampr_command_buffers: BTreeMap<u64, AmprCommandBufferState>,
    next_apr_submission_id: u32,
    apr_submissions: BTreeSet<u32>,
    exit_callbacks: Vec<GuestExitCallback>,
    /// Registered guest coredump handler and its requested stack size.
    ///
    /// The console runs this on a dedicated thread when the process faults. REmu records the
    /// registration so the ABI's already-registered and parameter errors are exact; generating a
    /// coredump to invoke it is a separate unimplemented capability.
    coredump_handler: Option<(u64, u64)>,
    process_param_address: u64,
    thread_atexit_count_callback: u64,
    thread_atexit_report_callback: u64,
    /// Per-guest-thread counts `_sceKernelRtldThreadAtexitIncrement` accumulates for CRT teardown.
    thread_atexit_counts: BTreeMap<u64, u64>,
    thread_dtors_callback: u64,
    /// Pending `__cxa_thread_atexit` destructor references keyed by module base.
    ///
    /// `_sceKernelRtldThreadAtexitIncrement` acquires one when the CRT registers a TLS dtor;
    /// `Decrement` releases it. Nonzero counts must keep the module mapped.
    module_tls_destructor_refs: BTreeMap<u64, u64>,
    application_heap_api_address: u64,
    bundled_modules: Vec<LinkedGuestModule>,
    unwind_modules: Vec<GuestUnwindModule>,
    installed_exception_handlers: BTreeMap<i32, u64>,
    next_ajm_context: u32,
    ajm_contexts: BTreeMap<u32, AjmContextState>,
    next_acm_context: u32,
    acm_contexts: BTreeSet<u32>,
    heap_trace_storage: Vec<u64>,
    guest_mspaces: BTreeMap<u64, GuestMspace>,
    json_runtime: JsonRuntimeState,
    av_player: AvPlayerRuntimeState,
    services: RuntimeServiceState,
    agc_primary_register_defaults: u64,
    agc_internal_register_defaults: u64,
    ctype_table: u64,
    tolower_table: u64,
    toupper_table: u64,
    locale_object: u64,
    locale_vtable: u64,
    tls_template: Vec<u8>,
    video_out: VideoOutState,
    game_root: PathBuf,
    np_universal_data: np::UniversalDataState,
    np_game_intent: np::GameIntentState,
    np_trophy: np::TrophyState,
    agc_submission_count: u64,
    agc_graphics: AgcGraphicsState,
    agc_compute_queues: BTreeMap<u32, AgcGraphicsState>,
    agc_shader_registry: AgcShaderRegistry,
    next_equeue: u64,
    kernel_equeues: BTreeSet<u64>,
    kernel_event_registrations: BTreeMap<(u64, i16, u64), u64>,
    pending_events: VecDeque<KernelQueuedEvent>,
    /// Accepted VideoOut flips awaiting retirement of their Metal presentation command.
    pending_video_out_flips: VecDeque<PendingVideoOutFlip>,
    /// Backend completion sequence paired with the AGC queue identifier registered through
    /// `sceAgcDriverAddEqEvent` (graphics is 0, ACB queues use their owner ID).
    pending_agc_completions: VecDeque<(u64, u64)>,
    pending_agc_user_interrupts: VecDeque<u64>,
    agc_completion_telemetry: AgcCompletionTelemetry,
    save_root: PathBuf,
    save_data_initialized: bool,
    save_mounts: BTreeMap<String, SaveDataMount>,
    next_save_transaction_resource: u32,
    save_transaction_resources: BTreeSet<u32>,
    prepared_save_resources: BTreeSet<u32>,
    save_transactions: BTreeMap<u32, SaveDataTransaction>,
    /// Opaque Videodec2 compute queues returned by `sceVideodec2AllocateComputeQueue`.
    videodec2_compute_queues: BTreeSet<u64>,
}

#[derive(Debug)]
struct GuestMspace {
    name: String,
    base: u64,
    end: u64,
    free_ranges: BTreeMap<u64, u64>,
    allocations: BTreeMap<u64, u64>,
}

#[derive(Debug, Default)]
struct JsonRuntimeState {
    allocators: BTreeMap<u64, (u64, u64)>,
    file_buffer_sizes: BTreeMap<u64, u64>,
    initialized_objects: BTreeMap<u64, u64>,
}

#[derive(Debug, Default)]
struct AvPlayerRuntimeState {
    log_callback: Option<(u64, u64)>,
    next_handle: u64,
    players: BTreeMap<u64, AvPlayerInstance>,
    pending_callbacks: Vec<AvPlayerEventCallback>,
    pending_texture_deallocations: Vec<AvPlayerTextureDeallocation>,
}

#[derive(Clone, Copy, Debug)]
struct AvPlayerEventCallback {
    function: u64,
    object: u64,
    event: u64,
    delivery: AvPlayerEventDelivery,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AvPlayerEventDelivery {
    Controller,
    CallingThread,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AvPlayerTextureDeallocation {
    function: u64,
    object: u64,
    address: u64,
}

#[derive(Debug)]
struct AvPlayerDecoder {
    child: Option<Child>,
    frames: Mutex<Receiver<Vec<u8>>>,
    cancelled: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
    progress: Option<Arc<AvPlayerDecoderProgress>>,
}

#[derive(Debug, Default)]
struct AvPlayerDecoderProgress {
    produced_frames: AtomicU64,
    queue_blocked: AtomicBool,
    last_progress_millis: AtomicU64,
}

#[derive(Debug)]
struct AvPlayerInstance {
    _initialization: Vec<u8>,
    allocator_object: u64,
    allocate_texture_callback: u64,
    deallocate_texture_callback: u64,
    event_object: u64,
    event_callback: u64,
    source: Option<String>,
    auto_start: bool,
    width: u32,
    height: u32,
    frames_per_second: f64,
    duration: Duration,
    has_audio: bool,
    /// Whether the source still belongs to an active playback session. Native AvPlayer reports a
    /// newly-added source as active while it is Ready, before Start establishes the media clock.
    /// This must therefore remain distinct from `playing_since` and `paused`.
    active_session: bool,
    paused: bool,
    end_of_stream: bool,
    playing_since: Option<Instant>,
    elapsed: Duration,
    /// Media timestamp and host instant which define the public presentation clock. Audio
    /// delivery owns this anchor when an audio stream exists. Default-sync video-only streams
    /// re-anchor it after every delivered frame so a slow guest cannot complete a movie using
    /// host time alone. None-sync video-only streams retain the independent playback wall clock.
    presentation_anchor_media: Duration,
    presentation_anchor_host: Option<Instant>,
    /// Last value exposed through CurrentTime or used to pace video. Re-anchoring must not make
    /// the public clock move backwards, even when the guest consumes media below real time.
    presentation_clock_reported: Duration,
    /// Whether the selected clock master has delivered a sample since Start or JumpToTime.
    /// Audio sources remain at their start/seek position until their first audio sample, matching
    /// native audio-master behavior. Video-only sources use the bounded start anchor immediately.
    presentation_master_delivered: bool,
    looping: bool,
    /// Media position freshly started decoders begin at. JumpToTime sets it so the ffmpeg
    /// processes are launched with an input-side seek instead of replaying from byte zero.
    decoder_seek: Duration,
    /// Playback-rate percentage from sceAvPlayerSetTrickSpeed; 100 is normal speed. It scales
    /// the player media clock, not media timestamps.
    trick_speed_percent: u32,
    /// Native Default mode (0) gates video by the presentation clock. None (1) hands frame
    /// selection to the caller and releases every decoded frame as soon as it is available,
    /// without changing the independent public clock.
    av_sync_mode: u32,
    video_decoder: Option<AvPlayerDecoder>,
    /// The video decoder reached EOF and its guest-visible frame queue has been drained. A
    /// multiplexed source remains active until every enabled stream reaches this state.
    video_decoder_drained: bool,
    raw_video_frame: Vec<u8>,
    padded_video_frame: Vec<u8>,
    video_buffer_base: u64,
    video_buffers: [u64; 3],
    video_buffer_stride: usize,
    next_video_buffer: usize,
    next_video_frame: u64,
    last_video_delivery: Option<Instant>,
    /// Host media-clock segment established only after the guest has consumed its first decoded
    /// frame. This prevents a long initial shader compile from seeking a short splash straight to
    /// EOF, while still allowing later frames to catch up when emulation runs below movie FPS.
    video_catch_up_origin: Option<(Instant, u64)>,
    audio_decoder: Option<AvPlayerDecoder>,
    /// The audio decoder reached EOF and its guest-visible frame queue has been drained.
    audio_decoder_drained: bool,
    raw_audio_frame: Vec<u8>,
    audio_buffer_base: u64,
    next_audio_buffer: usize,
    next_audio_frame: u64,
}

impl AvPlayerInstance {
    fn stop_decoders(&mut self) {
        if let Some(decoder) = self.video_decoder.take() {
            decoder.retire();
        }
        if let Some(decoder) = self.audio_decoder.take() {
            decoder.retire();
        }
        self.playing_since = None;
    }

    fn reset_playback(&mut self) {
        self.stop_decoders();
        self.elapsed = Duration::ZERO;
        self.presentation_anchor_media = Duration::ZERO;
        self.presentation_anchor_host = None;
        self.presentation_clock_reported = Duration::ZERO;
        self.presentation_master_delivered = false;
        self.end_of_stream = false;
        self.video_decoder_drained = false;
        self.audio_decoder_drained = false;
        self.next_video_frame = 0;
        self.last_video_delivery = None;
        self.video_catch_up_origin = None;
        self.next_audio_frame = 0;
        self.next_video_buffer = 0;
        self.next_audio_buffer = 0;
        self.decoder_seek = Duration::ZERO;
    }

    fn playback_elapsed(&self) -> Duration {
        self.playing_since
            .map_or(self.elapsed, |started| self.elapsed + started.elapsed())
    }
}

impl Drop for AvPlayerInstance {
    fn drop(&mut self) {
        self.stop_decoders();
        guest_free(self.video_buffer_base);
        guest_free(self.audio_buffer_base);
    }
}

impl GuestMspace {
    const ALLOCATION_ALIGNMENT: u64 = 16;
    const HEADER_SIZE: u64 = 0x100;

    fn new(name: String, base: u64, capacity: u64) -> Option<Self> {
        let end = base.checked_add(capacity)?;
        let first = align_up_u64(
            base.checked_add(Self::HEADER_SIZE)?,
            Self::ALLOCATION_ALIGNMENT,
        )?;
        if first >= end {
            return None;
        }
        let mut free_ranges = BTreeMap::new();
        free_ranges.insert(first, end - first);
        Some(Self {
            name,
            base,
            end,
            free_ranges,
            allocations: BTreeMap::new(),
        })
    }

    fn allocate(&mut self, requested_size: u64, alignment: u64) -> Option<u64> {
        if !alignment.is_power_of_two() {
            return None;
        }
        let alignment = alignment.max(Self::ALLOCATION_ALIGNMENT);
        let size = align_up_u64(requested_size.max(1), Self::ALLOCATION_ALIGNMENT)?;
        let candidate = self.free_ranges.iter().find_map(|(&start, &length)| {
            let aligned = align_up_u64(start, alignment)?;
            let allocation_end = aligned.checked_add(size)?;
            let range_end = start.checked_add(length)?;
            (allocation_end <= range_end).then_some((start, length, aligned, allocation_end))
        })?;
        let (range_start, range_length, allocation_start, allocation_end) = candidate;
        let range_end = range_start + range_length;
        self.free_ranges.remove(&range_start);
        if allocation_start > range_start {
            self.free_ranges
                .insert(range_start, allocation_start - range_start);
        }
        if allocation_end < range_end {
            self.free_ranges
                .insert(allocation_end, range_end - allocation_end);
        }
        self.allocations.insert(allocation_start, size);
        Some(allocation_start)
    }

    fn free(&mut self, address: u64) -> bool {
        let Some(size) = self.allocations.remove(&address) else {
            return address == 0;
        };
        self.insert_free_range(address, size);
        true
    }

    fn insert_free_range(&mut self, mut start: u64, mut size: u64) {
        if let Some((&previous_start, &previous_size)) = self.free_ranges.range(..start).next_back()
            && previous_start.checked_add(previous_size) == Some(start)
        {
            self.free_ranges.remove(&previous_start);
            start = previous_start;
            size += previous_size;
        }
        if let Some((&next_start, &next_size)) = self.free_ranges.range(start..).next()
            && start.checked_add(size) == Some(next_start)
        {
            self.free_ranges.remove(&next_start);
            size += next_size;
        }
        self.free_ranges.insert(start, size);
    }

    fn realloc(&mut self, address: u64, requested_size: u64) -> Option<u64> {
        if address == 0 {
            return self.allocate(requested_size, Self::ALLOCATION_ALIGNMENT);
        }
        if requested_size == 0 {
            self.free(address);
            return Some(0);
        }
        let old_size = *self.allocations.get(&address)?;
        let new_size = align_up_u64(requested_size, Self::ALLOCATION_ALIGNMENT)?;
        if new_size <= old_size {
            self.allocations.insert(address, new_size);
            if new_size < old_size {
                self.insert_free_range(address + new_size, old_size - new_size);
            }
            return Some(address);
        }
        let replacement = self.allocate(new_size, Self::ALLOCATION_ALIGNMENT)?;
        let copied = usize::try_from(old_size).ok()?;
        // SAFETY: Both ranges are disjoint, live allocations within the caller-provided mspace.
        unsafe {
            std::ptr::copy_nonoverlapping(address as *const u8, replacement as *mut u8, copied);
        }
        if remu_memory::guest_write_journal_enabled() {
            // SAFETY: the copy above just wrote exactly these bytes.
            let sample =
                unsafe { std::slice::from_raw_parts(replacement as *const u8, copied.min(4096)) };
            remu_memory::record_guest_write(
                remu_memory::GuestWriteSite::MspaceCopy,
                replacement,
                old_size,
                sample,
                0,
            );
        }
        self.free(address);
        Some(replacement)
    }

    fn usable_size(&self, address: u64) -> u64 {
        self.allocations.get(&address).copied().unwrap_or(0)
    }
}

#[derive(Debug, Default)]
struct AjmContextState {
    registered_codecs: BTreeSet<u32>,
    next_instance: u32,
    instances: BTreeSet<u32>,
}

const DEFAULT_THREAD_AFFINITY_MASK: u64 = 0x7f;
const DEFAULT_THREAD_STACK_SIZE: u64 = 0x10_0000;
const DEFAULT_THREAD_GUARD_SIZE: u64 = 0x1000;
const DEFAULT_THREAD_PRIORITY: i32 = 700;
/// Floor applied only to the host `std::thread` stack. Guest-visible pthread attributes keep the
/// size the title requested; Rosetta plus deep AGC HLE on graphics submit threads needs more than
/// the console's 16 KiB minimum.
const HOST_PTHREAD_STACK_FLOOR: usize = 0x20_0000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PthreadAttrState {
    affinity_mask: u64,
    detach_state: i32,
    stack_address: u64,
    stack_size: u64,
    guard_size: u64,
    inherit_schedule: i32,
    schedule_policy: i32,
    schedule_priority: i32,
}

impl Default for PthreadAttrState {
    fn default() -> Self {
        Self {
            affinity_mask: DEFAULT_THREAD_AFFINITY_MASK,
            detach_state: 0,
            stack_address: 0,
            stack_size: DEFAULT_THREAD_STACK_SIZE,
            guard_size: DEFAULT_THREAD_GUARD_SIZE,
            inherit_schedule: 4,
            schedule_policy: 1,
            schedule_priority: DEFAULT_THREAD_PRIORITY,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PthreadAttrObject {
    handle: u64,
    attributes: PthreadAttrState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PthreadMutexAttrState {
    mutex_type: i32,
    protocol: i32,
}

impl Default for PthreadMutexAttrState {
    fn default() -> Self {
        // Gen5's default mutex attribute is error-checking rather than host-pthread's normal type.
        Self {
            mutex_type: 1,
            protocol: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PthreadMutexAttrObject {
    handle: u64,
    attributes: PthreadMutexAttrState,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct AmprCommandBufferState {
    buffer: u64,
    size: u64,
    write_offset: u64,
    command_count: u64,
    /// File transfers recorded into this reusable command program.
    pending_reads: Vec<AmprReadCommand>,
    /// Queue completions recorded into this buffer, announced when it is submitted.
    pending_completions: Vec<AmprCompletion>,
    /// Completion words recorded into this buffer, written when it is submitted.
    pending_address_writes: Vec<AmprAddressWrite>,
}

/// One completion word an APR program publishes after its transfers finish.
///
/// This is the other half of the completion contract beside the event queue: a guest that does not
/// wait on an equeue polls this word instead, so a program that never writes it leaves that guest
/// waiting forever.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AmprAddressWrite {
    address: u64,
    value: u64,
}

/// One deferred APR file transfer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AmprReadCommand {
    file_id: u32,
    destination: u64,
    size: u64,
    file_offset: u64,
    record_offset: u64,
}

/// One recorded request to signal an event queue when a command buffer completes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AmprCompletion {
    equeue: u64,
    ident: u64,
    user_data: u64,
}

const PTHREAD_THREAD_OBJECT_SIZE: u64 = 0x1000;

#[derive(Clone, Debug, Eq, PartialEq)]
struct PthreadThreadObject {
    handle: u64,
    unique_id: u64,
    name: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PthreadKeyState {
    destructor: u64,
    generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GuestExitCallback {
    function: u64,
    argument: Option<u64>,
    module: u64,
}

#[derive(Debug, Default)]
struct RuntimeServiceState {
    libc_environment: Option<u64>,
    loaded_sysmodules: BTreeSet<i32>,
    ult_runtime: Option<u64>,
    splash: SplashState,
    user_service: InitializationState,
    user_login_event_delivered: bool,
    common_dialog: InitializationState,
    common_dialog_owner: Option<CommonDialogOwner>,
    /// Lifecycle state of the system message dialog, using the common-dialog status values.
    message_dialog_status: u32,
    message_dialog_sequence: u64,
    message_dialog_result_code: u32,
    message_dialog_button: u32,
    play_go_initialized: bool,
    play_go_open: bool,
    save_data_dialog_status: i32,
    save_data_dialog_sequence: u64,
    save_data_dialog_result_code: u32,
    agc_states: BTreeMap<u64, u32>,
    game_update: InitializationState,
    np_entitlement_access: Option<(u64, u64)>,
    np_session_signaling: Option<Vec<u8>>,
    next_np_web_api2_user_context: i32,
    np_web_api2_user_contexts: BTreeMap<i32, (i32, i32)>,
    next_np_web_api2_push_event_handle: i32,
    np_web_api2_push_event_handles: BTreeMap<i32, i32>,
    np_callback_polls: u64,
    np_title_id: Option<[u8; 16]>,
    share_feature_permitted: bool,
    share_content_parameter: Option<String>,
    /// Lifecycle state of `libSceErrorDialog`, using the common-dialog status values.
    error_dialog_status: u32,
    error_dialog_sequence: u64,
    error_dialog_result_code: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum InitializationState {
    #[default]
    Uninitialized,
    Initialized,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CommonDialogOwner {
    Message,
    SaveData,
    Error,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum SplashState {
    #[default]
    Visible,
    Hidden,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SaveDataMount {
    path: PathBuf,
    committed_path: PathBuf,
    dir_name: String,
    user_id: i32,
    resource: u32,
    read_only: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SaveDataTransaction {
    mount_point: String,
    committed_path: PathBuf,
    staging_path: PathBuf,
}

#[derive(Debug, Default)]
struct VideoOutState {
    is_open: bool,
    user_id: i32,
    bus_type: i32,
    index: i32,
    pixel_format: u64,
    tiling_mode: u32,
    width: u32,
    height: u32,
    display_buffers: Vec<u64>,
    display_buffer_attributes: Vec<Option<VideoOutBufferAttributes>>,
    flip_events: BTreeMap<u64, u64>,
    /// Equeues registered with `sceVideoOutAddVblankEvent`, keyed by queue handle.
    vblank_events: BTreeMap<u64, u64>,
    completed_flip_count: u64,
    /// Most recently completed flip, which `sceVideoOutGetFlipStatus` reports until the next one.
    last_completed_flip: Option<CompletedVideoOutFlip>,
}

/// Completion record `sceVideoOutGetFlipStatus` reads back to a title pacing its frames.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompletedVideoOutFlip {
    /// Process time, in microseconds, when the flip completed.
    process_time: u64,
    /// Process time counter, in nanoseconds, when the flip completed.
    process_time_counter: u64,
    /// Argument the title attached to the flip request.
    flip_argument: u64,
    /// Display-buffer index the flip scanned out.
    buffer_index: u32,
    /// Process time counter, in nanoseconds, when the flip was submitted.
    submit_process_time_counter: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct VideoOutBufferAttributes {
    width: u32,
    height: u32,
    pitch: u32,
    pixel_format: u64,
    tiling_mode: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingVideoOutFlip {
    completion: remu_gpu::GpuFlipCompletion,
    handle: u32,
    /// Process time counter, in nanoseconds, when the flip was queued for the GPU.
    submit_process_time_counter: u64,
}

// Shader evaluation keys are entirely emulator-owned integers. SipHash's collision resistance is
// unnecessary here and is costly because the hot miss path hashes up to 64 user-data words in as
// many as three cache tables. This small Fx-style mixer retains full equality checks in HashMap
// while making those repeated fixed-width writes substantially cheaper.
#[derive(Debug, Default)]
struct AgcGraphicsState {
    context_registers: BTreeMap<u32, u32>,
    shader_registers: BTreeMap<u32, u32>,
    user_config_registers: BTreeMap<u32, u32>,
    predicate_skip: bool,
    index_buffer_address: u64,
    index_resource_address: u64,
    index_buffer_count: u32,
    index_size: u32,
    instance_count: u32,
    draw_index_offset: u32,
    /// Active `SET_BASE` address for graphics-queue indirect draws and dispatches.
    ///
    /// This is command-processor state, not shader-stage state: the shader-type bit in a PM4
    /// header selects how the packet is decoded, but does not create a second indirect base.
    indirect_args_address: u64,
    /// Synthetic cumulative pixel count published by `PIXEL_PIPE_STAT_DUMP`.
    ///
    /// Gen5 exposes one cumulative counter per render backend. Keeping the sequence in the
    /// graphics command-processor state makes successive start/end query dumps differ while the
    /// high validity bit tells the guest that the ordered GPU write has retired.
    pixel_pipe_stat_counter: u64,
    emitted_draws: u64,
    emitted_compute_dispatches: u64,
    traced_shader_pairs: BTreeSet<(u64, u64)>,
    traced_scanout_shader_pairs: BTreeSet<(u64, u64)>,
    traced_target_shader_pairs: BTreeSet<(u64, u64)>,
    traced_sampled_target_shader_pairs: BTreeSet<(u64, u64)>,
    traced_sampled_compute_shaders: BTreeSet<u64>,
    sampled_texture_addresses: BTreeSet<u64>,
    failed_shader_evaluations: BTreeSet<(u8, u64)>,
    known_color_targets: BTreeMap<u64, remu_gpu::GpuColorTarget>,
    shader_evaluation_cache: ShaderEvaluationCache,
    shader_evaluation_workers: Option<ShaderEvaluationWorkers>,
}

#[derive(Debug, Default)]
struct AgcShaderRegistry {
    headers: BTreeMap<u64, u64>,
    code_sizes: BTreeMap<u64, u32>,
    registration_code_fingerprints: BTreeMap<u64, u64>,
    /// Separately allocated continuations selected by merged Gen5 shader tail calls.
    ///
    /// AGC programs the front half as the hardware entry point, but the fused stage descriptor is
    /// copied from the back half. Keying this relationship by the actually bound front address
    /// keeps code ownership and descriptor semantics together when a draw is snapshotted. The set
    /// deliberately retains ambiguity: if one entry point is fused with distinct back halves, a
    /// draw cannot safely select one from the ES program address alone.
    fused_continuations: BTreeMap<u64, BTreeSet<AgcFusedShaderContinuation>>,
    snapshots: BTreeMap<u64, remu_gpu::ShaderSnapshot>,
    decode_failures: BTreeSet<u64>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct AgcFusedShaderContinuation {
    front_header_address: u64,
    back_header_address: u64,
    back_program_address: u64,
    back_code_size: u32,
}

impl AgcGraphicsState {
    fn reset_pipeline(&mut self) {
        self.context_registers.clear();
        self.shader_registers.clear();
        self.user_config_registers.clear();
        self.predicate_skip = false;
        self.index_buffer_address = 0;
        self.index_resource_address = 0;
        self.index_buffer_count = 0;
        self.index_size = 0;
        self.instance_count = 1;
        self.draw_index_offset = 0;
        self.indirect_args_address = 0;
    }
}

#[derive(Clone, Copy, Debug)]
struct KernelQueuedEvent {
    equeue: u64,
    ident: u64,
    filter: i16,
    flags: u16,
    fflags: u32,
    data: u64,
    user_data: u64,
    /// Logical AGC graphics completions represented by this possibly coalesced event.
    agc_graphics_completions: u32,
    /// Logical AGC compute user interrupts represented by this possibly coalesced event.
    agc_user_interrupts: u32,
}

#[derive(Debug, Default)]
struct AgcCompletionTelemetry {
    refreshed_graphics: u64,
    refreshed_compute: u64,
    queued_graphics_notifications: u64,
    queued_user_interrupts: u64,
    delivered_graphics_notifications: u64,
    delivered_user_interrupts: u64,
    cancelled_graphics_notifications: u64,
    cancelled_user_interrupts: u64,
    missing_graphics_registration: u64,
    missing_user_interrupt_registration: u64,
    unmatched_backend_ids: u64,
    last_refreshed_graphics: Option<u64>,
    last_refreshed_compute: Option<u64>,
    last_unmatched_backend_id: Option<u64>,
}

const KERNEL_EVENT_FILTER_GRAPHICS: i16 = -14;
const KERNEL_EVENT_FILTER_USER: i16 = -11;
/// Gen5 AP/AMPR command-buffer completion. The Gen4 filter set ends at the high-resolution timer,
/// so the console assigns this newer source the next identifier below it.
const KERNEL_EVENT_FILTER_AMPR: i16 = -16;
/// Display output events, including mode changes and flips.
const KERNEL_EVENT_FILTER_VIDEO_OUT: i16 = -13;

static PROBE_DISPATCH: OnceLock<RwLock<ProbeDispatchState>> = OnceLock::new();

// Counts the `PROBE_DISPATCH` write guards this thread holds. Nothing reachable while one is alive
// may park on a guest synchronization object: the guest thread that would release it cannot reach
// its own release call without this same lock, so the whole process wedges. Test builds turn that
// class into a failed assertion instead of a hang.
#[cfg(test)]
thread_local! {
    static DISPATCH_WRITE_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Write access to [`PROBE_DISPATCH`], recorded for the guard's lifetime in test builds.
pub(crate) struct DispatchWriteGuard<'a>(std::sync::RwLockWriteGuard<'a, ProbeDispatchState>);

impl std::ops::Deref for DispatchWriteGuard<'_> {
    type Target = ProbeDispatchState;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for DispatchWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
impl Drop for DispatchWriteGuard<'_> {
    fn drop(&mut self) {
        DISPATCH_WRITE_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

/// Takes the process-wide dispatch lock for write, poisoning ignored.
pub(crate) fn lock_probe_dispatch_for_write(
    lock: &RwLock<ProbeDispatchState>,
) -> DispatchWriteGuard<'_> {
    let guard = lock
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    #[cfg(test)]
    DISPATCH_WRITE_DEPTH.with(|depth| depth.set(depth.get() + 1));
    DispatchWriteGuard(guard)
}

/// Rejects a guest-synchronization park attempted under the process-wide dispatch write lock.
///
/// Compiled away outside test builds; the constraint it names is structural, not diagnostic.
#[inline]
pub(crate) fn assert_wait_is_off_the_dispatch_lock(operation: &str) {
    let _ = operation;
    #[cfg(test)]
    DISPATCH_WRITE_DEPTH.with(|depth| {
        assert_eq!(
            depth.get(),
            0,
            "{operation} would park while this thread holds the process-wide dispatch write lock, \
             which the thread that must release it also needs"
        );
    });
}

/// Serializes APR/AMPR file and command-buffer operations in guest-observable order.
///
/// A file read deliberately releases `PROBE_DISPATCH` while the host blocks in `pread` and while
/// bytes are copied into guest memory. Holding this narrower lock across that gap prevents submit,
/// reset, destruction, or file re-resolution from overtaking the transfer.
/// A small fixed lock table preserves program order for each APR command buffer without forcing
/// unrelated streaming queues through one process-wide storage lock. A collision only reduces
/// concurrency; it cannot change guest-visible ordering.
static AMPR_COMMAND_ORDER: OnceLock<Vec<Mutex<()>>> = OnceLock::new();
/// Reusable host storage for APR reads. The pool lock is held only while borrowing or returning a
/// vector, never during host I/O or watched guest-memory copies.
static AMPR_READ_STAGING_POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());
static KERNEL_EQUEUE_WAKE: OnceLock<(Mutex<u64>, Condvar)> = OnceLock::new();
static AUDIO_OUT: OnceLock<Mutex<audio::AudioOutState>> = OnceLock::new();
static AUDIO_OUT2: OnceLock<Mutex<AudioOut2State>> = OnceLock::new();
static PROCESS_CLOCK_EPOCH: OnceLock<Instant> = OnceLock::new();
static PROCESS_START_NANOS: AtomicU64 = AtomicU64::new(0);
static NEXT_NET_POOL_HANDLE: AtomicUsize = AtomicUsize::new(1);
static NEXT_SSL_CONTEXT: AtomicUsize = AtomicUsize::new(1);
static NEXT_HTTP_CONTEXT: AtomicUsize = AtomicUsize::new(1);
static NEXT_NP_WEB_API2_CONTEXT: AtomicUsize = AtomicUsize::new(1);
static DISPATCH_LOCK_STALL_LOGS: AtomicUsize = AtomicUsize::new(0);
static DISPATCH_LOCK_HOLD_LOGS: AtomicUsize = AtomicUsize::new(0);
static GUEST_EXCEPTION_DELIVERY: OnceLock<Mutex<GuestExceptionDeliveryState>> = OnceLock::new();

static GUEST_EXCEPTION_PENDING: AtomicBool = AtomicBool::new(false);
#[cfg(target_os = "macos")]
static KERNEL_MEMORY_READABLE_RANGES: OnceLock<RwLock<Arc<[(u64, u64)]>>> = OnceLock::new();
#[cfg(target_os = "macos")]
static KERNEL_MEMORY_READABLE_RANGES_GENERATION: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingGuestException {
    handler: u64,
    exception: i32,
}

#[derive(Debug, Default)]
struct GuestExceptionDeliveryState {
    pending: BTreeMap<u64, PendingGuestException>,
    active: BTreeSet<u64>,
}

fn guest_exception_delivery() -> &'static Mutex<GuestExceptionDeliveryState> {
    GUEST_EXCEPTION_DELIVERY.get_or_init(|| Mutex::new(GuestExceptionDeliveryState::default()))
}

fn guest_exception_is_pending(thread: u64) -> bool {
    thread != 0
        && guest_exception_delivery()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending
            .contains_key(&thread)
}

#[derive(Debug)]
struct ImportTelemetry {
    nids: Box<[String]>,
    seen: Box<[AtomicBool]>,
    counts: Box<[AtomicUsize]>,
    hits: AtomicUsize,
    maximum_hits: usize,
}

impl ImportTelemetry {
    fn new(nids: Vec<String>, maximum_hits: usize) -> Self {
        let count = nids.len();
        Self {
            nids: nids.into_boxed_slice(),
            seen: (0..count)
                .map(|_| AtomicBool::new(false))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            counts: (0..count)
                .map(|_| AtomicUsize::new(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            hits: AtomicUsize::new(0),
            maximum_hits,
        }
    }

    fn hot_imports(&self, limit: usize) -> Vec<(&str, usize)> {
        let mut counts = self
            .nids
            .iter()
            .zip(&self.counts)
            .map(|(nid, count)| (nid.as_str(), count.load(Ordering::Relaxed)))
            .collect::<Vec<_>>();
        counts.sort_unstable_by_key(|(_, count)| std::cmp::Reverse(*count));
        counts.truncate(limit);
        counts
    }
}

static IMPORT_TELEMETRY: OnceLock<RwLock<Arc<ImportTelemetry>>> = OnceLock::new();
static IMPORT_TELEMETRY_GENERATION: AtomicUsize = AtomicUsize::new(0);
static TRACE_HOT_IMPORTS: OnceLock<bool> = OnceLock::new();
static GUEST_IMPORT_TRACE_EPOCH: OnceLock<Instant> = OnceLock::new();
static GUEST_IMPORT_TRACE_RECORDS: Mutex<Vec<Arc<GuestImportTraceRecord>>> = Mutex::new(Vec::new());
static GUEST_IMPORT_TRACE_REPORTER: OnceLock<()> = OnceLock::new();
static GUEST_IMPORT_TRACE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static CURRENT_GUEST_THREAD_POINTER: Cell<u64> = const { Cell::new(0) };
    static CURRENT_GUEST_THREAD_HANDLE: Cell<u64> = const { Cell::new(0) };
    static CURRENT_GUEST_THREAD_EXIT_CONTEXT: Cell<*mut GuestThreadExitContext> = const { Cell::new(std::ptr::null_mut()) };
    static GUEST_PTHREAD_SPECIFIC: RefCell<BTreeMap<u32, (u64, u64)>> = const { RefCell::new(BTreeMap::new()) };
    static IMPORT_TELEMETRY_CACHE: RefCell<Option<(usize, Arc<ImportTelemetry>)>> = const { RefCell::new(None) };
    static PROCESS_TIME_CALLSITES: RefCell<[(u64, u64); 8]> = const { RefCell::new([(0, 0); 8]) };
    static GUEST_IMPORT_TRACE_RECORD: Cell<*const GuestImportTraceRecord> = const { Cell::new(std::ptr::null()) };
    #[cfg(target_os = "macos")]
    static KERNEL_MEMORY_READABLE_RANGES_CACHE: RefCell<Option<(usize, Arc<[(u64, u64)]>)>> = const { RefCell::new(None) };
}

static PTHREAD_KEY_GENERATIONS: OnceLock<RwLock<[u64; 256]>> = OnceLock::new();
static NEXT_PTHREAD_KEY_GENERATION: AtomicU64 = AtomicU64::new(1);

fn pthread_key_generations() -> &'static RwLock<[u64; 256]> {
    PTHREAD_KEY_GENERATIONS.get_or_init(|| RwLock::new([0; 256]))
}

#[cfg(target_os = "macos")]
fn cached_kernel_memory_readable_range_end(address: u64) -> Option<u64> {
    let generation = KERNEL_MEMORY_READABLE_RANGES_GENERATION.load(Ordering::Acquire);
    KERNEL_MEMORY_READABLE_RANGES_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache
            .as_ref()
            .is_none_or(|(cached_generation, _)| *cached_generation != generation)
        {
            let ranges = KERNEL_MEMORY_READABLE_RANGES
                .get_or_init(|| RwLock::new(Arc::from([])))
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            *cache = Some((generation, ranges));
        }
        let ranges = &cache
            .as_ref()
            .expect("readable range cache was populated")
            .1;
        let index = ranges.partition_point(|&(start, _)| start <= address);
        index
            .checked_sub(1)
            .and_then(|index| ranges.get(index))
            .filter(|&&(start, end)| address >= start && address < end)
            .map(|&(_, end)| end)
    })
}

fn cached_import_telemetry() -> *const ImportTelemetry {
    let generation = IMPORT_TELEMETRY_GENERATION.load(Ordering::Acquire);
    IMPORT_TELEMETRY_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache
            .as_ref()
            .is_none_or(|(cached_generation, _)| *cached_generation != generation)
        {
            let telemetry = IMPORT_TELEMETRY
                .get_or_init(|| RwLock::new(Arc::new(ImportTelemetry::new(Vec::new(), 0))))
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            *cache = Some((generation, telemetry));
        }
        Arc::as_ptr(&cache.as_ref().expect("telemetry cache was populated").1)
    })
}

extern "C" fn current_guest_thread_pointer() -> u64 {
    CURRENT_GUEST_THREAD_POINTER.get()
}

fn set_current_guest_thread_pointer(thread_pointer: u64) {
    CURRENT_GUEST_THREAD_POINTER.set(thread_pointer);
}

thread_local! {
    /// Name the guest gave this thread, retained so diagnostics can attribute a stall without
    /// correlating against a separate creation log.
    static CURRENT_GUEST_THREAD_NAME: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn set_current_guest_thread_name(name: Option<String>) {
    CURRENT_GUEST_THREAD_NAME.with_borrow_mut(|current| *current = name);
}

/// Reports the current guest thread's name, or `main` for the thread that entered the guest.
fn current_guest_thread_name() -> String {
    CURRENT_GUEST_THREAD_NAME
        .with_borrow(|name| name.clone())
        .unwrap_or_else(|| "main".to_owned())
}

fn current_guest_thread_handle() -> u64 {
    CURRENT_GUEST_THREAD_HANDLE.get()
}

fn set_current_guest_thread_handle(thread_handle: u64) {
    CURRENT_GUEST_THREAD_HANDLE.set(thread_handle);
}

#[repr(C)]
#[derive(Debug, Default)]
struct GuestThreadExitContext {
    stack_pointer: u64,
    continuation: u64,
}

/// Calls a guest entry behind a stack boundary that `scePthreadExit` can return to.
///
/// Guest pthread implementations terminate from arbitrary depths and deliberately place a trap
/// after their non-returning exit import. Calling host `pthread_exit` would bypass Rust's thread
/// closure and abort the process. This boundary instead restores its own callee-saved frame and
/// lets the Rust wrapper complete normally with the guest-supplied return value.
#[cfg(target_arch = "x86_64")]
#[unsafe(no_mangle)]
#[unsafe(naked)]
unsafe extern "C" fn invoke_guest_entry_with_exit_boundary(
    _entry: u64,
    _argument0: u64,
    _argument1: u64,
    _context: *mut GuestThreadExitContext,
) -> u64 {
    core::arch::naked_asm!(
        "push rbp",
        "mov rbp, rsp",
        "push rbx",
        "push r12",
        "push r13",
        "push r14",
        "push r15",
        "sub rsp, 8",
        "mov [rcx], rsp",
        "lea rax, [rip + 2f]",
        "mov [rcx + 8], rax",
        "mov rax, rdi",
        "mov rdi, rsi",
        "mov rsi, rdx",
        "call rax",
        "2:",
        "add rsp, 8",
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbx",
        "pop rbp",
        "ret",
    );
}

#[cfg(target_arch = "x86_64")]
#[unsafe(no_mangle)]
#[unsafe(naked)]
unsafe extern "C" fn escape_guest_entry(_context: *const GuestThreadExitContext, _value: u64) -> ! {
    core::arch::naked_asm!("mov rsp, [rdi]", "mov rax, rsi", "jmp [rdi + 8]",);
}

#[cfg(target_arch = "x86_64")]
fn invoke_guest_entry_with_exit(entry: u64, argument0: u64, argument1: u64) -> u64 {
    let mut context = GuestThreadExitContext::default();
    let previous = CURRENT_GUEST_THREAD_EXIT_CONTEXT.replace(&raw mut context);
    // SAFETY: the mapped guest entry uses the System V ABI. The assembly boundary preserves every
    // callee-saved register and records a continuation in this live stack-owned context.
    let result = unsafe {
        invoke_guest_entry_with_exit_boundary(entry, argument0, argument1, &raw mut context)
    };
    CURRENT_GUEST_THREAD_EXIT_CONTEXT.set(previous);
    result
}

#[cfg(target_arch = "x86_64")]
fn guest_pthread_exit(value: u64) -> ! {
    let context = CURRENT_GUEST_THREAD_EXIT_CONTEXT.get();
    if context.is_null() {
        eprintln!("[REMu][THREAD] scePthreadExit called outside a guest thread boundary");
        terminate_probe(87);
    }
    // SAFETY: the context belongs to this native thread and remains live until its guest entry
    // either returns or exits. This import is handled before acquiring process-wide locks, so the
    // non-local escape cannot strand a Rust synchronization guard.
    unsafe { escape_guest_entry(context, value) }
}

#[cfg(not(target_arch = "x86_64"))]
fn guest_pthread_exit(_value: u64) -> ! {
    unreachable!("guest pthread execution requires an x86-64 host process")
}

#[cfg(not(target_arch = "x86_64"))]
fn invoke_guest_entry_with_exit(_entry: u64, _argument0: u64, _argument1: u64) -> u64 {
    unreachable!("guest pthread execution requires an x86-64 host process")
}

fn allocate_pthread_thread_object(state: &mut ProbeDispatchState, name: &str) -> Option<u64> {
    let unique_id = state.next_thread_unique_id.max(1);
    state.next_thread_unique_id = unique_id.checked_add(1)?;
    let handle = guest_calloc(1, PTHREAD_THREAD_OBJECT_SIZE);
    if handle == 0 {
        return None;
    }
    state.pthread_threads.insert(
        handle,
        PthreadThreadObject {
            handle,
            unique_id,
            name: if name.is_empty() {
                format!("Thread-{unique_id:X}")
            } else {
                name.to_owned()
            },
        },
    );
    Some(handle)
}

fn retire_detached_guest_thread(handle: u64) {
    let Some(dispatch) = PROBE_DISPATCH.get() else {
        return;
    };
    let object = {
        let mut state = lock_probe_dispatch_for_write(dispatch);
        if !state.detached_threads.remove(&handle) {
            return;
        }
        state.thread_handles.remove(&handle);
        state.thread_attributes.remove(&handle);
        state.pthread_threads.remove(&handle)
    };
    if let Some(object) = object {
        guest_free(object.handle);
    }
}

/// Spawns a REmu-owned thread that is allowed to call guest code.
///
/// Guest functions read `%fs` and may call imports, so such a thread needs the same identity a
/// `scePthreadCreate` thread receives: its own copy of the linked TLS image, a pthread object of
/// its own, and registration in the runtime's thread bookkeeping. Returns the guest thread handle.
fn spawn_guest_capable_thread(
    name: &str,
    state: &mut ProbeDispatchState,
    body: fn(),
) -> Option<u64> {
    // A thread started before an image is linked has no TLS to install. Refusing to start would
    // discard the work the thread owns, so it runs without a guest thread pointer instead.
    let guest_tls = u64::try_from(state.tls_template.len())
        .is_ok_and(|length| length == GUEST_TLS_PREFIX_SIZE + GUEST_TLS_TCB_SIZE)
        .then(|| clone_guest_tls(&state.tls_template));
    let handle = allocate_pthread_thread_object(state, name)?;
    let trace_name = name.to_owned();
    let spawned = std::thread::Builder::new()
        .name(trace_name.clone())
        .stack_size(HOST_PTHREAD_STACK_FLOOR)
        .spawn(move || {
            set_current_guest_thread_handle(handle);
            set_current_guest_thread_name(Some(trace_name));
            if let Some(guest_tls) = guest_tls.as_ref() {
                set_current_guest_thread_pointer(guest_tls.thread_pointer);
            }
            body();
            run_guest_thread_destructors();
            retire_current_guest_import_trace();
            set_current_guest_thread_pointer(0);
            set_current_guest_thread_handle(0);
            drop(guest_tls);
            0
        });
    match spawned {
        Ok(thread) => {
            state
                .thread_attributes
                .insert(handle, PthreadAttrState::default());
            state.thread_handles.insert(handle, thread);
            Some(handle)
        }
        Err(error) => {
            eprintln!("[REMu][THREAD] host spawn refused name={name:?}: {error}");
            if let Some(object) = state.pthread_threads.remove(&handle) {
                guest_free(object.handle);
            }
            None
        }
    }
}

const AV_PLAYER_CONTROLLER_THREAD_NAME: &str = "AvPlayerController";

/// Owner of the `AvPlayerEventDelivery::Controller` queue.
///
/// `libSceAvPlayer` issues Ready and end-of-stream from its own controller thread. A queue drained
/// by whichever guest thread happens to run the next stateful import instead re-enters that
/// thread's own `AvPlayer` frame, and strands every later notification once the guest stops issuing
/// such imports, so delivery needs a thread the runtime owns.
static AV_PLAYER_CONTROLLER: OnceLock<AvPlayerController> = OnceLock::new();

#[derive(Default)]
struct AvPlayerController {
    signal: Mutex<AvPlayerControllerSignal>,
    wakeup: Condvar,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum AvPlayerControllerState {
    #[default]
    NotStarted,
    Running,
    /// Exit requested; the thread leaves once the callback it is inside returns.
    Stopping,
    Finished,
}

#[derive(Default)]
struct AvPlayerControllerSignal {
    state: AvPlayerControllerState,
    /// Published under the dispatch write lock by every queue site, so it is never observed clear
    /// while a controller callback is still parked in `ProbeDispatchState`.
    queued: bool,
    delivered: u64,
    /// Event id and entry time of the callback currently executing guest code.
    in_flight: Option<(u64, Instant)>,
}

fn av_player_controller() -> &'static AvPlayerController {
    AV_PLAYER_CONTROLLER.get_or_init(AvPlayerController::default)
}

/// Publishes a queued controller notification, starting the controller thread on first use.
///
/// The caller holds the dispatch write lock. The callback itself runs on the controller thread,
/// which holds no runtime lock while it is inside guest code.
pub(crate) fn wake_av_player_controller(state: &mut ProbeDispatchState) {
    let controller = av_player_controller();
    let mut signal = controller
        .signal
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    signal.queued = true;
    if signal.state == AvPlayerControllerState::NotStarted {
        if let Some(handle) = spawn_guest_capable_thread(
            AV_PLAYER_CONTROLLER_THREAD_NAME,
            state,
            run_av_player_controller,
        ) {
            eprintln!("[REMu][AVPLAYER] controller thread started handle={handle:#x}");
            signal.state = AvPlayerControllerState::Running;
        } else {
            eprintln!(
                "[REMu][AVPLAYER] controller thread could not start; asynchronous events stay queued"
            );
        }
    }
    controller.wakeup.notify_all();
}

/// Delivers queued controller notifications until the process tears down.
fn run_av_player_controller() {
    // The wait is bounded so shutdown is observed without a notification, and so a queue site that
    // publishes without signalling still has its callback delivered.
    const IDLE_POLL: Duration = Duration::from_millis(250);
    let controller = av_player_controller();
    loop {
        {
            let mut signal = controller
                .signal
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            while !signal.queued && signal.state == AvPlayerControllerState::Running {
                signal = controller
                    .wakeup
                    .wait_timeout(signal, IDLE_POLL)
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .0;
            }
            if signal.state != AvPlayerControllerState::Running {
                break;
            }
            signal.queued = false;
        }
        let Some(dispatch) = PROBE_DISPATCH.get() else {
            continue;
        };
        // The dispatch lock is taken only to move the callbacks out. A guest event handler may
        // block indefinitely, and holding the process-wide lock across one would stop every import.
        let callbacks = {
            let mut state = lock_probe_dispatch_for_write(dispatch);
            take_av_player_controller_callbacks(&mut state)
        };
        for callback in callbacks {
            controller
                .signal
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .in_flight = Some((callback.event, Instant::now()));
            invoke_av_player_event_callback(callback);
            let mut signal = controller
                .signal
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            signal.in_flight = None;
            signal.delivered = signal.delivered.saturating_add(1);
        }
    }
    controller
        .signal
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .state = AvPlayerControllerState::Finished;
    controller.wakeup.notify_all();
}

/// Asks the controller thread to exit and waits a bounded interval for it.
///
/// A guest event handler can block forever, so teardown reports a wedged controller rather than
/// joining it: this runs on the paths that terminate the process, including the stall watchdog.
fn stop_av_player_controller() {
    const STOP_WAIT: Duration = Duration::from_millis(100);
    let Some(controller) = AV_PLAYER_CONTROLLER.get() else {
        return;
    };
    let mut signal = controller
        .signal
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if signal.state != AvPlayerControllerState::Running {
        return;
    }
    signal.state = AvPlayerControllerState::Stopping;
    controller.wakeup.notify_all();
    let (signal, _) = controller
        .wakeup
        .wait_timeout_while(signal, STOP_WAIT, |signal| {
            signal.state != AvPlayerControllerState::Finished
        })
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if signal.state != AvPlayerControllerState::Finished {
        eprintln!(
            "[REMu][AVPLAYER] controller thread still running after {}ms in_flight={:?}",
            STOP_WAIT.as_millis(),
            signal
                .in_flight
                .map(|(event, since)| (event, since.elapsed()))
        );
    }
}

/// Reports controller-thread liveness alongside the per-player stall snapshot.
///
/// A notification that was queued but never delivered is invisible in the player state: a player
/// waiting for Ready reports the same `inactive` decoder state as one that was never started.
fn report_av_player_controller_stall_attribution(queued: Option<usize>) {
    let queued = queued.map_or_else(|| "dispatch-busy".to_owned(), |queued| queued.to_string());
    let Some(controller) = AV_PLAYER_CONTROLLER.get() else {
        eprintln!("[REMu][STALL-AVPLAYER] controller=not-started queued={queued}");
        return;
    };
    let Ok(signal) = controller.signal.try_lock() else {
        eprintln!("[REMu][STALL-AVPLAYER] controller=signal-busy queued={queued}");
        return;
    };
    eprintln!(
        "[REMu][STALL-AVPLAYER] controller={:?} queued={queued} delivered={} in_flight={:?}",
        signal.state,
        signal.delivered,
        signal
            .in_flight
            .map(|(event, since)| (event, since.elapsed().as_millis())),
    );
}

#[cfg(target_arch = "x86_64")]
fn configure_probe_dispatch(image: &LinkedImage, game_root: &Path, maximum_hits: usize) {
    // Decode the installed title image before guest entry. Keeping PNG parsing on the runtime
    // side means the GPU seam carries a validated, immutable RGBA image instead of a host path
    // whose lifetime or casing the presenter would have to rediscover.
    let system_splash = load_system_splash(game_root);
    remu_gpu::install_gpu_completion_notifier(notify_kernel_equeue_waiters);
    let state = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
    let mut state = lock_probe_dispatch_for_write(state);
    let nids = image
        .import_stubs
        .iter()
        .map(|stub| stub.nid.clone())
        .collect();
    let telemetry =
        IMPORT_TELEMETRY.get_or_init(|| RwLock::new(Arc::new(ImportTelemetry::new(Vec::new(), 0))));
    *telemetry
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) =
        Arc::new(ImportTelemetry::new(nids, maximum_hits));
    IMPORT_TELEMETRY_GENERATION.fetch_add(1, Ordering::Release);
    state.unimplemented_nids.clear();
    #[cfg(target_os = "macos")]
    {
        state.kernel_memory = None;
        *KERNEL_MEMORY_READABLE_RANGES
            .get_or_init(|| RwLock::new(Arc::from([])))
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Arc::from([]);
        KERNEL_MEMORY_READABLE_RANGES_GENERATION.fetch_add(1, Ordering::Release);
        state.process_mapping_start = image.memory.mapping_start().get();
        state.process_mapping_end = state
            .process_mapping_start
            .checked_add(image.memory.mapping_size())
            .expect("the validated process mapping range cannot overflow");
    }
    state.thread_handles.clear();
    state.thread_attributes.clear();
    state.detached_threads.clear();
    for object in std::mem::take(&mut state.pthread_threads).into_values() {
        guest_free(object.handle);
    }
    state.next_thread_unique_id = 1;
    let main_thread = allocate_pthread_thread_object(&mut state, "main")
        .expect("the process main-thread object allocation must succeed");
    state
        .thread_attributes
        .insert(main_thread, PthreadAttrState::default());
    for object in std::mem::take(&mut state.pthread_attributes).into_values() {
        guest_free(object.handle);
    }
    for object in std::mem::take(&mut state.pthread_mutex_attributes).into_values() {
        guest_free(object.handle);
    }
    state.pthread_condition_attributes.clear();
    state.pthread_keys.clear();
    *pthread_key_generations()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = [0; 256];
    GUEST_PTHREAD_SPECIFIC.with_borrow_mut(BTreeMap::clear);
    state.exit_callbacks.clear();
    state.process_param_address = image.process_param.map_or(0, GuestAddress::get);
    state.thread_atexit_count_callback = 0;
    state.thread_atexit_report_callback = 0;
    state.thread_atexit_counts.clear();
    state.thread_dtors_callback = 0;
    state.module_tls_destructor_refs.clear();
    state.application_heap_api_address = 0;
    state.bundled_modules.clone_from(&image.bundled_modules);
    state.unwind_modules.clone_from(&image.unwind_modules);
    state.installed_exception_handlers.clear();
    state.next_ajm_context = 1;
    state.ajm_contexts.clear();
    state.next_acm_context = 1;
    state.acm_contexts.clear();
    *guest_exception_delivery()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) =
        GuestExceptionDeliveryState::default();
    GUEST_EXCEPTION_PENDING.store(false, Ordering::Release);
    state.heap_trace_storage.clear();
    state.heap_trace_storage.resize(65, 0);
    state.guest_mspaces.clear();
    state.json_runtime = JsonRuntimeState::default();
    state.av_player = AvPlayerRuntimeState::default();
    state.services = RuntimeServiceState::default();
    reset_pad_runtime();
    reset_kernel_process_time();
    *AUDIO_OUT
        .get_or_init(|| Mutex::new(audio::AudioOutState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = audio::AudioOutState::default();
    *AUDIO_OUT2
        .get_or_init(|| Mutex::new(AudioOut2State::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = AudioOut2State::default();
    state.agc_primary_register_defaults = 0;
    state.agc_internal_register_defaults = 0;
    state.ctype_table = 0;
    state.tolower_table = 0;
    state.toupper_table = 0;
    state.locale_object = 0;
    state.locale_vtable = 0;
    let tls_template_address = image.tls.thread_pointer.get() - GUEST_TLS_PREFIX_SIZE;
    let tls_template_size = usize::try_from(GUEST_TLS_PREFIX_SIZE + GUEST_TLS_TCB_SIZE)
        .expect("guest TLS template size fits usize");
    state.tls_template.resize(tls_template_size, 0);
    // SAFETY: The linked image owns this live TLS reservation for the entire probe. Snapshot it
    // before guest entry so later workers never inherit the main thread's mutated allocator caches.
    unsafe {
        std::ptr::copy_nonoverlapping(
            tls_template_address as *const u8,
            state.tls_template.as_mut_ptr(),
            tls_template_size,
        );
    }
    state.video_out = VideoOutState::default();
    game_root.clone_into(&mut state.game_root);
    state.np_universal_data = np::UniversalDataState::default();
    state.np_game_intent = np::GameIntentState::default();
    state.np_trophy = np::TrophyState::default();
    remu_gpu::reset_host_dialog_state();
    state.agc_submission_count = 0;
    state.agc_graphics = AgcGraphicsState::default();
    state.agc_compute_queues.clear();
    state.agc_shader_registry = AgcShaderRegistry::default();
    state.next_equeue = 1;
    state.kernel_equeues.clear();
    state.kernel_event_registrations.clear();
    state.pending_events.clear();
    state.pending_video_out_flips.clear();
    state.pending_agc_completions.clear();
    state.pending_agc_user_interrupts.clear();
    state.agc_completion_telemetry = AgcCompletionTelemetry::default();
    *KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = KernelFileState::default();
    state.ampr_files.clear();
    state.ampr_file_handles.clear();
    state.save_root = save_data_root(game_root);
    if let Err(error) = std::fs::create_dir_all(state.save_root.join("download0")) {
        eprintln!(
            "[REMu][VFS] could not create persistent /download0 backing under {}: {error}",
            state.save_root.display()
        );
    }
    state.save_data_initialized = false;
    state.save_mounts.clear();
    state.next_save_transaction_resource = 1;
    state.save_transaction_resources.clear();
    state.prepared_save_resources.clear();
    state.save_transactions.clear();
    let sync = GUEST_SYNC.get_or_init(|| (Mutex::new(GuestSyncState::default()), Condvar::new()));
    *sync
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = GuestSyncState::default();
    set_current_guest_thread_handle(main_thread);
    drop(state);
    publish_system_splash(system_splash);
}

/// Dumps selected linked guest ranges before any module initializer or process entry can mutate
/// them. `REMU_TRACE_GUEST_CODE` accepts comma-separated `address:length` pairs; both fields may
/// be decimal or `0x`-prefixed hexadecimal. The bytes are printed in one deterministic hexdump
/// and, when `REMU_DUMP_GUEST_CODE_DIR` is set, also retained as raw binaries for disassembly.
#[cfg(target_arch = "x86_64")]
fn trace_linked_guest_code(image: &LinkedImage) {
    const MAXIMUM_RANGE_BYTES: usize = 1024 * 1024;
    const MAXIMUM_RANGE_COUNT: usize = 16;

    let Ok(specifications) = std::env::var("REMU_TRACE_GUEST_CODE") else {
        return;
    };
    let output_directory = std::env::var_os("REMU_DUMP_GUEST_CODE_DIR").map(PathBuf::from);
    if let Some(directory) = &output_directory
        && let Err(error) = std::fs::create_dir_all(directory)
    {
        eprintln!(
            "[REMu][GUEST-CODE] cannot create output directory {}: {error}",
            directory.display()
        );
    }
    let mut total_bytes = 0_usize;
    for specification in specifications
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .take(MAXIMUM_RANGE_COUNT)
    {
        let Some((address, length)) = specification.split_once(':') else {
            eprintln!(
                "[REMu][GUEST-CODE] invalid range {specification:?}; expected address:length"
            );
            continue;
        };
        let Some(address) = parse_guest_code_number(address) else {
            eprintln!("[REMu][GUEST-CODE] invalid address {address:?}");
            continue;
        };
        let Some(length) =
            parse_guest_code_number(length).and_then(|value| usize::try_from(value).ok())
        else {
            eprintln!("[REMu][GUEST-CODE] invalid length {length:?}");
            continue;
        };
        let Some(next_total) = total_bytes.checked_add(length) else {
            eprintln!("[REMu][GUEST-CODE] rejected overflowing total range length");
            continue;
        };
        if length == 0 || next_total > MAXIMUM_RANGE_BYTES {
            eprintln!(
                "[REMu][GUEST-CODE] rejected range address={address:#x} length={length:#x}; maximum total={MAXIMUM_RANGE_BYTES:#x}"
            );
            continue;
        }
        let Some(end) = address.checked_add(u64::try_from(length).unwrap_or(u64::MAX)) else {
            eprintln!(
                "[REMu][GUEST-CODE] rejected overflowing range address={address:#x} length={length:#x}"
            );
            continue;
        };
        let executable = image.memory.regions().iter().any(|region| {
            region.flags & GUEST_EXECUTE != 0
                && address >= region.start.get()
                && region
                    .start
                    .get()
                    .checked_add(region.memory_size)
                    .is_some_and(|region_end| end <= region_end)
        });
        if !executable {
            eprintln!(
                "[REMu][GUEST-CODE] rejected non-executable range address={address:#x} length={length:#x}"
            );
            continue;
        }
        let mut bytes = vec![0_u8; length];
        if let Err(error) = image.memory.read(GuestAddress::new(address), &mut bytes) {
            eprintln!(
                "[REMu][GUEST-CODE] unreadable range address={address:#x} length={length:#x}: {error}"
            );
            continue;
        }
        total_bytes = next_total;
        eprintln!("[REMu][GUEST-CODE] begin address={address:#018x} length={length:#x}");
        for (offset, chunk) in bytes.chunks(16).enumerate() {
            let line_address = address + u64::try_from(offset * 16).unwrap_or(u64::MAX);
            let encoded = chunk
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!("[REMu][GUEST-CODE] {line_address:#018x}: {encoded}");
        }
        eprintln!("[REMu][GUEST-CODE] end address={address:#018x}");
        if let Some(directory) = &output_directory {
            let path = directory.join(format!("{address:016x}-{length:x}.bin"));
            match std::fs::write(&path, &bytes) {
                Ok(()) => eprintln!("[REMu][GUEST-CODE] wrote {}", path.display()),
                Err(error) => eprintln!(
                    "[REMu][GUEST-CODE] cannot write {}: {error}",
                    path.display()
                ),
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn parse_guest_code_number(value: &str) -> Option<u64> {
    let value = value.trim();
    value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
        .map_or_else(
            || value.parse().ok(),
            |hexadecimal| u64::from_str_radix(hexadecimal, 16).ok(),
        )
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
struct ImportCallFrame {
    integer_arguments: [u64; 6],
    rax: u64,
    guest_rsp: u64,
    guest_return_address: u64,
    dispatch_index: u64,
    xmm_arguments: [u128; 8],
    guest_rbx: u64,
    _reserved: u64,
}

#[derive(Debug)]
struct GuestImportTraceRecord {
    thread: u64,
    name: String,
    active: AtomicBool,
    phase: AtomicU8,
    dispatch_index: AtomicU64,
    integer_argument0: AtomicU64,
    integer_argument1: AtomicU64,
    guest_return_address: AtomicU64,
    guest_rsp: AtomicU64,
    calls: AtomicU64,
    transition_sequence: AtomicU64,
    reported_sequence: AtomicU64,
    transition_nanos: AtomicU64,
}

struct GuestImportTraceGuard {
    record: *const GuestImportTraceRecord,
    dispatch_index: u64,
    integer_argument0: u64,
    integer_argument1: u64,
    guest_return_address: u64,
    guest_rsp: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum GuestImportTracePhase {
    Guest = 0,
    Dispatch = 1,
    AgcDcbParse = 2,
    AgcAcbParse = 3,
    GpuPublication = 4,
    ImportCallbacks = 5,
    FlipPacing = 6,
}

impl GuestImportTracePhase {
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Dispatch,
            2 => Self::AgcDcbParse,
            3 => Self::AgcAcbParse,
            4 => Self::GpuPublication,
            5 => Self::ImportCallbacks,
            6 => Self::FlipPacing,
            _ => Self::Guest,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Guest => "in-guest",
            Self::Dispatch => "in-import-dispatch",
            Self::AgcDcbParse => "agc-dcb-parse",
            Self::AgcAcbParse => "agc-acb-parse",
            Self::GpuPublication => "gpu-publication",
            Self::ImportCallbacks => "import-callbacks",
            Self::FlipPacing => "flip-pacing",
        }
    }
}

pub(crate) struct GuestImportTracePhaseGuard {
    record: *const GuestImportTraceRecord,
    previous: GuestImportTracePhase,
}

impl GuestImportTracePhaseGuard {
    pub(crate) fn enter(phase: GuestImportTracePhase) -> Self {
        let record = GUEST_IMPORT_TRACE_RECORD.with(Cell::get);
        if record.is_null() {
            return Self {
                record,
                previous: GuestImportTracePhase::Guest,
            };
        }
        // SAFETY: the process-lifetime registry owns every trace record referenced by TLS.
        let trace = unsafe { &*record };
        let previous =
            GuestImportTracePhase::from_u8(trace.phase.swap(phase as u8, Ordering::Relaxed));
        trace
            .transition_nanos
            .store(guest_import_trace_nanos(), Ordering::Relaxed);
        trace.transition_sequence.store(
            GUEST_IMPORT_TRACE_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1,
            Ordering::Release,
        );
        Self { record, previous }
    }
}

impl Drop for GuestImportTracePhaseGuard {
    fn drop(&mut self) {
        if self.record.is_null() {
            return;
        }
        // SAFETY: the process-lifetime registry owns every trace record referenced by this guard.
        let trace = unsafe { &*self.record };
        trace.phase.store(self.previous as u8, Ordering::Relaxed);
        trace
            .transition_nanos
            .store(guest_import_trace_nanos(), Ordering::Relaxed);
        trace.transition_sequence.store(
            GUEST_IMPORT_TRACE_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1,
            Ordering::Release,
        );
    }
}

fn trace_guest_threads() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_GUEST_THREADS").is_some_and(|value| value == "1")
    })
}

fn guest_import_trace_nanos() -> u64 {
    u64::try_from(
        GUEST_IMPORT_TRACE_EPOCH
            .get_or_init(Instant::now)
            .elapsed()
            .as_nanos(),
    )
    .unwrap_or(u64::MAX)
}

fn start_guest_import_trace_reporter() {
    GUEST_IMPORT_TRACE_REPORTER.get_or_init(|| {
        std::thread::Builder::new()
            .name("REMu guest import trace".to_owned())
            .spawn(|| {
                const STALE_AFTER: Duration = Duration::from_secs(2);
                let stale_after_nanos =
                    u64::try_from(STALE_AFTER.as_nanos()).unwrap_or(u64::MAX);
                loop {
                    std::thread::sleep(STALE_AFTER);
                    let now = guest_import_trace_nanos();
                    let records = GUEST_IMPORT_TRACE_RECORDS
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone();
                    let telemetry = IMPORT_TELEMETRY
                        .get_or_init(|| {
                            RwLock::new(Arc::new(ImportTelemetry::new(Vec::new(), 0)))
                        })
                        .read()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone();
                    for record in records {
                        if !record.active.load(Ordering::Acquire) {
                            continue;
                        }
                        let transition_sequence =
                            record.transition_sequence.load(Ordering::Acquire);
                        let transition_nanos = record.transition_nanos.load(Ordering::Acquire);
                        let age_nanos = now.saturating_sub(transition_nanos);
                        if age_nanos < stale_after_nanos
                            || record.reported_sequence.swap(
                                transition_sequence,
                                Ordering::AcqRel,
                            ) == transition_sequence
                        {
                            continue;
                        }
                        let dispatch_index = record.dispatch_index.load(Ordering::Acquire);
                        let nid = usize::try_from(dispatch_index)
                            .ok()
                            .and_then(|index| telemetry.nids.get(index))
                            .map_or("?", String::as_str);
                        let phase = GuestImportTracePhase::from_u8(
                            record.phase.load(Ordering::Acquire),
                        );
                        eprintln!(
                            "[REMu][THREAD][STALE] thread={:#x} name={:?} age_ms={} phase={} nid={} import={} arg0={:#x} arg1={:#x} return={:#x} rsp={:#x} calls={} transition={transition_sequence}",
                            record.thread,
                            record.name,
                            age_nanos / 1_000_000,
                            phase.name(),
                            nid,
                            nid_display_name(nid).unwrap_or("unknown"),
                            record.integer_argument0.load(Ordering::Acquire),
                            record.integer_argument1.load(Ordering::Acquire),
                            record.guest_return_address.load(Ordering::Acquire),
                            record.guest_rsp.load(Ordering::Acquire),
                            record.calls.load(Ordering::Acquire),
                        );
                    }
                }
            })
            .expect("guest import trace reporter must start");
    });
}

impl GuestImportTraceGuard {
    fn enter(frame: &ImportCallFrame) -> Option<Self> {
        if !trace_guest_threads() {
            return None;
        }
        start_guest_import_trace_reporter();
        let record = GUEST_IMPORT_TRACE_RECORD.with(|current| {
            let existing = current.get();
            if !existing.is_null() {
                return existing;
            }
            let record = Arc::new(GuestImportTraceRecord {
                thread: current_guest_thread_handle(),
                name: current_guest_thread_name(),
                active: AtomicBool::new(true),
                phase: AtomicU8::new(GuestImportTracePhase::Guest as u8),
                dispatch_index: AtomicU64::new(0),
                integer_argument0: AtomicU64::new(0),
                integer_argument1: AtomicU64::new(0),
                guest_return_address: AtomicU64::new(0),
                guest_rsp: AtomicU64::new(0),
                calls: AtomicU64::new(0),
                transition_sequence: AtomicU64::new(0),
                reported_sequence: AtomicU64::new(0),
                transition_nanos: AtomicU64::new(guest_import_trace_nanos()),
            });
            let pointer = Arc::as_ptr(&record);
            GUEST_IMPORT_TRACE_RECORDS
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(record);
            current.set(pointer);
            pointer
        });
        // SAFETY: the process-lifetime registry retains this record after the current thread
        // publishes its stable pointer in thread-local storage.
        let record = unsafe { &*record };
        record
            .dispatch_index
            .store(frame.dispatch_index, Ordering::Relaxed);
        record
            .integer_argument0
            .store(frame.integer_arguments[0], Ordering::Relaxed);
        record
            .integer_argument1
            .store(frame.integer_arguments[1], Ordering::Relaxed);
        record
            .guest_return_address
            .store(frame.guest_return_address, Ordering::Relaxed);
        record.guest_rsp.store(frame.guest_rsp, Ordering::Relaxed);
        record.calls.fetch_add(1, Ordering::Relaxed);
        record
            .phase
            .store(GuestImportTracePhase::Dispatch as u8, Ordering::Relaxed);
        record
            .transition_nanos
            .store(guest_import_trace_nanos(), Ordering::Relaxed);
        record.transition_sequence.store(
            GUEST_IMPORT_TRACE_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1,
            Ordering::Release,
        );
        Some(Self {
            record,
            dispatch_index: frame.dispatch_index,
            integer_argument0: frame.integer_arguments[0],
            integer_argument1: frame.integer_arguments[1],
            guest_return_address: frame.guest_return_address,
            guest_rsp: frame.guest_rsp,
        })
    }
}

impl Drop for GuestImportTraceGuard {
    fn drop(&mut self) {
        // SAFETY: the process-lifetime registry retains every record published to a guard.
        let record = unsafe { &*self.record };
        record
            .dispatch_index
            .store(self.dispatch_index, Ordering::Relaxed);
        record
            .integer_argument0
            .store(self.integer_argument0, Ordering::Relaxed);
        record
            .integer_argument1
            .store(self.integer_argument1, Ordering::Relaxed);
        record
            .guest_return_address
            .store(self.guest_return_address, Ordering::Relaxed);
        record.guest_rsp.store(self.guest_rsp, Ordering::Relaxed);
        record
            .phase
            .store(GuestImportTracePhase::Guest as u8, Ordering::Relaxed);
        record
            .transition_nanos
            .store(guest_import_trace_nanos(), Ordering::Relaxed);
        record.transition_sequence.store(
            GUEST_IMPORT_TRACE_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1,
            Ordering::Release,
        );
    }
}

fn retire_current_guest_import_trace() {
    GUEST_IMPORT_TRACE_RECORD.with(|current| {
        let record = current.replace(std::ptr::null());
        if !record.is_null() {
            // SAFETY: the process-lifetime registry retains every thread-local record.
            unsafe { &*record }.active.store(false, Ordering::Release);
        }
    });
}

const IMPORT_CALL_FRAME_SIZE: usize = 0xe0;
const _: () = {
    assert!(std::mem::size_of::<ImportCallFrame>() == IMPORT_CALL_FRAME_SIZE);
    assert!(std::mem::align_of::<ImportCallFrame>() == 16);
    assert!(std::mem::offset_of!(ImportCallFrame, rax) == 0x30);
    assert!(std::mem::offset_of!(ImportCallFrame, guest_rsp) == 0x38);
    assert!(std::mem::offset_of!(ImportCallFrame, guest_return_address) == 0x40);
    assert!(std::mem::offset_of!(ImportCallFrame, dispatch_index) == 0x48);
    assert!(std::mem::offset_of!(ImportCallFrame, xmm_arguments) == 0x50);
    assert!(std::mem::offset_of!(ImportCallFrame, guest_rbx) == 0xd0);
};

#[repr(C, align(16))]
struct GuestExceptionContext([u8; 0x500]);

fn write_exception_context_word(context: &mut GuestExceptionContext, offset: usize, value: u64) {
    context.0[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn deliver_pending_guest_exception_at_import(frame: &ImportCallFrame) {
    let thread = current_guest_thread_handle();
    if thread == 0 || (!cfg!(test) && !GUEST_EXCEPTION_PENDING.load(Ordering::Acquire)) {
        return;
    }
    let pending = {
        let mut delivery = guest_exception_delivery()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if delivery.active.contains(&thread) {
            return;
        }
        let Some(pending) = delivery.pending.remove(&thread) else {
            return;
        };
        GUEST_EXCEPTION_PENDING.store(!delivery.pending.is_empty(), Ordering::Release);
        delivery.active.insert(thread);
        pending
    };

    let mut context = GuestExceptionContext([0; 0x500]);
    const MCONTEXT: usize = 0x40;
    for (offset, value) in [
        (0x08, frame.integer_arguments[0]),
        (0x10, frame.integer_arguments[1]),
        (0x18, frame.integer_arguments[2]),
        (0x20, frame.integer_arguments[3]),
        (0x28, frame.integer_arguments[4]),
        (0x30, frame.integer_arguments[5]),
        (0x38, frame.rax),
    ] {
        write_exception_context_word(&mut context, MCONTEXT + offset, value);
    }
    write_exception_context_word(&mut context, MCONTEXT + 0xa0, frame.guest_return_address);
    write_exception_context_word(&mut context, MCONTEXT + 0xb8, frame.guest_rsp);
    write_exception_context_word(&mut context, MCONTEXT + 0xc8, 0x480);
    write_exception_context_word(
        &mut context,
        MCONTEXT + 0x440,
        current_guest_thread_pointer(),
    );

    eprintln!(
        "[REMu][KERNEL] delivering cooperative exception target={thread:#x} type={} handler={:#x} rip={:#x} rsp={:#x}",
        pending.exception, pending.handler, frame.guest_return_address, frame.guest_rsp,
    );
    type GuestExceptionHandler = unsafe extern "C" fn(i32, *mut u8);
    // SAFETY: The process installed this mapped executable guest handler for the requested signal.
    // Delivery runs at an import boundary on the target's own native thread, so its guest TLS and
    // stack identity remain intact while the handler observes the synthesized Orbis ucontext.
    let handler: GuestExceptionHandler = unsafe { std::mem::transmute(pending.handler) };
    // SAFETY: `context` is live and writable for the complete synchronous handler invocation.
    unsafe { handler(pending.exception, context.0.as_mut_ptr()) };

    guest_exception_delivery()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .active
        .remove(&thread);
}

#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "C" fn import_dispatch_trampoline() {
    core::arch::naked_asm!(
        "sub rsp, 0xe0",
        "mov [rsp + 0x00], rdi",
        "mov [rsp + 0x08], rsi",
        "mov [rsp + 0x10], rdx",
        "mov [rsp + 0x18], rcx",
        "mov [rsp + 0x20], r8",
        "mov [rsp + 0x28], r9",
        "mov [rsp + 0x30], rax",
        "lea r10, [rsp + 0xe8]",
        "mov [rsp + 0x38], r10",
        "mov r10, [rsp + 0xe8]",
        "mov [rsp + 0x40], r10",
        "mov r10, [rsp + 0xe0]",
        "mov [rsp + 0x48], r10",
        "movdqu [rsp + 0x50], xmm0",
        "movdqu [rsp + 0x60], xmm1",
        "movdqu [rsp + 0x70], xmm2",
        "movdqu [rsp + 0x80], xmm3",
        "movdqu [rsp + 0x90], xmm4",
        "movdqu [rsp + 0xa0], xmm5",
        "movdqu [rsp + 0xb0], xmm6",
        "movdqu [rsp + 0xc0], xmm7",
        "mov [rsp + 0xd0], rbx",
        "mov qword ptr [rsp + 0xd8], 0",
        "mov rdi, rsp",
        "call {dispatch}",
        "add rsp, 0xe0",
        "add rsp, 8",
        "ret",
        dispatch = sym import_dispatch_entry,
    );
}

#[cfg(all(test, target_arch = "x86_64"))]
const TEST_CAPTURE_DISPATCH_INDEX: u64 = i32::MAX as u64;

#[cfg(all(test, target_arch = "x86_64"))]
static TEST_CAPTURED_IMPORT_FRAME: Mutex<Option<ImportCallFrame>> = Mutex::new(None);

#[cfg(all(test, target_arch = "x86_64"))]
#[unsafe(naked)]
unsafe extern "C" fn invoke_test_import_stub(_stub: u64) -> ProbeImportReturn {
    core::arch::naked_asm!(
        "mov r11, rdi",
        "push rbx",
        "mov rbx, 0x7172737475767778",
        "mov rdi, 0x11",
        "mov rsi, 0x22",
        "mov rdx, 0x33",
        "mov rcx, 0x44",
        "mov r8, 0x55",
        "mov r9, 0x66",
        "mov rax, 0x0102030405060708",
        "mov r10, 0x101",
        "movq xmm0, r10",
        "mov r10, 0x202",
        "movq xmm1, r10",
        "mov r10, 0x303",
        "movq xmm2, r10",
        "mov r10, 0x404",
        "movq xmm3, r10",
        "mov r10, 0x505",
        "movq xmm4, r10",
        "mov r10, 0x606",
        "movq xmm5, r10",
        "mov r10, 0x707",
        "movq xmm6, r10",
        "mov r10, 0x808",
        "movq xmm7, r10",
        "call r11",
        "pop rbx",
        "ret",
    );
}

#[cfg(not(target_arch = "x86_64"))]
extern "C" fn import_dispatch_trampoline() {
    std::hint::black_box(
        import_dispatch_entry as extern "C" fn(*const ImportCallFrame) -> ProbeImportReturn,
    );
    unreachable!("x86-64 guest import stubs cannot execute on this host architecture");
}

fn low_xmm_f64(value: u128) -> f64 {
    let bytes = value.to_le_bytes();
    f64::from_bits(u64::from_le_bytes(
        bytes[..8].try_into().expect("fixed XMM lane"),
    ))
}

#[allow(clippy::too_many_lines)]
/// Whether every failing guest import is reported with its arguments.
///
/// Titles routinely treat a refused call as fatal, so the first error a subsystem receives is
/// usually the defect; without this the failure is only visible as the assert it triggers.
fn trace_import_errors() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_IMPORT_ERRORS").is_some_and(|value| value == "1")
    })
}

fn trace_dispatch_stalls() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_DISPATCH_STALLS").is_some_and(|value| value == "1")
    })
}

/// How many times each dispatch site has reported a failure, so the log stays bounded.
static IMPORT_FAILURE_COUNTS: OnceLock<Mutex<BTreeMap<u64, u64>>> = OnceLock::new();

/// Reports an import that returned a console error code or a negative status.
fn report_failed_import(frame: *const ImportCallFrame) -> ProbeImportReturn {
    // SAFETY: the caller supplies one aligned, fully initialized frame that stays live for the
    // complete call and checked for null.
    let (index, arguments) = unsafe { ((*frame).dispatch_index, (*frame).integer_arguments) };
    let result = import_dispatch(frame);
    let value = result.integer;
    // Kernel errors reach the guest sign-extended from a negative 32-bit status. Graphics
    // errors instead carry their own facility in a plain 32-bit value, so match that facility
    // directly rather than widening the sign test and catching ordinary counters.
    let failed = value >> 32 == 0xffff_ffff || value >> 16 == 0x8a6c;
    // Reporting is unconditional: a failed import is the single most useful line in a bring-up
    // log, and gating it behind an environment variable meant the runs that needed it were the
    // ones without it. Volume is bounded per call site instead — the first failure of each is
    // always shown, then powers of two — so a hot call that legitimately fails, a trylock or a
    // file probe, cannot bury the rest. `REMU_TRACE_IMPORT_ERRORS=1` removes the bound.
    let failed = failed && {
        let seen = IMPORT_FAILURE_COUNTS
            .get_or_init(|| Mutex::new(BTreeMap::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(index)
            .and_modify(|count| *count += 1_u64)
            .or_insert(1_u64)
            .to_owned();
        trace_import_errors() || seen.is_power_of_two()
    };
    if failed {
        // SAFETY: the telemetry Arc is retained by this thread's cache for the whole call.
        let telemetry = unsafe { &*cached_import_telemetry() };
        let nid = usize::try_from(index)
            .ok()
            .and_then(|index| telemetry.nids.get(index))
            .map_or("?", String::as_str);
        eprintln!(
            "[REMu][FAIL] nid={nid} name={} thread={:?} result={value:#x} args={:#x},{:#x},{:#x},{:#x}",
            nid_display_name(nid).unwrap_or("unknown"),
            current_guest_thread_name(),
            arguments[0],
            arguments[1],
            arguments[2],
            arguments[3],
        );
    }
    result
}

/// Number of leading guest imports to trace with their return values.
///
/// Boot-order defects are easier to read as an exact call/return sequence than as first-hit
/// records, because a wrong result only becomes visible in the guest's next branch. The trace is
/// bounded so it cannot dominate a long run.
fn import_call_trace_limit() -> u64 {
    static LIMIT: OnceLock<u64> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("REMU_TRACE_IMPORT_CALLS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0)
    })
}

static IMPORT_CALL_TRACE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
/// Guards the one-time process-wide condition summary printed with the first stall report.
static STALL_SNAPSHOT_EMITTED: AtomicBool = AtomicBool::new(false);

extern "C" fn import_dispatch_entry(frame: *const ImportCallFrame) -> ProbeImportReturn {
    let _guest_import_trace = (!frame.is_null())
        // SAFETY: the import trampoline supplies one live frame for the complete dispatch.
        .then(|| GuestImportTraceGuard::enter(unsafe { &*frame }))
        .flatten();
    let limit = import_call_trace_limit();
    if limit == 0 {
        if !frame.is_null() {
            return report_failed_import(frame);
        }
        return import_dispatch(frame);
    }
    let sequence = IMPORT_CALL_TRACE_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1;
    if sequence > limit || frame.is_null() {
        return import_dispatch(frame);
    }
    // SAFETY: the caller supplies one aligned, fully initialized frame that stays live for the
    // complete call, and the null case returned above.
    let (index, arguments, guest_return_address) = unsafe {
        (
            (*frame).dispatch_index,
            (*frame).integer_arguments,
            (*frame).guest_return_address,
        )
    };
    // SAFETY: the telemetry Arc is retained by this thread's cache for the whole call.
    let telemetry = unsafe { &*cached_import_telemetry() };
    let nid = usize::try_from(index)
        .ok()
        .and_then(|index| telemetry.nids.get(index))
        .map_or("?", String::as_str);
    let name = nid_display_name(nid).unwrap_or("unknown");
    eprintln!(
        "[REMu][CALL] seq={sequence} nid={nid} name={name} thread={} return={guest_return_address:#x} args={:#x},{:#x},{:#x},{:#x},{:#x},{:#x}",
        current_guest_thread_handle(),
        arguments[0],
        arguments[1],
        arguments[2],
        arguments[3],
        arguments[4],
        arguments[5],
    );
    let result = import_dispatch(frame);
    eprintln!(
        "[REMu][CALL] seq={sequence} nid={nid} name={name} result={:#x}",
        result.integer
    );
    result
}

fn read_import_stack_arguments(frame: &ImportCallFrame) -> [u64; 6] {
    std::array::from_fn(|argument| {
        let offset = u64::try_from((argument + 1) * std::mem::size_of::<u64>())
            .expect("fixed stack argument offset fits u64");
        read_guest_u64(frame.guest_rsp.saturating_add(offset))
    })
}

fn is_serialized_ampr_import(nid: &str) -> bool {
    matches!(
        nid,
        // File resolution and metadata.
        "gEpBkcwxUjw"
            | "WT-5NKy42fw"
            | "eYAh2vlCY-U"
            | "QzB4O+bJQyA"
            | "i3HWvW35jao"
            | "w5fcCG+t31g"
            | "VB-BtuIW8Xc"
            | "C+Khtbbx2g8"
            | "WvEu7yl3Ivg"
            | "ApkYaHb8Sek"
            // Submission and completion.
            | "eE4Szl8sil8"
            | "qvMUCyyaCSI"
            | "ASoW5WE-UPo"
            | "rqwFKI4PAiM"
            | "bBfz7kMF2Ho"
            // Command-buffer construction, recording, and destruction.
            | "8aI7R7WaOlc"
            | "a8uLzYY--tM"
            | "N-FSPA4S3nI"
            | "baQO9ez2gL4"
            | "ULvXMDz56po"
            | "tZDDEo2tE5k"
            | "GnxKOHEawhk"
            | "H896Pt-yB4I"
            | "sSAUCCU1dv4"
            | "vWU-odnS+fU"
            | "GuchCTefuZw"
            | "Qs1xtplKo0U"
            | "mQ16-QdKv7k"
            // APR program builders implemented by the Ampr namespace.
            | "YPxkUDhgoNI"
            | "tNn5WBkta60"
            | "pFQ9UHpO52s"
            | "GmOguNIsuKk"
            | "4UkZbYKVF7c"
            | "4quckD2y7Pg"
            | "sWbST0oQKsc"
            | "dXPaz65HNmk"
            | "f12ObAMEi9A"
            | "mv0O8Zg0woU"
            | "o67gODLFpls"
            | "sJXyWHjP-F8"
    )
}

fn is_ordered_ampr_submit_import(nid: &str) -> bool {
    matches!(nid, "eE4Szl8sil8" | "qvMUCyyaCSI" | "ASoW5WE-UPo")
}

extern "C" fn import_dispatch(frame: *const ImportCallFrame) -> ProbeImportReturn {
    if frame.is_null() {
        eprintln!("[REMu][IMPORT] null captured register frame");
        terminate_probe(87);
    }
    // SAFETY: import_dispatch_trampoline supplies one aligned, fully initialized frame that stays
    // live until this call returns.
    let frame = unsafe { &*frame };
    #[cfg(all(test, target_arch = "x86_64"))]
    if frame.dispatch_index == TEST_CAPTURE_DISPATCH_INDEX {
        *TEST_CAPTURED_IMPORT_FRAME
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(*frame);
        return ProbeImportReturn {
            integer: frame.rax,
            float: low_xmm_f64(frame.xmm_arguments[7]),
        };
    }
    deliver_pending_guest_exception_at_import(frame);
    let call_args = frame.integer_arguments;
    let float_argument0 = low_xmm_f64(frame.xmm_arguments[0]);
    let index = frame.dispatch_index;
    let guest_return_address = frame.guest_return_address;
    // `configure_probe_dispatch` replaces telemetry only before guest threads start. The current
    // thread retains the Arc in `IMPORT_TELEMETRY_CACHE` for the complete call, so this pointer
    // remains live while avoiding a contended RwLock and atomic Arc clone on every guest import.
    let telemetry = unsafe { &*cached_import_telemetry() };
    let Some(index) = usize::try_from(index).ok() else {
        eprintln!("[REMu][IMPORT] invalid stub index {index}");
        terminate_probe(87);
    };
    let Some(nid) = telemetry.nids.get(index).map(String::as_str) else {
        eprintln!("[REMu][IMPORT] invalid stub index {index}");
        terminate_probe(87);
    };
    if matches!(nid, "3kg7rT0NQIs" | "FJrT5LuUBAU") {
        guest_pthread_exit(call_args[0]);
    }
    let hit = telemetry.hits.fetch_add(1, Ordering::Relaxed) + 1;
    telemetry.counts[index].fetch_add(1, Ordering::Relaxed);
    if nid == "fgxnMeTNUtY"
        && *TRACE_HOT_IMPORTS.get_or_init(|| {
            std::env::var_os("REMU_TRACE_HOT_IMPORTS").is_some_and(|value| value == "1")
        })
    {
        PROCESS_TIME_CALLSITES.with(|callsites| {
            let mut callsites = callsites.borrow_mut();
            let slot = callsites
                .iter()
                .position(|(address, _)| *address == guest_return_address)
                .or_else(|| callsites.iter().position(|(address, _)| *address == 0))
                .unwrap_or_else(|| {
                    callsites
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, (_, count))| *count)
                        .map_or(0, |(index, _)| index)
                });
            if callsites[slot].0 != guest_return_address {
                callsites[slot] = (guest_return_address, 0);
            }
            callsites[slot].1 = callsites[slot].1.saturating_add(1);
            let count = callsites[slot].1;
            if count >= 1_024 && count.is_power_of_two() {
                eprintln!(
                    "[REMu][IMPORT][TIMER] return={guest_return_address:#x} calls={count} rdi={:#x} rsi={:#x}",
                    call_args[0], call_args[1]
                );
            }
        });
    }
    if hit >= (1 << 20)
        && hit.is_power_of_two()
        && *TRACE_HOT_IMPORTS.get_or_init(|| {
            std::env::var_os("REMU_TRACE_HOT_IMPORTS").is_some_and(|value| value == "1")
        })
    {
        let summary = telemetry
            .hot_imports(8)
            .into_iter()
            .map(|(nid, count)| {
                format!(
                    "{}:{}={count}",
                    nid_display_name(nid).unwrap_or("unknown"),
                    nid
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!("[REMu][IMPORT] hits={hit} hot=[{summary}]");
    }
    // After the first call for this NID the flag never changes again, so a relaxed load keeps the
    // steady state a plain read instead of an atomic read-modify-write on every import.
    if !telemetry.seen[index].load(Ordering::Relaxed)
        && !telemetry.seen[index].swap(true, Ordering::Relaxed)
    {
        let name = nid_display_name(nid).unwrap_or("unknown");
        eprintln!(
            "[REMu][IMPORT] first_hit={} nid={} name={} return={guest_return_address:#x} args={:#x},{:#x},{:#x},{:#x},{:#x},{:#x}",
            hit,
            nid,
            name,
            call_args[0],
            call_args[1],
            call_args[2],
            call_args[3],
            call_args[4],
            call_args[5],
        );
    }
    if telemetry.maximum_hits != 0 && hit >= telemetry.maximum_hits {
        eprintln!("[REMu][PROBE] import hit limit reached at nid={nid}");
        for (nid, count) in telemetry.hot_imports(8) {
            eprintln!("[REMu][PROBE] hot_import nid={nid} hits={count}");
        }
        terminate_probe(86);
    }
    if matches!(nid, "uMei1W9uyNo" | "6Z83sYWFlA8") {
        let status = low_i32(call_args[0]);
        let run_callbacks = nid == "uMei1W9uyNo";
        if run_callbacks {
            run_all_guest_exit_callbacks();
        }
        eprintln!(
            "[REMu][PROBE] guest requested {}({status})",
            if run_callbacks { "exit" } else { "_exit" }
        );
        terminate_probe(status);
    }
    if nid == "T72hz6ffq08" {
        static RENDER_THREAD_YIELD_TRACE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
        const BLOCKING_RENDER_QUERY_YIELD_RETURN: u64 = 0x8011_42b17;
        // Naming the guest thread allocates, and this is the hottest spin-wait import in an
        // Unreal TaskGraph title, so test the trace gate before paying for the name.
        let thread_name = trace_guest_sync()
            .then(current_guest_thread_name)
            .unwrap_or_default();
        if thread_name.contains("RenderThread") {
            let sequence = RENDER_THREAD_YIELD_TRACE_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1;
            if trace_logarithmic_sequence(sequence, 32) {
                let stack = std::array::from_fn::<_, 32, _>(|index| {
                    try_read_process_guest_u64(
                        frame
                            .guest_rsp
                            .saturating_add(u64::try_from(index).unwrap_or(0).saturating_mul(8)),
                    )
                    .unwrap_or(0)
                });
                if guest_return_address == BLOCKING_RENDER_QUERY_YIELD_RETURN {
                    // The materialized query-result routine keeps its live query object in RBX.
                    // stack[4] is the caller's RBX saved by that routine's prologue, not the
                    // current object. The object owns a counter payload at +0x28 and a separate
                    // completion-label pointer at +0x30.
                    let query_object = frame.guest_rbx;
                    let query_payload = query_object
                        .checked_add(0x28)
                        .and_then(try_read_process_guest_u64);
                    let query_completion = query_object
                        .checked_add(0x30)
                        .and_then(try_read_process_guest_u64);
                    let query_completion_value = query_completion
                        .filter(|address| *address != 0)
                        .and_then(try_read_process_guest_u64);
                    eprintln!(
                        "[REMu][PTHREAD][YIELD] sequence={sequence} thread={thread_name:?}/{:#x} return={guest_return_address:#x} rsp={:#x} stack={stack:#x?} query_object={query_object:#x} query_payload={query_payload:#x?} query_completion={query_completion:#x?} query_completion_value={query_completion_value:#x?}",
                        current_guest_thread_handle(),
                        frame.guest_rsp,
                    );
                } else {
                    eprintln!(
                        "[REMu][PTHREAD][YIELD] sequence={sequence} thread={thread_name:?}/{:#x} return={guest_return_address:#x} rsp={:#x} stack={stack:#x?}",
                        current_guest_thread_handle(),
                        frame.guest_rsp,
                    );
                }
            }
        }
        std::thread::yield_now();
        return ProbeImportReturn::integer(0);
    }
    if matches!(nid, "4J2sUJmuHZQ" | "fgxnMeTNUtY" | "BNowx2l588E") {
        // Process timers are called by FMOD's mixer and streaming workers. They must not queue
        // behind graphics submission on the process-wide dispatch lock or audio production can
        // stall for an entire frame even though reading the monotonic clock is independent.
        let result = match nid {
            "4J2sUJmuHZQ" => kernel_process_time(false),
            "fgxnMeTNUtY" => kernel_process_time(true),
            "BNowx2l588E" => 1_000_000_000,
            _ => unreachable!("the process-timer NID set is exhaustive"),
        };
        return ProbeImportReturn::integer(result);
    }
    if is_guest_sync_import(nid) {
        let is_mutex_init = matches!(nid, "cmo1RIYva9o" | "ttHNfU+qDBU");
        let mutex_attributes = if is_mutex_init {
            let state = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
            let state = state
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // Missing attr objects fall back to Gen5 defaults. Titles sometimes pass a stack
            // attribute that was never routed through scePthreadMutexattrInit (zero-init / copy).
            Some(
                pthread_mutex_attr_state(call_args[1], &state)
                    .unwrap_or_default(),
            )
        } else {
            None
        };
        let result =
            dispatch_guest_sync_import_at_import(nid, call_args, mutex_attributes, Some(frame));
        // A kernel exception can interrupt a thread parked inside a synchronization import. The
        // wait returns as a legal spurious wake, then the target runs its handler on this exact
        // native thread before control crosses back into guest code.
        deliver_pending_guest_exception_at_import(frame);
        return ProbeImportReturn::integer(result);
    }
    if matches!(
        nid,
        "5OqszGpy7Mg" | "zlfEH8FmyUA" | "VOBg+iNwB-4" | "2vDqwBlpF-o"
    ) {
        return dispatch_guest_numeric_parser(nid, call_args);
    }
    if nid == "pztV4AF18iI" {
        return ProbeImportReturn::integer(guest_sincosf(
            low_f32(float_argument0),
            call_args[0],
            call_args[1],
        ));
    }
    if nid == "AEJdIVZTEmo" {
        return ProbeImportReturn::integer(guest_qsort(call_args));
    }
    if nid == "DYhhWbJSeRg" {
        return ProbeImportReturn::integer(platform_services::video_out_color_settings_set_gamma(
            call_args[0],
            low_f32(float_argument0),
        ));
    }
    if nid == "dhK16CKwhQg" {
        return ProbeImportReturn::integer(u64::from(float_argument0.is_finite()));
    }
    if matches!(
        nid,
        "JfEPXVxhFqA"
            | "ekNvsT22rsY"
            | "s1--uE9mBFw"
            | "b+uAV89IlxE"
            | "QOQtbeDqsT4"
            | "GrQ9s4IrNaQ"
            | "g2tViFIohHE"
            | "R7d0F1g2qsU"
            | "gatEUKG+Ea4"
            | "DImz2Ft9E2g"
            | "t5YrizufpQc"
            | "pDmme7Bgm6E"
            | "0x6o1VVAYSY"
            | "xywYcRB7nbQ"
            | "JK2wamZPzwM"
            | "IaZXJ9M79uo"
            | "cd+Rtw+D1x8"
            | "on6ZH7Abo10"
            | "8XTArSPyWHk"
            | "PE2zHMqLSHs"
            | "aII9h5nli9U"
            | "XHl38ZNknbs"
            | "v8iOE+j8a5o"
    ) {
        return ProbeImportReturn::integer(dispatch_audio_out_import(nid, call_args));
    }
    if let Some(result) = dispatch_pad_import(nid, call_args) {
        // Controller lifecycle and polling belong to one small service-local lock. In particular,
        // the per-frame read path must not queue behind command-buffer translation, which owns no
        // pad state and can take tens of milliseconds under shader-heavy workloads.
        return ProbeImportReturn::integer(result);
    }
    // Most hot imports return above without consuming stack arguments. Reading them lazily avoids
    // six guest-memory loads on every timer, synchronization, numeric, audio, and pad call while
    // preserving the complete ABI frame for the less common dispatch paths below.
    let stack_args = read_import_stack_arguments(frame);
    if let Some(result) = dispatch_stateless_import(nid, call_args, stack_args) {
        return ProbeImportReturn::integer(result);
    }
    if let Some(result) = dispatch_kernel_fd_import(nid, call_args) {
        return ProbeImportReturn::integer(result);
    }
    if nid == "fzyMKs9kim0" {
        // An untimed equeue wait must park without retaining the process-wide dispatch lock.
        // Graphics submission, flip, and completion producers all need that lock briefly to
        // publish their events; holding it here would create the exact producer/consumer
        // deadlock the wait is meant to resolve.
        let result = kernel_wait_equeue_blocking(call_args);
        deliver_pending_guest_exception_at_import(frame);
        return ProbeImportReturn::integer(result);
    }
    if nid == "wzvqT4UqKX8" {
        // Module initialization can itself call guest imports. Resolve and mark the object while
        // holding the process lock, then invoke its DT_INIT entries after releasing that lock.
        return ProbeImportReturn::integer(kernel_load_start_module(call_args));
    }
    if matches!(nid, "o3+RWnHViSg" | "JdksQu8pNdQ") {
        prepare_av_player_video_buffers(call_args[0], nid == "JdksQu8pNdQ");
    }
    if nid == "mQ16-QdKv7k" {
        return ProbeImportReturn::integer(ampr_command_buffer_read_file_ordered(
            call_args, stack_args,
        ));
    }
    if is_ordered_ampr_submit_import(nid) {
        return ProbeImportReturn::integer(ampr_submit_command_buffer_ordered(call_args));
    }
    let _ampr_order = is_serialized_ampr_import(nid).then(|| {
        ampr_command_order(call_args[0])
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    });
    let state = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
    let trace_lock_stall = trace_dispatch_stalls();
    let lock_started = trace_lock_stall.then(Instant::now);
    let mut state = lock_probe_dispatch_for_write(state);
    if let Some(lock_started) = lock_started {
        let waited = lock_started.elapsed();
        if waited >= Duration::from_millis(2)
            && DISPATCH_LOCK_STALL_LOGS.fetch_add(1, Ordering::Relaxed) < 64
        {
            eprintln!(
                "[REMu][IMPORT][LOCK] waited={:.3}ms nid={nid} name={} thread={:?} guest={:#x}",
                waited.as_secs_f64() * 1_000.0,
                nid_display_name(nid).unwrap_or("unknown"),
                std::thread::current().name().unwrap_or("unnamed"),
                current_guest_thread_handle(),
            );
        }
    }
    // Flip completions are produced asynchronously by Metal. Refresh them at every stateful import
    // boundary instead of only when the guest happens to poll an equeue; VideoOut pending queries
    // and AVPlayer's completed-vblank clock otherwise observe arbitrarily stale state and can hold
    // or skip visible frames. AGC submission completion remains equeue-owned.
    refresh_completed_video_out_flips(&mut state);
    if nid == "H2e8t5ScQGc" {
        let callbacks = take_exit_callbacks(&mut state, Some(call_args[0]));
        drop(state);
        invoke_guest_exit_callbacks(callbacks);
        return ProbeImportReturn::integer(0);
    }
    if matches!(nid, "6UgtwV+0zb4" | "OxhIB8LB-PQ" | "Jmi+9w9u0E4") {
        if call_args[0] == 0 || call_args[2] == 0 {
            return ProbeImportReturn::integer(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
        }
        let attributes = pthread_attr_state(call_args[1], &state).unwrap_or_default();
        let entry = call_args[2];
        let thread_argument = call_args[3];
        let guest_tls = clone_guest_tls(&state.tls_template);
        // The plain POSIX entry point has no name argument; both named forms pass it fifth.
        let thread_name = matches!(nid, "6UgtwV+0zb4" | "Jmi+9w9u0E4")
            .then(|| read_guest_cstring(call_args[4]))
            .flatten()
            .filter(|name| !name.is_empty());
        let Some(handle) =
            allocate_pthread_thread_object(&mut state, thread_name.as_deref().unwrap_or_default())
        else {
            return ProbeImportReturn::integer(sce_kernel_error(SCE_KERNEL_ERROR_TRY_AGAIN));
        };
        let mut builder = std::thread::Builder::new();
        let guest_stack_size = usize::try_from(attributes.stack_size).unwrap_or(0);
        let page_alignment = usize::try_from(MACOS_GUEST_PAGE_ALIGNMENT).unwrap_or(0x4000);
        let host_stack_size = if guest_stack_size >= page_alignment {
            guest_stack_size.max(HOST_PTHREAD_STACK_FLOOR)
        } else {
            0
        };
        if host_stack_size != 0 {
            builder = builder.stack_size(host_stack_size);
            if host_stack_size > guest_stack_size {
                eprintln!(
                    "[REMu][THREAD] raised host stack name={:?} guest={guest_stack_size:#x} host={host_stack_size:#x}",
                    thread_name.as_deref().unwrap_or(""),
                );
            }
        }
        if let Some(name) = thread_name.as_ref() {
            builder = builder.name(name.clone());
        }
        if trace_guest_sync() {
            let (inner_entry, inner_argument) =
                if process_guest_range_is_readable(thread_argument, 2 * std::mem::size_of::<u64>())
                {
                    (
                        read_guest_u64(thread_argument),
                        read_guest_u64(thread_argument + 8),
                    )
                } else {
                    (0, 0)
                };
            eprintln!(
                "[REMu][THREAD] create handle={handle} name={:?} entry={entry:#x} arg={thread_argument:#x} inner={inner_entry:#x}/{inner_argument:#x} detach={} stack={:#x} host_stack={host_stack_size:#x}",
                thread_name.as_deref().unwrap_or(""),
                attributes.detach_state,
                attributes.stack_size
            );
        }
        let trace_name = thread_name.clone();
        let Ok(thread) = builder.spawn(move || {
            if trace_guest_sync() {
                eprintln!(
                    "[REMu][THREAD] start handle={handle} name={:?}",
                    trace_name.as_deref().unwrap_or("")
                );
            }
            set_current_guest_thread_handle(handle);
            set_current_guest_thread_name(trace_name.clone());
            set_current_guest_thread_pointer(guest_tls.thread_pointer);
            let result = run_guest_thread(entry, thread_argument);
            run_guest_thread_destructors();
            retire_current_guest_import_trace();
            set_current_guest_thread_pointer(0);
            set_current_guest_thread_handle(0);
            drop(guest_tls);
            if trace_guest_sync() {
                eprintln!(
                    "[REMu][THREAD] exit handle={handle} name={:?} result={result:#x}",
                    trace_name.as_deref().unwrap_or("")
                );
            }
            retire_detached_guest_thread(handle);
            result
        }) else {
            // A refused thread is invisible to the guest beyond an error code it may not check,
            // and the work that thread was meant to perform then never happens.
            eprintln!(
                "[REMu][THREAD] host spawn refused name={:?} entry={entry:#x} stack={:#x}",
                thread_name.as_deref().unwrap_or(""),
                attributes.stack_size
            );
            if let Some(object) = state.pthread_threads.remove(&handle) {
                guest_free(object.handle);
            }
            return ProbeImportReturn::integer(sce_kernel_error(SCE_KERNEL_ERROR_TRY_AGAIN));
        };
        write_guest_u64(call_args[0], handle);
        state.thread_attributes.insert(handle, attributes);
        if attributes.detach_state == 0 {
            state.thread_handles.insert(handle, thread);
        } else {
            state.detached_threads.insert(handle);
            drop(thread);
        }
        return ProbeImportReturn::integer(0);
    }
    if matches!(nid, "onNY9Byn-W8" | "h9CcP3J0oVM") {
        let posix = nid == "h9CcP3J0oVM";
        if call_args[0] == 0 || state.detached_threads.contains(&call_args[0]) {
            return ProbeImportReturn::integer(join_result(
                sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT),
                posix,
            ));
        }
        let thread_id = call_args[0];
        let handle = state.thread_handles.remove(&thread_id);
        let Some(handle) = handle else {
            return ProbeImportReturn::integer(join_result(
                sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND),
                posix,
            ));
        };
        let object = state.pthread_threads.remove(&thread_id);
        state.thread_attributes.remove(&call_args[0]);
        drop(state);
        let return_value = handle.join().unwrap_or(u64::MAX);
        if let Some(object) = object {
            guest_free(object.handle);
        }
        if call_args[1] != 0 {
            write_guest_u64(call_args[1], return_value);
        }
        return ProbeImportReturn::integer(0);
    }
    if matches!(nid, "4qGrR6eoP9Y" | "+U1R4WtXvoc") {
        let handle = call_args[0];
        if handle == 0 || state.detached_threads.contains(&handle) {
            return ProbeImportReturn::integer(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
        }
        let Some(thread) = state.thread_handles.remove(&handle) else {
            return ProbeImportReturn::integer(sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND));
        };
        let finished = thread.is_finished();
        state.detached_threads.insert(handle);
        if let Some(attributes) = state.thread_attributes.get_mut(&handle) {
            attributes.detach_state = 1;
        }
        drop(thread);
        if finished {
            state.detached_threads.remove(&handle);
            state.thread_attributes.remove(&handle);
            if let Some(object) = state.pthread_threads.remove(&handle) {
                guest_free(object.handle);
            }
        }
        return ProbeImportReturn::integer(0);
    }
    // Asynchronous AvPlayer notifications belong to the controller thread and must not be drained
    // here: a guest event handler may block, and a guest thread that delivers one re-enters its own
    // AvPlayer frame. Playback-state callbacks are different, because the native implementation
    // emits Play/Pause/Stop synchronously from their initiating game-thread calls. Keeping those in
    // a process-wide deferred queue would let an unrelated renderer or audio worker steal the
    // callback and mutate its thread-local media state, so they are taken back below by delivery.
    let dispatch_started = trace_lock_stall.then(Instant::now);
    begin_deferred_gpu_publication();
    let result = dispatch_probe_import(nid, call_args, stack_args, &mut state);
    let av_player_calling_thread_callbacks = take_av_player_calling_thread_callbacks(&mut state);
    let av_player_texture_deallocations =
        std::mem::take(&mut state.av_player.pending_texture_deallocations);
    let gpu_publication = take_deferred_gpu_publication();
    if let Some(dispatch_started) = dispatch_started {
        let held = dispatch_started.elapsed();
        if held >= Duration::from_millis(2)
            && DISPATCH_LOCK_HOLD_LOGS.fetch_add(1, Ordering::Relaxed) < 64
        {
            eprintln!(
                "[REMu][IMPORT][HOLD] held={:.3}ms nid={nid} name={} thread={:?} guest={:#x}",
                held.as_secs_f64() * 1_000.0,
                nid_display_name(nid).unwrap_or("unknown"),
                std::thread::current().name().unwrap_or("unnamed"),
                current_guest_thread_handle(),
            );
        }
    }
    drop(state);
    let publication_succeeded = {
        let _phase = GuestImportTracePhaseGuard::enter(GuestImportTracePhase::GpuPublication);
        flush_deferred_gpu_publication(gpu_publication)
    };
    if !publication_succeeded {
        eprintln!(
            "[REMu][GPU] presenter became unavailable while publishing events from {}",
            nid_display_name(nid).unwrap_or(nid)
        );
    }
    {
        let _phase = GuestImportTracePhaseGuard::enter(GuestImportTracePhase::ImportCallbacks);
        invoke_av_player_event_callbacks(av_player_calling_thread_callbacks);
        invoke_av_player_texture_deallocations(av_player_texture_deallocations);
    }
    // Paced after the dispatch lock is released: the submitting thread waits for its own vblank,
    // exactly as it would on the console, without stalling other guest threads.
    let queued_flips = QUEUED_FLIPS.with(|queued| queued.replace(0));
    {
        let _phase = GuestImportTracePhaseGuard::enter(GuestImportTracePhase::FlipPacing);
        for _ in 0..queued_flips {
            pace_video_out_flip();
        }
    }
    ProbeImportReturn::integer(result)
}

fn dispatch_audio_out_import(nid: &str, arguments: [u64; 6]) -> u64 {
    if matches!(
        nid,
        "g2tViFIohHE"
            | "R7d0F1g2qsU"
            | "gatEUKG+Ea4"
            | "DImz2Ft9E2g"
            | "t5YrizufpQc"
            | "pDmme7Bgm6E"
            | "0x6o1VVAYSY"
            | "xywYcRB7nbQ"
            | "JK2wamZPzwM"
            | "IaZXJ9M79uo"
            | "cd+Rtw+D1x8"
            | "on6ZH7Abo10"
            | "8XTArSPyWHk"
            | "PE2zHMqLSHs"
            | "aII9h5nli9U"
            | "XHl38ZNknbs"
            | "v8iOE+j8a5o"
    ) {
        if nid == "g2tViFIohHE" {
            AUDIO_OUT
                .get_or_init(|| Mutex::new(audio::AudioOutState::default()))
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .initialize();
        }
        return dispatch_audio_out2_import(nid, arguments);
    }

    let state = AUDIO_OUT.get_or_init(|| Mutex::new(audio::AudioOutState::default()));
    let mut state = state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match nid {
        "JfEPXVxhFqA" => {
            state.initialize();
            0
        }
        "ekNvsT22rsY" => state.open(arguments).unwrap_or_else(audio_out_error),
        "s1--uE9mBFw" => state
            .close(arguments[0])
            .map_or_else(audio_out_error, |()| 0),
        "b+uAV89IlxE" => state
            .set_volume(arguments)
            .map_or_else(audio_out_error, |()| 0),
        "GrQ9s4IrNaQ" => state
            .get_port_state(arguments)
            .map_or_else(audio_out_error, |()| 0),
        "QOQtbeDqsT4" => {
            let output = state.output(arguments);
            drop(state);
            match output {
                Ok(delay) => {
                    if !delay.is_zero() {
                        std::thread::sleep(delay);
                    }
                    0
                }
                Err(error) => audio_out_error(error),
            }
        }
        _ => unreachable!("caller filters audio-out imports"),
    }
}

/// Validates an argument REmu treats as a guest output pointer.
///
/// A null check alone is not enough: when an argument position is inferred rather than
/// documented, the value can be a handle or selector, and writing through it corrupts whatever
/// the host has mapped at that address instead of failing the call.
fn guest_output_pointer(address: u64, size: usize) -> Option<u64> {
    (address != 0 && process_guest_range_is_readable(address, size)).then_some(address)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ProbeImportReturn {
    integer: u64,
    float: f64,
}

impl ProbeImportReturn {
    const fn integer(integer: u64) -> Self {
        Self {
            integer,
            float: 0.0,
        }
    }

    const fn float(float: f64) -> Self {
        Self { integer: 0, float }
    }
}

struct GuestThreadTls {
    _storage: Vec<u8>,
    thread_pointer: u64,
}

fn clone_guest_tls(template: &[u8]) -> GuestThreadTls {
    let size = usize::try_from(GUEST_TLS_PREFIX_SIZE + GUEST_TLS_TCB_SIZE)
        .expect("guest TLS allocation size fits usize");
    assert_eq!(
        template.len(),
        size,
        "guest TLS template has the linked size"
    );
    let mut storage = template.to_vec();
    let thread_pointer_offset =
        usize::try_from(GUEST_TLS_PREFIX_SIZE).expect("guest TLS prefix size fits usize");
    let original_thread_pointer = u64::from_le_bytes(
        storage[thread_pointer_offset..thread_pointer_offset + 8]
            .try_into()
            .expect("thread pointer word is present in TLS template"),
    );
    let thread_pointer = storage.as_mut_ptr() as usize as u64 + GUEST_TLS_PREFIX_SIZE;
    let dtv_offset = thread_pointer_offset + 0x100;
    let maximum_module_id = u64::from_le_bytes(
        storage[dtv_offset + 8..dtv_offset + 16]
            .try_into()
            .expect("DTV module count is present in TLS template"),
    );
    for module_id in 1..=maximum_module_id {
        let Some(entry_offset) =
            tls_dtv_entry_offset(module_id).and_then(|offset| usize::try_from(offset).ok())
        else {
            continue;
        };
        let entry_offset = dtv_offset + entry_offset;
        let original_block = u64::from_le_bytes(
            storage[entry_offset..entry_offset + 8]
                .try_into()
                .expect("DTV module entry is present in TLS template"),
        );
        if original_block == 0 {
            continue;
        }
        let displacement = i128::from(original_block) - i128::from(original_thread_pointer);
        let rebased = i128::from(thread_pointer) + displacement;
        if let Ok(rebased) = u64::try_from(rebased) {
            storage[entry_offset..entry_offset + 8].copy_from_slice(&rebased.to_le_bytes());
        }
    }
    write_guest_u64(thread_pointer, thread_pointer);
    write_guest_u64(thread_pointer + 0x08, thread_pointer + 0x100);
    write_guest_u64(thread_pointer + 0x10, thread_pointer);
    write_guest_u64(thread_pointer + 0x60, thread_pointer);
    write_guest_u64(thread_pointer + 0x100, 1);
    GuestThreadTls {
        _storage: storage,
        thread_pointer,
    }
}

/// Applies the caller's error convention to a thread-join result.
fn join_result(result: u64, posix: bool) -> u64 {
    if posix {
        posix_pthread_result(result)
    } else {
        result
    }
}

/// Converts an absolute `timespec` deadline into microseconds from now.
///
/// A deadline already in the past yields zero, which the wait treats as an immediate timeout
/// rather than an unbounded block.
fn relative_microseconds_until(deadline: u64) -> u64 {
    if deadline == 0 {
        return 0;
    }
    let seconds = i64::from_ne_bytes(read_guest_u64(deadline).to_ne_bytes());
    let nanoseconds = i64::from_ne_bytes(read_guest_u64(deadline + 8).to_ne_bytes());
    let deadline_microseconds = seconds
        .saturating_mul(1_000_000)
        .saturating_add(nanoseconds.div_euclid(1_000));
    let now_microseconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| {
            i64::try_from(elapsed.as_micros()).unwrap_or(i64::MAX)
        });
    u64::try_from(deadline_microseconds.saturating_sub(now_microseconds)).unwrap_or(0)
}

fn posix_pthread_result(result: u64) -> u64 {
    let result32 = low_u32(result);
    if result32 & 0xffff_0000 == 0x8002_0000 {
        u64::from(result32 & 0xffff)
    } else {
        result
    }
}

/// Whether a stalled wait is released as a spurious wakeup once it is reported.
///
/// This is a diagnostic: it cannot make an incorrect guest correct, but it separates a lost
/// wakeup from a predicate no producer ever satisfies.
fn stall_break_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("REMU_STALL_BREAK").is_some_and(|value| value == "1"))
}

/// Lists every event registration the guest is waiting on.
///
/// A queue with waiters but no registration, or a registration REmu never posts to, identifies
/// the notification the guest expects and is not receiving.
fn report_kernel_event_registrations() {
    let Some(state) = PROBE_DISPATCH.get() else {
        return;
    };
    let Ok(state) = state.try_read() else {
        eprintln!("[REMu][STALL] event registrations unavailable while dispatch is busy");
        return;
    };
    for (&(equeue, filter, ident), &user_data) in &state.kernel_event_registrations {
        eprintln!(
            "[REMu][STALL] registration equeue={equeue:#x} filter={filter} ident={ident:#x} user_data={user_data:#x}"
        );
    }
    for equeue in &state.kernel_equeues {
        eprintln!("[REMu][STALL] equeue={equeue:#x} exists");
    }
}

/// How long an unbounded guest wait may run before it reports itself once.
///
/// Opt-in through `REMU_TRACE_STALL_SECONDS`, because worker threads legitimately park for a long
/// time waiting for work and would otherwise dominate the log.
fn stall_report_interval() -> Option<Duration> {
    static INTERVAL: OnceLock<Option<Duration>> = OnceLock::new();
    *INTERVAL.get_or_init(|| {
        std::env::var("REMU_TRACE_STALL_SECONDS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|seconds| *seconds > 0)
            .map(Duration::from_secs)
    })
}

fn low_u32(value: u64) -> u32 {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

const fn trace_logarithmic_sequence(sequence: u64, initial: u64) -> bool {
    sequence <= initial || sequence.is_power_of_two()
}

fn low_i32(value: u64) -> i32 {
    let bytes = value.to_le_bytes();
    i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn low_f32(value: f64) -> f32 {
    let bytes = value.to_bits().to_le_bytes();
    f32::from_bits(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn guest_atomic_fetch_4(address: u64, value: u64, subtract: bool) -> u64 {
    if address == 0 || !address.is_multiple_of(std::mem::align_of::<AtomicU32>() as u64) {
        return u64::MAX;
    }
    let Ok(host_address) = usize::try_from(address) else {
        return u64::MAX;
    };
    let pointer = host_address as *mut u32;
    // SAFETY: Guest atomic imports provide a live, naturally aligned four-byte object in the
    // shared process address space. SeqCst is at least as strong as every guest memory-order mode.
    let atomic = unsafe { AtomicU32::from_ptr(pointer) };
    let value = low_u32(value);
    let previous = if subtract {
        atomic.fetch_sub(value, Ordering::SeqCst)
    } else {
        atomic.fetch_add(value, Ordering::SeqCst)
    };
    u64::from(previous)
}

fn dispatch_kernel_file_import(
    nid: &str,
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> Option<u64> {
    // These raw libKernel calls use Gen5 kernel errors, not POSIX -1/errno semantics.
    Some(match nid {
        "1G3lF1Gg1k8" => guest_kernel_open(arguments, state),
        "1-LFLmRFxxM" => guest_kernel_mkdir(arguments, state),
        "naInUjYt3so" => guest_kernel_rmdir(arguments, state),
        "eV9wAD2riIA" => guest_kernel_stat(arguments, state),
        "Cg4srZ6TKbU" => guest_kernel_read(arguments),
        "UK2Tl2DWUns" => guest_kernel_close(arguments),
        "oib76F-12fk" => guest_kernel_lseek(arguments),
        "kBwCPsYX-m4" => guest_kernel_fstat(arguments),
        "VW3TVZiM4-E" => guest_kernel_ftruncate(arguments),
        _ => return None,
    })
}

/// File-descriptor operations are independent of the process service graph. FMOD performs reads
/// and seeks on its streaming thread while the graphics thread can spend tens of milliseconds
/// translating one AGC submission. Keeping these calls behind `PROBE_DISPATCH` starved the audio
/// producer even though it touched a completely unrelated file.
fn dispatch_kernel_fd_import(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        // Raw libKernel error convention.
        "Cg4srZ6TKbU" => guest_kernel_read(arguments),
        "UK2Tl2DWUns" => guest_kernel_close(arguments),
        "oib76F-12fk" => guest_kernel_lseek(arguments),
        "kBwCPsYX-m4" => guest_kernel_fstat(arguments),
        "VW3TVZiM4-E" => guest_kernel_ftruncate(arguments),
        // POSIX wrappers translate kernel errors to -1 and guest errno.
        "DRuBt2pvICk" | "AqBioC2vF3I" => guest_posix_file_result(guest_kernel_read(arguments)),
        "NNtFaKJbPt0" | "bY-PO6JhzhQ" => guest_posix_file_result(guest_kernel_close(arguments)),
        "FxVZqBAA7ks" | "FN4gaPmuFV8" => guest_posix_file_result(guest_kernel_write(arguments)),
        "Oy6IpwgtYOk" => guest_posix_file_result(guest_kernel_lseek(arguments)),
        "mqQMh1zPPT8" => guest_posix_file_result(guest_kernel_fstat(arguments)),
        "ih4CD9-gghM" => guest_posix_file_result(guest_kernel_ftruncate(arguments)),
        "4wSze92BhLI" => guest_kernel_write(arguments),
        _ => return None,
    })
}

#[cfg(target_arch = "x86_64")]
type GuestModuleInitializer = unsafe extern "C" fn(u64, u64, u64) -> i32;

#[cfg(target_arch = "x86_64")]
fn invoke_guest_module_initializers(
    initializers: &[GuestAddress],
    argument_size: u64,
    argument: u64,
) -> i32 {
    let mut result = 0;
    for initializer in initializers {
        eprintln!(
            "[REMu][MODULE] initializing runtime at {initializer} args={argument_size:#x}/{argument:#x}"
        );
        // SAFETY: The module linker relocated this DT_INIT address in executable guest memory.
        let initializer: GuestModuleInitializer = unsafe { std::mem::transmute(initializer.get()) };
        // SAFETY: Gen5 module initializers use the SysV module-entry convention. The first two
        // values are sceKernelLoadStartModule's caller-owned argument block; a null third argument
        // asks the CRT wrapper to invoke the module's linked start function after its constructors.
        result = unsafe { initializer(argument_size, argument, 0) };
    }
    result
}

#[cfg(not(target_arch = "x86_64"))]
fn invoke_guest_module_initializers(
    _initializers: &[GuestAddress],
    _argument_size: u64,
    _argument: u64,
) -> i32 {
    0
}

fn kernel_load_start_module(arguments: [u64; 6]) -> u64 {
    if arguments[5] != 0 {
        write_guest_u32(arguments[5], 0);
    }
    let Some(requested_path) = read_guest_cstring(arguments[0]) else {
        let error = sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
        if arguments[5] != 0 {
            write_guest_u32(arguments[5], low_u32(error));
        }
        return error;
    };
    let requested_name = Path::new(&requested_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(&requested_path);
    let dispatch = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
    let mut state = lock_probe_dispatch_for_write(dispatch);
    let module = state.bundled_modules.iter_mut().find(|module| {
        module.name.eq_ignore_ascii_case(&requested_path)
            || Path::new(&module.name)
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.eq_ignore_ascii_case(requested_name))
    });
    let Some(module) = module else {
        let error = sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
        if arguments[5] != 0 {
            write_guest_u32(arguments[5], low_u32(error));
        }
        eprintln!("[REMu][MODULE] load-start could not find {requested_path}");
        return error;
    };
    let handle = module.handle;
    let module_name = module.name.clone();
    let initializers = if module.started {
        Vec::new()
    } else {
        module.started = true;
        module.initializers.clone()
    };
    drop(state);
    let start_result = invoke_guest_module_initializers(&initializers, arguments[1], arguments[2]);
    if arguments[5] != 0 {
        write_guest_u32(arguments[5], start_result.cast_unsigned());
    }
    eprintln!(
        "[REMu][MODULE] load-start {requested_path} -> handle {} ({}) result={start_result:#x}",
        handle, module_name,
    );
    u64::from(handle)
}

fn sce_symbol_nid(symbol: &str) -> String {
    const SALT: [u8; 16] = [
        0x51, 0x8d, 0x64, 0xa6, 0x35, 0xde, 0xd8, 0xc1, 0xe6, 0xb0, 0x39, 0xb1, 0xc3, 0xe5, 0x52,
        0x30,
    ];
    const CODES: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    let mut hasher = Sha1::new();
    hasher.update(symbol.as_bytes());
    hasher.update(SALT);
    let hash = hasher.finalize();
    let digest = u64::from_le_bytes(hash[..8].try_into().expect("SHA-1 prefix has eight bytes"));
    let mut nid = String::with_capacity(11);
    for index in 0..10 {
        let shift = 58 - index * 6;
        nid.push(char::from(CODES[((digest >> shift) & 0x3f) as usize]));
    }
    nid.push(char::from(CODES[((digest & 0x0f) * 4) as usize]));
    nid
}

fn kernel_dlsym(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let Some(symbol) = read_guest_cstring(arguments[1]) else {
        return u64::MAX;
    };
    let hle_target = match symbol.as_str() {
        "scriptingGetMem" => scripting_get_mem as *const () as usize as u64,
        "scriptingFreeMem" => libc::free as *const () as usize as u64,
        "scriptingRealloc" => libc::realloc as *const () as usize as u64,
        "scriptingCalloc" => libc::calloc as *const () as usize as u64,
        _ => 0,
    };
    let handle = low_u32(arguments[0]);
    let nid = sce_symbol_nid(&symbol);
    let guest_target = state
        .bundled_modules
        .iter()
        .find(|module| module.handle == handle)
        .and_then(|module| module.exports_by_nid.get(&nid))
        .copied()
        .unwrap_or_default();
    let target = if hle_target != 0 {
        hle_target
    } else {
        guest_target
    };
    if arguments[2] == 0 || target == 0 {
        if arguments[2] != 0 {
            write_guest_u64(arguments[2], 0);
        }
        eprintln!(
            "[REMu][MODULE] dlsym unresolved handle={} symbol={symbol:?} nid={nid}",
            arguments[0],
        );
        return u64::MAX;
    }
    write_guest_u64(arguments[2], target);
    eprintln!(
        "[REMu][MODULE] dlsym handle={} symbol={symbol:?} nid={nid} -> {target:#x}",
        arguments[0],
    );
    0
}

fn kernel_module_info_for_unwind(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let queried_address = arguments[0];
    let flags = low_i32(arguments[1]);
    let output = arguments[2];
    if output == 0 || !(0..3).contains(&flags) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if read_guest_u64(output) < 0x130 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(module) = state
        .unwind_modules
        .iter()
        .find(|module| (module.base..module.end).contains(&queried_address))
    else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    guest_memset(output, 0, 0x130);
    write_guest_u64(output, 0x130);
    let name = module.name.as_bytes();
    let name_length = name.len().min(0xff);
    write_guest_bytes(output + 8, &name[..name_length]);
    write_guest_u64(output + 0x108, module.eh_frame_header);
    write_guest_u64(output + 0x110, module.eh_frame);
    write_guest_u64(output + 0x118, module.eh_frame_size);
    write_guest_u64(output + 0x120, module.base);
    write_guest_u64(output + 0x128, module.end - module.base);
    0
}

/// `sceKernelGetModuleInfoFromAddr(addr, flags, SceKernelModuleInfoEx *info)`.
///
/// libc's TLS destructor path calls this with `flags == 2` immediately after
/// `_sceKernelRtldThreadAtexitIncrement` to recover the owning module's PHDRs.
fn kernel_module_info_from_addr(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    const MODULE_INFO_EX_SIZE: u64 = 0x1a8;
    const MAX_SEGMENTS: usize = 4;
    let queried_address = arguments[0];
    let flags = low_i32(arguments[1]);
    let output = arguments[2];
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    if !(0..3).contains(&flags) {
        guest_memset(output, 0, MODULE_INFO_EX_SIZE);
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if read_guest_u64(output) != MODULE_INFO_EX_SIZE {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some(module) = state
        .unwind_modules
        .iter()
        .find(|module| (module.base..module.end).contains(&queried_address))
    else {
        guest_memset(output, 0, MODULE_INFO_EX_SIZE);
        write_guest_u64(output, MODULE_INFO_EX_SIZE);
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let handle = state
        .bundled_modules
        .iter()
        .find(|bundled| {
            bundled.name == module.name
                || Path::new(&bundled.name)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name == module.name)
        })
        .map_or(0, |bundled| bundled.handle);

    guest_memset(output, 0, MODULE_INFO_EX_SIZE);
    write_guest_u64(output, MODULE_INFO_EX_SIZE);
    let name = module.name.as_bytes();
    let name_length = name.len().min(0xff);
    write_guest_bytes(output + 8, &name[..name_length]);
    write_guest_u32(output + 0x108, handle);
    write_guest_u64(output + 0x148, module.eh_frame_header);
    write_guest_u64(output + 0x150, module.eh_frame);
    write_guest_u32(output + 0x158, 0);
    write_guest_u32(
        output + 0x15c,
        u32::try_from(module.eh_frame_size.min(u64::from(u32::MAX))).unwrap_or(u32::MAX),
    );

    let mut segments = module.segments.clone();
    if segments.is_empty() {
        // Fallback when segment metadata was unavailable at link time: one aggregate R+X range.
        let size = u32::try_from((module.end - module.base).min(u64::from(u32::MAX)))
            .unwrap_or(u32::MAX);
        segments.push(GuestModuleSegment {
            address: module.base,
            size,
            protection: 5,
        });
    }
    let segment_count = segments.len().min(MAX_SEGMENTS);
    for (index, segment) in segments.into_iter().take(segment_count).enumerate() {
        let base = output + 0x160 + (index as u64) * 16;
        write_guest_u64(base, segment.address);
        write_guest_u32(base + 8, segment.size);
        write_guest_u32(base + 12, segment.protection);
    }
    write_guest_u32(output + 0x1a0, segment_count as u32);
    0
}

extern "C" fn scripting_get_mem(alignment: u64, size: u64) -> u64 {
    guest_aligned_alloc(alignment, size)
}

fn kernel_install_exception_handler(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let exception = low_i32(arguments[0]);
    if !matches!(exception, 1 | 4 | 8 | 10 | 11 | 30) || arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if state
        .installed_exception_handlers
        .insert(exception, arguments[1])
        .is_some()
    {
        return sce_kernel_error(SCE_KERNEL_ERROR_ALREADY_EXISTS);
    }
    eprintln!(
        "[REMu][KERNEL] exception handler installed type={exception} address={:#x}",
        arguments[1]
    );
    0
}

fn kernel_remove_exception_handler(exception: u64, state: &mut ProbeDispatchState) -> u64 {
    let exception = low_i32(exception);
    if !matches!(exception, 1 | 4 | 8 | 10 | 11 | 30) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state.installed_exception_handlers.remove(&exception);
    0
}

fn kernel_raise_exception(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let exception = low_i32(arguments[1]);
    if arguments[0] == 0 || !(0..128).contains(&exception) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if let Some(handler) = state.installed_exception_handlers.get(&exception) {
        let Some(target_thread) = state.pthread_threads.get(&arguments[0]) else {
            eprintln!(
                "[REMu][KERNEL] exception target is not a live pthread: {:#x}",
                arguments[0]
            );
            return sce_kernel_error(SCE_KERNEL_ERROR_BUSY);
        };
        let source = current_guest_thread_handle();
        guest_exception_delivery()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending
            .entry(arguments[0])
            .or_insert(PendingGuestException {
                handler: *handler,
                exception,
            });
        GUEST_EXCEPTION_PENDING.store(true, Ordering::Release);
        guest_sync().1.notify_all();
        notify_kernel_equeue_waiters();
        eprintln!(
            "[REMu][KERNEL] queued cooperative exception source={source:#x} target={:#x} target_name={:?} type={exception} handler={handler:#x}",
            arguments[0], target_thread.name
        );
    }
    0
}

fn process_clock_nanos() -> u64 {
    u64::try_from(
        PROCESS_CLOCK_EPOCH
            .get_or_init(Instant::now)
            .elapsed()
            .as_nanos(),
    )
    .unwrap_or(u64::MAX)
}

fn reset_kernel_process_time() {
    PROCESS_START_NANOS.store(process_clock_nanos().max(1), Ordering::Release);
}

fn kernel_process_time(counter: bool) -> u64 {
    let now = process_clock_nanos();
    let mut start = PROCESS_START_NANOS.load(Ordering::Acquire);
    if start == 0 {
        // Unit-level dispatch tests can call this export without configuring a complete image.
        // Let the first such caller establish the same process-relative epoch without a mutex.
        match PROCESS_START_NANOS.compare_exchange(
            0,
            now.max(1),
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => start = now,
            Err(installed) => start = installed,
        }
    }
    let elapsed_nanos = now.saturating_sub(start);
    if counter {
        elapsed_nanos
    } else {
        elapsed_nanos / 1_000
    }
}

const SCE_AJM_ERROR_INVALID_PARAMETER: u32 = 0x8093_0005;
const SCE_AJM_ERROR_INVALID_CONTEXT: u32 = 0x8093_0002;
const SCE_AJM_ERROR_INVALID_INSTANCE: u32 = 0x8093_0003;
const SCE_AJM_ERROR_CODEC_ALREADY_REGISTERED: u32 = 0x8093_0009;
const SCE_AJM_ERROR_CODEC_NOT_REGISTERED: u32 = 0x8093_000a;
const SCE_AJM_ERROR_JOB_CREATION: u32 = 0x8093_0012;
/// Sideband result written by control jobs (`result` + `internal_result`).
const AJM_SIDEBAND_RESULT_SIZE: usize = 8;
/// Bytes one control job (initialize / clear-context / gapless) occupies in a batch arena.
const AJM_JOB_CONTROL_SIZE: u64 = 48;
/// Smallest arena a caller may hand `sceAjmBatchInitialize`. One control job is 0x30 bytes;
/// rejecting that size strands audio workers that build single-job batches.
const AJM_BATCH_MIN_SIZE: u64 = AJM_JOB_CONTROL_SIZE;
const AJM_BATCH_MAX_SIZE: u64 = 0x10_000;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SceAjmBatchInfo {
    buffer: u64,
    offset: u64,
    size: u64,
    last_good_job: u64,
    last_good_job_return_address: u64,
}

const _: () = {
    assert!(std::mem::size_of::<SceAjmBatchInfo>() == 0x28);
    assert!(std::mem::offset_of!(SceAjmBatchInfo, buffer) == 0x00);
    assert!(std::mem::offset_of!(SceAjmBatchInfo, offset) == 0x08);
    assert!(std::mem::offset_of!(SceAjmBatchInfo, size) == 0x10);
    assert!(std::mem::offset_of!(SceAjmBatchInfo, last_good_job) == 0x18);
    assert!(std::mem::offset_of!(SceAjmBatchInfo, last_good_job_return_address) == 0x20);
};

/// Bytes the batch-error object a caller may supply occupies.
///
/// It carries an error code, the address of the offending job, a command offset, and that job's
/// return address. Clearing fewer bytes than this leaves the trailing fields holding whatever the
/// caller's stack held, and an engine that reads them back asserts on the garbage.
const AJM_BATCH_ERROR_SIZE: u64 = 24;

/// Identifiers handed out by [`ajm_batch_start`].
static AJM_NEXT_BATCH: AtomicU32 = AtomicU32::new(0);

/// `sceAjmBatchStart(context, info, priority, error, batchOut)`.
///
/// Accepts the batch the guest built and hands back an identifier it can wait on.
///
/// No decode runs. This host has no audio-job coprocessor and no general Ajm decoder, so a batch
/// is accepted and retired without producing samples; the job result fields the guest reads back
/// keep the zero-length output the arena was initialized with, which reports "nothing was
/// decoded" rather than claiming samples that do not exist. A title consuming this gets silence
/// on Ajm-decoded audio, and that is a real limitation, not a hidden one — but its control flow
/// proceeds, where refusing the call strands the audio thread and with it everything downstream.
fn ajm_batch_start(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let context = low_u32(arguments[0]);
    let info = arguments[1];
    let error = arguments[3];
    let output = arguments[4];
    if !state.ajm_contexts.contains_key(&context) {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
    }
    if info == 0
        || output == 0
        || guest_output_pointer(output, std::mem::size_of::<u32>()).is_none()
        || guest_output_pointer(info, std::mem::size_of::<SceAjmBatchInfo>()).is_none()
    {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    if error != 0 {
        if guest_output_pointer(error, AJM_BATCH_ERROR_SIZE as usize).is_none() {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
        }
        // No job failed, because no job ran.
        guest_memset(error, 0, AJM_BATCH_ERROR_SIZE);
    }
    let batch = AJM_NEXT_BATCH
        .fetch_add(1, Ordering::Relaxed)
        .wrapping_add(1)
        .max(1);
    write_guest_u32(output, batch);
    0
}

/// `sceAjmBatchWait(context, batch, timeout, error)`.
///
/// A batch this host accepted is already retired, so the wait completes immediately.
fn ajm_batch_wait(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let context = low_u32(arguments[0]);
    let error = arguments[3];
    if !state.ajm_contexts.contains_key(&context) {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
    }
    if error != 0 {
        if guest_output_pointer(error, AJM_BATCH_ERROR_SIZE as usize).is_none() {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
        }
        guest_memset(error, 0, AJM_BATCH_ERROR_SIZE);
    }
    0
}

/// `sceAjmBatchCancel(context, batch)`.
///
/// Cancelling a retired batch is a no-op that the guest is entitled to call.
fn ajm_batch_cancel(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if !state.ajm_contexts.contains_key(&low_u32(arguments[0])) {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
    }
    0
}

/// Bytes one `sceAjmBatchJobGetStatistics` job occupies in a batch arena.
const AJM_JOB_GET_STATISTICS_SIZE: u64 = 88;
/// Bytes the statistics result object occupies.
const AJM_STATISTICS_RESULT_SIZE: usize = 48;

/// Appends `job_size` bytes to a batch arena cursor and returns the reserved job address.
fn ajm_append_batch_job(info: u64, job_size: u64) -> Result<u64, u32> {
    if info == 0 || guest_output_pointer(info, std::mem::size_of::<SceAjmBatchInfo>()).is_none() {
        return Err(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    // SAFETY: the complete cursor object was validated immediately above.
    let mut batch_info = unsafe { std::ptr::read_unaligned(info as *const SceAjmBatchInfo) };
    let Some(end) = batch_info.offset.checked_add(job_size) else {
        return Err(SCE_AJM_ERROR_JOB_CREATION);
    };
    if batch_info.buffer == 0 || end > batch_info.size {
        return Err(SCE_AJM_ERROR_JOB_CREATION);
    }
    let job = batch_info.buffer.wrapping_add(batch_info.offset);
    guest_memset(job, 0, job_size);
    batch_info.offset = end;
    batch_info.last_good_job = job;
    // The arena cursor's return-address field belongs to the job just appended, not to whatever
    // occupied the slot before it.
    batch_info.last_good_job_return_address = 0;
    // SAFETY: the same validated cursor object.
    unsafe {
        std::ptr::write_unaligned(info as *mut SceAjmBatchInfo, batch_info);
    }
    Ok(job)
}

/// `sceAjmBatchJobGetStatistics(info, result)`.
///
/// Appends one statistics job to the caller's batch arena and clears the result object. The
/// counters this host can honestly report are all zero: no decode work runs on a coprocessor here,
/// so there is no elapsed time or job count to report, and reporting invented figures would be
/// worse than reporting none. The append itself is real, because the guest reads the arena cursor
/// back and builds the rest of its batch from it.
fn ajm_batch_job_get_statistics(arguments: [u64; 6]) -> u64 {
    let result = arguments[1];
    if let Err(error) = ajm_append_batch_job(arguments[0], AJM_JOB_GET_STATISTICS_SIZE) {
        return sce_kernel_error(error);
    }
    if result != 0 {
        if guest_output_pointer(result, AJM_STATISTICS_RESULT_SIZE).is_none() {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
        }
        guest_memset(result, 0, AJM_STATISTICS_RESULT_SIZE as u64);
    }
    0
}

/// `sceAjmBatchJobInitialize(info, instance, codecParameters, parametersSize, result)`.
///
/// Appends one control job and clears the sideband result. This host has no Ajm codec coprocessor,
/// so initialization records success for a live instance without consuming the parameter blob —
/// the guest still advances its batch cursor and later Start/Wait calls retire the batch as silence.
fn ajm_batch_job_initialize(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (info, instance, result) = (arguments[0], low_u32(arguments[1]), arguments[4]);
    if let Err(error) = ajm_append_batch_job(info, AJM_JOB_CONTROL_SIZE) {
        return sce_kernel_error(error);
    }
    let status = if state
        .ajm_contexts
        .values()
        .any(|context| context.instances.contains(&instance))
    {
        0_i32
    } else {
        // Sideband decode-result "invalid parameter" (not the SCE_AJM_ERROR_* facility).
        0x4_i32
    };
    if result != 0 {
        if guest_output_pointer(result, AJM_SIDEBAND_RESULT_SIZE).is_none() {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
        }
        guest_memset(result, 0, AJM_SIDEBAND_RESULT_SIZE as u64);
        write_guest_u32(result, status as u32);
    }
    0
}

fn ajm_batch_initialize(arguments: [u64; 6]) -> u64 {
    let batch = arguments[0];
    let size = arguments[1];
    let info = arguments[2];
    let Ok(batch_size) = usize::try_from(size) else {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    };
    if batch == 0
        || info == 0
        || !(AJM_BATCH_MIN_SIZE..=AJM_BATCH_MAX_SIZE).contains(&size)
        || !process_guest_range_has_read_access(batch, batch_size)
        || guest_output_pointer(info, std::mem::size_of::<SceAjmBatchInfo>()).is_none()
    {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    // The arena belongs to the caller and may already contain allocator/debug state. The PS5 ABI
    // initializes only this five-qword cursor; job builders subsequently append into `buffer`
    // using `offset` and update the two last-good fields.
    let batch_info = SceAjmBatchInfo {
        buffer: batch,
        offset: 0,
        size,
        last_good_job: 0,
        last_good_job_return_address: 0,
    };
    // SAFETY: the complete 0x28-byte output object was validated above.
    unsafe {
        std::ptr::write_unaligned(info as *mut SceAjmBatchInfo, batch_info);
    }
    0
}

fn guest_zeroed_constructor(address: u64, size: u64) -> u64 {
    if address != 0 {
        guest_memset(address, 0, size);
    }
    address
}

fn ajm_initialize(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let reserved = arguments[0];
    let output = arguments[1];
    // Gen5 titles tag this argument with the library revision in its high dword and leave the low
    // dword reserved. Rejecting the whole value as "must be zero" fails initialization for every
    // revision-tagged caller, which makes an audio engine tear its threads down during startup.
    if low_u32(reserved) != 0 || output == 0 {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }

    let context = state.next_ajm_context.max(1);
    state.next_ajm_context = context.saturating_add(1);
    state.ajm_contexts.insert(
        context,
        AjmContextState {
            next_instance: 1,
            ..AjmContextState::default()
        },
    );
    write_guest_u32(output, context);
    eprintln!("[REMu][AJM] initialized context={context}");
    0
}

/// `sceAcmContextCreate(contextOut)`.
///
/// An ACM context is an opaque identifier: the guest passes it back to the batch entry points and
/// never inspects it, so handing out a distinct live id is the whole contract of this call. The
/// signal-processing batch entry points it is used with stay unimplemented, and deliberately so —
/// this makes the audio engine keep its context instead of tearing its whole stack down at
/// creation, and moves the loud gap to the calls that would actually have to do the work.
fn acm_context_create(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let output = arguments[0];
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let context = state.next_acm_context.max(1);
    state.next_acm_context = context.saturating_add(1);
    state.acm_contexts.insert(context);
    write_guest_u32(output, context);
    0
}

/// `sceAcmContextDestroy(context)`.
fn acm_context_destroy(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let context = low_u32(arguments[0]);
    if !state.acm_contexts.remove(&context) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    0
}

fn ajm_module_register(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let context = low_u32(arguments[0]);
    let codec = low_u32(arguments[1]);
    if arguments[2] != 0 {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    let Some(context_state) = state.ajm_contexts.get_mut(&context) else {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
    };
    if !context_state.registered_codecs.insert(codec) {
        return sce_kernel_error(SCE_AJM_ERROR_CODEC_ALREADY_REGISTERED);
    }
    eprintln!("[REMu][AJM] registered codec={codec} context={context}");
    0
}

fn ajm_module_unregister(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let context = low_u32(arguments[0]);
    let codec = low_u32(arguments[1]);
    let Some(context_state) = state.ajm_contexts.get_mut(&context) else {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
    };
    context_state.registered_codecs.remove(&codec);
    eprintln!("[REMu][AJM] unregistered codec={codec} context={context}");
    0
}

fn ajm_instance_create(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let context = low_u32(arguments[0]);
    let codec = low_u32(arguments[1]);
    let output = arguments[3];
    if output == 0 {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    let Some(context_state) = state.ajm_contexts.get_mut(&context) else {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
    };
    if !context_state.registered_codecs.contains(&codec) {
        return sce_kernel_error(SCE_AJM_ERROR_CODEC_NOT_REGISTERED);
    }
    let index = context_state.next_instance.max(1) & 0x3fff;
    context_state.next_instance = index.saturating_add(1);
    let instance = index | (codec << 14);
    context_state.instances.insert(instance);
    write_guest_u32(output, instance);
    eprintln!(
        "[REMu][AJM] created instance={instance:#x} codec={codec} context={context} flags={:#x}",
        arguments[2]
    );
    0
}

fn ajm_instance_destroy(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let context = low_u32(arguments[0]);
    let instance = low_u32(arguments[1]);
    let Some(context_state) = state.ajm_contexts.get_mut(&context) else {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
    };
    if !context_state.instances.remove(&instance) {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_INSTANCE);
    }
    eprintln!("[REMu][AJM] destroyed instance={instance:#x} context={context}");
    0
}

fn ajm_finalize(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let context = low_u32(arguments[0]);
    if context == 0 {
        state.ajm_contexts.clear();
    } else if state.ajm_contexts.remove(&context).is_none() {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
    }
    eprintln!("[REMu][AJM] finalized context={context}");
    0
}

fn np_web_api2_initialize(arguments: [u64; 6]) -> u64 {
    const SCE_NP_WEB_API2_ERROR_LIB_CONTEXT_MAX: u32 = 0x8055_3418;
    const MAXIMUM_LIBRARY_CONTEXTS: usize = 0x7fff;

    let context =
        NEXT_NP_WEB_API2_CONTEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
            (next <= MAXIMUM_LIBRARY_CONTEXTS).then_some(next + 1)
        });
    let Ok(context) = context else {
        return sce_kernel_error(SCE_NP_WEB_API2_ERROR_LIB_CONTEXT_MAX);
    };
    eprintln!(
        "[REMu][NP_WEB_API2] initialized context={context} http_context={} pool_size={:#x}",
        low_i32(arguments[0]),
        arguments[1]
    );
    u64::try_from(context).expect("NP Web API2 context ID fits u64")
}

/// Handles imports whose result depends only on caller-owned memory or thread-local/host state.
///
/// These calls are deliberately kept outside `PROBE_DISPATCH`: Dreaming Sarah invokes the libc
/// memory/string and AGC command-building families from many asset workers, and serializing every
/// one behind the process-wide write lock can stall the render and audio threads even though the
/// calls do not touch `ProbeDispatchState`.
fn dispatch_stateless_import(
    nid: &str,
    arguments: [u64; 6],
    stack_arguments: [u64; 6],
) -> Option<u64> {
    if let Some(result) = agc_command_abi::dispatch_agc_command_abi(nid, arguments, stack_arguments)
    {
        return Some(result);
    }
    // Fibers run guest code that calls further imports, so they are served here rather than on the
    // locked path: holding the process dispatch lock across a fiber entry would deadlock it.
    #[cfg(target_arch = "x86_64")]
    if let Some(result) = fiber::dispatch(nid, arguments, stack_arguments) {
        return Some(result);
    }
    if let Some(result) = platform_services::dispatch(nid, arguments) {
        return Some(result);
    }
    Some(match nid {
        "bnZxYgAFeA0" => sanitizer_new_replace_external(),
        "vNe1w4diLCs" => guest_tls_get_addr(arguments[0]),
        "PI7jIZj4pcE" => random::sce_random_get_random_number(arguments),
        "aI+OeCz8xrQ" | "EotR8a3ASf4" => current_guest_thread_handle(),
        "pO96TwzOm5E" => 16 * 1024 * 1024 * 1024,
        "8zTFvBIAIN8" => guest_memset(arguments[0], arguments[1], arguments[2]),
        "Q3VBxCXhUHs" => guest_memcpy(arguments[0], arguments[1], arguments[2]),
        "+P6FRGH4LfA" => guest_memmove(arguments[0], arguments[1], arguments[2]),
        "DfivPArhucg" => guest_memcmp(arguments[0], arguments[1], arguments[2]),
        "8u8lPzUEq+U" => guest_memchr(arguments[0], arguments[1], arguments[2]),
        "fL3O02ypZFE" => guest_wmemcpy(arguments[0], arguments[1], arguments[2]),
        "Noj9PsJrsa8" => guest_wmemmove(arguments[0], arguments[1], arguments[2]),
        "fnUEjBCNRVU" => guest_wmemchr(arguments[0], arguments[1], arguments[2]),
        "QJ5xVfKkni0" => guest_wmemcmp(arguments[0], arguments[1], arguments[2]),
        "2HnmKiLmV6s" => guest_atomic_fetch_4(arguments[0], arguments[1], true),
        "iPBqs+YUUFw" => guest_atomic_fetch_4(arguments[0], arguments[1], false),
        "gQX+4GDQjpM" => guest_malloc(arguments[0]),
        "tIhsqj0qsFE" => guest_free(arguments[0]),
        "2X5agFjKxMc" => guest_calloc(arguments[0], arguments[1]),
        "Y7aJ1uydPMo" => guest_realloc(arguments[0], arguments[1]),
        "Ujf3KzMvRmI" | "2Btkg8k24Zg" => guest_aligned_alloc(arguments[0], arguments[1]),
        "cVSk9y8URbc" => guest_posix_memalign(arguments[0], arguments[1], arguments[2]),
        "j4ViWNHEgww" => guest_strlen(arguments[0]),
        "Ovb2dSJOAuE" => guest_strcmp(arguments[0], arguments[1]),
        "aesyjrHVWy4" => guest_strncmp(arguments[0], arguments[1], arguments[2]),
        "kiZSXIWd9vg" => guest_strcpy(arguments[0], arguments[1]),
        "6sJWiWSRuqk" => guest_strncpy(arguments[0], arguments[1], arguments[2]),
        "Ls4tzzhimqQ" => guest_strcat(arguments[0], arguments[1]),
        "AV6ipCNa4Rw" => guest_strcasecmp(arguments[0], arguments[1]),
        "viiwFMaNamA" => guest_strstr(arguments[0], arguments[1]),
        "rQFVBXp-Cxg" => guest_fseek(arguments[0], arguments[1], arguments[2]),
        "Qazy8LmXTvw" => guest_ftell(arguments[0]),
        "uodLYyUip20" => guest_fclose(arguments[0]),
        "lbB+UlZqVG0" => guest_fread(arguments[0], arguments[1], arguments[2], arguments[3]),
        "kALvdgEv5ME" | "9nf8joUTSaQ" | "Q1BL70XVV0o" => 0,
        "hqi8yMOCmG0" | "p6LrHjIQMdk" | "QW2jL1J5rwY" | "P8F2oavZXtY" => arguments[0],
        "wLlFkwG9UcQ" => guest_time(arguments[0]),
        "tcVi5SivF7Q" => guest_sprintf(arguments),
        "QrZZdJ8XsX0" => guest_fputs(arguments[0]),
        "3GPpjQdAMTw" => guest_guard_acquire(arguments[0]),
        "9rAeANT2tyE" => guest_guard_release(arguments[0]),
        "2emaaluWzUw" => guest_guard_abort(arguments[0]),
        "cfAXurvfl5o" => guest_cxa_allocate_exception(arguments[0]),
        "nOIEswYD4Ig" => guest_cxa_free_exception(arguments[0]),
        "vkuuLfhnSZI" => guest_cxa_throw(arguments),
        "9BcDykPmo1I" => guest_errno_address(),
        "HQOwnfMGipQ" => guest_net_errno_address(),
        "Q4qBuN-c0ZM" => guest_net_unavailable(45),
        "dgJBaeJnGpo" => NEXT_NET_POOL_HANDLE.fetch_add(1, Ordering::Relaxed) as u64,
        "hdpVEUDFW3s" => NEXT_SSL_CONTEXT.fetch_add(1, Ordering::Relaxed) as u64,
        "3JCe3lCbQ8A" | "A9cVMUtEp4Y" | "0gYjPTR-6cY" => {
            NEXT_HTTP_CONTEXT.fetch_add(1, Ordering::Relaxed) as u64
        }
        "+o9816YQhqQ" => np_web_api2_initialize(arguments),
        "MmpF1XsQiHw" => ajm_batch_initialize(arguments),
        "cK6bYHf-Q5E" | "-hJRce8wn1U" => guest_zeroed_constructor(arguments[0], 8),
        "WSOuge5IsCg" => guest_zeroed_constructor(arguments[0], 0x28),
        "TRO721eVt4g" => agc_dcb_reset_queue(arguments),
        "ZvwO9euwYzc" => agc_dcb_set_registers_indirect(arguments, AGC_R_CX_REGS_INDIRECT),
        "-HOOCn0JY48" => agc_dcb_set_registers_indirect(arguments, AGC_R_SH_REGS_INDIRECT),
        "hvUfkUIQcOE" => agc_dcb_set_registers_indirect(arguments, AGC_R_UC_REGS_INDIRECT),
        "vcmNN+AAXnY" | "Qrj4c+61z4A" | "6lNcCp+fxi4" => agc_set_indirect_patch_address(arguments),
        "d-6uF9sZDIU" | "z2duB-hHQSM" | "vRoArM9zaIk" => {
            agc_add_indirect_patch_registers(arguments)
        }
        "D9sr1xGUriE" => agc_create_primitive_state(arguments),
        "HV4j+E0MBHE" => agc_create_interpolant_mapping(arguments),
        "n2fD4A+pb+g" => agc_cb_set_sh_register_range_direct(arguments),
        "V++UgBtQhn0" => agc_get_data_packet_payload_address(arguments),
        "aJf+j5yntiU" => agc_dcb_event_write(arguments),
        "LtTouSCZjHM" => agc_cb_nop(arguments),
        "dolOmWH+huQ" => agc_get_fused_shader_size(arguments),
        "WmAc2MEj6Io" => agc_dcb_dma_data(arguments, stack_arguments),
        "wr23dPKyWc0" => agc_cb_release_mem(arguments, stack_arguments),
        "VmW0Tdpy420" => agc_dcb_wait_reg_mem(arguments, stack_arguments),
        "IxYiarKlXxM" => agc_dma_data_patch_destination(arguments),
        "3KDcnM3lrcU" => agc_wait_reg_mem_patch_address(arguments),
        "0fWWK5uG9rQ" => agc_release_mem_patch_address(arguments),
        "57labkp+rSQ" => agc_dcb_acquire_mem(arguments, stack_arguments),
        "l4fM9K-Lyks" => agc_dcb_set_index_buffer(arguments),
        "GIIW2J37e70" => agc_dcb_set_index_size(arguments),
        "B+aG9DUnTKA" => agc_dcb_draw_index_offset(arguments),
        "tSBxhAPyytQ" => agc_dcb_set_num_instances(arguments),
        "Yw0jKSqop+E" => agc_dcb_draw_index_auto(arguments),
        "YUeqkyT7mEQ" => agc_dcb_set_flip(arguments),
        "rPo6tV8D9bM" => system_service_get_status(arguments),
        "fZo48un7LK4" => system_service_param_get_int(arguments),
        "SsC-m-S9JTA" => system_service_param_get_string(arguments),
        "3RQ5aQfnstU" => system_service_get_notice_screen_skip_flag(arguments[0]),
        "Xjoosiw+XPI" => guest_uuid_create(arguments[0]),
        "yDBwVAolDgg" => guest_kernel_is_stack(arguments),
        "Qs0wWulgl7U" => 0,
        "RaqxZIf6DvE" => 0,
        "rbknaUjpqWo" => write_output_u64(arguments[1], 0),
        "XDncXQIJUSk" => np_get_online_id(arguments),
        "JT+t00a3TxA" => np_get_account_country(arguments[1]),
        "eQH7nWPcAgc" => write_output_i32(arguments[1], 1),
        "e-ZuhGEoeC4" => write_output_i32(arguments[1], 0),
        "1xxcMiGu2fo" => user_service_get_user_name(arguments),
        "woNpu+45RLk" => user_service_get_age_level(arguments),
        "-sD02mFDBh4" => user_service_get_game_presets(arguments),
        "R9lA82OraNs" => app_content_initialize(arguments),
        "99b82IKXpH4" => app_content_app_param_get_int(arguments),
        "eaFXjfJv3xs" => dialog::ime_keyboard_open(),
        "-4GCfYdNF1s" => 0,
        "18B2NS1y9UU" => rtc_get_current_tick(arguments[0]),
        "ZPD1YOKI+Kw" => rtc_get_current_clock_local_time(arguments[0]),
        "8w-H19ip48I" => rtc_get_tick(arguments),
        "ueega6v3GUw" => rtc_set_tick(arguments),
        "1n37q1Bvc5Y" => system_service_get_display_safe_area_info(arguments[0]),
        "9vA2aW+CHuA" => net_inet_ntop(arguments),
        // The console reports no connection when the title runs offline. Name resolution and
        // network status must fail explicitly so online features disable themselves instead of
        // waiting on a link that never appears.
        "C4UgDHHPvdw" | "obuxdTiwkF8" => guest_net_unavailable(50),
        // Network-state callbacks are accepted and never fire: an offline link never changes
        // state, so there is no notification to deliver and nothing pending to check.
        "UJ+Z7Q+4ck0" => net_ctl_register_callback(arguments),
        "iQw3iQPhvUQ" => 0,
        "nBDD66kiFW8" => 0,
        "zE-wXIZjLoM" => kernel_debug_raise_exception_on_release_mode(arguments),
        "j2AIqSqJP0w" | "2G6i6hMIUUY" => guest_kernel_getdents(arguments, false),
        "taRWhTJFTgE" | "sfKygSjIbI8" => guest_kernel_getdents(arguments, true),
        // Async I/O is initialized with tuning parameters only; REmu services the requests
        // synchronously through the same file paths, so there is no queue to size here.
        "nu4a0-arQis" | "vYU8P9Td2Zo" => 0,
        "EI-5-jlq2dE" => current_guest_thread_handle(),
        "dKadqZFgKKQ" => ime_keyboard_get_resource_ids(arguments),
        "VkqLPArfFdc" => ime_keyboard_get_info(arguments[1]),
        "x8qnXqh-tiM" => 0,
        "rnEhHqG-4xo" | "ZKJtxdgvzwg" | "-3Y5GO+-i78" | "hD-H81EN9Vg" | "O6IW1-Dwm-w" => {
            user_service_get_setting(arguments, 0)
        }
        "qWYHOFwqCxY" => user_service_get_setting(arguments, 1),
        "CdWp0oHWGr0" => write_output_i32(arguments[0], 0x1000_0000),
        "fPhymKNvK-A" => user_service_get_login_user_id_list(arguments),
        "h9z6+0hEydk" => {
            std::thread::yield_now();
            0
        }
        "1jfXLRVzisc" => guest_usleep(arguments[0]),
        // The POSIX spelling shares the sleep and reports errno rather than an SCE code.
        "QcteRwbsnV0" => posix_pthread_result(guest_usleep(arguments[0])),
        "lLMT9vJAck0" | "QBi7HCK03hw" => guest_clock_gettime(arguments[0], arguments[1]),
        "n88vx3C5nW8" => guest_posix_gettimeofday(arguments[0]),
        "ejekcaNQNq0" => guest_kernel_gettimeofday(arguments[0]),
        "0NTHN1NKONI" => guest_kernel_convert_localtime_to_utc(arguments),
        "-o5uEDpN+oY" => guest_kernel_convert_utc_to_localtime(arguments),
        "0-KXaS70xy4" | "eoht7mQOCmo" => pthread_get_specific(low_u32(arguments[0])),
        "WrOLvHU0yQM" => posix_pthread_result(pthread_set_specific(arguments)),
        "+BzXYkqYeLE" => pthread_set_specific(arguments),
        _ => return None,
    })
}

#[allow(clippy::match_same_arms, clippy::too_many_lines)]
fn dispatch_probe_import(
    nid: &str,
    arguments: [u64; 6],
    stack_arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    // Unit tests call the stateful dispatcher directly to exercise individual handlers. The live
    // import path already exhausted this table before acquiring `PROBE_DISPATCH`; repeating its
    // full guaranteed-miss scan here lengthens every stateful critical section.
    #[cfg(test)]
    if let Some(result) = dispatch_stateless_import(nid, arguments, stack_arguments) {
        return result;
    }
    if let Some(result) = dispatch_kernel_file_import(nid, arguments, state) {
        return result;
    }
    match nid {
        // Process-runtime initialization and destructor registration are observable lifecycle
        // operations. Both C and Itanium callbacks are retained in one registration-order list.
        "bzQExy189ZI" => initialize_guest_libc(arguments[0], state),
        "8G2LB+A3rzg" => register_guest_exit_callback(arguments[0], None, 0, state),
        "tsvEmnenz48" => {
            register_guest_exit_callback(arguments[0], Some(arguments[1]), arguments[2], state)
        }
        // The game-bundled C runtime registers its per-thread cleanup hooks during DT_INIT and
        // reads the main executable's PT_SCE_PROCPARAM block to configure its allocator.
        "rNhWz+lvOMU" => {
            state.thread_dtors_callback = arguments[0];
            0
        }
        "pB-yGZ2nQ9o" => {
            state.thread_atexit_count_callback = arguments[0];
            0
        }
        "WhCc1w3EhSI" => {
            state.thread_atexit_report_callback = arguments[0];
            0
        }
        // Pin/unpin the module that owns a pending TLS destructor (`__dso_handle` address).
        "Tz4RNUCBbGI" => rtld_atexit::rtld_thread_atexit_increment(arguments[0], state),
        "8OnWXlgQlvo" => rtld_atexit::rtld_thread_atexit_decrement(arguments[0], state),
        "959qrazPIrg" => state.process_param_address,
        "p5EcQeEeJAE" => {
            state.application_heap_api_address = arguments[0];
            0
        }
        "dl+4eHSzUu4" => ajm_initialize(arguments, state),
        // ACM context handles. The batch/DSP entry points stay unserved.
        "ZIXln2K3XMk" => acm_context_create(arguments, state),
        "jBgBjAj02R8" => acm_context_destroy(arguments, state),
        "Q3dyFuwGn64" => ajm_module_register(arguments, state),
        "3cAg7xN995U" => ajm_batch_job_get_statistics(arguments),
        "ezM2OhNxzck" => ajm_batch_job_initialize(arguments, state),
        "5tOfnaClcqM" => ajm_batch_start(arguments, state),
        "-qLsfDAywIY" => ajm_batch_wait(arguments, state),
        "NVDXiUesSbA" => ajm_batch_cancel(arguments, state),
        "Wi7DtlLV+KI" => ajm_module_unregister(arguments, state),
        "AxoDrINp4J8" => ajm_instance_create(arguments, state),
        "RbLbuKv8zho" => ajm_instance_destroy(arguments, state),
        "MHur6qCsUus" => ajm_finalize(arguments, state),
        // Bundled PRXs are mapped, relocated, and initialized before the main executable. Runtime
        // module loads must still resolve those objects to their stable process-local handles;
        // Unity treats a negative return here as a plug-in load failure and skips assembly setup.
        "LwG8g3niqwA" => kernel_dlsym(arguments, state),
        "WkwEd3N7w0Y" => kernel_install_exception_handler(arguments, state),
        "Qhv5ARAoOEc" => kernel_remove_exception_handler(arguments[0], state),
        "il03nluKfMk" => kernel_raise_exception(arguments, state),
        "6xVpy0Fdq+I" => 0,
        "bnZxYgAFeA0" => sanitizer_new_replace_external(),
        "vNe1w4diLCs" => guest_tls_get_addr(arguments[0]),
        "NWtTN10cJzE" => libc_heap_get_trace_info(arguments[0], state),
        "6c3rCVE-fTU" | "wuCroIGjt2g" => {
            guest_posix_file_result(guest_kernel_open(arguments, state))
        }
        "DRuBt2pvICk" | "AqBioC2vF3I" => guest_posix_file_result(guest_kernel_read(arguments)),
        "NNtFaKJbPt0" | "bY-PO6JhzhQ" => guest_posix_file_result(guest_kernel_close(arguments)),
        "FxVZqBAA7ks" | "FN4gaPmuFV8" => guest_posix_file_result(guest_kernel_write(arguments)),
        "Oy6IpwgtYOk" => guest_posix_file_result(guest_kernel_lseek(arguments)),
        "E6ao34wPw+U" => guest_posix_file_result(guest_kernel_stat(arguments, state)),
        "mqQMh1zPPT8" => guest_posix_file_result(guest_kernel_fstat(arguments)),
        "ih4CD9-gghM" => guest_posix_file_result(guest_kernel_ftruncate(arguments)),
        "4wSze92BhLI" => guest_kernel_write(arguments),
        "aI+OeCz8xrQ" | "EotR8a3ASf4" => current_guest_thread_handle(),
        "7Xl257M4VNI" => u64::from(arguments[0] == arguments[1]),
        // sceKernelGetDirectMemorySize
        "pO96TwzOm5E" => 16 * 1024 * 1024 * 1024,
        // sceKernelAllocateDirectMemory
        "rTXw65xmLIA" => allocate_direct_memory(arguments, state),
        // sceKernelAvailableDirectMemorySize
        "C0f7TJcbfac" => available_direct_memory_size(arguments, state),
        // sceKernelAvailableFlexibleMemorySize
        "aNz11fnnzi4" => available_flexible_memory_size(arguments, state),
        "B+vc2AO2Zrc" => allocate_main_direct_memory(arguments, state),
        "BHouLQzh0X0" => direct_memory_query(arguments, state),
        // sceKernelMapDirectMemory
        "L-Q3LEjIbgA" => map_direct_memory(arguments, state),
        "NcaWUxfMNIQ" => map_direct_memory(arguments, state),
        "mL8NDH86iQI" => map_named_flexible_memory(arguments, state),
        "-hn1tcVHq5Q" => guest_mspace_create(arguments, state),
        "LYo3GhIlB38" => guest_mspace_calloc(arguments, state),
        "OJjm-QOIHlI" => guest_mspace_malloc(arguments, state),
        "Vla-Z+eXlxo" => guest_mspace_free(arguments, state),
        "W6SiVSiCDtI" => guest_mspace_destroy(arguments, state),
        "fEoW6BJsPt4" => guest_mspace_usable_size(arguments, state),
        "gigoVHZvVPE" => guest_mspace_realloc(arguments, state),
        "iF1iQHzxBJU" | "ljkqMcC4-mk" => guest_mspace_memalign(arguments, state),
        "pzUa7KEoydw" => guest_mspace_is_heap_empty(arguments, state),
        "qWESlyXMI3E" => guest_mspace_posix_memalign(arguments, state),
        "I2QC8PYhJWY" => json_init_parameter_set_allocator(arguments, state),
        "Eu95jmqn5Rw" => json_init_parameter_set_file_buffer_size(arguments, state),
        "IXW-z8pggfg" | "Cxwy7wHq4J0" => json_initializer_initialize(arguments, state),
        "PR5k1penBLM" => json_initializer_terminate(arguments, state),
        "eBTreZ84JFY" => av_player_set_log_callback(arguments, state),
        "aS66RI0gGgo" => av_player_init(arguments, state),
        "o9eWRkSL+M4" => av_player_init_ex(arguments, state),
        "HD1YKVU26-M" => av_player_post_init(arguments, state),
        "KMcEa+rHsIo" => av_player_add_source(arguments, state),
        "x8uvuFOPZhU" => av_player_add_source_ex(arguments, state),
        "NkJwDzKmIlw" => av_player_close(arguments[0], state),
        "wwM99gjFf1Y" => av_player_current_time(arguments[0], state),
        "UbQoYawOsfY" => av_player_is_active(arguments[0], state),
        "9y5v+fGN4Wk" => av_player_pause(arguments[0], state),
        "XC9wM+xULz8" => av_player_jump_to_time(arguments, state),
        "w5moABNwnRY" => av_player_resume(arguments[0], state),
        "ET4Gr-Uu07s" => av_player_start(arguments[0], state),
        "ZC17w3vB5Lo" => av_player_stop(arguments[0], state),
        "OVths0xGfho" => av_player_set_looping(arguments, state),
        "buMCiJftcfw" => av_player_change_stream(arguments, state),
        "av8Z++94rs0" => av_player_set_trick_speed(arguments, state),
        "Wnp1OVcrZgk" => av_player_get_audio_data(arguments, state),
        "o3+RWnHViSg" => av_player_get_video_data(arguments, false, state),
        "JdksQu8pNdQ" => av_player_get_video_data(arguments, true, state),
        "d8FcbzfAdQw" => av_player_get_stream_info(arguments, false, state),
        "ctTAcF5DiKQ" => av_player_get_stream_info(arguments, true, state),
        "ODJK2sn9w4A" => av_player_validate_handle(arguments[0], state),
        "k-q+xOxdc3E" => av_player_set_av_sync_mode(arguments, state),
        "hdTyRzCXQeQ" => av_player_stream_count(arguments[0], state),
        // libkernel's APR resolver imports are POSIX-shaped: the SDK wrappers consume `-1` and
        // `errno`, then translate that pair into an SCE kernel error. Returning an SCE error
        // directly leaves the wrapper reading stale errno and can turn ENOENT into false success.
        "gEpBkcwxUjw" => guest_posix_file_result(ampr_resolve_file_paths(arguments, state)),
        // One resolver reached through eight names: the variants differ only in whether the
        // caller wants sizes, supplies a prefix, or collects a per-entry result.
        "WT-5NKy42fw" => guest_posix_file_result(ampr_resolve_paths_core(
            arguments[0],
            arguments[1],
            None,
            AprResolveOutputs {
                ids: arguments[2],
                error_index: arguments[3],
                ..AprResolveOutputs::default()
            },
            state,
        )),
        "eYAh2vlCY-U" => guest_posix_file_result(ampr_resolve_paths_core(
            arguments[0],
            arguments[1],
            None,
            AprResolveOutputs {
                ids: arguments[2],
                results: arguments[3],
                ..AprResolveOutputs::default()
            },
            state,
        )),
        "QzB4O+bJQyA" => guest_posix_file_result(ampr_resolve_paths_core(
            arguments[0],
            arguments[1],
            None,
            AprResolveOutputs {
                ids: arguments[2],
                sizes: arguments[3],
                results: arguments[4],
                ..AprResolveOutputs::default()
            },
            state,
        )),
        "i3HWvW35jao" => guest_posix_file_result(ampr_resolve_paths_core(
            arguments[1],
            arguments[2],
            read_guest_cstring(arguments[0]),
            AprResolveOutputs {
                ids: arguments[3],
                error_index: arguments[4],
                ..AprResolveOutputs::default()
            },
            state,
        )),
        "w5fcCG+t31g" => guest_posix_file_result(ampr_resolve_paths_core(
            arguments[1],
            arguments[2],
            read_guest_cstring(arguments[0]),
            AprResolveOutputs {
                ids: arguments[3],
                sizes: arguments[4],
                error_index: arguments[5],
                ..AprResolveOutputs::default()
            },
            state,
        )),
        "VB-BtuIW8Xc" => guest_posix_file_result(ampr_resolve_paths_core(
            arguments[1],
            arguments[2],
            read_guest_cstring(arguments[0]),
            AprResolveOutputs {
                ids: arguments[3],
                results: arguments[4],
                ..AprResolveOutputs::default()
            },
            state,
        )),
        "C+Khtbbx2g8" => guest_posix_file_result(ampr_resolve_paths_core(
            arguments[1],
            arguments[2],
            read_guest_cstring(arguments[0]),
            AprResolveOutputs {
                ids: arguments[3],
                sizes: arguments[4],
                results: arguments[5],
                ..AprResolveOutputs::default()
            },
            state,
        )),
        "WvEu7yl3Ivg" => ampr_get_file_size(arguments, state),
        "ApkYaHb8Sek" => ampr_get_file_stat(arguments, state),
        "eE4Szl8sil8" | "qvMUCyyaCSI" => ampr_submit_command_buffer(arguments, state),
        "ASoW5WE-UPo" => ampr_submit_command_buffer(arguments, state),
        "rqwFKI4PAiM" => ampr_wait_command_buffer(arguments, state),
        // sceKernelMapDirectMemory2 inserts memory type and passes alignment as argument seven.
        "BQQniolj9tQ" => map_direct_memory_2(arguments, stack_arguments[0], state),
        // Kernel VM ownership calls must describe and retire the mappings created above. Returning
        // success without filling VirtualQuery makes libc feed uninitialized lengths into munmap.
        "rVjRvHJ0X6c" => kernel_virtual_query(arguments, state),
        "cQke9UuBQOk" => kernel_munmap(arguments, state),
        "8zLSfEfW5AU" => coredump_register_handler(arguments, state),
        "2SKEx6bSq-4" => kernel_batch_map(arguments, state),
        "kBJzF8x4SyE" => kernel_batch_map_2(arguments, state),
        "9bfdLIyuwCY" => kernel_mtypeprotect(arguments, state),
        "hwVSPCmp5tM" => kernel_checked_release_direct_memory(arguments, state),
        "MBuItvba6z8" => kernel_release_direct_memory(arguments, state),
        "7oxv3PPCumo" => kernel_reserve_virtual_range(arguments, state),
        // The label is diagnostic metadata only. Validate the guest string so a bad caller still
        // gets an ABI error, but no host VM operation is required.
        "DGMG3JshrZU" => kernel_set_virtual_range_name(arguments),
        // libc memset
        "8zTFvBIAIN8" => guest_memset(arguments[0], arguments[1], arguments[2]),
        // libc memcpy
        "Q3VBxCXhUHs" => guest_memcpy(arguments[0], arguments[1], arguments[2]),
        // libc memmove
        "+P6FRGH4LfA" => guest_memmove(arguments[0], arguments[1], arguments[2]),
        // libc memcmp
        "DfivPArhucg" => guest_memcmp(arguments[0], arguments[1], arguments[2]),
        // libc memchr
        "8u8lPzUEq+U" => guest_memchr(arguments[0], arguments[1], arguments[2]),
        // Gen5 uses two-byte wchar_t values. macOS uses four-byte wchar_t values, so these
        // functions must not be bound directly to the host libc wide-memory family.
        "fL3O02ypZFE" => guest_wmemcpy(arguments[0], arguments[1], arguments[2]),
        "Noj9PsJrsa8" => guest_wmemmove(arguments[0], arguments[1], arguments[2]),
        "fnUEjBCNRVU" => guest_wmemchr(arguments[0], arguments[1], arguments[2]),
        "QJ5xVfKkni0" => guest_wmemcmp(arguments[0], arguments[1], arguments[2]),
        // The guest ABI returns the previous four-byte value; the third argument is memory order.
        "2HnmKiLmV6s" => guest_atomic_fetch_4(arguments[0], arguments[1], true),
        "iPBqs+YUUFw" => guest_atomic_fetch_4(arguments[0], arguments[1], false),
        // The native x86-64 process and guest share the libc allocation ABI.
        "gQX+4GDQjpM" => guest_malloc(arguments[0]),
        "tIhsqj0qsFE" => guest_free(arguments[0]),
        "2X5agFjKxMc" => guest_calloc(arguments[0], arguments[1]),
        "Y7aJ1uydPMo" => guest_realloc(arguments[0], arguments[1]),
        "Ujf3KzMvRmI" | "2Btkg8k24Zg" => guest_aligned_alloc(arguments[0], arguments[1]),
        "cVSk9y8URbc" => guest_posix_memalign(arguments[0], arguments[1], arguments[2]),
        // REmu does not instrument the guest address space with AddressSanitizer. Returning a
        // successful false result is ABI-observable: treating the missing import error as a
        // boolean true makes Unreal select ASan-only allocation and fault handling paths.
        "jh+8XiK4LeE" => 0,
        // libc strlen
        "j4ViWNHEgww" => guest_strlen(arguments[0]),
        "Ovb2dSJOAuE" => guest_strcmp(arguments[0], arguments[1]),
        "aesyjrHVWy4" => guest_strncmp(arguments[0], arguments[1], arguments[2]),
        "kiZSXIWd9vg" => guest_strcpy(arguments[0], arguments[1]),
        "6sJWiWSRuqk" => guest_strncpy(arguments[0], arguments[1], arguments[2]),
        "Ls4tzzhimqQ" => guest_strcat(arguments[0], arguments[1]),
        "AV6ipCNa4Rw" => guest_strcasecmp(arguments[0], arguments[1]),
        "viiwFMaNamA" => guest_strstr(arguments[0], arguments[1]),
        // libc stdio uses host FILE objects after resolving console paths inside the game dump.
        "xeYO4u7uyJ0" => guest_fopen(arguments[0], arguments[1], state),
        "rQFVBXp-Cxg" => guest_fseek(arguments[0], arguments[1], arguments[2]),
        "Qazy8LmXTvw" => guest_ftell(arguments[0]),
        "uodLYyUip20" => guest_fclose(arguments[0]),
        "lbB+UlZqVG0" => guest_fread(arguments[0], arguments[1], arguments[2], arguments[3]),
        // Dinkumware's locale helpers are used by the save-error formatting path. The character
        // table accessor must return a live table; its lock and lifetime helpers are process-local
        // bookkeeping in the single-process HLE runtime.
        "sUP1hBaouOw" => guest_ctype_table(state),
        "1uJgoVq3bQU" => guest_tolower_table(state),
        "rcQCUr0EaRU" => guest_toupper_table(state),
        "9rMML086SEE" | "hEQ2Yi4PJXA" => guest_global_locale(state),
        "kALvdgEv5ME" | "9nf8joUTSaQ" | "Q1BL70XVV0o" => 0,
        "hqi8yMOCmG0" | "p6LrHjIQMdk" | "QW2jL1J5rwY" | "P8F2oavZXtY" => arguments[0],
        // libc time returns Unix seconds and optionally stores the same value through its pointer.
        "wLlFkwG9UcQ" => guest_time(arguments[0]),
        // Preserve the SysV variadic register layout for the integer/pointer formats used by the
        // game's asset path builder. Floating-point varargs require a wider import frame later.
        "tcVi5SivF7Q" => guest_sprintf(arguments),
        // libc fputs
        "QrZZdJ8XsX0" => guest_fputs(arguments[0]),
        // Itanium C++ one-time initialization guards.
        "3GPpjQdAMTw" => guest_guard_acquire(arguments[0]),
        "9rAeANT2tyE" => guest_guard_release(arguments[0]),
        "2emaaluWzUw" => guest_guard_abort(arguments[0]),
        // libc++abi exception storage. Throwing currently ends the probe with a decoded diagnostic
        // until guest unwind metadata is registered with the host unwinder.
        "cfAXurvfl5o" => guest_cxa_allocate_exception(arguments[0]),
        "nOIEswYD4Ig" => guest_cxa_free_exception(arguments[0]),
        "vkuuLfhnSZI" => guest_cxa_throw(arguments),
        // libc __error returns a stable, writable errno cell owned by the current guest thread.
        "9BcDykPmo1I" => guest_errno_address(),
        // libKernel pthread attributes are opaque allocated objects. Their scheduling, stack, and
        // detach values feed thread creation and remain observable through the matching getters.
        "F8bUHwAG284" => create_pthread_mutex_attr(arguments[0], state),
        "smWEktiyyG0" => destroy_pthread_mutex_attr(arguments[0], state),
        "iMp8QpE+XO4" => pthread_mutex_attr_set_type(arguments, state),
        "1FGvU0i9saQ" => pthread_mutex_attr_set_protocol(arguments, state),
        // The POSIX forms return errno values instead of the signed SCE kernel namespace.
        "dQHWEsJtoE4" => posix_pthread_result(create_pthread_mutex_attr(arguments[0], state)),
        "HF7lK46xzjY" => posix_pthread_result(destroy_pthread_mutex_attr(arguments[0], state)),
        "mDmgMOGVUqg" => posix_pthread_result(pthread_mutex_attr_set_type(arguments, state)),
        "mqULNdimTn0" => posix_pthread_result(pthread_key_create(arguments, state)),
        "geDaqgH9lTg" => pthread_key_create(arguments, state),
        "PrdHuuDekhY" => pthread_key_delete(low_u32(arguments[0]), state),
        "6BpEZuDT7YI" => posix_pthread_result(pthread_key_delete(low_u32(arguments[0]), state)),
        "nsYoNRywwNg" | "wtkt-teR1so" => {
            create_pthread_attr(arguments[0], PthreadAttrState::default(), state)
        }
        "62KCwEMmzcM" | "zHchY8ft5pk" => destroy_pthread_attr(arguments[0], state),
        "-Wreprtu0Qs" => pthread_attr_set_detach_state(arguments, state),
        "UTXzJbWhhTE" | "2Q0z6rnBrTE" => pthread_attr_set_stack_size(arguments, state),
        "DzES9hQF4f4" => pthread_attr_set_schedule_param(arguments, state),
        "FXPWHNk8Of0" => pthread_attr_get_schedule_param(arguments, state),
        "eXbUSpEaTsA" => pthread_attr_set_inherit_schedule(arguments, state),
        "4+h9EzwKF4I" => pthread_attr_set_schedule_policy(arguments, state),
        "P41kTWUS3EI" => pthread_get_schedule_param(arguments, state),
        "oIRFTjoILbg" => pthread_set_schedule_param(arguments, state),
        "bt3CTBKmGyI" => pthread_set_affinity(arguments, state),
        "3qxgM4ezETA" => pthread_attr_set_affinity(arguments, state),
        "W0Hpm2X0uPE" => pthread_set_priority(arguments, state),
        "GBUY7ywdULE" => pthread_rename(arguments, state),
        "x1X76arYMxU" => pthread_attr_get(arguments, state),
        "-quPa4SEJUw" => pthread_attr_get_stack(arguments, state),
        "Ru36fiTtJzA" => pthread_attr_get_stack_address(arguments, state),
        "-fA+7ZlGDQs" => pthread_attr_get_stack_size(arguments, state),
        "8+s5BzZjxSg" => pthread_attr_get_affinity(arguments, state),
        "5txKfcMUAok" => pthread_mutex_attr_set_protocol(arguments, state),
        "m5-2bsNfv7s" => pthread_condition_attr_init(arguments[0], state),
        "waPcxYiR3WA" => pthread_condition_attr_destroy(arguments[0], state),
        // Initial single-device handles for input, video, and audio.
        "xk0AcarP3V4" => pad_open(arguments),
        "u1GRHp+oWoY" => pad_get_handle(arguments),
        "Up36PTk687E" => video_out_open(arguments, state),
        "utPrVdxio-8" => video_out_get_output_status(arguments, state),
        "CBiu4mCE1DA" => video_out_set_flip_rate(arguments, state),
        "U46NwOiJpys" => video_out_submit_flip(arguments, state),
        "zgXifHT9ErY" => video_out_is_flip_pending(arguments, state),
        "8aI7R7WaOlc" => ampr_command_buffer_constructor(arguments, state),
        "a8uLzYY--tM" => ampr_apr_command_buffer_constructor(arguments, state),
        "N-FSPA4S3nI" => ampr_command_buffer_set_buffer(arguments, state),
        "baQO9ez2gL4" => ampr_command_buffer_reset(arguments, state),
        "ULvXMDz56po" => ampr_command_buffer_clear_buffer(arguments, state),
        "tZDDEo2tE5k" => ampr_command_buffer_get_size(arguments, state),
        "GnxKOHEawhk" => ampr_command_buffer_get_current_offset(arguments, state),
        "H896Pt-yB4I" => ampr_command_buffer_write_kernel_event_queue(arguments, state),
        "sSAUCCU1dv4" | "vWU-odnS+fU" => ampr_measure_command_size(),
        "GuchCTefuZw" | "Qs1xtplKo0U" => ampr_command_buffer_destructor(arguments, state),
        "mQ16-QdKv7k" => {
            unreachable!("AMPR file reads are routed before acquiring PROBE_DISPATCH")
        }
        // AGC shader construction and VideoOut mutate caller-owned descriptors. Games consume the
        // resulting pointers and command data immediately, so successful no-ops are insufficient.
        "f3dg2CSgRKY" => agc_create_shader(arguments, state),
        "TRO721eVt4g" => agc_dcb_reset_queue(arguments),
        "ZvwO9euwYzc" => agc_dcb_set_registers_indirect(arguments, AGC_R_CX_REGS_INDIRECT),
        "-HOOCn0JY48" => agc_dcb_set_registers_indirect(arguments, AGC_R_SH_REGS_INDIRECT),
        "hvUfkUIQcOE" => agc_dcb_set_registers_indirect(arguments, AGC_R_UC_REGS_INDIRECT),
        "vcmNN+AAXnY" | "Qrj4c+61z4A" | "6lNcCp+fxi4" => agc_set_indirect_patch_address(arguments),
        "d-6uF9sZDIU" | "z2duB-hHQSM" | "vRoArM9zaIk" => {
            agc_add_indirect_patch_registers(arguments)
        }
        "D9sr1xGUriE" => agc_create_primitive_state(arguments),
        "HV4j+E0MBHE" => agc_create_interpolant_mapping(arguments),
        "fd5Bp5tGTgo" => agc_fuse_shader_halves(arguments, state),
        "n2fD4A+pb+g" => agc_cb_set_sh_register_range_direct(arguments),
        "V++UgBtQhn0" => agc_get_data_packet_payload_address(arguments),
        "aJf+j5yntiU" => agc_dcb_event_write(arguments),
        "LtTouSCZjHM" => agc_cb_nop(arguments),
        "WmAc2MEj6Io" => agc_dcb_dma_data(arguments, stack_arguments),
        "wr23dPKyWc0" => agc_cb_release_mem(arguments, stack_arguments),
        "VmW0Tdpy420" => agc_dcb_wait_reg_mem(arguments, stack_arguments),
        "IxYiarKlXxM" => agc_dma_data_patch_destination(arguments),
        "3KDcnM3lrcU" => agc_wait_reg_mem_patch_address(arguments),
        "0fWWK5uG9rQ" => agc_release_mem_patch_address(arguments),
        "57labkp+rSQ" => agc_dcb_acquire_mem(arguments, stack_arguments),
        "GIIW2J37e70" => agc_dcb_set_index_size(arguments),
        "B+aG9DUnTKA" => agc_dcb_draw_index_offset(arguments),
        "tSBxhAPyytQ" => agc_dcb_set_num_instances(arguments),
        "Yw0jKSqop+E" => agc_dcb_draw_index_auto(arguments),
        "YUeqkyT7mEQ" => agc_dcb_set_flip(arguments),
        "w2rJhmD+dsE" => agc_driver_add_eq_event(arguments, state),
        "XlNp7jzGiPo" | "MM4IZSEYytQ" => 0,
        "UglJIZjGssM" => agc_driver_submit_dcb(arguments, state),
        "6UzEidRZwkg" => agc_driver_submit_multi_dcbs(arguments, state),
        "gSRnr79F8tQ" => agc_driver_submit_acb(arguments, state),
        "HF3YllT3mXU" => agc_driver_submit_multi_acbs(arguments, state),
        "PjS5uASwcV8" => video_out_set_buffer_attribute2(arguments, state),
        "rKBUtgRrtbk" => video_out_register_buffers2(arguments, state),
        "HXzjK9yI30k" => video_out_add_flip_event(arguments, state),
        "Xru92wHJRmg" => video_out_add_vblank_event(arguments, state),
        "w0hLuNarQxY" => (low_i32(arguments[0]) == 1)
            .then_some(0)
            .unwrap_or_else(|| sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE)),
        // libScePad output structures must always be initialized; successful no-ops would expose
        // stale guest stack contents as controller capabilities and button state.
        "clVvL4ZDntw" => pad_set_motion_sensor_state(arguments),
        "gjP9-KQzoUk" => pad_get_controller_information(arguments),
        "6ncge5+l5Qs" => pad_close(arguments),
        "AcslpN1jHR8" => pad_get_extended_information(arguments),
        "YndgXqQVV7c" => pad_read_state(arguments),
        "q1cHNfGycLI" => pad_read(arguments),
        "W2G-yoyMF5U" => pad_set_vibration_mode(arguments),
        "rIZnR6eSpvk" | "DscD1i9HX1w" => pad_handle_operation(arguments),
        "r44mAxdSG+U" | "vDLMoJLde8I" => pad_boolean_operation(arguments),
        "yFVnOdGxvZY" | "2JgFB2n9oUM" | "RR4novUEENY" => pad_pointer_operation(arguments),
        "znaWI0gpuo8" => pad_get_trigger_effect_state(arguments),
        "IHPqcbc0zCA" => pad_device_class_parse_data(arguments),
        "hv1luiJrqQM" => pad_initialize(),
        // libSceSystemService status used by the foreground/background state machine.
        "rPo6tV8D9bM" => system_service_get_status(arguments),
        "fZo48un7LK4" => system_service_param_get_int(arguments),
        "SsC-m-S9JTA" => system_service_param_get_string(arguments),
        "RpQJJVKTiFM" | "4fU5yvOkVG4" => kernel_module_info_for_unwind(arguments, state),
        "f7KBOafysXo" => kernel_module_info_from_addr(arguments, state),
        "crb5j7mkk1c" | "OMDRKKAZ8I4" => 0,
        "Vo5V8KAwCmk" => system_service_hide_splash(state),
        // REmu presents in standard dynamic range, so tone mapping reports the SDR reference
        // luminance rather than an HDR display's peak.
        "mPpPxv5CZt4" => {
            let Some(output) = guest_output_pointer(arguments[0], 12) else {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            };
            guest_memset(output, 0, 12);
            write_guest_u32(output, 100);
            write_guest_u32(output + 4, 100);
            write_guest_u32(output + 8, 100);
            0
        }
        // libSceSaveData uses durable host-side transactions while preserving the guest ABI.
        "TywrFKCoLGY" => save_data_initialize(state),
        "dyIhnXq-0SM" | "X4MYzukPc3g" => save_data_dir_name_search(arguments, state),
        "ZP4e7rlzOUk" => save_data_mount3(arguments, state),
        "gjRZNnw0JPE" => save_data_create_transaction_resource(arguments, state),
        "lJUQuaKqoKY" => save_data_delete_transaction_resource(arguments, state),
        "uW4vfTwMQVo" => save_data_umount2(arguments, state),
        "sDCBrmc61XU" => save_data_prepare(arguments, state),
        "85zul--eGXs" => save_data_set_param(arguments, state),
        "c88Yy54Mx0w" => save_data_save_icon(arguments, state),
        "ie7qhZ4X0Cc" => save_data_commit(arguments, state),
        // User-service IDs use the retail slot encoding expected by the game's controller layer.
        "CdWp0oHWGr0" => write_output_i32(arguments[0], 0x1000_0000),
        "fPhymKNvK-A" => user_service_get_login_user_id_list(arguments),
        "j3YMu1MVNNo" => user_service_initialize(state),
        "yH17Q6NWtVg" => user_service_get_event(arguments, state),
        // Runtime services retain initialization and loaded-module state instead of accepting
        // unrelated calls through the unknown-import fallback.
        "g8cM39EUZ6o" => sysmodule_load(arguments[0], state),
        "eR2bZFAAU0Q" => sysmodule_unload(arguments[0], state),
        "fMP5NHUOaMk" => sysmodule_is_loaded(arguments[0], state),
        // PlayGo describes an installed package. REmu runs a fully local image, so every chunk is
        // resident; the calls still validate handles and pointers exactly.
        "ts6GlZOKRrE" => play_go_initialize(state),
        "MPe0EeBGM-E" => play_go_terminate(state),
        "M1Gma1ocrGE" => play_go_open(arguments, state),
        "Uco1I0dlDi8" => play_go_close(arguments[0], state),
        "uWIYLFkkwqk" => play_go_get_locus(arguments, state),
        "-RJWNMK3fC8" => play_go_get_progress(arguments, state),
        "v6EZ-YWRdMs" => play_go_get_eta(arguments, state),
        "rvBSfTimejE" => play_go_get_install_speed(arguments, state),
        "73fF1MFU8hA" => play_go_get_chunk_id(arguments, state),
        "3OMbYZBaa50" => play_go_get_language_mask(arguments, state),
        "4AAcTU9R3XM" | "LosLlHOpNqQ" | "gUPGiOQ1tmQ" | "Nn7zKwnA5q0" | "-Q1-u1a7p0g" => {
            play_go_accept_request(arguments[0], state)
        }
        "lDqxaY1UbEo" => message_dialog_initialize(state),
        "ePw-kqZmelo" => message_dialog_terminate(state),
        "b06Hh0DPEaE" => message_dialog_open(arguments[0], state),
        "HTrcDKlFKuM" => message_dialog_close(state),
        "CWVW78Qc3fI" => message_dialog_get_status(state),
        "6fIC3XKt2k0" => message_dialog_update_status(state),
        "Lr8ovHH9l6A" => message_dialog_get_result(arguments[0], state),
        "I88KChlynSs" => error_dialog_initialize(state),
        "9XAxK2PMwk8" => error_dialog_terminate(state),
        "M2ZF-ClLhgY" => error_dialog_open(arguments[0], state),
        "ekXHb1kDBl0" => error_dialog_close(state),
        "t2FvHRXzgqk" => error_dialog_get_status(state),
        "WWiGuh9XfgQ" => error_dialog_update_status(state),
        "PfLB0BjAuQw" => error_dialog_get_result(arguments[0], state),
        "buYbeLOGWmA" => app_content_temporary_data_mount(arguments, state),
        "YJtKLttI9fM" => game_update_initialize(state),
        "jO8DM8oyego" => np_entitlement_access_initialize(arguments, state),
        "ysmw6J-P8Ak" => np_session_signaling_initialize(arguments, state),
        "sk54bi6FtYM" => np_web_api2_create_user_context(arguments, state),
        "WV1GwM32NgY" => np_web_api2_push_event_create_handle(arguments, state),
        "3Zl8BePTh9Y" => np_check_callback(state),
        "Ec63y59l9tw" => np_set_np_title_id(arguments, state),
        "YBiIdcDPrxs" => share_feature_permit(state),
        "7QZtURYnXG4" => share_set_content_parameter(arguments[0], state),
        "hZIg1EWGsHM" => ult_initialize(arguments[0], state),
        "uoUpLGNkygk" => common_dialog_initialize(state),
        "s9e3+YpRnzw" => save_data_dialog_initialize(state),
        "YuH2FA7azqQ" => save_data_dialog_terminate(state),
        "4tPhsP6FpDI" => save_data_dialog_open(arguments[0], state),
        "fH46Lag88XY" => save_data_dialog_close(state),
        "ERKzksauAJA" => save_data_dialog_get_status(state),
        "KK3Bdg1RWK0" => save_data_dialog_update_status(state),
        "yEiJ-qqr6Cg" => save_data_dialog_get_result(arguments[0], state),
        "en7gNVnh878" => save_data_dialog_is_ready_to_display(state),
        "D0OdFMjp46I" => kernel_create_equeue(arguments[0], state),
        "jpFjmgAC5AE" => kernel_delete_equeue(arguments[0], state),
        "kmSe30JTs+E" => video_out_add_output_mode_event(arguments, state),
        "MTxxrOCeSig" => video_out_set_window_mode_margins(arguments, state),
        "WDszmSbWuDk" => kernel_add_user_event_edge(arguments, state),
        "bBfz7kMF2Ho" => kernel_add_ampr_event(arguments, state),
        "kwGyyjohI50" => kernel_get_event_data(arguments[0]),
        "F6e0kwo4cnk" => kernel_trigger_user_event(arguments, state),
        "mJ7aghmgvfc" => kernel_get_event_id(arguments[0]),
        "Xs9hdiD7sAA" => pthread_set_schedule_parameters(arguments),
        // Direct unit-dispatch callers use the nonblocking poll. Live imports are routed through
        // `kernel_wait_equeue_blocking` before acquiring the process-wide dispatch lock.
        "fzyMKs9kim0" => kernel_wait_equeue(arguments, state),
        // Stateful offline NP services. Context, handle, event/property, intent and callback
        // ownership follow the guest ABI even though REmu deliberately does not upload telemetry
        // or connect a dumped title to PlayStation Network.
        "sjaobBgqeB4" => np_universal_data_initialize(arguments[0], state),
        "47UAEuQl+iI" => np_universal_data_terminate(state),
        "5zBnau1uIEo" => np_universal_data_create_context(arguments, state),
        "wB7IWzGp2v0" => np_universal_data_destroy_context(arguments[0], state),
        "hT0IAEvN+M0" => np_universal_data_create_handle(arguments, state),
        "jZCqWFgMehE" => np_universal_data_abort_handle(arguments[0], state),
        "AUIHb7jUX3I" => np_universal_data_destroy_handle(arguments[0], state),
        "tpFJ8LIKvPw" => np_universal_data_register_context(arguments, state),
        "p+GcLqwpL9M" => np_universal_data_create_event(arguments, state),
        "s6W4Zl4Slgk" => np_universal_data_create_property_object(arguments[0], state),
        "Hm7qubT3b70" => np_universal_data_create_property_array(arguments[0], state),
        "W-0xwY0ZMjw" => np_universal_data_destroy_property_array(arguments[0], state),
        "wG+84pnNIuo" => np_universal_data_destroy_event(arguments[0], state),
        "YE4dbtbz6OE" => np_universal_data_object_set_i32(arguments, state),
        "MfDb+4Nln64" => np_universal_data_object_set_string(arguments, state),
        "Wxbg5x3pTXA" => np_universal_data_object_set_array(arguments, state),
        "4llLk7YJRTE" => np_universal_data_array_set_string(arguments, state),
        "CzkKf7ahIyU" => np_universal_data_post_event(arguments, state),
        "m87BHxt-H60" => np_game_intent_initialize(arguments[0], state),
        "jEIXUAr9XE8" => np_game_intent_receive(arguments[0], state),
        "rPl0INNc-M8" => np_game_intent_get_property_string(arguments, state),
        "Bagshr7OQ6Q" => np_trophy_create_context(arguments, state),
        "Gz1rmUZpROM" => np_trophy_create_handle(arguments, state),
        "sysY2FHYff4" => np_trophy_destroy_context(arguments[0], state),
        "d8P11CI40KE" => np_trophy_destroy_handle(arguments[0], state),
        "fYapWA9xVmA" => np_trophy_abort_handle(arguments[0], state),
        "bIDov3wBu5Q" => np_trophy_register_context(arguments, state),
        "4IzqhhUQ3nk" => np_trophy_get_game_info(arguments, state),
        "y3zHpdZO6ME" => np_trophy_get_trophy_info_array(arguments, stack_arguments, state),
        "sUXGfNMalIo" => np_trophy_register_unlock_callback(arguments, state),
        "wVqxM58sIKs" => np_trophy_unregister_unlock_callback(state),
        // Exact Gen5 register-default descriptors consumed by indirect AGC state packets.
        "2JtWUUiYBXs" => agc_register_defaults(arguments[0], false, state),
        "wRbq6ZjNop4" => agc_register_defaults(arguments[0], true, state),
        "23LRUSvYu1M" => agc_initialize(arguments, state),
        "h9z6+0hEydk" => {
            std::thread::yield_now();
            0
        }
        "vSMAm3cxYTY" => kernel_mprotect(arguments, state),
        // sceKernelUsleep
        "1jfXLRVzisc" => guest_usleep(arguments[0]),
        // POSIX clock_gettime. Darwin's timespec layout matches the x86-64 guest ABI.
        "lLMT9vJAck0" => guest_clock_gettime(arguments[0], arguments[1]),
        "QBi7HCK03hw" => guest_clock_gettime(arguments[0], arguments[1]),
        // Keep discovery moving, but identify the exact calls that reached the fallback. The
        // general first-hit log also includes known imports whose implementation lives above this
        // dispatcher, so it cannot distinguish a missing ABI from an incomplete name table.
        _ => {
            if let Some(result) = dispatch_extended_import(nid, arguments, state) {
                return result;
            }
            // A repeated unimplemented call is common; probing first keeps the `String` allocation
            // to the first report instead of every one, under the process-wide write lock.
            if !state.unimplemented_nids.contains(nid) {
                state.unimplemented_nids.insert(nid.to_owned());
                eprintln!(
                    "[REMu][IMPORT] unimplemented nid={nid} name={} args={:#x},{:#x},{:#x},{:#x},{:#x},{:#x}",
                    nid_display_name(nid).unwrap_or("unknown"),
                    arguments[0],
                    arguments[1],
                    arguments[2],
                    arguments[3],
                    arguments[4],
                    arguments[5]
                );
            }
            sce_kernel_error(SCE_KERNEL_ERROR_NOT_IMPLEMENTED)
        }
    }
}

/// Offers a NID to each per-area module in turn, returning the first result produced.
///
/// The main dispatcher consults this only after its own table misses, so a module here can add
/// an ABI without touching the dispatcher, and a NID no module owns still reaches the
/// unimplemented report unchanged.
fn dispatch_extended_import(
    nid: &str,
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> Option<u64> {
    if let Some(result) = agc_dcb::dispatch(nid, arguments, state) {
        return Some(result);
    }
    if let Some(result) = agc_acb::dispatch(nid, arguments, state) {
        return Some(result);
    }
    if let Some(result) = agc_state::dispatch(nid, arguments, state) {
        return Some(result);
    }
    if let Some(result) = agc_driver::dispatch(nid, arguments, state) {
        return Some(result);
    }
    if let Some(result) = ampr::dispatch(nid, arguments, state) {
        return Some(result);
    }
    if let Some(result) = kernel_api::dispatch(nid, arguments, state) {
        return Some(result);
    }
    if let Some(result) = av::dispatch(nid, arguments, state) {
        return Some(result);
    }
    if let Some(result) = videodec2::dispatch(nid, arguments, state) {
        return Some(result);
    }
    if let Some(result) = ces::dispatch(nid, arguments, state) {
        return Some(result);
    }
    None
}

/// Destroys an event queue and discards the registrations and undelivered events it owned.
fn kernel_delete_equeue(handle: u64, state: &mut ProbeDispatchState) -> u64 {
    if !state.kernel_equeues.remove(&handle) {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    state
        .kernel_event_registrations
        .retain(|&(equeue, _, _), _| equeue != handle);
    discard_kernel_events(state, |event| event.equeue == handle);
    notify_kernel_equeue_waiters();
    0
}

/// Registers interest in display output-mode changes.
///
/// REmu presents a single fixed output mode, so no change is ever announced; the registration is
/// still recorded so the queue reports the same shape the console would.
fn video_out_add_output_mode_event(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (equeue, handle, user_data) = (arguments[0], arguments[1], arguments[2]);
    if low_i32(handle) != 1 || !state.video_out.is_open {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    kernel_register_event(state, equeue, 0, KERNEL_EVENT_FILTER_VIDEO_OUT, user_data)
}

/// Accepts window-mode margins. REmu scales presentation rather than insetting the guest surface.
fn video_out_set_window_mode_margins(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if low_i32(arguments[0]) != 1 || !state.video_out.is_open {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    0
}

fn kernel_create_equeue(output: u64, state: &mut ProbeDispatchState) -> u64 {
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let handle = state.next_equeue.max(1);
    state.next_equeue = handle.saturating_add(1);
    state.kernel_equeues.insert(handle);
    write_guest_u64(output, handle);
    0
}

fn kernel_register_event(
    state: &mut ProbeDispatchState,
    equeue: u64,
    ident: u64,
    filter: i16,
    user_data: u64,
) -> u64 {
    if !state.kernel_equeues.contains(&equeue) {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    state
        .kernel_event_registrations
        .insert((equeue, filter, ident), user_data);
    0
}

fn kernel_add_user_event_edge(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    kernel_register_event(
        state,
        arguments[0],
        arguments[1],
        KERNEL_EVENT_FILTER_USER,
        0,
    )
}

/// Triggers a user event that a queue registered through `sceKernelAddUserEventEdge`.
///
/// The console delivers the registration's own user data, so a listener can resolve the event
/// back to its context; the trigger's identifier selects which registration fires.
fn kernel_trigger_user_event(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (equeue, ident) = (arguments[0], arguments[1]);
    let Some(&user_data) =
        state
            .kernel_event_registrations
            .get(&(equeue, KERNEL_EVENT_FILTER_USER, ident))
    else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    queue_kernel_event(
        state,
        KernelQueuedEvent {
            equeue,
            ident,
            filter: KERNEL_EVENT_FILTER_USER,
            flags: 0,
            fflags: 1,
            data: arguments[2],
            user_data,
            agc_graphics_completions: 0,
            agc_user_interrupts: 0,
        },
    );
    notify_kernel_equeue_waiters();
    0
}

/// Reads the identifier of a delivered kernel event.
fn kernel_get_event_id(event: u64) -> u64 {
    if event == 0 {
        return 0;
    }
    read_guest_u64(event)
}

/// Reads the `data` field of a delivered kernel event.
///
/// The accessor exists because the event layout is opaque to the title; the field sits after the
/// identifier, filter, flags, and filter flags.
fn kernel_get_event_data(event: u64) -> u64 {
    if event == 0 {
        return 0;
    }
    read_guest_u64(event + 0x10)
}

/// Applies a scheduling policy and priority to a running guest thread.
///
/// REmu runs guest threads on host threads whose priorities the host scheduler owns, so the
/// request is validated and accepted without changing host scheduling. Rejecting it would fail
/// engine worker setup for a property the guest cannot observe directly.
fn pthread_set_schedule_parameters(arguments: [u64; 6]) -> u64 {
    let (thread, parameters) = (arguments[0], arguments[2]);
    if thread == 0 || parameters == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    0
}

/// Registers interest in APR/AMPR command-buffer completion on an event queue.
///
/// Registration only expresses interest. Completions are delivered when a command buffer that
/// recorded a queue-write command is submitted; announcing one here would wake the guest's
/// listener for a transfer that never ran, and it would then resolve an event context that does
/// not exist yet.
fn kernel_add_ampr_event(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    kernel_register_event(
        state,
        arguments[0],
        arguments[1],
        KERNEL_EVENT_FILTER_AMPR,
        arguments[2],
    )
}

fn agc_driver_add_eq_event(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let result = kernel_register_event(
        state,
        arguments[0],
        arguments[1],
        KERNEL_EVENT_FILTER_GRAPHICS,
        arguments[2],
    );
    if result == 0 {
        eprintln!(
            "[REMu][AGC] registered completion event equeue={:#x} ident={:#x} user_data={:#x}",
            arguments[0], arguments[1], arguments[2]
        );
    }
    result
}

fn kernel_equeue_wake() -> &'static (Mutex<u64>, Condvar) {
    KERNEL_EQUEUE_WAKE.get_or_init(|| (Mutex::new(0), Condvar::new()))
}

fn notify_kernel_equeue_waiters() {
    let wake = kernel_equeue_wake();
    let mut generation = wake
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *generation = generation.wrapping_add(1);
    wake.1.notify_all();
}

fn queue_kernel_event(state: &mut ProbeDispatchState, mut event: KernelQueuedEvent) {
    if let Some(pending) = state.pending_events.iter_mut().find(|pending| {
        pending.equeue == event.equeue
            && pending.ident == event.ident
            && pending.filter == event.filter
    }) {
        event.fflags = pending.fflags.saturating_add(1).max(event.fflags);
        event.agc_graphics_completions = pending
            .agc_graphics_completions
            .saturating_add(event.agc_graphics_completions);
        event.agc_user_interrupts = pending
            .agc_user_interrupts
            .saturating_add(event.agc_user_interrupts);
        *pending = event;
    } else {
        state.pending_events.push_back(event);
    }
}

fn discard_kernel_events(
    state: &mut ProbeDispatchState,
    mut should_discard: impl FnMut(&KernelQueuedEvent) -> bool,
) {
    let mut cancelled_graphics = 0_u64;
    let mut cancelled_user = 0_u64;
    state.pending_events.retain(|event| {
        if should_discard(event) {
            cancelled_graphics =
                cancelled_graphics.saturating_add(u64::from(event.agc_graphics_completions));
            cancelled_user = cancelled_user.saturating_add(u64::from(event.agc_user_interrupts));
            false
        } else {
            true
        }
    });
    state
        .agc_completion_telemetry
        .cancelled_graphics_notifications = state
        .agc_completion_telemetry
        .cancelled_graphics_notifications
        .saturating_add(cancelled_graphics);
    state.agc_completion_telemetry.cancelled_user_interrupts = state
        .agc_completion_telemetry
        .cancelled_user_interrupts
        .saturating_add(cancelled_user);
}

fn refresh_completed_agc_submissions(state: &mut ProbeDispatchState) {
    const AGC_USER_INTERRUPT_EVENT: u64 = 0x1800;
    let completed = remu_gpu::take_gpu_completed_submission_ids()
        .into_iter()
        .collect::<BTreeSet<_>>();
    if completed.is_empty() {
        return;
    }
    let newest_completed = completed.iter().next_back().copied().unwrap_or_default();
    let mut completed_queues = Vec::new();
    let mut claimed = BTreeSet::new();
    state.pending_agc_completions.retain(|&(sequence, ident)| {
        if completed.contains(&sequence) {
            completed_queues.push((sequence, ident));
            claimed.insert(sequence);
            false
        } else {
            true
        }
    });
    let mut completed_compute = Vec::new();
    state.pending_agc_user_interrupts.retain(|sequence| {
        if completed.contains(sequence) {
            completed_compute.push(*sequence);
            claimed.insert(*sequence);
            false
        } else {
            true
        }
    });
    for &sequence in completed.difference(&claimed) {
        state.agc_completion_telemetry.unmatched_backend_ids = state
            .agc_completion_telemetry
            .unmatched_backend_ids
            .saturating_add(1);
        state.agc_completion_telemetry.last_unmatched_backend_id = Some(sequence);
    }
    let mut queued = false;
    for (sequence, completed_ident) in completed_queues {
        // `sceAgcDriverAddEqEvent` registers one hardware queue by ID. Queue zero is graphics and
        // an ACB queue uses its owner ID (Silent Hill uses 0x20). Broadcasting one queue's
        // completion to every registration wakes unrelated interrupt work before its command
        // buffer has retired, allowing the guest to recycle live RHI task nodes.
        let registrations = state
            .kernel_event_registrations
            .iter()
            .filter_map(|(&(equeue, filter, ident), &user_data)| {
                (filter == KERNEL_EVENT_FILTER_GRAPHICS && ident == completed_ident)
                    .then_some((equeue, ident, filter, user_data))
            })
            .collect::<Vec<_>>();
        state.agc_completion_telemetry.refreshed_graphics = state
            .agc_completion_telemetry
            .refreshed_graphics
            .saturating_add(1);
        state.agc_completion_telemetry.last_refreshed_graphics = Some(sequence);
        if registrations.is_empty() {
            state.agc_completion_telemetry.missing_graphics_registration = state
                .agc_completion_telemetry
                .missing_graphics_registration
                .saturating_add(1);
        }
        for (equeue, ident, filter, user_data) in registrations {
            queue_kernel_event(
                state,
                KernelQueuedEvent {
                    equeue,
                    ident,
                    filter,
                    flags: 0,
                    fflags: 1,
                    data: ident,
                    user_data,
                    agc_graphics_completions: 1,
                    agc_user_interrupts: 0,
                },
            );
            state.agc_completion_telemetry.queued_graphics_notifications = state
                .agc_completion_telemetry
                .queued_graphics_notifications
                .saturating_add(1);
            queued = true;
        }
        if trace_logarithmic_sequence(sequence, 8) {
            eprintln!(
                "[REMu][AGC] completed submission={sequence} newest_backend={newest_completed} equeues={}",
                state
                    .kernel_event_registrations
                    .keys()
                    .filter(|(_, filter, _)| *filter == KERNEL_EVENT_FILTER_GRAPHICS)
                    .count()
            );
        }
    }
    for sequence in completed_compute {
        // Compute submissions use the driver-wide AGC user interrupt rather than the graphics
        // end-of-pipe registrations. This is the signal the guest's AgcSubmissionThread waits on
        // before it can recycle the ACB and submit the next frame.
        let timestamp = kernel_process_time(true);
        let registrations = state
            .kernel_event_registrations
            .iter()
            .filter_map(|(&(equeue, filter, ident), &user_data)| {
                (filter == KERNEL_EVENT_FILTER_USER && ident == AGC_USER_INTERRUPT_EVENT)
                    .then_some((equeue, ident, filter, user_data))
            })
            .collect::<Vec<_>>();
        state.agc_completion_telemetry.refreshed_compute = state
            .agc_completion_telemetry
            .refreshed_compute
            .saturating_add(1);
        state.agc_completion_telemetry.last_refreshed_compute = Some(sequence);
        if registrations.is_empty() {
            state
                .agc_completion_telemetry
                .missing_user_interrupt_registration = state
                .agc_completion_telemetry
                .missing_user_interrupt_registration
                .saturating_add(1);
        }
        for (equeue, ident, filter, user_data) in registrations {
            queue_kernel_event(
                state,
                KernelQueuedEvent {
                    equeue,
                    ident,
                    filter,
                    flags: 0,
                    fflags: 1,
                    data: timestamp,
                    user_data,
                    agc_graphics_completions: 0,
                    agc_user_interrupts: 1,
                },
            );
            state.agc_completion_telemetry.queued_user_interrupts = state
                .agc_completion_telemetry
                .queued_user_interrupts
                .saturating_add(1);
            queued = true;
        }
        if trace_logarithmic_sequence(sequence, 8) {
            eprintln!(
                "[REMu][AGC] completed compute submission={sequence} newest_backend={newest_completed} user_interrupts={}",
                state
                    .kernel_event_registrations
                    .keys()
                    .filter(|(_, filter, ident)| {
                        *filter == KERNEL_EVENT_FILTER_USER && *ident == AGC_USER_INTERRUPT_EVENT
                    })
                    .count()
            );
        }
    }
    if queued {
        notify_kernel_equeue_waiters();
    }
}

fn kernel_poll_equeue(arguments: [u64; 6], state: &mut ProbeDispatchState) -> Result<u32, u64> {
    let handle = arguments[0];
    let events = arguments[1];
    let event_capacity = low_u32(arguments[2]);
    let out_count = arguments[3];
    if !state.kernel_equeues.contains(&handle) {
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND));
    }
    if events == 0 || event_capacity == 0 {
        return Err(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
    }
    refresh_completed_video_out_flips(state);
    refresh_completed_agc_submissions(state);
    let mut delivered = 0_u32;
    while delivered < event_capacity {
        let Some(index) = state
            .pending_events
            .iter()
            .position(|event| event.equeue == handle)
        else {
            break;
        };
        let event = state
            .pending_events
            .remove(index)
            .expect("the queued event index was just found");
        let Some(output) = events.checked_add(u64::from(delivered) * 0x20) else {
            return Err(sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT));
        };
        write_guest_u64(output, event.ident);
        write_guest_u16(output + 0x08, event.filter.cast_unsigned());
        write_guest_u16(output + 0x0a, event.flags);
        write_guest_u32(output + 0x0c, event.fflags);
        write_guest_u64(output + 0x10, event.data);
        write_guest_u64(output + 0x18, event.user_data);
        state
            .agc_completion_telemetry
            .delivered_graphics_notifications = state
            .agc_completion_telemetry
            .delivered_graphics_notifications
            .saturating_add(u64::from(event.agc_graphics_completions));
        state.agc_completion_telemetry.delivered_user_interrupts = state
            .agc_completion_telemetry
            .delivered_user_interrupts
            .saturating_add(u64::from(event.agc_user_interrupts));
        if event.filter == KERNEL_EVENT_FILTER_GRAPHICS {
            static GRAPHICS_DELIVERY_COUNT: AtomicUsize = AtomicUsize::new(0);
            let sequence = GRAPHICS_DELIVERY_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
            if sequence <= 64 || sequence.is_power_of_two() {
                eprintln!(
                    "[REMu][AGC][EQUEUE] delivered sequence={sequence} equeue={handle:#x} ident={:#x} flags={:#x} fflags={} data={:#x} user_data={:#x}",
                    event.ident, event.flags, event.fflags, event.data, event.user_data
                );
            }
        }
        delivered += 1;
    }
    if out_count != 0 {
        write_guest_u32(out_count, delivered);
    }
    Ok(delivered)
}

fn kernel_wait_equeue(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    kernel_poll_equeue(arguments, state).map_or_else(std::convert::identity, |_| 0)
}

fn kernel_wait_equeue_blocking(arguments: [u64; 6]) -> u64 {
    static WAIT_COUNT: AtomicUsize = AtomicUsize::new(0);
    let wait_sequence = WAIT_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    let timeout_pointer = arguments[4];
    // Orbis defines the timeout as a pointer to a 32-bit microsecond count. Reading or writing a
    // u64 here overwrites the first four bytes of an adjacent SceKernelEvent when callers keep the
    // timeout immediately before their event array (Unity does exactly that).
    let timeout_microseconds =
        (timeout_pointer != 0).then(|| u64::from(read_guest_u32(timeout_pointer)));
    let deadline = timeout_microseconds
        .and_then(|microseconds| Instant::now().checked_add(Duration::from_micros(microseconds)));
    let guest_thread = current_guest_thread_handle();
    if wait_sequence <= 64 || wait_sequence.is_power_of_two() {
        let thread_name = current_guest_thread_name();
        eprintln!(
            "[REMu][KERNEL][EQUEUE] wait sequence={wait_sequence} equeue={:#x} events={:#x} capacity={} out_count={:#x} timeout_ptr={timeout_pointer:#x} timeout_us={timeout_microseconds:?} thread={thread_name:?}/{guest_thread:#x}",
            arguments[0],
            arguments[1],
            low_u32(arguments[2]),
            arguments[3]
        );
    }
    loop {
        // Snapshot the wake generation without nesting the notifier and dispatch locks. Producers
        // hold the dispatch lock while enqueueing, so a nested inverse order could deadlock.
        let generation = *kernel_equeue_wake()
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let result = {
            let state = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
            let mut state = lock_probe_dispatch_for_write(state);
            kernel_poll_equeue(arguments, &mut state)
        };
        match result {
            Err(error) => return error,
            Ok(count) if count != 0 => {
                if let (Some(deadline), true) = (deadline, timeout_pointer != 0) {
                    let remaining = deadline
                        .checked_duration_since(Instant::now())
                        .map_or(0, |duration| {
                            u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
                        });
                    write_guest_u32(
                        timeout_pointer,
                        u32::try_from(remaining).unwrap_or(u32::MAX),
                    );
                }
                if wait_sequence <= 64 || wait_sequence.is_power_of_two() {
                    eprintln!(
                        "[REMu][KERNEL][EQUEUE] woke sequence={wait_sequence} equeue={:#x} delivered={count}",
                        arguments[0]
                    );
                }
                return 0;
            }
            Ok(_) => {}
        }
        if guest_exception_is_pending(guest_thread) {
            return sce_kernel_error(SCE_KERNEL_ERROR_INTERRUPTED);
        }
        let wait_for = if let Some(deadline) = deadline {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                write_guest_u32(timeout_pointer, 0);
                return sce_kernel_error(SCE_KERNEL_ERROR_TIMED_OUT);
            };
            Some(remaining)
        } else {
            None
        };
        let wake = kernel_equeue_wake();
        let wake_state = wake
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *wake_state != generation {
            continue;
        }
        if let Some(wait_for) = wait_for {
            let _ = wake
                .1
                .wait_timeout(wake_state, wait_for)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        } else {
            drop(
                wake.1
                    .wait(wake_state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner),
            );
        }
    }
}

fn initialize_guest_libc(environment: u64, state: &mut ProbeDispatchState) -> u64 {
    match state.services.libc_environment {
        Some(previous) if previous != environment => {
            sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT)
        }
        Some(_) => 0,
        None => {
            state.services.libc_environment = Some(environment);
            0
        }
    }
}

fn register_guest_exit_callback(
    function: u64,
    argument: Option<u64>,
    module: u64,
    state: &mut ProbeDispatchState,
) -> u64 {
    if function == 0 {
        return u64::MAX;
    }
    state.exit_callbacks.push(GuestExitCallback {
        function,
        argument,
        module,
    });
    0
}

fn take_exit_callbacks(
    state: &mut ProbeDispatchState,
    module: Option<u64>,
) -> Vec<GuestExitCallback> {
    let mut callbacks = Vec::new();
    let mut index = state.exit_callbacks.len();
    while index != 0 {
        index -= 1;
        let callback = state.exit_callbacks[index];
        let matches = module.is_none_or(|module| {
            module == 0 || (callback.argument.is_some() && callback.module == module)
        });
        if matches {
            callbacks.push(state.exit_callbacks.remove(index));
        }
    }
    callbacks
}

fn invoke_guest_exit_callbacks(callbacks: Vec<GuestExitCallback>) {
    for callback in callbacks {
        if let Some(argument) = callback.argument {
            type Destructor = unsafe extern "C" fn(u64);
            // SAFETY: The guest registered a live Itanium destructor function and its opaque
            // argument. It remains executable until the corresponding module is finalized.
            let destructor: Destructor = unsafe { std::mem::transmute(callback.function) };
            // SAFETY: The registration defines the function and argument ABI above.
            unsafe { destructor(argument) };
        } else {
            type Callback = unsafe extern "C" fn();
            // SAFETY: C atexit registers a live no-argument callback in guest executable memory.
            let callback: Callback = unsafe { std::mem::transmute(callback.function) };
            // SAFETY: The registration defines the no-argument callback ABI above.
            unsafe { callback() };
        }
    }
}

fn run_all_guest_exit_callbacks() {
    loop {
        let callbacks = PROBE_DISPATCH.get().map_or_else(Vec::new, |state| {
            let mut state = state
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            take_exit_callbacks(&mut state, None)
        });
        if callbacks.is_empty() {
            return;
        }
        invoke_guest_exit_callbacks(callbacks);
    }
}

fn sysmodule_load(module: u64, state: &mut ProbeDispatchState) -> u64 {
    let module = low_i32(module);
    if module < 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state.services.loaded_sysmodules.insert(module);
    0
}

fn sysmodule_unload(module: u64, state: &mut ProbeDispatchState) -> u64 {
    const SYSMODULE_INVALID_ID: u32 = 0x805a_1000;
    const SYSMODULE_NOT_LOADED: u32 = 0x805a_1001;
    let module = low_i32(module);
    if module <= 0 {
        return sce_kernel_error(SYSMODULE_INVALID_ID);
    }
    if !state.services.loaded_sysmodules.remove(&module) {
        return sce_kernel_error(SYSMODULE_NOT_LOADED);
    }
    0
}

fn sysmodule_is_loaded(module: u64, state: &ProbeDispatchState) -> u64 {
    const SYSMODULE_INVALID_ID: u32 = 0x805a_1000;
    const SYSMODULE_NOT_LOADED: u32 = 0x805a_1001;
    let module = low_i32(module);
    if module <= 0 {
        return sce_kernel_error(SYSMODULE_INVALID_ID);
    }
    if state.services.loaded_sysmodules.contains(&module) {
        0
    } else {
        sce_kernel_error(SYSMODULE_NOT_LOADED)
    }
}

fn game_update_initialize(state: &mut ProbeDispatchState) -> u64 {
    const GAME_UPDATE_SYSMODULE: i32 = 0xcc;
    if !state
        .services
        .loaded_sysmodules
        .contains(&GAME_UPDATE_SYSMODULE)
    {
        return sce_kernel_error(0x805a_1001);
    }
    state.services.game_update = InitializationState::Initialized;
    0
}

fn np_entitlement_access_initialize(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if arguments[0] == 0 || arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state.services.np_entitlement_access = Some((arguments[0], arguments[1]));
    0
}

fn np_session_signaling_initialize(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    const PARAMETER_SIZE: usize = 0x28;
    if arguments[0] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    // SAFETY: this ABI takes a fixed-size initialization structure that remains caller-owned.
    let parameter =
        unsafe { std::slice::from_raw_parts(arguments[0] as *const u8, PARAMETER_SIZE).to_vec() };
    state.services.np_session_signaling = Some(parameter);
    0
}

fn np_web_api2_create_user_context(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    const INVALID_ARGUMENT: u32 = 0x8055_3402;
    const INVALID_LIBRARY_CONTEXT: u32 = 0x8055_3403;
    const USER_CONTEXT_ALREADY_EXISTS: u32 = 0x8055_340a;
    let (library_context, user_id) = (low_i32(arguments[0]), low_i32(arguments[1]));
    if !(1..0x8000).contains(&library_context) {
        return sce_kernel_error(INVALID_LIBRARY_CONTEXT);
    }
    if user_id == -1 {
        return sce_kernel_error(INVALID_ARGUMENT);
    }
    if state
        .services
        .np_web_api2_user_contexts
        .values()
        .any(|&(library, user)| library == library_context && user == user_id)
    {
        return sce_kernel_error(USER_CONTEXT_ALREADY_EXISTS);
    }
    let context = state.services.next_np_web_api2_user_context.max(1);
    state.services.next_np_web_api2_user_context = context.saturating_add(1);
    state
        .services
        .np_web_api2_user_contexts
        .insert(context, (library_context, user_id));
    signed_i32_return(context)
}

/// `sceNpWebApi2PushEventCreateHandle(userCtxId) -> handle`.
///
/// Offline titles still allocate push-event handles so later register/unregister calls have a
/// stable identifier. No push traffic is ever delivered. The handle is returned like
/// `Initialize` / `CreateUserContext` (rax), not written through an out-parameter: observed
/// call sites pass a non-pointer second register that must not be treated as a store target.
fn np_web_api2_push_event_create_handle(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    const INVALID_USER_CONTEXT: u32 = 0x8055_3404;
    let user_context = low_i32(arguments[0]);
    if user_context <= 0 || user_context >= 0x8000 {
        return sce_kernel_error(INVALID_USER_CONTEXT);
    }
    let handle = state.services.next_np_web_api2_push_event_handle.max(1);
    state.services.next_np_web_api2_push_event_handle = handle.saturating_add(1);
    state
        .services
        .np_web_api2_push_event_handles
        .insert(handle, user_context);
    signed_i32_return(handle)
}

fn np_check_callback(state: &mut ProbeDispatchState) -> u64 {
    state.services.np_callback_polls = state.services.np_callback_polls.saturating_add(1);
    0
}

const SCE_NP_ERROR_INVALID_ARGUMENT: u32 = 0x8055_0003;

fn np_set_np_title_id(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    const NP_TITLE_ID_SIZE: usize = 16;
    const NP_TITLE_SECRET_SIZE: usize = 128;
    if arguments[0] == 0 || arguments[1] == 0 {
        return sce_kernel_error(SCE_NP_ERROR_INVALID_ARGUMENT);
    }

    let mut title_id = [0_u8; NP_TITLE_ID_SIZE];
    let mut title_secret = [0_u8; NP_TITLE_SECRET_SIZE];
    if !ProcessGuestMemory.read(arguments[0], &mut title_id)
        || !ProcessGuestMemory.read(arguments[1], &mut title_secret)
    {
        return sce_kernel_error(SCE_NP_ERROR_INVALID_ARGUMENT);
    }
    state.services.np_title_id = Some(title_id);
    // The secret is caller-owned authentication material. Validate the complete fixed-width ABI
    // record without retaining or logging it.
    std::hint::black_box(title_secret);
    0
}

fn share_feature_permit(state: &mut ProbeDispatchState) -> u64 {
    state.services.share_feature_permitted = true;
    0
}

fn share_set_content_parameter(address: u64, state: &mut ProbeDispatchState) -> u64 {
    const MAXIMUM_CONTENT_PARAMETER_BYTES: usize = 4_096;
    if address == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }

    let mut bytes = Vec::with_capacity(128);
    while bytes.len() < MAXIMUM_CONTENT_PARAMETER_BYTES {
        let Some(chunk_address) = address.checked_add(bytes.len() as u64) else {
            return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
        };
        // Reading only to the next 4 KiB boundary lets a valid short string end immediately before
        // an unmapped page. `ProcessGuestMemory` validates every chunk without dereferencing an
        // untrusted guest pointer in the host process.
        let page_bytes = 4_096 - usize::try_from(chunk_address & 0xfff).unwrap_or_default();
        let chunk_length =
            page_bytes.min(MAXIMUM_CONTENT_PARAMETER_BYTES.saturating_sub(bytes.len()));
        let mut chunk = vec![0_u8; chunk_length];
        if !ProcessGuestMemory.read(chunk_address, &mut chunk) {
            return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
        }
        if let Some(end) = chunk.iter().position(|byte| *byte == 0) {
            bytes.extend_from_slice(&chunk[..end]);
            let Ok(parameter) = String::from_utf8(bytes) else {
                return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
            };
            state.services.share_content_parameter = Some(parameter);
            return 0;
        }
        bytes.extend_from_slice(&chunk);
    }
    sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT)
}

fn ult_initialize(runtime: u64, state: &mut ProbeDispatchState) -> u64 {
    if runtime == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    match state.services.ult_runtime {
        Some(previous) if previous != runtime => {
            sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT)
        }
        Some(_) => 0,
        None => {
            state.services.ult_runtime = Some(runtime);
            0
        }
    }
}

const MAXIMUM_SYSTEM_SPLASH_PIXELS: usize = 64 * 1024 * 1024;

fn load_system_splash(game_root: &Path) -> Result<(PathBuf, remu_gpu::GpuHostImage), String> {
    let path = resolve_path_beneath(game_root, "sce_sys/pic0.png")
        .filter(|path| path.is_file())
        .ok_or_else(|| {
            format!(
                "installed title has no readable /sce_sys/pic0.png beneath {}",
                game_root.display()
            )
        })?;
    let file = std::fs::File::open(&path)
        .map_err(|error| format!("could not open {}: {error}", path.display()))?;
    let mut decoder = png::Decoder::new(BufReader::new(file));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder
        .read_info()
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    let metadata = reader.info();
    let pixel_count = usize::try_from(metadata.width)
        .ok()
        .and_then(|width| {
            usize::try_from(metadata.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .filter(|pixels| *pixels != 0 && *pixels <= MAXIMUM_SYSTEM_SPLASH_PIXELS)
        .ok_or_else(|| {
            format!(
                "{} has unsupported splash dimensions {}x{}",
                path.display(),
                metadata.width,
                metadata.height
            )
        })?;
    let mut decoded = vec![0_u8; reader.output_buffer_size()];
    let frame = reader
        .next_frame(&mut decoded)
        .map_err(|error| format!("could not decode {}: {error}", path.display()))?;
    let decoded = &decoded[..frame.buffer_size()];
    let mut rgba = Vec::with_capacity(
        pixel_count
            .checked_mul(4)
            .ok_or_else(|| format!("{} splash byte size overflows", path.display()))?,
    );
    match frame.color_type {
        png::ColorType::Rgba => rgba.extend_from_slice(decoded),
        png::ColorType::Rgb => {
            for pixel in decoded.chunks_exact(3) {
                rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], u8::MAX]);
            }
        }
        png::ColorType::GrayscaleAlpha => {
            for pixel in decoded.chunks_exact(2) {
                rgba.extend_from_slice(&[pixel[0], pixel[0], pixel[0], pixel[1]]);
            }
        }
        png::ColorType::Grayscale => {
            for &value in decoded {
                rgba.extend_from_slice(&[value, value, value, u8::MAX]);
            }
        }
        png::ColorType::Indexed => {
            return Err(format!(
                "{} remained indexed after PNG expansion",
                path.display()
            ));
        }
    }
    let image = remu_gpu::GpuHostImage::new(frame.width, frame.height, rgba)
        .ok_or_else(|| format!("{} did not decode to complete RGBA8 pixels", path.display()))?;
    Ok((path, image))
}

#[cfg(target_arch = "x86_64")]
fn publish_system_splash(splash: Result<(PathBuf, remu_gpu::GpuHostImage), String>) {
    match splash {
        Ok((path, image)) => {
            let (width, height) = (image.width(), image.height());
            if publish_gpu_event(remu_gpu::GpuEvent::ShowHostSplash(image)) {
                eprintln!(
                    "[REMu][SYSTEM] showing installed splash {} ({width}x{height})",
                    path.display()
                );
            } else {
                eprintln!(
                    "[REMu][SYSTEM] presenter unavailable for {}",
                    path.display()
                );
            }
        }
        Err(error) => eprintln!("[REMu][SYSTEM] {error}"),
    }
}

fn system_service_hide_splash(state: &mut ProbeDispatchState) -> u64 {
    if state.services.splash == SplashState::Hidden {
        return 0;
    }
    state.services.splash = SplashState::Hidden;
    if !publish_gpu_event(remu_gpu::GpuEvent::HideHostSplash) {
        eprintln!("[REMu][SYSTEM] presenter unavailable while hiding system splash");
    } else {
        eprintln!("[REMu][SYSTEM] system splash hidden by guest");
    }
    0
}

#[cfg(test)]
mod system_splash_tests;

fn user_service_initialize(state: &mut ProbeDispatchState) -> u64 {
    state.services.user_service = InitializationState::Initialized;
    0
}

fn user_service_get_event(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    const USER_SERVICE_ERROR_INVALID_ARGUMENT: u32 = 0x8096_0005;
    const USER_SERVICE_ERROR_NO_EVENT: u32 = 0x8096_0007;
    if arguments[0] == 0 {
        return sce_kernel_error(USER_SERVICE_ERROR_INVALID_ARGUMENT);
    }
    if state.services.user_login_event_delivered {
        return sce_kernel_error(USER_SERVICE_ERROR_NO_EVENT);
    }
    write_guest_u32(arguments[0], 0);
    write_guest_u32(arguments[0] + 4, 0x1000_0000);
    state.services.user_login_event_delivered = true;
    0
}

fn agc_initialize(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let agc_state = arguments[0];
    let version = low_u32(arguments[1]);
    if agc_state == 0 || !matches!(version, 7 | 8 | 10 | 13) {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state.services.agc_states.insert(agc_state, version);
    0
}

fn user_service_get_login_user_id_list(arguments: [u64; 6]) -> u64 {
    let output = arguments[0];
    if output == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    write_guest_u32(output, 0x1000_0000);
    for offset in [0x04, 0x08, 0x0c] {
        write_guest_u32(output + offset, u32::MAX);
    }
    0
}

fn np_get_account_country(output: u64) -> u64 {
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    // SceNpCountryCode is two ISO-3166 letters followed by a terminator and padding. Network
    // state is reported separately as signed out, but callers still expect deterministic profile
    // metadata rather than uninitialized stack bytes.
    write_guest_bytes(output, b"US\0\0");
    0
}

/// `sceNpGetOnlineId(userId, onlineIdOut)`.
///
/// There is no signed-in PSN account on this host. Callers still receive a fixed-width
/// `SceNpOnlineId` (16 data bytes including the terminator) so offline profile paths do not read
/// uninitialized memory.
fn np_get_online_id(arguments: [u64; 6]) -> u64 {
    const ONLINE_ID_SIZE: usize = 16;
    const ONLINE_ID: &[u8; ONLINE_ID_SIZE] = b"REmuUser\0\0\0\0\0\0\0\0";
    if low_i32(arguments[0]) != 0x1000_0000 {
        return sce_kernel_error(SCE_NP_ERROR_INVALID_ARGUMENT);
    }
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_NP_ERROR_INVALID_ARGUMENT);
    }
    write_guest_bytes(arguments[1], ONLINE_ID);
    0
}

fn system_service_get_status(arguments: [u64; 6]) -> u64 {
    let output = arguments[0];
    if output == 0 {
        return u64::from(libc::EINVAL.cast_unsigned());
    }
    guest_memset(output, 0, 0x0c);
    write_guest_u8(output + 0x06, 1);
    0
}

fn system_service_param_get_int(arguments: [u64; 6]) -> u64 {
    const SYSTEM_SERVICE_ERROR_PARAMETER: u32 = 0x80a1_0003;

    if arguments[1] == 0 {
        return sce_kernel_error(SYSTEM_SERVICE_ERROR_PARAMETER);
    }
    let value: i32 = match low_i32(arguments[0]) {
        1 | 2 | 3 | 1000 => 1,
        4 => 180,
        _ => 0,
    };
    write_guest_u32(arguments[1], value.cast_unsigned());
    0
}

fn system_service_param_get_string(arguments: [u64; 6]) -> u64 {
    const SYSTEM_SERVICE_ERROR_PARAMETER: u32 = 0x80a1_0003;

    let (param, output, capacity) = (low_i32(arguments[0]), arguments[1], arguments[2]);
    if output == 0 || capacity == 0 {
        return sce_kernel_error(SYSTEM_SERVICE_ERROR_PARAMETER);
    }
    let value: &[u8] = match param {
        1 => b"en\0",
        2 => b"en-US\0",
        3 | 6 => b"US\0",
        4 => b"America/Los_Angeles\0",
        5 => b"\0",
        _ => return sce_kernel_error(SYSTEM_SERVICE_ERROR_PARAMETER),
    };
    let Some(capacity) = usize::try_from(capacity).ok() else {
        return sce_kernel_error(SYSTEM_SERVICE_ERROR_PARAMETER);
    };
    if capacity < value.len() {
        return sce_kernel_error(SYSTEM_SERVICE_ERROR_PARAMETER);
    }
    write_guest_bytes(output, value);
    if capacity > value.len() {
        guest_memset(
            output + u64::try_from(value.len()).unwrap_or(u64::MAX),
            0,
            u64::try_from(capacity - value.len()).unwrap_or(0),
        );
    }
    0
}

/// Reports that the platform notice screen has not already been acknowledged by the system.
///
/// Titles use this one-byte flag to decide whether to show their legal/health notice during
/// first-run startup. Returning an error without initializing it leaves the guest's stack byte
/// stale and can make a fresh install skip the notice nondeterministically.
fn system_service_get_notice_screen_skip_flag(output: u64) -> u64 {
    let Some(output) = guest_output_pointer(output, 1) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    write_guest_u8(output, 0);
    0
}

fn guest_uuid_create(output: u64) -> u64 {
    static NEXT_UUID: AtomicU32 = AtomicU32::new(1);
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let sequence = u64::from(NEXT_UUID.fetch_add(1, Ordering::Relaxed));
    let seconds = current_unix_seconds().cast_unsigned();
    let mut bytes = [0_u8; 16];
    bytes[..8].copy_from_slice(&(seconds.rotate_left(17) ^ sequence).to_le_bytes());
    bytes[8..]
        .copy_from_slice(&(sequence.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ output).to_le_bytes());
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    write_guest_bytes(output, &bytes);
    0
}

fn guest_kernel_is_stack(arguments: [u64; 6]) -> u64 {
    if arguments[1] != 0 {
        write_guest_u64(arguments[1], 0);
    }
    if arguments[2] != 0 {
        write_guest_u64(arguments[2], 0);
    }
    0
}

fn user_service_get_user_name(arguments: [u64; 6]) -> u64 {
    const USER_SERVICE_ERROR_INVALID_ARGUMENT: u32 = 0x8096_0005;
    const USER_SERVICE_ERROR_INVALID_PARAMETER: u32 = 0x8096_0009;
    const USER_SERVICE_ERROR_BUFFER_TOO_SHORT: u32 = 0x8096_000a;
    const USER_NAME: &[u8] = b"REmu\0";

    if low_i32(arguments[0]) != 0x1000_0000 {
        return sce_kernel_error(USER_SERVICE_ERROR_INVALID_PARAMETER);
    }
    if arguments[1] == 0 {
        return sce_kernel_error(USER_SERVICE_ERROR_INVALID_ARGUMENT);
    }
    if arguments[2] < USER_NAME.len() as u64 {
        return sce_kernel_error(USER_SERVICE_ERROR_BUFFER_TOO_SHORT);
    }
    write_guest_bytes(arguments[1], USER_NAME);
    0
}

fn user_service_get_age_level(arguments: [u64; 6]) -> u64 {
    const USER_SERVICE_ERROR_INVALID_ARGUMENT: u32 = 0x8096_0005;
    const USER_SERVICE_ERROR_INVALID_PARAMETER: u32 = 0x8096_0009;
    if low_i32(arguments[0]) != 0x1000_0000 {
        return sce_kernel_error(USER_SERVICE_ERROR_INVALID_PARAMETER);
    }
    if arguments[1] == 0 {
        return sce_kernel_error(USER_SERVICE_ERROR_INVALID_ARGUMENT);
    }
    write_guest_u32(arguments[1], 18);
    0
}

fn user_service_get_game_presets(arguments: [u64; 6]) -> u64 {
    const USER_SERVICE_ERROR_INVALID_ARGUMENT: u32 = 0x8096_0005;
    const USER_SERVICE_ERROR_INVALID_PARAMETER: u32 = 0x8096_0009;
    if low_i32(arguments[0]) != 0x1000_0000 {
        return sce_kernel_error(USER_SERVICE_ERROR_INVALID_PARAMETER);
    }
    if arguments[1] == 0 {
        return sce_kernel_error(USER_SERVICE_ERROR_INVALID_ARGUMENT);
    }
    guest_memset(arguments[1], 0, 0x28);
    write_guest_u64(arguments[1], 0x28);
    0
}

fn user_service_get_setting(arguments: [u64; 6], value: u32) -> u64 {
    const USER_SERVICE_ERROR_INVALID_ARGUMENT: u32 = 0x8096_0005;
    const USER_SERVICE_ERROR_INVALID_PARAMETER: u32 = 0x8096_0009;
    if low_i32(arguments[0]) != 0x1000_0000 {
        return sce_kernel_error(USER_SERVICE_ERROR_INVALID_PARAMETER);
    }
    if arguments[1] == 0 {
        return sce_kernel_error(USER_SERVICE_ERROR_INVALID_ARGUMENT);
    }
    write_guest_u32(arguments[1], value);
    0
}

fn app_content_initialize(arguments: [u64; 6]) -> u64 {
    if arguments[0] == 0 || arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    write_guest_u32(arguments[1] + 4, 0);
    0
}

fn app_content_app_param_get_int(arguments: [u64; 6]) -> u64 {
    if arguments[1] == 0 || low_u32(arguments[0]) > 4 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let value = if low_u32(arguments[0]) == 0 { 3 } else { 0 };
    write_guest_u32(arguments[1], value);
    0
}

/// Decomposes a tick count into an `SceRtcDateTime`.
///
/// This is the inverse of [`rtc_get_tick`] and the accessor engines use to turn a stored
/// timestamp back into a calendar date. Leaving the structure untouched leaves the caller with
/// whatever the stack held, which fails date validation with implausible field values.
fn rtc_set_tick(arguments: [u64; 6]) -> u64 {
    let (output, input) = (arguments[0], arguments[1]);
    if output == 0 || input == 0 {
        return sce_kernel_error(RTC_ERROR_INVALID_POINTER);
    }
    let ticks = read_guest_u64(input);
    let epoch_ticks = RTC_UNIX_EPOCH_TICK_MICROSECONDS;
    // Ticks count from year one; convert to the Unix origin the civil decomposition uses.
    let unix_microseconds = i128::from(ticks) - i128::from(epoch_ticks);
    let seconds = i64::try_from(unix_microseconds.div_euclid(1_000_000)).unwrap_or(0);
    let microsecond = u32::try_from(unix_microseconds.rem_euclid(1_000_000)).unwrap_or(0);
    let (year, month, day, hour, minute, second) = civil_from_unix_seconds(seconds);
    if year == 0 {
        return sce_kernel_error(RTC_ERROR_INVALID_ARGUMENT);
    }
    write_guest_u16(output, year);
    write_guest_u16(output + 2, month);
    write_guest_u16(output + 4, day);
    write_guest_u16(output + 6, hour);
    write_guest_u16(output + 8, minute);
    write_guest_u16(output + 10, second);
    write_guest_u32(output + 12, microsecond);
    0
}

/// Reports the display's safe area as the full display.
///
/// The console lets users shrink the safe area; REmu presents the guest's full requested surface,
/// so the ratio is one and the insets are zero.
fn system_service_get_display_safe_area_info(output: u64) -> u64 {
    let Some(output) = guest_output_pointer(output, 32) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    guest_memset(output, 0, 32);
    write_guest_u32(output, 1.0_f32.to_bits());
    0
}

/// Records a network-state callback and reports its identifier.
fn net_ctl_register_callback(arguments: [u64; 6]) -> u64 {
    static NEXT_CALLBACK: AtomicU32 = AtomicU32::new(1);
    if arguments[0] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let id = NEXT_CALLBACK.fetch_add(1, Ordering::Relaxed);
    if arguments[2] != 0 {
        write_guest_u32(arguments[2], id);
    }
    0
}

/// Converts a binary network address into its textual form.
fn net_inet_ntop(arguments: [u64; 6]) -> u64 {
    const AF_INET: i32 = 2;
    const AF_INET6: i32 = 28;
    let (family, source, destination, capacity) = (
        low_i32(arguments[0]),
        arguments[1],
        arguments[2],
        arguments[3],
    );
    if source == 0 || destination == 0 {
        return 0;
    }
    let text = match family {
        AF_INET => {
            let octets = [
                read_guest_u8(source),
                read_guest_u8(source + 1),
                read_guest_u8(source + 2),
                read_guest_u8(source + 3),
            ];
            format!("{}.{}.{}.{}", octets[0], octets[1], octets[2], octets[3])
        }
        AF_INET6 => {
            let groups: Vec<String> = (0..8)
                .map(|group| format!("{:x}", read_guest_u16(source + group * 2).to_be()))
                .collect();
            groups.join(":")
        }
        _ => return 0,
    };
    if capacity <= text.len() as u64 {
        return 0;
    }
    write_guest_bytes(destination, text.as_bytes());
    write_guest_bytes(destination + text.len() as u64, b"\0");
    destination
}

/// Converts an `SceRtcDateTime` into a tick count.
///
/// Ticks are microseconds since the start of year one, which is the same origin
/// `sceRtcGetCurrentTick` reports, so values from both calls are directly comparable.
fn rtc_get_tick(arguments: [u64; 6]) -> u64 {
    let (input, output) = (arguments[0], arguments[1]);
    if input == 0 || output == 0 {
        return sce_kernel_error(RTC_ERROR_INVALID_POINTER);
    }
    let year = i64::from(read_guest_u16(input));
    let month = i64::from(read_guest_u16(input + 2));
    let day = i64::from(read_guest_u16(input + 4));
    let hour = i64::from(read_guest_u16(input + 6));
    let minute = i64::from(read_guest_u16(input + 8));
    let second = i64::from(read_guest_u16(input + 10));
    let microsecond = i64::from(read_guest_u32(input + 12));
    if !(1..=9999).contains(&year)
        || !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return sce_kernel_error(RTC_ERROR_INVALID_ARGUMENT);
    }
    let days = unix_days_from_civil(year, month, day);
    let seconds = days * 86_400 + hour * 3_600 + minute * 60 + second;
    let ticks = i64::try_from(RTC_UNIX_EPOCH_TICK_MICROSECONDS).unwrap_or(i64::MAX)
        + seconds.saturating_mul(1_000_000)
        + microsecond;
    write_guest_u64(output, u64::from_ne_bytes(ticks.to_ne_bytes()));
    0
}

/// Days from the Unix epoch to a proleptic Gregorian date; the inverse of [`civil_from_unix_seconds`].
fn unix_days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = year - i64::from(month <= 2);
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let shifted_month = if month > 2 { month - 3 } else { month + 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// Registers the writable per-title temporary data directory.
fn app_content_temporary_data_mount(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let output = arguments[1];
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let mount = state.save_root.join("temp0");
    if let Err(error) = std::fs::create_dir_all(&mount) {
        eprintln!(
            "[REMu][VFS] temporary data mount failed {:?}: {error}",
            mount.display()
        );
        return sce_kernel_io_error(&error);
    }
    // The console reports the guest-visible mount point, not a host path.
    write_guest_bytes(output, b"/temp0\0");
    0
}

/// Terminates the process for a guest-raised release-mode exception.
///
/// The title calls this after its own fatal-error handling has already run; continuing would let
/// the guest execute past an error it declared unrecoverable.
fn kernel_debug_raise_exception_on_release_mode(arguments: [u64; 6]) -> u64 {
    eprintln!(
        "[REMu][GUEST] fatal guest exception error={:#x} argument={:#x}",
        low_u32(arguments[0]),
        arguments[1]
    );
    terminate_probe(70)
}

fn rtc_get_current_tick(output: u64) -> u64 {
    if output == 0 {
        return sce_kernel_error(RTC_ERROR_INVALID_POINTER);
    }
    let unix_microseconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| {
            duration
                .as_secs()
                .saturating_mul(1_000_000)
                .saturating_add(u64::from(duration.subsec_micros()))
        });
    write_guest_u64(
        output,
        RTC_UNIX_EPOCH_TICK_MICROSECONDS.saturating_add(unix_microseconds),
    );
    0
}

const COREDUMP_ERROR_ALREADY_REGISTERED: u32 = 0x8002_0011;
const COREDUMP_ERROR_PARAM: u32 = 0x8002_0016;
/// Largest handler stack the console accepts.
const COREDUMP_MAXIMUM_STACK_SIZE: u64 = 0x2000_0000;

/// Records the guest's coredump handler so a later fault path can run it.
fn coredump_register_handler(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (handler, stack_size) = (arguments[0], arguments[1]);
    if state.coredump_handler.is_some() {
        return sce_kernel_error(COREDUMP_ERROR_ALREADY_REGISTERED);
    }
    if handler == 0 || stack_size >= COREDUMP_MAXIMUM_STACK_SIZE {
        return sce_kernel_error(COREDUMP_ERROR_PARAM);
    }
    state.coredump_handler = Some((handler, stack_size));
    0
}

/// Number of microseconds in the RTC tick epoch before the Unix epoch.
const RTC_UNIX_EPOCH_TICK_MICROSECONDS: u64 = 62_135_596_800_000_000;
const RTC_ERROR_INVALID_POINTER: u32 = 0x80b5_0002;
const RTC_ERROR_INVALID_ARGUMENT: u32 = 0x80b5_0001;

/// Fills an `SceRtcDateTime` with the current wall-clock time in the host's local zone.
///
/// The layout is six 16-bit fields followed by the microsecond remainder. Leaving it untouched is
/// not a benign failure: engines validate the decomposed date and abort the process when every
/// field reads zero, because year zero is not a representable date.
fn rtc_get_current_clock_local_time(output: u64) -> u64 {
    if output == 0 {
        return sce_kernel_error(RTC_ERROR_INVALID_POINTER);
    }
    let now = std::time::SystemTime::now();
    let (unix_seconds, microsecond) = match now.duration_since(std::time::UNIX_EPOCH) {
        Ok(duration) => (
            i64::try_from(duration.as_secs()).unwrap_or(0),
            duration.subsec_micros(),
        ),
        Err(error) => (-i64::try_from(error.duration().as_secs()).unwrap_or(0), 0),
    };
    let (offset_seconds, _) = guest_local_timezone(unix_seconds);
    let local_seconds = unix_seconds.saturating_add(offset_seconds);
    let (year, month, day, hour, minute, second) = civil_from_unix_seconds(local_seconds);
    write_guest_u16(output, year);
    write_guest_u16(output + 2, month);
    write_guest_u16(output + 4, day);
    write_guest_u16(output + 6, hour);
    write_guest_u16(output + 8, minute);
    write_guest_u16(output + 10, second);
    write_guest_u32(output + 12, microsecond);
    0
}

/// Decomposes a Unix timestamp into a proleptic Gregorian civil date and time of day.
///
/// This is Howard Hinnant's `civil_from_days` algorithm: it shifts the era origin to 0000-03-01 so
/// the leap day falls at the end of a 400-year era, which keeps the arithmetic branch-free and
/// correct for dates before 1970.
fn civil_from_unix_seconds(seconds: i64) -> (u16, u16, u16, u16, u16, u16) {
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400);
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    // March is month 0 in the shifted calendar, so months 10 and 11 belong to the next year.
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (
        u16::try_from(year).unwrap_or(0),
        u16::try_from(month).unwrap_or(0),
        u16::try_from(day).unwrap_or(0),
        u16::try_from(seconds_of_day / 3_600).unwrap_or(0),
        u16::try_from(seconds_of_day / 60 % 60).unwrap_or(0),
        u16::try_from(seconds_of_day % 60).unwrap_or(0),
    )
}

fn ime_keyboard_get_resource_ids(arguments: [u64; 6]) -> u64 {
    let output = arguments[1];
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    // The resource array contains the user id followed by five keyboard resource ids. REmu does
    // not expose a separate console keyboard device yet, so return an initialized empty list.
    guest_memset(output, 0, 24);
    write_guest_u32(output, low_u32(arguments[0]));
    0
}

fn ime_keyboard_get_info(output: u64) -> u64 {
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    // SceImeKeyboardInfo is six 32-bit fields plus twelve reserved bytes.
    guest_memset(output, 0, 36);
    0
}

fn relocate_guest_pointer(field: u64) {
    let relative = read_guest_u64(field);
    if relative != 0 {
        let absolute = field.wrapping_add(relative);
        write_guest_u64(field, absolute);
        // Not idempotent: a second sceAgcCreateShader over a reused header re-relocates every
        // pointer through heap garbage, so each mutation earns a journal entry.
        remu_memory::record_guest_write(
            remu_memory::GuestWriteSite::ShaderPatch,
            field,
            8,
            &absolute.to_le_bytes(),
            0,
        );
    }
}

fn patch_agc_shader_program_registers(header: u64, code: u64) -> bool {
    let registers = read_guest_u64(header + AGC_SHADER_SH_REGISTERS_OFFSET);
    let shader_type = read_guest_u8(header + AGC_SHADER_TYPE_OFFSET);
    let register_count = read_guest_u8(header + AGC_SHADER_SH_REGISTER_COUNT_OFFSET);
    // Every rejection here fails sceAgcCreateShader, and the guest treats that as a shader it
    // cannot use. Reporting the reason is the difference between "REmu declined 388 shaders" and
    // a silent rendering gap: the counts below are how a missing draw becomes attributable.
    let reject = |reason: &str, detail: String| {
        static REJECTED: AtomicU64 = AtomicU64::new(0);
        let sequence = REJECTED.fetch_add(1, Ordering::Relaxed) + 1;
        if sequence <= 16 || sequence.is_power_of_two() {
            eprintln!(
                "[REMu][AGC] declined shader registration sequence={sequence} header={header:#x} \
                 type={shader_type} reason={reason} {detail}"
            );
        }
        false
    };
    if registers == 0 || register_count < 2 {
        return reject(
            "unusable-sh-register-block",
            format!("registers={registers:#x} count={register_count}"),
        );
    }
    let expected = match shader_type {
        0 => (0x20c, 0x20d),
        1 => (0x008, 0x009),
        2 | 6 => (0x0c8, 0x0c9),
        4 => (0x08a, 0x08b),
        7 => (0x148, 0x149),
        // Gen5 defines shader stages REmu has no register mapping for yet; a title using one
        // loses those draws entirely, so name the stage rather than dropping it quietly.
        _ => return reject("unmapped-shader-stage", String::new()),
    };
    let observed = (read_guest_u32(registers), read_guest_u32(registers + 8));
    if observed != expected {
        return reject(
            "unexpected-program-registers",
            format!("observed={observed:#x?} expected={expected:#x?}"),
        );
    }
    write_guest_u32(registers + 4, ((code >> 8) & 0xffff_ffff) as u32);
    write_guest_u32(registers + 12, ((code >> 40) & 0xff) as u32);
    // `registers` is derived from a just-relocated header field; if the relocation was wrong,
    // these two stores land at an arbitrary heap address, so both are journaled.
    remu_memory::record_guest_write(
        remu_memory::GuestWriteSite::ShaderPatch,
        registers + 4,
        4,
        &(((code >> 8) & 0xffff_ffff) as u32).to_le_bytes(),
        0,
    );
    remu_memory::record_guest_write(
        remu_memory::GuestWriteSite::ShaderPatch,
        registers + 12,
        4,
        &(((code >> 40) & 0xff) as u32).to_le_bytes(),
        0,
    );
    true
}

thread_local! {
    /// Flips this dispatch queued, paced once the dispatch lock is released. Titles reach the
    /// display through either the video-out call or an AGC flip packet, so the count is kept
    /// where both paths converge.
    static QUEUED_FLIPS: Cell<u32> = const { Cell::new(0) };
    /// Order-visible events emitted while one AGC command buffer is decoded. Capturing a complete
    /// submission before publishing it preserves its resumable PM4 cursor and queue identity at
    /// the backend instead of irreversibly flattening graphics and async-compute work.
    static AGC_GPU_EVENT_CAPTURE: RefCell<Option<Vec<remu_gpu::GpuEvent>>> =
        const { RefCell::new(None) };
    /// GPU events prepared while the process-wide dispatch state is locked. The bounded backend
    /// channel may block, so publication happens only after the state guard has been released.
    static DEFERRED_GPU_PUBLICATION: RefCell<Option<DeferredGpuPublication>> =
        const { RefCell::new(None) };
}

struct DeferredGpuPublication {
    ticket: Option<u64>,
    events: Vec<remu_gpu::GpuEvent>,
}

#[derive(Debug, Default, Eq, PartialEq)]
struct DeferredGpuPublicationSummary {
    outer_events: usize,
    work_items: usize,
    queue_work_items: BTreeMap<u64, usize>,
}

impl DeferredGpuPublication {
    fn summary(&self) -> DeferredGpuPublicationSummary {
        let mut summary = DeferredGpuPublicationSummary {
            outer_events: self.events.len(),
            ..DeferredGpuPublicationSummary::default()
        };
        for event in &self.events {
            match event {
                remu_gpu::GpuEvent::QueueSubmission { queue_id, events } => {
                    summary.work_items = summary.work_items.saturating_add(events.len());
                    let queue = summary.queue_work_items.entry(*queue_id).or_default();
                    *queue = queue.saturating_add(events.len());
                }
                _ => summary.work_items = summary.work_items.saturating_add(1),
            }
        }
        summary
    }
}

static GPU_PUBLICATION_TICKET: AtomicU64 = AtomicU64::new(0);
static GPU_PUBLICATION_ORDER: OnceLock<(Mutex<u64>, Condvar)> = OnceLock::new();
static GPU_PUBLICATION_BARRIER: OnceLock<RwLock<()>> = OnceLock::new();

fn trace_gpu_publication() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_GPU_PUBLICATION").is_some_and(|value| value == "1")
    })
}

fn begin_deferred_gpu_publication() {
    DEFERRED_GPU_PUBLICATION.with_borrow_mut(|publication| {
        assert!(
            publication.is_none(),
            "nested GPU publication capture would lose event ordering"
        );
        *publication = Some(DeferredGpuPublication {
            ticket: None,
            events: Vec::new(),
        });
    });
}

fn take_deferred_gpu_publication() -> DeferredGpuPublication {
    DEFERRED_GPU_PUBLICATION.with_borrow_mut(|publication| {
        publication
            .take()
            .expect("GPU publication capture exists after dispatch")
    })
}

/// Publishes an event immediately unless the current import still owns the dispatch-state lock.
///
/// The first deferred event receives a process-order ticket while that lock is held. A queue-only
/// publication releases the ticket after joining its hardware queue's FIFO, so another queue can
/// publish a producer while this one is backpressured. Host lifecycle and mixed publications keep
/// the ticket through their full barrier.
fn publish_gpu_event(event: remu_gpu::GpuEvent) -> bool {
    let mut event = Some(event);
    let deferred = DEFERRED_GPU_PUBLICATION.with_borrow_mut(|publication| {
        let Some(publication) = publication.as_mut() else {
            return false;
        };
        publication
            .ticket
            .get_or_insert_with(|| GPU_PUBLICATION_TICKET.fetch_add(1, Ordering::Relaxed));
        publication
            .events
            .push(event.take().expect("deferred GPU event exists"));
        true
    });
    deferred || remu_gpu::emit_event(event.expect("immediate GPU event exists"))
}

fn flush_deferred_gpu_publication(publication: DeferredGpuPublication) -> bool {
    flush_deferred_gpu_publication_with_admission(publication, || {})
}

fn flush_deferred_gpu_publication_with_admission(
    publication: DeferredGpuPublication,
    admitted: impl FnOnce(),
) -> bool {
    let Some(ticket) = publication.ticket else {
        debug_assert!(publication.events.is_empty());
        admitted();
        return true;
    };
    let traced = trace_gpu_publication();
    let summary = traced.then(|| publication.summary());
    let started = traced.then(Instant::now);
    if let Some(summary) = &summary {
        eprintln!(
            "[REMu][GPU][PUBLICATION] begin ticket={ticket} outer={} work={} queues={:?} thread={:?} guest={:#x}",
            summary.outer_events,
            summary.work_items,
            summary.queue_work_items,
            std::thread::current().name().unwrap_or("unnamed"),
            current_guest_thread_handle(),
        );
    }
    let order = GPU_PUBLICATION_ORDER.get_or_init(|| (Mutex::new(0), Condvar::new()));
    let mut next = order
        .0
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    while *next != ticket {
        next = order
            .1
            .wait(next)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
    if traced {
        // Paired with `begin` and `end`, this identifies the blocking stage without changing the
        // ordered publisher: begin-without-turn is an order wait, while turn-without-end is
        // backend publication.
        eprintln!("[REMu][GPU][PUBLICATION] turn ticket={ticket}");
    }
    let queue_only = matches!(
        publication.events.as_slice(),
        [remu_gpu::GpuEvent::QueueSubmission { .. }]
    );
    let mut all_sent = true;
    if queue_only {
        let _queue_publication = GPU_PUBLICATION_BARRIER
            .get_or_init(|| RwLock::new(()))
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let event = publication
            .events
            .into_iter()
            .next()
            .expect("queue-only publication owns one event");
        // Assign the event its queue-local sequence while this process-order turn is still held.
        // Once assigned, independent hardware queues may enter the bounded channel concurrently;
        // a later event on this same queue still waits for its FIFO predecessor inside `remu-gpu`.
        all_sent &= remu_gpu::emit_event_with_admission(event, move || {
            debug_assert_eq!(*next, ticket);
            *next = next.saturating_add(1);
            order.1.notify_all();
            admitted();
            if traced {
                eprintln!("[REMu][GPU][PUBLICATION] admitted ticket={ticket} queue-only=true");
            }
        });
    } else {
        // Display lifecycle events and any mixed publication are process-wide barriers. They must
        // neither overtake an active hardware queue nor be overtaken by a later queue.
        let _full_barrier = GPU_PUBLICATION_BARRIER
            .get_or_init(|| RwLock::new(()))
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        admitted();
        for event in publication.events {
            all_sent &= remu_gpu::emit_event(event);
        }
        debug_assert_eq!(*next, ticket);
        *next = next.saturating_add(1);
        order.1.notify_all();
    }
    if let (Some(started), Some(summary)) = (started, summary) {
        eprintln!(
            "[REMu][GPU][PUBLICATION] end ticket={ticket} elapsed_ms={:.3} outer={} work={} queues={:?} success={all_sent}",
            started.elapsed().as_secs_f64() * 1_000.0,
            summary.outer_events,
            summary.work_items,
            summary.queue_work_items,
        );
    }
    all_sent
}

fn capture_agc_gpu_events<R>(operation: impl FnOnce() -> R) -> (R, Vec<remu_gpu::GpuEvent>) {
    AGC_GPU_EVENT_CAPTURE.with_borrow_mut(|capture| {
        assert!(
            capture.is_none(),
            "nested AGC GPU event capture would lose submission ownership"
        );
        *capture = Some(Vec::new());
    });
    let memory_shadow = begin_agc_memory_shadow();
    let result = operation();
    drop(memory_shadow);
    let events = AGC_GPU_EVENT_CAPTURE.with_borrow_mut(|capture| {
        capture
            .take()
            .expect("AGC GPU event capture exists after decoding")
    });
    (result, events)
}

fn emit_agc_gpu_event(event: remu_gpu::GpuEvent) {
    let mut event = Some(event);
    let captured = AGC_GPU_EVENT_CAPTURE.with_borrow_mut(|capture| {
        if let Some(events) = capture.as_mut() {
            events.push(event.take().expect("uncaptured AGC event exists"));
            true
        } else {
            false
        }
    });
    if !captured {
        publish_gpu_event(event.expect("AGC event was not captured"));
    }
}

struct ProcessGuestMemory;

impl GuestMemoryReader for ProcessGuestMemory {
    fn read(&self, address: u64, output: &mut [u8]) -> bool {
        if output.is_empty() {
            return true;
        }
        #[cfg(target_os = "macos")]
        {
            let Ok(length) = u64::try_from(output.len()) else {
                return false;
            };
            let Some(end) = address.checked_add(length) else {
                return false;
            };
            if cached_kernel_memory_readable_range_end(address)
                .is_some_and(|mapped_end| end <= mapped_end)
            {
                // SAFETY: guest mapping mutations are serialized by the process dispatch lock.
                // Shader evaluation runs while its graphics import retains that lock, and the
                // generation-checked range snapshot proves this identity-mapped span readable.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        address as *const u8,
                        output.as_mut_ptr(),
                        output.len(),
                    );
                }
                apply_agc_memory_shadow(address, output);
                return true;
            }
            let mut copied = 0_u64;
            // SAFETY: mach_vm_read_overwrite validates the source mapping in the current task and
            // writes at most `length` bytes to the live output slice.
            let result = unsafe {
                mach_vm_read_overwrite(
                    mach_task_self(),
                    address,
                    length,
                    output.as_mut_ptr() as u64,
                    &raw mut copied,
                )
            };
            let complete = result == 0 && copied == length;
            if complete {
                apply_agc_memory_shadow(address, output);
            }
            complete
        }
        #[cfg(target_os = "linux")]
        {
            let local = libc::iovec {
                iov_base: output.as_mut_ptr().cast(),
                iov_len: output.len(),
            };
            let remote = libc::iovec {
                iov_base: usize::try_from(address)
                    .map_or(std::ptr::null_mut(), |value| value as *mut libc::c_void),
                iov_len: output.len(),
            };
            if remote.iov_base.is_null() {
                return false;
            }
            // SAFETY: both iovecs describe live output storage and a source in this same process;
            // process_vm_readv returns EFAULT instead of dereferencing an unreadable guest range.
            let copied =
                unsafe { libc::process_vm_readv(libc::getpid(), &local, 1, &remote, 1, 0) };
            let complete = copied == isize::try_from(output.len()).unwrap_or(-1);
            if complete {
                apply_agc_memory_shadow(address, output);
            }
            complete
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = address;
            false
        }
    }

    fn readable_prefix_length(&self, address: u64, maximum_length: usize) -> Option<usize> {
        if maximum_length == 0 {
            return Some(0);
        }
        #[cfg(target_os = "macos")]
        {
            let mapped_end = cached_kernel_memory_readable_range_end(address)?;
            usize::try_from(mapped_end.saturating_sub(address))
                .ok()
                .map(|available| available.min(maximum_length))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (address, maximum_length);
            None
        }
    }
}

pub(crate) fn try_read_process_guest_u32(address: u64) -> Option<u32> {
    let mut bytes = [0_u8; 4];
    ProcessGuestMemory
        .read(address, &mut bytes)
        .then(|| u32::from_le_bytes(bytes))
}

pub(crate) fn try_read_process_guest_u64(address: u64) -> Option<u64> {
    let mut bytes = [0_u8; 8];
    ProcessGuestMemory
        .read(address, &mut bytes)
        .then(|| u64::from_le_bytes(bytes))
}

pub(crate) fn process_guest_range_is_readable(address: u64, length: usize) -> bool {
    let mut bytes = vec![0_u8; length];
    ProcessGuestMemory.read(address, &mut bytes)
}

pub(crate) fn process_guest_range_has_read_access(address: u64, length: usize) -> bool {
    match ProcessGuestMemory.readable_prefix_length(address, length) {
        Some(prefix) => prefix == length,
        None => process_guest_range_is_readable(address, length),
    }
}

extern "C" fn guest_locale_virtual_noop(locale: u64) -> u64 {
    locale
}

fn guest_global_locale(state: &mut ProbeDispatchState) -> u64 {
    if state.locale_object != 0 {
        return state.locale_object;
    }

    let vtable = guest_calloc(1, 0x80);
    if vtable == 0 {
        return 0;
    }
    let object = guest_calloc(1, 0x80);
    if object == 0 {
        guest_free(vtable);
        return 0;
    }
    let noop = guest_locale_virtual_noop as *const () as usize as u64;
    for offset in (0_u64..0x60).step_by(8) {
        write_guest_u64(vtable + offset, noop);
    }
    write_guest_u64(object, vtable);
    // Dinkumware's locale implementation begins with a vptr and reference count. The title only
    // invokes the virtual retain/release slots, so a process-lifetime classic locale is sufficient.
    write_guest_u64(object + 8, 1);
    state.locale_vtable = vtable;
    state.locale_object = object;
    object
}

fn run_guest_thread(entry: u64, argument: u64) -> u64 {
    if entry == 0 {
        return 0;
    }
    invoke_guest_entry_with_exit(entry, argument, 0)
}

fn guest_guard_acquire(address: u64) -> u64 {
    if address == 0 {
        return 0;
    }
    // SAFETY: C++ emits a writable eight-byte guard object and passes its live address.
    let value = unsafe { std::ptr::read_unaligned(address as *const u64) };
    if value & 1 != 0 || value & 0xff00 != 0 {
        return 0;
    }
    write_guest_u64(address, 0x100);
    1
}

fn guest_guard_release(address: u64) -> u64 {
    if address != 0 {
        write_guest_u64(address, 1);
    }
    0
}

fn guest_guard_abort(address: u64) -> u64 {
    if address != 0 {
        write_guest_u64(address, 0);
    }
    0
}

fn guest_cxa_allocate_exception(size: u64) -> u64 {
    let Some(allocation_size) = size.checked_add(0x80) else {
        return 0;
    };
    let allocation = guest_calloc(1, allocation_size);
    if allocation == 0 {
        0
    } else {
        allocation.checked_add(0x80).unwrap_or(0)
    }
}

fn guest_cxa_free_exception(exception: u64) -> u64 {
    if let Some(allocation) = exception.checked_sub(0x80) {
        guest_free(allocation);
    }
    0
}

fn guest_cxa_throw(arguments: [u64; 6]) -> u64 {
    let exception = arguments[0];
    let type_info = arguments[1];
    eprintln!(
        "[REMu][CXX] guest exception object={exception:#x} type_info={type_info:#x} destructor={:#x} object_word0={:#x} object_word1={:#x}",
        arguments[2],
        read_guest_u64(exception),
        read_guest_u64(exception + 8),
    );
    terminate_probe(88);
}

fn guest_net_unavailable(error: i32) -> u64 {
    let errno_address = guest_net_errno_address();
    if errno_address != 0 {
        write_guest_u32(errno_address, error.cast_unsigned());
    }
    // REmu runs dumped titles offline. Returning a real libSceNet error lets optional FMOD Live
    // Update/profiling sockets disable themselves without exposing a host listener.
    signed_i32_return(i32::from_ne_bytes(
        (0x8041_0100_u32 | error.cast_unsigned()).to_ne_bytes(),
    ))
}

fn guest_tls_get_addr(descriptor: u64) -> u64 {
    let thread_pointer = current_guest_thread_pointer();
    let Some(descriptor_offset) = descriptor.checked_add(8) else {
        return 0;
    };
    if descriptor == 0 || thread_pointer == 0 {
        return 0;
    }
    let module_id = read_guest_u64(descriptor);
    let offset = read_guest_u64(descriptor_offset);
    let Some(dtv_pointer_address) = thread_pointer.checked_add(8) else {
        return 0;
    };
    let dtv = read_guest_u64(dtv_pointer_address);
    let Some(maximum_module_address) = dtv.checked_add(8) else {
        return 0;
    };
    if dtv == 0 || module_id == 0 {
        return 0;
    }
    let maximum_module_id = read_guest_u64(maximum_module_address);
    if module_id > maximum_module_id {
        return 0;
    }
    let Some(entry_address) =
        tls_dtv_entry_offset(module_id).and_then(|entry_offset| dtv.checked_add(entry_offset))
    else {
        return 0;
    };
    let block = read_guest_u64(entry_address);
    if block == 0 {
        return 0;
    }
    block.checked_add(offset).unwrap_or(0)
}

fn write_guest_u64(address: u64, value: u64) {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe {
        std::ptr::write_unaligned(address as *mut u64, value);
    }
}

fn read_guest_u8(address: u64) -> u8 {
    if agc_memory_shadow_overlaps(address, 1) {
        let mut bytes = [0_u8; 1];
        if ProcessGuestMemory.read(address, &mut bytes) {
            return bytes[0];
        }
    }
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe { std::ptr::read_unaligned(address as *const u8) }
}

fn read_guest_u16(address: u64) -> u16 {
    if agc_memory_shadow_overlaps(address, 2) {
        let mut bytes = [0_u8; 2];
        if ProcessGuestMemory.read(address, &mut bytes) {
            return u16::from_le_bytes(bytes);
        }
    }
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe { std::ptr::read_unaligned(address as *const u16) }
}

fn read_guest_u32(address: u64) -> u32 {
    if agc_memory_shadow_overlaps(address, 4) {
        let mut bytes = [0_u8; 4];
        if ProcessGuestMemory.read(address, &mut bytes) {
            return u32::from_le_bytes(bytes);
        }
    }
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe { std::ptr::read_unaligned(address as *const u32) }
}

fn read_guest_u64(address: u64) -> u64 {
    if agc_memory_shadow_overlaps(address, 8) {
        let mut bytes = [0_u8; 8];
        if ProcessGuestMemory.read(address, &mut bytes) {
            return u64::from_le_bytes(bytes);
        }
    }
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe { std::ptr::read_unaligned(address as *const u64) }
}

fn write_guest_u8(address: u64, value: u8) {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe {
        std::ptr::write_unaligned(address as *mut u8, value);
    }
}

fn write_guest_u32(address: u64, value: u32) {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe {
        std::ptr::write_unaligned(address as *mut u32, value);
    }
}

fn write_guest_u16(address: u64, value: u16) {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe {
        std::ptr::write_unaligned(address as *mut u16, value);
    }
}

fn write_output_u64(address: u64, value: u64) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    write_guest_u64(address, value);
    0
}

fn write_output_i32(address: u64, value: i32) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    // SAFETY: The HLE ABI identifies `address` as a writable 32-bit output field.
    unsafe {
        std::ptr::write_unaligned(address as *mut i32, value);
    }
    0
}

fn trace_guest_vm() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("REMU_TRACE_VM").is_some_and(|value| value == "1"))
}

fn trace_guest_vfs() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("REMU_TRACE_VFS").is_some_and(|value| value == "1"))
}

fn trace_guest_vfs_miss() -> Option<u64> {
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1;
    (trace_guest_vfs() || trace_logarithmic_sequence(sequence, 32)).then_some(sequence)
}

fn trace_agc_shaders() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_AGC_SHADERS").is_some_and(|value| value == "1")
    })
}

fn trace_agc_target_address() -> Option<u64> {
    static ADDRESS: OnceLock<Option<u64>> = OnceLock::new();
    *ADDRESS.get_or_init(|| {
        let value = std::env::var("REMU_TRACE_AGC_TARGET").ok()?;
        let value = value
            .strip_prefix("0x")
            .or_else(|| value.strip_prefix("0X"))
            .unwrap_or(&value);
        u64::from_str_radix(value, 16).ok()
    })
}

fn trace_agc_draws() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var_os("REMU_TRACE_AGC_DRAWS").is_some_and(|value| value == "1"))
}

fn trace_agc_registers() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_AGC_REGISTERS").is_some_and(|value| value == "1")
    })
}

fn trace_agc_programs() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_AGC_PROGRAMS").is_some_and(|value| value == "1")
    })
}

fn trace_guest_time() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("REMU_TRACE_TIME").is_some_and(|value| value == "1"))
}

fn trace_fiber() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("REMU_TRACE_FIBER").is_some_and(|value| value == "1"))
}

fn trace_ampr_reads() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("REMU_TRACE_AMPR").is_some_and(|value| value == "1"))
}

fn trace_guest_sync() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("REMU_TRACE_SYNC").is_some_and(|value| value == "1"))
}

fn sanitizer_new_replace_external() -> u64 {
    const TLS_NEW_REPLACE_OFFSET: u64 = 0x300;
    const NEW_REPLACE_SIZE: usize = 0x68;
    let thread_pointer = current_guest_thread_pointer();
    if thread_pointer == 0 {
        return 0;
    }
    let address = thread_pointer + TLS_NEW_REPLACE_OFFSET;
    // SAFETY: every guest thread reserves a writable TCB region after its thread pointer, and this
    // kernel-owned compatibility structure fits entirely within that reservation.
    unsafe {
        std::ptr::write_bytes(address as *mut u8, 0, NEW_REPLACE_SIZE);
    }
    write_guest_u64(address, NEW_REPLACE_SIZE as u64);
    address
}

fn libc_heap_get_trace_info(info_address: u64, state: &mut ProbeDispatchState) -> u64 {
    const HEAP_TRACE_INFO_SIZE: u64 = 32;
    const HEAP_TRACE_STORAGE_WORDS: usize = 65;
    if info_address == 0 || read_guest_u64(info_address) != HEAP_TRACE_INFO_SIZE {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    if state.heap_trace_storage.len() != HEAP_TRACE_STORAGE_WORDS {
        state.heap_trace_storage.resize(HEAP_TRACE_STORAGE_WORDS, 0);
    }
    let storage_address = state.heap_trace_storage.as_mut_ptr() as usize as u64;
    write_guest_u64(info_address + 16, storage_address);
    write_guest_u64(info_address + 24, storage_address + 8);
    0
}

fn json_init_parameter_set_allocator(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (parameter, allocator, user_data) = (arguments[0], arguments[1], arguments[2]);
    if parameter == 0 || allocator == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state
        .json_runtime
        .allocators
        .insert(parameter, (allocator, user_data));
    write_guest_u64(parameter, allocator);
    write_guest_u64(parameter + 8, user_data);
    0
}

fn json_init_parameter_set_file_buffer_size(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let (parameter, size) = (arguments[0], arguments[1]);
    if parameter == 0 || size == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state.json_runtime.file_buffer_sizes.insert(parameter, size);
    write_guest_u64(parameter + 16, size);
    0
}

fn json_initializer_initialize(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let (initializer, parameter) = (arguments[0], arguments[1]);
    if initializer == 0 || parameter == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    state
        .json_runtime
        .initialized_objects
        .insert(initializer, parameter);
    0
}

fn json_initializer_terminate(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if state
        .json_runtime
        .initialized_objects
        .remove(&arguments[0])
        .is_some()
    {
        0
    } else {
        sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT)
    }
}

fn guest_malloc(size: u64) -> u64 {
    let Ok(size) = usize::try_from(size) else {
        return 0;
    };
    // SAFETY: The returned allocation is process-local and uses the same x86-64 pointer ABI.
    unsafe { host_malloc(size) as usize as u64 }
}

fn guest_free(address: u64) -> u64 {
    if address != 0 {
        // SAFETY: The guest passes an allocation previously returned by this libc HLE family.
        unsafe {
            host_free(address as *mut std::ffi::c_void);
        }
    }
    0
}

fn guest_calloc(count: u64, size: u64) -> u64 {
    let (Ok(count), Ok(size)) = (usize::try_from(count), usize::try_from(size)) else {
        return 0;
    };
    // SAFETY: libc validates the allocation-size multiplication and returns process-local memory.
    unsafe { host_calloc(count, size) as usize as u64 }
}

fn guest_realloc(address: u64, size: u64) -> u64 {
    let Ok(size) = usize::try_from(size) else {
        return 0;
    };
    // SAFETY: `address` is null or belongs to the same libc allocation family.
    unsafe { host_realloc(address as *mut std::ffi::c_void, size) as usize as u64 }
}

fn guest_aligned_alloc(alignment: u64, size: u64) -> u64 {
    let (Ok(mut alignment), Ok(size)) = (usize::try_from(alignment), usize::try_from(size)) else {
        return 0;
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return 0;
    }
    // Guest memalign accepts sub-pointer alignments; the host allocator requires at least
    // sizeof(void*), which still satisfies the guest's weaker request.
    alignment = alignment.max(std::mem::size_of::<usize>());
    unsafe { host_aligned_malloc(alignment, size) as usize as u64 }
}

fn guest_posix_memalign(output: u64, alignment: u64, size: u64) -> u64 {
    if output == 0 {
        return libc::EINVAL as u64;
    }
    let (Ok(mut alignment), Ok(size)) = (usize::try_from(alignment), usize::try_from(size)) else {
        return libc::ENOMEM as u64;
    };
    if alignment == 0 || !alignment.is_power_of_two() {
        return libc::EINVAL as u64;
    }
    alignment = alignment.max(std::mem::size_of::<usize>());
    let allocation = unsafe { host_aligned_malloc(alignment, size) };
    if allocation.is_null() {
        return libc::ENOMEM as u64;
    }
    write_guest_u64(output, allocation as usize as u64);
    0
}

#[cfg(unix)]
unsafe fn host_malloc(size: usize) -> *mut std::ffi::c_void {
    // SAFETY: size is representable; libc owns the returned block.
    unsafe { libc::malloc(size) }
}

#[cfg(unix)]
unsafe fn host_free(pointer: *mut std::ffi::c_void) {
    // SAFETY: pointer is null or from this host malloc family.
    unsafe { libc::free(pointer) }
}

#[cfg(unix)]
unsafe fn host_calloc(count: usize, size: usize) -> *mut std::ffi::c_void {
    // SAFETY: libc validates the multiplication.
    unsafe { libc::calloc(count, size) }
}

#[cfg(unix)]
unsafe fn host_realloc(pointer: *mut std::ffi::c_void, size: usize) -> *mut std::ffi::c_void {
    // SAFETY: pointer is null or from this host malloc family.
    unsafe { libc::realloc(pointer, size) }
}

#[cfg(unix)]
unsafe fn host_aligned_malloc(alignment: usize, size: usize) -> *mut std::ffi::c_void {
    let mut pointer = std::ptr::null_mut();
    // SAFETY: pointer is writable; alignment/size are host-representable.
    let result = unsafe { libc::posix_memalign(&raw mut pointer, alignment, size.max(1)) };
    if result == 0 { pointer } else { std::ptr::null_mut() }
}

#[cfg(windows)]
const WINDOWS_HEAP_ALIGNMENT: usize = 16;

#[cfg(windows)]
unsafe fn host_malloc(size: usize) -> *mut std::ffi::c_void {
    // SAFETY: MSVC aligned heap; paired with `aligned_free` in `host_free`.
    unsafe { libc::aligned_malloc(size.max(1), WINDOWS_HEAP_ALIGNMENT) }
}

#[cfg(windows)]
unsafe fn host_free(pointer: *mut std::ffi::c_void) {
    // SAFETY: pointer is null or from `aligned_malloc` / `aligned_realloc`.
    unsafe { libc::aligned_free(pointer) }
}

#[cfg(windows)]
unsafe fn host_calloc(count: usize, size: usize) -> *mut std::ffi::c_void {
    let Some(total) = count.checked_mul(size) else {
        return std::ptr::null_mut();
    };
    // SAFETY: paired with `aligned_free`.
    let pointer = unsafe { libc::aligned_malloc(total.max(1), WINDOWS_HEAP_ALIGNMENT) };
    if !pointer.is_null() {
        // SAFETY: freshly allocated writable block of `total` bytes.
        unsafe {
            std::ptr::write_bytes(pointer.cast::<u8>(), 0, total);
        }
    }
    pointer
}

#[cfg(windows)]
unsafe fn host_realloc(pointer: *mut std::ffi::c_void, size: usize) -> *mut std::ffi::c_void {
    // SAFETY: pointer is null or from the Windows aligned heap family.
    unsafe { libc::aligned_realloc(pointer, size.max(1), WINDOWS_HEAP_ALIGNMENT) }
}

#[cfg(windows)]
unsafe fn host_aligned_malloc(alignment: usize, size: usize) -> *mut std::ffi::c_void {
    // SAFETY: MSVC aligned heap; alignment is a power of two.
    unsafe { libc::aligned_malloc(size.max(1), alignment.max(WINDOWS_HEAP_ALIGNMENT)) }
}

fn guest_strlen(address: u64) -> u64 {
    if address == 0 {
        return 0;
    }
    // SAFETY: The guest libc call contract supplies a valid NUL-terminated string.
    unsafe { libc::strlen(address as *const libc::c_char) as u64 }
}

fn guest_strcmp(left: u64, right: u64) -> u64 {
    if left == 0 || right == 0 {
        return u64::MAX;
    }
    // SAFETY: strcmp's guest ABI contract supplies two NUL-terminated strings.
    let result = unsafe { libc::strcmp(left as *const libc::c_char, right as *const libc::c_char) };
    signed_i32_return(result)
}

fn guest_strncmp(left: u64, right: u64, length: u64) -> u64 {
    if left == 0 || right == 0 {
        return u64::MAX;
    }
    let Ok(length) = usize::try_from(length) else {
        return u64::MAX;
    };
    // SAFETY: strncmp's guest ABI contract supplies readable ranges through `length`.
    let result = unsafe {
        libc::strncmp(
            left as *const libc::c_char,
            right as *const libc::c_char,
            length,
        )
    };
    signed_i32_return(result)
}

fn guest_strcasecmp(left: u64, right: u64) -> u64 {
    if left == 0 || right == 0 {
        return u64::from(left != right);
    }
    // SAFETY: strcasecmp's guest ABI contract supplies two NUL-terminated strings.
    let result = unsafe { host_strcasecmp(left as *const libc::c_char, right as *const libc::c_char) };
    signed_i32_return(result)
}

#[cfg(unix)]
unsafe fn host_strcasecmp(left: *const libc::c_char, right: *const libc::c_char) -> i32 {
    // SAFETY: caller supplies two NUL-terminated C strings.
    unsafe { libc::strcasecmp(left, right) }
}

#[cfg(windows)]
unsafe fn host_strcasecmp(left: *const libc::c_char, right: *const libc::c_char) -> i32 {
    // SAFETY: caller supplies two NUL-terminated C strings; MSVC exports `_stricmp`.
    unsafe { libc::stricmp(left, right) }
}

#[cfg(not(any(unix, windows)))]
unsafe fn host_strcasecmp(left: *const libc::c_char, right: *const libc::c_char) -> i32 {
    // SAFETY: caller supplies two NUL-terminated C strings.
    unsafe {
        let mut left = left;
        let mut right = right;
        loop {
            let left_ch = (*left as u8 as char).to_ascii_lowercase() as i32;
            let right_ch = (*right as u8 as char).to_ascii_lowercase() as i32;
            if left_ch != right_ch || left_ch == 0 {
                return left_ch - right_ch;
            }
            left = left.add(1);
            right = right.add(1);
        }
    }
}

fn guest_strstr(haystack: u64, needle: u64) -> u64 {
    if haystack == 0 || needle == 0 {
        return 0;
    }
    // SAFETY: strstr's guest ABI contract supplies two NUL-terminated strings. Its result points
    // into the unchanged haystack, which is already process-addressable guest memory.
    unsafe {
        libc::strstr(
            haystack as *const libc::c_char,
            needle as *const libc::c_char,
        ) as usize as u64
    }
}

fn guest_sincosf(value: f32, sin_address: u64, cos_address: u64) -> u64 {
    let (sin, cos) = value.sin_cos();
    if sin_address != 0 {
        write_guest_u32(sin_address, sin.to_bits());
    }
    if cos_address != 0 {
        write_guest_u32(cos_address, cos.to_bits());
    }
    0
}

fn guest_qsort(arguments: [u64; 6]) -> u64 {
    let (base, count, size, comparator) = (arguments[0], arguments[1], arguments[2], arguments[3]);
    let (Ok(count), Ok(size), Ok(comparator)) = (
        usize::try_from(count),
        usize::try_from(size),
        usize::try_from(comparator),
    ) else {
        return 0;
    };
    if base == 0
        || comparator == 0
        || count < 2
        || size == 0
        || count > 1_000_000
        || size > 1_048_576
    {
        return 0;
    }
    let Ok(base) = usize::try_from(base) else {
        return 0;
    };
    // The guest comparator is native x86-64 code in this Rosetta process. Dropping the dispatcher
    // lock before entering this function permits a comparator that itself crosses an import stub.
    let compare: unsafe extern "C" fn(*const u8, *const u8) -> i32 =
        unsafe { std::mem::transmute(comparator) };
    let mut key = vec![0_u8; size];
    for index in 1..count {
        // SAFETY: qsort's ABI guarantees a writable `count * size` byte range at `base`.
        unsafe {
            let current = (base + index * size) as *mut u8;
            std::ptr::copy_nonoverlapping(current, key.as_mut_ptr(), size);
            let mut insertion = index;
            while insertion > 0 {
                let previous = (base + (insertion - 1) * size) as *mut u8;
                if compare(previous, key.as_ptr()) <= 0 {
                    break;
                }
                std::ptr::copy(previous, previous.add(size), size);
                insertion -= 1;
            }
            std::ptr::copy_nonoverlapping(key.as_ptr(), (base + insertion * size) as *mut u8, size);
        }
    }
    0
}

fn read_guest_cstring(address: u64) -> Option<String> {
    if address == 0 {
        return None;
    }
    // SAFETY: Guest libc string arguments are NUL-terminated within live guest memory.
    let string = unsafe { std::ffi::CStr::from_ptr(address as *const libc::c_char) };
    string.to_str().ok().map(ToOwned::to_owned)
}

const SCE_KERNEL_ERROR_OPERATION_NOT_PERMITTED: u32 = 0x8002_0001;
const SCE_KERNEL_ERROR_PERMISSION_DENIED: u32 = 0x8002_000d;
const SCE_KERNEL_ERROR_NOT_FOUND: u32 = 0x8002_0002;
const SCE_KERNEL_ERROR_INTERRUPTED: u32 = 0x8002_0004;
const SCE_KERNEL_ERROR_IO: u32 = 0x8002_0005;
const SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR: u32 = 0x8002_0009;
const SCE_KERNEL_ERROR_OUT_OF_MEMORY: u32 = 0x8002_000c;
const SCE_KERNEL_ERROR_MEMORY_FAULT: u32 = 0x8002_000e;
const SCE_KERNEL_ERROR_BUSY: u32 = 0x8002_0010;
const SCE_KERNEL_ERROR_ALREADY_EXISTS: u32 = 0x8002_0011;
const SCE_KERNEL_ERROR_CROSS_DEVICE: u32 = 0x8002_0012;
const SCE_KERNEL_ERROR_NOT_DIRECTORY: u32 = 0x8002_0014;
const SCE_KERNEL_ERROR_IS_DIRECTORY: u32 = 0x8002_0015;
const SCE_KERNEL_ERROR_INVALID_ARGUMENT: u32 = 0x8002_0016;
const SCE_KERNEL_ERROR_FILE_TOO_LARGE: u32 = 0x8002_001b;
const SCE_KERNEL_ERROR_NO_SPACE: u32 = 0x8002_001c;
const SCE_KERNEL_ERROR_READ_ONLY: u32 = 0x8002_001e;
const SCE_KERNEL_ERROR_BROKEN_PIPE: u32 = 0x8002_0020;
const SCE_KERNEL_ERROR_TRY_AGAIN: u32 = 0x8002_0023;
const SCE_KERNEL_ERROR_NOT_SUPPORTED: u32 = 0x8002_002d;
const SCE_KERNEL_ERROR_NAME_TOO_LONG: u32 = 0x8002_003f;
const SCE_KERNEL_ERROR_TIMED_OUT: u32 = 0x8002_003c;
const SCE_KERNEL_ERROR_NOT_IMPLEMENTED: u32 = 0x8002_004e;
const SCE_KERNEL_STAT_SIZE: usize = 120;

fn sce_kernel_error(code: u32) -> u64 {
    signed_i32_return(i32::from_ne_bytes(code.to_ne_bytes()))
}

fn guest_posix_file_result(result: u64) -> u64 {
    let code = low_u32(result);
    if code & 0x8000_0000 == 0 {
        return result;
    }
    let errno = if code & 0xffff_ff00 == 0x8002_0000 {
        i32::try_from(code & 0xff).expect("kernel errno fits i32")
    } else {
        libc::EIO
    };
    let errno_address = guest_errno_address();
    if errno_address != 0 {
        write_guest_u32(errno_address, errno.cast_unsigned());
    }
    u64::MAX
}

fn sce_kernel_io_error(error: &std::io::Error) -> u64 {
    use std::io::ErrorKind;

    let code = match error.kind() {
        ErrorKind::NotFound => SCE_KERNEL_ERROR_NOT_FOUND,
        ErrorKind::PermissionDenied => SCE_KERNEL_ERROR_PERMISSION_DENIED,
        ErrorKind::BrokenPipe => SCE_KERNEL_ERROR_BROKEN_PIPE,
        ErrorKind::AlreadyExists => SCE_KERNEL_ERROR_ALREADY_EXISTS,
        ErrorKind::WouldBlock => SCE_KERNEL_ERROR_TRY_AGAIN,
        ErrorKind::InvalidInput | ErrorKind::InvalidData => SCE_KERNEL_ERROR_INVALID_ARGUMENT,
        ErrorKind::TimedOut => 0x8002_003c,
        ErrorKind::Interrupted => SCE_KERNEL_ERROR_INTERRUPTED,
        ErrorKind::Unsupported => SCE_KERNEL_ERROR_NOT_SUPPORTED,
        ErrorKind::OutOfMemory => SCE_KERNEL_ERROR_OUT_OF_MEMORY,
        ErrorKind::NotADirectory => SCE_KERNEL_ERROR_NOT_DIRECTORY,
        ErrorKind::IsADirectory => SCE_KERNEL_ERROR_IS_DIRECTORY,
        ErrorKind::DirectoryNotEmpty => 0x8002_0042,
        ErrorKind::ReadOnlyFilesystem => SCE_KERNEL_ERROR_READ_ONLY,
        ErrorKind::StaleNetworkFileHandle => 0x8002_0046,
        ErrorKind::StorageFull | ErrorKind::QuotaExceeded => SCE_KERNEL_ERROR_NO_SPACE,
        ErrorKind::FileTooLarge => SCE_KERNEL_ERROR_FILE_TOO_LARGE,
        ErrorKind::ResourceBusy | ErrorKind::ExecutableFileBusy => SCE_KERNEL_ERROR_BUSY,
        ErrorKind::Deadlock => 0x8002_000b,
        ErrorKind::CrossesDevices => SCE_KERNEL_ERROR_CROSS_DEVICE,
        ErrorKind::TooManyLinks => 0x8002_001f,
        ErrorKind::InvalidFilename => 0x8002_003f,
        ErrorKind::ArgumentListTooLong => 0x8002_0007,
        _ => SCE_KERNEL_ERROR_IO,
    };
    sce_kernel_error(code)
}

fn audio_out_error(error: audio::AudioOutError) -> u64 {
    let code = match error {
        audio::AudioOutError::NotOpened => 0x8026_0001,
        audio::AudioOutError::Busy => 0x8026_0002,
        audio::AudioOutError::InvalidPort => 0x8026_0003,
        audio::AudioOutError::InvalidPointer => 0x8026_0004,
        audio::AudioOutError::PortFull => 0x8026_0005,
        audio::AudioOutError::InvalidSize => 0x8026_0006,
        audio::AudioOutError::InvalidFormat => 0x8026_0007,
        audio::AudioOutError::InvalidSampleFrequency => 0x8026_0008,
        audio::AudioOutError::InvalidVolume => 0x8026_0009,
        audio::AudioOutError::InvalidPortType => 0x8026_000a,
        audio::AudioOutError::OutOfMemory => 0x8026_000d,
    };
    sce_kernel_error(code)
}

fn guest_memcmp(left: u64, right: u64, length: u64) -> u64 {
    if length == 0 {
        return 0;
    }
    if left == 0 || right == 0 {
        return u64::MAX;
    }
    let Ok(length) = usize::try_from(length) else {
        return u64::MAX;
    };
    // SAFETY: memcmp's guest ABI contract supplies readable ranges through `length`.
    signed_i32_return(unsafe {
        libc::memcmp(
            left as *const libc::c_void,
            right as *const libc::c_void,
            length,
        )
    })
}

fn guest_memchr(address: u64, character: u64, length: u64) -> u64 {
    if address == 0 || length == 0 {
        return 0;
    }
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    // SAFETY: memchr's guest ABI contract supplies a readable range through `length`.
    unsafe {
        libc::memchr(address as *const libc::c_void, low_i32(character), length) as usize as u64
    }
}

const GUEST_WCHAR_SIZE: usize = std::mem::size_of::<u16>();

fn guest_wide_byte_length(count: u64) -> Option<usize> {
    usize::try_from(count).ok()?.checked_mul(GUEST_WCHAR_SIZE)
}

fn guest_wmemcpy(destination: u64, source: u64, count: u64) -> u64 {
    guest_wide_copy(destination, source, count)
}

fn guest_wmemmove(destination: u64, source: u64, count: u64) -> u64 {
    guest_wide_copy(destination, source, count)
}

fn guest_wide_copy(destination: u64, source: u64, count: u64) -> u64 {
    let Some(length) = guest_wide_byte_length(count) else {
        return 0;
    };
    if length == 0 {
        return destination;
    }
    if destination == 0 || source == 0 {
        return 0;
    }
    // SAFETY: the guest ABI supplies readable and writable ranges for `count` two-byte wide
    // characters. Gen5's libc copy is overlap-tolerant in the basic_string path used by this
    // title, so both wide-copy entry points intentionally use `copy`.
    unsafe {
        std::ptr::copy(source as *const u8, destination as *mut u8, length);
    }
    destination
}

fn guest_wmemchr(address: u64, character: u64, count: u64) -> u64 {
    if address == 0 || count == 0 {
        return 0;
    }
    let Ok(count) = usize::try_from(count) else {
        return 0;
    };
    let needle =
        u16::try_from(character & u64::from(u16::MAX)).expect("masked wide character fits u16");
    for index in 0..count {
        let offset = index * GUEST_WCHAR_SIZE;
        // SAFETY: wmemchr's guest ABI supplies `count` readable two-byte elements. Unaligned
        // reads keep the HLE correct even when a guest passes packed storage.
        let value: u16 =
            unsafe { std::ptr::read_unaligned((address as *const u8).add(offset).cast()) };
        if value == needle {
            return address + u64::try_from(offset).unwrap_or(u64::MAX);
        }
    }
    0
}

fn guest_wmemcmp(left: u64, right: u64, count: u64) -> u64 {
    if count == 0 {
        return 0;
    }
    if left == 0 || right == 0 {
        return u64::MAX;
    }
    let Ok(count) = usize::try_from(count) else {
        return u64::MAX;
    };
    for index in 0..count {
        let offset = index * GUEST_WCHAR_SIZE;
        // SAFETY: wmemcmp's guest ABI supplies `count` readable two-byte elements on each side.
        let left_value =
            unsafe { std::ptr::read_unaligned((left as *const u8).add(offset).cast::<u16>()) };
        // SAFETY: same contract as the left-side read above.
        let right_value =
            unsafe { std::ptr::read_unaligned((right as *const u8).add(offset).cast::<u16>()) };
        match left_value.cmp(&right_value) {
            std::cmp::Ordering::Less => return signed_i32_return(-1),
            std::cmp::Ordering::Greater => return signed_i32_return(1),
            std::cmp::Ordering::Equal => {}
        }
    }
    0
}

fn guest_sprintf(arguments: [u64; 6]) -> u64 {
    if arguments[0] == 0 || arguments[1] == 0 {
        return u64::MAX;
    }
    // SAFETY: The guest supplies a writable destination, a valid format string, and SysV integer
    // or pointer varargs. Passing all four available argument registers preserves their bit layout.
    signed_i32_return(unsafe {
        libc::sprintf(
            arguments[0] as *mut libc::c_char,
            arguments[1] as *const libc::c_char,
            arguments[2],
            arguments[3],
            arguments[4],
            arguments[5],
        )
    })
}

fn guest_strcpy(destination: u64, source: u64) -> u64 {
    if destination == 0 || source == 0 {
        return 0;
    }
    // SAFETY: strcpy's guest ABI contract supplies a large-enough destination and C string source.
    unsafe {
        libc::strcpy(
            destination as *mut libc::c_char,
            source as *const libc::c_char,
        );
    }
    destination
}

fn guest_strncpy(destination: u64, source: u64, length: u64) -> u64 {
    if destination == 0 || source == 0 {
        return 0;
    }
    let Ok(length) = usize::try_from(length) else {
        return 0;
    };
    // SAFETY: strncpy's guest ABI contract supplies readable/writable ranges through `length`.
    unsafe {
        libc::strncpy(
            destination as *mut libc::c_char,
            source as *const libc::c_char,
            length,
        );
    }
    destination
}

fn guest_strcat(destination: u64, source: u64) -> u64 {
    if destination == 0 || source == 0 {
        return 0;
    }
    // SAFETY: strcat's guest ABI contract supplies a large-enough destination and C string source.
    unsafe {
        libc::strcat(
            destination as *mut libc::c_char,
            source as *const libc::c_char,
        );
    }
    destination
}

fn signed_i32_return(value: i32) -> u64 {
    u64::from_ne_bytes(i64::from(value).to_ne_bytes())
}

fn guest_usleep(microseconds: u64) -> u64 {
    let bounded = microseconds.min(u64::from(u32::MAX));
    std::thread::sleep(std::time::Duration::from_micros(bounded));
    0
}

fn guest_clock_gettime(clock_id: u64, output: u64) -> u64 {
    if output == 0 {
        return u64::MAX;
    }
    // Console clock ids do not share Darwin's numeric constants. This POSIX compatibility export
    // returns wall-clock time, matching the console runtime's observable calendar-time contract.
    let elapsed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let seconds = elapsed.as_secs();
    let nanoseconds = u64::from(elapsed.subsec_nanos());
    write_guest_u64(output, seconds);
    write_guest_u64(output + 8, nanoseconds);
    if trace_guest_time() {
        eprintln!(
            "[REMu][TIME] clock_gettime id={} output={output:#x} status=0 sec={} nsec={}",
            clock_id, seconds, nanoseconds
        );
    }
    0
}

fn guest_kernel_gettimeofday(output: u64) -> u64 {
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    write_guest_timeval(output);
    0
}

fn guest_posix_gettimeofday(output: u64) -> u64 {
    if output == 0 {
        return u64::MAX;
    }
    write_guest_timeval(output);
    0
}

fn write_guest_timeval(output: u64) {
    let elapsed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    write_guest_u64(output, elapsed.as_secs());
    write_guest_u64(output + 8, u64::from(elapsed.subsec_micros()));
}

/// Returns the host's UTC offset and daylight-saving component for a Unix timestamp. Console
/// calendar conversions describe the machine's configured civil time zone, so they must not be
/// reduced to UTC-only stubs even though emulated monotonic clocks remain host-independent.
fn guest_local_timezone(unix_seconds: i64) -> (i64, i32) {
    #[cfg(unix)]
    {
        let host_time: libc::time_t = unix_seconds;
        // SAFETY: `local` is writable and `host_time` is a valid scalar time_t value.
        let mut local: libc::tm = unsafe { std::mem::zeroed() };
        let result = unsafe { libc::localtime_r(&raw const host_time, &raw mut local) };
        if result.is_null() {
            return (0, 0);
        }
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
        let offset = local.tm_gmtoff;
        #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "freebsd")))]
        let offset = 0;
        let daylight_seconds = if local.tm_isdst > 0 { 3_600 } else { 0 };
        (offset, daylight_seconds)
    }
    #[cfg(windows)]
    {
        let host_time: i64 = unix_seconds;
        // SAFETY: `local` is writable; UCRT's `_localtime64_s` fills it for a valid time_t.
        let mut local: libc::tm = unsafe { std::mem::zeroed() };
        let status = unsafe { _localtime64_s(&raw mut local, &raw const host_time) };
        if status != 0 {
            return (0, 0);
        }
        let daylight_seconds = if local.tm_isdst > 0 { 3_600 } else { 0 };
        (0, daylight_seconds)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = unix_seconds;
        (0, 0)
    }
}

fn guest_kernel_convert_localtime_to_utc(arguments: [u64; 6]) -> u64 {
    let local_seconds = i64::from_ne_bytes(arguments[0].to_ne_bytes());
    let utc_output = arguments[2];
    let timezone_output = arguments[3];
    let dst_output = arguments[4];
    if timezone_output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let (offset_seconds, daylight_seconds) = guest_local_timezone(local_seconds);
    let utc_seconds = local_seconds.saturating_sub(offset_seconds);
    let minutes_west = i32::try_from(-offset_seconds / 60).unwrap_or(0);
    write_guest_u32(
        timezone_output,
        u32::from_ne_bytes(minutes_west.to_ne_bytes()),
    );
    write_guest_u32(
        timezone_output + 4,
        u32::from_ne_bytes((daylight_seconds / 60).to_ne_bytes()),
    );
    if utc_output != 0 {
        write_guest_u64(utc_output, u64::from_ne_bytes(utc_seconds.to_ne_bytes()));
    }
    if dst_output != 0 {
        write_guest_u32(
            dst_output,
            u32::from_ne_bytes(daylight_seconds.to_ne_bytes()),
        );
    }
    0
}

fn guest_kernel_convert_utc_to_localtime(arguments: [u64; 6]) -> u64 {
    let utc_seconds = i64::from_ne_bytes(arguments[0].to_ne_bytes());
    let local_output = arguments[1];
    let timesec_output = arguments[2];
    let dst_output = arguments[3];
    if local_output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let (offset_seconds, daylight_seconds) = guest_local_timezone(utc_seconds);
    let local_seconds = utc_seconds.saturating_add(offset_seconds);
    write_guest_u64(
        local_output,
        u64::from_ne_bytes(local_seconds.to_ne_bytes()),
    );
    if timesec_output != 0 {
        write_guest_u64(
            timesec_output,
            u64::from_ne_bytes(utc_seconds.to_ne_bytes()),
        );
        write_guest_u32(
            timesec_output + 8,
            u32::try_from(offset_seconds).unwrap_or(0),
        );
        write_guest_u32(
            timesec_output + 12,
            u32::try_from(daylight_seconds).unwrap_or(0),
        );
    }
    if dst_output != 0 {
        write_guest_u64(dst_output, u64::try_from(daylight_seconds).unwrap_or(0));
    }
    0
}

fn guest_time(output: u64) -> u64 {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    if output != 0 {
        write_guest_u64(output, seconds);
    }
    if trace_guest_time() {
        eprintln!("[REMu][TIME] time output={output:#x} sec={seconds}");
    }
    seconds
}

fn guest_fputs(address: u64) -> u64 {
    if address == 0 {
        return u64::MAX;
    }
    // SAFETY: The guest libc call contract supplies a valid NUL-terminated string.
    let text = unsafe { std::ffi::CStr::from_ptr(address as *const libc::c_char) };
    eprint!("{}", text.to_string_lossy());
    0
}

const fn align_up_u64(value: u64, alignment: u64) -> Option<u64> {
    let mask = alignment - 1;
    match value.checked_add(mask) {
        Some(value) => Some(value & !mask),
        None => None,
    }
}

/// Raises the process descriptor limit to the hard maximum and reports the budget once.
///
/// launchd hands a 256-descriptor soft limit to the processes it starts, while the APR handle
/// cache retains one open descriptor per file a title has ever read. The budget therefore has to
/// be claimed before the guest starts opening files: past exhaustion every open fails, and the
/// titles that read through APR parse the untouched destination instead.
#[cfg(all(unix, target_arch = "x86_64"))]
fn raise_open_file_budget_once() {
    // macOS refuses RLIM_INFINITY for RLIMIT_NOFILE; the granted ceiling is kern.maxfilesperproc,
    // which OPEN_MAX bounds. Without this fallback an unlimited hard limit leaves the soft limit
    // exactly where it was.
    const DESCRIPTOR_CEILING: libc::rlim_t = 10240;
    static BUDGET: OnceLock<libc::rlim_t> = OnceLock::new();
    BUDGET.get_or_init(|| {
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: getrlimit writes the named resource's limits into the caller's storage.
        if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &raw mut limit) } != 0 {
            eprintln!(
                "[REMu] warning: file descriptor budget unavailable: {}",
                std::io::Error::last_os_error()
            );
            return 0;
        }
        for target in [limit.rlim_max, DESCRIPTOR_CEILING] {
            if target <= limit.rlim_cur {
                continue;
            }
            let raised = libc::rlimit {
                rlim_cur: target,
                rlim_max: limit.rlim_max,
            };
            // SAFETY: setrlimit reads the limits from the caller's storage.
            if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &raw const raised) } == 0 {
                limit.rlim_cur = target;
                break;
            }
        }
        let hard = if limit.rlim_max == libc::RLIM_INFINITY {
            "unlimited".to_owned()
        } else {
            limit.rlim_max.to_string()
        };
        eprintln!(
            "[REMu] file descriptor budget: {} open files (hard limit {hard})",
            limit.rlim_cur
        );
        limit.rlim_cur
    });
}

#[cfg(target_arch = "x86_64")]
fn probe_entry(
    image: &LinkedImage,
    process_image_name: &str,
    game_root: &Path,
    maximum_import_hits: usize,
    timeout_seconds: u32,
) -> Result<ProbeOutcome, LinkError> {
    #[repr(C)]
    struct EntryParameters {
        argument_count: u32,
        reserved: u32,
        argument0: u64,
        argument1: u64,
        argument2: u64,
    }
    // Claim the budget before the process opens anything on the guest's behalf.
    #[cfg(unix)]
    raise_open_file_budget_once();
    let image_name =
        CString::new(process_image_name).map_err(|_| LinkError::InvalidProcessImageName)?;
    let parameters = EntryParameters {
        argument_count: 1,
        reserved: 0,
        argument0: image_name.as_ptr() as usize as u64,
        argument1: 0,
        argument2: 0,
    };
    let unwind_registration_started = Instant::now();
    let registered_fdes = register_guest_unwind_frames(image)?;
    if registered_fdes != 0 {
        eprintln!(
            "[REMu][CXX] registered {registered_fdes} guest unwind records from {} sections in {:.3}s",
            image
                .unwind_modules
                .iter()
                .filter(|module| module.eh_frame != 0 && module.eh_frame_size != 0)
                .count(),
            unwind_registration_started.elapsed().as_secs_f64(),
        );
    }
    configure_probe_dispatch(image, game_root, maximum_import_hits);
    if let Err(error) = sha_ni::install_sha_ni_sigill_recovery() {
        eprintln!("[REMu] warning: SHA-NI SIGILL recovery unavailable: {error}");
    }
    trace_linked_guest_code(image);
    set_current_guest_thread_pointer(image.tls.thread_pointer.get());
    invoke_guest_module_initializers(&image.module_initializers, 0, 0);
    if timeout_seconds != 0 {
        start_diagnostic_timeout(timeout_seconds);
    }
    let return_value = invoke_guest_entry_with_exit(
        image.memory.entry_address().get(),
        (&raw const parameters).addr() as u64,
        guest_exit as *const () as usize as u64,
    ) as i32;
    run_guest_thread_destructors();
    set_current_guest_thread_pointer(0);
    let recovered = sha_ni::recovered_sha_ni_count();
    if recovered != 0 {
        eprintln!("[REMu][CPU] recovered {recovered} SHA-NI instructions via SIGILL emulation");
    }
    let recovered_sse4a = sha_ni::recovered_sse4a_count();
    if recovered_sse4a != 0 {
        eprintln!(
            "[REMu][CPU] recovered {recovered_sse4a} SSE4a instructions via SIGILL emulation"
        );
    }
    DIAGNOSTIC_TIMEOUT_CANCELLED.store(true, Ordering::Release);
    DIAGNOSTIC_TIMEOUT_SECONDS.store(0, Ordering::Release);
    remu_gpu::set_progress_telemetry_enabled(false);
    let import_hits = IMPORT_TELEMETRY.get().map_or(0, |telemetry| {
        telemetry
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .hits
            .load(Ordering::Relaxed)
    });
    Ok(ProbeOutcome {
        return_value,
        import_hits,
    })
}

#[cfg(not(target_arch = "x86_64"))]
const fn probe_entry(
    _image: &LinkedImage,
    _process_image_name: &str,
    _game_root: &Path,
    _maximum_import_hits: usize,
    _timeout_seconds: u32,
) -> Result<ProbeOutcome, LinkError> {
    Err(LinkError::ProbeRequiresX86_64)
}

#[cfg(target_arch = "x86_64")]
extern "C" fn guest_exit(code: i32) {
    eprintln!("[REMu][PROBE] guest exit handler called with code {code}");
    run_all_guest_exit_callbacks();
    terminate_probe(code);
}

/// Prints the shader opcode histogram accumulated by this run, one `[REMu][SHADER-OPCODES]` line
/// per opcode.
///
/// The report is empty unless `REMU_TRACE_SHADER_OPCODES=1` was set for the process, so every path
/// that ends a run can call this unconditionally.
pub fn report_shader_opcode_coverage() {
    for line in remu_shader::opcode_coverage_report() {
        eprintln!("[REMu][SHADER-OPCODES] {line}");
    }
}

/// Set once the guest has returned, so a run that finishes just as its limit expires still
/// reports the status the guest produced rather than the watchdog's.
static DIAGNOSTIC_TIMEOUT_CANCELLED: AtomicBool = AtomicBool::new(false);

/// Nonzero while a bounded diagnostic probe is active.
///
/// Hot-path attribution counters use this as a cheap guard, so regular emulator runs avoid
/// timestamp sampling and per-event diagnostic state updates.
static DIAGNOSTIC_TIMEOUT_SECONDS: AtomicU32 = AtomicU32::new(0);

fn diagnostic_timeout_active() -> bool {
    DIAGNOSTIC_TIMEOUT_SECONDS.load(Ordering::Acquire) != 0
        && !DIAGNOSTIC_TIMEOUT_CANCELLED.load(Ordering::Acquire)
}

/// The exit status a bounded diagnostic run reports when its time limit expires.
///
/// This matches what the shell reports for a process killed by `SIGALRM`, which is how the limit
/// used to be enforced, so scripts reading the status keep working.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const DIAGNOSTIC_TIMEOUT_EXIT_STATUS: i32 = 142;

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
fn report_diagnostic_stall_attribution() {
    const STALE_PROGRESS_MILLIS: u64 = 2_000;

    let wait = guest_sync::main_timed_wait_snapshot();
    let gpu_events = remu_gpu::gpu_events_in_flight();
    let gpu_work = remu_gpu::gpu_work_items_pending();
    let gpu_flips = remu_gpu::gpu_flips_pending();
    let gpu_completed = remu_gpu::gpu_completed_submission_sequence();
    let gpu_age = remu_gpu::gpu_progress_age_millis();
    let audio_frames = remu_gpu::audio_output_frames();
    let audio_underruns = remu_gpu::audio_underrun_frames();
    let audio_age = remu_gpu::audio_progress_age_millis();
    let audio_state = if audio_frames == 0 {
        "no-callback"
    } else if audio_age.is_some_and(|age| age >= STALE_PROGRESS_MILLIS) {
        "callback-stalled"
    } else {
        "callback-active"
    };
    let gpu_state = if gpu_events == 0 && gpu_work == 0 && gpu_flips == 0 {
        "drained"
    } else if gpu_age.is_some_and(|age| age >= STALE_PROGRESS_MILLIS) {
        "outstanding-stalled"
    } else {
        "outstanding-active"
    };
    if let Some(wait) = wait {
        eprintln!(
            "[REMu][STALL-ATTRIBUTION] main_timed_wait=active condition={:#x} return={:#x} timeout_usec={} consecutive_timeouts={} age_ms={} gpu={gpu_state} events={gpu_events} work={gpu_work} flips={gpu_flips} completed_submission={gpu_completed} progress_age_ms={gpu_age:?} audio={audio_state} frames={audio_frames} underrun_frames={audio_underruns} progress_age_ms={audio_age:?}",
            wait.condition,
            wait.return_address,
            wait.timeout_usec,
            wait.consecutive_timeouts,
            wait.age_millis,
        );
    } else {
        eprintln!(
            "[REMu][STALL-ATTRIBUTION] main_timed_wait=none gpu={gpu_state} events={gpu_events} work={gpu_work} flips={gpu_flips} completed_submission={gpu_completed} progress_age_ms={gpu_age:?} audio={audio_state} frames={audio_frames} underrun_frames={audio_underruns} progress_age_ms={audio_age:?}"
        );
    }
    let backend_unclaimed = remu_gpu::gpu_completed_submission_ids_snapshot();
    let backend_unclaimed_range = backend_unclaimed
        .first()
        .copied()
        .zip(backend_unclaimed.last().copied());
    if let Some(state) = PROBE_DISPATCH
        .get()
        .and_then(|dispatch| dispatch.try_read().ok())
    {
        let pending_graphics_range = state
            .pending_agc_completions
            .front()
            .map(|&(sequence, _)| sequence)
            .zip(
                state
                    .pending_agc_completions
                    .back()
                    .map(|&(sequence, _)| sequence),
            );
        let pending_compute_range = state
            .pending_agc_user_interrupts
            .front()
            .copied()
            .zip(state.pending_agc_user_interrupts.back().copied());
        let pending_graphics_notifications = state
            .pending_events
            .iter()
            .map(|event| u64::from(event.agc_graphics_completions))
            .sum::<u64>();
        let pending_user_interrupts = state
            .pending_events
            .iter()
            .map(|event| u64::from(event.agc_user_interrupts))
            .sum::<u64>();
        let graphics_registrations = state
            .kernel_event_registrations
            .keys()
            .filter(|(_, filter, _)| *filter == KERNEL_EVENT_FILTER_GRAPHICS)
            .count();
        let user_interrupt_registrations = state
            .kernel_event_registrations
            .keys()
            .filter(|(_, filter, ident)| *filter == KERNEL_EVENT_FILTER_USER && *ident == 0x1800)
            .count();
        let telemetry = &state.agc_completion_telemetry;
        eprintln!(
            "[REMu][STALL-AGC] submitted={} backend_newest={gpu_completed} backend_unclaimed={}/{backend_unclaimed_range:?} pending_graphics={}/{pending_graphics_range:?} pending_compute={}/{pending_compute_range:?} refreshed=graphics:{}@{:?},compute:{}@{:?} notifications=graphics:queued{}/delivered{}/cancelled{}/pending{},user:queued{}/delivered{}/cancelled{}/pending{} registrations=graphics:{graphics_registrations}/user:{user_interrupt_registrations} missing_registration=graphics:{}/user:{} unmatched_backend={}@{:?}",
            state.agc_submission_count,
            backend_unclaimed.len(),
            state.pending_agc_completions.len(),
            state.pending_agc_user_interrupts.len(),
            telemetry.refreshed_graphics,
            telemetry.last_refreshed_graphics,
            telemetry.refreshed_compute,
            telemetry.last_refreshed_compute,
            telemetry.queued_graphics_notifications,
            telemetry.delivered_graphics_notifications,
            telemetry.cancelled_graphics_notifications,
            pending_graphics_notifications,
            telemetry.queued_user_interrupts,
            telemetry.delivered_user_interrupts,
            telemetry.cancelled_user_interrupts,
            pending_user_interrupts,
            telemetry.missing_graphics_registration,
            telemetry.missing_user_interrupt_registration,
            telemetry.unmatched_backend_ids,
            telemetry.last_unmatched_backend_id,
        );
    } else {
        eprintln!(
            "[REMu][STALL-AGC] runtime=unavailable-or-dispatch-busy backend_newest={gpu_completed} backend_unclaimed={}/{backend_unclaimed_range:?}",
            backend_unclaimed.len(),
        );
    }
    guest_sync::report_condition_stall_attribution();

    let dispatch_state = PROBE_DISPATCH
        .get()
        .and_then(|dispatch| dispatch.try_read().ok());
    let players = dispatch_state.as_ref().map_or_else(Vec::new, |state| {
        av_player::av_player_stall_snapshots(state)
    });
    let queued_controller_callbacks = dispatch_state.as_ref().map(|state| {
        state
            .av_player
            .pending_callbacks
            .iter()
            .filter(|callback| callback.delivery == AvPlayerEventDelivery::Controller)
            .count()
    });
    drop(dispatch_state);
    report_av_player_controller_stall_attribution(queued_controller_callbacks);
    if players.is_empty() {
        eprintln!("[REMu][STALL-AVPLAYER] players=none-or-dispatch-busy");
    }
    for player in players {
        let decoder_state = if !player.active || !player.playing || player.paused {
            "inactive"
        } else if !player.decoder_present {
            "decoder-absent"
        } else if player.decoder_drained || player.end_of_stream {
            "drained"
        } else if player.decoder_queue_blocked
            && player
                .video_delivery_age_millis
                .is_some_and(|age| age >= STALE_PROGRESS_MILLIS)
        {
            "consumer-blocked"
        } else if !player.decoder_queue_blocked
            && player
                .decoder_progress_age_millis
                .is_some_and(|age| age >= STALE_PROGRESS_MILLIS)
        {
            "producer-stalled"
        } else if player.decoder_produced_frames == 0
            && player.decoder_progress_age_millis.is_none()
        {
            "producer-no-progress"
        } else {
            "active"
        };
        eprintln!(
            "[REMu][STALL-AVPLAYER] handle={:#x} state={decoder_state} active={} playing={} paused={} eos={} source={:?} delivered={} delivery_age_ms={:?} decoder_present={} decoder_drained={} produced={} decoder_progress_age_ms={:?} queue_blocked={}",
            player.handle,
            player.active,
            player.playing,
            player.paused,
            player.end_of_stream,
            player.source,
            player.delivered_video_frames,
            player.video_delivery_age_millis,
            player.decoder_present,
            player.decoder_drained,
            player.decoder_produced_frames,
            player.decoder_progress_age_millis,
            player.decoder_queue_blocked,
        );
    }
}

/// Ends the process once a bounded diagnostic run reaches its time limit.
///
/// A watchdog thread is used rather than `alarm`, whose default action terminates the process
/// outright: that discarded every end-of-run report, which is most of the reason for bounding a
/// run in the first place. The thread runs even when every guest thread is wedged, so it remains
/// a backstop against a hang and not merely a timer.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
fn start_diagnostic_timeout(timeout_seconds: u32) {
    DIAGNOSTIC_TIMEOUT_CANCELLED.store(false, Ordering::Release);
    DIAGNOSTIC_TIMEOUT_SECONDS.store(timeout_seconds, Ordering::Release);
    remu_gpu::set_progress_telemetry_enabled(true);
    let limit = Duration::from_secs(u64::from(timeout_seconds));
    std::thread::Builder::new()
        .name("remu-timeout".to_owned())
        .spawn(move || {
            std::thread::sleep(limit);
            if DIAGNOSTIC_TIMEOUT_CANCELLED.load(Ordering::Acquire) {
                return;
            }
            eprintln!("[REMu] diagnostic time limit of {timeout_seconds}s reached");
            report_diagnostic_stall_attribution();
            terminate_probe(DIAGNOSTIC_TIMEOUT_EXIT_STATUS);
        })
        .ok();
}

fn terminate_probe(code: i32) -> ! {
    report_shader_opcode_coverage();
    stop_av_player_controller();
    #[cfg(unix)]
    // SAFETY: `_exit` terminates the isolated CLI probe without unwinding across guest frames.
    unsafe {
        libc::_exit(code);
    }

    #[cfg(not(unix))]
    std::process::exit(code)
}

#[cfg(test)]
mod platform_independent_tests;
mod platform_services;

#[cfg(all(test, target_os = "macos", target_arch = "x86_64"))]
mod tests;
