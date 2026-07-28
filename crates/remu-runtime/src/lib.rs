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
mod av;
mod av_player;
mod dialog;
mod guest_libc;
mod guest_sync;
mod kernel_api;
#[cfg(target_os = "macos")]
mod kernel_memory;
mod memory;
mod nid_names;
mod nid_table;
mod np;
mod np_api;
mod pad;
mod playgo;
mod pthread;
mod save_data;
mod shader_eval;
mod tls;
mod unwind;
mod video_out;

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ffi::CString;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::io::{BufReader, Read, Seek, Write};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use remu_core::GuestAddress;
use remu_loader::{
    DynamicInfo, Elf64Rela, Elf64Symbol, ImportedSymbol, InspectError, LoadedExecutable,
    ModuleDependencyGraphError, UnresolvedModuleDependency, resolve_module_dependency_graph,
};

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
    AGC_IT_DRAW_INDEX_OFFSET, AGC_IT_EVENT_WRITE, AGC_IT_INDEX_BASE, AGC_IT_INDEX_BUFFER_SIZE,
    AGC_IT_INDEX_TYPE, AGC_IT_INDIRECT_BUFFER, AGC_IT_NOP, AGC_IT_NUM_INSTANCES,
    AGC_IT_RELEASE_MEM, AGC_IT_SET_BASE, AGC_IT_SET_CONTEXT_REG, AGC_IT_SET_CONTEXT_REG_INDIRECT,
    AGC_IT_SET_PREDICATION, AGC_IT_SET_SH_REG, AGC_IT_SET_SH_REG_INDIRECT, AGC_IT_SET_UCONFIG_REG,
    AGC_IT_SET_UCONFIG_REG_INDEX, AGC_IT_SET_UCONFIG_REG_INDIRECT, AGC_IT_WAIT_REG_MEM,
    AGC_IT_WRITE_DATA, AGC_R_ACB_RESET, AGC_R_ACQUIRE_MEM, AGC_R_CX_REGS_INDIRECT, AGC_R_DMA_DATA,
    AGC_R_DRAW_INDEX_AUTO, AGC_R_DRAW_RESET, AGC_R_FLIP, AGC_R_RELEASE_MEM, AGC_R_SH_REGS_INDIRECT,
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
    agc_get_fused_shader_size, agc_pm4, agc_register_defaults, agc_release_mem_patch_address,
    agc_set_indirect_patch_address, agc_wait_reg_mem_patch_address,
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
    agc_apply_set_base, agc_decode_compute_dispatch, agc_emit_draw, agc_emit_indexed_draw,
    agc_image_binding_address,
};
#[cfg(test)]
pub(crate) use agc_draw::{agc_prepare_draw_color_targets, agc_resolve_targetless_scanout};
pub(crate) use agc_flip::{
    AgcVideoOutTarget, agc_dcb_draw_index_auto, agc_dcb_draw_index_offset, agc_dcb_set_flip,
    agc_dcb_set_index_buffer, agc_dcb_set_index_size, agc_dcb_set_num_instances,
    agc_submit_ordered_video_out_flip, agc_video_out_target, agc_video_out_target_for_address,
    queue_agc_event_write, queue_video_out_flip, refresh_completed_video_out_flips,
    request_agc_compute_submission_completion, request_agc_submission_completion,
    video_out_add_flip_event, video_out_register_buffers2,
};
pub(crate) use agc_registers::{
    AgcColorTargetState, AgcComputeDispatchError, AgcSubmissionError,
    agc_apply_submission_registers, agc_color_target_state, agc_has_active_depth_state,
    agc_program_address, agc_register_snapshot, agc_shader_prefix_snapshot, agc_shader_snapshot,
};
pub(crate) use agc_submission::{
    AgcSubmissionContext, agc_driver_submit_acb, agc_driver_submit_dcb,
    agc_driver_submit_multi_acbs, agc_driver_submit_multi_dcbs, agc_parse_graphics_submission,
    agc_read_mapped_u64,
};
#[cfg(test)]
pub(crate) use agc_submission::{
    AgcSubmissionPacket, agc_future_scanout_targets, agc_handle_submission_packet,
    agc_parse_submission, decode_agc_dma_data, decode_agc_event_write, decode_agc_memory_acquire,
    decode_agc_memory_wait, decode_agc_pipeline_barrier,
};
pub(crate) use ampr_api::{
    AMPR_COMMAND_RECORD_SIZE, AprResolveOutputs, ampr_apr_command_buffer_constructor,
    ampr_command_buffer_clear_buffer, ampr_command_buffer_constructor,
    ampr_command_buffer_destructor, ampr_command_buffer_get_current_offset,
    ampr_command_buffer_get_size, ampr_command_buffer_read_file_ordered, ampr_command_buffer_reset,
    ampr_command_buffer_set_buffer, ampr_command_buffer_write_kernel_event_queue,
    ampr_get_file_size, ampr_get_file_stat, ampr_measure_command_size, ampr_resolve_file_paths,
    ampr_resolve_paths_core, ampr_submit_command_buffer, ampr_wait_command_buffer,
};
#[cfg(test)]
pub(crate) use ampr_api::{
    AMPR_INVALID_FILE_ID, ampr_command_buffer_read_file_two_phase_with, ampr_read_file_to_guest,
    join_guest_path,
};
pub(crate) use audio_out2::dispatch_audio_out2_import;
#[cfg(all(test, target_os = "macos", target_arch = "x86_64"))]
pub(crate) use av_player::av_player_nv12_layout;
#[cfg(test)]
pub(crate) use av_player::{
    AV_PLAYER_ERROR_OPERATION_FAILED, AV_PLAYER_STREAM_TYPE_AUDIO, AV_PLAYER_STREAM_TYPE_VIDEO,
    av_player_resolve_source, av_player_video_catch_up_target, av_player_video_frame_is_due,
    av_player_write_stream_info, av_player_write_video_frame_info,
};
pub(crate) use av_player::{
    av_player_add_source, av_player_add_source_ex, av_player_close, av_player_current_time,
    av_player_get_audio_data, av_player_get_stream_info, av_player_get_video_data,
    av_player_init_ex, av_player_is_active, av_player_jump_to_time, av_player_pause,
    av_player_post_init, av_player_set_log_callback, av_player_set_looping, av_player_start,
    av_player_stop, av_player_stream_count, av_player_validate_handle,
    invoke_av_player_event_callbacks, prepare_av_player_video_buffers,
};
#[cfg(test)]
pub(crate) use dialog::{
    COMMON_DIALOG_ERROR_BUSY, COMMON_DIALOG_ERROR_DIALOG_ALREADY_INITIALIZED,
    COMMON_DIALOG_ERROR_NOT_SYSTEM_INITIALIZED, COMMON_DIALOG_STATUS_INITIALIZED,
};
pub(crate) use dialog::{
    common_dialog_initialize, message_dialog_close, message_dialog_get_result,
    message_dialog_get_status, message_dialog_initialize, message_dialog_open,
    message_dialog_terminate, message_dialog_update_status, save_data_dialog_initialize,
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
    guest_pthread_once, guest_semaphore_wait,
};
#[cfg(any(test, target_arch = "x86_64"))]
pub(crate) use guest_sync::{GUEST_SYNC, GuestSyncState, guest_mutex_lock, guest_mutex_unlock};
pub(crate) use guest_sync::{
    dispatch_guest_sync_import_at_import, guest_rwlock_destroy, guest_rwlock_init,
    guest_rwlock_lock, guest_rwlock_unlock, guest_sync, is_guest_sync_import,
};
#[cfg(test)]
pub(crate) use memory::BATCH_MAP_OPERATION_MAP_DIRECT;
pub(crate) use memory::{
    allocate_direct_memory, allocate_main_direct_memory, direct_memory_query, kernel_batch_map,
    kernel_batch_map_2, kernel_checked_release_direct_memory, kernel_mprotect, kernel_mtypeprotect,
    kernel_munmap, kernel_release_direct_memory, kernel_reserve_virtual_range,
    kernel_set_virtual_range_name, kernel_virtual_query, map_direct_memory, map_direct_memory_2,
    map_flexible_memory_core, map_named_flexible_memory,
};
#[cfg(all(test, target_os = "macos", target_arch = "x86_64"))]
pub(crate) use memory::{host_protection_from_guest, process_mapping_contains};
pub use nid_table::{known_nid_name, nid_display_name};
pub(crate) use np_api::{
    np_game_intent_get_property_string, np_game_intent_initialize, np_game_intent_receive,
    np_trophy_abort_handle, np_trophy_create_context, np_trophy_create_handle,
    np_trophy_destroy_context, np_trophy_destroy_handle, np_trophy_register_context,
    np_trophy_register_unlock_callback, np_trophy_unregister_unlock_callback,
    np_universal_data_abort_handle, np_universal_data_array_set_string,
    np_universal_data_create_context, np_universal_data_create_event,
    np_universal_data_create_handle, np_universal_data_create_property_object,
    np_universal_data_destroy_context, np_universal_data_destroy_event,
    np_universal_data_destroy_handle, np_universal_data_initialize,
    np_universal_data_object_set_array, np_universal_data_object_set_i32,
    np_universal_data_object_set_string, np_universal_data_post_event,
    np_universal_data_register_context, np_universal_data_terminate,
};
#[cfg(test)]
pub(crate) use pad::{
    SCE_PAD_ERROR_ALREADY_OPENED, SCE_PAD_ERROR_DEVICE_NO_HANDLE, SCE_PAD_ERROR_INVALID_HANDLE,
    SCE_PAD_ERROR_NOT_INITIALIZED,
};
pub(crate) use pad::{
    pad_boolean_operation, pad_close, pad_device_class_parse_data, pad_get_controller_information,
    pad_get_extended_information, pad_get_handle, pad_get_trigger_effect_state,
    pad_handle_operation, pad_initialize, pad_open, pad_pointer_operation, pad_read,
    pad_read_state, pad_set_motion_sensor_state, pad_set_vibration_mode,
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
use remu_memory::{GUEST_EXECUTE, GUEST_READ, GuestImage, MappingError};
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
    save_data_create_transaction_resource, save_data_dir_name_search, save_data_initialize,
    save_data_mount3, save_data_prepare, save_data_save_icon, save_data_set_param,
    save_data_umount2, write_guest_bytes,
};
use sha1::{Digest, Sha1};
pub(crate) use shader_eval::{
    draw_required_vertex_record_count, evaluate_draw_shader_requests_parallel,
    evaluate_shader_register_snapshot_cached, export_user_data_register,
    shader_control_flow_summary, shader_evaluation_worker_request, snapshot_draw_indices,
    snapshot_small_readonly_global_bindings, trace_agc_shader_draw,
};
use thiserror::Error;
use tls::{
    GUEST_TLS_PREFIX_SIZE, GUEST_TLS_TCB_SIZE, TLS_HELPER_REGION_SIZE, TlsImageInput,
    install_tls_support, tls_dtv_entry_offset,
};
#[cfg(target_arch = "x86_64")]
use unwind::register_guest_unwind_frames;
use unwind::{GuestUnwindModule, collect_unwind_fdes_at, guest_unwind_module};
#[cfg(test)]
pub(crate) use video_out::{
    DEFAULT_FLIP_INTERVAL_MICROSECONDS, VIDEO_OUT_ERROR_RESOURCE_BUSY, VIDEO_OUT_FLIP_INTERVAL,
    flip_interval_microseconds,
};
pub(crate) use video_out::{
    VIDEO_OUT_ERROR_INVALID_ADDRESS, VIDEO_OUT_ERROR_INVALID_HANDLE, VIDEO_OUT_ERROR_INVALID_INDEX,
    VIDEO_OUT_ERROR_INVALID_VALUE, av_player_flip_interval_microseconds, pace_video_out_flip,
    video_out_get_output_status, video_out_is_flip_pending, video_out_open,
    video_out_set_buffer_attribute2, video_out_set_flip_rate, video_out_submit_flip,
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

const R_X86_64_64: u32 = 1;
const R_X86_64_GLOB_DAT: u32 = 6;
const R_X86_64_JUMP_SLOT: u32 = 7;
const R_X86_64_RELATIVE: u32 = 8;
const R_X86_64_DTPMOD64: u32 = 16;
const IMPORT_STUB_SIZE: u64 = 32;
const IMPORT_STUB_SIZE_USIZE: usize = 32;
const PT_TLS: u32 = 7;
const PT_SCE_PROC_PARAM: u32 = 0x6100_0001;
const MACOS_GUEST_PAGE_ALIGNMENT: u64 = 0x4000;
const SHADER_EVALUATION_CACHE_CAPACITY: usize = 16 * 1024;
const SHADER_EVALUATION_PENDING_CAPACITY: usize = SHADER_EVALUATION_CACHE_CAPACITY * 4;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportStub {
    pub nid: String,
    pub address: GuestAddress,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ImportResolutionStats {
    pub bundled_guest: usize,
    pub host_direct: usize,
    pub hle_dispatch: usize,
    pub unresolved_weak: usize,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ImportResolutionKind {
    BundledGuest,
    HostDirect,
    HleDispatch,
    UnresolvedWeak,
}

impl ImportResolutionKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BundledGuest => "bundled-guest",
            Self::HostDirect => "host-direct",
            Self::HleDispatch => "hle-dispatch",
            Self::UnresolvedWeak => "unresolved-weak",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportResolutionRecord {
    pub symbol: String,
    pub nid: String,
    pub weak: bool,
    pub kind: ImportResolutionKind,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RelocationStats {
    pub absolute: usize,
    pub global_data: usize,
    pub jump_slots: usize,
    pub relative: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuestTls {
    pub thread_pointer: GuestAddress,
    pub static_block_size: u64,
    pub patched_loads: usize,
}

impl RelocationStats {
    #[must_use]
    pub const fn total(self) -> usize {
        self.absolute + self.global_data + self.jump_slots + self.relative
    }
}

impl ImportResolutionStats {
    #[must_use]
    pub const fn total(self) -> usize {
        self.bundled_guest + self.host_direct + self.hle_dispatch + self.unresolved_weak
    }
}

#[derive(Debug)]
pub struct LinkedImage {
    pub memory: GuestImage,
    pub import_stubs: Vec<ImportStub>,
    pub relocation_stats: RelocationStats,
    pub tls: GuestTls,
    pub process_param: Option<GuestAddress>,
    import_resolution_stats: ImportResolutionStats,
    import_resolutions: Vec<ImportResolutionRecord>,
    unwind_fdes: Vec<GuestAddress>,
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    module_initializers: Vec<GuestAddress>,
    bundled_modules: Vec<LinkedGuestModule>,
    unwind_modules: Vec<GuestUnwindModule>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LinkedGuestModule {
    name: String,
    handle: u32,
    exports_by_nid: BTreeMap<String, u64>,
    initializers: Vec<GuestAddress>,
    started: bool,
}

/// Filesystem and dependency information retained from linking a complete guest process image.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModuleLinkReport {
    pub app_root: PathBuf,
    pub entry_path: PathBuf,
    /// Bundled PRX/SPRX objects in the dependency-first order used for mapping and initialization.
    pub loaded_modules: Vec<PathBuf>,
    /// Dependencies that must be supplied by the emulator's system-module implementations.
    pub unresolved_dependencies: Vec<UnresolvedModuleDependency>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProbeOutcome {
    pub return_value: i32,
    pub import_hits: usize,
}

#[derive(Debug, Error)]
pub enum LinkError {
    #[error(transparent)]
    Mapping(#[from] MappingError),
    #[error(transparent)]
    Inspect(#[from] InspectError),
    #[error(transparent)]
    ModuleGraph(#[from] ModuleDependencyGraphError),
    #[error("executable does not contain an ELF image")]
    MissingElf,
    #[error("executable does not contain dynamic-linker metadata")]
    MissingDynamicMetadata,
    #[error("import stub allocation size overflow")]
    StubAllocationOverflow,
    #[error("relocation references missing symbol {index}")]
    MissingSymbol { index: u32 },
    #[error("unresolved non-weak symbol {index} has no import stub")]
    MissingImportStub { index: u32 },
    #[error("dynamic symbol {symbol:?} refers to an unknown Sony library or module namespace")]
    InvalidSymbolNamespace { symbol: String },
    #[error("unsupported x86-64 relocation type {relocation_type} at {offset}")]
    UnsupportedRelocation {
        relocation_type: u32,
        offset: GuestAddress,
    },
    #[error("guest address arithmetic overflow")]
    AddressOverflow,
    #[error("guest TLS alignment 0x{0:x} is not a power of two")]
    InvalidTlsAlignment(u64),
    #[error(
        "guest static TLS requires 0x{required:x} bytes, exceeding the 0x{available:x} startup reservation"
    )]
    StaticTlsTooLarge { required: u64, available: u64 },
    #[error("executable region size cannot be represented on this host")]
    ExecutableRegionTooLarge,
    #[error("TLS helper at {helper} is outside rel32 range of guest instruction {instruction}")]
    TlsHelperOutOfRange {
        instruction: GuestAddress,
        helper: GuestAddress,
    },
    #[error("native guest probing requires an x86-64 process")]
    ProbeRequiresX86_64,
    #[error("process image name contains an interior NUL byte")]
    InvalidProcessImageName,
    #[error("unsupported guest .eh_frame pointer encoding 0x{0:02x}")]
    UnsupportedEhFrameEncoding(u8),
    #[error("malformed guest .eh_frame data: {0}")]
    MalformedEhFrame(&'static str),
    #[error("could not discover runtime modules under {path}: {source}")]
    RuntimeModuleDiscovery {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("guest {kind} byte size 0x{size:x} is not a multiple of eight")]
    InvalidInitializerArraySize { kind: &'static str, size: u64 },
    #[error("guest initializer address {address} is outside the process image")]
    InitializerOutsideImage { address: GuestAddress },
    #[error("native macOS C++ unwind runtime is unavailable")]
    NativeCxxRuntimeUnavailable,
    #[error("native macOS unwind symbol {0} is unavailable")]
    MissingNativeUnwindSymbol(&'static str),
}

fn discover_runtime_module_paths(app_root: &Path) -> Result<Vec<PathBuf>, LinkError> {
    // Unreal plug-ins are commonly unpacked under Media/Plugins, while Wwise titles such as
    // Silent Hill keep load-on-demand effect modules directly in /app0/prx. Both sets must be
    // mapped up front because the process image is finalized before sceKernelLoadStartModule can
    // ask to start one of them.
    let mut pending = [
        app_root.join("decrypted/Media/Plugins"),
        app_root.join("prx"),
    ]
    .into_iter()
    .filter(|root| root.is_dir())
    .collect::<Vec<_>>();
    let mut modules = Vec::new();
    while let Some(directory) = pending.pop() {
        let entries = std::fs::read_dir(&directory)
            .map_err(|source| LinkError::RuntimeModuleDiscovery {
                path: directory.clone(),
                source,
            })?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|source| LinkError::RuntimeModuleDiscovery {
                path: directory.clone(),
                source,
            })?;
        let mut entries = entries;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries.into_iter().rev() {
            let path = entry.path();
            let file_type =
                entry
                    .file_type()
                    .map_err(|source| LinkError::RuntimeModuleDiscovery {
                        path: path.clone(),
                        source,
                    })?;
            if file_type.is_dir() {
                pending.push(path);
                continue;
            }
            if (!file_type.is_file() && !file_type.is_symlink())
                || !path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| {
                        extension.eq_ignore_ascii_case("prx")
                            || extension.eq_ignore_ascii_case("sprx")
                    })
            {
                continue;
            }
            modules.push(std::fs::canonicalize(&path).map_err(|source| {
                LinkError::RuntimeModuleDiscovery {
                    path: path.clone(),
                    source,
                }
            })?);
        }
    }
    modules.sort();
    Ok(modules)
}

impl LinkedImage {
    /// Discovers, loads, maps, and relocates the complete bundled dependency graph for `path`.
    ///
    /// Missing system PRXs remain explicit in the returned report because those dependencies are
    /// fulfilled by `REMu`'s HLE services rather than files from the game dump.
    pub fn link_path(path: impl AsRef<Path>) -> Result<(Self, ModuleLinkReport), LinkError> {
        let graph = resolve_module_dependency_graph(path)?;
        let executable = remu_loader::load_path(&graph.entry.path)?;
        let mut dependency_order = graph.dependency_order;
        let mut unresolved_dependencies = graph.unresolved_dependencies;
        let mut loaded_paths = dependency_order
            .iter()
            .map(|node| node.path.clone())
            .collect::<BTreeSet<_>>();
        let startup_paths = loaded_paths.clone();
        for plugin_path in discover_runtime_module_paths(&graph.app_root)? {
            let plugin_graph = resolve_module_dependency_graph(&plugin_path)?;
            unresolved_dependencies.extend(plugin_graph.unresolved_dependencies);
            for node in plugin_graph
                .dependency_order
                .into_iter()
                .chain(std::iter::once(plugin_graph.entry))
            {
                if loaded_paths.insert(node.path.clone()) {
                    dependency_order.push(node);
                }
            }
        }
        let mut modules = Vec::with_capacity(dependency_order.len());
        for node in &dependency_order {
            let name = node
                .original_filename
                .clone()
                .or_else(|| {
                    node.path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .map(str::to_owned)
                })
                .unwrap_or_else(|| node.path.display().to_string());
            modules.push((name, remu_loader::load_path(&node.path)?));
        }
        let module_refs = modules
            .iter()
            .map(|(name, executable)| (name.as_str(), executable))
            .collect::<Vec<_>>();
        let module_startup = dependency_order
            .iter()
            .map(|node| startup_paths.contains(&node.path))
            .collect::<Vec<_>>();
        let image =
            Self::link_main_with_modules_and_startup(&executable, &module_refs, &module_startup)?;
        let report = ModuleLinkReport {
            app_root: graph.app_root,
            entry_path: graph.entry.path,
            loaded_modules: dependency_order.into_iter().map(|node| node.path).collect(),
            unresolved_dependencies,
        };
        Ok((image, report))
    }

    pub fn link_main(executable: &LoadedExecutable) -> Result<Self, LinkError> {
        Self::link_main_with_modules(executable, &[])
    }

    /// Links the main executable and dependency-ordered game-bundled PRX modules into one process
    /// address space and symbol scope.
    pub fn link_main_with_modules(
        executable: &LoadedExecutable,
        modules: &[(&str, &LoadedExecutable)],
    ) -> Result<Self, LinkError> {
        let module_startup = vec![true; modules.len()];
        Self::link_main_with_modules_and_startup(executable, modules, &module_startup)
    }

    fn link_main_with_modules_and_startup(
        executable: &LoadedExecutable,
        modules: &[(&str, &LoadedExecutable)],
        module_startup: &[bool],
    ) -> Result<Self, LinkError> {
        debug_assert_eq!(modules.len(), module_startup.len());
        let elf = executable.info.elf.as_ref().ok_or(LinkError::MissingElf)?;
        let dynamic = elf
            .dynamic
            .as_ref()
            .ok_or(LinkError::MissingDynamicMetadata)?;
        let auxiliary_reserve = calculate_process_auxiliary_reserve(dynamic, modules)?;
        let mut memory =
            GuestImage::map_main_with_auxiliary_reserve(executable, auxiliary_reserve)?;
        let linked_modules = map_bundled_modules(&mut memory, modules)?;

        let main_export = main_base_export_input(executable, dynamic, memory.image_base());
        let process_exports = collect_process_exports(&memory, &main_export, &linked_modules)?;
        let (import_resolution_stats, import_resolutions) = collect_import_resolutions(
            std::iter::once(dynamic).chain(linked_modules.iter().map(|module| module.dynamic)),
            &process_exports,
        )?;
        let mut import_stubs = Vec::new();
        let mut relocation_stats = RelocationStats::default();
        let mut unwind_fdes = Vec::new();
        for module in &linked_modules {
            let (module_stubs, symbol_stubs) = create_import_stubs(
                &mut memory,
                module.dynamic,
                import_stubs.len(),
                &process_exports,
            )?;
            import_stubs.extend(module_stubs);
            add_relocation_stats(
                &mut relocation_stats,
                apply_relocations_at(
                    &mut memory,
                    module.base,
                    module.tls_module_id,
                    module.dynamic,
                    &symbol_stubs,
                    &process_exports,
                )?,
            );
            unwind_fdes.extend(collect_unwind_fdes_at(
                &memory,
                module.base,
                module.executable,
            )?);
        }

        let (main_stubs, symbol_stubs) =
            create_import_stubs(&mut memory, dynamic, import_stubs.len(), &process_exports)?;
        import_stubs.extend(main_stubs);
        let main_base = memory.image_base();
        let process_param = resolve_process_param(elf, main_base)?;
        add_relocation_stats(
            &mut relocation_stats,
            apply_relocations_at(
                &mut memory,
                main_base,
                1,
                dynamic,
                &symbol_stubs,
                &process_exports,
            )?,
        );
        unwind_fdes.extend(collect_unwind_fdes_at(&memory, main_base, executable)?);
        let mut linked_module_initializers = Vec::with_capacity(linked_modules.len());
        for module in &linked_modules {
            let mut initializers = Vec::new();
            append_object_initializers(&mut initializers, &memory, module.base, module.dynamic)?;
            linked_module_initializers.push(initializers);
        }
        let module_initializers = linked_module_initializers
            .iter()
            .zip(module_startup)
            .filter(|(_, startup)| **startup)
            .flat_map(|(initializers, _)| initializers.iter().copied())
            .collect();
        let bundled_modules = modules
            .iter()
            .zip(&linked_modules)
            .zip(&linked_module_initializers)
            .zip(module_startup)
            .map(|((((name, _), module), initializers), startup)| {
                Ok(LinkedGuestModule {
                    name: (*name).to_owned(),
                    handle: u32::try_from(module.tls_module_id)
                        .expect("the module count was already represented by a platform usize"),
                    exports_by_nid: module_exports_by_nid(&memory, module)?,
                    initializers: initializers.clone(),
                    started: *startup,
                })
            })
            .collect::<Result<Vec<_>, LinkError>>()?;
        let unwind_modules = std::iter::once(("eboot.bin", executable, main_base))
            .chain(
                modules
                    .iter()
                    .zip(&linked_modules)
                    .map(|((name, executable), module)| (*name, *executable, module.base)),
            )
            .map(|(name, executable, base)| guest_unwind_module(&memory, name, base, executable))
            .collect::<Result<Vec<_>, _>>()?;
        // A Gen5 main executable enters through its own runtime bootstrap. Its DT_INIT commonly
        // aliases the ELF entry point, so invoking it here would start the title once with null
        // process arguments and then start it a second time below. Dependency modules have no
        // independent process entry and therefore still require loader-driven initialization.
        let tls_inputs = std::iter::once(TlsImageInput {
            executable,
            base: main_base,
            module_id: 1,
        })
        .chain(linked_modules.iter().map(|module| TlsImageInput {
            executable: module.executable,
            base: module.base,
            module_id: module.tls_module_id,
        }))
        .collect::<Vec<_>>();
        let tls = install_tls_support(&mut memory, &tls_inputs)?;
        memory.finalize_permissions()?;
        Ok(Self {
            memory,
            import_stubs,
            relocation_stats,
            tls,
            process_param,
            import_resolution_stats,
            import_resolutions,
            unwind_fdes,
            module_initializers,
            bundled_modules,
            unwind_modules,
        })
    }

    pub fn probe_entry(
        &self,
        process_image_name: &str,
        game_root: &Path,
        maximum_import_hits: usize,
        timeout_seconds: u32,
    ) -> Result<ProbeOutcome, LinkError> {
        probe_entry(
            self,
            process_image_name,
            game_root,
            maximum_import_hits,
            timeout_seconds,
        )
    }

    #[must_use]
    pub fn unwind_record_count(&self) -> usize {
        self.unwind_fdes.len()
    }

    #[must_use]
    pub fn import_resolution_stats(&self) -> ImportResolutionStats {
        self.import_resolution_stats
    }

    #[must_use]
    pub fn import_resolutions(&self) -> &[ImportResolutionRecord] {
        &self.import_resolutions
    }
}

fn resolve_process_param(
    elf: &remu_loader::Elf64Info,
    image_base: GuestAddress,
) -> Result<Option<GuestAddress>, LinkError> {
    elf.program_headers
        .iter()
        .find(|header| header.program_type == PT_SCE_PROC_PARAM)
        .map(|header| {
            image_base
                .get()
                .checked_add(header.virtual_address.get())
                .map(GuestAddress::new)
                .ok_or(LinkError::AddressOverflow)
        })
        .transpose()
}

struct LinkedModuleInput<'a> {
    executable: &'a LoadedExecutable,
    dynamic: &'a DynamicInfo,
    base: GuestAddress,
    tls_module_id: u64,
}

fn map_bundled_modules<'a>(
    memory: &mut GuestImage,
    modules: &[(&'a str, &'a LoadedExecutable)],
) -> Result<Vec<LinkedModuleInput<'a>>, LinkError> {
    let mut linked_modules = Vec::with_capacity(modules.len());
    for (index, (_name, module)) in modules.iter().enumerate() {
        let dynamic = module
            .info
            .elf
            .as_ref()
            .ok_or(LinkError::MissingElf)?
            .dynamic
            .as_ref()
            .ok_or(LinkError::MissingDynamicMetadata)?;
        let base = memory.map_module(module, MACOS_GUEST_PAGE_ALIGNMENT)?;
        linked_modules.push(LinkedModuleInput {
            executable: module,
            dynamic,
            base,
            tls_module_id: u64::try_from(index + 2).map_err(|_| LinkError::AddressOverflow)?,
        });
    }
    Ok(linked_modules)
}

fn calculate_process_auxiliary_reserve(
    main_dynamic: &DynamicInfo,
    modules: &[(&str, &LoadedExecutable)],
) -> Result<u64, LinkError> {
    let mut cursor = 0_u64;
    let mut import_count = main_dynamic.imports.len();
    for (_, module) in modules {
        cursor =
            align_up_u64(cursor, MACOS_GUEST_PAGE_ALIGNMENT).ok_or(LinkError::AddressOverflow)?;
        let module_size = module
            .load_segments
            .iter()
            .try_fold(0_u64, |maximum, segment| {
                segment
                    .virtual_address
                    .get()
                    .checked_add(segment.memory_size)
                    .map(|end| maximum.max(end))
                    .ok_or(LinkError::AddressOverflow)
            })?;
        if module_size == 0 {
            return Err(LinkError::Mapping(MappingError::NoLoadSegments));
        }
        cursor = cursor
            .checked_add(module_size)
            .ok_or(LinkError::AddressOverflow)?;
        let dynamic = module
            .info
            .elf
            .as_ref()
            .ok_or(LinkError::MissingElf)?
            .dynamic
            .as_ref()
            .ok_or(LinkError::MissingDynamicMetadata)?;
        import_count = import_count
            .checked_add(dynamic.imports.len())
            .ok_or(LinkError::StubAllocationOverflow)?;
    }

    cursor = align_up_u64(cursor, IMPORT_STUB_SIZE).ok_or(LinkError::AddressOverflow)?;
    let maximum_stub_bytes = u64::try_from(import_count)
        .map_err(|_| LinkError::StubAllocationOverflow)?
        .checked_mul(IMPORT_STUB_SIZE)
        .ok_or(LinkError::StubAllocationOverflow)?;
    cursor = cursor
        .checked_add(maximum_stub_bytes)
        .ok_or(LinkError::AddressOverflow)?;
    cursor = align_up_u64(cursor, 16).ok_or(LinkError::AddressOverflow)?;
    cursor = cursor
        .checked_add(TLS_HELPER_REGION_SIZE)
        .ok_or(LinkError::AddressOverflow)?;
    cursor = align_up_u64(cursor, MACOS_GUEST_PAGE_ALIGNMENT).ok_or(LinkError::AddressOverflow)?;
    cursor = cursor
        .checked_add(GUEST_TLS_PREFIX_SIZE + GUEST_TLS_TCB_SIZE)
        .ok_or(LinkError::AddressOverflow)?;
    // Keep one guard-sized cushion for alignment changes without hiding unbounded allocations.
    cursor = cursor
        .checked_add(MACOS_GUEST_PAGE_ALIGNMENT)
        .ok_or(LinkError::AddressOverflow)?;
    align_up_u64(cursor, MACOS_GUEST_PAGE_ALIGNMENT).ok_or(LinkError::AddressOverflow)
}

fn add_relocation_stats(total: &mut RelocationStats, addition: RelocationStats) {
    total.absolute += addition.absolute;
    total.global_data += addition.global_data;
    total.jump_slots += addition.jump_slots;
    total.relative += addition.relative;
}

fn append_object_initializers(
    output: &mut Vec<GuestAddress>,
    memory: &GuestImage,
    image_base: GuestAddress,
    dynamic: &DynamicInfo,
) -> Result<(), LinkError> {
    if dynamic.init_address != 0 {
        output.push(normalize_process_address(
            memory,
            image_base,
            dynamic.init_address,
        )?);
    }
    output.extend(collect_initializer_array(
        memory,
        image_base,
        dynamic.init_array_address,
        dynamic.init_array_size,
        "DT_INIT_ARRAY",
    )?);
    Ok(())
}

fn collect_initializer_array(
    memory: &GuestImage,
    image_base: GuestAddress,
    address: u64,
    size: u64,
    kind: &'static str,
) -> Result<Vec<GuestAddress>, LinkError> {
    if address == 0 || size == 0 {
        return Ok(Vec::new());
    }
    if !size.is_multiple_of(8) {
        return Err(LinkError::InvalidInitializerArraySize { kind, size });
    }
    let count = usize::try_from(size / 8).map_err(|_| LinkError::AddressOverflow)?;
    let byte_length = usize::try_from(size).map_err(|_| LinkError::AddressOverflow)?;
    let array = runtime_address_at(memory, image_base, GuestAddress::new(address), byte_length)?;
    let mut bytes = vec![0_u8; byte_length];
    memory.read(array, &mut bytes)?;
    let mut initializers = Vec::with_capacity(count);
    for entry in bytes.chunks_exact(8) {
        let value = u64::from_le_bytes(entry.try_into().expect("fixed initializer pointer"));
        if value != 0 && value != u64::MAX {
            initializers.push(normalize_process_address(memory, image_base, value)?);
        }
    }
    Ok(initializers)
}

fn normalize_process_address(
    memory: &GuestImage,
    image_base: GuestAddress,
    address: u64,
) -> Result<GuestAddress, LinkError> {
    let address = GuestAddress::new(address);
    if memory.contains(address, 1) {
        return Ok(address);
    }
    let rebased = GuestAddress::new(
        image_base
            .get()
            .checked_add(address.get())
            .ok_or(LinkError::AddressOverflow)?,
    );
    if memory.contains(rebased, 1) {
        Ok(rebased)
    } else {
        Err(LinkError::InitializerOutsideImage { address: rebased })
    }
}

#[derive(Clone, Debug)]
struct ProcessExport {
    address: u64,
    weak: bool,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum ProcessSymbolKey {
    Scoped {
        nid: String,
        library: String,
        library_version: u16,
        module: String,
        module_version_major: u8,
        module_version_minor: u8,
        symbol_type: u8,
    },
    Unscoped {
        nid: String,
        symbol_type: u8,
    },
}

impl ProcessSymbolKey {
    fn nid(&self) -> &str {
        match self {
            Self::Scoped { nid, .. } | Self::Unscoped { nid, .. } => nid,
        }
    }
}

fn main_base_export_input<'a>(
    executable: &'a LoadedExecutable,
    dynamic: &'a DynamicInfo,
    base: GuestAddress,
) -> LinkedModuleInput<'a> {
    LinkedModuleInput {
        executable,
        dynamic,
        base,
        tls_module_id: 1,
    }
}

fn collect_process_exports(
    memory: &GuestImage,
    main: &LinkedModuleInput<'_>,
    modules: &[LinkedModuleInput<'_>],
) -> Result<BTreeMap<ProcessSymbolKey, u64>, LinkError> {
    let mut exports = BTreeMap::<ProcessSymbolKey, ProcessExport>::new();
    for module in std::iter::once(main).chain(modules) {
        for symbol in module.dynamic.symbols.iter().filter(|symbol| {
            !symbol.is_undefined() && !symbol.name.is_empty() && matches!(symbol.binding(), 1 | 2)
        }) {
            let key = process_symbol_key(module.dynamic, symbol)?;
            if key.nid().is_empty() {
                continue;
            }
            let address = defined_symbol_address_at(memory, module.base, symbol)?;
            let candidate = ProcessExport {
                address,
                weak: symbol.is_weak(),
            };
            match exports.get_mut(&key) {
                None => {
                    exports.insert(key, candidate);
                }
                Some(current) if current.weak && !candidate.weak => {
                    *current = candidate;
                }
                Some(_) => {}
            }
        }
    }
    Ok(exports
        .into_iter()
        .map(|(nid, export)| (nid, export.address))
        .collect())
}

fn module_exports_by_nid(
    memory: &GuestImage,
    module: &LinkedModuleInput<'_>,
) -> Result<BTreeMap<String, u64>, LinkError> {
    let mut exports = BTreeMap::<String, ProcessExport>::new();
    for symbol in module.dynamic.symbols.iter().filter(|symbol| {
        !symbol.is_undefined() && !symbol.name.is_empty() && matches!(symbol.binding(), 1 | 2)
    }) {
        let nid = symbol.name.split('#').next().unwrap_or_default();
        if nid.is_empty() {
            continue;
        }
        let candidate = ProcessExport {
            address: defined_symbol_address_at(memory, module.base, symbol)?,
            weak: symbol.is_weak(),
        };
        match exports.get_mut(nid) {
            None => {
                exports.insert(nid.to_owned(), candidate);
            }
            Some(current) if current.weak && !candidate.weak => *current = candidate,
            Some(_) => {}
        }
    }
    Ok(exports
        .into_iter()
        .map(|(nid, export)| (nid, export.address))
        .collect())
}

fn process_symbol_key(
    dynamic: &DynamicInfo,
    symbol: &Elf64Symbol,
) -> Result<ProcessSymbolKey, LinkError> {
    let mut ids = symbol.name.split('#');
    let nid = ids.next().unwrap_or_default();
    let library_id = ids.next();
    let module_id = ids.next();
    if ids.next().is_none()
        && let (Some(library_id), Some(module_id)) = (library_id, module_id)
        && let Some(library) = dynamic
            .imported_libraries
            .iter()
            .chain(&dynamic.exported_libraries)
            .find(|library| encode_sce_id(library.id) == library_id)
        && let Some(module) = dynamic
            .required_modules
            .iter()
            .chain(&dynamic.exported_modules)
            .find(|module| encode_sce_id(module.id) == module_id)
    {
        return Ok(ProcessSymbolKey::Scoped {
            nid: nid.to_owned(),
            library: library.name.clone(),
            library_version: library.version,
            module: module.name.clone(),
            module_version_major: module.version_major,
            module_version_minor: module.version_minor,
            symbol_type: symbol.symbol_type(),
        });
    }
    if library_id.is_some()
        && module_id.is_some()
        && (!dynamic.imported_libraries.is_empty()
            || !dynamic.exported_libraries.is_empty()
            || !dynamic.required_modules.is_empty()
            || !dynamic.exported_modules.is_empty())
    {
        return Err(LinkError::InvalidSymbolNamespace {
            symbol: symbol.name.clone(),
        });
    }
    Ok(ProcessSymbolKey::Unscoped {
        nid: nid.to_owned(),
        symbol_type: symbol.symbol_type(),
    })
}

fn encode_sce_id(id: u16) -> String {
    const CODES: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    let mut encoded = String::with_capacity(3);
    if id < 0x40 {
        encoded.push(char::from(CODES[usize::from(id)]));
    } else if id < 0x1000 {
        encoded.push(char::from(CODES[usize::from((id >> 6) & 0x3f)]));
        encoded.push(char::from(CODES[usize::from(id & 0x3f)]));
    } else {
        encoded.push(char::from(CODES[usize::from((id >> 12) & 0x3f)]));
        encoded.push(char::from(CODES[usize::from((id >> 6) & 0x3f)]));
        encoded.push(char::from(CODES[usize::from(id & 0x3f)]));
    }
    encoded
}

fn imported_symbol<'a>(
    dynamic: &'a DynamicInfo,
    import: &ImportedSymbol,
) -> Result<&'a Elf64Symbol, LinkError> {
    let index = usize::try_from(import.symbol_index).map_err(|_| LinkError::MissingSymbol {
        index: import.symbol_index,
    })?;
    dynamic.symbols.get(index).ok_or(LinkError::MissingSymbol {
        index: import.symbol_index,
    })
}

fn classify_import(
    dynamic: &DynamicInfo,
    import: &ImportedSymbol,
    process_exports: &BTreeMap<ProcessSymbolKey, u64>,
) -> Result<ImportResolutionKind, LinkError> {
    let symbol = imported_symbol(dynamic, import)?;
    let key = process_symbol_key(dynamic, symbol)?;
    if process_exports.contains_key(&key) {
        return Ok(ImportResolutionKind::BundledGuest);
    }
    let nid = key.nid();
    if native_import_target(nid).is_some() || synthetic_data_import_target(nid).is_some() {
        return Ok(ImportResolutionKind::HostDirect);
    }
    if import.weak {
        Ok(ImportResolutionKind::UnresolvedWeak)
    } else {
        Ok(ImportResolutionKind::HleDispatch)
    }
}

fn collect_import_resolutions<'a>(
    dynamics: impl IntoIterator<Item = &'a DynamicInfo>,
    process_exports: &BTreeMap<ProcessSymbolKey, u64>,
) -> Result<(ImportResolutionStats, Vec<ImportResolutionRecord>), LinkError> {
    let mut stats = ImportResolutionStats::default();
    let mut records = Vec::new();
    for dynamic in dynamics {
        for import in &dynamic.imports {
            let kind = classify_import(dynamic, import, process_exports)?;
            match kind {
                ImportResolutionKind::BundledGuest => stats.bundled_guest += 1,
                ImportResolutionKind::HostDirect => stats.host_direct += 1,
                ImportResolutionKind::HleDispatch => stats.hle_dispatch += 1,
                ImportResolutionKind::UnresolvedWeak => stats.unresolved_weak += 1,
            }
            let symbol = imported_symbol(dynamic, import)?;
            let key = process_symbol_key(dynamic, symbol)?;
            records.push(ImportResolutionRecord {
                symbol: symbol.name.clone(),
                nid: key.nid().to_owned(),
                weak: import.weak,
                kind,
            });
        }
    }
    records.sort_by(|left, right| {
        (&left.symbol, left.kind, left.weak).cmp(&(&right.symbol, right.kind, right.weak))
    });
    Ok((stats, records))
}

#[cfg(test)]
mod linker_metadata_tests;

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

fn add_address(address: GuestAddress, offset: u64) -> Result<GuestAddress, LinkError> {
    Ok(GuestAddress::new(
        address
            .get()
            .checked_add(offset)
            .ok_or(LinkError::AddressOverflow)?,
    ))
}

fn subtract_address(address: GuestAddress, offset: u64) -> Result<GuestAddress, LinkError> {
    Ok(GuestAddress::new(
        address
            .get()
            .checked_sub(offset)
            .ok_or(LinkError::AddressOverflow)?,
    ))
}

fn write_u64(memory: &mut GuestImage, address: GuestAddress, value: u64) -> Result<(), LinkError> {
    memory.write(address, &value.to_le_bytes())?;
    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug)]
struct NativeCxxRuntime {
    cxxabi_handle: usize,
    unwind_handle: usize,
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn native_cxx_runtime() -> Option<NativeCxxRuntime> {
    static RUNTIME: OnceLock<Option<NativeCxxRuntime>> = OnceLock::new();
    *RUNTIME.get_or_init(|| {
        // SAFETY: Both paths are NUL-terminated constants naming macOS shared-cache libraries.
        let cxxabi_handle = unsafe {
            libc::dlopen(
                c"/usr/lib/libc++abi.dylib".as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            )
        };
        // SAFETY: Same as above; the handle is intentionally retained for the process lifetime.
        let unwind_handle = unsafe {
            libc::dlopen(
                c"/usr/lib/system/libunwind.dylib".as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            )
        };
        if cxxabi_handle.is_null() || unwind_handle.is_null() {
            None
        } else {
            Some(NativeCxxRuntime {
                cxxabi_handle: cxxabi_handle as usize,
                unwind_handle: unwind_handle as usize,
            })
        }
    })
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
impl NativeCxxRuntime {
    fn cxxabi_symbol(self, name: &std::ffi::CStr) -> Option<u64> {
        // SAFETY: `cxxabi_handle` is a retained live dlopen handle and `name` is NUL-terminated.
        let address =
            unsafe { libc::dlsym(self.cxxabi_handle as *mut libc::c_void, name.as_ptr()) };
        (!address.is_null()).then_some(address as usize as u64)
    }

    fn unwind_symbol(self, name: &std::ffi::CStr) -> Option<u64> {
        // SAFETY: `unwind_handle` is a retained live dlopen handle and `name` is NUL-terminated.
        let address =
            unsafe { libc::dlsym(self.unwind_handle as *mut libc::c_void, name.as_ptr()) };
        (!address.is_null()).then_some(address as usize as u64)
    }
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn native_cxx_symbol_name(nid: &str) -> Option<&'static std::ffi::CStr> {
    Some(match nid {
        "XwLA5cTHjt4" => c"__gxx_personality_v0",
        "IJKVjsmxxWI" => c"__cxa_allocate_dependent_exception",
        "cfAXurvfl5o" => c"__cxa_allocate_exception",
        "pBxafllkvt0" => c"__cxa_bad_cast",
        "xcc6DTcL8QA" => c"__cxa_bad_typeid",
        "3cUUypQzMiI" => c"__cxa_begin_catch",
        "BxPeH9TTcs4" => c"__cxa_current_exception_type",
        "lX+4FNUklF0" => c"__cxa_end_catch",
        "kBxt5LwtLA4" => c"__cxa_free_dependent_exception",
        "nOIEswYD4Ig" => c"__cxa_free_exception",
        "Y6Sl4Xw7gfA" => c"__cxa_get_exception_ptr",
        "3GPpjQdAMTw" => c"__cxa_guard_acquire",
        "9rAeANT2tyE" => c"__cxa_guard_release",
        "2emaaluWzUw" => c"__cxa_guard_abort",
        "zr094EQ39Ww" => c"__cxa_pure_virtual",
        "ZL9FV4mJXxo" => c"__cxa_rethrow",
        "vkuuLfhnSZI" => c"__cxa_throw",
        "XbAd9Ach9mo" => c"__cxa_throw_bad_array_new_length",
        "cmBRQsLPqwA" => c"__cxa_uncaught_exception",
        "qYhnoevd9bI" => c"_ZSt9terminatev",
        "ozMAr28BwSY" => c"_ZSt14_Xout_of_rangePKc",
        "eT2UsmTewbU" => c"_ZSt11_Xbad_allocv",
        "pZ9WXcClPO8" => c"_ZTVN10__cxxabiv120__si_class_type_infoE",
        "byV+FWlAnB4" => c"_ZTVN10__cxxabiv117__class_type_infoE",
        "tQIo+GIPklo" => c"_ZSt14_Xlength_errorPKc",
        "9ByRMdo7ywg" => c"_ZTVN10__cxxabiv121__vmi_class_type_infoE",
        "V23qt24VPVs" => c"_ZSt17iostream_categoryv",
        "tyHd3P7oDrU" => c"_ZNKSt9exception8_DoraiseEv",
        "Q1BL70XVV0o" => c"_ZSt18uncaught_exceptionv",
        "MELi-cKqWq0" => c"_ZSt19_Xbad_function_callv",
        "NU-T4QowTNA" => c"_ZSt18_Xinvalid_argumentPKc",
        "hMAe+TWS9mQ" => c"__dynamic_cast",
        "apPZ6HKZWaQ" => c"_ZNKSt9exception4whatEv",
        "7P1Wm-5KgAY" => c"_ZNSt12domain_errorD1Ev",
        "MOBxtefPZUg" => c"_ZNSt9exceptionD2Ev",
        "5BIbzIuDxTQ" => c"_ZTISt12domain_error",
        "oAidKrxuUv0" => c"_ZTVSt12domain_error",
        "udTM6Nxx-Ng" => c"_ZTVSt11logic_error",
        "N2f485TmJms" => c"_ZNSt8ios_base7failureD1Ev",
        "oe9tS0VztYk" => c"_ZNSt13runtime_errorD2Ev",
        "47RvLSo2HN8" => c"_ZNSt8bad_castD1Ev",
        "oQDS9nX05Qg" => c"_ZNSt16invalid_argumentD1Ev",
        "Kcb+MNSzZcc" => c"_ZNSt12out_of_rangeD1Ev",
        "BJCgW9-OxLA" => c"_ZTISt8ios_base",
        "Bq8m04PN1zw" => c"_ZTVSt12system_error",
        "-L+-8F0+gBc" => c"_ZTVSt13runtime_error",
        "bLPn1gfqSW8" => c"_ZTISt13runtime_error",
        "qOD-ksTkE08" => c"_ZTISt8bad_cast",
        "tVHE+C8vGXk" => c"_ZTVSt8bad_cast",
        "XZzWt0ygWdw" => c"_ZTISt16invalid_argument",
        "keXoyW-rV-0" => c"_ZTVSt16invalid_argument",
        "dKjhNUf9FBc" => c"_ZTISt12out_of_range",
        "n+aUKkC-3sI" => c"_ZTVSt12out_of_range",
        "n2kx+OmFUis" => c"_ZTISt9exception",
        "BgZcGDh7o9g" => c"_ZNSt9exceptionD1Ev",
        "dCzeFfg9WWI" => c"_ZTVSt9exception",
        _ => return None,
    })
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn native_unwind_symbol_name(nid: &str) -> Option<&'static std::ffi::CStr> {
    Some(match nid {
        "s62MgBhosjU" => c"_Unwind_Backtrace",
        "GslDM6l8E7U" => c"_Unwind_DeleteException",
        "go1LaO7f15k" => c"_Unwind_FindEnclosingFunction",
        "Jhba7poZWZ0" => c"_Unwind_Find_FDE",
        "zs817D8+hPA" => c"_Unwind_ForcedUnwind",
        "uKwYpqeUkzo" => c"_Unwind_GetBSP",
        "MmTshn7gIgU" => c"_Unwind_GetCFA",
        "nqD9bBu-W6U" => c"_Unwind_GetDataRelBase",
        "oQpl5qCT014" => c"_Unwind_GetGR",
        "sETNbyWsEHs" => c"_Unwind_GetIP",
        "5-3wdp3FVtc" => c"_Unwind_GetIPInfo",
        "gpc4YXZffkQ" => c"_Unwind_GetLanguageSpecificData",
        "p3BVovMPEE4" => c"_Unwind_GetRegionStart",
        "HPlmfRhMWwo" => c"_Unwind_GetTextRelBase",
        "wpNJwmDDtxw" => c"_Unwind_RaiseException",
        "f1zwJ3jAI2k" => c"_Unwind_Resume",
        "xUsJSLsdv9I" => c"_Unwind_Resume_or_Rethrow",
        "Xr4pio56GGQ" => c"_Unwind_SetGR",
        "BWAtsvaXToA" => c"_Unwind_SetIP",
        _ => return None,
    })
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn native_libc_symbol_name(nid: &str) -> Option<&'static std::ffi::CStr> {
    Some(match nid {
        "8zTFvBIAIN8" => c"memset",
        "Q3VBxCXhUHs" => c"memcpy",
        "+P6FRGH4LfA" => c"memmove",
        "DfivPArhucg" => c"memcmp",
        "8u8lPzUEq+U" => c"memchr",
        "j4ViWNHEgww" => c"strlen",
        "Ovb2dSJOAuE" => c"strcmp",
        "aesyjrHVWy4" => c"strncmp",
        "kiZSXIWd9vg" => c"strcpy",
        "6sJWiWSRuqk" => c"strncpy",
        "Ls4tzzhimqQ" => c"strcat",
        "AV6ipCNa4Rw" => c"strcasecmp",
        "pXvbDfchu6k" => c"strncasecmp",
        "viiwFMaNamA" => c"strstr",
        "g7zzzLDYGw0" => c"strdup",
        "SfQIZcqvvms" => c"strlcpy",
        "5OqszGpy7Mg" => c"strtoull",
        "VOBg+iNwB-4" => c"strtoll",
        "2vDqwBlpF-o" => c"strtod",
        "dhK16CKwhQg" => c"__isfinited",
        "GfxAp9Xyiqs" => c"__isnand",
        "n88vx3C5nW8" => c"gettimeofday",
        "eLdDw6l0-bU" => c"snprintf",
        "Q2V+iqvjgC0" => c"vsnprintf",
        "tcVi5SivF7Q" => c"sprintf",
        "1Pk0qZQGeWo" => c"sscanf",
        "hcuQgD53UxM" => c"printf",
        "YQ0navp+YIc" => c"puts",
        "HUbZmOnT-Dg" => c"atan2",
        "5ZkEP3Rq7As" => c"cbrt",
        "JBcgYuW8lPU" => c"acos",
        "7Ly52zaL44Q" => c"asin",
        "OXmauLdQ8kY" => c"atan",
        "T7uyNqP7vQA" => c"tan",
        "kn0yiYeExgA" => c"ldexpf",
        "weDug8QD-lE" => c"atanf",
        "wuAQt-j+p4o" => c"exp2f",
        "hsi9drzHR2k" => c"log2f",
        "1D0H2KNjshE" => c"powf",
        "nlaojL9hDtA" => c"round",
        "DDHG1a6+3q0" => c"roundf",
        "9LCjpWyQ5Zc" => c"pow",
        "jMB7EFyu30Y" => c"sincos",
        "pKwslsMUmSk" => c"fmod",
        "RQXLbdT2lc4" => c"logf",
        "pztV4AF18iI" => c"sincosf",
        "H8ya2H00jbI" => c"sin",
        "Q4rRL34CEeE" => c"sinf",
        "rtV7-jWC6Yg" => c"log",
        "WuMbPBKN1TU" => c"log10",
        "2WE3BTYVwKM" => c"cos",
        "Y5DhuDKGlnQ" => c"log2",
        "NVadfnzQhHQ" => c"exp",
        "8zsu04XNsZ4" => c"expf",
        "-P6FNMzk2Kc" => c"cosf",
        _ => return None,
    })
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_pthread_self() -> u64 {
    current_guest_thread_handle()
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_pthread_get_specific(key: u64) -> u64 {
    pthread_get_specific(low_u32(key))
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_guest_mutex_lock(address: u64) -> u64 {
    guest_mutex_lock(address, false)
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_guest_mutex_unlock(address: u64) -> u64 {
    guest_mutex_unlock(address)
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_guest_rwlock_write_lock(address: u64) -> u64 {
    guest_rwlock_lock(address, true, false)
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_guest_rwlock_unlock(address: u64) -> u64 {
    guest_rwlock_unlock(address)
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
extern "C" fn direct_process_time_counter() -> u64 {
    kernel_process_time(true)
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn direct_runtime_import_target(nid: &str) -> Option<u64> {
    Some(match nid {
        // Unreal's archive/name parser calls these leaf TLS and synchronization operations
        // millions of times during startup. They already use dedicated sharded/thread-local
        // state and need neither the process dispatch lock nor a captured eight-XMM import frame.
        "aI+OeCz8xrQ" | "EotR8a3ASf4" => direct_pthread_self as *const () as usize as u64,
        "0-KXaS70xy4" | "eoht7mQOCmo" => direct_pthread_get_specific as *const () as usize as u64,
        "9UK1vLZQft4" | "7H0iTOciTLo" => direct_guest_mutex_lock as *const () as usize as u64,
        "tn3VlD0hG60" | "2Z+PpY6CaJg" => direct_guest_mutex_unlock as *const () as usize as u64,
        "sIlRvQqsN2Y" => direct_guest_rwlock_write_lock as *const () as usize as u64,
        "EgmLo6EWgso" => direct_guest_rwlock_unlock as *const () as usize as u64,
        "EI-5-jlq2dE" => direct_pthread_self as *const () as usize as u64,
        "fgxnMeTNUtY" => direct_process_time_counter as *const () as usize as u64,
        _ => return None,
    })
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn native_import_target(nid: &str) -> Option<u64> {
    if let Some(target) = direct_runtime_import_target(nid) {
        return Some(target);
    }
    let runtime = native_cxx_runtime()?;
    if let Some(name) = native_cxx_symbol_name(nid) {
        return runtime.cxxabi_symbol(name);
    }
    if let Some(name) = native_unwind_symbol_name(nid) {
        return runtime.unwind_symbol(name);
    }
    runtime.cxxabi_symbol(native_libc_symbol_name(nid)?)
}

#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
const fn native_import_target(_nid: &str) -> Option<u64> {
    None
}

extern "C" fn synthetic_data_noop(
    _argument0: u64,
    _argument1: u64,
    _argument2: u64,
    _argument3: u64,
    _argument4: u64,
    _argument5: u64,
) -> u64 {
    0
}

struct SyntheticDataImports {
    _classic_vtable: Box<[u64; 32]>,
    _classic_object: Box<[u64; 8]>,
    classic_symbol: Box<u64>,
    cells: BTreeMap<&'static str, Box<[u64; 32]>>,
}

fn synthetic_data_imports() -> &'static SyntheticDataImports {
    static IMPORTS: OnceLock<SyntheticDataImports> = OnceLock::new();
    IMPORTS.get_or_init(|| {
        let noop = synthetic_data_noop as *const () as usize as u64;
        let mut classic_vtable = Box::new([0_u64; 32]);
        classic_vtable[2..].fill(noop);
        let mut classic_object = Box::new([0_u64; 8]);
        classic_object[0] = classic_vtable.as_ptr() as usize as u64 + 16;
        let classic_symbol = Box::new(classic_object.as_ptr() as usize as u64);

        let mut cells = BTreeMap::new();
        for nid in [
            "H8AprKeZtNg",
            "sBCTjFk7Gi4",
            "yLE5H3058Ao",
            "Cv+zC4EjGMA",
            "H4fcpQOpc08",
            "FQ9NFbBHb5Y",
            "wiR+rIcbnlc",
            "E14mW8pVpoE",
            "1kZFcktOm+s",
            "VmqsS6auJzo",
            "irGo1yaJ-vM",
            "2sWzhYqFH4E",
            "f7uOxY9mM1U",
        ] {
            let mut cell = Box::new([0_u64; 32]);
            if nid == "FQ9NFbBHb5Y" {
                cell[0] = u64::MAX;
            } else if nid == "f7uOxY9mM1U" {
                cell[0] = 0x9e37_79b9_7f4a_7c15;
            } else if matches!(nid, "yLE5H3058Ao" | "1kZFcktOm+s") {
                cell[2..].fill(noop);
            }
            cells.insert(nid, cell);
        }
        SyntheticDataImports {
            _classic_vtable: classic_vtable,
            _classic_object: classic_object,
            classic_symbol,
            cells,
        }
    })
}

fn synthetic_data_import_target(nid: &str) -> Option<u64> {
    let imports = synthetic_data_imports();
    if nid == "Qoo175Ig+-k" {
        return Some((&raw const *imports.classic_symbol) as usize as u64);
    }
    imports
        .cells
        .get(nid)
        .map(|cell| cell.as_ptr() as usize as u64)
}

fn create_import_stubs(
    memory: &mut GuestImage,
    dynamic: &DynamicInfo,
    first_dispatch_index: usize,
    process_exports: &BTreeMap<ProcessSymbolKey, u64>,
) -> Result<(Vec<ImportStub>, BTreeMap<u32, GuestAddress>), LinkError> {
    let mut dispatch_imports = BTreeMap::<ProcessSymbolKey, Vec<u32>>::new();
    for import in &dynamic.imports {
        if classify_import(dynamic, import, process_exports)? != ImportResolutionKind::HleDispatch {
            continue;
        }
        let symbol = imported_symbol(dynamic, import)?;
        dispatch_imports
            .entry(process_symbol_key(dynamic, symbol)?)
            .or_default()
            .push(import.symbol_index);
    }
    if dispatch_imports.is_empty() {
        return Ok((Vec::new(), BTreeMap::new()));
    }

    let stub_bytes = u64::try_from(dispatch_imports.len())
        .map_err(|_| LinkError::StubAllocationOverflow)?
        .checked_mul(IMPORT_STUB_SIZE)
        .ok_or(LinkError::StubAllocationOverflow)?;
    let stub_base =
        memory.allocate_staging(stub_bytes, IMPORT_STUB_SIZE, GUEST_READ | GUEST_EXECUTE)?;
    let bytes_len = usize::try_from(stub_bytes).map_err(|_| LinkError::StubAllocationOverflow)?;
    let mut bytes = vec![0_u8; bytes_len];
    let mut import_stubs = Vec::with_capacity(dispatch_imports.len());
    let mut symbol_stubs = BTreeMap::new();
    let dispatcher = import_dispatch_trampoline as *const () as usize as u64;
    for (index, (key, symbol_indices)) in dispatch_imports.into_iter().enumerate() {
        let byte_offset = index * IMPORT_STUB_SIZE_USIZE;
        let dispatch_index = first_dispatch_index
            .checked_add(index)
            .ok_or(LinkError::StubAllocationOverflow)?;
        write_import_stub(&mut bytes[byte_offset..], dispatch_index, dispatcher)?;
        let address = GuestAddress::new(
            stub_base
                .get()
                .checked_add(byte_offset as u64)
                .ok_or(LinkError::AddressOverflow)?,
        );
        for symbol_index in symbol_indices {
            symbol_stubs.insert(symbol_index, address);
        }
        import_stubs.push(ImportStub {
            nid: key.nid().to_owned(),
            address,
        });
    }
    memory.write(stub_base, &bytes)?;
    Ok((import_stubs, symbol_stubs))
}

fn apply_relocations_at(
    memory: &mut GuestImage,
    image_base: GuestAddress,
    tls_module_id: u64,
    dynamic: &DynamicInfo,
    symbol_stubs: &BTreeMap<u32, GuestAddress>,
    process_exports: &BTreeMap<ProcessSymbolKey, u64>,
) -> Result<RelocationStats, LinkError> {
    let mut stats = RelocationStats::default();
    for relocation in dynamic.relocations.iter().chain(&dynamic.jump_relocations) {
        let target = runtime_address_at(memory, image_base, relocation.offset, 8)?;
        let value = match relocation.relocation_type {
            R_X86_64_RELATIVE => {
                stats.relative += 1;
                image_base.get().wrapping_add_signed(relocation.addend)
            }
            R_X86_64_64 => {
                stats.absolute += 1;
                resolve_symbol_at(
                    memory,
                    image_base,
                    dynamic,
                    symbol_stubs,
                    process_exports,
                    relocation,
                )?
                .wrapping_add_signed(relocation.addend)
            }
            R_X86_64_GLOB_DAT => {
                stats.global_data += 1;
                resolve_symbol_at(
                    memory,
                    image_base,
                    dynamic,
                    symbol_stubs,
                    process_exports,
                    relocation,
                )?
            }
            R_X86_64_JUMP_SLOT => {
                stats.jump_slots += 1;
                resolve_symbol_at(
                    memory,
                    image_base,
                    dynamic,
                    symbol_stubs,
                    process_exports,
                    relocation,
                )?
            }
            R_X86_64_DTPMOD64 => tls_module_id,
            relocation_type => {
                return Err(LinkError::UnsupportedRelocation {
                    relocation_type,
                    offset: relocation.offset,
                });
            }
        };
        memory.write(target, &value.to_le_bytes())?;
    }
    Ok(stats)
}

fn resolve_symbol_at(
    memory: &GuestImage,
    image_base: GuestAddress,
    dynamic: &DynamicInfo,
    symbol_stubs: &BTreeMap<u32, GuestAddress>,
    process_exports: &BTreeMap<ProcessSymbolKey, u64>,
    relocation: &Elf64Rela,
) -> Result<u64, LinkError> {
    let index = usize::try_from(relocation.symbol_index).map_err(|_| LinkError::MissingSymbol {
        index: relocation.symbol_index,
    })?;
    let symbol = dynamic.symbols.get(index).ok_or(LinkError::MissingSymbol {
        index: relocation.symbol_index,
    })?;
    if symbol.is_undefined() {
        let key = process_symbol_key(dynamic, symbol)?;
        let nid = key.nid();
        if let Some(address) = process_exports.get(&key).copied() {
            return Ok(address);
        }
        if let Some(address) = native_import_target(nid) {
            return Ok(address);
        }
        if let Some(address) = synthetic_data_import_target(nid) {
            return Ok(address);
        }
        if symbol.is_weak() {
            return Ok(0);
        }
        return symbol_stubs
            .get(&relocation.symbol_index)
            .map(|address| address.get())
            .ok_or(LinkError::MissingImportStub {
                index: relocation.symbol_index,
            });
    }
    defined_symbol_address_at(memory, image_base, symbol)
}

fn defined_symbol_address_at(
    memory: &GuestImage,
    image_base: GuestAddress,
    symbol: &Elf64Symbol,
) -> Result<u64, LinkError> {
    if memory.contains(symbol.value, 1) {
        return Ok(symbol.value.get());
    }
    image_base
        .get()
        .checked_add(symbol.value.get())
        .ok_or(LinkError::AddressOverflow)
}

fn runtime_address_at(
    memory: &GuestImage,
    image_base: GuestAddress,
    value: GuestAddress,
    length: usize,
) -> Result<GuestAddress, LinkError> {
    if memory.contains(value, length) {
        return Ok(value);
    }
    Ok(GuestAddress::new(
        image_base
            .get()
            .checked_add(value.get())
            .ok_or(LinkError::AddressOverflow)?,
    ))
}

fn write_import_stub(slot: &mut [u8], index: usize, dispatcher: u64) -> Result<(), LinkError> {
    let index = i32::try_from(index).map_err(|_| LinkError::StubAllocationOverflow)?;
    // Push the dispatch index without altering the guest's argument registers, then tail-jump via
    // R11 into the shared frame-capture trampoline. The trampoline removes this word before
    // returning to the original guest call site.
    slot[0] = 0x68;
    slot[1..5].copy_from_slice(&index.to_le_bytes());
    slot[5] = 0x49;
    slot[6] = 0xbb;
    slot[7..15].copy_from_slice(&dispatcher.to_le_bytes());
    slot[15..18].copy_from_slice(&[0x41, 0xff, 0xe3]);
    Ok(())
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
    pthread_keys: BTreeMap<u32, u64>,
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
    thread_dtors_callback: u64,
    application_heap_api_address: u64,
    bundled_modules: Vec<LinkedGuestModule>,
    unwind_modules: Vec<GuestUnwindModule>,
    installed_exception_handlers: BTreeMap<i32, u64>,
    next_ajm_context: u32,
    ajm_contexts: BTreeMap<u32, AjmContextState>,
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
    agc_index_resources: BTreeMap<u64, u64>,
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
    save_root: PathBuf,
    save_data_initialized: bool,
    save_mounts: BTreeMap<String, SaveDataMount>,
    next_save_transaction_resource: u32,
    save_transaction_resources: BTreeSet<u32>,
    prepared_save_resources: BTreeSet<u32>,
    save_transactions: BTreeMap<u32, SaveDataTransaction>,
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
}

#[derive(Clone, Copy, Debug)]
struct AvPlayerEventCallback {
    function: u64,
    object: u64,
    event: u64,
}

#[derive(Debug)]
struct AvPlayerInstance {
    _initialization: Vec<u8>,
    allocator_object: u64,
    allocate_texture_callback: u64,
    event_object: u64,
    event_callback: u64,
    source: Option<String>,
    auto_start: bool,
    width: u32,
    height: u32,
    frames_per_second: f64,
    duration: Duration,
    has_audio: bool,
    paused: bool,
    end_of_stream: bool,
    playing_since: Option<Instant>,
    elapsed: Duration,
    looping: bool,
    video_decoder: Option<Child>,
    video_output: Option<ChildStdout>,
    raw_video_frame: Vec<u8>,
    padded_video_frame: Vec<u8>,
    video_buffer_base: u64,
    video_buffers: [u64; 3],
    video_buffer_stride: usize,
    next_video_buffer: usize,
    next_video_frame: u64,
    /// Completed VideoOut flip and decoded-frame indices at the start of the current media-clock
    /// segment. Native AVPlayer does not release every decoded frame merely because the guest
    /// polls quickly; tying the clock to retired scanout work also keeps its three buffers from
    /// being overwritten while the emulated GPU is still consuming them.
    video_clock_origin: Option<(u64, u64)>,
    /// Host media-clock segment established only after the guest has consumed its first decoded
    /// frame. This prevents a long initial shader compile from seeking a short splash straight to
    /// EOF, while still allowing later frames to catch up when emulation runs below movie FPS.
    video_catch_up_origin: Option<(Instant, u64)>,
    audio_decoder: Option<Child>,
    audio_output: Option<ChildStdout>,
    raw_audio_frame: Vec<u8>,
    audio_buffer_base: u64,
    next_audio_buffer: usize,
    next_audio_frame: u64,
}

impl AvPlayerInstance {
    fn stop_decoders(&mut self) {
        self.video_output = None;
        self.audio_output = None;
        if let Some(mut decoder) = self.video_decoder.take() {
            let _ = decoder.kill();
            let _ = decoder.wait();
        }
        if let Some(mut decoder) = self.audio_decoder.take() {
            let _ = decoder.kill();
            let _ = decoder.wait();
        }
        self.playing_since = None;
    }

    fn reset_playback(&mut self) {
        self.stop_decoders();
        self.elapsed = Duration::ZERO;
        self.end_of_stream = false;
        self.next_video_frame = 0;
        self.video_clock_origin = None;
        self.video_catch_up_origin = None;
        self.next_audio_frame = 0;
        self.next_video_buffer = 0;
        self.next_audio_buffer = 0;
    }

    fn playback_elapsed(&self) -> Duration {
        self.playing_since
            .map_or(self.elapsed, |started| self.elapsed + started.elapsed())
    }

    fn decoded_playback_elapsed(&self) -> Duration {
        if self.source.is_none() {
            return self.playback_elapsed();
        }
        let video = (self.frames_per_second.is_finite() && self.frames_per_second > 0.0)
            .then(|| Duration::from_secs_f64(self.next_video_frame as f64 / self.frames_per_second))
            .unwrap_or_default();
        let audio = Duration::from_secs_f64(self.next_audio_frame as f64 * 1_024.0 / 48_000.0);
        video.max(audio)
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
        // SAFETY: Both ranges are disjoint, live allocations within the caller-provided mspace.
        unsafe {
            std::ptr::copy_nonoverlapping(
                address as *const u8,
                replacement as *mut u8,
                usize::try_from(old_size).ok()?,
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
    /// Queue completions recorded into this buffer, announced when it is submitted.
    pending_completions: Vec<AmprCompletion>,
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
    pad: InitializationState,
    pad_handle: Option<i32>,
    next_pad_handle: i32,
    common_dialog: InitializationState,
    common_dialog_owner: Option<CommonDialogOwner>,
    /// Lifecycle state of the system message dialog, using the common-dialog status values.
    message_dialog_status: u32,
    play_go_initialized: bool,
    play_go_open: bool,
    save_data_dialog_status: i32,
    agc_states: BTreeMap<u64, u32>,
    game_update: InitializationState,
    np_entitlement_access: Option<(u64, u64)>,
    np_session_signaling: Option<Vec<u8>>,
    next_np_web_api2_user_context: i32,
    np_web_api2_user_contexts: BTreeMap<i32, (i32, i32)>,
    np_callback_polls: u64,
    np_title_id: Option<[u8; 16]>,
    share_feature_permitted: bool,
    share_content_parameter: Option<String>,
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
    completed_flip_count: u64,
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
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ShaderEvaluationCacheKey {
    program_address: u64,
    user_data: ShaderEvaluationUserDataKey,
    compute_system_registers: Option<remu_shader::ComputeSystemRegisters>,
    user_data_scalar_register_base: u32,
    options: remu_shader::EvaluationOptions,
}

#[derive(Clone, Debug)]
struct ShaderEvaluationUserDataKey {
    words: [u32; 64],
    len: u8,
}

impl ShaderEvaluationUserDataKey {
    fn as_slice(&self) -> &[u32] {
        &self.words[..usize::from(self.len)]
    }
}

impl PartialEq for ShaderEvaluationUserDataKey {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.as_slice() == other.as_slice()
    }
}

impl Eq for ShaderEvaluationUserDataKey {}

impl Hash for ShaderEvaluationUserDataKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len.hash(state);
        self.as_slice().hash(state);
    }
}

// Shader evaluation keys are entirely emulator-owned integers. SipHash's collision resistance is
// unnecessary here and is costly because the hot miss path hashes up to 64 user-data words in as
// many as three cache tables. This small Fx-style mixer retains full equality checks in HashMap
// while making those repeated fixed-width writes substantially cheaper.
#[derive(Debug)]
struct ShaderEvaluationCacheHasher(u64);

impl Default for ShaderEvaluationCacheHasher {
    fn default() -> Self {
        Self(0x517c_c1b7_2722_0a95)
    }
}

impl ShaderEvaluationCacheHasher {
    #[inline]
    fn mix(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for ShaderEvaluationCacheHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let mut word = [0_u8; 8];
            word.copy_from_slice(chunk);
            self.mix(u64::from_ne_bytes(word));
        }
        let remainder = chunks.remainder();
        if !remainder.is_empty() {
            let mut word = [0_u8; 8];
            word[..remainder.len()].copy_from_slice(remainder);
            self.mix(u64::from_ne_bytes(word) ^ ((remainder.len() as u64) << 56));
        }
    }

    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.mix(u64::from(value));
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.mix(u64::from(value));
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.mix(value);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.mix(value as u64);
    }
}

type ShaderEvaluationCacheBuildHasher = BuildHasherDefault<ShaderEvaluationCacheHasher>;

type ShaderEvaluationCacheValue = Result<Arc<remu_shader::ShaderEvaluation>, Arc<str>>;

fn retire_shader_evaluation(value: ShaderEvaluationCacheValue) {
    static RETIRER: OnceLock<std::sync::mpsc::Sender<ShaderEvaluationCacheValue>> = OnceLock::new();
    let sender = RETIRER.get_or_init(|| {
        let (sender, receiver) = std::sync::mpsc::channel::<ShaderEvaluationCacheValue>();
        if let Err(error) = std::thread::Builder::new()
            .name("REMu shader cache reclaimer".into())
            .spawn(move || {
                while let Ok(value) = receiver.recv() {
                    drop(value);
                }
            })
        {
            eprintln!("[REMu][AGC] could not start shader cache reclaimer: {error}");
        }
        sender
    });
    if let Err(error) = sender.send(value) {
        drop(error.0);
    }
}

#[derive(Debug)]
struct ShaderEvaluationCacheEntry {
    value: ShaderEvaluationCacheValue,
    admission_id: u64,
}

type ShaderEvaluationCacheEntries =
    HashMap<ShaderEvaluationCacheKey, ShaderEvaluationCacheEntry, ShaderEvaluationCacheBuildHasher>;
type ShaderEvaluationPendingEntries =
    HashMap<ShaderEvaluationCacheKey, u64, ShaderEvaluationCacheBuildHasher>;

#[derive(Debug, Default)]
struct ShaderEvaluationCache {
    entries: ShaderEvaluationCacheEntries,
    seen_once: ShaderEvaluationPendingEntries,
    pending_admission_order: VecDeque<(ShaderEvaluationCacheKey, u64)>,
    admission_order: VecDeque<(ShaderEvaluationCacheKey, u64)>,
    next_admission_id: u64,
    hits: u64,
    misses: u64,
    failures: u64,
    invalidations: u64,
    evictions: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct ShaderEvaluationCacheStats {
    entries: usize,
    pending_admissions: usize,
    hits: u64,
    misses: u64,
    failures: u64,
    invalidations: u64,
    evictions: u64,
}

impl ShaderEvaluationCache {
    fn stats(&self) -> ShaderEvaluationCacheStats {
        ShaderEvaluationCacheStats {
            entries: self.entries.len(),
            pending_admissions: self.seen_once.len(),
            hits: self.hits,
            misses: self.misses,
            failures: self.failures,
            invalidations: self.invalidations,
            evictions: self.evictions,
        }
    }
}

impl ShaderEvaluationCacheStats {
    fn merge(self, other: Self) -> Self {
        Self {
            entries: self.entries.saturating_add(other.entries),
            pending_admissions: self
                .pending_admissions
                .saturating_add(other.pending_admissions),
            hits: self.hits.saturating_add(other.hits),
            misses: self.misses.saturating_add(other.misses),
            failures: self.failures.saturating_add(other.failures),
            invalidations: self.invalidations.saturating_add(other.invalidations),
            evictions: self.evictions.saturating_add(other.evictions),
        }
    }
}

struct ShaderEvaluationWorkerRequest {
    state: remu_shader::ShaderState,
    options: remu_shader::EvaluationOptions,
}

struct ShaderEvaluationWorkerResult {
    evaluation: Result<Arc<remu_shader::ShaderEvaluation>, String>,
    stats: ShaderEvaluationCacheStats,
}

#[derive(Debug)]
struct ShaderEvaluationWorker {
    requests: std::sync::mpsc::SyncSender<ShaderEvaluationWorkerRequest>,
    results: Mutex<std::sync::mpsc::Receiver<ShaderEvaluationWorkerResult>>,
    stats: ShaderEvaluationCacheStats,
}

impl ShaderEvaluationWorker {
    fn spawn(name: &str) -> Result<Self, String> {
        let (requests, request_receiver) =
            std::sync::mpsc::sync_channel::<ShaderEvaluationWorkerRequest>(1);
        let (result_sender, results) =
            std::sync::mpsc::sync_channel::<ShaderEvaluationWorkerResult>(1);
        std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                let mut cache = ShaderEvaluationCache::default();
                while let Ok(request) = request_receiver.recv() {
                    let evaluation = cache.evaluate(&request.state, request.options);
                    if result_sender
                        .send(ShaderEvaluationWorkerResult {
                            evaluation,
                            stats: cache.stats(),
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .map_err(|error| format!("could not start {name}: {error}"))?;
        Ok(Self {
            requests,
            results: Mutex::new(results),
            stats: ShaderEvaluationCacheStats::default(),
        })
    }

    fn submit(&self, request: ShaderEvaluationWorkerRequest) -> Result<(), String> {
        self.requests
            .send(request)
            .map_err(|_| "shader evaluation worker stopped".to_owned())
    }

    fn receive(&mut self) -> Result<Arc<remu_shader::ShaderEvaluation>, String> {
        let result = self
            .results
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .recv()
            .map_err(|_| "shader evaluation worker stopped".to_owned())?;
        self.stats = result.stats;
        result.evaluation
    }
}

#[derive(Debug)]
struct ShaderEvaluationWorkers {
    parallel: ShaderEvaluationWorker,
}

impl ShaderEvaluationWorkers {
    fn new() -> Result<Self, String> {
        Ok(Self {
            parallel: ShaderEvaluationWorker::spawn("REMu shader evaluator")?,
        })
    }

    fn stats(&self) -> ShaderEvaluationCacheStats {
        self.parallel.stats
    }
}

fn canonical_vertex_record_count(record_count: Option<u32>) -> Option<u32> {
    record_count.map(|count| count.max(1).checked_next_power_of_two().unwrap_or(u32::MAX))
}

fn shader_evaluation_user_data_key(
    state: &remu_shader::ShaderState,
) -> ShaderEvaluationUserDataKey {
    let scalar_base = state.user_data_scalar_register_base;
    let mut words = [0_u32; 64];
    let mut len = 0_usize;
    for user_data_index in state
        .program
        .input_scalar_registers()
        .iter()
        .copied()
        .filter_map(|register| register.checked_sub(scalar_base))
        .filter(|offset| *offset < 64)
    {
        let user_data_index = usize::try_from(user_data_index).unwrap_or(usize::MAX);
        words[len] = state
            .user_data
            .get(user_data_index)
            .copied()
            .unwrap_or_default();
        len += 1;
    }
    ShaderEvaluationUserDataKey {
        words,
        len: u8::try_from(len).expect("hardware user data has at most 64 scalar registers"),
    }
}

fn shader_evaluation_cache_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var_os("REMU_DISABLE_SHADER_EVALUATION_CACHE").is_some_and(|value| value == "1")
    })
}

impl ShaderEvaluationCache {
    fn evaluate(
        &mut self,
        state: &remu_shader::ShaderState,
        mut options: remu_shader::EvaluationOptions,
    ) -> Result<Arc<remu_shader::ShaderEvaluation>, String> {
        // Draws using the same resources commonly differ only by a few indices. Binding the next
        // power-of-two record range is still bounded by the guest descriptor, yet lets those draws
        // share one evaluated shader instead of continuously rotating this cache.
        options.required_vertex_record_count =
            canonical_vertex_record_count(options.required_vertex_record_count);
        if shader_evaluation_cache_disabled() {
            self.misses = self.misses.saturating_add(1);
            return remu_shader::evaluate_shader(&ProcessGuestMemory, state, options)
                .map(Arc::new)
                .map_err(|error| {
                    self.failures = self.failures.saturating_add(1);
                    format!("evaluation-error({error})")
                });
        }
        let key = ShaderEvaluationCacheKey {
            program_address: state.program.address,
            // Only scalar inputs actually referenced by the decoded program can affect control
            // flow, resource discovery, or the generated Metal shader. Unity updates unused
            // hardware user-data slots extremely often, so including them here caused false
            // misses and expensive whole-generation cache destruction.
            user_data: shader_evaluation_user_data_key(state),
            compute_system_registers: state.compute_system_registers,
            user_data_scalar_register_base: state.user_data_scalar_register_base,
            options,
        };
        if let Some(cached) = self.entries.get(&key).map(|entry| entry.value.clone()) {
            if cached
                .as_ref()
                .is_ok_and(|evaluation| shader_evaluation_dependencies_are_current(evaluation))
            {
                self.hits = self.hits.saturating_add(1);
                return cached.map_err(|error| error.to_string());
            }
            // The pointer-valued user data can be identical while the guest rewrites the image,
            // sampler, or buffer descriptor behind it. A stale evaluation is worse than a miss:
            // it binds unrelated textures and can turn a complete frame into partial geometry.
            if let Some(retired) = self.entries.remove(&key) {
                retire_shader_evaluation(retired.value);
            }
            self.invalidations = self.invalidations.saturating_add(1);
        }
        self.misses = self.misses.saturating_add(1);
        let evaluated = remu_shader::evaluate_shader(&ProcessGuestMemory, state, options)
            .map(Arc::new)
            .map_err(|error| Arc::<str>::from(format!("evaluation-error({error})")));
        if evaluated.is_err() {
            self.failures = self.failures.saturating_add(1);
        }
        // Most Unity resource states are used by one draw only. Admit only successful states that
        // have been observed twice. Once the cache is full, evict one oldest live admission for
        // each replacement.
        // Whole-generation replacement caused large object graphs to retire together, visibly
        // stalling guest submission and starving audio for several frames.
        // Evaluation failures often describe a transient null descriptor while Unity is
        // populating a command buffer. Never make such a failure persistent for the same pointer
        // state; only successful evaluations participate in the two-hit admission filter.
        let admit = if evaluated.is_ok() {
            if self.seen_once.remove(&key).is_some() {
                true
            } else {
                while self.pending_admission_order.len() >= SHADER_EVALUATION_PENDING_CAPACITY {
                    let Some((old_key, admission_id)) = self.pending_admission_order.pop_front()
                    else {
                        break;
                    };
                    let is_current = self
                        .seen_once
                        .get(&old_key)
                        .is_some_and(|current| *current == admission_id);
                    if is_current {
                        self.seen_once.remove(&old_key);
                    }
                }
                let admission_id = self.next_admission_id;
                self.next_admission_id = self.next_admission_id.wrapping_add(1);
                self.seen_once.insert(key.clone(), admission_id);
                self.pending_admission_order
                    .push_back((key.clone(), admission_id));
                false
            }
        } else {
            false
        };
        if admit {
            self.seen_once.remove(&key);
            while self.entries.len() >= SHADER_EVALUATION_CACHE_CAPACITY {
                let Some((old_key, admission_id)) = self.admission_order.pop_front() else {
                    break;
                };
                let is_current = self
                    .entries
                    .get(&old_key)
                    .is_some_and(|entry| entry.admission_id == admission_id);
                if is_current {
                    if let Some(retired) = self.entries.remove(&old_key) {
                        retire_shader_evaluation(retired.value);
                        self.evictions = self.evictions.saturating_add(1);
                    }
                }
            }
            let admission_id = self.next_admission_id;
            self.next_admission_id = self.next_admission_id.wrapping_add(1);
            self.entries.insert(
                key.clone(),
                ShaderEvaluationCacheEntry {
                    value: evaluated.clone(),
                    admission_id,
                },
            );
            self.admission_order.push_back((key, admission_id));
        }
        evaluated.map_err(|error| error.to_string())
    }
}

fn shader_evaluation_dependencies_are_current(evaluation: &remu_shader::ShaderEvaluation) -> bool {
    #[cfg(target_os = "macos")]
    {
        // Dependencies are address-sorted by the evaluator. Reuse the containing readable range
        // for every following word instead of performing a TLS generation check and binary search
        // for each four-byte scalar load on every cache hit.
        let mut readable_end = 0_u64;
        for &(address, expected) in &evaluation.scalar_memory_reads {
            let Some(word_end) = address.checked_add(4) else {
                return false;
            };
            if word_end > readable_end {
                let Some(mapped_end) = cached_kernel_memory_readable_range_end(address) else {
                    return false;
                };
                if word_end > mapped_end {
                    return false;
                }
                readable_end = mapped_end;
            }
            // SAFETY: guest mapping mutations are serialized by the process dispatch lock. The
            // cached range snapshot proves this identity-mapped word readable for this dispatch.
            let current = unsafe { (address as *const u32).read_unaligned() };
            if u32::from_le(current) != expected {
                return false;
            }
        }
        true
    }
    #[cfg(not(target_os = "macos"))]
    {
        evaluation
            .scalar_memory_reads
            .iter()
            .all(|&(address, expected)| {
                let mut bytes = [0_u8; 4];
                ProcessGuestMemory.read(address, &mut bytes)
                    && u32::from_le_bytes(bytes) == expected
            })
    }
}

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
    compute_indirect_args_address: u64,
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
    snapshots: BTreeMap<u64, remu_gpu::ShaderSnapshot>,
    decode_failures: BTreeSet<u64>,
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
        self.compute_indirect_args_address = 0;
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
}

const KERNEL_EVENT_FILTER_GRAPHICS: i16 = -14;
const KERNEL_EVENT_FILTER_USER: i16 = -11;
/// Gen5 AP/AMPR command-buffer completion. The Gen4 filter set ends at the high-resolution timer,
/// so the console assigns this newer source the next identifier below it.
const KERNEL_EVENT_FILTER_AMPR: i16 = -16;
/// Display output events, including mode changes and flips.
const KERNEL_EVENT_FILTER_VIDEO_OUT: i16 = -13;

static PROBE_DISPATCH: OnceLock<RwLock<ProbeDispatchState>> = OnceLock::new();
/// Serializes APR/AMPR file and command-buffer operations in guest-observable order.
///
/// A file read deliberately releases `PROBE_DISPATCH` while the host blocks in `pread` and while
/// bytes are copied into guest memory. Holding this narrower lock across that gap prevents submit,
/// reset, destruction, or file re-resolution from overtaking the transfer.
static AMPR_IMPORT_ORDER: OnceLock<Mutex<()>> = OnceLock::new();
/// Reusable host storage for ordered AMPR reads.
///
/// Silent Hill issues thousands of mostly 32–256 KiB reads while opening Unreal containers. AMPR
/// operations are already serialized by `AMPR_IMPORT_ORDER`, so retaining one initialized buffer
/// avoids an allocation and zero-fill for every request without increasing the observable
/// concurrency or bypassing CPU write-watch faults during the final guest-memory copy.
static AMPR_READ_STAGING: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static KERNEL_EQUEUE_WAKE: OnceLock<(Mutex<u64>, Condvar)> = OnceLock::new();
static AUDIO_OUT: OnceLock<Mutex<audio::AudioOutState>> = OnceLock::new();
static AUDIO_OUT2: OnceLock<Mutex<AudioOut2State>> = OnceLock::new();
static KERNEL_FILES: OnceLock<Mutex<KernelFileState>> = OnceLock::new();
static PROCESS_CLOCK_EPOCH: OnceLock<Instant> = OnceLock::new();
static PROCESS_START_NANOS: AtomicU64 = AtomicU64::new(0);
static NEXT_NET_POOL_HANDLE: AtomicUsize = AtomicUsize::new(1);
static NEXT_SSL_CONTEXT: AtomicUsize = AtomicUsize::new(1);
static NEXT_HTTP_CONTEXT: AtomicUsize = AtomicUsize::new(1);
static NEXT_NP_WEB_API2_CONTEXT: AtomicUsize = AtomicUsize::new(1);
static DISPATCH_LOCK_STALL_LOGS: AtomicUsize = AtomicUsize::new(0);
static GUEST_EXCEPTION_DELIVERY: OnceLock<Mutex<GuestExceptionDeliveryState>> = OnceLock::new();

#[derive(Debug, Default)]
struct AudioOut2State {
    next_handle: u64,
    contexts: BTreeMap<u64, AudioOut2ContextState>,
    users: BTreeSet<u64>,
    ports: BTreeMap<u64, AudioOut2PortRuntimeState>,
    mastering_parameters: u64,
    user_creates: u64,
    user_destroys: u64,
    port_creates: u64,
    port_destroys: u64,
    queue_level_queries: u64,
    speaker_queries: u64,
    attribute_updates: u64,
    port_state_queries: u64,
    context_advances: u64,
    context_pushes: u64,
    #[cfg(test)]
    submitted_pcm: Vec<(u64, Vec<u8>)>,
}

#[derive(Clone, Copy, Debug)]
struct AudioOut2ContextState {
    queue_depth: u32,
    grain_frames: u32,
    queued: u32,
    last_queue_update: Instant,
    memory_address: u64,
    memory_size: u64,
}

#[derive(Debug)]
struct KernelFileState {
    next_fd: i32,
    files: BTreeMap<i32, std::fs::File>,
    directories: BTreeMap<i32, KernelDirectory>,
    next_file_number: u32,
}

impl Default for KernelFileState {
    fn default() -> Self {
        Self {
            next_fd: 3,
            files: BTreeMap::new(),
            directories: BTreeMap::new(),
            next_file_number: 1,
        }
    }
}

/// A directory opened by the guest, holding its entries already packed as console dirent records.
///
/// The listing is captured when the directory is opened so repeated reads see one stable
/// enumeration, matching a guest that walks a directory while writing files elsewhere.
#[derive(Debug, Default)]
struct KernelDirectory {
    records: Vec<u8>,
    offset: usize,
}

/// Bytes of fixed dirent header before the name: `d_fileno`, `d_reclen`, `d_type`, `d_namlen`.
const DIRENT_HEADER_SIZE: usize = 8;
/// Console dirent records never straddle this boundary, so a reader can consume whole blocks.
const DIRENT_BLOCK_SIZE: usize = 512;
const DIRENT_TYPE_DIRECTORY: u8 = 4;
const DIRENT_TYPE_FILE: u8 = 8;

impl KernelDirectory {
    /// Packs one directory listing into console dirent records.
    ///
    /// Each record is four-byte aligned, and a record that would cross a 512-byte block is pushed
    /// to the next block by extending the previous record's length. Readers rely on that
    /// invariant to consume a block at a time without partially decoding a trailing record.
    fn new(entries: &[(String, bool)], next_file_number: &mut u32) -> Self {
        let mut records: Vec<u8> = Vec::new();
        let mut last_record_start: Option<usize> = None;
        for (name, is_directory) in entries {
            let name_bytes = name.as_bytes();
            let Ok(name_length) = u8::try_from(name_bytes.len()) else {
                // The console's dirent name length is a single byte; longer names cannot be
                // represented and are skipped rather than truncated into a different name.
                continue;
            };
            let record_length = (DIRENT_HEADER_SIZE + name_bytes.len() + 1).next_multiple_of(4);
            let block_remaining = DIRENT_BLOCK_SIZE - records.len() % DIRENT_BLOCK_SIZE;
            if record_length > block_remaining {
                // Extend the previous record over the padding so the block stays fully described.
                if let Some(start) = last_record_start {
                    let previous = u16::from_le_bytes([records[start + 4], records[start + 5]]);
                    let extended =
                        previous.saturating_add(u16::try_from(block_remaining).unwrap_or(u16::MAX));
                    records[start + 4..start + 6].copy_from_slice(&extended.to_le_bytes());
                }
                records.resize(records.len() + block_remaining, 0);
            }
            let file_number = *next_file_number;
            *next_file_number = next_file_number.wrapping_add(1).max(1);
            last_record_start = Some(records.len());
            records.extend_from_slice(&file_number.to_le_bytes());
            records.extend_from_slice(
                &u16::try_from(record_length)
                    .unwrap_or(u16::MAX)
                    .to_le_bytes(),
            );
            records.push(if *is_directory {
                DIRENT_TYPE_DIRECTORY
            } else {
                DIRENT_TYPE_FILE
            });
            records.push(name_length);
            records.extend_from_slice(name_bytes);
            records.resize(last_record_start.unwrap_or(0) + record_length, 0);
        }
        if !records.is_empty() {
            // Pad the final block so every returned read is a whole number of blocks.
            let tail = records.len() % DIRENT_BLOCK_SIZE;
            if tail != 0 {
                let padding = DIRENT_BLOCK_SIZE - tail;
                if let Some(start) = last_record_start {
                    let previous = u16::from_le_bytes([records[start + 4], records[start + 5]]);
                    let extended =
                        previous.saturating_add(u16::try_from(padding).unwrap_or(u16::MAX));
                    records[start + 4..start + 6].copy_from_slice(&extended.to_le_bytes());
                }
                records.resize(records.len() + padding, 0);
            }
        }
        Self { records, offset: 0 }
    }
}

#[derive(Clone, Copy, Debug)]
struct AudioOut2PortRuntimeState {
    context: u64,
    backend_handle: Option<u64>,
    buffer_address: u64,
    channel_count: u8,
    gains: [f32; 8],
    parameter: SceAudioOut2PortParam,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct SceAudioOut2ContextParam {
    max_ports: u32,
    max_object_ports: u32,
    guarantee_object_ports: u32,
    queue_depth: u32,
    num_grains: u32,
    flags: u32,
    reserved: [u32; 10],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct SceAudioOut2PortParam {
    port_type: u16,
    pad: u16,
    data_format: u32,
    sampling_frequency: u32,
    flags: u32,
    user_handle: u64,
    reserved: [u32; 10],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct SceAudioOut2Attribute {
    attribute_id: u32,
    reserved: i32,
    value: u64,
    value_size: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct SceAudioOut2Pcm {
    data: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct SceAudioOut2PortState {
    output: u16,
    num_channels: u8,
    pad1: u8,
    volume: i16,
    reroute_counter: u16,
    flags: u32,
    pad2: u32,
    reserved: [u64; 6],
}

const _: () = {
    assert!(std::mem::size_of::<SceAudioOut2ContextParam>() == 0x40);
    assert!(std::mem::offset_of!(SceAudioOut2ContextParam, max_ports) == 0x00);
    assert!(std::mem::offset_of!(SceAudioOut2ContextParam, max_object_ports) == 0x04);
    assert!(std::mem::offset_of!(SceAudioOut2ContextParam, guarantee_object_ports) == 0x08);
    assert!(std::mem::offset_of!(SceAudioOut2ContextParam, queue_depth) == 0x0c);
    assert!(std::mem::offset_of!(SceAudioOut2ContextParam, num_grains) == 0x10);
    assert!(std::mem::offset_of!(SceAudioOut2ContextParam, flags) == 0x14);

    assert!(std::mem::size_of::<SceAudioOut2PortParam>() == 0x40);
    assert!(std::mem::offset_of!(SceAudioOut2PortParam, port_type) == 0x00);
    assert!(std::mem::offset_of!(SceAudioOut2PortParam, data_format) == 0x04);
    assert!(std::mem::offset_of!(SceAudioOut2PortParam, sampling_frequency) == 0x08);
    assert!(std::mem::offset_of!(SceAudioOut2PortParam, flags) == 0x0c);
    assert!(std::mem::offset_of!(SceAudioOut2PortParam, user_handle) == 0x10);

    assert!(std::mem::size_of::<SceAudioOut2Attribute>() == 0x18);
    assert!(std::mem::offset_of!(SceAudioOut2Attribute, attribute_id) == 0x00);
    assert!(std::mem::offset_of!(SceAudioOut2Attribute, reserved) == 0x04);
    assert!(std::mem::offset_of!(SceAudioOut2Attribute, value) == 0x08);
    assert!(std::mem::offset_of!(SceAudioOut2Attribute, value_size) == 0x10);

    assert!(std::mem::size_of::<SceAudioOut2Pcm>() == 0x08);
    assert!(std::mem::offset_of!(SceAudioOut2Pcm, data) == 0x00);

    assert!(std::mem::size_of::<SceAudioOut2PortState>() == 0x40);
    assert!(std::mem::offset_of!(SceAudioOut2PortState, output) == 0x00);
    assert!(std::mem::offset_of!(SceAudioOut2PortState, num_channels) == 0x02);
    assert!(std::mem::offset_of!(SceAudioOut2PortState, volume) == 0x04);
    assert!(std::mem::offset_of!(SceAudioOut2PortState, reroute_counter) == 0x06);
    assert!(std::mem::offset_of!(SceAudioOut2PortState, flags) == 0x08);
    assert!(std::mem::offset_of!(SceAudioOut2PortState, reserved) == 0x10);
};

const AUDIO_OUT2_CONTEXT_PARAM_SIZE: u64 = std::mem::size_of::<SceAudioOut2ContextParam>() as u64;
// `sceAudioOut2ContextQueryMemory` writes one `size_t`. Silent Hill's Wwise wrapper places that
// qword immediately before its 0x40-byte context parameter and reads it back as the allocation
// size after the call. SharpEmu currently writes a larger private memory-info guess here, but doing
// that overwrites the live parameter in this shipped caller.
const AUDIO_OUT2_CONTEXT_MEMORY_BASE_SIZE: u64 = 0x1_0000;
const AUDIO_OUT2_CONTEXT_MEMORY_QUEUE_STRIDE: u64 = 0x590;
const AUDIO_OUT2_DEFAULT_QUEUE_DEPTH: u32 = 4;
const AUDIO_OUT2_DEFAULT_GRAIN_FRAMES: u32 = 0x200;
const AUDIO_OUT2_MAX_QUEUE_DEPTH: u32 = 64;
/// Bytes one `SceAudioOut2PortState` occupies.
///
/// Silent Hill has two call sites that reserve and consume the full 0x40 bytes. One copies the
/// result into a long-lived Wwise object with two 32-byte vector loads, so leaving the upper half
/// untouched launders stale stack data into live device state.
const AUDIO_OUT2_PORT_STATE_SIZE: u64 = std::mem::size_of::<SceAudioOut2PortState>() as u64;
/// Bytes one `SceAudioOut2SpeakerInfo` occupies.
///
/// The Wwise caller reserves exactly 0x40 bytes at `rbp - 0x80`. Clearing the previous 0x50-byte
/// guess overwrote the next sixteen bytes of its stack frame.
const AUDIO_OUT2_SPEAKER_INFO_SIZE: u64 = 0x40;
const AUDIO_OUT2_PORT_TYPE_VIBRATION: u16 = 6;
const SCE_AUDIO_OUT2_ERROR_NOT_READY: u32 = 0x8026_8008;

const fn decode_audio_out2_channel_format(format: u32) -> Option<(u8, u64)> {
    match format {
        // AudioOut2's create descriptor encodes the channel count in this format word. Map it to
        // the corresponding floating-point ordinary AudioOut format used by the host mixer.
        0x100 => Some((1, 3)),
        0x200 => Some((2, 4)),
        0x880 => Some((8, 5)),
        _ => None,
    }
}

fn read_audio_out2_context_param(address: u64) -> SceAudioOut2ContextParam {
    // SAFETY: callers validate a complete readable 0x40-byte parameter object first.
    unsafe { std::ptr::read_unaligned(address as *const SceAudioOut2ContextParam) }
}

fn read_audio_out2_port_param(address: u64) -> SceAudioOut2PortParam {
    // SAFETY: callers validate a complete readable 0x40-byte parameter object first.
    unsafe { std::ptr::read_unaligned(address as *const SceAudioOut2PortParam) }
}

fn decode_audio_out2_attribute(address: u64) -> SceAudioOut2Attribute {
    // SAFETY: callers validate the complete attribute array before decoding an entry.
    unsafe { std::ptr::read_unaligned(address as *const SceAudioOut2Attribute) }
}

fn write_audio_out2_context_param(address: u64, parameter: SceAudioOut2ContextParam) {
    // SAFETY: callers validate one complete writable context-parameter object first.
    unsafe {
        std::ptr::write_unaligned(address as *mut SceAudioOut2ContextParam, parameter);
    }
}

fn write_audio_out2_port_state(address: u64, port_state: SceAudioOut2PortState) {
    // SAFETY: callers validate one complete writable port-state object first.
    unsafe {
        std::ptr::write_unaligned(address as *mut SceAudioOut2PortState, port_state);
    }
}

fn audio_out2_grain_duration(grain_frames: u32) -> Duration {
    Duration::from_nanos(u64::from(grain_frames) * 1_000_000_000 / 48_000)
}

fn audio_out2_update_context_queue(context: &mut AudioOut2ContextState, now: Instant) {
    if context.queued == 0 {
        context.last_queue_update = now;
        return;
    }
    let grain_duration = audio_out2_grain_duration(context.grain_frames);
    let elapsed = now.saturating_duration_since(context.last_queue_update);
    let drained = u32::try_from(
        (elapsed.as_nanos() / grain_duration.as_nanos()).min(u128::from(context.queued)),
    )
    .expect("the drained AudioOut2 queue count is bounded by u32");
    if drained == 0 {
        return;
    }
    context.queued -= drained;
    context.last_queue_update += grain_duration
        .checked_mul(drained)
        .expect("a bounded AudioOut2 queue duration cannot overflow");
    if context.queued == 0 {
        context.last_queue_update = now;
    }
}

fn audio_out2_queue_wait_duration(context: &AudioOut2ContextState, now: Instant) -> Duration {
    let grain_duration = audio_out2_grain_duration(context.grain_frames);
    grain_duration.saturating_sub(now.saturating_duration_since(context.last_queue_update))
}

fn trace_audio_out2_memory() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_AUDIO2_MEMORY").is_some_and(|value| value == "1")
    })
}

fn trace_audio_out2_context_memory(context: AudioOut2ContextState, push: u64, channels: &[u8]) {
    const PAGE_SIZE: u64 = 0x1000;
    const MAX_POINTERS: usize = 24;
    let Ok(memory_size) = usize::try_from(context.memory_size) else {
        return;
    };
    if memory_size == 0
        || memory_size > 0x20_0000
        || !process_guest_range_is_readable(context.memory_address, memory_size)
    {
        eprintln!(
            "[REMu][AUDIO2-MEM] push={push} unreadable context={:#x}+{:#x}",
            context.memory_address, context.memory_size
        );
        return;
    }

    let mut pages = Vec::new();
    let mut pointers = Vec::new();
    let mut page_offset = 0;
    while page_offset < context.memory_size {
        let page_bytes = (context.memory_size - page_offset).min(PAGE_SIZE);
        let qwords = page_bytes / 8;
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        let mut nonzero = 0_u64;
        let mut float_samples = 0_u64;
        let mut live_float_samples = 0_u64;
        for index in 0..qwords {
            let offset = page_offset + index * 8;
            let value = read_guest_u64(context.memory_address + offset);
            hash ^= value;
            hash = hash.wrapping_mul(0x1000_0000_01b3);
            nonzero += u64::from(value != 0);
            for bits in [value as u32, (value >> 32) as u32] {
                let sample = f32::from_bits(bits);
                if sample.is_finite() && sample.abs() <= 2.0 {
                    float_samples += 1;
                    live_float_samples += u64::from(sample.abs() >= f32::MIN_POSITIVE);
                }
            }
            if pointers.len() < MAX_POINTERS
                && value & 3 == 0
                && value != context.memory_address
                && process_guest_range_is_readable(value, 128)
                && !pointers
                    .iter()
                    .any(|(_, candidate): &(u64, u64)| *candidate == value)
            {
                pointers.push((offset, value));
            }
        }
        if nonzero != 0 {
            pages.push(format!(
                "{page_offset:#x}:nz={nonzero}/{qwords},f={live_float_samples}/{float_samples},h={hash:016x}"
            ));
        }
        page_offset += page_bytes;
    }

    let pointers = pointers
        .into_iter()
        .map(|(offset, address)| {
            let mut live_float_samples = 0_u32;
            let mut peak = 0.0_f32;
            for index in 0..16_u64 {
                let value = read_guest_u64(address + index * 8);
                for bits in [value as u32, (value >> 32) as u32] {
                    let sample = f32::from_bits(bits);
                    if sample.is_finite() && sample.abs() <= 2.0 {
                        live_float_samples += u32::from(sample.abs() >= f32::MIN_POSITIVE);
                        peak = peak.max(sample.abs());
                    }
                }
            }
            format!("{offset:#x}->{address:#x}(live-f32={live_float_samples}/32,peak={peak:.3})")
        })
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "[REMu][AUDIO2-MEM] push={push} context={:#x}+{:#x} channels={channels:?} pages=[{}] pointers=[{pointers}]",
        context.memory_address,
        context.memory_size,
        pages.join(","),
    );
}

fn trace_audio_out2_mastering_memory(parameters: u64, push: u64) {
    const MASTERING_OBJECT_PREFIX: u64 = 0x5b8;
    const MASTERING_OBJECT_SIZE: usize = 0x800;
    const TARGET_SAMPLE_SIZES: [usize; 3] = [0x4000, 0x2000, 0x1000];
    const MAX_TARGETS: usize = 48;
    const SECOND_LEVEL_SCAN_SIZES: [usize; 3] = [0x1_0000, 0x8000, 0x4000];
    const MAX_SECOND_LEVEL_TARGETS: usize = 64;

    let Some(object) = parameters.checked_sub(MASTERING_OBJECT_PREFIX) else {
        return;
    };
    if parameters == 0 || !process_guest_range_is_readable(object, MASTERING_OBJECT_SIZE) {
        eprintln!(
            "[REMu][AUDIO2-MASTER] push={push} unreadable parameters={parameters:#x} object={object:#x}"
        );
        return;
    }

    let mut targets = Vec::new();
    for offset in (0..MASTERING_OBJECT_SIZE).step_by(size_of::<u64>()) {
        let address = read_guest_u64(object + offset as u64);
        let sample_bytes = TARGET_SAMPLE_SIZES
            .into_iter()
            .find(|sample_bytes| process_guest_range_is_readable(address, *sample_bytes));
        if address & 3 != 0
            || sample_bytes.is_none()
            || targets
                .iter()
                .any(|(_, candidate, _, _, _, _)| *candidate == address)
        {
            continue;
        }
        let sample_bytes = sample_bytes.unwrap();
        let mut live_samples = 0_u32;
        let mut peak = 0.0_f32;
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for index in 0..sample_bytes / size_of::<u32>() {
            let bits = read_guest_u32(address + (index * size_of::<u32>()) as u64);
            hash ^= u64::from(bits);
            hash = hash.wrapping_mul(0x1000_0000_01b3);
            let sample = f32::from_bits(bits);
            if sample.is_finite() && (1.0e-8..=8.0).contains(&sample.abs()) {
                live_samples = live_samples.saturating_add(1);
                peak = peak.max(sample.abs());
            }
        }
        targets.push((offset, address, sample_bytes, live_samples, peak, hash));
        if targets.len() == MAX_TARGETS {
            break;
        }
    }
    let formatted_targets = targets
        .iter()
        .map(|(offset, address, sample_bytes, live_samples, peak, hash)| {
            format!(
                "{offset:#x}->{address:#x}(bytes={sample_bytes:#x},live-f32={live_samples}/{},peak={peak:.3},h={hash:016x})",
                sample_bytes / size_of::<u32>(),
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "[REMu][AUDIO2-MASTER] push={push} parameters={parameters:#x} object={object:#x} targets=[{formatted_targets}]"
    );

    let mut second_level_targets = Vec::new();
    for (parent_offset, parent, _, _, _, _) in &targets {
        let Some(scan_bytes) = SECOND_LEVEL_SCAN_SIZES
            .into_iter()
            .find(|scan_bytes| process_guest_range_is_readable(*parent, *scan_bytes))
        else {
            continue;
        };
        for child_offset in (0..scan_bytes).step_by(size_of::<u64>()) {
            let address = read_guest_u64(*parent + child_offset as u64);
            let sample_bytes = TARGET_SAMPLE_SIZES
                .into_iter()
                .find(|sample_bytes| process_guest_range_is_readable(address, *sample_bytes));
            if address & 3 != 0
                || address == object
                || address == *parent
                || sample_bytes.is_none()
                || second_level_targets
                    .iter()
                    .any(|(_, _, candidate, _, _, _, _)| *candidate == address)
            {
                continue;
            }
            let sample_bytes = sample_bytes.unwrap();
            let mut live_samples = 0_u32;
            let mut peak = 0.0_f32;
            let mut hash = 0xcbf2_9ce4_8422_2325_u64;
            for index in 0..sample_bytes / size_of::<u32>() {
                let bits = read_guest_u32(address + (index * size_of::<u32>()) as u64);
                hash ^= u64::from(bits);
                hash = hash.wrapping_mul(0x1000_0000_01b3);
                let sample = f32::from_bits(bits);
                if sample.is_finite() && (1.0e-8..=8.0).contains(&sample.abs()) {
                    live_samples = live_samples.saturating_add(1);
                    peak = peak.max(sample.abs());
                }
            }
            second_level_targets.push((
                *parent_offset,
                child_offset,
                address,
                sample_bytes,
                live_samples,
                peak,
                hash,
            ));
        }
    }
    second_level_targets
        .sort_unstable_by_key(|(_, _, _, _, live_samples, _, _)| std::cmp::Reverse(*live_samples));
    second_level_targets.truncate(MAX_SECOND_LEVEL_TARGETS);
    let second_level_targets = second_level_targets
        .into_iter()
        .map(
            |(parent_offset, child_offset, address, sample_bytes, live_samples, peak, hash)| {
                format!(
                    "{parent_offset:#x}+{child_offset:#x}->{address:#x}(bytes={sample_bytes:#x},live-f32={live_samples}/{},peak={peak:.3},h={hash:016x})",
                    sample_bytes / size_of::<u32>(),
                )
            },
        )
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "[REMu][AUDIO2-MASTER2] push={push} parameters={parameters:#x} object={object:#x} targets=[{second_level_targets}]"
    );
}

impl AudioOut2State {
    fn allocate_handle(&mut self) -> u64 {
        let handle = self.next_handle.max(1);
        self.next_handle = handle.saturating_add(1);
        handle
    }
}
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

thread_local! {
    static CURRENT_GUEST_THREAD_POINTER: Cell<u64> = const { Cell::new(0) };
    static CURRENT_GUEST_THREAD_HANDLE: Cell<u64> = const { Cell::new(0) };
    static CURRENT_GUEST_THREAD_EXIT_CONTEXT: Cell<*mut GuestThreadExitContext> = const { Cell::new(std::ptr::null_mut()) };
    static GUEST_PTHREAD_SPECIFIC: RefCell<BTreeMap<u32, u64>> = const { RefCell::new(BTreeMap::new()) };
    static IMPORT_TELEMETRY_CACHE: RefCell<Option<(usize, Arc<ImportTelemetry>)>> = const { RefCell::new(None) };
    static PROCESS_TIME_CALLSITES: RefCell<[(u64, u64); 8]> = const { RefCell::new([(0, 0); 8]) };
    #[cfg(target_os = "macos")]
    static KERNEL_MEMORY_READABLE_RANGES_CACHE: RefCell<Option<(usize, Arc<[(u64, u64)]>)>> = const { RefCell::new(None) };
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
        let mut state = dispatch
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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

#[cfg(target_arch = "x86_64")]
fn configure_probe_dispatch(image: &LinkedImage, game_root: &Path, maximum_hits: usize) {
    // Decode the installed title image before guest entry. Keeping PNG parsing on the runtime
    // side means the GPU seam carries a validated, immutable RGBA image instead of a host path
    // whose lifetime or casing the presenter would have to rediscover.
    let system_splash = load_system_splash(game_root);
    let state = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
    let mut state = state
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
    GUEST_PTHREAD_SPECIFIC.with_borrow_mut(BTreeMap::clear);
    state.exit_callbacks.clear();
    state.process_param_address = image.process_param.map_or(0, GuestAddress::get);
    state.thread_atexit_count_callback = 0;
    state.thread_atexit_report_callback = 0;
    state.thread_dtors_callback = 0;
    state.application_heap_api_address = 0;
    state.bundled_modules.clone_from(&image.bundled_modules);
    state.unwind_modules.clone_from(&image.unwind_modules);
    state.installed_exception_handlers.clear();
    state.next_ajm_context = 1;
    state.ajm_contexts.clear();
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
    state.agc_submission_count = 0;
    state.agc_graphics = AgcGraphicsState::default();
    state.agc_compute_queues.clear();
    state.agc_shader_registry = AgcShaderRegistry::default();
    state.agc_index_resources.clear();
    state.next_equeue = 1;
    state.kernel_equeues.clear();
    state.kernel_event_registrations.clear();
    state.pending_events.clear();
    state.pending_video_out_flips.clear();
    state.pending_agc_completions.clear();
    state.pending_agc_user_interrupts.clear();
    *KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = KernelFileState::default();
    state.ampr_files.clear();
    state.ampr_file_handles.clear();
    state.save_root = save_data_root(game_root);
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
}

const IMPORT_CALL_FRAME_SIZE: usize = 0xd0;
const _: () = {
    assert!(std::mem::size_of::<ImportCallFrame>() == IMPORT_CALL_FRAME_SIZE);
    assert!(std::mem::align_of::<ImportCallFrame>() == 16);
    assert!(std::mem::offset_of!(ImportCallFrame, rax) == 0x30);
    assert!(std::mem::offset_of!(ImportCallFrame, guest_rsp) == 0x38);
    assert!(std::mem::offset_of!(ImportCallFrame, guest_return_address) == 0x40);
    assert!(std::mem::offset_of!(ImportCallFrame, dispatch_index) == 0x48);
    assert!(std::mem::offset_of!(ImportCallFrame, xmm_arguments) == 0x50);
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
        "sub rsp, 0xd0",
        "mov [rsp + 0x00], rdi",
        "mov [rsp + 0x08], rsi",
        "mov [rsp + 0x10], rdx",
        "mov [rsp + 0x18], rcx",
        "mov [rsp + 0x20], r8",
        "mov [rsp + 0x28], r9",
        "mov [rsp + 0x30], rax",
        "lea r10, [rsp + 0xd8]",
        "mov [rsp + 0x38], r10",
        "mov r10, [rsp + 0xd8]",
        "mov [rsp + 0x40], r10",
        "mov r10, [rsp + 0xd0]",
        "mov [rsp + 0x48], r10",
        "movdqu [rsp + 0x50], xmm0",
        "movdqu [rsp + 0x60], xmm1",
        "movdqu [rsp + 0x70], xmm2",
        "movdqu [rsp + 0x80], xmm3",
        "movdqu [rsp + 0x90], xmm4",
        "movdqu [rsp + 0xa0], xmm5",
        "movdqu [rsp + 0xb0], xmm6",
        "movdqu [rsp + 0xc0], xmm7",
        "mov rdi, rsp",
        "call {dispatch}",
        "add rsp, 0xd0",
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
        "sub rsp, 8",
        "call r11",
        "add rsp, 8",
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
    let limit = import_call_trace_limit();
    if limit == 0 {
        if trace_import_errors() && !frame.is_null() {
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
    )
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
    if !telemetry.seen[index].swap(true, Ordering::Relaxed) {
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
            pthread_mutex_attr_state(call_args[1], &state)
        } else {
            None
        };
        if is_mutex_init && call_args[1] != 0 && mutex_attributes.is_none() {
            return ProbeImportReturn::integer(sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT));
        }
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
    // Most hot imports return above without consuming stack arguments. Reading them lazily avoids
    // six guest-memory loads on every timer, synchronization, numeric, and audio call while
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
    let _ampr_order = is_serialized_ampr_import(nid).then(|| {
        AMPR_IMPORT_ORDER
            .get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    });
    let state = PROBE_DISPATCH.get_or_init(|| RwLock::new(ProbeDispatchState::default()));
    let trace_lock_stall =
        std::env::var_os("REMU_TRACE_DISPATCH_STALLS").is_some_and(|value| value == "1");
    let lock_started = trace_lock_stall.then(Instant::now);
    let mut state = state
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
        if let Ok(stack_size) = usize::try_from(attributes.stack_size)
            && stack_size >= usize::try_from(MACOS_GUEST_PAGE_ALIGNMENT).unwrap_or(0x4000)
        {
            builder = builder.stack_size(stack_size);
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
                "[REMu][THREAD] create handle={handle} name={:?} entry={entry:#x} arg={thread_argument:#x} inner={inner_entry:#x}/{inner_argument:#x} detach={} stack={:#x}",
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
    let result = dispatch_probe_import(nid, call_args, stack_args, &mut state);
    let av_player_callbacks = std::mem::take(&mut state.av_player.pending_callbacks);
    drop(state);
    invoke_av_player_event_callbacks(av_player_callbacks);
    // Paced after the dispatch lock is released: the submitting thread waits for its own vblank,
    // exactly as it would on the console, without stalling other guest threads.
    let queued_flips = QUEUED_FLIPS.with(|queued| queued.replace(0));
    for _ in 0..queued_flips {
        pace_video_out_flip();
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

fn guest_kernel_rmdir(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let Some(guest_path) = read_guest_cstring(arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    let Some((host_path, writable)) = resolve_probe_guest_path(state, &guest_path) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if !writable {
        return sce_kernel_error(SCE_KERNEL_ERROR_PERMISSION_DENIED);
    }
    if !host_path.exists() {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    if !host_path.is_dir() {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_DIRECTORY);
    }
    // Only an empty directory may be removed; the guest deletes contents explicitly first.
    match std::fs::remove_dir(&host_path) {
        Ok(()) => 0,
        Err(error) => sce_kernel_io_error(&error),
    }
}

fn guest_kernel_mkdir(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let Some(guest_path) = read_guest_cstring(arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    let Some((host_path, writable)) = resolve_probe_guest_path(state, &guest_path) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if !writable {
        return sce_kernel_error(SCE_KERNEL_ERROR_PERMISSION_DENIED);
    }
    if host_path.exists() {
        return sce_kernel_error(SCE_KERNEL_ERROR_ALREADY_EXISTS);
    }
    let Some(parent) = host_path.parent() else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if !parent.is_dir() {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    match std::fs::create_dir(&host_path) {
        Ok(()) => 0,
        Err(error) => sce_kernel_io_error(&error),
    }
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
    let mut state = dispatch
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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

fn ajm_batch_initialize(arguments: [u64; 6]) -> u64 {
    let batch = arguments[0];
    let size = arguments[1];
    let info = arguments[2];
    let Ok(batch_size) = usize::try_from(size) else {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    };
    if batch == 0
        || info == 0
        || !(0x40..=0x10_000).contains(&size)
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
    Some(match nid {
        "bnZxYgAFeA0" => sanitizer_new_replace_external(),
        "vNe1w4diLCs" => guest_tls_get_addr(arguments[0]),
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
        "fd5Bp5tGTgo" => agc_fuse_shader_halves(arguments),
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
        "rPo6tV8D9bM" => system_service_get_status(arguments),
        "fZo48un7LK4" => system_service_param_get_int(arguments),
        "3RQ5aQfnstU" => system_service_get_notice_screen_skip_flag(arguments[0]),
        "Xjoosiw+XPI" => guest_uuid_create(arguments[0]),
        "yDBwVAolDgg" => guest_kernel_is_stack(arguments),
        "Qs0wWulgl7U" => 0,
        "RaqxZIf6DvE" => 0,
        "rbknaUjpqWo" => write_output_u64(arguments[1], 0),
        "JT+t00a3TxA" => np_get_account_country(arguments[1]),
        "eQH7nWPcAgc" => write_output_i32(arguments[1], 1),
        "e-ZuhGEoeC4" => write_output_i32(arguments[1], 0),
        "1xxcMiGu2fo" => user_service_get_user_name(arguments),
        "woNpu+45RLk" => user_service_get_age_level(arguments),
        "-sD02mFDBh4" => user_service_get_game_presets(arguments),
        "R9lA82OraNs" => app_content_initialize(arguments),
        "99b82IKXpH4" => app_content_app_param_get_int(arguments),
        "eaFXjfJv3xs" | "-4GCfYdNF1s" => 0,
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
        "ejekcaNQNq0" => guest_kernel_gettimeofday(arguments[0]),
        "0NTHN1NKONI" => guest_kernel_convert_localtime_to_utc(arguments),
        "-o5uEDpN+oY" => guest_kernel_convert_utc_to_localtime(arguments),
        "0-KXaS70xy4" | "eoht7mQOCmo" => pthread_get_specific(low_u32(arguments[0])),
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
        "959qrazPIrg" => state.process_param_address,
        "p5EcQeEeJAE" => {
            state.application_heap_api_address = arguments[0];
            0
        }
        "dl+4eHSzUu4" => ajm_initialize(arguments, state),
        "Q3dyFuwGn64" => ajm_module_register(arguments, state),
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
        "o9eWRkSL+M4" => av_player_init_ex(arguments, state),
        "HD1YKVU26-M" => av_player_post_init(arguments, state),
        "KMcEa+rHsIo" => av_player_add_source(arguments, state),
        "x8uvuFOPZhU" => av_player_add_source_ex(arguments, state),
        "NkJwDzKmIlw" => av_player_close(arguments[0], state),
        "wwM99gjFf1Y" => av_player_current_time(arguments[0], state),
        "UbQoYawOsfY" => av_player_is_active(arguments[0], state),
        "9y5v+fGN4Wk" => av_player_pause(arguments[0], state),
        "XC9wM+xULz8" => av_player_jump_to_time(arguments, state),
        "w5moABNwnRY" | "ET4Gr-Uu07s" => av_player_start(arguments[0], state),
        "ZC17w3vB5Lo" => av_player_stop(arguments[0], state),
        "OVths0xGfho" => av_player_set_looping(arguments, state),
        "Wnp1OVcrZgk" => av_player_get_audio_data(arguments, state),
        "o3+RWnHViSg" => av_player_get_video_data(arguments, false, state),
        "JdksQu8pNdQ" => av_player_get_video_data(arguments, true, state),
        "d8FcbzfAdQw" => av_player_get_stream_info(arguments, state),
        "ctTAcF5DiKQ" => av_player_get_stream_info(arguments, state),
        "ODJK2sn9w4A" | "k-q+xOxdc3E" => av_player_validate_handle(arguments[0], state),
        "hdTyRzCXQeQ" => av_player_stream_count(arguments[0], state),
        "gEpBkcwxUjw" => ampr_resolve_file_paths(arguments, state),
        // One resolver reached through eight names: the variants differ only in whether the
        // caller wants sizes, supplies a prefix, or collects a per-entry result.
        "WT-5NKy42fw" => ampr_resolve_paths_core(
            arguments[0],
            arguments[1],
            None,
            AprResolveOutputs {
                ids: arguments[2],
                error_index: arguments[3],
                ..AprResolveOutputs::default()
            },
            state,
        ),
        "eYAh2vlCY-U" => ampr_resolve_paths_core(
            arguments[0],
            arguments[1],
            None,
            AprResolveOutputs {
                ids: arguments[2],
                results: arguments[3],
                ..AprResolveOutputs::default()
            },
            state,
        ),
        "QzB4O+bJQyA" => ampr_resolve_paths_core(
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
        ),
        "i3HWvW35jao" => ampr_resolve_paths_core(
            arguments[1],
            arguments[2],
            read_guest_cstring(arguments[0]),
            AprResolveOutputs {
                ids: arguments[3],
                error_index: arguments[4],
                ..AprResolveOutputs::default()
            },
            state,
        ),
        "w5fcCG+t31g" => ampr_resolve_paths_core(
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
        ),
        "VB-BtuIW8Xc" => ampr_resolve_paths_core(
            arguments[1],
            arguments[2],
            read_guest_cstring(arguments[0]),
            AprResolveOutputs {
                ids: arguments[3],
                results: arguments[4],
                ..AprResolveOutputs::default()
            },
            state,
        ),
        "C+Khtbbx2g8" => ampr_resolve_paths_core(
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
        ),
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
        "WrOLvHU0yQM" => posix_pthread_result(pthread_set_specific(arguments, state)),
        "+BzXYkqYeLE" => pthread_set_specific(arguments, state),
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
        "xk0AcarP3V4" => pad_open(arguments, state),
        "u1GRHp+oWoY" => pad_get_handle(arguments, state),
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
        "l4fM9K-Lyks" => agc_dcb_set_index_buffer(arguments, state),
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
        "w0hLuNarQxY" => (low_i32(arguments[0]) == 1)
            .then_some(0)
            .unwrap_or_else(|| sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE)),
        // libScePad output structures must always be initialized; successful no-ops would expose
        // stale guest stack contents as controller capabilities and button state.
        "clVvL4ZDntw" => pad_set_motion_sensor_state(arguments, state),
        "gjP9-KQzoUk" => pad_get_controller_information(arguments, state),
        "6ncge5+l5Qs" => pad_close(arguments, state),
        "AcslpN1jHR8" => pad_get_extended_information(arguments, state),
        "YndgXqQVV7c" => pad_read_state(arguments, state),
        "q1cHNfGycLI" => pad_read(arguments, state),
        "W2G-yoyMF5U" => pad_set_vibration_mode(arguments, state),
        "rIZnR6eSpvk" | "DscD1i9HX1w" => pad_handle_operation(arguments, state),
        "r44mAxdSG+U" | "vDLMoJLde8I" => pad_boolean_operation(arguments, state),
        "yFVnOdGxvZY" | "2JgFB2n9oUM" | "RR4novUEENY" => pad_pointer_operation(arguments, state),
        "znaWI0gpuo8" => pad_get_trigger_effect_state(arguments, state),
        "IHPqcbc0zCA" => pad_device_class_parse_data(arguments, state),
        "hv1luiJrqQM" => pad_initialize(state),
        // libSceSystemService status used by the foreground/background state machine.
        "rPo6tV8D9bM" => system_service_get_status(arguments),
        "fZo48un7LK4" => system_service_param_get_int(arguments),
        "RpQJJVKTiFM" | "4fU5yvOkVG4" => kernel_module_info_for_unwind(arguments, state),
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
        "dyIhnXq-0SM" => save_data_dir_name_search(arguments, state),
        "ZP4e7rlzOUk" => save_data_mount3(arguments, state),
        "gjRZNnw0JPE" => save_data_create_transaction_resource(arguments, state),
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
        "buYbeLOGWmA" => app_content_temporary_data_mount(arguments, state),
        "YJtKLttI9fM" => game_update_initialize(state),
        "jO8DM8oyego" => np_entitlement_access_initialize(arguments, state),
        "ysmw6J-P8Ak" => np_session_signaling_initialize(arguments, state),
        "sk54bi6FtYM" => np_web_api2_create_user_context(arguments, state),
        "3Zl8BePTh9Y" => np_check_callback(state),
        "Ec63y59l9tw" => np_set_np_title_id(arguments, state),
        "YBiIdcDPrxs" => share_feature_permit(state),
        "7QZtURYnXG4" => share_set_content_parameter(arguments[0], state),
        "hZIg1EWGsHM" => ult_initialize(arguments[0], state),
        "uoUpLGNkygk" => common_dialog_initialize(state),
        "s9e3+YpRnzw" => save_data_dialog_initialize(state),
        "ERKzksauAJA" => state
            .services
            .save_data_dialog_status
            .cast_unsigned()
            .into(),
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
            if state.unimplemented_nids.insert(nid.to_owned()) {
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
    state.pending_events.retain(|event| event.equeue != handle);
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
        *pending = event;
    } else {
        state.pending_events.push_back(event);
    }
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
    state.pending_agc_completions.retain(|&(sequence, ident)| {
        if completed.contains(&sequence) {
            completed_queues.push((sequence, ident));
            false
        } else {
            true
        }
    });
    let mut completed_compute = Vec::new();
    state.pending_agc_user_interrupts.retain(|sequence| {
        if completed.contains(sequence) {
            completed_compute.push(*sequence);
            false
        } else {
            true
        }
    });
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
                },
            );
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
                },
            );
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
            let mut state = state
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
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
            remaining.min(Duration::from_millis(1))
        } else {
            // Backend submission completion is an atomic published from the presenter. Polling it
            // at a 1 ms ceiling avoids a cross-crate callback while keeping idle waits parked.
            Duration::from_millis(1)
        };
        let wake = kernel_equeue_wake();
        let wake_state = wake
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *wake_state != generation {
            continue;
        }
        let _ = wake
            .1
            .wait_timeout(wake_state, wait_for)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
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
            if remu_gpu::emit_event(remu_gpu::GpuEvent::ShowHostSplash(image)) {
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
    if !remu_gpu::emit_event(remu_gpu::GpuEvent::HideHostSplash) {
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
        write_guest_u64(field, field.wrapping_add(relative));
    }
}

fn patch_agc_shader_program_registers(header: u64, code: u64) -> bool {
    let registers = read_guest_u64(header + AGC_SHADER_SH_REGISTERS_OFFSET);
    let shader_type = read_guest_u8(header + AGC_SHADER_TYPE_OFFSET);
    let register_count = read_guest_u8(header + AGC_SHADER_SH_REGISTER_COUNT_OFFSET);
    if registers == 0 || register_count < 2 {
        return false;
    }
    let expected = match shader_type {
        0 => (0x20c, 0x20d),
        1 => (0x008, 0x009),
        2 | 6 => (0x0c8, 0x0c9),
        4 => (0x08a, 0x08b),
        7 => (0x148, 0x149),
        _ => return false,
    };
    if (read_guest_u32(registers), read_guest_u32(registers + 8)) != expected {
        return false;
    }
    write_guest_u32(registers + 4, ((code >> 8) & 0xffff_ffff) as u32);
    write_guest_u32(registers + 12, ((code >> 40) & 0xff) as u32);
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
}

fn capture_agc_gpu_events<R>(operation: impl FnOnce() -> R) -> (R, Vec<remu_gpu::GpuEvent>) {
    AGC_GPU_EVENT_CAPTURE.with_borrow_mut(|capture| {
        assert!(
            capture.is_none(),
            "nested AGC GPU event capture would lose submission ownership"
        );
        *capture = Some(Vec::new());
    });
    let result = operation();
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
        remu_gpu::emit_event(event.expect("AGC event was not captured"));
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
            result == 0 && copied == length
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
            copied == isize::try_from(output.len()).unwrap_or(-1)
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
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe { std::ptr::read_unaligned(address as *const u8) }
}

fn read_guest_u16(address: u64) -> u16 {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe { std::ptr::read_unaligned(address as *const u16) }
}

fn read_guest_u32(address: u64) -> u32 {
    // SAFETY: HLE calls use guest pointers into the process's live mapped address space.
    unsafe { std::ptr::read_unaligned(address as *const u32) }
}

fn read_guest_u64(address: u64) -> u64 {
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
    unsafe { libc::malloc(size) as usize as u64 }
}

fn guest_free(address: u64) -> u64 {
    if address != 0 {
        // SAFETY: The guest passes an allocation previously returned by this libc HLE family.
        unsafe {
            libc::free(address as *mut std::ffi::c_void);
        }
    }
    0
}

fn guest_calloc(count: u64, size: u64) -> u64 {
    let (Ok(count), Ok(size)) = (usize::try_from(count), usize::try_from(size)) else {
        return 0;
    };
    // SAFETY: libc validates the allocation-size multiplication and returns process-local memory.
    unsafe { libc::calloc(count, size) as usize as u64 }
}

fn guest_realloc(address: u64, size: u64) -> u64 {
    let Ok(size) = usize::try_from(size) else {
        return 0;
    };
    // SAFETY: `address` is null or belongs to the same libc allocation family.
    unsafe { libc::realloc(address as *mut std::ffi::c_void, size) as usize as u64 }
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
    let mut output = std::ptr::null_mut();
    // SAFETY: `output` is writable and alignment/size are represented in the host ABI.
    let result = unsafe { libc::posix_memalign(&raw mut output, alignment, size) };
    if result == 0 {
        output as usize as u64
    } else {
        0
    }
}

fn guest_posix_memalign(output: u64, alignment: u64, size: u64) -> u64 {
    if output == 0 {
        return libc::EINVAL as u64;
    }
    let (Ok(alignment), Ok(size)) = (usize::try_from(alignment), usize::try_from(size)) else {
        return libc::ENOMEM as u64;
    };
    let mut allocation = std::ptr::null_mut();
    // SAFETY: `allocation` is writable and libc validates the requested alignment.
    let result = unsafe { libc::posix_memalign(&raw mut allocation, alignment, size) };
    if result == 0 {
        write_guest_u64(output, allocation as usize as u64);
    }
    u64::try_from(result).unwrap_or(u64::MAX)
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
    let result =
        unsafe { libc::strcasecmp(left as *const libc::c_char, right as *const libc::c_char) };
    signed_i32_return(result)
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

fn guest_fopen(path_address: u64, mode_address: u64, state: &ProbeDispatchState) -> u64 {
    let (Some(guest_path), Some(mode)) = (
        read_guest_cstring(path_address),
        read_guest_cstring(mode_address),
    ) else {
        return 0;
    };
    let Some((host_path, writable)) = resolve_probe_guest_path(state, &guest_path) else {
        if let Some(miss) = trace_guest_vfs_miss() {
            eprintln!("[REMu][VFS] miss={miss} rejected path {guest_path:?}");
        }
        return 0;
    };
    if !mode.starts_with('r') && !writable {
        if let Some(miss) = trace_guest_vfs_miss() {
            eprintln!("[REMu][VFS] miss={miss} rejected write mode {mode:?} for {guest_path:?}");
        }
        return 0;
    }
    if writable
        && !mode.starts_with('r')
        && let Some(parent) = host_path.parent()
        && std::fs::create_dir_all(parent).is_err()
    {
        return 0;
    }
    let (Ok(host_path), Ok(mode)) = (
        CString::new(host_path.to_string_lossy().as_bytes()),
        CString::new(mode),
    ) else {
        return 0;
    };
    // SAFETY: Both strings are NUL-terminated and host libc owns the returned FILE object.
    let file = unsafe { libc::fopen(host_path.as_ptr(), mode.as_ptr()) };
    if trace_guest_vfs() {
        eprintln!(
            "[REMu][VFS] fopen guest={guest_path:?} host={:?} result={file:p}",
            host_path.to_string_lossy()
        );
    }
    file as usize as u64
}

const APP_PATH_CACHE_CAPACITY: usize = 65_536;

#[derive(Debug, Default)]
struct ImmutableDirectoryIndex {
    exact: HashMap<String, PathBuf>,
    folded: HashMap<String, PathBuf>,
}

impl ImmutableDirectoryIndex {
    fn read(parent: &Path) -> Self {
        let mut index = Self::default();
        let Ok(entries) = std::fs::read_dir(parent) else {
            return index;
        };
        for entry in entries.flatten() {
            let Some(name) = entry.file_name().to_str().map(ToOwned::to_owned) else {
                continue;
            };
            let path = entry.path();
            index
                .folded
                .entry(name.to_ascii_lowercase())
                .or_insert_with(|| path.clone());
            index.exact.insert(name, path);
        }
        index
    }

    fn get(&self, component: &str) -> Option<PathBuf> {
        self.exact
            .get(component)
            .or_else(|| self.folded.get(&component.to_ascii_lowercase()))
            .cloned()
    }

    fn entry_count(&self) -> usize {
        self.exact.len().saturating_add(self.folded.len())
    }
}

#[derive(Debug, Default)]
struct ImmutableAppPathCache {
    roots: HashMap<PathBuf, HashMap<String, PathBuf>>,
    directories: HashMap<PathBuf, ImmutableDirectoryIndex>,
    entries: usize,
}

impl ImmutableAppPathCache {
    fn get(&self, game_root: &Path, relative: &str) -> Option<PathBuf> {
        self.roots.get(game_root)?.get(relative).cloned()
    }

    fn insert(&mut self, game_root: &Path, relative: &str, resolved: PathBuf) {
        if self.entries >= APP_PATH_CACHE_CAPACITY {
            self.roots.clear();
            self.entries = 0;
        }
        let previous = self
            .roots
            .entry(game_root.to_owned())
            .or_default()
            .insert(relative.to_owned(), resolved);
        if previous.is_none() {
            self.entries += 1;
        }
    }

    fn directory_entry(&self, parent: &Path, component: &str) -> Option<Option<PathBuf>> {
        self.directories
            .get(parent)
            .map(|directory| directory.get(component))
    }

    fn insert_directory(&mut self, parent: PathBuf, directory: ImmutableDirectoryIndex) {
        let new_entries = directory.entry_count();
        if self.entries.saturating_add(new_entries) >= APP_PATH_CACHE_CAPACITY {
            self.roots.clear();
            self.directories.clear();
            self.entries = 0;
        }
        if let Some(previous) = self.directories.insert(parent, directory) {
            self.entries = self.entries.saturating_sub(previous.entry_count());
        }
        self.entries = self.entries.saturating_add(new_entries);
    }
}

fn resolve_immutable_app_path(
    root: &Path,
    relative: &str,
    cache: &RwLock<ImmutableAppPathCache>,
) -> Option<PathBuf> {
    let mut components: Vec<&str> = Vec::new();
    for component in relative.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                components.pop()?;
            }
            component => components.push(component),
        }
    }

    let mut host = root.to_owned();
    for component in components {
        let cached = cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .directory_entry(&host, component);
        match cached {
            Some(Some(matched)) => {
                host = matched;
            }
            Some(None) => host.push(component),
            None => {
                let directory = ImmutableDirectoryIndex::read(&host);
                let matched = directory.get(component);
                cache
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert_directory(host.clone(), directory);
                host = matched.unwrap_or_else(|| host.join(component));
            }
        }
    }
    Some(host)
}

fn resolve_guest_path(game_root: &Path, guest_path: &str) -> Option<PathBuf> {
    let normalized = guest_path.replace('\\', "/");
    let relative = if let Some(path) = normalized.strip_prefix("/app0/") {
        path
    } else if normalized == "/app0" {
        ""
    } else if let Some((_, path)) = normalized.split_once("/app0/") {
        path
    } else if normalized.starts_with('/') {
        return None;
    } else {
        normalized.as_str()
    };
    // `/app0` is immutable for the lifetime of a title. Unreal repeatedly probes the same
    // optional configuration and localization paths, and resolving each miss used to repeat an
    // `exists` syscall for every component plus a `read_dir` scan for every case-insensitive
    // fallback. Cache the resolved host spelling (including paths that ultimately do not exist)
    // so subsequent stat/open calls perform only their authoritative final syscall.
    static APP_PATH_CACHE: OnceLock<RwLock<ImmutableAppPathCache>> = OnceLock::new();
    let cache = APP_PATH_CACHE.get_or_init(|| RwLock::new(ImmutableAppPathCache::default()));
    if let Some(path) = cache
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(game_root, relative)
    {
        return Some(path);
    }
    let resolved = resolve_immutable_app_path(game_root, relative, cache)?;
    let mut cache = cache
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    cache.insert(game_root, relative, resolved.clone());
    Some(resolved)
}

/// Returns a host file that stands in for a guest path, if an overlay directory supplies one.
///
/// A dump is read-only and stays that way, so replacing one file in it -- a command line, a
/// configuration file -- otherwise means editing the dump itself. An overlay directory mirrors
/// the guest tree and is consulted first, which keeps experiments out of the dump entirely.
///
/// Set `REMU_OVERLAY_DIR` to a directory whose layout mirrors the guest filesystem, so
/// `/app0/ue4commandline.txt` is served from `$REMU_OVERLAY_DIR/app0/ue4commandline.txt`.
fn overlay_guest_path(normalized: &str) -> Option<PathBuf> {
    static OVERLAY: OnceLock<Option<PathBuf>> = OnceLock::new();
    let root = OVERLAY
        .get_or_init(|| std::env::var_os("REMU_OVERLAY_DIR").map(PathBuf::from))
        .as_ref()?;
    let candidate = resolve_path_beneath(root, normalized.trim_start_matches('/'))?;
    candidate.is_file().then_some(candidate)
}

fn resolve_probe_guest_path(
    state: &ProbeDispatchState,
    guest_path: &str,
) -> Option<(PathBuf, bool)> {
    let normalized = guest_path.replace('\\', "/");
    if let Some(overlaid) = overlay_guest_path(&normalized) {
        return Some((overlaid, false));
    }
    for (mount_point, mounted) in &state.save_mounts {
        if normalized == *mount_point {
            return Some((mounted.path.clone(), !mounted.read_only));
        }
        if let Some(relative) = normalized.strip_prefix(&format!("{mount_point}/")) {
            return resolve_path_beneath(&mounted.path, relative)
                .map(|path| (path, !mounted.read_only));
        }
    }
    resolve_guest_path(&state.game_root, &normalized).map(|path| (path, false))
}

/// Finds a sibling of `path` whose name differs only by case.
///
/// Returns `None` when the parent cannot be listed or nothing matches, leaving the caller's
/// original path so a genuine miss still reports the name the guest asked for.
fn case_insensitive_child(path: &Path) -> Option<PathBuf> {
    let name = path.file_name()?.to_str()?;
    let parent = path.parent()?;
    let mut matched = None;
    for entry in std::fs::read_dir(parent).ok()? {
        let entry = entry.ok()?;
        let candidate = entry.file_name();
        let candidate = candidate.to_str()?;
        if candidate.eq_ignore_ascii_case(name) {
            // An exact match always wins; otherwise keep the first case-insensitive hit so the
            // result stays stable regardless of directory order.
            if candidate == name {
                return Some(entry.path());
            }
            matched.get_or_insert_with(|| entry.path());
        }
    }
    matched
}

fn resolve_path_beneath(root: &Path, relative: &str) -> Option<PathBuf> {
    // Console paths routinely contain duplicate separators (for example
    // `/app0//shaders/presenter.vert`) and parent components: engines address their content
    // relative to a binaries directory, so `../../../game/config.ini` is an ordinary request, not
    // an attack. Resolve `..` by popping the component it cancels, and refuse only a path that
    // would leave the game root entirely.
    let mut components: Vec<&str> = Vec::new();
    for component in relative.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                components.pop()?;
            }
            component => components.push(component),
        }
    }
    let mut host = root.to_owned();
    for component in components {
        host.push(component);
        if !host.exists() {
            // Console lookups do not distinguish case, and dumps are frequently stored with
            // normalized names. Fall back to a case-insensitive match within the parent so a
            // title that asks for `AkRoomVerb.prx` still finds `akroomverb.prx`.
            if let Some(matched) = case_insensitive_child(&host) {
                host = matched;
            }
        }
    }
    Some(host)
}

fn read_guest_cstring(address: u64) -> Option<String> {
    if address == 0 {
        return None;
    }
    // SAFETY: Guest libc string arguments are NUL-terminated within live guest memory.
    let string = unsafe { std::ffi::CStr::from_ptr(address as *const libc::c_char) };
    string.to_str().ok().map(ToOwned::to_owned)
}

fn guest_fseek(handle: u64, offset: u64, whence: u64) -> u64 {
    if handle == 0 {
        return u64::MAX;
    }
    // SAFETY: fopen returned the handle, and libc validates offset/whence.
    signed_i32_return(unsafe {
        libc::fseek(
            handle as *mut libc::FILE,
            i64::from_ne_bytes(offset.to_ne_bytes()),
            low_i32(whence),
        )
    })
}

fn guest_ftell(handle: u64) -> u64 {
    if handle == 0 {
        return u64::MAX;
    }
    // SAFETY: fopen returned the live FILE handle.
    unsafe { libc::ftell(handle as *mut libc::FILE).cast_unsigned() }
}

fn guest_fclose(handle: u64) -> u64 {
    if handle == 0 {
        return u64::MAX;
    }
    // SAFETY: fopen returned the live FILE handle and ownership ends here.
    signed_i32_return(unsafe { libc::fclose(handle as *mut libc::FILE) })
}

fn guest_fread(destination: u64, size: u64, count: u64, handle: u64) -> u64 {
    if destination == 0 || handle == 0 || size == 0 || count == 0 {
        return 0;
    }
    let (Ok(size), Ok(count)) = (usize::try_from(size), usize::try_from(count)) else {
        return 0;
    };
    // SAFETY: The guest supplies a destination sized for size*count and fopen returned the handle.
    unsafe {
        libc::fread(
            destination as *mut libc::c_void,
            size,
            count,
            handle as *mut libc::FILE,
        ) as u64
    }
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

fn guest_kernel_open(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let Some(guest_path) = read_guest_cstring(arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    let flags = low_i32(arguments[1]);
    let access = flags & 0x3;
    if access == 0x3 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some((host_path, writable)) = resolve_probe_guest_path(state, &guest_path) else {
        if let Some(miss) = trace_guest_vfs_miss() {
            eprintln!("[REMu][VFS] miss={miss} sceKernelOpen rejected guest={guest_path:?}");
        }
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if access != 0 && !writable {
        if let Some(miss) = trace_guest_vfs_miss() {
            eprintln!(
                "[REMu][VFS] miss={miss} sceKernelOpen rejected write access guest={guest_path:?} flags={flags:#x}"
            );
        }
        return sce_kernel_error(SCE_KERNEL_ERROR_PERMISSION_DENIED);
    }
    let create = writable && flags & 0x200 != 0;
    let exclusive = create && flags & 0x800 != 0;
    let mut options = std::fs::OpenOptions::new();
    options
        .read(access != 1)
        .write(access != 0)
        .append(access != 0 && flags & 0x8 != 0)
        .create(create && !exclusive)
        .create_new(exclusive)
        .truncate(writable && access != 0 && flags & 0x400 != 0);
    #[cfg(unix)]
    {
        options.mode(low_u32(arguments[2]));
    }
    let file = match options.open(&host_path) {
        Ok(file) => file,
        Err(error) => {
            let miss = trace_guest_vfs_miss();
            if error.kind() != std::io::ErrorKind::NotFound || miss.is_some() {
                eprintln!(
                    "[REMu][VFS] miss={miss:?} sceKernelOpen failed guest={guest_path:?} host={:?}: {error}",
                    host_path.display()
                );
            }
            return sce_kernel_io_error(&error);
        }
    };
    if flags & 0x0002_0000 != 0 && file.metadata().is_ok_and(|metadata| !metadata.is_dir()) {
        if let Some(miss) = trace_guest_vfs_miss() {
            eprintln!(
                "[REMu][VFS] miss={miss} sceKernelOpen expected directory guest={guest_path:?} host={:?}",
                host_path.display()
            );
        }
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_DIRECTORY);
    }
    let is_directory = file.metadata().is_ok_and(|metadata| metadata.is_dir());
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fd = files.next_fd.max(3);
    files.next_fd = fd.saturating_add(1);
    if is_directory {
        let mut entries: Vec<(String, bool)> = match std::fs::read_dir(&host_path) {
            Ok(listing) => listing
                .filter_map(|entry| {
                    let entry = entry.ok()?;
                    Some((
                        entry.file_name().to_string_lossy().into_owned(),
                        entry.file_type().is_ok_and(|kind| kind.is_dir()),
                    ))
                })
                .collect(),
            Err(error) => {
                let miss = trace_guest_vfs_miss();
                if error.kind() != std::io::ErrorKind::NotFound || miss.is_some() {
                    eprintln!(
                        "[REMu][VFS] miss={miss:?} sceKernelOpen could not list guest={guest_path:?} host={:?}: {error}",
                        host_path.display()
                    );
                }
                return sce_kernel_io_error(&error);
            }
        };
        // A stable order keeps repeated enumerations reproducible across runs and hosts.
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        let mut next_file_number = files.next_file_number;
        let directory = KernelDirectory::new(&entries, &mut next_file_number);
        files.next_file_number = next_file_number;
        files.directories.insert(fd, directory);
        if trace_guest_vfs() {
            eprintln!(
                "[REMu][VFS] sceKernelOpen directory guest={guest_path:?} entries={} fd={fd}",
                entries.len()
            );
        }
    }
    files.files.insert(fd, file);
    if trace_guest_vfs() {
        eprintln!(
            "[REMu][VFS] sceKernelOpen guest={guest_path:?} host={:?} flags={flags:#x} fd={fd}",
            host_path.display()
        );
    }
    signed_i32_return(fd)
}

/// Reads packed directory entries into the guest buffer.
///
/// Returns the number of bytes written and zero once the enumeration is exhausted. `basep`, when
/// supplied, receives the enumeration position before the read so the guest can resume there.
fn guest_kernel_getdents(arguments: [u64; 6], report_position: bool) -> u64 {
    let (fd, buffer, capacity) = (low_i32(arguments[0]), arguments[1], arguments[2]);
    if buffer == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    if capacity < DIRENT_BLOCK_SIZE as u64 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !files.files.contains_key(&fd) {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    }
    let Some(directory) = files.directories.get_mut(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if report_position && arguments[3] != 0 {
        write_guest_u64(arguments[3], directory.offset as u64);
    }
    let remaining = directory.records.len().saturating_sub(directory.offset);
    if remaining == 0 {
        return 0;
    }
    // Only whole blocks are transferred, so a record is never split across two reads.
    let transfer = remaining.min((capacity as usize / DIRENT_BLOCK_SIZE) * DIRENT_BLOCK_SIZE);
    if transfer == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let end = directory.offset + transfer;
    write_guest_bytes(buffer, &directory.records[directory.offset..end]);
    directory.offset = end;
    transfer as u64
}

fn guest_kernel_stat(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let Some(guest_path) = read_guest_cstring(arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    let Some((host_path, _)) = resolve_probe_guest_path(state, &guest_path) else {
        if let Some(miss) = trace_guest_vfs_miss() {
            eprintln!("[REMu][VFS] miss={miss} sceKernelStat rejected guest={guest_path:?}");
        }
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let metadata = match std::fs::metadata(&host_path) {
        Ok(metadata) => metadata,
        Err(error) => {
            let miss = trace_guest_vfs_miss();
            if error.kind() != std::io::ErrorKind::NotFound || miss.is_some() {
                eprintln!(
                    "[REMu][VFS] miss={miss:?} sceKernelStat failed guest={guest_path:?} host={:?}: {error}",
                    host_path.display()
                );
            }
            return sce_kernel_io_error(&error);
        }
    };
    let payload = kernel_stat_payload_from_metadata(&metadata, guest_path.as_bytes());
    // SAFETY: sceKernelStat supplies a writable 120-byte Gen5 stat object in live guest memory.
    unsafe {
        std::ptr::copy_nonoverlapping(payload.as_ptr(), arguments[1] as *mut u8, payload.len());
    }
    if trace_guest_vfs() {
        eprintln!(
            "[REMu][VFS] sceKernelStat guest={guest_path:?} host={:?} size={}",
            host_path.display(),
            metadata.len()
        );
    }
    0
}

fn guest_kernel_read(arguments: [u64; 6]) -> u64 {
    let fd = low_i32(arguments[0]);
    let destination = arguments[1];
    let requested = arguments[2];
    if requested == 0 {
        return 0;
    }
    if destination == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let Ok(requested) = usize::try_from(requested) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(file) = files.files.get_mut(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    };
    // SAFETY: sceKernelRead supplies a writable range of exactly `requested` bytes in live guest
    // memory, and the global import dispatcher serializes access to the associated File.
    let destination = unsafe { std::slice::from_raw_parts_mut(destination as *mut u8, requested) };
    let read = match file.read(destination) {
        Ok(read) => read,
        Err(error) => return sce_kernel_io_error(&error),
    };
    if trace_guest_vfs() {
        eprintln!("[REMu][VFS] sceKernelRead fd={fd} requested={requested} read={read}");
    }
    u64::try_from(read).expect("read byte count fits u64")
}

fn guest_kernel_write(arguments: [u64; 6]) -> u64 {
    let fd = low_i32(arguments[0]);
    let Ok(length) = usize::try_from(arguments[2]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if length != 0 && arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let payload = if length == 0 {
        &[]
    } else {
        // SAFETY: `_write` supplies a readable guest buffer of exactly `length` bytes.
        unsafe { std::slice::from_raw_parts(arguments[1] as *const u8, length) }
    };
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let result = match fd {
        1 => std::io::stdout().write(payload),
        2 => std::io::stderr().write(payload),
        _ => files
            .files
            .get_mut(&fd)
            .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::NotFound))
            .and_then(|file| file.write(payload)),
    };
    match result {
        Ok(written) => u64::try_from(written).expect("written byte count fits u64"),
        Err(error) if fd != 1 && fd != 2 && !files.files.contains_key(&fd) => {
            sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR)
        }
        Err(error) => sce_kernel_io_error(&error),
    }
}

fn guest_kernel_close(arguments: [u64; 6]) -> u64 {
    let fd = low_i32(arguments[0]);
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    files.directories.remove(&fd);
    if files.files.remove(&fd).is_none() {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    }
    drop(files);
    if trace_guest_vfs() {
        eprintln!("[REMu][VFS] sceKernelClose fd={fd}");
    }
    0
}

fn guest_kernel_lseek(arguments: [u64; 6]) -> u64 {
    let fd = low_i32(arguments[0]);
    let offset = i64::from_ne_bytes(arguments[1].to_ne_bytes());
    let seek_from = match low_i32(arguments[2]) {
        0 if offset >= 0 => std::io::SeekFrom::Start(offset.cast_unsigned()),
        1 => std::io::SeekFrom::Current(offset),
        2 => std::io::SeekFrom::End(offset),
        _ => return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT),
    };
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(file) = files.files.get_mut(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    };
    match file.seek(seek_from) {
        Ok(position) => position,
        Err(error) => sce_kernel_io_error(&error),
    }
}

fn guest_kernel_fstat(arguments: [u64; 6]) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let fd = low_i32(arguments[0]);
    let files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(file) = files.files.get(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    };
    let metadata = match file.metadata() {
        Ok(metadata) => metadata,
        Err(error) => return sce_kernel_io_error(&error),
    };
    let payload = kernel_stat_payload_from_metadata(&metadata, &fd.to_le_bytes());
    // SAFETY: the guest ABI supplies writable storage for one complete 120-byte stat object.
    unsafe {
        std::ptr::copy_nonoverlapping(payload.as_ptr(), arguments[1] as *mut u8, payload.len());
    }
    0
}

fn guest_kernel_ftruncate(arguments: [u64; 6]) -> u64 {
    let fd = low_i32(arguments[0]);
    let length = i64::from_ne_bytes(arguments[1].to_ne_bytes());
    if length < 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(file) = files.files.get_mut(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    };
    file.set_len(length.cast_unsigned())
        .map_or_else(|error| sce_kernel_io_error(&error), |()| 0)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct GuestKernelStat {
    dev: u32,
    ino: u32,
    mode: u16,
    nlink: u16,
    uid: u32,
    gid: u32,
    rdev: u32,
    atime_sec: i64,
    atime_nsec: i64,
    mtime_sec: i64,
    mtime_nsec: i64,
    ctime_sec: i64,
    ctime_nsec: i64,
    size: i64,
    blocks: i64,
    block_size: u32,
    flags: u32,
    generation: u32,
    spare: i32,
    birthtime_sec: i64,
    birthtime_nsec: i64,
}

impl GuestKernelStat {
    fn encode(self) -> [u8; SCE_KERNEL_STAT_SIZE] {
        let mut payload = [0_u8; SCE_KERNEL_STAT_SIZE];
        payload[0..4].copy_from_slice(&self.dev.to_le_bytes());
        payload[4..8].copy_from_slice(&self.ino.to_le_bytes());
        payload[8..10].copy_from_slice(&self.mode.to_le_bytes());
        payload[10..12].copy_from_slice(&self.nlink.to_le_bytes());
        payload[12..16].copy_from_slice(&self.uid.to_le_bytes());
        payload[16..20].copy_from_slice(&self.gid.to_le_bytes());
        payload[20..24].copy_from_slice(&self.rdev.to_le_bytes());
        payload[24..32].copy_from_slice(&self.atime_sec.to_le_bytes());
        payload[32..40].copy_from_slice(&self.atime_nsec.to_le_bytes());
        payload[40..48].copy_from_slice(&self.mtime_sec.to_le_bytes());
        payload[48..56].copy_from_slice(&self.mtime_nsec.to_le_bytes());
        payload[56..64].copy_from_slice(&self.ctime_sec.to_le_bytes());
        payload[64..72].copy_from_slice(&self.ctime_nsec.to_le_bytes());
        payload[72..80].copy_from_slice(&self.size.to_le_bytes());
        payload[80..88].copy_from_slice(&self.blocks.to_le_bytes());
        payload[88..92].copy_from_slice(&self.block_size.to_le_bytes());
        payload[92..96].copy_from_slice(&self.flags.to_le_bytes());
        payload[96..100].copy_from_slice(&self.generation.to_le_bytes());
        payload[100..104].copy_from_slice(&self.spare.to_le_bytes());
        payload[104..112].copy_from_slice(&self.birthtime_sec.to_le_bytes());
        payload[112..120].copy_from_slice(&self.birthtime_nsec.to_le_bytes());
        payload
    }
}

fn fallback_inode(inode_seed: &[u8]) -> u32 {
    inode_seed.iter().fold(0x811c_9dc5_u32, |hash, &byte| {
        (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
    })
}

#[cfg(any(test, not(unix)))]
fn kernel_stat_payload(size: u64, is_directory: bool, inode_seed: &[u8]) -> [u8; 120] {
    let size = i64::try_from(size).unwrap_or(i64::MAX);
    GuestKernelStat {
        ino: fallback_inode(inode_seed),
        mode: if is_directory { 0x41ff } else { 0x81ff },
        nlink: 1,
        size,
        blocks: if is_directory {
            128
        } else {
            (size + 511) / 512
        },
        block_size: if is_directory { 65_536 } else { 512 },
        ..GuestKernelStat::default()
    }
    .encode()
}

fn kernel_stat_payload_from_metadata(
    metadata: &std::fs::Metadata,
    inode_seed: &[u8],
) -> [u8; SCE_KERNEL_STAT_SIZE] {
    #[cfg(unix)]
    {
        let (birth_seconds, birth_nanoseconds) = metadata
            .created()
            .ok()
            .and_then(|created| created.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or((metadata.ctime(), metadata.ctime_nsec()), |duration| {
                (
                    i64::try_from(duration.as_secs()).unwrap_or(i64::MAX),
                    i64::from(duration.subsec_nanos()),
                )
            });
        GuestKernelStat {
            dev: u32::try_from(metadata.dev()).unwrap_or(u32::MAX),
            ino: u32::try_from(metadata.ino()).unwrap_or_else(|_| fallback_inode(inode_seed)),
            mode: u16::try_from(metadata.mode() & u32::from(u16::MAX)).expect("masked mode fits"),
            nlink: u16::try_from(metadata.nlink()).unwrap_or(u16::MAX),
            uid: metadata.uid(),
            gid: metadata.gid(),
            rdev: u32::try_from(metadata.rdev()).unwrap_or(u32::MAX),
            atime_sec: metadata.atime(),
            atime_nsec: metadata.atime_nsec(),
            mtime_sec: metadata.mtime(),
            mtime_nsec: metadata.mtime_nsec(),
            ctime_sec: metadata.ctime(),
            ctime_nsec: metadata.ctime_nsec(),
            size: i64::try_from(metadata.size()).unwrap_or(i64::MAX),
            blocks: i64::try_from(metadata.blocks()).unwrap_or(i64::MAX),
            block_size: u32::try_from(metadata.blksize()).unwrap_or(u32::MAX),
            flags: 0,
            generation: 0,
            spare: 0,
            birthtime_sec: birth_seconds,
            birthtime_nsec: birth_nanoseconds,
        }
        .encode()
    }
    #[cfg(not(unix))]
    {
        kernel_stat_payload(metadata.len(), metadata.is_dir(), inode_seed)
    }
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
    let Ok(bounded) = u32::try_from(bounded) else {
        return u64::MAX;
    };
    // SAFETY: usleep has no pointer arguments and the duration is bounded to its host ABI width.
    signed_i32_return(unsafe { libc::usleep(bounded) })
}

fn guest_clock_gettime(clock_id: u64, output: u64) -> u64 {
    if output == 0 {
        return u64::MAX;
    }
    // Console clock ids do not share Darwin's numeric constants. This POSIX compatibility export
    // returns wall-clock time, matching the console runtime's observable calendar-time contract.
    let mut timestamp = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: timestamp is a valid writable host timespec.
    let status = unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &raw mut timestamp) };
    if status == 0 {
        write_guest_u64(output, u64::try_from(timestamp.tv_sec).unwrap_or(0));
        write_guest_u64(output + 8, u64::try_from(timestamp.tv_nsec).unwrap_or(0));
    }
    if trace_guest_time() {
        eprintln!(
            "[REMu][TIME] clock_gettime id={} output={output:#x} status={status} sec={} nsec={}",
            clock_id, timestamp.tv_sec, timestamp.tv_nsec
        );
    }
    signed_i32_return(status)
}

fn guest_kernel_gettimeofday(output: u64) -> u64 {
    if output == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let elapsed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    write_guest_u64(output, elapsed.as_secs());
    write_guest_u64(output + 8, u64::from(elapsed.subsec_micros()));
    0
}

/// Returns the host's UTC offset and daylight-saving component for a Unix timestamp. Console
/// calendar conversions describe the machine's configured civil time zone, so they must not be
/// reduced to UTC-only stubs even though emulated monotonic clocks remain host-independent.
fn guest_local_timezone(unix_seconds: i64) -> (i64, i32) {
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
    DIAGNOSTIC_TIMEOUT_CANCELLED.store(true, Ordering::Release);
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

/// The exit status a bounded diagnostic run reports when its time limit expires.
///
/// This matches what the shell reports for a process killed by `SIGALRM`, which is how the limit
/// used to be enforced, so scripts reading the status keep working.
const DIAGNOSTIC_TIMEOUT_EXIT_STATUS: i32 = 142;

/// Ends the process once a bounded diagnostic run reaches its time limit.
///
/// A watchdog thread is used rather than `alarm`, whose default action terminates the process
/// outright: that discarded every end-of-run report, which is most of the reason for bounding a
/// run in the first place. The thread runs even when every guest thread is wedged, so it remains
/// a backstop against a hang and not merely a timer.
fn start_diagnostic_timeout(timeout_seconds: u32) {
    let limit = Duration::from_secs(u64::from(timeout_seconds));
    std::thread::Builder::new()
        .name("remu-timeout".to_owned())
        .spawn(move || {
            std::thread::sleep(limit);
            if DIAGNOSTIC_TIMEOUT_CANCELLED.load(Ordering::Acquire) {
                return;
            }
            eprintln!("[REMu] diagnostic time limit of {timeout_seconds}s reached");
            terminate_probe(DIAGNOSTIC_TIMEOUT_EXIT_STATUS);
        })
        .ok();
}

fn terminate_probe(code: i32) -> ! {
    report_shader_opcode_coverage();
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

#[cfg(all(test, target_os = "macos", target_arch = "x86_64"))]
mod tests;
