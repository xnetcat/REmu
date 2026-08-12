// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use metal::{
    BinaryArchive, Buffer, CommandBuffer, ComputeCommandEncoder, ComputePipelineState,
    DepthStencilState, Device, Function, MTLBlendFactor, MTLBlendOperation, MTLCompareFunction,
    MTLCullMode, MTLPixelFormat, MTLPrimitiveTopologyClass, MTLScissorRect, MTLStencilOperation,
    MTLTriangleFillMode, MTLWinding, MetalLayer, RenderCommandEncoder, RenderPipelineState,
    SamplerState, Texture,
};
use remu_gpu::{
    ComputeDispatch, GpuEvent, GpuFlipWait, GpuMemoryRelease, GpuMemoryWait, GpuMemoryWrite,
    IndexedDraw,
};
use remu_memory::{
    CpuWriteWatch, CpuWriteWatchRetirement, GuestSharedBackingRange, GuestSharedMappingIdentity,
};
mod attachment;
mod cache;
mod completion;
mod decode;
mod diagnostics;
mod display_link;
mod execute;
mod format;
mod host_ui;
mod metal_bytes;
mod present;
mod readback;
mod sampler;
mod shader;
mod shared_event;
mod stats;
mod submission;
mod telemetry;
mod texture;
mod tiler;
mod tiling;
use self::cache::{GLOBAL_UPLOAD_ALIGNMENT, GLOBAL_UPLOAD_CHUNK_BYTE_COUNT};
use self::decode::{
    CB_BLEND_ALPHA, CB_BLEND_BLUE, CB_BLEND_GREEN, CB_BLEND_RED, CB_BLEND0_CONTROL,
    CB_COLOR_CONTROL, CB_COLOR_REGISTER_STRIDE, CB_COLOR0_FMASK, CB_COLOR0_FMASK_BASE_EXT,
    CB_TARGET_MASK, ColorOperationMode, DB_DEPTH_CONTROL, DB_DEPTH_SIZE_XY, DB_Z_INFO,
    DEFAULT_CB_COLOR_CONTROL, DEFAULT_PA_SU_SC_MODE_CNTL, NGG_MESHLET_WORDS_PER_GROUP,
    PA_SC_AA_MASK_X0Y0_X1Y0, PA_SC_AA_MASK_X0Y1_X1Y1, PA_SU_SC_MODE_CNTL,
    color_operation_is_raster_noop, decode_color_eliminate_fast_clear_target,
    decode_color_operation_mode, decode_color_resolve_targets, decode_depth_target,
    decode_depth_target_without_diagnostics, decode_image_descriptor,
    decode_image_descriptor_without_diagnostics, decode_metal_blend_attachment,
    decode_raster_state, decode_render_target_format, decode_render_targets, decode_scissor,
    decode_viewport, fallback_image_descriptor, graphics_pipeline_key, image_component_class,
    initial_depth_clear_for_target, initial_render_target_clear_color, metal_color_write_mask,
    metal_nonindexed_vertex_count, metal_primitive_type, ngg_mesh_base_vertex,
    noop_placeholder_cover_draw, prepare_indexed_ngg_indices, prepare_indexed_primitive,
    prepare_ngg_meshlets, prepare_nonindexed_ngg_indices, raw_video_out_texture_descriptor,
    read_draw_indices, register_value, render_target_cmask_address, render_target_fast_clear_color,
    sampled_alias_descriptor,
};
#[cfg(test)]
use self::decode::{
    CB_COLOR0_ATTRIB, CB_COLOR0_ATTRIB2, CB_COLOR0_ATTRIB3, CB_COLOR0_BASE, CB_COLOR0_BASE_EXT,
    CB_COLOR0_CLEAR_WORD0, CB_COLOR0_CLEAR_WORD1, CB_COLOR0_CMASK, CB_COLOR0_INFO, CB_COLOR0_VIEW,
    DB_DEPTH_CLEAR, DB_DEPTH_VIEW, DB_HTILE_DATA_BASE, DB_RENDER_CONTROL, DB_STENCIL_CONTROL,
    DB_STENCILREFMASK, DB_Z_READ_BASE, DB_Z_READ_BASE_HI, DB_Z_WRITE_BASE, DB_Z_WRITE_BASE_HI,
    SPI_PS_INPUT_CNTL_0, VIDEO_OUT_DISPLAY_TILE_MODE, VIDEO_OUT_TILING_MODE_LINEAR,
    VIDEO_OUT_TILING_MODE_TILE, apply_transparent_premultiplied_fill_clear,
    common_render_target_array_length, decode_fast_clear_color, decode_graphics_wave_state,
    decode_ngg_registers_from_state, decode_ngg_subgroup_limits,
    decode_ngg_subgroup_limits_with_uc, decode_render_targets_from_registers,
    decode_scanout_target, disabled_metal_blend_attachment, graphics_shader_resource_key,
    initial_depth_clear_value, is_transparent_premultiplied_fill_blend,
    metal_primitive_topology_class, ngg_fullscreen_cover_draw, pixel_input_controls,
    raw_video_out_layout_is_linear,
};
#[cfg(test)]
use self::diagnostics::parse_diagnostic_hex_address;
use self::diagnostics::{
    DIAGNOSTIC_MESH_POSITION_BUFFER_INDEX, DIAGNOSTIC_PIXEL_OUTPUT_BUFFER_INDEX,
    DIAGNOSTIC_PIXEL_OUTPUT_HEIGHT, DIAGNOSTIC_PIXEL_OUTPUT_WIDTH,
    DIAGNOSTIC_VERTEX_POSITION_BUFFER_INDEX, DIAGNOSTIC_VERTEX_POSITION_VERTEX_CAPACITY,
    DIAGNOSTIC_VERTEX_STATE_WORDS_PER_VERTEX, capture_target_sample_textures,
    diagnostic_capture_pixel_output, diagnostic_capture_vertex_position,
    diagnostic_compute_texture_dispatch_is_eligible, diagnostic_context_register_snapshot,
    diagnostic_disable_blending, diagnostic_disable_culling, diagnostic_disable_depth,
    diagnostic_disable_depth_and_culling, diagnostic_dump_complex_msaa,
    diagnostic_dump_compute_texture_minimum_dispatch, diagnostic_dump_depth_export_shader,
    diagnostic_dump_gbuffer, diagnostic_dump_glyph_draw, diagnostic_dump_shadow_consumer,
    diagnostic_dump_target_address, diagnostic_dump_target_address_range,
    diagnostic_dump_target_before_pixel_shader, diagnostic_dump_target_index_count,
    diagnostic_dump_target_match_index, diagnostic_dump_target_minimum_draw,
    diagnostic_dump_target_mrt_min, diagnostic_dump_target_pixel_instruction_count,
    diagnostic_dump_target_pixel_shader, diagnostic_dump_video_chain_consumer,
    diagnostic_dump_video_draw, diagnostic_dump_video_target_consumer,
    diagnostic_force_msaa_pixel_color, diagnostic_force_msaa_vertex_pattern,
    diagnostic_force_vertex_fetch_id, diagnostic_force_visible_clip_depth,
    diagnostic_global_binding_head, diagnostic_note_full_resolution_mrt_draw,
    diagnostic_overlapping_global_writers, diagnostic_overlapping_memory_copies,
    diagnostic_overlapping_texture_writers, diagnostic_related_texture_writers,
    diagnostic_scanout_writer, diagnostic_scanout_writer_slot,
    diagnostic_scanout_writer_target_selected, diagnostic_shader_summary,
    diagnostic_skip_export_shader, diagnostic_skip_pixel_shader, diagnostic_target_address_matches,
    diagnostic_target_after_mrt_ready, diagnostic_target_draw_filter, diagnostic_texture_writer,
    diagnostic_texture_writer_enabled, diagnostic_trace_all_target_contents,
    diagnostic_trace_target_contents, diagnostic_trace_target_maximum_draw,
    diagnostic_trace_target_stride, diagnostic_uploaded_global_binding_head,
    diagnostic_use_draw_merged_wave_counts, diagnostic_video_chain_selected_draw,
    draw_diagnostics_are_host_effectful, dump_msl_compute_filter_matches,
    dump_msl_instruction_filter_matches, dump_msl_pixel_filter_matches,
    evaluation_samples_glyph_atlas, evaluation_samples_large_linear_video,
    evaluation_samples_layered_comparison, record_diagnostic_global_writers,
    record_diagnostic_memory_copy, record_diagnostic_scanout_writer,
    record_diagnostic_texture_writer, should_record_diagnostic_target_writer, trace_glyph_atlas,
    trace_image_descriptors, trace_msaa_writes, trace_scanout_writes, trace_target_sampling,
    trace_video_target_compute_consumer, trace_video_target_consumers,
    trace_video_target_draw_consumer, video_target_binding_summary,
};
#[cfg(test)]
use self::execute::{export_resource_render_stage, ngg_mesh_dispatch_required};
use self::format::{
    compressed_texture_block_extent, decode_sampled_texture_format, half_float_to_f32,
    half_float_to_srgb_unorm8, half_float_to_unorm8, linear_unorm8_to_srgb_unorm8,
    texture_element_layout, unsigned_float_to_f32, unsigned_float_to_srgb_unorm8,
};
use self::metal_bytes::MetalBytes;
#[cfg(test)]
use self::present::{
    appkit_persistent_state_is_ignored, clear_appkit_persistent_state_override,
    headless_metal_value_is_enabled, parse_presentation_resolution, presentation_drawable_size,
    presentation_viewport,
};
use self::present::{
    create_presentation_resources, disable_appkit_state_restoration, headless_metal_requested,
    run_headless_presenter,
};
use self::readback::{
    WRITEBACK_TRIGGER_COUNT, WRITEBACK_TRIGGERS, WritebackTrigger, capture_guest_frame,
    diagnostic_f32_bits_magnitude_nonzero, diagnostic_texture_depth_plane_summaries,
    diagnostic_texture_magnitude_nonzero_texels, diagnostic_texture_numeric_range,
    guest_frame_fingerprint, metal_uncompressed_bytes_per_pixel, native_texture_capture_to_bgra,
    read_texture_array_linear, read_texture_linear, read_texture_linear_after_event,
    read_texture_volume_linear,
};
#[cfg(test)]
use self::sampler::sampler_lod_bias;
use self::sampler::{
    SAMPLER_ARGUMENT_CACHE_CAPACITY, canonical_sampler_bindings, canonical_texture_bindings,
    create_global_argument_buffer, create_sampler_argument_buffer, metal_buffer_with_bytes,
    sampled_image_bindings, sampler_uses_normalized_coordinates,
};
use self::shader::{
    SHADER_COMPILE_QUEUE_CAPACITY, compile_metal_compute_library, compile_metal_library_with_tools,
    compute_pipeline_key, compute_program_can_split_wave64, dump_shader_pair_msl,
    effective_global_memory_binding, fragmentless_pixel_evaluation, fragmentless_pixel_program,
    graphics_allocated_vector_register_count, image_resource_descriptor_is_null,
    is_storage_image_opcode, is_writable_image_opcode, ngg_memoryless_cover_export_stage,
    ngg_mesh_export_stage, ngg_mesh_stage_for_draw, pixel_scalars,
    procedural_rectangle_stage_for_draw, program_embeds_global_base,
    log_wave64_hazard_census, shader_scalar_metadata_for_registers,
    validate_compute_launch_for_metal, wave64_split_blockers,
};
#[cfg(test)]
use self::shader::{
    lift_expanded_vertex_buffer_num_records, missing_vertex_parameter_exports,
    ngg_export_uses_general_lds, ngg_export_uses_wave64_exec, ngg_mesh_draw_supported,
    procedural_rectangle_export_stage, reconcile_pixel_input_controls_with_export_producers,
    rectangle_coordinate_parameter_mask, rectangle_draw_has_incomplete_interface,
    shader_scalar_metadata, truncated_memoryless_ngg_rectangle_cover,
    unmatched_param_default_control,
};
use self::shared_event::{
    GpuWaitFenceMap, SharedEventFence, SharedEventProducerId, SharedEventTimeline,
};
#[cfg(test)]
use self::stats::process_resident_bytes;
use self::stats::{
    MIP_UPLOAD_BYTES, MIP_UPLOAD_COUNT, MIP_UPLOAD_NANOSECONDS, MODE27_DECODE_BYTES,
    MODE27_DECODE_COUNT, MODE27_DECODE_NANOSECONDS, OWNED_DEMOTIONS, OWNED_FALLBACK_FULL_RANGE,
    OWNED_RANGE_SET_COARSENED, OWNED_RECLAIM_DISCARD, OWNED_REMATERIALISE_BLIT,
    OWNED_REMATERIALISE_CPU, OWNED_REMATERIALISE_DETILE, OWNED_REMATERIALISE_FAILED,
    OWNED_REMATERIALISE_FAILED_CPU_GPU, OWNED_REMATERIALISE_FAILED_LAYOUT,
    OWNED_REMATERIALISE_FAILED_MAPPING, OWNED_REMATERIALISE_FAILED_METAL, OWNED_REMATERIALISE_VIEW,
    OWNED_SUBTRACT_TENANTS_SURVIVED, OWNED_UNRESOLVED_SPAN, PIPELINE_COMPILE_COMPLETED,
    PIPELINE_COMPILE_ENQUEUED, PIPELINE_COMPILE_FAILED, PIPELINE_COMPILE_QUEUE_FULL,
    PIPELINE_COMPUTE_GATES, PIPELINE_DRAW_GATES, RESIDENT_DEPTH_SAMPLE_HITS,
    RESIDENT_TEXTURE_ACQUIRE_INVALIDATIONS, RESIDENT_WRITE_ATTRIBUTION_MISSES,
    RESIDENT_WRITEBACK_DISCARDED, RESIDENT_WRITEBACK_DISCARDED_BYTES, RESIDENT_WRITEBACK_PUBLISHED,
    RESIDENT_WRITEBACK_PUBLISHED_BYTES, SAMPLED_ALIAS_WRITEBACK_BYTES,
    SAMPLED_ALIAS_WRITEBACK_COUNT, SAMPLED_ALIAS_WRITEBACK_NANOSECONDS,
    SAMPLED_TEXTURE_CACHE_ACQUIRE_INVALIDATIONS, SAMPLED_TEXTURE_CACHE_CAPACITY_EVICTIONS,
    SAMPLED_TEXTURE_CACHE_DIRTY_RELOADS, SAMPLED_TEXTURE_CACHE_HITS, SAMPLED_TEXTURE_CACHE_MISSES,
    SharedEventAckPhaseStatistics, SharedEventHostEffectLatencyPhase,
    SharedEventHostEffectLatencyStatistics, SharedEventHostEffectPendingStatistics,
    TEXTURE_DIRTY_VALIDATION_BYTES, TEXTURE_DIRTY_VALIDATION_COUNT,
    TEXTURE_DIRTY_VALIDATION_NANOSECONDS, TEXTURE_DIRTY_VALIDATION_UNCHANGED,
    TEXTURE_SNAPSHOT_BYTES, TEXTURE_SNAPSHOT_COUNT, TEXTURE_SNAPSHOT_NANOSECONDS,
    TEXTURE_SNAPSHOT_UNSTABLE_COUNT, TerminalPublicationPhase, TerminalPublicationStatistics,
    TerminalPublicationTimer, format_shared_event_ack_phase_report,
    format_shared_event_host_effect_latency_report, format_terminal_publication_report,
    process_memory_statistics, record_mode27_work_detail, record_texture_dirty_work_detail,
    record_texture_snapshot_work_detail, record_texture_work, take_mode27_work_details,
    take_texture_dirty_work_details, take_texture_snapshot_work_details,
};
use self::submission::{
    SAME_QUEUE_DMA_FILL_REJECT_COUNT, SHARED_EVENT_DRAW_REJECT_COUNT, SameQueueDmaFillReject,
    SharedEventDrawReject, append_unique_global_writebacks, append_unique_guest_write_ranges,
    collect_pending_global_writebacks, collect_pending_guest_write_ranges,
    collect_shared_event_global_versions, completed_command_buffer_gpu_duration,
    describe_guest_batch, guest_batch_writes_overlap,
};
#[cfg(test)]
use self::submission::{
    apply_pending_guest_memory_writes, pin_guest_memory_write, write_back_pending_globals,
};
use self::texture::{
    RESIDENT_FLIP_AGE_LABELS, RESIDENT_WORK_AGE_LABELS, RangeSet, ResidentTextureMap,
    SpanQueryOrder, TextureWritebackOutcome, WrittenRegion, decode_guest_texture_linear_from_raw,
    refresh_resident_texture_write_watch, resident_color_view_sample_is_eligible,
    resident_declared_span, resident_depth_sample_is_eligible,
    resident_same_image_sample_is_eligible, resident_texture_activity_stats,
    resident_texture_backing_overlap_stats, resident_texture_ownership_stats,
    resident_texture_raw_byte_count, retire_multisample_textures_overlapping_writeback,
    retire_sampled_textures_matching, retire_sampled_textures_overlapping_writeback,
    sampled_texture_raw_range, snapshot_guest_texture_raw, snapshot_watched_guest_range,
    storage_image_requires_mip_hierarchy, trace_resident_texture_removal, upload_image_textures,
    write_guest_texture_linear_into_raw_at_z, write_watched_guest_range,
    write_watched_guest_texture_linear,
};
#[cfg(test)]
use self::texture::{
    apply_image_destination_select, comparison_depth32_upload,
    create_exact_mode27_sampled_mip_view, create_sampled_guest_texture,
    decode_sampled_texture_linear_from_raw, depth_array_face_descriptor,
    depth_array_face_set_is_complete, ensure_mipmapped_resident_texture_in_map,
    guest_texture_raw_matches, guest_texture_raw_range, image_destination_select_is_native,
    new_color_alias_texture_view, new_red_blue_swizzled_texture_view,
    prepare_sampled_texture_upload, rematerialisation_blit_source_origin_y,
    resident_texture_eviction_plan, sampled_texture_cache_accepts,
    sampled_texture_cache_needs_reload, sampled_texture_cache_requires_eviction,
    sampled_texture_key, snapshot_guest_texture_linear, snapshot_watched_guest_texture_raw,
    watch_guest_texture, write_guest_texture_linear,
};
use self::tiling::{
    ResourceSpan, copy_linear_to_tiled_at_z, copy_mode27_mip_level_to_linear,
    copy_tiled_to_linear_at_z, decode_mode5_mip_chain_from_raw, decode_mode27_mip_chain_from_raw,
    depth_sample_view_must_not_steal_depth_authority, effective_linear_pitch,
    encode_mode27_mip_chain_into_raw, guest_physical_ranges_overlap, guest_ranges_overlap,
    guest_texture_byte_count, guest_texture_descriptors_have_same_layout,
    guest_texture_descriptors_overlap, guest_texture_descriptors_share_byte_storage,
    guest_texture_descriptors_share_color_view_storage,
    guest_texture_descriptors_share_depth_sample_storage,
    guest_texture_descriptors_share_gpu_sample_storage, guest_texture_descriptors_share_storage,
    mode5_mip_chain_layout, mode27_mip_chain_layout, mode27_surface_z_xor,
    render_resolve_candidate_matches, spans_overlap, tiled_block_bytes, tiled_block_dimensions,
};
#[cfg(test)]
use self::tiling::{
    encode_mode5_mip_chain_into_raw, guest_texture_descriptors_share_component_reordered_storage,
    mode5_within_block_byte_offset, mode9_within_block_byte_offset,
    mode24_within_block_byte_offset_for_format, mode27_block_dimensions, mode27_source_byte_offset,
    mode27_within_block_byte_offset, mode27_within_block_byte_offset_for_format,
};

#[cfg(test)]
use remu_memory::guest_shared_mapping_identity;
use remu_shader::{PixelExportFormat, PixelOutputKind};
use winit::event_loop::{ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::KeyCode;
use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
use winit::window::Window;
const EVENT_POLL_INTERVAL: Duration = Duration::from_millis(1);
const SHARED_HOST_KIND_ACQUIRE: usize = 0;
const SHARED_HOST_KIND_DRAW: usize = 1;
const SHARED_HOST_KIND_COMPUTE: usize = 2;
const SHARED_HOST_KIND_DMA: usize = 3;
const SHARED_HOST_KIND_WAIT: usize = 4;
const SHARED_HOST_KIND_FLIP: usize = 5;
const SHARED_HOST_KIND_UI: usize = 6;
const SHARED_HOST_KIND_OTHER: usize = 7;
const SHARED_HOST_KIND_COUNT: usize = 8;
/// A late Superliminal frame contains roughly 700 draws. Draining only 16 events per AppKit
/// callback imposed an artificial ~2k draw/s ceiling even when both the guest producer and Metal
/// encoder had capacity. This bound covers most of a frame while still returning to AppKit often
/// enough to service input and window events.
const GPU_EVENTS_PER_TICK: usize = 1024;
const GPU_EVENTS_PER_TICK_BACKLOG_THRESHOLD: usize = 256;
const GPU_EVENTS_PER_TICK_BACKLOG_MULTIPLIER: usize = 2;
// Unity can introduce dozens of distinct 200-400 KiB pixel programs at a scene boundary. Four
// workers let the ordered GPU queue grow by several thousand events while Apple's compiler is
// busy, which in turn withholds real guest flips and starves the guest audio producers. Each
// offline compiler is a separate memory-heavy process, so eight workers can amplify a large UE
// transition into system-wide pressure even when REmu's own RSS still looks reasonable.
/// Keep individual Metal submissions short enough for the shared Apple GPU to return regularly
/// to WindowServer. Unreal transition scenes can otherwise put hundreds of full-resolution draws
/// in one command buffer and monopolize the device for seconds.
const GUEST_BATCH_WORK_CAPACITY: usize = 64;

/// Maximum number of committed guest batches allowed ahead of the host GPU.
///
/// Each batch retains its transient uploads, render targets, and writeback resources until Metal
/// completes it. Bounding the queue prevents a stalled or simply overloaded game from growing
/// unified-memory use indefinitely and making the whole system unresponsive.
const MAX_COMMITTED_GUEST_BATCHES: usize = 4;
const SAMPLED_TEXTURE_CACHE_CAPACITY: usize = 2_048;
const DEFAULT_SAMPLED_TEXTURE_CACHE_MIB: u64 = 1_536;
/// Persistent GPU-authored render/storage images retain the architectural contents of guest
/// allocations whose tiled RAM is stale. Pressure collection is deliberately opt-in until title
/// telemetry establishes a safe working-set policy.
///
/// Unlike the sampled-image cache, pressure eviction of a GPU-owned resident first downloads and
/// retiles its exact contents into guest memory. This is a cache budget, not a correctness cap.
const MAX_RESIDENT_TEXTURE_EVICTIONS_PER_FLIP: usize = 32;
const DEFAULT_SHADER_COMPILER_WORKER_COUNT: usize = 4;
const METAL_TEXTURE_BUFFER_ROW_ALIGNMENT: u64 = 256;
const NGG_MESHLET_BUFFER_INDEX: u64 = 29;
const NGG_MESH_DRAW_BUFFER_INDEX: u64 = 30;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum NggOutputPrimitiveType {
    PointList,
    LineStrip,
    TriangleStrip,
    RectangleList,
}

/// Guest-programmed GFX10 NGG subgroup geometry limits.
///
/// A Metal mesh threadgroup still executes one guest wave64. These values describe how many
/// lanes of that wave carry input geometry and how much output storage the guest subgroup may
/// address; they are deliberately independent from the host threadgroup width.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct NggSubgroupLimits {
    es_vertices_per_subgroup: u32,
    gs_primitives_per_subgroup: u32,
    gs_instance_primitives_per_subgroup: u32,
    gs_instance_count: u32,
    /// Per-input-primitive API geometry-shader output limit.
    ///
    /// Zero means that no API GS limit is active. NGG passthrough primitive shaders still have a
    /// non-zero subgroup output capacity in `max_output_vertices_per_subgroup`.
    gs_max_vertices_out: u32,
    max_output_vertices_per_subgroup: u32,
    primitive_amplification_factor: u32,
    max_output_primitives_per_subgroup: u32,
    output_primitive_type: Option<NggOutputPrimitiveType>,
    threadgroup_threads: u32,
}

fn sampled_texture_cache_byte_capacity() -> Option<u64> {
    static CAPACITY: OnceLock<Option<u64>> = OnceLock::new();
    *CAPACITY.get_or_init(|| {
        let capacity_mib = std::env::var("REMU_SAMPLED_TEXTURE_CACHE_MIB")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_SAMPLED_TEXTURE_CACHE_MIB);
        (capacity_mib != 0).then(|| capacity_mib.saturating_mul(1024 * 1024))
    })
}

fn resident_texture_cache_byte_capacity() -> Option<u64> {
    static CAPACITY: OnceLock<Option<u64>> = OnceLock::new();
    *CAPACITY.get_or_init(|| {
        std::env::var("REMU_RESIDENT_TEXTURE_CACHE_MIB")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|&capacity_mib| capacity_mib != 0)
            .map(|capacity_mib| capacity_mib.saturating_mul(1024 * 1024))
    })
}

/// Graduated flags (plan A1) default on; the environment value survives one cycle as a
/// kill-switch, where only an explicit `0` disables the behaviour.
fn graduated_flag_enabled(value: Option<&std::ffi::OsStr>) -> bool {
    value != Some(std::ffi::OsStr::new("0"))
}

fn resident_texture_lifetime_diagnostics_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_TEXTURE_LIFETIMES").is_some_and(|value| value != "0")
    })
}

static RESIDENT_TEXTURE_ACTIVITY_LATEST_WORK: AtomicU64 = AtomicU64::new(0);

fn resident_texture_activity_latest_work() -> u64 {
    RESIDENT_TEXTURE_ACTIVITY_LATEST_WORK.load(Ordering::Relaxed)
}

fn shader_compiler_worker_count() -> usize {
    static COUNT: OnceLock<usize> = OnceLock::new();
    *COUNT.get_or_init(|| {
        std::env::var("REMU_SHADER_COMPILER_WORKERS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_SHADER_COMPILER_WORKER_COUNT)
            .clamp(1, 8)
    })
}

/// Leave one compiler worker available for an actual ordered queue-front miss.
///
/// Speculative requests never need to occupy the bounded channel: at most `workers - 1` can be
/// active, so a required front can be received immediately by the reserved worker. A single-worker
/// configuration disables speculative compilation rather than placing it ahead of required work.
fn speculative_pipeline_compile_in_flight_capacity() -> usize {
    shader_compiler_worker_count().saturating_sub(1)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PresenterFlow {
    Continue,
    Shutdown,
}

fn capture_interval() -> Duration {
    static INTERVAL: OnceLock<Duration> = OnceLock::new();
    *INTERVAL.get_or_init(|| {
        std::env::var("REMU_CAPTURE_INTERVAL_SECONDS")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|seconds| seconds.is_finite() && *seconds >= 0.1)
            .map_or(Duration::from_secs(1), Duration::from_secs_f64)
    })
}

// GFX10 context-register offsets are the MMIO dword address minus 0xA000. Color registers
// 0..7 use a 15-dword primary stride; the high address and GFX10 extent/swizzle registers are
// stored in separate dense eight-register arrays. These values match Mesa's generated gfx10
// register database and SharpEmu's AGC command-stream decoder.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PresentationResolution {
    LogicalWindow,
    NativeWindow,
    Fixed(u32, u32),
}

struct PresentationResources {
    /// Drawable + pipeline for linear float scanout (Unity RGBA16F): Metal writes sRGB.
    srgb_pipeline: RenderPipelineState,
    /// Drawable + pipeline for display-referred Unorm scanout (VT RGB10A2): passthrough.
    unorm_pipeline: RenderPipelineState,
    /// Alpha-blended system UI overlay for sRGB drawables.
    overlay_srgb_pipeline: RenderPipelineState,
    /// Alpha-blended system UI overlay for Unorm drawables.
    overlay_unorm_pipeline: RenderPipelineState,
    overlay_texture: Texture,
    _vertex_function: Function,
    _fragment_function: Function,
    _overlay_fragment_function: Function,
    sampler: SamplerState,
}

#[derive(Clone, Copy)]
struct DisplayConfiguration {
    width: u32,
    height: u32,
    pitch: u32,
    pixel_format: u64,
    tiling_mode: u32,
}

#[derive(Clone)]
struct HostSplashTexture {
    texture: Texture,
    width: u32,
    height: u32,
}

#[derive(Clone, Copy)]
struct PendingFlip {
    address: u64,
    buffer_index: u32,
    flip_argument: u64,
    submission_sequence: Option<u64>,
    sequence: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FlipTerminalSource {
    Presentation,
    Superseded,
    SurfaceLess,
}

impl FlipTerminalSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Presentation => "presentation",
            Self::Superseded => "superseded",
            Self::SurfaceLess => "surface-less",
        }
    }
}

struct FlipCompletionJoin {
    completion: remu_gpu::GpuFlipCompletion,
    submission_sequence: Option<u64>,
    submission_terminal: bool,
    terminal_claimed: bool,
    terminal_source: Option<FlipTerminalSource>,
    accepted_at: Instant,
}

#[derive(Default)]
struct FlipCompletionJoinStatistics {
    accepted: u64,
    deferred: u64,
    published: u64,
}

struct FlipBufferSafety {
    address: u64,
    buffer_index: u32,
    snapshot_command_buffer: Option<CommandBuffer>,
    snapshot_ready: Option<SharedEventFence>,
    safe: bool,
    accepted_at: Instant,
}

#[derive(Clone, Copy)]
struct ActiveTranslatedScanout {
    address: u64,
    width: u32,
    height: u32,
    sequence: u64,
    /// Orders the guest-queue snapshot copy before legacy-queue presentation/tail work.
    snapshot_ready: Option<SharedEventFence>,
    /// Resident `content_version` copied into the private scanout snapshot texture.
    snapshotted_content_version: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GuestTextureDescriptor {
    address: u64,
    width: u32,
    height: u32,
    pitch: u32,
    tile_mode: u32,
    pixel_format: MTLPixelFormat,
    bytes_per_pixel: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Mode27WorkOrigin {
    RenderTargetCreate,
    SampledCacheCreate,
    StorageCreate,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Mode27WorkKey {
    origin: Mode27WorkOrigin,
    descriptor: GuestTextureDescriptor,
    mip_level_count: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Mode27WorkAggregate {
    count: u64,
    bytes: u64,
    nanoseconds: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct TextureSnapshotWorkKey {
    address: u64,
    byte_count: usize,
    caller_line: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct TextureDirtyWorkAggregate {
    snapshots: u64,
    clean_snapshots: u64,
    imprecise_snapshots: u64,
    dirty_pages: u64,
    dirty_bytes: u64,
    full_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TextureReadbackLayout {
    packed_bytes_per_row: u64,
    staging_bytes_per_row: u64,
    staging_length: u64,
}

struct PendingTextureReadback {
    command_buffer: CommandBuffer,
    staging: Buffer,
    // Keep the source alive independently of Metal command-buffer retention policy.
    _source: Texture,
    layout: TextureReadbackLayout,
    height: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct RenderTargetFormat {
    pixel_format: MTLPixelFormat,
    output_kind: PixelOutputKind,
    bytes_per_pixel: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct RenderTargetPipelineState {
    guest_slot: u32,
    pixel_format: MTLPixelFormat,
    output_kind: PixelOutputKind,
    export_format: PixelExportFormat,
    write_mask: u32,
    blend: MetalBlendAttachmentState,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct DepthTargetPipelineState {
    pixel_format: MTLPixelFormat,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MetalBlendAttachmentState {
    enabled: bool,
    source_rgb: MTLBlendFactor,
    destination_rgb: MTLBlendFactor,
    rgb_operation: MTLBlendOperation,
    source_alpha: MTLBlendFactor,
    destination_alpha: MTLBlendFactor,
    alpha_operation: MTLBlendOperation,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
struct GraphicsDiagnosticPipelineState {
    force_pixel_color: bool,
    capture_pixel_output: bool,
    force_visible_clip_depth: bool,
    force_vertex_pattern: bool,
    force_vertex_fetch_id: bool,
    capture_vertex_position: bool,
}

/// Live graphics temp-ring and wave-width state, decoded from AGC registers.
///
/// Scratch is one ring for the whole graphics pipe, so the per-lane extent is the per-wave
/// allocation divided by the stage's wave width. `export_lane_count` is `None` when the draw never
/// programmed `CxVgtShaderStagesEn`, which is not the same as knowing the stage runs wave32.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
struct GraphicsWaveState {
    scratch_bytes_per_wave: u32,
    export_lane_count: Option<u32>,
}

impl GraphicsWaveState {
    /// Lane-private scratch bytes for a stage executing `lane_count` architectural lanes.
    const fn scratch_bytes_per_lane(self, lane_count: u32) -> u32 {
        self.scratch_bytes_per_wave / lane_count
    }

    /// Lane-private scratch bytes for the pixel stage, whose wave width no SDK register reports.
    ///
    /// The two candidate widths give per-lane extents that differ by a factor of two, and the
    /// backend bounds-checks every SCRATCH access against this value. Dividing by the narrower
    /// wave therefore yields an extent that is never smaller than the architectural one: a wave64
    /// pixel shader over-allocates its private array but no in-range guest access is ever clamped
    /// to zero. Halving it instead would silently zero the upper half of every spill.
    const fn pixel_scratch_bytes_per_lane(self) -> u32 {
        self.scratch_bytes_per_lane(32)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct GraphicsPipelineKey {
    export_shader_address: u64,
    pixel_shader_address: u64,
    /// Shader code allocations are recycled by Unreal during level and movie transitions.
    /// Guest addresses alone therefore do not identify a compiled program. Retain the decoded
    /// program allocations so re-registering different code at the same address cannot select a
    /// stale Metal pipeline.
    export_program_identity: ProgramIdentity,
    pixel_program_identity: ProgramIdentity,
    export_resources: GraphicsShaderResourceKey,
    pixel_resources: GraphicsShaderResourceKey,
    export_allocated_vector_register_count: u32,
    pixel_allocated_vector_register_count: u32,
    vertex_output_control: u32,
    /// Raw `PA_CL_VTE_CNTL` position-format state used to reconstruct Metal clip coordinates.
    viewport_transform_control: u32,
    pixel_input_controls: [u32; 32],
    /// Live NGG subgroup sizing registers. Non-NGG pipelines leave this absent; NGG compilation
    /// and meshlet preparation require it rather than assuming that execution wave width is also
    /// the guest input/output geometry capacity.
    ngg_subgroup_limits: Option<NggSubgroupLimits>,
    /// Raw SDK-defined registers proving the merged NGG stage's allocation and output bounds.
    /// The shader backend uses their presence to lower architectural empty-subgroup sequences;
    /// an incomplete command state remains a loud pipeline refusal.
    ngg_registers: Option<remu_shader_metal::NggRegisters>,
    /// `SPI_TMPRING_SIZE` scratch allocation and `CxVgtShaderStagesEn` export wave width. Both
    /// change the generated Metal function, so they belong in the pipeline cache key.
    wave_state: GraphicsWaveState,
    /// `DB_SHADER_CONTROL.Z_EXPORT_ENABLE`, gated by an active depth attachment.
    pixel_depth_export: bool,
    /// Diagnostic lowerings alter the generated Metal function body and interface. Keep the
    /// effective options in the cache key so a shader first seen outside the selected diagnostic
    /// draw cannot supply a hook-free pipeline to the later capture.
    diagnostics: GraphicsDiagnosticPipelineState,
    targets: Vec<RenderTargetPipelineState>,
    depth: Option<DepthTargetPipelineState>,
    sample_count: u32,
    render_target_array_length: u32,
    /// Shader stage selection depends on draw topology as well as program identity. Keep the
    /// resolved choice in the pipeline key so asynchronous compilation cannot reuse a vertex
    /// pipeline for the same NGG program when another draw requires mesh lowering.
    uses_mesh_stage: bool,
    /// Procedural rectangles synthesize a different vertex function from ordinary triangles even
    /// when their guest programs and Metal topology class are otherwise identical.
    uses_procedural_rectangle_stage: bool,
    primitive_topology_class: MTLPrimitiveTopologyClass,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct GraphicsShaderResourceKey {
    initial_scalar_count: usize,
    global_bindings: Vec<(Vec<u32>, u64, usize, bool)>,
    image_bindings: Vec<(u32, u32, bool, bool, u32, u32, u32, Option<u32>)>,
    vertex_inputs: Vec<(u32, u32, u32, u32, u32, u32, u32)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DecodedRenderTarget {
    guest_slot: u32,
    texture: GuestTextureDescriptor,
    /// `texture`'s guest bytes, resolved once here so no per-draw path resolves them per
    /// candidate. `None` when the descriptor's byte count overflows.
    span: Option<ResourceSpan>,
    output_kind: PixelOutputKind,
    write_mask: u32,
    direct_scanout: bool,
    sample_count: u32,
    texture_type: u32,
    layer_count: u32,
    base_layer: u32,
    mip_level: u32,
    mip_level_count: u32,
    dcc_enabled: bool,
    fast_clear_enabled: bool,
    /// `CB_COLOR_INFO.BLEND_BYPASS` — SDK requires this for integer targets.
    blend_bypass: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DiagnosticScanoutWriter {
    draw: u64,
    export_shader: u64,
    pixel_shader: u64,
    queue_id: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DecodedStencilFace {
    compare: MTLCompareFunction,
    stencil_fail: MTLStencilOperation,
    depth_fail: MTLStencilOperation,
    depth_pass: MTLStencilOperation,
    read_mask: u32,
    write_mask: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DecodedStencilState {
    enabled: bool,
    separate_back: bool,
    front: DecodedStencilFace,
    back: DecodedStencilFace,
    front_reference: u32,
    back_reference: u32,
}

impl Default for DecodedStencilState {
    fn default() -> Self {
        let face = DecodedStencilFace {
            compare: MTLCompareFunction::Never,
            stencil_fail: MTLStencilOperation::Keep,
            depth_fail: MTLStencilOperation::Keep,
            depth_pass: MTLStencilOperation::Keep,
            read_mask: 0,
            write_mask: 0,
        };
        Self {
            enabled: false,
            separate_back: false,
            front: face,
            back: face,
            front_reference: 0,
            back_reference: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DecodedDepthTarget {
    texture: GuestTextureDescriptor,
    span: Option<ResourceSpan>,
    layer_count: u32,
    htile_address: Option<u64>,
    compare: MTLCompareFunction,
    test_enabled: bool,
    write_enabled: bool,
    clear_enabled: bool,
    clear_depth: f64,
    sample_count: u32,
    stencil: DecodedStencilState,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DecodedDepthBias {
    slope_scale: f32,
    constant: f32,
    clamp: f32,
}

impl Default for DecodedDepthBias {
    fn default() -> Self {
        Self {
            slope_scale: 0.0,
            constant: 0.0,
            clamp: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct DepthStencilCacheKey {
    compare: u64,
    write_enabled: bool,
    stencil_enabled: bool,
    separate_back: bool,
    front_compare: u64,
    front_fail: u64,
    front_depth_fail: u64,
    front_depth_pass: u64,
    front_read_mask: u32,
    front_write_mask: u32,
    back_compare: u64,
    back_fail: u64,
    back_depth_fail: u64,
    back_depth_pass: u64,
    back_read_mask: u32,
    back_write_mask: u32,
}

fn consume_depth_metadata_clear(states: &mut HashMap<u64, bool>, address: Option<u64>) -> bool {
    let Some(address) = address else {
        return false;
    };
    let pending = states.entry(address).or_insert(true);
    let consumed = *pending;
    *pending = false;
    consumed
}

fn rearm_depth_metadata_clear(states: &mut HashMap<u64, bool>, address: u64) -> bool {
    let Some(pending) = states.get_mut(&address) else {
        return false;
    };
    *pending = true;
    true
}

fn consume_color_metadata_clear(
    states: &mut HashMap<u64, u32>,
    address: Option<u64>,
    slice: u32,
) -> bool {
    let Some(address) = address else {
        return false;
    };
    let Some(slice_mask) = 1_u32.checked_shl(slice) else {
        return false;
    };
    // shadPS4 registers a newly observed CMASK with all slices logically cleared. A metadata
    // compute/DMA fill can rearm that state; beginning a color attachment or eliminating the
    // fast clear consumes it once.
    let clear_mask = states.entry(address).or_insert(u32::MAX);
    let consumed = *clear_mask & slice_mask != 0;
    *clear_mask &= !slice_mask;
    consumed
}

fn rearm_color_metadata_clear(states: &mut HashMap<u64, u32>, address: u64) -> bool {
    let Some(clear_mask) = states.get_mut(&address) else {
        return false;
    };
    *clear_mask = u32::MAX;
    true
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DecodedRasterState {
    cull_mode: MTLCullMode,
    front_winding: MTLWinding,
    fill_mode: MTLTriangleFillMode,
    depth_bias: DecodedDepthBias,
}

struct DirectScanoutTexture {
    descriptor: GuestTextureDescriptor,
    texture: Texture,
    initialized: bool,
}

struct ScanoutSnapshotTexture {
    descriptor: GuestTextureDescriptor,
    texture: Texture,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MultisampleTextureKey {
    descriptor: GuestTextureDescriptor,
    sample_count: u32,
}

struct MultisampleResidentTexture {
    texture: Texture,
    initialized: bool,
}

struct CompiledShaderPair {
    pipeline: RenderPipelineState,
    vertex_function: Function,
    pixel_function: Function,
    vertex: remu_shader_metal::MetalShader,
    pixel: remu_shader_metal::MetalShader,
    reused_vertex_function: bool,
    reused_pixel_function: bool,
    lowering_duration: Duration,
    function_duration: Duration,
    pipeline_duration: Duration,
    uses_mesh_stage: bool,
    uses_procedural_rectangle_stage: bool,
}

struct CompiledComputePipeline {
    pipeline: ComputePipelineState,
    function: Function,
    shader: remu_shader_metal::MetalShader,
    lowering_duration: Duration,
    function_duration: Duration,
    pipeline_duration: Duration,
}

#[derive(Debug, Eq, Hash, PartialEq)]
struct SamplerArgumentCacheKey {
    function: usize,
    buffer_index: Option<u32>,
    descriptors: Vec<Vec<u32>>,
}

struct CachedSamplerArguments {
    buffer: Option<Buffer>,
    samplers: Vec<SamplerState>,
}

#[derive(Clone)]
struct ProgramIdentity(Arc<remu_shader::Program>);

impl std::fmt::Debug for ProgramIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProgramIdentity")
            .field("address", &self.0.address)
            .field("pointer", &Arc::as_ptr(&self.0))
            .finish()
    }
}

impl PartialEq for ProgramIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ProgramIdentity {}

impl Hash for ProgramIdentity {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ComputePipelineKey {
    program_identity: ProgramIdentity,
    lds_size_bytes: u32,
    compute_threadgroup_size: u32,
    scratch_size_bytes: u32,
    wave_lane_count: u32,
    allocated_vector_register_count: u32,
    compute_system_registers: Option<remu_shader::ComputeSystemRegisters>,
    initial_scalar_count: usize,
    global_bindings: Vec<(Vec<u32>, u64, usize, bool)>,
    image_bindings: Vec<(u32, bool, bool, u32, u32, u32, Option<u32>)>,
}

enum ShaderCompileResult {
    Graphics {
        key: GraphicsPipelineKey,
        compiled: Result<CompiledShaderPair, String>,
    },
    Compute {
        key: ComputePipelineKey,
        shader_address: u64,
        threads_per_group: [u32; 3],
        compiled: Result<CompiledComputePipeline, String>,
    },
}

enum ShaderCompileRequest {
    Graphics {
        key: GraphicsPipelineKey,
        draw: IndexedDraw,
    },
    Compute {
        key: ComputePipelineKey,
        dispatch: ComputeDispatch,
    },
}

const SHADER_COMPILE_PANIC_DESCRIPTION_CAPACITY: usize = 512;

impl ShaderCompileRequest {
    fn terminal_failure(&self, error: String) -> ShaderCompileResult {
        match self {
            Self::Graphics { key, .. } => ShaderCompileResult::Graphics {
                key: key.clone(),
                compiled: Err(error),
            },
            Self::Compute { key, dispatch } => ShaderCompileResult::Compute {
                key: key.clone(),
                shader_address: dispatch.shader_address,
                threads_per_group: dispatch.launch.threads_per_group,
                compiled: Err(error),
            },
        }
    }
}

fn shader_compile_panic_description(payload: &(dyn std::any::Any + Send)) -> String {
    let mut description = if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_owned()
    };
    if description.len() <= SHADER_COMPILE_PANIC_DESCRIPTION_CAPACITY {
        return description;
    }
    const SUFFIX: &str = "...";
    let mut end = SHADER_COMPILE_PANIC_DESCRIPTION_CAPACITY - SUFFIX.len();
    while !description.is_char_boundary(end) {
        end -= 1;
    }
    description.truncate(end);
    description.push_str(SUFFIX);
    description
}

fn compile_shader_request_catching_panics(
    request: &ShaderCompileRequest,
    compile: impl FnOnce() -> ShaderCompileResult,
) -> ShaderCompileResult {
    // Keep the catch outside the Objective-C pool. A compiler panic unwinds through
    // `autoreleasepool`, draining request-local objects before it becomes a terminal queue result.
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        objc::rc::autoreleasepool(compile)
    })) {
        Ok(result) => result,
        Err(payload) => request.terminal_failure(format!(
            "compiler worker panicked: {}",
            shader_compile_panic_description(payload.as_ref())
        )),
    }
}

#[derive(Debug, Default)]
struct RenderPipelineArchiveState {
    additions: usize,
    checkpoint_serialized: bool,
    quarantined: bool,
}

#[derive(Clone)]
struct PersistentRenderPipelineArchive {
    archive: BinaryArchive,
    path: Arc<PathBuf>,
    loaded: bool,
    state: Arc<Mutex<RenderPipelineArchiveState>>,
    driver_lock: Arc<RwLock<()>>,
}

enum ShaderFunctionCacheEntry {
    Compiling,
    Ready(Function),
    Failed(String),
}

#[derive(Default)]
struct ShaderFunctionStageCache {
    entries: Mutex<HashMap<String, ShaderFunctionCacheEntry>>,
    completed: Condvar,
}

#[derive(Clone, Debug)]
struct SharedEventHostEffectLatencyState {
    phase: SharedEventHostEffectLatencyPhase,
    phase_started_at: Instant,
}

impl SharedEventHostEffectLatencyState {
    fn new(now: Instant) -> Self {
        Self {
            phase: SharedEventHostEffectLatencyPhase::ConsumerAck,
            phase_started_at: now,
        }
    }

    fn complete_phase(
        &mut self,
        now: Instant,
        statistics: &mut SharedEventHostEffectLatencyStatistics,
    ) {
        statistics.record(
            self.phase,
            now.saturating_duration_since(self.phase_started_at),
        );
        self.phase = match self.phase {
            SharedEventHostEffectLatencyPhase::ConsumerAck => {
                SharedEventHostEffectLatencyPhase::ImmutablePrefix
            }
            SharedEventHostEffectLatencyPhase::ImmutablePrefix => {
                SharedEventHostEffectLatencyPhase::HostTerminal
            }
            SharedEventHostEffectLatencyPhase::HostTerminal => {
                SharedEventHostEffectLatencyPhase::HostTerminal
            }
        };
        self.phase_started_at = now;
    }
}

#[derive(Clone, Default)]
struct ShaderFunctionCache {
    vertex: Arc<ShaderFunctionStageCache>,
    mesh: Arc<ShaderFunctionStageCache>,
    pixel: Arc<ShaderFunctionStageCache>,
}

struct ScheduledGpuSubmission {
    ingress_sequence: u64,
    ingested_at: Option<Instant>,
    events: VecDeque<GpuEvent>,
}

struct ScheduledHostEvent {
    ingress_sequence: u64,
    ingested_at: Option<Instant>,
    event: GpuEvent,
}

#[derive(Clone, Copy, Debug, Default)]
struct SharedEventAckBatchCpuPhases {
    translation: Duration,
    metal_encoding: Duration,
    draws: u64,
    computes: u64,
}

#[derive(Debug)]
struct PendingSharedEventAckTrace {
    fence: SharedEventFence,
    event_queue: Duration,
    host_preflight: Duration,
    ack_encoding: Duration,
}

#[derive(Debug, Default)]
struct SharedEventAckNativeHostTimes {
    scheduled_at: Option<Instant>,
    completed_at: Option<Instant>,
}

#[derive(Clone, Copy, Debug)]
struct SharedEventAckTerminalTimes {
    kernel_to_gpu_start: Duration,
    gpu_execution: Duration,
    gpu_to_kernel_end: Duration,
}

#[derive(Debug)]
struct SharedEventAckTrace {
    fence: SharedEventFence,
    committed_at: Instant,
    event_queue: Duration,
    cpu: SharedEventAckBatchCpuPhases,
    ack_encoding: Duration,
    commit_call: Duration,
    native_host: Arc<Mutex<SharedEventAckNativeHostTimes>>,
    terminal: Option<SharedEventAckTerminalTimes>,
    ack_observed_at: Option<Instant>,
}

#[derive(Clone, Debug)]
enum QueueBlockReason {
    GuestMemoryWait(GpuMemoryWait),
    /// The satisfying value is already ordered behind a submitted Metal batch.
    ///
    /// Unlike a producerless wait, this queue does not need periodic CPU polling. Terminal
    /// publication wakes the exact overlapping wait after the label becomes guest-visible.
    GuestMemoryWaitPendingTerminal(GpuMemoryWait),
    /// Presenter-thread work must wait for both the consumer to cross its encoded wait and every
    /// producer's terminal host-side publication. `latency` observes the exact producer signal
    /// between those gates but does not publish memory or relax readiness.
    GuestSharedEventHostEffect {
        consumer_queue_id: u64,
        consumer_ack: Option<SharedEventFence>,
        terminal_dependencies: Vec<SharedEventFence>,
        latency: SharedEventHostEffectLatencyState,
    },
    FlipSafetyWait(GpuFlipWait),
    GraphicsPipeline(GraphicsPipelineKey),
    ComputePipeline(ComputePipelineKey),
}

enum PipelineCompileCandidate {
    Graphics(IndexedDraw),
    Compute(ComputeDispatch),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum OrderedEventSource {
    Host { ingress_sequence: u64 },
    GuestQueue(u64),
}

struct OrderedGpuEventReceiver {
    receiver: Receiver<GpuEvent>,
    submissions: HashMap<u64, VecDeque<ScheduledGpuSubmission>>,
    queue_order: Vec<u64>,
    next_queue_index: usize,
    next_prefetch_source_index: usize,
    active_queue: Option<u64>,
    blocked_queues: HashMap<u64, QueueBlockReason>,
    host_events: VecDeque<ScheduledHostEvent>,
    host_block: Option<QueueBlockReason>,
    next_ingress_sequence: u64,
    disconnected: bool,
    event_queue_timing: bool,
    last_event_ingested_at: Option<Instant>,
    last_event_queue_wait: Duration,
    /// Count of queued small producer writes per queue and 4-byte label cell.
    ///
    /// The dependency-ingress predicate must know whether an ingested queue can still satisfy
    /// a blocked memory wait. Scanning every queued event per check made the ingest control
    /// path O(queue²); this index answers it in O(1). Producers wider than the small-label
    /// cap are deliberately not counted: an undercount only widens ingress, which is the safe
    /// direction, while an overcount could starve a waiting queue of its producer.
    producer_labels: HashMap<(u64, u64), u32>,
}

#[derive(Debug)]
enum OrderedEventPoll {
    Event {
        source: OrderedEventSource,
        event: GpuEvent,
    },
    Blocked,
    Empty,
    Disconnected,
}

struct UploadedImageTexture {
    texture: Texture,
    descriptor: DecodedImageDescriptor,
    writable: bool,
    /// The key the residency map filed the served texture under, when it came from a resident.
    ///
    /// A storage image can be served by a resident whose pitch differs from the binding's, so
    /// attributing a later write to `descriptor.guest` would find no resident at all.
    selected_resident: Option<GuestTextureDescriptor>,
}

struct ResidentGuestTexture {
    texture: Texture,
    component_reorder_view: Option<Texture>,
    descriptor: GuestTextureDescriptor,
    texture_type: u32,
    layer_count: u32,
    mip_level_count: u32,
    write_watch: GuestTextureWriteWatch,
    /// The guest bytes this allocation declares, resolved once at creation.
    ///
    /// `None` disables the range algebra for this resident and leaves it on the descriptor-overlap
    /// behaviour. Note this is the resident's *whole* range — mip chain, array faces and all —
    /// while the interval index keys residents by their level-0 span, so the index answers "which
    /// residents might be involved" and `owned` answers "which bytes are actually theirs".
    span: Option<ResourceSpan>,
    /// Guest bytes this allocation is the authority for.
    owned: RangeSet,
    /// Subset of `owned` whose current value exists only in Metal.
    ///
    /// This is the publication debt the guest is owed, and it is what `gpu_written()` reports.
    /// Bookkeeping settles must not clear it without publishing — that lies that guest RAM agrees.
    gpu_authored: RangeSet,
    /// `Live` unless every owned byte was taken by another writer or by the CPU.
    state: ResidencyState,
    /// ACQUIRE_MEM proved guest bytes diverged from the Metal copy, but no consumer has bound
    /// this resident yet. Demotion is deferred until reclaim/bind (Phase 1H consumer-aware).
    acquire_invalidated: bool,
    content_version: u64,
    /// CPU-write generations observed when Metal most recently became authoritative.
    ///
    /// Guest heaps reuse image allocations as ICBs. A plain dirty bit cannot distinguish a later
    /// guest write from presenter-side watch renewal; generations establish the ownership boundary
    /// at the actual GPU write (kept after fingerprint deletion for the Active false-positive path).
    gpu_write_watch_generations: Vec<u64>,
    /// Most recent ordered guest work/flip that read or wrote this Metal allocation.
    ///
    /// This stamp is always maintained because pressure eviction must distinguish completed
    /// render-graph temporaries from resources still participating in the current frame. The
    /// more detailed per-operation activity counters remain opt-in diagnostics below.
    last_use: ResidentTextureActivityStamp,
    /// Read/write chronology used only by opt-in texture-lifetime diagnostics.
    ///
    /// Ordinary runs retain no counters and take only the predictable `Option::None` branch at
    /// resident sample/write sites.
    activity: Option<ResidentTextureActivity>,
}

/// Whether an allocation still answers for any of its guest bytes, and where they went if not.
///
/// The field has no `Default` on purpose: every construction site must name it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidencyState {
    Live,
    Stale(StaleSource),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StaleSource {
    /// Another resident wrote every byte, and its texels are recoverable on the GPU.
    Allocation {
        descriptor: GuestTextureDescriptor,
        /// The superseder's `content_version` at demotion time. A superseder that is gone or has
        /// moved on cannot be trusted as an identity, and recovery falls back to `GuestMemory`.
        content_version: u64,
    },
    /// Guest RAM became authoritative: a validated CPU write, an ACQUIRE_MEM invalidation, a DMA
    /// destination write, or REmu's own publish. Recovery is a rebuild from RAM, which is free
    /// whenever the watch proves the bytes are unchanged.
    GuestMemory,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ResidentTextureActivity {
    last_read_work: Option<u64>,
    last_read_flip: Option<u64>,
    last_write_work: Option<u64>,
    last_write_flip: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ResidentTextureActivityStamp {
    work: u64,
    flip: u64,
}

impl ResidentTextureActivity {
    fn enabled() -> Option<Self> {
        resident_texture_lifetime_diagnostics_enabled().then(Self::default)
    }

    fn note_read(&mut self, stamp: ResidentTextureActivityStamp) {
        RESIDENT_TEXTURE_ACTIVITY_LATEST_WORK.fetch_max(stamp.work, Ordering::Relaxed);
        self.last_read_work = Some(stamp.work);
        self.last_read_flip = Some(stamp.flip);
    }

    fn note_write(&mut self, stamp: ResidentTextureActivityStamp) {
        RESIDENT_TEXTURE_ACTIVITY_LATEST_WORK.fetch_max(stamp.work, Ordering::Relaxed);
        self.last_write_work = Some(stamp.work);
        self.last_write_flip = Some(stamp.flip);
    }
}

impl ResidentGuestTexture {
    /// A watch's span together with authority over every byte of it.
    ///
    /// A newly created allocation owns everything it declares: it either uploaded guest RAM or is
    /// about to be fully overwritten, and either way no other resident may answer for those bytes.
    fn declared_authority(
        write_watch: &GuestTextureWriteWatch,
    ) -> (Option<ResourceSpan>, RangeSet) {
        let span = resident_declared_span(write_watch);
        if span.is_none_or(|span| span.backing().is_none()) {
            OWNED_UNRESOLVED_SPAN.fetch_add(1, Ordering::Relaxed);
        }
        let owned = span.as_ref().map(RangeSet::for_span).unwrap_or_default();
        (span, owned)
    }

    /// True while any guest byte's current value exists only in Metal.
    fn gpu_written(&self) -> bool {
        debug_assert!(
            self.is_live() || self.gpu_authored.is_empty(),
            "a demoted allocation answers for no byte and can owe none"
        );
        !self.gpu_authored.is_empty()
    }

    fn is_live(&self) -> bool {
        matches!(self.state, ResidencyState::Live)
    }

    /// True when this allocation can serve a read of its whole declared range.
    ///
    /// Coverage must be total: a partially superseded tenant holds a mixture of current and
    /// superseded texels, and no read path can bind part of a Metal texture.
    fn serves_reads(&self) -> bool {
        self.is_live() && self.span.is_none_or(|span| self.owned.covers(&span))
    }

    /// Gives up `taken`, demoting to `source` when the last owned byte goes with it.
    fn subtract_authority(&mut self, taken: &RangeSet, source: StaleSource) {
        self.owned.subtract(taken);
        // A byte another allocation answers for can never be published from here.
        self.gpu_authored.subtract(taken);
        if self.owned.is_empty() {
            self.demote(source);
        }
    }

    /// Stops answering for any byte, in place: the Metal allocation, the write watch, its
    /// `mprotect` state and its fingerprint baseline all survive.
    fn demote(&mut self, source: StaleSource) {
        debug_assert!(
            self.write_watch.retirement() != Some(CpuWriteWatchRetirement::Replaced),
            "a replaced mapping proves nothing about its bytes and must be removed, not demoted"
        );
        self.owned = RangeSet::default();
        self.gpu_authored = RangeSet::default();
        self.acquire_invalidated = false;
        self.state = ResidencyState::Stale(source);
        OWNED_DEMOTIONS.fetch_add(1, Ordering::Relaxed);
    }

    /// Prepares a possibly demoted allocation for a bind, reporting whether it may serve it.
    ///
    /// A discarded target has every texel overwritten before one is read, so it needs no bytes —
    /// only the shape, the live watch and its fingerprints. `owned` stays empty until the mark
    /// that follows, which is the one window in which a `Live` resident owns nothing.
    fn reclaim_for_bind(&mut self, discard_guest_contents: bool) -> bool {
        // Consumer-aware ACQUIRE demotion: invalidate at first consumer bind, not at acquire.
        if self.acquire_invalidated && !discard_guest_contents {
            self.demote(StaleSource::GuestMemory);
            return false;
        }
        if discard_guest_contents {
            self.acquire_invalidated = false;
        }
        if self.serves_reads() {
            return true;
        }
        if !discard_guest_contents {
            return false;
        }
        self.state = ResidencyState::Live;
        OWNED_RECLAIM_DISCARD.fetch_add(1, Ordering::Relaxed);
        true
    }

    #[cfg(test)]
    fn note_gpu_authored_whole_span(&mut self) {
        let written = self
            .span
            .as_ref()
            .map(RangeSet::for_span)
            .unwrap_or_default();
        self.note_gpu_authored(&written);
    }

    /// Records a GPU write of `written` bytes: authority is claimed and the debt grows with it.
    fn note_gpu_authored(&mut self, written: &RangeSet) {
        self.owned.insert(written);
        self.gpu_authored.insert(written);
        self.state = ResidencyState::Live;
        debug_assert!(
            self.owned.contains_all(&self.gpu_authored),
            "a resident may only owe bytes it is the authority for"
        );
        debug_assert!(
            self.span
                .is_none_or(|span| RangeSet::for_span(&span).contains_all(&self.owned)),
            "a resident may only claim bytes inside its own declared span"
        );
    }

    /// The publication debt as byte offsets from this resident's declared base address.
    ///
    /// `None` means the debt cannot be expressed in that space and the caller must publish the
    /// whole declared range, which is what every writeback did before the algebra existed.
    fn gpu_authored_offsets(&self) -> Option<Vec<(u64, u64)>> {
        let base = self.span?.range_start();
        Some(
            self.gpu_authored
                .runs()
                .iter()
                .filter_map(|&(start, end)| {
                    Some((start.checked_sub(base)?, end.checked_sub(base)?))
                })
                .collect(),
        )
    }

    /// Drops the publication debt after guest RAM was made to agree with Metal.
    ///
    /// Authority is unchanged: the allocation still defines what these bytes mean, RAM merely
    /// holds the same value now.
    fn clear_gpu_authored(&mut self) {
        self.gpu_authored = RangeSet::default();
    }

    fn note_gpu_write(&mut self) {
        self.gpu_write_watch_generations = self.write_watch.generations();
    }

    fn cpu_write_obsoletes_gpu_writeback(
        &mut self,
        refresh: WriteWatchRefresh,
    ) -> Result<bool, String> {
        // Mapping renewal can report dirty without a guest store — remaps must not steal Metal
        // authority (`5b920ab`). On Active watches, generations still filter false-positive dirt
        // after fingerprint deletion: unchanged content re-anchors the GPU ownership boundary.
        match refresh {
            WriteWatchRefresh::Replaced => Ok(true),
            WriteWatchRefresh::Reprotecting | WriteWatchRefresh::Renewed { .. } => Ok(false),
            WriteWatchRefresh::Active => {
                let possibly_changed = self.write_watch.content_validation_pending()
                    || self.write_watch.is_dirty()
                    || self.write_watch.generations() != self.gpu_write_watch_generations;
                if !possibly_changed {
                    return Ok(false);
                }
                if self.write_watch.content_has_changed()? {
                    return Ok(true);
                }
                self.note_gpu_write();
                Ok(false)
            }
        }
    }

    fn note_read(&mut self, stamp: ResidentTextureActivityStamp) {
        self.last_use = stamp;
        if let Some(activity) = self.activity.as_mut() {
            activity.note_read(stamp);
        }
    }

    fn note_write(&mut self, stamp: ResidentTextureActivityStamp) {
        self.last_use = stamp;
        if let Some(activity) = self.activity.as_mut() {
            activity.note_write(stamp);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SampledTextureKey {
    descriptor: GuestTextureDescriptor,
    base_mip_level: u32,
    mip_level_count: u32,
    texture_type: u32,
    layer_count: u32,
    comparison: bool,
}

/// One layered depth-sample SRD whose faces are assembled on the GPU from resident depth targets.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct DepthArrayKey {
    descriptor: GuestTextureDescriptor,
    layer_count: u32,
}

/// The assembled depth-format array texture and the face content versions its slices hold.
struct DepthArrayResidentTexture {
    texture: Texture,
    slice_versions: Vec<u64>,
}

struct CachedSampledTexture {
    texture: Texture,
    write_watch: Arc<GuestTextureWriteWatch>,
    overlap: Option<ResourceSpan>,
    /// Actual Metal allocation charged to the sampled-image cache budget.
    ///
    /// Guest bytes can be block-compressed while the host texture is expanded, and generated mip
    /// levels also consume host storage that is absent from the watched guest range. Charging the
    /// write-watch span therefore does not bound unified-memory use.
    resident_bytes: u64,
    last_use: u64,
}

struct GuestTextureWriteWatch {
    address: u64,
    byte_count: u64,
    ranges: Vec<GuestTextureWatchRange>,
    content_state: Arc<Mutex<GuestTextureContentState>>,
}

struct GuestTextureWatchRange {
    watch: CpuWriteWatch,
    /// Address in the canonical GPU-physical resource corresponding to `watch.range().0`.
    ///
    /// One physical allocation can have several disjoint CPU aliases. Dirty faults arrive in
    /// the alias address space, while snapshots and fingerprints use the resource's canonical
    /// address. Retaining this translation keeps dirty-page validation coherent across aliases.
    canonical_address: u64,
}

struct GuestTextureContentState {
    pending_dirty_ranges: Vec<(u64, u64)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WriteWatchRefresh {
    Active,
    Reprotecting,
    Renewed { was_dirty: bool },
    Replaced,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GlobalBufferCacheKey {
    base_address: u64,
    byte_count: usize,
}

struct CachedGlobalBuffer {
    bytes: Vec<u8>,
    buffer: Buffer,
    offset: u64,
}

/// Metal submission state owned by one logical guest GPU queue.
///
/// The B17 per-queue path is the default (plan A1). It deliberately stays separate from the
/// legacy shared queue fields below so the `REMU_PER_QUEUE_BATCHES=0` kill-switch preserves the
/// old object graph and submission behaviour exactly for one cycle, after which the legacy path
/// is deleted.
struct GuestQueueBatchState {
    command_queue: metal::CommandQueue,
    shared_event_timeline: SharedEventTimeline,
    host_retired_fence_value: u64,
    host_dependencies: HashMap<SharedEventProducerId, u64>,
    resource_dependencies: HashMap<SharedEventProducerId, SharedEventSignalCheckpoint>,
    open_batch: Option<GuestCommandBatch>,
    committed_batches: VecDeque<GuestCommandBatch>,
    transient_global_buffer_cache: HashMap<GlobalBufferCacheKey, CachedGlobalBuffer>,
}

#[derive(Clone)]
struct UploadedGlobalBuffer {
    buffer: Buffer,
    offset: u64,
    guest_memory: bool,
    physical_identity: Option<GuestSharedMappingIdentity>,
    original_version: Option<GlobalBufferOriginalVersion>,
}

#[derive(Clone)]
struct GlobalBufferOriginalVersion {
    buffer: Buffer,
    offset: u64,
    /// Keeps a no-copy physical backing view alive until the GPU baseline copy is terminal.
    /// Ordinary retained GPU versions do not need this extra lease.
    _retained_source: Option<Buffer>,
}

struct PendingGlobalWriteback {
    buffer: Buffer,
    offset: u64,
    guest_address: u64,
    byte_count: usize,
    /// Capture-time claim on the destination (plan D2): identity resolved, shared-mapping
    /// epoch recorded, and the backing object pinned when the range was captured — not
    /// re-derived at publish time, seconds later on a deep queue.
    lease: Option<remu_memory::GuestRangeLease>,
    write_watch: Option<CpuWriteWatch>,
    original_bytes: Vec<u8>,
    original_version: Option<GlobalBufferOriginalVersion>,
}

impl PendingGlobalWriteback {
    fn mapping_identity(&self) -> Option<GuestSharedMappingIdentity> {
        self.lease
            .as_ref()
            .map(remu_memory::GuestRangeLease::identity)
    }
}

struct PendingPresentation {
    command_buffer: CommandBuffer,
    submitted_at: Instant,
    flip_sequence: Option<u64>,
    flip_terminal_source: FlipTerminalSource,
    /// Windowed flips become guest-visible when Core Animation presents the drawable, not when
    /// Metal merely finishes rendering it. Surface-less retirement tails have no drawable and
    /// therefore leave this unset.
    drawable_presented: Option<Arc<AtomicBool>>,
    await_presented_reported: bool,
}

/// Whether a flush may decline to commit the open guest batch.
///
/// `Opportunistic` treats `MAX_COMMITTED_GUEST_BATCHES` as backpressure and leaves the batch open
/// when the budget is exhausted. `Required` is for callers that then read GPU results the open
/// batch produces — a readback, a scanout snapshot, a CPU drain — so it must leave
/// `open_guest_batch` empty, settling the oldest committed batch to reclaim a slot when needed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FlushPolicy {
    Opportunistic,
    Required,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GuestBatchFlushReason {
    EventLoop,
    BlockedWait,
    SharedEventHostEffect,
    FlipSnapshot,
    WorkCapacity,
    CommandProcessorBarrier,
    CpuSynchronization,
}

struct PresenterStatistics {
    interval_started: Instant,
    last_logged: Instant,
    last_flip_reconciliation: Instant,
    guest_flips: u32,
    completed_presents: u32,
    submitted_guest_batches: u32,
    coalesced_pipeline_barriers: u32,
    encoded_draws: u32,
    encoded_compute_dispatches: u32,
    encoded_render_passes: u32,
    render_pass_ends_for_compute: u32,
    render_pass_ends_for_target_change: u32,
    render_pass_ends_for_color_operation: u32,
    render_pass_ends_for_batch_flush: u32,
    render_pass_key_transitions: GuestRenderPassKeyTransitionStatistics,
    batch_flushes_from_event_loop: u32,
    batch_flushes_for_guest_memory_write: u32,
    batch_flushes_for_blocked_wait: u32,
    batch_flushes_for_shared_event_host_effect: u32,
    batch_flushes_for_flip_snapshot: u32,
    batch_flushes_for_work_capacity: u32,
    batch_flushes_for_command_processor_barrier: u32,
    batch_flushes_for_cpu_synchronization: u32,
    shared_event_signals: u32,
    shared_event_waits: u32,
    /// A3: same-queue `WAIT_REG_MEM`s the producer index proved were encoded ahead on this very
    /// queue, so Metal FIFO order already satisfies them and no GPU event wait is encoded.
    same_queue_wait_elisions: u32,
    render_backend_flushes: u32,
    shared_event_host_blocks: u32,
    shared_event_host_block_kinds: [u32; SHARED_HOST_KIND_COUNT],
    shared_event_host_wakes: u32,
    shared_event_acquire_noop_drops: u32,
    shared_event_dma_fill_attempts: u32,
    shared_event_gpu_ordered_dma_fills: u32,
    shared_event_dma_fill_rejections: [u32; SAME_QUEUE_DMA_FILL_REJECT_COUNT],
    shared_event_gpu_ordered_registered_waits: u32,
    shared_event_gpu_ordered_compute_dispatches: u32,
    shared_event_gpu_ordered_draws: u32,
    shared_event_draw_attempts: u32,
    shared_event_draw_disjoint_candidates: u32,
    shared_event_draw_versioned_global_candidates: u32,
    shared_event_draw_resident_image_candidates: u32,
    shared_event_draw_internal_drains: u32,
    shared_event_draw_rejections: [u32; SHARED_EVENT_DRAW_REJECT_COUNT],
    shared_event_incomplete_signal_footprints: u32,
    shared_event_error_advances: u32,
    shared_event_fallbacks_no_exact_producer: u32,
    shared_event_fallbacks_timeline_overflow: u32,
    shared_event_host_effect_latency: SharedEventHostEffectLatencyStatistics,
    shared_event_ack_phases: SharedEventAckPhaseStatistics,
    terminal_publication: TerminalPublicationStatistics,
    gpu_execution_time: Duration,
    presentation_queue_latency: Duration,
    last_audio_output_frames: u64,
    last_audio_underrun_frames: u64,
}

#[derive(Debug)]
enum GuestBatchWork {
    Draw {
        sequence: u64,
        indexed: bool,
        indirect_arguments_address: Option<u64>,
        export: u64,
        pixel: u64,
        index_count: u32,
        instance_count: u32,
        target: u64,
        extent: [u32; 2],
        export_msl_bytes: usize,
        pixel_msl_bytes: usize,
        export_dispatcher: bool,
        pixel_dispatcher: bool,
    },
    Compute {
        sequence: u64,
        shader: u64,
        source: remu_gpu::ComputeDispatchSource,
        grid: remu_gpu::ComputeGrid,
        threads_per_group: [u32; 3],
        msl_bytes: usize,
        dispatcher: bool,
    },
}

struct GuestCommandBatch {
    queue_id: Option<u64>,
    command_buffer: CommandBuffer,
    work: Vec<GuestBatchWork>,
    committed_at: Option<Instant>,
    global_upload_chunks: Vec<GlobalUploadChunk>,
    /// Exact copied-buffer versions written by native work in this batch, including GPU-only
    /// outputs which do not require guest publication at terminal completion.
    global_buffer_versions: Vec<SharedEventGlobalBufferVersion>,
    global_writebacks: Vec<PendingGlobalWriteback>,
    guest_write_ranges: Vec<PendingGuestWriteRange>,
    memory_writes: Vec<PendingGuestMemoryWrite>,
    /// Immutable physical destinations written by native work encoded in this batch.
    ///
    /// CPU-published globals and command-processor writes remain in their existing vectors and
    /// are folded into a signal checkpoint at capture time. This journal covers resident/cache
    /// destinations whose Metal commands can still be in flight when a later queue waits on the
    /// signal.
    resident_write_footprint: Vec<PhysicalFootprintRange>,
    /// Exact Metal texture versions produced by resident writes in this batch.
    ///
    /// Shared-event checkpoints retain these objects so a same-queue consumer can sample the
    /// producer version directly without observing mutable guest RAM or a later residency-map
    /// tenant. Only ordinary render/storage destinations enter this journal; internal
    /// rematerialization and scanout work remain host dependencies.
    resident_texture_versions: Vec<SharedEventResidentTextureVersion>,
    /// False once native work was encoded for a destination without a stable physical identity.
    /// Unknown identity must remain a host dependency; virtual non-overlap is not a proof.
    resident_write_footprint_complete: bool,
    render_pass: Option<OpenGuestRenderPass>,
    compute_encoder: Option<ComputeCommandEncoder>,
    /// Highest SharedEvent signal encoded at an exact command-stream point in this batch.
    max_encoded_signal: Option<SharedEventFence>,
    signal_checkpoints: Vec<SharedEventSignalCheckpoint>,
    /// Guest submissions whose final ordered work is contained in this Metal batch.
    ///
    /// Queue scheduling can complete a newer global sequence before an older blocked queue, so
    /// every boundary is retained rather than incorrectly treating the maximum as a prefix.
    completion_sequences: Vec<u64>,
}

#[derive(Clone, Debug)]
struct SharedEventSignalCheckpoint {
    fence: SharedEventFence,
    global_writeback_count: usize,
    guest_write_range_count: usize,
    memory_write_count: usize,
    /// Immutable copied-buffer versions produced before `fence` was encoded.
    ///
    /// A checkpoint owns Metal references rather than vector indices alone. Later producer work
    /// can append or replace bookkeeping entries without changing the exact bytes visible at the
    /// signal. Zero-copy guest buffers and CPU-published writes are deliberately excluded: their
    /// storage remains mutable and therefore cannot cross a queue on a GPU wait alone.
    global_versions: Vec<SharedEventGlobalBufferVersion>,
    /// Exact resident texture objects visible at this queue signal.
    resident_texture_versions: Vec<SharedEventResidentTextureVersion>,
    unresolved_write_ranges: Vec<SharedEventUnresolvedWriteRange>,
    /// Complete physical write footprint of every unretired producer batch ordered before this
    /// exact queue signal, including the signal-owning RELEASE_MEM label.
    pending_write_footprint: Vec<PhysicalFootprintRange>,
    footprint_complete: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SharedEventWriteKind {
    ResidentDestination,
    DirectScanoutTarget,
    CopiedGlobal,
    ZeroCopyGlobal,
    TerminalGuestWrite,
    InternalResidentDestination,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PhysicalFootprintRange {
    kind: SharedEventWriteKind,
    guest_address: u64,
    byte_count: u64,
    mapping: GuestSharedMappingIdentity,
}

#[derive(Clone, Debug)]
struct SharedEventGlobalBufferVersion {
    buffer: Buffer,
    offset: u64,
    guest_address: u64,
    byte_count: u64,
    mapping_identity: Option<GuestSharedMappingIdentity>,
}

#[derive(Clone, Debug)]
struct SharedEventResidentTextureVersion {
    texture: Texture,
    descriptor: GuestTextureDescriptor,
    texture_type: u32,
    layer_count: u32,
    mip_level_count: u32,
    content_version: u64,
    span: ResourceSpan,
    /// Which write recorded this version: a guest render-target write
    /// (`ResidentDestination`) or an internal relayout (`InternalResidentDestination`,
    /// plan A2). A version only licenses producer ranges of its own kind, and internal
    /// versions do not require the resident to be GPU-authored — an internal relayout of
    /// clean CPU content leaves guest RAM authoritative.
    kind: SharedEventWriteKind,
}

#[derive(Clone, Debug)]
struct SharedEventDrawTextureOverride {
    sampled: DecodedImageDescriptor,
    texture: Texture,
}

impl SharedEventDrawTextureOverride {
    fn matches(&self, sampled: DecodedImageDescriptor) -> bool {
        self.sampled.guest == sampled.guest
            && self.sampled.destination_select == sampled.destination_select
            && self.sampled.base_mip_level == sampled.base_mip_level
            && self.sampled.mip_level_count == sampled.mip_level_count
            && self.sampled.texture_type == sampled.texture_type
            && self.sampled.layer_count == sampled.layer_count
    }
}

#[derive(Clone, Debug)]
struct SharedEventDirectDrawPlan {
    texture_overrides: Vec<SharedEventDrawTextureOverride>,
}

#[derive(Clone, Debug)]
struct SharedEventComputeGlobalBindingPlan {
    source: Option<Buffer>,
    source_offset: u64,
    guest_backing: Option<GuestSharedBackingRange>,
    physical_identity: GuestSharedMappingIdentity,
    byte_count: u64,
    writable_group: Option<usize>,
}

#[derive(Clone, Debug)]
struct SharedEventDirectComputePlan {
    global_bindings: Vec<SharedEventComputeGlobalBindingPlan>,
}

#[derive(Clone, Copy, Debug)]
struct SharedEventUnresolvedWriteRange {
    guest_address: u64,
    byte_count: u64,
    mapping_identity: Option<GuestSharedMappingIdentity>,
}

struct GlobalUploadChunk {
    buffer: Buffer,
    used: usize,
}

struct PendingGuestWriteRange {
    address: u64,
    byte_count: u64,
    buffer: Buffer,
    offset: u64,
    physical_identity: GuestSharedMappingIdentity,
}

#[derive(Clone, Debug)]
enum GuestMemoryWriteOperation {
    Release(GpuMemoryRelease),
    Write(GpuMemoryWrite),
    Fill {
        address: u64,
        value: u32,
        dword_count: u64,
    },
    /// A DMA_DATA copy's destination bytes, resolved at decode time and applied in order.
    ///
    /// Unreal writes its GPU fences through DMA_DATA as well as RELEASE_MEM (the runtime traces
    /// them as `fence_label_dma`). Applying the copy at event-dequeue time made those fences
    /// visible ahead of the frame's in-flight Metal work, and the guest recycled frame memory
    /// its own threads were still writing.
    Copy {
        address: u64,
        bytes: Vec<u8>,
    },
}

#[derive(Debug)]
struct PendingGuestMemoryWrite {
    operation: GuestMemoryWriteOperation,
    mapping_identity: Option<GuestSharedMappingIdentity>,
    write_watch: Option<CpuWriteWatch>,
    /// The command-processor write makes guest RAM authoritative for this destination.
    /// Apply the cache transition only after the identity-pinned write actually publishes.
    publishes_guest_memory_authority: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GuestMemoryWaitStatus {
    Satisfied,
    Blocked,
    BlockedOnPendingTerminal,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GuestMemoryWaitKey {
    source: OrderedEventSource,
    address: u64,
    reference: u64,
    mask: u64,
    compare_function: u32,
    is_64_bit: bool,
    operation: u32,
    cache_policy: u32,
}

#[derive(Clone, Copy, Debug)]
struct ActiveGuestMemoryWait {
    latched_value: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GuestMemoryWaitRecovery {
    None,
    ProducerLatched(u64),
}

#[derive(Default)]
struct GuestMemoryWaitTracker {
    active: HashMap<GuestMemoryWaitKey, ActiveGuestMemoryWait>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GuestRenderPassKey {
    targets: Vec<(GuestTextureDescriptor, u32, bool, u32, u32, u32, u32, u32)>,
    depth: Option<(GuestTextureDescriptor, u32, u32, Option<u32>, Option<u64>)>,
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
enum GuestRenderPassKeyField {
    TargetCount,
    TargetAddress,
    TargetWidth,
    TargetHeight,
    TargetPitch,
    TargetTileMode,
    TargetPixelFormat,
    TargetBytesPerPixel,
    TargetSampleCount,
    TargetDirectScanout,
    TargetTextureType,
    TargetLayerCount,
    TargetBaseLayer,
    TargetMipLevel,
    TargetMipLevelCount,
    DepthPresence,
    DepthAddress,
    DepthWidth,
    DepthHeight,
    DepthPitch,
    DepthTileMode,
    DepthPixelFormat,
    DepthBytesPerPixel,
    DepthSampleCount,
    DepthLayerCount,
    DepthClearValue,
    DepthClearSequence,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct GuestRenderPassKeyTransition(u32);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct GuestRenderPassKeyTransitionStatistics {
    target_count: u32,
    target_address: u32,
    target_width: u32,
    target_height: u32,
    target_pitch: u32,
    target_tile_mode: u32,
    target_pixel_format: u32,
    target_bytes_per_pixel: u32,
    target_sample_count: u32,
    target_direct_scanout: u32,
    target_texture_type: u32,
    target_layer_count: u32,
    target_base_layer: u32,
    target_mip_level: u32,
    target_mip_level_count: u32,
    depth_presence: u32,
    depth_address: u32,
    depth_width: u32,
    depth_height: u32,
    depth_pitch: u32,
    depth_tile_mode: u32,
    depth_pixel_format: u32,
    depth_bytes_per_pixel: u32,
    depth_sample_count: u32,
    depth_layer_count: u32,
    depth_clear_value: u32,
    depth_clear_sequence: u32,
}

impl GuestRenderPassKey {
    fn transition_to(&self, next: &Self) -> GuestRenderPassKeyTransition {
        let mut transition = GuestRenderPassKeyTransition::default();
        transition.mark(
            GuestRenderPassKeyField::TargetCount,
            self.targets.len() != next.targets.len(),
        );
        transition.mark(
            GuestRenderPassKeyField::DepthPresence,
            self.depth.is_some() != next.depth.is_some(),
        );
        for (current, next) in self.targets.iter().zip(&next.targets) {
            let current_descriptor = &current.0;
            let next_descriptor = &next.0;
            if current_descriptor != next_descriptor {
                transition.mark(
                    GuestRenderPassKeyField::TargetAddress,
                    current_descriptor.address != next_descriptor.address,
                );
                transition.mark(
                    GuestRenderPassKeyField::TargetWidth,
                    current_descriptor.width != next_descriptor.width,
                );
                transition.mark(
                    GuestRenderPassKeyField::TargetHeight,
                    current_descriptor.height != next_descriptor.height,
                );
                transition.mark(
                    GuestRenderPassKeyField::TargetPitch,
                    current_descriptor.pitch != next_descriptor.pitch,
                );
                transition.mark(
                    GuestRenderPassKeyField::TargetTileMode,
                    current_descriptor.tile_mode != next_descriptor.tile_mode,
                );
                transition.mark(
                    GuestRenderPassKeyField::TargetPixelFormat,
                    current_descriptor.pixel_format != next_descriptor.pixel_format,
                );
                transition.mark(
                    GuestRenderPassKeyField::TargetBytesPerPixel,
                    current_descriptor.bytes_per_pixel != next_descriptor.bytes_per_pixel,
                );
            }
            transition.mark(
                GuestRenderPassKeyField::TargetSampleCount,
                current.1 != next.1,
            );
            transition.mark(
                GuestRenderPassKeyField::TargetDirectScanout,
                current.2 != next.2,
            );
            transition.mark(
                GuestRenderPassKeyField::TargetTextureType,
                current.3 != next.3,
            );
            transition.mark(
                GuestRenderPassKeyField::TargetLayerCount,
                current.4 != next.4,
            );
            transition.mark(
                GuestRenderPassKeyField::TargetBaseLayer,
                current.5 != next.5,
            );
            transition.mark(GuestRenderPassKeyField::TargetMipLevel, current.6 != next.6);
            transition.mark(
                GuestRenderPassKeyField::TargetMipLevelCount,
                current.7 != next.7,
            );
        }
        if let (Some(current), Some(next)) = (&self.depth, &next.depth) {
            let current_descriptor = &current.0;
            let next_descriptor = &next.0;
            if current_descriptor != next_descriptor {
                transition.mark(
                    GuestRenderPassKeyField::DepthAddress,
                    current_descriptor.address != next_descriptor.address,
                );
                transition.mark(
                    GuestRenderPassKeyField::DepthWidth,
                    current_descriptor.width != next_descriptor.width,
                );
                transition.mark(
                    GuestRenderPassKeyField::DepthHeight,
                    current_descriptor.height != next_descriptor.height,
                );
                transition.mark(
                    GuestRenderPassKeyField::DepthPitch,
                    current_descriptor.pitch != next_descriptor.pitch,
                );
                transition.mark(
                    GuestRenderPassKeyField::DepthTileMode,
                    current_descriptor.tile_mode != next_descriptor.tile_mode,
                );
                transition.mark(
                    GuestRenderPassKeyField::DepthPixelFormat,
                    current_descriptor.pixel_format != next_descriptor.pixel_format,
                );
                transition.mark(
                    GuestRenderPassKeyField::DepthBytesPerPixel,
                    current_descriptor.bytes_per_pixel != next_descriptor.bytes_per_pixel,
                );
            }
            transition.mark(
                GuestRenderPassKeyField::DepthSampleCount,
                current.1 != next.1,
            );
            transition.mark(
                GuestRenderPassKeyField::DepthLayerCount,
                current.2 != next.2,
            );
            transition.mark(
                GuestRenderPassKeyField::DepthClearValue,
                current.3 != next.3,
            );
            transition.mark(
                GuestRenderPassKeyField::DepthClearSequence,
                current.4 != next.4,
            );
        }
        transition
    }
}

impl GuestRenderPassKeyTransition {
    fn mark(&mut self, field: GuestRenderPassKeyField, changed: bool) {
        if changed {
            self.0 |= 1_u32 << u32::from(field as u8);
        }
    }

    fn contains(self, field: GuestRenderPassKeyField) -> bool {
        self.0 & (1_u32 << u32::from(field as u8)) != 0
    }
}

impl GuestRenderPassKeyTransitionStatistics {
    fn record(&mut self, transition: GuestRenderPassKeyTransition) {
        macro_rules! record_fields {
            ($($field:ident => $cause:ident),+ $(,)?) => {
                $(
                    self.$field = self.$field.saturating_add(u32::from(
                        transition.contains(GuestRenderPassKeyField::$cause),
                    ));
                )+
            };
        }
        record_fields!(
            target_count => TargetCount,
            target_address => TargetAddress,
            target_width => TargetWidth,
            target_height => TargetHeight,
            target_pitch => TargetPitch,
            target_tile_mode => TargetTileMode,
            target_pixel_format => TargetPixelFormat,
            target_bytes_per_pixel => TargetBytesPerPixel,
            target_sample_count => TargetSampleCount,
            target_direct_scanout => TargetDirectScanout,
            target_texture_type => TargetTextureType,
            target_layer_count => TargetLayerCount,
            target_base_layer => TargetBaseLayer,
            target_mip_level => TargetMipLevel,
            target_mip_level_count => TargetMipLevelCount,
            depth_presence => DepthPresence,
            depth_address => DepthAddress,
            depth_width => DepthWidth,
            depth_height => DepthHeight,
            depth_pitch => DepthPitch,
            depth_tile_mode => DepthTileMode,
            depth_pixel_format => DepthPixelFormat,
            depth_bytes_per_pixel => DepthBytesPerPixel,
            depth_sample_count => DepthSampleCount,
            depth_layer_count => DepthLayerCount,
            depth_clear_value => DepthClearValue,
            depth_clear_sequence => DepthClearSequence,
        );
    }
}

struct OpenGuestRenderPass {
    key: GuestRenderPassKey,
    encoder: RenderCommandEncoder,
}

#[derive(Clone)]
struct CachedNggMeshlets {
    buffer: Buffer,
    group_count: u64,
    vertex_references: usize,
    primitive_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DecodedImageDescriptor {
    guest: GuestTextureDescriptor,
    span: Option<ResourceSpan>,
    destination_select: u32,
    base_mip_level: u32,
    mip_level_count: u32,
    texture_type: u32,
    layer_count: u32,
}

#[derive(Clone, Copy, Debug)]
enum GuestScissor {
    Full,
    Rect(MTLScissorRect),
    Empty,
}

struct MetalPresenter {
    receiver: OrderedGpuEventReceiver,
    window: Option<Window>,
    device: Option<Device>,
    layer: Option<MetalLayer>,
    command_queue: Option<metal::CommandQueue>,
    /// A4: presentation runs on its own `MTLCommandQueue`, distinct from the legacy guest command
    /// queue, so a drawable present never serializes against guest submissions on one queue. Its
    /// dependency on the guest snapshot it samples is expressed by a `SharedEvent` wait
    /// (`snapshot_wait` in `present.rs`), not by sharing a queue.
    presentation_queue: Option<metal::CommandQueue>,
    /// Wakes AppKit immediately when a guest Metal batch reaches a terminal state.
    event_loop_waker: Option<EventLoopProxy<()>>,
    per_queue_guest_batches: bool,
    /// Experimental B17 admission for direct draws proven disjoint from exact producer writes.
    /// This remains independently opt-in so the classifier can collect an unbiased OFF baseline.
    shared_event_direct_draw_admission: bool,
    /// Optional high-resolution attribution of presenter terminal-retirement work.
    terminal_publication_timing: bool,
    /// Bounded command-buffer phase attribution for consumer acknowledgement batches.
    shared_event_ack_phase_timing: bool,
    shared_event_ack_phase_samples_remaining: u32,
    /// Per-window structured records (plan E4); opened on the first 5-second window when the
    /// launcher exported a run directory.
    telemetry_log: Option<telemetry::TelemetryLog>,
    telemetry_log_attempted: bool,
    /// Timebase count of the vblank that carried the most recent drawable present (plan E1).
    last_presented_vblank: u64,
    shared_event_ack_batch_cpu: HashMap<u64, SharedEventAckBatchCpuPhases>,
    pending_shared_event_ack_traces: HashMap<u64, PendingSharedEventAckTrace>,
    shared_event_ack_traces: HashMap<usize, SharedEventAckTrace>,
    shared_event_ack_trace_buffers_by_fence: HashMap<SharedEventFence, usize>,
    /// Monotonic marker for any explicit CPU settlement of guest Metal batches.
    /// Interval statistics reset, so they cannot reliably identify a drain inside one draw.
    guest_batch_cpu_synchronization_epoch: u64,
    active_guest_queue: Option<u64>,
    guest_queue_batches: HashMap<u64, GuestQueueBatchState>,
    gpu_wait_fences: GpuWaitFenceMap,
    presentation: Option<PresentationResources>,
    presentation_resolution: PresentationResolution,
    guest_bgra_bytes: Vec<u8>,
    frame_texture: Option<Texture>,
    frame_texture_size: (u32, u32),
    host_splash: Option<HostSplashTexture>,
    display: Option<DisplayConfiguration>,
    /// Addresses from the most recent `DisplayRegistered` event (typically both VideoOut buffers).
    registered_video_out_buffers: Vec<u64>,
    pending_flip: Option<PendingFlip>,
    pending_flip_queue: Option<u64>,
    pending_flip_target_written_since_boundary: bool,
    /// Registered VideoOut buffer written since the last submission boundary when no flip arrived.
    pending_scanout_write_address: Option<u64>,
    active_translated_scanout: Option<ActiveTranslatedScanout>,
    next_flip_sequence: u64,
    finalized_flip_sequence: u64,
    uploaded_flip_sequence: u64,
    submitted_flip_sequence: u64,
    /// Accepted flips awaiting the two independently terminal halves of their completion join.
    ///
    /// The ordered map also prevents a newer cross-queue flip from becoming guest-visible ahead
    /// of an older accepted flip whose exact guest submission has not retired yet.
    flip_completion_joins: BTreeMap<u64, FlipCompletionJoin>,
    flip_completion_join_statistics: FlipCompletionJoinStatistics,
    /// Latest accepted flip generation for each registered `(address, index)` pair.
    last_flip_sequence_by_buffer: HashMap<(u64, u32), u64>,
    /// Private-snapshot safety for accepted flips newer than `safe_flip_sequence_prefix`.
    flip_buffer_safety: BTreeMap<u64, FlipBufferSafety>,
    safe_flip_sequence_prefix: u64,
    startup_error: Option<String>,
    logged_unsupported_layout: bool,
    logged_unsupported_guest_draws: HashSet<([u32; 4], [u32; 4])>,
    captured_guest_frames: HashSet<(u64, u32, u32, u32, u64)>,
    last_guest_capture: Option<Instant>,
    compiled_shader_pairs: HashMap<GraphicsPipelineKey, CompiledShaderPair>,
    compiling_shader_pairs: HashSet<GraphicsPipelineKey>,
    failed_shader_pairs: HashSet<GraphicsPipelineKey>,
    /// Draws whose execution failed after pipeline compilation, keyed by
    /// (export shader address, pixel shader address). Every failing execution counts, not just
    /// the first: a scene pass failing every frame must be visible in every run's perf trail
    /// instead of being silently dropped after one log line.
    dropped_draw_executions: HashMap<(u64, u64), u64>,
    compiled_compute_pipelines: HashMap<ComputePipelineKey, CompiledComputePipeline>,
    compiling_compute_pipelines: HashSet<ComputePipelineKey>,
    rectangle_indirect_pipeline: Option<ComputePipelineState>,
    ngg_indirect_pipeline: Option<ComputePipelineState>,
    mode27_rematerialisation_scatter_pipeline: Option<ComputePipelineState>,
    mode27_rematerialisation_gather_pipeline: Option<ComputePipelineState>,
    failed_compute_pipelines: HashSet<ComputePipelineKey>,
    failed_compute_execution: HashSet<u64>,
    /// Compute dispatches whose execution failed, keyed by shader address; same contract as
    /// `dropped_draw_executions`.
    dropped_compute_executions: HashMap<u64, u64>,
    resident_guest_textures: ResidentTextureMap,
    depth_array_textures: HashMap<DepthArrayKey, DepthArrayResidentTexture>,
    direct_scanout_textures: HashMap<u64, DirectScanoutTexture>,
    scanout_snapshot_textures: HashMap<u64, ScanoutSnapshotTexture>,
    multisample_textures: HashMap<MultisampleTextureKey, MultisampleResidentTexture>,
    /// J1: Shared `MTLVisibilityResultBuffer` for Counting-mode occlusion queries.
    visibility_result_buffer: Option<Buffer>,
    /// True after at least one draw armed Counting mode since the last OcclusionQuery publish.
    visibility_counting_armed: bool,
    /// Logical HTILE clear state keyed by the byte address in DB_HTILE_DATA_BASE. Metal cannot
    /// consume guest HTILE directly, so the next attachment use materializes a pending state as
    /// a depth load clear.
    depth_metadata_clears: HashMap<u64, bool>,
    /// Logical CMASK clear bits keyed by CB_COLOR_CMASK byte address. REmu stores color targets
    /// uncompressed, so attachment begin expands a pending bit through a Metal load clear and a
    /// later EliminateFastClear observes that the bit was already consumed.
    color_metadata_clears: HashMap<u64, u32>,
    next_depth_metadata_clear_sequence: u64,
    sampled_texture_cache: HashMap<SampledTextureKey, CachedSampledTexture>,
    sampled_texture_write_watches: HashMap<SampledTextureKey, Arc<GuestTextureWriteWatch>>,
    fallback_texture_cache: HashMap<SampledTextureKey, Texture>,
    next_sampled_texture_use: u64,
    sampler_argument_cache: HashMap<SamplerArgumentCacheKey, CachedSamplerArguments>,
    depth_stencil_cache: HashMap<DepthStencilCacheKey, DepthStencilState>,
    global_buffer_cache: HashMap<GlobalBufferCacheKey, CachedGlobalBuffer>,
    transient_global_buffer_cache: HashMap<GlobalBufferCacheKey, CachedGlobalBuffer>,
    guest_global_buffer_cache: HashMap<(GuestSharedMappingIdentity, usize), Buffer>,
    guest_global_buffer_cache_bytes: u64,
    global_buffer_cache_bytes: usize,
    index_buffer_cache: HashMap<Vec<u32>, Buffer>,
    ngg_meshlet_cache: HashMap<(NggSubgroupLimits, Vec<u32>), CachedNggMeshlets>,
    diagnostic_vertex_position_buffer: Option<Buffer>,
    diagnostic_pixel_output_buffer: Option<Buffer>,
    resident_content_version: u64,
    open_guest_batch: Option<GuestCommandBatch>,
    committed_guest_batches: VecDeque<GuestCommandBatch>,
    guest_memory_waits: GuestMemoryWaitTracker,
    pending_presentations: VecDeque<PendingPresentation>,
    statistics: PresenterStatistics,
    capture_directory: Option<PathBuf>,
    periodic_frame_capture: bool,
    pressed_keys: HashSet<KeyCode>,
    shader_compile_sender: SyncSender<ShaderCompileRequest>,
    shader_compile_receiver: Receiver<ShaderCompileResult>,
    render_pipeline_archive: Option<PersistentRenderPipelineArchive>,
    redraw_pending: bool,
}

impl MetalPresenter {
    fn new(receiver: Receiver<GpuEvent>) -> Result<Self, String> {
        // D5: register the read-watch materializer before any resident can arm a read-watch.
        // Without a materializer, `CpuWriteWatch::new_read_materialized` loudly refuses.
        let device = Device::system_default().ok_or("no Metal device is available")?;
        let command_queue = device.new_command_queue();
        command_queue.set_label("REmu guest command queue");
        // A4: a dedicated presentation queue. Ordering against the guest work it samples is a
        // SharedEvent wait, so this queue can drain drawable presents without queueing behind
        // guest submissions.
        let presentation_queue = device.new_command_queue();
        presentation_queue.set_label("REmu presentation queue");
        let presentation_resolution = PresentationResolution::from_environment()?;
        // Headless traces exercise the complete guest draw/compute/scanout path and capture the
        // finalized texture before CAMetalLayer presentation. They must not initialize the
        // window-only fullscreen pipeline: doing so both wastes startup time and makes a trace
        // depend on macOS's external MTLCompilerService even when every guest metallib is cached.
        let presentation = if headless_metal_requested() {
            None
        } else {
            Some(create_presentation_resources(&device)?)
        };
        let (shader_compile_sender, shader_compile_requests) =
            mpsc::sync_channel::<ShaderCompileRequest>(SHADER_COMPILE_QUEUE_CAPACITY);
        let (shader_compile_results, shader_compile_receiver) =
            mpsc::channel::<ShaderCompileResult>();
        let shader_compile_requests = Arc::new(Mutex::new(shader_compile_requests));
        let function_cache = ShaderFunctionCache::default();
        let render_pipeline_archive = if shader::offline_metal_compiler().is_some() {
            match PersistentRenderPipelineArchive::open(&device) {
                Ok(archive) => Some(archive),
                Err(error) => {
                    eprintln!(
                        "[REMu][METAL] persistent render pipeline cache unavailable: {error}"
                    );
                    None
                }
            }
        } else {
            None
        };
        let shader_compiler_workers = shader_compiler_worker_count();
        eprintln!(
            "[REMu][METAL] compiler workers={shader_compiler_workers} sampled texture cache={} entries/{} MiB resident texture cache={} MiB exact-shared-event-readiness=true retired-fence-range-bookkeeping=false",
            SAMPLED_TEXTURE_CACHE_CAPACITY,
            sampled_texture_cache_byte_capacity().map_or(0, |bytes| bytes / (1024 * 1024)),
            resident_texture_cache_byte_capacity().map_or(0, |bytes| bytes / (1024 * 1024)),
        );
        for worker in 0..shader_compiler_workers {
            let compiler_device = device.clone();
            let shader_compile_requests = Arc::clone(&shader_compile_requests);
            let shader_compile_results = shader_compile_results.clone();
            let function_cache = function_cache.clone();
            let render_pipeline_archive = render_pipeline_archive.clone();
            std::thread::Builder::new()
                .name(format!("remu-metal-compiler-{worker}"))
                .spawn(move || {
                    loop {
                        let request = shader_compile_requests
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .recv();
                        let Ok(request) = request else {
                            break;
                        };
                        let result =
                            compile_shader_request_catching_panics(&request, || match &request {
                                ShaderCompileRequest::Graphics { key, draw } => {
                                    let compiled = Self::compile_draw_shader_pair(
                                        &compiler_device,
                                        draw,
                                        key,
                                        &function_cache,
                                        render_pipeline_archive.as_ref(),
                                    );
                                    ShaderCompileResult::Graphics {
                                        key: key.clone(),
                                        compiled,
                                    }
                                }
                                ShaderCompileRequest::Compute { key, dispatch } => {
                                    let shader_address = dispatch.shader_address;
                                    let threads_per_group = dispatch.launch.threads_per_group;
                                    let compiled = Self::compile_compute_pipeline(
                                        &compiler_device,
                                        dispatch,
                                        key,
                                        render_pipeline_archive.as_ref(),
                                    );
                                    ShaderCompileResult::Compute {
                                        key: key.clone(),
                                        shader_address,
                                        threads_per_group,
                                        compiled,
                                    }
                                }
                            });
                        if shader_compile_results.send(result).is_err() {
                            break;
                        }
                    }
                })
                .map_err(|error| format!("could not start Metal compiler worker: {error}"))?;
        }
        drop(shader_compile_results);
        let per_queue_guest_batches =
            graduated_flag_enabled(std::env::var_os("REMU_PER_QUEUE_BATCHES").as_deref());
        let shared_event_direct_draw_admission = per_queue_guest_batches
            && graduated_flag_enabled(
                std::env::var_os("REMU_SHARED_EVENT_DIRECT_DRAWS").as_deref(),
            );
        let terminal_publication_timing =
            std::env::var_os("REMU_GPU_TERMINAL_PHASES").is_some_and(|value| value == "1");
        let shared_event_ack_phase_timing = per_queue_guest_batches
            && std::env::var_os("REMU_SHARED_EVENT_ACK_PHASES").is_some_and(|value| value == "1");
        if terminal_publication_timing {
            eprintln!(
                "[REMu][PERF-GPU-TERMINAL] optional presenter terminal phase attribution enabled"
            );
        }
        if shared_event_ack_phase_timing {
            eprintln!(
                "[REMu][PERF-SHARED-EVENT-ACK-PHASE] bounded consumer acknowledgement attribution enabled samples-per-window=64"
            );
        }
        let mut receiver = OrderedGpuEventReceiver::new(receiver);
        receiver.set_event_queue_timing(shared_event_ack_phase_timing);
        Ok(Self {
            receiver,
            window: None,
            device: Some(device),
            layer: None,
            command_queue: Some(command_queue),
            presentation_queue: Some(presentation_queue),
            event_loop_waker: None,
            per_queue_guest_batches,
            shared_event_direct_draw_admission,
            terminal_publication_timing,
            shared_event_ack_phase_timing,
            shared_event_ack_phase_samples_remaining: 64,
            telemetry_log: None,
            telemetry_log_attempted: false,
            last_presented_vblank: 0,
            shared_event_ack_batch_cpu: HashMap::new(),
            pending_shared_event_ack_traces: HashMap::new(),
            shared_event_ack_traces: HashMap::new(),
            shared_event_ack_trace_buffers_by_fence: HashMap::new(),
            guest_batch_cpu_synchronization_epoch: 0,
            active_guest_queue: None,
            guest_queue_batches: HashMap::new(),
            gpu_wait_fences: GpuWaitFenceMap::default(),
            presentation,
            presentation_resolution,
            guest_bgra_bytes: Vec::new(),
            frame_texture: None,
            frame_texture_size: (0, 0),
            host_splash: None,
            display: None,
            registered_video_out_buffers: Vec::new(),
            pending_flip: None,
            pending_flip_queue: None,
            pending_flip_target_written_since_boundary: false,
            pending_scanout_write_address: None,
            active_translated_scanout: None,
            next_flip_sequence: 0,
            finalized_flip_sequence: 0,
            uploaded_flip_sequence: 0,
            submitted_flip_sequence: 0,
            flip_completion_joins: BTreeMap::new(),
            flip_completion_join_statistics: FlipCompletionJoinStatistics::default(),
            last_flip_sequence_by_buffer: HashMap::new(),
            flip_buffer_safety: BTreeMap::new(),
            safe_flip_sequence_prefix: 0,
            startup_error: None,
            logged_unsupported_layout: false,
            logged_unsupported_guest_draws: HashSet::new(),
            captured_guest_frames: HashSet::new(),
            last_guest_capture: None,
            compiled_shader_pairs: HashMap::new(),
            compiling_shader_pairs: HashSet::new(),
            failed_shader_pairs: HashSet::new(),
            dropped_draw_executions: HashMap::new(),
            compiled_compute_pipelines: HashMap::new(),
            compiling_compute_pipelines: HashSet::new(),
            rectangle_indirect_pipeline: None,
            ngg_indirect_pipeline: None,
            mode27_rematerialisation_scatter_pipeline: None,
            mode27_rematerialisation_gather_pipeline: None,
            failed_compute_pipelines: HashSet::new(),
            failed_compute_execution: HashSet::new(),
            dropped_compute_executions: HashMap::new(),
            resident_guest_textures: ResidentTextureMap::new(),
            depth_array_textures: HashMap::new(),
            direct_scanout_textures: HashMap::new(),
            scanout_snapshot_textures: HashMap::new(),
            multisample_textures: HashMap::new(),
            visibility_result_buffer: None,
            visibility_counting_armed: false,
            depth_metadata_clears: HashMap::new(),
            color_metadata_clears: HashMap::new(),
            next_depth_metadata_clear_sequence: 0,
            sampled_texture_cache: HashMap::new(),
            sampled_texture_write_watches: HashMap::new(),
            fallback_texture_cache: HashMap::new(),
            next_sampled_texture_use: 0,
            sampler_argument_cache: HashMap::new(),
            depth_stencil_cache: HashMap::new(),
            global_buffer_cache: HashMap::new(),
            transient_global_buffer_cache: HashMap::new(),
            guest_global_buffer_cache: HashMap::new(),
            guest_global_buffer_cache_bytes: 0,
            global_buffer_cache_bytes: 0,
            index_buffer_cache: HashMap::new(),
            ngg_meshlet_cache: HashMap::new(),
            diagnostic_vertex_position_buffer: None,
            diagnostic_pixel_output_buffer: None,
            resident_content_version: 0,
            open_guest_batch: None,
            committed_guest_batches: VecDeque::new(),
            guest_memory_waits: GuestMemoryWaitTracker::default(),
            pending_presentations: VecDeque::new(),
            statistics: PresenterStatistics::new(),
            capture_directory: std::env::var_os("REMU_CAPTURE_DIR").map(PathBuf::from),
            periodic_frame_capture: std::env::var_os("REMU_CAPTURE_FRAMES")
                .is_some_and(|value| value == "1"),
            pressed_keys: HashSet::new(),
            shader_compile_sender,
            shader_compile_receiver,
            render_pipeline_archive,
            redraw_pending: false,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GuestMipLevelLayout {
    width: usize,
    height: usize,
    padded_width: usize,
    offset: usize,
    tail_x: usize,
    tail_y: usize,
    in_tail: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GuestMipChainLayout {
    levels: Vec<GuestMipLevelLayout>,
    byte_count: usize,
    block_width: usize,
    block_height: usize,
    bytes_per_element: usize,
}

pub fn run_presenter(receiver: Receiver<GpuEvent>) -> Result<(), String> {
    remu_gpu::record_startup_milestone(remu_gpu::StartupMilestone::PresenterEnter);
    if headless_metal_requested() {
        return run_headless_presenter(receiver);
    }
    disable_appkit_state_restoration();
    eprintln!("[REMu][METAL] starting AppKit event loop");
    let mut presenter = MetalPresenter::new(receiver)?;
    remu_gpu::record_startup_milestone(remu_gpu::StartupMilestone::PresenterResourcesReady);
    let mut builder = EventLoop::builder();
    builder
        .with_activation_policy(ActivationPolicy::Regular)
        .with_activate_ignoring_other_apps(true);
    let event_loop = builder
        .build()
        .map_err(|error| format!("event loop failed: {error}"))?;
    presenter.event_loop_waker = Some(event_loop.create_proxy());
    display_link::start_display_link(event_loop.create_proxy());
    if presenter.per_queue_guest_batches {
        eprintln!("[REMu][METAL] guest-batch completion wakeups enabled");
    }
    if presenter.shared_event_direct_draw_admission {
        eprintln!(
            "[REMu][METAL] experimental disjoint/fence-versioned/resident-versioned direct-draw SharedEvent admission enabled"
        );
    }
    event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + EVENT_POLL_INTERVAL));
    // Winit owns NSApplication's delegate and launch state. Calling `finishLaunching` or its
    // delegate callback ourselves marks the delegate as running before `run_app`, causing a
    // re-entrant AppKit loop assertion. Window and Metal-layer creation happen from `resumed`,
    // after Winit has delivered the canonical launch sequence on the main thread.
    event_loop
        .run_app(&mut presenter)
        .map_err(|error| format!("event loop failed: {error}"))?;
    eprintln!("[REMu][METAL] AppKit event loop stopped");
    if let Some(error) = presenter.startup_error {
        Err(error)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
