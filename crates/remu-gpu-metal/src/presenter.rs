// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use metal::{
    BinaryArchive, Buffer, CommandBuffer, ComputePipelineState, DepthStencilState, Device,
    Function, MTLBlendFactor, MTLBlendOperation, MTLCompareFunction, MTLCullMode, MTLPixelFormat,
    MTLPrimitiveTopologyClass, MTLScissorRect, MTLTriangleFillMode, MTLWinding, MetalLayer,
    RenderCommandEncoder, RenderPipelineState, SamplerState, Texture,
};
use remu_gpu::{GpuEvent, GpuMemoryRelease, GpuMemoryWait, GpuMemoryWrite, IndexedDraw};
use remu_memory::{CpuWriteWatch, GuestSharedMappingIdentity};
mod cache;
mod completion;
mod decode;
mod diagnostics;
mod execute;
mod format;
mod present;
mod readback;
mod sampler;
mod shader;
mod stats;
mod submission;
mod texture;
mod tiling;
use self::cache::{GLOBAL_UPLOAD_ALIGNMENT, GLOBAL_UPLOAD_CHUNK_BYTE_COUNT};
use self::decode::{
    CB_BLEND_ALPHA, CB_BLEND_BLUE, CB_BLEND_GREEN, CB_BLEND_RED, CB_BLEND0_CONTROL,
    CB_COLOR_CONTROL, CB_COLOR_REGISTER_STRIDE, CB_COLOR0_FMASK, CB_COLOR0_FMASK_BASE_EXT,
    CB_TARGET_MASK, ColorOperationMode, DB_DEPTH_CONTROL, DB_DEPTH_SIZE_XY, DB_Z_INFO,
    DEFAULT_CB_COLOR_CONTROL, PA_SC_AA_MASK_X0Y0_X1Y0, PA_SC_AA_MASK_X0Y1_X1Y1, PA_SU_SC_MODE_CNTL,
    color_operation_is_raster_noop, decode_color_eliminate_fast_clear_target,
    decode_color_operation_mode, decode_color_resolve_targets, decode_depth_target,
    decode_image_descriptor, decode_metal_blend_attachment, decode_raster_state,
    decode_render_target_format, decode_render_targets, decode_scissor, decode_viewport,
    fallback_image_descriptor, graphics_pipeline_key, image_component_class,
    initial_depth_clear_for_target, initial_render_target_clear_color, metal_color_write_mask,
    metal_nonindexed_vertex_count, metal_primitive_type, prepare_indexed_primitive,
    prepare_ngg_meshlets, raw_video_out_texture_descriptor, read_draw_indices, register_value,
    render_target_cmask_address, render_target_fast_clear_color, sampled_alias_descriptor,
};
#[cfg(test)]
use self::decode::{
    CB_COLOR0_ATTRIB, CB_COLOR0_ATTRIB2, CB_COLOR0_ATTRIB3, CB_COLOR0_BASE, CB_COLOR0_BASE_EXT,
    CB_COLOR0_CLEAR_WORD0, CB_COLOR0_CLEAR_WORD1, CB_COLOR0_CMASK, CB_COLOR0_INFO,
    CB_COLOR0_VIEW, DB_DEPTH_CLEAR, DB_DEPTH_VIEW, DB_HTILE_DATA_BASE, DB_RENDER_CONTROL,
    DB_Z_READ_BASE, DB_Z_READ_BASE_HI, DB_Z_WRITE_BASE, DB_Z_WRITE_BASE_HI,
    NGG_MESHLET_WORDS_PER_GROUP, SPI_PS_INPUT_CNTL_0, VIDEO_OUT_DISPLAY_TILE_MODE,
    VIDEO_OUT_TILING_MODE_LINEAR, VIDEO_OUT_TILING_MODE_TILE, common_render_target_array_length,
    decode_fast_clear_color,
    decode_render_targets_from_registers, decode_scanout_target, graphics_shader_resource_key,
    initial_depth_clear_value, metal_primitive_topology_class, pixel_input_controls,
    raw_video_out_layout_is_linear,
};
#[cfg(test)]
use self::diagnostics::parse_diagnostic_hex_address;
use self::diagnostics::{
    DIAGNOSTIC_PIXEL_OUTPUT_BUFFER_INDEX, DIAGNOSTIC_PIXEL_OUTPUT_HEIGHT,
    DIAGNOSTIC_PIXEL_OUTPUT_WIDTH, DIAGNOSTIC_VERTEX_POSITION_BUFFER_INDEX,
    DIAGNOSTIC_VERTEX_POSITION_VERTEX_CAPACITY, DIAGNOSTIC_VERTEX_STATE_WORDS_PER_VERTEX,
    capture_target_sample_textures, diagnostic_capture_pixel_output,
    diagnostic_capture_vertex_position, diagnostic_context_register_snapshot,
    diagnostic_disable_blending, diagnostic_disable_culling, diagnostic_disable_depth,
    diagnostic_disable_depth_and_culling, diagnostic_dump_complex_msaa,
    diagnostic_dump_shadow_consumer, diagnostic_dump_target_before_pixel_shader,
    diagnostic_dump_target_match_index, diagnostic_dump_target_minimum_draw,
    diagnostic_dump_target_pixel_instruction_count, diagnostic_dump_target_pixel_shader,
    diagnostic_dump_video_chain_consumer, diagnostic_dump_video_draw,
    diagnostic_dump_video_target_consumer, diagnostic_force_msaa_pixel_color,
    diagnostic_force_msaa_vertex_pattern, diagnostic_force_vertex_fetch_id,
    diagnostic_force_visible_clip_depth, diagnostic_global_binding_head,
    diagnostic_overlapping_global_writers, diagnostic_overlapping_memory_copies,
    diagnostic_overlapping_texture_writers, diagnostic_related_texture_writers,
    diagnostic_scanout_writer, diagnostic_shader_summary, diagnostic_target_draw_filter,
    diagnostic_texture_writer, diagnostic_texture_writer_enabled,
    diagnostic_trace_all_target_contents, diagnostic_trace_target_contents,
    diagnostic_trace_target_maximum_draw, diagnostic_trace_target_stride,
    diagnostic_use_draw_merged_wave_counts, diagnostic_video_chain_selected_draw,
    dump_msl_compute_filter_matches, dump_msl_instruction_filter_matches,
    dump_msl_pixel_filter_matches, evaluation_samples_large_linear_video,
    evaluation_samples_layered_comparison, record_diagnostic_global_writers,
    record_diagnostic_memory_copy, record_diagnostic_scanout_writer,
    record_diagnostic_texture_writer, should_record_diagnostic_target_writer,
    trace_image_descriptors, trace_msaa_writes, trace_scanout_writes, trace_target_sampling,
    trace_video_target_compute_consumer, trace_video_target_draw_consumer,
    video_target_binding_summary,
};
use self::format::{
    compressed_texture_block_extent, decode_sampled_texture_format, half_float_to_f32,
    half_float_to_srgb_unorm8, half_float_to_unorm8, linear_unorm8_to_srgb_unorm8,
    texture_element_layout, unsigned_float_to_f32, unsigned_float_to_srgb_unorm8,
};
use self::present::{
    create_presentation_resources, headless_metal_requested, run_headless_presenter,
};
#[cfg(test)]
use self::present::{
    headless_metal_value_is_enabled, parse_presentation_resolution, presentation_drawable_size,
    presentation_viewport,
};
use self::readback::{
    capture_guest_frame, diagnostic_texture_depth_plane_summaries, guest_frame_fingerprint,
    metal_uncompressed_bytes_per_pixel, native_texture_capture_to_bgra, read_texture_array_linear,
    read_texture_linear, read_texture_volume_linear, write_resident_texture_contents_to_guest,
};
#[cfg(test)]
use self::sampler::sampler_lod_bias;
use self::sampler::{
    SAMPLER_ARGUMENT_CACHE_CAPACITY, canonical_sampler_bindings, canonical_texture_bindings,
    create_global_argument_buffer, create_sampler_argument_buffer, guest_global_buffer_view,
    metal_buffer_with_bytes, sampled_image_bindings, sampler_uses_normalized_coordinates,
};
use self::shader::{
    SHADER_COMPILE_QUEUE_CAPACITY, compile_metal_compute_library, compile_metal_library_with_tools,
    compute_pipeline_key, compute_program_can_split_wave64, dump_compute_shader_msl,
    dump_shader_pair_msl, graphics_allocated_vector_register_count,
    image_resource_descriptor_is_null, is_storage_image_opcode, is_writable_image_opcode,
    pixel_scalars, program_embeds_global_base, shader_scalar_metadata_for_registers,
    validate_compute_launch_for_metal, wave64_split_blockers,
};
#[cfg(test)]
use self::shader::{
    missing_vertex_parameter_exports, procedural_rectangle_export_stage,
    rectangle_coordinate_parameter_mask, rectangle_draw_has_incomplete_interface,
    shader_scalar_metadata,
};
use self::stats::{
    MIP_UPLOAD_BYTES, MIP_UPLOAD_COUNT, MIP_UPLOAD_NANOSECONDS, MODE27_DECODE_BYTES,
    MODE27_DECODE_COUNT, MODE27_DECODE_NANOSECONDS, PIPELINE_COMPILE_COMPLETED,
    PIPELINE_COMPILE_ENQUEUED, PIPELINE_COMPILE_FAILED, PIPELINE_COMPILE_QUEUE_FULL,
    PIPELINE_DRAW_GATES, RESIDENT_DEPTH_SAMPLE_HITS, RESIDENT_TEXTURE_ACQUIRE_INVALIDATIONS,
    SAMPLED_ALIAS_WRITEBACK_BYTES, SAMPLED_ALIAS_WRITEBACK_COUNT,
    SAMPLED_ALIAS_WRITEBACK_NANOSECONDS, SAMPLED_TEXTURE_CACHE_ACQUIRE_INVALIDATIONS,
    SAMPLED_TEXTURE_CACHE_CAPACITY_EVICTIONS, SAMPLED_TEXTURE_CACHE_DIRTY_RELOADS,
    SAMPLED_TEXTURE_CACHE_HITS, SAMPLED_TEXTURE_CACHE_MISSES, TEXTURE_DIRTY_VALIDATION_BYTES,
    TEXTURE_DIRTY_VALIDATION_COUNT, TEXTURE_DIRTY_VALIDATION_NANOSECONDS,
    TEXTURE_DIRTY_VALIDATION_UNCHANGED, TEXTURE_SNAPSHOT_BYTES, TEXTURE_SNAPSHOT_COUNT,
    TEXTURE_SNAPSHOT_NANOSECONDS, process_memory_statistics, record_mode27_work_detail,
    record_texture_dirty_work_detail, record_texture_snapshot_work_detail, record_texture_work,
    take_mode27_work_details, take_texture_dirty_work_details, take_texture_snapshot_work_details,
};
#[cfg(test)]
use self::stats::process_resident_bytes;
use self::submission::{
    append_unique_global_writebacks, append_unique_guest_write_ranges,
    collect_pending_global_writebacks, collect_pending_guest_write_ranges,
    completed_command_buffer_gpu_duration, describe_guest_batch, guest_batch_writes_overlap,
};
#[cfg(test)]
use self::submission::{
    apply_pending_guest_memory_writes, pin_guest_memory_write, write_back_pending_globals,
};
use self::texture::{
    RESIDENT_FLIP_AGE_LABELS, RESIDENT_WORK_AGE_LABELS, decode_guest_texture_linear_from_raw,
    refresh_resident_texture_write_watch, resident_color_view_sample_is_eligible,
    resident_depth_sample_is_eligible, resident_texture_activity_stats,
    resident_texture_backing_overlap_stats, resident_texture_ownership_stats,
    retire_multisample_textures_overlapping_writeback,
    retire_resident_textures_overlapping_writeback, retire_sampled_textures_matching,
    retire_sampled_textures_overlapping_writeback, snapshot_guest_texture_raw,
    snapshot_watched_guest_range, storage_image_requires_mip_hierarchy,
    trace_resident_texture_removal, upload_image_textures,
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
    prepare_sampled_texture_upload, resident_texture_eviction_plan, sampled_texture_cache_accepts,
    sampled_texture_cache_needs_reload, sampled_texture_cache_requires_eviction,
    sampled_texture_key, sampled_texture_raw_range, snapshot_guest_texture_linear,
    snapshot_watched_guest_texture_raw, watch_guest_texture, write_guest_texture_linear,
};
use self::tiling::{
    GuestTextureOverlapQuery, copy_linear_to_tiled_at_z, copy_mode27_mip_level_to_linear,
    copy_tiled_to_linear_at_z, decode_mode5_mip_chain_from_raw, decode_mode27_mip_chain_from_raw,
    effective_linear_pitch, encode_mode27_mip_chain_into_raw, guest_physical_ranges_overlap,
    guest_ranges_overlap, guest_texture_byte_count, guest_texture_descriptors_have_same_layout,
    guest_texture_descriptors_overlap, guest_texture_descriptors_share_color_view_storage,
    guest_texture_descriptors_share_depth_sample_storage, guest_texture_descriptors_share_storage,
    mode5_mip_chain_layout, mode27_mip_chain_layout, render_resolve_candidate_matches,
    tiled_block_dimensions,
};
#[cfg(test)]
use self::tiling::{
    encode_mode5_mip_chain_into_raw, guest_texture_descriptors_share_component_reordered_storage,
    mode5_within_block_byte_offset, mode9_within_block_byte_offset,
    mode24_within_block_byte_offset_for_format, mode27_block_dimensions, mode27_source_byte_offset,
    mode27_surface_z_xor, mode27_within_block_byte_offset,
    mode27_within_block_byte_offset_for_format,
};

#[cfg(test)]
use remu_memory::guest_shared_mapping_identity;
use remu_shader::PixelOutputKind;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::KeyCode;
use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
use winit::window::Window;
const EVENT_POLL_INTERVAL: Duration = Duration::from_millis(1);
/// A late Superliminal frame contains roughly 700 draws. Draining only 16 events per AppKit
/// callback imposed an artificial ~2k draw/s ceiling even when both the guest producer and Metal
/// encoder had capacity. This bound covers most of a frame while still returning to AppKit often
/// enough to service input and window events.
const GPU_EVENTS_PER_TICK: usize = 512;
// Unity can introduce dozens of distinct 200-400 KiB pixel programs at a scene boundary. Four
// workers let the ordered GPU queue grow by several thousand events while Apple's compiler is
// busy, which in turn withholds real guest flips and starves the guest audio producers. Each
// offline compiler is a separate memory-heavy process, so eight workers can amplify a large UE
// transition into system-wide pressure even when REmu's own RSS still looks reasonable.
/// Keep individual Metal submissions short enough for the shared Apple GPU to return regularly
/// to WindowServer. Unreal transition scenes can otherwise put hundreds of full-resolution draws
/// in one command buffer and monopolize the device for seconds.
const GUEST_BATCH_WORK_CAPACITY: usize = 32;

/// Maximum number of committed guest batches allowed ahead of the host GPU.
///
/// Each batch retains its transient uploads, render targets, and writeback resources until Metal
/// completes it. Bounding the queue prevents a stalled or simply overloaded game from growing
/// unified-memory use indefinitely and making the whole system unresponsive.
const MAX_COMMITTED_GUEST_BATCHES: usize = 3;
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
const NGG_MESH_LDS_BYTES: u32 = 16 * 1024;
const NGG_MESH_THREADS: u64 = 64;

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
    pipeline: RenderPipelineState,
    _vertex_function: Function,
    _fragment_function: Function,
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
    sequence: u64,
}

#[derive(Clone, Copy)]
struct ActiveTranslatedScanout {
    address: u64,
    width: u32,
    height: u32,
    sequence: u64,
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
    AcquireReload,
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
    pixel_input_controls: [u32; 32],
    targets: Vec<RenderTargetPipelineState>,
    depth: Option<DepthTargetPipelineState>,
    sample_count: u32,
    render_target_array_length: u32,
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DiagnosticScanoutWriter {
    draw: u64,
    export_shader: u64,
    pixel_shader: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DecodedDepthTarget {
    texture: GuestTextureDescriptor,
    layer_count: u32,
    htile_address: Option<u64>,
    compare: MTLCompareFunction,
    test_enabled: bool,
    write_enabled: bool,
    clear_enabled: bool,
    clear_depth: f64,
    sample_count: u32,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DecodedRasterState {
    cull_mode: MTLCullMode,
    front_winding: MTLWinding,
    fill_mode: MTLTriangleFillMode,
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
    wave_lane_count: u32,
    allocated_vector_register_count: u32,
    compute_system_registers: Option<remu_shader::ComputeSystemRegisters>,
    initial_scalar_count: usize,
    global_bindings: Vec<(Vec<u32>, u64, usize, bool)>,
    image_bindings: Vec<(u32, bool, bool, u32, u32, u32, Option<u32>)>,
}

struct ShaderCompileResult {
    key: GraphicsPipelineKey,
    compiled: Result<CompiledShaderPair, String>,
}

struct ShaderCompileRequest {
    key: GraphicsPipelineKey,
    draw: IndexedDraw,
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

#[derive(Clone, Default)]
struct ShaderFunctionCache {
    vertex: Arc<ShaderFunctionStageCache>,
    mesh: Arc<ShaderFunctionStageCache>,
    pixel: Arc<ShaderFunctionStageCache>,
}

struct ScheduledGpuSubmission {
    ingress_sequence: u64,
    events: VecDeque<GpuEvent>,
}

struct ScheduledHostEvent {
    ingress_sequence: u64,
    event: GpuEvent,
}

#[derive(Clone, Debug)]
enum QueueBlockReason {
    GuestMemoryWait(GpuMemoryWait),
    GraphicsPipeline(GraphicsPipelineKey),
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
    active_queue: Option<u64>,
    blocked_queues: HashMap<u64, QueueBlockReason>,
    host_events: VecDeque<ScheduledHostEvent>,
    host_block: Option<QueueBlockReason>,
    next_ingress_sequence: u64,
    disconnected: bool,
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
}

struct ResidentGuestTexture {
    texture: Texture,
    component_reorder_view: Option<Texture>,
    descriptor: GuestTextureDescriptor,
    texture_type: u32,
    layer_count: u32,
    mip_level_count: u32,
    write_watch: GuestTextureWriteWatch,
    gpu_written: bool,
    content_version: u64,
    /// CPU-write generations observed when Metal most recently became authoritative.
    ///
    /// Guest heaps routinely reuse image allocations as indirect command buffers. A plain dirty
    /// bit cannot distinguish a later guest write from the presenter's own earlier readback,
    /// because both touch the same watched host pages. The monotonically increasing generations
    /// establish the ownership boundary at the actual GPU write.
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
    fn note_gpu_write(&mut self) {
        self.gpu_write_watch_generations = self.write_watch.generations();
    }

    fn cpu_write_obsoletes_gpu_writeback(
        &mut self,
        refresh: WriteWatchRefresh,
    ) -> Result<bool, String> {
        let possibly_changed = matches!(refresh, WriteWatchRefresh::Renewed { was_dirty: true })
            || self.write_watch.is_dirty()
            || self.write_watch.generations() != self.gpu_write_watch_generations;
        if !possibly_changed {
            return Ok(false);
        }
        if self.write_watch.content_has_changed()? {
            return Ok(true);
        }
        // False-positive page dirtiness and permission-only watch renewal do not transfer
        // ownership back to the CPU. Keep the GPU-current image and establish its boundary in the
        // current watch generation domain.
        self.note_gpu_write();
        Ok(false)
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
    overlap: GuestTextureOverlapQuery,
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
    chunk_fingerprints: Vec<u64>,
    initialized_chunks: Vec<bool>,
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

struct UploadedGlobalBuffer {
    buffer: Buffer,
    offset: u64,
    guest_memory: bool,
}

struct PendingGlobalWriteback {
    buffer: Buffer,
    offset: u64,
    guest_address: u64,
    byte_count: usize,
    mapping_identity: Option<GuestSharedMappingIdentity>,
    write_watch: Option<CpuWriteWatch>,
    original_bytes: Vec<u8>,
}

struct PendingPresentation {
    command_buffer: CommandBuffer,
    submitted_at: Instant,
    flip_completion: Option<remu_gpu::GpuFlipCompletion>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GuestBatchFlushReason {
    EventLoop,
    BlockedWait,
    FlipSnapshot,
    WorkCapacity,
    CommandProcessorBarrier,
    CpuSynchronization,
}

struct PresenterStatistics {
    interval_started: Instant,
    last_logged: Instant,
    guest_flips: u32,
    completed_presents: u32,
    submitted_guest_batches: u32,
    coalesced_pipeline_barriers: u32,
    encoded_draws: u32,
    encoded_compute_dispatches: u32,
    encoded_render_passes: u32,
    render_pass_ends_for_compute: u32,
    render_pass_ends_for_target_change: u32,
    render_pass_ends_for_batch_flush: u32,
    batch_flushes_from_event_loop: u32,
    batch_flushes_for_guest_memory_write: u32,
    batch_flushes_for_blocked_wait: u32,
    batch_flushes_for_flip_snapshot: u32,
    batch_flushes_for_work_capacity: u32,
    batch_flushes_for_command_processor_barrier: u32,
    batch_flushes_for_cpu_synchronization: u32,
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
    command_buffer: CommandBuffer,
    work: Vec<GuestBatchWork>,
    global_upload_chunks: Vec<GlobalUploadChunk>,
    global_writebacks: Vec<PendingGlobalWriteback>,
    guest_write_ranges: Vec<PendingGuestWriteRange>,
    memory_writes: Vec<PendingGuestMemoryWrite>,
    render_pass: Option<OpenGuestRenderPass>,
    /// Guest submissions whose final ordered work is contained in this Metal batch.
    ///
    /// Queue scheduling can complete a newer global sequence before an older blocked queue, so
    /// every boundary is retained rather than incorrectly treating the maximum as a prefix.
    completion_sequences: Vec<u64>,
}

struct GlobalUploadChunk {
    buffer: Buffer,
    used: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingGuestWriteRange {
    address: u64,
    byte_count: u64,
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
}

#[derive(Debug)]
struct PendingGuestMemoryWrite {
    operation: GuestMemoryWriteOperation,
    mapping_identity: Option<GuestSharedMappingIdentity>,
    write_watch: Option<CpuWriteWatch>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GuestMemoryWaitStatus {
    Satisfied,
    Blocked,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GuestMemoryWaitKey {
    source: OrderedEventSource,
    address: u64,
    reference: u64,
    mask: u64,
    compare_function: u32,
    is_64_bit: bool,
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

#[derive(Clone, Copy, Debug)]
struct DecodedImageDescriptor {
    guest: GuestTextureDescriptor,
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
    presentation: Option<PresentationResources>,
    presentation_resolution: PresentationResolution,
    guest_bgra_bytes: Vec<u8>,
    frame_texture: Option<Texture>,
    frame_texture_size: (u32, u32),
    host_splash: Option<HostSplashTexture>,
    display: Option<DisplayConfiguration>,
    pending_flip: Option<PendingFlip>,
    pending_flip_queue: Option<u64>,
    pending_flip_target_written_since_boundary: bool,
    active_translated_scanout: Option<ActiveTranslatedScanout>,
    next_flip_sequence: u64,
    finalized_flip_sequence: u64,
    uploaded_flip_sequence: u64,
    submitted_flip_sequence: u64,
    startup_error: Option<String>,
    logged_unsupported_layout: bool,
    logged_unsupported_guest_draws: HashSet<([u32; 4], [u32; 4])>,
    captured_guest_frames: HashSet<(u64, u32, u32, u32, u64)>,
    last_guest_capture: Option<Instant>,
    compiled_shader_pairs: HashMap<GraphicsPipelineKey, CompiledShaderPair>,
    compiling_shader_pairs: HashSet<GraphicsPipelineKey>,
    failed_shader_pairs: HashSet<GraphicsPipelineKey>,
    failed_draw_execution: HashSet<GraphicsPipelineKey>,
    /// Draws whose execution failed after pipeline compilation, keyed by
    /// (export shader address, pixel shader address). Every failing execution counts, not just
    /// the first: a scene pass failing every frame must be visible in every run's perf trail
    /// instead of being silently dropped after one log line.
    dropped_draw_executions: HashMap<(u64, u64), u64>,
    compiled_compute_pipelines: HashMap<ComputePipelineKey, CompiledComputePipeline>,
    rectangle_indirect_pipeline: Option<ComputePipelineState>,
    failed_compute_pipelines: HashSet<ComputePipelineKey>,
    failed_compute_execution: HashSet<u64>,
    /// Compute dispatches whose execution failed, keyed by shader address; same contract as
    /// `dropped_draw_executions`.
    dropped_compute_executions: HashMap<u64, u64>,
    resident_guest_textures: HashMap<GuestTextureDescriptor, ResidentGuestTexture>,
    depth_array_textures: HashMap<DepthArrayKey, DepthArrayResidentTexture>,
    direct_scanout_textures: HashMap<u64, DirectScanoutTexture>,
    scanout_snapshot_textures: HashMap<u64, ScanoutSnapshotTexture>,
    multisample_textures: HashMap<MultisampleTextureKey, MultisampleResidentTexture>,
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
    depth_stencil_cache: HashMap<(u64, bool), DepthStencilState>,
    global_buffer_cache: HashMap<GlobalBufferCacheKey, CachedGlobalBuffer>,
    transient_global_buffer_cache: HashMap<GlobalBufferCacheKey, CachedGlobalBuffer>,
    guest_global_buffer_cache: HashMap<(u64, usize, GuestSharedMappingIdentity), Buffer>,
    guest_global_buffer_cache_bytes: u64,
    global_buffer_cache_bytes: usize,
    index_buffer_cache: HashMap<Vec<u32>, Buffer>,
    ngg_meshlet_cache: HashMap<Vec<u32>, CachedNggMeshlets>,
    diagnostic_vertex_position_buffer: Option<Buffer>,
    diagnostic_pixel_output_buffer: Option<Buffer>,
    resident_content_version: u64,
    open_guest_batch: Option<GuestCommandBatch>,
    committed_guest_batches: VecDeque<GuestCommandBatch>,
    guest_memory_waits: GuestMemoryWaitTracker,
    pending_presentations: VecDeque<PendingPresentation>,
    statistics: PresenterStatistics,
    capture_directory: Option<PathBuf>,
    pressed_keys: HashSet<KeyCode>,
    shader_compile_sender: SyncSender<ShaderCompileRequest>,
    shader_compile_receiver: Receiver<ShaderCompileResult>,
    render_pipeline_archive: Option<PersistentRenderPipelineArchive>,
    redraw_pending: bool,
}

impl MetalPresenter {
    fn new(receiver: Receiver<GpuEvent>) -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device is available")?;
        let command_queue = device.new_command_queue();
        command_queue.set_label("REmu guest command queue");
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
        let render_pipeline_archive = if std::env::var_os("REMU_METAL_COMPILER").is_some() {
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
            "[REMu][METAL] compiler workers={shader_compiler_workers} sampled texture cache={} entries/{} MiB resident texture cache={} MiB",
            SAMPLED_TEXTURE_CACHE_CAPACITY,
            sampled_texture_cache_byte_capacity()
                .map_or(0, |bytes| bytes / (1024 * 1024)),
            resident_texture_cache_byte_capacity()
                .map_or(0, |bytes| bytes / (1024 * 1024)),
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
                        let Ok(ShaderCompileRequest { key, draw }) = request else {
                            break;
                        };
                        let compiled = objc::rc::autoreleasepool(|| {
                            Self::compile_draw_shader_pair(
                                &compiler_device,
                                &draw,
                                &key,
                                &function_cache,
                                render_pipeline_archive.as_ref(),
                            )
                        });
                        if shader_compile_results
                            .send(ShaderCompileResult { key, compiled })
                            .is_err()
                        {
                            break;
                        }
                    }
                })
                .map_err(|error| format!("could not start Metal compiler worker: {error}"))?;
        }
        drop(shader_compile_results);
        Ok(Self {
            receiver: OrderedGpuEventReceiver::new(receiver),
            window: None,
            device: Some(device),
            layer: None,
            command_queue: Some(command_queue),
            presentation,
            presentation_resolution,
            guest_bgra_bytes: Vec::new(),
            frame_texture: None,
            frame_texture_size: (0, 0),
            host_splash: None,
            display: None,
            pending_flip: None,
            pending_flip_queue: None,
            pending_flip_target_written_since_boundary: false,
            active_translated_scanout: None,
            next_flip_sequence: 0,
            finalized_flip_sequence: 0,
            uploaded_flip_sequence: 0,
            submitted_flip_sequence: 0,
            startup_error: None,
            logged_unsupported_layout: false,
            logged_unsupported_guest_draws: HashSet::new(),
            captured_guest_frames: HashSet::new(),
            last_guest_capture: None,
            compiled_shader_pairs: HashMap::new(),
            compiling_shader_pairs: HashSet::new(),
            failed_shader_pairs: HashSet::new(),
            failed_draw_execution: HashSet::new(),
            dropped_draw_executions: HashMap::new(),
            compiled_compute_pipelines: HashMap::new(),
            rectangle_indirect_pipeline: None,
            failed_compute_pipelines: HashSet::new(),
            failed_compute_execution: HashSet::new(),
            dropped_compute_executions: HashMap::new(),
            resident_guest_textures: HashMap::new(),
            depth_array_textures: HashMap::new(),
            direct_scanout_textures: HashMap::new(),
            scanout_snapshot_textures: HashMap::new(),
            multisample_textures: HashMap::new(),
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
    if headless_metal_requested() {
        return run_headless_presenter(receiver);
    }
    eprintln!("[REMu][METAL] starting AppKit event loop");
    let mut presenter = MetalPresenter::new(receiver)?;
    let mut builder = EventLoop::builder();
    builder
        .with_activation_policy(ActivationPolicy::Regular)
        .with_activate_ignoring_other_apps(true);
    let event_loop = builder
        .build()
        .map_err(|error| format!("event loop failed: {error}"))?;
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
