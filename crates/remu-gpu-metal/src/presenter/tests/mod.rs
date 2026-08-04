use remu_gpu::{ComputeGrid, ComputeLaunch, GpuRegister, IndexedDraw};

use super::{
    CB_COLOR_REGISTER_STRIDE, CB_COLOR0_ATTRIB, CB_COLOR0_ATTRIB2, CB_COLOR0_ATTRIB3,
    CB_COLOR0_BASE, CB_COLOR0_BASE_EXT, CB_COLOR0_CLEAR_WORD0, CB_COLOR0_CLEAR_WORD1,
    CB_COLOR0_CMASK, CB_COLOR0_INFO, CB_COLOR0_VIEW, CB_TARGET_MASK, ColorOperationMode,
    CompiledComputePipeline, ComputePipelineKey, DB_DEPTH_CLEAR, DB_DEPTH_CONTROL,
    DB_DEPTH_SIZE_XY, DB_DEPTH_VIEW, DB_HTILE_DATA_BASE, DB_RENDER_CONTROL, DB_Z_INFO,
    DB_Z_READ_BASE, DB_Z_READ_BASE_HI, DB_Z_WRITE_BASE, DB_Z_WRITE_BASE_HI, DecodedDepthTarget,
    DecodedImageDescriptor, DecodedRasterState, DecodedRenderTarget, DisplayConfiguration,
    FlushPolicy, GLOBAL_UPLOAD_ALIGNMENT, GraphicsDiagnosticPipelineState, GraphicsPipelineKey,
    GraphicsShaderResourceKey, GuestBatchFlushReason, GuestCommandBatch, GuestMemoryWaitKey,
    GuestMemoryWaitRecovery, GuestMemoryWaitStatus, GuestMemoryWaitTracker,
    GuestMemoryWriteOperation, GuestRenderPassKey, GuestScissor, GuestTextureDescriptor,
    MAX_COMMITTED_GUEST_BATCHES, MetalPresenter, Mode27WorkOrigin, NGG_MESHLET_WORDS_PER_GROUP,
    NggOutputPrimitiveType, NggSubgroupLimits, OpenGuestRenderPass, OrderedEventPoll,
    OrderedEventSource, OrderedGpuEventReceiver, PA_SU_SC_MODE_CNTL, PendingFlip,
    PendingGlobalWriteback, PendingGuestMemoryWrite, PendingTextureReadback,
    PipelineCompileCandidate, PresentationResolution, PresenterFlow, ProgramIdentity,
    QueueBlockReason, RangeSet, RenderTargetPipelineState, ResidentTextureActivity,
    ResidentTextureActivityStamp, ResourceSpan, SPI_PS_INPUT_CNTL_0, ShaderCompileRequest,
    ShaderCompileResult, TextureReadbackLayout, VIDEO_OUT_DISPLAY_TILE_MODE,
    VIDEO_OUT_TILING_MODE_LINEAR, VIDEO_OUT_TILING_MODE_TILE, WriteWatchRefresh,
    appkit_persistent_state_is_ignored, apply_image_destination_select,
    apply_pending_guest_memory_writes, apply_transparent_premultiplied_fill_clear,
    canonical_sampler_bindings, canonical_texture_bindings, clear_appkit_persistent_state_override,
    color_operation_is_raster_noop, common_render_target_array_length, comparison_depth32_upload,
    compile_shader_request_catching_panics, consume_color_metadata_clear,
    consume_depth_metadata_clear, copy_linear_to_tiled_at_z, create_exact_mode27_sampled_mip_view,
    create_sampled_guest_texture, decode_color_eliminate_fast_clear_target,
    decode_color_operation_mode, decode_color_resolve_targets, decode_depth_target,
    decode_fast_clear_color, decode_guest_texture_linear_from_raw, decode_image_descriptor,
    decode_metal_blend_attachment, decode_mode5_mip_chain_from_raw,
    decode_mode27_mip_chain_from_raw, decode_ngg_registers_from_state, decode_ngg_subgroup_limits,
    decode_raster_state, decode_render_target_format, decode_render_targets_from_registers,
    decode_sampled_texture_format, decode_sampled_texture_linear_from_raw, decode_scanout_target,
    decode_scissor, decode_viewport, depth_array_face_descriptor, depth_array_face_set_is_complete,
    diagnostic_texture_depth_plane_summaries, disable_appkit_state_restoration,
    disabled_metal_blend_attachment, effective_linear_pitch, encode_mode5_mip_chain_into_raw,
    encode_mode27_mip_chain_into_raw, ensure_mipmapped_resident_texture_in_map,
    export_resource_render_stage, fallback_image_descriptor, fragmentless_pixel_evaluation,
    fragmentless_pixel_program, graphics_pipeline_key, graphics_shader_resource_key,
    guest_frame_fingerprint, guest_shared_mapping_identity, guest_texture_byte_count,
    guest_texture_descriptors_overlap, guest_texture_descriptors_share_color_view_storage,
    guest_texture_descriptors_share_component_reordered_storage,
    guest_texture_descriptors_share_depth_sample_storage,
    guest_texture_descriptors_share_gpu_sample_storage, guest_texture_descriptors_share_storage,
    guest_texture_raw_matches, guest_texture_raw_range, half_float_to_srgb_unorm8,
    half_float_to_unorm8, headless_metal_value_is_enabled, image_component_class,
    image_destination_select_is_native, image_resource_descriptor_is_null,
    initial_depth_clear_for_target, initial_depth_clear_value, initial_render_target_clear_color,
    is_transparent_premultiplied_fill_blend, lift_expanded_vertex_buffer_num_records,
    linear_unorm8_to_srgb_unorm8, metal_color_write_mask, metal_nonindexed_vertex_count,
    metal_primitive_topology_class, metal_primitive_type, missing_vertex_parameter_exports,
    mode5_mip_chain_layout, mode5_within_block_byte_offset, mode9_within_block_byte_offset,
    mode24_within_block_byte_offset_for_format, mode27_block_dimensions, mode27_mip_chain_layout,
    mode27_source_byte_offset, mode27_surface_z_xor, mode27_within_block_byte_offset,
    mode27_within_block_byte_offset_for_format, native_texture_capture_to_bgra,
    new_color_alias_texture_view, new_red_blue_swizzled_texture_view, ngg_export_uses_general_lds,
    ngg_export_uses_wave64_exec, ngg_fullscreen_cover_draw, ngg_memoryless_cover_export_stage,
    ngg_mesh_base_vertex, ngg_mesh_dispatch_required, ngg_mesh_draw_supported,
    ngg_mesh_export_stage, ngg_mesh_stage_for_draw, noop_placeholder_cover_draw,
    parse_diagnostic_hex_address, parse_presentation_resolution, pin_guest_memory_write,
    pixel_input_controls, prepare_indexed_ngg_indices, prepare_indexed_primitive,
    prepare_ngg_meshlets, prepare_nonindexed_ngg_indices, prepare_sampled_texture_upload,
    presentation_drawable_size, presentation_viewport, procedural_rectangle_export_stage,
    procedural_rectangle_stage_for_draw, process_memory_statistics, process_resident_bytes,
    raw_video_out_layout_is_linear, raw_video_out_texture_descriptor, read_draw_indices,
    read_texture_array_linear, read_texture_linear, rearm_color_metadata_clear,
    rearm_depth_metadata_clear, rectangle_coordinate_parameter_mask,
    rectangle_draw_has_incomplete_interface, rematerialisation_blit_source_origin_y,
    render_resolve_candidate_matches, render_target_cmask_address, render_target_fast_clear_color,
    resident_texture_activity_stats, resident_texture_backing_overlap_stats,
    resident_texture_eviction_plan, resident_texture_ownership_stats, sampled_alias_descriptor,
    sampled_texture_cache_accepts, sampled_texture_cache_needs_reload,
    sampled_texture_cache_requires_eviction, sampled_texture_key, sampled_texture_raw_range,
    sampler_lod_bias, sampler_uses_normalized_coordinates, shader_scalar_metadata,
    shader_scalar_metadata_for_registers, should_record_diagnostic_target_writer,
    snapshot_guest_texture_linear, snapshot_guest_texture_raw, snapshot_watched_guest_texture_raw,
    speculative_pipeline_compile_in_flight_capacity, storage_image_requires_mip_hierarchy,
    tiled_block_dimensions, truncated_memoryless_ngg_rectangle_cover, unsigned_float_to_f32,
    upload_image_textures, validate_compute_launch_for_metal, watch_guest_texture,
    wave64_split_blockers, write_back_pending_globals, write_guest_texture_linear,
    write_watched_guest_texture_linear,
};
mod completion;
mod decode;
mod present;
mod readback;
mod render_pass;
mod shader;
mod submission;
mod texture;
mod tiling;

#[allow(clippy::too_many_arguments)]
fn append_color_target_registers(
    registers: &mut Vec<GpuRegister>,
    slot: u32,
    address: u64,
    width: u32,
    height: u32,
    data_format: u32,
    number_type: u32,
    tile_mode: u32,
) {
    let primary = slot * CB_COLOR_REGISTER_STRIDE;
    registers.extend([
        GpuRegister {
            offset: CB_COLOR0_BASE + primary,
            value: u32::try_from((address >> 8) & 0xffff_ffff).expect("base fits u32"),
        },
        GpuRegister {
            offset: CB_COLOR0_VIEW + primary,
            value: 0,
        },
        GpuRegister {
            offset: CB_COLOR0_INFO + primary,
            value: (data_format << 2) | (number_type << 8),
        },
        GpuRegister {
            offset: CB_COLOR0_ATTRIB + primary,
            value: 0,
        },
        GpuRegister {
            offset: CB_COLOR0_CMASK + primary,
            value: u32::try_from(((address + 0x1000_0000) >> 8) & 0xffff_ffff)
                .expect("CMASK base fits u32"),
        },
        GpuRegister {
            offset: CB_COLOR0_BASE_EXT + slot,
            value: u32::try_from((address >> 40) & 0xff).expect("base extension fits u32"),
        },
        GpuRegister {
            offset: CB_COLOR0_ATTRIB2 + slot,
            value: ((width - 1) << 14) | (height - 1),
        },
        GpuRegister {
            offset: CB_COLOR0_ATTRIB3 + slot,
            value: (tile_mode << 14) | (u32::from(height > 1) << 24),
        },
    ]);
}

/// A minimal graphics pipeline key, used where only its identity as a block reason matters.
fn cold_graphics_pipeline_key() -> GraphicsPipelineKey {
    GraphicsPipelineKey {
        export_shader_address: 0x1000,
        pixel_shader_address: 0x2000,
        export_program_identity: ProgramIdentity(std::sync::Arc::new(remu_shader::Program::new(
            0x1000,
            Vec::new(),
        ))),
        pixel_program_identity: ProgramIdentity(std::sync::Arc::new(remu_shader::Program::new(
            0x2000,
            Vec::new(),
        ))),
        export_resources: GraphicsShaderResourceKey {
            initial_scalar_count: 0,
            global_bindings: Vec::new(),
            image_bindings: Vec::new(),
            vertex_inputs: Vec::new(),
        },
        pixel_resources: GraphicsShaderResourceKey {
            initial_scalar_count: 0,
            global_bindings: Vec::new(),
            image_bindings: Vec::new(),
            vertex_inputs: Vec::new(),
        },
        export_allocated_vector_register_count: 256,
        pixel_allocated_vector_register_count: 256,
        vertex_output_control: 0,
        pixel_input_controls: std::array::from_fn(|index| {
            u32::try_from(index).expect("PS input index fits u32")
        }),
        pixel_depth_export: false,
        ngg_subgroup_limits: None,
        ngg_registers: None,
        diagnostics: GraphicsDiagnosticPipelineState::default(),
        targets: Vec::new(),
        depth: None,
        sample_count: 1,
        render_target_array_length: 1,
        uses_mesh_stage: false,
        uses_procedural_rectangle_stage: false,
        primitive_topology_class: metal::MTLPrimitiveTopologyClass::Triangle,
    }
}

fn indexed_draw(address: u64, index_size: u32, index_count: u32) -> IndexedDraw {
    IndexedDraw {
        sequence: 1,
        indexed: true,
        indirect_arguments_address: None,
        indirect_count_address: None,
        indirect_draw_index: 0,
        instance_count: 1,
        index_buffer_address: address,
        index_resource_address: 0,
        index_buffer_count: index_count,
        index_size,
        index_offset: 0,
        base_vertex: 0,
        index_count,
        index_snapshot: None,
        draw_flags: 0,
        primitive_type: 4,
        export_shader_address: 0,
        pixel_shader_address: 0,
        export_shader_prefix: [0; 4],
        pixel_shader_prefix: [0; 4],
        export_shader: None,
        pixel_shader: None,
        export_shader_evaluation: None,
        pixel_shader_evaluation: None,
        color_target_address: 0,
        scanout_target: None,
        context_registers: Vec::new(),
        shader_registers: Vec::new(),
        user_config_registers: Vec::new(),
    }
}

fn append_depth_target_registers(
    draw: &mut IndexedDraw,
    address: u64,
    width: u32,
    height: u32,
    z_info: u32,
    depth_control: u32,
) {
    let base = u32::try_from((address >> 8) & 0xffff_ffff).expect("depth base fits u32");
    let base_hi = u32::try_from((address >> 40) & 0xff).expect("depth base high fits u32");
    draw.context_registers.extend([
        GpuRegister {
            offset: DB_DEPTH_SIZE_XY,
            value: (width - 1) | ((height - 1) << 16),
        },
        GpuRegister {
            offset: DB_Z_INFO,
            value: z_info,
        },
        GpuRegister {
            offset: DB_Z_READ_BASE,
            value: base,
        },
        GpuRegister {
            offset: DB_Z_WRITE_BASE,
            value: base,
        },
        GpuRegister {
            offset: DB_Z_READ_BASE_HI,
            value: base_hi,
        },
        GpuRegister {
            offset: DB_Z_WRITE_BASE_HI,
            value: base_hi,
        },
        GpuRegister {
            offset: DB_DEPTH_CONTROL,
            value: depth_control,
        },
    ]);
    draw.context_registers
        .sort_unstable_by_key(|register| register.offset);
}

fn compute_launch() -> ComputeLaunch {
    ComputeLaunch {
        grid: ComputeGrid::Threadgroups([8, 4, 1]),
        group_count: [8, 4, 1],
        base_group: [0; 3],
        threads_per_group: [32, 2, 1],
        wave_lane_count: 32,
        lds_size_bytes: 16 * 1024,
        scratch_size_bytes: 0,
        initiator: 0x8045,
    }
}
