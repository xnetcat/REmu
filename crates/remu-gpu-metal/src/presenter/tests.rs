use metal::{
    CompileOptions, Device, MTLBlendFactor, MTLBlendOperation, MTLClearColor, MTLCompareFunction,
    MTLCullMode, MTLLoadAction, MTLPixelFormat, MTLPrimitiveTopologyClass, MTLPrimitiveType,
    MTLResourceOptions, MTLSize, MTLStorageMode, MTLStoreAction, MTLTextureType, MTLTextureUsage,
    MTLTriangleFillMode, MTLWinding, RenderPassDescriptor, RenderPipelineDescriptor,
    TextureDescriptor,
};
use remu_gpu::{
    ComputeGrid, ComputeLaunch, GpuColorTarget, GpuEvent, GpuHostImage, GpuMemoryAcquire,
    GpuMemoryRelease, GpuMemoryReleaseValue, GpuMemoryWait, GpuMemoryWrite, GpuPipelineBarrier,
    GpuPipelineStage, GpuRegister, GpuScanoutTarget, IndexedDraw, gpu_event_channel,
};
use remu_shader::{
    GlobalMemoryBinding, ImageBinding, ImageControl, Instruction, Opcode, Operand, PixelOutputKind,
    Program, ShaderEncoding, ShaderEvaluation, decode_program,
};

use super::{
    CB_COLOR_REGISTER_STRIDE, CB_COLOR0_ATTRIB, CB_COLOR0_ATTRIB2, CB_COLOR0_ATTRIB3,
    CB_COLOR0_BASE, CB_COLOR0_BASE_EXT, CB_COLOR0_CLEAR_WORD0, CB_COLOR0_CLEAR_WORD1,
    CB_COLOR0_INFO, CB_COLOR0_VIEW, CB_TARGET_MASK, DB_DEPTH_CLEAR, DB_DEPTH_CONTROL,
    DB_DEPTH_SIZE_XY, DB_RENDER_CONTROL, DB_Z_INFO, DB_Z_READ_BASE, DB_Z_READ_BASE_HI,
    DB_Z_WRITE_BASE, DB_Z_WRITE_BASE_HI, DecodedDepthTarget, DecodedRasterState,
    DecodedRenderTarget, DisplayConfiguration, GLOBAL_UPLOAD_ALIGNMENT, GraphicsPipelineKey,
    GraphicsShaderResourceKey, GuestBatchFlushReason, GuestCommandBatch, GuestMemoryWaitKey,
    GuestMemoryWaitRecovery, GuestMemoryWaitStatus, GuestMemoryWaitTracker, GuestRenderPassKey,
    GuestTextureDescriptor, MetalPresenter, OpenGuestRenderPass, OrderedEventPoll,
    OrderedEventSource, OrderedGpuEventReceiver, PA_SU_SC_MODE_CNTL, PendingFlip,
    PendingGlobalWriteback, PendingGuestMemoryWrite, PendingTextureReadback,
    PresentationResolution, QueueBlockReason, SPI_PS_INPUT_CNTL_0, TextureReadbackLayout,
    VIDEO_OUT_DISPLAY_TILE_MODE, VIDEO_OUT_TILING_MODE_LINEAR, VIDEO_OUT_TILING_MODE_TILE,
    WriteWatchRefresh, apply_image_destination_select, apply_pending_guest_memory_writes,
    canonical_sampler_bindings, canonical_texture_bindings, common_render_target_array_length,
    comparison_depth32_upload, decode_depth_target, decode_fast_clear_color,
    decode_image_descriptor, decode_metal_blend_attachment, decode_mode27_mip_chain_from_raw,
    decode_raster_state, decode_render_target_format, decode_render_targets_from_registers,
    decode_scanout_target, decode_scissor, decode_viewport, effective_linear_pitch,
    encode_mode27_mip_chain_into_raw, fallback_image_descriptor, graphics_shader_resource_key,
    guest_physical_ranges_overlap, guest_shared_mapping_identity, guest_texture_byte_count,
    guest_texture_descriptors_overlap, guest_texture_descriptors_share_component_reordered_storage,
    guest_texture_descriptors_share_depth_sample_storage, guest_texture_descriptors_share_storage,
    guest_texture_raw_matches, half_float_to_srgb_unorm8, half_float_to_unorm8,
    headless_metal_value_is_enabled, image_component_class, image_destination_select_is_native,
    image_resource_descriptor_is_null, initial_depth_clear_for_target, initial_depth_clear_value,
    initial_render_target_clear_color, linear_unorm8_to_srgb_unorm8, metal_color_write_mask,
    metal_nonindexed_vertex_count, metal_primitive_topology_class, metal_primitive_type,
    missing_vertex_parameter_exports, mode5_within_block_byte_offset,
    mode9_within_block_byte_offset, mode24_within_block_byte_offset_for_format,
    mode27_block_dimensions, mode27_mip_chain_layout, mode27_source_byte_offset,
    mode27_within_block_byte_offset, mode27_within_block_byte_offset_for_format,
    native_texture_capture_to_bgra, new_red_blue_swizzled_texture_view,
    parse_diagnostic_hex_address, parse_presentation_resolution, pixel_input_controls,
    presentation_drawable_size, presentation_viewport, procedural_rectangle_export_stage,
    process_resident_bytes, raw_video_out_layout_is_linear, raw_video_out_texture_descriptor,
    read_draw_indices, read_texture_linear, render_resolve_candidate_matches,
    render_target_fast_clear_color, sampled_alias_descriptor, sampled_texture_cache_needs_reload,
    sampler_lod_bias, sampler_uses_normalized_coordinates, shader_scalar_metadata,
    shader_scalar_metadata_for_registers, should_record_diagnostic_target_writer,
    snapshot_guest_texture_linear, snapshot_guest_texture_raw, snapshot_watched_guest_texture_raw,
    storage_image_requires_mip_hierarchy, tiled_block_dimensions, unsigned_float_to_f32,
    validate_compute_launch_for_metal, watch_guest_texture, wave64_split_blockers,
    write_back_pending_globals, write_guest_texture_linear, write_watched_guest_texture_linear,
};
use remu_memory::CpuWriteWatchRetirement;

#[test]
fn presenter_shows_and_hides_a_host_splash_without_guest_video_out() {
    let (sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let image = GpuHostImage::new(1, 1, vec![10, 20, 30, 255]).expect("complete host image");

    sender
        .send(GpuEvent::ShowHostSplash(image))
        .expect("show event reaches presenter");
    assert_eq!(presenter.receive_events(), super::PresenterFlow::Continue);
    let splash = presenter.host_splash.as_ref().expect("splash texture");
    assert_eq!((splash.width, splash.height), (1, 1));
    assert_eq!(
        splash.texture.pixel_format(),
        MTLPixelFormat::RGBA8Unorm_sRGB
    );

    sender
        .send(GpuEvent::HideHostSplash)
        .expect("hide event reaches presenter");
    assert_eq!(presenter.receive_events(), super::PresenterFlow::Continue);
    assert!(presenter.host_splash.is_none());
    assert!(presenter.redraw_pending);
}

#[test]
fn diagnostic_shader_address_accepts_prefixed_and_plain_hex() {
    assert_eq!(
        parse_diagnostic_hex_address("8007e20000"),
        Some(0x8007_e20000)
    );
    assert_eq!(
        parse_diagnostic_hex_address("0x8007E20000"),
        Some(0x8007_e20000)
    );
    assert_eq!(parse_diagnostic_hex_address(""), None);
    assert_eq!(parse_diagnostic_hex_address("not-hex"), None);
}

#[test]
fn sampled_texture_cache_reloads_cpu_dirty_content_without_an_acquire() {
    assert!(sampled_texture_cache_needs_reload(
        WriteWatchRefresh::Active,
        true
    ));
    assert!(!sampled_texture_cache_needs_reload(
        WriteWatchRefresh::Active,
        false
    ));
    assert!(sampled_texture_cache_needs_reload(
        WriteWatchRefresh::Renewed { was_dirty: true },
        false
    ));
    assert!(sampled_texture_cache_needs_reload(
        WriteWatchRefresh::Replaced,
        false
    ));
    assert!(!sampled_texture_cache_needs_reload(
        WriteWatchRefresh::Renewed { was_dirty: false },
        true
    ));
    assert!(!sampled_texture_cache_needs_reload(
        WriteWatchRefresh::Reprotecting,
        true
    ));
}

#[test]
fn pre_pixel_shader_diagnostic_records_ordinary_render_targets() {
    assert!(should_record_diagnostic_target_writer(true, false, false));
    assert!(should_record_diagnostic_target_writer(true, true, false));
    assert!(should_record_diagnostic_target_writer(false, true, true));
    assert!(!should_record_diagnostic_target_writer(false, true, false));
    assert!(!should_record_diagnostic_target_writer(false, false, true));
}

#[test]
fn headless_metal_is_an_explicit_opt_in() {
    assert!(!headless_metal_value_is_enabled(None));
    assert!(!headless_metal_value_is_enabled(Some(
        std::ffi::OsStr::new("0")
    )));
    assert!(!headless_metal_value_is_enabled(Some(
        std::ffi::OsStr::new("true")
    )));
    assert!(headless_metal_value_is_enabled(Some(std::ffi::OsStr::new(
        "1"
    ))));
}

#[test]
fn raw_video_out_upload_maps_both_abi_layouts() {
    assert!(raw_video_out_layout_is_linear(1, 3_840, 3_840));
    assert!(!raw_video_out_layout_is_linear(0, 3_840, 3_840));
    assert!(!raw_video_out_layout_is_linear(1, 1_919, 1_920));
    assert!(!raw_video_out_layout_is_linear(27, 3_840, 3_840));

    let display = DisplayConfiguration {
        width: 3_840,
        height: 2_160,
        pitch: 3_840,
        pixel_format: 0x8000_0000_0000_0000,
        tiling_mode: VIDEO_OUT_TILING_MODE_TILE,
    };
    assert_eq!(
        raw_video_out_texture_descriptor(0xefc0_0000_00, display)
            .expect("display-tiled VideoOut allocation")
            .tile_mode,
        VIDEO_OUT_DISPLAY_TILE_MODE
    );
    assert_eq!(
        raw_video_out_texture_descriptor(
            0xefc0_0000_00,
            DisplayConfiguration {
                tiling_mode: VIDEO_OUT_TILING_MODE_LINEAR,
                ..display
            }
        )
        .expect("linear VideoOut allocation")
        .tile_mode,
        0
    );
    assert!(
        raw_video_out_texture_descriptor(
            0xefc0_0000_00,
            DisplayConfiguration {
                pitch: 1_919,
                ..display
            }
        )
        .is_none()
    );
}

#[test]
fn reads_live_process_resident_memory_for_window_telemetry() {
    assert!(process_resident_bytes().is_some_and(|bytes| bytes > 0));
}

#[test]
fn partial_pipeline_flushes_coalesce_without_reordering_submission_completion() {
    let (sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal queue")
        .clone();
    let texture_descriptor = TextureDescriptor::new();
    texture_descriptor.set_texture_type(MTLTextureType::D2);
    texture_descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    texture_descriptor.set_width(4);
    texture_descriptor.set_height(4);
    texture_descriptor.set_storage_mode(MTLStorageMode::Private);
    texture_descriptor.set_usage(MTLTextureUsage::RenderTarget);
    let texture = device.new_texture(&texture_descriptor);
    let render_pass = RenderPassDescriptor::new();
    let attachment = render_pass
        .color_attachments()
        .object_at(0)
        .expect("color attachment zero");
    attachment.set_texture(Some(&texture));
    attachment.set_load_action(MTLLoadAction::Clear);
    attachment.set_store_action(MTLStoreAction::Store);
    attachment.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));

    let command_buffer = queue.new_command_buffer().to_owned();
    let original_command_buffer = command_buffer.clone();
    let encoder = command_buffer
        .new_render_command_encoder(render_pass)
        .to_owned();
    presenter.open_guest_batch = Some(GuestCommandBatch {
        command_buffer,
        work: Vec::new(),
        global_upload_chunks: Vec::new(),
        global_writebacks: Vec::new(),
        guest_write_ranges: Vec::new(),
        memory_writes: Vec::new(),
        render_pass: Some(OpenGuestRenderPass {
            key: GuestRenderPassKey {
                targets: Vec::new(),
                depth: None,
                resolve_multisample_targets: false,
            },
            encoder,
        }),
        completion_sequences: vec![40],
    });
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 7,
            events: vec![
                GpuEvent::PipelineBarrier(GpuPipelineBarrier {
                    wait_for: GpuPipelineStage::PixelShader,
                }),
                GpuEvent::SubmissionBoundary(41),
                GpuEvent::PipelineBarrier(GpuPipelineBarrier {
                    wait_for: GpuPipelineStage::ComputeShader,
                }),
                GpuEvent::SubmissionBoundary(42),
                GpuEvent::PipelineBarrier(GpuPipelineBarrier {
                    wait_for: GpuPipelineStage::VertexShader,
                }),
            ],
        })
        .expect("queue partial flushes");

    assert!(matches!(
        presenter.receive_events(),
        super::PresenterFlow::Continue
    ));
    let open = presenter
        .open_guest_batch
        .as_ref()
        .expect("partial flush retains the open Metal batch");
    assert!(
        open.render_pass.is_some(),
        "an in-pass barrier must not spill and reload unchanged render attachments"
    );
    assert!(std::ptr::eq(
        open.command_buffer.as_ref(),
        original_command_buffer.as_ref()
    ));
    assert_eq!(open.completion_sequences, [40, 41, 42]);
    assert!(presenter.committed_guest_batches.is_empty());
    assert_eq!(presenter.statistics.submitted_guest_batches, 0);
    assert_eq!(presenter.statistics.coalesced_pipeline_barriers, 3);

    let mut release_label = 0_u32;
    let release_address =
        u64::try_from((&raw mut release_label) as usize).expect("test label address fits u64");
    presenter
        .execute_guest_memory_release(GpuMemoryRelease {
            address: release_address,
            value: GpuMemoryReleaseValue::Immediate32(1),
        })
        .expect("a real release remains in the coalesced batch");
    let open = presenter
        .open_guest_batch
        .as_ref()
        .expect("release does not commit the open batch");
    assert!(open.render_pass.is_some());
    assert!(matches!(
        open.memory_writes.as_slice(),
        [PendingGuestMemoryWrite::Release(GpuMemoryRelease {
            address,
            value: GpuMemoryReleaseValue::Immediate32(1),
        })] if *address == release_address
    ));
    assert_eq!(release_label, 0, "the label remains deferred");
    assert!(presenter.committed_guest_batches.is_empty());
    assert_eq!(presenter.statistics.submitted_guest_batches, 0);
    assert_eq!(presenter.statistics.batch_flushes_for_guest_memory_write, 0);
    assert_eq!(presenter.statistics.render_pass_ends_for_batch_flush, 0);
    assert_eq!(presenter.statistics.render_pass_ends_for_compute, 0);
    assert_eq!(presenter.statistics.render_pass_ends_for_target_change, 0);

    presenter
        .flush_guest_command_batch(false, GuestBatchFlushReason::EventLoop)
        .expect("event loop commits deferred work");
    assert!(presenter.open_guest_batch.is_none());
    assert_eq!(presenter.statistics.submitted_guest_batches, 1);
    assert_eq!(presenter.statistics.batch_flushes_from_event_loop, 1);
    assert_eq!(presenter.statistics.render_pass_ends_for_batch_flush, 1);
    let committed = presenter
        .committed_guest_batches
        .pop_front()
        .expect("event loop retains one committed batch");
    assert_eq!(committed.completion_sequences, [40, 41, 42]);
    assert!(matches!(
        committed.memory_writes.as_slice(),
        [PendingGuestMemoryWrite::Release(GpuMemoryRelease {
            address,
            ..
        })] if *address == release_address
    ));
    committed.command_buffer.wait_until_completed();
    assert_eq!(
        committed.command_buffer.status(),
        metal::MTLCommandBufferStatus::Completed
    );
    apply_pending_guest_memory_writes(&committed.memory_writes)
        .expect("completed release publishes its label");
    assert_eq!(release_label, 1);
}

#[test]
fn pending_label_writes_coalesce_until_a_blocked_wait_commits_them() {
    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal queue")
        .clone();
    presenter.open_guest_batch = Some(GuestCommandBatch {
        command_buffer: queue.new_command_buffer().to_owned(),
        work: Vec::new(),
        global_upload_chunks: Vec::new(),
        global_writebacks: Vec::new(),
        guest_write_ranges: Vec::new(),
        memory_writes: Vec::new(),
        render_pass: None,
        completion_sequences: Vec::new(),
    });

    let mut label = 0_u32;
    let address = u64::try_from((&raw mut label) as usize).expect("test label address fits u64");
    for value in [1, 2] {
        presenter
            .execute_guest_memory_release(GpuMemoryRelease {
                address,
                value: GpuMemoryReleaseValue::Immediate32(value),
            })
            .expect("label release coalesces");
    }
    assert_eq!(label, 0, "open-batch label writes remain deferred");
    assert_eq!(
        presenter.pending_guest_memory_value(address, false),
        Some(2)
    );
    assert_eq!(
        presenter
            .open_guest_batch
            .as_ref()
            .expect("label writes retain the open batch")
            .memory_writes
            .len(),
        2
    );
    assert!(presenter.committed_guest_batches.is_empty());
    assert_eq!(presenter.statistics.submitted_guest_batches, 0);

    let satisfied = GpuMemoryWait {
        address,
        reference: 2,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    assert_eq!(
        presenter
            .execute_guest_memory_wait(OrderedEventSource::GuestQueue(77), None, satisfied)
            .expect("latest deferred value satisfies the ordered wait"),
        GuestMemoryWaitStatus::Satisfied
    );
    assert!(presenter.open_guest_batch.is_some());
    assert_eq!(presenter.statistics.submitted_guest_batches, 0);

    let blocked = GpuMemoryWait {
        reference: 3,
        ..satisfied
    };
    assert_eq!(
        presenter
            .execute_guest_memory_wait(OrderedEventSource::GuestQueue(77), None, blocked)
            .expect("nonmatching deferred value blocks"),
        GuestMemoryWaitStatus::Blocked
    );
    assert!(presenter.open_guest_batch.is_none());
    assert_eq!(presenter.committed_guest_batches.len(), 1);
    assert_eq!(presenter.statistics.submitted_guest_batches, 1);
    assert_eq!(presenter.statistics.batch_flushes_for_blocked_wait, 1);
    assert_eq!(label, 0, "blocked wait commits before CPU publication");
    assert_eq!(
        presenter.pending_guest_memory_value(address, false),
        Some(2)
    );

    presenter
        .wait_for_guest_command_batches()
        .expect("committed label batch completes");
    assert!(presenter.committed_guest_batches.is_empty());
    assert_eq!(label, 2, "deferred writes publish in exact packet order");
}

#[test]
fn terminal_gpu_failure_still_publishes_release_and_latches_ordered_wait() {
    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let mut label = 0_u32;
    let address = u64::try_from((&raw mut label) as usize).expect("test label address fits u64");
    let wait = GpuMemoryWait {
        address,
        reference: 7,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    let key = GuestMemoryWaitKey::new(OrderedEventSource::GuestQueue(19), wait);
    assert_eq!(
        presenter.guest_memory_waits.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::None
    );
    let pending = [PendingGuestMemoryWrite::Release(GpuMemoryRelease {
        address,
        value: GpuMemoryReleaseValue::Immediate32(7),
    })];

    let error = presenter
        .publish_terminal_guest_memory_writes(
            &pending,
            Err("Metal render pipeline rejected the ordered draw".to_owned()),
        )
        .expect_err("the original GPU failure must remain visible");

    assert_eq!(
        error, "Metal render pipeline rejected the ordered draw",
        "publishing the completion label must not erase the primary failure"
    );
    assert_eq!(
        label, 7,
        "terminal failure must not strand its following RELEASE_MEM"
    );
    assert_eq!(
        presenter.guest_memory_waits.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::ProducerLatched(7),
        "the ordered producer must wake a suspended WAIT_REG_MEM"
    );
}

#[test]
fn synchronous_failure_drains_later_batches_and_their_completion_labels() {
    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal queue")
        .clone();
    let mut completion_label = 0_u32;
    let completion_address =
        u64::try_from((&raw mut completion_label) as usize).expect("test label address fits u64");
    for memory_writes in [
        vec![PendingGuestMemoryWrite::Release(GpuMemoryRelease {
            address: 2,
            value: GpuMemoryReleaseValue::Immediate32(1),
        })],
        vec![PendingGuestMemoryWrite::Release(GpuMemoryRelease {
            address: completion_address,
            value: GpuMemoryReleaseValue::Immediate32(9),
        })],
    ] {
        let command_buffer = queue.new_command_buffer().to_owned();
        command_buffer.commit();
        presenter
            .committed_guest_batches
            .push_back(GuestCommandBatch {
                command_buffer,
                work: Vec::new(),
                global_upload_chunks: Vec::new(),
                global_writebacks: Vec::new(),
                guest_write_ranges: Vec::new(),
                memory_writes,
                render_pass: None,
                completion_sequences: Vec::new(),
            });
    }

    let error = presenter
        .wait_for_guest_command_batches()
        .expect_err("the invalid first release must remain an error");

    assert!(error.contains("ordered guest memory publication"));
    assert!(error.contains("unaligned"));
    assert!(
        presenter.committed_guest_batches.is_empty(),
        "a failed terminal batch must not abandon later committed tickets"
    );
    assert_eq!(
        completion_label, 9,
        "the later batch's ordered completion label must still publish"
    );
}

#[test]
fn guest_batch_global_uploads_are_aligned_and_write_back_from_their_slice() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let mut batch = GuestCommandBatch {
        command_buffer: queue.new_command_buffer().to_owned(),
        work: Vec::new(),
        global_upload_chunks: Vec::new(),
        global_writebacks: Vec::new(),
        guest_write_ranges: Vec::new(),
        memory_writes: Vec::new(),
        render_pass: None,
        completion_sequences: Vec::new(),
    };
    let (first_buffer, first_offset) = batch
        .upload_global_bytes(&device, &[1, 2, 3])
        .expect("first upload");
    let (second_buffer, second_offset) = batch
        .upload_global_bytes(&device, &[9, 8, 7, 6])
        .expect("second upload");
    assert_eq!(first_buffer.contents(), second_buffer.contents());
    assert_eq!(first_offset, 0);
    assert_eq!(second_offset, GLOBAL_UPLOAD_ALIGNMENT as u64);

    let mut destination = [0_u8; 4];
    let guest_address =
        u64::try_from(destination.as_mut_ptr() as usize).expect("host pointer fits u64");
    write_back_pending_globals(&[PendingGlobalWriteback {
        buffer: second_buffer,
        offset: second_offset,
        guest_address,
        byte_count: destination.len(),
    }])
    .expect("sliced global writeback");
    assert_eq!(destination, [9, 8, 7, 6]);
}

#[test]
fn decodes_force_unnormalized_sampler_coordinates() {
    assert!(sampler_uses_normalized_coordinates(&[0, 0, 0, 0]));
    assert!(!sampler_uses_normalized_coordinates(&[1 << 15, 0, 0, 0,]));
}

#[test]
fn decodes_signed_sampler_lod_bias() {
    assert_eq!(sampler_lod_bias(&[0, 0, 0x03bf, 0]), 959.0 / 256.0);
    assert_eq!(sampler_lod_bias(&[0, 0, 0x3fff, 0]), -1.0 / 256.0);
    assert_eq!(sampler_lod_bias(&[0, 0, 0x2000, 0]), -32.0);
}

#[test]
fn pixel_input_controls_default_to_identity_and_preserve_programmed_remaps() {
    assert_eq!(
        pixel_input_controls(&[]),
        std::array::from_fn(|index| u32::try_from(index).expect("index fits u32"))
    );
    let registers = [
        GpuRegister {
            offset: SPI_PS_INPUT_CNTL_0,
            value: 4,
        },
        GpuRegister {
            offset: SPI_PS_INPUT_CNTL_0 + 1,
            value: 0x220,
        },
    ];
    let controls = pixel_input_controls(&registers);
    assert_eq!(controls[0], 4);
    assert_eq!(controls[1], 0x220);
    assert_eq!(controls[2], 2);
}

#[test]
fn rejects_only_non_default_pixel_inputs_without_vertex_exports() {
    let mut controls =
        std::array::from_fn(|index| u32::try_from(index).expect("input-control index fits u32"));
    controls[0] = 0x320;
    controls[1] = 3;

    assert_eq!(
        missing_vertex_parameter_exports(1 << 3, 0b11, &controls),
        Vec::new(),
        "logical input zero uses architectural default 3 and input one maps to PARAM3"
    );
    assert_eq!(
        missing_vertex_parameter_exports(0, 0b11, &controls),
        vec![(1, 3, 3)],
        "a non-default PARAM3 input cannot be supplied by a position-only stage"
    );
}

#[test]
fn recognizes_the_architectural_procedural_rectangle_export_stage() {
    let words = [
        0xbfa0_0001,
        0x7e00_0000,
        0x7e00_0000,
        0x7e00_0000,
        0x93eb_ff03,
        0x0008_0008,
        0x8f6a_8c6b,
        0x8700_ff03,
        0x0000_00ff,
        0x887c_6a00,
        0xbf90_0009,
        0x81ea_6bc0,
        0x90fe_6ac1,
        0xf800_0941,
        0x0000_0000,
        0x81ea_00c0,
        0xbf8c_ff0f,
        0x90fe_6ac1,
        0x3604_0a81,
        0x2c06_0a81,
        0x7e00_0280,
        0x7e02_02f2,
        0xd746_0002,
        0x0305_0302,
        0xd746_0003,
        0x0305_0303,
        0x7e04_0b02,
        0x7e06_0b03,
        0xf800_08cf,
        0x0100_0302,
        0xbf81_0000,
    ];
    let program = decode_program(0x1000, &words).expect("observed AGC rectangle stage decodes");

    assert!(procedural_rectangle_export_stage(&program));
}

#[test]
fn comparison_sampling_converts_srgb_color_red_to_linear_depth() {
    let upload = comparison_depth32_upload(
        MTLPixelFormat::RGBA8Unorm_sRGB,
        &[0, 1, 2, 255, 128, 3, 4, 255, 255, 5, 6, 255],
    )
    .expect("sRGB color texture is comparison-compatible")
    .expect("sRGB color texture needs a depth upload");
    let depths = upload
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("one float")))
        .collect::<Vec<_>>();
    assert_eq!(depths[0], 0.0);
    assert!((depths[1] - 0.215_86).abs() < 0.000_1);
    assert_eq!(depths[2], 1.0);
}

#[test]
fn ordered_guest_label_writes_publish_only_the_declared_values() {
    let mut release_value = 0_u64;
    let mut words = [0_u32; 3];
    let mut fill_words = [0_u32; 3];
    let release = PendingGuestMemoryWrite::Release(GpuMemoryRelease {
        address: std::ptr::addr_of_mut!(release_value) as u64,
        value: GpuMemoryReleaseValue::Immediate64(0x1122_3344_5566_7788),
    });
    let write = PendingGuestMemoryWrite::Write(GpuMemoryWrite {
        address: words.as_mut_ptr() as u64,
        values: vec![3, 5, 8],
        increment_address: true,
    });
    let fill = PendingGuestMemoryWrite::Fill {
        address: fill_words.as_mut_ptr() as u64,
        value: 0xaabb_ccdd,
        dword_count: 3,
    };
    assert_eq!(
        release.value_at(std::ptr::addr_of!(release_value) as u64, true),
        Some(0x1122_3344_5566_7788)
    );
    assert_eq!(write.value_at(words.as_ptr() as u64 + 8, false), Some(8));
    assert_eq!(
        fill.value_at(fill_words.as_ptr() as u64 + 4, false),
        Some(0xaabb_ccdd)
    );
    assert_eq!(
        fill.value_at(fill_words.as_ptr() as u64, true),
        Some(0xaabb_ccdd_aabb_ccdd)
    );
    apply_pending_guest_memory_writes(&[release, write, fill]).expect("ordered guest writes");
    assert_eq!(release_value, 0x1122_3344_5566_7788);
    assert_eq!(words, [3, 5, 8]);
    assert_eq!(fill_words, [0xaabb_ccdd; 3]);
}

#[test]
fn graphics_shader_cache_key_keeps_dynamic_global_addresses_out_of_pipeline_identity() {
    let evaluation = |base_address| ShaderEvaluation {
        initial_scalar_registers: Vec::new().into(),
        scalar_registers: Vec::new().into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: Vec::new(),
        global_memory_bindings: vec![GlobalMemoryBinding {
            scalar_address: 4,
            base_address,
            instruction_pcs: vec![0x18c].into(),
            byte_count: 64,
            data: vec![0; 64],
            writable: false,
            write_back_to_guest: true,
        }],
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    };

    assert_eq!(
        graphics_shader_resource_key(&evaluation(0x7009_0605_ac), false),
        graphics_shader_resource_key(&evaluation(0x7009_1600_6c), false),
    );
    assert_eq!(
        graphics_shader_resource_key(&evaluation(0x7009_0605_ac), true),
        graphics_shader_resource_key(&evaluation(0x7009_1600_6c), true),
    );
}

#[test]
fn runtime_upload_representatives_follow_canonical_texture_and_sampler_slots() {
    let control = ImageControl {
        dmask: 0xf,
        vector_address: 0,
        address_registers: vec![0, 1],
        vector_data: 4,
        scalar_resource: 0,
        scalar_sampler: 8,
        dimension: 1,
        is_array: false,
        glc: false,
        slc: false,
        a16: false,
        d16: false,
    };
    let binding = |pc, opcode, descriptor: Vec<u32>| ImageBinding {
        pc,
        opcode: remu_shader::Opcode::Known(opcode),
        control: control.clone(),
        resource_descriptor: descriptor.into(),
        sampler_descriptor: if opcode.starts_with("ImageSample") {
            vec![7, 8, 9, 10].into()
        } else {
            Vec::new().into()
        },
        mip_level: None,
    };
    let descriptor_a = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let descriptor_b = vec![11, 12, 13, 14, 15, 16, 17, 18];
    let evaluation = ShaderEvaluation {
        initial_scalar_registers: Vec::new().into(),
        scalar_registers: Vec::new().into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: vec![
            binding(0, "ImageSample", descriptor_a.clone()),
            binding(8, "ImageSampleLz", descriptor_a.clone()),
            binding(16, "ImageLoad", descriptor_a),
            binding(24, "ImageSample", descriptor_b),
        ],
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    };

    assert_eq!(
        canonical_texture_bindings(&evaluation)
            .iter()
            .map(|binding| binding.pc)
            .collect::<Vec<_>>(),
        [0, 16, 24]
    );
    assert_eq!(
        canonical_sampler_bindings(&evaluation)
            .iter()
            .map(|binding| binding.pc)
            .collect::<Vec<_>>(),
        [0]
    );
}

#[test]
fn graphics_shader_cache_key_normalizes_dynamic_image_formats_and_swizzles() {
    let evaluation = |unified_format: u32, destination_select: u32| ShaderEvaluation {
        initial_scalar_registers: vec![0; 16].into(),
        scalar_registers: vec![0; 16].into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: vec![ImageBinding {
            pc: 0x20,
            opcode: remu_shader::Opcode::Known("ImageSample"),
            control: ImageControl {
                dmask: 0xf,
                vector_address: 0,
                address_registers: vec![0, 1],
                vector_data: 4,
                scalar_resource: 0,
                scalar_sampler: 8,
                dimension: 1,
                is_array: false,
                glc: false,
                slc: false,
                a16: false,
                d16: false,
            },
            resource_descriptor: vec![0, unified_format << 20, 0, destination_select].into(),
            sampler_descriptor: vec![0; 4].into(),
            mip_level: None,
        }],
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    };

    // Both unified formats are float-class interfaces; pixel format and component selection
    // are supplied by the texture and live resource SGPRs at execution time.
    let first = evaluation(56, 0x924); // RGBA8 unorm
    let second = evaluation(58, 0xfac); // RGBA8 unsigned-scaled
    assert_eq!(image_component_class(&first.image_bindings[0]), 0);
    assert_eq!(image_component_class(&second.image_bindings[0]), 0);
    assert_eq!(
        graphics_shader_resource_key(&first, false),
        graphics_shader_resource_key(&second, false),
    );
}

#[test]
fn appends_dynamic_global_base_and_size_metadata_after_guest_scalars() {
    let evaluation = ShaderEvaluation {
        initial_scalar_registers: vec![10, 20].into(),
        scalar_registers: vec![10, 20].into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: Vec::new(),
        global_memory_bindings: vec![GlobalMemoryBinding {
            scalar_address: 4,
            base_address: 0x0070_1234_5678,
            instruction_pcs: vec![0x18c].into(),
            byte_count: 96,
            data: vec![0; 96],
            writable: true,
            write_back_to_guest: true,
        }],
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    };

    assert_eq!(
        shader_scalar_metadata(&evaluation).expect("metadata fits"),
        [10, 20, 0x1234_5678, 0x70, 96, 1]
    );
    assert_eq!(
        shader_scalar_metadata_for_registers(
            &evaluation,
            &[
                GpuRegister {
                    offset: 0x240,
                    value: 30,
                },
                GpuRegister {
                    offset: 0x241,
                    value: 40,
                },
            ],
            0x240,
            0,
        )
        .expect("live scalar metadata fits"),
        [30, 40, 0x1234_5678, 0x70, 96, 1]
    );
}

#[test]
fn live_scalar_metadata_matches_sorted_register_snapshot_values() {
    let evaluation = ShaderEvaluation {
        initial_scalar_registers: vec![u32::MAX; 64].into(),
        scalar_registers: vec![u32::MAX; 64].into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: Vec::new(),
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    };
    let registers = (0..256_u32)
        .filter(|offset| *offset != 0x20)
        .map(|offset| GpuRegister {
            offset: 0x200 + offset,
            value: 0xa500_0000 | offset,
        })
        .collect::<Vec<_>>();
    let expected = std::array::from_fn::<_, 64, _>(|scalar| {
        let offset = 0x200 + u32::try_from(scalar).expect("scalar index fits u32");
        registers
            .iter()
            .find(|register| register.offset == offset)
            .map_or(0, |register| register.value)
    });

    assert_eq!(
        shader_scalar_metadata_for_registers(&evaluation, &registers, 0x200, 0)
            .expect("live scalar metadata fits"),
        expected
    );
    assert_eq!(expected[0], 0xa500_0000);
    assert_eq!(expected[0x20], 0);
    assert_eq!(expected[63], 0xa500_003f);
}

#[test]
fn reuses_immutable_sampler_argument_buffers() {
    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let device = presenter
        .device
        .as_ref()
        .expect("presenter retains its Metal device")
        .clone();
    let source = r"
#include <metal_stdlib>
using namespace metal;
struct RemuSamplers {
    sampler s0 [[id(0)]];
};
kernel void remu_sampler_cache_test(
    constant RemuSamplers& samplers [[buffer(0)]],
    device float4* output [[buffer(1)]],
    texture2d<float> texture [[texture(0)]],
    uint gid [[thread_position_in_grid]]) {
    output[gid] = texture.sample(samplers.s0, float2(0.5f));
}
";
    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .expect("sampler argument-buffer MSL compiles");
    let function = library
        .get_function("remu_sampler_cache_test", None)
        .expect("sampler cache function exists");
    let evaluation = ShaderEvaluation {
        initial_scalar_registers: Vec::new().into(),
        scalar_registers: Vec::new().into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: vec![ImageBinding {
            pc: 0x100,
            opcode: Opcode::Known("ImageSample"),
            control: ImageControl {
                dmask: 0xf,
                vector_address: 0,
                address_registers: vec![0, 1],
                vector_data: 4,
                scalar_resource: 0,
                scalar_sampler: 8,
                dimension: 1,
                is_array: false,
                glc: false,
                slc: false,
                a16: false,
                d16: false,
            },
            resource_descriptor: vec![0; 8].into(),
            sampler_descriptor: vec![0; 4].into(),
            mip_level: None,
        }],
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    };

    let (first_buffer, first_samplers) = presenter
        .cached_sampler_argument_buffer(&device, &function, Some(0), &evaluation)
        .expect("first sampler argument buffer is created");
    let (second_buffer, second_samplers) = presenter
        .cached_sampler_argument_buffer(&device, &function, Some(0), &evaluation)
        .expect("identical sampler argument buffer is reused");

    assert_eq!(presenter.sampler_argument_cache.len(), 1);
    assert_eq!(first_samplers.len(), 1);
    assert_eq!(second_samplers.len(), 1);
    assert_eq!(
        first_buffer.expect("first buffer").contents(),
        second_buffer.expect("second buffer").contents()
    );
}

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

#[test]
fn parses_configurable_presentation_resolution() {
    assert_eq!(
        parse_presentation_resolution("window").expect("logical window mode"),
        PresentationResolution::LogicalWindow
    );
    assert_eq!(
        parse_presentation_resolution("native").expect("native window mode"),
        PresentationResolution::NativeWindow
    );
    assert_eq!(
        parse_presentation_resolution("1920x1080").expect("fixed presentation size"),
        PresentationResolution::Fixed(1_920, 1_080)
    );
    assert!(parse_presentation_resolution("0x720").is_err());
    assert!(parse_presentation_resolution("garbage").is_err());
}

#[test]
fn logical_presentation_avoids_retina_and_guest_4k_drawables() {
    assert_eq!(
        presentation_drawable_size(PresentationResolution::LogicalWindow, 2_560, 1_440, 2.0),
        (1_280, 720)
    );
    assert_eq!(
        presentation_drawable_size(
            PresentationResolution::Fixed(1_920, 1_080),
            2_560,
            1_440,
            2.0
        ),
        (1_920, 1_080)
    );
}

#[test]
fn presentation_viewport_preserves_guest_aspect_ratio() {
    let viewport = presentation_viewport(3_840, 2_160, 1_024, 768);
    assert_eq!(
        (
            viewport.originX,
            viewport.originY,
            viewport.width,
            viewport.height
        ),
        (0.0, 96.0, 1_024.0, 576.0)
    );
}

fn indexed_draw(address: u64, index_size: u32, index_count: u32) -> IndexedDraw {
    IndexedDraw {
        sequence: 1,
        indexed: true,
        instance_count: 1,
        index_buffer_address: address,
        index_resource_address: 0,
        index_buffer_count: index_count,
        index_size,
        index_offset: 0,
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

#[test]
fn decodes_observed_depth16_greater_equal_read_only_target() {
    let address = 0x0070_1ef9_0000;
    let mut draw = indexed_draw(0, 0, 0);
    append_depth_target_registers(&mut draw, address, 8, 8, 0xa000_0001, 0x62);

    let depth = decode_depth_target(&draw)
        .expect("valid depth state")
        .expect("enabled depth target");
    assert_eq!(depth.texture.address, address);
    assert_eq!((depth.texture.width, depth.texture.height), (8, 8));
    assert_eq!(depth.texture.pitch, 256);
    assert_eq!(depth.texture.tile_mode, 24);
    assert_eq!(depth.texture.pixel_format, MTLPixelFormat::Depth16Unorm);
    assert_eq!(depth.compare, MTLCompareFunction::GreaterEqual);
    assert!(!depth.write_enabled);
    assert_eq!(initial_depth_clear_for_target(depth), 0.0);
}

#[test]
fn depth_target_uses_fixed_mode24_layout_and_rejects_nonzero_tile_index() {
    let address = 0x0070_1412_0000;
    let mut draw = indexed_draw(0, 0, 0);
    append_depth_target_registers(&mut draw, address, 3_840, 2_160, 0xa000_0003, 0x66);

    let depth = decode_depth_target(&draw)
        .expect("fixed Prospero depth layout")
        .expect("enabled depth target");
    assert_eq!(depth.texture.tile_mode, 24);
    assert_eq!(depth.texture.pitch, 3_840);
    assert_eq!(
        guest_texture_byte_count(depth.texture),
        Some(3_840 * 2_176 * 4)
    );

    draw.context_registers
        .iter_mut()
        .find(|register| register.offset == DB_Z_INFO)
        .expect("depth info")
        .value |= 1 << 20;
    let error = decode_depth_target(&draw).expect_err("nonzero tile index is unsupported");
    assert!(error.contains("tile-mode index 1"), "{error}");
}

#[test]
fn native_depth_surface_matches_shader_r32_view_without_writeback() {
    let rendered = GuestTextureDescriptor {
        address: 0x0080_16c1_0000,
        width: 3_840,
        height: 2_160,
        pitch: 3_840,
        tile_mode: 24,
        pixel_format: MTLPixelFormat::Depth32Float,
        bytes_per_pixel: 4,
    };
    let sampled = GuestTextureDescriptor {
        pixel_format: MTLPixelFormat::R32Float,
        ..rendered
    };
    assert!(guest_texture_descriptors_share_depth_sample_storage(
        rendered, sampled, 0x204
    ));
    assert!(!guest_texture_descriptors_share_depth_sample_storage(
        rendered, sampled, 0xfac
    ));
}

#[test]
fn decodes_depth32_write_target_and_ignores_stencil_only_state() {
    let mut draw = indexed_draw(0, 0, 0);
    append_depth_target_registers(&mut draw, 0x0070_1412_0000, 1_920, 1_080, 0xa000_0003, 0x66);
    let depth = decode_depth_target(&draw)
        .expect("valid depth state")
        .expect("enabled depth target");
    assert_eq!(depth.texture.pixel_format, MTLPixelFormat::Depth32Float);
    assert_eq!(depth.compare, MTLCompareFunction::GreaterEqual);
    assert!(depth.write_enabled);
    assert_eq!(initial_depth_clear_for_target(depth), 1.0);

    draw.context_registers
        .iter_mut()
        .find(|register| register.offset == DB_DEPTH_CONTROL)
        .expect("depth control")
        .value = 0x771;
    assert_eq!(
        decode_depth_target(&draw).expect("stencil-only state is valid"),
        None
    );

    draw.context_registers
        .iter_mut()
        .find(|register| register.offset == DB_DEPTH_CONTROL)
        .expect("depth control")
        .value = 0x66;
    draw.context_registers
        .iter_mut()
        .find(|register| register.offset == DB_Z_INFO)
        .expect("depth info")
        .value = 0x8000_0000;
    assert_eq!(
        decode_depth_target(&draw).expect("format-zero depth state is valid"),
        None
    );
}

#[test]
fn decodes_explicit_guest_depth_clear_without_depth_test() {
    let mut draw = indexed_draw(0, 0, 0);
    append_depth_target_registers(&mut draw, 0x0070_1412_0000, 1_920, 1_080, 0xa000_0003, 0);
    draw.context_registers.extend([
        GpuRegister {
            offset: DB_RENDER_CONTROL,
            value: 1,
        },
        GpuRegister {
            offset: DB_DEPTH_CLEAR,
            value: 0.25_f32.to_bits(),
        },
    ]);
    draw.context_registers
        .sort_unstable_by_key(|register| register.offset);

    let depth = decode_depth_target(&draw)
        .expect("valid explicit clear state")
        .expect("clear-only operation retains its depth target");
    assert!(!depth.test_enabled);
    assert!(!depth.write_enabled);
    assert!(depth.clear_enabled);
    assert_eq!(depth.clear_depth, 0.25);
    assert_eq!(initial_depth_clear_for_target(depth), 0.25);
}

#[test]
fn rejects_non_finite_guest_depth_clear_value() {
    let mut draw = indexed_draw(0, 0, 0);
    append_depth_target_registers(&mut draw, 0x0070_1412_0000, 1_920, 1_080, 0xa000_0003, 0x2);
    draw.context_registers.push(GpuRegister {
        offset: DB_DEPTH_CLEAR,
        value: f32::NAN.to_bits(),
    });
    draw.context_registers
        .sort_unstable_by_key(|register| register.offset);

    let depth = decode_depth_target(&draw)
        .expect("invalid clear register falls back safely")
        .expect("depth test remains active");
    assert_eq!(depth.clear_depth, 1.0);
}

#[test]
fn initializes_depth_to_the_far_value_for_the_active_comparison() {
    assert_eq!(
        initial_depth_clear_value(MTLCompareFunction::LessEqual),
        1.0
    );
    assert_eq!(
        initial_depth_clear_value(MTLCompareFunction::GreaterEqual),
        0.0
    );
    assert_eq!(initial_depth_clear_value(MTLCompareFunction::Greater), 0.0);
    assert_eq!(initial_depth_clear_value(MTLCompareFunction::Always), 1.0);
}

#[test]
fn decodes_observed_clockwise_back_face_culling() {
    let mut draw = indexed_draw(0, 0, 0);
    draw.context_registers.push(GpuRegister {
        offset: PA_SU_SC_MODE_CNTL,
        value: 0x246,
    });

    assert_eq!(
        decode_raster_state(&draw).expect("observed raster state"),
        DecodedRasterState {
            cull_mode: MTLCullMode::Back,
            front_winding: MTLWinding::Clockwise,
            fill_mode: MTLTriangleFillMode::Fill,
        }
    );
}

#[test]
fn compensates_winding_when_normalizing_negative_viewport_height() {
    let mut draw = indexed_draw(0, 0, 0);
    draw.context_registers.extend([
        GpuRegister {
            offset: PA_SU_SC_MODE_CNTL,
            value: 0x246,
        },
        GpuRegister {
            offset: 0x10f,
            value: 960.0_f32.to_bits(),
        },
        GpuRegister {
            offset: 0x110,
            value: 960.0_f32.to_bits(),
        },
        GpuRegister {
            offset: 0x111,
            value: (-540.0_f32).to_bits(),
        },
        GpuRegister {
            offset: 0x112,
            value: 540.0_f32.to_bits(),
        },
    ]);
    draw.context_registers
        .sort_by_key(|register| register.offset);

    assert_eq!(
        decode_raster_state(&draw).expect("negative-height viewport raster state"),
        DecodedRasterState {
            cull_mode: MTLCullMode::Back,
            front_winding: MTLWinding::CounterClockwise,
            fill_mode: MTLTriangleFillMode::Fill,
        }
    );
}

#[test]
fn decodes_dual_polygon_line_mode_and_rejects_unrepresentable_states() {
    let mut draw = indexed_draw(0, 0, 0);
    draw.context_registers.push(GpuRegister {
        offset: PA_SU_SC_MODE_CNTL,
        value: (1 << 2) | (1 << 3) | (1 << 5) | (1 << 8),
    });
    assert_eq!(
        decode_raster_state(&draw)
            .expect("matching front/back line mode")
            .fill_mode,
        MTLTriangleFillMode::Lines
    );

    draw.context_registers[0].value |= 1 | 2;
    assert!(
        decode_raster_state(&draw)
            .expect_err("Metal cannot cull both orientations")
            .contains("simultaneous front- and back-face")
    );
}

#[test]
fn queue_local_compile_gate_runs_compute_without_overtaking_graphics_or_flip() {
    let (sender, receiver) = gpu_event_channel();
    let mut ordered = OrderedGpuEventReceiver::new(receiver);
    let key = GraphicsPipelineKey {
        export_shader_address: 0x1000,
        pixel_shader_address: 0x2000,
        export_program_identity: super::ProgramIdentity(std::sync::Arc::new(Program::new(
            0x1000,
            Vec::new(),
        ))),
        pixel_program_identity: super::ProgramIdentity(std::sync::Arc::new(Program::new(
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
        pixel_input_controls: std::array::from_fn(|index| {
            u32::try_from(index).expect("PS input index fits u32")
        }),
        targets: Vec::new(),
        depth: None,
        sample_count: 1,
        render_target_array_length: 1,
        primitive_topology_class: MTLPrimitiveTopologyClass::Triangle,
    };
    let mut draw = indexed_draw(0, 0, 0);
    draw.sequence = 17;
    let mut later_draw = indexed_draw(0, 0, 0);
    later_draw.sequence = 23;
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 0,
            events: vec![
                GpuEvent::IndexedDraw(Box::new(draw)),
                GpuEvent::PipelineBarrier(remu_gpu::GpuPipelineBarrier {
                    wait_for: remu_gpu::GpuPipelineStage::PixelShader,
                }),
                GpuEvent::MemoryAcquire(GpuMemoryAcquire {
                    engine: 0,
                    cb_db_control: 0,
                    base_address: 0x2000,
                    size_bytes: 0x1000,
                    poll_interval: 0,
                    gcr_control: 0,
                }),
                GpuEvent::IndexedDraw(Box::new(later_draw)),
                GpuEvent::SubmissionBoundary(1),
            ],
        })
        .expect("queue graphics submission");
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 33,
            events: vec![
                GpuEvent::MemoryRelease(GpuMemoryRelease {
                    address: 0x3000,
                    value: GpuMemoryReleaseValue::Immediate32(1),
                }),
                GpuEvent::SubmissionBoundary(2),
            ],
        })
        .expect("queue compute submission");
    sender
        .send(GpuEvent::Flip {
            address: 0x7000_0000,
            buffer_index: 1,
            flip_argument: 9,
            width: 1_280,
            height: 720,
            pitch: 1_280,
            pixel_format: 0x8000_0000_2200_0000,
            tiling_mode: 0,
        })
        .expect("queue later flip");

    let (graphics_source, blocked_draw) = match ordered.try_recv(|_| false) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::IndexedDraw(draw),
        } => {
            assert_eq!(draw.sequence, 17);
            (source, GpuEvent::IndexedDraw(draw))
        }
        other => panic!("expected first graphics draw, got {other:?}"),
    };
    ordered.block_event(
        graphics_source,
        blocked_draw,
        QueueBlockReason::GraphicsPipeline(key),
    );

    let compute_source = match ordered.try_recv(|_| false) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::MemoryRelease(release),
        } => {
            assert_eq!(release.address, 0x3000);
            source
        }
        other => panic!("expected independent compute release, got {other:?}"),
    };
    ordered.finish_event(compute_source);
    let compute_source = match ordered.try_recv(|_| false) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::SubmissionBoundary(2),
        } => source,
        other => panic!("expected compute boundary, got {other:?}"),
    };
    ordered.finish_event(compute_source);

    assert!(matches!(
        ordered.try_recv(|_| false),
        OrderedEventPoll::Blocked
    ));
    let prefetched = ordered.prefetch_draws(16, 16);
    assert_eq!(
        prefetched
            .iter()
            .map(|draw| draw.sequence)
            .collect::<Vec<_>>(),
        [17, 23]
    );

    match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::IndexedDraw(draw),
        } => {
            assert_eq!(draw.sequence, 17);
            ordered.finish_event(source);
        }
        other => panic!("expected the blocked draw exactly once, got {other:?}"),
    }
    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event:
                GpuEvent::PipelineBarrier(remu_gpu::GpuPipelineBarrier {
                    wait_for: remu_gpu::GpuPipelineStage::PixelShader,
                }),
        } => source,
        other => panic!("expected graphics barrier, got {other:?}"),
    };
    ordered.finish_event(source);
    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event:
                GpuEvent::MemoryAcquire(GpuMemoryAcquire {
                    base_address: 0x2000,
                    size_bytes: 0x1000,
                    ..
                }),
        } => source,
        other => panic!("expected graphics acquire, got {other:?}"),
    };
    ordered.finish_event(source);
    match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::IndexedDraw(draw),
        } => {
            assert_eq!(draw.sequence, 23);
            ordered.finish_event(source);
        }
        other => panic!("expected the prefetched draw in exact order, got {other:?}"),
    }
    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::SubmissionBoundary(1),
        } => source,
        other => panic!("expected graphics boundary, got {other:?}"),
    };
    ordered.finish_event(source);
    assert!(matches!(
        ordered.try_recv(|_| true),
        OrderedEventPoll::Event {
            event: GpuEvent::Flip {
                address: 0x7000_0000,
                ..
            },
            ..
        }
    ));
    assert!(matches!(
        ordered.try_recv(|_| true),
        OrderedEventPoll::Empty
    ));
}

#[test]
fn ordered_ingress_stops_at_the_decoded_event_backpressure_limit() {
    let (sender, receiver) = std::sync::mpsc::channel();
    for _ in
        0..(remu_gpu::GPU_PENDING_EVENT_CAPACITY / remu_gpu::GPU_EVENT_SUBMISSION_CHUNK_SIZE + 4)
    {
        sender
            .send(GpuEvent::QueueSubmission {
                queue_id: 0,
                events: vec![GpuEvent::Shutdown; remu_gpu::GPU_EVENT_SUBMISSION_CHUNK_SIZE],
            })
            .expect("queue decoded event chunk");
    }
    let mut ordered = OrderedGpuEventReceiver::new(receiver);

    ordered.ingest(usize::MAX);

    assert_eq!(
        ordered.pending_event_count(),
        remu_gpu::GPU_PENDING_EVENT_CAPACITY
    );
}

#[test]
fn suspended_memory_wait_keeps_same_queue_tail_behind_it() {
    let (sender, receiver) = gpu_event_channel();
    let mut ordered = OrderedGpuEventReceiver::new(receiver);
    let wait = GpuMemoryWait {
        address: 0x4000,
        reference: 1,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 0,
            events: vec![
                GpuEvent::MemoryWait(wait),
                GpuEvent::MemoryWrite(GpuMemoryWrite {
                    address: 0x5000,
                    values: vec![9],
                    increment_address: true,
                }),
                GpuEvent::SubmissionBoundary(1),
            ],
        })
        .expect("queue waiting graphics submission");
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 33,
            events: vec![
                GpuEvent::MemoryRelease(GpuMemoryRelease {
                    address: wait.address,
                    value: GpuMemoryReleaseValue::Immediate32(1),
                }),
                GpuEvent::SubmissionBoundary(2),
            ],
        })
        .expect("queue satisfying compute submission");

    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::MemoryWait(actual),
        } => {
            assert_eq!(actual, wait);
            source
        }
        other => panic!("expected graphics wait, got {other:?}"),
    };
    ordered.block_event(
        source,
        GpuEvent::MemoryWait(wait),
        QueueBlockReason::GuestMemoryWait,
    );

    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::MemoryRelease(release),
        } => {
            assert_eq!(release.address, wait.address);
            source
        }
        other => panic!("expected compute release, got {other:?}"),
    };
    ordered.finish_event(source);
    ordered.retry_guest_memory_waits();

    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::SubmissionBoundary(2),
        } => source,
        other => panic!("expected compute boundary, got {other:?}"),
    };
    ordered.finish_event(source);
    let source = match ordered.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::MemoryWait(actual),
        } => {
            assert_eq!(actual, wait);
            source
        }
        other => panic!("expected retried graphics wait, got {other:?}"),
    };
    ordered.finish_event(source);

    assert!(matches!(
        ordered.try_recv(|_| true),
        OrderedEventPoll::Event {
            event: GpuEvent::MemoryWrite(GpuMemoryWrite {
                address: 0x5000,
                ..
            }),
            ..
        }
    ));
}

#[test]
fn guest_wait_latches_a_real_producer_before_guest_memory_can_be_recycled() {
    let mut tracker = GuestMemoryWaitTracker::default();
    let wait = GpuMemoryWait {
        address: 0x7000_1000,
        reference: 1,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    let key = GuestMemoryWaitKey::new(OrderedEventSource::GuestQueue(32), wait);
    assert_eq!(
        tracker.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::None
    );

    tracker.record_produced(wait.address, false, 1);
    assert_eq!(
        tracker.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::ProducerLatched(1)
    );
    assert!(!tracker.active.contains_key(&key));
}

#[test]
fn guest_wait_never_reuses_a_producer_from_before_its_registration() {
    let wait = GpuMemoryWait {
        address: 0x7000_2000,
        reference: 1,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    let key = GuestMemoryWaitKey::new(OrderedEventSource::GuestQueue(0), wait);
    let mut tracker = GuestMemoryWaitTracker::default();
    tracker.record_produced(wait.address, false, 1);

    assert_eq!(
        tracker.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::None
    );
    // Guest allocators recycle label addresses. Wall-clock age cannot turn a producer from an
    // earlier allocation generation into satisfaction for this wait.
    assert_eq!(
        tracker.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::None
    );

    tracker.record_produced(wait.address, false, 1);
    assert_eq!(
        tracker.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::ProducerLatched(1)
    );
}

#[test]
fn guest_wait_does_not_latch_missing_or_nonmatching_producers() {
    let wait = GpuMemoryWait {
        address: 0x7000_2000,
        reference: 1,
        mask: u64::from(u32::MAX),
        compare_function: 3,
        is_64_bit: false,
    };
    let key = GuestMemoryWaitKey::new(OrderedEventSource::GuestQueue(0), wait);
    let mut missing = GuestMemoryWaitTracker::default();
    assert_eq!(
        missing.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::None
    );
    assert_eq!(
        missing.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::None
    );

    let mut wrong = GuestMemoryWaitTracker::default();
    assert_eq!(
        wrong.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::None
    );
    wrong.record_produced(wait.address, false, 2);
    assert_eq!(
        wrong.observe_unsatisfied(key),
        GuestMemoryWaitRecovery::None
    );
}

#[test]
fn texture_readback_layout_keeps_exact_packed_rows_and_aligned_staging_rows() {
    let layout = TextureReadbackLayout::new(3, 2, 4).expect("valid layout");
    assert_eq!(layout.packed_bytes_per_row, 12);
    assert_eq!(layout.staging_bytes_per_row, 256);
    assert_eq!(layout.staging_length, 512);
    assert_eq!(layout.packed_length(2).expect("packed length"), 24);
}

#[test]
fn capture_encodes_linear_storage_bytes_with_the_standard_srgb_transfer() {
    assert_eq!(linear_unorm8_to_srgb_unorm8(0), 0);
    assert_eq!(linear_unorm8_to_srgb_unorm8(1), 13);
    assert_eq!(linear_unorm8_to_srgb_unorm8(5), 38);
    assert_eq!(linear_unorm8_to_srgb_unorm8(10), 56);
    assert_eq!(linear_unorm8_to_srgb_unorm8(23), 85);
    assert_eq!(linear_unorm8_to_srgb_unorm8(128), 188);
    assert_eq!(linear_unorm8_to_srgb_unorm8(255), 255);
}

#[test]
fn capture_maps_half_float_scanout_channels_to_unorm8() {
    assert_eq!(half_float_to_unorm8(0x0000), 0);
    assert_eq!(half_float_to_unorm8(0x0001), 0);
    assert_eq!(half_float_to_unorm8(0x3800), 128);
    assert_eq!(half_float_to_unorm8(0x3c00), 255);
    assert_eq!(half_float_to_unorm8(0x4000), 255);
    assert_eq!(half_float_to_unorm8(0xbc00), 0);
    assert_eq!(half_float_to_unorm8(0x7c00), 255);
    assert_eq!(half_float_to_unorm8(0x7e00), 0);
    assert_eq!(half_float_to_srgb_unorm8(0x3800), 188);
}

#[test]
fn capture_decodes_packed_rg11b10_float_channels() {
    assert_eq!(unsigned_float_to_f32(0, 6), 0.0);
    assert_eq!(unsigned_float_to_f32(1, 6), 2.0_f32.powi(-20));
    assert_eq!(unsigned_float_to_f32(15 << 6, 6), 1.0);
    assert_eq!(unsigned_float_to_f32(14 << 6, 6), 0.5);
    assert_eq!(unsigned_float_to_f32(13 << 5, 5), 0.25);
    assert!(unsigned_float_to_f32((31 << 6) | 1, 6).is_nan());

    let packed = (15_u32 << 6) | ((14_u32 << 6) << 11) | ((13_u32 << 5) << 22);
    let descriptor = GuestTextureDescriptor {
        address: 0x1_0000,
        width: 1,
        height: 1,
        pitch: 1,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RG11B10Float,
        bytes_per_pixel: 4,
    };
    assert_eq!(
        native_texture_capture_to_bgra(&packed.to_le_bytes(), descriptor),
        Some(vec![137, 188, 255, 255]),
    );
}

#[test]
fn private_render_target_readback_uses_owned_staging_resources() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let descriptor = GuestTextureDescriptor {
        address: 0,
        width: 3,
        height: 2,
        pitch: 3,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::BGRA8Unorm,
        bytes_per_pixel: 4,
    };
    let metal_descriptor = TextureDescriptor::new();
    metal_descriptor.set_texture_type(MTLTextureType::D2);
    metal_descriptor.set_pixel_format(descriptor.pixel_format);
    metal_descriptor.set_width(u64::from(descriptor.width));
    metal_descriptor.set_height(u64::from(descriptor.height));
    metal_descriptor.set_storage_mode(MTLStorageMode::Private);
    metal_descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    let texture = device.new_texture(&metal_descriptor);

    let pass = RenderPassDescriptor::new();
    let attachment = pass
        .color_attachments()
        .object_at(0)
        .expect("color attachment zero");
    attachment.set_texture(Some(&texture));
    attachment.set_load_action(MTLLoadAction::Clear);
    attachment.set_store_action(MTLStoreAction::Store);
    attachment.set_clear_color(MTLClearColor::new(1.0, 0.0, 0.0, 1.0));
    let render = queue.new_command_buffer().to_owned();
    let encoder = render.new_render_command_encoder(pass);
    encoder.end_encoding();
    render.commit();

    let pending = PendingTextureReadback::encode(&device, &queue, &texture, descriptor)
        .expect("encode private readback");
    drop(texture);
    let bytes = pending.finish().expect("finish private readback");
    assert_eq!(bytes.len(), 24);
    assert!(bytes.chunks_exact(4).all(|pixel| pixel == [0, 0, 255, 255]));
}

#[test]
fn multisample_resolve_target_is_readable_after_ordered_resolve() {
    let device = Device::system_default().expect("Metal device");
    if !device.supports_texture_sample_count(4) {
        return;
    }
    let queue = device.new_command_queue();
    let descriptor = GuestTextureDescriptor {
        address: 0,
        width: 2,
        height: 2,
        pitch: 2,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let multisample_descriptor = TextureDescriptor::new();
    multisample_descriptor.set_texture_type(MTLTextureType::D2Multisample);
    multisample_descriptor.set_pixel_format(descriptor.pixel_format);
    multisample_descriptor.set_width(u64::from(descriptor.width));
    multisample_descriptor.set_height(u64::from(descriptor.height));
    multisample_descriptor.set_sample_count(4);
    multisample_descriptor.set_storage_mode(MTLStorageMode::Private);
    multisample_descriptor.set_usage(MTLTextureUsage::RenderTarget);
    let multisample = device.new_texture(&multisample_descriptor);
    let unresolved_error =
        PendingTextureReadback::encode(&device, &queue, &multisample, descriptor)
            .err()
            .expect("unresolved multisample readback must fail");
    assert!(unresolved_error.contains("single-sample resolve target"));

    let resolve_descriptor = TextureDescriptor::new();
    resolve_descriptor.set_texture_type(MTLTextureType::D2);
    resolve_descriptor.set_pixel_format(descriptor.pixel_format);
    resolve_descriptor.set_width(u64::from(descriptor.width));
    resolve_descriptor.set_height(u64::from(descriptor.height));
    resolve_descriptor.set_storage_mode(MTLStorageMode::Private);
    resolve_descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    let resolve = device.new_texture(&resolve_descriptor);

    let pass = RenderPassDescriptor::new();
    let attachment = pass
        .color_attachments()
        .object_at(0)
        .expect("color attachment zero");
    attachment.set_texture(Some(&multisample));
    attachment.set_resolve_texture(Some(&resolve));
    attachment.set_load_action(MTLLoadAction::Clear);
    attachment.set_store_action(MTLStoreAction::MultisampleResolve);
    attachment.set_clear_color(MTLClearColor::new(0.0, 1.0, 0.0, 1.0));
    let render = queue.new_command_buffer().to_owned();
    let encoder = render.new_render_command_encoder(pass);
    encoder.end_encoding();
    render.commit();

    let bytes = PendingTextureReadback::encode(&device, &queue, &resolve, descriptor)
        .expect("encode resolve readback")
        .finish()
        .expect("finish resolve readback");
    assert!(bytes.chunks_exact(4).all(|pixel| pixel == [0, 255, 0, 255]));
}

#[test]
fn private_depth32_target_readback_preserves_float_samples() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let descriptor = GuestTextureDescriptor {
        address: 0,
        width: 2,
        height: 2,
        pitch: 2,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::Depth32Float,
        bytes_per_pixel: 4,
    };
    let metal_descriptor = TextureDescriptor::new();
    metal_descriptor.set_texture_type(MTLTextureType::D2);
    metal_descriptor.set_pixel_format(descriptor.pixel_format);
    metal_descriptor.set_width(u64::from(descriptor.width));
    metal_descriptor.set_height(u64::from(descriptor.height));
    metal_descriptor.set_storage_mode(MTLStorageMode::Private);
    metal_descriptor.set_usage(MTLTextureUsage::RenderTarget);
    let texture = device.new_texture(&metal_descriptor);

    let pass = RenderPassDescriptor::new();
    let attachment = pass.depth_attachment().expect("depth attachment");
    attachment.set_texture(Some(&texture));
    attachment.set_load_action(MTLLoadAction::Clear);
    attachment.set_store_action(MTLStoreAction::Store);
    attachment.set_clear_depth(0.25);
    let render = queue.new_command_buffer().to_owned();
    let encoder = render.new_render_command_encoder(pass);
    encoder.end_encoding();
    render.commit();

    let bytes = PendingTextureReadback::encode(&device, &queue, &texture, descriptor)
        .expect("encode depth readback")
        .finish()
        .expect("finish depth readback");
    assert!(
        bytes
            .chunks_exact(4)
            .map(|sample| f32::from_ne_bytes(sample.try_into().expect("four-byte depth")))
            .all(|sample| (sample - 0.25).abs() <= f32::EPSILON)
    );
}

#[test]
fn promotes_small_depth_target_to_the_color_framebuffer_extent() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let allocation_size = 4 * GUEST_PAGE_SIZE;
    let mut memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(allocation_size).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            allocation_size,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("guest depth mapping");
    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let descriptor = GuestTextureDescriptor {
        address: address.get(),
        width: 64,
        height: 64,
        pitch: 64,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::Depth32Float,
        bytes_per_pixel: 4,
    };
    let depth = DecodedDepthTarget {
        texture: descriptor,
        compare: MTLCompareFunction::Less,
        test_enabled: true,
        write_enabled: false,
        clear_enabled: false,
        clear_depth: 1.0,
        sample_count: 1,
    };

    let promoted = presenter
        .ensure_depth_render_target_texture(&device, depth, 128, 96)
        .expect("promote depth target");
    assert_eq!((promoted.width(), promoted.height()), (128, 96));

    let promoted_descriptor = GuestTextureDescriptor {
        width: 128,
        height: 96,
        pitch: 128,
        ..descriptor
    };
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal queue")
        .clone();
    let bytes = read_texture_linear(&device, &queue, &promoted, promoted_descriptor)
        .expect("read promoted depth target");
    let sample = |x: usize, y: usize| {
        let offset = (y * 128 + x) * 4;
        f32::from_ne_bytes(bytes[offset..offset + 4].try_into().expect("depth sample"))
    };
    assert_eq!(sample(0, 0), 0.0);
    assert_eq!(sample(100, 50), 1.0);
}

#[test]
fn decodes_agc_16_and_32_bit_index_types() {
    let indices_16 = [0_u16, 1, 2, 0, 2, 3];
    let draw_16 = indexed_draw(indices_16.as_ptr().addr() as u64, 0, 6);
    assert_eq!(
        read_draw_indices(&draw_16).as_deref(),
        Some([0, 1, 2, 0, 2, 3].as_slice())
    );

    let indices_32 = [0_u32, 1, 2, 0, 2, 3];
    let draw_32 = indexed_draw(indices_32.as_ptr().addr() as u64, 1, 6);
    assert_eq!(
        read_draw_indices(&draw_32).as_deref(),
        Some([0, 1, 2, 0, 2, 3].as_slice())
    );
}

#[test]
fn prefers_immutable_index_snapshot_over_recycled_guest_memory() {
    let mut transient_indices = [0_u16, 1, 2, 0, 2, 3];
    let mut draw = indexed_draw(transient_indices.as_ptr().addr() as u64, 0, 6);
    draw.index_snapshot = Some(std::sync::Arc::from([0_u32, 1, 2, 0, 2, 3]));
    transient_indices.fill(0xffff);

    assert_eq!(
        read_draw_indices(&draw).as_deref(),
        Some([0, 1, 2, 0, 2, 3].as_slice())
    );
}

#[test]
fn aligns_implicit_linear_rgba_texture_pitch() {
    assert_eq!(effective_linear_pitch(320, 4), Some(320));
    assert_eq!(effective_linear_pitch(980, 4), Some(1024));
    assert_eq!(effective_linear_pitch(250, 4), Some(256));
    assert_eq!(effective_linear_pitch(250, 1), Some(256));
    assert_eq!(effective_linear_pitch(250, 2), Some(256));
    assert_eq!(effective_linear_pitch(250, 8), Some(256));
    assert_eq!(effective_linear_pitch(250, 16), Some(256));
}

#[test]
fn decodes_all_gfx10_color_slots_and_dense_metal_output_layout() {
    let address0 = 0x0070_1234_5600_u64;
    let address3 = 0x0071_89ab_cd00_u64;
    let mut registers = vec![GpuRegister {
        offset: CB_TARGET_MASK,
        value: 0x0000_f00b,
    }];
    append_color_target_registers(&mut registers, 0, address0, 1_280, 720, 10, 0, 27);
    append_color_target_registers(&mut registers, 3, address3, 1_280, 720, 12, 4, 27);
    registers.sort_by_key(|register| register.offset);

    let targets = decode_render_targets_from_registers(&registers, 0x0000_5007)
        .expect("two valid MRT targets");
    assert_eq!(targets.len(), 2);
    assert_eq!(targets[0].guest_slot, 0);
    assert_eq!(targets[0].texture.address, address0);
    assert_eq!(targets[0].texture.pixel_format, MTLPixelFormat::RGBA8Unorm);
    assert_eq!(targets[0].texture.bytes_per_pixel, 4);
    assert_eq!(targets[0].output_kind, PixelOutputKind::Float);
    assert_eq!(targets[0].write_mask, 0x3);
    assert_eq!(targets[1].guest_slot, 3);
    assert_eq!(targets[1].texture.address, address3);
    assert_eq!(targets[1].texture.pixel_format, MTLPixelFormat::RGBA16Uint);
    assert_eq!(targets[1].texture.bytes_per_pixel, 8);
    assert_eq!(targets[1].output_kind, PixelOutputKind::Uint);
    assert_eq!(targets[1].write_mask, 0x5);
}

#[test]
fn zero_color_target_mask_is_a_valid_depth_only_attachment_set() {
    let registers = [GpuRegister {
        offset: CB_TARGET_MASK,
        value: 0,
    }];

    let targets = decode_render_targets_from_registers(&registers, 0xf)
        .expect("depth-only draws have no color attachments");
    assert!(targets.is_empty());
}

#[test]
fn decodes_explicit_scanout_target_without_fabricating_context_registers() {
    let surface = GpuColorTarget {
        slot: 0,
        address: 0x0070_1234_5600,
        view: 0,
        info: 10 << 2,
        attrib: 0,
        attrib2: ((3_840 - 1) << 14) | (2_160 - 1),
        attrib3: (27 << 14) | (1 << 24),
    };
    let target = GpuScanoutTarget {
        address: surface.address,
        width: 3_840,
        height: 2_160,
        pitch: 3_840,
        pixel_format: 0x8000_0000_2200_0000,
        tiling_mode: 0,
        export_slot: 0,
        write_mask: 0xf,
        color_surface: Some(surface),
    };

    let decoded = decode_scanout_target(target).expect("valid scanout target");

    assert_eq!(decoded.guest_slot, 0);
    assert_eq!(decoded.texture.address, target.address);
    assert_eq!(
        (decoded.texture.width, decoded.texture.height),
        (3_840, 2_160)
    );
    assert_eq!(decoded.texture.tile_mode, 27);
    assert_eq!(decoded.texture.pixel_format, MTLPixelFormat::RGBA8Unorm);
    assert_eq!(decoded.write_mask, 0xf);
    assert!(!decoded.direct_scanout);
}

#[test]
fn decodes_video_out_only_scanout_as_a_host_target() {
    let target = GpuScanoutTarget {
        address: 0x0070_1234_5600,
        width: 3_840,
        height: 2_160,
        pitch: 3_840,
        pixel_format: 0x8000_0000_2200_0000,
        tiling_mode: 0,
        export_slot: 0,
        write_mask: 0xf,
        color_surface: None,
    };

    let decoded = decode_scanout_target(target).expect("valid direct scanout target");

    assert!(decoded.direct_scanout);
    assert_eq!(
        decoded.texture.pixel_format,
        MTLPixelFormat::RGBA8Unorm_sRGB
    );
    assert_eq!(
        (decoded.texture.width, decoded.texture.height),
        (3_840, 2_160)
    );
}

#[test]
fn pending_flip_without_a_source_retries_at_the_submission_boundary() {
    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let descriptor = GuestTextureDescriptor {
        address: 0x0070_1234_5600,
        width: 640,
        height: 360,
        pitch: 640,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    presenter.ensure_direct_scanout_texture(&device, descriptor);
    presenter.display = Some(DisplayConfiguration {
        width: descriptor.width,
        height: descriptor.height,
        pitch: descriptor.pitch,
        pixel_format: 0x8000_0000_0000_0000,
        tiling_mode: 0,
    });
    presenter.pending_flip = Some(PendingFlip {
        address: descriptor.address,
        buffer_index: 0,
        flip_argument: 0,
        sequence: 1,
    });
    presenter.pending_flip_queue = Some(7);
    presenter.next_flip_sequence = 1;
    presenter.redraw_pending = false;

    presenter.mark_direct_scanout_initialized(descriptor.address);
    presenter.note_pending_flip_target_write(descriptor);

    assert!(!presenter.redraw_pending);
    assert!(presenter.pending_flip_target_written_since_boundary);
    assert_eq!(presenter.uploaded_flip_sequence, 0);
    presenter.finalize_pending_flip_at_submission_boundary(OrderedEventSource::GuestQueue(7));
    assert!(presenter.redraw_pending);
    assert!(!presenter.pending_flip_target_written_since_boundary);
    assert_eq!(presenter.uploaded_flip_sequence, 1);
    assert_eq!(presenter.frame_texture_size, (640, 360));
    assert!(presenter.frame_texture.is_some());

    // Once the flip has an immutable snapshot, later writes to the same allocation belong to
    // a future frame and must not replace it at another submission boundary.
    presenter.redraw_pending = false;
    presenter.note_pending_flip_target_write(descriptor);
    assert!(!presenter.pending_flip_target_written_since_boundary);
    assert!(!presenter.redraw_pending);
    presenter.finalize_pending_flip_at_submission_boundary(OrderedEventSource::GuestQueue(7));
    assert!(!presenter.redraw_pending);
}

#[test]
fn translated_scanout_resolution_prefers_the_current_resident_object() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let allocation_size = 4 * GUEST_PAGE_SIZE;
    let mut memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(allocation_size).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            allocation_size,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("guest texture mapping");
    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let descriptor = GuestTextureDescriptor {
        address: address.get(),
        width: 64,
        height: 64,
        pitch: 64,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let original = presenter
        .ensure_resident_texture(&device, descriptor)
        .expect("resident scanout");
    presenter.mark_resident_texture_written(descriptor);
    presenter.ensure_direct_scanout_texture(&device, descriptor);
    presenter.mark_direct_scanout_initialized(address.get());

    let (_, selected, source) = presenter
        .translated_scanout_source(address.get(), descriptor.width, descriptor.height)
        .expect("translated source");
    assert_eq!(source, "resident");
    assert!(std::ptr::eq(selected.as_ref(), original.as_ref()));

    let metal_descriptor = TextureDescriptor::new();
    metal_descriptor.set_texture_type(MTLTextureType::D2);
    metal_descriptor.set_pixel_format(descriptor.pixel_format);
    metal_descriptor.set_width(u64::from(descriptor.width));
    metal_descriptor.set_height(u64::from(descriptor.height));
    metal_descriptor.set_storage_mode(MTLStorageMode::Private);
    metal_descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    let replacement = device.new_texture(&metal_descriptor);
    let resident = presenter
        .resident_guest_textures
        .get_mut(&descriptor)
        .expect("resident map entry");
    resident.texture = replacement.clone();
    resident.content_version = resident.content_version.saturating_add(1);
    // A synchronization can write Metal's current contents back to guest memory and clear
    // this dirty bit without invalidating the Metal texture. Presentation validity must not
    // be tied to whether another writeback is presently required.
    resident.gpu_written = false;

    let (_, selected, source) = presenter
        .translated_scanout_source(address.get(), descriptor.width, descriptor.height)
        .expect("replacement translated source");
    assert_eq!(source, "resident");
    assert!(std::ptr::eq(selected.as_ref(), replacement.as_ref()));
}

#[test]
fn direct_scanout_texture_becomes_the_presented_frame_without_appkit() {
    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = super::MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let descriptor = GuestTextureDescriptor {
        address: 0x0070_1234_5600,
        width: 640,
        height: 360,
        pitch: 640,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let register_decoded_target = DecodedRenderTarget {
        guest_slot: 0,
        texture: descriptor,
        output_kind: PixelOutputKind::Float,
        write_mask: 0xf,
        direct_scanout: false,
        sample_count: 1,
        texture_type: 9,
        layer_count: 1,
        base_layer: 0,
        mip_level: 0,
        mip_level_count: 1,
        dcc_enabled: false,
        fast_clear_enabled: false,
    };
    assert!(!presenter.is_diagnostic_scanout_target(&register_decoded_target));
    assert!(
        presenter
            .diagnostic_capture_texture(&register_decoded_target)
            .is_none()
    );

    let texture = presenter.ensure_direct_scanout_texture(&device, descriptor);
    assert!(presenter.is_diagnostic_scanout_target(&register_decoded_target));
    assert!(
        presenter
            .diagnostic_capture_texture(&register_decoded_target)
            .is_some()
    );
    assert!(presenter.direct_scanout_needs_initial_clear(descriptor.address));
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal queue")
        .clone();
    let pass = RenderPassDescriptor::new();
    let attachment = pass
        .color_attachments()
        .object_at(0)
        .expect("color attachment zero");
    attachment.set_texture(Some(&texture));
    attachment.set_load_action(MTLLoadAction::Clear);
    attachment.set_store_action(MTLStoreAction::Store);
    attachment.set_clear_color(MTLClearColor::new(0.25, 0.5, 1.0, 0.0));
    let paint = queue.new_command_buffer().to_owned();
    let encoder = paint.new_render_command_encoder(pass);
    encoder.end_encoding();
    paint.commit();
    presenter.mark_direct_scanout_initialized(descriptor.address);
    assert!(!presenter.direct_scanout_needs_initial_clear(descriptor.address));
    let reused = presenter.ensure_direct_scanout_texture(&device, descriptor);
    assert_eq!(
        (reused.width(), reused.height()),
        (texture.width(), texture.height())
    );
    assert!(!presenter.direct_scanout_needs_initial_clear(descriptor.address));
    presenter.next_flip_sequence = 1;

    assert!(
        presenter
            .activate_translated_scanout(descriptor.address, descriptor.width, descriptor.height,)
            .expect("activate direct scanout")
    );
    assert_eq!(presenter.frame_texture_size, (640, 360));
    let output_texture = presenter.frame_texture.as_ref().expect("presented texture");
    assert_eq!(output_texture.width(), texture.width());
    assert_eq!(output_texture.height(), texture.height());

    // Reuse the original VideoOut allocation immediately after the ordered flip. The host
    // frame must retain the contents captured at the flip rather than following the mutation.
    let pass = RenderPassDescriptor::new();
    let attachment = pass
        .color_attachments()
        .object_at(0)
        .expect("color attachment zero");
    attachment.set_texture(Some(&texture));
    attachment.set_load_action(MTLLoadAction::Clear);
    attachment.set_store_action(MTLStoreAction::Store);
    attachment.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));
    let reuse = queue.new_command_buffer().to_owned();
    let encoder = reuse.new_render_command_encoder(pass);
    encoder.end_encoding();
    reuse.commit();

    let output = read_texture_linear(&device, &queue, output_texture, descriptor)
        .expect("read snapshotted scanout");
    assert!(
        output
            .chunks_exact(4)
            .all(|pixel| pixel == [64, 128, 255, 0])
    );

    let output_descriptor = GuestTextureDescriptor {
        address: 0,
        width: 320,
        height: 180,
        pitch: 320,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::BGRA8Unorm,
        bytes_per_pixel: 4,
    };
    let output_metal_descriptor = TextureDescriptor::new();
    output_metal_descriptor.set_texture_type(MTLTextureType::D2);
    output_metal_descriptor.set_pixel_format(output_descriptor.pixel_format);
    output_metal_descriptor.set_width(u64::from(output_descriptor.width));
    output_metal_descriptor.set_height(u64::from(output_descriptor.height));
    output_metal_descriptor.set_storage_mode(MTLStorageMode::Private);
    output_metal_descriptor.set_usage(MTLTextureUsage::RenderTarget);
    let presented = device.new_texture(&output_metal_descriptor);
    let pass = RenderPassDescriptor::new();
    let attachment = pass
        .color_attachments()
        .object_at(0)
        .expect("color attachment zero");
    attachment.set_texture(Some(&presented));
    attachment.set_load_action(MTLLoadAction::Clear);
    attachment.set_store_action(MTLStoreAction::Store);
    attachment.set_clear_color(MTLClearColor::new(1.0, 0.0, 1.0, 1.0));
    let present = queue.new_command_buffer().to_owned();
    let encoder = present.new_render_command_encoder(pass);
    let presentation = presenter
        .presentation
        .as_ref()
        .expect("test presenter has presentation resources");
    encoder.set_render_pipeline_state(&presentation.pipeline);
    encoder.set_viewport(presentation_viewport(640, 360, 320, 180));
    encoder.set_fragment_texture(0, Some(output_texture));
    encoder.set_fragment_sampler_state(0, Some(&presentation.sampler));
    encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 3);
    encoder.end_encoding();
    present.commit();
    let presented = read_texture_linear(&device, &queue, &presented, output_descriptor)
        .expect("read offscreen presentation");
    assert!(
        presented
            .chunks_exact(4)
            .all(|pixel| pixel == [255, 188, 137, 255]),
        "VideoOut scanout alpha must not make the host drawable transparent"
    );
}

#[test]
fn pre_pixel_shader_diagnostic_can_capture_an_exact_resident_mode27_target() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let allocation_size = 4 * GUEST_PAGE_SIZE;
    let mut memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(allocation_size).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            allocation_size,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("guest render target mapping");
    let descriptor = GuestTextureDescriptor {
        address: address.get(),
        width: 64,
        height: 64,
        pitch: 64,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    assert!(guest_texture_byte_count(descriptor).is_some_and(|bytes| bytes <= allocation_size));

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let resident = presenter
        .ensure_resident_texture(&device, descriptor)
        .expect("mode-27 target becomes resident");
    let target = DecodedRenderTarget {
        guest_slot: 0,
        texture: descriptor,
        output_kind: PixelOutputKind::Float,
        write_mask: 0xf,
        direct_scanout: false,
        sample_count: 1,
        texture_type: 9,
        layer_count: 1,
        base_layer: 0,
        mip_level: 0,
        mip_level_count: 1,
        dcc_enabled: false,
        fast_clear_enabled: false,
    };

    assert!(presenter.direct_scanout_textures.is_empty());
    let captured = presenter
        .diagnostic_capture_texture(&target)
        .expect("exact resident target is diagnostic-capturable");
    assert!(std::ptr::eq(captured.as_ref(), resident.as_ref()));
}

#[test]
fn mode27_texture_uses_physically_contiguous_disjoint_cpu_aliases() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(24 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(8 * GUEST_PAGE_SIZE).expect("guest physical backing");
    let base = memory.base();
    let first = memory
        .map_shared(
            GuestMapPlacement::Fixed(base),
            4 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("first physical half");
    let second_address = base
        .checked_add(16 * GUEST_PAGE_SIZE)
        .expect("second alias address");
    let second = memory
        .map_shared(
            GuestMapPlacement::Fixed(second_address),
            4 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            4 * GUEST_PAGE_SIZE,
        )
        .expect("second physical half");
    memory.write(first, &[0x31; 32]).expect("first bytes");
    memory.write(second, &[0x72; 32]).expect("second bytes");
    let descriptor = GuestTextureDescriptor {
        address: first.get(),
        width: 256,
        height: 128,
        pitch: 256,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    assert_eq!(
        guest_texture_byte_count(descriptor),
        Some(8 * GUEST_PAGE_SIZE)
    );
    assert!(
        guest_shared_mapping_identity(descriptor.address, 8 * GUEST_PAGE_SIZE).is_none(),
        "the CPU VA has an intentional hole"
    );

    let watch =
        watch_guest_texture(descriptor).expect("GPU-physical texture has complete backing aliases");
    assert_eq!(watch.ranges.len(), 2);
    let raw = snapshot_watched_guest_texture_raw(descriptor, &watch)
        .expect("disjoint aliases snapshot through physical backing");
    assert_eq!(&raw[..32], &[0x31; 32]);
    assert_eq!(
        &raw[usize::try_from(4 * GUEST_PAGE_SIZE).unwrap()
            ..usize::try_from(4 * GUEST_PAGE_SIZE).unwrap() + 32],
        &[0x72; 32]
    );
}

#[test]
fn texture_write_watch_observes_every_live_physical_alias_and_remap() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(16 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(4 * GUEST_PAGE_SIZE).expect("guest physical backing");
    let base = memory.base();
    let primary = memory
        .map_shared(
            GuestMapPlacement::Fixed(base),
            4 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("primary alias");
    let alias_address = base
        .checked_add(8 * GUEST_PAGE_SIZE)
        .expect("alias address");
    let alias = memory
        .map_shared(
            GuestMapPlacement::Fixed(alias_address),
            4 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("second alias");
    let descriptor = GuestTextureDescriptor {
        address: primary.get(),
        width: 64,
        height: 64,
        pitch: 64,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let watch = watch_guest_texture(descriptor).expect("watch every backing alias");
    assert_eq!(watch.ranges.len(), 2);

    memory.write(alias, &[0x5a]).expect("write through alias");
    assert!(
        watch.is_dirty(),
        "a coherent alias write must invalidate the GPU texture"
    );
    memory
        .unmap(alias, 4 * GUEST_PAGE_SIZE)
        .expect("replace alias");
    assert_eq!(
        watch.retirement(),
        Some(CpuWriteWatchRetirement::Replaced),
        "mapping replacement retires the resource watch before direct access"
    );
    assert!(
        watch
            .with_contiguous_mapping(primary.get(), 4 * GUEST_PAGE_SIZE, || ())
            .is_err(),
        "a retired multi-alias watch cannot authorize a later VA dereference"
    );
}

#[test]
fn contiguous_texture_snapshot_and_writeback_retain_direct_mapping_semantics() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(4 * GUEST_PAGE_SIZE).expect("guest physical backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            4 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("contiguous texture mapping");
    let descriptor = GuestTextureDescriptor {
        address: address.get(),
        width: 64,
        height: 64,
        pitch: 64,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let watch = watch_guest_texture(descriptor).expect("contiguous texture watch");
    assert_eq!(watch.ranges.len(), 1);
    let raw =
        snapshot_watched_guest_texture_raw(descriptor, &watch).expect("guarded direct snapshot");
    assert_eq!(raw.len(), usize::try_from(4 * GUEST_PAGE_SIZE).unwrap());

    let linear = vec![0x3c; 64 * 64 * 4];
    let generations = watch.generations();
    write_watched_guest_texture_linear(descriptor, &linear, &watch)
        .expect("guarded direct writeback");
    assert_eq!(watch.generations(), generations);
    assert!(!watch.is_dirty());
    let decoded = snapshot_watched_guest_texture_raw(descriptor, &watch).expect("updated snapshot");
    assert!(decoded.iter().any(|&byte| byte == 0x3c));
}

#[test]
fn contiguous_texture_access_pins_every_adjacent_mapping_record() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapCoalescing, GuestMapPlacement,
        GuestPhysicalBacking, GuestProtection, GuestSharedMapOptions,
    };

    let mut memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(4 * GUEST_PAGE_SIZE).expect("guest physical backing");
    let base = memory.base();
    let first = memory
        .map_shared_with_coalescing(
            GuestMapPlacement::Fixed(base),
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            GuestSharedMapOptions {
                backing_offset: 0,
                coalescing: GuestMapCoalescing::Disallowed,
            },
        )
        .expect("first direct-memory mapping");
    memory
        .map_shared_with_coalescing(
            GuestMapPlacement::Fixed(
                base.checked_add(2 * GUEST_PAGE_SIZE)
                    .expect("second mapping address"),
            ),
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            GuestSharedMapOptions {
                backing_offset: 2 * GUEST_PAGE_SIZE,
                coalescing: GuestMapCoalescing::Disallowed,
            },
        )
        .expect("second direct-memory mapping");
    let descriptor = GuestTextureDescriptor {
        address: first.get(),
        width: 64,
        height: 64,
        pitch: 64,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    assert!(guest_shared_mapping_identity(descriptor.address, 4 * GUEST_PAGE_SIZE).is_some());
    let watch = watch_guest_texture(descriptor).expect("adjacent mapping watches");
    assert_eq!(watch.ranges.len(), 2);
    assert!(!watch.has_single_covering_mapping(descriptor.address, 4 * GUEST_PAGE_SIZE));
    let raw = snapshot_watched_guest_texture_raw(descriptor, &watch)
        .expect("snapshot uses the coherent physical backing slow path");
    assert_eq!(raw.len(), usize::try_from(4 * GUEST_PAGE_SIZE).unwrap());
}

#[test]
fn silent_hill_mode27_mip_tail_round_trips_every_level() {
    let descriptor = GuestTextureDescriptor {
        address: 0x80b9_2700_00,
        width: 32,
        height: 32,
        pitch: 32,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RG16Uint,
        bytes_per_pixel: 4,
    };
    let layout = mode27_mip_chain_layout(descriptor, 6).expect("complete mip-tail layout");
    assert_eq!(layout.byte_count, 65_536);
    assert!(layout.levels.iter().all(|level| level.in_tail));
    assert_eq!(
        layout
            .levels
            .iter()
            .map(|level| (level.width, level.height, level.tail_x, level.tail_y))
            .collect::<Vec<_>>(),
        vec![
            (32, 32, 64, 0),
            (16, 16, 0, 64),
            (8, 8, 32, 0),
            (4, 4, 0, 32),
            (2, 2, 16, 0),
            (1, 1, 8, 16),
        ]
    );

    let levels = layout
        .levels
        .iter()
        .enumerate()
        .map(|(index, level)| {
            let byte_count = level.width * level.height * layout.bytes_per_element;
            (0..byte_count)
                .map(|byte| {
                    u8::try_from((index * 37 + byte) & 0xff).expect("masked test byte fits")
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut raw = vec![0xa5; layout.byte_count];
    encode_mode27_mip_chain_into_raw(&layout, &levels, &mut raw).expect("encode exact mip tail");
    let (decoded_layout, decoded) =
        decode_mode27_mip_chain_from_raw(descriptor, 6, &raw).expect("decode exact mip tail");
    assert_eq!(decoded_layout, layout);
    assert_eq!(decoded, levels);
}

#[test]
fn mode27_mip_chain_round_trips_multiple_nontail_blocks_and_tail() {
    let descriptor = GuestTextureDescriptor {
        address: 0x80b9_2800_00,
        width: 512,
        height: 256,
        pitch: 512,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let layout = mode27_mip_chain_layout(descriptor, 10).expect("complete mip-chain layout");
    assert!(layout.levels.iter().any(|level| !level.in_tail));
    assert!(layout.levels.iter().any(|level| level.in_tail));
    let levels = layout
        .levels
        .iter()
        .enumerate()
        .map(|(level_index, level)| {
            (0..level.width * level.height * layout.bytes_per_element)
                .map(|byte| {
                    u8::try_from((level_index * 41 + byte * 17 + byte / 251) & 0xff)
                        .expect("masked test byte fits")
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut raw = vec![0x5a; layout.byte_count];
    encode_mode27_mip_chain_into_raw(&layout, &levels, &mut raw)
        .expect("encode complete mip chain");
    let (decoded_layout, decoded) =
        decode_mode27_mip_chain_from_raw(descriptor, 10, &raw).expect("decode complete mip chain");
    assert_eq!(decoded_layout, layout);
    assert_eq!(decoded, levels);
}

#[test]
fn silent_hill_render_target_accepts_absolute_max_mip_hierarchy() {
    let address = 0x80b9_2700_00;
    let mut registers = vec![GpuRegister {
        offset: CB_TARGET_MASK,
        value: 0xf,
    }];
    append_color_target_registers(&mut registers, 0, address, 32, 32, 5, 4, 27);
    registers
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_ATTRIB2)
        .expect("color attrib2 exists")
        .value |= 5 << 28;
    registers.sort_by_key(|register| register.offset);
    let targets = decode_render_targets_from_registers(&registers, 0xf)
        .expect("observed mip=0/max_mip=5 target is complete");
    assert_eq!(targets.len(), 1);
    assert_eq!(targets[0].mip_level, 0);
    assert_eq!(targets[0].mip_level_count, 6);
    assert_eq!(targets[0].texture.pixel_format, MTLPixelFormat::RG16Uint);
}

#[test]
fn render_target_decode_rejects_incomplete_or_lossy_state() {
    assert!(
        decode_render_target_format(7, 7, 0)
            .expect_err("11_11_10 float needs explicit repacking and R/B remapping")
            .contains("format=7")
    );

    let address = 0x0070_1234_5600_u64;
    let mut registers = vec![GpuRegister {
        offset: CB_TARGET_MASK,
        value: 0xf,
    }];
    append_color_target_registers(&mut registers, 0, address, 640, 360, 10, 0, 27);
    registers.sort_by_key(|register| register.offset);
    registers
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_INFO)
        .expect("color info exists")
        .value |= 1 << 28;
    let dcc_targets = decode_render_targets_from_registers(&registers, 0xf)
        .expect("DCC targets use uncompressed resident Metal storage");
    assert!(dcc_targets[0].dcc_enabled);

    registers
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_INFO)
        .expect("color info exists")
        .value &= !(1 << 28);
    registers
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_INFO)
        .expect("color info exists")
        .value |= 1 << 13;
    let fast_clear_targets = decode_render_targets_from_registers(&registers, 0xf)
        .expect("fast-clear targets use cleared uncompressed resident Metal storage");
    assert!(fast_clear_targets[0].fast_clear_enabled);

    registers
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_INFO)
        .expect("color info exists")
        .value &= !(1 << 13);
    registers
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_VIEW)
        .expect("color view exists")
        .value = 1 << 26;
    assert!(
        decode_render_targets_from_registers(&registers, 0xf)
            .expect_err("nonzero mip view cannot be silently flattened")
            .contains("mip=1/0")
    );

    registers
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_VIEW)
        .expect("color view exists")
        .value = 0;
    registers
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_ATTRIB3)
        .expect("color attrib3 exists")
        .value = (27 << 14) | (3 << 24);
    assert!(
        decode_render_targets_from_registers(&registers, 0xf)
            .expect_err("reserved resource type cannot be treated as a 2D texture")
            .contains("type=3")
    );

    registers
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_ATTRIB3)
        .expect("color attrib3 exists")
        .value = (27 << 14) | (1 << 24);

    registers.retain(|register| register.offset != CB_COLOR0_ATTRIB2);
    assert!(
        decode_render_targets_from_registers(&registers, 0xf)
            .expect_err("missing extent register must fail")
            .contains("CB_COLOR_ATTRIB2")
    );
}

#[test]
fn decodes_silent_hill_volume_render_target() {
    let address = 0x0080_248f_0000_u64;
    let mut registers = vec![GpuRegister {
        offset: CB_TARGET_MASK,
        value: 0xf,
    }];
    append_color_target_registers(&mut registers, 0, address, 64, 64, 12, 7, 27);
    registers
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_VIEW)
        .expect("color view exists")
        .value = 64 << 13;
    registers
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_ATTRIB3)
        .expect("color attrib3 exists")
        .value = 63 | (27 << 14) | (2 << 24);
    registers.sort_by_key(|register| register.offset);

    let target = decode_render_targets_from_registers(&registers, 0xf)
        .expect("Silent Hill volume target is representable")
        .remove(0);
    assert_eq!(target.texture.address, address);
    assert_eq!((target.texture.width, target.texture.height), (64, 64));
    assert_eq!(target.texture_type, 10);
    assert_eq!(target.layer_count, 64);
    assert_eq!(target.base_layer, 0);
}

#[test]
fn mixed_render_targets_use_their_common_layer_capacity() {
    assert_eq!(common_render_target_array_length([1, 64]), 1);
    assert_eq!(common_render_target_array_length([64, 64]), 64);
    assert_eq!(common_render_target_array_length([]), 1);
}

#[test]
fn maps_guest_number_types_to_matching_metal_export_types() {
    let float = decode_render_target_format(10, 0, 0).expect("RGBA8 unorm");
    let uint = decode_render_target_format(10, 4, 0).expect("RGBA8 uint");
    let sint = decode_render_target_format(12, 5, 0).expect("RGBA16 sint");
    let standard_10_10_10_2 = decode_render_target_format(9, 0, 0).expect("standard packed color");
    let bgra = decode_render_target_format(10, 0, 1).expect("BGRA component swap");
    let rgba16_alt =
        decode_render_target_format(12, 7, 1).expect("RGBA16F alternate component swap");
    assert_eq!(float.output_kind, PixelOutputKind::Float);
    assert_eq!(uint.output_kind, PixelOutputKind::Uint);
    assert_eq!(sint.output_kind, PixelOutputKind::Sint);
    assert_eq!(
        standard_10_10_10_2.pixel_format,
        MTLPixelFormat::RGB10A2Unorm
    );
    assert_eq!(bgra.pixel_format, MTLPixelFormat::BGRA8Unorm);
    assert_eq!(rgba16_alt.pixel_format, MTLPixelFormat::RGBA16Float);
    assert!(decode_render_target_format(8, 0, 0).is_err());
    assert!(decode_render_target_format(10, 2, 0).is_err());
    assert!(decode_render_target_format(10, 0, 2).is_err());
}

#[test]
fn decodes_gfx10_fast_clear_words_for_metal_formats() {
    let close = |actual: f64, expected: f64| {
        assert!(
            (actual - expected).abs() < 0.000_01,
            "actual {actual} differs from expected {expected}"
        );
    };

    let rgba = decode_fast_clear_color(MTLPixelFormat::RGBA8Unorm, [0x8040_2010, 0])
        .expect("RGBA8 clear color");
    close(rgba.red, 16.0 / 255.0);
    close(rgba.green, 32.0 / 255.0);
    close(rgba.blue, 64.0 / 255.0);
    close(rgba.alpha, 128.0 / 255.0);

    let bgra = decode_fast_clear_color(MTLPixelFormat::BGRA8Unorm, [0x8040_2010, 0])
        .expect("BGRA8 clear color");
    close(bgra.red, 64.0 / 255.0);
    close(bgra.green, 32.0 / 255.0);
    close(bgra.blue, 16.0 / 255.0);
    close(bgra.alpha, 128.0 / 255.0);

    let srgb = decode_fast_clear_color(MTLPixelFormat::RGBA8Unorm_sRGB, [0xffbc_bcbc, 0])
        .expect("sRGB clear color");
    close(srgb.red, 0.502_886_458);
    close(srgb.green, 0.502_886_458);
    close(srgb.blue, 0.502_886_458);
    close(srgb.alpha, 1.0);

    let rgba16f = decode_fast_clear_color(MTLPixelFormat::RGBA16Float, [0x3c00_3800, 0xbc00_4000])
        .expect("RGBA16 float clear color");
    close(rgba16f.red, 0.5);
    close(rgba16f.green, 1.0);
    close(rgba16f.blue, 2.0);
    close(rgba16f.alpha, -1.0);

    let rg32u = decode_fast_clear_color(MTLPixelFormat::RG32Uint, [u32::MAX, 0x8000_0000])
        .expect("RG32 uint clear color");
    close(rg32u.red, f64::from(u32::MAX));
    close(rg32u.green, f64::from(0x8000_0000_u32));

    let rgb10a2 = decode_fast_clear_color(MTLPixelFormat::RGB10A2Unorm, [u32::MAX, 0])
        .expect("RGB10A2 clear color");
    close(rgb10a2.red, 1.0);
    close(rgb10a2.green, 1.0);
    close(rgb10a2.blue, 1.0);
    close(rgb10a2.alpha, 1.0);

    let rg11b10_word = 0x3c0 | (0x380 << 11) | (0x1a0 << 22);
    let rg11b10 = decode_fast_clear_color(MTLPixelFormat::RG11B10Float, [rg11b10_word, 0])
        .expect("RG11B10 clear color");
    close(rg11b10.red, 1.0);
    close(rg11b10.green, 0.5);
    close(rg11b10.blue, 0.25);

    let rgb9e5_word = (16 << 27) | 256 | (256 << 9) | (256 << 18);
    let rgb9e5 = decode_fast_clear_color(MTLPixelFormat::RGB9E5Float, [rgb9e5_word, 0])
        .expect("RGB9E5 clear color");
    close(rgb9e5.red, 1.0);
    close(rgb9e5.green, 1.0);
    close(rgb9e5.blue, 1.0);

    let rgba32f = decode_fast_clear_color(
        MTLPixelFormat::RGBA32Float,
        [0.25_f32.to_bits(), 0.75_f32.to_bits()],
    )
    .expect("RGBA32 float clear color");
    close(rgba32f.red, 0.25);
    close(rgba32f.green, 0.25);
    close(rgba32f.blue, 0.25);
    close(rgba32f.alpha, 0.75);
}

#[test]
fn reads_fast_clear_words_with_the_color_target_register_stride() {
    let slot = 3;
    let primary = slot * CB_COLOR_REGISTER_STRIDE;
    let mut registers = vec![GpuRegister {
        offset: CB_TARGET_MASK,
        value: 0xf << (slot * 4),
    }];
    append_color_target_registers(&mut registers, slot, 0x0070_1234_5600, 640, 360, 10, 0, 27);
    registers
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_INFO + primary)
        .expect("slot info exists")
        .value |= 1 << 13;
    registers.extend([
        GpuRegister {
            offset: CB_COLOR0_CLEAR_WORD0 + primary,
            value: 0x8040_2010,
        },
        GpuRegister {
            offset: CB_COLOR0_CLEAR_WORD1 + primary,
            value: 0xfeed_beef,
        },
    ]);
    registers.sort_by_key(|register| register.offset);

    let target = decode_render_targets_from_registers(&registers, 0xf << (slot * 4))
        .expect("slot-three target")
        .remove(0);
    let color =
        render_target_fast_clear_color(&target, &registers).expect("programmed clear color");
    assert!((color.red - 16.0 / 255.0).abs() < 0.000_01);
    assert!((color.alpha - 128.0 / 255.0).abs() < 0.000_01);

    registers.retain(|register| {
        !matches!(
            register.offset,
            offset
                if offset == CB_COLOR0_CLEAR_WORD0 + primary
                    || offset == CB_COLOR0_CLEAR_WORD1 + primary
        )
    });
    let reset = render_target_fast_clear_color(&target, &registers).expect("reset clear words");
    assert_eq!(
        (reset.red, reset.green, reset.blue, reset.alpha),
        (0.0, 0.0, 0.0, 0.0)
    );
}

#[test]
fn first_use_dcc_target_clears_instead_of_loading_compressed_backing() {
    let slot = 0;
    let mut registers = vec![GpuRegister {
        offset: CB_TARGET_MASK,
        value: 0xf,
    }];
    append_color_target_registers(&mut registers, slot, 0x0070_1234_5600, 640, 360, 10, 0, 27);
    registers.extend([
        GpuRegister {
            offset: CB_COLOR0_CLEAR_WORD0,
            value: 0x8040_2010,
        },
        GpuRegister {
            offset: CB_COLOR0_CLEAR_WORD1,
            value: 0,
        },
    ]);
    registers.sort_by_key(|register| register.offset);
    let ordinary = decode_render_targets_from_registers(&registers, 0xf)
        .expect("ordinary color target")
        .remove(0);

    assert!(
        initial_render_target_clear_color(&ordinary, &registers, false).is_none(),
        "ordinary first-use targets preserve their existing load behavior"
    );

    let mut dcc = ordinary;
    dcc.dcc_enabled = true;
    let discarded = initial_render_target_clear_color(&dcc, &registers, false)
        .expect("first-use DCC backing is discarded");
    assert_eq!(
        (
            discarded.red,
            discarded.green,
            discarded.blue,
            discarded.alpha
        ),
        (0.0, 0.0, 0.0, 1.0)
    );
    assert!(
        initial_render_target_clear_color(&dcc, &registers, true).is_none(),
        "resident DCC targets keep their accumulated Metal contents"
    );

    dcc.fast_clear_enabled = true;
    let programmed = initial_render_target_clear_color(&dcc, &registers, false)
        .expect("programmed DCC fast clear");
    assert!((programmed.red - 16.0 / 255.0).abs() < 0.000_01);
    assert!((programmed.green - 32.0 / 255.0).abs() < 0.000_01);
    assert!((programmed.blue - 64.0 / 255.0).abs() < 0.000_01);
    assert!((programmed.alpha - 128.0 / 255.0).abs() < 0.000_01);

    dcc.dcc_enabled = false;
    let ordinary_fast_clear = initial_render_target_clear_color(&dcc, &registers, false)
        .expect("ordinary fast clear behavior is preserved");
    assert_eq!(ordinary_fast_clear.red, programmed.red);
    assert_eq!(ordinary_fast_clear.green, programmed.green);
    assert_eq!(ordinary_fast_clear.blue, programmed.blue);
    assert_eq!(ordinary_fast_clear.alpha, programmed.alpha);
}

#[test]
fn translates_gfx10_write_mask_bit_order_to_metal() {
    assert_eq!(metal_color_write_mask(0x1).bits(), 0x8);
    assert_eq!(metal_color_write_mask(0x2).bits(), 0x4);
    assert_eq!(metal_color_write_mask(0x4).bits(), 0x2);
    assert_eq!(metal_color_write_mask(0x8).bits(), 0x1);
    assert_eq!(metal_color_write_mask(0xf).bits(), 0xf);
}

#[test]
fn translates_separate_premultiplied_guest_blending() {
    let blend = decode_metal_blend_attachment(0x6501_0501).expect("supported blend state");
    assert!(blend.enabled);
    assert_eq!(blend.source_rgb, MTLBlendFactor::One);
    assert_eq!(blend.destination_rgb, MTLBlendFactor::OneMinusSourceAlpha);
    assert_eq!(blend.rgb_operation, MTLBlendOperation::Add);
    assert_eq!(blend.source_alpha, MTLBlendFactor::One);
    assert_eq!(blend.destination_alpha, MTLBlendFactor::OneMinusSourceAlpha);
    assert_eq!(blend.alpha_operation, MTLBlendOperation::Add);
}

#[test]
fn shares_color_blending_with_alpha_without_separate_alpha() {
    let control = (1 << 30) | 4 | (5 << 8);
    let blend = decode_metal_blend_attachment(control).expect("supported blend state");
    assert_eq!(blend.source_rgb, MTLBlendFactor::SourceAlpha);
    assert_eq!(blend.destination_rgb, MTLBlendFactor::OneMinusSourceAlpha);
    assert_eq!(blend.source_alpha, blend.source_rgb);
    assert_eq!(blend.destination_alpha, blend.destination_rgb);
    assert_eq!(blend.alpha_operation, blend.rgb_operation);
}

#[test]
fn rejects_enabled_blend_encodings_that_cannot_be_lowered() {
    assert!(decode_metal_blend_attachment((1 << 30) | 11).is_err());
    assert!(decode_metal_blend_attachment((1 << 30) | (7 << 5)).is_err());
    assert!(
        !decode_metal_blend_attachment(0x1f1f_1f1f)
            .expect("disabled state ignores inactive factor fields")
            .enabled
    );
}

#[test]
fn apple_metal_compiles_mixed_float_uint_sint_mrt_pipeline() {
    let device = Device::system_default().expect("Apple test host has a Metal device");
    let source = r"
#include <metal_stdlib>
using namespace metal;
struct MixedMrtOutput {
    float4 floating [[color(0)]];
    uint4 unsigned_integer [[color(1)]];
    int4 signed_integer [[color(2)]];
};
vertex float4 remu_mrt_vertex(uint vertex_id [[vertex_id]]) {
    return float4(float(vertex_id & 1u), float((vertex_id >> 1u) & 1u), 0.0, 1.0);
}
fragment MixedMrtOutput remu_mrt_fragment() {
    return { float4(0.25), uint4(7u), int4(-3) };
}
";
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let library = device
        .new_library_with_source(source, &options)
        .expect("mixed MRT MSL compiles");
    let vertex = library
        .get_function("remu_mrt_vertex", None)
        .expect("MRT vertex function exists");
    let fragment = library
        .get_function("remu_mrt_fragment", None)
        .expect("MRT fragment function exists");
    let descriptor = RenderPipelineDescriptor::new();
    descriptor.set_vertex_function(Some(&vertex));
    descriptor.set_fragment_function(Some(&fragment));
    for (location, format, mask) in [
        (0, MTLPixelFormat::RGBA8Unorm, 0xf),
        (1, MTLPixelFormat::RGBA8Uint, 0x5),
        (2, MTLPixelFormat::RGBA16Sint, 0xa),
    ] {
        let attachment = descriptor
            .color_attachments()
            .object_at(location)
            .expect("Metal exposes eight color attachments");
        attachment.set_pixel_format(format);
        attachment.set_write_mask(metal_color_write_mask(mask));
    }
    device
        .new_render_pipeline_state(&descriptor)
        .expect("mixed typed MRT render pipeline compiles");
}

#[test]
fn every_decoded_render_target_format_is_apple_metal_renderable() {
    let device = Device::system_default().expect("Apple test host has a Metal device");
    let source = r"
#include <metal_stdlib>
using namespace metal;
vertex float4 remu_format_vertex(uint vertex_id [[vertex_id]]) {
    return float4(float(vertex_id & 1u), float((vertex_id >> 1u) & 1u), 0.0, 1.0);
}
fragment float4 remu_format_float() { return float4(0.25); }
fragment uint4 remu_format_uint() { return uint4(7u); }
fragment int4 remu_format_sint() { return int4(-3); }
";
    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .expect("format-cohort MSL compiles");
    let vertex = library
        .get_function("remu_format_vertex", None)
        .expect("format vertex exists");
    let cases = [
        (1, 0, 0),
        (1, 1, 0),
        (1, 4, 0),
        (1, 5, 0),
        (1, 6, 0),
        (2, 0, 0),
        (2, 1, 0),
        (2, 4, 0),
        (2, 5, 0),
        (2, 7, 0),
        (3, 0, 0),
        (3, 1, 0),
        (3, 4, 0),
        (3, 5, 0),
        (3, 6, 0),
        (4, 4, 0),
        (4, 5, 0),
        (4, 7, 0),
        (5, 0, 0),
        (5, 1, 0),
        (5, 4, 0),
        (5, 5, 0),
        (5, 7, 0),
        (6, 7, 0),
        (9, 0, 0),
        (10, 0, 0),
        (10, 0, 1),
        (10, 1, 0),
        (10, 4, 0),
        (10, 5, 0),
        (10, 6, 0),
        (10, 6, 1),
        (11, 4, 0),
        (11, 5, 0),
        (11, 7, 0),
        (12, 0, 0),
        (12, 1, 0),
        (12, 4, 0),
        (12, 5, 0),
        (12, 7, 0),
        (14, 4, 0),
        (14, 5, 0),
        (14, 7, 0),
        (24, 7, 0),
    ];
    for (data_format, number_type, component_swap) in cases {
        let decoded = decode_render_target_format(data_format, number_type, component_swap)
            .expect("listed format decodes");
        let fragment_name = match decoded.output_kind {
            PixelOutputKind::Float => "remu_format_float",
            PixelOutputKind::Uint => "remu_format_uint",
            PixelOutputKind::Sint => "remu_format_sint",
        };
        let fragment = library
            .get_function(fragment_name, None)
            .expect("typed fragment exists");
        let descriptor = RenderPipelineDescriptor::new();
        descriptor.set_vertex_function(Some(&vertex));
        descriptor.set_fragment_function(Some(&fragment));
        descriptor
            .color_attachments()
            .object_at(0)
            .expect("color attachment zero exists")
            .set_pixel_format(decoded.pixel_format);
        device.new_render_pipeline_state(&descriptor).unwrap_or_else(|error| {
                panic!(
                    "Metal rejected GFX10 format {data_format}/{number_type}/swap{component_swap} as {:?}: {error}",
                    decoded.pixel_format
                )
            });
    }
}

#[test]
fn expands_nonindexed_rect_lists_to_a_four_vertex_triangle_strip() {
    assert_eq!(metal_nonindexed_vertex_count(7, 3), 4);
    assert_eq!(metal_nonindexed_vertex_count(0x11, 3), 4);
    assert_eq!(metal_nonindexed_vertex_count(4, 3), 3);
}

#[test]
fn maps_legacy_and_gen5_triangle_list_codes_to_metal_triangles() {
    assert_eq!(metal_primitive_type(4), Ok(MTLPrimitiveType::Triangle));
    assert_eq!(metal_primitive_type(7), Ok(MTLPrimitiveType::TriangleStrip));
    assert_eq!(
        metal_primitive_topology_class(7),
        Ok(MTLPrimitiveTopologyClass::Triangle)
    );
}

#[test]
fn decodes_programmed_negative_height_viewport_and_intersected_scissors() {
    let mut draw = indexed_draw(0x10_000, 0, 3);
    draw.context_registers = vec![
        GpuRegister {
            offset: 0x081,
            value: (50 << 16) | 100,
        },
        GpuRegister {
            offset: 0x082,
            value: (900 << 16) | 1_500,
        },
        GpuRegister {
            offset: 0x090,
            value: (100 << 16) | 200,
        },
        GpuRegister {
            offset: 0x091,
            value: (800 << 16) | 1_200,
        },
        GpuRegister {
            offset: 0x10f,
            value: 1_920.0_f32.to_bits(),
        },
        GpuRegister {
            offset: 0x110,
            value: 1_920.0_f32.to_bits(),
        },
        GpuRegister {
            offset: 0x111,
            value: (-1_080.0_f32).to_bits(),
        },
        GpuRegister {
            offset: 0x112,
            value: 1_080.0_f32.to_bits(),
        },
    ];
    draw.context_registers
        .sort_by_key(|register| register.offset);

    let scissor = decode_scissor(&draw, 3_840, 2_160).expect("valid guest scissors");
    let super::GuestScissor::Rect(scissor_rect) = scissor else {
        panic!("expected an intersected scissor rectangle");
    };
    assert_eq!(
        (
            scissor_rect.x,
            scissor_rect.y,
            scissor_rect.width,
            scissor_rect.height
        ),
        (200, 100, 1_000, 700)
    );
    let viewport = decode_viewport(&draw, 3_840, 2_160, scissor);
    assert_eq!(
        (
            viewport.originX,
            viewport.originY,
            viewport.width,
            viewport.height
        ),
        (0.0, 0.0, 3_840.0, 2_160.0)
    );
}

#[test]
fn decodes_the_complete_observed_gfx10_image_descriptor() {
    let address = 0x0070_0119_0000_u64;
    let width = 1_280_u32;
    let height = 720_u32;
    let encoded_width = width - 1;
    let mut words = [0_u32; 8];
    words[0] = u32::try_from(address >> 8).expect("low image address fits");
    words[1] = u32::try_from(address >> 40).expect("high image address fits")
        | (56 << 20)
        | ((encoded_width & 0x3) << 30);
    words[2] = (encoded_width >> 2) | ((height - 1) << 14);
    words[3] = 0x204 | (27 << 20) | (9 << 28);
    words[4] = width - 1;

    let descriptor = decode_image_descriptor(&words).expect("valid RGBA8 2D descriptor");
    assert_eq!(descriptor.guest.address, address);
    assert_eq!(descriptor.guest.width, width);
    assert_eq!(descriptor.guest.height, height);
    assert_eq!(descriptor.guest.pitch, width);
    assert_eq!(descriptor.guest.tile_mode, 27);
    assert_eq!(descriptor.destination_select, 0x204);
    assert!(!image_destination_select_is_native(
        descriptor.guest.pixel_format,
        descriptor.destination_select,
    ));
    assert!(image_destination_select_is_native(
        descriptor.guest.pixel_format,
        0xfac,
    ));
}

#[test]
fn sampled_texture_cache_key_ignores_shader_applied_component_selection() {
    let descriptor = decode_image_descriptor(&[0x804b_9100, 0xc240_0000, 0x021b_c3bf, 0x91b0_03ac])
        .expect("observed Silent Hill HDR source descriptor decodes");
    let mut alternate = descriptor;
    alternate.destination_select = 0xfac;
    assert_eq!(
        super::sampled_texture_key(descriptor, false),
        super::sampled_texture_key(alternate, false)
    );
    assert_ne!(
        super::sampled_texture_key(descriptor, false),
        super::sampled_texture_key(alternate, true)
    );
}

#[test]
fn decodes_silent_hill_rg11b10_tone_map_source_descriptor() {
    let descriptor = decode_image_descriptor(&[0x804b_9100, 0xc240_0000, 0x021b_c3bf, 0x91b0_03ac])
        .expect("observed Silent Hill HDR source descriptor decodes");
    assert_eq!(descriptor.guest.address, 0x804b_9100_00);
    assert_eq!(
        (descriptor.guest.width, descriptor.guest.height),
        (3_840, 2_160)
    );
    assert_eq!(descriptor.guest.pitch, 3_840);
    assert_eq!(descriptor.guest.tile_mode, 27);
    assert_eq!(descriptor.guest.pixel_format, MTLPixelFormat::RG11B10Float);
    assert_eq!(descriptor.guest.bytes_per_pixel, 4);
    assert_eq!(descriptor.destination_select, 0x3ac);
    assert_eq!(descriptor.base_mip_level, 0);
    assert_eq!(descriptor.mip_level_count, 1);
    assert_eq!(descriptor.texture_type, 9);
    assert_eq!(descriptor.layer_count, 1);
}

#[test]
fn ordinary_storage_load_does_not_promote_a_full_mip_view() {
    let descriptor = decode_image_descriptor(&[0x804b_9100, 0xc240_0000, 0x021b_c3bf, 0x9098_03ac])
        .expect("observed-style mode-9 full mip view decodes");
    assert_eq!(descriptor.guest.tile_mode, 9);
    assert_eq!(descriptor.base_mip_level, 0);
    assert_eq!(descriptor.mip_level_count, 9);
    let mut binding = ImageBinding {
        pc: 0x190,
        opcode: Opcode::Known("ImageLoad"),
        control: ImageControl {
            dmask: 0xf,
            vector_address: 0,
            address_registers: vec![0, 1],
            vector_data: 4,
            scalar_resource: 12,
            scalar_sampler: 0,
            dimension: 1,
            is_array: false,
            glc: false,
            slc: false,
            a16: false,
            d16: false,
        },
        resource_descriptor: Vec::new().into(),
        sampler_descriptor: Vec::new().into(),
        mip_level: None,
    };
    assert!(!storage_image_requires_mip_hierarchy(&binding, descriptor));
    binding.opcode = Opcode::Known("ImageLoadMip");
    assert!(storage_image_requires_mip_hierarchy(&binding, descriptor));
}

#[test]
fn invalid_sampled_images_receive_a_dimension_compatible_fallback() {
    let binding = ImageBinding {
        pc: 0x50,
        opcode: Opcode::Known("ImageSample"),
        control: ImageControl {
            dmask: 0xf,
            vector_address: 0,
            address_registers: vec![0, 1],
            vector_data: 4,
            scalar_resource: 12,
            scalar_sampler: 20,
            dimension: 1,
            is_array: false,
            glc: false,
            slc: false,
            a16: false,
            d16: false,
        },
        resource_descriptor: vec![0; 8].into(),
        sampler_descriptor: vec![0; 4].into(),
        mip_level: None,
    };

    assert!(image_resource_descriptor_is_null(
        &binding.resource_descriptor
    ));
    assert!(!image_resource_descriptor_is_null(&[]));
    assert!(decode_image_descriptor(&binding.resource_descriptor).is_err());
    let fallback = fallback_image_descriptor(&binding).expect("fallback descriptor");
    assert_eq!(fallback.guest.address, 0);
    assert_eq!((fallback.guest.width, fallback.guest.height), (1, 1));
    assert_eq!(fallback.guest.pixel_format, MTLPixelFormat::RGBA8Unorm);
    assert_eq!(fallback.texture_type, 9);
    assert_eq!(fallback.layer_count, 1);

    let mut buffer_descriptor = binding.clone();
    buffer_descriptor.pc = 0xc4;
    buffer_descriptor.resource_descriptor =
        vec![0x804b_9100, 0xc240_0000, 0x021b_c3bf, 0x01b0_03ac].into();
    assert!(!image_resource_descriptor_is_null(
        &buffer_descriptor.resource_descriptor
    ));
    assert!(decode_image_descriptor(&buffer_descriptor.resource_descriptor).is_err());
    assert!(sampled_alias_descriptor(&buffer_descriptor).is_none());
    let fallback =
        fallback_image_descriptor(&buffer_descriptor).expect("type-zero sampled fallback");
    assert_eq!(fallback.texture_type, 9);

    buffer_descriptor.resource_descriptor =
        vec![0x804b_9100, 0xc240_0000, 0x021b_c3bf, 0x91b0_03ac].into();
    assert!(sampled_alias_descriptor(&buffer_descriptor).is_some());
}

#[test]
fn preserves_explicit_linear_video_plane_pitch() {
    let address = 0x0070_2180_0000_u64;
    let width = 1_440_u32;
    let height = 1_080_u32;
    let encoded_width = width - 1;
    let mut words = [0_u32; 8];
    words[0] = u32::try_from(address >> 8).expect("low image address fits");
    words[1] = u32::try_from(address >> 40).expect("high image address fits")
        | (1 << 20)
        | ((encoded_width & 0x3) << 30);
    words[2] = (encoded_width >> 2) | ((height - 1) << 14);
    words[3] = 0xfac | (9 << 28);
    words[4] = width - 1;

    let descriptor = decode_image_descriptor(&words).expect("valid linear R8 video plane");

    assert_eq!(descriptor.guest.pixel_format, MTLPixelFormat::R8Unorm);
    assert_eq!(descriptor.guest.bytes_per_pixel, 1);
    assert_eq!(descriptor.guest.pitch, width);
    assert_eq!(
        guest_texture_byte_count(descriptor.guest),
        Some(u64::from(width) * u64::from(height))
    );
}

#[test]
fn decodes_observed_r16_float_image_descriptor() {
    let address = 0x0070_0df8_0000_u64;
    let width = 320_u32;
    let height = 180_u32;
    let encoded_width = width - 1;
    let mut words = [0_u32; 8];
    words[0] = u32::try_from(address >> 8).expect("low image address fits");
    words[1] = u32::try_from(address >> 40).expect("high image address fits")
        | (13 << 20)
        | ((encoded_width & 0x3) << 30);
    words[2] = (encoded_width >> 2) | ((height - 1) << 14);
    words[3] = 0xfac | (27 << 20) | (9 << 28);
    words[4] = width - 1;

    let descriptor = decode_image_descriptor(&words).expect("valid R16-float descriptor");

    assert_eq!(descriptor.guest.pixel_format, MTLPixelFormat::R16Float);
    assert_eq!(descriptor.guest.bytes_per_pixel, 2);
    assert_eq!(descriptor.destination_select, 0xfac);
}

#[test]
fn decodes_observed_bc3_and_bc7_sampled_textures_as_four_by_four_blocks() {
    for (unified_format, tile_mode, expected_format) in [
        (174, 27, MTLPixelFormat::BC3_RGBA_sRGB),
        (182, 5, MTLPixelFormat::BC7_RGBAUnorm_sRGB),
    ] {
        let address = 0x0070_0df8_0000_u64;
        let width = 65_u32;
        let height = 65_u32;
        let encoded_width = width - 1;
        let mut words = [0_u32; 8];
        words[0] = u32::try_from(address >> 8).expect("low image address fits");
        words[1] = u32::try_from(address >> 40).expect("high image address fits")
            | (unified_format << 20)
            | ((encoded_width & 0x3) << 30);
        words[2] = (encoded_width >> 2) | ((height - 1) << 14);
        words[3] = 0xfac | (tile_mode << 20) | (9 << 28);
        words[4] = width - 1;

        let descriptor = decode_image_descriptor(&words).expect("valid BC descriptor");
        assert_eq!(descriptor.guest.pixel_format, expected_format);
        assert_eq!(descriptor.guest.bytes_per_pixel, 16);
        assert_eq!(descriptor.guest.tile_mode, tile_mode);
        let expected_bytes = if tile_mode == 5 { 16_384 } else { 65_536 };
        assert_eq!(
            guest_texture_byte_count(descriptor.guest),
            Some(expected_bytes)
        );
    }
}

#[test]
fn maps_unified_rgba8_srgb_sampled_texture_number_type() {
    assert_eq!(
        super::decode_sampled_texture_format(10, 9),
        Ok((MTLPixelFormat::RGBA8Unorm_sRGB, 4))
    );
}

#[test]
fn accepts_a_complete_base_zero_sampled_mip_range() {
    let width = 256_u32;
    let height = 128_u32;
    let encoded_width = width - 1;
    let mut words = [0_u32; 8];
    words[0] = 0x100;
    words[1] = (130 << 20) | ((encoded_width & 0x3) << 30);
    words[2] = (encoded_width >> 2) | ((height - 1) << 14);
    words[3] = 0xfac | (4 << 16) | (24 << 20) | (9 << 28);
    words[4] = width - 1;

    let descriptor = decode_image_descriptor(&words).expect("valid sampled mip hierarchy");
    assert_eq!(descriptor.base_mip_level, 0);
    assert_eq!(descriptor.mip_level_count, 5);
    assert_eq!(descriptor.guest.width, width);
    assert_eq!(descriptor.guest.height, height);
    assert_eq!(
        super::sampled_texture_raw_range(descriptor),
        Ok((
            usize::try_from(descriptor.guest.address).expect("test address fits usize"),
            usize::try_from(
                guest_texture_byte_count(descriptor.guest).expect("base texture byte count fits")
            )
            .expect("base texture byte count fits usize")
        ))
    );
}

#[test]
fn accepts_a_nonzero_sampled_mip_view() {
    let width = 2_048_u32;
    let height = 2_048_u32;
    let encoded_width = width - 1;
    let mut words = [0_u32; 8];
    words[0] = 0x100;
    words[1] = (130 << 20) | ((encoded_width & 0x3) << 30);
    words[2] = (encoded_width >> 2) | ((height - 1) << 14);
    words[3] = 0xfac | (1 << 12) | (4 << 16) | (27 << 20) | (9 << 28);
    words[4] = width - 1;

    let descriptor = decode_image_descriptor(&words).expect("valid nonzero mip view");
    assert_eq!(descriptor.base_mip_level, 1);
    assert_eq!(descriptor.mip_level_count, 4);
    assert_eq!(descriptor.guest.width, width);
    assert_eq!(descriptor.guest.height, height);
    assert_eq!(
        super::sampled_texture_raw_range(descriptor)
            .expect("mode-27 nonzero mip view covers its full underlying chain")
            .1,
        mode27_mip_chain_layout(descriptor.guest, 5)
            .expect("underlying mip layout")
            .byte_count
    );
}

#[test]
fn a_single_level_descriptor_does_not_gain_an_inferred_mip_hierarchy() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    // A mode-27 allocation large enough to hold a six-level tail. The presenter must still
    // create exactly the one level the descriptor declares: inferring the rest fabricates a
    // resource the guest never asked for.
    let allocation_size = 4 * GUEST_PAGE_SIZE;
    let mut memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(allocation_size).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            allocation_size,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("guest mip-tail mapping");
    let guest = GuestTextureDescriptor {
        address: address.get(),
        width: 32,
        height: 32,
        pitch: 32,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RG16Uint,
        bytes_per_pixel: 4,
    };

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let resident = presenter
        .ensure_resident_texture(&device, guest)
        .expect("single-level residency");
    assert_eq!(resident.mipmap_level_count(), 1);
    assert_eq!(
        presenter
            .resident_guest_textures
            .get(&guest)
            .expect("resident recorded")
            .mip_level_count,
        1
    );
}

#[test]
fn reuses_a_larger_resident_mip_hierarchy_for_a_smaller_view() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let allocation_size = 4 * GUEST_PAGE_SIZE;
    let mut memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(allocation_size).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            allocation_size,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("guest mip-tail mapping");
    let guest = GuestTextureDescriptor {
        address: address.get(),
        width: 32,
        height: 32,
        pitch: 32,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RG16Uint,
        bytes_per_pixel: 4,
    };
    assert_eq!(
        mode27_mip_chain_layout(guest, 6)
            .expect("six-level mode-27 tail")
            .byte_count,
        usize::try_from(allocation_size).expect("allocation size fits usize")
    );

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let complete = super::DecodedImageDescriptor {
        guest,
        destination_select: 0x22c,
        base_mip_level: 0,
        mip_level_count: 6,
        texture_type: 9,
        layer_count: 1,
    };
    // Residency comes from the hierarchy the guest declares. A single-level descriptor is
    // never promoted into a larger one on the presenter's own initiative.
    let original = super::ensure_mipmapped_resident_texture_in_map(
        &device,
        complete,
        &mut presenter.resident_guest_textures,
        super::Mode27WorkOrigin::StorageCreate,
    )
    .expect("declared complete mip hierarchy becomes resident");
    assert_eq!(original.mipmap_level_count(), 6);

    let complete_view = super::ensure_mipmapped_resident_texture_in_map(
        &device,
        complete,
        &mut presenter.resident_guest_textures,
        super::Mode27WorkOrigin::StorageCreate,
    )
    .expect("later complete mip view reuses first residency");
    assert!(std::ptr::eq(original.as_ref(), complete_view.as_ref()));
    assert_eq!(
        presenter
            .resident_guest_textures
            .get(&guest)
            .expect("resident survives preflight")
            .mip_level_count,
        6
    );

    let sampled = super::DecodedImageDescriptor {
        guest,
        destination_select: 0x22c,
        base_mip_level: 2,
        mip_level_count: 2,
        texture_type: 9,
        layer_count: 1,
    };
    let view = super::ensure_mipmapped_resident_texture_in_map(
        &device,
        sampled,
        &mut presenter.resident_guest_textures,
        super::Mode27WorkOrigin::StorageCreate,
    )
    .expect("larger resident hierarchy covers the sampled mip range");
    assert_eq!(view.width(), 8);
    assert_eq!(view.height(), 8);
    assert_eq!(view.mipmap_level_count(), 2);
    assert!(view.parent_texture().is_some());
    let resident = presenter
        .resident_guest_textures
        .get(&guest)
        .expect("resident survives narrower view");
    assert_eq!(resident.mip_level_count, 6);
    assert!(std::ptr::eq(original.as_ref(), resident.texture.as_ref()));
}

#[test]
fn creates_exact_initial_mode27_base_three_and_seven_views() {
    let Some(device) = Device::system_default() else {
        return;
    };
    for base_mip_level in [3, 7] {
        let descriptor = super::DecodedImageDescriptor {
            guest: GuestTextureDescriptor {
                address: 0x80c0_0000_00,
                width: 1_024,
                height: 1_024,
                pitch: 1_024,
                tile_mode: 27,
                pixel_format: MTLPixelFormat::RGBA8Unorm,
                bytes_per_pixel: 4,
            },
            destination_select: 0xfac,
            base_mip_level,
            mip_level_count: 1,
            texture_type: 9,
            layer_count: 1,
        };
        let underlying_mip_count = base_mip_level + 1;
        let layout = mode27_mip_chain_layout(descriptor.guest, underlying_mip_count)
            .expect("exact underlying mip layout");
        let raw = vec![0; layout.byte_count];
        let texture = super::create_exact_mode27_sampled_mip_view(&device, descriptor, &raw, false)
            .expect("exact Metal mip view");
        assert_eq!(
            texture.width(),
            u64::from((descriptor.guest.width >> base_mip_level).max(1))
        );
        assert_eq!(
            texture.height(),
            u64::from((descriptor.guest.height >> base_mip_level).max(1))
        );
        assert_eq!(texture.mipmap_level_count(), 1);
    }
}

#[test]
fn decodes_a_complete_six_face_cube_image_descriptor() {
    let address = 0x0070_1a00_0000_u64;
    let width = 128_u32;
    let height = 128_u32;
    let encoded_width = width - 1;
    let mut words = [0_u32; 8];
    words[0] = u32::try_from(address >> 8).expect("low image address fits");
    words[1] = u32::try_from(address >> 40).expect("high image address fits")
        | (130 << 20)
        | ((encoded_width & 0x3) << 30);
    words[2] = (encoded_width >> 2) | ((height - 1) << 14);
    words[3] = 0xfac | (27 << 20) | (11 << 28);
    // BASE_ARRAY=0, LAST_ARRAY=5 describes exactly one six-face cube.
    words[4] = 5;

    let descriptor = decode_image_descriptor(&words).expect("valid cube descriptor");
    assert_eq!(descriptor.texture_type, 11);
    assert_eq!(descriptor.layer_count, 6);
    assert_eq!(descriptor.guest.address, address);
    assert_eq!(descriptor.guest.width, width);
    assert_eq!(descriptor.guest.height, height);
    assert_eq!(descriptor.guest.pitch, width);
    assert_eq!(descriptor.guest.tile_mode, 27);
    assert_eq!(
        super::sampled_texture_raw_range(descriptor),
        Ok((
            usize::try_from(address).expect("test address fits usize"),
            6 * 65_536
        ))
    );
}

#[test]
fn decodes_a_tiled_eight_slice_3d_image_descriptor() {
    let address = 0x0070_1b00_0000_u64;
    let width = 64_u32;
    let height = 64_u32;
    let encoded_width = width - 1;
    let mut words = [0_u32; 8];
    words[0] = u32::try_from(address >> 8).expect("low image address fits");
    words[1] = u32::try_from(address >> 40).expect("high image address fits")
        | (130 << 20)
        | ((encoded_width & 0x3) << 30);
    words[2] = (encoded_width >> 2) | ((height - 1) << 14);
    words[3] = 0xfac | (27 << 20) | (10 << 28);
    words[4] = 7;

    let descriptor = decode_image_descriptor(&words).expect("valid 3D descriptor");
    assert_eq!(descriptor.texture_type, 10);
    assert_eq!(descriptor.layer_count, 8);
    assert_eq!(descriptor.guest.width, width);
    assert_eq!(descriptor.guest.height, height);
    assert_eq!(
        super::sampled_texture_raw_range(descriptor),
        Ok((
            usize::try_from(address).expect("test address fits usize"),
            8 * 65_536
        ))
    );
}

#[test]
fn applies_gfx10_constant_and_channel_image_swizzles() {
    let mut rgba = [10_u8, 20, 30, 40];
    let destination_select = 1 | (5 << 6) | (4 << 9);
    apply_image_destination_select(&mut rgba, destination_select).expect("valid swizzle");
    assert_eq!(rgba, [255, 0, 20, 10]);
}

#[test]
fn expands_r8_red_into_alpha_for_non_native_destination_select() {
    let descriptor = GuestTextureDescriptor {
        address: 0x1_0000,
        width: 2,
        height: 1,
        pitch: 2,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::R8Unorm,
        bytes_per_pixel: 1,
    };
    let (upload_descriptor, upload) =
        super::prepare_sampled_texture_upload(descriptor, vec![10, 20], 0x800)
            .expect("R8 red-to-alpha swizzle");
    assert_eq!(upload_descriptor.pixel_format, MTLPixelFormat::RGBA8Unorm);
    assert_eq!(upload_descriptor.bytes_per_pixel, 4);
    assert_eq!(upload, [0, 0, 0, 10, 0, 0, 0, 20]);
}

#[test]
fn mode27_lookup_matches_the_scalar_address_equation_across_blocks() {
    let width = 385_usize;
    for y in 0..257 {
        for x in 0..width {
            let expected = (y / 128 * width.div_ceil(128) + x / 128) * 65_536
                + mode27_within_block_byte_offset(x, y);
            assert_eq!(mode27_source_byte_offset(x, y, width), expected);
        }
    }
}

#[test]
fn mode24_uses_the_exact_gfx10_z_order_low_address_bits() {
    assert_eq!(mode24_within_block_byte_offset_for_format(0, 0, 4), 0);
    assert_eq!(mode24_within_block_byte_offset_for_format(1, 0, 4), 4);
    assert_eq!(mode24_within_block_byte_offset_for_format(0, 1, 4), 8);
    assert_eq!(mode24_within_block_byte_offset_for_format(2, 0, 4), 16);
    assert_eq!(mode24_within_block_byte_offset_for_format(0, 2, 4), 32);
    assert_eq!(mode24_within_block_byte_offset_for_format(3, 5, 1), 0x27);
    assert_eq!(mode24_within_block_byte_offset_for_format(3, 5, 2), 0x4e);
    assert_eq!(mode24_within_block_byte_offset_for_format(3, 5, 4), 0x9c);
}

#[test]
fn mode27_uses_the_current_gfx10_render_target_address_equation() {
    assert_eq!(mode27_within_block_byte_offset_for_format(0, 0, 4), 0);
    assert_eq!(mode27_within_block_byte_offset_for_format(1, 0, 4), 4);
    assert_eq!(mode27_within_block_byte_offset_for_format(2, 0, 4), 8);
    assert_eq!(mode27_within_block_byte_offset_for_format(0, 1, 4), 16);
    assert_eq!(mode27_within_block_byte_offset_for_format(0, 2, 4), 32);
    assert_eq!(mode27_within_block_byte_offset_for_format(4, 0, 4), 128);
    assert_eq!(super::mode27_surface_z_xor(0), 0);
    assert_eq!(super::mode27_surface_z_xor(1), 0x800);
    assert_eq!(super::mode27_surface_z_xor(2), 0x400);
    assert_eq!(super::mode27_surface_z_xor(4), 0x200);
    assert_eq!(super::mode27_surface_z_xor(8), 0x100);
    assert_eq!(super::mode27_surface_z_xor(15), 0xf00);
}

#[test]
fn mode27_volume_detile_applies_each_surface_z_xor() {
    let layer_count = 32_u32;
    let descriptor = GuestTextureDescriptor {
        address: 0x1_0000,
        width: 32,
        height: 32,
        pitch: 32,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGB10A2Unorm,
        bytes_per_pixel: 4,
    };
    let face_bytes = usize::try_from(guest_texture_byte_count(descriptor).expect("face size fits"))
        .expect("face size fits usize");
    assert_eq!(face_bytes, 65_536);
    let tight_face_bytes = 32 * 32 * 4;
    let mut expected = Vec::with_capacity(tight_face_bytes * layer_count as usize);
    let mut raw = vec![0_u8; face_bytes * layer_count as usize];
    for z in 0..layer_count as usize {
        let mut linear = vec![0_u8; tight_face_bytes];
        for (index, texel) in linear.chunks_exact_mut(4).enumerate() {
            texel.copy_from_slice(
                &(u32::try_from(z).expect("z fits") << 24
                    | u32::try_from(index).expect("texel index fits"))
                .to_ne_bytes(),
            );
        }
        expected.extend_from_slice(&linear);
        let mut tiled = vec![0_u8; face_bytes];
        super::copy_linear_to_tiled(&linear, descriptor, &mut tiled).expect("base slice tiles");
        let xor = super::mode27_surface_z_xor(z);
        let raw_face = &mut raw[z * face_bytes..(z + 1) * face_bytes];
        for source in (0..face_bytes).step_by(4) {
            let destination = source ^ xor;
            raw_face[destination..destination + 4].copy_from_slice(&tiled[source..source + 4]);
        }
    }
    let decoded = super::decode_sampled_texture_linear_from_raw(
        super::DecodedImageDescriptor {
            guest: descriptor,
            destination_select: 0xfac,
            base_mip_level: 0,
            mip_level_count: 1,
            texture_type: 10,
            layer_count,
        },
        &raw,
    )
    .expect("volume detiles");
    assert_eq!(decoded, expected);
}

#[test]
fn mode24_guest_copy_round_trips_partial_and_parity_blocks() {
    let width = 257_usize;
    let height = 257_usize;
    let mut descriptor = GuestTextureDescriptor {
        address: 0x1_0000,
        width: u32::try_from(width).expect("width fits u32"),
        height: u32::try_from(height).expect("height fits u32"),
        pitch: u32::try_from(width).expect("pitch fits u32"),
        tile_mode: 24,
        pixel_format: MTLPixelFormat::R32Float,
        bytes_per_pixel: 4,
    };
    let tiled_length =
        usize::try_from(guest_texture_byte_count(descriptor).expect("tiled byte count fits u64"))
            .expect("tiled byte count fits usize");
    let mut tiled = vec![0_u8; tiled_length];
    descriptor.address =
        u64::try_from(tiled.as_mut_ptr() as usize).expect("test allocation address fits u64");
    let linear = (0..width * height * 4)
        .map(|index| u8::try_from(index & 0xff).expect("masked byte fits u8"))
        .collect::<Vec<_>>();

    write_guest_texture_linear(descriptor, &linear).expect("mode 24 write succeeds");
    assert_eq!(
        snapshot_guest_texture_linear(descriptor).expect("mode 24 snapshot succeeds"),
        linear
    );
}

#[test]
fn mode9_uses_the_exact_standard_64k_address_bits() {
    assert_eq!(mode9_within_block_byte_offset(0, 0, 4), 0);
    assert_eq!(mode9_within_block_byte_offset(1, 0, 4), 4);
    assert_eq!(mode9_within_block_byte_offset(2, 0, 4), 8);
    assert_eq!(mode9_within_block_byte_offset(0, 1, 4), 16);
    assert_eq!(mode9_within_block_byte_offset(0, 2, 4), 32);
    assert_eq!(mode9_within_block_byte_offset(4, 0, 4), 128);
    assert_eq!(mode9_within_block_byte_offset(127, 127, 4), 65_532);
}

#[test]
fn mode9_round_trips_every_supported_element_size() {
    for (bytes_per_element, pixel_format) in [
        (1, MTLPixelFormat::R8Unorm),
        (2, MTLPixelFormat::RG8Unorm),
        (4, MTLPixelFormat::RGBA8Unorm),
        (8, MTLPixelFormat::RGBA16Float),
        (16, MTLPixelFormat::RGBA32Float),
    ] {
        let (block_width, block_height) = tiled_block_dimensions(65_536, bytes_per_element)
            .expect("element size has a mode 9 equation");
        let width = block_width + 1;
        let height = block_height + 1;
        let mut descriptor = GuestTextureDescriptor {
            address: 0x1_0000,
            width: u32::try_from(width).expect("test width fits u32"),
            height: u32::try_from(height).expect("test height fits u32"),
            pitch: u32::try_from(width).expect("test pitch fits u32"),
            tile_mode: 9,
            pixel_format,
            bytes_per_pixel: u32::try_from(bytes_per_element)
                .expect("test bytes per element fits u32"),
        };
        let tiled_length = usize::try_from(
            guest_texture_byte_count(descriptor).expect("tiled byte count fits u64"),
        )
        .expect("tiled byte count fits usize");
        let mut tiled = vec![0_u8; tiled_length];
        descriptor.address =
            u64::try_from(tiled.as_mut_ptr() as usize).expect("test allocation address fits u64");
        let linear = (0..width * height * bytes_per_element)
            .map(|index| u8::try_from(index & 0xff).expect("masked byte fits u8"))
            .collect::<Vec<_>>();
        write_guest_texture_linear(descriptor, &linear).expect("mode 9 write succeeds");
        assert_eq!(
            snapshot_guest_texture_linear(descriptor).expect("mode 9 snapshot succeeds"),
            linear,
            "bytes per element {bytes_per_element}",
        );
    }
}

#[test]
fn bulk_mode27_guest_copy_round_trips_partial_and_parity_blocks() {
    let width = 385_usize;
    let height = 257_usize;
    let mut tiled = vec![0_u8; width.div_ceil(128) * height.div_ceil(128) * 65_536];
    let descriptor = GuestTextureDescriptor {
        address: u64::try_from(tiled.as_mut_ptr() as usize).expect("test address fits u64"),
        width: u32::try_from(width).expect("width fits u32"),
        height: u32::try_from(height).expect("height fits u32"),
        pitch: u32::try_from(width).expect("pitch fits u32"),
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let mut linear = vec![0_u8; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 4;
            linear[offset..offset + 4].copy_from_slice(&[
                u8::try_from(x & 0xff).expect("masked X fits u8"),
                u8::try_from(y & 0xff).expect("masked Y fits u8"),
                u8::try_from((x ^ y) & 0xff).expect("masked xor fits u8"),
                0xff,
            ]);
        }
    }

    write_guest_texture_linear(descriptor, &linear).expect("mode 27 write succeeds");
    assert_eq!(
        snapshot_guest_texture_linear(descriptor).expect("mode 27 snapshot succeeds"),
        linear
    );
}

#[test]
fn tiled_guest_copy_uses_physical_pitch_between_block_rows() {
    let width = 128_usize;
    let height = 129_usize;
    let pitch = 256_usize;
    let mut descriptor = GuestTextureDescriptor {
        address: 0x1_0000,
        width: u32::try_from(width).expect("width fits u32"),
        height: u32::try_from(height).expect("height fits u32"),
        pitch: u32::try_from(pitch).expect("pitch fits u32"),
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let tiled_length =
        usize::try_from(guest_texture_byte_count(descriptor).expect("tiled byte count fits u64"))
            .expect("tiled byte count fits usize");
    assert_eq!(tiled_length, 4 * 65_536);
    let mut tiled = vec![0_u8; tiled_length];
    descriptor.address = u64::try_from(tiled.as_mut_ptr() as usize).expect("test address fits u64");
    let linear = (0..width * height * 4)
        .map(|index| u8::try_from(index & 0xff).expect("masked byte fits u8"))
        .collect::<Vec<_>>();

    write_guest_texture_linear(descriptor, &linear).expect("padded tile write succeeds");
    assert!(tiled[65_536..2 * 65_536].iter().all(|&byte| byte == 0));
    assert_eq!(
        snapshot_guest_texture_linear(descriptor).expect("padded tile snapshot succeeds"),
        linear
    );
}

#[test]
fn mode27_round_trips_every_supported_render_target_element_size() {
    for (bytes_per_pixel, pixel_format) in [
        (1, MTLPixelFormat::R8Unorm),
        (2, MTLPixelFormat::RG8Unorm),
        (4, MTLPixelFormat::RGBA8Unorm),
        (8, MTLPixelFormat::RGBA16Float),
        (16, MTLPixelFormat::RGBA32Float),
    ] {
        let (block_width, block_height) = mode27_block_dimensions(bytes_per_pixel)
            .expect("render-target element size has a mode 27 equation");
        let width = block_width + 1;
        let height = block_height + 1;
        let mut descriptor = GuestTextureDescriptor {
            address: 0x1_0000,
            width: u32::try_from(width).expect("test width fits u32"),
            height: u32::try_from(height).expect("test height fits u32"),
            pitch: u32::try_from(width).expect("test pitch fits u32"),
            tile_mode: 27,
            pixel_format,
            bytes_per_pixel: u32::try_from(bytes_per_pixel).expect("test bytes per pixel fits u32"),
        };
        let tiled_length = usize::try_from(
            guest_texture_byte_count(descriptor).expect("tiled byte count fits u64"),
        )
        .expect("tiled byte count fits usize");
        let mut tiled = vec![0_u8; tiled_length];
        descriptor.address =
            u64::try_from(tiled.as_mut_ptr() as usize).expect("test allocation address fits u64");
        let linear_length = width * height * bytes_per_pixel;
        let linear = (0..linear_length)
            .map(|index| u8::try_from(index & 0xff).expect("masked byte fits u8"))
            .collect::<Vec<_>>();
        write_guest_texture_linear(descriptor, &linear).expect("mode 27 write succeeds");
        assert_eq!(
            snapshot_guest_texture_linear(descriptor).expect("mode 27 snapshot succeeds"),
            linear,
            "bytes per pixel {bytes_per_pixel}",
        );
    }
}

#[test]
fn mode5_uses_the_exact_standard_4k_address_bits() {
    assert_eq!(mode5_within_block_byte_offset(0, 0, 4), 0);
    assert_eq!(mode5_within_block_byte_offset(1, 0, 4), 4);
    assert_eq!(mode5_within_block_byte_offset(2, 0, 4), 8);
    assert_eq!(mode5_within_block_byte_offset(0, 1, 4), 16);
    assert_eq!(mode5_within_block_byte_offset(0, 4, 4), 64);
    assert_eq!(mode5_within_block_byte_offset(4, 0, 4), 128);
}

#[test]
fn mode1_round_trips_every_supported_standard_element_size() {
    for (bytes_per_element, pixel_format) in [
        (1, MTLPixelFormat::R8Unorm),
        (2, MTLPixelFormat::RG8Unorm),
        (4, MTLPixelFormat::RGBA8Unorm),
        (8, MTLPixelFormat::RGBA16Float),
        (16, MTLPixelFormat::RGBA32Float),
    ] {
        let (block_width, block_height) = tiled_block_dimensions(256, bytes_per_element)
            .expect("element size has a mode 1 equation");
        let width = block_width + 1;
        let height = block_height + 1;
        let mut descriptor = GuestTextureDescriptor {
            address: 0x1_0000,
            width: u32::try_from(width).expect("test width fits u32"),
            height: u32::try_from(height).expect("test height fits u32"),
            pitch: u32::try_from(width).expect("test pitch fits u32"),
            tile_mode: 1,
            pixel_format,
            bytes_per_pixel: u32::try_from(bytes_per_element)
                .expect("test bytes per element fits u32"),
        };
        let tiled_length = usize::try_from(
            guest_texture_byte_count(descriptor).expect("tiled byte count fits u64"),
        )
        .expect("tiled byte count fits usize");
        let mut tiled = vec![0_u8; tiled_length];
        descriptor.address =
            u64::try_from(tiled.as_mut_ptr() as usize).expect("test allocation address fits u64");
        let linear = (0..width * height * bytes_per_element)
            .map(|index| u8::try_from(index & 0xff).expect("masked byte fits u8"))
            .collect::<Vec<_>>();

        write_guest_texture_linear(descriptor, &linear).expect("mode 1 write succeeds");
        assert_eq!(
            snapshot_guest_texture_linear(descriptor).expect("mode 1 snapshot succeeds"),
            linear,
            "bytes per element {bytes_per_element}",
        );
    }
}

#[test]
fn mode5_round_trips_uncompressed_and_bc_element_grids() {
    for (bytes_per_element, pixel_format, block_compressed) in [
        (1, MTLPixelFormat::R8Unorm, false),
        (2, MTLPixelFormat::RG8Unorm, false),
        (4, MTLPixelFormat::RGBA8Unorm, false),
        (8, MTLPixelFormat::RGBA16Float, false),
        (16, MTLPixelFormat::RGBA32Float, false),
        (16, MTLPixelFormat::BC7_RGBAUnorm_sRGB, true),
    ] {
        let (block_width, block_height) = tiled_block_dimensions(4_096, bytes_per_element)
            .expect("element size has a mode 5 equation");
        let elements_wide = block_width + 1;
        let elements_high = block_height + 1;
        let width = if block_compressed {
            elements_wide * 4
        } else {
            elements_wide
        };
        let height = if block_compressed {
            elements_high * 4
        } else {
            elements_high
        };
        let mut descriptor = GuestTextureDescriptor {
            address: 0x1_0000,
            width: u32::try_from(width).expect("test width fits u32"),
            height: u32::try_from(height).expect("test height fits u32"),
            pitch: u32::try_from(width).expect("test pitch fits u32"),
            tile_mode: 5,
            pixel_format,
            bytes_per_pixel: u32::try_from(bytes_per_element).expect("test element size fits u32"),
        };
        let tiled_length = usize::try_from(
            guest_texture_byte_count(descriptor).expect("tiled byte count fits u64"),
        )
        .expect("tiled byte count fits usize");
        let mut tiled = vec![0_u8; tiled_length];
        descriptor.address =
            u64::try_from(tiled.as_mut_ptr() as usize).expect("test allocation address fits u64");
        let linear_length = elements_wide * elements_high * bytes_per_element;
        let linear = (0..linear_length)
            .map(|index| u8::try_from(index & 0xff).expect("masked byte fits u8"))
            .collect::<Vec<_>>();

        write_guest_texture_linear(descriptor, &linear).expect("mode 5 write succeeds");
        assert_eq!(
            snapshot_guest_texture_linear(descriptor).expect("mode 5 snapshot succeeds"),
            linear,
            "element size {bytes_per_element}, compressed={block_compressed}",
        );
    }
}

#[test]
fn exact_raw_shadow_detects_cpu_writes_without_guessing_dirty_pages() {
    let mut pixels = vec![0x5a_u8; 16 * 8 * 4];
    let descriptor = GuestTextureDescriptor {
        address: u64::try_from(pixels.as_mut_ptr() as usize).expect("test address fits u64"),
        width: 8,
        height: 8,
        pitch: 16,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let shadow = snapshot_guest_texture_raw(descriptor).expect("raw snapshot succeeds");
    assert!(guest_texture_raw_matches(descriptor, &shadow).expect("raw comparison succeeds"));
    pixels[16 * 4 + 3] ^= 0xff;
    assert!(!guest_texture_raw_matches(descriptor, &shadow).expect("raw comparison succeeds"));
}

#[test]
fn guest_resource_aliasing_uses_byte_ranges_not_descriptor_identity() {
    let base = GuestTextureDescriptor {
        address: 0x1_0000,
        width: 64,
        height: 64,
        pitch: 64,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let overlapping_view = GuestTextureDescriptor {
        address: base.address + 8_192,
        width: 32,
        height: 32,
        pitch: 32,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let adjacent = GuestTextureDescriptor {
        address: base.address + 16_384,
        ..overlapping_view
    };
    assert!(guest_texture_descriptors_overlap(base, overlapping_view));
    assert!(!guest_texture_descriptors_overlap(base, adjacent));
}

#[test]
fn rgba_and_bgra_descriptors_can_share_component_reordered_storage() {
    let bgra = GuestTextureDescriptor {
        address: 0x8006_b7b00,
        width: 3_840,
        height: 2_160,
        pitch: 3_840,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::BGRA8Unorm,
        bytes_per_pixel: 4,
    };
    let rgba = GuestTextureDescriptor {
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        ..bgra
    };
    let rgba_srgb = GuestTextureDescriptor {
        pixel_format: MTLPixelFormat::RGBA8Unorm_sRGB,
        ..bgra
    };
    assert!(guest_texture_descriptors_share_component_reordered_storage(
        bgra, rgba
    ));
    assert!(!guest_texture_descriptors_share_storage(bgra, rgba));
    assert!(
        !guest_texture_descriptors_share_component_reordered_storage(bgra, rgba_srgb),
        "linear and sRGB views do not have equivalent sampling semantics"
    );
}

#[test]
fn metal_accepts_a_same_format_red_blue_swizzled_view_without_pixel_format_view_usage() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    descriptor.set_width(4);
    descriptor.set_height(4);
    descriptor.set_storage_mode(MTLStorageMode::Private);
    descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    let texture = device.new_texture(&descriptor);
    let view =
        new_red_blue_swizzled_texture_view(&texture, 0, 1).expect("swizzled Metal texture view");
    assert_eq!(view.pixel_format(), MTLPixelFormat::BGRA8Unorm);
    assert!(view.parent_texture().is_some());

    let pass = RenderPassDescriptor::new();
    let attachment = pass
        .color_attachments()
        .object_at(0)
        .expect("color attachment zero");
    attachment.set_texture(Some(&texture));
    attachment.set_load_action(MTLLoadAction::Clear);
    attachment.set_store_action(MTLStoreAction::Store);
    attachment.set_clear_color(MTLClearColor::new(0.1, 0.2, 0.3, 0.4));
    let clear = queue.new_command_buffer().to_owned();
    let render = clear.new_render_command_encoder(pass);
    render.end_encoding();
    clear.commit();
    clear.wait_until_completed();

    let source = r"
#include <metal_stdlib>
using namespace metal;
kernel void read_swizzled(
    texture2d<float, access::read> input [[texture(0)]],
    device float4* output [[buffer(0)]])
{
    output[0] = input.read(uint2(0));
}
";
    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .expect("swizzle test MSL compiles");
    let function = library
        .get_function("read_swizzled", None)
        .expect("swizzle test function exists");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .expect("swizzle test pipeline compiles");
    let output = device.new_buffer(16, MTLResourceOptions::StorageModeShared);
    let command = queue.new_command_buffer().to_owned();
    let compute = command.new_compute_command_encoder();
    compute.set_compute_pipeline_state(&pipeline);
    compute.set_texture(0, Some(&view));
    compute.set_buffer(0, Some(&output), 0);
    compute.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    compute.end_encoding();
    command.commit();
    command.wait_until_completed();
    // SAFETY: `output` is a 16-byte shared buffer and the completed kernel wrote one float4.
    let sampled =
        unsafe { std::slice::from_raw_parts(output.contents().cast::<f32>(), 4) }.to_vec();
    let guest_zyxw = [sampled[2], sampled[1], sampled[0], sampled[3]];
    for (actual, expected) in guest_zyxw.into_iter().zip([0.1, 0.2, 0.3, 0.4]) {
        assert!((actual - expected).abs() <= 1.0 / 255.0);
    }
}

#[test]
fn render_resolve_candidates_must_alias_the_sampled_storage() {
    let sampled = GuestTextureDescriptor {
        address: 0x1_0000,
        width: 64,
        height: 64,
        pitch: 64,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let overlapping_render = GuestTextureDescriptor {
        address: sampled.address + 8_192,
        ..sampled
    };
    let unrelated_render = GuestTextureDescriptor {
        address: sampled.address + 32_768,
        ..sampled
    };

    assert!(render_resolve_candidate_matches(
        sampled,
        overlapping_render,
        2,
    ));
    assert!(!render_resolve_candidate_matches(
        sampled,
        unrelated_render,
        2,
    ));
}

#[test]
fn render_resolve_candidate_accepts_view_after_large_page_rounded_msaa_storage() {
    let rendered = GuestTextureDescriptor {
        address: 0x7000_0000,
        width: 1_920,
        height: 1_080,
        pitch: 1_920,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA16Float,
        bytes_per_pixel: 8,
    };
    let sampled = GuestTextureDescriptor {
        address: rendered.address + 0x0200_0000,
        ..rendered
    };

    assert!(render_resolve_candidate_matches(sampled, rendered, 2));
}

#[test]
fn shared_backing_aliases_reuse_identical_metal_storage() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(4 * GUEST_PAGE_SIZE).unwrap();
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let first = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let second = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .unwrap();
    let first = GuestTextureDescriptor {
        address: first.get(),
        width: 64,
        height: 64,
        pitch: 64,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let second = GuestTextureDescriptor {
        address: second.get(),
        ..first
    };

    assert!(guest_texture_descriptors_overlap(first, second));
    assert!(guest_texture_descriptors_share_storage(first, second));
}

#[test]
fn persistent_target_removes_per_draw_full_surface_transfers() {
    let frame_bytes = 3_840_u64 * 2_160 * 4;
    let draws = 100_u64;
    // The replaced path copied guest-to-CPU, CPU-to-Metal, Metal-to-CPU, and CPU-to-guest for
    // every draw. A resident target needs one initial upload and one present-boundary readback.
    let previous_target_traffic = frame_bytes * 4 * draws;
    let persistent_target_traffic = frame_bytes * 2;
    assert_eq!(previous_target_traffic, 13_271_040_000);
    assert_eq!(persistent_target_traffic, 66_355_200);
    assert_eq!(previous_target_traffic / persistent_target_traffic, 200);
}

fn compute_launch() -> ComputeLaunch {
    ComputeLaunch {
        grid: ComputeGrid::Threadgroups([8, 4, 1]),
        group_count: [8, 4, 1],
        base_group: [0; 3],
        threads_per_group: [32, 2, 1],
        wave_lane_count: 32,
        lds_size_bytes: 16 * 1024,
        initiator: 0x8045,
    }
}

#[test]
fn validates_exact_metal_compute_execution_geometry() {
    validate_compute_launch_for_metal(compute_launch(), 32, false, 1_024, 32 * 1024, 0, 16 * 1024)
        .expect("representable compute launch");

    let mut splittable_wave64 = compute_launch();
    splittable_wave64.wave_lane_count = 64;
    validate_compute_launch_for_metal(splittable_wave64, 32, true, 1_024, 32 * 1024, 0, 16 * 1024)
        .expect("wave-local wave64 shader can execute as two SIMD32 subgroups");
    assert!(
        validate_compute_launch_for_metal(
            splittable_wave64,
            32,
            false,
            1_024,
            32 * 1024,
            0,
            16 * 1024,
        )
        .expect_err("cross-lane wave64 shader must not be split")
        .contains("wave64")
    );
}

#[test]
fn permits_split_wave64_readfirstlane_for_uniform_and_varying_values() {
    let uniform_move = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Vop1,
        opcode: Opcode::Known("VMovB32"),
        words: Vec::new(),
        sources: vec![Operand::scalar(18)],
        destinations: vec![Operand::vector(0)],
        control: None,
    };
    let readfirstlane = Instruction {
        pc: 4,
        encoding: ShaderEncoding::Vop1,
        opcode: Opcode::Known("VReadfirstlaneB32"),
        words: Vec::new(),
        sources: vec![Operand::vector(0)],
        destinations: vec![Operand::scalar(107)],
        control: None,
    };
    assert!(
        wave64_split_blockers(&Program::new(
            0x1000,
            vec![uniform_move, readfirstlane.clone()]
        ))
        .is_empty()
    );
    assert!(wave64_split_blockers(&Program::new(0x1000, vec![readfirstlane])).is_empty());
}

#[test]
fn permits_split_wave64_half_local_permlane16_operations() {
    for opcode in ["VPermlane16B32", "VPermlanex16B32"] {
        let instruction = Instruction {
            pc: 0x90,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known(opcode),
            words: Vec::new(),
            sources: vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
            destinations: vec![Operand::vector(8)],
            control: None,
        };
        assert!(
            wave64_split_blockers(&Program::new(0x1000, vec![instruction])).is_empty(),
            "{opcode} only permutes within one SIMD32 half"
        );
    }
}

#[test]
fn permits_exercised_split_wave64_lane_dpp_and_predication_operations() {
    let instructions = vec![
        Instruction {
            pc: 0x10,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VReadlaneB32"),
            words: Vec::new(),
            sources: vec![Operand::vector(1), Operand::scalar(2)],
            destinations: vec![Operand::scalar(4)],
            control: None,
        },
        Instruction {
            pc: 0x18,
            encoding: ShaderEncoding::Vop2,
            opcode: Opcode::Known("VMbcntLoU32B32"),
            words: Vec::new(),
            sources: vec![Operand::scalar(2), Operand::vector(1)],
            destinations: vec![Operand::vector(1)],
            control: Some(remu_shader::InstructionControl::Dpp(
                remu_shader::DppControl {
                    control: 0x101,
                    fetch_inactive: false,
                    row_mask: 0xf,
                    bank_mask: 0xf,
                    bound_control: false,
                    absolute_mask: 0,
                    negate_mask: 0,
                },
            )),
        },
        Instruction {
            pc: 0x20,
            encoding: ShaderEncoding::Sop1,
            opcode: Opcode::Known("SOrn2SaveexecB64"),
            words: Vec::new(),
            sources: vec![Operand::scalar(8)],
            destinations: vec![Operand::scalar(10)],
            control: None,
        },
    ];
    assert!(
        wave64_split_blockers(&Program::new(0x1000, instructions)).is_empty(),
        "implemented cross-half lane exchange and half-local DPP/predication are split-safe"
    );
}

#[test]
fn rejects_compute_geometry_metal_cannot_represent_exactly() {
    let mut based = compute_launch();
    based.base_group = [4, 0, 0];
    assert!(
        validate_compute_launch_for_metal(based, 32, false, 1_024, 32 * 1024, 0, 16 * 1024)
            .expect_err("dispatch base is not silently dropped")
            .contains("nonzero base")
    );

    assert!(
        validate_compute_launch_for_metal(
            compute_launch(),
            16,
            false,
            1_024,
            32 * 1024,
            0,
            16 * 1024,
        )
        .expect_err("wave width must match")
        .contains("wave32")
    );

    let mut oversized = compute_launch();
    oversized.threads_per_group = [1_025, 1, 1];
    assert!(
        validate_compute_launch_for_metal(oversized, 32, false, 1_024, 32 * 1024, 0, 16 * 1024,)
            .expect_err("oversized workgroup must fail")
            .contains("1025 threads")
    );

    assert!(
        validate_compute_launch_for_metal(
            compute_launch(),
            32,
            false,
            1_024,
            32 * 1024,
            0,
            8 * 1024,
        )
        .expect_err("LDS binding must equal lowering layout")
        .contains("8192 bytes of LDS")
    );
    assert!(
        validate_compute_launch_for_metal(
            compute_launch(),
            32,
            false,
            1_024,
            32 * 1024,
            20 * 1024,
            16 * 1024,
        )
        .expect_err("Metal LDS limit must be enforced")
        .contains("36864 bytes")
    );
}
