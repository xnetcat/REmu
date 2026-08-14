// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use metal::{
    CompileOptions, Device, MTLBlendFactor, MTLBlendOperation, MTLCompareFunction, MTLCullMode,
    MTLPixelFormat, MTLPrimitiveTopologyClass, MTLPrimitiveType, MTLTextureType,
    MTLTriangleFillMode, MTLWinding, RenderPipelineDescriptor,
};
use remu_gpu::{GpuRegister, ShaderSnapshot};
use remu_shader::{
    ImageBinding, ImageControl, Instruction, Opcode, Operand, PixelExportFormat, PixelOutputKind,
    Program, ShaderEncoding, ShaderEvaluation, decode_program, link_program_continuation,
};
use std::collections::HashMap;
use std::sync::Arc;

use super::super::decode::decode_render_target_state;
use super::{
    CB_COLOR_REGISTER_STRIDE, CB_COLOR0_ATTRIB, CB_COLOR0_ATTRIB2, CB_COLOR0_ATTRIB3,
    CB_COLOR0_CLEAR_WORD0, CB_COLOR0_CLEAR_WORD1, CB_COLOR0_INFO, CB_COLOR0_VIEW, CB_TARGET_MASK,
    ColorOperationMode, DB_DEPTH_CLEAR, DB_DEPTH_CONTROL, DB_DEPTH_VIEW, DB_HTILE_DATA_BASE,
    DB_RENDER_CONTROL, DB_STENCIL_CONTROL, DB_STENCILREFMASK, DB_Z_INFO, DecodedDepthBias,
    DecodedDepthTarget, DecodedRasterState, DecodedStencilState, GuestScissor,
    GuestTextureDescriptor, MetalPresenter, NGG_MESHLET_WORDS_PER_GROUP, NggOutputPrimitiveType,
    NggSubgroupLimits, PA_SU_SC_MODE_CNTL, RenderTargetPipelineState, SPI_PS_INPUT_CNTL_0,
    append_color_target_registers, append_depth_target_registers,
    apply_transparent_premultiplied_fill_clear, color_operation_is_raster_noop,
    common_render_target_array_length, consume_color_metadata_clear, consume_depth_metadata_clear,
    decode_color_eliminate_fast_clear_target, decode_color_operation_mode,
    decode_color_resolve_targets, decode_depth_target, decode_fast_clear_color,
    decode_graphics_wave_state, decode_metal_blend_attachment, decode_ngg_registers_from_state,
    decode_ngg_subgroup_limits, decode_ngg_subgroup_limits_with_uc, decode_raster_state,
    decode_render_target_format, decode_render_targets_from_registers, decode_scissor,
    decode_viewport, disabled_metal_blend_attachment, effective_linear_pitch, extend_gpu_registers,
    fragmentless_pixel_program, graphics_pipeline_key, guest_texture_byte_count,
    guest_texture_descriptors_share_depth_sample_storage, indexed_draw,
    initial_depth_clear_for_target, initial_depth_clear_value, initial_render_target_clear_color,
    is_transparent_premultiplied_fill_blend, metal_color_write_mask, metal_nonindexed_vertex_count,
    metal_primitive_topology_class, metal_primitive_type,
    ngg_fixed_allocation_fallback_for_program, ngg_mesh_base_vertex, ngg_mesh_dispatch_required,
    noop_placeholder_cover_draw, pixel_input_controls, prepare_indexed_ngg_indices,
    prepare_indexed_primitive, prepare_ngg_meshlets, prepare_nonindexed_ngg_indices,
    push_gpu_register, read_draw_indices, read_texture_array_linear, read_texture_linear,
    rearm_color_metadata_clear, rearm_depth_metadata_clear, render_target_cmask_address,
    render_target_fast_clear_color,
};

fn empty_shader_evaluation() -> Arc<ShaderEvaluation> {
    Arc::new(ShaderEvaluation {
        initial_scalar_registers: std::iter::once(0).collect(),
        scalar_registers: std::iter::once(0).collect(),
        scalar_memory_reads: Vec::new(),
        image_bindings: Vec::new(),
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    })
}

#[test]
fn programs_full_scissor_after_a_restricted_draw() {
    let restricted = super::super::execute::metal_scissor_rect_for_draw(
        GuestScissor::Rect(metal::MTLScissorRect {
            x: 32,
            y: 16,
            width: 640,
            height: 360,
        }),
        3_840,
        2_160,
    )
    .expect("restricted draw programs its guest scissor");
    assert_eq!(
        (
            restricted.x,
            restricted.y,
            restricted.width,
            restricted.height
        ),
        (32, 16, 640, 360)
    );

    let full = super::super::execute::metal_scissor_rect_for_draw(GuestScissor::Full, 3_840, 2_160)
        .expect("full draw must reset persistent Metal scissor state");
    assert_eq!(
        (full.x, full.y, full.width, full.height),
        (0, 0, 3_840, 2_160)
    );
    assert!(
        super::super::execute::metal_scissor_rect_for_draw(GuestScissor::Empty, 3_840, 2_160)
            .is_none(),
        "empty guest scissors remain draw suppression instead of invalid Metal rectangles"
    );
}

#[test]
fn clamps_signed_screen_scissor_coordinates_instead_of_dropping_bit_15() {
    let mut draw = indexed_draw(0, 0, 3);
    draw.context_registers = vec![
        GpuRegister {
            offset: 0x00c,
            // PA_SC_SCREEN_SCISSOR_TL: X=-20, Y=-10 (signed 16-bit components).
            value: 0xfff6_ffec,
        },
        GpuRegister {
            offset: 0x00d,
            value: (70 << 16) | 80,
        },
    ]
    .into();

    let scissor = decode_scissor(&draw, 100, 100).expect("signed screen scissor decodes");
    let GuestScissor::Rect(rect) = scissor else {
        panic!("negative top-left must clamp to zero without suppressing the draw");
    };
    assert_eq!((rect.x, rect.y, rect.width, rect.height), (0, 0, 80, 70));
}

#[test]
fn applies_window_offset_disable_independently_for_each_scissor() {
    let mut draw = indexed_draw(0, 0, 3);
    draw.context_registers = vec![
        GpuRegister {
            offset: 0x080,
            value: (50 << 16) | 50,
        },
        GpuRegister {
            offset: 0x081,
            value: 0,
        },
        GpuRegister {
            offset: 0x082,
            value: (150 << 16) | 150,
        },
        GpuRegister {
            offset: 0x090,
            // Generic scissor disables the offset independently of the window scissor.
            value: 0x8000_0000 | (10 << 16) | 10,
        },
        GpuRegister {
            offset: 0x091,
            value: (100 << 16) | 100,
        },
    ]
    .into();
    Arc::make_mut(&mut draw.context_registers).sort_by_key(|register| register.offset);

    let scissor = decode_scissor(&draw, 200, 200).expect("independent scissor flags decode");
    let GuestScissor::Rect(rect) = scissor else {
        panic!("expected intersected window and generic scissors");
    };
    assert_eq!((rect.x, rect.y, rect.width, rect.height), (50, 50, 50, 50));
}

#[test]
fn accepts_fragmentless_depth_only_pipeline() {
    let mut draw = indexed_draw(0, 0, 3);
    draw.indexed = false;
    draw.primitive_type = 7;
    draw.export_shader_address = 0x0080_07a8_0000;
    draw.export_shader = Some(ShaderSnapshot {
        header_address: 0,
        program: Arc::new(Program::new(draw.export_shader_address, Vec::new())),
        supplemental_programs: Arc::from([]),
        metadata: None,
    });
    draw.export_shader_evaluation = Some(empty_shader_evaluation().into());
    append_depth_target_registers(&mut draw, 0x0080_10b5_0000, 256, 256, 0xa000_0003, 0x66);

    let key = graphics_pipeline_key(&draw).expect("depth-only draw has a Metal pipeline");
    assert!(key.targets.is_empty());
    assert!(key.depth.is_some());
    assert_eq!(key.pixel_shader_address, 0);
    assert_eq!(key.pixel_program_identity.0.address, 0);
    assert!(Arc::ptr_eq(
        &key.pixel_program_identity.0,
        fragmentless_pixel_program()
    ));
    assert!(!key.pixel_depth_export);
    assert_eq!(
        key.pixel_allocated_vector_register_count, 4,
        "fragmentless pixel lowering still requires a non-zero VGPR allocation"
    );
    assert_eq!(key.pixel_resources.initial_scalar_count, 1);
    assert!(key.pixel_resources.global_bindings.is_empty());
    assert!(key.pixel_resources.image_bindings.is_empty());
}

#[test]
fn carries_pa_cl_vte_cntl_into_pipeline_key() {
    let mut draw = indexed_draw(0, 0, 3);
    draw.indexed = false;
    draw.primitive_type = 7;
    draw.export_shader_address = 0x0080_07a8_0000;
    draw.export_shader = Some(ShaderSnapshot {
        header_address: 0,
        program: Arc::new(Program::new(draw.export_shader_address, Vec::new())),
        supplemental_programs: Arc::from([]),
        metadata: None,
    });
    draw.export_shader_evaluation = Some(empty_shader_evaluation().into());
    append_depth_target_registers(&mut draw, 0x0080_10b5_0000, 256, 256, 0xa000_0003, 0x66);
    push_gpu_register(
        &mut draw.context_registers,
        GpuRegister {
            offset: 0x206,
            value: 0x43f,
        },
    );

    let key = graphics_pipeline_key(&draw).expect("depth-only draw has a Metal pipeline");
    assert_eq!(
        key.viewport_transform_control, 0x43f,
        "AGC-default PA_CL_VTE_CNTL (VTX_W0_FMT set) must reach the pipeline key"
    );
}

#[test]
fn recognizes_noop_placeholder_cover_without_attachments() {
    let words = [
        0xbfa0_0001,
        0x93ea_ff03,
        0x0008_0008,
        0x876b_ff03,
        0x0000_00ff,
        0x8f6a_8c6a,
        0x887c_6a6b,
        0xbf90_0009,
        0x906a_8803,
        0x81ea_6a80,
        0x90fe_6ac1,
        0xf800_0941,
        0x0000_0000,
        0x81ea_0380,
        0xbf8c_ff0f,
        0xbf81_0000,
    ];
    let program = decode_program(0x8051_4b00_00, &words).expect("memoryless cover decodes");
    let mut draw = super::indexed_draw(0, 0, 3);
    draw.indexed = false;
    draw.primitive_type = 7;
    draw.export_shader_address = 0x8051_4b00_00;
    draw.export_shader = Some(ShaderSnapshot {
        header_address: 0,
        program: Arc::new(program),
        supplemental_programs: Arc::from([]),
        metadata: None,
    });
    draw.export_shader_evaluation = Some(empty_shader_evaluation().into());
    assert!(noop_placeholder_cover_draw(&draw));
    assert!(
        graphics_pipeline_key(&draw)
            .expect_err("pipeline key stays invalid")
            .contains("neither an enabled color target nor an active depth target")
    );
}

#[test]
fn storage_only_image_store_draw_gets_a_dummy_color_target() {
    let words = [
        0xbfa0_0001,
        0x93ea_ff03,
        0x0008_0008,
        0x876b_ff03,
        0x0000_00ff,
        0x8f6a_8c6a,
        0x887c_6a6b,
        0xbf90_0009,
        0x906a_8803,
        0x81ea_6a80,
        0x90fe_6ac1,
        0xf800_0941,
        0x0000_0000,
        0x81ea_0380,
        0xbf8c_ff0f,
        0xbf81_0000,
    ];
    let program = decode_program(0x8051_4b00_00, &words).expect("cover program decodes");
    let snapshot = ShaderSnapshot {
        header_address: 0,
        program: Arc::new(program),
        supplemental_programs: Arc::from([]),
        metadata: None,
    };
    let store = ImageBinding {
        descriptor_index_is_runtime: false,
        pc: 0xc38,
        opcode: Opcode::Known("ImageStore"),
        control: ImageControl {
            dmask: 0xf,
            vector_address: 0,
            address_registers: vec![0, 1, 2],
            vector_data: 4,
            scalar_resource: 0,
            scalar_sampler: 0,
            dimension: 2,
            is_array: false,
            glc: false,
            slc: false,
            a16: false,
            d16: false,
            unorm: false,
        },
        // Live Superliminal cover-pass evaluation snapshots a null SRD; the TYPE-10
        // volume is resolved later from user data. The opcode is the write intent.
        resource_descriptor: vec![0, 0, 0, 0].into(),
        sampler_descriptor: Vec::new().into(),
        mip_level: None,
    };
    let mut pixel_evaluation = empty_shader_evaluation().as_ref().clone();
    pixel_evaluation.image_bindings = vec![store];
    let mut draw = super::indexed_draw(0, 0, 3);
    draw.indexed = false;
    draw.primitive_type = 7;
    draw.export_shader_address = 0x8051_4b00_00;
    draw.export_shader = Some(snapshot.clone());
    // Live Superliminal cover-pass evaluation is skipped: the ES has no resources.
    draw.export_shader_evaluation = None;
    draw.pixel_shader_address = 0x8051_4c00_00;
    draw.pixel_shader = Some(snapshot);
    draw.pixel_shader_evaluation = Some(Arc::new(pixel_evaluation));
    // Superliminal programs CB_TARGET_MASK=0 and leaves STENCIL_ENABLE latched
    // (DB_DEPTH_CONTROL=0x771) without a depth surface.
    push_gpu_register(
        &mut draw.context_registers,
        GpuRegister {
            offset: CB_TARGET_MASK,
            value: 0,
        },
    );
    push_gpu_register(
        &mut draw.context_registers,
        GpuRegister {
            offset: DB_DEPTH_CONTROL,
            value: 0x0000_0771,
        },
    );
    Arc::make_mut(&mut draw.context_registers).sort_unstable_by_key(|register| register.offset);
    let key = graphics_pipeline_key(&draw)
        .expect("ImageStore-only LUT bake must produce a Metal pipeline");
    assert_eq!(key.targets.len(), 1);
    assert_eq!(key.targets[0].pixel_format, MTLPixelFormat::R8Unorm);
    assert_eq!(key.targets[0].write_mask, 0);
}

#[test]
fn rejects_fragmentless_pipeline_with_color_writes() {
    let mut draw = indexed_draw(0, 0, 3);
    draw.export_shader_address = 0x0080_07a8_0000;
    draw.export_shader = Some(ShaderSnapshot {
        header_address: 0,
        program: Arc::new(Program::new(draw.export_shader_address, Vec::new())),
        supplemental_programs: Arc::from([]),
        metadata: None,
    });
    draw.export_shader_evaluation = Some(empty_shader_evaluation().into());
    push_gpu_register(
        &mut draw.context_registers,
        GpuRegister {
            offset: CB_TARGET_MASK,
            value: 0xf,
        },
    );
    Arc::make_mut(&mut draw.context_registers).sort_unstable_by_key(|register| register.offset);

    let error = graphics_pipeline_key(&draw).expect_err("color writes require a pixel shader");
    assert!(error.contains("requests color output"), "{error}");
}

#[test]
fn decodes_cb_color_control_operation_modes() {
    assert_eq!(
        decode_color_operation_mode(0x00cc_0000),
        Ok(ColorOperationMode::Disable)
    );
    assert_eq!(
        decode_color_operation_mode(0x00cc_0010),
        Ok(ColorOperationMode::Normal)
    );
    assert_eq!(
        decode_color_operation_mode(0x00cc_0030),
        Ok(ColorOperationMode::Resolve)
    );
    assert_eq!(
        decode_color_operation_mode(0x00cc_0020),
        Ok(ColorOperationMode::EliminateFastClear)
    );
    assert_eq!(
        decode_color_operation_mode(0x00cc_0050),
        Ok(ColorOperationMode::FmaskDecompress)
    );
    assert!(decode_color_operation_mode(0x00cc_0040).is_err());
}

#[test]
fn disabled_color_operation_without_depth_is_a_raster_noop() {
    assert!(color_operation_is_raster_noop(
        ColorOperationMode::Disable,
        false
    ));
    assert!(!color_operation_is_raster_noop(
        ColorOperationMode::Disable,
        true
    ));
    assert!(!color_operation_is_raster_noop(
        ColorOperationMode::Normal,
        false
    ));
}

#[test]
fn color_resolve_decodes_mrt1_without_a_target_mask_or_shader_export() {
    let source_address = 0x0070_649d_0000;
    let destination_address = 0x0070_7c0d_0000;
    let mut draw = indexed_draw(0, 0, 0);
    append_color_target_registers(
        &mut draw.context_registers,
        0,
        source_address,
        1_920,
        1_080,
        12,
        7,
        27,
    );
    append_color_target_registers(
        &mut draw.context_registers,
        1,
        destination_address,
        1_920,
        1_080,
        12,
        7,
        27,
    );
    push_gpu_register(
        &mut draw.context_registers,
        GpuRegister {
            offset: CB_TARGET_MASK,
            value: 0xf,
        },
    );
    Arc::make_mut(&mut draw.context_registers)
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_ATTRIB)
        .expect("MRT0 attributes")
        .value = (1 << 12) | (1 << 15);
    Arc::make_mut(&mut draw.context_registers).sort_unstable_by_key(|register| register.offset);

    let (source, destination) =
        decode_color_resolve_targets(&draw).expect("valid MRT0-to-MRT1 resolve");
    assert_eq!(source.guest_slot, 0);
    assert_eq!(source.texture.address, source_address);
    assert_eq!(source.sample_count, 2);
    assert_eq!(destination.guest_slot, 1);
    assert_eq!(destination.texture.address, destination_address);
    assert_eq!(destination.sample_count, 1);
}

#[test]
fn color_resolve_decodes_single_sample_mrt0_to_mrt1_copy() {
    let source_address = 0x0070_1214_0000;
    let destination_address = 0x0070_6481_0000;
    let mut draw = indexed_draw(0, 0, 0);
    append_color_target_registers(
        &mut draw.context_registers,
        0,
        source_address,
        1_920,
        1_080,
        12,
        7,
        27,
    );
    append_color_target_registers(
        &mut draw.context_registers,
        1,
        destination_address,
        1_920,
        1_080,
        12,
        7,
        27,
    );
    push_gpu_register(
        &mut draw.context_registers,
        GpuRegister {
            offset: CB_TARGET_MASK,
            value: 0xf,
        },
    );
    Arc::make_mut(&mut draw.context_registers).sort_unstable_by_key(|register| register.offset);

    let (source, destination) =
        decode_color_resolve_targets(&draw).expect("single-sample MRT0-to-MRT1 copy");
    assert_eq!(source.guest_slot, 0);
    assert_eq!(source.texture.address, source_address);
    assert_eq!(source.sample_count, 1);
    assert_eq!(destination.guest_slot, 1);
    assert_eq!(destination.texture.address, destination_address);
    assert_eq!(destination.sample_count, 1);
    assert_eq!(source.texture.width, 1_920);
    assert_eq!(destination.texture.width, 1_920);
}

#[test]
fn fast_clear_elimination_decodes_mrt0_without_a_target_mask_or_shader_export() {
    let address = 0x0070_649d_0000;
    let mut draw = indexed_draw(0, 0, 0);
    append_color_target_registers(
        &mut draw.context_registers,
        0,
        address,
        1_920,
        1_080,
        12,
        7,
        27,
    );
    Arc::make_mut(&mut draw.context_registers)
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_INFO)
        .expect("MRT0 info")
        .value |= 1 << 13;
    Arc::make_mut(&mut draw.context_registers).sort_unstable_by_key(|register| register.offset);

    let target = decode_color_eliminate_fast_clear_target(&draw)
        .expect("valid fast-clear state")
        .expect("fast-clear target");
    assert_eq!(target.guest_slot, 0);
    assert_eq!(target.texture.address, address);
    assert!(target.fast_clear_enabled);

    Arc::make_mut(&mut draw.context_registers)
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_INFO)
        .expect("MRT0 info")
        .value &= !(1 << 13);
    assert!(
        decode_color_eliminate_fast_clear_target(&draw)
            .expect("valid ordinary color state")
            .is_none(),
        "an elimination trigger is a no-op when CMASK fast clear is disabled"
    );
}

#[test]
fn pixel_input_controls_default_to_zero_and_preserve_programmed_remaps() {
    assert_eq!(pixel_input_controls(&[]), [0; 32]);
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
    assert_eq!(controls[2], 0);
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
fn decodes_htile_address_only_when_tile_surface_is_enabled() {
    let mut draw = indexed_draw(0, 0, 0);
    append_depth_target_registers(&mut draw, 0x0070_1412_0000, 1_920, 1_080, 0xa000_0003, 0x66);
    push_gpu_register(
        &mut draw.context_registers,
        GpuRegister {
            offset: DB_HTILE_DATA_BASE,
            value: 0x0012_3456,
        },
    );
    Arc::make_mut(&mut draw.context_registers).sort_unstable_by_key(|register| register.offset);

    let depth = decode_depth_target(&draw)
        .expect("valid HTILE depth target")
        .expect("enabled depth target");
    assert_eq!(depth.htile_address, Some(0x1234_5600));

    Arc::make_mut(&mut draw.context_registers)
        .iter_mut()
        .find(|register| register.offset == DB_Z_INFO)
        .expect("depth info")
        .value &= !(1 << 29);
    let depth = decode_depth_target(&draw)
        .expect("valid uncompressed depth target")
        .expect("enabled depth target");
    assert_eq!(depth.htile_address, None);
}

#[test]
fn depth_metadata_clear_is_one_shot_and_can_be_rearmed() {
    let address = 0x1234_5600;
    let mut states = HashMap::new();

    assert!(consume_depth_metadata_clear(&mut states, Some(address)));
    assert!(!consume_depth_metadata_clear(&mut states, Some(address)));
    assert!(rearm_depth_metadata_clear(&mut states, address));
    assert!(consume_depth_metadata_clear(&mut states, Some(address)));
    assert!(!consume_depth_metadata_clear(&mut states, Some(address)));
    assert!(!rearm_depth_metadata_clear(&mut states, address + 0x100));
    assert!(!consume_depth_metadata_clear(&mut states, None));
}

#[test]
fn color_metadata_clear_is_one_shot_per_slice_and_can_be_rearmed() {
    let address = 0x1234_5600;
    let mut states = HashMap::new();

    assert!(consume_color_metadata_clear(&mut states, Some(address), 0));
    assert!(!consume_color_metadata_clear(&mut states, Some(address), 0));
    assert!(consume_color_metadata_clear(&mut states, Some(address), 1));
    assert!(rearm_color_metadata_clear(&mut states, address));
    assert!(consume_color_metadata_clear(&mut states, Some(address), 0));
    assert!(!rearm_color_metadata_clear(&mut states, address + 0x100));
    assert!(!consume_color_metadata_clear(&mut states, None, 0));
    assert!(!consume_color_metadata_clear(
        &mut states,
        Some(address),
        32
    ));
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

    Arc::make_mut(&mut draw.context_registers)
        .iter_mut()
        .find(|register| register.offset == DB_Z_INFO)
        .expect("depth info")
        .value |= 1 << 20;
    let error = decode_depth_target(&draw).expect_err("nonzero tile index is unsupported");
    assert!(error.contains("tile-mode index 1"), "{error}");
}

#[test]
fn depth_target_view_selects_the_backing_array_slice() {
    let address = 0x0070_69b9_0000;
    let mut draw = indexed_draw(0, 0, 0);
    append_depth_target_registers(&mut draw, address, 1_024, 1_024, 0xa000_0001, 0x66);
    push_gpu_register(
        &mut draw.context_registers,
        GpuRegister {
            offset: DB_DEPTH_VIEW,
            value: 3 | (3 << 13),
        },
    );
    Arc::make_mut(&mut draw.context_registers).sort_unstable_by_key(|register| register.offset);

    let depth = decode_depth_target(&draw)
        .expect("single depth slice is supported")
        .expect("enabled depth target");
    let slice_bytes = 2 * 1_024 * 1_024;
    assert_eq!(depth.texture.address, address + 3 * slice_bytes);
    assert_eq!(depth.layer_count, 1);
    assert_eq!(guest_texture_byte_count(depth.texture), Some(slice_bytes));

    Arc::make_mut(&mut draw.context_registers)
        .iter_mut()
        .find(|register| register.offset == DB_DEPTH_VIEW)
        .expect("depth view")
        .value = 2 | (4 << 13);
    let depth = decode_depth_target(&draw)
        .expect("multi-slice depth views are supported")
        .expect("enabled layered depth target");
    assert_eq!(depth.texture.address, address + 2 * slice_bytes);
    assert_eq!(depth.layer_count, 3);
    assert_eq!(guest_texture_byte_count(depth.texture), Some(slice_bytes));
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
    assert!(
        guest_texture_descriptors_share_depth_sample_storage(rendered, sampled, 0xfac),
        "R32 depth views with XYZW identity still only need .x from the native Z plane"
    );

    // Poppy title: depth target pitch is RDNA2-padded while the R32 SRD reports visible width.
    let rendered = GuestTextureDescriptor {
        address: 0x9023_2c00_00,
        width: 2_848,
        height: 1_600,
        pitch: 2_944,
        tile_mode: 24,
        pixel_format: MTLPixelFormat::Depth32Float,
        bytes_per_pixel: 4,
    };
    let sampled = GuestTextureDescriptor {
        pitch: 2_848,
        pixel_format: MTLPixelFormat::R32Float,
        ..rendered
    };
    assert!(
        guest_texture_descriptors_share_depth_sample_storage(rendered, sampled, 0x204),
        "padded depth pitch vs visible-width R32 SRD must still bind the native resident"
    );
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
    assert_eq!(
        initial_depth_clear_for_target(depth),
        0.0,
        "an implicit first-use clear must not consume stale DB_DEPTH_CLEAR state"
    );

    Arc::make_mut(&mut draw.context_registers)
        .iter_mut()
        .find(|register| register.offset == DB_DEPTH_CONTROL)
        .expect("depth control")
        .value = 0x771;
    // Latched STENCIL_ENABLE with Z ops off is not a depth attachment. Color MRTs must still
    // build pipelines; true stencil-only (no color either) is refused at graphics_pipeline_key.
    assert_eq!(
        decode_depth_target(&draw).expect("latched stencil without Z ops is not a depth target"),
        None
    );
    assert!(
        graphics_pipeline_key(&draw)
            .expect_err("stencil-only without color or depth must refuse")
            .contains("stencil is enabled without an active depth/stencil surface")
    );

    Arc::make_mut(&mut draw.context_registers)
        .iter_mut()
        .find(|register| register.offset == DB_DEPTH_CONTROL)
        .expect("depth control")
        .value = 0x66;
    Arc::make_mut(&mut draw.context_registers)
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
fn ignores_stale_stencil_operations_when_stencil_testing_is_disabled() {
    let mut draw = indexed_draw(0, 0, 0);
    append_depth_target_registers(&mut draw, 0x0070_1412_0000, 1_920, 1_080, 0xa000_0003, 0x76);
    push_gpu_register(
        &mut draw.context_registers,
        GpuRegister {
            offset: DB_STENCIL_CONTROL,
            value: 0x00aa_aaaa,
        },
    );
    Arc::make_mut(&mut draw.context_registers).sort_by_key(|register| register.offset);

    let depth = decode_depth_target(&draw)
        .expect("disabled stencil ignores stale unrepresentable ops")
        .expect("depth target remains active");
    assert!(!depth.stencil.enabled);
}

#[test]
fn refuses_distinct_stencil_replace_operation_value_instead_of_using_test_value() {
    let mut draw = indexed_draw(0, 0, 0);
    // Stencil and depth enabled; both compare functions are Always.
    append_depth_target_registers(
        &mut draw,
        0x0070_1412_0000,
        1_920,
        1_080,
        0xa000_0003,
        0x777,
    );
    extend_gpu_registers(
        &mut draw.context_registers,
        [
            GpuRegister {
                offset: DB_STENCIL_CONTROL,
                // Front-face depth-pass operation is AMD REPLACE_OP.
                value: 4 << 4,
            },
            GpuRegister {
                offset: DB_STENCILREFMASK,
                // STENCILOPVAL=0xaa, STENCILTESTVAL=0x55.
                value: 0xaa_ff_ff55,
            },
        ],
    );
    Arc::make_mut(&mut draw.context_registers).sort_by_key(|register| register.offset);

    let error =
        decode_depth_target(&draw).expect_err("independent replace value is not Metal-exact");
    assert!(error.contains("REPLACE_OP value 0xaa differs from comparison value 0x55"));
}

#[test]
fn decodes_explicit_guest_depth_clear_without_depth_test() {
    let mut draw = indexed_draw(0, 0, 0);
    append_depth_target_registers(&mut draw, 0x0070_1412_0000, 1_920, 1_080, 0xa000_0003, 0);
    extend_gpu_registers(
        &mut draw.context_registers,
        [
            GpuRegister {
                offset: DB_RENDER_CONTROL,
                value: 1,
            },
            GpuRegister {
                offset: DB_DEPTH_CLEAR,
                value: 0.25_f32.to_bits(),
            },
        ],
    );
    Arc::make_mut(&mut draw.context_registers).sort_unstable_by_key(|register| register.offset);

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
    push_gpu_register(
        &mut draw.context_registers,
        GpuRegister {
            offset: DB_DEPTH_CLEAR,
            value: f32::NAN.to_bits(),
        },
    );
    Arc::make_mut(&mut draw.context_registers).sort_unstable_by_key(|register| register.offset);

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
fn absent_pa_su_sc_mode_cntl_uses_sdk_ccw_default() {
    let draw = indexed_draw(0, 0, 0);
    assert_eq!(
        decode_raster_state(&draw).expect("default raster state"),
        DecodedRasterState {
            cull_mode: MTLCullMode::None,
            front_winding: MTLWinding::CounterClockwise,
            fill_mode: MTLTriangleFillMode::Fill,
            depth_bias: DecodedDepthBias::default(),
        }
    );
}

#[test]
fn absent_viewport_registers_use_agc_unit_scale_and_zero_depth_range() {
    let draw = indexed_draw(0, 0, 0);
    let viewport = decode_viewport(&draw, 1_920, 1_080, GuestScissor::Full);
    assert_eq!(viewport.originX, -1.0);
    assert_eq!(viewport.originY, -1.0);
    assert_eq!(viewport.width, 2.0);
    assert_eq!(viewport.height, 2.0);
    assert_eq!(viewport.znear, 0.0);
    assert_eq!(viewport.zfar, 0.0);
}

#[test]
fn decodes_polygon_offset_depth_bias_in_sdk_sixteenth_units() {
    let mut draw = indexed_draw(0, 0, 0);
    extend_gpu_registers(
        &mut draw.context_registers,
        [
            GpuRegister {
                offset: 0x2e0,
                value: f32::to_bits(16.0),
            },
            GpuRegister {
                offset: 0x2e1,
                value: f32::to_bits(2.0),
            },
            GpuRegister {
                offset: 0x2df,
                value: f32::to_bits(0.25),
            },
        ],
    );
    Arc::make_mut(&mut draw.context_registers).sort_unstable_by_key(|register| register.offset);
    let raster = decode_raster_state(&draw).expect("polygon offset raster state");
    assert_eq!(raster.depth_bias.slope_scale, 1.0);
    assert_eq!(raster.depth_bias.constant, 2.0);
    assert_eq!(raster.depth_bias.clamp, 0.25);
}

#[test]
fn blend_bypass_in_cb_color_info_disables_metal_blend() {
    let mut registers = Vec::new();
    append_color_target_registers(&mut registers, 0, 0x0070_1000_0000, 64, 64, 10, 0, 27);
    registers.push(GpuRegister {
        offset: CB_TARGET_MASK,
        value: 0xf,
    });
    // CB_COLOR_INFO bit 16 = BlendBypass.
    registers
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_INFO)
        .expect("MRT0 info")
        .value |= 1 << 16;
    registers.sort_unstable_by_key(|register| register.offset);
    let targets =
        decode_render_targets_from_registers(&registers, 0xf).expect("blend-bypass color target");
    assert!(targets[0].blend_bypass);
}

#[test]
fn derives_ngg_subgroup_limits_from_uc_ge_cntl_when_cx_sizing_absent() {
    // UcGeCntl: PRIM_GRP_SIZE=8, VERT_GRP_SIZE=16. With no API GS, output amplification is
    // exactly one; VERT_GRP_SIZE is an input limit and must never be reused as PRIM_AMP_FACTOR.
    let limits = decode_ngg_subgroup_limits_with_uc(
        &[
            GpuRegister {
                offset: 0x2ce,
                // Stale VGT_GS_MAX_VERT_OUT must be inert when GS_EN is clear.
                value: 1,
            },
            GpuRegister {
                offset: 0x2d5,
                value: 0x2000,
            },
        ],
        &[GpuRegister {
            offset: 0x25b,
            value: 8 | (16 << 9),
        }],
    )
    .expect("UcGeCntl-derived NGG is representable")
    .expect("UcGeCntl-derived NGG is present");
    assert_eq!(limits.es_vertices_per_subgroup, 16);
    assert_eq!(limits.gs_primitives_per_subgroup, 8);
    assert_eq!(limits.gs_max_vertices_out, 0);
    assert_eq!(limits.primitive_amplification_factor, 1);
    assert_eq!(limits.max_output_primitives_per_subgroup, 8);
    assert!(limits.uses_fixed_allocation_fallback());
    assert_eq!(limits.meshlet_vertex_limit(), 64);
    assert_eq!(limits.meshlet_primitive_limit(), 64);
    assert_eq!(limits.metal_threadgroup_threads(), 64);
}

#[test]
fn derives_gs_amplification_from_gs_max_vert_out_not_uc_vertex_group_size() {
    let limits = decode_ngg_subgroup_limits_with_uc(
        &[
            GpuRegister {
                offset: 0x2ce,
                value: 24,
            },
            GpuRegister {
                offset: 0x2d5,
                // ES_EN plus GS_EN.
                value: 0x2020,
            },
        ],
        &[GpuRegister {
            offset: 0x25b,
            value: 4 | (16 << 9),
        }],
    )
    .expect("GS amplification is proven by VGT_GS_MAX_VERT_OUT")
    .expect("UcGeCntl-derived NGG is present");

    assert_eq!(limits.es_vertices_per_subgroup, 16);
    assert_eq!(limits.gs_primitives_per_subgroup, 4);
    assert_eq!(limits.gs_max_vertices_out, 24);
    assert_eq!(limits.primitive_amplification_factor, 24);
    assert_eq!(limits.max_output_vertices_per_subgroup, 96);
    assert_eq!(limits.max_output_primitives_per_subgroup, 96);
    assert!(!limits.uses_fixed_allocation_fallback());
}

#[test]
fn refuses_gs_amplification_when_only_uc_input_group_sizes_are_known() {
    let error = decode_ngg_subgroup_limits_with_uc(
        &[GpuRegister {
            offset: 0x2d5,
            value: 0x2020,
        }],
        &[GpuRegister {
            offset: 0x25b,
            value: 4 | (16 << 9),
        }],
    )
    .expect_err("UcGeCntl input sizes do not prove an API-GS output count");
    assert!(error.contains("missing VGT_GS_MAX_VERT_OUT"));
}

#[test]
fn decodes_graphics_scratch_ring_and_export_wave_width_from_agc_registers() {
    // SDK 10 CxGraphicsScratchRingSize (Cx 0x1ba): WaveSize[24:12] counts 1 KiB per wave, so two
    // chunks is 2 KiB per architectural wave — 64 bytes per lane at wave32, 32 at wave64.
    // CxVgtShaderStagesEn (Cx 0x2d5) GsWaveSize is bit 22: clear = k64, set = k32.
    let wave64 = decode_graphics_wave_state(&[
        GpuRegister {
            offset: 0x1ba,
            value: 2 << 12,
        },
        GpuRegister {
            offset: 0x2d5,
            value: 0x2000,
        },
    ]);
    assert_eq!(wave64.scratch_bytes_per_wave, 2048);
    assert_eq!(wave64.export_lane_count, Some(64));
    assert_eq!(wave64.scratch_bytes_per_lane(64), 32);

    let wave32 = decode_graphics_wave_state(&[
        GpuRegister {
            offset: 0x1ba,
            value: 2 << 12,
        },
        GpuRegister {
            offset: 0x2d5,
            value: 0x2000 | 0x0040_0000,
        },
    ]);
    assert_eq!(wave32.export_lane_count, Some(32));
    assert_eq!(wave32.scratch_bytes_per_lane(32), 64);

    // The pixel wave width has no SDK register. Dividing by the narrower wave can only
    // over-allocate the lane-private array; halving it would clamp real spills to zero.
    assert_eq!(wave64.pixel_scratch_bytes_per_lane(), 64);

    // An unprogrammed CxVgtShaderStagesEn is not a wave32 claim, and an unprogrammed temp ring
    // must stay zero so a SCRATCH instruction refuses instead of guessing an extent.
    let absent = decode_graphics_wave_state(&[]);
    assert_eq!(absent.scratch_bytes_per_wave, 0);
    assert_eq!(absent.export_lane_count, None);
}

#[test]
fn decodes_observed_clockwise_back_face_culling() {
    let mut draw = indexed_draw(0, 0, 0);
    push_gpu_register(
        &mut draw.context_registers,
        GpuRegister {
            offset: PA_SU_SC_MODE_CNTL,
            value: 0x246,
        },
    );

    assert_eq!(
        decode_raster_state(&draw).expect("observed raster state"),
        DecodedRasterState {
            cull_mode: MTLCullMode::Back,
            front_winding: MTLWinding::Clockwise,
            fill_mode: MTLTriangleFillMode::Fill,
            depth_bias: DecodedDepthBias::default(),
        }
    );
}

#[test]
fn negative_height_viewport_keeps_sdk_front_face_for_metal() {
    let mut draw = indexed_draw(0, 0, 0);
    extend_gpu_registers(
        &mut draw.context_registers,
        [
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
        ],
    );
    Arc::make_mut(&mut draw.context_registers).sort_by_key(|register| register.offset);

    assert_eq!(
        decode_raster_state(&draw).expect("negative-height viewport raster state"),
        DecodedRasterState {
            cull_mode: MTLCullMode::Back,
            front_winding: MTLWinding::Clockwise,
            fill_mode: MTLTriangleFillMode::Fill,
            depth_bias: DecodedDepthBias::default(),
        }
    );
    let viewport = decode_viewport(&draw, 1_920, 1_080, GuestScissor::Full);
    assert_eq!(viewport.originY, 0.0);
    assert_eq!(viewport.height, 1_080.0);
    // SCE AGC CxPrimitiveSetup::FrontFace::kCw is bit 2. Metal's positive viewport keeps that
    // SDK-facing value; callers must not compensate for the guest's negative Y scale again.
    assert_eq!(
        decode_raster_state(&draw)
            .expect("negative-height viewport front face")
            .front_winding,
        MTLWinding::Clockwise,
    );
}

#[test]
fn decodes_dual_polygon_line_mode_and_rejects_unrepresentable_states() {
    let mut draw = indexed_draw(0, 0, 0);
    push_gpu_register(
        &mut draw.context_registers,
        GpuRegister {
            offset: PA_SU_SC_MODE_CNTL,
            value: (1 << 2) | (1 << 3) | (1 << 5) | (1 << 8),
        },
    );
    assert_eq!(
        decode_raster_state(&draw)
            .expect("matching front/back line mode")
            .fill_mode,
        MTLTriangleFillMode::Lines
    );

    Arc::make_mut(&mut draw.context_registers)[0].value |= 1 | 2;
    assert!(
        decode_raster_state(&draw)
            .expect_err("Metal cannot cull both orientations")
            .contains("simultaneous front- and back-face")
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
        span: super::ResourceSpan::for_texture(descriptor),
        layer_count: 1,
        htile_address: None,
        compare: MTLCompareFunction::Less,
        test_enabled: true,
        write_enabled: false,
        clear_enabled: false,
        clear_depth: 1.0,
        sample_count: 1,
        stencil: DecodedStencilState::default(),
    };

    let promoted = presenter
        .ensure_depth_render_target_texture(&device, depth, 128, 96)
        .expect("promote depth target")
        .texture;
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
fn promotes_layered_depth_target_without_collapsing_its_slices() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let allocation_size = 8 * GUEST_PAGE_SIZE;
    let mut memory = GuestAddressSpace::new(16 * GUEST_PAGE_SIZE).expect("guest address space");
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
        .expect("guest layered depth mapping");
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
        span: super::ResourceSpan::for_texture(descriptor),
        layer_count: 3,
        htile_address: None,
        compare: MTLCompareFunction::Less,
        test_enabled: true,
        write_enabled: false,
        clear_enabled: false,
        clear_depth: 1.0,
        sample_count: 1,
        stencil: DecodedStencilState::default(),
    };

    let promoted = presenter
        .ensure_depth_render_target_texture(&device, depth, 128, 96)
        .expect("promote layered depth target")
        .texture;
    assert_eq!(promoted.texture_type(), MTLTextureType::D2Array);
    assert_eq!(promoted.array_length(), 3);
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
    let bytes = read_texture_array_linear(&device, &queue, &promoted, promoted_descriptor)
        .expect("read promoted layered depth target");
    let face_bytes = 128 * 96 * 4;
    for layer in 0..3 {
        let sample = |x: usize, y: usize| {
            let offset = layer * face_bytes + (y * 128 + x) * 4;
            f32::from_ne_bytes(bytes[offset..offset + 4].try_into().expect("depth sample"))
        };
        assert_eq!(sample(0, 0), 0.0);
        assert_eq!(sample(100, 50), 1.0);
    }
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
fn recovers_32_bit_index_buffers_mislabeled_as_16_bit() {
    // u32 [0,1,2,2,1,3] read as u16 is [0,0,1,0,2,0,2,0,1,0,3,0]. Poppy/UE often leave
    // VGT_INDEX_TYPE at the 16-bit CLEAR default while the buffer is 32-bit.
    let indices_32 = [0_u32, 1, 2, 2, 1, 3];
    let draw = indexed_draw(indices_32.as_ptr().addr() as u64, 0, 6);
    assert_eq!(
        read_draw_indices(&draw).as_deref(),
        Some([0, 1, 2, 2, 1, 3].as_slice())
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
fn gfx10_resource_level_one_is_not_a_render_target_mip() {
    let target = decode_render_target_state(
        0,
        0x00ef_c200_0000,
        0,
        0x0000_8024,
        0,
        ((3_840 - 1) << 14) | (2_160 - 1),
        0x4dc6_c000,
        0xf,
    )
    .expect("Poppy's GFX10 scanout resource state is representable");

    assert_eq!(target.texture.address, 0x00ef_c200_0000);
    assert_eq!(
        (target.texture.width, target.texture.height),
        (3_840, 2_160)
    );
    assert_eq!(target.texture.pixel_format, MTLPixelFormat::RGB10A2Unorm);
    assert_eq!(target.texture.tile_mode, 27);
    assert_eq!((target.mip_level, target.mip_level_count), (0, 1));
    assert_eq!((target.texture_type, target.layer_count), (9, 1));
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

    registers
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_ATTRIB3)
        .expect("color attrib3 exists")
        .value |= 2 << 27;
    assert!(
        decode_render_targets_from_registers(&registers, 0xf)
            .expect_err("undefined resource levels cannot be flattened")
            .contains("invalid GFX10 resource level 2")
    );
    registers
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_ATTRIB3)
        .expect("color attrib3 exists")
        .value = (27 << 14) | (1 << 24) | (1 << 31);
    assert!(
        decode_render_targets_from_registers(&registers, 0xf)
            .expect_err("Trinity VRS CMASK behavior cannot be silently dropped")
            .contains("Trinity VRS CMASK hints")
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
        .value = 63 | (27 << 14) | (2 << 24) | (1 << 27);
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
        initial_render_target_clear_color(&ordinary, &registers, false, false).is_none(),
        "ordinary first-use targets preserve their existing load behavior"
    );

    let mut dcc = ordinary;
    dcc.dcc_enabled = true;
    let discarded = initial_render_target_clear_color(&dcc, &registers, false, false)
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
        initial_render_target_clear_color(&dcc, &registers, true, false).is_none(),
        "resident DCC targets keep their accumulated Metal contents"
    );

    dcc.fast_clear_enabled = true;
    let programmed = initial_render_target_clear_color(&dcc, &registers, false, true)
        .expect("programmed DCC fast clear");
    assert!((programmed.red - 16.0 / 255.0).abs() < 0.000_01);
    assert!((programmed.green - 32.0 / 255.0).abs() < 0.000_01);
    assert!((programmed.blue - 64.0 / 255.0).abs() < 0.000_01);
    assert!((programmed.alpha - 128.0 / 255.0).abs() < 0.000_01);

    dcc.dcc_enabled = false;
    let ordinary_fast_clear = initial_render_target_clear_color(&dcc, &registers, false, true)
        .expect("ordinary fast clear behavior is preserved");
    assert_eq!(ordinary_fast_clear.red, programmed.red);
    assert_eq!(ordinary_fast_clear.green, programmed.green);
    assert_eq!(ordinary_fast_clear.blue, programmed.blue);
    assert_eq!(ordinary_fast_clear.alpha, programmed.alpha);
    assert!(
        initial_render_target_clear_color(&dcc, &registers, false, false).is_none(),
        "consumed CMASK must not clear an uncompressed first-use target again"
    );
    assert_eq!(
        render_target_cmask_address(&dcc, &registers).unwrap(),
        Some(0x70_2234_5600),
    );
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
    assert!(is_transparent_premultiplied_fill_blend(&blend));
}

#[test]
fn treats_chowdren_transparent_premultiplied_fills_as_overwrites() {
    // Guest CB_BLEND_CONTROL for enabled One / OneMinusSrcAlpha / Add (shared alpha).
    let blend = decode_metal_blend_attachment((1 << 30) | 1 | (5 << 8)).expect("premult blend");
    let mut targets = vec![RenderTargetPipelineState {
        guest_slot: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        output_kind: PixelOutputKind::Float,
        export_format: PixelExportFormat::Abgr32,
        write_mask: 0xf,
        blend,
    }];
    let export = empty_shader_evaluation();
    let pixel = Arc::new(ShaderEvaluation {
        initial_scalar_registers: vec![0, 0, 0, 0].into(),
        scalar_registers: vec![0, 0, 0, 0].into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: Vec::new(),
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    });

    apply_transparent_premultiplied_fill_clear(&mut targets, &export, &pixel);
    assert_eq!(targets[0].blend, disabled_metal_blend_attachment());

    targets[0].blend = blend;
    let tinted = Arc::new(ShaderEvaluation {
        initial_scalar_registers: vec![0, 0, 0, 0x3f80_0000].into(),
        ..(*pixel).clone()
    });
    apply_transparent_premultiplied_fill_clear(&mut targets, &export, &tinted);
    assert!(
        targets[0].blend.enabled,
        "non-zero fill color must keep premultiplied blending"
    );

    targets[0].blend = blend;
    let with_vertex_fetch = Arc::new(ShaderEvaluation {
        vertex_inputs: vec![remu_shader::VertexInputBinding {
            pc: 0,
            location: 0,
            component_count: 4,
            data_format: 14,
            number_format: 7,
            base_address: 0x1000,
            stride: 16,
            offset_bytes: 0,
            data: vec![0; 16],
        }],
        ..(*export).clone()
    });
    apply_transparent_premultiplied_fill_clear(&mut targets, &with_vertex_fetch, &pixel);
    assert!(
        targets[0].blend.enabled,
        "vertex-fetched draws must not be rewritten as clears"
    );
}

#[test]
fn does_not_rewrite_straight_alpha_transparent_fills_as_clears() {
    // ENABLE|SEPARATE_ALPHA|SrcAlpha/OneMinusSrcAlpha — also a no-op for RGBA=0, but rewriting
    // those draws as overwrites erases scene content that Chowdren meant to leave alone.
    let blend = decode_metal_blend_attachment(0x6504_0504).expect("straight-alpha blend");
    assert!(!is_transparent_premultiplied_fill_blend(&blend));

    let mut targets = vec![RenderTargetPipelineState {
        guest_slot: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        output_kind: PixelOutputKind::Float,
        export_format: PixelExportFormat::Abgr32,
        write_mask: 0xf,
        blend,
    }];
    let export = empty_shader_evaluation();
    let pixel = Arc::new(ShaderEvaluation {
        initial_scalar_registers: vec![0, 0, 0, 0].into(),
        scalar_registers: vec![0, 0, 0, 0].into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: Vec::new(),
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    });

    apply_transparent_premultiplied_fill_clear(&mut targets, &export, &pixel);
    assert!(
        targets[0].blend.enabled,
        "straight-alpha transparent fills must remain no-ops"
    );
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
fn decodes_captured_ngg_subgroup_limits_and_derives_ninety_six_threads() {
    let limits = decode_ngg_subgroup_limits(&[
        GpuRegister {
            offset: 0x1ff,
            value: 96,
        },
        GpuRegister {
            offset: 0x291,
            value: 0x0100_2018,
        },
        GpuRegister {
            offset: 0x29b,
            value: 2,
        },
        GpuRegister {
            offset: 0x2ce,
            value: 24,
        },
        GpuRegister {
            offset: 0x2d3,
            value: 24,
        },
        GpuRegister {
            offset: 0x2d5,
            // ES_EN plus GS_EN: VGT_GS_MAX_VERT_OUT is active for this captured GS stage.
            value: 0x2020,
        },
        GpuRegister {
            offset: 0x2e4,
            value: 1 << 2,
        },
    ])
    .expect("captured NGG registers are representable")
    .expect("captured NGG registers are present");

    assert_eq!(limits.es_vertices_per_subgroup, 24);
    assert_eq!(limits.gs_primitives_per_subgroup, 4);
    assert_eq!(limits.gs_instance_primitives_per_subgroup, 4);
    assert_eq!(limits.gs_instance_count, 1);
    assert_eq!(limits.gs_max_vertices_out, 24);
    assert_eq!(limits.max_output_vertices_per_subgroup, 96);
    assert_eq!(limits.primitive_amplification_factor, 24);
    assert_eq!(limits.max_output_primitives_per_subgroup, 96);
    assert_eq!(
        limits.output_primitive_type,
        Some(NggOutputPrimitiveType::TriangleStrip)
    );
    assert_eq!(limits.threadgroup_threads, 96);
}

#[test]
fn fixed_ngg_allocation_is_the_default_for_all_program_shapes() {
    let no_gs = Program::new(0x1000, Vec::new());
    assert!(ngg_fixed_allocation_fallback_for_program(&no_gs));

    let front = Program::new(
        0x2000,
        vec![Instruction {
            pc: 0,
            encoding: ShaderEncoding::Sop1,
            opcode: Opcode::Known("SSetpcB64"),
            words: vec![0xbe80_2006],
            sources: vec![Operand::scalar(6)],
            destinations: Vec::new(),
            control: None,
        }],
    );
    let continuation = Program::new(
        0x3000,
        vec![Instruction {
            pc: 0,
            encoding: ShaderEncoding::Sopp,
            opcode: Opcode::END_PROGRAM,
            words: vec![0xbf81_0000],
            sources: Vec::new(),
            destinations: Vec::new(),
            control: None,
        }],
    );
    let linked = link_program_continuation(&front, &continuation).expect("linked API GS fixture");
    assert!(ngg_fixed_allocation_fallback_for_program(&linked));
}

#[test]
fn captures_proven_ngg_compile_registers_from_the_live_draw_state() {
    // The merged export stage's PGM_RSRC1 sits two registers before its user-data base. The
    // geometry limits are context registers and GE_MAX_OUTPUT_PER_SUBGROUP—not
    // GE_NGG_SUBGRP_CNTL—owns the SDK's ten-bit post-amplification vertex cap.
    // PGM_RSRC2 LDS field 0x40 → SDK getLdsSize = 0x40 << 7 dwords = 32768 bytes.
    let registers = decode_ngg_registers_from_state(
        &[
            GpuRegister {
                offset: 0x08a,
                value: 0x25,
            },
            GpuRegister {
                offset: 0x08b,
                value: 0x40 << 19,
            },
        ],
        &[
            GpuRegister {
                offset: 0x1ff,
                value: 96,
            },
            GpuRegister {
                offset: 0x2ab,
                value: 4,
            },
            GpuRegister {
                offset: 0x2ce,
                value: 24,
            },
            GpuRegister {
                offset: 0x2d3,
                value: 7,
            },
            GpuRegister {
                offset: 0x2d5,
                value: 0x0200_0000,
            },
        ],
        0x08c,
    )
    .expect("the complete live register set proves NGG sizing");

    assert_eq!(registers.pgm_rsrc1, 0x25);
    assert_eq!(registers.pgm_rsrc2, 0x40 << 19);
    assert_eq!(registers.lds_size_bytes(), 32768);
    assert_eq!(registers.gs_max_vert_out, 24);
    assert_eq!(registers.ge_max_output_per_subgroup, 96);
    assert_eq!(registers.esgs_ring_itemsize, 4);
    assert!(registers.primgen_passthrough);
    assert_eq!(registers.max_verts_per_subgroup(), 96);
}

#[test]
fn decodes_passthrough_ngg_subgroup_without_an_api_gs_output_limit() {
    let limits = decode_ngg_subgroup_limits(&[
        GpuRegister {
            offset: 0x1ff,
            value: 64,
        },
        GpuRegister {
            offset: 0x291,
            value: 64 | (64 << 11) | (64 << 22),
        },
        GpuRegister {
            offset: 0x29b,
            value: 2,
        },
        GpuRegister {
            offset: 0x2ce,
            value: 0,
        },
        GpuRegister {
            offset: 0x2d3,
            value: 1,
        },
        GpuRegister {
            offset: 0x2e4,
            value: 0,
        },
    ])
    .expect("passthrough NGG state is representable")
    .expect("passthrough NGG registers are present");

    assert_eq!(limits.es_vertices_per_subgroup, 64);
    assert_eq!(limits.gs_primitives_per_subgroup, 64);
    assert_eq!(limits.gs_instance_primitives_per_subgroup, 64);
    assert_eq!(limits.gs_instance_count, 1);
    assert_eq!(
        limits.gs_max_vertices_out, 0,
        "zero preserves the inactive API-GS clamp"
    );
    assert_eq!(limits.max_output_vertices_per_subgroup, 64);
    assert_eq!(limits.primitive_amplification_factor, 1);
    assert_eq!(limits.max_output_primitives_per_subgroup, 64);
    assert_eq!(
        limits.output_primitive_type,
        Some(NggOutputPrimitiveType::TriangleStrip)
    );
    assert_eq!(limits.threadgroup_threads, 64);
}

#[test]
fn ordinary_agc_geometry_defaults_do_not_activate_ngg() {
    assert_eq!(
        decode_ngg_subgroup_limits(&[
            GpuRegister {
                offset: 0x1ff,
                value: 0,
            },
            GpuRegister {
                offset: 0x29b,
                value: 2,
            },
            GpuRegister {
                offset: 0x2ce,
                value: 0x400,
            },
        ]),
        Ok(None)
    );
}

#[test]
fn decodes_prospero_rectangle_gs_output_primitive_types() {
    for outprim in [3_u32, 4_u32] {
        let limits = decode_ngg_subgroup_limits(&[
            GpuRegister {
                offset: 0x1ff,
                value: 64,
            },
            GpuRegister {
                offset: 0x291,
                value: 64 | (64 << 11) | (64 << 22),
            },
            GpuRegister {
                offset: 0x29b,
                value: outprim,
            },
            GpuRegister {
                offset: 0x2ce,
                value: 0,
            },
            GpuRegister {
                offset: 0x2d3,
                value: 1,
            },
            GpuRegister {
                offset: 0x2e4,
                value: 0,
            },
        ])
        .unwrap_or_else(|error| panic!("OUTPRIM {outprim} should decode: {error}"))
        .expect("NGG registers are present");
        assert_eq!(
            limits.output_primitive_type,
            Some(NggOutputPrimitiveType::RectangleList),
            "Prospero k2dRectangle=3 and kRectList=4 both map to RectangleList"
        );
    }
}

#[test]
fn rejects_ngg_subgroups_that_exceed_metal_capacity() {
    let error = NggSubgroupLimits::new(
        24,
        16,
        16,
        24,
        257,
        16,
        Some(NggOutputPrimitiveType::TriangleStrip),
        Some(1),
    )
    .expect_err("257 guest outputs cannot be silently clamped to Metal's limit");
    assert!(error.contains("exceed Metal's limit"));
}

#[test]
fn still_enforces_a_programmed_api_gs_output_limit() {
    let error = NggSubgroupLimits::new(
        64,
        64,
        64,
        1,
        65,
        1,
        Some(NggOutputPrimitiveType::TriangleStrip),
        Some(1),
    )
    .expect_err("a programmed GS output clamp must bound subgroup vertices");

    assert!(error.contains("VGT_GS_MAX_VERT_OUT(1)"));
}

#[test]
fn packs_ngg_inputs_across_more_than_one_guest_wave() {
    let limits = NggSubgroupLimits::new(
        96,
        32,
        32,
        3,
        96,
        1,
        Some(NggOutputPrimitiveType::TriangleStrip),
        Some(1),
    )
    .expect("96-thread NGG subgroup is representable");
    let indices = (0_u32..66).collect::<Vec<_>>();
    let prepared = prepare_ngg_meshlets(&indices, limits);

    assert_eq!(limits.threadgroup_threads, 96);
    assert_eq!(prepared.group_count, 1);
    assert_eq!(prepared.vertex_references, 66);
    assert_eq!(prepared.primitive_count, 22);
    assert_eq!(prepared.words[0], 66 | (22 << 8));
}

#[test]
fn no_gs_ngg_meshlets_use_the_known_good_fixed_wave64_allocation() {
    let limits = NggSubgroupLimits::new(
        128,
        128,
        128,
        0,
        128,
        1,
        Some(NggOutputPrimitiveType::TriangleStrip),
        Some(1),
    )
    .expect("no-GS passthrough limits are representable");
    let indices = (0_u32..66).collect::<Vec<_>>();
    let prepared = prepare_ngg_meshlets(&indices, limits);

    assert!(limits.uses_fixed_allocation_fallback());
    assert_eq!(limits.threadgroup_threads, 128);
    assert_eq!(limits.metal_threadgroup_threads(), 64);
    assert_eq!(limits.meshlet_words_per_group(), 130);
    assert_eq!(prepared.group_count, 2);
    assert_eq!(prepared.words[0], 63 | (21 << 8));
    assert_eq!(prepared.words[130], 3 | (1 << 8));
}

#[test]
fn packs_ngg_meshlets_with_shared_vertex_indices() {
    let limits = NggSubgroupLimits::new(
        24,
        4,
        4,
        24,
        96,
        24,
        Some(NggOutputPrimitiveType::TriangleStrip),
        Some(1),
    )
    .expect("captured Silent Hill NGG limits");
    let prepared = prepare_ngg_meshlets(&[10, 11, 12, 12, 11, 13], limits);
    assert_eq!(prepared.group_count, 1);
    assert_eq!(prepared.vertex_references, 4);
    assert_eq!(prepared.primitive_count, 2);
    assert_eq!(prepared.words.len(), NGG_MESHLET_WORDS_PER_GROUP);
    assert_eq!(prepared.words[0], 4 | (2 << 8));
    assert_eq!(prepared.words[1], 0);
    assert_eq!(&prepared.words[2..6], &[10, 11, 12, 13]);
    assert_eq!(prepared.words[258], 1 << 10 | 2 << 20);
    assert_eq!(prepared.words[259], 2 | 1 << 10 | 3 << 20);
}

#[test]
fn synthesizes_consecutive_ids_for_nonindexed_triangle_lists() {
    let mut draw = indexed_draw(0, 0, 7);
    draw.indexed = false;
    draw.primitive_type = 4;
    draw.index_offset = 10;
    draw.base_vertex = 91;
    let indices = prepare_nonindexed_ngg_indices(&draw).expect("list expands");
    assert_eq!(
        indices.as_ref(),
        &[10, 11, 12, 13, 14, 15],
        "partial trailing vertices are dropped; indexed base_vertex must not shift auto IDs"
    );
    assert_eq!(ngg_mesh_base_vertex(&draw), 0);
}

#[test]
fn preserves_nonindexed_first_vertex_and_alternates_triangle_strip_winding() {
    let mut draw = indexed_draw(0, 0, 5);
    draw.indexed = false;
    draw.primitive_type = 6;
    draw.index_offset = 7;
    draw.base_vertex = 91;
    let indices = prepare_nonindexed_ngg_indices(&draw).expect("strip expands");
    assert_eq!(indices.as_ref(), &[7, 8, 9, 9, 8, 10, 9, 10, 11]);
    assert_eq!(
        ngg_mesh_base_vertex(&draw),
        0,
        "stale indexed base state must not shift non-indexed auto-generated vertex IDs"
    );
    let limits = NggSubgroupLimits::new(
        24,
        4,
        4,
        24,
        96,
        24,
        Some(NggOutputPrimitiveType::TriangleStrip),
        Some(1),
    )
    .expect("captured Silent Hill NGG limits");
    let prepared = prepare_ngg_meshlets(indices.as_ref(), limits);
    assert_eq!(prepared.group_count, 1);
    assert_eq!(prepared.vertex_references, 5);
    assert_eq!(prepared.primitive_count, 3);
}

#[test]
fn expands_indexed_triangle_strips_for_ngg_meshlets() {
    let mut draw = indexed_draw(0, 0, 4);
    draw.indexed = true;
    draw.primitive_type = 6;
    let expanded = prepare_indexed_ngg_indices(
        &draw,
        [10, 11, 12, 13].into(),
        MTLPrimitiveType::TriangleStrip,
    )
    .expect("indexed strip expands");
    assert_eq!(
        expanded.as_ref(),
        &[10, 11, 12, 12, 11, 13],
        "a four-vertex indexed strip becomes two triangles with GFX10 winding"
    );
    let already_list = prepare_indexed_ngg_indices(
        &draw,
        [10, 11, 12, 12, 11, 13].into(),
        MTLPrimitiveType::Triangle,
    )
    .expect("restart-expanded strips pass through");
    assert_eq!(already_list.as_ref(), &[10, 11, 12, 12, 11, 13]);
}

#[test]
fn expands_indexed_point_lists_for_ngg_meshlets() {
    let mut draw = indexed_draw(0, 0, 2);
    draw.indexed = true;
    draw.primitive_type = 1;
    let expanded =
        prepare_indexed_ngg_indices(&draw, [7, 9].into(), MTLPrimitiveType::Point).expect("points");
    assert_eq!(
        expanded.as_ref(),
        &[7, 7, 7, 9, 9, 9],
        "each point becomes one meshlet input primitive with duplicated local slots"
    );
    let prepared = prepare_ngg_meshlets(
        expanded.as_ref(),
        NggSubgroupLimits::new(
            24,
            4,
            4,
            24,
            96,
            24,
            Some(NggOutputPrimitiveType::TriangleStrip),
            Some(1),
        )
        .expect("limits"),
    );
    assert_eq!(prepared.primitive_count, 2);
    assert_eq!(prepared.vertex_references, 2);
}

#[test]
fn preserves_indexed_mesh_base_vertex() {
    let mut draw = indexed_draw(0, 0, 3);
    draw.base_vertex = -7;
    assert_eq!(ngg_mesh_base_vertex(&draw), -7);
}

#[test]
fn empty_nonindexed_triangle_strip_has_no_meshlets() {
    let mut draw = indexed_draw(0, 0, 2);
    draw.indexed = false;
    draw.primitive_type = 6;
    let indices = prepare_nonindexed_ngg_indices(&draw).expect("short strip is empty");
    assert!(indices.is_empty());
    let limits = NggSubgroupLimits::new(
        24,
        4,
        4,
        24,
        96,
        24,
        Some(NggOutputPrimitiveType::TriangleStrip),
        Some(1),
    )
    .expect("captured Silent Hill NGG limits");
    assert_eq!(
        prepare_ngg_meshlets(indices.as_ref(), limits).group_count,
        0
    );
}

#[test]
fn mesh_dispatch_respects_guest_no_work_conditions() {
    assert!(ngg_mesh_dispatch_required(GuestScissor::Full, 1, true));
    assert!(!ngg_mesh_dispatch_required(GuestScissor::Empty, 1, true));
    assert!(!ngg_mesh_dispatch_required(GuestScissor::Full, 0, true));
    assert!(!ngg_mesh_dispatch_required(GuestScissor::Full, 1, false));
}

#[test]
fn rejects_nonindexed_triangle_strip_vertex_id_overflow_before_allocating() {
    let mut draw = indexed_draw(0, 0, 3);
    draw.indexed = false;
    draw.primitive_type = 6;
    draw.index_offset = u32::MAX;
    assert_eq!(
        prepare_nonindexed_ngg_indices(&draw),
        Err("triangle-strip vertex ID overflow".to_owned())
    );
}

#[test]
fn splits_ngg_meshlets_at_the_guest_programmed_vertex_limit() {
    let limits = NggSubgroupLimits::new(
        24,
        9,
        9,
        24,
        24,
        1,
        Some(NggOutputPrimitiveType::TriangleStrip),
        Some(1),
    )
    .expect("captured Silent Hill NGG limits");
    let indices = (0_u32..27).collect::<Vec<_>>();
    let prepared = prepare_ngg_meshlets(&indices, limits);
    assert_eq!(prepared.group_count, 2);
    assert_eq!(prepared.vertex_references, 27);
    assert_eq!(prepared.primitive_count, 9);
    assert_eq!(prepared.words[NGG_MESHLET_WORDS_PER_GROUP], 3 | (1 << 8));
    assert_eq!(prepared.words[NGG_MESHLET_WORDS_PER_GROUP + 1], 8);
}

#[test]
fn splits_ngg_meshlets_at_the_guest_programmed_primitive_limit() {
    let limits = NggSubgroupLimits::new(
        24,
        4,
        4,
        24,
        96,
        24,
        Some(NggOutputPrimitiveType::TriangleStrip),
        Some(1),
    )
    .expect("captured Silent Hill NGG limits");
    let mut indices = Vec::new();
    for primitive in 0_u32..5 {
        indices.extend_from_slice(&[0, 1, 2 + primitive]);
    }
    let prepared = prepare_ngg_meshlets(&indices, limits);
    assert_eq!(prepared.group_count, 2);
    assert_eq!(prepared.primitive_count, 5);
    assert_eq!(prepared.words[0] >> 8, 4);
    assert_eq!(prepared.words[NGG_MESHLET_WORDS_PER_GROUP] >> 8, 1);
    assert_eq!(prepared.words[NGG_MESHLET_WORDS_PER_GROUP + 1], 4);
}

#[test]
fn removes_triangle_list_primitive_restart_indices() {
    let mut draw = indexed_draw(0x10_000, 0, 7);
    draw.context_registers = vec![
        GpuRegister {
            offset: 0x103,
            value: u32::MAX,
        },
        GpuRegister {
            offset: 0x2a5,
            value: 1,
        },
    ]
    .into();
    let (indices, primitive) =
        prepare_indexed_primitive(&draw, [0, 1, 2, u32::from(u16::MAX), 3, 4, 5].into())
            .expect("triangle-list primitive restart");
    assert_eq!(indices.as_ref(), &[0, 1, 2, 3, 4, 5]);
    assert_eq!(primitive, MTLPrimitiveType::Triangle);
}

#[test]
fn expands_restarted_triangle_strips_to_triangle_lists() {
    let mut draw = indexed_draw(0x10_000, 0, 8);
    draw.primitive_type = 6;
    draw.context_registers = vec![
        GpuRegister {
            offset: 0x103,
            value: u32::MAX,
        },
        GpuRegister {
            offset: 0x2a5,
            value: 1,
        },
    ]
    .into();
    let (indices, primitive) =
        prepare_indexed_primitive(&draw, [0, 1, 2, 3, u32::from(u16::MAX), 4, 5, 6].into())
            .expect("triangle-strip primitive restart");
    assert_eq!(indices.as_ref(), &[0, 1, 2, 2, 1, 3, 4, 5, 6]);
    assert_eq!(primitive, MTLPrimitiveType::Triangle);
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
    ]
    .into();
    Arc::make_mut(&mut draw.context_registers).sort_by_key(|register| register.offset);

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
