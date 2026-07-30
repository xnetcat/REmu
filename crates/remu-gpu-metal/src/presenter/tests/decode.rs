// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use metal::{
    CompileOptions, Device, MTLBlendFactor, MTLBlendOperation, MTLCompareFunction, MTLCullMode,
    MTLPixelFormat, MTLPrimitiveTopologyClass, MTLPrimitiveType, MTLTextureType,
    MTLTriangleFillMode, MTLWinding, RenderPipelineDescriptor,
};
use remu_gpu::GpuRegister;
use remu_shader::PixelOutputKind;
use std::collections::HashMap;

use super::{
    CB_COLOR_REGISTER_STRIDE, CB_COLOR0_ATTRIB, CB_COLOR0_ATTRIB2, CB_COLOR0_ATTRIB3,
    CB_COLOR0_CLEAR_WORD0, CB_COLOR0_CLEAR_WORD1, CB_COLOR0_INFO, CB_COLOR0_VIEW, CB_TARGET_MASK,
    ColorOperationMode, DB_DEPTH_CLEAR, DB_DEPTH_CONTROL, DB_DEPTH_VIEW, DB_HTILE_DATA_BASE,
    DB_RENDER_CONTROL, DB_Z_INFO, DecodedDepthTarget, DecodedRasterState, GuestTextureDescriptor,
    MetalPresenter, NGG_MESHLET_WORDS_PER_GROUP, PA_SU_SC_MODE_CNTL, SPI_PS_INPUT_CNTL_0,
    append_color_target_registers, append_depth_target_registers, color_operation_is_raster_noop,
    common_render_target_array_length, consume_color_metadata_clear,
    consume_depth_metadata_clear,
    decode_color_eliminate_fast_clear_target, decode_color_operation_mode,
    decode_color_resolve_targets, decode_depth_target, decode_fast_clear_color,
    decode_metal_blend_attachment, decode_raster_state, decode_render_target_format,
    decode_render_targets_from_registers, decode_scissor, decode_viewport, effective_linear_pitch,
    guest_texture_byte_count, guest_texture_descriptors_share_depth_sample_storage, indexed_draw,
    initial_depth_clear_for_target, initial_depth_clear_value, initial_render_target_clear_color,
    metal_color_write_mask, metal_nonindexed_vertex_count, metal_primitive_topology_class,
    metal_primitive_type, pixel_input_controls, prepare_indexed_primitive, prepare_ngg_meshlets,
    prepare_nonindexed_ngg_indices, read_draw_indices, read_texture_array_linear,
    read_texture_linear, rearm_color_metadata_clear, rearm_depth_metadata_clear,
    render_target_cmask_address, render_target_fast_clear_color,
};

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
    draw.context_registers.push(GpuRegister {
        offset: CB_TARGET_MASK,
        value: 0xf,
    });
    draw.context_registers
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_ATTRIB)
        .expect("MRT0 attributes")
        .value = (1 << 12) | (1 << 15);
    draw.context_registers
        .sort_unstable_by_key(|register| register.offset);

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
    draw.context_registers
        .iter_mut()
        .find(|register| register.offset == CB_COLOR0_INFO)
        .expect("MRT0 info")
        .value |= 1 << 13;
    draw.context_registers
        .sort_unstable_by_key(|register| register.offset);

    let target = decode_color_eliminate_fast_clear_target(&draw)
        .expect("valid fast-clear state")
        .expect("fast-clear target");
    assert_eq!(target.guest_slot, 0);
    assert_eq!(target.texture.address, address);
    assert!(target.fast_clear_enabled);

    draw.context_registers
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
    draw.context_registers.push(GpuRegister {
        offset: DB_HTILE_DATA_BASE,
        value: 0x0012_3456,
    });
    draw.context_registers
        .sort_unstable_by_key(|register| register.offset);

    let depth = decode_depth_target(&draw)
        .expect("valid HTILE depth target")
        .expect("enabled depth target");
    assert_eq!(depth.htile_address, Some(0x1234_5600));

    draw.context_registers
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
    assert!(!consume_color_metadata_clear(
        &mut states,
        Some(address),
        0
    ));
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

    draw.context_registers
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
    draw.context_registers.push(GpuRegister {
        offset: DB_DEPTH_VIEW,
        value: 3 | (3 << 13),
    });
    draw.context_registers
        .sort_unstable_by_key(|register| register.offset);

    let depth = decode_depth_target(&draw)
        .expect("single depth slice is supported")
        .expect("enabled depth target");
    let slice_bytes = 2 * 1_024 * 1_024;
    assert_eq!(depth.texture.address, address + 3 * slice_bytes);
    assert_eq!(depth.layer_count, 1);
    assert_eq!(guest_texture_byte_count(depth.texture), Some(slice_bytes));

    draw.context_registers
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
    assert_eq!(
        initial_depth_clear_for_target(depth),
        0.0,
        "an implicit first-use clear must not consume stale DB_DEPTH_CLEAR state"
    );

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
fn preserves_guest_winding_with_a_negative_viewport_height() {
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
            front_winding: MTLWinding::Clockwise,
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
        layer_count: 1,
        htile_address: None,
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
        layer_count: 3,
        htile_address: None,
        compare: MTLCompareFunction::Less,
        test_enabled: true,
        write_enabled: false,
        clear_enabled: false,
        clear_depth: 1.0,
        sample_count: 1,
    };

    let promoted = presenter
        .ensure_depth_render_target_texture(&device, depth, 128, 96)
        .expect("promote layered depth target");
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
fn packs_ngg_meshlets_with_shared_vertex_indices() {
    let prepared = prepare_ngg_meshlets(&[10, 11, 12, 12, 11, 13]);
    assert_eq!(prepared.group_count, 1);
    assert_eq!(prepared.vertex_references, 4);
    assert_eq!(prepared.primitive_count, 2);
    assert_eq!(prepared.words.len(), NGG_MESHLET_WORDS_PER_GROUP);
    assert_eq!(prepared.words[0], 4 | (2 << 8));
    assert_eq!(prepared.words[1], 0);
    assert_eq!(&prepared.words[2..6], &[10, 11, 12, 13]);
    assert_eq!(prepared.words[66], 1 << 10 | 2 << 20);
    assert_eq!(prepared.words[67], 2 | 1 << 10 | 3 << 20);
}

#[test]
fn preserves_nonindexed_first_vertex_and_alternates_triangle_strip_winding() {
    let mut draw = indexed_draw(0, 0, 5);
    draw.indexed = false;
    draw.primitive_type = 6;
    draw.index_offset = 7;
    let indices = prepare_nonindexed_ngg_indices(&draw).expect("strip expands");
    assert_eq!(indices.as_ref(), &[7, 8, 9, 9, 8, 10, 9, 10, 11]);
    let prepared = prepare_ngg_meshlets(indices.as_ref());
    assert_eq!(prepared.group_count, 1);
    assert_eq!(prepared.vertex_references, 5);
    assert_eq!(prepared.primitive_count, 3);
}

#[test]
fn empty_nonindexed_triangle_strip_has_no_meshlets() {
    let mut draw = indexed_draw(0, 0, 2);
    draw.indexed = false;
    draw.primitive_type = 6;
    let indices = prepare_nonindexed_ngg_indices(&draw).expect("short strip is empty");
    assert!(indices.is_empty());
    assert_eq!(prepare_ngg_meshlets(indices.as_ref()).group_count, 0);
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
fn splits_ngg_meshlets_at_the_architectural_vertex_limit() {
    let indices = (0_u32..66).collect::<Vec<_>>();
    let prepared = prepare_ngg_meshlets(&indices);
    assert_eq!(prepared.group_count, 2);
    assert_eq!(prepared.vertex_references, 66);
    assert_eq!(prepared.primitive_count, 22);
    assert_eq!(prepared.words[NGG_MESHLET_WORDS_PER_GROUP], 3 | (1 << 8));
    assert_eq!(prepared.words[NGG_MESHLET_WORDS_PER_GROUP + 1], 21);
}

#[test]
fn packs_up_to_sixty_four_ngg_primitives_per_meshlet() {
    let mut indices = Vec::new();
    for primitive in 0_u32..65 {
        indices.extend_from_slice(&[0, 1, 2 + primitive % 62]);
    }
    let prepared = prepare_ngg_meshlets(&indices);
    assert_eq!(prepared.group_count, 2);
    assert_eq!(prepared.primitive_count, 65);
    assert_eq!(prepared.words[0] >> 8, 64);
    assert_eq!(prepared.words[NGG_MESHLET_WORDS_PER_GROUP] >> 8, 1);
    assert_eq!(prepared.words[NGG_MESHLET_WORDS_PER_GROUP + 1], 64);
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
    ];
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
    ];
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
