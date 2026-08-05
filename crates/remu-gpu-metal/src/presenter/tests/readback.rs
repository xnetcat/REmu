// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use metal::{
    Device, MTLClearColor, MTLLoadAction, MTLPixelFormat, MTLRegion, MTLStorageMode,
    MTLStoreAction, MTLTextureType, MTLTextureUsage, RenderPassDescriptor, TextureDescriptor,
};
use remu_shader::PixelOutputKind;
use std::ffi::c_void;

use super::{
    DecodedRenderTarget, GuestTextureDescriptor, MetalPresenter, PendingTextureReadback,
    TextureReadbackLayout, diagnostic_texture_depth_plane_summaries, guest_frame_fingerprint,
    guest_texture_byte_count, half_float_to_srgb_unorm8, half_float_to_unorm8,
    linear_unorm8_to_srgb_unorm8, native_texture_capture_to_bgra, parse_diagnostic_hex_address,
    read_texture_array_linear, should_record_diagnostic_target_writer, unsigned_float_to_f32,
};

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
fn pre_pixel_shader_diagnostic_records_ordinary_render_targets() {
    assert!(should_record_diagnostic_target_writer(true, false, false));
    assert!(should_record_diagnostic_target_writer(true, true, false));
    assert!(should_record_diagnostic_target_writer(false, true, true));
    assert!(!should_record_diagnostic_target_writer(false, true, false));
    assert!(!should_record_diagnostic_target_writer(false, false, true));
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
fn capture_maps_depth16_to_grayscale() {
    let descriptor = GuestTextureDescriptor {
        address: 0x1_0000,
        width: 3,
        height: 1,
        pitch: 3,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::Depth16Unorm,
        bytes_per_pixel: 2,
    };
    assert_eq!(
        native_texture_capture_to_bgra(&[0, 0, 0, 128, 255, 255], descriptor),
        Some(vec![0, 0, 0, 255, 188, 188, 188, 255, 255, 255, 255, 255]),
    );
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
fn capture_decodes_packed_rgb10a2_unorm_channels() {
    let packed = 1023_u32 | (512_u32 << 10) | (256_u32 << 20) | (2_u32 << 30);
    let descriptor = GuestTextureDescriptor {
        address: 0x1_0000,
        width: 1,
        height: 1,
        pitch: 1,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGB10A2Unorm,
        bytes_per_pixel: 4,
    };
    assert_eq!(
        native_texture_capture_to_bgra(&packed.to_le_bytes(), descriptor),
        Some(vec![64, 128, 255, 170]),
    );
}

#[test]
fn diagnostic_volume_summary_keeps_each_depth_plane_distinct() {
    let descriptor = GuestTextureDescriptor {
        address: 0x1_0000,
        width: 2,
        height: 1,
        pitch: 2,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let bytes = [
        0, 0, 0, 0, 0, 0, 0, 0, // z=0
        255, 0, 0, 255, 0, 128, 0, 255, // z=1
        0, 0, 64, 255, 0, 0, 255, 255, // z=2
    ];
    let summaries =
        diagnostic_texture_depth_plane_summaries(&bytes, descriptor, 3).expect("valid volume");
    assert_eq!(summaries.len(), 3);
    assert_eq!(summaries[0].depth_plane, 0);
    assert_eq!(summaries[0].byte_count, 8);
    assert_eq!(summaries[0].nonzero_bytes, 0);
    assert_eq!(
        summaries[0].fingerprint,
        guest_frame_fingerprint(&bytes[..8])
    );
    assert_eq!(summaries[1].nonzero_bytes, 4);
    assert_eq!(
        summaries[1].fingerprint,
        guest_frame_fingerprint(&bytes[8..16])
    );
    let numeric = summaries[1].numeric.as_ref().expect("RGBA8 numeric range");
    assert_eq!(numeric.component_count, 4);
    assert_eq!(
        numeric.minimum,
        [Some(0.0), Some(0.0), Some(0.0), Some(1.0)]
    );
    assert_eq!(
        numeric.maximum,
        [Some(1.0), Some(128.0 / 255.0), Some(0.0), Some(1.0)]
    );
    assert_eq!(numeric.non_finite, [0; 4]);
    assert_ne!(summaries[1].fingerprint, summaries[2].fingerprint);
}

#[test]
fn diagnostic_volume_numeric_range_ignores_non_finite_half_values() {
    let descriptor = GuestTextureDescriptor {
        address: 0x2_0000,
        width: 2,
        height: 1,
        pitch: 2,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA16Float,
        bytes_per_pixel: 8,
    };
    let components = [
        0x3c00_u16, 0xc000, 0x7c00, 0x0000, // 1, -2, +inf, 0
        0x4000, 0x3800, 0x7e00, 0xbc00, // 2, 0.5, NaN, -1
    ];
    let bytes = components
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    let summaries =
        diagnostic_texture_depth_plane_summaries(&bytes, descriptor, 1).expect("valid volume");
    let numeric = summaries[0]
        .numeric
        .as_ref()
        .expect("RGBA16Float numeric range");
    assert_eq!(numeric.component_count, 4);
    assert_eq!(numeric.minimum, [Some(1.0), Some(-2.0), None, Some(-1.0)]);
    assert_eq!(numeric.maximum, [Some(2.0), Some(0.5), None, Some(0.0)]);
    assert_eq!(numeric.non_finite, [0, 0, 2, 0]);
}

#[test]
fn diagnostic_volume_summary_rejects_truncated_contents() {
    let descriptor = GuestTextureDescriptor {
        address: 0x3_0000,
        width: 2,
        height: 2,
        pitch: 2,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::R8Unorm,
        bytes_per_pixel: 1,
    };
    let error = diagnostic_texture_depth_plane_summaries(&[0; 7], descriptor, 2)
        .expect_err("truncated volume");
    assert!(error.contains("has 7 bytes, expected 8"));
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
fn texture_array_readback_flattens_slices_in_layer_order() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let descriptor = GuestTextureDescriptor {
        address: 0,
        width: 2,
        height: 2,
        pitch: 2,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::R16Unorm,
        bytes_per_pixel: 2,
    };
    let metal_descriptor = TextureDescriptor::new();
    metal_descriptor.set_texture_type(MTLTextureType::D2Array);
    metal_descriptor.set_pixel_format(descriptor.pixel_format);
    metal_descriptor.set_width(u64::from(descriptor.width));
    metal_descriptor.set_height(u64::from(descriptor.height));
    metal_descriptor.set_array_length(3);
    metal_descriptor.set_storage_mode(MTLStorageMode::Shared);
    metal_descriptor.set_usage(MTLTextureUsage::ShaderRead);
    let texture = device.new_texture(&metal_descriptor);
    for slice in 0..3_u64 {
        let values = [u16::try_from(slice + 1).expect("small slice"); 4];
        texture.replace_region_in_slice(
            MTLRegion::new_2d(0, 0, 2, 2),
            0,
            slice,
            values.as_ptr().cast::<c_void>(),
            4,
            8,
        );
    }

    let bytes = read_texture_array_linear(&device, &queue, &texture, descriptor)
        .expect("read all array slices");
    assert_eq!(bytes.len(), 24);
    for (slice, image) in bytes.chunks_exact(8).enumerate() {
        let expected = u16::try_from(slice + 1).expect("small slice");
        assert!(
            image
                .chunks_exact(2)
                .all(|value| u16::from_ne_bytes(value.try_into().expect("two bytes")) == expected)
        );
    }
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
        .expect("mode-27 target becomes resident")
        .texture;
    let target = DecodedRenderTarget {
        guest_slot: 0,
        texture: descriptor,
        span: super::ResourceSpan::for_texture(descriptor),
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
            blend_bypass: false,
    };

    assert!(presenter.direct_scanout_textures.is_empty());
    let captured = presenter
        .diagnostic_capture_texture(&target)
        .expect("exact resident target is diagnostic-capturable");
    assert!(std::ptr::eq(captured.as_ref(), resident.as_ref()));
}
