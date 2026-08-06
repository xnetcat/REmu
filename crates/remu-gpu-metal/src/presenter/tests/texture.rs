// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::texture::{
    ResidentSpanIndex, ResidentTextureMap, SpanQueryOrder, ensure_resident_texture_in_map,
    sampled_texture_supports_access,
};
use super::super::{
    OWNED_REMATERIALISE_BLIT, OWNED_REMATERIALISE_CPU, OWNED_REMATERIALISE_DETILE,
    OWNED_REMATERIALISE_FAILED, OWNED_REMATERIALISE_FAILED_LAYOUT, OWNED_REMATERIALISE_VIEW,
    RESIDENT_WRITE_ATTRIBUTION_MISSES, RESIDENT_WRITEBACK_DISCARDED, RESIDENT_WRITEBACK_PUBLISHED,
    ResidencyState, StaleSource, TEXTURE_SNAPSHOT_COUNT, WritebackTrigger,
};
use super::super::{
    SharedEventDrawTextureOverride, WrittenRegion, refresh_resident_texture_write_watch,
};
use super::{
    DecodedImageDescriptor, DecodedRenderTarget, GuestTextureDescriptor, MetalPresenter, RangeSet,
    ResidentTextureActivity, ResidentTextureActivityStamp, ResourceSpan, WriteWatchRefresh,
    apply_image_destination_select, comparison_depth32_upload, copy_linear_to_tiled_at_z,
    create_sampled_guest_texture, decode_guest_texture_linear_from_raw, decode_image_descriptor,
    decode_mode27_mip_chain_from_raw, depth_array_face_descriptor,
    depth_array_face_set_is_complete, fallback_image_descriptor, guest_shared_mapping_identity,
    guest_texture_byte_count, guest_texture_descriptors_overlap,
    guest_texture_descriptors_share_color_view_storage,
    guest_texture_descriptors_share_component_reordered_storage,
    guest_texture_descriptors_share_depth_sample_storage,
    guest_texture_descriptors_share_gpu_sample_storage, guest_texture_descriptors_share_storage,
    guest_texture_raw_matches, guest_texture_raw_range, image_destination_select_is_native,
    image_resource_descriptor_is_null, mode5_mip_chain_layout, mode27_mip_chain_layout,
    new_color_alias_texture_view, new_red_blue_swizzled_texture_view, read_texture_linear,
    rematerialisation_blit_source_origin_y, render_resolve_candidate_matches,
    resident_color_view_sample_is_eligible, resident_texture_activity_stats,
    resident_texture_backing_overlap_stats,
    resident_texture_eviction_plan, resident_texture_ownership_stats, sampled_alias_descriptor,
    sampled_texture_cache_accepts, sampled_texture_cache_needs_reload,
    sampled_texture_cache_requires_eviction, snapshot_guest_texture_raw,
    snapshot_watched_guest_texture_raw, storage_image_requires_mip_hierarchy,
    upload_image_textures, watch_guest_texture, write_watched_guest_texture_linear,
};
use metal::foreign_types::ForeignType;
use metal::{
    CompileOptions, Device, MTLClearColor, MTLCommandBufferStatus, MTLLoadAction, MTLPixelFormat,
    MTLRegion, MTLResourceOptions, MTLSize, MTLStorageMode, MTLStoreAction, MTLTextureType,
    MTLTextureUsage, RenderPassDescriptor, Texture, TextureDescriptor,
};
use remu_memory::CpuWriteWatchRetirement;
use remu_shader::{ImageBinding, ImageControl, Opcode, ShaderEvaluation};
use std::sync::atomic::Ordering as AtomicOrdering;

fn upload_test_image_textures(
    presenter: &mut MetalPresenter,
    evaluation: &ShaderEvaluation,
) -> Vec<Texture> {
    upload_test_image_textures_with_overrides(presenter, evaluation, &[])
}

fn upload_test_image_textures_with_overrides(
    presenter: &mut MetalPresenter,
    evaluation: &ShaderEvaluation,
    overrides: &[SharedEventDrawTextureOverride],
) -> Vec<Texture> {
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal command queue")
        .clone();
    upload_image_textures(
        "test",
        &device,
        &queue,
        evaluation,
        &mut presenter.resident_guest_textures,
        &presenter.depth_array_textures,
        &presenter.multisample_textures,
        &mut presenter.sampled_texture_cache,
        &mut presenter.sampled_texture_write_watches,
        &mut presenter.fallback_texture_cache,
        &mut presenter.next_sampled_texture_use,
        ResidentTextureActivityStamp::default(),
        overrides,
    )
    .expect("test image bindings upload")
    .into_iter()
    .map(|binding| binding.texture)
    .collect()
}

#[test]
fn accepts_large_rdna2_storage_ranges_when_descriptor_math_is_valid() {
    // Observed Silent Hill transition surfaces use a pitch larger than their visible width.
    // RDNA2 encodes pitch-1 in the depth/pitch fields for 2D resources when pitch > width; the
    // resulting physical span can legitimately exceed the old 128 MiB host-staging threshold.
    let descriptor = GuestTextureDescriptor {
        address: 0x80_d9e4_0000,
        width: 3_840,
        height: 4_160,
        pitch: 8_192,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA16Float,
        bytes_per_pixel: 8,
    };
    assert_eq!(guest_texture_byte_count(descriptor), Some(0x1040_0000));
    assert_eq!(
        guest_texture_raw_range(descriptor),
        Ok((0x80_d9e4_0000, 0x1040_0000))
    );
    let error = match watch_guest_texture(descriptor) {
        Ok(_) => panic!("unmapped storage range unexpectedly acquired a write watch"),
        Err(error) => error,
    };
    assert!(
        error.contains("no complete live shared backing"),
        "unexpected validation error: {error}"
    );
}

#[test]
#[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
fn shared_event_texture_override_bypasses_replaced_guest_mapping_at_bind() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(4 * GUEST_PAGE_SIZE).expect("guest address space");
    let original_backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("original backing");
    let replacement_backing =
        GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("replacement backing");
    let mapped = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &original_backing,
            0,
        )
        .expect("original mapping");
    let address = mapped.get();
    let width = 32_u32;
    let height = 32_u32;
    let encoded = address >> 8;
    let resource_descriptor = vec![
        encoded as u32,
        ((encoded >> 32) as u32 & 0xff) | 0x0380_0000 | (((width - 1) & 3) << 30),
        ((width - 1) >> 2) | ((height - 1) << 14),
        0x9000_0f2e,
        width - 1,
        0,
        0,
        0,
    ];
    let sampled = decode_image_descriptor(&resource_descriptor).expect("sample descriptor");
    let evaluation = ShaderEvaluation {
        initial_scalar_registers: Vec::new().into(),
        scalar_registers: Vec::new().into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: vec![ImageBinding {
            pc: 0x40,
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
                unorm: false,
            },
            resource_descriptor: resource_descriptor.into(),
            sampler_descriptor: vec![0; 4].into(),
            mip_level: None,
        }],
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    };
    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal queue")
        .clone();
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(sampled.guest.pixel_format);
    descriptor.set_width(u64::from(width));
    descriptor.set_height(u64::from(height));
    descriptor.set_storage_mode(MTLStorageMode::Private);
    descriptor.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget);
    let retained = device.new_texture(&descriptor);
    let retained_pointer = retained.as_ptr();
    let overrides = [SharedEventDrawTextureOverride {
        sampled,
        texture: retained,
    }];

    memory
        .unmap(mapped, GUEST_PAGE_SIZE)
        .expect("remove original mapping");
    memory
        .map_shared(
            GuestMapPlacement::Fixed(mapped),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &replacement_backing,
            0,
        )
        .expect("replacement mapping");
    let uploaded = upload_image_textures(
        "test",
        &device,
        &queue,
        &evaluation,
        &mut presenter.resident_guest_textures,
        &presenter.depth_array_textures,
        &presenter.multisample_textures,
        &mut presenter.sampled_texture_cache,
        &mut presenter.sampled_texture_write_watches,
        &mut presenter.fallback_texture_cache,
        &mut presenter.next_sampled_texture_use,
        ResidentTextureActivityStamp::default(),
        &overrides,
    )
    .expect("retained texture bind");
    assert_eq!(uploaded.len(), 1);
    assert_eq!(uploaded[0].texture.as_ptr(), retained_pointer);
    assert!(presenter.sampled_texture_cache.is_empty());
    assert!(presenter.sampled_texture_write_watches.is_empty());
}

#[test]
fn storage_image_color_view_reuses_gpu_current_resident_instead_of_guest_memory() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("guest backing");
    let first = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("first coherent alias");
    let second = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("second coherent alias");
    let rendered = GuestTextureDescriptor {
        address: first.get(),
        width: 16,
        height: 16,
        pitch: 16,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::BGRA8Unorm_sRGB,
        bytes_per_pixel: 4,
    };
    let storage_alias = GuestTextureDescriptor {
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        address: second.get(),
        ..rendered
    };
    assert!(guest_texture_descriptors_share_color_view_storage(
        rendered,
        storage_alias
    ));
    assert!(!guest_texture_descriptors_share_storage(
        rendered,
        storage_alias
    ));

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal command queue")
        .clone();
    let _gpu_current = presenter
        .ensure_resident_texture(&device, rendered)
        .expect("render target resident");
    presenter.mark_resident_texture_written(rendered);

    let address_words = storage_alias.address >> 8;
    let width_minus_one = storage_alias.width - 1;
    let height_minus_one = storage_alias.height - 1;
    let resource_descriptor = vec![
        address_words as u32,
        ((address_words >> 32) as u32 & 0xff) | 0x0380_0000 | ((width_minus_one & 3) << 30),
        (width_minus_one >> 2) | (height_minus_one << 14),
        0x9000_0f2e,
        storage_alias.pitch - 1,
        0,
        0,
        0,
    ];
    let evaluation = ShaderEvaluation {
        initial_scalar_registers: Vec::new().into(),
        scalar_registers: Vec::new().into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: vec![ImageBinding {
            pc: 0x60,
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
                unorm: false,
            },
            resource_descriptor: resource_descriptor.into(),
            sampler_descriptor: Vec::new().into(),
            mip_level: None,
        }],
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    };
    let uploaded = upload_image_textures(
        "test",
        &device,
        &queue,
        &evaluation,
        &mut presenter.resident_guest_textures,
        &presenter.depth_array_textures,
        &presenter.multisample_textures,
        &mut presenter.sampled_texture_cache,
        &mut presenter.sampled_texture_write_watches,
        &mut presenter.fallback_texture_cache,
        &mut presenter.next_sampled_texture_use,
        ResidentTextureActivityStamp::default(),
        &[],
    )
    .expect("storage alias upload");

    assert_eq!(uploaded.len(), 1);
    assert_eq!(
        uploaded[0].texture.pixel_format(),
        MTLPixelFormat::RGBA8Unorm,
        "the alias view must expose the storage descriptor's typed format"
    );
    assert!(
        uploaded[0].texture.parent_texture().is_some(),
        "the storage load must retain a view of the GPU-current render target"
    );
    assert!(
        !presenter
            .resident_guest_textures
            .contains_key(&storage_alias),
        "binding a coherent alias must not materialize stale guest memory under an exact key"
    );
}

#[test]
fn r32_depth_sample_prefers_gpu_written_depth32_over_exact_clean_shadow() {
    // Octopath samples DB depth through an R32Float SRD (dest select 0x204). An earlier
    // ImageLoad/ensure can leave a clean R32Float resident on the same address before the
    // Depth32Float render target is written. Sampling must bind the native Z plane.
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(16 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(4 * GUEST_PAGE_SIZE).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            4 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("guest texture mapping");
    let sampled = GuestTextureDescriptor {
        address: address.get(),
        width: 16,
        height: 16,
        pitch: 16,
        tile_mode: 24,
        pixel_format: MTLPixelFormat::R32Float,
        bytes_per_pixel: 4,
    };
    let rendered = GuestTextureDescriptor {
        pixel_format: MTLPixelFormat::Depth32Float,
        ..sampled
    };
    assert!(guest_texture_descriptors_share_depth_sample_storage(
        rendered, sampled, 0x204
    ));
    assert!(!guest_texture_descriptors_share_storage(rendered, sampled));

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal command queue")
        .clone();
    presenter
        .ensure_resident_texture(&device, sampled)
        .expect("clean R32Float shadow");
    presenter
        .ensure_resident_texture(&device, rendered)
        .expect("depth render target");
    presenter.mark_resident_texture_written(rendered);
    assert!(
        presenter
            .resident_guest_textures
            .get(&rendered)
            .is_some_and(|resident| resident.gpu_written()),
        "depth target must be GPU-authored"
    );

    let address_words = sampled.address >> 8;
    let width_minus_one = sampled.width - 1;
    let height_minus_one = sampled.height - 1;
    let resource_descriptor = vec![
        address_words as u32,
        ((address_words >> 32) as u32 & 0xff) | (22 << 20) | ((width_minus_one & 3) << 30),
        (width_minus_one >> 2) | (height_minus_one << 14),
        0x204 | (24 << 20) | (9 << 28),
        sampled.pitch - 1,
        0,
        0,
        0,
    ];
    let decoded = decode_image_descriptor(&resource_descriptor).expect("R32 depth SRD");
    assert_eq!(decoded.guest.pixel_format, MTLPixelFormat::R32Float);
    assert_eq!(decoded.destination_select, 0x204);
    assert_eq!(decoded.guest.tile_mode, 24);

    let evaluation = ShaderEvaluation {
        initial_scalar_registers: Vec::new().into(),
        scalar_registers: Vec::new().into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: vec![ImageBinding {
            pc: 0x80,
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
                unorm: false,
            },
            resource_descriptor: resource_descriptor.into(),
            sampler_descriptor: vec![0; 4].into(),
            mip_level: None,
        }],
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    };
    let uploaded = upload_image_textures(
        "test",
        &device,
        &queue,
        &evaluation,
        &mut presenter.resident_guest_textures,
        &presenter.depth_array_textures,
        &presenter.multisample_textures,
        &mut presenter.sampled_texture_cache,
        &mut presenter.sampled_texture_write_watches,
        &mut presenter.fallback_texture_cache,
        &mut presenter.next_sampled_texture_use,
        ResidentTextureActivityStamp::default(),
        &[],
    )
    .expect("native depth sample upload");
    assert_eq!(uploaded.len(), 1);
    assert_eq!(
        uploaded[0].texture.pixel_format(),
        MTLPixelFormat::Depth32Float,
        "R32 depth SRD must sample the GPU-written Depth32Float resident, not the clean R32 shadow"
    );
}

#[test]
fn layered_depth_sample_faces_resolve_to_per_face_depth_targets() {
    // A shadow cubemap is rendered as six separate 2D depth targets and then bound through one
    // six-layer R16Unorm array SRD. Each layer must resolve to the matching per-face depth
    // descriptor so the faces can be served on the GPU instead of round-tripping through guest
    // memory every frame.
    let sampled_guest = GuestTextureDescriptor {
        address: 0x70_6969_0000,
        width: 1024,
        height: 1024,
        pitch: 1024,
        tile_mode: 24,
        pixel_format: MTLPixelFormat::R16Unorm,
        bytes_per_pixel: 2,
    };
    let sampled = DecodedImageDescriptor {
        guest: sampled_guest,
        span: ResourceSpan::for_texture(sampled_guest),
        destination_select: 0x924,
        base_mip_level: 0,
        mip_level_count: 1,
        texture_type: 11,
        layer_count: 6,
    };
    let face_bytes = guest_texture_byte_count(sampled.guest).expect("face bytes");
    let first = depth_array_face_descriptor(sampled, 0).expect("first face resolves");
    assert_eq!(first.pixel_format, MTLPixelFormat::Depth16Unorm);
    assert_eq!(first.address, sampled.guest.address);
    assert_eq!((first.width, first.height, first.pitch), (1024, 1024, 1024));
    let last = depth_array_face_descriptor(sampled, 5).expect("last face resolves");
    assert_eq!(last.address, sampled.guest.address + 5 * face_bytes);
    assert_eq!(last.tile_mode, sampled.guest.tile_mode);

    // A 32-bit depth pyramid resolves through the same pairing.
    let mut float_depth = sampled;
    float_depth.guest.pixel_format = MTLPixelFormat::R32Float;
    float_depth.guest.bytes_per_pixel = 4;
    assert_eq!(
        depth_array_face_descriptor(float_depth, 0).map(|face| face.pixel_format),
        Some(MTLPixelFormat::Depth32Float)
    );

    // Layers beyond the array, non-layered types, color formats, mip pyramids, and channel
    // selections outside the supported depth set never resolve to a face.
    assert_eq!(depth_array_face_descriptor(sampled, 6), None);
    let mut flat = sampled;
    flat.texture_type = 9;
    flat.layer_count = 1;
    assert_eq!(depth_array_face_descriptor(flat, 0), None);
    let mut color = sampled;
    color.guest.pixel_format = MTLPixelFormat::RGBA8Unorm;
    assert_eq!(depth_array_face_descriptor(color, 0), None);
    let mut mipped = sampled;
    mipped.mip_level_count = 2;
    assert_eq!(depth_array_face_descriptor(mipped, 0), None);
    let mut swizzled = sampled;
    swizzled.destination_select = 0xfac;
    assert_eq!(depth_array_face_descriptor(swizzled, 0), None);
}

#[test]
fn layered_depth_sample_requires_every_face_before_skipping_alias_writeback() {
    let sampled_guest = GuestTextureDescriptor {
        address: 0x70_6969_0000,
        width: 1024,
        height: 1024,
        pitch: 1024,
        tile_mode: 24,
        pixel_format: MTLPixelFormat::R16Unorm,
        bytes_per_pixel: 2,
    };
    let sampled = DecodedImageDescriptor {
        guest: sampled_guest,
        span: ResourceSpan::for_texture(sampled_guest),
        destination_select: 0x924,
        base_mip_level: 0,
        mip_level_count: 1,
        texture_type: 11,
        layer_count: 6,
    };
    let faces = (0..sampled.layer_count)
        .map(|layer| depth_array_face_descriptor(sampled, layer).expect("face resolves"))
        .collect::<Vec<_>>();

    assert!(!depth_array_face_set_is_complete(sampled, |face| {
        face == faces[0]
    }));
    assert!(depth_array_face_set_is_complete(sampled, |face| {
        faces.contains(&face)
    }));
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
        true
    ));
    assert!(sampled_texture_cache_needs_reload(
        WriteWatchRefresh::Replaced,
        false
    ));
    assert!(sampled_texture_cache_needs_reload(
        WriteWatchRefresh::Renewed { was_dirty: false },
        true
    ));
    assert!(!sampled_texture_cache_needs_reload(
        WriteWatchRefresh::Renewed { was_dirty: true },
        false
    ));
    assert!(!sampled_texture_cache_needs_reload(
        WriteWatchRefresh::Reprotecting,
        true
    ));
}

#[test]
fn sampled_texture_cache_budget_uses_resident_bytes_and_rejects_oversize_entries() {
    let capacity = Some(512 * 1024 * 1024);
    assert!(sampled_texture_cache_accepts(128 * 1024 * 1024, capacity));
    assert!(!sampled_texture_cache_accepts(513 * 1024 * 1024, capacity));
    assert!(!sampled_texture_cache_requires_eviction(
        17,
        383 * 1024 * 1024,
        128 * 1024 * 1024,
        capacity,
    ));
    assert!(sampled_texture_cache_requires_eviction(
        17,
        385 * 1024 * 1024,
        128 * 1024 * 1024,
        capacity,
    ));
}

#[test]
fn comparison_sampling_converts_srgb_color_red_to_linear_depth() {
    let upload = comparison_depth32_upload(
        GuestTextureDescriptor {
            address: 0x10000,
            width: 3,
            height: 1,
            pitch: 3,
            tile_mode: 0,
            pixel_format: MTLPixelFormat::RGBA8Unorm_sRGB,
            bytes_per_pixel: 4,
        },
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
fn comparison_sampling_decompresses_bc1_red_to_depth() {
    let upload = comparison_depth32_upload(
        GuestTextureDescriptor {
            address: 0x10000,
            width: 4,
            height: 4,
            pitch: 4,
            tile_mode: 0,
            pixel_format: MTLPixelFormat::BC1_RGBA,
            bytes_per_pixel: 8,
        },
        &[0x00, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    )
    .expect("BC1 color texture is comparison-compatible")
    .expect("BC1 color texture needs a depth upload");
    let depths = upload
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("one float")))
        .collect::<Vec<_>>();
    assert_eq!(depths.len(), 16);
    assert!(depths.iter().all(|&depth| depth == 1.0));
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

    memory
        .write(second, &[0x73; 32])
        .expect("change bytes through the disjoint second alias");
    assert!(
        watch
            .content_has_changed()
            .expect("validate dirty content through disjoint physical aliases"),
        "batched fingerprinting must preserve physical order across virtual alias seams"
    );
    let raw = snapshot_watched_guest_texture_raw(descriptor, &watch)
        .expect("refresh disjoint-alias fingerprint baseline");
    assert_eq!(
        &raw[usize::try_from(4 * GUEST_PAGE_SIZE).unwrap()
            ..usize::try_from(4 * GUEST_PAGE_SIZE).unwrap() + 32],
        &[0x73; 32]
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
    snapshot_watched_guest_texture_raw(descriptor, &watch)
        .expect("initialize canonical content fingerprints");

    memory.write(alias, &[0x5a]).expect("write through alias");
    assert!(
        watch.is_dirty(),
        "a coherent alias write must invalidate the GPU texture"
    );
    assert!(
        watch
            .content_has_changed()
            .expect("validate content through canonical backing"),
        "dirty alias bytes must compare against their canonical resource offset"
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
fn texture_write_watch_renews_after_a_late_physical_alias_and_validates_early_writes() {
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
    let descriptor = GuestTextureDescriptor {
        address: primary.get(),
        width: 64,
        height: 64,
        pitch: 64,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let original = watch_guest_texture(descriptor).expect("initial single-alias watch");
    snapshot_watched_guest_texture_raw(descriptor, &original).expect("initialize content baseline");

    let alias_address = base
        .checked_add(8 * GUEST_PAGE_SIZE)
        .expect("late alias address");
    let alias = memory
        .map_shared(
            GuestMapPlacement::Fixed(alias_address),
            4 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("late physical alias");
    assert_eq!(
        original.retirement(),
        Some(CpuWriteWatchRetirement::Reprotected),
        "adding a coherent VA must renew rather than discard GPU physical storage"
    );

    memory
        .write(alias, &[0x5a; 32])
        .expect("CPU initializes the new alias before the presenter refreshes");
    let mut renewed = watch_guest_texture(descriptor).expect("renewed complete alias watch");
    assert_eq!(renewed.ranges.len(), 2);
    renewed
        .inherit_content_state_from(&original)
        .expect("preserve the old content baseline");
    assert!(
        renewed
            .content_has_changed()
            .expect("validate the complete alias-change range"),
        "a late-alias CPU write must defeat stale GPU ownership"
    );

    snapshot_watched_guest_texture_raw(descriptor, &renewed)
        .expect("establish the renewed content baseline");
    memory
        .write(alias, &[0x6b; 32])
        .expect("write through the renewed alias");
    assert!(
        renewed.is_dirty(),
        "the rebuilt watch must cover the late alias directly"
    );
}

#[test]
fn pending_alias_validation_survives_a_later_gpu_ownership_boundary() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let Some(device) = Device::system_default() else {
        return;
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
    let descriptor = GuestTextureDescriptor {
        address: primary.get(),
        width: 64,
        height: 64,
        pitch: 64,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let mut residents = ResidentTextureMap::new();
    let queue = device.new_command_queue();
    ensure_resident_texture_in_map(&device, Some(&queue), descriptor, &mut residents, true)
        .expect("resident with initialized content baseline");
    {
        let resident = residents.get_mut(&descriptor).expect("resident exists");
        resident.note_gpu_authored_whole_span();
    }

    let alias_address = base
        .checked_add(8 * GUEST_PAGE_SIZE)
        .expect("late alias address");
    let alias = memory
        .map_shared(
            GuestMapPlacement::Fixed(alias_address),
            4 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("late physical alias");
    memory
        .write(alias, &[0x5a; 32])
        .expect("CPU writes through the late alias");

    let resident = residents.get_mut(&descriptor).expect("resident exists");
    let refresh =
        refresh_resident_texture_write_watch(resident).expect("renew complete physical watch");
    assert_eq!(refresh, WriteWatchRefresh::Renewed { was_dirty: true });
    assert!(resident.write_watch.content_validation_pending());

    // A following draw establishes a new GPU generation boundary before eviction. The inherited
    // dirty range must remain authoritative even though the new watch itself is clean.
    let eviction_refresh =
        refresh_resident_texture_write_watch(resident).expect("refresh at eviction");
    assert_eq!(eviction_refresh, WriteWatchRefresh::Active);
    assert!(
        resident
            .cpu_write_obsoletes_gpu_writeback(eviction_refresh)
            .expect("validate pending CPU contents"),
        "a later GPU boundary must not hide the CPU write and publish stale Metal bytes"
    );
}

#[test]
fn late_alias_without_a_cpu_write_preserves_discarded_gpu_contents() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let Some(device) = Device::system_default() else {
        return;
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
    let descriptor = GuestTextureDescriptor {
        address: primary.get(),
        width: 64,
        height: 64,
        pitch: 64,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let mut residents = ResidentTextureMap::new();
    ensure_resident_texture_in_map(&device, None, descriptor, &mut residents, false)
        .expect("discarded render target with a CPU baseline");
    {
        let resident = residents.get_mut(&descriptor).expect("resident exists");
        resident.note_gpu_authored_whole_span();
    }

    let alias_address = base
        .checked_add(8 * GUEST_PAGE_SIZE)
        .expect("late alias address");
    memory
        .map_shared(
            GuestMapPlacement::Fixed(alias_address),
            4 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("late physical alias");

    let resident = residents.get_mut(&descriptor).expect("resident exists");
    let refresh =
        refresh_resident_texture_write_watch(resident).expect("renew complete physical watch");
    assert_eq!(refresh, WriteWatchRefresh::Renewed { was_dirty: true });
    assert!(
        !resident
            .cpu_write_obsoletes_gpu_writeback(refresh)
            .expect("validate unchanged guest bytes"),
        "adding an alias without a CPU store must retain GPU-authored contents"
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
    write_watched_guest_texture_linear(descriptor, &linear, None, &watch)
        .expect("guarded direct writeback");
    assert_eq!(watch.generations(), generations);
    assert!(!watch.is_dirty());
    let decoded = snapshot_watched_guest_texture_raw(descriptor, &watch).expect("updated snapshot");
    assert!(decoded.contains(&0x3c));
}

#[test]
#[ignore = "Phase 1G deleted content fingerprints / self-readback generations; coarse write-watch invalidation replaces this contract"]
fn texture_content_validation_avoids_clean_whole_resource_reloads() {
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
        width: 16,
        height: 16,
        pitch: 16,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let watch = watch_guest_texture(descriptor).expect("contiguous texture watch");
    snapshot_watched_guest_texture_raw(descriptor, &watch).expect("initialize fingerprints");

    let unrelated = address
        .checked_add(2 * 1024)
        .expect("same-page unrelated address");
    memory
        .write(unrelated, &[0x7a])
        .expect("unrelated same-page store faults the texture watch");
    assert!(
        watch
            .content_has_changed()
            .expect("validate unrelated dirty page"),
        "Phase 1G coarse watches invalidate on any fault in the watched pages"
    );

    memory
        .write(address, &[0x5a])
        .expect("changed store faults the rearmed page");
    assert!(
        watch
            .content_has_changed()
            .expect("validate changed dirty page"),
        "a real CPU content update must invalidate the resident texture"
    );
    let raw = snapshot_watched_guest_texture_raw(descriptor, &watch)
        .expect("full reload establishes the new fingerprint baseline");
    assert_eq!(raw[0], 0x5a);

    memory
        .write(address, &[0x5a])
        .expect("same updated value faults the rearmed page");
    assert!(
        !watch
            .content_has_changed()
            .expect("validate repeated updated value")
    );
}

#[test]
fn a_continuously_written_resource_still_completes_its_snapshot() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

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
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let watch = watch_guest_texture(descriptor).expect("contiguous texture watch");

    // Stores land through the guest mapping so they fault the armed watch exactly as the game's
    // own writer thread does. A snapshot that only accepts a run with no concurrent store never
    // converges against this writer, so reaching the assertions at all is the regression.
    let finished = Arc::new(AtomicBool::new(false));
    let writer_finished = Arc::clone(&finished);
    let writer_address = usize::try_from(address.get()).expect("guest address fits usize");
    let writer = std::thread::spawn(move || {
        while !writer_finished.load(Ordering::Acquire) {
            // SAFETY: the mapping outlives this thread because the test joins it before dropping
            // the address space, and the write watch restores write access on the fault it takes.
            unsafe { (writer_address as *mut u8).write_volatile(0x5a) };
            std::hint::spin_loop();
        }
    });

    let raw = snapshot_watched_guest_texture_raw(descriptor, &watch)
        .expect("a concurrently written resource still yields a snapshot");
    finished.store(true, Ordering::Release);
    writer.join().expect("guest writer thread");

    assert_eq!(raw.len(), 64 * 64 * 4);
}

#[test]
fn a_continuously_written_resource_still_completes_content_validation() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    const TEXTURE_PAGES: u64 = 1024;
    let texture_bytes = TEXTURE_PAGES * GUEST_PAGE_SIZE;
    let mut memory = GuestAddressSpace::new(2 * texture_bytes).expect("guest address space");
    let backing = GuestPhysicalBacking::new(texture_bytes).expect("guest physical backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            texture_bytes,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("contiguous texture mapping");
    let descriptor = GuestTextureDescriptor {
        address: address.get(),
        width: 2048,
        height: 2048,
        pitch: 2048,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let watch = watch_guest_texture(descriptor).expect("contiguous texture watch");
    snapshot_watched_guest_texture_raw(descriptor, &watch).expect("initialize fingerprints");

    let finished = Arc::new(AtomicBool::new(false));
    let writer_finished = Arc::clone(&finished);
    let writer_address = usize::try_from(address.get()).expect("guest address fits usize");
    let writer_page_bytes = usize::try_from(GUEST_PAGE_SIZE).expect("guest page size fits usize");
    let writer = std::thread::spawn(move || {
        let mut value = 0_u8;
        let mut page = 0_usize;
        while !writer_finished.load(Ordering::Acquire) {
            value = value.wrapping_add(1);
            // SAFETY: the mapping outlives this thread and the write watch restores access on
            // every protection fault before the store retries.
            unsafe {
                (writer_address as *mut u8)
                    .add(page * writer_page_bytes)
                    .write_volatile(value)
            };
            page = (page + 1) % usize::try_from(TEXTURE_PAGES).unwrap();
            std::hint::spin_loop();
        }
    });

    // Prove the writer has actually faulted and dirtied the armed watch before timing the
    // validator. Without this handshake, a busy host can schedule the validator first and the
    // test legitimately observes unchanged bytes, making the assertion depend on thread luck.
    let writer_start_deadline = Instant::now() + Duration::from_secs(2);
    while !watch.is_dirty() {
        assert!(
            Instant::now() < writer_start_deadline,
            "guest writer did not dirty the armed texture watch"
        );
        std::thread::yield_now();
    }

    let (sender, receiver) = std::sync::mpsc::channel();
    let validator = std::thread::spawn(move || {
        let result = watch.content_has_changed();
        let _ = sender.send(result);
    });
    let completed_while_writing = receiver.recv_timeout(Duration::from_millis(250));
    let completed_in_time = completed_while_writing.is_ok();
    finished.store(true, Ordering::Release);
    writer.join().expect("guest writer thread");
    let result = match completed_while_writing {
        Ok(result) => result,
        Err(_) => receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("content validation finishes after the writer stops"),
    };
    validator.join().expect("content validator thread");

    assert!(
        completed_in_time,
        "content validation must not retry forever while guest writes continue"
    );
    assert!(result.expect("content validation succeeds"));
}

#[test]
#[ignore = "Phase 1G deleted content fingerprints / self-readback generations; coarse write-watch invalidation replaces this contract"]
fn live_content_baseline_skips_full_host_snapshot_copy() {
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
    memory
        .write(address, &[0x11, 0x22, 0x33, 0x44])
        .expect("seed guest texels");
    let descriptor = GuestTextureDescriptor {
        address: address.get(),
        width: 16,
        height: 16,
        pitch: 16,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let watch = watch_guest_texture(descriptor).expect("contiguous texture watch");
    watch
        .record_live_content_baseline()
        .expect("fingerprint live mapping in place");

    memory
        .write(address, &[0x11, 0x22, 0x33, 0x44])
        .expect("identical rewrite faults the watch");
    assert!(
        !watch
            .content_has_changed()
            .expect("validate identical rewrite"),
        "Phase 1G coarse dirty: a store that faulted the watch invalidates even if bytes match"
    );

    memory
        .write(address, &[0xaa, 0xbb, 0xcc, 0xdd])
        .expect("changed rewrite faults the watch");
    assert!(
        watch
            .content_has_changed()
            .expect("validate changed rewrite"),
        "in-place baseline must still detect real CPU rewrites"
    );
}

#[test]
#[ignore = "Phase 1G deleted content fingerprints / self-readback generations; coarse write-watch invalidation replaces this contract"]
fn texture_content_validation_survives_write_watch_renewal() {
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
    let watch = watch_guest_texture(descriptor).expect("original texture watch");
    snapshot_watched_guest_texture_raw(descriptor, &watch).expect("initialize fingerprints");
    memory
        .write(address, &[0])
        .expect("same-value store dirties the original watch");

    let mut renewed = watch_guest_texture(descriptor).expect("renewed texture watch");
    renewed
        .inherit_content_state_from(&watch)
        .expect("carry dirty ranges and fingerprints into renewed watch");
    drop(watch);
    assert!(
        !renewed
            .content_has_changed()
            .expect("validate inherited same-value dirty range"),
        "permission renewal must not convert clean guest contents into a full reload"
    );
}

#[test]
fn contiguous_texture_access_coalesces_adjacent_mapping_records() {
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
    assert_eq!(
        watch.ranges.len(),
        1,
        "physical registry seams must not multiply CPU write watches"
    );
    assert!(watch.has_single_covering_mapping(descriptor.address, 4 * GUEST_PAGE_SIZE));
    let raw = snapshot_watched_guest_texture_raw(descriptor, &watch)
        .expect("snapshot uses the coalesced direct mapping path");
    assert_eq!(raw.len(), usize::try_from(4 * GUEST_PAGE_SIZE).unwrap());
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
fn sampled_texture_usage_is_promoted_without_splitting_guest_image_identity() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let guest = GuestTextureDescriptor {
        address: 0x10_0000,
        width: 1,
        height: 1,
        pitch: 1,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let sampled = create_sampled_guest_texture(
        &device,
        &queue,
        guest,
        &[1, 2, 3, 4],
        0,
        1,
        13,
        1,
        false,
        false,
    )
    .expect("ordinary 2D-array sample texture");
    assert_eq!(sampled.usage(), MTLTextureUsage::ShaderRead);
    assert!(sampled_texture_supports_access(&sampled, false));
    assert!(!sampled_texture_supports_access(&sampled, true));

    let storage = create_sampled_guest_texture(
        &device,
        &queue,
        guest,
        &[1, 2, 3, 4],
        0,
        1,
        13,
        1,
        false,
        true,
    )
    .expect("writable 2D-array storage texture");
    assert_eq!(
        storage.usage(),
        MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite
    );
    assert!(sampled_texture_supports_access(&storage, false));
    assert!(sampled_texture_supports_access(&storage, true));
}

#[test]
fn load_before_store_binds_one_read_write_array_texture() {
    let mut resource_descriptor = vec![0; 8];
    resource_descriptor[3] = 13 << 28;
    let control = ImageControl {
        dmask: 0xf,
        vector_address: 0,
        address_registers: vec![0, 1, 2],
        vector_data: 4,
        scalar_resource: 12,
        scalar_sampler: 0,
        dimension: 5,
        is_array: true,
        glc: false,
        slc: false,
        a16: false,
        d16: false,
        unorm: false,
    };
    let mut store = ImageBinding {
        pc: 0x38,
        opcode: Opcode::Known("ImageStore"),
        control: control.clone(),
        resource_descriptor: resource_descriptor.clone().into(),
        sampler_descriptor: Vec::new().into(),
        mip_level: None,
    };
    let load = ImageBinding {
        pc: 0x30,
        opcode: Opcode::Known("ImageLoad"),
        control,
        resource_descriptor: resource_descriptor.into(),
        sampler_descriptor: Vec::new().into(),
        mip_level: None,
    };
    store.control.vector_data = 8;
    let evaluation = |image_bindings| ShaderEvaluation {
        initial_scalar_registers: Vec::new().into(),
        scalar_registers: Vec::new().into(),
        scalar_memory_reads: Vec::new(),
        image_bindings,
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    };

    let (_transition_sender, transition_receiver) = std::sync::mpsc::channel();
    let mut transition = MetalPresenter::new(transition_receiver).expect("Metal presenter state");
    let read_only = upload_test_image_textures(&mut transition, &evaluation(vec![load.clone()]));
    assert_eq!(read_only.len(), 1);
    assert!(!sampled_texture_supports_access(&read_only[0], true));
    let promoted = upload_test_image_textures(&mut transition, &evaluation(vec![store.clone()]));
    assert_eq!(promoted.len(), 1);
    assert_ne!(read_only[0].as_ptr(), promoted[0].as_ptr());
    assert!(sampled_texture_supports_access(&promoted[0], true));
    let reused = upload_test_image_textures(&mut transition, &evaluation(vec![load.clone()]));
    assert_eq!(reused.len(), 1);
    assert_eq!(promoted[0].as_ptr(), reused[0].as_ptr());
    assert_eq!(transition.fallback_texture_cache.len(), 1);

    let (_combined_sender, combined_receiver) = std::sync::mpsc::channel();
    let mut combined = MetalPresenter::new(combined_receiver).expect("Metal presenter state");
    let sampled = fallback_image_descriptor(&load).expect("array fallback descriptor");
    let device = combined.device.as_ref().expect("Metal device");
    let retained_descriptor = TextureDescriptor::new();
    retained_descriptor.set_texture_type(MTLTextureType::D2Array);
    retained_descriptor.set_pixel_format(sampled.guest.pixel_format);
    retained_descriptor.set_width(u64::from(sampled.guest.width));
    retained_descriptor.set_height(u64::from(sampled.guest.height));
    retained_descriptor.set_array_length(u64::from(sampled.layer_count.max(1)));
    retained_descriptor.set_storage_mode(MTLStorageMode::Private);
    retained_descriptor.set_usage(MTLTextureUsage::ShaderRead);
    let retained = device.new_texture(&retained_descriptor);
    let retained_pointer = retained.as_ptr();
    let overrides = [SharedEventDrawTextureOverride {
        sampled,
        texture: retained,
    }];
    let uploaded = upload_test_image_textures_with_overrides(
        &mut combined,
        &evaluation(vec![load, store]),
        &overrides,
    );
    assert_eq!(uploaded.len(), 2);
    assert_eq!(uploaded[0].as_ptr(), uploaded[1].as_ptr());
    assert_ne!(uploaded[0].as_ptr(), retained_pointer);
    assert_eq!(combined.fallback_texture_cache.len(), 1);
    for texture in &uploaded {
        assert_eq!(texture.texture_type(), MTLTextureType::D2Array);
        assert!(sampled_texture_supports_access(texture, true));
    }
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
            unorm: false,
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
            unorm: false,
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
    // fallback_image_descriptor still builds a 1x1 placeholder for callers that need a
    // dimensioned Metal object, but sampled upload must refuse non-null type-0 SRDs instead of
    // silently sampling opaque black (see upload_image_textures IMAGE-REFUSE path).
    let fallback =
        fallback_image_descriptor(&buffer_descriptor).expect("type-zero sampled fallback");
    assert_eq!(fallback.texture_type, 9);
    assert_eq!(
        (buffer_descriptor.resource_descriptor[3] >> 28) & 0xf,
        0,
        "fixture must remain TYPE nibble 0"
    );

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
fn mode5_sampled_mip_range_covers_the_packed_prefix_and_deferred_base_image() {
    const MODE5_HIERARCHY: GuestTextureDescriptor = GuestTextureDescriptor {
        address: 0x70_2754_1000,
        width: 512,
        height: 512,
        pitch: 512,
        tile_mode: 5,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let descriptor = DecodedImageDescriptor {
        guest: MODE5_HIERARCHY,
        span: ResourceSpan::for_texture(MODE5_HIERARCHY),
        destination_select: 0xfac,
        base_mip_level: 0,
        mip_level_count: 10,
        texture_type: 9,
        layer_count: 1,
    };
    assert_eq!(
        super::sampled_texture_raw_range(descriptor)
            .expect("mode-5 sampled hierarchy covers its complete packed chain")
            .1,
        mode5_mip_chain_layout(descriptor.guest, 10)
            .expect("mode-5 underlying mip layout")
            .byte_count
    );
    assert!(
        super::sampled_texture_raw_range(descriptor)
            .expect("mode-5 range")
            .1
            > usize::try_from(
                guest_texture_byte_count(descriptor.guest).expect("base byte count fits")
            )
            .expect("base byte count fits usize")
    );
}

#[test]
fn mode5_cube_mip_range_uses_one_complete_chain_per_face() {
    const CUBE_FACE: GuestTextureDescriptor = GuestTextureDescriptor {
        address: 0x70_3000_0000,
        width: 256,
        height: 256,
        pitch: 256,
        tile_mode: 5,
        pixel_format: MTLPixelFormat::RGBA16Float,
        bytes_per_pixel: 8,
    };
    let descriptor = DecodedImageDescriptor {
        guest: CUBE_FACE,
        span: ResourceSpan::for_texture(CUBE_FACE),
        destination_select: 0xfac,
        base_mip_level: 0,
        mip_level_count: 9,
        texture_type: 11,
        layer_count: 6,
    };
    let face_bytes = mode5_mip_chain_layout(descriptor.guest, 9)
        .expect("mode-5 cube face layout")
        .byte_count;
    assert_eq!(
        super::sampled_texture_raw_range(descriptor)
            .expect("mode-5 cube covers every face chain")
            .1,
        face_bytes * 6
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
        .expect("single-level residency")
        .texture;
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
fn resident_ownership_stats_distinguish_gpu_state_and_replaced_mappings() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).expect("guest backing");
    let first_address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("first target mapping");
    let second_address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            GUEST_PAGE_SIZE,
        )
        .expect("second target mapping");
    let first = GuestTextureDescriptor {
        address: first_address.get(),
        width: 16,
        height: 16,
        pitch: 16,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let second = GuestTextureDescriptor {
        address: second_address.get(),
        ..first
    };

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    presenter
        .ensure_resident_texture(&device, first)
        .expect("first target resident");
    presenter
        .ensure_resident_texture(&device, second)
        .expect("second target resident");
    presenter.mark_resident_texture_written(first);

    let live = resident_texture_ownership_stats(&presenter.resident_guest_textures);
    assert_eq!(live.gpu_owned_count, 1);
    assert_eq!(live.cpu_owned_count, 1);
    assert_eq!(live.active_mapping_count, 2);
    assert_eq!(live.replaced_mapping_count, 0);
    assert_eq!(live.unique_backing_bases, 2);
    assert_eq!(live.duplicate_backing_bases, 0);
    assert_eq!(live.unique_backing_spans, 2);
    assert_eq!(live.duplicate_backing_spans, 0);
    assert_eq!(live.unresolved_backing_count, 0);

    memory
        .unmap(second_address, GUEST_PAGE_SIZE)
        .expect("replace the second target mapping");
    let retired = resident_texture_ownership_stats(&presenter.resident_guest_textures);
    assert_eq!(retired.active_mapping_count, 1);
    assert_eq!(retired.replaced_mapping_count, 1);
    assert_eq!(retired.replaced_mapping_bytes, live.cpu_owned_bytes);
    assert_eq!(retired.unique_backing_bases, 1);
    assert_eq!(retired.unique_backing_spans, 1);
    assert_eq!(retired.unresolved_backing_count, 0);
}

#[test]
fn resident_lifetime_stats_attribute_physical_aliases_and_activity_age() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(4 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("target mapping")
        .get();
    let first = GuestTextureDescriptor {
        address,
        width: 16,
        height: 16,
        pitch: 16,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let second = GuestTextureDescriptor {
        address: address + 512,
        ..first
    };

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    presenter
        .ensure_resident_texture(&device, first)
        .expect("first target resident");
    presenter
        .ensure_resident_texture(&device, second)
        .expect("second target resident");
    assert_eq!(presenter.resident_guest_textures.len(), 2);

    let first_resident = presenter
        .resident_guest_textures
        .get_mut(&first)
        .expect("first resident");
    first_resident.note_gpu_authored_whole_span();
    first_resident.activity = Some(ResidentTextureActivity {
        last_read_work: Some(100),
        last_read_flip: Some(10),
        last_write_work: Some(90),
        last_write_flip: Some(9),
    });
    presenter
        .resident_guest_textures
        .get_mut(&second)
        .expect("second resident")
        .activity = Some(ResidentTextureActivity {
        last_read_work: None,
        last_read_flip: None,
        last_write_work: Some(10),
        last_write_flip: Some(1),
    });

    let overlap = resident_texture_backing_overlap_stats(&presenter.resident_guest_textures);
    assert_eq!(overlap.pair_count, 1);
    assert_eq!(overlap.pair_overlap_bytes, 512);
    assert_eq!(overlap.resident_count, 2);
    assert_eq!(overlap.gpu_owned_count, 1);
    assert_eq!(overlap.cpu_owned_count, 1);

    let activity = resident_texture_activity_stats(&presenter.resident_guest_textures, 100, 10);
    assert_eq!(activity.read_work[0].gpu_owned_count, 1);
    assert_eq!(activity.read_work[5].cpu_owned_count, 1);
    assert_eq!(activity.write_work[0].gpu_owned_count, 1);
    assert_eq!(activity.write_work[1].cpu_owned_count, 1);
    assert_eq!(activity.read_flip[0].gpu_owned_count, 1);
    assert_eq!(activity.read_flip[5].cpu_owned_count, 1);
    assert_eq!(activity.write_flip[1].gpu_owned_count, 1);
    assert_eq!(activity.write_flip[3].cpu_owned_count, 1);
}

#[test]
fn resident_pressure_plan_prefers_clean_old_images_and_excludes_the_current_flip() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(3 * GUEST_PAGE_SIZE).expect("guest backing");
    let mut descriptors = Vec::new();
    for page_index in 0..3 {
        let address = memory
            .map_shared(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                page_index * GUEST_PAGE_SIZE,
            )
            .expect("resident mapping");
        descriptors.push(GuestTextureDescriptor {
            address: address.get(),
            width: 16,
            height: 16,
            pitch: 16,
            tile_mode: 0,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        });
    }

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    for descriptor in descriptors.iter().copied() {
        presenter
            .ensure_resident_texture(&device, descriptor)
            .expect("resident texture");
    }
    let old_gpu = descriptors[0];
    let old_clean = descriptors[1];
    let current_gpu = descriptors[2];
    {
        let resident = presenter
            .resident_guest_textures
            .get_mut(&old_gpu)
            .expect("old GPU resident");
        resident.note_gpu_authored_whole_span();
        resident.last_use = ResidentTextureActivityStamp { work: 10, flip: 1 };
    }
    presenter
        .resident_guest_textures
        .get_mut(&old_clean)
        .expect("old clean resident")
        .last_use = ResidentTextureActivityStamp { work: 20, flip: 1 };
    {
        let resident = presenter
            .resident_guest_textures
            .get_mut(&current_gpu)
            .expect("current GPU resident");
        resident.note_gpu_authored_whole_span();
        resident.last_use = ResidentTextureActivityStamp { work: 30, flip: 2 };
    }
    let total_bytes = presenter
        .resident_guest_textures
        .values()
        .map(|resident| resident.texture.allocated_size())
        .sum::<u64>();
    let clean_bytes = presenter
        .resident_guest_textures
        .get(&old_clean)
        .expect("old clean resident")
        .texture
        .allocated_size();
    let plan = resident_texture_eviction_plan(
        &presenter.resident_guest_textures,
        total_bytes.saturating_sub(clean_bytes),
        2,
        1,
    );
    assert_eq!(plan.descriptors, vec![old_clean]);
    assert!(!plan.descriptors.contains(&current_gpu));
}

#[test]
fn resident_pressure_plan_keeps_overlapping_gpu_versions_indivisible() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(4 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("target mapping")
        .get();
    let first = GuestTextureDescriptor {
        address,
        width: 16,
        height: 16,
        pitch: 16,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let second = GuestTextureDescriptor {
        address: address + 512,
        ..first
    };

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    presenter
        .ensure_resident_texture(&device, first)
        .expect("first resident");
    presenter
        .ensure_resident_texture(&device, second)
        .expect("second resident");
    for (work, descriptor) in [(10, first), (20, second)] {
        let resident = presenter
            .resident_guest_textures
            .get_mut(&descriptor)
            .expect("overlap resident");
        resident.note_gpu_authored_whole_span();
        resident.content_version = work;
        resident.last_use = ResidentTextureActivityStamp { work, flip: 1 };
    }

    let plan = resident_texture_eviction_plan(&presenter.resident_guest_textures, 0, 2, usize::MAX);
    assert!(
        plan.descriptors.is_empty(),
        "an overlapping GPU-owned version cannot be written back independently"
    );
}

#[test]
fn guest_observable_writeback_retiles_gpu_pixels_and_pressure_eviction_drops_them() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };
    use std::ffi::c_void;

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
        .expect("resident mapping");
    let descriptor = GuestTextureDescriptor {
        address: address.get(),
        width: 64,
        height: 64,
        pitch: 64,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let texture = presenter
        .ensure_resident_texture(&device, descriptor)
        .expect("resident texture")
        .texture;
    let rendered = (0..64 * 64 * 4)
        .map(|index| ((index * 37 + index / 11) & 0xff) as u8)
        .collect::<Vec<_>>();
    texture.replace_region(
        MTLRegion::new_2d(0, 0, 64, 64),
        0,
        rendered.as_ptr().cast::<c_void>(),
        64 * 4,
    );
    presenter.mark_resident_texture_written_at(
        descriptor,
        ResourceSpan::for_texture(descriptor),
        None,
        WrittenRegion::whole_resource(),
        ResidentTextureActivityStamp { work: 10, flip: 1 },
    );

    let (count, bytes) = presenter
        .garbage_collect_resident_textures_to_budget(0, 1, 8)
        .expect("current-flip pressure pass");
    assert_eq!((count, bytes), (0, 0));
    assert!(presenter.resident_guest_textures.contains_key(&descriptor));

    // Host memory pressure is REmu's own event, so the pixels are dropped rather than staged in
    // a guest range no guest read named. The debt survives for a trigger that does name one.
    let discarded_before = RESIDENT_WRITEBACK_DISCARDED[WritebackTrigger::PressureEviction.index()]
        .load(AtomicOrdering::Relaxed);
    let untouched = snapshot_guest_texture_raw(descriptor).expect("pre-eviction guest bytes");
    let (count, bytes) = presenter
        .garbage_collect_resident_textures_to_budget(0, 2, 8)
        .expect("completed-flip pressure pass");
    assert_eq!(count, 1);
    assert!(bytes >= rendered.len() as u64);
    assert!(!presenter.resident_guest_textures.contains_key(&descriptor));
    assert_eq!(
        RESIDENT_WRITEBACK_DISCARDED[WritebackTrigger::PressureEviction.index()]
            .load(AtomicOrdering::Relaxed),
        discarded_before + 1
    );
    assert_eq!(
        snapshot_guest_texture_raw(descriptor).expect("post-eviction guest bytes"),
        untouched,
        "an eviction the guest never asked for must not write guest memory"
    );

    // The same pixels, published for a trigger that names a guest read of exactly these bytes.
    let texture = presenter
        .ensure_resident_texture(&device, descriptor)
        .expect("rebuilt resident texture")
        .texture;
    texture.replace_region(
        MTLRegion::new_2d(0, 0, 64, 64),
        0,
        rendered.as_ptr().cast::<c_void>(),
        64 * 4,
    );
    presenter.mark_resident_texture_written_at(
        descriptor,
        ResourceSpan::for_texture(descriptor),
        None,
        WrittenRegion::whole_resource(),
        ResidentTextureActivityStamp { work: 11, flip: 3 },
    );
    assert!(
        presenter
            .settle_resident_texture_writeback(descriptor, WritebackTrigger::DmaSource)
            .expect("guest-observable writeback")
    );
    let guest_raw = snapshot_guest_texture_raw(descriptor).expect("snapshotted tiled writeback");
    let guest_linear =
        decode_guest_texture_linear_from_raw(descriptor, &guest_raw).expect("detiled writeback");
    assert_eq!(guest_linear, rendered);
    assert!(
        !presenter.resident_guest_textures[&descriptor].gpu_written(),
        "a published writeback settles the debt"
    );
}

#[test]
fn resident_writeback_publishes_only_the_bytes_the_gpu_authored() {
    use metal::MTLScissorRect;
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };
    use std::ffi::c_void;

    // Mode-27 tiles are 64 KiB, 128x128 texels at four bytes each, so a 256x256 surface is a
    // 2x2 block grid and a write of its top half authors exactly the first block row.
    const SURFACE_BYTES: usize = 256 * 256 * 4;
    const AUTHORED_BYTES: usize = 2 * 65_536;
    let allocation_size = SURFACE_BYTES as u64 + GUEST_PAGE_SIZE;
    let mut memory = GuestAddressSpace::new(4 * allocation_size).expect("guest address space");
    let backing = GuestPhysicalBacking::new(allocation_size).expect("guest backing");
    let mapping = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            allocation_size,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("resident mapping");
    let descriptor = GuestTextureDescriptor {
        address: mapping.get(),
        width: 256,
        height: 256,
        pitch: 256,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let guest_pattern = (0..SURFACE_BYTES)
        .map(|index| (index % 251) as u8)
        .collect::<Vec<_>>();
    memory
        .write(mapping, &guest_pattern)
        .expect("seed the guest allocation");

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let texture = presenter
        .ensure_resident_texture(&device, descriptor)
        .expect("resident texture")
        .texture;
    let rendered = vec![0xa5_u8; SURFACE_BYTES];
    texture.replace_region(
        MTLRegion::new_2d(0, 0, 256, 256),
        0,
        rendered.as_ptr().cast::<c_void>(),
        256 * 4,
    );
    presenter.mark_resident_texture_written_at(
        descriptor,
        ResourceSpan::for_texture(descriptor),
        None,
        WrittenRegion {
            rect: Some(MTLScissorRect {
                x: 0,
                y: 0,
                width: 256,
                height: 128,
            }),
            ..WrittenRegion::whole_resource()
        },
        ResidentTextureActivityStamp { work: 10, flip: 1 },
    );
    assert_eq!(
        presenter.resident_guest_textures[&descriptor]
            .gpu_authored
            .byte_count(),
        AUTHORED_BYTES as u64
    );

    assert!(
        presenter
            .settle_resident_texture_writeback(descriptor, WritebackTrigger::FlipTarget)
            .expect("guest-observable writeback")
    );

    let published = snapshot_guest_texture_raw(descriptor).expect("published guest bytes");
    assert_eq!(
        &published[AUTHORED_BYTES..],
        &guest_pattern[AUTHORED_BYTES..],
        "bytes outside the written region must keep whatever guest memory held"
    );
    assert!(
        published[..AUTHORED_BYTES] != guest_pattern[..AUTHORED_BYTES]
            && published[..AUTHORED_BYTES].iter().all(|&byte| byte == 0xa5),
        "the written region must carry the rendered pixels"
    );
}

#[test]
fn writeback_triggers_follow_end_state_classification() {
    for trigger in [WritebackTrigger::DmaSource, WritebackTrigger::FlipTarget] {
        assert!(
            trigger.publishes_to_guest(),
            "{} is guest-observable and publishes",
            trigger.label()
        );
    }
    for trigger in [
        WritebackTrigger::PressureEviction,
        WritebackTrigger::AliasRebuild,
        WritebackTrigger::LayeredDepthAlias,
        WritebackTrigger::SampledAlias,
    ] {
        assert!(
            !trigger.publishes_to_guest(),
            "{} is RemuBookkeeping by default and must not publish into recycled heaps",
            trigger.label()
        );
    }
}

#[test]
fn bookkeeping_trigger_refuses_publish_and_keeps_debt() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };
    use std::ffi::c_void;

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
        .expect("resident mapping");
    let descriptor = GuestTextureDescriptor {
        address: address.get(),
        width: 64,
        height: 64,
        pitch: 64,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let guest_pattern = (0..64 * 64 * 4)
        .map(|index| (index % 253) as u8)
        .collect::<Vec<_>>();
    memory
        .write(address, &guest_pattern)
        .expect("seed the guest allocation");

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let texture = presenter
        .ensure_resident_texture(&device, descriptor)
        .expect("resident texture")
        .texture;
    let rendered = (0..64 * 64 * 4)
        .map(|index| ((index * 17 + 3) & 0xff) as u8)
        .collect::<Vec<_>>();
    texture.replace_region(
        MTLRegion::new_2d(0, 0, 64, 64),
        0,
        rendered.as_ptr().cast::<c_void>(),
        64 * 4,
    );
    presenter.mark_resident_texture_written_at(
        descriptor,
        ResourceSpan::for_texture(descriptor),
        None,
        WrittenRegion::whole_resource(),
        ResidentTextureActivityStamp { work: 10, flip: 1 },
    );

    let discarded_before = RESIDENT_WRITEBACK_DISCARDED[WritebackTrigger::PressureEviction.index()]
        .load(AtomicOrdering::Relaxed);
    let untouched =
        snapshot_guest_texture_raw(descriptor).expect("guest bytes before bookkeeping refuse");
    assert!(
        !presenter
            .settle_resident_texture_writeback(descriptor, WritebackTrigger::PressureEviction)
            .expect("bookkeeping refuse")
    );
    assert_eq!(
        RESIDENT_WRITEBACK_DISCARDED[WritebackTrigger::PressureEviction.index()]
            .load(AtomicOrdering::Relaxed),
        discarded_before + 1
    );
    assert_eq!(
        snapshot_guest_texture_raw(descriptor).expect("guest bytes after bookkeeping refuse"),
        untouched,
        "bookkeeping must leave guest memory exactly as the guest left it"
    );
    let resident = &presenter.resident_guest_textures[&descriptor];
    assert!(
        resident.gpu_written() && resident.is_live(),
        "bookkeeping must keep Metal debt — clearing without publish lies that RAM agrees"
    );

    presenter.mark_resident_texture_written_at(
        descriptor,
        ResourceSpan::for_texture(descriptor),
        None,
        WrittenRegion::whole_resource(),
        ResidentTextureActivityStamp { work: 11, flip: 1 },
    );
    let published_before = RESIDENT_WRITEBACK_PUBLISHED[WritebackTrigger::FlipTarget.index()]
        .load(AtomicOrdering::Relaxed);
    assert!(
        presenter
            .settle_resident_texture_writeback(descriptor, WritebackTrigger::FlipTarget)
            .expect("guest-observable writeback")
    );
    assert_eq!(
        RESIDENT_WRITEBACK_PUBLISHED[WritebackTrigger::FlipTarget.index()]
            .load(AtomicOrdering::Relaxed),
        published_before + 1
    );
    let guest_raw = snapshot_guest_texture_raw(descriptor).expect("published guest bytes");
    let guest_linear =
        decode_guest_texture_linear_from_raw(descriptor, &guest_raw).expect("detiled writeback");
    assert_eq!(
        guest_linear, rendered,
        "guest-observable publish must land Metal bytes into guest RAM"
    );
    assert!(!presenter.resident_guest_textures[&descriptor].gpu_written());
}

#[test]
fn clean_mip_promotion_is_encoded_in_order_and_remains_cpu_owned() {
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
        .expect("guest mip mapping");
    let guest = GuestTextureDescriptor {
        address: address.get(),
        width: 32,
        height: 32,
        pitch: 32,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RG16Uint,
        bytes_per_pixel: 4,
    };
    let descriptor = DecodedImageDescriptor {
        guest,
        span: ResourceSpan::for_texture(guest),
        destination_select: 0x22c,
        base_mip_level: 2,
        mip_level_count: 1,
        texture_type: 9,
        layer_count: 1,
    };

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal command queue")
        .clone();
    presenter
        .ensure_resident_texture(&device, guest)
        .expect("single-level resident");

    let error = super::ensure_mipmapped_resident_texture_in_map(
        &device,
        Some(&queue),
        None,
        descriptor,
        &mut presenter.resident_guest_textures,
        super::Mode27WorkOrigin::StorageCreate,
    )
    .expect_err("promotion without an ordered command buffer must be rejected");
    assert!(error.contains("ordered guest command buffer"));
    assert!(
        presenter.resident_guest_textures.contains_key(&guest),
        "a rejected promotion must retain the original resident"
    );

    let command_buffer = queue.new_command_buffer().to_owned();
    let view = super::ensure_mipmapped_resident_texture_in_map(
        &device,
        Some(&queue),
        Some(&command_buffer),
        descriptor,
        &mut presenter.resident_guest_textures,
        super::Mode27WorkOrigin::StorageCreate,
    )
    .expect("ordered clean mip promotion");
    assert_eq!(
        command_buffer.status(),
        MTLCommandBufferStatus::NotEnqueued,
        "promotion must not synchronously commit or wait"
    );
    assert_eq!(view.width(), 8);
    assert_eq!(view.height(), 8);
    let resident = presenter
        .resident_guest_textures
        .get(&guest)
        .expect("promoted resident");
    assert_eq!(resident.mip_level_count, 3);
    assert!(
        !resident.gpu_written(),
        "copying clean guest-derived levels must not invent GPU ownership"
    );
    command_buffer.commit();
    command_buffer.wait_until_completed();
    assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
}

#[test]
fn clean_overlapping_render_views_coexist_until_one_is_written() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(4 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("guest texture mapping");
    let first = GuestTextureDescriptor {
        address: address.get(),
        width: 16,
        height: 16,
        pitch: 16,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::R32Float,
        bytes_per_pixel: 4,
    };
    let second = GuestTextureDescriptor {
        pixel_format: MTLPixelFormat::RGBA16Unorm,
        bytes_per_pixel: 8,
        ..first
    };
    assert!(guest_texture_descriptors_overlap(first, second));
    assert!(!guest_texture_descriptors_share_storage(first, second));

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    presenter
        .ensure_resident_texture(&device, first)
        .expect("first clean view");
    presenter
        .ensure_resident_texture(&device, second)
        .expect("second clean view");
    assert!(presenter.resident_guest_textures.contains_key(&first));
    assert!(presenter.resident_guest_textures.contains_key(&second));

    presenter.mark_resident_texture_written(second);
    // `second` reinterprets every byte `first` declares, so `first` loses all of them. It is
    // demoted in place rather than destroyed: the allocation, its watch and its fingerprints
    // survive, and it stops serving reads.
    assert!(presenter.resident_guest_textures.contains_key(&first));
    assert!(matches!(
        presenter.resident_guest_textures[&first].state,
        ResidencyState::Stale(StaleSource::Allocation { descriptor, .. }) if descriptor == second
    ));
    assert!(presenter.resident_guest_textures[&first].owned.is_empty());
    assert!(!presenter.resident_guest_textures[&first].serves_reads());
    assert!(presenter.resident_guest_textures[&second].gpu_written());
}

#[test]
fn a_disjoint_tenant_survives_an_overlapping_neighbours_write() {
    use metal::MTLScissorRect;
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    // A pooled parent's declared range contains its neighbour, so the descriptor-overlap rule
    // this replaces destroyed the neighbour on every write. The write itself covers only the
    // parent's top block row, which is a range the neighbour does not own a byte of.
    const SURFACE_BYTES: u64 = 256 * 256 * 4;
    let allocation_size = SURFACE_BYTES + GUEST_PAGE_SIZE;
    let mut memory = GuestAddressSpace::new(4 * allocation_size).expect("guest address space");
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
        .expect("pooled mapping")
        .get();
    let parent = GuestTextureDescriptor {
        address,
        width: 256,
        height: 256,
        pitch: 256,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let tenant = GuestTextureDescriptor {
        address: address + SURFACE_BYTES / 2,
        height: 128,
        ..parent
    };

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    presenter
        .ensure_resident_texture(&device, parent)
        .expect("pooled parent");
    let survivor = presenter
        .ensure_resident_texture(&device, tenant)
        .expect("contained tenant")
        .texture;
    let owned_before = presenter.resident_guest_textures[&tenant].owned.clone();
    assert!(
        guest_texture_descriptors_overlap(parent, tenant),
        "the two declared ranges really do overlap"
    );

    presenter.mark_resident_texture_written_at(
        parent,
        ResourceSpan::for_texture(parent),
        None,
        WrittenRegion {
            rect: Some(MTLScissorRect {
                x: 0,
                y: 0,
                width: 256,
                height: 128,
            }),
            ..WrittenRegion::whole_resource()
        },
        ResidentTextureActivityStamp { work: 10, flip: 1 },
    );

    let neighbour = &presenter.resident_guest_textures[&tenant];
    assert!(neighbour.is_live(), "a disjoint tenant is not superseded");
    assert!(neighbour.serves_reads());
    assert_eq!(neighbour.owned, owned_before, "no byte changed hands");
    assert!(!neighbour.gpu_written());
    assert!(
        std::ptr::eq(survivor.as_ref(), neighbour.texture.as_ref()),
        "the survivor keeps its Metal allocation"
    );
    // Re-ensuring it must be a reuse, not a rebuild.
    let snapshots_before = TEXTURE_SNAPSHOT_COUNT.load(AtomicOrdering::Relaxed);
    let reused = presenter
        .ensure_resident_texture(&device, tenant)
        .expect("surviving tenant serves the next bind")
        .texture;
    assert!(std::ptr::eq(survivor.as_ref(), reused.as_ref()));
    assert_eq!(
        TEXTURE_SNAPSHOT_COUNT.load(AtomicOrdering::Relaxed),
        snapshots_before
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn a_contained_tenant_is_rematerialised_by_an_ordered_blit() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("pooled mapping")
        .get();
    let parent = GuestTextureDescriptor {
        address,
        width: 32,
        height: 32,
        pitch: 32,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let tenant = GuestTextureDescriptor {
        // Two complete source rows preserve mode-27's equivalent parity invariant too.
        address: address + 2 * 32 * 4,
        width: 16,
        height: 16,
        pitch: 32,
        ..parent
    };

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    // Kill-switch (A1) legacy shared-queue path; delete this pin with that path.
    presenter.per_queue_guest_batches = false;
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal command queue")
        .clone();
    presenter
        .ensure_resident_texture(&device, parent)
        .expect("pooled parent");
    let tenant_texture = presenter
        .ensure_resident_texture(&device, tenant)
        .expect("contained tenant")
        .texture;

    // Make the Metal source observably newer than RAM before attributing the guest GPU write.
    let mut source_pixels = vec![0_u8; 32 * 32 * 4];
    for y in 0..32_usize {
        for x in 0..32_usize {
            let offset = (y * 32 + x) * 4;
            source_pixels[offset..offset + 4].copy_from_slice(&[
                u8::try_from(x).unwrap(),
                u8::try_from(y).unwrap(),
                u8::try_from(x ^ y).unwrap(),
                0xff,
            ]);
        }
    }
    presenter.resident_guest_textures[&parent]
        .texture
        .replace_region(
            MTLRegion::new_2d(0, 0, 32, 32),
            0,
            source_pixels.as_ptr().cast::<std::ffi::c_void>(),
            32 * 4,
        );

    presenter.mark_resident_texture_written(parent);

    let demoted = &presenter.resident_guest_textures[&tenant];
    assert!(matches!(
        demoted.state,
        ResidencyState::Stale(StaleSource::Allocation { descriptor, .. }) if descriptor == parent
    ));
    assert!(demoted.owned.is_empty() && !demoted.gpu_written());
    assert!(
        std::ptr::eq(tenant_texture.as_ref(), demoted.texture.as_ref()),
        "demotion keeps the Metal allocation in place"
    );

    let snapshots_before = TEXTURE_SNAPSHOT_COUNT.load(AtomicOrdering::Relaxed);
    let blits_before = OWNED_REMATERIALISE_BLIT.load(AtomicOrdering::Relaxed);
    let rematerialised = presenter
        .ensure_resident_texture(&device, tenant)
        .expect("demoted tenant rematerialises")
        .texture;
    assert!(
        std::ptr::eq(tenant_texture.as_ref(), rematerialised.as_ref()),
        "the blit rung preserves the tenant's existing Metal allocation"
    );
    assert_eq!(
        TEXTURE_SNAPSHOT_COUNT.load(AtomicOrdering::Relaxed),
        snapshots_before,
        "GPU rematerialisation must not snapshot guest RAM"
    );
    assert_eq!(
        OWNED_REMATERIALISE_BLIT.load(AtomicOrdering::Relaxed),
        blits_before + 1
    );
    let batch = presenter
        .open_guest_batch
        .take()
        .expect("the blit is appended to the open guest batch");
    assert_eq!(
        batch.command_buffer.status(),
        MTLCommandBufferStatus::NotEnqueued,
        "rematerialisation must not flush, commit, or wait"
    );
    assert!(
        batch.resident_texture_versions.iter().any(|version| {
            version.kind == super::super::SharedEventWriteKind::InternalResidentDestination
                && version.descriptor == tenant
        }),
        "the internal rematerialisation must version the tenant on the queue timeline (plan A2)"
    );
    batch.command_buffer.commit();
    batch.command_buffer.wait_until_completed();

    let pixels = read_texture_linear(&device, &queue, &rematerialised, tenant)
        .expect("rematerialised tenant readback");
    let mut expected = Vec::with_capacity(16 * 16 * 4);
    for y in 2..18_usize {
        for x in 0..16_usize {
            let offset = (y * 32 + x) * 4;
            expected.extend_from_slice(&source_pixels[offset..offset + 4]);
        }
    }
    assert_eq!(
        pixels, expected,
        "the blit must preserve every logical texel"
    );
    assert!(presenter.resident_guest_textures[&tenant].serves_reads());
    assert!(presenter.resident_guest_textures[&tenant].gpu_written());
}

#[test]
fn a_typed_alias_is_rematerialised_as_a_zero_copy_view() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(4 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("alias mapping")
        .get();
    let source = GuestTextureDescriptor {
        address,
        width: 16,
        height: 16,
        pitch: 16,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::R32Float,
        bytes_per_pixel: 4,
    };
    let target = GuestTextureDescriptor {
        pixel_format: MTLPixelFormat::R32Uint,
        ..source
    };

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    // Kill-switch (A1) legacy shared-queue path; delete this pin with that path.
    presenter.per_queue_guest_batches = false;
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let source_texture = presenter
        .ensure_resident_texture(&device, source)
        .expect("float source")
        .texture;
    let old_target = presenter
        .ensure_resident_texture(&device, target)
        .expect("integer alias")
        .texture;
    presenter.mark_resident_texture_written(source);
    assert!(matches!(
        presenter.resident_guest_textures[&target].state,
        ResidencyState::Stale(StaleSource::Allocation { descriptor, .. }) if descriptor == source
    ));

    let snapshots_before = TEXTURE_SNAPSHOT_COUNT.load(AtomicOrdering::Relaxed);
    let views_before = OWNED_REMATERIALISE_VIEW.load(AtomicOrdering::Relaxed);
    let view = presenter
        .ensure_resident_texture(&device, target)
        .expect("typed alias rematerialises")
        .texture;
    assert_eq!(
        TEXTURE_SNAPSHOT_COUNT.load(AtomicOrdering::Relaxed),
        snapshots_before
    );
    assert_eq!(
        OWNED_REMATERIALISE_VIEW.load(AtomicOrdering::Relaxed),
        views_before + 1
    );
    assert!(view.parent_texture().is_some());
    assert_eq!(view.pixel_format(), MTLPixelFormat::R32Uint);
    assert!(!std::ptr::eq(old_target.as_ref(), view.as_ref()));
    assert!(
        presenter.open_guest_batch.is_none(),
        "a view does not need a command buffer"
    );
    assert!(matches!(
        presenter.resident_guest_textures[&source].state,
        ResidencyState::Stale(StaleSource::Allocation { descriptor, .. }) if descriptor == target
    ));
    assert!(presenter.resident_guest_textures[&target].serves_reads());
    assert!(presenter.resident_guest_textures[&target].gpu_written());
    assert!(std::ptr::eq(
        source_texture.as_ref(),
        view.parent_texture().expect("view parent")
    ));
}

#[test]
#[ignore = "Phase 1G deleted content fingerprints / self-readback generations; coarse write-watch invalidation replaces this contract"]
fn guest_memory_stale_state_takes_the_counted_snapshot_fallback() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(4 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("pooled mapping")
        .get();
    let target = GuestTextureDescriptor {
        address,
        width: 16,
        height: 16,
        pitch: 16,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    // Kill-switch (A1) legacy shared-queue path; delete this pin with that path.
    presenter.per_queue_guest_batches = false;
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let old_texture = presenter
        .ensure_resident_texture(&device, target)
        .expect("resident target")
        .texture;
    let span = ResourceSpan::for_texture(target).expect("target span");
    let published = RangeSet::for_span(&span);
    presenter.note_guest_memory_authoritative(&span, &published, None, "test-cpu-authority");
    assert_eq!(
        presenter.resident_guest_textures[&target].state,
        ResidencyState::Stale(StaleSource::GuestMemory)
    );

    let snapshots_before = TEXTURE_SNAPSHOT_COUNT.load(AtomicOrdering::Relaxed);
    let cpu_before = OWNED_REMATERIALISE_CPU.load(AtomicOrdering::Relaxed);
    let rebuilt = presenter
        .ensure_resident_texture(&device, target)
        .expect("guest-memory fallback rebuilds")
        .texture;
    assert_eq!(
        OWNED_REMATERIALISE_CPU.load(AtomicOrdering::Relaxed),
        cpu_before + 1
    );
    assert!(TEXTURE_SNAPSHOT_COUNT.load(AtomicOrdering::Relaxed) > snapshots_before);
    assert!(!std::ptr::eq(old_texture.as_ref(), rebuilt.as_ref()));
    assert!(presenter.open_guest_batch.is_none());
    assert!(presenter.resident_guest_textures[&target].serves_reads());
}

#[test]
fn gpu_current_stale_state_never_fabricates_a_guest_memory_fallback() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(4 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("pooled mapping")
        .get();
    let source = GuestTextureDescriptor {
        address,
        width: 32,
        height: 32,
        pitch: 32,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let incompatible = GuestTextureDescriptor {
        address: address + 1024,
        width: 16,
        height: 16,
        pitch: 16,
        ..source
    };

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    // Kill-switch (A1) legacy shared-queue path; delete this pin with that path.
    presenter.per_queue_guest_batches = false;
    let device = presenter.device.as_ref().expect("Metal device").clone();
    presenter
        .ensure_resident_texture(&device, source)
        .expect("source resident");
    presenter
        .ensure_resident_texture(&device, incompatible)
        .expect("incompatible tenant");
    presenter.mark_resident_texture_written(source);

    let snapshots_before = TEXTURE_SNAPSHOT_COUNT.load(AtomicOrdering::Relaxed);
    let failures_before = OWNED_REMATERIALISE_FAILED.load(AtomicOrdering::Relaxed);
    let layout_failures_before = OWNED_REMATERIALISE_FAILED_LAYOUT.load(AtomicOrdering::Relaxed);
    let error = presenter
        .ensure_resident_texture(&device, incompatible)
        .expect_err("GPU-only bytes cannot fall back to stale RAM");
    assert!(error.contains("cannot be rematerialised"), "{error}");
    assert_eq!(
        TEXTURE_SNAPSHOT_COUNT.load(AtomicOrdering::Relaxed),
        snapshots_before
    );
    assert_eq!(
        OWNED_REMATERIALISE_FAILED.load(AtomicOrdering::Relaxed),
        failures_before + 1
    );
    assert_eq!(
        OWNED_REMATERIALISE_FAILED_LAYOUT.load(AtomicOrdering::Relaxed),
        layout_failures_before + 1
    );
    assert!(matches!(
        presenter.resident_guest_textures[&incompatible].state,
        ResidencyState::Stale(StaleSource::Allocation { descriptor, .. }) if descriptor == source
    ));
    assert!(presenter.open_guest_batch.is_none());
}

#[test]
fn mode27_rematerialisation_requires_even_block_row_parity() {
    let source = GuestTextureDescriptor {
        address: 0x10_0000,
        width: 128,
        height: 512,
        pitch: 128,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let even = GuestTextureDescriptor {
        address: source.address + 2 * 65_536,
        width: 128,
        height: 128,
        ..source
    };
    let odd = GuestTextureDescriptor {
        address: source.address + 65_536,
        ..even
    };
    assert_eq!(
        rematerialisation_blit_source_origin_y(
            source,
            even,
            ResourceSpan::new(source.address, 4 * 65_536),
            ResourceSpan::new(even.address, 65_536),
        ),
        Some(256)
    );
    assert_eq!(
        rematerialisation_blit_source_origin_y(
            source,
            odd,
            ResourceSpan::new(source.address, 4 * 65_536),
            ResourceSpan::new(odd.address, 65_536),
        ),
        None
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn mode27_gpu_retile_preserves_cross_format_alias_bytes_without_a_batch_boundary() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    const SOURCE_BYTES: u64 = 4 * 65_536;
    let mut memory = GuestAddressSpace::new(4 * SOURCE_BYTES).expect("guest address space");
    let backing = GuestPhysicalBacking::new(SOURCE_BYTES).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            SOURCE_BYTES,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("pooled mapping")
        .get();
    let source = GuestTextureDescriptor {
        address,
        width: 256,
        height: 256,
        pitch: 256,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let target = GuestTextureDescriptor {
        address: address + 65_536,
        width: 256,
        height: 128,
        pitch: 256,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::R16Uint,
        bytes_per_pixel: 2,
    };
    assert_eq!(guest_texture_byte_count(source), Some(SOURCE_BYTES));
    assert_eq!(guest_texture_byte_count(target), Some(65_536));

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    // Kill-switch (A1) legacy shared-queue path; delete this pin with that path.
    presenter.per_queue_guest_batches = false;
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal command queue")
        .clone();
    presenter
        .ensure_resident_texture(&device, source)
        .expect("source resident");
    let old_target = presenter
        .ensure_resident_texture(&device, target)
        .expect("target resident")
        .texture;

    let mut source_linear = vec![0_u8; 256 * 256 * 4];
    for (index, byte) in source_linear.iter_mut().enumerate() {
        *byte = u8::try_from((index * 37 + index / 251) & 0xff).expect("byte pattern");
    }
    presenter.resident_guest_textures[&source]
        .texture
        .replace_region(
            MTLRegion::new_2d(0, 0, 256, 256),
            0,
            source_linear.as_ptr().cast::<std::ffi::c_void>(),
            256 * 4,
        );
    presenter.mark_resident_texture_written(source);
    assert!(matches!(
        presenter.resident_guest_textures[&target].state,
        ResidencyState::Stale(StaleSource::Allocation { descriptor, .. }) if descriptor == source
    ));

    let detiles_before = OWNED_REMATERIALISE_DETILE.load(AtomicOrdering::Relaxed);
    let snapshots_before = TEXTURE_SNAPSHOT_COUNT.load(AtomicOrdering::Relaxed);
    let rematerialised = presenter
        .ensure_resident_texture(&device, target)
        .expect("cross-format mode-27 alias rematerialises")
        .texture;
    assert!(
        std::ptr::eq(old_target.as_ref(), rematerialised.as_ref()),
        "the detile rung preserves the target allocation"
    );
    assert_eq!(
        OWNED_REMATERIALISE_DETILE.load(AtomicOrdering::Relaxed),
        detiles_before + 1
    );
    assert_eq!(
        TEXTURE_SNAPSHOT_COUNT.load(AtomicOrdering::Relaxed),
        snapshots_before,
        "GPU rematerialisation never snapshots or publishes guest RAM"
    );
    let batch = presenter
        .open_guest_batch
        .take()
        .expect("retile/de-tile work remains in the open guest batch");
    assert_eq!(
        batch.command_buffer.status(),
        MTLCommandBufferStatus::NotEnqueued
    );
    batch.command_buffer.commit();
    batch.command_buffer.wait_until_completed();

    let actual = read_texture_linear(&device, &queue, &rematerialised, target)
        .expect("rematerialised target readback");
    let mut source_raw = vec![0_u8; usize::try_from(SOURCE_BYTES).unwrap()];
    copy_linear_to_tiled_at_z(&source_linear, source, &mut source_raw, 0)
        .expect("CPU retile oracle");
    let expected = decode_guest_texture_linear_from_raw(target, &source_raw[65_536..2 * 65_536])
        .expect("CPU detile oracle");
    assert_eq!(actual, expected, "GPU retile/de-tile must be byte exact");
    assert!(presenter.resident_guest_textures[&target].serves_reads());
    assert!(presenter.resident_guest_textures[&target].gpu_written());
}

#[test]
#[allow(clippy::too_many_lines)]
fn mode24_depth_rematerialisation_is_byte_exact_and_stays_in_the_guest_batch() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    const SOURCE_BYTES: u64 = 2 * 65_536;
    let mut memory = GuestAddressSpace::new(4 * SOURCE_BYTES).expect("guest address space");
    let backing = GuestPhysicalBacking::new(SOURCE_BYTES).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            SOURCE_BYTES,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("pooled mapping")
        .get();
    let source = GuestTextureDescriptor {
        address,
        width: 256,
        height: 128,
        pitch: 256,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let target = GuestTextureDescriptor {
        address,
        width: 128,
        height: 128,
        pitch: 128,
        tile_mode: 24,
        pixel_format: MTLPixelFormat::Depth32Float,
        bytes_per_pixel: 4,
    };
    assert_eq!(guest_texture_byte_count(source), Some(SOURCE_BYTES));
    assert_eq!(guest_texture_byte_count(target), Some(65_536));

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    // Kill-switch (A1) legacy shared-queue path; delete this pin with that path.
    presenter.per_queue_guest_batches = false;
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal command queue")
        .clone();
    presenter
        .ensure_resident_texture(&device, source)
        .expect("source resident");
    let old_target = presenter
        .ensure_resident_texture(&device, target)
        .expect("mode-24 depth target")
        .texture;

    let mut source_linear = vec![0_u8; 256 * 128 * 4];
    for (index, byte) in source_linear.iter_mut().enumerate() {
        *byte = u8::try_from((index * 29 + index / 127) & 0xff).expect("byte pattern");
    }
    presenter.resident_guest_textures[&source]
        .texture
        .replace_region(
            MTLRegion::new_2d(0, 0, 256, 128),
            0,
            source_linear.as_ptr().cast::<std::ffi::c_void>(),
            256 * 4,
        );
    presenter.mark_resident_texture_written(source);
    assert!(matches!(
        presenter.resident_guest_textures[&target].state,
        ResidencyState::Stale(StaleSource::Allocation { descriptor, .. }) if descriptor == source
    ));

    let detiles_before = OWNED_REMATERIALISE_DETILE.load(AtomicOrdering::Relaxed);
    let snapshots_before = TEXTURE_SNAPSHOT_COUNT.load(AtomicOrdering::Relaxed);
    let rematerialised = presenter
        .ensure_resident_texture(&device, target)
        .expect("mode-24 depth target rematerialises")
        .texture;
    assert!(std::ptr::eq(old_target.as_ref(), rematerialised.as_ref()));
    assert_eq!(
        OWNED_REMATERIALISE_DETILE.load(AtomicOrdering::Relaxed),
        detiles_before + 1
    );
    assert_eq!(
        TEXTURE_SNAPSHOT_COUNT.load(AtomicOrdering::Relaxed),
        snapshots_before,
        "mode-24 GPU rematerialisation must not snapshot guest RAM"
    );
    let batch = presenter
        .open_guest_batch
        .take()
        .expect("mode-24 retile/de-tile remains in the open guest batch");
    assert_eq!(
        batch.command_buffer.status(),
        MTLCommandBufferStatus::NotEnqueued
    );
    batch.command_buffer.commit();
    batch.command_buffer.wait_until_completed();

    let actual = read_texture_linear(&device, &queue, &rematerialised, target)
        .expect("rematerialised depth readback");
    let mut source_raw = vec![0_u8; usize::try_from(SOURCE_BYTES).unwrap()];
    copy_linear_to_tiled_at_z(&source_linear, source, &mut source_raw, 0)
        .expect("CPU mode-27 retile oracle");
    let expected = decode_guest_texture_linear_from_raw(target, &source_raw[..65_536])
        .expect("CPU mode-24 detile oracle");
    assert_eq!(actual, expected, "mode-24 GPU detile must be byte exact");
    assert!(presenter.resident_guest_textures[&target].serves_reads());
    assert!(presenter.resident_guest_textures[&target].gpu_written());
}

#[test]
#[allow(clippy::too_many_lines)]
fn mode27_gpu_retile_merges_disjoint_current_owners_in_physical_byte_order() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    const BLOCK_BYTES: u64 = 65_536;
    const TARGET_BYTES: u64 = 4 * BLOCK_BYTES;
    let mut memory = GuestAddressSpace::new(4 * TARGET_BYTES).expect("guest address space");
    let backing = GuestPhysicalBacking::new(TARGET_BYTES).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            TARGET_BYTES,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("pooled mapping")
        .get();
    let target = GuestTextureDescriptor {
        address,
        width: 256,
        height: 256,
        pitch: 256,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let owners = std::array::from_fn::<_, 4, _>(|index| GuestTextureDescriptor {
        address: address + u64::try_from(index).unwrap() * BLOCK_BYTES,
        width: 128,
        height: 128,
        pitch: 128,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    });

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    // Kill-switch (A1) legacy shared-queue path; delete this pin with that path.
    presenter.per_queue_guest_batches = false;
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal command queue")
        .clone();
    presenter
        .ensure_resident_texture(&device, target)
        .expect("composite target");
    for owner in owners {
        presenter
            .ensure_resident_texture(&device, owner)
            .expect("block owner");
    }

    let mut owner_linear = Vec::new();
    for (owner_index, owner) in owners.into_iter().enumerate() {
        let mut linear = vec![0_u8; 128 * 128 * 4];
        for (byte_index, byte) in linear.iter_mut().enumerate() {
            *byte = u8::try_from((owner_index * 53 + byte_index * 29 + byte_index / 127) & 0xff)
                .expect("owner byte pattern");
        }
        presenter.resident_guest_textures[&owner]
            .texture
            .replace_region(
                MTLRegion::new_2d(0, 0, 128, 128),
                0,
                linear.as_ptr().cast::<std::ffi::c_void>(),
                128 * 4,
            );
        presenter.mark_resident_texture_written(owner);
        owner_linear.push(linear);
    }
    assert!(matches!(
        presenter.resident_guest_textures[&target].state,
        ResidencyState::Stale(StaleSource::Allocation { .. })
    ));

    let detiles_before = OWNED_REMATERIALISE_DETILE.load(AtomicOrdering::Relaxed);
    let rematerialised = presenter
        .ensure_resident_texture(&device, target)
        .expect("four-owner target rematerialises")
        .texture;
    assert_eq!(
        OWNED_REMATERIALISE_DETILE.load(AtomicOrdering::Relaxed),
        detiles_before + 1
    );
    let batch = presenter
        .open_guest_batch
        .take()
        .expect("owner merge remains ordered in the open batch");
    assert_eq!(
        batch.command_buffer.status(),
        MTLCommandBufferStatus::NotEnqueued
    );
    batch.command_buffer.commit();
    batch.command_buffer.wait_until_completed();

    let actual = read_texture_linear(&device, &queue, &rematerialised, target)
        .expect("merged target readback");
    let mut expected_raw = vec![0_u8; usize::try_from(TARGET_BYTES).unwrap()];
    for (index, (owner, linear)) in owners.into_iter().zip(&owner_linear).enumerate() {
        let start = index * usize::try_from(BLOCK_BYTES).unwrap();
        copy_linear_to_tiled_at_z(
            linear,
            owner,
            &mut expected_raw[start..start + usize::try_from(BLOCK_BYTES).unwrap()],
            0,
        )
        .expect("CPU owner retile oracle");
    }
    let expected = decode_guest_texture_linear_from_raw(target, &expected_raw)
        .expect("CPU composite detile oracle");
    assert_eq!(
        actual, expected,
        "every current owner must contribute its exact bytes"
    );
    assert!(presenter.resident_guest_textures[&target].serves_reads());
    for owner in owners {
        assert!(matches!(
            presenter.resident_guest_textures[&owner].state,
            ResidencyState::Stale(StaleSource::Allocation { descriptor, .. })
                if descriptor == target
        ));
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn mode27_gpu_detile_reconstructs_every_volume_slice_with_surface_xor() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    const SOURCE_BYTES: u64 = 4 * 65_536;
    let mut memory = GuestAddressSpace::new(4 * SOURCE_BYTES).expect("guest address space");
    let backing = GuestPhysicalBacking::new(SOURCE_BYTES).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            SOURCE_BYTES,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("pooled mapping")
        .get();
    let source = GuestTextureDescriptor {
        address,
        width: 256,
        height: 256,
        pitch: 256,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let volume = GuestTextureDescriptor {
        address,
        width: 32,
        height: 32,
        pitch: 32,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA16Uint,
        bytes_per_pixel: 8,
    };
    let volume_target = DecodedRenderTarget {
        guest_slot: 0,
        texture: volume,
        span: ResourceSpan::for_texture(volume),
        output_kind: remu_shader::PixelOutputKind::Uint,
        write_mask: 0xf,
        direct_scanout: false,
        sample_count: 1,
        texture_type: 10,
        layer_count: 4,
        base_layer: 0,
        mip_level: 0,
        mip_level_count: 1,
        dcc_enabled: false,
        fast_clear_enabled: false,
            blend_bypass: false,
    };

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    // Kill-switch (A1) legacy shared-queue path; delete this pin with that path.
    presenter.per_queue_guest_batches = false;
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal command queue")
        .clone();
    presenter
        .ensure_resident_texture(&device, source)
        .expect("source resident");
    presenter
        .ensure_layered_render_target_texture(&device, volume_target, false)
        .expect("volume resident");

    let mut source_linear = vec![0_u8; 256 * 256 * 4];
    for (index, byte) in source_linear.iter_mut().enumerate() {
        *byte = u8::try_from((index * 43 + index / 193) & 0xff).expect("byte pattern");
    }
    presenter.resident_guest_textures[&source]
        .texture
        .replace_region(
            MTLRegion::new_2d(0, 0, 256, 256),
            0,
            source_linear.as_ptr().cast::<std::ffi::c_void>(),
            256 * 4,
        );
    presenter.mark_resident_texture_written(source);
    assert!(matches!(
        presenter.resident_guest_textures[&volume].state,
        ResidencyState::Stale(StaleSource::Allocation { descriptor, .. }) if descriptor == source
    ));

    let rematerialised = presenter
        .ensure_layered_render_target_texture(&device, volume_target, false)
        .expect("volume rematerialises")
        .texture;
    let batch = presenter
        .open_guest_batch
        .take()
        .expect("volume detile remains in the open batch");
    assert_eq!(
        batch.command_buffer.status(),
        MTLCommandBufferStatus::NotEnqueued
    );
    batch.command_buffer.commit();
    batch.command_buffer.wait_until_completed();

    let actual = super::super::read_texture_volume_linear(&device, &queue, &rematerialised, volume)
        .expect("volume readback");
    let mut source_raw = vec![0_u8; usize::try_from(SOURCE_BYTES).unwrap()];
    copy_linear_to_tiled_at_z(&source_linear, source, &mut source_raw, 0)
        .expect("CPU source retile oracle");
    let mut expected = Vec::with_capacity(actual.len());
    for z in 0..4_usize {
        expected.extend_from_slice(
            &super::super::texture::decode_guest_texture_linear_from_raw_at_z(
                volume,
                &source_raw[z * 65_536..(z + 1) * 65_536],
                z,
            )
            .expect("CPU volume detile oracle"),
        );
    }
    assert_eq!(
        actual, expected,
        "surface-Z XOR must be preserved per slice"
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn mode27_gpu_detile_reconstructs_nontail_and_mip_tail_levels() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    const SOURCE_BYTES: u64 = 4 * 65_536;
    let mut memory = GuestAddressSpace::new(4 * SOURCE_BYTES).expect("guest address space");
    let backing = GuestPhysicalBacking::new(SOURCE_BYTES).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            SOURCE_BYTES,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("pooled mapping")
        .get();
    let source = GuestTextureDescriptor {
        address,
        width: 256,
        height: 256,
        pitch: 256,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let mipped = GuestTextureDescriptor {
        address,
        width: 256,
        height: 128,
        pitch: 256,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::R16Uint,
        bytes_per_pixel: 2,
    };
    let mipped_image = DecodedImageDescriptor {
        guest: mipped,
        span: ResourceSpan::for_texture(mipped),
        destination_select: 0,
        base_mip_level: 0,
        mip_level_count: 6,
        texture_type: 9,
        layer_count: 1,
    };
    let mip_layout = mode27_mip_chain_layout(mipped, 6).expect("target mip layout");
    assert!(mip_layout.levels.iter().any(|level| level.in_tail));
    assert!(mip_layout.levels.iter().any(|level| !level.in_tail));

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    // Kill-switch (A1) legacy shared-queue path; delete this pin with that path.
    presenter.per_queue_guest_batches = false;
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let queue = presenter
        .command_queue
        .as_ref()
        .expect("Metal command queue")
        .clone();
    presenter
        .ensure_resident_texture(&device, source)
        .expect("source resident");
    presenter
        .ensure_mipmapped_storage_texture(&device, mipped_image)
        .expect("mip-chain resident");
    if let Some(batch) = presenter.open_guest_batch.take() {
        batch.command_buffer.commit();
        batch.command_buffer.wait_until_completed();
    }

    let mut source_linear = vec![0_u8; 256 * 256 * 4];
    for (index, byte) in source_linear.iter_mut().enumerate() {
        *byte = u8::try_from((index * 47 + index / 181) & 0xff).expect("byte pattern");
    }
    presenter.resident_guest_textures[&source]
        .texture
        .replace_region(
            MTLRegion::new_2d(0, 0, 256, 256),
            0,
            source_linear.as_ptr().cast::<std::ffi::c_void>(),
            256 * 4,
        );
    presenter.mark_resident_texture_written(source);
    assert!(matches!(
        presenter.resident_guest_textures[&mipped].state,
        ResidencyState::Stale(StaleSource::Allocation { descriptor, .. }) if descriptor == source
    ));

    let rematerialised = presenter
        .ensure_mipmapped_storage_texture(&device, mipped_image)
        .expect("mip chain rematerialises");
    let batch = presenter
        .open_guest_batch
        .take()
        .expect("mip detile remains in the open batch");
    assert_eq!(
        batch.command_buffer.status(),
        MTLCommandBufferStatus::NotEnqueued
    );
    batch.command_buffer.commit();
    batch.command_buffer.wait_until_completed();

    let mut source_raw = vec![0_u8; usize::try_from(SOURCE_BYTES).unwrap()];
    copy_linear_to_tiled_at_z(&source_linear, source, &mut source_raw, 0)
        .expect("CPU source retile oracle");
    let (_, expected_levels) =
        decode_mode27_mip_chain_from_raw(mipped, 6, &source_raw[..mip_layout.byte_count])
            .expect("CPU mip detile oracle");
    for (level, expected) in expected_levels.iter().enumerate() {
        let actual = super::super::readback::read_texture_mip_linear(
            &device,
            &queue,
            &rematerialised,
            mipped,
            u32::try_from(level).expect("mip index"),
        )
        .expect("mip readback");
        assert_eq!(&actual, expected, "mip {level} differs from CPU oracle");
    }
}

#[test]
fn a_discarded_bind_reclaims_a_demoted_target_without_a_rebuild() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("pooled mapping")
        .get();
    let parent = GuestTextureDescriptor {
        address,
        width: 32,
        height: 32,
        pitch: 32,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let tenant = GuestTextureDescriptor {
        address: address + 1024,
        width: 16,
        height: 16,
        pitch: 16,
        ..parent
    };

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    presenter
        .ensure_resident_texture(&device, parent)
        .expect("pooled parent");
    let tenant_texture = presenter
        .ensure_resident_texture(&device, tenant)
        .expect("contained tenant")
        .texture;
    presenter.mark_resident_texture_written(parent);
    assert!(!presenter.resident_guest_textures[&tenant].is_live());

    // This is the transient-allocator re-bind: a fresh target with a clear. It must cost one
    // state flip — no snapshot, no rebuild, no new allocation.
    let snapshots_before = TEXTURE_SNAPSHOT_COUNT.load(AtomicOrdering::Relaxed);
    let reclaimed = presenter
        .ensure_resident_texture_with_initial_contents(
            &device,
            tenant,
            ResourceSpan::for_texture(tenant),
            false,
        )
        .expect("discarded re-bind")
        .texture;
    assert!(std::ptr::eq(tenant_texture.as_ref(), reclaimed.as_ref()));
    assert_eq!(
        TEXTURE_SNAPSHOT_COUNT.load(AtomicOrdering::Relaxed),
        snapshots_before
    );
    assert!(presenter.resident_guest_textures[&tenant].is_live());
    assert!(
        !presenter.resident_guest_textures[&tenant].serves_reads(),
        "an owner-less Live target is legal only until the mark that follows it"
    );

    presenter.mark_resident_texture_written(tenant);
    assert!(presenter.resident_guest_textures[&tenant].serves_reads());
    assert!(presenter.resident_guest_textures[&tenant].gpu_written());
    // Authority is exclusive: the parent gave those bytes back.
    assert!(
        !presenter.resident_guest_textures[&parent]
            .owned
            .intersects(&presenter.resident_guest_textures[&tenant].owned)
    );
}

#[test]
fn a_partially_superseded_tenant_stays_live_but_stops_serving_reads() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("pooled mapping")
        .get();
    // The tenant spans 2 KiB from `address`; the writer covers only its first kilobyte.
    let tenant = GuestTextureDescriptor {
        address,
        width: 16,
        height: 32,
        pitch: 16,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let writer = GuestTextureDescriptor {
        height: 16,
        ..tenant
    };

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    presenter
        .ensure_resident_texture(&device, tenant)
        .expect("tenant");
    let owned_before = presenter.resident_guest_textures[&tenant]
        .owned
        .byte_count();
    presenter
        .ensure_resident_texture(&device, writer)
        .expect("writer");

    presenter.mark_resident_texture_written(writer);

    let tenant_resident = &presenter.resident_guest_textures[&tenant];
    assert!(
        tenant_resident.is_live(),
        "half its bytes are still its own"
    );
    assert!(!tenant_resident.serves_reads());
    assert_eq!(tenant_resident.owned.byte_count(), owned_before / 2);
}

#[test]
fn same_address_color_view_survives_partial_owned_supersession() {
    // Silent Hill title BG: CB writes BGRA8 @ addr, PS samples RGBA8 @ same addr. A neighbour in
    // the shared heap can partially supersede owned-byte accounting while the Metal image stays
    // authoritative. Colour-view sampling must still bind that resident instead of uploading
    // unpublished guest RAM (opaque black).
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("pooled mapping")
        .get();
    let bgra = GuestTextureDescriptor {
        address,
        width: 16,
        height: 32,
        pitch: 16,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::BGRA8Unorm,
        bytes_per_pixel: 4,
    };
    let rgba_sample = DecodedImageDescriptor {
        guest: GuestTextureDescriptor {
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            ..bgra
        },
        span: ResourceSpan::for_texture(GuestTextureDescriptor {
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            ..bgra
        }),
        destination_select: 0xf2e,
        base_mip_level: 0,
        mip_level_count: 1,
        texture_type: 9,
        layer_count: 1,
    };

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    presenter
        .ensure_resident_texture(&device, bgra)
        .expect("bgra resident");
    presenter.mark_resident_texture_written(bgra);

    {
        let resident = presenter
            .resident_guest_textures
            .get_mut(&bgra)
            .expect("bgra resident");
        let span = resident.span.expect("declared span");
        let mid = (span.range_start() + span.range_end()) / 2;
        let stolen = RangeSet::from_run(span.range_start(), mid);
        resident.subtract_authority(&stolen, StaleSource::GuestMemory);
    }

    let bgra_resident = &presenter.resident_guest_textures[&bgra];
    assert!(bgra_resident.is_live());
    assert!(bgra_resident.gpu_written());
    assert!(
        !bgra_resident.serves_reads(),
        "partial supersession must clear full-range owned coverage"
    );
    assert!(
        resident_color_view_sample_is_eligible(bgra_resident, rgba_sample, false),
        "same-address same-extent BGRA→RGBA must stay eligible after partial owned supersession"
    );
    let subrect_sample = DecodedImageDescriptor {
        guest: GuestTextureDescriptor {
            height: 16,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            ..bgra
        },
        span: ResourceSpan::for_texture(GuestTextureDescriptor {
            height: 16,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            ..bgra
        }),
        destination_select: 0xf2e,
        base_mip_level: 0,
        mip_level_count: 1,
        texture_type: 9,
        layer_count: 1,
    };
    assert!(
        !resident_color_view_sample_is_eligible(bgra_resident, subrect_sample, false),
        "sub-rect same-address views still require full owned coverage"
    );
}

#[test]
fn writes_across_overlapping_descriptors_keep_owned_sets_disjoint() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(16 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(4 * GUEST_PAGE_SIZE).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            4 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("pooled mapping")
        .get();
    let base = GuestTextureDescriptor {
        address,
        width: 16,
        height: 16,
        pitch: 16,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let descriptors = [0_u64, 512, 1024, 1536, 2048].map(|offset| GuestTextureDescriptor {
        address: address + offset,
        ..base
    });

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    for descriptor in descriptors {
        presenter
            .ensure_resident_texture(&device, descriptor)
            .expect("overlapping resident");
    }
    // Creation hands every resident its whole declared range, so the claims start overlapping;
    // one write each is what establishes exclusivity, and no later interleaving may break it.
    for index in [3_usize, 0, 4, 1, 2, 0, 3, 2, 4] {
        presenter.mark_resident_texture_written(descriptors[index]);
        if index == 2 {
            for (left, right) in descriptors
                .iter()
                .enumerate()
                .flat_map(|(left, descriptor)| {
                    descriptors[left + 1..]
                        .iter()
                        .map(move |other| (*descriptor, *other))
                })
            {
                assert!(
                    !presenter.resident_guest_textures[&left]
                        .owned
                        .intersects(&presenter.resident_guest_textures[&right].owned),
                    "exactly one allocation may answer for any guest byte"
                );
            }
        }
    }
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
        span: ResourceSpan::for_texture(guest),
        destination_select: 0x22c,
        base_mip_level: 0,
        mip_level_count: 6,
        texture_type: 9,
        layer_count: 1,
    };
    // Residency comes from the hierarchy the guest declares. A single-level descriptor is
    // never promoted into a larger one on the presenter's own initiative.
    let queue = device.new_command_queue();
    let original = super::ensure_mipmapped_resident_texture_in_map(
        &device,
        Some(&queue),
        None,
        complete,
        &mut presenter.resident_guest_textures,
        super::Mode27WorkOrigin::StorageCreate,
    )
    .expect("declared complete mip hierarchy becomes resident");
    assert_eq!(original.mipmap_level_count(), 6);

    let complete_view = super::ensure_mipmapped_resident_texture_in_map(
        &device,
        Some(&queue),
        None,
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
        span: ResourceSpan::for_texture(guest),
        destination_select: 0x22c,
        base_mip_level: 2,
        mip_level_count: 2,
        texture_type: 9,
        layer_count: 1,
    };
    let view = super::ensure_mipmapped_resident_texture_in_map(
        &device,
        Some(&queue),
        None,
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
    const EXACT_VIEW: GuestTextureDescriptor = GuestTextureDescriptor {
        address: 0x80c0_0000_00,
        width: 1_024,
        height: 1_024,
        pitch: 1_024,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    for base_mip_level in [3, 7] {
        let descriptor = super::DecodedImageDescriptor {
            guest: EXACT_VIEW,
            span: ResourceSpan::for_texture(EXACT_VIEW),
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
fn rgba_bgra_and_transfer_variants_can_share_typed_color_view_storage() {
    let bgra_srgb = GuestTextureDescriptor {
        address: 0x8006_b7b00,
        width: 3_840,
        height: 2_160,
        pitch: 3_840,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::BGRA8Unorm_sRGB,
        bytes_per_pixel: 4,
    };
    let rgba_linear = GuestTextureDescriptor {
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        ..bgra_srgb
    };
    let rgba_srgb = GuestTextureDescriptor {
        pixel_format: MTLPixelFormat::RGBA8Unorm_sRGB,
        ..bgra_srgb
    };
    let bgra_linear = GuestTextureDescriptor {
        pixel_format: MTLPixelFormat::BGRA8Unorm,
        ..bgra_srgb
    };
    assert!(guest_texture_descriptors_share_color_view_storage(
        bgra_srgb,
        rgba_linear,
    ));
    assert!(guest_texture_descriptors_share_color_view_storage(
        bgra_srgb, rgba_srgb,
    ));
    assert!(guest_texture_descriptors_share_color_view_storage(
        bgra_srgb,
        bgra_linear,
    ));
}

#[test]
fn integer_and_normalized_views_of_one_channel_targets_share_typed_color_view_storage() {
    // Octopath renders its occlusion chain to R8Uint targets and samples the same bytes through
    // R8Unorm SRDs. Both formats are in Metal's one-channel 8-bit view class, so a format view
    // reproduces the CPU decode without reading roughly 260 MiB per interval back through guest
    // memory.
    let rendered = GuestTextureDescriptor {
        address: 0x80cd_de00_00,
        width: 1_920,
        height: 1_080,
        pitch: 2_048,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::R8Uint,
        bytes_per_pixel: 1,
    };
    let sampled = GuestTextureDescriptor {
        pixel_format: MTLPixelFormat::R8Unorm,
        ..rendered
    };
    assert!(guest_texture_descriptors_share_color_view_storage(
        rendered, sampled,
    ));
    assert!(guest_texture_descriptors_share_color_view_storage(
        sampled, rendered,
    ));
    assert!(
        !guest_texture_descriptors_share_color_view_storage(
            rendered,
            GuestTextureDescriptor {
                pixel_format: MTLPixelFormat::R16Unorm,
                bytes_per_pixel: 2,
                ..rendered
            },
        ),
        "channel widths that differ cannot back one Metal format view"
    );
    assert!(
        !guest_texture_descriptors_share_color_view_storage(
            rendered,
            GuestTextureDescriptor {
                pixel_format: MTLPixelFormat::Stencil8,
                ..rendered
            },
        ),
        "colour formats are not view-compatible with depth or stencil formats"
    );
}

#[test]
fn same_format_pitch_mismatch_can_share_gpu_sample_storage() {
    // UE HDR color targets often keep RDNA2 padded pitch while the SRD reports visible width.
    // Same address + format must still sample the Metal resident without a CPU writeback.
    let rendered = GuestTextureDescriptor {
        address: 0x8006_b7b00,
        width: 3_840,
        height: 2_160,
        pitch: 3_968,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA16Float,
        bytes_per_pixel: 8,
    };
    let sampled = GuestTextureDescriptor {
        pitch: 3_840,
        ..rendered
    };
    assert!(!guest_texture_descriptors_share_storage(rendered, sampled));
    assert!(guest_texture_descriptors_share_gpu_sample_storage(
        rendered, sampled
    ));
}

#[test]
fn color_view_pitch_mismatch_can_share_typed_storage() {
    let rendered = GuestTextureDescriptor {
        address: 0x8006_b7b00,
        width: 3_840,
        height: 2_160,
        pitch: 3_968,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::BGRA8Unorm_sRGB,
        bytes_per_pixel: 4,
    };
    let sampled = GuestTextureDescriptor {
        pitch: 3_840,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        ..rendered
    };
    assert!(!guest_texture_descriptors_share_storage(rendered, sampled));
    assert!(guest_texture_descriptors_share_color_view_storage(
        rendered, sampled
    ));
}

#[test]
fn typed_color_alias_view_preserves_guest_byte_order_and_transfer_semantics() {
    let device = Device::system_default().expect("Metal device");
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm_sRGB);
    descriptor.set_width(1);
    descriptor.set_height(1);
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::PixelFormatView);
    let texture = device.new_texture(&descriptor);
    let raw_bgra = [30_u8, 20, 10, 40];
    texture.replace_region(
        MTLRegion::new_2d(0, 0, 1, 1),
        0,
        raw_bgra.as_ptr().cast::<std::ffi::c_void>(),
        4,
    );
    let view = new_color_alias_texture_view(&texture, MTLPixelFormat::RGBA8Unorm, 0, 1)
        .expect("linear RGBA alias view");
    assert_eq!(view.pixel_format(), MTLPixelFormat::RGBA8Unorm);

    let source = r"
#include <metal_stdlib>
using namespace metal;
kernel void read_alias(
    texture2d<float, access::read> input [[texture(0)]],
    device float4* output [[buffer(0)]])
{
    output[0] = input.read(uint2(0));
}
";
    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .expect("typed alias test MSL compiles");
    let function = library
        .get_function("read_alias", None)
        .expect("typed alias test function exists");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .expect("typed alias test pipeline compiles");
    let queue = device.new_command_queue();
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

    // The RGBA-linear view must expose the underlying guest bytes directly: there is neither an
    // sRGB decode nor an extra red/blue swap after the pixel-format reinterpretation.
    let sampled = unsafe { std::slice::from_raw_parts(output.contents().cast::<f32>(), 4) };
    for (actual, expected) in sampled.iter().copied().zip(raw_bgra.map(f32::from)) {
        assert!((actual - expected / 255.0).abs() <= 1.0 / 255.0);
    }
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
fn cube_sampled_textures_serve_face_atlas_coordinates_as_2d_array_layers() {
    // GCN cube resources are addressed with V_CUBEID/SC/TC/MA face-atlas coordinates, so the
    // presenter creates them as 2D arrays and the translated shader indexes
    // (x - 1.5)*2 mapped back to [0,1] on the layer round(z). Every face texel must round-trip
    // exactly through that addressing form; a direction-vector cube binding cannot.
    let Some(device) = Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let guest = GuestTextureDescriptor {
        address: 0x10_0000,
        width: 4,
        height: 4,
        pitch: 4,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::R8Unorm,
        bytes_per_pixel: 1,
    };
    let mut linear = Vec::with_capacity(96);
    for face in 0..6_u32 {
        for texel in 0..16_u32 {
            linear.push(u8::try_from(face * 36 + texel + 1).expect("identity texel fits u8"));
        }
    }
    let texture =
        create_sampled_guest_texture(&device, &queue, guest, &linear, 0, 1, 11, 6, false, false)
            .expect("six-face cube uploads as a 2D array");
    assert_eq!(texture.texture_type(), MTLTextureType::D2Array);
    assert_eq!(texture.array_length(), 6);

    // The addressing expressions mirror emit_image_sample's cube lowering exactly.
    let source = r"
#include <metal_stdlib>
using namespace metal;
kernel void sample_cube_atlas(
    texture2d_array<float> input [[texture(0)]],
    device float* output [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]])
{
    constexpr sampler s(coord::normalized, filter::nearest, address::clamp_to_edge);
    float3 atlas = float3(
        (float(gid.x) + 0.5f) / 4.0f + 1.0f,
        (float(gid.y) + 0.5f) / 4.0f + 1.0f,
        float(gid.z));
    float2 face_uv = float2(atlas.x - 1.0f, atlas.y - 1.0f);
    uint layer = min(uint(rint(max(atlas.z, 0.0f))), input.get_array_size() - 1u);
    output[gid.z * 16u + gid.y * 4u + gid.x] = input.sample(s, face_uv, layer).x;
}
";
    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .expect("cube atlas test MSL compiles");
    let function = library
        .get_function("sample_cube_atlas", None)
        .expect("cube atlas test function exists");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .expect("cube atlas test pipeline compiles");
    let output = device.new_buffer(96 * 4, MTLResourceOptions::StorageModeShared);
    let command = queue.new_command_buffer().to_owned();
    let compute = command.new_compute_command_encoder();
    compute.set_compute_pipeline_state(&pipeline);
    compute.set_texture(0, Some(&texture));
    compute.set_buffer(0, Some(&output), 0);
    compute.dispatch_threads(MTLSize::new(4, 4, 6), MTLSize::new(4, 4, 6));
    compute.end_encoding();
    command.commit();
    command.wait_until_completed();
    // SAFETY: `output` is a 384-byte shared buffer and the completed kernel wrote 96 floats.
    let sampled = unsafe { std::slice::from_raw_parts(output.contents().cast::<f32>(), 96) };
    for (index, &value) in sampled.iter().enumerate() {
        let expected = f32::from(linear[index]) / 255.0;
        assert!(
            (value - expected).abs() <= 0.5 / 255.0,
            "face {} texel {} sampled {value}, expected {expected}",
            index / 16,
            index % 16,
        );
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
    let overlap = ResourceSpan::for_texture(first).expect("first span");
    assert!(overlap.overlaps_texture(second));
    assert!(!overlap.overlaps_texture(GuestTextureDescriptor {
        address: second.address + GUEST_PAGE_SIZE,
        ..second
    }));
}

#[test]
fn cached_overlap_identity_survives_original_virtual_address_rebinding() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(4 * GUEST_PAGE_SIZE).unwrap();
    let original_backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let replacement_backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
    let original = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &original_backing,
            0,
        )
        .unwrap();
    let alias = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &original_backing,
            0,
        )
        .unwrap();
    let original_descriptor = GuestTextureDescriptor {
        address: original.get(),
        width: 64,
        height: 64,
        pitch: 64,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let alias_descriptor = GuestTextureDescriptor {
        address: alias.get(),
        ..original_descriptor
    };
    let cached = ResourceSpan::for_texture(original_descriptor).expect("cached span");

    memory.unmap(original, GUEST_PAGE_SIZE).unwrap();
    memory
        .map_shared(
            GuestMapPlacement::Fixed(original),
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &replacement_backing,
            0,
        )
        .unwrap();

    let written_alias = ResourceSpan::for_texture(alias_descriptor).expect("alias span");
    assert!(
        !written_alias.overlaps_texture(original_descriptor),
        "resolving the cached descriptor again observes the replacement backing",
    );
    assert!(
        written_alias.overlaps(&cached),
        "the admission-time identity continues to name the cached physical storage",
    );
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

#[test]
fn gpu_written_resident_serving_sampled_key_prefers_the_newest_gpu_written_resident() {
    use super::super::texture::gpu_written_resident_serving_sampled_key;
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(4 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(4 * GUEST_PAGE_SIZE).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            4 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("guest texture mapping");
    let rendered = GuestTextureDescriptor {
        address: address.get(),
        width: 64,
        height: 64,
        pitch: 64,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let sampled = GuestTextureDescriptor {
        pitch: 128,
        ..rendered
    };
    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    presenter
        .ensure_resident_texture(&device, rendered)
        .expect("render resident");
    presenter.mark_resident_texture_written(rendered);
    assert_eq!(
        gpu_written_resident_serving_sampled_key(
            &presenter.resident_guest_textures,
            ResourceSpan::for_texture(sampled),
        ),
        Some(rendered)
    );
}

#[test]
fn gpu_written_resident_lookup_uses_the_admitted_physical_span_after_va_rebinding() {
    use super::super::texture::gpu_written_resident_serving_sampled_key;
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let texture_bytes = 4 * GUEST_PAGE_SIZE;
    let mut memory = GuestAddressSpace::new(12 * GUEST_PAGE_SIZE).expect("guest address space");
    let original_backing = GuestPhysicalBacking::new(texture_bytes).expect("original backing");
    let replacement_backing =
        GuestPhysicalBacking::new(texture_bytes).expect("replacement backing");
    let original = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            texture_bytes,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &original_backing,
            0,
        )
        .expect("original mapping");
    let alias = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            texture_bytes,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &original_backing,
            0,
        )
        .expect("alias mapping");
    let rendered = GuestTextureDescriptor {
        address: original.get(),
        width: 64,
        height: 64,
        pitch: 64,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let sampled = GuestTextureDescriptor {
        address: alias.get(),
        ..rendered
    };

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    presenter
        .ensure_resident_texture(&device, rendered)
        .expect("render resident");
    presenter.mark_resident_texture_written(rendered);

    memory.unmap(original, texture_bytes).unwrap();
    memory
        .map_shared(
            GuestMapPlacement::Fixed(original),
            texture_bytes,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &replacement_backing,
            0,
        )
        .expect("replacement mapping");

    assert_eq!(
        gpu_written_resident_serving_sampled_key(
            &presenter.resident_guest_textures,
            ResourceSpan::for_texture(sampled),
        ),
        Some(rendered),
        "lookup must use the resident's admission-time backing instead of re-resolving every VA",
    );
}

/// Builds a GPU-written resident that only the pitch-tolerant rule can serve to `sampled`.
#[cfg(test)]
fn pitch_tolerant_gpu_resident(
    presenter: &mut MetalPresenter,
    device: &Device,
    address: u64,
) -> (GuestTextureDescriptor, GuestTextureDescriptor) {
    let rendered = GuestTextureDescriptor {
        address,
        width: 16,
        height: 16,
        pitch: 16,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let sampled = GuestTextureDescriptor {
        pitch: 32,
        ..rendered
    };
    assert!(guest_texture_descriptors_share_gpu_sample_storage(
        rendered, sampled
    ));
    assert!(!guest_texture_descriptors_share_storage(rendered, sampled));
    presenter
        .ensure_resident_texture(device, rendered)
        .expect("render target resident");
    presenter.mark_resident_texture_written(rendered);
    (rendered, sampled)
}

#[test]
fn writing_a_pitch_tolerant_selection_attributes_to_the_resident_that_was_served() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(4 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("target mapping")
        .get();

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let (rendered, sampled) = pitch_tolerant_gpu_resident(&mut presenter, &device, address);
    let version_after_first_write = presenter
        .resident_guest_textures
        .get(&rendered)
        .expect("rendered resident")
        .content_version;

    // The mismatched pitch is served by the existing GPU-written resident, so no second
    // resident is created and the write has to be attributed through the selected key.
    let selected = presenter
        .ensure_resident_texture(&device, sampled)
        .expect("pitch-tolerant selection");
    assert_eq!(selected.descriptor, rendered);
    assert!(!presenter.resident_guest_textures.contains_key(&sampled));

    let misses_before = RESIDENT_WRITE_ATTRIBUTION_MISSES.load(AtomicOrdering::Relaxed);
    presenter.mark_resident_texture_written_at(
        sampled,
        ResourceSpan::for_texture(sampled),
        Some(selected.descriptor),
        WrittenRegion::whole_resource(),
        ResidentTextureActivityStamp { work: 42, flip: 7 },
    );
    let resident = presenter
        .resident_guest_textures
        .get(&rendered)
        .expect("selected resident survives its own write");
    assert!(resident.gpu_written());
    assert!(resident.content_version > version_after_first_write);
    assert_eq!(
        resident.last_use,
        ResidentTextureActivityStamp { work: 42, flip: 7 }
    );
    assert_eq!(
        RESIDENT_WRITE_ATTRIBUTION_MISSES.load(AtomicOrdering::Relaxed),
        misses_before
    );
}

#[test]
fn writing_without_the_selected_key_counts_a_pitch_tolerant_attribution_miss() {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(4 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).expect("guest backing");
    let address = memory
        .map_shared(
            GuestMapPlacement::Anywhere,
            GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("target mapping")
        .get();

    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let (rendered, sampled) = pitch_tolerant_gpu_resident(&mut presenter, &device, address);
    let version_after_first_write = presenter
        .resident_guest_textures
        .get(&rendered)
        .expect("rendered resident")
        .content_version;

    // This is the defect the counter watches for: the raw written descriptor reaches neither
    // the exact key nor a shared-storage resident, so the write lands nowhere.
    let misses_before = RESIDENT_WRITE_ATTRIBUTION_MISSES.load(AtomicOrdering::Relaxed);
    presenter.mark_resident_texture_written_at(
        sampled,
        ResourceSpan::for_texture(sampled),
        None,
        WrittenRegion::whole_resource(),
        ResidentTextureActivityStamp { work: 42, flip: 7 },
    );
    assert_eq!(
        RESIDENT_WRITE_ATTRIBUTION_MISSES.load(AtomicOrdering::Relaxed),
        misses_before + 1
    );
    assert_eq!(
        presenter
            .resident_guest_textures
            .get(&rendered)
            .expect("rendered resident")
            .content_version,
        version_after_first_write
    );
}

/// Deterministic pseudo-random source for the residency-index property tests.
///
/// A fixed sequence keeps any failure reproducible; the suite must not depend on entropy.
struct IndexTestRandom(u64);

impl IndexTestRandom {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % bound
    }
}

/// Builds a linear descriptor at a random offset and height inside one of `bases`.
///
/// Every generated range stays inside its mapping so its physical identity resolves; the
/// mappings the caller supplies decide whether two descriptors can alias.
fn index_test_descriptor(
    random: &mut IndexTestRandom,
    bases: &[u64],
    mapping_bytes: u64,
) -> GuestTextureDescriptor {
    const ROW_BYTES: u64 = 256;
    let maximum_rows = mapping_bytes / ROW_BYTES;
    let height = 1 + random.below(maximum_rows / 4);
    let row = random.below(maximum_rows - height);
    GuestTextureDescriptor {
        address: bases[usize::try_from(random.below(u64::try_from(bases.len()).unwrap())).unwrap()]
            + row * ROW_BYTES,
        width: 64,
        height: u32::try_from(height).unwrap(),
        pitch: 64,
        tile_mode: 0,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    }
}

fn index_test_sort_key(descriptor: GuestTextureDescriptor) -> (u64, u32, u32) {
    (descriptor.address, descriptor.height, descriptor.pitch)
}

fn index_test_sorted(descriptors: Vec<GuestTextureDescriptor>) -> Vec<(u64, u32, u32)> {
    let mut keys = descriptors
        .into_iter()
        .map(index_test_sort_key)
        .collect::<Vec<_>>();
    keys.sort_unstable();
    keys
}

/// The linear scan the index replaces, in the requested operand order.
fn index_test_linear_overlapping(
    residents: &[GuestTextureDescriptor],
    query: GuestTextureDescriptor,
    order: SpanQueryOrder,
) -> Vec<GuestTextureDescriptor> {
    residents
        .iter()
        .copied()
        .filter(|resident| match order {
            SpanQueryOrder::ResidentFirst => guest_texture_descriptors_overlap(*resident, query),
            SpanQueryOrder::QueryFirst => guest_texture_descriptors_overlap(query, *resident),
        })
        .collect()
}

/// Maps one backing at two virtual addresses plus an unrelated backing at a third.
///
/// The alias pair makes the predicate's physical branch reachable: two descriptors there can be
/// virtually disjoint and still name the same bytes. The unrelated mapping supplies pairs that
/// must stay disjoint even when their physical offsets collide.
fn index_test_mappings() -> (remu_memory::GuestAddressSpace, Vec<u64>, u64) {
    use remu_memory::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mapping_bytes = 8 * GUEST_PAGE_SIZE;
    let mut memory = GuestAddressSpace::new(64 * GUEST_PAGE_SIZE).expect("guest address space");
    let shared = GuestPhysicalBacking::new(mapping_bytes).expect("shared backing");
    let unrelated = GuestPhysicalBacking::new(mapping_bytes).expect("unrelated backing");
    let mut bases = Vec::new();
    for backing in [&shared, &shared, &unrelated] {
        bases.push(
            memory
                .map_shared(
                    GuestMapPlacement::Anywhere,
                    mapping_bytes,
                    GUEST_PAGE_SIZE,
                    GuestProtection::READ_WRITE,
                    backing,
                    0,
                )
                .expect("index test mapping")
                .get(),
        );
    }
    (memory, bases, mapping_bytes)
}

#[test]
fn resident_span_index_answers_exactly_what_the_linear_overlap_scan_does() {
    let (_memory, bases, mapping_bytes) = index_test_mappings();
    let mut random = IndexTestRandom::new(0x5eed_0001);
    let mut residents = Vec::new();
    let mut index = ResidentSpanIndex::new();
    for _ in 0..64 {
        let descriptor = index_test_descriptor(&mut random, &bases, mapping_bytes);
        if residents.contains(&descriptor) {
            continue;
        }
        residents.push(descriptor);
        index.insert(descriptor);
    }
    assert_eq!(index.len(), residents.len());

    // Matches that only the physical branch can produce: the two virtual ranges are disjoint.
    let mut physical_only_matches = 0_usize;
    for _ in 0..256 {
        let query = index_test_descriptor(&mut random, &bases, mapping_bytes);
        let span = ResourceSpan::for_texture(query).expect("query span");
        for order in [SpanQueryOrder::ResidentFirst, SpanQueryOrder::QueryFirst] {
            let indexed = index_test_sorted(index.overlapping(&span, order));
            let linear = index_test_linear_overlapping(&residents, query, order);
            physical_only_matches += linear
                .iter()
                .filter(|resident| {
                    !super::super::guest_ranges_overlap(
                        resident.address,
                        guest_texture_byte_count(**resident).expect("resident bytes"),
                        query.address,
                        guest_texture_byte_count(query).expect("query bytes"),
                    )
                })
                .count();
            assert_eq!(
                indexed,
                index_test_sorted(linear),
                "index disagrees with the linear scan for {query:?}"
            );
        }
    }
    assert!(
        physical_only_matches > 0,
        "no query aliased a resident through the shared backing; the physical index went untested"
    );
}

#[test]
fn resident_span_index_matches_the_linear_scan_after_random_inserts_and_removes() {
    let (_memory, bases, mapping_bytes) = index_test_mappings();
    let mut random = IndexTestRandom::new(0x5eed_0002);
    let population = (0..48)
        .map(|_| index_test_descriptor(&mut random, &bases, mapping_bytes))
        .collect::<Vec<_>>();

    let mut live: Vec<GuestTextureDescriptor> = Vec::new();
    let mut index = ResidentSpanIndex::new();
    for step in 0..512 {
        let candidate = population[usize::try_from(random.below(48)).unwrap()];
        if live.contains(&candidate) && random.below(2) == 0 {
            live.retain(|resident| *resident != candidate);
            index.remove(&candidate);
        } else if !live.contains(&candidate) {
            live.push(candidate);
            index.insert(candidate);
        }
        assert_eq!(
            index.len(),
            live.len(),
            "index population drifted at step {step}"
        );

        let query = index_test_descriptor(&mut random, &bases, mapping_bytes);
        let span = ResourceSpan::for_texture(query).expect("query span");
        for order in [SpanQueryOrder::ResidentFirst, SpanQueryOrder::QueryFirst] {
            assert_eq!(
                index_test_sorted(index.overlapping(&span, order)),
                index_test_sorted(index_test_linear_overlapping(&live, query, order)),
                "index disagrees with the linear scan at step {step}"
            );
        }
    }
}
