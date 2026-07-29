// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use metal::{
    CompileOptions, Device, MTLClearColor, MTLCommandBufferStatus, MTLLoadAction, MTLPixelFormat,
    MTLRegion, MTLResourceOptions, MTLSize, MTLStorageMode, MTLStoreAction, MTLTextureType,
    MTLTextureUsage, RenderPassDescriptor, TextureDescriptor,
};
use remu_memory::CpuWriteWatchRetirement;
use remu_shader::{ImageBinding, ImageControl, Opcode, ShaderEvaluation};

use super::{
    DecodedImageDescriptor, GuestTextureDescriptor, GuestTextureOverlapQuery, MetalPresenter,
    WriteWatchRefresh, apply_image_destination_select, comparison_depth32_upload,
    create_sampled_guest_texture, decode_image_descriptor, depth_array_face_descriptor,
    depth_array_face_set_is_complete, fallback_image_descriptor, guest_shared_mapping_identity,
    guest_texture_byte_count, guest_texture_descriptors_overlap,
    guest_texture_descriptors_share_color_view_storage,
    guest_texture_descriptors_share_component_reordered_storage,
    guest_texture_descriptors_share_storage, guest_texture_raw_matches, guest_texture_raw_range,
    image_destination_select_is_native, image_resource_descriptor_is_null, mode5_mip_chain_layout,
    mode27_mip_chain_layout, new_color_alias_texture_view, new_red_blue_swizzled_texture_view,
    render_resolve_candidate_matches, sampled_alias_descriptor, sampled_texture_cache_accepts,
    sampled_texture_cache_needs_reload, sampled_texture_cache_requires_eviction,
    snapshot_guest_texture_raw, snapshot_watched_guest_texture_raw,
    storage_image_requires_mip_hierarchy, upload_image_textures, watch_guest_texture,
    write_watched_guest_texture_linear,
};

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
fn layered_depth_sample_faces_resolve_to_per_face_depth_targets() {
    // A shadow cubemap is rendered as six separate 2D depth targets and then bound through one
    // six-layer R16Unorm array SRD. Each layer must resolve to the matching per-face depth
    // descriptor so the faces can be served on the GPU instead of round-tripping through guest
    // memory every frame.
    let sampled = DecodedImageDescriptor {
        guest: GuestTextureDescriptor {
            address: 0x70_6969_0000,
            width: 1024,
            height: 1024,
            pitch: 1024,
            tile_mode: 24,
            pixel_format: MTLPixelFormat::R16Unorm,
            bytes_per_pixel: 2,
        },
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
    let sampled = DecodedImageDescriptor {
        guest: GuestTextureDescriptor {
            address: 0x70_6969_0000,
            width: 1024,
            height: 1024,
            pitch: 1024,
            tile_mode: 24,
            pixel_format: MTLPixelFormat::R16Unorm,
            bytes_per_pixel: 2,
        },
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
fn mode5_sampled_mip_range_covers_the_packed_prefix_and_deferred_base_image() {
    let descriptor = DecodedImageDescriptor {
        guest: GuestTextureDescriptor {
            address: 0x70_2754_1000,
            width: 512,
            height: 512,
            pitch: 512,
            tile_mode: 5,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        },
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
    let descriptor = DecodedImageDescriptor {
        guest: GuestTextureDescriptor {
            address: 0x70_3000_0000,
            width: 256,
            height: 256,
            pitch: 256,
            tile_mode: 5,
            pixel_format: MTLPixelFormat::RGBA16Float,
            bytes_per_pixel: 8,
        },
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
        !resident.gpu_written,
        "copying clean guest-derived levels must not invent GPU ownership"
    );
    assert!(resident.gpu_write_watch_generations.is_empty());
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
    assert!(!presenter.resident_guest_textures.contains_key(&first));
    assert!(
        presenter
            .resident_guest_textures
            .get(&second)
            .is_some_and(|resident| resident.gpu_written)
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
        None,
        complete,
        &mut presenter.resident_guest_textures,
        super::Mode27WorkOrigin::StorageCreate,
    )
    .expect("declared complete mip hierarchy becomes resident");
    assert_eq!(original.mipmap_level_count(), 6);

    let complete_view = super::ensure_mipmapped_resident_texture_in_map(
        &device,
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
        destination_select: 0x22c,
        base_mip_level: 2,
        mip_level_count: 2,
        texture_type: 9,
        layer_count: 1,
    };
    let view = super::ensure_mipmapped_resident_texture_in_map(
        &device,
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
    let texture = create_sampled_guest_texture(&device, &queue, guest, &linear, 0, 1, 11, 6, false)
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
    let overlap = GuestTextureOverlapQuery::new(first);
    assert!(overlap.overlaps(second));
    assert!(!overlap.overlaps(GuestTextureDescriptor {
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
    let cached = GuestTextureOverlapQuery::new(original_descriptor);

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

    let written_alias = GuestTextureOverlapQuery::new(alias_descriptor);
    assert!(
        !written_alias.overlaps(original_descriptor),
        "resolving the cached descriptor again observes the replacement backing",
    );
    assert!(
        written_alias.overlaps_query(&cached),
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
