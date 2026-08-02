// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use metal::{
    MTLClearColor, MTLLoadAction, MTLPixelFormat, MTLStorageMode, MTLStoreAction, MTLTextureType,
    MTLTextureUsage, RenderPassDescriptor, TextureDescriptor,
};
use remu_gpu::{
    GpuColorTarget, GpuEvent, GpuFlipCompletion, GpuHostImage, GpuScanoutTarget, gpu_event_channel,
};
use remu_shader::PixelOutputKind;
use std::time::{Duration, Instant};

use super::{
    DecodedRenderTarget, DisplayConfiguration, GuestTextureDescriptor, MetalPresenter,
    OrderedEventPoll, OrderedEventSource, PendingFlip, PresentationResolution, QueueBlockReason,
    VIDEO_OUT_DISPLAY_TILE_MODE, VIDEO_OUT_TILING_MODE_LINEAR, VIDEO_OUT_TILING_MODE_TILE,
    appkit_persistent_state_is_ignored, clear_appkit_persistent_state_override,
    decode_scanout_target, disable_appkit_state_restoration, headless_metal_value_is_enabled,
    parse_presentation_resolution, presentation_drawable_size, presentation_viewport,
    process_memory_statistics, process_resident_bytes, raw_video_out_layout_is_linear,
    raw_video_out_texture_descriptor, read_texture_linear,
};

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
fn windowed_presenter_disables_appkit_persistent_state() {
    clear_appkit_persistent_state_override();
    assert!(!appkit_persistent_state_is_ignored());

    disable_appkit_state_restoration();
    assert!(appkit_persistent_state_is_ignored());

    clear_appkit_persistent_state_override();
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
    let statistics = process_memory_statistics().expect("live process memory statistics");
    assert!(statistics.physical_footprint_bytes > 0);
    assert!(
        statistics.peak_physical_footprint_bytes >= statistics.physical_footprint_bytes,
        "lifetime peak cannot be lower than the current physical footprint"
    );
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
fn a_write_to_the_other_registered_buffer_waits_for_its_own_flip() {
    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let device = presenter.device.as_ref().expect("Metal device").clone();
    let front = GuestTextureDescriptor {
        address: 0xefc0_0000_00,
        width: 640,
        height: 360,
        pitch: 640,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::BGRA8Unorm,
        bytes_per_pixel: 4,
    };
    let back = GuestTextureDescriptor {
        address: 0xefc2_0000_00,
        ..front
    };
    presenter.registered_video_out_buffers = vec![front.address, back.address];
    presenter.display = Some(DisplayConfiguration {
        width: front.width,
        height: front.height,
        pitch: front.pitch,
        pixel_format: 0x8000_0000_0000_0000,
        tiling_mode: 27,
    });
    presenter.pending_flip = Some(PendingFlip {
        address: front.address,
        buffer_index: 0,
        flip_argument: 0,
        sequence: 1,
    });
    presenter.uploaded_flip_sequence = 1;
    presenter.next_flip_sequence = 1;
    presenter.ensure_direct_scanout_texture(&device, back);
    presenter.mark_direct_scanout_initialized(back.address);

    // The guest composites the next frame into the buffer it has not flipped yet.
    presenter.note_pending_flip_target_write(back);
    assert!(presenter.pending_flip_target_written_since_boundary);
    assert_eq!(presenter.pending_scanout_write_address, Some(back.address));

    presenter.redraw_pending = false;
    presenter.finalize_pending_flip_at_submission_boundary(OrderedEventSource::GuestQueue(7));
    assert!(
        !presenter.redraw_pending,
        "a write to the buffer the guest has not flipped must not present a second frame"
    );
    assert!(presenter.active_translated_scanout.is_none());
    assert_eq!(presenter.uploaded_flip_sequence, 1);
    assert_eq!(
        presenter.pending_scanout_write_address,
        Some(back.address),
        "the recorded write stays available to the flip that claims it"
    );

    // The flip for that buffer is what publishes it.
    presenter.pending_flip = Some(PendingFlip {
        address: back.address,
        buffer_index: 1,
        flip_argument: 0,
        sequence: 2,
    });
    presenter.next_flip_sequence = 2;
    presenter.pending_scanout_write_address = None;
    presenter.finalize_pending_flip();
    assert!(presenter.redraw_pending);
    assert_eq!(presenter.uploaded_flip_sequence, 2);
    assert_eq!(
        presenter
            .active_translated_scanout
            .map(|scanout| scanout.address),
        Some(back.address)
    );
}

#[test]
fn a_flip_without_a_presentable_surface_still_completes() {
    let (_sender, receiver) = std::sync::mpsc::channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let _ = remu_gpu::take_gpu_completed_flips();
    presenter.display = Some(DisplayConfiguration {
        width: 640,
        height: 360,
        pitch: 640,
        pixel_format: 0x8000_0000_0000_0000,
        tiling_mode: 0,
    });
    presenter.pending_flip = Some(PendingFlip {
        address: 0xefc0_0000_00,
        buffer_index: 1,
        flip_argument: 0x42,
        sequence: 1,
    });
    presenter.next_flip_sequence = 1;
    assert!(
        presenter.layer.is_none(),
        "this presenter stands in for a headless or occluded run"
    );

    presenter.finalize_pending_flip();
    assert_eq!(presenter.submitted_flip_sequence, 1);
    assert_eq!(
        remu_gpu::take_gpu_completed_flips(),
        [GpuFlipCompletion {
            address: 0xefc0_0000_00,
            buffer_index: 1,
            flip_argument: 0x42,
        }]
    );

    presenter.finalize_pending_flip();
    assert!(
        remu_gpu::take_gpu_completed_flips().is_empty(),
        "a flip is retired exactly once"
    );
}

#[test]
fn a_superseded_flip_is_completed_when_the_next_flip_replaces_it() {
    let (sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    let _ = remu_gpu::take_gpu_completed_flips();
    // A flip the presenter accepted but could never present: no drawable was available for it.
    presenter.display = Some(DisplayConfiguration {
        width: 640,
        height: 360,
        pitch: 640,
        pixel_format: 0x8000_0000_0000_0000,
        tiling_mode: 0,
    });
    presenter.pending_flip = Some(PendingFlip {
        address: 0xefc0_0000_00,
        buffer_index: 0,
        flip_argument: 0x11,
        sequence: 1,
    });
    presenter.next_flip_sequence = 1;
    presenter.finalized_flip_sequence = 1;
    presenter.uploaded_flip_sequence = 1;

    sender
        .send(GpuEvent::Flip {
            address: 0xefc2_0000_00,
            buffer_index: 1,
            flip_argument: 0x22,
            width: 640,
            height: 360,
            pitch: 640,
            pixel_format: 0x8000_0000_0000_0000,
            tiling_mode: 0,
        })
        .expect("flip event reaches presenter");
    assert_eq!(presenter.receive_events(), super::PresenterFlow::Continue);

    assert_eq!(presenter.submitted_flip_sequence, 2);
    assert_eq!(
        remu_gpu::take_gpu_completed_flips(),
        [
            GpuFlipCompletion {
                address: 0xefc0_0000_00,
                buffer_index: 0,
                flip_argument: 0x11,
            },
            GpuFlipCompletion {
                address: 0xefc2_0000_00,
                buffer_index: 1,
                flip_argument: 0x22,
            }
        ],
        "the replaced flip is retired before the flip that replaced it"
    );
}

#[test]
fn leaked_flip_pacing_is_reconciled_against_the_presenter_sequences() {
    let (sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
    remu_gpu::install_event_sink(sender);
    // A flip counted against guest pacing which this presenter never sequences is the permanent
    // backpressure reconciliation exists to clear.
    assert!(remu_gpu::emit_event(GpuEvent::Flip {
        address: 0xefc0_0000_00,
        buffer_index: 0,
        flip_argument: 0,
        width: 640,
        height: 360,
        pitch: 640,
        pixel_format: 0x8000_0000_0000_0000,
        tiling_mode: 0,
    }));
    remu_gpu::record_gpu_work_item_processed();
    assert_eq!(remu_gpu::gpu_flips_pending(), 1);

    presenter.reconcile_pending_flips();
    assert_eq!(
        remu_gpu::gpu_flips_pending(),
        1,
        "reconciliation runs at most once per second"
    );

    presenter.statistics.last_flip_reconciliation = Instant::now()
        .checked_sub(Duration::from_secs(2))
        .expect("the monotonic clock predates this test");
    presenter.reconcile_pending_flips();
    assert_eq!(remu_gpu::gpu_flips_pending(), 0);
    remu_gpu::clear_event_sink();
}

#[test]
fn a_cold_pipeline_cover_holds_its_frame_back_from_a_foreign_submission_boundary() {
    // Octopath's compositor clears the VideoOut allocation and then covers it with a fullscreen
    // NGG draw. When that cover parks on a cold pipeline, the compute queue's boundary must not
    // publish the bare clear: the cover's own output would then look like the next frame.
    let (sender, receiver) = gpu_event_channel();
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
    presenter.mark_direct_scanout_initialized(descriptor.address);

    let mut cover = super::indexed_draw(0x4000, 4, 3);
    cover.color_target_address = descriptor.address;
    sender
        .send(GpuEvent::QueueSubmission {
            queue_id: 7,
            events: vec![GpuEvent::IndexedDraw(Box::new(cover.clone()))],
        })
        .expect("queue the cover draw");
    let source = match presenter.receiver.try_recv(|_| true) {
        OrderedEventPoll::Event {
            source,
            event: GpuEvent::IndexedDraw(_),
        } => source,
        other => panic!("expected the cover draw, got {other:?}"),
    };
    presenter.receiver.block_event(
        source,
        GpuEvent::IndexedDraw(Box::new(cover)),
        QueueBlockReason::GraphicsPipeline(super::cold_graphics_pipeline_key()),
    );
    assert!(
        presenter
            .receiver
            .pipeline_blocked_draw_writes(descriptor.address)
    );

    // The clear pass already wrote the allocation before the cover was gated.
    presenter.note_pending_flip_target_write(descriptor);
    assert!(presenter.pending_flip_target_written_since_boundary);
    presenter.redraw_pending = false;
    presenter.finalize_pending_flip_at_submission_boundary(OrderedEventSource::GuestQueue(33));
    assert!(
        !presenter.redraw_pending,
        "a foreign boundary must not publish a frame whose cover is still parked"
    );
    assert_eq!(presenter.uploaded_flip_sequence, 0);
    assert!(
        presenter.pending_flip_target_written_since_boundary,
        "the deferred boundary must keep the pending write so the frame is not lost"
    );

    // The cover's own queue still finalizes the frame once it drains.
    presenter.finalize_pending_flip_at_submission_boundary(OrderedEventSource::GuestQueue(7));
    assert!(presenter.redraw_pending);
    assert_eq!(presenter.uploaded_flip_sequence, 1);
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
        .expect("resident scanout")
        .texture;
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
            .activate_translated_scanout(descriptor.address, descriptor.width, descriptor.height)
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
    encoder.set_render_pipeline_state(&presentation.srgb_pipeline);
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
