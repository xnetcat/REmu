// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use block::ConcreteBlock;
use metal::{
    CompileOptions, Device, MTLBlendFactor, MTLBlendOperation, MTLClearColor, MTLLoadAction,
    MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLSamplerAddressMode,
    MTLSamplerMinMagFilter, MTLSamplerMipFilter, MTLSize, MTLStorageMode, MTLStoreAction,
    MTLTextureUsage, MTLViewport, RenderPassDescriptor, RenderPipelineDescriptor,
    SamplerDescriptor, TextureDescriptor,
};
use remu_gpu::{SYSTEM_UI_HEIGHT, SYSTEM_UI_WIDTH};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use super::{
    ActiveTranslatedScanout, EVENT_POLL_INTERVAL, FlipTerminalSource, FlushPolicy,
    GuestBatchFlushReason, GuestTextureDescriptor, HostSplashTexture, MIP_UPLOAD_BYTES,
    MIP_UPLOAD_COUNT, MIP_UPLOAD_NANOSECONDS, MODE27_DECODE_BYTES, MODE27_DECODE_COUNT,
    MODE27_DECODE_NANOSECONDS, MetalPresenter, OrderedEventSource, PIPELINE_COMPILE_COMPLETED,
    PIPELINE_COMPILE_ENQUEUED, PIPELINE_COMPILE_FAILED, PIPELINE_COMPILE_QUEUE_FULL,
    PIPELINE_COMPUTE_GATES, PIPELINE_DRAW_GATES, PendingPresentation, PresentationResolution,
    PresentationResources, PresenterFlow, RESIDENT_DEPTH_SAMPLE_HITS, RESIDENT_FLIP_AGE_LABELS,
    RESIDENT_TEXTURE_ACQUIRE_INVALIDATIONS, RESIDENT_WORK_AGE_LABELS,
    RESIDENT_WRITE_ATTRIBUTION_MISSES, SAMPLED_ALIAS_WRITEBACK_BYTES,
    SAMPLED_ALIAS_WRITEBACK_COUNT, SAMPLED_ALIAS_WRITEBACK_NANOSECONDS,
    SAMPLED_TEXTURE_CACHE_ACQUIRE_INVALIDATIONS, SAMPLED_TEXTURE_CACHE_CAPACITY_EVICTIONS,
    SAMPLED_TEXTURE_CACHE_DIRTY_RELOADS, SAMPLED_TEXTURE_CACHE_HITS, SAMPLED_TEXTURE_CACHE_MISSES,
    TEXTURE_DIRTY_VALIDATION_BYTES, TEXTURE_DIRTY_VALIDATION_COUNT,
    TEXTURE_DIRTY_VALIDATION_NANOSECONDS, TEXTURE_DIRTY_VALIDATION_UNCHANGED,
    TEXTURE_SNAPSHOT_BYTES, TEXTURE_SNAPSHOT_COUNT, TEXTURE_SNAPSHOT_NANOSECONDS,
    TEXTURE_SNAPSHOT_UNSTABLE_COUNT, WriteWatchRefresh, capture_interval,
    compile_metal_library_with_tools, completed_command_buffer_gpu_duration,
    decode_guest_texture_linear_from_raw, describe_guest_batch, diagnostic_scanout_writer,
    diagnostic_texture_writer, process_memory_statistics, raw_video_out_texture_descriptor,
    read_texture_linear, refresh_resident_texture_write_watch,
    resident_texture_activity_latest_work, resident_texture_activity_stats,
    resident_texture_backing_overlap_stats, resident_texture_lifetime_diagnostics_enabled,
    resident_texture_ownership_stats, snapshot_guest_texture_raw, take_mode27_work_details,
    take_texture_dirty_work_details, take_texture_snapshot_work_details,
    trace_resident_texture_removal, trace_scanout_writes,
};
mod scanout;
mod window;
#[cfg(test)]
pub(crate) use self::window::{
    appkit_persistent_state_is_ignored, clear_appkit_persistent_state_override,
    headless_metal_value_is_enabled,
};
pub(crate) use self::window::{
    disable_appkit_state_restoration, headless_metal_requested, run_headless_presenter,
};

impl MetalPresenter {
    pub(super) fn render(&mut self) {
        if !self.redraw_pending {
            return;
        }
        self.upload_pending_frame();
        self.refresh_translated_scanout_before_present();
        let active_translated_scanout = self
            .active_translated_scanout
            .filter(|scanout| scanout.sequence == self.uploaded_flip_sequence);
        let snapshot_ready = active_translated_scanout.and_then(|scanout| scanout.snapshot_ready);
        let translated_snapshot = active_translated_scanout.and_then(|scanout| {
            self.scanout_snapshot_textures
                .get(&scanout.address)
                .filter(|snapshot| {
                    snapshot.descriptor.width == scanout.width
                        && snapshot.descriptor.height == scanout.height
                })
                .map(|snapshot| (snapshot.descriptor, snapshot.texture.clone()))
        });
        let guest_frame_texture = self.frame_texture.clone();
        let host_splash = self
            .host_splash
            .as_ref()
            .map(|splash| (splash.texture.clone(), splash.width, splash.height));
        let presentation_source = if host_splash.is_some() {
            "host-splash"
        } else if translated_snapshot.is_some() {
            "translated"
        } else if guest_frame_texture.is_some() {
            "raw"
        } else {
            "clear"
        };
        let (frame_texture, source_dimensions) = if let Some((texture, width, height)) = host_splash
        {
            (Some(texture), Some((width, height)))
        } else {
            (
                guest_frame_texture,
                self.display.map(|display| (display.width, display.height)),
            )
        };
        let snapshot_wait = (presentation_source == "translated")
            .then_some(snapshot_ready)
            .flatten()
            .map(|fence| {
                let event = self
                    .guest_queue_batches
                    .get(&fence.producer().queue_id())
                    .expect("snapshot fence producer queue remains live")
                    .shared_event_timeline
                    .event_owned();
                (fence, event)
            });
        // Raw VideoOut fallback can be captured immediately before the translated render target
        // becomes available. Do not let that fallback consume the capture interval: the first
        // translated snapshot is the authoritative evidence for what Metal is actually about to
        // present.
        let first_translated_capture = self.periodic_frame_capture
            && translated_snapshot.as_ref().is_some_and(|(descriptor, _)| {
                self.captured_guest_frames.len() < 128
                    && !self.captured_guest_frames.iter().any(
                        |(address, width, height, tile_mode, _)| {
                            *address == descriptor.address
                                && *width == descriptor.width
                                && *height == descriptor.height
                                && *tile_mode == descriptor.tile_mode
                        },
                    )
            });
        if first_translated_capture {
            self.last_guest_capture = None;
        }
        let capture_due = self.periodic_frame_capture
            && self.capture_directory.is_some()
            && (first_translated_capture
                || self
                    .last_guest_capture
                    .is_none_or(|last| last.elapsed() >= capture_interval()));
        if capture_due
            && let Some((descriptor, texture)) = translated_snapshot.as_ref()
            && matches!(
                descriptor.pixel_format,
                MTLPixelFormat::RGBA8Unorm
                    | MTLPixelFormat::RGBA8Unorm_sRGB
                    | MTLPixelFormat::BGRA8Unorm
                    | MTLPixelFormat::BGRA8Unorm_sRGB
                    | MTLPixelFormat::RGBA16Float
                    | MTLPixelFormat::RGB10A2Unorm
            )
            && let (Some(device), Some(queue)) = (
                self.device.as_ref().cloned(),
                self.command_queue.as_ref().cloned(),
            )
        {
            match read_texture_linear(&device, &queue, texture, *descriptor) {
                Ok(rendered) => self.capture_translated_frame_if_changed(*descriptor, &rendered),
                Err(error) => {
                    eprintln!("[REMu][METAL] could not capture live translated frame: {error}");
                }
            }
        }
        // A host splash is presentable before VideoOut registration. After HideSplash, a missing
        // guest texture intentionally clears the old system-owned image to black until the first
        // guest scanout is ready.
        let encode_srgb = frame_texture
            .as_ref()
            .is_none_or(|texture| presentation_applies_srgb_encode(texture.pixel_format()));
        let drawable_format = if encode_srgb {
            MTLPixelFormat::BGRA8Unorm_sRGB
        } else {
            MTLPixelFormat::BGRA8Unorm
        };
        let (Some(layer), Some(queue)) = (self.layer.as_mut(), self.command_queue.as_ref()) else {
            return;
        };
        if layer.pixel_format() != drawable_format {
            layer.set_pixel_format(drawable_format);
        }
        let Some(drawable) = layer.next_drawable() else {
            if trace_scanout_writes() {
                eprintln!(
                    "[REMu][METAL][SCANOUT-PRESENT] outcome=no-drawable source={presentation_source} pending={} finalized={} uploaded={} submitted={}",
                    self.pending_flip.map_or(0, |flip| flip.sequence),
                    self.finalized_flip_sequence,
                    self.uploaded_flip_sequence,
                    self.submitted_flip_sequence,
                );
            }
            return;
        };
        let command_buffer = queue.new_command_buffer().to_owned();
        command_buffer.set_label("REMu guest presentation");
        if let Some((fence, event)) = snapshot_wait.as_ref() {
            command_buffer.encode_wait_for_event(event, fence.value());
        }
        let Ok(destination_width) = u32::try_from(drawable.texture().width()) else {
            return;
        };
        let Ok(destination_height) = u32::try_from(drawable.texture().height()) else {
            return;
        };
        let pass = RenderPassDescriptor::new();
        let attachment = pass
            .color_attachments()
            .object_at(0)
            .expect("Metal presentation pass has color attachment zero");
        attachment.set_texture(Some(drawable.texture()));
        attachment.set_load_action(MTLLoadAction::Clear);
        attachment.set_store_action(MTLStoreAction::Store);
        attachment.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));
        let encoder = command_buffer.new_render_command_encoder(pass);
        encoder.set_label("REMu scaled display-buffer presentation");
        if let Some(texture) = frame_texture.as_ref() {
            let presentation = self
                .presentation
                .as_ref()
                .expect("windowed presenter has presentation resources");
            let (source_width, source_height) = source_dimensions
                .expect("every presentation texture has validated source dimensions");
            let pipeline = if encode_srgb {
                &presentation.srgb_pipeline
            } else {
                &presentation.unorm_pipeline
            };
            encoder.set_render_pipeline_state(pipeline);
            encoder.set_viewport(presentation_viewport(
                source_width,
                source_height,
                destination_width,
                destination_height,
            ));
            encoder.set_fragment_texture(0, Some(texture));
            encoder.set_fragment_sampler_state(0, Some(&presentation.sampler));
            encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
        }
        if crate::presenter::host_ui::system_ui_overlay_visible() {
            let presentation = self
                .presentation
                .as_mut()
                .expect("windowed presenter has presentation resources");
            if let Some(overlay) = crate::presenter::host_ui::take_system_ui_overlay_upload() {
                upload_system_ui_overlay(&presentation.overlay_texture, &overlay.rgba);
            }
            let overlay_pipeline = if encode_srgb {
                &presentation.overlay_srgb_pipeline
            } else {
                &presentation.overlay_unorm_pipeline
            };
            encoder.set_render_pipeline_state(overlay_pipeline);
            encoder.set_viewport(MTLViewport {
                originX: 0.0,
                originY: 0.0,
                width: f64::from(destination_width),
                height: f64::from(destination_height),
                znear: 0.0,
                zfar: 1.0,
            });
            encoder.set_fragment_texture(0, Some(&presentation.overlay_texture));
            encoder.set_fragment_sampler_state(0, Some(&presentation.sampler));
            encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
        }
        encoder.end_encoding();
        let drawable_presented = Arc::new(AtomicBool::new(false));
        let presented_latch = Arc::clone(&drawable_presented);
        let presented_handler = ConcreteBlock::new(move |_: &metal::DrawableRef| {
            presented_latch.store(true, Ordering::Release);
        })
        .copy();
        drawable.add_presented_handler(&presented_handler);
        command_buffer.present_drawable(drawable);
        command_buffer.commit();
        if trace_scanout_writes() {
            eprintln!(
                "[REMu][METAL][SCANOUT-PRESENT] outcome=committed source={presentation_source} source_size={source_dimensions:?} source_format={:?} drawable={}x{} pending={} finalized={} uploaded={} submitted_before={}",
                frame_texture.as_ref().map(|texture| texture.pixel_format()),
                destination_width,
                destination_height,
                self.pending_flip.map_or(0, |flip| flip.sequence),
                self.finalized_flip_sequence,
                self.uploaded_flip_sequence,
                self.submitted_flip_sequence,
            );
        }
        // Presentation is where a finalized flip becomes guest-visible; an unfinalized flip has no
        // snapshot in this drawable and stays pending.
        let flip_sequence = self.claim_flip_terminal(self.finalized_flip_sequence);
        if flip_sequence.is_some() {
            remu_gpu::record_startup_milestone(
                remu_gpu::StartupMilestone::FirstGuestPresentSubmitted,
            );
        }
        self.pending_presentations.push_back(PendingPresentation {
            command_buffer,
            submitted_at: Instant::now(),
            flip_sequence,
            flip_terminal_source: FlipTerminalSource::Presentation,
            drawable_presented: Some(drawable_presented),
            await_presented_reported: false,
        });
        self.redraw_pending = false;
    }
}
pub(super) const PRESENTATION_SHADER_SOURCE: &str = r"
#include <metal_stdlib>
using namespace metal;

struct REmuPresentationVertex {
    float4 position [[position]];
    float2 texture_coordinate;
};

vertex REmuPresentationVertex remu_presentation_vertex(uint vertex_id [[vertex_id]]) {
    constexpr float2 positions[] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0),
    };
    constexpr float2 texture_coordinates[] = {
        float2(0.0,  1.0),
        float2(2.0,  1.0),
        float2(0.0, -1.0),
    };
    return {
        float4(positions[vertex_id], 0.0, 1.0),
        texture_coordinates[vertex_id],
    };
}

fragment float4 remu_presentation_fragment(
    REmuPresentationVertex input [[stage_in]],
    texture2d<float> source [[texture(0)]],
    sampler source_sampler [[sampler(0)]]) {
    // VideoOut treats scanout alpha as padding; games are free to leave it at zero. Passing that
    // value into CAMetalLayer makes AppKit composite an otherwise valid frame as transparent
    // black, so host presentation must always be opaque.
    return float4(source.sample(source_sampler, input.texture_coordinate).rgb, 1.0);
}

fragment float4 remu_system_ui_fragment(
    REmuPresentationVertex input [[stage_in]],
    texture2d<float> source [[texture(0)]],
    sampler source_sampler [[sampler(0)]]) {
    // Same UV space as guest present; overlay pixels are uploaded top-left like VideoOut.
    return source.sample(source_sampler, input.texture_coordinate);
}
";

impl PresentationResolution {
    pub(super) fn from_environment() -> Result<Self, String> {
        let Some(value) = std::env::var_os("REMU_PRESENTATION_SIZE") else {
            return Ok(Self::LogicalWindow);
        };
        let value = value
            .into_string()
            .map_err(|_| "REMU_PRESENTATION_SIZE is not valid UTF-8".to_owned())?;
        parse_presentation_resolution(&value)
    }
}

/// Most guest scanouts are treated as linear and need an sRGB drawable (Unity RGBA16F and the
/// common RGBA8 path). Packed `RGB10A2` VideoOut — Void Terrarium's flip format — is already
/// display-referred; writing it into `BGRA8Unorm_sRGB` applies a second linear→sRGB encode and
/// lifts near-blacks into the washed title look.
pub(super) fn presentation_applies_srgb_encode(source_format: MTLPixelFormat) -> bool {
    !matches!(source_format, MTLPixelFormat::RGB10A2Unorm)
}

fn create_presentation_pipeline(
    device: &Device,
    vertex_function: &metal::Function,
    fragment_function: &metal::Function,
    pixel_format: MTLPixelFormat,
    label: &str,
) -> Result<metal::RenderPipelineState, String> {
    let descriptor = RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_function));
    descriptor.set_fragment_function(Some(fragment_function));
    descriptor
        .color_attachments()
        .object_at(0)
        .ok_or("Metal has no presentation color attachment")?
        .set_pixel_format(pixel_format);
    device
        .new_render_pipeline_state(&descriptor)
        .map_err(|error| format!("Metal presentation pipeline ({label}): {error}"))
}

fn create_overlay_pipeline(
    device: &Device,
    vertex_function: &metal::Function,
    fragment_function: &metal::Function,
    pixel_format: MTLPixelFormat,
    label: &str,
) -> Result<metal::RenderPipelineState, String> {
    let descriptor = RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_function));
    descriptor.set_fragment_function(Some(fragment_function));
    let attachment = descriptor
        .color_attachments()
        .object_at(0)
        .ok_or("Metal has no overlay color attachment")?;
    attachment.set_pixel_format(pixel_format);
    attachment.set_blending_enabled(true);
    attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
    attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
    attachment.set_rgb_blend_operation(MTLBlendOperation::Add);
    attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
    attachment.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
    attachment.set_alpha_blend_operation(MTLBlendOperation::Add);
    device
        .new_render_pipeline_state(&descriptor)
        .map_err(|error| format!("Metal overlay pipeline ({label}): {error}"))
}

fn create_system_ui_overlay_texture(device: &Device) -> metal::Texture {
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(metal::MTLTextureType::D2);
    descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    descriptor.set_width(u64::from(SYSTEM_UI_WIDTH));
    descriptor.set_height(u64::from(SYSTEM_UI_HEIGHT));
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(MTLTextureUsage::ShaderRead);
    device.new_texture(&descriptor)
}

fn upload_system_ui_overlay(texture: &metal::Texture, rgba: &[u8]) {
    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: u64::from(SYSTEM_UI_WIDTH),
            height: u64::from(SYSTEM_UI_HEIGHT),
            depth: 1,
        },
    };
    texture.replace_region(
        region,
        0,
        rgba.as_ptr().cast(),
        u64::from(SYSTEM_UI_WIDTH) * 4,
    );
}

pub(super) fn create_presentation_resources(
    device: &Device,
) -> Result<PresentationResources, String> {
    let library = if super::shader::offline_metal_compiler().is_some() {
        compile_metal_library_with_tools(device, "presentation", PRESENTATION_SHADER_SOURCE)?
    } else {
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        device
            .new_library_with_source(PRESENTATION_SHADER_SOURCE, &options)
            .map_err(|error| format!("Metal presentation shader compilation: {error}"))?
    };
    let vertex_function = library
        .get_function("remu_presentation_vertex", None)
        .map_err(|error| format!("Metal presentation vertex function: {error}"))?;
    let fragment_function = library
        .get_function("remu_presentation_fragment", None)
        .map_err(|error| format!("Metal presentation fragment function: {error}"))?;
    let overlay_fragment_function = library
        .get_function("remu_system_ui_fragment", None)
        .map_err(|error| format!("Metal system UI fragment function: {error}"))?;
    let srgb_pipeline = create_presentation_pipeline(
        device,
        &vertex_function,
        &fragment_function,
        MTLPixelFormat::BGRA8Unorm_sRGB,
        "REMu scaled display presentation (sRGB)",
    )?;
    let unorm_pipeline = create_presentation_pipeline(
        device,
        &vertex_function,
        &fragment_function,
        MTLPixelFormat::BGRA8Unorm,
        "REMu scaled display presentation (Unorm)",
    )?;
    let overlay_srgb_pipeline = create_overlay_pipeline(
        device,
        &vertex_function,
        &overlay_fragment_function,
        MTLPixelFormat::BGRA8Unorm_sRGB,
        "REMu system UI overlay (sRGB)",
    )?;
    let overlay_unorm_pipeline = create_overlay_pipeline(
        device,
        &vertex_function,
        &overlay_fragment_function,
        MTLPixelFormat::BGRA8Unorm,
        "REMu system UI overlay (Unorm)",
    )?;
    let sampler_descriptor = SamplerDescriptor::new();
    sampler_descriptor.set_address_mode_s(MTLSamplerAddressMode::ClampToEdge);
    sampler_descriptor.set_address_mode_t(MTLSamplerAddressMode::ClampToEdge);
    sampler_descriptor.set_min_filter(MTLSamplerMinMagFilter::Linear);
    sampler_descriptor.set_mag_filter(MTLSamplerMinMagFilter::Linear);
    sampler_descriptor.set_mip_filter(MTLSamplerMipFilter::NotMipmapped);
    let sampler = device.new_sampler(&sampler_descriptor);
    let overlay_texture = create_system_ui_overlay_texture(device);
    Ok(PresentationResources {
        srgb_pipeline,
        unorm_pipeline,
        overlay_srgb_pipeline,
        overlay_unorm_pipeline,
        overlay_texture,
        _vertex_function: vertex_function,
        _fragment_function: fragment_function,
        _overlay_fragment_function: overlay_fragment_function,
        sampler,
    })
}

pub(super) fn parse_presentation_resolution(value: &str) -> Result<PresentationResolution, String> {
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "window" | "logical" => Ok(PresentationResolution::LogicalWindow),
        "native" => Ok(PresentationResolution::NativeWindow),
        _ => {
            let (width, height) = normalized.split_once('x').ok_or_else(|| {
                format!(
                    "invalid REMU_PRESENTATION_SIZE {value:?}; expected window, native, or WIDTHxHEIGHT"
                )
            })?;
            let width = width
                .parse::<u32>()
                .map_err(|_| format!("invalid width in REMU_PRESENTATION_SIZE {value:?}"))?;
            let height = height
                .parse::<u32>()
                .map_err(|_| format!("invalid height in REMU_PRESENTATION_SIZE {value:?}"))?;
            if width == 0 || height == 0 || width > 16_384 || height > 16_384 {
                return Err(format!(
                    "REMU_PRESENTATION_SIZE dimensions must each be in 1..=16384, got {width}x{height}"
                ));
            }
            Ok(PresentationResolution::Fixed(width, height))
        }
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(super) fn presentation_drawable_size(
    resolution: PresentationResolution,
    physical_width: u32,
    physical_height: u32,
    scale_factor: f64,
) -> (u32, u32) {
    match resolution {
        PresentationResolution::Fixed(width, height) => (width, height),
        PresentationResolution::NativeWindow => (physical_width.max(1), physical_height.max(1)),
        PresentationResolution::LogicalWindow => {
            let scale_factor = if scale_factor.is_finite() && scale_factor > 0.0 {
                scale_factor
            } else {
                1.0
            };
            let logical = |physical: u32| {
                // The result is rounded and clamped to the complete accepted u32 subrange before
                // conversion, so neither truncation nor a negative result is possible.
                (f64::from(physical) / scale_factor)
                    .round()
                    .clamp(1.0, 16_384.0) as u32
            };
            (logical(physical_width), logical(physical_height))
        }
    }
}

pub(super) fn presentation_viewport(
    source_width: u32,
    source_height: u32,
    destination_width: u32,
    destination_height: u32,
) -> MTLViewport {
    let source_width = f64::from(source_width.max(1));
    let source_height = f64::from(source_height.max(1));
    let destination_width = f64::from(destination_width.max(1));
    let destination_height = f64::from(destination_height.max(1));
    let scale = (destination_width / source_width).min(destination_height / source_height);
    let width = source_width * scale;
    let height = source_height * scale;
    MTLViewport {
        originX: (destination_width - width) * 0.5,
        originY: (destination_height - height) * 0.5,
        width,
        height,
        znear: 0.0,
        zfar: 1.0,
    }
}
