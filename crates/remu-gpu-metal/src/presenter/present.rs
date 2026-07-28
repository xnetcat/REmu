// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use metal::{
    CompileOptions, Device, MTLClearColor, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType,
    MTLSamplerAddressMode, MTLSamplerMinMagFilter, MTLSamplerMipFilter, MTLStoreAction,
    MTLViewport, RenderPassDescriptor, RenderPipelineDescriptor, SamplerDescriptor,
};
use std::time::Instant;

use super::{
    ActiveTranslatedScanout, EVENT_POLL_INTERVAL, GuestBatchFlushReason, GuestTextureDescriptor,
    HostSplashTexture, MIP_UPLOAD_BYTES, MIP_UPLOAD_COUNT, MIP_UPLOAD_NANOSECONDS,
    MODE27_DECODE_BYTES, MODE27_DECODE_COUNT, MODE27_DECODE_NANOSECONDS, MetalPresenter,
    OrderedEventSource, PIPELINE_COMPILE_COMPLETED, PIPELINE_COMPILE_ENQUEUED,
    PIPELINE_COMPILE_FAILED, PIPELINE_COMPILE_QUEUE_FULL, PIPELINE_DRAW_GATES, PendingPresentation,
    PresentationResolution, PresentationResources, PresenterFlow, RESIDENT_DEPTH_SAMPLE_HITS,
    RESIDENT_TEXTURE_ACQUIRE_INVALIDATIONS, SAMPLED_ALIAS_WRITEBACK_BYTES,
    SAMPLED_ALIAS_WRITEBACK_COUNT, SAMPLED_ALIAS_WRITEBACK_NANOSECONDS,
    SAMPLED_TEXTURE_CACHE_ACQUIRE_INVALIDATIONS, SAMPLED_TEXTURE_CACHE_CAPACITY_EVICTIONS,
    SAMPLED_TEXTURE_CACHE_DIRTY_RELOADS, SAMPLED_TEXTURE_CACHE_HITS, SAMPLED_TEXTURE_CACHE_MISSES,
    TEXTURE_SNAPSHOT_BYTES, TEXTURE_SNAPSHOT_COUNT, TEXTURE_SNAPSHOT_NANOSECONDS,
    WriteWatchRefresh, capture_interval, compile_metal_library_with_tools,
    completed_command_buffer_gpu_duration, decode_guest_texture_linear_from_raw,
    diagnostic_scanout_writer, diagnostic_texture_writer, process_resident_bytes,
    raw_video_out_texture_descriptor, read_texture_linear, refresh_resident_texture_write_watch,
    retire_multisample_textures_overlapping_writeback,
    retire_resident_textures_overlapping_writeback, retire_sampled_textures_overlapping_writeback,
    snapshot_guest_texture_raw, take_mode27_work_details, take_texture_dirty_work_details,
    take_texture_snapshot_work_details, trace_scanout_writes,
    write_resident_texture_contents_to_guest,
};
mod scanout;
mod window;
#[cfg(test)]
pub(crate) use self::window::headless_metal_value_is_enabled;
pub(crate) use self::window::{headless_metal_requested, run_headless_presenter};

impl MetalPresenter {
    pub(super) fn render(&mut self) {
        if !self.redraw_pending {
            return;
        }
        self.upload_pending_frame();
        let translated_snapshot = self
            .active_translated_scanout
            .filter(|scanout| scanout.sequence == self.uploaded_flip_sequence)
            .and_then(|scanout| {
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
        let (frame_texture, source_dimensions) = if let Some((texture, width, height)) = host_splash
        {
            (Some(texture), Some((width, height)))
        } else {
            (
                guest_frame_texture,
                self.display.map(|display| (display.width, display.height)),
            )
        };
        let capture_due = self.capture_directory.is_some()
            && self
                .last_guest_capture
                .is_none_or(|last| last.elapsed() >= capture_interval());
        if capture_due
            && let Some((descriptor, texture)) = translated_snapshot.as_ref()
            && matches!(
                descriptor.pixel_format,
                MTLPixelFormat::RGBA8Unorm
                    | MTLPixelFormat::RGBA8Unorm_sRGB
                    | MTLPixelFormat::BGRA8Unorm
                    | MTLPixelFormat::BGRA8Unorm_sRGB
                    | MTLPixelFormat::RGBA16Float
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
        let (Some(layer), Some(queue)) = (self.layer.as_ref(), self.command_queue.as_ref()) else {
            return;
        };
        let Some(drawable) = layer.next_drawable() else {
            return;
        };
        let command_buffer = queue.new_command_buffer().to_owned();
        command_buffer.set_label("REMu guest presentation");
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
            encoder.set_render_pipeline_state(&presentation.pipeline);
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
        encoder.end_encoding();
        command_buffer.present_drawable(drawable);
        command_buffer.commit();
        let flip_completion = self
            .pending_flip
            .filter(|flip| {
                flip.sequence == self.finalized_flip_sequence
                    && flip.sequence > self.submitted_flip_sequence
            })
            .map(|flip| {
                self.submitted_flip_sequence = flip.sequence;
                remu_gpu::GpuFlipCompletion {
                    address: flip.address,
                    buffer_index: flip.buffer_index,
                    flip_argument: flip.flip_argument,
                }
            });
        self.pending_presentations.push_back(PendingPresentation {
            command_buffer,
            submitted_at: Instant::now(),
            flip_completion,
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

pub(super) fn create_presentation_resources(
    device: &Device,
) -> Result<PresentationResources, String> {
    let library = if std::env::var_os("REMU_METAL_COMPILER").is_some() {
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
    let descriptor = RenderPipelineDescriptor::new();
    descriptor.set_label("REMu scaled display presentation");
    descriptor.set_vertex_function(Some(&vertex_function));
    descriptor.set_fragment_function(Some(&fragment_function));
    descriptor
        .color_attachments()
        .object_at(0)
        .ok_or("Metal has no presentation color attachment")?
        .set_pixel_format(MTLPixelFormat::BGRA8Unorm_sRGB);
    let pipeline = device
        .new_render_pipeline_state(&descriptor)
        .map_err(|error| format!("Metal presentation pipeline: {error}"))?;
    let sampler_descriptor = SamplerDescriptor::new();
    sampler_descriptor.set_address_mode_s(MTLSamplerAddressMode::ClampToEdge);
    sampler_descriptor.set_address_mode_t(MTLSamplerAddressMode::ClampToEdge);
    sampler_descriptor.set_min_filter(MTLSamplerMinMagFilter::Linear);
    sampler_descriptor.set_mag_filter(MTLSamplerMinMagFilter::Linear);
    sampler_descriptor.set_mip_filter(MTLSamplerMipFilter::NotMipmapped);
    let sampler = device.new_sampler(&sampler_descriptor);
    Ok(PresentationResources {
        pipeline,
        _vertex_function: vertex_function,
        _fragment_function: fragment_function,
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
