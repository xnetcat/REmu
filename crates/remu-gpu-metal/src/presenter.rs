// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::ffi::c_void;
use std::sync::mpsc::{Receiver, TryRecvError};

use core_graphics_types::geometry::CGSize;
use metal::{
    CompileOptions, Device, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRegion,
    MTLStorageMode, MTLStoreAction, MTLTextureType, MTLTextureUsage, MetalLayer,
    RenderPassDescriptor, RenderPipelineDescriptor, RenderPipelineState, Texture,
    TextureDescriptor,
};
use objc::runtime::{Object, YES};
use objc::{msg_send, sel, sel_impl};
use remu_gpu::GpuEvent;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::{Window, WindowId};

const PRESENT_SHADER: &str = r"
#include <metal_stdlib>
using namespace metal;

struct PresentVertex {
    float4 position [[position]];
    float2 uv;
};

vertex PresentVertex remu_present_vertex(uint vertex_id [[vertex_id]]) {
    const float2 positions[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    const float2 uvs[3] = { float2(0.0, 1.0), float2(2.0, 1.0), float2(0.0, -1.0) };
    PresentVertex output;
    output.position = float4(positions[vertex_id], 0.0, 1.0);
    output.uv = uvs[vertex_id];
    return output;
}

fragment float4 remu_present_fragment(
    PresentVertex input [[stage_in]],
    texture2d<float> frame [[texture(0)]]) {
    constexpr sampler frame_sampler(coord::normalized, address::clamp_to_edge, filter::linear);
    return frame.sample(frame_sampler, input.uv);
}
";

#[derive(Clone, Copy)]
struct DisplayConfiguration {
    width: u32,
    height: u32,
    pitch: u32,
    pixel_format: u64,
    tiling_mode: u32,
}

#[derive(Clone, Copy)]
struct PendingFlip {
    address: u64,
    sequence: u64,
}

struct MetalPresenter {
    receiver: Receiver<GpuEvent>,
    window: Option<Window>,
    device: Option<Device>,
    layer: Option<MetalLayer>,
    command_queue: Option<metal::CommandQueue>,
    pipeline: Option<RenderPipelineState>,
    frame_texture: Option<Texture>,
    frame_texture_size: (u32, u32),
    display: Option<DisplayConfiguration>,
    pending_flip: Option<PendingFlip>,
    next_flip_sequence: u64,
    uploaded_flip_sequence: u64,
    startup_error: Option<String>,
    logged_unsupported_layout: bool,
}

impl MetalPresenter {
    fn new(receiver: Receiver<GpuEvent>) -> Self {
        Self {
            receiver,
            window: None,
            device: None,
            layer: None,
            command_queue: None,
            pipeline: None,
            frame_texture: None,
            frame_texture_size: (0, 0),
            display: None,
            pending_flip: None,
            next_flip_sequence: 0,
            uploaded_flip_sequence: 0,
            startup_error: None,
            logged_unsupported_layout: false,
        }
    }

    fn initialize(&mut self, event_loop: &ActiveEventLoop) -> Result<(), String> {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("REmu — Dreaming Sarah")
                    .with_inner_size(LogicalSize::new(1280, 720)),
            )
            .map_err(|error| format!("could not create the macOS window: {error}"))?;
        let device = Device::system_default().ok_or("no Metal device is available")?;
        let mut layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_framebuffer_only(true);
        layer.set_presents_with_transaction(false);

        // SAFETY: Winit owns a live NSView on the main thread. NSView retains the CAMetalLayer;
        // this presenter also keeps its owning MetalLayer handle until the event loop exits.
        unsafe {
            let handle = window
                .window_handle()
                .map_err(|error| format!("could not get the AppKit window handle: {error}"))?;
            let RawWindowHandle::AppKit(appkit) = handle.as_raw() else {
                return Err("winit did not create an AppKit window".to_owned());
            };
            let view = appkit.ns_view.as_ptr().cast::<Object>();
            let _: () = msg_send![view, setWantsLayer: YES];
            let _: () = msg_send![view, setLayer: layer.as_mut()];
        }

        let size = window.inner_size();
        layer.set_drawable_size(CGSize::new(f64::from(size.width), f64::from(size.height)));
        let pipeline = create_present_pipeline(&device)?;
        let command_queue = device.new_command_queue();
        eprintln!("[REMu][METAL] presenter started on {}", device.name());

        self.window = Some(window);
        self.device = Some(device);
        self.layer = Some(layer);
        self.command_queue = Some(command_queue);
        self.pipeline = Some(pipeline);
        Ok(())
    }

    fn receive_events(&mut self, event_loop: &ActiveEventLoop) {
        loop {
            match self.receiver.try_recv() {
                Ok(GpuEvent::DisplayRegistered {
                    width,
                    height,
                    pitch,
                    pixel_format,
                    tiling_mode,
                    ..
                }) => {
                    self.display = Some(DisplayConfiguration {
                        width,
                        height,
                        pitch,
                        pixel_format,
                        tiling_mode,
                    });
                    self.frame_texture = None;
                    eprintln!(
                        "[REMu][METAL] display registered {width}x{height} pitch={pitch} format={pixel_format:#x} tiling={tiling_mode}"
                    );
                }
                Ok(GpuEvent::Flip { address, .. }) => {
                    self.next_flip_sequence = self.next_flip_sequence.saturating_add(1);
                    self.pending_flip = Some(PendingFlip {
                        address,
                        sequence: self.next_flip_sequence,
                    });
                }
                Ok(GpuEvent::Shutdown) | Err(TryRecvError::Disconnected) => {
                    event_loop.exit();
                    return;
                }
                Err(TryRecvError::Empty) => return,
            }
        }
    }

    fn upload_pending_frame(&mut self) {
        let (Some(display), Some(flip), Some(device)) =
            (self.display, self.pending_flip, self.device.as_ref())
        else {
            return;
        };
        if flip.sequence == self.uploaded_flip_sequence {
            return;
        }
        if display.tiling_mode != 0 || display.pitch < display.width {
            if !self.logged_unsupported_layout {
                eprintln!(
                    "[REMu][METAL] cannot upload display layout tiling={} pitch={} width={} yet",
                    display.tiling_mode, display.pitch, display.width
                );
                self.logged_unsupported_layout = true;
            }
            return;
        }
        let Some(row_bytes) = u64::from(display.pitch).checked_mul(4) else {
            return;
        };
        let Some(frame_bytes) = row_bytes.checked_mul(u64::from(display.height)) else {
            return;
        };
        if flip.address == 0 || frame_bytes > 128 * 1024 * 1024 {
            return;
        }
        if self.frame_texture_size != (display.width, display.height) {
            let descriptor = TextureDescriptor::new();
            descriptor.set_texture_type(MTLTextureType::D2);
            descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            descriptor.set_width(u64::from(display.width));
            descriptor.set_height(u64::from(display.height));
            descriptor.set_storage_mode(MTLStorageMode::Shared);
            descriptor.set_usage(MTLTextureUsage::ShaderRead);
            self.frame_texture = Some(device.new_texture(&descriptor));
            self.frame_texture_size = (display.width, display.height);
        }
        let Some(texture) = self.frame_texture.as_ref() else {
            return;
        };
        let Ok(bytes_per_row) = usize::try_from(row_bytes) else {
            return;
        };
        let Ok(source) = usize::try_from(flip.address) else {
            return;
        };
        texture.replace_region(
            MTLRegion::new_2d(0, 0, u64::from(display.width), u64::from(display.height)),
            0,
            source as *const c_void,
            bytes_per_row as u64,
        );
        self.uploaded_flip_sequence = flip.sequence;
        if flip.sequence == 1 {
            eprintln!(
                "[REMu][METAL] uploaded first guest display buffer format={:#x}",
                display.pixel_format
            );
        }
    }

    fn render(&mut self) {
        self.upload_pending_frame();
        let (Some(layer), Some(queue), Some(pipeline)) = (
            self.layer.as_ref(),
            self.command_queue.as_ref(),
            self.pipeline.as_ref(),
        ) else {
            return;
        };
        let Some(drawable) = layer.next_drawable() else {
            return;
        };
        let pass = RenderPassDescriptor::new();
        let Some(color) = pass.color_attachments().object_at(0) else {
            return;
        };
        color.set_texture(Some(drawable.texture()));
        color.set_load_action(MTLLoadAction::Clear);
        color.set_clear_color(metal::MTLClearColor::new(0.015, 0.02, 0.04, 1.0));
        color.set_store_action(MTLStoreAction::Store);

        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_render_command_encoder(pass);
        if let Some(texture) = self.frame_texture.as_ref() {
            encoder.set_render_pipeline_state(pipeline);
            encoder.set_fragment_texture(0, Some(texture));
            encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
        }
        encoder.end_encoding();
        command_buffer.present_drawable(drawable);
        command_buffer.commit();
    }
}

impl ApplicationHandler for MetalPresenter {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() || self.startup_error.is_some() {
            return;
        }
        if let Err(error) = self.initialize(event_loop) {
            eprintln!("[REMu][METAL] {error}");
            self.startup_error = Some(error);
            event_loop.exit();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if size.width != 0
                    && size.height != 0
                    && let Some(layer) = self.layer.as_ref()
                {
                    layer.set_drawable_size(CGSize::new(
                        f64::from(size.width),
                        f64::from(size.height),
                    ));
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.receive_events(event_loop);
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

fn create_present_pipeline(device: &metal::DeviceRef) -> Result<RenderPipelineState, String> {
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(PRESENT_SHADER, &options)
        .map_err(|error| format!("could not compile the Metal presenter shader: {error}"))?;
    let vertex = library
        .get_function("remu_present_vertex", None)
        .map_err(|error| format!("missing presenter vertex function: {error}"))?;
    let fragment = library
        .get_function("remu_present_fragment", None)
        .map_err(|error| format!("missing presenter fragment function: {error}"))?;
    let descriptor = RenderPipelineDescriptor::new();
    descriptor.set_vertex_function(Some(&vertex));
    descriptor.set_fragment_function(Some(&fragment));
    let Some(color) = descriptor.color_attachments().object_at(0) else {
        return Err("Metal did not expose color attachment zero".to_owned());
    };
    color.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    device
        .new_render_pipeline_state(&descriptor)
        .map_err(|error| format!("could not create the Metal presenter pipeline: {error}"))
}

pub fn run_presenter(receiver: Receiver<GpuEvent>) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|error| format!("event loop failed: {error}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut presenter = MetalPresenter::new(receiver);
    event_loop
        .run_app(&mut presenter)
        .map_err(|error| format!("event loop failed: {error}"))?;
    if let Some(error) = presenter.startup_error {
        Err(error)
    } else {
        Ok(())
    }
}
