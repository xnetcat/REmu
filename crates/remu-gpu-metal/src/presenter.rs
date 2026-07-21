// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::ffi::c_void;
use std::sync::mpsc::{Receiver, TryRecvError};

use core_graphics_types::geometry::CGSize;
use metal::{
    CompileOptions, Device, MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRegion,
    MTLSize, MTLStorageMode, MTLStoreAction, MTLTextureType, MTLTextureUsage, MetalLayer,
    RenderPassDescriptor, RenderPipelineDescriptor, RenderPipelineState, Texture,
    TextureDescriptor,
};
use objc::runtime::{Class, Object, YES};
use objc::{msg_send, sel, sel_impl};
use remu_gpu::{GpuEvent, GpuRegister, IndexedDraw};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::{Window, WindowId};

const GUEST_PRESENTER_SHADER: &str = r"
#include <metal_stdlib>
using namespace metal;

struct GuestPresentVertex {
    float4 position [[position]];
    float2 uv;
};

struct GuestPresentConstants {
    float4 multiply;
    float4 add;
};

vertex GuestPresentVertex remu_guest_present_vertex(uint vertex_id [[vertex_id]]) {
    const float2 positions[3] = {
        float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)
    };
    const float2 uvs[3] = {
        float2(0.0, 1.0), float2(2.0, 1.0), float2(0.0, -1.0)
    };
    GuestPresentVertex output;
    output.position = float4(positions[vertex_id], 0.0, 1.0);
    output.uv = uvs[vertex_id];
    return output;
}

fragment float4 remu_guest_present_fragment(
    GuestPresentVertex input [[stage_in]],
    texture2d<float> source [[texture(0)]],
    constant GuestPresentConstants& constants [[buffer(0)]]) {
    constexpr sampler source_sampler(
        coord::normalized, address::clamp_to_edge, filter::linear);
    return source.sample(source_sampler, input.uv) * constants.multiply + constants.add;
}
";

const DREAMING_PRESENTER_ES_PREFIX: [u32; 4] = [0xbfa0_0003, 0x93eb_ff03, 0x0008_0008, 0x8f6a_8c6b];
const DREAMING_PRESENTER_PS_PREFIX: [u32; 4] = [0xbfa0_0001, 0xbefc_0310, 0xf42c_0406, 0xfa00_0000];

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

#[derive(Clone, Copy)]
struct GuestPresentConstants {
    multiply: [f32; 4],
    add: [f32; 4],
}

#[derive(Clone, Copy)]
struct GuestTextureDescriptor {
    address: u64,
    width: u32,
    height: u32,
    pitch: u32,
}

struct MetalPresenter {
    receiver: Receiver<GpuEvent>,
    window: Option<Window>,
    device: Option<Device>,
    layer: Option<MetalLayer>,
    command_queue: Option<metal::CommandQueue>,
    guest_present_pipeline: RenderPipelineState,
    guest_texture: Option<Texture>,
    guest_texture_size: (u32, u32),
    pending_guest_draw: Option<IndexedDraw>,
    guest_present_constants: Option<GuestPresentConstants>,
    uploaded_guest_draw_sequence: u64,
    frame_texture: Option<Texture>,
    frame_texture_size: (u32, u32),
    display: Option<DisplayConfiguration>,
    pending_flip: Option<PendingFlip>,
    next_flip_sequence: u64,
    uploaded_flip_sequence: u64,
    startup_error: Option<String>,
    logged_unsupported_layout: bool,
    logged_unsupported_guest_draw: bool,
}

impl MetalPresenter {
    fn new(receiver: Receiver<GpuEvent>) -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device is available")?;
        let command_queue = device.new_command_queue();
        let guest_present_pipeline = create_guest_present_pipeline(&device)?;
        Ok(Self {
            receiver,
            window: None,
            device: Some(device),
            layer: None,
            command_queue: Some(command_queue),
            guest_present_pipeline,
            guest_texture: None,
            guest_texture_size: (0, 0),
            pending_guest_draw: None,
            guest_present_constants: None,
            uploaded_guest_draw_sequence: 0,
            frame_texture: None,
            frame_texture_size: (0, 0),
            display: None,
            pending_flip: None,
            next_flip_sequence: 0,
            uploaded_flip_sequence: 0,
            startup_error: None,
            logged_unsupported_layout: false,
            logged_unsupported_guest_draw: false,
        })
    }

    fn initialize(&mut self, event_loop: &ActiveEventLoop) -> Result<(), String> {
        let window = event_loop
            .create_window(presenter_window_attributes())
            .map_err(|error| format!("could not create the macOS window: {error}"))?;
        self.initialize_window(window)
    }

    fn initialize_window(&mut self, window: Window) -> Result<(), String> {
        let device = self
            .device
            .as_ref()
            .ok_or("Metal presenter resources are unavailable")?;
        let mut layer = MetalLayer::new();
        layer.set_device(device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        // The compiler-free presenter blits into each drawable instead of running a fullscreen
        // shader, so the drawable must permit blit writes.
        layer.set_framebuffer_only(false);
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
        eprintln!("[REMu][METAL] presenter started on {}", device.name());

        self.window = Some(window);
        self.layer = Some(layer);
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
                    if let Some(layer) = self.layer.as_ref() {
                        // Keep the drawable at guest resolution. Core Animation scales it into the
                        // window while Metal performs a simple one-to-one copy.
                        layer.set_drawable_size(CGSize::new(f64::from(width), f64::from(height)));
                    }
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
                Ok(GpuEvent::IndexedDraw(draw)) => {
                    self.pending_guest_draw = Some(draw);
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

    fn upload_pending_guest_draw(&mut self) {
        let (Some(draw), Some(device)) = (self.pending_guest_draw.as_ref(), self.device.as_ref())
        else {
            return;
        };
        if draw.sequence == self.uploaded_guest_draw_sequence {
            return;
        }
        self.uploaded_guest_draw_sequence = draw.sequence;
        let Some((descriptor, constants)) = decode_dreaming_presenter_draw(draw) else {
            if !self.logged_unsupported_guest_draw {
                eprintln!(
                    "[REMu][METAL] no translated Metal path for es={:#x} ps={:#x}",
                    draw.export_shader_address, draw.pixel_shader_address
                );
                self.logged_unsupported_guest_draw = true;
            }
            return;
        };
        let Some(row_bytes) = u64::from(descriptor.pitch).checked_mul(4) else {
            return;
        };
        let Some(byte_count) = row_bytes.checked_mul(u64::from(descriptor.height)) else {
            return;
        };
        if descriptor.address < 0x10_000 || byte_count > 128 * 1024 * 1024 {
            return;
        }
        if self.guest_texture_size != (descriptor.width, descriptor.height) {
            let texture_descriptor = TextureDescriptor::new();
            texture_descriptor.set_texture_type(MTLTextureType::D2);
            texture_descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
            texture_descriptor.set_width(u64::from(descriptor.width));
            texture_descriptor.set_height(u64::from(descriptor.height));
            texture_descriptor.set_storage_mode(MTLStorageMode::Shared);
            texture_descriptor.set_usage(MTLTextureUsage::ShaderRead);
            self.guest_texture = Some(device.new_texture(&texture_descriptor));
            self.guest_texture_size = (descriptor.width, descriptor.height);
        }
        let (Some(texture), Ok(source)) = (
            self.guest_texture.as_ref(),
            usize::try_from(descriptor.address),
        ) else {
            return;
        };
        texture.replace_region(
            MTLRegion::new_2d(
                0,
                0,
                u64::from(descriptor.width),
                u64::from(descriptor.height),
            ),
            0,
            source as *const c_void,
            row_bytes,
        );
        self.guest_present_constants = Some(constants);
        if draw.sequence == 1 {
            eprintln!(
                "[REMu][METAL] translated Dreaming Sarah presenter draw source={:#x} {}x{} target={:#x}",
                descriptor.address, descriptor.width, descriptor.height, draw.color_target_address
            );
        }
    }

    fn render(&mut self) {
        self.upload_pending_frame();
        self.upload_pending_guest_draw();
        let (Some(layer), Some(queue)) = (self.layer.as_ref(), self.command_queue.as_ref()) else {
            return;
        };
        let Some(drawable) = layer.next_drawable() else {
            return;
        };
        let command_buffer = queue.new_command_buffer();
        if let (Some(texture), Some(constants)) =
            (self.guest_texture.as_ref(), self.guest_present_constants)
        {
            let pass = RenderPassDescriptor::new();
            let Some(color) = pass.color_attachments().object_at(0) else {
                return;
            };
            color.set_texture(Some(drawable.texture()));
            color.set_load_action(MTLLoadAction::Clear);
            color.set_clear_color(metal::MTLClearColor::new(0.015, 0.02, 0.04, 1.0));
            color.set_store_action(MTLStoreAction::Store);
            let encoder = command_buffer.new_render_command_encoder(pass);
            encoder.set_render_pipeline_state(&self.guest_present_pipeline);
            encoder.set_fragment_texture(0, Some(texture));
            let constant_values = [constants.multiply, constants.add];
            encoder.set_fragment_bytes(
                0,
                u64::try_from(std::mem::size_of_val(&constant_values))
                    .expect("guest presenter constants fit in Metal's length type"),
                constant_values.as_ptr().cast::<c_void>(),
            );
            encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
            encoder.end_encoding();
        } else if let (Some(display), Some(texture)) = (self.display, self.frame_texture.as_ref()) {
            let encoder = command_buffer.new_blit_command_encoder();
            encoder.copy_from_texture(
                texture,
                0,
                0,
                MTLOrigin::default(),
                MTLSize::new(u64::from(display.width), u64::from(display.height), 1),
                drawable.texture(),
                0,
                0,
                MTLOrigin::default(),
            );
            encoder.end_encoding();
        } else {
            return;
        }
        command_buffer.present_drawable(drawable);
        command_buffer.commit();
    }
}

fn create_guest_present_pipeline(device: &metal::DeviceRef) -> Result<RenderPipelineState, String> {
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(GUEST_PRESENTER_SHADER, &options)
        .map_err(|error| format!("could not compile the Dreaming Sarah Metal shader: {error}"))?;
    let vertex = library
        .get_function("remu_guest_present_vertex", None)
        .map_err(|error| format!("could not load the guest presenter vertex shader: {error}"))?;
    let fragment = library
        .get_function("remu_guest_present_fragment", None)
        .map_err(|error| format!("could not load the guest presenter fragment shader: {error}"))?;
    let descriptor = RenderPipelineDescriptor::new();
    descriptor.set_vertex_function(Some(&vertex));
    descriptor.set_fragment_function(Some(&fragment));
    let color = descriptor
        .color_attachments()
        .object_at(0)
        .ok_or("Metal did not provide color attachment zero")?;
    color.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    device
        .new_render_pipeline_state(&descriptor)
        .map_err(|error| format!("could not create the guest presenter pipeline: {error}"))
}

fn register_value(registers: &[GpuRegister], offset: u32) -> Option<u32> {
    registers
        .binary_search_by_key(&offset, |register| register.offset)
        .ok()
        .map(|index| registers[index].value)
}

fn shader_matches_prefix(code: &[u32], prefix: &[u32]) -> bool {
    code.get(..prefix.len()) == Some(prefix)
}

fn decode_dreaming_presenter_draw(
    draw: &IndexedDraw,
) -> Option<(GuestTextureDescriptor, GuestPresentConstants)> {
    if !shader_matches_prefix(&draw.export_shader_prefix, &DREAMING_PRESENTER_ES_PREFIX)
        || !shader_matches_prefix(&draw.pixel_shader_prefix, &DREAMING_PRESENTER_PS_PREFIX)
    {
        return None;
    }

    let fields: [u32; 8] = std::array::from_fn(|index| {
        register_value(
            &draw.shader_registers,
            0x00c + u32::try_from(index).expect("texture descriptor index fits in u32"),
        )
        .unwrap_or_default()
    });
    let address = ((u64::from(fields[1] & 0xff) << 32) | u64::from(fields[0])) << 8;
    let width = ((fields[1] >> 30) | ((fields[2] & 0x3fff) << 2)).checked_add(1)?;
    let height = ((fields[2] >> 14) & 0xffff).checked_add(1)?;
    let unified_format = (fields[1] >> 20) & 0x1ff;
    let tile_mode = (fields[3] >> 20) & 0x1f;
    let texture_type = fields[3] >> 28;
    if address < 0x10_000
        || width == 0
        || width > 8_192
        || height == 0
        || height > 8_192
        || unified_format != 56
        || tile_mode != 0
        || texture_type != 9
    {
        return None;
    }
    let pitch = if fields[4] == 0 {
        width
    } else {
        (fields[4] & 0x3fff).checked_add(1)?
    };
    if pitch < width || pitch > 8_192 {
        return None;
    }

    let constants_low = register_value(&draw.shader_registers, 0x018)?;
    let constants_high = register_value(&draw.shader_registers, 0x019)?;
    let constants_address = (u64::from(constants_high & 0xffff) << 32) | u64::from(constants_low);
    let constants =
        read_guest_present_constants(constants_address).unwrap_or(GuestPresentConstants {
            multiply: [1.0; 4],
            add: [0.0; 4],
        });
    Some((
        GuestTextureDescriptor {
            address,
            width,
            height,
            pitch,
        },
        constants,
    ))
}

fn read_guest_present_constants(address: u64) -> Option<GuestPresentConstants> {
    if address < 0x10_000 {
        return None;
    }
    let address = usize::try_from(address).ok()?;
    let values: [f32; 8] = std::array::from_fn(|index| {
        // SAFETY: The AGC user-data descriptor references a live, CPU-visible guest allocation.
        // The runtime owns that mapping for the duration of the presenter thread.
        unsafe { std::ptr::read_unaligned((address + index * 4) as *const f32) }
    });
    if !values.iter().all(|value| value.is_finite()) {
        return None;
    }
    Some(GuestPresentConstants {
        multiply: values[..4].try_into().expect("four multiply constants"),
        add: values[4..].try_into().expect("four add constants"),
    })
}

fn presenter_window_attributes() -> winit::window::WindowAttributes {
    Window::default_attributes()
        .with_title("REmu — Dreaming Sarah")
        .with_inner_size(LogicalSize::new(1280, 720))
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
                    let (width, height) =
                        self.display.map_or((size.width, size.height), |display| {
                            (display.width, display.height)
                        });
                    layer.set_drawable_size(CGSize::new(f64::from(width), f64::from(height)));
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

pub fn run_presenter(receiver: Receiver<GpuEvent>) -> Result<(), String> {
    eprintln!("[REMu][METAL] starting AppKit event loop");
    let mut presenter = MetalPresenter::new(receiver)?;
    let mut builder = EventLoop::builder();
    builder
        .with_activation_policy(ActivationPolicy::Regular)
        .with_activate_ignoring_other_apps(true);
    let event_loop = builder
        .build()
        .map_err(|error| format!("event loop failed: {error}"))?;
    // A Rosetta-hosted command-line binary is not launched through an application bundle, and
    // LaunchServices does not reliably deliver NSApplicationDidFinishLaunching. Complete the
    // AppKit launch explicitly, as SharpEmu's bare Metal presenter does. Winit sees the launched
    // application when `run_app` installs its handler and then emits the normal init/resume events.
    unsafe {
        let application_class =
            Class::get("NSApplication").ok_or("could not find the NSApplication class")?;
        let application: *mut Object = msg_send![application_class, sharedApplication];
        if application.is_null() {
            return Err("could not create the shared NSApplication".to_owned());
        }
        let _: () = msg_send![application, finishLaunching];
        // LaunchServices can still omit the delegate notification for an unbundled translated
        // executable. Deliver the standard callback to Winit's installed application delegate so
        // its `run_app` path knows AppKit is launched and dispatches `Resumed`.
        let delegate: *mut Object = msg_send![application, delegate];
        if delegate.is_null() {
            return Err("AppKit did not install an application delegate".to_owned());
        }
        let string_class = Class::get("NSString").ok_or("could not find the NSString class")?;
        let notification_class =
            Class::get("NSNotification").ok_or("could not find the NSNotification class")?;
        let notification_name: *mut Object =
            msg_send![string_class, stringWithUTF8String: c"REMuDidFinishLaunching".as_ptr()];
        let notification: *mut Object = msg_send![notification_class,
            notificationWithName: notification_name
            object: application
        ];
        if notification.is_null() {
            return Err("could not create the AppKit launch notification".to_owned());
        }
        let _: () = msg_send![delegate, applicationDidFinishLaunching: notification];
    }
    event_loop.set_control_flow(ControlFlow::Poll);
    // Build the window outside Winit's launch/resume callback. This keeps AppKit lifecycle work
    // out of the callback and lets the compiler-free blit presenter start immediately.
    #[allow(deprecated)]
    let window = event_loop
        .create_window(presenter_window_attributes())
        .map_err(|error| format!("could not create the macOS window: {error}"))?;
    presenter.initialize_window(window)?;
    eprintln!("[REMu][METAL] AppKit event loop ready");
    event_loop
        .run_app(&mut presenter)
        .map_err(|error| format!("event loop failed: {error}"))?;
    if let Some(error) = presenter.startup_error {
        Err(error)
    } else {
        Ok(())
    }
}
