// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::ffi::c_void;
use std::sync::mpsc::{Receiver, TryRecvError};

use core_graphics_types::geometry::CGSize;
use metal::{
    Device, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize, MTLStorageMode, MTLTextureType,
    MTLTextureUsage, MetalLayer, Texture, TextureDescriptor,
};
use objc::runtime::{Class, Object, YES};
use objc::{msg_send, sel, sel_impl};
use remu_gpu::GpuEvent;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::{Window, WindowId};

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
    fn new(receiver: Receiver<GpuEvent>) -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device is available")?;
        let command_queue = device.new_command_queue();
        Ok(Self {
            receiver,
            window: None,
            device: Some(device),
            layer: None,
            command_queue: Some(command_queue),
            frame_texture: None,
            frame_texture_size: (0, 0),
            display: None,
            pending_flip: None,
            next_flip_sequence: 0,
            uploaded_flip_sequence: 0,
            startup_error: None,
            logged_unsupported_layout: false,
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
                Ok(GpuEvent::IndexedDraw(_)) => {
                    // The next renderer milestone consumes these snapshots after translating the
                    // referenced GFX10 programs to MSL. Flip presentation remains independent.
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
        let (Some(display), Some(layer), Some(queue), Some(texture)) = (
            self.display,
            self.layer.as_ref(),
            self.command_queue.as_ref(),
            self.frame_texture.as_ref(),
        ) else {
            return;
        };
        let Some(drawable) = layer.next_drawable() else {
            return;
        };
        let command_buffer = queue.new_command_buffer();
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
        command_buffer.present_drawable(drawable);
        command_buffer.commit();
    }
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
