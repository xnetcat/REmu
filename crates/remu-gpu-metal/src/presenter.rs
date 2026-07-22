// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::time::{Duration, Instant};

use core_graphics_types::geometry::CGSize;
use metal::{
    Buffer, CompileOptions, Device, Function, MTLCommandBufferStatus, MTLIndexType, MTLLoadAction,
    MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLResourceOptions,
    MTLSamplerAddressMode, MTLSamplerMinMagFilter, MTLSamplerMipFilter, MTLScissorRect, MTLSize,
    MTLStorageMode, MTLStoreAction, MTLTextureType, MTLTextureUsage, MTLViewport, MetalLayer,
    RenderPassDescriptor, RenderPipelineDescriptor, RenderPipelineState, SamplerDescriptor,
    SamplerState, Texture, TextureDescriptor,
};
use objc::runtime::{Class, Object, YES};
use objc::{msg_send, sel, sel_impl};
use remu_gpu::{
    GpuEvent, GpuRegister, HostPadState, IndexedDraw, PAD_BUTTON_CIRCLE, PAD_BUTTON_CROSS,
    PAD_BUTTON_DOWN, PAD_BUTTON_L1, PAD_BUTTON_L2, PAD_BUTTON_LEFT, PAD_BUTTON_OPTIONS,
    PAD_BUTTON_R1, PAD_BUTTON_R2, PAD_BUTTON_RIGHT, PAD_BUTTON_SQUARE, PAD_BUTTON_TRIANGLE,
    PAD_BUTTON_UP, set_host_pad_state,
};
use remu_shader::{ImageBinding, ShaderEvaluation};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::{Window, WindowId};

const DREAMING_PRESENTER_ES_PREFIX: [u32; 4] = [0xbfa0_0003, 0x93eb_ff03, 0x0008_0008, 0x8f6a_8c6b];
const DREAMING_PRESENTER_PS_PREFIX: [u32; 4] = [0xbfa0_0001, 0xbefc_0310, 0xf42c_0406, 0xfa00_0000];
const DREAMING_FILL_ES_PREFIX: [u32; 4] = [0xbfa0_0001, 0x7e00_0000, 0x7e00_0000, 0x7e00_0000];
const DREAMING_FILL_PS_PREFIX: [u32; 4] = [0xd52f_0000, 0x0000_0200, 0xd52f_0001, 0x0000_0602];
const DREAMING_SOLID_PS_PREFIX: [u32; 4] = [0xf428_0100, 0xfa00_0000, 0xbf8c_c07f, 0xd52f_0000];
const DREAMING_TEXTURE_PS_PREFIX: [u32; 4] = [0xbfa0_0001, 0xbefc_030c, 0xbeea_047e, 0xbefe_0a7e];
const DREAMING_COMPOSITOR_PS_PREFIX: [u32; 4] =
    [0xbfa0_0003, 0xf424_1a86, 0xfa00_0000, 0xbefc_0310];

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
    tile_mode: u32,
}

#[derive(Clone, Copy)]
struct ExecutedFill {
    address: u64,
    width: u32,
    height: u32,
    format: u32,
    number_type: u32,
    tile_mode: u32,
    rgba: [u8; 4],
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum SoftwareDrawKind {
    Solid,
    Texture,
    Compositor,
}

#[derive(Clone, Copy)]
struct ExecutedSoftwareDraw {
    kind: SoftwareDrawKind,
    source: u64,
    target: u64,
    width: u32,
    height: u32,
}

#[derive(Clone, Copy, Debug)]
struct SpriteVertex {
    position: [f32; 2],
    uv: [f32; 2],
    opacity: f32,
}

struct CompiledShaderPair {
    pipeline: RenderPipelineState,
    vertex_function: Function,
    pixel_function: Function,
    vertex: remu_shader_metal::MetalShader,
    pixel: remu_shader_metal::MetalShader,
}

struct ShaderCompileResult {
    key: (u64, u64),
    compiled: Result<CompiledShaderPair, String>,
}

struct UploadedImageTexture {
    texture: Texture,
    descriptor: DecodedImageDescriptor,
    writable: bool,
}

#[derive(Clone, Copy)]
struct DecodedImageDescriptor {
    guest: GuestTextureDescriptor,
    destination_select: u32,
}

#[derive(Clone, Copy, Debug)]
enum GuestScissor {
    Full,
    Rect(MTLScissorRect),
    Empty,
}

struct MetalPresenter {
    receiver: Receiver<GpuEvent>,
    window: Option<Window>,
    device: Option<Device>,
    layer: Option<MetalLayer>,
    command_queue: Option<metal::CommandQueue>,
    guest_texture: Option<Texture>,
    guest_texture_size: (u32, u32),
    guest_bgra_bytes: Vec<u8>,
    pending_guest_draw: Option<IndexedDraw>,
    uploaded_guest_draw_sequence: u64,
    frame_texture: Option<Texture>,
    frame_texture_size: (u32, u32),
    display: Option<DisplayConfiguration>,
    pending_flip: Option<PendingFlip>,
    next_flip_sequence: u64,
    uploaded_flip_sequence: u64,
    startup_error: Option<String>,
    logged_unsupported_layout: bool,
    logged_unsupported_guest_draws: HashSet<([u32; 4], [u32; 4])>,
    logged_guest_layouts: HashSet<(u64, u32, u32, u32)>,
    captured_guest_frames: HashSet<(u64, u32, u32, u32, u64)>,
    last_guest_capture: Option<Instant>,
    logged_fill_draw: bool,
    logged_software_draws: HashSet<(SoftwareDrawKind, u64, u64)>,
    compiled_shader_pairs: HashMap<(u64, u64), CompiledShaderPair>,
    compiling_shader_pairs: HashSet<(u64, u64)>,
    failed_shader_pairs: HashSet<(u64, u64)>,
    failed_draw_execution: HashSet<(u64, u64)>,
    capture_directory: Option<PathBuf>,
    pressed_keys: HashSet<KeyCode>,
    shader_compile_sender: Sender<IndexedDraw>,
    shader_compile_receiver: Receiver<ShaderCompileResult>,
}

impl MetalPresenter {
    fn new(receiver: Receiver<GpuEvent>) -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device is available")?;
        let command_queue = device.new_command_queue();
        let (shader_compile_sender, shader_compile_requests) = mpsc::channel::<IndexedDraw>();
        let (shader_compile_results, shader_compile_receiver) =
            mpsc::channel::<ShaderCompileResult>();
        let compiler_device = device.clone();
        std::thread::Builder::new()
            .name("remu-metal-compiler".to_owned())
            .spawn(move || {
                while let Ok(draw) = shader_compile_requests.recv() {
                    let key = (draw.export_shader_address, draw.pixel_shader_address);
                    let compiled = objc::rc::autoreleasepool(|| {
                        Self::compile_draw_shader_pair(&compiler_device, &draw)
                    });
                    if shader_compile_results
                        .send(ShaderCompileResult { key, compiled })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .map_err(|error| format!("could not start Metal compiler worker: {error}"))?;
        Ok(Self {
            receiver,
            window: None,
            device: Some(device),
            layer: None,
            command_queue: Some(command_queue),
            guest_texture: None,
            guest_texture_size: (0, 0),
            guest_bgra_bytes: Vec::new(),
            pending_guest_draw: None,
            uploaded_guest_draw_sequence: 0,
            frame_texture: None,
            frame_texture_size: (0, 0),
            display: None,
            pending_flip: None,
            next_flip_sequence: 0,
            uploaded_flip_sequence: 0,
            startup_error: None,
            logged_unsupported_layout: false,
            logged_unsupported_guest_draws: HashSet::new(),
            logged_guest_layouts: HashSet::new(),
            captured_guest_frames: HashSet::new(),
            last_guest_capture: None,
            logged_fill_draw: false,
            logged_software_draws: HashSet::new(),
            compiled_shader_pairs: HashMap::new(),
            compiling_shader_pairs: HashSet::new(),
            failed_shader_pairs: HashSet::new(),
            failed_draw_execution: HashSet::new(),
            capture_directory: std::env::var_os("REMU_CAPTURE_DIR").map(PathBuf::from),
            pressed_keys: HashSet::new(),
            shader_compile_sender,
            shader_compile_receiver,
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

    #[allow(clippy::too_many_lines)]
    fn receive_events(&mut self, event_loop: &ActiveEventLoop) {
        self.receive_shader_compilations();
        // A bounded batch prevents a continuously producing guest from starving redraw and input
        // delivery. The bounded sender applies backpressure until the next AppKit iteration.
        for _ in 0..16 {
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
                    if self.guest_texture.is_none()
                        && let Some(layer) = self.layer.as_ref()
                    {
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
                    self.compile_draw_shaders(&draw);
                    match self.execute_compiled_draw(&draw) {
                        Ok(()) => continue,
                        Err(error) => {
                            let key = (draw.export_shader_address, draw.pixel_shader_address);
                            if self.failed_draw_execution.insert(key) {
                                eprintln!(
                                    "[REMu][METAL] Gen5 draw es={:#x} ps={:#x} could not execute: {error}",
                                    key.0, key.1
                                );
                            }
                        }
                    }
                    if is_dreaming_presenter_draw(&draw) {
                        self.pending_guest_draw = Some(*draw);
                    } else if let Some(fill) = execute_dreaming_fill_draw(&draw) {
                        if !self.logged_fill_draw {
                            eprintln!(
                                "[REMu][METAL] executed Dreaming Sarah fill target={:#x} {}x{} format={} number={} tile={} rgba={:02x?}",
                                fill.address,
                                fill.width,
                                fill.height,
                                fill.format,
                                fill.number_type,
                                fill.tile_mode,
                                fill.rgba,
                            );
                            self.logged_fill_draw = true;
                        }
                    } else if let Some(executed) = execute_dreaming_software_draw(&draw) {
                        if self.logged_software_draws.len() < 32
                            && self.logged_software_draws.insert((
                                executed.kind,
                                executed.source,
                                executed.target,
                            ))
                        {
                            eprintln!(
                                "[REMu][METAL] executed Dreaming Sarah {:?} compatibility draw source={:#x} target={:#x} {}x{}",
                                executed.kind,
                                executed.source,
                                executed.target,
                                executed.width,
                                executed.height,
                            );
                        }
                    } else if self.logged_unsupported_guest_draws.len() < 16
                        && self
                            .logged_unsupported_guest_draws
                            .insert((draw.export_shader_prefix, draw.pixel_shader_prefix))
                    {
                        let pixel_scalars = pixel_scalars(&draw);
                        eprintln!(
                            "[REMu][METAL] no translated Metal path for es={:#x}/{:x?} ps={:#x}/{:x?} indexed={} count={} instances={} target={:#x} cb_info={:#x} cb_attrib2={:#x} cb_attrib3={:#x} target_mask={:#x} ps_scalars={pixel_scalars:x?}",
                            draw.export_shader_address,
                            draw.export_shader_prefix,
                            draw.pixel_shader_address,
                            draw.pixel_shader_prefix,
                            draw.indexed,
                            draw.index_count,
                            draw.instance_count,
                            draw.color_target_address,
                            register_value(&draw.context_registers, 0x31c).unwrap_or_default(),
                            register_value(&draw.context_registers, 0x3b0).unwrap_or_default(),
                            register_value(&draw.context_registers, 0x3b8).unwrap_or_default(),
                            register_value(&draw.context_registers, 0x08e).unwrap_or_default(),
                        );
                    }
                }
                Ok(GpuEvent::Shutdown) | Err(TryRecvError::Disconnected) => {
                    event_loop.exit();
                    return;
                }
                Err(TryRecvError::Empty) => return,
            }
        }
    }

    fn compile_draw_shaders(&mut self, draw: &IndexedDraw) {
        let key = (draw.export_shader_address, draw.pixel_shader_address);
        if self.compiled_shader_pairs.contains_key(&key)
            || self.compiling_shader_pairs.contains(&key)
            || self.failed_shader_pairs.contains(&key)
        {
            return;
        }
        self.compiling_shader_pairs.insert(key);
        if self.shader_compile_sender.send(draw.clone()).is_err() {
            self.compiling_shader_pairs.remove(&key);
            self.failed_shader_pairs.insert(key);
            eprintln!(
                "[REMu][METAL] Gen5 shader pair es={:#x} ps={:#x} failed: compiler worker stopped",
                key.0, key.1
            );
        }
    }

    fn receive_shader_compilations(&mut self) {
        while let Ok(result) = self.shader_compile_receiver.try_recv() {
            self.compiling_shader_pairs.remove(&result.key);
            match result.compiled {
                Ok(compiled) => {
                    eprintln!(
                        "[REMu][METAL] compiled Gen5 shader pair es={:#x} ps={:#x} MSL={vertex_bytes}/{pixel_bytes} bytes",
                        result.key.0,
                        result.key.1,
                        vertex_bytes = compiled.vertex.source.len(),
                        pixel_bytes = compiled.pixel.source.len(),
                    );
                    self.compiled_shader_pairs.insert(result.key, compiled);
                }
                Err(error) => {
                    self.failed_shader_pairs.insert(result.key);
                    eprintln!(
                        "[REMu][METAL] Gen5 shader pair es={:#x} ps={:#x} failed: {error}",
                        result.key.0, result.key.1
                    );
                }
            }
        }
    }

    fn compile_draw_shader_pair(
        device: &Device,
        draw: &IndexedDraw,
    ) -> Result<CompiledShaderPair, String> {
        let export = draw
            .export_shader
            .as_ref()
            .ok_or("draw has no decoded export shader")?;
        let pixel = draw
            .pixel_shader
            .as_ref()
            .ok_or("draw has no decoded pixel shader")?;
        let export_evaluation = draw
            .export_shader_evaluation
            .as_ref()
            .ok_or("draw has no evaluated export shader resources")?;
        let pixel_evaluation = draw
            .pixel_shader_evaluation
            .as_ref()
            .ok_or("draw has no evaluated pixel shader resources")?;
        let mut host_location = 0_u32;
        let pixel_outputs = (0..8)
            .filter_map(|guest_slot| {
                let mask = (pixel.program.pixel_color_export_masks() >> (guest_slot * 4)) & 0xf;
                (mask != 0).then(|| {
                    let binding = remu_shader::PixelOutputBinding {
                        guest_slot,
                        host_location,
                        kind: remu_shader::PixelOutputKind::Float,
                    };
                    host_location += 1;
                    binding
                })
            })
            .collect();
        let pixel_options = remu_shader_metal::MetalCompileOptions::pixel(
            pixel_outputs,
            register_value(&draw.context_registers, 0x1b3).unwrap_or_default(),
            register_value(&draw.context_registers, 0x1b4).unwrap_or_default(),
        );
        let pixel_msl =
            remu_shader_metal::compile_msl(&pixel.program, pixel_evaluation, &pixel_options)
                .map_err(|error| format!("pixel lowering: {error}"))?;
        let vertex_options =
            remu_shader_metal::MetalCompileOptions::vertex(pixel_msl.interpolated_attribute_count);
        let vertex_msl =
            remu_shader_metal::compile_msl(&export.program, export_evaluation, &vertex_options)
                .map_err(|error| format!("vertex lowering: {error}"))?;
        let compile_options = CompileOptions::new();
        let vertex_library = device
            .new_library_with_source(&vertex_msl.source, &compile_options)
            .map_err(|error| format!("Metal vertex compilation: {error}"))?;
        let pixel_library = device
            .new_library_with_source(&pixel_msl.source, &compile_options)
            .map_err(|error| format!("Metal pixel compilation: {error}"))?;
        let vertex_function = vertex_library
            .get_function(&vertex_msl.entry_point, None)
            .map_err(|error| format!("Metal vertex function: {error}"))?;
        let pixel_function = pixel_library
            .get_function(&pixel_msl.entry_point, None)
            .map_err(|error| format!("Metal pixel function: {error}"))?;
        let descriptor = RenderPipelineDescriptor::new();
        descriptor.set_vertex_function(Some(&vertex_function));
        descriptor.set_fragment_function(Some(&pixel_function));
        for location in 0..pixel_options.pixel_outputs.len() {
            descriptor
                .color_attachments()
                .object_at(u64::try_from(location).expect("color location fits u64"))
                .ok_or_else(|| format!("Metal has no color attachment {location}"))?
                .set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        }
        let pipeline = device
            .new_render_pipeline_state(&descriptor)
            .map_err(|error| format!("Metal render pipeline: {error}"))?;
        Ok(CompiledShaderPair {
            pipeline,
            vertex_function,
            pixel_function,
            vertex: vertex_msl,
            pixel: pixel_msl,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn execute_compiled_draw(&mut self, draw: &IndexedDraw) -> Result<(), String> {
        let started = Instant::now();
        let trace = draw.sequence <= 4;
        let key = (draw.export_shader_address, draw.pixel_shader_address);
        let compiled = self
            .compiled_shader_pairs
            .get(&key)
            .ok_or("shader pair has no compiled Metal pipeline")?;
        let export_evaluation = draw
            .export_shader_evaluation
            .as_ref()
            .ok_or("draw has no export resource evaluation")?;
        let pixel_evaluation = draw
            .pixel_shader_evaluation
            .as_ref()
            .ok_or("draw has no pixel resource evaluation")?;
        let device = self.device.as_ref().ok_or("Metal device is unavailable")?;
        let queue = self
            .command_queue
            .as_ref()
            .ok_or("Metal command queue is unavailable")?;
        let target = decode_render_target(draw).ok_or("unsupported render-target descriptor")?;
        let target_bytes = snapshot_guest_texture_rgba(target)
            .map_err(|error| format!("render target guest memory is unreadable: {error}"))?;
        if trace {
            eprintln!(
                "[REMu][METAL] Gen5 draw={} target snapshot completed in {:?}",
                draw.sequence,
                started.elapsed()
            );
        }
        let target_texture = create_rgba_texture(
            device,
            target.width,
            target.height,
            MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead,
            &target_bytes,
        )?;

        let export_buffers = upload_global_buffers(device, export_evaluation)?;
        let pixel_buffers = upload_global_buffers(device, pixel_evaluation)?;
        let export_textures = upload_image_textures(device, export_evaluation)?;
        let pixel_textures = upload_image_textures(device, pixel_evaluation)?;
        let (export_sampler_buffer, _export_samplers) = create_sampler_argument_buffer(
            device,
            &compiled.vertex_function,
            compiled.vertex.bindings.sampler_argument_buffer_index,
            export_evaluation,
        )?;
        let (pixel_sampler_buffer, _pixel_samplers) = create_sampler_argument_buffer(
            device,
            &compiled.pixel_function,
            compiled.pixel.bindings.sampler_argument_buffer_index,
            pixel_evaluation,
        )?;
        if trace {
            eprintln!(
                "[REMu][METAL] Gen5 draw={} resources uploaded in {:?}",
                draw.sequence,
                started.elapsed()
            );
        }

        let pass = RenderPassDescriptor::new();
        let attachment = pass
            .color_attachments()
            .object_at(0)
            .ok_or("Metal render pass has no color attachment zero")?;
        attachment.set_texture(Some(&target_texture));
        attachment.set_load_action(MTLLoadAction::Load);
        attachment.set_store_action(MTLStoreAction::Store);
        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_render_command_encoder(pass);
        encoder.set_render_pipeline_state(&compiled.pipeline);
        let scissor = decode_scissor(draw, target.width, target.height)?;
        let viewport = decode_viewport(draw, target.width, target.height, scissor);
        encoder.set_viewport(viewport);
        if let GuestScissor::Rect(rect) = scissor {
            encoder.set_scissor_rect(rect);
        }
        if trace {
            eprintln!(
                "[REMu][METAL] Gen5 draw={} viewport=({:.1},{:.1} {:.1}x{:.1} {:.3}..{:.3}) scissor={scissor:?} registers=[{:08x},{:08x},{:08x},{:08x}]",
                draw.sequence,
                viewport.originX,
                viewport.originY,
                viewport.width,
                viewport.height,
                viewport.znear,
                viewport.zfar,
                register_value(&draw.context_registers, 0x10f).unwrap_or_default(),
                register_value(&draw.context_registers, 0x110).unwrap_or_default(),
                register_value(&draw.context_registers, 0x111).unwrap_or_default(),
                register_value(&draw.context_registers, 0x112).unwrap_or_default(),
            );
        }
        for (index, buffer) in export_buffers.iter().enumerate() {
            encoder.set_vertex_buffer(
                u64::try_from(index).map_err(|_| "vertex buffer index overflow")?,
                Some(buffer),
                0,
            );
        }
        for (index, buffer) in pixel_buffers.iter().enumerate() {
            encoder.set_fragment_buffer(
                u64::try_from(index).map_err(|_| "pixel buffer index overflow")?,
                Some(buffer),
                0,
            );
        }
        let export_scalar_bytes = export_evaluation.initial_scalar_registers.len() * 4;
        encoder.set_vertex_bytes(
            u64::from(compiled.vertex.bindings.initial_scalar_buffer_index),
            u64::try_from(export_scalar_bytes).map_err(|_| "vertex scalar byte count overflow")?,
            export_evaluation
                .initial_scalar_registers
                .as_ptr()
                .cast::<c_void>(),
        );
        let pixel_scalar_bytes = pixel_evaluation.initial_scalar_registers.len() * 4;
        encoder.set_fragment_bytes(
            u64::from(compiled.pixel.bindings.initial_scalar_buffer_index),
            u64::try_from(pixel_scalar_bytes).map_err(|_| "pixel scalar byte count overflow")?,
            pixel_evaluation
                .initial_scalar_registers
                .as_ptr()
                .cast::<c_void>(),
        );
        for (index, texture) in export_textures.iter().enumerate() {
            encoder.set_vertex_texture(
                u64::try_from(index).map_err(|_| "vertex texture index overflow")?,
                Some(&texture.texture),
            );
        }
        for (index, texture) in pixel_textures.iter().enumerate() {
            encoder.set_fragment_texture(
                u64::try_from(index).map_err(|_| "pixel texture index overflow")?,
                Some(&texture.texture),
            );
        }
        if let (Some(index), Some(buffer)) = (
            compiled.vertex.bindings.sampler_argument_buffer_index,
            export_sampler_buffer.as_ref(),
        ) {
            encoder.set_vertex_buffer(u64::from(index), Some(buffer), 0);
        }
        if let (Some(index), Some(buffer)) = (
            compiled.pixel.bindings.sampler_argument_buffer_index,
            pixel_sampler_buffer.as_ref(),
        ) {
            encoder.set_fragment_buffer(u64::from(index), Some(buffer), 0);
        }

        let primitive = metal_primitive_type(draw.primitive_type)?;
        if matches!(scissor, GuestScissor::Empty) {
            // An empty guest scissor is a valid no-op draw. Metal rejects zero-sized scissors, so
            // retain the render pass and omit the primitive command instead.
        } else if draw.indexed {
            let indices = read_draw_indices(draw).ok_or("index buffer is unreadable")?;
            let index_buffer = device.new_buffer_with_data(
                indices.as_ptr().cast::<c_void>(),
                u64::try_from(indices.len() * 4).map_err(|_| "index byte count overflow")?,
                MTLResourceOptions::StorageModeShared,
            );
            encoder.draw_indexed_primitives_instanced(
                primitive,
                u64::try_from(indices.len()).map_err(|_| "index count overflow")?,
                MTLIndexType::UInt32,
                &index_buffer,
                0,
                u64::from(draw.instance_count),
            );
        } else {
            let vertex_count = metal_nonindexed_vertex_count(draw.primitive_type, draw.index_count);
            encoder.draw_primitives_instanced(
                primitive,
                u64::from(draw.index_offset),
                u64::from(vertex_count),
                u64::from(draw.instance_count),
            );
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(format!(
                "Metal command buffer ended in {:?}",
                command_buffer.status()
            ));
        }
        if trace {
            eprintln!(
                "[REMu][METAL] Gen5 draw={} GPU completed in {:?}",
                draw.sequence,
                started.elapsed()
            );
        }

        write_back_global_buffers(&export_buffers, export_evaluation)?;
        write_back_global_buffers(&pixel_buffers, pixel_evaluation)?;
        write_back_image_textures(&export_textures)?;
        write_back_image_textures(&pixel_textures)?;

        let mut rendered = vec![0_u8; target_bytes.len()];
        target_texture.get_bytes(
            rendered.as_mut_ptr().cast::<c_void>(),
            u64::from(target.width) * 4,
            MTLRegion::new_2d(0, 0, u64::from(target.width), u64::from(target.height)),
            0,
        );
        write_guest_texture_rgba(target, &rendered).map_err(|error| {
            format!("could not write Metal render target back to guest memory: {error}")
        })?;
        self.capture_translated_frame_if_changed(target, &rendered);
        if trace {
            eprintln!(
                "[REMu][METAL] executed Gen5 draw={} es={:#x} ps={:#x} in {:?}",
                draw.sequence,
                key.0,
                key.1,
                started.elapsed()
            );
        }
        Ok(())
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
            return;
        };
        let Some(byte_count) = guest_texture_byte_count(descriptor) else {
            return;
        };
        let Some(output_byte_count) = u64::from(descriptor.width)
            .checked_mul(u64::from(descriptor.height))
            .and_then(|pixels| pixels.checked_mul(4))
        else {
            return;
        };
        if descriptor.address < 0x10_000
            || byte_count > 128 * 1024 * 1024
            || output_byte_count > 128 * 1024 * 1024
        {
            return;
        }
        if self.guest_texture_size != (descriptor.width, descriptor.height) {
            let texture_descriptor = TextureDescriptor::new();
            texture_descriptor.set_texture_type(MTLTextureType::D2);
            texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            texture_descriptor.set_width(u64::from(descriptor.width));
            texture_descriptor.set_height(u64::from(descriptor.height));
            texture_descriptor.set_storage_mode(MTLStorageMode::Shared);
            texture_descriptor.set_usage(MTLTextureUsage::ShaderRead);
            self.guest_texture = Some(device.new_texture(&texture_descriptor));
            self.guest_texture_size = (descriptor.width, descriptor.height);
        }
        let (Some(texture), Ok(source), Ok(source_length), Ok(output_length)) = (
            self.guest_texture.as_ref(),
            usize::try_from(descriptor.address),
            usize::try_from(byte_count),
            usize::try_from(output_byte_count),
        ) else {
            return;
        };
        self.guest_bgra_bytes.resize(output_length, 0);
        // SAFETY: The AGC texture descriptor refers to a live CPU-visible guest allocation. The
        // runtime owns the mapping while the presenter thread is active, and the descriptor was
        // bounded to 128 MiB above before constructing this read-only view.
        let source_bytes =
            unsafe { std::slice::from_raw_parts(source as *const u8, source_length) };
        convert_guest_texture_to_bgra(
            source_bytes,
            descriptor,
            constants,
            &mut self.guest_bgra_bytes,
        );
        texture.replace_region(
            MTLRegion::new_2d(
                0,
                0,
                u64::from(descriptor.width),
                u64::from(descriptor.height),
            ),
            0,
            self.guest_bgra_bytes.as_ptr().cast::<c_void>(),
            u64::from(descriptor.width) * 4,
        );
        if let Some(layer) = self.layer.as_ref() {
            layer.set_drawable_size(CGSize::new(
                f64::from(descriptor.width),
                f64::from(descriptor.height),
            ));
        }
        let layout = (
            descriptor.address,
            descriptor.width,
            descriptor.height,
            descriptor.tile_mode,
        );
        if self.logged_guest_layouts.len() < 8 && self.logged_guest_layouts.insert(layout) {
            log_guest_draw_upload(
                &self.guest_bgra_bytes,
                descriptor,
                draw.color_target_address,
                output_length,
            );
        }
        self.capture_guest_frame_if_changed(descriptor);
    }

    fn capture_guest_frame_if_changed(&mut self, descriptor: GuestTextureDescriptor) {
        let Some(directory) = self.capture_directory.as_deref() else {
            return;
        };
        if self
            .last_guest_capture
            .is_some_and(|last| last.elapsed() < Duration::from_secs(1))
        {
            return;
        }
        let fingerprint = guest_frame_fingerprint(&self.guest_bgra_bytes);
        let captured_frame = (
            descriptor.address,
            descriptor.width,
            descriptor.height,
            descriptor.tile_mode,
            fingerprint,
        );
        if self.captured_guest_frames.len() >= 128
            || !self.captured_guest_frames.insert(captured_frame)
        {
            return;
        }
        self.last_guest_capture = Some(Instant::now());
        match capture_guest_frame(directory, &self.guest_bgra_bytes, descriptor, fingerprint) {
            Ok(path) => eprintln!("[REMu][METAL] captured frame {}", path.display()),
            Err(error) => eprintln!("[REMu][METAL] could not capture frame: {error}"),
        }
    }

    fn capture_translated_frame_if_changed(
        &mut self,
        descriptor: GuestTextureDescriptor,
        rgba: &[u8],
    ) {
        if self.capture_directory.is_none()
            || self
                .last_guest_capture
                .is_some_and(|last| last.elapsed() < Duration::from_secs(1))
        {
            return;
        }
        self.guest_bgra_bytes.resize(rgba.len(), 0);
        for (source, destination) in rgba
            .chunks_exact(4)
            .zip(self.guest_bgra_bytes.chunks_exact_mut(4))
        {
            destination.copy_from_slice(&[source[2], source[1], source[0], source[3]]);
        }
        self.capture_guest_frame_if_changed(descriptor);
    }

    fn update_keyboard_input(&mut self, code: KeyCode, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.pressed_keys.insert(code);
            }
            ElementState::Released => {
                self.pressed_keys.remove(&code);
            }
        }
        self.publish_pad_state();
    }

    fn publish_pad_state(&self) {
        let down = |code| self.pressed_keys.contains(&code);
        let mut buttons = 0;
        for (pressed, button) in [
            (down(KeyCode::ArrowUp), PAD_BUTTON_UP),
            (down(KeyCode::ArrowRight), PAD_BUTTON_RIGHT),
            (down(KeyCode::ArrowDown), PAD_BUTTON_DOWN),
            (down(KeyCode::ArrowLeft), PAD_BUTTON_LEFT),
            (
                down(KeyCode::Enter) || down(KeyCode::NumpadEnter) || down(KeyCode::KeyZ),
                PAD_BUTTON_CROSS,
            ),
            (
                down(KeyCode::Escape) || down(KeyCode::KeyX),
                PAD_BUTTON_CIRCLE,
            ),
            (down(KeyCode::KeyC), PAD_BUTTON_SQUARE),
            (down(KeyCode::KeyV), PAD_BUTTON_TRIANGLE),
            (down(KeyCode::KeyQ), PAD_BUTTON_L1),
            (down(KeyCode::KeyE), PAD_BUTTON_R1),
            (down(KeyCode::KeyR), PAD_BUTTON_L2),
            (down(KeyCode::KeyF), PAD_BUTTON_R2),
            (
                down(KeyCode::Tab) || down(KeyCode::Backspace),
                PAD_BUTTON_OPTIONS,
            ),
        ] {
            if pressed {
                buttons |= button;
            }
        }
        let axis = |negative, positive| match (down(negative), down(positive)) {
            (true, false) => 0,
            (false, true) => 255,
            _ => 128,
        };
        set_host_pad_state(HostPadState {
            buttons,
            left_x: axis(KeyCode::KeyA, KeyCode::KeyD),
            left_y: axis(KeyCode::KeyW, KeyCode::KeyS),
            right_x: axis(KeyCode::KeyJ, KeyCode::KeyL),
            right_y: axis(KeyCode::KeyI, KeyCode::KeyK),
            l2: if down(KeyCode::KeyR) { 255 } else { 0 },
            r2: if down(KeyCode::KeyF) { 255 } else { 0 },
        });
    }

    fn render(&mut self) {
        self.upload_pending_guest_draw();
        if self.guest_texture.is_none() {
            self.upload_pending_frame();
        }
        let (Some(layer), Some(queue)) = (self.layer.as_ref(), self.command_queue.as_ref()) else {
            return;
        };
        let Some(drawable) = layer.next_drawable() else {
            return;
        };
        let command_buffer = queue.new_command_buffer();
        if let Some(texture) = self.guest_texture.as_ref() {
            let encoder = command_buffer.new_blit_command_encoder();
            encoder.copy_from_texture(
                texture,
                0,
                0,
                MTLOrigin::default(),
                MTLSize::new(
                    u64::from(self.guest_texture_size.0),
                    u64::from(self.guest_texture_size.1),
                    1,
                ),
                drawable.texture(),
                0,
                0,
                MTLOrigin::default(),
            );
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

fn decode_scissor(
    draw: &IndexedDraw,
    target_width: u32,
    target_height: u32,
) -> Result<GuestScissor, String> {
    let mut bounds = [0_i64, 0, i64::from(target_width), i64::from(target_height)];
    let window_scissor_tl = register_value(&draw.context_registers, 0x081);
    let enable_window_offset = window_scissor_tl.is_none_or(|value| value & 0x8000_0000 == 0);
    let (window_offset_x, window_offset_y) = if enable_window_offset {
        register_value(&draw.context_registers, 0x080).map_or((0_i64, 0_i64), |value| {
            (
                i64::from(i16::from_ne_bytes(
                    u16::try_from(value & 0xffff)
                        .expect("masked window X offset fits u16")
                        .to_ne_bytes(),
                )),
                i64::from(i16::from_ne_bytes(
                    u16::try_from(value >> 16)
                        .expect("window Y offset fits u16")
                        .to_ne_bytes(),
                )),
            )
        })
    } else {
        (0, 0)
    };
    intersect_scissor_pair(draw, 0x00c, 0x00d, 0, 0, true, &mut bounds);
    intersect_scissor_pair(
        draw,
        0x081,
        0x082,
        window_offset_x,
        window_offset_y,
        false,
        &mut bounds,
    );
    intersect_scissor_pair(
        draw,
        0x090,
        0x091,
        window_offset_x,
        window_offset_y,
        false,
        &mut bounds,
    );
    if register_value(&draw.context_registers, 0x292).is_none_or(|mode| mode & 0x2 != 0) {
        intersect_scissor_pair(draw, 0x094, 0x095, 0, 0, false, &mut bounds);
    }
    bounds[0] = bounds[0].clamp(0, i64::from(target_width));
    bounds[1] = bounds[1].clamp(0, i64::from(target_height));
    bounds[2] = bounds[2].clamp(bounds[0], i64::from(target_width));
    bounds[3] = bounds[3].clamp(bounds[1], i64::from(target_height));
    if bounds == [0, 0, i64::from(target_width), i64::from(target_height)] {
        return Ok(GuestScissor::Full);
    }
    let width = bounds[2] - bounds[0];
    let height = bounds[3] - bounds[1];
    if width == 0 || height == 0 {
        return Ok(GuestScissor::Empty);
    }
    Ok(GuestScissor::Rect(MTLScissorRect {
        x: u64::try_from(bounds[0]).map_err(|_| "scissor X does not fit Metal")?,
        y: u64::try_from(bounds[1]).map_err(|_| "scissor Y does not fit Metal")?,
        width: u64::try_from(width).map_err(|_| "scissor width does not fit Metal")?,
        height: u64::try_from(height).map_err(|_| "scissor height does not fit Metal")?,
    }))
}

#[allow(clippy::too_many_arguments)]
fn intersect_scissor_pair(
    draw: &IndexedDraw,
    top_left_register: u32,
    bottom_right_register: u32,
    offset_x: i64,
    offset_y: i64,
    ignore_all_zero: bool,
    bounds: &mut [i64; 4],
) {
    let (Some(top_left), Some(bottom_right)) = (
        register_value(&draw.context_registers, top_left_register),
        register_value(&draw.context_registers, bottom_right_register),
    ) else {
        return;
    };
    if ignore_all_zero && top_left == 0 && bottom_right == 0 {
        return;
    }
    let pair = [
        i64::from(top_left & 0x7fff) + offset_x,
        i64::from((top_left >> 16) & 0x7fff) + offset_y,
        i64::from(bottom_right & 0x7fff) + offset_x,
        i64::from((bottom_right >> 16) & 0x7fff) + offset_y,
    ];
    bounds[0] = bounds[0].max(pair[0]);
    bounds[1] = bounds[1].max(pair[1]);
    bounds[2] = bounds[2].min(pair[2]);
    bounds[3] = bounds[3].min(pair[3]);
}

fn decode_viewport(
    draw: &IndexedDraw,
    target_width: u32,
    target_height: u32,
    scissor: GuestScissor,
) -> MTLViewport {
    let mut minimum_depth = 0.0_f32;
    let mut maximum_depth = 1.0_f32;
    if let (Some(minimum), Some(maximum)) = (
        register_value(&draw.context_registers, 0x0b4).map(f32::from_bits),
        register_value(&draw.context_registers, 0x0b5).map(f32::from_bits),
    ) && minimum.is_finite()
        && maximum.is_finite()
        && maximum > minimum
    {
        minimum_depth = minimum;
        maximum_depth = maximum;
    }
    let programmed = [0x10f, 0x110, 0x111, 0x112]
        .map(|register| register_value(&draw.context_registers, register).map(f32::from_bits));
    if let [Some(x_scale), Some(x_offset), Some(y_scale), Some(y_offset)] = programmed
        && [x_scale, x_offset, y_scale, y_offset]
            .into_iter()
            .all(f32::is_finite)
        && x_scale > 0.0
        && y_scale != 0.0
    {
        let mut origin_y = y_offset - y_scale;
        let mut height = y_scale * 2.0;
        if height < 0.0 {
            origin_y += height;
            height = -height;
        }
        return MTLViewport {
            originX: f64::from(x_offset - x_scale),
            originY: f64::from(origin_y),
            width: f64::from(x_scale * 2.0),
            height: f64::from(height),
            znear: f64::from(minimum_depth),
            zfar: f64::from(maximum_depth),
        };
    }
    let (x, y, width, height) = match scissor {
        GuestScissor::Rect(rect) => (rect.x, rect.y, rect.width, rect.height),
        GuestScissor::Full | GuestScissor::Empty => {
            (0, 0, u64::from(target_width), u64::from(target_height))
        }
    };
    MTLViewport {
        #[allow(clippy::cast_precision_loss)]
        originX: x as f64,
        #[allow(clippy::cast_precision_loss)]
        originY: y as f64,
        #[allow(clippy::cast_precision_loss)]
        width: width as f64,
        #[allow(clippy::cast_precision_loss)]
        height: height as f64,
        znear: f64::from(minimum_depth),
        zfar: f64::from(maximum_depth),
    }
}

fn snapshot_guest_texture_rgba(descriptor: GuestTextureDescriptor) -> Result<Vec<u8>, String> {
    let byte_count = guest_texture_byte_count(descriptor)
        .ok_or("guest texture byte count overflows the host address space")?;
    if descriptor.address < 0x10_000 || byte_count > 128 * 1024 * 1024 {
        return Err(format!(
            "invalid guest texture range address={:#x} bytes={byte_count}",
            descriptor.address
        ));
    }
    let source_address = usize::try_from(descriptor.address)
        .map_err(|_| "guest texture address does not fit usize")?;
    let source_length =
        usize::try_from(byte_count).map_err(|_| "guest texture byte count does not fit usize")?;
    let output_length = u64::from(descriptor.width)
        .checked_mul(u64::from(descriptor.height))
        .and_then(|pixels| pixels.checked_mul(4))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or("guest texture dimensions overflow")?;
    // SAFETY: The resource descriptor points into a live guest direct-memory mapping. The decoded
    // dimensions, pitch, and supported tiling modes bound the view to at most 128 MiB.
    let source = unsafe { std::slice::from_raw_parts(source_address as *const u8, source_length) };
    let width =
        usize::try_from(descriptor.width).map_err(|_| "texture width does not fit usize")?;
    let height =
        usize::try_from(descriptor.height).map_err(|_| "texture height does not fit usize")?;
    let mut output = vec![0_u8; output_length];
    if descriptor.tile_mode == 0 {
        let source_row_bytes = usize::try_from(descriptor.pitch)
            .map_err(|_| "texture pitch does not fit usize")?
            .checked_mul(4)
            .ok_or("source row byte count overflows")?;
        let output_row_bytes = width
            .checked_mul(4)
            .ok_or("output row byte count overflows")?;
        for y in 0..height {
            let source_offset = y
                .checked_mul(source_row_bytes)
                .ok_or("source row offset overflows")?;
            let destination_offset = y
                .checked_mul(output_row_bytes)
                .ok_or("destination row offset overflows")?;
            output[destination_offset..destination_offset + output_row_bytes].copy_from_slice(
                source
                    .get(source_offset..source_offset + output_row_bytes)
                    .ok_or("guest texture row exceeds its decoded allocation")?,
            );
        }
    } else {
        copy_mode27_to_linear(source, descriptor, &mut output)?;
    }
    Ok(output)
}

fn write_guest_texture_rgba(descriptor: GuestTextureDescriptor, rgba: &[u8]) -> Result<(), String> {
    let byte_count = guest_texture_byte_count(descriptor)
        .ok_or("guest texture byte count overflows the host address space")?;
    if descriptor.address < 0x10_000 || byte_count > 128 * 1024 * 1024 {
        return Err(format!(
            "invalid guest texture range address={:#x} bytes={byte_count}",
            descriptor.address
        ));
    }
    let width =
        usize::try_from(descriptor.width).map_err(|_| "texture width does not fit usize")?;
    let height =
        usize::try_from(descriptor.height).map_err(|_| "texture height does not fit usize")?;
    let required = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or("guest texture dimensions overflow")?;
    if rgba.len() != required {
        return Err(format!(
            "tight RGBA image has {} bytes, expected {required}",
            rgba.len()
        ));
    }
    let destination_address = usize::try_from(descriptor.address)
        .map_err(|_| "guest texture address does not fit usize")?;
    let destination_length =
        usize::try_from(byte_count).map_err(|_| "guest texture byte count does not fit usize")?;
    // SAFETY: The resource descriptor points into a live writable guest direct-memory mapping.
    // The decoded dimensions, pitch, and tiling mode bound this mutable view to 128 MiB.
    let destination = unsafe {
        std::slice::from_raw_parts_mut(destination_address as *mut u8, destination_length)
    };
    if descriptor.tile_mode == 0 {
        let destination_row_bytes = usize::try_from(descriptor.pitch)
            .map_err(|_| "texture pitch does not fit usize")?
            .checked_mul(4)
            .ok_or("destination row byte count overflows")?;
        let source_row_bytes = width
            .checked_mul(4)
            .ok_or("source row byte count overflows")?;
        for y in 0..height {
            let source_offset = y
                .checked_mul(source_row_bytes)
                .ok_or("source row offset overflows")?;
            let destination_offset = y
                .checked_mul(destination_row_bytes)
                .ok_or("destination row offset overflows")?;
            destination
                .get_mut(destination_offset..destination_offset + source_row_bytes)
                .ok_or("guest texture row exceeds its decoded allocation")?
                .copy_from_slice(&rgba[source_offset..source_offset + source_row_bytes]);
        }
    } else {
        copy_linear_to_mode27(rgba, descriptor, destination)?;
    }
    Ok(())
}

fn create_rgba_texture(
    device: &Device,
    width: u32,
    height: u32,
    usage: MTLTextureUsage,
    rgba: &[u8],
) -> Result<Texture, String> {
    let required = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(4))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or("Metal texture dimensions overflow")?;
    if width == 0 || height == 0 || rgba.len() != required {
        return Err(format!(
            "invalid RGBA upload {}x{} with {} bytes (expected {required})",
            width,
            height,
            rgba.len()
        ));
    }
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    descriptor.set_width(u64::from(width));
    descriptor.set_height(u64::from(height));
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(usage);
    let texture = device.new_texture(&descriptor);
    texture.replace_region(
        MTLRegion::new_2d(0, 0, u64::from(width), u64::from(height)),
        0,
        rgba.as_ptr().cast::<c_void>(),
        u64::from(width) * 4,
    );
    Ok(texture)
}

fn upload_global_buffers(
    device: &Device,
    evaluation: &ShaderEvaluation,
) -> Result<Vec<Buffer>, String> {
    evaluation
        .global_memory_bindings
        .iter()
        .map(|binding| {
            let empty = [0_u8];
            let bytes = if binding.data.is_empty() {
                empty.as_slice()
            } else {
                binding.data.as_slice()
            };
            let length = u64::try_from(bytes.len())
                .map_err(|_| "global buffer byte count does not fit Metal")?;
            Ok(device.new_buffer_with_data(
                bytes.as_ptr().cast::<c_void>(),
                length,
                MTLResourceOptions::StorageModeShared,
            ))
        })
        .collect()
}

fn write_back_global_buffers(
    buffers: &[Buffer],
    evaluation: &ShaderEvaluation,
) -> Result<(), String> {
    if buffers.len() != evaluation.global_memory_bindings.len() {
        return Err("Metal/global buffer binding count mismatch".to_owned());
    }
    for (buffer, binding) in buffers
        .iter()
        .zip(&evaluation.global_memory_bindings)
        .filter(|(_, binding)| binding.writable && binding.write_back_to_guest)
    {
        if binding.base_address < 0x10_000 || binding.data.len() > 128 * 1024 * 1024 {
            return Err(format!(
                "invalid writable global buffer address={:#x} bytes={}",
                binding.base_address,
                binding.data.len()
            ));
        }
        let destination = usize::try_from(binding.base_address)
            .map_err(|_| "writable global buffer address does not fit usize")?;
        // SAFETY: Shader evaluation retained this live writable guest binding, whose captured data
        // length is the exact range represented by the corresponding Metal buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer.contents().cast::<u8>(),
                destination as *mut u8,
                binding.data.len(),
            );
        }
    }
    Ok(())
}

fn decode_image_descriptor(words: &[u32]) -> Result<DecodedImageDescriptor, String> {
    if words.len() < 4 {
        return Err(format!(
            "image resource descriptor has {} dwords, expected at least four",
            words.len()
        ));
    }
    let address = ((u64::from(words[1] & 0xff) << 32) | u64::from(words[0])) << 8;
    let width = ((words[1] >> 30) | ((words[2] & 0x3fff) << 2))
        .checked_add(1)
        .ok_or("image width overflows")?;
    let height = ((words[2] >> 14) & 0xffff)
        .checked_add(1)
        .ok_or("image height overflows")?;
    let unified_format = (words[1] >> 20) & 0x1ff;
    let (data_format, number_format) = remu_shader::decode_unified_format(unified_format)
        .ok_or_else(|| format!("unknown GFX10 unified image format {unified_format}"))?;
    let tile_mode = (words[3] >> 20) & 0x1f;
    let texture_type = (words[3] >> 28) & 0xf;
    let base_level = (words[3] >> 12) & 0xf;
    let last_level = (words[3] >> 16) & 0xf;
    let destination_select = words[3] & 0xfff;
    let descriptor_pitch =
        if matches!(texture_type, 8 | 9 | 14) && words.get(4).copied().unwrap_or_default() != 0 {
            (words[4] & 0x3fff)
                .checked_add(1)
                .ok_or("image pitch overflows")?
        } else {
            width
        };
    let pitch = if tile_mode == 0 {
        effective_linear_rgba_pitch(width.max(descriptor_pitch))
            .ok_or("linear image pitch overflows")?
    } else {
        descriptor_pitch
    };
    if address < 0x10_000 || width > 8_192 || height > 8_192 || pitch < width || pitch > 8_192 {
        return Err(format!(
            "invalid image range address={address:#x} extent={width}x{height} pitch={pitch}"
        ));
    }
    if (data_format, number_format) != (10, 0) {
        return Err(format!(
            "Metal image upload does not yet implement GFX10 format {data_format}/{number_format} (unified {unified_format})"
        ));
    }
    if texture_type != 9 {
        return Err(format!(
            "Metal image upload does not yet implement texture type {texture_type}"
        ));
    }
    if base_level != 0 || last_level != 0 {
        return Err(format!(
            "Metal image upload does not yet implement mip range {base_level}..={last_level}"
        ));
    }
    if !matches!(tile_mode, 0 | 27) {
        return Err(format!(
            "Metal image upload does not yet implement tile mode {tile_mode}"
        ));
    }
    Ok(DecodedImageDescriptor {
        guest: GuestTextureDescriptor {
            address,
            width,
            height,
            pitch,
            tile_mode,
        },
        destination_select,
    })
}

fn apply_image_destination_select(rgba: &mut [u8], destination_select: u32) -> Result<(), String> {
    if destination_select == 0 || destination_select == 0xfac {
        return Ok(());
    }
    let selectors =
        std::array::from_fn::<_, 4, _>(|channel| (destination_select >> (channel * 3)) & 0x7);
    if selectors
        .iter()
        .any(|selector| !matches!(selector, 0 | 1 | 4..=7))
    {
        return Err(format!(
            "invalid image destination select {destination_select:#x}"
        ));
    }
    for pixel in rgba.chunks_exact_mut(4) {
        let source: [u8; 4] = pixel.try_into().expect("RGBA pixel has four channels");
        for (channel, selector) in selectors.iter().copied().enumerate() {
            pixel[channel] = match selector {
                0 => 0,
                1 => 255,
                4..=7 => source[usize::try_from(selector - 4).expect("channel fits usize")],
                _ => unreachable!("validated destination selector"),
            };
        }
    }
    Ok(())
}

fn upload_image_textures(
    device: &Device,
    evaluation: &ShaderEvaluation,
) -> Result<Vec<UploadedImageTexture>, String> {
    evaluation
        .image_bindings
        .iter()
        .map(|binding| {
            let descriptor = decode_image_descriptor(&binding.resource_descriptor)
                .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
            let writable = binding.opcode.starts_with("ImageStore")
                || binding.opcode.starts_with("ImageAtomic");
            if writable && !matches!(descriptor.destination_select, 0 | 0xfac) {
                return Err(format!(
                    "writable image at PC {:#x} has non-identity destination select {:#x}",
                    binding.pc, descriptor.destination_select
                ));
            }
            let mut rgba = snapshot_guest_texture_rgba(descriptor.guest)
                .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
            if !writable {
                apply_image_destination_select(&mut rgba, descriptor.destination_select)
                    .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
            }
            let usage = if writable {
                MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite
            } else {
                MTLTextureUsage::ShaderRead
            };
            let texture = create_rgba_texture(
                device,
                descriptor.guest.width,
                descriptor.guest.height,
                usage,
                &rgba,
            )?;
            Ok(UploadedImageTexture {
                texture,
                descriptor,
                writable,
            })
        })
        .collect()
}

fn write_back_image_textures(textures: &[UploadedImageTexture]) -> Result<(), String> {
    for uploaded in textures.iter().filter(|texture| texture.writable) {
        let descriptor = uploaded.descriptor.guest;
        let length = u64::from(descriptor.width)
            .checked_mul(u64::from(descriptor.height))
            .and_then(|pixels| pixels.checked_mul(4))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or("writable image dimensions overflow")?;
        let mut rgba = vec![0_u8; length];
        uploaded.texture.get_bytes(
            rgba.as_mut_ptr().cast::<c_void>(),
            u64::from(descriptor.width) * 4,
            MTLRegion::new_2d(
                0,
                0,
                u64::from(descriptor.width),
                u64::from(descriptor.height),
            ),
            0,
        );
        write_guest_texture_rgba(descriptor, &rgba)?;
    }
    Ok(())
}

fn sampler_address_mode(value: u32) -> MTLSamplerAddressMode {
    match value {
        0 => MTLSamplerAddressMode::Repeat,
        1 => MTLSamplerAddressMode::MirrorRepeat,
        2 => MTLSamplerAddressMode::ClampToEdge,
        3 | 5 | 7 => MTLSamplerAddressMode::MirrorClampToEdge,
        4 | 6 => MTLSamplerAddressMode::ClampToBorderColor,
        _ => unreachable!("three-bit sampler address mode"),
    }
}

fn sampler_min_mag_filter(value: u32) -> MTLSamplerMinMagFilter {
    if matches!(value, 1 | 3) {
        MTLSamplerMinMagFilter::Linear
    } else {
        MTLSamplerMinMagFilter::Nearest
    }
}

fn sampler_mip_filter(value: u32) -> MTLSamplerMipFilter {
    match value {
        0 => MTLSamplerMipFilter::NotMipmapped,
        1 => MTLSamplerMipFilter::Nearest,
        2 | 3 => MTLSamplerMipFilter::Linear,
        _ => unreachable!("two-bit sampler mip filter"),
    }
}

fn sampled_image_bindings(evaluation: &ShaderEvaluation) -> impl Iterator<Item = &ImageBinding> {
    evaluation.image_bindings.iter().filter(|binding| {
        binding.opcode.starts_with("ImageSample") || binding.opcode.starts_with("ImageGather")
    })
}

fn create_sampler_argument_buffer(
    device: &Device,
    function: &Function,
    buffer_index: Option<u32>,
    evaluation: &ShaderEvaluation,
) -> Result<(Option<Buffer>, Vec<SamplerState>), String> {
    let sampled_bindings: Vec<_> = sampled_image_bindings(evaluation).collect();
    let Some(buffer_index) = buffer_index else {
        if sampled_bindings.is_empty() {
            return Ok((None, Vec::new()));
        }
        return Err("Metal shader has sampled images but no sampler argument buffer".to_owned());
    };
    if sampled_bindings.is_empty() {
        return Err("Metal shader declares a sampler argument buffer without samplers".to_owned());
    }
    let argument_encoder = function.new_argument_encoder(u64::from(buffer_index));
    let argument_buffer = device.new_buffer(
        argument_encoder.encoded_length(),
        MTLResourceOptions::StorageModeShared,
    );
    argument_encoder.set_argument_buffer(&argument_buffer, 0);
    let mut samplers = Vec::with_capacity(sampled_bindings.len());
    for (slot, binding) in sampled_bindings.iter().enumerate() {
        if binding.sampler_descriptor.len() < 4 {
            return Err(format!(
                "sampler at PC {:#x} has {} dwords, expected four",
                binding.pc,
                binding.sampler_descriptor.len()
            ));
        }
        let words = &binding.sampler_descriptor;
        let lod_bias = words[2] & 0x3fff;
        if lod_bias != 0 {
            return Err(format!(
                "sampler at PC {:#x} has unimplemented LOD bias {lod_bias:#x}",
                binding.pc
            ));
        }
        let descriptor = SamplerDescriptor::new();
        descriptor.set_address_mode_s(sampler_address_mode(words[0] & 0x7));
        descriptor.set_address_mode_t(sampler_address_mode((words[0] >> 3) & 0x7));
        descriptor.set_address_mode_r(sampler_address_mode((words[0] >> 6) & 0x7));
        descriptor.set_max_anisotropy(u64::from(1_u32 << ((words[0] >> 9) & 0x7).min(4)));
        descriptor.set_mag_filter(sampler_min_mag_filter((words[2] >> 20) & 0x3));
        descriptor.set_min_filter(sampler_min_mag_filter((words[2] >> 22) & 0x3));
        descriptor.set_mip_filter(sampler_mip_filter((words[2] >> 26) & 0x3));
        descriptor.set_lod_min_clamp(
            f32::from(u16::try_from(words[1] & 0xfff).expect("12-bit LOD fits u16")) / 256.0,
        );
        descriptor.set_lod_max_clamp(
            f32::from(u16::try_from((words[1] >> 12) & 0xfff).expect("12-bit LOD fits u16"))
                / 256.0,
        );
        descriptor.set_support_argument_buffers(true);
        let sampler = device.new_sampler(&descriptor);
        argument_encoder.set_sampler_state(
            u64::try_from(slot).map_err(|_| "sampler slot does not fit Metal")?,
            &sampler,
        );
        samplers.push(sampler);
    }
    Ok((Some(argument_buffer), samplers))
}

fn metal_primitive_type(guest: u32) -> Result<MTLPrimitiveType, String> {
    match guest {
        1 => Ok(MTLPrimitiveType::Point),
        2 => Ok(MTLPrimitiveType::Line),
        3 => Ok(MTLPrimitiveType::LineStrip),
        4 => Ok(MTLPrimitiveType::Triangle),
        6 | 0x11 => Ok(MTLPrimitiveType::TriangleStrip),
        5 => Err("triangle-fan topology requires explicit Metal index expansion".to_owned()),
        _ => Err(format!("unknown guest primitive topology {guest:#x}")),
    }
}

const fn metal_nonindexed_vertex_count(guest_primitive_type: u32, guest_count: u32) -> u32 {
    if guest_primitive_type == 0x11 {
        4
    } else {
        guest_count
    }
}

fn log_guest_draw_upload(
    bytes: &[u8],
    descriptor: GuestTextureDescriptor,
    target_address: u64,
    output_length: usize,
) {
    let nonzero_bytes = bytes.iter().filter(|&&value| value != 0).count();
    let minimum = bytes.iter().copied().min().unwrap_or(0);
    let maximum = bytes.iter().copied().max().unwrap_or(0);
    let fingerprint = guest_frame_fingerprint(bytes);
    eprintln!(
        "[REMu][METAL] fast-presented Dreaming Sarah source={:#x} {}x{} tile={} target={target_address:#x} nonzero={nonzero_bytes}/{output_length} range={minimum}..={maximum} fingerprint={fingerprint:016x}",
        descriptor.address, descriptor.width, descriptor.height, descriptor.tile_mode,
    );
}

fn guest_frame_fingerprint(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, &value| {
            (hash ^ u64::from(value)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

fn capture_guest_frame(
    directory: &Path,
    bgra: &[u8],
    descriptor: GuestTextureDescriptor,
    fingerprint: u64,
) -> Result<PathBuf, String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
    let path = directory.join(format!(
        "remu-{:x}-{}x{}-tile{}-{fingerprint:016x}.ppm",
        descriptor.address, descriptor.width, descriptor.height, descriptor.tile_mode
    ));
    let file = std::fs::File::create(&path)
        .map_err(|error| format!("could not create {}: {error}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "P6\n{} {}\n255",
        descriptor.width, descriptor.height
    )
    .map_err(|error| format!("could not write {}: {error}", path.display()))?;
    let pixel_count = u64::from(descriptor.width)
        .checked_mul(u64::from(descriptor.height))
        .and_then(|pixels| usize::try_from(pixels).ok())
        .ok_or("captured frame dimensions overflow")?;
    if bgra.len() < pixel_count.saturating_mul(4) {
        return Err("captured frame buffer is truncated".to_owned());
    }
    let mut rgb = Vec::with_capacity(pixel_count.saturating_mul(3));
    for pixel in bgra.chunks_exact(4).take(pixel_count) {
        rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
    }
    writer
        .write_all(&rgb)
        .map_err(|error| format!("could not write {}: {error}", path.display()))?;
    Ok(path)
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

fn is_dreaming_presenter_draw(draw: &IndexedDraw) -> bool {
    draw.indexed
        && shader_matches_prefix(&draw.export_shader_prefix, &DREAMING_PRESENTER_ES_PREFIX)
        && shader_matches_prefix(&draw.pixel_shader_prefix, &DREAMING_PRESENTER_PS_PREFIX)
}

fn is_dreaming_fill_draw(draw: &IndexedDraw) -> bool {
    !draw.indexed
        && draw.index_count == 3
        && shader_matches_prefix(&draw.export_shader_prefix, &DREAMING_FILL_ES_PREFIX)
        && shader_matches_prefix(&draw.pixel_shader_prefix, &DREAMING_FILL_PS_PREFIX)
}

fn pixel_scalars(draw: &IndexedDraw) -> [u32; 4] {
    std::array::from_fn(|index| {
        register_value(
            &draw.shader_registers,
            0x00c + u32::try_from(index).expect("scalar index fits"),
        )
        .unwrap_or_default()
    })
}

fn decode_render_target(draw: &IndexedDraw) -> Option<GuestTextureDescriptor> {
    let info = register_value(&draw.context_registers, 0x31c)?;
    let attrib2 = register_value(&draw.context_registers, 0x3b0)?;
    let attrib3 = register_value(&draw.context_registers, 0x3b8)?;
    let target_mask = register_value(&draw.context_registers, 0x08e).unwrap_or(0xf);
    let width = ((attrib2 >> 14) & 0x3fff).checked_add(1)?;
    let height = (attrib2 & 0x3fff).checked_add(1)?;
    let format = (info >> 2) & 0x1f;
    let number_type = (info >> 8) & 0x7;
    let tile_mode = (attrib3 >> 14) & 0x1f;
    if draw.color_target_address < 0x10_000
        || width > 8_192
        || height > 8_192
        || format != 10
        || number_type != 0
        || !matches!(tile_mode, 0 | 27)
        || target_mask.trailing_zeros() >= 4
    {
        return None;
    }
    Some(GuestTextureDescriptor {
        address: draw.color_target_address,
        width,
        height,
        pitch: width,
        tile_mode,
    })
}

fn dreaming_software_draw_kind(draw: &IndexedDraw) -> Option<SoftwareDrawKind> {
    if !draw.indexed
        || draw.index_count == 0
        || !shader_matches_prefix(&draw.export_shader_prefix, &DREAMING_PRESENTER_ES_PREFIX)
    {
        return None;
    }
    if shader_matches_prefix(&draw.pixel_shader_prefix, &DREAMING_SOLID_PS_PREFIX) {
        Some(SoftwareDrawKind::Solid)
    } else if shader_matches_prefix(&draw.pixel_shader_prefix, &DREAMING_TEXTURE_PS_PREFIX) {
        Some(SoftwareDrawKind::Texture)
    } else if shader_matches_prefix(&draw.pixel_shader_prefix, &DREAMING_COMPOSITOR_PS_PREFIX) {
        Some(SoftwareDrawKind::Compositor)
    } else {
        None
    }
}

fn execute_dreaming_software_draw(draw: &IndexedDraw) -> Option<ExecutedSoftwareDraw> {
    let kind = dreaming_software_draw_kind(draw)?;
    let target = decode_render_target(draw)?;
    let source = match kind {
        SoftwareDrawKind::Solid => {
            execute_dreaming_solid_draw(draw, target)?;
            0
        }
        SoftwareDrawKind::Texture => {
            let source = decode_guest_texture(draw, 0x00c)?;
            if software_rasterize_sprite(draw, source, target).is_none() {
                software_blit_guest_texture(source, target, true)?;
            }
            source.address
        }
        SoftwareDrawKind::Compositor => {
            let source = decode_guest_texture(draw, 0x00c)?;
            software_blit_guest_texture(source, target, false)?;
            source.address
        }
    };
    Some(ExecutedSoftwareDraw {
        kind,
        source,
        target: target.address,
        width: target.width,
        height: target.height,
    })
}

fn execute_dreaming_solid_draw(draw: &IndexedDraw, target: GuestTextureDescriptor) -> Option<()> {
    let descriptor = pixel_scalars(draw);
    let color_address = (u64::from(descriptor[1] & 0xffff) << 32) | u64::from(descriptor[0]);
    if color_address < 0x10_000 {
        return None;
    }
    let color_address = usize::try_from(color_address).ok()?;
    let color = std::array::from_fn(|index| {
        // SAFETY: The observed PS buffer descriptor identifies a live 16-byte uniform allocation.
        unsafe { std::ptr::read_unaligned((color_address + index * 4) as *const f32) }
    });
    if !color.iter().all(|value| value.is_finite()) {
        return None;
    }
    let rgba = color.map(|value| transform_color_channel(255, value, 0.0));
    fill_guest_texture(target, rgba)
}

fn fill_guest_texture(descriptor: GuestTextureDescriptor, rgba: [u8; 4]) -> Option<()> {
    let length = usize::try_from(guest_texture_byte_count(descriptor)?).ok()?;
    if length > 128 * 1024 * 1024 {
        return None;
    }
    let address = usize::try_from(descriptor.address).ok()?;
    // SAFETY: The validated guest render-target descriptor bounds this live direct-memory write.
    let target = unsafe { std::slice::from_raw_parts_mut(address as *mut u8, length) };
    for pixel in target.chunks_exact_mut(4) {
        pixel.copy_from_slice(&rgba);
    }
    Some(())
}

fn software_rasterize_sprite(
    draw: &IndexedDraw,
    source: GuestTextureDescriptor,
    target: GuestTextureDescriptor,
) -> Option<()> {
    let vertices = decode_sprite_vertices(draw, target)?;
    if vertices.len() < 3 || !matches!(source.tile_mode, 0 | 27) {
        return None;
    }
    let source_length = usize::try_from(guest_texture_byte_count(source)?).ok()?;
    let target_length = usize::try_from(guest_texture_byte_count(target)?).ok()?;
    if source_length > 128 * 1024 * 1024 || target_length > 128 * 1024 * 1024 {
        return None;
    }
    let source_address = usize::try_from(source.address).ok()?;
    let target_address = usize::try_from(target.address).ok()?;
    let source_end = source_address.checked_add(source_length)?;
    let target_end = target_address.checked_add(target_length)?;
    if source_address < target_end && target_address < source_end {
        return None;
    }
    // SAFETY: The decoded, bounded texture and target descriptors identify non-overlapping live
    // guest allocations for the duration of this synchronous compatibility draw.
    let source_bytes =
        unsafe { std::slice::from_raw_parts(source_address as *const u8, source_length) };
    let target_bytes =
        unsafe { std::slice::from_raw_parts_mut(target_address as *mut u8, target_length) };
    capture_sampled_texture_if_requested(source_bytes, source);
    rasterize_sprite_triangles(source_bytes, source, target_bytes, target, &vertices);
    Some(())
}

fn capture_sampled_texture_if_requested(source: &[u8], descriptor: GuestTextureDescriptor) {
    static CAPTURE_COUNT: AtomicUsize = AtomicUsize::new(0);
    let Some(directory) = std::env::var_os("REMU_CAPTURE_TEXTURE_DIR").map(PathBuf::from) else {
        return;
    };
    if CAPTURE_COUNT.fetch_add(1, Ordering::Relaxed) >= 24 {
        return;
    }
    let Some(pixel_count) = u64::from(descriptor.width)
        .checked_mul(u64::from(descriptor.height))
        .and_then(|pixels| usize::try_from(pixels).ok())
    else {
        return;
    };
    let mut bgra = vec![0_u8; pixel_count.saturating_mul(4)];
    let width = usize::try_from(descriptor.width).expect("validated texture width");
    let height = usize::try_from(descriptor.height).expect("validated texture height");
    for y in 0..height {
        for x in 0..width {
            let source_offset = guest_pixel_byte_offset(descriptor, x, y);
            let target_offset = (y * width + x) * 4;
            let pixel = &source[source_offset..source_offset + 4];
            bgra[target_offset..target_offset + 4]
                .copy_from_slice(&[pixel[2], pixel[1], pixel[0], pixel[3]]);
        }
    }
    let fingerprint = guest_frame_fingerprint(&bgra);
    match capture_guest_frame(&directory, &bgra, descriptor, fingerprint) {
        Ok(path) => eprintln!("[REMu][METAL] captured sampled texture {}", path.display()),
        Err(error) => eprintln!("[REMu][METAL] could not capture sampled texture: {error}"),
    }
}

fn decode_sprite_vertices(
    draw: &IndexedDraw,
    target: GuestTextureDescriptor,
) -> Option<Vec<SpriteVertex>> {
    let descriptor_table = shader_buffer_address(draw, 0x090)?;
    let attribute_table = shader_pointer_address(draw, 0x092)?;
    let matrix_address = shader_buffer_address(draw, 0x08c)?;
    let matrices: [f32; 32] = std::array::from_fn(|index| {
        read_address_f32(matrix_address + u64::try_from(index).unwrap_or(0) * 4)
    });
    if !matrices.iter().all(|value| value.is_finite()) {
        return None;
    }
    let attribute_entries: [u32; 3] = std::array::from_fn(|attribute| {
        read_address_u32(attribute_table + u64::try_from(attribute).unwrap_or(0) * 4)
    });
    let attribute_descriptors: [u64; 3] = std::array::from_fn(|attribute| {
        let entry = attribute_entries[attribute];
        let descriptor_address = descriptor_table + u64::from((entry << 4) & 0x1f0);
        let low = read_address_u32(descriptor_address);
        let high = read_address_u32(descriptor_address + 4);
        (u64::from(high & 0xffff) << 32) | u64::from(low)
    });
    if attribute_descriptors
        .iter()
        .any(|&address| address < 0x10_000)
    {
        return None;
    }
    let indices = read_draw_indices(draw)?;
    let mut vertices = Vec::with_capacity(indices.len());
    for index in indices {
        let byte_offset = u64::from(index).checked_mul(24)?;
        let position_address = attribute_descriptors[0].checked_add(byte_offset)?;
        let opacity_address = attribute_descriptors[1].checked_add(byte_offset)?;
        let uv_address = attribute_descriptors[2].checked_add(byte_offset)?;
        let position = [
            read_address_f32(position_address),
            read_address_f32(position_address + 4),
            read_address_f32(position_address + 8),
        ];
        let vertex = SpriteVertex {
            position: transform_sprite_position(position, &matrices, target)?,
            uv: [
                read_address_f32(uv_address),
                read_address_f32(uv_address + 4),
            ],
            opacity: read_address_f32(opacity_address),
        };
        if !vertex
            .position
            .into_iter()
            .chain(vertex.uv)
            .chain([vertex.opacity])
            .all(f32::is_finite)
        {
            return None;
        }
        vertices.push(vertex);
    }
    trace_sprite_vertices(
        draw,
        descriptor_table,
        attribute_table,
        attribute_entries,
        attribute_descriptors,
        &vertices,
    );
    Some(vertices)
}

fn transform_sprite_position(
    position: [f32; 3],
    matrices: &[f32; 32],
    target: GuestTextureDescriptor,
) -> Option<[f32; 2]> {
    if !position.iter().all(|value| value.is_finite()) {
        return None;
    }
    let model = transform_column_major(
        [position[0], position[1], position[2], 1.0],
        matrices[16..].try_into().ok()?,
    );
    let clip = transform_column_major(model, matrices[..16].try_into().ok()?);
    if !clip.iter().all(|value| value.is_finite()) || clip[3].abs() < f32::EPSILON {
        return None;
    }
    let width = f32::from(u16::try_from(target.width).ok()?);
    let height = f32::from(u16::try_from(target.height).ok()?);
    Some([
        (clip[0] / clip[3] + 1.0) * width * 0.5,
        (1.0 - clip[1] / clip[3]) * height * 0.5,
    ])
}

fn transform_column_major(vector: [f32; 4], matrix: &[f32; 16]) -> [f32; 4] {
    std::array::from_fn(|row| {
        vector[0] * matrix[row]
            + vector[1] * matrix[4 + row]
            + vector[2] * matrix[8 + row]
            + vector[3] * matrix[12 + row]
    })
}

fn trace_sprite_vertices(
    draw: &IndexedDraw,
    descriptor_table: u64,
    attribute_table: u64,
    attribute_entries: [u32; 3],
    attribute_descriptors: [u64; 3],
    vertices: &[SpriteVertex],
) {
    static TRACE_COUNT: AtomicUsize = AtomicUsize::new(0);
    if std::env::var_os("REMU_TRACE_SPRITES").is_none()
        || TRACE_COUNT.fetch_add(1, Ordering::Relaxed) >= 12
    {
        return;
    }
    let matrix_address = shader_buffer_address(draw, 0x08c).unwrap_or_default();
    let matrices: [f32; 32] = if matrix_address >= 0x10_000 {
        std::array::from_fn(|index| {
            read_address_f32(matrix_address + u64::try_from(index).unwrap_or(0) * 4)
        })
    } else {
        [0.0; 32]
    };
    eprintln!(
        "[REMu][METAL][SPRITE] descriptors={descriptor_table:#x}/{attribute_table:#x} entries={attribute_entries:?} bases={attribute_descriptors:x?} indices={} vertices={vertices:?} matrix={matrix_address:#x}/{matrices:?} viewport=[{:08x}, {:08x}, {:08x}, {:08x}]",
        vertices.len(),
        register_value(&draw.context_registers, 0x10f).unwrap_or_default(),
        register_value(&draw.context_registers, 0x110).unwrap_or_default(),
        register_value(&draw.context_registers, 0x111).unwrap_or_default(),
        register_value(&draw.context_registers, 0x112).unwrap_or_default(),
    );
}

fn shader_buffer_address(draw: &IndexedDraw, register_base: u32) -> Option<u64> {
    let low = register_value(&draw.shader_registers, register_base)?;
    let high = register_value(&draw.shader_registers, register_base + 1)?;
    let address = (u64::from(high & 0xffff) << 32) | u64::from(low);
    (address >= 0x10_000).then_some(address)
}

fn shader_pointer_address(draw: &IndexedDraw, register_base: u32) -> Option<u64> {
    let low = register_value(&draw.shader_registers, register_base)?;
    let high = register_value(&draw.shader_registers, register_base + 1)?;
    let address = (u64::from(high) << 32) | u64::from(low);
    (address >= 0x10_000).then_some(address)
}

fn read_draw_indices(draw: &IndexedDraw) -> Option<Vec<u32>> {
    if draw.index_buffer_address < 0x10_000 || draw.index_count > 65_536 {
        return None;
    }
    // AGC encodes zero as 16-bit and every observed nonzero index type as 32-bit. This matches
    // SharpEmu's `CreateGuestIndexBuffer` handling and, importantly, is not a byte-size enum.
    let element_size = if draw.index_size == 0 { 2_u64 } else { 4 };
    let first = u64::from(draw.index_offset).checked_mul(element_size)?;
    let base = draw.index_buffer_address.checked_add(first)?;
    Some(
        (0..draw.index_count)
            .map(|index| {
                let address = base + u64::from(index) * element_size;
                let address = usize::try_from(address).expect("validated index address");
                // SAFETY: The draw snapshot retains a live index allocation containing the
                // bounded index range used by this synchronous software draw.
                unsafe {
                    match element_size {
                        1 => u32::from(std::ptr::read_unaligned(address as *const u8)),
                        2 => u32::from(std::ptr::read_unaligned(address as *const u16)),
                        _ => std::ptr::read_unaligned(address as *const u32),
                    }
                }
            })
            .collect(),
    )
}

fn read_address_u32(address: u64) -> u32 {
    let address = usize::try_from(address).expect("validated guest address");
    // SAFETY: Callers derive this address from a live, validated guest resource descriptor.
    unsafe { std::ptr::read_unaligned(address as *const u32) }
}

fn read_address_f32(address: u64) -> f32 {
    f32::from_bits(read_address_u32(address))
}

fn rasterize_sprite_triangles(
    source_bytes: &[u8],
    source: GuestTextureDescriptor,
    target_bytes: &mut [u8],
    target: GuestTextureDescriptor,
    vertices: &[SpriteVertex],
) {
    for triangle in vertices.chunks_exact(3) {
        rasterize_sprite_triangle(source_bytes, source, target_bytes, target, triangle);
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn rasterize_sprite_triangle(
    source_bytes: &[u8],
    source: GuestTextureDescriptor,
    target_bytes: &mut [u8],
    target: GuestTextureDescriptor,
    vertices: &[SpriteVertex],
) {
    let edge = |a: [f32; 2], b: [f32; 2], point: [f32; 2]| {
        (point[0] - a[0]) * (b[1] - a[1]) - (point[1] - a[1]) * (b[0] - a[0])
    };
    let area = edge(
        vertices[1].position,
        vertices[2].position,
        vertices[0].position,
    );
    if area.abs() < f32::EPSILON {
        return;
    }
    let (Ok(target_width), Ok(target_height)) =
        (i32::try_from(target.width), i32::try_from(target.height))
    else {
        return;
    };
    let minimum_x = vertices
        .iter()
        .map(|vertex| vertex.position[0])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as i32;
    let maximum_x = vertices
        .iter()
        .map(|vertex| vertex.position[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(target_width as f32) as i32;
    let minimum_y = vertices
        .iter()
        .map(|vertex| vertex.position[1])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .max(0.0) as i32;
    let maximum_y = vertices
        .iter()
        .map(|vertex| vertex.position[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .min(target_height as f32) as i32;
    for y in minimum_y..maximum_y {
        for x in minimum_x..maximum_x {
            let point = [x as f32 + 0.5, y as f32 + 0.5];
            let weights = [
                edge(vertices[1].position, vertices[2].position, point) / area,
                edge(vertices[2].position, vertices[0].position, point) / area,
                edge(vertices[0].position, vertices[1].position, point) / area,
            ];
            if weights.iter().any(|&weight| weight < -0.000_1) {
                continue;
            }
            let uv = [
                weights[0] * vertices[0].uv[0]
                    + weights[1] * vertices[1].uv[0]
                    + weights[2] * vertices[2].uv[0],
                weights[0] * vertices[0].uv[1]
                    + weights[1] * vertices[1].uv[1]
                    + weights[2] * vertices[2].uv[1],
            ];
            let opacity = (weights[0] * vertices[0].opacity
                + weights[1] * vertices[1].opacity
                + weights[2] * vertices[2].opacity)
                .clamp(0.0, 1.0);
            let source_x =
                (uv[0].clamp(0.0, 1.0) * (source.width.saturating_sub(1)) as f32).round() as usize;
            let source_y =
                (uv[1].clamp(0.0, 1.0) * (source.height.saturating_sub(1)) as f32).round() as usize;
            let source_offset = guest_pixel_byte_offset(source, source_x, source_y);
            let target_offset = guest_pixel_byte_offset(target, x as usize, y as usize);
            let source_pixel = &source_bytes[source_offset..source_offset + 4];
            let mut modulated = [0_u8; 4];
            for channel in 0..4 {
                modulated[channel] = transform_color_channel(source_pixel[channel], opacity, 0.0);
            }
            blend_premultiplied_rgba(
                &modulated,
                &mut target_bytes[target_offset..target_offset + 4],
            );
        }
    }
}

fn blend_premultiplied_rgba(source: &[u8], target: &mut [u8]) {
    let inverse = 255 - u32::from(source[3]);
    for channel in 0..4 {
        target[channel] = u8::try_from(
            u32::from(source[channel])
                .saturating_add((u32::from(target[channel]) * inverse + 127) / 255)
                .min(255),
        )
        .expect("clamped premultiplied blend channel fits in u8");
    }
}

fn software_blit_guest_texture(
    source: GuestTextureDescriptor,
    target: GuestTextureDescriptor,
    alpha_blend: bool,
) -> Option<()> {
    if !matches!(source.tile_mode, 0 | 27) || !matches!(target.tile_mode, 0 | 27) {
        return None;
    }
    let source_length = usize::try_from(guest_texture_byte_count(source)?).ok()?;
    let target_length = usize::try_from(guest_texture_byte_count(target)?).ok()?;
    if source_length > 128 * 1024 * 1024 || target_length > 128 * 1024 * 1024 {
        return None;
    }
    let source_address = usize::try_from(source.address).ok()?;
    let target_address = usize::try_from(target.address).ok()?;
    let source_end = source_address.checked_add(source_length)?;
    let target_end = target_address.checked_add(target_length)?;
    if source_address < target_end && target_address < source_end {
        return None;
    }
    // SAFETY: Both non-overlapping ranges are bounded live guest allocations described above.
    let source_bytes =
        unsafe { std::slice::from_raw_parts(source_address as *const u8, source_length) };
    let target_bytes =
        unsafe { std::slice::from_raw_parts_mut(target_address as *mut u8, target_length) };
    if !alpha_blend
        && source.width == target.width
        && source.height == target.height
        && source.pitch == target.pitch
        && source.tile_mode == target.tile_mode
        && source_length == target_length
    {
        target_bytes.copy_from_slice(source_bytes);
    } else {
        blit_guest_texture_bytes(source_bytes, source, target_bytes, target, alpha_blend);
    }
    Some(())
}

fn blit_guest_texture_bytes(
    source_bytes: &[u8],
    source: GuestTextureDescriptor,
    target_bytes: &mut [u8],
    target: GuestTextureDescriptor,
    alpha_blend: bool,
) {
    let source_width = usize::try_from(source.width).expect("validated source width");
    let source_height = usize::try_from(source.height).expect("validated source height");
    let target_width = usize::try_from(target.width).expect("validated target width");
    let target_height = usize::try_from(target.height).expect("validated target height");
    let copy_width = source_width.min(target_width);
    let copy_height = source_height.min(target_height);
    let source_x = source_width.saturating_sub(target_width) / 2;
    let source_y = source_height.saturating_sub(target_height) / 2;
    let target_x = target_width.saturating_sub(source_width) / 2;
    let target_y = target_height.saturating_sub(source_height) / 2;
    for y in 0..copy_height {
        for x in 0..copy_width {
            let source_offset = guest_pixel_byte_offset(source, source_x + x, source_y + y);
            let target_offset = guest_pixel_byte_offset(target, target_x + x, target_y + y);
            let source_pixel = &source_bytes[source_offset..source_offset + 4];
            let target_pixel = &mut target_bytes[target_offset..target_offset + 4];
            if alpha_blend {
                alpha_blend_rgba(source_pixel, target_pixel);
            } else {
                target_pixel.copy_from_slice(source_pixel);
            }
        }
    }
}

fn guest_pixel_byte_offset(descriptor: GuestTextureDescriptor, x: usize, y: usize) -> usize {
    if descriptor.tile_mode == 27 {
        mode27_source_byte_offset(
            x,
            y,
            usize::try_from(descriptor.width).expect("validated width"),
        )
    } else {
        (y * usize::try_from(descriptor.pitch).expect("validated pitch") + x) * 4
    }
}

fn copy_mode27_to_linear(
    source: &[u8],
    descriptor: GuestTextureDescriptor,
    destination: &mut [u8],
) -> Result<(), String> {
    let width =
        usize::try_from(descriptor.width).map_err(|_| "texture width does not fit usize")?;
    let height =
        usize::try_from(descriptor.height).map_err(|_| "texture height does not fit usize")?;
    let expected_source = usize::try_from(
        guest_texture_byte_count(descriptor).ok_or("mode 27 texture byte count overflows")?,
    )
    .map_err(|_| "mode 27 texture byte count does not fit usize")?;
    let expected_destination = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or("mode 27 linear byte count overflows")?;
    if source.len() != expected_source || destination.len() != expected_destination {
        return Err(format!(
            "mode 27 copy has source/destination lengths {}/{}, expected {expected_source}/{expected_destination}",
            source.len(),
            destination.len(),
        ));
    }
    let blocks_per_row = width.div_ceil(128);
    for block_y in 0..height.div_ceil(128) {
        let block_height = (height - block_y * 128).min(128);
        for block_x in 0..blocks_per_row {
            let block_width = (width - block_x * 128).min(128);
            let block_base = (block_y * blocks_per_row + block_x) * 65_536;
            let offsets = mode27_block_byte_offsets(block_x, block_y);
            for local_y in 0..block_height {
                let destination_row = ((block_y * 128 + local_y) * width + block_x * 128) * 4;
                let offset_row = local_y * 128;
                for local_x in 0..block_width {
                    let source_offset = block_base + usize::from(offsets[offset_row + local_x]);
                    let destination_offset = destination_row + local_x * 4;
                    destination[destination_offset..destination_offset + 4]
                        .copy_from_slice(&source[source_offset..source_offset + 4]);
                }
            }
        }
    }
    Ok(())
}

fn copy_linear_to_mode27(
    source: &[u8],
    descriptor: GuestTextureDescriptor,
    destination: &mut [u8],
) -> Result<(), String> {
    let width =
        usize::try_from(descriptor.width).map_err(|_| "texture width does not fit usize")?;
    let height =
        usize::try_from(descriptor.height).map_err(|_| "texture height does not fit usize")?;
    let expected_source = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or("mode 27 linear byte count overflows")?;
    let expected_destination = usize::try_from(
        guest_texture_byte_count(descriptor).ok_or("mode 27 texture byte count overflows")?,
    )
    .map_err(|_| "mode 27 texture byte count does not fit usize")?;
    if source.len() != expected_source || destination.len() != expected_destination {
        return Err(format!(
            "mode 27 copy has source/destination lengths {}/{}, expected {expected_source}/{expected_destination}",
            source.len(),
            destination.len(),
        ));
    }
    let blocks_per_row = width.div_ceil(128);
    for block_y in 0..height.div_ceil(128) {
        let block_height = (height - block_y * 128).min(128);
        for block_x in 0..blocks_per_row {
            let block_width = (width - block_x * 128).min(128);
            let block_base = (block_y * blocks_per_row + block_x) * 65_536;
            let offsets = mode27_block_byte_offsets(block_x, block_y);
            for local_y in 0..block_height {
                let source_row = ((block_y * 128 + local_y) * width + block_x * 128) * 4;
                let offset_row = local_y * 128;
                for local_x in 0..block_width {
                    let source_offset = source_row + local_x * 4;
                    let destination_offset =
                        block_base + usize::from(offsets[offset_row + local_x]);
                    destination[destination_offset..destination_offset + 4]
                        .copy_from_slice(&source[source_offset..source_offset + 4]);
                }
            }
        }
    }
    Ok(())
}

fn alpha_blend_rgba(source: &[u8], target: &mut [u8]) {
    let alpha = u32::from(source[3]);
    let inverse = 255 - alpha;
    for channel in 0..3 {
        target[channel] = u8::try_from(
            (u32::from(source[channel]) * alpha + u32::from(target[channel]) * inverse + 127) / 255,
        )
        .expect("blended color channel fits in u8");
    }
    target[3] = u8::try_from(alpha + (u32::from(target[3]) * inverse + 127) / 255)
        .expect("blended alpha channel fits in u8");
}

fn execute_dreaming_fill_draw(draw: &IndexedDraw) -> Option<ExecutedFill> {
    if !is_dreaming_fill_draw(draw) {
        return None;
    }
    let info = register_value(&draw.context_registers, 0x31c)?;
    let attrib2 = register_value(&draw.context_registers, 0x3b0)?;
    let attrib3 = register_value(&draw.context_registers, 0x3b8)?;
    let target_mask = register_value(&draw.context_registers, 0x08e).unwrap_or(0xf);
    let width = ((attrib2 >> 14) & 0x3fff).checked_add(1)?;
    let height = (attrib2 & 0x3fff).checked_add(1)?;
    let format = (info >> 2) & 0x1f;
    let number_type = (info >> 8) & 0x7;
    let tile_mode = (attrib3 >> 14) & 0x1f;
    if draw.color_target_address < 0x10_000
        || width > 8_192
        || height > 8_192
        || format != 10
        || number_type != 0
        || target_mask.trailing_zeros() >= 4
    {
        return None;
    }
    let byte_count = guest_texture_byte_count(GuestTextureDescriptor {
        address: draw.color_target_address,
        width,
        height,
        pitch: width,
        tile_mode,
    })?;
    if byte_count > 128 * 1024 * 1024 {
        return None;
    }
    let scalar_bits = pixel_scalars(draw);
    let rgba = scalar_bits.map(|bits| {
        let value = f32::from_bits(bits & 0xffff_e000);
        if value.is_finite() {
            transform_color_channel(255, value, 0.0)
        } else {
            0
        }
    });
    let address = usize::try_from(draw.color_target_address).ok()?;
    let length = usize::try_from(byte_count).ok()?;
    // SAFETY: The AGC color-target registers identify a live guest direct-memory allocation. Its
    // decoded dimensions and format bound this write to at most 128 MiB. Constant fills are
    // layout-independent, so this also works for Dreaming Sarah's observed tile mode 27 surface.
    let target = unsafe { std::slice::from_raw_parts_mut(address as *mut u8, length) };
    for pixel in target.chunks_exact_mut(4) {
        pixel.copy_from_slice(&rgba);
    }
    Some(ExecutedFill {
        address: draw.color_target_address,
        width,
        height,
        format,
        number_type,
        tile_mode,
        rgba,
    })
}

fn decode_dreaming_presenter_draw(
    draw: &IndexedDraw,
) -> Option<(GuestTextureDescriptor, GuestPresentConstants)> {
    if !is_dreaming_presenter_draw(draw) {
        return None;
    }
    let descriptor = decode_guest_texture(draw, 0x00c)?;
    if !matches!(descriptor.tile_mode, 0 | 27) {
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
    Some((descriptor, constants))
}

fn decode_guest_texture(draw: &IndexedDraw, register_base: u32) -> Option<GuestTextureDescriptor> {
    let fields: [u32; 8] = std::array::from_fn(|index| {
        register_value(
            &draw.shader_registers,
            register_base + u32::try_from(index).expect("texture descriptor index fits in u32"),
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
        || texture_type != 9
    {
        return None;
    }
    let descriptor_pitch = if fields[4] == 0 {
        width
    } else {
        (fields[4] & 0x3fff).checked_add(1)?
    };
    let pitch = if tile_mode == 0 {
        effective_linear_rgba_pitch(width.max(descriptor_pitch))?
    } else {
        descriptor_pitch
    };
    if pitch < width || pitch > 8_192 {
        return None;
    }

    Some(GuestTextureDescriptor {
        address,
        width,
        height,
        pitch,
        tile_mode,
    })
}

fn effective_linear_rgba_pitch(pitch: u32) -> Option<u32> {
    // GNM linear surfaces align RGBA8 rows to 256 bytes. This is implicit for common 128-bit
    // texture descriptors, so widths such as 980 and 250 must be read with 1024- and 256-pixel
    // strides even when the descriptor's extended pitch word is zero.
    pitch.checked_add(63).map(|value| value & !63)
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

fn guest_texture_byte_count(descriptor: GuestTextureDescriptor) -> Option<u64> {
    if descriptor.tile_mode == 27 {
        let blocks_wide = u64::from(descriptor.width).div_ceil(128);
        let blocks_high = u64::from(descriptor.height).div_ceil(128);
        blocks_wide.checked_mul(blocks_high)?.checked_mul(65_536)
    } else {
        u64::from(descriptor.pitch)
            .checked_mul(u64::from(descriptor.height))?
            .checked_mul(4)
    }
}

fn mode27_source_byte_offset(x: usize, y: usize, width: usize) -> usize {
    let blocks_per_row = width.div_ceil(128);
    let block_x = x / 128;
    let block_y = y / 128;
    let block_index = block_y * blocks_per_row + block_x;
    let offsets = mode27_block_byte_offsets(block_x, block_y);
    block_index * 65_536 + usize::from(offsets[(y & 127) * 128 + (x & 127)])
}

fn mode27_block_byte_offsets(block_x: usize, block_y: usize) -> &'static [u16] {
    const PIXELS_PER_BLOCK: usize = 128 * 128;
    static OFFSETS: OnceLock<Box<[u16]>> = OnceLock::new();
    let offsets = OFFSETS.get_or_init(|| {
        let mut offsets = vec![0_u16; 4 * PIXELS_PER_BLOCK];
        for block_y_parity in 0..2 {
            for block_x_parity in 0..2 {
                let table = (block_y_parity * 2 + block_x_parity) * PIXELS_PER_BLOCK;
                for local_y in 0..128 {
                    for local_x in 0..128 {
                        let global_x = local_x | (block_x_parity << 7);
                        let global_y = local_y | (block_y_parity << 7);
                        offsets[table + local_y * 128 + local_x] =
                            u16::try_from(mode27_within_block_byte_offset(global_x, global_y))
                                .expect("mode 27 byte offset fits u16");
                    }
                }
            }
        }
        offsets.into_boxed_slice()
    });
    let table = ((block_y & 1) * 2 + (block_x & 1)) * PIXELS_PER_BLOCK;
    &offsets[table..table + PIXELS_PER_BLOCK]
}

fn mode27_within_block_byte_offset(x: usize, y: usize) -> usize {
    let bit = |value: usize, index: usize| (value >> index) & 1;
    let mut within = 0;
    within |= bit(x, 0) << 2;
    within |= bit(x, 1) << 3;
    within |= bit(y, 0) << 4;
    within |= bit(y, 1) << 5;
    within |= bit(x, 2) << 6;
    within |= bit(y, 2) << 7;
    within |= (bit(x, 7) ^ bit(y, 4) ^ bit(y, 7)) << 8;
    within |= (bit(x, 4) ^ bit(y, 4)) << 9;
    within |= (bit(x, 6) ^ bit(y, 5)) << 10;
    within |= (bit(x, 5) ^ bit(y, 6)) << 11;
    within |= bit(x, 3) << 12;
    within |= bit(y, 3) << 13;
    within |= (bit(x, 6) ^ bit(y, 7)) << 14;
    within |= (bit(x, 7) ^ bit(y, 6)) << 15;
    within
}

fn convert_mode27_rgba_to_bgra(
    source: &[u8],
    width: u32,
    height: u32,
    constants: GuestPresentConstants,
    destination: &mut [u8],
) {
    let width = usize::try_from(width).expect("validated width fits");
    let height = usize::try_from(height).expect("validated height fits");
    let blocks_per_row = width.div_ceil(128);
    for block_y in 0..height.div_ceil(128) {
        let block_height = (height - block_y * 128).min(128);
        for block_x in 0..blocks_per_row {
            let block_width = (width - block_x * 128).min(128);
            let block_base = (block_y * blocks_per_row + block_x) * 65_536;
            let offsets = mode27_block_byte_offsets(block_x, block_y);
            for local_y in 0..block_height {
                let destination_row = ((block_y * 128 + local_y) * width + block_x * 128) * 4;
                let offset_row = local_y * 128;
                for local_x in 0..block_width {
                    let source_offset = block_base + usize::from(offsets[offset_row + local_x]);
                    let destination_offset = destination_row + local_x * 4;
                    let rgba = &source[source_offset..source_offset + 4];
                    let bgra = &mut destination[destination_offset..destination_offset + 4];
                    bgra[0] =
                        transform_color_channel(rgba[2], constants.multiply[2], constants.add[2]);
                    bgra[1] =
                        transform_color_channel(rgba[1], constants.multiply[1], constants.add[1]);
                    bgra[2] =
                        transform_color_channel(rgba[0], constants.multiply[0], constants.add[0]);
                    bgra[3] =
                        transform_color_channel(rgba[3], constants.multiply[3], constants.add[3]);
                }
            }
        }
    }
}

fn convert_guest_texture_to_bgra(
    source: &[u8],
    descriptor: GuestTextureDescriptor,
    constants: GuestPresentConstants,
    destination: &mut [u8],
) {
    if descriptor.tile_mode == 27 {
        convert_mode27_rgba_to_bgra(
            source,
            descriptor.width,
            descriptor.height,
            constants,
            destination,
        );
    } else {
        convert_guest_rgba_to_bgra(
            source,
            descriptor.pitch,
            descriptor.width,
            descriptor.height,
            constants,
            destination,
        );
    }
}

fn convert_guest_rgba_to_bgra(
    source: &[u8],
    source_pitch: u32,
    width: u32,
    height: u32,
    constants: GuestPresentConstants,
    destination: &mut [u8],
) {
    let source_pitch_bytes = usize::try_from(source_pitch).expect("validated pitch fits") * 4;
    let width = usize::try_from(width).expect("validated width fits");
    let height = usize::try_from(height).expect("validated height fits");
    let destination_pitch_bytes = width * 4;
    for row in 0..height {
        let source_row = &source[row * source_pitch_bytes..][..width * 4];
        let destination_row =
            &mut destination[row * destination_pitch_bytes..][..destination_pitch_bytes];
        for (rgba, bgra) in source_row
            .chunks_exact(4)
            .zip(destination_row.chunks_exact_mut(4))
        {
            bgra[0] = transform_color_channel(rgba[2], constants.multiply[2], constants.add[2]);
            bgra[1] = transform_color_channel(rgba[1], constants.multiply[1], constants.add[1]);
            bgra[2] = transform_color_channel(rgba[0], constants.multiply[0], constants.add[0]);
            bgra[3] = transform_color_channel(rgba[3], constants.multiply[3], constants.add[3]);
        }
    }
}

fn transform_color_channel(value: u8, multiply: f32, add: f32) -> u8 {
    let transformed = ((f32::from(value) / 255.0 * multiply + add).clamp(0.0, 1.0) * 255.0).round();
    // The clamp proves this finite value is in u8's inclusive range.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        transformed as u8
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
            WindowEvent::Focused(false) => {
                self.pressed_keys.clear();
                self.publish_pad_state();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.update_keyboard_input(code, event.state);
                }
            }
            WindowEvent::Resized(size) => {
                if size.width != 0
                    && size.height != 0
                    && let Some(layer) = self.layer.as_ref()
                {
                    let (width, height) = if self.guest_texture.is_some() {
                        self.guest_texture_size
                    } else {
                        self.display.map_or((size.width, size.height), |display| {
                            (display.width, display.height)
                        })
                    };
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

#[cfg(test)]
mod tests {
    use remu_gpu::{GpuRegister, IndexedDraw};

    use super::{
        DREAMING_FILL_ES_PREFIX, DREAMING_FILL_PS_PREFIX, GuestPresentConstants,
        GuestTextureDescriptor, apply_image_destination_select, blend_premultiplied_rgba,
        blit_guest_texture_bytes, convert_guest_rgba_to_bgra, convert_mode27_rgba_to_bgra,
        decode_image_descriptor, decode_scissor, decode_viewport, effective_linear_rgba_pitch,
        execute_dreaming_fill_draw, metal_nonindexed_vertex_count, mode27_source_byte_offset,
        mode27_within_block_byte_offset, read_draw_indices, snapshot_guest_texture_rgba,
        transform_sprite_position, write_guest_texture_rgba,
    };

    fn indexed_draw(address: u64, index_size: u32, index_count: u32) -> IndexedDraw {
        IndexedDraw {
            sequence: 1,
            indexed: true,
            instance_count: 1,
            index_buffer_address: address,
            index_resource_address: 0,
            index_buffer_count: index_count,
            index_size,
            index_offset: 0,
            index_count,
            draw_flags: 0,
            primitive_type: 4,
            export_shader_address: 0,
            pixel_shader_address: 0,
            export_shader_prefix: [0; 4],
            pixel_shader_prefix: [0; 4],
            export_shader: None,
            pixel_shader: None,
            export_shader_evaluation: None,
            pixel_shader_evaluation: None,
            color_target_address: 0,
            context_registers: Vec::new(),
            shader_registers: Vec::new(),
            user_config_registers: Vec::new(),
        }
    }

    #[test]
    fn decodes_agc_16_and_32_bit_index_types() {
        let indices_16 = [0_u16, 1, 2, 0, 2, 3];
        let draw_16 = indexed_draw(indices_16.as_ptr().addr() as u64, 0, 6);
        assert_eq!(read_draw_indices(&draw_16), Some(vec![0, 1, 2, 0, 2, 3]));

        let indices_32 = [0_u32, 1, 2, 0, 2, 3];
        let draw_32 = indexed_draw(indices_32.as_ptr().addr() as u64, 1, 6);
        assert_eq!(read_draw_indices(&draw_32), Some(vec![0, 1, 2, 0, 2, 3]));
    }

    #[test]
    fn aligns_implicit_linear_rgba_texture_pitch() {
        assert_eq!(effective_linear_rgba_pitch(320), Some(320));
        assert_eq!(effective_linear_rgba_pitch(980), Some(1024));
        assert_eq!(effective_linear_rgba_pitch(250), Some(256));
    }

    #[test]
    fn expands_nonindexed_rect_lists_to_a_four_vertex_triangle_strip() {
        assert_eq!(metal_nonindexed_vertex_count(0x11, 3), 4);
        assert_eq!(metal_nonindexed_vertex_count(4, 3), 3);
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
        words[3] = 0xfac | (27 << 20) | (9 << 28);
        words[4] = width - 1;

        let descriptor = decode_image_descriptor(&words).expect("valid RGBA8 2D descriptor");
        assert_eq!(descriptor.guest.address, address);
        assert_eq!(descriptor.guest.width, width);
        assert_eq!(descriptor.guest.height, height);
        assert_eq!(descriptor.guest.pitch, width);
        assert_eq!(descriptor.guest.tile_mode, 27);
        assert_eq!(descriptor.destination_select, 0xfac);
    }

    #[test]
    fn applies_gfx10_constant_and_channel_image_swizzles() {
        let mut rgba = [10_u8, 20, 30, 40];
        let destination_select = 1 | (5 << 6) | (4 << 9);
        apply_image_destination_select(&mut rgba, destination_select).expect("valid swizzle");
        assert_eq!(rgba, [255, 0, 20, 10]);
    }

    #[test]
    fn converts_pitched_rgba_and_applies_presenter_constants() {
        let source = [
            10, 20, 30, 40, 100, 120, 140, 160, 0, 0, 0, 0, 20, 40, 60, 80, 200, 220, 240, 255, 0,
            0, 0, 0,
        ];
        let constants = GuestPresentConstants {
            multiply: [1.0, 0.5, 1.0, 1.0],
            add: [0.0, 0.0, 0.1, 0.0],
        };
        let mut destination = vec![0; 16];
        convert_guest_rgba_to_bgra(&source, 3, 2, 2, constants, &mut destination);
        assert_eq!(
            destination,
            [
                56, 10, 10, 40, 166, 60, 100, 160, 86, 20, 20, 80, 255, 110, 200, 255
            ]
        );
    }

    #[test]
    fn executes_the_observed_constant_fill_into_guest_memory() {
        let mut target = vec![0xcc_u8; 65_536];
        let draw = IndexedDraw {
            sequence: 1,
            indexed: false,
            instance_count: 1,
            index_buffer_address: 0,
            index_resource_address: 0,
            index_buffer_count: 0,
            index_size: 0,
            index_offset: 0,
            index_count: 3,
            draw_flags: 0,
            primitive_type: 4,
            export_shader_address: 0,
            pixel_shader_address: 0,
            export_shader_prefix: DREAMING_FILL_ES_PREFIX,
            pixel_shader_prefix: DREAMING_FILL_PS_PREFIX,
            export_shader: None,
            pixel_shader: None,
            export_shader_evaluation: None,
            pixel_shader_evaluation: None,
            color_target_address: u64::try_from(target.as_mut_ptr().addr())
                .expect("target address"),
            context_registers: vec![
                GpuRegister {
                    offset: 0x08e,
                    value: 0xf,
                },
                GpuRegister {
                    offset: 0x31c,
                    value: 10 << 2,
                },
                GpuRegister {
                    offset: 0x3b0,
                    value: (1 << 14) | 1,
                },
                GpuRegister {
                    offset: 0x3b8,
                    value: 27 << 14,
                },
            ],
            shader_registers: [1.0_f32, 0.5, 0.0, 1.0]
                .into_iter()
                .enumerate()
                .map(|(index, value)| GpuRegister {
                    offset: 0x00c + u32::try_from(index).expect("register index"),
                    value: value.to_bits(),
                })
                .collect(),
            user_config_registers: Vec::new(),
        };

        let fill = execute_dreaming_fill_draw(&draw).expect("recognized fill draw");
        assert_eq!(fill.rgba, [255, 128, 0, 255]);
        assert!(
            target
                .chunks_exact(4)
                .all(|pixel| pixel == [255, 128, 0, 255])
        );
    }

    #[test]
    fn detiles_the_observed_mode27_rgba_layout() {
        let width = 130_usize;
        let height = 129_usize;
        let mut tiled = vec![0_u8; 4 * 65_536];
        for y in 0..height {
            for x in 0..width {
                let offset = mode27_source_byte_offset(x, y, width);
                tiled[offset..offset + 4].copy_from_slice(&[
                    u8::try_from(x).unwrap_or(0xff),
                    u8::try_from(y).unwrap_or(0xff),
                    u8::try_from(x ^ y).unwrap_or(0xff),
                    0xff,
                ]);
            }
        }
        let mut linear = vec![0_u8; width * height * 4];
        convert_mode27_rgba_to_bgra(
            &tiled,
            u32::try_from(width).expect("width"),
            u32::try_from(height).expect("height"),
            GuestPresentConstants {
                multiply: [1.0; 4],
                add: [0.0; 4],
            },
            &mut linear,
        );
        for y in 0..height {
            for x in 0..width {
                let offset = (y * width + x) * 4;
                assert_eq!(
                    linear[offset..offset + 4],
                    [
                        u8::try_from(x ^ y).unwrap_or(0xff),
                        u8::try_from(y).unwrap_or(0xff),
                        u8::try_from(x).unwrap_or(0xff),
                        0xff,
                    ]
                );
            }
        }
    }

    #[test]
    fn mode27_lookup_matches_the_scalar_address_equation_across_blocks() {
        let width = 385_usize;
        for y in 0..257 {
            for x in 0..width {
                let expected = (y / 128 * width.div_ceil(128) + x / 128) * 65_536
                    + mode27_within_block_byte_offset(x, y);
                assert_eq!(mode27_source_byte_offset(x, y, width), expected);
            }
        }
    }

    #[test]
    fn bulk_mode27_guest_copy_round_trips_partial_and_parity_blocks() {
        let width = 385_usize;
        let height = 257_usize;
        let mut tiled = vec![0_u8; width.div_ceil(128) * height.div_ceil(128) * 65_536];
        let descriptor = GuestTextureDescriptor {
            address: u64::try_from(tiled.as_mut_ptr() as usize).expect("test address fits u64"),
            width: u32::try_from(width).expect("width fits u32"),
            height: u32::try_from(height).expect("height fits u32"),
            pitch: u32::try_from(width).expect("pitch fits u32"),
            tile_mode: 27,
        };
        let mut linear = vec![0_u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let offset = (y * width + x) * 4;
                linear[offset..offset + 4].copy_from_slice(&[
                    u8::try_from(x & 0xff).expect("masked X fits u8"),
                    u8::try_from(y & 0xff).expect("masked Y fits u8"),
                    u8::try_from((x ^ y) & 0xff).expect("masked xor fits u8"),
                    0xff,
                ]);
            }
        }

        write_guest_texture_rgba(descriptor, &linear).expect("mode 27 write succeeds");
        assert_eq!(
            snapshot_guest_texture_rgba(descriptor).expect("mode 27 snapshot succeeds"),
            linear
        );
    }

    #[test]
    fn alpha_blits_a_linear_sprite_into_a_mode27_target() {
        let source = [
            100, 0, 0, 255, 0, 100, 0, 128, 0, 0, 100, 255, 200, 100, 50, 0,
        ];
        let source_descriptor = GuestTextureDescriptor {
            address: 0,
            width: 2,
            height: 2,
            pitch: 2,
            tile_mode: 0,
        };
        let target_descriptor = GuestTextureDescriptor {
            address: 0,
            width: 4,
            height: 4,
            pitch: 4,
            tile_mode: 27,
        };
        let mut target = [10, 20, 30, 255].repeat(65_536 / 4);

        blit_guest_texture_bytes(
            &source,
            source_descriptor,
            &mut target,
            target_descriptor,
            true,
        );

        let top_left = mode27_source_byte_offset(1, 1, 4);
        let top_right = mode27_source_byte_offset(2, 1, 4);
        let untouched = mode27_source_byte_offset(0, 0, 4);
        assert_eq!(target[top_left..top_left + 4], [100, 0, 0, 255]);
        assert_eq!(target[top_right..top_right + 4], [5, 60, 15, 255]);
        assert_eq!(target[untouched..untouched + 4], [10, 20, 30, 255]);
    }

    #[test]
    fn transforms_dreaming_sarah_fullscreen_composite_vertices() {
        let matrices = [
            1.357_995_2,
            0.0,
            0.0,
            0.0,
            0.0,
            2.414_213_4,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.002_002,
            -1.0,
            0.0,
            0.0,
            -2.002_002,
            0.0,
            0.115_059_41,
            0.0,
            0.0,
            0.0,
            -0.0,
            -0.115_059_41,
            -0.0,
            -0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            -100.0,
            1.0,
        ];
        let target = GuestTextureDescriptor {
            address: 0,
            width: 1280,
            height: 720,
            pitch: 1280,
            tile_mode: 27,
        };

        let top_left = transform_sprite_position([-640.0, -360.0, 0.0], &matrices, target)
            .expect("observed transform is valid");
        let bottom_right = transform_sprite_position([640.0, 360.0, 0.0], &matrices, target)
            .expect("observed transform is valid");

        assert!(top_left[0].abs() < 0.001);
        assert!(top_left[1].abs() < 0.001);
        assert!((bottom_right[0] - 1280.0).abs() < 0.001);
        assert!((bottom_right[1] - 720.0).abs() < 0.001);
    }

    #[test]
    fn blends_the_observed_premultiplied_sprite_mode() {
        let mut target = [20, 40, 60, 127];
        blend_premultiplied_rgba(&[100, 50, 25, 128], &mut target);
        assert_eq!(target, [110, 70, 55, 191]);
    }
}
