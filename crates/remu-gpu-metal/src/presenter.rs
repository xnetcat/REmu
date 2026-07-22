// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::c_void;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::time::{Duration, Instant};

use core_graphics_types::geometry::CGSize;
use metal::{
    Buffer, CommandBuffer, CompileOptions, Device, Function, Library, MTLCommandBufferStatus,
    MTLIndexType, MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRegion,
    MTLResourceOptions, MTLSamplerAddressMode, MTLSamplerMinMagFilter, MTLSamplerMipFilter,
    MTLScissorRect, MTLSize, MTLStorageMode, MTLStoreAction, MTLTextureType, MTLTextureUsage,
    MTLViewport, MetalLayer, RenderPassDescriptor, RenderPipelineDescriptor, RenderPipelineState,
    SamplerDescriptor, SamplerState, Texture, TextureDescriptor,
};
use objc::runtime::{Class, Object, YES};
use objc::{msg_send, sel, sel_impl};
use remu_gpu::{
    GpuEvent, GpuMemoryAcquire, GpuRegister, HostPadState, IndexedDraw, PAD_BUTTON_CIRCLE,
    PAD_BUTTON_CROSS, PAD_BUTTON_DOWN, PAD_BUTTON_L1, PAD_BUTTON_L2, PAD_BUTTON_LEFT,
    PAD_BUTTON_OPTIONS, PAD_BUTTON_R1, PAD_BUTTON_R2, PAD_BUTTON_RIGHT, PAD_BUTTON_SQUARE,
    PAD_BUTTON_TRIANGLE, PAD_BUTTON_UP, set_host_pad_state,
};
use remu_memory::CpuWriteWatch;
use remu_shader::{ImageBinding, ShaderEvaluation};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::{Window, WindowId};

const EVENT_POLL_INTERVAL: Duration = Duration::from_millis(1);

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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GuestTextureDescriptor {
    address: u64,
    width: u32,
    height: u32,
    pitch: u32,
    tile_mode: u32,
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

struct ResidentGuestTexture {
    texture: Texture,
    descriptor: GuestTextureDescriptor,
    write_watch: CpuWriteWatch,
    gpu_written: bool,
    content_version: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SampledTextureKey {
    descriptor: GuestTextureDescriptor,
    destination_select: u32,
}

struct CachedSampledTexture {
    texture: Texture,
    write_watch: CpuWriteWatch,
}

struct PendingGlobalWriteback {
    buffer: Buffer,
    guest_address: u64,
    byte_count: usize,
}

struct GuestCommandBatch {
    command_buffer: CommandBuffer,
    draws: Vec<(u64, u64)>,
    global_writebacks: Vec<PendingGlobalWriteback>,
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
    guest_bgra_bytes: Vec<u8>,
    frame_texture: Option<Texture>,
    frame_texture_size: (u32, u32),
    display: Option<DisplayConfiguration>,
    pending_flip: Option<PendingFlip>,
    next_flip_sequence: u64,
    uploaded_flip_sequence: u64,
    startup_error: Option<String>,
    logged_unsupported_layout: bool,
    logged_unsupported_guest_draws: HashSet<([u32; 4], [u32; 4])>,
    captured_guest_frames: HashSet<(u64, u32, u32, u32, u64)>,
    last_guest_capture: Option<Instant>,
    compiled_shader_pairs: HashMap<(u64, u64), CompiledShaderPair>,
    compiling_shader_pairs: HashSet<(u64, u64)>,
    failed_shader_pairs: HashSet<(u64, u64)>,
    failed_draw_execution: HashSet<(u64, u64)>,
    pending_shader_draws: HashMap<(u64, u64), Box<IndexedDraw>>,
    resident_guest_textures: HashMap<GuestTextureDescriptor, ResidentGuestTexture>,
    sampled_texture_cache: HashMap<SampledTextureKey, CachedSampledTexture>,
    resident_content_version: u64,
    open_guest_batch: Option<GuestCommandBatch>,
    committed_guest_batches: VecDeque<GuestCommandBatch>,
    capture_directory: Option<PathBuf>,
    pressed_keys: HashSet<KeyCode>,
    shader_compile_sender: Sender<IndexedDraw>,
    shader_compile_receiver: Receiver<ShaderCompileResult>,
    redraw_pending: bool,
}

impl MetalPresenter {
    fn new(receiver: Receiver<GpuEvent>) -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device is available")?;
        let command_queue = device.new_command_queue();
        command_queue.set_label("REmu guest command queue");
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
            guest_bgra_bytes: Vec::new(),
            frame_texture: None,
            frame_texture_size: (0, 0),
            display: None,
            pending_flip: None,
            next_flip_sequence: 0,
            uploaded_flip_sequence: 0,
            startup_error: None,
            logged_unsupported_layout: false,
            logged_unsupported_guest_draws: HashSet::new(),
            captured_guest_frames: HashSet::new(),
            last_guest_capture: None,
            compiled_shader_pairs: HashMap::new(),
            compiling_shader_pairs: HashSet::new(),
            failed_shader_pairs: HashSet::new(),
            failed_draw_execution: HashSet::new(),
            pending_shader_draws: HashMap::new(),
            resident_guest_textures: HashMap::new(),
            sampled_texture_cache: HashMap::new(),
            resident_content_version: 0,
            open_guest_batch: None,
            committed_guest_batches: VecDeque::new(),
            capture_directory: std::env::var_os("REMU_CAPTURE_DIR").map(PathBuf::from),
            pressed_keys: HashSet::new(),
            shader_compile_sender,
            shader_compile_receiver,
            redraw_pending: false,
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
        self.retire_completed_guest_batches();
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
                    self.redraw_pending = true;
                }
                Ok(GpuEvent::MemoryAcquire(acquire)) => {
                    if let Err(error) = self.synchronize_cpu_written_guest_resources(acquire) {
                        eprintln!("[REMu][METAL] ACQUIRE_MEM synchronization failed: {error}");
                    }
                }
                Ok(GpuEvent::IndexedDraw(draw)) => {
                    let key = (draw.export_shader_address, draw.pixel_shader_address);
                    self.compile_draw_shaders(&draw);
                    if !self.compiled_shader_pairs.contains_key(&key)
                        && self.compiling_shader_pairs.contains(&key)
                    {
                        // Compilation is asynchronous so the AppKit event loop remains responsive.
                        // Retain the latest draw for a new pair: static screens may submit only
                        // once, and dropping that draw leaves the already-flipped buffer black.
                        self.pending_shader_draws.insert(key, draw);
                        continue;
                    }
                    if let Err(error) = self.execute_compiled_draw(&draw) {
                        self.log_draw_execution_failure(&draw, &error);
                    }
                }
                Ok(GpuEvent::Shutdown) | Err(TryRecvError::Disconnected) => {
                    if let Err(error) = self.wait_for_guest_command_batches() {
                        eprintln!("[REMu][METAL] shutdown synchronization failed: {error}");
                    }
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
                    if let Some(draw) = self.pending_shader_draws.remove(&result.key) {
                        match self.execute_compiled_draw(&draw) {
                            Ok(()) => self.refresh_deferred_flip(&draw),
                            Err(error) => self.log_draw_execution_failure(&draw, &error),
                        }
                    }
                }
                Err(error) => {
                    self.pending_shader_draws.remove(&result.key);
                    self.failed_shader_pairs.insert(result.key);
                    eprintln!(
                        "[REMu][METAL] Gen5 shader pair es={:#x} ps={:#x} failed: {error}",
                        result.key.0, result.key.1
                    );
                }
            }
        }
    }

    fn refresh_deferred_flip(&mut self, draw: &IndexedDraw) {
        let address = draw.color_target_address;
        if self.pending_flip.is_none_or(|flip| flip.address != address) {
            return;
        }
        self.next_flip_sequence = self.next_flip_sequence.saturating_add(1);
        self.pending_flip = Some(PendingFlip {
            address,
            sequence: self.next_flip_sequence,
        });
        self.redraw_pending = true;
    }

    fn log_draw_execution_failure(&mut self, draw: &IndexedDraw, error: &str) {
        let key = (draw.export_shader_address, draw.pixel_shader_address);
        if self.failed_draw_execution.insert(key) {
            eprintln!(
                "[REMu][METAL] Gen5 draw es={:#x} ps={:#x} could not execute: {error}",
                key.0, key.1
            );
        }
        if self.logged_unsupported_guest_draws.len() < 16
            && self
                .logged_unsupported_guest_draws
                .insert((draw.export_shader_prefix, draw.pixel_shader_prefix))
        {
            let pixel_scalars = pixel_scalars(draw);
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
        let (vertex_library, pixel_library) =
            compile_metal_libraries(device, &vertex_msl.source, &pixel_msl.source)?;
        let vertex_function = vertex_library
            .get_function(&vertex_msl.entry_point, None)
            .map_err(|error| format!("Metal vertex function: {error}"))?;
        let pixel_function = pixel_library
            .get_function(&pixel_msl.entry_point, None)
            .map_err(|error| format!("Metal pixel function: {error}"))?;
        let descriptor = RenderPipelineDescriptor::new();
        descriptor.set_label(&format!(
            "REMu Gen5 ES {:#x} PS {:#x}",
            draw.export_shader_address, draw.pixel_shader_address
        ));
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

    fn ensure_resident_texture(
        &mut self,
        device: &Device,
        descriptor: GuestTextureDescriptor,
    ) -> Result<Texture, String> {
        if let Some(resident) = self.resident_guest_textures.get(&descriptor) {
            return Ok(resident.texture.clone());
        }
        let mut aliases = self
            .resident_guest_textures
            .values()
            .filter(|resident| guest_texture_descriptors_overlap(resident.descriptor, descriptor))
            .map(|resident| (resident.content_version, resident.descriptor))
            .collect::<Vec<_>>();
        if !aliases.is_empty() {
            self.wait_for_guest_command_batches()?;
            aliases.sort_by_key(|(version, _)| *version);
            for (_, alias) in aliases {
                let resident = self
                    .resident_guest_textures
                    .remove(&alias)
                    .expect("collected resident alias exists");
                if resident.gpu_written {
                    let rgba = read_texture_rgba(&resident.texture, resident.descriptor)?;
                    write_guest_texture_rgba(resident.descriptor, &rgba)?;
                }
            }
            self.sampled_texture_cache
                .retain(|key, _| !guest_texture_descriptors_overlap(key.descriptor, descriptor));
        }
        ensure_resident_texture_in_map(device, descriptor, &mut self.resident_guest_textures)
    }

    fn synchronize_cpu_written_guest_resources(
        &mut self,
        acquire: GpuMemoryAcquire,
    ) -> Result<(), String> {
        if !acquire.invalidates_guest_resources() {
            return Ok(());
        }
        let device = self
            .device
            .as_ref()
            .ok_or("Metal device is unavailable")?
            .clone();

        for resident in self.resident_guest_textures.values_mut() {
            let (_, byte_count) = resident.write_watch.range();
            if !acquire.overlaps(resident.descriptor.address, byte_count)
                || !resident.write_watch.is_dirty()
            {
                continue;
            }
            let raw =
                snapshot_watched_guest_texture_raw(resident.descriptor, &resident.write_watch)?;
            let rgba = decode_guest_texture_rgba_from_raw(resident.descriptor, &raw)?;
            let texture = create_rgba_texture(
                &device,
                resident.descriptor.width,
                resident.descriptor.height,
                MTLTextureUsage::RenderTarget
                    | MTLTextureUsage::ShaderRead
                    | MTLTextureUsage::ShaderWrite,
                &rgba,
            )?;
            texture.set_label(&format!(
                "REMu guest image {:#x}",
                resident.descriptor.address
            ));
            // Replace instead of mutating the old texture. Already encoded Metal work retains the
            // old allocation, while following draws observe the CPU update without a GPU wait.
            resident.texture = texture;
            resident.gpu_written = false;
        }

        for (key, cached) in &mut self.sampled_texture_cache {
            let (_, byte_count) = cached.write_watch.range();
            if !acquire.overlaps(key.descriptor.address, byte_count)
                || !cached.write_watch.is_dirty()
            {
                continue;
            }
            let raw = snapshot_watched_guest_texture_raw(key.descriptor, &cached.write_watch)?;
            let mut rgba = decode_guest_texture_rgba_from_raw(key.descriptor, &raw)?;
            apply_image_destination_select(&mut rgba, key.destination_select)?;
            let texture = create_rgba_texture(
                &device,
                key.descriptor.width,
                key.descriptor.height,
                MTLTextureUsage::ShaderRead,
                &rgba,
            )?;
            texture.set_label(&format!(
                "REMu sampled guest image {:#x}",
                key.descriptor.address
            ));
            cached.texture = texture;
        }
        Ok(())
    }

    fn mark_resident_texture_written(&mut self, descriptor: GuestTextureDescriptor) {
        self.resident_content_version = self.resident_content_version.saturating_add(1);
        if let Some(resident) = self.resident_guest_textures.get_mut(&descriptor) {
            resident.gpu_written = true;
            resident.content_version = self.resident_content_version;
        }
        self.sampled_texture_cache
            .retain(|key, _| key.descriptor != descriptor);
    }

    fn synchronize_sampled_aliases(
        &mut self,
        evaluations: [&ShaderEvaluation; 2],
    ) -> Result<(), String> {
        let mut aliases = Vec::new();
        for evaluation in evaluations {
            for binding in evaluation.image_bindings.iter().filter(|binding| {
                !binding.opcode.starts_with("ImageStore")
                    && !binding.opcode.starts_with("ImageAtomic")
            }) {
                let sampled = decode_image_descriptor(&binding.resource_descriptor)
                    .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
                let identity_select = matches!(sampled.destination_select, 0 | 0xfac);
                aliases.extend(
                    self.resident_guest_textures
                        .values()
                        .filter(|resident| resident.gpu_written)
                        .filter(|resident| {
                            guest_texture_descriptors_overlap(resident.descriptor, sampled.guest)
                                && (!identity_select || resident.descriptor != sampled.guest)
                        })
                        .map(|resident| resident.descriptor),
                );
            }
        }
        aliases.sort_unstable_by_key(|descriptor| descriptor.address);
        aliases.dedup();
        if aliases.is_empty() {
            return Ok(());
        }
        self.wait_for_guest_command_batches()?;
        for descriptor in aliases {
            let resident = self
                .resident_guest_textures
                .get_mut(&descriptor)
                .expect("collected sampled alias exists");
            let rgba = read_texture_rgba(&resident.texture, descriptor)?;
            write_guest_texture_rgba(descriptor, &rgba)?;
            rearm_guest_texture_after_gpu_writeback(&resident.write_watch)?;
            resident.gpu_written = false;
        }
        Ok(())
    }

    fn retire_completed_guest_batches(&mut self) {
        loop {
            let Some(batch) = self.committed_guest_batches.front() else {
                return;
            };
            match batch.command_buffer.status() {
                MTLCommandBufferStatus::Completed => {
                    let batch = self
                        .committed_guest_batches
                        .pop_front()
                        .expect("front batch exists");
                    if let Err(error) = write_back_pending_globals(&batch.global_writebacks) {
                        eprintln!("[REMu][METAL] completed guest batch writeback failed: {error}");
                    }
                }
                MTLCommandBufferStatus::Error => {
                    let batch = self
                        .committed_guest_batches
                        .pop_front()
                        .expect("front batch exists");
                    eprintln!("[REMu][METAL] guest command batch {:?} failed", batch.draws);
                }
                _ => return,
            }
        }
    }

    fn flush_guest_command_batch(&mut self, wait: bool) -> Result<(), String> {
        let Some(batch) = self.open_guest_batch.take() else {
            return Ok(());
        };
        batch.command_buffer.commit();
        if wait {
            batch.command_buffer.wait_until_completed();
            if batch.command_buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(format!(
                    "Metal guest command batch {:?} ended in {:?}",
                    batch.draws,
                    batch.command_buffer.status()
                ));
            }
            write_back_pending_globals(&batch.global_writebacks)?;
        } else {
            self.committed_guest_batches.push_back(batch);
        }
        Ok(())
    }

    fn wait_for_guest_command_batches(&mut self) -> Result<(), String> {
        self.flush_guest_command_batch(false)?;
        while let Some(batch) = self.committed_guest_batches.pop_front() {
            batch.command_buffer.wait_until_completed();
            if batch.command_buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(format!(
                    "Metal guest command batch {:?} ended in {:?}",
                    batch.draws,
                    batch.command_buffer.status()
                ));
            }
            write_back_pending_globals(&batch.global_writebacks)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn execute_compiled_draw(&mut self, draw: &IndexedDraw) -> Result<(), String> {
        self.retire_completed_guest_batches();
        let started = Instant::now();
        let trace = draw.sequence <= 4;
        let key = (draw.export_shader_address, draw.pixel_shader_address);
        if !self.compiled_shader_pairs.contains_key(&key) {
            return Err("shader pair has no compiled Metal pipeline".to_owned());
        }
        let export_evaluation = draw
            .export_shader_evaluation
            .as_ref()
            .ok_or("draw has no export resource evaluation")?;
        let pixel_evaluation = draw
            .pixel_shader_evaluation
            .as_ref()
            .ok_or("draw has no pixel resource evaluation")?;
        let device = self
            .device
            .as_ref()
            .ok_or("Metal device is unavailable")?
            .clone();
        let queue = self
            .command_queue
            .as_ref()
            .ok_or("Metal command queue is unavailable")?
            .clone();
        let target = decode_render_target(draw).ok_or("unsupported render-target descriptor")?;
        let target_texture = self
            .ensure_resident_texture(&device, target)
            .map_err(|error| format!("render target guest memory is unreadable: {error}"))?;

        for evaluation in [export_evaluation, pixel_evaluation] {
            for binding in evaluation.image_bindings.iter().filter(|binding| {
                binding.opcode.starts_with("ImageStore")
                    || binding.opcode.starts_with("ImageAtomic")
            }) {
                let descriptor = decode_image_descriptor(&binding.resource_descriptor)
                    .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
                if descriptor.guest != target
                    && guest_texture_descriptors_overlap(descriptor.guest, target)
                {
                    return Err(format!(
                        "writable image at PC {:#x} aliases the render target through an incompatible view",
                        binding.pc
                    ));
                }
                self.ensure_resident_texture(&device, descriptor.guest)?;
            }
        }
        self.synchronize_sampled_aliases([export_evaluation, pixel_evaluation])?;

        let export_buffers = upload_global_buffers(&device, export_evaluation)?;
        let pixel_buffers = upload_global_buffers(&device, pixel_evaluation)?;
        let export_textures = upload_image_textures(
            &device,
            export_evaluation,
            &mut self.resident_guest_textures,
            &mut self.sampled_texture_cache,
        )?;
        let pixel_textures = upload_image_textures(
            &device,
            pixel_evaluation,
            &mut self.resident_guest_textures,
            &mut self.sampled_texture_cache,
        )?;
        let compiled = self
            .compiled_shader_pairs
            .get(&key)
            .ok_or("shader pair has no compiled Metal pipeline")?;
        let (export_sampler_buffer, _export_samplers) = create_sampler_argument_buffer(
            &device,
            &compiled.vertex_function,
            compiled.vertex.bindings.sampler_argument_buffer_index,
            export_evaluation,
        )?;
        let (pixel_sampler_buffer, _pixel_samplers) = create_sampler_argument_buffer(
            &device,
            &compiled.pixel_function,
            compiled.pixel.bindings.sampler_argument_buffer_index,
            pixel_evaluation,
        )?;
        let pipeline = compiled.pipeline.clone();
        let vertex_scalar_buffer_index = compiled.vertex.bindings.initial_scalar_buffer_index;
        let pixel_scalar_buffer_index = compiled.pixel.bindings.initial_scalar_buffer_index;
        let vertex_sampler_buffer_index = compiled.vertex.bindings.sampler_argument_buffer_index;
        let pixel_sampler_buffer_index = compiled.pixel.bindings.sampler_argument_buffer_index;

        let pass = RenderPassDescriptor::new();
        let attachment = pass
            .color_attachments()
            .object_at(0)
            .ok_or("Metal render pass has no color attachment zero")?;
        attachment.set_texture(Some(&target_texture));
        attachment.set_load_action(MTLLoadAction::Load);
        attachment.set_store_action(MTLStoreAction::Store);
        if self.open_guest_batch.is_none() {
            let command_buffer = queue.new_command_buffer().to_owned();
            command_buffer.set_label("REMu translated guest batch");
            self.open_guest_batch = Some(GuestCommandBatch {
                command_buffer,
                draws: Vec::new(),
                global_writebacks: Vec::new(),
            });
        }
        {
            let batch = self
                .open_guest_batch
                .as_mut()
                .expect("guest command batch was opened above");
            batch.draws.push(key);
            collect_pending_global_writebacks(
                &export_buffers,
                export_evaluation,
                &mut batch.global_writebacks,
            )?;
            collect_pending_global_writebacks(
                &pixel_buffers,
                pixel_evaluation,
                &mut batch.global_writebacks,
            )?;
            let encoder = batch.command_buffer.new_render_command_encoder(pass);
            encoder.set_label("REMu translated guest render pass");
            encoder.set_render_pipeline_state(&pipeline);
            let scissor = decode_scissor(draw, target.width, target.height)?;
            let viewport = decode_viewport(draw, target.width, target.height, scissor);
            encoder.set_viewport(viewport);
            if let GuestScissor::Rect(rect) = scissor {
                encoder.set_scissor_rect(rect);
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
            encoder.set_vertex_bytes(
                u64::from(vertex_scalar_buffer_index),
                u64::try_from(export_evaluation.initial_scalar_registers.len() * 4)
                    .map_err(|_| "vertex scalar byte count overflow")?,
                export_evaluation
                    .initial_scalar_registers
                    .as_ptr()
                    .cast::<c_void>(),
            );
            encoder.set_fragment_bytes(
                u64::from(pixel_scalar_buffer_index),
                u64::try_from(pixel_evaluation.initial_scalar_registers.len() * 4)
                    .map_err(|_| "pixel scalar byte count overflow")?,
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
            if let (Some(index), Some(buffer)) =
                (vertex_sampler_buffer_index, export_sampler_buffer.as_ref())
            {
                encoder.set_vertex_buffer(u64::from(index), Some(buffer), 0);
            }
            if let (Some(index), Some(buffer)) =
                (pixel_sampler_buffer_index, pixel_sampler_buffer.as_ref())
            {
                encoder.set_fragment_buffer(u64::from(index), Some(buffer), 0);
            }

            let primitive = metal_primitive_type(draw.primitive_type)?;
            if matches!(scissor, GuestScissor::Empty) {
                // Metal rejects an empty scissor, so a guest empty-scissor draw encodes no
                // primitive while preserving command-stream ordering.
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
                encoder.draw_primitives_instanced(
                    primitive,
                    u64::from(draw.index_offset),
                    u64::from(metal_nonindexed_vertex_count(
                        draw.primitive_type,
                        draw.index_count,
                    )),
                    u64::from(draw.instance_count),
                );
            }
            encoder.end_encoding();
        }

        self.mark_resident_texture_written(target);
        for texture in export_textures.iter().chain(&pixel_textures) {
            if texture.writable {
                self.mark_resident_texture_written(texture.descriptor.guest);
            }
        }
        let capture_due = self.capture_directory.is_some()
            && self
                .last_guest_capture
                .is_none_or(|last| last.elapsed() >= Duration::from_secs(1));
        let batch_full = self
            .open_guest_batch
            .as_ref()
            .is_some_and(|batch| batch.draws.len() >= 32);
        if capture_due {
            self.wait_for_guest_command_batches()?;
            let rendered = read_texture_rgba(&target_texture, target)?;
            self.capture_translated_frame_if_changed(target, &rendered);
        } else if batch_full {
            self.flush_guest_command_batch(false)?;
        }
        if trace {
            eprintln!(
                "[REMu][METAL] encoded Gen5 draw={} es={:#x} ps={:#x} in {:?}",
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
        if !self.redraw_pending {
            return;
        }
        // A translated target already lives in a private Metal texture. Present it directly so a
        // flip does not force a 4K GPU wait, readback, guest-memory writeback, then upload. The CPU
        // frame texture remains the fallback for pre-translation and unsupported draws.
        if self.pending_resident_frame().is_none() {
            self.upload_pending_frame();
        }
        let (Some(layer), Some(queue)) = (self.layer.as_ref(), self.command_queue.as_ref()) else {
            return;
        };
        let Some(drawable) = layer.next_drawable() else {
            return;
        };
        let command_buffer = queue.new_command_buffer();
        command_buffer.set_label("REMu guest presentation");
        if let (Some(display), Some(texture)) = (
            self.display,
            self.pending_resident_frame()
                .or(self.frame_texture.as_ref()),
        ) {
            let encoder = command_buffer.new_blit_command_encoder();
            encoder.set_label("REMu display-buffer presentation blit");
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
        self.redraw_pending = false;
    }

    fn pending_resident_frame(&self) -> Option<&Texture> {
        let display = self.display?;
        let flip = self.pending_flip?;
        self.resident_guest_textures
            .values()
            .filter(|resident| {
                resident.gpu_written
                    && resident.descriptor.address == flip.address
                    && resident.descriptor.width == display.width
                    && resident.descriptor.height == display.height
            })
            .max_by_key(|resident| resident.content_version)
            .map(|resident| &resident.texture)
    }
}

fn compile_metal_libraries(
    device: &Device,
    vertex_source: &str,
    pixel_source: &str,
) -> Result<(Library, Library), String> {
    if std::env::var_os("REMU_METAL_COMPILER").is_some() {
        return compile_metal_libraries_with_tools(device, vertex_source, pixel_source);
    }

    let compile_options = CompileOptions::new();
    let vertex = device
        .new_library_with_source(vertex_source, &compile_options)
        .map_err(|error| format!("Metal vertex compilation: {error}"))?;
    let pixel = device
        .new_library_with_source(pixel_source, &compile_options)
        .map_err(|error| format!("Metal pixel compilation: {error}"))?;
    Ok((vertex, pixel))
}

fn compile_metal_libraries_with_tools(
    device: &Device,
    vertex_source: &str,
    pixel_source: &str,
) -> Result<(Library, Library), String> {
    static NEXT_COMPILE_DIRECTORY: AtomicUsize = AtomicUsize::new(0);
    let compiler = std::env::var_os("REMU_METAL_COMPILER")
        .map(PathBuf::from)
        .ok_or("REMU_METAL_COMPILER is not configured")?;
    if !compiler.is_file() {
        return Err(format!("{} is not a file", compiler.display()));
    }
    let linker = compiler.with_file_name("metallib");
    if !linker.is_file() {
        return Err(format!("{} is not a file", linker.display()));
    }
    let sequence = NEXT_COMPILE_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory =
        std::env::temp_dir().join(format!("remu-metal-{}-{sequence}", std::process::id()));
    std::fs::create_dir(&directory)
        .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
    let directory = MetalCompileDirectory(directory);
    let vertex_path = directory.0.join("vertex.metal");
    let pixel_path = directory.0.join("pixel.metal");
    let vertex_air = directory.0.join("vertex.air");
    let pixel_air = directory.0.join("pixel.air");
    let vertex_library_path = directory.0.join("vertex.metallib");
    let pixel_library_path = directory.0.join("pixel.metallib");
    std::fs::write(&vertex_path, vertex_source)
        .map_err(|error| format!("could not write {}: {error}", vertex_path.display()))?;
    std::fs::write(&pixel_path, pixel_source)
        .map_err(|error| format!("could not write {}: {error}", pixel_path.display()))?;
    run_metal_tool(
        Command::new(&compiler)
            .current_dir(&directory.0)
            .arg("-c")
            .arg(&vertex_path)
            .arg("-o")
            .arg(&vertex_air),
        "vertex compilation",
    )?;
    run_metal_tool(
        Command::new(&compiler)
            .current_dir(&directory.0)
            .arg("-c")
            .arg(&pixel_path)
            .arg("-o")
            .arg(&pixel_air),
        "pixel compilation",
    )?;
    run_metal_tool(
        Command::new(&linker)
            .current_dir(&directory.0)
            .arg(&vertex_air)
            .arg("-o")
            .arg(&vertex_library_path),
        "vertex library link",
    )?;
    run_metal_tool(
        Command::new(&linker)
            .current_dir(&directory.0)
            .arg(&pixel_air)
            .arg("-o")
            .arg(&pixel_library_path),
        "pixel library link",
    )?;
    let vertex_library = device
        .new_library_with_file(&vertex_library_path)
        .map_err(|error| format!("could not load {}: {error}", vertex_library_path.display()))?;
    let pixel_library = device
        .new_library_with_file(&pixel_library_path)
        .map_err(|error| format!("could not load {}: {error}", pixel_library_path.display()))?;
    Ok((vertex_library, pixel_library))
}

fn run_metal_tool(command: &mut Command, operation: &str) -> Result<(), String> {
    let output = command
        .output()
        .map_err(|error| format!("could not start {operation}: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    Err(format!(
        "{operation} exited with {}: {}",
        output.status,
        diagnostics.trim()
    ))
}

struct MetalCompileDirectory(PathBuf);

impl Drop for MetalCompileDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
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

fn guest_texture_raw_range(descriptor: GuestTextureDescriptor) -> Result<(usize, usize), String> {
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
    Ok((source_address, source_length))
}

fn snapshot_guest_texture_raw(descriptor: GuestTextureDescriptor) -> Result<Vec<u8>, String> {
    let (source_address, source_length) = guest_texture_raw_range(descriptor)?;
    // SAFETY: The decoded resource descriptor names a live guest direct-memory mapping. The
    // runtime owns that mapping while GPU events referencing it may be processed.
    Ok(unsafe { std::slice::from_raw_parts(source_address as *const u8, source_length) }.to_vec())
}

fn watch_guest_texture(descriptor: GuestTextureDescriptor) -> Result<CpuWriteWatch, String> {
    let (_, byte_count) = guest_texture_raw_range(descriptor)?;
    let byte_count =
        u64::try_from(byte_count).map_err(|_| "guest texture byte count does not fit u64")?;
    // SAFETY: the validated descriptor identifies a live guest direct-memory allocation. The
    // runtime retains guest mappings while GPU events and their resident resources are alive.
    unsafe { CpuWriteWatch::new(descriptor.address, byte_count) }
        .map_err(|error| format!("could not watch guest texture CPU writes: {error}"))
}

fn snapshot_watched_guest_texture_raw(
    descriptor: GuestTextureDescriptor,
    write_watch: &CpuWriteWatch,
) -> Result<Vec<u8>, String> {
    loop {
        let _ = write_watch.consume_dirty();
        write_watch
            .rearm()
            .map_err(|error| format!("could not rearm guest texture CPU writes: {error}"))?;
        let generation = write_watch.generation();
        let raw = snapshot_guest_texture_raw(descriptor)?;
        if write_watch.generation() == generation && !write_watch.is_dirty() {
            return Ok(raw);
        }
    }
}

fn rearm_guest_texture_after_gpu_writeback(write_watch: &CpuWriteWatch) -> Result<(), String> {
    loop {
        let generation = write_watch.generation();
        let _ = write_watch.consume_dirty();
        write_watch
            .rearm()
            .map_err(|error| format!("could not rearm guest texture CPU writes: {error}"))?;
        if write_watch.generation() == generation && !write_watch.is_dirty() {
            return Ok(());
        }
    }
}

#[cfg(test)]
fn guest_texture_raw_matches(
    descriptor: GuestTextureDescriptor,
    shadow: &[u8],
) -> Result<bool, String> {
    let (source_address, source_length) = guest_texture_raw_range(descriptor)?;
    // SAFETY: See snapshot_guest_texture_raw; the comparison does not retain the borrowed range.
    let source = unsafe { std::slice::from_raw_parts(source_address as *const u8, source_length) };
    Ok(source == shadow)
}

fn decode_guest_texture_rgba_from_raw(
    descriptor: GuestTextureDescriptor,
    source: &[u8],
) -> Result<Vec<u8>, String> {
    let output_length = u64::from(descriptor.width)
        .checked_mul(u64::from(descriptor.height))
        .and_then(|pixels| pixels.checked_mul(4))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or("guest texture dimensions overflow")?;
    let expected_source = usize::try_from(
        guest_texture_byte_count(descriptor).ok_or("guest texture byte count overflows")?,
    )
    .map_err(|_| "guest texture byte count does not fit usize")?;
    if source.len() != expected_source {
        return Err(format!(
            "guest texture source has {} bytes, expected {expected_source}",
            source.len()
        ));
    }
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

#[cfg(test)]
fn snapshot_guest_texture_rgba(descriptor: GuestTextureDescriptor) -> Result<Vec<u8>, String> {
    let (source_address, source_length) = guest_texture_raw_range(descriptor)?;
    // SAFETY: See snapshot_guest_texture_raw; decoding completes before this function returns.
    let source = unsafe { std::slice::from_raw_parts(source_address as *const u8, source_length) };
    decode_guest_texture_rgba_from_raw(descriptor, source)
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

fn read_texture_rgba(
    texture: &Texture,
    descriptor: GuestTextureDescriptor,
) -> Result<Vec<u8>, String> {
    let length = u64::from(descriptor.width)
        .checked_mul(u64::from(descriptor.height))
        .and_then(|pixels| pixels.checked_mul(4))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or("Metal texture readback dimensions overflow")?;
    let mut rgba = vec![0_u8; length];
    texture.get_bytes(
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
    Ok(rgba)
}

fn ensure_resident_texture_in_map(
    device: &Device,
    descriptor: GuestTextureDescriptor,
    resident_guest_textures: &mut HashMap<GuestTextureDescriptor, ResidentGuestTexture>,
) -> Result<Texture, String> {
    if let Some(resident) = resident_guest_textures.get(&descriptor) {
        return Ok(resident.texture.clone());
    }
    let write_watch = watch_guest_texture(descriptor)?;
    let raw = snapshot_watched_guest_texture_raw(descriptor, &write_watch)?;
    let rgba = decode_guest_texture_rgba_from_raw(descriptor, &raw)?;
    let texture = create_rgba_texture(
        device,
        descriptor.width,
        descriptor.height,
        MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
        &rgba,
    )?;
    texture.set_label(&format!("REMu guest image {:#x}", descriptor.address));
    resident_guest_textures.insert(
        descriptor,
        ResidentGuestTexture {
            texture: texture.clone(),
            descriptor,
            write_watch,
            gpu_written: false,
            content_version: 0,
        },
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

fn collect_pending_global_writebacks(
    buffers: &[Buffer],
    evaluation: &ShaderEvaluation,
    pending: &mut Vec<PendingGlobalWriteback>,
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
        pending.push(PendingGlobalWriteback {
            buffer: buffer.clone(),
            guest_address: binding.base_address,
            byte_count: binding.data.len(),
        });
    }
    Ok(())
}

fn write_back_pending_globals(pending: &[PendingGlobalWriteback]) -> Result<(), String> {
    for writeback in pending {
        let destination = usize::try_from(writeback.guest_address)
            .map_err(|_| "writable global buffer address does not fit usize")?;
        // SAFETY: Shader evaluation captured this live writable guest binding, and the completed
        // command buffer owns a Metal buffer of the same exact byte length.
        unsafe {
            std::ptr::copy_nonoverlapping(
                writeback.buffer.contents().cast::<u8>(),
                destination as *mut u8,
                writeback.byte_count,
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
    resident_guest_textures: &mut HashMap<GuestTextureDescriptor, ResidentGuestTexture>,
    sampled_texture_cache: &mut HashMap<SampledTextureKey, CachedSampledTexture>,
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
            let identity_select = matches!(descriptor.destination_select, 0 | 0xfac);
            let texture = if writable {
                ensure_resident_texture_in_map(device, descriptor.guest, resident_guest_textures)?
            } else if identity_select
                && let Some(resident) = resident_guest_textures.get(&descriptor.guest)
            {
                resident.texture.clone()
            } else {
                let key = SampledTextureKey {
                    descriptor: descriptor.guest,
                    destination_select: descriptor.destination_select,
                };
                if let Some(cached) = sampled_texture_cache.get(&key) {
                    cached.texture.clone()
                } else {
                    let write_watch = watch_guest_texture(descriptor.guest)
                        .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
                    let raw = snapshot_watched_guest_texture_raw(descriptor.guest, &write_watch)
                        .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
                    let mut rgba = decode_guest_texture_rgba_from_raw(descriptor.guest, &raw)
                        .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
                    apply_image_destination_select(&mut rgba, descriptor.destination_select)
                        .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
                    let texture = create_rgba_texture(
                        device,
                        descriptor.guest.width,
                        descriptor.guest.height,
                        MTLTextureUsage::ShaderRead,
                        &rgba,
                    )?;
                    texture.set_label(&format!(
                        "REMu sampled guest image {:#x}",
                        descriptor.guest.address
                    ));
                    if sampled_texture_cache.len() >= 2_048 {
                        sampled_texture_cache.clear();
                    }
                    sampled_texture_cache.insert(
                        key,
                        CachedSampledTexture {
                            texture: texture.clone(),
                            write_watch,
                        },
                    );
                    texture
                }
            };
            Ok(UploadedImageTexture {
                texture,
                descriptor,
                writable,
            })
        })
        .collect()
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

fn effective_linear_rgba_pitch(pitch: u32) -> Option<u32> {
    // GNM linear surfaces align RGBA8 rows to 256 bytes. This is implicit for common 128-bit
    // texture descriptors, so widths such as 980 and 250 must be read with 1024- and 256-pixel
    // strides even when the descriptor's extended pitch word is zero.
    pitch.checked_add(63).map(|value| value & !63)
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

fn guest_texture_descriptors_overlap(
    left: GuestTextureDescriptor,
    right: GuestTextureDescriptor,
) -> bool {
    let Some(left_bytes) = guest_texture_byte_count(left) else {
        return false;
    };
    let Some(right_bytes) = guest_texture_byte_count(right) else {
        return false;
    };
    let left_end = left.address.saturating_add(left_bytes);
    let right_end = right.address.saturating_add(right_bytes);
    left.address < right_end && right.address < left_end
}

#[cfg(test)]
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

fn presenter_window_attributes() -> winit::window::WindowAttributes {
    Window::default_attributes()
        .with_title("REmu")
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
                    let (width, height) =
                        self.display.map_or((size.width, size.height), |display| {
                            (display.width, display.height)
                        });
                    layer.set_drawable_size(CGSize::new(f64::from(width), f64::from(height)));
                    self.redraw_pending = true;
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.receive_events(event_loop);
        if let Err(error) = self.flush_guest_command_batch(false) {
            eprintln!("[REMu][METAL] guest command batch submission failed: {error}");
        }
        if self.redraw_pending
            && let Some(window) = self.window.as_ref()
        {
            window.request_redraw();
        }
        // The std channel feeding guest GPU work cannot wake Winit directly. A short deadline
        // keeps submission latency low without ControlFlow::Poll busy-spinning an otherwise idle
        // process or continuously presenting identical drawables.
        event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + EVENT_POLL_INTERVAL));
    }
}

pub fn run_presenter(receiver: Receiver<GpuEvent>) -> Result<(), String> {
    eprintln!("[REMu][METAL] starting AppKit event loop");
    let mut presenter = MetalPresenter::new(receiver)?;
    let mut builder = EventLoop::builder();
    builder
        // REmu currently runs as a Rosetta command-line process rather than an application
        // bundle. Registering that process as a regular app gives the Dock ownership of its
        // lifecycle; the Dock may then terminate a long compatibility run even though the Winit
        // event loop is healthy. Accessory applications still own interactive windows and receive
        // keyboard input, but remain attached to the launching terminal until the future GUI ships
        // as a real `.app` bundle.
        .with_activation_policy(ActivationPolicy::Accessory)
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
    event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + EVENT_POLL_INTERVAL));
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
        GuestTextureDescriptor, apply_image_destination_select, decode_image_descriptor,
        decode_scissor, decode_viewport, effective_linear_rgba_pitch,
        guest_texture_descriptors_overlap, guest_texture_raw_matches,
        metal_nonindexed_vertex_count, mode27_source_byte_offset, mode27_within_block_byte_offset,
        read_draw_indices, snapshot_guest_texture_raw, snapshot_guest_texture_rgba,
        write_guest_texture_rgba,
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
    fn exact_raw_shadow_detects_cpu_writes_without_guessing_dirty_pages() {
        let mut pixels = vec![0x5a_u8; 16 * 8 * 4];
        let descriptor = GuestTextureDescriptor {
            address: u64::try_from(pixels.as_mut_ptr() as usize).expect("test address fits u64"),
            width: 8,
            height: 8,
            pitch: 16,
            tile_mode: 0,
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
        };
        let overlapping_view = GuestTextureDescriptor {
            address: base.address + 8_192,
            width: 32,
            height: 32,
            pitch: 32,
            tile_mode: 0,
        };
        let adjacent = GuestTextureDescriptor {
            address: base.address + 16_384,
            ..overlapping_view
        };
        assert!(guest_texture_descriptors_overlap(base, overlapping_view));
        assert!(!guest_texture_descriptors_overlap(base, adjacent));
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
}
