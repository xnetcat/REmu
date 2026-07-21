// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::mpsc::{Receiver, TryRecvError};

use core_graphics_types::geometry::CGSize;
use metal::{
    Device, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize, MTLStorageMode, MTLTextureType,
    MTLTextureUsage, MetalLayer, Texture, TextureDescriptor,
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
    logged_fill_draw: bool,
    logged_software_draws: HashSet<SoftwareDrawKind>,
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
            logged_fill_draw: false,
            logged_software_draws: HashSet::new(),
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
                    if is_dreaming_presenter_draw(&draw) {
                        self.pending_guest_draw = Some(draw);
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
                        if self.logged_software_draws.insert(executed.kind) {
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

fn log_guest_draw_upload(
    bytes: &[u8],
    descriptor: GuestTextureDescriptor,
    target_address: u64,
    output_length: usize,
) {
    let nonzero_bytes = bytes.iter().filter(|&&value| value != 0).count();
    let minimum = bytes.iter().copied().min().unwrap_or(0);
    let maximum = bytes.iter().copied().max().unwrap_or(0);
    let fingerprint = bytes
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, &value| {
            (hash ^ u64::from(value)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    eprintln!(
        "[REMu][METAL] fast-presented Dreaming Sarah source={:#x} {}x{} tile={} target={target_address:#x} nonzero={nonzero_bytes}/{output_length} range={minimum}..={maximum} fingerprint={fingerprint:016x}",
        descriptor.address, descriptor.width, descriptor.height, descriptor.tile_mode,
    );
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
            software_blit_guest_texture(source, target, true)?;
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
    let pitch = if fields[4] == 0 {
        width
    } else {
        (fields[4] & 0x3fff).checked_add(1)?
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
    let bit = |value: usize, index: usize| (value >> index) & 1;
    let blocks_per_row = width.div_ceil(128);
    let block_index = y / 128 * blocks_per_row + x / 128;
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
    block_index * 65_536 + within
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
    for y in 0..height {
        for x in 0..width {
            let source_offset = mode27_source_byte_offset(x, y, width);
            let destination_offset = (y * width + x) * 4;
            let rgba = &source[source_offset..source_offset + 4];
            let bgra = &mut destination[destination_offset..destination_offset + 4];
            bgra[0] = transform_color_channel(rgba[2], constants.multiply[2], constants.add[2]);
            bgra[1] = transform_color_channel(rgba[1], constants.multiply[1], constants.add[1]);
            bgra[2] = transform_color_channel(rgba[0], constants.multiply[0], constants.add[0]);
            bgra[3] = transform_color_channel(rgba[3], constants.multiply[3], constants.add[3]);
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
        convert_guest_rgba_to_bgra, convert_mode27_rgba_to_bgra, execute_dreaming_fill_draw,
        mode27_source_byte_offset,
    };

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
}
