// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::{OsStr, c_void};
use std::hash::{Hash, Hasher};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use core_graphics_types::geometry::CGSize;
use metal::{
    Buffer, CommandBuffer, CompileOptions, ComputePipelineState, Device, Function, Library,
    MTLBlitOption, MTLClearColor, MTLColorWriteMask, MTLCommandBufferStatus, MTLIndexType,
    MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLResourceOptions,
    MTLSamplerAddressMode, MTLSamplerMinMagFilter, MTLSamplerMipFilter, MTLScissorRect, MTLSize,
    MTLStorageMode, MTLStoreAction, MTLTextureType, MTLTextureUsage, MTLViewport, MetalLayer,
    RenderPassDescriptor, RenderPipelineDescriptor, RenderPipelineState, SamplerDescriptor,
    SamplerState, Texture, TextureDescriptor,
};
use objc::runtime::{Object, YES};
use objc::{msg_send, sel, sel_impl};
use remu_gpu::{
    ComputeDispatch, ComputeGrid, GpuEvent, GpuMemoryAcquire, GpuRegister, GpuScanoutTarget,
    HostPadState, IndexedDraw, PAD_BUTTON_CIRCLE, PAD_BUTTON_CROSS, PAD_BUTTON_DOWN, PAD_BUTTON_L1,
    PAD_BUTTON_L2, PAD_BUTTON_LEFT, PAD_BUTTON_OPTIONS, PAD_BUTTON_R1, PAD_BUTTON_R2,
    PAD_BUTTON_RIGHT, PAD_BUTTON_SQUARE, PAD_BUTTON_TRIANGLE, PAD_BUTTON_UP, set_host_pad_state,
};
use remu_memory::CpuWriteWatch;
use remu_shader::{ImageBinding, PixelOutputKind, ShaderEvaluation};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::{Window, WindowId};

const EVENT_POLL_INTERVAL: Duration = Duration::from_millis(1);
const GPU_EVENTS_PER_TICK: usize = 16;
const SHADER_COMPILE_QUEUE_CAPACITY: usize = 1;
const METAL_TEXTURE_BUFFER_ROW_ALIGNMENT: u64 = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PresenterFlow {
    Continue,
    Shutdown,
}

fn headless_metal_value_is_enabled(value: Option<&OsStr>) -> bool {
    value.is_some_and(|value| value == "1")
}

fn headless_metal_requested() -> bool {
    headless_metal_value_is_enabled(std::env::var_os("REMU_HEADLESS_METAL").as_deref())
}

// GFX10 context-register offsets are the MMIO dword address minus 0xA000. Color registers
// 0..7 use a 15-dword primary stride; the high address and GFX10 extent/swizzle registers are
// stored in separate dense eight-register arrays. These values match Mesa's generated gfx10
// register database and SharpEmu's AGC command-stream decoder.
const CB_TARGET_MASK: u32 = 0x08e;
const CB_COLOR0_BASE: u32 = 0x318;
const CB_COLOR_REGISTER_STRIDE: u32 = 15;
const CB_COLOR0_VIEW: u32 = 0x31b;
const CB_COLOR0_INFO: u32 = 0x31c;
const CB_COLOR0_ATTRIB: u32 = 0x31d;
const CB_COLOR0_BASE_EXT: u32 = 0x390;
const CB_COLOR0_ATTRIB2: u32 = 0x3b0;
const CB_COLOR0_ATTRIB3: u32 = 0x3b8;
const COLOR_TARGET_COUNT: u32 = 8;

const PRESENTATION_SHADER_SOURCE: &str = r"
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
    return source.sample(source_sampler, input.texture_coordinate);
}
";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PresentationResolution {
    LogicalWindow,
    NativeWindow,
    Fixed(u32, u32),
}

impl PresentationResolution {
    fn from_environment() -> Result<Self, String> {
        let Some(value) = std::env::var_os("REMU_PRESENTATION_SIZE") else {
            return Ok(Self::LogicalWindow);
        };
        let value = value
            .into_string()
            .map_err(|_| "REMU_PRESENTATION_SIZE is not valid UTF-8".to_owned())?;
        parse_presentation_resolution(&value)
    }
}

struct PresentationResources {
    pipeline: RenderPipelineState,
    _vertex_function: Function,
    _fragment_function: Function,
    sampler: SamplerState,
}

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
    pixel_format: MTLPixelFormat,
    bytes_per_pixel: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TextureReadbackLayout {
    packed_bytes_per_row: u64,
    staging_bytes_per_row: u64,
    staging_length: u64,
}

impl TextureReadbackLayout {
    fn new(width: u32, height: u32, bytes_per_pixel: u32) -> Result<Self, String> {
        if width == 0 || height == 0 || bytes_per_pixel == 0 {
            return Err(format!(
                "invalid Metal texture readback extent {width}x{height} at {bytes_per_pixel} bytes per pixel"
            ));
        }
        let packed_bytes_per_row = u64::from(width)
            .checked_mul(u64::from(bytes_per_pixel))
            .ok_or("Metal texture readback row size overflows")?;
        let staging_bytes_per_row = packed_bytes_per_row
            .checked_add(METAL_TEXTURE_BUFFER_ROW_ALIGNMENT - 1)
            .map(|value| value & !(METAL_TEXTURE_BUFFER_ROW_ALIGNMENT - 1))
            .ok_or("Metal texture readback aligned row size overflows")?;
        let staging_length = staging_bytes_per_row
            .checked_mul(u64::from(height))
            .ok_or("Metal texture readback staging size overflows")?;
        Ok(Self {
            packed_bytes_per_row,
            staging_bytes_per_row,
            staging_length,
        })
    }

    fn packed_length(self, height: u32) -> Result<usize, String> {
        self.packed_bytes_per_row
            .checked_mul(u64::from(height))
            .and_then(|length| usize::try_from(length).ok())
            .ok_or_else(|| "Metal texture readback packed size overflows".to_owned())
    }
}

struct PendingTextureReadback {
    command_buffer: CommandBuffer,
    staging: Buffer,
    // Keep the source alive independently of Metal command-buffer retention policy.
    _source: Texture,
    layout: TextureReadbackLayout,
    height: u32,
}

impl PendingTextureReadback {
    // Every source writer must already be committed. The presenter either completes guest batches
    // first or relies on this queue's serial order, so the blit cannot overtake its producer.
    fn encode(
        device: &Device,
        queue: &metal::CommandQueue,
        texture: &Texture,
        descriptor: GuestTextureDescriptor,
    ) -> Result<Self, String> {
        validate_texture_readback_source(texture, descriptor)?;
        let layout = TextureReadbackLayout::new(
            descriptor.width,
            descriptor.height,
            descriptor.bytes_per_pixel,
        )?;
        let staging = device.new_buffer(
            layout.staging_length,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache,
        );
        staging.set_label("REMu texture readback staging");
        let command_buffer = queue.new_command_buffer().to_owned();
        command_buffer.set_label("REMu texture readback");
        let encoder = command_buffer.new_blit_command_encoder();
        encoder.set_label("REMu texture-to-buffer readback blit");
        encoder.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize::new(u64::from(descriptor.width), u64::from(descriptor.height), 1),
            &staging,
            0,
            layout.staging_bytes_per_row,
            layout.staging_length,
            MTLBlitOption::None,
        );
        encoder.end_encoding();
        command_buffer.commit();
        Ok(Self {
            command_buffer,
            staging,
            _source: texture.clone(),
            layout,
            height: descriptor.height,
        })
    }

    fn finish(self) -> Result<Vec<u8>, String> {
        self.command_buffer.wait_until_completed();
        if self.command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(format!(
                "Metal texture readback ended in {:?}",
                self.command_buffer.status()
            ));
        }
        let packed_row = usize::try_from(self.layout.packed_bytes_per_row)
            .map_err(|_| "Metal texture readback packed row does not fit usize")?;
        let staging_row = usize::try_from(self.layout.staging_bytes_per_row)
            .map_err(|_| "Metal texture readback staging row does not fit usize")?;
        let staging_length = usize::try_from(self.layout.staging_length)
            .map_err(|_| "Metal texture readback staging size does not fit usize")?;
        let packed_length = self.layout.packed_length(self.height)?;
        let contents = self.staging.contents().cast::<u8>();
        if contents.is_null() {
            return Err("Metal shared texture readback buffer has no CPU mapping".to_owned());
        }
        // SAFETY: The shared buffer remains owned by `self`, its completed blit wrote exactly
        // `staging_length` bytes, and `contents` is non-null for this CPU-visible storage mode.
        let staged = unsafe { std::slice::from_raw_parts(contents, staging_length) };
        let mut packed = vec![0_u8; packed_length];
        for row in 0..usize::try_from(self.height).map_err(|_| "height does not fit usize")? {
            let source = row * staging_row;
            let destination = row * packed_row;
            packed[destination..destination + packed_row]
                .copy_from_slice(&staged[source..source + packed_row]);
        }
        Ok(packed)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct RenderTargetFormat {
    pixel_format: MTLPixelFormat,
    output_kind: PixelOutputKind,
    bytes_per_pixel: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct RenderTargetPipelineState {
    guest_slot: u32,
    pixel_format: MTLPixelFormat,
    output_kind: PixelOutputKind,
    write_mask: u32,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct GraphicsPipelineKey {
    export_shader_address: u64,
    pixel_shader_address: u64,
    targets: Vec<RenderTargetPipelineState>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DecodedRenderTarget {
    guest_slot: u32,
    texture: GuestTextureDescriptor,
    output_kind: PixelOutputKind,
    write_mask: u32,
    direct_scanout: bool,
}

struct DirectScanoutTexture {
    descriptor: GuestTextureDescriptor,
    texture: Texture,
}

struct CompiledShaderPair {
    pipeline: RenderPipelineState,
    vertex_function: Function,
    pixel_function: Function,
    vertex: remu_shader_metal::MetalShader,
    pixel: remu_shader_metal::MetalShader,
}

struct CompiledComputePipeline {
    pipeline: ComputePipelineState,
    function: Function,
    shader: remu_shader_metal::MetalShader,
}

#[derive(Clone)]
struct ProgramIdentity(Arc<remu_shader::Program>);

impl std::fmt::Debug for ProgramIdentity {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProgramIdentity")
            .field("address", &self.0.address)
            .field("pointer", &Arc::as_ptr(&self.0))
            .finish()
    }
}

impl PartialEq for ProgramIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ProgramIdentity {}

impl Hash for ProgramIdentity {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.0).hash(state);
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ComputePipelineKey {
    program_identity: ProgramIdentity,
    lds_size_bytes: u32,
    wave_lane_count: u32,
    compute_system_registers: Option<remu_shader::ComputeSystemRegisters>,
    global_bindings: Vec<(Vec<u32>, u64, usize, bool)>,
    image_bindings: Vec<(u32, bool)>,
}

struct ShaderCompileResult {
    key: GraphicsPipelineKey,
    compiled: Result<CompiledShaderPair, String>,
}

struct ShaderCompileRequest {
    key: GraphicsPipelineKey,
    draw: IndexedDraw,
}

struct BlockedGraphicsDraw {
    key: GraphicsPipelineKey,
    draw: Box<IndexedDraw>,
}

struct OrderedGpuEventReceiver {
    receiver: Receiver<GpuEvent>,
    blocked_draw: Option<BlockedGraphicsDraw>,
}

#[derive(Debug)]
enum OrderedEventPoll {
    Event(GpuEvent),
    Blocked,
    Empty,
    Disconnected,
}

impl OrderedGpuEventReceiver {
    fn new(receiver: Receiver<GpuEvent>) -> Self {
        Self {
            receiver,
            blocked_draw: None,
        }
    }

    fn block_draw(&mut self, key: GraphicsPipelineKey, draw: Box<IndexedDraw>) {
        assert!(
            self.blocked_draw.is_none(),
            "ordered Metal event gate already contains a blocked draw"
        );
        self.blocked_draw = Some(BlockedGraphicsDraw { key, draw });
    }

    fn try_recv(
        &mut self,
        pipeline_resolved: impl FnOnce(&GraphicsPipelineKey) -> bool,
    ) -> OrderedEventPoll {
        if let Some(blocked) = self.blocked_draw.as_ref() {
            if !pipeline_resolved(&blocked.key) {
                return OrderedEventPoll::Blocked;
            }
            let blocked = self
                .blocked_draw
                .take()
                .expect("blocked draw exists after readiness check");
            return OrderedEventPoll::Event(GpuEvent::IndexedDraw(blocked.draw));
        }
        match self.receiver.try_recv() {
            Ok(event) => OrderedEventPoll::Event(event),
            Err(TryRecvError::Empty) => OrderedEventPoll::Empty,
            Err(TryRecvError::Disconnected) => OrderedEventPoll::Disconnected,
        }
    }
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

#[derive(Debug)]
enum GuestBatchWork {
    Draw { export: u64, pixel: u64 },
    Compute { shader: u64 },
}

fn describe_guest_batch(work: &[GuestBatchWork]) -> String {
    work.iter()
        .map(|item| match item {
            GuestBatchWork::Draw { export, pixel } => format!("draw({export:#x},{pixel:#x})"),
            GuestBatchWork::Compute { shader } => format!("compute({shader:#x})"),
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn compute_pipeline_key(
    dispatch: &ComputeDispatch,
    evaluation: &ShaderEvaluation,
    program: &std::sync::Arc<remu_shader::Program>,
) -> ComputePipelineKey {
    let embeds_global_base = program.instructions.iter().any(|instruction| {
        matches!(
            instruction.control.as_ref(),
            Some(remu_shader::InstructionControl::GlobalMemory(_))
        )
    });
    ComputePipelineKey {
        // Retaining the Arc makes pointer identity stable for the lifetime of both successful and
        // negative cache entries; an allocator cannot recycle a dropped Program into this key.
        program_identity: ProgramIdentity(Arc::clone(program)),
        lds_size_bytes: dispatch.launch.lds_size_bytes,
        wave_lane_count: dispatch.launch.wave_lane_count,
        compute_system_registers: evaluation.compute_system_registers,
        global_bindings: evaluation
            .global_memory_bindings
            .iter()
            .map(|binding| {
                (
                    binding.instruction_pcs.clone(),
                    if embeds_global_base {
                        binding.base_address
                    } else {
                        0
                    },
                    binding.data.len(),
                    binding.writable,
                )
            })
            .collect(),
        image_bindings: evaluation
            .image_bindings
            .iter()
            .map(|binding| {
                let unified_format = binding
                    .resource_descriptor
                    .get(1)
                    .map_or(0, |word| (word >> 20) & 0x1ff);
                let storage = is_storage_image_opcode(binding.opcode);
                (unified_format, storage)
            })
            .collect(),
    }
}

fn is_writable_image_opcode(opcode: remu_shader::Opcode) -> bool {
    opcode.starts_with("ImageStore") || opcode.starts_with("ImageAtomic")
}

fn is_storage_image_opcode(opcode: remu_shader::Opcode) -> bool {
    is_writable_image_opcode(opcode) || opcode.starts_with("ImageLoad")
}

struct GuestCommandBatch {
    command_buffer: CommandBuffer,
    work: Vec<GuestBatchWork>,
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
    receiver: OrderedGpuEventReceiver,
    window: Option<Window>,
    device: Option<Device>,
    layer: Option<MetalLayer>,
    command_queue: Option<metal::CommandQueue>,
    presentation: PresentationResources,
    presentation_resolution: PresentationResolution,
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
    compiled_shader_pairs: HashMap<GraphicsPipelineKey, CompiledShaderPair>,
    compiling_shader_pairs: HashSet<GraphicsPipelineKey>,
    failed_shader_pairs: HashSet<GraphicsPipelineKey>,
    failed_draw_execution: HashSet<GraphicsPipelineKey>,
    compiled_compute_pipelines: HashMap<ComputePipelineKey, CompiledComputePipeline>,
    failed_compute_pipelines: HashSet<ComputePipelineKey>,
    failed_compute_execution: HashSet<u64>,
    resident_guest_textures: HashMap<GuestTextureDescriptor, ResidentGuestTexture>,
    direct_scanout_textures: HashMap<u64, DirectScanoutTexture>,
    sampled_texture_cache: HashMap<SampledTextureKey, CachedSampledTexture>,
    resident_content_version: u64,
    open_guest_batch: Option<GuestCommandBatch>,
    committed_guest_batches: VecDeque<GuestCommandBatch>,
    capture_directory: Option<PathBuf>,
    pressed_keys: HashSet<KeyCode>,
    shader_compile_sender: SyncSender<ShaderCompileRequest>,
    shader_compile_receiver: Receiver<ShaderCompileResult>,
    redraw_pending: bool,
}

impl MetalPresenter {
    fn new(receiver: Receiver<GpuEvent>) -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device is available")?;
        let command_queue = device.new_command_queue();
        command_queue.set_label("REmu guest command queue");
        let presentation_resolution = PresentationResolution::from_environment()?;
        let presentation = create_presentation_resources(&device)?;
        let (shader_compile_sender, shader_compile_requests) =
            mpsc::sync_channel::<ShaderCompileRequest>(SHADER_COMPILE_QUEUE_CAPACITY);
        let (shader_compile_results, shader_compile_receiver) =
            mpsc::channel::<ShaderCompileResult>();
        let compiler_device = device.clone();
        std::thread::Builder::new()
            .name("remu-metal-compiler".to_owned())
            .spawn(move || {
                while let Ok(request) = shader_compile_requests.recv() {
                    let ShaderCompileRequest { key, draw } = request;
                    let compiled = objc::rc::autoreleasepool(|| {
                        Self::compile_draw_shader_pair(&compiler_device, &draw, &key)
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
            receiver: OrderedGpuEventReceiver::new(receiver),
            window: None,
            device: Some(device),
            layer: None,
            command_queue: Some(command_queue),
            presentation,
            presentation_resolution,
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
            compiled_compute_pipelines: HashMap::new(),
            failed_compute_pipelines: HashSet::new(),
            failed_compute_execution: HashSet::new(),
            resident_guest_textures: HashMap::new(),
            direct_scanout_textures: HashMap::new(),
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
        // Presentation uses a fullscreen sampling pass so a high-resolution guest display buffer
        // does not force an equally large host drawable. The layer remains framebuffer-only.
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
        let (width, height) = presentation_drawable_size(
            self.presentation_resolution,
            size.width,
            size.height,
            window.scale_factor(),
        );
        layer.set_drawable_size(CGSize::new(f64::from(width), f64::from(height)));
        eprintln!(
            "[REMu][METAL] presenter started on {} at {width}x{height} ({:?})",
            device.name(),
            self.presentation_resolution
        );

        self.window = Some(window);
        self.layer = Some(layer);
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn receive_events(&mut self) -> PresenterFlow {
        self.receive_shader_compilations();
        self.retire_completed_guest_batches();
        // A bounded batch prevents a continuously producing guest from starving redraw and input
        // delivery. If the oldest event is waiting on shader compilation, the ordered receiver
        // stops here without touching any later event in the bounded upstream channel.
        for _ in 0..GPU_EVENTS_PER_TICK {
            let event = match self.receiver.try_recv(|key| {
                self.compiled_shader_pairs.contains_key(key)
                    || self.failed_shader_pairs.contains(key)
            }) {
                OrderedEventPoll::Event(event) => event,
                OrderedEventPoll::Blocked | OrderedEventPoll::Empty => {
                    return PresenterFlow::Continue;
                }
                OrderedEventPoll::Disconnected => {
                    eprintln!("[REMu][METAL] GPU event channel disconnected; closing presenter");
                    if let Err(error) = self.wait_for_guest_command_batches() {
                        eprintln!("[REMu][METAL] shutdown synchronization failed: {error}");
                    }
                    return PresenterFlow::Shutdown;
                }
            };
            match event {
                GpuEvent::DisplayRegistered {
                    width,
                    height,
                    pitch,
                    pixel_format,
                    tiling_mode,
                    ..
                } => {
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
                GpuEvent::Flip {
                    address,
                    width,
                    height,
                    pitch,
                    pixel_format,
                    tiling_mode,
                    ..
                } => {
                    self.display = Some(DisplayConfiguration {
                        width,
                        height,
                        pitch,
                        pixel_format,
                        tiling_mode,
                    });
                    self.next_flip_sequence = self.next_flip_sequence.saturating_add(1);
                    match self.activate_direct_scanout(address, width, height) {
                        Ok(true) => {
                            self.uploaded_flip_sequence = self.next_flip_sequence;
                        }
                        Ok(false) => {
                            if let Err(error) = self.synchronize_flip_target(address) {
                                eprintln!("[REMu][METAL] flip synchronization failed: {error}");
                            }
                        }
                        Err(error) => {
                            eprintln!(
                                "[REMu][METAL] direct scanout synchronization failed: {error}"
                            );
                        }
                    }
                    self.pending_flip = Some(PendingFlip {
                        address,
                        sequence: self.next_flip_sequence,
                    });
                    self.redraw_pending = true;
                }
                GpuEvent::PipelineBarrier(_barrier) => {
                    // EVENT_WRITE partial flushes are GPU ordering boundaries, not CPU waits.
                    // Closing the current batch places all following work in a later command
                    // buffer on the same serial Metal queue.
                    if let Err(error) = self.flush_guest_command_batch(false) {
                        eprintln!("[REMu][METAL] pipeline barrier failed: {error}");
                    }
                }
                GpuEvent::MemoryAcquire(acquire) => {
                    if let Err(error) = self.synchronize_cpu_written_guest_resources(acquire) {
                        eprintln!("[REMu][METAL] ACQUIRE_MEM synchronization failed: {error}");
                    }
                }
                GpuEvent::ComputeDispatch(dispatch) => {
                    if let Err(error) = self.execute_compute_dispatch(&dispatch)
                        && self
                            .failed_compute_execution
                            .insert(dispatch.shader_address)
                    {
                        eprintln!(
                            "[REMu][METAL] Gen5 compute cs={:#x} could not execute: {error}",
                            dispatch.shader_address
                        );
                    }
                }
                GpuEvent::IndexedDraw(draw) => {
                    let key = graphics_pipeline_key(&draw);
                    self.compile_draw_shaders(&draw);
                    if let Ok(key) = key.as_ref()
                        && !self.compiled_shader_pairs.contains_key(key)
                        && !self.failed_shader_pairs.contains(key)
                    {
                        debug_assert!(self.compiling_shader_pairs.contains(key));
                        self.receiver.block_draw(key.clone(), draw);
                        return PresenterFlow::Continue;
                    }
                    if let Err(error) = self.execute_compiled_draw(&draw) {
                        if key
                            .as_ref()
                            .is_ok_and(|key| self.failed_draw_execution.insert(key.clone()))
                        {
                            eprintln!(
                                "[REMu][METAL] Gen5 draw es={:#x} ps={:#x} could not execute: {error}",
                                draw.export_shader_address, draw.pixel_shader_address
                            );
                        }
                        if self.logged_unsupported_guest_draws.len() < 16
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
                }
                GpuEvent::Shutdown => {
                    eprintln!("[REMu][METAL] guest runtime requested presenter shutdown");
                    if let Err(error) = self.wait_for_guest_command_batches() {
                        eprintln!("[REMu][METAL] shutdown synchronization failed: {error}");
                    }
                    return PresenterFlow::Shutdown;
                }
            }
        }
        PresenterFlow::Continue
    }

    fn compile_draw_shaders(&mut self, draw: &IndexedDraw) {
        let Ok(key) = graphics_pipeline_key(draw) else {
            return;
        };
        if self.compiled_shader_pairs.contains_key(&key)
            || self.compiling_shader_pairs.contains(&key)
            || self.failed_shader_pairs.contains(&key)
        {
            return;
        }
        self.compiling_shader_pairs.insert(key.clone());
        if self
            .shader_compile_sender
            .send(ShaderCompileRequest {
                key: key.clone(),
                draw: draw.clone(),
            })
            .is_err()
        {
            self.compiling_shader_pairs.remove(&key);
            self.failed_shader_pairs.insert(key.clone());
            eprintln!(
                "[REMu][METAL] Gen5 shader pair es={:#x} ps={:#x} failed: compiler worker stopped",
                key.export_shader_address, key.pixel_shader_address
            );
        }
    }

    fn receive_shader_compilations(&mut self) {
        loop {
            let result = match self.shader_compile_receiver.try_recv() {
                Ok(result) => result,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    for key in self.compiling_shader_pairs.drain() {
                        eprintln!(
                            "[REMu][METAL] Gen5 shader pair es={:#x} ps={:#x} failed: compiler worker stopped",
                            key.export_shader_address, key.pixel_shader_address
                        );
                        self.failed_shader_pairs.insert(key);
                    }
                    break;
                }
            };
            self.compiling_shader_pairs.remove(&result.key);
            match result.compiled {
                Ok(compiled) => {
                    eprintln!(
                        "[REMu][METAL] compiled Gen5 shader pair es={:#x} ps={:#x} MSL={vertex_bytes}/{pixel_bytes} bytes",
                        result.key.export_shader_address,
                        result.key.pixel_shader_address,
                        vertex_bytes = compiled.vertex.source.len(),
                        pixel_bytes = compiled.pixel.source.len(),
                    );
                    self.compiled_shader_pairs.insert(result.key, compiled);
                }
                Err(error) => {
                    self.failed_shader_pairs.insert(result.key.clone());
                    eprintln!(
                        "[REMu][METAL] Gen5 shader pair es={:#x} ps={:#x} failed: {error}",
                        result.key.export_shader_address, result.key.pixel_shader_address
                    );
                }
            }
        }
    }

    fn compile_draw_shader_pair(
        device: &Device,
        draw: &IndexedDraw,
        key: &GraphicsPipelineKey,
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
        let pixel_outputs = key
            .targets
            .iter()
            .enumerate()
            .map(|(host_location, target)| remu_shader::PixelOutputBinding {
                guest_slot: target.guest_slot,
                host_location: u32::try_from(host_location)
                    .expect("at most eight color targets fit u32"),
                kind: target.output_kind,
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
        for (location, target) in key.targets.iter().enumerate() {
            let attachment = descriptor
                .color_attachments()
                .object_at(u64::try_from(location).expect("color location fits u64"))
                .ok_or_else(|| format!("Metal has no color attachment {location}"))?;
            attachment.set_pixel_format(target.pixel_format);
            attachment.set_write_mask(metal_color_write_mask(target.write_mask));
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
            let queue = self
                .command_queue
                .as_ref()
                .ok_or("Metal command queue is unavailable")?
                .clone();
            aliases.sort_by_key(|(version, _)| *version);
            for (_, alias) in aliases {
                let resident = self
                    .resident_guest_textures
                    .remove(&alias)
                    .expect("collected resident alias exists");
                if resident.gpu_written {
                    let linear = read_texture_linear(
                        device,
                        &queue,
                        &resident.texture,
                        resident.descriptor,
                    )?;
                    write_guest_texture_linear(resident.descriptor, &linear)?;
                }
            }
            self.sampled_texture_cache
                .retain(|key, _| !guest_texture_descriptors_overlap(key.descriptor, descriptor));
        }
        ensure_resident_texture_in_map(device, descriptor, &mut self.resident_guest_textures)
    }

    fn ensure_direct_scanout_texture(
        &mut self,
        device: &Device,
        descriptor: GuestTextureDescriptor,
    ) -> Texture {
        if let Some(scanout) = self.direct_scanout_textures.get(&descriptor.address)
            && scanout.descriptor == descriptor
        {
            return scanout.texture.clone();
        }
        let metal_descriptor = TextureDescriptor::new();
        metal_descriptor.set_texture_type(MTLTextureType::D2);
        metal_descriptor.set_pixel_format(descriptor.pixel_format);
        metal_descriptor.set_width(u64::from(descriptor.width));
        metal_descriptor.set_height(u64::from(descriptor.height));
        metal_descriptor.set_storage_mode(MTLStorageMode::Private);
        metal_descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        let texture = device.new_texture(&metal_descriptor);
        self.direct_scanout_textures.insert(
            descriptor.address,
            DirectScanoutTexture {
                descriptor,
                texture: texture.clone(),
            },
        );
        texture
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
            let linear = decode_guest_texture_linear_from_raw(resident.descriptor, &raw)?;
            let texture = create_guest_texture(
                &device,
                resident.descriptor,
                MTLTextureUsage::RenderTarget
                    | MTLTextureUsage::ShaderRead
                    | MTLTextureUsage::ShaderWrite,
                &linear,
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
            let mut rgba = decode_guest_texture_linear_from_raw(key.descriptor, &raw)?;
            apply_image_destination_select(&mut rgba, key.destination_select)?;
            let texture =
                create_guest_texture(&device, key.descriptor, MTLTextureUsage::ShaderRead, &rgba)?;
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
        evaluations: &[&ShaderEvaluation],
    ) -> Result<(), String> {
        let mut aliases = Vec::new();
        for &evaluation in evaluations {
            for binding in evaluation
                .image_bindings
                .iter()
                .filter(|binding| !is_storage_image_opcode(binding.opcode))
            {
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
        for descriptor in aliases {
            let resident = self
                .resident_guest_textures
                .get_mut(&descriptor)
                .expect("collected sampled alias exists");
            let linear = read_texture_linear(&device, &queue, &resident.texture, descriptor)?;
            write_guest_texture_linear(descriptor, &linear)?;
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
                    eprintln!(
                        "[REMu][METAL] guest command batch [{}] failed",
                        describe_guest_batch(&batch.work)
                    );
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
                    "Metal guest command batch [{}] ended in {:?}",
                    describe_guest_batch(&batch.work),
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
                    "Metal guest command batch [{}] ended in {:?}",
                    describe_guest_batch(&batch.work),
                    batch.command_buffer.status()
                ));
            }
            write_back_pending_globals(&batch.global_writebacks)?;
        }
        Ok(())
    }

    fn synchronize_flip_target(&mut self, address: u64) -> Result<(), String> {
        let display = self.display;
        let descriptor = self
            .resident_guest_textures
            .values()
            .filter(|resident| resident.descriptor.address == address && resident.gpu_written)
            .filter(|resident| {
                display.is_none_or(|display| {
                    resident.descriptor.width == display.width
                        && resident.descriptor.height == display.height
                })
            })
            .max_by_key(|resident| resident.content_version)
            .map(|resident| resident.descriptor);
        let Some(descriptor) = descriptor else {
            return Ok(());
        };
        self.wait_for_guest_command_batches()?;
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
        let resident = self
            .resident_guest_textures
            .get_mut(&descriptor)
            .expect("selected resident texture exists");
        let linear = read_texture_linear(&device, &queue, &resident.texture, descriptor)?;
        write_guest_texture_linear(descriptor, &linear)?;
        rearm_guest_texture_after_gpu_writeback(&resident.write_watch)?;
        resident.gpu_written = false;
        Ok(())
    }

    fn activate_direct_scanout(
        &mut self,
        address: u64,
        width: u32,
        height: u32,
    ) -> Result<bool, String> {
        let Some((descriptor, texture)) = self
            .direct_scanout_textures
            .get(&address)
            .filter(|scanout| {
                scanout.descriptor.width == width && scanout.descriptor.height == height
            })
            .map(|scanout| (scanout.descriptor, scanout.texture.clone()))
        else {
            return Ok(false);
        };
        self.wait_for_guest_command_batches()?;
        self.frame_texture = Some(texture);
        self.frame_texture_size = (descriptor.width, descriptor.height);
        if self.next_flip_sequence == 1 {
            eprintln!(
                "[REMu][METAL] presenting translated direct scanout at {address:#x} {}x{}",
                descriptor.width, descriptor.height
            );
        }
        Ok(true)
    }

    #[allow(clippy::too_many_lines)]
    fn execute_compute_dispatch(&mut self, dispatch: &ComputeDispatch) -> Result<(), String> {
        self.retire_completed_guest_batches();
        if dispatch.launch.base_group != [0; 3] {
            return Err(format!(
                "Metal compute does not yet support nonzero base groups {:?}",
                dispatch.launch.base_group
            ));
        }
        let snapshot = dispatch
            .shader
            .as_ref()
            .ok_or("dispatch has no decoded compute shader")?;
        let evaluation = dispatch
            .shader_evaluation
            .as_ref()
            .ok_or("dispatch has no evaluated compute resources")?;
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

        let pipeline_key = compute_pipeline_key(dispatch, evaluation, &snapshot.program);
        if self.failed_compute_pipelines.contains(&pipeline_key) {
            return Err("compute pipeline previously failed to compile".to_owned());
        }
        if !self.compiled_compute_pipelines.contains_key(&pipeline_key) {
            let compiled = (|| {
                let options =
                    remu_shader_metal::MetalCompileOptions::compute(dispatch.launch.lds_size_bytes);
                let lowered =
                    remu_shader_metal::compile_msl(&snapshot.program, evaluation, &options)
                        .map_err(|error| format!("compute lowering: {error}"))?;
                if lowered.stage != remu_shader_metal::MetalShaderStage::Compute {
                    return Err("compute lowering returned a non-compute Metal stage".to_owned());
                }
                let library = compile_metal_compute_library(&device, &lowered.source)?;
                let function = library
                    .get_function(&lowered.entry_point, None)
                    .map_err(|error| format!("Metal compute function: {error}"))?;
                let pipeline = device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|error| format!("Metal compute pipeline: {error}"))?;
                Ok::<_, String>(CompiledComputePipeline {
                    pipeline,
                    function,
                    shader: lowered,
                })
            })();
            let compiled = match compiled {
                Ok(compiled) => compiled,
                Err(error) => {
                    self.failed_compute_pipelines.insert(pipeline_key.clone());
                    return Err(error);
                }
            };
            eprintln!(
                "[REMu][METAL] compiled Gen5 compute cs={:#x} MSL={} bytes local={:?} LDS={} bytes",
                dispatch.shader_address,
                compiled.shader.source.len(),
                dispatch.launch.threads_per_group,
                compiled.shader.bindings.threadgroup_memory_bytes,
            );
            self.compiled_compute_pipelines
                .insert(pipeline_key.clone(), compiled);
        }
        let compiled = self
            .compiled_compute_pipelines
            .get(&pipeline_key)
            .expect("compute pipeline was cached above");
        let pipeline = compiled.pipeline.clone();
        let function = compiled.function.clone();
        let bindings = compiled.shader.bindings.clone();
        let execution_width = u32::try_from(pipeline.thread_execution_width())
            .map_err(|_| "Metal compute execution width does not fit u32")?;
        let dynamic_lds = bindings.threadgroup_memory_bytes;
        validate_compute_launch_for_metal(
            dispatch.launch,
            execution_width,
            pipeline.max_total_threads_per_threadgroup(),
            device.max_threadgroup_memory_length(),
            pipeline.static_threadgroup_memory_length(),
            dynamic_lds,
        )?;

        for binding in evaluation
            .image_bindings
            .iter()
            .filter(|binding| is_storage_image_opcode(binding.opcode))
        {
            let descriptor = decode_image_descriptor(&binding.resource_descriptor)
                .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
            self.ensure_resident_texture(&device, descriptor.guest)?;
        }
        self.synchronize_sampled_aliases(&[evaluation])?;

        let buffers = upload_global_buffers(&device, evaluation)?;
        let textures = upload_image_textures(
            &device,
            evaluation,
            &mut self.resident_guest_textures,
            &mut self.sampled_texture_cache,
        )?;
        let (sampler_buffer, _samplers) = create_sampler_argument_buffer(
            &device,
            &function,
            bindings.sampler_argument_buffer_index,
            evaluation,
        )?;
        if self.open_guest_batch.is_none() {
            let command_buffer = queue.new_command_buffer().to_owned();
            command_buffer.set_label("REMu translated guest batch");
            self.open_guest_batch = Some(GuestCommandBatch {
                command_buffer,
                work: Vec::new(),
                global_writebacks: Vec::new(),
            });
        }
        {
            let batch = self
                .open_guest_batch
                .as_mut()
                .expect("guest command batch was opened above");
            batch.work.push(GuestBatchWork::Compute {
                shader: dispatch.shader_address,
            });
            collect_pending_global_writebacks(&buffers, evaluation, &mut batch.global_writebacks)?;
            let encoder = batch.command_buffer.new_compute_command_encoder();
            encoder.set_label(&format!("REMu Gen5 compute {:#x}", dispatch.shader_address));
            encoder.set_compute_pipeline_state(&pipeline);
            for (index, buffer) in buffers.iter().enumerate() {
                encoder.set_buffer(
                    u64::try_from(index).map_err(|_| "compute buffer index overflow")?,
                    Some(buffer),
                    0,
                );
            }
            encoder.set_bytes(
                u64::from(bindings.initial_scalar_buffer_index),
                u64::try_from(evaluation.initial_scalar_registers.len() * 4)
                    .map_err(|_| "compute scalar byte count overflow")?,
                evaluation
                    .initial_scalar_registers
                    .as_ptr()
                    .cast::<c_void>(),
            );
            for (index, texture) in textures.iter().enumerate() {
                encoder.set_texture(
                    u64::try_from(index).map_err(|_| "compute texture index overflow")?,
                    Some(&texture.texture),
                );
            }
            if let (Some(index), Some(buffer)) = (
                bindings.sampler_argument_buffer_index,
                sampler_buffer.as_ref(),
            ) {
                encoder.set_buffer(u64::from(index), Some(buffer), 0);
            }
            if dynamic_lds != 0 {
                encoder.set_threadgroup_memory_length(0, u64::from(dynamic_lds));
            }
            let local = MTLSize::new(
                u64::from(dispatch.launch.threads_per_group[0]),
                u64::from(dispatch.launch.threads_per_group[1]),
                u64::from(dispatch.launch.threads_per_group[2]),
            );
            match dispatch.launch.grid {
                ComputeGrid::Threadgroups(groups) => encoder.dispatch_thread_groups(
                    MTLSize::new(
                        u64::from(groups[0]),
                        u64::from(groups[1]),
                        u64::from(groups[2]),
                    ),
                    local,
                ),
                ComputeGrid::Threads(threads) => encoder
                    .dispatch_threads(MTLSize::new(threads[0], threads[1], threads[2]), local),
            }
            encoder.end_encoding();
        }
        for texture in &textures {
            if texture.writable {
                self.mark_resident_texture_written(texture.descriptor.guest);
            }
        }
        if self
            .open_guest_batch
            .as_ref()
            .is_some_and(|batch| batch.work.len() >= 32)
        {
            self.flush_guest_command_batch(false)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn execute_compiled_draw(&mut self, draw: &IndexedDraw) -> Result<(), String> {
        self.retire_completed_guest_batches();
        let started = Instant::now();
        let trace = draw.sequence <= 4;
        let targets = decode_render_targets(draw)?;
        let key = graphics_pipeline_key_from_targets(draw, &targets);
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
        let mut target_textures = Vec::with_capacity(targets.len());
        for target in &targets {
            let texture = if target.direct_scanout {
                self.ensure_direct_scanout_texture(&device, target.texture)
            } else {
                self.ensure_resident_texture(&device, target.texture)
                    .map_err(|error| {
                        format!(
                            "render target {} guest memory is unreadable: {error}",
                            target.guest_slot
                        )
                    })?
            };
            target_textures.push(texture);
        }

        for evaluation in [export_evaluation, pixel_evaluation] {
            for binding in evaluation
                .image_bindings
                .iter()
                .filter(|binding| is_storage_image_opcode(binding.opcode))
            {
                let descriptor = decode_image_descriptor(&binding.resource_descriptor)
                    .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
                for target in &targets {
                    if descriptor.guest != target.texture
                        && guest_texture_descriptors_overlap(descriptor.guest, target.texture)
                    {
                        return Err(format!(
                            "writable image at PC {:#x} aliases render target {} through an incompatible view",
                            binding.pc, target.guest_slot
                        ));
                    }
                }
                self.ensure_resident_texture(&device, descriptor.guest)?;
            }
        }
        self.synchronize_sampled_aliases(&[export_evaluation, pixel_evaluation])?;

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
        for (location, (texture, target)) in target_textures.iter().zip(&targets).enumerate() {
            let attachment = pass
                .color_attachments()
                .object_at(u64::try_from(location).map_err(|_| "color location overflow")?)
                .ok_or_else(|| format!("Metal render pass has no color attachment {location}"))?;
            attachment.set_texture(Some(texture));
            if target.direct_scanout {
                attachment.set_load_action(MTLLoadAction::Clear);
                attachment.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));
            } else {
                attachment.set_load_action(MTLLoadAction::Load);
            }
            attachment.set_store_action(MTLStoreAction::Store);
        }
        if self.open_guest_batch.is_none() {
            let command_buffer = queue.new_command_buffer().to_owned();
            command_buffer.set_label("REMu translated guest batch");
            self.open_guest_batch = Some(GuestCommandBatch {
                command_buffer,
                work: Vec::new(),
                global_writebacks: Vec::new(),
            });
        }
        {
            let batch = self
                .open_guest_batch
                .as_mut()
                .expect("guest command batch was opened above");
            batch.work.push(GuestBatchWork::Draw {
                export: key.export_shader_address,
                pixel: key.pixel_shader_address,
            });
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
            let primary_target = targets
                .first()
                .expect("render-target decoder requires at least one target")
                .texture;
            let scissor = decode_scissor(draw, primary_target.width, primary_target.height)?;
            let viewport =
                decode_viewport(draw, primary_target.width, primary_target.height, scissor);
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

        for target in &targets {
            if !target.direct_scanout {
                self.mark_resident_texture_written(target.texture);
            }
        }
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
            .is_some_and(|batch| batch.work.len() >= 32);
        if capture_due {
            self.wait_for_guest_command_batches()?;
            let primary_target = targets
                .first()
                .expect("render-target decoder requires at least one target")
                .texture;
            if matches!(
                primary_target.pixel_format,
                MTLPixelFormat::RGBA8Unorm
                    | MTLPixelFormat::RGBA8Unorm_sRGB
                    | MTLPixelFormat::BGRA8Unorm
                    | MTLPixelFormat::BGRA8Unorm_sRGB
            ) {
                let rendered =
                    read_texture_linear(&device, &queue, &target_textures[0], primary_target)?;
                self.capture_translated_frame_if_changed(primary_target, &rendered);
            }
        } else if batch_full {
            self.flush_guest_command_batch(false)?;
        }
        if trace {
            eprintln!(
                "[REMu][METAL] encoded Gen5 draw={} es={:#x} ps={:#x} in {:?}",
                draw.sequence,
                key.export_shader_address,
                key.pixel_shader_address,
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
        let capture = (self.capture_directory.is_some()
            && self
                .last_guest_capture
                .is_none_or(|last| last.elapsed() >= Duration::from_secs(1)))
        .then(|| {
            let descriptor = GuestTextureDescriptor {
                address: flip.address,
                width: display.width,
                height: display.height,
                pitch: display.width,
                tile_mode: 0,
                pixel_format: MTLPixelFormat::BGRA8Unorm,
                bytes_per_pixel: 4,
            };
            self.command_queue
                .as_ref()
                .ok_or_else(|| "Metal command queue is unavailable".to_owned())
                .and_then(|queue| read_texture_linear(device, queue, texture, descriptor))
                .map(|bytes| (descriptor, bytes))
        });
        self.uploaded_flip_sequence = flip.sequence;
        if flip.sequence == 1 {
            eprintln!(
                "[REMu][METAL] uploaded first guest display buffer format={:#x}",
                display.pixel_format
            );
        }
        if let Some(capture) = capture {
            match capture {
                Ok((descriptor, bgra)) => {
                    self.guest_bgra_bytes = bgra;
                    self.capture_guest_frame_if_changed(descriptor);
                }
                Err(error) => eprintln!("[REMu][METAL] could not read back flipped frame: {error}"),
            }
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
        native_bytes: &[u8],
    ) {
        if self.capture_directory.is_none()
            || self
                .last_guest_capture
                .is_some_and(|last| last.elapsed() < Duration::from_secs(1))
        {
            return;
        }
        self.guest_bgra_bytes.resize(native_bytes.len(), 0);
        match descriptor.pixel_format {
            MTLPixelFormat::RGBA8Unorm => {
                for (source, destination) in native_bytes
                    .chunks_exact(4)
                    .zip(self.guest_bgra_bytes.chunks_exact_mut(4))
                {
                    destination.copy_from_slice(&[source[2], source[1], source[0], source[3]]);
                }
            }
            MTLPixelFormat::RGBA8Unorm_sRGB => {
                for (source, destination) in native_bytes
                    .chunks_exact(4)
                    .zip(self.guest_bgra_bytes.chunks_exact_mut(4))
                {
                    destination.copy_from_slice(&[
                        srgb_unorm8_to_linear_unorm8(source[2]),
                        srgb_unorm8_to_linear_unorm8(source[1]),
                        srgb_unorm8_to_linear_unorm8(source[0]),
                        source[3],
                    ]);
                }
            }
            MTLPixelFormat::BGRA8Unorm => {
                self.guest_bgra_bytes.copy_from_slice(native_bytes);
            }
            MTLPixelFormat::BGRA8Unorm_sRGB => {
                for (source, destination) in native_bytes
                    .chunks_exact(4)
                    .zip(self.guest_bgra_bytes.chunks_exact_mut(4))
                {
                    destination.copy_from_slice(&[
                        srgb_unorm8_to_linear_unorm8(source[0]),
                        srgb_unorm8_to_linear_unorm8(source[1]),
                        srgb_unorm8_to_linear_unorm8(source[2]),
                        source[3],
                    ]);
                }
            }
            _ => return,
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
        self.upload_pending_frame();
        let (Some(layer), Some(queue)) = (self.layer.as_ref(), self.command_queue.as_ref()) else {
            return;
        };
        let Some(drawable) = layer.next_drawable() else {
            return;
        };
        let command_buffer = queue.new_command_buffer();
        command_buffer.set_label("REMu guest presentation");
        if let (Some(display), Some(texture)) = (self.display, self.frame_texture.as_ref()) {
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
            encoder.set_render_pipeline_state(&self.presentation.pipeline);
            encoder.set_viewport(presentation_viewport(
                display.width,
                display.height,
                destination_width,
                destination_height,
            ));
            encoder.set_fragment_texture(0, Some(texture));
            encoder.set_fragment_sampler_state(0, Some(&self.presentation.sampler));
            encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 3);
            encoder.end_encoding();
        } else {
            return;
        }
        command_buffer.present_drawable(drawable);
        command_buffer.commit();
        self.redraw_pending = false;
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn srgb_unorm8_to_linear_unorm8(encoded: u8) -> u8 {
    static LUT: OnceLock<[u8; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        std::array::from_fn(|encoded| {
            let encoded =
                f64::from(u8::try_from(encoded).expect("sRGB table index fits u8")) / 255.0;
            let linear = if encoded <= 0.040_45 {
                encoded / 12.92
            } else {
                ((encoded + 0.055) / 1.055).powf(2.4)
            };
            // Both transfer-function branches map the unorm input into 0..=1, so rounding the
            // scaled result cannot produce a negative value or exceed u8.
            (linear * 255.0).round() as u8
        })
    })[usize::from(encoded)]
}

fn validate_compute_launch_for_metal(
    launch: remu_gpu::ComputeLaunch,
    execution_width: u32,
    maximum_threads_per_group: u64,
    maximum_threadgroup_memory_bytes: u64,
    static_threadgroup_memory_bytes: u64,
    dynamic_threadgroup_memory_bytes: u32,
) -> Result<(), String> {
    if launch.base_group != [0; 3] {
        return Err(format!(
            "Metal compute does not yet support nonzero base groups {:?}",
            launch.base_group
        ));
    }
    if execution_width != launch.wave_lane_count {
        return Err(format!(
            "guest wave{} cannot execute on Metal SIMD{} without changing wave semantics",
            launch.wave_lane_count, execution_width
        ));
    }
    let threadgroup_threads = launch.threads_per_group_total();
    if threadgroup_threads == 0 || threadgroup_threads > maximum_threads_per_group {
        return Err(format!(
            "guest compute threadgroup {:?} has {threadgroup_threads} threads, Metal permits {maximum_threads_per_group}",
            launch.threads_per_group
        ));
    }
    if dynamic_threadgroup_memory_bytes != 0
        && dynamic_threadgroup_memory_bytes != launch.lds_size_bytes
    {
        return Err(format!(
            "Metal shader requests {dynamic_threadgroup_memory_bytes} bytes of LDS, dispatch declares {}",
            launch.lds_size_bytes
        ));
    }
    let total_threadgroup_memory = static_threadgroup_memory_bytes
        .checked_add(u64::from(dynamic_threadgroup_memory_bytes))
        .ok_or("Metal threadgroup memory size overflows")?;
    if total_threadgroup_memory > maximum_threadgroup_memory_bytes {
        return Err(format!(
            "compute pipeline needs {total_threadgroup_memory} bytes of threadgroup memory, this Metal device permits {maximum_threadgroup_memory_bytes}"
        ));
    }
    Ok(())
}

fn create_presentation_resources(device: &Device) -> Result<PresentationResources, String> {
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
        .set_pixel_format(MTLPixelFormat::BGRA8Unorm);
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

fn compile_metal_libraries(
    device: &Device,
    vertex_source: &str,
    pixel_source: &str,
) -> Result<(Library, Library), String> {
    if std::env::var_os("REMU_METAL_COMPILER").is_some() {
        return compile_metal_libraries_with_tools(device, vertex_source, pixel_source);
    }

    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(false);
    let vertex = device
        .new_library_with_source(vertex_source, &compile_options)
        .map_err(|error| format!("Metal vertex compilation: {error}"))?;
    let pixel = device
        .new_library_with_source(pixel_source, &compile_options)
        .map_err(|error| format!("Metal pixel compilation: {error}"))?;
    Ok((vertex, pixel))
}

fn compile_metal_compute_library(device: &Device, source: &str) -> Result<Library, String> {
    if std::env::var_os("REMU_METAL_COMPILER").is_some() {
        return compile_metal_library_with_tools(device, "compute", source);
    }
    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(false);
    device
        .new_library_with_source(source, &compile_options)
        .map_err(|error| format!("Metal compute compilation: {error}"))
}

fn compile_metal_libraries_with_tools(
    device: &Device,
    vertex_source: &str,
    pixel_source: &str,
) -> Result<(Library, Library), String> {
    Ok((
        compile_metal_library_with_tools(device, "vertex", vertex_source)?,
        compile_metal_library_with_tools(device, "pixel", pixel_source)?,
    ))
}

fn compile_metal_library_with_tools(
    device: &Device,
    stage: &str,
    source: &str,
) -> Result<Library, String> {
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
    let source_path = directory.0.join(format!("{stage}.metal"));
    let air_path = directory.0.join(format!("{stage}.air"));
    let library_path = directory.0.join(format!("{stage}.metallib"));
    std::fs::write(&source_path, source)
        .map_err(|error| format!("could not write {}: {error}", source_path.display()))?;
    run_metal_tool(
        Command::new(&compiler)
            .current_dir(&directory.0)
            .arg("-fmetal-math-mode=safe")
            .arg("-c")
            .arg(&source_path)
            .arg("-o")
            .arg(&air_path),
        &format!("{stage} compilation"),
    )?;
    run_metal_tool(
        Command::new(&linker)
            .current_dir(&directory.0)
            .arg(&air_path)
            .arg("-o")
            .arg(&library_path),
        &format!("{stage} library link"),
    )?;
    device
        .new_library_with_file(&library_path)
        .map_err(|error| format!("could not load {}: {error}", library_path.display()))
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

fn decode_guest_texture_linear_from_raw(
    descriptor: GuestTextureDescriptor,
    source: &[u8],
) -> Result<Vec<u8>, String> {
    let output_length = u64::from(descriptor.width)
        .checked_mul(u64::from(descriptor.height))
        .and_then(|pixels| pixels.checked_mul(u64::from(descriptor.bytes_per_pixel)))
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
            .checked_mul(
                usize::try_from(descriptor.bytes_per_pixel)
                    .map_err(|_| "texture bytes per pixel does not fit usize")?,
            )
            .ok_or("source row byte count overflows")?;
        let output_row_bytes = width
            .checked_mul(
                usize::try_from(descriptor.bytes_per_pixel)
                    .map_err(|_| "texture bytes per pixel does not fit usize")?,
            )
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
fn snapshot_guest_texture_linear(descriptor: GuestTextureDescriptor) -> Result<Vec<u8>, String> {
    let (source_address, source_length) = guest_texture_raw_range(descriptor)?;
    // SAFETY: See snapshot_guest_texture_raw; decoding completes before this function returns.
    let source = unsafe { std::slice::from_raw_parts(source_address as *const u8, source_length) };
    decode_guest_texture_linear_from_raw(descriptor, source)
}

fn write_guest_texture_linear(
    descriptor: GuestTextureDescriptor,
    linear: &[u8],
) -> Result<(), String> {
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
        .and_then(|pixels| pixels.checked_mul(usize::try_from(descriptor.bytes_per_pixel).ok()?))
        .ok_or("guest texture dimensions overflow")?;
    if linear.len() != required {
        return Err(format!(
            "tight guest image has {} bytes, expected {required}",
            linear.len()
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
            .checked_mul(
                usize::try_from(descriptor.bytes_per_pixel)
                    .map_err(|_| "texture bytes per pixel does not fit usize")?,
            )
            .ok_or("destination row byte count overflows")?;
        let source_row_bytes = width
            .checked_mul(
                usize::try_from(descriptor.bytes_per_pixel)
                    .map_err(|_| "texture bytes per pixel does not fit usize")?,
            )
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
                .copy_from_slice(&linear[source_offset..source_offset + source_row_bytes]);
        }
    } else {
        copy_linear_to_mode27(linear, descriptor, destination)?;
    }
    Ok(())
}

fn create_guest_texture(
    device: &Device,
    guest: GuestTextureDescriptor,
    usage: MTLTextureUsage,
    linear: &[u8],
) -> Result<Texture, String> {
    let required = u64::from(guest.width)
        .checked_mul(u64::from(guest.height))
        .and_then(|pixels| pixels.checked_mul(u64::from(guest.bytes_per_pixel)))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or("Metal texture dimensions overflow")?;
    if guest.width == 0 || guest.height == 0 || linear.len() != required {
        return Err(format!(
            "invalid guest texture upload {}x{} format {:?} with {} bytes (expected {required})",
            guest.width,
            guest.height,
            guest.pixel_format,
            linear.len()
        ));
    }
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(guest.pixel_format);
    descriptor.set_width(u64::from(guest.width));
    descriptor.set_height(u64::from(guest.height));
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(usage);
    let texture = device.new_texture(&descriptor);
    texture.replace_region(
        MTLRegion::new_2d(0, 0, u64::from(guest.width), u64::from(guest.height)),
        0,
        linear.as_ptr().cast::<c_void>(),
        u64::from(guest.width) * u64::from(guest.bytes_per_pixel),
    );
    Ok(texture)
}

fn validate_texture_readback_source(
    texture: &Texture,
    descriptor: GuestTextureDescriptor,
) -> Result<(), String> {
    if texture.framebuffer_only() {
        return Err("Metal framebuffer-only textures cannot be copied for readback".to_owned());
    }
    if texture.storage_mode() == MTLStorageMode::Memoryless {
        return Err("Metal memoryless textures have no stored contents to read back".to_owned());
    }
    if texture.sample_count() != 1 {
        return Err(format!(
            "Metal texture readback requires the single-sample resolve target, got {} samples",
            texture.sample_count()
        ));
    }
    if texture.texture_type() != MTLTextureType::D2 {
        return Err(format!(
            "Metal texture readback requires a 2D single-slice source, got {:?}",
            texture.texture_type()
        ));
    }
    if texture.width() != u64::from(descriptor.width)
        || texture.height() != u64::from(descriptor.height)
    {
        return Err(format!(
            "Metal texture readback extent mismatch: texture={}x{} descriptor={}x{}",
            texture.width(),
            texture.height(),
            descriptor.width,
            descriptor.height
        ));
    }
    if texture.pixel_format() != descriptor.pixel_format {
        return Err(format!(
            "Metal texture readback format mismatch: texture={:?} descriptor={:?}",
            texture.pixel_format(),
            descriptor.pixel_format
        ));
    }
    let expected_bytes =
        metal_uncompressed_bytes_per_pixel(descriptor.pixel_format).ok_or_else(|| {
            format!(
                "Metal texture readback has no exact uncompressed layout for {:?}",
                descriptor.pixel_format
            )
        })?;
    if descriptor.bytes_per_pixel != expected_bytes {
        return Err(format!(
            "Metal texture readback format {:?} requires {expected_bytes} bytes per pixel, got {}",
            descriptor.pixel_format, descriptor.bytes_per_pixel
        ));
    }
    Ok(())
}

fn metal_uncompressed_bytes_per_pixel(format: MTLPixelFormat) -> Option<u32> {
    match format {
        MTLPixelFormat::R8Unorm
        | MTLPixelFormat::R8Snorm
        | MTLPixelFormat::R8Uint
        | MTLPixelFormat::R8Sint
        | MTLPixelFormat::R8Unorm_sRGB
        | MTLPixelFormat::Stencil8 => Some(1),
        MTLPixelFormat::R16Unorm
        | MTLPixelFormat::R16Snorm
        | MTLPixelFormat::R16Uint
        | MTLPixelFormat::R16Sint
        | MTLPixelFormat::R16Float
        | MTLPixelFormat::RG8Unorm
        | MTLPixelFormat::RG8Snorm
        | MTLPixelFormat::RG8Uint
        | MTLPixelFormat::RG8Sint
        | MTLPixelFormat::RG8Unorm_sRGB
        | MTLPixelFormat::Depth16Unorm => Some(2),
        MTLPixelFormat::R32Uint
        | MTLPixelFormat::R32Sint
        | MTLPixelFormat::R32Float
        | MTLPixelFormat::RG16Unorm
        | MTLPixelFormat::RG16Snorm
        | MTLPixelFormat::RG16Uint
        | MTLPixelFormat::RG16Sint
        | MTLPixelFormat::RG16Float
        | MTLPixelFormat::RG11B10Float
        | MTLPixelFormat::RGB10A2Unorm
        | MTLPixelFormat::BGR10A2Unorm
        | MTLPixelFormat::RGBA8Unorm
        | MTLPixelFormat::RGBA8Snorm
        | MTLPixelFormat::RGBA8Uint
        | MTLPixelFormat::RGBA8Sint
        | MTLPixelFormat::RGBA8Unorm_sRGB
        | MTLPixelFormat::BGRA8Unorm
        | MTLPixelFormat::BGRA8Unorm_sRGB
        | MTLPixelFormat::RGB9E5Float
        | MTLPixelFormat::Depth32Float => Some(4),
        MTLPixelFormat::RG32Uint
        | MTLPixelFormat::RG32Sint
        | MTLPixelFormat::RG32Float
        | MTLPixelFormat::RGBA16Unorm
        | MTLPixelFormat::RGBA16Snorm
        | MTLPixelFormat::RGBA16Uint
        | MTLPixelFormat::RGBA16Sint
        | MTLPixelFormat::RGBA16Float => Some(8),
        MTLPixelFormat::RGBA32Uint | MTLPixelFormat::RGBA32Sint | MTLPixelFormat::RGBA32Float => {
            Some(16)
        }
        _ => None,
    }
}

fn read_texture_linear(
    device: &Device,
    queue: &metal::CommandQueue,
    texture: &Texture,
    descriptor: GuestTextureDescriptor,
) -> Result<Vec<u8>, String> {
    PendingTextureReadback::encode(device, queue, texture, descriptor)?.finish()
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
    let linear = decode_guest_texture_linear_from_raw(descriptor, &raw)?;
    let texture = create_guest_texture(
        device,
        descriptor,
        MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
        &linear,
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
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
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
            let writable = is_writable_image_opcode(binding.opcode);
            let storage = is_storage_image_opcode(binding.opcode);
            if storage && !matches!(descriptor.destination_select, 0 | 0xfac) {
                return Err(format!(
                    "storage image at PC {:#x} has non-identity destination select {:#x}",
                    binding.pc, descriptor.destination_select
                ));
            }
            let identity_select = matches!(descriptor.destination_select, 0 | 0xfac);
            let texture = if storage {
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
                    let mut rgba = decode_guest_texture_linear_from_raw(descriptor.guest, &raw)
                        .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
                    apply_image_destination_select(&mut rgba, descriptor.destination_select)
                        .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
                    let texture = create_guest_texture(
                        device,
                        descriptor.guest,
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

fn decode_render_target_format(
    data_format: u32,
    number_type: u32,
    component_swap: u32,
) -> Result<RenderTargetFormat, String> {
    // Mesa's ac_get_cb_format/ac_get_cb_number_type tables establish the valid GFX10
    // combinations. Metal has direct, bit-compatible render-target formats for this cohort.
    // Scaled number types have no Metal color-attachment equivalent and are rejected.
    let (pixel_format, bytes_per_pixel) = match (data_format, number_type, component_swap) {
        (1, 0, 0) => (MTLPixelFormat::R8Unorm, 1),
        (1, 1, 0) => (MTLPixelFormat::R8Snorm, 1),
        (1, 4, 0) => (MTLPixelFormat::R8Uint, 1),
        (1, 5, 0) => (MTLPixelFormat::R8Sint, 1),
        (1, 6, 0) => (MTLPixelFormat::R8Unorm_sRGB, 1),
        (2, 0, 0) => (MTLPixelFormat::R16Unorm, 2),
        (2, 1, 0) => (MTLPixelFormat::R16Snorm, 2),
        (2, 4, 0) => (MTLPixelFormat::R16Uint, 2),
        (2, 5, 0) => (MTLPixelFormat::R16Sint, 2),
        (2, 7, 0) => (MTLPixelFormat::R16Float, 2),
        (3, 0, 0) => (MTLPixelFormat::RG8Unorm, 2),
        (3, 1, 0) => (MTLPixelFormat::RG8Snorm, 2),
        (3, 4, 0) => (MTLPixelFormat::RG8Uint, 2),
        (3, 5, 0) => (MTLPixelFormat::RG8Sint, 2),
        (3, 6, 0) => (MTLPixelFormat::RG8Unorm_sRGB, 2),
        (4, 4, 0) => (MTLPixelFormat::R32Uint, 4),
        (4, 5, 0) => (MTLPixelFormat::R32Sint, 4),
        (4, 7, 0) => (MTLPixelFormat::R32Float, 4),
        (5, 0, 0) => (MTLPixelFormat::RG16Unorm, 4),
        (5, 1, 0) => (MTLPixelFormat::RG16Snorm, 4),
        (5, 4, 0) => (MTLPixelFormat::RG16Uint, 4),
        (5, 5, 0) => (MTLPixelFormat::RG16Sint, 4),
        (5, 7, 0) => (MTLPixelFormat::RG16Float, 4),
        (6 | 7, 7, 0) => (MTLPixelFormat::RG11B10Float, 4),
        (8, 0, 0) => (MTLPixelFormat::RGB10A2Unorm, 4),
        (9, 0, 0) => (MTLPixelFormat::BGR10A2Unorm, 4),
        (10, 0, 0) => (MTLPixelFormat::RGBA8Unorm, 4),
        (10, 0, 1) => (MTLPixelFormat::BGRA8Unorm, 4),
        (10, 1, 0) => (MTLPixelFormat::RGBA8Snorm, 4),
        (10, 4, 0) => (MTLPixelFormat::RGBA8Uint, 4),
        (10, 5, 0) => (MTLPixelFormat::RGBA8Sint, 4),
        (10, 6, 0) => (MTLPixelFormat::RGBA8Unorm_sRGB, 4),
        (10, 6, 1) => (MTLPixelFormat::BGRA8Unorm_sRGB, 4),
        (11, 4, 0) => (MTLPixelFormat::RG32Uint, 8),
        (11, 5, 0) => (MTLPixelFormat::RG32Sint, 8),
        (11, 7, 0) => (MTLPixelFormat::RG32Float, 8),
        (12, 0, 0) => (MTLPixelFormat::RGBA16Unorm, 8),
        (12, 1, 0) => (MTLPixelFormat::RGBA16Snorm, 8),
        (12, 4, 0) => (MTLPixelFormat::RGBA16Uint, 8),
        (12, 5, 0) => (MTLPixelFormat::RGBA16Sint, 8),
        (12, 7, 0) => (MTLPixelFormat::RGBA16Float, 8),
        (14, 4, 0) => (MTLPixelFormat::RGBA32Uint, 16),
        (14, 5, 0) => (MTLPixelFormat::RGBA32Sint, 16),
        (14, 7, 0) => (MTLPixelFormat::RGBA32Float, 16),
        (24, 7, 0) => (MTLPixelFormat::RGB9E5Float, 4),
        _ => {
            return Err(format!(
                "unsupported GFX10 color format={data_format} number_type={number_type} component_swap={component_swap}"
            ));
        }
    };
    let output_kind = match number_type {
        4 => PixelOutputKind::Uint,
        5 => PixelOutputKind::Sint,
        _ => PixelOutputKind::Float,
    };
    Ok(RenderTargetFormat {
        pixel_format,
        output_kind,
        bytes_per_pixel,
    })
}

#[allow(clippy::too_many_arguments)]
fn decode_render_target_state(
    guest_slot: u32,
    address: u64,
    view: u32,
    info: u32,
    attrib: u32,
    attrib2: u32,
    attrib3: u32,
    write_mask: u32,
) -> Result<DecodedRenderTarget, String> {
    let width = ((attrib2 >> 14) & 0x3fff)
        .checked_add(1)
        .ok_or("render-target width overflows")?;
    let height = (attrib2 & 0x3fff)
        .checked_add(1)
        .ok_or("render-target height overflows")?;
    if address < 0x10_000 || width > 16_384 || height > 16_384 {
        return Err(format!(
            "invalid color target {guest_slot} address={address:#x} extent={width}x{height}"
        ));
    }

    let endian = info & 0x3;
    let fast_clear = (info >> 13) & 1;
    let fmask_compression = (info >> 14) & 1;
    let dcc_enable = (info >> 28) & 1;
    if endian != 0 || fast_clear != 0 || fmask_compression != 0 || dcc_enable != 0 {
        return Err(format!(
            "color target {guest_slot} requires unsupported metadata/endian state endian={endian} fast_clear={fast_clear} fmask={fmask_compression} dcc={dcc_enable}"
        ));
    }
    let sample_log2 = (attrib >> 12) & 0x7;
    let fragment_log2 = (attrib >> 15) & 0x3;
    if sample_log2 != 0 || fragment_log2 != 0 {
        return Err(format!(
            "color target {guest_slot} is multisampled (samples_log2={sample_log2}, fragments_log2={fragment_log2})"
        ));
    }
    let slice_start = view & 0x1fff;
    let slice_max = (view >> 13) & 0x1fff;
    let mip_level = (view >> 26) & 0xf;
    let max_mip = (attrib2 >> 28) & 0xf;
    let mip0_depth = attrib3 & 0x1fff;
    if slice_start != 0 || slice_max != 0 || mip_level != 0 || max_mip != 0 || mip0_depth != 0 {
        return Err(format!(
            "color target {guest_slot} requires unsupported slice/mip view slices={slice_start}..={slice_max} mip={mip_level}/{max_mip} depth={mip0_depth}"
        ));
    }
    let tile_mode = (attrib3 >> 14) & 0x1f;
    if !matches!(tile_mode, 0 | 27) {
        return Err(format!(
            "color target {guest_slot} uses unsupported GFX10 swizzle mode {tile_mode}"
        ));
    }
    let resource_type = (attrib3 >> 24) & 0x3;
    let resource_level = (attrib3 >> 27) & 0x7;
    let non_block_compressed_view = (attrib3 >> 31) & 1;
    if !matches!(resource_type, 0 | 1)
        || (resource_type == 0 && height != 1)
        || resource_level != 0
        || non_block_compressed_view != 0
    {
        return Err(format!(
            "color target {guest_slot} requires unsupported resource state type={resource_type} level={resource_level} nbc={non_block_compressed_view} extent={width}x{height}"
        ));
    }
    let format =
        decode_render_target_format((info >> 2) & 0x1f, (info >> 8) & 0x7, (info >> 11) & 0x3)?;
    let pitch = if tile_mode == 0 {
        effective_linear_pitch(width, format.bytes_per_pixel)
            .ok_or("linear render-target pitch overflows")?
    } else {
        width
    };
    Ok(DecodedRenderTarget {
        guest_slot,
        texture: GuestTextureDescriptor {
            address,
            width,
            height,
            pitch,
            tile_mode,
            pixel_format: format.pixel_format,
            bytes_per_pixel: format.bytes_per_pixel,
        },
        output_kind: format.output_kind,
        write_mask,
        direct_scanout: false,
    })
}

fn decode_render_targets(draw: &IndexedDraw) -> Result<Vec<DecodedRenderTarget>, String> {
    let pixel = draw
        .pixel_shader
        .as_ref()
        .ok_or("draw has no decoded pixel shader")?;
    let export_masks = pixel.program.pixel_color_export_masks();
    if let Some(target) = draw.scanout_target {
        let shader_write_mask = (export_masks >> (target.export_slot * 4)) & 0xf;
        if shader_write_mask == 0 || shader_write_mask & target.write_mask == 0 {
            return Err(format!(
                "scanout target slot {} has no matching pixel export ({export_masks:#010x})",
                target.export_slot
            ));
        }
        return decode_scanout_target(target).map(|target| vec![target]);
    }
    decode_render_targets_from_registers(&draw.context_registers, export_masks)
}

fn decode_scanout_target(target: GpuScanoutTarget) -> Result<DecodedRenderTarget, String> {
    if let Some(surface) = target.color_surface {
        let mut decoded = decode_render_target_state(
            target.export_slot,
            surface.address,
            surface.view,
            surface.info,
            surface.attrib,
            surface.attrib2,
            surface.attrib3,
            target.write_mask,
        )?;
        if decoded.texture.address != target.address
            || decoded.texture.width != target.width
            || decoded.texture.height != target.height
        {
            return Err(
                "programmed scanout surface disagrees with VideoOut registration".to_owned(),
            );
        }
        decoded.guest_slot = target.export_slot;
        return Ok(decoded);
    }
    let (pixel_format, bytes_per_pixel) = decode_video_out_pixel_format(target.pixel_format)?;
    if target.address < 0x10_000
        || target.width == 0
        || target.height == 0
        || target.width > 16_384
        || target.height > 16_384
        || target.pitch < target.width
    {
        return Err(format!(
            "invalid direct scanout address={:#x} extent={}x{} pitch={}",
            target.address, target.width, target.height, target.pitch
        ));
    }
    Ok(DecodedRenderTarget {
        guest_slot: target.export_slot,
        texture: GuestTextureDescriptor {
            address: target.address,
            width: target.width,
            height: target.height,
            pitch: target.pitch,
            tile_mode: 0,
            pixel_format,
            bytes_per_pixel,
        },
        output_kind: PixelOutputKind::Float,
        write_mask: target.write_mask,
        direct_scanout: true,
    })
}

fn decode_video_out_pixel_format(pixel_format: u64) -> Result<(MTLPixelFormat, u32), String> {
    const A8_R8_G8_B8_SRGB: u64 = 0x8000_0000;
    const A8_B8_G8_R8_SRGB: u64 = 0x8000_2200;
    const A2_R10_G10_B10: u64 = 0x8806_0000;
    const A2_R10_G10_B10_SRGB: u64 = 0x8800_0000;
    const A2_R10_G10_B10_BT2020_PQ: u64 = 0x8874_0000;
    const F2_R8_G8_B8_A8_SRGB: u64 = 0x8000_0000_2200_0000;
    const F2_B8_G8_R8_A8_SRGB: u64 = 0x8000_0000_0000_0000;
    const F2_R10_G10_B10_A2: u64 = 0x8100_0006_2200_0000;
    const F2_B10_G10_R10_A2: u64 = 0x8100_0006_0000_0000;
    const F2_R10_G10_B10_A2_SRGB: u64 = 0x8100_0000_2200_0000;
    const F2_B10_G10_R10_A2_SRGB: u64 = 0x8100_0000_0000_0000;
    const F2_R10_G10_B10_A2_BT2100_PQ: u64 = 0x8100_0704_2200_0000;
    const F2_B10_G10_R10_A2_BT2100_PQ: u64 = 0x8100_0704_0000_0000;
    let pixel_format = match pixel_format {
        A8_R8_G8_B8_SRGB
        | A8_B8_G8_R8_SRGB
        | A2_R10_G10_B10
        | A2_R10_G10_B10_SRGB
        | A2_R10_G10_B10_BT2020_PQ
        | F2_R8_G8_B8_A8_SRGB
        | F2_B8_G8_R8_A8_SRGB
        | F2_R10_G10_B10_A2
        | F2_B10_G10_R10_A2
        | F2_R10_G10_B10_A2_SRGB
        | F2_B10_G10_R10_A2_SRGB
        | F2_R10_G10_B10_A2_BT2100_PQ
        | F2_B10_G10_R10_A2_BT2100_PQ => pixel_format,
        _ => {
            return Err(format!(
                "unsupported VideoOut pixel format {pixel_format:#018x}"
            ));
        }
    };
    let metal = match pixel_format {
        A8_B8_G8_R8_SRGB | F2_R8_G8_B8_A8_SRGB => MTLPixelFormat::RGBA8Unorm_sRGB,
        A8_R8_G8_B8_SRGB | F2_B8_G8_R8_A8_SRGB => MTLPixelFormat::BGRA8Unorm_sRGB,
        F2_R10_G10_B10_A2 | F2_R10_G10_B10_A2_SRGB | F2_R10_G10_B10_A2_BT2100_PQ => {
            MTLPixelFormat::RGB10A2Unorm
        }
        A2_R10_G10_B10
        | A2_R10_G10_B10_SRGB
        | A2_R10_G10_B10_BT2020_PQ
        | F2_B10_G10_R10_A2
        | F2_B10_G10_R10_A2_SRGB
        | F2_B10_G10_R10_A2_BT2100_PQ => MTLPixelFormat::BGR10A2Unorm,
        _ => unreachable!("VideoOut format cohort was exhaustively matched"),
    };
    Ok((metal, 4))
}

#[allow(clippy::too_many_lines)]
fn decode_render_targets_from_registers(
    context_registers: &[GpuRegister],
    export_masks: u32,
) -> Result<Vec<DecodedRenderTarget>, String> {
    let target_mask = register_value(context_registers, CB_TARGET_MASK)
        .ok_or("draw has no CB_TARGET_MASK register")?;
    let mut targets = Vec::new();
    for guest_slot in 0..COLOR_TARGET_COUNT {
        let target_write_mask = (target_mask >> (guest_slot * 4)) & 0xf;
        let shader_write_mask = (export_masks >> (guest_slot * 4)) & 0xf;
        let write_mask = target_write_mask & shader_write_mask;
        if write_mask == 0 {
            continue;
        }
        let primary = guest_slot * CB_COLOR_REGISTER_STRIDE;
        let required = |offset, name| {
            register_value(context_registers, offset).ok_or_else(|| {
                format!("color target {guest_slot} has no {name} register ({offset:#x})")
            })
        };
        let base = required(CB_COLOR0_BASE + primary, "CB_COLOR_BASE")?;
        let base_ext = required(CB_COLOR0_BASE_EXT + guest_slot, "CB_COLOR_BASE_EXT")?;
        let view = required(CB_COLOR0_VIEW + primary, "CB_COLOR_VIEW")?;
        let info = required(CB_COLOR0_INFO + primary, "CB_COLOR_INFO")?;
        let attrib = required(CB_COLOR0_ATTRIB + primary, "CB_COLOR_ATTRIB")?;
        let attrib2 = required(CB_COLOR0_ATTRIB2 + guest_slot, "CB_COLOR_ATTRIB2")?;
        let attrib3 = required(CB_COLOR0_ATTRIB3 + guest_slot, "CB_COLOR_ATTRIB3")?;

        let address = (u64::from(base_ext & 0xff) << 40) | (u64::from(base) << 8);
        targets.push(decode_render_target_state(
            guest_slot, address, view, info, attrib, attrib2, attrib3, write_mask,
        )?);
    }
    if targets.is_empty() {
        return Err(format!(
            "draw has no color target enabled by both CB_TARGET_MASK={target_mask:#010x} and pixel exports={export_masks:#010x}"
        ));
    }
    Ok(targets)
}

fn graphics_pipeline_key(draw: &IndexedDraw) -> Result<GraphicsPipelineKey, String> {
    let targets = decode_render_targets(draw)?;
    Ok(graphics_pipeline_key_from_targets(draw, &targets))
}

fn graphics_pipeline_key_from_targets(
    draw: &IndexedDraw,
    targets: &[DecodedRenderTarget],
) -> GraphicsPipelineKey {
    GraphicsPipelineKey {
        export_shader_address: draw.export_shader_address,
        pixel_shader_address: draw.pixel_shader_address,
        targets: targets
            .iter()
            .map(|target| RenderTargetPipelineState {
                guest_slot: target.guest_slot,
                pixel_format: target.texture.pixel_format,
                output_kind: target.output_kind,
                write_mask: target.write_mask,
            })
            .collect(),
    }
}

fn metal_color_write_mask(guest_mask: u32) -> MTLColorWriteMask {
    // GFX10 CB_TARGET_MASK is R=1,G=2,B=4,A=8; MTLColorWriteMask uses the reverse bits.
    let metal_mask = u64::from(guest_mask & 1 != 0) << 3
        | u64::from(guest_mask & 2 != 0) << 2
        | u64::from(guest_mask & 4 != 0) << 1
        | u64::from(guest_mask & 8 != 0);
    MTLColorWriteMask::from_bits_retain(metal_mask)
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
    let bytes_per_pixel = usize::try_from(descriptor.bytes_per_pixel)
        .map_err(|_| "texture bytes per pixel does not fit usize")?;
    let (block_width, block_height) = mode27_block_dimensions(bytes_per_pixel)
        .ok_or("mode 27 requires a 1, 2, 4, 8, or 16-byte pixel format")?;
    let expected_source = usize::try_from(
        guest_texture_byte_count(descriptor).ok_or("mode 27 texture byte count overflows")?,
    )
    .map_err(|_| "mode 27 texture byte count does not fit usize")?;
    let expected_destination = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(bytes_per_pixel))
        .ok_or("mode 27 linear byte count overflows")?;
    if source.len() != expected_source || destination.len() != expected_destination {
        return Err(format!(
            "mode 27 copy has source/destination lengths {}/{}, expected {expected_source}/{expected_destination}",
            source.len(),
            destination.len(),
        ));
    }
    let blocks_per_row = width.div_ceil(block_width);
    for block_y in 0..height.div_ceil(block_height) {
        let rows = (height - block_y * block_height).min(block_height);
        for block_x in 0..blocks_per_row {
            let columns = (width - block_x * block_width).min(block_width);
            let block_base = (block_y * blocks_per_row + block_x) * 65_536;
            let offsets = mode27_block_byte_offsets(
                block_x,
                block_y,
                bytes_per_pixel,
                block_width,
                block_height,
            );
            for local_y in 0..rows {
                let destination_row = ((block_y * block_height + local_y) * width
                    + block_x * block_width)
                    * bytes_per_pixel;
                let offset_row = local_y * block_width;
                for local_x in 0..columns {
                    let source_offset = block_base + usize::from(offsets[offset_row + local_x]);
                    let destination_offset = destination_row + local_x * bytes_per_pixel;
                    destination[destination_offset..destination_offset + bytes_per_pixel]
                        .copy_from_slice(&source[source_offset..source_offset + bytes_per_pixel]);
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
    let bytes_per_pixel = usize::try_from(descriptor.bytes_per_pixel)
        .map_err(|_| "texture bytes per pixel does not fit usize")?;
    let (block_width, block_height) = mode27_block_dimensions(bytes_per_pixel)
        .ok_or("mode 27 requires a 1, 2, 4, 8, or 16-byte pixel format")?;
    let expected_source = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(bytes_per_pixel))
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
    let blocks_per_row = width.div_ceil(block_width);
    for block_y in 0..height.div_ceil(block_height) {
        let rows = (height - block_y * block_height).min(block_height);
        for block_x in 0..blocks_per_row {
            let columns = (width - block_x * block_width).min(block_width);
            let block_base = (block_y * blocks_per_row + block_x) * 65_536;
            let offsets = mode27_block_byte_offsets(
                block_x,
                block_y,
                bytes_per_pixel,
                block_width,
                block_height,
            );
            for local_y in 0..rows {
                let source_row = ((block_y * block_height + local_y) * width
                    + block_x * block_width)
                    * bytes_per_pixel;
                let offset_row = local_y * block_width;
                for local_x in 0..columns {
                    let source_offset = source_row + local_x * bytes_per_pixel;
                    let destination_offset =
                        block_base + usize::from(offsets[offset_row + local_x]);
                    destination[destination_offset..destination_offset + bytes_per_pixel]
                        .copy_from_slice(&source[source_offset..source_offset + bytes_per_pixel]);
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
    effective_linear_pitch(pitch, 4)
}

fn effective_linear_pitch(width: u32, bytes_per_pixel: u32) -> Option<u32> {
    if !matches!(bytes_per_pixel, 1 | 2 | 4 | 8 | 16) {
        return None;
    }
    let alignment = 256 / bytes_per_pixel;
    width
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
}

fn guest_texture_byte_count(descriptor: GuestTextureDescriptor) -> Option<u64> {
    if descriptor.tile_mode == 27 {
        let (block_width, block_height) =
            mode27_block_dimensions(usize::try_from(descriptor.bytes_per_pixel).ok()?)?;
        let blocks_wide = u64::from(descriptor.width).div_ceil(u64::try_from(block_width).ok()?);
        let blocks_high = u64::from(descriptor.height).div_ceil(u64::try_from(block_height).ok()?);
        blocks_wide.checked_mul(blocks_high)?.checked_mul(65_536)
    } else {
        u64::from(descriptor.pitch)
            .checked_mul(u64::from(descriptor.height))?
            .checked_mul(u64::from(descriptor.bytes_per_pixel))
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
    let offsets = mode27_block_byte_offsets(block_x, block_y, 4, 128, 128);
    block_index * 65_536 + usize::from(offsets[(y & 127) * 128 + (x & 127)])
}

fn mode27_block_byte_offsets(
    block_x: usize,
    block_y: usize,
    bytes_per_pixel: usize,
    block_width: usize,
    block_height: usize,
) -> &'static [u16] {
    static OFFSETS: OnceLock<[Box<[u16]>; 5]> = OnceLock::new();
    let offsets = OFFSETS.get_or_init(|| {
        std::array::from_fn(|log2_bytes| {
            let bytes_per_pixel = 1 << log2_bytes;
            let (block_width, block_height) = mode27_block_dimensions(bytes_per_pixel)
                .expect("mode 27 table bytes per pixel is supported");
            let elements_per_block = block_width * block_height;
            let mut offsets = vec![0_u16; 4 * elements_per_block];
            for block_y_parity in 0..2 {
                for block_x_parity in 0..2 {
                    let table = (block_y_parity * 2 + block_x_parity) * elements_per_block;
                    for local_y in 0..block_height {
                        for local_x in 0..block_width {
                            let global_x = local_x | (block_x_parity * block_width);
                            let global_y = local_y | (block_y_parity * block_height);
                            offsets[table + local_y * block_width + local_x] =
                                u16::try_from(mode27_within_block_byte_offset_for_format(
                                    global_x,
                                    global_y,
                                    bytes_per_pixel,
                                ))
                                .expect("mode 27 byte offset fits u16");
                        }
                    }
                }
            }
            offsets.into_boxed_slice()
        })
    });
    let log2_bytes =
        usize::try_from(bytes_per_pixel.trailing_zeros()).expect("mode 27 log2 bytes fits usize");
    let elements_per_block = block_width * block_height;
    let table = ((block_y & 1) * 2 + (block_x & 1)) * elements_per_block;
    &offsets[log2_bytes][table..table + elements_per_block]
}

#[cfg(test)]
fn mode27_within_block_byte_offset(x: usize, y: usize) -> usize {
    mode27_within_block_byte_offset_for_format(x, y, 4)
}

fn mode27_block_dimensions(bytes_per_pixel: usize) -> Option<(usize, usize)> {
    if !bytes_per_pixel.is_power_of_two() || !(1..=16).contains(&bytes_per_pixel) {
        return None;
    }
    let element_bits = usize::try_from((65_536 / bytes_per_pixel).trailing_zeros())
        .expect("mode 27 element bit count fits usize");
    let width_bits = element_bits.div_ceil(2);
    Some((1 << width_bits, 1 << (element_bits - width_bits)))
}

fn mode27_within_block_byte_offset_for_format(x: usize, y: usize, bytes_per_pixel: usize) -> usize {
    let bit = |value: usize, index: usize| (value >> index) & 1;
    let log2_bytes =
        usize::try_from(bytes_per_pixel.trailing_zeros()).expect("mode 27 log2 bytes fits usize");
    let address_bits: [(usize, usize); 16] = match log2_bytes {
        0 => [
            (bit(x, 0), 0),
            (bit(x, 1), 0),
            (bit(x, 2), 0),
            (bit(x, 3), 0),
            (bit(y, 0), 0),
            (bit(y, 1), 0),
            (bit(y, 2), 0),
            (bit(y, 3), 0),
            (bit(x, 7) ^ bit(y, 4) ^ bit(y, 7), 0),
            (bit(x, 4) ^ bit(y, 4), 0),
            (bit(x, 6) ^ bit(y, 5), 0),
            (bit(x, 5) ^ bit(y, 6), 0),
            (bit(x, 6), 0),
            (bit(y, 6), 0),
            (bit(x, 7) ^ bit(y, 8), 0),
            (bit(x, 8) ^ bit(y, 7), 0),
        ],
        1 => [
            (0, 0),
            (bit(x, 0), 0),
            (bit(x, 1), 0),
            (bit(x, 2), 0),
            (bit(y, 0), 0),
            (bit(y, 1), 0),
            (bit(y, 2), 0),
            (bit(x, 3), 0),
            (bit(x, 7) ^ bit(y, 4) ^ bit(y, 7), 0),
            (bit(x, 4) ^ bit(y, 4), 0),
            (bit(x, 6) ^ bit(y, 5), 0),
            (bit(x, 5) ^ bit(y, 6), 0),
            (bit(y, 3), 0),
            (bit(x, 6), 0),
            (bit(x, 7) ^ bit(y, 7), 0),
            (bit(x, 8) ^ bit(y, 6), 0),
        ],
        2 => [
            (0, 0),
            (0, 0),
            (bit(x, 0), 0),
            (bit(x, 1), 0),
            (bit(y, 0), 0),
            (bit(y, 1), 0),
            (bit(x, 2), 0),
            (bit(y, 2), 0),
            (bit(x, 7) ^ bit(y, 4) ^ bit(y, 7), 0),
            (bit(x, 4) ^ bit(y, 4), 0),
            (bit(x, 6) ^ bit(y, 5), 0),
            (bit(x, 5) ^ bit(y, 6), 0),
            (bit(x, 3), 0),
            (bit(y, 3), 0),
            (bit(x, 6) ^ bit(y, 7), 0),
            (bit(x, 7) ^ bit(y, 6), 0),
        ],
        3 => [
            (0, 0),
            (0, 0),
            (0, 0),
            (bit(x, 0), 0),
            (bit(y, 0), 0),
            (bit(x, 1), 0),
            (bit(x, 2), 0),
            (bit(y, 1), 0),
            (bit(x, 7) ^ bit(y, 4) ^ bit(y, 7), 0),
            (bit(x, 4) ^ bit(y, 4), 0),
            (bit(x, 6) ^ bit(y, 5), 0),
            (bit(x, 5) ^ bit(y, 6), 0),
            (bit(y, 2), 0),
            (bit(x, 3), 0),
            (bit(x, 7) ^ bit(y, 3), 0),
            (bit(x, 6) ^ bit(y, 6), 0),
        ],
        4 => [
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (bit(x, 0), 0),
            (bit(y, 0), 0),
            (bit(x, 1), 0),
            (bit(y, 1), 0),
            (bit(x, 7) ^ bit(y, 4) ^ bit(y, 7), 0),
            (bit(x, 4) ^ bit(y, 4), 0),
            (bit(x, 6) ^ bit(y, 5), 0),
            (bit(x, 5) ^ bit(y, 6), 0),
            (bit(x, 2), 0),
            (bit(y, 2), 0),
            (bit(x, 6) ^ bit(y, 3), 0),
            (bit(x, 3) ^ bit(y, 6), 0),
        ],
        _ => unreachable!("mode 27 bytes per pixel was validated"),
    };
    address_bits
        .iter()
        .enumerate()
        .fold(0, |offset, (index, &(value, _))| offset | (value << index))
}

fn presenter_window_attributes() -> winit::window::WindowAttributes {
    Window::default_attributes()
        .with_title("REmu")
        .with_inner_size(LogicalSize::new(1280, 720))
}

fn parse_presentation_resolution(value: &str) -> Result<PresentationResolution, String> {
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
fn presentation_drawable_size(
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

fn presentation_viewport(
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
            WindowEvent::CloseRequested => {
                eprintln!("[REMu][METAL] window close requested by AppKit");
                event_loop.exit();
            }
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
                    let scale_factor = self.window.as_ref().map_or(1.0, Window::scale_factor);
                    let (width, height) = presentation_drawable_size(
                        self.presentation_resolution,
                        size.width,
                        size.height,
                        scale_factor,
                    );
                    layer.set_drawable_size(CGSize::new(f64::from(width), f64::from(height)));
                    self.redraw_pending = true;
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let (Some(window), Some(layer)) = (self.window.as_ref(), self.layer.as_ref()) {
                    let size = window.inner_size();
                    let (width, height) = presentation_drawable_size(
                        self.presentation_resolution,
                        size.width,
                        size.height,
                        scale_factor,
                    );
                    layer.set_drawable_size(CGSize::new(f64::from(width), f64::from(height)));
                    self.redraw_pending = true;
                }
            }
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.receive_events() == PresenterFlow::Shutdown {
            event_loop.exit();
            return;
        }
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

fn run_headless_presenter(receiver: Receiver<GpuEvent>) -> Result<(), String> {
    eprintln!("[REMu][METAL] starting headless Metal event loop");
    let mut presenter = MetalPresenter::new(receiver)?;
    eprintln!(
        "[REMu][METAL] headless presenter started on {}",
        presenter
            .device
            .as_ref()
            .ok_or("Metal presenter resources are unavailable")?
            .name()
    );
    loop {
        let (flow, submission) = objc::rc::autoreleasepool(|| {
            let flow = presenter.receive_events();
            let submission = if flow == PresenterFlow::Continue {
                presenter.flush_guest_command_batch(false)
            } else {
                Ok(())
            };
            (flow, submission)
        });
        submission.map_err(|error| format!("guest command batch submission failed: {error}"))?;
        if flow == PresenterFlow::Shutdown {
            eprintln!("[REMu][METAL] headless event loop stopped");
            return Ok(());
        }
        std::thread::sleep(EVENT_POLL_INTERVAL);
    }
}

pub fn run_presenter(receiver: Receiver<GpuEvent>) -> Result<(), String> {
    if headless_metal_requested() {
        return run_headless_presenter(receiver);
    }
    eprintln!("[REMu][METAL] starting AppKit event loop");
    let mut presenter = MetalPresenter::new(receiver)?;
    let mut builder = EventLoop::builder();
    builder
        .with_activation_policy(ActivationPolicy::Regular)
        .with_activate_ignoring_other_apps(true);
    let event_loop = builder
        .build()
        .map_err(|error| format!("event loop failed: {error}"))?;
    event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + EVENT_POLL_INTERVAL));
    // Winit owns NSApplication's delegate and launch state. Calling `finishLaunching` or its
    // delegate callback ourselves marks the delegate as running before `run_app`, causing a
    // re-entrant AppKit loop assertion. Window and Metal-layer creation happen from `resumed`,
    // after Winit has delivered the canonical launch sequence on the main thread.
    event_loop
        .run_app(&mut presenter)
        .map_err(|error| format!("event loop failed: {error}"))?;
    eprintln!("[REMu][METAL] AppKit event loop stopped");
    if let Some(error) = presenter.startup_error {
        Err(error)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use metal::{
        CompileOptions, Device, MTLClearColor, MTLLoadAction, MTLPixelFormat, MTLStorageMode,
        MTLStoreAction, MTLTextureType, MTLTextureUsage, RenderPassDescriptor,
        RenderPipelineDescriptor, TextureDescriptor,
    };
    use remu_gpu::{
        ComputeGrid, ComputeLaunch, GpuColorTarget, GpuEvent, GpuMemoryAcquire, GpuRegister,
        GpuScanoutTarget, IndexedDraw, gpu_event_channel,
    };
    use remu_shader::PixelOutputKind;

    use super::{
        CB_COLOR_REGISTER_STRIDE, CB_COLOR0_ATTRIB, CB_COLOR0_ATTRIB2, CB_COLOR0_ATTRIB3,
        CB_COLOR0_BASE, CB_COLOR0_BASE_EXT, CB_COLOR0_INFO, CB_COLOR0_VIEW, CB_TARGET_MASK,
        GraphicsPipelineKey, GuestTextureDescriptor, OrderedEventPoll, OrderedGpuEventReceiver,
        PendingTextureReadback, PresentationResolution, TextureReadbackLayout,
        apply_image_destination_select, decode_image_descriptor, decode_render_target_format,
        decode_render_targets_from_registers, decode_scanout_target, decode_scissor,
        decode_viewport, effective_linear_pitch, effective_linear_rgba_pitch,
        guest_texture_byte_count, guest_texture_descriptors_overlap, guest_texture_raw_matches,
        headless_metal_value_is_enabled, metal_color_write_mask, metal_nonindexed_vertex_count,
        mode27_block_dimensions, mode27_source_byte_offset, mode27_within_block_byte_offset,
        parse_presentation_resolution, presentation_drawable_size, presentation_viewport,
        read_draw_indices, snapshot_guest_texture_linear, snapshot_guest_texture_raw,
        srgb_unorm8_to_linear_unorm8, validate_compute_launch_for_metal,
        write_guest_texture_linear,
    };

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

    #[allow(clippy::too_many_arguments)]
    fn append_color_target_registers(
        registers: &mut Vec<GpuRegister>,
        slot: u32,
        address: u64,
        width: u32,
        height: u32,
        data_format: u32,
        number_type: u32,
        tile_mode: u32,
    ) {
        let primary = slot * CB_COLOR_REGISTER_STRIDE;
        registers.extend([
            GpuRegister {
                offset: CB_COLOR0_BASE + primary,
                value: u32::try_from((address >> 8) & 0xffff_ffff).expect("base fits u32"),
            },
            GpuRegister {
                offset: CB_COLOR0_VIEW + primary,
                value: 0,
            },
            GpuRegister {
                offset: CB_COLOR0_INFO + primary,
                value: (data_format << 2) | (number_type << 8),
            },
            GpuRegister {
                offset: CB_COLOR0_ATTRIB + primary,
                value: 0,
            },
            GpuRegister {
                offset: CB_COLOR0_BASE_EXT + slot,
                value: u32::try_from((address >> 40) & 0xff).expect("base extension fits u32"),
            },
            GpuRegister {
                offset: CB_COLOR0_ATTRIB2 + slot,
                value: ((width - 1) << 14) | (height - 1),
            },
            GpuRegister {
                offset: CB_COLOR0_ATTRIB3 + slot,
                value: (tile_mode << 14) | (u32::from(height > 1) << 24),
            },
        ]);
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
            scanout_target: None,
            context_registers: Vec::new(),
            shader_registers: Vec::new(),
            user_config_registers: Vec::new(),
        }
    }

    #[test]
    fn ordered_compile_gate_replays_draw_before_barrier_acquire_and_flip() {
        let (sender, receiver) = gpu_event_channel();
        let mut ordered = OrderedGpuEventReceiver::new(receiver);
        let key = GraphicsPipelineKey {
            export_shader_address: 0x1000,
            pixel_shader_address: 0x2000,
            targets: Vec::new(),
        };
        let mut draw = indexed_draw(0, 0, 0);
        draw.sequence = 17;
        ordered.block_draw(key, Box::new(draw));
        sender
            .send(GpuEvent::PipelineBarrier(remu_gpu::GpuPipelineBarrier {
                wait_for: remu_gpu::GpuPipelineStage::PixelShader,
            }))
            .expect("queue later barrier");
        sender
            .send(GpuEvent::MemoryAcquire(GpuMemoryAcquire {
                engine: 0,
                cb_db_control: 0,
                base_address: 0x2000,
                size_bytes: 0x1000,
                poll_interval: 0,
                gcr_control: 0,
            }))
            .expect("queue later acquire");
        sender
            .send(GpuEvent::Flip {
                address: 0x7000_0000,
                buffer_index: 1,
                flip_argument: 9,
                width: 1_280,
                height: 720,
                pitch: 1_280,
                pixel_format: 0x8000_0000_2200_0000,
                tiling_mode: 0,
            })
            .expect("queue later flip");

        assert!(matches!(
            ordered.try_recv(|_| false),
            OrderedEventPoll::Blocked
        ));
        assert!(matches!(
            ordered.try_recv(|_| false),
            OrderedEventPoll::Blocked
        ));
        match ordered.try_recv(|_| true) {
            OrderedEventPoll::Event(GpuEvent::IndexedDraw(draw)) => {
                assert_eq!(draw.sequence, 17);
            }
            other => panic!("expected the blocked draw exactly once, got {other:?}"),
        }
        assert!(matches!(
            ordered.try_recv(|_| true),
            OrderedEventPoll::Event(GpuEvent::PipelineBarrier(remu_gpu::GpuPipelineBarrier {
                wait_for: remu_gpu::GpuPipelineStage::PixelShader,
            }))
        ));
        assert!(matches!(
            ordered.try_recv(|_| true),
            OrderedEventPoll::Event(GpuEvent::MemoryAcquire(GpuMemoryAcquire {
                base_address: 0x2000,
                size_bytes: 0x1000,
                ..
            }))
        ));
        assert!(matches!(
            ordered.try_recv(|_| true),
            OrderedEventPoll::Event(GpuEvent::Flip {
                address: 0x7000_0000,
                ..
            })
        ));
        assert!(matches!(
            ordered.try_recv(|_| true),
            OrderedEventPoll::Empty
        ));
    }

    #[test]
    fn texture_readback_layout_keeps_exact_packed_rows_and_aligned_staging_rows() {
        let layout = TextureReadbackLayout::new(3, 2, 4).expect("valid layout");
        assert_eq!(layout.packed_bytes_per_row, 12);
        assert_eq!(layout.staging_bytes_per_row, 256);
        assert_eq!(layout.staging_length, 512);
        assert_eq!(layout.packed_length(2).expect("packed length"), 24);
    }

    #[test]
    fn capture_linearizes_srgb_storage_bytes_with_the_standard_transfer() {
        assert_eq!(srgb_unorm8_to_linear_unorm8(0), 0);
        assert_eq!(srgb_unorm8_to_linear_unorm8(10), 1);
        assert_eq!(srgb_unorm8_to_linear_unorm8(38), 5);
        assert_eq!(srgb_unorm8_to_linear_unorm8(56), 10);
        assert_eq!(srgb_unorm8_to_linear_unorm8(85), 23);
        assert_eq!(srgb_unorm8_to_linear_unorm8(255), 255);
    }

    #[test]
    fn private_render_target_readback_uses_owned_staging_resources() {
        let device = Device::system_default().expect("Metal device");
        let queue = device.new_command_queue();
        let descriptor = GuestTextureDescriptor {
            address: 0,
            width: 3,
            height: 2,
            pitch: 3,
            tile_mode: 0,
            pixel_format: MTLPixelFormat::BGRA8Unorm,
            bytes_per_pixel: 4,
        };
        let metal_descriptor = TextureDescriptor::new();
        metal_descriptor.set_texture_type(MTLTextureType::D2);
        metal_descriptor.set_pixel_format(descriptor.pixel_format);
        metal_descriptor.set_width(u64::from(descriptor.width));
        metal_descriptor.set_height(u64::from(descriptor.height));
        metal_descriptor.set_storage_mode(MTLStorageMode::Private);
        metal_descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        let texture = device.new_texture(&metal_descriptor);

        let pass = RenderPassDescriptor::new();
        let attachment = pass
            .color_attachments()
            .object_at(0)
            .expect("color attachment zero");
        attachment.set_texture(Some(&texture));
        attachment.set_load_action(MTLLoadAction::Clear);
        attachment.set_store_action(MTLStoreAction::Store);
        attachment.set_clear_color(MTLClearColor::new(1.0, 0.0, 0.0, 1.0));
        let render = queue.new_command_buffer().to_owned();
        let encoder = render.new_render_command_encoder(pass);
        encoder.end_encoding();
        render.commit();

        let pending = PendingTextureReadback::encode(&device, &queue, &texture, descriptor)
            .expect("encode private readback");
        drop(texture);
        let bytes = pending.finish().expect("finish private readback");
        assert_eq!(bytes.len(), 24);
        assert!(bytes.chunks_exact(4).all(|pixel| pixel == [0, 0, 255, 255]));
    }

    #[test]
    fn multisample_resolve_target_is_readable_after_ordered_resolve() {
        let device = Device::system_default().expect("Metal device");
        if !device.supports_texture_sample_count(4) {
            return;
        }
        let queue = device.new_command_queue();
        let descriptor = GuestTextureDescriptor {
            address: 0,
            width: 2,
            height: 2,
            pitch: 2,
            tile_mode: 0,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        let multisample_descriptor = TextureDescriptor::new();
        multisample_descriptor.set_texture_type(MTLTextureType::D2Multisample);
        multisample_descriptor.set_pixel_format(descriptor.pixel_format);
        multisample_descriptor.set_width(u64::from(descriptor.width));
        multisample_descriptor.set_height(u64::from(descriptor.height));
        multisample_descriptor.set_sample_count(4);
        multisample_descriptor.set_storage_mode(MTLStorageMode::Private);
        multisample_descriptor.set_usage(MTLTextureUsage::RenderTarget);
        let multisample = device.new_texture(&multisample_descriptor);
        let unresolved_error =
            PendingTextureReadback::encode(&device, &queue, &multisample, descriptor)
                .err()
                .expect("unresolved multisample readback must fail");
        assert!(unresolved_error.contains("single-sample resolve target"));

        let resolve_descriptor = TextureDescriptor::new();
        resolve_descriptor.set_texture_type(MTLTextureType::D2);
        resolve_descriptor.set_pixel_format(descriptor.pixel_format);
        resolve_descriptor.set_width(u64::from(descriptor.width));
        resolve_descriptor.set_height(u64::from(descriptor.height));
        resolve_descriptor.set_storage_mode(MTLStorageMode::Private);
        resolve_descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        let resolve = device.new_texture(&resolve_descriptor);

        let pass = RenderPassDescriptor::new();
        let attachment = pass
            .color_attachments()
            .object_at(0)
            .expect("color attachment zero");
        attachment.set_texture(Some(&multisample));
        attachment.set_resolve_texture(Some(&resolve));
        attachment.set_load_action(MTLLoadAction::Clear);
        attachment.set_store_action(MTLStoreAction::MultisampleResolve);
        attachment.set_clear_color(MTLClearColor::new(0.0, 1.0, 0.0, 1.0));
        let render = queue.new_command_buffer().to_owned();
        let encoder = render.new_render_command_encoder(pass);
        encoder.end_encoding();
        render.commit();

        let bytes = PendingTextureReadback::encode(&device, &queue, &resolve, descriptor)
            .expect("encode resolve readback")
            .finish()
            .expect("finish resolve readback");
        assert!(bytes.chunks_exact(4).all(|pixel| pixel == [0, 255, 0, 255]));
    }

    #[test]
    fn private_depth32_target_readback_preserves_float_samples() {
        let device = Device::system_default().expect("Metal device");
        let queue = device.new_command_queue();
        let descriptor = GuestTextureDescriptor {
            address: 0,
            width: 2,
            height: 2,
            pitch: 2,
            tile_mode: 0,
            pixel_format: MTLPixelFormat::Depth32Float,
            bytes_per_pixel: 4,
        };
        let metal_descriptor = TextureDescriptor::new();
        metal_descriptor.set_texture_type(MTLTextureType::D2);
        metal_descriptor.set_pixel_format(descriptor.pixel_format);
        metal_descriptor.set_width(u64::from(descriptor.width));
        metal_descriptor.set_height(u64::from(descriptor.height));
        metal_descriptor.set_storage_mode(MTLStorageMode::Private);
        metal_descriptor.set_usage(MTLTextureUsage::RenderTarget);
        let texture = device.new_texture(&metal_descriptor);

        let pass = RenderPassDescriptor::new();
        let attachment = pass.depth_attachment().expect("depth attachment");
        attachment.set_texture(Some(&texture));
        attachment.set_load_action(MTLLoadAction::Clear);
        attachment.set_store_action(MTLStoreAction::Store);
        attachment.set_clear_depth(0.25);
        let render = queue.new_command_buffer().to_owned();
        let encoder = render.new_render_command_encoder(pass);
        encoder.end_encoding();
        render.commit();

        let bytes = PendingTextureReadback::encode(&device, &queue, &texture, descriptor)
            .expect("encode depth readback")
            .finish()
            .expect("finish depth readback");
        assert!(
            bytes
                .chunks_exact(4)
                .map(|sample| f32::from_ne_bytes(sample.try_into().expect("four-byte depth")))
                .all(|sample| (sample - 0.25).abs() <= f32::EPSILON)
        );
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
        assert_eq!(effective_linear_pitch(250, 1), Some(256));
        assert_eq!(effective_linear_pitch(250, 2), Some(256));
        assert_eq!(effective_linear_pitch(250, 8), Some(256));
        assert_eq!(effective_linear_pitch(250, 16), Some(256));
    }

    #[test]
    fn decodes_all_gfx10_color_slots_and_dense_metal_output_layout() {
        let address0 = 0x0070_1234_5600_u64;
        let address3 = 0x0071_89ab_cd00_u64;
        let mut registers = vec![GpuRegister {
            offset: CB_TARGET_MASK,
            value: 0x0000_f00b,
        }];
        append_color_target_registers(&mut registers, 0, address0, 1_280, 720, 10, 0, 27);
        append_color_target_registers(&mut registers, 3, address3, 1_280, 720, 12, 4, 27);
        registers.sort_by_key(|register| register.offset);

        let targets = decode_render_targets_from_registers(&registers, 0x0000_5007)
            .expect("two valid MRT targets");
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0].guest_slot, 0);
        assert_eq!(targets[0].texture.address, address0);
        assert_eq!(targets[0].texture.pixel_format, MTLPixelFormat::RGBA8Unorm);
        assert_eq!(targets[0].texture.bytes_per_pixel, 4);
        assert_eq!(targets[0].output_kind, PixelOutputKind::Float);
        assert_eq!(targets[0].write_mask, 0x3);
        assert_eq!(targets[1].guest_slot, 3);
        assert_eq!(targets[1].texture.address, address3);
        assert_eq!(targets[1].texture.pixel_format, MTLPixelFormat::RGBA16Uint);
        assert_eq!(targets[1].texture.bytes_per_pixel, 8);
        assert_eq!(targets[1].output_kind, PixelOutputKind::Uint);
        assert_eq!(targets[1].write_mask, 0x5);
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
            pixel_format: MTLPixelFormat::RGBA8Unorm_sRGB,
            bytes_per_pixel: 4,
        };

        let texture = presenter.ensure_direct_scanout_texture(&device, descriptor);
        presenter.next_flip_sequence = 1;

        assert!(
            presenter
                .activate_direct_scanout(descriptor.address, descriptor.width, descriptor.height)
                .expect("activate direct scanout")
        );
        assert_eq!(presenter.frame_texture_size, (640, 360));
        let output_texture = presenter.frame_texture.as_ref().expect("presented texture");
        assert_eq!(output_texture.width(), texture.width());
        assert_eq!(output_texture.height(), texture.height());
    }

    #[test]
    fn render_target_decode_rejects_incomplete_or_lossy_state() {
        let address = 0x0070_1234_5600_u64;
        let mut registers = vec![GpuRegister {
            offset: CB_TARGET_MASK,
            value: 0xf,
        }];
        append_color_target_registers(&mut registers, 0, address, 640, 360, 10, 0, 27);
        registers.sort_by_key(|register| register.offset);
        registers
            .iter_mut()
            .find(|register| register.offset == CB_COLOR0_INFO)
            .expect("color info exists")
            .value |= 1 << 28;
        assert!(
            decode_render_targets_from_registers(&registers, 0xf)
                .expect_err("DCC cannot be silently treated as raw color")
                .contains("dcc=1")
        );

        registers
            .iter_mut()
            .find(|register| register.offset == CB_COLOR0_INFO)
            .expect("color info exists")
            .value &= !(1 << 28);
        registers
            .iter_mut()
            .find(|register| register.offset == CB_COLOR0_VIEW)
            .expect("color view exists")
            .value = 1 << 26;
        assert!(
            decode_render_targets_from_registers(&registers, 0xf)
                .expect_err("nonzero mip view cannot be silently flattened")
                .contains("mip=1/0")
        );

        registers
            .iter_mut()
            .find(|register| register.offset == CB_COLOR0_VIEW)
            .expect("color view exists")
            .value = 0;
        registers
            .iter_mut()
            .find(|register| register.offset == CB_COLOR0_ATTRIB3)
            .expect("color attrib3 exists")
            .value = (27 << 14) | (3 << 24);
        assert!(
            decode_render_targets_from_registers(&registers, 0xf)
                .expect_err("reserved resource type cannot be treated as a 2D texture")
                .contains("type=3")
        );

        registers
            .iter_mut()
            .find(|register| register.offset == CB_COLOR0_ATTRIB3)
            .expect("color attrib3 exists")
            .value = (27 << 14) | (1 << 24);

        registers.retain(|register| register.offset != CB_COLOR0_ATTRIB2);
        assert!(
            decode_render_targets_from_registers(&registers, 0xf)
                .expect_err("missing extent register must fail")
                .contains("CB_COLOR_ATTRIB2")
        );
    }

    #[test]
    fn maps_guest_number_types_to_matching_metal_export_types() {
        let float = decode_render_target_format(10, 0, 0).expect("RGBA8 unorm");
        let uint = decode_render_target_format(10, 4, 0).expect("RGBA8 uint");
        let sint = decode_render_target_format(12, 5, 0).expect("RGBA16 sint");
        let bgra = decode_render_target_format(10, 0, 1).expect("BGRA component swap");
        assert_eq!(float.output_kind, PixelOutputKind::Float);
        assert_eq!(uint.output_kind, PixelOutputKind::Uint);
        assert_eq!(sint.output_kind, PixelOutputKind::Sint);
        assert_eq!(bgra.pixel_format, MTLPixelFormat::BGRA8Unorm);
        assert!(decode_render_target_format(10, 2, 0).is_err());
        assert!(decode_render_target_format(10, 0, 2).is_err());
    }

    #[test]
    fn translates_gfx10_write_mask_bit_order_to_metal() {
        assert_eq!(metal_color_write_mask(0x1).bits(), 0x8);
        assert_eq!(metal_color_write_mask(0x2).bits(), 0x4);
        assert_eq!(metal_color_write_mask(0x4).bits(), 0x2);
        assert_eq!(metal_color_write_mask(0x8).bits(), 0x1);
        assert_eq!(metal_color_write_mask(0xf).bits(), 0xf);
    }

    #[test]
    fn apple_metal_compiles_mixed_float_uint_sint_mrt_pipeline() {
        let device = Device::system_default().expect("Apple test host has a Metal device");
        let source = r"
#include <metal_stdlib>
using namespace metal;
struct MixedMrtOutput {
    float4 floating [[color(0)]];
    uint4 unsigned_integer [[color(1)]];
    int4 signed_integer [[color(2)]];
};
vertex float4 remu_mrt_vertex(uint vertex_id [[vertex_id]]) {
    return float4(float(vertex_id & 1u), float((vertex_id >> 1u) & 1u), 0.0, 1.0);
}
fragment MixedMrtOutput remu_mrt_fragment() {
    return { float4(0.25), uint4(7u), int4(-3) };
}
";
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(source, &options)
            .expect("mixed MRT MSL compiles");
        let vertex = library
            .get_function("remu_mrt_vertex", None)
            .expect("MRT vertex function exists");
        let fragment = library
            .get_function("remu_mrt_fragment", None)
            .expect("MRT fragment function exists");
        let descriptor = RenderPipelineDescriptor::new();
        descriptor.set_vertex_function(Some(&vertex));
        descriptor.set_fragment_function(Some(&fragment));
        for (location, format, mask) in [
            (0, MTLPixelFormat::RGBA8Unorm, 0xf),
            (1, MTLPixelFormat::RGBA8Uint, 0x5),
            (2, MTLPixelFormat::RGBA16Sint, 0xa),
        ] {
            let attachment = descriptor
                .color_attachments()
                .object_at(location)
                .expect("Metal exposes eight color attachments");
            attachment.set_pixel_format(format);
            attachment.set_write_mask(metal_color_write_mask(mask));
        }
        device
            .new_render_pipeline_state(&descriptor)
            .expect("mixed typed MRT render pipeline compiles");
    }

    #[test]
    fn every_decoded_render_target_format_is_apple_metal_renderable() {
        let device = Device::system_default().expect("Apple test host has a Metal device");
        let source = r"
#include <metal_stdlib>
using namespace metal;
vertex float4 remu_format_vertex(uint vertex_id [[vertex_id]]) {
    return float4(float(vertex_id & 1u), float((vertex_id >> 1u) & 1u), 0.0, 1.0);
}
fragment float4 remu_format_float() { return float4(0.25); }
fragment uint4 remu_format_uint() { return uint4(7u); }
fragment int4 remu_format_sint() { return int4(-3); }
";
        let library = device
            .new_library_with_source(source, &CompileOptions::new())
            .expect("format-cohort MSL compiles");
        let vertex = library
            .get_function("remu_format_vertex", None)
            .expect("format vertex exists");
        let cases = [
            (1, 0, 0),
            (1, 1, 0),
            (1, 4, 0),
            (1, 5, 0),
            (1, 6, 0),
            (2, 0, 0),
            (2, 1, 0),
            (2, 4, 0),
            (2, 5, 0),
            (2, 7, 0),
            (3, 0, 0),
            (3, 1, 0),
            (3, 4, 0),
            (3, 5, 0),
            (3, 6, 0),
            (4, 4, 0),
            (4, 5, 0),
            (4, 7, 0),
            (5, 0, 0),
            (5, 1, 0),
            (5, 4, 0),
            (5, 5, 0),
            (5, 7, 0),
            (6, 7, 0),
            (7, 7, 0),
            (8, 0, 0),
            (9, 0, 0),
            (10, 0, 0),
            (10, 0, 1),
            (10, 1, 0),
            (10, 4, 0),
            (10, 5, 0),
            (10, 6, 0),
            (10, 6, 1),
            (11, 4, 0),
            (11, 5, 0),
            (11, 7, 0),
            (12, 0, 0),
            (12, 1, 0),
            (12, 4, 0),
            (12, 5, 0),
            (12, 7, 0),
            (14, 4, 0),
            (14, 5, 0),
            (14, 7, 0),
            (24, 7, 0),
        ];
        for (data_format, number_type, component_swap) in cases {
            let decoded = decode_render_target_format(data_format, number_type, component_swap)
                .expect("listed format decodes");
            let fragment_name = match decoded.output_kind {
                PixelOutputKind::Float => "remu_format_float",
                PixelOutputKind::Uint => "remu_format_uint",
                PixelOutputKind::Sint => "remu_format_sint",
            };
            let fragment = library
                .get_function(fragment_name, None)
                .expect("typed fragment exists");
            let descriptor = RenderPipelineDescriptor::new();
            descriptor.set_vertex_function(Some(&vertex));
            descriptor.set_fragment_function(Some(&fragment));
            descriptor
                .color_attachments()
                .object_at(0)
                .expect("color attachment zero exists")
                .set_pixel_format(decoded.pixel_format);
            device.new_render_pipeline_state(&descriptor).unwrap_or_else(|error| {
                panic!(
                    "Metal rejected GFX10 format {data_format}/{number_type}/swap{component_swap} as {:?}: {error}",
                    decoded.pixel_format
                )
            });
        }
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
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
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

        write_guest_texture_linear(descriptor, &linear).expect("mode 27 write succeeds");
        assert_eq!(
            snapshot_guest_texture_linear(descriptor).expect("mode 27 snapshot succeeds"),
            linear
        );
    }

    #[test]
    fn mode27_round_trips_every_supported_render_target_element_size() {
        for (bytes_per_pixel, pixel_format) in [
            (1, MTLPixelFormat::R8Unorm),
            (2, MTLPixelFormat::RG8Unorm),
            (4, MTLPixelFormat::RGBA8Unorm),
            (8, MTLPixelFormat::RGBA16Float),
            (16, MTLPixelFormat::RGBA32Float),
        ] {
            let (block_width, block_height) = mode27_block_dimensions(bytes_per_pixel)
                .expect("render-target element size has a mode 27 equation");
            let width = block_width + 1;
            let height = block_height + 1;
            let mut descriptor = GuestTextureDescriptor {
                address: 0x1_0000,
                width: u32::try_from(width).expect("test width fits u32"),
                height: u32::try_from(height).expect("test height fits u32"),
                pitch: u32::try_from(width).expect("test pitch fits u32"),
                tile_mode: 27,
                pixel_format,
                bytes_per_pixel: u32::try_from(bytes_per_pixel)
                    .expect("test bytes per pixel fits u32"),
            };
            let tiled_length = usize::try_from(
                guest_texture_byte_count(descriptor).expect("tiled byte count fits u64"),
            )
            .expect("tiled byte count fits usize");
            let mut tiled = vec![0_u8; tiled_length];
            descriptor.address = u64::try_from(tiled.as_mut_ptr() as usize)
                .expect("test allocation address fits u64");
            let linear_length = width * height * bytes_per_pixel;
            let linear = (0..linear_length)
                .map(|index| u8::try_from(index & 0xff).expect("masked byte fits u8"))
                .collect::<Vec<_>>();
            write_guest_texture_linear(descriptor, &linear).expect("mode 27 write succeeds");
            assert_eq!(
                snapshot_guest_texture_linear(descriptor).expect("mode 27 snapshot succeeds"),
                linear,
                "bytes per pixel {bytes_per_pixel}",
            );
        }
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

    fn compute_launch() -> ComputeLaunch {
        ComputeLaunch {
            grid: ComputeGrid::Threadgroups([8, 4, 1]),
            group_count: [8, 4, 1],
            base_group: [0; 3],
            threads_per_group: [32, 2, 1],
            wave_lane_count: 32,
            lds_size_bytes: 16 * 1024,
            initiator: 0x8045,
        }
    }

    #[test]
    fn validates_exact_metal_compute_execution_geometry() {
        validate_compute_launch_for_metal(compute_launch(), 32, 1_024, 32 * 1024, 0, 16 * 1024)
            .expect("representable compute launch");
    }

    #[test]
    fn rejects_compute_geometry_metal_cannot_represent_exactly() {
        let mut based = compute_launch();
        based.base_group = [4, 0, 0];
        assert!(
            validate_compute_launch_for_metal(based, 32, 1_024, 32 * 1024, 0, 16 * 1024)
                .expect_err("dispatch base is not silently dropped")
                .contains("nonzero base")
        );

        assert!(
            validate_compute_launch_for_metal(
                compute_launch(),
                16,
                1_024,
                32 * 1024,
                0,
                16 * 1024,
            )
                .expect_err("wave width must match")
                .contains("wave32")
        );

        let mut oversized = compute_launch();
        oversized.threads_per_group = [1_025, 1, 1];
        assert!(
            validate_compute_launch_for_metal(oversized, 32, 1_024, 32 * 1024, 0, 16 * 1024)
                .expect_err("oversized workgroup must fail")
                .contains("1025 threads")
        );

        assert!(
            validate_compute_launch_for_metal(compute_launch(), 32, 1_024, 32 * 1024, 0, 8 * 1024)
                .expect_err("LDS binding must equal lowering layout")
                .contains("8192 bytes of LDS")
        );
        assert!(
            validate_compute_launch_for_metal(
                compute_launch(),
                32,
                1_024,
                32 * 1024,
                20 * 1024,
                16 * 1024,
            )
            .expect_err("Metal LDS limit must be enforced")
            .contains("36864 bytes")
        );
    }
}
