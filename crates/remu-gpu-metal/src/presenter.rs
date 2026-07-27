// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::{OsStr, c_void};
use std::hash::{Hash, Hasher};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use core_graphics_types::geometry::CGSize;
use metal::foreign_types::ForeignType;
use metal::{
    BinaryArchive, BinaryArchiveDescriptor, Buffer, CommandBuffer, CompileOptions,
    ComputePipelineState, DepthStencilDescriptor, DepthStencilState, Device, Function, Library,
    MTLBlendFactor, MTLBlendOperation, MTLBlitOption, MTLClearColor, MTLColorWriteMask,
    MTLCommandBufferStatus, MTLCompareFunction, MTLCullMode, MTLIndexType, MTLLoadAction,
    MTLOrigin, MTLPixelFormat, MTLPrimitiveTopologyClass, MTLPrimitiveType, MTLRegion,
    MTLRenderStages, MTLResourceOptions, MTLSamplerAddressMode, MTLSamplerMinMagFilter,
    MTLSamplerMipFilter, MTLScissorRect, MTLSize, MTLStorageMode, MTLStoreAction, MTLTextureType,
    MTLTextureUsage, MTLTriangleFillMode, MTLViewport, MTLWinding, MetalLayer, NSRange, NSUInteger,
    RenderCommandEncoder, RenderPassDescriptor, RenderPipelineDescriptor, RenderPipelineState,
    SamplerDescriptor, SamplerState, Texture, TextureDescriptor, URL,
};
use objc::runtime::{Object, YES};
use objc::{msg_send, sel, sel_impl};
use remu_gpu::{
    ComputeDispatch, ComputeGrid, GpuEvent, GpuMemoryAcquire, GpuMemoryCopy, GpuMemoryRelease,
    GpuMemoryReleaseValue, GpuMemoryWait, GpuMemoryWrite, GpuRegister, GpuScanoutTarget, GpuWaitId,
    HostPadState, IndexedDraw, PAD_BUTTON_CIRCLE, PAD_BUTTON_CROSS, PAD_BUTTON_DOWN, PAD_BUTTON_L1,
    PAD_BUTTON_L2, PAD_BUTTON_LEFT, PAD_BUTTON_OPTIONS, PAD_BUTTON_R1, PAD_BUTTON_R2,
    PAD_BUTTON_RIGHT, PAD_BUTTON_SQUARE, PAD_BUTTON_TRIANGLE, PAD_BUTTON_UP, set_host_pad_state,
};
use remu_memory::{
    CpuWriteWatch, CpuWriteWatchRetirement, guest_shared_backing_alias_segments,
    guest_shared_mapping_identity, read_guest_shared_backing, write_guest_shared_backing,
};
use remu_shader::{
    GlobalMemoryBinding, ImageBinding, InstructionControl, PixelOutputKind, Program,
    ShaderEvaluation, image_is_comparison,
};
use sha1::{Digest, Sha1};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::{Window, WindowId};

const EVENT_POLL_INTERVAL: Duration = Duration::from_millis(1);
/// A late Superliminal frame contains roughly 700 draws. Draining only 16 events per AppKit
/// callback imposed an artificial ~2k draw/s ceiling even when both the guest producer and Metal
/// encoder had capacity. This bound covers most of a frame while still returning to AppKit often
/// enough to service input and window events.
const GPU_EVENTS_PER_TICK: usize = 512;
const SHADER_COMPILE_QUEUE_CAPACITY: usize = 64;
// Unity can introduce dozens of distinct 200-400 KiB pixel programs at a scene boundary. Four
// workers let the ordered GPU queue grow by several thousand events while Apple's compiler is
// busy, which in turn withholds real guest flips and starves the guest audio producers. Current
// Apple silicon has enough performance cores for eight independent offline compiler processes
// without making pipeline construction on the presenter thread synchronous.
const SHADER_COMPILER_WORKER_COUNT: usize = 8;
const GUEST_BATCH_WORK_CAPACITY: usize = 256;
const SAMPLER_ARGUMENT_CACHE_CAPACITY: usize = 1_024;
const GLOBAL_BUFFER_CACHE_CAPACITY: usize = 4_096;
const GLOBAL_BUFFER_CACHE_BYTE_CAPACITY: usize = 64 * 1024 * 1024;
const GUEST_GLOBAL_BUFFER_CACHE_CAPACITY: usize = 4_096;
const INDEX_BUFFER_CACHE_CAPACITY: usize = 256;
const SAMPLED_TEXTURE_CACHE_CAPACITY: usize = 2_048;
const METAL_TEXTURE_BUFFER_ROW_ALIGNMENT: u64 = 256;
const METAL_LIBRARY_CACHE_FORMAT: &[u8] = b"remu-metal-library-v1\0";
const METAL_RENDER_PIPELINE_ARCHIVE_FORMAT: &[u8] = b"remu-metal-render-pipeline-v1\0";
const DIAGNOSTIC_VERTEX_POSITION_VERTEX_CAPACITY: usize = 262_144;
const DIAGNOSTIC_VERTEX_POSITION_BUFFER_INDEX: u64 = 30;
const DIAGNOSTIC_PIXEL_OUTPUT_WIDTH: usize = 1_920;
const DIAGNOSTIC_PIXEL_OUTPUT_HEIGHT: usize = 1_080;
const DIAGNOSTIC_PIXEL_OUTPUT_BUFFER_INDEX: u64 = 30;

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

fn capture_interval() -> Duration {
    static INTERVAL: OnceLock<Duration> = OnceLock::new();
    *INTERVAL.get_or_init(|| {
        std::env::var("REMU_CAPTURE_INTERVAL_SECONDS")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|seconds| seconds.is_finite() && *seconds >= 0.1)
            .map_or(Duration::from_secs(1), Duration::from_secs_f64)
    })
}

fn process_resident_bytes() -> Option<u64> {
    let mut task = std::mem::MaybeUninit::<libc::proc_taskinfo>::zeroed();
    let byte_count = std::mem::size_of::<libc::proc_taskinfo>();
    let byte_count = i32::try_from(byte_count).ok()?;
    // SAFETY: `task` names writable storage of exactly the size passed to `proc_pidinfo` for the
    // current process and is read only after libproc reports a complete `PROC_PIDTASKINFO` result.
    let written = unsafe {
        libc::proc_pidinfo(
            i32::try_from(std::process::id()).ok()?,
            libc::PROC_PIDTASKINFO,
            0,
            task.as_mut_ptr().cast::<c_void>(),
            byte_count,
        )
    };
    if written != byte_count {
        return None;
    }
    // SAFETY: A complete result initialized every field of `proc_taskinfo` above.
    Some(unsafe { task.assume_init() }.pti_resident_size)
}

// GFX10 context-register offsets are the MMIO dword address minus 0xA000. Color registers
// 0..7 use a 15-dword primary stride; the high address and GFX10 extent/swizzle registers are
// stored in separate dense eight-register arrays. These values match Mesa's generated gfx10
// register database and SharpEmu's AGC command-stream decoder.
const CB_TARGET_MASK: u32 = 0x08e;
const CB_BLEND_RED: u32 = 0x105;
const CB_BLEND_GREEN: u32 = 0x106;
const CB_BLEND_BLUE: u32 = 0x107;
const CB_BLEND_ALPHA: u32 = 0x108;
const SPI_PS_INPUT_CNTL_0: u32 = 0x191;
const CB_BLEND0_CONTROL: u32 = 0x1e0;
const PA_SU_SC_MODE_CNTL: u32 = 0x205;
const CB_COLOR0_BASE: u32 = 0x318;
const CB_COLOR_REGISTER_STRIDE: u32 = 15;
const CB_COLOR0_VIEW: u32 = 0x31b;
const CB_COLOR0_INFO: u32 = 0x31c;
const CB_COLOR0_ATTRIB: u32 = 0x31d;
const CB_COLOR0_FMASK: u32 = 0x321;
const CB_COLOR0_BASE_EXT: u32 = 0x390;
const CB_COLOR0_FMASK_BASE_EXT: u32 = 0x3a0;
const CB_COLOR0_ATTRIB2: u32 = 0x3b0;
const CB_COLOR0_ATTRIB3: u32 = 0x3b8;
const PA_SC_AA_MASK_X0Y0_X1Y0: u32 = 0x30e;
const PA_SC_AA_MASK_X0Y1_X1Y1: u32 = 0x30f;
const COLOR_TARGET_COUNT: u32 = 8;
const DB_DEPTH_SIZE_XY: u32 = 0x007;
const DB_Z_INFO: u32 = 0x010;
const DB_Z_READ_BASE: u32 = 0x012;
const DB_Z_WRITE_BASE: u32 = 0x014;
const DB_Z_READ_BASE_HI: u32 = 0x01a;
const DB_Z_WRITE_BASE_HI: u32 = 0x01c;
const DB_DEPTH_CONTROL: u32 = 0x200;

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
    // VideoOut treats scanout alpha as padding; games are free to leave it at zero. Passing that
    // value into CAMetalLayer makes AppKit composite an otherwise valid frame as transparent
    // black, so host presentation must always be opaque.
    return float4(source.sample(source_sampler, input.texture_coordinate).rgb, 1.0);
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

const VIDEO_OUT_TILING_MODE_LINEAR: u32 = 1;

const fn raw_video_out_layout_is_linear(tiling_mode: u32, pitch: u32, width: u32) -> bool {
    tiling_mode == VIDEO_OUT_TILING_MODE_LINEAR && pitch >= width
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
        Self::encode_mip(device, queue, texture, descriptor, 0)
    }

    fn encode_mip(
        device: &Device,
        queue: &metal::CommandQueue,
        texture: &Texture,
        descriptor: GuestTextureDescriptor,
        mip_level: u32,
    ) -> Result<Self, String> {
        validate_texture_readback_source(texture, descriptor)?;
        if u64::from(mip_level) >= texture.mipmap_level_count() {
            return Err(format!(
                "Metal texture has {} mip levels, cannot read level {mip_level}",
                texture.mipmap_level_count()
            ));
        }
        let width = (descriptor.width >> mip_level).max(1);
        let height = (descriptor.height >> mip_level).max(1);
        let layout = TextureReadbackLayout::new(width, height, descriptor.bytes_per_pixel)?;
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
            u64::from(mip_level),
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize::new(u64::from(width), u64::from(height), 1),
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
            height,
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
    blend: MetalBlendAttachmentState,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct DepthTargetPipelineState {
    pixel_format: MTLPixelFormat,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MetalBlendAttachmentState {
    enabled: bool,
    source_rgb: MTLBlendFactor,
    destination_rgb: MTLBlendFactor,
    rgb_operation: MTLBlendOperation,
    source_alpha: MTLBlendFactor,
    destination_alpha: MTLBlendFactor,
    alpha_operation: MTLBlendOperation,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct GraphicsPipelineKey {
    export_shader_address: u64,
    pixel_shader_address: u64,
    export_resources: GraphicsShaderResourceKey,
    pixel_resources: GraphicsShaderResourceKey,
    pixel_input_controls: [u32; 32],
    targets: Vec<RenderTargetPipelineState>,
    depth: Option<DepthTargetPipelineState>,
    sample_count: u32,
    render_target_array_length: u32,
    primitive_topology_class: MTLPrimitiveTopologyClass,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct GraphicsShaderResourceKey {
    initial_scalar_count: usize,
    global_bindings: Vec<(Vec<u32>, u64, usize, bool)>,
    image_bindings: Vec<(u32, u32, bool, bool, u32, u32, u32, Option<u32>)>,
    vertex_inputs: Vec<(u32, u32, u32, u32, u32, u32, u32)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DecodedRenderTarget {
    guest_slot: u32,
    texture: GuestTextureDescriptor,
    output_kind: PixelOutputKind,
    write_mask: u32,
    direct_scanout: bool,
    sample_count: u32,
    texture_type: u32,
    layer_count: u32,
    base_layer: u32,
    mip_level: u32,
    mip_level_count: u32,
    dcc_enabled: bool,
    fast_clear_enabled: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DiagnosticScanoutWriter {
    draw: u64,
    export_shader: u64,
    pixel_shader: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DecodedDepthTarget {
    texture: GuestTextureDescriptor,
    compare: MTLCompareFunction,
    write_enabled: bool,
    sample_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DecodedRasterState {
    cull_mode: MTLCullMode,
    front_winding: MTLWinding,
    fill_mode: MTLTriangleFillMode,
}

struct DirectScanoutTexture {
    descriptor: GuestTextureDescriptor,
    texture: Texture,
    initialized: bool,
}

struct ScanoutSnapshotTexture {
    descriptor: GuestTextureDescriptor,
    texture: Texture,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MultisampleTextureKey {
    descriptor: GuestTextureDescriptor,
    sample_count: u32,
}

struct MultisampleResidentTexture {
    texture: Texture,
    initialized: bool,
}

struct CompiledShaderPair {
    pipeline: RenderPipelineState,
    vertex_function: Function,
    pixel_function: Function,
    vertex: remu_shader_metal::MetalShader,
    pixel: remu_shader_metal::MetalShader,
    reused_vertex_function: bool,
    reused_pixel_function: bool,
}

struct CompiledComputePipeline {
    pipeline: ComputePipelineState,
    function: Function,
    shader: remu_shader_metal::MetalShader,
}

#[derive(Debug, Eq, Hash, PartialEq)]
struct SamplerArgumentCacheKey {
    function: usize,
    buffer_index: u32,
    descriptors: Vec<Vec<u32>>,
}

struct CachedSamplerArguments {
    buffer: Buffer,
    samplers: Vec<SamplerState>,
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
    initial_scalar_count: usize,
    global_bindings: Vec<(Vec<u32>, u64, usize, bool)>,
    image_bindings: Vec<(u32, bool, bool, u32, u32, u32, Option<u32>)>,
}

struct ShaderCompileResult {
    key: GraphicsPipelineKey,
    compiled: Result<CompiledShaderPair, String>,
}

struct ShaderCompileRequest {
    key: GraphicsPipelineKey,
    draw: IndexedDraw,
}

#[derive(Debug, Default)]
struct RenderPipelineArchiveState {
    additions: usize,
}

#[derive(Clone)]
struct PersistentRenderPipelineArchive {
    archive: BinaryArchive,
    path: Arc<PathBuf>,
    state: Arc<Mutex<RenderPipelineArchiveState>>,
    driver_lock: Arc<Mutex<()>>,
}

impl PersistentRenderPipelineArchive {
    fn open(device: &Device) -> Result<Self, String> {
        let path = metal_render_pipeline_archive_path(device);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| {
                format!(
                    "could not create Metal pipeline cache {}: {error}",
                    parent.display()
                )
            })?;
        }
        // metal-rs 0.33 wraps +[NSURL URLWithString:] (an autoreleased object) as an owned URL.
        // Suppress the wrapper's release and let the active Objective-C pool perform the single
        // balanced release; otherwise archive serialization crashes at autoreleasePoolPop.
        let url = std::mem::ManuallyDrop::new(URL::new_with_string(&format!(
            "file://{}",
            path.display()
        )));
        let descriptor = BinaryArchiveDescriptor::new();
        if path.is_file() {
            descriptor.set_url(&url);
        }
        let archive = match device.new_binary_archive_with_descriptor(&descriptor) {
            Ok(archive) => archive,
            Err(error) if path.is_file() => {
                eprintln!(
                    "[REMu][METAL] discarding unreadable pipeline archive {}: {error}",
                    path.display()
                );
                std::fs::remove_file(&path).map_err(|remove_error| {
                    format!(
                        "could not remove invalid pipeline archive {}: {remove_error}",
                        path.display()
                    )
                })?;
                device
                    .new_binary_archive_with_descriptor(&BinaryArchiveDescriptor::new())
                    .map_err(|error| format!("could not create Metal pipeline archive: {error}"))?
            }
            Err(error) => return Err(format!("could not create Metal pipeline archive: {error}")),
        };
        archive.set_label("REMu persistent render pipelines");
        eprintln!(
            "[REMu][METAL] {} render pipeline archive {}",
            if path.is_file() { "loaded" } else { "created" },
            path.display()
        );
        Ok(Self {
            archive,
            path: Arc::new(path),
            state: Arc::new(Mutex::new(RenderPipelineArchiveState::default())),
            driver_lock: Arc::new(Mutex::new(())),
        })
    }

    fn new_render_pipeline_state(
        &self,
        device: &Device,
        make_descriptor: impl FnOnce() -> Result<RenderPipelineDescriptor, String>,
    ) -> Result<RenderPipelineState, String> {
        // MTLBinaryArchive is documented as a resource shared by pipeline descriptors, but its
        // Rosetta driver path crashes while autoreleased descriptor internals from one thread are
        // drained concurrently with another archive mutation. Keep just descriptor/archive/
        // pipeline creation serialized and drain its local pool before admitting the next worker;
        // MSL lowering and metallib loading remain parallel outside this critical section.
        let _driver = self
            .driver_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        objc::rc::autoreleasepool(|| {
            let descriptor = make_descriptor()?;
            descriptor.set_binary_archives(&[&self.archive]);
            self.archive
                .add_render_pipeline_functions_with_descriptor(&descriptor)
                .map_err(|error| {
                    format!("could not add Metal render pipeline to archive: {error}")
                })?;
            let should_serialize = {
                let mut state = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                state.additions = state.additions.saturating_add(1);
                state.additions.is_power_of_two() && state.additions >= 8
            };
            let pipeline = device
                .new_render_pipeline_state(&descriptor)
                .map_err(|error| format!("Metal render pipeline: {error}"))?;
            // Persist progressively because diagnostic and interactive game runs are often
            // stopped by SIGALRM or from the window close button rather than a clean guest
            // shutdown. Powers of two bound serialization overhead while retaining nearly all
            // newly discovered pipelines.
            if should_serialize {
                self.serialize_locked()?;
            }
            Ok(pipeline)
        })
    }

    fn serialize_locked(&self) -> Result<(), String> {
        let url = std::mem::ManuallyDrop::new(URL::new_with_string(&format!(
            "file://{}",
            self.path.display()
        )));
        self.archive
            .serialize_to_url(&url)
            .map(|_| ())
            .map_err(|error| {
                format!(
                    "could not persist Metal render pipeline archive {}: {error}",
                    self.path.display()
                )
            })
    }

    fn serialize(&self) -> Result<(), String> {
        let _driver = self
            .driver_lock
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.serialize_locked()
    }
}

#[derive(Clone, Default)]
struct ShaderFunctionCache {
    vertex: Arc<Mutex<HashMap<String, Function>>>,
    pixel: Arc<Mutex<HashMap<String, Function>>>,
}

impl ShaderFunctionCache {
    fn function(
        cache: &Mutex<HashMap<String, Function>>,
        device: &Device,
        stage: &str,
        shader: &remu_shader_metal::MetalShader,
    ) -> Result<(Function, bool), String> {
        if let Some(function) = cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&shader.source)
        {
            return Ok((function.clone(), true));
        }
        let library = compile_metal_shader_library(device, stage, &shader.source)?;
        let function = library
            .get_function(&shader.entry_point, None)
            .map_err(|error| format!("Metal {stage} function: {error}"))?;
        let mut cache = cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(existing) = cache.get(&shader.source) {
            // Another compiler may have completed the same stage while this one was in Apple's
            // compiler. Keep one canonical retained function for all later pipeline variants.
            return Ok((existing.clone(), true));
        }
        cache.insert(shader.source.clone(), function.clone());
        Ok((function, false))
    }

    fn vertex(
        &self,
        device: &Device,
        shader: &remu_shader_metal::MetalShader,
    ) -> Result<(Function, bool), String> {
        Self::function(&self.vertex, device, "vertex", shader)
    }

    fn pixel(
        &self,
        device: &Device,
        shader: &remu_shader_metal::MetalShader,
    ) -> Result<(Function, bool), String> {
        Self::function(&self.pixel, device, "pixel", shader)
    }
}

struct ScheduledGpuSubmission {
    ingress_sequence: u64,
    events: VecDeque<GpuEvent>,
}

struct ScheduledHostEvent {
    ingress_sequence: u64,
    event: GpuEvent,
}

#[derive(Clone, Debug)]
enum QueueBlockReason {
    GuestMemoryWait,
    GraphicsPipeline(GraphicsPipelineKey),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum OrderedEventSource {
    Host { ingress_sequence: u64 },
    GuestQueue(u64),
}

struct OrderedGpuEventReceiver {
    receiver: Receiver<GpuEvent>,
    submissions: HashMap<u64, VecDeque<ScheduledGpuSubmission>>,
    queue_order: Vec<u64>,
    next_queue_index: usize,
    active_queue: Option<u64>,
    blocked_queues: HashMap<u64, QueueBlockReason>,
    host_events: VecDeque<ScheduledHostEvent>,
    host_block: Option<QueueBlockReason>,
    next_ingress_sequence: u64,
    disconnected: bool,
}

#[derive(Debug)]
enum OrderedEventPoll {
    Event {
        source: OrderedEventSource,
        event: GpuEvent,
    },
    Blocked,
    Empty,
    Disconnected,
}

impl OrderedGpuEventReceiver {
    fn new(receiver: Receiver<GpuEvent>) -> Self {
        Self {
            receiver,
            submissions: HashMap::new(),
            queue_order: Vec::new(),
            next_queue_index: 0,
            active_queue: None,
            blocked_queues: HashMap::new(),
            host_events: VecDeque::new(),
            host_block: None,
            next_ingress_sequence: 0,
            disconnected: false,
        }
    }

    fn ingest(&mut self, receive_limit: usize) {
        if self.disconnected {
            return;
        }
        for _ in 0..receive_limit {
            let event = match self.receiver.try_recv() {
                Ok(event) => {
                    remu_gpu::record_gpu_event_consumed();
                    event
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.disconnected = true;
                    break;
                }
            };
            let ingress_sequence = self.next_ingress_sequence;
            self.next_ingress_sequence = self.next_ingress_sequence.saturating_add(1);
            match event {
                GpuEvent::QueueSubmission { queue_id, events } => {
                    if events.is_empty() {
                        continue;
                    }
                    if !self.submissions.contains_key(&queue_id) {
                        self.queue_order.push(queue_id);
                    }
                    self.submissions.entry(queue_id).or_default().push_back(
                        ScheduledGpuSubmission {
                            ingress_sequence,
                            events: events.into(),
                        },
                    );
                }
                event => self.host_events.push_back(ScheduledHostEvent {
                    ingress_sequence,
                    event,
                }),
            }
        }
    }

    fn try_recv(
        &mut self,
        pipeline_resolved: impl Fn(&GraphicsPipelineKey) -> bool,
    ) -> OrderedEventPoll {
        self.ingest(GPU_EVENTS_PER_TICK);
        self.blocked_queues.retain(|_, reason| match reason {
            QueueBlockReason::GuestMemoryWait => true,
            QueueBlockReason::GraphicsPipeline(key) => !pipeline_resolved(key),
        });
        if self.host_block.as_ref().is_some_and(|reason| match reason {
            QueueBlockReason::GuestMemoryWait => false,
            QueueBlockReason::GraphicsPipeline(key) => pipeline_resolved(key),
        }) {
            self.host_block = None;
        }

        let oldest_submission = self
            .submissions
            .values()
            .filter_map(|queue| queue.front())
            .map(|submission| submission.ingress_sequence)
            .min();
        if self.host_block.is_none()
            && self.host_events.front().is_some_and(|host| {
                oldest_submission.is_none_or(|oldest| host.ingress_sequence < oldest)
            })
        {
            let host = self
                .host_events
                .pop_front()
                .expect("host event exists after readiness check");
            return OrderedEventPoll::Event {
                source: OrderedEventSource::Host {
                    ingress_sequence: host.ingress_sequence,
                },
                event: host.event,
            };
        }

        if let Some(queue_id) = self.active_queue {
            if !self.blocked_queues.contains_key(&queue_id)
                && let Some(event) = self.pop_queue_event(queue_id)
            {
                return OrderedEventPoll::Event {
                    source: OrderedEventSource::GuestQueue(queue_id),
                    event,
                };
            }
            self.active_queue = None;
        }

        for offset in 0..self.queue_order.len() {
            let index = (self.next_queue_index + offset) % self.queue_order.len();
            let queue_id = self.queue_order[index];
            if self.blocked_queues.contains_key(&queue_id) {
                continue;
            }
            if let Some(event) = self.pop_queue_event(queue_id) {
                self.active_queue = Some(queue_id);
                return OrderedEventPoll::Event {
                    source: OrderedEventSource::GuestQueue(queue_id),
                    event,
                };
            }
        }

        if self.pending_event_count() != 0 {
            OrderedEventPoll::Blocked
        } else if self.disconnected {
            return OrderedEventPoll::Disconnected;
        } else {
            OrderedEventPoll::Empty
        }
    }

    fn pop_queue_event(&mut self, queue_id: u64) -> Option<GpuEvent> {
        let queue = self.submissions.get_mut(&queue_id)?;
        while queue
            .front()
            .is_some_and(|submission| submission.events.is_empty())
        {
            queue.pop_front();
        }
        queue.front_mut()?.events.pop_front()
    }

    fn finish_event(&mut self, source: OrderedEventSource) {
        let OrderedEventSource::GuestQueue(queue_id) = source else {
            return;
        };
        let completed_submission = self.submissions.get_mut(&queue_id).is_some_and(|queue| {
            if queue
                .front()
                .is_some_and(|submission| submission.events.is_empty())
            {
                queue.pop_front();
                true
            } else {
                false
            }
        });
        if completed_submission {
            self.blocked_queues.remove(&queue_id);
            self.active_queue = None;
            self.advance_after(queue_id);
        }
    }

    fn block_event(
        &mut self,
        source: OrderedEventSource,
        event: GpuEvent,
        reason: QueueBlockReason,
    ) {
        match source {
            OrderedEventSource::Host { ingress_sequence } => {
                self.host_events.push_front(ScheduledHostEvent {
                    ingress_sequence,
                    event,
                });
                self.host_block = Some(reason);
            }
            OrderedEventSource::GuestQueue(queue_id) => {
                self.submissions
                    .get_mut(&queue_id)
                    .and_then(|queue| queue.front_mut())
                    .expect("blocked guest event retains its owning submission")
                    .events
                    .push_front(event);
                self.blocked_queues.insert(queue_id, reason);
                self.active_queue = None;
                self.advance_after(queue_id);
            }
        }
    }

    fn advance_after(&mut self, queue_id: u64) {
        if let Some(index) = self.queue_order.iter().position(|&id| id == queue_id) {
            self.next_queue_index = (index + 1) % self.queue_order.len().max(1);
        }
    }

    fn retry_guest_memory_waits(&mut self) {
        self.blocked_queues
            .retain(|_, reason| !matches!(reason, QueueBlockReason::GuestMemoryWait));
        if matches!(self.host_block, Some(QueueBlockReason::GuestMemoryWait)) {
            self.host_block = None;
        }
    }

    fn prefetch_draws(&mut self, receive_limit: usize, draw_limit: usize) -> Vec<IndexedDraw> {
        self.ingest(receive_limit);
        let host_draws = self
            .host_events
            .iter()
            .filter_map(|host| match &host.event {
                GpuEvent::IndexedDraw(draw) => Some(draw.as_ref().clone()),
                _ => None,
            });
        let queue_draws = self.queue_order.iter().flat_map(|queue_id| {
            self.submissions
                .get(queue_id)
                .into_iter()
                .flatten()
                .flat_map(|submission| submission.events.iter())
                .filter_map(|event| match event {
                    GpuEvent::IndexedDraw(draw) => Some(draw.as_ref().clone()),
                    _ => None,
                })
        });
        host_draws.chain(queue_draws).take(draw_limit).collect()
    }

    fn pending_event_count(&self) -> usize {
        self.host_events.len().saturating_add(
            self.submissions
                .values()
                .flatten()
                .map(|submission| submission.events.len())
                .sum::<usize>(),
        )
    }
}

struct UploadedImageTexture {
    texture: Texture,
    descriptor: DecodedImageDescriptor,
    writable: bool,
}

struct ResidentGuestTexture {
    texture: Texture,
    component_reorder_view: Option<Texture>,
    descriptor: GuestTextureDescriptor,
    texture_type: u32,
    layer_count: u32,
    mip_level_count: u32,
    write_watch: GuestTextureWriteWatch,
    gpu_written: bool,
    content_version: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SampledTextureKey {
    descriptor: GuestTextureDescriptor,
    destination_select: u32,
    base_mip_level: u32,
    mip_level_count: u32,
    texture_type: u32,
    layer_count: u32,
    comparison: bool,
}

struct CachedSampledTexture {
    texture: Texture,
    write_watch: Arc<GuestTextureWriteWatch>,
    admission_id: u64,
}

struct GuestTextureWriteWatch {
    address: u64,
    byte_count: u64,
    ranges: Vec<CpuWriteWatch>,
}

impl GuestTextureWriteWatch {
    fn new(address: u64, byte_count: u64) -> Result<Self, String> {
        let segments = guest_shared_backing_alias_segments(address, byte_count).ok_or_else(|| {
            format!(
                "guest texture range {address:#x}+{byte_count:#x} has no complete live shared backing"
            )
        })?;
        let ranges = segments
            .into_iter()
            .map(|segment| {
                // SAFETY: the backing resolver returned live readable/writable CPU aliases that
                // cover this exact GPU-physical resource span.
                unsafe { CpuWriteWatch::new(segment.address, segment.byte_count) }.map_err(
                    |error| {
                        format!(
                            "could not watch guest texture CPU writes at {:#x}+{:#x}: {error}",
                            segment.address, segment.byte_count
                        )
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            address,
            byte_count,
            ranges,
        })
    }

    fn retirement(&self) -> Option<CpuWriteWatchRetirement> {
        let mut retirement = None;
        for watch in &self.ranges {
            match watch.retirement() {
                Some(CpuWriteWatchRetirement::Replaced) => {
                    return Some(CpuWriteWatchRetirement::Replaced);
                }
                Some(CpuWriteWatchRetirement::Reprotecting) => {
                    retirement = Some(CpuWriteWatchRetirement::Reprotecting);
                }
                Some(CpuWriteWatchRetirement::Reprotected)
                    if retirement != Some(CpuWriteWatchRetirement::Reprotecting) =>
                {
                    retirement = Some(CpuWriteWatchRetirement::Reprotected);
                }
                None | Some(CpuWriteWatchRetirement::Reprotected) => {}
            }
        }
        retirement
    }

    fn consume_dirty(&self) -> bool {
        self.ranges
            .iter()
            .fold(false, |dirty, watch| watch.consume_dirty() || dirty)
    }

    fn is_dirty(&self) -> bool {
        self.ranges.iter().any(CpuWriteWatch::is_dirty)
    }

    fn precise_tracking(&self) -> bool {
        self.ranges.iter().all(CpuWriteWatch::precise_tracking)
    }

    fn generations(&self) -> Vec<u64> {
        self.ranges.iter().map(CpuWriteWatch::generation).collect()
    }

    fn rearm(&self) -> Result<(), String> {
        for watch in &self.ranges {
            watch
                .rearm()
                .map_err(|error| format!("could not rearm guest texture CPU writes: {error}"))?;
        }
        Ok(())
    }

    fn range(&self) -> (u64, u64) {
        (self.address, self.byte_count)
    }

    fn with_contiguous_mapping<T>(
        &self,
        address: u64,
        byte_count: u64,
        operation: impl FnOnce() -> T,
    ) -> Result<T, String> {
        if let Some(retirement) = self.retirement() {
            return Err(format!(
                "guest texture mapping {address:#x}+{byte_count:#x} is retired ({retirement:?})"
            ));
        }
        let end = address
            .checked_add(byte_count)
            .ok_or("guest texture mapping range overflows")?;
        let watch = self
            .ranges
            .iter()
            .find(|watch| {
                let (watch_address, watch_bytes) = watch.range();
                watch_address <= address
                    && watch_address
                        .checked_add(watch_bytes)
                        .is_some_and(|watch_end| end <= watch_end)
            })
            .ok_or_else(|| {
                format!(
                    "guest texture mapping {address:#x}+{byte_count:#x} has no single covering write watch"
                )
            })?;
        watch.with_live_mapping(operation).map_err(|error| {
            format!(
                "guest texture mapping {address:#x}+{byte_count:#x} was retired during access: {error}"
            )
        })
    }

    fn has_single_covering_mapping(&self, address: u64, byte_count: u64) -> bool {
        let Some(end) = address.checked_add(byte_count) else {
            return false;
        };
        self.ranges.iter().any(|watch| {
            let (watch_address, watch_bytes) = watch.range();
            watch_address <= address
                && watch_address
                    .checked_add(watch_bytes)
                    .is_some_and(|watch_end| end <= watch_end)
        })
    }
}

/// Metal exposes this C structure by value in the swizzled texture-view Objective-C method.
/// Its members are `uint8_t` in the macOS SDK; keep the layout explicit because metal-rs 0.33
/// does not yet bind this overload.
#[repr(C)]
#[derive(Clone, Copy)]
struct MetalTextureSwizzleChannels {
    red: u8,
    green: u8,
    blue: u8,
    alpha: u8,
}

fn new_red_blue_swizzled_texture_view(
    texture: &Texture,
    base_mip_level: u32,
    mip_level_count: u32,
) -> Option<Texture> {
    let underlying_mip_count = base_mip_level.checked_add(mip_level_count)?;
    if mip_level_count == 0 || underlying_mip_count > texture.mipmap_level_count() as u32 {
        return None;
    }
    const RED: u8 = 2;
    const GREEN: u8 = 3;
    const BLUE: u8 = 4;
    const ALPHA: u8 = 5;
    let swizzle = MetalTextureSwizzleChannels {
        red: BLUE,
        green: GREEN,
        blue: RED,
        alpha: ALPHA,
    };
    // SAFETY: `texture` is a live MTLTexture. The selector is available on every supported macOS
    // version (10.15+), the enum arguments use Metal's NSUInteger ABI, NSRange is repr(C), and the
    // swizzle structure above exactly matches MTLTextureSwizzleChannels. A `new...` selector
    // returns an owned (+1) object, which `Texture::from_ptr` assumes.
    let view: *mut Object = unsafe {
        msg_send![
            texture.as_ref(),
            newTextureViewWithPixelFormat: texture.pixel_format()
            textureType: texture.texture_type()
            levels: NSRange::new(u64::from(base_mip_level), u64::from(mip_level_count))
            slices: NSRange::new(0, texture.array_length())
            swizzle: swizzle
        ]
    };
    if view.is_null() {
        None
    } else {
        // SAFETY: The non-null Objective-C result conforms to MTLTexture by method contract and
        // carries the retained ownership documented above.
        Some(unsafe { Texture::from_ptr(view.cast()) })
    }
}

fn resident_texture_for_sample(
    resident: &mut ResidentGuestTexture,
    sampled: DecodedImageDescriptor,
) -> Option<Texture> {
    let underlying_mip_count = sampled
        .base_mip_level
        .checked_add(sampled.mip_level_count.max(1))?;
    if resident.mip_level_count < underlying_mip_count {
        return None;
    }
    if guest_texture_descriptors_share_storage(resident.descriptor, sampled.guest) {
        if sampled.base_mip_level == 0 && sampled.mip_level_count.max(1) == resident.mip_level_count
        {
            return Some(resident.texture.clone());
        }
        return Some(resident.texture.new_texture_view_from_slice(
            resident.texture.pixel_format(),
            MTLTextureType::D2,
            NSRange::new(
                u64::from(sampled.base_mip_level),
                u64::from(sampled.mip_level_count.max(1)),
            ),
            NSRange::new(0, 1),
        ));
    }
    if !guest_texture_descriptors_share_component_reordered_storage(
        resident.descriptor,
        sampled.guest,
    ) {
        return None;
    }
    let sampled_mip_level_count = sampled.mip_level_count.max(1);
    let uses_complete_hierarchy =
        sampled.base_mip_level == 0 && sampled_mip_level_count == resident.mip_level_count;
    if uses_complete_hierarchy && resident.component_reorder_view.is_none() {
        let view = new_red_blue_swizzled_texture_view(
            &resident.texture,
            sampled.base_mip_level,
            sampled_mip_level_count,
        )?;
        view.set_label(&format!(
            "REMu component-reordered guest image {:#x}",
            resident.descriptor.address
        ));
        resident.component_reorder_view = Some(view);
        if trace_target_sampling() {
            static CREATED_COMPONENT_REORDER_VIEWS: AtomicUsize = AtomicUsize::new(0);
            let sequence = CREATED_COMPONENT_REORDER_VIEWS.fetch_add(1, Ordering::Relaxed) + 1;
            if sequence <= 16 || sequence.is_power_of_two() {
                eprintln!(
                    "[REMu][METAL][ALIAS-VIEW] sequence={sequence} address={:#x} rendered={:?} sampled={:?} extent={}x{} tile={}",
                    resident.descriptor.address,
                    resident.descriptor.pixel_format,
                    sampled.guest.pixel_format,
                    sampled.guest.width,
                    sampled.guest.height,
                    sampled.guest.tile_mode,
                );
            }
        }
    }
    if uses_complete_hierarchy {
        resident.component_reorder_view.clone()
    } else {
        let view = new_red_blue_swizzled_texture_view(
            &resident.texture,
            sampled.base_mip_level,
            sampled_mip_level_count,
        )?;
        view.set_label(&format!(
            "REMu component-reordered guest image {:#x} mips {}..{}",
            resident.descriptor.address,
            sampled.base_mip_level,
            sampled.base_mip_level + sampled_mip_level_count,
        ));
        Some(view)
    }
}

fn resident_component_reordered_sample_is_eligible(
    resident: &ResidentGuestTexture,
    sampled: DecodedImageDescriptor,
    comparison: bool,
) -> bool {
    let underlying_mip_count = sampled
        .base_mip_level
        .checked_add(sampled.mip_level_count.max(1));
    !comparison
        && resident.gpu_written
        && resident.texture_type == 9
        && resident.layer_count == 1
        && sampled.texture_type == 9
        && sampled.layer_count == 1
        && underlying_mip_count.is_some_and(|count| resident.mip_level_count >= count)
        && guest_texture_descriptors_share_component_reordered_storage(
            resident.descriptor,
            sampled.guest,
        )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WriteWatchRefresh {
    Active,
    Reprotecting,
    Renewed { was_dirty: bool },
    Replaced,
}

fn refresh_resident_texture_write_watch(
    resident: &mut ResidentGuestTexture,
) -> Result<WriteWatchRefresh, String> {
    match resident.write_watch.retirement() {
        None => Ok(WriteWatchRefresh::Active),
        Some(CpuWriteWatchRetirement::Reprotecting) => Ok(WriteWatchRefresh::Reprotecting),
        Some(CpuWriteWatchRetirement::Reprotected) => {
            let was_dirty = resident.write_watch.is_dirty();
            resident.write_watch = watch_resident_texture(
                resident.descriptor,
                resident.texture_type,
                resident.layer_count,
                resident.mip_level_count,
            )?;
            Ok(WriteWatchRefresh::Renewed { was_dirty })
        }
        Some(CpuWriteWatchRetirement::Replaced) => Ok(WriteWatchRefresh::Replaced),
    }
}

fn watch_resident_texture(
    descriptor: GuestTextureDescriptor,
    texture_type: u32,
    layer_count: u32,
    mip_level_count: u32,
) -> Result<GuestTextureWriteWatch, String> {
    if texture_type == 9 && layer_count == 1 {
        if mip_level_count == 1 {
            watch_guest_texture(descriptor)
        } else {
            let layout = mode27_mip_chain_layout(descriptor, mip_level_count)?;
            GuestTextureWriteWatch::new(
                descriptor.address,
                u64::try_from(layout.byte_count)
                    .map_err(|_| "mip chain byte count does not fit u64")?,
            )
        }
    } else {
        if mip_level_count != 1 {
            return Err(format!(
                "mipmapped resident texture type={texture_type} layers={layer_count} is unsupported"
            ));
        }
        watch_sampled_texture(DecodedImageDescriptor {
            guest: descriptor,
            destination_select: 0,
            base_mip_level: 0,
            mip_level_count: 1,
            texture_type,
            layer_count,
        })
    }
}

fn refresh_sampled_texture_write_watch(
    key: SampledTextureKey,
    cached: &mut CachedSampledTexture,
    write_watches: &mut HashMap<SampledTextureKey, Arc<GuestTextureWriteWatch>>,
) -> Result<WriteWatchRefresh, String> {
    match cached.write_watch.retirement() {
        None => Ok(WriteWatchRefresh::Active),
        Some(CpuWriteWatchRetirement::Reprotecting) => Ok(WriteWatchRefresh::Reprotecting),
        Some(CpuWriteWatchRetirement::Reprotected) => {
            let was_dirty = cached.write_watch.is_dirty();
            let write_watch = Arc::new(watch_sampled_texture(DecodedImageDescriptor {
                guest: key.descriptor,
                destination_select: key.destination_select,
                base_mip_level: key.base_mip_level,
                mip_level_count: key.mip_level_count,
                texture_type: key.texture_type,
                layer_count: key.layer_count,
            })?);
            cached.write_watch = Arc::clone(&write_watch);
            write_watches.insert(key, write_watch);
            Ok(WriteWatchRefresh::Renewed { was_dirty })
        }
        Some(CpuWriteWatchRetirement::Replaced) => Ok(WriteWatchRefresh::Replaced),
    }
}

fn persistent_sampled_texture_write_watch(
    key: SampledTextureKey,
    descriptor: DecodedImageDescriptor,
    write_watches: &mut HashMap<SampledTextureKey, Arc<GuestTextureWriteWatch>>,
) -> Result<Arc<GuestTextureWriteWatch>, String> {
    if let Some(write_watch) = write_watches.get(&key)
        && !matches!(
            write_watch.retirement(),
            Some(CpuWriteWatchRetirement::Reprotected | CpuWriteWatchRetirement::Replaced)
        )
    {
        return Ok(Arc::clone(write_watch));
    }
    write_watches.remove(&key);
    let write_watch = Arc::new(watch_sampled_texture(descriptor)?);
    // The texture cache has the same capacity. Keeping one coherent watch per admitted key lets
    // render-target invalidation discard stale Metal pixels without repeatedly rebuilding the
    // process-wide signal-handler registry. Rare overflow keys retain their watch only as long as
    // their sampled texture entry.
    if write_watches.len() < SAMPLED_TEXTURE_CACHE_CAPACITY {
        write_watches.insert(key, Arc::clone(&write_watch));
    }
    Ok(write_watch)
}

fn retire_sampled_texture(texture: CachedSampledTexture) {
    static RETIRER: OnceLock<mpsc::Sender<CachedSampledTexture>> = OnceLock::new();
    let sender = RETIRER.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<CachedSampledTexture>();
        if let Err(error) = std::thread::Builder::new()
            .name("REMu texture reclaimer".into())
            .spawn(move || {
                while let Ok(texture) = receiver.recv() {
                    drop(texture);
                }
            })
        {
            eprintln!("[REMu][METAL] could not start texture reclaimer: {error}");
        }
        sender
    });
    if let Err(error) = sender.send(texture) {
        drop(error.0);
    }
}

fn retire_sampled_textures_matching(
    cache: &mut HashMap<SampledTextureKey, CachedSampledTexture>,
    mut should_retire: impl FnMut(SampledTextureKey) -> bool,
) {
    // Dropping a write watch republishes the signal-handler's immutable page-owner snapshot.
    // Invalidating a render graph can evict hundreds of sampled aliases at once, so doing those
    // drops inside HashMap::retain stalls the presenter for an entire guest frame. Remove the
    // entries immediately for correctness, then let the existing reclaimer perform their costly
    // destruction away from the event-loop thread.
    let retired = cache
        .keys()
        .copied()
        .filter(|key| should_retire(*key))
        .collect::<Vec<_>>();
    for key in retired {
        if let Some(texture) = cache.remove(&key) {
            retire_sampled_texture(texture);
        }
    }
}

fn retire_overlapping_sampled_textures(
    cache: &mut HashMap<SampledTextureKey, CachedSampledTexture>,
    descriptor: GuestTextureDescriptor,
) {
    retire_sampled_textures_matching(cache, |key| {
        guest_texture_descriptors_overlap(key.descriptor, descriptor)
    });
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GlobalBufferCacheKey {
    base_address: u64,
    byte_count: usize,
}

struct CachedGlobalBuffer {
    bytes: Vec<u8>,
    buffer: Buffer,
}

struct UploadedGlobalBuffer {
    buffer: Buffer,
    offset: u64,
    guest_memory: bool,
}

struct PendingGlobalWriteback {
    buffer: Buffer,
    guest_address: u64,
    byte_count: usize,
}

struct PendingPresentation {
    command_buffer: CommandBuffer,
    submitted_at: Instant,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GuestBatchFlushReason {
    EventLoop,
    BlockedWait,
    FlipSnapshot,
    WorkCapacity,
    CpuSynchronization,
}

struct PresenterStatistics {
    interval_started: Instant,
    last_logged: Instant,
    guest_flips: u32,
    completed_presents: u32,
    submitted_guest_batches: u32,
    coalesced_pipeline_barriers: u32,
    encoded_draws: u32,
    encoded_compute_dispatches: u32,
    encoded_render_passes: u32,
    render_pass_ends_for_compute: u32,
    render_pass_ends_for_target_change: u32,
    render_pass_ends_for_batch_flush: u32,
    batch_flushes_from_event_loop: u32,
    batch_flushes_for_guest_memory_write: u32,
    batch_flushes_for_blocked_wait: u32,
    batch_flushes_for_flip_snapshot: u32,
    batch_flushes_for_work_capacity: u32,
    batch_flushes_for_cpu_synchronization: u32,
    gpu_execution_time: Duration,
    presentation_queue_latency: Duration,
    last_audio_output_frames: u64,
    last_audio_underrun_frames: u64,
}

impl PresenterStatistics {
    fn new() -> Self {
        Self {
            interval_started: Instant::now(),
            last_logged: Instant::now(),
            guest_flips: 0,
            completed_presents: 0,
            submitted_guest_batches: 0,
            coalesced_pipeline_barriers: 0,
            encoded_draws: 0,
            encoded_compute_dispatches: 0,
            encoded_render_passes: 0,
            render_pass_ends_for_compute: 0,
            render_pass_ends_for_target_change: 0,
            render_pass_ends_for_batch_flush: 0,
            batch_flushes_from_event_loop: 0,
            batch_flushes_for_guest_memory_write: 0,
            batch_flushes_for_blocked_wait: 0,
            batch_flushes_for_flip_snapshot: 0,
            batch_flushes_for_work_capacity: 0,
            batch_flushes_for_cpu_synchronization: 0,
            gpu_execution_time: Duration::ZERO,
            presentation_queue_latency: Duration::ZERO,
            last_audio_output_frames: remu_gpu::audio_output_frames(),
            last_audio_underrun_frames: remu_gpu::audio_underrun_frames(),
        }
    }
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

fn merge_guest_batch_result(
    current: Result<(), String>,
    operation: &str,
    next: Result<(), String>,
) -> Result<(), String> {
    match (current, next) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(format!("{operation}: {error}")),
        (Err(primary), Err(secondary)) => {
            Err(format!("{primary}; {operation} also failed: {secondary}"))
        }
    }
}

fn compute_pipeline_key(
    dispatch: &ComputeDispatch,
    evaluation: &ShaderEvaluation,
    program: &std::sync::Arc<remu_shader::Program>,
) -> ComputePipelineKey {
    let image_slots = evaluation.image_binding_slots();
    ComputePipelineKey {
        // Retaining the Arc makes pointer identity stable for the lifetime of both successful and
        // negative cache entries; an allocator cannot recycle a dropped Program into this key.
        program_identity: ProgramIdentity(Arc::clone(program)),
        lds_size_bytes: dispatch.launch.lds_size_bytes,
        wave_lane_count: dispatch.launch.wave_lane_count,
        compute_system_registers: evaluation.compute_system_registers,
        initial_scalar_count: evaluation.initial_scalar_registers.len(),
        global_bindings: evaluation
            .global_memory_bindings
            .iter()
            .map(|binding| (binding.instruction_pcs.to_vec(), 0, 0, binding.writable))
            .collect(),
        image_bindings: evaluation
            .image_bindings
            .iter()
            .enumerate()
            .map(|(index, binding)| {
                let storage = is_storage_image_opcode(binding.opcode);
                let comparison = image_is_comparison(binding.opcode);
                let texture_type = binding
                    .resource_descriptor
                    .get(3)
                    .map_or(0, |word| (word >> 28) & 0xf);
                (
                    image_component_class(binding),
                    storage,
                    comparison,
                    binding.control.dimension,
                    texture_type,
                    image_slots.texture_slots[index],
                    image_slots.sampler_slots[index],
                )
            })
            .collect(),
    }
}

fn program_embeds_global_base(program: &remu_shader::Program) -> bool {
    program.instructions.iter().any(|instruction| {
        matches!(
            instruction.control.as_ref(),
            Some(remu_shader::InstructionControl::GlobalMemory(_))
        )
    })
}

fn is_writable_image_opcode(opcode: remu_shader::Opcode) -> bool {
    opcode.starts_with("ImageStore") || opcode.starts_with("ImageAtomic")
}

fn is_storage_image_opcode(opcode: remu_shader::Opcode) -> bool {
    is_writable_image_opcode(opcode) || opcode.starts_with("ImageLoad")
}

fn image_resource_descriptor_is_null(words: &[u32]) -> bool {
    !words.is_empty() && words.iter().all(|word| *word == 0)
}

struct GuestCommandBatch {
    command_buffer: CommandBuffer,
    work: Vec<GuestBatchWork>,
    global_writebacks: Vec<PendingGlobalWriteback>,
    guest_write_ranges: Vec<PendingGuestWriteRange>,
    memory_writes: Vec<PendingGuestMemoryWrite>,
    render_pass: Option<OpenGuestRenderPass>,
    /// Guest submissions whose final ordered work is contained in this Metal batch.
    ///
    /// Queue scheduling can complete a newer global sequence before an older blocked queue, so
    /// every boundary is retained rather than incorrectly treating the maximum as a prefix.
    completion_sequences: Vec<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingGuestWriteRange {
    address: u64,
    byte_count: u64,
}

#[derive(Clone, Debug)]
enum PendingGuestMemoryWrite {
    Release(GpuMemoryRelease),
    Write(GpuMemoryWrite),
    Fill {
        address: u64,
        value: u32,
        dword_count: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GuestMemoryWaitStatus {
    Satisfied,
    Blocked,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GuestMemoryWaitKey {
    source: OrderedEventSource,
    address: u64,
    reference: u64,
    mask: u64,
    compare_function: u32,
    is_64_bit: bool,
}

impl GuestMemoryWaitKey {
    const fn new(source: OrderedEventSource, wait: GpuMemoryWait) -> Self {
        Self {
            source,
            address: wait.address,
            reference: wait.reference,
            mask: wait.mask,
            compare_function: wait.compare_function,
            is_64_bit: wait.is_64_bit,
        }
    }

    const fn wait(self) -> GpuMemoryWait {
        GpuMemoryWait {
            address: self.address,
            reference: self.reference,
            mask: self.mask,
            compare_function: self.compare_function,
            is_64_bit: self.is_64_bit,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ActiveGuestMemoryWait {
    latched_value: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GuestMemoryWaitRecovery {
    None,
    ProducerLatched(u64),
}

#[derive(Default)]
struct GuestMemoryWaitTracker {
    active: HashMap<GuestMemoryWaitKey, ActiveGuestMemoryWait>,
}

impl GuestMemoryWaitTracker {
    fn observe_unsatisfied(&mut self, key: GuestMemoryWaitKey) -> GuestMemoryWaitRecovery {
        let active = self.active.entry(key).or_insert(ActiveGuestMemoryWait {
            latched_value: None,
        });
        if let Some(value) = active.latched_value {
            self.active.remove(&key);
            return GuestMemoryWaitRecovery::ProducerLatched(value);
        }
        GuestMemoryWaitRecovery::None
    }

    fn complete(&mut self, key: GuestMemoryWaitKey) {
        self.active.remove(&key);
    }

    fn record_produced(&mut self, address: u64, is_64_bit: bool, value: u64) {
        for (key, active) in &mut self.active {
            if key.address == address
                && key.is_64_bit == is_64_bit
                && key.wait().is_satisfied_by(value)
            {
                active.latched_value = Some(value);
            }
        }
    }
}

impl PendingGuestMemoryWrite {
    fn range(&self) -> (u64, u64) {
        match self {
            Self::Release(release) => (
                release.address,
                match release.value {
                    GpuMemoryReleaseValue::Immediate32(_) => 4,
                    GpuMemoryReleaseValue::Immediate64(_) | GpuMemoryReleaseValue::Timestamp => 8,
                },
            ),
            Self::Write(write) => (
                write.address,
                if write.increment_address {
                    u64::try_from(write.values.len())
                        .unwrap_or(u64::MAX / 4)
                        .saturating_mul(4)
                } else {
                    4
                },
            ),
            Self::Fill {
                address,
                dword_count,
                ..
            } => (*address, dword_count.saturating_mul(4)),
        }
    }

    fn value_at(&self, address: u64, is_64_bit: bool) -> Option<u64> {
        match self {
            Self::Release(release) if release.address == address => match release.value {
                GpuMemoryReleaseValue::Immediate32(value) if !is_64_bit => Some(u64::from(value)),
                GpuMemoryReleaseValue::Immediate64(value) => Some(value),
                GpuMemoryReleaseValue::Timestamp | GpuMemoryReleaseValue::Immediate32(_) => None,
            },
            Self::Write(write) if !is_64_bit => write
                .values
                .iter()
                .copied()
                .enumerate()
                .rev()
                .find_map(|(index, value)| {
                    let offset = if write.increment_address {
                        u64::try_from(index).ok()?.checked_mul(4)?
                    } else {
                        0
                    };
                    (write.address.checked_add(offset)? == address).then_some(u64::from(value))
                }),
            Self::Fill {
                address: fill_address,
                value,
                dword_count,
            } => {
                let offset = address.checked_sub(*fill_address)?;
                if !offset.is_multiple_of(4) {
                    return None;
                }
                let index = offset / 4;
                let required_dwords = if is_64_bit { 2 } else { 1 };
                (index.checked_add(required_dwords)? <= *dword_count).then_some(if is_64_bit {
                    u64::from(*value) | (u64::from(*value) << 32)
                } else {
                    u64::from(*value)
                })
            }
            Self::Write(_) | Self::Release(_) => None,
        }
    }

    fn overlaps(&self, address: u64, byte_count: u64) -> bool {
        let (write_address, write_byte_count) = self.range();
        guest_ranges_overlap(write_address, write_byte_count, address, byte_count)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GuestRenderPassKey {
    targets: Vec<(GuestTextureDescriptor, u32, bool, u32, u32, u32, u32, u32)>,
    depth: Option<(GuestTextureDescriptor, u32)>,
    resolve_multisample_targets: bool,
}

struct OpenGuestRenderPass {
    key: GuestRenderPassKey,
    encoder: RenderCommandEncoder,
}

impl GuestCommandBatch {
    fn end_render_pass(&mut self) -> bool {
        if let Some(pass) = self.render_pass.take() {
            pass.encoder.end_encoding();
            true
        } else {
            false
        }
    }
}

fn guest_batch_writes_overlap(batch: &GuestCommandBatch, address: u64, byte_count: u64) -> bool {
    batch
        .guest_write_ranges
        .iter()
        .any(|range| guest_ranges_overlap(range.address, range.byte_count, address, byte_count))
        || batch.global_writebacks.iter().any(|writeback| {
            u64::try_from(writeback.byte_count).is_ok_and(|length| {
                guest_ranges_overlap(writeback.guest_address, length, address, byte_count)
            })
        })
        || batch
            .memory_writes
            .iter()
            .any(|write| write.overlaps(address, byte_count))
}

fn publish_guest_batch_completion(batch: &GuestCommandBatch) {
    for &sequence in &batch.completion_sequences {
        remu_gpu::record_gpu_submission_completed(sequence);
    }
}

fn completed_command_buffer_gpu_duration(command_buffer: &CommandBuffer) -> Duration {
    // Metal exposes GPUStartTime/GPUEndTime only after a command buffer completes. The metal-rs
    // version used here does not wrap them yet, so query the public MTLCommandBuffer selectors
    // directly. Unlike wall time from commit to retirement, this excludes time waiting behind
    // older command buffers and is therefore suitable for the title bar's GPU utilization.
    let (start, end): (f64, f64) = unsafe {
        (
            msg_send![command_buffer.as_ref(), GPUStartTime],
            msg_send![command_buffer.as_ref(), GPUEndTime],
        )
    };
    if !start.is_finite() || !end.is_finite() || start <= 0.0 || end < start {
        return Duration::ZERO;
    }
    Duration::from_secs_f64(end - start)
}

#[derive(Clone, Copy, Debug)]
struct DecodedImageDescriptor {
    guest: GuestTextureDescriptor,
    destination_select: u32,
    base_mip_level: u32,
    mip_level_count: u32,
    texture_type: u32,
    layer_count: u32,
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
    pending_flip_queue: Option<u64>,
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
    scanout_snapshot_textures: HashMap<u64, ScanoutSnapshotTexture>,
    multisample_textures: HashMap<MultisampleTextureKey, MultisampleResidentTexture>,
    sampled_texture_cache: HashMap<SampledTextureKey, CachedSampledTexture>,
    sampled_texture_write_watches: HashMap<SampledTextureKey, Arc<GuestTextureWriteWatch>>,
    fallback_texture_cache: HashMap<SampledTextureKey, Texture>,
    sampled_texture_admission_order: VecDeque<(SampledTextureKey, u64)>,
    next_sampled_texture_admission_id: u64,
    sampler_argument_cache: HashMap<SamplerArgumentCacheKey, CachedSamplerArguments>,
    depth_stencil_cache: HashMap<(u64, bool), DepthStencilState>,
    global_buffer_cache: HashMap<GlobalBufferCacheKey, CachedGlobalBuffer>,
    transient_global_buffer_cache: HashMap<GlobalBufferCacheKey, CachedGlobalBuffer>,
    guest_global_buffer_cache: HashMap<(u64, usize), Buffer>,
    global_buffer_cache_bytes: usize,
    index_buffer_cache: HashMap<Vec<u32>, Buffer>,
    diagnostic_vertex_position_buffer: Option<Buffer>,
    diagnostic_pixel_output_buffer: Option<Buffer>,
    resident_content_version: u64,
    open_guest_batch: Option<GuestCommandBatch>,
    committed_guest_batches: VecDeque<GuestCommandBatch>,
    guest_memory_waits: GuestMemoryWaitTracker,
    pending_presentations: VecDeque<PendingPresentation>,
    statistics: PresenterStatistics,
    capture_directory: Option<PathBuf>,
    pressed_keys: HashSet<KeyCode>,
    shader_compile_sender: SyncSender<ShaderCompileRequest>,
    shader_compile_receiver: Receiver<ShaderCompileResult>,
    render_pipeline_archive: Option<PersistentRenderPipelineArchive>,
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
        let shader_compile_requests = Arc::new(Mutex::new(shader_compile_requests));
        let function_cache = ShaderFunctionCache::default();
        let render_pipeline_archive = if std::env::var_os("REMU_METAL_COMPILER").is_some() {
            match PersistentRenderPipelineArchive::open(&device) {
                Ok(archive) => Some(archive),
                Err(error) => {
                    eprintln!(
                        "[REMu][METAL] persistent render pipeline cache unavailable: {error}"
                    );
                    None
                }
            }
        } else {
            None
        };
        for worker in 0..SHADER_COMPILER_WORKER_COUNT {
            let compiler_device = device.clone();
            let shader_compile_requests = Arc::clone(&shader_compile_requests);
            let shader_compile_results = shader_compile_results.clone();
            let function_cache = function_cache.clone();
            let render_pipeline_archive = render_pipeline_archive.clone();
            std::thread::Builder::new()
                .name(format!("remu-metal-compiler-{worker}"))
                .spawn(move || {
                    loop {
                        let request = shader_compile_requests
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .recv();
                        let Ok(ShaderCompileRequest { key, draw }) = request else {
                            break;
                        };
                        let compiled = objc::rc::autoreleasepool(|| {
                            Self::compile_draw_shader_pair(
                                &compiler_device,
                                &draw,
                                &key,
                                &function_cache,
                                render_pipeline_archive.as_ref(),
                            )
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
        }
        drop(shader_compile_results);
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
            pending_flip_queue: None,
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
            scanout_snapshot_textures: HashMap::new(),
            multisample_textures: HashMap::new(),
            sampled_texture_cache: HashMap::new(),
            sampled_texture_write_watches: HashMap::new(),
            fallback_texture_cache: HashMap::new(),
            sampled_texture_admission_order: VecDeque::new(),
            next_sampled_texture_admission_id: 0,
            sampler_argument_cache: HashMap::new(),
            depth_stencil_cache: HashMap::new(),
            global_buffer_cache: HashMap::new(),
            transient_global_buffer_cache: HashMap::new(),
            guest_global_buffer_cache: HashMap::new(),
            global_buffer_cache_bytes: 0,
            index_buffer_cache: HashMap::new(),
            diagnostic_vertex_position_buffer: None,
            diagnostic_pixel_output_buffer: None,
            resident_content_version: 0,
            open_guest_batch: None,
            committed_guest_batches: VecDeque::new(),
            guest_memory_waits: GuestMemoryWaitTracker::default(),
            pending_presentations: VecDeque::new(),
            statistics: PresenterStatistics::new(),
            capture_directory: std::env::var_os("REMU_CAPTURE_DIR").map(PathBuf::from),
            pressed_keys: HashSet::new(),
            shader_compile_sender,
            shader_compile_receiver,
            render_pipeline_archive,
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
        // Guest render targets are linear (including the RGBA16F scanout used by Unity). An
        // sRGB drawable applies the display transfer function while storing into the 8-bit
        // swapchain. Presenting into BGRA8Unorm copied those linear values verbatim and crushed
        // the entire lower end of the scene toward black.
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm_sRGB);
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

    fn finalize_pending_flip(&mut self) {
        let (Some(display), Some(flip)) = (self.display, self.pending_flip) else {
            return;
        };
        match self.activate_translated_scanout(flip.address, display.width, display.height) {
            Ok(true) => {
                self.uploaded_flip_sequence = flip.sequence;
            }
            Ok(false) => {
                if let Err(error) = self.synchronize_flip_target(flip.address) {
                    eprintln!("[REMu][METAL] flip synchronization failed: {error}");
                }
            }
            Err(error) => {
                eprintln!("[REMu][METAL] translated scanout synchronization failed: {error}");
            }
        }
        self.redraw_pending = true;
    }

    #[allow(clippy::too_many_lines)]
    fn receive_events(&mut self) -> PresenterFlow {
        self.receive_shader_compilations();
        self.retire_completed_guest_batches();
        self.retire_completed_presentations();
        // A guest label may have been published by a Metal batch that retired since the previous
        // AppKit tick. Retry suspended queue fronts once; an unsatisfied wait is re-suspended
        // without permitting any later event from that queue to overtake it.
        self.receiver.retry_guest_memory_waits();
        // A bounded batch prevents a continuously producing guest from starving redraw and input
        // delivery. Queue-local shader and memory waits leave independent guest queues runnable.
        for _ in 0..GPU_EVENTS_PER_TICK {
            let (source, event) = match self.receiver.try_recv(|key| {
                self.compiled_shader_pairs.contains_key(key)
                    || self.failed_shader_pairs.contains(key)
            }) {
                OrderedEventPoll::Event { source, event } => (source, event),
                OrderedEventPoll::Blocked => {
                    // Look ahead far enough to feed all compiler workers while memory-wait queues
                    // remain suspended. Prefetching never executes or reorders an event.
                    for draw in self.receiver.prefetch_draws(512, 128) {
                        if let Ok(key) = graphics_pipeline_key(&draw) {
                            self.compile_draw_shaders(&draw, &key);
                        }
                    }
                    return PresenterFlow::Continue;
                }
                OrderedEventPoll::Empty => {
                    return PresenterFlow::Continue;
                }
                OrderedEventPoll::Disconnected => {
                    eprintln!("[REMu][METAL] GPU event channel disconnected; closing presenter");
                    if let Err(error) = self.wait_for_guest_command_batches() {
                        eprintln!("[REMu][METAL] shutdown synchronization failed: {error}");
                    }
                    self.persist_render_pipeline_archive();
                    return PresenterFlow::Shutdown;
                }
            };
            match event {
                GpuEvent::QueueSubmission { .. } => {
                    eprintln!("[REMu][METAL] ignored invalid nested GPU queue submission");
                }
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
                    self.statistics.guest_flips = self.statistics.guest_flips.saturating_add(1);
                    self.display = Some(DisplayConfiguration {
                        width,
                        height,
                        pitch,
                        pixel_format,
                        tiling_mode,
                    });
                    self.next_flip_sequence = self.next_flip_sequence.saturating_add(1);
                    self.pending_flip = Some(PendingFlip {
                        address,
                        sequence: self.next_flip_sequence,
                    });
                    self.pending_flip_queue = match source {
                        OrderedEventSource::GuestQueue(queue_id) => Some(queue_id),
                        OrderedEventSource::Host { .. } => {
                            self.finalize_pending_flip();
                            None
                        }
                    };
                }
                GpuEvent::PipelineBarrier(barrier) => self.end_guest_pipeline_barrier(barrier),
                GpuEvent::MemoryAcquire(acquire) => {
                    if let Err(error) = self.synchronize_cpu_written_guest_resources(acquire) {
                        eprintln!("[REMu][METAL] ACQUIRE_MEM synchronization failed: {error}");
                    }
                }
                GpuEvent::MemoryCopy(copy) => {
                    if let Err(error) = self.execute_guest_memory_copy(copy) {
                        eprintln!("[REMu][METAL] DMA_DATA synchronization failed: {error}");
                    }
                }
                GpuEvent::MemoryWrite(write) => {
                    if let Err(error) = self.execute_guest_memory_write(&write) {
                        eprintln!("[REMu][METAL] WRITE_DATA synchronization failed: {error}");
                    }
                    self.receiver.retry_guest_memory_waits();
                }
                GpuEvent::MemoryRelease(release) => {
                    if let Err(error) = self.execute_guest_memory_release(release) {
                        eprintln!("[REMu][METAL] RELEASE_MEM synchronization failed: {error}");
                    }
                    self.receiver.retry_guest_memory_waits();
                }
                GpuEvent::MemoryWait(wait) => {
                    match self.execute_guest_memory_wait(source, None, wait) {
                        Ok(GuestMemoryWaitStatus::Satisfied) => {}
                        Ok(GuestMemoryWaitStatus::Blocked) => {
                            self.receiver.block_event(
                                source,
                                GpuEvent::MemoryWait(wait),
                                QueueBlockReason::GuestMemoryWait,
                            );
                            return PresenterFlow::Continue;
                        }
                        Err(error) => {
                            eprintln!("[REMu][METAL] WAIT_REG_MEM synchronization failed: {error}");
                        }
                    }
                }
                GpuEvent::RegisteredMemoryWait { id, wait } => {
                    match self.execute_guest_memory_wait(source, Some(id), wait) {
                        Ok(GuestMemoryWaitStatus::Satisfied) => {}
                        Ok(GuestMemoryWaitStatus::Blocked) => {
                            self.receiver.block_event(
                                source,
                                GpuEvent::RegisteredMemoryWait { id, wait },
                                QueueBlockReason::GuestMemoryWait,
                            );
                            return PresenterFlow::Continue;
                        }
                        Err(error) => {
                            eprintln!(
                                "[REMu][METAL] registered WAIT_REG_MEM synchronization failed: {error}"
                            );
                        }
                    }
                }
                GpuEvent::ComputeDispatch(dispatch) => {
                    match self.execute_compute_dispatch(&dispatch) {
                        Ok(()) => {
                            self.statistics.encoded_compute_dispatches =
                                self.statistics.encoded_compute_dispatches.saturating_add(1);
                        }
                        Err(error)
                            if self
                                .failed_compute_execution
                                .insert(dispatch.shader_address) =>
                        {
                            eprintln!(
                                "[REMu][METAL] Gen5 compute cs={:#x} could not execute: {error}",
                                dispatch.shader_address
                            );
                        }
                        Err(_) => {}
                    }
                }
                GpuEvent::IndexedDraw(draw) => {
                    let key = graphics_pipeline_key(&draw);
                    if let Ok(key) = key.as_ref() {
                        self.compile_draw_shaders(&draw, key);
                        if !self.compiled_shader_pairs.contains_key(key)
                            && !self.failed_shader_pairs.contains(key)
                        {
                            debug_assert!(self.compiling_shader_pairs.contains(key));
                            self.receiver.block_event(
                                source,
                                GpuEvent::IndexedDraw(draw),
                                QueueBlockReason::GraphicsPipeline(key.clone()),
                            );
                            return PresenterFlow::Continue;
                        }
                    }
                    let result = match key.as_ref() {
                        Ok(key) => self.execute_compiled_draw(&draw, key),
                        Err(error) => Err(error.clone()),
                    };
                    if let Err(error) = result {
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
                            let pipeline_key_error =
                                key.as_ref().err().map_or("none", |error| error.as_str());
                            eprintln!(
                                "[REMu][METAL] no translated Metal path for es={:#x}/{:x?} ps={:#x}/{:x?} indexed={} count={} instances={} target={:#x} cb_info={:#x} cb_attrib2={:#x} cb_attrib3={:#x} target_mask={:#x} ps_scalars={pixel_scalars:x?} pipeline_key_error={pipeline_key_error}",
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
                    } else {
                        self.statistics.encoded_draws =
                            self.statistics.encoded_draws.saturating_add(1);
                    }
                }
                GpuEvent::SubmissionBoundary(sequence) => {
                    // Batch across guest submissions without losing their individual completion
                    // identities. Graphics and compute queues can retire out of global sequence
                    // order, and the shared completion tracker exposes only the contiguous prefix.
                    if let Some(batch) = self.open_guest_batch.as_mut() {
                        batch.completion_sequences.push(sequence);
                    } else if let Some(batch) = self.committed_guest_batches.back_mut() {
                        // Empty submissions and boundaries following a forced flush still depend
                        // on all earlier ordered work. Attaching them to the newest committed
                        // batch avoids publishing completion before that work reaches Metal.
                        batch.completion_sequences.push(sequence);
                    } else {
                        // No open or committed work remains, so this submission is genuinely
                        // empty and can complete immediately.
                        remu_gpu::record_gpu_submission_completed(sequence);
                    }
                    if matches!(
                        (source, self.pending_flip_queue),
                        (OrderedEventSource::GuestQueue(queue_id), Some(pending_queue))
                            if queue_id == pending_queue
                    ) {
                        self.pending_flip_queue = None;
                        self.finalize_pending_flip();
                    }
                }
                GpuEvent::Shutdown => {
                    eprintln!("[REMu][METAL] guest runtime requested presenter shutdown");
                    if let Err(error) = self.wait_for_guest_command_batches() {
                        eprintln!("[REMu][METAL] shutdown synchronization failed: {error}");
                    }
                    self.persist_render_pipeline_archive();
                    return PresenterFlow::Shutdown;
                }
            }
            self.receiver.finish_event(source);
            // Do not let a large guest GPU backlog delay an already-requested presentation.
            // In particular, the first flip can precede creation of its translated scanout
            // texture; once a later draw produces that texture, return to Winit immediately so
            // `about_to_wait` can request the redraw instead of processing hundreds more events.
            if self.redraw_pending {
                return PresenterFlow::Continue;
            }
        }
        PresenterFlow::Continue
    }

    fn persist_render_pipeline_archive(&self) {
        if let Some(archive) = self.render_pipeline_archive.as_ref()
            && let Err(error) = archive.serialize()
        {
            eprintln!("[REMu][METAL] {error}");
        }
    }

    fn compile_draw_shaders(&mut self, draw: &IndexedDraw, key: &GraphicsPipelineKey) {
        if self.compiled_shader_pairs.contains_key(key)
            || self.compiling_shader_pairs.contains(key)
            || self.failed_shader_pairs.contains(key)
        {
            return;
        }
        let request = ShaderCompileRequest {
            key: key.clone(),
            draw: draw.clone(),
        };
        match self.shader_compile_sender.try_send(request) {
            Ok(()) => {
                self.compiling_shader_pairs.insert(key.clone());
            }
            Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => {
                self.failed_shader_pairs.insert(key.clone());
                eprintln!(
                    "[REMu][METAL] Gen5 shader pair es={:#x} ps={:#x} failed: compiler worker stopped",
                    key.export_shader_address, key.pixel_shader_address
                );
            }
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
                        "[REMu][METAL] compiled Gen5 shader pair es={:#x} ps={:#x} MSL={vertex_bytes}/{pixel_bytes} bytes reuse={vertex_reuse}/{pixel_reuse} diagnostic={vertex_pattern}/{pixel_color}",
                        result.key.export_shader_address,
                        result.key.pixel_shader_address,
                        vertex_bytes = compiled.vertex.source.len(),
                        pixel_bytes = compiled.pixel.source.len(),
                        vertex_reuse = compiled.reused_vertex_function,
                        pixel_reuse = compiled.reused_pixel_function,
                        vertex_pattern = compiled.vertex.source.contains("remu_diagnostic_vertex"),
                        pixel_color = compiled
                            .pixel
                            .source
                            .contains("float4(1.0f, 0.0f, 1.0f, 1.0f)"),
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
        function_cache: &ShaderFunctionCache,
        render_pipeline_archive: Option<&PersistentRenderPipelineArchive>,
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
        let mut pixel_options = remu_shader_metal::MetalCompileOptions::pixel(
            pixel_outputs,
            register_value(&draw.context_registers, 0x1b3).unwrap_or_default(),
            register_value(&draw.context_registers, 0x1b4).unwrap_or_default(),
        );
        pixel_options.pixel_input_controls = key
            .pixel_input_controls
            .map(remu_shader_metal::PixelInputControl::from_register);
        let selected_diagnostic_shader = diagnostic_dump_target_pixel_shader()
            == Some(draw.pixel_shader_address)
            || diagnostic_dump_target_pixel_instruction_count()
                == Some(pixel.program.instructions.len());
        pixel_options.diagnostic_force_pixel_color = key.sample_count > 1
            && pixel.program.instructions.len() > 64
            && diagnostic_force_msaa_pixel_color();
        pixel_options.diagnostic_capture_pixel_output = diagnostic_capture_pixel_output()
            && ((key.sample_count > 1 && pixel.program.instructions.len() == 370)
                || selected_diagnostic_shader
                || (diagnostic_dump_video_draw()
                    && evaluation_samples_large_linear_video(pixel_evaluation)));
        let pixel_msl =
            remu_shader_metal::compile_msl(&pixel.program, pixel_evaluation, &pixel_options)
                .map_err(|error| format!("pixel lowering: {error}"))?;
        let mut vertex_options =
            remu_shader_metal::MetalCompileOptions::vertex(pixel_msl.interpolated_attribute_count);
        vertex_options.render_target_array_length = key.render_target_array_length;
        vertex_options.diagnostic_force_visible_clip_depth =
            key.sample_count > 1 && diagnostic_force_visible_clip_depth();
        vertex_options.diagnostic_force_vertex_pattern =
            key.sample_count > 1 && diagnostic_force_msaa_vertex_pattern();
        vertex_options.diagnostic_force_vertex_fetch_id = diagnostic_force_vertex_fetch_id()
            && (key.sample_count > 1 || selected_diagnostic_shader);
        vertex_options.diagnostic_capture_vertex_position = diagnostic_capture_vertex_position()
            && (key.sample_count > 1
                || selected_diagnostic_shader
                || (diagnostic_dump_video_draw()
                    && evaluation_samples_large_linear_video(pixel_evaluation)));
        let vertex_msl = if key.render_target_array_length > 1 {
            remu_shader_metal::compile_layered_rectangle_vertex(
                pixel_msl.interpolated_attribute_count,
                key.render_target_array_length,
            )
        } else {
            remu_shader_metal::compile_msl(&export.program, export_evaluation, &vertex_options)
                .map_err(|error| format!("vertex lowering: {error}"))?
        };
        if let Some(directory) = std::env::var_os("REMU_DUMP_MSL_DIR")
            && dump_msl_pixel_filter_matches(draw.pixel_shader_address)
            && dump_msl_instruction_filter_matches(pixel.program.instructions.len())
        {
            let directory = Path::new(&directory);
            if let Err(error) = dump_shader_pair_ir(
                directory,
                draw.export_shader_address,
                draw.pixel_shader_address,
                &export.program.instructions,
                &pixel.program.instructions,
            ) {
                eprintln!(
                    "[REMu][METAL] could not dump shader IR for es={:#x} ps={:#x}: {error}",
                    draw.export_shader_address, draw.pixel_shader_address
                );
            }
            if let Err(error) = dump_shader_pair_msl(
                directory,
                draw.export_shader_address,
                draw.pixel_shader_address,
                &vertex_msl.source,
                &pixel_msl.source,
            ) {
                eprintln!(
                    "[REMu][METAL] could not dump MSL for es={:#x} ps={:#x}: {error}",
                    draw.export_shader_address, draw.pixel_shader_address
                );
            }
        }
        if let Some(directory) = std::env::var_os("REMU_DUMP_VIDEO_MSL_DIR")
            && evaluation_samples_large_linear_video(pixel_evaluation)
        {
            let directory = Path::new(&directory);
            if let Err(error) = dump_shader_pair_msl(
                directory,
                draw.export_shader_address,
                draw.pixel_shader_address,
                &vertex_msl.source,
                &pixel_msl.source,
            ) {
                eprintln!(
                    "[REMu][METAL] could not dump video MSL for es={:#x} ps={:#x}: {error}",
                    draw.export_shader_address, draw.pixel_shader_address
                );
            }
        }
        // Render-target formats, blending, depth state, and sample count produce distinct Metal
        // pipelines but do not alter the generated stage functions. Unity frequently emits many
        // such variants during scene loading, so compiling the same 70-100 KiB MSL function for
        // each variant stalls the guest for tens of seconds. Retain functions independently from
        // pipeline states and compile each exact MSL source only once on this worker.
        let (vertex_function, reused_vertex_function) =
            function_cache.vertex(device, &vertex_msl)?;
        let (pixel_function, reused_pixel_function) = function_cache.pixel(device, &pixel_msl)?;
        let make_descriptor = || {
            let descriptor = RenderPipelineDescriptor::new();
            descriptor.set_label(&format!(
                "REMu Gen5 ES {:#x} PS {:#x}",
                draw.export_shader_address, draw.pixel_shader_address
            ));
            descriptor.set_vertex_function(Some(&vertex_function));
            descriptor.set_fragment_function(Some(&pixel_function));
            descriptor.set_raster_sample_count(u64::from(key.sample_count));
            descriptor.set_input_primitive_topology(key.primitive_topology_class);
            for (location, target) in key.targets.iter().enumerate() {
                let attachment = descriptor
                    .color_attachments()
                    .object_at(u64::try_from(location).expect("color location fits u64"))
                    .ok_or_else(|| format!("Metal has no color attachment {location}"))?;
                attachment.set_pixel_format(target.pixel_format);
                attachment.set_write_mask(metal_color_write_mask(target.write_mask));
                let blending_enabled = target.blend.enabled
                    && !pixel_options.diagnostic_force_pixel_color
                    && !diagnostic_disable_blending();
                attachment.set_blending_enabled(blending_enabled);
                if blending_enabled {
                    attachment.set_source_rgb_blend_factor(target.blend.source_rgb);
                    attachment.set_destination_rgb_blend_factor(target.blend.destination_rgb);
                    attachment.set_rgb_blend_operation(target.blend.rgb_operation);
                    attachment.set_source_alpha_blend_factor(target.blend.source_alpha);
                    attachment.set_destination_alpha_blend_factor(target.blend.destination_alpha);
                    attachment.set_alpha_blend_operation(target.blend.alpha_operation);
                }
            }
            if let Some(depth) = key.depth {
                descriptor.set_depth_attachment_pixel_format(depth.pixel_format);
            }
            Ok(descriptor)
        };
        let pipeline = if let Some(archive) = render_pipeline_archive {
            archive.new_render_pipeline_state(device, make_descriptor)?
        } else {
            let descriptor = make_descriptor()?;
            device
                .new_render_pipeline_state(&descriptor)
                .map_err(|error| format!("Metal render pipeline: {error}"))?
        };
        Ok(CompiledShaderPair {
            pipeline,
            vertex_function,
            pixel_function,
            vertex: vertex_msl,
            pixel: pixel_msl,
            reused_vertex_function,
            reused_pixel_function,
        })
    }

    fn ensure_resident_texture(
        &mut self,
        device: &Device,
        descriptor: GuestTextureDescriptor,
    ) -> Result<Texture, String> {
        if let Some(resident) =
            self.resident_guest_textures
                .get_mut(&descriptor)
                .filter(|resident| {
                    resident.texture_type == 9
                        && resident.layer_count == 1
                        && resident.mip_level_count >= 1
                })
        {
            match refresh_resident_texture_write_watch(resident)? {
                WriteWatchRefresh::Active
                | WriteWatchRefresh::Reprotecting
                | WriteWatchRefresh::Renewed { .. } => return Ok(resident.texture.clone()),
                WriteWatchRefresh::Replaced => {}
            }
        }
        self.resident_guest_textures.remove(&descriptor);
        let shared_resident = self
            .resident_guest_textures
            .values()
            .find(|resident| {
                resident.texture_type == 9
                    && resident.layer_count == 1
                    && resident.mip_level_count >= 1
                    && guest_texture_descriptors_share_storage(resident.descriptor, descriptor)
            })
            .map(|resident| resident.descriptor);
        if let Some(shared_resident) = shared_resident {
            let resident = self
                .resident_guest_textures
                .get_mut(&shared_resident)
                .expect("selected shared resident texture exists");
            match refresh_resident_texture_write_watch(resident)? {
                WriteWatchRefresh::Active
                | WriteWatchRefresh::Reprotecting
                | WriteWatchRefresh::Renewed { .. } => return Ok(resident.texture.clone()),
                WriteWatchRefresh::Replaced => {
                    self.resident_guest_textures.remove(&shared_resident);
                }
            }
        }
        let mut aliases = self
            .resident_guest_textures
            .values()
            .filter(|resident| guest_texture_descriptors_overlap(resident.descriptor, descriptor))
            .map(|resident| (resident.content_version, resident.descriptor))
            .collect::<Vec<_>>();
        if !aliases.is_empty() {
            self.wait_for_guest_command_batches()?;
            self.multisample_textures
                .retain(|key, _| !guest_texture_descriptors_overlap(key.descriptor, descriptor));
            let queue = self
                .command_queue
                .as_ref()
                .ok_or("Metal command queue is unavailable")?
                .clone();
            aliases.sort_by_key(|(version, _)| *version);
            for (_, alias) in aliases {
                let refresh = refresh_resident_texture_write_watch(
                    self.resident_guest_textures
                        .get_mut(&alias)
                        .expect("collected resident alias exists"),
                )?;
                if refresh == WriteWatchRefresh::Reprotecting {
                    return Err(format!(
                        "resident alias mapping {:#x} is being reprotected",
                        alias.address
                    ));
                }
                let resident = self
                    .resident_guest_textures
                    .remove(&alias)
                    .expect("collected resident alias exists");
                if refresh != WriteWatchRefresh::Replaced && resident.gpu_written {
                    write_resident_texture_contents_to_guest(device, &queue, &resident)?;
                }
            }
            retire_overlapping_sampled_textures(&mut self.sampled_texture_cache, descriptor);
        }
        // GFX10 image descriptors may expose only the mip selected by one shader even though
        // later passes reinterpret the same mode-27 allocation as a larger hierarchy. Metal
        // cannot grow a texture after a compute/render pass has written it, so create the complete
        // hierarchy at first residency whenever the guest allocation covers it. This keeps the
        // first GPU-written level live and lets later single-level and mip-tail views share the
        // same backing instead of dropping the later compute pass.
        if descriptor.tile_mode == 27 {
            let mip_level_count = complete_2d_mip_level_count(descriptor);
            if mip_level_count > 1
                && let Ok(layout) = mode27_mip_chain_layout(descriptor, mip_level_count)
                && let Ok(byte_count) = u64::try_from(layout.byte_count)
                && guest_shared_backing_alias_segments(descriptor.address, byte_count).is_some()
            {
                return ensure_mipmapped_resident_texture_in_map(
                    device,
                    DecodedImageDescriptor {
                        guest: descriptor,
                        destination_select: 0xfac,
                        base_mip_level: 0,
                        mip_level_count,
                        texture_type: 9,
                        layer_count: 1,
                    },
                    &mut self.resident_guest_textures,
                );
            }
        }
        ensure_resident_texture_in_map(device, descriptor, &mut self.resident_guest_textures)
    }

    fn ensure_mipmapped_render_target_texture(
        &mut self,
        device: &Device,
        target: DecodedRenderTarget,
    ) -> Result<Texture, String> {
        if target.texture_type != 9
            || target.layer_count != 1
            || target.mip_level_count <= 1
            || target.sample_count != 1
        {
            return Err(format!(
                "mipmapped render target requires a single-sample 2D resource, got type={} layers={} levels={} samples={}",
                target.texture_type,
                target.layer_count,
                target.mip_level_count,
                target.sample_count
            ));
        }
        if let Some(resident) = self
            .resident_guest_textures
            .get_mut(&target.texture)
            .filter(|resident| {
                resident.texture_type == 9
                    && resident.layer_count == 1
                    && resident.mip_level_count >= target.mip_level_count
            })
        {
            match refresh_resident_texture_write_watch(resident)? {
                WriteWatchRefresh::Active
                | WriteWatchRefresh::Reprotecting
                | WriteWatchRefresh::Renewed { .. } => return Ok(resident.texture.clone()),
                WriteWatchRefresh::Replaced => {}
            }
        }
        if self
            .resident_guest_textures
            .get(&target.texture)
            .is_some_and(|resident| resident.gpu_written)
        {
            return Err(format!(
                "cannot reinterpret GPU-written target {:#x} as a {}-level hierarchy",
                target.texture.address, target.mip_level_count
            ));
        }
        self.resident_guest_textures.remove(&target.texture);
        retire_overlapping_sampled_textures(&mut self.sampled_texture_cache, target.texture);
        let layout = mode27_mip_chain_layout(target.texture, target.mip_level_count)?;
        let write_watch = watch_resident_texture(
            target.texture,
            target.texture_type,
            target.layer_count,
            target.mip_level_count,
        )?;
        let raw =
            snapshot_watched_guest_range(target.texture.address, layout.byte_count, &write_watch)?;
        let (_, linear_levels) =
            decode_mode27_mip_chain_from_raw(target.texture, target.mip_level_count, &raw)?;
        let texture = create_mipmapped_guest_texture(
            device,
            target.texture,
            target.mip_level_count,
            &linear_levels,
        )?;
        texture.set_label(&format!(
            "REMu {}-level guest image {:#x}",
            target.mip_level_count, target.texture.address
        ));
        self.resident_guest_textures.insert(
            target.texture,
            ResidentGuestTexture {
                texture: texture.clone(),
                component_reorder_view: None,
                descriptor: target.texture,
                texture_type: target.texture_type,
                layer_count: target.layer_count,
                mip_level_count: target.mip_level_count,
                write_watch,
                gpu_written: false,
                content_version: 0,
            },
        );
        Ok(texture)
    }

    fn ensure_layered_render_target_texture(
        &mut self,
        device: &Device,
        target: DecodedRenderTarget,
    ) -> Result<Texture, String> {
        if target.texture_type == 9 && target.layer_count == 1 {
            return if target.mip_level_count == 1 {
                self.ensure_resident_texture(device, target.texture)
            } else {
                self.ensure_mipmapped_render_target_texture(device, target)
            };
        }
        if target.texture_type != 10 || target.layer_count <= 1 {
            return Err(format!(
                "unsupported layered target texture type={} layers={}",
                target.texture_type, target.layer_count
            ));
        }
        if target.sample_count != 1 {
            return Err("multisampled 3D color targets are not implemented".to_owned());
        }
        if target.mip_level_count != 1 || target.mip_level != 0 {
            return Err(format!(
                "mipmapped 3D color targets are not implemented (mip={}/{})",
                target.mip_level,
                target.mip_level_count - 1
            ));
        }
        if let Some(resident) = self
            .resident_guest_textures
            .get_mut(&target.texture)
            .filter(|resident| {
                resident.texture_type == target.texture_type
                    && resident.layer_count == target.layer_count
                    && resident.mip_level_count == 1
            })
        {
            match refresh_resident_texture_write_watch(resident)? {
                WriteWatchRefresh::Active
                | WriteWatchRefresh::Reprotecting
                | WriteWatchRefresh::Renewed { .. } => return Ok(resident.texture.clone()),
                WriteWatchRefresh::Replaced => {}
            }
        }
        if self
            .resident_guest_textures
            .get(&target.texture)
            .is_some_and(|resident| resident.gpu_written)
        {
            return Err(format!(
                "cannot replace GPU-written target {:#x} with a {}-layer 3D view",
                target.texture.address, target.layer_count
            ));
        }
        self.resident_guest_textures.remove(&target.texture);
        retire_overlapping_sampled_textures(&mut self.sampled_texture_cache, target.texture);
        let descriptor = DecodedImageDescriptor {
            guest: target.texture,
            destination_select: 0,
            base_mip_level: 0,
            mip_level_count: 1,
            texture_type: target.texture_type,
            layer_count: target.layer_count,
        };
        let write_watch = watch_sampled_texture(descriptor)?;
        let raw = snapshot_watched_sampled_texture_raw(descriptor, &write_watch)?;
        let linear = decode_sampled_texture_linear_from_raw(descriptor, &raw)?;
        let texture = create_layered_render_target_texture(device, descriptor, &linear)?;
        texture.set_label(&format!(
            "REMu {}-layer guest volume {:#x}",
            target.layer_count, target.texture.address
        ));
        self.resident_guest_textures.insert(
            target.texture,
            ResidentGuestTexture {
                texture: texture.clone(),
                component_reorder_view: None,
                descriptor: target.texture,
                texture_type: target.texture_type,
                layer_count: target.layer_count,
                mip_level_count: 1,
                write_watch,
                gpu_written: false,
                content_version: 0,
            },
        );
        Ok(texture)
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
                initialized: false,
            },
        );
        texture
    }

    fn ensure_scanout_snapshot_texture(
        &mut self,
        device: &Device,
        descriptor: GuestTextureDescriptor,
    ) -> Texture {
        if let Some(snapshot) = self.scanout_snapshot_textures.get(&descriptor.address)
            && snapshot.descriptor == descriptor
        {
            return snapshot.texture.clone();
        }
        let metal_descriptor = TextureDescriptor::new();
        metal_descriptor.set_texture_type(MTLTextureType::D2);
        metal_descriptor.set_pixel_format(descriptor.pixel_format);
        metal_descriptor.set_width(u64::from(descriptor.width));
        metal_descriptor.set_height(u64::from(descriptor.height));
        metal_descriptor.set_storage_mode(MTLStorageMode::Private);
        metal_descriptor.set_usage(MTLTextureUsage::ShaderRead);
        let texture = device.new_texture(&metal_descriptor);
        texture.set_label(&format!("REMu scanout snapshot {:#x}", descriptor.address));
        self.scanout_snapshot_textures.insert(
            descriptor.address,
            ScanoutSnapshotTexture {
                descriptor,
                texture: texture.clone(),
            },
        );
        texture
    }

    fn ensure_multisample_texture(
        &mut self,
        device: &Device,
        descriptor: GuestTextureDescriptor,
        sample_count: u32,
    ) -> Result<(Texture, bool), String> {
        if sample_count <= 1 {
            return Err("multisample texture requires at least two samples".to_owned());
        }
        if !device.supports_texture_sample_count(u64::from(sample_count)) {
            return Err(format!(
                "Metal device does not support {sample_count} samples for {:?}",
                descriptor.pixel_format
            ));
        }
        let key = MultisampleTextureKey {
            descriptor,
            sample_count,
        };
        if let Some(resident) = self.multisample_textures.get(&key) {
            return Ok((resident.texture.clone(), !resident.initialized));
        }
        let metal_descriptor = TextureDescriptor::new();
        metal_descriptor.set_texture_type(MTLTextureType::D2Multisample);
        metal_descriptor.set_pixel_format(descriptor.pixel_format);
        metal_descriptor.set_width(u64::from(descriptor.width));
        metal_descriptor.set_height(u64::from(descriptor.height));
        metal_descriptor.set_sample_count(u64::from(sample_count));
        metal_descriptor.set_storage_mode(MTLStorageMode::Private);
        metal_descriptor.set_usage(MTLTextureUsage::RenderTarget);
        let texture = device.new_texture(&metal_descriptor);
        texture.set_label(&format!(
            "REMu {}x multisample guest image {:#x}",
            sample_count, descriptor.address
        ));
        self.multisample_textures.insert(
            key,
            MultisampleResidentTexture {
                texture: texture.clone(),
                initialized: false,
            },
        );
        Ok((texture, true))
    }

    fn mark_multisample_texture_initialized(
        &mut self,
        descriptor: GuestTextureDescriptor,
        sample_count: u32,
    ) {
        if let Some(resident) = self.multisample_textures.get_mut(&MultisampleTextureKey {
            descriptor,
            sample_count,
        }) {
            resident.initialized = true;
        }
    }

    fn cached_sampler_argument_buffer(
        &mut self,
        device: &Device,
        function: &Function,
        buffer_index: Option<u32>,
        evaluation: &ShaderEvaluation,
    ) -> Result<(Option<Buffer>, Vec<SamplerState>), String> {
        let sampled_descriptors = canonical_sampler_bindings(evaluation)
            .into_iter()
            .map(|binding| binding.sampler_descriptor.clone())
            .collect::<Vec<_>>();
        let key = buffer_index.map(|buffer_index| SamplerArgumentCacheKey {
            function: function.as_ptr() as usize,
            buffer_index,
            descriptors: sampled_descriptors
                .into_iter()
                .map(|descriptor| descriptor.to_vec())
                .collect(),
        });
        if let Some(cached) = key
            .as_ref()
            .and_then(|key| self.sampler_argument_cache.get(key))
        {
            return Ok((Some(cached.buffer.clone()), cached.samplers.clone()));
        }
        let created = create_sampler_argument_buffer(device, function, buffer_index, evaluation)?;
        if self.sampler_argument_cache.len() < SAMPLER_ARGUMENT_CACHE_CAPACITY
            && let (Some(key), Some(buffer)) = (key, created.0.as_ref())
        {
            self.sampler_argument_cache.insert(
                key,
                CachedSamplerArguments {
                    buffer: buffer.clone(),
                    samplers: created.1.clone(),
                },
            );
        }
        Ok(created)
    }

    fn cached_depth_stencil_state(
        &mut self,
        device: &Device,
        depth: DecodedDepthTarget,
    ) -> DepthStencilState {
        let key = (depth.compare as u64, depth.write_enabled);
        if let Some(state) = self.depth_stencil_cache.get(&key) {
            return state.clone();
        }
        let descriptor = DepthStencilDescriptor::new();
        descriptor.set_depth_compare_function(depth.compare);
        descriptor.set_depth_write_enabled(depth.write_enabled);
        let state = device.new_depth_stencil_state(&descriptor);
        self.depth_stencil_cache.insert(key, state.clone());
        state
    }

    fn upload_global_buffers(
        &mut self,
        device: &Device,
        evaluation: &ShaderEvaluation,
    ) -> Result<Vec<UploadedGlobalBuffer>, String> {
        evaluation
            .global_memory_bindings
            .iter()
            .map(|binding| self.upload_global_buffer(device, binding))
            .collect()
    }

    fn upload_global_buffer(
        &mut self,
        device: &Device,
        binding: &GlobalMemoryBinding,
    ) -> Result<UploadedGlobalBuffer, String> {
        if let Some((page_address, page_byte_count, offset)) = guest_global_buffer_view(binding)? {
            let key = (page_address, page_byte_count);
            let buffer = if let Some(buffer) = self.guest_global_buffer_cache.get(&key) {
                buffer.clone()
            } else {
                if self.guest_global_buffer_cache.len() >= GUEST_GLOBAL_BUFFER_CACHE_CAPACITY {
                    self.guest_global_buffer_cache.clear();
                }
                let length = u64::try_from(page_byte_count)
                    .map_err(|_| "page-aligned guest global size does not fit Metal")?;
                let buffer = device.new_buffer_with_bytes_no_copy(
                    usize::try_from(page_address)
                        .map_err(|_| "guest global address does not fit the host pointer")?
                        as *const c_void,
                    length,
                    MTLResourceOptions::StorageModeShared,
                    None,
                );
                buffer.set_label(&format!(
                    "REMu zero-copy guest global {page_address:#x}/{page_byte_count}"
                ));
                self.guest_global_buffer_cache.insert(key, buffer.clone());
                buffer
            };
            return Ok(UploadedGlobalBuffer {
                buffer,
                offset,
                guest_memory: true,
            });
        }
        if binding.data.len() != binding.byte_count {
            return Err(format!(
                "global binding {:#x}/{} is neither identity-mapped nor captured",
                binding.base_address, binding.byte_count
            ));
        }
        let key = GlobalBufferCacheKey {
            base_address: binding.base_address,
            byte_count: binding.data.len(),
        };
        if binding.writable || binding.data.len() > GLOBAL_BUFFER_CACHE_BYTE_CAPACITY {
            if let Some(cached) = self.transient_global_buffer_cache.get(&key)
                && cached.bytes == binding.data
            {
                return Ok(UploadedGlobalBuffer {
                    buffer: cached.buffer.clone(),
                    offset: 0,
                    guest_memory: false,
                });
            }
            let buffer = metal_buffer_with_bytes(device, &binding.data)?;
            buffer.set_label(&format!(
                "REMu transient global {:#x}/{}",
                binding.base_address,
                binding.data.len()
            ));
            self.transient_global_buffer_cache.insert(
                key,
                CachedGlobalBuffer {
                    bytes: binding.data.clone(),
                    buffer: buffer.clone(),
                },
            );
            return Ok(UploadedGlobalBuffer {
                buffer,
                offset: 0,
                guest_memory: false,
            });
        }
        if let Some(cached) = self.global_buffer_cache.get(&key)
            && cached.bytes == binding.data
        {
            return Ok(UploadedGlobalBuffer {
                buffer: cached.buffer.clone(),
                offset: 0,
                guest_memory: false,
            });
        }
        let buffer = metal_buffer_with_bytes(device, &binding.data)?;
        if self.global_buffer_cache.len() >= GLOBAL_BUFFER_CACHE_CAPACITY
            || self
                .global_buffer_cache_bytes
                .saturating_add(binding.data.len())
                > GLOBAL_BUFFER_CACHE_BYTE_CAPACITY
        {
            self.global_buffer_cache.clear();
            self.global_buffer_cache_bytes = 0;
        }
        if let Some(replaced) = self.global_buffer_cache.insert(
            key,
            CachedGlobalBuffer {
                bytes: binding.data.clone(),
                buffer: buffer.clone(),
            },
        ) {
            self.global_buffer_cache_bytes = self
                .global_buffer_cache_bytes
                .saturating_sub(replaced.bytes.len());
        }
        self.global_buffer_cache_bytes = self
            .global_buffer_cache_bytes
            .saturating_add(binding.data.len());
        Ok(UploadedGlobalBuffer {
            buffer,
            offset: 0,
            guest_memory: false,
        })
    }

    fn cached_index_buffer(&mut self, device: &Device, indices: &[u32]) -> Result<Buffer, String> {
        if let Some(cached) = self.index_buffer_cache.get(indices) {
            return Ok(cached.clone());
        }
        let byte_count = indices
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or("index byte count overflow")?;
        let bytes = if byte_count == 0 {
            &[][..]
        } else {
            // SAFETY: u32 has no invalid bit patterns and this view has the exact byte extent of
            // the live index slice. Metal copies it before this method returns.
            unsafe { std::slice::from_raw_parts(indices.as_ptr().cast::<u8>(), byte_count) }
        };
        let buffer = metal_buffer_with_bytes(device, bytes)?;
        if self.index_buffer_cache.len() >= INDEX_BUFFER_CACHE_CAPACITY {
            self.index_buffer_cache.clear();
        }
        self.index_buffer_cache
            .insert(indices.to_vec(), buffer.clone());
        Ok(buffer)
    }

    fn direct_scanout_needs_initial_clear(&self, address: u64) -> bool {
        self.direct_scanout_textures
            .get(&address)
            .is_none_or(|scanout| !scanout.initialized)
    }

    fn is_diagnostic_scanout_target(&self, target: &DecodedRenderTarget) -> bool {
        // A render target decoded from ordinary CB registers does not carry `direct_scanout`
        // even when its address is an already registered VideoOut buffer. Diagnostics need the
        // address identity as well so they can follow writers that arrive without the synthetic
        // scanout target metadata. This predicate is intentionally not used for texture routing
        // or initialization, and therefore cannot alter guest rendering.
        target.direct_scanout
            || self
                .direct_scanout_textures
                .contains_key(&target.texture.address)
    }

    fn diagnostic_capture_texture(&self, target: &DecodedRenderTarget) -> Option<Texture> {
        self.direct_scanout_textures
            .get(&target.texture.address)
            .filter(|scanout| scanout.descriptor == target.texture)
            .map(|scanout| scanout.texture.clone())
            .or_else(|| {
                self.resident_guest_textures
                    .get(&target.texture)
                    .map(|resident| resident.texture.clone())
            })
    }

    fn mark_direct_scanout_initialized(&mut self, address: u64) {
        if let Some(scanout) = self.direct_scanout_textures.get_mut(&address) {
            scanout.initialized = true;
        }
    }

    fn refresh_pending_flip_after_target_write(&mut self, target: DecodedRenderTarget) {
        let (Some(display), Some(flip)) = (self.display, self.pending_flip) else {
            return;
        };
        if flip.address != target.texture.address
            || display.width != target.texture.width
            || display.height != target.texture.height
        {
            return;
        }

        // A guest may announce a flip before its final compositor pass, and reinterpreting an
        // overlapping allocation can replace the Metal texture object after that announcement.
        // Refresh the presented object after every matching target write instead of treating an
        // already-uploaded flip sequence as immutable. Otherwise the window keeps sampling the
        // stale solid texture while later passes render the real menu into its replacement.
        let refreshed = if target.direct_scanout {
            self.direct_scanout_textures
                .get(&target.texture.address)
                .filter(|scanout| scanout.initialized && scanout.descriptor == target.texture)
                .map(|scanout| scanout.texture.clone())
        } else {
            self.resident_guest_textures
                .values()
                .filter(|resident| {
                    resident.gpu_written
                        && resident.descriptor.address == target.texture.address
                        && resident.descriptor.width == target.texture.width
                        && resident.descriptor.height == target.texture.height
                })
                .max_by_key(|resident| resident.content_version)
                .map(|resident| resident.texture.clone())
        };
        if let Some(texture) = refreshed {
            self.frame_texture = Some(texture);
            self.frame_texture_size = (target.texture.width, target.texture.height);
            self.uploaded_flip_sequence = flip.sequence;
        }
        self.redraw_pending = true;
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
        let queue = self
            .command_queue
            .as_ref()
            .ok_or("Metal command queue is unavailable")?
            .clone();

        let mut invalidated_multisample_descriptors = Vec::new();
        let mut dirty_reprotected_residents = HashSet::new();
        let resident_descriptors = self
            .resident_guest_textures
            .keys()
            .copied()
            .collect::<Vec<_>>();
        for descriptor in resident_descriptors {
            let refresh = refresh_resident_texture_write_watch(
                self.resident_guest_textures
                    .get_mut(&descriptor)
                    .expect("collected resident texture exists"),
            )?;
            match refresh {
                WriteWatchRefresh::Renewed { was_dirty: true } => {
                    dirty_reprotected_residents.insert(descriptor);
                }
                WriteWatchRefresh::Replaced => {
                    self.resident_guest_textures.remove(&descriptor);
                }
                WriteWatchRefresh::Active
                | WriteWatchRefresh::Reprotecting
                | WriteWatchRefresh::Renewed { was_dirty: false } => {}
            }
        }
        let mut dirty_reprotected_sampled = HashSet::new();
        let sampled_keys = self
            .sampled_texture_cache
            .keys()
            .copied()
            .collect::<Vec<_>>();
        for key in sampled_keys {
            let refresh = refresh_sampled_texture_write_watch(
                key,
                self.sampled_texture_cache
                    .get_mut(&key)
                    .expect("collected sampled texture exists"),
                &mut self.sampled_texture_write_watches,
            )?;
            match refresh {
                WriteWatchRefresh::Renewed { was_dirty: true } => {
                    dirty_reprotected_sampled.insert(key);
                }
                WriteWatchRefresh::Replaced => {
                    if let Some(retired) = self.sampled_texture_cache.remove(&key) {
                        retire_sampled_texture(retired);
                    }
                }
                WriteWatchRefresh::Active
                | WriteWatchRefresh::Reprotecting
                | WriteWatchRefresh::Renewed { was_dirty: false } => {}
            }
        }
        for resident in self.resident_guest_textures.values_mut() {
            let (_, byte_count) = resident.write_watch.range();
            if !acquire.overlaps(resident.descriptor.address, byte_count)
                || (!resident.write_watch.is_dirty()
                    && !dirty_reprotected_residents.contains(&resident.descriptor))
            {
                continue;
            }
            let texture = if resident.mip_level_count == 1 {
                let raw =
                    snapshot_watched_guest_texture_raw(resident.descriptor, &resident.write_watch)?;
                let linear = decode_guest_texture_linear_from_raw(resident.descriptor, &raw)?;
                create_guest_texture(
                    &device,
                    resident.descriptor,
                    MTLTextureUsage::RenderTarget
                        | MTLTextureUsage::ShaderRead
                        | MTLTextureUsage::ShaderWrite,
                    &linear,
                )?
            } else {
                let layout =
                    mode27_mip_chain_layout(resident.descriptor, resident.mip_level_count)?;
                let raw = snapshot_watched_guest_range(
                    resident.descriptor.address,
                    layout.byte_count,
                    &resident.write_watch,
                )?;
                let (_, linear_levels) = decode_mode27_mip_chain_from_raw(
                    resident.descriptor,
                    resident.mip_level_count,
                    &raw,
                )?;
                create_mipmapped_guest_texture(
                    &device,
                    resident.descriptor,
                    resident.mip_level_count,
                    &linear_levels,
                )?
            };
            texture.set_label(&format!(
                "REMu guest image {:#x}",
                resident.descriptor.address
            ));
            // Replace instead of mutating the old texture. Already encoded Metal work retains the
            // old allocation, while following draws observe the CPU update without a GPU wait.
            resident.texture = texture;
            resident.component_reorder_view = None;
            resident.gpu_written = false;
            invalidated_multisample_descriptors.push(resident.descriptor);
        }
        self.multisample_textures.retain(|key, _| {
            !invalidated_multisample_descriptors
                .iter()
                .any(|descriptor| guest_texture_descriptors_overlap(key.descriptor, *descriptor))
        });

        for (key, cached) in &mut self.sampled_texture_cache {
            let (_, byte_count) = cached.write_watch.range();
            if !acquire.overlaps(key.descriptor.address, byte_count)
                || (!cached.write_watch.is_dirty() && !dirty_reprotected_sampled.contains(key))
            {
                continue;
            }
            let descriptor = DecodedImageDescriptor {
                guest: key.descriptor,
                destination_select: key.destination_select,
                base_mip_level: key.base_mip_level,
                mip_level_count: key.mip_level_count,
                texture_type: key.texture_type,
                layer_count: key.layer_count,
            };
            let raw = snapshot_watched_sampled_texture_raw(descriptor, &cached.write_watch)?;
            let texture = if sampled_texture_uses_exact_mode27_mips(descriptor) {
                create_exact_mode27_sampled_mip_view(&device, descriptor, &raw, key.comparison)?
            } else {
                let linear = decode_sampled_texture_linear_from_raw(descriptor, &raw)?;
                create_sampled_guest_texture(
                    &device,
                    &queue,
                    key.descriptor,
                    &linear,
                    key.base_mip_level,
                    key.mip_level_count,
                    key.texture_type,
                    key.layer_count,
                    key.comparison,
                )?
            };
            texture.set_label(&format!(
                "REMu sampled guest image {:#x}",
                key.descriptor.address
            ));
            cached.texture = texture;
        }
        Ok(())
    }

    fn mark_resident_texture_written(&mut self, descriptor: GuestTextureDescriptor) {
        self.mark_resident_texture_written_preserving_multisample(descriptor);
        self.multisample_textures
            .retain(|key, _| !guest_texture_descriptors_overlap(key.descriptor, descriptor));
    }

    fn mark_resident_texture_written_preserving_multisample(
        &mut self,
        descriptor: GuestTextureDescriptor,
    ) {
        self.resident_content_version = self.resident_content_version.saturating_add(1);
        let resident_descriptor = self
            .resident_guest_textures
            .contains_key(&descriptor)
            .then_some(descriptor)
            .or_else(|| {
                self.resident_guest_textures
                    .values()
                    .find(|resident| {
                        guest_texture_descriptors_share_storage(resident.descriptor, descriptor)
                    })
                    .map(|resident| resident.descriptor)
            });
        if let Some(resident) = resident_descriptor
            .and_then(|descriptor| self.resident_guest_textures.get_mut(&descriptor))
        {
            resident.gpu_written = true;
            resident.content_version = self.resident_content_version;
        }
        retire_overlapping_sampled_textures(&mut self.sampled_texture_cache, descriptor);
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
                .filter(|binding| !image_resource_descriptor_is_null(&binding.resource_descriptor))
            {
                let sampled = decode_image_descriptor(&binding.resource_descriptor)
                    .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
                let comparison = image_is_comparison(binding.opcode);
                // A Gen5 MSAA allocation can expose its shader-readable resolved image through
                // an overlapping SRD whose base follows the allocation's mask metadata. Keep the
                // Metal resolve GPU-current so `upload_image_textures` can bind it directly. A
                // CPU writeback here both discards that provenance and decodes from the shifted
                // SRD base, producing an empty post-process input.
                aliases.extend(
                    self.resident_guest_textures
                        .values()
                        .filter(|resident| resident.gpu_written)
                        .filter(|resident| {
                            guest_texture_descriptors_overlap(resident.descriptor, sampled.guest)
                                && resident.descriptor != sampled.guest
                                && !guest_texture_descriptors_share_storage(
                                    resident.descriptor,
                                    sampled.guest,
                                )
                                && !resident_component_reordered_sample_is_eligible(
                                    resident, sampled, comparison,
                                )
                                && !self.multisample_textures.iter().any(|(key, multisample)| {
                                    multisample.initialized
                                        && key.descriptor == resident.descriptor
                                        && render_resolve_candidate_matches(
                                            sampled.guest,
                                            key.descriptor,
                                            key.sample_count,
                                        )
                                })
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
            let refresh = refresh_resident_texture_write_watch(
                self.resident_guest_textures
                    .get_mut(&descriptor)
                    .expect("collected sampled alias exists"),
            )?;
            if refresh == WriteWatchRefresh::Replaced {
                self.resident_guest_textures.remove(&descriptor);
                continue;
            }
            if refresh == WriteWatchRefresh::Reprotecting {
                return Err(format!(
                    "sampled alias mapping {:#x} is being reprotected",
                    descriptor.address
                ));
            }
            let resident = self
                .resident_guest_textures
                .get_mut(&descriptor)
                .expect("collected sampled alias exists");
            write_resident_texture_contents_to_guest(&device, &queue, resident)?;
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
                    let result = self.finish_terminal_guest_batch(&batch, None);
                    self.statistics.gpu_execution_time +=
                        completed_command_buffer_gpu_duration(&batch.command_buffer);
                    publish_guest_batch_completion(&batch);
                    if let Err(error) = result {
                        eprintln!(
                            "[REMu][METAL] completed guest batch finalization failed: {error}"
                        );
                    }
                }
                MTLCommandBufferStatus::Error => {
                    let batch = self
                        .committed_guest_batches
                        .pop_front()
                        .expect("front batch exists");
                    let execution_error = format!(
                        "Metal guest command batch [{}] ended in {:?}",
                        describe_guest_batch(&batch.work),
                        batch.command_buffer.status()
                    );
                    let result = self.finish_terminal_guest_batch(&batch, Some(execution_error));
                    self.statistics.gpu_execution_time +=
                        completed_command_buffer_gpu_duration(&batch.command_buffer);
                    // A terminal native failure cannot be retried in-place. Publish both its
                    // ordered CPU labels and submission ID after reporting the error so the guest
                    // observes one failed queue item instead of a producerless 120-second wait.
                    publish_guest_batch_completion(&batch);
                    if let Err(error) = result {
                        eprintln!("[REMu][METAL] {error}");
                    }
                }
                _ => return,
            }
        }
    }

    fn publish_terminal_guest_memory_writes(
        &mut self,
        pending: &[PendingGuestMemoryWrite],
        result: Result<(), String>,
    ) -> Result<(), String> {
        let publication = apply_pending_guest_memory_writes(pending);
        if publication.is_ok() {
            self.record_produced_guest_memory_writes(pending);
        }
        merge_guest_batch_result(result, "ordered guest memory publication", publication)
    }

    fn finish_terminal_guest_batch(
        &mut self,
        batch: &GuestCommandBatch,
        execution_error: Option<String>,
    ) -> Result<(), String> {
        let execution_succeeded = execution_error.is_none();
        let mut result = execution_error.map_or(Ok(()), Err);
        if execution_succeeded {
            let writeback = write_back_pending_globals(&batch.global_writebacks);
            if writeback.is_ok() {
                self.record_produced_global_writebacks(&batch.global_writebacks);
            }
            result = merge_guest_batch_result(result, "global writeback", writeback);
        }
        // RELEASE_MEM, WRITE_DATA, and DMA fills are command-processor effects ordered after the
        // native work, not shader results. Once the Metal command buffer reaches either terminal
        // state, publish them even when the draw/dispatch or an unrelated global readback failed.
        // This matches the existing policy of retiring the submission after a native error while
        // retaining the error as the primary result.
        self.publish_terminal_guest_memory_writes(&batch.memory_writes, result)
    }

    fn retire_completed_presentations(&mut self) {
        loop {
            let Some(presentation) = self.pending_presentations.front() else {
                return;
            };
            match presentation.command_buffer.status() {
                MTLCommandBufferStatus::Completed => {
                    let presentation = self
                        .pending_presentations
                        .pop_front()
                        .expect("front presentation exists");
                    self.statistics.completed_presents =
                        self.statistics.completed_presents.saturating_add(1);
                    self.statistics.gpu_execution_time +=
                        completed_command_buffer_gpu_duration(&presentation.command_buffer);
                    self.statistics.presentation_queue_latency +=
                        presentation.submitted_at.elapsed();
                }
                MTLCommandBufferStatus::Error => {
                    self.pending_presentations.pop_front();
                    eprintln!("[REMu][METAL] presentation command buffer failed");
                }
                _ => return,
            }
        }
    }

    fn update_window_title(&mut self) {
        const UPDATE_INTERVAL: Duration = Duration::from_millis(500);
        let elapsed = self.statistics.interval_started.elapsed();
        if elapsed < UPDATE_INTERVAL {
            return;
        }
        let seconds = elapsed.as_secs_f64();
        let completed = self.statistics.completed_presents;
        let flip_rate = f64::from(self.statistics.guest_flips) / seconds;
        let present_rate = f64::from(completed) / seconds;
        // The user-visible FPS is the rate of new guest frames, not redraws of the most recent
        // drawable. Presentation can keep running while the guest is stalled on rendering, so
        // reporting it as FPS hid the exact black-screen/audio-starvation condition this title is
        // intended to diagnose.
        let frame_milliseconds = if flip_rate > 0.0 {
            1_000.0 / flip_rate
        } else {
            0.0
        };
        let draw_rate = f64::from(self.statistics.encoded_draws) / seconds;
        let compute_rate = f64::from(self.statistics.encoded_compute_dispatches) / seconds;
        let render_pass_rate = f64::from(self.statistics.encoded_render_passes) / seconds;
        let presentation_latency_milliseconds = if completed == 0 {
            0.0
        } else {
            self.statistics.presentation_queue_latency.as_secs_f64() * 1_000.0
                / f64::from(completed)
        };
        let gpu_utilization =
            (self.statistics.gpu_execution_time.as_secs_f64() * 100.0 / seconds).clamp(0.0, 100.0);
        let guest_batch_rate = f64::from(self.statistics.submitted_guest_batches) / seconds;
        let pipeline_barrier_rate =
            f64::from(self.statistics.coalesced_pipeline_barriers) / seconds;
        let resident_mib = process_resident_bytes().map(|bytes| bytes.div_ceil(1024 * 1024));
        let audio_output = remu_gpu::audio_output_frames();
        let interval_audio_output =
            audio_output.saturating_sub(self.statistics.last_audio_output_frames);
        let audio_output_rate = interval_audio_output as f64 / seconds;
        let audio_underruns = remu_gpu::audio_underrun_frames();
        let interval_underruns =
            audio_underruns.saturating_sub(self.statistics.last_audio_underrun_frames);
        let audio_underrun_percent = if interval_audio_output == 0 {
            0.0
        } else {
            let output = u32::try_from(interval_audio_output).unwrap_or(u32::MAX);
            let underruns = u32::try_from(interval_underruns).unwrap_or(u32::MAX);
            f64::from(underruns) * 100.0 / f64::from(output)
        };
        let resident =
            resident_mib.map_or_else(|| "RSS ?".to_owned(), |mib| format!("RSS {mib} MiB"));
        let pending_gpu_events =
            remu_gpu::gpu_events_in_flight().saturating_add(self.receiver.pending_event_count());
        let title = format!(
            "REMu | FPS {flip_rate:.1} ({frame_milliseconds:.1} ms) | Present {present_rate:.1}/s | GPU {gpu_utilization:.0}% Lat {presentation_latency_milliseconds:.1} ms | {resident} | Draw {draw_rate:.0}/s RP {render_pass_rate:.0}/s CS {compute_rate:.0}/s | CB {guest_batch_rate:.0}/s PB {pipeline_barrier_rate:.0}/s ({}) | Q {}/{} | Audio {audio_output_rate:.0}/s U {audio_underrun_percent:.1}%",
            self.committed_guest_batches.len(),
            pending_gpu_events,
            self.pending_presentations.len(),
        );
        if let Some(window) = self.window.as_ref() {
            window.set_title(&title);
        }
        if self.statistics.last_logged.elapsed() >= Duration::from_secs(5) {
            eprintln!("[REMu][PERF] {title}");
            eprintln!(
                "[REMu][PERF-CAUSES] RP-end compute={:.0}/s key={:.0}/s flush={:.0}/s | CB tick={:.0}/s memory={:.0}/s wait={:.0}/s flip={:.0}/s capacity={:.0}/s sync={:.0}/s",
                f64::from(self.statistics.render_pass_ends_for_compute) / seconds,
                f64::from(self.statistics.render_pass_ends_for_target_change) / seconds,
                f64::from(self.statistics.render_pass_ends_for_batch_flush) / seconds,
                f64::from(self.statistics.batch_flushes_from_event_loop) / seconds,
                f64::from(self.statistics.batch_flushes_for_guest_memory_write) / seconds,
                f64::from(self.statistics.batch_flushes_for_blocked_wait) / seconds,
                f64::from(self.statistics.batch_flushes_for_flip_snapshot) / seconds,
                f64::from(self.statistics.batch_flushes_for_work_capacity) / seconds,
                f64::from(self.statistics.batch_flushes_for_cpu_synchronization) / seconds,
            );
            self.statistics.last_logged = Instant::now();
        }
        self.statistics.interval_started = Instant::now();
        self.statistics.guest_flips = 0;
        self.statistics.completed_presents = 0;
        self.statistics.submitted_guest_batches = 0;
        self.statistics.coalesced_pipeline_barriers = 0;
        self.statistics.encoded_draws = 0;
        self.statistics.encoded_compute_dispatches = 0;
        self.statistics.encoded_render_passes = 0;
        self.statistics.render_pass_ends_for_compute = 0;
        self.statistics.render_pass_ends_for_target_change = 0;
        self.statistics.render_pass_ends_for_batch_flush = 0;
        self.statistics.batch_flushes_from_event_loop = 0;
        self.statistics.batch_flushes_for_guest_memory_write = 0;
        self.statistics.batch_flushes_for_blocked_wait = 0;
        self.statistics.batch_flushes_for_flip_snapshot = 0;
        self.statistics.batch_flushes_for_work_capacity = 0;
        self.statistics.batch_flushes_for_cpu_synchronization = 0;
        self.statistics.gpu_execution_time = Duration::ZERO;
        self.statistics.presentation_queue_latency = Duration::ZERO;
        self.statistics.last_audio_output_frames = audio_output;
        self.statistics.last_audio_underrun_frames = audio_underruns;
    }

    fn end_guest_pipeline_barrier(&mut self, _barrier: remu_gpu::GpuPipelineBarrier) {
        let Some(batch) = self.open_guest_batch.as_mut() else {
            return;
        };
        // EVENT_WRITE partial flushes are memory dependencies, not submission boundaries. Kyty
        // emits a Vulkan memory barrier in the existing command buffer and shadPS4 likewise does
        // not submit at these events. Keeping a Metal render pass open is particularly important
        // on a tile GPU: ending it here stores every attachment, and the next draw immediately
        // reloads the same attachments.
        //
        // Compute encoders are already ended immediately after each dispatch, so Metal's serial
        // encoder order and tracked resource hazards cover inter-pass dependencies. When the
        // current work is a render pass, use Metal's in-pass all-resource barrier to make earlier
        // vertex/fragment writes visible to following draws without spilling the attachments.
        if let Some(pass) = batch.render_pass.as_ref() {
            const ALL_RESOURCE_SCOPES: NSUInteger = (1 << 0) | (1 << 1) | (1 << 2);
            let render_stages = MTLRenderStages::Vertex | MTLRenderStages::Fragment;
            // SAFETY: `pass.encoder` is a live MTLRenderCommandEncoder and the selector has been
            // available on macOS since 10.14. The three NSUInteger arguments exactly match
            // MTLBarrierScope, `afterStages`, and `beforeStages`.
            unsafe {
                let _: () = msg_send![
                    pass.encoder.as_ref(),
                    memoryBarrierWithScope: ALL_RESOURCE_SCOPES
                    afterStages: render_stages.bits()
                    beforeStages: render_stages.bits()
                ];
            }
        }
        self.statistics.coalesced_pipeline_barriers = self
            .statistics
            .coalesced_pipeline_barriers
            .saturating_add(1);
    }

    fn flush_guest_command_batch(
        &mut self,
        wait: bool,
        reason: GuestBatchFlushReason,
    ) -> Result<(), String> {
        let Some(mut batch) = self.open_guest_batch.take() else {
            return Ok(());
        };
        match reason {
            GuestBatchFlushReason::EventLoop => {
                self.statistics.batch_flushes_from_event_loop = self
                    .statistics
                    .batch_flushes_from_event_loop
                    .saturating_add(1);
            }
            GuestBatchFlushReason::BlockedWait => {
                self.statistics.batch_flushes_for_blocked_wait = self
                    .statistics
                    .batch_flushes_for_blocked_wait
                    .saturating_add(1);
            }
            GuestBatchFlushReason::FlipSnapshot => {
                self.statistics.batch_flushes_for_flip_snapshot = self
                    .statistics
                    .batch_flushes_for_flip_snapshot
                    .saturating_add(1);
            }
            GuestBatchFlushReason::WorkCapacity => {
                self.statistics.batch_flushes_for_work_capacity = self
                    .statistics
                    .batch_flushes_for_work_capacity
                    .saturating_add(1);
            }
            GuestBatchFlushReason::CpuSynchronization => {
                self.statistics.batch_flushes_for_cpu_synchronization = self
                    .statistics
                    .batch_flushes_for_cpu_synchronization
                    .saturating_add(1);
            }
        }
        if batch.end_render_pass() {
            self.statistics.render_pass_ends_for_batch_flush = self
                .statistics
                .render_pass_ends_for_batch_flush
                .saturating_add(1);
        }
        // Writable and oversized bindings are shared only within this ordered batch. Rotating
        // them here keeps completed writebacks immutable while reducing per-draw Metal allocation
        // to at most one buffer per unchanged guest address range per batch.
        self.transient_global_buffer_cache.clear();
        batch.command_buffer.commit();
        self.statistics.submitted_guest_batches =
            self.statistics.submitted_guest_batches.saturating_add(1);
        if wait {
            batch.command_buffer.wait_until_completed();
            let execution_error =
                (batch.command_buffer.status() != MTLCommandBufferStatus::Completed).then(|| {
                    format!(
                        "Metal guest command batch [{}] ended in {:?}",
                        describe_guest_batch(&batch.work),
                        batch.command_buffer.status()
                    )
                });
            let result = self.finish_terminal_guest_batch(&batch, execution_error);
            self.statistics.gpu_execution_time +=
                completed_command_buffer_gpu_duration(&batch.command_buffer);
            publish_guest_batch_completion(&batch);
            result?;
        } else {
            self.committed_guest_batches.push_back(batch);
        }
        Ok(())
    }

    fn wait_for_guest_command_batches(&mut self) -> Result<(), String> {
        self.flush_guest_command_batch(false, GuestBatchFlushReason::CpuSynchronization)?;
        let mut result = Ok(());
        while let Some(batch) = self.committed_guest_batches.pop_front() {
            batch.command_buffer.wait_until_completed();
            let execution_error =
                (batch.command_buffer.status() != MTLCommandBufferStatus::Completed).then(|| {
                    format!(
                        "Metal guest command batch [{}] ended in {:?}",
                        describe_guest_batch(&batch.work),
                        batch.command_buffer.status()
                    )
                });
            let batch_result = self.finish_terminal_guest_batch(&batch, execution_error);
            self.statistics.gpu_execution_time +=
                completed_command_buffer_gpu_duration(&batch.command_buffer);
            publish_guest_batch_completion(&batch);
            // Drain every already-committed batch even after one fails. Returning early here used
            // to abandon later RELEASE_MEM labels and completion IDs during shutdown or a
            // synchronous readback, converting a preserved Metal error into an unrelated guest
            // RenderThread timeout.
            result = merge_guest_batch_result(result, "later ordered guest batch", batch_result);
        }
        result
    }

    fn execute_guest_memory_release(&mut self, release: GpuMemoryRelease) -> Result<(), String> {
        if release.address == 0 {
            return Err("release destination is null".to_owned());
        }
        validate_guest_memory_release(release)?;
        self.queue_pending_guest_memory_write(PendingGuestMemoryWrite::Release(release))
    }

    fn execute_guest_memory_copy(&mut self, copy: GpuMemoryCopy) -> Result<(), String> {
        // GFX10 selector 3 is the regular address path with L2 participation; selector 0 is the
        // uncached address path. Both name guest virtual memory. Source selector 2 repeats the
        // packet's inline low dword across the destination; selector 1 names unsupported GDS.
        if !matches!(copy.source_selection, 0 | 2 | 3)
            || !matches!(copy.destination_selection, 0 | 3)
        {
            return Err(format!(
                "unsupported selectors source={} destination={}",
                copy.source_selection, copy.destination_selection
            ));
        }
        let byte_count = usize::try_from(copy.byte_count)
            .map_err(|_| "DMA byte count does not fit the host address space")?;
        if byte_count == 0 || copy.byte_count > 512 * 1024 * 1024 {
            return Err(format!("invalid DMA byte count {:#x}", copy.byte_count));
        }
        copy.destination_address
            .checked_add(copy.byte_count)
            .ok_or("DMA destination range overflows")?;
        record_diagnostic_memory_copy(copy);

        if copy.source_selection == 2 {
            if !copy.destination_address.is_multiple_of(4) || !copy.byte_count.is_multiple_of(4) {
                return Err(format!(
                    "immediate DMA fill destination={:#x} bytes={:#x} is not dword aligned",
                    copy.destination_address, copy.byte_count
                ));
            }
            self.queue_pending_guest_memory_write(PendingGuestMemoryWrite::Fill {
                address: copy.destination_address,
                value: copy.source_address as u32,
                dword_count: copy.byte_count / 4,
            })?;
        } else {
            copy.source_address
                .checked_add(copy.byte_count)
                .ok_or("DMA source range overflows")?;

            // DMA observes preceding GPU writes to its source, but it does not need to drain
            // unrelated rendering. Most Unity DMA_DATA packets move CPU-populated upload pages;
            // synchronously waiting for the complete Metal queue for every one of those packets
            // cuts frame rate in half. Track actual writable guest ranges so only a real
            // read-after-write dependency pays for retirement/readback.
            let source_descriptors = self
                .resident_guest_textures
                .values()
                .filter(|resident| resident.gpu_written)
                .filter(|resident| {
                    guest_texture_byte_count(resident.descriptor).is_some_and(|length| {
                        guest_ranges_overlap(
                            resident.descriptor.address,
                            length,
                            copy.source_address,
                            copy.byte_count,
                        )
                    })
                })
                .map(|resident| resident.descriptor)
                .collect::<Vec<_>>();
            let pending_source_write = self.open_guest_batch.as_ref().is_some_and(|batch| {
                guest_batch_writes_overlap(batch, copy.source_address, copy.byte_count)
            }) || self.committed_guest_batches.iter().any(|batch| {
                guest_batch_writes_overlap(batch, copy.source_address, copy.byte_count)
            });
            let source_was_gpu_written = !source_descriptors.is_empty();
            if !source_descriptors.is_empty() || pending_source_write {
                self.wait_for_guest_command_batches()?;
            }
            if !source_descriptors.is_empty() {
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
                for descriptor in source_descriptors {
                    let refresh = refresh_resident_texture_write_watch(
                        self.resident_guest_textures
                            .get_mut(&descriptor)
                            .expect("collected DMA source texture exists"),
                    )?;
                    if refresh == WriteWatchRefresh::Replaced {
                        self.resident_guest_textures.remove(&descriptor);
                        continue;
                    }
                    if refresh == WriteWatchRefresh::Reprotecting {
                        return Err(format!(
                            "DMA source mapping {:#x} is being reprotected",
                            descriptor.address
                        ));
                    }
                    let resident = self
                        .resident_guest_textures
                        .get_mut(&descriptor)
                        .expect("collected DMA source texture exists");
                    write_resident_texture_contents_to_guest(&device, &queue, resident)?;
                    rearm_guest_texture_after_gpu_writeback(&resident.write_watch)?;
                    resident.gpu_written = false;
                }
            }

            if !copy.cpu_applied || source_was_gpu_written || pending_source_write {
                // SAFETY: the runtime decodes DMA only for bounded non-null guest virtual
                // addresses. The guest owns these mappings for the lifetime of the ordered GPU
                // event stream. `copy` intentionally provides memmove semantics for legal
                // overlapping packets.
                unsafe {
                    std::ptr::copy(
                        copy.source_address as *const u8,
                        copy.destination_address as *mut u8,
                        byte_count,
                    );
                }
            }
            self.record_submitted_gpu_waits_in_range(copy.destination_address, copy.byte_count);
        }

        let destination_descriptors = self
            .resident_guest_textures
            .keys()
            .copied()
            .filter(|descriptor| {
                guest_texture_byte_count(*descriptor).is_some_and(|length| {
                    guest_ranges_overlap(
                        descriptor.address,
                        length,
                        copy.destination_address,
                        copy.byte_count,
                    )
                })
            })
            .collect::<Vec<_>>();
        for descriptor in destination_descriptors {
            self.resident_guest_textures.remove(&descriptor);
        }
        self.multisample_textures.retain(|key, _| {
            guest_texture_byte_count(key.descriptor).is_none_or(|length| {
                !guest_ranges_overlap(
                    key.descriptor.address,
                    length,
                    copy.destination_address,
                    copy.byte_count,
                )
            })
        });
        retire_sampled_textures_matching(&mut self.sampled_texture_cache, |key| {
            guest_texture_byte_count(key.descriptor).is_some_and(|length| {
                guest_ranges_overlap(
                    key.descriptor.address,
                    length,
                    copy.destination_address,
                    copy.byte_count,
                )
            })
        });
        Ok(())
    }

    fn execute_guest_memory_write(&mut self, write: &GpuMemoryWrite) -> Result<(), String> {
        if write.address == 0 || write.values.is_empty() {
            return Ok(());
        }
        if !write.address.is_multiple_of(4) {
            return Err(format!(
                "WRITE_DATA destination {:#x} is unaligned",
                write.address
            ));
        }
        validate_guest_memory_write(write)?;
        self.queue_pending_guest_memory_write(PendingGuestMemoryWrite::Write(write.clone()))
    }

    fn execute_guest_memory_wait(
        &mut self,
        source: OrderedEventSource,
        ticket_id: Option<GpuWaitId>,
        wait: GpuMemoryWait,
    ) -> Result<GuestMemoryWaitStatus, String> {
        let key = GuestMemoryWaitKey::new(source, wait);
        if let Some(ticket_id) = ticket_id
            && let Some(value) = remu_gpu::take_gpu_wait_latch(ticket_id)
        {
            self.guest_memory_waits.complete(key);
            eprintln!(
                "[REMu][METAL] resumed WAIT_REG_MEM from submitted producer ticket address={:#x} value={value:#x}",
                wait.address
            );
            return Ok(GuestMemoryWaitStatus::Satisfied);
        }
        if wait.address == 0 {
            self.guest_memory_waits.complete(key);
            if let Some(ticket_id) = ticket_id {
                remu_gpu::retire_gpu_wait(ticket_id);
            }
            return Ok(GuestMemoryWaitStatus::Satisfied);
        }
        let required_alignment = if wait.is_64_bit { 8 } else { 4 };
        if !wait.address.is_multiple_of(required_alignment) {
            if let Some(ticket_id) = ticket_id {
                remu_gpu::retire_gpu_wait(ticket_id);
            }
            return Err(format!(
                "{}-bit wait address {:#x} is unaligned",
                required_alignment * 8,
                wait.address
            ));
        }
        // A release/write queued after the preceding Metal batch is the value the ordered GPU
        // wait will observe. When it already satisfies the comparison, the serial Metal queue
        // provides the required ordering without a CPU round trip. The actual guest-memory write
        // is published as that batch retires, before its completion sequence becomes visible.
        // A WAIT whose condition is already true is only an ordering point. Later Metal command
        // buffers are submitted to the same serial queue, so completing every earlier batch on
        // the CPU before reading the label adds no ordering that the queue does not already
        // provide. This is the overwhelmingly common Unity fence pattern (for example 0 < a
        // monotonically increasing reference) and synchronously draining the queue here turns
        // hundreds of otherwise asynchronous draws into one CPU/GPU round trip per frame.
        //
        // A queued write to this exact label takes precedence over the current memory contents:
        // the guest WAIT is ordered after that write. If no such write exists, the live value can
        // be tested immediately. If the queued value does not satisfy the comparison, retirement
        // below publishes it before the next live-memory test.
        match self.pending_guest_memory_value(wait.address, wait.is_64_bit) {
            Some(value) if wait.is_satisfied_by(value) => {
                self.guest_memory_waits.complete(key);
                if let Some(ticket_id) = ticket_id {
                    remu_gpu::retire_gpu_wait(ticket_id);
                }
                return Ok(GuestMemoryWaitStatus::Satisfied);
            }
            Some(_) => {}
            None if wait.is_satisfied_by(read_guest_memory_wait_value(wait)) => {
                self.guest_memory_waits.complete(key);
                if let Some(ticket_id) = ticket_id {
                    remu_gpu::retire_gpu_wait(ticket_id);
                }
                return Ok(GuestMemoryWaitStatus::Satisfied);
            }
            None => {}
        }
        self.retire_completed_guest_batches();
        match self.pending_guest_memory_value(wait.address, wait.is_64_bit) {
            Some(value) if wait.is_satisfied_by(value) => {
                self.guest_memory_waits.complete(key);
                if let Some(ticket_id) = ticket_id {
                    remu_gpu::retire_gpu_wait(ticket_id);
                }
                return Ok(GuestMemoryWaitStatus::Satisfied);
            }
            Some(_) => {}
            None if wait.is_satisfied_by(read_guest_memory_wait_value(wait)) => {
                self.guest_memory_waits.complete(key);
                if let Some(ticket_id) = ticket_id {
                    remu_gpu::retire_gpu_wait(ticket_id);
                }
                return Ok(GuestMemoryWaitStatus::Satisfied);
            }
            None => {}
        }

        // Guest graphics and async-compute queues can wait on labels released by one another.
        // Closing this queue's current Metal batch preserves its work before the wait. The queue
        // scheduler retains this packet at the submission front and runs another guest queue;
        // later commands from this queue are never allowed to overtake the unsatisfied condition.
        if let Err(error) =
            self.flush_guest_command_batch(false, GuestBatchFlushReason::BlockedWait)
        {
            if let Some(ticket_id) = ticket_id {
                remu_gpu::retire_gpu_wait(ticket_id);
            }
            return Err(error);
        }
        if ticket_id.is_none() {
            match self.guest_memory_waits.observe_unsatisfied(key) {
                GuestMemoryWaitRecovery::ProducerLatched(value) => {
                    eprintln!(
                        "[REMu][METAL] resumed WAIT_REG_MEM from a real producer latch address={:#x} value={value:#x}",
                        wait.address
                    );
                    return Ok(GuestMemoryWaitStatus::Satisfied);
                }
                GuestMemoryWaitRecovery::None => {}
            }
        }
        static SUSPENDED_WAITS: AtomicUsize = AtomicUsize::new(0);
        let sequence = SUSPENDED_WAITS.fetch_add(1, Ordering::Relaxed) + 1;
        if sequence <= 16 || sequence.is_power_of_two() {
            let value = read_guest_memory_wait_value(wait);
            let pending_value = self.pending_guest_memory_value(wait.address, wait.is_64_bit);
            eprintln!(
                "[REMu][METAL] suspended WAIT_REG_MEM sequence={sequence} address={:#x} value={value:#x} pending={pending_value:?} mask={:#x} reference={:#x} compare={}",
                wait.address, wait.mask, wait.reference, wait.compare_function
            );
        }
        Ok(GuestMemoryWaitStatus::Blocked)
    }

    fn queue_pending_guest_memory_write(
        &mut self,
        write: PendingGuestMemoryWrite,
    ) -> Result<(), String> {
        // RELEASE_MEM/WRITE_DATA label updates are ordered after the Metal work already encoded
        // in this batch, but do not themselves require a new Metal submission. Publish them when
        // the owning command buffer retires. A later wait can inspect the deferred value, while
        // a genuinely blocked wait and the event-loop boundary still commit the batch.
        if let Some(batch) = self.open_guest_batch.as_mut() {
            batch.memory_writes.push(write);
            return Ok(());
        }
        if let Some(batch) = self.committed_guest_batches.back_mut() {
            batch.memory_writes.push(write);
        } else {
            apply_pending_guest_memory_writes(std::slice::from_ref(&write))?;
            self.record_produced_guest_memory_writes(std::slice::from_ref(&write));
        }
        Ok(())
    }

    fn record_guest_memory_producer(&mut self, address: u64, is_64_bit: bool, value: u64) {
        self.guest_memory_waits
            .record_produced(address, is_64_bit, value);
        remu_gpu::record_gpu_wait_producer(address, is_64_bit, value);
    }

    fn record_submitted_gpu_waits_in_range(&mut self, address: u64, byte_count: u64) {
        let waits = remu_gpu::pending_gpu_waits_overlapping(address, byte_count);
        for (_, wait) in waits {
            let value = read_guest_memory_wait_value(wait);
            self.record_guest_memory_producer(wait.address, wait.is_64_bit, value);
        }
    }

    fn record_produced_global_writebacks(&mut self, pending: &[PendingGlobalWriteback]) {
        for writeback in pending {
            if let Ok(byte_count) = u64::try_from(writeback.byte_count) {
                self.record_submitted_gpu_waits_in_range(writeback.guest_address, byte_count);
            }
        }
    }

    fn record_produced_guest_memory_writes(&mut self, pending: &[PendingGuestMemoryWrite]) {
        for write in pending {
            match write {
                PendingGuestMemoryWrite::Release(release) => match release.value {
                    GpuMemoryReleaseValue::Immediate32(value) => {
                        self.record_guest_memory_producer(release.address, false, u64::from(value));
                    }
                    GpuMemoryReleaseValue::Immediate64(value) => {
                        self.record_guest_memory_producer(release.address, true, value);
                        self.record_guest_memory_producer(
                            release.address,
                            false,
                            value & u64::from(u32::MAX),
                        );
                        self.record_guest_memory_producer(
                            release.address.saturating_add(4),
                            false,
                            value >> 32,
                        );
                    }
                    GpuMemoryReleaseValue::Timestamp => {
                        let value = read_guest_memory_wait_value(GpuMemoryWait {
                            address: release.address,
                            reference: 0,
                            mask: u64::MAX,
                            compare_function: 0,
                            is_64_bit: true,
                        });
                        self.record_guest_memory_producer(release.address, true, value);
                        self.record_guest_memory_producer(
                            release.address,
                            false,
                            value & u64::from(u32::MAX),
                        );
                        self.record_guest_memory_producer(
                            release.address.saturating_add(4),
                            false,
                            value >> 32,
                        );
                    }
                },
                PendingGuestMemoryWrite::Write(write) => {
                    for (index, value) in write.values.iter().copied().enumerate() {
                        let offset = if write.increment_address {
                            u64::try_from(index).unwrap_or(u64::MAX).saturating_mul(4)
                        } else {
                            0
                        };
                        self.record_guest_memory_producer(
                            write.address.saturating_add(offset),
                            false,
                            u64::from(value),
                        );
                    }
                    if write.increment_address {
                        for (index, pair) in write.values.windows(2).enumerate() {
                            let address = write.address.saturating_add(
                                u64::try_from(index).unwrap_or(u64::MAX).saturating_mul(4),
                            );
                            if address.is_multiple_of(8) {
                                let value = u64::from(pair[0]) | (u64::from(pair[1]) << 32);
                                self.record_guest_memory_producer(address, true, value);
                            }
                        }
                    }
                }
                PendingGuestMemoryWrite::Fill { address, value, .. } => {
                    self.record_guest_memory_producer(*address, false, u64::from(*value));
                    if address.is_multiple_of(8) {
                        self.record_guest_memory_producer(
                            *address,
                            true,
                            u64::from(*value) | (u64::from(*value) << 32),
                        );
                    }
                }
            }
        }

        // Large DMA fills are not expanded into the bounded produced-label table. Active waits
        // still latch the exact value they observed at producer completion, including waits on
        // an interior dword or qword of a fill.
        let active = self
            .guest_memory_waits
            .active
            .keys()
            .copied()
            .collect::<Vec<_>>();
        for key in active {
            let width = if key.is_64_bit { 8 } else { 4 };
            if pending
                .iter()
                .any(|write| write.overlaps(key.address, width))
            {
                let value = read_guest_memory_wait_value(key.wait());
                self.record_guest_memory_producer(key.address, key.is_64_bit, value);
            }
        }
        for write in pending {
            let (address, byte_count) = write.range();
            self.record_submitted_gpu_waits_in_range(address, byte_count);
        }
    }

    fn pending_guest_memory_value(&self, address: u64, is_64_bit: bool) -> Option<u64> {
        self.open_guest_batch
            .iter()
            .flat_map(|batch| batch.memory_writes.iter().rev())
            .chain(
                self.committed_guest_batches
                    .iter()
                    .rev()
                    .flat_map(|batch| batch.memory_writes.iter().rev()),
            )
            // The open batch is newer than every committed batch; within each batch, the last
            // write wins. This mirrors guest command-stream order without publishing the label
            // before its preceding Metal work completes.
            .find_map(|write| write.value_at(address, is_64_bit))
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
        match refresh_resident_texture_write_watch(resident)? {
            WriteWatchRefresh::Replaced => {
                self.resident_guest_textures.remove(&descriptor);
                return Ok(());
            }
            WriteWatchRefresh::Reprotecting => return Ok(()),
            WriteWatchRefresh::Active | WriteWatchRefresh::Renewed { .. } => {}
        }
        let resident = self
            .resident_guest_textures
            .get_mut(&descriptor)
            .expect("refreshed resident texture exists");
        write_resident_texture_contents_to_guest(&device, &queue, resident)?;
        rearm_guest_texture_after_gpu_writeback(&resident.write_watch)?;
        resident.gpu_written = false;
        Ok(())
    }

    fn activate_translated_scanout(
        &mut self,
        address: u64,
        width: u32,
        height: u32,
    ) -> Result<bool, String> {
        let direct = self
            .direct_scanout_textures
            .get(&address)
            .filter(|scanout| {
                scanout.initialized
                    && scanout.descriptor.width == width
                    && scanout.descriptor.height == height
            })
            .map(|scanout| (scanout.descriptor, scanout.texture.clone(), "direct"));
        let resident = self
            .resident_guest_textures
            .values()
            .filter(|resident| {
                resident.gpu_written
                    && resident.descriptor.address == address
                    && resident.descriptor.width == width
                    && resident.descriptor.height == height
            })
            .max_by_key(|resident| resident.content_version)
            .map(|resident| (resident.descriptor, resident.texture.clone(), "resident"));
        let Some((descriptor, source_texture, source)) = direct.or(resident) else {
            return Ok(false);
        };
        // Keep the scanout resident on the GPU and let the serial Metal queue order presentation
        // after the guest work that produced it. AGC's embedded flip marker can precede Silent
        // Hill's final composition work, so copying at that marker preserves a stale solid buffer.
        // Presenting the live resident also avoids a 32 MiB 4K blit on every frame.
        self.flush_guest_command_batch(false, GuestBatchFlushReason::FlipSnapshot)?;
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
        let capture_due = self.capture_directory.is_some()
            && self
                .last_guest_capture
                .is_none_or(|last| last.elapsed() >= capture_interval());
        if capture_due
            && matches!(
                descriptor.pixel_format,
                MTLPixelFormat::RGBA8Unorm
                    | MTLPixelFormat::RGBA8Unorm_sRGB
                    | MTLPixelFormat::BGRA8Unorm
                    | MTLPixelFormat::BGRA8Unorm_sRGB
                    | MTLPixelFormat::RGBA16Float
            )
        {
            let rendered = read_texture_linear(&device, &queue, &source_texture, descriptor)?;
            self.capture_translated_frame_if_changed(descriptor, &rendered);
        }
        self.frame_texture = Some(source_texture);
        self.frame_texture_size = (descriptor.width, descriptor.height);
        static LOGGED_TRANSLATED_SCANOUTS: OnceLock<Mutex<HashSet<(u64, &'static str)>>> =
            OnceLock::new();
        let first_for_source = LOGGED_TRANSLATED_SCANOUTS
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert((address, source));
        if first_for_source {
            eprintln!(
                "[REMu][METAL] presenting translated {source} scanout sequence={} at {address:#x} {}x{} format={:?} tile={}",
                self.next_flip_sequence,
                descriptor.width,
                descriptor.height,
                descriptor.pixel_format,
                descriptor.tile_mode,
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
                let options = remu_shader_metal::MetalCompileOptions::compute_wave(
                    dispatch.launch.lds_size_bytes,
                    dispatch.launch.wave_lane_count,
                );
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
            if let Some(directory) = std::env::var_os("REMU_DUMP_MSL_DIR")
                && dump_msl_compute_filter_matches(dispatch.shader_address)
                && let Err(error) = dump_compute_shader_msl(
                    Path::new(&directory),
                    dispatch.shader_address,
                    &snapshot.program.instructions,
                    &compiled.shader.source,
                )
            {
                eprintln!(
                    "[REMu][METAL] could not dump compute shader cs={:#x}: {error}",
                    dispatch.shader_address
                );
            }
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
        let wave64_split_safe = compute_program_can_split_wave64(&snapshot.program);
        validate_compute_launch_for_metal(
            dispatch.launch,
            execution_width,
            wave64_split_safe,
            pipeline.max_total_threads_per_threadgroup(),
            device.max_threadgroup_memory_length(),
            pipeline.static_threadgroup_memory_length(),
            dynamic_lds,
        )
        .map_err(|error| {
            if dispatch.launch.wave_lane_count == 64 && execution_width == 32 && !wave64_split_safe
            {
                format!(
                    "{error}; wave-sensitive instructions [{}]",
                    wave64_split_blockers(&snapshot.program).join(",")
                )
            } else {
                error
            }
        })?;

        for binding in evaluation
            .image_bindings
            .iter()
            .filter(|binding| is_storage_image_opcode(binding.opcode))
            .filter(|binding| !image_resource_descriptor_is_null(&binding.resource_descriptor))
        {
            let descriptor = decode_image_descriptor(&binding.resource_descriptor)
                .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
            if descriptor.texture_type == 9 {
                self.ensure_resident_texture(&device, descriptor.guest)?;
            }
        }
        self.synchronize_sampled_aliases(&[evaluation])?;

        let buffers = self.upload_global_buffers(&device, evaluation)?;
        let textures = upload_image_textures(
            &device,
            &queue,
            evaluation,
            &mut self.resident_guest_textures,
            &self.multisample_textures,
            &mut self.sampled_texture_cache,
            &mut self.sampled_texture_write_watches,
            &mut self.fallback_texture_cache,
            &mut self.sampled_texture_admission_order,
            &mut self.next_sampled_texture_admission_id,
        )?;
        let (sampler_buffer, _samplers) = self.cached_sampler_argument_buffer(
            &device,
            &function,
            bindings.sampler_argument_buffer_index,
            evaluation,
        )?;
        let buffer_indices = (0..buffers.len())
            .map(|index| u64::try_from(index).map_err(|_| "compute buffer index overflow"))
            .collect::<Result<Vec<_>, _>>()?;
        let texture_indices = (0..textures.len())
            .map(|index| u64::try_from(index).map_err(|_| "compute texture index overflow"))
            .collect::<Result<Vec<_>, _>>()?;
        let scalar_metadata =
            shader_scalar_metadata_for_registers(evaluation, &dispatch.shader_registers, 0x240, 0)?;
        let scalar_byte_count = scalar_metadata
            .len()
            .checked_mul(4)
            .and_then(|count| u64::try_from(count).ok())
            .ok_or("compute scalar byte count overflow")?;
        let mut pending_writebacks = Vec::new();
        collect_pending_global_writebacks(&buffers, evaluation, &mut pending_writebacks)?;
        let mut pending_guest_write_ranges = Vec::new();
        collect_pending_guest_write_ranges(&buffers, evaluation, &mut pending_guest_write_ranges)?;
        if self.open_guest_batch.is_none() {
            let command_buffer = queue.new_command_buffer().to_owned();
            command_buffer.set_label("REMu translated guest batch");
            self.open_guest_batch = Some(GuestCommandBatch {
                command_buffer,
                work: Vec::new(),
                global_writebacks: Vec::new(),
                guest_write_ranges: Vec::new(),
                memory_writes: Vec::new(),
                render_pass: None,
                completion_sequences: Vec::new(),
            });
        }
        {
            let batch = self
                .open_guest_batch
                .as_mut()
                .expect("guest command batch was opened above");
            if batch.end_render_pass() {
                self.statistics.render_pass_ends_for_compute = self
                    .statistics
                    .render_pass_ends_for_compute
                    .saturating_add(1);
            }
            batch.work.push(GuestBatchWork::Compute {
                shader: dispatch.shader_address,
            });
            append_unique_global_writebacks(&mut batch.global_writebacks, pending_writebacks);
            append_unique_guest_write_ranges(
                &mut batch.guest_write_ranges,
                pending_guest_write_ranges,
            );
            let encoder = batch.command_buffer.new_compute_command_encoder();
            encoder.set_label(&format!("REMu Gen5 compute {:#x}", dispatch.shader_address));
            encoder.set_compute_pipeline_state(&pipeline);
            for (index, buffer) in buffer_indices.iter().zip(&buffers) {
                encoder.set_buffer(*index, Some(&buffer.buffer), buffer.offset);
            }
            encoder.set_bytes(
                u64::from(bindings.initial_scalar_buffer_index),
                scalar_byte_count,
                scalar_metadata.as_ptr().cast::<c_void>(),
            );
            for (index, texture) in texture_indices.iter().zip(&textures) {
                encoder.set_texture(*index, Some(&texture.texture));
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
        let diagnostic_sampled = diagnostic_texture_writer_enabled().then(|| {
            sampled_image_bindings(evaluation)
                .filter_map(|binding| decode_image_descriptor(&binding.resource_descriptor).ok())
                .map(|descriptor| descriptor.guest)
                .collect::<Vec<_>>()
        });
        record_diagnostic_global_writers(
            "compute",
            dispatch.sequence,
            dispatch.shader_address,
            evaluation,
        );
        for texture in &textures {
            if texture.writable {
                if let Some(sampled) = &diagnostic_sampled {
                    record_diagnostic_texture_writer(
                        texture.descriptor.guest,
                        "compute",
                        dispatch.sequence,
                        dispatch.shader_address,
                        sampled.clone(),
                    );
                }
                self.mark_resident_texture_written(texture.descriptor.guest);
            }
        }
        if self
            .open_guest_batch
            .as_ref()
            .is_some_and(|batch| batch.work.len() >= GUEST_BATCH_WORK_CAPACITY)
        {
            self.flush_guest_command_batch(false, GuestBatchFlushReason::WorkCapacity)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn execute_compiled_draw(
        &mut self,
        draw: &IndexedDraw,
        key: &GraphicsPipelineKey,
    ) -> Result<(), String> {
        self.retire_completed_guest_batches();
        let started = Instant::now();
        let trace = draw.sequence <= 4 || draw.sequence.is_power_of_two();
        let targets = decode_render_targets(draw)?;
        let depth = decode_depth_target(draw)?;
        let raster = decode_raster_state(draw)?;
        self.capture_scanout_before_pixel_shader(draw, &targets)?;
        if trace_scanout_writes() && draw.sequence >= diagnostic_dump_target_minimum_draw() {
            static LOGGED_SCANOUT_WRITES: AtomicUsize = AtomicUsize::new(0);
            for target in &targets {
                if self
                    .scanout_snapshot_textures
                    .contains_key(&target.texture.address)
                {
                    let write = LOGGED_SCANOUT_WRITES.fetch_add(1, Ordering::Relaxed);
                    if write < 512 {
                        eprintln!(
                            "[REMu][METAL][SCANOUT-WRITE] index={write} draw={} es={:#x} ps={:#x} ps_insns={} indexed={} count={} instances={} primitive={:#x} target={:?} depth={depth:?} pipeline_targets={:?} images={}/{} globals={}/{}",
                            draw.sequence,
                            draw.export_shader_address,
                            draw.pixel_shader_address,
                            draw.pixel_shader
                                .as_ref()
                                .map_or(0, |shader| shader.program.instructions.len()),
                            draw.indexed,
                            draw.index_count,
                            draw.instance_count,
                            draw.primitive_type,
                            target,
                            key.targets,
                            draw.export_shader_evaluation
                                .as_ref()
                                .map_or(0, |evaluation| evaluation.image_bindings.len()),
                            draw.pixel_shader_evaluation
                                .as_ref()
                                .map_or(0, |evaluation| evaluation.image_bindings.len()),
                            draw.export_shader_evaluation
                                .as_ref()
                                .map_or(0, |evaluation| evaluation.global_memory_bindings.len()),
                            draw.pixel_shader_evaluation
                                .as_ref()
                                .map_or(0, |evaluation| evaluation.global_memory_bindings.len()),
                        );
                    }
                }
            }
        }
        let diagnostic_complex_msaa_raster = targets.iter().any(|target| target.sample_count > 1)
            && draw
                .pixel_shader
                .as_ref()
                .is_some_and(|shader| shader.program.instructions.len() == 370);
        // The combined full-raster diagnostic is intended to probe the complex gameplay pass.
        // Scoping it here avoids turning every tiny MSAA clear/copy draw into catastrophic
        // full-screen overdraw during loading. The independent cull/depth probes retain their
        // original all-draw behavior.
        let diagnostic_disable_combined_raster = diagnostic_disable_depth_and_culling();
        let diagnostic_disable_culling_for_draw = diagnostic_disable_culling()
            && (!diagnostic_disable_combined_raster || diagnostic_complex_msaa_raster);
        let diagnostic_disable_depth_for_draw = diagnostic_disable_depth()
            && (!diagnostic_disable_combined_raster || diagnostic_complex_msaa_raster);
        if trace_target_sampling() {
            static LOGGED_TARGETS: OnceLock<Mutex<HashSet<(u64, u32, u32, MTLPixelFormat, u32)>>> =
                OnceLock::new();
            let mut logged = LOGGED_TARGETS
                .get_or_init(|| Mutex::new(HashSet::new()))
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for target in &targets {
                let descriptor = target.texture;
                let key = (
                    descriptor.address,
                    descriptor.width,
                    descriptor.height,
                    descriptor.pixel_format,
                    descriptor.tile_mode,
                );
                if descriptor.width == 1_920
                    && descriptor.height == 1_080
                    && logged.len() < 256
                    && logged.insert(key)
                {
                    let primary = target.guest_slot * CB_COLOR_REGISTER_STRIDE;
                    let fmask = register_value(&draw.context_registers, CB_COLOR0_FMASK + primary);
                    let fmask_ext = register_value(
                        &draw.context_registers,
                        CB_COLOR0_FMASK_BASE_EXT + target.guest_slot,
                    );
                    let fmask_address = fmask
                        .zip(fmask_ext)
                        .map(|(base, ext)| (u64::from(ext & 0xff) << 40) | (u64::from(base) << 8));
                    eprintln!(
                        "[REMu][METAL][TARGET-WRITE] draw={} es={:#x} ps={:#x} ps_insns={} slot={} address={:#x} fmask={fmask_address:x?} extent={}x{} pitch={} format={:?} tile={} mask={:#x} samples={}",
                        draw.sequence,
                        draw.export_shader_address,
                        draw.pixel_shader_address,
                        draw.pixel_shader
                            .as_ref()
                            .map_or(0, |shader| shader.program.instructions.len()),
                        target.guest_slot,
                        descriptor.address,
                        descriptor.width,
                        descriptor.height,
                        descriptor.pitch,
                        descriptor.pixel_format,
                        descriptor.tile_mode,
                        target.write_mask,
                        target.sample_count,
                    );
                }
            }
            static GAMEPLAY_TARGET: OnceLock<Mutex<Option<GuestTextureDescriptor>>> =
                OnceLock::new();
            let gameplay_target = {
                let mut gameplay_target = GAMEPLAY_TARGET
                    .get_or_init(|| Mutex::new(None))
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if gameplay_target.is_none()
                    && draw
                        .pixel_shader
                        .as_ref()
                        .is_some_and(|shader| shader.program.instructions.len() == 370)
                {
                    *gameplay_target = targets
                        .iter()
                        .find(|target| target.sample_count > 1)
                        .map(|target| target.texture);
                }
                *gameplay_target
            };
            if let Some(gameplay_target) = gameplay_target {
                static WRITES: AtomicUsize = AtomicUsize::new(0);
                for target in &targets {
                    if target.texture.address == gameplay_target.address {
                        let write = WRITES.fetch_add(1, Ordering::Relaxed);
                        if write < 128 {
                            eprintln!(
                                "[REMu][METAL][GAMEPLAY-WRITE] draw={} es={:#x} ps={:#x} ps_insns={} samples={} format={:?} mask={:#x}",
                                draw.sequence,
                                draw.export_shader_address,
                                draw.pixel_shader_address,
                                draw.pixel_shader
                                    .as_ref()
                                    .map_or(0, |shader| shader.program.instructions.len()),
                                target.sample_count,
                                target.texture.pixel_format,
                                target.write_mask,
                            );
                        }
                    }
                }
            }
        }
        if !self.compiled_shader_pairs.contains_key(key) {
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
        // Unity lowers color clears to a resource-free fullscreen draw. On Gen5 the clear updates
        // the multisample attachment without publishing a new shader-readable resolve; a later
        // content pass performs that resolve. Resolving every Metal pass made the final white
        // clear overwrite the adjacent single-sample image sampled by Superliminal's postprocess.
        let multisample_constant_clear = targets.iter().any(|target| target.sample_count > 1)
            && depth.is_none()
            && !draw.indexed
            && draw.index_count == 3
            && draw.instance_count == 1
            && draw
                .pixel_shader
                .as_ref()
                .is_some_and(|shader| shader.program.instructions.len() <= 6)
            && pixel_evaluation.image_bindings.is_empty()
            && pixel_evaluation.global_memory_bindings.is_empty();
        let resolve_multisample_targets = !multisample_constant_clear;
        if trace_msaa_writes() && draw.sequence >= diagnostic_dump_target_minimum_draw() {
            static LOGGED_MULTISAMPLE_WRITES: AtomicUsize = AtomicUsize::new(0);
            for target in targets.iter().filter(|target| target.sample_count > 1) {
                let write = LOGGED_MULTISAMPLE_WRITES.fetch_add(1, Ordering::Relaxed);
                if write < 2_048 {
                    eprintln!(
                        "[REMu][METAL][MSAA-WRITE] draw={} es={:#x} ps={:#x} instructions={} indexed={} count={} instances={} target={:#x} samples={} resolve={} depth={} sampled={} globals={}",
                        draw.sequence,
                        draw.export_shader_address,
                        draw.pixel_shader_address,
                        draw.pixel_shader
                            .as_ref()
                            .map_or(0, |shader| shader.program.instructions.len()),
                        draw.indexed,
                        draw.index_count,
                        draw.instance_count,
                        target.texture.address,
                        target.sample_count,
                        resolve_multisample_targets,
                        depth.is_some(),
                        pixel_evaluation.image_bindings.len(),
                        pixel_evaluation.global_memory_bindings.len(),
                    );
                }
            }
        }
        if multisample_constant_clear && trace_target_sampling() {
            static LOGGED_MULTISAMPLE_CLEARS: AtomicUsize = AtomicUsize::new(0);
            let sequence = LOGGED_MULTISAMPLE_CLEARS.fetch_add(1, Ordering::Relaxed) + 1;
            if sequence <= 16 || sequence.is_power_of_two() {
                eprintln!(
                    "[REMu][METAL][MSAA-CLEAR] draw={} ps={:#x} instructions={} preserves prior resolve",
                    draw.sequence,
                    draw.pixel_shader_address,
                    draw.pixel_shader
                        .as_ref()
                        .map_or(0, |shader| shader.program.instructions.len()),
                );
            }
        }
        if diagnostic_texture_writer_enabled() {
            let diagnostic_sampled = sampled_image_bindings(pixel_evaluation)
                .filter_map(|binding| decode_image_descriptor(&binding.resource_descriptor).ok())
                .map(|descriptor| descriptor.guest)
                .collect::<Vec<_>>();
            for target in &targets {
                record_diagnostic_texture_writer(
                    target.texture,
                    "draw",
                    draw.sequence,
                    draw.pixel_shader_address,
                    diagnostic_sampled.clone(),
                );
            }
            record_diagnostic_global_writers(
                "vertex",
                draw.sequence,
                draw.export_shader_address,
                export_evaluation,
            );
            record_diagnostic_global_writers(
                "pixel",
                draw.sequence,
                draw.pixel_shader_address,
                pixel_evaluation,
            );
        }
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
        let mut target_resolve_textures = Vec::with_capacity(targets.len());
        let mut clear_uninitialized_targets = Vec::with_capacity(targets.len());
        for target in &targets {
            let resolved_texture = if target.direct_scanout {
                let texture = self.ensure_direct_scanout_texture(&device, target.texture);
                clear_uninitialized_targets
                    .push(self.direct_scanout_needs_initial_clear(target.texture.address));
                texture
            } else {
                let has_resident_storage = self.resident_guest_textures.values().any(|resident| {
                    guest_texture_descriptors_share_storage(resident.descriptor, target.texture)
                });
                let clear_compressed_initial_contents =
                    (target.dcc_enabled || target.fast_clear_enabled) && !has_resident_storage;
                clear_uninitialized_targets.push(clear_compressed_initial_contents);
                if clear_compressed_initial_contents {
                    static DISCARDED_DCC_TARGETS: AtomicUsize = AtomicUsize::new(0);
                    let sequence = DISCARDED_DCC_TARGETS.fetch_add(1, Ordering::Relaxed) + 1;
                    if sequence <= 16 || sequence.is_power_of_two() {
                        eprintln!(
                            "[REMu][METAL] initializing compressed color target {} at {:#x} as uncompressed Metal storage dcc={} fast_clear={} sequence={sequence}",
                            target.guest_slot,
                            target.texture.address,
                            target.dcc_enabled,
                            target.fast_clear_enabled,
                        );
                    }
                }
                self.ensure_layered_render_target_texture(&device, *target)
                    .map_err(|error| {
                        format!(
                            "render target {} guest memory is unreadable: {error}",
                            target.guest_slot
                        )
                    })?
            };
            if target.sample_count > 1 {
                let (multisample, clear_multisample) =
                    self.ensure_multisample_texture(&device, target.texture, target.sample_count)?;
                target_textures.push(multisample);
                target_resolve_textures
                    .push(resolve_multisample_targets.then_some(resolved_texture));
                *clear_uninitialized_targets
                    .last_mut()
                    .expect("target clear state was just added") = clear_multisample;
            } else {
                target_textures.push(resolved_texture);
                target_resolve_textures.push(None);
            }
        }
        let resolved_depth_texture = depth
            .map(|depth| {
                self.ensure_resident_texture(&device, depth.texture)
                    .map_err(|error| format!("depth target guest memory is unreadable: {error}"))
            })
            .transpose()?;
        let (depth_texture, depth_resolve_texture, clear_uninitialized_depth) =
            if let (Some(depth), Some(resolved)) = (depth, resolved_depth_texture) {
                if depth.sample_count > 1 {
                    let (multisample, clear) = self.ensure_multisample_texture(
                        &device,
                        depth.texture,
                        depth.sample_count,
                    )?;
                    (Some(multisample), Some(resolved), clear)
                } else {
                    (Some(resolved), None, false)
                }
            } else {
                (None, None, false)
            };

        for evaluation in [export_evaluation, pixel_evaluation] {
            for binding in evaluation
                .image_bindings
                .iter()
                .filter(|binding| is_storage_image_opcode(binding.opcode))
                .filter(|binding| !image_resource_descriptor_is_null(&binding.resource_descriptor))
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
                if let Some(depth) = depth
                    && descriptor.guest != depth.texture
                    && guest_texture_descriptors_overlap(descriptor.guest, depth.texture)
                {
                    return Err(format!(
                        "writable image at PC {:#x} aliases the depth target through an incompatible view",
                        binding.pc
                    ));
                }
                if descriptor.texture_type == 9 {
                    self.ensure_resident_texture(&device, descriptor.guest)?;
                }
            }
        }
        self.synchronize_sampled_aliases(&[export_evaluation, pixel_evaluation])?;

        let export_buffers = self.upload_global_buffers(&device, export_evaluation)?;
        let pixel_buffers = self.upload_global_buffers(&device, pixel_evaluation)?;
        let export_textures = upload_image_textures(
            &device,
            &queue,
            export_evaluation,
            &mut self.resident_guest_textures,
            &self.multisample_textures,
            &mut self.sampled_texture_cache,
            &mut self.sampled_texture_write_watches,
            &mut self.fallback_texture_cache,
            &mut self.sampled_texture_admission_order,
            &mut self.next_sampled_texture_admission_id,
        )?;
        let pixel_textures = upload_image_textures(
            &device,
            &queue,
            pixel_evaluation,
            &mut self.resident_guest_textures,
            &self.multisample_textures,
            &mut self.sampled_texture_cache,
            &mut self.sampled_texture_write_watches,
            &mut self.fallback_texture_cache,
            &mut self.sampled_texture_admission_order,
            &mut self.next_sampled_texture_admission_id,
        )?;
        let (
            pipeline,
            vertex_function,
            pixel_function,
            vertex_scalar_buffer_index,
            pixel_scalar_buffer_index,
            vertex_sampler_buffer_index,
            pixel_sampler_buffer_index,
        ) = {
            let compiled = self
                .compiled_shader_pairs
                .get(key)
                .ok_or("shader pair has no compiled Metal pipeline")?;
            if let Some(directory) = std::env::var_os("REMU_DUMP_SCANOUT_MSL_DIR")
                && (targets.iter().any(|target| {
                    self.scanout_snapshot_textures
                        .contains_key(&target.texture.address)
                }) || diagnostic_dump_target_pixel_shader() == Some(draw.pixel_shader_address))
            {
                static DUMPED_SCANOUT_PAIRS: OnceLock<Mutex<HashSet<(u64, u64)>>> = OnceLock::new();
                let pair = (draw.export_shader_address, draw.pixel_shader_address);
                let mut dumped = DUMPED_SCANOUT_PAIRS
                    .get_or_init(|| Mutex::new(HashSet::new()))
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if dumped.len() < 128 && dumped.insert(pair) {
                    let sampled = sampled_image_bindings(pixel_evaluation)
                        .map(|binding| {
                            let descriptor =
                                decode_image_descriptor(&binding.resource_descriptor).ok();
                            (
                                binding.pc,
                                binding.opcode,
                                descriptor.map(|descriptor| descriptor.guest),
                                binding.sampler_descriptor.clone(),
                            )
                        })
                        .collect::<Vec<_>>();
                    eprintln!(
                        "[REMu][METAL][SCANOUT] draw={} es={:#x} ps={:#x} targets={targets:?} sampled={sampled:?}",
                        draw.sequence, draw.export_shader_address, draw.pixel_shader_address,
                    );
                    if let Err(error) = dump_shader_pair_msl(
                        Path::new(&directory),
                        draw.export_shader_address,
                        draw.pixel_shader_address,
                        &compiled.vertex.source,
                        &compiled.pixel.source,
                    ) {
                        eprintln!(
                            "[REMu][METAL] could not dump scanout MSL for es={:#x} ps={:#x}: {error}",
                            draw.export_shader_address, draw.pixel_shader_address,
                        );
                    }
                }
            }
            (
                compiled.pipeline.clone(),
                compiled.vertex_function.clone(),
                compiled.pixel_function.clone(),
                compiled.vertex.bindings.initial_scalar_buffer_index,
                compiled.pixel.bindings.initial_scalar_buffer_index,
                compiled.vertex.bindings.sampler_argument_buffer_index,
                compiled.pixel.bindings.sampler_argument_buffer_index,
            )
        };
        let (export_sampler_buffer, _export_samplers) = self.cached_sampler_argument_buffer(
            &device,
            &vertex_function,
            vertex_sampler_buffer_index,
            export_evaluation,
        )?;
        let (pixel_sampler_buffer, _pixel_samplers) = self.cached_sampler_argument_buffer(
            &device,
            &pixel_function,
            pixel_sampler_buffer_index,
            pixel_evaluation,
        )?;
        let indexed_draw = if draw.indexed {
            let indices = read_draw_indices(draw).ok_or("index buffer is unreadable")?;
            let buffer = self.cached_index_buffer(&device, indices.as_ref())?;
            Some((indices, buffer))
        } else {
            None
        };
        let primary_target = targets
            .first()
            .expect("render-target decoder requires at least one target")
            .texture;
        // Vulkan framebuffers use the shared minimum attachment extent. Unreal relies on that
        // behavior for passes that bind a small hierarchical depth surface beside a larger color
        // target. Metal exposes the same explicit render-target extent on the pass descriptor.
        let render_width = targets
            .iter()
            .map(|target| (target.texture.width >> target.mip_level).max(1))
            .chain(depth.map(|depth| depth.texture.width))
            .min()
            .unwrap_or(primary_target.width);
        let render_height = targets
            .iter()
            .map(|target| (target.texture.height >> target.mip_level).max(1))
            .chain(depth.map(|depth| depth.texture.height))
            .min()
            .unwrap_or(primary_target.height);
        let scissor = decode_scissor(draw, render_width, render_height)?;
        let diagnostic_force_triangle = diagnostic_disable_combined_raster
            && diagnostic_complex_msaa_raster
            && diagnostic_force_msaa_vertex_pattern();
        let viewport = if diagnostic_force_triangle {
            MTLViewport {
                originX: 0.0,
                originY: 0.0,
                width: f64::from(render_width),
                height: f64::from(render_height),
                znear: 0.0,
                zfar: 1.0,
            }
        } else {
            decode_viewport(draw, render_width, render_height, scissor)
        };
        if trace_image_descriptors() && evaluation_samples_large_linear_video(pixel_evaluation) {
            static LOGGED_VIDEO_DRAWS: AtomicUsize = AtomicUsize::new(0);
            if LOGGED_VIDEO_DRAWS.fetch_add(1, Ordering::Relaxed) < 16 {
                let indices = indexed_draw.as_ref().map(|(indices, _)| indices.as_ref());
                eprintln!(
                    "[REMu][METAL][VIDEO] draw={} es={:#x} ps={:#x} indexed={} index_size={} index_offset={} index_buffer={:#x} indices={indices:?} count={} primitive={:#x} targets={targets:?} depth={depth:?} raster={raster:?} scissor={scissor:?} viewport=({:.1},{:.1} {:.1}x{:.1} {:.3}..{:.3}) depth_control={:#010x} z_info={:#010x} depth_size={:#010x} raster_control={:#010x}",
                    draw.sequence,
                    draw.export_shader_address,
                    draw.pixel_shader_address,
                    draw.indexed,
                    draw.index_size,
                    draw.index_offset,
                    draw.index_buffer_address,
                    draw.index_count,
                    draw.primitive_type,
                    viewport.originX,
                    viewport.originY,
                    viewport.width,
                    viewport.height,
                    viewport.znear,
                    viewport.zfar,
                    register_value(&draw.context_registers, DB_DEPTH_CONTROL).unwrap_or_default(),
                    register_value(&draw.context_registers, DB_Z_INFO).unwrap_or_default(),
                    register_value(&draw.context_registers, DB_DEPTH_SIZE_XY).unwrap_or_default(),
                    register_value(&draw.context_registers, PA_SU_SC_MODE_CNTL).unwrap_or(1 << 2),
                );
                let global_bindings = export_evaluation
                    .global_memory_bindings
                    .iter()
                    .map(|binding| {
                        (
                            binding.scalar_address,
                            binding.base_address,
                            binding.byte_count,
                            binding.instruction_pcs.as_slice(),
                        )
                    })
                    .collect::<Vec<_>>();
                eprintln!(
                    "[REMu][METAL][VIDEO] export initial_scalars={} globals={global_bindings:x?} vertex_inputs={:?}",
                    export_evaluation.initial_scalar_registers.len(),
                    export_evaluation.vertex_inputs,
                );
            }
        }
        let primitive = metal_primitive_type(draw.primitive_type)?;
        let export_buffer_indices = (0..export_buffers.len())
            .map(|index| u64::try_from(index).map_err(|_| "vertex buffer index overflow"))
            .collect::<Result<Vec<_>, _>>()?;
        let pixel_buffer_indices = (0..pixel_buffers.len())
            .map(|index| u64::try_from(index).map_err(|_| "pixel buffer index overflow"))
            .collect::<Result<Vec<_>, _>>()?;
        let export_texture_indices = (0..export_textures.len())
            .map(|index| u64::try_from(index).map_err(|_| "vertex texture index overflow"))
            .collect::<Result<Vec<_>, _>>()?;
        let pixel_texture_indices = (0..pixel_textures.len())
            .map(|index| u64::try_from(index).map_err(|_| "pixel texture index overflow"))
            .collect::<Result<Vec<_>, _>>()?;
        let export_user_data_register = [0x08c, 0x0cc, 0x04c]
            .into_iter()
            .find(|base| {
                draw.shader_registers
                    .iter()
                    .any(|register| register.offset == base - 1)
            })
            .unwrap_or(0x0cc);
        let mut vertex_scalar_metadata = shader_scalar_metadata_for_registers(
            export_evaluation,
            &draw.shader_registers,
            export_user_data_register,
            8,
        )?;
        if diagnostic_use_draw_merged_wave_counts() && vertex_scalar_metadata.len() > 3 {
            let vertex_threads = if let Some((indices, _)) = indexed_draw.as_ref() {
                let mut unique = HashSet::with_capacity(indices.len().min(64));
                for &index in indices.iter() {
                    unique.insert(index);
                    if unique.len() == 64 {
                        break;
                    }
                }
                unique.len()
            } else {
                usize::try_from(metal_nonindexed_vertex_count(
                    draw.primitive_type,
                    draw.index_count,
                ))
                .unwrap_or(64)
                .min(64)
            };
            let element_count = indexed_draw.as_ref().map_or_else(
                || usize::try_from(draw.index_count).unwrap_or(64),
                |(indices, _)| indices.len(),
            );
            let primitive_threads = match primitive {
                MTLPrimitiveType::Point => element_count,
                MTLPrimitiveType::Line => element_count / 2,
                MTLPrimitiveType::LineStrip => element_count.saturating_sub(1),
                MTLPrimitiveType::Triangle => element_count / 3,
                MTLPrimitiveType::TriangleStrip => element_count.saturating_sub(2),
            }
            .clamp(1, 64);
            let vertex_threads = vertex_threads.clamp(1, 64);
            vertex_scalar_metadata[3] = u32::try_from(vertex_threads).expect("wave count fits u32")
                | (u32::try_from(primitive_threads).expect("wave count fits u32") << 8);
        }
        let pixel_scalar_metadata = shader_scalar_metadata_for_registers(
            pixel_evaluation,
            &draw.shader_registers,
            0x00c,
            0,
        )?;
        let vertex_scalar_byte_count = vertex_scalar_metadata
            .len()
            .checked_mul(4)
            .and_then(|count| u64::try_from(count).ok())
            .ok_or("vertex scalar byte count overflow")?;
        let pixel_scalar_byte_count = pixel_scalar_metadata
            .len()
            .checked_mul(4)
            .and_then(|count| u64::try_from(count).ok())
            .ok_or("pixel scalar byte count overflow")?;
        let indexed_count = indexed_draw
            .as_ref()
            .map(|(indices, _)| u64::try_from(indices.len()).map_err(|_| "index count overflow"))
            .transpose()?;
        let mut pending_writebacks = Vec::new();
        collect_pending_global_writebacks(
            &export_buffers,
            export_evaluation,
            &mut pending_writebacks,
        )?;
        collect_pending_global_writebacks(
            &pixel_buffers,
            pixel_evaluation,
            &mut pending_writebacks,
        )?;
        let mut pending_guest_write_ranges = Vec::new();
        collect_pending_guest_write_ranges(
            &export_buffers,
            export_evaluation,
            &mut pending_guest_write_ranges,
        )?;
        collect_pending_guest_write_ranges(
            &pixel_buffers,
            pixel_evaluation,
            &mut pending_guest_write_ranges,
        )?;
        let depth_stencil_state =
            depth.map(|depth| self.cached_depth_stencil_state(&device, depth));
        let diagnostic_vertex_position_buffer = if diagnostic_capture_vertex_position() {
            if self.diagnostic_vertex_position_buffer.is_none() {
                let byte_count = DIAGNOSTIC_VERTEX_POSITION_VERTEX_CAPACITY
                    .checked_mul(4)
                    .and_then(|words| words.checked_mul(std::mem::size_of::<u32>()))
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .ok_or("diagnostic vertex-position byte count overflow")?;
                let buffer = device.new_buffer(byte_count, MTLResourceOptions::StorageModeShared);
                buffer.set_label("REMu diagnostic vertex positions");
                self.diagnostic_vertex_position_buffer = Some(buffer);
            }
            self.diagnostic_vertex_position_buffer.clone()
        } else {
            None
        };
        let diagnostic_pixel_output_buffer = if diagnostic_capture_pixel_output() {
            if self.diagnostic_pixel_output_buffer.is_none() {
                let byte_count = DIAGNOSTIC_PIXEL_OUTPUT_WIDTH
                    .checked_mul(DIAGNOSTIC_PIXEL_OUTPUT_HEIGHT)
                    .and_then(|pixels| pixels.checked_mul(5))
                    .and_then(|words| words.checked_mul(std::mem::size_of::<u32>()))
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .ok_or("diagnostic pixel-output byte count overflow")?;
                let buffer = device.new_buffer(byte_count, MTLResourceOptions::StorageModeShared);
                buffer.set_label("REMu diagnostic pixel outputs");
                // SAFETY: the newly allocated shared buffer exposes exactly `byte_count` bytes.
                unsafe {
                    std::ptr::write_bytes(
                        buffer.contents().cast::<u8>(),
                        0,
                        usize::try_from(byte_count)
                            .expect("diagnostic pixel-output byte count fits usize"),
                    );
                }
                self.diagnostic_pixel_output_buffer = Some(buffer);
            }
            self.diagnostic_pixel_output_buffer.clone()
        } else {
            None
        };

        if self.open_guest_batch.is_none() {
            let command_buffer = queue.new_command_buffer().to_owned();
            command_buffer.set_label("REMu translated guest batch");
            self.open_guest_batch = Some(GuestCommandBatch {
                command_buffer,
                work: Vec::new(),
                global_writebacks: Vec::new(),
                guest_write_ranges: Vec::new(),
                memory_writes: Vec::new(),
                render_pass: None,
                completion_sequences: Vec::new(),
            });
        }
        let render_pass_key = GuestRenderPassKey {
            targets: targets
                .iter()
                .map(|target| {
                    (
                        target.texture,
                        target.sample_count,
                        target.direct_scanout,
                        target.texture_type,
                        target.layer_count,
                        target.base_layer,
                        target.mip_level,
                        target.mip_level_count,
                    )
                })
                .collect(),
            depth: depth.map(|depth| (depth.texture, depth.sample_count)),
            resolve_multisample_targets,
        };
        let needs_new_render_pass = self
            .open_guest_batch
            .as_ref()
            .and_then(|batch| batch.render_pass.as_ref())
            .is_none_or(|open| open.key != render_pass_key);
        let pass = if needs_new_render_pass {
            let pass = RenderPassDescriptor::new();
            pass.set_render_target_width(u64::from(render_width));
            pass.set_render_target_height(u64::from(render_height));
            pass.set_render_target_array_length(u64::from(key.render_target_array_length));
            for (location, ((texture, resolve_texture), clear_uninitialized)) in target_textures
                .iter()
                .zip(&target_resolve_textures)
                .zip(&clear_uninitialized_targets)
                .enumerate()
            {
                let attachment = pass
                    .color_attachments()
                    .object_at(u64::try_from(location).map_err(|_| "color location overflow")?)
                    .ok_or_else(|| {
                        format!("Metal render pass has no color attachment {location}")
                    })?;
                attachment.set_texture(Some(texture));
                attachment.set_level(u64::from(targets[location].mip_level));
                if targets[location].texture_type == 10 {
                    attachment.set_depth_plane(u64::from(targets[location].base_layer));
                } else {
                    attachment.set_slice(u64::from(targets[location].base_layer));
                }
                if *clear_uninitialized {
                    attachment.set_load_action(MTLLoadAction::Clear);
                    attachment.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));
                } else {
                    attachment.set_load_action(MTLLoadAction::Load);
                }
                if let Some(resolve_texture) = resolve_texture {
                    attachment.set_resolve_texture(Some(resolve_texture));
                    attachment.set_store_action(MTLStoreAction::StoreAndMultisampleResolve);
                } else {
                    attachment.set_store_action(MTLStoreAction::Store);
                }
            }
            if let (Some(depth), Some(texture)) = (depth, depth_texture.as_ref()) {
                let attachment = pass
                    .depth_attachment()
                    .ok_or("Metal render pass has no depth attachment")?;
                attachment.set_texture(Some(texture));
                if clear_uninitialized_depth {
                    attachment.set_load_action(MTLLoadAction::Clear);
                    attachment.set_clear_depth(initial_depth_clear_value(depth.compare));
                } else {
                    attachment.set_load_action(MTLLoadAction::Load);
                }
                if let Some(resolve_texture) = depth_resolve_texture.as_ref() {
                    attachment.set_resolve_texture(Some(resolve_texture));
                    attachment.set_store_action(MTLStoreAction::StoreAndMultisampleResolve);
                } else {
                    attachment.set_store_action(MTLStoreAction::Store);
                }
            }
            Some(pass)
        } else {
            None
        };
        {
            let batch = self
                .open_guest_batch
                .as_mut()
                .expect("guest command batch was opened above");
            batch.work.push(GuestBatchWork::Draw {
                export: key.export_shader_address,
                pixel: key.pixel_shader_address,
            });
            append_unique_global_writebacks(&mut batch.global_writebacks, pending_writebacks);
            append_unique_guest_write_ranges(
                &mut batch.guest_write_ranges,
                pending_guest_write_ranges,
            );
            if needs_new_render_pass {
                if batch.end_render_pass() {
                    self.statistics.render_pass_ends_for_target_change = self
                        .statistics
                        .render_pass_ends_for_target_change
                        .saturating_add(1);
                }
                let encoder = batch
                    .command_buffer
                    .new_render_command_encoder(
                        pass.expect("new guest render pass descriptor was prepared"),
                    )
                    .to_owned();
                encoder.set_label("REMu translated guest render pass");
                batch.render_pass = Some(OpenGuestRenderPass {
                    key: render_pass_key,
                    encoder,
                });
                self.statistics.encoded_render_passes =
                    self.statistics.encoded_render_passes.saturating_add(1);
            }
            let encoder = &batch
                .render_pass
                .as_ref()
                .expect("compatible guest render pass was opened above")
                .encoder;
            encoder.set_render_pipeline_state(&pipeline);
            encoder.set_front_facing_winding(raster.front_winding);
            encoder.set_cull_mode(if diagnostic_disable_culling_for_draw {
                MTLCullMode::None
            } else {
                raster.cull_mode
            });
            encoder.set_triangle_fill_mode(raster.fill_mode);
            if !diagnostic_disable_depth_for_draw
                && let Some(depth_stencil_state) = depth_stencil_state.as_ref()
            {
                encoder.set_depth_stencil_state(depth_stencil_state);
            }
            encoder.set_blend_color(
                f32::from_bits(
                    register_value(&draw.context_registers, CB_BLEND_RED).unwrap_or_default(),
                ),
                f32::from_bits(
                    register_value(&draw.context_registers, CB_BLEND_GREEN).unwrap_or_default(),
                ),
                f32::from_bits(
                    register_value(&draw.context_registers, CB_BLEND_BLUE).unwrap_or_default(),
                ),
                f32::from_bits(
                    register_value(&draw.context_registers, CB_BLEND_ALPHA).unwrap_or_default(),
                ),
            );
            encoder.set_viewport(viewport);
            if !diagnostic_force_triangle && let GuestScissor::Rect(rect) = scissor {
                encoder.set_scissor_rect(rect);
            }
            for (index, buffer) in export_buffer_indices.iter().zip(&export_buffers) {
                encoder.set_vertex_buffer(*index, Some(&buffer.buffer), buffer.offset);
            }
            for (index, buffer) in pixel_buffer_indices.iter().zip(&pixel_buffers) {
                encoder.set_fragment_buffer(*index, Some(&buffer.buffer), buffer.offset);
            }
            encoder.set_vertex_bytes(
                u64::from(vertex_scalar_buffer_index),
                vertex_scalar_byte_count,
                vertex_scalar_metadata.as_ptr().cast::<c_void>(),
            );
            if let Some(buffer) = diagnostic_vertex_position_buffer.as_ref() {
                encoder.set_vertex_buffer(DIAGNOSTIC_VERTEX_POSITION_BUFFER_INDEX, Some(buffer), 0);
            }
            encoder.set_fragment_bytes(
                u64::from(pixel_scalar_buffer_index),
                pixel_scalar_byte_count,
                pixel_scalar_metadata.as_ptr().cast::<c_void>(),
            );
            if let Some(buffer) = diagnostic_pixel_output_buffer.as_ref() {
                encoder.set_fragment_buffer(DIAGNOSTIC_PIXEL_OUTPUT_BUFFER_INDEX, Some(buffer), 0);
            }
            for (index, texture) in export_texture_indices.iter().zip(&export_textures) {
                encoder.set_vertex_texture(*index, Some(&texture.texture));
            }
            for (index, texture) in pixel_texture_indices.iter().zip(&pixel_textures) {
                encoder.set_fragment_texture(*index, Some(&texture.texture));
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

            if diagnostic_force_triangle {
                encoder.draw_primitives_instanced(MTLPrimitiveType::Triangle, 0, 3, 1);
            } else if matches!(scissor, GuestScissor::Empty) {
                // Metal rejects an empty scissor, so a guest empty-scissor draw encodes no
                // primitive while preserving command-stream ordering.
            } else if let Some((_, index_buffer)) = indexed_draw.as_ref() {
                encoder.draw_indexed_primitives_instanced(
                    primitive,
                    indexed_count.expect("indexed draw count was precomputed"),
                    MTLIndexType::UInt32,
                    index_buffer,
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
        }

        for target in &targets {
            if target.direct_scanout {
                self.mark_direct_scanout_initialized(target.texture.address);
            } else {
                if target.sample_count > 1 {
                    self.mark_multisample_texture_initialized(target.texture, target.sample_count);
                    if resolve_multisample_targets {
                        self.mark_resident_texture_written_preserving_multisample(target.texture);
                    }
                } else {
                    self.mark_resident_texture_written(target.texture);
                }
            }
            // A flip can arrive before the presenter has translated the render target that
            // backs it, and the translated texture can later be replaced by an overlapping
            // reinterpretation. Always refresh the live scanout after a matching target write.
            self.refresh_pending_flip_after_target_write(*target);
        }
        let dump_target_before_pixel_shader =
            diagnostic_dump_target_before_pixel_shader().is_some();
        let trace_scanout_writers = trace_scanout_writes();
        for target in &targets {
            if should_record_diagnostic_target_writer(
                dump_target_before_pixel_shader,
                trace_scanout_writers,
                self.is_diagnostic_scanout_target(target),
            ) {
                record_diagnostic_scanout_writer(
                    target.texture.address,
                    DiagnosticScanoutWriter {
                        draw: draw.sequence,
                        export_shader: draw.export_shader_address,
                        pixel_shader: draw.pixel_shader_address,
                    },
                );
            }
        }
        if let Some(depth) = depth {
            if depth.sample_count > 1 {
                self.mark_multisample_texture_initialized(depth.texture, depth.sample_count);
            }
            if depth.write_enabled {
                if depth.sample_count > 1 {
                    self.mark_resident_texture_written_preserving_multisample(depth.texture);
                } else {
                    self.mark_resident_texture_written(depth.texture);
                }
            }
        }
        for texture in export_textures.iter().chain(&pixel_textures) {
            if texture.writable {
                self.mark_resident_texture_written(texture.descriptor.guest);
            }
        }
        let diagnostic_target = diagnostic_dump_target_pixel_shader()
            .filter(|&pixel| pixel == draw.pixel_shader_address)
            .and_then(|_| targets.first())
            .or_else(|| {
                diagnostic_dump_target_pixel_instruction_count()
                    .filter(|&count| {
                        draw.pixel_shader
                            .as_ref()
                            .is_some_and(|shader| shader.program.instructions.len() == count)
                    })
                    .and_then(|_| targets.first())
            })
            .or_else(|| {
                (diagnostic_dump_complex_msaa()
                    && draw
                        .pixel_shader
                        .as_ref()
                        .is_some_and(|shader| shader.program.instructions.len() == 370))
                .then(|| targets.iter().find(|target| target.sample_count > 1))
                .flatten()
            })
            .or_else(|| {
                let address = diagnostic_dump_target_address()?;
                targets.iter().find(|target| {
                    if address == 0 {
                        target.texture.width == 1_920 && target.texture.height == 1_080
                    } else {
                        target.texture.address == address
                    }
                })
            })
            .or_else(|| {
                (diagnostic_dump_video_draw()
                    && evaluation_samples_large_linear_video(pixel_evaluation))
                .then(|| targets.first())
                .flatten()
            });
        if draw.sequence >= diagnostic_dump_target_minimum_draw()
            && let Some(target) = diagnostic_target
        {
            static TARGET_MATCHES: AtomicUsize = AtomicUsize::new(0);
            let target_match = TARGET_MATCHES.fetch_add(1, Ordering::Relaxed);
            static DUMPED: AtomicBool = AtomicBool::new(false);
            if target_match == diagnostic_dump_target_match_index()
                && DUMPED
                    .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            {
                self.wait_for_guest_command_batches()?;
                let descriptor = target.texture;
                let texture = if target.direct_scanout {
                    self.direct_scanout_textures
                        .get(&descriptor.address)
                        .ok_or("diagnostic target has no direct scanout Metal texture")?
                        .texture
                        .clone()
                } else {
                    self.resident_guest_textures
                        .get(&descriptor)
                        .ok_or("diagnostic target has no resident Metal texture")?
                        .texture
                        .clone()
                };
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
                let bytes = read_texture_linear(&device, &queue, &texture, descriptor)?;
                let nonzero = bytes.iter().filter(|&&value| value != 0).count();
                let fingerprint = guest_frame_fingerprint(&bytes);
                let sampled_summaries = pixel_evaluation
                    .image_bindings
                    .iter()
                    .zip(&pixel_textures)
                    .filter(|(binding, _)| {
                        binding.opcode.starts_with("ImageSample")
                            || binding.opcode.starts_with("ImageGather")
                    })
                    .filter_map(|(binding, uploaded)| {
                        // A zero-copy component-reordered alias deliberately keeps its parent's
                        // Metal pixel format and applies R/B swizzles in the view. Read it back
                        // using that actual format; the guest sampling descriptor can differ.
                        let readback_descriptor = GuestTextureDescriptor {
                            pixel_format: uploaded.texture.pixel_format(),
                            bytes_per_pixel: metal_uncompressed_bytes_per_pixel(
                                uploaded.texture.pixel_format(),
                            )?,
                            ..uploaded.descriptor.guest
                        };
                        let sampled_bytes = match if uploaded.texture.texture_type()
                            == MTLTextureType::D3
                        {
                            read_texture_volume_linear(
                                &device,
                                &queue,
                                &uploaded.texture,
                                readback_descriptor,
                            )
                        } else {
                            read_texture_linear(
                                &device,
                                &queue,
                                &uploaded.texture,
                                readback_descriptor,
                            )
                        } {
                            Ok(bytes) => bytes,
                            Err(error) => {
                                eprintln!(
                                    "[REMu][METAL][TARGET-SAMPLE-READ] pc={:#x} opcode={} guest={:?} metal={}x{} format={:?} type={:?} error={error}",
                                    binding.pc,
                                    binding.opcode,
                                    uploaded.descriptor.guest,
                                    uploaded.texture.width(),
                                    uploaded.texture.height(),
                                    uploaded.texture.pixel_format(),
                                    uploaded.texture.texture_type(),
                                );
                                return None;
                            }
                        };
                        let sampled_fingerprint = guest_frame_fingerprint(&sampled_bytes);
                        if capture_target_sample_textures()
                            && let Some(directory) = self.capture_directory.as_deref()
                        {
                            let capture_descriptor =
                                if uploaded.texture.texture_type() == MTLTextureType::D3 {
                                    let Some(height) = readback_descriptor
                                        .height
                                        .checked_mul(
                                            u32::try_from(uploaded.texture.depth())
                                                .unwrap_or(u32::MAX),
                                        )
                                    else {
                                        eprintln!(
                                            "[REMu][METAL] could not flatten target 3D sample pc={:#x}: extent overflow",
                                            binding.pc
                                        );
                                        return None;
                                    };
                                    GuestTextureDescriptor {
                                        height,
                                        pitch: readback_descriptor.width,
                                        tile_mode: 0,
                                        ..readback_descriptor
                                    }
                                } else {
                                    readback_descriptor
                                };
                            match native_texture_capture_to_bgra(
                                &sampled_bytes,
                                capture_descriptor,
                            ) {
                                Some(bgra) => match capture_guest_frame(
                                    directory,
                                    &bgra,
                                    capture_descriptor,
                                    sampled_fingerprint,
                                ) {
                                    Ok(path) => eprintln!(
                                        "[REMu][METAL] captured target sample pc={:#x} {}",
                                        binding.pc,
                                        path.display()
                                    ),
                                    Err(error) => eprintln!(
                                        "[REMu][METAL] could not capture target sample pc={:#x}: {error}",
                                        binding.pc
                                    ),
                                },
                                None => eprintln!(
                                    "[REMu][METAL] could not convert target sample pc={:#x} format={:?} bytes={}",
                                    binding.pc,
                                    readback_descriptor.pixel_format,
                                    sampled_bytes.len(),
                                ),
                            }
                        }
                        let resident = self
                            .resident_guest_textures
                            .get(&uploaded.descriptor.guest)
                            .map(|resident| (resident.gpu_written, resident.content_version));
                        let aliases = self
                            .resident_guest_textures
                            .values()
                            .filter(|resident| {
                                guest_texture_descriptors_overlap(
                                    resident.descriptor,
                                    uploaded.descriptor.guest,
                                )
                            })
                            .map(|resident| {
                                (
                                    resident.descriptor,
                                    resident.gpu_written,
                                    resident.content_version,
                                )
                            })
                            .collect::<Vec<_>>();
                        let writer = diagnostic_texture_writer(uploaded.descriptor.guest);
                        let guest_raw = snapshot_guest_texture_raw(uploaded.descriptor.guest)
                            .ok()
                            .map(|raw| {
                                (
                                    raw.iter().filter(|&&value| value != 0).count(),
                                    guest_frame_fingerprint(&raw),
                                    raw.get(..raw.len().min(32)).unwrap_or(&[]).to_vec(),
                                )
                            });
                        let upstream = writer
                            .as_ref()
                            .map(|writer| {
                                writer
                                    .3
                                    .iter()
                                    .map(|descriptor| {
                                        let mut chain = Vec::new();
                                        let mut current = *descriptor;
                                        for _ in 0..12 {
                                            let resident = self
                                                .resident_guest_textures
                                                .get(&current)
                                                .map(|resident| {
                                                    let contents = read_texture_linear(
                                                        &device,
                                                        &queue,
                                                        &resident.texture,
                                                        current,
                                                    )
                                                    .ok();
                                                    (
                                                        resident.gpu_written,
                                                        resident.content_version,
                                                        contents.as_ref().map(|contents| {
                                                            contents
                                                                .iter()
                                                                .filter(|&&value| value != 0)
                                                                .count()
                                                        }),
                                                        contents.as_ref().map(|contents| {
                                                            guest_frame_fingerprint(contents)
                                                        }),
                                                    )
                                                });
                                            let cached = self
                                                .sampled_texture_cache
                                                .iter()
                                                .filter(|(key, _)| key.descriptor == current)
                                                .max_by_key(|(_, cached)| cached.admission_id)
                                                .and_then(|(_, cached)| {
                                                    let contents = read_texture_linear(
                                                        &device,
                                                        &queue,
                                                        &cached.texture,
                                                        current,
                                                    )
                                                    .ok()?;
                                                    Some((
                                                        contents
                                                            .iter()
                                                            .filter(|&&value| value != 0)
                                                            .count(),
                                                        guest_frame_fingerprint(&contents),
                                                    ))
                                                });
                                            let aliases = self
                                                .resident_guest_textures
                                                .values()
                                                .filter(|resident| {
                                                    guest_texture_descriptors_overlap(
                                                        resident.descriptor,
                                                        current,
                                                    )
                                                })
                                                .map(|resident| {
                                                    (
                                                        resident.descriptor,
                                                        resident.gpu_written,
                                                        resident.content_version,
                                                    )
                                                })
                                                .collect::<Vec<_>>();
                                            let current_writer = diagnostic_texture_writer(current);
                                            let overlapping_writers =
                                                diagnostic_overlapping_texture_writers(current);
                                            let memory_copies =
                                                diagnostic_overlapping_memory_copies(current);
                                            chain.push((
                                                current,
                                                resident,
                                                cached,
                                                aliases,
                                                current_writer.clone(),
                                                overlapping_writers,
                                                memory_copies,
                                            ));
                                            let Some(next) = current_writer
                                                .as_ref()
                                                .and_then(|writer| writer.3.first())
                                                .copied()
                                            else {
                                                break;
                                            };
                                            if chain.iter().any(|(descriptor, _, _, _, _, _, _)| {
                                                *descriptor == next
                                            }) {
                                                break;
                                            }
                                            current = next;
                                        }
                                        chain
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        Some((
                            binding.pc,
                            uploaded.descriptor.guest,
                            resident,
                            writer,
                            diagnostic_overlapping_texture_writers(uploaded.descriptor.guest),
                            diagnostic_related_texture_writers(uploaded.descriptor.guest),
                            (
                                diagnostic_overlapping_memory_copies(uploaded.descriptor.guest),
                                diagnostic_overlapping_global_writers(uploaded.descriptor.guest),
                            ),
                            (guest_raw, upstream),
                            aliases,
                            sampled_bytes.iter().filter(|&&value| value != 0).count(),
                            sampled_fingerprint,
                            sampled_bytes
                                .get(..sampled_bytes.len().min(32))
                                .unwrap_or(&[])
                                .to_vec(),
                        ))
                    })
                    .collect::<Vec<_>>();
                let rgba8_summary = if descriptor.pixel_format == MTLPixelFormat::RGBA8Unorm {
                    let first = bytes.get(..4).unwrap_or(&[]);
                    let mut minimum = [u8::MAX; 4];
                    let mut maximum = [u8::MIN; 4];
                    let mut different = 0_usize;
                    for pixel in bytes.chunks_exact(4) {
                        if pixel != first {
                            different += 1;
                        }
                        for component in 0..4 {
                            minimum[component] = minimum[component].min(pixel[component]);
                            maximum[component] = maximum[component].max(pixel[component]);
                        }
                    }
                    format!(
                        " first={first:02x?} min={minimum:02x?} max={maximum:02x?} different={different}"
                    )
                } else if descriptor.pixel_format == MTLPixelFormat::RGBA16Float {
                    const MAGENTA_RGBA16F: [u8; 8] = [0x00, 0x3c, 0, 0, 0x00, 0x3c, 0x00, 0x3c];
                    const WHITE_RGBA16F: [u8; 8] = [0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c];
                    let first = bytes.get(..8).unwrap_or(&[]);
                    let magenta = bytes
                        .chunks_exact(8)
                        .filter(|pixel| *pixel == MAGENTA_RGBA16F)
                        .count();
                    let white = bytes
                        .chunks_exact(8)
                        .filter(|pixel| *pixel == WHITE_RGBA16F)
                        .count();
                    let zero = bytes
                        .chunks_exact(8)
                        .filter(|pixel| pixel.iter().all(|&component| component == 0))
                        .count();
                    let different = bytes
                        .chunks_exact(8)
                        .filter(|pixel| *pixel != first)
                        .count();
                    format!(
                        " first={first:02x?} magenta={magenta} white={white} zero={zero} different={different}"
                    )
                } else {
                    String::new()
                };
                let diagnostic_vertex_positions = self
                    .diagnostic_vertex_position_buffer
                    .as_ref()
                    .and_then(|buffer| {
                        let indices = indexed_draw.as_ref()?.0.as_ref();
                        let word_count =
                            DIAGNOSTIC_VERTEX_POSITION_VERTEX_CAPACITY.checked_mul(4)?;
                        // SAFETY: the diagnostic buffer is allocated above with exactly this many
                        // shared u32 words and the completed command batch is waited before readback.
                        let words = unsafe {
                            std::slice::from_raw_parts(buffer.contents().cast::<u32>(), word_count)
                        };
                        let mut seen = HashSet::new();
                        Some(
                            indices
                                .iter()
                                .copied()
                                .filter(|index| {
                                    usize::try_from(*index).is_ok_and(|index| {
                                        index < DIAGNOSTIC_VERTEX_POSITION_VERTEX_CAPACITY
                                    }) && seen.insert(*index)
                                })
                                .take(24)
                                .map(|index| {
                                    let base = usize::try_from(index)
                                        .expect("filtered diagnostic vertex index fits usize")
                                        * 4;
                                    let raw = [
                                        words[base],
                                        words[base + 1],
                                        words[base + 2],
                                        words[base + 3],
                                    ];
                                    let value = raw.map(f32::from_bits);
                                    (index, raw, value)
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .unwrap_or_default();
                let diagnostic_pixel_outputs = self
                    .diagnostic_pixel_output_buffer
                    .as_ref()
                    .and_then(|buffer| {
                        let word_count = DIAGNOSTIC_PIXEL_OUTPUT_WIDTH
                            .checked_mul(DIAGNOSTIC_PIXEL_OUTPUT_HEIGHT)?
                            .checked_mul(5)?;
                        // SAFETY: the shared diagnostic buffer is allocated with exactly this
                        // many words and the command batch has completed before this readback.
                        let words = unsafe {
                            std::slice::from_raw_parts(buffer.contents().cast::<u32>(), word_count)
                        };
                        let covered = words.chunks_exact(5).filter(|pixel| pixel[0] != 0).count();
                        let nonzero = words
                            .chunks_exact(5)
                            .filter(|pixel| pixel[0] != 0 && pixel[1..].iter().any(|&v| v != 0))
                            .count();
                        let first = words
                            .chunks_exact(5)
                            .enumerate()
                            .filter(|(_, pixel)| pixel[0] != 0)
                            .take(8)
                            .map(|(pixel, words)| {
                                let raw = [words[1], words[2], words[3], words[4]];
                                let value = raw.map(f32::from_bits);
                                (
                                    pixel % DIAGNOSTIC_PIXEL_OUTPUT_WIDTH,
                                    pixel / DIAGNOSTIC_PIXEL_OUTPUT_WIDTH,
                                    raw,
                                    value,
                                )
                            })
                            .collect::<Vec<_>>();
                        Some((covered, nonzero, first))
                    });
                eprintln!(
                    "[REMu][METAL][TARGET] draw={} es={:#x} ps={:#x} indexed={} count={} instances={} primitive={:#x} indices={:?} address={:#x} {}x{} format={:?} bytes={} nonzero={} fingerprint={fingerprint:016x}{rgba8_summary} pipeline_targets={:?} ps_inputs={:08x?} vertex_positions={diagnostic_vertex_positions:?} pixel_outputs={diagnostic_pixel_outputs:?} sampled={sampled_summaries:x?} vertex_scalars={:08x?} pixel_scalars={:08x?} vertex_globals={:x?} pixel_globals={:x?}",
                    draw.sequence,
                    draw.export_shader_address,
                    draw.pixel_shader_address,
                    draw.indexed,
                    draw.index_count,
                    draw.instance_count,
                    draw.primitive_type,
                    indexed_draw
                        .as_ref()
                        .map(|(indices, _)| &indices[..indices.len().min(24)]),
                    descriptor.address,
                    descriptor.width,
                    descriptor.height,
                    descriptor.pixel_format,
                    bytes.len(),
                    nonzero,
                    key.targets,
                    key.pixel_input_controls,
                    vertex_scalar_metadata,
                    pixel_evaluation.initial_scalar_registers,
                    export_evaluation
                        .global_memory_bindings
                        .iter()
                        .map(|binding| (
                            binding.scalar_address,
                            binding.base_address,
                            binding.byte_count,
                            diagnostic_global_binding_head(binding),
                        ))
                        .collect::<Vec<_>>(),
                    pixel_evaluation
                        .global_memory_bindings
                        .iter()
                        .map(|binding| (
                            binding.scalar_address,
                            binding.base_address,
                            binding.byte_count,
                            diagnostic_global_binding_head(binding),
                        ))
                        .collect::<Vec<_>>(),
                );
                // A requested one-shot target dump must not be suppressed by the periodic
                // flip-capture interval. It is commonly used to inspect two adjacent draws in
                // the same frame, where the normal capture throttle intentionally keeps only one.
                self.last_guest_capture = None;
                self.capture_translated_frame_if_changed(descriptor, &bytes);
            }
        }
        if diagnostic_trace_target_contents()
            && draw.sequence >= diagnostic_dump_target_minimum_draw()
            && draw.sequence <= diagnostic_trace_target_maximum_draw()
            && let Some(target) = targets.iter().find(|target| {
                target.sample_count > 1
                    && target.texture.width == 1_920
                    && target.texture.height == 1_080
                    && target.texture.pixel_format == MTLPixelFormat::RGBA16Float
            })
        {
            self.wait_for_guest_command_batches()?;
            let descriptor = target.texture;
            let texture = self
                .resident_guest_textures
                .get(&descriptor)
                .ok_or("traced target has no resident Metal texture")?
                .texture
                .clone();
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
            let bytes = read_texture_linear(&device, &queue, &texture, descriptor)?;
            const WHITE_RGBA16F: [u8; 8] = [0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c, 0x00, 0x3c];
            let pixels = bytes.chunks_exact(8);
            let pixel_count = pixels.len();
            let zero = pixels
                .clone()
                .filter(|pixel| pixel.iter().all(|&component| component == 0))
                .count();
            let white = pixels
                .clone()
                .filter(|pixel| *pixel == WHITE_RGBA16F)
                .count();
            let class = if zero == pixel_count {
                0_u8
            } else if white == pixel_count {
                2_u8
            } else {
                1_u8
            };
            static LAST_CLASSES: OnceLock<Mutex<HashMap<u64, u8>>> = OnceLock::new();
            let changed = LAST_CLASSES
                .get_or_init(|| Mutex::new(HashMap::new()))
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(descriptor.address, class)
                != Some(class);
            if changed || diagnostic_trace_all_target_contents() {
                let class_name = match class {
                    0 => "zero",
                    2 => "white",
                    _ => "mixed",
                };
                eprintln!(
                    "[REMu][METAL][TARGET-CONTENT] draw={} ps={:#x} ps_insns={} count={} instances={} address={:#x} class={} zero={} white={} pixels={} fingerprint={:016x} mask={:#x} aa_mask={:08x}/{:08x} scissor={scissor:?} viewport=({:.1},{:.1} {:.1}x{:.1} {:.3}..{:.3}) depth={depth:?} raster={raster:?}",
                    draw.sequence,
                    draw.pixel_shader_address,
                    draw.pixel_shader
                        .as_ref()
                        .map_or(0, |shader| shader.program.instructions.len()),
                    draw.index_count,
                    draw.instance_count,
                    descriptor.address,
                    class_name,
                    zero,
                    white,
                    pixel_count,
                    guest_frame_fingerprint(&bytes),
                    target.write_mask,
                    register_value(&draw.context_registers, PA_SC_AA_MASK_X0Y0_X1Y0)
                        .unwrap_or(u32::MAX),
                    register_value(&draw.context_registers, PA_SC_AA_MASK_X0Y1_X1Y1)
                        .unwrap_or(u32::MAX),
                    viewport.originX,
                    viewport.originY,
                    viewport.width,
                    viewport.height,
                    viewport.znear,
                    viewport.zfar,
                );
            }
        }
        let batch_full = self
            .open_guest_batch
            .as_ref()
            .is_some_and(|batch| batch.work.len() >= GUEST_BATCH_WORK_CAPACITY);
        if batch_full {
            self.flush_guest_command_batch(false, GuestBatchFlushReason::WorkCapacity)?;
        }
        if let Some(directory) = std::env::var_os("REMU_DUMP_VIDEO_TARGET_DIR")
            && evaluation_samples_large_linear_video(pixel_evaluation)
        {
            self.wait_for_guest_command_batches()?;
            let descriptor = targets
                .first()
                .expect("render-target decoder requires at least one target")
                .texture;
            let texture = self
                .resident_guest_textures
                .get(&descriptor)
                .ok_or("video render target has no resident Metal texture")?
                .texture
                .clone();
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
            let bytes = read_texture_linear(&device, &queue, &texture, descriptor)?;
            let directory = Path::new(&directory);
            std::fs::create_dir_all(directory)
                .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
            let path = directory.join(format!(
                "draw-{}-target-{:x}-{}x{}-{:?}.bin",
                draw.sequence,
                descriptor.address,
                descriptor.width,
                descriptor.height,
                descriptor.pixel_format,
            ));
            std::fs::write(&path, &bytes)
                .map_err(|error| format!("could not write {}: {error}", path.display()))?;
            for (index, (_binding, uploaded)) in pixel_evaluation
                .image_bindings
                .iter()
                .zip(&pixel_textures)
                .enumerate()
                .filter(|(_, (binding, _))| {
                    binding.opcode.starts_with("ImageSample")
                        || binding.opcode.starts_with("ImageGather")
                })
            {
                let sampled_descriptor = uploaded.descriptor.guest;
                let sampled_bytes =
                    read_texture_linear(&device, &queue, &uploaded.texture, sampled_descriptor)?;
                let sampled_path = directory.join(format!(
                    "draw-{}-sample-{index}-{:x}-{}x{}-{:?}.bin",
                    draw.sequence,
                    sampled_descriptor.address,
                    sampled_descriptor.width,
                    sampled_descriptor.height,
                    sampled_descriptor.pixel_format,
                ));
                std::fs::write(&sampled_path, &sampled_bytes).map_err(|error| {
                    format!("could not write {}: {error}", sampled_path.display())
                })?;
                if let Ok(raw) = snapshot_guest_texture_raw(sampled_descriptor) {
                    let raw_path = directory.join(format!(
                        "draw-{}-sample-{index}-{:x}-guest-raw.bin",
                        draw.sequence, sampled_descriptor.address,
                    ));
                    std::fs::write(&raw_path, raw).map_err(|error| {
                        format!("could not write {}: {error}", raw_path.display())
                    })?;
                }
            }
            let scalar_path = directory.join(format!("draw-{}-export-scalars.txt", draw.sequence));
            std::fs::write(
                &scalar_path,
                format!("{:08x?}\n", export_evaluation.initial_scalar_registers),
            )
            .map_err(|error| format!("could not write {}: {error}", scalar_path.display()))?;
            for (index, (binding, uploaded)) in export_evaluation
                .global_memory_bindings
                .iter()
                .zip(&export_buffers)
                .enumerate()
                .filter(|(_, (binding, _))| binding.byte_count <= 4 * 1024 * 1024)
            {
                let contents = uploaded.buffer.contents().cast::<u8>();
                let offset = usize::try_from(uploaded.offset)
                    .map_err(|_| "video global-buffer offset does not fit usize")?;
                if contents.is_null() {
                    continue;
                }
                // SAFETY: every uploaded global buffer is shared CPU-visible storage and the
                // binding extent was validated when the buffer and its byte offset were created.
                let data =
                    unsafe { std::slice::from_raw_parts(contents.add(offset), binding.byte_count) };
                let binding_path = directory.join(format!(
                    "draw-{}-export-binding-{index}-{:x}-{}.bin",
                    draw.sequence, binding.base_address, binding.byte_count,
                ));
                std::fs::write(&binding_path, data).map_err(|error| {
                    format!("could not write {}: {error}", binding_path.display())
                })?;
            }
            eprintln!(
                "[REMu][METAL][VIDEO] dumped immediate target {} ({} bytes)",
                path.display(),
                bytes.len()
            );
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

    fn capture_scanout_before_pixel_shader(
        &mut self,
        draw: &IndexedDraw,
        targets: &[DecodedRenderTarget],
    ) -> Result<(), String> {
        let Some(pixel_shader) = diagnostic_dump_target_before_pixel_shader() else {
            return Ok(());
        };
        if draw.pixel_shader_address != pixel_shader
            || draw.sequence < diagnostic_dump_target_minimum_draw()
        {
            return Ok(());
        }
        let Some((descriptor, previous_writer, texture)) = targets.iter().find_map(|target| {
            let previous_writer = diagnostic_scanout_writer(target.texture.address)?;
            let texture = self.diagnostic_capture_texture(target)?;
            Some((target.texture, previous_writer, texture))
        }) else {
            return Ok(());
        };
        let Some(directory) = self.capture_directory.clone() else {
            eprintln!(
                "[REMu][METAL][PRE-PS-TARGET] draw={} ps={:#x} skipped: REMU_CAPTURE_DIR is unset",
                draw.sequence, draw.pixel_shader_address
            );
            return Ok(());
        };
        static DUMPED: AtomicBool = AtomicBool::new(false);
        if DUMPED
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
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
        let bytes = read_texture_linear(&device, &queue, &texture, descriptor)?;
        let fingerprint = guest_frame_fingerprint(&bytes);
        let nonzero = bytes.iter().filter(|&&value| value != 0).count();
        let Some(bgra) = native_texture_capture_to_bgra(&bytes, descriptor) else {
            return Err(format!(
                "cannot convert pre-pixel-shader target format {:?} to BGRA",
                descriptor.pixel_format
            ));
        };
        let path = capture_guest_frame(&directory, &bgra, descriptor, fingerprint)?;
        eprintln!(
            "[REMu][METAL][PRE-PS-TARGET] draw={} es={:#x} ps={:#x} address={:#x} {}x{} format={:?} bytes={} nonzero={} fingerprint={fingerprint:016x} previous={previous_writer:?} captured={}",
            draw.sequence,
            draw.export_shader_address,
            draw.pixel_shader_address,
            descriptor.address,
            descriptor.width,
            descriptor.height,
            descriptor.pixel_format,
            bytes.len(),
            nonzero,
            path.display(),
        );
        Ok(())
    }

    fn upload_pending_frame(&mut self) {
        let (Some(display), Some(flip)) = (self.display, self.pending_flip) else {
            return;
        };
        if flip.sequence == self.uploaded_flip_sequence {
            return;
        }
        match self.activate_translated_scanout(flip.address, display.width, display.height) {
            Ok(true) => {
                self.uploaded_flip_sequence = flip.sequence;
                return;
            }
            Ok(false) => {}
            Err(error) => {
                eprintln!("[REMu][METAL] translated scanout retry failed: {error}");
            }
        }
        // The VideoOut ABI uses zero for the console's tiled scanout and one for a linear
        // allocation. Raw guest-memory upload is valid only for the latter; translated render
        // targets take the texture path above and retain their architectural tile mode.
        if !raw_video_out_layout_is_linear(display.tiling_mode, display.pitch, display.width) {
            if !self.logged_unsupported_layout {
                eprintln!(
                    "[REMu][METAL] cannot upload display layout tiling={} pitch={} width={} yet",
                    display.tiling_mode, display.pitch, display.width
                );
                self.logged_unsupported_layout = true;
            }
            return;
        }
        let Some(device) = self.device.as_ref() else {
            return;
        };
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
                .is_none_or(|last| last.elapsed() >= capture_interval()))
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
            .is_some_and(|last| last.elapsed() < capture_interval())
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
                .is_some_and(|last| last.elapsed() < capture_interval())
        {
            return;
        }
        let pixel_count = u64::from(descriptor.width)
            .checked_mul(u64::from(descriptor.height))
            .and_then(|pixels| usize::try_from(pixels).ok());
        let Some(bgra_length) = pixel_count.and_then(|pixels| pixels.checked_mul(4)) else {
            return;
        };
        let Some(bgra) = native_texture_capture_to_bgra(native_bytes, descriptor) else {
            return;
        };
        debug_assert_eq!(bgra.len(), bgra_length);
        self.guest_bgra_bytes = bgra;
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
        // A resize can request a redraw before the guest has registered or flipped a display
        // buffer. Retire that request without acquiring a drawable: leaving it pending makes
        // Winit call this path every millisecond, needlessly presenting an empty surface while
        // the guest is still starting. The first real flip sets `redraw_pending` again.
        let (Some(display), Some(texture)) = (self.display, self.frame_texture.as_ref()) else {
            self.redraw_pending = false;
            return;
        };
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
        command_buffer.present_drawable(drawable);
        command_buffer.commit();
        self.pending_presentations.push_back(PendingPresentation {
            command_buffer,
            submitted_at: Instant::now(),
        });
        self.redraw_pending = false;
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn linear_unorm8_to_srgb_unorm8(linear: u8) -> u8 {
    static LUT: OnceLock<[u8; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        std::array::from_fn(|linear| {
            let linear =
                f64::from(u8::try_from(linear).expect("linear table index fits u8")) / 255.0;
            let encoded = if linear <= 0.003_130_8 {
                linear * 12.92
            } else {
                1.055 * linear.powf(1.0 / 2.4) - 0.055
            };
            // Both transfer-function branches map the unorm input into 0..=1, so rounding the
            // scaled result cannot produce a negative value or exceed u8.
            (encoded * 255.0).round() as u8
        })
    })[usize::from(linear)]
}

fn half_float_to_srgb_unorm8(bits: u16) -> u8 {
    linear_unorm8_to_srgb_unorm8(half_float_to_unorm8(bits))
}

fn unsigned_float_to_f32(bits: u32, mantissa_bits: u32) -> f32 {
    debug_assert!(matches!(mantissa_bits, 5 | 6));
    let mantissa_mask = (1_u32 << mantissa_bits) - 1;
    let exponent = (bits >> mantissa_bits) & 0x1f;
    let mantissa = bits & mantissa_mask;
    let float_bits = match exponent {
        0 if mantissa == 0 => 0,
        0 => {
            let leading_bit = 31 - mantissa.leading_zeros();
            let float_exponent = 113 + leading_bit - mantissa_bits;
            let float_mantissa = (mantissa ^ (1 << leading_bit)) << (23 - leading_bit);
            (float_exponent << 23) | float_mantissa
        }
        0x1f => 0x7f80_0000 | (mantissa << (23 - mantissa_bits)),
        _ => ((exponent + 112) << 23) | (mantissa << (23 - mantissa_bits)),
    };
    f32::from_bits(float_bits)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn float_to_unorm8(value: f32) -> u8 {
    if value.is_nan() || value <= 0.0 {
        0
    } else if value >= 1.0 {
        255
    } else {
        (value * 255.0).round() as u8
    }
}

fn unsigned_float_to_srgb_unorm8(bits: u32, mantissa_bits: u32) -> u8 {
    linear_unorm8_to_srgb_unorm8(float_to_unorm8(unsigned_float_to_f32(bits, mantissa_bits)))
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn half_float_to_unorm8(bits: u16) -> u8 {
    let sign = u32::from(bits >> 15);
    let exponent = u32::from((bits >> 10) & 0x1f);
    let significand = u32::from(bits & 0x03ff);
    let float_bits = match exponent {
        0 if significand == 0 => sign << 31,
        0 => {
            let leading_zeros = significand.leading_zeros() - 21;
            let normalized = (significand << leading_zeros) & 0x03ff;
            let exponent = 113_u32.saturating_sub(leading_zeros);
            (sign << 31) | (exponent << 23) | (normalized << 13)
        }
        0x1f => (sign << 31) | 0x7f80_0000 | (significand << 13),
        _ => (sign << 31) | ((exponent + 112) << 23) | (significand << 13),
    };
    float_to_unorm8(f32::from_bits(float_bits))
}

fn native_texture_capture_to_bgra(
    native_bytes: &[u8],
    descriptor: GuestTextureDescriptor,
) -> Option<Vec<u8>> {
    let pixel_count = u64::from(descriptor.width)
        .checked_mul(u64::from(descriptor.height))
        .and_then(|pixels| usize::try_from(pixels).ok())?;
    let mut bgra = vec![0; pixel_count.checked_mul(4)?];
    match descriptor.pixel_format {
        MTLPixelFormat::RGBA8Unorm => {
            if native_bytes.len() < bgra.len() {
                return None;
            }
            for (source, destination) in native_bytes.chunks_exact(4).zip(bgra.chunks_exact_mut(4))
            {
                destination.copy_from_slice(&[
                    linear_unorm8_to_srgb_unorm8(source[2]),
                    linear_unorm8_to_srgb_unorm8(source[1]),
                    linear_unorm8_to_srgb_unorm8(source[0]),
                    source[3],
                ]);
            }
        }
        MTLPixelFormat::RGBA8Unorm_sRGB => {
            if native_bytes.len() < bgra.len() {
                return None;
            }
            for (source, destination) in native_bytes.chunks_exact(4).zip(bgra.chunks_exact_mut(4))
            {
                destination.copy_from_slice(&[source[2], source[1], source[0], source[3]]);
            }
        }
        MTLPixelFormat::BGRA8Unorm => {
            if native_bytes.len() < bgra.len() {
                return None;
            }
            for (source, destination) in native_bytes.chunks_exact(4).zip(bgra.chunks_exact_mut(4))
            {
                destination.copy_from_slice(&[
                    linear_unorm8_to_srgb_unorm8(source[0]),
                    linear_unorm8_to_srgb_unorm8(source[1]),
                    linear_unorm8_to_srgb_unorm8(source[2]),
                    source[3],
                ]);
            }
        }
        MTLPixelFormat::BGRA8Unorm_sRGB => {
            if native_bytes.len() < bgra.len() {
                return None;
            }
            let byte_count = bgra.len();
            bgra.copy_from_slice(&native_bytes[..byte_count]);
        }
        MTLPixelFormat::RGBA16Float => {
            if native_bytes.len() < pixel_count.checked_mul(8)? {
                return None;
            }
            for (source, destination) in native_bytes.chunks_exact(8).zip(bgra.chunks_exact_mut(4))
            {
                let red = u16::from_le_bytes([source[0], source[1]]);
                let green = u16::from_le_bytes([source[2], source[3]]);
                let blue = u16::from_le_bytes([source[4], source[5]]);
                let alpha = u16::from_le_bytes([source[6], source[7]]);
                destination.copy_from_slice(&[
                    half_float_to_srgb_unorm8(blue),
                    half_float_to_srgb_unorm8(green),
                    half_float_to_srgb_unorm8(red),
                    half_float_to_unorm8(alpha),
                ]);
            }
        }
        MTLPixelFormat::RG11B10Float => {
            if native_bytes.len() < bgra.len() {
                return None;
            }
            for (source, destination) in native_bytes.chunks_exact(4).zip(bgra.chunks_exact_mut(4))
            {
                let packed = u32::from_le_bytes(source.try_into().expect("four-byte pixel"));
                let red = packed & 0x7ff;
                let green = (packed >> 11) & 0x7ff;
                let blue = (packed >> 22) & 0x3ff;
                destination.copy_from_slice(&[
                    unsigned_float_to_srgb_unorm8(blue, 5),
                    unsigned_float_to_srgb_unorm8(green, 6),
                    unsigned_float_to_srgb_unorm8(red, 6),
                    255,
                ]);
            }
        }
        _ => return None,
    }
    Some(bgra)
}

fn validate_compute_launch_for_metal(
    launch: remu_gpu::ComputeLaunch,
    execution_width: u32,
    wave64_split_safe: bool,
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
    if execution_width != launch.wave_lane_count
        && !(launch.wave_lane_count == 64 && execution_width == 32 && wave64_split_safe)
    {
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

fn compute_program_can_split_wave64(program: &Program) -> bool {
    wave64_split_blockers(program).is_empty()
}

fn wave64_split_blockers(program: &Program) -> Vec<String> {
    let mut blockers = Vec::new();
    for instruction in &program.instructions {
        let dpp = matches!(
            instruction.control.as_ref(),
            Some(InstructionControl::Dpp(_) | InstructionControl::Dpp8(_))
        );
        let name = instruction.opcode.known_name();
        let wave_sensitive = name.is_none_or(|name| {
            [
                "Readlane",
                "Writelane",
                "Mbcnt",
                "Swizzle",
                "Ballot",
                "Saveexec",
            ]
            .iter()
            .any(|wave_operation| name.contains(wave_operation))
                // Structured wave64 predication is represented half-locally across the two
                // Metal SIMD32 groups. The observed SAndSaveexecB64 operation is therefore
                // split-safe; other saveexec variants remain rejected until their full-mask
                // behavior is proven.
                && name != "SAndSaveexecB64"
        });
        if dpp || wave_sensitive {
            blockers.push(format!(
                "{:#x}:{}{}",
                instruction.pc,
                instruction.opcode,
                if dpp { "/dpp" } else { "" }
            ));
        }
    }
    blockers
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

fn compile_metal_shader_library(
    device: &Device,
    stage: &str,
    source: &str,
) -> Result<Library, String> {
    if std::env::var_os("REMU_METAL_COMPILER").is_some() {
        return compile_metal_library_with_tools(device, stage, source);
    }

    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(false);
    device
        .new_library_with_source(source, &compile_options)
        .map_err(|error| format!("Metal {stage} compilation: {error}"))
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

fn compile_metal_library_with_tools(
    device: &Device,
    stage: &str,
    source: &str,
) -> Result<Library, String> {
    static NEXT_COMPILE_DIRECTORY: AtomicUsize = AtomicUsize::new(0);
    static CACHE_HITS: AtomicUsize = AtomicUsize::new(0);
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
    let cache_path = metal_library_cache_path(source);
    if cache_path.is_file() {
        match device.new_library_with_file(&cache_path) {
            Ok(library) => {
                let hit = CACHE_HITS.fetch_add(1, Ordering::Relaxed) + 1;
                if hit <= 8 || hit.is_power_of_two() {
                    eprintln!(
                        "[REMu][METAL] persistent shader cache hit={hit} stage={stage} path={}",
                        cache_path.display()
                    );
                }
                return Ok(library);
            }
            Err(error) => {
                eprintln!(
                    "[REMu][METAL] discarding unreadable cached {stage} library {}: {error}",
                    cache_path.display()
                );
                let _ = std::fs::remove_file(&cache_path);
            }
        }
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
    if let Some(parent) = cache_path.parent()
        && let Err(error) = std::fs::create_dir_all(parent)
    {
        eprintln!(
            "[REMu][METAL] could not create persistent shader cache {}: {error}",
            parent.display()
        );
    } else if let Err(error) = std::fs::rename(&library_path, &cache_path) {
        eprintln!(
            "[REMu][METAL] could not persist {stage} shader library {}: {error}",
            cache_path.display()
        );
    }
    let load_path = if cache_path.is_file() {
        &cache_path
    } else {
        &library_path
    };
    device
        .new_library_with_file(load_path)
        .map_err(|error| format!("could not load {}: {error}", load_path.display()))
}

fn metal_library_cache_path(source: &str) -> PathBuf {
    let mut hasher = Sha1::new();
    hasher.update(METAL_LIBRARY_CACHE_FORMAT);
    hasher.update(source.as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    metal_cache_root().join(format!("{digest}.metallib"))
}

fn metal_cache_root() -> PathBuf {
    std::env::var_os("REMU_METAL_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join("Library/Caches/REMu/metal"))
        })
        .unwrap_or_else(|| std::env::temp_dir().join("remu-metal-cache"))
}

fn metal_render_pipeline_archive_path(device: &Device) -> PathBuf {
    let mut hasher = Sha1::new();
    hasher.update(METAL_RENDER_PIPELINE_ARCHIVE_FORMAT);
    hasher.update(device.name().as_bytes());
    hasher.update(std::env::consts::ARCH.as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    metal_cache_root().join(format!("render-{digest}.metallibarchive"))
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

fn programmed_viewport_has_negative_height(draw: &IndexedDraw) -> bool {
    let programmed = [0x10f, 0x110, 0x111, 0x112]
        .map(|register| register_value(&draw.context_registers, register).map(f32::from_bits));
    matches!(
        programmed,
        [Some(x_scale), Some(x_offset), Some(y_scale), Some(y_offset)]
            if [x_scale, x_offset, y_scale, y_offset]
                .into_iter()
                .all(f32::is_finite)
                && x_scale > 0.0
                && y_scale < 0.0
    )
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
    let (_source_address, source_length) = guest_texture_raw_range(descriptor)?;
    let mut raw = vec![0; source_length];
    if read_guest_shared_backing(descriptor.address, &mut raw) {
        return Ok(raw);
    }
    #[cfg(test)]
    {
        // Tests also exercise the tiler with ordinary process-local allocations.
        // SAFETY: those test descriptors retain their complete source allocation.
        return Ok(unsafe {
            std::slice::from_raw_parts(_source_address as *const u8, source_length)
        }
        .to_vec());
    }
    #[cfg(not(test))]
    Err(format!(
        "guest texture range {:#x}+{source_length:#x} has no complete live shared backing",
        descriptor.address
    ))
}

fn watch_guest_texture(
    descriptor: GuestTextureDescriptor,
) -> Result<GuestTextureWriteWatch, String> {
    let (_, byte_count) = guest_texture_raw_range(descriptor)?;
    let byte_count =
        u64::try_from(byte_count).map_err(|_| "guest texture byte count does not fit u64")?;
    GuestTextureWriteWatch::new(descriptor.address, byte_count)
}

fn sampled_texture_raw_range(descriptor: DecodedImageDescriptor) -> Result<(usize, usize), String> {
    let exact_mip_chain = sampled_texture_uses_exact_mode27_mips(descriptor);
    let byte_count = if exact_mip_chain {
        if descriptor.texture_type != 9 || descriptor.layer_count != 1 {
            return Err(format!(
                "exact sampled mip-chain upload requires a 2D single-layer mode-27 resource, got type={} layers={} tile={} base={} count={}",
                descriptor.texture_type,
                descriptor.layer_count,
                descriptor.guest.tile_mode,
                descriptor.base_mip_level,
                descriptor.mip_level_count
            ));
        }
        let underlying_mip_count = descriptor
            .base_mip_level
            .checked_add(descriptor.mip_level_count)
            .ok_or("sampled mip range overflows")?;
        u64::try_from(mode27_mip_chain_layout(descriptor.guest, underlying_mip_count)?.byte_count)
            .map_err(|_| "sampled mip-chain byte count does not fit u64")?
    } else {
        guest_texture_byte_count(descriptor.guest)
            .ok_or("sampled texture face byte count overflows")?
            .checked_mul(u64::from(descriptor.layer_count))
            .ok_or("sampled texture byte count overflows")?
    };
    if descriptor.guest.address < 0x10_000 || byte_count > 512 * 1024 * 1024 {
        return Err(format!(
            "invalid sampled texture range address={:#x} bytes={byte_count}",
            descriptor.guest.address
        ));
    }
    Ok((
        usize::try_from(descriptor.guest.address)
            .map_err(|_| "sampled texture address does not fit usize")?,
        usize::try_from(byte_count).map_err(|_| "sampled texture byte count does not fit usize")?,
    ))
}

fn sampled_texture_uses_exact_mode27_mips(descriptor: DecodedImageDescriptor) -> bool {
    descriptor.guest.tile_mode == 27
        && (descriptor.base_mip_level != 0 || descriptor.mip_level_count != 1)
}

fn watch_sampled_texture(
    descriptor: DecodedImageDescriptor,
) -> Result<GuestTextureWriteWatch, String> {
    let (_, byte_count) = sampled_texture_raw_range(descriptor)?;
    GuestTextureWriteWatch::new(
        descriptor.guest.address,
        u64::try_from(byte_count).map_err(|_| "sampled byte count does not fit u64")?,
    )
}

fn snapshot_watched_sampled_texture_raw(
    descriptor: DecodedImageDescriptor,
    write_watch: &GuestTextureWriteWatch,
) -> Result<Vec<u8>, String> {
    let (_, source_length) = sampled_texture_raw_range(descriptor)?;
    snapshot_watched_guest_range(descriptor.guest.address, source_length, write_watch)
}

fn decode_sampled_texture_linear_from_raw(
    descriptor: DecodedImageDescriptor,
    raw: &[u8],
) -> Result<Vec<u8>, String> {
    let face_bytes = usize::try_from(
        guest_texture_byte_count(descriptor.guest)
            .ok_or("sampled texture face byte count overflows")?,
    )
    .map_err(|_| "sampled texture face byte count does not fit usize")?;
    let expected = face_bytes
        .checked_mul(
            usize::try_from(descriptor.layer_count)
                .map_err(|_| "sampled layer count does not fit usize")?,
        )
        .ok_or("sampled texture byte count overflows")?;
    if raw.len() != expected {
        return Err(format!(
            "sampled texture source has {} bytes, expected {expected}",
            raw.len()
        ));
    }
    let mut linear = Vec::new();
    for (surface_z, face) in raw.chunks_exact(face_bytes).enumerate() {
        linear.extend(decode_guest_texture_linear_from_raw_at_z(
            descriptor.guest,
            face,
            surface_z,
        )?);
    }
    Ok(linear)
}

fn snapshot_watched_guest_range(
    address: u64,
    byte_count: usize,
    write_watch: &GuestTextureWriteWatch,
) -> Result<Vec<u8>, String> {
    let byte_count_u64 =
        u64::try_from(byte_count).map_err(|_| "guest range length does not fit u64")?;
    let source_address =
        usize::try_from(address).map_err(|_| "guest range address does not fit usize")?;
    let direct_snapshot = || {
        // SAFETY: with_contiguous_mapping pins the complete live mapping around this copy.
        unsafe { std::slice::from_raw_parts(source_address as *const u8, byte_count) }.to_vec()
    };
    let snapshot = || {
        if guest_shared_mapping_identity(address, byte_count_u64).is_some()
            && write_watch.has_single_covering_mapping(address, byte_count_u64)
        {
            write_watch.with_contiguous_mapping(address, byte_count_u64, direct_snapshot)
        } else {
            let mut raw = vec![0; byte_count];
            if read_guest_shared_backing(address, &mut raw) {
                Ok(raw)
            } else {
                Err(format!(
                    "guest range {address:#x}+{byte_count:#x} has no complete live shared backing"
                ))
            }
        }
    };
    if !write_watch.precise_tracking() {
        return snapshot();
    }
    loop {
        let _ = write_watch.consume_dirty();
        write_watch.rearm()?;
        let generations = write_watch.generations();
        let raw = snapshot()?;
        if write_watch.generations() == generations && !write_watch.is_dirty() {
            return Ok(raw);
        }
    }
}

fn snapshot_watched_guest_texture_raw(
    descriptor: GuestTextureDescriptor,
    write_watch: &GuestTextureWriteWatch,
) -> Result<Vec<u8>, String> {
    let (_, source_length) = guest_texture_raw_range(descriptor)?;
    snapshot_watched_guest_range(descriptor.address, source_length, write_watch)
}

fn write_watched_guest_range(
    address: u64,
    raw: &[u8],
    write_watch: &GuestTextureWriteWatch,
) -> Result<(), String> {
    if write_watch.retirement() == Some(CpuWriteWatchRetirement::Replaced) {
        return Err(format!(
            "guest range mapping {address:#x} was retired before writeback"
        ));
    }
    let byte_count = u64::try_from(raw.len()).map_err(|_| "guest range length does not fit u64")?;
    if guest_shared_mapping_identity(address, byte_count).is_some()
        && write_watch.has_single_covering_mapping(address, byte_count)
    {
        let destination =
            usize::try_from(address).map_err(|_| "guest range address does not fit usize")?;
        return write_watch.with_contiguous_mapping(address, byte_count, || {
            // SAFETY: with_contiguous_mapping pins the complete writable mapping for this copy.
            unsafe {
                std::ptr::copy_nonoverlapping(raw.as_ptr(), destination as *mut u8, raw.len());
            }
        });
    }
    if write_guest_shared_backing(address, raw) {
        Ok(())
    } else {
        Err(format!(
            "guest range {address:#x}+{byte_count:#x} lost shared backing during writeback"
        ))
    }
}

fn rearm_guest_texture_after_gpu_writeback(
    write_watch: &GuestTextureWriteWatch,
) -> Result<(), String> {
    if !write_watch.precise_tracking() {
        return Ok(());
    }
    loop {
        let generations = write_watch.generations();
        let _ = write_watch.consume_dirty();
        write_watch.rearm()?;
        if write_watch.generations() == generations && !write_watch.is_dirty() {
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
    decode_guest_texture_linear_from_raw_at_z(descriptor, source, 0)
}

fn decode_guest_texture_linear_from_raw_at_z(
    descriptor: GuestTextureDescriptor,
    source: &[u8],
    surface_z: usize,
) -> Result<Vec<u8>, String> {
    let (width, height, pitch, bytes_per_element) = texture_element_layout(descriptor)?;
    let output_length = u64::try_from(width)
        .ok()
        .and_then(|width| width.checked_mul(u64::try_from(height).ok()?))
        .and_then(|elements| elements.checked_mul(u64::try_from(bytes_per_element).ok()?))
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
    let mut output = vec![0_u8; output_length];
    if descriptor.tile_mode == 0 {
        let source_row_bytes = pitch
            .checked_mul(bytes_per_element)
            .ok_or("source row byte count overflows")?;
        let output_row_bytes = width
            .checked_mul(bytes_per_element)
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
        copy_tiled_to_linear_at_z(source, descriptor, &mut output, surface_z)?;
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
    let (width, height, _pitch, bytes_per_element) = texture_element_layout(descriptor)?;
    let required = width
        .checked_mul(height)
        .and_then(|elements| elements.checked_mul(bytes_per_element))
        .ok_or("guest texture dimensions overflow")?;
    if linear.len() != required {
        return Err(format!(
            "tight guest image has {} bytes, expected {required}",
            linear.len()
        ));
    }
    let destination_length =
        usize::try_from(byte_count).map_err(|_| "guest texture byte count does not fit usize")?;
    if guest_shared_mapping_identity(descriptor.address, byte_count).is_some() {
        let destination_address = usize::try_from(descriptor.address)
            .map_err(|_| "guest texture address does not fit usize")?;
        // SAFETY: the registry proved the full writable texture range is one live contiguous
        // mapping, and the caller's write watch has not been retired.
        let destination = unsafe {
            std::slice::from_raw_parts_mut(destination_address as *mut u8, destination_length)
        };
        write_guest_texture_linear_into_raw(descriptor, linear, destination)?;
        return Ok(());
    }
    let mut destination = vec![0; destination_length];
    let has_shared_backing = read_guest_shared_backing(descriptor.address, &mut destination);
    #[cfg(test)]
    if !has_shared_backing {
        let destination_address = usize::try_from(descriptor.address)
            .map_err(|_| "guest texture address does not fit usize")?;
        // SAFETY: process-local unit-test descriptors retain the full destination allocation.
        destination.copy_from_slice(unsafe {
            std::slice::from_raw_parts(destination_address as *const u8, destination_length)
        });
    }
    #[cfg(not(test))]
    if !has_shared_backing {
        return Err(format!(
            "guest texture range {:#x}+{byte_count:#x} has no complete live shared backing",
            descriptor.address
        ));
    }
    write_guest_texture_linear_into_raw(descriptor, linear, &mut destination)?;
    if has_shared_backing {
        if !write_guest_shared_backing(descriptor.address, &destination) {
            return Err(format!(
                "guest texture range {:#x}+{byte_count:#x} lost shared backing during writeback",
                descriptor.address
            ));
        }
    } else {
        #[cfg(test)]
        {
            let destination_address = usize::try_from(descriptor.address)
                .map_err(|_| "guest texture address does not fit usize")?;
            // SAFETY: process-local unit-test descriptors retain the full destination allocation.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    destination.as_ptr(),
                    destination_address as *mut u8,
                    destination.len(),
                );
            }
        }
    }
    Ok(())
}

fn write_guest_texture_linear_into_raw(
    descriptor: GuestTextureDescriptor,
    linear: &[u8],
    destination: &mut [u8],
) -> Result<(), String> {
    let (width, height, pitch, bytes_per_element) = texture_element_layout(descriptor)?;
    if descriptor.tile_mode == 0 {
        let destination_row_bytes = pitch
            .checked_mul(bytes_per_element)
            .ok_or("destination row byte count overflows")?;
        let source_row_bytes = width
            .checked_mul(bytes_per_element)
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
        copy_linear_to_tiled(linear, descriptor, destination)?;
    }
    Ok(())
}

fn write_watched_guest_texture_linear(
    descriptor: GuestTextureDescriptor,
    linear: &[u8],
    write_watch: &GuestTextureWriteWatch,
) -> Result<(), String> {
    if write_watch.retirement() == Some(CpuWriteWatchRetirement::Replaced) {
        return Err(format!(
            "guest texture mapping {:#x} was retired before writeback",
            descriptor.address
        ));
    }
    let byte_count = guest_texture_byte_count(descriptor)
        .ok_or("guest texture byte count overflows the host address space")?;
    if guest_shared_mapping_identity(descriptor.address, byte_count).is_some()
        && write_watch.has_single_covering_mapping(descriptor.address, byte_count)
    {
        return write_watch.with_contiguous_mapping(descriptor.address, byte_count, || {
            write_guest_texture_linear(descriptor, linear)
        })?;
    }
    let destination_length =
        usize::try_from(byte_count).map_err(|_| "guest texture byte count does not fit usize")?;
    let mut destination = vec![0; destination_length];
    if !read_guest_shared_backing(descriptor.address, &mut destination) {
        return Err(format!(
            "guest texture range {:#x}+{byte_count:#x} has no complete live shared backing",
            descriptor.address
        ));
    }
    write_guest_texture_linear_into_raw(descriptor, linear, &mut destination)?;
    if !write_guest_shared_backing(descriptor.address, &destination) {
        return Err(format!(
            "guest texture range {:#x}+{byte_count:#x} lost shared backing during writeback",
            descriptor.address
        ));
    }
    Ok(())
}

fn create_guest_texture(
    device: &Device,
    guest: GuestTextureDescriptor,
    usage: MTLTextureUsage,
    linear: &[u8],
) -> Result<Texture, String> {
    let (elements_wide, elements_high, _, bytes_per_element) = texture_element_layout(guest)?;
    let required = u64::try_from(elements_wide)
        .ok()
        .and_then(|width| width.checked_mul(u64::try_from(elements_high).ok()?))
        .and_then(|elements| elements.checked_mul(u64::try_from(bytes_per_element).ok()?))
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
        u64::try_from(elements_wide).expect("validated texture element width fits u64")
            * u64::try_from(bytes_per_element)
                .expect("validated texture element byte count fits u64"),
    );
    Ok(texture)
}

fn create_mipmapped_guest_texture(
    device: &Device,
    guest: GuestTextureDescriptor,
    mip_level_count: u32,
    linear_levels: &[Vec<u8>],
) -> Result<Texture, String> {
    let layout = mode27_mip_chain_layout(guest, mip_level_count)?;
    if linear_levels.len() != layout.levels.len() {
        return Err(format!(
            "Metal mip upload has {} levels, expected {}",
            linear_levels.len(),
            layout.levels.len()
        ));
    }
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(guest.pixel_format);
    descriptor.set_width(u64::from(guest.width));
    descriptor.set_height(u64::from(guest.height));
    descriptor.set_mipmap_level_count(u64::from(mip_level_count));
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(
        MTLTextureUsage::RenderTarget
            | MTLTextureUsage::ShaderRead
            | MTLTextureUsage::ShaderWrite
            | MTLTextureUsage::PixelFormatView,
    );
    let texture = device.new_texture(&descriptor);
    for (level_index, (&level, linear)) in layout.levels.iter().zip(linear_levels).enumerate() {
        let expected = level
            .width
            .checked_mul(level.height)
            .and_then(|elements| elements.checked_mul(layout.bytes_per_element))
            .ok_or("Metal mip upload size overflows")?;
        if linear.len() != expected {
            return Err(format!(
                "Metal mip {level_index} upload has {} bytes, expected {expected}",
                linear.len()
            ));
        }
        texture.replace_region(
            MTLRegion::new_2d(
                0,
                0,
                u64::try_from(level.width).expect("validated mip width fits u64"),
                u64::try_from(level.height).expect("validated mip height fits u64"),
            ),
            u64::try_from(level_index).expect("validated mip index fits u64"),
            linear.as_ptr().cast::<c_void>(),
            u64::try_from(level.width * layout.bytes_per_element)
                .expect("validated mip row size fits u64"),
        );
    }
    Ok(texture)
}

fn create_exact_mode27_sampled_mip_view(
    device: &Device,
    descriptor: DecodedImageDescriptor,
    raw: &[u8],
    comparison: bool,
) -> Result<Texture, String> {
    if comparison {
        return Err(format!(
            "exact comparison-sampled mip view base={} count={} is not implemented",
            descriptor.base_mip_level, descriptor.mip_level_count
        ));
    }
    if descriptor.texture_type != 9
        || descriptor.layer_count != 1
        || descriptor.guest.tile_mode != 27
        || descriptor.mip_level_count == 0
    {
        return Err(format!(
            "exact sampled mip view requires a nonempty 2D single-layer mode-27 resource, got type={} layers={} tile={} base={} count={}",
            descriptor.texture_type,
            descriptor.layer_count,
            descriptor.guest.tile_mode,
            descriptor.base_mip_level,
            descriptor.mip_level_count
        ));
    }
    let underlying_mip_count = descriptor
        .base_mip_level
        .checked_add(descriptor.mip_level_count)
        .ok_or("sampled mip range overflows")?;
    let (_, linear_levels) =
        decode_mode27_mip_chain_from_raw(descriptor.guest, underlying_mip_count, raw)?;
    let texture = create_mipmapped_guest_texture(
        device,
        descriptor.guest,
        underlying_mip_count,
        &linear_levels,
    )?;
    if descriptor.base_mip_level == 0 && descriptor.mip_level_count == underlying_mip_count {
        return Ok(texture);
    }
    Ok(texture.new_texture_view_from_slice(
        descriptor.guest.pixel_format,
        MTLTextureType::D2,
        NSRange::new(
            u64::from(descriptor.base_mip_level),
            u64::from(descriptor.mip_level_count),
        ),
        NSRange::new(0, 1),
    ))
}

fn create_layered_render_target_texture(
    device: &Device,
    descriptor: DecodedImageDescriptor,
    linear: &[u8],
) -> Result<Texture, String> {
    if descriptor.texture_type != 10 || descriptor.layer_count <= 1 {
        return Err(format!(
            "layered render target requires a 3D texture, got type={} layers={}",
            descriptor.texture_type, descriptor.layer_count
        ));
    }
    let (elements_wide, elements_high, _, bytes_per_element) =
        texture_element_layout(descriptor.guest)?;
    let bytes_per_row = elements_wide
        .checked_mul(bytes_per_element)
        .ok_or("layered Metal texture row size overflows")?;
    let bytes_per_image = bytes_per_row
        .checked_mul(elements_high)
        .ok_or("layered Metal texture image size overflows")?;
    let required = bytes_per_image
        .checked_mul(
            usize::try_from(descriptor.layer_count)
                .map_err(|_| "layered Metal texture depth does not fit usize")?,
        )
        .ok_or("layered Metal texture byte count overflows")?;
    if linear.len() != required {
        return Err(format!(
            "invalid layered guest texture upload {}x{}x{} format {:?} with {} bytes (expected {required})",
            descriptor.guest.width,
            descriptor.guest.height,
            descriptor.layer_count,
            descriptor.guest.pixel_format,
            linear.len()
        ));
    }
    let metal_descriptor = TextureDescriptor::new();
    metal_descriptor.set_texture_type(MTLTextureType::D3);
    metal_descriptor.set_pixel_format(descriptor.guest.pixel_format);
    metal_descriptor.set_width(u64::from(descriptor.guest.width));
    metal_descriptor.set_height(u64::from(descriptor.guest.height));
    metal_descriptor.set_depth(u64::from(descriptor.layer_count));
    metal_descriptor.set_storage_mode(MTLStorageMode::Shared);
    metal_descriptor.set_usage(
        MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite,
    );
    let texture = device.new_texture(&metal_descriptor);
    texture.replace_region_in_slice(
        MTLRegion::new_3d(
            0,
            0,
            0,
            u64::from(descriptor.guest.width),
            u64::from(descriptor.guest.height),
            u64::from(descriptor.layer_count),
        ),
        0,
        0,
        linear.as_ptr().cast::<c_void>(),
        u64::try_from(bytes_per_row).expect("validated row size fits u64"),
        u64::try_from(bytes_per_image).expect("validated image size fits u64"),
    );
    Ok(texture)
}

fn create_sampled_guest_texture(
    device: &Device,
    queue: &metal::CommandQueue,
    guest: GuestTextureDescriptor,
    linear: &[u8],
    base_mip_level: u32,
    requested_mip_level_count: u32,
    texture_type: u32,
    layer_count: u32,
    comparison: bool,
) -> Result<Texture, String> {
    let (elements_wide, elements_high, _, bytes_per_element) = texture_element_layout(guest)?;
    let source_face_bytes = elements_wide
        .checked_mul(elements_high)
        .and_then(|elements| elements.checked_mul(bytes_per_element))
        .ok_or("Metal sampled texture dimensions overflow")?;
    let required = source_face_bytes
        .checked_mul(
            usize::try_from(layer_count).map_err(|_| "sampled layer count does not fit usize")?,
        )
        .ok_or("Metal sampled texture layer dimensions overflow")?;
    if guest.width == 0 || guest.height == 0 || linear.len() != required {
        return Err(format!(
            "invalid sampled guest texture upload {}x{} format {:?} with {} bytes (expected {required})",
            guest.width,
            guest.height,
            guest.pixel_format,
            linear.len()
        ));
    }
    if trace_image_descriptors()
        && guest.tile_mode == 0
        && guest.width >= 700
        && guest.height >= 500
        && matches!(
            guest.pixel_format,
            MTLPixelFormat::R8Unorm | MTLPixelFormat::RG8Unorm
        )
    {
        static LOGGED_VIDEO_UPLOADS: OnceLock<Mutex<HashSet<(u64, u64)>>> = OnceLock::new();
        let fingerprint = guest_frame_fingerprint(linear);
        let mut logged = LOGGED_VIDEO_UPLOADS
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if logged.len() < 32 && logged.insert((guest.address, fingerprint)) {
            let (minimum, maximum, total) = linear.iter().fold(
                (u8::MAX, u8::MIN, 0_u64),
                |(minimum, maximum, total), &value| {
                    (
                        minimum.min(value),
                        maximum.max(value),
                        total + u64::from(value),
                    )
                },
            );
            let average = total as f64 / linear.len().max(1) as f64;
            eprintln!(
                "[REMu][METAL][IMAGE] uploaded linear video plane address={:#x} extent={}x{} format={:?} bytes={} hash={fingerprint:016x} range={minimum}..={maximum} average={average:.1}",
                guest.address,
                guest.width,
                guest.height,
                guest.pixel_format,
                linear.len(),
            );
        }
    }
    let comparison_linear = comparison
        .then(|| comparison_depth32_upload(guest.pixel_format, linear))
        .transpose()?
        .flatten();
    let upload = comparison_linear.as_deref().unwrap_or(linear);
    let upload_bytes_per_element = if comparison_linear.is_some() {
        std::mem::size_of::<f32>()
    } else {
        bytes_per_element
    };
    let face_bytes = elements_wide
        .checked_mul(elements_high)
        .and_then(|elements| elements.checked_mul(upload_bytes_per_element))
        .ok_or("Metal sampled texture upload dimensions overflow")?;
    // GFX10 packs small mip levels into a tiled mip tail. Until that guest layout is decoded,
    // generate the same filtered hierarchy from the exact base image on Metal. Compressed BC
    // formats cannot be filtered by Metal's blit mip generator, so they remain base-level views.
    let underlying_mip_level_count = base_mip_level
        .checked_add(requested_mip_level_count.max(1))
        .ok_or("sampled Metal mip range overflows")?;
    let mip_level_count = if compressed_texture_block_extent(guest.pixel_format).is_some() {
        if base_mip_level != 0 {
            return Err(format!(
                "compressed sampled Metal texture cannot synthesize base mip {base_mip_level}"
            ));
        }
        1
    } else {
        underlying_mip_level_count
    };
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(match texture_type {
        9 => MTLTextureType::D2,
        10 => MTLTextureType::D3,
        11 if layer_count == 6 => MTLTextureType::Cube,
        11 => MTLTextureType::CubeArray,
        13 => MTLTextureType::D2Array,
        _ => {
            return Err(format!(
                "unsupported sampled Metal texture type {texture_type}"
            ));
        }
    });
    let pixel_format = if comparison {
        match guest.pixel_format {
            MTLPixelFormat::R16Unorm => MTLPixelFormat::Depth16Unorm,
            MTLPixelFormat::R32Float => MTLPixelFormat::Depth32Float,
            _ => MTLPixelFormat::Depth32Float,
        }
    } else {
        guest.pixel_format
    };
    descriptor.set_pixel_format(pixel_format);
    descriptor.set_width(u64::from(guest.width));
    descriptor.set_height(u64::from(guest.height));
    if texture_type == 10 {
        descriptor.set_depth(u64::from(layer_count));
    }
    descriptor.set_mipmap_level_count(u64::from(mip_level_count));
    if texture_type == 13 {
        descriptor.set_array_length(u64::from(layer_count));
    }
    if texture_type == 11 {
        descriptor.set_array_length(u64::from(layer_count / 6));
    }
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(if texture_type == 10 && !comparison {
        MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite
    } else {
        MTLTextureUsage::ShaderRead
    });
    let texture = device.new_texture(&descriptor);
    let bytes_per_row = u64::try_from(elements_wide)
        .expect("validated texture element width fits u64")
        * u64::try_from(upload_bytes_per_element)
            .expect("validated texture element byte count fits u64");
    if texture_type == 10 {
        texture.replace_region_in_slice(
            MTLRegion::new_3d(
                0,
                0,
                0,
                u64::from(guest.width),
                u64::from(guest.height),
                u64::from(layer_count),
            ),
            0,
            0,
            upload.as_ptr().cast::<c_void>(),
            bytes_per_row,
            u64::try_from(face_bytes).expect("validated image byte count fits u64"),
        );
    } else if matches!(texture_type, 11 | 13) {
        for layer in 0..layer_count {
            let offset = usize::try_from(layer)
                .expect("validated layer fits usize")
                .checked_mul(face_bytes)
                .expect("validated layer offset fits usize");
            texture.replace_region_in_slice(
                MTLRegion::new_2d(0, 0, u64::from(guest.width), u64::from(guest.height)),
                0,
                u64::from(layer),
                upload[offset..].as_ptr().cast::<c_void>(),
                bytes_per_row,
                u64::try_from(face_bytes).expect("validated face byte count fits u64"),
            );
        }
    } else {
        texture.replace_region(
            MTLRegion::new_2d(0, 0, u64::from(guest.width), u64::from(guest.height)),
            0,
            upload.as_ptr().cast::<c_void>(),
            bytes_per_row,
        );
    }
    if mip_level_count > 1 {
        let command_buffer = queue.new_command_buffer().to_owned();
        command_buffer.set_label("REMu sampled image mip generation");
        let encoder = command_buffer.new_blit_command_encoder();
        encoder.set_label("REMu sampled image generated mip hierarchy");
        encoder.generate_mipmaps(&texture);
        encoder.end_encoding();
        command_buffer.commit();
    }
    if base_mip_level == 0 {
        return Ok(texture);
    }
    if texture_type != 9 {
        return Err(format!(
            "nonzero sampled base mip {base_mip_level} is not implemented for texture type {texture_type}"
        ));
    }
    Ok(texture.new_texture_view_from_slice(
        pixel_format,
        MTLTextureType::D2,
        NSRange::new(
            u64::from(base_mip_level),
            u64::from(requested_mip_level_count.max(1)),
        ),
        NSRange::new(0, 1),
    ))
}

fn comparison_depth32_upload(
    format: MTLPixelFormat,
    linear: &[u8],
) -> Result<Option<Vec<u8>>, String> {
    let (stride, red_offset, srgb) = match format {
        MTLPixelFormat::R16Unorm | MTLPixelFormat::R32Float => return Ok(None),
        MTLPixelFormat::R8Unorm => (1, 0, false),
        MTLPixelFormat::R8Unorm_sRGB => (1, 0, true),
        MTLPixelFormat::RG8Unorm => (2, 0, false),
        MTLPixelFormat::RG8Unorm_sRGB => (2, 0, true),
        MTLPixelFormat::RGBA8Unorm => (4, 0, false),
        MTLPixelFormat::RGBA8Unorm_sRGB => (4, 0, true),
        MTLPixelFormat::BGRA8Unorm => (4, 2, false),
        MTLPixelFormat::BGRA8Unorm_sRGB => (4, 2, true),
        _ => {
            return Err(format!(
                "comparison sampling cannot convert Metal image format {format:?} to depth"
            ));
        }
    };
    if !linear.len().is_multiple_of(stride) {
        return Err(format!(
            "comparison image byte count {} is not aligned to its {stride}-byte texel",
            linear.len()
        ));
    }
    let mut depth = Vec::with_capacity(linear.len() / stride * std::mem::size_of::<f32>());
    for texel in linear.chunks_exact(stride) {
        let encoded = f32::from(texel[red_offset]) / 255.0;
        let value = if srgb {
            if encoded <= 0.040_45 {
                encoded / 12.92
            } else {
                ((encoded + 0.055) / 1.055).powf(2.4)
            }
        } else {
            encoded
        };
        depth.extend_from_slice(&value.to_ne_bytes());
    }
    Ok(Some(depth))
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

fn read_texture_mip_linear(
    device: &Device,
    queue: &metal::CommandQueue,
    texture: &Texture,
    descriptor: GuestTextureDescriptor,
    mip_level: u32,
) -> Result<Vec<u8>, String> {
    PendingTextureReadback::encode_mip(device, queue, texture, descriptor, mip_level)?.finish()
}

fn write_resident_texture_contents_to_guest(
    device: &Device,
    queue: &metal::CommandQueue,
    resident: &ResidentGuestTexture,
) -> Result<(), String> {
    if resident.mip_level_count == 1 {
        let linear = read_texture_linear(device, queue, &resident.texture, resident.descriptor)?;
        return write_watched_guest_texture_linear(
            resident.descriptor,
            &linear,
            &resident.write_watch,
        );
    }
    if resident.texture_type != 9 || resident.layer_count != 1 {
        return Err(format!(
            "mip writeback requires a 2D single-layer resident, got type={} layers={}",
            resident.texture_type, resident.layer_count
        ));
    }
    let layout = mode27_mip_chain_layout(resident.descriptor, resident.mip_level_count)?;
    let mut raw = snapshot_watched_guest_range(
        resident.descriptor.address,
        layout.byte_count,
        &resident.write_watch,
    )?;
    let mut levels = Vec::with_capacity(layout.levels.len());
    for mip_level in 0..resident.mip_level_count {
        levels.push(read_texture_mip_linear(
            device,
            queue,
            &resident.texture,
            resident.descriptor,
            mip_level,
        )?);
    }
    encode_mode27_mip_chain_into_raw(&layout, &levels, &mut raw)?;
    write_watched_guest_range(resident.descriptor.address, &raw, &resident.write_watch)
}

fn read_texture_volume_linear(
    device: &Device,
    queue: &metal::CommandQueue,
    texture: &Texture,
    descriptor: GuestTextureDescriptor,
) -> Result<Vec<u8>, String> {
    if texture.texture_type() != MTLTextureType::D3 {
        return Err(format!(
            "Metal volume readback requires a 3D texture, got {:?}",
            texture.texture_type()
        ));
    }
    if texture.framebuffer_only() || texture.storage_mode() == MTLStorageMode::Memoryless {
        return Err("Metal 3D texture has no readable stored contents".to_owned());
    }
    if texture.sample_count() != 1
        || texture.width() != u64::from(descriptor.width)
        || texture.height() != u64::from(descriptor.height)
        || texture.pixel_format() != descriptor.pixel_format
    {
        return Err(format!(
            "Metal volume readback mismatch: texture={}x{}x{} {:?} descriptor={}x{} {:?}",
            texture.width(),
            texture.height(),
            texture.depth(),
            texture.pixel_format(),
            descriptor.width,
            descriptor.height,
            descriptor.pixel_format,
        ));
    }
    let expected_bytes =
        metal_uncompressed_bytes_per_pixel(descriptor.pixel_format).ok_or_else(|| {
            format!(
                "Metal volume readback has no exact uncompressed layout for {:?}",
                descriptor.pixel_format
            )
        })?;
    if descriptor.bytes_per_pixel != expected_bytes {
        return Err(format!(
            "Metal volume readback format {:?} requires {expected_bytes} bytes per pixel, got {}",
            descriptor.pixel_format, descriptor.bytes_per_pixel
        ));
    }
    let layout = TextureReadbackLayout::new(
        descriptor.width,
        descriptor.height,
        descriptor.bytes_per_pixel,
    )?;
    let depth = texture.depth();
    let staging_bytes_per_image = layout.staging_length;
    let staging_length = staging_bytes_per_image
        .checked_mul(depth)
        .ok_or("Metal volume staging size overflows")?;
    let staging = device.new_buffer(
        staging_length,
        MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache,
    );
    staging.set_label("REMu 3D texture readback staging");
    let command_buffer = queue.new_command_buffer().to_owned();
    command_buffer.set_label("REMu 3D texture readback");
    let encoder = command_buffer.new_blit_command_encoder();
    encoder.set_label("REMu 3D texture-to-buffer readback blit");
    encoder.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        MTLOrigin { x: 0, y: 0, z: 0 },
        MTLSize::new(
            u64::from(descriptor.width),
            u64::from(descriptor.height),
            depth,
        ),
        &staging,
        0,
        layout.staging_bytes_per_row,
        staging_bytes_per_image,
        MTLBlitOption::None,
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    if command_buffer.status() != MTLCommandBufferStatus::Completed {
        return Err(format!(
            "Metal volume readback ended in {:?}",
            command_buffer.status()
        ));
    }
    let packed_row = usize::try_from(layout.packed_bytes_per_row)
        .map_err(|_| "Metal volume packed row does not fit usize")?;
    let staging_row = usize::try_from(layout.staging_bytes_per_row)
        .map_err(|_| "Metal volume staging row does not fit usize")?;
    let staging_image = usize::try_from(staging_bytes_per_image)
        .map_err(|_| "Metal volume staging image does not fit usize")?;
    let staging_length = usize::try_from(staging_length)
        .map_err(|_| "Metal volume staging size does not fit usize")?;
    let height =
        usize::try_from(descriptor.height).map_err(|_| "Metal volume height does not fit usize")?;
    let depth = usize::try_from(depth).map_err(|_| "Metal volume depth does not fit usize")?;
    let packed_image = packed_row
        .checked_mul(height)
        .ok_or("Metal volume packed image size overflows")?;
    let packed_length = packed_image
        .checked_mul(depth)
        .ok_or("Metal volume packed size overflows")?;
    let contents = staging.contents().cast::<u8>();
    if contents.is_null() {
        return Err("Metal shared volume readback buffer has no CPU mapping".to_owned());
    }
    // SAFETY: the shared staging buffer remains live, and the completed blit initialized its
    // complete declared extent.
    let staged = unsafe { std::slice::from_raw_parts(contents, staging_length) };
    let mut packed = vec![0_u8; packed_length];
    for z in 0..depth {
        for y in 0..height {
            let source = z * staging_image + y * staging_row;
            let destination = z * packed_image + y * packed_row;
            packed[destination..destination + packed_row]
                .copy_from_slice(&staged[source..source + packed_row]);
        }
    }
    Ok(packed)
}

fn ensure_resident_texture_in_map(
    device: &Device,
    descriptor: GuestTextureDescriptor,
    resident_guest_textures: &mut HashMap<GuestTextureDescriptor, ResidentGuestTexture>,
) -> Result<Texture, String> {
    if let Some(resident) = resident_guest_textures
        .get_mut(&descriptor)
        .filter(|resident| {
            resident.texture_type == 9 && resident.layer_count == 1 && resident.mip_level_count >= 1
        })
    {
        match refresh_resident_texture_write_watch(resident)? {
            WriteWatchRefresh::Active
            | WriteWatchRefresh::Reprotecting
            | WriteWatchRefresh::Renewed { .. } => return Ok(resident.texture.clone()),
            WriteWatchRefresh::Replaced => {}
        }
    }
    resident_guest_textures.remove(&descriptor);
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
            component_reorder_view: None,
            descriptor,
            texture_type: 9,
            layer_count: 1,
            mip_level_count: 1,
            write_watch,
            gpu_written: false,
            content_version: 0,
        },
    );
    Ok(texture)
}

fn complete_2d_mip_level_count(descriptor: GuestTextureDescriptor) -> u32 {
    let maximum_extent = descriptor.width.max(descriptor.height).max(1);
    (u32::BITS - maximum_extent.leading_zeros()).min(16)
}

fn ensure_mipmapped_resident_texture_in_map(
    device: &Device,
    descriptor: DecodedImageDescriptor,
    resident_guest_textures: &mut HashMap<GuestTextureDescriptor, ResidentGuestTexture>,
) -> Result<Texture, String> {
    if descriptor.texture_type != 9
        || descriptor.layer_count != 1
        || descriptor.guest.tile_mode != 27
        || descriptor.mip_level_count == 0
    {
        return Err(format!(
            "mipmapped storage image requires a nonempty 2D single-layer mode-27 resource, got type={} layers={} tile={} base={} count={}",
            descriptor.texture_type,
            descriptor.layer_count,
            descriptor.guest.tile_mode,
            descriptor.base_mip_level,
            descriptor.mip_level_count
        ));
    }
    let underlying_mip_count = descriptor
        .base_mip_level
        .checked_add(descriptor.mip_level_count)
        .ok_or("storage-image mip range overflows")?;
    if let Some(resident) = resident_guest_textures
        .get_mut(&descriptor.guest)
        .filter(|resident| {
            resident.texture_type == 9
                && resident.layer_count == 1
                && resident.mip_level_count >= underlying_mip_count
        })
    {
        match refresh_resident_texture_write_watch(resident)? {
            WriteWatchRefresh::Active
            | WriteWatchRefresh::Reprotecting
            | WriteWatchRefresh::Renewed { .. } => {
                return resident_texture_for_sample(resident, descriptor).ok_or_else(|| {
                    "resident mip hierarchy cannot represent storage view".to_owned()
                });
            }
            WriteWatchRefresh::Replaced => {}
        }
    }
    if resident_guest_textures
        .get(&descriptor.guest)
        .is_some_and(|resident| resident.gpu_written)
    {
        return Err(format!(
            "cannot reinterpret GPU-written image {:#x} as a {underlying_mip_count}-level hierarchy",
            descriptor.guest.address
        ));
    }
    resident_guest_textures.remove(&descriptor.guest);
    let layout = mode27_mip_chain_layout(descriptor.guest, underlying_mip_count)?;
    let write_watch = watch_resident_texture(
        descriptor.guest,
        descriptor.texture_type,
        descriptor.layer_count,
        underlying_mip_count,
    )?;
    let raw =
        snapshot_watched_guest_range(descriptor.guest.address, layout.byte_count, &write_watch)?;
    let (_, linear_levels) =
        decode_mode27_mip_chain_from_raw(descriptor.guest, underlying_mip_count, &raw)?;
    let texture = create_mipmapped_guest_texture(
        device,
        descriptor.guest,
        underlying_mip_count,
        &linear_levels,
    )?;
    texture.set_label(&format!(
        "REMu {}-level storage image {:#x}",
        underlying_mip_count, descriptor.guest.address
    ));
    resident_guest_textures.insert(
        descriptor.guest,
        ResidentGuestTexture {
            texture,
            component_reorder_view: None,
            descriptor: descriptor.guest,
            texture_type: descriptor.texture_type,
            layer_count: descriptor.layer_count,
            mip_level_count: underlying_mip_count,
            write_watch,
            gpu_written: false,
            content_version: 0,
        },
    );
    resident_texture_for_sample(
        resident_guest_textures
            .get_mut(&descriptor.guest)
            .expect("inserted mipmapped resident exists"),
        descriptor,
    )
    .ok_or_else(|| "created mip hierarchy cannot represent storage view".to_owned())
}

fn metal_buffer_with_bytes(device: &Device, bytes: &[u8]) -> Result<Buffer, String> {
    let empty = [0_u8];
    let bytes = if bytes.is_empty() {
        empty.as_slice()
    } else {
        bytes
    };
    let length =
        u64::try_from(bytes.len()).map_err(|_| "global buffer byte count does not fit Metal")?;
    Ok(device.new_buffer_with_data(
        bytes.as_ptr().cast::<c_void>(),
        length,
        MTLResourceOptions::StorageModeShared,
    ))
}

fn guest_global_buffer_view(
    binding: &GlobalMemoryBinding,
) -> Result<Option<(u64, usize, u64)>, String> {
    if binding.byte_count == 0
        || binding.base_address < 0x10_000
        || !binding.base_address.is_multiple_of(4)
    {
        return Ok(None);
    }
    static PAGE_SIZE: OnceLock<usize> = OnceLock::new();
    let page_size = *PAGE_SIZE.get_or_init(|| {
        // SAFETY: sysconf is side-effect free for this process-wide constant.
        let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        usize::try_from(value)
            .ok()
            .filter(|value| value.is_power_of_two())
            .unwrap_or(16 * 1024)
    });
    let base = usize::try_from(binding.base_address)
        .map_err(|_| "guest global address does not fit the host pointer")?;
    let end = base
        .checked_add(binding.byte_count)
        .ok_or("guest global byte range overflows")?;
    let page_base = base & !(page_size - 1);
    let page_end = end
        .checked_add(page_size - 1)
        .map(|value| value & !(page_size - 1))
        .ok_or("page-aligned guest global byte range overflows")?;
    let page_byte_count = page_end
        .checked_sub(page_base)
        .ok_or("page-aligned guest global byte range underflows")?;
    let offset = u64::try_from(base - page_base)
        .map_err(|_| "guest global page offset does not fit Metal")?;
    Ok(Some((
        u64::try_from(page_base).map_err(|_| "guest global page address does not fit u64")?,
        page_byte_count,
        offset,
    )))
}

fn shader_scalar_metadata(evaluation: &ShaderEvaluation) -> Result<Vec<u32>, String> {
    let mut words = evaluation.initial_scalar_registers.to_vec();
    words.reserve(evaluation.global_memory_bindings.len().saturating_mul(4));
    for binding in &evaluation.global_memory_bindings {
        words.extend([
            binding.base_address as u32,
            (binding.base_address >> 32) as u32,
            u32::try_from(binding.byte_count)
                .map_err(|_| "shader global binding exceeds the 32-bit Metal byte range")?,
            u32::from(binding.writable),
        ]);
    }
    Ok(words)
}

fn shader_scalar_metadata_for_registers(
    evaluation: &ShaderEvaluation,
    shader_registers: &[GpuRegister],
    user_data_register: u32,
    scalar_register_base: usize,
) -> Result<Vec<u32>, String> {
    let mut words = shader_scalar_metadata(evaluation)?;
    let scalar_end = scalar_register_base
        .saturating_add(64)
        .min(evaluation.initial_scalar_registers.len());
    for scalar_register in scalar_register_base..scalar_end {
        let offset = u32::try_from(scalar_register - scalar_register_base)
            .map_err(|_| "shader scalar register offset does not fit u32")?;
        words[scalar_register] = shader_registers
            .iter()
            .find(|register| register.offset == user_data_register + offset)
            .map_or(0, |register| register.value);
    }
    Ok(words)
}

fn collect_pending_global_writebacks(
    buffers: &[UploadedGlobalBuffer],
    evaluation: &ShaderEvaluation,
    pending: &mut Vec<PendingGlobalWriteback>,
) -> Result<(), String> {
    if buffers.len() != evaluation.global_memory_bindings.len() {
        return Err("Metal/global buffer binding count mismatch".to_owned());
    }
    for (buffer, binding) in buffers
        .iter()
        .zip(&evaluation.global_memory_bindings)
        .filter(|(buffer, binding)| {
            !buffer.guest_memory && binding.writable && binding.write_back_to_guest
        })
    {
        if binding.base_address < 0x10_000 || binding.byte_count > 128 * 1024 * 1024 {
            return Err(format!(
                "invalid writable global buffer address={:#x} bytes={}",
                binding.base_address, binding.byte_count
            ));
        }
        if pending.iter().any(|writeback| {
            writeback.buffer.as_ptr() == buffer.buffer.as_ptr()
                && writeback.guest_address == binding.base_address
                && writeback.byte_count == binding.byte_count
        }) {
            continue;
        }
        pending.push(PendingGlobalWriteback {
            buffer: buffer.buffer.clone(),
            guest_address: binding.base_address,
            byte_count: binding.byte_count,
        });
    }
    Ok(())
}

fn collect_pending_guest_write_ranges(
    buffers: &[UploadedGlobalBuffer],
    evaluation: &ShaderEvaluation,
    pending: &mut Vec<PendingGuestWriteRange>,
) -> Result<(), String> {
    if buffers.len() != evaluation.global_memory_bindings.len() {
        return Err("Metal/global buffer binding count mismatch".to_owned());
    }
    for (_, binding) in buffers
        .iter()
        .zip(&evaluation.global_memory_bindings)
        .filter(|(buffer, binding)| {
            binding.writable && (buffer.guest_memory || binding.write_back_to_guest)
        })
    {
        let byte_count = u64::try_from(binding.byte_count)
            .map_err(|_| "writable guest buffer byte count does not fit u64")?;
        let range = PendingGuestWriteRange {
            address: binding.base_address,
            byte_count,
        };
        if !pending.contains(&range) {
            pending.push(range);
        }
    }
    Ok(())
}

fn append_unique_global_writebacks(
    destination: &mut Vec<PendingGlobalWriteback>,
    source: Vec<PendingGlobalWriteback>,
) {
    for writeback in source {
        let duplicate = destination.iter().any(|existing| {
            existing.buffer.as_ptr() == writeback.buffer.as_ptr()
                && existing.guest_address == writeback.guest_address
                && existing.byte_count == writeback.byte_count
        });
        if !duplicate {
            destination.push(writeback);
        }
    }
}

fn append_unique_guest_write_ranges(
    destination: &mut Vec<PendingGuestWriteRange>,
    source: Vec<PendingGuestWriteRange>,
) {
    for range in source {
        if !destination.contains(&range) {
            destination.push(range);
        }
    }
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

fn read_guest_memory_wait_value(wait: GpuMemoryWait) -> u64 {
    // SAFETY: execute_guest_memory_wait validates the non-null address and required alignment
    // before calling this helper. The runtime retains the guest mapping while its ordered GPU
    // events are pending. Volatile access observes label writes from guest CPU threads.
    unsafe {
        if wait.is_64_bit {
            std::ptr::read_volatile(wait.address as *const u64)
        } else {
            u64::from(std::ptr::read_volatile(wait.address as *const u32))
        }
    }
}

fn validate_guest_memory_release(release: GpuMemoryRelease) -> Result<(), String> {
    let alignment = match release.value {
        GpuMemoryReleaseValue::Immediate32(_) => 4,
        GpuMemoryReleaseValue::Immediate64(_) | GpuMemoryReleaseValue::Timestamp => 8,
    };
    if release.address.is_multiple_of(alignment) {
        Ok(())
    } else {
        Err(format!(
            "{}-bit release destination {:#x} is unaligned",
            alignment * 8,
            release.address
        ))
    }
}

fn validate_guest_memory_write(write: &GpuMemoryWrite) -> Result<(), String> {
    if !write.address.is_multiple_of(4) {
        return Err(format!(
            "WRITE_DATA destination {:#x} is unaligned",
            write.address
        ));
    }
    if write.increment_address && !write.values.is_empty() {
        let final_index = u64::try_from(write.values.len() - 1)
            .map_err(|_| "WRITE_DATA value count does not fit u64")?;
        write
            .address
            .checked_add(
                final_index
                    .checked_mul(4)
                    .ok_or("WRITE_DATA destination offset overflowed")?,
            )
            .ok_or_else(|| "WRITE_DATA destination range overflowed".to_owned())?;
    }
    Ok(())
}

fn apply_pending_guest_memory_writes(pending: &[PendingGuestMemoryWrite]) -> Result<(), String> {
    static GPU_CLOCK_START: OnceLock<Instant> = OnceLock::new();
    for write in pending {
        match write {
            PendingGuestMemoryWrite::Release(release) => {
                validate_guest_memory_release(*release)?;
                // SAFETY: packet decoding admits only aligned, live process-local guest
                // destinations. The owning Metal batch has completed, and volatile writes publish
                // the ordered label update before submission completion becomes visible.
                unsafe {
                    match release.value {
                        GpuMemoryReleaseValue::Immediate32(value) => {
                            std::ptr::write_volatile(release.address as *mut u32, value);
                        }
                        GpuMemoryReleaseValue::Immediate64(value) => {
                            std::ptr::write_volatile(release.address as *mut u64, value);
                        }
                        GpuMemoryReleaseValue::Timestamp => {
                            let timestamp = u64::try_from(
                                GPU_CLOCK_START
                                    .get_or_init(Instant::now)
                                    .elapsed()
                                    .as_nanos(),
                            )
                            .unwrap_or(u64::MAX)
                            .max(1);
                            std::ptr::write_volatile(release.address as *mut u64, timestamp);
                        }
                    }
                }
            }
            PendingGuestMemoryWrite::Write(write) => {
                validate_guest_memory_write(write)?;
                // SAFETY: validation above proves aligned, non-overflowing destinations in the
                // guest mapping supplied by the runtime.
                unsafe {
                    for (index, value) in write.values.iter().copied().enumerate() {
                        let offset = if write.increment_address {
                            u64::try_from(index)
                                .map_err(|_| "WRITE_DATA index does not fit u64")?
                                .checked_mul(4)
                                .ok_or("WRITE_DATA destination offset overflowed")?
                        } else {
                            0
                        };
                        let address = write
                            .address
                            .checked_add(offset)
                            .ok_or("WRITE_DATA destination range overflowed")?;
                        std::ptr::write_volatile(address as *mut u32, value);
                    }
                }
            }
            PendingGuestMemoryWrite::Fill {
                address,
                value,
                dword_count,
            } => {
                let byte_count = dword_count
                    .checked_mul(4)
                    .ok_or("DMA fill byte count overflowed")?;
                if !address.is_multiple_of(4) || address.checked_add(byte_count).is_none() {
                    return Err(format!(
                        "DMA fill destination {address:#x} dwords={dword_count:#x} is invalid"
                    ));
                }
                // SAFETY: packet decoding admitted an aligned, non-overflowing guest range. The
                // owning Metal batch has retired, so publishing the repeated dwords here preserves
                // DMA ordering for following waits and CPU readers.
                unsafe {
                    for index in 0..*dword_count {
                        let offset = index
                            .checked_mul(4)
                            .ok_or("DMA fill destination offset overflowed")?;
                        std::ptr::write_volatile((*address + offset) as *mut u32, *value);
                    }
                }
            }
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
    let (pixel_format, bytes_per_element) =
        decode_sampled_texture_format(data_format, number_format).map_err(|_| {
        format!(
            "Metal image upload does not yet implement GFX10 format {data_format}/{number_format} (unified {unified_format})"
        )
    })?;
    let tile_mode = (words[3] >> 20) & 0x1f;
    let texture_type = (words[3] >> 28) & 0xf;
    let base_level = (words[3] >> 12) & 0xf;
    let last_level = (words[3] >> 16) & 0xf;
    let destination_select = words[3] & 0xfff;
    let descriptor_pitch_word = words.get(4).copied().unwrap_or_default();
    let has_explicit_descriptor_pitch =
        matches!(texture_type, 8 | 9 | 14) && descriptor_pitch_word != 0;
    let descriptor_pitch = if has_explicit_descriptor_pitch {
        (descriptor_pitch_word & 0x3fff)
            .checked_add(1)
            .ok_or("image pitch overflows")?
    } else {
        width
    };
    if trace_image_descriptors() && (tile_mode == 9 || (width >= 700 && height >= 500)) {
        static LOGGED_IMAGES: OnceLock<Mutex<HashSet<(u64, u32, u32, u32, u32, u32)>>> =
            OnceLock::new();
        let key = (
            address,
            width,
            height,
            unified_format,
            descriptor_pitch,
            tile_mode,
        );
        let mut logged = LOGGED_IMAGES
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if logged.len() < 128 && logged.insert(key) {
            eprintln!(
                "[REMu][METAL][IMAGE] mode={tile_mode} address={address:#x} extent={width}x{height} pitch={descriptor_pitch} format={data_format}/{number_format} unified={unified_format} type={texture_type}"
            );
        }
    }
    let pitch = if tile_mode == 0 && !has_explicit_descriptor_pitch {
        effective_linear_pitch(width, bytes_per_element).ok_or("linear image pitch overflows")?
    } else {
        descriptor_pitch
    };
    if address < 0x10_000 || width > 8_192 || height > 8_192 || pitch < width || pitch > 8_192 {
        return Err(format!(
            "invalid image range address={address:#x} extent={width}x{height} pitch={pitch}"
        ));
    }
    if !matches!(texture_type, 9..=11 | 13) {
        return Err(format!(
            "Metal image upload does not yet implement texture type {texture_type}"
        ));
    }
    let layer_count = if texture_type == 10 {
        (words.get(4).copied().unwrap_or_default() & 0x1fff)
            .checked_add(1)
            .ok_or("3D image depth overflows")?
    } else if texture_type == 11 {
        let last_array_slice = words.get(4).copied().unwrap_or_default() & 0x1fff;
        let base_array_slice = (words.get(4).copied().unwrap_or_default() >> 16) & 0x1fff;
        if base_array_slice > last_array_slice {
            return Err(format!(
                "invalid cube array slice range {base_array_slice}..={last_array_slice}"
            ));
        }
        let layers = last_array_slice - base_array_slice + 1;
        if !layers.is_multiple_of(6) {
            return Err(format!(
                "cube resource exposes {layers} faces instead of a multiple of six"
            ));
        }
        layers
    } else if texture_type == 13 {
        let slices = words.get(4).copied().unwrap_or_default();
        let last_array_slice = slices & 0x1fff;
        let base_array_slice = (slices >> 16) & 0x1fff;
        if base_array_slice > last_array_slice {
            return Err(format!(
                "invalid array slice range {base_array_slice}..={last_array_slice}"
            ));
        }
        last_array_slice - base_array_slice + 1
    } else {
        1
    };
    let maximum_mip_level = 31 - width.max(height).leading_zeros();
    if base_level > last_level || last_level > maximum_mip_level {
        return Err(format!(
            "invalid image mip range {base_level}..={last_level} for {width}x{height}"
        ));
    }
    if !matches!(tile_mode, 0 | 1 | 5 | 9 | 24 | 27) {
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
            pixel_format,
            bytes_per_pixel: bytes_per_element,
        },
        destination_select,
        base_mip_level: base_level,
        mip_level_count: last_level - base_level + 1,
        texture_type,
        layer_count,
    })
}

fn fallback_image_descriptor(binding: &ImageBinding) -> Result<DecodedImageDescriptor, String> {
    let unified_format = binding
        .resource_descriptor
        .get(1)
        .map_or(0, |word| (word >> 20) & 0x1ff);
    let decoded_format = remu_shader::decode_unified_format(unified_format);
    let (pixel_format, bytes_per_pixel) = decoded_format
        .and_then(|(data_format, number_format)| {
            decode_sampled_texture_format(data_format, number_format).ok()
        })
        .unwrap_or_else(
            || match decoded_format.map(|(_, number_format)| number_format) {
                Some(4) => (MTLPixelFormat::RGBA8Uint, 4),
                Some(5) => (MTLPixelFormat::RGBA8Sint, 4),
                _ => (MTLPixelFormat::RGBA8Unorm, 4),
            },
        );
    let storage = is_storage_image_opcode(binding.opcode);
    let encoded_texture_type = binding
        .resource_descriptor
        .get(3)
        .map_or(0, |word| (word >> 28) & 0xf);
    // Match the texture type emitted by remu-shader-metal. Sample instructions select their
    // Metal dimension from the instruction itself, while storage instructions use the descriptor
    // only to distinguish 3D from 2D.
    let texture_type = if storage {
        if encoded_texture_type == 10 { 10 } else { 9 }
    } else {
        match binding.control.dimension {
            2 => 10,
            3 => 11,
            _ => 9,
        }
    };
    let layer_count = if texture_type == 11 { 6 } else { 1 };
    Ok(DecodedImageDescriptor {
        guest: GuestTextureDescriptor {
            address: 0,
            width: 1,
            height: 1,
            pitch: 1,
            tile_mode: 0,
            pixel_format,
            bytes_per_pixel,
        },
        destination_select: 0,
        base_mip_level: 0,
        mip_level_count: 1,
        texture_type,
        layer_count,
    })
}

fn trace_image_descriptors() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("REMU_TRACE_IMAGES").is_some_and(|value| value == "1"))
}

fn trace_target_sampling() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_TARGET_SAMPLING").is_some_and(|value| value == "1")
    })
}

fn capture_target_sample_textures() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_CAPTURE_TARGET_SAMPLES").is_some_and(|value| value == "1")
    })
}

fn trace_scanout_writes() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_SCANOUT_WRITES").is_some_and(|value| value == "1")
    })
}

fn diagnostic_dump_target_before_pixel_shader() -> Option<u64> {
    static ADDRESS: OnceLock<Option<u64>> = OnceLock::new();
    *ADDRESS.get_or_init(|| {
        std::env::var("REMU_DUMP_TARGET_BEFORE_PS")
            .ok()
            .and_then(|value| parse_diagnostic_hex_address(&value))
    })
}

const fn should_record_diagnostic_target_writer(
    dump_target_before_pixel_shader: bool,
    trace_scanout_writes: bool,
    is_known_scanout: bool,
) -> bool {
    dump_target_before_pixel_shader || (trace_scanout_writes && is_known_scanout)
}

fn parse_diagnostic_hex_address(value: &str) -> Option<u64> {
    let value = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
        .unwrap_or(value);
    (!value.is_empty())
        .then(|| u64::from_str_radix(value, 16).ok())
        .flatten()
}

fn diagnostic_scanout_writers() -> &'static Mutex<HashMap<u64, DiagnosticScanoutWriter>> {
    static WRITERS: OnceLock<Mutex<HashMap<u64, DiagnosticScanoutWriter>>> = OnceLock::new();
    WRITERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn diagnostic_scanout_writer(address: u64) -> Option<DiagnosticScanoutWriter> {
    diagnostic_scanout_writers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&address)
        .copied()
}

fn record_diagnostic_scanout_writer(address: u64, writer: DiagnosticScanoutWriter) {
    diagnostic_scanout_writers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(address, writer);
}

fn trace_msaa_writes() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_MSAA_WRITES").is_some_and(|value| value == "1")
    })
}

fn diagnostic_disable_depth_and_culling() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DIAGNOSTIC_RASTER").is_some_and(|value| value == "1")
    })
}

fn diagnostic_disable_culling() -> bool {
    diagnostic_disable_depth_and_culling() || {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var_os("REMU_DIAGNOSTIC_CULL").is_some_and(|value| value == "1")
        })
    }
}

fn diagnostic_disable_depth() -> bool {
    diagnostic_disable_depth_and_culling() || {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var_os("REMU_DIAGNOSTIC_DEPTH").is_some_and(|value| value == "1")
        })
    }
}

fn diagnostic_force_visible_clip_depth() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var_os("REMU_DIAGNOSTIC_CLIP").is_some_and(|value| value == "1"))
}

fn diagnostic_force_msaa_pixel_color() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DIAGNOSTIC_MSAA_COLOR").is_some_and(|value| value == "1")
    })
}

fn diagnostic_force_msaa_vertex_pattern() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DIAGNOSTIC_MSAA_PATTERN").is_some_and(|value| value == "1")
    })
}

fn diagnostic_force_vertex_fetch_id() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DIAGNOSTIC_VERTEX_FETCH_ID").is_some_and(|value| value == "1")
    })
}

fn diagnostic_capture_vertex_position() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DIAGNOSTIC_CAPTURE_VERTEX_POSITION")
            .is_some_and(|value| value == "1")
    })
}

fn diagnostic_capture_pixel_output() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DIAGNOSTIC_CAPTURE_PIXEL_OUTPUT").is_some_and(|value| value == "1")
    })
}

fn diagnostic_disable_blending() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var_os("REMU_DIAGNOSTIC_BLEND").is_some_and(|value| value == "1"))
}

fn diagnostic_use_draw_merged_wave_counts() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DIAGNOSTIC_MERGED_COUNTS").is_some_and(|value| value == "1")
    })
}

fn diagnostic_dump_target_address() -> Option<u64> {
    static ADDRESS: OnceLock<Option<u64>> = OnceLock::new();
    *ADDRESS.get_or_init(|| {
        let value = std::env::var("REMU_DUMP_TARGET_ADDRESS").ok()?;
        if value.eq_ignore_ascii_case("auto") {
            return Some(0);
        }
        let value = value.strip_prefix("0x").unwrap_or(&value);
        u64::from_str_radix(value, 16).ok()
    })
}

fn diagnostic_dump_target_pixel_shader() -> Option<u64> {
    static ADDRESS: OnceLock<Option<u64>> = OnceLock::new();
    *ADDRESS.get_or_init(|| {
        let value = std::env::var("REMU_DUMP_TARGET_PS").ok()?;
        let value = value.strip_prefix("0x").unwrap_or(&value);
        u64::from_str_radix(value, 16).ok()
    })
}

fn diagnostic_dump_target_pixel_instruction_count() -> Option<usize> {
    static COUNT: OnceLock<Option<usize>> = OnceLock::new();
    *COUNT.get_or_init(|| {
        std::env::var("REMU_DUMP_TARGET_PS_INSNS")
            .ok()
            .and_then(|value| value.parse().ok())
    })
}

fn diagnostic_dump_complex_msaa() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DUMP_COMPLEX_MSAA").is_some_and(|value| value == "1")
    })
}

fn diagnostic_texture_writer_enabled() -> bool {
    diagnostic_dump_target_address().is_some()
        || diagnostic_dump_target_pixel_shader().is_some()
        || diagnostic_dump_target_pixel_instruction_count().is_some()
        || diagnostic_dump_complex_msaa()
}

fn diagnostic_dump_target_minimum_draw() -> u64 {
    static DRAW: OnceLock<u64> = OnceLock::new();
    *DRAW.get_or_init(|| {
        std::env::var("REMU_DUMP_TARGET_MIN_DRAW")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(131_072)
    })
}

fn diagnostic_dump_target_match_index() -> usize {
    static INDEX: OnceLock<usize> = OnceLock::new();
    *INDEX.get_or_init(|| {
        std::env::var("REMU_DUMP_TARGET_MATCH_INDEX")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    })
}

fn diagnostic_trace_target_contents() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("REMU_TRACE_TARGET_CONTENTS")
            .ok()
            .is_some_and(|value| value != "0")
    })
}

fn diagnostic_dump_video_draw() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("REMU_DUMP_VIDEO_DRAW")
            .ok()
            .is_some_and(|value| value != "0")
    })
}

fn diagnostic_trace_all_target_contents() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("REMU_TRACE_ALL_TARGET_CONTENTS")
            .ok()
            .is_some_and(|value| value != "0")
    })
}

fn diagnostic_trace_target_maximum_draw() -> u64 {
    static DRAW: OnceLock<u64> = OnceLock::new();
    *DRAW.get_or_init(|| {
        std::env::var("REMU_TRACE_TARGET_MAX_DRAW")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(u64::MAX)
    })
}

fn diagnostic_global_binding_head(binding: &GlobalMemoryBinding) -> Vec<u8> {
    let byte_count = binding.byte_count.min(32);
    if byte_count == 0 {
        return Vec::new();
    }
    if let Some(bytes) = binding.data.get(..byte_count) {
        return bytes.to_vec();
    }
    if binding.base_address < 0x10_000 {
        return Vec::new();
    }
    // SAFETY: Shader evaluation has already validated this exact live guest binding and the
    // presenter consumes it synchronously while the draw snapshot owns the allocation.
    unsafe { std::slice::from_raw_parts(binding.base_address as *const u8, byte_count).to_vec() }
}

fn record_diagnostic_texture_writer(
    descriptor: GuestTextureDescriptor,
    stage: &'static str,
    sequence: u64,
    shader: u64,
    sampled: Vec<GuestTextureDescriptor>,
) {
    if !diagnostic_texture_writer_enabled() {
        return;
    }
    diagnostic_texture_writers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(descriptor, (stage, sequence, shader, sampled));
}

type DiagnosticTextureWriter = (&'static str, u64, u64, Vec<GuestTextureDescriptor>);

fn diagnostic_texture_writers()
-> &'static Mutex<HashMap<GuestTextureDescriptor, DiagnosticTextureWriter>> {
    static WRITERS: OnceLock<Mutex<HashMap<GuestTextureDescriptor, DiagnosticTextureWriter>>> =
        OnceLock::new();
    WRITERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn diagnostic_texture_writer(
    descriptor: GuestTextureDescriptor,
) -> Option<DiagnosticTextureWriter> {
    diagnostic_texture_writers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&descriptor)
        .cloned()
}

fn diagnostic_overlapping_texture_writers(
    descriptor: GuestTextureDescriptor,
) -> Vec<(GuestTextureDescriptor, DiagnosticTextureWriter)> {
    diagnostic_texture_writers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .filter(|(candidate, _)| {
            **candidate != descriptor && guest_texture_descriptors_overlap(**candidate, descriptor)
        })
        .map(|(candidate, writer)| (*candidate, writer.clone()))
        .collect()
}

fn diagnostic_related_texture_writers(
    descriptor: GuestTextureDescriptor,
) -> Vec<(GuestTextureDescriptor, DiagnosticTextureWriter)> {
    let mut writers = diagnostic_texture_writers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .filter(|(candidate, _)| {
            candidate.width == descriptor.width
                && candidate.height == descriptor.height
                && candidate.pixel_format == descriptor.pixel_format
        })
        .map(|(candidate, writer)| (*candidate, writer.clone()))
        .collect::<Vec<_>>();
    writers.sort_unstable_by_key(|(_, writer)| writer.1);
    if writers.len() > 64 {
        writers.drain(..writers.len() - 64);
    }
    writers
}

fn record_diagnostic_memory_copy(copy: GpuMemoryCopy) {
    if !diagnostic_texture_writer_enabled() {
        return;
    }
    const CAPACITY: usize = 8_192;
    let mut copies = diagnostic_memory_copies()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if copies.len() == CAPACITY {
        copies.pop_front();
    }
    copies.push_back(copy);
}

fn diagnostic_memory_copies() -> &'static Mutex<VecDeque<GpuMemoryCopy>> {
    static COPIES: OnceLock<Mutex<VecDeque<GpuMemoryCopy>>> = OnceLock::new();
    COPIES.get_or_init(|| Mutex::new(VecDeque::new()))
}

fn diagnostic_overlapping_memory_copies(descriptor: GuestTextureDescriptor) -> Vec<GpuMemoryCopy> {
    let Some(byte_count) = guest_texture_byte_count(descriptor) else {
        return Vec::new();
    };
    diagnostic_memory_copies()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .rev()
        .filter(|copy| {
            guest_ranges_overlap(
                descriptor.address,
                byte_count,
                copy.destination_address,
                copy.byte_count,
            )
        })
        .take(16)
        .copied()
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct DiagnosticGlobalWriter {
    stage: &'static str,
    sequence: u64,
    shader: u64,
    address: u64,
    byte_count: u64,
}

fn record_diagnostic_global_writers(
    stage: &'static str,
    sequence: u64,
    shader: u64,
    evaluation: &ShaderEvaluation,
) {
    if !diagnostic_texture_writer_enabled() {
        return;
    }
    let writable = evaluation
        .global_memory_bindings
        .iter()
        .filter(|binding| binding.writable && binding.base_address >= 0x10_000)
        .filter_map(|binding| {
            Some(DiagnosticGlobalWriter {
                stage,
                sequence,
                shader,
                address: binding.base_address,
                byte_count: u64::try_from(binding.byte_count).ok()?,
            })
        })
        .collect::<Vec<_>>();
    if writable.is_empty() {
        return;
    }
    const CAPACITY: usize = 16_384;
    let mut writers = diagnostic_global_writers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for writer in writable {
        if writers.len() == CAPACITY {
            writers.pop_front();
        }
        writers.push_back(writer);
    }
}

fn diagnostic_global_writers() -> &'static Mutex<VecDeque<DiagnosticGlobalWriter>> {
    static WRITERS: OnceLock<Mutex<VecDeque<DiagnosticGlobalWriter>>> = OnceLock::new();
    WRITERS.get_or_init(|| Mutex::new(VecDeque::new()))
}

fn diagnostic_overlapping_global_writers(
    descriptor: GuestTextureDescriptor,
) -> Vec<DiagnosticGlobalWriter> {
    let Some(byte_count) = guest_texture_byte_count(descriptor) else {
        return Vec::new();
    };
    diagnostic_global_writers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .rev()
        .filter(|writer| {
            guest_ranges_overlap(
                descriptor.address,
                byte_count,
                writer.address,
                writer.byte_count,
            )
        })
        .take(32)
        .copied()
        .collect()
}

fn evaluation_samples_large_linear_video(evaluation: &ShaderEvaluation) -> bool {
    sampled_image_bindings(evaluation).any(|binding| {
        decode_image_descriptor(&binding.resource_descriptor).is_ok_and(|descriptor| {
            descriptor.guest.tile_mode == 0
                && descriptor.guest.width >= 700
                && descriptor.guest.height >= 500
                && matches!(
                    descriptor.guest.pixel_format,
                    MTLPixelFormat::R8Unorm | MTLPixelFormat::RG8Unorm
                )
        })
    })
}

fn dump_msl_pixel_filter_matches(pixel_shader_address: u64) -> bool {
    let Some(filter) = std::env::var_os("REMU_DUMP_MSL_PS") else {
        return true;
    };
    let filter = filter.to_string_lossy();
    let filter = filter.strip_prefix("0x").unwrap_or(&filter);
    u64::from_str_radix(filter, 16).is_ok_and(|filter| filter == pixel_shader_address)
}

fn dump_msl_compute_filter_matches(compute_shader_address: u64) -> bool {
    let Some(filter) = std::env::var_os("REMU_DUMP_MSL_CS") else {
        return false;
    };
    let filter = filter.to_string_lossy();
    let filter = filter.strip_prefix("0x").unwrap_or(&filter);
    u64::from_str_radix(filter, 16).is_ok_and(|filter| filter == compute_shader_address)
}

fn dump_msl_instruction_filter_matches(pixel_instruction_count: usize) -> bool {
    let Some(filter) = std::env::var_os("REMU_DUMP_MSL_PS_INSNS") else {
        return true;
    };
    filter
        .to_string_lossy()
        .parse::<usize>()
        .is_ok_and(|filter| filter == pixel_instruction_count)
}

fn decode_sampled_texture_format(
    data_format: u32,
    number_format: u32,
) -> Result<(MTLPixelFormat, u32), String> {
    match (data_format, number_format) {
        (10, 9) => Ok((MTLPixelFormat::RGBA8Unorm_sRGB, 4)),
        (169, _) => Ok((MTLPixelFormat::BC1_RGBA, 8)),
        (170, _) => Ok((MTLPixelFormat::BC1_RGBA_sRGB, 8)),
        (171, _) => Ok((MTLPixelFormat::BC2_RGBA, 16)),
        (172, _) => Ok((MTLPixelFormat::BC2_RGBA_sRGB, 16)),
        (173, _) => Ok((MTLPixelFormat::BC3_RGBA, 16)),
        (174, _) => Ok((MTLPixelFormat::BC3_RGBA_sRGB, 16)),
        (175, _) => Ok((MTLPixelFormat::BC4_RUnorm, 8)),
        (176, _) => Ok((MTLPixelFormat::BC4_RSnorm, 8)),
        (177, _) => Ok((MTLPixelFormat::BC5_RGUnorm, 16)),
        (178, _) => Ok((MTLPixelFormat::BC5_RGSnorm, 16)),
        (179, _) => Ok((MTLPixelFormat::BC6H_RGBUfloat, 16)),
        (180, _) => Ok((MTLPixelFormat::BC6H_RGBFloat, 16)),
        (181, _) => Ok((MTLPixelFormat::BC7_RGBAUnorm, 16)),
        (182, _) => Ok((MTLPixelFormat::BC7_RGBAUnorm_sRGB, 16)),
        _ => decode_render_target_format(data_format, number_format, 0)
            .map(|format| (format.pixel_format, format.bytes_per_pixel)),
    }
}

fn compressed_texture_block_extent(format: MTLPixelFormat) -> Option<(usize, usize)> {
    match format {
        MTLPixelFormat::BC1_RGBA
        | MTLPixelFormat::BC1_RGBA_sRGB
        | MTLPixelFormat::BC2_RGBA
        | MTLPixelFormat::BC2_RGBA_sRGB
        | MTLPixelFormat::BC3_RGBA
        | MTLPixelFormat::BC3_RGBA_sRGB
        | MTLPixelFormat::BC4_RUnorm
        | MTLPixelFormat::BC4_RSnorm
        | MTLPixelFormat::BC5_RGUnorm
        | MTLPixelFormat::BC5_RGSnorm
        | MTLPixelFormat::BC6H_RGBFloat
        | MTLPixelFormat::BC6H_RGBUfloat
        | MTLPixelFormat::BC7_RGBAUnorm
        | MTLPixelFormat::BC7_RGBAUnorm_sRGB => Some((4, 4)),
        _ => None,
    }
}

fn texture_element_layout(
    descriptor: GuestTextureDescriptor,
) -> Result<(usize, usize, usize, usize), String> {
    let width =
        usize::try_from(descriptor.width).map_err(|_| "texture width does not fit usize")?;
    let height =
        usize::try_from(descriptor.height).map_err(|_| "texture height does not fit usize")?;
    let pitch =
        usize::try_from(descriptor.pitch).map_err(|_| "texture pitch does not fit usize")?;
    let bytes_per_element = usize::try_from(descriptor.bytes_per_pixel)
        .map_err(|_| "texture element byte count does not fit usize")?;
    if let Some((block_width, block_height)) =
        compressed_texture_block_extent(descriptor.pixel_format)
    {
        Ok((
            width.div_ceil(block_width),
            height.div_ceil(block_height),
            pitch.div_ceil(block_width),
            bytes_per_element,
        ))
    } else {
        Ok((width, height, pitch, bytes_per_element))
    }
}

#[cfg(test)]
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

#[cfg(test)]
fn prepare_sampled_texture_upload(
    descriptor: GuestTextureDescriptor,
    mut linear: Vec<u8>,
    destination_select: u32,
) -> Result<(GuestTextureDescriptor, Vec<u8>), String> {
    if image_destination_select_is_native(descriptor.pixel_format, destination_select) {
        return Ok((descriptor, linear));
    }
    if descriptor.bytes_per_pixel == 4
        && matches!(
            descriptor.pixel_format,
            MTLPixelFormat::RGBA8Unorm
                | MTLPixelFormat::RGBA8Unorm_sRGB
                | MTLPixelFormat::RGBA8Snorm
                | MTLPixelFormat::RGBA8Uint
                | MTLPixelFormat::RGBA8Sint
                | MTLPixelFormat::BGRA8Unorm
                | MTLPixelFormat::BGRA8Unorm_sRGB
        )
    {
        apply_image_destination_select(&mut linear, destination_select)?;
        return Ok((descriptor, linear));
    }
    if matches!(
        descriptor.pixel_format,
        MTLPixelFormat::R8Unorm | MTLPixelFormat::R8Unorm_sRGB
    ) {
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
        let mut rgba = Vec::with_capacity(linear.len().saturating_mul(4));
        for red in linear {
            for &selector in &selectors {
                rgba.push(match selector {
                    0 | 5 | 6 => 0,
                    1 | 7 => u8::MAX,
                    4 => red,
                    _ => unreachable!("validated destination selector"),
                });
            }
        }
        return Ok((
            GuestTextureDescriptor {
                pixel_format: if descriptor.pixel_format == MTLPixelFormat::R8Unorm_sRGB {
                    MTLPixelFormat::RGBA8Unorm_sRGB
                } else {
                    MTLPixelFormat::RGBA8Unorm
                },
                bytes_per_pixel: 4,
                ..descriptor
            },
            rgba,
        ));
    }
    Err(format!(
        "destination select {destination_select:#x} is not representable for {:?}",
        descriptor.pixel_format
    ))
}

#[cfg(test)]
fn image_destination_select_is_native(
    pixel_format: MTLPixelFormat,
    destination_select: u32,
) -> bool {
    if destination_select == 0 {
        return true;
    }
    let native_select = match pixel_format {
        MTLPixelFormat::R8Unorm
        | MTLPixelFormat::R8Snorm
        | MTLPixelFormat::R8Uint
        | MTLPixelFormat::R8Sint
        | MTLPixelFormat::R8Unorm_sRGB
        | MTLPixelFormat::R16Unorm
        | MTLPixelFormat::R16Snorm
        | MTLPixelFormat::R16Uint
        | MTLPixelFormat::R16Sint
        | MTLPixelFormat::R16Float
        | MTLPixelFormat::R32Uint
        | MTLPixelFormat::R32Sint
        | MTLPixelFormat::R32Float => 0x204,
        MTLPixelFormat::RG8Unorm
        | MTLPixelFormat::RG8Snorm
        | MTLPixelFormat::RG8Uint
        | MTLPixelFormat::RG8Sint
        | MTLPixelFormat::RG8Unorm_sRGB
        | MTLPixelFormat::RG16Unorm
        | MTLPixelFormat::RG16Snorm
        | MTLPixelFormat::RG16Uint
        | MTLPixelFormat::RG16Sint
        | MTLPixelFormat::RG16Float
        | MTLPixelFormat::RG32Uint
        | MTLPixelFormat::RG32Sint
        | MTLPixelFormat::RG32Float => 0x22c,
        MTLPixelFormat::RG11B10Float | MTLPixelFormat::RGB9E5Float => 0x3ac,
        _ => 0xfac,
    };
    destination_select == native_select
}

fn upload_image_textures(
    device: &Device,
    queue: &metal::CommandQueue,
    evaluation: &ShaderEvaluation,
    resident_guest_textures: &mut HashMap<GuestTextureDescriptor, ResidentGuestTexture>,
    multisample_textures: &HashMap<MultisampleTextureKey, MultisampleResidentTexture>,
    sampled_texture_cache: &mut HashMap<SampledTextureKey, CachedSampledTexture>,
    sampled_texture_write_watches: &mut HashMap<SampledTextureKey, Arc<GuestTextureWriteWatch>>,
    fallback_texture_cache: &mut HashMap<SampledTextureKey, Texture>,
    sampled_texture_admission_order: &mut VecDeque<(SampledTextureKey, u64)>,
    next_sampled_texture_admission_id: &mut u64,
) -> Result<Vec<UploadedImageTexture>, String> {
    canonical_texture_bindings(evaluation)
        .into_iter()
        .map(|binding| {
            let writable = is_writable_image_opcode(binding.opcode);
            let storage = is_storage_image_opcode(binding.opcode);
            let comparison = image_is_comparison(binding.opcode);
            let decoded = decode_image_descriptor(&binding.resource_descriptor);
            let fallback = decoded.is_err();
            let descriptor = decoded
                .or_else(|_| fallback_image_descriptor(binding))
                .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
            if trace_target_sampling()
                && descriptor.guest.width == 1_920
                && descriptor.guest.height == 1_080
            {
                static LOGGED: OnceLock<Mutex<HashSet<(u64, u32, u32, u32, bool, bool)>>> =
                    OnceLock::new();
                let normalized_coordinates =
                    sampler_uses_normalized_coordinates(&binding.sampler_descriptor);
                let key = (
                    descriptor.guest.address,
                    descriptor.guest.tile_mode,
                    descriptor.destination_select,
                    descriptor.mip_level_count,
                    comparison,
                    normalized_coordinates,
                );
                let mut logged = LOGGED
                    .get_or_init(|| Mutex::new(HashSet::new()))
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if logged.len() < 128 && logged.insert(key) {
                    let byte_count = guest_texture_byte_count(descriptor.guest);
                    let mapping = byte_count.and_then(|byte_count| {
                        guest_shared_mapping_identity(descriptor.guest.address, byte_count)
                    });
                    let mapping_start =
                        guest_shared_mapping_identity(descriptor.guest.address, 1);
                    let mapping_end = byte_count
                        .and_then(|byte_count| descriptor.guest.address.checked_add(byte_count - 1))
                        .and_then(|address| guest_shared_mapping_identity(address, 1));
                    let exact = resident_guest_textures
                        .get(&descriptor.guest)
                        .map(|resident| (resident.gpu_written, resident.content_version));
                    let aliases = resident_guest_textures
                        .values()
                        .filter(|resident| {
                            guest_texture_descriptors_overlap(
                                resident.descriptor,
                                descriptor.guest,
                            )
                        })
                        .map(|resident| {
                            (
                                resident.descriptor.address,
                                resident.descriptor.width,
                                resident.descriptor.height,
                                resident.descriptor.pixel_format,
                                resident.descriptor.tile_mode,
                                resident.gpu_written,
                                resident.content_version,
                            )
                        })
                        .collect::<Vec<_>>();
                    let multisample_candidates = multisample_textures
                        .iter()
                        .filter(|(key, _)| {
                            guest_texture_descriptors_have_same_layout(
                                descriptor.guest,
                                key.descriptor,
                            )
                        })
                        .map(|(key, multisample)| {
                            (
                                key.descriptor.address,
                                key.sample_count,
                                multisample.initialized,
                                guest_texture_descriptors_overlap(
                                    descriptor.guest,
                                    key.descriptor,
                                ),
                                resident_guest_textures
                                    .get(&key.descriptor)
                                    .map(|resident| {
                                        (resident.gpu_written, resident.content_version)
                                    }),
                            )
                        })
                        .collect::<Vec<_>>();
                    eprintln!(
                        "[REMu][METAL][SAMPLE] pc={:#x} op={} guest={:#x} 1920x1080 format={:?} tile={} select={:#x} mips={} type={} comparison={} normalized={} sampler={:08x?} bytes={byte_count:x?} mapping={mapping:?} endpoints={mapping_start:?}/{mapping_end:?} exact={exact:?} aliases={aliases:?} msaa={multisample_candidates:?}",
                        binding.pc,
                        binding.opcode,
                        descriptor.guest.address,
                        descriptor.guest.pixel_format,
                        descriptor.guest.tile_mode,
                        descriptor.destination_select,
                        descriptor.mip_level_count,
                        descriptor.texture_type,
                        comparison,
                        normalized_coordinates,
                        binding.sampler_descriptor,
                    );
                }
            }
            let underlying_mip_count = descriptor
                .base_mip_level
                .checked_add(descriptor.mip_level_count.max(1))
                .ok_or("sampled resident mip range overflows")?;
            if matches!(descriptor.texture_type, 9 | 10) {
                let replaced = if let Some(resident) = resident_guest_textures
                .get_mut(&descriptor.guest)
                .filter(|resident| {
                    resident.texture_type == descriptor.texture_type
                        && resident.layer_count == descriptor.layer_count
                        && resident.mip_level_count >= underlying_mip_count
                })
                {
                    refresh_resident_texture_write_watch(resident)?
                        == WriteWatchRefresh::Replaced
                } else {
                    false
                };
                if replaced {
                    resident_guest_textures.remove(&descriptor.guest);
                }
            }
            let multisample_resolve = (!comparison && descriptor.texture_type == 9)
                .then(|| {
                    multisample_textures
                        .iter()
                        .filter(|(_, multisample)| multisample.initialized)
                        .filter(|(key, _)| {
                            render_resolve_candidate_matches(
                                descriptor.guest,
                                key.descriptor,
                                key.sample_count,
                            )
                        })
                        .filter_map(|(key, _)| {
                            let resident = resident_guest_textures.get(&key.descriptor)?;
                            resident.gpu_written.then_some((
                                resident.content_version,
                                key.descriptor,
                                resident.texture.clone(),
                            ))
                        })
                        .max_by_key(|(version, _, _)| *version)
                })
                .flatten();
            let resident_descriptor =
                (!comparison && matches!(descriptor.texture_type, 9 | 10))
                    .then(|| {
                        resident_guest_textures
                            .get(&descriptor.guest)
                            .filter(|resident| {
                                resident.texture_type == descriptor.texture_type
                                    && resident.layer_count == descriptor.layer_count
                                    && resident.mip_level_count >= underlying_mip_count
                            })
                            .map(|resident| resident.descriptor)
                            .or_else(|| {
                                (descriptor.texture_type == 9).then(|| {
                                    resident_guest_textures
                                        .values()
                                        .filter(|resident| {
                                            resident.texture_type == 9
                                                && resident.layer_count == 1
                                                && resident.mip_level_count
                                                    >= underlying_mip_count
                                                && (guest_texture_descriptors_share_storage(
                                                    resident.descriptor,
                                                    descriptor.guest,
                                                ) || resident_component_reordered_sample_is_eligible(
                                                    resident,
                                                    descriptor,
                                                    comparison,
                                                ))
                                        })
                                        .max_by_key(|resident| resident.content_version)
                                        .map(|resident| resident.descriptor)
                                })?
                            })
                    })
                    .flatten();
            let resident_resolve = resident_descriptor.and_then(|resident_descriptor| {
                let resident = resident_guest_textures.get_mut(&resident_descriptor)?;
                let version = resident.content_version;
                resident_texture_for_sample(resident, descriptor)
                    .map(|texture| (version, texture))
            });
            // A guest descriptor can retain an exact single-sample resident from an earlier
            // render phase while the current frame is produced through a distinct MSAA target
            // with the same layout. Address equality alone is therefore not sufficient: choose
            // whichever compatible GPU image was written most recently. This also naturally
            // retains an exact post-process result when it is newer than the MSAA resolve.
            let select_multisample = match (&resident_resolve, &multisample_resolve) {
                (Some((resident_version, _)), Some((multisample_version, _, _))) => {
                    multisample_version > resident_version
                }
                (None, Some(_)) => true,
                _ => false,
            };
            if trace_target_sampling()
                && descriptor.guest.width == 1_920
                && descriptor.guest.height == 1_080
                && select_multisample
                && let Some((version, source, _)) = multisample_resolve.as_ref()
            {
                static LOGGED_RESOLVES: OnceLock<Mutex<HashSet<(u64, u64)>>> = OnceLock::new();
                let key = (descriptor.guest.address, source.address);
                let mut logged = LOGGED_RESOLVES
                    .get_or_init(|| Mutex::new(HashSet::new()))
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if logged.len() < 256 && logged.insert(key) {
                    eprintln!(
                        "[REMu][METAL][MSAA-SAMPLE] guest={:#x} source={:#x} version={} format={:?} tile={}",
                        descriptor.guest.address,
                        source.address,
                        version,
                        source.pixel_format,
                        source.tile_mode,
                    );
                }
            }
            let render_resolve = if select_multisample {
                multisample_resolve
                    .as_ref()
                    .map(|(_, _, texture)| texture.clone())
            } else {
                resident_resolve
                    .as_ref()
                    .map(|(_, texture)| texture.clone())
            };
            let texture = if fallback {
                let key = SampledTextureKey {
                    descriptor: descriptor.guest,
                    destination_select: descriptor.destination_select,
                    base_mip_level: descriptor.base_mip_level,
                    mip_level_count: descriptor.mip_level_count,
                    texture_type: descriptor.texture_type,
                    layer_count: descriptor.layer_count,
                    comparison,
                };
                if let Some(texture) = fallback_texture_cache.get(&key) {
                    texture.clone()
                } else {
                    let (_, _, _, bytes_per_element) =
                        texture_element_layout(descriptor.guest)?;
                    let byte_count = bytes_per_element
                        .checked_mul(
                            usize::try_from(descriptor.layer_count)
                                .map_err(|_| "fallback image layer count does not fit usize")?,
                        )
                        .ok_or("fallback image byte count overflows")?;
                    let mut linear = vec![0; byte_count];
                    if matches!(
                        descriptor.guest.pixel_format,
                        MTLPixelFormat::RGBA8Unorm
                            | MTLPixelFormat::RGBA8Unorm_sRGB
                            | MTLPixelFormat::RGBA8Uint
                            | MTLPixelFormat::RGBA8Sint
                    ) {
                        for texel in linear.chunks_exact_mut(4) {
                            texel[3] = u8::MAX;
                        }
                    }
                    let texture = create_sampled_guest_texture(
                        device,
                        queue,
                        descriptor.guest,
                        &linear,
                        0,
                        1,
                        descriptor.texture_type,
                        descriptor.layer_count,
                        comparison,
                    )?;
                    texture.set_label("REMu fallback guest image");
                    fallback_texture_cache.insert(key, texture.clone());
                    texture
                }
            } else if storage && descriptor.texture_type == 9 {
                if descriptor.base_mip_level != 0 || descriptor.mip_level_count != 1 {
                    ensure_mipmapped_resident_texture_in_map(
                        device,
                        descriptor,
                        resident_guest_textures,
                    )?
                } else {
                    ensure_resident_texture_in_map(
                        device,
                        descriptor.guest,
                        resident_guest_textures,
                    )?
                }
            } else if let Some(texture) = render_resolve {
                texture
            } else {
                let key = SampledTextureKey {
                    descriptor: descriptor.guest,
                    destination_select: descriptor.destination_select,
                    base_mip_level: descriptor.base_mip_level,
                    mip_level_count: descriptor.mip_level_count,
                    texture_type: descriptor.texture_type,
                    layer_count: descriptor.layer_count,
                    comparison,
                };
                let replaced = if let Some(cached) = sampled_texture_cache.get_mut(&key) {
                    refresh_sampled_texture_write_watch(
                        key,
                        cached,
                        sampled_texture_write_watches,
                    )?
                        == WriteWatchRefresh::Replaced
                } else {
                    false
                };
                if replaced {
                    if let Some(retired) = sampled_texture_cache.remove(&key) {
                        retire_sampled_texture(retired);
                    }
                }
                if let Some(cached) = sampled_texture_cache.get(&key) {
                    cached.texture.clone()
                } else {
                    let write_watch = persistent_sampled_texture_write_watch(
                        key,
                        descriptor,
                        sampled_texture_write_watches,
                    )
                        .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
                    let raw = snapshot_watched_sampled_texture_raw(descriptor, &write_watch)
                        .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
                    let texture =
                        if sampled_texture_uses_exact_mode27_mips(descriptor) {
                            create_exact_mode27_sampled_mip_view(
                                device,
                                descriptor,
                                &raw,
                                comparison,
                            )
                            .map_err(|error| {
                                format!("image at PC {:#x}: {error}", binding.pc)
                            })?
                        } else {
                            let linear =
                                decode_sampled_texture_linear_from_raw(descriptor, &raw).map_err(
                                    |error| format!("image at PC {:#x}: {error}", binding.pc),
                                )?;
                            create_sampled_guest_texture(
                                device,
                                queue,
                                descriptor.guest,
                                &linear,
                                descriptor.base_mip_level,
                                descriptor.mip_level_count,
                                descriptor.texture_type,
                                descriptor.layer_count,
                                comparison,
                            )?
                        };
                    texture.set_label(&format!(
                        "REMu sampled guest image {:#x}",
                        descriptor.guest.address
                    ));
                    while sampled_texture_cache.len() >= SAMPLED_TEXTURE_CACHE_CAPACITY {
                        let Some((old_key, admission_id)) =
                            sampled_texture_admission_order.pop_front()
                        else {
                            break;
                        };
                        let is_current = sampled_texture_cache
                            .get(&old_key)
                            .is_some_and(|cached| cached.admission_id == admission_id);
                        if is_current {
                            if let Some(retired) = sampled_texture_cache.remove(&old_key) {
                                retire_sampled_texture(retired);
                            }
                        }
                    }
                    let admission_id = *next_sampled_texture_admission_id;
                    *next_sampled_texture_admission_id =
                        next_sampled_texture_admission_id.wrapping_add(1);
                    sampled_texture_cache.insert(
                        key,
                        CachedSampledTexture {
                            texture: texture.clone(),
                            write_watch,
                            admission_id,
                        },
                    );
                    sampled_texture_admission_order.push_back((key, admission_id));
                    texture
                }
            };
            Ok(UploadedImageTexture {
                texture,
                descriptor,
                // The resident 2D path has exact guest writeback today. A writable 3D image stays
                // resident in the shared sampled-image cache so later GPU reads see its contents;
                // it must not invalidate that cache through the 2D writeback path.
                writable: !fallback && writable && descriptor.texture_type == 9,
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

fn sampler_compare_function(value: u32) -> MTLCompareFunction {
    match value & 0x7 {
        0 => MTLCompareFunction::Never,
        1 => MTLCompareFunction::Less,
        2 => MTLCompareFunction::Equal,
        3 => MTLCompareFunction::LessEqual,
        4 => MTLCompareFunction::Greater,
        5 => MTLCompareFunction::NotEqual,
        6 => MTLCompareFunction::GreaterEqual,
        7 => MTLCompareFunction::Always,
        _ => unreachable!("three-bit sampler comparison function"),
    }
}

fn sampled_image_bindings(evaluation: &ShaderEvaluation) -> impl Iterator<Item = &ImageBinding> {
    evaluation.image_bindings.iter().filter(|binding| {
        binding.opcode.starts_with("ImageSample") || binding.opcode.starts_with("ImageGather")
    })
}

fn canonical_texture_bindings(evaluation: &ShaderEvaluation) -> Vec<&ImageBinding> {
    let slots = evaluation.image_binding_slots();
    let mut representatives = vec![
        None;
        usize::try_from(slots.texture_count)
            .expect("canonical texture count fits the host index")
    ];
    for (binding, slot) in evaluation
        .image_bindings
        .iter()
        .zip(slots.texture_slots.iter().copied())
    {
        let slot = usize::try_from(slot).expect("canonical texture slot fits the host index");
        representatives[slot].get_or_insert(binding);
    }
    representatives
        .into_iter()
        .map(|binding| binding.expect("canonical texture slots are dense"))
        .collect()
}

fn canonical_sampler_bindings(evaluation: &ShaderEvaluation) -> Vec<&ImageBinding> {
    let slots = evaluation.image_binding_slots();
    let mut representatives = vec![
        None;
        usize::try_from(slots.sampler_count)
            .expect("canonical sampler count fits the host index")
    ];
    for (binding, slot) in evaluation
        .image_bindings
        .iter()
        .zip(slots.sampler_slots.iter().copied())
    {
        let Some(slot) = slot else {
            continue;
        };
        let slot = usize::try_from(slot).expect("canonical sampler slot fits the host index");
        representatives[slot].get_or_insert(binding);
    }
    representatives
        .into_iter()
        .map(|binding| binding.expect("canonical sampler slots are dense"))
        .collect()
}

fn sampler_uses_normalized_coordinates(words: &[u32]) -> bool {
    words.first().is_none_or(|word| word & (1 << 15) == 0)
}

fn sampler_lod_bias(words: &[u32]) -> f32 {
    let raw = words.get(2).copied().unwrap_or_default() & 0x3fff;
    let signed = ((raw.cast_signed() ^ 0x2000) - 0x2000) as f32;
    signed / 256.0
}

fn create_sampler_argument_buffer(
    device: &Device,
    function: &Function,
    buffer_index: Option<u32>,
    evaluation: &ShaderEvaluation,
) -> Result<(Option<Buffer>, Vec<SamplerState>), String> {
    let sampled_bindings = canonical_sampler_bindings(evaluation);
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
        let descriptor = SamplerDescriptor::new();
        descriptor.set_address_mode_s(sampler_address_mode(words[0] & 0x7));
        descriptor.set_address_mode_t(sampler_address_mode((words[0] >> 3) & 0x7));
        descriptor.set_address_mode_r(sampler_address_mode((words[0] >> 6) & 0x7));
        descriptor.set_max_anisotropy(u64::from(1_u32 << ((words[0] >> 9) & 0x7).min(4)));
        descriptor.set_normalized_coordinates(sampler_uses_normalized_coordinates(words));
        // GFX10 stores a signed 14-bit fixed-point (8 fractional bits) sampler bias. Vulkan
        // implementations such as SharpEmu pass this directly to the native sampler; modern
        // Metal exposes the equivalent descriptor property as well.
        unsafe {
            descriptor.set_lod_bias(sampler_lod_bias(words));
        }
        descriptor.set_mag_filter(sampler_min_mag_filter((words[2] >> 20) & 0x3));
        descriptor.set_min_filter(sampler_min_mag_filter((words[2] >> 22) & 0x3));
        descriptor.set_mip_filter(sampler_mip_filter((words[2] >> 26) & 0x3));
        if image_is_comparison(binding.opcode) {
            descriptor.set_compare_function(sampler_compare_function((words[0] >> 12) & 0x7));
        }
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
        // GFX10's 2D rectangle and rectangle-list topologies consume three guest vertices but
        // rasterize a complete four-corner rectangle. The observed AGC shaders derive the fourth
        // corner from vertex ID, so invoking the translated vertex function once more preserves
        // the native rectangle while using Metal's triangle-strip topology.
        6 | 7 | 0x11 => Ok(MTLPrimitiveType::TriangleStrip),
        5 => Err("triangle-fan topology requires explicit Metal index expansion".to_owned()),
        _ => Err(format!("unknown guest primitive topology {guest:#x}")),
    }
}

fn metal_primitive_topology_class(guest: u32) -> Result<MTLPrimitiveTopologyClass, String> {
    match guest {
        1 => Ok(MTLPrimitiveTopologyClass::Point),
        2 | 3 => Ok(MTLPrimitiveTopologyClass::Line),
        4 | 5 | 6 | 7 | 0x11 => Ok(MTLPrimitiveTopologyClass::Triangle),
        _ => Err(format!("unknown guest primitive topology {guest:#x}")),
    }
}

const fn metal_nonindexed_vertex_count(guest_primitive_type: u32, guest_count: u32) -> u32 {
    if matches!(guest_primitive_type, 7 | 0x11) && guest_count == 3 {
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

fn dump_shader_pair_msl(
    directory: &Path,
    export_address: u64,
    pixel_address: u64,
    vertex_source: &str,
    pixel_source: &str,
) -> Result<(), String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
    let stem = format!("es-{export_address:x}-ps-{pixel_address:x}");
    let vertex_path = directory.join(format!("{stem}-vertex.metal"));
    let pixel_path = directory.join(format!("{stem}-pixel.metal"));
    std::fs::write(&vertex_path, vertex_source)
        .map_err(|error| format!("could not write {}: {error}", vertex_path.display()))?;
    std::fs::write(&pixel_path, pixel_source)
        .map_err(|error| format!("could not write {}: {error}", pixel_path.display()))?;
    Ok(())
}

fn dump_shader_pair_ir(
    directory: &Path,
    export_address: u64,
    pixel_address: u64,
    export: &[remu_shader::Instruction],
    pixel: &[remu_shader::Instruction],
) -> Result<(), String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
    let stem = format!("es-{export_address:x}-ps-{pixel_address:x}");
    let path = directory.join(format!("{stem}-ir.txt"));
    let contents = format!("export={export:#?}\n\npixel={pixel:#?}\n");
    std::fs::write(&path, contents)
        .map_err(|error| format!("could not write {}: {error}", path.display()))
}

fn dump_compute_shader_msl(
    directory: &Path,
    compute_address: u64,
    instructions: &[remu_shader::Instruction],
    source: &str,
) -> Result<(), String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
    let source_path = directory.join(format!("cs-{compute_address:x}.metal"));
    std::fs::write(&source_path, source)
        .map_err(|error| format!("could not write {}: {error}", source_path.display()))?;
    let ir_path = directory.join(format!("cs-{compute_address:x}-ir.txt"));
    std::fs::write(&ir_path, format!("{instructions:#?}\n"))
        .map_err(|error| format!("could not write {}: {error}", ir_path.display()))
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

fn pixel_input_controls(registers: &[GpuRegister]) -> [u32; 32] {
    std::array::from_fn(|index| {
        let index = u32::try_from(index).expect("PS input-control index fits u32");
        register_value(registers, SPI_PS_INPUT_CNTL_0 + index).unwrap_or(index)
    })
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
        // Metal's RG11B10Float is bit-compatible only with AMD 10_11_11_FLOAT. The reversed
        // 11_11_10 layout requires repacking and an R/B selector remap; treating it as the same
        // attachment silently exchanges both component widths and positions.
        (6, 7, 0) => (MTLPixelFormat::RG11B10Float, 4),
        // Gen5 2_10_10_10 places a 2-bit red component in the low bits and cannot be represented
        // by either Metal packed-10 format without repacking. Standard 10_10_10_2 puts R in bits
        // 0..9 and is bit-compatible with Metal RGB10A2.
        (9, 0, 0) => (MTLPixelFormat::RGB10A2Unorm, 4),
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
        // Gen5 commonly programs ALT component swap for 16-bit float scanout surfaces. Metal has
        // no BGRA16Float attachment; the shader export remains in logical component order, which
        // is the behavior used by the native Gen5 path for this attachment class.
        (12, 7, 0 | 1) => (MTLPixelFormat::RGBA16Float, 8),
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
    if endian != 0 || fmask_compression != 0 {
        return Err(format!(
            "color target {guest_slot} requires unsupported metadata/endian state endian={endian} fast_clear={fast_clear} fmask={fmask_compression} dcc={dcc_enable}"
        ));
    }
    let sample_log2 = (attrib >> 12) & 0x7;
    let fragment_log2 = (attrib >> 15) & 0x3;
    if sample_log2 != fragment_log2 || sample_log2 > 3 {
        return Err(format!(
            "color target {guest_slot} has unrepresentable sample/fragment counts (samples_log2={sample_log2}, fragments_log2={fragment_log2})"
        ));
    }
    let sample_count = 1_u32 << sample_log2;
    let slice_start = view & 0x1fff;
    let slice_max = (view >> 13) & 0x1fff;
    let mip_level = (view >> 26) & 0xf;
    let max_mip = (attrib2 >> 28) & 0xf;
    let mip0_depth = attrib3 & 0x1fff;
    let tile_mode = (attrib3 >> 14) & 0x1f;
    let maximum_extent_mip = 31 - width.max(height).leading_zeros();
    if mip_level > max_mip || max_mip > maximum_extent_mip {
        return Err(format!(
            "color target {guest_slot} has invalid mip view slices={slice_start}..={slice_max} mip={mip_level}/{max_mip} maximum={maximum_extent_mip} depth={mip0_depth}"
        ));
    }
    if max_mip != 0 && (sample_count != 1 || tile_mode != 27) {
        return Err(format!(
            "color target {guest_slot} mip hierarchy requires single-sample mode-27 storage, got samples={sample_count} tile={tile_mode} mip={mip_level}/{max_mip}"
        ));
    }
    if !matches!(tile_mode, 0 | 27) {
        return Err(format!(
            "color target {guest_slot} uses unsupported GFX10 swizzle mode {tile_mode}"
        ));
    }
    let resource_type = (attrib3 >> 24) & 0x3;
    let resource_level = (attrib3 >> 27) & 0x7;
    let non_block_compressed_view = (attrib3 >> 31) & 1;
    if resource_level != 0 || non_block_compressed_view != 0 {
        return Err(format!(
            "color target {guest_slot} requires unsupported resource state type={resource_type} level={resource_level} nbc={non_block_compressed_view} extent={width}x{height}"
        ));
    }
    let (texture_type, layer_count, base_layer) = match resource_type {
        0 if height == 1 && slice_start == 0 && slice_max == 0 && mip0_depth == 0 => (9, 1, 0),
        1 if slice_start == 0 && slice_max == 0 && mip0_depth == 0 => (9, 1, 0),
        2 if slice_start == 0 && mip0_depth != 0 => {
            let layers = mip0_depth
                .checked_add(1)
                .ok_or("3D color-target depth overflows")?;
            // GFX10 programs MIP0_DEPTH as depth-1. Some Prospero command streams program
            // SLICE_MAX as either the inclusive final layer or the exclusive layer count.
            if slice_max != mip0_depth && slice_max != layers {
                return Err(format!(
                    "color target {guest_slot} has inconsistent 3D slice/depth state slices={slice_start}..={slice_max} depth={mip0_depth}"
                ));
            }
            (10, layers, 0)
        }
        _ => {
            return Err(format!(
                "color target {guest_slot} requires unsupported resource state type={resource_type} slices={slice_start}..={slice_max} depth={mip0_depth} extent={width}x{height}"
            ));
        }
    };
    if max_mip != 0 && texture_type != 9 {
        return Err(format!(
            "color target {guest_slot} mipmapped resource type={resource_type} is not implemented"
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
        sample_count,
        texture_type,
        layer_count,
        base_layer,
        mip_level,
        mip_level_count: max_mip + 1,
        dcc_enabled: dcc_enable != 0,
        fast_clear_enabled: fast_clear != 0,
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
        sample_count: 1,
        texture_type: 9,
        layer_count: 1,
        base_layer: 0,
        mip_level: 0,
        mip_level_count: 1,
        dcc_enabled: false,
        fast_clear_enabled: false,
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

fn decode_raster_state(draw: &IndexedDraw) -> Result<DecodedRasterState, String> {
    // GFX10's reset value has FACE set and all culling/polygon-mode bits clear.
    let control = register_value(&draw.context_registers, PA_SU_SC_MODE_CNTL).unwrap_or(1 << 2);
    let cull_front = control & 1 != 0;
    let cull_back = control & 2 != 0;
    let cull_mode = match (cull_front, cull_back) {
        (false, false) => MTLCullMode::None,
        (true, false) => MTLCullMode::Front,
        (false, true) => MTLCullMode::Back,
        (true, true) => {
            return Err(
                "simultaneous front- and back-face culling has no exact Metal raster state"
                    .to_owned(),
            );
        }
    };
    let mut front_winding = if control & (1 << 2) != 0 {
        MTLWinding::Clockwise
    } else {
        MTLWinding::CounterClockwise
    };
    // Gen5 titles commonly program a negative viewport Y scale to account for the framebuffer's
    // top-left origin. `decode_viewport` represents that transform with a positive Metal viewport
    // (Metal rejects a negative height), which mirrors the projected triangle orientation. Keep
    // guest front/back-face classification intact by applying the corresponding winding flip.
    if programmed_viewport_has_negative_height(draw) {
        front_winding = match front_winding {
            MTLWinding::Clockwise => MTLWinding::CounterClockwise,
            MTLWinding::CounterClockwise => MTLWinding::Clockwise,
        };
    }

    let polygon_mode = (control >> 3) & 0x3;
    let fill_mode = match polygon_mode {
        0 => MTLTriangleFillMode::Fill,
        1 => {
            let front_type = (control >> 5) & 0x7;
            let back_type = (control >> 8) & 0x7;
            let visible_type = match cull_mode {
                MTLCullMode::Front => back_type,
                MTLCullMode::Back => front_type,
                MTLCullMode::None if front_type == back_type => front_type,
                MTLCullMode::None => {
                    return Err(format!(
                        "different front/back polygon modes ({front_type}/{back_type}) have no exact Metal raster state"
                    ));
                }
            };
            match visible_type {
                1 => MTLTriangleFillMode::Lines,
                2 => MTLTriangleFillMode::Fill,
                0 => {
                    return Err(
                        "point polygon mode has no exact Metal triangle raster state".to_owned(),
                    );
                }
                value => return Err(format!("reserved GFX10 polygon primitive type {value}")),
            }
        }
        value => return Err(format!("reserved GFX10 polygon mode {value}")),
    };

    Ok(DecodedRasterState {
        cull_mode,
        front_winding,
        fill_mode,
    })
}

fn decode_depth_target(draw: &IndexedDraw) -> Result<Option<DecodedDepthTarget>, String> {
    let Some(control) = register_value(&draw.context_registers, DB_DEPTH_CONTROL) else {
        return Ok(None);
    };
    let depth_enabled = control & (1 << 1) != 0;
    let write_enabled = control & (1 << 2) != 0;
    if !depth_enabled {
        return Ok(None);
    }

    let required = |offset, name| {
        register_value(&draw.context_registers, offset)
            .ok_or_else(|| format!("active depth target is missing {name}"))
    };
    let z_info = required(DB_Z_INFO, "DB_Z_INFO")?;
    let size = required(DB_DEPTH_SIZE_XY, "DB_DEPTH_SIZE_XY")?;
    let format = match z_info & 0x3 {
        // The AGC register defaults leave the high metadata bit set while the format is zero.
        // Unreal can carry that default into a color-only composite even with Z_ENABLE still
        // latched. SharpEmu treats guest format zero as no depth surface.
        0 => return Ok(None),
        1 => (MTLPixelFormat::Depth16Unorm, 2),
        3 => (MTLPixelFormat::Depth32Float, 4),
        value => {
            return Err(format!(
                "active depth target uses unsupported Z format {value} in DB_Z_INFO={z_info:#010x}"
            ));
        }
    };
    let samples_log2 = (z_info >> 2) & 0x3;
    let sample_count = 1_u32 << samples_log2;
    let tile_mode = (z_info >> 4) & 0x1f;
    if !matches!(tile_mode, 0 | 27) {
        return Err(format!(
            "active depth target uses unsupported GFX10 swizzle mode {tile_mode}"
        ));
    }

    let width = (size & 0x3fff)
        .checked_add(1)
        .ok_or("depth-target width overflows")?;
    let height = ((size >> 16) & 0x3fff)
        .checked_add(1)
        .ok_or("depth-target height overflows")?;
    let read_address = (u64::from(
        register_value(&draw.context_registers, DB_Z_READ_BASE_HI).unwrap_or_default() & 0xff,
    ) << 40)
        | (u64::from(register_value(&draw.context_registers, DB_Z_READ_BASE).unwrap_or_default())
            << 8);
    let write_address = (u64::from(
        register_value(&draw.context_registers, DB_Z_WRITE_BASE_HI).unwrap_or_default() & 0xff,
    ) << 40)
        | (u64::from(register_value(&draw.context_registers, DB_Z_WRITE_BASE).unwrap_or_default())
            << 8);
    if read_address != 0 && write_address != 0 && read_address != write_address {
        return Err(format!(
            "separate depth read/write surfaces are unsupported (read={read_address:#x}, write={write_address:#x})"
        ));
    }
    let address = if write_enabled && write_address != 0 {
        write_address
    } else if read_address != 0 {
        read_address
    } else {
        write_address
    };
    if address < 0x10_000 || width > 16_384 || height > 16_384 {
        return Err(format!(
            "invalid depth target address={address:#x} extent={width}x{height}"
        ));
    }
    let pitch = if tile_mode == 0 {
        effective_linear_pitch(width, format.1).ok_or("linear depth-target pitch overflows")?
    } else {
        width
    };
    let compare = match (control >> 4) & 0x7 {
        0 => MTLCompareFunction::Never,
        1 => MTLCompareFunction::Less,
        2 => MTLCompareFunction::Equal,
        3 => MTLCompareFunction::LessEqual,
        4 => MTLCompareFunction::Greater,
        5 => MTLCompareFunction::NotEqual,
        6 => MTLCompareFunction::GreaterEqual,
        7 => MTLCompareFunction::Always,
        _ => unreachable!("three-bit compare function"),
    };
    Ok(Some(DecodedDepthTarget {
        texture: GuestTextureDescriptor {
            address,
            width,
            height,
            pitch,
            tile_mode,
            pixel_format: format.0,
            bytes_per_pixel: format.1,
        },
        compare,
        write_enabled,
        sample_count,
    }))
}

fn initial_depth_clear_value(compare: MTLCompareFunction) -> f64 {
    if matches!(
        compare,
        MTLCompareFunction::Greater | MTLCompareFunction::GreaterEqual
    ) {
        0.0
    } else {
        1.0
    }
}

fn graphics_pipeline_key(draw: &IndexedDraw) -> Result<GraphicsPipelineKey, String> {
    let targets = decode_render_targets(draw)?;
    graphics_pipeline_key_from_targets(draw, &targets)
}

fn graphics_pipeline_key_from_targets(
    draw: &IndexedDraw,
    targets: &[DecodedRenderTarget],
) -> Result<GraphicsPipelineKey, String> {
    let depth = decode_depth_target(draw)?;
    let export_evaluation = draw
        .export_shader_evaluation
        .as_ref()
        .ok_or("draw has no export resource evaluation")?;
    let pixel_evaluation = draw
        .pixel_shader_evaluation
        .as_ref()
        .ok_or("draw has no pixel resource evaluation")?;
    let export_program = &draw
        .export_shader
        .as_ref()
        .ok_or("draw has no decoded export shader")?
        .program;
    let pixel_program = &draw
        .pixel_shader
        .as_ref()
        .ok_or("draw has no decoded pixel shader")?
        .program;
    let sample_count = targets.first().map_or(1, |target| target.sample_count);
    let render_target_array_length = targets.first().map_or(1, |target| target.layer_count);
    if let Some(target) = targets
        .iter()
        .find(|target| target.sample_count != sample_count)
    {
        return Err(format!(
            "color target {} uses {} samples while the render pass uses {sample_count}",
            target.guest_slot, target.sample_count
        ));
    }
    if let Some(depth) = depth
        && depth.sample_count != sample_count
    {
        return Err(format!(
            "depth target uses {} samples while color targets use {sample_count}",
            depth.sample_count
        ));
    }
    if let Some(target) = targets
        .iter()
        .find(|target| target.layer_count != render_target_array_length)
    {
        return Err(format!(
            "color target {} exposes {} layers while the render pass exposes {render_target_array_length}",
            target.guest_slot, target.layer_count
        ));
    }
    if render_target_array_length > 1 && depth.is_some() {
        return Err(
            "layered color rendering with a depth attachment is not implemented".to_owned(),
        );
    }
    Ok(GraphicsPipelineKey {
        export_shader_address: draw.export_shader_address,
        pixel_shader_address: draw.pixel_shader_address,
        export_resources: graphics_shader_resource_key(
            export_evaluation,
            program_embeds_global_base(export_program),
        ),
        pixel_resources: graphics_shader_resource_key(
            pixel_evaluation,
            program_embeds_global_base(pixel_program),
        ),
        pixel_input_controls: pixel_input_controls(&draw.context_registers),
        targets: targets
            .iter()
            .map(|target| {
                let control = register_value(
                    &draw.context_registers,
                    CB_BLEND0_CONTROL + target.guest_slot,
                )
                .unwrap_or(0x2001_0001);
                Ok(RenderTargetPipelineState {
                    guest_slot: target.guest_slot,
                    pixel_format: target.texture.pixel_format,
                    output_kind: target.output_kind,
                    write_mask: target.write_mask,
                    blend: decode_metal_blend_attachment(control).map_err(|error| {
                        format!(
                            "color target {} has invalid CB_BLEND_CONTROL={control:#010x}: {error}",
                            target.guest_slot
                        )
                    })?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
        depth: depth.map(|depth| DepthTargetPipelineState {
            pixel_format: depth.texture.pixel_format,
        }),
        sample_count,
        render_target_array_length,
        primitive_topology_class: metal_primitive_topology_class(draw.primitive_type)?,
    })
}

fn graphics_shader_resource_key(
    evaluation: &ShaderEvaluation,
    _embeds_global_base: bool,
) -> GraphicsShaderResourceKey {
    let image_slots = evaluation.image_binding_slots();
    GraphicsShaderResourceKey {
        initial_scalar_count: evaluation.initial_scalar_registers.len(),
        // Every lowering embeds the byte extent. Only GLOBAL instructions still embed a full
        // guest base; descriptor-backed buffer instructions address the current dword-aligned
        // Metal binding through the descriptor's dynamic low-bit bias.
        global_bindings: evaluation
            .global_memory_bindings
            .iter()
            .map(|binding| (binding.instruction_pcs.to_vec(), 0, 0, binding.writable))
            .collect(),
        // Image addresses, component selection, exact pixel formats, and sampler state are bound
        // dynamically. Only the resource order, MSL component class, and access/interface shape
        // alter the generated Metal function signature.
        image_bindings: evaluation
            .image_bindings
            .iter()
            .enumerate()
            .map(|(index, binding)| {
                (
                    binding.pc,
                    image_component_class(binding),
                    is_storage_image_opcode(binding.opcode),
                    image_is_comparison(binding.opcode),
                    binding.control.dimension,
                    binding
                        .resource_descriptor
                        .get(3)
                        .map_or(0, |word| (word >> 28) & 0xf),
                    image_slots.texture_slots[index],
                    image_slots.sampler_slots[index],
                )
            })
            .collect(),
        vertex_inputs: evaluation
            .vertex_inputs
            .iter()
            .map(|input| {
                (
                    input.pc,
                    input.location,
                    input.component_count,
                    input.data_format,
                    input.number_format,
                    input.stride,
                    input.offset_bytes,
                )
            })
            .collect(),
    }
}

fn image_component_class(binding: &ImageBinding) -> u32 {
    let unified_format = binding
        .resource_descriptor
        .get(1)
        .map_or(0, |word| (word >> 20) & 0x1ff);
    let number_format =
        remu_shader::decode_unified_format(unified_format).map_or(0, |(_, number)| number);
    match number_format {
        4 => 1, // uint
        5 => 2, // int
        _ => 0, // float, normalized, scaled, and sRGB
    }
}

fn decode_metal_blend_attachment(control: u32) -> Result<MetalBlendAttachmentState, String> {
    let disabled = MetalBlendAttachmentState {
        enabled: false,
        source_rgb: MTLBlendFactor::One,
        destination_rgb: MTLBlendFactor::Zero,
        rgb_operation: MTLBlendOperation::Add,
        source_alpha: MTLBlendFactor::One,
        destination_alpha: MTLBlendFactor::Zero,
        alpha_operation: MTLBlendOperation::Add,
    };
    if control & (1 << 30) == 0 {
        return Ok(disabled);
    }

    let source_rgb = metal_blend_factor(control & 0x1f)?;
    let rgb_operation = metal_blend_operation((control >> 5) & 0x7)?;
    let destination_rgb = metal_blend_factor((control >> 8) & 0x1f)?;
    let separate_alpha = control & (1 << 29) != 0;
    let (source_alpha, alpha_operation, destination_alpha) = if separate_alpha {
        (
            metal_blend_factor((control >> 16) & 0x1f)?,
            metal_blend_operation((control >> 21) & 0x7)?,
            metal_blend_factor((control >> 24) & 0x1f)?,
        )
    } else {
        (source_rgb, rgb_operation, destination_rgb)
    };
    Ok(MetalBlendAttachmentState {
        enabled: true,
        source_rgb,
        destination_rgb,
        rgb_operation,
        source_alpha,
        destination_alpha,
        alpha_operation,
    })
}

fn metal_blend_factor(value: u32) -> Result<MTLBlendFactor, String> {
    match value {
        0 => Ok(MTLBlendFactor::Zero),
        1 => Ok(MTLBlendFactor::One),
        2 => Ok(MTLBlendFactor::SourceColor),
        3 => Ok(MTLBlendFactor::OneMinusSourceColor),
        4 => Ok(MTLBlendFactor::SourceAlpha),
        5 => Ok(MTLBlendFactor::OneMinusSourceAlpha),
        6 => Ok(MTLBlendFactor::DestinationAlpha),
        7 => Ok(MTLBlendFactor::OneMinusDestinationAlpha),
        8 => Ok(MTLBlendFactor::DestinationColor),
        9 => Ok(MTLBlendFactor::OneMinusDestinationColor),
        10 => Ok(MTLBlendFactor::SourceAlphaSaturated),
        13 => Ok(MTLBlendFactor::BlendColor),
        14 => Ok(MTLBlendFactor::OneMinusBlendColor),
        19 => Ok(MTLBlendFactor::BlendAlpha),
        20 => Ok(MTLBlendFactor::OneMinusBlendAlpha),
        15..=18 => Err(format!(
            "dual-source blend factor {value} requires a second pixel-shader color output"
        )),
        _ => Err(format!("unsupported blend factor {value}")),
    }
}

fn metal_blend_operation(value: u32) -> Result<MTLBlendOperation, String> {
    match value {
        0 => Ok(MTLBlendOperation::Add),
        1 => Ok(MTLBlendOperation::Subtract),
        2 => Ok(MTLBlendOperation::Min),
        3 => Ok(MTLBlendOperation::Max),
        4 => Ok(MTLBlendOperation::ReverseSubtract),
        _ => Err(format!("unsupported blend operation {value}")),
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

fn read_draw_indices(draw: &IndexedDraw) -> Option<Arc<[u32]>> {
    if let Some(indices) = &draw.index_snapshot {
        return Some(indices.clone());
    }
    if draw.index_buffer_address < 0x10_000 || draw.index_count > 65_536 {
        return None;
    }
    let element_size = match draw.index_size {
        0 => 2_u64,
        1 => 4,
        2 => 1,
        _ => return None,
    };
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
            .collect::<Vec<_>>()
            .into(),
    )
}

fn copy_tiled_to_linear_at_z(
    source: &[u8],
    descriptor: GuestTextureDescriptor,
    destination: &mut [u8],
    surface_z: usize,
) -> Result<(), String> {
    let (width, height, pitch, bytes_per_element) = texture_element_layout(descriptor)?;
    let block_bytes = tiled_block_bytes(descriptor.tile_mode)
        .ok_or_else(|| format!("unsupported texture tile mode {}", descriptor.tile_mode))?;
    let (block_width, block_height) = tiled_block_dimensions(block_bytes, bytes_per_element)
        .ok_or("tiled texture requires a 1, 2, 4, 8, or 16-byte element format")?;
    let expected_source = usize::try_from(
        guest_texture_byte_count(descriptor).ok_or("tiled texture byte count overflows")?,
    )
    .map_err(|_| "tiled texture byte count does not fit usize")?;
    let expected_destination = width
        .checked_mul(height)
        .and_then(|elements| elements.checked_mul(bytes_per_element))
        .ok_or("tiled texture linear byte count overflows")?;
    if source.len() != expected_source || destination.len() != expected_destination {
        return Err(format!(
            "tiled texture copy has source/destination lengths {}/{}, expected {expected_source}/{expected_destination}",
            source.len(),
            destination.len(),
        ));
    }
    // The guest pitch is the physical row width. Visible width only bounds copied texels;
    // using it as the block stride shifts every later tile row on padded PS5 surfaces.
    let blocks_per_row = pitch.div_ceil(block_width);
    let visible_blocks_per_row = width.div_ceil(block_width);
    for block_y in 0..height.div_ceil(block_height) {
        let rows = (height - block_y * block_height).min(block_height);
        for block_x in 0..visible_blocks_per_row {
            let columns = (width - block_x * block_width).min(block_width);
            let block_base = (block_y * blocks_per_row + block_x) * block_bytes;
            let offsets = tiled_block_byte_offsets(
                descriptor.tile_mode,
                block_x,
                block_y,
                bytes_per_element,
                block_width,
                block_height,
            )?;
            for local_y in 0..rows {
                let destination_row = ((block_y * block_height + local_y) * width
                    + block_x * block_width)
                    * bytes_per_element;
                let offset_row = local_y * block_width;
                for local_x in 0..columns {
                    let within_block = usize::from(offsets[offset_row + local_x])
                        ^ if descriptor.tile_mode == 27 {
                            mode27_surface_z_xor(surface_z)
                        } else {
                            0
                        };
                    let source_offset = block_base + within_block;
                    let destination_offset = destination_row + local_x * bytes_per_element;
                    destination[destination_offset..destination_offset + bytes_per_element]
                        .copy_from_slice(&source[source_offset..source_offset + bytes_per_element]);
                }
            }
        }
    }
    Ok(())
}

fn copy_linear_to_tiled(
    source: &[u8],
    descriptor: GuestTextureDescriptor,
    destination: &mut [u8],
) -> Result<(), String> {
    let (width, height, pitch, bytes_per_element) = texture_element_layout(descriptor)?;
    let block_bytes = tiled_block_bytes(descriptor.tile_mode)
        .ok_or_else(|| format!("unsupported texture tile mode {}", descriptor.tile_mode))?;
    let (block_width, block_height) = tiled_block_dimensions(block_bytes, bytes_per_element)
        .ok_or("tiled texture requires a 1, 2, 4, 8, or 16-byte element format")?;
    let expected_source = width
        .checked_mul(height)
        .and_then(|elements| elements.checked_mul(bytes_per_element))
        .ok_or("tiled texture linear byte count overflows")?;
    let expected_destination = usize::try_from(
        guest_texture_byte_count(descriptor).ok_or("tiled texture byte count overflows")?,
    )
    .map_err(|_| "tiled texture byte count does not fit usize")?;
    if source.len() != expected_source || destination.len() != expected_destination {
        return Err(format!(
            "tiled texture copy has source/destination lengths {}/{}, expected {expected_source}/{expected_destination}",
            source.len(),
            destination.len(),
        ));
    }
    let blocks_per_row = pitch.div_ceil(block_width);
    let visible_blocks_per_row = width.div_ceil(block_width);
    for block_y in 0..height.div_ceil(block_height) {
        let rows = (height - block_y * block_height).min(block_height);
        for block_x in 0..visible_blocks_per_row {
            let columns = (width - block_x * block_width).min(block_width);
            let block_base = (block_y * blocks_per_row + block_x) * block_bytes;
            let offsets = tiled_block_byte_offsets(
                descriptor.tile_mode,
                block_x,
                block_y,
                bytes_per_element,
                block_width,
                block_height,
            )?;
            for local_y in 0..rows {
                let source_row = ((block_y * block_height + local_y) * width
                    + block_x * block_width)
                    * bytes_per_element;
                let offset_row = local_y * block_width;
                for local_x in 0..columns {
                    let source_offset = source_row + local_x * bytes_per_element;
                    let destination_offset =
                        block_base + usize::from(offsets[offset_row + local_x]);
                    destination[destination_offset..destination_offset + bytes_per_element]
                        .copy_from_slice(&source[source_offset..source_offset + bytes_per_element]);
                }
            }
        }
    }
    Ok(())
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
    let (_width, height, pitch, bytes_per_element) = texture_element_layout(descriptor).ok()?;
    if let Some(block_bytes) = tiled_block_bytes(descriptor.tile_mode) {
        let (block_width, block_height) = tiled_block_dimensions(block_bytes, bytes_per_element)?;
        let blocks_wide = u64::try_from(pitch)
            .ok()?
            .div_ceil(u64::try_from(block_width).ok()?);
        let blocks_high = u64::try_from(height)
            .ok()?
            .div_ceil(u64::try_from(block_height).ok()?);
        blocks_wide
            .checked_mul(blocks_high)?
            .checked_mul(u64::try_from(block_bytes).ok()?)
    } else {
        u64::try_from(pitch)
            .ok()?
            .checked_mul(u64::try_from(height).ok()?)?
            .checked_mul(u64::try_from(bytes_per_element).ok()?)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GuestMipLevelLayout {
    width: usize,
    height: usize,
    padded_width: usize,
    offset: usize,
    tail_x: usize,
    tail_y: usize,
    in_tail: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GuestMipChainLayout {
    levels: Vec<GuestMipLevelLayout>,
    byte_count: usize,
    block_width: usize,
    block_height: usize,
    bytes_per_element: usize,
}

const MODE27_MIP_TAIL_LOCATIONS: [[(usize, usize); 12]; 5] = [
    [
        (128, 0),
        (0, 128),
        (64, 0),
        (0, 64),
        (32, 0),
        (16, 32),
        (0, 48),
        (0, 32),
        (16, 16),
        (16, 0),
        (0, 16),
        (0, 0),
    ],
    [
        (128, 0),
        (0, 64),
        (64, 0),
        (0, 32),
        (32, 0),
        (16, 16),
        (0, 24),
        (0, 16),
        (16, 8),
        (16, 0),
        (0, 8),
        (0, 0),
    ],
    [
        (64, 0),
        (0, 64),
        (32, 0),
        (0, 32),
        (16, 0),
        (8, 16),
        (0, 24),
        (0, 16),
        (8, 8),
        (8, 0),
        (0, 8),
        (0, 0),
    ],
    [
        (64, 0),
        (0, 32),
        (32, 0),
        (0, 16),
        (16, 0),
        (8, 8),
        (0, 12),
        (0, 8),
        (8, 4),
        (8, 0),
        (0, 4),
        (0, 0),
    ],
    [
        (32, 0),
        (0, 32),
        (16, 0),
        (0, 16),
        (8, 0),
        (4, 8),
        (0, 12),
        (0, 8),
        (4, 4),
        (4, 0),
        (0, 4),
        (0, 0),
    ],
];

fn shift_ceil(value: usize, shift: u32) -> Option<usize> {
    let divisor = 1_usize.checked_shl(shift)?;
    value.checked_add(divisor - 1).map(|value| value / divisor)
}

fn mode27_mip_chain_layout(
    descriptor: GuestTextureDescriptor,
    mip_level_count: u32,
) -> Result<GuestMipChainLayout, String> {
    if descriptor.tile_mode != 27 || mip_level_count == 0 || mip_level_count > 16 {
        return Err(format!(
            "mode-27 mip layout requires 1..=16 levels, got tile={} levels={mip_level_count}",
            descriptor.tile_mode
        ));
    }
    let bytes_per_element = usize::try_from(descriptor.bytes_per_pixel)
        .map_err(|_| "mip element byte count does not fit usize")?;
    let bytes_log2 = usize::try_from(bytes_per_element.trailing_zeros())
        .map_err(|_| "mip element byte logarithm does not fit usize")?;
    if !bytes_per_element.is_power_of_two() || bytes_log2 >= MODE27_MIP_TAIL_LOCATIONS.len() {
        return Err(format!(
            "mode-27 mip layout has unsupported {}-byte elements",
            descriptor.bytes_per_pixel
        ));
    }
    let (block_width, block_height) = tiled_block_dimensions(65_536, bytes_per_element)
        .ok_or("mode-27 mip layout requires a 1, 2, 4, 8, or 16-byte element format")?;
    let width = usize::try_from(descriptor.width).map_err(|_| "mip width does not fit usize")?;
    let height = usize::try_from(descriptor.height).map_err(|_| "mip height does not fit usize")?;
    let level_count =
        usize::try_from(mip_level_count).map_err(|_| "mip count does not fit usize")?;
    let tail_width = block_width / 2;
    let mut first_tail = level_count;
    if level_count > 1 {
        for level in 0..level_count {
            let shift = u32::try_from(level).expect("level count is bounded");
            if shift_ceil(width, shift).is_some_and(|width| width <= tail_width)
                && shift_ceil(height, shift).is_some_and(|height| height <= block_height)
                && level_count - level <= 12
            {
                first_tail = level;
                break;
            }
        }
    }
    let empty = GuestMipLevelLayout {
        width: 0,
        height: 0,
        padded_width: 0,
        offset: 0,
        tail_x: 0,
        tail_y: 0,
        in_tail: false,
    };
    let mut levels = vec![empty; level_count];
    let mut offset = if first_tail < level_count {
        65_536_usize
    } else {
        0
    };
    for level in (0..first_tail).rev() {
        let shift = u32::try_from(level).expect("level count is bounded");
        let mip_width = shift_ceil(width, shift)
            .ok_or("mip width overflows")?
            .max(1);
        let mip_height = shift_ceil(height, shift)
            .ok_or("mip height overflows")?
            .max(1);
        let padded_width = mip_width
            .div_ceil(block_width)
            .checked_mul(block_width)
            .ok_or("mip padded width overflows")?;
        let padded_height = mip_height
            .div_ceil(block_height)
            .checked_mul(block_height)
            .ok_or("mip padded height overflows")?;
        let size = padded_width
            .checked_mul(padded_height)
            .and_then(|elements| elements.checked_mul(bytes_per_element))
            .ok_or("mip storage size overflows")?;
        levels[level] = GuestMipLevelLayout {
            width: (width >> shift).max(1),
            height: (height >> shift).max(1),
            padded_width,
            offset,
            ..empty
        };
        offset = offset.checked_add(size).ok_or("mip chain size overflows")?;
    }
    for level in first_tail..level_count {
        let shift = u32::try_from(level).expect("level count is bounded");
        let (tail_x, tail_y) = MODE27_MIP_TAIL_LOCATIONS[bytes_log2][level - first_tail];
        let mip_width = (width >> shift).max(1);
        let mip_height = (height >> shift).max(1);
        if tail_x + mip_width > block_width || tail_y + mip_height > block_height {
            return Err(format!(
                "mode-27 mip {level} tail placement ({tail_x},{tail_y}) {mip_width}x{mip_height} exceeds {block_width}x{block_height}"
            ));
        }
        levels[level] = GuestMipLevelLayout {
            width: mip_width,
            height: mip_height,
            padded_width: block_width,
            offset: 0,
            tail_x,
            tail_y,
            in_tail: true,
        };
    }
    Ok(GuestMipChainLayout {
        levels,
        byte_count: offset,
        block_width,
        block_height,
        bytes_per_element,
    })
}

fn mode27_mip_texel_offset(
    layout: &GuestMipChainLayout,
    level: GuestMipLevelLayout,
    x: usize,
    y: usize,
) -> Result<usize, String> {
    let (block_x, block_y, local_x, local_y) = if level.in_tail {
        (0, 0, level.tail_x + x, level.tail_y + y)
    } else {
        (
            x / layout.block_width,
            y / layout.block_height,
            x % layout.block_width,
            y % layout.block_height,
        )
    };
    let blocks_per_row = level.padded_width / layout.block_width;
    let block_offset = if level.in_tail {
        0
    } else {
        (block_y * blocks_per_row + block_x)
            .checked_mul(65_536)
            .ok_or("mip block offset overflows")?
    };
    let offsets = mode27_block_byte_offsets(
        block_x,
        block_y,
        layout.bytes_per_element,
        layout.block_width,
        layout.block_height,
    );
    level
        .offset
        .checked_add(block_offset)
        .and_then(|offset| {
            offset.checked_add(usize::from(offsets[local_y * layout.block_width + local_x]))
        })
        .ok_or_else(|| "mip texel offset overflows".to_owned())
}

fn decode_mode27_mip_chain_from_raw(
    descriptor: GuestTextureDescriptor,
    mip_level_count: u32,
    raw: &[u8],
) -> Result<(GuestMipChainLayout, Vec<Vec<u8>>), String> {
    let layout = mode27_mip_chain_layout(descriptor, mip_level_count)?;
    if raw.len() != layout.byte_count {
        return Err(format!(
            "mode-27 mip source has {} bytes, expected {}",
            raw.len(),
            layout.byte_count
        ));
    }
    let mut decoded = Vec::with_capacity(layout.levels.len());
    for &level in &layout.levels {
        let mut linear = vec![
            0_u8;
            level
                .width
                .checked_mul(level.height)
                .and_then(|elements| elements.checked_mul(layout.bytes_per_element))
                .ok_or("mip linear size overflows")?
        ];
        for y in 0..level.height {
            for x in 0..level.width {
                let source = mode27_mip_texel_offset(&layout, level, x, y)?;
                let destination = (y * level.width + x) * layout.bytes_per_element;
                linear[destination..destination + layout.bytes_per_element].copy_from_slice(
                    raw.get(source..source + layout.bytes_per_element)
                        .ok_or("mode-27 mip texel exceeds backing")?,
                );
            }
        }
        decoded.push(linear);
    }
    Ok((layout, decoded))
}

fn encode_mode27_mip_chain_into_raw(
    layout: &GuestMipChainLayout,
    linear_levels: &[Vec<u8>],
    raw: &mut [u8],
) -> Result<(), String> {
    if raw.len() != layout.byte_count || linear_levels.len() != layout.levels.len() {
        return Err("mode-27 mip writeback layout mismatch".to_owned());
    }
    for (&level, linear) in layout.levels.iter().zip(linear_levels) {
        let expected = level
            .width
            .checked_mul(level.height)
            .and_then(|elements| elements.checked_mul(layout.bytes_per_element))
            .ok_or("mip linear size overflows")?;
        if linear.len() != expected {
            return Err(format!(
                "mode-27 mip writeback has {} bytes, expected {expected}",
                linear.len()
            ));
        }
        for y in 0..level.height {
            for x in 0..level.width {
                let source = (y * level.width + x) * layout.bytes_per_element;
                let destination = mode27_mip_texel_offset(layout, level, x, y)?;
                raw[destination..destination + layout.bytes_per_element]
                    .copy_from_slice(&linear[source..source + layout.bytes_per_element]);
            }
        }
    }
    Ok(())
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
    if guest_ranges_overlap(left.address, left_bytes, right.address, right_bytes) {
        return true;
    }
    let Some(left_mapping) = guest_shared_mapping_identity(left.address, left_bytes) else {
        return false;
    };
    let Some(right_mapping) = guest_shared_mapping_identity(right.address, right_bytes) else {
        return false;
    };
    left_mapping.backing_id == right_mapping.backing_id
        && guest_ranges_overlap(
            left_mapping.backing_offset,
            left_bytes,
            right_mapping.backing_offset,
            right_bytes,
        )
}

fn guest_texture_descriptors_share_storage(
    left: GuestTextureDescriptor,
    right: GuestTextureDescriptor,
) -> bool {
    if left == right {
        return true;
    }
    if !guest_texture_descriptors_have_same_layout(left, right) {
        return false;
    }
    guest_texture_descriptors_share_byte_storage(left, right)
}

fn guest_texture_descriptors_share_component_reordered_storage(
    left: GuestTextureDescriptor,
    right: GuestTextureDescriptor,
) -> bool {
    let formats_are_red_blue_reorders = matches!(
        (left.pixel_format, right.pixel_format),
        (MTLPixelFormat::BGRA8Unorm, MTLPixelFormat::RGBA8Unorm)
            | (MTLPixelFormat::RGBA8Unorm, MTLPixelFormat::BGRA8Unorm)
            | (
                MTLPixelFormat::BGRA8Unorm_sRGB,
                MTLPixelFormat::RGBA8Unorm_sRGB
            )
            | (
                MTLPixelFormat::RGBA8Unorm_sRGB,
                MTLPixelFormat::BGRA8Unorm_sRGB
            )
    );
    formats_are_red_blue_reorders
        && guest_texture_descriptors_have_same_byte_layout(left, right)
        && guest_texture_descriptors_share_byte_storage(left, right)
}

fn guest_texture_descriptors_share_byte_storage(
    left: GuestTextureDescriptor,
    right: GuestTextureDescriptor,
) -> bool {
    let Some(byte_count) = guest_texture_byte_count(left) else {
        return false;
    };
    if guest_texture_byte_count(right) != Some(byte_count) {
        return false;
    }
    if left.address == right.address {
        return true;
    }
    let Some(left_mapping) = guest_shared_mapping_identity(left.address, byte_count) else {
        return false;
    };
    let Some(right_mapping) = guest_shared_mapping_identity(right.address, byte_count) else {
        return false;
    };
    left_mapping == right_mapping
}

fn guest_texture_descriptors_have_same_layout(
    left: GuestTextureDescriptor,
    right: GuestTextureDescriptor,
) -> bool {
    guest_texture_descriptors_have_same_byte_layout(left, right)
        && left.pixel_format == right.pixel_format
}

fn guest_texture_descriptors_have_same_byte_layout(
    left: GuestTextureDescriptor,
    right: GuestTextureDescriptor,
) -> bool {
    left.width == right.width
        && left.height == right.height
        && left.pitch == right.pitch
        && left.tile_mode == right.tile_mode
        && left.bytes_per_pixel == right.bytes_per_pixel
}

fn render_resolve_candidate_matches(
    sampled: GuestTextureDescriptor,
    rendered: GuestTextureDescriptor,
    sample_count: u32,
) -> bool {
    if !guest_texture_descriptors_have_same_layout(sampled, rendered) {
        return false;
    }
    if guest_texture_descriptors_overlap(sampled, rendered) {
        return true;
    }
    // Gen5's shader-readable single-sample view commonly follows the multisample allocation in
    // guest memory. GNM allocators round that allocation to a large GPU page, so the SRD base is
    // just beyond the raw sample storage rather than overlapping CB_COLOR_BASE. This relationship
    // is narrow enough to retain allocation identity while rejecting unrelated same-size targets.
    const GPU_LARGE_PAGE_BYTES: u64 = 2 * 1_024 * 1_024;
    let Some(multisample_bytes) = guest_texture_byte_count(rendered)
        .and_then(|bytes| bytes.checked_mul(u64::from(sample_count)))
    else {
        return false;
    };
    let Some(rounded_bytes) = multisample_bytes
        .checked_add(GPU_LARGE_PAGE_BYTES - 1)
        .map(|bytes| bytes & !(GPU_LARGE_PAGE_BYTES - 1))
    else {
        return false;
    };
    let Some(minimum_resolve_address) = rendered.address.checked_add(multisample_bytes) else {
        return false;
    };
    let Some(maximum_resolve_address) = rendered.address.checked_add(rounded_bytes) else {
        return false;
    };
    (minimum_resolve_address..=maximum_resolve_address).contains(&sampled.address)
}

fn guest_ranges_overlap(left: u64, left_bytes: u64, right: u64, right_bytes: u64) -> bool {
    let left_end = left.saturating_add(left_bytes);
    let right_end = right.saturating_add(right_bytes);
    left < right_end && right < left_end
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

const fn tiled_block_bytes(tile_mode: u32) -> Option<usize> {
    match tile_mode {
        1 => Some(256),
        5 => Some(4_096),
        9 | 24 | 27 => Some(65_536),
        _ => None,
    }
}

fn tiled_block_dimensions(block_bytes: usize, bytes_per_element: usize) -> Option<(usize, usize)> {
    if !bytes_per_element.is_power_of_two()
        || !(1..=16).contains(&bytes_per_element)
        || !block_bytes.is_power_of_two()
        || block_bytes < bytes_per_element
    {
        return None;
    }
    let element_bits = usize::try_from((block_bytes / bytes_per_element).trailing_zeros()).ok()?;
    let width_bits = element_bits.div_ceil(2);
    Some((1 << width_bits, 1 << (element_bits - width_bits)))
}

fn tiled_block_byte_offsets(
    tile_mode: u32,
    block_x: usize,
    block_y: usize,
    bytes_per_element: usize,
    block_width: usize,
    block_height: usize,
) -> Result<&'static [u16], String> {
    match tile_mode {
        1 => Ok(mode1_block_byte_offsets(
            bytes_per_element,
            block_width,
            block_height,
        )),
        5 => Ok(mode5_block_byte_offsets(
            bytes_per_element,
            block_width,
            block_height,
        )),
        9 => Ok(mode9_block_byte_offsets(
            bytes_per_element,
            block_width,
            block_height,
        )),
        24 if bytes_per_element <= 8 => Ok(mode24_block_byte_offsets(
            block_x,
            block_y,
            bytes_per_element,
            block_width,
            block_height,
        )),
        24 => Err(format!(
            "unsupported mode 24 depth element size {bytes_per_element}"
        )),
        27 => Ok(mode27_block_byte_offsets(
            block_x,
            block_y,
            bytes_per_element,
            block_width,
            block_height,
        )),
        _ => Err(format!("unsupported texture tile mode {tile_mode}")),
    }
}

fn mode1_block_byte_offsets(
    bytes_per_element: usize,
    block_width: usize,
    block_height: usize,
) -> &'static [u16] {
    static OFFSETS: OnceLock<[Box<[u16]>; 5]> = OnceLock::new();
    let offsets = OFFSETS.get_or_init(|| {
        std::array::from_fn(|log2_bytes| {
            let bytes_per_element = 1 << log2_bytes;
            let (block_width, block_height) = tiled_block_dimensions(256, bytes_per_element)
                .expect("mode 1 table element size is supported");
            let mut offsets = vec![0_u16; block_width * block_height];
            for local_y in 0..block_height {
                for local_x in 0..block_width {
                    // GFX10 SW_256_S uses the same low eight address equations as
                    // SW_64K_S; its 256-byte block ends before the latter's higher bits.
                    offsets[local_y * block_width + local_x] = u16::try_from(
                        mode9_within_block_byte_offset(local_x, local_y, bytes_per_element) & 0xff,
                    )
                    .expect("mode 1 byte offset fits u16");
                }
            }
            offsets.into_boxed_slice()
        })
    });
    let log2_bytes =
        usize::try_from(bytes_per_element.trailing_zeros()).expect("mode 1 log2 bytes fits usize");
    let table = &offsets[log2_bytes];
    debug_assert_eq!(table.len(), block_width * block_height);
    table
}

fn mode9_block_byte_offsets(
    bytes_per_element: usize,
    block_width: usize,
    block_height: usize,
) -> &'static [u16] {
    static OFFSETS: OnceLock<[Box<[u16]>; 5]> = OnceLock::new();
    let offsets = OFFSETS.get_or_init(|| {
        std::array::from_fn(|log2_bytes| {
            let bytes_per_element = 1 << log2_bytes;
            let (block_width, block_height) = tiled_block_dimensions(65_536, bytes_per_element)
                .expect("mode 9 table element size is supported");
            let mut offsets = vec![0_u16; block_width * block_height];
            for local_y in 0..block_height {
                for local_x in 0..block_width {
                    offsets[local_y * block_width + local_x] = u16::try_from(
                        mode9_within_block_byte_offset(local_x, local_y, bytes_per_element),
                    )
                    .expect("mode 9 byte offset fits u16");
                }
            }
            offsets.into_boxed_slice()
        })
    });
    let log2_bytes =
        usize::try_from(bytes_per_element.trailing_zeros()).expect("mode 9 log2 bytes fits usize");
    let table = &offsets[log2_bytes];
    debug_assert_eq!(table.len(), block_width * block_height);
    table
}

fn mode9_within_block_byte_offset(x: usize, y: usize, bytes_per_element: usize) -> usize {
    let bit = |value: usize, index: usize| (value >> index) & 1;
    let log2_bytes =
        usize::try_from(bytes_per_element.trailing_zeros()).expect("mode 9 log2 bytes fits usize");
    // ADDR_SW_64KB_S is the standard, non-XOR GFX10 64 KiB layout. These are the
    // exact address bits assembled by GFX10_SW_64K_S_PATINFO for one through
    // sixteen byte elements. Low constant bits select a byte within the element;
    // the remaining bits interleave X and Y according to the hardware equation.
    let address_bits = match log2_bytes {
        0 => [
            bit(x, 0),
            bit(x, 1),
            bit(x, 2),
            bit(x, 3),
            bit(y, 0),
            bit(y, 1),
            bit(y, 2),
            bit(y, 3),
            bit(y, 4),
            bit(x, 4),
            bit(y, 5),
            bit(x, 5),
            bit(y, 6),
            bit(x, 6),
            bit(y, 7),
            bit(x, 7),
        ],
        1 => [
            0,
            bit(x, 0),
            bit(x, 1),
            bit(x, 2),
            bit(y, 0),
            bit(y, 1),
            bit(y, 2),
            bit(x, 3),
            bit(y, 3),
            bit(x, 4),
            bit(y, 4),
            bit(x, 5),
            bit(y, 5),
            bit(x, 6),
            bit(y, 6),
            bit(x, 7),
        ],
        2 => [
            0,
            0,
            bit(x, 0),
            bit(x, 1),
            bit(y, 0),
            bit(y, 1),
            bit(y, 2),
            bit(x, 2),
            bit(y, 3),
            bit(x, 3),
            bit(y, 4),
            bit(x, 4),
            bit(y, 5),
            bit(x, 5),
            bit(y, 6),
            bit(x, 6),
        ],
        3 => [
            0,
            0,
            0,
            bit(x, 0),
            bit(y, 0),
            bit(y, 1),
            bit(x, 1),
            bit(x, 2),
            bit(y, 2),
            bit(x, 3),
            bit(y, 3),
            bit(x, 4),
            bit(y, 4),
            bit(x, 5),
            bit(y, 5),
            bit(x, 6),
        ],
        4 => [
            0,
            0,
            0,
            0,
            bit(y, 0),
            bit(y, 1),
            bit(x, 0),
            bit(x, 1),
            bit(y, 2),
            bit(x, 2),
            bit(y, 3),
            bit(x, 3),
            bit(y, 4),
            bit(x, 4),
            bit(y, 5),
            bit(x, 5),
        ],
        _ => unreachable!("mode 9 element size was validated"),
    };
    address_bits
        .iter()
        .enumerate()
        .fold(0, |offset, (index, &value)| offset | (value << index))
}

fn mode24_block_byte_offsets(
    block_x: usize,
    block_y: usize,
    bytes_per_element: usize,
    block_width: usize,
    block_height: usize,
) -> &'static [u16] {
    static OFFSETS: OnceLock<[Box<[u16]>; 4]> = OnceLock::new();
    let offsets = OFFSETS.get_or_init(|| {
        std::array::from_fn(|log2_bytes| {
            let bytes_per_element = 1 << log2_bytes;
            let (block_width, block_height) = tiled_block_dimensions(65_536, bytes_per_element)
                .expect("mode 24 table element size is supported");
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
                                u16::try_from(mode24_within_block_byte_offset_for_format(
                                    global_x,
                                    global_y,
                                    bytes_per_element,
                                ))
                                .expect("mode 24 byte offset fits u16");
                        }
                    }
                }
            }
            offsets.into_boxed_slice()
        })
    });
    let log2_bytes =
        usize::try_from(bytes_per_element.trailing_zeros()).expect("mode 24 log2 bytes fits usize");
    let elements_per_block = block_width * block_height;
    let table = ((block_y & 1) * 2 + (block_x & 1)) * elements_per_block;
    &offsets[log2_bytes][table..table + elements_per_block]
}

fn mode24_within_block_byte_offset_for_format(
    x: usize,
    y: usize,
    bytes_per_element: usize,
) -> usize {
    let log2_bytes =
        usize::try_from(bytes_per_element.trailing_zeros()).expect("mode 24 log2 bytes fits usize");
    // GFX10 64KB_Z_X has a distinct complete address equation; only replacing the
    // low byte of the render-target equation corrupts the upper nibble ordering.
    // Keep this in sync with KytyPS5's current Depth64KB*OffsetBytes reference.
    match log2_bytes {
        0 => {
            (x & 0x0001)
                ^ ((x << 1) & 0x0004)
                ^ ((x << 2) & 0x0010)
                ^ ((x << 3) & 0x0040)
                ^ ((x << 5) & 0x0300)
                ^ ((x << 4) & 0x0400)
                ^ ((x << 6) & 0x0800)
                ^ ((x << 7) & 0x2000)
                ^ ((x << 8) & 0x8000)
                ^ ((y << 1) & 0x0002)
                ^ ((y << 2) & 0x0008)
                ^ ((y << 3) & 0x00a0)
                ^ ((y << 5) & 0x0f00)
                ^ ((y << 6) & 0x1000)
                ^ ((y << 7) & 0x4000)
        }
        1 => {
            ((x << 1) & 0x0002)
                ^ ((x << 2) & 0x0008)
                ^ ((x << 3) & 0x0020)
                ^ ((x << 4) & 0x0480)
                ^ ((x << 5) & 0x0300)
                ^ ((x << 6) & 0x0800)
                ^ ((x << 7) & 0x2000)
                ^ ((x << 8) & 0x8000)
                ^ ((y << 2) & 0x0004)
                ^ ((y << 3) & 0x0010)
                ^ ((y << 4) & 0x0040)
                ^ ((y << 5) & 0x0f00)
                ^ ((y << 8) & 0x5000)
        }
        2 => {
            ((x << 2) & 0x0004)
                ^ ((x << 3) & 0x0010)
                ^ ((x << 4) & 0x0440)
                ^ ((x << 5) & 0x0300)
                ^ ((x << 6) & 0x0800)
                ^ ((x << 9) & 0xa000)
                ^ ((y << 3) & 0x0008)
                ^ ((y << 4) & 0x0020)
                ^ ((y << 5) & 0x0f80)
                ^ ((y << 9) & 0x1000)
                ^ ((y << 8) & 0x4000)
        }
        3 => {
            ((x << 3) & 0x0008)
                ^ ((x << 4) & 0x0420)
                ^ ((x << 5) & 0x0380)
                ^ ((x << 6) & 0x0800)
                ^ ((x << 10) & 0x2000)
                ^ ((x << 9) & 0x8000)
                ^ ((y << 4) & 0x0010)
                ^ ((y << 5) & 0x0f40)
                ^ ((y << 10) & 0x5000)
        }
        // KytyPS5 and the GFX10 depth layout support at most 64-bit elements.
        _ => unreachable!("mode 24 element size was validated"),
    }
}

fn mode5_block_byte_offsets(
    bytes_per_element: usize,
    block_width: usize,
    block_height: usize,
) -> &'static [u16] {
    static OFFSETS: OnceLock<[Box<[u16]>; 5]> = OnceLock::new();
    let offsets = OFFSETS.get_or_init(|| {
        std::array::from_fn(|log2_bytes| {
            let bytes_per_element = 1 << log2_bytes;
            let (block_width, block_height) = tiled_block_dimensions(4_096, bytes_per_element)
                .expect("mode 5 table element size is supported");
            let mut offsets = vec![0_u16; block_width * block_height];
            for local_y in 0..block_height {
                for local_x in 0..block_width {
                    offsets[local_y * block_width + local_x] = u16::try_from(
                        mode5_within_block_byte_offset(local_x, local_y, bytes_per_element),
                    )
                    .expect("mode 5 byte offset fits u16");
                }
            }
            offsets.into_boxed_slice()
        })
    });
    let log2_bytes =
        usize::try_from(bytes_per_element.trailing_zeros()).expect("mode 5 log2 bytes fits usize");
    let table = &offsets[log2_bytes];
    debug_assert_eq!(table.len(), block_width * block_height);
    table
}

fn mode5_within_block_byte_offset(x: usize, y: usize, bytes_per_element: usize) -> usize {
    let bit = |value: usize, index: usize| (value >> index) & 1;
    let log2_bytes =
        usize::try_from(bytes_per_element.trailing_zeros()).expect("mode 5 log2 bytes fits usize");
    let address_bits = match log2_bytes {
        0 => [
            bit(x, 0),
            bit(x, 1),
            bit(x, 2),
            bit(x, 3),
            bit(y, 0),
            bit(y, 1),
            bit(y, 2),
            bit(y, 3),
            bit(y, 4),
            bit(x, 4),
            bit(y, 5),
            bit(x, 5),
        ],
        1 => [
            0,
            bit(x, 0),
            bit(x, 1),
            bit(x, 2),
            bit(y, 0),
            bit(y, 1),
            bit(y, 2),
            bit(x, 3),
            bit(y, 3),
            bit(x, 4),
            bit(y, 4),
            bit(x, 5),
        ],
        2 => [
            0,
            0,
            bit(x, 0),
            bit(x, 1),
            bit(y, 0),
            bit(y, 1),
            bit(y, 2),
            bit(x, 2),
            bit(y, 3),
            bit(x, 3),
            bit(y, 4),
            bit(x, 4),
        ],
        3 => [
            0,
            0,
            0,
            bit(x, 0),
            bit(y, 0),
            bit(y, 1),
            bit(x, 1),
            bit(x, 2),
            bit(y, 2),
            bit(x, 3),
            bit(y, 3),
            bit(x, 4),
        ],
        4 => [
            0,
            0,
            0,
            0,
            bit(y, 0),
            bit(y, 1),
            bit(x, 0),
            bit(x, 1),
            bit(y, 2),
            bit(x, 2),
            bit(y, 3),
            bit(x, 3),
        ],
        _ => unreachable!("mode 5 element size was validated"),
    };
    address_bits
        .iter()
        .enumerate()
        .fold(0, |offset, (index, &value)| offset | (value << index))
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
    tiled_block_dimensions(65_536, bytes_per_pixel)
}

const fn mode27_surface_z_xor(surface_z: usize) -> usize {
    // GFX10 RenderTarget64KB folds the low four surface-Z bits into address bits 11..8.
    // This applies to each depth/array slice independently even though the block itself is thin.
    ((surface_z & 8) << 5)
        ^ ((surface_z & 4) << 7)
        ^ ((surface_z & 2) << 9)
        ^ ((surface_z & 1) << 11)
}

fn mode27_within_block_byte_offset_for_format(x: usize, y: usize, bytes_per_pixel: usize) -> usize {
    let log2_bytes =
        usize::try_from(bytes_per_pixel.trailing_zeros()).expect("mode 27 log2 bytes fits usize");
    // Exact GFX10 64KB_R_X equation used by current KytyPS5. The previous
    // PATINFO-derived bit table was a different pipe configuration and produced
    // the striped/barcode render-target uploads visible in Silent Hill.
    match log2_bytes {
        0 => {
            ((y << 2) & 0x0008)
                ^ ((y << 4) & 0x0010)
                ^ ((y << 3) & 0x00a0)
                ^ ((y << 5) & 0x0f00)
                ^ ((y << 6) & 0x1000)
                ^ ((y << 7) & 0x4000)
                ^ (x & 0x0007)
                ^ ((x << 3) & 0x0040)
                ^ ((x << 5) & 0x0300)
                ^ ((x << 4) & 0x0400)
                ^ ((x << 6) & 0x0800)
                ^ ((x << 7) & 0x2000)
                ^ ((x << 8) & 0x8000)
        }
        1 => {
            ((y << 4) & 0x0070)
                ^ ((y << 5) & 0x0f00)
                ^ ((y << 8) & 0x5000)
                ^ ((x << 1) & 0x000e)
                ^ ((x << 4) & 0x0480)
                ^ ((x << 5) & 0x0300)
                ^ ((x << 6) & 0x0800)
                ^ ((x << 7) & 0x2000)
                ^ ((x << 8) & 0x8000)
        }
        2 => {
            ((y << 4) & 0x0070)
                ^ ((y << 5) & 0x0f00)
                ^ ((y << 9) & 0x1000)
                ^ ((y << 8) & 0x4000)
                ^ ((x << 2) & 0x000c)
                ^ ((x << 5) & 0x0380)
                ^ ((x << 4) & 0x0400)
                ^ ((x << 6) & 0x0800)
                ^ ((x << 9) & 0xa000)
        }
        3 => {
            ((y << 4) & 0x0010)
                ^ ((y << 6) & 0x0080)
                ^ ((y << 5) & 0x0f00)
                ^ ((y << 10) & 0x5000)
                ^ ((x << 3) & 0x0008)
                ^ ((x << 4) & 0x0460)
                ^ ((x << 5) & 0x0300)
                ^ ((x << 6) & 0x0800)
                ^ ((x << 10) & 0x2000)
                ^ ((x << 9) & 0x8000)
        }
        4 => {
            ((x << 4) & 0x0410)
                ^ ((x << 5) & 0x0340)
                ^ ((x << 6) & 0x0800)
                ^ ((x << 11) & 0xa000)
                ^ ((y << 5) & 0x0f20)
                ^ ((y << 6) & 0x0080)
                ^ ((y << 10) & 0x1000)
                ^ ((y << 11) & 0x4000)
        }
        _ => unreachable!("mode 27 bytes per pixel was validated"),
    }
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
                self.persist_render_pipeline_archive();
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
        if let Err(error) = self.flush_guest_command_batch(false, GuestBatchFlushReason::EventLoop)
        {
            eprintln!("[REMu][METAL] guest command batch submission failed: {error}");
        }
        self.retire_completed_presentations();
        self.update_window_title();
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
                presenter.flush_guest_command_batch(false, GuestBatchFlushReason::EventLoop)
            } else {
                Ok(())
            };
            (flow, submission)
        });
        submission.map_err(|error| format!("guest command batch submission failed: {error}"))?;
        presenter.retire_completed_presentations();
        presenter.update_window_title();
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
        CompileOptions, Device, MTLBlendFactor, MTLBlendOperation, MTLClearColor,
        MTLCompareFunction, MTLCullMode, MTLLoadAction, MTLPixelFormat, MTLPrimitiveTopologyClass,
        MTLPrimitiveType, MTLResourceOptions, MTLSize, MTLStorageMode, MTLStoreAction,
        MTLTextureType, MTLTextureUsage, MTLTriangleFillMode, MTLWinding, RenderPassDescriptor,
        RenderPipelineDescriptor, TextureDescriptor,
    };
    use remu_gpu::{
        ComputeGrid, ComputeLaunch, GpuColorTarget, GpuEvent, GpuMemoryAcquire, GpuMemoryRelease,
        GpuMemoryReleaseValue, GpuMemoryWait, GpuMemoryWrite, GpuPipelineBarrier, GpuPipelineStage,
        GpuRegister, GpuScanoutTarget, IndexedDraw, gpu_event_channel,
    };
    use remu_shader::{
        GlobalMemoryBinding, ImageBinding, ImageControl, Instruction, Opcode, Operand,
        PixelOutputKind, Program, ShaderEncoding, ShaderEvaluation,
    };

    use super::{
        CB_COLOR_REGISTER_STRIDE, CB_COLOR0_ATTRIB, CB_COLOR0_ATTRIB2, CB_COLOR0_ATTRIB3,
        CB_COLOR0_BASE, CB_COLOR0_BASE_EXT, CB_COLOR0_INFO, CB_COLOR0_VIEW, CB_TARGET_MASK,
        DB_DEPTH_CONTROL, DB_DEPTH_SIZE_XY, DB_Z_INFO, DB_Z_READ_BASE, DB_Z_READ_BASE_HI,
        DB_Z_WRITE_BASE, DB_Z_WRITE_BASE_HI, DecodedRasterState, DecodedRenderTarget,
        DisplayConfiguration, GraphicsPipelineKey, GraphicsShaderResourceKey,
        GuestBatchFlushReason, GuestCommandBatch, GuestMemoryWaitKey, GuestMemoryWaitRecovery,
        GuestMemoryWaitStatus, GuestMemoryWaitTracker, GuestRenderPassKey, GuestTextureDescriptor,
        MetalPresenter, OpenGuestRenderPass, OrderedEventPoll, OrderedEventSource,
        OrderedGpuEventReceiver, PA_SU_SC_MODE_CNTL, PendingFlip, PendingGuestMemoryWrite,
        PendingTextureReadback, PresentationResolution, QueueBlockReason, SPI_PS_INPUT_CNTL_0,
        TextureReadbackLayout, apply_image_destination_select, apply_pending_guest_memory_writes,
        canonical_sampler_bindings, canonical_texture_bindings, comparison_depth32_upload,
        decode_depth_target, decode_image_descriptor, decode_metal_blend_attachment,
        decode_mode27_mip_chain_from_raw, decode_raster_state, decode_render_target_format,
        decode_render_targets_from_registers, decode_scanout_target, decode_scissor,
        decode_viewport, effective_linear_pitch, encode_mode27_mip_chain_into_raw,
        fallback_image_descriptor, graphics_shader_resource_key, guest_shared_mapping_identity,
        guest_texture_byte_count, guest_texture_descriptors_overlap,
        guest_texture_descriptors_share_component_reordered_storage,
        guest_texture_descriptors_share_storage, guest_texture_raw_matches,
        half_float_to_srgb_unorm8, half_float_to_unorm8, headless_metal_value_is_enabled,
        image_component_class, image_destination_select_is_native,
        image_resource_descriptor_is_null, initial_depth_clear_value, linear_unorm8_to_srgb_unorm8,
        metal_color_write_mask, metal_nonindexed_vertex_count, metal_primitive_topology_class,
        metal_primitive_type, mode5_within_block_byte_offset, mode9_within_block_byte_offset,
        mode24_within_block_byte_offset_for_format, mode27_block_dimensions,
        mode27_mip_chain_layout, mode27_source_byte_offset, mode27_within_block_byte_offset,
        mode27_within_block_byte_offset_for_format, native_texture_capture_to_bgra,
        new_red_blue_swizzled_texture_view, parse_diagnostic_hex_address,
        parse_presentation_resolution, pixel_input_controls, presentation_drawable_size,
        presentation_viewport, process_resident_bytes, raw_video_out_layout_is_linear,
        read_draw_indices, read_texture_linear, render_resolve_candidate_matches, sampler_lod_bias,
        sampler_uses_normalized_coordinates, shader_scalar_metadata,
        shader_scalar_metadata_for_registers, should_record_diagnostic_target_writer,
        snapshot_guest_texture_linear, snapshot_guest_texture_raw,
        snapshot_watched_guest_texture_raw, tiled_block_dimensions, unsigned_float_to_f32,
        validate_compute_launch_for_metal, watch_guest_texture, wave64_split_blockers,
        write_guest_texture_linear, write_watched_guest_texture_linear,
    };
    use remu_memory::CpuWriteWatchRetirement;

    #[test]
    fn diagnostic_shader_address_accepts_prefixed_and_plain_hex() {
        assert_eq!(
            parse_diagnostic_hex_address("8007e20000"),
            Some(0x8007_e20000)
        );
        assert_eq!(
            parse_diagnostic_hex_address("0x8007E20000"),
            Some(0x8007_e20000)
        );
        assert_eq!(parse_diagnostic_hex_address(""), None);
        assert_eq!(parse_diagnostic_hex_address("not-hex"), None);
    }

    #[test]
    fn pre_pixel_shader_diagnostic_records_ordinary_render_targets() {
        assert!(should_record_diagnostic_target_writer(true, false, false));
        assert!(should_record_diagnostic_target_writer(true, true, false));
        assert!(should_record_diagnostic_target_writer(false, true, true));
        assert!(!should_record_diagnostic_target_writer(false, true, false));
        assert!(!should_record_diagnostic_target_writer(false, false, true));
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
    fn raw_video_out_upload_accepts_only_the_abi_linear_tiling_mode() {
        assert!(raw_video_out_layout_is_linear(1, 3_840, 3_840));
        assert!(!raw_video_out_layout_is_linear(0, 3_840, 3_840));
        assert!(!raw_video_out_layout_is_linear(1, 1_919, 1_920));
        assert!(!raw_video_out_layout_is_linear(27, 3_840, 3_840));
    }

    #[test]
    fn reads_live_process_resident_memory_for_window_telemetry() {
        assert!(process_resident_bytes().is_some_and(|bytes| bytes > 0));
    }

    #[test]
    fn partial_pipeline_flushes_coalesce_without_reordering_submission_completion() {
        let (sender, receiver) = gpu_event_channel();
        let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
        let device = presenter.device.as_ref().expect("Metal device").clone();
        let queue = presenter
            .command_queue
            .as_ref()
            .expect("Metal queue")
            .clone();
        let texture_descriptor = TextureDescriptor::new();
        texture_descriptor.set_texture_type(MTLTextureType::D2);
        texture_descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        texture_descriptor.set_width(4);
        texture_descriptor.set_height(4);
        texture_descriptor.set_storage_mode(MTLStorageMode::Private);
        texture_descriptor.set_usage(MTLTextureUsage::RenderTarget);
        let texture = device.new_texture(&texture_descriptor);
        let render_pass = RenderPassDescriptor::new();
        let attachment = render_pass
            .color_attachments()
            .object_at(0)
            .expect("color attachment zero");
        attachment.set_texture(Some(&texture));
        attachment.set_load_action(MTLLoadAction::Clear);
        attachment.set_store_action(MTLStoreAction::Store);
        attachment.set_clear_color(MTLClearColor::new(0.0, 0.0, 0.0, 1.0));

        let command_buffer = queue.new_command_buffer().to_owned();
        let original_command_buffer = command_buffer.clone();
        let encoder = command_buffer
            .new_render_command_encoder(render_pass)
            .to_owned();
        presenter.open_guest_batch = Some(GuestCommandBatch {
            command_buffer,
            work: Vec::new(),
            global_writebacks: Vec::new(),
            guest_write_ranges: Vec::new(),
            memory_writes: Vec::new(),
            render_pass: Some(OpenGuestRenderPass {
                key: GuestRenderPassKey {
                    targets: Vec::new(),
                    depth: None,
                    resolve_multisample_targets: false,
                },
                encoder,
            }),
            completion_sequences: vec![40],
        });
        sender
            .send(GpuEvent::QueueSubmission {
                queue_id: 7,
                events: vec![
                    GpuEvent::PipelineBarrier(GpuPipelineBarrier {
                        wait_for: GpuPipelineStage::PixelShader,
                    }),
                    GpuEvent::SubmissionBoundary(41),
                    GpuEvent::PipelineBarrier(GpuPipelineBarrier {
                        wait_for: GpuPipelineStage::ComputeShader,
                    }),
                    GpuEvent::SubmissionBoundary(42),
                    GpuEvent::PipelineBarrier(GpuPipelineBarrier {
                        wait_for: GpuPipelineStage::VertexShader,
                    }),
                ],
            })
            .expect("queue partial flushes");

        assert!(matches!(
            presenter.receive_events(),
            super::PresenterFlow::Continue
        ));
        let open = presenter
            .open_guest_batch
            .as_ref()
            .expect("partial flush retains the open Metal batch");
        assert!(
            open.render_pass.is_some(),
            "an in-pass barrier must not spill and reload unchanged render attachments"
        );
        assert!(std::ptr::eq(
            open.command_buffer.as_ref(),
            original_command_buffer.as_ref()
        ));
        assert_eq!(open.completion_sequences, [40, 41, 42]);
        assert!(presenter.committed_guest_batches.is_empty());
        assert_eq!(presenter.statistics.submitted_guest_batches, 0);
        assert_eq!(presenter.statistics.coalesced_pipeline_barriers, 3);

        let mut release_label = 0_u32;
        let release_address =
            u64::try_from((&raw mut release_label) as usize).expect("test label address fits u64");
        presenter
            .execute_guest_memory_release(GpuMemoryRelease {
                address: release_address,
                value: GpuMemoryReleaseValue::Immediate32(1),
            })
            .expect("a real release remains in the coalesced batch");
        let open = presenter
            .open_guest_batch
            .as_ref()
            .expect("release does not commit the open batch");
        assert!(open.render_pass.is_some());
        assert!(matches!(
            open.memory_writes.as_slice(),
            [PendingGuestMemoryWrite::Release(GpuMemoryRelease {
                address,
                value: GpuMemoryReleaseValue::Immediate32(1),
            })] if *address == release_address
        ));
        assert_eq!(release_label, 0, "the label remains deferred");
        assert!(presenter.committed_guest_batches.is_empty());
        assert_eq!(presenter.statistics.submitted_guest_batches, 0);
        assert_eq!(presenter.statistics.batch_flushes_for_guest_memory_write, 0);
        assert_eq!(presenter.statistics.render_pass_ends_for_batch_flush, 0);
        assert_eq!(presenter.statistics.render_pass_ends_for_compute, 0);
        assert_eq!(presenter.statistics.render_pass_ends_for_target_change, 0);

        presenter
            .flush_guest_command_batch(false, GuestBatchFlushReason::EventLoop)
            .expect("event loop commits deferred work");
        assert!(presenter.open_guest_batch.is_none());
        assert_eq!(presenter.statistics.submitted_guest_batches, 1);
        assert_eq!(presenter.statistics.batch_flushes_from_event_loop, 1);
        assert_eq!(presenter.statistics.render_pass_ends_for_batch_flush, 1);
        let committed = presenter
            .committed_guest_batches
            .pop_front()
            .expect("event loop retains one committed batch");
        assert_eq!(committed.completion_sequences, [40, 41, 42]);
        assert!(matches!(
            committed.memory_writes.as_slice(),
            [PendingGuestMemoryWrite::Release(GpuMemoryRelease {
                address,
                ..
            })] if *address == release_address
        ));
        committed.command_buffer.wait_until_completed();
        assert_eq!(
            committed.command_buffer.status(),
            metal::MTLCommandBufferStatus::Completed
        );
        apply_pending_guest_memory_writes(&committed.memory_writes)
            .expect("completed release publishes its label");
        assert_eq!(release_label, 1);
    }

    #[test]
    fn pending_label_writes_coalesce_until_a_blocked_wait_commits_them() {
        let (_sender, receiver) = gpu_event_channel();
        let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
        let queue = presenter
            .command_queue
            .as_ref()
            .expect("Metal queue")
            .clone();
        presenter.open_guest_batch = Some(GuestCommandBatch {
            command_buffer: queue.new_command_buffer().to_owned(),
            work: Vec::new(),
            global_writebacks: Vec::new(),
            guest_write_ranges: Vec::new(),
            memory_writes: Vec::new(),
            render_pass: None,
            completion_sequences: Vec::new(),
        });

        let mut label = 0_u32;
        let address =
            u64::try_from((&raw mut label) as usize).expect("test label address fits u64");
        for value in [1, 2] {
            presenter
                .execute_guest_memory_release(GpuMemoryRelease {
                    address,
                    value: GpuMemoryReleaseValue::Immediate32(value),
                })
                .expect("label release coalesces");
        }
        assert_eq!(label, 0, "open-batch label writes remain deferred");
        assert_eq!(
            presenter.pending_guest_memory_value(address, false),
            Some(2)
        );
        assert_eq!(
            presenter
                .open_guest_batch
                .as_ref()
                .expect("label writes retain the open batch")
                .memory_writes
                .len(),
            2
        );
        assert!(presenter.committed_guest_batches.is_empty());
        assert_eq!(presenter.statistics.submitted_guest_batches, 0);

        let satisfied = GpuMemoryWait {
            address,
            reference: 2,
            mask: u64::from(u32::MAX),
            compare_function: 3,
            is_64_bit: false,
        };
        assert_eq!(
            presenter
                .execute_guest_memory_wait(OrderedEventSource::GuestQueue(77), None, satisfied)
                .expect("latest deferred value satisfies the ordered wait"),
            GuestMemoryWaitStatus::Satisfied
        );
        assert!(presenter.open_guest_batch.is_some());
        assert_eq!(presenter.statistics.submitted_guest_batches, 0);

        let blocked = GpuMemoryWait {
            reference: 3,
            ..satisfied
        };
        assert_eq!(
            presenter
                .execute_guest_memory_wait(OrderedEventSource::GuestQueue(77), None, blocked)
                .expect("nonmatching deferred value blocks"),
            GuestMemoryWaitStatus::Blocked
        );
        assert!(presenter.open_guest_batch.is_none());
        assert_eq!(presenter.committed_guest_batches.len(), 1);
        assert_eq!(presenter.statistics.submitted_guest_batches, 1);
        assert_eq!(presenter.statistics.batch_flushes_for_blocked_wait, 1);
        assert_eq!(label, 0, "blocked wait commits before CPU publication");
        assert_eq!(
            presenter.pending_guest_memory_value(address, false),
            Some(2)
        );

        presenter
            .wait_for_guest_command_batches()
            .expect("committed label batch completes");
        assert!(presenter.committed_guest_batches.is_empty());
        assert_eq!(label, 2, "deferred writes publish in exact packet order");
    }

    #[test]
    fn terminal_gpu_failure_still_publishes_release_and_latches_ordered_wait() {
        let (_sender, receiver) = gpu_event_channel();
        let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
        let mut label = 0_u32;
        let address =
            u64::try_from((&raw mut label) as usize).expect("test label address fits u64");
        let wait = GpuMemoryWait {
            address,
            reference: 7,
            mask: u64::from(u32::MAX),
            compare_function: 3,
            is_64_bit: false,
        };
        let key = GuestMemoryWaitKey::new(OrderedEventSource::GuestQueue(19), wait);
        assert_eq!(
            presenter.guest_memory_waits.observe_unsatisfied(key),
            GuestMemoryWaitRecovery::None
        );
        let pending = [PendingGuestMemoryWrite::Release(GpuMemoryRelease {
            address,
            value: GpuMemoryReleaseValue::Immediate32(7),
        })];

        let error = presenter
            .publish_terminal_guest_memory_writes(
                &pending,
                Err("Metal render pipeline rejected the ordered draw".to_owned()),
            )
            .expect_err("the original GPU failure must remain visible");

        assert_eq!(
            error, "Metal render pipeline rejected the ordered draw",
            "publishing the completion label must not erase the primary failure"
        );
        assert_eq!(
            label, 7,
            "terminal failure must not strand its following RELEASE_MEM"
        );
        assert_eq!(
            presenter.guest_memory_waits.observe_unsatisfied(key),
            GuestMemoryWaitRecovery::ProducerLatched(7),
            "the ordered producer must wake a suspended WAIT_REG_MEM"
        );
    }

    #[test]
    fn synchronous_failure_drains_later_batches_and_their_completion_labels() {
        let (_sender, receiver) = gpu_event_channel();
        let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
        let queue = presenter
            .command_queue
            .as_ref()
            .expect("Metal queue")
            .clone();
        let mut completion_label = 0_u32;
        let completion_address = u64::try_from((&raw mut completion_label) as usize)
            .expect("test label address fits u64");
        for memory_writes in [
            vec![PendingGuestMemoryWrite::Release(GpuMemoryRelease {
                address: 2,
                value: GpuMemoryReleaseValue::Immediate32(1),
            })],
            vec![PendingGuestMemoryWrite::Release(GpuMemoryRelease {
                address: completion_address,
                value: GpuMemoryReleaseValue::Immediate32(9),
            })],
        ] {
            let command_buffer = queue.new_command_buffer().to_owned();
            command_buffer.commit();
            presenter
                .committed_guest_batches
                .push_back(GuestCommandBatch {
                    command_buffer,
                    work: Vec::new(),
                    global_writebacks: Vec::new(),
                    guest_write_ranges: Vec::new(),
                    memory_writes,
                    render_pass: None,
                    completion_sequences: Vec::new(),
                });
        }

        let error = presenter
            .wait_for_guest_command_batches()
            .expect_err("the invalid first release must remain an error");

        assert!(error.contains("ordered guest memory publication"));
        assert!(error.contains("unaligned"));
        assert!(
            presenter.committed_guest_batches.is_empty(),
            "a failed terminal batch must not abandon later committed tickets"
        );
        assert_eq!(
            completion_label, 9,
            "the later batch's ordered completion label must still publish"
        );
    }

    #[test]
    fn decodes_force_unnormalized_sampler_coordinates() {
        assert!(sampler_uses_normalized_coordinates(&[0, 0, 0, 0]));
        assert!(!sampler_uses_normalized_coordinates(&[1 << 15, 0, 0, 0,]));
    }

    #[test]
    fn decodes_signed_sampler_lod_bias() {
        assert_eq!(sampler_lod_bias(&[0, 0, 0x03bf, 0]), 959.0 / 256.0);
        assert_eq!(sampler_lod_bias(&[0, 0, 0x3fff, 0]), -1.0 / 256.0);
        assert_eq!(sampler_lod_bias(&[0, 0, 0x2000, 0]), -32.0);
    }

    #[test]
    fn pixel_input_controls_default_to_identity_and_preserve_programmed_remaps() {
        assert_eq!(
            pixel_input_controls(&[]),
            std::array::from_fn(|index| u32::try_from(index).expect("index fits u32"))
        );
        let registers = [
            GpuRegister {
                offset: SPI_PS_INPUT_CNTL_0,
                value: 4,
            },
            GpuRegister {
                offset: SPI_PS_INPUT_CNTL_0 + 1,
                value: 0x220,
            },
        ];
        let controls = pixel_input_controls(&registers);
        assert_eq!(controls[0], 4);
        assert_eq!(controls[1], 0x220);
        assert_eq!(controls[2], 2);
    }

    #[test]
    fn comparison_sampling_converts_srgb_color_red_to_linear_depth() {
        let upload = comparison_depth32_upload(
            MTLPixelFormat::RGBA8Unorm_sRGB,
            &[0, 1, 2, 255, 128, 3, 4, 255, 255, 5, 6, 255],
        )
        .expect("sRGB color texture is comparison-compatible")
        .expect("sRGB color texture needs a depth upload");
        let depths = upload
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("one float")))
            .collect::<Vec<_>>();
        assert_eq!(depths[0], 0.0);
        assert!((depths[1] - 0.215_86).abs() < 0.000_1);
        assert_eq!(depths[2], 1.0);
    }

    #[test]
    fn ordered_guest_label_writes_publish_only_the_declared_values() {
        let mut release_value = 0_u64;
        let mut words = [0_u32; 3];
        let mut fill_words = [0_u32; 3];
        let release = PendingGuestMemoryWrite::Release(GpuMemoryRelease {
            address: std::ptr::addr_of_mut!(release_value) as u64,
            value: GpuMemoryReleaseValue::Immediate64(0x1122_3344_5566_7788),
        });
        let write = PendingGuestMemoryWrite::Write(GpuMemoryWrite {
            address: words.as_mut_ptr() as u64,
            values: vec![3, 5, 8],
            increment_address: true,
        });
        let fill = PendingGuestMemoryWrite::Fill {
            address: fill_words.as_mut_ptr() as u64,
            value: 0xaabb_ccdd,
            dword_count: 3,
        };
        assert_eq!(
            release.value_at(std::ptr::addr_of!(release_value) as u64, true),
            Some(0x1122_3344_5566_7788)
        );
        assert_eq!(write.value_at(words.as_ptr() as u64 + 8, false), Some(8));
        assert_eq!(
            fill.value_at(fill_words.as_ptr() as u64 + 4, false),
            Some(0xaabb_ccdd)
        );
        assert_eq!(
            fill.value_at(fill_words.as_ptr() as u64, true),
            Some(0xaabb_ccdd_aabb_ccdd)
        );
        apply_pending_guest_memory_writes(&[release, write, fill]).expect("ordered guest writes");
        assert_eq!(release_value, 0x1122_3344_5566_7788);
        assert_eq!(words, [3, 5, 8]);
        assert_eq!(fill_words, [0xaabb_ccdd; 3]);
    }

    #[test]
    fn graphics_shader_cache_key_keeps_dynamic_global_addresses_out_of_pipeline_identity() {
        let evaluation = |base_address| ShaderEvaluation {
            initial_scalar_registers: Vec::new().into(),
            scalar_registers: Vec::new().into(),
            scalar_memory_reads: Vec::new(),
            image_bindings: Vec::new(),
            global_memory_bindings: vec![GlobalMemoryBinding {
                scalar_address: 4,
                base_address,
                instruction_pcs: vec![0x18c].into(),
                byte_count: 64,
                data: vec![0; 64],
                writable: false,
                write_back_to_guest: true,
            }],
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        };

        assert_eq!(
            graphics_shader_resource_key(&evaluation(0x7009_0605_ac), false),
            graphics_shader_resource_key(&evaluation(0x7009_1600_6c), false),
        );
        assert_eq!(
            graphics_shader_resource_key(&evaluation(0x7009_0605_ac), true),
            graphics_shader_resource_key(&evaluation(0x7009_1600_6c), true),
        );
    }

    #[test]
    fn runtime_upload_representatives_follow_canonical_texture_and_sampler_slots() {
        let control = ImageControl {
            dmask: 0xf,
            vector_address: 0,
            address_registers: vec![0, 1],
            vector_data: 4,
            scalar_resource: 0,
            scalar_sampler: 8,
            dimension: 1,
            is_array: false,
            glc: false,
            slc: false,
            a16: false,
            d16: false,
        };
        let binding = |pc, opcode, descriptor: Vec<u32>| ImageBinding {
            pc,
            opcode: remu_shader::Opcode::Known(opcode),
            control: control.clone(),
            resource_descriptor: descriptor.into(),
            sampler_descriptor: if opcode.starts_with("ImageSample") {
                vec![7, 8, 9, 10].into()
            } else {
                Vec::new().into()
            },
            mip_level: None,
        };
        let descriptor_a = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let descriptor_b = vec![11, 12, 13, 14, 15, 16, 17, 18];
        let evaluation = ShaderEvaluation {
            initial_scalar_registers: Vec::new().into(),
            scalar_registers: Vec::new().into(),
            scalar_memory_reads: Vec::new(),
            image_bindings: vec![
                binding(0, "ImageSample", descriptor_a.clone()),
                binding(8, "ImageSampleLz", descriptor_a.clone()),
                binding(16, "ImageLoad", descriptor_a),
                binding(24, "ImageSample", descriptor_b),
            ],
            global_memory_bindings: Vec::new(),
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        };

        assert_eq!(
            canonical_texture_bindings(&evaluation)
                .iter()
                .map(|binding| binding.pc)
                .collect::<Vec<_>>(),
            [0, 16, 24]
        );
        assert_eq!(
            canonical_sampler_bindings(&evaluation)
                .iter()
                .map(|binding| binding.pc)
                .collect::<Vec<_>>(),
            [0]
        );
    }

    #[test]
    fn graphics_shader_cache_key_normalizes_dynamic_image_formats_and_swizzles() {
        let evaluation = |unified_format: u32, destination_select: u32| ShaderEvaluation {
            initial_scalar_registers: vec![0; 16].into(),
            scalar_registers: vec![0; 16].into(),
            scalar_memory_reads: Vec::new(),
            image_bindings: vec![ImageBinding {
                pc: 0x20,
                opcode: remu_shader::Opcode::Known("ImageSample"),
                control: ImageControl {
                    dmask: 0xf,
                    vector_address: 0,
                    address_registers: vec![0, 1],
                    vector_data: 4,
                    scalar_resource: 0,
                    scalar_sampler: 8,
                    dimension: 1,
                    is_array: false,
                    glc: false,
                    slc: false,
                    a16: false,
                    d16: false,
                },
                resource_descriptor: vec![0, unified_format << 20, 0, destination_select].into(),
                sampler_descriptor: vec![0; 4].into(),
                mip_level: None,
            }],
            global_memory_bindings: Vec::new(),
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        };

        // Both unified formats are float-class interfaces; pixel format and component selection
        // are supplied by the texture and live resource SGPRs at execution time.
        let first = evaluation(56, 0x924); // RGBA8 unorm
        let second = evaluation(58, 0xfac); // RGBA8 unsigned-scaled
        assert_eq!(image_component_class(&first.image_bindings[0]), 0);
        assert_eq!(image_component_class(&second.image_bindings[0]), 0);
        assert_eq!(
            graphics_shader_resource_key(&first, false),
            graphics_shader_resource_key(&second, false),
        );
    }

    #[test]
    fn appends_dynamic_global_base_and_size_metadata_after_guest_scalars() {
        let evaluation = ShaderEvaluation {
            initial_scalar_registers: vec![10, 20].into(),
            scalar_registers: vec![10, 20].into(),
            scalar_memory_reads: Vec::new(),
            image_bindings: Vec::new(),
            global_memory_bindings: vec![GlobalMemoryBinding {
                scalar_address: 4,
                base_address: 0x0070_1234_5678,
                instruction_pcs: vec![0x18c].into(),
                byte_count: 96,
                data: vec![0; 96],
                writable: true,
                write_back_to_guest: true,
            }],
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        };

        assert_eq!(
            shader_scalar_metadata(&evaluation).expect("metadata fits"),
            [10, 20, 0x1234_5678, 0x70, 96, 1]
        );
        assert_eq!(
            shader_scalar_metadata_for_registers(
                &evaluation,
                &[
                    GpuRegister {
                        offset: 0x240,
                        value: 30,
                    },
                    GpuRegister {
                        offset: 0x241,
                        value: 40,
                    },
                ],
                0x240,
                0,
            )
            .expect("live scalar metadata fits"),
            [30, 40, 0x1234_5678, 0x70, 96, 1]
        );
    }

    #[test]
    fn reuses_immutable_sampler_argument_buffers() {
        let (_sender, receiver) = gpu_event_channel();
        let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
        let device = presenter
            .device
            .as_ref()
            .expect("presenter retains its Metal device")
            .clone();
        let source = r"
#include <metal_stdlib>
using namespace metal;
struct RemuSamplers {
    sampler s0 [[id(0)]];
};
kernel void remu_sampler_cache_test(
    constant RemuSamplers& samplers [[buffer(0)]],
    device float4* output [[buffer(1)]],
    texture2d<float> texture [[texture(0)]],
    uint gid [[thread_position_in_grid]]) {
    output[gid] = texture.sample(samplers.s0, float2(0.5f));
}
";
        let library = device
            .new_library_with_source(source, &CompileOptions::new())
            .expect("sampler argument-buffer MSL compiles");
        let function = library
            .get_function("remu_sampler_cache_test", None)
            .expect("sampler cache function exists");
        let evaluation = ShaderEvaluation {
            initial_scalar_registers: Vec::new().into(),
            scalar_registers: Vec::new().into(),
            scalar_memory_reads: Vec::new(),
            image_bindings: vec![ImageBinding {
                pc: 0x100,
                opcode: Opcode::Known("ImageSample"),
                control: ImageControl {
                    dmask: 0xf,
                    vector_address: 0,
                    address_registers: vec![0, 1],
                    vector_data: 4,
                    scalar_resource: 0,
                    scalar_sampler: 8,
                    dimension: 1,
                    is_array: false,
                    glc: false,
                    slc: false,
                    a16: false,
                    d16: false,
                },
                resource_descriptor: vec![0; 8].into(),
                sampler_descriptor: vec![0; 4].into(),
                mip_level: None,
            }],
            global_memory_bindings: Vec::new(),
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        };

        let (first_buffer, first_samplers) = presenter
            .cached_sampler_argument_buffer(&device, &function, Some(0), &evaluation)
            .expect("first sampler argument buffer is created");
        let (second_buffer, second_samplers) = presenter
            .cached_sampler_argument_buffer(&device, &function, Some(0), &evaluation)
            .expect("identical sampler argument buffer is reused");

        assert_eq!(presenter.sampler_argument_cache.len(), 1);
        assert_eq!(first_samplers.len(), 1);
        assert_eq!(second_samplers.len(), 1);
        assert_eq!(
            first_buffer.expect("first buffer").contents(),
            second_buffer.expect("second buffer").contents()
        );
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
            index_snapshot: None,
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

    fn append_depth_target_registers(
        draw: &mut IndexedDraw,
        address: u64,
        width: u32,
        height: u32,
        z_info: u32,
        depth_control: u32,
    ) {
        let base = u32::try_from((address >> 8) & 0xffff_ffff).expect("depth base fits u32");
        let base_hi = u32::try_from((address >> 40) & 0xff).expect("depth base high fits u32");
        draw.context_registers.extend([
            GpuRegister {
                offset: DB_DEPTH_SIZE_XY,
                value: (width - 1) | ((height - 1) << 16),
            },
            GpuRegister {
                offset: DB_Z_INFO,
                value: z_info,
            },
            GpuRegister {
                offset: DB_Z_READ_BASE,
                value: base,
            },
            GpuRegister {
                offset: DB_Z_WRITE_BASE,
                value: base,
            },
            GpuRegister {
                offset: DB_Z_READ_BASE_HI,
                value: base_hi,
            },
            GpuRegister {
                offset: DB_Z_WRITE_BASE_HI,
                value: base_hi,
            },
            GpuRegister {
                offset: DB_DEPTH_CONTROL,
                value: depth_control,
            },
        ]);
    }

    #[test]
    fn decodes_observed_depth16_greater_equal_read_only_target() {
        let address = 0x0070_1ef9_0000;
        let mut draw = indexed_draw(0, 0, 0);
        append_depth_target_registers(&mut draw, address, 8, 8, 0xa000_0001, 0x62);

        let depth = decode_depth_target(&draw)
            .expect("valid depth state")
            .expect("enabled depth target");
        assert_eq!(depth.texture.address, address);
        assert_eq!((depth.texture.width, depth.texture.height), (8, 8));
        assert_eq!(depth.texture.pitch, 128);
        assert_eq!(depth.texture.tile_mode, 0);
        assert_eq!(depth.texture.pixel_format, MTLPixelFormat::Depth16Unorm);
        assert_eq!(depth.compare, MTLCompareFunction::GreaterEqual);
        assert!(!depth.write_enabled);
    }

    #[test]
    fn decodes_depth32_write_target_and_ignores_stencil_only_state() {
        let mut draw = indexed_draw(0, 0, 0);
        append_depth_target_registers(&mut draw, 0x0070_1412_0000, 1_920, 1_080, 0xa000_0003, 0x66);
        let depth = decode_depth_target(&draw)
            .expect("valid depth state")
            .expect("enabled depth target");
        assert_eq!(depth.texture.pixel_format, MTLPixelFormat::Depth32Float);
        assert_eq!(depth.compare, MTLCompareFunction::GreaterEqual);
        assert!(depth.write_enabled);

        draw.context_registers
            .iter_mut()
            .find(|register| register.offset == DB_DEPTH_CONTROL)
            .expect("depth control")
            .value = 0x771;
        assert_eq!(
            decode_depth_target(&draw).expect("stencil-only state is valid"),
            None
        );

        draw.context_registers
            .iter_mut()
            .find(|register| register.offset == DB_DEPTH_CONTROL)
            .expect("depth control")
            .value = 0x66;
        draw.context_registers
            .iter_mut()
            .find(|register| register.offset == DB_Z_INFO)
            .expect("depth info")
            .value = 0x8000_0000;
        assert_eq!(
            decode_depth_target(&draw).expect("format-zero depth state is valid"),
            None
        );
    }

    #[test]
    fn initializes_depth_to_the_far_value_for_the_active_comparison() {
        assert_eq!(
            initial_depth_clear_value(MTLCompareFunction::LessEqual),
            1.0
        );
        assert_eq!(
            initial_depth_clear_value(MTLCompareFunction::GreaterEqual),
            0.0
        );
        assert_eq!(initial_depth_clear_value(MTLCompareFunction::Greater), 0.0);
        assert_eq!(initial_depth_clear_value(MTLCompareFunction::Always), 1.0);
    }

    #[test]
    fn decodes_observed_clockwise_back_face_culling() {
        let mut draw = indexed_draw(0, 0, 0);
        draw.context_registers.push(GpuRegister {
            offset: PA_SU_SC_MODE_CNTL,
            value: 0x246,
        });

        assert_eq!(
            decode_raster_state(&draw).expect("observed raster state"),
            DecodedRasterState {
                cull_mode: MTLCullMode::Back,
                front_winding: MTLWinding::Clockwise,
                fill_mode: MTLTriangleFillMode::Fill,
            }
        );
    }

    #[test]
    fn compensates_winding_when_normalizing_negative_viewport_height() {
        let mut draw = indexed_draw(0, 0, 0);
        draw.context_registers.extend([
            GpuRegister {
                offset: PA_SU_SC_MODE_CNTL,
                value: 0x246,
            },
            GpuRegister {
                offset: 0x10f,
                value: 960.0_f32.to_bits(),
            },
            GpuRegister {
                offset: 0x110,
                value: 960.0_f32.to_bits(),
            },
            GpuRegister {
                offset: 0x111,
                value: (-540.0_f32).to_bits(),
            },
            GpuRegister {
                offset: 0x112,
                value: 540.0_f32.to_bits(),
            },
        ]);
        draw.context_registers
            .sort_by_key(|register| register.offset);

        assert_eq!(
            decode_raster_state(&draw).expect("negative-height viewport raster state"),
            DecodedRasterState {
                cull_mode: MTLCullMode::Back,
                front_winding: MTLWinding::CounterClockwise,
                fill_mode: MTLTriangleFillMode::Fill,
            }
        );
    }

    #[test]
    fn decodes_dual_polygon_line_mode_and_rejects_unrepresentable_states() {
        let mut draw = indexed_draw(0, 0, 0);
        draw.context_registers.push(GpuRegister {
            offset: PA_SU_SC_MODE_CNTL,
            value: (1 << 2) | (1 << 3) | (1 << 5) | (1 << 8),
        });
        assert_eq!(
            decode_raster_state(&draw)
                .expect("matching front/back line mode")
                .fill_mode,
            MTLTriangleFillMode::Lines
        );

        draw.context_registers[0].value |= 1 | 2;
        assert!(
            decode_raster_state(&draw)
                .expect_err("Metal cannot cull both orientations")
                .contains("simultaneous front- and back-face")
        );
    }

    #[test]
    fn queue_local_compile_gate_runs_compute_without_overtaking_graphics_or_flip() {
        let (sender, receiver) = gpu_event_channel();
        let mut ordered = OrderedGpuEventReceiver::new(receiver);
        let key = GraphicsPipelineKey {
            export_shader_address: 0x1000,
            pixel_shader_address: 0x2000,
            export_resources: GraphicsShaderResourceKey {
                initial_scalar_count: 0,
                global_bindings: Vec::new(),
                image_bindings: Vec::new(),
                vertex_inputs: Vec::new(),
            },
            pixel_resources: GraphicsShaderResourceKey {
                initial_scalar_count: 0,
                global_bindings: Vec::new(),
                image_bindings: Vec::new(),
                vertex_inputs: Vec::new(),
            },
            pixel_input_controls: std::array::from_fn(|index| {
                u32::try_from(index).expect("PS input index fits u32")
            }),
            targets: Vec::new(),
            depth: None,
            sample_count: 1,
            render_target_array_length: 1,
            primitive_topology_class: MTLPrimitiveTopologyClass::Triangle,
        };
        let mut draw = indexed_draw(0, 0, 0);
        draw.sequence = 17;
        let mut later_draw = indexed_draw(0, 0, 0);
        later_draw.sequence = 23;
        sender
            .send(GpuEvent::QueueSubmission {
                queue_id: 0,
                events: vec![
                    GpuEvent::IndexedDraw(Box::new(draw)),
                    GpuEvent::PipelineBarrier(remu_gpu::GpuPipelineBarrier {
                        wait_for: remu_gpu::GpuPipelineStage::PixelShader,
                    }),
                    GpuEvent::MemoryAcquire(GpuMemoryAcquire {
                        engine: 0,
                        cb_db_control: 0,
                        base_address: 0x2000,
                        size_bytes: 0x1000,
                        poll_interval: 0,
                        gcr_control: 0,
                    }),
                    GpuEvent::IndexedDraw(Box::new(later_draw)),
                    GpuEvent::SubmissionBoundary(1),
                ],
            })
            .expect("queue graphics submission");
        sender
            .send(GpuEvent::QueueSubmission {
                queue_id: 33,
                events: vec![
                    GpuEvent::MemoryRelease(GpuMemoryRelease {
                        address: 0x3000,
                        value: GpuMemoryReleaseValue::Immediate32(1),
                    }),
                    GpuEvent::SubmissionBoundary(2),
                ],
            })
            .expect("queue compute submission");
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

        let (graphics_source, blocked_draw) = match ordered.try_recv(|_| false) {
            OrderedEventPoll::Event {
                source,
                event: GpuEvent::IndexedDraw(draw),
            } => {
                assert_eq!(draw.sequence, 17);
                (source, GpuEvent::IndexedDraw(draw))
            }
            other => panic!("expected first graphics draw, got {other:?}"),
        };
        ordered.block_event(
            graphics_source,
            blocked_draw,
            QueueBlockReason::GraphicsPipeline(key),
        );

        let compute_source = match ordered.try_recv(|_| false) {
            OrderedEventPoll::Event {
                source,
                event: GpuEvent::MemoryRelease(release),
            } => {
                assert_eq!(release.address, 0x3000);
                source
            }
            other => panic!("expected independent compute release, got {other:?}"),
        };
        ordered.finish_event(compute_source);
        let compute_source = match ordered.try_recv(|_| false) {
            OrderedEventPoll::Event {
                source,
                event: GpuEvent::SubmissionBoundary(2),
            } => source,
            other => panic!("expected compute boundary, got {other:?}"),
        };
        ordered.finish_event(compute_source);

        assert!(matches!(
            ordered.try_recv(|_| false),
            OrderedEventPoll::Blocked
        ));
        let prefetched = ordered.prefetch_draws(16, 16);
        assert_eq!(
            prefetched
                .iter()
                .map(|draw| draw.sequence)
                .collect::<Vec<_>>(),
            [17, 23]
        );

        match ordered.try_recv(|_| true) {
            OrderedEventPoll::Event {
                source,
                event: GpuEvent::IndexedDraw(draw),
            } => {
                assert_eq!(draw.sequence, 17);
                ordered.finish_event(source);
            }
            other => panic!("expected the blocked draw exactly once, got {other:?}"),
        }
        let source = match ordered.try_recv(|_| true) {
            OrderedEventPoll::Event {
                source,
                event:
                    GpuEvent::PipelineBarrier(remu_gpu::GpuPipelineBarrier {
                        wait_for: remu_gpu::GpuPipelineStage::PixelShader,
                    }),
            } => source,
            other => panic!("expected graphics barrier, got {other:?}"),
        };
        ordered.finish_event(source);
        let source = match ordered.try_recv(|_| true) {
            OrderedEventPoll::Event {
                source,
                event:
                    GpuEvent::MemoryAcquire(GpuMemoryAcquire {
                        base_address: 0x2000,
                        size_bytes: 0x1000,
                        ..
                    }),
            } => source,
            other => panic!("expected graphics acquire, got {other:?}"),
        };
        ordered.finish_event(source);
        match ordered.try_recv(|_| true) {
            OrderedEventPoll::Event {
                source,
                event: GpuEvent::IndexedDraw(draw),
            } => {
                assert_eq!(draw.sequence, 23);
                ordered.finish_event(source);
            }
            other => panic!("expected the prefetched draw in exact order, got {other:?}"),
        }
        let source = match ordered.try_recv(|_| true) {
            OrderedEventPoll::Event {
                source,
                event: GpuEvent::SubmissionBoundary(1),
            } => source,
            other => panic!("expected graphics boundary, got {other:?}"),
        };
        ordered.finish_event(source);
        assert!(matches!(
            ordered.try_recv(|_| true),
            OrderedEventPoll::Event {
                event: GpuEvent::Flip {
                    address: 0x7000_0000,
                    ..
                },
                ..
            }
        ));
        assert!(matches!(
            ordered.try_recv(|_| true),
            OrderedEventPoll::Empty
        ));
    }

    #[test]
    fn suspended_memory_wait_keeps_same_queue_tail_behind_it() {
        let (sender, receiver) = gpu_event_channel();
        let mut ordered = OrderedGpuEventReceiver::new(receiver);
        let wait = GpuMemoryWait {
            address: 0x4000,
            reference: 1,
            mask: u64::from(u32::MAX),
            compare_function: 3,
            is_64_bit: false,
        };
        sender
            .send(GpuEvent::QueueSubmission {
                queue_id: 0,
                events: vec![
                    GpuEvent::MemoryWait(wait),
                    GpuEvent::MemoryWrite(GpuMemoryWrite {
                        address: 0x5000,
                        values: vec![9],
                        increment_address: true,
                    }),
                    GpuEvent::SubmissionBoundary(1),
                ],
            })
            .expect("queue waiting graphics submission");
        sender
            .send(GpuEvent::QueueSubmission {
                queue_id: 33,
                events: vec![
                    GpuEvent::MemoryRelease(GpuMemoryRelease {
                        address: wait.address,
                        value: GpuMemoryReleaseValue::Immediate32(1),
                    }),
                    GpuEvent::SubmissionBoundary(2),
                ],
            })
            .expect("queue satisfying compute submission");

        let source = match ordered.try_recv(|_| true) {
            OrderedEventPoll::Event {
                source,
                event: GpuEvent::MemoryWait(actual),
            } => {
                assert_eq!(actual, wait);
                source
            }
            other => panic!("expected graphics wait, got {other:?}"),
        };
        ordered.block_event(
            source,
            GpuEvent::MemoryWait(wait),
            QueueBlockReason::GuestMemoryWait,
        );

        let source = match ordered.try_recv(|_| true) {
            OrderedEventPoll::Event {
                source,
                event: GpuEvent::MemoryRelease(release),
            } => {
                assert_eq!(release.address, wait.address);
                source
            }
            other => panic!("expected compute release, got {other:?}"),
        };
        ordered.finish_event(source);
        ordered.retry_guest_memory_waits();

        let source = match ordered.try_recv(|_| true) {
            OrderedEventPoll::Event {
                source,
                event: GpuEvent::SubmissionBoundary(2),
            } => source,
            other => panic!("expected compute boundary, got {other:?}"),
        };
        ordered.finish_event(source);
        let source = match ordered.try_recv(|_| true) {
            OrderedEventPoll::Event {
                source,
                event: GpuEvent::MemoryWait(actual),
            } => {
                assert_eq!(actual, wait);
                source
            }
            other => panic!("expected retried graphics wait, got {other:?}"),
        };
        ordered.finish_event(source);

        assert!(matches!(
            ordered.try_recv(|_| true),
            OrderedEventPoll::Event {
                event: GpuEvent::MemoryWrite(GpuMemoryWrite {
                    address: 0x5000,
                    ..
                }),
                ..
            }
        ));
    }

    #[test]
    fn guest_wait_latches_a_real_producer_before_guest_memory_can_be_recycled() {
        let mut tracker = GuestMemoryWaitTracker::default();
        let wait = GpuMemoryWait {
            address: 0x7000_1000,
            reference: 1,
            mask: u64::from(u32::MAX),
            compare_function: 3,
            is_64_bit: false,
        };
        let key = GuestMemoryWaitKey::new(OrderedEventSource::GuestQueue(32), wait);
        assert_eq!(
            tracker.observe_unsatisfied(key),
            GuestMemoryWaitRecovery::None
        );

        tracker.record_produced(wait.address, false, 1);
        assert_eq!(
            tracker.observe_unsatisfied(key),
            GuestMemoryWaitRecovery::ProducerLatched(1)
        );
        assert!(!tracker.active.contains_key(&key));
    }

    #[test]
    fn guest_wait_never_reuses_a_producer_from_before_its_registration() {
        let wait = GpuMemoryWait {
            address: 0x7000_2000,
            reference: 1,
            mask: u64::from(u32::MAX),
            compare_function: 3,
            is_64_bit: false,
        };
        let key = GuestMemoryWaitKey::new(OrderedEventSource::GuestQueue(0), wait);
        let mut tracker = GuestMemoryWaitTracker::default();
        tracker.record_produced(wait.address, false, 1);

        assert_eq!(
            tracker.observe_unsatisfied(key),
            GuestMemoryWaitRecovery::None
        );
        // Guest allocators recycle label addresses. Wall-clock age cannot turn a producer from an
        // earlier allocation generation into satisfaction for this wait.
        assert_eq!(
            tracker.observe_unsatisfied(key),
            GuestMemoryWaitRecovery::None
        );

        tracker.record_produced(wait.address, false, 1);
        assert_eq!(
            tracker.observe_unsatisfied(key),
            GuestMemoryWaitRecovery::ProducerLatched(1)
        );
    }

    #[test]
    fn guest_wait_does_not_latch_missing_or_nonmatching_producers() {
        let wait = GpuMemoryWait {
            address: 0x7000_2000,
            reference: 1,
            mask: u64::from(u32::MAX),
            compare_function: 3,
            is_64_bit: false,
        };
        let key = GuestMemoryWaitKey::new(OrderedEventSource::GuestQueue(0), wait);
        let mut missing = GuestMemoryWaitTracker::default();
        assert_eq!(
            missing.observe_unsatisfied(key),
            GuestMemoryWaitRecovery::None
        );
        assert_eq!(
            missing.observe_unsatisfied(key),
            GuestMemoryWaitRecovery::None
        );

        let mut wrong = GuestMemoryWaitTracker::default();
        assert_eq!(
            wrong.observe_unsatisfied(key),
            GuestMemoryWaitRecovery::None
        );
        wrong.record_produced(wait.address, false, 2);
        assert_eq!(
            wrong.observe_unsatisfied(key),
            GuestMemoryWaitRecovery::None
        );
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
    fn capture_encodes_linear_storage_bytes_with_the_standard_srgb_transfer() {
        assert_eq!(linear_unorm8_to_srgb_unorm8(0), 0);
        assert_eq!(linear_unorm8_to_srgb_unorm8(1), 13);
        assert_eq!(linear_unorm8_to_srgb_unorm8(5), 38);
        assert_eq!(linear_unorm8_to_srgb_unorm8(10), 56);
        assert_eq!(linear_unorm8_to_srgb_unorm8(23), 85);
        assert_eq!(linear_unorm8_to_srgb_unorm8(128), 188);
        assert_eq!(linear_unorm8_to_srgb_unorm8(255), 255);
    }

    #[test]
    fn capture_maps_half_float_scanout_channels_to_unorm8() {
        assert_eq!(half_float_to_unorm8(0x0000), 0);
        assert_eq!(half_float_to_unorm8(0x0001), 0);
        assert_eq!(half_float_to_unorm8(0x3800), 128);
        assert_eq!(half_float_to_unorm8(0x3c00), 255);
        assert_eq!(half_float_to_unorm8(0x4000), 255);
        assert_eq!(half_float_to_unorm8(0xbc00), 0);
        assert_eq!(half_float_to_unorm8(0x7c00), 255);
        assert_eq!(half_float_to_unorm8(0x7e00), 0);
        assert_eq!(half_float_to_srgb_unorm8(0x3800), 188);
    }

    #[test]
    fn capture_decodes_packed_rg11b10_float_channels() {
        assert_eq!(unsigned_float_to_f32(0, 6), 0.0);
        assert_eq!(unsigned_float_to_f32(1, 6), 2.0_f32.powi(-20));
        assert_eq!(unsigned_float_to_f32(15 << 6, 6), 1.0);
        assert_eq!(unsigned_float_to_f32(14 << 6, 6), 0.5);
        assert_eq!(unsigned_float_to_f32(13 << 5, 5), 0.25);
        assert!(unsigned_float_to_f32((31 << 6) | 1, 6).is_nan());

        let packed = (15_u32 << 6) | ((14_u32 << 6) << 11) | ((13_u32 << 5) << 22);
        let descriptor = GuestTextureDescriptor {
            address: 0x1_0000,
            width: 1,
            height: 1,
            pitch: 1,
            tile_mode: 0,
            pixel_format: MTLPixelFormat::RG11B10Float,
            bytes_per_pixel: 4,
        };
        assert_eq!(
            native_texture_capture_to_bgra(&packed.to_le_bytes(), descriptor),
            Some(vec![137, 188, 255, 255]),
        );
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
        assert_eq!(
            read_draw_indices(&draw_16).as_deref(),
            Some([0, 1, 2, 0, 2, 3].as_slice())
        );

        let indices_32 = [0_u32, 1, 2, 0, 2, 3];
        let draw_32 = indexed_draw(indices_32.as_ptr().addr() as u64, 1, 6);
        assert_eq!(
            read_draw_indices(&draw_32).as_deref(),
            Some([0, 1, 2, 0, 2, 3].as_slice())
        );
    }

    #[test]
    fn prefers_immutable_index_snapshot_over_recycled_guest_memory() {
        let mut transient_indices = [0_u16, 1, 2, 0, 2, 3];
        let mut draw = indexed_draw(transient_indices.as_ptr().addr() as u64, 0, 6);
        draw.index_snapshot = Some(std::sync::Arc::from([0_u32, 1, 2, 0, 2, 3]));
        transient_indices.fill(0xffff);

        assert_eq!(
            read_draw_indices(&draw).as_deref(),
            Some([0, 1, 2, 0, 2, 3].as_slice())
        );
    }

    #[test]
    fn aligns_implicit_linear_rgba_texture_pitch() {
        assert_eq!(effective_linear_pitch(320, 4), Some(320));
        assert_eq!(effective_linear_pitch(980, 4), Some(1024));
        assert_eq!(effective_linear_pitch(250, 4), Some(256));
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
    fn pending_flip_retries_after_its_translated_target_is_written() {
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
            sequence: 1,
        });
        presenter.next_flip_sequence = 1;
        presenter.redraw_pending = false;

        presenter.mark_direct_scanout_initialized(descriptor.address);
        let target = DecodedRenderTarget {
            guest_slot: 0,
            texture: descriptor,
            output_kind: PixelOutputKind::Float,
            write_mask: 0xf,
            direct_scanout: true,
            sample_count: 1,
            texture_type: 9,
            layer_count: 1,
            base_layer: 0,
            mip_level: 0,
            mip_level_count: 1,
            dcc_enabled: false,
            fast_clear_enabled: false,
        };
        presenter.refresh_pending_flip_after_target_write(target);

        assert!(presenter.redraw_pending);
        assert_eq!(presenter.uploaded_flip_sequence, 1);
        presenter.upload_pending_frame();
        assert_eq!(presenter.uploaded_flip_sequence, 1);
        assert_eq!(presenter.frame_texture_size, (640, 360));
        assert!(presenter.frame_texture.is_some());

        // The same guest flip remains live while later compositor passes update its target.
        // An uploaded sequence must not suppress the redraw for those newer contents.
        presenter.redraw_pending = false;
        presenter.refresh_pending_flip_after_target_write(target);
        assert!(presenter.redraw_pending);
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
                .activate_translated_scanout(
                    descriptor.address,
                    descriptor.width,
                    descriptor.height,
                )
                .expect("activate direct scanout")
        );
        assert_eq!(presenter.frame_texture_size, (640, 360));
        let output_texture = presenter.frame_texture.as_ref().expect("presented texture");
        assert_eq!(output_texture.width(), texture.width());
        assert_eq!(output_texture.height(), texture.height());
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
        encoder.set_render_pipeline_state(&presenter.presentation.pipeline);
        encoder.set_viewport(presentation_viewport(640, 360, 320, 180));
        encoder.set_fragment_texture(0, Some(output_texture));
        encoder.set_fragment_sampler_state(0, Some(&presenter.presentation.sampler));
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

    #[test]
    fn pre_pixel_shader_diagnostic_can_capture_an_exact_resident_mode27_target() {
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
            .expect("guest render target mapping");
        let descriptor = GuestTextureDescriptor {
            address: address.get(),
            width: 64,
            height: 64,
            pitch: 64,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        assert!(guest_texture_byte_count(descriptor).is_some_and(|bytes| bytes <= allocation_size));

        let (_sender, receiver) = std::sync::mpsc::channel();
        let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
        let device = presenter.device.as_ref().expect("Metal device").clone();
        let resident = presenter
            .ensure_resident_texture(&device, descriptor)
            .expect("mode-27 target becomes resident");
        let target = DecodedRenderTarget {
            guest_slot: 0,
            texture: descriptor,
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

        assert!(presenter.direct_scanout_textures.is_empty());
        let captured = presenter
            .diagnostic_capture_texture(&target)
            .expect("exact resident target is diagnostic-capturable");
        assert!(std::ptr::eq(captured.as_ref(), resident.as_ref()));
    }

    #[test]
    fn mode27_texture_uses_physically_contiguous_disjoint_cpu_aliases() {
        use remu_memory::{
            GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
            GuestProtection,
        };

        let mut memory = GuestAddressSpace::new(24 * GUEST_PAGE_SIZE).expect("guest address space");
        let backing =
            GuestPhysicalBacking::new(8 * GUEST_PAGE_SIZE).expect("guest physical backing");
        let base = memory.base();
        let first = memory
            .map_shared(
                GuestMapPlacement::Fixed(base),
                4 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .expect("first physical half");
        let second_address = base
            .checked_add(16 * GUEST_PAGE_SIZE)
            .expect("second alias address");
        let second = memory
            .map_shared(
                GuestMapPlacement::Fixed(second_address),
                4 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                4 * GUEST_PAGE_SIZE,
            )
            .expect("second physical half");
        memory.write(first, &[0x31; 32]).expect("first bytes");
        memory.write(second, &[0x72; 32]).expect("second bytes");
        let descriptor = GuestTextureDescriptor {
            address: first.get(),
            width: 256,
            height: 128,
            pitch: 256,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        assert_eq!(
            guest_texture_byte_count(descriptor),
            Some(8 * GUEST_PAGE_SIZE)
        );
        assert!(
            guest_shared_mapping_identity(descriptor.address, 8 * GUEST_PAGE_SIZE).is_none(),
            "the CPU VA has an intentional hole"
        );

        let watch = watch_guest_texture(descriptor)
            .expect("GPU-physical texture has complete backing aliases");
        assert_eq!(watch.ranges.len(), 2);
        let raw = snapshot_watched_guest_texture_raw(descriptor, &watch)
            .expect("disjoint aliases snapshot through physical backing");
        assert_eq!(&raw[..32], &[0x31; 32]);
        assert_eq!(
            &raw[usize::try_from(4 * GUEST_PAGE_SIZE).unwrap()
                ..usize::try_from(4 * GUEST_PAGE_SIZE).unwrap() + 32],
            &[0x72; 32]
        );
    }

    #[test]
    fn texture_write_watch_observes_every_live_physical_alias_and_remap() {
        use remu_memory::{
            GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
            GuestProtection,
        };

        let mut memory = GuestAddressSpace::new(16 * GUEST_PAGE_SIZE).expect("guest address space");
        let backing =
            GuestPhysicalBacking::new(4 * GUEST_PAGE_SIZE).expect("guest physical backing");
        let base = memory.base();
        let primary = memory
            .map_shared(
                GuestMapPlacement::Fixed(base),
                4 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .expect("primary alias");
        let alias_address = base
            .checked_add(8 * GUEST_PAGE_SIZE)
            .expect("alias address");
        let alias = memory
            .map_shared(
                GuestMapPlacement::Fixed(alias_address),
                4 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .expect("second alias");
        let descriptor = GuestTextureDescriptor {
            address: primary.get(),
            width: 64,
            height: 64,
            pitch: 64,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        let watch = watch_guest_texture(descriptor).expect("watch every backing alias");
        assert_eq!(watch.ranges.len(), 2);

        memory.write(alias, &[0x5a]).expect("write through alias");
        assert!(
            watch.is_dirty(),
            "a coherent alias write must invalidate the GPU texture"
        );
        memory
            .unmap(alias, 4 * GUEST_PAGE_SIZE)
            .expect("replace alias");
        assert_eq!(
            watch.retirement(),
            Some(CpuWriteWatchRetirement::Replaced),
            "mapping replacement retires the resource watch before direct access"
        );
        assert!(
            watch
                .with_contiguous_mapping(primary.get(), 4 * GUEST_PAGE_SIZE, || ())
                .is_err(),
            "a retired multi-alias watch cannot authorize a later VA dereference"
        );
    }

    #[test]
    fn contiguous_texture_snapshot_and_writeback_retain_direct_mapping_semantics() {
        use remu_memory::{
            GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
            GuestProtection,
        };

        let mut memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
        let backing =
            GuestPhysicalBacking::new(4 * GUEST_PAGE_SIZE).expect("guest physical backing");
        let address = memory
            .map_shared(
                GuestMapPlacement::Anywhere,
                4 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .expect("contiguous texture mapping");
        let descriptor = GuestTextureDescriptor {
            address: address.get(),
            width: 64,
            height: 64,
            pitch: 64,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        let watch = watch_guest_texture(descriptor).expect("contiguous texture watch");
        assert_eq!(watch.ranges.len(), 1);
        let raw = snapshot_watched_guest_texture_raw(descriptor, &watch)
            .expect("guarded direct snapshot");
        assert_eq!(raw.len(), usize::try_from(4 * GUEST_PAGE_SIZE).unwrap());

        let linear = vec![0x3c; 64 * 64 * 4];
        write_watched_guest_texture_linear(descriptor, &linear, &watch)
            .expect("guarded direct writeback");
        let decoded =
            snapshot_watched_guest_texture_raw(descriptor, &watch).expect("updated snapshot");
        assert!(decoded.iter().any(|&byte| byte == 0x3c));
    }

    #[test]
    fn contiguous_texture_access_pins_every_adjacent_mapping_record() {
        use remu_memory::{
            GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapCoalescing, GuestMapPlacement,
            GuestPhysicalBacking, GuestProtection, GuestSharedMapOptions,
        };

        let mut memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
        let backing =
            GuestPhysicalBacking::new(4 * GUEST_PAGE_SIZE).expect("guest physical backing");
        let base = memory.base();
        let first = memory
            .map_shared_with_coalescing(
                GuestMapPlacement::Fixed(base),
                2 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                GuestSharedMapOptions {
                    backing_offset: 0,
                    coalescing: GuestMapCoalescing::Disallowed,
                },
            )
            .expect("first direct-memory mapping");
        memory
            .map_shared_with_coalescing(
                GuestMapPlacement::Fixed(
                    base.checked_add(2 * GUEST_PAGE_SIZE)
                        .expect("second mapping address"),
                ),
                2 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                GuestSharedMapOptions {
                    backing_offset: 2 * GUEST_PAGE_SIZE,
                    coalescing: GuestMapCoalescing::Disallowed,
                },
            )
            .expect("second direct-memory mapping");
        let descriptor = GuestTextureDescriptor {
            address: first.get(),
            width: 64,
            height: 64,
            pitch: 64,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        assert!(guest_shared_mapping_identity(descriptor.address, 4 * GUEST_PAGE_SIZE).is_some());
        let watch = watch_guest_texture(descriptor).expect("adjacent mapping watches");
        assert_eq!(watch.ranges.len(), 2);
        assert!(!watch.has_single_covering_mapping(descriptor.address, 4 * GUEST_PAGE_SIZE));
        let raw = snapshot_watched_guest_texture_raw(descriptor, &watch)
            .expect("snapshot uses the coherent physical backing slow path");
        assert_eq!(raw.len(), usize::try_from(4 * GUEST_PAGE_SIZE).unwrap());
    }

    #[test]
    fn silent_hill_mode27_mip_tail_round_trips_every_level() {
        let descriptor = GuestTextureDescriptor {
            address: 0x80b9_2700_00,
            width: 32,
            height: 32,
            pitch: 32,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RG16Uint,
            bytes_per_pixel: 4,
        };
        let layout = mode27_mip_chain_layout(descriptor, 6).expect("complete mip-tail layout");
        assert_eq!(layout.byte_count, 65_536);
        assert!(layout.levels.iter().all(|level| level.in_tail));
        assert_eq!(
            layout
                .levels
                .iter()
                .map(|level| (level.width, level.height, level.tail_x, level.tail_y))
                .collect::<Vec<_>>(),
            vec![
                (32, 32, 64, 0),
                (16, 16, 0, 64),
                (8, 8, 32, 0),
                (4, 4, 0, 32),
                (2, 2, 16, 0),
                (1, 1, 8, 16),
            ]
        );

        let levels = layout
            .levels
            .iter()
            .enumerate()
            .map(|(index, level)| {
                let byte_count = level.width * level.height * layout.bytes_per_element;
                (0..byte_count)
                    .map(|byte| {
                        u8::try_from((index * 37 + byte) & 0xff).expect("masked test byte fits")
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut raw = vec![0xa5; layout.byte_count];
        encode_mode27_mip_chain_into_raw(&layout, &levels, &mut raw)
            .expect("encode exact mip tail");
        let (decoded_layout, decoded) =
            decode_mode27_mip_chain_from_raw(descriptor, 6, &raw).expect("decode exact mip tail");
        assert_eq!(decoded_layout, layout);
        assert_eq!(decoded, levels);
    }

    #[test]
    fn silent_hill_render_target_accepts_absolute_max_mip_hierarchy() {
        let address = 0x80b9_2700_00;
        let mut registers = vec![GpuRegister {
            offset: CB_TARGET_MASK,
            value: 0xf,
        }];
        append_color_target_registers(&mut registers, 0, address, 32, 32, 5, 4, 27);
        registers
            .iter_mut()
            .find(|register| register.offset == CB_COLOR0_ATTRIB2)
            .expect("color attrib2 exists")
            .value |= 5 << 28;
        registers.sort_by_key(|register| register.offset);
        let targets = decode_render_targets_from_registers(&registers, 0xf)
            .expect("observed mip=0/max_mip=5 target is complete");
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].mip_level, 0);
        assert_eq!(targets[0].mip_level_count, 6);
        assert_eq!(targets[0].texture.pixel_format, MTLPixelFormat::RG16Uint);
    }

    #[test]
    fn render_target_decode_rejects_incomplete_or_lossy_state() {
        assert!(
            decode_render_target_format(7, 7, 0)
                .expect_err("11_11_10 float needs explicit repacking and R/B remapping")
                .contains("format=7")
        );

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
        let dcc_targets = decode_render_targets_from_registers(&registers, 0xf)
            .expect("DCC targets use uncompressed resident Metal storage");
        assert!(dcc_targets[0].dcc_enabled);

        registers
            .iter_mut()
            .find(|register| register.offset == CB_COLOR0_INFO)
            .expect("color info exists")
            .value &= !(1 << 28);
        registers
            .iter_mut()
            .find(|register| register.offset == CB_COLOR0_INFO)
            .expect("color info exists")
            .value |= 1 << 13;
        let fast_clear_targets = decode_render_targets_from_registers(&registers, 0xf)
            .expect("fast-clear targets use cleared uncompressed resident Metal storage");
        assert!(fast_clear_targets[0].fast_clear_enabled);

        registers
            .iter_mut()
            .find(|register| register.offset == CB_COLOR0_INFO)
            .expect("color info exists")
            .value &= !(1 << 13);
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
    fn decodes_silent_hill_volume_render_target() {
        let address = 0x0080_248f_0000_u64;
        let mut registers = vec![GpuRegister {
            offset: CB_TARGET_MASK,
            value: 0xf,
        }];
        append_color_target_registers(&mut registers, 0, address, 64, 64, 12, 7, 27);
        registers
            .iter_mut()
            .find(|register| register.offset == CB_COLOR0_VIEW)
            .expect("color view exists")
            .value = 64 << 13;
        registers
            .iter_mut()
            .find(|register| register.offset == CB_COLOR0_ATTRIB3)
            .expect("color attrib3 exists")
            .value = 63 | (27 << 14) | (2 << 24);
        registers.sort_by_key(|register| register.offset);

        let target = decode_render_targets_from_registers(&registers, 0xf)
            .expect("Silent Hill volume target is representable")
            .remove(0);
        assert_eq!(target.texture.address, address);
        assert_eq!((target.texture.width, target.texture.height), (64, 64));
        assert_eq!(target.texture_type, 10);
        assert_eq!(target.layer_count, 64);
        assert_eq!(target.base_layer, 0);
    }

    #[test]
    fn maps_guest_number_types_to_matching_metal_export_types() {
        let float = decode_render_target_format(10, 0, 0).expect("RGBA8 unorm");
        let uint = decode_render_target_format(10, 4, 0).expect("RGBA8 uint");
        let sint = decode_render_target_format(12, 5, 0).expect("RGBA16 sint");
        let standard_10_10_10_2 =
            decode_render_target_format(9, 0, 0).expect("standard packed color");
        let bgra = decode_render_target_format(10, 0, 1).expect("BGRA component swap");
        let rgba16_alt =
            decode_render_target_format(12, 7, 1).expect("RGBA16F alternate component swap");
        assert_eq!(float.output_kind, PixelOutputKind::Float);
        assert_eq!(uint.output_kind, PixelOutputKind::Uint);
        assert_eq!(sint.output_kind, PixelOutputKind::Sint);
        assert_eq!(
            standard_10_10_10_2.pixel_format,
            MTLPixelFormat::RGB10A2Unorm
        );
        assert_eq!(bgra.pixel_format, MTLPixelFormat::BGRA8Unorm);
        assert_eq!(rgba16_alt.pixel_format, MTLPixelFormat::RGBA16Float);
        assert!(decode_render_target_format(8, 0, 0).is_err());
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
    fn translates_separate_premultiplied_guest_blending() {
        let blend = decode_metal_blend_attachment(0x6501_0501).expect("supported blend state");
        assert!(blend.enabled);
        assert_eq!(blend.source_rgb, MTLBlendFactor::One);
        assert_eq!(blend.destination_rgb, MTLBlendFactor::OneMinusSourceAlpha);
        assert_eq!(blend.rgb_operation, MTLBlendOperation::Add);
        assert_eq!(blend.source_alpha, MTLBlendFactor::One);
        assert_eq!(blend.destination_alpha, MTLBlendFactor::OneMinusSourceAlpha);
        assert_eq!(blend.alpha_operation, MTLBlendOperation::Add);
    }

    #[test]
    fn shares_color_blending_with_alpha_without_separate_alpha() {
        let control = (1 << 30) | 4 | (5 << 8);
        let blend = decode_metal_blend_attachment(control).expect("supported blend state");
        assert_eq!(blend.source_rgb, MTLBlendFactor::SourceAlpha);
        assert_eq!(blend.destination_rgb, MTLBlendFactor::OneMinusSourceAlpha);
        assert_eq!(blend.source_alpha, blend.source_rgb);
        assert_eq!(blend.destination_alpha, blend.destination_rgb);
        assert_eq!(blend.alpha_operation, blend.rgb_operation);
    }

    #[test]
    fn rejects_enabled_blend_encodings_that_cannot_be_lowered() {
        assert!(decode_metal_blend_attachment((1 << 30) | 11).is_err());
        assert!(decode_metal_blend_attachment((1 << 30) | (7 << 5)).is_err());
        assert!(
            !decode_metal_blend_attachment(0x1f1f_1f1f)
                .expect("disabled state ignores inactive factor fields")
                .enabled
        );
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
        assert_eq!(metal_nonindexed_vertex_count(7, 3), 4);
        assert_eq!(metal_nonindexed_vertex_count(0x11, 3), 4);
        assert_eq!(metal_nonindexed_vertex_count(4, 3), 3);
    }

    #[test]
    fn maps_legacy_and_gen5_triangle_list_codes_to_metal_triangles() {
        assert_eq!(metal_primitive_type(4), Ok(MTLPrimitiveType::Triangle));
        assert_eq!(metal_primitive_type(7), Ok(MTLPrimitiveType::TriangleStrip));
        assert_eq!(
            metal_primitive_topology_class(7),
            Ok(MTLPrimitiveTopologyClass::Triangle)
        );
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
        words[3] = 0x204 | (27 << 20) | (9 << 28);
        words[4] = width - 1;

        let descriptor = decode_image_descriptor(&words).expect("valid RGBA8 2D descriptor");
        assert_eq!(descriptor.guest.address, address);
        assert_eq!(descriptor.guest.width, width);
        assert_eq!(descriptor.guest.height, height);
        assert_eq!(descriptor.guest.pitch, width);
        assert_eq!(descriptor.guest.tile_mode, 27);
        assert_eq!(descriptor.destination_select, 0x204);
        assert!(!image_destination_select_is_native(
            descriptor.guest.pixel_format,
            descriptor.destination_select,
        ));
        assert!(image_destination_select_is_native(
            descriptor.guest.pixel_format,
            0xfac,
        ));
    }

    #[test]
    fn decodes_silent_hill_rg11b10_tone_map_source_descriptor() {
        let descriptor =
            decode_image_descriptor(&[0x804b_9100, 0xc240_0000, 0x021b_c3bf, 0x91b0_03ac])
                .expect("observed Silent Hill HDR source descriptor decodes");
        assert_eq!(descriptor.guest.address, 0x804b_9100_00);
        assert_eq!(
            (descriptor.guest.width, descriptor.guest.height),
            (3_840, 2_160)
        );
        assert_eq!(descriptor.guest.pitch, 3_840);
        assert_eq!(descriptor.guest.tile_mode, 27);
        assert_eq!(descriptor.guest.pixel_format, MTLPixelFormat::RG11B10Float);
        assert_eq!(descriptor.guest.bytes_per_pixel, 4);
        assert_eq!(descriptor.destination_select, 0x3ac);
        assert_eq!(descriptor.base_mip_level, 0);
        assert_eq!(descriptor.mip_level_count, 1);
        assert_eq!(descriptor.texture_type, 9);
        assert_eq!(descriptor.layer_count, 1);
    }

    #[test]
    fn invalid_sampled_images_receive_a_dimension_compatible_fallback() {
        let binding = ImageBinding {
            pc: 0x50,
            opcode: Opcode::Known("ImageSample"),
            control: ImageControl {
                dmask: 0xf,
                vector_address: 0,
                address_registers: vec![0, 1],
                vector_data: 4,
                scalar_resource: 12,
                scalar_sampler: 20,
                dimension: 1,
                is_array: false,
                glc: false,
                slc: false,
                a16: false,
                d16: false,
            },
            resource_descriptor: vec![0; 8].into(),
            sampler_descriptor: vec![0; 4].into(),
            mip_level: None,
        };

        assert!(image_resource_descriptor_is_null(
            &binding.resource_descriptor
        ));
        assert!(!image_resource_descriptor_is_null(&[]));
        assert!(decode_image_descriptor(&binding.resource_descriptor).is_err());
        let fallback = fallback_image_descriptor(&binding).expect("fallback descriptor");
        assert_eq!(fallback.guest.address, 0);
        assert_eq!((fallback.guest.width, fallback.guest.height), (1, 1));
        assert_eq!(fallback.guest.pixel_format, MTLPixelFormat::RGBA8Unorm);
        assert_eq!(fallback.texture_type, 9);
        assert_eq!(fallback.layer_count, 1);
    }

    #[test]
    fn preserves_explicit_linear_video_plane_pitch() {
        let address = 0x0070_2180_0000_u64;
        let width = 1_440_u32;
        let height = 1_080_u32;
        let encoded_width = width - 1;
        let mut words = [0_u32; 8];
        words[0] = u32::try_from(address >> 8).expect("low image address fits");
        words[1] = u32::try_from(address >> 40).expect("high image address fits")
            | (1 << 20)
            | ((encoded_width & 0x3) << 30);
        words[2] = (encoded_width >> 2) | ((height - 1) << 14);
        words[3] = 0xfac | (9 << 28);
        words[4] = width - 1;

        let descriptor = decode_image_descriptor(&words).expect("valid linear R8 video plane");

        assert_eq!(descriptor.guest.pixel_format, MTLPixelFormat::R8Unorm);
        assert_eq!(descriptor.guest.bytes_per_pixel, 1);
        assert_eq!(descriptor.guest.pitch, width);
        assert_eq!(
            guest_texture_byte_count(descriptor.guest),
            Some(u64::from(width) * u64::from(height))
        );
    }

    #[test]
    fn decodes_observed_r16_float_image_descriptor() {
        let address = 0x0070_0df8_0000_u64;
        let width = 320_u32;
        let height = 180_u32;
        let encoded_width = width - 1;
        let mut words = [0_u32; 8];
        words[0] = u32::try_from(address >> 8).expect("low image address fits");
        words[1] = u32::try_from(address >> 40).expect("high image address fits")
            | (13 << 20)
            | ((encoded_width & 0x3) << 30);
        words[2] = (encoded_width >> 2) | ((height - 1) << 14);
        words[3] = 0xfac | (27 << 20) | (9 << 28);
        words[4] = width - 1;

        let descriptor = decode_image_descriptor(&words).expect("valid R16-float descriptor");

        assert_eq!(descriptor.guest.pixel_format, MTLPixelFormat::R16Float);
        assert_eq!(descriptor.guest.bytes_per_pixel, 2);
        assert_eq!(descriptor.destination_select, 0xfac);
    }

    #[test]
    fn decodes_observed_bc3_and_bc7_sampled_textures_as_four_by_four_blocks() {
        for (unified_format, tile_mode, expected_format) in [
            (174, 27, MTLPixelFormat::BC3_RGBA_sRGB),
            (182, 5, MTLPixelFormat::BC7_RGBAUnorm_sRGB),
        ] {
            let address = 0x0070_0df8_0000_u64;
            let width = 65_u32;
            let height = 65_u32;
            let encoded_width = width - 1;
            let mut words = [0_u32; 8];
            words[0] = u32::try_from(address >> 8).expect("low image address fits");
            words[1] = u32::try_from(address >> 40).expect("high image address fits")
                | (unified_format << 20)
                | ((encoded_width & 0x3) << 30);
            words[2] = (encoded_width >> 2) | ((height - 1) << 14);
            words[3] = 0xfac | (tile_mode << 20) | (9 << 28);
            words[4] = width - 1;

            let descriptor = decode_image_descriptor(&words).expect("valid BC descriptor");
            assert_eq!(descriptor.guest.pixel_format, expected_format);
            assert_eq!(descriptor.guest.bytes_per_pixel, 16);
            assert_eq!(descriptor.guest.tile_mode, tile_mode);
            let expected_bytes = if tile_mode == 5 { 16_384 } else { 65_536 };
            assert_eq!(
                guest_texture_byte_count(descriptor.guest),
                Some(expected_bytes)
            );
        }
    }

    #[test]
    fn maps_unified_rgba8_srgb_sampled_texture_number_type() {
        assert_eq!(
            super::decode_sampled_texture_format(10, 9),
            Ok((MTLPixelFormat::RGBA8Unorm_sRGB, 4))
        );
    }

    #[test]
    fn accepts_a_complete_base_zero_sampled_mip_range() {
        let width = 256_u32;
        let height = 128_u32;
        let encoded_width = width - 1;
        let mut words = [0_u32; 8];
        words[0] = 0x100;
        words[1] = (130 << 20) | ((encoded_width & 0x3) << 30);
        words[2] = (encoded_width >> 2) | ((height - 1) << 14);
        words[3] = 0xfac | (4 << 16) | (24 << 20) | (9 << 28);
        words[4] = width - 1;

        let descriptor = decode_image_descriptor(&words).expect("valid sampled mip hierarchy");
        assert_eq!(descriptor.base_mip_level, 0);
        assert_eq!(descriptor.mip_level_count, 5);
        assert_eq!(descriptor.guest.width, width);
        assert_eq!(descriptor.guest.height, height);
        assert_eq!(
            super::sampled_texture_raw_range(descriptor),
            Ok((
                usize::try_from(descriptor.guest.address).expect("test address fits usize"),
                usize::try_from(
                    guest_texture_byte_count(descriptor.guest)
                        .expect("base texture byte count fits")
                )
                .expect("base texture byte count fits usize")
            ))
        );
    }

    #[test]
    fn accepts_a_nonzero_sampled_mip_view() {
        let width = 2_048_u32;
        let height = 2_048_u32;
        let encoded_width = width - 1;
        let mut words = [0_u32; 8];
        words[0] = 0x100;
        words[1] = (130 << 20) | ((encoded_width & 0x3) << 30);
        words[2] = (encoded_width >> 2) | ((height - 1) << 14);
        words[3] = 0xfac | (1 << 12) | (4 << 16) | (27 << 20) | (9 << 28);
        words[4] = width - 1;

        let descriptor = decode_image_descriptor(&words).expect("valid nonzero mip view");
        assert_eq!(descriptor.base_mip_level, 1);
        assert_eq!(descriptor.mip_level_count, 4);
        assert_eq!(descriptor.guest.width, width);
        assert_eq!(descriptor.guest.height, height);
        assert_eq!(
            super::sampled_texture_raw_range(descriptor)
                .expect("mode-27 nonzero mip view covers its full underlying chain")
                .1,
            mode27_mip_chain_layout(descriptor.guest, 5)
                .expect("underlying mip layout")
                .byte_count
        );
    }

    #[test]
    fn reuses_a_larger_resident_mip_hierarchy_for_a_smaller_view() {
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
            .expect("guest mip-tail mapping");
        let guest = GuestTextureDescriptor {
            address: address.get(),
            width: 32,
            height: 32,
            pitch: 32,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RG16Uint,
            bytes_per_pixel: 4,
        };
        assert_eq!(
            mode27_mip_chain_layout(guest, 6)
                .expect("six-level mode-27 tail")
                .byte_count,
            usize::try_from(allocation_size).expect("allocation size fits usize")
        );

        let (_sender, receiver) = std::sync::mpsc::channel();
        let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter state");
        let device = presenter.device.as_ref().expect("Metal device").clone();
        let complete = super::DecodedImageDescriptor {
            guest,
            destination_select: 0x22c,
            base_mip_level: 0,
            mip_level_count: 6,
            texture_type: 9,
            layer_count: 1,
        };
        let original = presenter
            .ensure_resident_texture(&device, guest)
            .expect("first single-level use reserves the complete mapped hierarchy");
        assert_eq!(original.mipmap_level_count(), 6);

        let complete_view = super::ensure_mipmapped_resident_texture_in_map(
            &device,
            complete,
            &mut presenter.resident_guest_textures,
        )
        .expect("later complete mip view reuses first residency");
        assert!(std::ptr::eq(original.as_ref(), complete_view.as_ref()));
        assert_eq!(
            presenter
                .resident_guest_textures
                .get(&guest)
                .expect("resident survives preflight")
                .mip_level_count,
            6
        );

        let sampled = super::DecodedImageDescriptor {
            guest,
            destination_select: 0x22c,
            base_mip_level: 2,
            mip_level_count: 2,
            texture_type: 9,
            layer_count: 1,
        };
        let view = super::ensure_mipmapped_resident_texture_in_map(
            &device,
            sampled,
            &mut presenter.resident_guest_textures,
        )
        .expect("larger resident hierarchy covers the sampled mip range");
        assert_eq!(view.width(), 8);
        assert_eq!(view.height(), 8);
        assert_eq!(view.mipmap_level_count(), 2);
        assert!(view.parent_texture().is_some());
        let resident = presenter
            .resident_guest_textures
            .get(&guest)
            .expect("resident survives narrower view");
        assert_eq!(resident.mip_level_count, 6);
        assert!(std::ptr::eq(original.as_ref(), resident.texture.as_ref()));
    }

    #[test]
    fn creates_exact_initial_mode27_base_three_and_seven_views() {
        let Some(device) = Device::system_default() else {
            return;
        };
        for base_mip_level in [3, 7] {
            let descriptor = super::DecodedImageDescriptor {
                guest: GuestTextureDescriptor {
                    address: 0x80c0_0000_00,
                    width: 1_024,
                    height: 1_024,
                    pitch: 1_024,
                    tile_mode: 27,
                    pixel_format: MTLPixelFormat::RGBA8Unorm,
                    bytes_per_pixel: 4,
                },
                destination_select: 0xfac,
                base_mip_level,
                mip_level_count: 1,
                texture_type: 9,
                layer_count: 1,
            };
            let underlying_mip_count = base_mip_level + 1;
            let layout = mode27_mip_chain_layout(descriptor.guest, underlying_mip_count)
                .expect("exact underlying mip layout");
            let raw = vec![0; layout.byte_count];
            let texture =
                super::create_exact_mode27_sampled_mip_view(&device, descriptor, &raw, false)
                    .expect("exact Metal mip view");
            assert_eq!(
                texture.width(),
                u64::from((descriptor.guest.width >> base_mip_level).max(1))
            );
            assert_eq!(
                texture.height(),
                u64::from((descriptor.guest.height >> base_mip_level).max(1))
            );
            assert_eq!(texture.mipmap_level_count(), 1);
        }
    }

    #[test]
    fn decodes_a_complete_six_face_cube_image_descriptor() {
        let address = 0x0070_1a00_0000_u64;
        let width = 128_u32;
        let height = 128_u32;
        let encoded_width = width - 1;
        let mut words = [0_u32; 8];
        words[0] = u32::try_from(address >> 8).expect("low image address fits");
        words[1] = u32::try_from(address >> 40).expect("high image address fits")
            | (130 << 20)
            | ((encoded_width & 0x3) << 30);
        words[2] = (encoded_width >> 2) | ((height - 1) << 14);
        words[3] = 0xfac | (27 << 20) | (11 << 28);
        // BASE_ARRAY=0, LAST_ARRAY=5 describes exactly one six-face cube.
        words[4] = 5;

        let descriptor = decode_image_descriptor(&words).expect("valid cube descriptor");
        assert_eq!(descriptor.texture_type, 11);
        assert_eq!(descriptor.layer_count, 6);
        assert_eq!(descriptor.guest.address, address);
        assert_eq!(descriptor.guest.width, width);
        assert_eq!(descriptor.guest.height, height);
        assert_eq!(descriptor.guest.pitch, width);
        assert_eq!(descriptor.guest.tile_mode, 27);
        assert_eq!(
            super::sampled_texture_raw_range(descriptor),
            Ok((
                usize::try_from(address).expect("test address fits usize"),
                6 * 65_536
            ))
        );
    }

    #[test]
    fn decodes_a_tiled_eight_slice_3d_image_descriptor() {
        let address = 0x0070_1b00_0000_u64;
        let width = 64_u32;
        let height = 64_u32;
        let encoded_width = width - 1;
        let mut words = [0_u32; 8];
        words[0] = u32::try_from(address >> 8).expect("low image address fits");
        words[1] = u32::try_from(address >> 40).expect("high image address fits")
            | (130 << 20)
            | ((encoded_width & 0x3) << 30);
        words[2] = (encoded_width >> 2) | ((height - 1) << 14);
        words[3] = 0xfac | (27 << 20) | (10 << 28);
        words[4] = 7;

        let descriptor = decode_image_descriptor(&words).expect("valid 3D descriptor");
        assert_eq!(descriptor.texture_type, 10);
        assert_eq!(descriptor.layer_count, 8);
        assert_eq!(descriptor.guest.width, width);
        assert_eq!(descriptor.guest.height, height);
        assert_eq!(
            super::sampled_texture_raw_range(descriptor),
            Ok((
                usize::try_from(address).expect("test address fits usize"),
                8 * 65_536
            ))
        );
    }

    #[test]
    fn applies_gfx10_constant_and_channel_image_swizzles() {
        let mut rgba = [10_u8, 20, 30, 40];
        let destination_select = 1 | (5 << 6) | (4 << 9);
        apply_image_destination_select(&mut rgba, destination_select).expect("valid swizzle");
        assert_eq!(rgba, [255, 0, 20, 10]);
    }

    #[test]
    fn expands_r8_red_into_alpha_for_non_native_destination_select() {
        let descriptor = GuestTextureDescriptor {
            address: 0x1_0000,
            width: 2,
            height: 1,
            pitch: 2,
            tile_mode: 0,
            pixel_format: MTLPixelFormat::R8Unorm,
            bytes_per_pixel: 1,
        };
        let (upload_descriptor, upload) =
            super::prepare_sampled_texture_upload(descriptor, vec![10, 20], 0x800)
                .expect("R8 red-to-alpha swizzle");
        assert_eq!(upload_descriptor.pixel_format, MTLPixelFormat::RGBA8Unorm);
        assert_eq!(upload_descriptor.bytes_per_pixel, 4);
        assert_eq!(upload, [0, 0, 0, 10, 0, 0, 0, 20]);
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
    fn mode24_uses_the_exact_gfx10_z_order_low_address_bits() {
        assert_eq!(mode24_within_block_byte_offset_for_format(0, 0, 4), 0);
        assert_eq!(mode24_within_block_byte_offset_for_format(1, 0, 4), 4);
        assert_eq!(mode24_within_block_byte_offset_for_format(0, 1, 4), 8);
        assert_eq!(mode24_within_block_byte_offset_for_format(2, 0, 4), 16);
        assert_eq!(mode24_within_block_byte_offset_for_format(0, 2, 4), 32);
        assert_eq!(mode24_within_block_byte_offset_for_format(3, 5, 1), 0x27);
        assert_eq!(mode24_within_block_byte_offset_for_format(3, 5, 2), 0x4e);
        assert_eq!(mode24_within_block_byte_offset_for_format(3, 5, 4), 0x9c);
    }

    #[test]
    fn mode27_uses_the_current_gfx10_render_target_address_equation() {
        assert_eq!(mode27_within_block_byte_offset_for_format(0, 0, 4), 0);
        assert_eq!(mode27_within_block_byte_offset_for_format(1, 0, 4), 4);
        assert_eq!(mode27_within_block_byte_offset_for_format(2, 0, 4), 8);
        assert_eq!(mode27_within_block_byte_offset_for_format(0, 1, 4), 16);
        assert_eq!(mode27_within_block_byte_offset_for_format(0, 2, 4), 32);
        assert_eq!(mode27_within_block_byte_offset_for_format(4, 0, 4), 128);
        assert_eq!(super::mode27_surface_z_xor(0), 0);
        assert_eq!(super::mode27_surface_z_xor(1), 0x800);
        assert_eq!(super::mode27_surface_z_xor(2), 0x400);
        assert_eq!(super::mode27_surface_z_xor(4), 0x200);
        assert_eq!(super::mode27_surface_z_xor(8), 0x100);
        assert_eq!(super::mode27_surface_z_xor(15), 0xf00);
    }

    #[test]
    fn mode27_volume_detile_applies_each_surface_z_xor() {
        let layer_count = 32_u32;
        let descriptor = GuestTextureDescriptor {
            address: 0x1_0000,
            width: 32,
            height: 32,
            pitch: 32,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RGB10A2Unorm,
            bytes_per_pixel: 4,
        };
        let face_bytes =
            usize::try_from(guest_texture_byte_count(descriptor).expect("face size fits"))
                .expect("face size fits usize");
        assert_eq!(face_bytes, 65_536);
        let tight_face_bytes = 32 * 32 * 4;
        let mut expected = Vec::with_capacity(tight_face_bytes * layer_count as usize);
        let mut raw = vec![0_u8; face_bytes * layer_count as usize];
        for z in 0..layer_count as usize {
            let mut linear = vec![0_u8; tight_face_bytes];
            for (index, texel) in linear.chunks_exact_mut(4).enumerate() {
                texel.copy_from_slice(
                    &(u32::try_from(z).expect("z fits") << 24
                        | u32::try_from(index).expect("texel index fits"))
                    .to_ne_bytes(),
                );
            }
            expected.extend_from_slice(&linear);
            let mut tiled = vec![0_u8; face_bytes];
            super::copy_linear_to_tiled(&linear, descriptor, &mut tiled).expect("base slice tiles");
            let xor = super::mode27_surface_z_xor(z);
            let raw_face = &mut raw[z * face_bytes..(z + 1) * face_bytes];
            for source in (0..face_bytes).step_by(4) {
                let destination = source ^ xor;
                raw_face[destination..destination + 4].copy_from_slice(&tiled[source..source + 4]);
            }
        }
        let decoded = super::decode_sampled_texture_linear_from_raw(
            super::DecodedImageDescriptor {
                guest: descriptor,
                destination_select: 0xfac,
                base_mip_level: 0,
                mip_level_count: 1,
                texture_type: 10,
                layer_count,
            },
            &raw,
        )
        .expect("volume detiles");
        assert_eq!(decoded, expected);
    }

    #[test]
    fn mode24_guest_copy_round_trips_partial_and_parity_blocks() {
        let width = 257_usize;
        let height = 257_usize;
        let mut descriptor = GuestTextureDescriptor {
            address: 0x1_0000,
            width: u32::try_from(width).expect("width fits u32"),
            height: u32::try_from(height).expect("height fits u32"),
            pitch: u32::try_from(width).expect("pitch fits u32"),
            tile_mode: 24,
            pixel_format: MTLPixelFormat::R32Float,
            bytes_per_pixel: 4,
        };
        let tiled_length = usize::try_from(
            guest_texture_byte_count(descriptor).expect("tiled byte count fits u64"),
        )
        .expect("tiled byte count fits usize");
        let mut tiled = vec![0_u8; tiled_length];
        descriptor.address =
            u64::try_from(tiled.as_mut_ptr() as usize).expect("test allocation address fits u64");
        let linear = (0..width * height * 4)
            .map(|index| u8::try_from(index & 0xff).expect("masked byte fits u8"))
            .collect::<Vec<_>>();

        write_guest_texture_linear(descriptor, &linear).expect("mode 24 write succeeds");
        assert_eq!(
            snapshot_guest_texture_linear(descriptor).expect("mode 24 snapshot succeeds"),
            linear
        );
    }

    #[test]
    fn mode9_uses_the_exact_standard_64k_address_bits() {
        assert_eq!(mode9_within_block_byte_offset(0, 0, 4), 0);
        assert_eq!(mode9_within_block_byte_offset(1, 0, 4), 4);
        assert_eq!(mode9_within_block_byte_offset(2, 0, 4), 8);
        assert_eq!(mode9_within_block_byte_offset(0, 1, 4), 16);
        assert_eq!(mode9_within_block_byte_offset(0, 2, 4), 32);
        assert_eq!(mode9_within_block_byte_offset(4, 0, 4), 128);
        assert_eq!(mode9_within_block_byte_offset(127, 127, 4), 65_532);
    }

    #[test]
    fn mode9_round_trips_every_supported_element_size() {
        for (bytes_per_element, pixel_format) in [
            (1, MTLPixelFormat::R8Unorm),
            (2, MTLPixelFormat::RG8Unorm),
            (4, MTLPixelFormat::RGBA8Unorm),
            (8, MTLPixelFormat::RGBA16Float),
            (16, MTLPixelFormat::RGBA32Float),
        ] {
            let (block_width, block_height) = tiled_block_dimensions(65_536, bytes_per_element)
                .expect("element size has a mode 9 equation");
            let width = block_width + 1;
            let height = block_height + 1;
            let mut descriptor = GuestTextureDescriptor {
                address: 0x1_0000,
                width: u32::try_from(width).expect("test width fits u32"),
                height: u32::try_from(height).expect("test height fits u32"),
                pitch: u32::try_from(width).expect("test pitch fits u32"),
                tile_mode: 9,
                pixel_format,
                bytes_per_pixel: u32::try_from(bytes_per_element)
                    .expect("test bytes per element fits u32"),
            };
            let tiled_length = usize::try_from(
                guest_texture_byte_count(descriptor).expect("tiled byte count fits u64"),
            )
            .expect("tiled byte count fits usize");
            let mut tiled = vec![0_u8; tiled_length];
            descriptor.address = u64::try_from(tiled.as_mut_ptr() as usize)
                .expect("test allocation address fits u64");
            let linear = (0..width * height * bytes_per_element)
                .map(|index| u8::try_from(index & 0xff).expect("masked byte fits u8"))
                .collect::<Vec<_>>();
            write_guest_texture_linear(descriptor, &linear).expect("mode 9 write succeeds");
            assert_eq!(
                snapshot_guest_texture_linear(descriptor).expect("mode 9 snapshot succeeds"),
                linear,
                "bytes per element {bytes_per_element}",
            );
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
    fn tiled_guest_copy_uses_physical_pitch_between_block_rows() {
        let width = 128_usize;
        let height = 129_usize;
        let pitch = 256_usize;
        let mut descriptor = GuestTextureDescriptor {
            address: 0x1_0000,
            width: u32::try_from(width).expect("width fits u32"),
            height: u32::try_from(height).expect("height fits u32"),
            pitch: u32::try_from(pitch).expect("pitch fits u32"),
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        let tiled_length = usize::try_from(
            guest_texture_byte_count(descriptor).expect("tiled byte count fits u64"),
        )
        .expect("tiled byte count fits usize");
        assert_eq!(tiled_length, 4 * 65_536);
        let mut tiled = vec![0_u8; tiled_length];
        descriptor.address =
            u64::try_from(tiled.as_mut_ptr() as usize).expect("test address fits u64");
        let linear = (0..width * height * 4)
            .map(|index| u8::try_from(index & 0xff).expect("masked byte fits u8"))
            .collect::<Vec<_>>();

        write_guest_texture_linear(descriptor, &linear).expect("padded tile write succeeds");
        assert!(tiled[65_536..2 * 65_536].iter().all(|&byte| byte == 0));
        assert_eq!(
            snapshot_guest_texture_linear(descriptor).expect("padded tile snapshot succeeds"),
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
    fn mode5_uses_the_exact_standard_4k_address_bits() {
        assert_eq!(mode5_within_block_byte_offset(0, 0, 4), 0);
        assert_eq!(mode5_within_block_byte_offset(1, 0, 4), 4);
        assert_eq!(mode5_within_block_byte_offset(2, 0, 4), 8);
        assert_eq!(mode5_within_block_byte_offset(0, 1, 4), 16);
        assert_eq!(mode5_within_block_byte_offset(0, 4, 4), 64);
        assert_eq!(mode5_within_block_byte_offset(4, 0, 4), 128);
    }

    #[test]
    fn mode1_round_trips_every_supported_standard_element_size() {
        for (bytes_per_element, pixel_format) in [
            (1, MTLPixelFormat::R8Unorm),
            (2, MTLPixelFormat::RG8Unorm),
            (4, MTLPixelFormat::RGBA8Unorm),
            (8, MTLPixelFormat::RGBA16Float),
            (16, MTLPixelFormat::RGBA32Float),
        ] {
            let (block_width, block_height) = tiled_block_dimensions(256, bytes_per_element)
                .expect("element size has a mode 1 equation");
            let width = block_width + 1;
            let height = block_height + 1;
            let mut descriptor = GuestTextureDescriptor {
                address: 0x1_0000,
                width: u32::try_from(width).expect("test width fits u32"),
                height: u32::try_from(height).expect("test height fits u32"),
                pitch: u32::try_from(width).expect("test pitch fits u32"),
                tile_mode: 1,
                pixel_format,
                bytes_per_pixel: u32::try_from(bytes_per_element)
                    .expect("test bytes per element fits u32"),
            };
            let tiled_length = usize::try_from(
                guest_texture_byte_count(descriptor).expect("tiled byte count fits u64"),
            )
            .expect("tiled byte count fits usize");
            let mut tiled = vec![0_u8; tiled_length];
            descriptor.address = u64::try_from(tiled.as_mut_ptr() as usize)
                .expect("test allocation address fits u64");
            let linear = (0..width * height * bytes_per_element)
                .map(|index| u8::try_from(index & 0xff).expect("masked byte fits u8"))
                .collect::<Vec<_>>();

            write_guest_texture_linear(descriptor, &linear).expect("mode 1 write succeeds");
            assert_eq!(
                snapshot_guest_texture_linear(descriptor).expect("mode 1 snapshot succeeds"),
                linear,
                "bytes per element {bytes_per_element}",
            );
        }
    }

    #[test]
    fn mode5_round_trips_uncompressed_and_bc_element_grids() {
        for (bytes_per_element, pixel_format, block_compressed) in [
            (1, MTLPixelFormat::R8Unorm, false),
            (2, MTLPixelFormat::RG8Unorm, false),
            (4, MTLPixelFormat::RGBA8Unorm, false),
            (8, MTLPixelFormat::RGBA16Float, false),
            (16, MTLPixelFormat::RGBA32Float, false),
            (16, MTLPixelFormat::BC7_RGBAUnorm_sRGB, true),
        ] {
            let (block_width, block_height) = tiled_block_dimensions(4_096, bytes_per_element)
                .expect("element size has a mode 5 equation");
            let elements_wide = block_width + 1;
            let elements_high = block_height + 1;
            let width = if block_compressed {
                elements_wide * 4
            } else {
                elements_wide
            };
            let height = if block_compressed {
                elements_high * 4
            } else {
                elements_high
            };
            let mut descriptor = GuestTextureDescriptor {
                address: 0x1_0000,
                width: u32::try_from(width).expect("test width fits u32"),
                height: u32::try_from(height).expect("test height fits u32"),
                pitch: u32::try_from(width).expect("test pitch fits u32"),
                tile_mode: 5,
                pixel_format,
                bytes_per_pixel: u32::try_from(bytes_per_element)
                    .expect("test element size fits u32"),
            };
            let tiled_length = usize::try_from(
                guest_texture_byte_count(descriptor).expect("tiled byte count fits u64"),
            )
            .expect("tiled byte count fits usize");
            let mut tiled = vec![0_u8; tiled_length];
            descriptor.address = u64::try_from(tiled.as_mut_ptr() as usize)
                .expect("test allocation address fits u64");
            let linear_length = elements_wide * elements_high * bytes_per_element;
            let linear = (0..linear_length)
                .map(|index| u8::try_from(index & 0xff).expect("masked byte fits u8"))
                .collect::<Vec<_>>();

            write_guest_texture_linear(descriptor, &linear).expect("mode 5 write succeeds");
            assert_eq!(
                snapshot_guest_texture_linear(descriptor).expect("mode 5 snapshot succeeds"),
                linear,
                "element size {bytes_per_element}, compressed={block_compressed}",
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
    fn rgba_and_bgra_descriptors_can_share_component_reordered_storage() {
        let bgra = GuestTextureDescriptor {
            address: 0x8006_b7b00,
            width: 3_840,
            height: 2_160,
            pitch: 3_840,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::BGRA8Unorm,
            bytes_per_pixel: 4,
        };
        let rgba = GuestTextureDescriptor {
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            ..bgra
        };
        let rgba_srgb = GuestTextureDescriptor {
            pixel_format: MTLPixelFormat::RGBA8Unorm_sRGB,
            ..bgra
        };
        assert!(guest_texture_descriptors_share_component_reordered_storage(
            bgra, rgba
        ));
        assert!(!guest_texture_descriptors_share_storage(bgra, rgba));
        assert!(
            !guest_texture_descriptors_share_component_reordered_storage(bgra, rgba_srgb),
            "linear and sRGB views do not have equivalent sampling semantics"
        );
    }

    #[test]
    fn metal_accepts_a_same_format_red_blue_swizzled_view_without_pixel_format_view_usage() {
        let device = Device::system_default().expect("Metal device");
        let queue = device.new_command_queue();
        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        descriptor.set_width(4);
        descriptor.set_height(4);
        descriptor.set_storage_mode(MTLStorageMode::Private);
        descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        let texture = device.new_texture(&descriptor);
        let view = new_red_blue_swizzled_texture_view(&texture, 0, 1)
            .expect("swizzled Metal texture view");
        assert_eq!(view.pixel_format(), MTLPixelFormat::BGRA8Unorm);
        assert!(view.parent_texture().is_some());

        let pass = RenderPassDescriptor::new();
        let attachment = pass
            .color_attachments()
            .object_at(0)
            .expect("color attachment zero");
        attachment.set_texture(Some(&texture));
        attachment.set_load_action(MTLLoadAction::Clear);
        attachment.set_store_action(MTLStoreAction::Store);
        attachment.set_clear_color(MTLClearColor::new(0.1, 0.2, 0.3, 0.4));
        let clear = queue.new_command_buffer().to_owned();
        let render = clear.new_render_command_encoder(pass);
        render.end_encoding();
        clear.commit();
        clear.wait_until_completed();

        let source = r"
#include <metal_stdlib>
using namespace metal;
kernel void read_swizzled(
    texture2d<float, access::read> input [[texture(0)]],
    device float4* output [[buffer(0)]])
{
    output[0] = input.read(uint2(0));
}
";
        let library = device
            .new_library_with_source(source, &CompileOptions::new())
            .expect("swizzle test MSL compiles");
        let function = library
            .get_function("read_swizzled", None)
            .expect("swizzle test function exists");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .expect("swizzle test pipeline compiles");
        let output = device.new_buffer(16, MTLResourceOptions::StorageModeShared);
        let command = queue.new_command_buffer().to_owned();
        let compute = command.new_compute_command_encoder();
        compute.set_compute_pipeline_state(&pipeline);
        compute.set_texture(0, Some(&view));
        compute.set_buffer(0, Some(&output), 0);
        compute.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        compute.end_encoding();
        command.commit();
        command.wait_until_completed();
        // SAFETY: `output` is a 16-byte shared buffer and the completed kernel wrote one float4.
        let sampled =
            unsafe { std::slice::from_raw_parts(output.contents().cast::<f32>(), 4) }.to_vec();
        let guest_zyxw = [sampled[2], sampled[1], sampled[0], sampled[3]];
        for (actual, expected) in guest_zyxw.into_iter().zip([0.1, 0.2, 0.3, 0.4]) {
            assert!((actual - expected).abs() <= 1.0 / 255.0);
        }
    }

    #[test]
    fn render_resolve_candidates_must_alias_the_sampled_storage() {
        let sampled = GuestTextureDescriptor {
            address: 0x1_0000,
            width: 64,
            height: 64,
            pitch: 64,
            tile_mode: 0,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        let overlapping_render = GuestTextureDescriptor {
            address: sampled.address + 8_192,
            ..sampled
        };
        let unrelated_render = GuestTextureDescriptor {
            address: sampled.address + 32_768,
            ..sampled
        };

        assert!(render_resolve_candidate_matches(
            sampled,
            overlapping_render,
            2,
        ));
        assert!(!render_resolve_candidate_matches(
            sampled,
            unrelated_render,
            2,
        ));
    }

    #[test]
    fn render_resolve_candidate_accepts_view_after_large_page_rounded_msaa_storage() {
        let rendered = GuestTextureDescriptor {
            address: 0x7000_0000,
            width: 1_920,
            height: 1_080,
            pitch: 1_920,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RGBA16Float,
            bytes_per_pixel: 8,
        };
        let sampled = GuestTextureDescriptor {
            address: rendered.address + 0x0200_0000,
            ..rendered
        };

        assert!(render_resolve_candidate_matches(sampled, rendered, 2));
    }

    #[test]
    fn shared_backing_aliases_reuse_identical_metal_storage() {
        use remu_memory::{
            GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
            GuestProtection,
        };

        let mut memory = GuestAddressSpace::new(4 * GUEST_PAGE_SIZE).unwrap();
        let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
        let first = memory
            .map_shared(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .unwrap();
        let second = memory
            .map_shared(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .unwrap();
        let first = GuestTextureDescriptor {
            address: first.get(),
            width: 64,
            height: 64,
            pitch: 64,
            tile_mode: 0,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        let second = GuestTextureDescriptor {
            address: second.get(),
            ..first
        };

        assert!(guest_texture_descriptors_overlap(first, second));
        assert!(guest_texture_descriptors_share_storage(first, second));
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
        validate_compute_launch_for_metal(
            compute_launch(),
            32,
            false,
            1_024,
            32 * 1024,
            0,
            16 * 1024,
        )
        .expect("representable compute launch");

        let mut splittable_wave64 = compute_launch();
        splittable_wave64.wave_lane_count = 64;
        validate_compute_launch_for_metal(
            splittable_wave64,
            32,
            true,
            1_024,
            32 * 1024,
            0,
            16 * 1024,
        )
        .expect("wave-local wave64 shader can execute as two SIMD32 subgroups");
        assert!(
            validate_compute_launch_for_metal(
                splittable_wave64,
                32,
                false,
                1_024,
                32 * 1024,
                0,
                16 * 1024,
            )
            .expect_err("cross-lane wave64 shader must not be split")
            .contains("wave64")
        );
    }

    #[test]
    fn permits_split_wave64_readfirstlane_for_uniform_and_varying_values() {
        let uniform_move = Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop1,
            opcode: Opcode::Known("VMovB32"),
            words: Vec::new(),
            sources: vec![Operand::scalar(18)],
            destinations: vec![Operand::vector(0)],
            control: None,
        };
        let readfirstlane = Instruction {
            pc: 4,
            encoding: ShaderEncoding::Vop1,
            opcode: Opcode::Known("VReadfirstlaneB32"),
            words: Vec::new(),
            sources: vec![Operand::vector(0)],
            destinations: vec![Operand::scalar(107)],
            control: None,
        };
        assert!(
            wave64_split_blockers(&Program::new(
                0x1000,
                vec![uniform_move, readfirstlane.clone()]
            ))
            .is_empty()
        );
        assert!(wave64_split_blockers(&Program::new(0x1000, vec![readfirstlane])).is_empty());
    }

    #[test]
    fn permits_split_wave64_half_local_permlane16_operations() {
        for opcode in ["VPermlane16B32", "VPermlanex16B32"] {
            let instruction = Instruction {
                pc: 0x90,
                encoding: ShaderEncoding::Vop3,
                opcode: Opcode::Known(opcode),
                words: Vec::new(),
                sources: vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
                destinations: vec![Operand::vector(8)],
                control: None,
            };
            assert!(
                wave64_split_blockers(&Program::new(0x1000, vec![instruction])).is_empty(),
                "{opcode} only permutes within one SIMD32 half"
            );
        }
    }

    #[test]
    fn rejects_compute_geometry_metal_cannot_represent_exactly() {
        let mut based = compute_launch();
        based.base_group = [4, 0, 0];
        assert!(
            validate_compute_launch_for_metal(based, 32, false, 1_024, 32 * 1024, 0, 16 * 1024)
                .expect_err("dispatch base is not silently dropped")
                .contains("nonzero base")
        );

        assert!(
            validate_compute_launch_for_metal(
                compute_launch(),
                16,
                false,
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
            validate_compute_launch_for_metal(
                oversized,
                32,
                false,
                1_024,
                32 * 1024,
                0,
                16 * 1024,
            )
                .expect_err("oversized workgroup must fail")
                .contains("1025 threads")
        );

        assert!(
            validate_compute_launch_for_metal(
                compute_launch(),
                32,
                false,
                1_024,
                32 * 1024,
                0,
                8 * 1024,
            )
            .expect_err("LDS binding must equal lowering layout")
            .contains("8192 bytes of LDS")
        );
        assert!(
            validate_compute_launch_for_metal(
                compute_launch(),
                32,
                false,
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
