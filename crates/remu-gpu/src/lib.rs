// SPDX-License-Identifier: GPL-2.0-or-later

use std::fmt;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Mutex, OnceLock};

use remu_shader::{ComputeSystemRegisters, Program, ShaderEvaluation, ShaderMetadata};
use thiserror::Error;

pub const PAD_BUTTON_L3: u32 = 0x0002;
pub const PAD_BUTTON_R3: u32 = 0x0004;
pub const PAD_BUTTON_OPTIONS: u32 = 0x0008;
pub const PAD_BUTTON_UP: u32 = 0x0010;
pub const PAD_BUTTON_RIGHT: u32 = 0x0020;
pub const PAD_BUTTON_DOWN: u32 = 0x0040;
pub const PAD_BUTTON_LEFT: u32 = 0x0080;
pub const PAD_BUTTON_L2: u32 = 0x0100;
pub const PAD_BUTTON_R2: u32 = 0x0200;
pub const PAD_BUTTON_L1: u32 = 0x0400;
pub const PAD_BUTTON_R1: u32 = 0x0800;
pub const PAD_BUTTON_TRIANGLE: u32 = 0x1000;
pub const PAD_BUTTON_CIRCLE: u32 = 0x2000;
pub const PAD_BUTTON_CROSS: u32 = 0x4000;
pub const PAD_BUTTON_SQUARE: u32 = 0x8000;

/// Maximum number of decoded GPU events retained between the guest command processor and the
/// host presenter. Modern Unity frames contain hundreds of draws plus synchronization packets;
/// retaining several thousand events lets the guest command processor and Metal encoder run in
/// parallel instead of context-switching every few draws. The queue remains bounded so a shader
/// compilation stall cannot grow memory without limit.
pub const GPU_EVENT_QUEUE_CAPACITY: usize = 2048;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostPadState {
    pub buttons: u32,
    pub left_x: u8,
    pub left_y: u8,
    pub right_x: u8,
    pub right_y: u8,
    pub l2: u8,
    pub r2: u8,
}

impl Default for HostPadState {
    fn default() -> Self {
        Self {
            buttons: 0,
            left_x: 128,
            left_y: 128,
            right_x: 128,
            right_y: 128,
            l2: 0,
            r2: 0,
        }
    }
}

static HOST_PAD_BUTTONS: AtomicU32 = AtomicU32::new(0);
static HOST_PAD_AXES: AtomicU32 = AtomicU32::new(0x8080_8080);
static HOST_PAD_TRIGGERS: AtomicU32 = AtomicU32::new(0);
static GPU_EVENTS_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
static GPU_COMPLETED_SUBMISSION_SEQUENCE: AtomicU64 = AtomicU64::new(0);
static AUDIO_OUTPUT_FRAMES: AtomicU64 = AtomicU64::new(0);
static AUDIO_UNDERRUN_FRAMES: AtomicU64 = AtomicU64::new(0);

pub fn set_host_pad_state(state: HostPadState) {
    let axes = u32::from_le_bytes([state.left_x, state.left_y, state.right_x, state.right_y]);
    let triggers = u32::from_le_bytes([state.l2, state.r2, 0, 0]);
    HOST_PAD_AXES.store(axes, Ordering::Release);
    HOST_PAD_TRIGGERS.store(triggers, Ordering::Release);
    HOST_PAD_BUTTONS.store(state.buttons, Ordering::Release);
}

#[must_use]
pub fn host_pad_state() -> HostPadState {
    let buttons = HOST_PAD_BUTTONS.load(Ordering::Acquire);
    let [left_x, left_y, right_x, right_y] = HOST_PAD_AXES.load(Ordering::Acquire).to_le_bytes();
    let [l2, r2, _, _] = HOST_PAD_TRIGGERS.load(Ordering::Acquire).to_le_bytes();
    HostPadState {
        buttons,
        left_x,
        left_y,
        right_x,
        right_y,
        l2,
        r2,
    }
}

/// Returns the number of guest GPU events queued or blocked on the bounded presenter channel.
#[must_use]
pub fn gpu_events_in_flight() -> usize {
    GPU_EVENTS_IN_FLIGHT.load(Ordering::Relaxed)
}

/// Records that the presenter removed one event from the bounded guest GPU channel.
pub fn record_gpu_event_consumed() {
    let _ = GPU_EVENTS_IN_FLIGHT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
        Some(value.saturating_sub(1))
    });
}

/// Adds one host audio callback's output and underrun frame counts.
pub fn report_audio_frames(output_frames: u64, underrun_frames: u64) {
    AUDIO_OUTPUT_FRAMES.fetch_add(output_frames, Ordering::Relaxed);
    AUDIO_UNDERRUN_FRAMES.fetch_add(underrun_frames, Ordering::Relaxed);
}

/// Returns the cumulative number of frames requested by the host audio device while a port was
/// open.
#[must_use]
pub fn audio_output_frames() -> u64 {
    AUDIO_OUTPUT_FRAMES.load(Ordering::Relaxed)
}

/// Returns the cumulative number of host audio frames affected by a guest-port underrun.
#[must_use]
pub fn audio_underrun_frames() -> u64 {
    AUDIO_UNDERRUN_FRAMES.load(Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphicsApi {
    Metal,
    Direct3D12,
    Vulkan,
}

impl fmt::Display for GraphicsApi {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metal => formatter.write_str("Metal"),
            Self::Direct3D12 => formatter.write_str("Direct3D 12"),
            Self::Vulkan => formatter.write_str("Vulkan"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuAdapter {
    pub api: GraphicsApi,
    pub name: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuRegister {
    pub offset: u32,
    pub value: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderSnapshot {
    pub header_address: u64,
    pub program: Arc<Program>,
    pub metadata: Option<Arc<ShaderMetadata>>,
}

/// Exact GFX10 color-surface state retained for a display buffer that is named by a later
/// `VideoOut` flip. Some games issue their final sampled blit without rebinding `CB_COLOR`; the
/// command processor associates that draw with a previously programmed scanout surface instead
/// of inventing context-register writes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuColorTarget {
    pub slot: u32,
    pub address: u64,
    pub view: u32,
    pub info: u32,
    pub attrib: u32,
    pub attrib2: u32,
    pub attrib3: u32,
}

/// A display attachment named explicitly by a following `VideoOut` flip. `color_surface` retains
/// exact prior GFX10 surface state when one exists; otherwise the backend renders directly to a
/// host scanout image using the registered display format without fabricating guest registers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuScanoutTarget {
    pub address: u64,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub pixel_format: u64,
    pub tiling_mode: u32,
    pub export_slot: u32,
    pub write_mask: u32,
    pub color_surface: Option<GpuColorTarget>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexedDraw {
    pub sequence: u64,
    pub indexed: bool,
    pub instance_count: u32,
    pub index_buffer_address: u64,
    pub index_resource_address: u64,
    pub index_buffer_count: u32,
    pub index_size: u32,
    pub index_offset: u32,
    pub index_count: u32,
    /// Immutable indices captured while the guest submission still owns its transient ring data.
    /// The presenter can lag command decoding during shader compilation, so retaining only the
    /// guest address would allow Unity to recycle and overwrite the indices before Metal reads
    /// them.
    pub index_snapshot: Option<Arc<[u32]>>,
    pub draw_flags: u32,
    pub primitive_type: u32,
    pub export_shader_address: u64,
    pub pixel_shader_address: u64,
    pub export_shader_prefix: [u32; 4],
    pub pixel_shader_prefix: [u32; 4],
    pub export_shader: Option<ShaderSnapshot>,
    pub pixel_shader: Option<ShaderSnapshot>,
    pub export_shader_evaluation: Option<Arc<ShaderEvaluation>>,
    pub pixel_shader_evaluation: Option<Arc<ShaderEvaluation>>,
    pub color_target_address: u64,
    pub scanout_target: Option<GpuScanoutTarget>,
    pub context_registers: Vec<GpuRegister>,
    pub shader_registers: Vec<GpuRegister>,
    pub user_config_registers: Vec<GpuRegister>,
}

pub const COMPUTE_START_X_REGISTER: u32 = 0x204;
pub const COMPUTE_START_Y_REGISTER: u32 = 0x205;
pub const COMPUTE_START_Z_REGISTER: u32 = 0x206;
pub const COMPUTE_NUM_THREAD_X_REGISTER: u32 = 0x207;
pub const COMPUTE_NUM_THREAD_Y_REGISTER: u32 = 0x208;
pub const COMPUTE_NUM_THREAD_Z_REGISTER: u32 = 0x209;
pub const COMPUTE_PGM_LO_REGISTER: u32 = 0x20c;
pub const COMPUTE_PGM_HI_REGISTER: u32 = 0x20d;
pub const COMPUTE_PGM_RSRC2_REGISTER: u32 = 0x213;

const PM4_TYPE3: u32 = 3;
const PM4_DISPATCH_DIRECT: u32 = 0x15;
const PM4_DISPATCH_INDIRECT: u32 = 0x16;
const DISPATCH_ENABLE: u32 = 1;
const DISPATCH_PARTIAL_THREADGROUP: u32 = 1 << 1;
const DISPATCH_FORCE_START_AT_ZERO: u32 = 1 << 2;
const DISPATCH_USE_THREAD_DIMENSIONS: u32 = 1 << 5;
const DISPATCH_WAVE32: u32 = 1 << 15;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ComputeGrid {
    /// A regular grid whose dimensions are workgroup counts.
    Threadgroups([u32; 3]),
    /// An exact thread grid. Metal must use `dispatchThreads` so the final workgroup is not padded.
    Threads([u64; 3]),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ComputeLaunch {
    pub grid: ComputeGrid,
    pub group_count: [u32; 3],
    pub base_group: [u32; 3],
    pub threads_per_group: [u32; 3],
    pub wave_lane_count: u32,
    pub lds_size_bytes: u32,
    pub initiator: u32,
}

impl ComputeLaunch {
    #[must_use]
    pub const fn threads_per_group_total(self) -> u64 {
        self.threads_per_group[0] as u64
            * self.threads_per_group[1] as u64
            * self.threads_per_group[2] as u64
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ComputeDispatchSource {
    Direct,
    Indirect { dimensions_address: u64 },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ComputeIndirectDimensions {
    BaseOffset(u32),
    Absolute(u64),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ComputeIndirectPacket {
    pub dimensions: ComputeIndirectDimensions,
    pub initiator: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComputeDispatch {
    pub sequence: u64,
    pub source: ComputeDispatchSource,
    pub launch: ComputeLaunch,
    pub shader_address: u64,
    pub shader_prefix: [u32; 4],
    pub shader: Option<ShaderSnapshot>,
    pub shader_evaluation: Option<Arc<ShaderEvaluation>>,
    pub shader_registers: Vec<GpuRegister>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub enum ComputeDispatchDecodeError {
    #[error("compute packet has {actual} dwords, expected {expected}")]
    PacketLength { expected: u32, actual: usize },
    #[error("compute packet declares unsupported length of {0} dwords")]
    DeclaredPacketLength(u32),
    #[error("PM4 header 0x{0:08X} is not DISPATCH_DIRECT")]
    NotDispatchDirect(u32),
    #[error("PM4 header 0x{0:08X} is not DISPATCH_INDIRECT")]
    NotDispatchIndirect(u32),
    #[error("dispatch initiator 0x{0:08X} does not enable compute")]
    Disabled(u32),
    #[error("compute {axis}-dimension end {end} is not after base {base}")]
    EndNotAfterBase { axis: char, end: u64, base: u64 },
    #[error("compute {axis}-dimension partial size {partial} exceeds full size {full}")]
    InvalidPartialSize { axis: char, partial: u32, full: u32 },
    #[error("compute {axis}-dimension thread count overflows")]
    ThreadCountOverflow { axis: char },
}

/// Decodes the complete geometry of a Gen5 direct or resolved-indirect compute dispatch.
///
/// Packet dimensions are end coordinates unless `USE_THREAD_DIMENSIONS` is set. When
/// `FORCE_START_AT_000` is clear, `COMPUTE_START_{X,Y,Z}` supplies the base coordinates. Partial
/// workgroups become an exact thread grid instead of being rounded up with extra invocations.
pub fn decode_compute_launch(
    dimensions: [u32; 3],
    initiator: u32,
    shader_registers: &[GpuRegister],
) -> Result<ComputeLaunch, ComputeDispatchDecodeError> {
    if initiator & DISPATCH_ENABLE == 0 {
        return Err(ComputeDispatchDecodeError::Disabled(initiator));
    }
    let register = |offset| {
        shader_registers
            .iter()
            .find(|register| register.offset == offset)
            .map_or(0, |register| register.value)
    };
    let thread_registers = [
        register(COMPUTE_NUM_THREAD_X_REGISTER),
        register(COMPUTE_NUM_THREAD_Y_REGISTER),
        register(COMPUTE_NUM_THREAD_Z_REGISTER),
    ];
    let threads_per_group = thread_registers.map(|value| (value & 0xffff).max(1));
    let base_group = if initiator & DISPATCH_FORCE_START_AT_ZERO != 0 {
        [0; 3]
    } else {
        [
            register(COMPUTE_START_X_REGISTER),
            register(COMPUTE_START_Y_REGISTER),
            register(COMPUTE_START_Z_REGISTER),
        ]
    };
    let axes = ['x', 'y', 'z'];
    let mut group_count = [0_u32; 3];
    let grid = if initiator & DISPATCH_USE_THREAD_DIMENSIONS != 0 {
        let mut threads = [0_u64; 3];
        for axis in 0..3 {
            let start_thread = u64::from(base_group[axis])
                .checked_mul(u64::from(threads_per_group[axis]))
                .ok_or(ComputeDispatchDecodeError::ThreadCountOverflow { axis: axes[axis] })?;
            let end_thread = u64::from(dimensions[axis]);
            if end_thread <= start_thread {
                return Err(ComputeDispatchDecodeError::EndNotAfterBase {
                    axis: axes[axis],
                    end: end_thread,
                    base: start_thread,
                });
            }
            threads[axis] = end_thread - start_thread;
            let groups = threads[axis].div_ceil(u64::from(threads_per_group[axis]));
            group_count[axis] = u32::try_from(groups).map_err(|_| {
                ComputeDispatchDecodeError::ThreadCountOverflow { axis: axes[axis] }
            })?;
        }
        ComputeGrid::Threads(threads)
    } else {
        for axis in 0..3 {
            if dimensions[axis] <= base_group[axis] {
                return Err(ComputeDispatchDecodeError::EndNotAfterBase {
                    axis: axes[axis],
                    end: u64::from(dimensions[axis]),
                    base: u64::from(base_group[axis]),
                });
            }
            group_count[axis] = dimensions[axis] - base_group[axis];
        }
        if initiator & DISPATCH_PARTIAL_THREADGROUP == 0 {
            ComputeGrid::Threadgroups(group_count)
        } else {
            let mut threads = [0_u64; 3];
            for axis in 0..3 {
                let partial = thread_registers[axis] >> 16;
                let full = threads_per_group[axis];
                if partial == 0 || partial > full {
                    return Err(ComputeDispatchDecodeError::InvalidPartialSize {
                        axis: axes[axis],
                        partial,
                        full,
                    });
                }
                threads[axis] = u64::from(group_count[axis] - 1)
                    .checked_mul(u64::from(full))
                    .and_then(|prefix| prefix.checked_add(u64::from(partial)))
                    .ok_or(ComputeDispatchDecodeError::ThreadCountOverflow { axis: axes[axis] })?;
            }
            ComputeGrid::Threads(threads)
        }
    };
    let resource2 = register(COMPUTE_PGM_RSRC2_REGISTER);
    Ok(ComputeLaunch {
        grid,
        group_count,
        base_group,
        threads_per_group,
        wave_lane_count: if initiator & DISPATCH_WAVE32 != 0 {
            32
        } else {
            64
        },
        lds_size_bytes: ((resource2 >> 15) & 0x1ff) * 512,
        initiator,
    })
}

/// Decodes one complete type-3 `DISPATCH_DIRECT` packet and its active compute registers.
pub fn decode_compute_dispatch_direct(
    packet: &[u32],
    shader_registers: &[GpuRegister],
) -> Result<ComputeLaunch, ComputeDispatchDecodeError> {
    let Some(&header) = packet.first() else {
        return Err(ComputeDispatchDecodeError::PacketLength {
            expected: 5,
            actual: 0,
        });
    };
    let declared = ((header >> 16) & 0x3fff) + 2;
    if declared != 5 {
        return Err(ComputeDispatchDecodeError::DeclaredPacketLength(declared));
    }
    if packet.len() != 5 {
        return Err(ComputeDispatchDecodeError::PacketLength {
            expected: 5,
            actual: packet.len(),
        });
    }
    if header >> 30 != PM4_TYPE3 || (header >> 8) & 0xff != PM4_DISPATCH_DIRECT {
        return Err(ComputeDispatchDecodeError::NotDispatchDirect(header));
    }
    decode_compute_launch(
        [packet[1], packet[2], packet[3]],
        packet[4],
        shader_registers,
    )
}

/// Decodes either observed Gen5 `DISPATCH_INDIRECT` packet layout without dereferencing guest
/// memory. Three-dword packets carry an offset from the active indirect-argument base; four-dword
/// packets carry an absolute 64-bit dimensions address.
pub fn decode_compute_dispatch_indirect(
    packet: &[u32],
) -> Result<ComputeIndirectPacket, ComputeDispatchDecodeError> {
    let Some(&header) = packet.first() else {
        return Err(ComputeDispatchDecodeError::PacketLength {
            expected: 3,
            actual: 0,
        });
    };
    if header >> 30 != PM4_TYPE3 || (header >> 8) & 0xff != PM4_DISPATCH_INDIRECT {
        return Err(ComputeDispatchDecodeError::NotDispatchIndirect(header));
    }
    let declared = ((header >> 16) & 0x3fff) + 2;
    if !matches!(declared, 3 | 4) {
        return Err(ComputeDispatchDecodeError::DeclaredPacketLength(declared));
    }
    if packet.len() != usize::try_from(declared).unwrap_or(usize::MAX) {
        return Err(ComputeDispatchDecodeError::PacketLength {
            expected: declared,
            actual: packet.len(),
        });
    }
    if declared == 3 {
        Ok(ComputeIndirectPacket {
            dimensions: ComputeIndirectDimensions::BaseOffset(packet[1]),
            initiator: packet[2],
        })
    } else {
        Ok(ComputeIndirectPacket {
            dimensions: ComputeIndirectDimensions::Absolute(
                u64::from(packet[1]) | (u64::from(packet[2]) << 32),
            ),
            initiator: packet[3],
        })
    }
}

#[must_use]
pub fn compute_program_address(shader_registers: &[GpuRegister]) -> Option<u64> {
    let low = shader_registers
        .iter()
        .find(|register| register.offset == COMPUTE_PGM_LO_REGISTER)?
        .value;
    let high = shader_registers
        .iter()
        .find(|register| register.offset == COMPUTE_PGM_HI_REGISTER)?
        .value;
    // GFX10 COMPUTE_PGM_HI carries only address bits 47:40. Reserved high bits are not part of
    // the virtual address (Kyty's command processor applies the same 8-bit mask).
    Some((u64::from(high & 0xff) << 40) | (u64::from(low) << 8))
}

#[must_use]
pub fn decode_compute_system_registers(shader_registers: &[GpuRegister]) -> ComputeSystemRegisters {
    let resource2 = shader_registers
        .iter()
        .find(|register| register.offset == COMPUTE_PGM_RSRC2_REGISTER)
        .map_or(0, |register| register.value);
    let mut next_register = (resource2 >> 1) & 0x1f;
    let mut consume_enabled = |mask| {
        if resource2 & mask == 0 {
            return None;
        }
        let register = next_register;
        next_register += 1;
        Some(register)
    };
    ComputeSystemRegisters {
        work_group_x: consume_enabled(1 << 7),
        work_group_y: consume_enabled(1 << 8),
        work_group_z: consume_enabled(1 << 9),
        thread_group_size: consume_enabled(1 << 10),
    }
}

#[derive(Debug, Error)]
pub enum GpuProbeError {
    #[error("{0} is unavailable on this host")]
    Unsupported(GraphicsApi),
    #[error("no {0} device is available")]
    NoDevice(GraphicsApi),
}

pub trait GpuProbe {
    fn probe_default() -> Result<GpuAdapter, GpuProbeError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuEvent {
    DisplayRegistered {
        width: u32,
        height: u32,
        pitch: u32,
        pixel_format: u64,
        tiling_mode: u32,
        buffers: Vec<u64>,
    },
    Flip {
        address: u64,
        buffer_index: u32,
        flip_argument: u64,
        width: u32,
        height: u32,
        pitch: u32,
        pixel_format: u64,
        tiling_mode: u32,
    },
    PipelineBarrier(GpuPipelineBarrier),
    MemoryAcquire(GpuMemoryAcquire),
    MemoryCopy(GpuMemoryCopy),
    MemoryWrite(GpuMemoryWrite),
    MemoryRelease(GpuMemoryRelease),
    MemoryWait(GpuMemoryWait),
    ComputeDispatch(Box<ComputeDispatch>),
    IndexedDraw(Box<IndexedDraw>),
    /// An ordered CPU-visible end-of-pipe boundary for one guest graphics submission.
    ///
    /// The backend records this sequence only after all preceding translated GPU work and guest
    /// memory writebacks have completed. The kernel equeue implementation uses the resulting
    /// monotonic value to deliver `sceAgcDriverAddEqEvent` completion notifications.
    SubmissionBoundary(u64),
    Shutdown,
}

/// A guest pipeline-stage completion boundary emitted by a PM4 partial-flush event.
///
/// Backends may use a lighter native primitive than a CPU wait, but following GPU work must not
/// overtake work from the named stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuPipelineBarrier {
    pub wait_for: GpuPipelineStage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuPipelineStage {
    ComputeShader,
    VertexShader,
    PixelShader,
}

/// A decoded Gen5 `ACQUIRE_MEM` packet in guest byte-address form.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMemoryAcquire {
    pub engine: u32,
    pub cb_db_control: u32,
    pub base_address: u64,
    pub size_bytes: u64,
    pub poll_interval: u32,
    pub gcr_control: u32,
}

/// An ordered command-processor byte copy emitted by a Gen5 `DMA_DATA` packet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMemoryCopy {
    pub source_address: u64,
    pub destination_address: u64,
    pub byte_count: u64,
    pub source_selection: u32,
    pub destination_selection: u32,
    /// The command parser already mirrored this raw guest-memory copy for CPU-side shader
    /// descriptor discovery. Backends still consume the event for ordering and resource-cache
    /// invalidation, and must redo it if an earlier GPU write supplies the source.
    pub cpu_applied: bool,
}

/// An ordered CPU-visible label write produced by a Gen5 end-of-pipe packet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMemoryRelease {
    pub address: u64,
    pub value: GpuMemoryReleaseValue,
}

/// An ordered command-processor write of one or more dwords to guest memory.
///
/// `WRITE_DATA` can either advance the destination for every dword or repeatedly update one
/// address. The latter is used for register-like labels, while Unity's submit-done fence uses an
/// incrementing memory destination.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuMemoryWrite {
    pub address: u64,
    pub values: Vec<u32>,
    pub increment_address: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuMemoryReleaseValue {
    Immediate32(u32),
    Immediate64(u64),
    Timestamp,
}

/// An ordered Gen5 command-processor wait on a guest-memory label.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuMemoryWait {
    pub address: u64,
    pub reference: u64,
    pub mask: u64,
    pub compare_function: u32,
    pub is_64_bit: bool,
}

impl GpuMemoryWait {
    #[must_use]
    pub const fn is_satisfied_by(self, value: u64) -> bool {
        let value = value & self.mask;
        match self.compare_function {
            0 => true,
            1 => value < self.reference,
            2 => value <= self.reference,
            3 => value == self.reference,
            4 => value != self.reference,
            5 => value >= self.reference,
            6 => value > self.reference,
            _ => true,
        }
    }
}

impl GpuMemoryAcquire {
    const GLI_INVALIDATE_MASK: u32 = 0x3;
    const GL1_RANGE_SHIFT: u32 = 2;
    const GL1_RANGE_MASK: u32 = 0x3;
    const GLM_INVALIDATE: u32 = 1 << 5;
    const GLK_INVALIDATE: u32 = 1 << 7;
    const GLV_INVALIDATE: u32 = 1 << 8;
    const GL1_INVALIDATE: u32 = 1 << 9;
    const GL2_RANGE_SHIFT: u32 = 11;
    const GL2_RANGE_MASK: u32 = 0x3;
    const GL2_DISCARD: u32 = 1 << 13;
    const GL2_INVALIDATE: u32 = 1 << 14;

    /// Returns whether this packet invalidates host-cached guest resources.
    #[must_use]
    pub const fn invalidates_guest_resources(self) -> bool {
        self.gcr_control
            & (Self::GLI_INVALIDATE_MASK
                | Self::GLM_INVALIDATE
                | Self::GLK_INVALIDATE
                | Self::GLV_INVALIDATE
                | Self::GL1_INVALIDATE
                | Self::GL2_DISCARD
                | Self::GL2_INVALIDATE)
            != 0
    }

    /// Returns whether the packet's effective invalidation scope covers all guest memory.
    #[must_use]
    pub const fn covers_all_guest_memory(self) -> bool {
        self.size_bytes == 0
            || self.gcr_control & Self::GLI_INVALIDATE_MASK == 1
            || (self.gcr_control
                & (Self::GLM_INVALIDATE
                    | Self::GLK_INVALIDATE
                    | Self::GLV_INVALIDATE
                    | Self::GL1_INVALIDATE)
                != 0
                && (self.gcr_control >> Self::GL1_RANGE_SHIFT) & Self::GL1_RANGE_MASK == 0)
            || (self.gcr_control & (Self::GL2_DISCARD | Self::GL2_INVALIDATE) != 0
                && (self.gcr_control >> Self::GL2_RANGE_SHIFT) & Self::GL2_RANGE_MASK == 0)
    }

    /// Returns whether a guest resource intersects the packet's effective invalidation scope.
    #[must_use]
    pub const fn overlaps(self, address: u64, byte_count: u64) -> bool {
        if self.covers_all_guest_memory() {
            return true;
        }
        let resource_end = address.saturating_add(byte_count);
        let acquire_end = self.base_address.saturating_add(self.size_bytes);
        address < acquire_end && self.base_address < resource_end
    }
}

static GPU_EVENT_SINK: OnceLock<Mutex<Option<SyncSender<GpuEvent>>>> = OnceLock::new();

/// Creates the bounded, ordered event channel used by the guest GPU command processor.
#[must_use]
pub fn gpu_event_channel() -> (SyncSender<GpuEvent>, Receiver<GpuEvent>) {
    std::sync::mpsc::sync_channel(GPU_EVENT_QUEUE_CAPACITY)
}

pub fn install_event_sink(sender: SyncSender<GpuEvent>) {
    GPU_EVENTS_IN_FLIGHT.store(0, Ordering::Relaxed);
    GPU_COMPLETED_SUBMISSION_SEQUENCE.store(0, Ordering::Release);
    AUDIO_OUTPUT_FRAMES.store(0, Ordering::Relaxed);
    AUDIO_UNDERRUN_FRAMES.store(0, Ordering::Relaxed);
    let sink = GPU_EVENT_SINK.get_or_init(|| Mutex::new(None));
    *sink
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(sender);
}

pub fn clear_event_sink() {
    GPU_EVENTS_IN_FLIGHT.store(0, Ordering::Relaxed);
    GPU_COMPLETED_SUBMISSION_SEQUENCE.store(0, Ordering::Release);
    let Some(sink) = GPU_EVENT_SINK.get() else {
        return;
    };
    *sink
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
}

/// Returns the newest guest submission whose ordered backend work is fully complete.
#[must_use]
pub fn gpu_completed_submission_sequence() -> u64 {
    GPU_COMPLETED_SUBMISSION_SEQUENCE.load(Ordering::Acquire)
}

/// Publishes completion of an ordered guest submission.
///
/// Backends may retire more than one boundary between guest polls, so completion is monotonic
/// even if callbacks are observed out of order on a future multi-queue implementation.
pub fn record_gpu_submission_completed(sequence: u64) {
    let _ = GPU_COMPLETED_SUBMISSION_SEQUENCE.fetch_max(sequence, Ordering::AcqRel);
}

pub fn emit_event(event: GpuEvent) {
    let Some(sink) = GPU_EVENT_SINK.get() else {
        return;
    };
    let sender = sink
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone();
    if let Some(sender) = sender {
        GPU_EVENTS_IN_FLIGHT.fetch_add(1, Ordering::Relaxed);
        if sender.send(event).is_err() {
            record_gpu_event_consumed();
        }
    }
}

/// Emits an explicit guest GPU cache-invalidation boundary. Event ordering keeps the invalidation
/// between preceding and following guest GPU work without stalling the guest command processor.
pub fn acquire_memory(acquire: GpuMemoryAcquire) {
    emit_event(GpuEvent::MemoryAcquire(acquire));
}

/// Emits an ordered guest pipeline-stage completion boundary.
pub fn pipeline_barrier(barrier: GpuPipelineBarrier) {
    emit_event(GpuEvent::PipelineBarrier(barrier));
}

#[cfg(test)]
mod tests {
    use super::{
        COMPUTE_NUM_THREAD_X_REGISTER, COMPUTE_NUM_THREAD_Y_REGISTER,
        COMPUTE_NUM_THREAD_Z_REGISTER, COMPUTE_PGM_HI_REGISTER, COMPUTE_PGM_LO_REGISTER,
        COMPUTE_PGM_RSRC2_REGISTER, COMPUTE_START_X_REGISTER, COMPUTE_START_Y_REGISTER,
        COMPUTE_START_Z_REGISTER, ComputeDispatchDecodeError, ComputeGrid,
        GPU_EVENT_QUEUE_CAPACITY, GpuEvent, GpuMemoryAcquire, GpuMemoryWait, GpuRegister,
        HostPadState, audio_output_frames, audio_underrun_frames, clear_event_sink,
        compute_program_address, decode_compute_dispatch_direct, decode_compute_dispatch_indirect,
        decode_compute_launch, decode_compute_system_registers, emit_event,
        gpu_completed_submission_sequence, gpu_event_channel, gpu_events_in_flight, host_pad_state,
        install_event_sink, record_gpu_event_consumed, record_gpu_submission_completed,
        report_audio_frames, set_host_pad_state,
    };

    #[test]
    fn gpu_event_channel_applies_bounded_backpressure() {
        let (sender, receiver) = gpu_event_channel();
        for _ in 0..GPU_EVENT_QUEUE_CAPACITY {
            sender
                .try_send(GpuEvent::Shutdown)
                .expect("event fits bounded queue");
        }
        assert!(matches!(
            sender.try_send(GpuEvent::Shutdown),
            Err(std::sync::mpsc::TrySendError::Full(GpuEvent::Shutdown))
        ));
        assert_eq!(receiver.try_recv(), Ok(GpuEvent::Shutdown));
        sender
            .try_send(GpuEvent::Shutdown)
            .expect("receiver progress releases producer backpressure");
    }

    #[test]
    fn runtime_telemetry_tracks_gpu_queue_and_audio_callback_progress() {
        let (sender, receiver) = gpu_event_channel();
        install_event_sink(sender);
        emit_event(GpuEvent::Shutdown);
        assert_eq!(gpu_events_in_flight(), 1);
        assert_eq!(receiver.recv(), Ok(GpuEvent::Shutdown));
        record_gpu_event_consumed();
        assert_eq!(gpu_events_in_flight(), 0);

        record_gpu_submission_completed(7);
        record_gpu_submission_completed(3);
        assert_eq!(gpu_completed_submission_sequence(), 7);

        report_audio_frames(480, 12);
        assert_eq!(audio_output_frames(), 480);
        assert_eq!(audio_underrun_frames(), 12);
        clear_event_sink();
    }

    #[test]
    fn shares_a_host_pad_snapshot_across_the_gpu_runtime_seam() {
        let state = HostPadState {
            buttons: 0x4030,
            left_x: 1,
            left_y: 2,
            right_x: 3,
            right_y: 4,
            l2: 5,
            r2: 6,
        };
        set_host_pad_state(state);
        assert_eq!(host_pad_state(), state);
        set_host_pad_state(HostPadState::default());
    }

    #[test]
    fn decodes_acquire_invalidation_scope_without_backend_assumptions() {
        let range = GpuMemoryAcquire {
            engine: 1,
            cb_db_control: 0x7fc0,
            base_address: 0x1234_0000,
            size_bytes: 0x8000,
            poll_interval: 10,
            gcr_control: 0x1_0388,
        };
        assert!(range.invalidates_guest_resources());
        assert!(!range.covers_all_guest_memory());
        assert!(range.overlaps(0x1234_7000, 0x2000));
        assert!(!range.overlaps(0x1235_0000, 0x1000));

        let all = GpuMemoryAcquire {
            size_bytes: 0,
            gcr_control: 0x280,
            ..range
        };
        assert!(all.invalidates_guest_resources());
        assert!(all.covers_all_guest_memory());
        assert!(all.overlaps(0x7000_0000_0000, 0x1000));
    }

    #[test]
    fn memory_wait_comparisons_apply_the_packet_mask() {
        let wait = GpuMemoryWait {
            address: 0x1000,
            reference: 0x34,
            mask: 0xff,
            compare_function: 3,
            is_64_bit: false,
        };
        assert!(wait.is_satisfied_by(0xabcd_1234));
        assert!(!wait.is_satisfied_by(0xabcd_1235));
        assert!(
            GpuMemoryWait {
                compare_function: 4,
                ..wait
            }
            .is_satisfied_by(0xabcd_1235)
        );
    }

    fn compute_registers(thread_words: [u32; 3], starts: [u32; 3]) -> Vec<GpuRegister> {
        [
            (COMPUTE_NUM_THREAD_X_REGISTER, thread_words[0]),
            (COMPUTE_NUM_THREAD_Y_REGISTER, thread_words[1]),
            (COMPUTE_NUM_THREAD_Z_REGISTER, thread_words[2]),
            (COMPUTE_START_X_REGISTER, starts[0]),
            (COMPUTE_START_Y_REGISTER, starts[1]),
            (COMPUTE_START_Z_REGISTER, starts[2]),
            (COMPUTE_PGM_RSRC2_REGISTER, 64 << 15),
        ]
        .map(|(offset, value)| GpuRegister { offset, value })
        .to_vec()
    }

    #[test]
    fn decodes_direct_group_dimensions_wave_size_and_lds() {
        let registers = compute_registers([64, 2, 1], [99, 99, 99]);
        let packet = [0xc003_1500, 10, 4, 2, 0x8045];
        let launch =
            decode_compute_dispatch_direct(&packet, &registers).expect("valid direct dispatch");
        assert_eq!(launch.grid, ComputeGrid::Threadgroups([10, 4, 2]));
        assert_eq!(launch.group_count, [10, 4, 2]);
        assert_eq!(launch.base_group, [0; 3]);
        assert_eq!(launch.threads_per_group, [64, 2, 1]);
        assert_eq!(launch.threads_per_group_total(), 128);
        assert_eq!(launch.wave_lane_count, 32);
        assert_eq!(launch.lds_size_bytes, 32 * 1024);
    }

    #[test]
    fn decodes_thread_dimensions_as_an_exact_unpadded_grid() {
        let registers = compute_registers([64, 1, 1], [0; 3]);
        let launch = decode_compute_launch([130, 3, 1], 0x8025, &registers)
            .expect("valid thread-dimension dispatch");
        assert_eq!(launch.grid, ComputeGrid::Threads([130, 3, 1]));
        assert_eq!(launch.group_count, [3, 3, 1]);
    }

    #[test]
    fn decodes_partial_final_workgroups_without_padded_invocations() {
        let registers = compute_registers([(32 << 16) | 64, (1 << 16) | 4, (1 << 16) | 1], [0; 3]);
        let launch = decode_compute_launch([3, 2, 1], 0x8007, &registers)
            .expect("valid partial-threadgroup dispatch");
        assert_eq!(launch.grid, ComputeGrid::Threads([160, 5, 1]));
        assert_eq!(launch.group_count, [3, 2, 1]);
    }

    #[test]
    fn preserves_nonzero_compute_start_for_backend_validation() {
        let registers = compute_registers([8, 4, 2], [4, 3, 2]);
        let launch = decode_compute_launch([10, 8, 5], 0x8001, &registers)
            .expect("valid based dispatch geometry");
        assert_eq!(launch.base_group, [4, 3, 2]);
        assert_eq!(launch.group_count, [6, 5, 3]);
        assert_eq!(launch.grid, ComputeGrid::Threadgroups([6, 5, 3]));
    }

    #[test]
    fn rejects_disabled_and_malformed_compute_dispatches() {
        let registers = compute_registers([64, 1, 1], [0; 3]);
        assert_eq!(
            decode_compute_launch([1; 3], 0x8004, &registers),
            Err(ComputeDispatchDecodeError::Disabled(0x8004))
        );
        assert!(matches!(
            decode_compute_dispatch_direct(&[0xc002_1500, 1, 1, 1], &registers),
            Err(ComputeDispatchDecodeError::DeclaredPacketLength(4))
        ));
        assert!(matches!(
            decode_compute_launch([1; 3], 0x8003, &compute_registers([64, 1, 1], [0; 3])),
            Err(ComputeDispatchDecodeError::InvalidPartialSize { axis: 'x', .. })
        ));
    }

    #[test]
    fn decodes_compute_program_and_system_sgpr_state() {
        let registers = [
            GpuRegister {
                offset: COMPUTE_PGM_LO_REGISTER,
                value: 0x1234_5678,
            },
            GpuRegister {
                offset: COMPUTE_PGM_HI_REGISTER,
                value: 0xdead_be5a,
            },
            GpuRegister {
                offset: COMPUTE_PGM_RSRC2_REGISTER,
                value: (4 << 1) | (1 << 7) | (1 << 9) | (1 << 10),
            },
        ];
        assert_eq!(compute_program_address(&registers), Some(0x5a12_3456_7800));
        let system = decode_compute_system_registers(&registers);
        assert_eq!(system.work_group_x, Some(4));
        assert_eq!(system.work_group_y, None);
        assert_eq!(system.work_group_z, Some(5));
        assert_eq!(system.thread_group_size, Some(6));
    }

    #[test]
    fn decodes_base_relative_and_absolute_indirect_dispatch_packets() {
        let relative = decode_compute_dispatch_indirect(&[0xc001_1600, 0x280, 0x8041])
            .expect("relative indirect packet");
        assert_eq!(
            relative.dimensions,
            super::ComputeIndirectDimensions::BaseOffset(0x280)
        );
        assert_eq!(relative.initiator, 0x8041);

        let absolute =
            decode_compute_dispatch_indirect(&[0xc002_1600, 0x7654_3200, 0x1234, 0x8001])
                .expect("absolute indirect packet");
        assert_eq!(
            absolute.dimensions,
            super::ComputeIndirectDimensions::Absolute(0x1234_7654_3200)
        );
        assert_eq!(absolute.initiator, 0x8001);
    }
}
