// SPDX-License-Identifier: GPL-2.0-or-later

use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, Mutex, OnceLock};

use remu_shader::{Program, ShaderEvaluation, ShaderMetadata};
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
    pub metadata: Option<ShaderMetadata>,
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
    pub context_registers: Vec<GpuRegister>,
    pub shader_registers: Vec<GpuRegister>,
    pub user_config_registers: Vec<GpuRegister>,
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
    },
    IndexedDraw(Box<IndexedDraw>),
    Shutdown,
}

static GPU_EVENT_SINK: OnceLock<Mutex<Option<SyncSender<GpuEvent>>>> = OnceLock::new();

pub fn install_event_sink(sender: SyncSender<GpuEvent>) {
    let sink = GPU_EVENT_SINK.get_or_init(|| Mutex::new(None));
    *sink
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(sender);
}

pub fn clear_event_sink() {
    let Some(sink) = GPU_EVENT_SINK.get() else {
        return;
    };
    *sink
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
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
        let _ = sender.send(event);
    }
}

#[cfg(test)]
mod tests {
    use super::{HostPadState, host_pad_state, set_host_pad_state};

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
}
