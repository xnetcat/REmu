// SPDX-License-Identifier: GPL-2.0-or-later

use std::fmt;
use std::sync::mpsc::Sender;
use std::sync::{Mutex, OnceLock};

use thiserror::Error;

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
pub struct IndexedDraw {
    pub sequence: u64,
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
    IndexedDraw(IndexedDraw),
    Shutdown,
}

static GPU_EVENT_SINK: OnceLock<Mutex<Option<Sender<GpuEvent>>>> = OnceLock::new();

pub fn install_event_sink(sender: Sender<GpuEvent>) {
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
