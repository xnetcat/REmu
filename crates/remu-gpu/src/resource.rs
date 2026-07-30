// SPDX-License-Identifier: GPL-2.0-or-later

//! Guest-domain GPU resource, adapter and draw description types.

use std::fmt;
use std::sync::Arc;

use remu_shader::{Program, ShaderEvaluation, ShaderMetadata};
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
pub struct ShaderSnapshot {
    pub header_address: u64,
    pub program: Arc<Program>,
    /// Additional entry points retained in execution order. Independently terminated programs
    /// from the leading AGC allocation come first. When AGC fuses a separately allocated back
    /// half, its exact leading program is appended last; a leading `S_SETPC_B64` makes that final
    /// entry a linked continuation rather than an independent segment.
    pub supplemental_programs: Arc<[Program]>,
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
    /// Guest address of the native draw arguments when this draw came from
    /// `DRAW_{INDEX_}INDIRECT`.
    ///
    /// The scalar count fields below retain the packet-time CPU observation for diagnostics and
    /// resource-analysis hints. They are not authoritative for an indirect draw because an
    /// earlier GPU pass can still be producing the argument dwords. Backends must consume this
    /// buffer in queue order instead.
    pub indirect_arguments_address: Option<u64>,
    pub instance_count: u32,
    pub index_buffer_address: u64,
    pub index_resource_address: u64,
    pub index_buffer_count: u32,
    pub index_size: u32,
    pub index_offset: u32,
    /// Signed GFX10 `INDEX_OFFSET` added to each indexed vertex ID (Metal `baseVertex`).
    ///
    /// This is independent of `index_offset`, which selects the first element in the index
    /// buffer for `DRAW_INDEX_OFFSET_2`.
    pub base_vertex: i32,
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
pub struct GpuHostImage {
    width: u32,
    height: u32,
    rgba8: Arc<[u8]>,
}

impl GpuHostImage {
    const MAXIMUM_BYTE_COUNT: usize = 256 * 1024 * 1024;

    /// Builds a validated, tightly packed RGBA8 host image.
    #[must_use]
    pub fn new(width: u32, height: u32, rgba8: impl Into<Arc<[u8]>>) -> Option<Self> {
        let rgba8 = rgba8.into();
        let expected = usize::try_from(width)
            .ok()?
            .checked_mul(usize::try_from(height).ok()?)?
            .checked_mul(4)?;
        if width == 0
            || height == 0
            || expected > Self::MAXIMUM_BYTE_COUNT
            || rgba8.len() != expected
        {
            return None;
        }
        Some(Self {
            width,
            height,
            rgba8,
        })
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub fn rgba8(&self) -> &[u8] {
        &self.rgba8
    }
}
