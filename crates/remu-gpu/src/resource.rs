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

/// Reusable content identity for an immutable index-buffer capture.
///
/// The 128-bit fingerprint records content identity once at capture time without making cache
/// lookups copy or re-hash the complete index vector. Cache keys pair it with the guest source
/// address and exact range, so identical packet captures can reuse prepared buffers while changed
/// bytes at a recycled VA cannot select the old entry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct IndexSnapshotIdentity {
    pub fingerprint: u128,
}

/// Immutable packet-time indices together with a reusable content identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexSnapshot {
    indices: Arc<[u32]>,
    identity: IndexSnapshotIdentity,
}

impl IndexSnapshot {
    #[must_use]
    pub fn new(indices: Arc<[u32]>) -> Self {
        Self {
            identity: IndexSnapshotIdentity {
                fingerprint: index_snapshot_fingerprint(&indices),
            },
            indices,
        }
    }

    #[must_use]
    pub const fn identity(&self) -> IndexSnapshotIdentity {
        self.identity
    }

    #[must_use]
    pub fn indices(&self) -> &Arc<[u32]> {
        &self.indices
    }
}

impl std::ops::Deref for IndexSnapshot {
    type Target = [u32];

    fn deref(&self) -> &Self::Target {
        &self.indices
    }
}

impl AsRef<[u32]> for IndexSnapshot {
    fn as_ref(&self) -> &[u32] {
        &self.indices
    }
}

impl From<Arc<[u32]>> for IndexSnapshot {
    fn from(indices: Arc<[u32]>) -> Self {
        Self::new(indices)
    }
}

impl From<Vec<u32>> for IndexSnapshot {
    fn from(indices: Vec<u32>) -> Self {
        Self::new(indices.into())
    }
}

impl<const N: usize> From<[u32; N]> for IndexSnapshot {
    fn from(indices: [u32; N]) -> Self {
        Self::new(Arc::from(indices))
    }
}

fn index_snapshot_fingerprint(indices: &[u32]) -> u128 {
    // FNV-1a 128. Including the logical element count separates different slices whose byte
    // encodings might otherwise share a prefix. A 128-bit content identity makes collision risk
    // negligible while keeping cache lookup O(1); the immutable bytes remain owned by the draw.
    const FNV_OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const FNV_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
    let mut fingerprint = FNV_OFFSET;
    for byte in u64::try_from(indices.len())
        .unwrap_or(u64::MAX)
        .to_le_bytes()
        .into_iter()
        .chain(indices.iter().flat_map(|index| index.to_le_bytes()))
    {
        fingerprint ^= u128::from(byte);
        fingerprint = fingerprint.wrapping_mul(FNV_PRIME);
    }
    fingerprint
}

/// One cached shader evaluation together with packet-time copies of transient read-only globals.
///
/// Resource discovery is shared across draws through `evaluation`. Only the guest bytes which can
/// be recycled before the presenter consumes the draw are retained per packet. Keeping the two in
/// one value makes it impossible to attach a sidecar to a different evaluation accidentally.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderEvaluationSnapshot {
    evaluation: Arc<ShaderEvaluation>,
    readonly_global_data: Arc<[Option<Arc<[u8]>>]>,
}

impl ShaderEvaluationSnapshot {
    /// Attaches index-aligned packet snapshots to `evaluation` after validating every override.
    pub fn with_readonly_global_data(
        evaluation: Arc<ShaderEvaluation>,
        readonly_global_data: Vec<Option<Arc<[u8]>>>,
    ) -> Result<Self, String> {
        if readonly_global_data.len() != evaluation.global_memory_bindings.len() {
            return Err(format!(
                "shader evaluation has {} global bindings but {} packet snapshots",
                evaluation.global_memory_bindings.len(),
                readonly_global_data.len(),
            ));
        }
        for (index, (binding, data)) in evaluation
            .global_memory_bindings
            .iter()
            .zip(&readonly_global_data)
            .enumerate()
        {
            let Some(data) = data else {
                continue;
            };
            if binding.writable {
                return Err(format!(
                    "packet snapshot {index} overrides a writable global binding"
                ));
            }
            if !binding.data.is_empty() {
                return Err(format!(
                    "packet snapshot {index} overrides data already captured by evaluation"
                ));
            }
            if data.len() != binding.byte_count {
                return Err(format!(
                    "packet snapshot {index} has {} bytes, expected {}",
                    data.len(),
                    binding.byte_count,
                ));
            }
        }
        Ok(Self {
            evaluation,
            readonly_global_data: readonly_global_data.into(),
        })
    }

    /// The shared resource-discovery result. Draw snapshots never clone its owned vectors.
    #[must_use]
    pub fn evaluation(&self) -> &Arc<ShaderEvaluation> {
        &self.evaluation
    }

    /// Packet-time bytes for one binding, when that binding needed a transient snapshot.
    #[must_use]
    pub fn captured_global_binding_data(&self, index: usize) -> Option<&[u8]> {
        self.readonly_global_data
            .get(index)
            .and_then(Option::as_deref)
    }

    /// Effective bytes for one binding: packet sidecar first, cached evaluation second.
    #[must_use]
    pub fn global_binding_data(&self, index: usize) -> Option<&[u8]> {
        let binding = self.evaluation.global_memory_bindings.get(index)?;
        Some(
            self.captured_global_binding_data(index)
                .unwrap_or(&binding.data),
        )
    }
}

impl From<Arc<ShaderEvaluation>> for ShaderEvaluationSnapshot {
    fn from(evaluation: Arc<ShaderEvaluation>) -> Self {
        Self {
            evaluation,
            readonly_global_data: Arc::from([]),
        }
    }
}

impl std::ops::Deref for ShaderEvaluationSnapshot {
    type Target = ShaderEvaluation;

    fn deref(&self) -> &Self::Target {
        &self.evaluation
    }
}

impl AsRef<ShaderEvaluation> for ShaderEvaluationSnapshot {
    fn as_ref(&self) -> &ShaderEvaluation {
        &self.evaluation
    }
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

impl GpuColorTarget {
    /// AGC `CxRenderTarget` / `CB_COLORn_ATTRIB2` extent: height-1 in bits 0..13, width-1 in
    /// bits 14..27 (not GFX10's width@0..15 / height@16..31 packing).
    pub const fn agc_extent(self) -> (u32, u32) {
        agc_color_attrib2_extent(self.attrib2)
    }
}

/// Decode Prospero CB color ATTRIB2 into `(width, height)`.
pub const fn agc_color_attrib2_extent(attrib2: u32) -> (u32, u32) {
    let width = ((attrib2 >> 14) & 0x3fff).saturating_add(1);
    let height = (attrib2 & 0x3fff).saturating_add(1);
    (width, height)
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
    pub index_snapshot: Option<IndexSnapshot>,
    pub draw_flags: u32,
    pub primitive_type: u32,
    pub export_shader_address: u64,
    pub pixel_shader_address: u64,
    pub export_shader_prefix: [u32; 4],
    pub pixel_shader_prefix: [u32; 4],
    pub export_shader: Option<ShaderSnapshot>,
    pub pixel_shader: Option<ShaderSnapshot>,
    pub export_shader_evaluation: Option<ShaderEvaluationSnapshot>,
    pub pixel_shader_evaluation: Option<ShaderEvaluationSnapshot>,
    pub color_target_address: u64,
    pub scanout_target: Option<GpuScanoutTarget>,
    pub context_registers: Arc<[GpuRegister]>,
    pub shader_registers: Arc<[GpuRegister]>,
    pub user_config_registers: Arc<[GpuRegister]>,
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

#[cfg(test)]
mod tests {
    use super::{GpuColorTarget, agc_color_attrib2_extent};

    #[test]
    fn agc_color_attrib2_extent_matches_cx_render_target_packing() {
        // Prospero CxRenderTarget: ((width-1) << 14) | (height-1). 1920×1080 → 0x1dfc437.
        let attrib2 = ((1920 - 1) << 14) | (1080 - 1);
        assert_eq!(attrib2, 0x01df_c437);
        assert_eq!(agc_color_attrib2_extent(attrib2), (1920, 1080));
        // GFX10-style width@0..15/height@16..31 mis-decode yields the bogus 0xc438×0x1e0
        // that polluted Silent Hill COLOR-EMIT logs before the AGC helper.
        assert_ne!((attrib2 & 0xffff).saturating_add(1), 1920);
        let target = GpuColorTarget {
            slot: 0,
            address: 0x8068_e0_0000,
            view: 0,
            info: 0,
            attrib: 0,
            attrib2,
            attrib3: 0,
        };
        assert_eq!(target.agc_extent(), (1920, 1080));
    }
}
