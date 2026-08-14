// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Backend lowering from `REMu`'s Gen5 shader IR to Metal Shading Language.
//!
//! Translation is deliberately strict. An instruction is either emitted with its architectural
//! semantics or compilation fails with the instruction PC and opcode; this crate never substitutes
//! a placeholder color, drops an unknown memory access, or selects behavior by game identity.

use std::fmt::Write as _;

mod compiler;

pub(crate) use compiler::Compiler;
#[cfg(test)]
use compiler::memory::{buffer_oob_expression, format_buffer_element_oob_expression};
pub(crate) use compiler::operand::format_u32;
#[cfg(test)]
use compiler::vector::is_supported_dpp_control;
#[cfg(test)]
use remu_shader::{
    ExportControl, ImageBinding, ImageControl, InstructionControl, Operand, OperandKind,
    PixelExportFormat, PixelOutputKind, SdwaControl, ShaderEncoding,
};
use remu_shader::{Opcode, PixelOutputBinding, Program, ShaderEvaluation, VertexInputBinding};

const SCALAR_REGISTER_COUNT: u32 = 128;
const VECTOR_REGISTER_COUNT: u32 = 256;
const METAL_DIRECT_TEXTURE_LIMIT: u32 = 128;
const VCC_LO: u32 = 106;
const VCC_HI: u32 = 107;
const EXEC_LO: u32 = 126;
const EXEC_HI: u32 = 127;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MetalShaderStage {
    Vertex,
    Pixel,
    Compute,
    /// A fused GFX10 export/geometry program lowered to a Metal mesh function.
    ///
    /// Unlike a vertex function, a mesh function has workgroup-shared memory and can publish
    /// primitive connectivity as well as vertex outputs. That is the closest Metal execution
    /// model for RDNA2 NGG programs which exchange ES/GS records through LDS.
    Mesh,
}

pub const PIXEL_INPUT_CONTROL_COUNT: usize = 32;

/// Decoded `SPI_PS_INPUT_CNTL_n` state for one logical pixel-shader input.
///
/// Pixel shaders name logical attributes through `V_INTERP`, while this register selects the
/// physical vertex-export parameter which supplies that attribute. A missing semantic can instead
/// request one of four architectural constants. The bit layout matches RDNA2/Prospero:
/// `OFFSET[4:0]`, `USE_DEFAULT[5]`, `DEFAULT_VALUE[9:8]`, and `FLAT_SHADE[10]`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PixelInputControl {
    pub input_offset: u8,
    pub use_default: bool,
    pub default_value: u8,
    pub flat_shade: bool,
}

impl PixelInputControl {
    const ZERO: Self = Self {
        input_offset: 0,
        use_default: false,
        default_value: 0,
        flat_shade: false,
    };

    #[must_use]
    pub const fn identity(index: u32) -> Self {
        Self {
            input_offset: (index & 0x1f) as u8,
            ..Self::ZERO
        }
    }

    #[must_use]
    pub const fn from_register(value: u32) -> Self {
        Self {
            input_offset: (value & 0x1f) as u8,
            use_default: value & (1 << 5) != 0,
            default_value: ((value >> 8) & 0x3) as u8,
            flat_shade: value & (1 << 10) != 0,
        }
    }

    #[must_use]
    pub const fn register_value(self) -> u32 {
        self.input_offset as u32
            | ((self.use_default as u32) << 5)
            | ((self.default_value as u32) << 8)
            | ((self.flat_shade as u32) << 10)
    }

    /// `USE_DEFAULT` and `FLAT_SHADE` share a hardware mode. Matching the GCN runtime contract,
    /// the constant table is selected only when defaulting is enabled without flat shading.
    #[must_use]
    pub const fn uses_default(self) -> bool {
        self.use_default && !self.flat_shade
    }
}

#[must_use]
pub const fn identity_pixel_input_controls() -> [PixelInputControl; PIXEL_INPUT_CONTROL_COUNT] {
    let mut controls = [PixelInputControl::ZERO; PIXEL_INPUT_CONTROL_COUNT];
    let mut index = 0;
    while index < PIXEL_INPUT_CONTROL_COUNT {
        controls[index] = PixelInputControl::identity(index as u32);
        index += 1;
    }
    controls
}

/// Raw guest NGG geometry-stage registers used to size a Metal mesh pipeline (workstream Q5).
///
/// These are the unmodified guest register dwords. The derived accessors apply the exact bit
/// layouts the SDK's register structs encode (`sdk/target/include/agc/registerstructs.h`), so a
/// caller — the `remu-gpu-metal` context/state decoder (co-designed with B2) — can hand REmu the
/// values it already reads from the guest command stream instead of re-deriving the arithmetic.
///
/// The dynamic `M0` `GS_ALLOC_REQ` allocation (vertices `[10:0]`, primitives `[22:12]`) is a
/// per-launch value the translated mesh program already reads at runtime (see the `S_SENDMSG`
/// message-9 lowering in `compiler/driver.rs`); it is therefore not a compile-time field here.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NggRegisters {
    /// `SPI_SHADER_PGM_RSRC1_*` for the merged NGG stage. Only the low `VGPRS` field is consumed.
    pub pgm_rsrc1: u32,
    /// `SPI_SHADER_PGM_RSRC2_*` / `ShShaderResource2Gs` for the merged NGG stage. Supplies LDS.
    pub pgm_rsrc2: u32,
    /// `VGT_GS_MAX_VERT_OUT`: the geometry stage's maximum output vertex count.
    pub gs_max_vert_out: u32,
    /// `GE_MAX_OUTPUT_PER_SUBGROUP`: the per-subgroup output-vertex cap after amplification.
    pub ge_max_output_per_subgroup: u32,
    /// `VGT_ESGS_RING_ITEMSIZE`: dwords reserved for one ES vertex in linked ES/GS LDS.
    /// GFX9-11 hardware scales the packed GS input offsets by this value.
    pub esgs_ring_itemsize: u32,
    /// `VGT_SHADER_STAGES_EN.PRIMGEN_PASSTHRU_EN`: selects the packed-primitive VGPR0 ABI.
    /// When false, GFX10 supplies three packed 16-bit vertex offsets in VGPR0/VGPR1 and the
    /// primitive ID in VGPR2 instead.
    pub primgen_passthrough: bool,
}

impl NggRegisters {
    /// Allocated 32-wide vector registers, from `PGM_RSRC1.VGPRS`.
    ///
    /// SDK `ShShaderResource1Cs::getVgprs()` (registerstructs.h:1504-1507): `((VGPRS & 0x3f) + 1)
    /// << 3`, documented as "the number of 32-wide vector general purpose registers. Divide by two
    /// for 64-wide wavefronts." (registerstructs.h:1500). The default field value is `0x8`
    /// (registerstructs.h:1427).
    #[must_use]
    pub const fn allocated_vgprs_wave32(self) -> u32 {
        ((self.pgm_rsrc1 & 0x3f) + 1) << 3
    }

    /// Guest LDS reservation in bytes for one NGG threadgroup, from `PGM_RSRC2.LDS_SIZE`.
    ///
    /// SDK `ShShaderResource2Gs::getLdsSize()` (registerstructs.h:12154-12157):
    /// `((m_regs[0].m_value & 0x07f80000u) >> 12)`. That getter returns a dword count: the
    /// 8-bit field in bits `[26:19]` is scaled by the GFX10 *encode* granule of 128 dwords
    /// (512 bytes). Hardware then allocates in 1024-byte granules on GFX10.3 (Mesa
    /// `ac_shader_get_lds_alloc_granularity`), so a shader may legally address up to the
    /// 1-KiB-aligned size; sizing Metal threadgroup memory below that guards out valid
    /// accesses in the rounded tail (field `1` → 1 KiB; field `0x11` → 9,216 bytes).
    #[must_use]
    pub const fn lds_size_bytes(self) -> u32 {
        let encoded_bytes = ((self.pgm_rsrc2 & 0x07f8_0000) >> 12) * 4;
        (encoded_bytes + 1023) & !1023
    }

    /// Maximum geometry output vertices, from `VGT_GS_MAX_VERT_OUT`.
    ///
    /// SDK `CxGsMaxVertOut::getVertices()` (registerstructs.h:13165-13168): the 11-bit field
    /// `value & 0x7ff`.
    #[must_use]
    pub const fn gs_max_output_vertices(self) -> u32 {
        self.gs_max_vert_out & 0x7ff
    }

    /// Maximum per-subgroup output vertices after amplification, from
    /// `GE_MAX_OUTPUT_PER_SUBGROUP`.
    ///
    /// SDK `CxGeMaxOutputPerSubgroup::getMaxVertsPerSubgroup()` (registerstructs.h:6241-6246): the
    /// 10-bit field `value & 0x3ff`.
    #[must_use]
    pub const fn max_verts_per_subgroup(self) -> u32 {
        self.ge_max_output_per_subgroup & 0x3ff
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalCompileOptions {
    pub stage: MetalShaderStage,
    pub required_vertex_output_count: u32,
    /// Raw `PA_CL_VS_OUT_CNTL` state used to map `EXP POS1..3` onto clip/cull distances and
    /// miscellaneous fixed-function vertex outputs.
    ///
    /// GFX10 packs only the enabled output vectors, so the same POS target can denote the misc
    /// vector or either clip/cull-distance vector depending on this register. Keeping the raw
    /// value in the compile options makes that mapping part of the generated Metal interface.
    pub vertex_output_control: u32,
    /// Raw `PA_CL_VTE_CNTL` state used to convert the guest position export into Metal clip
    /// coordinates.
    ///
    /// `VTX_W0_FMT` says whether the exported W is W0 (set) or 1/W0 (clear); it does not force
    /// W to one. `VTX_XY_FMT` and `VTX_Z_FMT` say whether those components were already
    /// multiplied by 1/W0. Metal always performs the perspective divide itself, so translated
    /// shaders must undo only the guest's pre-divided components and preserve the real W0.
    pub viewport_transform_control: u32,
    /// Number of active color/depth target layers. A guest misc position export selects the
    /// layer; procedural volume draws fall back to their instance index.
    pub render_target_array_length: u32,
    pub pixel_outputs: Vec<PixelOutputBinding>,
    /// `DB_SHADER_CONTROL.Z_EXPORT_ENABLE`: the pixel shader's MRTZ component zero replaces the
    /// fixed-function fragment depth.
    ///
    /// This is pipeline state rather than shader-code state on RDNA2, so callers must include it
    /// in their pipeline key before enabling it here.
    pub pixel_depth_export: bool,
    pub pixel_input_enable: u32,
    pub pixel_input_address: u32,
    /// Logical-to-physical fragment input mapping from `SPI_PS_INPUT_CNTL_0..31`.
    ///
    /// Constructors initialize this to an identity mapping so existing callers retain their
    /// previous interface until they provide live context-register state.
    pub pixel_input_controls: [PixelInputControl; PIXEL_INPUT_CONTROL_COUNT],
    /// Diagnostic-only overrides used to isolate translated geometry from fixed-function state.
    /// Production callers leave all overrides disabled.
    pub diagnostic_force_visible_clip_depth: bool,
    pub diagnostic_force_vertex_pattern: bool,
    pub diagnostic_force_vertex_fetch_id: bool,
    pub diagnostic_capture_vertex_position: bool,
    pub diagnostic_capture_pixel_output: bool,
    pub diagnostic_force_pixel_color: bool,
    /// Maximum vertex and primitive records exposed by a Metal mesh shader.
    ///
    /// These are guest NGG subgroup output capacities, not the number of host threads in the
    /// mesh threadgroup. They are zero for non-mesh stages.
    pub mesh_max_vertex_count: u32,
    pub mesh_max_primitive_count: u32,
    /// Total Metal threads used to execute one NGG subgroup. Guest wave32 numbering is derived
    /// from this value, so a 96-thread subgroup executes three architectural waves.
    pub mesh_threadgroup_size: u32,
    /// Whether GFX10 primitive generation uses its packed passthrough input-VGPR ABI.
    pub ngg_primgen_passthrough: bool,
    /// Dword stride applied to packed GFX9-11 GS input offsets for a linked ES/GS program.
    /// Zero means the register was not proven and is valid only for a no-GS NGG stage.
    pub ngg_esgs_ring_itemsize: u32,
    /// Bytes of guest LDS allocated to each compute workgroup.
    ///
    /// A non-zero, dword-aligned value is required only when the program contains an LDS memory
    /// operation. `DS_SWIZZLE_B32` does not allocate or access LDS and therefore does not require
    /// this field. Apple GPUs expose at most 32 KiB of threadgroup memory to this backend; larger
    /// guest allocations are rejected instead of being silently wrapped or spilled with different
    /// synchronization semantics.
    pub lds_size_bytes: u32,
    /// Total threads in one compute workgroup.
    ///
    /// A wave64 compute workgroup containing at most one architectural wave can use Metal's
    /// whole-threadgroup barrier to make LDS writes visible between its two SIMD32 halves. Larger
    /// workgroups cannot use that lowering at wave-local reconvergence points because independent
    /// guest waves may reach them at different times. Non-compute stages leave this at zero.
    pub compute_threadgroup_size: u32,
    /// Lane-private scratch allocation in bytes for one shader invocation.
    ///
    /// Compute callers derive this from `COMPUTE_TMPRING_SIZE.WAVESIZE`; graphics callers from
    /// `SPI_TMPRING_SIZE.WAVESIZE` (`CxGraphicsScratchRingSize`, one temp ring for the whole
    /// graphics pipe). Both registers count 1 KiB per architectural wave, so the per-lane value is
    /// that quotient over the stage's wave width. A shader containing a SCRATCH instruction is
    /// rejected when this is zero instead of being redirected into guest global memory.
    pub scratch_size_bytes: u32,
    /// Architectural lanes per guest wave. Apple executes SIMD32; wave64 shaders whose cross-lane
    /// operations are split-safe use this to place ballot masks in the correct half.
    ///
    /// Only threadgroup stages (compute, mesh) can host a split 64-lane wave. Graphics callers
    /// therefore leave this at 32 and must refuse a wave64 program whose result depends on the
    /// wave width rather than lowering half of it.
    pub wave_lane_count: u32,
    /// Number of vector registers allocated by the guest stage's `PGM_RSRC1` state.
    ///
    /// Relative vector moves address `VGPR[base + M0]`. Metal does not allow the backend's
    /// scalarized per-register locals to retain a dynamic subscript, so lowering expands the
    /// selection only across this architectural allocation.
    pub allocated_vector_register_count: u32,
    /// Set when [`MetalCompileOptions::mesh_from_ngg_registers`] built these options from real
    /// guest NGG register dwords (workstream Q5). J3 peephole deletions require this proof before
    /// they will lower NGG glyph sequences; without it those patterns fail loudly instead of
    /// applying a title-specific heuristic.
    pub ngg_registers_proven: bool,
    /// Use the original fixed-wave64 passthrough allocation for a no-API-GS NGG stage.
    ///
    /// `VGT_GS_MAX_VERT_OUT == 0` leaves the dynamic `GS_ALLOC_REQ` payload unproven for these
    /// stages. The presenter enables this compatibility ABI only for that class; linked geometry
    /// shaders retain their register-sized output allocation.
    pub ngg_fixed_allocation_fallback: bool,
    /// Lower a wave64 threadgroup program by carrying two guest lanes per Metal thread on one
    /// SIMD32 group, instead of splitting the wave across two SIMD32 groups.
    ///
    /// Under this mapping guest wave scope equals Metal simdgroup scope, so 64-bit EXEC/VCC and
    /// every mask SGPR hold exact architectural values and wave-internal ordering needs only
    /// `simdgroup_barrier`. See docs/design/2026-08-13-wave64-lane-doubling.md.
    pub wave64_lane_doubled: bool,
    /// Resolve dynamically-indexed image descriptors through runtime heap texture tables
    /// instead of the evaluation snapshot (docs/design/2026-08-13-runtime-image-descriptor-
    /// fetch.md). Off by default until the encoder stage binds the tables.
    pub runtime_heap_tables: bool,
}

impl MetalCompileOptions {
    #[must_use]
    pub const fn vertex(required_vertex_output_count: u32) -> Self {
        Self {
            stage: MetalShaderStage::Vertex,
            required_vertex_output_count,
            vertex_output_control: 0,
            // Canonical clip-space position: X/Y/Z are not pre-divided and W contains W0.
            viewport_transform_control: 1 << 10,
            render_target_array_length: 1,
            pixel_outputs: Vec::new(),
            pixel_depth_export: false,
            pixel_input_enable: 0,
            pixel_input_address: 0,
            pixel_input_controls: identity_pixel_input_controls(),
            diagnostic_force_visible_clip_depth: false,
            diagnostic_force_vertex_pattern: false,
            diagnostic_force_vertex_fetch_id: false,
            diagnostic_capture_vertex_position: false,
            diagnostic_capture_pixel_output: false,
            diagnostic_force_pixel_color: false,
            mesh_max_vertex_count: 0,
            mesh_max_primitive_count: 0,
            mesh_threadgroup_size: 0,
            ngg_primgen_passthrough: false,
            ngg_esgs_ring_itemsize: 0,
            lds_size_bytes: 0,
            compute_threadgroup_size: 0,
            scratch_size_bytes: 0,
            wave_lane_count: 32,
            allocated_vector_register_count: VECTOR_REGISTER_COUNT,
            ngg_registers_proven: false,
            ngg_fixed_allocation_fallback: false,
            wave64_lane_doubled: false,
            runtime_heap_tables: false,
        }
    }

    #[must_use]
    pub fn pixel(
        pixel_outputs: Vec<PixelOutputBinding>,
        pixel_input_enable: u32,
        pixel_input_address: u32,
    ) -> Self {
        Self {
            stage: MetalShaderStage::Pixel,
            required_vertex_output_count: 0,
            vertex_output_control: 0,
            viewport_transform_control: 1 << 10,
            render_target_array_length: 1,
            pixel_outputs,
            pixel_depth_export: false,
            pixel_input_enable,
            pixel_input_address,
            pixel_input_controls: identity_pixel_input_controls(),
            diagnostic_force_visible_clip_depth: false,
            diagnostic_force_vertex_pattern: false,
            diagnostic_force_vertex_fetch_id: false,
            diagnostic_capture_vertex_position: false,
            diagnostic_capture_pixel_output: false,
            diagnostic_force_pixel_color: false,
            mesh_max_vertex_count: 0,
            mesh_max_primitive_count: 0,
            mesh_threadgroup_size: 0,
            ngg_primgen_passthrough: false,
            ngg_esgs_ring_itemsize: 0,
            lds_size_bytes: 0,
            compute_threadgroup_size: 0,
            scratch_size_bytes: 0,
            wave_lane_count: 32,
            allocated_vector_register_count: VECTOR_REGISTER_COUNT,
            ngg_registers_proven: false,
            ngg_fixed_allocation_fallback: false,
            wave64_lane_doubled: false,
            runtime_heap_tables: false,
        }
    }

    #[must_use]
    pub const fn compute(lds_size_bytes: u32) -> Self {
        Self {
            stage: MetalShaderStage::Compute,
            required_vertex_output_count: 0,
            vertex_output_control: 0,
            viewport_transform_control: 1 << 10,
            render_target_array_length: 1,
            pixel_outputs: Vec::new(),
            pixel_depth_export: false,
            pixel_input_enable: 0,
            pixel_input_address: 0,
            pixel_input_controls: identity_pixel_input_controls(),
            diagnostic_force_visible_clip_depth: false,
            diagnostic_force_vertex_pattern: false,
            diagnostic_force_vertex_fetch_id: false,
            diagnostic_capture_vertex_position: false,
            diagnostic_capture_pixel_output: false,
            diagnostic_force_pixel_color: false,
            mesh_max_vertex_count: 0,
            mesh_max_primitive_count: 0,
            mesh_threadgroup_size: 0,
            ngg_primgen_passthrough: false,
            ngg_esgs_ring_itemsize: 0,
            lds_size_bytes,
            compute_threadgroup_size: 0,
            scratch_size_bytes: 0,
            wave_lane_count: 32,
            allocated_vector_register_count: VECTOR_REGISTER_COUNT,
            ngg_registers_proven: false,
            ngg_fixed_allocation_fallback: false,
            wave64_lane_doubled: false,
            runtime_heap_tables: false,
        }
    }

    #[must_use]
    pub const fn compute_wave(lds_size_bytes: u32, wave_lane_count: u32) -> Self {
        Self {
            stage: MetalShaderStage::Compute,
            required_vertex_output_count: 0,
            vertex_output_control: 0,
            viewport_transform_control: 1 << 10,
            render_target_array_length: 1,
            pixel_outputs: Vec::new(),
            pixel_depth_export: false,
            pixel_input_enable: 0,
            pixel_input_address: 0,
            pixel_input_controls: identity_pixel_input_controls(),
            diagnostic_force_visible_clip_depth: false,
            diagnostic_force_vertex_pattern: false,
            diagnostic_force_vertex_fetch_id: false,
            diagnostic_capture_vertex_position: false,
            diagnostic_capture_pixel_output: false,
            diagnostic_force_pixel_color: false,
            mesh_max_vertex_count: 0,
            mesh_max_primitive_count: 0,
            mesh_threadgroup_size: 0,
            ngg_primgen_passthrough: false,
            ngg_esgs_ring_itemsize: 0,
            lds_size_bytes,
            compute_threadgroup_size: 0,
            scratch_size_bytes: 0,
            wave_lane_count,
            allocated_vector_register_count: VECTOR_REGISTER_COUNT,
            ngg_registers_proven: false,
            ngg_fixed_allocation_fallback: false,
            wave64_lane_doubled: false,
            runtime_heap_tables: false,
        }
    }

    /// Builds options for one GFX10 NGG mesh workgroup, defaulting to wave32 lane IDs.
    ///
    /// The presenter supplies the register-derived subgroup width. Guest registers independently
    /// determine how many input lanes are populated and how many vertex/primitive output records
    /// Metal exposes. Merged NGG export shaders on Prospero recover their subgroup thread index
    /// as `s3[27:24] * wave_size + v_mbcnt`, so the presenter raises `wave_lane_count` to 64 for
    /// exports that manipulate the 64-bit EXEC pair (`S_*_SAVEEXEC_B64` rather than
    /// `S_AND_SAVEEXEC_B32`) and are safe to split across two SIMD32 halves.
    #[must_use]
    pub const fn mesh(
        required_vertex_output_count: u32,
        lds_size_bytes: u32,
        mesh_max_vertex_count: u32,
        mesh_max_primitive_count: u32,
        mesh_threadgroup_size: u32,
    ) -> Self {
        Self {
            stage: MetalShaderStage::Mesh,
            required_vertex_output_count,
            vertex_output_control: 0,
            viewport_transform_control: 1 << 10,
            render_target_array_length: 1,
            pixel_outputs: Vec::new(),
            pixel_depth_export: false,
            pixel_input_enable: 0,
            pixel_input_address: 0,
            pixel_input_controls: identity_pixel_input_controls(),
            diagnostic_force_visible_clip_depth: false,
            diagnostic_force_vertex_pattern: false,
            diagnostic_force_vertex_fetch_id: false,
            diagnostic_capture_vertex_position: false,
            diagnostic_capture_pixel_output: false,
            diagnostic_force_pixel_color: false,
            mesh_max_vertex_count,
            mesh_max_primitive_count,
            mesh_threadgroup_size,
            ngg_primgen_passthrough: false,
            ngg_esgs_ring_itemsize: 0,
            lds_size_bytes,
            compute_threadgroup_size: 0,
            scratch_size_bytes: 0,
            wave_lane_count: 32,
            allocated_vector_register_count: VECTOR_REGISTER_COUNT,
            ngg_registers_proven: false,
            ngg_fixed_allocation_fallback: false,
            wave64_lane_doubled: false,
            runtime_heap_tables: false,
        }
    }

    /// Builds mesh options with the vertex/primitive capacities and VGPR allocation derived from
    /// real guest NGG registers (workstream Q5), rather than caller-supplied estimates.
    ///
    /// The Metal-side maxima are `1..=256` (validated in `compiler/driver.rs`), while the guest
    /// fields are wider (11-bit vertices, 6-bit VGPR granule up to 512), so each derived quantity
    /// is clamped into the Metal window. Sizing the vertex records by the per-subgroup cap
    /// (`GE_MAX_OUTPUT_PER_SUBGROUP`) matches the amplified output one Metal threadgroup exposes;
    /// a stage that never programmed that register falls back to `VGT_GS_MAX_VERT_OUT`.
    ///
    /// Primitive capacity is bounded by the vertex capacity: the SDK register structs surfaced in
    /// this payload expose a per-subgroup *vertex* cap but no per-subgroup *primitive* cap, and a
    /// triangle-output subgroup can emit at most as many primitives as vertices. Over-provisioning
    /// the Metal primitive records within `1..=256` is safe; it never drops geometry.
    #[must_use]
    pub fn mesh_from_ngg_registers(
        required_vertex_output_count: u32,
        mesh_threadgroup_size: u32,
        registers: NggRegisters,
    ) -> Self {
        let subgroup_verts = registers.max_verts_per_subgroup();
        let vertex_capacity = if subgroup_verts != 0 {
            subgroup_verts
        } else if registers.gs_max_output_vertices() != 0 {
            registers.gs_max_output_vertices()
        } else {
            // A no-GS pipeline leaves VGT_GS_MAX_VERT_OUT inactive; with
            // GE_MAX_OUTPUT_PER_SUBGROUP also zero the natural capacity is one vertex per
            // input thread. Collapsing to 1 would silently drop every vertex slot above 0.
            mesh_threadgroup_size
        }
        .clamp(1, 256);
        let allocated_vector_register_count = registers
            .allocated_vgprs_wave32()
            .clamp(1, VECTOR_REGISTER_COUNT);
        Self {
            allocated_vector_register_count,
            ngg_registers_proven: true,
            ngg_primgen_passthrough: registers.primgen_passthrough,
            ngg_esgs_ring_itemsize: registers.esgs_ring_itemsize & 0x7fff,
            ..Self::mesh(
                required_vertex_output_count,
                registers.lds_size_bytes(),
                vertex_capacity,
                vertex_capacity,
                mesh_threadgroup_size,
            )
        }
    }

    /// Selects the known-good fixed-wave64 ABI for a no-API-GS passthrough stage whose dynamic
    /// allocation payload is not architecturally proven by `VGT_GS_MAX_VERT_OUT`.
    pub fn enable_fixed_ngg_allocation_fallback(&mut self) {
        debug_assert_eq!(self.stage, MetalShaderStage::Mesh);
        self.mesh_max_vertex_count = 64;
        self.mesh_max_primitive_count = 64;
        self.mesh_threadgroup_size = 64;
        self.ngg_fixed_allocation_fallback = true;
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalBindingLayout {
    pub global_buffer_count: u32,
    /// Metal buffer slot containing an argument buffer of guest global-memory pointers.
    ///
    /// Direct stage bindings are cheaper for ordinary shaders. Large Unreal shaders can expose
    /// more than Metal's 31 per-stage buffer slots, so those shaders place every guest pointer in
    /// one tier-2 argument buffer instead.
    pub global_argument_buffer_index: Option<u32>,
    pub initial_scalar_buffer_index: u32,
    pub sampler_argument_buffer_index: Option<u32>,
    pub texture_count: u32,
    pub sampler_count: u32,
    /// Dynamic threadgroup-memory length that the compute encoder must bind at index zero.
    pub threadgroup_memory_bytes: u32,
    /// Runtime heap texture tables: (global-memory binding index of the descriptor heap,
    /// Metal buffer slot expecting the argument buffer of 512 texture handles).
    pub heap_texture_tables: Vec<(u32, u32)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalShader {
    pub source: String,
    pub entry_point: String,
    pub stage: MetalShaderStage,
    pub bindings: MetalBindingLayout,
    pub vertex_inputs: Vec<VertexInputBinding>,
    pub interpolated_attribute_count: u32,
    /// Architectural register values which remain as independent MSL locals after lowering.
    ///
    /// The old backend forced all 384 guest registers into two addressable arrays. Keeping the
    /// values independent lets Metal's SSA and liveness passes allocate only the values which are
    /// actually referenced by the translated program.
    pub register_footprint: MetalRegisterFootprint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetalRegisterFootprint {
    pub scalar_values: u32,
    pub vector_values: u32,
    /// True when all reachable control flow was statically resolved and no PC dispatcher was
    /// emitted.
    pub direct_control_flow: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum MetalCompileError {
    #[error("shader program is empty")]
    EmptyProgram,
    #[error("invalid Metal shader interface: {0}")]
    InvalidInterface(String),
    #[error("pc 0x{pc:X} {opcode}: {message}")]
    Instruction {
        pc: u32,
        opcode: Opcode,
        message: String,
    },
}

/// Lowers one fully decoded and evaluated Gen5 shader to MSL.
///
/// # Errors
///
/// Returns an error for malformed operands, unresolved evaluated resources, invalid stage IO, or
/// any opcode/control mode that has not been implemented with complete semantics.
pub fn compile_msl(
    program: &Program,
    evaluation: &ShaderEvaluation,
    options: &MetalCompileOptions,
) -> Result<MetalShader, MetalCompileError> {
    Compiler::new(program, evaluation, options)?.compile()
}

/// Builds the fixed-function equivalent of a procedural Gen5 rectangle export stage.
///
/// The common AGC fullscreen stage emits primitive data and position, but no conventional
/// `EXP PARAM` instructions. On Prospero those rectangle parameters are generated as part of the
/// primitive path. Metal exposes no equivalent stage, so emit a single oversized triangle that
/// covers clip space `[-1, 1]^2` (the same covering used by host presentation) and synthesize the
/// screen-space parameters directly. Drawing one triangle avoids guest triangle-strip / cull
/// combinations that otherwise survive as a diagonal half-screen. The paired pixel shader
/// determines which physical parameters carry rectangle coordinates; every other missing
/// parameter receives the architectural unit modulation value. Metal's render-target rows run
/// opposite to the guest rectangle's interpolated Y coordinate, so coordinate parameters reverse
/// Y while the covering triangle keeps a consistent winding.
#[must_use]
pub fn compile_procedural_rectangle_vertex(
    interpolated_attribute_count: u32,
    coordinate_parameter_mask: u32,
) -> MetalShader {
    let mut source = String::from(
        "#include <metal_stdlib>\nusing namespace metal;\n\n\
         struct RemuVsOut {\n\
         \x20   float4 position [[position]];\n",
    );
    for location in 0..interpolated_attribute_count {
        let _ = writeln!(
            source,
            "    float4 param{location} [[user(locn{location})]];"
        );
    }
    source.push_str(
        "};\n\n\
         vertex RemuVsOut remu_main(uint remu_vertex_id [[vertex_id]]) {\n\
         \x20   RemuVsOut remu_out = {};\n\
         \x20   constexpr float2 remu_positions[3] = {\n\
         \x20       float2(-1.0f, -1.0f),\n\
         \x20       float2( 3.0f, -1.0f),\n\
         \x20       float2(-1.0f,  3.0f),\n\
         \x20   };\n\
         \x20   constexpr float2 remu_uvs[3] = {\n\
         \x20       float2(0.0f, 0.0f),\n\
         \x20       float2(2.0f, 0.0f),\n\
         \x20       float2(0.0f, 2.0f),\n\
         \x20   };\n\
         \x20   uint remu_index = min(remu_vertex_id, 2u);\n\
         \x20   float2 remu_uv = remu_uvs[remu_index];\n\
         \x20   remu_out.position = float4(remu_positions[remu_index], 0.0f, 1.0f);\n",
    );
    for location in 0..interpolated_attribute_count {
        let value = if coordinate_parameter_mask & (1 << location) != 0 {
            "float4(remu_uv.x, 1.0f - remu_uv.y, 1.0f, 1.0f)"
        } else {
            "float4(1.0f)"
        };
        let _ = writeln!(source, "    remu_out.param{location} = {value};");
    }
    source.push_str("    return remu_out;\n}\n");
    MetalShader {
        source,
        entry_point: "remu_main".to_owned(),
        stage: MetalShaderStage::Vertex,
        bindings: MetalBindingLayout {
            global_buffer_count: 0,
            global_argument_buffer_index: None,
            initial_scalar_buffer_index: 0,
            sampler_argument_buffer_index: None,
            texture_count: 0,
            sampler_count: 0,
            threadgroup_memory_bytes: 0,
            heap_texture_tables: Vec::new(),
        },
        vertex_inputs: Vec::new(),
        interpolated_attribute_count,
        register_footprint: MetalRegisterFootprint {
            scalar_values: 0,
            vector_values: 0,
            direct_control_flow: true,
        },
    }
}

#[cfg(test)]
mod tests;
