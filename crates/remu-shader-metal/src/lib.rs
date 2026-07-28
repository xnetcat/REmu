// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Backend lowering from `REMu`'s Gen5 shader IR to Metal Shading Language.
//!
//! Translation is deliberately strict. An instruction is either emitted with its architectural
//! semantics or compilation fails with the instruction PC and opcode; this crate never substitutes
//! a placeholder color, drops an unknown memory access, or selects behavior by game identity.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

mod compiler;

use compiler::control_flow::{
    ControlFlowLowering, LoweringFailure, analyze_control_flow, branch_target,
    combined_lowering_error, dispatcher_basic_blocks,
};
use compiler::image::{ImageKind, image_component_type, is_storage_image, is_writable_image};
#[cfg(test)]
use compiler::memory::{buffer_oob_expression, format_buffer_element_oob_expression};
use compiler::memory::{is_format_buffer_memory, is_lds_memory_instruction, is_writable_memory};
use compiler::prelude::{
    emit_data_share_prelude, emit_div_fixup_f32_prelude, emit_format_prelude,
    emit_global_memory_prelude, emit_vop3p_prelude,
};
#[cfg(test)]
use compiler::vector::is_supported_dpp_control;
use compiler::vector::{destination_vector, vector_operand};
use remu_shader::{
    ExportControl, ImageBindingSlotMap, Instruction, InstructionControl, Opcode, OperandKind,
    PixelOutputBinding, PixelOutputKind, Program, ShaderEvaluation, VertexInputBinding,
    decode_inline_constant, image_is_comparison,
};
#[cfg(test)]
use remu_shader::{ImageBinding, ImageControl, Operand, SdwaControl, ShaderEncoding};

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalCompileOptions {
    pub stage: MetalShaderStage,
    pub required_vertex_output_count: u32,
    /// Number of active color-target layers. Gen5 volume draws route each guest instance to the
    /// corresponding layer.
    pub render_target_array_length: u32,
    pub pixel_outputs: Vec<PixelOutputBinding>,
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
    /// Bytes of guest LDS allocated to each compute workgroup.
    ///
    /// A non-zero, dword-aligned value is required only when the program contains an LDS memory
    /// operation. `DS_SWIZZLE_B32` does not allocate or access LDS and therefore does not require
    /// this field. Apple GPUs expose at most 32 KiB of threadgroup memory to this backend; larger
    /// guest allocations are rejected instead of being silently wrapped or spilled with different
    /// synchronization semantics.
    pub lds_size_bytes: u32,
    /// Architectural lanes per guest compute wave. Apple executes SIMD32; wave64 shaders whose
    /// cross-lane operations are split-safe use this to place ballot masks in the correct half.
    pub wave_lane_count: u32,
}

impl MetalCompileOptions {
    #[must_use]
    pub const fn vertex(required_vertex_output_count: u32) -> Self {
        Self {
            stage: MetalShaderStage::Vertex,
            required_vertex_output_count,
            render_target_array_length: 1,
            pixel_outputs: Vec::new(),
            pixel_input_enable: 0,
            pixel_input_address: 0,
            pixel_input_controls: identity_pixel_input_controls(),
            diagnostic_force_visible_clip_depth: false,
            diagnostic_force_vertex_pattern: false,
            diagnostic_force_vertex_fetch_id: false,
            diagnostic_capture_vertex_position: false,
            diagnostic_capture_pixel_output: false,
            diagnostic_force_pixel_color: false,
            lds_size_bytes: 0,
            wave_lane_count: 32,
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
            render_target_array_length: 1,
            pixel_outputs,
            pixel_input_enable,
            pixel_input_address,
            pixel_input_controls: identity_pixel_input_controls(),
            diagnostic_force_visible_clip_depth: false,
            diagnostic_force_vertex_pattern: false,
            diagnostic_force_vertex_fetch_id: false,
            diagnostic_capture_vertex_position: false,
            diagnostic_capture_pixel_output: false,
            diagnostic_force_pixel_color: false,
            lds_size_bytes: 0,
            wave_lane_count: 32,
        }
    }

    #[must_use]
    pub const fn compute(lds_size_bytes: u32) -> Self {
        Self {
            stage: MetalShaderStage::Compute,
            required_vertex_output_count: 0,
            render_target_array_length: 1,
            pixel_outputs: Vec::new(),
            pixel_input_enable: 0,
            pixel_input_address: 0,
            pixel_input_controls: identity_pixel_input_controls(),
            diagnostic_force_visible_clip_depth: false,
            diagnostic_force_vertex_pattern: false,
            diagnostic_force_vertex_fetch_id: false,
            diagnostic_capture_vertex_position: false,
            diagnostic_capture_pixel_output: false,
            diagnostic_force_pixel_color: false,
            lds_size_bytes,
            wave_lane_count: 32,
        }
    }

    #[must_use]
    pub const fn compute_wave(lds_size_bytes: u32, wave_lane_count: u32) -> Self {
        Self {
            stage: MetalShaderStage::Compute,
            required_vertex_output_count: 0,
            render_target_array_length: 1,
            pixel_outputs: Vec::new(),
            pixel_input_enable: 0,
            pixel_input_address: 0,
            pixel_input_controls: identity_pixel_input_controls(),
            diagnostic_force_visible_clip_depth: false,
            diagnostic_force_vertex_pattern: false,
            diagnostic_force_vertex_fetch_id: false,
            diagnostic_capture_vertex_position: false,
            diagnostic_capture_pixel_output: false,
            diagnostic_force_pixel_color: false,
            lds_size_bytes,
            wave_lane_count,
        }
    }

    /// Builds options for one wave64 GFX10 NGG mesh workgroup.
    ///
    /// The presenter supplies 64 threads. Up to 63 lanes carry triangle-list vertices so every
    /// workgroup contains 21 whole primitives; the remaining lane participates in wave/LDS
    /// operations but exports only degenerate initialization data.
    #[must_use]
    pub const fn mesh(required_vertex_output_count: u32, lds_size_bytes: u32) -> Self {
        Self {
            stage: MetalShaderStage::Mesh,
            required_vertex_output_count,
            render_target_array_length: 1,
            pixel_outputs: Vec::new(),
            pixel_input_enable: 0,
            pixel_input_address: 0,
            pixel_input_controls: identity_pixel_input_controls(),
            diagnostic_force_visible_clip_depth: false,
            diagnostic_force_vertex_pattern: false,
            diagnostic_force_vertex_fetch_id: false,
            diagnostic_capture_vertex_position: false,
            diagnostic_capture_pixel_output: false,
            diagnostic_force_pixel_color: false,
            lds_size_bytes,
            wave_lane_count: 64,
        }
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

/// Builds the fixed-function equivalent of the common fused ES/GS rectangle expansion used for
/// layered volume generation. Prospero runs the export half through LDS and transfers to a
/// geometry half with `S_SETPC`; Metal has neither graphics-stage LDS nor geometry shaders, so the
/// equivalent four-vertex strip is emitted directly and each instance is routed to one layer.
#[must_use]
pub fn compile_layered_rectangle_vertex(
    interpolated_attribute_count: u32,
    render_target_array_length: u32,
) -> MetalShader {
    assert!(render_target_array_length > 1);
    let mut source = String::from(
        "#include <metal_stdlib>\nusing namespace metal;\n\n\
         struct RemuVsOut {\n\
         \x20   float4 position [[position]];\n\
         \x20   uint render_target_array_index [[render_target_array_index]];\n",
    );
    for location in 0..interpolated_attribute_count {
        let _ = writeln!(
            source,
            "    float4 param{location} [[user(locn{location})]];"
        );
    }
    source.push_str(
        "};\n\n\
         vertex RemuVsOut remu_main(\n\
         \x20   uint remu_vertex_id [[vertex_id]],\n\
         \x20   uint remu_instance_id [[instance_id]]\n\
         ) {\n\
         \x20   RemuVsOut remu_out = {};\n\
         \x20   uint remu_corner = remu_vertex_id & 3u;\n\
         \x20   float2 remu_uv = float2(float(remu_corner & 1u), float((remu_corner >> 1u) & 1u));\n\
         \x20   remu_out.position = float4(remu_uv * 2.0f - 1.0f, 0.5f, 1.0f);\n",
    );
    let _ = writeln!(
        source,
        "    remu_out.render_target_array_index = min(remu_instance_id, {}u);",
        render_target_array_length - 1
    );
    let _ = writeln!(
        source,
        "    float remu_layer = (float(remu_instance_id) + 0.5f) / {}.0f;",
        render_target_array_length
    );
    for location in 0..interpolated_attribute_count {
        let _ = writeln!(
            source,
            "    remu_out.param{location} = float4(remu_uv, remu_layer, 1.0f);"
        );
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

/// Builds the fixed-function equivalent of a procedural Gen5 rectangle export stage.
///
/// The common AGC fullscreen stage emits primitive data and position, but no conventional
/// `EXP PARAM` instructions. On Prospero those rectangle parameters are generated as part of the
/// primitive path. Metal exposes no equivalent stage, so emit the complete four-corner strip and
/// its screen-space parameters directly. Attribute zero is the conventional modulation/tint
/// parameter when a paired UI shader consumes more than one attribute; subsequent attributes
/// carry UV and unit scale. A single attribute is treated as UV.
#[must_use]
pub fn compile_procedural_rectangle_vertex(interpolated_attribute_count: u32) -> MetalShader {
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
         \x20   uint remu_corner = remu_vertex_id & 3u;\n\
         \x20   float2 remu_uv = float2(float(remu_corner & 1u), float((remu_corner >> 1u) & 1u));\n\
         \x20   remu_out.position = float4(remu_uv * 2.0f - 1.0f, 0.0f, 1.0f);\n",
    );
    for location in 0..interpolated_attribute_count {
        let value = if interpolated_attribute_count > 1 && location == 0 {
            "float4(1.0f)"
        } else {
            "float4(remu_uv, 1.0f, 1.0f)"
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

struct Compiler<'a> {
    program: &'a Program,
    evaluation: &'a ShaderEvaluation,
    options: &'a MetalCompileOptions,
    body: String,
    next_temp: u32,
    vertex_outputs: BTreeSet<u32>,
    pixel_attributes: BTreeSet<u32>,
    pixel_interpolations: BTreeMap<u32, PixelInterpolation>,
    image_kinds: Vec<ImageKind>,
    image_slots: ImageBindingSlotMap,
    control_flow: ControlFlowLowering,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PixelInterpolation {
    CenterPerspective,
    CentroidPerspective,
    SamplePerspective,
    CenterNoPerspective,
    CentroidNoPerspective,
    SampleNoPerspective,
    Flat,
}

impl PixelInterpolation {
    const fn msl_attribute(self) -> &'static str {
        match self {
            Self::CenterPerspective => "center_perspective",
            Self::CentroidPerspective => "centroid_perspective",
            Self::SamplePerspective => "sample_perspective",
            Self::CenterNoPerspective => "center_no_perspective",
            Self::CentroidNoPerspective => "centroid_no_perspective",
            Self::SampleNoPerspective => "sample_no_perspective",
            Self::Flat => "flat",
        }
    }
}

fn interpolation_for_barycentric_vgpr(
    pixel_input_address: u32,
    register: u32,
) -> Result<PixelInterpolation, String> {
    // VGPRs are allocated in SPI_PS_INPUT_ADDR order. Each enabled interpolation mode contributes
    // its I/J pair; pull-model contributes I/W, J/W, and 1/W, and line stipple contributes one.
    // With no barycentric address bits, retain Metal's historical/default center-perspective mode
    // for API compatibility with callers that have not supplied register state yet.
    if pixel_input_address & 0xff == 0 {
        return Ok(PixelInterpolation::CenterPerspective);
    }
    let mut first_register = 0_u32;
    for (bit, count, interpolation) in [
        (0, 2, Some(PixelInterpolation::SamplePerspective)),
        (1, 2, Some(PixelInterpolation::CenterPerspective)),
        (2, 2, Some(PixelInterpolation::CentroidPerspective)),
        (3, 3, None),
        (4, 2, Some(PixelInterpolation::SampleNoPerspective)),
        (5, 2, Some(PixelInterpolation::CenterNoPerspective)),
        (6, 2, Some(PixelInterpolation::CentroidNoPerspective)),
        (7, 1, None),
    ] {
        if pixel_input_address & (1 << bit) == 0 {
            continue;
        }
        if (first_register..first_register + count).contains(&register) {
            return interpolation.ok_or_else(|| {
                format!(
                    "V_INTERP source v{register} selects pull-model or line-stipple state, which \
                     requires explicit interpolation"
                )
            });
        }
        first_register += count;
    }
    Err(format!(
        "V_INTERP source v{register} is not an enabled barycentric VGPR in \
         SPI_PS_INPUT_ADDR={pixel_input_address:#010x}"
    ))
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct RegisterReferences {
    scalar: BTreeSet<u32>,
    vector: BTreeSet<u32>,
}

impl<'a> Compiler<'a> {
    fn is_threadgroup_stage(&self) -> bool {
        matches!(
            self.options.stage,
            MetalShaderStage::Compute | MetalShaderStage::Mesh
        )
    }

    fn split_wave64(&self) -> bool {
        self.is_threadgroup_stage() && self.options.wave_lane_count == 64
    }

    fn uses_lds_memory(&self) -> bool {
        self.program
            .instructions
            .iter()
            .any(is_lds_memory_instruction)
    }

    fn graphics_addtid_write_slot_count(&self) -> usize {
        if self.is_threadgroup_stage() {
            return 0;
        }
        self.program
            .instructions
            .iter()
            .filter(|instruction| instruction.opcode.is("DsWriteAddtidB32"))
            .count()
    }

    fn graphics_addtid_write_slot(&self, pc: u32) -> Option<usize> {
        self.program
            .instructions
            .iter()
            .filter(|instruction| instruction.opcode.is("DsWriteAddtidB32"))
            .position(|instruction| instruction.pc == pc)
    }

    fn architectural_lane(&self) -> &'static str {
        if self.split_wave64() {
            "remu_wave_lane"
        } else {
            "remu_lane"
        }
    }

    fn lane_mask_predicate(&self, low: &str, high: &str) -> String {
        if self.split_wave64() {
            format!(
                "(remu_wave_half == 0u ? ((({low}) >> remu_lane) & 1u) : ((({high}) >> remu_lane) & 1u)) != 0u"
            )
        } else {
            format!("((({low}) >> remu_lane) & 1u) != 0u")
        }
    }

    fn ballot_words(&self, condition: &str) -> (String, String) {
        let ballot = Self::ballot(condition);
        if self.split_wave64() {
            (
                format!("remu_wave_half == 0u ? {ballot} : 0u"),
                format!("remu_wave_half == 0u ? 0u : {ballot}"),
            )
        } else {
            (ballot, "0u".to_owned())
        }
    }

    fn global_metadata_word(&self, binding_index: usize, word: usize) -> String {
        let scalar_words = self.evaluation.initial_scalar_registers.len();
        let index = scalar_words + binding_index * 4 + word;
        format!("remu_initial_s[{index}]")
    }

    fn global_binding_base(&self, binding_index: usize) -> String {
        let low = self.global_metadata_word(binding_index, 0);
        let high = self.global_metadata_word(binding_index, 1);
        format!("((ulong){low} | ((ulong){high} << 32))")
    }

    fn global_binding_length(&self, binding_index: usize) -> String {
        self.global_metadata_word(binding_index, 2)
    }

    fn new(
        program: &'a Program,
        evaluation: &'a ShaderEvaluation,
        options: &'a MetalCompileOptions,
    ) -> Result<Self, MetalCompileError> {
        if program.instructions.is_empty() {
            return Err(MetalCompileError::EmptyProgram);
        }
        validate_interface(options)?;
        if let Some(system_registers) = evaluation.compute_system_registers {
            if options.stage != MetalShaderStage::Compute {
                return Err(MetalCompileError::InvalidInterface(
                    "compute system SGPRs require a compute shader stage".into(),
                ));
            }
            for register in [
                system_registers.work_group_x,
                system_registers.work_group_y,
                system_registers.work_group_z,
                system_registers.thread_group_size,
            ]
            .into_iter()
            .flatten()
            {
                if register >= SCALAR_REGISTER_COUNT {
                    return Err(MetalCompileError::InvalidInterface(format!(
                        "compute system SGPR s{register} exceeds the scalar register file"
                    )));
                }
            }
        }

        let mut vertex_outputs = BTreeSet::new();
        let mut pixel_attributes = BTreeSet::new();
        let mut pixel_interpolations = BTreeMap::new();
        for instruction in &program.instructions {
            match instruction.control.as_ref() {
                Some(InstructionControl::Export(export))
                    if matches!(
                        options.stage,
                        MetalShaderStage::Vertex | MetalShaderStage::Mesh
                    ) && (32..64).contains(&export.target) =>
                {
                    vertex_outputs.insert(export.target - 32);
                }
                Some(InstructionControl::Interpolation(interpolation))
                    if options.stage == MetalShaderStage::Pixel =>
                {
                    let attribute_index =
                        usize::try_from(interpolation.attribute).map_err(|_| {
                            MetalCompileError::InvalidInterface(format!(
                                "pixel input attribute {} does not fit the host index",
                                interpolation.attribute
                            ))
                        })?;
                    let Some(input) = options.pixel_input_controls.get(attribute_index).copied()
                    else {
                        return Err(MetalCompileError::InvalidInterface(format!(
                            "pixel input attribute {} exceeds the {} SPI_PS_INPUT_CNTL registers",
                            interpolation.attribute, PIXEL_INPUT_CONTROL_COUNT
                        )));
                    };
                    if input.input_offset >= 32 {
                        return Err(MetalCompileError::InvalidInterface(format!(
                            "pixel input attribute {} maps to invalid parameter {}",
                            interpolation.attribute, input.input_offset
                        )));
                    }
                    if input.default_value > 3 {
                        return Err(MetalCompileError::InvalidInterface(format!(
                            "pixel input attribute {} uses invalid default selector {}",
                            interpolation.attribute, input.default_value
                        )));
                    }
                    pixel_attributes.insert(interpolation.attribute);
                    if input.flat_shade {
                        pixel_interpolations
                            .insert(interpolation.attribute, PixelInterpolation::Flat);
                    } else if !input.uses_default() && instruction.opcode.starts_with("VInterpP2") {
                        let source = instruction.sources.first().copied().ok_or_else(|| {
                            MetalCompileError::InvalidInterface(format!(
                                "pc 0x{:X} {} has no P2 barycentric source",
                                instruction.pc, instruction.opcode
                            ))
                        })?;
                        if source.kind != OperandKind::VectorRegister {
                            return Err(MetalCompileError::InvalidInterface(format!(
                                "pc 0x{:X} {} P2 barycentric source is not a VGPR",
                                instruction.pc, instruction.opcode
                            )));
                        }
                        let interpolation_mode = interpolation_for_barycentric_vgpr(
                            options.pixel_input_address,
                            source.value,
                        )
                        .map_err(MetalCompileError::InvalidInterface)?;
                        if let Some(previous) =
                            pixel_interpolations.insert(interpolation.attribute, interpolation_mode)
                            && previous != interpolation_mode
                        {
                            return Err(MetalCompileError::InvalidInterface(format!(
                                "pixel input attribute {} mixes {:?} and {:?} interpolation",
                                interpolation.attribute, previous, interpolation_mode
                            )));
                        }
                    }
                }
                _ => {}
            }
        }
        if matches!(
            options.stage,
            MetalShaderStage::Vertex | MetalShaderStage::Mesh
        ) {
            vertex_outputs.extend(0..options.required_vertex_output_count);
        }

        let image_slots = evaluation.image_binding_slots();
        if image_slots.texture_count > METAL_DIRECT_TEXTURE_LIMIT {
            return Err(MetalCompileError::InvalidInterface(format!(
                "{} distinct image descriptors/types/access modes exceed Metal's {} direct \
                 texture slots",
                image_slots.texture_count, METAL_DIRECT_TEXTURE_LIMIT
            )));
        }
        let mut image_kinds = Vec::with_capacity(
            usize::try_from(image_slots.texture_count)
                .map_err(|_| MetalCompileError::InvalidInterface("too many textures".into()))?,
        );
        for (binding_index, binding) in evaluation.image_bindings.iter().enumerate() {
            let texture_slot = usize::try_from(image_slots.texture_slots[binding_index])
                .map_err(|_| MetalCompileError::InvalidInterface("texture slot overflow".into()))?;
            if texture_slot == image_kinds.len() {
                let storage = is_storage_image(binding.opcode);
                let writable = is_writable_image(binding.opcode);
                let comparison = image_is_comparison(binding.opcode);
                image_kinds.push(ImageKind {
                    component: image_component_type(binding),
                    storage,
                    writable,
                    comparison,
                    dimension: binding.control.dimension,
                    texture_type: binding
                        .resource_descriptor
                        .get(3)
                        .map_or(0, |word| (word >> 28) & 0xf),
                });
            } else if texture_slot > image_kinds.len() {
                return Err(MetalCompileError::InvalidInterface(
                    "non-dense canonical texture slots".into(),
                ));
            }
        }
        let control_flow = analyze_control_flow(program)?;

        Ok(Self {
            program,
            evaluation,
            options,
            body: String::new(),
            next_temp: 0,
            vertex_outputs,
            pixel_attributes,
            pixel_interpolations,
            image_kinds,
            image_slots,
            control_flow,
        })
    }

    fn compile(mut self) -> Result<MetalShader, MetalCompileError> {
        // Lowering continues past a failed instruction so that one attempt names every opcode a
        // shader needs. A body produced after a failure is only ever discarded with the error.
        let mut failures = Vec::new();
        let mut failed_opcodes = BTreeSet::new();
        match self.control_flow.clone() {
            ControlFlowLowering::Direct(indices) => {
                for index in indices {
                    let instruction = &self.program.instructions[index];
                    if let Err(message) = self.emit_instruction(instruction, index, false)
                        && failed_opcodes.insert(instruction.opcode)
                    {
                        failures.push(LoweringFailure {
                            pc: instruction.pc,
                            opcode: instruction.opcode,
                            message,
                        });
                    }
                }
            }
            ControlFlowLowering::Dispatcher => {
                // Dispatch whole basic blocks instead of one instruction per switch iteration.
                // The architectural PC only has to be materialized at control-flow boundaries;
                // straight-line instructions inside a block execute in their original order.
                // Large Unity pixel shaders contain hundreds of instructions but only a handful
                // of branches, so this avoids hundreds of divergent switch operations per pixel
                // without changing branch, loop, or EXEC/VCC semantics.
                for block in dispatcher_basic_blocks(self.program) {
                    let first = &self.program.instructions[block.start];
                    let block_instruction_count = block.len();
                    self.line(2, &format!("case {}u: {{", first.pc));
                    let last = block.end - 1;
                    for index in block {
                        let instruction = &self.program.instructions[index];
                        if let Err(message) =
                            self.emit_instruction(instruction, index, index == last)
                            && failed_opcodes.insert(instruction.opcode)
                        {
                            failures.push(LoweringFailure {
                                pc: instruction.pc,
                                opcode: instruction.opcode,
                                message,
                            });
                        }
                    }
                    self.line(3, &format!("steps += {block_instruction_count}u;"));
                    self.line(3, "break;");
                    self.line(2, "}");
                }
            }
        }
        if let Some(error) = combined_lowering_error(failures) {
            return Err(error);
        }

        let direct_control_flow = matches!(self.control_flow, ControlFlowLowering::Direct(_));
        let (source, register_references) = self.emit_module()?;
        let global_buffer_count = u32::try_from(self.evaluation.global_memory_bindings.len())
            .map_err(|_| MetalCompileError::InvalidInterface("too many global buffers".into()))?;
        let sampler_count = self.image_slots.sampler_count;
        let global_argument_buffer_index = (global_buffer_count > 30).then_some(0);
        let initial_scalar_buffer_index = if global_argument_buffer_index.is_some() {
            1
        } else {
            global_buffer_count
        };
        let direct_sampler_bindings =
            sampler_count != 0 && initial_scalar_buffer_index.saturating_add(1) > 30;
        let threadgroup_memory_bytes = if self
            .program
            .instructions
            .iter()
            .any(is_lds_memory_instruction)
        {
            self.options.lds_size_bytes
        } else {
            0
        };
        Ok(MetalShader {
            source,
            entry_point: "remu_main".into(),
            stage: self.options.stage,
            bindings: MetalBindingLayout {
                global_buffer_count,
                global_argument_buffer_index,
                initial_scalar_buffer_index,
                sampler_argument_buffer_index: (sampler_count != 0 && !direct_sampler_bindings)
                    .then_some(initial_scalar_buffer_index + 1),
                texture_count: self.image_slots.texture_count,
                sampler_count,
                threadgroup_memory_bytes,
            },
            vertex_inputs: self.evaluation.vertex_inputs.clone(),
            interpolated_attribute_count: self
                .pixel_attributes
                .iter()
                .filter_map(|attribute| {
                    let input = self.options.pixel_input_controls[*attribute as usize];
                    (!input.uses_default()).then_some(u32::from(input.input_offset) + 1)
                })
                .max()
                .unwrap_or(0),
            register_footprint: MetalRegisterFootprint {
                scalar_values: u32::try_from(register_references.scalar.len()).map_err(|_| {
                    MetalCompileError::InvalidInterface("too many live scalar values".into())
                })?,
                vector_values: u32::try_from(register_references.vector.len()).map_err(|_| {
                    MetalCompileError::InvalidInterface("too many live vector values".into())
                })?,
                direct_control_flow,
            },
        })
    }

    fn emit_instruction(
        &mut self,
        instruction: &Instruction,
        index: usize,
        dispatched: bool,
    ) -> Result<(), String> {
        let next_pc = self.program.instructions.get(index + 1).map(|next| next.pc);
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw opcode has no lowering semantics".to_owned())?;

        match name {
            "SEndpgm" => {
                if dispatched {
                    self.line(3, "active = false;");
                }
                return Ok(());
            }
            "SBranch" => {
                if dispatched {
                    let target = branch_target(instruction)?;
                    self.line(3, &format!("pc = {target}u;"));
                }
                return Ok(());
            }
            "SCbranchScc0" | "SCbranchScc1" | "SCbranchVccz" | "SCbranchVccnz"
            | "SCbranchExecz" | "SCbranchExecnz" => {
                if !dispatched {
                    return Err("conditional branch reached direct control-flow lowering".into());
                }
                let target = branch_target(instruction)?;
                let fallthrough =
                    next_pc.ok_or_else(|| "conditional branch has no fallthrough".to_owned())?;
                let condition = match name {
                    "SCbranchScc0" => "!scc",
                    "SCbranchScc1" => "scc",
                    "SCbranchVccz" => "(s[106] | s[107]) == 0u",
                    "SCbranchVccnz" => "(s[106] | s[107]) != 0u",
                    "SCbranchExecz" => "(s[126] | s[127]) == 0u",
                    "SCbranchExecnz" => "(s[126] | s[127]) != 0u",
                    _ => unreachable!(),
                };
                self.line(
                    3,
                    &format!("pc = ({condition}) ? {target}u : {fallthrough}u;"),
                );
                return Ok(());
            }
            // These operations order or provision guest hardware. MSL executes the translated
            // operations in program order, and NGG exports are emitted directly, so they require
            // no host instruction and retain their full architectural effect.
            "SNop" | "SWaitcnt" | "SWaitcntDepctr" | "SInstPrefetch" | "STtraceData"
            | "SClause" | "SSendmsg" | "VNop" => {}
            "SBarrier" => {
                if !self.is_threadgroup_stage() {
                    return Err(
                        "workgroup barriers cannot be represented by a Metal graphics stage"
                            .to_owned(),
                    );
                }
                self.line(3, "threadgroup_barrier(mem_flags::mem_threadgroup);");
            }
            "Exp" => self.emit_export(instruction)?,
            _ if matches!(
                instruction.control,
                Some(InstructionControl::Interpolation(_))
            ) =>
            {
                self.emit_interpolation(instruction)?;
            }
            _ if matches!(instruction.control, Some(InstructionControl::Image(_))) => {
                self.emit_image(instruction)?;
            }
            _ if matches!(
                instruction.control,
                Some(InstructionControl::ScalarMemory(_))
            ) =>
            {
                self.emit_scalar_memory(instruction)?;
            }
            _ if matches!(
                instruction.control,
                Some(InstructionControl::BufferMemory(_))
            ) =>
            {
                self.emit_buffer_memory(instruction)?;
            }
            _ if matches!(
                instruction.control,
                Some(InstructionControl::GlobalMemory(_))
            ) =>
            {
                self.emit_global_memory(instruction)?;
            }
            _ if matches!(instruction.control, Some(InstructionControl::DataShare(_))) => {
                self.emit_data_share(instruction)?;
            }
            _ if instruction.opcode.starts_with("V") => self.emit_vector_alu(instruction)?,
            _ if instruction.opcode.starts_with("S") => self.emit_scalar_alu(instruction)?,
            _ => return Err("unsupported instruction family".to_owned()),
        }
        if dispatched {
            if let Some(next_pc) = next_pc {
                self.line(3, &format!("pc = {next_pc}u;"));
            } else {
                self.line(3, "active = false;");
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn emit_module(&self) -> Result<(String, RegisterReferences), MetalCompileError> {
        let global_count = self.evaluation.global_memory_bindings.len();
        let global_argument_buffer = global_count > 30;
        let initial_index = if global_argument_buffer {
            1
        } else {
            global_count
        };
        let sampler_count = usize::try_from(self.image_slots.sampler_count)
            .map_err(|_| MetalCompileError::InvalidInterface("too many samplers".into()))?;
        let direct_sampler_bindings = sampler_count != 0 && initial_index.saturating_add(1) > 30;
        if direct_sampler_bindings && sampler_count > 16 {
            return Err(MetalCompileError::InvalidInterface(format!(
                "{sampler_count} direct samplers exceed Metal's per-stage limit of 16"
            )));
        }
        if self.options.stage == MetalShaderStage::Mesh
            && !global_argument_buffer
            && (initial_index >= 29
                || (!direct_sampler_bindings && sampler_count != 0 && initial_index + 1 >= 29))
        {
            return Err(MetalCompileError::InvalidInterface(
                "mesh lowering reserves Metal buffers 29 and 30 for indices and draw geometry"
                    .into(),
            ));
        }
        let mut source = String::from(
            "#include <metal_stdlib>\nusing namespace metal;\n\n\
             inline bool remu_range_valid(uint offset, uint size, uint length) {\n\
             \x20   return offset <= length && size <= length - offset;\n\
             }\n\n\
             inline uint remu_load_u8(const device uchar* data, uint offset, uint length) {\n\
             \x20   return remu_range_valid(offset, 1u, length) ? (uint)data[offset] : 0u;\n\
             }\n\n\
             inline uint remu_load_u16(const device uchar* data, uint offset, uint length) {\n\
             \x20   if (!remu_range_valid(offset, 2u, length)) return 0u;\n\
             \x20   return (uint)data[offset] | ((uint)data[offset + 1u] << 8u);\n\
             }\n\n\
             inline uint remu_load_u32(const device uchar* data, uint offset, uint length) {\n\
             \x20   if (!remu_range_valid(offset, 4u, length)) return 0u;\n\
             \x20   if ((offset & 3u) == 0u) {\n\
             \x20       return *((const device uint*)(data + offset));\n\
             \x20   }\n\
             \x20   uint value = 0u;\n\
             \x20   for (uint i = 0u; i < 4u; ++i) {\n\
             \x20       value |= (uint)data[offset + i] << (i * 8u);\n\
             \x20   }\n\
             \x20   return value;\n}\n\n",
        );
        if self
            .program
            .instructions
            .iter()
            .any(is_format_buffer_memory)
        {
            emit_format_prelude(&mut source);
        }
        if self.program.instructions.iter().any(is_writable_memory) {
            emit_global_memory_prelude(&mut source);
        }
        if self.program.instructions.iter().any(|instruction| {
            matches!(instruction.control, Some(InstructionControl::Vop3p(_)))
                || instruction.opcode.is("VCvtPkrtzF16F32")
        }) {
            emit_vop3p_prelude(&mut source);
        }
        if self
            .program
            .instructions
            .iter()
            .any(|instruction| instruction.opcode.is("VDivFixupF32"))
        {
            emit_div_fixup_f32_prelude(&mut source);
        }
        let uses_lds_memory = self
            .program
            .instructions
            .iter()
            .any(is_lds_memory_instruction);
        if uses_lds_memory {
            emit_data_share_prelude(&mut source);
        }
        if sampler_count != 0 && !direct_sampler_bindings {
            source.push_str("struct RemuSamplers {\n");
            for slot in 0..sampler_count {
                let _ = writeln!(source, "    sampler s{slot} [[id({slot})]];");
            }
            source.push_str("};\n\n");
        }
        if global_argument_buffer {
            source.push_str("struct RemuGlobalBuffers {\n");
            for (index, binding) in self.evaluation.global_memory_bindings.iter().enumerate() {
                let qualifier = if binding.writable {
                    "device"
                } else {
                    "const device"
                };
                let _ = writeln!(source, "    {qualifier} uchar* b{index} [[id({index})]];");
            }
            source.push_str("};\n\n");
        }

        match self.options.stage {
            MetalShaderStage::Vertex | MetalShaderStage::Mesh => {
                self.emit_vertex_interface(&mut source);
            }
            MetalShaderStage::Pixel => self.emit_pixel_interface(&mut source),
            MetalShaderStage::Compute => {}
        }

        let mut parameters = Vec::new();
        match self.options.stage {
            MetalShaderStage::Vertex => {
                if !self.evaluation.vertex_inputs.is_empty() {
                    parameters.push("RemuVsIn remu_in [[stage_in]]".to_owned());
                }
            }
            MetalShaderStage::Pixel => parameters.push("RemuPsIn remu_in [[stage_in]]".to_owned()),
            MetalShaderStage::Compute | MetalShaderStage::Mesh => {}
        }
        if global_argument_buffer {
            parameters.push("constant RemuGlobalBuffers& remu_globals [[buffer(0)]]".to_owned());
        } else {
            for (index, binding) in self.evaluation.global_memory_bindings.iter().enumerate() {
                let qualifier = if binding.writable {
                    "device"
                } else {
                    "const device"
                };
                parameters.push(format!("{qualifier} uchar* b{index} [[buffer({index})]]"));
            }
        }
        parameters.push(format!(
            "constant uint* remu_initial_s [[buffer({initial_index})]]"
        ));
        for (index, kind) in self.image_kinds.iter().enumerate() {
            let access = if kind.writable {
                ", access::read_write"
            } else if kind.storage {
                ", access::read"
            } else {
                ""
            };
            let texture_kind = if kind.comparison {
                match kind.dimension {
                    3 => "depthcube",
                    _ => "depth2d",
                }
            } else if kind.storage {
                if kind.texture_type == 10 {
                    "texture3d"
                } else {
                    "texture2d"
                }
            } else {
                match kind.dimension {
                    2 => "texture3d",
                    3 => "texturecube",
                    _ => "texture2d",
                }
            };
            let component = if kind.comparison {
                "float"
            } else {
                kind.component
            };
            parameters.push(format!(
                "{texture_kind}<{component}{access}> tex{index} [[texture({index})]]"
            ));
        }
        if direct_sampler_bindings {
            for slot in 0..sampler_count {
                parameters.push(format!("sampler remu_sampler_{slot} [[sampler({slot})]]"));
            }
        } else if sampler_count != 0 {
            parameters.push(format!(
                "constant RemuSamplers& remu_samplers [[buffer({})]]",
                initial_index + 1
            ));
        }
        if self.options.stage == MetalShaderStage::Vertex
            && self.options.diagnostic_capture_vertex_position
        {
            parameters.push("device uint* remu_debug_position [[buffer(30)]]".to_owned());
        }
        if self.options.stage == MetalShaderStage::Pixel
            && self.options.diagnostic_capture_pixel_output
        {
            parameters.push("device atomic_uint* remu_debug_pixel [[buffer(30)]]".to_owned());
        }
        match self.options.stage {
            MetalShaderStage::Vertex => {
                parameters.push("uint remu_vertex_id [[vertex_id]]".to_owned());
                parameters.push("uint remu_instance_id [[instance_id]]".to_owned());
            }
            MetalShaderStage::Pixel => {}
            MetalShaderStage::Compute => {
                if uses_lds_memory {
                    parameters
                        .push("threadgroup atomic_uint* remu_lds [[threadgroup(0)]]".to_owned());
                }
                parameters
                    .push("uint3 remu_local_id [[thread_position_in_threadgroup]]".to_owned());
                parameters.push("uint3 remu_group_id [[threadgroup_position_in_grid]]".to_owned());
                parameters.push("uint remu_lane [[thread_index_in_simdgroup]]".to_owned());
                if self.split_wave64()
                    || self.program.instructions.iter().any(|instruction| {
                        matches!(
                            instruction.opcode.known_name(),
                            Some("DsWriteAddtidB32" | "DsReadAddtidB32")
                        )
                    })
                    || self
                        .evaluation
                        .compute_system_registers
                        .and_then(|registers| registers.thread_group_size)
                        .is_some_and(|register| {
                            contains_register_reference(&self.body, 's', register)
                        })
                {
                    parameters
                        .push("uint remu_thread_index [[thread_index_in_threadgroup]]".to_owned());
                    parameters.push("uint3 remu_group_size [[threads_per_threadgroup]]".to_owned());
                }
            }
            MetalShaderStage::Mesh => {
                parameters
                    .push("mesh<RemuVsOut, void, 64, 64, topology::triangle> remu_mesh".to_owned());
                parameters.push("const device uint* remu_indices [[buffer(29)]]".to_owned());
                parameters.push("constant uint* remu_draw [[buffer(30)]]".to_owned());
                parameters
                    .push("uint3 remu_local_id [[thread_position_in_threadgroup]]".to_owned());
                parameters.push("uint3 remu_group_id [[threadgroup_position_in_grid]]".to_owned());
                parameters.push("uint remu_lane [[thread_index_in_simdgroup]]".to_owned());
                parameters
                    .push("uint remu_thread_index [[thread_index_in_threadgroup]]".to_owned());
                parameters.push("uint3 remu_group_size [[threads_per_threadgroup]]".to_owned());
            }
        }
        let signature = match self.options.stage {
            MetalShaderStage::Vertex => "vertex RemuVsOut remu_main",
            MetalShaderStage::Pixel if self.options.pixel_outputs.is_empty() => {
                "fragment void remu_main"
            }
            MetalShaderStage::Pixel => "fragment RemuPsOut remu_main",
            MetalShaderStage::Compute => "kernel void remu_main",
            MetalShaderStage::Mesh => "[[mesh]] void remu_main",
        };
        let _ = writeln!(
            source,
            "{signature}(\n    {}\n) {{",
            parameters.join(",\n    ")
        );
        if global_argument_buffer {
            for (index, binding) in self.evaluation.global_memory_bindings.iter().enumerate() {
                let qualifier = if binding.writable {
                    "device"
                } else {
                    "const device"
                };
                let _ = writeln!(
                    source,
                    "    {qualifier} uchar* b{index} = remu_globals.b{index};"
                );
            }
        }
        source.push_str("    /* REMU_REGISTER_STATE */\n");
        source.push_str("    bool exec = true;\n    bool vcc = false;\n    bool scc = false;\n");
        if self.options.stage == MetalShaderStage::Mesh && uses_lds_memory {
            let _ = writeln!(
                source,
                "    threadgroup atomic_uint remu_lds_storage[{}];\n    threadgroup atomic_uint* remu_lds = remu_lds_storage;",
                self.options.lds_size_bytes / 4
            );
        }
        if self.split_wave64() {
            source.push_str(
                "    uint remu_wave_index = remu_thread_index >> 6u;\n\
                 \x20   uint remu_wave_half = (remu_thread_index >> 5u) & 1u;\n\
                 \x20   uint remu_wave_lane = remu_lane + (remu_wave_half << 5u);\n",
            );
            if self
                .program
                .instructions
                .iter()
                .any(|instruction| instruction.opcode.is("VReadfirstlaneB32"))
            {
                // A Metal threadgroup can contain at most 1,024 threads, or sixteen guest
                // wave64s. Each wave owns two half-wave records used to select the first live
                // architectural lane across both SIMD32 groups.
                source.push_str(
                    "    threadgroup uint remu_readfirstlane_values[32];\n\
                     \x20   threadgroup uint remu_readfirstlane_active[32];\n",
                );
            }
            if self
                .program
                .instructions
                .iter()
                .any(|instruction| instruction.opcode.is("VReadlaneB32"))
            {
                // V_READLANE may select either half of a guest wave64. Metal's simd_shuffle
                // cannot cross the two SIMD32 groups, so exchange every architectural lane
                // through threadgroup memory. The maximum Metal threadgroup is 1,024 threads.
                source.push_str("    threadgroup uint remu_readlane_values[1024];\n");
            }
        }
        if matches!(self.control_flow, ControlFlowLowering::Dispatcher) {
            source.push_str("    uint pc = 0u;\n    bool active = true;\n    uint steps = 0u;\n");
        }
        if matches!(
            self.options.stage,
            MetalShaderStage::Vertex | MetalShaderStage::Pixel
        ) {
            // Graphics stages cannot declare thread_index_in_simdgroup, but the exclusive prefix
            // of one is the architectural lane index. Wave operations are emitted before the
            // EXEC-guarded write, keeping every host lane converged at simdgroup intrinsics.
            source.push_str("    uint remu_lane = simd_prefix_exclusive_sum(1u);\n");
        }
        let graphics_addtid_slots = self.graphics_addtid_write_slot_count();
        if graphics_addtid_slots != 0 {
            // DS_*_ADDTID_B32 is the architectural lane-private spill form: the only implicit
            // address term is TID*4, and compiler-generated spill regions are separated by the
            // 256-byte wave footprint. Metal graphics stages have no threadgroup address space,
            // so preserve each invocation's dynamically selected M0+offset records locally.
            // One slot per static write plus a sequence number also preserves branches and loops.
            let _ = writeln!(
                source,
                "    uint remu_addtid_address[{graphics_addtid_slots}] = {{}};"
            );
            let _ = writeln!(
                source,
                "    uint remu_addtid_value[{graphics_addtid_slots}] = {{}};"
            );
            let _ = writeln!(
                source,
                "    uint remu_addtid_sequence[{graphics_addtid_slots}] = {{}};"
            );
            let _ = writeln!(
                source,
                "    bool remu_addtid_valid[{graphics_addtid_slots}] = {{}};"
            );
            source.push_str("    uint remu_addtid_clock = 0u;\n");
        }
        match self.options.stage {
            MetalShaderStage::Vertex => {
                source.push_str("    RemuVsOut remu_out = {};\n");
                if self.options.render_target_array_length > 1 {
                    let _ = writeln!(
                        source,
                        "    remu_out.render_target_array_index = min(remu_instance_id, {}u);",
                        self.options.render_target_array_length - 1
                    );
                }
                if contains_register_reference(&self.body, 'v', 5) {
                    source.push_str("    v[5] = remu_vertex_id;\n");
                }
                if contains_register_reference(&self.body, 'v', 8) {
                    source.push_str("    v[8] = remu_instance_id;\n");
                }
            }
            MetalShaderStage::Pixel => {
                if !self.options.pixel_outputs.is_empty() {
                    source.push_str("    RemuPsOut remu_out = {};\n");
                }
                self.emit_pixel_input_state(&mut source);
            }
            MetalShaderStage::Compute => {
                for (register, component) in [(0, "x"), (1, "y"), (2, "z")] {
                    if contains_register_reference(&self.body, 'v', register) {
                        let _ = writeln!(source, "    v[{register}] = remu_local_id.{component};");
                    }
                }
                self.emit_compute_system_registers(&mut source);
            }
            MetalShaderStage::Mesh => {
                source.push_str(
                    "    RemuVsOut remu_out = {};\n\
                     \x20   remu_out.position = float4(0.0f, 0.0f, 0.0f, 1.0f);\n\
                     \x20   uint remu_mesh_vertex_base = remu_group_id.x * 63u;\n\
                     \x20   uint remu_mesh_remaining = remu_draw[0] - min(remu_draw[0], remu_mesh_vertex_base);\n\
                     \x20   uint remu_mesh_vertex_count = min(remu_mesh_remaining, 63u);\n\
                     \x20   uint remu_mesh_primitive_count = remu_mesh_vertex_count / 3u;\n\
                     \x20   uint remu_mesh_primitive_vertex = remu_thread_index * 3u;\n\
                     \x20   uint remu_mesh_primitive = remu_mesh_primitive_vertex | ((remu_mesh_primitive_vertex + 1u) << 10u) | ((remu_mesh_primitive_vertex + 2u) << 20u);\n\
                     \x20   if (remu_thread_index >= remu_mesh_primitive_count) remu_mesh_primitive = 0x80000000u;\n\
                     \x20   remu_mesh.set_index(remu_thread_index * 3u + 0u, 0u);\n\
                     \x20   remu_mesh.set_index(remu_thread_index * 3u + 1u, 0u);\n\
                     \x20   remu_mesh.set_index(remu_thread_index * 3u + 2u, 0u);\n",
                );
                if contains_register_reference(&self.body, 's', 3) {
                    source.push_str(
                        "    s[3] = remu_mesh_vertex_count | (remu_mesh_primitive_count << 8u);\n",
                    );
                }
                if contains_register_reference(&self.body, 'v', 0) {
                    source.push_str("    v[0] = remu_mesh_primitive;\n");
                }
                if contains_register_reference(&self.body, 'v', 1) {
                    source.push_str("    v[1] = remu_group_id.x * 21u + remu_thread_index;\n");
                }
                if contains_register_reference(&self.body, 'v', 2) {
                    source.push_str("    v[2] = 0u;\n");
                }
                if contains_register_reference(&self.body, 'v', 5) {
                    source.push_str(
                        "    v[5] = remu_thread_index < remu_mesh_vertex_count ? remu_indices[remu_mesh_vertex_base + remu_thread_index] : 0u;\n",
                    );
                }
                if contains_register_reference(&self.body, 'v', 8) {
                    source.push_str("    v[8] = remu_draw[1];\n");
                }
            }
        }
        for register in [VCC_LO, VCC_HI] {
            if contains_register_reference(&self.body, 's', register) {
                let _ = writeln!(source, "    s[{register}] = 0u;");
            }
        }
        if contains_register_reference(&self.body, 's', EXEC_LO)
            || contains_register_reference(&self.body, 's', EXEC_HI)
        {
            let (low, high) = self.ballot_words("true");
            if contains_register_reference(&self.body, 's', EXEC_LO) {
                let _ = writeln!(source, "    s[{EXEC_LO}] = {low};");
            }
            if contains_register_reference(&self.body, 's', EXEC_HI) {
                let _ = writeln!(source, "    s[{EXEC_HI}] = {high};");
            }
        }
        match self.control_flow {
            ControlFlowLowering::Direct(_) => source.push_str(&self.body),
            ControlFlowLowering::Dispatcher => {
                source.push_str("    while (active) {\n        switch (pc) {\n");
                source.push_str(&self.body);
                source.push_str(
                    "        default: active = false; break;\n        }\n\
                     \x20       if (steps >= 16384u) active = false;\n    }\n",
                );
            }
        }
        match self.options.stage {
            MetalShaderStage::Vertex | MetalShaderStage::Pixel => {
                if self.options.stage == MetalShaderStage::Pixel {
                    if self.options.diagnostic_capture_pixel_output
                        && let Some(output) = self.options.pixel_outputs.first()
                    {
                        let diagnostic_enable_index = self
                            .evaluation
                            .initial_scalar_registers
                            .len()
                            .saturating_add(
                                self.evaluation
                                    .global_memory_bindings
                                    .len()
                                    .saturating_mul(4),
                            );
                        let field = format!("remu_out.mrt{}", output.guest_slot);
                        let _ = writeln!(
                            source,
                            "    uint2 remu_debug_pixel_xy = uint2(remu_in.frag_coord.xy);\n\
                             \x20   if (remu_initial_s[{diagnostic_enable_index}] != 0u && remu_debug_pixel_xy.x < 1920u && remu_debug_pixel_xy.y < 1080u) {{\n\
                             \x20       uint remu_debug_pixel_base = (remu_debug_pixel_xy.y * 1920u + remu_debug_pixel_xy.x) * 5u;"
                        );
                        for component in 0..4 {
                            let value = match output.kind {
                                PixelOutputKind::Float => {
                                    format!("as_type<uint>({field}[{component}])")
                                }
                                PixelOutputKind::Uint => format!("{field}[{component}]"),
                                PixelOutputKind::Sint => {
                                    format!("as_type<uint>({field}[{component}])")
                                }
                            };
                            let _ = writeln!(
                                source,
                                "        atomic_store_explicit(&remu_debug_pixel[remu_debug_pixel_base + {}u], {value}, memory_order_relaxed);",
                                component + 1
                            );
                        }
                        source.push_str(
                            "        atomic_store_explicit(&remu_debug_pixel[remu_debug_pixel_base], 1u, memory_order_relaxed);\n\
                             \x20   }\n",
                        );
                    }
                    if self.options.diagnostic_force_pixel_color {
                        for output in &self.options.pixel_outputs {
                            let value = match output.kind {
                                PixelOutputKind::Float => "float4(1.0f, 0.0f, 1.0f, 1.0f)",
                                PixelOutputKind::Uint => {
                                    "uint4(0xffffffffu, 0u, 0xffffffffu, 0xffffffffu)"
                                }
                                PixelOutputKind::Sint => {
                                    "int4(0x7fffffff, 0, 0x7fffffff, 0x7fffffff)"
                                }
                            };
                            let _ = writeln!(
                                source,
                                "    remu_out.mrt{} = {value};",
                                output.guest_slot
                            );
                        }
                    } else {
                        source.push_str("    if (!exec) discard_fragment();\n");
                    }
                } else if self.options.diagnostic_force_vertex_pattern {
                    source.push_str(
                        "    uint remu_diagnostic_vertex = remu_vertex_id % 3u;\n\
                         \x20   float2 remu_diagnostic_xy = remu_diagnostic_vertex == 0u ? float2(-1.0f, -1.0f) : (remu_diagnostic_vertex == 1u ? float2(3.0f, -1.0f) : float2(-1.0f, 3.0f));\n\
                         \x20   remu_out.position = float4(remu_diagnostic_xy, 0.5f, 1.0f);\n",
                    );
                } else if self.options.diagnostic_force_visible_clip_depth {
                    // Keep the guest X/Y projection intact while forcing a finite positive W and
                    // an in-range Metal clip-space Z. This is a diagnostic for separating broken
                    // vertex/resource evaluation from Z/W clipping; it is never enabled by
                    // default and deliberately does not alter ordinary translated shaders.
                    source.push_str(
                        "    float remu_diagnostic_w = max(fabs(remu_out.position.w), 1.0e-6f);\n\
                         \x20   remu_out.position.w = remu_diagnostic_w;\n\
                         \x20   remu_out.position.z = 0.5f * remu_diagnostic_w;\n",
                    );
                }
                if self.options.diagnostic_capture_vertex_position {
                    let diagnostic_enable_index = self
                        .evaluation
                        .initial_scalar_registers
                        .len()
                        .saturating_add(
                            self.evaluation
                                .global_memory_bindings
                                .len()
                                .saturating_mul(4),
                        );
                    let _ = writeln!(
                        source,
                        "    if (remu_initial_s[{diagnostic_enable_index}] != 0u && remu_vertex_id < 262144u) {{\n\
                         \x20       uint remu_debug_base = remu_vertex_id * 4u;\n\
                         \x20       remu_debug_position[remu_debug_base + 0u] = as_type<uint>(remu_out.position.x);\n\
                         \x20       remu_debug_position[remu_debug_base + 1u] = as_type<uint>(remu_out.position.y);\n\
                         \x20       remu_debug_position[remu_debug_base + 2u] = as_type<uint>(remu_out.position.z);\n\
                         \x20       remu_debug_position[remu_debug_base + 3u] = as_type<uint>(remu_out.position.w);\n\
                         \x20   }}"
                    );
                }
                if self.options.stage == MetalShaderStage::Vertex
                    || !self.options.pixel_outputs.is_empty()
                {
                    source.push_str("    return remu_out;\n");
                }
            }
            MetalShaderStage::Compute => {}
            MetalShaderStage::Mesh => {
                source.push_str(
                    "    remu_mesh.set_vertex(remu_thread_index, remu_out);\n\
                     \x20   if (remu_thread_index == 0u) remu_mesh.set_primitive_count(64u);\n",
                );
            }
        }
        source.push_str("}\n");
        scalarize_register_state(&source)
    }

    fn emit_compute_system_registers(&self, source: &mut String) {
        let Some(registers) = self.evaluation.compute_system_registers else {
            return;
        };
        for (register, component) in [
            (registers.work_group_x, "x"),
            (registers.work_group_y, "y"),
            (registers.work_group_z, "z"),
        ] {
            if let Some(register) = register
                && contains_register_reference(&self.body, 's', register)
            {
                let _ = writeln!(source, "    s[{register}] = remu_group_id.{component};");
            }
        }
        if let Some(register) = registers.thread_group_size
            && contains_register_reference(&self.body, 's', register)
        {
            let wave_shift = if self.options.wave_lane_count == 64 {
                6
            } else {
                5
            };
            let wave_mask = self.options.wave_lane_count - 1;
            let _ = writeln!(
                source,
                "    uint remu_wave_id = remu_thread_index >> {wave_shift}u;\n\
                 \x20   uint remu_threadgroup_threads = remu_group_size.x * remu_group_size.y * remu_group_size.z;\n\
                 \x20   uint remu_threadgroup_waves = (remu_threadgroup_threads + {wave_mask}u) >> {wave_shift}u;"
            );
            // Ordinary compute dispatches have a zero ordered-append term. The remaining fields
            // match the RDNA2 initial-wave SGPR layout: FIRST_WAVE[31], WAVE_ID[24:20], and the
            // number of architectural waves in this threadgroup in [5:0].
            let _ = writeln!(
                source,
                "    s[{register}] = (remu_wave_id == 0u ? 0x80000000u : 0u) | ((remu_wave_id & 31u) << 20u) | (remu_threadgroup_waves & 63u);"
            );
        }
    }

    fn emit_vertex_interface(&self, source: &mut String) {
        if !self.evaluation.vertex_inputs.is_empty() {
            source.push_str("struct RemuVsIn {\n");
            let mut locations = BTreeSet::new();
            for input in &self.evaluation.vertex_inputs {
                if locations.insert(input.location) {
                    let ty = if input.component_count == 1 {
                        "float".to_owned()
                    } else {
                        format!("float{}", input.component_count)
                    };
                    let _ = writeln!(
                        source,
                        "    {ty} in{} [[attribute({})]];",
                        input.location, input.location
                    );
                }
            }
            source.push_str("};\n\n");
        }
        source.push_str("struct RemuVsOut {\n    float4 position [[position]];\n");
        if self.options.render_target_array_length > 1 {
            source.push_str("    uint render_target_array_index [[render_target_array_index]];\n");
        }
        for location in &self.vertex_outputs {
            let _ = writeln!(
                source,
                "    float4 param{location} [[user(locn{location})]];"
            );
        }
        source.push_str("};\n\n");
    }

    fn emit_pixel_interface(&self, source: &mut String) {
        source.push_str("struct RemuPsIn {\n    float4 frag_coord [[position]];\n");
        if self.options.render_target_array_length > 1 {
            source.push_str("    uint render_target_array_index [[render_target_array_index]];\n");
        }
        for attribute in &self.pixel_attributes {
            let input = self.options.pixel_input_controls[*attribute as usize];
            if input.uses_default() {
                continue;
            }
            let interpolation = self
                .pixel_interpolations
                .get(attribute)
                .copied()
                .unwrap_or(PixelInterpolation::CenterPerspective);
            let _ = writeln!(
                source,
                "    float4 attr{attribute} [[user(locn{}), {}]];",
                input.input_offset,
                interpolation.msl_attribute()
            );
        }
        source.push_str("};\n\nstruct RemuPsOut {\n");
        for output in &self.options.pixel_outputs {
            let ty = match output.kind {
                PixelOutputKind::Float => "float4",
                PixelOutputKind::Uint => "uint4",
                PixelOutputKind::Sint => "int4",
            };
            let _ = writeln!(
                source,
                "    {ty} mrt{} [[color({})]];",
                output.guest_slot, output.host_location
            );
        }
        source.push_str("};\n\n");
    }

    fn emit_pixel_input_state(&self, source: &mut String) {
        let mut vgpr = 0_u32;
        for (bit, count) in [
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 3),
            (4, 2),
            (5, 2),
            (6, 2),
            (7, 1),
        ] {
            if self.options.pixel_input_address & (1 << bit) != 0 {
                vgpr += count;
            }
        }
        for (bit, component) in [(8, "x"), (9, "y"), (10, "z"), (11, "w")] {
            let mask = 1_u32 << bit;
            if self.options.pixel_input_address & mask != 0 {
                if self.options.pixel_input_enable & mask != 0
                    && contains_register_reference(&self.body, 'v', vgpr)
                {
                    let _ = writeln!(
                        source,
                        "    remu_v_{vgpr} = as_type<uint>(remu_in.frag_coord.{component});"
                    );
                }
                vgpr += 1;
            }
        }
        for bit in 12..=15 {
            let mask = 1_u32 << bit;
            if self.options.pixel_input_address & mask == 0 {
                continue;
            }
            if bit == 13
                && self.options.pixel_input_enable & mask != 0
                && self.options.render_target_array_length > 1
                && contains_register_reference(&self.body, 'v', vgpr)
            {
                // GFX10 ANCILLARY packs the render-target array index into bits 28:16. Metal
                // exposes the exact layer selected by the vertex-stage system value to the
                // fragment stage, so preserve the architectural VGPR layout instead of leaving
                // every volume slice at the zero-initialized layer.
                let _ = writeln!(
                    source,
                    "    remu_v_{vgpr} = remu_in.render_target_array_index << 16u;"
                );
            }
            // SPI_PS_INPUT_ADDR controls compact VGPR allocation independently of INPUT_ENA.
            // Reserve every addressed system input even while its host semantic is not enabled.
            vgpr += 1;
        }
    }

    fn emit_export(&mut self, instruction: &Instruction) -> Result<(), String> {
        let Some(InstructionControl::Export(export)) = instruction.control.as_ref() else {
            return Err("export instruction lacks export control".to_owned());
        };
        if instruction.sources.len() < 4 {
            return Err("export requires four source operands".to_owned());
        }
        match self.options.stage {
            MetalShaderStage::Vertex => self.emit_vertex_export(instruction, export),
            MetalShaderStage::Pixel => self.emit_pixel_export(instruction, export),
            MetalShaderStage::Compute => Ok(()),
            MetalShaderStage::Mesh => self.emit_mesh_export(instruction, export),
        }
    }

    fn emit_mesh_export(
        &mut self,
        instruction: &Instruction,
        export: &ExportControl,
    ) -> Result<(), String> {
        if export.target == 20 {
            let primitive = self.temp("uint", &self.raw_source(instruction, 0)?);
            let lane = self.architectural_lane();
            self.line(
                3,
                &format!("if (exec && ({primitive} & 0x80000000u) == 0u) {{"),
            );
            self.line(
                4,
                &format!("remu_mesh.set_index({lane} * 3u + 0u, ({primitive} >> 0u) & 0x1ffu);"),
            );
            self.line(
                4,
                &format!("remu_mesh.set_index({lane} * 3u + 1u, ({primitive} >> 10u) & 0x1ffu);"),
            );
            self.line(
                4,
                &format!("remu_mesh.set_index({lane} * 3u + 2u, ({primitive} >> 20u) & 0x1ffu);"),
            );
            self.line(3, "}");
            return Ok(());
        }
        self.emit_vertex_export(instruction, export)
    }

    fn emit_vertex_export(
        &mut self,
        instruction: &Instruction,
        export: &ExportControl,
    ) -> Result<(), String> {
        let field = if export.target == 12 {
            "position".to_owned()
        } else if (32..64).contains(&export.target)
            && self.vertex_outputs.contains(&(export.target - 32))
        {
            format!("param{}", export.target - 32)
        } else {
            return Ok(());
        };
        let values = self.export_float_values(instruction, export)?;
        self.line(
            3,
            &format!(
                "if (exec) remu_out.{field} = float4({}, {}, {}, {});",
                values[0], values[1], values[2], values[3]
            ),
        );
        Ok(())
    }

    fn emit_pixel_export(
        &mut self,
        instruction: &Instruction,
        export: &ExportControl,
    ) -> Result<(), String> {
        let Some(output) = self
            .options
            .pixel_outputs
            .iter()
            .find(|output| output.guest_slot == export.target)
            .copied()
        else {
            return Ok(());
        };
        let field = format!("remu_out.mrt{}", output.guest_slot);
        let mut values = Vec::with_capacity(4);
        for component in 0..4 {
            if export.enable_mask & (1 << component) == 0 {
                values.push(format!("{field}[{component}]"));
                continue;
            }
            let raw = if export.compressed {
                let register = vector_operand(instruction.sources[component >> 1])?;
                format!("(float)as_type<half2>(v[{register}])[{}]", component & 1)
            } else {
                let register = vector_operand(instruction.sources[component])?;
                format!("as_type<float>(v[{register}])")
            };
            values.push(match output.kind {
                PixelOutputKind::Float => raw,
                PixelOutputKind::Uint => format!("(uint)({raw})"),
                PixelOutputKind::Sint => format!("(int)({raw})"),
            });
        }
        let ty = match output.kind {
            PixelOutputKind::Float => "float4",
            PixelOutputKind::Uint => "uint4",
            PixelOutputKind::Sint => "int4",
        };
        self.line(
            3,
            &format!(
                "if (exec) {field} = {ty}({}, {}, {}, {});",
                values[0], values[1], values[2], values[3]
            ),
        );
        Ok(())
    }

    #[allow(clippy::unused_self)]
    fn export_float_values(
        &self,
        instruction: &Instruction,
        export: &ExportControl,
    ) -> Result<[String; 4], String> {
        let mut values = [String::new(), String::new(), String::new(), String::new()];
        for (component, value) in values.iter_mut().enumerate() {
            if export.enable_mask & (1 << component) == 0 {
                *value = String::from(if component == 3 { "1.0f" } else { "0.0f" });
            } else if export.compressed {
                let register = vector_operand(instruction.sources[component >> 1])?;
                *value = format!("(float)as_type<half2>(v[{register}])[{}]", component & 1);
            } else {
                let register = vector_operand(instruction.sources[component])?;
                *value = format!("as_type<float>(v[{register}])");
            }
        }
        Ok(values)
    }

    fn emit_interpolation(&mut self, instruction: &Instruction) -> Result<(), String> {
        let Some(InstructionControl::Interpolation(interpolation)) = instruction.control.as_ref()
        else {
            return Err("missing interpolation control".to_owned());
        };
        if self.options.stage != MetalShaderStage::Pixel
            || !self.pixel_attributes.contains(&interpolation.attribute)
        {
            return Err("interpolation is valid only for a declared pixel input".to_owned());
        }
        // Host rasterization performs the complete interpolation. P1 only forms an intermediate
        // hardware partial product; the translated destination becomes architecturally visible
        // when P2 consumes J, matching shadPS4's non-manual interpolation path.
        if instruction.opcode.starts_with("VInterpP1") {
            return Ok(());
        }
        let destination = destination_vector(instruction)?;
        let input = self.options.pixel_input_controls[interpolation.attribute as usize];
        let value = if input.uses_default() {
            let one = match input.default_value {
                0 => false,
                1 => interpolation.channel == 3,
                2 => interpolation.channel != 3,
                3 => true,
                _ => unreachable!("pixel input controls are validated during compiler creation"),
            };
            if one { "0x3F800000u" } else { "0u" }.to_owned()
        } else {
            format!(
                "as_type<uint>(remu_in.attr{}[{}])",
                interpolation.attribute, interpolation.channel
            )
        };
        self.store_vector(destination, &value);
        Ok(())
    }

    fn raw_source(&self, instruction: &Instruction, index: usize) -> Result<String, String> {
        self.raw_source_with_sdwa_modifiers(instruction, index, true)
    }

    #[allow(clippy::unused_self)]
    fn raw_source_with_sdwa_modifiers(
        &self,
        instruction: &Instruction,
        index: usize,
        apply_integer_modifiers: bool,
    ) -> Result<String, String> {
        let operand = instruction
            .sources
            .get(index)
            .copied()
            .ok_or_else(|| format!("missing source operand {index}"))?;
        let mut value = match operand.kind {
            OperandKind::ScalarRegister if operand.value < SCALAR_REGISTER_COUNT => {
                Ok(format!("s[{}]", operand.value))
            }
            OperandKind::VectorRegister if operand.value < VECTOR_REGISTER_COUNT => {
                Ok(format!("v[{}]", operand.value))
            }
            OperandKind::LiteralConstant => Ok(format_u32(operand.value)),
            OperandKind::EncodedConstant => match operand.value {
                251 => Ok(format!("((s[{VCC_LO}] | s[{VCC_HI}]) == 0u ? 1u : 0u)")),
                252 => Ok(format!("((s[{EXEC_LO}] | s[{EXEC_HI}]) == 0u ? 1u : 0u)")),
                253 => Ok("(scc ? 1u : 0u)".to_owned()),
                encoded => decode_inline_constant(encoded)
                    .map(format_u32)
                    .ok_or_else(|| format!("unsupported encoded constant {encoded}")),
            },
            _ => Err(format!(
                "register operand is outside the architectural file: {operand}"
            )),
        }?;
        if index == 0 {
            value = match instruction.control.as_ref() {
                Some(InstructionControl::Dpp(control)) => Self::apply_dpp_source(*control, &value)?,
                Some(InstructionControl::Dpp8(control)) => {
                    Self::apply_dpp8_source(*control, &value)
                }
                _ => value,
            };
        }
        let Some(InstructionControl::Sdwa(control)) = instruction.control.as_ref() else {
            return Ok(value);
        };
        let selector = match index {
            0 => control.source0_select,
            1 => control.source1_select,
            _ => 6,
        };
        value = match selector {
            0 => format!("(({value}) & 0xFFu)"),
            1 => format!("((({value}) >> 8) & 0xFFu)"),
            2 => format!("((({value}) >> 16) & 0xFFu)"),
            3 => format!("((({value}) >> 24) & 0xFFu)"),
            4 => format!("(({value}) & 0xFFFFu)"),
            5 => format!("((({value}) >> 16) & 0xFFFFu)"),
            6 => value,
            _ => return Err(format!("reserved SDWA source {index} selector {selector}")),
        };
        let sign_extend = match index {
            0 => control.source0_sign_extend,
            1 => control.source1_sign_extend,
            _ => false,
        };
        if sign_extend && selector != 6 {
            let width = if selector <= 3 { 8 } else { 16 };
            value = format!("(uint)extract_bits(as_type<int>({value}), 0u, {width}u)");
        }
        if apply_integer_modifiers {
            if control.absolute_mask & (1 << index) != 0 {
                value = format!("(uint)abs(as_type<int>({value}))");
            }
            if control.negate_mask & (1 << index) != 0 {
                value = format!("(0u - ({value}))");
            }
        }
        Ok(value)
    }

    #[allow(clippy::unused_self)]
    fn raw_source64(&self, instruction: &Instruction, index: usize) -> Result<String, String> {
        let operand = instruction
            .sources
            .get(index)
            .copied()
            .ok_or_else(|| format!("missing 64-bit source operand {index}"))?;
        match operand.kind {
            OperandKind::ScalarRegister if operand.value + 1 < SCALAR_REGISTER_COUNT => {
                Ok(format!(
                    "((ulong)s[{}] | ((ulong)s[{}] << 32))",
                    operand.value,
                    operand.value + 1
                ))
            }
            OperandKind::VectorRegister if operand.value + 1 < VECTOR_REGISTER_COUNT => {
                Ok(format!(
                    "((ulong)v[{}] | ((ulong)v[{}] << 32))",
                    operand.value,
                    operand.value + 1
                ))
            }
            OperandKind::LiteralConstant => Ok(format!("(ulong){}", format_u32(operand.value))),
            OperandKind::EncodedConstant => decode_inline_constant(operand.value)
                .map(|value| {
                    // Integer inline constants -1..-16 are architecturally sign-extended for
                    // 64-bit SALU operations. Zero extension turns `S_MOV_B64 EXEC, -1` into a
                    // half-wave mask and disables lanes 32..63 in every split wave64.
                    let value = if (193..=208).contains(&operand.value) {
                        u64::from(value) | (u64::from(u32::MAX) << 32)
                    } else {
                        u64::from(value)
                    };
                    format!("0x{value:016X}ul")
                })
                .ok_or_else(|| format!("unsupported 64-bit encoded constant {}", operand.value)),
            _ => Err(format!("invalid 64-bit source operand {operand}")),
        }
    }

    fn float_source(&self, instruction: &Instruction, index: usize) -> Result<String, String> {
        let mut value = format!(
            "as_type<float>({})",
            self.raw_source_with_sdwa_modifiers(instruction, index, false)?
        );
        let (absolute_mask, negate_mask) = match instruction.control.as_ref() {
            Some(InstructionControl::Vop3(control)) => (control.absolute_mask, control.negate_mask),
            Some(InstructionControl::Sdwa(control)) => (control.absolute_mask, control.negate_mask),
            Some(InstructionControl::Dpp(control)) => (control.absolute_mask, control.negate_mask),
            _ => (0, 0),
        };
        if absolute_mask & (1 << index) != 0 {
            value = format!("fabs({value})");
        }
        if negate_mask & (1 << index) != 0 {
            value = format!("-({value})");
        }
        Ok(value)
    }

    fn has_float_modifiers(instruction: &Instruction) -> bool {
        match instruction.control.as_ref() {
            Some(InstructionControl::Vop3(control)) => {
                control.absolute_mask != 0
                    || control.negate_mask != 0
                    || control.output_modifier != 0
                    || control.clamp
            }
            Some(InstructionControl::Sdwa(control)) => {
                control.absolute_mask != 0
                    || control.negate_mask != 0
                    || control.output_modifier != 0
                    || control.clamp
            }
            Some(InstructionControl::Dpp(control)) => {
                control.absolute_mask != 0 || control.negate_mask != 0
            }
            _ => false,
        }
    }

    fn f16_result(
        &mut self,
        instruction: &Instruction,
        destination: u32,
        expression: &str,
    ) -> String {
        let result = self.temp("uint", &self.float_result(instruction, expression));
        let packed = format!("remu_f32_to_f16_bits({result})");
        match instruction.control.as_ref() {
            Some(InstructionControl::Vop3(control)) if control.operand_select & 8 != 0 => {
                format!("(v[{destination}] & 0xFFFFu) | ({packed} << 16)")
            }
            Some(InstructionControl::Vop3(_)) => {
                format!("(v[{destination}] & 0xFFFF0000u) | {packed}")
            }
            _ => packed,
        }
    }

    fn half_float_source(&self, instruction: &Instruction, index: usize) -> Result<String, String> {
        let raw = self.raw_source_with_sdwa_modifiers(instruction, index, false)?;
        let shift = match instruction.control.as_ref() {
            Some(InstructionControl::Vop3(control))
                if control.operand_select & (1 << index) != 0 =>
            {
                16
            }
            _ => 0,
        };
        let mut value =
            format!("as_type<float>(remu_f16_to_f32_bits((({raw}) >> {shift}u) & 0xFFFFu))");
        let (absolute_mask, negate_mask) = match instruction.control.as_ref() {
            Some(InstructionControl::Vop3(control)) => (control.absolute_mask, control.negate_mask),
            Some(InstructionControl::Sdwa(control)) => (control.absolute_mask, control.negate_mask),
            Some(InstructionControl::Dpp(control)) => (control.absolute_mask, control.negate_mask),
            _ => (0, 0),
        };
        if absolute_mask & (1 << index) != 0 {
            value = format!("fabs({value})");
        }
        if negate_mask & (1 << index) != 0 {
            value = format!("-({value})");
        }
        Ok(value)
    }

    #[allow(clippy::unused_self)]
    fn float_result(&self, instruction: &Instruction, expression: &str) -> String {
        let (output_modifier, clamp) = match instruction.control.as_ref() {
            Some(InstructionControl::Vop3(control)) => (control.output_modifier, control.clamp),
            Some(InstructionControl::Sdwa(control)) => (control.output_modifier, control.clamp),
            _ => (0, false),
        };
        let mut value = match output_modifier {
            1 => format!("(({expression}) * 2.0f)"),
            2 => format!("(({expression}) * 4.0f)"),
            3 => format!("(({expression}) * 0.5f)"),
            _ => format!("({expression})"),
        };
        if clamp {
            value = format!("clamp({value}, 0.0f, 1.0f)");
        }
        format!("as_type<uint>({value})")
    }

    fn line(&mut self, indent: usize, text: &str) {
        for _ in 0..indent {
            self.body.push_str("    ");
        }
        self.body.push_str(text);
        self.body.push('\n');
    }

    fn temp(&mut self, ty: &str, expression: &str) -> String {
        let name = format!("t{}", self.next_temp);
        self.next_temp += 1;
        self.line(3, &format!("{ty} {name} = {expression};"));
        name
    }
}

fn validate_interface(options: &MetalCompileOptions) -> Result<(), MetalCompileError> {
    if options.pixel_outputs.len() > 8 {
        return Err(MetalCompileError::InvalidInterface(
            "Metal supports at most eight color outputs".into(),
        ));
    }
    let mut guest_slots = BTreeSet::new();
    let mut host_locations = BTreeSet::new();
    for output in &options.pixel_outputs {
        if output.guest_slot > 7
            || !guest_slots.insert(output.guest_slot)
            || !host_locations.insert(output.host_location)
        {
            return Err(MetalCompileError::InvalidInterface(
                "pixel output guest slots and host locations must be unique and in range".into(),
            ));
        }
    }
    if host_locations
        .iter()
        .copied()
        .ne(0..u32::try_from(host_locations.len()).unwrap_or(0))
    {
        return Err(MetalCompileError::InvalidInterface(
            "pixel output host locations must be dense from zero".into(),
        ));
    }
    Ok(())
}

fn contains_register_reference(source: &str, kind: char, register: u32) -> bool {
    register_references(source).is_ok_and(|references| match kind {
        's' => references.scalar.contains(&register),
        'v' => references.vector.contains(&register),
        _ => false,
    })
}

fn scalarize_register_state(
    source: &str,
) -> Result<(String, RegisterReferences), MetalCompileError> {
    const MARKER: &str = "    /* REMU_REGISTER_STATE */\n";
    let references = register_references(source)?;
    let mut rewritten = String::with_capacity(source.len());
    let bytes = source.as_bytes();
    let mut cursor = 0;
    while cursor < bytes.len() {
        if let Some((kind, register, end)) = parse_register_reference(bytes, cursor)? {
            let _ = write!(rewritten, "remu_{kind}_{register}");
            cursor = end;
        } else {
            rewritten.push(char::from(bytes[cursor]));
            cursor += 1;
        }
    }

    let mut declarations = String::new();
    for register in &references.scalar {
        let _ = writeln!(
            declarations,
            "    uint remu_s_{register} = remu_initial_s[{register}];"
        );
    }
    for register in &references.vector {
        let _ = writeln!(declarations, "    uint remu_v_{register} = 0u;");
    }
    if !rewritten.contains(MARKER) {
        return Err(MetalCompileError::InvalidInterface(
            "internal register-state marker was lost".into(),
        ));
    }
    Ok((rewritten.replacen(MARKER, &declarations, 1), references))
}

fn register_references(source: &str) -> Result<RegisterReferences, MetalCompileError> {
    let bytes = source.as_bytes();
    let mut references = RegisterReferences::default();
    for cursor in 0..bytes.len() {
        let Some((kind, register, _)) = parse_register_reference(bytes, cursor)? else {
            continue;
        };
        match kind {
            's' => {
                references.scalar.insert(register);
            }
            'v' => {
                references.vector.insert(register);
            }
            _ => unreachable!("register parser accepts only scalar and vector files"),
        }
    }
    Ok(references)
}

fn parse_register_reference(
    bytes: &[u8],
    start: usize,
) -> Result<Option<(char, u32, usize)>, MetalCompileError> {
    let Some(&kind @ (b's' | b'v')) = bytes.get(start) else {
        return Ok(None);
    };
    if start != 0
        && bytes
            .get(start - 1)
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
    {
        return Ok(None);
    }
    if bytes.get(start + 1) != Some(&b'[') {
        return Ok(None);
    }
    let mut cursor = start + 2;
    let digits_start = cursor;
    while bytes.get(cursor).is_some_and(u8::is_ascii_digit) {
        cursor += 1;
    }
    if cursor == digits_start || bytes.get(cursor) != Some(&b']') {
        return Err(MetalCompileError::InvalidInterface(format!(
            "dynamic {} register indexing survived lowering",
            char::from(kind)
        )));
    }
    let register = std::str::from_utf8(&bytes[digits_start..cursor])
        .ok()
        .and_then(|digits| digits.parse::<u32>().ok())
        .ok_or_else(|| {
            MetalCompileError::InvalidInterface("invalid generated register index".into())
        })?;
    let limit = if kind == b's' {
        SCALAR_REGISTER_COUNT
    } else {
        VECTOR_REGISTER_COUNT
    };
    if register >= limit {
        return Err(MetalCompileError::InvalidInterface(format!(
            "generated {} register {} exceeds architectural limit {}",
            char::from(kind),
            register,
            limit
        )));
    }
    Ok(Some((char::from(kind), register, cursor + 1)))
}

fn format_u32(value: u32) -> String {
    if value <= 9 {
        format!("{value}u")
    } else {
        format!("0x{value:X}u")
    }
}

#[cfg(test)]
mod tests;
