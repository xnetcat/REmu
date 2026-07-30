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
    PixelOutputKind, SdwaControl, ShaderEncoding,
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
    /// Number of vector registers allocated by the guest stage's `PGM_RSRC1` state.
    ///
    /// Relative vector moves address `VGPR[base + M0]`. Metal does not allow the backend's
    /// scalarized per-register locals to retain a dynamic subscript, so lowering expands the
    /// selection only across this architectural allocation.
    pub allocated_vector_register_count: u32,
}

impl MetalCompileOptions {
    #[must_use]
    pub const fn vertex(required_vertex_output_count: u32) -> Self {
        Self {
            stage: MetalShaderStage::Vertex,
            required_vertex_output_count,
            vertex_output_control: 0,
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
            lds_size_bytes: 0,
            wave_lane_count: 32,
            allocated_vector_register_count: VECTOR_REGISTER_COUNT,
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
            lds_size_bytes: 0,
            wave_lane_count: 32,
            allocated_vector_register_count: VECTOR_REGISTER_COUNT,
        }
    }

    #[must_use]
    pub const fn compute(lds_size_bytes: u32) -> Self {
        Self {
            stage: MetalShaderStage::Compute,
            required_vertex_output_count: 0,
            vertex_output_control: 0,
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
            lds_size_bytes,
            wave_lane_count: 32,
            allocated_vector_register_count: VECTOR_REGISTER_COUNT,
        }
    }

    #[must_use]
    pub const fn compute_wave(lds_size_bytes: u32, wave_lane_count: u32) -> Self {
        Self {
            stage: MetalShaderStage::Compute,
            required_vertex_output_count: 0,
            vertex_output_control: 0,
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
            lds_size_bytes,
            wave_lane_count,
            allocated_vector_register_count: VECTOR_REGISTER_COUNT,
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
            vertex_output_control: 0,
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
            lds_size_bytes,
            wave_lane_count: 64,
            allocated_vector_register_count: VECTOR_REGISTER_COUNT,
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
/// Metal's render-target rows run opposite to the guest rectangle's interpolated Y coordinate,
/// so the synthesized parameter reverses Y while preserving the strip's position and winding.
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
            "    remu_out.param{location} = float4(remu_uv.x, 1.0f - remu_uv.y, remu_layer, 1.0f);"
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
/// its screen-space parameters directly. The paired pixel shader determines which physical
/// parameters carry rectangle coordinates; every other missing parameter receives the
/// architectural unit modulation value. Metal's render-target rows run opposite to the guest
/// rectangle's interpolated Y coordinate, so coordinate parameters reverse Y while preserving
/// the strip's position and winding.
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
         \x20   uint remu_corner = remu_vertex_id & 3u;\n\
         \x20   float2 remu_uv = float2(float(remu_corner & 1u), float((remu_corner >> 1u) & 1u));\n\
         \x20   remu_out.position = float4(remu_uv * 2.0f - 1.0f, 0.0f, 1.0f);\n",
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
