// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Backend lowering from `REMu`'s Gen5 shader IR to Metal Shading Language.
//!
//! Translation is deliberately strict. An instruction is either emitted with its architectural
//! semantics or compilation fails with the instruction PC and opcode; this crate never substitutes
//! a placeholder color, drops an unknown memory access, or selects behavior by game identity.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use remu_shader::{
    DataShareControl, Dpp8Control, DppControl, ExportControl, ImageBinding, ImageControl,
    Instruction, InstructionControl, Opcode, Operand, OperandKind, PixelOutputBinding,
    PixelOutputKind, Program, SdwaControl, ShaderEncoding, ShaderEvaluation, VertexInputBinding,
    Vop3pControl, decode_inline_constant, decode_unified_format, decode_v2f16_inline_constant,
};

const SCALAR_REGISTER_COUNT: u32 = 128;
const VECTOR_REGISTER_COUNT: u32 = 256;
const VCC_LO: u32 = 106;
const VCC_HI: u32 = 107;
const EXEC_LO: u32 = 126;
const EXEC_HI: u32 = 127;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MetalShaderStage {
    Vertex,
    Pixel,
    Compute,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalCompileOptions {
    pub stage: MetalShaderStage,
    pub required_vertex_output_count: u32,
    pub pixel_outputs: Vec<PixelOutputBinding>,
    pub pixel_input_enable: u32,
    pub pixel_input_address: u32,
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
            pixel_outputs: Vec::new(),
            pixel_input_enable: 0,
            pixel_input_address: 0,
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
            pixel_outputs,
            pixel_input_enable,
            pixel_input_address,
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
            pixel_outputs: Vec::new(),
            pixel_input_enable: 0,
            pixel_input_address: 0,
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
            pixel_outputs: Vec::new(),
            pixel_input_enable: 0,
            pixel_input_address: 0,
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
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalBindingLayout {
    pub global_buffer_count: u32,
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

struct Compiler<'a> {
    program: &'a Program,
    evaluation: &'a ShaderEvaluation,
    options: &'a MetalCompileOptions,
    body: String,
    next_temp: u32,
    vertex_outputs: BTreeSet<u32>,
    pixel_attributes: BTreeSet<u32>,
    image_kinds: Vec<ImageKind>,
    control_flow: ControlFlowLowering,
}

#[derive(Clone, Debug)]
enum ControlFlowLowering {
    /// Instruction indices in their exact execution order. This includes forward or backward
    /// unconditional branches, provided the resulting path is acyclic.
    Direct(Vec<usize>),
    /// Conditional control flow or a cycle still requires the bounded architectural PC machine.
    Dispatcher,
}

/// One opcode a lowering attempt could not translate. Failures are kept per distinct opcode, so a
/// shader that uses an unimplemented opcode hundreds of times still reports it once.
#[derive(Clone, Debug, Eq, PartialEq)]
struct LoweringFailure {
    pc: u32,
    opcode: Opcode,
    message: String,
}

/// Combines every opcode one shader failed to lower into a single error. The first failure keeps
/// the shape it would have had alone, and the remaining opcodes are appended so a translation gap
/// can be closed as a batch instead of one launch per missing opcode.
fn combined_lowering_error(failures: Vec<LoweringFailure>) -> Option<MetalCompileError> {
    let mut failures = failures.into_iter();
    let first = failures.next()?;
    let remaining = failures
        .map(|failure| {
            format!(
                "pc 0x{:X} {}: {}",
                failure.pc, failure.opcode, failure.message
            )
        })
        .collect::<Vec<_>>();
    let message = if remaining.is_empty() {
        first.message
    } else {
        format!(
            "{}; the same shader also fails at {}",
            first.message,
            remaining.join(", ")
        )
    };
    Some(MetalCompileError::Instruction {
        pc: first.pc,
        opcode: first.opcode,
        message,
    })
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct RegisterReferences {
    scalar: BTreeSet<u32>,
    vector: BTreeSet<u32>,
}

#[derive(Clone, Copy)]
struct ImageKind {
    component: &'static str,
    storage: bool,
    comparison: bool,
    sampler_slot: Option<u32>,
    dimension: u32,
    texture_type: u32,
}

impl<'a> Compiler<'a> {
    fn split_wave64(&self) -> bool {
        self.options.stage == MetalShaderStage::Compute && self.options.wave_lane_count == 64
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
        for instruction in &program.instructions {
            match instruction.control.as_ref() {
                Some(InstructionControl::Export(export))
                    if options.stage == MetalShaderStage::Vertex
                        && (32..64).contains(&export.target) =>
                {
                    vertex_outputs.insert(export.target - 32);
                }
                Some(InstructionControl::Interpolation(interpolation))
                    if options.stage == MetalShaderStage::Pixel =>
                {
                    pixel_attributes.insert(interpolation.attribute);
                }
                _ => {}
            }
        }
        if options.stage == MetalShaderStage::Vertex {
            vertex_outputs.extend(0..options.required_vertex_output_count);
        }

        let mut sampler_count = 0_u32;
        let image_kinds = evaluation
            .image_bindings
            .iter()
            .map(|binding| {
                let storage = is_storage_image(binding.opcode);
                let comparison = binding.opcode.contains("SampleC");
                let sampler_slot = (!storage).then(|| {
                    let slot = sampler_count;
                    sampler_count += 1;
                    slot
                });
                ImageKind {
                    component: image_component_type(binding),
                    storage,
                    comparison,
                    sampler_slot,
                    dimension: binding.control.dimension,
                    texture_type: binding
                        .resource_descriptor
                        .get(3)
                        .map_or(0, |word| (word >> 28) & 0xf),
                }
            })
            .collect();
        let control_flow = analyze_control_flow(program)?;

        Ok(Self {
            program,
            evaluation,
            options,
            body: String::new(),
            next_temp: 0,
            vertex_outputs,
            pixel_attributes,
            image_kinds,
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
        let sampler_count = self
            .image_kinds
            .iter()
            .filter(|kind| kind.sampler_slot.is_some())
            .count();
        let sampler_count = u32::try_from(sampler_count)
            .map_err(|_| MetalCompileError::InvalidInterface("too many samplers".into()))?;
        let initial_scalar_buffer_index = global_buffer_count;
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
                initial_scalar_buffer_index,
                sampler_argument_buffer_index: (sampler_count != 0)
                    .then_some(initial_scalar_buffer_index + 1),
                texture_count: u32::try_from(self.image_kinds.len())
                    .map_err(|_| MetalCompileError::InvalidInterface("too many textures".into()))?,
                sampler_count,
                threadgroup_memory_bytes,
            },
            vertex_inputs: self.evaluation.vertex_inputs.clone(),
            interpolated_attribute_count: self
                .pixel_attributes
                .last()
                .map_or(0, |attribute| attribute + 1),
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
                if self.options.stage != MetalShaderStage::Compute {
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
        let initial_index = global_count;
        let sampler_count = self
            .image_kinds
            .iter()
            .filter(|kind| kind.sampler_slot.is_some())
            .count();
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
        if self
            .program
            .instructions
            .iter()
            .any(|instruction| matches!(instruction.control, Some(InstructionControl::Vop3p(_))))
        {
            emit_vop3p_prelude(&mut source);
        }
        let uses_lds_memory = self
            .program
            .instructions
            .iter()
            .any(is_lds_memory_instruction);
        if uses_lds_memory {
            emit_data_share_prelude(&mut source);
        }
        if sampler_count != 0 {
            source.push_str("struct RemuSamplers {\n");
            for slot in 0..sampler_count {
                let _ = writeln!(source, "    sampler s{slot} [[id({slot})]];");
            }
            source.push_str("};\n\n");
        }

        match self.options.stage {
            MetalShaderStage::Vertex => self.emit_vertex_interface(&mut source),
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
            MetalShaderStage::Compute => {}
        }
        for (index, binding) in self.evaluation.global_memory_bindings.iter().enumerate() {
            let qualifier = if binding.writable {
                "device"
            } else {
                "const device"
            };
            parameters.push(format!("{qualifier} uchar* b{index} [[buffer({index})]]"));
        }
        parameters.push(format!(
            "constant uint* remu_initial_s [[buffer({initial_index})]]"
        ));
        for (index, kind) in self.image_kinds.iter().enumerate() {
            let access = if kind.storage {
                ", access::read_write"
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
        if sampler_count != 0 {
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
        }
        let signature = match self.options.stage {
            MetalShaderStage::Vertex => "vertex RemuVsOut remu_main",
            MetalShaderStage::Pixel => "fragment RemuPsOut remu_main",
            MetalShaderStage::Compute => "kernel void remu_main",
        };
        let _ = writeln!(
            source,
            "{signature}(\n    {}\n) {{",
            parameters.join(",\n    ")
        );
        source.push_str("    /* REMU_REGISTER_STATE */\n");
        source.push_str("    bool exec = true;\n    bool vcc = false;\n    bool scc = false;\n");
        if self.split_wave64() {
            source.push_str(
                "    uint remu_wave_half = (remu_thread_index >> 5u) & 1u;\n\
                 \x20   uint remu_wave_lane = remu_lane + (remu_wave_half << 5u);\n",
            );
        }
        if matches!(self.control_flow, ControlFlowLowering::Dispatcher) {
            source.push_str("    uint pc = 0u;\n    bool active = true;\n    uint steps = 0u;\n");
        }
        if self.options.stage != MetalShaderStage::Compute {
            // Graphics stages cannot declare thread_index_in_simdgroup, but the exclusive prefix
            // of one is the architectural lane index. Wave operations are emitted before the
            // EXEC-guarded write, keeping every host lane converged at simdgroup intrinsics.
            source.push_str("    uint remu_lane = simd_prefix_exclusive_sum(1u);\n");
        }
        match self.options.stage {
            MetalShaderStage::Vertex => {
                source.push_str("    RemuVsOut remu_out = {};\n");
                if contains_register_reference(&self.body, 'v', 5) {
                    source.push_str("    v[5] = remu_vertex_id;\n");
                }
                if contains_register_reference(&self.body, 'v', 8) {
                    source.push_str("    v[8] = remu_instance_id;\n");
                }
            }
            MetalShaderStage::Pixel => {
                source.push_str("    RemuPsOut remu_out = {};\n");
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
                        let field = format!("remu_out.mrt{}", output.guest_slot);
                        source.push_str(
                            "    uint2 remu_debug_pixel_xy = uint2(remu_in.frag_coord.xy);\n\
                             \x20   if (remu_debug_pixel_xy.x < 1920u && remu_debug_pixel_xy.y < 1080u) {\n\
                             \x20       uint remu_debug_pixel_base = (remu_debug_pixel_xy.y * 1920u + remu_debug_pixel_xy.x) * 5u;\n",
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
                    source.push_str(
                        "    if (remu_vertex_id < 262144u) {\n\
                         \x20       uint remu_debug_base = remu_vertex_id * 4u;\n\
                         \x20       remu_debug_position[remu_debug_base + 0u] = as_type<uint>(remu_out.position.x);\n\
                         \x20       remu_debug_position[remu_debug_base + 1u] = as_type<uint>(remu_out.position.y);\n\
                         \x20       remu_debug_position[remu_debug_base + 2u] = as_type<uint>(remu_out.position.z);\n\
                         \x20       remu_debug_position[remu_debug_base + 3u] = as_type<uint>(remu_out.position.w);\n\
                         \x20   }\n",
                    );
                }
                source.push_str("    return remu_out;\n");
            }
            MetalShaderStage::Compute => {}
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
            source.push_str(
                "    uint remu_wave_id = remu_thread_index >> 5u;\n\
                 \x20   uint remu_threadgroup_threads = remu_group_size.x * remu_group_size.y * remu_group_size.z;\n\
                 \x20   uint remu_threadgroup_waves = (remu_threadgroup_threads + 31u) >> 5u;\n",
            );
            // Ordinary compute dispatches have a zero ordered-append term. The remaining fields
            // match the RDNA2 initial-wave SGPR layout: FIRST_WAVE[31], WAVE_ID[24:20], and the
            // number of wave32s in this threadgroup in [5:0].
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
        for attribute in &self.pixel_attributes {
            let _ = writeln!(
                source,
                "    float4 attr{attribute} [[user(locn{attribute})]];"
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
                        "    v[{vgpr}] = as_type<uint>(remu_in.frag_coord.{component});"
                    );
                }
                vgpr += 1;
            }
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
        }
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
        let destination = destination_vector(instruction)?;
        self.store_vector(
            destination,
            &format!(
                "as_type<uint>(remu_in.attr{}[{}])",
                interpolation.attribute, interpolation.channel
            ),
        );
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
            OperandKind::LiteralConstant => Ok(format!("(ulong){}", format_u32(operand.value))),
            OperandKind::EncodedConstant => decode_inline_constant(operand.value)
                .map(|value| format!("(ulong){}", format_u32(value)))
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

    fn dpp_source_lane(control: DppControl) -> Result<(String, String), String> {
        let dpp = control.control;
        if dpp <= 0xff {
            let selected = format!("(({dpp}u >> ((remu_lane & 3u) * 2u)) & 3u)");
            let target = format!("((remu_lane & 0xFFFFFFFCu) + {selected})");
            return Ok((target, "true".to_owned()));
        }
        if (0x101..=0x10f).contains(&dpp) {
            let shift = dpp & 15;
            let shifted = format!("((remu_lane & 15u) + {shift}u)");
            let in_range = format!("({shifted} < 16u)");
            let target = format!("((remu_lane & 0xFFFFFFF0u) + ({shifted} & 15u))");
            return Ok((target, in_range));
        }
        if (0x111..=0x11f).contains(&dpp) {
            let shift = dpp & 15;
            let in_range = format!("((remu_lane & 15u) >= {shift}u)");
            let target =
                format!("((remu_lane & 0xFFFFFFF0u) + (((remu_lane & 15u) - {shift}u) & 15u))");
            return Ok((target, in_range));
        }
        if (0x121..=0x12f).contains(&dpp) {
            let shift = dpp & 15;
            let target =
                format!("((remu_lane & 0xFFFFFFF0u) + (((remu_lane & 15u) - {shift}u) & 15u))");
            return Ok((target, "true".to_owned()));
        }
        let target = match dpp {
            0x140 => "(remu_lane & 0xFFFFFFF0u) + (15u - (remu_lane & 15u))".to_owned(),
            0x141 => "(remu_lane & 0xFFFFFFF8u) + (7u - (remu_lane & 7u))".to_owned(),
            0x150..=0x15f => format!("(remu_lane & 0xFFFFFFF0u) + {}u", dpp & 15),
            0x160..=0x16f => format!(
                "(remu_lane & 0xFFFFFFF0u) + ((remu_lane & 15u) ^ {}u)",
                dpp & 15
            ),
            _ => return Err(format!("unsupported DPP16 control 0x{dpp:03X}")),
        };
        Ok((format!("({target})"), "true".to_owned()))
    }

    fn shuffle_lane(value: &str, target_lane: &str) -> String {
        format!("simd_shuffle({value}, (ushort){target_lane})")
    }

    fn lane_active_expression(target_lane: &str) -> String {
        format!("simd_shuffle(exec ? 1u : 0u, (ushort){target_lane}) != 0u")
    }

    fn apply_dpp_source(control: DppControl, value: &str) -> Result<String, String> {
        let (target_lane, in_range) = Self::dpp_source_lane(control)?;
        let safe_target = format!("((({in_range}) ? {target_lane} : remu_lane) & 31u)");
        let shuffled = Self::shuffle_lane(value, &safe_target);
        if control.fetch_inactive {
            return Ok(shuffled);
        }
        let source_active = Self::lane_active_expression(&safe_target);
        Ok(format!(
            "((({in_range}) && ({source_active})) ? ({shuffled}) : 0u)"
        ))
    }

    fn apply_dpp8_source(control: Dpp8Control, value: &str) -> String {
        let selector = format!(
            "(({}u >> ((remu_lane & 7u) * 3u)) & 7u)",
            control.lane_selectors
        );
        let target_lane = format!("(((remu_lane & 0xFFFFFFF8u) + {selector}) & 31u)");
        let shuffled = Self::shuffle_lane(value, &target_lane);
        if control.fetch_inactive {
            return shuffled;
        }
        let source_active = Self::lane_active_expression(&target_lane);
        format!("(({source_active}) ? ({shuffled}) : 0u)")
    }

    fn emit_dpp_write_enabled(&mut self, control: DppControl) -> Result<String, String> {
        let (_, in_range) = Self::dpp_source_lane(control)?;
        let row_enabled = format!("(({}u >> (remu_lane >> 4)) & 1u) != 0u", control.row_mask);
        let bank_enabled = format!("(({}u >> (remu_lane & 3u)) & 1u) != 0u", control.bank_mask);
        let source_allows = if control.bound_control {
            "true"
        } else {
            &in_range
        };
        Ok(self.temp(
            "bool",
            &format!("({row_enabled}) && ({bank_enabled}) && ({source_allows})"),
        ))
    }

    fn emit_vop3p(&mut self, instruction: &Instruction, name: &str) -> Result<(), String> {
        let Some(InstructionControl::Vop3p(control)) = instruction.control.as_ref() else {
            return Err(format!("missing VOP3P control for {name}"));
        };
        let destination = destination_vector(instruction)?;
        let result = match name {
            "VPkFmaF16" | "VPkAddF16" | "VPkMulF16" | "VPkMinF16" | "VPkMaxF16" => {
                self.emit_packed_f16(instruction, name, *control)?
            }
            "VFmaMixF32" | "VFmaMixloF16" | "VFmaMixhiF16" => {
                self.emit_fma_mix(instruction, name, destination, *control)?
            }
            _ => return Err(format!("unsupported packed VOP3 opcode {name}")),
        };
        self.store_vector(destination, &result);
        Ok(())
    }

    fn emit_packed_f16(
        &mut self,
        instruction: &Instruction,
        name: &str,
        control: Vop3pControl,
    ) -> Result<String, String> {
        let source_count = usize::from(name == "VPkFmaF16") + 2;
        let mut raw_sources = Vec::with_capacity(source_count);
        for index in 0..source_count {
            let source = self.vop3p_packed_raw_source(instruction, index)?;
            raw_sources.push(self.temp("uint", &source));
        }
        let low = self.emit_packed_f16_lane(name, &raw_sources, control, false);
        let high = self.emit_packed_f16_lane(name, &raw_sources, control, true);
        Ok(format!("({low} | ({high} << 16))"))
    }

    fn emit_packed_f16_lane(
        &mut self,
        name: &str,
        raw_sources: &[String],
        control: Vop3pControl,
        high_lane: bool,
    ) -> String {
        let select_mask = if high_lane {
            control.operand_select_high_mask
        } else {
            control.operand_select_mask
        };
        let negate_mask = if high_lane {
            control.negate_high_mask
        } else {
            control.negate_low_mask
        };
        let mut operands = Vec::with_capacity(raw_sources.len());
        for (index, raw) in raw_sources.iter().enumerate() {
            let shift = if select_mask & (1 << index) != 0 {
                16
            } else {
                0
            };
            let mut half_bits = format!("(({raw} >> {shift}) & 0xFFFFu)");
            if negate_mask & (1 << index) != 0 {
                half_bits = format!("({half_bits} ^ 0x8000u)");
            }
            operands.push(self.temp(
                "float",
                &format!("as_type<float>(remu_f16_to_f32_bits({half_bits}))"),
            ));
        }
        let mut result_bits = match name {
            "VPkFmaF16" => self.temp(
                "uint",
                &format!(
                    "remu_f16_fma_rto_bits({}, {}, {})",
                    operands[0], operands[1], operands[2]
                ),
            ),
            "VPkAddF16" => self.temp(
                "uint",
                &format!("as_type<uint>({} + {})", operands[0], operands[1]),
            ),
            "VPkMulF16" => self.temp(
                "uint",
                &format!("as_type<uint>({} * {})", operands[0], operands[1]),
            ),
            "VPkMinF16" => self.temp(
                "uint",
                &format!(
                    "remu_f16_minmax_bits({}, {}, false)",
                    operands[0], operands[1]
                ),
            ),
            "VPkMaxF16" => self.temp(
                "uint",
                &format!(
                    "remu_f16_minmax_bits({}, {}, true)",
                    operands[0], operands[1]
                ),
            ),
            _ => unreachable!("packed opcode was classified before lane lowering"),
        };
        if control.clamp {
            result_bits = self.temp("uint", &format!("remu_clamp_unit_f32_bits({result_bits})"));
        }
        self.temp("uint", &format!("remu_f32_to_f16_bits({result_bits})"))
    }

    fn emit_fma_mix(
        &mut self,
        instruction: &Instruction,
        name: &str,
        destination: u32,
        control: Vop3pControl,
    ) -> Result<String, String> {
        let mut operands = Vec::with_capacity(3);
        for index in 0..3 {
            operands.push(self.fma_mix_operand(instruction, control, index)?);
        }
        let mut result_bits = self.temp(
            "uint",
            &format!(
                "as_type<uint>(metal::precise::fma({}, {}, {}))",
                operands[0], operands[1], operands[2]
            ),
        );
        if control.clamp {
            result_bits = self.temp("uint", &format!("remu_clamp_unit_f32_bits({result_bits})"));
        }
        Ok(match name {
            "VFmaMixF32" => result_bits,
            "VFmaMixloF16" => {
                let half = self.temp("uint", &format!("remu_f32_to_f16_bits({result_bits})"));
                format!("(v[{destination}] & 0xFFFF0000u) | {half}")
            }
            "VFmaMixhiF16" => {
                let half = self.temp("uint", &format!("remu_f32_to_f16_bits({result_bits})"));
                format!("(v[{destination}] & 0x0000FFFFu) | ({half} << 16)")
            }
            _ => unreachable!("mix opcode was classified before result lowering"),
        })
    }

    fn fma_mix_operand(
        &mut self,
        instruction: &Instruction,
        control: Vop3pControl,
        index: usize,
    ) -> Result<String, String> {
        let operand = instruction
            .sources
            .get(index)
            .copied()
            .ok_or_else(|| format!("missing VOP3P source operand {index}"))?;
        let read_as_half = control.operand_select_high_mask & (1 << index) != 0
            && matches!(
                operand.kind,
                OperandKind::ScalarRegister | OperandKind::VectorRegister
            );
        let mut bits = if read_as_half {
            let raw = self.raw_source(instruction, index)?;
            let shift = if control.operand_select_mask & (1 << index) != 0 {
                16
            } else {
                0
            };
            format!("remu_f16_to_f32_bits((({raw} >> {shift}) & 0xFFFFu))")
        } else {
            format!("as_type<uint>({})", self.float_source(instruction, index)?)
        };
        // FMA_MIX repurposes NEG_HI as absolute value and NEG as negate. Apply them
        // in architectural abs-then-neg order using sign-bit operations.
        if control.negate_high_mask & (1 << index) != 0 {
            bits = format!("({bits} & 0x7FFFFFFFu)");
        }
        if control.negate_low_mask & (1 << index) != 0 {
            bits = format!("({bits} ^ 0x80000000u)");
        }
        Ok(self.temp("float", &format!("as_type<float>({bits})")))
    }

    #[allow(clippy::unused_self)]
    fn vop3p_packed_raw_source(
        &self,
        instruction: &Instruction,
        index: usize,
    ) -> Result<String, String> {
        let operand = instruction
            .sources
            .get(index)
            .copied()
            .ok_or_else(|| format!("missing VOP3P source operand {index}"))?;
        match operand.kind {
            OperandKind::ScalarRegister if operand.value < SCALAR_REGISTER_COUNT => {
                Ok(format!("s[{}]", operand.value))
            }
            OperandKind::VectorRegister if operand.value < VECTOR_REGISTER_COUNT => {
                Ok(format!("v[{}]", operand.value))
            }
            OperandKind::LiteralConstant => Ok(format_u32(operand.value)),
            OperandKind::EncodedConstant => decode_v2f16_inline_constant(operand.value)
                .map(format_u32)
                .ok_or_else(|| {
                    format!("unsupported packed-f16 encoded constant {}", operand.value)
                }),
            _ => Err(format!(
                "VOP3P register operand is outside the architectural file: {operand}"
            )),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn emit_vector_alu(&mut self, instruction: &Instruction) -> Result<(), String> {
        if let Some(InstructionControl::Sdwa(control)) = instruction.control.as_ref()
            && (control.source0_select == 7
                || control.source1_select == 7
                || control.destination_select == 7
                || control.destination_unused == 3)
        {
            return Err("reserved SDWA selector or destination-unused mode".to_owned());
        }
        if let Some(InstructionControl::Dpp(control)) = instruction.control
            && !is_supported_dpp_control(control.control)
        {
            return Err(format!(
                "unsupported DPP16 control 0x{:03X}",
                control.control
            ));
        }
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw vector opcode".to_owned())?;
        if matches!(instruction.control, Some(InstructionControl::Vop3p(_))) {
            return self.emit_vop3p(instruction, name);
        }
        if name.starts_with("VCmp") {
            return self.emit_vector_compare(instruction, name);
        }
        if matches!(
            name,
            "VAddcU32"
                | "VSubbU32"
                | "VSubbrevU32"
                | "VAddCoU32"
                | "VSubCoU32"
                | "VSubrevCoU32"
                | "VAddCoCiU32"
        ) {
            return self.emit_vector_carry(instruction, name);
        }
        if name == "VReadlaneB32" {
            let destination = destination_scalar(instruction)?;
            let value = self.raw_source(instruction, 0)?;
            let lane = self.raw_source(instruction, 1)?;
            let selected = self.temp(
                "uint",
                &format!("simd_shuffle({value}, (ushort)({lane} & 31u))"),
            );
            // V_READLANE is a wave operation and is architecturally independent of EXEC.
            self.store_scalar(destination, &selected);
            return Ok(());
        }
        if name == "VWritelaneB32" {
            let destination = destination_vector(instruction)?;
            let value = self.raw_source(instruction, 0)?;
            let lane = self.raw_source(instruction, 1)?;
            // V_WRITELANE also ignores EXEC: exactly the selected physical lane is replaced and
            // all other lanes retain the old VGPR value.
            self.line(
                3,
                &format!("if (remu_lane == (({lane}) & 31u)) v[{destination}] = {value};"),
            );
            return Ok(());
        }
        let destination = destination_vector(instruction)?;
        let expression = match name {
            "VCndmaskB32" => {
                let mask = if instruction.sources.len() > 2 {
                    self.mask_source(instruction.sources[2])?
                } else {
                    "vcc".to_owned()
                };
                let false_value = self.raw_source(instruction, 0)?;
                let true_value = self.raw_source(instruction, 1)?;
                format!("({mask}) ? ({true_value}) : ({false_value})")
            }
            "VMovB32" => self.raw_source(instruction, 0)?,
            "VBfrevB32" => format!("reverse_bits({})", self.raw_source(instruction, 0)?),
            "VAddF32" => self.float_result(
                instruction,
                &format!(
                    "{} + {}",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                ),
            ),
            "VSubF32" => self.float_result(
                instruction,
                &format!(
                    "{} - {}",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                ),
            ),
            "VSubrevF32" => self.float_result(
                instruction,
                &format!(
                    "{} - {}",
                    self.float_source(instruction, 1)?,
                    self.float_source(instruction, 0)?
                ),
            ),
            "VMulF32" => self.float_result(
                instruction,
                &format!(
                    "{} * {}",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                ),
            ),
            "VMinF32" => self.float_result(
                instruction,
                &format!(
                    "fmin({}, {})",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                ),
            ),
            "VMaxF32" => self.float_result(
                instruction,
                &format!(
                    "fmax({}, {})",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                ),
            ),
            "VMin3F32" => self.float_result(
                instruction,
                &format!(
                    "fmin(fmin({}, {}), {})",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?,
                    self.float_source(instruction, 2)?
                ),
            ),
            "VMax3F32" => self.float_result(
                instruction,
                &format!(
                    "fmax(fmax({}, {}), {})",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?,
                    self.float_source(instruction, 2)?
                ),
            ),
            "VMed3F32" => {
                let first = self.float_source(instruction, 0)?;
                let second = self.float_source(instruction, 1)?;
                let third = self.float_source(instruction, 2)?;
                self.float_result(
                    instruction,
                    &format!(
                        "fmax(fmin({first}, {second}), fmin(fmax({first}, {second}), {third}))"
                    ),
                )
            }
            "VMin3I32" => format!(
                "as_type<uint>(min(min(as_type<int>({}), as_type<int>({})), as_type<int>({})))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VMax3I32" => format!(
                "as_type<uint>(max(max(as_type<int>({}), as_type<int>({})), as_type<int>({})))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VMed3I32" => {
                let first = format!("as_type<int>({})", self.raw_source(instruction, 0)?);
                let second = format!("as_type<int>({})", self.raw_source(instruction, 1)?);
                let third = format!("as_type<int>({})", self.raw_source(instruction, 2)?);
                format!(
                    "as_type<uint>(max(min({first}, {second}), min(max({first}, {second}), {third})))"
                )
            }
            "VMin3U32" => format!(
                "min(min({}, {}), {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VMax3U32" => format!(
                "max(max({}, {}), {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VMed3U32" => {
                let first = self.raw_source(instruction, 0)?;
                let second = self.raw_source(instruction, 1)?;
                let third = self.raw_source(instruction, 2)?;
                format!("max(min({first}, {second}), min(max({first}, {second}), {third}))")
            }
            "VFmacF32" | "VMacF32" => self.float_result(
                instruction,
                &format!(
                    "fma({}, {}, as_type<float>(v[{destination}]))",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                ),
            ),
            "VFloorF32" => self.float_result(
                instruction,
                &format!("floor({})", self.float_source(instruction, 0)?),
            ),
            "VCeilF32" => self.float_result(
                instruction,
                &format!("ceil({})", self.float_source(instruction, 0)?),
            ),
            "VTruncF32" => self.float_result(
                instruction,
                &format!("trunc({})", self.float_source(instruction, 0)?),
            ),
            "VRndneF32" => self.float_result(
                instruction,
                &format!("rint({})", self.float_source(instruction, 0)?),
            ),
            "VFractF32" => self.float_result(
                instruction,
                &format!("fract({})", self.float_source(instruction, 0)?),
            ),
            "VSqrtF32" => self.float_result(
                instruction,
                &format!("sqrt({})", self.float_source(instruction, 0)?),
            ),
            "VRsqF32" => self.float_result(
                instruction,
                &format!("rsqrt({})", self.float_source(instruction, 0)?),
            ),
            "VRcpF32" | "VRcpIflagF32" => self.float_result(
                instruction,
                &format!("1.0f / {}", self.float_source(instruction, 0)?),
            ),
            "VLogF32" => self.float_result(
                instruction,
                &format!("log2({})", self.float_source(instruction, 0)?),
            ),
            "VExpF32" => self.float_result(
                instruction,
                &format!("exp2({})", self.float_source(instruction, 0)?),
            ),
            "VSinF32" => self.float_result(
                instruction,
                &format!(
                    "sin({} * 6.28318530717958647692f)",
                    self.float_source(instruction, 0)?
                ),
            ),
            "VCosF32" => self.float_result(
                instruction,
                &format!(
                    "cos({} * 6.28318530717958647692f)",
                    self.float_source(instruction, 0)?
                ),
            ),
            "VAndB32" => format!(
                "({} & {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VOrB32" => format!(
                "({} | {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VXorB32" => format!(
                "({} ^ {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VXnorB32" => format!(
                "~({} ^ {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VNotB32" => format!("~{}", self.raw_source(instruction, 0)?),
            "VAddI32" => format!(
                "({} + {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VSubI32" => format!(
                "({} - {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VSubrevI32" => format!(
                "({} - {})",
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 0)?
            ),
            "VMinI32" => format!(
                "as_type<uint>(min(as_type<int>({}), as_type<int>({})))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMaxI32" => format!(
                "as_type<uint>(max(as_type<int>({}), as_type<int>({})))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VLshrB32" => format!(
                "({} >> ({} & 31u))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VLshrrevB32" => format!(
                "({} >> ({} & 31u))",
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 0)?
            ),
            "VAshrI32" => format!(
                "as_type<uint>(as_type<int>({}) >> ({} & 31u))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VAshrrevI32" => format!(
                "as_type<uint>(as_type<int>({}) >> ({} & 31u))",
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 0)?
            ),
            "VLshlB32" => format!(
                "({} << ({} & 31u))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VLshlrevB32" => format!(
                "({} << ({} & 31u))",
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 0)?
            ),
            "VLshlAddU32" => format!(
                "(({} << ({} & 31u)) + {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VCvtF32I32" => self.float_result(
                instruction,
                &format!("(float)as_type<int>({})", self.raw_source(instruction, 0)?),
            ),
            "VCvtF32U32" => self.float_result(
                instruction,
                &format!("(float){}", self.raw_source(instruction, 0)?),
            ),
            "VMinU32" => format!(
                "min({}, {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMaxU32" => format!(
                "max({}, {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMulLoU32" => format!(
                "({} * {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMulU32U24" => format!(
                "(({} & 0xFFFFFFu) * ({} & 0xFFFFFFu))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMulHiU32U24" => format!(
                "(uint)(((ulong)({} & 0xFFFFFFu) * (ulong)({} & 0xFFFFFFu)) >> 32)",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VCvtU32F32" => {
                let source = self.temp("float", &self.float_source(instruction, 0)?);
                format!(
                    "(isnan({source}) || {source} <= 0.0f) ? 0u : ({source} >= 4294967296.0f ? 0xFFFFFFFFu : (uint)trunc({source}))"
                )
            }
            "VCvtI32F32" => {
                let source = self.temp("float", &self.float_source(instruction, 0)?);
                format!(
                    "isnan({source}) ? 0u : ({source} <= -2147483648.0f ? 0x80000000u : ({source} >= 2147483648.0f ? 0x7FFFFFFFu : as_type<uint>((int)trunc({source}))))"
                )
            }
            "VCvtOffF32I4" => {
                let source = self.temp("uint", &self.raw_source(instruction, 0)?);
                self.float_result(
                    instruction,
                    &format!("(float)((int)({source} << 28) >> 28) * 0.0625f"),
                )
            }
            "VCubeidF32" | "VCubescF32" | "VCubetcF32" | "VCubemaF32" => {
                self.cube_result(instruction, name)?
            }
            "VCvtPkrtzF16F32" => {
                let first = self.temp(
                    "float",
                    &format!(
                        "as_type<float>(as_type<uint>({}) & 0xFFFFE000u)",
                        self.float_source(instruction, 0)?
                    ),
                );
                let second = self.temp(
                    "float",
                    &format!(
                        "as_type<float>(as_type<uint>({}) & 0xFFFFE000u)",
                        self.float_source(instruction, 1)?
                    ),
                );
                format!(
                    "((uint)as_type<ushort>(half({first})) | ((uint)as_type<ushort>(half({second})) << 16))"
                )
            }
            "VMadF32" | "VFmaF32" | "VMadAkF32" | "VMadMkF32" | "VFmaAkF32" | "VFmaMkF32" => self
                .float_result(
                    instruction,
                    &format!(
                        "fma({}, {}, {})",
                        self.float_source(instruction, 0)?,
                        self.float_source(instruction, 1)?,
                        self.float_source(instruction, 2)?
                    ),
                ),
            _ => return Err(format!("unsupported vector opcode {name}")),
        };
        let expression =
            if let Some(InstructionControl::Dpp(control)) = instruction.control.as_ref() {
                let result = self.temp("uint", &expression);
                let write_enabled = self.emit_dpp_write_enabled(*control)?;
                format!("({write_enabled}) ? {result} : v[{destination}]")
            } else if matches!(instruction.control, Some(InstructionControl::Dpp8(_))) {
                // Keep the shuffle converged across inactive guest lanes; only the architectural
                // destination write is EXEC-masked below.
                self.temp("uint", &expression)
            } else if let Some(InstructionControl::Sdwa(control)) = instruction.control.as_ref()
                && control.scalar_destination.is_none()
            {
                let result = self.temp("uint", &expression);
                self.apply_sdwa_destination(control, &result, &format!("v[{destination}]"))?
            } else {
                expression
            };
        self.store_vector(destination, &expression);
        Ok(())
    }

    fn emit_vector_carry(&mut self, instruction: &Instruction, name: &str) -> Result<(), String> {
        let destination = destination_vector(instruction)?;
        let first = self.temp("uint", &self.raw_source(instruction, 0)?);
        let second = self.temp("uint", &self.raw_source(instruction, 1)?);
        let carry_in = match name {
            "VAddcU32" | "VSubbU32" | "VSubbrevU32" => self.temp("bool", "vcc"),
            "VAddCoCiU32" => {
                let operand = instruction
                    .sources
                    .get(2)
                    .copied()
                    .ok_or_else(|| "missing VOP3 carry-in lane mask".to_owned())?;
                self.temp("bool", &self.mask_source(operand)?)
            }
            _ => self.temp("bool", "false"),
        };
        let (expression, carry) = match name {
            "VAddcU32" | "VAddCoCiU32" => {
                let wide = self.temp(
                    "ulong",
                    &format!("(ulong){first} + (ulong){second} + (ulong){carry_in}"),
                );
                (format!("(uint){wide}"), format!("({wide} >> 32) != 0ul"))
            }
            "VAddCoU32" => {
                let wide = self.temp("ulong", &format!("(ulong){first} + (ulong){second}"));
                (format!("(uint){wide}"), format!("({wide} >> 32) != 0ul"))
            }
            "VSubbU32" | "VSubCoU32" => {
                let subtrahend = self.temp(
                    "ulong",
                    &format!(
                        "(ulong){second} + (ulong){}",
                        if name == "VSubbU32" {
                            carry_in.as_str()
                        } else {
                            "false"
                        }
                    ),
                );
                (
                    format!("{first} - (uint){subtrahend}"),
                    format!("(ulong){first} < {subtrahend}"),
                )
            }
            "VSubbrevU32" | "VSubrevCoU32" => {
                let subtrahend = self.temp(
                    "ulong",
                    &format!(
                        "(ulong){first} + (ulong){}",
                        if name == "VSubbrevU32" {
                            carry_in.as_str()
                        } else {
                            "false"
                        }
                    ),
                );
                (
                    format!("{second} - (uint){subtrahend}"),
                    format!("(ulong){second} < {subtrahend}"),
                )
            }
            _ => unreachable!("carry opcode was selected from a closed set"),
        };
        let result = self.temp("uint", &expression);
        let mut carry = self.temp("bool", &format!("exec && ({carry})"));
        let expression =
            if let Some(InstructionControl::Dpp(control)) = instruction.control.as_ref() {
                let write_enabled = self.emit_dpp_write_enabled(*control)?;
                carry = self.temp("bool", &format!("({write_enabled}) ? {carry} : vcc"));
                format!("({write_enabled}) ? {result} : v[{destination}]")
            } else if matches!(instruction.control, Some(InstructionControl::Dpp8(_))) {
                result
            } else if let Some(InstructionControl::Sdwa(control)) = instruction.control.as_ref()
                && control.scalar_destination.is_none()
            {
                self.apply_sdwa_destination(control, &result, &format!("v[{destination}]"))?
            } else {
                result
            };
        let scalar_destination = match instruction.control.as_ref() {
            Some(InstructionControl::Vop3(control)) => control.scalar_destination.unwrap_or(VCC_LO),
            _ => VCC_LO,
        };
        self.store_lane_mask(scalar_destination, &carry);
        self.store_vector(destination, &expression);
        Ok(())
    }

    fn cube_result(&mut self, instruction: &Instruction, name: &str) -> Result<String, String> {
        let x = self.temp("float", &self.float_source(instruction, 0)?);
        let y = self.temp("float", &self.float_source(instruction, 1)?);
        let z = self.temp("float", &self.float_source(instruction, 2)?);
        let z_major = format!("(fabs({z}) >= fabs({x}) && fabs({z}) >= fabs({y}))");
        let y_major = format!("(fabs({y}) >= fabs({x}))");
        let value = match name {
            "VCubeidF32" => format!(
                "{z_major} ? ({z} < 0.0f ? 5.0f : 4.0f) : ({y_major} ? ({y} < 0.0f ? 3.0f : 2.0f) : ({x} < 0.0f ? 1.0f : 0.0f))"
            ),
            "VCubescF32" => format!(
                "{z_major} ? ({z} < 0.0f ? -{x} : {x}) : ({y_major} ? {x} : ({x} < 0.0f ? {z} : -{z}))"
            ),
            "VCubetcF32" => {
                format!("{z_major} ? -{y} : ({y_major} ? ({y} < 0.0f ? -{z} : {z}) : -{y})")
            }
            "VCubemaF32" => {
                format!("2.0f * ({z_major} ? {z} : ({y_major} ? {y} : {x}))")
            }
            _ => unreachable!("cube result is selected from a closed opcode set"),
        };
        Ok(self.float_result(instruction, &value))
    }

    fn emit_vector_compare(&mut self, instruction: &Instruction, name: &str) -> Result<(), String> {
        let condition = match classify_vector_compare(name)? {
            VectorCompare::Always(value) => value.to_string(),
            VectorCompare::Ordered => format!(
                "(!isnan({}) && !isnan({}))",
                self.float_source(instruction, 0)?,
                self.float_source(instruction, 1)?
            ),
            VectorCompare::Unordered => format!(
                "(isnan({}) || isnan({}))",
                self.float_source(instruction, 0)?,
                self.float_source(instruction, 1)?
            ),
            VectorCompare::Float { operator, negate } => {
                let comparison = format!(
                    "({} {operator} {})",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                );
                if negate {
                    format!("!{comparison}")
                } else {
                    comparison
                }
            }
            VectorCompare::Integer { operator, signed } => {
                let left = self.raw_source(instruction, 0)?;
                let right = self.raw_source(instruction, 1)?;
                if signed {
                    format!("(as_type<int>({left}) {operator} as_type<int>({right}))")
                } else {
                    format!("(({left}) {operator} ({right}))")
                }
            }
            VectorCompare::Class => self.emit_float_class_compare(instruction)?,
        };
        let mut active = self.temp("bool", &format!("exec && ({condition})"));
        if let Some(InstructionControl::Dpp(control)) = instruction.control.as_ref() {
            let write_enabled = self.emit_dpp_write_enabled(*control)?;
            active = self.temp("bool", &format!("({write_enabled}) ? {active} : vcc"));
        }
        if name.starts_with("VCmpx") {
            self.store_exec_mask(&active);
        } else {
            let destination = match instruction.control.as_ref() {
                Some(InstructionControl::Sdwa(control)) => {
                    control.scalar_destination.unwrap_or(VCC_LO)
                }
                _ => VCC_LO,
            };
            self.store_lane_mask(destination, &active);
        }
        Ok(())
    }

    fn emit_float_class_compare(&mut self, instruction: &Instruction) -> Result<String, String> {
        let value = self.temp("float", &self.float_source(instruction, 0)?);
        let raw = self.temp("uint", &self.raw_source(instruction, 0)?);
        let mask = self.temp("uint", &self.raw_source(instruction, 1)?);
        let negative = self.temp("bool", &format!("({raw} & 0x80000000u) != 0u"));
        let nan = self.temp("bool", &format!("isnan({value})"));
        let infinite = self.temp("bool", &format!("isinf({value})"));
        let zero = self.temp("bool", &format!("{value} == 0.0f"));
        let subnormal = self.temp(
            "bool",
            &format!("fabs({value}) > 0.0f && fabs({value}) < as_type<float>(0x00800000u)"),
        );
        let normal = self.temp(
            "bool",
            &format!("!({nan} || {infinite} || {zero} || {subnormal})"),
        );
        Ok(format!(
            "((({mask} & 3u) != 0u) && {nan}) || \
             ((({mask} >> 2) & 1u) != 0u && {infinite} && {negative}) || \
             ((({mask} >> 3) & 1u) != 0u && {normal} && {negative}) || \
             ((({mask} >> 4) & 1u) != 0u && {subnormal} && {negative}) || \
             ((({mask} >> 5) & 1u) != 0u && {zero} && {negative}) || \
             ((({mask} >> 6) & 1u) != 0u && {zero} && !{negative}) || \
             ((({mask} >> 7) & 1u) != 0u && {subnormal} && !{negative}) || \
             ((({mask} >> 8) & 1u) != 0u && {normal} && !{negative}) || \
             ((({mask} >> 9) & 1u) != 0u && {infinite} && !{negative})"
        ))
    }

    fn store_lane_mask(&mut self, register: u32, condition: &str) {
        let (low, high) = self.ballot_words(condition);
        match register {
            VCC_LO => {
                self.line(3, &format!("vcc = {condition};"));
                self.line(3, &format!("s[{VCC_LO}] = {low};"));
                self.line(3, &format!("s[{VCC_HI}] = {high};"));
            }
            EXEC_LO => self.store_exec_mask(condition),
            register if register < SCALAR_REGISTER_COUNT => {
                self.line(3, &format!("s[{register}] = {low};"));
                if register + 1 < SCALAR_REGISTER_COUNT {
                    self.line(3, &format!("s[{}] = {high};", register + 1));
                }
            }
            _ => {}
        }
    }

    fn store_exec_mask(&mut self, condition: &str) {
        let (low, high) = self.ballot_words(condition);
        self.line(3, &format!("exec = {condition};"));
        self.line(3, &format!("s[{EXEC_LO}] = {low};"));
        self.line(3, &format!("s[{EXEC_HI}] = {high};"));
    }

    fn ballot(condition: &str) -> String {
        format!("(uint)(uint64_t)simd_ballot({condition})")
    }

    fn apply_sdwa_destination(
        &mut self,
        control: &SdwaControl,
        value: &str,
        previous: &str,
    ) -> Result<String, String> {
        let (shift, width) = match control.destination_select {
            0 => (0, 8),
            1 => (8, 8),
            2 => (16, 8),
            3 => (24, 8),
            4 => (0, 16),
            5 => (16, 16),
            6 => (0, 32),
            selector => return Err(format!("reserved SDWA destination selector {selector}")),
        };
        if width == 32 {
            return Ok(value.to_owned());
        }
        let low_mask = if width == 8 { 0xff_u32 } else { 0xffff_u32 };
        let field_mask = low_mask << shift;
        let upper_start = shift + width;
        let upper_mask = if upper_start == 32 {
            0
        } else {
            u32::MAX << upper_start
        };
        let sign_bit = 1_u32 << (shift + width - 1);
        let positioned = self.temp("uint", &format!("(({value}) & 0x{low_mask:X}u) << {shift}"));
        match control.destination_unused {
            0 => Ok(positioned),
            1 => Ok(format!(
                "{positioned} | ((({positioned} & 0x{sign_bit:X}u) != 0u) ? 0x{upper_mask:X}u : 0u)"
            )),
            2 => Ok(format!(
                "(({previous}) & 0x{:X}u) | {positioned}",
                !field_mask
            )),
            mode => Err(format!("reserved SDWA destination-unused mode {mode}")),
        }
    }

    #[allow(clippy::unused_self)]
    fn mask_source(&self, operand: Operand) -> Result<String, String> {
        match operand.kind {
            OperandKind::ScalarRegister if operand.value + 1 < SCALAR_REGISTER_COUNT => {
                Ok(format!("((s[{}] >> remu_lane) & 1u) != 0u", operand.value))
            }
            _ => Err(format!("invalid lane-mask operand {operand}")),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn emit_scalar_alu(&mut self, instruction: &Instruction) -> Result<(), String> {
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw scalar opcode".to_owned())?;
        if instruction.encoding == ShaderEncoding::Sopc {
            return self.emit_scalar_compare(instruction, name, "SCmp");
        }
        let destination = destination_scalar(instruction)?;
        if instruction.encoding == ShaderEncoding::Sopk {
            let immediate = u16::try_from(instruction.words[0] & 0xffff)
                .expect("masked immediate fits u16")
                .cast_signed();
            if name.starts_with("SCmpk") {
                return self.emit_scalar_compare_values(
                    name,
                    "SCmpk",
                    &format!("s[{destination}]"),
                    &format_u32(u32::from_ne_bytes(i32::from(immediate).to_ne_bytes())),
                );
            }
            let value = match name {
                "SMovkI32" => format_u32(u32::from_ne_bytes(i32::from(immediate).to_ne_bytes())),
                "SAddkI32" => format!(
                    "(s[{destination}] + {})",
                    format_u32(u32::from_ne_bytes(i32::from(immediate).to_ne_bytes()))
                ),
                "SMulkI32" => format!(
                    "(s[{destination}] * {})",
                    format_u32(u32::from_ne_bytes(i32::from(immediate).to_ne_bytes()))
                ),
                _ => return Err(format!("unsupported scalar immediate opcode {name}")),
            };
            self.store_scalar(destination, &value);
            return Ok(());
        }
        if name == "SGetpcB64" {
            let address = self.program.address
                + u64::from(instruction.pc)
                + u64::try_from(instruction.words.len() * 4)
                    .map_err(|_| "instruction size overflow".to_owned())?;
            let low = u32::try_from(address & u64::from(u32::MAX))
                .expect("masked address low half fits u32");
            let high = u32::try_from(address >> 32).expect("address high half fits u32");
            self.store_scalar(destination, &format_u32(low));
            self.store_scalar(destination + 1, &format_u32(high));
            return Ok(());
        }
        if name.ends_with("SaveexecB64") {
            return self.emit_saveexec64(instruction, name, destination);
        }
        if name.ends_with("B64") || matches!(name, "SBfeU64" | "SBfeI64") {
            return self.emit_scalar64(instruction, name, destination);
        }

        let left = self.raw_source(instruction, 0)?;
        match name {
            "SMovB32" => {
                self.store_scalar(destination, &left);
                return Ok(());
            }
            "SNotB32" => {
                let result = self.temp("uint", &format!("~({left})"));
                self.store_scalar(destination, &result);
                self.line(3, &format!("scc = {result} != 0u;"));
                return Ok(());
            }
            "SBrevB32" => {
                let result = self.temp("uint", &format!("reverse_bits({left})"));
                self.store_scalar(destination, &result);
                self.line(3, &format!("scc = {result} != 0u;"));
                return Ok(());
            }
            "SBcnt1I32B32" => {
                let result = self.temp("uint", &format!("popcount({left})"));
                self.store_scalar(destination, &result);
                self.line(3, &format!("scc = {result} != 0u;"));
                return Ok(());
            }
            _ => {}
        }
        let right = self.raw_source(instruction, 1)?;
        if matches!(name, "SAddcU32" | "SSubbU32") {
            let carry = self.temp("ulong", "scc ? 1ul : 0ul");
            if name == "SAddcU32" {
                let wide = self.temp(
                    "ulong",
                    &format!("(ulong){left} + (ulong){right} + {carry}"),
                );
                self.store_scalar(destination, &format!("(uint){wide}"));
                self.line(3, &format!("scc = ({wide} >> 32) != 0ul;"));
            } else {
                let subtrahend = self.temp("ulong", &format!("(ulong){right} + {carry}"));
                self.store_scalar(destination, &format!("{left} - (uint){subtrahend}"));
                self.line(3, &format!("scc = (ulong){left} < {subtrahend};"));
            }
            return Ok(());
        }
        let (expression, scc) = match name {
            "SAddU32" => (
                format!("({left} + {right})"),
                Some(format!("RESULT < {left}")),
            ),
            "SSubU32" => (
                format!("({left} - {right})"),
                Some(format!("{right} > {left}")),
            ),
            "SAddI32" => (
                format!("({left} + {right})"),
                Some(format!(
                    "((~({left} ^ {right}) & ({left} ^ RESULT)) >> 31) != 0u"
                )),
            ),
            "SSubI32" => (
                format!("({left} - {right})"),
                Some(format!(
                    "((({left} ^ {right}) & ({left} ^ RESULT)) >> 31) != 0u"
                )),
            ),
            "SMulI32" => (format!("({left} * {right})"), None),
            "SMulHiU32" => (format!("mulhi({left}, {right})"), None),
            "SAndB32" => (format!("({left} & {right})"), Some("RESULT != 0u".into())),
            "SOrB32" => (format!("({left} | {right})"), Some("RESULT != 0u".into())),
            "SXorB32" => (format!("({left} ^ {right})"), Some("RESULT != 0u".into())),
            "SAndn2B32" => (format!("({left} & ~{right})"), Some("RESULT != 0u".into())),
            "SOrn2B32" => (format!("({left} | ~{right})"), Some("RESULT != 0u".into())),
            "SLshlB32" => (
                format!("({left} << ({right} & 31u))"),
                Some("RESULT != 0u".into()),
            ),
            "SLshrB32" => (
                format!("({left} >> ({right} & 31u))"),
                Some("RESULT != 0u".into()),
            ),
            "SAshrI32" => (
                format!("as_type<uint>(as_type<int>({left}) >> ({right} & 31u))"),
                Some("RESULT != 0u".into()),
            ),
            "SBfmB32" => (
                format!("(((1u << ({left} & 31u)) - 1u) << ({right} & 31u))"),
                None,
            ),
            "SBfeU32" | "SBfeI32" => {
                let offset = self.temp("uint", &format!("{right} & 31u"));
                let width = self.temp(
                    "uint",
                    &format!("min(({right} >> 16) & 0x7Fu, 32u - {offset})"),
                );
                let value = if name == "SBfeI32" {
                    format!(
                        "{width} == 0u ? 0u : as_type<uint>(extract_bits(as_type<int>({left}), {offset}, {width}))"
                    )
                } else {
                    format!("{width} == 0u ? 0u : extract_bits({left}, {offset}, {width})")
                };
                (value, Some("RESULT != 0u".into()))
            }
            "SCselectB32" => (format!("(scc ? {left} : {right})"), None),
            "SMinU32" => (
                format!("min({left}, {right})"),
                Some(format!("{left} < {right}")),
            ),
            "SMaxU32" => (
                format!("max({left}, {right})"),
                Some(format!("{left} > {right}")),
            ),
            "SMinI32" => (
                format!("as_type<uint>(min(as_type<int>({left}), as_type<int>({right})))"),
                Some(format!("as_type<int>({left}) < as_type<int>({right})")),
            ),
            "SMaxI32" => (
                format!("as_type<uint>(max(as_type<int>({left}), as_type<int>({right})))"),
                Some(format!("as_type<int>({left}) > as_type<int>({right})")),
            ),
            "SLshl1AddU32" | "SLshl2AddU32" | "SLshl3AddU32" | "SLshl4AddU32" => {
                let shift = name.as_bytes()[5] - b'0';
                (format!("(({left} << {shift}) + {right})"), None)
            }
            _ => return Err(format!("unsupported scalar opcode {name}")),
        };
        let result = self.temp("uint", &expression);
        self.store_scalar(destination, &result);
        if let Some(scc) = scc {
            self.line(3, &format!("scc = {};", scc.replace("RESULT", &result)));
        }
        Ok(())
    }

    fn emit_saveexec64(
        &mut self,
        instruction: &Instruction,
        name: &str,
        destination: u32,
    ) -> Result<(), String> {
        let source = self.raw_source64(instruction, 0)?;
        // Preserve the complete local portion of old EXEC in its architectural 64-bit position.
        // A split guest wave64 runs as two SIMD32 groups: the first group owns bits 0..31 and the
        // second owns bits 32..63. Keeping masks half-local lets structured EXEC predication and
        // later B64 restore operations remain exact without cross-SIMD communication.
        let old_expression = if self.split_wave64() {
            "((ulong)(uint64_t)simd_ballot(exec) << (remu_wave_half * 32u))"
        } else {
            "(ulong)(uint64_t)simd_ballot(exec)"
        };
        let old = self.temp("ulong", old_expression);
        let expression = match name {
            "SAndSaveexecB64" => format!("({old} & {source})"),
            "SOrSaveexecB64" => format!("({old} | {source})"),
            "SXorSaveexecB64" => format!("({old} ^ {source})"),
            "SAndn1SaveexecB64" => format!("(~{source} & {old})"),
            "SAndn2SaveexecB64" => format!("({source} & ~{old})"),
            "SOrn1SaveexecB64" => format!("(~{source} | {old})"),
            "SOrn2SaveexecB64" => format!("({source} | ~{old})"),
            "SNandSaveexecB64" => format!("~({source} & {old})"),
            "SNorSaveexecB64" => format!("~({source} | {old})"),
            "SXnorSaveexecB64" => format!("~({source} ^ {old})"),
            _ => return Err(format!("unsupported 64-bit saveexec opcode {name}")),
        };
        let architectural_mask = if self.split_wave64() {
            "0xFFFFFFFFFFFFFFFFul"
        } else {
            "0xFFFFFFFFul"
        };
        let new = self.temp("ulong", &format!("({expression}) & {architectural_mask}"));
        self.store_scalar_pair(destination, &old);
        self.line(3, &format!("s[{EXEC_LO}] = (uint){new};"));
        self.line(3, &format!("s[{EXEC_HI}] = (uint)({new} >> 32);"));
        let exec =
            self.lane_mask_predicate(&format!("(uint){new}"), &format!("(uint)({new} >> 32)"));
        self.line(3, &format!("exec = {exec};"));
        self.line(3, &format!("scc = {new} != 0ul;"));
        Ok(())
    }

    fn emit_scalar_compare(
        &mut self,
        instruction: &Instruction,
        name: &str,
        prefix: &str,
    ) -> Result<(), String> {
        let left = self.raw_source(instruction, 0)?;
        let right = self.raw_source(instruction, 1)?;
        self.emit_scalar_compare_values(name, prefix, &left, &right)
    }

    fn emit_scalar_compare_values(
        &mut self,
        name: &str,
        prefix: &str,
        left: &str,
        right: &str,
    ) -> Result<(), String> {
        let suffix = name
            .strip_prefix(prefix)
            .ok_or_else(|| format!("invalid scalar compare name {name}"))?;
        let signed = suffix.ends_with("I32");
        let relation = suffix
            .get(..suffix.len().saturating_sub(3))
            .ok_or_else(|| format!("invalid scalar compare suffix {suffix}"))?;
        let operator = match relation {
            "Eq" => "==",
            "Lg" => "!=",
            "Gt" => ">",
            "Ge" => ">=",
            "Lt" => "<",
            "Le" => "<=",
            _ => return Err(format!("unsupported scalar comparison {name}")),
        };
        if signed {
            self.line(
                3,
                &format!("scc = as_type<int>({left}) {operator} as_type<int>({right});"),
            );
        } else {
            self.line(3, &format!("scc = ({left}) {operator} ({right});"));
        }
        Ok(())
    }

    fn emit_scalar64(
        &mut self,
        instruction: &Instruction,
        name: &str,
        destination: u32,
    ) -> Result<(), String> {
        let left = self.raw_source64(instruction, 0)?;
        let (value, sets_scc) = match name {
            "SMovB64" => (left, false),
            "SNotB64" => (format!("~({left})"), true),
            "SWqmB64" => {
                let quad = self.temp(
                    "ulong",
                    &format!("({left} | ({left} >> 1) | ({left} >> 2) | ({left} >> 3)) & 0x1111111111111111ul"),
                );
                (format!("({quad} * 0xFul)"), true)
            }
            "SLshlB64" | "SLshrB64" => {
                let shift = self.raw_source(instruction, 1)?;
                let operator = if name == "SLshlB64" { "<<" } else { ">>" };
                (format!("({left} {operator} ({shift} & 63u))"), true)
            }
            "SBfeU64" | "SBfeI64" => {
                let control = self.raw_source(instruction, 1)?;
                let offset = self.temp("uint", &format!("{control} & 63u"));
                let width = self.temp(
                    "uint",
                    &format!("min(({control} >> 16) & 0x7Fu, 64u - {offset})"),
                );
                let mask = self.temp(
                    "ulong",
                    &format!("{width} >= 64u ? 0xFFFFFFFFFFFFFFFFul : ((1ul << {width}) - 1ul)"),
                );
                let extracted = format!("({left} >> {offset}) & {mask}");
                if name == "SBfeI64" {
                    (
                        format!(
                            "({width} == 0u || {width} >= 64u) ? {extracted} : (({extracted} ^ (1ul << ({width} - 1u))) - (1ul << ({width} - 1u)))"
                        ),
                        true,
                    )
                } else {
                    (extracted, true)
                }
            }
            "SCselectB64" => {
                let right = self.raw_source64(instruction, 1)?;
                (format!("(scc ? {left} : {right})"), false)
            }
            "SAndB64" | "SOrB64" | "SXorB64" | "SAndn2B64" | "SOrn2B64" | "SNandB64"
            | "SNorB64" | "SXnorB64" => {
                let right = self.raw_source64(instruction, 1)?;
                let expression = match name {
                    "SAndB64" => format!("({left} & {right})"),
                    "SOrB64" => format!("({left} | {right})"),
                    "SXorB64" => format!("({left} ^ {right})"),
                    "SAndn2B64" => format!("({left} & ~{right})"),
                    "SOrn2B64" => format!("({left} | ~{right})"),
                    "SNandB64" => format!("~({left} & {right})"),
                    "SNorB64" => format!("~({left} | {right})"),
                    "SXnorB64" => format!("~({left} ^ {right})"),
                    _ => unreachable!("64-bit bitwise opcode is matched above"),
                };
                (expression, true)
            }
            _ => return Err(format!("unsupported 64-bit scalar opcode {name}")),
        };
        let result = self.temp("ulong", &value);
        self.store_scalar_pair(destination, &result);
        if sets_scc {
            self.line(3, &format!("scc = {result} != 0ul;"));
        }
        Ok(())
    }

    fn emit_scalar_memory(&mut self, instruction: &Instruction) -> Result<(), String> {
        let Some(InstructionControl::ScalarMemory(control)) = instruction.control.as_ref() else {
            return Err("missing scalar-memory control".to_owned());
        };
        let binding_index = self
            .evaluation
            .global_memory_bindings
            .iter()
            .position(|binding| binding.instruction_pcs.contains(&instruction.pc))
            .ok_or_else(|| "scalar load has no evaluated memory binding".to_owned())?;
        let binding_length = self.global_binding_length(binding_index);
        let dynamic = control
            .dynamic_offset_register
            .map_or_else(|| "0u".to_owned(), |register| format!("s[{register}]"));
        let address = self.temp(
            "uint",
            &format!(
                "({dynamic} + {})",
                format_u32(control.immediate_offset_bytes.cast_unsigned())
            ),
        );
        for (index, destination) in instruction.destinations.iter().enumerate() {
            if destination.kind != OperandKind::ScalarRegister {
                return Err("scalar load has a non-scalar destination".to_owned());
            }
            let offset = u32::try_from(index)
                .map_err(|_| "scalar load destination count overflow".to_owned())?
                * 4;
            let expression =
                format!("remu_load_u32(b{binding_index}, {address} + {offset}u, {binding_length})");
            self.store_scalar(destination.value, &expression);
        }
        Ok(())
    }

    #[allow(clippy::similar_names, clippy::too_many_lines)]
    fn emit_buffer_memory(&mut self, instruction: &Instruction) -> Result<(), String> {
        let Some(InstructionControl::BufferMemory(control)) = instruction.control.as_ref() else {
            return Err("missing buffer-memory control".to_owned());
        };
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw buffer opcode".to_owned())?;
        if control.lds && (!name.starts_with("BufferLoad") || control.tfe) {
            return Err("MUBUF LDS transfer is legal only for non-TFE buffer loads".to_owned());
        }
        if control.lds && control.vector_data != 124 {
            return Err("MUBUF LDS transfer must encode M0 in VDATA".to_owned());
        }
        if self.options.stage == MetalShaderStage::Vertex
            && let Some(input) = self
                .evaluation
                .vertex_inputs
                .iter()
                .find(|input| input.pc == instruction.pc)
        {
            if control.dword_count == 0 || control.dword_count > input.component_count {
                return Err(format!(
                    "vertex fetch requests {} components from a {}-component input",
                    control.dword_count, input.component_count
                ));
            }
            for component in 0..control.dword_count {
                let field = if input.component_count == 1 {
                    format!("remu_in.in{}", input.location)
                } else {
                    format!("remu_in.in{}[{component}]", input.location)
                };
                self.store_vector(
                    control.vector_data + component,
                    &format!("as_type<uint>({field})"),
                );
            }
            return Ok(());
        }
        let binding_index = self
            .evaluation
            .global_memory_bindings
            .iter()
            .position(|binding| binding.instruction_pcs.contains(&instruction.pc))
            .ok_or_else(|| format!("{} has no evaluated buffer binding", instruction.opcode))?;
        let binding_length = self.global_binding_length(binding_index);
        let scalar_offset = self.raw_source(instruction, 2)?;
        let vector_index = if self.options.stage == MetalShaderStage::Vertex
            && self.options.diagnostic_force_vertex_fetch_id
            && name.starts_with("BufferLoadFormat")
            && control.index_enabled
        {
            "remu_vertex_id".to_owned()
        } else if control.index_enabled {
            format!("v[{}]", control.vector_address)
        } else {
            "0u".to_owned()
        };
        let vector_offset = if control.offset_enabled {
            format!(
                "v[{}]",
                control.vector_address + u32::from(control.index_enabled)
            )
        } else {
            "0u".to_owned()
        };
        let resource0 = format!("s[{}]", control.scalar_resource);
        let resource1 = format!("s[{}]", control.scalar_resource + 1);
        let resource2 = format!("s[{}]", control.scalar_resource + 2);
        let resource3 = format!("s[{}]", control.scalar_resource + 3);
        let resource_base = self.temp(
            "ulong",
            &format!("(ulong){resource0} | (((ulong){resource1} & 0xFFFFul) << 32)"),
        );
        let resource_bias = self.temp("uint", &format!("(uint)({resource_base} & 3ul)"));
        let stride = self.temp("uint", &format!("({resource1} >> 16) & 0x3FFFu"));
        let swizzled = self.temp("bool", &format!("({resource1} & 0x80000000u) != 0u"));
        let add_tid = self.temp("bool", &format!("({resource3} & 0x00800000u) != 0u"));
        let index = self.temp(
            "uint",
            &format!(
                "{vector_index} + ({add_tid} ? ({} & 63u) : 0u)",
                self.architectural_lane()
            ),
        );
        let offset = self.temp(
            "uint",
            &format!(
                "{vector_offset} + {}",
                format_u32(control.offset_bytes.cast_unsigned())
            ),
        );
        let index_stride = self.temp("uint", &format!("8u << (({resource3} >> 21) & 3u)"));
        let element_size = "4u";
        let index_msb = self.temp("uint", &format!("{index} / {index_stride}"));
        let index_lsb = self.temp("uint", &format!("{index} % {index_stride}"));
        let offset_msb = self.temp("uint", &format!("{offset} / {element_size}"));
        let offset_lsb = self.temp("uint", &format!("{offset} % {element_size}"));
        let linear_offset = self.temp("uint", &format!("{index} * {stride} + {offset}"));
        let swizzled_offset = self.temp(
            "uint",
            &format!(
                "({index_msb} * {stride} + {offset_msb} * {element_size}) * {index_stride} + {index_lsb} * {element_size} + {offset_lsb}"
            ),
        );
        let buffer_offset = self.temp(
            "uint",
            &format!("{swizzled} ? {swizzled_offset} : {linear_offset}"),
        );
        let address = self.temp(
            "uint",
            &format!("{resource_bias} + {scalar_offset} + {buffer_offset}"),
        );
        let payload = if is_format_buffer_memory(instruction) {
            let unified = control
                .instruction_format
                .map_or_else(|| format!("({resource3} >> 12) & 0x7Fu"), format_u32);
            let table = self.temp("uint", &format!("remu_gfx10_formats[{unified}]"));
            let data_format = self.temp("uint", &format!("{table} & 0xFFu"));
            self.temp("uint", &format!("remu_format_size({data_format})"))
        } else {
            format_u32(memory_payload_bytes(name, control.dword_count)?)
        };
        let record_count = self.temp("uint", &resource2);
        let oob_select = self.temp("uint", &format!("({resource3} >> 28) & 3u"));
        let descriptor_null = self.temp(
            "bool",
            &format!("({resource0} | {resource1} | {resource2} | {resource3}) == 0u"),
        );
        let reduced_records = self.temp(
            "uint",
            &format!("{record_count} > {scalar_offset} ? {record_count} - {scalar_offset} : 0u"),
        );
        let oob = self.temp(
            "bool",
            &buffer_oob_expression(
                &descriptor_null,
                &oob_select,
                &index,
                &record_count,
                &offset,
                &stride,
                &swizzled,
                &buffer_offset,
                &reduced_records,
                &payload,
                0,
            ),
        );
        if is_format_buffer_memory(instruction) {
            return self.emit_format_buffer_memory(
                instruction,
                control,
                binding_index,
                &binding_length,
                &address,
                &oob,
            );
        }
        let component_oobs = (control.dword_count > 1
            && name.contains("Dword")
            && !name.contains("Atomic"))
        .then(|| {
            (0..control.dword_count)
                .map(|component| {
                    let expression = buffer_oob_expression(
                        &descriptor_null,
                        &oob_select,
                        &index,
                        &record_count,
                        &offset,
                        &stride,
                        &swizzled,
                        &buffer_offset,
                        &reduced_records,
                        "4u",
                        component * 4,
                    );
                    self.temp("bool", &expression)
                })
                .collect::<Vec<_>>()
        });
        self.emit_raw_memory_operation(
            name,
            control.vector_data,
            control.vector_data,
            control.dword_count,
            binding_index,
            &binding_length,
            &address,
            &oob,
            component_oobs.as_deref(),
            control.glc,
            control.lds,
            control.tfe,
        )
    }

    fn emit_global_memory(&mut self, instruction: &Instruction) -> Result<(), String> {
        let Some(InstructionControl::GlobalMemory(control)) = instruction.control.as_ref() else {
            return Err("missing global-memory control".to_owned());
        };
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw global opcode".to_owned())?;
        let guest_address = if let Some(scalar_address) = control.scalar_address {
            self.temp(
                "ulong",
                &format!(
                    "((ulong)s[{}] | ((ulong)s[{}] << 32)) + (ulong)v[{}] + (ulong)(long){}",
                    scalar_address,
                    scalar_address + 1,
                    control.vector_address,
                    control.offset_bytes
                ),
            )
        } else {
            self.temp(
                "ulong",
                &format!(
                    "((ulong)v[{}] | ((ulong)v[{}] << 32)) + (ulong)(long){}",
                    control.vector_address,
                    control.vector_address + 1,
                    control.offset_bytes
                ),
            )
        };
        let binding_index = self
            .evaluation
            .global_memory_bindings
            .iter()
            .position(|binding| binding.instruction_pcs.contains(&instruction.pc))
            .or_else(|| {
                (control.scalar_address.is_none()
                    && self.evaluation.global_memory_bindings.len() == 1)
                    .then_some(0)
            })
            .ok_or_else(|| {
                "GLOBAL with a VGPR-only address has no statically resolved guest memory range"
                    .to_owned()
            })?;
        let binding_base = self.temp("ulong", &self.global_binding_base(binding_index));
        let binding_length = self.global_binding_length(binding_index);
        let local_address = self.temp("uint", &format!("(uint)({guest_address} - {binding_base})"));
        let before = self.temp("bool", &format!("{guest_address} < {binding_base}"));
        let oob = self.temp("bool", &before);
        self.emit_raw_memory_operation(
            name,
            control.vector_data,
            control.vector_destination,
            control.dword_count,
            binding_index,
            &binding_length,
            &local_address,
            &oob,
            None,
            control.glc,
            control.lds,
            false,
        )
    }

    #[allow(clippy::too_many_lines)]
    fn emit_data_share(&mut self, instruction: &Instruction) -> Result<(), String> {
        let Some(InstructionControl::DataShare(control)) = instruction.control.as_ref() else {
            return Err("missing data-share control".to_owned());
        };
        if control.gds {
            return Err("GDS operations require a system-wide data-share resource".to_owned());
        }
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw data-share opcode".to_owned())?;
        if name == "DsSwizzleB32" {
            return self.emit_ds_swizzle(instruction, control);
        }
        if self.options.stage != MetalShaderStage::Compute {
            return Err(
                "LDS memory is workgroup-shared and cannot be represented by a Metal graphics stage"
                    .to_owned(),
            );
        }
        if self.options.lds_size_bytes == 0 || self.options.lds_size_bytes & 3 != 0 {
            return Err(
                "LDS memory operations require a non-zero dword-aligned allocation size".to_owned(),
            );
        }
        if self.options.lds_size_bytes > 32 * 1024 {
            return Err(format!(
                "{} bytes of guest LDS exceed Apple Metal's 32768-byte threadgroup-memory limit",
                self.options.lds_size_bytes
            ));
        }

        match name {
            "DsWriteB32" => self.emit_ds_write(instruction, control, 1),
            "DsWriteB64" => self.emit_ds_write(instruction, control, 2),
            "DsWriteB96" => self.emit_ds_write(instruction, control, 3),
            "DsWriteB128" => self.emit_ds_write(instruction, control, 4),
            "DsWrite2B32" => self.emit_ds_write_pair(instruction, control, false),
            "DsWrite2St64B32" => self.emit_ds_write_pair(instruction, control, true),
            "DsReadB32" => self.emit_ds_read(instruction, control, 1),
            "DsReadB96" => self.emit_ds_read(instruction, control, 3),
            "DsReadB128" => self.emit_ds_read(instruction, control, 4),
            "DsRead2B32" => self.emit_ds_read_pair(instruction, control, false),
            "DsRead2St64B32" => self.emit_ds_read_pair(instruction, control, true),
            _ if is_lds_atomic_name(name) => self.emit_ds_atomic(instruction, control, name),
            _ => Err(format!("unsupported LDS opcode {name}")),
        }
    }

    fn emit_ds_write(
        &mut self,
        instruction: &Instruction,
        control: &DataShareControl,
        dword_count: usize,
    ) -> Result<(), String> {
        if instruction.sources.len() < dword_count + 1 {
            return Err(format!("LDS write requires {dword_count} data dwords"));
        }
        let base = self.temp("uint", &self.raw_source(instruction, 0)?);
        let first_offset = ds_combined_offset(control);
        let mut values = Vec::with_capacity(dword_count);
        for component in 0..dword_count {
            values.push(self.temp("uint", &self.raw_source(instruction, component + 1)?));
        }
        for (component, value) in values.iter().enumerate() {
            let component_offset = u32::try_from(component)
                .map_err(|_| "LDS component index overflow".to_owned())?
                * 4;
            let address = self.ds_address(&base, first_offset + component_offset);
            let index = self.temp("uint", &format!("{address} >> 2"));
            self.emit_ds_store_index(&index, value);
        }
        Ok(())
    }

    fn emit_ds_write_pair(
        &mut self,
        instruction: &Instruction,
        control: &DataShareControl,
        stride64: bool,
    ) -> Result<(), String> {
        if instruction.sources.len() < 3 {
            return Err("LDS write2 requires two data dwords".to_owned());
        }
        let base = self.temp("uint", &self.raw_source(instruction, 0)?);
        let first = self.temp("uint", &self.raw_source(instruction, 1)?);
        let second = self.temp("uint", &self.raw_source(instruction, 2)?);
        let scale = if stride64 { 4 * 64 } else { 4 };
        for (offset, value) in [
            (control.offset0 * scale, first),
            (control.offset1 * scale, second),
        ] {
            let address = self.ds_address(&base, offset);
            let index = self.temp("uint", &format!("{address} >> 2"));
            self.emit_ds_store_index(&index, &value);
        }
        Ok(())
    }

    fn emit_ds_read(
        &mut self,
        instruction: &Instruction,
        control: &DataShareControl,
        dword_count: usize,
    ) -> Result<(), String> {
        if instruction.destinations.len() < dword_count {
            return Err(format!(
                "LDS read requires {dword_count} destination dwords"
            ));
        }
        let base = self.temp("uint", &self.raw_source(instruction, 0)?);
        let first_offset = ds_combined_offset(control);
        let mut values = Vec::with_capacity(dword_count);
        for component in 0..dword_count {
            let component_offset = u32::try_from(component)
                .map_err(|_| "LDS component index overflow".to_owned())?
                * 4;
            let address = self.ds_address(&base, first_offset + component_offset);
            let index = self.temp("uint", &format!("{address} >> 2"));
            values.push(self.emit_ds_load_index(&index));
        }
        for (destination, value) in instruction.destinations.iter().zip(values) {
            self.store_vector(vector_operand(*destination)?, &value);
        }
        Ok(())
    }

    fn emit_ds_read_pair(
        &mut self,
        instruction: &Instruction,
        control: &DataShareControl,
        stride64: bool,
    ) -> Result<(), String> {
        if instruction.destinations.len() < 2 {
            return Err("LDS read2 requires two destination dwords".to_owned());
        }
        let base = self.temp("uint", &self.raw_source(instruction, 0)?);
        let scale = if stride64 { 4 * 64 } else { 4 };
        let mut values = Vec::with_capacity(2);
        for offset in [control.offset0 * scale, control.offset1 * scale] {
            let address = self.ds_address(&base, offset);
            let index = self.temp("uint", &format!("{address} >> 2"));
            values.push(self.emit_ds_load_index(&index));
        }
        for (destination, value) in instruction.destinations.iter().zip(values) {
            self.store_vector(vector_operand(*destination)?, &value);
        }
        Ok(())
    }

    fn emit_ds_atomic(
        &mut self,
        instruction: &Instruction,
        control: &DataShareControl,
        name: &str,
    ) -> Result<(), String> {
        let required_sources = if matches!(name, "DsCmpstB32" | "DsCmpstRtnB32") {
            3
        } else {
            2
        };
        if instruction.sources.len() < required_sources {
            return Err(format!("{name} has missing atomic operands"));
        }
        let base = self.temp("uint", &self.raw_source(instruction, 0)?);
        let address = self.ds_address(&base, ds_combined_offset(control));
        let index = self.temp("uint", &format!("{address} >> 2"));
        let data0 = self.temp("uint", &self.raw_source(instruction, 1)?);
        let data1 = if required_sources == 3 {
            Some(self.temp("uint", &self.raw_source(instruction, 2)?))
        } else {
            None
        };
        let original = self.temp("uint", "0u");
        let pointer = format!("remu_lds + {index}");
        let operation = match name {
            "DsAddU32" | "DsAddRtnU32" => {
                format!("atomic_fetch_add_explicit({pointer}, {data0}, memory_order_relaxed)")
            }
            "DsSubU32" | "DsSubRtnU32" => {
                format!("atomic_fetch_sub_explicit({pointer}, {data0}, memory_order_relaxed)")
            }
            "DsIncU32" | "DsIncRtnU32" => format!("remu_ds_inc({pointer}, {data0})"),
            "DsDecU32" | "DsDecRtnU32" => format!("remu_ds_dec({pointer}, {data0})"),
            "DsMinI32" | "DsMinRtnI32" => format!("remu_ds_smin({pointer}, {data0})"),
            "DsMaxI32" | "DsMaxRtnI32" => format!("remu_ds_smax({pointer}, {data0})"),
            "DsMinU32" | "DsMinRtnU32" => {
                format!("atomic_fetch_min_explicit({pointer}, {data0}, memory_order_relaxed)")
            }
            "DsMaxU32" | "DsMaxRtnU32" => {
                format!("atomic_fetch_max_explicit({pointer}, {data0}, memory_order_relaxed)")
            }
            "DsAndB32" | "DsAndRtnB32" => {
                format!("atomic_fetch_and_explicit({pointer}, {data0}, memory_order_relaxed)")
            }
            "DsOrB32" | "DsOrRtnB32" => {
                format!("atomic_fetch_or_explicit({pointer}, {data0}, memory_order_relaxed)")
            }
            "DsXorB32" | "DsXorRtnB32" => {
                format!("atomic_fetch_xor_explicit({pointer}, {data0}, memory_order_relaxed)")
            }
            "DsWrxchgRtnB32" => {
                format!("atomic_exchange_explicit({pointer}, {data0}, memory_order_relaxed)")
            }
            "DsCmpstB32" | "DsCmpstRtnB32" => format!(
                "remu_ds_cmpst({pointer}, {data0}, {})",
                data1.expect("compare/store requires DATA1")
            ),
            _ => return Err(format!("unsupported LDS atomic opcode {name}")),
        };
        self.line(
            3,
            &format!(
                "if (exec && {index} < {}u && {index} <= (s[124] >> 2)) {original} = {operation};",
                self.options.lds_size_bytes / 4
            ),
        );
        if name.contains("Rtn") {
            let destination = instruction
                .destinations
                .first()
                .copied()
                .ok_or_else(|| format!("{name} has no return destination"))?;
            self.store_vector(vector_operand(destination)?, &original);
        }
        Ok(())
    }

    fn emit_ds_swizzle(
        &mut self,
        instruction: &Instruction,
        control: &DataShareControl,
    ) -> Result<(), String> {
        if instruction.sources.len() != 1 || instruction.destinations.len() != 1 {
            return Err("DS_SWIZZLE_B32 requires one source and one destination".to_owned());
        }
        let source = vector_operand(instruction.sources[0])?;
        let destination = vector_operand(instruction.destinations[0])?;
        let offset = u16::try_from(ds_combined_offset(control))
            .expect("two 8-bit data-share offsets fit u16");
        let source_lane = self.temp("uint", &ds_swizzle_lane_expression(offset));
        let shuffled = self.temp(
            "uint",
            &format!("simd_shuffle(v[{source}], (ushort)({source_lane} & 31u))"),
        );
        let source_active = self.temp(
            "uint",
            &format!("simd_shuffle(exec ? 1u : 0u, (ushort)({source_lane} & 31u))"),
        );
        self.store_vector(
            destination,
            &format!("{source_active} != 0u ? {shuffled} : 0u"),
        );
        Ok(())
    }

    fn ds_address(&mut self, base: &str, offset_bytes: u32) -> String {
        if offset_bytes == 0 {
            base.to_owned()
        } else {
            self.temp("uint", &format!("{base} + {}", format_u32(offset_bytes)))
        }
    }

    fn emit_ds_load_index(&mut self, index: &str) -> String {
        let value = self.temp("uint", "0u");
        self.line(
            3,
            &format!(
                "if (exec && {index} < {}u && {index} <= (s[124] >> 2)) {value} = atomic_load_explicit(remu_lds + {index}, memory_order_relaxed);",
                self.options.lds_size_bytes / 4
            ),
        );
        value
    }

    fn emit_ds_store_index(&mut self, index: &str, value: &str) {
        self.line(
            3,
            &format!(
                "if (exec && {index} < {}u && {index} <= (s[124] >> 2)) atomic_store_explicit(remu_lds + {index}, {value}, memory_order_relaxed);",
                self.options.lds_size_bytes / 4
            ),
        );
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn emit_raw_memory_operation(
        &mut self,
        name: &str,
        vector_data: u32,
        vector_destination: u32,
        dword_count: u32,
        binding_index: usize,
        binding_length: &str,
        address: &str,
        descriptor_oob: &str,
        component_descriptor_oobs: Option<&[String]>,
        glc: bool,
        lds: bool,
        tfe: bool,
    ) -> Result<(), String> {
        let load = name.contains("Load");
        let store = name.contains("Store");
        let atomic = name.contains("Atomic");
        if !load && !store && !atomic {
            return Err(format!("unsupported memory opcode {name}"));
        }
        let subdword = name.contains("Byte")
            || name.contains("Ubyte")
            || name.contains("Sbyte")
            || name.contains("Short")
            || name.contains("Ushort")
            || name.contains("Sshort");
        let aligned_address = if subdword {
            address.to_owned()
        } else {
            self.temp("uint", &format!("{address} & ~3u"))
        };

        if load {
            let (width, signed, d16, high) = raw_load_shape(name)?;
            if lds {
                if self.options.stage != MetalShaderStage::Compute {
                    return Err("memory-to-LDS transfers require a Metal compute stage".to_owned());
                }
                if dword_count != 1 || tfe {
                    return Err(
                        "memory-to-LDS transfer must return exactly one non-TFE dword".to_owned(),
                    );
                }
            }
            for component in 0..dword_count {
                let descriptor_oob = component_descriptor_oobs
                    .and_then(|bounds| bounds.get(component as usize))
                    .map_or(descriptor_oob, String::as_str);
                let component_address = if component == 0 {
                    aligned_address.clone()
                } else {
                    format!("{aligned_address} + {}u", component * 4)
                };
                let loaded = match width {
                    1 => format!(
                        "remu_load_u8(b{binding_index}, {component_address}, {binding_length})"
                    ),
                    2 => format!(
                        "remu_load_u16(b{binding_index}, {component_address}, {binding_length})"
                    ),
                    4 => format!(
                        "remu_load_u32(b{binding_index}, {component_address}, {binding_length})"
                    ),
                    _ => unreachable!("raw load widths are fixed"),
                };
                let in_range = format!(
                    "!{descriptor_oob} && remu_range_valid({component_address}, {width}u, {binding_length})"
                );
                let raw = self.temp("uint", &format!("{in_range} ? {loaded} : 0u"));
                let converted = if signed && width == 1 {
                    format!("({raw} & 0x80u) != 0u ? ({raw} | 0xFFFFFF00u) : {raw}")
                } else if signed && width == 2 {
                    format!("({raw} & 0x8000u) != 0u ? ({raw} | 0xFFFF0000u) : {raw}")
                } else {
                    raw.clone()
                };
                let value = self.temp("uint", &converted);
                if lds {
                    let lds_index = self.temp("uint", "((s[124] & 0xFFFFu) >> 2) + remu_lane");
                    self.line(
                        3,
                        &format!(
                            "if (exec && {lds_index} < {}u) atomic_store_explicit(remu_lds + {lds_index}, {value}, memory_order_relaxed);",
                            self.options.lds_size_bytes / 4
                        ),
                    );
                } else if d16 {
                    let register = vector_destination + component;
                    let merged = if high {
                        format!("(v[{register}] & 0xFFFFu) | (({value} & 0xFFFFu) << 16)")
                    } else {
                        format!("(v[{register}] & 0xFFFF0000u) | ({value} & 0xFFFFu)")
                    };
                    self.store_vector(register, &merged);
                } else {
                    self.store_vector(vector_destination + component, &value);
                }
            }
            if tfe {
                self.store_vector(vector_destination + dword_count, "0u");
            }
            return Ok(());
        }

        if store {
            let (width, high) = raw_store_shape(name)?;
            for component in 0..dword_count {
                let descriptor_oob = component_descriptor_oobs
                    .and_then(|bounds| bounds.get(component as usize))
                    .map_or(descriptor_oob, String::as_str);
                let component_address = if component == 0 {
                    aligned_address.clone()
                } else {
                    format!("{aligned_address} + {}u", component * 4)
                };
                let source = format!("v[{}]", vector_data + component);
                let value = if high {
                    format!("{source} >> 16")
                } else {
                    source
                };
                self.line(
                    3,
                    &format!(
                        "if (exec && !{descriptor_oob} && remu_range_valid({component_address}, {width}u, {binding_length})) remu_store_u{bits}(b{binding_index}, {component_address}, {value});",
                        bits = width * 8
                    ),
                );
            }
            return Ok(());
        }

        let atomic_address = aligned_address;
        let old = self.temp("uint", "0u");
        let source = format!("v[{vector_data}]");
        let operation = match name
            .strip_prefix("BufferAtomic")
            .or_else(|| name.strip_prefix("GlobalAtomic"))
        {
            Some("Swap") => {
                format!("remu_atomic_swap(b{binding_index}, {atomic_address}, {source})")
            }
            Some("Cmpswap") => format!(
                "remu_atomic_cmpswap(b{binding_index}, {atomic_address}, {source}, v[{}])",
                vector_data + 1
            ),
            Some("Add") => format!("remu_atomic_add(b{binding_index}, {atomic_address}, {source})"),
            Some("Sub") => format!("remu_atomic_sub(b{binding_index}, {atomic_address}, {source})"),
            Some("Smin") => {
                format!("remu_atomic_smin(b{binding_index}, {atomic_address}, {source})")
            }
            Some("Umin") => {
                format!("remu_atomic_umin(b{binding_index}, {atomic_address}, {source})")
            }
            Some("Smax") => {
                format!("remu_atomic_smax(b{binding_index}, {atomic_address}, {source})")
            }
            Some("Umax" | "UMax") => {
                format!("remu_atomic_umax(b{binding_index}, {atomic_address}, {source})")
            }
            Some("And") => format!("remu_atomic_and(b{binding_index}, {atomic_address}, {source})"),
            Some("Or") => format!("remu_atomic_or(b{binding_index}, {atomic_address}, {source})"),
            Some("Xor") => format!("remu_atomic_xor(b{binding_index}, {atomic_address}, {source})"),
            Some("Inc") => format!("remu_atomic_inc(b{binding_index}, {atomic_address}, {source})"),
            Some("Dec") => format!("remu_atomic_dec(b{binding_index}, {atomic_address}, {source})"),
            _ => return Err(format!("unsupported atomic opcode {name}")),
        };
        self.line(
            3,
            &format!(
                "if (exec && !{descriptor_oob} && remu_range_valid({atomic_address}, 4u, {binding_length})) {old} = {operation};"
            ),
        );
        if glc {
            self.store_vector(vector_destination, &old);
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn emit_format_buffer_memory(
        &mut self,
        instruction: &Instruction,
        control: &remu_shader::BufferMemoryControl,
        binding_index: usize,
        binding_length: &str,
        address: &str,
        descriptor_oob: &str,
    ) -> Result<(), String> {
        if control.dword_count == 0 || control.dword_count > 4 {
            return Err(format!(
                "format load has invalid component count {}",
                control.dword_count
            ));
        }
        let descriptor_word = self.temp("uint", &format!("s[{}]", control.scalar_resource + 3));
        let unified_format = control
            .instruction_format
            .map_or_else(|| format!("({descriptor_word} >> 12) & 0x7Fu"), format_u32);
        let table_entry = self.temp("uint", &format!("remu_gfx10_formats[{unified_format}]"));
        let data_format = self.temp("uint", &format!("{table_entry} & 0xFFu"));
        let number_format = self.temp("uint", &format!("({table_entry} >> 8) & 0xFFu"));
        let element_size = self.temp("uint", &format!("remu_format_size({data_format})"));
        let oob = self.temp(
            "bool",
            &format_buffer_element_oob_expression(
                descriptor_oob,
                address,
                &element_size,
                binding_length,
            ),
        );
        if instruction.opcode.contains("Store") {
            return self.emit_format_buffer_store(
                instruction,
                control,
                binding_index,
                address,
                &descriptor_word,
                &data_format,
                &number_format,
                &oob,
            );
        }
        let mut canonical = Vec::with_capacity(4);
        for component in 0..4 {
            let byte_offset = self.temp("uint", "0u");
            let bit_offset = self.temp("uint", "0u");
            let bits = self.temp("uint", "0u");
            self.line(
                3,
                &format!(
                    "remu_format_layout({data_format}, {component}u, {byte_offset}, {bit_offset}, {bits});"
                ),
            );
            let raw = self.temp(
                "uint",
                &format!(
                    "{oob} || {bits} == 0u ? 0u : remu_load_format_bits(b{binding_index}, {address}, {byte_offset}, {bit_offset}, {bits})"
                ),
            );
            let missing = if component == 3 {
                format!("remu_format_one({number_format})")
            } else {
                "0u".to_owned()
            };
            canonical.push(self.temp(
                "uint",
                &format!(
                    "{bits} == 0u ? {missing} : remu_format_convert({raw}, {bits}, {number_format}, {data_format})"
                ),
            ));
        }
        let component_count = format_component_count(
            instruction
                .opcode
                .known_name()
                .ok_or_else(|| "raw format opcode".to_owned())?,
        );
        let mut selected = Vec::with_capacity(component_count as usize);
        for destination in 0..component_count {
            let selector = if control.instruction_format.is_some() {
                format_u32(destination + 4)
            } else {
                self.temp(
                    "uint",
                    &format!("({descriptor_word} >> {}u) & 7u", destination * 3),
                )
            };
            selected.push(self.temp(
                "uint",
                &format!(
                    "{selector} == 1u ? remu_format_one({number_format}) : {oob} ? 0u : {selector} == 4u ? {} : {selector} == 5u ? {} : {selector} == 6u ? {} : {selector} == 7u ? {} : 0u",
                    canonical[0], canonical[1], canonical[2], canonical[3]
                ),
            ));
        }
        let d16 = instruction.opcode.contains("FormatD16");
        let d16_hi = instruction.opcode.contains("FormatD16Hi");
        if control.lds {
            if self.options.stage != MetalShaderStage::Compute
                || component_count != 1
                || d16
                || control.tfe
            {
                return Err(
                    "formatted memory-to-LDS requires a non-D16 FORMAT_X compute load".to_owned(),
                );
            }
            let lds_index = self.temp("uint", "((s[124] & 0xFFFFu) >> 2) + remu_lane");
            self.line(
                3,
                &format!(
                    "if (exec && {lds_index} < {}u) atomic_store_explicit(remu_lds + {lds_index}, {}, memory_order_relaxed);",
                    self.options.lds_size_bytes / 4,
                    selected[0]
                ),
            );
        } else if d16_hi {
            let converted = self.temp(
                "uint",
                &format!("remu_format_to_d16({}, {number_format})", selected[0]),
            );
            self.store_vector(
                control.vector_data,
                &format!(
                    "(v[{}] & 0xFFFFu) | ({converted} << 16)",
                    control.vector_data
                ),
            );
        } else if d16 {
            for register in 0..control.dword_count {
                let low_index = register * 2;
                let low = self.temp(
                    "uint",
                    &format!(
                        "remu_format_to_d16({}, {number_format})",
                        selected[low_index as usize]
                    ),
                );
                let expression = if low_index + 1 < component_count {
                    let high = self.temp(
                        "uint",
                        &format!(
                            "remu_format_to_d16({}, {number_format})",
                            selected[(low_index + 1) as usize]
                        ),
                    );
                    format!("{low} | ({high} << 16)")
                } else {
                    format!(
                        "(v[{}] & 0xFFFF0000u) | {low}",
                        control.vector_data + register
                    )
                };
                self.store_vector(control.vector_data + register, &expression);
            }
        } else {
            for (destination, value) in selected.iter().enumerate() {
                self.store_vector(
                    control.vector_data
                        + u32::try_from(destination)
                            .map_err(|_| "format destination overflow".to_owned())?,
                    value,
                );
            }
        }
        if control.tfe {
            self.store_vector(control.vector_data + control.dword_count, "0u");
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_format_buffer_store(
        &mut self,
        instruction: &Instruction,
        control: &remu_shader::BufferMemoryControl,
        binding_index: usize,
        address: &str,
        descriptor_word: &str,
        data_format: &str,
        number_format: &str,
        oob: &str,
    ) -> Result<(), String> {
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw format store opcode".to_owned())?;
        let component_count = format_component_count(name);
        let d16 = name.contains("FormatD16");
        let d16_hi = name.contains("FormatD16Hi");
        for memory_component in 0..4_u32 {
            let byte_offset = self.temp("uint", "0u");
            let bit_offset = self.temp("uint", "0u");
            let bits = self.temp("uint", "0u");
            self.line(
                3,
                &format!(
                    "remu_format_layout({data_format}, {memory_component}u, {byte_offset}, {bit_offset}, {bits});"
                ),
            );
            let enabled = self.temp("bool", "false");
            let canonical = self.temp("uint", "0u");
            if control.instruction_format.is_some() {
                if memory_component < component_count {
                    let register = control.vector_data + memory_component / if d16 { 2 } else { 1 };
                    let source = if d16_hi {
                        format!("v[{register}] >> 16")
                    } else if d16 {
                        format!("v[{register}] >> {}u", (memory_component & 1) * 16)
                    } else {
                        format!("v[{register}]")
                    };
                    let expanded = if d16 {
                        format!("remu_format_from_d16({source} & 0xFFFFu, {number_format})")
                    } else {
                        source
                    };
                    self.line(3, &format!("{enabled} = true; {canonical} = {expanded};"));
                }
            } else {
                for source_component in 0..component_count {
                    let selector = self.temp(
                        "uint",
                        &format!("({descriptor_word} >> {}u) & 7u", source_component * 3),
                    );
                    let register = control.vector_data + source_component / if d16 { 2 } else { 1 };
                    let source = if d16_hi {
                        format!("v[{register}] >> 16")
                    } else if d16 {
                        format!("v[{register}] >> {}u", (source_component & 1) * 16)
                    } else {
                        format!("v[{register}]")
                    };
                    let expanded = if d16 {
                        format!("remu_format_from_d16({source} & 0xFFFFu, {number_format})")
                    } else {
                        source
                    };
                    self.line(
                        3,
                        &format!(
                            "if ({selector} == {}u) {{ {enabled} = true; {canonical} = {expanded}; }}",
                            memory_component + 4
                        ),
                    );
                }
            }
            let packed = self.temp(
                "uint",
                &format!("remu_format_pack({canonical}, {bits}, {number_format}, {data_format})"),
            );
            self.line(
                3,
                &format!(
                    "if (exec && !{oob} && {enabled} && {bits} != 0u) remu_store_format_bits(b{binding_index}, {address}, {byte_offset}, {bit_offset}, {bits}, {packed});"
                ),
            );
        }
        Ok(())
    }

    fn emit_image(&mut self, instruction: &Instruction) -> Result<(), String> {
        let Some(InstructionControl::Image(image)) = instruction.control.as_ref() else {
            return Err("missing image control".to_owned());
        };
        let binding_index = self
            .evaluation
            .image_bindings
            .iter()
            .position(|binding| binding.pc == instruction.pc)
            .ok_or_else(|| "image instruction has no evaluated image binding".to_owned())?;
        let kind = self.image_kinds[binding_index];
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw image opcode".to_owned())?;
        if name.starts_with("ImageLoad") {
            return self.emit_image_load(image, name, binding_index);
        }
        if name.starts_with("ImageStore") {
            return self.emit_image_store(image, name, binding_index);
        }
        if !name.starts_with("ImageSample") {
            return Err(format!("unsupported image opcode {name}"));
        }
        if kind.storage {
            return Err("sample operation resolved to a storage image".to_owned());
        }
        if image.is_array || !matches!(image.dimension, 0 | 1 | 2 | 3) {
            return Err(format!(
                "sampled image dimension {} array={} requires its matching Metal texture type",
                image.dimension, image.is_array
            ));
        }
        let sampler_slot = kind
            .sampler_slot
            .ok_or_else(|| "sampled image has no sampler slot".to_owned())?;
        let sampled = self.emit_image_sample(image, name, binding_index, sampler_slot)?;
        let mut output = 0_u32;
        for component in 0..4 {
            if image.dmask & (1 << component) == 0 {
                continue;
            }
            let raw = if kind.component == "uint" {
                format!("{sampled}[{component}]")
            } else {
                format!("as_type<uint>({sampled}[{component}])")
            };
            self.store_vector(image.vector_data + output, &raw);
            output += 1;
        }
        Ok(())
    }

    fn image_destination_select(&mut self, image: &ImageControl) -> String {
        let descriptor_word = format!("s[{}] & 0xFFFu", image.scalar_resource + 3);
        // A zeroed descriptor is used by a few partially initialized Unity paths; retain the
        // architectural XYZW identity fallback while keeping real component selection dynamic.
        // The latter prevents render-material swizzles from creating new MSL functions.
        self.temp(
            "uint",
            &format!("({descriptor_word}) == 0u ? 0xFACu : ({descriptor_word})"),
        )
    }

    fn emit_image_store(
        &mut self,
        image: &ImageControl,
        _name: &str,
        binding_index: usize,
    ) -> Result<(), String> {
        // GFX10 DIM still describes the address operand layout for storage operations, but a
        // descriptor-backed storage view remains a 2D resource. In particular Unity emits DIM=5
        // for its single-layer stores; the third address component must not change the Metal view.
        let image_kind = self.image_kinds[binding_index];
        let is_3d = image_kind.texture_type == 10;
        if !(matches!(image.dimension, 0 | 1 | 4 | 5) || (is_3d && image.dimension == 2)) {
            return Err(format!(
                "storage image dimension {} requires a non-2D Metal texture type",
                image.dimension
            ));
        }
        let raw_x = self.image_uint_address(image, 0);
        let raw_y = if matches!(image.dimension, 0 | 4) {
            "0u".to_owned()
        } else {
            self.image_uint_address(image, 1)
        };
        let x = self.temp("int", &format!("as_type<int>({raw_x})"));
        let y = self.temp("int", &format!("as_type<int>({raw_y})"));
        let z = is_3d.then(|| {
            let raw_z = self.image_uint_address(image, 2);
            self.temp("int", &format!("as_type<int>({raw_z})"))
        });
        let kind = image_kind.component;
        let destination_select = self.image_destination_select(image);
        let mut logical_components = Vec::with_capacity(4);
        let mut input = 0_u32;
        for component in 0..4 {
            if image.dmask & (1 << component) == 0 {
                logical_components.push(if kind == "float" { "0.0f" } else { "0" }.to_owned());
                continue;
            }
            let raw = self.image_store_raw_component(image, kind, input);
            logical_components.push(match kind {
                "uint" => raw,
                "int" => format!("as_type<int>({raw})"),
                "float" => format!("as_type<float>({raw})"),
                _ => unreachable!("image component type is selected from a closed set"),
            });
            input += 1;
        }
        let zero = if kind == "float" { "0.0f" } else { "0" };
        let physical_components = std::array::from_fn::<_, 4, _>(|physical| {
            let physical_selector = 4 + physical;
            (0..4).rev().fold(zero.to_owned(), |fallback, logical| {
                let shift = logical * 3;
                format!(
                    "((({destination_select} >> {shift}u) & 7u) == {physical_selector}u ? {} : {fallback})",
                    logical_components[logical]
                )
            })
        });
        let z_bounds = z.as_ref().map_or_else(String::new, |z| {
            format!(" && {z} >= 0 && {z} < (int)tex{binding_index}.get_depth()")
        });
        let coordinates = z.as_ref().map_or_else(
            || format!("uint2((uint){x}, (uint){y})"),
            |z| format!("uint3((uint){x}, (uint){y}, (uint){z})"),
        );
        self.line(3, &format!("if (exec && {x} >= 0 && {y} >= 0 && {x} < (int)tex{binding_index}.get_width() && {y} < (int)tex{binding_index}.get_height(){z_bounds}) {{"));
        self.line(
            4,
            &format!(
                "tex{binding_index}.write(vec<{kind}, 4>({}, {}, {}, {}), {coordinates});",
                physical_components[0],
                physical_components[1],
                physical_components[2],
                physical_components[3]
            ),
        );
        self.line(3, "}");
        Ok(())
    }

    fn emit_image_load(
        &mut self,
        image: &ImageControl,
        _name: &str,
        binding_index: usize,
    ) -> Result<(), String> {
        let image_kind = self.image_kinds[binding_index];
        let is_3d = image_kind.texture_type == 10;
        if !(matches!(image.dimension, 0 | 1 | 4 | 5) || (is_3d && image.dimension == 2)) {
            return Err(format!(
                "storage image dimension {} requires a non-2D Metal texture type",
                image.dimension
            ));
        }
        let raw_x = self.image_uint_address(image, 0);
        let raw_y = if matches!(image.dimension, 0 | 4) {
            "0u".to_owned()
        } else {
            self.image_uint_address(image, 1)
        };
        let x = self.temp(
            "uint",
            &format!(
                "(uint)clamp(as_type<int>({raw_x}), 0, (int)tex{binding_index}.get_width() - 1)"
            ),
        );
        let y = self.temp(
            "uint",
            &format!(
                "(uint)clamp(as_type<int>({raw_y}), 0, (int)tex{binding_index}.get_height() - 1)"
            ),
        );
        let z = is_3d.then(|| {
            let raw_z = self.image_uint_address(image, 2);
            self.temp(
                "uint",
                &format!(
                    "(uint)clamp(as_type<int>({raw_z}), 0, (int)tex{binding_index}.get_depth() - 1)"
                ),
            )
        });
        let coordinates = z.as_ref().map_or_else(
            || format!("uint2({x}, {y})"),
            |z| format!("uint3({x}, {y}, {z})"),
        );
        let loaded = self.temp(
            &format!("vec<{}, 4>", self.image_kinds[binding_index].component),
            &format!("tex{binding_index}.read({coordinates})"),
        );
        let destination_select = self.image_destination_select(image);
        let mut values = Vec::with_capacity(4);
        for component in 0..4 {
            if image.dmask & (1 << component) == 0 {
                continue;
            }
            let kind = self.image_kinds[binding_index];
            let selector = format!("(({destination_select} >> {}u) & 7u)", component * 3);
            let one = if kind.component == "float" {
                "0x3F800000u"
            } else {
                "1u"
            };
            let physical = match kind.component {
                "uint" => format!("{loaded}[{selector} - 4u]"),
                "int" | "float" => {
                    format!("as_type<uint>({loaded}[{selector} - 4u])")
                }
                _ => unreachable!("image component type is selected from a closed set"),
            };
            let raw = format!(
                "({selector} == 1u ? {one} : ({selector} >= 4u && {selector} <= 7u ? {physical} : 0u))"
            );
            values.push(raw);
        }
        if image.d16 {
            for (index, pair) in values.chunks(2).enumerate() {
                let low = &pair[0];
                let high = pair.get(1).map_or("0u", String::as_str);
                let packed =
                    self.pack_image_d16(self.image_kinds[binding_index].component, low, high);
                self.store_vector(
                    image.vector_data
                        + u32::try_from(index).map_err(|_| "D16 image output overflow")?,
                    &packed,
                );
            }
        } else {
            for (index, raw) in values.iter().enumerate() {
                self.store_vector(
                    image.vector_data
                        + u32::try_from(index).map_err(|_| "image output overflow")?,
                    raw,
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::unused_self)]
    fn image_store_raw_component(
        &self,
        image: &ImageControl,
        kind: &str,
        component: u32,
    ) -> String {
        if !image.d16 {
            return format!("v[{}]", image.vector_data + component);
        }
        let packed = format!("v[{}]", image.vector_data + component / 2);
        if kind == "float" {
            return format!(
                "as_type<uint>((float)as_type<half2>({packed})[{}])",
                component & 1
            );
        }
        let low = format!("(({packed} >> {}u) & 0xFFFFu)", (component & 1) * 16);
        if kind == "int" {
            format!("(uint)extract_bits(as_type<int>({low}), 0u, 16u)")
        } else {
            low
        }
    }

    #[allow(clippy::unused_self)]
    fn pack_image_d16(&self, kind: &str, low: &str, high: &str) -> String {
        if kind == "float" {
            format!(
                "((uint)as_type<ushort>(half(as_type<float>({low}))) | ((uint)as_type<ushort>(half(as_type<float>({high}))) << 16))"
            )
        } else {
            format!("(({low} & 0xFFFFu) | (({high} & 0xFFFFu) << 16))")
        }
    }

    fn emit_image_sample(
        &mut self,
        image: &ImageControl,
        name: &str,
        binding_index: usize,
        sampler_slot: u32,
    ) -> Result<String, String> {
        let has_offset = name.ends_with('O');
        let has_compare = name.contains("SampleC");
        let has_gradients = name.contains("SampleD");
        let has_zero_lod = name.contains("Lz");
        let has_lod = !has_zero_lod && name.contains("SampleL");
        let has_bias = name.contains("SampleB");
        let mut cursor = 0_usize;
        let mut offset_x = "0".to_owned();
        let mut offset_y = "0".to_owned();
        let mut offset_z = "0".to_owned();
        if has_offset {
            cursor = align_full_image_address(image, cursor);
            let packed = self.temp(
                "int",
                &format!(
                    "as_type<int>(v[{}])",
                    image.address_register(image_address_register(image, cursor))
                ),
            );
            offset_x = self.temp("int", &format!("extract_bits({packed}, 0u, 6u)"));
            offset_y = self.temp("int", &format!("extract_bits({packed}, 8u, 6u)"));
            if image.dimension == 2 {
                offset_z = self.temp("int", &format!("extract_bits({packed}, 16u, 6u)"));
            }
            cursor += image_full_address_slots(image);
        }
        let bias = if has_bias {
            let value = self.image_float_address(image, cursor);
            cursor += 1;
            self.temp("float", &value)
        } else {
            "0.0f".to_owned()
        };
        let compare = if has_compare {
            let value = self.image_float_address(image, cursor);
            cursor += 1;
            self.temp("float", &value)
        } else {
            String::new()
        };
        let mut gradient_x = String::new();
        let mut gradient_y = String::new();
        if has_gradients {
            let gradient_components = if matches!(image.dimension, 2 | 3) {
                3
            } else {
                2
            };
            let gradient_kind = if gradient_components == 3 {
                "float3"
            } else {
                "float2"
            };
            gradient_x = self.temp(
                gradient_kind,
                &if gradient_components == 3 {
                    format!(
                        "float3({}, {}, {})",
                        self.image_float_address(image, cursor),
                        self.image_float_address(image, cursor + 1),
                        self.image_float_address(image, cursor + 2)
                    )
                } else {
                    format!(
                        "float2({}, {})",
                        self.image_float_address(image, cursor),
                        self.image_float_address(image, cursor + 1)
                    )
                },
            );
            gradient_y = self.temp(
                gradient_kind,
                &if gradient_components == 3 {
                    format!(
                        "float3({}, {}, {})",
                        self.image_float_address(image, cursor + 3),
                        self.image_float_address(image, cursor + 4),
                        self.image_float_address(image, cursor + 5)
                    )
                } else {
                    format!(
                        "float2({}, {})",
                        self.image_float_address(image, cursor + 2),
                        self.image_float_address(image, cursor + 3)
                    )
                },
            );
            cursor += gradient_components * 2;
        }
        let coordinate_kind = if matches!(image.dimension, 2 | 3) {
            "float3"
        } else {
            "float2"
        };
        let mut coordinates = self.temp(
            coordinate_kind,
            &if matches!(image.dimension, 2 | 3) {
                format!(
                    "float3({}, {}, {})",
                    self.image_float_address(image, cursor),
                    self.image_float_address(image, cursor + 1),
                    self.image_float_address(image, cursor + 2)
                )
            } else {
                format!(
                    "float2({}, {})",
                    self.image_float_address(image, cursor),
                    self.image_float_address(image, cursor + 1)
                )
            },
        );
        let lod = if has_zero_lod {
            "0.0f".to_owned()
        } else if has_lod {
            self.temp(
                "float",
                &self.image_float_address(
                    image,
                    cursor
                        + if matches!(image.dimension, 2 | 3) {
                            3
                        } else {
                            2
                        },
                ),
            )
        } else {
            bias.clone()
        };
        if has_offset {
            if image.dimension == 3 {
                return Err("cube sampling does not support integer texel offsets".to_owned());
            }
            let offset_lod = if (has_zero_lod || has_lod) && !has_gradients {
                lod.clone()
            } else {
                "0.0f".to_owned()
            };
            let mip = self.temp("uint", &format!("(uint)max((int)({offset_lod}), 0)"));
            coordinates = if image.dimension == 2 {
                self.temp(
                    "float3",
                    &format!(
                        "{coordinates} + float3((float){offset_x} / (float)tex{binding_index}.get_width({mip}), (float){offset_y} / (float)tex{binding_index}.get_height({mip}), (float){offset_z} / (float)tex{binding_index}.get_depth({mip}))"
                    ),
                )
            } else {
                self.temp(
                    "float2",
                    &format!(
                        "{coordinates} + float2((float){offset_x} / (float)tex{binding_index}.get_width({mip}), (float){offset_y} / (float)tex{binding_index}.get_height({mip}))"
                    ),
                )
            };
        }
        let extra = if has_gradients {
            let gradient = match image.dimension {
                2 => "gradient3d",
                3 => "gradientcube",
                _ => "gradient2d",
            };
            format!(", {gradient}({gradient_x}, {gradient_y})")
        } else if has_zero_lod || has_lod {
            format!(", level({lod})")
        } else if has_bias {
            format!(", bias({bias})")
        } else {
            String::new()
        };
        let kind = self.image_kinds[binding_index];
        let vector_kind = format!("vec<{}, 4>", kind.component);
        let physical = if has_compare {
            let compared = self.temp(
                "float",
                &format!(
                    "tex{binding_index}.sample_compare(remu_samplers.s{sampler_slot}, {coordinates}, {compare}{extra})"
                ),
            );
            self.temp(
                &vector_kind,
                &format!("{vector_kind}({compared}, {compared}, {compared}, {compared})"),
            )
        } else {
            self.temp(
                &vector_kind,
                &format!(
                    "tex{binding_index}.sample(remu_samplers.s{sampler_slot}, {coordinates}{extra})"
                ),
            )
        };
        let zero = if kind.component == "float" {
            "0.0f"
        } else if kind.component == "uint" {
            "0u"
        } else {
            "0"
        };
        let one = if kind.component == "float" {
            "1.0f"
        } else if kind.component == "uint" {
            "1u"
        } else {
            "1"
        };
        let destination_select = self.image_destination_select(image);
        let mut logical = Vec::with_capacity(4);
        for component in 0..4 {
            let selector = format!("(({destination_select} >> {}u) & 7u)", component * 3);
            logical.push(format!(
                "({selector} == 1u ? {one} : ({selector} >= 4u && {selector} <= 7u ? {physical}[{selector} - 4u] : {zero}))"
            ));
        }
        Ok(self.temp(
            &vector_kind,
            &format!(
                "{vector_kind}({}, {}, {}, {})",
                logical[0], logical[1], logical[2], logical[3]
            ),
        ))
    }

    #[allow(clippy::unused_self)]
    fn image_float_address(&self, image: &ImageControl, component: usize) -> String {
        let register = image.address_register(image_address_register(image, component));
        if image.a16 {
            format!("(float)as_type<half2>(v[{register}])[{}]", component & 1)
        } else {
            format!("as_type<float>(v[{register}])")
        }
    }

    #[allow(clippy::unused_self)]
    fn image_uint_address(&self, image: &ImageControl, component: usize) -> String {
        let register = image.address_register(image_address_register(image, component));
        if image.a16 {
            format!("((v[{register}] >> {}u) & 0xFFFFu)", (component & 1) * 16)
        } else {
            format!("v[{register}]")
        }
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

    fn store_vector(&mut self, register: u32, expression: &str) {
        if register < VECTOR_REGISTER_COUNT {
            self.line(3, &format!("if (exec) v[{register}] = {expression};"));
        }
    }

    fn store_scalar(&mut self, register: u32, expression: &str) {
        if register >= SCALAR_REGISTER_COUNT {
            return;
        }
        match register {
            VCC_LO | VCC_HI => {
                let value = self.temp("uint", expression);
                self.line(3, &format!("s[{register}] = {value};"));
                let predicate =
                    self.lane_mask_predicate(&format!("s[{VCC_LO}]"), &format!("s[{VCC_HI}]"));
                self.line(3, &format!("vcc = {predicate};"));
            }
            EXEC_LO | EXEC_HI => {
                let value = self.temp("uint", expression);
                self.line(3, &format!("s[{register}] = {value};"));
                let predicate =
                    self.lane_mask_predicate(&format!("s[{EXEC_LO}]"), &format!("s[{EXEC_HI}]"));
                self.line(3, &format!("exec = {predicate};"));
            }
            _ => self.line(3, &format!("s[{register}] = {expression};")),
        }
    }

    fn store_scalar_pair(&mut self, register: u32, expression: &str) {
        if register + 1 >= SCALAR_REGISTER_COUNT {
            return;
        }
        self.line(3, &format!("s[{register}] = (uint){expression};"));
        self.line(
            3,
            &format!("s[{}] = (uint)({expression} >> 32);", register + 1),
        );
        if register == VCC_LO {
            let predicate =
                self.lane_mask_predicate(&format!("s[{VCC_LO}]"), &format!("s[{VCC_HI}]"));
            self.line(3, &format!("vcc = {predicate};"));
        } else if register == EXEC_LO {
            let predicate =
                self.lane_mask_predicate(&format!("s[{EXEC_LO}]"), &format!("s[{EXEC_HI}]"));
            self.line(3, &format!("exec = {predicate};"));
        }
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

fn branch_target(instruction: &Instruction) -> Result<u32, String> {
    if instruction.encoding != ShaderEncoding::Sopp || instruction.words.len() != 1 {
        return Err("scalar branch is not one complete SOPP instruction".to_owned());
    }
    let word = instruction
        .words
        .first()
        .copied()
        .ok_or_else(|| "branch has no encoding word".to_owned())?;
    let immediate = u16::try_from(word & 0xffff)
        .expect("masked branch immediate fits u16")
        .cast_signed();
    let end = i64::from(instruction.pc)
        + i64::try_from(instruction.words.len() * 4)
            .map_err(|_| "branch instruction size overflow".to_owned())?;
    let target = end + i64::from(immediate) * 4;
    u32::try_from(target).map_err(|_| "branch target is outside the Gen5 PC range".to_owned())
}

fn analyze_control_flow(program: &Program) -> Result<ControlFlowLowering, MetalCompileError> {
    let mut instruction_by_pc = BTreeMap::new();
    for (index, instruction) in program.instructions.iter().enumerate() {
        if instruction_by_pc.insert(instruction.pc, index).is_some() {
            return Err(MetalCompileError::InvalidInterface(format!(
                "duplicate instruction pc 0x{:X}",
                instruction.pc
            )));
        }
    }

    // Validate every encoded branch, including unreachable code. An invalid target previously
    // fell through the dispatcher's default arm and silently terminated the shader.
    for instruction in &program.instructions {
        if !is_scalar_branch(instruction.opcode) {
            continue;
        }
        let target =
            branch_target(instruction).map_err(|message| MetalCompileError::Instruction {
                pc: instruction.pc,
                opcode: instruction.opcode,
                message,
            })?;
        if !instruction_by_pc.contains_key(&target) {
            return Err(MetalCompileError::Instruction {
                pc: instruction.pc,
                opcode: instruction.opcode,
                message: format!("branch target 0x{target:X} is not an instruction boundary"),
            });
        }
    }

    let mut path = Vec::new();
    let mut visited = BTreeSet::new();
    let mut index = 0_usize;
    while let Some(instruction) = program.instructions.get(index) {
        if !visited.insert(index) {
            return Ok(ControlFlowLowering::Dispatcher);
        }
        path.push(index);
        match instruction.opcode.known_name() {
            Some("SEndpgm") => break,
            Some("SBranch") => {
                let target = branch_target(instruction).expect("all branch targets were validated");
                index = instruction_by_pc[&target];
            }
            Some(name) if name.starts_with("SCbranch") => {
                return Ok(ControlFlowLowering::Dispatcher);
            }
            _ => {
                index += 1;
            }
        }
    }
    Ok(ControlFlowLowering::Direct(path))
}

fn is_scalar_branch(opcode: Opcode) -> bool {
    opcode
        .known_name()
        .is_some_and(|name| name == "SBranch" || name.starts_with("SCbranch"))
}

fn dispatcher_basic_blocks(program: &Program) -> Vec<std::ops::Range<usize>> {
    let mut instruction_by_pc = BTreeMap::new();
    for (index, instruction) in program.instructions.iter().enumerate() {
        instruction_by_pc.insert(instruction.pc, index);
    }

    let mut leaders = BTreeSet::from([0_usize]);
    for (index, instruction) in program.instructions.iter().enumerate() {
        let terminator = is_scalar_branch(instruction.opcode)
            || instruction.opcode.known_name() == Some("SEndpgm");
        if terminator && index + 1 < program.instructions.len() {
            leaders.insert(index + 1);
        }
        if is_scalar_branch(instruction.opcode) {
            let target = branch_target(instruction)
                .expect("dispatcher block construction follows validated control flow");
            leaders.insert(instruction_by_pc[&target]);
        }
    }

    let leaders = leaders.into_iter().collect::<Vec<_>>();
    leaders
        .iter()
        .copied()
        .enumerate()
        .map(|(index, start)| {
            let end = leaders
                .get(index + 1)
                .copied()
                .unwrap_or(program.instructions.len());
            start..end
        })
        .collect()
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

fn raw_load_shape(name: &str) -> Result<(u32, bool, bool, bool), String> {
    if !name.contains("Load") {
        return Err(format!("{name} is not a raw load"));
    }
    let width = if name.contains("Ubyte") || name.contains("Sbyte") {
        1
    } else if name.contains("Ushort") || name.contains("Sshort") || name.contains("ShortD16") {
        2
    } else if name.contains("Dword") {
        4
    } else {
        return Err(format!("unsupported raw load width for {name}"));
    };
    Ok((
        width,
        name.contains("Sbyte") || name.contains("Sshort"),
        name.contains("D16"),
        name.ends_with("D16Hi"),
    ))
}

fn raw_store_shape(name: &str) -> Result<(u32, bool), String> {
    if !name.contains("Store") {
        return Err(format!("{name} is not a raw store"));
    }
    if name.contains("Byte") {
        Ok((1, name.ends_with("D16Hi")))
    } else if name.contains("Short") {
        Ok((2, name.ends_with("D16Hi")))
    } else if name.contains("Dword") {
        Ok((4, false))
    } else {
        Err(format!("unsupported raw store width for {name}"))
    }
}

#[allow(clippy::too_many_arguments)]
fn buffer_oob_expression(
    descriptor_null: &str,
    oob_select: &str,
    index: &str,
    record_count: &str,
    offset: &str,
    stride: &str,
    swizzled: &str,
    buffer_offset: &str,
    reduced_records: &str,
    payload: &str,
    component_offset: u32,
) -> String {
    let offset = if component_offset == 0 {
        offset.to_owned()
    } else {
        format!("{offset} + {component_offset}u")
    };
    let buffer_offset = if component_offset == 0 {
        buffer_offset.to_owned()
    } else {
        format!("{buffer_offset} + {component_offset}u")
    };
    format!(
        "{descriptor_null} || ({oob_select} == 0u ? ({index} >= {record_count} || {offset} >= {stride}) : {oob_select} == 1u ? ({index} >= {record_count}) : {oob_select} == 2u ? ({record_count} == 0u) : ({swizzled} && {stride} != 0u ? ({index} >= {record_count} || !remu_range_valid({offset}, {payload}, {stride})) : !remu_range_valid({buffer_offset}, {payload}, {reduced_records})))"
    )
}

fn format_buffer_element_oob_expression(
    descriptor_oob: &str,
    address: &str,
    element_size: &str,
    binding_length: &str,
) -> String {
    // A structured buffer's stride, not the total formatted element size, positions records.
    // Requiring a float3 payload to begin at a multiple of 12 would reject legal records in a
    // 28-byte vertex layout. The bytewise format loader below already handles every component at
    // its architectural byte offset, so only format validity and the actual bound range matter.
    format!(
        "{descriptor_oob} || {element_size} == 0u || !remu_range_valid({address}, {element_size}, {binding_length})"
    )
}

fn memory_payload_bytes(name: &str, _dword_count: u32) -> Result<u32, String> {
    if name.contains("Atomic") || name.contains("Dword") {
        Ok(4)
    } else if name.contains("Byte") || name.contains("Ubyte") || name.contains("Sbyte") {
        Ok(1)
    } else if name.contains("Short") || name.contains("Ushort") || name.contains("Sshort") {
        Ok(2)
    } else {
        Err(format!("unknown memory payload width for {name}"))
    }
}

fn format_component_count(name: &str) -> u32 {
    if name.ends_with("Xyzw") {
        4
    } else if name.ends_with("Xyz") {
        3
    } else if name.ends_with("Xy") {
        2
    } else {
        1
    }
}

fn is_lds_memory_instruction(instruction: &Instruction) -> bool {
    match instruction.control {
        Some(InstructionControl::DataShare(_)) => !instruction.opcode.is("DsSwizzleB32"),
        Some(InstructionControl::BufferMemory(control)) => control.lds,
        Some(InstructionControl::GlobalMemory(control)) => control.lds,
        _ => false,
    }
}

fn is_lds_atomic_name(name: &str) -> bool {
    matches!(
        name,
        "DsAddU32"
            | "DsSubU32"
            | "DsIncU32"
            | "DsDecU32"
            | "DsMinI32"
            | "DsMaxI32"
            | "DsMinU32"
            | "DsMaxU32"
            | "DsAndB32"
            | "DsOrB32"
            | "DsXorB32"
            | "DsCmpstB32"
            | "DsAddRtnU32"
            | "DsSubRtnU32"
            | "DsIncRtnU32"
            | "DsDecRtnU32"
            | "DsMinRtnI32"
            | "DsMaxRtnI32"
            | "DsMinRtnU32"
            | "DsMaxRtnU32"
            | "DsAndRtnB32"
            | "DsOrRtnB32"
            | "DsXorRtnB32"
            | "DsWrxchgRtnB32"
            | "DsCmpstRtnB32"
    )
}

const fn ds_combined_offset(control: &DataShareControl) -> u32 {
    control.offset0 | (control.offset1 << 8)
}

fn ds_swizzle_lane_expression(offset: u16) -> String {
    if offset >= 0xe000 {
        let mask = u32::from(offset & 0x1f);
        let reverse = if mask.count_ones() == 5 {
            "0u".to_owned()
        } else {
            format!(
                "(reverse_bits(remu_lane & 31u) >> {}u)",
                27 + mask.count_ones()
            )
        };
        format!("({reverse} | (remu_lane & {mask}u))")
    } else if offset >= 0xc000 {
        let rotate = u32::from((offset >> 5) & 0x1f);
        let mask = u32::from(offset & 0x1f);
        let delta = if offset & (1 << 10) == 0 {
            rotate
        } else {
            32 - rotate
        };
        format!(
            "((remu_lane & {mask}u) | (((remu_lane + {delta}u) & 31u) & {}u))",
            31 ^ mask
        )
    } else if offset & 0x8000 != 0 {
        format!("((remu_lane & 0xFFFFFFFCu) | ((0x{offset:04X}u >> ((remu_lane & 3u) * 2u)) & 3u))")
    } else {
        let and_mask = u32::from(offset & 0x1f);
        let or_mask = u32::from((offset >> 5) & 0x1f);
        let xor_mask = u32::from((offset >> 10) & 0x1f);
        format!(
            "((remu_lane & 0xFFFFFFE0u) | ((((remu_lane & 31u) & {and_mask}u) | {or_mask}u) ^ {xor_mask}u))"
        )
    }
}

fn emit_data_share_prelude(source: &mut String) {
    source.push_str(
        r"inline uint remu_ds_inc(threadgroup atomic_uint* location, uint ceiling) {
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint desired = observed >= ceiling ? 0u : observed + 1u;
        uint expected = observed;
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

inline uint remu_ds_dec(threadgroup atomic_uint* location, uint ceiling) {
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint desired = (observed == 0u || observed > ceiling) ? ceiling : observed - 1u;
        uint expected = observed;
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

inline uint remu_ds_smin(threadgroup atomic_uint* location, uint value) {
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint desired = as_type<int>(value) < as_type<int>(observed) ? value : observed;
        uint expected = observed;
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

inline uint remu_ds_smax(threadgroup atomic_uint* location, uint value) {
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint desired = as_type<int>(value) > as_type<int>(observed) ? value : observed;
        uint expected = observed;
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

inline uint remu_ds_cmpst(threadgroup atomic_uint* location, uint comparator,
                         uint replacement) {
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (observed == comparator) {
        uint expected = observed;
        if (atomic_compare_exchange_weak_explicit(location, &expected, replacement,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
    return observed;
}

",
    );
}

fn is_format_buffer_memory(instruction: &Instruction) -> bool {
    instruction.opcode.starts_with("BufferLoadFormat")
        || instruction.opcode.starts_with("BufferStoreFormat")
        || instruction.opcode.starts_with("TBufferLoadFormat")
        || instruction.opcode.starts_with("TBufferStoreFormat")
}

fn is_writable_memory(instruction: &Instruction) -> bool {
    instruction.opcode.contains("Store") || instruction.opcode.contains("Atomic")
}

#[allow(clippy::too_many_lines)]
fn emit_global_memory_prelude(source: &mut String) {
    source.push_str(
        r"inline void remu_store_u8(device uchar* data, uint offset, uint value) {
    data[offset] = (uchar)value;
}

inline void remu_store_u16(device uchar* data, uint offset, uint value) {
    data[offset] = (uchar)value;
    data[offset + 1u] = (uchar)(value >> 8u);
}

inline void remu_store_u32(device uchar* data, uint offset, uint value) {
    for (uint i = 0u; i < 4u; ++i) data[offset + i] = (uchar)(value >> (i * 8u));
}

inline device atomic_uint* remu_atomic_pointer(device uchar* data, uint offset) {
    return (device atomic_uint*)(data + offset);
}

inline uint remu_atomic_swap(device uchar* data, uint offset, uint value) {
    return atomic_exchange_explicit(remu_atomic_pointer(data, offset), value,
                                    memory_order_relaxed);
}

inline uint remu_atomic_add(device uchar* data, uint offset, uint value) {
    return atomic_fetch_add_explicit(remu_atomic_pointer(data, offset), value,
                                     memory_order_relaxed);
}

inline uint remu_atomic_sub(device uchar* data, uint offset, uint value) {
    return atomic_fetch_sub_explicit(remu_atomic_pointer(data, offset), value,
                                     memory_order_relaxed);
}

inline uint remu_atomic_and(device uchar* data, uint offset, uint value) {
    return atomic_fetch_and_explicit(remu_atomic_pointer(data, offset), value,
                                     memory_order_relaxed);
}

inline uint remu_atomic_or(device uchar* data, uint offset, uint value) {
    return atomic_fetch_or_explicit(remu_atomic_pointer(data, offset), value,
                                    memory_order_relaxed);
}

inline uint remu_atomic_xor(device uchar* data, uint offset, uint value) {
    return atomic_fetch_xor_explicit(remu_atomic_pointer(data, offset), value,
                                     memory_order_relaxed);
}

inline uint remu_atomic_cmpswap(device uchar* data, uint offset, uint replacement,
                                uint comparator) {
    device atomic_uint* location = remu_atomic_pointer(data, offset);
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (observed == comparator) {
        uint expected = observed;
        if (atomic_compare_exchange_weak_explicit(location, &expected, replacement,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
    return observed;
}

inline uint remu_atomic_umin(device uchar* data, uint offset, uint value) {
    device atomic_uint* location = remu_atomic_pointer(data, offset);
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint expected = observed;
        uint desired = min(observed, value);
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

inline uint remu_atomic_umax(device uchar* data, uint offset, uint value) {
    device atomic_uint* location = remu_atomic_pointer(data, offset);
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint expected = observed;
        uint desired = max(observed, value);
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

inline uint remu_atomic_smin(device uchar* data, uint offset, uint value) {
    device atomic_uint* location = remu_atomic_pointer(data, offset);
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint expected = observed;
        uint desired = as_type<int>(value) < as_type<int>(observed) ? value : observed;
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

inline uint remu_atomic_smax(device uchar* data, uint offset, uint value) {
    device atomic_uint* location = remu_atomic_pointer(data, offset);
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint expected = observed;
        uint desired = as_type<int>(value) > as_type<int>(observed) ? value : observed;
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

inline uint remu_atomic_inc(device uchar* data, uint offset, uint limit) {
    device atomic_uint* location = remu_atomic_pointer(data, offset);
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint expected = observed;
        uint desired = observed >= limit ? 0u : observed + 1u;
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

inline uint remu_atomic_dec(device uchar* data, uint offset, uint limit) {
    device atomic_uint* location = remu_atomic_pointer(data, offset);
    uint observed = atomic_load_explicit(location, memory_order_relaxed);
    while (true) {
        uint expected = observed;
        uint desired = observed == 0u || observed > limit ? limit : observed - 1u;
        if (atomic_compare_exchange_weak_explicit(location, &expected, desired,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) return observed;
        observed = expected;
    }
}

",
    );
}

fn emit_vop3p_prelude(source: &mut String) {
    source.push_str(
        r"inline bool remu_f32_bits_is_nan(uint bits) {
    return (bits & 0x7FFFFFFFu) > 0x7F800000u;
}

inline uint remu_f16_to_f32_bits(uint half_bits) {
    half_bits &= 0xFFFFu;
    uint sign = (half_bits & 0x8000u) << 16;
    uint exponent = (half_bits >> 10) & 0x1Fu;
    uint mantissa = half_bits & 0x3FFu;
    uint normal = ((exponent + 112u) << 23) | (mantissa << 13);
    uint infinity_nan = 0x7F800000u | (mantissa << 13);
    uint high_bit = 31u - clz(mantissa | 1u);
    uint shift = 23u - high_bit;
    uint sub_fraction = (mantissa << shift) & 0x7FFFFFu;
    uint subnormal = mantissa != 0u ? ((high_bit + 103u) << 23) | sub_fraction : 0u;
    uint magnitude = exponent == 0u ? subnormal : (exponent == 31u ? infinity_nan : normal);
    return sign | magnitude;
}

inline uint remu_f32_to_f16_bits(uint bits) {
    uint sign = (bits >> 16) & 0x8000u;
    uint absolute = bits & 0x7FFFFFFFu;
    bool infinity_nan_input = absolute >= 0x7F800000u;
    bool nan_input = absolute > 0x7F800000u;
    uint infinity_nan = sign | 0x7C00u | (nan_input ? 0x200u : 0u);

    uint exponent = absolute >> 23;
    uint mantissa = absolute & 0x7FFFFFu;
    uint significand = mantissa | 0x800000u;

    uint round_bit = (significand >> 13) & 1u;
    uint rounded = (significand + 0xFFFu + round_bit) >> 13;
    uint half_exponent = exponent - 112u;
    uint normal_bits = (half_exponent << 10) + (rounded - 0x400u);
    uint normal = exponent >= 113u ? (normal_bits >= 0x7C00u ? 0x7C00u : normal_bits) : 0u;

    uint distance = 126u - exponent;
    uint subnormal_shift = min(max(distance, 1u), 25u);
    uint shift_mask = (1u << subnormal_shift) - 1u;
    uint halfway = 1u << (subnormal_shift - 1u);
    uint low_bits = significand & shift_mask;
    uint quotient = significand >> subnormal_shift;
    bool round_up = low_bits > halfway || (low_bits == halfway && (quotient & 1u) != 0u);
    uint subnormal = quotient + (round_up ? 1u : 0u);

    uint finite = exponent <= 112u ? subnormal : normal;
    return infinity_nan_input ? infinity_nan : (sign | finite);
}

inline uint remu_clamp_unit_f32_bits(uint bits) {
    uint magnitude = bits & 0x7FFFFFFFu;
    if (remu_f32_bits_is_nan(bits) || (bits & 0x80000000u) != 0u || magnitude == 0u) return 0u;
    return magnitude > 0x3F800000u ? 0x3F800000u : bits;
}

inline uint remu_f16_minmax_bits(float left, float right, bool maximum) {
    uint left_bits = as_type<uint>(left);
    uint right_bits = as_type<uint>(right);
    if (remu_f32_bits_is_nan(left_bits)) return right_bits;
    if (remu_f32_bits_is_nan(right_bits)) return left_bits;
    return as_type<uint>(maximum ? (left > right ? left : right) : (left < right ? left : right));
}

inline uint remu_f16_fma_rto_bits(float left, float right, float addend) {
#pragma clang fp reassociate(off) contract(off)
    float product = left * right;
    float sum = product + addend;
    float product_part = sum - addend;
    float addend_part = sum - product_part;
    float product_error = product - product_part;
    float addend_error = addend - addend_part;
    float residual = product_error + addend_error;

    uint sum_bits = as_type<uint>(sum);
    uint residual_bits = as_type<uint>(residual);
    uint residual_magnitude = residual_bits & 0x7FFFFFFFu;
    bool inexact = residual_magnitude != 0u && residual_magnitude < 0x7F800000u;
    bool adjust = inexact && (sum_bits & 1u) == 0u;
    bool toward_zero = ((sum_bits ^ residual_bits) & 0x80000000u) != 0u;
    uint stepped = toward_zero ? sum_bits - 1u : sum_bits + 1u;
    return adjust ? stepped : sum_bits;
}

",
    );
}

#[allow(clippy::too_many_lines)]
fn emit_format_prelude(source: &mut String) {
    source.push_str("static constant uint remu_gfx10_formats[128] = {\n    ");
    for unified in 0..128 {
        let packed =
            decode_unified_format(unified).map_or(0, |(data, number)| data | (number << 8));
        let _ = write!(source, "0x{packed:X}u,");
        if unified % 16 == 15 && unified != 127 {
            source.push_str("\n    ");
        } else {
            source.push(' ');
        }
    }
    source.push_str(
        "\n};\n\n\
         static inline void remu_format_layout(uint dfmt, uint component, thread uint& byte_off, thread uint& bit_off, thread uint& bits) {\n\
         \x20   byte_off = 0u; bit_off = 0u; bits = 0u;\n\
         \x20   switch (component * 16u + dfmt) {\n",
    );
    for &(component, format, bytes, bit_offset, bits) in FORMAT_COMPONENT_LAYOUTS {
        let key = component * 16 + format;
        let _ = writeln!(
            source,
            "    case {key}u: byte_off = {bytes}u; bit_off = {bit_offset}u; bits = {bits}u; break;"
        );
    }
    source.push_str(
        "    default: break;\n    }\n}\n\n\
         static inline uint remu_minifloat(uint raw, uint bits) {\n\
         \x20   uint mantissa_bits = bits - 5u;\n\
         \x20   uint mantissa = raw & ((1u << mantissa_bits) - 1u);\n\
         \x20   uint exponent = (raw >> mantissa_bits) & 0x1Fu;\n\
         \x20   uint shift = 23u - mantissa_bits;\n\
         \x20   if (exponent == 31u) return 0x7F800000u | (mantissa << shift);\n\
         \x20   if (exponent == 0u) {\n\
         \x20       float scale = mantissa_bits == 6u ? (1.0f / 1048576.0f) : (1.0f / 524288.0f);\n\
         \x20       return as_type<uint>((float)mantissa * scale);\n\
         \x20   }\n\
         \x20   return ((exponent + 112u) << 23) | (mantissa << shift);\n}\n\n\
         static inline uint remu_format_one(uint nfmt) {\n\
         \x20   return (nfmt == 4u || nfmt == 5u) ? 1u : 0x3F800000u;\n}\n\n\
         static inline uint remu_format_convert(uint raw, uint bits, uint nfmt, uint dfmt) {\n\
         \x20   uint low_mask = bits >= 32u ? 0xFFFFFFFFu : ((1u << bits) - 1u);\n\
         \x20   int signed_raw = extract_bits(as_type<int>(raw), 0u, bits);\n\
         \x20   switch (nfmt) {\n\
         \x20   case 0u: return as_type<uint>((float)raw / (float)low_mask);\n\
         \x20   case 1u: return as_type<uint>(fmax((float)signed_raw / (float)(low_mask >> 1), -1.0f));\n\
         \x20   case 2u: return as_type<uint>((float)raw);\n\
         \x20   case 3u: return as_type<uint>((float)signed_raw);\n\
         \x20   case 5u: return (uint)signed_raw;\n\
         \x20   case 7u:\n\
         \x20       if (dfmt == 6u || dfmt == 7u) return remu_minifloat(raw, bits);\n\
         \x20       if (bits == 16u) return as_type<uint>((float)as_type<half>((ushort)(raw & 0xFFFFu)));\n\
         \x20       return raw;\n\
         \x20   default: return raw;\n\
         \x20   }\n}\n\n",
    );
    source.push_str("static inline uint remu_format_size(uint dfmt) {\n    switch (dfmt) {\n");
    for format in 1..=14_u32 {
        let size = FORMAT_COMPONENT_LAYOUTS
            .iter()
            .filter(|(_, candidate, _, _, _)| *candidate == format)
            .map(|(_, _, byte_offset, bit_offset, bits)| {
                byte_offset + (bit_offset + bits).div_ceil(8)
            })
            .max()
            .unwrap_or(0);
        let _ = writeln!(source, "    case {format}u: return {size}u;");
    }
    source.push_str(
        r"    default: return 0u;
    }
}

static inline uint remu_load_format_bits(const device uchar* data, uint address,
                                         uint byte_off, uint bit_off, uint bits) {
    uint value = 0u;
    for (uint bit = 0u; bit < bits; ++bit) {
        uint source_bit = bit_off + bit;
        uint source_byte = address + byte_off + source_bit / 8u;
        value |= (((uint)data[source_byte] >> (source_bit & 7u)) & 1u) << bit;
    }
    return value;
}

static inline void remu_store_format_bits(device uchar* data, uint address, uint byte_off,
                                          uint bit_off, uint bits, uint value) {
    for (uint bit = 0u; bit < bits; ++bit) {
        uint destination_bit = bit_off + bit;
        uint destination_byte = address + byte_off + destination_bit / 8u;
        uint mask = 1u << (destination_bit & 7u);
        uint byte_value = (uint)data[destination_byte];
        byte_value = (byte_value & ~mask) | (((value >> bit) & 1u) * mask);
        data[destination_byte] = (uchar)byte_value;
    }
}

static inline uint remu_format_minifloat_pack(uint value, uint bits) {
    float converted = fmax(as_type<float>(value), 0.0f);
    uint half_bits = (uint)as_type<ushort>(half(converted));
    uint exponent = (half_bits >> 10) & 0x1Fu;
    uint mantissa = half_bits & 0x3FFu;
    uint mantissa_bits = bits - 5u;
    uint shift = 10u - mantissa_bits;
    uint truncated = mantissa >> shift;
    uint remainder = mantissa & ((1u << shift) - 1u);
    uint halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (truncated & 1u) != 0u)) {
        truncated += 1u;
        if (truncated == (1u << mantissa_bits)) {
            truncated = 0u;
            exponent = min(exponent + 1u, 31u);
        }
    }
    return (exponent << mantissa_bits) | truncated;
}

static inline uint remu_format_pack(uint value, uint bits, uint nfmt, uint dfmt) {
    uint mask = bits >= 32u ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    float floating = as_type<float>(value);
    switch (nfmt) {
    case 0u:
        return (uint)rint(clamp(floating, 0.0f, 1.0f) * (float)mask) & mask;
    case 1u: {
        int maximum = (int)(mask >> 1);
        int packed = (int)rint(clamp(floating, -1.0f, 1.0f) * (float)maximum);
        return (uint)packed & mask;
    }
    case 2u:
        return (uint)rint(clamp(floating, 0.0f, (float)mask)) & mask;
    case 3u: {
        int maximum = (int)(mask >> 1);
        int minimum = -maximum - 1;
        return (uint)(int)rint(clamp(floating, (float)minimum, (float)maximum)) & mask;
    }
    case 4u: return value & mask;
    case 5u: return (uint)as_type<int>(value) & mask;
    case 7u:
        if (dfmt == 6u || dfmt == 7u) return remu_format_minifloat_pack(value, bits);
        if (bits == 16u) return (uint)as_type<ushort>(half(floating));
        return value;
    default: return value & mask;
    }
}

static inline uint remu_format_from_d16(uint value, uint nfmt) {
    value &= 0xFFFFu;
    if (nfmt == 4u) return value;
    if (nfmt == 5u) return (value & 0x8000u) != 0u ? value | 0xFFFF0000u : value;
    return as_type<uint>((float)as_type<half>((ushort)value));
}

static inline uint remu_format_to_d16(uint value, uint nfmt) {
    if (nfmt == 4u || nfmt == 5u) return value & 0xFFFFu;
    return (uint)as_type<ushort>(half(as_type<float>(value)));
}

",
    );
}

const FORMAT_COMPONENT_LAYOUTS: &[(u32, u32, u32, u32, u32)] = &[
    (0, 1, 0, 0, 8),
    (0, 2, 0, 0, 16),
    (0, 3, 0, 0, 8),
    (0, 4, 0, 0, 32),
    (0, 5, 0, 0, 16),
    (0, 6, 0, 0, 10),
    (0, 7, 0, 0, 11),
    (0, 8, 0, 0, 10),
    (0, 9, 0, 0, 2),
    (0, 10, 0, 0, 8),
    (0, 11, 0, 0, 32),
    (0, 12, 0, 0, 16),
    (0, 13, 0, 0, 32),
    (0, 14, 0, 0, 32),
    (1, 3, 1, 0, 8),
    (1, 5, 2, 0, 16),
    (1, 6, 0, 10, 11),
    (1, 7, 0, 11, 11),
    (1, 8, 0, 10, 10),
    (1, 9, 0, 2, 10),
    (1, 10, 1, 0, 8),
    (1, 11, 4, 0, 32),
    (1, 12, 2, 0, 16),
    (1, 13, 4, 0, 32),
    (1, 14, 4, 0, 32),
    (2, 6, 0, 21, 11),
    (2, 7, 0, 22, 10),
    (2, 8, 0, 20, 10),
    (2, 9, 0, 12, 10),
    (2, 10, 2, 0, 8),
    (2, 12, 4, 0, 16),
    (2, 13, 8, 0, 32),
    (2, 14, 8, 0, 32),
    (3, 8, 0, 30, 2),
    (3, 9, 0, 22, 10),
    (3, 10, 3, 0, 8),
    (3, 12, 6, 0, 16),
    (3, 14, 12, 0, 32),
];

const fn image_address_register(image: &ImageControl, component: usize) -> usize {
    if image.a16 { component / 2 } else { component }
}

const fn image_full_address_slots(image: &ImageControl) -> usize {
    if image.a16 { 2 } else { 1 }
}

const fn align_full_image_address(image: &ImageControl, component: usize) -> usize {
    if image.a16 {
        (component + 1) & !1
    } else {
        component
    }
}

const fn is_supported_dpp_control(control: u32) -> bool {
    control <= 0xff
        || matches!(
            control,
            0x101..=0x10f
                | 0x111..=0x11f
                | 0x121..=0x12f
                | 0x140
                | 0x141
                | 0x150..=0x15f
                | 0x160..=0x16f
        )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VectorCompare {
    Always(bool),
    Ordered,
    Unordered,
    Float {
        operator: &'static str,
        negate: bool,
    },
    Integer {
        operator: &'static str,
        signed: bool,
    },
    Class,
}

fn classify_vector_compare(name: &str) -> Result<VectorCompare, String> {
    if name == "VCmpClassF32" {
        return Ok(VectorCompare::Class);
    }
    let tail = name
        .strip_prefix("VCmpx")
        .or_else(|| name.strip_prefix("VCmp"))
        .ok_or_else(|| format!("invalid vector compare opcode {name}"))?;
    let (relation, kind) = tail
        .strip_suffix("F32")
        .map(|relation| (relation, "float"))
        .or_else(|| {
            tail.strip_suffix("I32")
                .map(|relation| (relation, "signed"))
        })
        .or_else(|| {
            tail.strip_suffix("U32")
                .map(|relation| (relation, "unsigned"))
        })
        .ok_or_else(|| format!("invalid vector compare type in {name}"))?;
    match kind {
        "float" => match relation {
            "F" => Ok(VectorCompare::Always(false)),
            "Tru" => Ok(VectorCompare::Always(true)),
            "O" => Ok(VectorCompare::Ordered),
            "U" => Ok(VectorCompare::Unordered),
            "Lt" => Ok(VectorCompare::Float {
                operator: "<",
                negate: false,
            }),
            "Eq" => Ok(VectorCompare::Float {
                operator: "==",
                negate: false,
            }),
            "Le" => Ok(VectorCompare::Float {
                operator: "<=",
                negate: false,
            }),
            "Gt" => Ok(VectorCompare::Float {
                operator: ">",
                negate: false,
            }),
            "Lg" => Ok(VectorCompare::Float {
                operator: "!=",
                negate: false,
            }),
            "Ge" => Ok(VectorCompare::Float {
                operator: ">=",
                negate: false,
            }),
            "Nge" => Ok(VectorCompare::Float {
                operator: ">=",
                negate: true,
            }),
            "Nlg" => Ok(VectorCompare::Float {
                operator: "!=",
                negate: true,
            }),
            "Ngt" => Ok(VectorCompare::Float {
                operator: ">",
                negate: true,
            }),
            "Nle" => Ok(VectorCompare::Float {
                operator: "<=",
                negate: true,
            }),
            "Neq" => Ok(VectorCompare::Float {
                operator: "==",
                negate: true,
            }),
            "Nlt" => Ok(VectorCompare::Float {
                operator: "<",
                negate: true,
            }),
            _ => Err(format!("unsupported float comparison {name}")),
        },
        "signed" | "unsigned" => {
            let operator = match relation {
                "F" | "T" => {
                    return Ok(VectorCompare::Always(relation == "T"));
                }
                "Lt" => "<",
                "Eq" => "==",
                "Le" => "<=",
                "Gt" => ">",
                "Ne" => "!=",
                "Ge" => ">=",
                _ => return Err(format!("unsupported integer comparison {name}")),
            };
            Ok(VectorCompare::Integer {
                operator,
                signed: kind == "signed",
            })
        }
        _ => unreachable!("comparison kind is selected above"),
    }
}

fn destination_vector(instruction: &Instruction) -> Result<u32, String> {
    instruction
        .destinations
        .first()
        .copied()
        .filter(|operand| operand.kind == OperandKind::VectorRegister)
        .map(|operand| operand.value)
        .ok_or_else(|| "missing vector destination".to_owned())
}

fn destination_scalar(instruction: &Instruction) -> Result<u32, String> {
    instruction
        .destinations
        .first()
        .copied()
        .filter(|operand| operand.kind == OperandKind::ScalarRegister)
        .map(|operand| operand.value)
        .ok_or_else(|| "missing scalar destination".to_owned())
}

fn vector_operand(operand: Operand) -> Result<u32, String> {
    (operand.kind == OperandKind::VectorRegister)
        .then_some(operand.value)
        .ok_or_else(|| format!("expected vector operand, got {operand}"))
}

fn is_storage_image(opcode: Opcode) -> bool {
    opcode.starts_with("ImageStore")
        || opcode.starts_with("ImageLoad")
        || opcode.starts_with("ImageAtomic")
}

fn image_component_type(binding: &ImageBinding) -> &'static str {
    let Some(word) = binding.resource_descriptor.get(1) else {
        return "float";
    };
    let unified_format = (word >> 20) & 0x1ff;
    let Some((_, number_format)) = decode_unified_format(unified_format) else {
        return "float";
    };
    match number_format {
        4 => "uint",
        5 => "int",
        _ => "float",
    }
}

#[cfg(test)]
mod tests {
    use remu_shader::{
        ComputeSystemRegisters, DataShareControl, Dpp8Control, DppControl, Instruction, Program,
        RawOpcodeFamily, ShaderEvaluation, Vop3pControl, decode_program,
    };

    use super::*;

    fn empty_evaluation() -> ShaderEvaluation {
        ShaderEvaluation {
            initial_scalar_registers: vec![0; 128].into(),
            scalar_registers: vec![0; 128].into(),
            scalar_memory_reads: Vec::new(),
            image_bindings: Vec::new(),
            global_memory_bindings: Vec::new(),
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        }
    }

    fn vector_instruction(
        name: &'static str,
        source_count: usize,
        control: InstructionControl,
    ) -> Instruction {
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop2,
            opcode: Opcode::Known(name),
            words: vec![0, 0],
            sources: (1..=source_count)
                .map(|register| {
                    Operand::vector(u32::try_from(register).expect("test source index fits u32"))
                })
                .collect(),
            destinations: vec![Operand::vector(8)],
            control: Some(control),
        }
    }

    fn vop3p_instruction(name: &'static str, control: Vop3pControl) -> Instruction {
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop3p,
            opcode: Opcode::Known(name),
            words: vec![0, 0],
            sources: vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
            destinations: vec![Operand::vector(8)],
            control: Some(InstructionControl::Vop3p(control)),
        }
    }

    const DS_OPCODES: [(u32, &str); 37] = [
        (0x00, "DsAddU32"),
        (0x01, "DsSubU32"),
        (0x03, "DsIncU32"),
        (0x04, "DsDecU32"),
        (0x05, "DsMinI32"),
        (0x06, "DsMaxI32"),
        (0x07, "DsMinU32"),
        (0x08, "DsMaxU32"),
        (0x09, "DsAndB32"),
        (0x0a, "DsOrB32"),
        (0x0b, "DsXorB32"),
        (0x0d, "DsWriteB32"),
        (0x0e, "DsWrite2B32"),
        (0x0f, "DsWrite2St64B32"),
        (0x10, "DsCmpstB32"),
        (0x20, "DsAddRtnU32"),
        (0x21, "DsSubRtnU32"),
        (0x23, "DsIncRtnU32"),
        (0x24, "DsDecRtnU32"),
        (0x25, "DsMinRtnI32"),
        (0x26, "DsMaxRtnI32"),
        (0x27, "DsMinRtnU32"),
        (0x28, "DsMaxRtnU32"),
        (0x29, "DsAndRtnB32"),
        (0x2a, "DsOrRtnB32"),
        (0x2b, "DsXorRtnB32"),
        (0x2d, "DsWrxchgRtnB32"),
        (0x30, "DsCmpstRtnB32"),
        (0x35, "DsSwizzleB32"),
        (0x36, "DsReadB32"),
        (0x37, "DsRead2B32"),
        (0x38, "DsRead2St64B32"),
        (0x4d, "DsWriteB64"),
        (0xde, "DsWriteB96"),
        (0xdf, "DsWriteB128"),
        (0xfe, "DsReadB96"),
        (0xff, "DsReadB128"),
    ];

    fn decoded_ds_program(offset: u16, gds: bool) -> Program {
        let mut words = Vec::with_capacity(DS_OPCODES.len() * 2 + 1);
        for (opcode, _) in DS_OPCODES {
            words.push(0xd800_0000 | (opcode << 18) | (u32::from(gds) << 17) | u32::from(offset));
            words.push(1u32 | (2u32 << 8) | (3u32 << 16) | (8u32 << 24));
        }
        words.extend([0xbf8a_0000, 0xbf81_0000]);
        decode_program(0x1000, &words).expect("complete DS test program")
    }

    fn ds_swizzle_instruction(offset: u16) -> Instruction {
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Ds,
            opcode: Opcode::Known("DsSwizzleB32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(2)],
            destinations: vec![Operand::vector(8)],
            control: Some(InstructionControl::DataShare(DataShareControl {
                offset0: u32::from(offset & 0xff),
                offset1: u32::from(offset >> 8),
                gds: false,
            })),
        }
    }

    fn compile_vector_instruction(
        instruction: Instruction,
        stage: MetalShaderStage,
    ) -> MetalShader {
        compile_msl(
            &Program::new(0x1000, vec![instruction]),
            &empty_evaluation(),
            &MetalCompileOptions {
                stage,
                required_vertex_output_count: 0,
                pixel_outputs: Vec::new(),
                pixel_input_enable: 0,
                pixel_input_address: 0,
                diagnostic_force_visible_clip_depth: false,
                diagnostic_force_vertex_pattern: false,
                diagnostic_force_vertex_fetch_id: false,
                diagnostic_capture_vertex_position: false,
                diagnostic_capture_pixel_output: false,
                diagnostic_force_pixel_color: false,
                lds_size_bytes: 0,
                wave_lane_count: 32,
            },
        )
        .expect("vector instruction should lower")
    }

    #[test]
    fn lowers_observed_64_bit_and_not_scalar_operation() {
        let shader = compile_vector_instruction(
            Instruction {
                pc: 0x30,
                encoding: ShaderEncoding::Sop2,
                opcode: Opcode::Known("SAndn2B64"),
                words: vec![0],
                sources: vec![Operand::scalar(10), Operand::scalar(12)],
                destinations: vec![Operand::scalar(20)],
                control: None,
            },
            MetalShaderStage::Pixel,
        );

        assert!(shader.source.contains("& ~"));
        assert!(shader.source.contains("s[20]"));
        assert!(shader.source.contains("s[21]"));
    }

    #[test]
    fn lowers_observed_bit_reverse_and_wave32_saveexec_operations() {
        let reverse = compile_vector_instruction(
            Instruction {
                pc: 0x194,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VBfrevB32"),
                words: vec![0],
                sources: vec![Operand::vector(2)],
                destinations: vec![Operand::vector(8)],
                control: None,
            },
            MetalShaderStage::Pixel,
        );
        assert!(reverse.source.contains("reverse_bits(remu_v_2)"));

        let saveexec = compile_vector_instruction(
            Instruction {
                pc: 0x74,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::Known("SAndSaveexecB64"),
                words: vec![0],
                sources: vec![Operand::scalar(10)],
                destinations: vec![Operand::scalar(20)],
                control: None,
            },
            MetalShaderStage::Pixel,
        );
        assert!(saveexec.source.contains("simd_ballot(exec)"));
        assert!(saveexec.source.contains("& 0xFFFFFFFFul"));
        assert!(saveexec.source.contains("remu_s_20 = (uint)"));
        assert!(saveexec.source.contains("exec = (("));
    }

    #[test]
    fn lowers_structured_wave64_saveexec_into_the_correct_simd32_half() {
        let program = Program::new(
            0x1000,
            vec![Instruction {
                pc: 0x7cc,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::Known("SAndSaveexecB64"),
                words: vec![0],
                sources: vec![Operand::scalar(10)],
                destinations: vec![Operand::scalar(20)],
                control: None,
            }],
        );
        let shader = compile_msl(
            &program,
            &empty_evaluation(),
            &MetalCompileOptions::compute_wave(0, 64),
        )
        .expect("split wave64 saveexec should lower");

        assert!(
            shader
                .source
                .contains("uint remu_thread_index [[thread_index_in_threadgroup]]")
        );
        assert!(shader.source.contains("uint remu_wave_half ="));
        assert!(shader.source.contains("remu_wave_half * 32u"));
        assert!(shader.source.contains("0xFFFFFFFFFFFFFFFFul"));
        assert!(shader.source.contains("remu_s_127 = (uint)(t"));
        assert!(shader.source.contains("remu_wave_half == 0u ?"));
    }

    #[test]
    fn lowers_readlane_and_writelane_independently_of_exec() {
        let read = compile_vector_instruction(
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Vop3,
                opcode: Opcode::Known("VReadlaneB32"),
                words: vec![0, 0],
                sources: vec![Operand::vector(2), Operand::scalar(4)],
                destinations: vec![Operand::scalar(8)],
                control: None,
            },
            MetalShaderStage::Pixel,
        );
        assert!(
            read.source
                .contains("simd_shuffle(remu_v_2, (ushort)(remu_s_4 & 31u))")
        );
        assert!(read.source.contains("remu_s_8 = "));

        let write = compile_vector_instruction(
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Vop3,
                opcode: Opcode::Known("VWritelaneB32"),
                words: vec![0, 0],
                sources: vec![Operand::scalar(2), Operand::scalar(4)],
                destinations: vec![Operand::vector(8)],
                control: None,
            },
            MetalShaderStage::Pixel,
        );
        assert!(
            write
                .source
                .contains("if (remu_lane == ((remu_s_4) & 31u)) remu_v_8 = remu_s_2;")
        );
        assert!(!write.source.contains("if (exec) remu_v_8 ="));
    }

    #[test]
    fn lowers_vector_carry_and_borrow_with_wave32_lane_masks() {
        let mut instructions = ["VAddcU32", "VSubbU32", "VSubbrevU32"]
            .into_iter()
            .enumerate()
            .map(|(index, name)| Instruction {
                pc: u32::try_from(index * 4).expect("small test pc"),
                encoding: ShaderEncoding::Vop2,
                opcode: Opcode::Known(name),
                words: vec![0],
                sources: vec![Operand::vector(1), Operand::vector(2)],
                destinations: vec![Operand::vector(
                    8 + u32::try_from(index).expect("small destination"),
                )],
                control: None,
            })
            .collect::<Vec<_>>();
        for (index, name) in ["VAddCoU32", "VSubCoU32", "VSubrevCoU32", "VAddCoCiU32"]
            .into_iter()
            .enumerate()
        {
            instructions.push(Instruction {
                pc: 12 + u32::try_from(index * 8).expect("small test pc"),
                encoding: ShaderEncoding::Vop3,
                opcode: Opcode::Known(name),
                words: vec![0, 0],
                sources: vec![Operand::vector(1), Operand::vector(2), Operand::scalar(20)],
                destinations: vec![Operand::vector(
                    12 + u32::try_from(index).expect("small destination"),
                )],
                control: Some(InstructionControl::Vop3(remu_shader::Vop3Control {
                    absolute_mask: 0,
                    negate_mask: 0,
                    output_modifier: 0,
                    clamp: false,
                    operand_select: 0,
                    scalar_destination: Some(12),
                })),
            });
        }
        let shader = compile_msl(
            &Program::new(0x1000, instructions),
            &empty_evaluation(),
            &MetalCompileOptions::pixel(Vec::new(), 0, 0),
        )
        .expect("carry and borrow operations should lower");

        assert!(shader.source.contains("ulong t"));
        assert!(shader.source.contains(">> 32) != 0ul"));
        assert!(shader.source.contains(" + (ulong)"));
        assert!(shader.source.contains(" < t"));
        assert!(shader.source.contains("simd_ballot"));
        assert!(shader.source.contains("remu_s_12 ="));
        assert!(shader.source.contains("remu_s_106 ="));
    }

    #[test]
    fn lowers_scalar_bitfield_mask_operation() {
        let shader = compile_vector_instruction(
            Instruction {
                pc: 0x220,
                encoding: ShaderEncoding::Sop2,
                opcode: Opcode::Known("SBfmB32"),
                words: vec![0],
                sources: vec![Operand::scalar(10), Operand::scalar(12)],
                destinations: vec![Operand::scalar(20)],
                control: None,
            },
            MetalShaderStage::Pixel,
        );

        assert!(shader.source.contains("1u << ("));
        assert!(shader.source.contains("& 31u)) - 1u) << ("));
        assert!(shader.source.contains("remu_s_20"));
    }

    #[test]
    fn directly_lowers_single_path_control_flow_and_scalarizes_live_registers() {
        let program = Program::new(
            0x1000,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Sop1,
                    opcode: Opcode::Known("SMovB32"),
                    words: vec![0],
                    sources: vec![Operand::scalar(12)],
                    destinations: vec![Operand::scalar(80)],
                    control: None,
                },
                Instruction {
                    pc: 4,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![Operand::scalar(80)],
                    destinations: vec![Operand::vector(200)],
                    control: None,
                },
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ],
        );
        let shader = compile_msl(
            &program,
            &empty_evaluation(),
            &MetalCompileOptions::compute(0),
        )
        .expect("single-path program should lower directly");

        assert!(shader.register_footprint.direct_control_flow);
        assert_eq!(shader.register_footprint.scalar_values, 2);
        assert_eq!(shader.register_footprint.vector_values, 1);
        assert!(
            shader
                .source
                .contains("uint remu_s_12 = remu_initial_s[12];")
        );
        assert!(shader.source.contains("uint remu_v_200 = 0u;"));
        assert!(!shader.source.contains("uint s[128]"));
        assert!(!shader.source.contains("uint v[256]"));
        assert!(!shader.source.contains("while (active)"));
        assert!(!shader.source.contains("switch (pc)"));
    }

    #[test]
    fn resolves_acyclic_unconditional_branches_without_lowering_dead_instructions() {
        let program = Program::new(
            0x1000,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::Known("SBranch"),
                    // PC 0 + 4-byte instruction + one dword = PC 8.
                    words: vec![1],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
                Instruction {
                    pc: 4,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![Operand::vector(99)],
                    destinations: vec![Operand::vector(100)],
                    control: None,
                },
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ],
        );
        let shader = compile_msl(
            &program,
            &empty_evaluation(),
            &MetalCompileOptions::compute(0),
        )
        .expect("static unconditional path should lower");

        assert!(shader.register_footprint.direct_control_flow);
        assert_eq!(shader.register_footprint.vector_values, 0);
        assert!(!shader.source.contains("remu_v_99"));
        assert!(!shader.source.contains("switch (pc)"));
    }

    #[test]
    fn conditional_control_flow_keeps_exact_dispatcher_but_not_register_arrays() {
        let program = Program::new(
            0x1000,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::Known("SCbranchScc1"),
                    words: vec![1],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
                Instruction {
                    pc: 4,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ],
        );
        let shader = compile_msl(
            &program,
            &empty_evaluation(),
            &MetalCompileOptions::compute(0),
        )
        .expect("conditional program should retain dispatcher");

        assert!(!shader.register_footprint.direct_control_flow);
        assert!(shader.source.contains("while (active)"));
        assert!(shader.source.contains("switch (pc)"));
        assert!(!shader.source.contains("uint s[128]"));
        assert!(!shader.source.contains("uint v[256]"));
    }

    #[test]
    fn dispatcher_switches_once_per_basic_block_instead_of_once_per_instruction() {
        let program = Program::new(
            0x1000,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::Known("SCbranchScc1"),
                    // PC 0 + 4-byte instruction + three dwords = PC 16.
                    words: vec![3],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
                Instruction {
                    pc: 4,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![Operand::vector(1)],
                    destinations: vec![Operand::vector(2)],
                    control: None,
                },
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![Operand::vector(2)],
                    destinations: vec![Operand::vector(3)],
                    control: None,
                },
                Instruction {
                    pc: 12,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::Known("SBranch"),
                    // PC 12 + 4-byte instruction + one dword = PC 20.
                    words: vec![1],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
                Instruction {
                    pc: 16,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![Operand::vector(4)],
                    destinations: vec![Operand::vector(5)],
                    control: None,
                },
                Instruction {
                    pc: 20,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ],
        );
        let shader = compile_msl(
            &program,
            &empty_evaluation(),
            &MetalCompileOptions::compute(0),
        )
        .expect("conditional blocks should lower");

        for pc in [0, 4, 16, 20] {
            assert!(shader.source.contains(&format!("case {pc}u")));
        }
        for pc in [8, 12] {
            assert!(!shader.source.contains(&format!("case {pc}u")));
        }
        assert_eq!(shader.source.matches("case ").count(), 4);
    }

    #[test]
    fn rejects_branches_to_non_instruction_boundaries() {
        let program = Program::new(
            0x1000,
            vec![Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::Known("SBranch"),
                words: vec![1],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            }],
        );
        let error = compile_msl(
            &program,
            &empty_evaluation(),
            &MetalCompileOptions::compute(0),
        )
        .expect_err("invalid branch target must not silently end the shader");
        assert!(error.to_string().contains("is not an instruction boundary"));
    }

    #[test]
    fn reports_bounded_codegen_footprint_for_a_long_straight_line_shader() {
        let mut instructions = (0..64_u32)
            .map(|index| Instruction {
                pc: index * 4,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VMovB32"),
                words: vec![0],
                sources: vec![Operand::vector(1)],
                destinations: vec![Operand::vector(8)],
                control: None,
            })
            .collect::<Vec<_>>();
        instructions.push(Instruction {
            pc: 64 * 4,
            encoding: ShaderEncoding::Sopp,
            opcode: Opcode::END_PROGRAM,
            words: vec![0xbf81_0000],
            sources: Vec::new(),
            destinations: Vec::new(),
            control: None,
        });
        let shader = compile_msl(
            &Program::new(0x1000, instructions),
            &empty_evaluation(),
            &MetalCompileOptions::compute(0),
        )
        .expect("long straight-line shader should lower");

        eprintln!(
            "remu Metal codegen: guest_instructions=65 source_bytes={} scalar_values={} vector_values={} dispatcher={}",
            shader.source.len(),
            shader.register_footprint.scalar_values,
            shader.register_footprint.vector_values,
            !shader.register_footprint.direct_control_flow,
        );
        assert_eq!(shader.register_footprint.scalar_values, 0);
        assert_eq!(shader.register_footprint.vector_values, 2);
        assert!(shader.register_footprint.direct_control_flow);
        assert!(!shader.source.contains("case 0u"));
        assert!(!shader.source.contains("steps"));
    }

    const BUFFER_MEMORY_OPCODES: [&str; 60] = [
        "BufferLoadFormatX",
        "BufferLoadFormatXy",
        "BufferLoadFormatXyz",
        "BufferLoadFormatXyzw",
        "BufferStoreFormatX",
        "BufferStoreFormatXy",
        "BufferStoreFormatXyz",
        "BufferStoreFormatXyzw",
        "BufferLoadUbyte",
        "BufferLoadSbyte",
        "BufferLoadUshort",
        "BufferLoadSshort",
        "BufferLoadDword",
        "BufferLoadDwordx2",
        "BufferLoadDwordx4",
        "BufferLoadDwordx3",
        "BufferStoreByte",
        "BufferStoreByteD16Hi",
        "BufferStoreShort",
        "BufferStoreShortD16Hi",
        "BufferStoreDword",
        "BufferStoreDwordx2",
        "BufferStoreDwordx4",
        "BufferStoreDwordx3",
        "BufferLoadUbyteD16",
        "BufferLoadUbyteD16Hi",
        "BufferLoadSbyteD16",
        "BufferLoadSbyteD16Hi",
        "BufferLoadShortD16",
        "BufferLoadShortD16Hi",
        "BufferLoadFormatD16HiX",
        "BufferStoreFormatD16HiX",
        "BufferAtomicSwap",
        "BufferAtomicCmpswap",
        "BufferAtomicAdd",
        "BufferAtomicSub",
        "BufferAtomicSmin",
        "BufferAtomicUmin",
        "BufferAtomicSmax",
        "BufferAtomicUmax",
        "BufferAtomicAnd",
        "BufferAtomicOr",
        "BufferAtomicXor",
        "BufferAtomicInc",
        "BufferAtomicDec",
        "BufferLoadFormatD16X",
        "BufferLoadFormatD16Xy",
        "BufferLoadFormatD16Xyz",
        "BufferLoadFormatD16Xyzw",
        "BufferStoreFormatD16X",
        "BufferStoreFormatD16Xy",
        "BufferStoreFormatD16Xyz",
        "BufferStoreFormatD16Xyzw",
        "TBufferLoadFormatX",
        "TBufferLoadFormatXy",
        "TBufferLoadFormatXyz",
        "TBufferLoadFormatXyzw",
        "TBufferStoreFormatX",
        "TBufferStoreFormatXy",
        "TBufferStoreFormatXyz",
    ];

    const EXTRA_BUFFER_MEMORY_OPCODE: &str = "TBufferStoreFormatXyzw";
    const TYPED_D16_MEMORY_OPCODES: [&str; 8] = [
        "TBufferLoadFormatD16X",
        "TBufferLoadFormatD16Xy",
        "TBufferLoadFormatD16Xyz",
        "TBufferLoadFormatD16Xyzw",
        "TBufferStoreFormatD16X",
        "TBufferStoreFormatD16Xy",
        "TBufferStoreFormatD16Xyz",
        "TBufferStoreFormatD16Xyzw",
    ];

    const GLOBAL_MEMORY_OPCODES: [&str; 24] = [
        "GlobalLoadUbyte",
        "GlobalLoadSbyte",
        "GlobalLoadUshort",
        "GlobalLoadSshort",
        "GlobalLoadDword",
        "GlobalLoadDwordx2",
        "GlobalLoadDwordx4",
        "GlobalLoadDwordx3",
        "GlobalStoreByte",
        "GlobalStoreByteD16Hi",
        "GlobalStoreShort",
        "GlobalStoreShortD16Hi",
        "GlobalStoreDword",
        "GlobalStoreDwordx2",
        "GlobalStoreDwordx4",
        "GlobalStoreDwordx3",
        "GlobalLoadUbyteD16",
        "GlobalLoadUbyteD16Hi",
        "GlobalLoadSbyteD16",
        "GlobalLoadSbyteD16Hi",
        "GlobalLoadShortD16",
        "GlobalLoadShortD16Hi",
        "GlobalAtomicAdd",
        "GlobalAtomicUMax",
    ];

    fn memory_dword_count_for_test(name: &str) -> u32 {
        if name.contains("FormatD16") {
            if name.ends_with("Xyz") || name.ends_with("Xyzw") {
                2
            } else {
                1
            }
        } else if name.ends_with("Xyzw") || name.ends_with("Dwordx4") {
            4
        } else if name.ends_with("Xyz") || name.ends_with("Dwordx3") {
            3
        } else if name.ends_with("Xy") || name.ends_with("Dwordx2") || name.ends_with("Cmpswap") {
            2
        } else {
            1
        }
    }

    fn complete_memory_program() -> (Program, ShaderEvaluation) {
        let names = BUFFER_MEMORY_OPCODES
            .into_iter()
            .chain([EXTRA_BUFFER_MEMORY_OPCODE])
            .chain(TYPED_D16_MEMORY_OPCODES)
            .chain(GLOBAL_MEMORY_OPCODES);
        let mut pcs = Vec::new();
        let mut instructions = names
            .enumerate()
            .map(|(index, name)| {
                let pc = u32::try_from(index * 8).expect("small memory test program");
                pcs.push(pc);
                let dword_count = memory_dword_count_for_test(name);
                let (encoding, control) = if name.starts_with("Global") {
                    (
                        ShaderEncoding::Flat,
                        InstructionControl::GlobalMemory(remu_shader::GlobalMemoryControl {
                            dword_count,
                            vector_address: 1,
                            vector_data: 16,
                            vector_destination: 32,
                            scalar_address: Some(8),
                            offset_bytes: -4,
                            dlc: true,
                            lds: false,
                            glc: true,
                            slc: true,
                        }),
                    )
                } else {
                    (
                        if name.starts_with("TBuffer") {
                            ShaderEncoding::Mtbuf
                        } else {
                            ShaderEncoding::Mubuf
                        },
                        InstructionControl::BufferMemory(remu_shader::BufferMemoryControl {
                            dword_count,
                            vector_address: 1,
                            vector_data: 16,
                            scalar_resource: 0,
                            instruction_format: name.starts_with("TBuffer").then_some(60),
                            offset_bytes: 4,
                            index_enabled: true,
                            offset_enabled: true,
                            dlc: true,
                            lds: false,
                            glc: true,
                            slc: true,
                            tfe: name.contains("Load") && !name.contains("Atomic"),
                        }),
                    )
                };
                Instruction {
                    pc,
                    encoding,
                    opcode: Opcode::Known(name),
                    words: vec![0, 0],
                    sources: vec![
                        Operand::vector(1),
                        Operand::scalar(0),
                        Operand::source(128, None),
                    ],
                    destinations: Vec::new(),
                    control: Some(control),
                }
            })
            .collect::<Vec<_>>();
        let image_pc = u32::try_from(instructions.len() * 8).expect("small memory test program");
        let image_control = remu_shader::ImageControl {
            dmask: 0xf,
            vector_address: 1,
            address_registers: vec![1, 2],
            vector_data: 64,
            scalar_resource: 16,
            scalar_sampler: 24,
            dimension: 1,
            is_array: false,
            glc: true,
            slc: true,
            a16: false,
            d16: false,
        };
        instructions.push(Instruction {
            pc: image_pc,
            encoding: ShaderEncoding::Mimg,
            opcode: Opcode::Known("ImageLoad"),
            words: vec![0, 0],
            sources: vec![Operand::vector(1), Operand::vector(2), Operand::scalar(16)],
            destinations: (64..68).map(Operand::vector).collect(),
            control: Some(InstructionControl::Image(image_control.clone())),
        });
        let image_store_pc = image_pc + 8;
        let mut image_store_control = image_control.clone();
        image_store_control.vector_data = 68;
        image_store_control.dimension = 5;
        image_store_control.is_array = true;
        instructions.push(Instruction {
            pc: image_store_pc,
            encoding: ShaderEncoding::Mimg,
            opcode: Opcode::Known("ImageStore"),
            words: vec![0, 0],
            sources: vec![
                Operand::vector(1),
                Operand::vector(2),
                Operand::scalar(16),
                Operand::vector(68),
                Operand::vector(69),
                Operand::vector(70),
                Operand::vector(71),
            ],
            destinations: Vec::new(),
            control: Some(InstructionControl::Image(image_store_control.clone())),
        });
        let end_pc = u32::try_from(instructions.len() * 8).expect("small memory test program");
        instructions.push(Instruction {
            pc: end_pc,
            encoding: ShaderEncoding::Sopp,
            opcode: Opcode::END_PROGRAM,
            words: vec![0xbf81_0000],
            sources: Vec::new(),
            destinations: Vec::new(),
            control: None,
        });
        let mut evaluation = empty_evaluation();
        let selectors = 4 | (5 << 3) | (6 << 6) | (7 << 9);
        evaluation.initial_scalar_registers[0] = 0x1000;
        evaluation.initial_scalar_registers[1] = 16 << 16;
        evaluation.initial_scalar_registers[2] = 64 * 1024;
        evaluation.initial_scalar_registers[3] = selectors | (60 << 12) | (1 << 24) | (3 << 28);
        evaluation.initial_scalar_registers[8] = 0x1000;
        evaluation.scalar_registers = evaluation.initial_scalar_registers.clone();
        evaluation
            .global_memory_bindings
            .push(remu_shader::GlobalMemoryBinding {
                scalar_address: 0,
                base_address: 0x1000,
                instruction_pcs: pcs.into(),
                byte_count: 64 * 1024,
                data: vec![0; 64 * 1024],
                writable: true,
                write_back_to_guest: true,
            });
        let mut identity_image_descriptor = vec![0; 8];
        identity_image_descriptor[3] = 0xfac;
        evaluation.image_bindings.push(remu_shader::ImageBinding {
            pc: image_pc,
            opcode: Opcode::Known("ImageLoad"),
            control: image_control,
            resource_descriptor: identity_image_descriptor.clone().into(),
            sampler_descriptor: Vec::new().into(),
            mip_level: None,
        });
        let mut swizzled_store_descriptor = identity_image_descriptor;
        swizzled_store_descriptor[3] = 0x9f5;
        evaluation.image_bindings.push(remu_shader::ImageBinding {
            pc: image_store_pc,
            opcode: Opcode::Known("ImageStore"),
            control: image_store_control,
            resource_descriptor: swizzled_store_descriptor.into(),
            sampler_descriptor: Vec::new().into(),
            mip_level: None,
        });
        (Program::new(0x1000, instructions), evaluation)
    }

    #[test]
    fn lowers_complete_decoded_global_mubuf_and_mtbuf_memory_surface() {
        let (program, evaluation) = complete_memory_program();
        let shader = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
            .expect("complete decoded VMEM surface should lower");
        for operation in [
            "remu_atomic_cmpswap",
            "remu_atomic_inc",
            "remu_store_format_bits",
            "remu_load_format_bits",
            "remu_format_to_d16",
            "remu_format_from_d16",
        ] {
            assert!(shader.source.contains(operation), "missing {operation}");
        }
        assert!(shader.source.contains("0xFFFF0000u"));
        assert!(shader.source.contains("0xFFFFFF00u"));
        assert!(shader.source.contains("remu_range_valid"));
        assert!(
            shader
                .source
                .contains("texture2d<float, access::read_write> tex0")
        );
        assert!(shader.source.contains("tex0.read(uint2("));
        assert!(shader.source.contains("tex1.write("));
        assert!(
            shader.source.contains(" & 3ul"),
            "buffer descriptors must address their dynamically bound, dword-aligned Metal buffer without embedding the evaluated guest base"
        );
    }

    #[test]
    fn sampled_images_apply_dynamic_descriptor_component_selection_in_the_shader() {
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
        let instruction = Instruction {
            pc: 0,
            encoding: ShaderEncoding::Mimg,
            opcode: Opcode::Known("ImageSample"),
            words: vec![0, 0],
            sources: vec![Operand::vector(0), Operand::vector(1), Operand::scalar(0)],
            destinations: (4..8).map(Operand::vector).collect(),
            control: Some(InstructionControl::Image(control.clone())),
        };
        let mut evaluation = empty_evaluation();
        let mut resource_descriptor = vec![0; 8];
        // R32F sampled as R,R,R,R, the live Superliminal descriptor that previously failed
        // because the host tried to materialize the swizzle by changing the texture format.
        resource_descriptor[3] = 0x924;
        evaluation.image_bindings.push(ImageBinding {
            pc: 0,
            opcode: Opcode::Known("ImageSample"),
            control,
            resource_descriptor: resource_descriptor.into(),
            sampler_descriptor: vec![0; 4].into(),
            mip_level: None,
        });
        let shader = compile_msl(
            &Program::new(0x1000, vec![instruction]),
            &evaluation,
            &MetalCompileOptions::compute(0),
        )
        .expect("sampled R32 swizzle should lower");
        assert!(shader.source.contains("remu_s_3 & 0xFFFu"));
        assert!(shader.source.contains("== 0u ? 0xFACu"));
        assert!(shader.source.contains(">= 4u &&"));
        assert!(shader.source.contains("<= 7u ?"));
        assert!(shader.source.contains("tex0.sample("));
    }

    #[test]
    fn comparison_samples_use_depth_texture_typing_and_the_z_reference_operand() {
        let control = ImageControl {
            dmask: 1,
            vector_address: 0,
            address_registers: vec![0, 1, 2],
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
        let instruction = Instruction {
            pc: 0,
            encoding: ShaderEncoding::Mimg,
            opcode: Opcode::Known("ImageSampleCLz"),
            words: vec![0, 0],
            sources: vec![
                Operand::vector(0),
                Operand::vector(1),
                Operand::vector(2),
                Operand::scalar(0),
                Operand::scalar(8),
            ],
            destinations: vec![Operand::vector(4)],
            control: Some(InstructionControl::Image(control.clone())),
        };
        let mut evaluation = empty_evaluation();
        let mut resource_descriptor = vec![0; 8];
        resource_descriptor[3] = 0x204 | (9 << 28);
        evaluation.image_bindings.push(ImageBinding {
            pc: 0,
            opcode: Opcode::Known("ImageSampleCLz"),
            control,
            resource_descriptor: resource_descriptor.into(),
            sampler_descriptor: vec![3 << 12, 0, 0, 0].into(),
            mip_level: None,
        });
        let shader = compile_msl(
            &Program::new(0x1000, vec![instruction]),
            &evaluation,
            &MetalCompileOptions::pixel(Vec::new(), 0, 0),
        )
        .expect("depth comparison sample should lower");

        assert!(shader.source.contains("depth2d<float> tex0"));
        assert!(shader.source.contains("tex0.sample_compare("));
        assert!(shader.source.contains("as_type<float>(remu_v_0)"));
        assert!(
            shader
                .source
                .contains("float2(as_type<float>(remu_v_1), as_type<float>(remu_v_2))")
        );
        assert!(shader.source.contains(", level(0.0f))"));
    }

    #[test]
    fn range_checks_raw_dword_vectors_per_component() {
        let expression = buffer_oob_expression(
            "null",
            "mode",
            "index",
            "records",
            "offset",
            "stride",
            "swizzled",
            "buffer_offset",
            "reduced_records",
            "4u",
            12,
        );

        assert!(expression.contains("offset + 12u >= stride"));
        assert!(expression.contains("remu_range_valid(offset + 12u, 4u, stride)"));
        assert!(expression.contains("remu_range_valid(buffer_offset + 12u, 4u, reduced_records)"));
    }

    #[test]
    fn formatted_elements_use_record_stride_without_total_size_alignment() {
        let expression =
            format_buffer_element_oob_expression("descriptor_oob", "record_address", "12u", "28u");

        assert_eq!(
            expression,
            "descriptor_oob || 12u == 0u || !remu_range_valid(record_address, 12u, 28u)"
        );
        assert!(
            !expression.contains('%'),
            "a 12-byte float3 at offsets 0, 28, and 56 is legal in a 28-byte record layout"
        );
    }

    #[test]
    fn initializes_complete_rdna2_compute_system_register_state() {
        let program = Program::new(
            0x1000,
            [4_u32, 5, 6, 7]
                .into_iter()
                .enumerate()
                .map(|(index, source)| Instruction {
                    pc: u32::try_from(index * 4).expect("small system-register program"),
                    encoding: ShaderEncoding::Sop1,
                    opcode: Opcode::Known("SMovB32"),
                    words: vec![0],
                    sources: vec![Operand::scalar(source)],
                    destinations: vec![Operand::scalar(20 + source)],
                    control: None,
                })
                .chain([
                    Instruction {
                        pc: 16,
                        encoding: ShaderEncoding::Vop1,
                        opcode: Opcode::Known("VMovB32"),
                        words: vec![0],
                        sources: vec![Operand::vector(0)],
                        destinations: vec![Operand::vector(20)],
                        control: None,
                    },
                    Instruction {
                        pc: 20,
                        encoding: ShaderEncoding::Sopp,
                        opcode: Opcode::END_PROGRAM,
                        words: vec![0],
                        sources: Vec::new(),
                        destinations: Vec::new(),
                        control: None,
                    },
                ])
                .collect(),
        );
        let mut evaluation = empty_evaluation();
        evaluation.compute_system_registers = Some(ComputeSystemRegisters {
            work_group_x: Some(4),
            work_group_y: Some(5),
            work_group_z: Some(6),
            thread_group_size: Some(7),
        });

        let shader = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
            .expect("compute system register lowering");

        assert!(shader.source.contains("remu_v_0 = remu_local_id.x"));
        assert!(shader.source.contains("remu_s_4 = remu_group_id.x;"));
        assert!(shader.source.contains("remu_s_5 = remu_group_id.y;"));
        assert!(shader.source.contains("remu_s_6 = remu_group_id.z;"));
        assert!(
            shader
                .source
                .contains("uint remu_thread_index [[thread_index_in_threadgroup]]")
        );
        assert!(
            shader
                .source
                .contains("uint3 remu_group_size [[threads_per_threadgroup]]")
        );
        assert!(
            shader
                .source
                .contains("(remu_wave_id == 0u ? 0x80000000u : 0u)")
        );
        assert!(shader.source.contains("((remu_wave_id & 31u) << 20u)"));
        assert!(shader.source.contains("(remu_threadgroup_waves & 63u)"));
    }

    #[test]
    fn rejects_compute_system_registers_outside_compute_and_sgpr_bounds() {
        let program = Program::new(
            0x1000,
            vec![Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            }],
        );
        let mut evaluation = empty_evaluation();
        evaluation.compute_system_registers = Some(ComputeSystemRegisters {
            work_group_x: Some(128),
            ..ComputeSystemRegisters::default()
        });
        let error = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
            .expect_err("out-of-range system SGPR must be rejected");
        assert!(
            error
                .to_string()
                .contains("exceeds the scalar register file")
        );

        evaluation.compute_system_registers = Some(ComputeSystemRegisters {
            work_group_x: Some(4),
            ..ComputeSystemRegisters::default()
        });
        let error = compile_msl(&program, &evaluation, &MetalCompileOptions::vertex(0))
            .expect_err("graphics stage must reject compute system SGPRs");
        assert!(error.to_string().contains("require a compute shader stage"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn generated_memory_msl_compiles_with_apple_metal() {
        use std::path::PathBuf;
        use std::process::Command;
        use std::sync::atomic::{AtomicUsize, Ordering};

        static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

        fn find_metal_compiler() -> Option<PathBuf> {
            if let Some(path) = std::env::var_os("REMU_METAL_COMPILER").map(PathBuf::from)
                && path.is_file()
            {
                return Some(path);
            }
            let bundled = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../.remu/toolchains/Metal.xctoolchain/usr/bin/metal");
            if bundled.is_file() {
                return Some(bundled);
            }
            let output = Command::new("xcrun")
                .args(["--find", "metal"])
                .output()
                .ok()?;
            output
                .status
                .success()
                .then(|| PathBuf::from(String::from_utf8_lossy(&output.stdout).trim().to_owned()))
        }

        let Some(compiler) = find_metal_compiler() else {
            eprintln!("skipping Apple Metal memory compile test: no metal compiler found");
            return;
        };
        let (program, mut evaluation) = complete_memory_program();
        evaluation.compute_system_registers = Some(ComputeSystemRegisters {
            work_group_x: Some(116),
            work_group_y: Some(117),
            work_group_z: Some(118),
            thread_group_size: Some(119),
        });
        let shader = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
            .expect("complete decoded VMEM surface should lower");
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "remu-memory-metal-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).expect("create Metal memory test directory");
        let source = directory.join("memory.metal");
        let air = directory.join("memory.air");
        std::fs::write(&source, shader.source).expect("write generated memory MSL");
        let output = Command::new(&compiler)
            .args(["-c", "-target", "air64-apple-macosx15.0"])
            .arg("-fmetal-math-mode=safe")
            .arg("-o")
            .arg(&air)
            .arg(&source)
            .output()
            .expect("run Apple Metal compiler");
        assert!(
            output.status.success(),
            "memory MSL failed to compile:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        std::fs::remove_dir_all(directory).expect("remove Metal memory test directory");
    }

    #[test]
    fn rejects_raw_opcodes_instead_of_emitting_placeholder_code() {
        let program = Program::new(
            0x1000,
            vec![Instruction {
                pc: 0,
                encoding: ShaderEncoding::Vop3,
                opcode: Opcode::Raw {
                    family: remu_shader::RawOpcodeFamily::Vop3,
                    code: 0x1ff,
                },
                words: vec![0, 0],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            }],
        );
        let error = compile_msl(
            &program,
            &empty_evaluation(),
            &MetalCompileOptions::vertex(0),
        )
        .expect_err("raw opcode must fail");
        assert!(error.to_string().contains("raw opcode"));
    }

    #[test]
    fn lowers_every_decoded_ds_operation_with_architectural_offsets_and_atomics() {
        let program = decoded_ds_program(0x1234, false);
        for (instruction, (_, name)) in program.instructions.iter().zip(DS_OPCODES) {
            assert!(instruction.opcode.is(name));
        }
        let shader = compile_msl(
            &program,
            &empty_evaluation(),
            &MetalCompileOptions::compute(32 * 1024),
        )
        .expect("every decoded DS operation should lower");
        assert_eq!(shader.bindings.threadgroup_memory_bytes, 32 * 1024);
        assert!(
            shader
                .source
                .contains("threadgroup atomic_uint* remu_lds [[threadgroup(0)]]")
        );
        assert!(shader.source.contains("+ 0x1234u"));
        assert!(shader.source.contains("+ 0xD0u"));
        assert!(shader.source.contains("+ 0x48u"));
        assert!(shader.source.contains("+ 0x3400u"));
        assert!(shader.source.contains("+ 0x1200u"));
        assert!(shader.source.contains("atomic_fetch_add_explicit"));
        assert!(shader.source.contains("atomic_fetch_sub_explicit"));
        assert!(shader.source.contains("atomic_fetch_min_explicit"));
        assert!(shader.source.contains("atomic_fetch_max_explicit"));
        assert!(shader.source.contains("atomic_fetch_and_explicit"));
        assert!(shader.source.contains("atomic_fetch_or_explicit"));
        assert!(shader.source.contains("atomic_fetch_xor_explicit"));
        assert!(shader.source.contains("atomic_exchange_explicit"));
        assert!(shader.source.contains("remu_ds_inc"));
        assert!(
            shader
                .source
                .contains("observed >= ceiling ? 0u : observed + 1u")
        );
        assert!(
            shader
                .source
                .contains("(observed == 0u || observed > ceiling) ? ceiling : observed - 1u")
        );
        assert!(shader.source.contains("remu_ds_cmpst"));
        assert!(shader.source.contains("if (exec &&"));
        assert!(shader.source.contains("< 8192u"));
        assert!(shader.source.contains("<= (remu_s_124 >> 2)"));
        assert!(
            shader
                .source
                .contains("threadgroup_barrier(mem_flags::mem_threadgroup)")
        );
    }

    #[test]
    fn ds_rejects_unrepresentable_resources_instead_of_wrapping_or_privatizing_memory() {
        let memory_program = Program::new(
            0x1000,
            vec![decoded_ds_program(0, false).instructions[0].clone()],
        );
        let graphics_error = compile_msl(
            &memory_program,
            &empty_evaluation(),
            &MetalCompileOptions::vertex(0),
        )
        .expect_err("graphics LDS must not become lane-private scratch");
        assert!(graphics_error.to_string().contains("workgroup-shared"));

        let missing_size = compile_msl(
            &memory_program,
            &empty_evaluation(),
            &MetalCompileOptions::compute(0),
        )
        .expect_err("compute LDS requires its architectural allocation size");
        assert!(missing_size.to_string().contains("non-zero dword-aligned"));

        let oversized = compile_msl(
            &memory_program,
            &empty_evaluation(),
            &MetalCompileOptions::compute(64 * 1024),
        )
        .expect_err("64 KiB guest LDS cannot silently alias 32 KiB Metal memory");
        assert!(oversized.to_string().contains("32768-byte"));

        let gds_program = decoded_ds_program(0, true);
        let gds_error = compile_msl(
            &gds_program,
            &empty_evaluation(),
            &MetalCompileOptions::compute(32 * 1024),
        )
        .expect_err("GDS needs a system-wide resource");
        assert!(gds_error.to_string().contains("system-wide"));
    }

    #[test]
    fn lowers_all_65536_ds_swizzle_controls_and_all_four_isa_modes() {
        let seed = ds_swizzle_instruction(0);
        let program = Program::new(0x1000, vec![seed]);
        let evaluation = empty_evaluation();
        let options = MetalCompileOptions::vertex(0);
        let mut compiler = Compiler::new(&program, &evaluation, &options).expect("compiler");
        for offset in 0u16..=u16::MAX {
            compiler.body.clear();
            compiler.next_temp = 0;
            compiler
                .emit_data_share(&ds_swizzle_instruction(offset))
                .unwrap_or_else(|error| panic!("swizzle 0x{offset:04X} did not lower: {error}"));
            assert!(compiler.body.contains("simd_shuffle(v[2]"));
            assert!(compiler.body.contains("simd_shuffle(exec ? 1u : 0u"));
            assert!(compiler.body.contains("if (exec) v[8]"));
        }

        let basic =
            compile_vector_instruction(ds_swizzle_instruction(0x041f), MetalShaderStage::Vertex);
        assert!(basic.source.contains("remu_lane & 31u"));
        let quad =
            compile_vector_instruction(ds_swizzle_instruction(0x80f6), MetalShaderStage::Vertex);
        assert!(quad.source.contains("0x80F6u >> ((remu_lane & 3u) * 2u)"));
        let rotate =
            compile_vector_instruction(ds_swizzle_instruction(0xc020), MetalShaderStage::Vertex);
        assert!(rotate.source.contains("remu_lane + 1u"));
        let fft =
            compile_vector_instruction(ds_swizzle_instruction(0xe000), MetalShaderStage::Vertex);
        assert!(fft.source.contains("reverse_bits(remu_lane & 31u)"));
        assert!(!fft.source.contains("threadgroup atomic_uint* remu_lds"));
    }

    #[test]
    fn lowers_common_gen5_float_arithmetic_with_destination_accumulation() {
        let instruction = |pc, encoding, name, sources: Vec<Operand>, destination| Instruction {
            pc,
            encoding,
            opcode: Opcode::Known(name),
            words: vec![0],
            sources,
            destinations: vec![Operand::vector(destination)],
            control: None,
        };
        let program = Program::new(
            0x1000,
            vec![
                instruction(
                    0,
                    ShaderEncoding::Vop2,
                    "VMulF32",
                    vec![Operand::vector(1), Operand::vector(2)],
                    0,
                ),
                instruction(
                    4,
                    ShaderEncoding::Vop2,
                    "VMacF32",
                    vec![Operand::vector(3), Operand::vector(4)],
                    0,
                ),
                instruction(
                    8,
                    ShaderEncoding::Vop1,
                    "VRcpF32",
                    vec![Operand::vector(0)],
                    5,
                ),
            ],
        );
        let shader = compile_msl(
            &program,
            &empty_evaluation(),
            &MetalCompileOptions::vertex(0),
        )
        .expect("common float ALU should lower");
        assert!(
            shader
                .source
                .contains("as_type<float>(remu_v_1) * as_type<float>(remu_v_2)")
        );
        assert!(shader.source.contains(
            "fma(as_type<float>(remu_v_3), as_type<float>(remu_v_4), as_type<float>(remu_v_0))"
        ));
        assert!(shader.source.contains("1.0f / as_type<float>(remu_v_0)"));
    }

    #[test]
    fn lowers_sdwa_source_and_destination_controls() {
        let program = Program::new(
            0x1000,
            vec![Instruction {
                pc: 0,
                encoding: ShaderEncoding::Vop2,
                opcode: Opcode::Known("VMulF32"),
                words: vec![0, 0],
                sources: vec![Operand::vector(1), Operand::vector(2)],
                destinations: vec![Operand::vector(3)],
                control: Some(InstructionControl::Sdwa(SdwaControl {
                    destination_select: 1,
                    destination_unused: 2,
                    source0_select: 4,
                    source1_select: 6,
                    source0_sign_extend: false,
                    source1_sign_extend: false,
                    absolute_mask: 1,
                    negate_mask: 0,
                    output_modifier: 2,
                    clamp: true,
                    scalar_destination: None,
                })),
            }],
        );
        let shader = compile_msl(
            &program,
            &empty_evaluation(),
            &MetalCompileOptions::vertex(0),
        )
        .expect("SDWA float arithmetic should lower");
        assert!(shader.source.contains("(remu_v_1) & 0xFFFFu"));
        assert!(shader.source.contains("fabs(as_type<float>"));
        assert!(shader.source.contains("* 4.0f"));
        assert!(shader.source.contains("clamp("));
        assert!(shader.source.contains("& 0xFFFF00FFu"));
        assert!(shader.source.contains("<< 8"));
    }

    #[test]
    fn lowers_every_decoded_vector_compare_opcode() {
        const FLOAT_RELATIONS: [&str; 16] = [
            "F", "Lt", "Eq", "Le", "Gt", "Lg", "Ge", "O", "U", "Nge", "Nlg", "Ngt", "Nle", "Neq",
            "Nlt", "Tru",
        ];
        const INTEGER_RELATIONS: [&str; 8] = ["F", "Lt", "Eq", "Le", "Gt", "Ne", "Ge", "T"];
        let mut names = Vec::new();
        for prefix in ["VCmp", "VCmpx"] {
            for relation in FLOAT_RELATIONS {
                names.push(format!("{prefix}{relation}F32"));
            }
            for suffix in ["I32", "U32"] {
                for relation in INTEGER_RELATIONS {
                    names.push(format!("{prefix}{relation}{suffix}"));
                }
            }
        }
        names.push("VCmpClassF32".to_owned());
        assert_eq!(names.len(), 65);

        for name in names {
            let name: &'static str = Box::leak(name.into_boxed_str());
            let program = Program::new(
                0x1000,
                vec![Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Vopc,
                    opcode: Opcode::Known(name),
                    words: vec![0],
                    sources: vec![Operand::vector(1), Operand::vector(2)],
                    destinations: Vec::new(),
                    control: None,
                }],
            );
            compile_msl(
                &program,
                &empty_evaluation(),
                &MetalCompileOptions::vertex(0),
            )
            .unwrap_or_else(|error| panic!("{name} did not lower: {error}"));
        }
    }

    #[test]
    fn vector_compares_preserve_float_integer_and_mask_semantics() {
        let compare = |pc, name, control| Instruction {
            pc,
            encoding: ShaderEncoding::Vopc,
            opcode: Opcode::Known(name),
            words: vec![0],
            sources: vec![Operand::vector(1), Operand::vector(2)],
            destinations: Vec::new(),
            control,
        };
        let program = Program::new(
            0x1000,
            vec![
                compare(0, "VCmpLtF32", None),
                compare(4, "VCmpNgeF32", None),
                compare(8, "VCmpLtI32", None),
                compare(12, "VCmpLtU32", None),
                compare(16, "VCmpxEqF32", None),
                compare(
                    20,
                    "VCmpEqU32",
                    Some(InstructionControl::Sdwa(SdwaControl {
                        destination_select: 6,
                        destination_unused: 0,
                        source0_select: 6,
                        source1_select: 6,
                        source0_sign_extend: false,
                        source1_sign_extend: false,
                        absolute_mask: 0,
                        negate_mask: 0,
                        output_modifier: 0,
                        clamp: false,
                        scalar_destination: Some(12),
                    })),
                ),
            ],
        );
        let shader = compile_msl(
            &program,
            &empty_evaluation(),
            &MetalCompileOptions::vertex(0),
        )
        .expect("compare family should lower");
        assert!(
            shader
                .source
                .contains("as_type<float>(remu_v_1) < as_type<float>(remu_v_2)")
        );
        assert!(
            shader
                .source
                .contains("!(as_type<float>(remu_v_1) >= as_type<float>(remu_v_2))")
        );
        assert!(
            shader
                .source
                .contains("as_type<int>(remu_v_1) < as_type<int>(remu_v_2)")
        );
        assert!(shader.source.contains("((remu_v_1) < (remu_v_2))"));
        assert!(shader.source.contains("exec = t"));
        assert!(
            shader
                .source
                .contains("remu_s_126 = (uint)(uint64_t)simd_ballot(t")
        );
        assert!(
            shader
                .source
                .contains("remu_s_12 = (uint)(uint64_t)simd_ballot(t")
        );
        assert!(shader.source.contains("remu_s_13 = 0u;"));
    }

    #[test]
    fn compute_vector_compares_use_real_simd_lane_ballots() {
        let program = Program::new(
            0x1000,
            vec![Instruction {
                pc: 0,
                encoding: ShaderEncoding::Vopc,
                opcode: Opcode::Known("VCmpEqU32"),
                words: vec![0],
                sources: vec![Operand::vector(1), Operand::vector(2)],
                destinations: Vec::new(),
                control: None,
            }],
        );
        let shader = compile_msl(
            &program,
            &empty_evaluation(),
            &MetalCompileOptions {
                stage: MetalShaderStage::Compute,
                required_vertex_output_count: 0,
                pixel_outputs: Vec::new(),
                pixel_input_enable: 0,
                pixel_input_address: 0,
                diagnostic_force_visible_clip_depth: false,
                diagnostic_force_vertex_pattern: false,
                diagnostic_force_vertex_fetch_id: false,
                diagnostic_capture_vertex_position: false,
                diagnostic_capture_pixel_output: false,
                diagnostic_force_pixel_color: false,
                lds_size_bytes: 0,
                wave_lane_count: 32,
            },
        )
        .expect("compute compare should lower");
        assert!(
            shader
                .source
                .contains("uint remu_lane [[thread_index_in_simdgroup]]")
        );
        assert!(shader.source.contains("(uint)(uint64_t)simd_ballot("));
        assert!(!shader.source.contains("const uint remu_lane = 0u"));
    }

    #[test]
    fn accepts_exactly_the_architectural_dpp16_control_space() {
        for control in 0..=0x1ff {
            let instruction = vector_instruction(
                "VMovB32",
                1,
                InstructionControl::Dpp(DppControl {
                    control,
                    fetch_inactive: false,
                    bound_control: false,
                    absolute_mask: 0,
                    negate_mask: 0,
                    bank_mask: 0xf,
                    row_mask: 0xf,
                }),
            );
            let result = compile_msl(
                &Program::new(0x1000, vec![instruction]),
                &empty_evaluation(),
                &MetalCompileOptions::vertex(0),
            );
            assert_eq!(
                result.is_ok(),
                is_supported_dpp_control(control),
                "DPP control 0x{control:03X} acceptance differs from the ISA table: {result:?}"
            );
        }
    }

    #[test]
    fn lowers_dpp_and_dpp8_for_every_otherwise_supported_vector_alu() {
        const ALU: [(&str, usize); 56] = [
            ("VCndmaskB32", 2),
            ("VMovB32", 1),
            ("VAddF32", 2),
            ("VSubF32", 2),
            ("VSubrevF32", 2),
            ("VMulF32", 2),
            ("VMinF32", 2),
            ("VMaxF32", 2),
            ("VMin3F32", 3),
            ("VMax3F32", 3),
            ("VMed3F32", 3),
            ("VMin3I32", 3),
            ("VMax3I32", 3),
            ("VMed3I32", 3),
            ("VMin3U32", 3),
            ("VMax3U32", 3),
            ("VMed3U32", 3),
            ("VFmacF32", 2),
            ("VMacF32", 2),
            ("VFloorF32", 1),
            ("VCeilF32", 1),
            ("VTruncF32", 1),
            ("VRndneF32", 1),
            ("VFractF32", 1),
            ("VSqrtF32", 1),
            ("VRsqF32", 1),
            ("VRcpF32", 1),
            ("VRcpIflagF32", 1),
            ("VLogF32", 1),
            ("VExpF32", 1),
            ("VSinF32", 1),
            ("VCosF32", 1),
            ("VAndB32", 2),
            ("VAddI32", 2),
            ("VLshrrevB32", 2),
            ("VLshlrevB32", 2),
            ("VLshlAddU32", 3),
            ("VCvtF32I32", 1),
            ("VCvtF32U32", 1),
            ("VMinU32", 2),
            ("VMaxU32", 2),
            ("VMulLoU32", 2),
            ("VCvtU32F32", 1),
            ("VCvtI32F32", 1),
            ("VCvtOffF32I4", 1),
            ("VCubeidF32", 3),
            ("VCubescF32", 3),
            ("VCubetcF32", 3),
            ("VCubemaF32", 3),
            ("VCvtPkrtzF16F32", 2),
            ("VMadF32", 3),
            ("VFmaF32", 3),
            ("VMadAkF32", 3),
            ("VMadMkF32", 3),
            ("VFmaAkF32", 3),
            ("VFmaMkF32", 3),
        ];
        const DPP_FAMILIES: [u32; 8] = [0xe4, 0x101, 0x111, 0x121, 0x140, 0x141, 0x150, 0x160];
        for (name, source_count) in ALU {
            for control in DPP_FAMILIES {
                compile_vector_instruction(
                    vector_instruction(
                        name,
                        source_count,
                        InstructionControl::Dpp(DppControl {
                            control,
                            fetch_inactive: false,
                            bound_control: false,
                            absolute_mask: 0,
                            negate_mask: 0,
                            bank_mask: 0xf,
                            row_mask: 0xf,
                        }),
                    ),
                    MetalShaderStage::Compute,
                );
            }
            for fetch_inactive in [false, true] {
                compile_vector_instruction(
                    vector_instruction(
                        name,
                        source_count,
                        InstructionControl::Dpp8(Dpp8Control {
                            lane_selectors: 0x00fa_c688,
                            fetch_inactive,
                        }),
                    ),
                    MetalShaderStage::Compute,
                );
            }
        }
    }

    #[test]
    fn lowers_dpp_and_dpp8_for_all_decoded_vector_compares() {
        const FLOAT_RELATIONS: [&str; 16] = [
            "F", "Lt", "Eq", "Le", "Gt", "Lg", "Ge", "O", "U", "Nge", "Nlg", "Ngt", "Nle", "Neq",
            "Nlt", "Tru",
        ];
        const INTEGER_RELATIONS: [&str; 8] = ["F", "Lt", "Eq", "Le", "Gt", "Ne", "Ge", "T"];
        let mut names = Vec::new();
        for prefix in ["VCmp", "VCmpx"] {
            for relation in FLOAT_RELATIONS {
                names.push(format!("{prefix}{relation}F32"));
            }
            for suffix in ["I32", "U32"] {
                for relation in INTEGER_RELATIONS {
                    names.push(format!("{prefix}{relation}{suffix}"));
                }
            }
        }
        names.push("VCmpClassF32".to_owned());
        for name in names {
            let name: &'static str = Box::leak(name.into_boxed_str());
            for control in [
                InstructionControl::Dpp(DppControl {
                    control: 0x111,
                    fetch_inactive: false,
                    bound_control: false,
                    absolute_mask: 0,
                    negate_mask: 0,
                    bank_mask: 0xf,
                    row_mask: 0xf,
                }),
                InstructionControl::Dpp8(Dpp8Control {
                    lane_selectors: 0x0005_3977,
                    fetch_inactive: false,
                }),
            ] {
                let mut instruction = vector_instruction(name, 2, control);
                instruction.encoding = ShaderEncoding::Vopc;
                instruction.destinations.clear();
                compile_vector_instruction(instruction, MetalShaderStage::Compute);
            }
        }
    }

    #[test]
    fn lowers_every_vop3p_opcode_for_every_modifier_bit_pattern() {
        const OPCODES: [&str; 8] = [
            "VPkFmaF16",
            "VPkAddF16",
            "VPkMulF16",
            "VPkMinF16",
            "VPkMaxF16",
            "VFmaMixF32",
            "VFmaMixloF16",
            "VFmaMixhiF16",
        ];
        let seed = vop3p_instruction(
            OPCODES[0],
            Vop3pControl {
                operand_select_mask: 0,
                operand_select_high_mask: 0,
                negate_low_mask: 0,
                negate_high_mask: 0,
                clamp: false,
            },
        );
        let program = Program::new(0x1000, vec![seed]);
        let evaluation = empty_evaluation();
        let options = MetalCompileOptions::vertex(0);
        let mut compiler = Compiler::new(&program, &evaluation, &options).expect("compiler");

        for bits in 0u32..(1 << 13) {
            let control = Vop3pControl {
                operand_select_mask: bits & 7,
                operand_select_high_mask: (bits >> 3) & 7,
                negate_low_mask: (bits >> 6) & 7,
                negate_high_mask: (bits >> 9) & 7,
                clamp: bits >> 12 != 0,
            };
            for name in OPCODES {
                compiler.body.clear();
                compiler.next_temp = 0;
                let instruction = vop3p_instruction(name, control);
                compiler
                    .emit_vector_alu(&instruction)
                    .unwrap_or_else(|error| {
                        panic!("{name} modifiers 0x{bits:04X} did not lower: {error}")
                    });
                assert!(compiler.body.contains("if (exec) v[8]"));
            }
        }
    }

    #[test]
    fn vop3p_preserves_exact_packed_half_and_mix_destination_semantics() {
        let packed = compile_vector_instruction(
            vop3p_instruction(
                "VPkFmaF16",
                Vop3pControl {
                    operand_select_mask: 0b001,
                    operand_select_high_mask: 0b010,
                    negate_low_mask: 0b100,
                    negate_high_mask: 0b010,
                    clamp: true,
                },
            ),
            MetalShaderStage::Vertex,
        );
        assert!(packed.source.contains("remu_f16_to_f32_bits"));
        assert!(packed.source.contains("remu_f32_to_f16_bits"));
        assert!(packed.source.contains("remu_f16_fma_rto_bits"));
        assert!(
            packed
                .source
                .contains("#pragma clang fp reassociate(off) contract(off)")
        );
        assert!(packed.source.contains("remu_clamp_unit_f32_bits"));
        assert!(packed.source.contains(">> 16) & 0xFFFFu"));
        assert!(packed.source.contains("^ 0x8000u"));
        assert!(packed.source.contains("| (t"));
        assert!(packed.source.contains("<< 16)"));

        let mut inline = vop3p_instruction(
            "VPkAddF16",
            Vop3pControl {
                operand_select_mask: 0,
                operand_select_high_mask: 0,
                negate_low_mask: 0,
                negate_high_mask: 0,
                clamp: false,
            },
        );
        inline.sources[0] = Operand {
            kind: OperandKind::EncodedConstant,
            value: 242,
        };
        let inline = compile_vector_instruction(inline, MetalShaderStage::Vertex);
        assert!(inline.source.contains("uint t0 = 0x3C00u"));
        assert!(!inline.source.contains("uint t0 = 0x3F800000u"));

        let mix_control = Vop3pControl {
            operand_select_mask: 0b001,
            operand_select_high_mask: 0b101,
            negate_low_mask: 0b010,
            negate_high_mask: 0b001,
            clamp: true,
        };
        let mut mix = vop3p_instruction("VFmaMixF32", mix_control);
        mix.sources[2] = Operand {
            kind: OperandKind::EncodedConstant,
            value: 242,
        };
        let mix = compile_vector_instruction(mix, MetalShaderStage::Vertex);
        assert!(
            mix.source
                .contains("remu_f16_to_f32_bits(((remu_v_1 >> 16) & 0xFFFFu))")
        );
        assert!(mix.source.contains("as_type<float>(0x3F800000u)"));
        assert!(mix.source.contains("& 0x7FFFFFFFu"));
        assert!(mix.source.contains("^ 0x80000000u"));
        assert!(mix.source.contains("metal::precise::fma"));

        let low = compile_vector_instruction(
            vop3p_instruction("VFmaMixloF16", mix_control),
            MetalShaderStage::Vertex,
        );
        assert!(low.source.contains("(remu_v_8 & 0xFFFF0000u) |"));
        let high = compile_vector_instruction(
            vop3p_instruction("VFmaMixhiF16", mix_control),
            MetalShaderStage::Vertex,
        );
        assert!(high.source.contains("(remu_v_8 & 0x0000FFFFu) |"));
        assert!(high.source.contains("<< 16)"));
    }

    #[test]
    fn packed_f16_fma_round_to_odd_avoids_the_pinned_double_rounding_case() {
        fn f16_to_f32(bits: u16) -> f32 {
            let bits = u32::from(bits);
            let sign = (bits & 0x8000) << 16;
            let exponent = (bits >> 10) & 0x1f;
            let mantissa = bits & 0x3ff;
            let normal = ((exponent + 112) << 23) | (mantissa << 13);
            let infinity_nan = 0x7f80_0000 | (mantissa << 13);
            let high_bit = 31 - (mantissa | 1).leading_zeros();
            let shift = 23 - high_bit;
            let subnormal = if mantissa == 0 {
                0
            } else {
                ((high_bit + 103) << 23) | ((mantissa << shift) & 0x7f_ffff)
            };
            f32::from_bits(
                sign | if exponent == 0 {
                    subnormal
                } else if exponent == 31 {
                    infinity_nan
                } else {
                    normal
                },
            )
        }

        fn f32_to_f16(bits: u32) -> u16 {
            let sign = (bits >> 16) & 0x8000;
            let absolute = bits & 0x7fff_ffff;
            if absolute >= 0x7f80_0000 {
                return u16::try_from(sign | 0x7c00 | (u32::from(absolute > 0x7f80_0000) * 0x200))
                    .expect("half bits");
            }
            let exponent = absolute >> 23;
            let significand = (absolute & 0x7f_ffff) | 0x80_0000;
            let round_bit = (significand >> 13) & 1;
            let rounded = (significand + 0xfff + round_bit) >> 13;
            let normal_bits = (exponent.wrapping_sub(112) << 10) + (rounded - 0x400);
            let normal = if exponent >= 113 {
                normal_bits.min(0x7c00)
            } else {
                0
            };
            let distance = 126u32.wrapping_sub(exponent);
            let shift = distance.clamp(1, 25);
            let low_bits = significand & ((1 << shift) - 1);
            let halfway = 1 << (shift - 1);
            let quotient = significand >> shift;
            let round_up = low_bits > halfway || (low_bits == halfway && quotient & 1 != 0);
            let subnormal = quotient + u32::from(round_up);
            u16::try_from(sign | if exponent <= 112 { subnormal } else { normal })
                .expect("half bits")
        }

        fn fma_round_to_odd(left: f32, right: f32, addend: f32) -> u32 {
            let product = std::hint::black_box(left * right);
            let sum = std::hint::black_box(product + addend);
            let product_part = std::hint::black_box(sum - addend);
            let addend_part = std::hint::black_box(sum - product_part);
            let product_error = std::hint::black_box(product - product_part);
            let addend_error = std::hint::black_box(addend - addend_part);
            let residual = std::hint::black_box(product_error + addend_error);
            let sum_bits = sum.to_bits();
            let residual_bits = residual.to_bits();
            let residual_magnitude = residual_bits & 0x7fff_ffff;
            let adjust =
                residual_magnitude != 0 && residual_magnitude < 0x7f80_0000 && sum_bits & 1 == 0;
            if !adjust {
                return sum_bits;
            }
            if (sum_bits ^ residual_bits) & 0x8000_0000 != 0 {
                sum_bits - 1
            } else {
                sum_bits + 1
            }
        }

        let left = f16_to_f32(0x4100); // 2.5
        let right = f16_to_f32(0x7522); // 21024
        let addend = f16_to_f32(0x04ea); // approximately 7.496e-5
        let double_rounded = f32_to_f16(std::hint::black_box(left * right + addend).to_bits());
        let single_rounded = f32_to_f16(fma_round_to_odd(left, right, addend));
        assert_eq!(double_rounded, 0x7a6a);
        assert_eq!(single_rounded, 0x7a6b);
        assert_eq!(f32_to_f16(fma_round_to_odd(left, right, -addend)), 0x7a6a);
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[allow(clippy::too_many_lines)]
    fn generated_vop3p_msl_compiles_with_apple_metal() {
        use std::path::PathBuf;
        use std::process::Command;
        use std::sync::atomic::{AtomicUsize, Ordering};

        static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

        fn find_metal_compiler() -> Option<PathBuf> {
            if let Some(path) = std::env::var_os("REMU_METAL_COMPILER").map(PathBuf::from)
                && path.is_file()
            {
                return Some(path);
            }
            let bundled = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../.remu/toolchains/Metal.xctoolchain/usr/bin/metal");
            if bundled.is_file() {
                return Some(bundled);
            }
            let output = Command::new("xcrun")
                .args(["--find", "metal"])
                .output()
                .ok()?;
            output
                .status
                .success()
                .then(|| PathBuf::from(String::from_utf8_lossy(&output.stdout).trim().to_owned()))
        }

        let Some(compiler) = find_metal_compiler() else {
            eprintln!("skipping Apple Metal compile test: no metal compiler found");
            return;
        };
        let control = Vop3pControl {
            operand_select_mask: 0b101,
            operand_select_high_mask: 0b011,
            negate_low_mask: 0b110,
            negate_high_mask: 0b101,
            clamp: true,
        };
        let mut instructions = [
            "VPkFmaF16",
            "VPkAddF16",
            "VPkMulF16",
            "VPkMinF16",
            "VPkMaxF16",
            "VFmaMixF32",
            "VFmaMixloF16",
            "VFmaMixhiF16",
        ]
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            let mut instruction = vop3p_instruction(name, control);
            instruction.pc = u32::try_from(index * 8).expect("small VOP3P test program");
            instruction.destinations = vec![Operand::vector(
                8 + u32::try_from(index).expect("small destination index"),
            )];
            instruction
        })
        .collect::<Vec<_>>();
        instructions[1].sources[0] = Operand {
            kind: OperandKind::EncodedConstant,
            value: 242,
        };
        let program = Program::new(0x1000, instructions);
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "remu-vop3p-metal-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).expect("create Metal test directory");

        for stage in [
            MetalShaderStage::Vertex,
            MetalShaderStage::Pixel,
            MetalShaderStage::Compute,
        ] {
            let options = match stage {
                MetalShaderStage::Vertex => MetalCompileOptions::vertex(0),
                MetalShaderStage::Pixel => MetalCompileOptions::pixel(
                    vec![PixelOutputBinding {
                        guest_slot: 0,
                        host_location: 0,
                        kind: PixelOutputKind::Float,
                    }],
                    0,
                    0,
                ),
                MetalShaderStage::Compute => MetalCompileOptions {
                    stage,
                    required_vertex_output_count: 0,
                    pixel_outputs: Vec::new(),
                    pixel_input_enable: 0,
                    pixel_input_address: 0,
                    diagnostic_force_visible_clip_depth: false,
                    diagnostic_force_vertex_pattern: false,
                    diagnostic_force_vertex_fetch_id: false,
                    diagnostic_capture_vertex_position: false,
                    diagnostic_capture_pixel_output: false,
                    diagnostic_force_pixel_color: false,
                    lds_size_bytes: 0,
                    wave_lane_count: 32,
                },
            };
            let shader = compile_msl(&program, &empty_evaluation(), &options)
                .expect("VOP3P program should lower");
            let stem = match stage {
                MetalShaderStage::Vertex => "vertex",
                MetalShaderStage::Pixel => "pixel",
                MetalShaderStage::Compute => "compute",
            };
            let source = directory.join(format!("{stem}.metal"));
            let air = directory.join(format!("{stem}.air"));
            std::fs::write(&source, shader.source).expect("write generated MSL");
            let output = Command::new(&compiler)
                .args(["-c", "-target", "air64-apple-macosx15.0"])
                .arg("-fmetal-math-mode=safe")
                .arg("-o")
                .arg(&air)
                .arg(&source)
                .output()
                .expect("run Apple Metal compiler");
            assert!(
                output.status.success(),
                "{stage:?} VOP3P MSL failed to compile:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::fs::remove_dir_all(directory).expect("remove Metal test directory");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn generated_ds_msl_compiles_with_apple_metal_in_every_valid_stage() {
        use std::path::PathBuf;
        use std::process::Command;
        use std::sync::atomic::{AtomicUsize, Ordering};

        static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

        fn find_metal_compiler() -> Option<PathBuf> {
            if let Some(path) = std::env::var_os("REMU_METAL_COMPILER").map(PathBuf::from)
                && path.is_file()
            {
                return Some(path);
            }
            let bundled = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../.remu/toolchains/Metal.xctoolchain/usr/bin/metal");
            if bundled.is_file() {
                return Some(bundled);
            }
            let output = Command::new("xcrun")
                .args(["--find", "metal"])
                .output()
                .ok()?;
            output
                .status
                .success()
                .then(|| PathBuf::from(String::from_utf8_lossy(&output.stdout).trim().to_owned()))
        }

        let Some(compiler) = find_metal_compiler() else {
            eprintln!("skipping Apple Metal DS compile test: no metal compiler found");
            return;
        };
        let mut swizzles = [0x041f, 0x80f6, 0xc020, 0xe000]
            .into_iter()
            .enumerate()
            .map(|(index, offset)| {
                let mut instruction = ds_swizzle_instruction(offset);
                instruction.pc = u32::try_from(index * 8).expect("small swizzle program");
                instruction
            })
            .collect::<Vec<_>>();
        swizzles.push(Instruction {
            pc: 32,
            encoding: ShaderEncoding::Sopp,
            opcode: Opcode::END_PROGRAM,
            words: vec![0xbf81_0000],
            sources: Vec::new(),
            destinations: Vec::new(),
            control: None,
        });
        let swizzle_program = Program::new(0x1000, swizzles);
        let programs = [
            (
                "compute",
                decoded_ds_program(0x1234, false),
                MetalCompileOptions::compute(32 * 1024),
            ),
            (
                "vertex",
                swizzle_program.clone(),
                MetalCompileOptions::vertex(0),
            ),
            (
                "pixel",
                swizzle_program,
                MetalCompileOptions::pixel(
                    vec![PixelOutputBinding {
                        guest_slot: 0,
                        host_location: 0,
                        kind: PixelOutputKind::Float,
                    }],
                    0,
                    0,
                ),
            ),
        ];
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let directory =
            std::env::temp_dir().join(format!("remu-ds-metal-{}-{sequence}", std::process::id()));
        std::fs::create_dir_all(&directory).expect("create Metal DS test directory");

        for (stem, program, options) in programs {
            let shader = compile_msl(&program, &empty_evaluation(), &options)
                .expect("valid DS program should lower");
            let source = directory.join(format!("{stem}.metal"));
            let air = directory.join(format!("{stem}.air"));
            std::fs::write(&source, shader.source).expect("write generated DS MSL");
            let output = Command::new(&compiler)
                .args(["-c", "-target", "air64-apple-macosx15.0"])
                .arg("-fmetal-math-mode=safe")
                .arg("-o")
                .arg(&air)
                .arg(&source)
                .output()
                .expect("run Apple Metal compiler");
            assert!(
                output.status.success(),
                "{stem} DS MSL failed to compile:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::fs::remove_dir_all(directory).expect("remove Metal DS test directory");
    }

    #[test]
    fn dpp_semantics_cover_lane_selection_activity_modifiers_and_write_masks() {
        let shader = compile_vector_instruction(
            vector_instruction(
                "VAddF32",
                2,
                InstructionControl::Dpp(DppControl {
                    control: 0x111,
                    fetch_inactive: false,
                    bound_control: false,
                    absolute_mask: 1,
                    negate_mask: 2,
                    bank_mask: 0xa,
                    row_mask: 5,
                }),
            ),
            MetalShaderStage::Compute,
        );
        assert!(shader.source.contains("simd_shuffle(remu_v_1, (ushort)"));
        assert!(
            shader
                .source
                .contains("simd_shuffle(exec ? 1u : 0u, (ushort)")
        );
        assert!(shader.source.contains("(remu_lane & 15u) >= 1u"));
        assert!(shader.source.contains("fabs(as_type<float>"));
        assert!(shader.source.contains("-(as_type<float>(remu_v_2))"));
        assert!(shader.source.contains("5u >> (remu_lane >> 4)"));
        assert!(shader.source.contains("10u >> (remu_lane & 3u)"));
        assert!(shader.source.contains("? t"));
        assert!(shader.source.contains(": remu_v_8"));

        let fetch_inactive = compile_vector_instruction(
            vector_instruction(
                "VMovB32",
                1,
                InstructionControl::Dpp(DppControl {
                    control: 0x101,
                    fetch_inactive: true,
                    bound_control: true,
                    absolute_mask: 0,
                    negate_mask: 0,
                    bank_mask: 0xf,
                    row_mask: 0xf,
                }),
            ),
            MetalShaderStage::Compute,
        );
        assert!(
            !fetch_inactive
                .source
                .contains("simd_shuffle(exec ? 1u : 0u")
        );
        assert!(fetch_inactive.source.contains("&& (true)"));
    }

    #[test]
    fn dpp8_and_compare_masks_preserve_architectural_lane_state() {
        let dpp8 = compile_vector_instruction(
            vector_instruction(
                "VMovB32",
                1,
                InstructionControl::Dpp8(Dpp8Control {
                    lane_selectors: 0x0005_3977,
                    fetch_inactive: false,
                }),
            ),
            MetalShaderStage::Compute,
        );
        assert!(dpp8.source.contains("342391u >> ((remu_lane & 7u) * 3u)"));
        assert!(dpp8.source.contains("remu_lane & 0xFFFFFFF8u"));
        assert!(dpp8.source.contains("simd_shuffle(remu_v_1, (ushort)"));
        assert!(
            dpp8.source
                .contains("simd_shuffle(exec ? 1u : 0u, (ushort)")
        );

        let mut compare = vector_instruction(
            "VCmpEqF32",
            2,
            InstructionControl::Dpp(DppControl {
                control: 0xe4,
                fetch_inactive: false,
                bound_control: false,
                absolute_mask: 0,
                negate_mask: 0,
                bank_mask: 0,
                row_mask: 0,
            }),
        );
        compare.encoding = ShaderEncoding::Vopc;
        compare.destinations.clear();
        let compare = compile_vector_instruction(compare, MetalShaderStage::Vertex);
        assert!(compare.source.contains(": vcc"));
        assert!(compare.source.contains("remu_s_106 ="));
        assert!(compare.source.contains("remu_s_107 = 0u"));
    }

    #[test]
    fn graphics_dpp_uses_converged_real_wave_operations() {
        let shader = compile_vector_instruction(
            vector_instruction(
                "VMovB32",
                1,
                InstructionControl::Dpp8(Dpp8Control {
                    lane_selectors: 0x00fa_c688,
                    fetch_inactive: false,
                }),
            ),
            MetalShaderStage::Vertex,
        );
        assert!(
            shader
                .source
                .contains("uint remu_lane = simd_prefix_exclusive_sum(1u)")
        );
        assert!(shader.source.contains("simd_shuffle(remu_v_1, (ushort)"));
        assert_eq!(shader.register_footprint.scalar_values, 0);
        let shuffle = shader
            .source
            .find("simd_shuffle(remu_v_1, (ushort)")
            .expect("DPP8 shuffle");
        let write = shader
            .source
            .find("if (exec) remu_v_8")
            .expect("EXEC-guarded destination write");
        assert!(
            shuffle < write,
            "wave shuffle must be converged before EXEC guard"
        );
    }

    // Digests of the MSL that lowering emits for two fixed programs, one per control-flow mode.
    // Translated source is compiled and cached per guest shader, so any difference here changes
    // every pipeline a game builds and has to be an intended backend change.
    const DIRECT_SOURCE_DIGEST: u64 = 0xc2f6_b13e_f46d_8b8d;
    const DISPATCHER_SOURCE_DIGEST: u64 = 0x029c_2bd0_49d1_c94c;

    fn source_digest(source: &str) -> u64 {
        const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        source.bytes().fold(FNV_OFFSET_BASIS, |digest, byte| {
            (digest ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
        })
    }

    fn direct_reference_program() -> Program {
        Program::new(
            0x1000,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Sop1,
                    opcode: Opcode::Known("SMovB32"),
                    words: vec![0],
                    sources: vec![Operand::scalar(12)],
                    destinations: vec![Operand::scalar(80)],
                    control: None,
                },
                Instruction {
                    pc: 4,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![Operand::scalar(80)],
                    destinations: vec![Operand::vector(3)],
                    control: None,
                },
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VAddF32"),
                    words: vec![0],
                    sources: vec![Operand::vector(3), Operand::vector(4)],
                    destinations: vec![Operand::vector(5)],
                    control: None,
                },
                Instruction {
                    pc: 12,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VMulF32"),
                    words: vec![0],
                    sources: vec![Operand::vector(5), Operand::vector(5)],
                    destinations: vec![Operand::vector(6)],
                    control: None,
                },
                Instruction {
                    pc: 16,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VAddcU32"),
                    words: vec![0],
                    sources: vec![Operand::vector(6), Operand::vector(7)],
                    destinations: vec![Operand::vector(8)],
                    control: None,
                },
                Instruction {
                    pc: 20,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ],
        )
    }

    fn dispatcher_reference_program() -> Program {
        Program::new(
            0x2000,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::Known("SCbranchScc1"),
                    // PC 0 + 4-byte instruction + two dwords = PC 12.
                    words: vec![2],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
                Instruction {
                    pc: 4,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![Operand::vector(1)],
                    destinations: vec![Operand::vector(2)],
                    control: None,
                },
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VMulF32"),
                    words: vec![0],
                    sources: vec![Operand::vector(2), Operand::vector(2)],
                    destinations: vec![Operand::vector(3)],
                    control: None,
                },
                Instruction {
                    pc: 12,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ],
        )
    }

    fn unsupported_reference_program() -> Program {
        Program::new(
            0x1000,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Vop3,
                    opcode: Opcode::Known("VAdd3U32"),
                    words: vec![0, 0],
                    sources: vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
                    destinations: vec![Operand::vector(4)],
                    control: None,
                },
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![Operand::vector(4)],
                    destinations: vec![Operand::vector(5)],
                    control: None,
                },
                Instruction {
                    pc: 12,
                    encoding: ShaderEncoding::Vop3,
                    opcode: Opcode::Known("VBfeU32"),
                    words: vec![0, 0],
                    sources: vec![Operand::vector(5), Operand::vector(6), Operand::vector(7)],
                    destinations: vec![Operand::vector(8)],
                    control: None,
                },
                Instruction {
                    pc: 20,
                    encoding: ShaderEncoding::Vop3,
                    opcode: Opcode::Known("VAdd3U32"),
                    words: vec![0, 0],
                    sources: vec![Operand::vector(8), Operand::vector(9), Operand::vector(10)],
                    destinations: vec![Operand::vector(11)],
                    control: None,
                },
                Instruction {
                    pc: 28,
                    encoding: ShaderEncoding::Sop2,
                    opcode: Opcode::Known("SAbsdiffI32"),
                    words: vec![0],
                    sources: vec![Operand::scalar(10), Operand::scalar(11)],
                    destinations: vec![Operand::scalar(12)],
                    control: None,
                },
                Instruction {
                    pc: 32,
                    encoding: ShaderEncoding::Vop3,
                    opcode: Opcode::Raw {
                        family: RawOpcodeFamily::Vop3,
                        code: 0x2a5,
                    },
                    words: vec![0, 0],
                    sources: vec![Operand::vector(12)],
                    destinations: vec![Operand::vector(13)],
                    control: None,
                },
                Instruction {
                    pc: 40,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ],
        )
    }

    #[test]
    fn a_shader_using_several_unsupported_opcodes_reports_all_of_them() {
        let error = compile_msl(
            &unsupported_reference_program(),
            &empty_evaluation(),
            &MetalCompileOptions::compute(0),
        )
        .expect_err("unsupported opcodes should fail lowering");

        let message = error.to_string();
        assert!(
            message.starts_with("pc 0x0 VAdd3U32: unsupported vector opcode VAdd3U32;"),
            "{message}"
        );
        assert!(
            message.contains("pc 0xC VBfeU32: unsupported vector opcode VBfeU32"),
            "{message}"
        );
        assert!(
            message.contains("pc 0x1C SAbsdiffI32: unsupported scalar opcode SAbsdiffI32"),
            "{message}"
        );
        assert!(
            message.contains("pc 0x20 Vop3[0x2A5]: raw opcode has no lowering semantics"),
            "{message}"
        );
        assert!(!message.contains("pc 0x14"), "{message}");
    }

    #[test]
    fn a_shader_with_one_unsupported_opcode_reports_only_that_opcode() {
        let mut instructions = unsupported_reference_program().instructions;
        instructions.retain(|instruction| !instruction.opcode.is("VAdd3U32"));
        instructions.truncate(2);
        let error = compile_msl(
            &Program::new(0x1000, instructions),
            &empty_evaluation(),
            &MetalCompileOptions::compute(0),
        )
        .expect_err("an unsupported opcode should fail lowering");

        assert_eq!(
            error.to_string(),
            "pc 0xC VBfeU32: unsupported vector opcode VBfeU32"
        );
    }

    #[test]
    fn a_shader_that_lowers_today_still_emits_byte_identical_metal_source() {
        let direct = compile_msl(
            &direct_reference_program(),
            &empty_evaluation(),
            &MetalCompileOptions::compute(0),
        )
        .expect("direct reference program should lower");
        let dispatcher = compile_msl(
            &dispatcher_reference_program(),
            &empty_evaluation(),
            &MetalCompileOptions::compute(0),
        )
        .expect("dispatcher reference program should lower");

        assert!(direct.register_footprint.direct_control_flow);
        assert!(!dispatcher.register_footprint.direct_control_flow);
        assert_eq!(
            source_digest(&direct.source),
            DIRECT_SOURCE_DIGEST,
            "direct lowering emitted different MSL:\n{}",
            direct.source
        );
        assert_eq!(
            source_digest(&dispatcher.source),
            DISPATCHER_SOURCE_DIGEST,
            "dispatcher lowering emitted different MSL:\n{}",
            dispatcher.source
        );
    }
}
