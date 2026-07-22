// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Backend lowering from `REMu`'s Gen5 shader IR to Metal Shading Language.
//!
//! Translation is deliberately strict. An instruction is either emitted with its architectural
//! semantics or compilation fails with the instruction PC and opcode; this crate never substitutes
//! a placeholder color, drops an unknown memory access, or selects behavior by game identity.

use std::collections::BTreeSet;
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
    /// Bytes of guest LDS allocated to each compute workgroup.
    ///
    /// A non-zero, dword-aligned value is required only when the program contains an LDS memory
    /// operation. `DS_SWIZZLE_B32` does not allocate or access LDS and therefore does not require
    /// this field. Apple GPUs expose at most 32 KiB of threadgroup memory to this backend; larger
    /// guest allocations are rejected instead of being silently wrapped or spilled with different
    /// synchronization semantics.
    pub lds_size_bytes: u32,
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
            lds_size_bytes: 0,
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
            lds_size_bytes: 0,
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
            lds_size_bytes,
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
}

#[derive(Clone, Copy)]
struct ImageKind {
    component: &'static str,
    storage: bool,
    sampler_slot: Option<u32>,
}

impl<'a> Compiler<'a> {
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
                let sampler_slot = (!storage).then(|| {
                    let slot = sampler_count;
                    sampler_count += 1;
                    slot
                });
                ImageKind {
                    component: image_component_type(binding),
                    storage,
                    sampler_slot,
                }
            })
            .collect();

        Ok(Self {
            program,
            evaluation,
            options,
            body: String::new(),
            next_temp: 0,
            vertex_outputs,
            pixel_attributes,
            image_kinds,
        })
    }

    fn compile(mut self) -> Result<MetalShader, MetalCompileError> {
        for (index, instruction) in self.program.instructions.iter().enumerate() {
            self.line(2, &format!("case {}u: {{", instruction.pc));
            self.emit_instruction(instruction, index)
                .map_err(|message| MetalCompileError::Instruction {
                    pc: instruction.pc,
                    opcode: instruction.opcode,
                    message,
                })?;
            self.line(3, "break;");
            self.line(2, "}");
        }

        let source = self.emit_module();
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
        })
    }

    fn emit_instruction(&mut self, instruction: &Instruction, index: usize) -> Result<(), String> {
        let next_pc = self.program.instructions.get(index + 1).map(|next| next.pc);
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw opcode has no lowering semantics".to_owned())?;

        match name {
            "SEndpgm" => {
                self.line(3, "active = false;");
                return Ok(());
            }
            "SBranch" => {
                let target = branch_target(instruction)?;
                self.line(3, &format!("pc = {target}u;"));
                return Ok(());
            }
            "SCbranchScc0" | "SCbranchScc1" | "SCbranchVccz" | "SCbranchVccnz"
            | "SCbranchExecz" | "SCbranchExecnz" => {
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
        if let Some(next_pc) = next_pc {
            self.line(3, &format!("pc = {next_pc}u;"));
        } else {
            self.line(3, "active = false;");
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn emit_module(&self) -> String {
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
            parameters.push(format!(
                "texture2d<{}{access}> tex{index} [[texture({index})]]",
                kind.component
            ));
        }
        if sampler_count != 0 {
            parameters.push(format!(
                "constant RemuSamplers& remu_samplers [[buffer({})]]",
                initial_index + 1
            ));
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
                if self
                    .evaluation
                    .compute_system_registers
                    .is_some_and(|registers| registers.thread_group_size.is_some())
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
        source.push_str("    uint s[128] = {};\n    uint v[256] = {};\n");
        source.push_str("    bool exec = true;\n    bool vcc = false;\n    bool scc = false;\n");
        source.push_str("    uint pc = 0u;\n    bool active = true;\n    uint steps = 0u;\n");
        source.push_str("    for (uint i = 0u; i < 128u; ++i) s[i] = remu_initial_s[i];\n");
        if self.options.stage != MetalShaderStage::Compute {
            // Graphics stages cannot declare thread_index_in_simdgroup, but the exclusive prefix
            // of one is the architectural lane index. Wave operations are emitted before the
            // EXEC-guarded write, keeping every host lane converged at simdgroup intrinsics.
            source.push_str("    uint remu_lane = simd_prefix_exclusive_sum(1u);\n");
        }
        match self.options.stage {
            MetalShaderStage::Vertex => {
                source.push_str("    RemuVsOut remu_out = {};\n");
                source.push_str("    v[5] = remu_vertex_id;\n    v[8] = remu_instance_id;\n");
            }
            MetalShaderStage::Pixel => {
                source.push_str("    RemuPsOut remu_out = {};\n");
                self.emit_pixel_input_state(&mut source);
            }
            MetalShaderStage::Compute => {
                source.push_str(
                    "    v[0] = remu_local_id.x; v[1] = remu_local_id.y; v[2] = remu_local_id.z;\n",
                );
                self.emit_compute_system_registers(&mut source);
            }
        }
        source.push_str(
            "    s[106] = 0u; s[107] = 0u; s[126] = (uint)(uint64_t)simd_ballot(true); s[127] = 0u;\n",
        );
        source.push_str("    while (active) {\n        switch (pc) {\n");
        source.push_str(&self.body);
        source.push_str(
            "        default: active = false; break;\n        }\n\
             \x20       if (++steps >= 16384u) active = false;\n    }\n",
        );
        match self.options.stage {
            MetalShaderStage::Vertex | MetalShaderStage::Pixel => {
                if self.options.stage == MetalShaderStage::Pixel {
                    source.push_str("    if (!exec) discard_fragment();\n");
                }
                source.push_str("    return remu_out;\n");
            }
            MetalShaderStage::Compute => {}
        }
        source.push_str("}\n");
        source
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
            if let Some(register) = register {
                let _ = writeln!(source, "    s[{register}] = remu_group_id.{component};");
            }
        }
        if let Some(register) = registers.thread_group_size {
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
                if self.options.pixel_input_enable & mask != 0 {
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
            "VLshrrevB32" => format!(
                "({} >> ({} & 31u))",
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
        let ballot = Self::ballot(condition);
        match register {
            VCC_LO => {
                self.line(3, &format!("vcc = {condition};"));
                self.line(3, &format!("s[{VCC_LO}] = {ballot};"));
                self.line(3, &format!("s[{VCC_HI}] = 0u;"));
            }
            EXEC_LO => self.store_exec_mask(condition),
            register if register < SCALAR_REGISTER_COUNT => {
                self.line(3, &format!("s[{register}] = {ballot};"));
                if register + 1 < SCALAR_REGISTER_COUNT {
                    self.line(3, &format!("s[{}] = 0u;", register + 1));
                }
            }
            _ => {}
        }
    }

    fn store_exec_mask(&mut self, condition: &str) {
        let ballot = Self::ballot(condition);
        self.line(3, &format!("exec = {condition};"));
        self.line(3, &format!("s[{EXEC_LO}] = {ballot};"));
        self.line(3, &format!("s[{EXEC_HI}] = 0u;"));
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
            "SAndB64" => {
                let right = self.raw_source64(instruction, 1)?;
                (format!("({left} & {right})"), true)
            }
            "SOrB64" => {
                let right = self.raw_source64(instruction, 1)?;
                (format!("({left} | {right})"), true)
            }
            _ => return Err(format!("unsupported 64-bit scalar opcode {name}")),
        };
        let result = self.temp("ulong", &value);
        self.store_scalar(destination, &format!("(uint){result}"));
        self.store_scalar(destination + 1, &format!("(uint)({result} >> 32)"));
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
        let binding = &self.evaluation.global_memory_bindings[binding_index];
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
            let expression = format!(
                "remu_load_u32(b{binding_index}, {address} + {offset}u, {}u)",
                binding.data.len()
            );
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
        let binding = &self.evaluation.global_memory_bindings[binding_index];
        let binding_length = binding.data.len();
        let binding_base = binding.base_address;
        let scalar_offset = self.raw_source(instruction, 2)?;
        let vector_index = if control.index_enabled {
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
        let resource_bias = self.temp(
            "uint",
            &format!("(uint)({resource_base} - 0x{binding_base:X}ul)"),
        );
        let stride = self.temp("uint", &format!("({resource1} >> 16) & 0x3FFFu"));
        let swizzled = self.temp("bool", &format!("({resource1} & 0x80000000u) != 0u"));
        let add_tid = self.temp("bool", &format!("({resource3} & 0x00800000u) != 0u"));
        let index = self.temp(
            "uint",
            &format!("{vector_index} + ({add_tid} ? (remu_lane & 63u) : 0u)"),
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
            &format!(
                "{descriptor_null} || ({oob_select} == 0u ? ({index} >= {record_count} || {offset} >= {stride}) : {oob_select} == 1u ? ({index} >= {record_count}) : {oob_select} == 2u ? ({record_count} == 0u) : ({swizzled} && {stride} != 0u ? ({index} >= {record_count} || !remu_range_valid({offset}, {payload}, {stride})) : !remu_range_valid({buffer_offset}, {payload}, {reduced_records})))"
            ),
        );
        if is_format_buffer_memory(instruction) {
            return self.emit_format_buffer_memory(
                instruction,
                control,
                binding_index,
                binding_length,
                &address,
                &oob,
            );
        }
        self.emit_raw_memory_operation(
            name,
            control.vector_data,
            control.vector_data,
            control.dword_count,
            binding_index,
            binding_length,
            &address,
            &oob,
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
        let binding = &self.evaluation.global_memory_bindings[binding_index];
        let local_address = self.temp(
            "uint",
            &format!("(uint)({guest_address} - 0x{:X}ul)", binding.base_address),
        );
        let before = self.temp(
            "bool",
            &format!("{guest_address} < 0x{:X}ul", binding.base_address),
        );
        let oob = self.temp("bool", &before);
        self.emit_raw_memory_operation(
            name,
            control.vector_data,
            control.vector_destination,
            control.dword_count,
            binding_index,
            binding.data.len(),
            &local_address,
            &oob,
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
        binding_length: usize,
        address: &str,
        descriptor_oob: &str,
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
                let component_address = if component == 0 {
                    aligned_address.clone()
                } else {
                    format!("{aligned_address} + {}u", component * 4)
                };
                let loaded = match width {
                    1 => format!(
                        "remu_load_u8(b{binding_index}, {component_address}, {binding_length}u)"
                    ),
                    2 => format!(
                        "remu_load_u16(b{binding_index}, {component_address}, {binding_length}u)"
                    ),
                    4 => format!(
                        "remu_load_u32(b{binding_index}, {component_address}, {binding_length}u)"
                    ),
                    _ => unreachable!("raw load widths are fixed"),
                };
                let in_range = format!(
                    "!{descriptor_oob} && remu_range_valid({component_address}, {width}u, {binding_length}u)"
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
                        "if (exec && !{descriptor_oob} && remu_range_valid({component_address}, {width}u, {binding_length}u)) remu_store_u{bits}(b{binding_index}, {component_address}, {value});",
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
                "if (exec && !{descriptor_oob} && remu_range_valid({atomic_address}, 4u, {binding_length}u)) {old} = {operation};"
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
        binding_length: usize,
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
        let aligned = self.temp(
            "bool",
            &format!("{element_size} != 0u && ({address} % {element_size}) == 0u"),
        );
        let oob = self.temp(
            "bool",
            &format!(
                "{descriptor_oob} || !{aligned} || !remu_range_valid({address}, {element_size}, {binding_length}u)"
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
        if !name.starts_with("ImageSample") {
            return Err(format!("unsupported image opcode {name}"));
        }
        if kind.storage {
            return Err("sample operation resolved to a storage image".to_owned());
        }
        if image.is_array || !matches!(image.dimension, 0 | 1) {
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
            cursor += image_full_address_slots(image);
        }
        let bias = if has_bias {
            let value = self.image_float_address(image, cursor);
            cursor += 1;
            self.temp("float", &value)
        } else {
            "0.0f".to_owned()
        };
        if has_compare {
            return Err("comparison sampling requires depth-texture Metal typing".to_owned());
        }
        let mut gradient_x = String::new();
        let mut gradient_y = String::new();
        if has_gradients {
            gradient_x = self.temp(
                "float2",
                &format!(
                    "float2({}, {})",
                    self.image_float_address(image, cursor),
                    self.image_float_address(image, cursor + 1)
                ),
            );
            gradient_y = self.temp(
                "float2",
                &format!(
                    "float2({}, {})",
                    self.image_float_address(image, cursor + 2),
                    self.image_float_address(image, cursor + 3)
                ),
            );
            cursor += 4;
        }
        let mut coordinates = self.temp(
            "float2",
            &format!(
                "float2({}, {})",
                self.image_float_address(image, cursor),
                self.image_float_address(image, cursor + 1)
            ),
        );
        let lod = if has_zero_lod {
            "0.0f".to_owned()
        } else if has_lod {
            self.temp("float", &self.image_float_address(image, cursor + 2))
        } else {
            bias.clone()
        };
        if has_offset {
            let offset_lod = if (has_zero_lod || has_lod) && !has_gradients {
                lod.clone()
            } else {
                "0.0f".to_owned()
            };
            let mip = self.temp("uint", &format!("(uint)max((int)({offset_lod}), 0)"));
            coordinates = self.temp(
                "float2",
                &format!(
                    "{coordinates} + float2((float){offset_x} / (float)tex{binding_index}.get_width({mip}), (float){offset_y} / (float)tex{binding_index}.get_height({mip}))"
                ),
            );
        }
        let extra = if has_gradients {
            format!(", gradient2d({gradient_x}, {gradient_y})")
        } else if has_zero_lod || has_lod {
            format!(", level({lod})")
        } else if has_bias {
            format!(", bias({bias})")
        } else {
            String::new()
        };
        Ok(self.temp(
            &format!("vec<{}, 4>", self.image_kinds[binding_index].component),
            &format!(
                "tex{binding_index}.sample(remu_samplers.s{sampler_slot}, {coordinates}{extra})"
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
            VCC_LO => {
                let value = self.temp("uint", expression);
                self.line(3, &format!("s[{VCC_LO}] = {value};"));
                self.line(3, &format!("vcc = (({value} >> remu_lane) & 1u) != 0u;"));
            }
            EXEC_LO => {
                let value = self.temp("uint", expression);
                self.line(3, &format!("s[{EXEC_LO}] = {value};"));
                self.line(3, &format!("exec = (({value} >> remu_lane) & 1u) != 0u;"));
            }
            _ => self.line(3, &format!("s[{register}] = {expression};")),
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
        ShaderEvaluation, Vop3pControl, decode_program,
    };

    use super::*;

    fn empty_evaluation() -> ShaderEvaluation {
        ShaderEvaluation {
            initial_scalar_registers: vec![0; 128],
            scalar_registers: vec![0; 128],
            image_bindings: Vec::new(),
            global_memory_bindings: Vec::new(),
            compute_system_registers: None,
            runtime_scalar_registers: BTreeSet::new(),
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
                lds_size_bytes: 0,
            },
        )
        .expect("vector instruction should lower")
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
                instruction_pcs: pcs,
                data: vec![0; 64 * 1024],
                writable: true,
                write_back_to_guest: true,
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
    }

    #[test]
    fn initializes_complete_rdna2_compute_system_register_state() {
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
            work_group_x: Some(4),
            work_group_y: Some(5),
            work_group_z: Some(6),
            thread_group_size: Some(7),
        });

        let shader = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
            .expect("compute system register lowering");

        assert!(shader.source.contains("v[0] = remu_local_id.x"));
        assert!(shader.source.contains("s[4] = remu_group_id.x;"));
        assert!(shader.source.contains("s[5] = remu_group_id.y;"));
        assert!(shader.source.contains("s[6] = remu_group_id.z;"));
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
        assert!(shader.source.contains("<= (s[124] >> 2)"));
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
                .contains("as_type<float>(v[1]) * as_type<float>(v[2])")
        );
        assert!(
            shader
                .source
                .contains("fma(as_type<float>(v[3]), as_type<float>(v[4]), as_type<float>(v[0]))")
        );
        assert!(shader.source.contains("1.0f / as_type<float>(v[0])"));
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
        assert!(shader.source.contains("(v[1]) & 0xFFFFu"));
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
                .contains("as_type<float>(v[1]) < as_type<float>(v[2])")
        );
        assert!(
            shader
                .source
                .contains("!(as_type<float>(v[1]) >= as_type<float>(v[2]))")
        );
        assert!(
            shader
                .source
                .contains("as_type<int>(v[1]) < as_type<int>(v[2])")
        );
        assert!(shader.source.contains("((v[1]) < (v[2]))"));
        assert!(shader.source.contains("exec = t"));
        assert!(
            shader
                .source
                .contains("s[126] = (uint)(uint64_t)simd_ballot(t")
        );
        assert!(
            shader
                .source
                .contains("s[12] = (uint)(uint64_t)simd_ballot(t")
        );
        assert!(shader.source.contains("s[13] = 0u;"));
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
                lds_size_bytes: 0,
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
        const ALU: [(&str, usize); 34] = [
            ("VCndmaskB32", 2),
            ("VMovB32", 1),
            ("VAddF32", 2),
            ("VSubF32", 2),
            ("VSubrevF32", 2),
            ("VMulF32", 2),
            ("VMinF32", 2),
            ("VMaxF32", 2),
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
            ("VLshrrevB32", 2),
            ("VLshlAddU32", 3),
            ("VCvtF32I32", 1),
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
                .contains("remu_f16_to_f32_bits(((v[1] >> 16) & 0xFFFFu))")
        );
        assert!(mix.source.contains("as_type<float>(0x3F800000u)"));
        assert!(mix.source.contains("& 0x7FFFFFFFu"));
        assert!(mix.source.contains("^ 0x80000000u"));
        assert!(mix.source.contains("metal::precise::fma"));

        let low = compile_vector_instruction(
            vop3p_instruction("VFmaMixloF16", mix_control),
            MetalShaderStage::Vertex,
        );
        assert!(low.source.contains("(v[8] & 0xFFFF0000u) |"));
        let high = compile_vector_instruction(
            vop3p_instruction("VFmaMixhiF16", mix_control),
            MetalShaderStage::Vertex,
        );
        assert!(high.source.contains("(v[8] & 0x0000FFFFu) |"));
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
                    lds_size_bytes: 0,
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
        assert!(shader.source.contains("simd_shuffle(v[1], (ushort)"));
        assert!(
            shader
                .source
                .contains("simd_shuffle(exec ? 1u : 0u, (ushort)")
        );
        assert!(shader.source.contains("(remu_lane & 15u) >= 1u"));
        assert!(shader.source.contains("fabs(as_type<float>"));
        assert!(shader.source.contains("-(as_type<float>(v[2]))"));
        assert!(shader.source.contains("5u >> (remu_lane >> 4)"));
        assert!(shader.source.contains("10u >> (remu_lane & 3u)"));
        assert!(shader.source.contains("? t"));
        assert!(shader.source.contains(": v[8]"));

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
        assert!(dpp8.source.contains("simd_shuffle(v[1], (ushort)"));
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
        assert!(compare.source.contains("s[106] ="));
        assert!(compare.source.contains("s[107] = 0u"));
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
        assert!(shader.source.contains("simd_shuffle(v[1], (ushort)"));
        assert!(
            shader
                .source
                .contains("s[126] = (uint)(uint64_t)simd_ballot(true)")
        );
        let shuffle = shader
            .source
            .find("simd_shuffle(v[1], (ushort)")
            .expect("DPP8 shuffle");
        let write = shader
            .source
            .find("if (exec) v[8]")
            .expect("EXEC-guarded destination write");
        assert!(
            shuffle < write,
            "wave shuffle must be converged before EXEC guard"
        );
    }
}
