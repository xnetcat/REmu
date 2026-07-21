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
    ExportControl, ImageBinding, ImageControl, Instruction, InstructionControl, Opcode, Operand,
    OperandKind, PixelOutputBinding, PixelOutputKind, Program, ShaderEncoding, ShaderEvaluation,
    VertexInputBinding, decode_inline_constant, decode_unified_format,
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
             inline uint remu_load_u32(const device uchar* data, uint offset, uint length) {\n\
             \x20   uint value = 0u;\n\
             \x20   for (uint i = 0u; i < 4u; ++i) {\n\
             \x20       if (offset + i < length) value |= (uint)data[offset + i] << (i * 8u);\n\
             \x20   }\n\
             \x20   return value;\n}\n\n",
        );
        if self.program.instructions.iter().any(is_format_buffer_load) {
            emit_format_prelude(&mut source);
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
        for index in 0..global_count {
            parameters.push(format!("const device uchar* b{index} [[buffer({index})]]"));
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
                parameters
                    .push("uint3 remu_local_id [[thread_position_in_threadgroup]]".to_owned());
                parameters.push("uint3 remu_group_id [[threadgroup_position_in_grid]]".to_owned());
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
        source.push_str("    const uint remu_lane = 0u;\n");
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
            }
        }
        source.push_str("    s[106] = 0u; s[107] = 0u; s[126] = 1u; s[127] = 0u;\n");
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

    #[allow(clippy::unused_self)]
    fn raw_source(&self, instruction: &Instruction, index: usize) -> Result<String, String> {
        let operand = instruction
            .sources
            .get(index)
            .copied()
            .ok_or_else(|| format!("missing source operand {index}"))?;
        match operand.kind {
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
        }
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
        let mut value = format!("as_type<float>({})", self.raw_source(instruction, index)?);
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

    fn emit_vector_alu(&mut self, instruction: &Instruction) -> Result<(), String> {
        if matches!(
            instruction.control,
            Some(
                InstructionControl::Sdwa(_)
                    | InstructionControl::Dpp(_)
                    | InstructionControl::Dpp8(_)
                    | InstructionControl::Vop3p(_)
            )
        ) {
            return Err(
                "SDWA, DPP, DPP8, and packed VOP3 controls require their dedicated lowering"
                    .to_owned(),
            );
        }
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw vector opcode".to_owned())?;
        if name == "VCndmaskB32" {
            let destination = destination_vector(instruction)?;
            let mask = if instruction.sources.len() > 2 {
                self.mask_source(instruction.sources[2])?
            } else {
                "vcc".to_owned()
            };
            let false_value = self.raw_source(instruction, 0)?;
            let true_value = self.raw_source(instruction, 1)?;
            self.store_vector(
                destination,
                &format!("({mask}) ? ({true_value}) : ({false_value})"),
            );
            return Ok(());
        }
        let destination = destination_vector(instruction)?;
        let expression = match name {
            "VMovB32" => self.raw_source(instruction, 0)?,
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
        self.store_vector(destination, &expression);
        Ok(())
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

    fn emit_buffer_memory(&mut self, instruction: &Instruction) -> Result<(), String> {
        let Some(InstructionControl::BufferMemory(control)) = instruction.control.as_ref() else {
            return Err("missing buffer-memory control".to_owned());
        };
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
        let binding_length = self.evaluation.global_memory_bindings[binding_index]
            .data
            .len();
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
        let stride = format!("((s[{}] >> 16) & 0x3FFFu)", control.scalar_resource + 1);
        let address = self.temp(
            "uint",
            &format!(
                "({} + {scalar_offset} + {vector_offset} + ({vector_index} * {stride}))",
                format_u32(control.offset_bytes.cast_unsigned())
            ),
        );
        if is_format_buffer_load(instruction) {
            return self.emit_format_buffer_load(control, binding_index, binding_length, &address);
        }
        let Some(name) = instruction.opcode.known_name() else {
            return Err("raw buffer opcode".to_owned());
        };
        if name.starts_with("BufferLoad") {
            for component in 0..control.dword_count {
                self.store_vector(
                    control.vector_data + component,
                    &format!(
                        "remu_load_u32(b{binding_index}, {address} + {}u, {binding_length}u)",
                        component * 4
                    ),
                );
            }
            return Ok(());
        }
        Err(format!("unsupported generic buffer opcode {name}"))
    }

    fn emit_format_buffer_load(
        &mut self,
        control: &remu_shader::BufferMemoryControl,
        binding_index: usize,
        binding_length: usize,
        address: &str,
    ) -> Result<(), String> {
        if control.dword_count == 0 || control.dword_count > 4 {
            return Err(format!(
                "format load has invalid component count {}",
                control.dword_count
            ));
        }
        let descriptor_word = self.temp("uint", &format!("s[{}]", control.scalar_resource + 3));
        let table_entry = self.temp(
            "uint",
            &format!("remu_gfx10_formats[({descriptor_word} >> 12) & 0x7Fu]"),
        );
        let data_format = self.temp("uint", &format!("{table_entry} & 0xFFu"));
        let number_format = self.temp("uint", &format!("({table_entry} >> 8) & 0xFFu"));
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
            let packed = self.temp(
                "uint",
                &format!(
                    "remu_load_u32(b{binding_index}, {address} + {byte_offset}, {binding_length}u)"
                ),
            );
            let raw = self.temp(
                "uint",
                &format!("{bits} == 0u ? 0u : extract_bits({packed}, {bit_offset}, {bits})"),
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
        for destination in 0..control.dword_count {
            let selector = self.temp(
                "uint",
                &format!("({descriptor_word} >> {}u) & 7u", destination * 3),
            );
            self.store_vector(
                control.vector_data + destination,
                &format!(
                    "{selector} == 1u ? remu_format_one({number_format}) : {selector} == 4u ? {} : {selector} == 5u ? {} : {selector} == 6u ? {} : {selector} == 7u ? {} : 0u",
                    canonical[0], canonical[1], canonical[2], canonical[3]
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

fn is_format_buffer_load(instruction: &Instruction) -> bool {
    instruction.opcode.starts_with("BufferLoadFormat")
        || instruction.opcode.starts_with("TBufferLoadFormat")
}

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
    use remu_shader::{Instruction, Program, ShaderEvaluation};

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
}
