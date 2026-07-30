// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Compiler construction and the per-instruction lowering loop: interface
//! validation, initial register state and the opcode class dispatch.

use std::collections::{BTreeMap, BTreeSet};

use remu_shader::{
    Instruction, InstructionControl, OperandKind, Program, ShaderEvaluation, image_is_comparison,
};

use crate::compiler::control_flow::{
    ControlFlowLowering, LoweringFailure, analyze_control_flow, branch_target,
    combined_lowering_error, dispatcher_basic_blocks,
};
use crate::compiler::image::{
    ImageKind, image_component_type, is_storage_image, is_writable_image,
};
use crate::compiler::interface::{PixelInterpolation, interpolation_for_barycentric_vgpr};
use crate::compiler::memory::is_lds_memory_instruction;
use crate::{
    Compiler, METAL_DIRECT_TEXTURE_LIMIT, MetalBindingLayout, MetalCompileError,
    MetalCompileOptions, MetalRegisterFootprint, MetalShader, MetalShaderStage,
    PIXEL_INPUT_CONTROL_COUNT, SCALAR_REGISTER_COUNT, VECTOR_REGISTER_COUNT,
};

impl<'a> Compiler<'a> {
    pub(crate) fn new(
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
        let (readfirstlane_site_bases, disjoint_readfirstlane_slot_count) =
            Self::plan_disjoint_readfirstlane_layout(program, options, &control_flow);

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
            readfirstlane_site_bases,
            disjoint_readfirstlane_slot_count,
        })
    }

    pub(crate) fn compile(mut self) -> Result<MetalShader, MetalCompileError> {
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
            ControlFlowLowering::Forward | ControlFlowLowering::Dispatcher => {
                // Dispatch whole basic blocks instead of one instruction per switch iteration.
                // The architectural PC only has to be materialized at control-flow boundaries;
                // straight-line instructions inside a block execute in their original order.
                // Large Unity pixel shaders contain hundreds of instructions but only a handful
                // of branches, so this avoids hundreds of divergent switch operations per pixel
                // without changing branch, loop, or EXEC/VCC semantics.
                let forward = matches!(self.control_flow, ControlFlowLowering::Forward);
                for block in dispatcher_basic_blocks(self.program) {
                    let first = &self.program.instructions[block.start];
                    let block_instruction_count = (!forward).then_some(block.len());
                    if forward {
                        self.line(2, &format!("if (active && pc == {}u) {{", first.pc));
                    } else {
                        self.line(2, &format!("case {}u: {{", first.pc));
                    }
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
                    if let Some(block_instruction_count) = block_instruction_count {
                        self.line(3, &format!("steps += {block_instruction_count}u;"));
                        self.line(3, "break;");
                    }
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
            "SEndpgm" | "SSetpcB64" => {
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
}
fn validate_interface(options: &MetalCompileOptions) -> Result<(), MetalCompileError> {
    if !(1..=VECTOR_REGISTER_COUNT).contains(&options.allocated_vector_register_count) {
        return Err(MetalCompileError::InvalidInterface(format!(
            "allocated vector-register count {} is outside 1..={VECTOR_REGISTER_COUNT}",
            options.allocated_vector_register_count
        )));
    }
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
