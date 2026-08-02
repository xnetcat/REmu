// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#[cfg(test)]
use std::collections::BTreeMap;

use smallvec::SmallVec;

mod control_flow;
mod resources;
mod scalar_alu;

#[cfg(test)]
use crate::{BufferMemoryControl, GlobalMemoryControl, ImageControl, ScalarMemoryControl};
use crate::{
    GlobalMemoryBinding, GuestMemoryReader, ImageBinding, Instruction, InstructionControl, Opcode,
    ShaderEncoding, ShaderEvaluation, ShaderState, VertexInputBinding,
};
#[cfg(test)]
use crate::{Operand, OperandKind};
use control_flow::{
    ScalarPath, VisitedPaths, branch_condition_depends_on_vector_state, branch_taken,
    branch_target, is_scalar_branch, scalar_state_hash,
};
#[cfg(test)]
use resources::decode_buffer_descriptor;
#[cfg(test)]
use scalar_alu::execute_scalar_binary;
use scalar_alu::{
    execute_scalar_alu, execute_scalar_compare, execute_scalar_compare_k, write_scalar_pair,
};

// Descriptor-control fields can retain extended scalar indices even when ordinary source operand
// encodings in the same range denote inline constants. Keep the full decoder address space here;
// reducing this to the architectural SGPR count made those controls index past the path state.
const SCALAR_REGISTER_COUNT: usize = 256;
const SCALAR_REGISTER_COUNT_U32: u32 = 256;
// Gen5 export shaders are merged ES+GS programs. Hardware places the merged-wave descriptor in
// s3 and starts GS user data at s8. Metal launches the translated vertex function in SIMD32
// groups, while these guest prologues use the wave64 mask idiom `exec >> (64 - count)`. Present a
// full virtual wave here so the prologue preserves Metal's exact active-lane ballot; the Metal
// draw's vertex count still supplies the real tail mask for partial groups.
const MERGED_EXPORT_WAVE_INFO: u32 = 64 | (64 << 8);
const IMAGE_DESCRIPTOR_DWORDS: usize = 8;
const SAMPLER_DESCRIPTOR_DWORDS: usize = 4;
const MAX_PATH_STEPS: usize = 1 << 20;
const RDNA_WAVE_MASK: u64 = u32::MAX as u64;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EvaluationOptions {
    pub resolve_vertex_inputs: bool,
    pub required_vertex_record_count: Option<u32>,
    pub maximum_binding_bytes: usize,
    pub discover_static_control_flow: bool,
    /// Retain CPU copies of global bindings in the result. Runtime backends that bind
    /// identity-mapped guest memory directly can disable this to avoid copying multi-megabyte
    /// buffers once per draw.
    pub capture_global_binding_data: bool,
}

impl Default for EvaluationOptions {
    fn default() -> Self {
        Self {
            resolve_vertex_inputs: false,
            required_vertex_record_count: None,
            maximum_binding_bytes: 16 * 1024 * 1024,
            discover_static_control_flow: true,
            capture_global_binding_data: true,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum EvaluationError {
    #[error("invalid {detail} at pc 0x{pc:X} ({opcode})")]
    InvalidInstruction {
        pc: u32,
        opcode: Opcode,
        detail: &'static str,
    },
    #[error("unsupported scalar opcode {opcode} at pc 0x{pc:X}")]
    UnsupportedScalarOpcode { pc: u32, opcode: Opcode },
    #[error("branch at pc 0x{pc:X} targets invalid pc 0x{target:X}")]
    InvalidBranchTarget { pc: u32, target: u32 },
    #[error(
        "shader control flow exceeded {limit} evaluated instructions at pc 0x{pc:X} \
         with {pending_paths} pending paths"
    )]
    ControlFlowLimit {
        pc: u32,
        pending_paths: usize,
        limit: usize,
    },
    #[error("guest memory read failed at pc 0x{pc:X}: address {address:#x}, {length} bytes")]
    MemoryFault {
        pc: u32,
        address: u64,
        length: usize,
    },
    #[error(
        "buffer descriptor in s{register} is invalid at pc 0x{pc:X}: \
         [{words0:08X}, {words1:08X}, {words2:08X}, {words3:08X}]"
    )]
    InvalidBufferDescriptor {
        pc: u32,
        register: u32,
        words0: u32,
        words1: u32,
        words2: u32,
        words3: u32,
    },
    #[error("global address in s{register} is null at pc 0x{pc:X}")]
    NullGlobalAddress { pc: u32, register: u32 },
    #[error("image descriptors at pc 0x{pc:X} vary across scalar control-flow paths")]
    DynamicImageDescriptor { pc: u32 },
    #[error("binding size arithmetic overflowed at pc 0x{pc:X}")]
    BindingSizeOverflow { pc: u32 },
}

struct ScalarMemoryReads {
    sorted: Vec<(u64, u32)>,
}

impl ScalarMemoryReads {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            sorted: Vec::with_capacity(capacity),
        }
    }

    fn insert(&mut self, address: u64, value: u32) {
        match self.sorted.last_mut() {
            Some((last_address, last_value)) if *last_address == address => {
                *last_value = value;
                return;
            }
            Some((last_address, _)) if *last_address < address => {
                self.sorted.push((address, value));
                return;
            }
            _ => {}
        }
        match self
            .sorted
            .binary_search_by_key(&address, |(current, _)| *current)
        {
            Ok(index) => self.sorted[index].1 = value,
            Err(index) => self.sorted.insert(index, (address, value)),
        }
    }

    fn into_sorted_vec(self) -> Vec<(u64, u32)> {
        self.sorted
    }
}

struct Evaluator<'a, M> {
    memory: &'a M,
    state: &'a ShaderState,
    options: EvaluationOptions,
    initial_registers: [u32; SCALAR_REGISTER_COUNT],
    final_registers: [u32; SCALAR_REGISTER_COUNT],
    pending_paths: Vec<ScalarPath>,
    visited_paths: VisitedPaths,
    image_bindings: Vec<ImageBinding>,
    scalar_memory_reads: ScalarMemoryReads,
    global_bindings: Vec<GlobalMemoryBinding>,
    vertex_inputs: Vec<VertexInputBinding>,
    evaluated_steps: usize,
    maximum_steps: usize,
}

pub fn evaluate_shader(
    memory: &impl GuestMemoryReader,
    state: &ShaderState,
    options: EvaluationOptions,
) -> Result<ShaderEvaluation, EvaluationError> {
    evaluate_shader_within(memory, state, options, MAX_PATH_STEPS)
}

fn evaluate_shader_within(
    memory: &impl GuestMemoryReader,
    state: &ShaderState,
    options: EvaluationOptions,
    maximum_steps: usize,
) -> Result<ShaderEvaluation, EvaluationError> {
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    let user_base =
        usize::try_from(state.user_data_scalar_register_base).unwrap_or(SCALAR_REGISTER_COUNT);
    for (index, value) in state.user_data.iter().copied().enumerate() {
        let Some(destination) = user_base.checked_add(index) else {
            break;
        };
        let Some(register) = registers.get_mut(destination) else {
            break;
        };
        *register = value;
    }
    if state.user_data_scalar_register_base == 8 {
        registers[3] = MERGED_EXPORT_WAVE_INFO;
    }
    if let Some(system_registers) = state.compute_system_registers {
        system_registers.clear_static_values(&mut registers);
    }
    let mut exec_mask = RDNA_WAVE_MASK;
    write_scalar_pair(&mut registers, 106, 0, &mut exec_mask);
    write_scalar_pair(&mut registers, 126, exec_mask, &mut exec_mask);

    let mut evaluator = Evaluator {
        memory,
        state,
        options,
        initial_registers: registers,
        final_registers: registers,
        pending_paths: Vec::with_capacity(4),
        visited_paths: VisitedPaths::new(),
        image_bindings: Vec::with_capacity(state.program.image_binding_capacity_hint()),
        scalar_memory_reads: ScalarMemoryReads::with_capacity(
            state.program.scalar_memory_read_capacity_hint(),
        ),
        global_bindings: Vec::with_capacity(state.program.global_binding_capacity_hint()),
        vertex_inputs: Vec::with_capacity(4),
        evaluated_steps: 0,
        maximum_steps,
    };
    if !state.program.instructions.is_empty() {
        evaluator.queue_path(ScalarPath {
            instruction_index: 0,
            registers,
            exec_mask,
            scalar_condition: false,
            supplemental: false,
            backward_branch_visits: SmallVec::new(),
        });
    }
    while let Some(path) = evaluator.pending_paths.pop() {
        evaluator.execute_path(path)?;
    }
    evaluator.bind_statically_inactive_resources()?;
    let scalar_register_count = state
        .program
        .runtime_scalar_registers()
        .last()
        .and_then(|register| usize::try_from(*register).ok())
        .and_then(|register| register.checked_add(1))
        .unwrap_or_default();
    Ok(ShaderEvaluation {
        initial_scalar_registers: SmallVec::from_slice(
            &evaluator.initial_registers[..scalar_register_count],
        ),
        scalar_registers: SmallVec::from_slice(&evaluator.final_registers[..scalar_register_count]),
        scalar_memory_reads: evaluator.scalar_memory_reads.into_sorted_vec(),
        image_bindings: evaluator.image_bindings,
        global_memory_bindings: evaluator.global_bindings,
        compute_system_registers: state.compute_system_registers,
        vertex_inputs: evaluator.vertex_inputs,
    })
}

impl<M: GuestMemoryReader> Evaluator<'_, M> {
    fn queue_path(&mut self, mut path: ScalarPath) {
        let Some(instruction_index) = self
            .state
            .program
            .next_evaluation_instruction_index(path.instruction_index)
        else {
            return;
        };
        path.instruction_index = instruction_index;
        let instruction = &self.state.program.instructions[instruction_index];
        let key = (
            instruction.pc,
            scalar_state_hash(
                &path.registers,
                self.state
                    .program
                    .evaluation_live_scalar_register_indices(instruction_index),
                path.exec_mask,
                path.scalar_condition,
            ),
        );
        if self.visited_paths.insert(key) {
            self.pending_paths.push(path);
        }
    }

    #[allow(clippy::too_many_lines)]
    fn execute_path(&mut self, mut path: ScalarPath) -> Result<(), EvaluationError> {
        while let Some(instruction_index) = self
            .state
            .program
            .next_evaluation_instruction_index(path.instruction_index)
        {
            path.instruction_index = instruction_index;
            let instruction = &self.state.program.instructions[instruction_index];
            self.evaluated_steps = self.evaluated_steps.saturating_add(1);
            if self.evaluated_steps > self.maximum_steps {
                return Err(EvaluationError::ControlFlowLimit {
                    pc: instruction.pc,
                    pending_paths: self.pending_paths.len(),
                    limit: self.maximum_steps,
                });
            }
            if instruction.opcode == Opcode::END_PROGRAM {
                break;
            }
            if instruction.opcode.is("SSetpcB64") {
                let Some(target_pc) = self.state.program.linked_transfer_target(instruction.pc)
                else {
                    break;
                };
                path.instruction_index = self
                    .state
                    .program
                    .instructions
                    .binary_search_by_key(&target_pc, |candidate| candidate.pc)
                    .map_err(|_| EvaluationError::InvalidBranchTarget {
                        pc: instruction.pc,
                        target: target_pc,
                    })?;
                continue;
            }
            if instruction.encoding == ShaderEncoding::Sopp && is_scalar_branch(instruction.opcode)
            {
                let target_pc = branch_target(instruction)?;
                let target_index = self
                    .state
                    .program
                    .instructions
                    .binary_search_by_key(&target_pc, |candidate| candidate.pc)
                    .map_err(|_| EvaluationError::InvalidBranchTarget {
                        pc: instruction.pc,
                        target: target_pc,
                    })?;
                let taken = branch_taken(
                    instruction.opcode,
                    &path.registers,
                    path.exec_mask,
                    path.scalar_condition,
                )?;
                let backward = target_pc <= instruction.pc;
                let backward_visits = path.backward_branch_visits(instruction.pc);
                // SCC is produced entirely by scalar instructions that this evaluator executes,
                // so its branch direction is exact. Exploring the opposite SCC edge on every
                // scalar-loop iteration creates thousands of impossible paths and can exhaust the
                // global path budget. EXEC and VCC may be changed by skipped vector instructions;
                // retain alternate-edge discovery only for those runtime-vector conditions.
                if self.options.discover_static_control_flow
                    && branch_condition_depends_on_vector_state(instruction.opcode)
                    && (!backward || backward_visits == 0)
                {
                    let alternate_index = if taken {
                        path.instruction_index.saturating_add(1)
                    } else {
                        target_index
                    };
                    if backward && !taken {
                        // The alternate edge of a not-taken backward branch is the loop header, so
                        // it re-enters the loop exactly as taking the edge would. Charge it to the
                        // same budget before cloning so the fork inherits the charge.
                        path.record_backward_branch_visit(instruction.pc);
                    }
                    let mut alternate = path.clone();
                    alternate.instruction_index = alternate_index;
                    alternate.supplemental = true;
                    self.queue_path(alternate);
                }
                if taken {
                    if path.supplemental && target_pc <= instruction.pc {
                        break;
                    }
                    if backward {
                        // EXEC/VCC loops are controlled by vector lane state that static scalar
                        // evaluation intentionally does not execute. Follow one iteration to
                        // discover in-loop resources, then continue through the already queued
                        // exit path. Scalar SCC loops can still initialize bounded tables exactly.
                        let maximum_visits = if matches!(
                            instruction.opcode.known_name(),
                            Some(
                                "SBranch"
                                    | "SCbranchVccz"
                                    | "SCbranchVccnz"
                                    | "SCbranchExecz"
                                    | "SCbranchExecnz"
                            )
                        ) {
                            1
                        } else {
                            4_096
                        };
                        if backward_visits >= maximum_visits {
                            path.instruction_index = path.instruction_index.saturating_add(1);
                            continue;
                        }
                        path.record_backward_branch_visit(instruction.pc);
                    }
                    path.instruction_index = target_index;
                } else {
                    path.instruction_index = path.instruction_index.saturating_add(1);
                }
                continue;
            }
            if instruction.encoding == ShaderEncoding::Sopc {
                path.scalar_condition =
                    execute_scalar_compare(instruction, &path.registers, path.exec_mask)?;
            } else if instruction.encoding == ShaderEncoding::Sopk
                && instruction.opcode.starts_with("SCmpk")
            {
                path.scalar_condition = execute_scalar_compare_k(instruction, &path.registers)?;
            } else if matches!(
                instruction.encoding,
                ShaderEncoding::Sop1 | ShaderEncoding::Sop2 | ShaderEncoding::Sopk
            ) {
                if instruction.opcode.is("SSwappcB64") {
                    break;
                }
                execute_scalar_alu(
                    instruction,
                    self.state.program.instruction_address(instruction.pc),
                    &mut path.registers,
                    &mut path.exec_mask,
                    &mut path.scalar_condition,
                )?;
            } else if let Some(InstructionControl::ScalarMemory(control)) = &instruction.control {
                self.execute_scalar_load(instruction, *control, &mut path.registers)?;
            } else if let Some(InstructionControl::GlobalMemory(control)) = &instruction.control {
                self.resolve_global_memory(
                    instruction,
                    *control,
                    &path.registers,
                    path.supplemental,
                )?;
            } else if let Some(InstructionControl::BufferMemory(control)) = &instruction.control {
                self.resolve_buffer_memory(
                    instruction,
                    *control,
                    &path.registers,
                    path.supplemental,
                )?;
            } else if let Some(InstructionControl::Image(control)) = &instruction.control {
                self.resolve_image(instruction, control, &path.registers, path.supplemental)?;
            }
            path.instruction_index = path.instruction_index.saturating_add(1);
        }
        if !path.supplemental {
            self.final_registers = path.registers;
        }
        Ok(())
    }
}

fn invalid(instruction: &Instruction, detail: &'static str) -> EvaluationError {
    EvaluationError::InvalidInstruction {
        pc: instruction.pc,
        opcode: instruction.opcode,
        detail,
    }
}

#[cfg(test)]
mod tests;
