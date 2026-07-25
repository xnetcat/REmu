// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#[cfg(test)]
use std::collections::BTreeMap;
use std::collections::HashSet;

use smallvec::SmallVec;

use crate::{
    BufferMemoryControl, GlobalMemoryBinding, GlobalMemoryControl, GuestMemoryReader, ImageBinding,
    ImageControl, Instruction, InstructionControl, Opcode, Operand, OperandKind,
    ScalarMemoryControl, ShaderEncoding, ShaderEvaluation, ShaderState, VertexInputBinding,
    decode_inline_constant, decode_unified_format,
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
    #[error("shader control flow exceeded {MAX_PATH_STEPS} evaluated instructions")]
    ControlFlowLimit,
    #[error("guest memory read failed at pc 0x{pc:X}: address {address:#x}, {length} bytes")]
    MemoryFault {
        pc: u32,
        address: u64,
        length: usize,
    },
    #[error("buffer descriptor in s{register} is invalid at pc 0x{pc:X}")]
    InvalidBufferDescriptor { pc: u32, register: u32 },
    #[error("global address in s{register} is null at pc 0x{pc:X}")]
    NullGlobalAddress { pc: u32, register: u32 },
    #[error("image descriptors at pc 0x{pc:X} vary across scalar control-flow paths")]
    DynamicImageDescriptor { pc: u32 },
    #[error("binding size arithmetic overflowed at pc 0x{pc:X}")]
    BindingSizeOverflow { pc: u32 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BufferDescriptor {
    base_address: u64,
    stride: u32,
    swizzle_enabled: bool,
    record_count: u32,
    size_bytes: u64,
    number_format: u32,
    data_format: u32,
}

#[derive(Clone, Copy)]
struct BindingRequest {
    scalar_address: u32,
    base_address: u64,
    requested_bytes: u64,
    writable: bool,
    supplemental: bool,
    allow_shorter: bool,
}

#[derive(Clone)]
struct ScalarPath {
    instruction_index: usize,
    registers: [u32; SCALAR_REGISTER_COUNT],
    exec_mask: u64,
    scalar_condition: bool,
    supplemental: bool,
}

struct VisitedPaths {
    small: Vec<(u32, u64)>,
    large: Option<HashSet<(u32, u64)>>,
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

impl VisitedPaths {
    fn new() -> Self {
        Self {
            small: Vec::with_capacity(8),
            large: None,
        }
    }

    fn insert(&mut self, key: (u32, u64)) -> bool {
        if let Some(large) = self.large.as_mut() {
            return large.insert(key);
        }
        if self.small.contains(&key) {
            return false;
        }
        if self.small.len() < 16 {
            self.small.push(key);
            return true;
        }
        let mut large = HashSet::with_capacity(32);
        large.extend(self.small.drain(..));
        let inserted = large.insert(key);
        self.large = Some(large);
        inserted
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
}

pub fn evaluate_shader(
    memory: &impl GuestMemoryReader,
    state: &ShaderState,
    options: EvaluationOptions,
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
    };
    if !state.program.instructions.is_empty() {
        evaluator.queue_path(ScalarPath {
            instruction_index: 0,
            registers,
            exec_mask,
            scalar_condition: false,
            supplemental: false,
        });
    }
    while let Some(path) = evaluator.pending_paths.pop() {
        evaluator.execute_path(path)?;
    }
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
                self.state.program.runtime_scalar_register_indices(),
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
        let mut backward_branch_visits = Vec::<(u32, usize)>::with_capacity(4);
        while let Some(instruction_index) = self
            .state
            .program
            .next_evaluation_instruction_index(path.instruction_index)
        {
            path.instruction_index = instruction_index;
            let instruction = &self.state.program.instructions[instruction_index];
            self.evaluated_steps = self.evaluated_steps.saturating_add(1);
            if self.evaluated_steps > MAX_PATH_STEPS {
                return Err(EvaluationError::ControlFlowLimit);
            }
            if instruction.opcode == Opcode::END_PROGRAM {
                break;
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
                let backward_visits = backward_branch_visits
                    .iter()
                    .find(|(pc, _)| *pc == instruction.pc)
                    .map(|(_, visits)| *visits)
                    .unwrap_or_default();
                if self.options.discover_static_control_flow && (!backward || backward_visits == 0)
                {
                    let alternate_index = if taken {
                        path.instruction_index.saturating_add(1)
                    } else {
                        target_index
                    };
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
                        if let Some((_, visits)) = backward_branch_visits
                            .iter_mut()
                            .find(|(pc, _)| *pc == instruction.pc)
                        {
                            *visits = visits.saturating_add(1);
                        } else {
                            backward_branch_visits.push((instruction.pc, 1));
                        }
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
                if instruction.opcode.is("SSetpcB64") || instruction.opcode.is("SSwappcB64") {
                    break;
                }
                execute_scalar_alu(
                    instruction,
                    self.state.program.address,
                    &mut path.registers,
                    &mut path.exec_mask,
                    &mut path.scalar_condition,
                )?;
            } else if let Some(InstructionControl::ScalarMemory(control)) = &instruction.control {
                self.execute_scalar_load(
                    instruction,
                    *control,
                    &mut path.registers,
                    path.supplemental,
                )?;
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

impl<M: GuestMemoryReader> Evaluator<'_, M> {
    fn execute_scalar_load(
        &mut self,
        instruction: &Instruction,
        control: ScalarMemoryControl,
        registers: &mut [u32; SCALAR_REGISTER_COUNT],
        supplemental: bool,
    ) -> Result<(), EvaluationError> {
        let scalar_base = match instruction.sources.first() {
            Some(Operand {
                kind: OperandKind::ScalarRegister,
                value,
            }) if *value < SCALAR_REGISTER_COUNT_U32 - 1 => *value,
            _ => return Err(invalid(instruction, "scalar-memory base register pair")),
        };
        let is_buffer = instruction.opcode.starts_with("SBufferLoad");
        let descriptor = is_buffer.then(|| decode_buffer_descriptor(registers, scalar_base, false));
        let descriptor =
            descriptor
                .transpose()
                .map_err(|()| EvaluationError::InvalidBufferDescriptor {
                    pc: instruction.pc,
                    register: scalar_base,
                })?;
        let base_address = descriptor.map_or_else(
            || read_scalar_pair(registers, scalar_base).unwrap_or_default(),
            |descriptor| descriptor.base_address,
        );
        let dynamic_offset = control
            .dynamic_offset_register
            .and_then(|register| registers.get(register as usize))
            .copied()
            .unwrap_or_default();
        let byte_offset = u64::from(dynamic_offset)
            .wrapping_add(i64::from(control.immediate_offset_bytes).cast_unsigned());
        let address = base_address.wrapping_add(byte_offset) & !3;
        let unbound =
            base_address == 0 || descriptor.is_some_and(|descriptor| descriptor.size_bytes == 0);
        if unbound && !supplemental {
            return Err(EvaluationError::NullGlobalAddress {
                pc: instruction.pc,
                register: scalar_base,
            });
        }

        let binding_size = if let Some(descriptor) = descriptor {
            descriptor.size_bytes
        } else {
            byte_offset
                .checked_add(
                    u64::try_from(instruction.destinations.len().max(1) * 4)
                        .map_err(|_| EvaluationError::BindingSizeOverflow { pc: instruction.pc })?,
                )
                .ok_or(EvaluationError::BindingSizeOverflow { pc: instruction.pc })?
        };
        self.record_global_binding(
            instruction,
            BindingRequest {
                scalar_address: scalar_base,
                base_address,
                requested_bytes: binding_size,
                writable: false,
                supplemental,
                allow_shorter: !is_buffer,
            },
        )?;

        for destination in &instruction.destinations {
            if destination.kind != OperandKind::ScalarRegister
                || destination.value >= SCALAR_REGISTER_COUNT_U32
            {
                return Err(invalid(instruction, "scalar-memory destination"));
            }
        }
        if unbound {
            for destination in &instruction.destinations {
                registers[destination.value as usize] = 0;
            }
            return Ok(());
        }

        // Hardware scalar loads fetch a contiguous vector. Read the complete vector in one guest
        // access as well: ProcessGuestMemory can validate one mapped range and copy it directly,
        // instead of repeating its TLS generation/range lookup for every four-byte component.
        // GFX10 scalar loads top out at 16 dwords, so this stays allocation-free.
        let component_byte_count = instruction.destinations.len().saturating_mul(4);
        let bulk_in_bounds = descriptor.is_none_or(|descriptor| {
            byte_offset
                .checked_add(component_byte_count as u64)
                .is_some_and(|end| end <= descriptor.size_bytes)
        });
        if component_byte_count <= 64 && bulk_in_bounds {
            let mut bytes = [0_u8; 64];
            if !self
                .memory
                .read(address, &mut bytes[..component_byte_count])
            {
                return Err(EvaluationError::MemoryFault {
                    pc: instruction.pc,
                    address,
                    length: component_byte_count,
                });
            }
            for (index, destination) in instruction.destinations.iter().enumerate() {
                let component_address = address.wrapping_add((index as u64) * 4);
                let component = index * 4;
                let value = u32::from_le_bytes(
                    bytes[component..component + 4]
                        .try_into()
                        .expect("scalar component is four bytes"),
                );
                self.scalar_memory_reads.insert(component_address, value);
                registers[destination.value as usize] = value;
            }
            return Ok(());
        }

        for (index, destination) in instruction.destinations.iter().enumerate() {
            let component_offset = byte_offset.wrapping_add((index as u64) * 4);
            if descriptor.is_some_and(|descriptor| {
                component_offset
                    .checked_add(4)
                    .is_none_or(|end| end > descriptor.size_bytes)
            }) {
                if supplemental {
                    registers[destination.value as usize] = 0;
                    continue;
                }
                return Err(invalid(instruction, "scalar buffer load bounds"));
            }
            let component_address = address.wrapping_add((index as u64) * 4);
            let mut bytes = [0_u8; 4];
            if !self.memory.read(component_address, &mut bytes) {
                return Err(EvaluationError::MemoryFault {
                    pc: instruction.pc,
                    address: component_address,
                    length: 4,
                });
            }
            let value = u32::from_le_bytes(bytes);
            self.scalar_memory_reads.insert(component_address, value);
            registers[destination.value as usize] = value;
        }
        Ok(())
    }

    fn resolve_global_memory(
        &mut self,
        instruction: &Instruction,
        control: GlobalMemoryControl,
        registers: &[u32; SCALAR_REGISTER_COUNT],
        supplemental: bool,
    ) -> Result<(), EvaluationError> {
        let Some(scalar_address) = control.scalar_address else {
            // With SADDR=NULL the complete 64-bit address is supplied per lane by VGPRs.
            // There is no scalar value to resolve statically; Metal lowering selects from the
            // guest ranges discovered by the other memory instructions at run time.
            return Ok(());
        };
        let Some(base_address) = read_scalar_pair(registers, scalar_address) else {
            return Err(invalid(instruction, "global-memory scalar address pair"));
        };
        if base_address == 0 && !supplemental {
            return Err(EvaluationError::NullGlobalAddress {
                pc: instruction.pc,
                register: scalar_address,
            });
        }
        let minimum_address = if control.offset_bytes.is_negative() {
            base_address.wrapping_sub(u64::from(control.offset_bytes.unsigned_abs()))
        } else {
            base_address
        };
        let binding_base = minimum_address & !3;
        let prefix_bytes = base_address.wrapping_sub(binding_base);
        let writable = instruction.opcode.starts_with("GlobalStore")
            || instruction.opcode.starts_with("GlobalAtomic");
        self.record_global_binding(
            instruction,
            BindingRequest {
                scalar_address,
                base_address: binding_base,
                requested_bytes: (self.options.maximum_binding_bytes as u64)
                    .saturating_add(prefix_bytes),
                writable,
                supplemental,
                allow_shorter: true,
            },
        )
    }

    fn resolve_buffer_memory(
        &mut self,
        instruction: &Instruction,
        control: BufferMemoryControl,
        registers: &[u32; SCALAR_REGISTER_COUNT],
        supplemental: bool,
    ) -> Result<(), EvaluationError> {
        let descriptor = decode_buffer_descriptor(registers, control.scalar_resource, true)
            .map_err(|()| EvaluationError::InvalidBufferDescriptor {
                pc: instruction.pc,
                register: control.scalar_resource,
            })?;
        let writable =
            instruction.opcode.contains("Store") || instruction.opcode.contains("Atomic");
        if descriptor.base_address == 0 {
            if !supplemental {
                return Err(EvaluationError::NullGlobalAddress {
                    pc: instruction.pc,
                    register: control.scalar_resource,
                });
            }
            return self.record_global_binding(
                instruction,
                BindingRequest {
                    scalar_address: control.scalar_resource,
                    base_address: 0,
                    requested_bytes: 4,
                    writable,
                    supplemental: true,
                    allow_shorter: false,
                },
            );
        }
        let bounded_vertex_fetch = is_vertex_fetch(instruction, control, descriptor)
            && (self.options.resolve_vertex_inputs
                || self.options.required_vertex_record_count.is_some());
        if bounded_vertex_fetch {
            let scalar_offset = instruction
                .sources
                .get(2)
                .and_then(|operand| evaluate_operand(operand, registers))
                .ok_or_else(|| invalid(instruction, "vertex scalar offset"))?;
            let binding_offset = control
                .offset_bytes
                .cast_unsigned()
                .wrapping_add(scalar_offset);
            let desired_size = required_vertex_bytes(
                descriptor,
                binding_offset,
                control.dword_count,
                self.options.required_vertex_record_count,
            )
            .min(self.options.maximum_binding_bytes as u64);
            if self.options.resolve_vertex_inputs {
                let data =
                    self.read_exact_binding(instruction.pc, descriptor.base_address, desired_size)?;
                self.vertex_inputs.push(VertexInputBinding {
                    pc: instruction.pc,
                    location: u32::try_from(self.vertex_inputs.len())
                        .map_err(|_| EvaluationError::BindingSizeOverflow { pc: instruction.pc })?,
                    component_count: control.dword_count,
                    data_format: descriptor.data_format,
                    number_format: descriptor.number_format,
                    base_address: descriptor.base_address,
                    stride: descriptor.stride,
                    offset_bytes: binding_offset,
                    data,
                });
            } else {
                let binding_base = descriptor.base_address & !3;
                let prefix_bytes = descriptor.base_address - binding_base;
                self.record_global_binding(
                    instruction,
                    BindingRequest {
                        scalar_address: control.scalar_resource,
                        base_address: binding_base,
                        requested_bytes: desired_size.saturating_add(prefix_bytes),
                        writable,
                        supplemental,
                        allow_shorter: false,
                    },
                )?;
            }
            return Ok(());
        }
        let binding_base = descriptor.base_address & !3;
        let prefix_bytes = descriptor.base_address - binding_base;
        self.record_global_binding(
            instruction,
            BindingRequest {
                scalar_address: control.scalar_resource,
                base_address: binding_base,
                requested_bytes: descriptor.size_bytes.saturating_add(prefix_bytes),
                writable,
                supplemental,
                allow_shorter: false,
            },
        )
    }

    fn resolve_image(
        &mut self,
        instruction: &Instruction,
        control: &ImageControl,
        registers: &[u32; SCALAR_REGISTER_COUNT],
        supplemental: bool,
    ) -> Result<(), EvaluationError> {
        if supplemental
            && self
                .image_bindings
                .iter()
                .any(|binding| binding.pc == instruction.pc)
        {
            return Ok(());
        }
        let resource_descriptor =
            copy_registers::<8>(registers, control.scalar_resource, IMAGE_DESCRIPTOR_DWORDS)
                .ok_or_else(|| invalid(instruction, "image resource descriptor registers"))?;
        let sampler_descriptor = if uses_sampler(instruction.opcode) {
            copy_registers::<4>(registers, control.scalar_sampler, SAMPLER_DESCRIPTOR_DWORDS)
                .ok_or_else(|| invalid(instruction, "image sampler descriptor registers"))?
        } else {
            SmallVec::new()
        };
        let mip_level = (instruction.opcode.is("ImageLoadMip")
            || instruction.opcode.is("ImageStoreMip"))
        .then(|| resolve_image_mip_level(self.state, instruction.pc, control))
        .flatten();
        let binding = ImageBinding {
            pc: instruction.pc,
            opcode: instruction.opcode,
            control: control.clone(),
            resource_descriptor,
            sampler_descriptor,
            mip_level,
        };
        if let Some(index) = self
            .image_bindings
            .iter()
            .position(|existing| existing.pc == instruction.pc)
        {
            let existing = &self.image_bindings[index];
            let existing_null = existing.resource_descriptor.iter().all(|word| *word == 0);
            let candidate_null = binding.resource_descriptor.iter().all(|word| *word == 0);
            if existing_null && !candidate_null {
                self.image_bindings[index] = binding;
            } else if !candidate_null
                && (existing.resource_descriptor != binding.resource_descriptor
                    || existing.sampler_descriptor != binding.sampler_descriptor)
            {
                return Err(EvaluationError::DynamicImageDescriptor { pc: instruction.pc });
            }
        } else {
            self.image_bindings.push(binding);
        }
        Ok(())
    }

    fn record_global_binding(
        &mut self,
        instruction: &Instruction,
        request: BindingRequest,
    ) -> Result<(), EvaluationError> {
        if let Some(index) = self.global_bindings.iter().position(|binding| {
            binding.scalar_address == request.scalar_address
                && binding.base_address == request.base_address
        }) {
            if request.requested_bytes > self.global_bindings[index].byte_count as u64 {
                let (byte_count, expanded) = self.read_binding_request(instruction.pc, request)?;
                if byte_count > self.global_bindings[index].byte_count {
                    self.global_bindings[index].byte_count = byte_count;
                    self.global_bindings[index].data = expanded;
                }
            }
            let binding = &mut self.global_bindings[index];
            if !binding.instruction_pcs.contains(&instruction.pc) {
                binding.instruction_pcs.push(instruction.pc);
            }
            binding.writable |= request.writable;
            return Ok(());
        }
        let (byte_count, data) = self.read_binding_request(instruction.pc, request)?;
        let mut instruction_pcs = SmallVec::new();
        instruction_pcs.push(instruction.pc);
        let binding = GlobalMemoryBinding {
            scalar_address: request.scalar_address,
            base_address: request.base_address,
            instruction_pcs,
            byte_count,
            data,
            writable: request.writable,
            write_back_to_guest: request.base_address != 0,
        };
        self.global_bindings.push(binding);
        Ok(())
    }

    fn read_binding_request(
        &self,
        pc: u32,
        request: BindingRequest,
    ) -> Result<(usize, Vec<u8>), EvaluationError> {
        let data = if request.base_address == 0 && request.supplemental {
            vec![0; usize::try_from(request.requested_bytes.clamp(4, 4096)).unwrap_or(4)]
        } else if !self.options.capture_global_binding_data {
            let maximum_length = usize::try_from(
                request
                    .requested_bytes
                    .max(4)
                    .min(self.options.maximum_binding_bytes as u64),
            )
            .map_err(|_| EvaluationError::BindingSizeOverflow { pc })?;
            if let Some(readable_length) = self
                .memory
                .readable_prefix_length(request.base_address, maximum_length)
            {
                let byte_count = if request.allow_shorter {
                    readable_length.min(maximum_length)
                } else if readable_length >= maximum_length {
                    maximum_length
                } else {
                    return Err(EvaluationError::MemoryFault {
                        pc,
                        address: request.base_address,
                        length: maximum_length,
                    });
                };
                if byte_count < 4 {
                    return Err(EvaluationError::MemoryFault {
                        pc,
                        address: request.base_address,
                        length: maximum_length,
                    });
                }
                return Ok((byte_count, Vec::new()));
            }
            if request.allow_shorter {
                self.read_adaptive_binding(pc, request.base_address, request.requested_bytes)?
            } else {
                self.read_exact_binding(pc, request.base_address, request.requested_bytes)?
            }
        } else if request.allow_shorter {
            self.read_adaptive_binding(pc, request.base_address, request.requested_bytes)?
        } else {
            self.read_exact_binding(pc, request.base_address, request.requested_bytes)?
        };
        Ok((data.len(), data))
    }

    fn read_exact_binding(
        &self,
        pc: u32,
        address: u64,
        requested_bytes: u64,
    ) -> Result<Vec<u8>, EvaluationError> {
        let length = usize::try_from(
            requested_bytes
                .max(4)
                .min(self.options.maximum_binding_bytes as u64),
        )
        .map_err(|_| EvaluationError::BindingSizeOverflow { pc })?;
        let mut data = vec![0; length];
        if !self.memory.read(address, &mut data) {
            return Err(EvaluationError::MemoryFault {
                pc,
                address,
                length,
            });
        }
        Ok(data)
    }

    fn read_adaptive_binding(
        &self,
        pc: u32,
        address: u64,
        requested_bytes: u64,
    ) -> Result<Vec<u8>, EvaluationError> {
        let mut length = usize::try_from(
            requested_bytes
                .max(4)
                .min(self.options.maximum_binding_bytes as u64),
        )
        .map_err(|_| EvaluationError::BindingSizeOverflow { pc })?;
        loop {
            let mut data = vec![0; length];
            if self.memory.read(address, &mut data) {
                return Ok(data);
            }
            if length == 4 {
                return Err(EvaluationError::MemoryFault {
                    pc,
                    address,
                    length,
                });
            }
            length = (length / 2).max(4);
        }
    }
}

fn scalar_state_hash(
    registers: &[u32],
    register_indices: &[u8],
    exec_mask: u64,
    scalar_condition: bool,
) -> u64 {
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    let mut hash = OFFSET;
    for &index in register_indices {
        hash = (hash ^ u64::from(registers[usize::from(index)])).wrapping_mul(PRIME);
    }
    hash = (hash ^ exec_mask).wrapping_mul(PRIME);
    (hash ^ u64::from(scalar_condition)).wrapping_mul(PRIME)
}

#[allow(clippy::cast_possible_truncation)]
const fn low_u16(value: u32) -> u16 {
    (value & u16::MAX as u32) as u16
}

#[allow(clippy::cast_possible_truncation)]
const fn low_u32(value: u64) -> u32 {
    (value & u32::MAX as u64) as u32
}

fn branch_target(instruction: &Instruction) -> Result<u32, EvaluationError> {
    let offset = low_u16(instruction.words[0]).cast_signed();
    let next_pc =
        i64::from(instruction.pc) + i64::try_from(instruction.words.len() * 4).unwrap_or(i64::MAX);
    let target = next_pc + i64::from(offset) * 4;
    u32::try_from(target).map_err(|_| EvaluationError::InvalidBranchTarget {
        pc: instruction.pc,
        target: if target < 0 { 0 } else { u32::MAX },
    })
}

fn branch_taken(
    opcode: Opcode,
    registers: &[u32; SCALAR_REGISTER_COUNT],
    exec_mask: u64,
    scalar_condition: bool,
) -> Result<bool, EvaluationError> {
    let taken = match opcode.known_name() {
        Some("SBranch") => true,
        Some("SCbranchScc0") => !scalar_condition,
        Some("SCbranchScc1") => scalar_condition,
        Some("SCbranchVccz") => registers[106] == 0 && registers[107] == 0,
        Some("SCbranchVccnz") => registers[106] != 0 || registers[107] != 0,
        Some("SCbranchExecz") => exec_mask == 0,
        Some("SCbranchExecnz") => exec_mask != 0,
        _ => {
            return Err(EvaluationError::InvalidInstruction {
                pc: 0,
                opcode,
                detail: "scalar branch opcode",
            });
        }
    };
    Ok(taken)
}

fn is_scalar_branch(opcode: Opcode) -> bool {
    matches!(
        opcode.known_name(),
        Some(
            "SBranch"
                | "SCbranchScc0"
                | "SCbranchScc1"
                | "SCbranchVccz"
                | "SCbranchVccnz"
                | "SCbranchExecz"
                | "SCbranchExecnz"
        )
    )
}

fn invalid(instruction: &Instruction, detail: &'static str) -> EvaluationError {
    EvaluationError::InvalidInstruction {
        pc: instruction.pc,
        opcode: instruction.opcode,
        detail,
    }
}

fn scalar_destination(instruction: &Instruction) -> Result<u32, EvaluationError> {
    match instruction.destinations.as_slice() {
        [
            Operand {
                kind: OperandKind::ScalarRegister,
                value,
            },
        ] if *value < SCALAR_REGISTER_COUNT_U32 => Ok(*value),
        _ => Err(invalid(instruction, "scalar destination")),
    }
}

#[allow(clippy::too_many_lines)]
fn execute_scalar_alu(
    instruction: &Instruction,
    program_address: u64,
    registers: &mut [u32; SCALAR_REGISTER_COUNT],
    exec_mask: &mut u64,
    scalar_condition: &mut bool,
) -> Result<(), EvaluationError> {
    let destination = scalar_destination(instruction)?;
    if instruction.opcode.is("SMovkI32") {
        registers[destination as usize] =
            i32::from(low_u16(instruction.sources[0].value).cast_signed()).cast_unsigned();
        return Ok(());
    }
    if instruction.opcode.is("SAddkI32") || instruction.opcode.is("SMulkI32") {
        let immediate =
            i32::from(low_u16(instruction.sources[0].value).cast_signed()).cast_unsigned();
        let current = registers[destination as usize];
        registers[destination as usize] = if instruction.opcode.is("SAddkI32") {
            current.wrapping_add(immediate)
        } else {
            current
                .cast_signed()
                .wrapping_mul(immediate.cast_signed())
                .cast_unsigned()
        };
        return Ok(());
    }
    if instruction.opcode.is("SGetpcB64") {
        let pc = program_address
            .wrapping_add(u64::from(instruction.pc))
            .wrapping_add(u64::try_from(instruction.words.len() * 4).unwrap_or_default());
        write_scalar_pair(registers, destination, pc, exec_mask);
        return Ok(());
    }
    if execute_save_exec(instruction, registers, exec_mask, scalar_condition)? {
        return Ok(());
    }

    if matches!(
        instruction.opcode.known_name(),
        Some("SMovB64" | "SWqmB64" | "SNotB64")
    ) {
        let mut value = evaluate_operand64(
            instruction
                .sources
                .first()
                .ok_or_else(|| invalid(instruction, "64-bit source"))?,
            registers,
            *exec_mask,
        )
        .ok_or_else(|| invalid(instruction, "64-bit source"))?;
        if instruction.opcode.is("SNotB64") {
            value = !value;
            *scalar_condition = value != 0;
        } else if instruction.opcode.is("SWqmB64") {
            let quad_any =
                (value | (value >> 1) | (value >> 2) | (value >> 3)) & 0x1111_1111_1111_1111;
            value = quad_any.wrapping_mul(0xf);
            *scalar_condition = value != 0;
        }
        write_scalar_pair(registers, destination, value, exec_mask);
        return Ok(());
    }

    if matches!(
        instruction.opcode.known_name(),
        Some("SLshlB64" | "SLshrB64" | "SAshrI64")
    ) {
        let left = evaluate_operand64(
            instruction
                .sources
                .first()
                .ok_or_else(|| invalid(instruction, "64-bit source"))?,
            registers,
            *exec_mask,
        )
        .ok_or_else(|| invalid(instruction, "64-bit source"))?;
        let right = evaluate_operand(
            instruction
                .sources
                .get(1)
                .ok_or_else(|| invalid(instruction, "shift source"))?,
            registers,
        )
        .ok_or_else(|| invalid(instruction, "shift source"))?;
        let shift = right & 63;
        let result = if instruction.opcode.is("SLshlB64") {
            left << shift
        } else if instruction.opcode.is("SLshrB64") {
            left >> shift
        } else {
            (left.cast_signed() >> shift).cast_unsigned()
        };
        write_scalar_pair(registers, destination, result, exec_mask);
        *scalar_condition = result != 0;
        return Ok(());
    }

    if instruction.opcode.is("SBfeU64") || instruction.opcode.is("SBfeI64") {
        let source = evaluate_operand64(
            instruction
                .sources
                .first()
                .ok_or_else(|| invalid(instruction, "64-bit source"))?,
            registers,
            *exec_mask,
        )
        .ok_or_else(|| invalid(instruction, "64-bit source"))?;
        let control = evaluate_operand(
            instruction
                .sources
                .get(1)
                .ok_or_else(|| invalid(instruction, "bitfield control"))?,
            registers,
        )
        .ok_or_else(|| invalid(instruction, "bitfield control"))?;
        let offset = control & 63;
        let width = ((control >> 16) & 0x7f).min(64 - offset);
        let mut result = if width == 0 { 0 } else { source >> offset };
        if width != 0 && width < 64 {
            result &= u64::MAX >> (64 - width);
            if instruction.opcode.is("SBfeI64") {
                result = ((result << (64 - width)).cast_signed() >> (64 - width)).cast_unsigned();
            }
        }
        write_scalar_pair(registers, destination, result, exec_mask);
        *scalar_condition = result != 0;
        return Ok(());
    }

    if instruction.opcode.is("SBfmB64") {
        let width = evaluate_operand(
            instruction
                .sources
                .first()
                .ok_or_else(|| invalid(instruction, "bitmask width"))?,
            registers,
        )
        .ok_or_else(|| invalid(instruction, "bitmask width"))?
            & 63;
        let offset = evaluate_operand(
            instruction
                .sources
                .get(1)
                .ok_or_else(|| invalid(instruction, "bitmask offset"))?,
            registers,
        )
        .ok_or_else(|| invalid(instruction, "bitmask offset"))?
            & 63;
        let result = if width == 0 {
            0
        } else {
            (u64::MAX >> (64 - width)) << offset
        };
        write_scalar_pair(registers, destination, result, exec_mask);
        *scalar_condition = result != 0;
        return Ok(());
    }

    if matches!(
        instruction.opcode.known_name(),
        Some(
            "SCselectB64"
                | "SAndB64"
                | "SOrB64"
                | "SXorB64"
                | "SAndn2B64"
                | "SOrn2B64"
                | "SNandB64"
                | "SNorB64"
                | "SXnorB64"
        )
    ) {
        let left = evaluate_operand64(
            instruction
                .sources
                .first()
                .ok_or_else(|| invalid(instruction, "64-bit source 0"))?,
            registers,
            *exec_mask,
        )
        .ok_or_else(|| invalid(instruction, "64-bit source 0"))?;
        let right = evaluate_operand64(
            instruction
                .sources
                .get(1)
                .ok_or_else(|| invalid(instruction, "64-bit source 1"))?,
            registers,
            *exec_mask,
        )
        .ok_or_else(|| invalid(instruction, "64-bit source 1"))?;
        let result = match instruction.opcode.known_name() {
            Some("SCselectB64") => {
                if *scalar_condition {
                    left
                } else {
                    right
                }
            }
            Some("SAndB64") => left & right,
            Some("SOrB64") => left | right,
            Some("SXorB64") => left ^ right,
            Some("SAndn2B64") => left & !right,
            Some("SOrn2B64") => left | !right,
            Some("SNandB64") => !(left & right),
            Some("SNorB64") => !(left | right),
            Some("SXnorB64") => !(left ^ right),
            _ => unreachable!(),
        };
        write_scalar_pair(registers, destination, result, exec_mask);
        *scalar_condition = result != 0;
        return Ok(());
    }

    let left = evaluate_wave_operand(
        instruction
            .sources
            .first()
            .ok_or_else(|| invalid(instruction, "scalar source 0"))?,
        registers,
        *exec_mask,
        *scalar_condition,
    )
    .ok_or_else(|| invalid(instruction, "scalar source 0"))?;
    if instruction.opcode.is("SMovB32") {
        registers[destination as usize] = left;
        return Ok(());
    }
    if matches!(
        instruction.opcode.known_name(),
        Some("SNotB32" | "SBrevB32" | "SBcnt1I32B32" | "SFF1I32B32" | "SBitset1B32")
    ) {
        let result = match instruction.opcode.known_name() {
            Some("SNotB32") => !left,
            Some("SBrevB32") => left.reverse_bits(),
            Some("SBcnt1I32B32") => left.count_ones(),
            Some("SFF1I32B32") => {
                if left == 0 {
                    u32::MAX
                } else {
                    left.trailing_zeros()
                }
            }
            Some("SBitset1B32") => registers[destination as usize] | (1 << (left & 31)),
            _ => unreachable!(),
        };
        registers[destination as usize] = result;
        if !instruction.opcode.is("SBitset1B32") {
            *scalar_condition = result != 0;
        }
        return Ok(());
    }
    let right = evaluate_wave_operand(
        instruction
            .sources
            .get(1)
            .ok_or_else(|| invalid(instruction, "scalar source 1"))?,
        registers,
        *exec_mask,
        *scalar_condition,
    )
    .ok_or_else(|| invalid(instruction, "scalar source 1"))?;
    let result = execute_scalar_binary(instruction, left, right, scalar_condition)?;
    registers[destination as usize] = result;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn execute_scalar_binary(
    instruction: &Instruction,
    left: u32,
    right: u32,
    scalar_condition: &mut bool,
) -> Result<u32, EvaluationError> {
    let result = match instruction.opcode.known_name() {
        Some("SAddU32") => {
            let (value, carry) = left.overflowing_add(right);
            *scalar_condition = carry;
            value
        }
        Some("SSubU32") => {
            let (value, borrow) = left.overflowing_sub(right);
            *scalar_condition = borrow;
            value
        }
        Some("SAddI32") => {
            let value = left.wrapping_add(right);
            *scalar_condition = ((left ^ value) & (right ^ value) & 0x8000_0000) != 0;
            value
        }
        Some("SSubI32") => {
            let value = left.wrapping_sub(right);
            *scalar_condition = ((left ^ right) & (left ^ value) & 0x8000_0000) != 0;
            value
        }
        Some("SAddcU32") => {
            let wide = u64::from(left) + u64::from(right) + u64::from(*scalar_condition);
            *scalar_condition = wide > u64::from(u32::MAX);
            low_u32(wide)
        }
        Some("SSubbU32") => {
            let subtrahend = u64::from(right) + u64::from(*scalar_condition);
            *scalar_condition = subtrahend > u64::from(left);
            left.wrapping_sub(low_u32(subtrahend))
        }
        Some("SMinI32") => {
            *scalar_condition = left.cast_signed() < right.cast_signed();
            left.cast_signed().min(right.cast_signed()).cast_unsigned()
        }
        Some("SMinU32") => {
            *scalar_condition = left < right;
            left.min(right)
        }
        Some("SMaxI32") => {
            *scalar_condition = left.cast_signed() > right.cast_signed();
            left.cast_signed().max(right.cast_signed()).cast_unsigned()
        }
        Some("SMaxU32") => {
            *scalar_condition = left > right;
            left.max(right)
        }
        Some("SCselectB32") => {
            if *scalar_condition {
                left
            } else {
                right
            }
        }
        Some("SAndB32") => {
            let value = left & right;
            *scalar_condition = value != 0;
            value
        }
        Some("SOrB32") => {
            let value = left | right;
            *scalar_condition = value != 0;
            value
        }
        Some("SXorB32") => {
            let value = left ^ right;
            *scalar_condition = value != 0;
            value
        }
        Some("SAndn2B32") => {
            let value = left & !right;
            *scalar_condition = value != 0;
            value
        }
        Some("SOrn2B32") => {
            let value = left | !right;
            *scalar_condition = value != 0;
            value
        }
        Some("SNandB32") => {
            let value = !(left & right);
            *scalar_condition = value != 0;
            value
        }
        Some("SNorB32") => {
            let value = !(left | right);
            *scalar_condition = value != 0;
            value
        }
        Some("SXnorB32") => {
            let value = !(left ^ right);
            *scalar_condition = value != 0;
            value
        }
        Some("SLshlB32") => {
            let value = left << (right & 31);
            *scalar_condition = value != 0;
            value
        }
        Some("SLshrB32") => {
            let value = left >> (right & 31);
            *scalar_condition = value != 0;
            value
        }
        Some("SAshrI32") => {
            let value = (left.cast_signed() >> (right & 31)).cast_unsigned();
            *scalar_condition = value != 0;
            value
        }
        Some("SBfmB32") => {
            let width = left & 31;
            let offset = right & 31;
            if width == 0 {
                0
            } else {
                (u32::MAX >> (32 - width)) << offset
            }
        }
        Some("SMulI32") => left
            .cast_signed()
            .wrapping_mul(right.cast_signed())
            .cast_unsigned(),
        Some("SBfeU32") => {
            let offset = right & 31;
            let width = ((right >> 16) & 0x7f).min(32 - offset);
            let value = if width == 0 {
                0
            } else {
                (left >> offset) & (u32::MAX >> (32 - width))
            };
            *scalar_condition = value != 0;
            value
        }
        Some("SBfeI32") => {
            let offset = right & 31;
            let width = ((right >> 16) & 0x7f).min(32 - offset);
            let value = if width == 0 {
                0
            } else {
                ((left << (32 - width - offset)).cast_signed() >> (32 - width)).cast_unsigned()
            };
            *scalar_condition = value != 0;
            value
        }
        Some("SAbsdiffI32") => {
            u32::try_from(i64::from(left.cast_signed()).abs_diff(i64::from(right.cast_signed())))
                .expect("i32 absolute difference fits u32")
        }
        Some("SLshl1AddU32" | "SLshl2AddU32" | "SLshl3AddU32" | "SLshl4AddU32") => {
            let shift = match instruction.opcode.known_name() {
                Some("SLshl1AddU32") => 1,
                Some("SLshl2AddU32") => 2,
                Some("SLshl3AddU32") => 3,
                _ => 4,
            };
            let wide = (u64::from(left) << shift) + u64::from(right);
            *scalar_condition = wide > u64::from(u32::MAX);
            low_u32(wide)
        }
        Some("SPackLlB32B16") => (left & 0xffff) | (right << 16),
        Some("SPackLhB32B16") => (left & 0xffff) | (right & 0xffff_0000),
        Some("SPackHhB32B16") => (left >> 16) | (right & 0xffff_0000),
        Some("SMulHiU32") => ((u64::from(left) * u64::from(right)) >> 32) as u32,
        Some("SMulHiI32") => low_u32(
            ((i64::from(left.cast_signed()) * i64::from(right.cast_signed())) >> 32)
                .cast_unsigned(),
        ),
        _ => {
            return Err(EvaluationError::UnsupportedScalarOpcode {
                pc: instruction.pc,
                opcode: instruction.opcode,
            });
        }
    };
    Ok(result)
}

fn execute_save_exec(
    instruction: &Instruction,
    registers: &mut [u32; SCALAR_REGISTER_COUNT],
    exec_mask: &mut u64,
    scalar_condition: &mut bool,
) -> Result<bool, EvaluationError> {
    if instruction
        .opcode
        .known_name()
        .is_some_and(|name| name.ends_with("SaveexecB32"))
    {
        let destination = scalar_destination(instruction)?;
        let source = evaluate_operand(
            instruction
                .sources
                .first()
                .ok_or_else(|| invalid(instruction, "saveexec source"))?,
            registers,
        )
        .ok_or_else(|| invalid(instruction, "saveexec source"))?;
        let old = low_u32(*exec_mask);
        let new = match instruction.opcode.known_name() {
            Some("SAndSaveexecB32") => old & source,
            Some("SOrSaveexecB32") => old | source,
            Some("SXorSaveexecB32") => old ^ source,
            Some("SAndn1SaveexecB32") => !source & old,
            Some("SAndn2SaveexecB32") => source & !old,
            Some("SOrn1SaveexecB32") => !source | old,
            Some("SOrn2SaveexecB32") => source | !old,
            Some("SNandSaveexecB32") => !(source & old),
            Some("SNorSaveexecB32") => !(source | old),
            Some("SXnorSaveexecB32") => !(old ^ source),
            _ => {
                return Err(EvaluationError::UnsupportedScalarOpcode {
                    pc: instruction.pc,
                    opcode: instruction.opcode,
                });
            }
        };
        registers[destination as usize] = old;
        *exec_mask = u64::from(new);
        registers[126] = new;
        registers[127] = 0;
        *scalar_condition = new != 0;
        return Ok(true);
    }
    if !matches!(
        instruction.opcode.known_name(),
        Some(
            "SAndSaveexecB64"
                | "SOrSaveexecB64"
                | "SXorSaveexecB64"
                | "SAndn2SaveexecB64"
                | "SOrn2SaveexecB64"
                | "SNandSaveexecB64"
                | "SNorSaveexecB64"
                | "SXnorSaveexecB64"
                | "SAndn1SaveexecB64"
                | "SOrn1SaveexecB64"
        )
    ) {
        return Ok(false);
    }
    let destination = scalar_destination(instruction)?;
    let source = evaluate_operand64(
        instruction
            .sources
            .first()
            .ok_or_else(|| invalid(instruction, "saveexec source"))?,
        registers,
        *exec_mask,
    )
    .ok_or_else(|| invalid(instruction, "saveexec source"))?;
    let old = *exec_mask;
    let new = match instruction.opcode.known_name() {
        Some("SAndSaveexecB64") => old & source,
        Some("SOrSaveexecB64") => old | source,
        Some("SXorSaveexecB64") => old ^ source,
        Some("SAndn1SaveexecB64") => !source & old,
        Some("SAndn2SaveexecB64") => source & !old,
        Some("SOrn1SaveexecB64") => !source | old,
        Some("SOrn2SaveexecB64") => source | !old,
        Some("SNandSaveexecB64") => !(source & old),
        Some("SNorSaveexecB64") => !(source | old),
        Some("SXnorSaveexecB64") => !(old ^ source),
        _ => {
            return Err(EvaluationError::UnsupportedScalarOpcode {
                pc: instruction.pc,
                opcode: instruction.opcode,
            });
        }
    } & RDNA_WAVE_MASK;
    write_scalar_pair(registers, destination, old, exec_mask);
    *exec_mask = new;
    write_scalar_pair(registers, 126, new, exec_mask);
    *scalar_condition = new != 0;
    Ok(true)
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn evaluate_operand(operand: &Operand, registers: &[u32; SCALAR_REGISTER_COUNT]) -> Option<u32> {
    match operand.kind {
        OperandKind::ScalarRegister => registers.get(operand.value as usize).copied(),
        OperandKind::LiteralConstant => Some(operand.value),
        OperandKind::EncodedConstant => decode_inline_constant(operand.value),
        OperandKind::VectorRegister => None,
    }
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn evaluate_wave_operand(
    operand: &Operand,
    registers: &[u32; SCALAR_REGISTER_COUNT],
    exec_mask: u64,
    scalar_condition: bool,
) -> Option<u32> {
    if operand.kind == OperandKind::EncodedConstant {
        match operand.value {
            251 => return Some(u32::from(registers[106] == 0 && registers[107] == 0)),
            252 => return Some(u32::from(exec_mask == 0)),
            253 => return Some(u32::from(scalar_condition)),
            _ => {}
        }
    }
    evaluate_operand(operand, registers)
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn evaluate_operand64(
    operand: &Operand,
    registers: &[u32; SCALAR_REGISTER_COUNT],
    exec_mask: u64,
) -> Option<u64> {
    if operand.kind == OperandKind::ScalarRegister && operand.value == 126 {
        return Some(exec_mask);
    }
    if operand.kind == OperandKind::ScalarRegister {
        let index = usize::try_from(operand.value).ok()?;
        let high = index.checked_add(1)?;
        return Some(u64::from(*registers.get(index)?) | (u64::from(*registers.get(high)?) << 32));
    }
    let low = evaluate_operand(operand, registers)?;
    Some(
        if operand.kind == OperandKind::EncodedConstant && (193..=208).contains(&operand.value) {
            u64::from(low) | (u64::from(u32::MAX) << 32)
        } else {
            u64::from(low)
        },
    )
}

fn write_scalar_pair(
    registers: &mut [u32; SCALAR_REGISTER_COUNT],
    destination: u32,
    mut value: u64,
    exec_mask: &mut u64,
) {
    let Ok(index) = usize::try_from(destination) else {
        return;
    };
    let Some(high_index) = index
        .checked_add(1)
        .filter(|index| *index < registers.len())
    else {
        return;
    };
    if destination == 126 {
        value &= RDNA_WAVE_MASK;
    }
    registers[index] = low_u32(value);
    registers[high_index] = low_u32(value >> 32);
    if destination == 126 {
        *exec_mask = value;
    }
}

fn execute_scalar_compare(
    instruction: &Instruction,
    registers: &[u32; SCALAR_REGISTER_COUNT],
    exec_mask: u64,
) -> Result<bool, EvaluationError> {
    let [left_operand, right_operand] = instruction.sources.as_slice() else {
        return Err(invalid(instruction, "scalar compare operands"));
    };
    let left = evaluate_operand(left_operand, registers)
        .ok_or_else(|| invalid(instruction, "scalar compare source 0"))?;
    let right = evaluate_operand(right_operand, registers)
        .ok_or_else(|| invalid(instruction, "scalar compare source 1"))?;
    let result = match instruction.opcode.known_name() {
        Some("SBitcmp0B32" | "SBitcmp1B32") => {
            let set = (left >> (right & 31)) & 1 != 0;
            if instruction.opcode.is("SBitcmp1B32") {
                set
            } else {
                !set
            }
        }
        Some("SBitcmp0B64" | "SBitcmp1B64") => {
            let wide = evaluate_operand64(left_operand, registers, exec_mask)
                .ok_or_else(|| invalid(instruction, "64-bit compare source"))?;
            let set = (wide >> (right & 63)) & 1 != 0;
            if instruction.opcode.is("SBitcmp1B64") {
                set
            } else {
                !set
            }
        }
        Some("SCmpEqI32") => left.cast_signed() == right.cast_signed(),
        Some("SCmpLgI32") => left.cast_signed() != right.cast_signed(),
        Some("SCmpGtI32") => left.cast_signed() > right.cast_signed(),
        Some("SCmpGeI32") => left.cast_signed() >= right.cast_signed(),
        Some("SCmpLtI32") => left.cast_signed() < right.cast_signed(),
        Some("SCmpLeI32") => left.cast_signed() <= right.cast_signed(),
        Some("SCmpEqU32") => left == right,
        Some("SCmpLgU32") => left != right,
        Some("SCmpGtU32") => left > right,
        Some("SCmpGeU32") => left >= right,
        Some("SCmpLtU32") => left < right,
        Some("SCmpLeU32") => left <= right,
        _ => {
            return Err(EvaluationError::UnsupportedScalarOpcode {
                pc: instruction.pc,
                opcode: instruction.opcode,
            });
        }
    };
    Ok(result)
}

fn execute_scalar_compare_k(
    instruction: &Instruction,
    registers: &[u32; SCALAR_REGISTER_COUNT],
) -> Result<bool, EvaluationError> {
    let destination = scalar_destination(instruction)? as usize;
    let left = registers[destination];
    let right = i32::from(
        low_u16(
            instruction
                .sources
                .first()
                .ok_or_else(|| invalid(instruction, "compare-k source"))?
                .value,
        )
        .cast_signed(),
    )
    .cast_unsigned();
    let result = match instruction.opcode.known_name() {
        Some("SCmpkEqI32") => left.cast_signed() == right.cast_signed(),
        Some("SCmpkLgI32") => left.cast_signed() != right.cast_signed(),
        Some("SCmpkGtI32") => left.cast_signed() > right.cast_signed(),
        Some("SCmpkGeI32") => left.cast_signed() >= right.cast_signed(),
        Some("SCmpkLtI32") => left.cast_signed() < right.cast_signed(),
        Some("SCmpkLeI32") => left.cast_signed() <= right.cast_signed(),
        Some("SCmpkEqU32") => left == right,
        Some("SCmpkLgU32") => left != right,
        Some("SCmpkGtU32") => left > right,
        Some("SCmpkGeU32") => left >= right,
        Some("SCmpkLtU32") => left < right,
        Some("SCmpkLeU32") => left <= right,
        _ => {
            return Err(EvaluationError::UnsupportedScalarOpcode {
                pc: instruction.pc,
                opcode: instruction.opcode,
            });
        }
    };
    Ok(result)
}

fn read_scalar_pair(registers: &[u32; SCALAR_REGISTER_COUNT], scalar_base: u32) -> Option<u64> {
    let index = usize::try_from(scalar_base).ok()?;
    Some(
        u64::from(*registers.get(index)?)
            | (u64::from(*registers.get(index.checked_add(1)?)?) << 32),
    )
}

fn decode_buffer_descriptor(
    registers: &[u32; SCALAR_REGISTER_COUNT],
    scalar_base: u32,
    strict_type: bool,
) -> Result<BufferDescriptor, ()> {
    let index = usize::try_from(scalar_base).map_err(|_| ())?;
    let descriptor_words = registers
        .get(index..index.checked_add(4).ok_or(())?)
        .ok_or(())?;
    let [word0, word1, word2, word3] = *descriptor_words else {
        return Err(());
    };
    if word0 == 0 && word1 == 0 && word2 == 0 && word3 == 0 {
        return Ok(BufferDescriptor {
            base_address: 0,
            stride: 0,
            swizzle_enabled: false,
            record_count: 0,
            size_bytes: 0,
            number_format: 0,
            data_format: 0,
        });
    }
    if word3 >> 30 != 0 {
        if strict_type {
            return Err(());
        }
        return Ok(BufferDescriptor {
            base_address: 0,
            stride: 0,
            swizzle_enabled: false,
            record_count: 0,
            size_bytes: 0,
            number_format: 0,
            data_format: 0,
        });
    }
    let base_address = u64::from(word0) | (u64::from(word1 & 0xffff) << 32);
    let stride = (word1 >> 16) & 0x3fff;
    let swizzle_enabled = word1 >> 31 != 0;
    let oob_select = (word3 >> 28) & 0x3;
    let unified_format = (word3 >> 12) & 0x7f;
    let (data_format, number_format) = decode_unified_format(unified_format).ok_or(())?;
    let size_bytes = if stride != 0 && (swizzle_enabled || oob_select != 3) {
        u64::from(stride) * u64::from(word2)
    } else {
        u64::from(word2)
    };
    Ok(BufferDescriptor {
        base_address,
        stride,
        swizzle_enabled,
        record_count: word2,
        size_bytes,
        number_format,
        data_format,
    })
}

fn is_vertex_fetch(
    instruction: &Instruction,
    control: BufferMemoryControl,
    descriptor: BufferDescriptor,
) -> bool {
    control.index_enabled
        && !control.offset_enabled
        && (1..=4).contains(&control.dword_count)
        && descriptor.base_address != 0
        && descriptor.stride != 0
        && (instruction.opcode.starts_with("BufferLoadFormat")
            || instruction.opcode.starts_with("TBufferLoadFormat"))
}

fn required_vertex_bytes(
    descriptor: BufferDescriptor,
    binding_offset: u32,
    component_count: u32,
    record_count: Option<u32>,
) -> u64 {
    let Some(record_count) = record_count
        .filter(|count| *count != 0)
        .map(|count| count.min(descriptor.record_count))
        .filter(|count| *count != 0)
    else {
        return descriptor.size_bytes;
    };
    let last_record = u64::from(record_count - 1).saturating_mul(u64::from(descriptor.stride));
    let element_bytes = u64::from(component_count.max(1)).saturating_mul(4);
    let record_span =
        u64::from(descriptor.stride).max(u64::from(binding_offset).saturating_add(element_bytes));
    descriptor
        .size_bytes
        .min(last_record.saturating_add(record_span))
}

fn copy_registers<const N: usize>(
    registers: &[u32; SCALAR_REGISTER_COUNT],
    start: u32,
    count: usize,
) -> Option<SmallVec<[u32; N]>>
where
    [u32; N]: smallvec::Array<Item = u32>,
{
    let start = usize::try_from(start).ok()?;
    let end = start.checked_add(count)?;
    Some(SmallVec::from_slice(registers.get(start..end)?))
}

fn uses_sampler(opcode: Opcode) -> bool {
    opcode.starts_with("ImageSample") || opcode.starts_with("ImageGather")
}

fn resolve_image_mip_level(state: &ShaderState, pc: u32, image: &ImageControl) -> Option<u32> {
    let register = image.address_register(if image.a16 { 1 } else { 2 });
    for instruction in state.program.instructions.iter().rev() {
        if instruction.pc >= pc {
            continue;
        }
        let [destination] = instruction.destinations.as_slice() else {
            continue;
        };
        if destination.kind != OperandKind::VectorRegister || destination.value != register {
            continue;
        }
        if instruction.opcode.is("VMovB32") {
            let [source] = instruction.sources.as_slice() else {
                return None;
            };
            let raw = match source.kind {
                OperandKind::LiteralConstant => source.value,
                OperandKind::EncodedConstant => decode_inline_constant(source.value)?,
                _ => return None,
            };
            return Some(if image.a16 { raw & 0xffff } else { raw });
        }
        return None;
    }
    None
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{ExportControl, Program};

    #[derive(Default)]
    struct TestMemory(BTreeMap<u64, u8>);

    impl TestMemory {
        fn write(&mut self, address: u64, bytes: &[u8]) {
            for (offset, byte) in bytes.iter().copied().enumerate() {
                self.0.insert(address + offset as u64, byte);
            }
        }
    }

    impl GuestMemoryReader for TestMemory {
        fn read(&self, address: u64, output: &mut [u8]) -> bool {
            for (offset, destination) in output.iter_mut().enumerate() {
                let Some(byte) = self.0.get(&(address + offset as u64)) else {
                    return false;
                };
                *destination = *byte;
            }
            true
        }
    }

    fn instruction(
        pc: u32,
        encoding: ShaderEncoding,
        opcode: &'static str,
        sources: Vec<Operand>,
        destinations: Vec<Operand>,
        control: Option<InstructionControl>,
    ) -> Instruction {
        Instruction {
            pc,
            encoding,
            opcode: Opcode::from_static(opcode),
            words: vec![0],
            sources,
            destinations,
            control,
        }
    }

    fn end(pc: u32) -> Instruction {
        instruction(
            pc,
            ShaderEncoding::Sopp,
            "SEndpgm",
            Vec::new(),
            Vec::new(),
            None,
        )
    }

    fn state(instructions: Vec<Instruction>, user_data: Vec<u32>) -> ShaderState {
        ShaderState {
            program: Arc::new(Program::new(0x1000, instructions)),
            user_data: user_data.into(),
            compute_system_registers: None,
            user_data_scalar_register_base: 0,
        }
    }

    #[test]
    fn initializes_merged_export_wave_info_before_user_sgprs() {
        let shader = ShaderState {
            program: Arc::new(Program::new(
                0x1000,
                vec![
                    instruction(
                        0,
                        ShaderEncoding::Sop1,
                        "SMovB32",
                        vec![Operand::scalar(3)],
                        vec![Operand::scalar(4)],
                        None,
                    ),
                    instruction(
                        4,
                        ShaderEncoding::Sop1,
                        "SMovB32",
                        vec![Operand::scalar(8)],
                        vec![Operand::scalar(5)],
                        None,
                    ),
                    end(8),
                ],
            )),
            user_data: vec![0x1234_5678].into(),
            compute_system_registers: None,
            user_data_scalar_register_base: 8,
        };

        let evaluation = evaluate_shader(&TestMemory::default(), &shader, Default::default())
            .expect("merged export evaluation");

        assert_eq!(evaluation.initial_scalar_registers[3], 0x4040);
        assert_eq!(evaluation.scalar_registers[4], 0x4040);
        assert_eq!(evaluation.initial_scalar_registers[8], 0x1234_5678);
        assert_eq!(evaluation.scalar_registers[5], 0x1234_5678);
    }

    #[test]
    fn sizes_structured_buffer_descriptors_in_records() {
        let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
        registers[..4].copy_from_slice(&[0x2000, 16 << 16, 32, 0]);

        let descriptor = decode_buffer_descriptor(&registers, 0, true).expect("descriptor");

        assert_eq!(descriptor.base_address, 0x2000);
        assert_eq!(descriptor.stride, 16);
        assert_eq!(descriptor.record_count, 32);
        assert_eq!(descriptor.size_bytes, 512);
    }

    #[test]
    fn sizes_raw_buffer_descriptors_in_bytes() {
        let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
        registers[..4].copy_from_slice(&[0x2000, 16 << 16, 512, 3 << 28]);

        let descriptor = decode_buffer_descriptor(&registers, 0, true).expect("descriptor");

        assert_eq!(descriptor.stride, 16);
        assert_eq!(descriptor.record_count, 512);
        assert_eq!(descriptor.size_bytes, 512);
    }

    #[test]
    fn sizes_swizzled_buffers_in_records_even_with_raw_oob_selection() {
        let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
        registers[..4].copy_from_slice(&[0x2000, (16 << 16) | (1 << 31), 32, 3 << 28]);

        let descriptor = decode_buffer_descriptor(&registers, 0, true).expect("descriptor");

        assert!(descriptor.swizzle_enabled);
        assert_eq!(descriptor.size_bytes, 512);
    }

    #[test]
    fn bounds_runtime_vertex_buffers_to_the_draws_referenced_records() {
        let mut memory = TestMemory::default();
        memory.write(0x2000, &[0; 96]);
        let selectors = 4 | (5 << 3) | (6 << 6) | (7 << 9);
        let shader = state(
            vec![
                instruction(
                    0,
                    ShaderEncoding::Mubuf,
                    "BufferLoadFormatX",
                    vec![
                        Operand::vector(5),
                        Operand::scalar(0),
                        Operand::source(128, None),
                    ],
                    vec![Operand::vector(7)],
                    Some(InstructionControl::BufferMemory(BufferMemoryControl {
                        dword_count: 1,
                        vector_address: 5,
                        vector_data: 7,
                        scalar_resource: 0,
                        instruction_format: None,
                        offset_bytes: 0,
                        index_enabled: true,
                        offset_enabled: false,
                        dlc: false,
                        lds: false,
                        glc: false,
                        slc: false,
                        tfe: false,
                    })),
                ),
                end(8),
            ],
            vec![0x2000, 24 << 16, 0xaaaa, selectors | (60 << 12) | (3 << 28)],
        );

        let evaluation = evaluate_shader(
            &memory,
            &shader,
            EvaluationOptions {
                required_vertex_record_count: Some(4),
                ..EvaluationOptions::default()
            },
        )
        .expect("bounded runtime vertex fetch evaluation");

        assert!(evaluation.vertex_inputs.is_empty());
        assert_eq!(evaluation.global_memory_bindings.len(), 1);
        assert_eq!(evaluation.global_memory_bindings[0].data.len(), 96);
    }

    #[test]
    fn evaluates_scalar_arithmetic_and_taken_control_flow() {
        let mut branch = instruction(
            12,
            ShaderEncoding::Sopp,
            "SCbranchScc1",
            vec![Operand {
                kind: OperandKind::EncodedConstant,
                value: 1,
            }],
            Vec::new(),
            None,
        );
        branch.words[0] = 1;
        let shader = state(
            vec![
                instruction(
                    0,
                    ShaderEncoding::Sopk,
                    "SMovkI32",
                    vec![Operand {
                        kind: OperandKind::EncodedConstant,
                        value: 5,
                    }],
                    vec![Operand::scalar(0)],
                    None,
                ),
                instruction(
                    4,
                    ShaderEncoding::Sopk,
                    "SAddkI32",
                    vec![Operand {
                        kind: OperandKind::EncodedConstant,
                        value: u32::from(u16::MAX),
                    }],
                    vec![Operand::scalar(0)],
                    None,
                ),
                instruction(
                    8,
                    ShaderEncoding::Sopk,
                    "SCmpkEqI32",
                    vec![Operand {
                        kind: OperandKind::EncodedConstant,
                        value: 4,
                    }],
                    vec![Operand::scalar(0)],
                    None,
                ),
                branch,
                instruction(
                    16,
                    ShaderEncoding::Sopk,
                    "SMovkI32",
                    vec![Operand {
                        kind: OperandKind::EncodedConstant,
                        value: 7,
                    }],
                    vec![Operand::scalar(1)],
                    None,
                ),
                instruction(
                    20,
                    ShaderEncoding::Sopk,
                    "SMovkI32",
                    vec![Operand {
                        kind: OperandKind::EncodedConstant,
                        value: 9,
                    }],
                    vec![Operand::scalar(1)],
                    None,
                ),
                end(24),
            ],
            Vec::new(),
        );
        let evaluation = evaluate_shader(
            &TestMemory::default(),
            &shader,
            EvaluationOptions::default(),
        )
        .expect("scalar evaluation");
        assert_eq!(evaluation.scalar_registers[0], 4);
        assert_eq!(evaluation.scalar_registers[1], 9);
    }

    #[test]
    fn bounds_vector_controlled_backward_loops_during_scalar_discovery() {
        let mut branch = instruction(
            4,
            ShaderEncoding::Sopp,
            "SCbranchExecnz",
            Vec::new(),
            Vec::new(),
            None,
        );
        branch.words[0] = u32::from((-2_i16).cast_unsigned());
        let shader = state(
            vec![
                instruction(
                    0,
                    ShaderEncoding::Vop1,
                    "VMovB32",
                    vec![Operand::vector(0)],
                    vec![Operand::vector(1)],
                    None,
                ),
                branch,
                end(8),
            ],
            Vec::new(),
        );

        let evaluation = evaluate_shader(
            &TestMemory::default(),
            &shader,
            EvaluationOptions::default(),
        )
        .expect("vector-controlled runtime loop must not exhaust static discovery");
        assert!(evaluation.global_memory_bindings.is_empty());
    }

    #[test]
    fn exits_an_unconditional_back_edge_after_discovering_the_vector_loop_once() {
        let mut exit = instruction(
            4,
            ShaderEncoding::Sopp,
            "SCbranchExecz",
            Vec::new(),
            Vec::new(),
            None,
        );
        exit.words[0] = 2;
        let mut back_edge = instruction(
            12,
            ShaderEncoding::Sopp,
            "SBranch",
            Vec::new(),
            Vec::new(),
            None,
        );
        back_edge.words[0] = u32::from((-3_i16).cast_unsigned());
        let shader = state(
            vec![
                instruction(
                    0,
                    ShaderEncoding::Vop1,
                    "VMovB32",
                    vec![Operand::vector(0)],
                    vec![Operand::vector(1)],
                    None,
                ),
                exit,
                instruction(
                    8,
                    ShaderEncoding::Vop1,
                    "VMovB32",
                    vec![Operand::vector(1)],
                    vec![Operand::vector(2)],
                    None,
                ),
                back_edge,
                end(16),
            ],
            Vec::new(),
        );

        evaluate_shader(
            &TestMemory::default(),
            &shader,
            EvaluationOptions::default(),
        )
        .expect("unconditional vector loop back edge must remain bounded");
    }

    #[test]
    fn executes_scalar_loads_and_records_the_backing_binding() {
        let mut memory = TestMemory::default();
        memory.write(0x2000, &0x1122_3344_u32.to_le_bytes());
        memory.write(0x2004, &0xaabb_ccdd_u32.to_le_bytes());
        let shader = state(
            vec![
                instruction(
                    0,
                    ShaderEncoding::Smem,
                    "SLoadDwordx2",
                    vec![Operand::scalar(0)],
                    vec![Operand::scalar(4), Operand::scalar(5)],
                    Some(InstructionControl::ScalarMemory(ScalarMemoryControl {
                        destination_count: 2,
                        immediate_offset_bytes: 0,
                        dynamic_offset_register: None,
                    })),
                ),
                end(8),
            ],
            vec![0x2000, 0],
        );
        let evaluation = evaluate_shader(&memory, &shader, EvaluationOptions::default())
            .expect("scalar load evaluation");
        assert_eq!(evaluation.scalar_registers[4], 0x1122_3344);
        assert_eq!(evaluation.scalar_registers[5], 0xaabb_ccdd);
        assert_eq!(
            evaluation.scalar_memory_reads,
            vec![(0x2000, 0x1122_3344), (0x2004, 0xaabb_ccdd)]
        );
        assert_eq!(evaluation.global_memory_bindings.len(), 1);
        assert_eq!(evaluation.global_memory_bindings[0].data.len(), 8);
    }

    #[test]
    fn grows_shared_scalar_binding_to_the_largest_observed_load() {
        let mut memory = TestMemory::default();
        memory.write(0x2000, &[0; 68]);
        let scalar_load = |pc, immediate_offset_bytes, destination| {
            instruction(
                pc,
                ShaderEncoding::Smem,
                "SLoadDword",
                vec![Operand::scalar(0)],
                vec![Operand::scalar(destination)],
                Some(InstructionControl::ScalarMemory(ScalarMemoryControl {
                    destination_count: 1,
                    immediate_offset_bytes,
                    dynamic_offset_register: None,
                })),
            )
        };
        let shader = state(
            vec![scalar_load(0, 0, 4), scalar_load(8, 64, 5), end(16)],
            vec![0x2000, 0],
        );

        let evaluation = evaluate_shader(&memory, &shader, EvaluationOptions::default())
            .expect("scalar binding growth evaluation");

        assert_eq!(evaluation.global_memory_bindings.len(), 1);
        assert_eq!(evaluation.global_memory_bindings[0].data.len(), 68);
        assert_eq!(
            evaluation.global_memory_bindings[0]
                .instruction_pcs
                .as_slice(),
            [0, 8]
        );
    }

    #[test]
    fn resolves_image_descriptors_without_executing_vector_math() {
        let image = ImageControl {
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
        let shader = state(
            vec![
                instruction(
                    0,
                    ShaderEncoding::Mimg,
                    "ImageSample",
                    Vec::new(),
                    Vec::new(),
                    Some(InstructionControl::Image(image.clone())),
                ),
                instruction(
                    8,
                    ShaderEncoding::Exp,
                    "Exp",
                    Vec::new(),
                    Vec::new(),
                    Some(InstructionControl::Export(ExportControl {
                        target: 0,
                        enable_mask: 0xf,
                        compressed: false,
                        done: true,
                        valid_mask: false,
                    })),
                ),
                end(16),
            ],
            (1..=12).collect(),
        );
        let evaluation = evaluate_shader(
            &TestMemory::default(),
            &shader,
            EvaluationOptions::default(),
        )
        .expect("image evaluation");
        assert_eq!(evaluation.image_bindings.len(), 1);
        assert_eq!(
            evaluation.image_bindings[0].resource_descriptor.as_slice(),
            (1..=8).collect::<Vec<_>>().as_slice()
        );
        assert_eq!(
            evaluation.image_bindings[0].sampler_descriptor.as_slice(),
            (9..=12).collect::<Vec<_>>().as_slice()
        );
    }
}
