// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Resource discovery: scalar loads, buffer/global/image descriptor resolution and binding records.

use smallvec::SmallVec;

use crate::{
    BufferMemoryControl, GlobalMemoryBinding, GlobalMemoryControl, GuestMemoryReader, ImageBinding,
    ImageControl, Instruction, InstructionControl, Opcode, Operand, OperandKind,
    ScalarMemoryControl, ShaderState, VertexInputBinding, decode_inline_constant,
    decode_unified_format,
};

use super::scalar_alu::{copy_registers, evaluate_operand, read_scalar_pair};
use super::{
    EvaluationError, Evaluator, IMAGE_DESCRIPTOR_DWORDS, SAMPLER_DESCRIPTOR_DWORDS,
    SCALAR_REGISTER_COUNT, SCALAR_REGISTER_COUNT_U32, invalid,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BufferDescriptor {
    pub(crate) base_address: u64,
    pub(crate) stride: u32,
    pub(crate) swizzle_enabled: bool,
    pub(crate) record_count: u32,
    pub(crate) size_bytes: u64,
    pub(crate) number_format: u32,
    pub(crate) data_format: u32,
}

#[derive(Clone, Copy)]
pub(crate) struct BindingRequest {
    pub(crate) scalar_address: u32,
    pub(crate) base_address: u64,
    pub(crate) requested_bytes: u64,
    pub(crate) writable: bool,
    pub(crate) allow_shorter: bool,
}

pub(crate) fn buffer_descriptor_words(
    registers: &[u32; SCALAR_REGISTER_COUNT],
    scalar_base: u32,
) -> Option<[u32; 4]> {
    let index = usize::try_from(scalar_base).ok()?;
    registers.get(index..index.checked_add(4)?)?.try_into().ok()
}

pub(crate) fn decode_buffer_descriptor(
    registers: &[u32; SCALAR_REGISTER_COUNT],
    scalar_base: u32,
) -> Result<BufferDescriptor, ()> {
    let [word0, word1, word2, word3] = buffer_descriptor_words(registers, scalar_base).ok_or(())?;
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
    // GFX10 repurposes the legacy TYPE field at the top of word 3. Bit 31 is
    // RESOURCE_LEVEL and is set on valid RDNA2 buffer descriptors. Interpreting those two bits as
    // the pre-GFX10 resource type rejects real Gen5 descriptors before their base address can be
    // evaluated.
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

pub(crate) fn is_vertex_fetch(
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

pub(crate) fn required_vertex_bytes(
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

pub(crate) fn uses_sampler(opcode: Opcode) -> bool {
    opcode.starts_with("ImageSample") || opcode.starts_with("ImageGather")
}

pub(crate) fn resolve_image_mip_level(
    state: &ShaderState,
    pc: u32,
    image: &ImageControl,
) -> Option<u32> {
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
impl<M: GuestMemoryReader> Evaluator<'_, M> {
    pub(crate) fn bind_statically_inactive_resources(&mut self) -> Result<(), EvaluationError> {
        let mut instruction_pcs = SmallVec::<[u32; 8]>::new();
        let mut scalar_address = 0;
        let mut writable = false;
        for instruction in &self.state.program.instructions {
            let candidate_scalar_address = match instruction.control.as_ref() {
                Some(InstructionControl::ScalarMemory(_)) => instruction
                    .sources
                    .first()
                    .filter(|source| source.kind == OperandKind::ScalarRegister)
                    .map(|source| source.value),
                Some(InstructionControl::BufferMemory(control)) => Some(control.scalar_resource),
                Some(InstructionControl::GlobalMemory(control)) => control.scalar_address,
                _ => None,
            };
            let Some(candidate_scalar_address) = candidate_scalar_address else {
                continue;
            };
            if self
                .global_bindings
                .iter()
                .any(|binding| binding.instruction_pcs.contains(&instruction.pc))
            {
                continue;
            }
            if instruction_pcs.is_empty() {
                scalar_address = candidate_scalar_address;
            }
            instruction_pcs.push(instruction.pc);
            writable |=
                instruction.opcode.contains("Store") || instruction.opcode.contains("Atomic");
        }
        if instruction_pcs.is_empty() {
            self.bind_statically_inactive_images()?;
            return Ok(());
        }
        // Scalar control flow is evaluated exactly for the current descriptor and dependency
        // snapshot, but Metal must still type-check every basic block in the shader. Give memory
        // instructions on statically inactive scalar paths one bounds-checked zero buffer. If a
        // dependency changes and activates the path, evaluation creates a new real binding and a
        // different pipeline key before that dispatch is encoded.
        self.global_bindings.push(GlobalMemoryBinding {
            scalar_address,
            base_address: 0,
            instruction_pcs,
            byte_count: 4,
            data: vec![0; 4],
            writable,
            write_back_to_guest: false,
        });
        self.bind_statically_inactive_images()?;
        Ok(())
    }

    pub(crate) fn bind_statically_inactive_images(&mut self) -> Result<(), EvaluationError> {
        let missing = self
            .state
            .program
            .instructions
            .iter()
            .filter(|instruction| {
                matches!(
                    instruction.control.as_ref(),
                    Some(InstructionControl::Image(_))
                ) && !self
                    .image_bindings
                    .iter()
                    .any(|binding| binding.pc == instruction.pc)
            })
            .cloned()
            .collect::<Vec<_>>();
        let initial_registers = self.initial_registers;
        for instruction in missing {
            let Some(InstructionControl::Image(control)) = instruction.control.clone() else {
                continue;
            };
            // Metal type-checks every scalar-control-flow block even when exact evaluation proves
            // one inactive for this draw. Resource descriptors are shader inputs, so preserve the
            // initial descriptor for those blocks instead of omitting the texture parameter and
            // making otherwise valid MSL fail to lower. A later scalar-path activation still
            // replaces this snapshot with the dynamically evaluated descriptor.
            self.resolve_image(&instruction, &control, &initial_registers, true)?;
        }
        Ok(())
    }

    pub(crate) fn execute_scalar_load(
        &mut self,
        instruction: &Instruction,
        control: ScalarMemoryControl,
        registers: &mut [u32; SCALAR_REGISTER_COUNT],
    ) -> Result<(), EvaluationError> {
        let scalar_base = match instruction.sources.first() {
            Some(Operand {
                kind: OperandKind::ScalarRegister,
                value,
            }) if *value < SCALAR_REGISTER_COUNT_U32 - 1 => *value,
            _ => return Err(invalid(instruction, "scalar-memory base register pair")),
        };
        let is_buffer = instruction.opcode.starts_with("SBufferLoad");
        let descriptor = is_buffer.then(|| decode_buffer_descriptor(registers, scalar_base));
        let descriptor = descriptor.transpose().map_err(|()| {
            let [words0, words1, words2, words3] =
                buffer_descriptor_words(registers, scalar_base).unwrap_or_default();
            EvaluationError::InvalidBufferDescriptor {
                pc: instruction.pc,
                register: scalar_base,
                words0,
                words1,
                words2,
                words3,
            }
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
        let mut unbound =
            base_address == 0 || descriptor.is_some_and(|descriptor| descriptor.size_bytes == 0);

        let binding_size = if unbound {
            4
        } else if let Some(descriptor) = descriptor {
            descriptor.size_bytes
        } else {
            byte_offset
                .checked_add(
                    u64::try_from(instruction.destinations.len().max(1) * 4)
                        .map_err(|_| EvaluationError::BindingSizeOverflow { pc: instruction.pc })?,
                )
                .ok_or(EvaluationError::BindingSizeOverflow { pc: instruction.pc })?
        };
        let binding_result = self.record_global_binding(
            instruction,
            BindingRequest {
                scalar_address: scalar_base,
                base_address,
                requested_bytes: binding_size,
                writable: false,
                // A buffer descriptor's extent is the hardware OOB limit, not a promise that
                // every page in that virtual range is resident. The scalar load below validates
                // the exact components it touches; retain the readable prefix needed by Metal
                // instead of rejecting a valid sparsely backed descriptor up front.
                allow_shorter: true,
            },
        );
        if let Err(error) = binding_result {
            if !matches!(error, EvaluationError::MemoryFault { .. }) {
                return Err(error);
            }
            // Resource-table pointers can transiently contain stale sentinels for a material
            // branch that the vector shader will not execute. Match the permissive console
            // resource behavior used by SharpEmu: keep translating, feed zero to the scalar
            // destinations, and bind a bounded zero buffer for Metal.
            unbound = true;
            self.record_global_binding(
                instruction,
                BindingRequest {
                    scalar_address: scalar_base,
                    base_address: 0,
                    requested_bytes: 4,
                    writable: false,
                    allow_shorter: false,
                },
            )?;
        }

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
                // Scalar resource tables can name allocations that are not resident for the
                // active material permutation. Console drivers return zero for these speculative
                // loads. SharpEmu applies the same compatibility rule; rejecting the complete
                // compute submission here turns a harmless unused resource into a guest fatal.
                for destination in &instruction.destinations {
                    registers[destination.value as usize] = 0;
                }
                return Ok(());
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
                registers[destination.value as usize] = 0;
                continue;
            }
            let component_address = address.wrapping_add((index as u64) * 4);
            let mut bytes = [0_u8; 4];
            if !self.memory.read(component_address, &mut bytes) {
                registers[destination.value as usize] = 0;
                continue;
            }
            let value = u32::from_le_bytes(bytes);
            self.scalar_memory_reads.insert(component_address, value);
            registers[destination.value as usize] = value;
        }
        Ok(())
    }

    pub(crate) fn resolve_global_memory(
        &mut self,
        instruction: &Instruction,
        control: GlobalMemoryControl,
        registers: &[u32; SCALAR_REGISTER_COUNT],
        _supplemental: bool,
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
        if base_address == 0 {
            // Null resources are legal placeholders in shaders shared across material variants.
            // The vector control flow that guards their use is not available to this scalar
            // evaluator. Preserve the pass with a bounded zero resource instead of rejecting the
            // complete shader during resource discovery.
            return self.record_global_binding(
                instruction,
                BindingRequest {
                    scalar_address,
                    base_address: 0,
                    requested_bytes: 4,
                    writable: false,
                    allow_shorter: false,
                },
            );
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
        self.record_global_binding_or_readonly_zero(
            instruction,
            BindingRequest {
                scalar_address,
                base_address: binding_base,
                requested_bytes: (self.options.maximum_binding_bytes as u64)
                    .saturating_add(prefix_bytes),
                writable,
                allow_shorter: true,
            },
        )
    }

    pub(crate) fn resolve_buffer_memory(
        &mut self,
        instruction: &Instruction,
        control: BufferMemoryControl,
        registers: &[u32; SCALAR_REGISTER_COUNT],
        _supplemental: bool,
    ) -> Result<(), EvaluationError> {
        let descriptor =
            decode_buffer_descriptor(registers, control.scalar_resource).map_err(|()| {
                let [words0, words1, words2, words3] =
                    buffer_descriptor_words(registers, control.scalar_resource).unwrap_or_default();
                EvaluationError::InvalidBufferDescriptor {
                    pc: instruction.pc,
                    register: control.scalar_resource,
                    words0,
                    words1,
                    words2,
                    words3,
                }
            })?;
        let writable =
            instruction.opcode.contains("Store") || instruction.opcode.contains("Atomic");
        if descriptor.base_address == 0 {
            return self.record_global_binding(
                instruction,
                BindingRequest {
                    scalar_address: control.scalar_resource,
                    base_address: 0,
                    requested_bytes: 4,
                    // Never attempt to write a placeholder back through address zero.
                    writable: false,
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
                self.record_global_binding_or_readonly_zero(
                    instruction,
                    BindingRequest {
                        scalar_address: control.scalar_resource,
                        base_address: binding_base,
                        requested_bytes: desired_size.saturating_add(prefix_bytes),
                        writable,
                        allow_shorter: false,
                    },
                )?;
            }
            return Ok(());
        }
        let binding_base = descriptor.base_address & !3;
        let prefix_bytes = descriptor.base_address - binding_base;
        self.record_global_binding_or_readonly_zero(
            instruction,
            BindingRequest {
                scalar_address: control.scalar_resource,
                base_address: binding_base,
                requested_bytes: descriptor.size_bytes.saturating_add(prefix_bytes),
                writable,
                allow_shorter: false,
            },
        )
    }

    pub(crate) fn resolve_image(
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

    pub(crate) fn record_global_binding(
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

    pub(crate) fn record_global_binding_or_readonly_zero(
        &mut self,
        instruction: &Instruction,
        request: BindingRequest,
    ) -> Result<(), EvaluationError> {
        match self.record_global_binding(instruction, request) {
            Ok(()) => Ok(()),
            Err(EvaluationError::MemoryFault { .. }) if !request.writable => {
                // Material permutations can retain an unreadable sentinel descriptor for a
                // vector-memory path whose lanes are inactive or out of bounds. Metal still
                // needs a binding for the instruction to type-check. Its generated load helpers
                // bounds-check against this four-byte placeholder and return zero, matching the
                // scalar-resource compatibility path without making stores silently disappear.
                self.record_global_binding(
                    instruction,
                    BindingRequest {
                        scalar_address: request.scalar_address,
                        base_address: 0,
                        requested_bytes: 4,
                        writable: false,
                        allow_shorter: false,
                    },
                )
            }
            Err(error) => Err(error),
        }
    }

    pub(crate) fn read_binding_request(
        &self,
        pc: u32,
        request: BindingRequest,
    ) -> Result<(usize, Vec<u8>), EvaluationError> {
        let data = if request.base_address == 0 {
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

    pub(crate) fn read_exact_binding(
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

    pub(crate) fn read_adaptive_binding(
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
