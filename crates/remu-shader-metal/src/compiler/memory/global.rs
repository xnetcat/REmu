// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Scalar, global and scratch access lowering: SMEM constant loads and the
//! FLAT/GLOBAL/SCRATCH address space.

use remu_shader::{Instruction, InstructionControl, OperandKind};

use crate::{Compiler, MetalShaderStage, format_u32};

use super::buffer::{raw_load_shape, raw_store_shape};

impl Compiler<'_> {
    pub(crate) fn emit_scalar_memory(&mut self, instruction: &Instruction) -> Result<(), String> {
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

    pub(crate) fn emit_global_memory(&mut self, instruction: &Instruction) -> Result<(), String> {
        let Some(InstructionControl::GlobalMemory(control)) = instruction.control.as_ref() else {
            return Err("missing global-memory control".to_owned());
        };
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw global opcode".to_owned())?;
        if name.starts_with("Scratch") {
            return self.emit_scratch_memory(instruction, control, name);
        }
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

    /// Lowers RDNA2 SCRATCH into Metal's lane-private `thread` address space.
    ///
    /// The guest offset is 32-bit and comes from either one VGPR or one SGPR; unlike GLOBAL it is
    /// never a 64-bit guest pointer.  The hardware swizzles lanes into the per-wave temp ring.
    /// Each Metal invocation already owns one private array, so using the unswizzled byte offset
    /// within that array is the equivalent representation.
    fn emit_scratch_memory(
        &mut self,
        instruction: &Instruction,
        control: &remu_shader::GlobalMemoryControl,
        name: &str,
    ) -> Result<(), String> {
        if self.options.scratch_size_bytes == 0 {
            // The temp ring is sized by a different register per pipe: compute dispatches program
            // `COMPUTE_TMPRING_SIZE` (`ShComputeScratchRingSize`) while every graphics stage shares
            // `SPI_TMPRING_SIZE` (`CxGraphicsScratchRingSize`). Naming the compute register on a
            // graphics stage sent the reader looking for state the draw never programs.
            let register = if self.options.stage == MetalShaderStage::Compute {
                "COMPUTE_TMPRING_SIZE"
            } else {
                "SPI_TMPRING_SIZE"
            };
            return Err(format!(
                "SCRATCH instruction has no {register} lane allocation"
            ));
        }
        if control.lds || name.contains("Atomic") {
            return Err(format!(
                "unsupported SCRATCH transfer shape lds={} opcode={name}",
                control.lds
            ));
        }
        let register_offset = instruction
            .sources
            .iter()
            .find(|operand| operand.kind == OperandKind::ScalarRegister)
            .map_or_else(
                || format!("v[{}]", control.vector_address),
                |operand| format!("s[{}]", operand.value),
            );
        let address = self.temp(
            "uint",
            &format!(
                "{register_offset} + {}",
                format_u32(control.offset_bytes.cast_unsigned())
            ),
        );
        let scratch_length = format_u32(self.options.scratch_size_bytes);

        if name.contains("Load") {
            let (width, signed, d16, high) = raw_load_shape(name)?;
            for component in 0..control.dword_count {
                let component_address = if component == 0 {
                    address.clone()
                } else {
                    format!("{address} + {}u", component * 4)
                };
                let loaded = match width {
                    1 => format!("(uint)remu_scratch[{component_address}]"),
                    2 => format!(
                        "(uint)remu_scratch[{component_address}] | ((uint)remu_scratch[{component_address} + 1u] << 8u)"
                    ),
                    4 => format!(
                        "(uint)remu_scratch[{component_address}] | ((uint)remu_scratch[{component_address} + 1u] << 8u) | ((uint)remu_scratch[{component_address} + 2u] << 16u) | ((uint)remu_scratch[{component_address} + 3u] << 24u)"
                    ),
                    _ => unreachable!("raw scratch load widths are fixed"),
                };
                let raw = self.temp(
                    "uint",
                    &format!(
                        "remu_range_valid({component_address}, {width}u, {scratch_length}) ? ({loaded}) : 0u"
                    ),
                );
                let converted = if signed && width == 1 {
                    format!("({raw} & 0x80u) != 0u ? ({raw} | 0xFFFFFF00u) : {raw}")
                } else if signed && width == 2 {
                    format!("({raw} & 0x8000u) != 0u ? ({raw} | 0xFFFF0000u) : {raw}")
                } else {
                    raw
                };
                let register = control.vector_destination + component;
                let value = if d16 && high {
                    format!("(v[{register}] & 0xFFFFu) | (({converted} & 0xFFFFu) << 16)")
                } else if d16 {
                    format!("(v[{register}] & 0xFFFF0000u) | ({converted} & 0xFFFFu)")
                } else {
                    converted
                };
                self.store_vector(register, &value);
            }
            return Ok(());
        }

        if name.contains("Store") {
            let (width, high) = raw_store_shape(name)?;
            for component in 0..control.dword_count {
                let component_address = if component == 0 {
                    address.clone()
                } else {
                    format!("{address} + {}u", component * 4)
                };
                let source = format!("v[{}]", control.vector_data + component);
                let value = if high {
                    format!("{source} >> 16u")
                } else {
                    source
                };
                self.line(
                    3,
                    &format!(
                        "if (exec && remu_range_valid({component_address}, {width}u, {scratch_length})) {{"
                    ),
                );
                for byte in 0..width {
                    self.line(
                        4,
                        &format!(
                            "remu_scratch[{component_address} + {byte}u] = (uchar)({value} >> {}u);",
                            byte * 8
                        ),
                    );
                }
                self.line(3, "}");
            }
            return Ok(());
        }

        Err(format!("unsupported SCRATCH opcode {name}"))
    }
}
