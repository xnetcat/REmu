// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Scalar, global and scratch access lowering: SMEM constant loads and the
//! FLAT/GLOBAL/SCRATCH address space.

use remu_shader::{Instruction, InstructionControl, OperandKind};

use crate::{Compiler, format_u32};

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
}
