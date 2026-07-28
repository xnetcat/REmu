// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Wide and carry-producing vector lowering: 64-bit multiply-add and shifts,
//! and the add/sub opcodes that write a carry-out lane mask.

use remu_shader::{Instruction, InstructionControl};

use crate::compiler::vector::destination_vector;
use crate::{Compiler, SCALAR_REGISTER_COUNT, VCC_LO, VECTOR_REGISTER_COUNT};

impl Compiler<'_> {
    pub(crate) fn emit_mad_u64_u32(&mut self, instruction: &Instruction) -> Result<(), String> {
        let destination = destination_vector(instruction)?;
        if destination + 1 >= VECTOR_REGISTER_COUNT {
            return Err("V_MAD_U64_U32 destination pair exceeds the VGPR file".to_owned());
        }
        let Some(InstructionControl::Vop3(control)) = instruction.control else {
            return Err("V_MAD_U64_U32 requires VOP3B control".to_owned());
        };
        let scalar_destination = control
            .scalar_destination
            .ok_or_else(|| "V_MAD_U64_U32 requires an SDST carry mask".to_owned())?;
        if scalar_destination + 1 >= SCALAR_REGISTER_COUNT {
            return Err("V_MAD_U64_U32 SDST pair exceeds the SGPR file".to_owned());
        }
        if control.absolute_mask != 0
            || control.negate_mask != 0
            || control.output_modifier != 0
            || control.clamp
            || control.operand_select != 0
        {
            return Err("V_MAD_U64_U32 uses reserved VOP3B modifier bits".to_owned());
        }

        let first = self.temp("uint", &self.raw_source(instruction, 0)?);
        let second = self.temp("uint", &self.raw_source(instruction, 1)?);
        let addend = self.temp("ulong", &self.raw_source64(instruction, 2)?);
        let product = self.temp("ulong", &format!("(ulong){first} * (ulong){second}"));
        let result = self.temp("ulong", &format!("{product} + {addend}"));
        let carry = self.temp("bool", &format!("{result} < {product}"));
        self.store_lane_mask(scalar_destination, &format!("exec && {carry}"));
        self.store_vector(destination, &format!("(uint){result}"));
        self.store_vector(destination + 1, &format!("(uint)({result} >> 32)"));
        Ok(())
    }

    pub(crate) fn emit_vector_shift64(
        &mut self,
        instruction: &Instruction,
        shift_left: bool,
    ) -> Result<(), String> {
        let destination = destination_vector(instruction)?;
        if destination + 1 >= VECTOR_REGISTER_COUNT {
            return Err("64-bit vector shift destination pair exceeds the VGPR file".to_owned());
        }
        let Some(InstructionControl::Vop3(control)) = instruction.control else {
            return Err("64-bit vector shift requires VOP3 control".to_owned());
        };
        if control.absolute_mask != 0
            || control.negate_mask != 0
            || control.output_modifier != 0
            || control.clamp
            || control.operand_select != 0
            || control.scalar_destination.is_some()
        {
            return Err("64-bit vector shift uses reserved VOP3 modifier bits".to_owned());
        }

        let shift = self.temp(
            "uint",
            &format!("{} & 63u", self.raw_source(instruction, 0)?),
        );
        let value = self.temp("ulong", &self.raw_source64(instruction, 1)?);
        let result = self.temp(
            "ulong",
            &format!("{value} {} {shift}", if shift_left { "<<" } else { ">>" }),
        );
        self.store_vector(destination, &format!("(uint){result}"));
        self.store_vector(destination + 1, &format!("(uint)({result} >> 32)"));
        Ok(())
    }

    pub(crate) fn emit_vector_carry(
        &mut self,
        instruction: &Instruction,
        name: &str,
    ) -> Result<(), String> {
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
}
