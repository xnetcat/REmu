// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! VALU lowering: vector arithmetic, packed math, DPP/SDWA modifiers and vector compares.

use remu_shader::{Instruction, Operand, OperandKind, SdwaControl};

use crate::{Compiler, VECTOR_REGISTER_COUNT};

mod alu;
mod compare;
mod lane;
mod packed;
mod wide;

pub(crate) const fn is_supported_dpp_control(control: u32) -> bool {
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

pub(crate) fn destination_vector(instruction: &Instruction) -> Result<u32, String> {
    instruction
        .destinations
        .first()
        .copied()
        .filter(|operand| operand.kind == OperandKind::VectorRegister)
        .map(|operand| operand.value)
        .ok_or_else(|| "missing vector destination".to_owned())
}

pub(crate) fn vector_operand(operand: Operand) -> Result<u32, String> {
    (operand.kind == OperandKind::VectorRegister)
        .then_some(operand.value)
        .ok_or_else(|| format!("expected vector operand, got {operand}"))
}

impl Compiler<'_> {
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

    pub(crate) fn store_vector(&mut self, register: u32, expression: &str) {
        if register < VECTOR_REGISTER_COUNT {
            self.line(3, &format!("if (exec) v[{register}] = {expression};"));
        }
    }
}
