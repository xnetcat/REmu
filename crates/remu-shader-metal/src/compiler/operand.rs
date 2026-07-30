// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Operand sourcing: raw, 64-bit, float and half-float source expressions,
//! the SDWA and VOP3 modifiers applied to them, and result clamping.

use remu_shader::{
    Instruction, InstructionControl, OperandKind, decode_inline_constant,
    decode_v2f16_inline_constant,
};

use crate::{
    Compiler, EXEC_HI, EXEC_LO, SCALAR_REGISTER_COUNT, VCC_HI, VCC_LO, VECTOR_REGISTER_COUNT,
};

impl Compiler<'_> {
    pub(crate) fn raw_source(
        &self,
        instruction: &Instruction,
        index: usize,
    ) -> Result<String, String> {
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
    pub(crate) fn raw_source64(
        &self,
        instruction: &Instruction,
        index: usize,
    ) -> Result<String, String> {
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
            OperandKind::VectorRegister if operand.value + 1 < VECTOR_REGISTER_COUNT => {
                Ok(format!(
                    "((ulong)v[{}] | ((ulong)v[{}] << 32))",
                    operand.value,
                    operand.value + 1
                ))
            }
            OperandKind::LiteralConstant => Ok(format!("(ulong){}", format_u32(operand.value))),
            OperandKind::EncodedConstant => decode_inline_constant(operand.value)
                .map(|value| {
                    // Integer inline constants -1..-16 are architecturally sign-extended for
                    // 64-bit SALU operations. Zero extension turns `S_MOV_B64 EXEC, -1` into a
                    // half-wave mask and disables lanes 32..63 in every split wave64.
                    let value = if (193..=208).contains(&operand.value) {
                        u64::from(value) | (u64::from(u32::MAX) << 32)
                    } else {
                        u64::from(value)
                    };
                    format!("0x{value:016X}ul")
                })
                .ok_or_else(|| format!("unsupported 64-bit encoded constant {}", operand.value)),
            _ => Err(format!("invalid 64-bit source operand {operand}")),
        }
    }

    pub(crate) fn float_source(
        &self,
        instruction: &Instruction,
        index: usize,
    ) -> Result<String, String> {
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

    pub(crate) fn has_float_modifiers(instruction: &Instruction) -> bool {
        match instruction.control.as_ref() {
            Some(InstructionControl::Vop3(control)) => {
                control.absolute_mask != 0
                    || control.negate_mask != 0
                    || control.output_modifier != 0
                    || control.clamp
            }
            Some(InstructionControl::Sdwa(control)) => {
                control.absolute_mask != 0
                    || control.negate_mask != 0
                    || control.output_modifier != 0
                    || control.clamp
            }
            Some(InstructionControl::Dpp(control)) => {
                control.absolute_mask != 0 || control.negate_mask != 0
            }
            _ => false,
        }
    }

    pub(crate) fn f16_result(
        &mut self,
        instruction: &Instruction,
        destination: u32,
        expression: &str,
    ) -> String {
        let result = self.temp("uint", &self.float_result(instruction, expression));
        let packed = format!("remu_f32_to_f16_bits({result})");
        match instruction.control.as_ref() {
            Some(InstructionControl::Vop3(control)) if control.operand_select & 8 != 0 => {
                format!("(v[{destination}] & 0xFFFFu) | ({packed} << 16)")
            }
            Some(InstructionControl::Vop3(_)) => {
                format!("(v[{destination}] & 0xFFFF0000u) | {packed}")
            }
            _ => packed,
        }
    }

    pub(crate) fn half_float_source(
        &self,
        instruction: &Instruction,
        index: usize,
    ) -> Result<String, String> {
        let operand = instruction
            .sources
            .get(index)
            .copied()
            .ok_or_else(|| format!("missing half-float source operand {index}"))?;
        let (raw, shift) = match operand.kind {
            // Ordinary F16 instructions receive inline floating-point constants already
            // converted to F16. They are scalar values rather than packed register operands, so
            // OPSEL does not select a half of the ordinary FP32 inline-constant representation.
            OperandKind::EncodedConstant => {
                let raw = match operand.value {
                    251..=253 => self.raw_source_with_sdwa_modifiers(instruction, index, false)?,
                    encoded => decode_v2f16_inline_constant(encoded)
                        .map(format_u32)
                        .ok_or_else(|| {
                            format!("unsupported half-float encoded constant {encoded}")
                        })?,
                };
                (raw, 0)
            }
            _ => {
                let raw = self.raw_source_with_sdwa_modifiers(instruction, index, false)?;
                let shift = match instruction.control.as_ref() {
                    Some(InstructionControl::Vop3(control))
                        if control.operand_select & (1 << index) != 0 =>
                    {
                        16
                    }
                    _ => 0,
                };
                (raw, shift)
            }
        };
        let mut value =
            format!("as_type<float>(remu_f16_to_f32_bits((({raw}) >> {shift}u) & 0xFFFFu))");
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
    pub(crate) fn float_result(&self, instruction: &Instruction, expression: &str) -> String {
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
}
pub(crate) fn format_u32(value: u32) -> String {
    if value <= 9 {
        format!("{value}u")
    } else {
        format!("0x{value:X}u")
    }
}
