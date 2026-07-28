// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! VOP3P packed maths lowering: packed f16 pairs, FMA-mix and the opsel/neg
//! source selection those encodings apply.

use remu_shader::{
    Instruction, InstructionControl, OperandKind, Vop3pControl, decode_v2f16_inline_constant,
};

use crate::compiler::vector::destination_vector;
use crate::{Compiler, SCALAR_REGISTER_COUNT, VECTOR_REGISTER_COUNT, format_u32};

impl Compiler<'_> {
    pub(crate) fn emit_vop3p(
        &mut self,
        instruction: &Instruction,
        name: &str,
    ) -> Result<(), String> {
        let Some(InstructionControl::Vop3p(control)) = instruction.control.as_ref() else {
            return Err(format!("missing VOP3P control for {name}"));
        };
        let destination = destination_vector(instruction)?;
        let result = match name {
            "VPkFmaF16" | "VPkAddF16" | "VPkMulF16" | "VPkMinF16" | "VPkMaxF16" => {
                self.emit_packed_f16(instruction, name, *control)?
            }
            "VFmaMixF32" | "VFmaMixloF16" | "VFmaMixhiF16" => {
                self.emit_fma_mix(instruction, name, destination, *control)?
            }
            _ => return Err(format!("unsupported packed VOP3 opcode {name}")),
        };
        self.store_vector(destination, &result);
        Ok(())
    }

    fn emit_packed_f16(
        &mut self,
        instruction: &Instruction,
        name: &str,
        control: Vop3pControl,
    ) -> Result<String, String> {
        let source_count = usize::from(name == "VPkFmaF16") + 2;
        let mut raw_sources = Vec::with_capacity(source_count);
        for index in 0..source_count {
            let source = self.vop3p_packed_raw_source(instruction, index)?;
            raw_sources.push(self.temp("uint", &source));
        }
        let low = self.emit_packed_f16_lane(name, &raw_sources, control, false);
        let high = self.emit_packed_f16_lane(name, &raw_sources, control, true);
        Ok(format!("({low} | ({high} << 16))"))
    }

    fn emit_packed_f16_lane(
        &mut self,
        name: &str,
        raw_sources: &[String],
        control: Vop3pControl,
        high_lane: bool,
    ) -> String {
        let select_mask = if high_lane {
            control.operand_select_high_mask
        } else {
            control.operand_select_mask
        };
        let negate_mask = if high_lane {
            control.negate_high_mask
        } else {
            control.negate_low_mask
        };
        let mut operands = Vec::with_capacity(raw_sources.len());
        for (index, raw) in raw_sources.iter().enumerate() {
            let shift = if select_mask & (1 << index) != 0 {
                16
            } else {
                0
            };
            let mut half_bits = format!("(({raw} >> {shift}) & 0xFFFFu)");
            if negate_mask & (1 << index) != 0 {
                half_bits = format!("({half_bits} ^ 0x8000u)");
            }
            operands.push(self.temp(
                "float",
                &format!("as_type<float>(remu_f16_to_f32_bits({half_bits}))"),
            ));
        }
        let mut result_bits = match name {
            "VPkFmaF16" => self.temp(
                "uint",
                &format!(
                    "remu_f16_fma_rto_bits({}, {}, {})",
                    operands[0], operands[1], operands[2]
                ),
            ),
            "VPkAddF16" => self.temp(
                "uint",
                &format!("as_type<uint>({} + {})", operands[0], operands[1]),
            ),
            "VPkMulF16" => self.temp(
                "uint",
                &format!("as_type<uint>({} * {})", operands[0], operands[1]),
            ),
            "VPkMinF16" => self.temp(
                "uint",
                &format!(
                    "remu_f16_minmax_bits({}, {}, false)",
                    operands[0], operands[1]
                ),
            ),
            "VPkMaxF16" => self.temp(
                "uint",
                &format!(
                    "remu_f16_minmax_bits({}, {}, true)",
                    operands[0], operands[1]
                ),
            ),
            _ => unreachable!("packed opcode was classified before lane lowering"),
        };
        if control.clamp {
            result_bits = self.temp("uint", &format!("remu_clamp_unit_f32_bits({result_bits})"));
        }
        self.temp("uint", &format!("remu_f32_to_f16_bits({result_bits})"))
    }

    fn emit_fma_mix(
        &mut self,
        instruction: &Instruction,
        name: &str,
        destination: u32,
        control: Vop3pControl,
    ) -> Result<String, String> {
        let mut operands = Vec::with_capacity(3);
        for index in 0..3 {
            operands.push(self.fma_mix_operand(instruction, control, index)?);
        }
        let mut result_bits = self.temp(
            "uint",
            &format!(
                "as_type<uint>(metal::precise::fma({}, {}, {}))",
                operands[0], operands[1], operands[2]
            ),
        );
        if control.clamp {
            result_bits = self.temp("uint", &format!("remu_clamp_unit_f32_bits({result_bits})"));
        }
        Ok(match name {
            "VFmaMixF32" => result_bits,
            "VFmaMixloF16" => {
                let half = self.temp("uint", &format!("remu_f32_to_f16_bits({result_bits})"));
                format!("(v[{destination}] & 0xFFFF0000u) | {half}")
            }
            "VFmaMixhiF16" => {
                let half = self.temp("uint", &format!("remu_f32_to_f16_bits({result_bits})"));
                format!("(v[{destination}] & 0x0000FFFFu) | ({half} << 16)")
            }
            _ => unreachable!("mix opcode was classified before result lowering"),
        })
    }

    fn fma_mix_operand(
        &mut self,
        instruction: &Instruction,
        control: Vop3pControl,
        index: usize,
    ) -> Result<String, String> {
        let operand = instruction
            .sources
            .get(index)
            .copied()
            .ok_or_else(|| format!("missing VOP3P source operand {index}"))?;
        // FMA_MIX combines OPSEL_HI and OPSEL into a two-bit source mode:
        // 0/1 select the complete FP32 source while 2/3 select its low/high FP16 half.
        // The selection applies to literals and inline constants as well as registers.
        let read_as_half = control.operand_select_high_mask & (1 << index) != 0;
        let mut bits = if read_as_half {
            let raw = if operand.kind == OperandKind::EncodedConstant {
                decode_v2f16_inline_constant(operand.value)
                    .map(format_u32)
                    .map_or_else(|| self.raw_source(instruction, index), Ok)?
            } else {
                self.raw_source(instruction, index)?
            };
            let shift = if control.operand_select_mask & (1 << index) != 0 {
                16
            } else {
                0
            };
            format!("remu_f16_to_f32_bits((({raw} >> {shift}) & 0xFFFFu))")
        } else {
            format!("as_type<uint>({})", self.float_source(instruction, index)?)
        };
        // FMA_MIX repurposes NEG_HI as absolute value and NEG as negate. Apply them
        // in architectural abs-then-neg order using sign-bit operations.
        if control.negate_high_mask & (1 << index) != 0 {
            bits = format!("({bits} & 0x7FFFFFFFu)");
        }
        if control.negate_low_mask & (1 << index) != 0 {
            bits = format!("({bits} ^ 0x80000000u)");
        }
        Ok(self.temp("float", &format!("as_type<float>({bits})")))
    }

    #[allow(clippy::unused_self)]
    fn vop3p_packed_raw_source(
        &self,
        instruction: &Instruction,
        index: usize,
    ) -> Result<String, String> {
        let operand = instruction
            .sources
            .get(index)
            .copied()
            .ok_or_else(|| format!("missing VOP3P source operand {index}"))?;
        match operand.kind {
            OperandKind::ScalarRegister if operand.value < SCALAR_REGISTER_COUNT => {
                Ok(format!("s[{}]", operand.value))
            }
            OperandKind::VectorRegister if operand.value < VECTOR_REGISTER_COUNT => {
                Ok(format!("v[{}]", operand.value))
            }
            OperandKind::LiteralConstant => Ok(format_u32(operand.value)),
            OperandKind::EncodedConstant => decode_v2f16_inline_constant(operand.value)
                .map(format_u32)
                .ok_or_else(|| {
                    format!("unsupported packed-f16 encoded constant {}", operand.value)
                }),
            _ => Err(format!(
                "VOP3P register operand is outside the architectural file: {operand}"
            )),
        }
    }
}
