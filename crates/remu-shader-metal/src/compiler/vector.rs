// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! VALU lowering: vector arithmetic, packed math, DPP/SDWA modifiers and vector compares.

use remu_shader::{
    Dpp8Control, DppControl, Instruction, InstructionControl, Operand, OperandKind, SdwaControl,
    Vop3pControl, decode_v2f16_inline_constant,
};

use crate::compiler::scalar::destination_scalar;
use crate::{Compiler, SCALAR_REGISTER_COUNT, VCC_LO, VECTOR_REGISTER_COUNT, format_u32};

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VectorCompare {
    Always(bool),
    Ordered,
    Unordered,
    Float {
        operator: &'static str,
        negate: bool,
    },
    Integer {
        operator: &'static str,
        signed: bool,
    },
    Class,
}

pub(crate) fn classify_vector_compare(name: &str) -> Result<VectorCompare, String> {
    if name == "VCmpClassF32" {
        return Ok(VectorCompare::Class);
    }
    let tail = name
        .strip_prefix("VCmpx")
        .or_else(|| name.strip_prefix("VCmp"))
        .ok_or_else(|| format!("invalid vector compare opcode {name}"))?;
    let (relation, kind) = tail
        .strip_suffix("F32")
        .map(|relation| (relation, "float"))
        .or_else(|| tail.strip_suffix("F16").map(|relation| (relation, "float")))
        .or_else(|| {
            tail.strip_suffix("I32")
                .map(|relation| (relation, "signed"))
        })
        .or_else(|| {
            tail.strip_suffix("U32")
                .map(|relation| (relation, "unsigned"))
        })
        .ok_or_else(|| format!("invalid vector compare type in {name}"))?;
    match kind {
        "float" => match relation {
            "F" => Ok(VectorCompare::Always(false)),
            "Tru" => Ok(VectorCompare::Always(true)),
            "O" => Ok(VectorCompare::Ordered),
            "U" => Ok(VectorCompare::Unordered),
            "Lt" => Ok(VectorCompare::Float {
                operator: "<",
                negate: false,
            }),
            "Eq" => Ok(VectorCompare::Float {
                operator: "==",
                negate: false,
            }),
            "Le" => Ok(VectorCompare::Float {
                operator: "<=",
                negate: false,
            }),
            "Gt" => Ok(VectorCompare::Float {
                operator: ">",
                negate: false,
            }),
            "Lg" => Ok(VectorCompare::Float {
                operator: "!=",
                negate: false,
            }),
            "Ge" => Ok(VectorCompare::Float {
                operator: ">=",
                negate: false,
            }),
            "Nge" => Ok(VectorCompare::Float {
                operator: ">=",
                negate: true,
            }),
            "Nlg" => Ok(VectorCompare::Float {
                operator: "!=",
                negate: true,
            }),
            "Ngt" => Ok(VectorCompare::Float {
                operator: ">",
                negate: true,
            }),
            "Nle" => Ok(VectorCompare::Float {
                operator: "<=",
                negate: true,
            }),
            "Neq" => Ok(VectorCompare::Float {
                operator: "==",
                negate: true,
            }),
            "Nlt" => Ok(VectorCompare::Float {
                operator: "<",
                negate: true,
            }),
            _ => Err(format!("unsupported float comparison {name}")),
        },
        "signed" | "unsigned" => {
            let operator = match relation {
                "F" | "T" => {
                    return Ok(VectorCompare::Always(relation == "T"));
                }
                "Lt" => "<",
                "Eq" => "==",
                "Le" => "<=",
                "Gt" => ">",
                "Ne" => "!=",
                "Ge" => ">=",
                _ => return Err(format!("unsupported integer comparison {name}")),
            };
            Ok(VectorCompare::Integer {
                operator,
                signed: kind == "signed",
            })
        }
        _ => unreachable!("comparison kind is selected above"),
    }
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
    fn dpp_source_lane(control: DppControl) -> Result<(String, String), String> {
        let dpp = control.control;
        if dpp <= 0xff {
            let selected = format!("(({dpp}u >> ((remu_lane & 3u) * 2u)) & 3u)");
            let target = format!("((remu_lane & 0xFFFFFFFCu) + {selected})");
            return Ok((target, "true".to_owned()));
        }
        if (0x101..=0x10f).contains(&dpp) {
            let shift = dpp & 15;
            let shifted = format!("((remu_lane & 15u) + {shift}u)");
            let in_range = format!("({shifted} < 16u)");
            let target = format!("((remu_lane & 0xFFFFFFF0u) + ({shifted} & 15u))");
            return Ok((target, in_range));
        }
        if (0x111..=0x11f).contains(&dpp) {
            let shift = dpp & 15;
            let in_range = format!("((remu_lane & 15u) >= {shift}u)");
            let target =
                format!("((remu_lane & 0xFFFFFFF0u) + (((remu_lane & 15u) - {shift}u) & 15u))");
            return Ok((target, in_range));
        }
        if (0x121..=0x12f).contains(&dpp) {
            let shift = dpp & 15;
            let target =
                format!("((remu_lane & 0xFFFFFFF0u) + (((remu_lane & 15u) - {shift}u) & 15u))");
            return Ok((target, "true".to_owned()));
        }
        let target = match dpp {
            0x140 => "(remu_lane & 0xFFFFFFF0u) + (15u - (remu_lane & 15u))".to_owned(),
            0x141 => "(remu_lane & 0xFFFFFFF8u) + (7u - (remu_lane & 7u))".to_owned(),
            0x150..=0x15f => format!("(remu_lane & 0xFFFFFFF0u) + {}u", dpp & 15),
            0x160..=0x16f => format!(
                "(remu_lane & 0xFFFFFFF0u) + ((remu_lane & 15u) ^ {}u)",
                dpp & 15
            ),
            _ => return Err(format!("unsupported DPP16 control 0x{dpp:03X}")),
        };
        Ok((format!("({target})"), "true".to_owned()))
    }

    fn shuffle_lane(value: &str, target_lane: &str) -> String {
        format!("simd_shuffle({value}, (ushort){target_lane})")
    }

    fn lane_active_expression(target_lane: &str) -> String {
        format!("simd_shuffle(exec ? 1u : 0u, (ushort){target_lane}) != 0u")
    }

    pub(crate) fn apply_dpp_source(control: DppControl, value: &str) -> Result<String, String> {
        let (target_lane, in_range) = Self::dpp_source_lane(control)?;
        let safe_target = format!("((({in_range}) ? {target_lane} : remu_lane) & 31u)");
        let shuffled = Self::shuffle_lane(value, &safe_target);
        if control.fetch_inactive {
            return Ok(shuffled);
        }
        let source_active = Self::lane_active_expression(&safe_target);
        Ok(format!(
            "((({in_range}) && ({source_active})) ? ({shuffled}) : 0u)"
        ))
    }

    pub(crate) fn apply_dpp8_source(control: Dpp8Control, value: &str) -> String {
        let selector = format!(
            "(({}u >> ((remu_lane & 7u) * 3u)) & 7u)",
            control.lane_selectors
        );
        let target_lane = format!("(((remu_lane & 0xFFFFFFF8u) + {selector}) & 31u)");
        let shuffled = Self::shuffle_lane(value, &target_lane);
        if control.fetch_inactive {
            return shuffled;
        }
        let source_active = Self::lane_active_expression(&target_lane);
        format!("(({source_active}) ? ({shuffled}) : 0u)")
    }

    fn emit_dpp_write_enabled(&mut self, control: DppControl) -> Result<String, String> {
        let (_, in_range) = Self::dpp_source_lane(control)?;
        let row_enabled = format!("(({}u >> (remu_lane >> 4)) & 1u) != 0u", control.row_mask);
        let bank_enabled = format!("(({}u >> (remu_lane & 3u)) & 1u) != 0u", control.bank_mask);
        let source_allows = if control.bound_control {
            "true"
        } else {
            &in_range
        };
        Ok(self.temp(
            "bool",
            &format!("({row_enabled}) && ({bank_enabled}) && ({source_allows})"),
        ))
    }

    fn emit_vop3p(&mut self, instruction: &Instruction, name: &str) -> Result<(), String> {
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

    #[allow(clippy::too_many_lines)]
    pub(crate) fn emit_vector_alu(&mut self, instruction: &Instruction) -> Result<(), String> {
        if let Some(InstructionControl::Sdwa(control)) = instruction.control.as_ref()
            && (control.source0_select == 7
                || control.source1_select == 7
                || control.destination_select == 7
                || control.destination_unused == 3)
        {
            return Err("reserved SDWA selector or destination-unused mode".to_owned());
        }
        if let Some(InstructionControl::Dpp(control)) = instruction.control
            && !is_supported_dpp_control(control.control)
        {
            return Err(format!(
                "unsupported DPP16 control 0x{:03X}",
                control.control
            ));
        }
        let name = instruction
            .opcode
            .known_name()
            .ok_or_else(|| "raw vector opcode".to_owned())?;
        if matches!(instruction.control, Some(InstructionControl::Vop3p(_))) {
            return self.emit_vop3p(instruction, name);
        }
        if name.starts_with("VCmp") {
            return self.emit_vector_compare(instruction, name);
        }
        if matches!(
            name,
            "VAddcU32"
                | "VSubbU32"
                | "VSubbrevU32"
                | "VAddCoU32"
                | "VSubCoU32"
                | "VSubrevCoU32"
                | "VAddCoCiU32"
        ) {
            return self.emit_vector_carry(instruction, name);
        }
        if name == "VReadlaneB32" {
            let destination = destination_scalar(instruction)?;
            let value = self.raw_source(instruction, 0)?;
            let lane = self.raw_source(instruction, 1)?;
            let selected = if self.split_wave64() {
                self.line(
                    3,
                    &format!("remu_readlane_values[remu_thread_index] = {value};"),
                );
                self.line(3, "threadgroup_barrier(mem_flags::mem_threadgroup);");
                let selected_lane = self.temp("uint", &format!("({lane}) & 63u"));
                let wave_base = self.temp("uint", "remu_wave_index << 6u");
                let selected = self.temp(
                    "uint",
                    &format!("remu_readlane_values[{wave_base} + {selected_lane}]"),
                );
                // No lane may reuse its slot for a later V_READLANE until both SIMD32 halves
                // have consumed the current value.
                self.line(3, "threadgroup_barrier(mem_flags::mem_threadgroup);");
                selected
            } else {
                self.temp(
                    "uint",
                    &format!("simd_shuffle({value}, (ushort)({lane} & 31u))"),
                )
            };
            // V_READLANE is a wave operation and is architecturally independent of EXEC.
            self.store_scalar(destination, &selected);
            return Ok(());
        }
        if name == "VReadfirstlaneB32" {
            let destination = destination_scalar(instruction)?;
            let value = self.raw_source(instruction, 0)?;
            // The lane read is the lowest one EXEC has enabled, which is not necessarily lane
            // zero and is not the same as the SIMD group's own notion of an active thread. A
            // wave with nothing enabled still has to produce a value, and the hardware reads
            // lane zero in that case, which is what the mask of zero falls through to here.
            let active = self.temp("uint", "(uint)(uint64_t)simd_ballot(exec)");
            let lane = self.temp("uint", &format!("{active} == 0u ? 0u : ctz({active})"));
            let mut selected = self.temp("uint", &format!("simd_shuffle({value}, (ushort){lane})"));
            if self.split_wave64() {
                let slot = self.temp("uint", "(remu_wave_index << 1u) | remu_wave_half");
                self.line(
                    3,
                    &format!(
                        "if (remu_lane == 0u) {{ remu_readfirstlane_values[{slot}] = {selected}; remu_readfirstlane_active[{slot}] = {active} != 0u; }}"
                    ),
                );
                self.line(3, "threadgroup_barrier(mem_flags::mem_threadgroup);");
                let low_slot = self.temp("uint", "remu_wave_index << 1u");
                selected = self.temp(
                    "uint",
                    &format!(
                        "remu_readfirstlane_active[{low_slot}] != 0u ? remu_readfirstlane_values[{low_slot}] : (remu_readfirstlane_active[{low_slot} + 1u] != 0u ? remu_readfirstlane_values[{low_slot} + 1u] : remu_readfirstlane_values[{low_slot}])"
                    ),
                );
                self.line(3, "threadgroup_barrier(mem_flags::mem_threadgroup);");
            }
            self.store_scalar(destination, &selected);
            return Ok(());
        }
        if matches!(name, "VPermlane16B32" | "VPermlanex16B32") {
            return self.emit_permlane16(instruction, name == "VPermlanex16B32");
        }
        if name == "VMadU64U32" {
            return self.emit_mad_u64_u32(instruction);
        }
        if matches!(name, "VLshlrevB64" | "VLshrrevB64") {
            return self.emit_vector_shift64(instruction, name == "VLshlrevB64");
        }
        if name == "VWritelaneB32" {
            let destination = destination_vector(instruction)?;
            let value = self.raw_source(instruction, 0)?;
            let lane = self.raw_source(instruction, 1)?;
            // V_WRITELANE also ignores EXEC: exactly the selected physical lane is replaced and
            // all other lanes retain the old VGPR value.
            let architectural_lane = if self.split_wave64() {
                "remu_wave_lane"
            } else {
                "remu_lane"
            };
            let lane_mask = if self.options.wave_lane_count == 64 {
                63
            } else {
                31
            };
            self.line(
                3,
                &format!(
                    "if ({architectural_lane} == (({lane}) & {lane_mask}u)) v[{destination}] = {value};"
                ),
            );
            return Ok(());
        }
        let destination = destination_vector(instruction)?;
        let expression = match name {
            "VCndmaskB32" => {
                let mask = if instruction.sources.len() > 2 {
                    self.mask_source(instruction.sources[2])?
                } else {
                    "vcc".to_owned()
                };
                if Self::has_float_modifiers(instruction) {
                    // V_CNDMASK_B32 is bitwise only without modifiers. RDNA2 explicitly permits
                    // VOP3/SDWA floating-point ABS/NEG (and result OMOD/CLAMP) so shaders can
                    // conditionally change a float's sign without a separate VALU instruction.
                    // Treating those sources as raw uints silently drops the modifier.
                    let false_value = self.float_source(instruction, 0)?;
                    let true_value = self.float_source(instruction, 1)?;
                    self.float_result(
                        instruction,
                        &format!("({mask}) ? ({true_value}) : ({false_value})"),
                    )
                } else {
                    let false_value = self.raw_source(instruction, 0)?;
                    let true_value = self.raw_source(instruction, 1)?;
                    format!("({mask}) ? ({true_value}) : ({false_value})")
                }
            }
            "VMovB32" => {
                // RDNA2 defines V_MOV_B32 as an untyped move only when no floating-point
                // modifiers are present. ABS/NEG (and SDWA CLAMP/OMOD) explicitly reinterpret
                // S0 as f32. Applying SDWA NEG to the raw uint would compute two's-complement
                // negation instead of toggling the floating-point sign, e.g. 1.0 became -4.0.
                if Self::has_float_modifiers(instruction) {
                    let source = self.float_source(instruction, 0)?;
                    self.float_result(instruction, &source)
                } else {
                    self.raw_source(instruction, 0)?
                }
            }
            "VBfrevB32" => format!("reverse_bits({})", self.raw_source(instruction, 0)?),
            "VBcntU32B32" => format!(
                "popcount({}) + {}",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMbcntLoU32B32" | "VMbcntHiU32B32" => {
                let source = self.raw_source(instruction, 0)?;
                let addend = self.raw_source(instruction, 1)?;
                let below = self.temp("uint", "remu_lane == 0u ? 0u : ((1u << remu_lane) - 1u)");
                let mask = match (name, self.split_wave64()) {
                    ("VMbcntLoU32B32", true) => {
                        format!("remu_wave_half == 0u ? {below} : 0xFFFFFFFFu")
                    }
                    ("VMbcntHiU32B32", true) => {
                        format!("remu_wave_half == 0u ? 0u : {below}")
                    }
                    ("VMbcntLoU32B32", false) => below,
                    ("VMbcntHiU32B32", false) => "0u".to_owned(),
                    _ => unreachable!("matched masked bit-count operation"),
                };
                format!("popcount(({source}) & ({mask})) + {addend}")
            }
            "VXor3B32" => format!(
                "{} ^ {} ^ {}",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VAddF32" => self.float_result(
                instruction,
                &format!(
                    "{} + {}",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                ),
            ),
            "VSubF32" => self.float_result(
                instruction,
                &format!(
                    "{} - {}",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                ),
            ),
            "VSubrevF32" => self.float_result(
                instruction,
                &format!(
                    "{} - {}",
                    self.float_source(instruction, 1)?,
                    self.float_source(instruction, 0)?
                ),
            ),
            "VMulF32" => self.float_result(
                instruction,
                &format!(
                    "{} * {}",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                ),
            ),
            "VAddF16" | "VSubF16" | "VSubrevF16" | "VMulF16" | "VMaxF16" | "VMinF16" => {
                let first = self.half_float_source(instruction, 0)?;
                let second = self.half_float_source(instruction, 1)?;
                let expression = match name {
                    "VAddF16" => format!("{first} + {second}"),
                    "VSubF16" => format!("{first} - {second}"),
                    "VSubrevF16" => format!("{second} - {first}"),
                    "VMulF16" => format!("{first} * {second}"),
                    "VMaxF16" => format!("fmax({first}, {second})"),
                    "VMinF16" => format!("fmin({first}, {second})"),
                    _ => unreachable!("matched observed half-float arithmetic"),
                };
                self.f16_result(instruction, destination, &expression)
            }
            "VFmacF16" => {
                let first = self.half_float_source(instruction, 0)?;
                let second = self.half_float_source(instruction, 1)?;
                let previous_raw = self.temp("uint", &format!("v[{destination}] & 0xFFFFu"));
                let previous = self.temp(
                    "float",
                    &format!("as_type<float>(remu_f16_to_f32_bits({previous_raw}))"),
                );
                self.f16_result(
                    instruction,
                    destination,
                    &format!("fma({first}, {second}, {previous})"),
                )
            }
            "VFmaF16" | "VFmaMkF16" | "VFmaAkF16" => {
                let first = self.half_float_source(instruction, 0)?;
                let second = self.half_float_source(instruction, 1)?;
                let third = self.half_float_source(instruction, 2)?;
                self.f16_result(
                    instruction,
                    destination,
                    &format!("fma({first}, {second}, {third})"),
                )
            }
            "VLdexpF16" => {
                let first = self.half_float_source(instruction, 0)?;
                let second_raw = self.temp("uint", &self.raw_source(instruction, 1)?);
                self.f16_result(
                    instruction,
                    destination,
                    &format!("ldexp({first}, (int)(short)({second_raw} & 0xFFFFu))"),
                )
            }
            "VMinF32" => self.float_result(
                instruction,
                &format!(
                    "fmin({}, {})",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                ),
            ),
            "VMaxF32" => self.float_result(
                instruction,
                &format!(
                    "fmax({}, {})",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                ),
            ),
            "VMin3F32" => self.float_result(
                instruction,
                &format!(
                    "fmin(fmin({}, {}), {})",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?,
                    self.float_source(instruction, 2)?
                ),
            ),
            "VMax3F32" => self.float_result(
                instruction,
                &format!(
                    "fmax(fmax({}, {}), {})",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?,
                    self.float_source(instruction, 2)?
                ),
            ),
            "VMed3F32" => {
                let first = self.float_source(instruction, 0)?;
                let second = self.float_source(instruction, 1)?;
                let third = self.float_source(instruction, 2)?;
                self.float_result(
                    instruction,
                    &format!(
                        "fmax(fmin({first}, {second}), fmin(fmax({first}, {second}), {third}))"
                    ),
                )
            }
            "VMin3I32" => format!(
                "as_type<uint>(min(min(as_type<int>({}), as_type<int>({})), as_type<int>({})))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VMax3I32" => format!(
                "as_type<uint>(max(max(as_type<int>({}), as_type<int>({})), as_type<int>({})))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VMed3I32" => {
                let first = format!("as_type<int>({})", self.raw_source(instruction, 0)?);
                let second = format!("as_type<int>({})", self.raw_source(instruction, 1)?);
                let third = format!("as_type<int>({})", self.raw_source(instruction, 2)?);
                format!(
                    "as_type<uint>(max(min({first}, {second}), min(max({first}, {second}), {third})))"
                )
            }
            "VMin3U32" => format!(
                "min(min({}, {}), {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VMax3U32" => format!(
                "max(max({}, {}), {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VMed3U32" => {
                let first = self.raw_source(instruction, 0)?;
                let second = self.raw_source(instruction, 1)?;
                let third = self.raw_source(instruction, 2)?;
                format!("max(min({first}, {second}), min(max({first}, {second}), {third}))")
            }
            "VSadU32" => {
                let first = self.raw_source(instruction, 0)?;
                let second = self.raw_source(instruction, 1)?;
                let addend = self.raw_source(instruction, 2)?;
                format!("(max({first}, {second}) - min({first}, {second}) + {addend})")
            }
            "VFmacF32" | "VMacF32" => self.float_result(
                instruction,
                &format!(
                    "fma({}, {}, as_type<float>(v[{destination}]))",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?
                ),
            ),
            "VFloorF32" => self.float_result(
                instruction,
                &format!("floor({})", self.float_source(instruction, 0)?),
            ),
            "VCeilF32" => self.float_result(
                instruction,
                &format!("ceil({})", self.float_source(instruction, 0)?),
            ),
            "VTruncF32" => self.float_result(
                instruction,
                &format!("trunc({})", self.float_source(instruction, 0)?),
            ),
            "VRndneF32" => self.float_result(
                instruction,
                &format!("rint({})", self.float_source(instruction, 0)?),
            ),
            "VFractF32" => self.float_result(
                instruction,
                &format!("fract({})", self.float_source(instruction, 0)?),
            ),
            "VSqrtF32" => self.float_result(
                instruction,
                &format!("sqrt({})", self.float_source(instruction, 0)?),
            ),
            "VRsqF32" => self.float_result(
                instruction,
                &format!("rsqrt({})", self.float_source(instruction, 0)?),
            ),
            "VRsqF16" => {
                let result = self.temp(
                    "uint",
                    &format!(
                        "as_type<uint>(rsqrt({}))",
                        self.half_float_source(instruction, 0)?
                    ),
                );
                format!("remu_f32_to_f16_bits({result})")
            }
            "VRcpF16" => {
                let result = self.temp(
                    "uint",
                    &format!(
                        "as_type<uint>(1.0f / {})",
                        self.half_float_source(instruction, 0)?
                    ),
                );
                format!("remu_f32_to_f16_bits({result})")
            }
            "VSqrtF16" => {
                let result = self.temp(
                    "uint",
                    &format!(
                        "as_type<uint>(sqrt({}))",
                        self.half_float_source(instruction, 0)?
                    ),
                );
                format!("remu_f32_to_f16_bits({result})")
            }
            "VLogF16" => {
                let result = self.temp(
                    "uint",
                    &format!(
                        "as_type<uint>(log2({}))",
                        self.half_float_source(instruction, 0)?
                    ),
                );
                format!("remu_f32_to_f16_bits({result})")
            }
            "VExpF16" => {
                let result = self.temp(
                    "uint",
                    &format!(
                        "as_type<uint>(exp2({}))",
                        self.half_float_source(instruction, 0)?
                    ),
                );
                format!("remu_f32_to_f16_bits({result})")
            }
            "VRcpF32" | "VRcpIflagF32" => self.float_result(
                instruction,
                &format!("1.0f / {}", self.float_source(instruction, 0)?),
            ),
            "VLogF32" => self.float_result(
                instruction,
                &format!("log2({})", self.float_source(instruction, 0)?),
            ),
            "VExpF32" => self.float_result(
                instruction,
                &format!("exp2({})", self.float_source(instruction, 0)?),
            ),
            "VLdexpF32" => self.float_result(
                instruction,
                &format!(
                    "ldexp({}, as_type<int>({}))",
                    self.float_source(instruction, 0)?,
                    self.raw_source(instruction, 1)?
                ),
            ),
            "VDivFixupF32" => self.float_result(
                instruction,
                &format!(
                    "as_type<float>(remu_div_fixup_f32_bits(as_type<uint>({}), as_type<uint>({}), as_type<uint>({})))",
                    self.float_source(instruction, 0)?,
                    self.float_source(instruction, 1)?,
                    self.float_source(instruction, 2)?
                ),
            ),
            "VSinF32" => self.float_result(
                instruction,
                &format!(
                    "sin({} * 6.28318530717958647692f)",
                    self.float_source(instruction, 0)?
                ),
            ),
            "VCosF32" => self.float_result(
                instruction,
                &format!(
                    "cos({} * 6.28318530717958647692f)",
                    self.float_source(instruction, 0)?
                ),
            ),
            "VAndB32" => format!(
                "({} & {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VOrB32" => format!(
                "({} | {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VXorB32" => format!(
                "({} ^ {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VXnorB32" => format!(
                "~({} ^ {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VNotB32" => format!("~{}", self.raw_source(instruction, 0)?),
            "VAddI32" => format!(
                "({} + {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VSubI32" => format!(
                "({} - {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VSubrevI32" => format!(
                "({} - {})",
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 0)?
            ),
            "VMinI32" => format!(
                "as_type<uint>(min(as_type<int>({}), as_type<int>({})))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMaxI32" => format!(
                "as_type<uint>(max(as_type<int>({}), as_type<int>({})))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VLshrB32" => format!(
                "({} >> ({} & 31u))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VLshrrevB32" => format!(
                "({} >> ({} & 31u))",
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 0)?
            ),
            "VAshrI32" => format!(
                "as_type<uint>(as_type<int>({}) >> ({} & 31u))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VAshrrevI32" => format!(
                "as_type<uint>(as_type<int>({}) >> ({} & 31u))",
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 0)?
            ),
            "VLshlB32" => format!(
                "({} << ({} & 31u))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VLshlrevB32" => format!(
                "({} << ({} & 31u))",
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 0)?
            ),
            "VLshlAddU32" => format!(
                "(({} << ({} & 31u)) + {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VAddLshlU32" => format!(
                "(({} + {}) << ({} & 31u))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VBfeU32" => {
                let value = self.temp("uint", &self.raw_source(instruction, 0)?);
                let offset = self.temp(
                    "uint",
                    &format!("{} & 31u", self.raw_source(instruction, 1)?),
                );
                let width = self.temp(
                    "uint",
                    &format!(
                        "min({} & 31u, 32u - {offset})",
                        self.raw_source(instruction, 2)?
                    ),
                );
                format!("{width} == 0u ? 0u : extract_bits({value}, {offset}, {width})")
            }
            "VBfeI32" => {
                let value = self.temp(
                    "int",
                    &format!("as_type<int>({})", self.raw_source(instruction, 0)?),
                );
                let offset = self.temp(
                    "uint",
                    &format!("{} & 31u", self.raw_source(instruction, 1)?),
                );
                let width = self.temp(
                    "uint",
                    &format!("{} & 31u", self.raw_source(instruction, 2)?),
                );
                let mask = self.temp(
                    "uint",
                    &format!("{width} == 0u ? 0u : ((1u << {width}) - 1u)"),
                );
                format!("(as_type<uint>({value} >> {offset}) & {mask})")
            }
            "VBfiB32" => {
                let mask = self.temp("uint", &self.raw_source(instruction, 0)?);
                let insert = self.temp("uint", &self.raw_source(instruction, 1)?);
                let base = self.temp("uint", &self.raw_source(instruction, 2)?);
                format!("(({mask} & {insert}) | (~{mask} & {base}))")
            }
            // Three-operand integer forms fold a second ALU step into one instruction. The shift
            // amount is masked because the hardware uses only the low five bits.
            "VAdd3U32" => format!(
                "({} + {} + {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VXadU32" => format!(
                "(({} ^ {}) + {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VOr3U32" => format!(
                "({} | {} | {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VAndOrB32" => format!(
                "(({} & {}) | {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VLshlOrU32" => format!(
                "(({} << ({} & 31u)) | {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VCvtF32I32" => self.float_result(
                instruction,
                &format!("(float)as_type<int>({})", self.raw_source(instruction, 0)?),
            ),
            "VCvtF32U32" => self.float_result(
                instruction,
                &format!("(float){}", self.raw_source(instruction, 0)?),
            ),
            "VCvtF16F32" => {
                let source = self.float_source(instruction, 0)?;
                self.f16_result(instruction, destination, &source)
            }
            "VCvtF32F16" => {
                let source = self.half_float_source(instruction, 0)?;
                self.float_result(instruction, &source)
            }
            "VCvtU16F16" => {
                let source = self.temp("float", &self.half_float_source(instruction, 0)?);
                format!(
                    "(isnan({source}) || {source} <= 0.0f) ? 0u : ({source} >= 65535.0f ? 0xFFFFu : (uint)trunc({source}))"
                )
            }
            "VMinU32" => format!(
                "min({}, {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMaxU32" => format!(
                "max({}, {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMulLoU32" => format!(
                "({} * {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMulHiU32" => format!(
                "(uint)(((ulong){} * (ulong){}) >> 32)",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMulHiI32" => format!(
                "(uint)(int)(((long)(int){} * (long)(int){}) >> 32)",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMulU32U24" => format!(
                "(({} & 0xFFFFFFu) * ({} & 0xFFFFFFu))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMadU32U24" => format!(
                "(({} & 0xFFFFFFu) * ({} & 0xFFFFFFu) + {})",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?,
                self.raw_source(instruction, 2)?
            ),
            "VMulHiU32U24" => format!(
                "(uint)(((ulong)({} & 0xFFFFFFu) * (ulong)({} & 0xFFFFFFu)) >> 32)",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VCvtU32F32" => {
                let source = self.temp("float", &self.float_source(instruction, 0)?);
                format!(
                    "(isnan({source}) || {source} <= 0.0f) ? 0u : ({source} >= 4294967296.0f ? 0xFFFFFFFFu : (uint)trunc({source}))"
                )
            }
            "VCvtI32F32" => {
                let source = self.temp("float", &self.float_source(instruction, 0)?);
                format!(
                    "isnan({source}) ? 0u : ({source} <= -2147483648.0f ? 0x80000000u : ({source} >= 2147483648.0f ? 0x7FFFFFFFu : as_type<uint>((int)trunc({source}))))"
                )
            }
            "VCvtFlrI32F32" => {
                let source = self.temp("float", &self.float_source(instruction, 0)?);
                // The conversion is floor followed by the architectural saturating F32->I32
                // conversion. Guard both endpoints before the MSL cast: converting NaN,
                // infinity, or an out-of-range float to int is otherwise undefined.
                format!(
                    "isnan({source}) ? 0u : ({source} <= -2147483648.0f ? 0x80000000u : ({source} >= 2147483648.0f ? 0x7FFFFFFFu : as_type<uint>((int)floor({source}))))"
                )
            }
            "VCvtOffF32I4" => {
                let source = self.temp("uint", &self.raw_source(instruction, 0)?);
                self.float_result(
                    instruction,
                    &format!("(float)((int)({source} << 28) >> 28) * 0.0625f"),
                )
            }
            "VCubeidF32" | "VCubescF32" | "VCubetcF32" | "VCubemaF32" => {
                self.cube_result(instruction, name)?
            }
            "VCvtPkrtzF16F32" => {
                let first = self.temp(
                    "uint",
                    &format!("as_type<uint>({})", self.float_source(instruction, 0)?),
                );
                let second = self.temp(
                    "uint",
                    &format!("as_type<uint>({})", self.float_source(instruction, 1)?),
                );
                format!(
                    "(remu_f32_to_f16_rtz_bits({first}) | (remu_f32_to_f16_rtz_bits({second}) << 16))"
                )
            }
            "VCvtPknormU16F32" => {
                let first = self.temp("float", &self.float_source(instruction, 0)?);
                let second = self.temp("float", &self.float_source(instruction, 1)?);
                // RDNA2's UNORM conversion is round-to-nearest-even after saturation. Keep the
                // cast on the finite, in-range branch so Metal never observes an undefined
                // float-to-uint conversion. The usual DX10 NaN mode maps NaN to zero.
                let low = self.temp(
                    "uint",
                    &format!(
                        "(isnan({first}) || {first} <= 0.0f) ? 0u : ({first} >= 1.0f ? 0xFFFFu : (uint)rint({first} * 65535.0f))"
                    ),
                );
                let high = self.temp(
                    "uint",
                    &format!(
                        "(isnan({second}) || {second} <= 0.0f) ? 0u : ({second} >= 1.0f ? 0xFFFFu : (uint)rint({second} * 65535.0f))"
                    ),
                );
                format!("({low} | ({high} << 16))")
            }
            "VCvtPkU16U32" => format!(
                "(({} & 0xFFFFu) | (({} & 0xFFFFu) << 16))",
                self.raw_source(instruction, 0)?,
                self.raw_source(instruction, 1)?
            ),
            "VMadF32" | "VFmaF32" | "VMadAkF32" | "VMadMkF32" | "VFmaAkF32" | "VFmaMkF32" => self
                .float_result(
                    instruction,
                    &format!(
                        "fma({}, {}, {})",
                        self.float_source(instruction, 0)?,
                        self.float_source(instruction, 1)?,
                        self.float_source(instruction, 2)?
                    ),
                ),
            _ => return Err(format!("unsupported vector opcode {name}")),
        };
        let expression =
            if let Some(InstructionControl::Dpp(control)) = instruction.control.as_ref() {
                let result = self.temp("uint", &expression);
                let write_enabled = self.emit_dpp_write_enabled(*control)?;
                format!("({write_enabled}) ? {result} : v[{destination}]")
            } else if matches!(instruction.control, Some(InstructionControl::Dpp8(_))) {
                // Keep the shuffle converged across inactive guest lanes; only the architectural
                // destination write is EXEC-masked below.
                self.temp("uint", &expression)
            } else if let Some(InstructionControl::Sdwa(control)) = instruction.control.as_ref()
                && control.scalar_destination.is_none()
            {
                let result = self.temp("uint", &expression);
                self.apply_sdwa_destination(control, &result, &format!("v[{destination}]"))?
            } else {
                expression
            };
        self.store_vector(destination, &expression);
        Ok(())
    }

    fn emit_permlane16(
        &mut self,
        instruction: &Instruction,
        exchange_rows: bool,
    ) -> Result<(), String> {
        let destination = destination_vector(instruction)?;
        let source = instruction
            .sources
            .first()
            .copied()
            .filter(|operand| {
                operand.kind == OperandKind::VectorRegister && operand.value < VECTOR_REGISTER_COUNT
            })
            .ok_or_else(|| "V_PERMLANE16 source 0 must be a VGPR".to_owned())?;
        for index in 1..=2 {
            let selector = instruction
                .sources
                .get(index)
                .copied()
                .ok_or_else(|| format!("V_PERMLANE16 is missing selector source {index}"))?;
            if selector.kind == OperandKind::VectorRegister {
                return Err(format!(
                    "V_PERMLANE16 selector source {index} must be scalar"
                ));
            }
        }
        let Some(InstructionControl::Vop3(control)) = instruction.control else {
            return Err("V_PERMLANE16 requires VOP3 control".to_owned());
        };
        if control.absolute_mask != 0
            || control.negate_mask != 0
            || control.output_modifier != 0
            || control.clamp
            || control.operand_select & !3 != 0
            || control.scalar_destination.is_some()
        {
            return Err("V_PERMLANE16 uses reserved VOP3 modifier bits".to_owned());
        }

        let fetch_inactive = control.operand_select & 1 != 0;
        let bound_control = control.operand_select & 2 != 0;
        let selector_low = self.raw_source(instruction, 1)?;
        let selector_high = self.raw_source(instruction, 2)?;
        let row_lane = self.temp("uint", "remu_lane & 15u");
        let selector_word = self.temp(
            "uint",
            &format!("({row_lane} < 8u) ? {selector_low} : {selector_high}"),
        );
        let selector = self.temp(
            "uint",
            &format!("({selector_word} >> (({row_lane} & 7u) * 4u)) & 15u"),
        );
        let row = if exchange_rows {
            "(remu_lane & 16u) ^ 16u"
        } else {
            "remu_lane & 16u"
        };
        let source_lane = self.temp("uint", &format!("({row}) | {selector}"));
        // Keep both shuffles converged across the Metal SIMD group. Guest EXEC only controls the
        // architectural write and, when FI=0, whether the selected guest lane is considered live.
        let fetched = self.temp(
            "uint",
            &format!("simd_shuffle(v[{}], (ushort){source_lane})", source.value),
        );
        if fetch_inactive {
            self.store_vector(destination, &fetched);
            return Ok(());
        }

        let source_active = self.temp(
            "bool",
            &format!("simd_shuffle(exec ? 1u : 0u, (ushort){source_lane}) != 0u"),
        );
        if bound_control {
            self.store_vector(destination, &format!("{source_active} ? {fetched} : 0u"));
        } else if destination < VECTOR_REGISTER_COUNT {
            // With FI=0 and BC=0, an inactive selected source suppresses the destination write.
            self.line(
                3,
                &format!("if (exec && {source_active}) v[{destination}] = {fetched};"),
            );
        }
        Ok(())
    }

    fn emit_mad_u64_u32(&mut self, instruction: &Instruction) -> Result<(), String> {
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

    fn emit_vector_shift64(
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

    fn emit_vector_carry(&mut self, instruction: &Instruction, name: &str) -> Result<(), String> {
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

    fn emit_vector_compare(&mut self, instruction: &Instruction, name: &str) -> Result<(), String> {
        let float_source = |index| {
            if name.ends_with("F16") {
                self.half_float_source(instruction, index)
            } else {
                self.float_source(instruction, index)
            }
        };
        let condition = match classify_vector_compare(name)? {
            VectorCompare::Always(value) => value.to_string(),
            VectorCompare::Ordered => format!(
                "(!isnan({}) && !isnan({}))",
                float_source(0)?,
                float_source(1)?
            ),
            VectorCompare::Unordered => format!(
                "(isnan({}) || isnan({}))",
                float_source(0)?,
                float_source(1)?
            ),
            VectorCompare::Float { operator, negate } => {
                let comparison = format!("({} {operator} {})", float_source(0)?, float_source(1)?);
                if negate {
                    format!("!{comparison}")
                } else {
                    comparison
                }
            }
            VectorCompare::Integer { operator, signed } => {
                let left = self.raw_source(instruction, 0)?;
                let right = self.raw_source(instruction, 1)?;
                if signed {
                    format!("(as_type<int>({left}) {operator} as_type<int>({right}))")
                } else {
                    format!("(({left}) {operator} ({right}))")
                }
            }
            VectorCompare::Class => self.emit_float_class_compare(instruction)?,
        };
        let mut active = self.temp("bool", &format!("exec && ({condition})"));
        if let Some(InstructionControl::Dpp(control)) = instruction.control.as_ref() {
            let write_enabled = self.emit_dpp_write_enabled(*control)?;
            active = self.temp("bool", &format!("({write_enabled}) ? {active} : vcc"));
        }
        if name.starts_with("VCmpx") {
            self.store_exec_mask(&active);
        } else {
            let destination = match instruction.control.as_ref() {
                Some(InstructionControl::Sdwa(control)) => {
                    control.scalar_destination.unwrap_or(VCC_LO)
                }
                _ => VCC_LO,
            };
            self.store_lane_mask(destination, &active);
        }
        Ok(())
    }

    fn emit_float_class_compare(&mut self, instruction: &Instruction) -> Result<String, String> {
        let value = self.temp("float", &self.float_source(instruction, 0)?);
        let raw = self.temp("uint", &self.raw_source(instruction, 0)?);
        let mask = self.temp("uint", &self.raw_source(instruction, 1)?);
        let negative = self.temp("bool", &format!("({raw} & 0x80000000u) != 0u"));
        let nan = self.temp("bool", &format!("isnan({value})"));
        let infinite = self.temp("bool", &format!("isinf({value})"));
        let zero = self.temp("bool", &format!("{value} == 0.0f"));
        let subnormal = self.temp(
            "bool",
            &format!("fabs({value}) > 0.0f && fabs({value}) < as_type<float>(0x00800000u)"),
        );
        let normal = self.temp(
            "bool",
            &format!("!({nan} || {infinite} || {zero} || {subnormal})"),
        );
        Ok(format!(
            "((({mask} & 3u) != 0u) && {nan}) || \
             ((({mask} >> 2) & 1u) != 0u && {infinite} && {negative}) || \
             ((({mask} >> 3) & 1u) != 0u && {normal} && {negative}) || \
             ((({mask} >> 4) & 1u) != 0u && {subnormal} && {negative}) || \
             ((({mask} >> 5) & 1u) != 0u && {zero} && {negative}) || \
             ((({mask} >> 6) & 1u) != 0u && {zero} && !{negative}) || \
             ((({mask} >> 7) & 1u) != 0u && {subnormal} && !{negative}) || \
             ((({mask} >> 8) & 1u) != 0u && {normal} && !{negative}) || \
             ((({mask} >> 9) & 1u) != 0u && {infinite} && !{negative})"
        ))
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
