// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Vector comparison lowering: VOPC predicate classification and the lane mask
//! each comparison writes.

use remu_shader::{Instruction, InstructionControl};

use crate::{Compiler, VCC_LO};

impl Compiler<'_> {
    pub(crate) fn emit_vector_compare(
        &mut self,
        instruction: &Instruction,
        name: &str,
    ) -> Result<(), String> {
        // J3: the Poppy glyph `V_READFIRSTLANE`→`V_CMPX`→`S_MOV M0`→`V_MOVRELS` serialization
        // peephole is deleted. Metal fragment simdgroups cannot prove divergent-lane survival for
        // that idiom, so detecting it is a loud refuse rather than silently skipping V_CMPX
        // (which would leave EXEC wrong) or running the architectural serialization (tofu).
        if name.starts_with("VCmpx")
            && self
                .pixel_movrels_serialization_index_vgpr(instruction)
                .is_some()
        {
            return Err(
                "pixel V_MOVRELS serialization idiom (V_READFIRSTLANE/V_CMPX/S_MOV M0/V_MOVRELS) has no proven Metal lowering; refusing the title-specific EXEC-preserving peephole"
                    .into(),
            );
        }
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
