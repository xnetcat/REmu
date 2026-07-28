// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Lane and permute lowering: DPP16/DPP8 source selection and row masking, and
//! the RDNA2 PERMLANE16 cross-lane exchange.

use remu_shader::{Dpp8Control, DppControl, Instruction, InstructionControl, OperandKind};

use crate::compiler::vector::destination_vector;
use crate::{Compiler, VECTOR_REGISTER_COUNT};

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

    pub(crate) fn emit_dpp_write_enabled(&mut self, control: DppControl) -> Result<String, String> {
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

    pub(crate) fn emit_permlane16(
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
}
