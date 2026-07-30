// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Scalar ALU evaluation: SOP arithmetic, scalar compares and operand reads.

use smallvec::SmallVec;

use crate::{Instruction, Operand, OperandKind, decode_inline_constant};

use super::{
    EvaluationError, RDNA_WAVE_MASK, SCALAR_REGISTER_COUNT, SCALAR_REGISTER_COUNT_U32, invalid,
};

#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn low_u16(value: u32) -> u16 {
    (value & u16::MAX as u32) as u16
}

#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn low_u32(value: u64) -> u32 {
    (value & u32::MAX as u64) as u32
}

pub(crate) fn scalar_destination(instruction: &Instruction) -> Result<u32, EvaluationError> {
    match instruction.destinations.as_slice() {
        [
            Operand {
                kind: OperandKind::ScalarRegister,
                value,
            },
        ] if *value < SCALAR_REGISTER_COUNT_U32 => Ok(*value),
        _ => Err(invalid(instruction, "scalar destination")),
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) fn execute_scalar_alu(
    instruction: &Instruction,
    program_address: u64,
    registers: &mut [u32; SCALAR_REGISTER_COUNT],
    exec_mask: &mut u64,
    scalar_condition: &mut bool,
) -> Result<(), EvaluationError> {
    let destination = scalar_destination(instruction)?;
    if instruction.opcode.is("SMovkI32") {
        registers[destination as usize] =
            i32::from(low_u16(instruction.sources[0].value).cast_signed()).cast_unsigned();
        return Ok(());
    }
    if instruction.opcode.is("SAddkI32") || instruction.opcode.is("SMulkI32") {
        let immediate =
            i32::from(low_u16(instruction.sources[0].value).cast_signed()).cast_unsigned();
        let current = registers[destination as usize];
        registers[destination as usize] = if instruction.opcode.is("SAddkI32") {
            current.wrapping_add(immediate)
        } else {
            current
                .cast_signed()
                .wrapping_mul(immediate.cast_signed())
                .cast_unsigned()
        };
        return Ok(());
    }
    if instruction.opcode.is("SGetpcB64") {
        let pc = program_address
            .wrapping_add(u64::from(instruction.pc))
            .wrapping_add(u64::try_from(instruction.words.len() * 4).unwrap_or_default());
        write_scalar_pair(registers, destination, pc, exec_mask);
        return Ok(());
    }
    if execute_save_exec(instruction, registers, exec_mask, scalar_condition)? {
        return Ok(());
    }

    if instruction.opcode.is("SBcnt1I32B64") {
        let source = evaluate_operand64(
            instruction
                .sources
                .first()
                .ok_or_else(|| invalid(instruction, "64-bit source"))?,
            registers,
            *exec_mask,
        )
        .ok_or_else(|| invalid(instruction, "64-bit source"))?;
        let result = source.count_ones();
        registers[destination as usize] = result;
        *scalar_condition = result != 0;
        return Ok(());
    }

    if matches!(
        instruction.opcode.known_name(),
        Some("SMovB64" | "SWqmB64" | "SNotB64")
    ) {
        let mut value = evaluate_operand64(
            instruction
                .sources
                .first()
                .ok_or_else(|| invalid(instruction, "64-bit source"))?,
            registers,
            *exec_mask,
        )
        .ok_or_else(|| invalid(instruction, "64-bit source"))?;
        if instruction.opcode.is("SNotB64") {
            value = !value;
            *scalar_condition = value != 0;
        } else if instruction.opcode.is("SWqmB64") {
            let quad_any =
                (value | (value >> 1) | (value >> 2) | (value >> 3)) & 0x1111_1111_1111_1111;
            value = quad_any.wrapping_mul(0xf);
            *scalar_condition = value != 0;
        }
        write_scalar_pair(registers, destination, value, exec_mask);
        return Ok(());
    }

    if matches!(
        instruction.opcode.known_name(),
        Some("SLshlB64" | "SLshrB64" | "SAshrI64")
    ) {
        let left = evaluate_operand64(
            instruction
                .sources
                .first()
                .ok_or_else(|| invalid(instruction, "64-bit source"))?,
            registers,
            *exec_mask,
        )
        .ok_or_else(|| invalid(instruction, "64-bit source"))?;
        let right = evaluate_operand(
            instruction
                .sources
                .get(1)
                .ok_or_else(|| invalid(instruction, "shift source"))?,
            registers,
        )
        .ok_or_else(|| invalid(instruction, "shift source"))?;
        let shift = right & 63;
        let result = if instruction.opcode.is("SLshlB64") {
            left << shift
        } else if instruction.opcode.is("SLshrB64") {
            left >> shift
        } else {
            (left.cast_signed() >> shift).cast_unsigned()
        };
        write_scalar_pair(registers, destination, result, exec_mask);
        *scalar_condition = result != 0;
        return Ok(());
    }

    if instruction.opcode.is("SBfeU64") || instruction.opcode.is("SBfeI64") {
        let source = evaluate_operand64(
            instruction
                .sources
                .first()
                .ok_or_else(|| invalid(instruction, "64-bit source"))?,
            registers,
            *exec_mask,
        )
        .ok_or_else(|| invalid(instruction, "64-bit source"))?;
        let control = evaluate_operand(
            instruction
                .sources
                .get(1)
                .ok_or_else(|| invalid(instruction, "bitfield control"))?,
            registers,
        )
        .ok_or_else(|| invalid(instruction, "bitfield control"))?;
        let offset = control & 63;
        let width = ((control >> 16) & 0x7f).min(64 - offset);
        let mut result = if width == 0 { 0 } else { source >> offset };
        if width != 0 && width < 64 {
            result &= u64::MAX >> (64 - width);
            if instruction.opcode.is("SBfeI64") {
                result = ((result << (64 - width)).cast_signed() >> (64 - width)).cast_unsigned();
            }
        }
        write_scalar_pair(registers, destination, result, exec_mask);
        *scalar_condition = result != 0;
        return Ok(());
    }

    if instruction.opcode.is("SBfmB64") {
        let width = evaluate_operand(
            instruction
                .sources
                .first()
                .ok_or_else(|| invalid(instruction, "bitmask width"))?,
            registers,
        )
        .ok_or_else(|| invalid(instruction, "bitmask width"))?
            & 63;
        let offset = evaluate_operand(
            instruction
                .sources
                .get(1)
                .ok_or_else(|| invalid(instruction, "bitmask offset"))?,
            registers,
        )
        .ok_or_else(|| invalid(instruction, "bitmask offset"))?
            & 63;
        let result = if width == 0 {
            0
        } else {
            (u64::MAX >> (64 - width)) << offset
        };
        write_scalar_pair(registers, destination, result, exec_mask);
        return Ok(());
    }

    if matches!(
        instruction.opcode.known_name(),
        Some(
            "SCselectB64"
                | "SAndB64"
                | "SOrB64"
                | "SXorB64"
                | "SAndn2B64"
                | "SOrn2B64"
                | "SNandB64"
                | "SNorB64"
                | "SXnorB64"
        )
    ) {
        let left = evaluate_operand64(
            instruction
                .sources
                .first()
                .ok_or_else(|| invalid(instruction, "64-bit source 0"))?,
            registers,
            *exec_mask,
        )
        .ok_or_else(|| invalid(instruction, "64-bit source 0"))?;
        let right = evaluate_operand64(
            instruction
                .sources
                .get(1)
                .ok_or_else(|| invalid(instruction, "64-bit source 1"))?,
            registers,
            *exec_mask,
        )
        .ok_or_else(|| invalid(instruction, "64-bit source 1"))?;
        let result = match instruction.opcode.known_name() {
            Some("SCselectB64") => {
                if *scalar_condition {
                    left
                } else {
                    right
                }
            }
            Some("SAndB64") => left & right,
            Some("SOrB64") => left | right,
            Some("SXorB64") => left ^ right,
            Some("SAndn2B64") => left & !right,
            Some("SOrn2B64") => left | !right,
            Some("SNandB64") => !(left & right),
            Some("SNorB64") => !(left | right),
            Some("SXnorB64") => !(left ^ right),
            _ => unreachable!(),
        };
        write_scalar_pair(registers, destination, result, exec_mask);
        *scalar_condition = result != 0;
        return Ok(());
    }

    let left = evaluate_wave_operand(
        instruction
            .sources
            .first()
            .ok_or_else(|| invalid(instruction, "scalar source 0"))?,
        registers,
        *exec_mask,
        *scalar_condition,
    )
    .ok_or_else(|| invalid(instruction, "scalar source 0"))?;
    if instruction.opcode.is("SMovB32") {
        registers[destination as usize] = left;
        return Ok(());
    }
    if matches!(
        instruction.opcode.known_name(),
        Some("SNotB32" | "SBrevB32" | "SBcnt1I32B32" | "SFF1I32B32" | "SBitset1B32")
    ) {
        let result = match instruction.opcode.known_name() {
            Some("SNotB32") => !left,
            Some("SBrevB32") => left.reverse_bits(),
            Some("SBcnt1I32B32") => left.count_ones(),
            Some("SFF1I32B32") => {
                if left == 0 {
                    u32::MAX
                } else {
                    left.trailing_zeros()
                }
            }
            Some("SBitset1B32") => registers[destination as usize] | (1 << (left & 31)),
            _ => unreachable!(),
        };
        registers[destination as usize] = result;
        if !instruction.opcode.is("SBitset1B32") {
            *scalar_condition = result != 0;
        }
        return Ok(());
    }
    let right = evaluate_wave_operand(
        instruction
            .sources
            .get(1)
            .ok_or_else(|| invalid(instruction, "scalar source 1"))?,
        registers,
        *exec_mask,
        *scalar_condition,
    )
    .ok_or_else(|| invalid(instruction, "scalar source 1"))?;
    let result = execute_scalar_binary(instruction, left, right, scalar_condition)?;
    registers[destination as usize] = result;
    Ok(())
}

#[allow(clippy::too_many_lines)]
pub(crate) fn execute_scalar_binary(
    instruction: &Instruction,
    left: u32,
    right: u32,
    scalar_condition: &mut bool,
) -> Result<u32, EvaluationError> {
    let result = match instruction.opcode.known_name() {
        Some("SAddU32") => {
            let (value, carry) = left.overflowing_add(right);
            *scalar_condition = carry;
            value
        }
        Some("SSubU32") => {
            let (value, borrow) = left.overflowing_sub(right);
            *scalar_condition = borrow;
            value
        }
        Some("SAddI32") => {
            let value = left.wrapping_add(right);
            *scalar_condition = ((left ^ value) & (right ^ value) & 0x8000_0000) != 0;
            value
        }
        Some("SSubI32") => {
            let value = left.wrapping_sub(right);
            *scalar_condition = ((left ^ right) & (left ^ value) & 0x8000_0000) != 0;
            value
        }
        Some("SAddcU32") => {
            let wide = u64::from(left) + u64::from(right) + u64::from(*scalar_condition);
            *scalar_condition = wide > u64::from(u32::MAX);
            low_u32(wide)
        }
        Some("SSubbU32") => {
            let subtrahend = u64::from(right) + u64::from(*scalar_condition);
            *scalar_condition = subtrahend > u64::from(left);
            left.wrapping_sub(low_u32(subtrahend))
        }
        Some("SMinI32") => {
            *scalar_condition = left.cast_signed() < right.cast_signed();
            left.cast_signed().min(right.cast_signed()).cast_unsigned()
        }
        Some("SMinU32") => {
            *scalar_condition = left < right;
            left.min(right)
        }
        Some("SMaxI32") => {
            *scalar_condition = left.cast_signed() > right.cast_signed();
            left.cast_signed().max(right.cast_signed()).cast_unsigned()
        }
        Some("SMaxU32") => {
            *scalar_condition = left > right;
            left.max(right)
        }
        Some("SCselectB32") => {
            if *scalar_condition {
                left
            } else {
                right
            }
        }
        Some("SAndB32") => {
            let value = left & right;
            *scalar_condition = value != 0;
            value
        }
        Some("SOrB32") => {
            let value = left | right;
            *scalar_condition = value != 0;
            value
        }
        Some("SXorB32") => {
            let value = left ^ right;
            *scalar_condition = value != 0;
            value
        }
        Some("SAndn2B32") => {
            let value = left & !right;
            *scalar_condition = value != 0;
            value
        }
        Some("SOrn2B32") => {
            let value = left | !right;
            *scalar_condition = value != 0;
            value
        }
        Some("SNandB32") => {
            let value = !(left & right);
            *scalar_condition = value != 0;
            value
        }
        Some("SNorB32") => {
            let value = !(left | right);
            *scalar_condition = value != 0;
            value
        }
        Some("SXnorB32") => {
            let value = !(left ^ right);
            *scalar_condition = value != 0;
            value
        }
        Some("SLshlB32") => {
            let value = left << (right & 31);
            *scalar_condition = value != 0;
            value
        }
        Some("SLshrB32") => {
            let value = left >> (right & 31);
            *scalar_condition = value != 0;
            value
        }
        Some("SAshrI32") => {
            let value = (left.cast_signed() >> (right & 31)).cast_unsigned();
            *scalar_condition = value != 0;
            value
        }
        Some("SBfmB32") => {
            let width = left & 31;
            let offset = right & 31;
            if width == 0 {
                0
            } else {
                (u32::MAX >> (32 - width)) << offset
            }
        }
        Some("SMulI32") => left
            .cast_signed()
            .wrapping_mul(right.cast_signed())
            .cast_unsigned(),
        Some("SBfeU32") => {
            let offset = right & 31;
            let width = ((right >> 16) & 0x7f).min(32 - offset);
            let value = if width == 0 {
                0
            } else {
                (left >> offset) & (u32::MAX >> (32 - width))
            };
            *scalar_condition = value != 0;
            value
        }
        Some("SBfeI32") => {
            let offset = right & 31;
            let width = ((right >> 16) & 0x7f).min(32 - offset);
            let value = if width == 0 {
                0
            } else {
                ((left << (32 - width - offset)).cast_signed() >> (32 - width)).cast_unsigned()
            };
            *scalar_condition = value != 0;
            value
        }
        Some("SAbsdiffI32") => {
            // The ISA performs both the subtraction and conditional negation at
            // i32 width. This intentionally leaves INT_MIN negative and makes
            // INT_MIN - 1 wrap to INT_MAX.
            let difference = left.cast_signed().wrapping_sub(right.cast_signed());
            let value = if difference < 0 {
                difference.wrapping_neg()
            } else {
                difference
            }
            .cast_unsigned();
            *scalar_condition = value != 0;
            value
        }
        Some("SLshl1AddU32" | "SLshl2AddU32" | "SLshl3AddU32" | "SLshl4AddU32") => {
            let shift = match instruction.opcode.known_name() {
                Some("SLshl1AddU32") => 1,
                Some("SLshl2AddU32") => 2,
                Some("SLshl3AddU32") => 3,
                _ => 4,
            };
            let wide = (u64::from(left) << shift) + u64::from(right);
            *scalar_condition = wide > u64::from(u32::MAX);
            low_u32(wide)
        }
        Some("SPackLlB32B16") => (left & 0xffff) | ((right & 0xffff) << 16),
        Some("SPackLhB32B16") => (left & 0xffff) | (right & 0xffff_0000),
        Some("SPackHhB32B16") => (left >> 16) | (right & 0xffff_0000),
        Some("SMulHiU32") => ((u64::from(left) * u64::from(right)) >> 32) as u32,
        Some("SMulHiI32") => low_u32(
            ((i64::from(left.cast_signed()) * i64::from(right.cast_signed())) >> 32)
                .cast_unsigned(),
        ),
        _ => {
            return Err(EvaluationError::UnsupportedScalarOpcode {
                pc: instruction.pc,
                opcode: instruction.opcode,
            });
        }
    };
    Ok(result)
}

pub(crate) fn execute_save_exec(
    instruction: &Instruction,
    registers: &mut [u32; SCALAR_REGISTER_COUNT],
    exec_mask: &mut u64,
    scalar_condition: &mut bool,
) -> Result<bool, EvaluationError> {
    if instruction
        .opcode
        .known_name()
        .is_some_and(|name| name.ends_with("SaveexecB32"))
    {
        let destination = scalar_destination(instruction)?;
        let source = evaluate_operand(
            instruction
                .sources
                .first()
                .ok_or_else(|| invalid(instruction, "saveexec source"))?,
            registers,
        )
        .ok_or_else(|| invalid(instruction, "saveexec source"))?;
        let old = low_u32(*exec_mask);
        let new = match instruction.opcode.known_name() {
            Some("SAndSaveexecB32") => old & source,
            Some("SOrSaveexecB32") => old | source,
            Some("SXorSaveexecB32") => old ^ source,
            Some("SAndn1SaveexecB32") => !source & old,
            Some("SAndn2SaveexecB32") => source & !old,
            Some("SOrn1SaveexecB32") => !source | old,
            Some("SOrn2SaveexecB32") => source | !old,
            Some("SNandSaveexecB32") => !(source & old),
            Some("SNorSaveexecB32") => !(source | old),
            Some("SXnorSaveexecB32") => !(old ^ source),
            _ => {
                return Err(EvaluationError::UnsupportedScalarOpcode {
                    pc: instruction.pc,
                    opcode: instruction.opcode,
                });
            }
        };
        registers[destination as usize] = old;
        *exec_mask = u64::from(new);
        registers[126] = new;
        registers[127] = 0;
        *scalar_condition = new != 0;
        return Ok(true);
    }
    if !matches!(
        instruction.opcode.known_name(),
        Some(
            "SAndSaveexecB64"
                | "SOrSaveexecB64"
                | "SXorSaveexecB64"
                | "SAndn2SaveexecB64"
                | "SOrn2SaveexecB64"
                | "SNandSaveexecB64"
                | "SNorSaveexecB64"
                | "SXnorSaveexecB64"
                | "SAndn1SaveexecB64"
                | "SOrn1SaveexecB64"
        )
    ) {
        return Ok(false);
    }
    let destination = scalar_destination(instruction)?;
    let source = evaluate_operand64(
        instruction
            .sources
            .first()
            .ok_or_else(|| invalid(instruction, "saveexec source"))?,
        registers,
        *exec_mask,
    )
    .ok_or_else(|| invalid(instruction, "saveexec source"))?;
    let old = *exec_mask;
    let new = match instruction.opcode.known_name() {
        Some("SAndSaveexecB64") => old & source,
        Some("SOrSaveexecB64") => old | source,
        Some("SXorSaveexecB64") => old ^ source,
        Some("SAndn1SaveexecB64") => !source & old,
        Some("SAndn2SaveexecB64") => source & !old,
        Some("SOrn1SaveexecB64") => !source | old,
        Some("SOrn2SaveexecB64") => source | !old,
        Some("SNandSaveexecB64") => !(source & old),
        Some("SNorSaveexecB64") => !(source | old),
        Some("SXnorSaveexecB64") => !(old ^ source),
        _ => {
            return Err(EvaluationError::UnsupportedScalarOpcode {
                pc: instruction.pc,
                opcode: instruction.opcode,
            });
        }
    } & RDNA_WAVE_MASK;
    write_scalar_pair(registers, destination, old, exec_mask);
    *exec_mask = new;
    write_scalar_pair(registers, 126, new, exec_mask);
    *scalar_condition = new != 0;
    Ok(true)
}

#[allow(clippy::trivially_copy_pass_by_ref)]
pub(crate) fn evaluate_operand(
    operand: &Operand,
    registers: &[u32; SCALAR_REGISTER_COUNT],
) -> Option<u32> {
    match operand.kind {
        OperandKind::ScalarRegister => registers.get(operand.value as usize).copied(),
        OperandKind::LiteralConstant => Some(operand.value),
        OperandKind::EncodedConstant => decode_inline_constant(operand.value),
        OperandKind::VectorRegister => None,
    }
}

#[allow(clippy::trivially_copy_pass_by_ref)]
pub(crate) fn evaluate_wave_operand(
    operand: &Operand,
    registers: &[u32; SCALAR_REGISTER_COUNT],
    exec_mask: u64,
    scalar_condition: bool,
) -> Option<u32> {
    if operand.kind == OperandKind::EncodedConstant {
        match operand.value {
            251 => return Some(u32::from(registers[106] == 0 && registers[107] == 0)),
            252 => return Some(u32::from(exec_mask == 0)),
            253 => return Some(u32::from(scalar_condition)),
            _ => {}
        }
    }
    evaluate_operand(operand, registers)
}

#[allow(clippy::trivially_copy_pass_by_ref)]
pub(crate) fn evaluate_operand64(
    operand: &Operand,
    registers: &[u32; SCALAR_REGISTER_COUNT],
    exec_mask: u64,
) -> Option<u64> {
    if operand.kind == OperandKind::ScalarRegister && operand.value == 126 {
        return Some(exec_mask);
    }
    if operand.kind == OperandKind::ScalarRegister {
        let index = usize::try_from(operand.value).ok()?;
        let high = index.checked_add(1)?;
        return Some(u64::from(*registers.get(index)?) | (u64::from(*registers.get(high)?) << 32));
    }
    let low = evaluate_operand(operand, registers)?;
    Some(
        if operand.kind == OperandKind::EncodedConstant && (193..=208).contains(&operand.value) {
            u64::from(low) | (u64::from(u32::MAX) << 32)
        } else {
            u64::from(low)
        },
    )
}

pub(crate) fn write_scalar_pair(
    registers: &mut [u32; SCALAR_REGISTER_COUNT],
    destination: u32,
    mut value: u64,
    exec_mask: &mut u64,
) {
    let Ok(index) = usize::try_from(destination) else {
        return;
    };
    let Some(high_index) = index
        .checked_add(1)
        .filter(|index| *index < registers.len())
    else {
        return;
    };
    if destination == 126 {
        value &= RDNA_WAVE_MASK;
    }
    registers[index] = low_u32(value);
    registers[high_index] = low_u32(value >> 32);
    if destination == 126 {
        *exec_mask = value;
    }
}

pub(crate) fn execute_scalar_compare(
    instruction: &Instruction,
    registers: &[u32; SCALAR_REGISTER_COUNT],
    exec_mask: u64,
) -> Result<bool, EvaluationError> {
    let [left_operand, right_operand] = instruction.sources.as_slice() else {
        return Err(invalid(instruction, "scalar compare operands"));
    };
    if matches!(
        instruction.opcode.known_name(),
        Some("SCmpEqU64" | "SCmpLgU64")
    ) {
        let left = evaluate_operand64(left_operand, registers, exec_mask)
            .ok_or_else(|| invalid(instruction, "64-bit scalar compare source 0"))?;
        let right = evaluate_operand64(right_operand, registers, exec_mask)
            .ok_or_else(|| invalid(instruction, "64-bit scalar compare source 1"))?;
        return Ok(if instruction.opcode.is("SCmpEqU64") {
            left == right
        } else {
            left != right
        });
    }
    let left = evaluate_operand(left_operand, registers)
        .ok_or_else(|| invalid(instruction, "scalar compare source 0"))?;
    let right = evaluate_operand(right_operand, registers)
        .ok_or_else(|| invalid(instruction, "scalar compare source 1"))?;
    let result = match instruction.opcode.known_name() {
        Some("SBitcmp0B32" | "SBitcmp1B32") => {
            let set = (left >> (right & 31)) & 1 != 0;
            if instruction.opcode.is("SBitcmp1B32") {
                set
            } else {
                !set
            }
        }
        Some("SBitcmp0B64" | "SBitcmp1B64") => {
            let wide = evaluate_operand64(left_operand, registers, exec_mask)
                .ok_or_else(|| invalid(instruction, "64-bit compare source"))?;
            let set = (wide >> (right & 63)) & 1 != 0;
            if instruction.opcode.is("SBitcmp1B64") {
                set
            } else {
                !set
            }
        }
        Some("SCmpEqI32") => left.cast_signed() == right.cast_signed(),
        Some("SCmpLgI32") => left.cast_signed() != right.cast_signed(),
        Some("SCmpGtI32") => left.cast_signed() > right.cast_signed(),
        Some("SCmpGeI32") => left.cast_signed() >= right.cast_signed(),
        Some("SCmpLtI32") => left.cast_signed() < right.cast_signed(),
        Some("SCmpLeI32") => left.cast_signed() <= right.cast_signed(),
        Some("SCmpEqU32") => left == right,
        Some("SCmpLgU32") => left != right,
        Some("SCmpGtU32") => left > right,
        Some("SCmpGeU32") => left >= right,
        Some("SCmpLtU32") => left < right,
        Some("SCmpLeU32") => left <= right,
        _ => {
            return Err(EvaluationError::UnsupportedScalarOpcode {
                pc: instruction.pc,
                opcode: instruction.opcode,
            });
        }
    };
    Ok(result)
}

pub(crate) fn execute_scalar_compare_k(
    instruction: &Instruction,
    registers: &[u32; SCALAR_REGISTER_COUNT],
) -> Result<bool, EvaluationError> {
    let destination = scalar_destination(instruction)? as usize;
    let left = registers[destination];
    let right = i32::from(
        low_u16(
            instruction
                .sources
                .first()
                .ok_or_else(|| invalid(instruction, "compare-k source"))?
                .value,
        )
        .cast_signed(),
    )
    .cast_unsigned();
    let result = match instruction.opcode.known_name() {
        Some("SCmpkEqI32") => left.cast_signed() == right.cast_signed(),
        Some("SCmpkLgI32") => left.cast_signed() != right.cast_signed(),
        Some("SCmpkGtI32") => left.cast_signed() > right.cast_signed(),
        Some("SCmpkGeI32") => left.cast_signed() >= right.cast_signed(),
        Some("SCmpkLtI32") => left.cast_signed() < right.cast_signed(),
        Some("SCmpkLeI32") => left.cast_signed() <= right.cast_signed(),
        Some("SCmpkEqU32") => left == right,
        Some("SCmpkLgU32") => left != right,
        Some("SCmpkGtU32") => left > right,
        Some("SCmpkGeU32") => left >= right,
        Some("SCmpkLtU32") => left < right,
        Some("SCmpkLeU32") => left <= right,
        _ => {
            return Err(EvaluationError::UnsupportedScalarOpcode {
                pc: instruction.pc,
                opcode: instruction.opcode,
            });
        }
    };
    Ok(result)
}

pub(crate) fn read_scalar_pair(
    registers: &[u32; SCALAR_REGISTER_COUNT],
    scalar_base: u32,
) -> Option<u64> {
    let index = usize::try_from(scalar_base).ok()?;
    Some(
        u64::from(*registers.get(index)?)
            | (u64::from(*registers.get(index.checked_add(1)?)?) << 32),
    )
}

pub(crate) fn copy_registers<const N: usize>(
    registers: &[u32; SCALAR_REGISTER_COUNT],
    start: u32,
    count: usize,
) -> Option<SmallVec<[u32; N]>>
where
    [u32; N]: smallvec::Array<Item = u32>,
{
    let start = usize::try_from(start).ok()?;
    let end = start.checked_add(count)?;
    Some(SmallVec::from_slice(registers.get(start..end)?))
}
