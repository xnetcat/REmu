// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Scalar liveness analysis used to size the evaluation slice of a decoded program.

use std::collections::{BTreeMap, BTreeSet};

use super::{Instruction, InstructionControl, Opcode, OperandKind, ShaderEncoding};

pub(crate) fn instruction_requires_evaluation(instruction: &Instruction) -> bool {
    matches!(
        instruction.encoding,
        ShaderEncoding::Sop1
            | ShaderEncoding::Sop2
            | ShaderEncoding::Sopc
            | ShaderEncoding::Sopp
            | ShaderEncoding::Sopk
    ) || matches!(
        instruction.control,
        Some(
            InstructionControl::ScalarMemory(_)
                | InstructionControl::GlobalMemory(_)
                | InstructionControl::BufferMemory(_)
                | InstructionControl::Image(_)
        )
    )
}

const SCALAR_LIVE_WORDS: usize = 4;

type ScalarLiveSet = [u64; SCALAR_LIVE_WORDS];

#[derive(Clone, Copy, Default, Eq, PartialEq)]
struct EvaluationLiveSet {
    registers: ScalarLiveSet,
    scalar_condition: bool,
}

fn scalar_operand_width(instruction: &Instruction) -> u32 {
    if instruction
        .opcode
        .known_name()
        .is_some_and(|name| name.ends_with("B64") || name.ends_with("I64") || name.ends_with("U64"))
    {
        2
    } else {
        1
    }
}

fn scalar_instruction_defines_condition(instruction: &Instruction) -> bool {
    if instruction.encoding == ShaderEncoding::Sopc
        || (instruction.encoding == ShaderEncoding::Sopk && instruction.opcode.starts_with("SCmpk"))
    {
        return true;
    }
    matches!(
        instruction.opcode.known_name(),
        Some(
            "SNotB32"
                | "SBrevB32"
                | "SBcnt1I32B32"
                | "SFF1I32B32"
                | "SBcnt1I32B64"
                | "SNotB64"
                | "SWqmB64"
                | "SLshlB64"
                | "SLshrB64"
                | "SAshrI64"
                | "SBfeU64"
                | "SBfeI64"
                | "SCselectB64"
                | "SAndB64"
                | "SOrB64"
                | "SXorB64"
                | "SAndn2B64"
                | "SOrn2B64"
                | "SNandB64"
                | "SNorB64"
                | "SXnorB64"
                | "SAddU32"
                | "SSubU32"
                | "SAddI32"
                | "SSubI32"
                | "SAddcU32"
                | "SSubbU32"
                | "SMinI32"
                | "SMinU32"
                | "SMaxI32"
                | "SMaxU32"
                | "SAndB32"
                | "SOrB32"
                | "SXorB32"
                | "SAndn2B32"
                | "SOrn2B32"
                | "SNandB32"
                | "SNorB32"
                | "SXnorB32"
                | "SLshlB32"
                | "SLshrB32"
                | "SAshrI32"
                | "SBfeU32"
                | "SBfeI32"
                | "SAbsdiffI32"
                | "SLshl1AddU32"
                | "SLshl2AddU32"
                | "SLshl3AddU32"
                | "SLshl4AddU32"
                | "SAndSaveexecB32"
                | "SOrSaveexecB32"
                | "SXorSaveexecB32"
                | "SAndn1SaveexecB32"
                | "SAndn2SaveexecB32"
                | "SOrn1SaveexecB32"
                | "SOrn2SaveexecB32"
                | "SNandSaveexecB32"
                | "SNorSaveexecB32"
                | "SXnorSaveexecB32"
                | "SAndSaveexecB64"
                | "SOrSaveexecB64"
                | "SXorSaveexecB64"
                | "SAndn1SaveexecB64"
                | "SAndn2SaveexecB64"
                | "SOrn1SaveexecB64"
                | "SOrn2SaveexecB64"
                | "SNandSaveexecB64"
                | "SNorSaveexecB64"
                | "SXnorSaveexecB64"
        )
    )
}

fn scalar_instruction_reads_condition(instruction: &Instruction) -> bool {
    matches!(
        instruction.opcode.known_name(),
        Some("SCselectB32" | "SCselectB64" | "SAddcU32" | "SSubbU32")
    ) || instruction
        .sources
        .iter()
        .any(|operand| operand.kind == OperandKind::EncodedConstant && operand.value == 253)
}

fn scalar_destination_is_also_source(instruction: &Instruction) -> bool {
    matches!(
        instruction.opcode.known_name(),
        Some("SAddkI32" | "SMulkI32" | "SBitset1B32")
    )
}

#[allow(clippy::too_many_lines)]
pub(crate) fn evaluation_scalar_liveness(
    instructions: &[Instruction],
    next_evaluation_instruction_indices: &[u32],
    linked_transfer_targets: &BTreeMap<u32, u32>,
) -> (Vec<Vec<u8>>, Vec<bool>) {
    let instruction_count = instructions.len();
    let mut successors = vec![Vec::<usize>::with_capacity(2); instruction_count];

    let next_evaluation = |index: usize| {
        next_evaluation_instruction_indices
            .get(index)
            .copied()
            .filter(|index| *index != u32::MAX)
            .and_then(|index| usize::try_from(index).ok())
    };

    for (index, instruction) in instructions.iter().enumerate() {
        if !instruction_requires_evaluation(instruction) {
            continue;
        }
        let name = instruction.opcode.known_name();
        if instruction.opcode == Opcode::END_PROGRAM || matches!(name, Some("SSwappcB64")) {
            continue;
        }
        if matches!(name, Some("SSetpcB64")) {
            if let Some(target) = linked_transfer_targets.get(&instruction.pc)
                && let Ok(target_index) =
                    instructions.binary_search_by_key(target, |candidate| candidate.pc)
                && let Some(target_evaluation) = next_evaluation(target_index)
            {
                successors[index].push(target_evaluation);
            }
            continue;
        }
        let is_branch = matches!(
            name,
            Some(
                "SBranch"
                    | "SCbranchScc0"
                    | "SCbranchScc1"
                    | "SCbranchVccz"
                    | "SCbranchVccnz"
                    | "SCbranchExecz"
                    | "SCbranchExecnz"
            )
        );
        if is_branch
            && let Some(target) = evaluation_branch_target(instruction)
            && let Ok(target_index) =
                instructions.binary_search_by_key(&target, |candidate| candidate.pc)
            && let Some(target_evaluation) = next_evaluation(target_index)
        {
            successors[index].push(target_evaluation);
        }
        if !matches!(name, Some("SBranch"))
            && let Some(fallthrough) = next_evaluation(index.saturating_add(1))
            && !successors[index].contains(&fallthrough)
        {
            successors[index].push(fallthrough);
        }
    }

    let mut live_in = vec![EvaluationLiveSet::default(); instruction_count];
    loop {
        let mut changed = false;
        for index in (0..instruction_count).rev() {
            if !instruction_requires_evaluation(&instructions[index]) {
                continue;
            }
            let instruction = &instructions[index];
            let mut live_out = EvaluationLiveSet::default();
            for &successor in &successors[index] {
                for word in 0..SCALAR_LIVE_WORDS {
                    live_out.registers[word] |= live_in[successor].registers[word];
                }
                live_out.scalar_condition |= live_in[successor].scalar_condition;
            }

            let compare_k = instruction.encoding == ShaderEncoding::Sopk
                && instruction.opcode.starts_with("SCmpk");
            let destination_live = instruction.destinations.iter().any(|operand| {
                operand.kind == OperandKind::ScalarRegister
                    && scalar_live_contains(&live_out.registers, operand.value)
            });
            let condition_live =
                scalar_instruction_defines_condition(instruction) && live_out.scalar_condition;
            let result_live = destination_live || condition_live;
            let mut updated = live_out;

            if !compare_k {
                let destination_width = scalar_operand_width(instruction);
                for operand in &instruction.destinations {
                    if operand.kind == OperandKind::ScalarRegister {
                        scalar_live_remove_range(
                            &mut updated.registers,
                            operand.value,
                            destination_width,
                        );
                    }
                }
            }
            if scalar_instruction_defines_condition(instruction) {
                updated.scalar_condition = false;
            }

            match instruction.control.as_ref() {
                Some(InstructionControl::Image(image)) => {
                    scalar_live_insert_range(&mut updated.registers, image.scalar_resource, 8);
                    if instruction.opcode.starts_with("ImageSample")
                        || instruction.opcode.starts_with("ImageGather")
                    {
                        scalar_live_insert_range(&mut updated.registers, image.scalar_sampler, 4);
                    }
                }
                Some(InstructionControl::GlobalMemory(global)) => {
                    if let Some(register) = global.scalar_address {
                        scalar_live_insert_range(&mut updated.registers, register, 2);
                    }
                }
                Some(InstructionControl::BufferMemory(buffer)) => {
                    scalar_live_insert_range(&mut updated.registers, buffer.scalar_resource, 4);
                }
                Some(InstructionControl::ScalarMemory(scalar)) => {
                    if let Some(base) = instruction.sources.first()
                        && base.kind == OperandKind::ScalarRegister
                    {
                        let base_width = if instruction.opcode.starts_with("SBufferLoad") {
                            4
                        } else {
                            2
                        };
                        scalar_live_insert_range(&mut updated.registers, base.value, base_width);
                    }
                    // The offset selects loaded scalar values, but not the Metal buffer identity.
                    // It only participates in recurrence when a loaded destination can affect a
                    // later binding or scalar branch.
                    if destination_live && let Some(register) = scalar.dynamic_offset_register {
                        scalar_live_insert_range(&mut updated.registers, register, 1);
                    }
                }
                _ => {}
            }

            match instruction.opcode.known_name() {
                Some("SCbranchScc0" | "SCbranchScc1") => {
                    updated.scalar_condition = true;
                }
                Some("SCbranchVccz" | "SCbranchVccnz") => {
                    scalar_live_insert_range(&mut updated.registers, 106, 2);
                }
                Some("SCbranchExecz" | "SCbranchExecnz") => {
                    scalar_live_insert_range(&mut updated.registers, 126, 2);
                }
                _ => {}
            }

            let scalar_operation = matches!(
                instruction.encoding,
                ShaderEncoding::Sop1
                    | ShaderEncoding::Sop2
                    | ShaderEncoding::Sopc
                    | ShaderEncoding::Sopk
            );
            if scalar_operation && result_live {
                let source_width = scalar_operand_width(instruction);
                for operand in &instruction.sources {
                    if operand.kind == OperandKind::ScalarRegister {
                        scalar_live_insert_range(
                            &mut updated.registers,
                            operand.value,
                            source_width,
                        );
                    }
                }
                if compare_k || scalar_destination_is_also_source(instruction) {
                    for operand in &instruction.destinations {
                        if operand.kind == OperandKind::ScalarRegister {
                            scalar_live_insert_range(
                                &mut updated.registers,
                                operand.value,
                                source_width,
                            );
                        }
                    }
                }
                if scalar_instruction_reads_condition(instruction) {
                    updated.scalar_condition = true;
                }
            }

            if updated != live_in[index] {
                live_in[index] = updated;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let registers = live_in
        .iter()
        .map(|set| {
            (0_u32..256)
                .filter(|register| scalar_live_contains(&set.registers, *register))
                .filter_map(|register| u8::try_from(register).ok())
                .collect()
        })
        .collect();
    let scalar_conditions = live_in
        .into_iter()
        .map(|set| set.scalar_condition)
        .collect();
    (registers, scalar_conditions)
}

pub(crate) fn evaluation_branch_target(instruction: &Instruction) -> Option<u32> {
    let encoded = instruction.words.first().copied()?;
    #[allow(clippy::cast_possible_truncation)]
    let offset = (encoded as u16).cast_signed();
    let next_pc = i64::from(instruction.pc)
        .checked_add(i64::try_from(instruction.words.len().checked_mul(4)?).ok()?)?;
    let target = next_pc.checked_add(i64::from(offset).checked_mul(4)?)?;
    u32::try_from(target).ok()
}

pub(crate) fn scalar_live_insert_range(set: &mut ScalarLiveSet, first: u32, count: u32) {
    for register in first..first.saturating_add(count).min(256) {
        let word = usize::try_from(register / 64).expect("scalar register word fits usize");
        let bit = register % 64;
        set[word] |= 1_u64 << bit;
    }
}

pub(crate) fn scalar_live_contains(set: &ScalarLiveSet, register: u32) -> bool {
    let word = usize::try_from(register / 64).expect("scalar register word fits usize");
    let bit = register % 64;
    (set[word] & (1_u64 << bit)) != 0
}

pub(crate) fn scalar_live_remove_range(set: &mut ScalarLiveSet, first: u32, count: u32) {
    for register in first..first.saturating_add(count).min(256) {
        let word = usize::try_from(register / 64).expect("scalar register word fits usize");
        let bit = register % 64;
        set[word] &= !(1_u64 << bit);
    }
}

pub(crate) fn insert_scalar_range(registers: &mut BTreeSet<u32>, first: u32, count: u32) {
    for register in first..first.saturating_add(count) {
        if register < 256 {
            registers.insert(register);
        }
    }
}

pub(crate) fn insert_input_scalar_range(
    inputs: &mut BTreeSet<u32>,
    defined: &BTreeSet<u32>,
    first: u32,
    count: u32,
    linear_scalar_flow: bool,
) {
    for register in first..first.saturating_add(count) {
        if register < 256 && (!linear_scalar_flow || !defined.contains(&register)) {
            inputs.insert(register);
        }
    }
}
