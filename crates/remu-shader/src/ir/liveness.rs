// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Scalar liveness analysis used to size the evaluation slice of a decoded program.

use std::collections::BTreeSet;

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

#[allow(clippy::too_many_lines)]
pub(crate) fn evaluation_scalar_liveness(
    instructions: &[Instruction],
    next_evaluation_instruction_indices: &[u32],
) -> Vec<Vec<u8>> {
    let instruction_count = instructions.len();
    let mut uses = vec![[0_u64; SCALAR_LIVE_WORDS]; instruction_count];
    let mut definitions = vec![[0_u64; SCALAR_LIVE_WORDS]; instruction_count];
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
        for operand in &instruction.sources {
            if operand.kind == OperandKind::ScalarRegister {
                scalar_live_insert_range(&mut uses[index], operand.value, 2);
            }
        }
        match instruction.control.as_ref() {
            Some(InstructionControl::Image(image)) => {
                scalar_live_insert_range(&mut uses[index], image.scalar_resource, 8);
                scalar_live_insert_range(&mut uses[index], image.scalar_sampler, 4);
            }
            Some(InstructionControl::GlobalMemory(global)) => {
                if let Some(register) = global.scalar_address {
                    scalar_live_insert_range(&mut uses[index], register, 2);
                }
            }
            Some(InstructionControl::BufferMemory(buffer)) => {
                scalar_live_insert_range(&mut uses[index], buffer.scalar_resource, 4);
            }
            Some(InstructionControl::ScalarMemory(scalar)) => {
                if matches!(
                    instruction.encoding,
                    ShaderEncoding::Smem | ShaderEncoding::Smrd
                ) && let Some(base) = instruction.sources.first()
                    && base.kind == OperandKind::ScalarRegister
                {
                    scalar_live_insert_range(&mut uses[index], base.value, 4);
                }
                if let Some(register) = scalar.dynamic_offset_register {
                    scalar_live_insert_range(&mut uses[index], register, 2);
                }
            }
            _ => {}
        }
        match instruction.opcode.known_name() {
            Some("SCbranchVccz" | "SCbranchVccnz") => {
                scalar_live_insert_range(&mut uses[index], 106, 2);
            }
            Some("SCbranchExecz" | "SCbranchExecnz") => {
                scalar_live_insert_range(&mut uses[index], 126, 2);
            }
            _ => {}
        }
        for operand in &instruction.destinations {
            if operand.kind == OperandKind::ScalarRegister {
                scalar_live_insert_range(&mut definitions[index], operand.value, 1);
            }
        }

        let name = instruction.opcode.known_name();
        if instruction.opcode == Opcode::END_PROGRAM
            || matches!(name, Some("SSetpcB64" | "SSwappcB64"))
        {
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

    let mut live_in = vec![[0_u64; SCALAR_LIVE_WORDS]; instruction_count];
    loop {
        let mut changed = false;
        for index in (0..instruction_count).rev() {
            if !instruction_requires_evaluation(&instructions[index]) {
                continue;
            }
            let mut live_out = [0_u64; SCALAR_LIVE_WORDS];
            for &successor in &successors[index] {
                for word in 0..SCALAR_LIVE_WORDS {
                    live_out[word] |= live_in[successor][word];
                }
            }
            let mut updated = uses[index];
            for word in 0..SCALAR_LIVE_WORDS {
                updated[word] |= live_out[word] & !definitions[index][word];
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

    live_in
        .into_iter()
        .map(|set| {
            (0_u32..256)
                .filter(|register| scalar_live_contains(&set, *register))
                .filter_map(|register| u8::try_from(register).ok())
                .collect()
        })
        .collect()
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
