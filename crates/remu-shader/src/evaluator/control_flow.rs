// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Scalar control flow: path bookkeeping, branch targets and branch predicates.

use std::collections::HashSet;

use crate::{Instruction, Opcode};

use super::scalar_alu::low_u16;
use super::{EvaluationError, SCALAR_REGISTER_COUNT};

#[derive(Clone)]
pub(crate) struct ScalarPath {
    pub(crate) instruction_index: usize,
    pub(crate) registers: [u32; SCALAR_REGISTER_COUNT],
    pub(crate) exec_mask: u64,
    pub(crate) scalar_condition: bool,
    pub(crate) supplemental: bool,
}

pub(crate) struct VisitedPaths {
    pub(crate) small: Vec<(u32, u64)>,
    pub(crate) large: Option<HashSet<(u32, u64)>>,
}

pub(crate) fn scalar_state_hash(
    registers: &[u32],
    register_indices: &[u8],
    exec_mask: u64,
    scalar_condition: bool,
) -> u64 {
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    let mut hash = OFFSET;
    for &index in register_indices {
        hash = (hash ^ u64::from(registers[usize::from(index)])).wrapping_mul(PRIME);
    }
    hash = (hash ^ exec_mask).wrapping_mul(PRIME);
    (hash ^ u64::from(scalar_condition)).wrapping_mul(PRIME)
}

pub(crate) fn branch_target(instruction: &Instruction) -> Result<u32, EvaluationError> {
    let offset = low_u16(instruction.words[0]).cast_signed();
    let next_pc =
        i64::from(instruction.pc) + i64::try_from(instruction.words.len() * 4).unwrap_or(i64::MAX);
    let target = next_pc + i64::from(offset) * 4;
    u32::try_from(target).map_err(|_| EvaluationError::InvalidBranchTarget {
        pc: instruction.pc,
        target: if target < 0 { 0 } else { u32::MAX },
    })
}

pub(crate) fn branch_taken(
    opcode: Opcode,
    registers: &[u32; SCALAR_REGISTER_COUNT],
    exec_mask: u64,
    scalar_condition: bool,
) -> Result<bool, EvaluationError> {
    let taken = match opcode.known_name() {
        Some("SBranch") => true,
        Some("SCbranchScc0") => !scalar_condition,
        Some("SCbranchScc1") => scalar_condition,
        Some("SCbranchVccz") => registers[106] == 0 && registers[107] == 0,
        Some("SCbranchVccnz") => registers[106] != 0 || registers[107] != 0,
        Some("SCbranchExecz") => exec_mask == 0,
        Some("SCbranchExecnz") => exec_mask != 0,
        _ => {
            return Err(EvaluationError::InvalidInstruction {
                pc: 0,
                opcode,
                detail: "scalar branch opcode",
            });
        }
    };
    Ok(taken)
}

pub(crate) fn is_scalar_branch(opcode: Opcode) -> bool {
    matches!(
        opcode.known_name(),
        Some(
            "SBranch"
                | "SCbranchScc0"
                | "SCbranchScc1"
                | "SCbranchVccz"
                | "SCbranchVccnz"
                | "SCbranchExecz"
                | "SCbranchExecnz"
        )
    )
}

pub(crate) fn branch_condition_depends_on_vector_state(opcode: Opcode) -> bool {
    matches!(
        opcode.known_name(),
        Some("SCbranchVccz" | "SCbranchVccnz" | "SCbranchExecz" | "SCbranchExecnz")
    )
}

impl VisitedPaths {
    pub(crate) fn new() -> Self {
        Self {
            small: Vec::with_capacity(8),
            large: None,
        }
    }

    pub(crate) fn insert(&mut self, key: (u32, u64)) -> bool {
        if let Some(large) = self.large.as_mut() {
            return large.insert(key);
        }
        if self.small.contains(&key) {
            return false;
        }
        if self.small.len() < 16 {
            self.small.push(key);
            return true;
        }
        let mut large = HashSet::with_capacity(32);
        large.extend(self.small.drain(..));
        let inserted = large.insert(key);
        self.large = Some(large);
        inserted
    }
}
