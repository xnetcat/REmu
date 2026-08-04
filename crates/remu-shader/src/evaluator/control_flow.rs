// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Scalar control flow: path bookkeeping, branch targets and branch predicates.

use std::collections::HashSet;

use smallvec::SmallVec;

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
    /// This compute path is walking the CFG once for supplemental resource discovery. Back edges
    /// are forced to their fall-through successor and forward conditional edges are both explored
    /// so runtime loops and snapshot-only SCC decisions cannot hide descriptor loads.
    pub(crate) structural_rescue: bool,
    /// Backward edges already traversed by this lineage, keyed by branch pc. Forked paths inherit
    /// these counts so static alternate-edge discovery is performed once per loop lineage.
    pub(crate) backward_branch_visits: SmallVec<[(u32, u32); 4]>,
    /// Exact observable scalar recurrence at each vector-controlled back edge. Brent's cycle
    /// detector keeps this bounded while still recognizing cycles longer than one iteration.
    pub(crate) backward_branch_recurrences: SmallVec<[(u32, ScalarRecurrenceTracker); 4]>,
}

#[derive(Clone, Eq, PartialEq)]
pub(crate) struct ScalarRecurrenceState {
    registers: SmallVec<[(u8, u32); 16]>,
    exec_mask: Option<u64>,
    scalar_condition: Option<bool>,
}

#[derive(Clone)]
pub(crate) struct ScalarRecurrenceTracker {
    checkpoint: ScalarRecurrenceState,
    power: u64,
    distance: u64,
}

impl ScalarPath {
    pub(crate) fn backward_branch_visits(&self, pc: u32) -> u32 {
        self.backward_branch_visits
            .iter()
            .find(|(branch, _)| *branch == pc)
            .map_or(0, |(_, visits)| *visits)
    }

    pub(crate) fn record_backward_branch_visit(&mut self, pc: u32) {
        if let Some((_, visits)) = self
            .backward_branch_visits
            .iter_mut()
            .find(|(branch, _)| *branch == pc)
        {
            *visits = visits.saturating_add(1);
        } else {
            self.backward_branch_visits.push((pc, 1));
        }
    }

    /// Observes one latch state and returns the exact period when it recurs.
    ///
    /// This is Brent's constant-space cycle detector. The scalar evaluator is deterministic for
    /// the immutable guest-memory snapshot, so returning to the same evaluation-observable state
    /// proves every later binding and scalar branch will repeat.
    pub(crate) fn observe_backward_branch_recurrence(
        &mut self,
        pc: u32,
        state: ScalarRecurrenceState,
    ) -> Option<u64> {
        let Some((_, tracker)) = self
            .backward_branch_recurrences
            .iter_mut()
            .find(|(branch, _)| *branch == pc)
        else {
            self.backward_branch_recurrences.push((
                pc,
                ScalarRecurrenceTracker {
                    checkpoint: state,
                    power: 1,
                    distance: 0,
                },
            ));
            return None;
        };
        tracker.distance = tracker.distance.saturating_add(1);
        if tracker.checkpoint == state {
            return Some(tracker.distance);
        }
        if tracker.distance == tracker.power {
            tracker.checkpoint = state;
            tracker.power = tracker.power.saturating_mul(2);
            tracker.distance = 0;
        }
        None
    }
}

pub(crate) struct VisitedPaths {
    pub(crate) small: Vec<(u32, u64, bool)>,
    pub(crate) large: Option<HashSet<(u32, u64, bool)>>,
}

pub(crate) fn scalar_state_hash(
    registers: &[u32],
    register_indices: &[u8],
    exec_mask: u64,
    scalar_condition: bool,
    scalar_condition_live: bool,
) -> u64 {
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    let mut hash = OFFSET;
    for &index in register_indices {
        hash = (hash ^ u64::from(registers[usize::from(index)])).wrapping_mul(PRIME);
    }
    if register_indices.contains(&126) || register_indices.contains(&127) {
        hash = (hash ^ exec_mask).wrapping_mul(PRIME);
    }
    if scalar_condition_live {
        hash = (hash ^ u64::from(scalar_condition)).wrapping_mul(PRIME);
    }
    hash
}

pub(crate) fn scalar_recurrence_state(
    registers: &[u32],
    register_indices: &[u8],
    exec_mask: u64,
    scalar_condition: bool,
    scalar_condition_live: bool,
) -> ScalarRecurrenceState {
    ScalarRecurrenceState {
        registers: register_indices
            .iter()
            .map(|&index| (index, registers[usize::from(index)]))
            .collect(),
        exec_mask: (register_indices.contains(&126) || register_indices.contains(&127))
            .then_some(exec_mask),
        scalar_condition: scalar_condition_live.then_some(scalar_condition),
    }
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

    pub(crate) fn insert(&mut self, key: (u32, u64, bool)) -> bool {
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
