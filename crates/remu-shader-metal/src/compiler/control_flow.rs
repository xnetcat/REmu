// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Control-flow analysis and the choice between straight-line and dispatcher lowering.

use std::collections::{BTreeMap, BTreeSet};

use remu_shader::{Instruction, Opcode, Program, ShaderEncoding};

use crate::MetalCompileError;

#[derive(Clone, Debug)]
pub(crate) enum ControlFlowLowering {
    /// Instruction indices in their exact execution order. This includes forward or backward
    /// unconditional branches, provided the resulting path is acyclic.
    Direct(Vec<usize>),
    /// Conditional control flow whose every encoded edge moves forward. Each basic block can be
    /// considered once in guest-PC order, without a per-invocation loop or switch dispatcher.
    Forward,
    /// Conditional control flow or a cycle still requires the bounded architectural PC machine.
    Dispatcher,
}

/// One opcode a lowering attempt could not translate. Failures are kept per distinct opcode, so a
/// shader that uses an unimplemented opcode hundreds of times still reports it once.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LoweringFailure {
    pub(crate) pc: u32,
    pub(crate) opcode: Opcode,
    pub(crate) message: String,
}

/// Combines every opcode one shader failed to lower into a single error. The first failure keeps
/// the shape it would have had alone, and the remaining opcodes are appended so a translation gap
/// can be closed as a batch instead of one launch per missing opcode.
pub(crate) fn combined_lowering_error(failures: Vec<LoweringFailure>) -> Option<MetalCompileError> {
    let mut failures = failures.into_iter();
    let first = failures.next()?;
    let remaining = failures
        .map(|failure| {
            format!(
                "pc 0x{:X} {}: {}",
                failure.pc, failure.opcode, failure.message
            )
        })
        .collect::<Vec<_>>();
    let message = if remaining.is_empty() {
        first.message
    } else {
        format!(
            "{}; the same shader also fails at {}",
            first.message,
            remaining.join(", ")
        )
    };
    Some(MetalCompileError::Instruction {
        pc: first.pc,
        opcode: first.opcode,
        message,
    })
}

pub(crate) fn branch_target(instruction: &Instruction) -> Result<u32, String> {
    if instruction.encoding != ShaderEncoding::Sopp || instruction.words.len() != 1 {
        return Err("scalar branch is not one complete SOPP instruction".to_owned());
    }
    let word = instruction
        .words
        .first()
        .copied()
        .ok_or_else(|| "branch has no encoding word".to_owned())?;
    let immediate = u16::try_from(word & 0xffff)
        .expect("masked branch immediate fits u16")
        .cast_signed();
    let end = i64::from(instruction.pc)
        + i64::try_from(instruction.words.len() * 4)
            .map_err(|_| "branch instruction size overflow".to_owned())?;
    let target = end + i64::from(immediate) * 4;
    u32::try_from(target).map_err(|_| "branch target is outside the Gen5 PC range".to_owned())
}

pub(crate) fn analyze_control_flow(
    program: &Program,
) -> Result<ControlFlowLowering, MetalCompileError> {
    let mut instruction_by_pc = BTreeMap::new();
    for (index, instruction) in program.instructions.iter().enumerate() {
        if instruction_by_pc.insert(instruction.pc, index).is_some() {
            return Err(MetalCompileError::InvalidInterface(format!(
                "duplicate instruction pc 0x{:X}",
                instruction.pc
            )));
        }
    }

    // Validate every encoded branch, including unreachable code. An invalid target previously
    // fell through the dispatcher's default arm and silently terminated the shader.
    for instruction in &program.instructions {
        if !is_scalar_branch(instruction.opcode) {
            continue;
        }
        let target =
            branch_target(instruction).map_err(|message| MetalCompileError::Instruction {
                pc: instruction.pc,
                opcode: instruction.opcode,
                message,
            })?;
        if !instruction_by_pc.contains_key(&target) {
            return Err(MetalCompileError::Instruction {
                pc: instruction.pc,
                opcode: instruction.opcode,
                message: format!("branch target 0x{target:X} is not an instruction boundary"),
            });
        }
    }

    let forward_only = program
        .instructions
        .iter()
        .filter(|instruction| is_scalar_branch(instruction.opcode))
        .all(|instruction| branch_target(instruction).is_ok_and(|target| target > instruction.pc));
    let mut path = Vec::new();
    let mut visited = BTreeSet::new();
    let mut index = 0_usize;
    while let Some(instruction) = program.instructions.get(index) {
        if !visited.insert(index) {
            return Ok(ControlFlowLowering::Dispatcher);
        }
        path.push(index);
        match instruction.opcode.known_name() {
            Some("SEndpgm" | "SSetpcB64") => break,
            Some("SBranch") => {
                let target = branch_target(instruction).expect("all branch targets were validated");
                index = instruction_by_pc[&target];
            }
            Some(name) if name.starts_with("SCbranch") => {
                return Ok(if forward_only {
                    ControlFlowLowering::Forward
                } else {
                    ControlFlowLowering::Dispatcher
                });
            }
            _ => {
                index += 1;
            }
        }
    }
    Ok(ControlFlowLowering::Direct(path))
}

pub(crate) fn is_scalar_branch(opcode: Opcode) -> bool {
    opcode
        .known_name()
        .is_some_and(|name| name == "SBranch" || name.starts_with("SCbranch"))
}

pub(crate) fn is_terminal_transfer(opcode: Opcode) -> bool {
    matches!(opcode.known_name(), Some("SEndpgm" | "SSetpcB64"))
}

pub(crate) fn dispatcher_basic_blocks(program: &Program) -> Vec<std::ops::Range<usize>> {
    let mut instruction_by_pc = BTreeMap::new();
    for (index, instruction) in program.instructions.iter().enumerate() {
        instruction_by_pc.insert(instruction.pc, index);
    }

    let mut leaders = BTreeSet::from([0_usize]);
    for (index, instruction) in program.instructions.iter().enumerate() {
        let terminator =
            is_scalar_branch(instruction.opcode) || is_terminal_transfer(instruction.opcode);
        if terminator && index + 1 < program.instructions.len() {
            leaders.insert(index + 1);
        }
        if is_scalar_branch(instruction.opcode) {
            let target = branch_target(instruction)
                .expect("dispatcher block construction follows validated control flow");
            leaders.insert(instruction_by_pc[&target]);
        }
    }

    let leaders = leaders.into_iter().collect::<Vec<_>>();
    leaders
        .iter()
        .copied()
        .enumerate()
        .map(|(index, start)| {
            let end = leaders
                .get(index + 1)
                .copied()
                .unwrap_or(program.instructions.len());
            start..end
        })
        .collect()
}
