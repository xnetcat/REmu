// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Control-flow analysis and the choice between straight-line, structured, and dispatcher lowering.

use std::collections::{BTreeMap, BTreeSet};

use remu_shader::{
    BasicBlock, Instruction, Opcode, Program, ShaderEncoding, StructuredCfg,
    build_control_flow_graph,
};

use crate::MetalCompileError;

#[derive(Clone, Debug)]
pub(crate) enum ControlFlowLowering {
    /// Instruction indices in their exact execution order. This includes forward or backward
    /// unconditional branches, provided the resulting path is acyclic.
    Direct(Vec<usize>),
    /// Fully structured reducible control flow from [`remu_shader::ControlFlowGraph::structure`].
    /// Emitted as nested MSL `if`/`while` with no architectural PC machine and no step-cap.
    Structured {
        tree: StructuredCfg,
        blocks: Vec<BasicBlock>,
    },
    /// Conditional control flow whose every encoded edge moves forward. Each basic block can be
    /// considered once in guest-PC order, without a per-invocation loop or switch dispatcher.
    ///
    /// Prefer [`ControlFlowLowering::Structured`] when the remu-shader structurer proves a
    /// Dispatcher-free tree; this arm remains for shapes the structurer demotes.
    Forward,
    /// Irreducible or non-canonical reducible shape: the bounded architectural PC machine.
    /// Step-cap truncation remains possible on this arm — that is the documented seam until the
    /// structurer can prove every title shape (or the backend refuses irreducible shaders).
    Dispatcher,
}

/// True when `structure()` produced only Sequence/If/Loop/Block/Continue/Break — never a
/// [`StructuredCfg::Dispatcher`]. That is the anti-truncation proof for structured MSL emission.
pub(crate) fn structured_cfg_is_fully_reducible(tree: &StructuredCfg) -> bool {
    match tree {
        StructuredCfg::Block(_) | StructuredCfg::Continue | StructuredCfg::Break => true,
        StructuredCfg::Sequence(nodes) => nodes.iter().all(structured_cfg_is_fully_reducible),
        StructuredCfg::If {
            then_branch,
            else_branch,
            ..
        } => {
            structured_cfg_is_fully_reducible(then_branch)
                && else_branch
                    .as_ref()
                    .is_none_or(|branch| structured_cfg_is_fully_reducible(branch))
        }
        StructuredCfg::Loop(body) => structured_cfg_is_fully_reducible(body),
        StructuredCfg::Dispatcher { .. } => false,
    }
}

fn structured_cfg_needs_structured_emission(tree: &StructuredCfg) -> bool {
    match tree {
        StructuredCfg::If { .. } | StructuredCfg::Loop(_) => true,
        StructuredCfg::Sequence(nodes) => {
            nodes.iter().any(structured_cfg_needs_structured_emission)
        }
        StructuredCfg::Block(_) | StructuredCfg::Continue | StructuredCfg::Break => false,
        StructuredCfg::Dispatcher { .. } => false,
    }
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

/// MSL boolean expression for a scalar conditional branch's taken edge (same predicates the
/// dispatcher PC machine uses).
pub(crate) fn scalar_branch_taken_condition(name: &str) -> Result<&'static str, String> {
    Ok(match name {
        "SCbranchScc0" => "!scc",
        "SCbranchScc1" => "scc",
        "SCbranchVccz" => "(s[106] | s[107]) == 0u",
        "SCbranchVccnz" => "(s[106] | s[107]) != 0u",
        "SCbranchExecz" => "(s[126] | s[127]) == 0u",
        "SCbranchExecnz" => "(s[126] | s[127]) != 0u",
        _ => return Err(format!("not a scalar conditional branch: {name}")),
    })
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
            if let Some(target) = program.linked_transfer_target(instruction.pc)
                && !instruction_by_pc.contains_key(&target)
            {
                return Err(MetalCompileError::Instruction {
                    pc: instruction.pc,
                    opcode: instruction.opcode,
                    message: format!(
                        "linked S_SETPC target 0x{target:X} is not an instruction boundary"
                    ),
                });
            }
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

    // Prefer remu-shader's StructuredCfg when it proves a Dispatcher-free reducible tree that
    // actually needs structured emission (If/Loop). That path deletes the step-cap truncation
    // class for those shapes. Anything the structurer demotes to Dispatcher falls through to the
    // legacy Forward/Dispatcher arms below — loud in the sense that we never emit structured MSL
    // from an unproven tree (silently-wrong pixels risk).
    let cfg = build_control_flow_graph(program);
    let structured = cfg.structure();
    if structured_cfg_is_fully_reducible(&structured)
        && structured_cfg_needs_structured_emission(&structured)
    {
        return Ok(ControlFlowLowering::Structured {
            tree: structured,
            blocks: cfg.blocks().to_vec(),
        });
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
            // Irreducible / non-canonical cycle: keep the step-capped PC machine. Seam: step-cap
            // truncation is still possible here; StructuredCfg refused to prove this shape.
            return Ok(ControlFlowLowering::Dispatcher);
        }
        path.push(index);
        match instruction.opcode.known_name() {
            Some("SEndpgm") => break,
            Some("SSetpcB64") => {
                let Some(target) = program.linked_transfer_target(instruction.pc) else {
                    break;
                };
                index = instruction_by_pc[&target];
            }
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
        } else if let Some(target) = program.linked_transfer_target(instruction.pc) {
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
