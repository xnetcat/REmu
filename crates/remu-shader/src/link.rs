// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Linking for separately allocated merged-shader entry points.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::{Instruction, Opcode, OperandKind, Program, ShaderEncoding};

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ProgramLinkError {
    #[error("shader front program is empty")]
    EmptyFront,
    #[error("shader continuation program is empty")]
    EmptyContinuation,
    #[error("shader front program has no S_SETPC_B64 s[6:7] continuation transfer")]
    MissingSetpc,
    #[error(
        "shader front program has {count} S_SETPC_B64 s[6:7] continuation transfers; exactly one is required"
    )]
    AmbiguousSetpc { count: usize },
    #[error(
        "S_SETPC_B64 s[6:7] at pc 0x{pc:X} is not the terminal instruction ending at pc 0x{front_end:X}"
    )]
    NonTerminalSetpc { pc: u32, front_end: u32 },
    #[error("linked shader program counter overflow")]
    ProgramCounterOverflow,
    #[error(
        "S_SETPC_B64 at pc 0x{pc:X} cannot reach linked continuation pc 0x{target:X} with a SOPP branch"
    )]
    TransferOutOfRange { pc: u32, target: u32 },
}

/// Links an AGC merged-shader front and its exact separately allocated continuation.
///
/// AGC's fuse operation programs the front allocation as the stage entry point and initializes
/// s[6:7] with the back allocation's address. The front's `S_SETPC_B64` is therefore a known
/// tail-call edge, not shader termination. The executable IR represents that edge as an ordinary
/// unconditional branch while retaining exact guest addresses for relocated continuation
/// instructions so `S_GETPC_B64` remains architectural.
pub fn link_tail_call_programs(
    front: &Program,
    continuation: &Program,
) -> Result<Program, ProgramLinkError> {
    let front_end = front
        .instructions
        .iter()
        .map(instruction_end)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or(ProgramLinkError::EmptyFront)?;
    if continuation.instructions.is_empty() {
        return Err(ProgramLinkError::EmptyContinuation);
    }
    let matching_setpc = front
        .instructions
        .iter()
        .enumerate()
        .filter(|(_, instruction)| fused_continuation_transfer(instruction))
        .collect::<Vec<_>>();
    let (setpc_index, setpc) = match matching_setpc.as_slice() {
        [] => return Err(ProgramLinkError::MissingSetpc),
        [(index, instruction)] => (*index, *instruction),
        transfers => {
            return Err(ProgramLinkError::AmbiguousSetpc {
                count: transfers.len(),
            });
        }
    };
    if instruction_end(setpc)? != front_end {
        return Err(ProgramLinkError::NonTerminalSetpc {
            pc: setpc.pc,
            front_end,
        });
    }
    let continuation_pc = front_end;
    let mut instructions =
        Vec::with_capacity(front.instructions.len() + continuation.instructions.len());
    let mut instruction_addresses = BTreeMap::new();

    for (index, instruction) in front.instructions.iter().enumerate() {
        if index == setpc_index {
            instructions.push(linked_branch(instruction, continuation_pc)?);
        } else {
            instructions.push(instruction.clone());
        }
        let architectural = front.instruction_address(instruction.pc);
        if architectural != front.address.wrapping_add(u64::from(instruction.pc)) {
            instruction_addresses.insert(instruction.pc, architectural);
        }
    }
    for instruction in &continuation.instructions {
        let mut relocated = instruction.clone();
        relocated.pc = continuation_pc
            .checked_add(instruction.pc)
            .ok_or(ProgramLinkError::ProgramCounterOverflow)?;
        instruction_addresses.insert(
            relocated.pc,
            continuation.instruction_address(instruction.pc),
        );
        instructions.push(relocated);
    }

    Ok(Program::new_with_instruction_addresses(
        front.address,
        instructions,
        instruction_addresses,
        Some(continuation.address),
    ))
}

fn fused_continuation_transfer(instruction: &Instruction) -> bool {
    instruction.opcode.is("SSetpcB64")
        && instruction
            .sources
            .first()
            .is_some_and(|source| source.kind == OperandKind::ScalarRegister && source.value == 6)
}

fn instruction_end(instruction: &Instruction) -> Result<u32, ProgramLinkError> {
    let byte_count = u32::try_from(
        instruction
            .words
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(ProgramLinkError::ProgramCounterOverflow)?,
    )
    .map_err(|_| ProgramLinkError::ProgramCounterOverflow)?;
    instruction
        .pc
        .checked_add(byte_count)
        .ok_or(ProgramLinkError::ProgramCounterOverflow)
}

fn linked_branch(instruction: &Instruction, target: u32) -> Result<Instruction, ProgramLinkError> {
    let end = instruction_end(instruction)?;
    let delta = i64::from(target) - i64::from(end);
    if delta % 4 != 0 {
        return Err(ProgramLinkError::TransferOutOfRange {
            pc: instruction.pc,
            target,
        });
    }
    let immediate = i16::try_from(delta / 4).map_err(|_| ProgramLinkError::TransferOutOfRange {
        pc: instruction.pc,
        target,
    })?;
    Ok(Instruction {
        pc: instruction.pc,
        encoding: ShaderEncoding::Sopp,
        opcode: Opcode::Known("SBranch"),
        words: vec![0xbf82_0000 | u32::from(u16::from_ne_bytes(immediate.to_ne_bytes()))],
        sources: Vec::new(),
        destinations: Vec::new(),
        control: None,
    })
}

#[cfg(test)]
mod tests {
    use crate::{InstructionControl, decode_program};

    use super::*;

    #[test]
    fn links_setpc_to_external_exports_and_preserves_getpc_address() {
        let front = decode_program(0x0080_0aed_0000, &[0xbe80_2006])
            .expect("front S_SETPC program decodes");
        let continuation = decode_program(
            0x0080_0aee_0000,
            &[
                0xbe84_1f00, // s_getpc_b64 s[4:5]
                0xf800_08cf, // exp pos0, v0-v3
                0x0302_0100,
                0xf800_0941, // exp primitive, v4
                0x0404_0404,
                0xbf81_0000,
            ],
        )
        .expect("continuation program decodes");

        let linked = link_tail_call_programs(&front, &continuation).expect("programs link");
        assert_eq!(linked.instructions[0].opcode.known_name(), Some("SBranch"));
        assert_eq!(
            linked.instructions[1].opcode.known_name(),
            Some("SGetpcB64")
        );
        assert_eq!(
            linked.instruction_address(linked.instructions[1].pc),
            continuation.address
        );
        assert_eq!(
            linked.linked_continuation_address(),
            Some(continuation.address)
        );
        let targets = linked
            .instructions
            .iter()
            .filter_map(|instruction| {
                let InstructionControl::Export(export) = instruction.control.as_ref()? else {
                    return None;
                };
                Some(export.target)
            })
            .collect::<Vec<_>>();
        assert_eq!(targets, [12, 20]);
    }

    #[test]
    fn refuses_a_front_without_an_external_transfer() {
        let front = decode_program(0x1000, &[0xbf81_0000]).unwrap();
        let continuation = decode_program(0x2000, &[0xbf81_0000]).unwrap();
        assert_eq!(
            link_tail_call_programs(&front, &continuation),
            Err(ProgramLinkError::MissingSetpc)
        );
    }

    #[test]
    fn refuses_a_setpc_not_using_the_fused_continuation_registers() {
        let front = decode_program(0x1000, &[0xbe80_2008]).unwrap();
        let continuation = decode_program(0x2000, &[0xbf81_0000]).unwrap();
        assert_eq!(
            link_tail_call_programs(&front, &continuation),
            Err(ProgramLinkError::MissingSetpc)
        );
    }

    #[test]
    fn refuses_multiple_matching_continuation_transfers() {
        let transfer = decode_program(0x1000, &[0xbe80_2006]).unwrap().instructions[0].clone();
        let mut second = transfer.clone();
        second.pc = 4;
        let front = Program::new(0x1000, vec![transfer, second]);
        let continuation = decode_program(0x2000, &[0xbf81_0000]).unwrap();
        assert_eq!(
            link_tail_call_programs(&front, &continuation),
            Err(ProgramLinkError::AmbiguousSetpc { count: 2 })
        );
    }

    #[test]
    fn refuses_a_nonterminal_continuation_transfer() {
        let transfer = decode_program(0x1000, &[0xbe80_2006]).unwrap().instructions[0].clone();
        let mut end = decode_program(0x1004, &[0xbf81_0000]).unwrap().instructions[0].clone();
        end.pc = 4;
        let front = Program::new(0x1000, vec![transfer, end]);
        let continuation = decode_program(0x2000, &[0xbf81_0000]).unwrap();
        assert_eq!(
            link_tail_call_programs(&front, &continuation),
            Err(ProgramLinkError::NonTerminalSetpc {
                pc: 0,
                front_end: 8,
            })
        );
    }
}
