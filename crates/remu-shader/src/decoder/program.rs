// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Program-level decoding: stream walking, segment discovery and guest-memory reads.

use crate::ir::{Instruction, Opcode, Program, ShaderEncoding};
use crate::metadata::GuestMemoryReader;

use super::{DecodeError, FALLBACK_MAX_INSTRUCTIONS, create_instruction, decode_frame};

/// Decodes one bounded Gen5/RDNA shader stream. The input must reach an architectural terminal
/// control transfer that is not before any statically encoded forward branch target; trailing
/// allocation contents are ignored.
pub fn decode_program(address: u64, words: &[u32]) -> Result<Program, DecodeError> {
    if words.is_empty() {
        return Err(DecodeError::Empty);
    }

    let mut instructions = Vec::new();
    let mut word_index = 0usize;
    let mut furthest_forward_branch_target = None;
    while word_index < words.len() {
        let pc = u32::try_from(
            word_index
                .checked_mul(4)
                .ok_or(DecodeError::ProgramCounterOverflow)?,
        )
        .map_err(|_| DecodeError::ProgramCounterOverflow)?;
        let remaining = words.get(word_index..).ok_or(DecodeError::Unterminated {
            inspected_instructions: instructions.len(),
        })?;
        if remaining.is_empty() {
            return Err(DecodeError::Unterminated {
                inspected_instructions: instructions.len(),
            });
        }
        let frame = decode_frame(pc, remaining)?;
        if remaining.len() < frame.dwords {
            return Err(DecodeError::Truncated {
                pc,
                needed_dwords: frame.dwords,
                available_dwords: remaining.len(),
            });
        }
        let instruction_words = remaining[..frame.dwords].to_vec();
        let instruction = create_instruction(pc, frame.encoding, frame.opcode, instruction_words);
        if let Some(target) = forward_branch_target(&instruction)? {
            furthest_forward_branch_target = Some(
                furthest_forward_branch_target.map_or(target, |current: u32| current.max(target)),
            );
        }
        word_index += frame.dwords;
        let ended = instruction_terminates_program(&instruction);
        instructions.push(instruction);
        if ended && furthest_forward_branch_target.is_none_or(|target| pc >= target) {
            let program = Program::new(address, instructions);
            crate::coverage::record_program(&program);
            return Ok(program);
        }
    }
    Err(DecodeError::Unterminated {
        inspected_instructions: instructions.len(),
    })
}

/// Decodes every adjacent entry point within one exact AGC shader-code allocation.
///
/// Geometry and other merged Gen5 binaries can contain multiple `S_ENDPGM`-delimited programs in
/// the same allocation. `decode_program` intentionally returns the first entry point; this variant
/// preserves those adjacent segments so callers can lower the complete merged stage instead of
/// silently discarding its trailing producer/consumer half. An indirect `S_SETPC_B64` transfer
/// ends the allocation's statically decodable stream rather than delimiting an adjacent entry
/// point.
pub fn decode_program_segments(address: u64, words: &[u32]) -> Result<Vec<Program>, DecodeError> {
    if words.is_empty() {
        return Err(DecodeError::Empty);
    }

    let mut segments = Vec::new();
    let mut word_index = 0usize;
    while word_index < words.len() {
        // AGC code extents are dword aligned but some toolchains retain zero-filled allocation
        // padding after the final entry point. It carries no executable program.
        if words[word_index..].iter().all(|word| *word == 0) {
            break;
        }
        let byte_offset = word_index
            .checked_mul(4)
            .ok_or(DecodeError::ProgramCounterOverflow)?;
        let segment_address = address
            .checked_add(
                u64::try_from(byte_offset).map_err(|_| DecodeError::ProgramCounterOverflow)?,
            )
            .ok_or(DecodeError::AddressOverflow {
                pc: u32::try_from(byte_offset).unwrap_or(u32::MAX),
            })?;
        let segment = decode_program(segment_address, &words[word_index..])?;
        let last = segment.instructions.last().ok_or(DecodeError::Empty)?;
        let segment_dwords = usize::try_from(last.pc / 4)
            .map_err(|_| DecodeError::ProgramCounterOverflow)?
            .checked_add(last.words.len())
            .ok_or(DecodeError::ProgramCounterOverflow)?;
        if segment_dwords == 0 {
            return Err(DecodeError::Unterminated {
                inspected_instructions: 0,
            });
        }
        word_index = word_index
            .checked_add(segment_dwords)
            .ok_or(DecodeError::ProgramCounterOverflow)?;
        let transferred_indirectly = last.opcode.is("SSetpcB64");
        segments.push(segment);
        if transferred_indirectly {
            break;
        }
    }

    if segments.is_empty() {
        Err(DecodeError::Empty)
    } else {
        Ok(segments)
    }
}

/// Reads and decodes a program through a fault-reporting guest-memory interface. Only the words
/// required by each decoded frame are touched, so an `S_ENDPGM` at the end of a mapping does not
/// cause speculative reads into the following page.
pub fn read_program(memory: &impl GuestMemoryReader, address: u64) -> Result<Program, DecodeError> {
    read_program_with_limits(memory, address, FALLBACK_MAX_INSTRUCTIONS, usize::MAX)
}

/// Reads a shader whose exact AGC code extent is known from its registered binary header.
///
/// Modern game shaders can legitimately exceed the decoder's conservative fallback scan. The
/// binary-provided size is the authoritative boundary: it permits every instruction in that
/// allocation while still preventing the decoder from walking into the next guest object when a
/// corrupt shader has no `S_ENDPGM`.
pub fn read_program_bounded(
    memory: &impl GuestMemoryReader,
    address: u64,
    code_size_bytes: usize,
) -> Result<Program, DecodeError> {
    if code_size_bytes == 0 || !code_size_bytes.is_multiple_of(4) {
        return Err(DecodeError::InvalidCodeSize {
            bytes: code_size_bytes,
        });
    }
    let code_dwords = code_size_bytes / 4;
    read_program_with_limits(memory, address, code_dwords, code_dwords)
}

/// Reads an exact AGC shader-code allocation and decodes each adjacent entry point in it.
pub fn read_program_segments_bounded(
    memory: &impl GuestMemoryReader,
    address: u64,
    code_size_bytes: usize,
) -> Result<Vec<Program>, DecodeError> {
    if code_size_bytes == 0 || !code_size_bytes.is_multiple_of(4) {
        return Err(DecodeError::InvalidCodeSize {
            bytes: code_size_bytes,
        });
    }
    let mut words = Vec::with_capacity(code_size_bytes / 4);
    for word_index in 0..(code_size_bytes / 4) {
        let pc = u32::try_from(
            word_index
                .checked_mul(4)
                .ok_or(DecodeError::ProgramCounterOverflow)?,
        )
        .map_err(|_| DecodeError::ProgramCounterOverflow)?;
        words.push(read_memory_word(memory, address, pc, 0)?);
    }
    decode_program_segments(address, &words)
}

pub(crate) fn read_program_with_limits(
    memory: &impl GuestMemoryReader,
    address: u64,
    max_instructions: usize,
    max_dwords: usize,
) -> Result<Program, DecodeError> {
    let mut raw_words = Vec::new();
    let mut furthest_forward_branch_target = None;
    let mut decoded_instructions = 0usize;
    while decoded_instructions < max_instructions && raw_words.len() < max_dwords {
        let pc = u32::try_from(
            raw_words
                .len()
                .checked_mul(4)
                .ok_or(DecodeError::ProgramCounterOverflow)?,
        )
        .map_err(|_| DecodeError::ProgramCounterOverflow)?;
        let first = read_memory_word(memory, address, pc, 0)?;
        let mut prefix = vec![first];
        let frame = match decode_frame(pc, &prefix) {
            Err(DecodeError::Truncated {
                needed_dwords: 2, ..
            }) => {
                prefix.push(read_memory_word(memory, address, pc, 1)?);
                decode_frame(pc, &prefix)?
            }
            result => result?,
        };
        if raw_words.len().saturating_add(frame.dwords) > max_dwords {
            return Err(DecodeError::Truncated {
                pc,
                needed_dwords: frame.dwords,
                available_dwords: max_dwords.saturating_sub(raw_words.len()),
            });
        }
        for index in prefix.len()..frame.dwords {
            prefix.push(read_memory_word(memory, address, pc, index)?);
        }
        let instruction = create_instruction(pc, frame.encoding, frame.opcode, prefix.clone());
        if let Some(target) = forward_branch_target(&instruction)? {
            furthest_forward_branch_target = Some(
                furthest_forward_branch_target.map_or(target, |current: u32| current.max(target)),
            );
        }
        let ended = instruction_terminates_program(&instruction);
        raw_words.extend(prefix);
        decoded_instructions += 1;
        if ended && furthest_forward_branch_target.is_none_or(|target| pc >= target) {
            return decode_program(address, &raw_words);
        }
    }
    Err(DecodeError::Unterminated {
        inspected_instructions: decoded_instructions,
    })
}

fn instruction_terminates_program(instruction: &Instruction) -> bool {
    instruction.opcode == Opcode::END_PROGRAM || instruction.opcode.is("SSetpcB64")
}

pub(crate) fn forward_branch_target(instruction: &Instruction) -> Result<Option<u32>, DecodeError> {
    if instruction.encoding != ShaderEncoding::Sopp
        || !matches!(
            instruction.opcode.known_name(),
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
    {
        return Ok(None);
    }
    let offset = i16::from_ne_bytes(
        u16::try_from(instruction.words[0] & u32::from(u16::MAX))
            .expect("masked branch offset fits u16")
            .to_ne_bytes(),
    );
    let next_pc = i64::from(instruction.pc)
        + i64::try_from(instruction.words.len().saturating_mul(4))
            .map_err(|_| DecodeError::ProgramCounterOverflow)?;
    let target = next_pc + i64::from(offset) * 4;
    let target = u32::try_from(target).map_err(|_| DecodeError::ProgramCounterOverflow)?;
    Ok((target > instruction.pc).then_some(target))
}

pub(crate) fn read_memory_word(
    memory: &impl GuestMemoryReader,
    base: u64,
    pc: u32,
    word_index: usize,
) -> Result<u32, DecodeError> {
    let offset = u64::from(pc)
        .checked_add(
            u64::try_from(word_index)
                .map_err(|_| DecodeError::AddressOverflow { pc })?
                .checked_mul(4)
                .ok_or(DecodeError::AddressOverflow { pc })?,
        )
        .ok_or(DecodeError::AddressOverflow { pc })?;
    let word_address = base
        .checked_add(offset)
        .ok_or(DecodeError::AddressOverflow { pc })?;
    let mut bytes = [0_u8; 4];
    if !memory.read(word_address, &mut bytes) {
        return Err(DecodeError::MemoryFault {
            pc,
            address: word_address,
        });
    }
    Ok(u32::from_le_bytes(bytes))
}
