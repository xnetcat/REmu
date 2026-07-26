// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::ir::{
    BufferMemoryControl, DataShareControl, Dpp8Control, DppControl, ExportControl,
    GlobalMemoryControl, ImageControl, Instruction, InstructionControl, InterpolationControl,
    Opcode, Operand, OperandKind, Program, RawOpcodeFamily, ScalarMemoryControl, SdwaControl,
    ShaderEncoding, Vop3Control, Vop3pControl,
};
use crate::metadata::GuestMemoryReader;

const FALLBACK_MAX_INSTRUCTIONS: usize = 4096;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum DecodeError {
    #[error("shader instruction stream is empty")]
    Empty,
    #[error(
        "truncated shader instruction at pc 0x{pc:X}: needs {needed_dwords} dwords, only {available_dwords} remain"
    )]
    Truncated {
        pc: u32,
        needed_dwords: usize,
        available_dwords: usize,
    },
    #[error("unknown top-level shader encoding at pc 0x{pc:X}: word 0x{word:08X}")]
    UnknownEncoding { pc: u32, word: u32 },
    #[error("unknown {encoding:?} opcode 0x{opcode:X} at pc 0x{pc:X}: word 0x{word:08X}")]
    UnknownOpcode {
        pc: u32,
        encoding: ShaderEncoding,
        opcode: u32,
        word: u32,
    },
    #[error("unsupported FLAT address segment 0x{segment:X} at pc 0x{pc:X}: opcode 0x{opcode:X}")]
    FlatSegment { pc: u32, segment: u32, opcode: u32 },
    #[error("shader has no S_ENDPGM within {inspected_instructions} decoded instructions")]
    Unterminated { inspected_instructions: usize },
    #[error("shader code size {bytes} is not a non-zero number of dwords")]
    InvalidCodeSize { bytes: usize },
    #[error("shader program counter exceeds the 32-bit Gen5 address space")]
    ProgramCounterOverflow,
    #[error("shader address arithmetic overflowed at pc 0x{pc:X}")]
    AddressOverflow { pc: u32 },
    #[error("shader memory is not readable at {address:#x} (pc 0x{pc:X})")]
    MemoryFault { pc: u32, address: u64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Frame {
    encoding: ShaderEncoding,
    opcode: Opcode,
    dwords: usize,
}

/// Decodes one bounded Gen5/RDNA shader stream. The input must end at an `S_ENDPGM` that is not
/// before any statically encoded forward branch target; trailing allocation contents are ignored.
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
        let ended = instruction.opcode == Opcode::END_PROGRAM;
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

fn read_program_with_limits(
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
        let ended = frame.opcode == Opcode::END_PROGRAM;
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

fn forward_branch_target(instruction: &Instruction) -> Result<Option<u32>, DecodeError> {
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

fn read_memory_word(
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

fn decode_frame(pc: u32, words: &[u32]) -> Result<Frame, DecodeError> {
    let word = words[0];
    if word & 0x8000_0000 == 0 {
        let opcode = (word >> 25) & 0x3f;
        return if opcode == 0x3e {
            decode_vopc(pc, word)
        } else if opcode == 0x3f {
            decode_vop1(pc, word)
        } else {
            decode_vop2(pc, word)
        };
    }
    if word & 0xf800_0000 == 0xc000_0000 {
        return decode_smrd(pc, word);
    }
    if word & 0xc000_0000 == 0x8000_0000 {
        let opcode = (word >> 23) & 0x7f;
        return match opcode {
            0x7d => decode_sop1(pc, word),
            0x7e => decode_sopc(pc, word),
            0x7f => decode_sopp(pc, word),
            0x60.. => decode_sopk(pc, word),
            _ => decode_sop2(pc, word),
        };
    }
    if word & 0xff80_0000 == 0xcc00_0000 {
        let extra = required_word(pc, words, 1, 2)?;
        return Ok(decode_vop3p(word, extra));
    }

    match word >> 26 {
        0x32 => decode_vintrp(pc, word),
        0x33 | 0x3d => decode_smem(pc, word),
        0x34 | 0x35 => {
            let extra = required_word(pc, words, 1, 2)?;
            Ok(decode_vop3(word, extra))
        }
        0x36 => decode_ds(pc, word),
        0x37 => decode_flat(pc, word),
        0x38 => {
            let extra = required_word(pc, words, 1, 2)?;
            Ok(decode_mubuf(word, extra))
        }
        0x3a => {
            let extra = required_word(pc, words, 1, 2)?;
            Ok(decode_mtbuf(word, extra))
        }
        0x3c => Ok(decode_mimg(word)),
        0x3e => Ok(Frame {
            encoding: ShaderEncoding::Exp,
            opcode: Opcode::EXPORT,
            dwords: 2,
        }),
        0x3f => Ok(Frame {
            encoding: ShaderEncoding::Vop3p,
            opcode: Opcode::raw(RawOpcodeFamily::Vop3p, ((word >> 24) & 0xff) as u16),
            dwords: 2,
        }),
        _ => Err(DecodeError::UnknownEncoding { pc, word }),
    }
}

fn required_word(
    pc: u32,
    words: &[u32],
    index: usize,
    needed_dwords: usize,
) -> Result<u32, DecodeError> {
    words.get(index).copied().ok_or(DecodeError::Truncated {
        pc,
        needed_dwords,
        available_dwords: words.len(),
    })
}

fn known_frame(
    pc: u32,
    word: u32,
    encoding: ShaderEncoding,
    opcode_code: u32,
    name: Option<&'static str>,
    dwords: usize,
) -> Result<Frame, DecodeError> {
    let name = name.ok_or(DecodeError::UnknownOpcode {
        pc,
        encoding,
        opcode: opcode_code,
        word,
    })?;
    Ok(Frame {
        encoding,
        opcode: Opcode::from_static(name),
        dwords,
    })
}

fn decode_sop1(pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 8) & 0xff;
    let name = match code {
        0x03 => Some("SMovB32"),
        0x04 => Some("SMovB64"),
        0x07 => Some("SNotB32"),
        0x08 => Some("SNotB64"),
        0x0a => Some("SWqmB64"),
        0x0b => Some("SBrevB32"),
        0x0f => Some("SBcnt1I32B32"),
        0x10 => Some("SBcnt1I32B64"),
        0x13 => Some("SFF1I32B32"),
        0x1d => Some("SBitset1B32"),
        0x1f => Some("SGetpcB64"),
        0x20 => Some("SSetpcB64"),
        0x21 => Some("SSwappcB64"),
        0x24 => Some("SAndSaveexecB64"),
        0x25 => Some("SOrSaveexecB64"),
        0x26 => Some("SXorSaveexecB64"),
        0x27 => Some("SAndn2SaveexecB64"),
        0x28 => Some("SOrn2SaveexecB64"),
        0x29 => Some("SNandSaveexecB64"),
        0x2a => Some("SNorSaveexecB64"),
        0x2b => Some("SXnorSaveexecB64"),
        0x37 => Some("SAndn1SaveexecB64"),
        0x38 => Some("SOrn1SaveexecB64"),
        0x3c => Some("SAndSaveexecB32"),
        0x3d => Some("SOrSaveexecB32"),
        0x3e => Some("SXorSaveexecB32"),
        0x3f => Some("SAndn2SaveexecB32"),
        0x40 => Some("SOrn2SaveexecB32"),
        0x41 => Some("SNandSaveexecB32"),
        0x42 => Some("SNorSaveexecB32"),
        0x43 => Some("SXnorSaveexecB32"),
        0x44 => Some("SAndn1SaveexecB32"),
        0x45 => Some("SOrn1SaveexecB32"),
        _ => None,
    };
    known_frame(
        pc,
        word,
        ShaderEncoding::Sop1,
        code,
        name,
        1 + usize::from(word & 0xff == 0xff),
    )
}

fn decode_sop2(pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 23) & 0x7f;
    let name = match code {
        0x00 => Some("SAddU32"),
        0x01 => Some("SSubU32"),
        0x02 => Some("SAddI32"),
        0x03 => Some("SSubI32"),
        0x04 => Some("SAddcU32"),
        0x05 => Some("SSubbU32"),
        0x06 => Some("SMinI32"),
        0x07 => Some("SMinU32"),
        0x08 => Some("SMaxI32"),
        0x09 => Some("SMaxU32"),
        0x0a => Some("SCselectB32"),
        0x0b => Some("SCselectB64"),
        0x0e => Some("SAndB32"),
        0x0f => Some("SAndB64"),
        0x10 => Some("SOrB32"),
        0x11 => Some("SOrB64"),
        0x12 => Some("SXorB32"),
        0x13 => Some("SXorB64"),
        0x14 => Some("SAndn2B32"),
        0x15 => Some("SAndn2B64"),
        0x16 => Some("SOrn2B32"),
        0x17 => Some("SOrn2B64"),
        0x18 => Some("SNandB32"),
        0x19 => Some("SNandB64"),
        0x1a => Some("SNorB32"),
        0x1b => Some("SNorB64"),
        0x1c => Some("SXnorB32"),
        0x1d => Some("SXnorB64"),
        0x1e => Some("SLshlB32"),
        0x1f => Some("SLshlB64"),
        0x20 => Some("SLshrB32"),
        0x21 => Some("SLshrB64"),
        0x22 => Some("SAshrI32"),
        0x23 => Some("SAshrI64"),
        0x24 => Some("SBfmB32"),
        0x25 => Some("SBfmB64"),
        0x26 => Some("SMulI32"),
        0x27 => Some("SBfeU32"),
        0x28 => Some("SBfeI32"),
        0x29 => Some("SBfeU64"),
        0x2a => Some("SBfeI64"),
        0x2c => Some("SAbsdiffI32"),
        0x2e => Some("SLshl1AddU32"),
        0x2f => Some("SLshl2AddU32"),
        0x30 => Some("SLshl3AddU32"),
        0x31 => Some("SLshl4AddU32"),
        0x32 => Some("SPackLlB32B16"),
        0x33 => Some("SPackLhB32B16"),
        0x34 => Some("SPackHhB32B16"),
        0x35 => Some("SMulHiU32"),
        0x36 => Some("SMulHiI32"),
        _ => None,
    };
    let literal = word & 0xff == 0xff || (word >> 8) & 0xff == 0xff;
    known_frame(
        pc,
        word,
        ShaderEncoding::Sop2,
        code,
        name,
        1 + usize::from(literal),
    )
}

fn decode_sopc(pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 16) & 0x7f;
    let name = match code {
        0x00 => Some("SCmpEqI32"),
        0x01 => Some("SCmpLgI32"),
        0x02 => Some("SCmpGtI32"),
        0x03 => Some("SCmpGeI32"),
        0x04 => Some("SCmpLtI32"),
        0x05 => Some("SCmpLeI32"),
        0x06 => Some("SCmpEqU32"),
        0x07 => Some("SCmpLgU32"),
        0x08 => Some("SCmpGtU32"),
        0x09 => Some("SCmpGeU32"),
        0x0a => Some("SCmpLtU32"),
        0x0b => Some("SCmpLeU32"),
        0x0c => Some("SBitcmp0B32"),
        0x0d => Some("SBitcmp1B32"),
        0x0e => Some("SBitcmp0B64"),
        0x0f => Some("SBitcmp1B64"),
        0x12 => Some("SCmpEqU64"),
        0x13 => Some("SCmpLgU64"),
        _ => None,
    };
    let literal = word & 0xff == 0xff || (word >> 8) & 0xff == 0xff;
    known_frame(
        pc,
        word,
        ShaderEncoding::Sopc,
        code,
        name,
        1 + usize::from(literal),
    )
}

fn decode_sopp(pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 16) & 0x7f;
    let name = match code {
        0x00 => Some("SNop"),
        0x01 => Some("SEndpgm"),
        0x02 => Some("SBranch"),
        0x04 => Some("SCbranchScc0"),
        0x05 => Some("SCbranchScc1"),
        0x06 => Some("SCbranchVccz"),
        0x07 => Some("SCbranchVccnz"),
        0x08 => Some("SCbranchExecz"),
        0x09 => Some("SCbranchExecnz"),
        0x0a => Some("SBarrier"),
        0x0c => Some("SWaitcnt"),
        0x10 => Some("SSendmsg"),
        0x16 => Some("STtraceData"),
        // RDNA2 uses S_CODE_END to pad bytes beyond the executable shader. Treating it as the
        // bounded stream terminator keeps the decoder from interpreting padding as instructions.
        0x1f => {
            return Ok(Frame {
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                dwords: 1,
            });
        }
        0x20 => Some("SInstPrefetch"),
        0x21 => Some("SClause"),
        0x23 => Some("SWaitcntDepctr"),
        _ => None,
    };
    known_frame(pc, word, ShaderEncoding::Sopp, code, name, 1)
}

fn decode_sopk(pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = ((word >> 23) & 0x7f) - 0x60;
    let name = match code {
        0x00 => Some("SMovkI32"),
        0x03 => Some("SCmpkEqI32"),
        0x04 => Some("SCmpkLgI32"),
        0x05 => Some("SCmpkGtI32"),
        0x06 => Some("SCmpkGeI32"),
        0x07 => Some("SCmpkLtI32"),
        0x08 => Some("SCmpkLeI32"),
        0x09 => Some("SCmpkEqU32"),
        0x0a => Some("SCmpkLgU32"),
        0x0b => Some("SCmpkGtU32"),
        0x0c => Some("SCmpkGeU32"),
        0x0d => Some("SCmpkLtU32"),
        0x0e => Some("SCmpkLeU32"),
        0x0f => Some("SAddkI32"),
        0x10 => Some("SMulkI32"),
        _ => None,
    };
    known_frame(pc, word, ShaderEncoding::Sopk, code, name, 1)
}

fn vector_extension_dwords(source: u32) -> usize {
    usize::from(matches!(source, 0xe9 | 0xea | 0xf9 | 0xfa | 0xff))
}

fn decode_vop1(pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 9) & 0xff;
    let name = match code {
        0x00 => Some("VNop"),
        0x01 => Some("VMovB32"),
        0x02 => Some("VReadfirstlaneB32"),
        0x05 => Some("VCvtF32I32"),
        0x06 => Some("VCvtF32U32"),
        0x07 => Some("VCvtU32F32"),
        0x08 => Some("VCvtI32F32"),
        0x0a => Some("VCvtF16F32"),
        0x0b => Some("VCvtF32F16"),
        0x0c => Some("VCvtRpiI32F32"),
        0x0d => Some("VCvtFlrI32F32"),
        0x0e => Some("VCvtOffF32I4"),
        0x11 => Some("VCvtF32Ubyte0"),
        0x12 => Some("VCvtF32Ubyte1"),
        0x13 => Some("VCvtF32Ubyte2"),
        0x14 => Some("VCvtF32Ubyte3"),
        0x20 => Some("VFractF32"),
        0x21 => Some("VTruncF32"),
        0x22 => Some("VCeilF32"),
        0x23 => Some("VRndneF32"),
        0x24 => Some("VFloorF32"),
        0x25 => Some("VExpF32"),
        0x27 => Some("VLogF32"),
        0x2a => Some("VRcpF32"),
        0x2b => Some("VRcpIflagF32"),
        0x2e => Some("VRsqF32"),
        0x33 => Some("VSqrtF32"),
        0x35 => Some("VSinF32"),
        0x36 => Some("VCosF32"),
        0x37 => Some("VNotB32"),
        0x38 => Some("VBfrevB32"),
        0x3a => Some("VFfblB32"),
        0x42 => Some("VMovreldB32"),
        0x43 => Some("VMovrelsB32"),
        0x44 => Some("VMovrelsdB32"),
        0x52 => Some("VCvtU16F16"),
        0x56 => Some("VRsqF16"),
        _ => None,
    };
    known_frame(
        pc,
        word,
        ShaderEncoding::Vop1,
        code,
        name,
        1 + vector_extension_dwords(word & 0x1ff),
    )
}

fn decode_vop2(pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 25) & 0x3f;
    let name = match code {
        0x01 => Some("VCndmaskB32"),
        0x02 => Some("VDot2cF32F16"),
        0x03 => Some("VAddF32"),
        0x04 => Some("VSubF32"),
        0x05 => Some("VSubrevF32"),
        0x08 => Some("VMulF32"),
        0x0b => Some("VMulU32U24"),
        0x0c => Some("VMulHiU32U24"),
        0x0f => Some("VMinF32"),
        0x10 => Some("VMaxF32"),
        0x11 => Some("VMinI32"),
        0x12 => Some("VMaxI32"),
        0x13 => Some("VMinU32"),
        0x14 => Some("VMaxU32"),
        0x15 => Some("VLshrB32"),
        0x16 => Some("VLshrrevB32"),
        0x17 => Some("VAshrI32"),
        0x18 => Some("VAshrrevI32"),
        0x19 => Some("VLshlB32"),
        0x1a => Some("VLshlrevB32"),
        0x1b => Some("VAndB32"),
        0x1c => Some("VOrB32"),
        0x1d => Some("VXorB32"),
        0x1e => Some("VXnorB32"),
        0x1f => Some("VMacF32"),
        0x20 => Some("VMadMkF32"),
        0x21 => Some("VMadAkF32"),
        0x22 => Some("VBcntU32B32"),
        0x23 => Some("VMbcntLoU32B32"),
        0x24 => Some("VMbcntHiU32B32"),
        0x25 => Some("VAddI32"),
        0x26 => Some("VSubI32"),
        0x27 => Some("VSubrevI32"),
        0x28 => Some("VAddcU32"),
        0x29 => Some("VSubbU32"),
        0x2a => Some("VSubbrevU32"),
        0x2b => Some("VFmacF32"),
        0x2c => Some("VFmaMkF32"),
        0x2d => Some("VFmaAkF32"),
        0x2f => Some("VCvtPkrtzF16F32"),
        0x30 => Some("VCvtPkU16U32"),
        0x31 => Some("VCvtPkI16I32"),
        0x32 => Some("VAddF16"),
        0x33 => Some("VSubF16"),
        0x34 => Some("VSubrevF16"),
        0x35 => Some("VMulF16"),
        0x36 => Some("VFmacF16"),
        0x37 => Some("VFmaMkF16"),
        0x38 => Some("VFmaAkF16"),
        0x39 => Some("VMaxF16"),
        0x3a => Some("VMinF16"),
        0x3b => Some("VLdexpF16"),
        _ => None,
    };
    let forced_literal = matches!(code, 0x20 | 0x21 | 0x2c | 0x2d);
    known_frame(
        pc,
        word,
        ShaderEncoding::Vop2,
        code,
        name,
        1 + usize::from(forced_literal).max(vector_extension_dwords(word & 0x1ff)),
    )
}

fn decode_vopc(pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 17) & 0xff;
    let name = match code {
        0x00 => Some("VCmpFF32"),
        0x01 => Some("VCmpLtF32"),
        0x02 => Some("VCmpEqF32"),
        0x03 => Some("VCmpLeF32"),
        0x04 => Some("VCmpGtF32"),
        0x05 => Some("VCmpLgF32"),
        0x06 => Some("VCmpGeF32"),
        0x07 => Some("VCmpOF32"),
        0x08 => Some("VCmpUF32"),
        0x09 => Some("VCmpNgeF32"),
        0x0a => Some("VCmpNlgF32"),
        0x0b => Some("VCmpNgtF32"),
        0x0c => Some("VCmpNleF32"),
        0x0d => Some("VCmpNeqF32"),
        0x0e => Some("VCmpNltF32"),
        0x0f => Some("VCmpTruF32"),
        0x10 => Some("VCmpxFF32"),
        0x11 => Some("VCmpxLtF32"),
        0x12 => Some("VCmpxEqF32"),
        0x13 => Some("VCmpxLeF32"),
        0x14 => Some("VCmpxGtF32"),
        0x15 => Some("VCmpxLgF32"),
        0x16 => Some("VCmpxGeF32"),
        0x17 => Some("VCmpxOF32"),
        0x18 => Some("VCmpxUF32"),
        0x19 => Some("VCmpxNgeF32"),
        0x1a => Some("VCmpxNlgF32"),
        0x1b => Some("VCmpxNgtF32"),
        0x1c => Some("VCmpxNleF32"),
        0x1d => Some("VCmpxNeqF32"),
        0x1e => Some("VCmpxNltF32"),
        0x1f => Some("VCmpxTruF32"),
        0x80 => Some("VCmpFI32"),
        0x81 => Some("VCmpLtI32"),
        0x82 => Some("VCmpEqI32"),
        0x83 => Some("VCmpLeI32"),
        0x84 => Some("VCmpGtI32"),
        0x85 => Some("VCmpNeI32"),
        0x86 => Some("VCmpGeI32"),
        0x87 => Some("VCmpTI32"),
        0x88 => Some("VCmpClassF32"),
        0x90 => Some("VCmpxFI32"),
        0x91 => Some("VCmpxLtI32"),
        0x92 => Some("VCmpxEqI32"),
        0x93 => Some("VCmpxLeI32"),
        0x94 => Some("VCmpxGtI32"),
        0x95 => Some("VCmpxNeI32"),
        0x96 => Some("VCmpxGeI32"),
        0x97 => Some("VCmpxTI32"),
        0xc0 => Some("VCmpFU32"),
        0xc1 => Some("VCmpLtU32"),
        0xc2 => Some("VCmpEqU32"),
        0xc3 => Some("VCmpLeU32"),
        0xc4 => Some("VCmpGtU32"),
        0xc5 => Some("VCmpNeU32"),
        0xc6 => Some("VCmpGeU32"),
        0xc7 => Some("VCmpTU32"),
        0xc8 => Some("VCmpFF16"),
        0xc9 => Some("VCmpLtF16"),
        0xca => Some("VCmpEqF16"),
        0xcb => Some("VCmpLeF16"),
        0xcc => Some("VCmpGtF16"),
        0xcd => Some("VCmpLgF16"),
        0xce => Some("VCmpGeF16"),
        0xcf => Some("VCmpOF16"),
        0xd0 => Some("VCmpxFU32"),
        0xd1 => Some("VCmpxLtU32"),
        0xd2 => Some("VCmpxEqU32"),
        0xd3 => Some("VCmpxLeU32"),
        0xd4 => Some("VCmpxGtU32"),
        0xd5 => Some("VCmpxNeU32"),
        0xd6 => Some("VCmpxGeU32"),
        0xd7 => Some("VCmpxTU32"),
        0xd8 => Some("VCmpxFF16"),
        0xd9 => Some("VCmpxLtF16"),
        0xda => Some("VCmpxEqF16"),
        0xdb => Some("VCmpxLeF16"),
        0xdc => Some("VCmpxGtF16"),
        0xdd => Some("VCmpxLgF16"),
        0xde => Some("VCmpxGeF16"),
        0xdf => Some("VCmpxOF16"),
        0xe8 => Some("VCmpUF16"),
        0xe9 => Some("VCmpNgeF16"),
        0xea => Some("VCmpNlgF16"),
        0xeb => Some("VCmpNgtF16"),
        0xec => Some("VCmpNleF16"),
        0xed => Some("VCmpNeqF16"),
        0xee => Some("VCmpNltF16"),
        0xef => Some("VCmpTruF16"),
        0xf8 => Some("VCmpxUF16"),
        0xf9 => Some("VCmpxNgeF16"),
        0xfa => Some("VCmpxNlgF16"),
        0xfb => Some("VCmpxNgtF16"),
        0xfc => Some("VCmpxNleF16"),
        0xfd => Some("VCmpxNeqF16"),
        0xfe => Some("VCmpxNltF16"),
        0xff => Some("VCmpxTruF16"),
        _ => None,
    };
    known_frame(
        pc,
        word,
        ShaderEncoding::Vopc,
        code,
        name,
        1 + vector_extension_dwords(word & 0x1ff),
    )
}

fn decode_vop3(word: u32, extra: u32) -> Frame {
    let code = (word >> 16) & 0x3ff;
    let is_b = is_vop3b_opcode(code);
    let name = if is_b {
        match code {
            0x128 => Some("VAddCoCiU32"),
            0x30f => Some("VAddCoU32"),
            0x310 => Some("VSubCoU32"),
            0x319 => Some("VSubrevCoU32"),
            0x176 => Some("VMadU64U32"),
            _ => None,
        }
    } else {
        match code {
            0x101 => Some("VCndmaskB32"),
            0x103 => Some("VAddF32"),
            0x104 => Some("VSubF32"),
            0x108 => Some("VMulF32"),
            0x10f => Some("VMinF32"),
            0x110 => Some("VMaxF32"),
            0x11f => Some("VMacF32"),
            0x12b => Some("VFmacF32"),
            0x12f => Some("VCvtPkrtzF16F32"),
            0x141 => Some("VMadF32"),
            0x143 => Some("VMadU32U24"),
            0x144 => Some("VCubeidF32"),
            0x145 => Some("VCubescF32"),
            0x146 => Some("VCubetcF32"),
            0x147 => Some("VCubemaF32"),
            0x148 => Some("VBfeU32"),
            0x149 => Some("VBfeI32"),
            0x14a => Some("VBfiB32"),
            0x14b => Some("VFmaF32"),
            0x151 => Some("VMin3F32"),
            0x152 => Some("VMin3I32"),
            0x153 => Some("VMin3U32"),
            0x154 => Some("VMax3F32"),
            0x155 => Some("VMax3I32"),
            0x156 => Some("VMax3U32"),
            0x157 => Some("VMed3F32"),
            0x158 => Some("VMed3I32"),
            0x159 => Some("VMed3U32"),
            0x15a => Some("VSadU8"),
            0x15b => Some("VSadHiU8"),
            0x15c => Some("VSadU16"),
            0x15d => Some("VSadU32"),
            0x15e => Some("VCvtPkU8F32"),
            0x169 => Some("VMulLoU32"),
            0x16a => Some("VMulHiU32"),
            0x16b => Some("VMulLoI32"),
            0x16c => Some("VMulHiI32"),
            // VOP1 opcodes use their architectural opcode plus 0x180 in the VOP3/e64
            // encoding. V_CVT_FLR_I32_F32 is opcode 0x0d, hence 0x18d here.
            0x18d => Some("VCvtFlrI32F32"),
            0x345 => Some("VXadU32"),
            0x346 => Some("VLshlAddU32"),
            0x347 => Some("VAddLshlU32"),
            0x34b => Some("VFmaF16"),
            0x360 => Some("VReadlaneB32"),
            0x361 => Some("VWritelaneB32"),
            0x362 => Some("VLdexpF32"),
            0x363 => Some("VBfmB32"),
            0x364 => Some("VBcntU32B32"),
            0x365 => Some("VMbcntLoU32B32"),
            0x366 => Some("VMbcntHiU32B32"),
            0x368 => Some("VCvtPknormI16F32"),
            0x369 => Some("VCvtPknormU16F32"),
            0x36a => Some("VCvtPkU16U32"),
            0x36d => Some("VAdd3U32"),
            0x36f => Some("VLshlOrU32"),
            0x371 => Some("VAndOrB32"),
            0x372 => Some("VOr3U32"),
            0x373 => Some("VMadU32U16"),
            0x377 => Some("VPermlane16B32"),
            0x378 => Some("VPermlanex16B32"),
            _ => None,
        }
    };
    let opcode = name.map_or_else(
        || {
            Opcode::raw(
                if is_b {
                    RawOpcodeFamily::Vop3b
                } else {
                    RawOpcodeFamily::Vop3
                },
                code as u16,
            )
        },
        Opcode::from_static,
    );
    let literal = [extra & 0x1ff, (extra >> 9) & 0x1ff, (extra >> 18) & 0x1ff].contains(&0xff);
    Frame {
        encoding: ShaderEncoding::Vop3,
        opcode,
        dwords: 2 + usize::from(literal),
    }
}

const fn is_vop3b_opcode(code: u32) -> bool {
    matches!(
        code,
        0x128 | 0x16d | 0x16e | 0x176 | 0x177 | 0x30f | 0x310 | 0x319
    )
}

fn decode_vop3p(word: u32, extra: u32) -> Frame {
    let code = (word >> 16) & 0x7f;
    let name = match code {
        0x0e => Some("VPkFmaF16"),
        0x0f => Some("VPkAddF16"),
        0x10 => Some("VPkMulF16"),
        0x11 => Some("VPkMinF16"),
        0x12 => Some("VPkMaxF16"),
        0x20 => Some("VFmaMixF32"),
        0x21 => Some("VFmaMixloF16"),
        0x22 => Some("VFmaMixhiF16"),
        _ => None,
    };
    let opcode = name.map_or_else(
        || Opcode::raw(RawOpcodeFamily::Vop3p, code as u16),
        Opcode::from_static,
    );
    let literal = [extra & 0x1ff, (extra >> 9) & 0x1ff, (extra >> 18) & 0x1ff].contains(&0xff);
    Frame {
        encoding: ShaderEncoding::Vop3p,
        opcode,
        dwords: 2 + usize::from(literal),
    }
}

fn decode_ds(pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 18) & 0xff;
    let name = match code {
        0x00 => Some("DsAddU32"),
        0x01 => Some("DsSubU32"),
        0x03 => Some("DsIncU32"),
        0x04 => Some("DsDecU32"),
        0x05 => Some("DsMinI32"),
        0x06 => Some("DsMaxI32"),
        0x07 => Some("DsMinU32"),
        0x08 => Some("DsMaxU32"),
        0x09 => Some("DsAndB32"),
        0x0a => Some("DsOrB32"),
        0x0b => Some("DsXorB32"),
        0x0d => Some("DsWriteB32"),
        0x0e => Some("DsWrite2B32"),
        0x0f => Some("DsWrite2St64B32"),
        0x10 => Some("DsCmpstB32"),
        0x20 => Some("DsAddRtnU32"),
        0x21 => Some("DsSubRtnU32"),
        0x23 => Some("DsIncRtnU32"),
        0x24 => Some("DsDecRtnU32"),
        0x25 => Some("DsMinRtnI32"),
        0x26 => Some("DsMaxRtnI32"),
        0x27 => Some("DsMinRtnU32"),
        0x28 => Some("DsMaxRtnU32"),
        0x29 => Some("DsAndRtnB32"),
        0x2a => Some("DsOrRtnB32"),
        0x2b => Some("DsXorRtnB32"),
        0x2d => Some("DsWrxchgRtnB32"),
        0x30 => Some("DsCmpstRtnB32"),
        0x35 => Some("DsSwizzleB32"),
        0x36 => Some("DsReadB32"),
        0x37 => Some("DsRead2B32"),
        0x38 => Some("DsRead2St64B32"),
        0x4d => Some("DsWriteB64"),
        0xb0 => Some("DsWriteAddtidB32"),
        0xde => Some("DsWriteB96"),
        0xdf => Some("DsWriteB128"),
        0xfe => Some("DsReadB96"),
        0xff => Some("DsReadB128"),
        _ => None,
    };
    known_frame(pc, word, ShaderEncoding::Ds, code, name, 2)
}

fn decode_mtbuf(word: u32, extra: u32) -> Frame {
    let code = ((extra >> 21) & 1) << 3 | ((word >> 16) & 0x7);
    let name = match code {
        0x00 => "TBufferLoadFormatX",
        0x01 => "TBufferLoadFormatXy",
        0x02 => "TBufferLoadFormatXyz",
        0x03 => "TBufferLoadFormatXyzw",
        0x04 => "TBufferStoreFormatX",
        0x05 => "TBufferStoreFormatXy",
        0x06 => "TBufferStoreFormatXyz",
        0x07 => "TBufferStoreFormatXyzw",
        0x08 => "TBufferLoadFormatD16X",
        0x09 => "TBufferLoadFormatD16Xy",
        0x0a => "TBufferLoadFormatD16Xyz",
        0x0b => "TBufferLoadFormatD16Xyzw",
        0x0c => "TBufferStoreFormatD16X",
        0x0d => "TBufferStoreFormatD16Xy",
        0x0e => "TBufferStoreFormatD16Xyz",
        0x0f => "TBufferStoreFormatD16Xyzw",
        _ => unreachable!("four-bit opcode"),
    };
    Frame {
        encoding: ShaderEncoding::Mtbuf,
        opcode: Opcode::from_static(name),
        dwords: 2 + usize::from(extra >> 24 == 0xff),
    }
}

fn decode_mubuf(word: u32, extra: u32) -> Frame {
    let code = (word >> 18) & 0xff;
    let name = match code {
        0x00 => Some("BufferLoadFormatX"),
        0x01 => Some("BufferLoadFormatXy"),
        0x02 => Some("BufferLoadFormatXyz"),
        0x03 => Some("BufferLoadFormatXyzw"),
        0x04 => Some("BufferStoreFormatX"),
        0x05 => Some("BufferStoreFormatXy"),
        0x06 => Some("BufferStoreFormatXyz"),
        0x07 => Some("BufferStoreFormatXyzw"),
        0x08 => Some("BufferLoadUbyte"),
        0x09 => Some("BufferLoadSbyte"),
        0x0a => Some("BufferLoadUshort"),
        0x0b => Some("BufferLoadSshort"),
        0x0c => Some("BufferLoadDword"),
        0x0d => Some("BufferLoadDwordx2"),
        0x0e => Some("BufferLoadDwordx4"),
        0x0f => Some("BufferLoadDwordx3"),
        0x18 => Some("BufferStoreByte"),
        0x19 => Some("BufferStoreByteD16Hi"),
        0x1a => Some("BufferStoreShort"),
        0x1b => Some("BufferStoreShortD16Hi"),
        0x1c => Some("BufferStoreDword"),
        0x1d => Some("BufferStoreDwordx2"),
        0x1e => Some("BufferStoreDwordx4"),
        0x1f => Some("BufferStoreDwordx3"),
        0x20 => Some("BufferLoadUbyteD16"),
        0x21 => Some("BufferLoadUbyteD16Hi"),
        0x22 => Some("BufferLoadSbyteD16"),
        0x23 => Some("BufferLoadSbyteD16Hi"),
        0x24 => Some("BufferLoadShortD16"),
        0x25 => Some("BufferLoadShortD16Hi"),
        0x26 => Some("BufferLoadFormatD16HiX"),
        0x27 => Some("BufferStoreFormatD16HiX"),
        0x30 => Some("BufferAtomicSwap"),
        0x31 => Some("BufferAtomicCmpswap"),
        0x32 => Some("BufferAtomicAdd"),
        0x33 => Some("BufferAtomicSub"),
        0x35 => Some("BufferAtomicSmin"),
        0x36 => Some("BufferAtomicUmin"),
        0x37 => Some("BufferAtomicSmax"),
        0x38 => Some("BufferAtomicUmax"),
        0x39 => Some("BufferAtomicAnd"),
        0x3a => Some("BufferAtomicOr"),
        0x3b => Some("BufferAtomicXor"),
        0x3c => Some("BufferAtomicInc"),
        0x3d => Some("BufferAtomicDec"),
        0x80 => Some("BufferLoadFormatD16X"),
        0x81 => Some("BufferLoadFormatD16Xy"),
        0x82 => Some("BufferLoadFormatD16Xyz"),
        0x83 => Some("BufferLoadFormatD16Xyzw"),
        0x84 => Some("BufferStoreFormatD16X"),
        0x85 => Some("BufferStoreFormatD16Xy"),
        0x86 => Some("BufferStoreFormatD16Xyz"),
        0x87 => Some("BufferStoreFormatD16Xyzw"),
        _ => None,
    };
    Frame {
        encoding: ShaderEncoding::Mubuf,
        opcode: name.map_or_else(
            || Opcode::raw(RawOpcodeFamily::Mubuf, code as u16),
            Opcode::from_static,
        ),
        dwords: 2 + usize::from(extra >> 24 == 0xff),
    }
}

fn decode_flat(pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let segment = (word >> 14) & 0x3;
    let code = (word >> 18) & 0x7f;
    if segment != 2 {
        return Err(DecodeError::FlatSegment {
            pc,
            segment,
            opcode: code,
        });
    }
    let name = match code {
        0x08 => Some("GlobalLoadUbyte"),
        0x09 => Some("GlobalLoadSbyte"),
        0x0a => Some("GlobalLoadUshort"),
        0x0b => Some("GlobalLoadSshort"),
        0x0c => Some("GlobalLoadDword"),
        0x0d => Some("GlobalLoadDwordx2"),
        0x0e => Some("GlobalLoadDwordx4"),
        0x0f => Some("GlobalLoadDwordx3"),
        0x18 => Some("GlobalStoreByte"),
        0x19 => Some("GlobalStoreByteD16Hi"),
        0x1a => Some("GlobalStoreShort"),
        0x1b => Some("GlobalStoreShortD16Hi"),
        0x1c => Some("GlobalStoreDword"),
        0x1d => Some("GlobalStoreDwordx2"),
        0x1e => Some("GlobalStoreDwordx4"),
        0x1f => Some("GlobalStoreDwordx3"),
        0x20 => Some("GlobalLoadUbyteD16"),
        0x21 => Some("GlobalLoadUbyteD16Hi"),
        0x22 => Some("GlobalLoadSbyteD16"),
        0x23 => Some("GlobalLoadSbyteD16Hi"),
        0x24 => Some("GlobalLoadShortD16"),
        0x25 => Some("GlobalLoadShortD16Hi"),
        0x32 => Some("GlobalAtomicAdd"),
        0x38 => Some("GlobalAtomicUMax"),
        _ => None,
    };
    known_frame(pc, word, ShaderEncoding::Flat, code, name, 2)
}

fn decode_smrd(pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 22) & 0x1f;
    let name = scalar_load_name(code);
    let literal = word >> 8 & 1 == 0 && word & 0xff == 0xff;
    known_frame(
        pc,
        word,
        ShaderEncoding::Smrd,
        code,
        name,
        1 + usize::from(literal),
    )
}

fn decode_smem(pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 18) & 0xff;
    known_frame(
        pc,
        word,
        ShaderEncoding::Smem,
        code,
        scalar_load_name(code),
        2,
    )
}

const fn scalar_load_name(code: u32) -> Option<&'static str> {
    match code {
        0x00 => Some("SLoadDword"),
        0x01 => Some("SLoadDwordx2"),
        0x02 => Some("SLoadDwordx4"),
        0x03 => Some("SLoadDwordx8"),
        0x04 => Some("SLoadDwordx16"),
        0x08 => Some("SBufferLoadDword"),
        0x09 => Some("SBufferLoadDwordx2"),
        0x0a => Some("SBufferLoadDwordx4"),
        0x0b => Some("SBufferLoadDwordx8"),
        0x0c => Some("SBufferLoadDwordx16"),
        _ => None,
    }
}

fn decode_mimg(word: u32) -> Frame {
    // MIMG is the one RDNA2 encoding whose opcode is split: OP[6:0] lives in
    // bits 24:18 while OP[7] lives in bit 0. Bit 0 must not be mistaken for
    // part of NSA; dropping it aliases IMAGE_MSAA_LOAD and later operations to
    // unrelated low-numbered image operations.
    let code = ((word >> 18) & 0x7f) | ((word & 1) << 7);
    let name = match code {
        0x00 => Some("ImageLoad"),
        0x01 => Some("ImageLoadMip"),
        0x08 => Some("ImageStore"),
        0x09 => Some("ImageStoreMip"),
        0x0e => Some("ImageGetResinfo"),
        0x0f => Some("ImageAtomicSwap"),
        0x10 => Some("ImageAtomicCmpswap"),
        0x11 => Some("ImageAtomicAdd"),
        0x12 => Some("ImageAtomicSub"),
        0x14 => Some("ImageAtomicSmin"),
        0x15 => Some("ImageAtomicUmin"),
        0x16 => Some("ImageAtomicSmax"),
        0x17 => Some("ImageAtomicUmax"),
        0x18 => Some("ImageAtomicAnd"),
        0x19 => Some("ImageAtomicOr"),
        0x1a => Some("ImageAtomicXor"),
        0x1b => Some("ImageAtomicInc"),
        0x1c => Some("ImageAtomicDec"),
        0x20 => Some("ImageSample"),
        0x22 => Some("ImageSampleD"),
        0x24 => Some("ImageSampleL"),
        0x25 => Some("ImageSampleB"),
        0x27 => Some("ImageSampleLz"),
        0x2f => Some("ImageSampleCLz"),
        0x30 => Some("ImageSampleO"),
        0x34 => Some("ImageSampleLO"),
        0x37 => Some("ImageSampleLzO"),
        0x40 => Some("ImageGather4"),
        0x47 => Some("ImageGather4Lz"),
        0x48 => Some("ImageGather4C"),
        0x4e => Some("ImageGather4CBCl"),
        0x57 => Some("ImageGather4LzO"),
        0x5f => Some("ImageGather4CLzO"),
        _ => None,
    };
    Frame {
        encoding: ShaderEncoding::Mimg,
        opcode: name.map_or_else(
            || {
                Opcode::raw(
                    RawOpcodeFamily::Mimg,
                    u16::try_from(code).expect("eight-bit MIMG opcode fits u16"),
                )
            },
            Opcode::from_static,
        ),
        dwords: 2 + ((word >> 1) & 3) as usize,
    }
}

fn decode_vintrp(pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 16) & 3;
    let name = match code {
        0 => Some("VInterpP1F32"),
        1 => Some("VInterpP2F32"),
        2 => Some("VInterpMovF32"),
        _ => None,
    };
    known_frame(pc, word, ShaderEncoding::Vintrp, code, name, 1)
}

#[allow(clippy::too_many_lines)]
fn create_instruction(
    pc: u32,
    encoding: ShaderEncoding,
    opcode: Opcode,
    words: Vec<u32>,
) -> Instruction {
    let word = words[0];
    let extended_vector = matches!(
        encoding,
        ShaderEncoding::Vop1 | ShaderEncoding::Vop2 | ShaderEncoding::Vopc
    );
    let is_sdwa = extended_vector && word & 0x1ff == 0xf9;
    let is_dpp = extended_vector && word & 0x1ff == 0xfa;
    let is_dpp8 = extended_vector && matches!(word & 0x1ff, 0xe9 | 0xea);
    let minimum = minimum_encoding_dwords(encoding);
    let literal =
        (!is_sdwa && !is_dpp && !is_dpp8 && words.len() > minimum).then(|| words[words.len() - 1]);

    let mut sources = Vec::new();
    let mut destinations = Vec::new();
    let mut control = None;

    match encoding {
        ShaderEncoding::Sop1 => {
            sources.push(Operand::source(word & 0xff, literal));
            destinations.push(Operand::scalar((word >> 16) & 0x7f));
        }
        ShaderEncoding::Sop2 => {
            sources.push(Operand::source(word & 0xff, literal));
            sources.push(Operand::source((word >> 8) & 0xff, literal));
            destinations.push(Operand::scalar((word >> 16) & 0x7f));
        }
        ShaderEncoding::Sopc => {
            sources.push(Operand::source(word & 0xff, literal));
            sources.push(Operand::source((word >> 8) & 0xff, literal));
        }
        ShaderEncoding::Sopp => {
            sources.push(Operand {
                kind: OperandKind::EncodedConstant,
                value: word & 0xffff,
            });
        }
        ShaderEncoding::Sopk => {
            sources.push(Operand {
                kind: OperandKind::EncodedConstant,
                value: word & 0xffff,
            });
            destinations.push(Operand::scalar((word >> 16) & 0x7f));
        }
        ShaderEncoding::Smrd => {
            let scalar_base = ((word >> 9) & 0x3f) * 2;
            let scalar_destination = (word >> 15) & 0x7f;
            let immediate = word >> 8 & 1 != 0;
            let offset = word & 0xff;
            let count = scalar_load_dword_count(opcode);
            let (immediate_offset_bytes, dynamic_offset_register) = if immediate {
                (
                    i32::try_from(offset * 4).expect("8-bit SMRD offset fits i32"),
                    None,
                )
            } else if offset == 0xff {
                (literal.unwrap_or_default().cast_signed(), None)
            } else {
                (0, Some(offset))
            };
            sources.push(Operand::scalar(scalar_base));
            if let Some(register) = dynamic_offset_register {
                sources.push(Operand::scalar(register));
            }
            push_scalar_range(&mut destinations, scalar_destination, count);
            control = Some(InstructionControl::ScalarMemory(ScalarMemoryControl {
                destination_count: count,
                immediate_offset_bytes,
                dynamic_offset_register,
            }));
        }
        ShaderEncoding::Smem => {
            let extra = words[1];
            let scalar_base = (word & 0x3f) * 2;
            let scalar_destination = (word >> 6) & 0x7f;
            let scalar_offset = (extra >> 25) & 0x7f;
            let offset = sign_extend(extra & 0x1f_ffff, 21);
            let count = scalar_load_dword_count(opcode);
            let scalar_offset_operand = Operand::source(scalar_offset, None);
            let dynamic_offset_register = (scalar_offset_operand.kind
                == OperandKind::ScalarRegister)
                .then_some(scalar_offset_operand.value);
            sources.push(Operand::scalar(scalar_base));
            sources.push(scalar_offset_operand);
            push_scalar_range(&mut destinations, scalar_destination, count);
            control = Some(InstructionControl::ScalarMemory(ScalarMemoryControl {
                destination_count: count,
                immediate_offset_bytes: offset,
                dynamic_offset_register,
            }));
        }
        ShaderEncoding::Vop1 => {
            decode_vop_extension(
                word,
                &words,
                literal,
                is_sdwa,
                is_dpp,
                is_dpp8,
                false,
                false,
                &mut sources,
                &mut destinations,
                &mut control,
            );
            if opcode.is("VReadfirstlaneB32") {
                destinations.push(Operand::scalar((word >> 17) & 0x7f));
            } else {
                destinations.push(Operand::vector((word >> 17) & 0xff));
            }
        }
        ShaderEncoding::Vop2 => {
            decode_vop_extension(
                word,
                &words,
                literal,
                is_sdwa,
                is_dpp,
                is_dpp8,
                true,
                false,
                &mut sources,
                &mut destinations,
                &mut control,
            );
            if !is_sdwa && !is_dpp && !is_dpp8 {
                if (opcode.is("VMadMkF32") || opcode.is("VFmaMkF32") || opcode.is("VFmaMkF16"))
                    && let Some(literal) = literal
                {
                    sources.insert(1, literal_operand(literal));
                } else if (opcode.is("VMadAkF32")
                    || opcode.is("VFmaAkF32")
                    || opcode.is("VFmaAkF16"))
                    && let Some(literal) = literal
                {
                    sources.push(literal_operand(literal));
                }
            }
            destinations.push(Operand::vector((word >> 17) & 0xff));
        }
        ShaderEncoding::Vopc => {
            decode_vop_extension(
                word,
                &words,
                literal,
                is_sdwa,
                is_dpp,
                is_dpp8,
                true,
                true,
                &mut sources,
                &mut destinations,
                &mut control,
            );
        }
        ShaderEncoding::Vop3 => {
            let extra = words[1];
            sources.extend([
                Operand::source(extra & 0x1ff, literal),
                Operand::source((extra >> 9) & 0x1ff, literal),
                Operand::source((extra >> 18) & 0x1ff, literal),
            ]);
            if opcode.is("VReadlaneB32") {
                destinations.push(Operand::scalar(word & 0xff));
            } else if opcode.is("VMadU64U32") {
                push_vector_range(&mut destinations, word & 0xff, 2);
            } else {
                destinations.push(Operand::vector(word & 0xff));
            }
            let is_b = is_vop3b_opcode((word >> 16) & 0x3ff);
            control = Some(InstructionControl::Vop3(Vop3Control {
                absolute_mask: if is_b { 0 } else { (word >> 8) & 7 },
                negate_mask: (extra >> 29) & 7,
                output_modifier: (extra >> 27) & 3,
                clamp: word >> 15 & 1 != 0,
                operand_select: if is_b { 0 } else { (word >> 11) & 0xf },
                scalar_destination: is_b.then_some((word >> 8) & 0x7f),
            }));
        }
        ShaderEncoding::Vop3p => {
            let extra = words[1];
            sources.extend([
                Operand::source(extra & 0x1ff, literal),
                Operand::source((extra >> 9) & 0x1ff, literal),
                Operand::source((extra >> 18) & 0x1ff, literal),
            ]);
            destinations.push(Operand::vector(word & 0xff));
            let operand_select_high = ((extra >> 27) & 3) | (((word >> 14) & 1) << 2);
            control = Some(InstructionControl::Vop3p(Vop3pControl {
                operand_select_mask: (word >> 11) & 7,
                operand_select_high_mask: operand_select_high,
                negate_low_mask: (extra >> 29) & 7,
                negate_high_mask: (word >> 8) & 7,
                clamp: word >> 15 & 1 != 0,
            }));
        }
        ShaderEncoding::Ds => {
            decode_data_share(word, words[1], opcode, &mut sources, &mut destinations);
            control = Some(InstructionControl::DataShare(DataShareControl {
                offset0: word & 0xff,
                offset1: (word >> 8) & 0xff,
                gds: word >> 17 & 1 != 0,
            }));
        }
        ShaderEncoding::Vintrp => {
            sources.push(Operand::vector(word & 0xff));
            destinations.push(Operand::vector((word >> 18) & 0xff));
            control = Some(InstructionControl::Interpolation(InterpolationControl {
                attribute: (word >> 10) & 0x3f,
                channel: (word >> 8) & 3,
            }));
        }
        ShaderEncoding::Flat => {
            control = Some(decode_global_memory(
                word,
                words[1],
                opcode,
                &mut sources,
                &mut destinations,
            ));
        }
        ShaderEncoding::Mubuf | ShaderEncoding::Mtbuf => {
            control = Some(decode_buffer_memory(
                word,
                words[1],
                opcode,
                literal,
                &mut sources,
                &mut destinations,
            ));
        }
        ShaderEncoding::Mimg => {
            control = Some(decode_image(
                word,
                &words,
                opcode,
                &mut sources,
                &mut destinations,
            ));
        }
        ShaderEncoding::Exp => {
            let extra = words[1];
            sources.extend([
                Operand::vector(extra & 0xff),
                Operand::vector((extra >> 8) & 0xff),
                Operand::vector((extra >> 16) & 0xff),
                Operand::vector((extra >> 24) & 0xff),
            ]);
            control = Some(InstructionControl::Export(ExportControl {
                target: (word >> 4) & 0x3f,
                enable_mask: word & 0xf,
                compressed: word >> 10 & 1 != 0,
                done: word >> 11 & 1 != 0,
                valid_mask: word >> 12 & 1 != 0,
            }));
        }
    }

    Instruction {
        pc,
        encoding,
        opcode,
        words,
        sources,
        destinations,
        control,
    }
}

#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
fn decode_vop_extension(
    word: u32,
    words: &[u32],
    literal: Option<u32>,
    is_sdwa: bool,
    is_dpp: bool,
    is_dpp8: bool,
    has_source1: bool,
    is_compare: bool,
    sources: &mut Vec<Operand>,
    destinations: &mut Vec<Operand>,
    control: &mut Option<InstructionControl>,
) {
    if is_dpp8 {
        let extra = words[1];
        sources.push(Operand::vector(extra & 0xff));
        if has_source1 {
            sources.push(Operand::vector((word >> 9) & 0xff));
        }
        *control = Some(InstructionControl::Dpp8(Dpp8Control {
            lane_selectors: extra >> 8,
            fetch_inactive: word & 0x1ff == 0xea,
        }));
    } else if is_dpp {
        let extra = words[1];
        sources.push(Operand::vector(extra & 0xff));
        if has_source1 {
            sources.push(Operand::vector((word >> 9) & 0xff));
        }
        *control = Some(InstructionControl::Dpp(create_dpp_control(extra)));
    } else if is_sdwa {
        let extra = words[1];
        let encoded_source0 = (extra & 0xff) + if extra >> 23 & 1 == 0 { 256 } else { 0 };
        sources.push(Operand::source(encoded_source0, None));
        if has_source1 {
            let encoded_source1 = ((word >> 9) & 0xff) + if extra >> 31 & 1 == 0 { 256 } else { 0 };
            sources.push(Operand::source(encoded_source1, None));
        }
        let sdwa = create_sdwa_control(extra, is_compare, has_source1);
        if is_compare
            && let Some(scalar_destination) = sdwa.scalar_destination
            && scalar_destination != 106
        {
            destinations.push(Operand::scalar(scalar_destination));
        }
        *control = Some(InstructionControl::Sdwa(sdwa));
    } else {
        sources.push(Operand::source(word & 0x1ff, literal));
        if has_source1 {
            sources.push(Operand::vector((word >> 9) & 0xff));
        }
    }
}

fn decode_data_share(
    _word: u32,
    extra: u32,
    opcode: Opcode,
    sources: &mut Vec<Operand>,
    destinations: &mut Vec<Operand>,
) {
    let vector_address = extra & 0xff;
    let vector_data0 = (extra >> 8) & 0xff;
    let vector_data1 = (extra >> 16) & 0xff;
    let vector_destination = (extra >> 24) & 0xff;
    sources.push(Operand::vector(vector_address));
    if opcode.is("DsWriteAddtidB32") {
        sources.clear();
        sources.push(Operand::vector(vector_data0));
    } else if opcode.is("DsWriteB32") {
        sources.push(Operand::vector(vector_data0));
    } else if opcode.is("DsWriteB64") {
        push_vector_range(sources, vector_data0, 2);
    } else if opcode.is("DsWriteB96") {
        push_vector_range(sources, vector_data0, 3);
    } else if opcode.is("DsWriteB128") {
        push_vector_range(sources, vector_data0, 4);
    } else if opcode.is("DsWrite2B32") || opcode.is("DsWrite2St64B32") {
        sources.extend([Operand::vector(vector_data0), Operand::vector(vector_data1)]);
    } else if opcode.is("DsSwizzleB32") {
        sources.clear();
        sources.push(Operand::vector(vector_data0));
    } else if opcode.is("DsCmpstB32") || opcode.is("DsCmpstRtnB32") {
        sources.extend([Operand::vector(vector_data0), Operand::vector(vector_data1)]);
    } else if is_data_share_atomic(opcode) {
        sources.push(Operand::vector(vector_data0));
    }

    if opcode.is("DsReadB32") || opcode.is("DsSwizzleB32") {
        destinations.push(Operand::vector(vector_destination));
    } else if opcode.is("DsRead2B32") || opcode.is("DsRead2St64B32") {
        push_vector_range(destinations, vector_destination, 2);
    } else if opcode.is("DsReadB96") {
        push_vector_range(destinations, vector_destination, 3);
    } else if opcode.is("DsReadB128") {
        push_vector_range(destinations, vector_destination, 4);
    } else if is_data_share_atomic(opcode) && opcode.contains("Rtn") {
        destinations.push(Operand::vector(vector_destination));
    }
}

fn decode_global_memory(
    word: u32,
    extra: u32,
    opcode: Opcode,
    sources: &mut Vec<Operand>,
    destinations: &mut Vec<Operand>,
) -> InstructionControl {
    let vector_address = extra & 0xff;
    let vector_data = (extra >> 8) & 0xff;
    let scalar_address_field = (extra >> 16) & 0x7f;
    let scalar_address = (scalar_address_field != 0x7f).then_some(scalar_address_field);
    let vector_destination = extra >> 24;
    let dword_count = memory_dword_count(opcode);
    sources.push(Operand::vector(vector_address));
    if let Some(scalar_address) = scalar_address {
        sources.push(Operand::scalar(scalar_address));
    } else {
        sources.push(Operand::vector(vector_address.saturating_add(1)));
    }
    let lds = word >> 13 & 1 != 0;
    if opcode.starts_with("GlobalLoad") {
        if !lds {
            push_vector_range(destinations, vector_destination, dword_count);
        }
    } else {
        push_vector_range(sources, vector_data, dword_count);
        if opcode.starts_with("GlobalAtomic") && word >> 16 & 1 != 0 {
            destinations.push(Operand::vector(vector_destination));
        }
    }
    InstructionControl::GlobalMemory(GlobalMemoryControl {
        dword_count,
        vector_address,
        vector_data,
        vector_destination,
        scalar_address,
        offset_bytes: sign_extend(word & 0xfff, 12),
        dlc: word >> 12 & 1 != 0,
        lds,
        glc: word >> 16 & 1 != 0,
        slc: word >> 17 & 1 != 0,
    })
}

fn decode_buffer_memory(
    word: u32,
    extra: u32,
    opcode: Opcode,
    literal: Option<u32>,
    sources: &mut Vec<Operand>,
    destinations: &mut Vec<Operand>,
) -> InstructionControl {
    let vector_address = extra & 0xff;
    let vector_data = (extra >> 8) & 0xff;
    let scalar_resource = ((extra >> 16) & 0x1f) * 4;
    let scalar_offset = (extra >> 24) & 0xff;
    let dword_count = memory_dword_count(opcode);
    sources.extend([
        Operand::vector(vector_address),
        Operand::scalar(scalar_resource),
        Operand::source(scalar_offset, literal),
    ]);
    let lds = !opcode.starts_with("TBuffer") && word >> 16 & 1 != 0;
    let tfe = extra >> 23 & 1 != 0;
    if opcode.contains("Load") {
        if !lds {
            push_vector_range(destinations, vector_data, dword_count);
            if tfe {
                destinations.push(Operand::vector(vector_data.saturating_add(dword_count)));
            }
        }
    } else {
        push_vector_range(sources, vector_data, dword_count);
        if opcode.starts_with("BufferAtomic") && word >> 14 & 1 != 0 {
            destinations.push(Operand::vector(vector_data));
        }
    }
    InstructionControl::BufferMemory(BufferMemoryControl {
        dword_count,
        vector_address,
        vector_data,
        scalar_resource,
        instruction_format: opcode.starts_with("TBuffer").then_some((word >> 19) & 0x7f),
        offset_bytes: i32::try_from(word & 0xfff).expect("12-bit buffer offset fits i32"),
        index_enabled: word >> 13 & 1 != 0,
        offset_enabled: word >> 12 & 1 != 0,
        dlc: word >> 15 & 1 != 0,
        lds,
        glc: word >> 14 & 1 != 0,
        slc: extra >> 22 & 1 != 0,
        tfe,
    })
}

fn decode_image(
    word: u32,
    words: &[u32],
    opcode: Opcode,
    sources: &mut Vec<Operand>,
    destinations: &mut Vec<Operand>,
) -> InstructionControl {
    let extra = words[1];
    let vector_address = extra & 0xff;
    let vector_data = (extra >> 8) & 0xff;
    let scalar_resource = ((extra >> 16) & 0x1f) * 4;
    let scalar_sampler = ((extra >> 21) & 0x1f) * 4;
    let mut address_registers = Vec::with_capacity(1 + words.len().saturating_sub(2) * 4);
    address_registers.push(vector_address);
    for extension in &words[2..] {
        for shift in [0, 8, 16, 24] {
            address_registers.push((extension >> shift) & 0xff);
        }
    }
    sources.extend(address_registers.iter().copied().map(Operand::vector));
    sources.extend([
        Operand::scalar(scalar_resource),
        Operand::scalar(scalar_sampler),
    ]);
    let result_count = (word >> 8 & 0xf).count_ones().max(1);
    let d16 = extra >> 31 & 1 != 0;
    let data_register_count = if d16 {
        result_count.div_ceil(2)
    } else {
        result_count
    };
    if opcode.starts_with("ImageStore") || opcode.starts_with("ImageAtomic") {
        let data_count = if opcode.is("ImageAtomicCmpswap") {
            2
        } else {
            data_register_count
        };
        push_vector_range(sources, vector_data, data_count);
        if opcode.starts_with("ImageAtomic") && word >> 13 & 1 != 0 {
            destinations.push(Operand::vector(vector_data));
        }
    } else {
        push_vector_range(destinations, vector_data, data_register_count);
    }
    let dimension = (word >> 3) & 7;
    InstructionControl::Image(ImageControl {
        dmask: (word >> 8) & 0xf,
        vector_address,
        address_registers,
        vector_data,
        scalar_resource,
        scalar_sampler,
        dimension,
        is_array: matches!(dimension, 4 | 5 | 7),
        glc: word >> 13 & 1 != 0,
        slc: word >> 25 & 1 != 0,
        a16: extra >> 30 & 1 != 0,
        d16,
    })
}

const fn create_dpp_control(word: u32) -> DppControl {
    DppControl {
        control: (word >> 8) & 0x1ff,
        fetch_inactive: word >> 18 & 1 != 0,
        bound_control: word >> 19 & 1 != 0,
        absolute_mask: ((word >> 21) & 1) | (((word >> 23) & 1) << 1),
        negate_mask: ((word >> 20) & 1) | (((word >> 22) & 1) << 1),
        bank_mask: (word >> 24) & 0xf,
        row_mask: (word >> 28) & 0xf,
    }
}

const fn create_sdwa_control(word: u32, is_compare: bool, has_source1: bool) -> SdwaControl {
    let scalar_destination = if is_compare {
        Some(if word >> 15 & 1 != 0 {
            (word >> 8) & 0x7f
        } else {
            106
        })
    } else {
        None
    };
    SdwaControl {
        destination_select: if is_compare { 6 } else { (word >> 8) & 7 },
        destination_unused: if is_compare { 0 } else { (word >> 11) & 3 },
        source0_select: (word >> 16) & 7,
        source1_select: if has_source1 { (word >> 24) & 7 } else { 6 },
        source0_sign_extend: word >> 19 & 1 != 0,
        source1_sign_extend: has_source1 && word >> 27 & 1 != 0,
        absolute_mask: ((word >> 21) & 1)
            | if has_source1 {
                ((word >> 29) & 1) << 1
            } else {
                0
            },
        negate_mask: ((word >> 20) & 1)
            | if has_source1 {
                ((word >> 28) & 1) << 1
            } else {
                0
            },
        output_modifier: if is_compare { 0 } else { (word >> 14) & 3 },
        clamp: !is_compare && word >> 13 & 1 != 0,
        scalar_destination,
    }
}

const fn minimum_encoding_dwords(encoding: ShaderEncoding) -> usize {
    match encoding {
        ShaderEncoding::Vop3
        | ShaderEncoding::Smem
        | ShaderEncoding::Mubuf
        | ShaderEncoding::Mtbuf
        | ShaderEncoding::Ds
        | ShaderEncoding::Flat
        | ShaderEncoding::Vop3p
        | ShaderEncoding::Mimg
        | ShaderEncoding::Exp => 2,
        _ => 1,
    }
}

fn scalar_load_dword_count(opcode: Opcode) -> u32 {
    match opcode.known_name() {
        Some("SLoadDword" | "SBufferLoadDword") => 1,
        Some("SLoadDwordx2" | "SBufferLoadDwordx2") => 2,
        Some("SLoadDwordx4" | "SBufferLoadDwordx4") => 4,
        Some("SLoadDwordx8" | "SBufferLoadDwordx8") => 8,
        Some("SLoadDwordx16" | "SBufferLoadDwordx16") => 16,
        _ => 0,
    }
}

fn memory_dword_count(opcode: Opcode) -> u32 {
    let Some(name) = opcode.known_name() else {
        return 0;
    };
    if name.contains("FormatD16") {
        if name.ends_with("Xyz") || name.ends_with("Xyzw") {
            2
        } else {
            1
        }
    } else if name.ends_with("Xyzw") || name.ends_with("Dwordx4") {
        4
    } else if name.ends_with("Xyz") || name.ends_with("Dwordx3") {
        3
    } else if name.ends_with("Xy") || name.ends_with("Dwordx2") || name.ends_with("Cmpswap") {
        2
    } else {
        1
    }
}

fn is_data_share_atomic(opcode: Opcode) -> bool {
    matches!(
        opcode.known_name(),
        Some(
            "DsAddU32"
                | "DsSubU32"
                | "DsIncU32"
                | "DsDecU32"
                | "DsMinI32"
                | "DsMaxI32"
                | "DsMinU32"
                | "DsMaxU32"
                | "DsAndB32"
                | "DsOrB32"
                | "DsXorB32"
                | "DsCmpstB32"
                | "DsAddRtnU32"
                | "DsSubRtnU32"
                | "DsIncRtnU32"
                | "DsDecRtnU32"
                | "DsMinRtnI32"
                | "DsMaxRtnI32"
                | "DsMinRtnU32"
                | "DsMaxRtnU32"
                | "DsAndRtnB32"
                | "DsOrRtnB32"
                | "DsXorRtnB32"
                | "DsWrxchgRtnB32"
                | "DsCmpstRtnB32"
        )
    )
}

const fn literal_operand(value: u32) -> Operand {
    Operand {
        kind: OperandKind::LiteralConstant,
        value,
    }
}

fn push_scalar_range(output: &mut Vec<Operand>, first: u32, count: u32) {
    output.extend((first..first.saturating_add(count)).map(Operand::scalar));
}

fn push_vector_range(output: &mut Vec<Operand>, first: u32, count: u32) {
    output.extend((first..first.saturating_add(count)).map(Operand::vector));
}

const fn sign_extend(value: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    (value << shift).cast_signed() >> shift
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SliceMemory {
        base: u64,
        words: Vec<u32>,
    }

    impl GuestMemoryReader for SliceMemory {
        fn read(&self, address: u64, output: &mut [u8]) -> bool {
            if output.len() != 4 {
                return false;
            }
            let Some(offset) = address.checked_sub(self.base) else {
                return false;
            };
            if offset & 3 != 0 {
                return false;
            }
            let Ok(index) = usize::try_from(offset / 4) else {
                return false;
            };
            let Some(word) = self.words.get(index) else {
                return false;
            };
            output.copy_from_slice(&word.to_le_bytes());
            true
        }
    }

    #[test]
    fn decodes_known_pixel_program_through_architectural_terminator() {
        let program = decode_program(
            0x1234_0000,
            &[
                0xd52f_0000,
                0x0000_0200,
                0xd52f_0001,
                0x0000_0602,
                0xf800_1c0f,
                0x0000_0100,
                0xbf81_0000,
                0xdead_beef,
            ],
        )
        .expect("well-formed shader");
        assert_eq!(program.instructions.len(), 4);
        assert!(program.instructions[0].opcode.is("VCvtPkrtzF16F32"));
        assert_eq!(program.instructions[0].encoding, ShaderEncoding::Vop3);
        assert_eq!(program.instructions[2].opcode, Opcode::EXPORT);
        assert_eq!(program.instructions[3].opcode, Opcode::END_PROGRAM);
        assert_eq!(program.pixel_color_export_masks(), 0x0f);
    }

    #[test]
    fn decodes_past_an_early_endpgm_needed_by_a_forward_branch() {
        let words = [
            0xbf82_0002, // s_branch from pc 0 to pc 12
            0xbf81_0000, // early fallthrough endpgm at pc 4
            0xbf80_0000, // padding/fallthrough instruction at pc 8
            0xbf80_0000, // forward target at pc 12
            0xbf81_0000, // terminating endpgm at pc 16
            0xdead_beef,
        ];

        let program = decode_program(0x1000, &words).expect("forward branch program");

        assert_eq!(program.instructions.len(), 5);
        assert_eq!(program.instructions[1].opcode, Opcode::END_PROGRAM);
        assert_eq!(program.instructions[3].pc, 12);
        assert_eq!(program.instructions[4].opcode, Opcode::END_PROGRAM);

        let memory = SliceMemory {
            base: 0x1000,
            words: words.to_vec(),
        };
        let program = read_program(&memory, memory.base).expect("memory-backed forward branch");
        assert_eq!(program.instructions.len(), 5);
    }

    #[test]
    fn registered_code_extent_allows_a_shader_larger_than_the_fallback_scan() {
        let mut words = vec![0xbf80_0000; FALLBACK_MAX_INSTRUCTIONS + 1];
        words.push(0xbf81_0000);
        let memory = SliceMemory {
            base: 0x4000,
            words,
        };
        assert!(matches!(
            read_program(&memory, memory.base),
            Err(DecodeError::Unterminated {
                inspected_instructions: FALLBACK_MAX_INSTRUCTIONS
            })
        ));

        let program = read_program_bounded(
            &memory,
            memory.base,
            memory.words.len() * std::mem::size_of::<u32>(),
        )
        .expect("AGC's registered shader size is the authoritative decoder boundary");
        assert_eq!(program.instructions.len(), FALLBACK_MAX_INSTRUCTIONS + 2);
        assert_eq!(
            program
                .instructions
                .last()
                .map(|instruction| &instruction.opcode),
            Some(&Opcode::END_PROGRAM)
        );
    }

    #[test]
    fn rejects_a_missing_literal_extension_without_reading_past_the_slice() {
        let error = decode_program(0x1000, &[0x7e00_02ff]).expect_err("missing literal");
        assert_eq!(
            error,
            DecodeError::Truncated {
                pc: 0,
                needed_dwords: 2,
                available_dwords: 1,
            }
        );
    }

    #[test]
    fn retains_framed_unknown_vop3_opcode_without_guessing_semantics() {
        let unknown_vop3 = 0xd000_0000;
        let program = decode_program(0x1000, &[unknown_vop3, 0, 0xbf81_0000])
            .expect("unknown VOP3 has an architecturally defined frame");
        assert_eq!(
            program.instructions[0].opcode,
            Opcode::raw(RawOpcodeFamily::Vop3, 0)
        );
        assert_eq!(program.instructions[1].pc, 8);
    }

    #[test]
    fn decodes_silent_hill_xor_add_and_unsigned_pack_vop3_opcodes() {
        let sources = 257_u32 | (258_u32 << 9) | (259_u32 << 18);
        let program = decode_program(
            0x1000,
            &[0xd345_0008, sources, 0xd36a_0009, sources, 0xbf81_0000],
        )
        .expect("observed RDNA2 VOP3 stream");
        assert!(program.instructions[0].opcode.is("VXadU32"));
        assert_eq!(
            program.instructions[0].sources,
            [Operand::vector(1), Operand::vector(2), Operand::vector(3)]
        );
        assert_eq!(program.instructions[0].destinations, [Operand::vector(8)]);
        assert!(program.instructions[1].opcode.is("VCvtPkU16U32"));
        assert_eq!(
            program.instructions[1].sources,
            [Operand::vector(1), Operand::vector(2), Operand::vector(3)]
        );
        assert_eq!(program.instructions[1].destinations, [Operand::vector(9)]);
    }

    #[test]
    fn decodes_silent_hill_half_to_u16_and_signed_bitfield_extract() {
        let sources = 257_u32 | (258_u32 << 9) | (259_u32 << 18);
        let program = decode_program(0x1000, &[0x7e06_a4f9, 0, 0xd149_0008, sources, 0xbf81_0000])
            .expect("observed RDNA2 VOP1 SDWA and VOP3 stream");

        assert!(program.instructions[0].opcode.is("VCvtU16F16"));
        assert_eq!(program.instructions[0].sources, [Operand::vector(0)]);
        assert_eq!(program.instructions[0].destinations, [Operand::vector(3)]);
        assert!(matches!(
            program.instructions[0].control,
            Some(InstructionControl::Sdwa(_))
        ));

        assert!(program.instructions[1].opcode.is("VBfeI32"));
        assert_eq!(
            program.instructions[1].sources,
            [Operand::vector(1), Operand::vector(2), Operand::vector(3)]
        );
        assert_eq!(program.instructions[1].destinations, [Operand::vector(8)]);
    }

    #[test]
    fn decodes_rdna2_floor_and_packed_unorm_conversion_encodings() {
        let unary_sources = 257_u32 | (128_u32 << 9) | (128_u32 << 18);
        let packed_sources = 257_u32 | (258_u32 << 9) | (128_u32 << 18);
        let program = decode_program(
            0x1000,
            &[
                // v_cvt_flr_i32_f32_e32 v8, v1
                0x7e10_1b01,
                // v_cvt_flr_i32_f32_e64 v9, -|v1|
                0xd18d_0109,
                unary_sources | (1_u32 << 29),
                // v_cvt_pknorm_u16_f32 v10, |v1|, -v2
                0xd369_010a,
                packed_sources | (2_u32 << 29),
                0xbf81_0000,
            ],
        )
        .expect("RDNA2 conversion stream");

        let floor_e32 = &program.instructions[0];
        assert_eq!(floor_e32.encoding, ShaderEncoding::Vop1);
        assert!(floor_e32.opcode.is("VCvtFlrI32F32"));
        assert_eq!(floor_e32.sources, [Operand::vector(1)]);
        assert_eq!(floor_e32.destinations, [Operand::vector(8)]);

        let floor_e64 = &program.instructions[1];
        assert_eq!(floor_e64.encoding, ShaderEncoding::Vop3);
        assert!(floor_e64.opcode.is("VCvtFlrI32F32"));
        assert_eq!(floor_e64.destinations, [Operand::vector(9)]);
        assert_eq!(
            floor_e64.control,
            Some(InstructionControl::Vop3(Vop3Control {
                absolute_mask: 1,
                negate_mask: 1,
                output_modifier: 0,
                clamp: false,
                operand_select: 0,
                scalar_destination: None,
            }))
        );

        let packed = &program.instructions[2];
        assert_eq!(packed.encoding, ShaderEncoding::Vop3);
        assert!(packed.opcode.is("VCvtPknormU16F32"));
        assert_eq!(
            packed.sources[..2],
            [Operand::vector(1), Operand::vector(2)]
        );
        assert_eq!(packed.destinations, [Operand::vector(10)]);
        assert_eq!(
            packed.control,
            Some(InstructionControl::Vop3(Vop3Control {
                absolute_mask: 1,
                negate_mask: 2,
                output_modifier: 0,
                clamp: false,
                operand_select: 0,
                scalar_destination: None,
            }))
        );
    }

    #[test]
    fn decodes_every_vop3p_opcode_and_every_modifier_bit_pattern() {
        const OPCODES: [(u32, &str); 8] = [
            (0x0e, "VPkFmaF16"),
            (0x0f, "VPkAddF16"),
            (0x10, "VPkMulF16"),
            (0x11, "VPkMinF16"),
            (0x12, "VPkMaxF16"),
            (0x20, "VFmaMixF32"),
            (0x21, "VFmaMixloF16"),
            (0x22, "VFmaMixhiF16"),
        ];
        let sources = 257u32 | (258u32 << 9) | (259u32 << 18);

        for (opcode, name) in OPCODES {
            let word = 0xcc00_0005 | (opcode << 16);
            let frame = decode_frame(0, &[word, sources]).expect("VOP3P frame");
            assert_eq!(frame.encoding, ShaderEncoding::Vop3p);
            assert!(frame.opcode.is(name));
            let instruction =
                create_instruction(0, frame.encoding, frame.opcode, vec![word, sources]);
            assert_eq!(
                instruction.sources,
                vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)]
            );
            assert_eq!(instruction.destinations, vec![Operand::vector(5)]);
        }

        // Five independent fields occupy thirteen encoding bits. Check all 8192 combinations so
        // a split OP_SEL_HI bit or swapped NEG lane cannot regress unnoticed.
        for bits in 0u32..(1 << 13) {
            let operand_select_mask = bits & 7;
            let operand_select_high_mask = (bits >> 3) & 7;
            let negate_low_mask = (bits >> 6) & 7;
            let negate_high_mask = (bits >> 9) & 7;
            let clamp = bits >> 12 != 0;
            let word = 0xcc0e_0005
                | (operand_select_mask << 11)
                | ((operand_select_high_mask >> 2) << 14)
                | (negate_high_mask << 8)
                | (u32::from(clamp) << 15);
            let extra = sources | ((operand_select_high_mask & 3) << 27) | (negate_low_mask << 29);
            let instruction = create_instruction(
                0,
                ShaderEncoding::Vop3p,
                Opcode::Known("VPkFmaF16"),
                vec![word, extra],
            );
            assert_eq!(
                instruction.control,
                Some(InstructionControl::Vop3p(Vop3pControl {
                    operand_select_mask,
                    operand_select_high_mask,
                    negate_low_mask,
                    negate_high_mask,
                    clamp,
                })),
                "incorrect VOP3P modifiers for packed bits 0x{bits:04X}"
            );
        }
    }

    #[test]
    fn decodes_every_supported_ds_opcode_and_every_control_bit_pattern() {
        const OPCODES: [(u32, &str, usize, usize); 38] = [
            (0x00, "DsAddU32", 2, 0),
            (0x01, "DsSubU32", 2, 0),
            (0x03, "DsIncU32", 2, 0),
            (0x04, "DsDecU32", 2, 0),
            (0x05, "DsMinI32", 2, 0),
            (0x06, "DsMaxI32", 2, 0),
            (0x07, "DsMinU32", 2, 0),
            (0x08, "DsMaxU32", 2, 0),
            (0x09, "DsAndB32", 2, 0),
            (0x0a, "DsOrB32", 2, 0),
            (0x0b, "DsXorB32", 2, 0),
            (0x0d, "DsWriteB32", 2, 0),
            (0x0e, "DsWrite2B32", 3, 0),
            (0x0f, "DsWrite2St64B32", 3, 0),
            (0x10, "DsCmpstB32", 3, 0),
            (0x20, "DsAddRtnU32", 2, 1),
            (0x21, "DsSubRtnU32", 2, 1),
            (0x23, "DsIncRtnU32", 2, 1),
            (0x24, "DsDecRtnU32", 2, 1),
            (0x25, "DsMinRtnI32", 2, 1),
            (0x26, "DsMaxRtnI32", 2, 1),
            (0x27, "DsMinRtnU32", 2, 1),
            (0x28, "DsMaxRtnU32", 2, 1),
            (0x29, "DsAndRtnB32", 2, 1),
            (0x2a, "DsOrRtnB32", 2, 1),
            (0x2b, "DsXorRtnB32", 2, 1),
            (0x2d, "DsWrxchgRtnB32", 2, 1),
            (0x30, "DsCmpstRtnB32", 3, 1),
            (0x35, "DsSwizzleB32", 1, 1),
            (0x36, "DsReadB32", 1, 1),
            (0x37, "DsRead2B32", 1, 2),
            (0x38, "DsRead2St64B32", 1, 2),
            (0x4d, "DsWriteB64", 3, 0),
            (0xb0, "DsWriteAddtidB32", 1, 0),
            (0xde, "DsWriteB96", 4, 0),
            (0xdf, "DsWriteB128", 5, 0),
            (0xfe, "DsReadB96", 1, 3),
            (0xff, "DsReadB128", 1, 4),
        ];
        let extra = 1u32 | (2u32 << 8) | (3u32 << 16) | (4u32 << 24);
        for (opcode, name, source_count, destination_count) in OPCODES {
            let word = 0xd800_0000 | (opcode << 18);
            let frame = decode_frame(0, &[word, extra]).expect("DS frame");
            assert_eq!(frame.encoding, ShaderEncoding::Ds);
            assert!(frame.opcode.is(name));
            let instruction =
                create_instruction(0, frame.encoding, frame.opcode, vec![word, extra]);
            assert_eq!(instruction.sources.len(), source_count, "{name} sources");
            assert_eq!(
                instruction.destinations.len(),
                destination_count,
                "{name} destinations"
            );
        }

        // OFFSET0, OFFSET1, and GDS are independent fields. Cover all 131072 combinations to
        // prevent a one-bit opcode shift or a truncated concatenated byte offset from regressing.
        for bits in 0u32..(1 << 17) {
            let word = 0xd800_0000 | (bits & 0xffff) | ((bits >> 16) << 17);
            let instruction = create_instruction(
                0,
                ShaderEncoding::Ds,
                Opcode::Known("DsReadB32"),
                vec![word, extra],
            );
            assert_eq!(
                instruction.control,
                Some(InstructionControl::DataShare(DataShareControl {
                    offset0: bits & 0xff,
                    offset1: (bits >> 8) & 0xff,
                    gds: bits >> 16 != 0,
                })),
                "incorrect DS control for packed bits 0x{bits:05X}"
            );
        }
    }

    #[test]
    fn decodes_silent_hill_ds_write_addtid_words_without_an_address_source() {
        let extra = 1u32 | (2u32 << 8) | (3u32 << 16) | (4u32 << 24);
        for (word, expected_offset1) in [
            (0xdac0_0000, 0_u32),
            (0xdac0_0100, 1_u32),
            (0xdac0_0200, 2_u32),
        ] {
            let program = decode_program(0x1000, &[word, extra, 0xbf81_0000])
                .expect("observed Silent Hill DS_WRITE_ADDTID_B32 stream");
            let instruction = &program.instructions[0];

            assert!(instruction.opcode.is("DsWriteAddtidB32"));
            assert_eq!(
                instruction.sources,
                [Operand::vector(2)],
                "ADDR and DATA1 are architecturally ignored"
            );
            assert!(instruction.destinations.is_empty());
            assert_eq!(
                instruction.control,
                Some(InstructionControl::DataShare(DataShareControl {
                    offset0: 0,
                    offset1: expected_offset1,
                    gds: false,
                }))
            );
        }
    }

    #[test]
    fn decodes_complete_mubuf_and_mtbuf_surfaces_with_all_control_fields() {
        const MUBUF: &[(u32, &str)] = &[
            (0x00, "BufferLoadFormatX"),
            (0x01, "BufferLoadFormatXy"),
            (0x02, "BufferLoadFormatXyz"),
            (0x03, "BufferLoadFormatXyzw"),
            (0x04, "BufferStoreFormatX"),
            (0x05, "BufferStoreFormatXy"),
            (0x06, "BufferStoreFormatXyz"),
            (0x07, "BufferStoreFormatXyzw"),
            (0x08, "BufferLoadUbyte"),
            (0x09, "BufferLoadSbyte"),
            (0x0a, "BufferLoadUshort"),
            (0x0b, "BufferLoadSshort"),
            (0x0c, "BufferLoadDword"),
            (0x0d, "BufferLoadDwordx2"),
            (0x0e, "BufferLoadDwordx4"),
            (0x0f, "BufferLoadDwordx3"),
            (0x18, "BufferStoreByte"),
            (0x19, "BufferStoreByteD16Hi"),
            (0x1a, "BufferStoreShort"),
            (0x1b, "BufferStoreShortD16Hi"),
            (0x1c, "BufferStoreDword"),
            (0x1d, "BufferStoreDwordx2"),
            (0x1e, "BufferStoreDwordx4"),
            (0x1f, "BufferStoreDwordx3"),
            (0x20, "BufferLoadUbyteD16"),
            (0x21, "BufferLoadUbyteD16Hi"),
            (0x22, "BufferLoadSbyteD16"),
            (0x23, "BufferLoadSbyteD16Hi"),
            (0x24, "BufferLoadShortD16"),
            (0x25, "BufferLoadShortD16Hi"),
            (0x26, "BufferLoadFormatD16HiX"),
            (0x27, "BufferStoreFormatD16HiX"),
            (0x30, "BufferAtomicSwap"),
            (0x31, "BufferAtomicCmpswap"),
            (0x32, "BufferAtomicAdd"),
            (0x33, "BufferAtomicSub"),
            (0x35, "BufferAtomicSmin"),
            (0x36, "BufferAtomicUmin"),
            (0x37, "BufferAtomicSmax"),
            (0x38, "BufferAtomicUmax"),
            (0x39, "BufferAtomicAnd"),
            (0x3a, "BufferAtomicOr"),
            (0x3b, "BufferAtomicXor"),
            (0x3c, "BufferAtomicInc"),
            (0x3d, "BufferAtomicDec"),
            (0x80, "BufferLoadFormatD16X"),
            (0x81, "BufferLoadFormatD16Xy"),
            (0x82, "BufferLoadFormatD16Xyz"),
            (0x83, "BufferLoadFormatD16Xyzw"),
            (0x84, "BufferStoreFormatD16X"),
            (0x85, "BufferStoreFormatD16Xy"),
            (0x86, "BufferStoreFormatD16Xyz"),
            (0x87, "BufferStoreFormatD16Xyzw"),
        ];
        const MTBUF: [&str; 16] = [
            "TBufferLoadFormatX",
            "TBufferLoadFormatXy",
            "TBufferLoadFormatXyz",
            "TBufferLoadFormatXyzw",
            "TBufferStoreFormatX",
            "TBufferStoreFormatXy",
            "TBufferStoreFormatXyz",
            "TBufferStoreFormatXyzw",
            "TBufferLoadFormatD16X",
            "TBufferLoadFormatD16Xy",
            "TBufferLoadFormatD16Xyz",
            "TBufferLoadFormatD16Xyzw",
            "TBufferStoreFormatD16X",
            "TBufferStoreFormatD16Xy",
            "TBufferStoreFormatD16Xyz",
            "TBufferStoreFormatD16Xyzw",
        ];
        let extra = 1u32 | (16u32 << 8) | (1u32 << 22) | (1u32 << 23) | (128u32 << 24);
        for &(opcode, name) in MUBUF {
            let word = 0xe000_0000 | (opcode << 18) | 0x0000_fabc;
            let frame = decode_frame(0, &[word, extra]).expect("MUBUF frame");
            assert_eq!(frame.encoding, ShaderEncoding::Mubuf);
            assert!(frame.opcode.is(name), "opcode 0x{opcode:02X}");
        }
        for (opcode, name) in MTBUF.into_iter().enumerate() {
            let opcode = u32::try_from(opcode).expect("small MTBUF opcode");
            let word = 0xe800_0000 | ((opcode & 7) << 16) | (60 << 19) | 0x0000_cabc;
            let extra = extra | ((opcode >> 3) << 21);
            let frame = decode_frame(0, &[word, extra]).expect("MTBUF frame");
            assert_eq!(frame.encoding, ShaderEncoding::Mtbuf);
            assert!(frame.opcode.is(name), "opcode 0x{opcode:02X}");
            let instruction =
                create_instruction(0, frame.encoding, frame.opcode, vec![word, extra]);
            let Some(InstructionControl::BufferMemory(control)) = instruction.control else {
                panic!("expected MTBUF control");
            };
            assert_eq!(control.instruction_format, Some(60));
            assert!(!control.lds, "MTBUF OP bits must not alias MUBUF LDS");
            assert!(control.dlc && control.glc && control.slc && control.tfe);
        }
    }

    #[test]
    fn decodes_global_vdst_vdata_address_modes_and_twelve_bit_offset() {
        let word =
            0xdc00_0000 | (0x32 << 18) | (2 << 14) | (1 << 12) | (1 << 16) | (1 << 17) | 0x800;
        let extra = 1 | (16 << 8) | (0x7f << 16) | (32 << 24);
        let frame = decode_frame(0, &[word, extra]).expect("GLOBAL frame");
        let instruction = create_instruction(0, frame.encoding, frame.opcode, vec![word, extra]);
        assert_eq!(instruction.sources[0], Operand::vector(1));
        assert_eq!(instruction.sources[1], Operand::vector(2));
        assert_eq!(instruction.sources[2], Operand::vector(16));
        assert_eq!(instruction.destinations, [Operand::vector(32)]);
        assert_eq!(
            instruction.control,
            Some(InstructionControl::GlobalMemory(GlobalMemoryControl {
                dword_count: 1,
                vector_address: 1,
                vector_data: 16,
                vector_destination: 32,
                scalar_address: None,
                offset_bytes: -2048,
                dlc: true,
                lds: false,
                glc: true,
                slc: true,
            }))
        );
    }

    #[test]
    fn rejects_reserved_scalar_opcode_instead_of_emitting_a_placeholder() {
        let error = decode_program(0x1000, &[0xbe80_0000]).expect_err("reserved SOP1 opcode");
        assert!(matches!(
            error,
            DecodeError::UnknownOpcode {
                encoding: ShaderEncoding::Sop1,
                ..
            }
        ));
    }

    #[test]
    fn decodes_s_bcnt1_i32_b64_from_the_silent_hill_word() {
        let program =
            decode_program(0x1000, &[0xbe8a_1000, 0xbf81_0000]).expect("well-formed SOP1 program");
        let instruction = &program.instructions[0];

        assert!(instruction.opcode.is("SBcnt1I32B64"));
        assert_eq!(instruction.sources, [Operand::scalar(0)]);
        assert_eq!(instruction.destinations, [Operand::scalar(10)]);
    }

    #[test]
    fn records_all_mimg_nsa_address_registers() {
        // IMAGE_LOAD with NSA=2 carries eight additional byte-packed VGPR indices.
        let word = 0xf000_0004;
        let program = decode_program(0x1000, &[word, 0, 0x0403_0201, 0x0807_0605, 0xbf81_0000])
            .expect("MIMG NSA program");
        let InstructionControl::Image(image) = program.instructions[0]
            .control
            .as_ref()
            .expect("image control")
        else {
            panic!("expected image control");
        };
        assert_eq!(image.address_registers, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(program.instructions[1].pc, 16);
    }

    #[test]
    fn image_d16_liveness_uses_one_vgpr_for_each_packed_component_pair() {
        let extra = 1 | (20 << 8) | (1 << 31);
        for (opcode, name) in [(0_u32, "ImageLoad"), (8, "ImageStore")] {
            let word = 0xf000_0000 | (opcode << 18) | (0xf << 8);
            let instruction = create_instruction(
                0,
                ShaderEncoding::Mimg,
                Opcode::Known(name),
                vec![word, extra],
            );
            let Some(InstructionControl::Image(control)) = instruction.control else {
                panic!("expected image control");
            };
            assert!(control.d16);
            if name == "ImageLoad" {
                assert_eq!(
                    instruction.destinations,
                    [Operand::vector(20), Operand::vector(21)]
                );
            } else {
                assert_eq!(
                    &instruction.sources[instruction.sources.len() - 2..],
                    [Operand::vector(20), Operand::vector(21)]
                );
                assert!(!instruction.sources.contains(&Operand::vector(22)));
            }
        }
    }

    #[test]
    fn retains_high_mimg_opcodes_without_aliasing_low_operations() {
        for (opcode, nsa_dwords) in [(0x80_u32, 0_u32), (0xa2, 1), (0xe6, 3)] {
            let word =
                0xf000_0000 | ((opcode & 0x7f) << 18) | ((opcode >> 7) & 1) | (nsa_dwords << 1);
            let mut words = vec![word, 0];
            words.resize(2 + usize::try_from(nsa_dwords).expect("small NSA count"), 0);
            words.push(0xbf81_0000);

            let program = decode_program(0x1000, &words).expect("framed high MIMG opcode");
            assert_eq!(
                program.instructions[0].opcode,
                Opcode::Raw {
                    family: RawOpcodeFamily::Mimg,
                    code: u16::try_from(opcode).expect("eight-bit MIMG opcode fits u16"),
                }
            );
            assert_eq!(
                program.instructions[0].words.len(),
                2 + usize::try_from(nsa_dwords).expect("small NSA count")
            );
            assert_eq!(program.instructions[1].pc, (2 + nsa_dwords) * 4);
            assert!(
                !program.instructions[0].opcode.is("ImageLoad")
                    && !program.instructions[0].opcode.is("ImageSampleD"),
                "high opcode 0x{opcode:02X} aliased a low operation"
            );
        }
    }

    #[test]
    fn decodes_absdiff_at_the_rdna2_sop2_opcode() {
        let word = 0x8000_0000 | (44 << 23);
        let frame = decode_frame(0, &[word]).expect("S_ABSDIFF_I32 opcode 44");
        assert_eq!(frame.encoding, ShaderEncoding::Sop2);
        assert!(frame.opcode.is("SAbsdiffI32"));

        let reserved = 0x8000_0000 | (45 << 23);
        assert!(matches!(
            decode_frame(0, &[reserved]),
            Err(DecodeError::UnknownOpcode {
                encoding: ShaderEncoding::Sop2,
                opcode: 45,
                ..
            })
        ));
    }

    #[test]
    fn memory_reader_stops_exactly_at_endpgm() {
        let memory = SliceMemory {
            base: 0x8000,
            words: vec![0xbf81_0000],
        };
        let program = read_program(&memory, memory.base).expect("single-word program");
        assert_eq!(program.instructions.len(), 1);
        assert_eq!(program.instructions[0].opcode, Opcode::END_PROGRAM);
    }

    #[test]
    fn rdna2_code_end_terminates_padding_and_half_float_vop2_opcodes_decode() {
        let program = decode_program(
            0x8000,
            &[
                0x6406_0601, // v_add_f16 v3, v1, v3
                0x6600_00f9, // v_sub_f16 v0, src_sdwa, v0
                0,
                0x6812_0202, // v_subrev_f16 v9, v2, v1
                0x6a12_0202, // v_mul_f16 v9, v2, v1
                0x7404_04f9, // v_min_f16 v2, src_sdwa, v2
                0,
                0xd74b_600a, // v_fma_f16 v10, v240, v12, v10 op_sel:[0,0,1,1]
                0x042a_18f0,
                0xbf13_1e6a, // s_cmp_lg_u64 s106:s107, s30:s31
                0x7e12_acf9, // v_rsq_f16 v9, src_dpp
                0,
                0x7d98_04f9, // v_cmp_gt_f16 src_dpp, v2
                0,
                0x7dfa_04f9, // v_cmpx_neq_f16 src_dpp, v2
                0,
                0x7dfc_62f9, // v_cmpx_nlt_f16 src_dpp, v49
                0,
                0xbf9f_0000, // s_code_end
                0,
            ],
        )
        .expect("observed RDNA2 shader subset");

        assert!(program.instructions[0].opcode.is("VAddF16"));
        assert!(program.instructions[1].opcode.is("VSubF16"));
        assert!(program.instructions[2].opcode.is("VSubrevF16"));
        assert!(program.instructions[3].opcode.is("VMulF16"));
        assert!(program.instructions[4].opcode.is("VMinF16"));
        assert!(program.instructions[5].opcode.is("VFmaF16"));
        assert!(program.instructions[6].opcode.is("SCmpLgU64"));
        assert!(program.instructions[7].opcode.is("VRsqF16"));
        assert!(program.instructions[8].opcode.is("VCmpGtF16"));
        assert!(program.instructions[9].opcode.is("VCmpxNeqF16"));
        assert!(program.instructions[10].opcode.is("VCmpxNltF16"));
        assert_eq!(program.instructions[11].opcode, Opcode::END_PROGRAM);
        assert_eq!(program.instructions.len(), 12);
    }

    #[test]
    fn decodes_silent_hill_lane_permute_bitcount_and_u64_mad_shapes() {
        let v_bcnt = (0x22_u32 << 25) | (4_u32 << 17) | (3_u32 << 9) | (0x100_u32 + 2);
        let v_permlane = (0x35_u32 << 26) | (0x377_u32 << 16) | (2_u32 << 11) | 8;
        let v_permlanex = (0x35_u32 << 26) | (0x378_u32 << 16) | (1_u32 << 11) | 9;
        let selectors = (0x100_u32 + 5) | (20_u32 << 9) | (21_u32 << 18);
        let v_mad_u64 = (0x35_u32 << 26) | (0x176_u32 << 16) | (30_u32 << 8) | 40;
        let mad_sources = (0x100_u32 + 6) | ((0x100_u32 + 7) << 9) | ((0x100_u32 + 50) << 18);
        let program = decode_program(
            0xa000,
            &[
                v_bcnt,
                v_permlane,
                selectors,
                v_permlanex,
                selectors,
                v_mad_u64,
                mad_sources,
                0xbf81_0000,
            ],
        )
        .expect("observed Silent Hill compute opcode subset");

        let bcnt = &program.instructions[0];
        assert!(bcnt.opcode.is("VBcntU32B32"));
        assert_eq!(bcnt.sources, [Operand::vector(2), Operand::vector(3)]);
        assert_eq!(bcnt.destinations, [Operand::vector(4)]);

        for (index, opcode, operand_select, destination) in
            [(1, "VPermlane16B32", 2, 8), (2, "VPermlanex16B32", 1, 9)]
        {
            let instruction = &program.instructions[index];
            assert!(instruction.opcode.is(opcode));
            assert_eq!(
                instruction.sources,
                [Operand::vector(5), Operand::scalar(20), Operand::scalar(21)]
            );
            assert_eq!(instruction.destinations, [Operand::vector(destination)]);
            let Some(InstructionControl::Vop3(control)) = instruction.control else {
                panic!("lane permutation must retain VOP3 control");
            };
            assert_eq!(control.operand_select, operand_select);
        }

        let mad = &program.instructions[3];
        assert!(mad.opcode.is("VMadU64U32"));
        assert_eq!(
            mad.sources,
            [Operand::vector(6), Operand::vector(7), Operand::vector(50)]
        );
        assert_eq!(mad.destinations, [Operand::vector(40), Operand::vector(41)]);
        let Some(InstructionControl::Vop3(control)) = mad.control else {
            panic!("64-bit MAD must retain VOP3B control");
        };
        assert_eq!(control.scalar_destination, Some(30));
    }

    #[test]
    fn decodes_clean_room_ps5_extension_regression_envelopes() {
        // These are synthetic raw envelopes from the clean-room Ps5IsaExtensions manifest, not
        // retained guest captures. They are still valuable regressions for the PS5 opcode holes:
        // the two constant MAD forms must consume their unconditional literal or every following
        // instruction in the shader is decoded at the wrong address.
        let vop2 = decode_program(
            0x9000,
            &[
                0x3e00_0501, // v_mac_f32 v0, v1, v2
                0x4000_0501, // v_madmk_f32 v0, v1, 1.0, v2
                0x3f80_0000,
                0x4200_0501, // v_madak_f32 v0, v1, v2, 1.0
                0x3f80_0000,
                0xbf81_0000,
            ],
        )
        .expect("synthetic PS5 VOP2 extension envelopes");
        assert!(vop2.instructions[0].opcode.is("VMacF32"));
        assert_eq!(vop2.instructions[0].pc, 0);
        assert_eq!(vop2.instructions[0].words.len(), 1);
        assert!(vop2.instructions[1].opcode.is("VMadMkF32"));
        assert_eq!(vop2.instructions[1].pc, 4);
        assert_eq!(vop2.instructions[1].words, [0x4000_0501, 0x3f80_0000]);
        assert_eq!(vop2.instructions[1].sources.len(), 3);
        assert!(vop2.instructions[2].opcode.is("VMadAkF32"));
        assert_eq!(vop2.instructions[2].pc, 12);
        assert_eq!(vop2.instructions[2].words, [0x4200_0501, 0x3f80_0000]);
        assert_eq!(vop2.instructions[2].sources.len(), 3);
        assert_eq!(vop2.instructions[3].pc, 20);
        assert_eq!(vop2.instructions[3].opcode, Opcode::END_PROGRAM);

        let vop3 = decode_program(
            0xa000,
            &[
                0xd541_0000, // PS5 VOP3A opcode 321: v_mad_f32 v0, v1, v2, v3
                0x040e_0501,
                0xbf81_0000,
            ],
        )
        .expect("synthetic PS5 VOP3 extension envelope");
        assert!(vop3.instructions[0].opcode.is("VMadF32"));
        assert_eq!(vop3.instructions[0].words, [0xd541_0000, 0x040e_0501]);
        assert_eq!(vop3.instructions[0].sources.len(), 3);
        assert_eq!(vop3.instructions[1].pc, 8);
    }
}
