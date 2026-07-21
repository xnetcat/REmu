// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::ir::{
    BufferMemoryControl, DataShareControl, Dpp8Control, DppControl, ExportControl,
    GlobalMemoryControl, ImageControl, Instruction, InstructionControl, InterpolationControl,
    Opcode, Operand, OperandKind, Program, RawOpcodeFamily, ScalarMemoryControl, SdwaControl,
    ShaderEncoding, Vop3Control, Vop3pControl,
};

const MAX_INSTRUCTIONS: usize = 4096;

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
    #[error("shader has no S_ENDPGM within {MAX_INSTRUCTIONS} instructions")]
    Unterminated,
    #[error("shader program counter exceeds the 32-bit Gen5 address space")]
    ProgramCounterOverflow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Frame {
    encoding: ShaderEncoding,
    opcode: Opcode,
    dwords: usize,
}

/// Decodes one bounded Gen5/RDNA shader stream. The input must end at an `S_ENDPGM`; trailing
/// allocation contents are deliberately ignored after that architectural terminator.
pub fn decode_program(address: u64, words: &[u32]) -> Result<Program, DecodeError> {
    if words.is_empty() {
        return Err(DecodeError::Empty);
    }

    let mut instructions = Vec::new();
    let mut word_index = 0usize;
    while instructions.len() < MAX_INSTRUCTIONS {
        let pc = u32::try_from(
            word_index
                .checked_mul(4)
                .ok_or(DecodeError::ProgramCounterOverflow)?,
        )
        .map_err(|_| DecodeError::ProgramCounterOverflow)?;
        let remaining = words.get(word_index..).ok_or(DecodeError::Unterminated)?;
        if remaining.is_empty() {
            return Err(DecodeError::Unterminated);
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
        word_index += frame.dwords;
        let ended = instruction.opcode == Opcode::END_PROGRAM;
        instructions.push(instruction);
        if ended {
            return Ok(Program::new(address, instructions));
        }
    }
    Err(DecodeError::Unterminated)
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
        0x3c => decode_mimg(pc, word),
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
        0x2d => Some("SAbsdiffI32"),
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
        0xd0 => Some("VCmpxFU32"),
        0xd1 => Some("VCmpxLtU32"),
        0xd2 => Some("VCmpxEqU32"),
        0xd3 => Some("VCmpxLeU32"),
        0xd4 => Some("VCmpxGtU32"),
        0xd5 => Some("VCmpxNeU32"),
        0xd6 => Some("VCmpxGeU32"),
        0xd7 => Some("VCmpxTU32"),
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
            0x346 => Some("VLshlAddU32"),
            0x347 => Some("VAddLshlU32"),
            0x360 => Some("VReadlaneB32"),
            0x361 => Some("VWritelaneB32"),
            0x362 => Some("VLdexpF32"),
            0x363 => Some("VBfmB32"),
            0x364 => Some("VBcntU32B32"),
            0x365 => Some("VMbcntLoU32B32"),
            0x366 => Some("VMbcntHiU32B32"),
            0x368 => Some("VCvtPknormI16F32"),
            0x369 => Some("VCvtPknormU16F32"),
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
        0xde => Some("DsWriteB96"),
        0xdf => Some("DsWriteB128"),
        0xfe => Some("DsReadB96"),
        0xff => Some("DsReadB128"),
        _ => None,
    };
    known_frame(pc, word, ShaderEncoding::Ds, code, name, 2)
}

fn decode_mtbuf(word: u32, extra: u32) -> Frame {
    let code = (word >> 16) & 0x7;
    let name = match code {
        0x00 => "TBufferLoadFormatX",
        0x01 => "TBufferLoadFormatXy",
        0x02 => "TBufferLoadFormatXyz",
        0x03 => "TBufferLoadFormatXyzw",
        0x04 => "TBufferStoreFormatX",
        0x05 => "TBufferStoreFormatXy",
        0x06 => "TBufferStoreFormatXyz",
        0x07 => "TBufferStoreFormatXyzw",
        _ => unreachable!("three-bit opcode"),
    };
    Frame {
        encoding: ShaderEncoding::Mtbuf,
        opcode: Opcode::from_static(name),
        dwords: 2 + usize::from(extra >> 24 == 0xff),
    }
}

fn decode_mubuf(word: u32, extra: u32) -> Frame {
    let code = (word >> 18) & 0x7f;
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

fn decode_mimg(pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 18) & 0x7f;
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
    known_frame(
        pc,
        word,
        ShaderEncoding::Mimg,
        code,
        name,
        2 + ((word >> 1) & 3) as usize,
    )
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
                if (opcode.is("VMadMkF32") || opcode.is("VFmaMkF32"))
                    && let Some(literal) = literal
                {
                    sources.insert(1, literal_operand(literal));
                } else if (opcode.is("VMadAkF32") || opcode.is("VFmaAkF32"))
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
    if opcode.is("DsWriteB32") {
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
    let scalar_address = (extra >> 16) & 0x7f;
    let dword_count = memory_dword_count(opcode);
    sources.extend([
        Operand::vector(vector_address),
        Operand::scalar(scalar_address),
    ]);
    if opcode.starts_with("GlobalLoad") {
        push_vector_range(destinations, vector_data, dword_count);
    } else {
        push_vector_range(sources, vector_data, dword_count);
        if opcode.starts_with("GlobalAtomic") && word >> 16 & 1 != 0 {
            destinations.push(Operand::vector(vector_data));
        }
    }
    InstructionControl::GlobalMemory(GlobalMemoryControl {
        dword_count,
        vector_address,
        vector_data,
        scalar_address,
        offset_bytes: sign_extend(word & 0x1fff, 13),
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
    if opcode.contains("Load") {
        push_vector_range(destinations, vector_data, dword_count);
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
        offset_bytes: i32::try_from(word & 0xfff).expect("12-bit buffer offset fits i32"),
        index_enabled: word >> 13 & 1 != 0,
        offset_enabled: word >> 12 & 1 != 0,
        glc: word >> 14 & 1 != 0,
        slc: extra >> 22 & 1 != 0,
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
    if opcode.starts_with("ImageStore") || opcode.starts_with("ImageAtomic") {
        let data_count = if opcode.is("ImageAtomicCmpswap") {
            2
        } else {
            result_count
        };
        push_vector_range(sources, vector_data, data_count);
        if opcode.starts_with("ImageAtomic") && word >> 13 & 1 != 0 {
            destinations.push(Operand::vector(vector_data));
        }
    } else {
        push_vector_range(destinations, vector_data, result_count);
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
        d16: extra >> 31 & 1 != 0,
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
    if name.ends_with("Xyzw") || name.ends_with("Dwordx4") {
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
}
