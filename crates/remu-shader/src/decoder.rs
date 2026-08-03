// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

mod memory;
mod program;
mod scalar;
mod vector;

use memory::{
    decode_buffer_memory, decode_data_share, decode_ds, decode_flat, decode_global_memory,
    decode_image, decode_mimg, decode_mtbuf, decode_mubuf,
};
pub use program::{
    decode_program, decode_program_segments, read_program, read_program_bounded,
    read_program_segments_bounded,
};
use scalar::{
    decode_smem, decode_smrd, decode_sop1, decode_sop2, decode_sopc, decode_sopk, decode_sopp,
    scalar_load_dword_count,
};
use vector::{
    decode_vintrp, decode_vop_extension, decode_vop1, decode_vop2, decode_vop3, decode_vop3p,
    decode_vopc, is_vop3b_opcode, vop1_writes_vector_pair,
};

#[cfg(test)]
use crate::ir::GlobalMemoryControl;
use crate::ir::{
    DataShareControl, ExportControl, Instruction, InstructionControl, InterpolationControl, Opcode,
    Operand, OperandKind, RawOpcodeFamily, ScalarMemoryControl, ShaderEncoding, Vop3Control,
    Vop3pControl,
};
#[cfg(test)]
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

/// Frames one instruction whose length is fully determined by the encoding word itself.
///
/// An opcode outside the semantic table decodes as a typed [`Opcode::Raw`] value instead of
/// failing the whole shader: the framing stays exact, the evaluator refuses raw scalar
/// operations loudly, and a backend reports the precise per-opcode translation gap.
#[allow(clippy::cast_possible_truncation)]
fn frame_or_raw(
    encoding: ShaderEncoding,
    family: RawOpcodeFamily,
    code: u32,
    name: Option<&'static str>,
    dwords: usize,
) -> Frame {
    Frame {
        encoding,
        opcode: name.map_or_else(|| Opcode::raw(family, code as u16), Opcode::from_static),
        dwords,
    }
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
            if opcode.is("SVersion") {
                // A pure toolchain hint: the SDST field carries no register operand.
            } else if opcode.is("SSetregB32") {
                // The SDST field is the value READ into the hardware register.
                sources.push(Operand::scalar((word >> 16) & 0x7f));
            } else if opcode.is("SSetregImm32B32") {
                if let Some(literal) = literal {
                    sources.push(literal_operand(literal));
                }
            } else if matches!(
                opcode.known_name(),
                Some("SWaitcntVscnt" | "SWaitcntVmcnt" | "SWaitcntExpcnt" | "SWaitcntLgkmcnt")
            ) {
                // The SDST field supplies an additional wait count and is read, not written.
                sources.push(Operand::source((word >> 16) & 0x7f, None));
            } else {
                destinations.push(Operand::scalar((word >> 16) & 0x7f));
            }
        }
        ShaderEncoding::Smrd if opcode.known_name().is_none() => {
            // A raw scalar-memory opcode has unknown field semantics; fabricating operands or a
            // load control from them could invent a binding. The backend reports the gap.
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
        ShaderEncoding::Smem if opcode.known_name().is_none() => {
            // Same rule as raw SMRD: nothing may be fabricated from unknown fields.
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
            } else if vop1_writes_vector_pair(opcode) {
                push_vector_range(&mut destinations, (word >> 17) & 0xff, 2);
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
            } else if opcode.is("VMadU64U32")
                || opcode.is("VLshlrevB64")
                || opcode.is("VLshrrevB64")
            {
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
        ShaderEncoding::Flat if opcode.known_name().is_none() => {
            // Raw FLAT-encoded operations keep exact framing but expose no operands: an unknown
            // operation's address fields must never turn into a guest-memory binding.
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
mod tests;
