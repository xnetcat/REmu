// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Vector encoding families: VOP1/VOP2/VOPC/VOP3/VOP3P, VINTRP and the DPP/SDWA extensions.

use crate::ir::{
    Dpp8Control, DppControl, InstructionControl, Opcode, Operand, RawOpcodeFamily, SdwaControl,
    ShaderEncoding,
};

use super::{DecodeError, Frame, frame_or_raw};

pub(crate) fn vector_extension_dwords(source: u32) -> usize {
    usize::from(matches!(source, 0xe9 | 0xea | 0xf9 | 0xfa | 0xff))
}

pub(crate) fn decode_vop1(_pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 9) & 0xff;
    let name = match code {
        0x00 => Some("VNop"),
        0x01 => Some("VMovB32"),
        0x02 => Some("VReadfirstlaneB32"),
        0x03 => Some("VCvtI32F64"),
        0x04 => Some("VCvtF64I32"),
        0x05 => Some("VCvtF32I32"),
        0x06 => Some("VCvtF32U32"),
        0x07 => Some("VCvtU32F32"),
        0x08 => Some("VCvtI32F32"),
        0x0a => Some("VCvtF16F32"),
        0x0b => Some("VCvtF32F16"),
        0x0c => Some("VCvtRpiI32F32"),
        0x0d => Some("VCvtFlrI32F32"),
        0x0e => Some("VCvtOffF32I4"),
        0x0f => Some("VCvtF32F64"),
        0x10 => Some("VCvtF64F32"),
        0x11 => Some("VCvtF32Ubyte0"),
        0x12 => Some("VCvtF32Ubyte1"),
        0x13 => Some("VCvtF32Ubyte2"),
        0x14 => Some("VCvtF32Ubyte3"),
        0x15 => Some("VCvtU32F64"),
        0x16 => Some("VCvtF64U32"),
        0x17 => Some("VTruncF64"),
        0x18 => Some("VCeilF64"),
        0x19 => Some("VRndneF64"),
        0x1a => Some("VFloorF64"),
        0x1b => Some("VPipeflush"),
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
        0x2f => Some("VRcpF64"),
        0x31 => Some("VRsqF64"),
        0x33 => Some("VSqrtF32"),
        0x34 => Some("VSqrtF64"),
        0x35 => Some("VSinF32"),
        0x36 => Some("VCosF32"),
        0x37 => Some("VNotB32"),
        0x38 => Some("VBfrevB32"),
        0x39 => Some("VFfbhU32"),
        0x3a => Some("VFfblB32"),
        0x3b => Some("VFfbhI32"),
        0x3c => Some("VFrexpExpI32F64"),
        0x3d => Some("VFrexpMantF64"),
        0x3e => Some("VFractF64"),
        0x3f => Some("VFrexpExpI32F32"),
        0x40 => Some("VFrexpMantF32"),
        0x42 => Some("VMovreldB32"),
        0x43 => Some("VMovrelsB32"),
        0x44 => Some("VMovrelsdB32"),
        0x48 => Some("VMovrelsd2B32"),
        0x50 => Some("VCvtF16U16"),
        0x51 => Some("VCvtF16I16"),
        0x52 => Some("VCvtU16F16"),
        0x53 => Some("VCvtI16F16"),
        0x54 => Some("VRcpF16"),
        0x55 => Some("VSqrtF16"),
        0x56 => Some("VRsqF16"),
        0x57 => Some("VLogF16"),
        0x58 => Some("VExpF16"),
        0x59 => Some("VFrexpMantF16"),
        0x5a => Some("VFrexpExpI16F16"),
        0x5b => Some("VFloorF16"),
        0x5c => Some("VCeilF16"),
        0x5d => Some("VTruncF16"),
        0x5e => Some("VRndneF16"),
        0x5f => Some("VFractF16"),
        0x60 => Some("VSinF16"),
        0x61 => Some("VCosF16"),
        0x62 => Some("VSatPkU8I16"),
        0x63 => Some("VCvtNormI16F16"),
        0x64 => Some("VCvtNormU16F16"),
        0x65 => Some("VSwapB32"),
        0x68 => Some("VSwaprelB32"),
        _ => None,
    };
    Ok(frame_or_raw(
        ShaderEncoding::Vop1,
        RawOpcodeFamily::Vop1,
        code,
        name,
        1 + vector_extension_dwords(word & 0x1ff),
    ))
}

/// VOP1 operations whose result is a 64-bit value written to a VGPR pair.
pub(crate) fn vop1_writes_vector_pair(opcode: Opcode) -> bool {
    matches!(
        opcode.known_name(),
        Some(
            "VCvtF64I32"
                | "VCvtF64F32"
                | "VCvtF64U32"
                | "VTruncF64"
                | "VCeilF64"
                | "VRndneF64"
                | "VFloorF64"
                | "VRcpF64"
                | "VRsqF64"
                | "VSqrtF64"
                | "VFrexpMantF64"
                | "VFractF64"
        )
    )
}

pub(crate) fn decode_vop2(_pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 25) & 0x3f;
    let name = match code {
        0x01 => Some("VCndmaskB32"),
        0x02 => Some("VDot2cF32F16"),
        0x03 => Some("VAddF32"),
        0x04 => Some("VSubF32"),
        0x05 => Some("VSubrevF32"),
        0x06 => Some("VFmacLegacyF32"),
        0x07 => Some("VMulLegacyF32"),
        0x08 => Some("VMulF32"),
        0x09 => Some("VMulI32I24"),
        0x0a => Some("VMulHiI32I24"),
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
        0x3c => Some("VPkFmacF16"),
        _ => None,
    };
    let forced_literal = matches!(code, 0x20 | 0x21 | 0x2c | 0x2d | 0x37 | 0x38);
    Ok(frame_or_raw(
        ShaderEncoding::Vop2,
        RawOpcodeFamily::Vop2,
        code,
        name,
        1 + usize::from(forced_literal).max(vector_extension_dwords(word & 0x1ff)),
    ))
}

#[allow(clippy::too_many_lines)]
pub(crate) fn decode_vopc(_pc: u32, word: u32) -> Result<Frame, DecodeError> {
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
        0x20 => Some("VCmpFF64"),
        0x21 => Some("VCmpLtF64"),
        0x22 => Some("VCmpEqF64"),
        0x23 => Some("VCmpLeF64"),
        0x24 => Some("VCmpGtF64"),
        0x25 => Some("VCmpLgF64"),
        0x26 => Some("VCmpGeF64"),
        0x27 => Some("VCmpOF64"),
        0x28 => Some("VCmpUF64"),
        0x29 => Some("VCmpNgeF64"),
        0x2a => Some("VCmpNlgF64"),
        0x2b => Some("VCmpNgtF64"),
        0x2c => Some("VCmpNleF64"),
        0x2d => Some("VCmpNeqF64"),
        0x2e => Some("VCmpNltF64"),
        0x2f => Some("VCmpTruF64"),
        0x30 => Some("VCmpxFF64"),
        0x31 => Some("VCmpxLtF64"),
        0x32 => Some("VCmpxEqF64"),
        0x33 => Some("VCmpxLeF64"),
        0x34 => Some("VCmpxGtF64"),
        0x35 => Some("VCmpxLgF64"),
        0x36 => Some("VCmpxGeF64"),
        0x37 => Some("VCmpxOF64"),
        0x38 => Some("VCmpxUF64"),
        0x39 => Some("VCmpxNgeF64"),
        0x3a => Some("VCmpxNlgF64"),
        0x3b => Some("VCmpxNgtF64"),
        0x3c => Some("VCmpxNleF64"),
        0x3d => Some("VCmpxNeqF64"),
        0x3e => Some("VCmpxNltF64"),
        0x3f => Some("VCmpxTruF64"),
        0x80 => Some("VCmpFI32"),
        0x81 => Some("VCmpLtI32"),
        0x82 => Some("VCmpEqI32"),
        0x83 => Some("VCmpLeI32"),
        0x84 => Some("VCmpGtI32"),
        0x85 => Some("VCmpNeI32"),
        0x86 => Some("VCmpGeI32"),
        0x87 => Some("VCmpTI32"),
        0x88 => Some("VCmpClassF32"),
        0x89 => Some("VCmpLtI16"),
        0x8a => Some("VCmpEqI16"),
        0x8b => Some("VCmpLeI16"),
        0x8c => Some("VCmpGtI16"),
        0x8d => Some("VCmpNeI16"),
        0x8e => Some("VCmpGeI16"),
        0x8f => Some("VCmpClassF16"),
        0x90 => Some("VCmpxFI32"),
        0x91 => Some("VCmpxLtI32"),
        0x92 => Some("VCmpxEqI32"),
        0x93 => Some("VCmpxLeI32"),
        0x94 => Some("VCmpxGtI32"),
        0x95 => Some("VCmpxNeI32"),
        0x96 => Some("VCmpxGeI32"),
        0x97 => Some("VCmpxTI32"),
        0x98 => Some("VCmpxClassF32"),
        0x99 => Some("VCmpxLtI16"),
        0x9a => Some("VCmpxEqI16"),
        0x9b => Some("VCmpxLeI16"),
        0x9c => Some("VCmpxGtI16"),
        0x9d => Some("VCmpxNeI16"),
        0x9e => Some("VCmpxGeI16"),
        0x9f => Some("VCmpxClassF16"),
        0xa8 => Some("VCmpClassF64"),
        0xa9 => Some("VCmpLtU16"),
        0xaa => Some("VCmpEqU16"),
        0xab => Some("VCmpLeU16"),
        0xac => Some("VCmpGtU16"),
        0xad => Some("VCmpNeU16"),
        0xae => Some("VCmpGeU16"),
        0xb8 => Some("VCmpxClassF64"),
        0xb9 => Some("VCmpxLtU16"),
        0xba => Some("VCmpxEqU16"),
        0xbb => Some("VCmpxLeU16"),
        0xbc => Some("VCmpxGtU16"),
        0xbd => Some("VCmpxNeU16"),
        0xbe => Some("VCmpxGeU16"),
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
    Ok(frame_or_raw(
        ShaderEncoding::Vopc,
        RawOpcodeFamily::Vopc,
        code,
        name,
        1 + vector_extension_dwords(word & 0x1ff),
    ))
}

pub(crate) fn decode_vop3(word: u32, extra: u32) -> Frame {
    let code = (word >> 16) & 0x3ff;
    let is_b = is_vop3b_opcode(code);
    let name = if is_b {
        match code {
            0x128 => Some("VAddCoCiU32"),
            0x30f => Some("VAddCoU32"),
            0x310 => Some("VSubCoU32"),
            0x319 => Some("VSubrevCoU32"),
            0x16d => Some("VDivScaleF32"),
            0x16e => Some("VDivScaleF64"),
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
            // GFX10 keeps the precise-division sequence contiguous with the multiplies at
            // 0x15F and 0x16D..0x170; the 0x340 OPSEL block that starts at V_FMA_F16 holds the
            // half-float three-operand minima instead.
            0x15f => Some("VDivFixupF32"),
            0x169 => Some("VMulLoU32"),
            0x16a => Some("VMulHiU32"),
            0x16b => Some("VMulLoI32"),
            0x16c => Some("VMulHiI32"),
            0x16f => Some("VDivFmasF32"),
            0x170 => Some("VDivFmasF64"),
            0x178 => Some("VXor3B32"),
            // VOP1 opcodes use their architectural opcode plus 0x180 in the VOP3/e64
            // encoding. V_CVT_FLR_I32_F32 is opcode 0x0d, hence 0x18d here.
            0x18d => Some("VCvtFlrI32F32"),
            0x2ff => Some("VLshlrevB64"),
            0x300 => Some("VLshrrevB64"),
            0x345 => Some("VXadU32"),
            0x346 => Some("VLshlAddU32"),
            0x347 => Some("VAddLshlU32"),
            0x34b => Some("VFmaF16"),
            0x351 => Some("VMin3F16"),
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

pub(crate) const fn is_vop3b_opcode(code: u32) -> bool {
    matches!(
        code,
        0x128 | 0x16d | 0x16e | 0x176 | 0x177 | 0x30f | 0x310 | 0x319
    )
}

pub(crate) fn decode_vop3p(word: u32, extra: u32) -> Frame {
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

pub(crate) fn decode_vintrp(_pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 16) & 3;
    let name = match code {
        0 => Some("VInterpP1F32"),
        1 => Some("VInterpP2F32"),
        2 => Some("VInterpMovF32"),
        _ => None,
    };
    Ok(frame_or_raw(
        ShaderEncoding::Vintrp,
        RawOpcodeFamily::Vintrp,
        code,
        name,
        1,
    ))
}

#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
pub(crate) fn decode_vop_extension(
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

pub(crate) const fn create_dpp_control(word: u32) -> DppControl {
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

pub(crate) const fn create_sdwa_control(
    word: u32,
    is_compare: bool,
    has_source1: bool,
) -> SdwaControl {
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
