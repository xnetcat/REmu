// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Scalar encoding families: SOP1/SOP2/SOPC/SOPK/SOPP and the scalar memory encodings.

use crate::ir::{Opcode, RawOpcodeFamily, ShaderEncoding};

use super::{DecodeError, Frame, frame_or_raw};

pub(crate) fn decode_sop1(_pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 8) & 0xff;
    let name = match code {
        0x03 => Some("SMovB32"),
        0x04 => Some("SMovB64"),
        0x05 => Some("SCmovB32"),
        0x06 => Some("SCmovB64"),
        0x07 => Some("SNotB32"),
        0x08 => Some("SNotB64"),
        0x09 => Some("SWqmB32"),
        0x0a => Some("SWqmB64"),
        0x0b => Some("SBrevB32"),
        0x0c => Some("SBrevB64"),
        0x0d => Some("SBcnt0I32B32"),
        0x0e => Some("SBcnt0I32B64"),
        0x0f => Some("SBcnt1I32B32"),
        0x10 => Some("SBcnt1I32B64"),
        0x11 => Some("SFF0I32B32"),
        0x12 => Some("SFF0I32B64"),
        0x13 => Some("SFF1I32B32"),
        0x14 => Some("SFF1I32B64"),
        0x15 => Some("SFlbitI32B32"),
        0x16 => Some("SFlbitI32B64"),
        0x17 => Some("SFlbitI32"),
        0x18 => Some("SFlbitI32I64"),
        0x19 => Some("SSextI32I8"),
        0x1a => Some("SSextI32I16"),
        0x1b => Some("SBitset0B32"),
        0x1c => Some("SBitset0B64"),
        0x1d => Some("SBitset1B32"),
        0x1e => Some("SBitset1B64"),
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
        0x2c => Some("SQuadmaskB32"),
        0x2d => Some("SQuadmaskB64"),
        0x2e => Some("SMovrelsB32"),
        0x2f => Some("SMovrelsB64"),
        0x30 => Some("SMovreldB32"),
        0x31 => Some("SMovreldB64"),
        0x34 => Some("SAbsI32"),
        0x36 => Some("SBitreplicateB64B32"),
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
    Ok(frame_or_raw(
        ShaderEncoding::Sop1,
        RawOpcodeFamily::Sop1,
        code,
        name,
        1 + usize::from(word & 0xff == 0xff),
    ))
}

pub(crate) fn decode_sop2(_pc: u32, word: u32) -> Result<Frame, DecodeError> {
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
    Ok(frame_or_raw(
        ShaderEncoding::Sop2,
        RawOpcodeFamily::Sop2,
        code,
        name,
        1 + usize::from(literal),
    ))
}

pub(crate) fn decode_sopc(_pc: u32, word: u32) -> Result<Frame, DecodeError> {
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
    Ok(frame_or_raw(
        ShaderEncoding::Sopc,
        RawOpcodeFamily::Sopc,
        code,
        name,
        1 + usize::from(literal),
    ))
}

pub(crate) fn decode_sopp(_pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 16) & 0x7f;
    let name = match code {
        0x00 => Some("SNop"),
        0x01 => Some("SEndpgm"),
        0x02 => Some("SBranch"),
        0x03 => Some("SWakeup"),
        0x04 => Some("SCbranchScc0"),
        0x05 => Some("SCbranchScc1"),
        0x06 => Some("SCbranchVccz"),
        0x07 => Some("SCbranchVccnz"),
        0x08 => Some("SCbranchExecz"),
        0x09 => Some("SCbranchExecnz"),
        0x0a => Some("SBarrier"),
        0x0c => Some("SWaitcnt"),
        0x0d => Some("SSethalt"),
        0x0e => Some("SSleep"),
        0x0f => Some("SSetprio"),
        0x10 => Some("SSendmsg"),
        0x11 => Some("SSendmsghalt"),
        0x13 => Some("SIcacheInv"),
        0x14 => Some("SIncperflevel"),
        0x15 => Some("SDecperflevel"),
        0x16 => Some("STtraceData"),
        // Mode-control ops. They set rounding/denormal/priority state that resource discovery
        // does not model, so decoding them keeps a shader that merely toggled a mode alive. SDK
        // cross-check: s_round_mode / s_denorm_mode / s_sleep / s_ttracedata_imm mnemonics in
        // libSceShaderIsaP.dll; OP numbers continue the RDNA2 (GFX10) SOPP block already decoded
        // above (0x20-0x23).
        0x22 => Some("SWaitIdle"),
        0x24 => Some("SRoundMode"),
        0x25 => Some("SDenormMode"),
        0x28 => Some("STtracedataImm"),
        // S_ENDPGM_SAVED (0x1b) and S_ENDPGM_ORDERED_PS_DONE (0x1e) end the wave exactly like
        // S_ENDPGM (both mnemonics are in libSceShaderIsaP.dll); RDNA2 additionally uses
        // S_CODE_END to pad bytes beyond the executable shader. Treating all three as the
        // bounded stream terminator keeps the decoder from interpreting padding as instructions
        // while preserving the architectural wave end.
        0x1b | 0x1e | 0x1f => {
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
    // Every SOPP control transfer is in the named table (including the debug-conditional
    // branches, which production compilers never emit); a reserved code is a hint-space
    // operation whose raw fallthrough matches hardware behavior without a debugger attached.
    Ok(frame_or_raw(
        ShaderEncoding::Sopp,
        RawOpcodeFamily::Sopp,
        code,
        name,
        1,
    ))
}

pub(crate) fn decode_sopk(_pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = ((word >> 23) & 0x7f) - 0x60;
    let name = match code {
        0x00 => Some("SMovkI32"),
        0x01 => Some("SVersion"),
        0x02 => Some("SCmovkI32"),
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
        // Hardware-register access. S_GETREG reads a mode/status register into an SGPR;
        // S_SETREG(_IMM32) writes one. Resource discovery treats the mode registers as opaque, so
        // decoding these keeps a prologue that configures rounding or reads HW_ID alive. SDK
        // cross-check: s_getreg_b32 / s_setreg_b32 / s_setreg_imm32_b32 in libSceShaderIsaP.dll;
        // OP numbers follow the RDNA2 (GFX10) SOPK block. S_SETREG_IMM32_B32 carries a trailing
        // 32-bit literal, so it frames as two dwords.
        0x11 => Some("SGetregB32"),
        0x12 => Some("SSetregB32"),
        0x13 => Some("SSetregImm32B32"),
        // S_CALL_B64 (20) links the return address and takes a static relative target; the
        // scalar wait counts (21-24) only park the wave. RDNA2 SOPK numbering continues the
        // block above; every mnemonic is present in libSceShaderIsaP.dll.
        0x14 => Some("SCallB64"),
        0x15 => Some("SWaitcntVscnt"),
        0x16 => Some("SWaitcntVmcnt"),
        0x17 => Some("SWaitcntExpcnt"),
        0x18 => Some("SWaitcntLgkmcnt"),
        _ => None,
    };
    // S_SETREG_IMM32_B32 is the one SOPK instruction that carries a 32-bit literal.
    Ok(frame_or_raw(
        ShaderEncoding::Sopk,
        RawOpcodeFamily::Sopk,
        code,
        name,
        1 + usize::from(code == 0x13),
    ))
}

pub(crate) fn decode_smrd(_pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 22) & 0x1f;
    let name = scalar_load_name(code);
    let literal = word >> 8 & 1 == 0 && word & 0xff == 0xff;
    Ok(frame_or_raw(
        ShaderEncoding::Smrd,
        RawOpcodeFamily::Smrd,
        code,
        name,
        1 + usize::from(literal),
    ))
}

pub(crate) fn decode_smem(_pc: u32, word: u32) -> Result<Frame, DecodeError> {
    let code = (word >> 18) & 0xff;
    Ok(frame_or_raw(
        ShaderEncoding::Smem,
        RawOpcodeFamily::Smem,
        code,
        scalar_load_name(code),
        2,
    ))
}

pub(crate) const fn scalar_load_name(code: u32) -> Option<&'static str> {
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

pub(crate) fn scalar_load_dword_count(opcode: Opcode) -> u32 {
    match opcode.known_name() {
        Some("SLoadDword" | "SBufferLoadDword") => 1,
        Some("SLoadDwordx2" | "SBufferLoadDwordx2") => 2,
        Some("SLoadDwordx4" | "SBufferLoadDwordx4") => 4,
        Some("SLoadDwordx8" | "SBufferLoadDwordx8") => 8,
        Some("SLoadDwordx16" | "SBufferLoadDwordx16") => 16,
        _ => 0,
    }
}
