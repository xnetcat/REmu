// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Scalar ALU lowering tests: SOP1/SOP2/SOPC bit operations, 64-bit scalar
//! arithmetic and the SCC side effect.

use super::*;

#[test]
fn lowers_observed_64_bit_and_not_scalar_operation() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0x30,
            encoding: ShaderEncoding::Sop2,
            opcode: Opcode::Known("SAndn2B64"),
            words: vec![0],
            sources: vec![Operand::scalar(10), Operand::scalar(12)],
            destinations: vec![Operand::scalar(20)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );

    assert!(shader.source.contains("& ~"));
    assert!(shader.source.contains("s[20]"));
    assert!(shader.source.contains("s[21]"));
}

#[test]
fn lowers_s_bcnt1_i32_b64_to_one_uint_destination_and_scc() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Sop1,
            opcode: Opcode::Known("SBcnt1I32B64"),
            words: vec![0xbe8a_1000],
            sources: vec![Operand::scalar(0)],
            destinations: vec![Operand::scalar(10)],
            control: None,
        },
        MetalShaderStage::Compute,
    );

    assert!(
        shader
            .source
            .contains("((ulong)remu_s_0 | ((ulong)remu_s_1 << 32))")
    );
    assert!(shader.source.contains("popcount((uint)"));
    assert!(shader.source.contains("popcount((uint)("));
    assert!(shader.source.contains(" >> 32))"));
    assert!(shader.source.contains("remu_s_10 = "));
    assert!(shader.source.contains("scc = "));
    assert!(shader.source.contains(" != 0u;"));
    assert!(
        !shader.source.contains("remu_s_11 = "),
        "the 32-bit destination must not be lowered as a B64 register pair"
    );
}

#[test]
fn lowers_s_bitcmp1_b32_as_an_indexed_bit_test() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Sopc,
            opcode: Opcode::Known("SBitcmp1B32"),
            words: vec![0],
            sources: vec![Operand::scalar(10), Operand::scalar(11)],
            destinations: Vec::new(),
            control: None,
        },
        MetalShaderStage::Compute,
    );

    assert!(shader.source.contains("remu_s_10"));
    assert!(shader.source.contains("remu_s_11"));
    assert!(shader.source.contains("& 31u"));
    assert!(shader.source.contains(">>"));
    assert!(shader.source.contains("& 1u"));
    assert!(shader.source.contains("scc = "));
}

#[test]
fn lowers_scalar_absdiff_with_wrapping_i32_arithmetic_and_scc() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Sop2,
            opcode: Opcode::Known("SAbsdiffI32"),
            words: vec![0],
            sources: vec![Operand::scalar(10), Operand::scalar(11)],
            destinations: vec![Operand::scalar(12)],
            control: None,
        },
        MetalShaderStage::Compute,
    );

    assert!(shader.source.contains("remu_s_10 - remu_s_11"));
    assert!(shader.source.contains("as_type<int>("));
    assert!(shader.source.contains("? 0u - "));
    assert!(shader.source.contains("scc = "));
    assert!(shader.source.contains(" != 0u;"));
}

#[test]
fn lowers_observed_bit_reverse_and_wave32_saveexec_operations() {
    let reverse = compile_vector_instruction(
        Instruction {
            pc: 0x194,
            encoding: ShaderEncoding::Vop1,
            opcode: Opcode::Known("VBfrevB32"),
            words: vec![0],
            sources: vec![Operand::vector(2)],
            destinations: vec![Operand::vector(8)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );
    assert!(reverse.source.contains("reverse_bits(remu_v_2)"));

    let saveexec = compile_vector_instruction(
        Instruction {
            pc: 0x74,
            encoding: ShaderEncoding::Sop1,
            opcode: Opcode::Known("SAndSaveexecB64"),
            words: vec![0],
            sources: vec![Operand::scalar(10)],
            destinations: vec![Operand::scalar(20)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );
    assert!(saveexec.source.contains("simd_ballot(exec)"));
    assert!(saveexec.source.contains("& 0xFFFFFFFFul"));
    assert!(saveexec.source.contains("remu_s_20 = (uint)"));
    assert!(saveexec.source.contains("exec = (("));

    let saveexec32 = compile_vector_instruction(
        Instruction {
            pc: 0xb0c,
            encoding: ShaderEncoding::Sop1,
            opcode: Opcode::Known("SAndSaveexecB32"),
            words: vec![0],
            sources: vec![Operand::scalar(10)],
            destinations: vec![Operand::scalar(20)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );
    assert!(
        saveexec32
            .source
            .contains("(uint)(uint64_t)simd_ballot(exec)")
    );
    assert!(saveexec32.source.contains("s[20] = ") || saveexec32.source.contains("remu_s_20 = "));
    assert!(saveexec32.source.contains("s[126]") || saveexec32.source.contains("remu_s_126"));
    assert!(
        saveexec32.source.contains("s[127] = 0u") || saveexec32.source.contains("remu_s_127 = 0u")
    );
    assert!(saveexec32.source.contains("exec = "));
}

#[test]
fn lowers_unsigned_64_bit_scalar_compare_from_register_pairs() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0x1c0,
            encoding: ShaderEncoding::Sopc,
            opcode: Opcode::Known("SCmpLgU64"),
            words: vec![0],
            sources: vec![Operand::scalar(106), Operand::scalar(30)],
            destinations: Vec::new(),
            control: None,
        },
        MetalShaderStage::Pixel,
    );

    assert!(
        shader
            .source
            .contains("((ulong)remu_s_106 | ((ulong)remu_s_107 << 32))")
    );
    assert!(
        shader
            .source
            .contains("((ulong)remu_s_30 | ((ulong)remu_s_31 << 32))")
    );
    assert!(shader.source.contains("!="));
}

#[test]
fn lowers_scalar_bitfield_mask_operation() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0x220,
            encoding: ShaderEncoding::Sop2,
            opcode: Opcode::Known("SBfmB32"),
            words: vec![0],
            sources: vec![Operand::scalar(10), Operand::scalar(12)],
            destinations: vec![Operand::scalar(20)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );

    assert!(shader.source.contains("1u << ("));
    assert!(shader.source.contains("& 31u)) - 1u) << ("));
    assert!(shader.source.contains("remu_s_20"));
}

#[test]
fn lowers_s_bfm_b64_from_two_u32_controls_without_modifying_scc() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0x224,
            encoding: ShaderEncoding::Sop2,
            opcode: Opcode::Known("SBfmB64"),
            words: vec![0x9294_0c0a],
            sources: vec![Operand::scalar(10), Operand::scalar(12)],
            destinations: vec![Operand::scalar(20)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );

    assert!(shader.source.contains("1ul << (remu_s_10 & 63u)"));
    assert!(shader.source.contains("<< (remu_s_12 & 63u)"));
    assert!(shader.source.contains("remu_s_20 = (uint)"));
    assert!(shader.source.contains("remu_s_21 = (uint)"));
    assert!(!shader.source.contains("remu_s_11"));
    assert!(!shader.source.contains("remu_s_13"));
    assert_eq!(
        shader.source.matches("scc = ").count(),
        1,
        "S_BFM_B64 must preserve the initialized SCC value"
    );
}

#[test]
fn lowers_s_pack_ll_b32_b16_from_both_low_halves_without_modifying_scc() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0x228,
            encoding: ShaderEncoding::Sop2,
            opcode: Opcode::Known("SPackLlB32B16"),
            words: vec![0x9914_0c0a],
            sources: vec![Operand::scalar(10), Operand::scalar(12)],
            destinations: vec![Operand::scalar(20)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );

    assert!(shader.source.contains("(remu_s_10 & 0xFFFFu)"));
    assert!(shader.source.contains("((remu_s_12 & 0xFFFFu) << 16)"));
    assert!(shader.source.contains("remu_s_20 = "));
    assert_eq!(
        shader.source.matches("scc = ").count(),
        1,
        "S_PACK_LL_B32_B16 must preserve the initialized SCC value"
    );
}
