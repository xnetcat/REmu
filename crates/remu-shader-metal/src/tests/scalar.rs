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
fn unsigned_sopk_compare_zero_extends_its_immediate() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0x34,
            encoding: ShaderEncoding::Sopk,
            opcode: Opcode::Known("SCmpkLeU32"),
            words: vec![0xb704_ffff],
            sources: vec![Operand {
                kind: OperandKind::EncodedConstant,
                value: u32::from(u16::MAX),
            }],
            destinations: vec![Operand::scalar(4)],
            control: None,
        },
        MetalShaderStage::Compute,
    );

    assert!(shader.source.contains("(remu_s_4) <= (0xFFFFu)"));
    assert!(!shader.source.contains("0xFFFFFFFFu"));
}

#[test]
fn s_addk_i32_lowers_its_signed_overflow_condition() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0x38,
            encoding: ShaderEncoding::Sopk,
            opcode: Opcode::Known("SAddkI32"),
            words: vec![0xb784_0001],
            sources: vec![Operand {
                kind: OperandKind::EncodedConstant,
                value: 1,
            }],
            destinations: vec![Operand::scalar(4)],
            control: None,
        },
        MetalShaderStage::Compute,
    );

    assert!(shader.source.contains("0x80000000u"));
    assert!(shader.source.contains(" ^ "));
    assert_eq!(
        shader.source.matches("scc = ").count(),
        2,
        "S_ADDK_I32 must write SCC in addition to its initialized value:\n{}",
        shader.source
    );
}

#[test]
fn hardware_register_writes_do_not_fabricate_an_sgpr_destination() {
    for (name, sources, words) in [
        (
            "SSetregB32",
            vec![
                Operand {
                    kind: OperandKind::EncodedConstant,
                    value: 0xf801,
                },
                Operand::scalar(5),
            ],
            vec![0xb985_f801],
        ),
        (
            "SSetregImm32B32",
            vec![
                Operand {
                    kind: OperandKind::EncodedConstant,
                    value: 0xf801,
                },
                Operand {
                    kind: OperandKind::LiteralConstant,
                    value: 0x1122_3344,
                },
            ],
            vec![0xba85_f801, 0x1122_3344],
        ),
    ] {
        let shader = compile_vector_instruction(
            Instruction {
                pc: 0x24,
                encoding: ShaderEncoding::Sopk,
                opcode: Opcode::Known(name),
                words,
                sources,
                destinations: Vec::new(),
                control: None,
            },
            MetalShaderStage::Compute,
        );
        assert!(shader.source.contains("kernel void remu_main"));
        assert!(
            !shader.source.contains("remu_s_5 ="),
            "{name} reads hardware-state operands but writes no SGPR"
        );
    }
}

#[test]
fn wave_mode_and_scheduling_hints_lower_to_nothing_but_reading_one_refuses() {
    for name in [
        "SRoundMode",
        "SDenormMode",
        "SSetprio",
        "SIcacheInv",
        "SSleep",
        "SWakeup",
        "SSethalt",
        "SIncperflevel",
        "SDecperflevel",
        "STtracedataImm",
        "SWaitIdle",
        "SWaitcntVscnt",
        "SWaitcntVmcnt",
        "SWaitcntExpcnt",
        "SWaitcntLgkmcnt",
    ] {
        let shader = compile_vector_instruction(
            Instruction {
                pc: 0x30,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::Known(name),
                words: vec![0],
                sources: vec![Operand {
                    kind: OperandKind::EncodedConstant,
                    value: 0,
                }],
                destinations: Vec::new(),
                control: None,
            },
            MetalShaderStage::Compute,
        );
        assert!(
            shader.source.contains("kernel void remu_main"),
            "{name} adjusts wave state Metal fixes for us and must not fail the translation"
        );
    }

    // Reading a hardware register is different in kind: the value is unmodeled state, and a
    // shader that goes on to compute an address from it would read the wrong memory.
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0x1000,
            encoding: ShaderEncoding::Sopk,
            opcode: Opcode::Known("SGetregB32"),
            words: vec![0xb905_f801],
            sources: vec![Operand {
                kind: OperandKind::EncodedConstant,
                value: 0xf801,
            }],
            destinations: vec![Operand::scalar(5)],
            control: None,
        }],
    );
    let error = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(0),
    )
    .expect_err("reading an unmodeled hardware register must refuse");
    assert!(
        error.to_string().contains("SGetregB32"),
        "the refusal must name the hardware register read: {error}"
    );
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
    assert!(
        saveexec
            .source
            .contains("exec ? 0xfffffffffffffffful : 0ul"),
        "fragment saveexec cannot use simd_ballot; a zero ballot restores EXEC empty and discards the quad"
    );
    assert!(!saveexec.source.contains("simd_ballot(exec)"));
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
        saveexec32.source.contains("(exec) ? 0xffffffffu : 0u"),
        "wave32 fragment saveexec must publish this thread's EXEC in every local SIMD32 bit"
    );
    assert!(!saveexec32.source.contains("simd_ballot(exec)"));
    assert!(saveexec32.source.contains("s[20] = ") || saveexec32.source.contains("remu_s_20 = "));
    assert!(saveexec32.source.contains("s[126]") || saveexec32.source.contains("remu_s_126"));
    assert!(
        saveexec32.source.contains("s[127] = 0u") || saveexec32.source.contains("remu_s_127 = 0u")
    );
    assert!(saveexec32.source.contains("exec = "));
}

#[test]
fn s_brev_b32_leaves_the_scalar_condition_alone() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0x40,
            encoding: ShaderEncoding::Sop1,
            opcode: Opcode::Known("SBrevB32"),
            words: vec![0],
            sources: vec![Operand::scalar(10)],
            destinations: vec![Operand::scalar(12)],
            control: None,
        },
        MetalShaderStage::Compute,
    );

    assert!(shader.source.contains("reverse_bits(remu_s_10)"));
    assert!(shader.source.contains("remu_s_12 = "));
    assert_eq!(
        shader.source.matches("scc = ").count(),
        1,
        "S_BREV_B32 must preserve the initialized SCC value:\n{}",
        shader.source
    );
}

#[test]
fn lowers_unary_first_set_and_destination_bit_update_without_touching_scc() {
    let first_set = compile_vector_instruction(
        Instruction {
            pc: 0x44,
            encoding: ShaderEncoding::Sop1,
            opcode: Opcode::Known("SFF1I32B32"),
            words: vec![0],
            sources: vec![Operand::scalar(10)],
            destinations: vec![Operand::scalar(12)],
            control: None,
        },
        MetalShaderStage::Compute,
    );
    assert!(first_set.source.contains("remu_s_10 == 0u ? 0xFFFFFFFFu"));
    assert!(first_set.source.contains("ctz(remu_s_10)"));
    assert!(first_set.source.contains("remu_s_12 = "));
    assert_eq!(first_set.source.matches("scc = ").count(), 1);

    for (name, operation) in [
        ("SBitset0B32", "& ~(1u << (remu_s_10 & 31u))"),
        ("SBitset1B32", "| (1u << (remu_s_10 & 31u))"),
    ] {
        let shader = compile_vector_instruction(
            Instruction {
                pc: 0x48,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::Known(name),
                words: vec![0],
                sources: vec![Operand::scalar(10)],
                destinations: vec![Operand::scalar(12)],
                control: None,
            },
            MetalShaderStage::Compute,
        );
        assert!(shader.source.contains(&format!("remu_s_12 {operation}")));
        assert_eq!(shader.source.matches("scc = ").count(), 1, "{name}");
    }
}

#[test]
fn lowers_shifted_scalar_add_with_its_unsigned_carry_out_condition() {
    for (name, shift) in [
        ("SLshl1AddU32", 1),
        ("SLshl2AddU32", 2),
        ("SLshl3AddU32", 3),
        ("SLshl4AddU32", 4),
    ] {
        let shader = compile_vector_instruction(
            Instruction {
                pc: 0x44,
                encoding: ShaderEncoding::Sop2,
                opcode: Opcode::Known(name),
                words: vec![0],
                sources: vec![Operand::scalar(10), Operand::scalar(12)],
                destinations: vec![Operand::scalar(14)],
                control: None,
            },
            MetalShaderStage::Compute,
        );

        assert!(
            shader
                .source
                .contains(&format!("((ulong)remu_s_10 << {shift}) + (ulong)remu_s_12")),
            "{name} must form the sum at 64-bit width:\n{}",
            shader.source
        );
        assert!(shader.source.contains("remu_s_14 = "));
        assert!(
            shader.source.contains(">> 32) != 0ul;"),
            "{name} defines SCC as the unsigned carry-out:\n{}",
            shader.source
        );
    }
}

#[test]
fn fixed_ngg_compatibility_preserves_legacy_shifted_scalar_add_dataflow() {
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0x44,
            encoding: ShaderEncoding::Sop2,
            opcode: Opcode::Known("SLshl4AddU32"),
            words: vec![0],
            sources: vec![Operand::scalar(10), Operand::scalar(12)],
            destinations: vec![Operand::scalar(14)],
            control: None,
        }],
    );
    let mut options = MetalCompileOptions::mesh(0, 16 * 1024, 64, 64, 64);
    options.ngg_fixed_allocation_fallback = true;
    options.wave_lane_count = 64;
    let shader = compile_msl(&program, &empty_evaluation(), &options)
        .expect("fixed-NGG shifted scalar add should lower");

    assert!(shader.source.contains("((remu_s_10 << 4) + remu_s_12)"));
    assert!(!shader.source.contains("((ulong)remu_s_10 << 4)"));
    assert!(!shader.source.contains(">> 32) != 0ul"));
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
