// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Scalar ALU evaluation tests: SOP1/SOP2/SOPC results, the SCC side effect and
//! the initial merged-stage register state.

use super::*;

#[test]
fn initializes_merged_export_wave_info_before_user_sgprs() {
    let shader = ShaderState {
        program: Arc::new(Program::new(
            0x1000,
            vec![
                instruction(
                    0,
                    ShaderEncoding::Sop1,
                    "SMovB32",
                    vec![Operand::scalar(3)],
                    vec![Operand::scalar(4)],
                    None,
                ),
                instruction(
                    4,
                    ShaderEncoding::Sop1,
                    "SMovB32",
                    vec![Operand::scalar(8)],
                    vec![Operand::scalar(5)],
                    None,
                ),
                end(8),
            ],
        )),
        user_data: vec![0x1234_5678].into(),
        compute_system_registers: None,
        user_data_scalar_register_base: 8,
    };

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("merged export evaluation");

    assert_eq!(evaluation.initial_scalar_registers[3], 0x4040);
    assert_eq!(evaluation.scalar_registers[4], 0x4040);
    assert_eq!(evaluation.initial_scalar_registers[8], 0x1234_5678);
    assert_eq!(evaluation.scalar_registers[5], 0x1234_5678);
}

#[test]
fn s_bcnt1_i32_b64_counts_the_source_pair_and_updates_scc() {
    let instruction = instruction(
        0,
        ShaderEncoding::Sop1,
        "SBcnt1I32B64",
        vec![Operand::scalar(0)],
        vec![Operand::scalar(10)],
        None,
    );
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    let mut exec_mask = u64::MAX;
    let mut scalar_condition = false;

    registers[0] = 0xf0f0_0001;
    registers[1] = 0x8000_ffff;
    execute_scalar_alu(
        &instruction,
        0x1000,
        &mut registers,
        &mut exec_mask,
        &mut scalar_condition,
    )
    .expect("64-bit bit count");
    let source = u64::from(registers[0]) | (u64::from(registers[1]) << 32);
    assert_eq!(registers[10], source.count_ones());
    assert!(scalar_condition);

    registers[0] = 0;
    registers[1] = 0;
    scalar_condition = true;
    execute_scalar_alu(
        &instruction,
        0x1000,
        &mut registers,
        &mut exec_mask,
        &mut scalar_condition,
    )
    .expect("zero 64-bit bit count");
    assert_eq!(registers[10], 0);
    assert!(!scalar_condition);
}

#[test]
fn s_bfm_b64_uses_32_bit_controls_and_preserves_scc() {
    let operation = instruction(
        0,
        ShaderEncoding::Sop2,
        "SBfmB64",
        vec![Operand::scalar(0), Operand::scalar(2)],
        vec![Operand::scalar(10)],
        None,
    );

    for (width, offset, expected) in [
        (0_u32, 12_u32, 0_u64),
        (32, 16, 0x0000_ffff_ffff_0000),
        (0x43, 0x7c, 0x7000_0000_0000_0000),
        (63, 63, 0x8000_0000_0000_0000),
    ] {
        for original_scc in [false, true] {
            let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
            let mut exec_mask = u64::MAX;
            let mut scalar_condition = original_scc;
            registers[0] = width;
            registers[2] = offset;

            execute_scalar_alu(
                &operation,
                0x1000,
                &mut registers,
                &mut exec_mask,
                &mut scalar_condition,
            )
            .expect("S_BFM_B64 evaluates");

            let result = u64::from(registers[10]) | (u64::from(registers[11]) << 32);
            assert_eq!(
                result, expected,
                "width=0x{width:08X}, offset=0x{offset:08X}"
            );
            assert_eq!(
                scalar_condition, original_scc,
                "S_BFM_B64 must not modify SCC"
            );
        }
    }
}

#[test]
fn s_pack_ll_b32_b16_packs_low_halves_and_preserves_scc() {
    let operation = instruction(
        0,
        ShaderEncoding::Sop2,
        "SPackLlB32B16",
        vec![Operand::scalar(0), Operand::scalar(1)],
        vec![Operand::scalar(2)],
        None,
    );

    for original_scc in [false, true] {
        let mut scalar_condition = original_scc;
        assert_eq!(
            execute_scalar_binary(&operation, 0x1234_abcd, 0x5678_ef01, &mut scalar_condition)
                .expect("S_PACK_LL_B32_B16 evaluates"),
            0xef01_abcd
        );
        assert_eq!(
            scalar_condition, original_scc,
            "S_PACK_LL_B32_B16 must not modify SCC"
        );
    }
}

#[test]
fn s_bitcmp1_b32_uses_the_low_five_index_bits() {
    let instruction = instruction(
        0,
        ShaderEncoding::Sopc,
        "SBitcmp1B32",
        vec![Operand::scalar(0), Operand::scalar(1)],
        Vec::new(),
        None,
    );
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    registers[0] = 1 << 7;

    for index in [7_u32, 39_u32] {
        registers[1] = index;
        assert!(
            execute_scalar_compare(&instruction, &registers, u64::MAX)
                .expect("S_BITCMP1_B32 scalar evaluation"),
            "bit index {index} must select source bit 7"
        );
    }
    registers[1] = 6;
    assert!(
        !execute_scalar_compare(&instruction, &registers, u64::MAX)
            .expect("clear S_BITCMP1_B32 scalar evaluation")
    );
}

#[test]
fn evaluates_scalar_arithmetic_and_taken_control_flow() {
    let mut branch = instruction(
        12,
        ShaderEncoding::Sopp,
        "SCbranchScc1",
        vec![Operand {
            kind: OperandKind::EncodedConstant,
            value: 1,
        }],
        Vec::new(),
        None,
    );
    branch.words[0] = 1;
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sopk,
                "SMovkI32",
                vec![Operand {
                    kind: OperandKind::EncodedConstant,
                    value: 5,
                }],
                vec![Operand::scalar(0)],
                None,
            ),
            instruction(
                4,
                ShaderEncoding::Sopk,
                "SAddkI32",
                vec![Operand {
                    kind: OperandKind::EncodedConstant,
                    value: u32::from(u16::MAX),
                }],
                vec![Operand::scalar(0)],
                None,
            ),
            instruction(
                8,
                ShaderEncoding::Sopk,
                "SCmpkEqI32",
                vec![Operand {
                    kind: OperandKind::EncodedConstant,
                    value: 4,
                }],
                vec![Operand::scalar(0)],
                None,
            ),
            branch,
            instruction(
                16,
                ShaderEncoding::Sopk,
                "SMovkI32",
                vec![Operand {
                    kind: OperandKind::EncodedConstant,
                    value: 7,
                }],
                vec![Operand::scalar(1)],
                None,
            ),
            instruction(
                20,
                ShaderEncoding::Sopk,
                "SMovkI32",
                vec![Operand {
                    kind: OperandKind::EncodedConstant,
                    value: 9,
                }],
                vec![Operand::scalar(1)],
                None,
            ),
            end(24),
        ],
        Vec::new(),
    );
    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("scalar evaluation");
    assert_eq!(evaluation.scalar_registers[0], 4);
    assert_eq!(evaluation.scalar_registers[1], 9);
}

#[test]
fn scalar_absdiff_wraps_at_i32_width_and_sets_scc_from_the_result() {
    let operation = instruction(
        0,
        ShaderEncoding::Sop2,
        "SAbsdiffI32",
        vec![Operand::scalar(0), Operand::scalar(1)],
        vec![Operand::scalar(2)],
        None,
    );
    for (left, right, expected, expected_scc) in [
        (0x0000_0002, 0x0000_0005, 0x0000_0003, true),
        (0xffff_ffff, 0x0000_0000, 0x0000_0001, true),
        (0x8000_0000, 0x0000_0000, 0x8000_0000, true),
        (0x8000_0000, 0x0000_0001, 0x7fff_ffff, true),
        (0x8000_0000, 0xffff_ffff, 0x7fff_ffff, true),
        (0x8000_0000, 0xffff_fffe, 0x7fff_fffe, true),
        (0x1234_5678, 0x1234_5678, 0x0000_0000, false),
    ] {
        let mut scc = !expected_scc;
        assert_eq!(
            execute_scalar_binary(&operation, left, right, &mut scc)
                .expect("S_ABSDIFF_I32 evaluates"),
            expected,
            "left=0x{left:08X}, right=0x{right:08X}"
        );
        assert_eq!(scc, expected_scc, "left=0x{left:08X}, right=0x{right:08X}");
    }
}

#[test]
fn evaluates_unsigned_64_bit_scalar_comparisons_as_register_pairs() {
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    registers[10] = 0x89ab_cdef;
    registers[11] = 0x0123_4567;
    registers[20] = 0x89ab_cdef;
    registers[21] = 0x0123_4567;
    let compare = instruction(
        0,
        ShaderEncoding::Sopc,
        "SCmpEqU64",
        vec![Operand::scalar(10), Operand::scalar(20)],
        Vec::new(),
        None,
    );
    assert!(
        execute_scalar_compare(&compare, &registers, u64::MAX)
            .expect("equal register pairs compare")
    );

    registers[21] ^= 1;
    let not_equal = Instruction {
        opcode: Opcode::from_static("SCmpLgU64"),
        ..compare
    };
    assert!(
        execute_scalar_compare(&not_equal, &registers, u64::MAX)
            .expect("different register pairs compare")
    );
}
