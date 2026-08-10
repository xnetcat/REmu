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
fn unsigned_sopk_compare_zero_extends_its_immediate() {
    let compare = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Sopk,
        opcode: Opcode::Known("SCmpkLeU32"),
        words: vec![0xb704_ffff],
        sources: vec![Operand {
            kind: OperandKind::EncodedConstant,
            value: u32::from(u16::MAX),
        }],
        destinations: vec![Operand::scalar(4)],
        control: None,
    };
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    registers[4] = 0x1_0000;

    assert!(!execute_scalar_compare_k(&compare, &registers).expect("unsigned compare-k"));

    let mut signed = compare;
    signed.opcode = Opcode::Known("SCmpkLeI32");
    registers[4] = u32::MAX;
    assert!(execute_scalar_compare_k(&signed, &registers).expect("signed compare-k"));
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

fn unary(opcode: &'static str, source: u32, destination: u32) -> Instruction {
    instruction(
        0,
        ShaderEncoding::Sop1,
        opcode,
        vec![Operand::scalar(source)],
        vec![Operand::scalar(destination)],
        None,
    )
}

fn run_unary(
    operation: &Instruction,
    registers: &mut [u32; SCALAR_REGISTER_COUNT],
    scalar_condition: &mut bool,
) {
    let mut exec_mask = u64::MAX;
    execute_scalar_alu(
        operation,
        0x1000,
        registers,
        &mut exec_mask,
        scalar_condition,
    )
    .expect("SOP1 batch operation evaluates");
}

#[test]
fn s_cmovk_and_s_cmov_write_only_when_scc_is_set() {
    let cmovk = instruction(
        0,
        ShaderEncoding::Sopk,
        "SCmovkI32",
        vec![Operand {
            kind: OperandKind::EncodedConstant,
            value: 0xfffe,
        }],
        vec![Operand::scalar(4)],
        None,
    );
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    let mut exec_mask = u64::MAX;
    registers[4] = 11;
    let mut scalar_condition = false;
    execute_scalar_alu(
        &cmovk,
        0x1000,
        &mut registers,
        &mut exec_mask,
        &mut scalar_condition,
    )
    .expect("clear-SCC S_CMOVK_I32");
    assert_eq!(
        registers[4], 11,
        "S_CMOVK_I32 must not write while SCC is clear"
    );
    scalar_condition = true;
    execute_scalar_alu(
        &cmovk,
        0x1000,
        &mut registers,
        &mut exec_mask,
        &mut scalar_condition,
    )
    .expect("set-SCC S_CMOVK_I32");
    assert_eq!(registers[4], 0xffff_fffe, "simm16 sign-extends");
    assert!(scalar_condition, "S_CMOVK_I32 must not modify SCC");

    let cmov32 = unary("SCmovB32", 2, 4);
    registers[2] = 0x1234_5678;
    registers[4] = 5;
    scalar_condition = false;
    run_unary(&cmov32, &mut registers, &mut scalar_condition);
    assert_eq!(registers[4], 5);
    scalar_condition = true;
    run_unary(&cmov32, &mut registers, &mut scalar_condition);
    assert_eq!(registers[4], 0x1234_5678);

    let cmov64 = unary("SCmovB64", 2, 6);
    registers[2] = 0xaaaa_bbbb;
    registers[3] = 0xcccc_dddd;
    registers[6] = 1;
    registers[7] = 2;
    scalar_condition = false;
    run_unary(&cmov64, &mut registers, &mut scalar_condition);
    assert_eq!((registers[6], registers[7]), (1, 2));
    scalar_condition = true;
    run_unary(&cmov64, &mut registers, &mut scalar_condition);
    assert_eq!((registers[6], registers[7]), (0xaaaa_bbbb, 0xcccc_dddd));
}

#[test]
fn s_brev_and_s_ff1_do_not_modify_scc() {
    // RDNA2 ISA: S_BREV_B32/B64 and S_FF1_I32_B32/B64 write no SCC. The evaluator previously
    // fabricated SCC = (result != 0) for both.
    for (opcode, source_value) in [("SBrevB32", 0x8000_0001_u32), ("SFF1I32B32", 0x10)] {
        for original_scc in [false, true] {
            let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
            registers[2] = source_value;
            let mut scalar_condition = original_scc;
            run_unary(&unary(opcode, 2, 4), &mut registers, &mut scalar_condition);
            assert_eq!(scalar_condition, original_scc, "{opcode} must preserve SCC");
        }
    }
}

#[test]
fn sop1_bit_scans_match_the_isa_reference() {
    let mut scalar_condition = false;

    // S_FF0_I32_B32: position of the first zero from bit 0; -1 when the source is all ones.
    for (source, expected) in [(0_u32, 0_u32), (0b0111, 3), (u32::MAX, u32::MAX)] {
        let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
        registers[2] = source;
        run_unary(
            &unary("SFF0I32B32", 2, 4),
            &mut registers,
            &mut scalar_condition,
        );
        assert_eq!(registers[4], expected, "S_FF0_I32_B32 of 0x{source:08X}");
    }

    // S_FF0_I32_B64 over the register pair.
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    registers[2] = u32::MAX;
    registers[3] = 0b0111;
    run_unary(
        &unary("SFF0I32B64", 2, 4),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!(registers[4], 35, "first zero sits at bit 35 of the pair");

    // S_FF1_I32_B64: first one from bit 0; -1 when zero.
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    registers[2] = 0;
    registers[3] = 0x8000_0000;
    run_unary(
        &unary("SFF1I32B64", 2, 4),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!(registers[4], 63);
    registers[2] = 0;
    registers[3] = 0;
    run_unary(
        &unary("SFF1I32B64", 2, 4),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!(registers[4], u32::MAX);

    // S_FLBIT_I32_B32: position of the first one counting from bit 31; -1 when zero.
    for (source, expected) in [(0_u32, u32::MAX), (0x8000_0000, 0), (1, 31)] {
        let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
        registers[2] = source;
        run_unary(
            &unary("SFlbitI32B32", 2, 4),
            &mut registers,
            &mut scalar_condition,
        );
        assert_eq!(registers[4], expected, "S_FLBIT_I32_B32 of 0x{source:08X}");
    }

    // S_FLBIT_I32: first bit differing from the sign, from the MSB; -1 for 0 and -1.
    for (source, expected) in [
        (0_u32, u32::MAX),
        (u32::MAX, u32::MAX),
        (1, 31),
        (0xffff_fffe, 31),
        (0x7fff_ffff, 1),
    ] {
        let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
        registers[2] = source;
        run_unary(
            &unary("SFlbitI32", 2, 4),
            &mut registers,
            &mut scalar_condition,
        );
        assert_eq!(registers[4], expected, "S_FLBIT_I32 of 0x{source:08X}");
    }

    // S_FLBIT_I32_I64 over the pair.
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    registers[2] = 1;
    registers[3] = 0;
    run_unary(
        &unary("SFlbitI32I64", 2, 4),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!(registers[4], 63);

    // S_BCNT0 counts zeros and sets SCC from a nonzero result.
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    registers[2] = u32::MAX;
    scalar_condition = true;
    run_unary(
        &unary("SBcnt0I32B32", 2, 4),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!(registers[4], 0);
    assert!(!scalar_condition, "S_BCNT0 SCC follows the result");
    registers[2] = 0xffff_0000;
    run_unary(
        &unary("SBcnt0I32B32", 2, 4),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!(registers[4], 16);
    assert!(scalar_condition);
    registers[2] = u32::MAX;
    registers[3] = 0;
    run_unary(
        &unary("SBcnt0I32B64", 2, 4),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!(registers[4], 32);
    assert!(scalar_condition);
}

#[test]
fn s_sext_abs_wqm_and_quadmask_semantics() {
    let mut scalar_condition = false;

    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    registers[2] = 0x0000_1280;
    run_unary(
        &unary("SSextI32I8", 2, 4),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!(registers[4], 0xffff_ff80);
    registers[2] = 0x0001_8000;
    run_unary(
        &unary("SSextI32I16", 2, 4),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!(registers[4], 0xffff_8000);

    registers[2] = 0x8000_0000;
    scalar_condition = false;
    run_unary(
        &unary("SAbsI32", 2, 4),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!(registers[4], 0x8000_0000, "|INT_MIN| wraps to INT_MIN");
    assert!(scalar_condition);
    registers[2] = 0xffff_fffb;
    run_unary(
        &unary("SAbsI32", 2, 4),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!(registers[4], 5);

    // S_WQM_B32: whole-quad mode over eight quads, SCC from the result.
    registers[2] = 0x0000_0012;
    scalar_condition = false;
    run_unary(
        &unary("SWqmB32", 2, 4),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!(registers[4], 0x0000_00ff);
    assert!(scalar_condition);

    // S_QUADMASK_B32: one output bit per nonzero input nibble.
    registers[2] = 0x0000_0102;
    run_unary(
        &unary("SQuadmaskB32", 2, 4),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!(registers[4], 0b101);
    assert!(scalar_condition);

    // S_QUADMASK_B64 packs sixteen nibble bits into the destination pair.
    registers[2] = 0x0000_0001;
    registers[3] = 0x1000_0000;
    run_unary(
        &unary("SQuadmaskB64", 2, 6),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!(registers[6], 0x8001);
    assert_eq!(registers[7], 0);
    assert!(scalar_condition);

    // S_BREV_B64 mirrors the complete pair.
    registers[2] = 0x8000_0000;
    registers[3] = 0x0000_0001;
    run_unary(
        &unary("SBrevB64", 2, 6),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!((registers[6], registers[7]), (0x8000_0000, 0x0000_0001));
}

#[test]
fn s_bitset_and_bitreplicate_modify_the_destination_pair() {
    let mut scalar_condition = false;

    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    registers[2] = 39; // low six bits select the pair bit
    registers[4] = 0;
    registers[5] = 0;
    run_unary(
        &unary("SBitset1B64", 2, 4),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!((registers[4], registers[5]), (0, 1 << 7));

    registers[2] = 3;
    registers[4] = 0b1111;
    run_unary(
        &unary("SBitset0B32", 2, 4),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!(registers[4], 0b0111);

    registers[2] = 39;
    registers[4] = u32::MAX;
    registers[5] = u32::MAX;
    run_unary(
        &unary("SBitset0B64", 2, 4),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!((registers[4], registers[5]), (u32::MAX, !(1 << 7)));

    registers[2] = 0b101;
    run_unary(
        &unary("SBitreplicateB64B32", 2, 4),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!((registers[4], registers[5]), (0b110011, 0));
}

#[test]
fn s_movrels_reads_and_s_movreld_writes_the_m0_relative_register() {
    let mut scalar_condition = false;

    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    registers[124] = 3; // M0
    registers[13] = 0xfeed_beef; // s10 + M0
    run_unary(
        &unary("SMovrelsB32", 10, 4),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!(registers[4], 0xfeed_beef);

    registers[14] = 0x1111_2222;
    run_unary(
        &unary("SMovrelsB64", 10, 6),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!((registers[6], registers[7]), (0xfeed_beef, 0x1111_2222));

    registers[2] = 0xcafe_f00d;
    run_unary(
        &unary("SMovreldB32", 2, 20),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!(registers[23], 0xcafe_f00d, "s20 + M0 receives the value");

    registers[2] = 0x0102_0304;
    registers[3] = 0x0506_0708;
    run_unary(
        &unary("SMovreldB64", 2, 30),
        &mut registers,
        &mut scalar_condition,
    );
    assert_eq!((registers[33], registers[34]), (0x0102_0304, 0x0506_0708));

    // An index escaping the architectural register file is a loud error, not a wrap.
    registers[124] = 300;
    let error = {
        let mut exec_mask = u64::MAX;
        execute_scalar_alu(
            &unary("SMovrelsB32", 10, 4),
            0x1000,
            &mut registers,
            &mut exec_mask,
            &mut scalar_condition,
        )
        .expect_err("out-of-range relative scalar index")
    };
    assert!(matches!(error, EvaluationError::InvalidInstruction { .. }));
}

#[test]
fn scalar_hint_opcodes_are_benign_and_getreg_is_loudly_unsupported() {
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    let mut exec_mask = u64::MAX;
    let mut scalar_condition = true;
    registers[5] = 77;

    for opcode in [
        "SVersion",
        "SSetregB32",
        "SSetregImm32B32",
        "SWaitcntVscnt",
        "SWaitcntVmcnt",
        "SWaitcntExpcnt",
        "SWaitcntLgkmcnt",
    ] {
        let hint = instruction(
            0,
            ShaderEncoding::Sopk,
            opcode,
            vec![Operand {
                kind: OperandKind::EncodedConstant,
                value: 0,
            }],
            Vec::new(),
            None,
        );
        execute_scalar_alu(
            &hint,
            0x1000,
            &mut registers,
            &mut exec_mask,
            &mut scalar_condition,
        )
        .unwrap_or_else(|error| panic!("{opcode} must be benign: {error}"));
        assert_eq!(registers[5], 77);
        assert!(scalar_condition, "{opcode} must not modify SCC");
    }

    let getreg = instruction(
        0,
        ShaderEncoding::Sopk,
        "SGetregB32",
        vec![Operand {
            kind: OperandKind::EncodedConstant,
            value: 0xf801,
        }],
        vec![Operand::scalar(5)],
        None,
    );
    let error = execute_scalar_alu(
        &getreg,
        0x1000,
        &mut registers,
        &mut exec_mask,
        &mut scalar_condition,
    )
    .expect_err("S_GETREG_B32 reads unmodeled hardware state and must fail loudly");
    assert!(matches!(
        error,
        EvaluationError::UnsupportedScalarOpcode { .. }
    ));
}

#[test]
fn raw_scalar_opcodes_fail_loudly_and_raw_vector_opcodes_stay_benign() {
    // A raw scalar op on an executed path must never produce fabricated register state.
    let raw_scalar = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Sop1,
        opcode: Opcode::Raw {
            family: crate::RawOpcodeFamily::Sop1,
            code: 0x69,
        },
        words: vec![0],
        sources: vec![Operand::scalar(2)],
        destinations: vec![Operand::scalar(4)],
        control: None,
    };
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    let mut exec_mask = u64::MAX;
    let mut scalar_condition = false;
    let error = execute_scalar_alu(
        &raw_scalar,
        0x1000,
        &mut registers,
        &mut exec_mask,
        &mut scalar_condition,
    )
    .expect_err("raw scalar opcode must be loudly unsupported");
    assert!(matches!(
        error,
        EvaluationError::UnsupportedScalarOpcode { .. }
    ));

    // A raw vector op is invisible to scalar discovery: the program still evaluates.
    let mut raw_vector = instruction(
        0,
        ShaderEncoding::Vop1,
        "placeholder",
        vec![Operand::vector(1)],
        vec![Operand::vector(2)],
        None,
    );
    raw_vector.opcode = Opcode::Raw {
        family: crate::RawOpcodeFamily::Vop1,
        code: 0x6a,
    };
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sopk,
                "SMovkI32",
                vec![Operand {
                    kind: OperandKind::EncodedConstant,
                    value: 21,
                }],
                vec![Operand::scalar(0)],
                None,
            ),
            raw_vector,
            end(8),
        ],
        Vec::new(),
    );
    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("raw vector opcodes must not fail scalar evaluation");
    assert_eq!(evaluation.scalar_registers[0], 21);
}
