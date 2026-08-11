// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Control-flow evaluation tests: statically inactive blocks, bounded backward
//! edges and the vector-controlled loops scalar discovery must terminate on.

use super::*;

fn relative_branch(pc: u32, opcode: &'static str, offset: i16) -> Instruction {
    let mut branch = instruction(
        pc,
        ShaderEncoding::Sopp,
        opcode,
        Vec::new(),
        Vec::new(),
        None,
    );
    branch.words[0] = u32::from(offset.cast_unsigned());
    branch
}

#[test]
fn preserves_initial_image_descriptors_for_statically_inactive_blocks() {
    let image = ImageControl {
        dmask: 0xf,
        vector_address: 0,
        address_registers: vec![0, 1],
        vector_data: 4,
        scalar_resource: 0,
        scalar_sampler: 8,
        dimension: 1,
        is_array: false,
        glc: false,
        slc: false,
        a16: false,
        d16: false,
        unorm: false,
    };
    let mut branch = instruction(
        4,
        ShaderEncoding::Sopp,
        "SCbranchScc1",
        Vec::new(),
        Vec::new(),
        None,
    );
    branch.words[0] = 2;
    let mut registers = (1..=12).collect::<Vec<_>>();
    registers.resize(21, 0);
    registers[20] = 1;
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sopk,
                "SCmpkEqI32",
                vec![Operand {
                    kind: OperandKind::EncodedConstant,
                    value: 1,
                }],
                vec![Operand::scalar(20)],
                None,
            ),
            branch,
            instruction(
                8,
                ShaderEncoding::Mimg,
                "ImageSampleLz",
                Vec::new(),
                Vec::new(),
                Some(InstructionControl::Image(image)),
            ),
            end(16),
        ],
        registers,
    );

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("inactive image descriptor should remain available to Metal lowering");

    assert_eq!(evaluation.image_bindings.len(), 1);
    assert_eq!(evaluation.image_bindings[0].pc, 8);
    assert_eq!(
        evaluation.image_bindings[0].resource_descriptor.as_slice(),
        (1..=8).collect::<Vec<_>>().as_slice()
    );
    assert_eq!(
        evaluation.image_bindings[0].sampler_descriptor.as_slice(),
        (9..=12).collect::<Vec<_>>().as_slice()
    );
}

#[test]
fn forward_scalar_discovery_loads_an_inactive_image_descriptor_over_its_user_pointer() {
    let pointer = 0x0000_0070_4030_0090_u64;
    let descriptor: [u32; 8] = [
        0x1234_5000,
        0x0000_0070,
        0x00ff_00ff,
        0x9008_1000,
        0,
        0,
        0,
        0,
    ];
    let mut memory = TestMemory::default();
    let bytes = descriptor
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    memory.write(pointer, &bytes);

    let scalar_load = instruction(
        8,
        ShaderEncoding::Smem,
        "SLoadDwordx8",
        vec![Operand::scalar(0)],
        (0..8).map(Operand::scalar).collect(),
        Some(InstructionControl::ScalarMemory(ScalarMemoryControl {
            destination_count: 8,
            immediate_offset_bytes: 0,
            dynamic_offset_register: None,
        })),
    );
    let compare = instruction(
        0,
        ShaderEncoding::Sopk,
        "SCmpkEqI32",
        vec![Operand {
            kind: OperandKind::EncodedConstant,
            value: 1,
        }],
        vec![Operand::scalar(8)],
        None,
    );
    let mut branch = instruction(
        4,
        ShaderEncoding::Sopp,
        "SCbranchScc1",
        Vec::new(),
        Vec::new(),
        None,
    );
    branch.words[0] = 4;
    let image = instruction(
        16,
        ShaderEncoding::Mimg,
        "ImageStore",
        Vec::new(),
        Vec::new(),
        Some(InstructionControl::Image(ImageControl {
            dmask: 0xf,
            vector_address: 0,
            address_registers: vec![0, 1],
            vector_data: 4,
            scalar_resource: 0,
            scalar_sampler: 0,
            dimension: 1,
            is_array: false,
            glc: false,
            slc: false,
            a16: false,
            d16: false,
            unorm: false,
        })),
    );
    let mut user_data = vec![pointer as u32, (pointer >> 32) as u32];
    user_data.resize(8, 0);
    user_data.push(1);
    let shader = state(
        vec![compare, branch, scalar_load, image, end(24)],
        user_data,
    );

    let evaluation = evaluate_shader(&memory, &shader, EvaluationOptions::default())
        .expect("the skipped-block scalar load must supply the inactive image descriptor");

    assert_eq!(evaluation.image_bindings.len(), 1);
    assert_eq!(
        evaluation.image_bindings[0].resource_descriptor.as_slice(),
        descriptor
    );
}

#[test]
fn bounds_vector_controlled_backward_loops_during_scalar_discovery() {
    let mut branch = instruction(
        4,
        ShaderEncoding::Sopp,
        "SCbranchExecnz",
        Vec::new(),
        Vec::new(),
        None,
    );
    branch.words[0] = u32::from((-2_i16).cast_unsigned());
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Vop1,
                "VMovB32",
                vec![Operand::vector(0)],
                vec![Operand::vector(1)],
                None,
            ),
            branch,
            end(8),
        ],
        Vec::new(),
    );

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("vector-controlled runtime loop must not exhaust static discovery");
    assert!(evaluation.global_memory_bindings.is_empty());
}

#[test]
fn conversion_only_vector_alu_is_skipped_by_scalar_resource_evaluation() {
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Vop1,
                "VCvtFlrI32F32",
                vec![Operand::vector(1)],
                vec![Operand::vector(8)],
                None,
            ),
            instruction(
                4,
                ShaderEncoding::Vop3,
                "VCvtPknormU16F32",
                vec![
                    Operand::vector(2),
                    Operand::vector(3),
                    Operand::source(128, None),
                ],
                vec![Operand::vector(9)],
                Some(InstructionControl::Vop3(crate::Vop3Control {
                    absolute_mask: 0,
                    negate_mask: 0,
                    output_modifier: 0,
                    clamp: false,
                    operand_select: 0,
                    scalar_destination: None,
                })),
            ),
            end(12),
        ],
        Vec::new(),
    );

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("vector conversions must not enter scalar resource evaluation");
    assert!(evaluation.scalar_memory_reads.is_empty());
    assert!(evaluation.global_memory_bindings.is_empty());
    assert!(evaluation.image_bindings.is_empty());
}

#[test]
fn exits_an_unconditional_back_edge_after_reaching_an_observable_recurrence() {
    let mut exit = instruction(
        4,
        ShaderEncoding::Sopp,
        "SCbranchExecz",
        Vec::new(),
        Vec::new(),
        None,
    );
    exit.words[0] = 2;
    let mut back_edge = instruction(
        12,
        ShaderEncoding::Sopp,
        "SBranch",
        Vec::new(),
        Vec::new(),
        None,
    );
    back_edge.words[0] = u32::from((-3_i16).cast_unsigned());
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Vop1,
                "VMovB32",
                vec![Operand::vector(0)],
                vec![Operand::vector(1)],
                None,
            ),
            exit,
            instruction(
                8,
                ShaderEncoding::Vop1,
                "VMovB32",
                vec![Operand::vector(1)],
                vec![Operand::vector(2)],
                None,
            ),
            back_edge,
            end(16),
        ],
        Vec::new(),
    );

    evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("unconditional vector loop back edge must remain bounded");
}

#[test]
fn detects_a_non_unit_scalar_recurrence_in_constant_space() {
    let mut path = ScalarPath {
        instruction_index: 0,
        registers: [0; SCALAR_REGISTER_COUNT],
        exec_mask: RDNA_WAVE_MASK,
        scalar_condition: false,
        supplemental: false,
        backward_branch_visits: SmallVec::new(),
        backward_branch_recurrences: SmallVec::new(),
    };
    let recurrence = |value| {
        let mut registers = [0; SCALAR_REGISTER_COUNT];
        registers[20] = value;
        scalar_recurrence_state(&registers, &[20], RDNA_WAVE_MASK, false, false)
    };

    assert_eq!(
        path.observe_backward_branch_recurrence(0x40, recurrence(1)),
        None
    );
    assert_eq!(
        path.observe_backward_branch_recurrence(0x40, recurrence(2)),
        None
    );
    assert_eq!(
        path.observe_backward_branch_recurrence(0x40, recurrence(1)),
        None
    );
    assert_eq!(
        path.observe_backward_branch_recurrence(0x40, recurrence(2)),
        Some(2)
    );
}

#[test]
fn executes_a_scalar_counted_loop_closed_by_an_unconditional_latch() {
    let mut exit = instruction(
        8,
        ShaderEncoding::Sopp,
        "SCbranchScc0",
        Vec::new(),
        Vec::new(),
        None,
    );
    exit.words[0] = 2;
    let mut back_edge = instruction(
        16,
        ShaderEncoding::Sopp,
        "SBranch",
        Vec::new(),
        Vec::new(),
        None,
    );
    back_edge.words[0] = u32::from((-4_i16).cast_unsigned());
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sop1,
                "SMovB32",
                vec![Operand::source(128, None)],
                vec![Operand::scalar(107)],
                None,
            ),
            instruction(
                4,
                ShaderEncoding::Sopc,
                "SCmpLtU32",
                vec![Operand::scalar(107), Operand::scalar(20)],
                Vec::new(),
                None,
            ),
            exit,
            instruction(
                12,
                ShaderEncoding::Sop2,
                "SAddU32",
                vec![Operand::scalar(107), Operand::source(132, None)],
                vec![Operand::scalar(107)],
                None,
            ),
            back_edge,
            end(20),
        ],
        {
            let mut registers = vec![0; 21];
            registers[20] = 16_388;
            registers
        },
    );

    let evaluation = evaluate_shader_within(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
        20_000,
    )
    .expect("scalar counted loop should run to its SCC exit");

    assert_eq!(evaluation.scalar_registers[107], 16_388);
}

#[test]
fn converges_an_scc_exit_derived_from_exec_in_an_unconditional_latch_loop() {
    let exit = relative_branch(4, "SCbranchScc0", 3);
    let internal_exec_branch = relative_branch(8, "SCbranchExecz", 1);
    let back_edge = relative_branch(16, "SBranch", -5);
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sop2,
                "SAndn2B64",
                vec![Operand::scalar(0), Operand::scalar(126)],
                vec![Operand::scalar(0)],
                None,
            ),
            exit,
            internal_exec_branch,
            instruction(
                12,
                ShaderEncoding::Sopp,
                "SNop",
                Vec::new(),
                Vec::new(),
                None,
            ),
            back_edge,
            end(20),
        ],
        vec![u32::MAX, u32::MAX],
    );

    evaluate_shader_within(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
        16,
    )
    .expect("EXEC-derived SCC loop exit should use vector-loop recurrence discovery");
}

#[test]
fn keeps_a_scalar_only_scc_exit_exact_in_an_unconditional_latch_loop() {
    let exit = relative_branch(4, "SCbranchScc0", 2);
    let back_edge = relative_branch(12, "SBranch", -4);
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sop2,
                "SAndB64",
                vec![Operand::scalar(0), Operand::scalar(0)],
                vec![Operand::scalar(0)],
                None,
            ),
            exit,
            instruction(
                8,
                ShaderEncoding::Sopp,
                "SNop",
                Vec::new(),
                Vec::new(),
                None,
            ),
            back_edge,
            end(16),
        ],
        vec![u32::MAX, u32::MAX],
    );

    let error = evaluate_shader_within(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
        16,
    )
    .expect_err("scalar-only SCC loop exit must remain exact");

    assert!(matches!(error, EvaluationError::ControlFlowLimit { .. }));
}

#[test]
fn converges_a_saveexec_scc_exit_in_an_unconditional_latch_loop() {
    let exit = relative_branch(4, "SCbranchScc0", 2);
    let back_edge = relative_branch(12, "SBranch", -4);
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sop1,
                "SAndSaveexecB64",
                vec![Operand::scalar(2)],
                vec![Operand::scalar(4)],
                None,
            ),
            exit,
            instruction(
                8,
                ShaderEncoding::Sopp,
                "SNop",
                Vec::new(),
                Vec::new(),
                None,
            ),
            back_edge,
            end(16),
        ],
        vec![0, 0, u32::MAX, u32::MAX],
    );

    evaluate_shader_within(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
        16,
    )
    .expect("saveexec SCC loop exit should use vector-loop recurrence discovery");
}

#[test]
fn converges_an_scc_exit_derived_from_an_arbitrary_vop3b_scalar_destination() {
    let exit = relative_branch(8, "SCbranchScc0", 2);
    let back_edge = relative_branch(16, "SBranch", -5);
    let mut registers = vec![u32::MAX, u32::MAX];
    registers.resize(22, u32::MAX);
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Vop3,
                "VDivScaleF32",
                vec![Operand::vector(0), Operand::vector(1), Operand::vector(2)],
                vec![Operand::vector(3)],
                Some(InstructionControl::Vop3(crate::Vop3Control {
                    absolute_mask: 0,
                    negate_mask: 0,
                    output_modifier: 0,
                    clamp: false,
                    operand_select: 0,
                    scalar_destination: Some(20),
                })),
            ),
            instruction(
                4,
                ShaderEncoding::Sop2,
                "SAndB64",
                vec![Operand::scalar(0), Operand::scalar(20)],
                vec![Operand::scalar(0)],
                None,
            ),
            exit,
            instruction(
                12,
                ShaderEncoding::Sopp,
                "SNop",
                Vec::new(),
                Vec::new(),
                None,
            ),
            back_edge,
            end(20),
        ],
        registers,
    );

    evaluate_shader_within(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
        20,
    )
    .expect("VOP3B-derived SCC loop exit should use vector-loop recurrence discovery");
}

#[test]
fn carries_a_preheader_vop3b_scalar_destination_into_the_loop() {
    let exit = relative_branch(8, "SCbranchScc0", 2);
    let back_edge = relative_branch(16, "SBranch", -4);
    let mut registers = vec![u32::MAX, u32::MAX];
    registers.resize(22, u32::MAX);
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Vop3,
                "VDivScaleF32",
                vec![Operand::vector(0), Operand::vector(1), Operand::vector(2)],
                vec![Operand::vector(3)],
                Some(InstructionControl::Vop3(crate::Vop3Control {
                    absolute_mask: 0,
                    negate_mask: 0,
                    output_modifier: 0,
                    clamp: false,
                    operand_select: 0,
                    scalar_destination: Some(20),
                })),
            ),
            instruction(
                4,
                ShaderEncoding::Sop2,
                "SAndB64",
                vec![Operand::scalar(0), Operand::scalar(20)],
                vec![Operand::scalar(0)],
                None,
            ),
            exit,
            instruction(
                12,
                ShaderEncoding::Sopp,
                "SNop",
                Vec::new(),
                Vec::new(),
                None,
            ),
            back_edge,
            end(20),
        ],
        registers,
    );

    evaluate_shader_within(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
        20,
    )
    .expect("a preheader vector scalar result must taint the SCC value inside the loop");
}

#[test]
fn scalar_constant_overwrites_detaint_exec_and_vcc() {
    for mask_register in [106, 126] {
        let exit = relative_branch(8, "SCbranchScc0", 2);
        let back_edge = relative_branch(16, "SBranch", -5);
        let shader = state(
            vec![
                instruction(
                    0,
                    ShaderEncoding::Sop1,
                    "SMovB64",
                    vec![Operand::source(128, None)],
                    vec![Operand::scalar(mask_register)],
                    None,
                ),
                instruction(
                    4,
                    ShaderEncoding::Sop2,
                    "SOrB64",
                    vec![Operand::scalar(0), Operand::scalar(mask_register)],
                    vec![Operand::scalar(0)],
                    None,
                ),
                exit,
                instruction(
                    12,
                    ShaderEncoding::Sopp,
                    "SNop",
                    Vec::new(),
                    Vec::new(),
                    None,
                ),
                back_edge,
                end(20),
            ],
            vec![u32::MAX, u32::MAX],
        );

        let error = evaluate_shader_within(
            &TestMemory::default(),
            &shader,
            EvaluationOptions::default(),
            20,
        )
        .expect_err("a scalar constant mask must keep the SCC loop exact");

        assert!(matches!(error, EvaluationError::ControlFlowLimit { .. }));
    }
}

#[test]
fn vector_compares_retaint_their_implicit_default_masks() {
    for (opcode, mask_register) in [("VCmpEqF32", 106), ("VCmpxEqF32", 126)] {
        let exit = relative_branch(12, "SCbranchScc0", 2);
        let back_edge = relative_branch(20, "SBranch", -6);
        let shader = state(
            vec![
                instruction(
                    0,
                    ShaderEncoding::Sop1,
                    "SMovB64",
                    vec![Operand::source(128, None)],
                    vec![Operand::scalar(mask_register)],
                    None,
                ),
                instruction(
                    4,
                    ShaderEncoding::Vopc,
                    opcode,
                    vec![Operand::vector(0), Operand::vector(1)],
                    Vec::new(),
                    None,
                ),
                instruction(
                    8,
                    ShaderEncoding::Sop2,
                    "SOrB64",
                    vec![Operand::scalar(0), Operand::scalar(mask_register)],
                    vec![Operand::scalar(0)],
                    None,
                ),
                exit,
                instruction(
                    16,
                    ShaderEncoding::Sopp,
                    "SNop",
                    Vec::new(),
                    Vec::new(),
                    None,
                ),
                back_edge,
                end(24),
            ],
            vec![u32::MAX, u32::MAX],
        );

        evaluate_shader_within(
            &TestMemory::default(),
            &shader,
            EvaluationOptions::default(),
            24,
        )
        .expect("a vector compare must re-taint its implicit lane mask destination");
    }
}

#[test]
fn vcmpx_does_not_retaint_vcc_or_an_sdwa_scalar_destination() {
    for (mask_register, destinations, control) in [
        (106, Vec::new(), None),
        (
            20,
            vec![Operand::scalar(20)],
            Some(InstructionControl::Sdwa(crate::SdwaControl {
                destination_select: 6,
                destination_unused: 0,
                source0_select: 6,
                source1_select: 6,
                source0_sign_extend: false,
                source1_sign_extend: false,
                absolute_mask: 0,
                negate_mask: 0,
                output_modifier: 0,
                clamp: false,
                scalar_destination: Some(20),
            })),
        ),
    ] {
        let exit = relative_branch(12, "SCbranchScc0", 2);
        let back_edge = relative_branch(20, "SBranch", -6);
        let shader = state(
            vec![
                instruction(
                    0,
                    ShaderEncoding::Sop1,
                    "SMovB64",
                    vec![Operand::source(128, None)],
                    vec![Operand::scalar(mask_register)],
                    None,
                ),
                instruction(
                    4,
                    ShaderEncoding::Vopc,
                    "VCmpxEqF32",
                    vec![Operand::vector(0), Operand::vector(1)],
                    destinations,
                    control,
                ),
                instruction(
                    8,
                    ShaderEncoding::Sop2,
                    "SOrB64",
                    vec![Operand::scalar(0), Operand::scalar(mask_register)],
                    vec![Operand::scalar(0)],
                    None,
                ),
                exit,
                instruction(
                    16,
                    ShaderEncoding::Sopp,
                    "SNop",
                    Vec::new(),
                    Vec::new(),
                    None,
                ),
                back_edge,
                end(24),
            ],
            vec![u32::MAX, u32::MAX],
        );

        let error = evaluate_shader_within(
            &TestMemory::default(),
            &shader,
            EvaluationOptions::default(),
            24,
        )
        .expect_err("VCMPX must not taint a non-EXEC mask destination");

        assert!(matches!(error, EvaluationError::ControlFlowLimit { .. }));
    }
}

#[test]
fn ordinary_vop3_compare_retaints_default_vcc() {
    let exit = relative_branch(12, "SCbranchScc0", 2);
    let back_edge = relative_branch(20, "SBranch", -6);
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sop1,
                "SMovB64",
                vec![Operand::source(128, None)],
                vec![Operand::scalar(106)],
                None,
            ),
            instruction(
                4,
                ShaderEncoding::Vop3,
                "VCmpEqF32",
                vec![Operand::vector(0), Operand::vector(1)],
                vec![Operand::vector(2)],
                Some(InstructionControl::Vop3(crate::Vop3Control {
                    absolute_mask: 0,
                    negate_mask: 0,
                    output_modifier: 0,
                    clamp: false,
                    operand_select: 0,
                    scalar_destination: None,
                })),
            ),
            instruction(
                8,
                ShaderEncoding::Sop2,
                "SOrB64",
                vec![Operand::scalar(0), Operand::scalar(106)],
                vec![Operand::scalar(0)],
                None,
            ),
            exit,
            instruction(
                16,
                ShaderEncoding::Sopp,
                "SNop",
                Vec::new(),
                Vec::new(),
                None,
            ),
            back_edge,
            end(24),
        ],
        vec![u32::MAX, u32::MAX],
    );

    evaluate_shader_within(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
        24,
    )
    .expect("an ordinary VOP3 compare must taint its implicit default VCC destination");
}

#[test]
fn saveexec_destination_retains_only_the_old_exec_taint() {
    let exit = relative_branch(16, "SCbranchScc1", 2);
    let back_edge = relative_branch(24, "SBranch", -7);
    let mut registers = vec![u32::MAX; 22];
    registers[4] = 0;
    registers[5] = 0;
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sop1,
                "SMovB64",
                vec![Operand::source(128, None)],
                vec![Operand::scalar(126)],
                None,
            ),
            instruction(
                4,
                ShaderEncoding::Vop3,
                "VDivScaleF32",
                vec![Operand::vector(0), Operand::vector(1), Operand::vector(2)],
                vec![Operand::vector(3)],
                Some(InstructionControl::Vop3(crate::Vop3Control {
                    absolute_mask: 0,
                    negate_mask: 0,
                    output_modifier: 0,
                    clamp: false,
                    operand_select: 0,
                    scalar_destination: Some(20),
                })),
            ),
            instruction(
                8,
                ShaderEncoding::Sop1,
                "SOrSaveexecB64",
                vec![Operand::scalar(20)],
                vec![Operand::scalar(4)],
                None,
            ),
            instruction(
                12,
                ShaderEncoding::Sop2,
                "SAndB64",
                vec![Operand::scalar(4), Operand::scalar(4)],
                vec![Operand::scalar(4)],
                None,
            ),
            exit,
            instruction(
                20,
                ShaderEncoding::Sopp,
                "SNop",
                Vec::new(),
                Vec::new(),
                None,
            ),
            back_edge,
            end(28),
        ],
        registers,
    );

    let error = evaluate_shader_within(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
        28,
    )
    .expect_err(
        "saved old EXEC must remain untainted when only the new EXEC source is vector-tainted",
    );

    assert!(matches!(error, EvaluationError::ControlFlowLimit { .. }));
}

#[test]
fn saveexec_b32_ignores_exec_hi_taint() {
    let exit = relative_branch(16, "SCbranchScc1", 2);
    let back_edge = relative_branch(24, "SBranch", -7);
    let mut registers = vec![0; 21];
    registers[20] = u32::MAX;
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sop1,
                "SMovB64",
                vec![Operand::source(128, None)],
                vec![Operand::scalar(126)],
                None,
            ),
            instruction(
                4,
                ShaderEncoding::Vop3,
                "VReadlaneB32",
                vec![Operand::vector(0), Operand::source(128, None)],
                vec![Operand::scalar(127)],
                Some(InstructionControl::Vop3(crate::Vop3Control {
                    absolute_mask: 0,
                    negate_mask: 0,
                    output_modifier: 0,
                    clamp: false,
                    operand_select: 0,
                    scalar_destination: None,
                })),
            ),
            instruction(
                8,
                ShaderEncoding::Sop1,
                "SOrSaveexecB32",
                vec![Operand::scalar(20)],
                vec![Operand::scalar(4)],
                None,
            ),
            instruction(
                12,
                ShaderEncoding::Sop2,
                "SAndB32",
                vec![Operand::scalar(4), Operand::scalar(4)],
                vec![Operand::scalar(4)],
                None,
            ),
            exit,
            instruction(
                20,
                ShaderEncoding::Sopp,
                "SNop",
                Vec::new(),
                Vec::new(),
                None,
            ),
            back_edge,
            end(28),
        ],
        registers,
    );

    let error = evaluate_shader_within(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
        28,
    )
    .expect_err("SaveexecB32 must ignore EXEC_HI when deriving destination, EXEC_LO and SCC taint");

    assert!(matches!(error, EvaluationError::ControlFlowLimit { .. }));
}

#[test]
fn taints_both_halves_of_a_bitreplicate_destination() {
    let exit = relative_branch(8, "SCbranchScc0", 2);
    let back_edge = relative_branch(16, "SBranch", -5);
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sop1,
                "SBitreplicateB64B32",
                vec![Operand::scalar(126)],
                vec![Operand::scalar(0)],
                None,
            ),
            instruction(
                4,
                ShaderEncoding::Sop2,
                "SAndB32",
                vec![Operand::scalar(1), Operand::scalar(1)],
                vec![Operand::scalar(2)],
                None,
            ),
            exit,
            instruction(
                12,
                ShaderEncoding::Sopp,
                "SNop",
                Vec::new(),
                Vec::new(),
                None,
            ),
            back_edge,
            end(20),
        ],
        Vec::new(),
    );

    evaluate_shader_within(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
        20,
    )
    .expect("the high half of a mask-derived bitreplicate remains mask-derived");
}

#[test]
fn propagates_a_mask_derived_scalar_memory_address_to_the_loaded_value() {
    let pointer = 0x1000_u64;
    let mut memory = TestMemory::default();
    memory.write(pointer, &1_u32.to_le_bytes());
    let exit = relative_branch(12, "SCbranchScc0", 2);
    let back_edge = relative_branch(20, "SBranch", -6);
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sop2,
                "SAndB64",
                vec![Operand::scalar(0), Operand::scalar(126)],
                vec![Operand::scalar(0)],
                None,
            ),
            instruction(
                4,
                ShaderEncoding::Smem,
                "SLoadDword",
                vec![Operand::scalar(0)],
                vec![Operand::scalar(4)],
                Some(InstructionControl::ScalarMemory(ScalarMemoryControl {
                    destination_count: 1,
                    immediate_offset_bytes: 0,
                    dynamic_offset_register: None,
                })),
            ),
            instruction(
                8,
                ShaderEncoding::Sop2,
                "SAndB32",
                vec![Operand::scalar(4), Operand::scalar(4)],
                vec![Operand::scalar(4)],
                None,
            ),
            exit,
            instruction(
                16,
                ShaderEncoding::Sopp,
                "SNop",
                Vec::new(),
                Vec::new(),
                None,
            ),
            back_edge,
            end(24),
        ],
        vec![pointer as u32, (pointer >> 32) as u32],
    );

    evaluate_shader_within(&memory, &shader, EvaluationOptions::default(), 28)
        .expect("a load selected by a mask-derived address must keep the loop vector-controlled");
}

#[test]
fn does_not_taint_the_register_beside_a_single_dword_vector_scalar_destination() {
    let exit = relative_branch(8, "SCbranchScc0", 2);
    let back_edge = relative_branch(16, "SBranch", -5);
    let mut registers = vec![u32::MAX];
    registers.resize(22, u32::MAX);
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Vop3,
                "VReadlaneB32",
                vec![Operand::vector(0), Operand::source(128, None)],
                vec![Operand::scalar(20)],
                Some(InstructionControl::Vop3(crate::Vop3Control {
                    absolute_mask: 0,
                    negate_mask: 0,
                    output_modifier: 0,
                    clamp: false,
                    operand_select: 0,
                    scalar_destination: None,
                })),
            ),
            instruction(
                4,
                ShaderEncoding::Sop2,
                "SAndB32",
                vec![Operand::scalar(21), Operand::scalar(21)],
                vec![Operand::scalar(0)],
                None,
            ),
            exit,
            instruction(
                12,
                ShaderEncoding::Sopp,
                "SNop",
                Vec::new(),
                Vec::new(),
                None,
            ),
            back_edge,
            end(20),
        ],
        registers,
    );

    let error = evaluate_shader_within(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
        20,
    )
    .expect_err("V_READLANE_B32 must not taint the neighboring scalar register");

    assert!(matches!(error, EvaluationError::ControlFlowLimit { .. }));
}

#[test]
fn follows_scc_taint_across_a_branch_target() {
    let skip_unreachable_scalar_definition = relative_branch(4, "SBranch", 1);
    let exit = relative_branch(12, "SCbranchScc0", 2);
    let back_edge = relative_branch(20, "SBranch", -6);
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sop2,
                "SAndn2B64",
                vec![Operand::scalar(0), Operand::scalar(126)],
                vec![Operand::scalar(0)],
                None,
            ),
            skip_unreachable_scalar_definition,
            instruction(
                8,
                ShaderEncoding::Sop2,
                "SAndB64",
                vec![Operand::scalar(2), Operand::scalar(2)],
                vec![Operand::scalar(2)],
                None,
            ),
            exit,
            instruction(
                16,
                ShaderEncoding::Sopp,
                "SNop",
                Vec::new(),
                Vec::new(),
                None,
            ),
            back_edge,
            end(24),
        ],
        vec![u32::MAX, u32::MAX, 0, 0],
    );

    evaluate_shader_within(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
        20,
    )
    .expect("the SCC definition reaching through a branch target remains mask-derived");
}

#[test]
fn merges_mask_taint_from_one_predecessor_at_an_scc_join() {
    let select_scalar_arm = relative_branch(0, "SCbranchExecz", 2);
    let join = relative_branch(8, "SBranch", 1);
    let exit = relative_branch(16, "SCbranchScc0", 2);
    let back_edge = relative_branch(24, "SBranch", -7);
    let shader = state(
        vec![
            select_scalar_arm,
            instruction(
                4,
                ShaderEncoding::Sop2,
                "SAndn2B64",
                vec![Operand::scalar(0), Operand::scalar(126)],
                vec![Operand::scalar(0)],
                None,
            ),
            join,
            instruction(
                12,
                ShaderEncoding::Sop2,
                "SAndB64",
                vec![Operand::scalar(2), Operand::scalar(2)],
                vec![Operand::scalar(2)],
                None,
            ),
            exit,
            instruction(
                20,
                ShaderEncoding::Sopp,
                "SNop",
                Vec::new(),
                Vec::new(),
                None,
            ),
            back_edge,
            end(28),
        ],
        vec![u32::MAX, u32::MAX, 0, 0],
    );

    evaluate_shader_within(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
        24,
    )
    .expect("one mask-derived predecessor makes the SCC join vector-controlled");
}

#[test]
fn ignores_an_unreachable_mask_derived_scc_definition_before_a_branch_target() {
    let skip_unreachable_mask_definition = relative_branch(4, "SBranch", 1);
    let exit = relative_branch(12, "SCbranchScc0", 2);
    let back_edge = relative_branch(20, "SBranch", -6);
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sop2,
                "SAndB64",
                vec![Operand::scalar(0), Operand::scalar(0)],
                vec![Operand::scalar(0)],
                None,
            ),
            skip_unreachable_mask_definition,
            instruction(
                8,
                ShaderEncoding::Sop2,
                "SAndn2B64",
                vec![Operand::scalar(2), Operand::scalar(126)],
                vec![Operand::scalar(2)],
                None,
            ),
            exit,
            instruction(
                16,
                ShaderEncoding::Sopp,
                "SNop",
                Vec::new(),
                Vec::new(),
                None,
            ),
            back_edge,
            end(24),
        ],
        vec![u32::MAX, u32::MAX, u32::MAX, u32::MAX],
    );

    let error = evaluate_shader_within(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
        20,
    )
    .expect_err("an unreachable mask definition must not taint the scalar SCC loop");

    assert!(matches!(error, EvaluationError::ControlFlowLimit { .. }));
}

#[test]
fn keeps_a_b64_shift_with_a_scalar_s125_count_exact() {
    let exit = relative_branch(4, "SCbranchScc0", 2);
    let back_edge = relative_branch(12, "SBranch", -4);
    let mut registers = vec![u32::MAX, u32::MAX];
    registers.resize(126, 0);
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sop2,
                "SLshlB64",
                vec![Operand::scalar(0), Operand::scalar(125)],
                vec![Operand::scalar(0)],
                None,
            ),
            exit,
            instruction(
                8,
                ShaderEncoding::Sopp,
                "SNop",
                Vec::new(),
                Vec::new(),
                None,
            ),
            back_edge,
            end(16),
        ],
        registers,
    );

    let error = evaluate_shader_within(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
        16,
    )
    .expect_err("a 32-bit shift count beside EXEC must not taint the scalar SCC loop");

    assert!(matches!(error, EvaluationError::ControlFlowLimit { .. }));
}

#[test]
fn keeps_b64_bitmask_controls_beside_exec_exact() {
    let exit = relative_branch(8, "SCbranchScc0", 2);
    let back_edge = relative_branch(16, "SBranch", -5);
    let mut registers = vec![0, 0, u32::MAX, u32::MAX];
    registers.resize(126, 0);
    registers[124] = 1;
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sop2,
                "SBfmB64",
                vec![Operand::scalar(124), Operand::scalar(125)],
                vec![Operand::scalar(0)],
                None,
            ),
            instruction(
                4,
                ShaderEncoding::Sop2,
                "SAndB64",
                vec![Operand::scalar(0), Operand::scalar(2)],
                vec![Operand::scalar(2)],
                None,
            ),
            exit,
            instruction(
                12,
                ShaderEncoding::Sopp,
                "SNop",
                Vec::new(),
                Vec::new(),
                None,
            ),
            back_edge,
            end(20),
        ],
        registers,
    );

    let error = evaluate_shader_within(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
        20,
    )
    .expect_err("S_BFM_B64 controls beside EXEC must not taint the scalar SCC loop");

    assert!(matches!(error, EvaluationError::ControlFlowLimit { .. }));
}

#[test]
fn linked_setpc_preserves_scalar_state_and_continuation_absolute_pc() {
    let front = Program::new(
        0x1000,
        vec![
            instruction(
                0,
                ShaderEncoding::Sop1,
                "SMovB32",
                vec![Operand::scalar(0)],
                vec![Operand::scalar(20)],
                None,
            ),
            instruction(
                4,
                ShaderEncoding::Sop1,
                "SSetpcB64",
                vec![Operand::scalar(6)],
                Vec::new(),
                None,
            ),
        ],
    );
    let continuation = Program::new(
        0x800,
        vec![
            instruction(
                0,
                ShaderEncoding::Sop1,
                "SGetpcB64",
                Vec::new(),
                vec![Operand::scalar(30)],
                None,
            ),
            instruction(
                4,
                ShaderEncoding::Sop2,
                "SAddU32",
                vec![Operand::scalar(20), Operand::scalar(30)],
                vec![Operand::scalar(40)],
                None,
            ),
            end(8),
        ],
    );
    let shader = ShaderState {
        program: Arc::new(
            crate::link_program_continuation(&front, &continuation)
                .expect("exact linked continuation"),
        ),
        user_data: vec![7].into(),
        compute_system_registers: None,
        user_data_scalar_register_base: 0,
    };

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("linked scalar control flow");

    assert_eq!(evaluation.scalar_registers[20], 7);
    assert_eq!(evaluation.scalar_registers[30], 0x804);
    assert_eq!(evaluation.scalar_registers[31], 0);
    assert_eq!(evaluation.scalar_registers[40], 0x80b);
}

#[test]
fn linked_program_resources_keep_distinct_front_and_continuation_pcs() {
    let image = ImageControl {
        dmask: 0xf,
        vector_address: 0,
        address_registers: vec![0, 1],
        vector_data: 4,
        scalar_resource: 0,
        scalar_sampler: 8,
        dimension: 1,
        is_array: false,
        glc: false,
        slc: false,
        a16: false,
        d16: false,
        unorm: false,
    };
    let image_instruction = |pc| {
        instruction(
            pc,
            ShaderEncoding::Mimg,
            "ImageSampleLz",
            Vec::new(),
            Vec::new(),
            Some(InstructionControl::Image(image.clone())),
        )
    };
    let front = Program::new(
        0x1000,
        vec![
            image_instruction(0),
            instruction(
                4,
                ShaderEncoding::Sop1,
                "SSetpcB64",
                vec![Operand::scalar(6)],
                Vec::new(),
                None,
            ),
        ],
    );
    let continuation = Program::new(0x800, vec![image_instruction(0), end(4)]);
    let shader = ShaderState {
        program: Arc::new(
            crate::link_program_continuation(&front, &continuation)
                .expect("exact linked continuation"),
        ),
        user_data: (1..=12).collect::<Vec<_>>().into(),
        compute_system_registers: None,
        user_data_scalar_register_base: 0,
    };

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("both linked resource sites should evaluate");

    assert_eq!(
        evaluation
            .image_bindings
            .iter()
            .map(|binding| binding.pc)
            .collect::<Vec<_>>(),
        vec![0, 8]
    );
}

/// The control-flow shape of the Poppy Playtime compute shader at 0x77C0F10000: nested
/// vector-condition branches wrapped around a vector-controlled loop whose body advances a live
/// scalar induction register, so no two loop iterations share a memoization key.
fn nested_vector_loop_program() -> Vec<Instruction> {
    let branch = |pc: u32, opcode: &'static str, target: u32| {
        let mut branch = instruction(
            pc,
            ShaderEncoding::Sopp,
            opcode,
            Vec::new(),
            Vec::new(),
            None,
        );
        let offset = (i64::from(target) - i64::from(pc) - 4) / 4;
        let offset = i16::try_from(offset).expect("branch offset");
        branch.words[0] = u32::from(offset.cast_unsigned());
        branch
    };
    let move_zero = |pc: u32, destination: u32| {
        instruction(
            pc,
            ShaderEncoding::Sop1,
            "SMovB32",
            vec![Operand::source(128, None)],
            vec![Operand::scalar(destination)],
            None,
        )
    };
    let move_one = |pc: u32, destination: u32| {
        instruction(
            pc,
            ShaderEncoding::Sopk,
            "SMovkI32",
            vec![Operand {
                kind: OperandKind::LiteralConstant,
                value: 1,
            }],
            vec![Operand::scalar(destination)],
            None,
        )
    };
    let compare = |pc: u32, register: u32| {
        instruction(
            pc,
            ShaderEncoding::Sopk,
            "SCmpkEqI32",
            vec![Operand {
                kind: OperandKind::EncodedConstant,
                value: 1,
            }],
            vec![Operand::scalar(register)],
            None,
        )
    };
    let scalar_load = |pc: u32, base: u32, dynamic_offset_register: Option<u32>| {
        instruction(
            pc,
            ShaderEncoding::Smem,
            "SLoadDwordx2",
            vec![Operand::scalar(base)],
            vec![Operand::scalar(16), Operand::scalar(17)],
            Some(InstructionControl::ScalarMemory(ScalarMemoryControl {
                destination_count: 2,
                immediate_offset_bytes: 0,
                dynamic_offset_register,
            })),
        )
    };
    vec![
        move_zero(0x30, 12),
        branch(0x34, "SCbranchScc1", 0x76c),
        compare(0x54, 13),
        branch(0x58, "SCbranchScc0", 0x750),
        branch(0x70, "SCbranchExecz", 0xb8),
        move_zero(0x74, 14),
        move_one(0xb8, 15),
        compare(0xc0, 15),
        branch(0xc4, "SCbranchScc0", 0x11c),
        instruction(
            0xc8,
            ShaderEncoding::Sop1,
            "SMovB64",
            vec![Operand::source(128, None)],
            vec![Operand::scalar(126)],
            None,
        ),
        branch(0xcc, "SCbranchExecz", 0x114),
        scalar_load(0xd0, 4, Some(12)),
        branch(0xe8, "SCbranchVccz", 0x114),
        instruction(
            0xec,
            ShaderEncoding::Sop2,
            "SAddU32",
            vec![
                Operand::scalar(12),
                Operand {
                    kind: OperandKind::LiteralConstant,
                    value: 8,
                },
            ],
            vec![Operand::scalar(12)],
            None,
        ),
        branch(0x110, "SCbranchExecnz", 0xcc),
        move_zero(0x114, 20),
        branch(0x118, "SBranch", 0x71c),
        move_zero(0x11c, 21),
        move_zero(0x71c, 22),
        branch(0x74c, "SBranch", 0x76c),
        move_zero(0x750, 23),
        move_zero(0x76c, 24),
        branch(0x770, "SCbranchExecz", 0x798),
        scalar_load(0x774, 8, None),
        end(0x798),
    ]
}

#[test]
fn nested_vector_branches_around_a_loop_stay_within_the_evaluation_budget() {
    let mut memory = TestMemory::default();
    memory.write(0x2000, &[0xab; 512]);
    memory.write(0x4000, &[0xcd; 512]);
    let mut user_data = vec![0_u32; 14];
    user_data[4] = 0x2000;
    user_data[8] = 0x4000;
    user_data[13] = 1;
    let shader = state(nested_vector_loop_program(), user_data);
    let loop_header = shader
        .program
        .instructions
        .binary_search_by_key(&0xcc, |instruction| instruction.pc)
        .expect("Poppy-derived vector loop header");
    let recurrence_registers = shader
        .program
        .evaluation_live_scalar_register_indices(loop_header);
    assert!(
        recurrence_registers.contains(&4),
        "SMEM base identity must remain observable"
    );
    assert!(
        !recurrence_registers.contains(&12),
        "dead SMEM induction offset must not prevent convergence"
    );

    let evaluation = evaluate_shader_within(&memory, &shader, EvaluationOptions::default(), 4096)
        .expect("nested vector control flow must terminate within a small step budget");

    let bindings = evaluation
        .global_memory_bindings
        .iter()
        .map(|binding| (binding.scalar_address, binding.base_address))
        .collect::<Vec<_>>();
    assert!(
        bindings.contains(&(4, 0x2000)),
        "loop-body binding missing: {bindings:?}"
    );
    assert!(
        bindings.contains(&(8, 0x4000)),
        "tail binding missing: {bindings:?}"
    );
}

#[test]
fn s_call_enters_the_subroutine_and_the_indirect_return_resumes_the_call_site() {
    // A real encoded program: main body calls a subroutine placed after its endpgm; the
    // subroutine returns through S_SETPC_B64 on the register-known return address. Before this
    // support every call abandoned the path as the "subroutine-call" truncation class.
    let program = crate::decode_program(
        0x1000,
        &[
            0xb000_0005, // s_movk_i32 s0, 5
            0xbb0a_0002, // s_call_b64 s[10:11], +2 (subroutine at pc 16)
            0xb001_0009, // s_movk_i32 s1, 9 (return continuation)
            0xbf81_0000, // s_endpgm
            0xb000_002a, // s_movk_i32 s0, 42 (subroutine body)
            0xbe80_200a, // s_setpc_b64 s[10:11] (return)
        ],
    )
    .expect("call/return program decodes");
    let shader = ShaderState {
        program: Arc::new(program),
        user_data: Vec::new().into(),
        compute_system_registers: None,
        user_data_scalar_register_base: 0,
    };

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("call/return evaluation");

    assert_eq!(
        evaluation.scalar_registers[0], 42,
        "the subroutine body ran"
    );
    assert_eq!(
        evaluation.scalar_registers[1], 9,
        "the return resumed at the call continuation"
    );
    assert_eq!(
        evaluation.scalar_registers[10], 0x1008,
        "the call linked the architectural return address"
    );
    assert_eq!(evaluation.scalar_registers[11], 0);
}

#[test]
fn an_unresolvable_indirect_transfer_still_ends_the_path_quietly() {
    // S_SETPC_B64 on a register value that names no instruction in the program keeps its
    // truncation behavior: the path ends, prior discovery survives, and no error is fabricated.
    let program = crate::decode_program(
        0x1000,
        &[
            0xb000_0007, // s_movk_i32 s0, 7
            0xbe80_200a, // s_setpc_b64 s[10:11] (s[10:11] holds 0 -> unresolvable)
            0xb000_0063, // s_movk_i32 s0, 99 (unreachable)
            0xbf81_0000, // s_endpgm
        ],
    )
    .expect("indirect transfer program decodes");
    let shader = ShaderState {
        program: Arc::new(program),
        user_data: Vec::new().into(),
        compute_system_registers: None,
        user_data_scalar_register_base: 0,
    };

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("unresolvable transfer evaluation");

    assert_eq!(
        evaluation.scalar_registers[0], 7,
        "state before the abandoned transfer is preserved"
    );
}
