// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Control-flow evaluation tests: statically inactive blocks, bounded backward
//! edges and the vector-controlled loops scalar discovery must terminate on.

use super::*;

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
fn exits_an_unconditional_back_edge_after_discovering_the_vector_loop_once() {
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
        0x4000,
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
    assert_eq!(evaluation.scalar_registers[30], 0x4004);
    assert_eq!(evaluation.scalar_registers[31], 0);
    assert_eq!(evaluation.scalar_registers[40], 0x400b);
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
    let continuation = Program::new(0x4000, vec![image_instruction(0), end(4)]);
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
        vec![0, 0x3000]
    );
}
