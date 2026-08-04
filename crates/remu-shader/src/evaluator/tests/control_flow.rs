// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Control-flow evaluation tests: statically inactive blocks, bounded backward
//! edges and the vector-controlled loops scalar discovery must terminate on.

use super::*;
use crate::ComputeSystemRegisters;

#[test]
fn structural_rescue_revisits_states_seen_during_exact_evaluation() {
    let mut visited = VisitedPaths::new();

    assert!(visited.insert((0x40, 0x1234, false)));
    assert!(!visited.insert((0x40, 0x1234, false)));
    assert!(visited.insert((0x40, 0x1234, true)));
    assert!(!visited.insert((0x40, 0x1234, true)));
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
    let mut shader = state(
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
    shader.compute_system_registers = Some(ComputeSystemRegisters::default());

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("vector-controlled runtime loop must not exhaust static discovery");
    assert!(evaluation.global_memory_bindings.is_empty());
    assert!(
        evaluation
            .compute_system_registers
            .expect("compute loop hints")
            .has_evaluated_vector_controlled_loop
    );
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
        structural_rescue: false,
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
    let mut shader = state(
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
    shader.compute_system_registers = Some(ComputeSystemRegisters::default());

    let evaluation = evaluate_shader_within(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
        20_000,
    )
    .expect("scalar counted loop should run to its SCC exit");

    assert_eq!(evaluation.scalar_registers[107], 16_388);
    assert_eq!(
        evaluation
            .compute_system_registers
            .expect("compute loop hints")
            .maximum_evaluated_scalar_loop_iterations,
        4_097
    );
}

#[test]
fn compute_resource_discovery_degrades_to_a_finite_loop_hint_at_its_step_limit() {
    let mut back_edge = instruction(
        4,
        ShaderEncoding::Sopp,
        "SBranch",
        Vec::new(),
        Vec::new(),
        None,
    );
    back_edge.words[0] = u32::from((-2_i16).cast_unsigned());
    let mut shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sop2,
                "SAddU32",
                vec![Operand::scalar(20), Operand::source(132, None)],
                vec![Operand::scalar(20)],
                None,
            ),
            back_edge,
            end(8),
        ],
        vec![0; 21],
    );
    shader.compute_system_registers = Some(ComputeSystemRegisters::default());

    let evaluation = evaluate_shader_within(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
        16,
    )
    .expect("compute discovery should defer termination to the HLSL watchdog");

    assert_eq!(
        evaluation
            .compute_system_registers
            .expect("compute loop hints")
            .maximum_evaluated_scalar_loop_iterations,
        u32::MAX
    );
}

#[test]
fn compute_structural_pass_recovers_descriptors_from_an_inactive_scc_arm() {
    let mut skip_inactive_arm = instruction(
        4,
        ShaderEncoding::Sopp,
        "SCbranchScc1",
        Vec::new(),
        Vec::new(),
        None,
    );
    skip_inactive_arm.words[0] = 6; // pc 4 + 4 + 6*4 = pc 32
    let scalar_load = |pc, destination, count, offset| {
        instruction(
            pc,
            ShaderEncoding::Smem,
            if count == 8 {
                "SLoadDwordx8"
            } else {
                "SLoadDwordx4"
            },
            vec![Operand::scalar(0)],
            (destination..destination + count)
                .map(Operand::scalar)
                .collect(),
            Some(InstructionControl::ScalarMemory(ScalarMemoryControl {
                destination_count: count,
                immediate_offset_bytes: offset,
                dynamic_offset_register: None,
            })),
        )
    };
    let image = ImageControl {
        dmask: 0xf,
        vector_address: 0,
        address_registers: vec![0, 1],
        vector_data: 4,
        scalar_resource: 16,
        scalar_sampler: 28,
        dimension: 1,
        is_array: false,
        glc: false,
        slc: false,
        a16: false,
        d16: false,
        unorm: false,
    };
    let mut user_data = vec![0_u32; 21];
    user_data[0] = 0x2000;
    user_data[20] = 1;
    let mut shader = state(
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
            skip_inactive_arm,
            scalar_load(8, 16, 8, 0),
            scalar_load(16, 28, 4, 32),
            instruction(
                24,
                ShaderEncoding::Mimg,
                "ImageSampleLz",
                Vec::new(),
                Vec::new(),
                Some(InstructionControl::Image(image)),
            ),
            end(32),
        ],
        user_data,
    );
    shader.compute_system_registers = Some(ComputeSystemRegisters::default());
    let mut memory = TestMemory::default();
    let descriptors = (1_u32..=12).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
    memory.write(0x2000, &descriptors);

    let evaluation = evaluate_shader(&memory, &shader, EvaluationOptions::default())
        .expect("structural pass should enter the inactive SCC arm");

    assert_eq!(evaluation.image_bindings.len(), 1);
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
fn compute_step_limit_rescue_executes_descriptor_loads_after_a_long_scalar_loop() {
    let mut exit = instruction(
        4,
        ShaderEncoding::Sopp,
        "SCbranchScc0",
        Vec::new(),
        Vec::new(),
        None,
    );
    exit.words[0] = 3; // pc 4 + 4 + 3*4 = pc 20
    let mut back_edge = instruction(
        12,
        ShaderEncoding::Sopp,
        "SBranch",
        Vec::new(),
        Vec::new(),
        None,
    );
    back_edge.words[0] = u32::from((-4_i16).cast_unsigned());
    let scalar_load = |pc, destination, count, offset| {
        instruction(
            pc,
            ShaderEncoding::Smem,
            if count == 8 {
                "SLoadDwordx8"
            } else {
                "SLoadDwordx4"
            },
            vec![Operand::scalar(0)],
            (destination..destination + count)
                .map(Operand::scalar)
                .collect(),
            Some(InstructionControl::ScalarMemory(ScalarMemoryControl {
                destination_count: count,
                immediate_offset_bytes: offset,
                dynamic_offset_register: None,
            })),
        )
    };
    let image = ImageControl {
        dmask: 0xf,
        vector_address: 0,
        address_registers: vec![0, 1],
        vector_data: 4,
        scalar_resource: 16,
        scalar_sampler: 28,
        dimension: 1,
        is_array: false,
        glc: false,
        slc: false,
        a16: false,
        d16: false,
        unorm: false,
    };
    let mut user_data = vec![0_u32; 22];
    user_data[0] = 0x2000;
    user_data[21] = 100;
    let mut shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sopc,
                "SCmpLtU32",
                vec![Operand::scalar(20), Operand::scalar(21)],
                Vec::new(),
                None,
            ),
            exit,
            instruction(
                8,
                ShaderEncoding::Sop2,
                "SAddU32",
                vec![
                    Operand::scalar(20),
                    Operand {
                        kind: OperandKind::LiteralConstant,
                        value: 1,
                    },
                ],
                vec![Operand::scalar(20)],
                None,
            ),
            back_edge,
            scalar_load(20, 16, 8, 0),
            scalar_load(28, 28, 4, 32),
            instruction(
                36,
                ShaderEncoding::Mimg,
                "ImageSampleLz",
                Vec::new(),
                Vec::new(),
                Some(InstructionControl::Image(image)),
            ),
            end(44),
        ],
        user_data,
    );
    shader.compute_system_registers = Some(ComputeSystemRegisters::default());
    let mut memory = TestMemory::default();
    let descriptors = (1_u32..=12).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
    memory.write(0x2000, &descriptors);

    let evaluation = evaluate_shader_within(&memory, &shader, EvaluationOptions::default(), 6)
        .expect("compute rescue should walk descriptor loads after the scalar loop");

    assert_eq!(evaluation.image_bindings.len(), 1);
    assert_eq!(
        evaluation.image_bindings[0].resource_descriptor.as_slice(),
        (1..=8).collect::<Vec<_>>().as_slice()
    );
    assert_eq!(
        evaluation.image_bindings[0].sampler_descriptor.as_slice(),
        (9..=12).collect::<Vec<_>>().as_slice()
    );
    assert_eq!(
        evaluation
            .compute_system_registers
            .expect("compute loop hints")
            .maximum_evaluated_scalar_loop_iterations,
        u32::MAX
    );
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
            0xba0a_0002, // s_call_b64 s[10:11], +2 (subroutine at pc 16)
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
