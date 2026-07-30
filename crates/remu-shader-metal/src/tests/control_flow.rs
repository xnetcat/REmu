// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Control-flow lowering tests: straight-line direct emission, forward-only basic-block chains,
//! the bounded loop dispatcher, branch validation and codegen footprint bounds.

use super::*;

#[test]
fn directly_lowers_single_path_control_flow_and_scalarizes_live_registers() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::Known("SMovB32"),
                words: vec![0],
                sources: vec![Operand::scalar(12)],
                destinations: vec![Operand::scalar(80)],
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VMovB32"),
                words: vec![0],
                sources: vec![Operand::scalar(80)],
                destinations: vec![Operand::vector(200)],
                control: None,
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0xbf81_0000],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(0),
    )
    .expect("single-path program should lower directly");

    assert!(shader.register_footprint.direct_control_flow);
    assert_eq!(shader.register_footprint.scalar_values, 2);
    assert_eq!(shader.register_footprint.vector_values, 1);
    assert!(
        shader
            .source
            .contains("uint remu_s_12 = remu_initial_s[12];")
    );
    assert!(shader.source.contains("uint remu_v_200 = 0u;"));
    assert!(!shader.source.contains("uint s[128]"));
    assert!(!shader.source.contains("uint v[256]"));
    assert!(!shader.source.contains("while (active)"));
    assert!(!shader.source.contains("switch (pc)"));
}

#[test]
fn resolves_acyclic_unconditional_branches_without_lowering_dead_instructions() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::Known("SBranch"),
                // PC 0 + 4-byte instruction + one dword = PC 8.
                words: vec![1],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VMovB32"),
                words: vec![0],
                sources: vec![Operand::vector(99)],
                destinations: vec![Operand::vector(100)],
                control: None,
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0xbf81_0000],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(0),
    )
    .expect("static unconditional path should lower");

    assert!(shader.register_footprint.direct_control_flow);
    assert_eq!(shader.register_footprint.vector_values, 0);
    assert!(!shader.source.contains("remu_v_99"));
    assert!(!shader.source.contains("switch (pc)"));
}

fn setpc(pc: u32) -> Instruction {
    Instruction {
        pc,
        encoding: ShaderEncoding::Sop1,
        opcode: Opcode::Known("SSetpcB64"),
        words: vec![0xbe80_2006],
        sources: vec![Operand::scalar(6)],
        destinations: Vec::new(),
        control: None,
    }
}

#[test]
fn setpc_terminates_direct_lowering_without_emitting_a_dynamic_pc_operation() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VMovB32"),
                words: vec![0],
                sources: vec![Operand::vector(1)],
                destinations: vec![Operand::vector(2)],
                control: None,
            },
            setpc(4),
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::Known("SSwappcB64"),
                words: vec![0xbe80_2106],
                sources: vec![Operand::scalar(6)],
                destinations: vec![Operand::scalar(0)],
                control: None,
            },
        ],
    );

    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(0),
    )
    .expect("terminal setpc must lower without reaching trailing allocation contents");

    assert!(shader.register_footprint.direct_control_flow);
    assert!(!shader.source.contains("while (active)"));
    assert!(!shader.source.contains("switch (pc)"));
}

#[test]
fn setpc_deactivates_each_forward_control_flow_path() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::Known("SCbranchScc1"),
                words: vec![1],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
            setpc(4),
            setpc(8),
        ],
    );

    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(0),
    )
    .expect("terminal setpc must lower in forward conditional blocks");

    assert!(!shader.register_footprint.direct_control_flow);
    assert_eq!(shader.source.matches("active = false;").count(), 2);
}

#[test]
fn linked_setpc_lowers_real_continuation_param_exports() {
    let front = Program::new(0x1000, vec![setpc(0)]);
    // Decode an architectural RDNA2 EXP PARAM0 rather than constructing export IR by hand.
    let export_param0 = 0xf800_0000 | (32 << 4) | 0x0f | (1 << 11) | (1 << 12);
    let continuation = decode_program(0x4000, &[export_param0, 0x0403_0201, 0xbf81_0000])
        .expect("copy program PARAM export");
    let linked = remu_shader::link_program_continuation(&front, &continuation)
        .expect("exact registered front/back pair");

    let shader = compile_msl(
        &linked,
        &empty_evaluation(),
        &MetalCompileOptions::vertex(1),
    )
    .expect("linked PARAM export should lower");

    assert!(shader.register_footprint.direct_control_flow);
    assert!(shader.source.contains(
        "remu_out.param0 = float4(as_type<float>(remu_v_1), as_type<float>(remu_v_2), as_type<float>(remu_v_3), as_type<float>(remu_v_4));"
    ));
    assert!(!shader.source.contains("active = false;"));
}

#[test]
fn linked_continuation_getpc_uses_its_original_absolute_address() {
    let front = Program::new(0x1000, vec![setpc(0)]);
    let continuation = Program::new(
        0x4000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::Known("SGetpcB64"),
                words: vec![0xbe80_1f14],
                sources: Vec::new(),
                destinations: vec![Operand::scalar(20)],
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VMovB32"),
                words: vec![0],
                sources: vec![Operand::scalar(20)],
                destinations: vec![Operand::vector(1)],
                control: None,
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0xbf81_0000],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    );
    let linked = remu_shader::link_program_continuation(&front, &continuation)
        .expect("exact registered front/back pair");

    let shader = compile_msl(
        &linked,
        &empty_evaluation(),
        &MetalCompileOptions::vertex(0),
    )
    .expect("linked GETPC should lower");

    assert!(shader.source.contains("remu_s_20 = 0x4004u;"));
}

#[test]
fn swappc_remains_a_nonterminal_unsupported_dynamic_call() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::Known("SSwappcB64"),
                words: vec![0xbe80_2106],
                sources: vec![Operand::scalar(6)],
                destinations: vec![Operand::scalar(0)],
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0xbf81_0000],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    );

    let error = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(0),
    )
    .expect_err("swappc is a dynamic call whose continuation cannot be discarded");

    assert!(error.to_string().contains("SSwappcB64"));
}

#[test]
fn forward_conditional_control_flow_avoids_the_loop_dispatcher_and_register_arrays() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::Known("SCbranchScc1"),
                words: vec![1],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0xbf81_0000],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0xbf81_0000],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(0),
    )
    .expect("forward conditional program should lower");

    assert!(!shader.register_footprint.direct_control_flow);
    assert!(shader.source.contains("if (active && pc == 0u)"));
    assert!(!shader.source.contains("while (active)"));
    assert!(!shader.source.contains("switch (pc)"));
    assert!(!shader.source.contains("uint s[128]"));
    assert!(!shader.source.contains("uint v[256]"));
}

#[test]
fn forward_chain_checks_once_per_basic_block_instead_of_once_per_instruction() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::Known("SCbranchScc1"),
                // PC 0 + 4-byte instruction + three dwords = PC 16.
                words: vec![3],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VMovB32"),
                words: vec![0],
                sources: vec![Operand::vector(1)],
                destinations: vec![Operand::vector(2)],
                control: None,
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VMovB32"),
                words: vec![0],
                sources: vec![Operand::vector(2)],
                destinations: vec![Operand::vector(3)],
                control: None,
            },
            Instruction {
                pc: 12,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::Known("SBranch"),
                // PC 12 + 4-byte instruction + one dword = PC 20.
                words: vec![1],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
            Instruction {
                pc: 16,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VMovB32"),
                words: vec![0],
                sources: vec![Operand::vector(4)],
                destinations: vec![Operand::vector(5)],
                control: None,
            },
            Instruction {
                pc: 20,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0xbf81_0000],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(0),
    )
    .expect("conditional blocks should lower");

    for pc in [0, 4, 16, 20] {
        assert!(
            shader
                .source
                .contains(&format!("if (active && pc == {pc}u)"))
        );
    }
    for pc in [8, 12] {
        assert!(
            !shader
                .source
                .contains(&format!("if (active && pc == {pc}u)"))
        );
    }
    assert_eq!(shader.source.matches("if (active && pc == ").count(), 4);
    assert!(!shader.source.contains("while (active)"));
}

#[test]
fn backward_conditional_control_flow_keeps_the_bounded_dispatcher() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VMovB32"),
                words: vec![0],
                sources: vec![Operand::vector(1)],
                destinations: vec![Operand::vector(2)],
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::Known("SCbranchScc1"),
                // PC 4 + 4-byte instruction - two dwords = PC 0.
                words: vec![0xfffe],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0xbf81_0000],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(0),
    )
    .expect("backward conditional program should retain dispatcher");

    assert!(!shader.register_footprint.direct_control_flow);
    assert!(shader.source.contains("while (active)"));
    assert!(shader.source.contains("switch (pc)"));
    assert!(shader.source.contains("if (steps >= 16384u)"));
}

#[test]
fn cyclic_wave64_readfirstlane_retains_shared_records_and_reuse_barrier() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VReadfirstlaneB32"),
                words: vec![0],
                sources: vec![Operand::vector(4)],
                destinations: vec![Operand::scalar(44)],
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::Known("SCbranchScc1"),
                words: vec![0xfffe],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0xbf81_0000],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute_wave(0, 64),
    )
    .expect("cyclic readfirstlane should retain shared exchange records");

    assert!(
        shader
            .source
            .contains("threadgroup uint remu_readfirstlane_values[32]")
    );
    assert_eq!(
        shader
            .source
            .matches("threadgroup_barrier(mem_flags::mem_threadgroup)")
            .count(),
        2
    );
}

#[test]
fn rejects_branches_to_non_instruction_boundaries() {
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0,
            encoding: ShaderEncoding::Sopp,
            opcode: Opcode::Known("SBranch"),
            words: vec![1],
            sources: Vec::new(),
            destinations: Vec::new(),
            control: None,
        }],
    );
    let error = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(0),
    )
    .expect_err("invalid branch target must not silently end the shader");
    assert!(error.to_string().contains("is not an instruction boundary"));
}

#[test]
fn reports_bounded_codegen_footprint_for_a_long_straight_line_shader() {
    let mut instructions = (0..64_u32)
        .map(|index| Instruction {
            pc: index * 4,
            encoding: ShaderEncoding::Vop1,
            opcode: Opcode::Known("VMovB32"),
            words: vec![0],
            sources: vec![Operand::vector(1)],
            destinations: vec![Operand::vector(8)],
            control: None,
        })
        .collect::<Vec<_>>();
    instructions.push(Instruction {
        pc: 64 * 4,
        encoding: ShaderEncoding::Sopp,
        opcode: Opcode::END_PROGRAM,
        words: vec![0xbf81_0000],
        sources: Vec::new(),
        destinations: Vec::new(),
        control: None,
    });
    let shader = compile_msl(
        &Program::new(0x1000, instructions),
        &empty_evaluation(),
        &MetalCompileOptions::compute(0),
    )
    .expect("long straight-line shader should lower");

    eprintln!(
        "remu Metal codegen: guest_instructions=65 source_bytes={} scalar_values={} vector_values={} dispatcher={}",
        shader.source.len(),
        shader.register_footprint.scalar_values,
        shader.register_footprint.vector_values,
        !shader.register_footprint.direct_control_flow,
    );
    assert_eq!(shader.register_footprint.scalar_values, 0);
    assert_eq!(shader.register_footprint.vector_values, 2);
    assert!(shader.register_footprint.direct_control_flow);
    assert!(!shader.source.contains("case 0u"));
    assert!(!shader.source.contains("steps"));
}
