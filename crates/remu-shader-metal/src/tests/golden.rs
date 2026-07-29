// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Whole-program regression tests: unsupported opcode reporting and the
//! byte-identical Metal source digests for the direct and forward-control-flow paths.

use super::*;

#[test]
fn rejects_raw_opcodes_instead_of_emitting_placeholder_code() {
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Raw {
                family: remu_shader::RawOpcodeFamily::Vop3,
                code: 0x1ff,
            },
            words: vec![0, 0],
            sources: Vec::new(),
            destinations: Vec::new(),
            control: None,
        }],
    );
    let error = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::vertex(0),
    )
    .expect_err("raw opcode must fail");
    assert!(error.to_string().contains("raw opcode"));
}

const DIRECT_SOURCE_DIGEST: u64 = 0xfdb5_3279_d8e0_bd9f;

const FORWARD_SOURCE_DIGEST: u64 = 0x9e0f_43b3_c561_81a4;

fn source_digest(source: &str) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    source.bytes().fold(FNV_OFFSET_BASIS, |digest, byte| {
        (digest ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
    })
}

fn direct_reference_program() -> Program {
    Program::new(
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
                destinations: vec![Operand::vector(3)],
                control: None,
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Vop2,
                opcode: Opcode::Known("VAddF32"),
                words: vec![0],
                sources: vec![Operand::vector(3), Operand::vector(4)],
                destinations: vec![Operand::vector(5)],
                control: None,
            },
            Instruction {
                pc: 12,
                encoding: ShaderEncoding::Vop2,
                opcode: Opcode::Known("VMulF32"),
                words: vec![0],
                sources: vec![Operand::vector(5), Operand::vector(5)],
                destinations: vec![Operand::vector(6)],
                control: None,
            },
            Instruction {
                pc: 16,
                encoding: ShaderEncoding::Vop2,
                opcode: Opcode::Known("VAddcU32"),
                words: vec![0],
                sources: vec![Operand::vector(6), Operand::vector(7)],
                destinations: vec![Operand::vector(8)],
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
    )
}

fn forward_reference_program() -> Program {
    Program::new(
        0x2000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::Known("SCbranchScc1"),
                // PC 0 + 4-byte instruction + two dwords = PC 12.
                words: vec![2],
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
                encoding: ShaderEncoding::Vop2,
                opcode: Opcode::Known("VMulF32"),
                words: vec![0],
                sources: vec![Operand::vector(2), Operand::vector(2)],
                destinations: vec![Operand::vector(3)],
                control: None,
            },
            Instruction {
                pc: 12,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0xbf81_0000],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    )
}

fn unsupported_reference_program() -> Program {
    Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Vop3,
                opcode: Opcode::Raw {
                    family: RawOpcodeFamily::Vop3,
                    code: 0x2a4,
                },
                words: vec![0, 0],
                sources: vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
                destinations: vec![Operand::vector(4)],
                control: None,
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VMovB32"),
                words: vec![0],
                sources: vec![Operand::vector(4)],
                destinations: vec![Operand::vector(5)],
                control: None,
            },
            Instruction {
                pc: 12,
                encoding: ShaderEncoding::Vop3,
                opcode: Opcode::Known("VBfiB32"),
                words: vec![0, 0],
                sources: vec![Operand::vector(5), Operand::vector(6), Operand::vector(7)],
                destinations: vec![Operand::vector(8)],
                control: None,
            },
            Instruction {
                pc: 20,
                encoding: ShaderEncoding::Vop3,
                opcode: Opcode::Known("VAdd3U32"),
                words: vec![0, 0],
                sources: vec![Operand::vector(8), Operand::vector(9), Operand::vector(10)],
                destinations: vec![Operand::vector(11)],
                control: None,
            },
            Instruction {
                pc: 28,
                encoding: ShaderEncoding::Sop2,
                opcode: Opcode::Known("SAbsdiffI32"),
                words: vec![0],
                sources: vec![Operand::scalar(10), Operand::scalar(11)],
                destinations: vec![Operand::scalar(12)],
                control: None,
            },
            Instruction {
                pc: 32,
                encoding: ShaderEncoding::Vop3,
                opcode: Opcode::Raw {
                    family: RawOpcodeFamily::Vop3,
                    code: 0x2a5,
                },
                words: vec![0, 0],
                sources: vec![Operand::vector(12)],
                destinations: vec![Operand::vector(13)],
                control: None,
            },
            Instruction {
                pc: 40,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0xbf81_0000],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    )
}

#[test]
fn a_shader_using_several_unsupported_opcodes_reports_all_of_them() {
    let error = compile_msl(
        &unsupported_reference_program(),
        &empty_evaluation(),
        &MetalCompileOptions::compute(0),
    )
    .expect_err("unsupported opcodes should fail lowering");

    let message = error.to_string();
    assert!(
        message.starts_with("pc 0x0 Vop3[0x2A4]: raw opcode has no lowering semantics;"),
        "{message}"
    );
    assert!(!message.contains("SAbsdiffI32"), "{message}");
    assert!(
        message.contains("pc 0x20 Vop3[0x2A5]: raw opcode has no lowering semantics"),
        "{message}"
    );
    assert!(!message.contains("pc 0xC"), "{message}");
    assert!(!message.contains("pc 0x14"), "{message}");
}

#[test]
fn a_shader_with_one_raw_opcode_reports_only_that_opcode() {
    let mut instructions = unsupported_reference_program().instructions;
    instructions.retain(|instruction| matches!(instruction.pc, 0 | 40));
    let error = compile_msl(
        &Program::new(0x1000, instructions),
        &empty_evaluation(),
        &MetalCompileOptions::compute(0),
    )
    .expect_err("an unsupported opcode should fail lowering");

    assert_eq!(
        error.to_string(),
        "pc 0x0 Vop3[0x2A4]: raw opcode has no lowering semantics"
    );
}

#[test]
fn a_shader_that_lowers_today_still_emits_byte_identical_metal_source() {
    let direct = compile_msl(
        &direct_reference_program(),
        &empty_evaluation(),
        &MetalCompileOptions::compute(0),
    )
    .expect("direct reference program should lower");
    let forward = compile_msl(
        &forward_reference_program(),
        &empty_evaluation(),
        &MetalCompileOptions::compute(0),
    )
    .expect("forward-control-flow reference program should lower");

    assert!(direct.register_footprint.direct_control_flow);
    assert!(!forward.register_footprint.direct_control_flow);
    assert_eq!(
        source_digest(&direct.source),
        DIRECT_SOURCE_DIGEST,
        "direct lowering emitted different MSL:\n{}",
        direct.source
    );
    assert_eq!(
        source_digest(&forward.source),
        FORWARD_SOURCE_DIGEST,
        "forward-control-flow lowering emitted different MSL:\n{}",
        forward.source
    );
}
