use super::*;

#[test]
fn classifies_operands_without_conflating_literals_and_inline_constants() {
    assert_eq!(Operand::source(12, None), Operand::scalar(12));
    assert_eq!(Operand::source(256, None), Operand::vector(0));
    assert_eq!(
        Operand::source(249, Some(0xdead_beef)),
        Operand {
            kind: OperandKind::LiteralConstant,
            value: 0xdead_beef,
        }
    );
    assert_eq!(
        Operand::source(242, None),
        Operand {
            kind: OperandKind::EncodedConstant,
            value: 242,
        }
    );
}

#[test]
fn computes_pixel_masks_and_runtime_scalar_registers_once() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sop2,
                opcode: Opcode::from_static("SAddU32"),
                words: vec![0],
                sources: vec![Operand::scalar(4), Operand::vector(1)],
                destinations: vec![Operand::scalar(6)],
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Exp,
                opcode: Opcode::EXPORT,
                words: vec![0, 0],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: Some(InstructionControl::Export(ExportControl {
                    target: 2,
                    enable_mask: 0b1011,
                    compressed: false,
                    done: true,
                    valid_mask: false,
                })),
            },
        ],
    );
    assert_eq!(program.pixel_color_export_masks(), 0x0b00);
    assert!(
        program
            .runtime_scalar_registers()
            .is_superset(&BTreeSet::from([4, 5, 6, 106, 107, 124, 125, 126, 127,]))
    );
    assert!(program.input_scalar_registers().contains(&4));
    assert!(!program.input_scalar_registers().contains(&6));
}

#[test]
fn straight_line_temporaries_do_not_become_shader_inputs() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::from_static("SMovB32"),
                words: vec![0],
                sources: vec![Operand::scalar(4)],
                destinations: vec![Operand::scalar(6)],
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Sop2,
                opcode: Opcode::from_static("SAddU32"),
                words: vec![0],
                sources: vec![Operand::scalar(6), Operand::scalar(8)],
                destinations: vec![Operand::scalar(10)],
                control: None,
            },
        ],
    );
    assert!(program.input_scalar_registers().contains(&4));
    assert!(program.input_scalar_registers().contains(&8));
    assert!(!program.input_scalar_registers().contains(&6));
    assert!(!program.input_scalar_registers().contains(&10));
}

#[test]
fn vector_only_scalar_sources_do_not_become_evaluation_inputs() {
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop2,
            opcode: Opcode::from_static("VAddF32"),
            words: vec![0],
            sources: vec![Operand::scalar(4), Operand::vector(1)],
            destinations: vec![Operand::vector(0)],
            control: None,
        }],
    );
    assert!(program.runtime_scalar_registers().contains(&4));
    assert!(!program.input_scalar_registers().contains(&4));
}

#[test]
fn branched_programs_keep_conservative_shader_inputs() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::from_static("SMovB32"),
                words: vec![0],
                sources: vec![Operand::scalar(4)],
                destinations: vec![Operand::scalar(6)],
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::from_static("SCbranchScc0"),
                words: vec![0],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Sop2,
                opcode: Opcode::from_static("SAddU32"),
                words: vec![0],
                sources: vec![Operand::scalar(6), Operand::scalar(8)],
                destinations: vec![Operand::scalar(10)],
                control: None,
            },
        ],
    );
    assert!(program.input_scalar_registers().contains(&6));
}
