// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Vector ALU lowering tests: VOP1/VOP2/VOP3 arithmetic, bitfield operations,
//! half-float maths, carry/borrow lane masks and SDWA modifiers.

use super::*;

fn compile_relative_move(name: &'static str, source: u32, destination: u32) -> MetalShader {
    let mut options = MetalCompileOptions::compute(0);
    options.allocated_vector_register_count = 8;
    compile_msl(
        &Program::new(
            0x1000,
            vec![Instruction {
                pc: 0x1000,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known(name),
                words: vec![0],
                sources: vec![Operand::vector(source)],
                destinations: vec![Operand::vector(destination)],
                control: None,
            }],
        ),
        &empty_evaluation(),
        &options,
    )
    .expect("relative vector move should lower")
}

#[test]
fn lowers_relative_vector_moves_with_m0_and_the_allocated_register_limit() {
    let source_relative = compile_relative_move("VMovrelsB32", 5, 2);
    assert!(source_relative.source.contains("uint t0 = remu_v_0;"));
    assert!(source_relative.source.contains("switch (remu_s_124)"));
    assert!(
        source_relative
            .source
            .contains("case 0u: t0 = remu_v_5; break;")
    );
    assert!(
        source_relative
            .source
            .contains("case 2u: t0 = remu_v_7; break;")
    );
    assert!(!source_relative.source.contains("remu_v_8"));
    assert!(source_relative.source.contains("remu_v_2 = t0;"));

    let destination_relative = compile_relative_move("VMovreldB32", 2, 4);
    assert!(
        destination_relative
            .source
            .contains("exec && remu_s_124 == 0u) remu_v_4 = remu_v_2;")
    );
    assert!(
        destination_relative
            .source
            .contains("exec && remu_s_124 == 3u) remu_v_7 = remu_v_2;")
    );
    assert!(!destination_relative.source.contains("remu_v_8"));

    let both_relative = compile_relative_move("VMovrelsdB32", 5, 4);
    assert!(
        both_relative
            .source
            .contains("case 2u: t0 = remu_v_7; break;")
    );
    assert!(
        both_relative
            .source
            .contains("exec && remu_s_124 == 3u) remu_v_7 = t0;")
    );
}

#[test]
fn pixel_movrels_serialization_refuses_without_a_proven_metal_path() {
    // J3: the EXEC-preserving serialization peephole is deleted. The idiom must fail loudly.
    let mut options = MetalCompileOptions::pixel(Vec::new(), 0, 0);
    options.allocated_vector_register_count = 8;
    let error = compile_msl(
        &Program::new(
            0x1000,
            vec![
                Instruction {
                    pc: 0x1000,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VReadfirstlaneB32"),
                    words: vec![0],
                    sources: vec![Operand::vector(5)],
                    destinations: vec![Operand::scalar(4)],
                    control: None,
                },
                Instruction {
                    pc: 0x1004,
                    encoding: ShaderEncoding::Vopc,
                    opcode: Opcode::Known("VCmpxEqU32"),
                    words: vec![0],
                    sources: vec![Operand::scalar(4), Operand::vector(5)],
                    destinations: vec![],
                    control: None,
                },
                Instruction {
                    pc: 0x1008,
                    encoding: ShaderEncoding::Sop1,
                    opcode: Opcode::Known("SMovB32"),
                    words: vec![0],
                    sources: vec![Operand::scalar(4)],
                    destinations: vec![Operand::scalar(124)],
                    control: None,
                },
                Instruction {
                    pc: 0x100c,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovrelsB32"),
                    words: vec![0],
                    sources: vec![Operand::vector(0)],
                    destinations: vec![Operand::vector(2)],
                    control: None,
                },
            ],
        ),
        &empty_evaluation(),
        &options,
    )
    .expect_err("pixel movrels serialization must refuse");
    assert!(
        error.to_string().contains("V_MOVRELS serialization"),
        "loud refuse must name the idiom: {error}"
    );
}

#[test]
fn lowers_unsigned_vector_bitfield_extract_without_overrunning_the_word() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0x240,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VBfeU32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(5), Operand::vector(6), Operand::vector(7)],
            destinations: vec![Operand::vector(8)],
            control: None,
        },
        MetalShaderStage::Vertex,
    );

    assert!(shader.source.contains("remu_v_6 & 31u"));
    assert!(shader.source.contains("min(remu_v_7 & 31u, 32u - t"));
    assert!(shader.source.contains("== 0u ? 0u : extract_bits("));
}

#[test]
fn lowers_vector_bitfield_mask_from_width_and_offset() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0x250,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VBfmB32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(5), Operand::vector(6), Operand::vector(7)],
            destinations: vec![Operand::vector(8)],
            control: None,
        },
        MetalShaderStage::Compute,
    );

    assert!(shader.source.contains("(1u << (remu_v_5 & 31u)) - 1u"));
    assert!(shader.source.contains("<< (remu_v_6 & 31u)"));
    assert!(!shader.source.contains("remu_v_7"));
}

#[test]
fn lowers_each_unsigned_byte_to_f32_conversion_from_the_selected_source_bits() {
    for (name, shift) in [
        ("VCvtF32Ubyte0", 0),
        ("VCvtF32Ubyte1", 8),
        ("VCvtF32Ubyte2", 16),
        ("VCvtF32Ubyte3", 24),
    ] {
        let shader = compile_vector_instruction(
            Instruction {
                pc: 0xd48,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known(name),
                words: vec![0],
                sources: vec![Operand::vector(5)],
                destinations: vec![Operand::vector(8)],
                control: None,
            },
            MetalShaderStage::Compute,
        );

        assert!(
            shader
                .source
                .contains(&format!("(float)((remu_v_5 >> {shift}u) & 0xFFu)")),
            "{name} must extract byte {shift}/8 before conversion:\n{}",
            shader.source
        );
        assert!(
            shader.source.contains("as_type<uint>("),
            "the f32 result must be stored as architectural VGPR bits"
        );
    }
}

const fn clamped_vop3() -> InstructionControl {
    InstructionControl::Vop3(remu_shader::Vop3Control {
        absolute_mask: 0,
        negate_mask: 0,
        output_modifier: 0,
        clamp: true,
        operand_select: 0,
        scalar_destination: None,
    })
}

#[test]
fn signed_vector_bitfield_extract_replicates_the_field_sign() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0x244,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VBfeI32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(5), Operand::vector(6), Operand::vector(7)],
            destinations: vec![Operand::vector(8)],
            control: Some(unmodified_vop3()),
        },
        MetalShaderStage::Vertex,
    );

    assert!(shader.source.contains("as_type<int>(remu_v_5)"));
    assert!(shader.source.contains("min(remu_v_7 & 31u, 32u - t"));
    assert!(
        shader.source.contains("as_type<uint>(extract_bits(t0,"),
        "the signed extract must sign-extend the field:\n{}",
        shader.source
    );
    assert!(
        !shader.source.contains("1u << "),
        "a width mask would zero-fill the sign bits the extract has to replicate:\n{}",
        shader.source
    );
}

#[test]
fn clamp_saturates_the_carryless_integer_add_and_subtract() {
    for (name, exact) in [
        ("VAddI32", "(long)remu_v_1 + (long)remu_v_2"),
        ("VSubI32", "(long)remu_v_1 - (long)remu_v_2"),
        ("VSubrevI32", "(long)remu_v_2 - (long)remu_v_1"),
    ] {
        let instruction = |control| Instruction {
            pc: 0x2c,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known(name),
            words: vec![0, 0],
            sources: vec![Operand::vector(1), Operand::vector(2)],
            destinations: vec![Operand::vector(8)],
            control: Some(control),
        };

        let wrapping =
            compile_vector_instruction(instruction(unmodified_vop3()), MetalShaderStage::Compute);
        assert!(
            wrapping
                .source
                .contains(&format!("remu_v_8 = (uint)({exact})")),
            "{name} without CLAMP must keep the architectural wrapping result:\n{}",
            wrapping.source
        );
        assert!(!wrapping.source.contains("4294967295l"));

        let saturating =
            compile_vector_instruction(instruction(clamped_vop3()), MetalShaderStage::Compute);
        assert!(saturating.source.contains(&format!("long t0 = {exact};")));
        assert!(
            saturating.source.contains("t0 < 0l ? 0l")
                && saturating.source.contains("t0 > 4294967295l ? 4294967295l"),
            "{name} with CLAMP must saturate to the unsigned destination range:\n{}",
            saturating.source
        );
    }
}

#[test]
fn clamp_saturates_the_carry_producing_add_without_changing_its_carry_mask() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0x30,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VAddCoU32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(1), Operand::vector(2)],
            destinations: vec![Operand::vector(8)],
            control: Some(InstructionControl::Vop3(remu_shader::Vop3Control {
                absolute_mask: 0,
                negate_mask: 0,
                output_modifier: 0,
                clamp: true,
                operand_select: 0,
                scalar_destination: Some(20),
            })),
        },
        MetalShaderStage::Compute,
    );

    assert!(shader.source.contains("? 0xFFFFFFFFu : ("));
    assert!(shader.source.contains("remu_s_20 = "));
    assert!(
        shader.source.contains(">> 32) != 0ul"),
        "the carry-out lane mask still reports the unsaturated carry:\n{}",
        shader.source
    );
}

#[test]
fn lowers_vector_bitfield_insert_select_with_exact_mask_semantics() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0xb54,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VBfiB32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(5), Operand::vector(6), Operand::vector(7)],
            destinations: vec![Operand::vector(8)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );

    assert!(shader.source.contains("= remu_v_5;"));
    assert!(shader.source.contains("= remu_v_6;"));
    assert!(shader.source.contains("= remu_v_7;"));
    assert!(shader.source.contains(" | (~"));
    assert!(shader.source.contains("remu_v_8 = (("));
}

#[test]
fn lowers_observed_rdna2_half_float_arithmetic() {
    for (name, operation) in [
        ("VAddF16", " + "),
        ("VSubF16", " - "),
        ("VSubrevF16", " - "),
        ("VMulF16", " * "),
        ("VMaxF16", "fmax("),
        ("VMinF16", "fmin("),
    ] {
        let shader = compile_vector_instruction(
            Instruction {
                pc: 0x118,
                encoding: ShaderEncoding::Vop2,
                opcode: Opcode::Known(name),
                words: vec![0],
                sources: vec![Operand::vector(5), Operand::vector(6)],
                destinations: vec![Operand::vector(8)],
                control: None,
            },
            MetalShaderStage::Pixel,
        );

        assert!(shader.source.contains("remu_f16_to_f32_bits"));
        assert!(shader.source.contains(operation));
        assert!(shader.source.contains("remu_f32_to_f16_bits"));
    }

    let fma = compile_vector_instruction(
        Instruction {
            pc: 0xf8,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VFmaF16"),
            words: vec![0, 0],
            sources: vec![Operand::vector(5), Operand::vector(6), Operand::vector(7)],
            destinations: vec![Operand::vector(8)],
            control: Some(InstructionControl::Vop3(remu_shader::Vop3Control {
                absolute_mask: 0,
                negate_mask: 0,
                output_modifier: 0,
                clamp: false,
                operand_select: 0b1100,
                scalar_destination: None,
            })),
        },
        MetalShaderStage::Pixel,
    );
    assert!(fma.source.contains("fma("));
    assert!(fma.source.contains("remu_v_8 & 0xFFFFu"));
    assert!(fma.source.contains("remu_f32_to_f16_bits"));

    for (name, conversion) in [
        ("VCvtF16F32", "remu_f32_to_f16_bits"),
        ("VCvtF32F16", "remu_f16_to_f32_bits"),
    ] {
        let shader = compile_vector_instruction(
            Instruction {
                pc: 0x138,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known(name),
                words: vec![0],
                sources: vec![Operand::vector(5)],
                destinations: vec![Operand::vector(8)],
                control: None,
            },
            MetalShaderStage::Pixel,
        );
        assert!(shader.source.contains(conversion));
        assert!(
            shader
                .source
                .contains("inline uint remu_f16_to_f32_bits(uint half_bits)"),
            "ordinary {name} must include the FP16 helper prelude its generated body calls"
        );
    }
}

#[test]
fn ordinary_f16_inline_constants_use_their_f16_values() {
    let inline = compile_vector_instruction(
        Instruction {
            pc: 0x54c,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VSubF16"),
            words: vec![0, 0],
            sources: vec![
                Operand::source(242, None),
                Operand::vector(4),
                Operand::source(128, None),
            ],
            destinations: vec![Operand::vector(4)],
            control: Some(InstructionControl::Vop3(remu_shader::Vop3Control {
                absolute_mask: 0,
                negate_mask: 0,
                output_modifier: 0,
                clamp: false,
                // Inline constants are scalar F16 values, so OPSEL must not turn 1.0 into
                // the zero-filled upper half of an implementation-side packed value.
                operand_select: 0b0001,
                scalar_destination: None,
            })),
        },
        MetalShaderStage::Pixel,
    );
    assert!(
        inline
            .source
            .contains("remu_f16_to_f32_bits(((0x3C00u) >> 0u)")
    );
    assert!(
        !inline
            .source
            .contains("remu_f16_to_f32_bits(((0x3F800000u)")
    );
    assert!(!inline.source.contains("0x3C00u) >> 16u"));

    let inline_half = compile_vector_instruction(
        Instruction {
            pc: 0x5a4,
            encoding: ShaderEncoding::Vop2,
            opcode: Opcode::Known("VMulF16"),
            words: vec![0],
            sources: vec![Operand::source(240, None), Operand::vector(5)],
            destinations: vec![Operand::vector(5)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );
    assert!(
        inline_half
            .source
            .contains("remu_f16_to_f32_bits(((0x3800u) >> 0u)")
    );
    assert!(
        !inline_half
            .source
            .contains("remu_f16_to_f32_bits(((0x3F000000u)")
    );
}

#[test]
fn lowers_observed_rdna2_half_rsqrt_compare_and_integer_folds() {
    let rsqrt = compile_vector_instruction(
        Instruction {
            pc: 0x258,
            encoding: ShaderEncoding::Vop1,
            opcode: Opcode::Known("VRsqF16"),
            words: vec![0],
            sources: vec![Operand::vector(5)],
            destinations: vec![Operand::vector(9)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );
    assert!(rsqrt.source.contains("rsqrt("));
    assert!(rsqrt.source.contains("remu_f16_to_f32_bits"));
    assert!(rsqrt.source.contains("remu_f32_to_f16_bits"));

    for (name, operation) in [
        ("VRcpF16", "1.0f /"),
        ("VSqrtF16", "sqrt("),
        ("VLogF16", "log2("),
        ("VExpF16", "exp2("),
    ] {
        let unary = compile_vector_instruction(
            Instruction {
                pc: 0x644,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known(name),
                words: vec![0],
                sources: vec![Operand::vector(5)],
                destinations: vec![Operand::vector(6)],
                control: None,
            },
            MetalShaderStage::Pixel,
        );
        assert!(unary.source.contains(operation));
        assert!(unary.source.contains("remu_f16_to_f32_bits"));
        assert!(unary.source.contains("remu_f32_to_f16_bits"));
    }

    let ldexp = compile_vector_instruction(
        Instruction {
            pc: 0xd80,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VLdexpF32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(1), Operand::vector(2)],
            destinations: vec![Operand::vector(3)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );
    assert!(ldexp.source.contains("ldexp("));
    assert!(ldexp.source.contains("as_type<int>"));

    let div_fixup = compile_vector_instruction(
        Instruction {
            pc: 0x9d8,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VDivFixupF32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
            destinations: vec![Operand::vector(4)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );
    assert!(
        div_fixup
            .source
            .contains("remu_div_fixup_f32_bits(as_type<uint>")
    );
    assert!(div_fixup.source.contains("0xFFC00000u"));
    assert!(div_fixup.source.contains("numerator_bits"));
    assert!(div_fixup.source.contains("denominator_bits"));

    let compare = compile_vector_instruction(
        Instruction {
            pc: 0x190,
            encoding: ShaderEncoding::Vopc,
            opcode: Opcode::Known("VCmpGtF16"),
            words: vec![0],
            sources: vec![Operand::vector(1), Operand::vector(2)],
            destinations: Vec::new(),
            control: None,
        },
        MetalShaderStage::Pixel,
    );
    assert!(compare.source.contains("remu_f16_to_f32_bits"));
    assert!(compare.source.contains(" > "));

    for (name, expression) in [("VMadU32U24", "0xFFFFFFu"), ("VAddLshlU32", "<< (")] {
        let shader = compile_vector_instruction(
            Instruction {
                pc: 0x148,
                encoding: ShaderEncoding::Vop3,
                opcode: Opcode::Known(name),
                words: vec![0, 0],
                sources: vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
                destinations: vec![Operand::vector(4)],
                control: None,
            },
            MetalShaderStage::Compute,
        );
        assert!(shader.source.contains(expression), "{name} did not lower");
    }
}

#[test]
fn lowers_the_division_fmas_and_half_float_min3_that_share_the_division_opcode_block() {
    let fmas = compile_vector_instruction(
        Instruction {
            pc: 0x9dc,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VDivFmasF32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
            destinations: vec![Operand::vector(4)],
            control: Some(unmodified_vop3()),
        },
        MetalShaderStage::Pixel,
    );
    assert!(fmas.source.contains("remu_div_fmas_f32(as_type<float>"));
    assert!(
        fmas.source.contains(", vcc)"),
        "the scale decision comes from VCC, not an explicit operand:\n{}",
        fmas.source
    );
    assert!(fmas.source.contains("4294967296.0f"));

    let min3 = compile_vector_instruction(
        Instruction {
            pc: 0x9e4,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VMin3F16"),
            words: vec![0, 0],
            sources: vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
            destinations: vec![Operand::vector(4)],
            control: Some(unmodified_vop3()),
        },
        MetalShaderStage::Pixel,
    );
    assert!(min3.source.contains("remu_f16_to_f32_bits"));
    assert!(min3.source.contains("fmin(fmin("));
    assert!(min3.source.contains("remu_f32_to_f16_bits"));
}

#[test]
fn lowers_division_operand_scaling_with_vcc_lane() {
    let scale = compile_msl(
        &Program::new(
            0x1000,
            vec![Instruction {
                pc: 0x1000,
                encoding: ShaderEncoding::Vop3,
                opcode: Opcode::Known("VDivScaleF32"),
                words: vec![0, 0],
                sources: vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
                destinations: vec![Operand::vector(4)],
                control: Some(InstructionControl::Vop3(remu_shader::Vop3Control {
                    absolute_mask: 0,
                    negate_mask: 0,
                    output_modifier: 0,
                    clamp: false,
                    operand_select: 0,
                    scalar_destination: Some(106),
                })),
            }],
        ),
        &empty_evaluation(),
        &MetalCompileOptions::vertex(0),
    )
    .expect("V_DIV_SCALE_F32 must lower");
    assert!(
        scale.source.contains("remu_div_scale_f32("),
        "scale helper must be emitted:\n{}",
        scale.source
    );
    assert!(
        scale.source.contains("frexp") || scale.source.contains("remu_div_scale_f32"),
        "scale must use the frexp-based prelude helper:\n{}",
        scale.source
    );
}

#[test]
fn lowers_exercised_rdna2_xor3_and_vector_shifts() {
    let xor = compile_vector_instruction(
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VXor3B32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(1), Operand::vector(2), Operand::scalar(4)],
            destinations: vec![Operand::vector(8)],
            control: Some(unmodified_vop3()),
        },
        MetalShaderStage::Compute,
    );
    assert!(xor.source.contains("remu_v_1 ^ remu_v_2 ^ remu_s_4"));

    for (opcode, operator) in [("VLshlrevB64", "<<"), ("VLshrrevB64", ">>")] {
        let shift = compile_vector_instruction(
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Vop3,
                opcode: Opcode::Known(opcode),
                words: vec![0, 0],
                sources: vec![Operand::scalar(4), Operand::vector(2), Operand::vector(3)],
                destinations: vec![Operand::vector(8), Operand::vector(9)],
                control: Some(unmodified_vop3()),
            },
            MetalShaderStage::Compute,
        );
        assert!(shift.source.contains("remu_s_4 & 63u"));
        assert!(
            shift
                .source
                .contains("((ulong)remu_v_2 | ((ulong)remu_v_3 << 32))")
        );
        assert!(shift.source.contains(&format!(" {operator} ")));
        assert!(shift.source.contains("if (exec) remu_v_8 = (uint)"));
        assert!(shift.source.contains("if (exec) remu_v_9 = (uint)("));
    }
}

#[test]
fn lowers_vector_carry_and_borrow_with_wave32_lane_masks() {
    let mut instructions = ["VAddcU32", "VSubbU32", "VSubbrevU32"]
        .into_iter()
        .enumerate()
        .map(|(index, name)| Instruction {
            pc: u32::try_from(index * 4).expect("small test pc"),
            encoding: ShaderEncoding::Vop2,
            opcode: Opcode::Known(name),
            words: vec![0],
            sources: vec![Operand::vector(1), Operand::vector(2)],
            destinations: vec![Operand::vector(
                8 + u32::try_from(index).expect("small destination"),
            )],
            control: None,
        })
        .collect::<Vec<_>>();
    for (index, name) in ["VAddCoU32", "VSubCoU32", "VSubrevCoU32", "VAddCoCiU32"]
        .into_iter()
        .enumerate()
    {
        instructions.push(Instruction {
            pc: 12 + u32::try_from(index * 8).expect("small test pc"),
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known(name),
            words: vec![0, 0],
            sources: vec![Operand::vector(1), Operand::vector(2), Operand::scalar(20)],
            destinations: vec![Operand::vector(
                12 + u32::try_from(index).expect("small destination"),
            )],
            control: Some(InstructionControl::Vop3(remu_shader::Vop3Control {
                absolute_mask: 0,
                negate_mask: 0,
                output_modifier: 0,
                clamp: false,
                operand_select: 0,
                scalar_destination: Some(12),
            })),
        });
    }
    let shader = compile_msl(
        &Program::new(0x1000, instructions),
        &empty_evaluation(),
        &MetalCompileOptions::pixel(Vec::new(), 0, 0),
    )
    .expect("carry and borrow operations should lower");

    assert!(shader.source.contains("ulong t"));
    assert!(shader.source.contains(">> 32) != 0ul"));
    assert!(shader.source.contains(" + (ulong)"));
    assert!(shader.source.contains(" < t"));
    assert!(
        shader.source.contains("? 0xffffffffu : 0u"),
        "pixel carry/borrow VCC must not depend on fragment simd_ballot"
    );
    assert!(!shader.source.contains("simd_ballot"));
    assert!(shader.source.contains("remu_s_12 ="));
    assert!(shader.source.contains("remu_s_106 ="));
}

#[test]
fn lowers_common_gen5_float_arithmetic_with_destination_accumulation() {
    let instruction = |pc, encoding, name, sources: Vec<Operand>, destination| Instruction {
        pc,
        encoding,
        opcode: Opcode::Known(name),
        words: vec![0],
        sources,
        destinations: vec![Operand::vector(destination)],
        control: None,
    };
    let program = Program::new(
        0x1000,
        vec![
            instruction(
                0,
                ShaderEncoding::Vop2,
                "VMulF32",
                vec![Operand::vector(1), Operand::vector(2)],
                0,
            ),
            instruction(
                4,
                ShaderEncoding::Vop2,
                "VMacF32",
                vec![Operand::vector(3), Operand::vector(4)],
                0,
            ),
            instruction(
                8,
                ShaderEncoding::Vop1,
                "VRcpF32",
                vec![Operand::vector(0)],
                5,
            ),
        ],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::vertex(0),
    )
    .expect("common float ALU should lower");
    assert!(
        shader
            .source
            .contains("as_type<float>(remu_v_1) * as_type<float>(remu_v_2)")
    );
    assert!(shader.source.contains(
        "fma(as_type<float>(remu_v_3), as_type<float>(remu_v_4), as_type<float>(remu_v_0))"
    ));
    assert!(shader.source.contains("1.0f / as_type<float>(remu_v_0)"));
}

#[test]
fn lowers_sdwa_source_and_destination_controls() {
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop2,
            opcode: Opcode::Known("VMulF32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(1), Operand::vector(2)],
            destinations: vec![Operand::vector(3)],
            control: Some(InstructionControl::Sdwa(SdwaControl {
                destination_select: 1,
                destination_unused: 2,
                source0_select: 4,
                source1_select: 6,
                source0_sign_extend: false,
                source1_sign_extend: false,
                absolute_mask: 1,
                negate_mask: 0,
                output_modifier: 2,
                clamp: true,
                scalar_destination: None,
            })),
        }],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::vertex(0),
    )
    .expect("SDWA float arithmetic should lower");
    assert!(shader.source.contains("(remu_v_1) & 0xFFFFu"));
    assert!(shader.source.contains("fabs(as_type<float>"));
    assert!(shader.source.contains("* 4.0f"));
    assert!(shader.source.contains("clamp("));
    assert!(shader.source.contains("& 0xFFFF00FFu"));
    assert!(shader.source.contains("<< 8"));
}

#[test]
fn sdwa_v_mov_b32_uses_floating_point_negation() {
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop1,
            opcode: Opcode::Known("VMovB32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(1)],
            destinations: vec![Operand::vector(2)],
            control: Some(InstructionControl::Sdwa(SdwaControl {
                destination_select: 6,
                destination_unused: 0,
                source0_select: 6,
                source1_select: 6,
                source0_sign_extend: false,
                source1_sign_extend: false,
                absolute_mask: 0,
                negate_mask: 1,
                output_modifier: 0,
                clamp: false,
                scalar_destination: None,
            })),
        }],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::vertex(0),
    )
    .expect("floating-point-modified V_MOV_B32 should lower");
    assert!(shader.source.contains("-(as_type<float>(remu_v_1))"));
    assert!(!shader.source.contains("0u - (remu_v_1)"));
}

#[test]
fn vop3_v_cndmask_b32_applies_floating_point_source_modifiers() {
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0x1c8,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VCndmaskB32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(2), Operand::vector(2), Operand::scalar(4)],
            destinations: vec![Operand::vector(3)],
            control: Some(InstructionControl::Vop3(remu_shader::Vop3Control {
                absolute_mask: 0,
                negate_mask: 2,
                output_modifier: 0,
                clamp: false,
                operand_select: 0,
                scalar_destination: None,
            })),
        }],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute_wave(0, 64),
    )
    .expect("floating-point-modified V_CNDMASK_B32 should lower");

    assert!(shader.source.contains("-(as_type<float>(remu_v_2))"));
    assert!(
        !shader.source.contains("? (remu_v_2) : (remu_v_2)"),
        "the selected true source must retain its VOP3 NEG modifier"
    );
}

#[test]
fn signed_and_unsigned_low_multiplies_share_one_lowering() {
    for name in ["VMulLoU32", "VMulLoI32"] {
        let shader = compile_vector_instruction(
            Instruction {
                pc: 0x2c0,
                encoding: ShaderEncoding::Vop3,
                opcode: Opcode::Known(name),
                words: vec![0, 0],
                sources: vec![Operand::vector(5), Operand::vector(6)],
                destinations: vec![Operand::vector(8)],
                control: Some(unmodified_vop3()),
            },
            MetalShaderStage::Compute,
        );

        assert!(
            shader.source.contains("(remu_v_5 * remu_v_6)"),
            "{name} keeps the low half of the product:\n{}",
            shader.source
        );
    }
}

#[test]
fn lowers_the_vector_bit_scans_with_their_no_bit_result() {
    for (name, expected) in [
        ("VFfbhU32", "clz(t0)"),
        ("VFfblB32", "ctz(t0)"),
        ("VFfbhI32", "clz(t1)"),
    ] {
        let shader = compile_vector_instruction(
            Instruction {
                pc: 0x2c8,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known(name),
                words: vec![0],
                sources: vec![Operand::vector(5)],
                destinations: vec![Operand::vector(8)],
                control: None,
            },
            MetalShaderStage::Compute,
        );

        assert!(
            shader.source.contains(expected),
            "{name} counts from the end the hardware scans:\n{}",
            shader.source
        );
        assert!(
            shader.source.contains("0xFFFFFFFFu :"),
            "{name} reports 0xFFFFFFFF when no bit matches:\n{}",
            shader.source
        );
    }

    // The signed scan finds the highest bit that differs from the sign, so a negative input is
    // inverted before the count. Scanning the value itself would answer 0 for every negative
    // number, which is a plausible-looking shift amount rather than an obvious failure.
    let signed = compile_vector_instruction(
        Instruction {
            pc: 0x2c8,
            encoding: ShaderEncoding::Vop1,
            opcode: Opcode::Known("VFfbhI32"),
            words: vec![0],
            sources: vec![Operand::vector(5)],
            destinations: vec![Operand::vector(8)],
            control: None,
        },
        MetalShaderStage::Compute,
    );
    assert!(
        signed.source.contains("as_type<int>(t0) < 0 ? ~t0 : t0"),
        "V_FFBH_I32 inverts a negative value before scanning:\n{}",
        signed.source
    );
}

#[test]
fn the_legacy_multiply_keeps_a_zero_operand_from_producing_a_nan() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0x2d0,
            encoding: ShaderEncoding::Vop2,
            opcode: Opcode::Known("VMulLegacyF32"),
            words: vec![0],
            sources: vec![Operand::vector(5), Operand::vector(6)],
            destinations: vec![Operand::vector(8)],
            control: None,
        },
        MetalShaderStage::Compute,
    );

    assert!(
        shader
            .source
            .contains("t0 == 0.0f || t1 == 0.0f) ? 0.0f : t0 * t1"),
        "the DX9 multiply answers zero for a zero operand instead of the IEEE NaN:\n{}",
        shader.source
    );
}
