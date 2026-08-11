// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Conversion and pack/unpack lowering tests: f32/f16 rounding, integer
//! conversions and the packed 16-bit destination formats.

use super::*;

fn emulated_f32_to_f16_rtz(bits: u32) -> u16 {
    let sign = ((bits >> 16) & 0x8000) as u16;
    let absolute = bits & 0x7fff_ffff;
    let exponent = (absolute >> 23) & 0xff;
    let mantissa = absolute & 0x7f_ffff;

    if exponent == 0xff {
        if mantissa == 0 {
            return sign | 0x7c00;
        }
        let payload = ((mantissa >> 13) | 0x200) as u16;
        return sign | 0x7c00 | (payload & 0x03ff);
    }
    if exponent > 142 {
        return sign | 0x7bff;
    }
    if exponent >= 113 {
        return sign | (((exponent - 112) << 10) as u16) | ((mantissa >> 13) as u16);
    }
    if exponent >= 103 {
        let significand = mantissa | 0x80_0000;
        return sign | ((significand >> (126 - exponent)) as u16);
    }
    sign
}

fn positive_f16_to_f32_bits(bits: u16) -> u32 {
    let exponent = (u32::from(bits) >> 10) & 0x1f;
    let mantissa = u32::from(bits) & 0x03ff;
    if exponent != 0 {
        return ((exponent + 112) << 23) | (mantissa << 13);
    }
    if mantissa == 0 {
        return 0;
    }
    let high_bit = 31 - mantissa.leading_zeros();
    ((high_bit + 103) << 23) | ((mantissa << (23 - high_bit)) & 0x7f_ffff)
}

fn reference_f32_to_f16_rtz(bits: u32, finite_half_values: &[u32]) -> u16 {
    let sign = ((bits >> 16) & 0x8000) as u16;
    let absolute = bits & 0x7fff_ffff;
    let mantissa = absolute & 0x7f_ffff;
    if absolute >= 0x7f80_0000 {
        if mantissa == 0 {
            return sign | 0x7c00;
        }
        let payload = ((mantissa >> 13) | 0x200) as u16;
        return sign | 0x7c00 | (payload & 0x03ff);
    }

    // Positive finite IEEE-754 bit patterns have the same ordering as their values. The
    // round-toward-zero result is therefore the greatest representable half no larger than
    // the input magnitude.
    let upper = finite_half_values.partition_point(|candidate| *candidate <= absolute);
    sign | u16::try_from(upper.saturating_sub(1)).expect("finite half index fits u16")
}

#[test]
fn lowers_packed_f32_to_f16_with_an_integer_rtz_conversion() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop2,
            opcode: Opcode::Known("VCvtPkrtzF16F32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(1), Operand::vector(2)],
            destinations: vec![Operand::vector(8)],
            control: None,
        },
        MetalShaderStage::Vertex,
    );

    assert_eq!(
        shader.source.matches("remu_f32_to_f16_rtz_bits(").count(),
        3,
        "the prelude helper must be called once for each packed lane"
    );
    assert!(!shader.source.contains("& 0xFFFFE000u"));
    assert!(!shader.source.contains("as_type<ushort>(half("));
}

#[test]
fn lowers_silent_hill_xor_add_and_unsigned_u16_pack() {
    let xor_add = compile_vector_instruction(
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VXadU32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
            destinations: vec![Operand::vector(8)],
            control: None,
        },
        MetalShaderStage::Compute,
    );
    assert!(
        xor_add
            .source
            .contains("((remu_v_1 ^ remu_v_2) + remu_v_3)")
    );

    let pack = compile_vector_instruction(
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VCvtPkU16U32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
            destinations: vec![Operand::vector(8)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );
    assert!(
        pack.source
            .contains("((remu_v_1 & 0xFFFFu) | ((remu_v_2 & 0xFFFFu) << 16))")
    );
}

fn reference_cvt_flr_i32_f32(value: f32) -> u32 {
    if value.is_nan() {
        0
    } else if value <= -2_147_483_648.0 {
        0x8000_0000
    } else if value >= 2_147_483_648.0 {
        0x7fff_ffff
    } else {
        (value.floor() as i32).cast_unsigned()
    }
}

fn reference_cvt_unorm16_f32(value: f32) -> u16 {
    if value.is_nan() || value <= 0.0 {
        0
    } else if value >= 1.0 {
        u16::MAX
    } else {
        (value * 65_535.0).round_ties_even() as u16
    }
}

#[test]
fn rdna2_floor_and_packed_unorm_conversions_preserve_exact_integer_semantics() {
    for (value, expected) in [
        (f32::NAN, 0),
        (f32::NEG_INFINITY, 0x8000_0000),
        (-2_147_483_648.0, 0x8000_0000),
        (-1.000_000_1, 0xffff_fffe),
        (-1.0, 0xffff_ffff),
        (-0.0, 0),
        (0.999_999_94, 0),
        (1.0, 1),
        (2_147_483_520.0, 0x7fff_ff80),
        (2_147_483_648.0, 0x7fff_ffff),
        (f32::INFINITY, 0x7fff_ffff),
    ] {
        assert_eq!(
            reference_cvt_flr_i32_f32(value),
            expected,
            "floor conversion mismatch for {value:?}"
        );
    }

    for (value, expected) in [
        (f32::NAN, 0),
        (f32::NEG_INFINITY, 0),
        (-1.0, 0),
        (-0.0, 0),
        (0.0, 0),
        (0.5 / 65_535.0, 0),
        (1.5 / 65_535.0, 2),
        (0.5, 0x8000),
        (1.0, 0xffff),
        (f32::INFINITY, 0xffff),
    ] {
        assert_eq!(
            reference_cvt_unorm16_f32(value),
            expected,
            "UNORM16 conversion mismatch for {value:?}"
        );
    }

    let floor = compile_vector_instruction(
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VCvtFlrI32F32"),
            words: vec![0, 0],
            sources: vec![
                Operand::vector(1),
                Operand::source(128, None),
                Operand::source(128, None),
            ],
            destinations: vec![Operand::vector(8)],
            control: Some(InstructionControl::Vop3(remu_shader::Vop3Control {
                absolute_mask: 1,
                negate_mask: 1,
                output_modifier: 1,
                clamp: true,
                operand_select: 0,
                scalar_destination: None,
            })),
        },
        MetalShaderStage::Pixel,
    );
    assert!(floor.source.contains("floor(t0)"));
    assert!(floor.source.contains("t0 <= -2147483648.0f ? 0x80000000u"));
    assert!(floor.source.contains("t0 >= 2147483648.0f ? 0x7FFFFFFFu"));
    assert!(
        floor
            .source
            .contains("float t0 = -(fabs(as_type<float>(remu_v_1)));")
    );
    assert!(
        !floor.source.contains("floor(t0) * 2.0f"),
        "OMOD is ignored for this integer result"
    );

    let packed = compile_vector_instruction(
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VCvtPknormU16F32"),
            words: vec![0, 0],
            sources: vec![
                Operand::vector(1),
                Operand::vector(2),
                Operand::source(128, None),
            ],
            destinations: vec![Operand::vector(8)],
            control: Some(InstructionControl::Vop3(remu_shader::Vop3Control {
                absolute_mask: 1,
                negate_mask: 2,
                output_modifier: 2,
                clamp: true,
                operand_select: 0,
                scalar_destination: None,
            })),
        },
        MetalShaderStage::Pixel,
    );
    assert!(
        packed
            .source
            .contains("float t0 = fabs(as_type<float>(remu_v_1));")
    );
    assert!(
        packed
            .source
            .contains("float t1 = -(as_type<float>(remu_v_2));")
    );
    assert_eq!(
        packed.source.matches("(uint)rint(").count(),
        2,
        "each source must independently round to nearest-even"
    );
    assert!(packed.source.contains("t0 * 65535.0f"));
    assert!(packed.source.contains("t1 * 65535.0f"));
    assert!(packed.source.contains("(t2 | (t3 << 16))"));
}

#[test]
fn lowers_rdna_integer_round_positive_and_u8_insert_conversions() {
    let round_positive = compile_vector_instruction(
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop1,
            opcode: Opcode::Known("VCvtRpiI32F32"),
            words: vec![0],
            sources: vec![Operand::vector(1)],
            destinations: vec![Operand::vector(8)],
            control: None,
        },
        MetalShaderStage::Compute,
    );
    assert!(
        round_positive
            .source
            .contains("floor(as_type<float>(remu_v_1) + 0.5f)")
    );
    assert!(round_positive.source.contains("+ 0.5f"));
    assert!(
        round_positive
            .source
            .contains("<= -2147483648.0f ? 0x80000000u")
    );
    assert!(
        round_positive
            .source
            .contains(">= 2147483648.0f ? 0x7FFFFFFFu")
    );

    let insert_byte = compile_vector_instruction(
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VCvtPkU8F32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
            destinations: vec![Operand::vector(8)],
            control: None,
        },
        MetalShaderStage::Compute,
    );
    assert!(insert_byte.source.contains("remu_v_2 & 3u"));
    assert!(insert_byte.source.contains("0xFFu <<"));
    assert!(insert_byte.source.contains("remu_v_3 & ~"));
    assert!(insert_byte.source.contains(">= 255.0f ? 0xFFu"));
    assert!(insert_byte.source.contains("(uint)trunc("));
}

#[test]
fn lowers_silent_hill_half_to_u16_and_signed_bitfield_extract() {
    let conversion = compile_vector_instruction(
        Instruction {
            pc: 0x638,
            encoding: ShaderEncoding::Vop1,
            opcode: Opcode::Known("VCvtU16F16"),
            words: vec![0],
            sources: vec![Operand::vector(1)],
            destinations: vec![Operand::vector(8)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );
    assert!(conversion.source.contains("remu_f16_to_f32_bits"));
    assert!(conversion.source.contains("isnan("));
    assert!(conversion.source.contains(">= 65535.0f"));
    assert!(conversion.source.contains("(uint)trunc("));

    let signed_extract = compile_vector_instruction(
        Instruction {
            pc: 0xa0,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VBfeI32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
            destinations: vec![Operand::vector(8)],
            control: None,
        },
        MetalShaderStage::Compute,
    );
    assert!(signed_extract.source.contains("as_type<int>(remu_v_1)"));
    assert!(signed_extract.source.contains("remu_v_2 & 31u"));
    assert!(signed_extract.source.contains("remu_v_3 & 31u"));
    assert!(
        signed_extract
            .source
            .contains("as_type<uint>(extract_bits(t0,")
    );
}

#[test]
fn packed_f32_to_f16_rtz_matches_reference_boundaries_and_special_values() {
    let finite_half_values = (0u16..=0x7bff)
        .map(positive_f16_to_f32_bits)
        .collect::<Vec<_>>();
    let largest_half_subnormal = positive_f16_to_f32_bits(0x03ff);
    let smallest_half_normal = positive_f16_to_f32_bits(0x0400);
    let cases = [
        (0x0000_0000, 0x0000),
        (0x8000_0000, 0x8000),
        (0x0000_0001, 0x0000),
        (0x8000_0001, 0x8000),
        (0x337f_ffff, 0x0000),
        (0x3380_0000, 0x0001),
        (0x33c0_0000, 0x0001),
        (largest_half_subnormal, 0x03ff),
        (smallest_half_normal - 1, 0x03ff),
        (smallest_half_normal, 0x0400),
        (0x3f80_0000, 0x3c00),
        (0x3f80_1fff, 0x3c00),
        (0x3f80_2000, 0x3c01),
        (0xbf80_1fff, 0xbc00),
        (0x477f_e000, 0x7bff),
        (0x477f_ffff, 0x7bff),
        (0x4780_0000, 0x7bff),
        (0x7f7f_ffff, 0x7bff),
        (0xff7f_ffff, 0xfbff),
        (0x7f80_0000, 0x7c00),
        (0xff80_0000, 0xfc00),
        (0x7f80_0001, 0x7e00),
        (0x7fc1_2345, 0x7e09),
        (0xffc1_2345, 0xfe09),
        (0x7fff_ffff, 0x7fff),
    ];

    for (input, expected) in cases {
        assert_eq!(
            emulated_f32_to_f16_rtz(input),
            expected,
            "incorrect RTZ conversion for f32 bits 0x{input:08x}"
        );
        assert_eq!(
            reference_f32_to_f16_rtz(input, &finite_half_values),
            expected,
            "reference disagrees for f32 bits 0x{input:08x}"
        );
    }
}

#[test]
fn packed_f32_to_f16_rtz_exhausts_all_semantically_distinct_bit_fields() {
    let finite_half_values = (0u16..=0x7bff)
        .map(positive_f16_to_f32_bits)
        .collect::<Vec<_>>();

    // RTZ discards at least the low 13 mantissa bits. Exhaust every sign, exponent, and
    // retained ten-bit mantissa, then check both ends of the discarded-bit interval.
    for sign in [0, 0x8000_0000] {
        for exponent in 0..=0xff {
            for retained_mantissa in 0..=0x03ff {
                for discarded_mantissa in [0, 0x1fff] {
                    let input =
                        sign | (exponent << 23) | (retained_mantissa << 13) | discarded_mantissa;
                    assert_eq!(
                        emulated_f32_to_f16_rtz(input),
                        reference_f32_to_f16_rtz(input, &finite_half_values),
                        "RTZ mismatch for f32 bits 0x{input:08x}"
                    );
                }
            }
        }
    }
}
