use remu_shader::{
    ComputeSystemRegisters, DataShareControl, Dpp8Control, DppControl, Instruction, Program,
    RawOpcodeFamily, ShaderEvaluation, Vop3pControl, decode_program,
};

use super::*;

fn empty_evaluation() -> ShaderEvaluation {
    ShaderEvaluation {
        initial_scalar_registers: vec![0; 128].into(),
        scalar_registers: vec![0; 128].into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: Vec::new(),
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    }
}

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
fn emits_instance_routing_for_layered_render_targets() {
    let program = decode_program(0x1000, &[0xbf81_0000]).expect("minimal vertex program decodes");
    let mut options = MetalCompileOptions::vertex(0);
    options.render_target_array_length = 64;
    let shader =
        compile_msl(&program, &empty_evaluation(), &options).expect("layered vertex lowers");

    assert!(
        shader
            .source
            .contains("uint render_target_array_index [[render_target_array_index]]")
    );
    assert!(
        shader
            .source
            .contains("remu_out.render_target_array_index = min(remu_instance_id, 63u);")
    );
}

#[test]
fn builds_fused_es_gs_rectangle_equivalent_for_volume_generation() {
    let shader = compile_layered_rectangle_vertex(2, 64);

    assert_eq!(shader.stage, MetalShaderStage::Vertex);
    assert_eq!(shader.interpolated_attribute_count, 2);
    assert!(
        shader
            .source
            .contains("uint render_target_array_index [[render_target_array_index]]")
    );
    assert!(
        shader
            .source
            .contains("remu_out.render_target_array_index = min(remu_instance_id, 63u);")
    );
    assert!(
        shader
            .source
            .contains("float remu_layer = (float(remu_instance_id) + 0.5f) / 64.0f;")
    );
    assert!(
        shader
            .source
            .contains("remu_out.param1 = float4(remu_uv, remu_layer, 1.0f);")
    );
}

#[test]
fn builds_procedural_rectangle_with_tint_and_scaled_uv_parameters() {
    let shader = compile_procedural_rectangle_vertex(2);

    assert_eq!(shader.stage, MetalShaderStage::Vertex);
    assert_eq!(shader.interpolated_attribute_count, 2);
    assert!(
        shader
            .source
            .contains("remu_out.position = float4(remu_uv * 2.0f - 1.0f, 0.0f, 1.0f);")
    );
    assert!(shader.source.contains("remu_out.param0 = float4(1.0f);"));
    assert!(
        shader
            .source
            .contains("remu_out.param1 = float4(remu_uv, 1.0f, 1.0f);")
    );
}

fn compile_pixel_system_input_shader(
    referenced_vgpr: u32,
    pixel_input_enable: u32,
    pixel_input_address: u32,
    render_target_array_length: u32,
) -> MetalShader {
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop1,
            opcode: Opcode::Known("VMovB32"),
            words: vec![0],
            sources: vec![Operand::vector(referenced_vgpr)],
            destinations: vec![Operand::vector(8)],
            control: None,
        }],
    );
    let mut options =
        MetalCompileOptions::pixel(Vec::new(), pixel_input_enable, pixel_input_address);
    options.render_target_array_length = render_target_array_length;
    compile_msl(&program, &empty_evaluation(), &options).expect("pixel system input should lower")
}

#[test]
fn routes_layered_render_target_index_into_ancillary_vgpr() {
    let addressed = (1 << 0) | (1 << 13);
    let shader = compile_pixel_system_input_shader(2, addressed, addressed, 32);

    assert!(
        shader
            .source
            .contains("uint render_target_array_index [[render_target_array_index]];")
    );
    assert!(
        shader
            .source
            .contains("remu_v_2 = remu_in.render_target_array_index << 16u;")
    );
}

#[test]
fn disabled_system_inputs_still_reserve_compacted_pixel_vgprs() {
    let address = (1 << 0) | (1 << 12) | (1 << 13);
    let enable = (1 << 0) | (1 << 13);
    let shader = compile_pixel_system_input_shader(3, enable, address, 32);

    assert!(
        shader
            .source
            .contains("remu_v_3 = remu_in.render_target_array_index << 16u;")
    );
    assert!(
        !shader
            .source
            .contains("remu_v_2 = remu_in.render_target_array_index << 16u;")
    );
}

#[test]
fn omits_layer_index_input_for_single_layer_pixel_shader() {
    let addressed = (1 << 0) | (1 << 13);
    let shader = compile_pixel_system_input_shader(2, addressed, addressed, 1);

    assert!(
        !shader
            .source
            .contains("uint render_target_array_index [[render_target_array_index]];")
    );
    assert!(!shader.source.contains("remu_in.render_target_array_index"));
}

#[test]
fn materializes_pixel_position_into_the_named_vgpr() {
    let shader = compile_pixel_system_input_shader(0, 1 << 8, 1 << 8, 1);

    assert!(
        shader
            .source
            .contains("remu_v_0 = as_type<uint>(remu_in.frag_coord.x);")
    );
    assert!(!shader.source.contains("    v[0] ="));
}

fn interpolation_instruction(
    pc: u32,
    attribute: u32,
    channel: u32,
    barycentric_register: u32,
    destination: u32,
) -> Instruction {
    Instruction {
        pc,
        encoding: ShaderEncoding::Vintrp,
        opcode: Opcode::Known("VInterpP2F32"),
        words: vec![0],
        sources: vec![Operand::vector(barycentric_register)],
        destinations: vec![Operand::vector(destination)],
        control: Some(InstructionControl::Interpolation(
            remu_shader::InterpolationControl { attribute, channel },
        )),
    }
}

fn compile_interpolation_shader(
    instructions: Vec<Instruction>,
    pixel_input_address: u32,
    pixel_input_controls: [PixelInputControl; PIXEL_INPUT_CONTROL_COUNT],
) -> MetalShader {
    let mut options =
        MetalCompileOptions::pixel(Vec::new(), pixel_input_address, pixel_input_address);
    options.pixel_input_controls = pixel_input_controls;
    compile_msl(
        &Program::new(0x1000, instructions),
        &empty_evaluation(),
        &options,
    )
    .expect("pixel interpolation should lower")
}

#[test]
fn maps_logical_pixel_input_to_physical_vertex_parameter() {
    let mut controls = identity_pixel_input_controls();
    controls[0] = PixelInputControl::from_register(4);
    assert_eq!(
        controls[0],
        PixelInputControl {
            input_offset: 4,
            use_default: false,
            default_value: 0,
            flat_shade: false,
        }
    );
    assert_eq!(controls[0].register_value(), 4);

    let pixel = compile_interpolation_shader(
        vec![interpolation_instruction(0, 0, 0, 1, 8)],
        1 << 1,
        controls,
    );
    assert!(
        pixel
            .source
            .contains("float4 attr0 [[user(locn4), center_perspective]];")
    );
    assert!(pixel.source.contains("as_type<uint>(remu_in.attr0[0])"));
    assert_eq!(pixel.interpolated_attribute_count, 5);

    let vertex_program =
        decode_program(0x1000, &[0xbf81_0000]).expect("minimal vertex program decodes");
    let vertex = compile_msl(
        &vertex_program,
        &empty_evaluation(),
        &MetalCompileOptions::vertex(pixel.interpolated_attribute_count),
    )
    .expect("mapped vertex interface should lower");
    assert!(vertex.source.contains("float4 param4 [[user(locn4)]];"));
}

#[test]
fn materializes_architectural_pixel_input_default_two() {
    let mut controls = identity_pixel_input_controls();
    controls[0] = PixelInputControl::from_register((1 << 5) | (2 << 8));
    let pixel = compile_interpolation_shader(
        vec![
            interpolation_instruction(0, 0, 2, 1, 8),
            interpolation_instruction(4, 0, 3, 1, 9),
        ],
        1 << 1,
        controls,
    );

    assert!(!pixel.source.contains("float4 attr0"));
    assert!(pixel.source.contains("remu_v_8 = 0x3F800000u;"));
    assert!(pixel.source.contains("remu_v_9 = 0u;"));
    assert_eq!(pixel.interpolated_attribute_count, 0);
}

#[test]
fn emits_flat_interpolation_from_pixel_input_control() {
    let mut controls = identity_pixel_input_controls();
    controls[0] = PixelInputControl::from_register(3 | (1 << 10));
    let pixel = compile_interpolation_shader(
        vec![interpolation_instruction(0, 0, 0, 1, 8)],
        1 << 1,
        controls,
    );

    assert!(pixel.source.contains("float4 attr0 [[user(locn3), flat]];"));
    assert_eq!(pixel.interpolated_attribute_count, 4);
}

#[test]
fn derives_all_metal_interpolation_modes_from_p2_barycentric_vgpr() {
    let pixel = compile_interpolation_shader(
        [(0, 1), (1, 3), (2, 5), (3, 7), (4, 9), (5, 11)]
            .into_iter()
            .enumerate()
            .map(|(index, (attribute, barycentric_register))| {
                interpolation_instruction(
                    u32::try_from(index * 4).expect("test PC fits u32"),
                    attribute,
                    0,
                    barycentric_register,
                    16 + attribute,
                )
            })
            .collect(),
        0x77,
        identity_pixel_input_controls(),
    );

    for (attribute, qualifier) in [
        (0, "sample_perspective"),
        (1, "center_perspective"),
        (2, "centroid_perspective"),
        (3, "sample_no_perspective"),
        (4, "center_no_perspective"),
        (5, "centroid_no_perspective"),
    ] {
        assert!(
            pixel.source.contains(&format!(
                "float4 attr{attribute} [[user(locn{attribute}), {qualifier}]];"
            )),
            "missing {qualifier} for attr{attribute}"
        );
    }
}

fn vector_instruction(
    name: &'static str,
    source_count: usize,
    control: InstructionControl,
) -> Instruction {
    Instruction {
        pc: 0,
        encoding: ShaderEncoding::Vop2,
        opcode: Opcode::Known(name),
        words: vec![0, 0],
        sources: (1..=source_count)
            .map(|register| {
                Operand::vector(u32::try_from(register).expect("test source index fits u32"))
            })
            .collect(),
        destinations: vec![Operand::vector(8)],
        control: Some(control),
    }
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

fn permlane_instruction(operand_select: u32, exchange_rows: bool) -> Instruction {
    Instruction {
        pc: 0x80,
        encoding: ShaderEncoding::Vop3,
        opcode: Opcode::Known(if exchange_rows {
            "VPermlanex16B32"
        } else {
            "VPermlane16B32"
        }),
        words: vec![0, 0],
        sources: vec![Operand::vector(1), Operand::scalar(10), Operand::scalar(11)],
        destinations: vec![Operand::vector(8)],
        control: Some(InstructionControl::Vop3(remu_shader::Vop3Control {
            absolute_mask: 0,
            negate_mask: 0,
            output_modifier: 0,
            clamp: false,
            operand_select,
            scalar_destination: None,
        })),
    }
}

#[test]
fn lowers_rdna2_permlane16_lane_selection_and_inactive_lane_modes() {
    let suppress_write =
        compile_vector_instruction(permlane_instruction(0, false), MetalShaderStage::Compute);
    assert!(suppress_write.source.contains("remu_lane & 15u"));
    assert!(suppress_write.source.contains("< 8u"));
    assert!(suppress_write.source.contains("& 7u) * 4u"));
    assert!(suppress_write.source.contains("remu_lane & 16u"));
    assert!(
        suppress_write
            .source
            .contains("simd_shuffle(remu_v_1, (ushort)")
    );
    assert!(
        suppress_write
            .source
            .contains("simd_shuffle(exec ? 1u : 0u, (ushort)")
    );
    assert!(suppress_write.source.contains("if (exec && t"));

    let zero_inactive =
        compile_vector_instruction(permlane_instruction(2, false), MetalShaderStage::Compute);
    assert!(zero_inactive.source.contains("? t"));
    assert!(zero_inactive.source.contains(": 0u"));
    assert!(!zero_inactive.source.contains("if (exec &&"));

    let fetch_inactive =
        compile_vector_instruction(permlane_instruction(1, true), MetalShaderStage::Compute);
    assert!(fetch_inactive.source.contains("(remu_lane & 16u) ^ 16u"));
    assert!(
        !fetch_inactive
            .source
            .contains("simd_shuffle(exec ? 1u : 0u")
    );
    assert!(fetch_inactive.source.contains("if (exec) remu_v_8 = t"));
}

#[test]
fn lowers_rdna2_bitcount_and_u64_mad_with_carry_mask() {
    let bcnt = compile_vector_instruction(
        Instruction {
            pc: 0x90,
            encoding: ShaderEncoding::Vop2,
            opcode: Opcode::Known("VBcntU32B32"),
            words: vec![0],
            sources: vec![Operand::vector(1), Operand::vector(2)],
            destinations: vec![Operand::vector(8)],
            control: None,
        },
        MetalShaderStage::Compute,
    );
    assert!(bcnt.source.contains("popcount(remu_v_1) + remu_v_2"));

    let mad = compile_vector_instruction(
        Instruction {
            pc: 0x98,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VMadU64U32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
            destinations: vec![Operand::vector(8), Operand::vector(9)],
            control: Some(InstructionControl::Vop3(remu_shader::Vop3Control {
                absolute_mask: 0,
                negate_mask: 0,
                output_modifier: 0,
                clamp: false,
                operand_select: 0,
                scalar_destination: Some(20),
            })),
        },
        MetalShaderStage::Compute,
    );
    assert!(
        mad.source
            .contains("((ulong)remu_v_3 | ((ulong)remu_v_4 << 32))")
    );
    assert!(mad.source.contains("(ulong)t"));
    assert!(mad.source.contains(" * (ulong)t"));
    assert!(mad.source.contains(" < t"));
    assert!(mad.source.contains("simd_ballot(exec && t"));
    assert!(mad.source.contains("remu_s_20 = "));
    assert!(mad.source.contains("remu_s_21 = "));
    assert!(mad.source.contains("if (exec) remu_v_8 = (uint)t"));
    assert!(mad.source.contains("if (exec) remu_v_9 = (uint)(t"));
    assert!(mad.source.contains(" >> 32)"));
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
    assert!(signed_extract.source.contains(">>"));
    assert!(signed_extract.source.contains("&"));
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

fn vop3p_instruction(name: &'static str, control: Vop3pControl) -> Instruction {
    Instruction {
        pc: 0,
        encoding: ShaderEncoding::Vop3p,
        opcode: Opcode::Known(name),
        words: vec![0, 0],
        sources: vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
        destinations: vec![Operand::vector(8)],
        control: Some(InstructionControl::Vop3p(control)),
    }
}

const DS_OPCODES: [(u32, &str); 39] = [
    (0x00, "DsAddU32"),
    (0x01, "DsSubU32"),
    (0x03, "DsIncU32"),
    (0x04, "DsDecU32"),
    (0x05, "DsMinI32"),
    (0x06, "DsMaxI32"),
    (0x07, "DsMinU32"),
    (0x08, "DsMaxU32"),
    (0x09, "DsAndB32"),
    (0x0a, "DsOrB32"),
    (0x0b, "DsXorB32"),
    (0x0d, "DsWriteB32"),
    (0x0e, "DsWrite2B32"),
    (0x0f, "DsWrite2St64B32"),
    (0x10, "DsCmpstB32"),
    (0x20, "DsAddRtnU32"),
    (0x21, "DsSubRtnU32"),
    (0x23, "DsIncRtnU32"),
    (0x24, "DsDecRtnU32"),
    (0x25, "DsMinRtnI32"),
    (0x26, "DsMaxRtnI32"),
    (0x27, "DsMinRtnU32"),
    (0x28, "DsMaxRtnU32"),
    (0x29, "DsAndRtnB32"),
    (0x2a, "DsOrRtnB32"),
    (0x2b, "DsXorRtnB32"),
    (0x2d, "DsWrxchgRtnB32"),
    (0x30, "DsCmpstRtnB32"),
    (0x35, "DsSwizzleB32"),
    (0x36, "DsReadB32"),
    (0x37, "DsRead2B32"),
    (0x38, "DsRead2St64B32"),
    (0x4d, "DsWriteB64"),
    (0xb0, "DsWriteAddtidB32"),
    (0xb1, "DsReadAddtidB32"),
    (0xde, "DsWriteB96"),
    (0xdf, "DsWriteB128"),
    (0xfe, "DsReadB96"),
    (0xff, "DsReadB128"),
];

fn decoded_ds_program(offset: u16, gds: bool) -> Program {
    let mut words = Vec::with_capacity(DS_OPCODES.len() * 2 + 1);
    for (opcode, _) in DS_OPCODES {
        words.push(0xd800_0000 | (opcode << 18) | (u32::from(gds) << 17) | u32::from(offset));
        words.push(1u32 | (2u32 << 8) | (3u32 << 16) | (8u32 << 24));
    }
    words.extend([0xbf8a_0000, 0xbf81_0000]);
    decode_program(0x1000, &words).expect("complete DS test program")
}

fn ds_swizzle_instruction(offset: u16) -> Instruction {
    Instruction {
        pc: 0,
        encoding: ShaderEncoding::Ds,
        opcode: Opcode::Known("DsSwizzleB32"),
        words: vec![0, 0],
        sources: vec![Operand::vector(2)],
        destinations: vec![Operand::vector(8)],
        control: Some(InstructionControl::DataShare(DataShareControl {
            offset0: u32::from(offset & 0xff),
            offset1: u32::from(offset >> 8),
            gds: false,
        })),
    }
}

fn compile_vector_instruction(instruction: Instruction, stage: MetalShaderStage) -> MetalShader {
    compile_msl(
        &Program::new(0x1000, vec![instruction]),
        &empty_evaluation(),
        &MetalCompileOptions {
            stage,
            required_vertex_output_count: 0,
            render_target_array_length: 1,
            pixel_outputs: Vec::new(),
            pixel_input_enable: 0,
            pixel_input_address: 0,
            pixel_input_controls: identity_pixel_input_controls(),
            diagnostic_force_visible_clip_depth: false,
            diagnostic_force_vertex_pattern: false,
            diagnostic_force_vertex_fetch_id: false,
            diagnostic_capture_vertex_position: false,
            diagnostic_capture_pixel_output: false,
            diagnostic_force_pixel_color: false,
            lds_size_bytes: 0,
            wave_lane_count: 32,
        },
    )
    .expect("vector instruction should lower")
}

const fn unmodified_vop3() -> InstructionControl {
    InstructionControl::Vop3(remu_shader::Vop3Control {
        absolute_mask: 0,
        negate_mask: 0,
        output_modifier: 0,
        clamp: false,
        operand_select: 0,
        scalar_destination: None,
    })
}

#[test]
fn lowers_observed_64_bit_and_not_scalar_operation() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0x30,
            encoding: ShaderEncoding::Sop2,
            opcode: Opcode::Known("SAndn2B64"),
            words: vec![0],
            sources: vec![Operand::scalar(10), Operand::scalar(12)],
            destinations: vec![Operand::scalar(20)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );

    assert!(shader.source.contains("& ~"));
    assert!(shader.source.contains("s[20]"));
    assert!(shader.source.contains("s[21]"));
}

#[test]
fn outputless_pixel_shader_lowers_to_void_depth_fragment() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop1,
            opcode: Opcode::Known("VMovB32"),
            words: vec![0],
            sources: vec![Operand::vector(1)],
            destinations: vec![Operand::vector(2)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );

    assert!(shader.source.contains("fragment void remu_main"));
    assert!(!shader.source.contains("RemuPsOut remu_out"));
    assert!(!shader.source.contains("return remu_out"));
}

#[test]
fn lowers_s_bcnt1_i32_b64_to_one_uint_destination_and_scc() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Sop1,
            opcode: Opcode::Known("SBcnt1I32B64"),
            words: vec![0xbe8a_1000],
            sources: vec![Operand::scalar(0)],
            destinations: vec![Operand::scalar(10)],
            control: None,
        },
        MetalShaderStage::Compute,
    );

    assert!(
        shader
            .source
            .contains("((ulong)remu_s_0 | ((ulong)remu_s_1 << 32))")
    );
    assert!(shader.source.contains("popcount((uint)"));
    assert!(shader.source.contains("popcount((uint)("));
    assert!(shader.source.contains(" >> 32))"));
    assert!(shader.source.contains("remu_s_10 = "));
    assert!(shader.source.contains("scc = "));
    assert!(shader.source.contains(" != 0u;"));
    assert!(
        !shader.source.contains("remu_s_11 = "),
        "the 32-bit destination must not be lowered as a B64 register pair"
    );
}

#[test]
fn lowers_s_bitcmp1_b32_as_an_indexed_bit_test() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Sopc,
            opcode: Opcode::Known("SBitcmp1B32"),
            words: vec![0],
            sources: vec![Operand::scalar(10), Operand::scalar(11)],
            destinations: Vec::new(),
            control: None,
        },
        MetalShaderStage::Compute,
    );

    assert!(shader.source.contains("remu_s_10"));
    assert!(shader.source.contains("remu_s_11"));
    assert!(shader.source.contains("& 31u"));
    assert!(shader.source.contains(">>"));
    assert!(shader.source.contains("& 1u"));
    assert!(shader.source.contains("scc = "));
}

#[test]
fn lowers_scalar_absdiff_with_wrapping_i32_arithmetic_and_scc() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Sop2,
            opcode: Opcode::Known("SAbsdiffI32"),
            words: vec![0],
            sources: vec![Operand::scalar(10), Operand::scalar(11)],
            destinations: vec![Operand::scalar(12)],
            control: None,
        },
        MetalShaderStage::Compute,
    );

    assert!(shader.source.contains("remu_s_10 - remu_s_11"));
    assert!(shader.source.contains("as_type<int>("));
    assert!(shader.source.contains("? 0u - "));
    assert!(shader.source.contains("scc = "));
    assert!(shader.source.contains(" != 0u;"));
}

#[test]
fn lowers_observed_bit_reverse_and_wave32_saveexec_operations() {
    let reverse = compile_vector_instruction(
        Instruction {
            pc: 0x194,
            encoding: ShaderEncoding::Vop1,
            opcode: Opcode::Known("VBfrevB32"),
            words: vec![0],
            sources: vec![Operand::vector(2)],
            destinations: vec![Operand::vector(8)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );
    assert!(reverse.source.contains("reverse_bits(remu_v_2)"));

    let saveexec = compile_vector_instruction(
        Instruction {
            pc: 0x74,
            encoding: ShaderEncoding::Sop1,
            opcode: Opcode::Known("SAndSaveexecB64"),
            words: vec![0],
            sources: vec![Operand::scalar(10)],
            destinations: vec![Operand::scalar(20)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );
    assert!(saveexec.source.contains("simd_ballot(exec)"));
    assert!(saveexec.source.contains("& 0xFFFFFFFFul"));
    assert!(saveexec.source.contains("remu_s_20 = (uint)"));
    assert!(saveexec.source.contains("exec = (("));
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
    }
}

#[test]
fn lowers_unsigned_64_bit_scalar_compare_from_register_pairs() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0x1c0,
            encoding: ShaderEncoding::Sopc,
            opcode: Opcode::Known("SCmpLgU64"),
            words: vec![0],
            sources: vec![Operand::scalar(106), Operand::scalar(30)],
            destinations: Vec::new(),
            control: None,
        },
        MetalShaderStage::Pixel,
    );

    assert!(
        shader
            .source
            .contains("((ulong)remu_s_106 | ((ulong)remu_s_107 << 32))")
    );
    assert!(
        shader
            .source
            .contains("((ulong)remu_s_30 | ((ulong)remu_s_31 << 32))")
    );
    assert!(shader.source.contains("!="));
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
fn lowers_structured_wave64_saveexec_into_the_correct_simd32_half() {
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0x7cc,
            encoding: ShaderEncoding::Sop1,
            opcode: Opcode::Known("SAndSaveexecB64"),
            words: vec![0],
            sources: vec![Operand::scalar(10)],
            destinations: vec![Operand::scalar(20)],
            control: None,
        }],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute_wave(0, 64),
    )
    .expect("split wave64 saveexec should lower");

    assert!(
        shader
            .source
            .contains("uint remu_thread_index [[thread_index_in_threadgroup]]")
    );
    assert!(shader.source.contains("uint remu_wave_half ="));
    assert!(shader.source.contains("remu_wave_half * 32u"));
    assert!(shader.source.contains("0xFFFFFFFFFFFFFFFFul"));
    assert!(shader.source.contains("remu_s_127 = (uint)(t"));
    assert!(shader.source.contains("remu_wave_half == 0u ?"));
}

#[test]
fn sign_extends_negative_inline_constants_for_wave64_exec_restores() {
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0x164,
            encoding: ShaderEncoding::Sop1,
            opcode: Opcode::Known("SMovB64"),
            words: vec![0],
            sources: vec![Operand::source(193, None)],
            destinations: vec![Operand::scalar(EXEC_LO)],
            control: None,
        }],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute_wave(0, 64),
    )
    .expect("wave64 EXEC=-1 restore should lower");

    assert!(shader.source.contains("0xFFFFFFFFFFFFFFFFul"));
    assert!(
        !shader.source.contains("(ulong)0xFFFFFFFFu"),
        "a zero-extended -1 disables the high 32 lanes"
    );
    assert!(shader.source.contains("remu_s_127 = (uint)("));
}

#[test]
fn synchronizes_split_wave64_lds_at_full_exec_reconvergence() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0x100,
                encoding: ShaderEncoding::Ds,
                opcode: Opcode::Known("DsWriteB32"),
                words: vec![0, 0],
                sources: vec![Operand::vector(0), Operand::vector(1)],
                destinations: Vec::new(),
                control: Some(InstructionControl::DataShare(DataShareControl {
                    offset0: 0,
                    offset1: 0,
                    gds: false,
                })),
            },
            Instruction {
                pc: 0x108,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::Known("SMovB64"),
                words: vec![0],
                sources: vec![Operand::source(193, None)],
                destinations: vec![Operand::scalar(EXEC_LO)],
                control: None,
            },
        ],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute_wave(2_048, 64),
    )
    .expect("split wave64 LDS reconvergence should lower");

    assert!(
        shader
            .source
            .contains("threadgroup_barrier(mem_flags::mem_threadgroup)")
    );
}

#[test]
fn wave64_explicit_lane_masks_select_the_matching_sgpr_half() {
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0x258,
            encoding: ShaderEncoding::Vop2,
            opcode: Opcode::Known("VCndmaskB32"),
            words: vec![0],
            sources: vec![Operand::vector(1), Operand::vector(2), Operand::scalar(10)],
            destinations: vec![Operand::vector(3)],
            control: None,
        }],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute_wave(0, 64),
    )
    .expect("wave64 explicit lane mask should lower");

    assert!(shader.source.contains("remu_wave_half == 0u ?"));
    assert!(shader.source.contains("remu_s_10"));
    assert!(shader.source.contains("remu_s_11"));
}

#[test]
fn lowers_wave64_readfirstlane_across_both_simd32_halves() {
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0xec,
            encoding: ShaderEncoding::Vop1,
            opcode: Opcode::Known("VReadfirstlaneB32"),
            words: vec![0],
            sources: vec![Operand::vector(4)],
            destinations: vec![Operand::scalar(44)],
            control: None,
        }],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions {
            stage: MetalShaderStage::Compute,
            required_vertex_output_count: 0,
            render_target_array_length: 1,
            pixel_outputs: Vec::new(),
            pixel_input_enable: 0,
            pixel_input_address: 0,
            pixel_input_controls: identity_pixel_input_controls(),
            diagnostic_force_visible_clip_depth: false,
            diagnostic_force_vertex_pattern: false,
            diagnostic_force_vertex_fetch_id: false,
            diagnostic_capture_vertex_position: false,
            diagnostic_capture_pixel_output: false,
            diagnostic_force_pixel_color: false,
            lds_size_bytes: 2_048,
            wave_lane_count: 64,
        },
    )
    .expect("wave64 readfirstlane should lower with threadgroup exchange");

    assert!(
        shader
            .source
            .contains("threadgroup uint remu_readfirstlane_values[32]")
    );
    assert!(
        shader
            .source
            .contains("threadgroup_barrier(mem_flags::mem_threadgroup)")
    );
    assert!(shader.source.contains("remu_readfirstlane_active"));
    assert!(shader.source.contains("remu_s_44 ="));
}

#[test]
fn lowers_readlane_and_writelane_independently_of_exec() {
    let read = compile_vector_instruction(
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VReadlaneB32"),
            words: vec![0, 0],
            sources: vec![Operand::vector(2), Operand::scalar(4)],
            destinations: vec![Operand::scalar(8)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );
    assert!(
        read.source
            .contains("simd_shuffle(remu_v_2, (ushort)(remu_s_4 & 31u))")
    );
    assert!(read.source.contains("remu_s_8 = "));

    let write = compile_vector_instruction(
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VWritelaneB32"),
            words: vec![0, 0],
            sources: vec![Operand::scalar(2), Operand::scalar(4)],
            destinations: vec![Operand::vector(8)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );
    assert!(
        write
            .source
            .contains("if (remu_lane == ((remu_s_4) & 31u)) remu_v_8 = remu_s_2;")
    );
    assert!(!write.source.contains("if (exec) remu_v_8 ="));
}

#[test]
fn lowers_wave64_readlane_and_writelane_across_simd32_halves() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Vop3,
                opcode: Opcode::Known("VReadlaneB32"),
                words: vec![0, 0],
                sources: vec![Operand::vector(2), Operand::scalar(4)],
                destinations: vec![Operand::scalar(8)],
                control: Some(unmodified_vop3()),
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Vop3,
                opcode: Opcode::Known("VWritelaneB32"),
                words: vec![0, 0],
                sources: vec![Operand::scalar(2), Operand::scalar(4)],
                destinations: vec![Operand::vector(8)],
                control: Some(unmodified_vop3()),
            },
        ],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute_wave(0, 64),
    )
    .expect("wave64 explicit lane operations should lower");

    assert!(
        shader
            .source
            .contains("threadgroup uint remu_readlane_values[1024]")
    );
    assert!(
        shader
            .source
            .contains("remu_readlane_values[remu_thread_index] = remu_v_2;")
    );
    assert!(shader.source.contains("remu_wave_index << 6u"));
    assert!(shader.source.contains("remu_s_4) & 63u"));
    assert!(
        shader
            .source
            .contains("if (remu_wave_lane == ((remu_s_4) & 63u)) remu_v_8 = remu_s_2;")
    );
}

#[test]
fn lowers_masked_bit_count_for_wave32_and_both_wave64_halves() {
    let low = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Vop2,
        opcode: Opcode::Known("VMbcntLoU32B32"),
        words: vec![0],
        sources: vec![Operand::scalar(2), Operand::vector(1)],
        destinations: vec![Operand::vector(1)],
        control: None,
    };
    let high = Instruction {
        pc: 4,
        encoding: ShaderEncoding::Vop2,
        opcode: Opcode::Known("VMbcntHiU32B32"),
        words: vec![0],
        sources: vec![Operand::scalar(2), Operand::vector(1)],
        destinations: vec![Operand::vector(1)],
        control: None,
    };
    let wave32 = compile_vector_instruction(low.clone(), MetalShaderStage::Compute);
    assert!(wave32.source.contains("popcount((remu_s_2) & (t"));
    let wave32_high = compile_vector_instruction(high.clone(), MetalShaderStage::Compute);
    assert!(wave32_high.source.contains("& (0u)"));

    let wave64 = compile_msl(
        &Program::new(0x1000, vec![low, high]),
        &empty_evaluation(),
        &MetalCompileOptions::compute_wave(0, 64),
    )
    .expect("wave64 masked bit count should lower");
    assert!(
        wave64
            .source
            .contains("remu_wave_half == 0u ? t0 : 0xFFFFFFFFu")
    );
    assert!(wave64.source.contains("remu_wave_half == 0u ? 0u : t"));
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
    assert!(shader.source.contains("simd_ballot"));
    assert!(shader.source.contains("remu_s_12 ="));
    assert!(shader.source.contains("remu_s_106 ="));
}

#[test]
fn lowers_scalar_bitfield_mask_operation() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0x220,
            encoding: ShaderEncoding::Sop2,
            opcode: Opcode::Known("SBfmB32"),
            words: vec![0],
            sources: vec![Operand::scalar(10), Operand::scalar(12)],
            destinations: vec![Operand::scalar(20)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );

    assert!(shader.source.contains("1u << ("));
    assert!(shader.source.contains("& 31u)) - 1u) << ("));
    assert!(shader.source.contains("remu_s_20"));
}

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

#[test]
fn conditional_control_flow_keeps_exact_dispatcher_but_not_register_arrays() {
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
    .expect("conditional program should retain dispatcher");

    assert!(!shader.register_footprint.direct_control_flow);
    assert!(shader.source.contains("while (active)"));
    assert!(shader.source.contains("switch (pc)"));
    assert!(!shader.source.contains("uint s[128]"));
    assert!(!shader.source.contains("uint v[256]"));
}

#[test]
fn dispatcher_switches_once_per_basic_block_instead_of_once_per_instruction() {
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
        assert!(shader.source.contains(&format!("case {pc}u")));
    }
    for pc in [8, 12] {
        assert!(!shader.source.contains(&format!("case {pc}u")));
    }
    assert_eq!(shader.source.matches("case ").count(), 4);
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

const BUFFER_MEMORY_OPCODES: [&str; 60] = [
    "BufferLoadFormatX",
    "BufferLoadFormatXy",
    "BufferLoadFormatXyz",
    "BufferLoadFormatXyzw",
    "BufferStoreFormatX",
    "BufferStoreFormatXy",
    "BufferStoreFormatXyz",
    "BufferStoreFormatXyzw",
    "BufferLoadUbyte",
    "BufferLoadSbyte",
    "BufferLoadUshort",
    "BufferLoadSshort",
    "BufferLoadDword",
    "BufferLoadDwordx2",
    "BufferLoadDwordx4",
    "BufferLoadDwordx3",
    "BufferStoreByte",
    "BufferStoreByteD16Hi",
    "BufferStoreShort",
    "BufferStoreShortD16Hi",
    "BufferStoreDword",
    "BufferStoreDwordx2",
    "BufferStoreDwordx4",
    "BufferStoreDwordx3",
    "BufferLoadUbyteD16",
    "BufferLoadUbyteD16Hi",
    "BufferLoadSbyteD16",
    "BufferLoadSbyteD16Hi",
    "BufferLoadShortD16",
    "BufferLoadShortD16Hi",
    "BufferLoadFormatD16HiX",
    "BufferStoreFormatD16HiX",
    "BufferAtomicSwap",
    "BufferAtomicCmpswap",
    "BufferAtomicAdd",
    "BufferAtomicSub",
    "BufferAtomicSmin",
    "BufferAtomicUmin",
    "BufferAtomicSmax",
    "BufferAtomicUmax",
    "BufferAtomicAnd",
    "BufferAtomicOr",
    "BufferAtomicXor",
    "BufferAtomicInc",
    "BufferAtomicDec",
    "BufferLoadFormatD16X",
    "BufferLoadFormatD16Xy",
    "BufferLoadFormatD16Xyz",
    "BufferLoadFormatD16Xyzw",
    "BufferStoreFormatD16X",
    "BufferStoreFormatD16Xy",
    "BufferStoreFormatD16Xyz",
    "BufferStoreFormatD16Xyzw",
    "TBufferLoadFormatX",
    "TBufferLoadFormatXy",
    "TBufferLoadFormatXyz",
    "TBufferLoadFormatXyzw",
    "TBufferStoreFormatX",
    "TBufferStoreFormatXy",
    "TBufferStoreFormatXyz",
];

const EXTRA_BUFFER_MEMORY_OPCODE: &str = "TBufferStoreFormatXyzw";
const TYPED_D16_MEMORY_OPCODES: [&str; 8] = [
    "TBufferLoadFormatD16X",
    "TBufferLoadFormatD16Xy",
    "TBufferLoadFormatD16Xyz",
    "TBufferLoadFormatD16Xyzw",
    "TBufferStoreFormatD16X",
    "TBufferStoreFormatD16Xy",
    "TBufferStoreFormatD16Xyz",
    "TBufferStoreFormatD16Xyzw",
];

const GLOBAL_MEMORY_OPCODES: [&str; 24] = [
    "GlobalLoadUbyte",
    "GlobalLoadSbyte",
    "GlobalLoadUshort",
    "GlobalLoadSshort",
    "GlobalLoadDword",
    "GlobalLoadDwordx2",
    "GlobalLoadDwordx4",
    "GlobalLoadDwordx3",
    "GlobalStoreByte",
    "GlobalStoreByteD16Hi",
    "GlobalStoreShort",
    "GlobalStoreShortD16Hi",
    "GlobalStoreDword",
    "GlobalStoreDwordx2",
    "GlobalStoreDwordx4",
    "GlobalStoreDwordx3",
    "GlobalLoadUbyteD16",
    "GlobalLoadUbyteD16Hi",
    "GlobalLoadSbyteD16",
    "GlobalLoadSbyteD16Hi",
    "GlobalLoadShortD16",
    "GlobalLoadShortD16Hi",
    "GlobalAtomicAdd",
    "GlobalAtomicUMax",
];

fn memory_dword_count_for_test(name: &str) -> u32 {
    if name.contains("FormatD16") {
        if name.ends_with("Xyz") || name.ends_with("Xyzw") {
            2
        } else {
            1
        }
    } else if name.ends_with("Xyzw") || name.ends_with("Dwordx4") {
        4
    } else if name.ends_with("Xyz") || name.ends_with("Dwordx3") {
        3
    } else if name.ends_with("Xy") || name.ends_with("Dwordx2") || name.ends_with("Cmpswap") {
        2
    } else {
        1
    }
}

fn complete_memory_program() -> (Program, ShaderEvaluation) {
    let names = BUFFER_MEMORY_OPCODES
        .into_iter()
        .chain([EXTRA_BUFFER_MEMORY_OPCODE])
        .chain(TYPED_D16_MEMORY_OPCODES)
        .chain(GLOBAL_MEMORY_OPCODES);
    let mut pcs = Vec::new();
    let mut instructions = names
        .enumerate()
        .map(|(index, name)| {
            let pc = u32::try_from(index * 8).expect("small memory test program");
            pcs.push(pc);
            let dword_count = memory_dword_count_for_test(name);
            let (encoding, control) = if name.starts_with("Global") {
                (
                    ShaderEncoding::Flat,
                    InstructionControl::GlobalMemory(remu_shader::GlobalMemoryControl {
                        dword_count,
                        vector_address: 1,
                        vector_data: 16,
                        vector_destination: 32,
                        scalar_address: Some(8),
                        offset_bytes: -4,
                        dlc: true,
                        lds: false,
                        glc: true,
                        slc: true,
                    }),
                )
            } else {
                (
                    if name.starts_with("TBuffer") {
                        ShaderEncoding::Mtbuf
                    } else {
                        ShaderEncoding::Mubuf
                    },
                    InstructionControl::BufferMemory(remu_shader::BufferMemoryControl {
                        dword_count,
                        vector_address: 1,
                        vector_data: 16,
                        scalar_resource: 0,
                        instruction_format: name.starts_with("TBuffer").then_some(60),
                        offset_bytes: 4,
                        index_enabled: true,
                        offset_enabled: true,
                        dlc: true,
                        lds: false,
                        glc: true,
                        slc: true,
                        tfe: name.contains("Load") && !name.contains("Atomic"),
                    }),
                )
            };
            Instruction {
                pc,
                encoding,
                opcode: Opcode::Known(name),
                words: vec![0, 0],
                sources: vec![
                    Operand::vector(1),
                    Operand::scalar(0),
                    Operand::source(128, None),
                ],
                destinations: Vec::new(),
                control: Some(control),
            }
        })
        .collect::<Vec<_>>();
    let image_pc = u32::try_from(instructions.len() * 8).expect("small memory test program");
    let image_control = remu_shader::ImageControl {
        dmask: 0xf,
        vector_address: 1,
        address_registers: vec![1, 2],
        vector_data: 64,
        scalar_resource: 16,
        scalar_sampler: 24,
        dimension: 1,
        is_array: false,
        glc: true,
        slc: true,
        a16: false,
        d16: false,
    };
    instructions.push(Instruction {
        pc: image_pc,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageLoad"),
        words: vec![0, 0],
        sources: vec![Operand::vector(1), Operand::vector(2), Operand::scalar(16)],
        destinations: (64..68).map(Operand::vector).collect(),
        control: Some(InstructionControl::Image(image_control.clone())),
    });
    let image_store_pc = image_pc + 8;
    let mut image_store_control = image_control.clone();
    image_store_control.vector_data = 68;
    image_store_control.dimension = 5;
    image_store_control.is_array = true;
    instructions.push(Instruction {
        pc: image_store_pc,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageStore"),
        words: vec![0, 0],
        sources: vec![
            Operand::vector(1),
            Operand::vector(2),
            Operand::scalar(16),
            Operand::vector(68),
            Operand::vector(69),
            Operand::vector(70),
            Operand::vector(71),
        ],
        destinations: Vec::new(),
        control: Some(InstructionControl::Image(image_store_control.clone())),
    });
    let end_pc = u32::try_from(instructions.len() * 8).expect("small memory test program");
    instructions.push(Instruction {
        pc: end_pc,
        encoding: ShaderEncoding::Sopp,
        opcode: Opcode::END_PROGRAM,
        words: vec![0xbf81_0000],
        sources: Vec::new(),
        destinations: Vec::new(),
        control: None,
    });
    let mut evaluation = empty_evaluation();
    let selectors = 4 | (5 << 3) | (6 << 6) | (7 << 9);
    evaluation.initial_scalar_registers[0] = 0x1000;
    evaluation.initial_scalar_registers[1] = 16 << 16;
    evaluation.initial_scalar_registers[2] = 64 * 1024;
    evaluation.initial_scalar_registers[3] = selectors | (60 << 12) | (1 << 24) | (3 << 28);
    evaluation.initial_scalar_registers[8] = 0x1000;
    evaluation.scalar_registers = evaluation.initial_scalar_registers.clone();
    evaluation
        .global_memory_bindings
        .push(remu_shader::GlobalMemoryBinding {
            scalar_address: 0,
            base_address: 0x1000,
            instruction_pcs: pcs.into(),
            byte_count: 64 * 1024,
            data: vec![0; 64 * 1024],
            writable: true,
            write_back_to_guest: true,
        });
    let mut identity_image_descriptor = vec![0; 8];
    identity_image_descriptor[3] = 0xfac;
    evaluation.image_bindings.push(remu_shader::ImageBinding {
        pc: image_pc,
        opcode: Opcode::Known("ImageLoad"),
        control: image_control,
        resource_descriptor: identity_image_descriptor.clone().into(),
        sampler_descriptor: Vec::new().into(),
        mip_level: None,
    });
    let mut swizzled_store_descriptor = identity_image_descriptor;
    swizzled_store_descriptor[3] = 0x9f5;
    evaluation.image_bindings.push(remu_shader::ImageBinding {
        pc: image_store_pc,
        opcode: Opcode::Known("ImageStore"),
        control: image_store_control,
        resource_descriptor: swizzled_store_descriptor.into(),
        sampler_descriptor: Vec::new().into(),
        mip_level: None,
    });
    (Program::new(0x1000, instructions), evaluation)
}

#[test]
fn lowers_complete_decoded_global_mubuf_and_mtbuf_memory_surface() {
    let (program, evaluation) = complete_memory_program();
    let shader = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
        .expect("complete decoded VMEM surface should lower");
    for operation in [
        "remu_atomic_cmpswap",
        "remu_atomic_inc",
        "remu_store_format_bits",
        "remu_load_format_bits",
        "remu_format_to_d16",
        "remu_format_from_d16",
    ] {
        assert!(shader.source.contains(operation), "missing {operation}");
    }
    assert!(shader.source.contains("0xFFFF0000u"));
    assert!(shader.source.contains("0xFFFFFF00u"));
    assert!(shader.source.contains("remu_range_valid"));
    assert!(
        shader
            .source
            .contains("texture2d<float, access::read> tex0")
    );
    assert!(
        shader
            .source
            .contains("texture2d<float, access::read_write> tex1")
    );
    assert!(shader.source.contains("tex0.read(uint2("));
    assert!(shader.source.contains("tex1.write("));
    assert!(
            shader.source.contains(
                "vec<float, 4>(as_type<float>(remu_v_68), as_type<float>(remu_v_69), as_type<float>(remu_v_70), as_type<float>(remu_v_71))"
            ),
            "image stores must map enabled DMASK components directly to physical RGBA channels"
        );
    assert!(
        !shader.source.contains("clamp(as_type<int>"),
        "out-of-range image loads must return zero instead of sampling a clamped edge texel"
    );
    assert!(
        shader.source.contains("= vec<float, 4>(0.0f);"),
        "image loads must initialize an architectural zero result before their bounds-checked read"
    );
    assert!(
        shader.source.contains(" & 3ul"),
        "buffer descriptors must address their dynamically bound, dword-aligned Metal buffer without embedding the evaluated guest base"
    );
}

#[test]
fn sampled_images_apply_dynamic_descriptor_component_selection_in_the_shader() {
    let control = ImageControl {
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
    let instruction = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageSample"),
        words: vec![0, 0],
        sources: vec![Operand::vector(0), Operand::vector(1), Operand::scalar(0)],
        destinations: (4..8).map(Operand::vector).collect(),
        control: Some(InstructionControl::Image(control.clone())),
    };
    let mut evaluation = empty_evaluation();
    let mut resource_descriptor = vec![0; 8];
    // R32F sampled as R,R,R,R, the live Superliminal descriptor that previously failed
    // because the host tried to materialize the swizzle by changing the texture format.
    resource_descriptor[3] = 0x924;
    evaluation.image_bindings.push(ImageBinding {
        pc: 0,
        opcode: Opcode::Known("ImageSample"),
        control,
        resource_descriptor: resource_descriptor.into(),
        sampler_descriptor: vec![0; 4].into(),
        mip_level: None,
    });
    let shader = compile_msl(
        &Program::new(0x1000, vec![instruction]),
        &evaluation,
        &MetalCompileOptions::compute(0),
    )
    .expect("sampled R32 swizzle should lower");
    assert!(shader.source.contains("remu_s_3 & 0xFFFu"));
    assert!(shader.source.contains("== 0u ? 0xFACu"));
    assert!(shader.source.contains(">= 4u &&"));
    assert!(shader.source.contains("<= 7u ?"));
    assert!(shader.source.contains("tex0.sample("));
}

#[test]
fn zero_lod_gather_returns_four_neighbor_values_from_the_selected_component() {
    let control = ImageControl {
        dmask: 4,
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
    let instruction = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageGather4Lz"),
        words: vec![0, 0],
        sources: vec![Operand::vector(0), Operand::vector(1), Operand::scalar(0)],
        destinations: (4..8).map(Operand::vector).collect(),
        control: Some(InstructionControl::Image(control.clone())),
    };
    let mut evaluation = empty_evaluation();
    let mut resource_descriptor = vec![0; 8];
    resource_descriptor[3] = 0xfac | (9 << 28);
    evaluation.image_bindings.push(ImageBinding {
        pc: 0,
        opcode: Opcode::Known("ImageGather4Lz"),
        control,
        resource_descriptor: resource_descriptor.into(),
        sampler_descriptor: vec![0; 4].into(),
        mip_level: Some(0),
    });

    let shader = compile_msl(
        &Program::new(0x1000, vec![instruction]),
        &evaluation,
        &MetalCompileOptions::compute(0),
    )
    .expect("zero-LOD image gather should lower");

    for component in ["x", "y", "z", "w"] {
        assert!(
            shader
                .source
                .contains(&format!("int2(0), component::{component}")),
            "missing physical component gather {component}"
        );
    }
    assert!(shader.source.contains("remu_s_3 & 0xFFFu"));
    assert!(shader.source.contains("remu_v_4 = "));
    assert!(shader.source.contains("remu_v_7 = "));
}

#[test]
fn canonicalizes_silent_hill_scale_image_uses_below_the_direct_texture_limit() {
    const IMAGE_INSTRUCTION_COUNT: usize = 147;
    let control = ImageControl {
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
    let mut instructions = Vec::with_capacity(IMAGE_INSTRUCTION_COUNT);
    let mut evaluation = empty_evaluation();
    let mut resource_descriptor = vec![0; 8];
    resource_descriptor[3] = 0xfac | (9 << 28);

    for index in 0..IMAGE_INSTRUCTION_COUNT {
        let pc = u32::try_from(index * 8).expect("test PC fits u32");
        instructions.push(Instruction {
            pc,
            encoding: ShaderEncoding::Mimg,
            opcode: Opcode::Known("ImageSampleLz"),
            words: vec![0, 0],
            sources: vec![Operand::vector(0), Operand::vector(1), Operand::scalar(0)],
            destinations: (4..8).map(Operand::vector).collect(),
            control: Some(InstructionControl::Image(control.clone())),
        });
        evaluation.image_bindings.push(ImageBinding {
            pc,
            opcode: Opcode::Known("ImageSampleLz"),
            control: control.clone(),
            resource_descriptor: resource_descriptor.clone().into(),
            sampler_descriptor: vec![0; 4].into(),
            mip_level: Some(0),
        });
    }

    let program = Program::new(0x800c_1100_00, instructions);
    let shader = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
        .expect("repeated descriptor uses must share one Metal binding");

    assert_eq!(evaluation.image_bindings.len(), IMAGE_INSTRUCTION_COUNT);
    assert_eq!(shader.bindings.texture_count, 1);
    assert_eq!(shader.bindings.sampler_count, 1);
    assert_eq!(shader.source.matches("[[texture(").count(), 1);
    assert!(shader.source.contains("tex0 [[texture(0)]]"));
    assert!(!shader.source.contains("[[texture(128)]]"));
    assert!(!shader.source.contains("tex128"));

    let mut genuinely_distinct = evaluation.clone();
    for (index, binding) in genuinely_distinct.image_bindings.iter_mut().enumerate() {
        binding.resource_descriptor[0] =
            u32::try_from(index).expect("test descriptor index fits u32");
    }
    let error = compile_msl(
        &program,
        &genuinely_distinct,
        &MetalCompileOptions::compute(0),
    )
    .expect_err("129+ genuinely distinct textures must be rejected, not dropped");
    assert!(error.to_string().contains("distinct image descriptors"));
}

#[test]
fn comparison_gather_uses_distinct_depth_resources_and_gather_compare_codegen() {
    let ordinary_control = ImageControl {
        dmask: 1,
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
    let comparison_control = ImageControl {
        address_registers: vec![0, 1, 2],
        vector_data: 8,
        ..ordinary_control.clone()
    };
    let ordinary = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageGather4"),
        words: vec![0, 0],
        sources: vec![Operand::vector(0), Operand::vector(1), Operand::scalar(0)],
        destinations: (4..8).map(Operand::vector).collect(),
        control: Some(InstructionControl::Image(ordinary_control.clone())),
    };
    let comparison = Instruction {
        pc: 8,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageGather4C"),
        words: vec![0, 0],
        sources: vec![
            Operand::vector(0),
            Operand::vector(1),
            Operand::vector(2),
            Operand::scalar(0),
            Operand::scalar(8),
        ],
        destinations: (8..12).map(Operand::vector).collect(),
        control: Some(InstructionControl::Image(comparison_control.clone())),
    };
    let mut evaluation = empty_evaluation();
    let mut resource_descriptor = vec![0; 8];
    resource_descriptor[3] = 0x204 | (9 << 28);
    for (instruction, control) in [
        (&ordinary, ordinary_control),
        (&comparison, comparison_control),
    ] {
        evaluation.image_bindings.push(ImageBinding {
            pc: instruction.pc,
            opcode: instruction.opcode,
            control,
            resource_descriptor: resource_descriptor.clone().into(),
            sampler_descriptor: vec![3 << 12, 0, 0, 0].into(),
            mip_level: None,
        });
    }

    let shader = compile_msl(
        &Program::new(0x1000, vec![ordinary.clone(), comparison.clone()]),
        &evaluation,
        &MetalCompileOptions::compute(0),
    )
    .expect("ordinary and comparison gathers should lower");

    assert_eq!(shader.bindings.texture_count, 2);
    assert_eq!(shader.bindings.sampler_count, 2);
    assert!(
        shader
            .source
            .contains("texture2d<float> tex0 [[texture(0)]]")
    );
    assert!(shader.source.contains("depth2d<float> tex1 [[texture(1)]]"));
    assert!(shader.source.contains("tex0.gather(remu_samplers.s0"));
    assert!(
        shader
            .source
            .contains("tex1.gather_compare(remu_samplers.s1")
    );

    for index in 0..30_u32 {
        evaluation
            .global_memory_bindings
            .push(remu_shader::GlobalMemoryBinding {
                scalar_address: index * 4,
                base_address: 0x10_0000 + u64::from(index) * 0x1000,
                instruction_pcs: Vec::new().into(),
                byte_count: 4,
                data: vec![0; 4],
                writable: false,
                write_back_to_guest: false,
            });
    }
    let direct = compile_msl(
        &Program::new(0x1000, vec![ordinary.clone(), comparison.clone()]),
        &evaluation,
        &MetalCompileOptions::compute(0),
    )
    .expect("samplers should fall back to direct stage bindings at Metal's buffer limit");
    assert_eq!(direct.bindings.initial_scalar_buffer_index, 30);
    assert_eq!(direct.bindings.sampler_argument_buffer_index, None);
    assert!(
        direct
            .source
            .contains("sampler remu_sampler_0 [[sampler(0)]]")
    );
    assert!(direct.source.contains("tex0.gather(remu_sampler_0"));
    assert!(
        !direct
            .source
            .contains("RemuSamplers& remu_samplers [[buffer(31)]]")
    );

    evaluation
        .global_memory_bindings
        .push(remu_shader::GlobalMemoryBinding {
            scalar_address: 30 * 4,
            base_address: 0x20_0000,
            instruction_pcs: Vec::new().into(),
            byte_count: 4,
            data: vec![0; 4],
            writable: false,
            write_back_to_guest: false,
        });
    let packed = compile_msl(
        &Program::new(0x1000, vec![ordinary, comparison]),
        &evaluation,
        &MetalCompileOptions::compute(0),
    )
    .expect("large global interfaces should use one Metal argument buffer");
    assert_eq!(packed.bindings.global_buffer_count, 31);
    assert_eq!(packed.bindings.global_argument_buffer_index, Some(0));
    assert_eq!(packed.bindings.initial_scalar_buffer_index, 1);
    assert_eq!(packed.bindings.sampler_argument_buffer_index, Some(2));
    assert!(
        packed
            .source
            .contains("constant RemuGlobalBuffers& remu_globals [[buffer(0)]]")
    );
    assert!(
        packed
            .source
            .contains("const device uchar* b30 = remu_globals.b30;")
    );
    assert!(packed.source.contains("tex0.gather(remu_samplers.s0"));
    assert!(
        packed
            .source
            .contains("tex1.gather_compare(remu_samplers.s1")
    );
    assert!(!packed.source.contains("tex0.gather(remu_sampler_0"));
    assert!(!packed.source.contains("[[buffer(31)]]"));
}

#[test]
fn comparison_samples_use_depth_texture_typing_and_the_z_reference_operand() {
    let control = ImageControl {
        dmask: 1,
        vector_address: 0,
        address_registers: vec![0, 1, 2],
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
    let instruction = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageSampleCLz"),
        words: vec![0, 0],
        sources: vec![
            Operand::vector(0),
            Operand::vector(1),
            Operand::vector(2),
            Operand::scalar(0),
            Operand::scalar(8),
        ],
        destinations: vec![Operand::vector(4)],
        control: Some(InstructionControl::Image(control.clone())),
    };
    let mut evaluation = empty_evaluation();
    let mut resource_descriptor = vec![0; 8];
    resource_descriptor[3] = 0x204 | (9 << 28);
    evaluation.image_bindings.push(ImageBinding {
        pc: 0,
        opcode: Opcode::Known("ImageSampleCLz"),
        control,
        resource_descriptor: resource_descriptor.into(),
        sampler_descriptor: vec![3 << 12, 0, 0, 0].into(),
        mip_level: None,
    });
    let shader = compile_msl(
        &Program::new(0x1000, vec![instruction]),
        &evaluation,
        &MetalCompileOptions::pixel(Vec::new(), 0, 0),
    )
    .expect("depth comparison sample should lower");

    assert!(shader.source.contains("depth2d<float> tex0"));
    assert!(shader.source.contains("tex0.sample_compare("));
    assert!(shader.source.contains("as_type<float>(remu_v_0)"));
    assert!(
        shader
            .source
            .contains("float2(as_type<float>(remu_v_1), as_type<float>(remu_v_2))")
    );
    assert!(shader.source.contains(", level(0.0f))"));
}

#[test]
fn range_checks_raw_dword_vectors_per_component() {
    let expression = buffer_oob_expression(
        "null",
        "mode",
        "index",
        "records",
        "offset",
        "stride",
        "swizzled",
        "buffer_offset",
        "reduced_records",
        "4u",
        12,
    );

    assert!(expression.contains("offset + 12u >= stride"));
    assert!(expression.contains("remu_range_valid(offset + 12u, 4u, stride)"));
    assert!(expression.contains("remu_range_valid(buffer_offset + 12u, 4u, reduced_records)"));
}

#[test]
fn formatted_elements_use_record_stride_without_total_size_alignment() {
    let expression =
        format_buffer_element_oob_expression("descriptor_oob", "record_address", "12u", "28u");

    assert_eq!(
        expression,
        "descriptor_oob || 12u == 0u || !remu_range_valid(record_address, 12u, 28u)"
    );
    assert!(
        !expression.contains('%'),
        "a 12-byte float3 at offsets 0, 28, and 56 is legal in a 28-byte record layout"
    );
}

#[test]
fn initializes_complete_rdna2_compute_system_register_state() {
    let program = Program::new(
        0x1000,
        [4_u32, 5, 6, 7]
            .into_iter()
            .enumerate()
            .map(|(index, source)| Instruction {
                pc: u32::try_from(index * 4).expect("small system-register program"),
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::Known("SMovB32"),
                words: vec![0],
                sources: vec![Operand::scalar(source)],
                destinations: vec![Operand::scalar(20 + source)],
                control: None,
            })
            .chain([
                Instruction {
                    pc: 16,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![Operand::vector(0)],
                    destinations: vec![Operand::vector(20)],
                    control: None,
                },
                Instruction {
                    pc: 20,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ])
            .collect(),
    );
    let mut evaluation = empty_evaluation();
    evaluation.compute_system_registers = Some(ComputeSystemRegisters {
        work_group_x: Some(4),
        work_group_y: Some(5),
        work_group_z: Some(6),
        thread_group_size: Some(7),
    });

    let shader = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
        .expect("compute system register lowering");

    assert!(shader.source.contains("remu_v_0 = remu_local_id.x"));
    assert!(shader.source.contains("remu_s_4 = remu_group_id.x;"));
    assert!(shader.source.contains("remu_s_5 = remu_group_id.y;"));
    assert!(shader.source.contains("remu_s_6 = remu_group_id.z;"));
    assert!(
        shader
            .source
            .contains("uint remu_thread_index [[thread_index_in_threadgroup]]")
    );
    assert!(
        shader
            .source
            .contains("uint3 remu_group_size [[threads_per_threadgroup]]")
    );
    assert!(
        shader
            .source
            .contains("(remu_wave_id == 0u ? 0x80000000u : 0u)")
    );
    assert!(shader.source.contains("((remu_wave_id & 31u) << 20u)"));
    assert!(shader.source.contains("(remu_threadgroup_waves & 63u)"));

    let wave64 = compile_msl(
        &program,
        &evaluation,
        &MetalCompileOptions::compute_wave(0, 64),
    )
    .expect("wave64 compute system register lowering");
    assert!(wave64.source.contains("remu_thread_index >> 6u"));
    assert!(
        wave64
            .source
            .contains("(remu_threadgroup_threads + 63u) >> 6u")
    );
    assert!(
        !wave64
            .source
            .contains("uint remu_wave_id = remu_thread_index >> 5u;"),
        "TG_SIZE must count guest wave64s rather than host SIMD32 halves"
    );
}

#[test]
fn rejects_compute_system_registers_outside_compute_and_sgpr_bounds() {
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0,
            encoding: ShaderEncoding::Sopp,
            opcode: Opcode::END_PROGRAM,
            words: vec![0],
            sources: Vec::new(),
            destinations: Vec::new(),
            control: None,
        }],
    );
    let mut evaluation = empty_evaluation();
    evaluation.compute_system_registers = Some(ComputeSystemRegisters {
        work_group_x: Some(128),
        ..ComputeSystemRegisters::default()
    });
    let error = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
        .expect_err("out-of-range system SGPR must be rejected");
    assert!(
        error
            .to_string()
            .contains("exceeds the scalar register file")
    );

    evaluation.compute_system_registers = Some(ComputeSystemRegisters {
        work_group_x: Some(4),
        ..ComputeSystemRegisters::default()
    });
    let error = compile_msl(&program, &evaluation, &MetalCompileOptions::vertex(0))
        .expect_err("graphics stage must reject compute system SGPRs");
    assert!(error.to_string().contains("require a compute shader stage"));
}

#[cfg(target_os = "macos")]
#[test]
fn generated_memory_msl_compiles_with_apple_metal() {
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

    fn find_metal_compiler() -> Option<PathBuf> {
        if let Some(path) = std::env::var_os("REMU_METAL_COMPILER").map(PathBuf::from)
            && path.is_file()
        {
            return Some(path);
        }
        let bundled = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../.remu/toolchains/Metal.xctoolchain/usr/bin/metal");
        if bundled.is_file() {
            return Some(bundled);
        }
        let output = Command::new("xcrun")
            .args(["--find", "metal"])
            .output()
            .ok()?;
        output
            .status
            .success()
            .then(|| PathBuf::from(String::from_utf8_lossy(&output.stdout).trim().to_owned()))
    }

    let Some(compiler) = find_metal_compiler() else {
        eprintln!("skipping Apple Metal memory compile test: no metal compiler found");
        return;
    };
    let (program, mut evaluation) = complete_memory_program();
    evaluation.compute_system_registers = Some(ComputeSystemRegisters {
        work_group_x: Some(116),
        work_group_y: Some(117),
        work_group_z: Some(118),
        thread_group_size: Some(119),
    });
    let shader = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
        .expect("complete decoded VMEM surface should lower");
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "remu-memory-metal-{}-{sequence}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).expect("create Metal memory test directory");
    let source = directory.join("memory.metal");
    let air = directory.join("memory.air");
    std::fs::write(&source, shader.source).expect("write generated memory MSL");
    let output = Command::new(&compiler)
        .args(["-c", "-target", "air64-apple-macosx15.0"])
        .arg("-fmetal-math-mode=safe")
        .arg("-o")
        .arg(&air)
        .arg(&source)
        .output()
        .expect("run Apple Metal compiler");
    assert!(
        output.status.success(),
        "memory MSL failed to compile:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::remove_dir_all(directory).expect("remove Metal memory test directory");
}

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

#[test]
fn lowers_every_decoded_ds_operation_with_architectural_offsets_and_atomics() {
    let program = decoded_ds_program(0x1234, false);
    for (instruction, (_, name)) in program.instructions.iter().zip(DS_OPCODES) {
        assert!(instruction.opcode.is(name));
    }
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(32 * 1024),
    )
    .expect("every decoded DS operation should lower");
    assert_eq!(shader.bindings.threadgroup_memory_bytes, 32 * 1024);
    assert!(
        shader
            .source
            .contains("threadgroup atomic_uint* remu_lds [[threadgroup(0)]]")
    );
    assert!(shader.source.contains("+ 0x1234u"));
    assert!(shader.source.contains("+ 0xD0u"));
    assert!(shader.source.contains("+ 0x48u"));
    assert!(shader.source.contains("+ 0x3400u"));
    assert!(shader.source.contains("+ 0x1200u"));
    assert!(shader.source.contains("atomic_fetch_add_explicit"));
    assert!(shader.source.contains("atomic_fetch_sub_explicit"));
    assert!(shader.source.contains("atomic_fetch_min_explicit"));
    assert!(shader.source.contains("atomic_fetch_max_explicit"));
    assert!(shader.source.contains("atomic_fetch_and_explicit"));
    assert!(shader.source.contains("atomic_fetch_or_explicit"));
    assert!(shader.source.contains("atomic_fetch_xor_explicit"));
    assert!(shader.source.contains("atomic_exchange_explicit"));
    assert!(shader.source.contains("remu_ds_inc"));
    assert!(
        shader
            .source
            .contains("observed >= ceiling ? 0u : observed + 1u")
    );
    assert!(
        shader
            .source
            .contains("(observed == 0u || observed > ceiling) ? ceiling : observed - 1u")
    );
    assert!(shader.source.contains("remu_ds_cmpst"));
    assert!(shader.source.contains("if (exec &&"));
    assert!(shader.source.contains("< 8192u"));
    assert!(
        shader
            .source
            .contains("uint remu_thread_index [[thread_index_in_threadgroup]]")
    );
    assert!(
        shader.source.contains("(remu_s_124 & 0xFFFFu)")
            && shader.source.contains("remu_thread_index * 4u"),
        "DS_WRITE_ADDTID_B32 must address M0[15:0] + concatenated offset + TID*4"
    );
    assert!(
        shader
            .source
            .contains("threadgroup_barrier(mem_flags::mem_threadgroup)")
    );
}

#[test]
fn ds_rejects_unrepresentable_resources_instead_of_wrapping_or_privatizing_memory() {
    let memory_program = Program::new(
        0x1000,
        vec![decoded_ds_program(0, false).instructions[0].clone()],
    );
    let graphics_error = compile_msl(
        &memory_program,
        &empty_evaluation(),
        &MetalCompileOptions::vertex(0),
    )
    .expect_err("graphics LDS must not become lane-private scratch");
    assert!(graphics_error.to_string().contains("workgroup-shared"));

    let missing_size = compile_msl(
        &memory_program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(0),
    )
    .expect_err("compute LDS requires its architectural allocation size");
    assert!(missing_size.to_string().contains("non-zero dword-aligned"));

    let oversized = compile_msl(
        &memory_program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(64 * 1024),
    )
    .expect_err("64 KiB guest LDS cannot silently alias 32 KiB Metal memory");
    assert!(oversized.to_string().contains("32768-byte"));

    let gds_program = decoded_ds_program(0, true);
    let gds_error = compile_msl(
        &gds_program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(32 * 1024),
    )
    .expect_err("GDS needs a system-wide resource");
    assert!(gds_error.to_string().contains("system-wide"));
}

#[test]
fn lowers_ngg_lds_and_primitive_exports_to_a_mesh_function() {
    let data_share = |pc, opcode, sources, destinations| Instruction {
        pc,
        encoding: ShaderEncoding::Ds,
        opcode: Opcode::Known(opcode),
        words: vec![0, 0],
        sources,
        destinations,
        control: Some(InstructionControl::DataShare(DataShareControl {
            offset0: 0,
            offset1: 0,
            gds: false,
        })),
    };
    let export = |pc, target, sources| Instruction {
        pc,
        encoding: ShaderEncoding::Exp,
        opcode: Opcode::Known("Exp"),
        words: vec![0, 0],
        sources,
        destinations: Vec::new(),
        control: Some(InstructionControl::Export(ExportControl {
            target,
            enable_mask: if target == 20 { 1 } else { 15 },
            compressed: false,
            done: true,
            valid_mask: false,
        })),
    };
    let program = Program::new(
        0x1000,
        vec![
            data_share(
                0,
                "DsWriteB32",
                vec![Operand::vector(0), Operand::vector(1)],
                Vec::new(),
            ),
            data_share(
                8,
                "DsReadB32",
                vec![Operand::vector(0)],
                vec![Operand::vector(2)],
            ),
            export(
                16,
                20,
                vec![
                    Operand::vector(3),
                    Operand::vector(3),
                    Operand::vector(3),
                    Operand::vector(3),
                ],
            ),
            export(
                24,
                12,
                vec![
                    Operand::vector(4),
                    Operand::vector(5),
                    Operand::vector(6),
                    Operand::vector(7),
                ],
            ),
            Instruction {
                pc: 32,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::mesh(2, 16 * 1024),
    )
    .expect("NGG mesh lowering");

    assert_eq!(shader.stage, MetalShaderStage::Mesh);
    assert!(shader.source.contains("[[mesh]] void remu_main"));
    assert!(
        shader
            .source
            .contains("mesh<RemuVsOut, void, 64, 64, topology::triangle>")
    );
    assert!(
        shader
            .source
            .contains("threadgroup atomic_uint remu_lds_storage[4096]")
    );
    assert!(
        shader
            .source
            .contains("threadgroup_barrier(mem_flags::mem_threadgroup)")
    );
    assert!(shader.source.contains("remu_mesh.set_index("));
    assert!(shader.source.contains("remu_mesh.set_vertex("));
    assert!(shader.source.contains("remu_mesh.set_primitive_count(64u)"));
    #[cfg(target_os = "macos")]
    if let Some(compiler) = std::env::var_os("REMU_METAL_COMPILER") {
        let directory =
            std::env::temp_dir().join(format!("remu-ngg-mesh-metal-{}", std::process::id()));
        std::fs::create_dir_all(&directory).expect("create mesh Metal test directory");
        let source = directory.join("mesh.metal");
        let air = directory.join("mesh.air");
        std::fs::write(&source, &shader.source).expect("write generated mesh MSL");
        let output = std::process::Command::new(compiler)
            .args(["-c", "-target", "air64-apple-macosx15.0"])
            .arg("-fmetal-math-mode=safe")
            .arg("-o")
            .arg(&air)
            .arg(&source)
            .output()
            .expect("run Apple Metal compiler");
        assert!(
            output.status.success(),
            "mesh MSL failed to compile:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        std::fs::remove_dir_all(directory).expect("remove mesh Metal test directory");
    }
}

#[test]
fn lowers_graphics_addtid_spills_to_lane_private_records() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0x100,
                encoding: ShaderEncoding::Ds,
                opcode: Opcode::Known("DsWriteAddtidB32"),
                words: vec![0, 0],
                sources: vec![Operand::vector(7)],
                destinations: Vec::new(),
                control: Some(InstructionControl::DataShare(DataShareControl {
                    offset0: 0,
                    offset1: 2,
                    gds: false,
                })),
            },
            Instruction {
                pc: 0x108,
                encoding: ShaderEncoding::Ds,
                opcode: Opcode::Known("DsReadAddtidB32"),
                words: vec![0, 0],
                sources: Vec::new(),
                destinations: vec![Operand::vector(8)],
                control: Some(InstructionControl::DataShare(DataShareControl {
                    offset0: 0,
                    offset1: 2,
                    gds: false,
                })),
            },
        ],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::pixel(Vec::new(), 0, 0),
    )
    .expect("lane-private ADDTID spill should lower in a graphics stage");

    assert!(shader.source.contains("uint remu_addtid_address[1] = {};"));
    assert!(shader.source.contains("remu_addtid_address[0] ="));
    assert!(shader.source.contains("remu_addtid_sequence[0] ="));
    assert!(shader.source.contains("remu_addtid_i < 1u"));
    assert!(shader.source.contains("(remu_s_124 & 0xFFFFu) + 0x200u"));
    assert!(!shader.source.contains("threadgroup atomic_uint* remu_lds"));
    assert_eq!(shader.bindings.threadgroup_memory_bytes, 0);
}

#[test]
fn lowers_all_65536_ds_swizzle_controls_and_all_four_isa_modes() {
    let seed = ds_swizzle_instruction(0);
    let program = Program::new(0x1000, vec![seed]);
    let evaluation = empty_evaluation();
    let options = MetalCompileOptions::vertex(0);
    let mut compiler = Compiler::new(&program, &evaluation, &options).expect("compiler");
    for offset in 0u16..=u16::MAX {
        compiler.body.clear();
        compiler.next_temp = 0;
        compiler
            .emit_data_share(&ds_swizzle_instruction(offset))
            .unwrap_or_else(|error| panic!("swizzle 0x{offset:04X} did not lower: {error}"));
        assert!(compiler.body.contains("simd_shuffle(v[2]"));
        assert!(compiler.body.contains("simd_shuffle(exec ? 1u : 0u"));
        assert!(compiler.body.contains("if (exec) v[8]"));
    }

    let basic =
        compile_vector_instruction(ds_swizzle_instruction(0x041f), MetalShaderStage::Vertex);
    assert!(basic.source.contains("remu_lane & 31u"));
    let quad = compile_vector_instruction(ds_swizzle_instruction(0x80f6), MetalShaderStage::Vertex);
    assert!(quad.source.contains("0x80F6u >> ((remu_lane & 3u) * 2u)"));
    let rotate =
        compile_vector_instruction(ds_swizzle_instruction(0xc020), MetalShaderStage::Vertex);
    assert!(rotate.source.contains("remu_lane + 1u"));
    let fft = compile_vector_instruction(ds_swizzle_instruction(0xe000), MetalShaderStage::Vertex);
    assert!(fft.source.contains("reverse_bits(remu_lane & 31u)"));
    assert!(!fft.source.contains("threadgroup atomic_uint* remu_lds"));
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
fn lowers_every_decoded_vector_compare_opcode() {
    const FLOAT_RELATIONS: [&str; 16] = [
        "F", "Lt", "Eq", "Le", "Gt", "Lg", "Ge", "O", "U", "Nge", "Nlg", "Ngt", "Nle", "Neq",
        "Nlt", "Tru",
    ];
    const INTEGER_RELATIONS: [&str; 8] = ["F", "Lt", "Eq", "Le", "Gt", "Ne", "Ge", "T"];
    let mut names = Vec::new();
    for prefix in ["VCmp", "VCmpx"] {
        for relation in FLOAT_RELATIONS {
            names.push(format!("{prefix}{relation}F32"));
        }
        for suffix in ["I32", "U32"] {
            for relation in INTEGER_RELATIONS {
                names.push(format!("{prefix}{relation}{suffix}"));
            }
        }
    }
    names.push("VCmpClassF32".to_owned());
    assert_eq!(names.len(), 65);

    for name in names {
        let name: &'static str = Box::leak(name.into_boxed_str());
        let program = Program::new(
            0x1000,
            vec![Instruction {
                pc: 0,
                encoding: ShaderEncoding::Vopc,
                opcode: Opcode::Known(name),
                words: vec![0],
                sources: vec![Operand::vector(1), Operand::vector(2)],
                destinations: Vec::new(),
                control: None,
            }],
        );
        compile_msl(
            &program,
            &empty_evaluation(),
            &MetalCompileOptions::vertex(0),
        )
        .unwrap_or_else(|error| panic!("{name} did not lower: {error}"));
    }
}

#[test]
fn vector_compares_preserve_float_integer_and_mask_semantics() {
    let compare = |pc, name, control| Instruction {
        pc,
        encoding: ShaderEncoding::Vopc,
        opcode: Opcode::Known(name),
        words: vec![0],
        sources: vec![Operand::vector(1), Operand::vector(2)],
        destinations: Vec::new(),
        control,
    };
    let program = Program::new(
        0x1000,
        vec![
            compare(0, "VCmpLtF32", None),
            compare(4, "VCmpNgeF32", None),
            compare(8, "VCmpLtI32", None),
            compare(12, "VCmpLtU32", None),
            compare(16, "VCmpxEqF32", None),
            compare(
                20,
                "VCmpEqU32",
                Some(InstructionControl::Sdwa(SdwaControl {
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
                    scalar_destination: Some(12),
                })),
            ),
        ],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::vertex(0),
    )
    .expect("compare family should lower");
    assert!(
        shader
            .source
            .contains("as_type<float>(remu_v_1) < as_type<float>(remu_v_2)")
    );
    assert!(
        shader
            .source
            .contains("!(as_type<float>(remu_v_1) >= as_type<float>(remu_v_2))")
    );
    assert!(
        shader
            .source
            .contains("as_type<int>(remu_v_1) < as_type<int>(remu_v_2)")
    );
    assert!(shader.source.contains("((remu_v_1) < (remu_v_2))"));
    assert!(shader.source.contains("exec = t"));
    assert!(
        shader
            .source
            .contains("remu_s_126 = (uint)(uint64_t)simd_ballot(t")
    );
    assert!(
        shader
            .source
            .contains("remu_s_12 = (uint)(uint64_t)simd_ballot(t")
    );
    assert!(shader.source.contains("remu_s_13 = 0u;"));
}

#[test]
fn compute_vector_compares_use_real_simd_lane_ballots() {
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vopc,
            opcode: Opcode::Known("VCmpEqU32"),
            words: vec![0],
            sources: vec![Operand::vector(1), Operand::vector(2)],
            destinations: Vec::new(),
            control: None,
        }],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions {
            stage: MetalShaderStage::Compute,
            required_vertex_output_count: 0,
            render_target_array_length: 1,
            pixel_outputs: Vec::new(),
            pixel_input_enable: 0,
            pixel_input_address: 0,
            pixel_input_controls: identity_pixel_input_controls(),
            diagnostic_force_visible_clip_depth: false,
            diagnostic_force_vertex_pattern: false,
            diagnostic_force_vertex_fetch_id: false,
            diagnostic_capture_vertex_position: false,
            diagnostic_capture_pixel_output: false,
            diagnostic_force_pixel_color: false,
            lds_size_bytes: 0,
            wave_lane_count: 32,
        },
    )
    .expect("compute compare should lower");
    assert!(
        shader
            .source
            .contains("uint remu_lane [[thread_index_in_simdgroup]]")
    );
    assert!(shader.source.contains("(uint)(uint64_t)simd_ballot("));
    assert!(!shader.source.contains("const uint remu_lane = 0u"));
}

#[test]
fn accepts_exactly_the_architectural_dpp16_control_space() {
    for control in 0..=0x1ff {
        let instruction = vector_instruction(
            "VMovB32",
            1,
            InstructionControl::Dpp(DppControl {
                control,
                fetch_inactive: false,
                bound_control: false,
                absolute_mask: 0,
                negate_mask: 0,
                bank_mask: 0xf,
                row_mask: 0xf,
            }),
        );
        let result = compile_msl(
            &Program::new(0x1000, vec![instruction]),
            &empty_evaluation(),
            &MetalCompileOptions::vertex(0),
        );
        assert_eq!(
            result.is_ok(),
            is_supported_dpp_control(control),
            "DPP control 0x{control:03X} acceptance differs from the ISA table: {result:?}"
        );
    }
}

#[test]
fn lowers_dpp_and_dpp8_for_every_otherwise_supported_vector_alu() {
    const ALU: [(&str, usize); 75] = [
        ("VCndmaskB32", 2),
        ("VMovB32", 1),
        ("VAddF32", 2),
        ("VSubF32", 2),
        ("VSubrevF32", 2),
        ("VMulF32", 2),
        ("VAddF16", 2),
        ("VSubF16", 2),
        ("VSubrevF16", 2),
        ("VMulF16", 2),
        ("VFmacF16", 2),
        ("VMaxF16", 2),
        ("VMinF16", 2),
        ("VLdexpF16", 2),
        ("VMinF32", 2),
        ("VMaxF32", 2),
        ("VMin3F32", 3),
        ("VMax3F32", 3),
        ("VMed3F32", 3),
        ("VMin3I32", 3),
        ("VMax3I32", 3),
        ("VMed3I32", 3),
        ("VMin3U32", 3),
        ("VMax3U32", 3),
        ("VMed3U32", 3),
        ("VSadU32", 3),
        ("VFmacF32", 2),
        ("VMacF32", 2),
        ("VFloorF32", 1),
        ("VCeilF32", 1),
        ("VTruncF32", 1),
        ("VRndneF32", 1),
        ("VFractF32", 1),
        ("VSqrtF32", 1),
        ("VRsqF32", 1),
        ("VRcpF32", 1),
        ("VRcpIflagF32", 1),
        ("VLogF32", 1),
        ("VExpF32", 1),
        ("VSinF32", 1),
        ("VCosF32", 1),
        ("VAndB32", 2),
        ("VAddI32", 2),
        ("VLshrrevB32", 2),
        ("VLshlrevB32", 2),
        ("VLshlAddU32", 3),
        ("VBfeU32", 3),
        ("VBfiB32", 3),
        ("VAdd3U32", 3),
        ("VOr3U32", 3),
        ("VAndOrB32", 3),
        ("VLshlOrU32", 3),
        ("VCvtF32I32", 1),
        ("VCvtF32U32", 1),
        ("VCvtF16F32", 1),
        ("VCvtF32F16", 1),
        ("VMinU32", 2),
        ("VMaxU32", 2),
        ("VMulLoU32", 2),
        ("VMulHiU32", 2),
        ("VMulHiI32", 2),
        ("VCvtU32F32", 1),
        ("VCvtI32F32", 1),
        ("VCvtOffF32I4", 1),
        ("VCubeidF32", 3),
        ("VCubescF32", 3),
        ("VCubetcF32", 3),
        ("VCubemaF32", 3),
        ("VCvtPkrtzF16F32", 2),
        ("VMadF32", 3),
        ("VFmaF32", 3),
        ("VMadAkF32", 3),
        ("VMadMkF32", 3),
        ("VFmaAkF32", 3),
        ("VFmaMkF32", 3),
    ];
    const DPP_FAMILIES: [u32; 8] = [0xe4, 0x101, 0x111, 0x121, 0x140, 0x141, 0x150, 0x160];
    for (name, source_count) in ALU {
        for control in DPP_FAMILIES {
            compile_vector_instruction(
                vector_instruction(
                    name,
                    source_count,
                    InstructionControl::Dpp(DppControl {
                        control,
                        fetch_inactive: false,
                        bound_control: false,
                        absolute_mask: 0,
                        negate_mask: 0,
                        bank_mask: 0xf,
                        row_mask: 0xf,
                    }),
                ),
                MetalShaderStage::Compute,
            );
        }
        for fetch_inactive in [false, true] {
            compile_vector_instruction(
                vector_instruction(
                    name,
                    source_count,
                    InstructionControl::Dpp8(Dpp8Control {
                        lane_selectors: 0x00fa_c688,
                        fetch_inactive,
                    }),
                ),
                MetalShaderStage::Compute,
            );
        }
    }
}

#[test]
fn lowers_dpp_and_dpp8_for_all_decoded_vector_compares() {
    const FLOAT_RELATIONS: [&str; 16] = [
        "F", "Lt", "Eq", "Le", "Gt", "Lg", "Ge", "O", "U", "Nge", "Nlg", "Ngt", "Nle", "Neq",
        "Nlt", "Tru",
    ];
    const INTEGER_RELATIONS: [&str; 8] = ["F", "Lt", "Eq", "Le", "Gt", "Ne", "Ge", "T"];
    let mut names = Vec::new();
    for prefix in ["VCmp", "VCmpx"] {
        for relation in FLOAT_RELATIONS {
            names.push(format!("{prefix}{relation}F32"));
        }
        for suffix in ["I32", "U32"] {
            for relation in INTEGER_RELATIONS {
                names.push(format!("{prefix}{relation}{suffix}"));
            }
        }
    }
    names.push("VCmpClassF32".to_owned());
    for name in names {
        let name: &'static str = Box::leak(name.into_boxed_str());
        for control in [
            InstructionControl::Dpp(DppControl {
                control: 0x111,
                fetch_inactive: false,
                bound_control: false,
                absolute_mask: 0,
                negate_mask: 0,
                bank_mask: 0xf,
                row_mask: 0xf,
            }),
            InstructionControl::Dpp8(Dpp8Control {
                lane_selectors: 0x0005_3977,
                fetch_inactive: false,
            }),
        ] {
            let mut instruction = vector_instruction(name, 2, control);
            instruction.encoding = ShaderEncoding::Vopc;
            instruction.destinations.clear();
            compile_vector_instruction(instruction, MetalShaderStage::Compute);
        }
    }
}

#[test]
fn lowers_every_vop3p_opcode_for_every_modifier_bit_pattern() {
    const OPCODES: [&str; 8] = [
        "VPkFmaF16",
        "VPkAddF16",
        "VPkMulF16",
        "VPkMinF16",
        "VPkMaxF16",
        "VFmaMixF32",
        "VFmaMixloF16",
        "VFmaMixhiF16",
    ];
    let seed = vop3p_instruction(
        OPCODES[0],
        Vop3pControl {
            operand_select_mask: 0,
            operand_select_high_mask: 0,
            negate_low_mask: 0,
            negate_high_mask: 0,
            clamp: false,
        },
    );
    let program = Program::new(0x1000, vec![seed]);
    let evaluation = empty_evaluation();
    let options = MetalCompileOptions::vertex(0);
    let mut compiler = Compiler::new(&program, &evaluation, &options).expect("compiler");

    for bits in 0u32..(1 << 13) {
        let control = Vop3pControl {
            operand_select_mask: bits & 7,
            operand_select_high_mask: (bits >> 3) & 7,
            negate_low_mask: (bits >> 6) & 7,
            negate_high_mask: (bits >> 9) & 7,
            clamp: bits >> 12 != 0,
        };
        for name in OPCODES {
            compiler.body.clear();
            compiler.next_temp = 0;
            let instruction = vop3p_instruction(name, control);
            compiler
                .emit_vector_alu(&instruction)
                .unwrap_or_else(|error| {
                    panic!("{name} modifiers 0x{bits:04X} did not lower: {error}")
                });
            assert!(compiler.body.contains("if (exec) v[8]"));
        }
    }
}

#[test]
fn vop3p_preserves_exact_packed_half_and_mix_destination_semantics() {
    let packed = compile_vector_instruction(
        vop3p_instruction(
            "VPkFmaF16",
            Vop3pControl {
                operand_select_mask: 0b001,
                operand_select_high_mask: 0b010,
                negate_low_mask: 0b100,
                negate_high_mask: 0b010,
                clamp: true,
            },
        ),
        MetalShaderStage::Vertex,
    );
    assert!(packed.source.contains("remu_f16_to_f32_bits"));
    assert!(packed.source.contains("remu_f32_to_f16_bits"));
    assert!(packed.source.contains("remu_f16_fma_rto_bits"));
    assert!(
        packed
            .source
            .contains("#pragma clang fp reassociate(off) contract(off)")
    );
    assert!(packed.source.contains("remu_clamp_unit_f32_bits"));
    assert!(packed.source.contains(">> 16) & 0xFFFFu"));
    assert!(packed.source.contains("^ 0x8000u"));
    assert!(packed.source.contains("| (t"));
    assert!(packed.source.contains("<< 16)"));

    let mut inline = vop3p_instruction(
        "VPkAddF16",
        Vop3pControl {
            operand_select_mask: 0,
            operand_select_high_mask: 0,
            negate_low_mask: 0,
            negate_high_mask: 0,
            clamp: false,
        },
    );
    inline.sources[0] = Operand {
        kind: OperandKind::EncodedConstant,
        value: 242,
    };
    let inline = compile_vector_instruction(inline, MetalShaderStage::Vertex);
    assert!(inline.source.contains("uint t0 = 0x3C00u"));
    assert!(!inline.source.contains("uint t0 = 0x3F800000u"));

    let mix_control = Vop3pControl {
        operand_select_mask: 0b001,
        operand_select_high_mask: 0b101,
        negate_low_mask: 0b010,
        negate_high_mask: 0b001,
        clamp: true,
    };
    let mut mix = vop3p_instruction("VFmaMixF32", mix_control);
    mix.sources[2] = Operand {
        kind: OperandKind::EncodedConstant,
        value: 242,
    };
    let mix = compile_vector_instruction(mix, MetalShaderStage::Vertex);
    assert!(
        mix.source
            .contains("remu_f16_to_f32_bits(((remu_v_1 >> 16) & 0xFFFFu))")
    );
    assert!(
        mix.source
            .contains("remu_f16_to_f32_bits(((0x3C00u >> 0) & 0xFFFFu))")
    );
    assert!(mix.source.contains("& 0x7FFFFFFFu"));
    assert!(mix.source.contains("^ 0x80000000u"));
    assert!(mix.source.contains("metal::precise::fma"));

    let mut literal_half = vop3p_instruction(
        "VFmaMixF32",
        Vop3pControl {
            operand_select_mask: 0,
            operand_select_high_mask: 0b001,
            negate_low_mask: 0,
            negate_high_mask: 0,
            clamp: false,
        },
    );
    literal_half.sources[0] = Operand {
        kind: OperandKind::LiteralConstant,
        value: 0xAC00,
    };
    let literal_half = compile_vector_instruction(literal_half, MetalShaderStage::Vertex);
    assert!(
        literal_half
            .source
            .contains("remu_f16_to_f32_bits(((0xAC00u >> 0) & 0xFFFFu))")
    );
    assert!(!literal_half.source.contains("as_type<float>(0xAC00u)"));

    let low = compile_vector_instruction(
        vop3p_instruction("VFmaMixloF16", mix_control),
        MetalShaderStage::Vertex,
    );
    assert!(low.source.contains("(remu_v_8 & 0xFFFF0000u) |"));
    let high = compile_vector_instruction(
        vop3p_instruction("VFmaMixhiF16", mix_control),
        MetalShaderStage::Vertex,
    );
    assert!(high.source.contains("(remu_v_8 & 0x0000FFFFu) |"));
    assert!(high.source.contains("<< 16)"));
}

#[test]
fn packed_f16_fma_round_to_odd_avoids_the_pinned_double_rounding_case() {
    fn f16_to_f32(bits: u16) -> f32 {
        let bits = u32::from(bits);
        let sign = (bits & 0x8000) << 16;
        let exponent = (bits >> 10) & 0x1f;
        let mantissa = bits & 0x3ff;
        let normal = ((exponent + 112) << 23) | (mantissa << 13);
        let infinity_nan = 0x7f80_0000 | (mantissa << 13);
        let high_bit = 31 - (mantissa | 1).leading_zeros();
        let shift = 23 - high_bit;
        let subnormal = if mantissa == 0 {
            0
        } else {
            ((high_bit + 103) << 23) | ((mantissa << shift) & 0x7f_ffff)
        };
        f32::from_bits(
            sign | if exponent == 0 {
                subnormal
            } else if exponent == 31 {
                infinity_nan
            } else {
                normal
            },
        )
    }

    fn f32_to_f16(bits: u32) -> u16 {
        let sign = (bits >> 16) & 0x8000;
        let absolute = bits & 0x7fff_ffff;
        if absolute >= 0x7f80_0000 {
            return u16::try_from(sign | 0x7c00 | (u32::from(absolute > 0x7f80_0000) * 0x200))
                .expect("half bits");
        }
        let exponent = absolute >> 23;
        let significand = (absolute & 0x7f_ffff) | 0x80_0000;
        let round_bit = (significand >> 13) & 1;
        let rounded = (significand + 0xfff + round_bit) >> 13;
        let normal_bits = (exponent.wrapping_sub(112) << 10) + (rounded - 0x400);
        let normal = if exponent >= 113 {
            normal_bits.min(0x7c00)
        } else {
            0
        };
        let distance = 126u32.wrapping_sub(exponent);
        let shift = distance.clamp(1, 25);
        let low_bits = significand & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let quotient = significand >> shift;
        let round_up = low_bits > halfway || (low_bits == halfway && quotient & 1 != 0);
        let subnormal = quotient + u32::from(round_up);
        u16::try_from(sign | if exponent <= 112 { subnormal } else { normal }).expect("half bits")
    }

    fn fma_round_to_odd(left: f32, right: f32, addend: f32) -> u32 {
        let product = std::hint::black_box(left * right);
        let sum = std::hint::black_box(product + addend);
        let product_part = std::hint::black_box(sum - addend);
        let addend_part = std::hint::black_box(sum - product_part);
        let product_error = std::hint::black_box(product - product_part);
        let addend_error = std::hint::black_box(addend - addend_part);
        let residual = std::hint::black_box(product_error + addend_error);
        let sum_bits = sum.to_bits();
        let residual_bits = residual.to_bits();
        let residual_magnitude = residual_bits & 0x7fff_ffff;
        let adjust =
            residual_magnitude != 0 && residual_magnitude < 0x7f80_0000 && sum_bits & 1 == 0;
        if !adjust {
            return sum_bits;
        }
        if (sum_bits ^ residual_bits) & 0x8000_0000 != 0 {
            sum_bits - 1
        } else {
            sum_bits + 1
        }
    }

    let left = f16_to_f32(0x4100); // 2.5
    let right = f16_to_f32(0x7522); // 21024
    let addend = f16_to_f32(0x04ea); // approximately 7.496e-5
    let double_rounded = f32_to_f16(std::hint::black_box(left * right + addend).to_bits());
    let single_rounded = f32_to_f16(fma_round_to_odd(left, right, addend));
    assert_eq!(double_rounded, 0x7a6a);
    assert_eq!(single_rounded, 0x7a6b);
    assert_eq!(f32_to_f16(fma_round_to_odd(left, right, -addend)), 0x7a6a);
}

#[cfg(target_os = "macos")]
#[test]
#[allow(clippy::too_many_lines)]
fn generated_vop3p_conversion_and_lane_msl_compiles_with_apple_metal() {
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

    fn find_metal_compiler() -> Option<PathBuf> {
        if let Some(path) = std::env::var_os("REMU_METAL_COMPILER").map(PathBuf::from)
            && path.is_file()
        {
            return Some(path);
        }
        let bundled = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../.remu/toolchains/Metal.xctoolchain/usr/bin/metal");
        if bundled.is_file() {
            return Some(bundled);
        }
        let output = Command::new("xcrun")
            .args(["--find", "metal"])
            .output()
            .ok()?;
        output
            .status
            .success()
            .then(|| PathBuf::from(String::from_utf8_lossy(&output.stdout).trim().to_owned()))
    }

    let Some(compiler) = find_metal_compiler() else {
        eprintln!("skipping Apple Metal compile test: no metal compiler found");
        return;
    };
    let control = Vop3pControl {
        operand_select_mask: 0b101,
        operand_select_high_mask: 0b011,
        negate_low_mask: 0b110,
        negate_high_mask: 0b101,
        clamp: true,
    };
    let mut instructions = [
        "VPkFmaF16",
        "VPkAddF16",
        "VPkMulF16",
        "VPkMinF16",
        "VPkMaxF16",
        "VFmaMixF32",
        "VFmaMixloF16",
        "VFmaMixhiF16",
    ]
    .into_iter()
    .enumerate()
    .map(|(index, name)| {
        let mut instruction = vop3p_instruction(name, control);
        instruction.pc = u32::try_from(index * 8).expect("small VOP3P test program");
        instruction.destinations = vec![Operand::vector(
            8 + u32::try_from(index).expect("small destination index"),
        )];
        instruction
    })
    .collect::<Vec<_>>();
    instructions[1].sources[0] = Operand {
        kind: OperandKind::EncodedConstant,
        value: 242,
    };
    instructions.push(Instruction {
        pc: u32::try_from(instructions.len() * 8).expect("small conversion test program"),
        encoding: ShaderEncoding::Vop3,
        opcode: Opcode::Known("VCvtFlrI32F32"),
        words: vec![0, 0],
        sources: vec![
            Operand::vector(1),
            Operand::source(128, None),
            Operand::source(128, None),
        ],
        destinations: vec![Operand::vector(24)],
        control: None,
    });
    instructions.push(Instruction {
        pc: u32::try_from(instructions.len() * 8).expect("small conversion test program"),
        encoding: ShaderEncoding::Vop3,
        opcode: Opcode::Known("VCvtPknormU16F32"),
        words: vec![0, 0],
        sources: vec![
            Operand::vector(2),
            Operand::vector(3),
            Operand::source(128, None),
        ],
        destinations: vec![Operand::vector(25)],
        control: None,
    });
    instructions.push(Instruction {
        pc: u32::try_from(instructions.len() * 8).expect("small bitcount test program"),
        encoding: ShaderEncoding::Vop2,
        opcode: Opcode::Known("VBcntU32B32"),
        words: vec![0],
        sources: vec![Operand::vector(1), Operand::vector(2)],
        destinations: vec![Operand::vector(26)],
        control: None,
    });
    for exchange_rows in [false, true] {
        let mut instruction =
            permlane_instruction(if exchange_rows { 1 } else { 2 }, exchange_rows);
        instruction.pc =
            u32::try_from(instructions.len() * 8).expect("small permlane test program");
        instruction.destinations = vec![Operand::vector(if exchange_rows { 28 } else { 27 })];
        instructions.push(instruction);
    }
    instructions.push(Instruction {
        pc: u32::try_from(instructions.len() * 8).expect("small u64 MAD test program"),
        encoding: ShaderEncoding::Vop3,
        opcode: Opcode::Known("VMadU64U32"),
        words: vec![0, 0],
        sources: vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
        destinations: vec![Operand::vector(30), Operand::vector(31)],
        control: Some(InstructionControl::Vop3(remu_shader::Vop3Control {
            absolute_mask: 0,
            negate_mask: 0,
            output_modifier: 0,
            clamp: false,
            operand_select: 0,
            scalar_destination: Some(20),
        })),
    });
    let program = Program::new(0x1000, instructions);
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "remu-vop3p-metal-{}-{sequence}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).expect("create Metal test directory");

    for stage in [
        MetalShaderStage::Vertex,
        MetalShaderStage::Pixel,
        MetalShaderStage::Compute,
    ] {
        let options = match stage {
            MetalShaderStage::Vertex => MetalCompileOptions::vertex(0),
            MetalShaderStage::Pixel => MetalCompileOptions::pixel(
                vec![PixelOutputBinding {
                    guest_slot: 0,
                    host_location: 0,
                    kind: PixelOutputKind::Float,
                }],
                0,
                0,
            ),
            MetalShaderStage::Compute => MetalCompileOptions {
                stage,
                required_vertex_output_count: 0,
                render_target_array_length: 1,
                pixel_outputs: Vec::new(),
                pixel_input_enable: 0,
                pixel_input_address: 0,
                pixel_input_controls: identity_pixel_input_controls(),
                diagnostic_force_visible_clip_depth: false,
                diagnostic_force_vertex_pattern: false,
                diagnostic_force_vertex_fetch_id: false,
                diagnostic_capture_vertex_position: false,
                diagnostic_capture_pixel_output: false,
                diagnostic_force_pixel_color: false,
                lds_size_bytes: 0,
                wave_lane_count: 32,
            },
            MetalShaderStage::Mesh => MetalCompileOptions::mesh(0, 16 * 1024),
        };
        let shader = compile_msl(&program, &empty_evaluation(), &options)
            .expect("VOP3P, conversion, lane, and integer program should lower");
        let stem = match stage {
            MetalShaderStage::Vertex => "vertex",
            MetalShaderStage::Pixel => "pixel",
            MetalShaderStage::Compute => "compute",
            MetalShaderStage::Mesh => "mesh",
        };
        let source = directory.join(format!("{stem}.metal"));
        let air = directory.join(format!("{stem}.air"));
        std::fs::write(&source, shader.source).expect("write generated MSL");
        let output = Command::new(&compiler)
            .args(["-c", "-target", "air64-apple-macosx15.0"])
            .arg("-fmetal-math-mode=safe")
            .arg("-o")
            .arg(&air)
            .arg(&source)
            .output()
            .expect("run Apple Metal compiler");
        assert!(
            output.status.success(),
            "{stage:?} VOP3P/conversion/lane MSL failed to compile:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    std::fs::remove_dir_all(directory).expect("remove Metal test directory");
}

#[cfg(target_os = "macos")]
#[test]
fn generated_ds_msl_compiles_with_apple_metal_in_every_valid_stage() {
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

    fn find_metal_compiler() -> Option<PathBuf> {
        if let Some(path) = std::env::var_os("REMU_METAL_COMPILER").map(PathBuf::from)
            && path.is_file()
        {
            return Some(path);
        }
        let bundled = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../.remu/toolchains/Metal.xctoolchain/usr/bin/metal");
        if bundled.is_file() {
            return Some(bundled);
        }
        let output = Command::new("xcrun")
            .args(["--find", "metal"])
            .output()
            .ok()?;
        output
            .status
            .success()
            .then(|| PathBuf::from(String::from_utf8_lossy(&output.stdout).trim().to_owned()))
    }

    let Some(compiler) = find_metal_compiler() else {
        eprintln!("skipping Apple Metal DS compile test: no metal compiler found");
        return;
    };
    let mut swizzles = [0x041f, 0x80f6, 0xc020, 0xe000]
        .into_iter()
        .enumerate()
        .map(|(index, offset)| {
            let mut instruction = ds_swizzle_instruction(offset);
            instruction.pc = u32::try_from(index * 8).expect("small swizzle program");
            instruction
        })
        .collect::<Vec<_>>();
    swizzles.push(Instruction {
        pc: 32,
        encoding: ShaderEncoding::Sopp,
        opcode: Opcode::END_PROGRAM,
        words: vec![0xbf81_0000],
        sources: Vec::new(),
        destinations: Vec::new(),
        control: None,
    });
    let swizzle_program = Program::new(0x1000, swizzles);
    let programs = [
        (
            "compute",
            decoded_ds_program(0x1234, false),
            MetalCompileOptions::compute(32 * 1024),
        ),
        (
            "vertex",
            swizzle_program.clone(),
            MetalCompileOptions::vertex(0),
        ),
        (
            "pixel",
            swizzle_program,
            MetalCompileOptions::pixel(
                vec![PixelOutputBinding {
                    guest_slot: 0,
                    host_location: 0,
                    kind: PixelOutputKind::Float,
                }],
                0,
                0,
            ),
        ),
    ];
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory =
        std::env::temp_dir().join(format!("remu-ds-metal-{}-{sequence}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("create Metal DS test directory");

    for (stem, program, options) in programs {
        let shader = compile_msl(&program, &empty_evaluation(), &options)
            .expect("valid DS program should lower");
        let source = directory.join(format!("{stem}.metal"));
        let air = directory.join(format!("{stem}.air"));
        std::fs::write(&source, shader.source).expect("write generated DS MSL");
        let output = Command::new(&compiler)
            .args(["-c", "-target", "air64-apple-macosx15.0"])
            .arg("-fmetal-math-mode=safe")
            .arg("-o")
            .arg(&air)
            .arg(&source)
            .output()
            .expect("run Apple Metal compiler");
        assert!(
            output.status.success(),
            "{stem} DS MSL failed to compile:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    std::fs::remove_dir_all(directory).expect("remove Metal DS test directory");
}

#[test]
fn dpp_semantics_cover_lane_selection_activity_modifiers_and_write_masks() {
    let shader = compile_vector_instruction(
        vector_instruction(
            "VAddF32",
            2,
            InstructionControl::Dpp(DppControl {
                control: 0x111,
                fetch_inactive: false,
                bound_control: false,
                absolute_mask: 1,
                negate_mask: 2,
                bank_mask: 0xa,
                row_mask: 5,
            }),
        ),
        MetalShaderStage::Compute,
    );
    assert!(shader.source.contains("simd_shuffle(remu_v_1, (ushort)"));
    assert!(
        shader
            .source
            .contains("simd_shuffle(exec ? 1u : 0u, (ushort)")
    );
    assert!(shader.source.contains("(remu_lane & 15u) >= 1u"));
    assert!(shader.source.contains("fabs(as_type<float>"));
    assert!(shader.source.contains("-(as_type<float>(remu_v_2))"));
    assert!(shader.source.contains("5u >> (remu_lane >> 4)"));
    assert!(shader.source.contains("10u >> (remu_lane & 3u)"));
    assert!(shader.source.contains("? t"));
    assert!(shader.source.contains(": remu_v_8"));

    let fetch_inactive = compile_vector_instruction(
        vector_instruction(
            "VMovB32",
            1,
            InstructionControl::Dpp(DppControl {
                control: 0x101,
                fetch_inactive: true,
                bound_control: true,
                absolute_mask: 0,
                negate_mask: 0,
                bank_mask: 0xf,
                row_mask: 0xf,
            }),
        ),
        MetalShaderStage::Compute,
    );
    assert!(
        !fetch_inactive
            .source
            .contains("simd_shuffle(exec ? 1u : 0u")
    );
    assert!(fetch_inactive.source.contains("&& (true)"));
}

#[test]
fn dpp8_and_compare_masks_preserve_architectural_lane_state() {
    let dpp8 = compile_vector_instruction(
        vector_instruction(
            "VMovB32",
            1,
            InstructionControl::Dpp8(Dpp8Control {
                lane_selectors: 0x0005_3977,
                fetch_inactive: false,
            }),
        ),
        MetalShaderStage::Compute,
    );
    assert!(dpp8.source.contains("342391u >> ((remu_lane & 7u) * 3u)"));
    assert!(dpp8.source.contains("remu_lane & 0xFFFFFFF8u"));
    assert!(dpp8.source.contains("simd_shuffle(remu_v_1, (ushort)"));
    assert!(
        dpp8.source
            .contains("simd_shuffle(exec ? 1u : 0u, (ushort)")
    );

    let mut compare = vector_instruction(
        "VCmpEqF32",
        2,
        InstructionControl::Dpp(DppControl {
            control: 0xe4,
            fetch_inactive: false,
            bound_control: false,
            absolute_mask: 0,
            negate_mask: 0,
            bank_mask: 0,
            row_mask: 0,
        }),
    );
    compare.encoding = ShaderEncoding::Vopc;
    compare.destinations.clear();
    let compare = compile_vector_instruction(compare, MetalShaderStage::Vertex);
    assert!(compare.source.contains(": vcc"));
    assert!(compare.source.contains("remu_s_106 ="));
    assert!(compare.source.contains("remu_s_107 = 0u"));
}

#[test]
fn graphics_dpp_uses_converged_real_wave_operations() {
    let shader = compile_vector_instruction(
        vector_instruction(
            "VMovB32",
            1,
            InstructionControl::Dpp8(Dpp8Control {
                lane_selectors: 0x00fa_c688,
                fetch_inactive: false,
            }),
        ),
        MetalShaderStage::Vertex,
    );
    assert!(
        shader
            .source
            .contains("uint remu_lane = simd_prefix_exclusive_sum(1u)")
    );
    assert!(shader.source.contains("simd_shuffle(remu_v_1, (ushort)"));
    assert_eq!(shader.register_footprint.scalar_values, 0);
    let shuffle = shader
        .source
        .find("simd_shuffle(remu_v_1, (ushort)")
        .expect("DPP8 shuffle");
    let write = shader
        .source
        .find("if (exec) remu_v_8")
        .expect("EXEC-guarded destination write");
    assert!(
        shuffle < write,
        "wave shuffle must be converged before EXEC guard"
    );
}

// Digests of the MSL that lowering emits for two fixed programs, one per control-flow mode.
// Translated source is compiled and cached per guest shader, so any difference here changes
// every pipeline a game builds and has to be an intended backend change.
const DIRECT_SOURCE_DIGEST: u64 = 0xfdb5_3279_d8e0_bd9f;
const DISPATCHER_SOURCE_DIGEST: u64 = 0x049b_d3d1_c39b_d2da;

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

fn dispatcher_reference_program() -> Program {
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
    let dispatcher = compile_msl(
        &dispatcher_reference_program(),
        &empty_evaluation(),
        &MetalCompileOptions::compute(0),
    )
    .expect("dispatcher reference program should lower");

    assert!(direct.register_footprint.direct_control_flow);
    assert!(!dispatcher.register_footprint.direct_control_flow);
    assert_eq!(
        source_digest(&direct.source),
        DIRECT_SOURCE_DIGEST,
        "direct lowering emitted different MSL:\n{}",
        direct.source
    );
    assert_eq!(
        source_digest(&dispatcher.source),
        DISPATCHER_SOURCE_DIGEST,
        "dispatcher lowering emitted different MSL:\n{}",
        dispatcher.source
    );
}
