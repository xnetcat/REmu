// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Vector comparison lowering tests: VOPC float/integer predicates and the lane
//! mask ballots they produce.

use super::*;

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
            vertex_output_control: 0,
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
