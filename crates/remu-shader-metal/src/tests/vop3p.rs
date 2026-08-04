// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! VOP3P packed maths lowering tests: every opcode across the modifier bit
//! space, packed half destinations and round-to-odd FMA behaviour.

use super::*;

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
    for destination in [44, 45] {
        instructions.push(Instruction {
            pc: u32::try_from(instructions.len() * 8).expect("small readfirstlane test program"),
            encoding: ShaderEncoding::Vop1,
            opcode: Opcode::Known("VReadfirstlaneB32"),
            words: vec![0],
            sources: vec![Operand::vector(4)],
            destinations: vec![Operand::scalar(destination)],
            control: None,
        });
    }
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
                    export_format: PixelExportFormat::Abgr32,
                }],
                0,
                0,
            ),
            MetalShaderStage::Compute => MetalCompileOptions {
                stage,
                required_vertex_output_count: 0,
                required_vertex_output_locations: None,
                vertex_output_control: 0,
                render_target_array_length: 1,
                pixel_outputs: Vec::new(),
                pixel_depth_export: false,
                pixel_input_enable: 0,
                pixel_input_address: 0,
                pixel_input_controls: identity_pixel_input_controls(),
                diagnostic_force_visible_clip_depth: false,
                diagnostic_force_vertex_pattern: false,
                diagnostic_force_vertex_fetch_id: false,
                diagnostic_capture_vertex_position: false,
                diagnostic_capture_pixel_output: false,
                diagnostic_force_pixel_color: false,
                mesh_max_vertex_count: 0,
                mesh_max_primitive_count: 0,
                mesh_threadgroup_size: 0,
                lds_size_bytes: 0,
                scratch_size_bytes: 0,
                wave_lane_count: 64,
                hlsl_wave64_group_isolation_axis: None,
                allocated_vector_register_count: 256,
                ngg_registers_proven: false,
            },
            MetalShaderStage::Mesh => MetalCompileOptions::mesh(0, 16 * 1024, 64, 64, 64),
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
