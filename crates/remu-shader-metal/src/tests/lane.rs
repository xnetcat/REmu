// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Lane and permute lowering tests: PERMLANE, READLANE/WRITELANE, lane-mask
//! bit counts and the wave64 SIMD32-half split.

use super::*;

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
            lds_size_bytes: 2_048,
            wave_lane_count: 64,
            allocated_vector_register_count: 256,
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
    assert_eq!(
        shader
            .source
            .matches("threadgroup_barrier(mem_flags::mem_threadgroup)")
            .count(),
        1,
        "an acyclic static site does not need a record-reuse barrier"
    );
    assert!(shader.source.contains("remu_readfirstlane_active"));
    assert!(shader.source.contains("remu_s_44 ="));
}

#[test]
fn acyclic_wave64_readfirstlane_sites_use_disjoint_exchange_records() {
    let instructions = [0_u32, 4]
        .into_iter()
        .map(|pc| Instruction {
            pc,
            encoding: ShaderEncoding::Vop1,
            opcode: Opcode::Known("VReadfirstlaneB32"),
            words: vec![0],
            sources: vec![Operand::vector(4)],
            destinations: vec![Operand::scalar(44 + pc)],
            control: None,
        })
        .collect();
    let shader = compile_msl(
        &Program::new(0x1000, instructions),
        &empty_evaluation(),
        &MetalCompileOptions::compute_wave(0, 64),
    )
    .expect("acyclic readfirstlane sites should lower with disjoint records");

    assert!(
        shader
            .source
            .contains("threadgroup uint remu_readfirstlane_values[64]")
    );
    assert!(shader.source.contains("32u + ((remu_wave_index << 1u)"));
    assert!(shader.source.contains("32u + (remu_wave_index << 1u)"));
    assert_eq!(
        shader
            .source
            .matches("threadgroup_barrier(mem_flags::mem_threadgroup)")
            .count(),
        2,
        "each static site needs only its producer-consumer barrier"
    );
}

#[test]
fn captured_acyclic_wave64_site_count_halves_whole_group_barriers() {
    // Silent Hill's largest captured acyclic compute shader contains 44 static sites. The
    // previous shared-record protocol emitted two whole-threadgroup barriers for every site.
    let instructions = (0_u32..44)
        .map(|site| Instruction {
            pc: site * 4,
            encoding: ShaderEncoding::Vop1,
            opcode: Opcode::Known("VReadfirstlaneB32"),
            words: vec![0],
            sources: vec![Operand::vector(4)],
            destinations: vec![Operand::scalar(44 + site)],
            control: None,
        })
        .collect();
    let shader = compile_msl(
        &Program::new(0x1000, instructions),
        &empty_evaluation(),
        &MetalCompileOptions::compute_wave(0, 64),
    )
    .expect("captured acyclic site count fits the disjoint exchange budget");

    assert!(
        shader
            .source
            .contains("threadgroup uint remu_readfirstlane_values[1408]")
    );
    assert_eq!(
        shader
            .source
            .matches("threadgroup_barrier(mem_flags::mem_threadgroup)")
            .count(),
        44,
        "disjoint records retain the producer-consumer barrier and remove only reuse barriers"
    );
}

#[test]
fn memory_heavy_acyclic_readfirstlane_retains_shared_records() {
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0,
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
        &MetalCompileOptions::compute_wave(32 * 1_024, 64),
    )
    .expect("memory-heavy readfirstlane should lower with shared records");

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
