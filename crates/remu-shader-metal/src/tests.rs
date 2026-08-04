// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Metal lowering tests, split by instruction family.
//!
//! This module holds the fixtures shared across families; each submodule
//! owns the tests for one instruction class and its family-local helpers.

use remu_shader::{
    ComputeSystemRegisters, DataShareControl, Dpp8Control, DppControl, Instruction, Program,
    RawOpcodeFamily, ShaderEvaluation, Vop3pControl, decode_program,
};

use super::*;

mod compare;
mod control_flow;
mod conversion;
mod data_share;
mod dpp;
mod golden;
mod image;
mod interface;
mod lane;
mod memory;
mod ngg_registers;
mod scalar;
mod vector_alu;
mod vop3p;

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

#[cfg(target_os = "macos")]
fn compile_generated_msl_with_safe_math(
    compiler: &std::path::Path,
    source: &std::path::Path,
    air: &std::path::Path,
) -> std::process::Output {
    fn run(
        compiler: &std::path::Path,
        math_option: &str,
        source: &std::path::Path,
        air: &std::path::Path,
    ) -> std::process::Output {
        std::process::Command::new(compiler)
            .args(["-c", "-target", "air64-apple-macosx15.0"])
            .arg(math_option)
            .arg("-o")
            .arg(air)
            .arg(source)
            .output()
            .expect("run Apple Metal compiler")
    }

    let output = run(compiler, "-fmetal-math-mode=safe", source, air);
    if output.status.success() {
        return output;
    }
    let diagnostics = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !diagnostics.contains("unknown argument: '-fmetal-math-mode=safe'")
        && !diagnostics.contains("unknown argument: \"-fmetal-math-mode=safe\"")
        && !diagnostics.contains("unsupported option '-fmetal-math-mode=safe'")
        && !diagnostics.contains("unsupported option \"-fmetal-math-mode=safe\"")
    {
        return output;
    }

    let _ = std::fs::remove_file(air);
    run(compiler, "-fno-fast-math", source, air)
}

#[test]
fn hlsl_evaluation_key_excludes_live_resource_addresses_and_captured_data() {
    let uint_format = (0..0x200)
        .find(|format| remu_shader::decode_unified_format(*format) == Some((10, 4)))
        .expect("RGBA8 uint unified format exists");
    let mut first = empty_evaluation();
    first.initial_scalar_registers[0] = 0x1111_0000;
    first.global_memory_bindings = vec![remu_shader::GlobalMemoryBinding {
        scalar_address: 8,
        base_address: 0x1_0000,
        instruction_pcs: [0x20].into_iter().collect(),
        byte_count: 256,
        data: vec![1; 16],
        writable: false,
        write_back_to_guest: false,
    }];
    first.image_bindings = vec![remu_shader::ImageBinding {
        pc: 0x40,
        opcode: Opcode::Known("ImageSample"),
        control: remu_shader::ImageControl {
            dmask: 0xf,
            vector_address: 0,
            address_registers: vec![0, 1],
            vector_data: 4,
            scalar_resource: 16,
            scalar_sampler: 24,
            dimension: 1,
            is_array: false,
            glc: false,
            slc: false,
            a16: false,
            d16: false,
            unorm: false,
        },
        // Word zero and the low byte of word one carry the live texture address. The unified
        // format in word one and TYPE in word three are code-facing resource shape.
        resource_descriptor: vec![0x1234_5678, uint_format << 20, 63, 9 << 28].into(),
        sampler_descriptor: vec![1, 2, 3, 4].into(),
        mip_level: Some(0),
    }];
    first.vertex_inputs = vec![remu_shader::VertexInputBinding {
        pc: 0x20,
        location: 0,
        component_count: 4,
        data_format: 10,
        number_format: 0,
        base_address: 0x2_0000,
        stride: 16,
        offset_bytes: 0,
        data: vec![2; 64],
    }];

    let mut second = first.clone();
    second.initial_scalar_registers[0] = 0x9999_0000;
    second.global_memory_bindings[0].base_address = 0x8_0000;
    second.global_memory_bindings[0].byte_count = 4_096;
    second.global_memory_bindings[0].data = vec![9; 32];
    second.global_memory_bindings[0].write_back_to_guest = true;
    second.image_bindings[0].resource_descriptor[0] = 0xabcd_1234;
    second.image_bindings[0].sampler_descriptor = vec![8, 7, 6, 5].into();
    second.image_bindings[0].mip_level = Some(4);
    second.vertex_inputs[0].base_address = 0x9_0000;
    second.vertex_inputs[0].stride = 32;
    second.vertex_inputs[0].data = vec![3; 128];

    assert_eq!(
        HlslEvaluationKey::new(&first),
        HlslEvaluationKey::new(&second)
    );

    second.image_bindings[0].resource_descriptor[1] = 0;
    assert_ne!(
        HlslEvaluationKey::new(&first),
        HlslEvaluationKey::new(&second),
        "changing the component type must select different HLSL"
    );
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

const DS_OPCODES: [(u32, &str); 44] = [
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
    (0x4e, "DsWrite2B64"),
    (0x4f, "DsWrite2St64B64"),
    (0x76, "DsReadB64"),
    (0x77, "DsRead2B64"),
    (0x78, "DsRead2St64B64"),
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
            wave_lane_count: 32,
            hlsl_wave64_group_isolation_axis: None,
            allocated_vector_register_count: 256,
            ngg_registers_proven: false,
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

// Digests of the MSL that lowering emits for two fixed programs, one per control-flow mode.
// Translated source is compiled and cached per guest shader, so any difference here changes
// every pipeline a game builds and has to be an intended backend change.
