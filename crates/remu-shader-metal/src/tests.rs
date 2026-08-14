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
mod heap_tables;
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

const DS_OPCODES: [(u32, &str); 50] = [
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
    (0x1e, "DsWriteB8"),
    (0x1f, "DsWriteB16"),
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
    (0x39, "DsReadI8"),
    (0x3a, "DsReadU8"),
    (0x3b, "DsReadI16"),
    (0x3c, "DsReadU16"),
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
            vertex_output_control: 0,
            viewport_transform_control: 1 << 10,
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
            ngg_primgen_passthrough: false,
            ngg_esgs_ring_itemsize: 0,
            lds_size_bytes: 0,
            compute_threadgroup_size: 0,
            scratch_size_bytes: 0,
            wave_lane_count: 32,
            allocated_vector_register_count: 256,
            ngg_registers_proven: false,
            wave64_lane_doubled: false,
        runtime_heap_tables: false,
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
