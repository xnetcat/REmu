// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Runtime heap texture table lowering.
//!
//! See docs/design/2026-08-13-runtime-image-descriptor-fetch.md. The whole path is gated on
//! `MetalCompileOptions::runtime_heap_tables`, which the presenter sets from REMU_HEAP_TABLES=1
//! for the pixel stage only; these tests drive it directly.

use super::*;
use crate::compiler::image::{ImageKind, heap_texture_table_type, metal_texture_declaration};
use remu_shader::ImageViewReinterpretation;

/// The SGPR holding the flagged binding's T#.
const HEAP_BASE: u32 = 12;

fn image_control(scalar_resource: u32) -> ImageControl {
    ImageControl {
        dmask: 0xf,
        vector_address: 0,
        address_registers: vec![0, 1],
        vector_data: 4,
        scalar_resource,
        scalar_sampler: 100,
        dimension: 1,
        is_array: false,
        glc: false,
        slc: false,
        a16: false,
        d16: false,
        unorm: false,
    }
}

fn image_sample(pc: u32, scalar_resource: u32) -> Instruction {
    Instruction {
        pc,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageSample"),
        words: vec![0, 0],
        sources: vec![
            Operand::vector(0),
            Operand::vector(1),
            Operand::scalar(scalar_resource),
        ],
        destinations: (4..8).map(Operand::vector).collect(),
        control: Some(InstructionControl::Image(image_control(scalar_resource))),
    }
}

fn scalar_load(pc: u32, first_destination: u32, destination_count: u32) -> Instruction {
    Instruction {
        pc,
        encoding: ShaderEncoding::Smem,
        opcode: if destination_count == 16 {
            Opcode::Known("SLoadDwordx16")
        } else {
            Opcode::Known("SLoadDwordx8")
        },
        words: vec![0, 0],
        sources: vec![Operand::scalar(0)],
        destinations: (first_destination..first_destination + destination_count)
            .map(Operand::scalar)
            .collect(),
        control: Some(InstructionControl::ScalarMemory(
            remu_shader::ScalarMemoryControl {
                destination_count,
                immediate_offset_bytes: 0,
                dynamic_offset_register: Some(20),
            },
        )),
    }
}

fn end_program(pc: u32) -> Instruction {
    Instruction {
        pc,
        encoding: ShaderEncoding::Sopp,
        opcode: Opcode::END_PROGRAM,
        words: vec![0xbf81_0000],
        sources: Vec::new(),
        destinations: Vec::new(),
        control: None,
    }
}

fn scalar_move(pc: u32, destination: u32) -> Instruction {
    Instruction {
        pc,
        encoding: ShaderEncoding::Sop1,
        opcode: Opcode::Known("SMovB32"),
        words: vec![0],
        sources: vec![Operand {
            kind: OperandKind::EncodedConstant,
            value: 128,
        }],
        destinations: vec![Operand::scalar(destination)],
        control: None,
    }
}

fn sampled_binding(
    pc: u32,
    scalar_resource: u32,
    descriptor_word3: u32,
    descriptor_index_is_runtime: bool,
) -> ImageBinding {
    let mut resource_descriptor = vec![0_u32; 8];
    resource_descriptor[3] = descriptor_word3;
    ImageBinding {
        descriptor_index_is_runtime,
        pc,
        opcode: Opcode::Known("ImageSample"),
        control: image_control(scalar_resource),
        resource_descriptor: resource_descriptor.into(),
        sampler_descriptor: vec![0; 4].into(),
        mip_level: None,
    }
}

fn heap_binding(instruction_pcs: Vec<u32>) -> remu_shader::GlobalMemoryBinding {
    remu_shader::GlobalMemoryBinding {
        scalar_address: 0,
        base_address: 0x1_0000,
        instruction_pcs: instruction_pcs.into(),
        byte_count: 4096,
        data: vec![0; 4096],
        writable: false,
        write_back_to_guest: false,
    }
}

fn heap_table_options() -> MetalCompileOptions {
    let mut options = MetalCompileOptions::compute(0);
    options.runtime_heap_tables = true;
    options
}

#[test]
fn heap_tables_are_keyed_by_canonical_texture_slot_not_evaluation_index() {
    // Two of the three sampled bindings share a descriptor and canonicalize onto texture slot 0,
    // so the flagged binding's evaluation index (2) and its canonical texture slot (1) differ.
    // Keying the map by the evaluation index still registers the route, still emits the argument
    // buffer and the slot shadow — and then never indexes it, because the sample site resolves
    // the route by canonical slot. The binding keeps sampling its stale snapshot texture, which
    // is the failure the whole design exists to remove.
    let instructions = vec![
        // The descriptor starts four dwords into a wider SMEM tuple. Discovery and shadow
        // emission must agree on that exact +16-byte offset.
        scalar_load(0x00, 8, 16),
        image_sample(0x08, 4),
        image_sample(0x10, 4),
        image_sample(0x18, HEAP_BASE),
        end_program(0x20),
    ];
    let mut evaluation = empty_evaluation();
    evaluation
        .global_memory_bindings
        .push(heap_binding(vec![0x00]));
    evaluation
        .image_bindings
        .push(sampled_binding(0x08, 4, 0xfac, false));
    evaluation
        .image_bindings
        .push(sampled_binding(0x10, 4, 0xfac, false));
    evaluation
        .image_bindings
        .push(sampled_binding(0x18, HEAP_BASE, 0x924, true));
    // The descriptor starts 16 bytes into the wide tuple, while the dynamic array offset is
    // another 16 bytes. Their sum is exactly one published 32-byte table record.
    evaluation.initial_scalar_registers[20] = 16;
    evaluation.scalar_registers[20] = 16;
    assert_eq!(
        evaluation.image_binding_slots().texture_slots.as_slice(),
        [0, 0, 1],
        "the fixture must exercise a slot map that is not the identity",
    );

    let shader = compile_msl(
        &Program::new(0x1000, instructions),
        &evaluation,
        &heap_table_options(),
    )
    .expect("flagged sample should lower through its heap table");

    assert!(
        shader
            .source
            .contains("remu_heap_1_0.tex[remu_srd_slot_12]"),
        "the flagged binding must sample through its heap table:\n{}",
        shader.source,
    );
    assert!(
        !shader.source.contains("tex1.sample("),
        "the flagged binding must not fall back to its stale snapshot texture:\n{}",
        shader.source,
    );
    assert!(
        shader
            .source
            .contains("uint remu_srd_slot_12 = 0xffffffffu;")
            && shader.source.contains(" + 16u")
            && shader.source.contains("% 32u")
            && !shader.source.contains("min(remu_srd_slot_12"),
        "the shadow must encode the complete destination offset/stride and never clamp invalid state:\n{}",
        shader.source,
    );
    assert_eq!(shader.bindings.heap_texture_tables.len(), 1);
    let table = shader.bindings.heap_texture_tables[0];
    assert_eq!(table.global_binding_index, 0);
    assert_eq!(table.canonical_texture_slot, 1);
    assert_eq!(
        table.descriptor_stride_bytes,
        HEAP_TEXTURE_DESCRIPTOR_STRIDE_BYTES
    );
}

#[test]
fn heap_table_refuses_a_descriptor_phase_that_disagrees_with_its_binding_base() {
    let instructions = vec![
        // The runtime descriptor begins 16 bytes into the load and the observed dynamic offset
        // is zero, so it is half a record away from the argument table's binding-base phase.
        scalar_load(0x00, 8, 16),
        image_sample(0x08, HEAP_BASE),
        end_program(0x10),
    ];
    let mut evaluation = empty_evaluation();
    evaluation
        .global_memory_bindings
        .push(heap_binding(vec![0x00]));
    evaluation
        .image_bindings
        .push(sampled_binding(0x08, HEAP_BASE, 0x924, true));

    let error = compile_msl(
        &Program::new(0x1000, instructions),
        &evaluation,
        &heap_table_options(),
    )
    .expect_err("a mixed-width descriptor phase must not index a 32-byte table");
    assert!(
        error.to_string().contains("descriptor phase 16")
            && error.to_string().contains("publishes 32-byte records"),
        "unexpected refusal: {error}",
    );
}

#[test]
fn canonical_texture_slot_refuses_mixed_snapshot_and_runtime_routes() {
    let instructions = vec![
        scalar_load(0x00, HEAP_BASE, 8),
        image_sample(0x08, 4),
        image_sample(0x10, HEAP_BASE),
        end_program(0x18),
    ];
    let mut evaluation = empty_evaluation();
    evaluation
        .global_memory_bindings
        .push(heap_binding(vec![0x00]));
    evaluation
        .image_bindings
        .push(sampled_binding(0x08, 4, 0xfac, false));
    evaluation
        .image_bindings
        .push(sampled_binding(0x10, HEAP_BASE, 0xfac, true));
    assert_eq!(
        evaluation.image_binding_slots().texture_slots.as_slice(),
        [0, 0]
    );

    let error = compile_msl(
        &Program::new(0x1000, instructions),
        &evaluation,
        &heap_table_options(),
    )
    .expect_err("one canonical slot cannot mix snapshot and heap lowerings");
    assert!(error.to_string().contains("mixes snapshot/runtime"));
}

#[test]
fn runtime_route_refuses_an_incomplete_descriptor_suffix() {
    let instructions = vec![
        scalar_load(0x00, 8, 8),
        image_sample(0x08, HEAP_BASE),
        end_program(0x10),
    ];
    let mut evaluation = empty_evaluation();
    evaluation
        .global_memory_bindings
        .push(heap_binding(vec![0x00]));
    evaluation
        .image_bindings
        .push(sampled_binding(0x08, HEAP_BASE, 0x924, true));

    let error = compile_msl(
        &Program::new(0x1000, instructions),
        &evaluation,
        &heap_table_options(),
    )
    .expect_err("four trailing dwords are not a complete T#");
    assert!(
        error
            .to_string()
            .contains("without all eight descriptor dwords")
    );
}

#[test]
fn scalar_overwrite_invalidates_a_runtime_descriptor_shadow() {
    let instructions = vec![
        scalar_load(0x00, HEAP_BASE, 8),
        scalar_move(0x08, HEAP_BASE + 1),
        image_sample(0x10, HEAP_BASE),
        end_program(0x18),
    ];
    let mut evaluation = empty_evaluation();
    evaluation
        .global_memory_bindings
        .push(heap_binding(vec![0x00]));
    evaluation
        .image_bindings
        .push(sampled_binding(0x10, HEAP_BASE, 0x924, true));
    let shader = compile_msl(
        &Program::new(0x1000, instructions),
        &evaluation,
        &heap_table_options(),
    )
    .expect("complete load followed by an overwrite still lowers with invalid shadow state");
    assert!(
        shader
            .source
            .matches("remu_srd_slot_12 = 0xffffffffu;")
            .count()
            >= 2,
        "declaration and overlapping scalar overwrite must both publish invalid: {}",
        shader.source
    );
}

#[test]
fn heap_table_msl_type_is_derived_from_the_canonical_image_kind() {
    let uint_3d_read = ImageKind {
        component: "uint",
        storage: true,
        writable: false,
        comparison: false,
        dimension: 2,
        texture_type: 10,
        view_reinterpretation: ImageViewReinterpretation::Direct,
    };
    assert_eq!(
        metal_texture_declaration(uint_3d_read),
        "texture3d<uint, access::read>"
    );
    assert_eq!(
        heap_texture_table_type(uint_3d_read),
        HeapTextureTableType {
            component: HeapTextureComponent::Uint,
            dimension: HeapTextureDimension::D3,
            comparison: false,
            writable: false,
        }
    );

    let sint_array_write = ImageKind {
        component: "int",
        storage: true,
        writable: true,
        comparison: false,
        dimension: 3,
        texture_type: 11,
        view_reinterpretation: ImageViewReinterpretation::Direct,
    };
    assert_eq!(
        metal_texture_declaration(sint_array_write),
        "texture2d_array<int, access::read_write>"
    );

    let comparison_array = ImageKind {
        component: "float",
        storage: false,
        writable: false,
        comparison: true,
        dimension: 3,
        texture_type: 11,
        view_reinterpretation: ImageViewReinterpretation::Direct,
    };
    assert_eq!(
        metal_texture_declaration(comparison_array),
        "depth2d_array<float>"
    );
}
