// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Scalar memory and image resource tests: SMEM load execution, the bindings
//! discovery records for them, and image descriptor canonicalization.

use super::*;

#[test]
fn gives_unreachable_scalar_memory_a_zero_binding_without_reading_it() {
    let mut memory = TestMemory::default();
    memory.write(0x2000, &0x1122_3344_u32.to_le_bytes());
    memory.write(0x3000, &0xaabb_ccdd_u32.to_le_bytes());
    let scalar_load = |pc, scalar_base, destination| {
        instruction(
            pc,
            ShaderEncoding::Smem,
            "SLoadDword",
            vec![Operand::scalar(scalar_base)],
            vec![Operand::scalar(destination)],
            Some(InstructionControl::ScalarMemory(ScalarMemoryControl {
                destination_count: 1,
                immediate_offset_bytes: 0,
                dynamic_offset_register: None,
            })),
        )
    };
    let mut branch = instruction(
        4,
        ShaderEncoding::Sopp,
        "SCbranchScc1",
        Vec::new(),
        Vec::new(),
        None,
    );
    branch.words[0] = 2;
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sopk,
                "SCmpkEqI32",
                vec![Operand {
                    kind: OperandKind::EncodedConstant,
                    value: 1,
                }],
                vec![Operand::scalar(4)],
                None,
            ),
            branch,
            scalar_load(8, 0, 6),
            scalar_load(16, 2, 7),
            end(24),
        ],
        vec![0x2000, 0, 0x3000, 0, 1],
    );

    let evaluation = evaluate_shader(&memory, &shader, EvaluationOptions::default())
        .expect("the exact scalar branch should skip its unreachable load");

    assert_eq!(evaluation.scalar_registers[7], 0xaabb_ccdd);
    assert_eq!(evaluation.scalar_memory_reads, vec![(0x3000, 0xaabb_ccdd)]);
    assert_eq!(evaluation.global_memory_bindings.len(), 2);
    assert_eq!(
        evaluation.global_memory_bindings[0]
            .instruction_pcs
            .as_slice(),
        [16]
    );
    assert_eq!(evaluation.global_memory_bindings[1].base_address, 0);
    assert_eq!(
        evaluation.global_memory_bindings[1]
            .instruction_pcs
            .as_slice(),
        [8]
    );
}

#[test]
fn executes_scalar_loads_and_records_the_backing_binding() {
    let mut memory = TestMemory::default();
    memory.write(0x2000, &0x1122_3344_u32.to_le_bytes());
    memory.write(0x2004, &0xaabb_ccdd_u32.to_le_bytes());
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Smem,
                "SLoadDwordx2",
                vec![Operand::scalar(0)],
                vec![Operand::scalar(4), Operand::scalar(5)],
                Some(InstructionControl::ScalarMemory(ScalarMemoryControl {
                    destination_count: 2,
                    immediate_offset_bytes: 0,
                    dynamic_offset_register: None,
                })),
            ),
            end(8),
        ],
        vec![0x2000, 0],
    );
    let evaluation = evaluate_shader(&memory, &shader, EvaluationOptions::default())
        .expect("scalar load evaluation");
    assert_eq!(evaluation.scalar_registers[4], 0x1122_3344);
    assert_eq!(evaluation.scalar_registers[5], 0xaabb_ccdd);
    assert_eq!(
        evaluation.scalar_memory_reads,
        vec![(0x2000, 0x1122_3344), (0x2004, 0xaabb_ccdd)]
    );
    assert_eq!(evaluation.global_memory_bindings.len(), 1);
    assert_eq!(evaluation.global_memory_bindings[0].data.len(), 8);
}

#[test]
fn null_scalar_loads_return_zero_and_keep_a_zero_binding() {
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Smem,
                "SLoadDwordx2",
                vec![Operand::scalar(0)],
                vec![Operand::scalar(4), Operand::scalar(5)],
                Some(InstructionControl::ScalarMemory(ScalarMemoryControl {
                    destination_count: 2,
                    immediate_offset_bytes: 0,
                    dynamic_offset_register: None,
                })),
            ),
            end(8),
        ],
        vec![0, 0],
    );

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("null scalar resource must not discard the shader");

    assert_eq!(evaluation.scalar_registers[4], 0);
    assert_eq!(evaluation.scalar_registers[5], 0);
    assert!(evaluation.scalar_memory_reads.is_empty());
    assert_eq!(evaluation.global_memory_bindings.len(), 1);
    assert_eq!(evaluation.global_memory_bindings[0].base_address, 0);
    assert_eq!(evaluation.global_memory_bindings[0].data, vec![0; 4]);
}

#[test]
fn unreadable_scalar_resource_sentinels_fall_back_to_zero() {
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Smem,
                "SLoadDword",
                vec![Operand::scalar(0)],
                vec![Operand::scalar(4)],
                Some(InstructionControl::ScalarMemory(ScalarMemoryControl {
                    destination_count: 1,
                    immediate_offset_bytes: 0,
                    dynamic_offset_register: None,
                })),
            ),
            end(8),
        ],
        vec![0x3f, 0],
    );

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("unreadable resource sentinel must not discard the shader");

    assert_eq!(evaluation.scalar_registers[4], 0);
    assert!(evaluation.scalar_memory_reads.is_empty());
    assert_eq!(evaluation.global_memory_bindings.len(), 1);
    assert_eq!(evaluation.global_memory_bindings[0].base_address, 0);
    assert_eq!(evaluation.global_memory_bindings[0].data, vec![0; 4]);
}

#[test]
fn unreadable_scalar_components_after_binding_discovery_return_zero() {
    struct ResidentRangeWithUnavailableContents;

    impl GuestMemoryReader for ResidentRangeWithUnavailableContents {
        fn read(&self, _address: u64, _output: &mut [u8]) -> bool {
            false
        }

        fn readable_prefix_length(&self, _address: u64, maximum_length: usize) -> Option<usize> {
            Some(maximum_length)
        }
    }

    let shader = state(
        vec![
            instruction(
                0x0c,
                ShaderEncoding::Smem,
                "SLoadDword",
                vec![Operand::scalar(0)],
                vec![Operand::scalar(4)],
                Some(InstructionControl::ScalarMemory(ScalarMemoryControl {
                    destination_count: 1,
                    immediate_offset_bytes: 0,
                    dynamic_offset_register: None,
                })),
            ),
            end(0x14),
        ],
        vec![0x8037_3300, 0],
    );
    let evaluation = evaluate_shader(
        &ResidentRangeWithUnavailableContents,
        &shader,
        EvaluationOptions {
            capture_global_binding_data: false,
            ..EvaluationOptions::default()
        },
    )
    .expect("an unavailable scalar component must behave like a zero-filled resource");

    assert_eq!(evaluation.scalar_registers[4], 0);
    assert!(evaluation.scalar_memory_reads.is_empty());
    assert_eq!(evaluation.global_memory_bindings.len(), 1);
    assert_eq!(
        evaluation.global_memory_bindings[0].base_address,
        0x8037_3300
    );
}

#[test]
fn scalar_buffer_loads_do_not_require_the_complete_descriptor_range_to_be_mapped() {
    let mut memory = TestMemory::default();
    memory.write(0x2000, &0x1122_3344_u32.to_le_bytes());
    memory.write(0x2004, &0xaabb_ccdd_u32.to_le_bytes());
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Smem,
                "SBufferLoadDwordx2",
                vec![Operand::scalar(0)],
                vec![Operand::scalar(4), Operand::scalar(5)],
                Some(InstructionControl::ScalarMemory(ScalarMemoryControl {
                    destination_count: 2,
                    immediate_offset_bytes: 0,
                    dynamic_offset_register: None,
                })),
            ),
            end(8),
        ],
        vec![0x2000, 16 << 16, 4096, (1 << 31) | (1 << 28)],
    );

    let evaluation = evaluate_shader(&memory, &shader, EvaluationOptions::default())
        .expect("only the scalar components that execute need resident backing");

    assert_eq!(evaluation.scalar_registers[4], 0x1122_3344);
    assert_eq!(evaluation.scalar_registers[5], 0xaabb_ccdd);
    assert_eq!(evaluation.global_memory_bindings.len(), 1);
    assert_eq!(evaluation.global_memory_bindings[0].byte_count, 8);
}

#[test]
fn noncanonical_scalar_buffer_sentinels_read_as_zero() {
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Smem,
                "SBufferLoadDwordx2",
                vec![Operand::scalar(0)],
                vec![Operand::scalar(4), Operand::scalar(5)],
                Some(InstructionControl::ScalarMemory(ScalarMemoryControl {
                    destination_count: 2,
                    immediate_offset_bytes: 2_064,
                    dynamic_offset_register: None,
                })),
            ),
            end(8),
        ],
        vec![0xda1e_d51e, 0xdc1e_db1e, 0xdc1e_dd1e, 0xde1e_db1e],
    );

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("a noncanonical read-only scalar resource must not discard the shader");

    assert_eq!(evaluation.scalar_registers[4], 0);
    assert_eq!(evaluation.scalar_registers[5], 0);
    assert_eq!(evaluation.global_memory_bindings.len(), 1);
    assert_eq!(evaluation.global_memory_bindings[0].scalar_address, 0);
    assert_eq!(evaluation.global_memory_bindings[0].base_address, 0);
    assert_eq!(evaluation.global_memory_bindings[0].byte_count, 4);
    assert_eq!(evaluation.global_memory_bindings[0].data, vec![0; 4]);
    assert!(!evaluation.global_memory_bindings[0].writable);
}

#[test]
fn grows_shared_scalar_binding_to_the_largest_observed_load() {
    let mut memory = TestMemory::default();
    memory.write(0x2000, &[0; 68]);
    let scalar_load = |pc, immediate_offset_bytes, destination| {
        instruction(
            pc,
            ShaderEncoding::Smem,
            "SLoadDword",
            vec![Operand::scalar(0)],
            vec![Operand::scalar(destination)],
            Some(InstructionControl::ScalarMemory(ScalarMemoryControl {
                destination_count: 1,
                immediate_offset_bytes,
                dynamic_offset_register: None,
            })),
        )
    };
    let shader = state(
        vec![scalar_load(0, 0, 4), scalar_load(8, 64, 5), end(16)],
        vec![0x2000, 0],
    );

    let evaluation = evaluate_shader(&memory, &shader, EvaluationOptions::default())
        .expect("scalar binding growth evaluation");

    assert_eq!(evaluation.global_memory_bindings.len(), 1);
    assert_eq!(evaluation.global_memory_bindings[0].data.len(), 68);
    assert_eq!(
        evaluation.global_memory_bindings[0]
            .instruction_pcs
            .as_slice(),
        [0, 8]
    );
}

#[test]
fn resolves_image_descriptors_without_executing_vector_math() {
    let image = ImageControl {
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
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Mimg,
                "ImageSample",
                Vec::new(),
                Vec::new(),
                Some(InstructionControl::Image(image.clone())),
            ),
            instruction(
                8,
                ShaderEncoding::Exp,
                "Exp",
                Vec::new(),
                Vec::new(),
                Some(InstructionControl::Export(ExportControl {
                    target: 0,
                    enable_mask: 0xf,
                    compressed: false,
                    done: true,
                    valid_mask: false,
                })),
            ),
            end(16),
        ],
        (1..=12).collect(),
    );
    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("image evaluation");
    assert_eq!(evaluation.image_bindings.len(), 1);
    assert_eq!(
        evaluation.image_bindings[0].resource_descriptor.as_slice(),
        (1..=8).collect::<Vec<_>>().as_slice()
    );
    assert_eq!(
        evaluation.image_bindings[0].sampler_descriptor.as_slice(),
        (9..=12).collect::<Vec<_>>().as_slice()
    );
}

#[test]
fn canonicalizes_repeated_image_descriptors_without_merging_type_or_access() {
    let image_instruction = |pc, opcode, scalar_resource, scalar_sampler, dimension| {
        instruction(
            pc,
            ShaderEncoding::Mimg,
            opcode,
            Vec::new(),
            Vec::new(),
            Some(InstructionControl::Image(ImageControl {
                dmask: 0xf,
                vector_address: 0,
                address_registers: vec![0, 1, 2],
                vector_data: 4,
                scalar_resource,
                scalar_sampler,
                dimension,
                is_array: false,
                glc: false,
                slc: false,
                a16: false,
                d16: false,
            })),
        )
    };
    let mut registers = (1..=12).collect::<Vec<_>>();
    registers.resize(16, 0);
    registers.extend(101..=108);
    let shader = state(
        vec![
            image_instruction(0, "ImageSample", 0, 8, 1),
            image_instruction(8, "ImageSampleLz", 0, 8, 1),
            image_instruction(16, "ImageLoad", 0, 0, 1),
            image_instruction(24, "ImageSampleCLz", 0, 8, 1),
            image_instruction(32, "ImageSample", 0, 8, 2),
            image_instruction(40, "ImageSample", 16, 8, 1),
            image_instruction(48, "ImageStore", 0, 0, 1),
            image_instruction(56, "ImageGather4", 0, 8, 1),
            image_instruction(64, "ImageGather4C", 0, 8, 1),
            end(72),
        ],
        registers,
    );

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("image resource discovery");
    let slots = evaluation.image_binding_slots();

    assert_eq!(evaluation.image_bindings.len(), 9);
    assert_eq!(slots.texture_slots, [0, 0, 1, 2, 3, 4, 5, 0, 2]);
    assert_eq!(
        slots.sampler_slots,
        [
            Some(0),
            Some(0),
            None,
            Some(1),
            Some(0),
            Some(0),
            None,
            Some(0),
            Some(1)
        ]
    );
    assert_eq!(slots.texture_count, 6);
    assert_eq!(slots.sampler_count, 2);
}
