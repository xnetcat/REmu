// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Buffer and global memory binding tests: descriptor sizing, null and
//! unreadable resources, and the record range a draw's vertex buffers cover.

use super::*;
use crate::evaluator::resources::required_vertex_bytes;

#[test]
fn sizes_structured_buffer_descriptors_in_records() {
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    registers[..4].copy_from_slice(&[0x2000, 16 << 16, 32, 0]);

    let descriptor = decode_buffer_descriptor(&registers, 0).expect("descriptor");

    assert_eq!(descriptor.base_address, 0x2000);
    assert_eq!(descriptor.stride, 16);
    assert_eq!(descriptor.record_count, 32);
    assert_eq!(descriptor.size_bytes, 512);
}

#[test]
fn sizes_raw_buffer_descriptors_in_bytes() {
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    registers[..4].copy_from_slice(&[0x2000, 16 << 16, 512, 3 << 28]);

    let descriptor = decode_buffer_descriptor(&registers, 0).expect("descriptor");

    assert_eq!(descriptor.stride, 16);
    assert_eq!(descriptor.record_count, 512);
    assert_eq!(descriptor.size_bytes, 512);
}

#[test]
fn extends_vertex_bindings_across_draw_instances_for_all_oob_modes() {
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    registers[..4].copy_from_slice(&[0x2000, 24 << 16, 4, 2 << 28]);
    let descriptor = decode_buffer_descriptor(&registers, 0).expect("descriptor");

    assert_eq!(descriptor.oob_select, 2);
    assert_eq!(descriptor.size_bytes, 96);
    assert_eq!(
        required_vertex_bytes(descriptor, 0, 4, Some(4 * 64)),
        24 * 4 * 64,
        "OOB_SELECT=2 permits instance-indexed records beyond NUM_RECORDS"
    );

    registers[3] = 1 << 28;
    let bounded = decode_buffer_descriptor(&registers, 0).expect("bounded descriptor");
    assert_eq!(
        required_vertex_bytes(bounded, 0, 4, Some(4 * 64)),
        24 * 4 * 64,
        "host binding length must not preempt the shader's OOB selection semantics"
    );
}

#[test]
fn sizes_swizzled_buffers_in_records_even_with_raw_oob_selection() {
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    registers[..4].copy_from_slice(&[0x2000, (16 << 16) | (1 << 31), 32, 3 << 28]);

    let descriptor = decode_buffer_descriptor(&registers, 0).expect("descriptor");

    assert!(descriptor.swizzle_enabled);
    assert_eq!(descriptor.size_bytes, 512);
}

#[test]
fn accepts_gfx10_buffer_descriptors_with_resource_level() {
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    registers[..4].copy_from_slice(&[0x8065_a700, 0xcb60_0000, 0x0046_c19b, 0x9050_0fac]);

    let descriptor = decode_buffer_descriptor(&registers, 0)
        .expect("RESOURCE_LEVEL is valid in an RDNA2 buffer descriptor");

    assert_eq!(descriptor.base_address, 0x8065_a700);
    assert_eq!(descriptor.stride, 0xb60);
    assert!(descriptor.swizzle_enabled);
    assert_eq!(descriptor.record_count, 0x0046_c19b);
    assert_ne!(descriptor.size_bytes, 0);
}

#[test]
fn keeps_null_global_resources_as_bounded_zero_bindings() {
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Flat,
                "GlobalLoadDword",
                Vec::new(),
                vec![Operand::vector(0)],
                Some(InstructionControl::GlobalMemory(GlobalMemoryControl {
                    dword_count: 1,
                    vector_address: 0,
                    vector_data: 0,
                    vector_destination: 0,
                    scalar_address: Some(12),
                    offset_bytes: 0,
                    dlc: false,
                    lds: false,
                    glc: false,
                    slc: false,
                })),
            ),
            end(8),
        ],
        vec![0; 14],
    );

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("null global resource must not discard the shader");

    assert_eq!(evaluation.global_memory_bindings.len(), 1);
    assert_eq!(evaluation.global_memory_bindings[0].scalar_address, 12);
    assert_eq!(evaluation.global_memory_bindings[0].base_address, 0);
    assert_eq!(evaluation.global_memory_bindings[0].data, vec![0; 4]);
    assert!(!evaluation.global_memory_bindings[0].writable);
    assert!(!evaluation.global_memory_bindings[0].write_back_to_guest);
}

#[test]
fn keeps_null_buffer_resources_as_bounded_zero_bindings() {
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Mubuf,
                "BufferLoadDword",
                Vec::new(),
                vec![Operand::vector(0)],
                Some(InstructionControl::BufferMemory(BufferMemoryControl {
                    dword_count: 1,
                    vector_address: 0,
                    vector_data: 0,
                    scalar_resource: 8,
                    instruction_format: None,
                    offset_bytes: 0,
                    index_enabled: false,
                    offset_enabled: false,
                    dlc: false,
                    lds: false,
                    glc: false,
                    slc: false,
                    tfe: false,
                })),
            ),
            end(8),
        ],
        vec![0; 12],
    );

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("null buffer resource must not discard the shader");

    assert_eq!(evaluation.global_memory_bindings.len(), 1);
    assert_eq!(evaluation.global_memory_bindings[0].scalar_address, 8);
    assert_eq!(evaluation.global_memory_bindings[0].base_address, 0);
    assert_eq!(evaluation.global_memory_bindings[0].data, vec![0; 4]);
    assert!(!evaluation.global_memory_bindings[0].writable);
    assert!(!evaluation.global_memory_bindings[0].write_back_to_guest);
}

#[test]
fn noncanonical_buffer_loads_use_a_bounded_zero_binding() {
    let shader = state(
        vec![
            instruction(
                0x904,
                ShaderEncoding::Mubuf,
                "BufferLoadDword",
                Vec::new(),
                vec![Operand::vector(0)],
                Some(InstructionControl::BufferMemory(BufferMemoryControl {
                    dword_count: 1,
                    vector_address: 0,
                    vector_data: 0,
                    scalar_resource: 8,
                    instruction_format: None,
                    offset_bytes: 0,
                    index_enabled: false,
                    offset_enabled: false,
                    dlc: false,
                    lds: false,
                    glc: false,
                    slc: false,
                    tfe: false,
                })),
            ),
            end(0x90c),
        ],
        {
            let mut user_data = vec![0; 12];
            user_data[8..12].copy_from_slice(&[1, u32::MAX, 1, u32::MAX]);
            user_data
        },
    );

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("a noncanonical read-only buffer sharp must resolve as null");

    assert_eq!(evaluation.global_memory_bindings.len(), 1);
    assert_eq!(evaluation.global_memory_bindings[0].scalar_address, 8);
    assert_eq!(evaluation.global_memory_bindings[0].base_address, 0);
    assert_eq!(evaluation.global_memory_bindings[0].byte_count, 4);
    assert_eq!(evaluation.global_memory_bindings[0].data, vec![0; 4]);
    assert!(!evaluation.global_memory_bindings[0].writable);
    assert!(!evaluation.global_memory_bindings[0].write_back_to_guest);
}

#[test]
fn noncanonical_buffer_stores_and_atomics_remain_errors() {
    for opcode in ["BufferStoreDword", "BufferAtomicAdd"] {
        let shader = state(
            vec![
                instruction(
                    0x904,
                    ShaderEncoding::Mubuf,
                    opcode,
                    Vec::new(),
                    Vec::new(),
                    Some(InstructionControl::BufferMemory(BufferMemoryControl {
                        dword_count: 1,
                        vector_address: 0,
                        vector_data: 0,
                        scalar_resource: 8,
                        instruction_format: None,
                        offset_bytes: 0,
                        index_enabled: false,
                        offset_enabled: false,
                        dlc: false,
                        lds: false,
                        glc: false,
                        slc: false,
                        tfe: false,
                    })),
                ),
                end(0x90c),
            ],
            {
                let mut user_data = vec![0; 12];
                user_data[8..12].copy_from_slice(&[1, u32::MAX, 1, u32::MAX]);
                user_data
            },
        );

        assert!(matches!(
            evaluate_shader(
                &TestMemory::default(),
                &shader,
                EvaluationOptions::default(),
            ),
            Err(EvaluationError::InvalidBufferDescriptor {
                pc: 0x904,
                register: 8,
                words0: 1,
                words1: u32::MAX,
                words2: 1,
                words3: u32::MAX,
            })
        ));
    }
}

#[test]
fn unreadable_global_loads_use_a_bounded_zero_binding() {
    let sentinel = 0xf000_0000_0090_u64;
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Flat,
                "GlobalLoadDword",
                Vec::new(),
                vec![Operand::vector(0)],
                Some(InstructionControl::GlobalMemory(GlobalMemoryControl {
                    dword_count: 1,
                    vector_address: 0,
                    vector_data: 0,
                    vector_destination: 0,
                    scalar_address: Some(12),
                    offset_bytes: 0,
                    dlc: false,
                    lds: false,
                    glc: false,
                    slc: false,
                })),
            ),
            end(8),
        ],
        {
            let mut user_data = vec![0; 14];
            user_data[12] = u32::try_from(sentinel & u64::from(u32::MAX)).unwrap();
            user_data[13] = u32::try_from(sentinel >> 32).unwrap();
            user_data
        },
    );

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions {
            maximum_binding_bytes: 64,
            ..EvaluationOptions::default()
        },
    )
    .expect("an unreadable read-only global must receive a zero binding");

    assert_eq!(evaluation.global_memory_bindings.len(), 1);
    assert_eq!(evaluation.global_memory_bindings[0].scalar_address, 12);
    assert_eq!(evaluation.global_memory_bindings[0].base_address, 0);
    assert_eq!(evaluation.global_memory_bindings[0].byte_count, 4);
    assert_eq!(evaluation.global_memory_bindings[0].data, vec![0; 4]);
    assert!(!evaluation.global_memory_bindings[0].writable);
    assert!(!evaluation.global_memory_bindings[0].write_back_to_guest);
}

#[test]
fn unreadable_buffer_loads_use_a_bounded_zero_binding() {
    let sentinel = 0xf000_0000_0090_u64;
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Mubuf,
                "BufferLoadDword",
                Vec::new(),
                vec![Operand::vector(0)],
                Some(InstructionControl::BufferMemory(BufferMemoryControl {
                    dword_count: 1,
                    vector_address: 0,
                    vector_data: 0,
                    scalar_resource: 8,
                    instruction_format: None,
                    offset_bytes: 0,
                    index_enabled: false,
                    offset_enabled: false,
                    dlc: false,
                    lds: false,
                    glc: false,
                    slc: false,
                    tfe: false,
                })),
            ),
            end(8),
        ],
        {
            let mut user_data = vec![0; 12];
            user_data[8] = u32::try_from(sentinel & u64::from(u32::MAX)).unwrap();
            user_data[9] = u32::try_from(sentinel >> 32).unwrap();
            user_data[10] = 16;
            user_data[11] = 3 << 28;
            user_data
        },
    );

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("an unreadable read-only buffer must receive a zero binding");

    assert_eq!(evaluation.global_memory_bindings.len(), 1);
    assert_eq!(evaluation.global_memory_bindings[0].scalar_address, 8);
    assert_eq!(evaluation.global_memory_bindings[0].base_address, 0);
    assert_eq!(evaluation.global_memory_bindings[0].byte_count, 4);
    assert_eq!(evaluation.global_memory_bindings[0].data, vec![0; 4]);
    assert!(!evaluation.global_memory_bindings[0].writable);
    assert!(!evaluation.global_memory_bindings[0].write_back_to_guest);
}

#[test]
fn unreadable_global_stores_and_atomics_remain_errors() {
    let sentinel = 0xf000_0000_0090_u64;
    for opcode in ["GlobalStoreDword", "GlobalAtomicAdd"] {
        let shader = state(
            vec![
                instruction(
                    0,
                    ShaderEncoding::Flat,
                    opcode,
                    Vec::new(),
                    Vec::new(),
                    Some(InstructionControl::GlobalMemory(GlobalMemoryControl {
                        dword_count: 1,
                        vector_address: 0,
                        vector_data: 0,
                        vector_destination: 0,
                        scalar_address: Some(12),
                        offset_bytes: 0,
                        dlc: false,
                        lds: false,
                        glc: false,
                        slc: false,
                    })),
                ),
                end(8),
            ],
            {
                let mut user_data = vec![0; 14];
                user_data[12] = u32::try_from(sentinel & u64::from(u32::MAX)).unwrap();
                user_data[13] = u32::try_from(sentinel >> 32).unwrap();
                user_data
            },
        );

        assert!(matches!(
            evaluate_shader(
                &TestMemory::default(),
                &shader,
                EvaluationOptions {
                    maximum_binding_bytes: 64,
                    ..EvaluationOptions::default()
                },
            ),
            Err(EvaluationError::MemoryFault { address, .. }) if address == sentinel
        ));
    }
}

#[test]
fn unreadable_buffer_stores_and_atomics_remain_errors() {
    let sentinel = 0xf000_0000_0090_u64;
    for opcode in ["BufferStoreDword", "BufferAtomicAdd"] {
        let shader = state(
            vec![
                instruction(
                    0,
                    ShaderEncoding::Mubuf,
                    opcode,
                    Vec::new(),
                    Vec::new(),
                    Some(InstructionControl::BufferMemory(BufferMemoryControl {
                        dword_count: 1,
                        vector_address: 0,
                        vector_data: 0,
                        scalar_resource: 8,
                        instruction_format: None,
                        offset_bytes: 0,
                        index_enabled: false,
                        offset_enabled: false,
                        dlc: false,
                        lds: false,
                        glc: false,
                        slc: false,
                        tfe: false,
                    })),
                ),
                end(8),
            ],
            {
                let mut user_data = vec![0; 12];
                user_data[8] = u32::try_from(sentinel & u64::from(u32::MAX)).unwrap();
                user_data[9] = u32::try_from(sentinel >> 32).unwrap();
                user_data[10] = 16;
                user_data[11] = 3 << 28;
                user_data
            },
        );

        assert!(matches!(
            evaluate_shader(
                &TestMemory::default(),
                &shader,
                EvaluationOptions::default(),
            ),
            Err(EvaluationError::MemoryFault { address, .. }) if address == sentinel
        ));
    }
}

#[test]
fn bounds_runtime_vertex_buffers_to_the_draws_referenced_records() {
    let mut memory = TestMemory::default();
    memory.write(0x2000, &[0; 96]);
    let selectors = 4 | (5 << 3) | (6 << 6) | (7 << 9);
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Mubuf,
                "BufferLoadFormatX",
                vec![
                    Operand::vector(5),
                    Operand::scalar(0),
                    Operand::source(128, None),
                ],
                vec![Operand::vector(7)],
                Some(InstructionControl::BufferMemory(BufferMemoryControl {
                    dword_count: 1,
                    vector_address: 5,
                    vector_data: 7,
                    scalar_resource: 0,
                    instruction_format: None,
                    offset_bytes: 0,
                    index_enabled: true,
                    offset_enabled: false,
                    dlc: false,
                    lds: false,
                    glc: false,
                    slc: false,
                    tfe: false,
                })),
            ),
            end(8),
        ],
        vec![0x2000, 24 << 16, 0xaaaa, selectors | (60 << 12) | (3 << 28)],
    );

    let evaluation = evaluate_shader(
        &memory,
        &shader,
        EvaluationOptions {
            required_vertex_record_count: Some(4),
            ..EvaluationOptions::default()
        },
    )
    .expect("bounded runtime vertex fetch evaluation");

    assert!(evaluation.vertex_inputs.is_empty());
    assert_eq!(evaluation.global_memory_bindings.len(), 1);
    assert_eq!(evaluation.global_memory_bindings[0].data.len(), 96);
}

// ---------------------------------------------------------------------------------------------
// Workstream Q1: SCRATCH spills, the full GLOBAL atomic set, and benign mode-control opcodes must
// all evaluate through resource discovery instead of dropping the whole shader.
// ---------------------------------------------------------------------------------------------

fn scratch_control() -> InstructionControl {
    // SCRATCH addresses private per-lane spill memory; there is no guest scalar base to resolve,
    // so `decode_global_memory` records `scalar_address: None` for the segment.
    InstructionControl::GlobalMemory(GlobalMemoryControl {
        dword_count: 1,
        vector_address: 0,
        vector_data: 1,
        vector_destination: 2,
        scalar_address: None,
        offset_bytes: 0,
        dlc: false,
        lds: false,
        glc: false,
        slc: false,
    })
}

#[test]
fn a_scratch_spill_reload_creates_no_binding_and_keeps_the_shader() {
    // A register spill (ScratchStore) and reload (ScratchLoad) touch private scratch, not a guest
    // resource. Previously the FLAT decoder rejected the segment outright and the whole shader was
    // dropped; now the pair evaluates and contributes no global binding.
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Flat,
                "ScratchStoreDword",
                vec![Operand::vector(0), Operand::vector(1)],
                Vec::new(),
                Some(scratch_control()),
            ),
            instruction(
                8,
                ShaderEncoding::Flat,
                "ScratchLoadDword",
                vec![Operand::vector(0)],
                vec![Operand::vector(2)],
                Some(scratch_control()),
            ),
            end(16),
        ],
        vec![0; 8],
    );

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("a scratch spill must not drop the shader");

    assert!(
        evaluation.global_memory_bindings.is_empty(),
        "scratch is private memory and binds no guest resource"
    );
}

#[test]
fn the_full_global_atomic_set_resolves_a_writable_binding() {
    let base = 0x2000_u64;
    for opcode in [
        "GlobalAtomicSwap",
        "GlobalAtomicCmpswap",
        "GlobalAtomicSub",
        "GlobalAtomicSmin",
        "GlobalAtomicUmin",
        "GlobalAtomicSmax",
        "GlobalAtomicUmax",
        "GlobalAtomicAnd",
        "GlobalAtomicOr",
        "GlobalAtomicXor",
        "GlobalAtomicInc",
        "GlobalAtomicDec",
    ] {
        let shader = state(
            vec![
                instruction(
                    0,
                    ShaderEncoding::Flat,
                    opcode,
                    Vec::new(),
                    Vec::new(),
                    Some(InstructionControl::GlobalMemory(GlobalMemoryControl {
                        dword_count: 1,
                        vector_address: 0,
                        vector_data: 0,
                        vector_destination: 0,
                        scalar_address: Some(12),
                        offset_bytes: 0,
                        dlc: false,
                        lds: false,
                        glc: false,
                        slc: false,
                    })),
                ),
                end(8),
            ],
            {
                let mut user_data = vec![0; 14];
                user_data[12] = u32::try_from(base & u64::from(u32::MAX)).unwrap();
                user_data[13] = u32::try_from(base >> 32).unwrap();
                user_data
            },
        );

        let mut memory = TestMemory::default();
        memory.write(base, &[0; 64]);
        let evaluation = evaluate_shader(
            &memory,
            &shader,
            EvaluationOptions {
                maximum_binding_bytes: 64,
                ..EvaluationOptions::default()
            },
        )
        .unwrap_or_else(|error| panic!("{opcode} must resolve a writable binding: {error}"));

        assert_eq!(evaluation.global_memory_bindings.len(), 1, "{opcode}");
        let binding = &evaluation.global_memory_bindings[0];
        assert_eq!(binding.base_address, base, "{opcode}");
        assert!(binding.writable, "{opcode} binds writable guest memory");
        assert!(binding.write_back_to_guest, "{opcode}");
    }
}

#[test]
fn benign_mode_control_and_hwreg_opcodes_do_not_drop_the_shader() {
    // S_DENORM_MODE / S_ROUND_MODE toggle FP mode; S_SETREG writes an opaque HW register and must
    // not clobber the SGPR the decoder parked in its destination slot; S_GETREG reads an opaque HW
    // register, resolved to a conservative zero.
    let mut user_data = vec![0; 8];
    user_data[5] = 0xdead_beef;
    user_data[6] = 0x1234_5678;
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sopp,
                "SDenormMode",
                Vec::new(),
                Vec::new(),
                None,
            ),
            instruction(
                4,
                ShaderEncoding::Sopp,
                "SRoundMode",
                Vec::new(),
                Vec::new(),
                None,
            ),
            instruction(
                8,
                ShaderEncoding::Sopk,
                "SGetregB32",
                Vec::new(),
                vec![Operand::scalar(5)],
                None,
            ),
            instruction(
                12,
                ShaderEncoding::Sopk,
                "SSetregB32",
                vec![Operand::scalar(6)],
                vec![Operand::scalar(6)],
                None,
            ),
            end(16),
        ],
        user_data,
    );

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("benign mode-control and hwreg opcodes must not drop the shader");

    assert_eq!(
        evaluation.scalar_registers[5], 0,
        "S_GETREG resolves to a conservative zero"
    );
    assert_eq!(
        evaluation.scalar_registers[6], 0x1234_5678,
        "S_SETREG must not clobber its data-source SGPR"
    );
}
