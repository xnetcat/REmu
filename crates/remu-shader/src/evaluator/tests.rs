use std::sync::Arc;

use super::*;
use crate::{ExportControl, Program};

#[derive(Default)]
struct TestMemory(BTreeMap<u64, u8>);

impl TestMemory {
    fn write(&mut self, address: u64, bytes: &[u8]) {
        for (offset, byte) in bytes.iter().copied().enumerate() {
            self.0.insert(address + offset as u64, byte);
        }
    }
}

impl GuestMemoryReader for TestMemory {
    fn read(&self, address: u64, output: &mut [u8]) -> bool {
        for (offset, destination) in output.iter_mut().enumerate() {
            let Some(byte) = self.0.get(&(address + offset as u64)) else {
                return false;
            };
            *destination = *byte;
        }
        true
    }
}

fn instruction(
    pc: u32,
    encoding: ShaderEncoding,
    opcode: &'static str,
    sources: Vec<Operand>,
    destinations: Vec<Operand>,
    control: Option<InstructionControl>,
) -> Instruction {
    Instruction {
        pc,
        encoding,
        opcode: Opcode::from_static(opcode),
        words: vec![0],
        sources,
        destinations,
        control,
    }
}

fn end(pc: u32) -> Instruction {
    instruction(
        pc,
        ShaderEncoding::Sopp,
        "SEndpgm",
        Vec::new(),
        Vec::new(),
        None,
    )
}

fn state(instructions: Vec<Instruction>, user_data: Vec<u32>) -> ShaderState {
    ShaderState {
        program: Arc::new(Program::new(0x1000, instructions)),
        user_data: user_data.into(),
        compute_system_registers: None,
        user_data_scalar_register_base: 0,
    }
}

#[test]
fn initializes_merged_export_wave_info_before_user_sgprs() {
    let shader = ShaderState {
        program: Arc::new(Program::new(
            0x1000,
            vec![
                instruction(
                    0,
                    ShaderEncoding::Sop1,
                    "SMovB32",
                    vec![Operand::scalar(3)],
                    vec![Operand::scalar(4)],
                    None,
                ),
                instruction(
                    4,
                    ShaderEncoding::Sop1,
                    "SMovB32",
                    vec![Operand::scalar(8)],
                    vec![Operand::scalar(5)],
                    None,
                ),
                end(8),
            ],
        )),
        user_data: vec![0x1234_5678].into(),
        compute_system_registers: None,
        user_data_scalar_register_base: 8,
    };

    let evaluation = evaluate_shader(&TestMemory::default(), &shader, Default::default())
        .expect("merged export evaluation");

    assert_eq!(evaluation.initial_scalar_registers[3], 0x4040);
    assert_eq!(evaluation.scalar_registers[4], 0x4040);
    assert_eq!(evaluation.initial_scalar_registers[8], 0x1234_5678);
    assert_eq!(evaluation.scalar_registers[5], 0x1234_5678);
}

#[test]
fn s_bcnt1_i32_b64_counts_the_source_pair_and_updates_scc() {
    let instruction = instruction(
        0,
        ShaderEncoding::Sop1,
        "SBcnt1I32B64",
        vec![Operand::scalar(0)],
        vec![Operand::scalar(10)],
        None,
    );
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    let mut exec_mask = u64::MAX;
    let mut scalar_condition = false;

    registers[0] = 0xf0f0_0001;
    registers[1] = 0x8000_ffff;
    execute_scalar_alu(
        &instruction,
        0x1000,
        &mut registers,
        &mut exec_mask,
        &mut scalar_condition,
    )
    .expect("64-bit bit count");
    let source = u64::from(registers[0]) | (u64::from(registers[1]) << 32);
    assert_eq!(registers[10], source.count_ones());
    assert!(scalar_condition);

    registers[0] = 0;
    registers[1] = 0;
    scalar_condition = true;
    execute_scalar_alu(
        &instruction,
        0x1000,
        &mut registers,
        &mut exec_mask,
        &mut scalar_condition,
    )
    .expect("zero 64-bit bit count");
    assert_eq!(registers[10], 0);
    assert!(!scalar_condition);
}

#[test]
fn s_bitcmp1_b32_uses_the_low_five_index_bits() {
    let instruction = instruction(
        0,
        ShaderEncoding::Sopc,
        "SBitcmp1B32",
        vec![Operand::scalar(0), Operand::scalar(1)],
        Vec::new(),
        None,
    );
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    registers[0] = 1 << 7;

    for index in [7_u32, 39_u32] {
        registers[1] = index;
        assert!(
            execute_scalar_compare(&instruction, &registers, u64::MAX)
                .expect("S_BITCMP1_B32 scalar evaluation"),
            "bit index {index} must select source bit 7"
        );
    }
    registers[1] = 6;
    assert!(
        !execute_scalar_compare(&instruction, &registers, u64::MAX)
            .expect("clear S_BITCMP1_B32 scalar evaluation")
    );
}

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
            user_data[12] = sentinel as u32;
            user_data[13] = (sentinel >> 32) as u32;
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
            user_data[8] = sentinel as u32;
            user_data[9] = (sentinel >> 32) as u32;
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
                user_data[12] = sentinel as u32;
                user_data[13] = (sentinel >> 32) as u32;
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
                user_data[8] = sentinel as u32;
                user_data[9] = (sentinel >> 32) as u32;
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

#[test]
fn evaluates_scalar_arithmetic_and_taken_control_flow() {
    let mut branch = instruction(
        12,
        ShaderEncoding::Sopp,
        "SCbranchScc1",
        vec![Operand {
            kind: OperandKind::EncodedConstant,
            value: 1,
        }],
        Vec::new(),
        None,
    );
    branch.words[0] = 1;
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Sopk,
                "SMovkI32",
                vec![Operand {
                    kind: OperandKind::EncodedConstant,
                    value: 5,
                }],
                vec![Operand::scalar(0)],
                None,
            ),
            instruction(
                4,
                ShaderEncoding::Sopk,
                "SAddkI32",
                vec![Operand {
                    kind: OperandKind::EncodedConstant,
                    value: u32::from(u16::MAX),
                }],
                vec![Operand::scalar(0)],
                None,
            ),
            instruction(
                8,
                ShaderEncoding::Sopk,
                "SCmpkEqI32",
                vec![Operand {
                    kind: OperandKind::EncodedConstant,
                    value: 4,
                }],
                vec![Operand::scalar(0)],
                None,
            ),
            branch,
            instruction(
                16,
                ShaderEncoding::Sopk,
                "SMovkI32",
                vec![Operand {
                    kind: OperandKind::EncodedConstant,
                    value: 7,
                }],
                vec![Operand::scalar(1)],
                None,
            ),
            instruction(
                20,
                ShaderEncoding::Sopk,
                "SMovkI32",
                vec![Operand {
                    kind: OperandKind::EncodedConstant,
                    value: 9,
                }],
                vec![Operand::scalar(1)],
                None,
            ),
            end(24),
        ],
        Vec::new(),
    );
    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("scalar evaluation");
    assert_eq!(evaluation.scalar_registers[0], 4);
    assert_eq!(evaluation.scalar_registers[1], 9);
}

#[test]
fn scalar_absdiff_wraps_at_i32_width_and_sets_scc_from_the_result() {
    let operation = instruction(
        0,
        ShaderEncoding::Sop2,
        "SAbsdiffI32",
        vec![Operand::scalar(0), Operand::scalar(1)],
        vec![Operand::scalar(2)],
        None,
    );
    for (left, right, expected, expected_scc) in [
        (0x0000_0002, 0x0000_0005, 0x0000_0003, true),
        (0xffff_ffff, 0x0000_0000, 0x0000_0001, true),
        (0x8000_0000, 0x0000_0000, 0x8000_0000, true),
        (0x8000_0000, 0x0000_0001, 0x7fff_ffff, true),
        (0x8000_0000, 0xffff_ffff, 0x7fff_ffff, true),
        (0x8000_0000, 0xffff_fffe, 0x7fff_fffe, true),
        (0x1234_5678, 0x1234_5678, 0x0000_0000, false),
    ] {
        let mut scc = !expected_scc;
        assert_eq!(
            execute_scalar_binary(&operation, left, right, &mut scc)
                .expect("S_ABSDIFF_I32 evaluates"),
            expected,
            "left=0x{left:08X}, right=0x{right:08X}"
        );
        assert_eq!(scc, expected_scc, "left=0x{left:08X}, right=0x{right:08X}");
    }
}

#[test]
fn evaluates_unsigned_64_bit_scalar_comparisons_as_register_pairs() {
    let mut registers = [0_u32; SCALAR_REGISTER_COUNT];
    registers[10] = 0x89ab_cdef;
    registers[11] = 0x0123_4567;
    registers[20] = 0x89ab_cdef;
    registers[21] = 0x0123_4567;
    let compare = instruction(
        0,
        ShaderEncoding::Sopc,
        "SCmpEqU64",
        vec![Operand::scalar(10), Operand::scalar(20)],
        Vec::new(),
        None,
    );
    assert!(
        execute_scalar_compare(&compare, &registers, u64::MAX)
            .expect("equal register pairs compare")
    );

    registers[21] ^= 1;
    let not_equal = Instruction {
        opcode: Opcode::from_static("SCmpLgU64"),
        ..compare
    };
    assert!(
        execute_scalar_compare(&not_equal, &registers, u64::MAX)
            .expect("different register pairs compare")
    );
}

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
fn preserves_initial_image_descriptors_for_statically_inactive_blocks() {
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
    let mut branch = instruction(
        4,
        ShaderEncoding::Sopp,
        "SCbranchScc1",
        Vec::new(),
        Vec::new(),
        None,
    );
    branch.words[0] = 2;
    let mut registers = (1..=12).collect::<Vec<_>>();
    registers.resize(21, 0);
    registers[20] = 1;
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
                vec![Operand::scalar(20)],
                None,
            ),
            branch,
            instruction(
                8,
                ShaderEncoding::Mimg,
                "ImageSampleLz",
                Vec::new(),
                Vec::new(),
                Some(InstructionControl::Image(image)),
            ),
            end(16),
        ],
        registers,
    );

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("inactive image descriptor should remain available to Metal lowering");

    assert_eq!(evaluation.image_bindings.len(), 1);
    assert_eq!(evaluation.image_bindings[0].pc, 8);
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
fn bounds_vector_controlled_backward_loops_during_scalar_discovery() {
    let mut branch = instruction(
        4,
        ShaderEncoding::Sopp,
        "SCbranchExecnz",
        Vec::new(),
        Vec::new(),
        None,
    );
    branch.words[0] = u32::from((-2_i16).cast_unsigned());
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Vop1,
                "VMovB32",
                vec![Operand::vector(0)],
                vec![Operand::vector(1)],
                None,
            ),
            branch,
            end(8),
        ],
        Vec::new(),
    );

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("vector-controlled runtime loop must not exhaust static discovery");
    assert!(evaluation.global_memory_bindings.is_empty());
}

#[test]
fn conversion_only_vector_alu_is_skipped_by_scalar_resource_evaluation() {
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Vop1,
                "VCvtFlrI32F32",
                vec![Operand::vector(1)],
                vec![Operand::vector(8)],
                None,
            ),
            instruction(
                4,
                ShaderEncoding::Vop3,
                "VCvtPknormU16F32",
                vec![
                    Operand::vector(2),
                    Operand::vector(3),
                    Operand::source(128, None),
                ],
                vec![Operand::vector(9)],
                Some(InstructionControl::Vop3(crate::Vop3Control {
                    absolute_mask: 0,
                    negate_mask: 0,
                    output_modifier: 0,
                    clamp: false,
                    operand_select: 0,
                    scalar_destination: None,
                })),
            ),
            end(12),
        ],
        Vec::new(),
    );

    let evaluation = evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("vector conversions must not enter scalar resource evaluation");
    assert!(evaluation.scalar_memory_reads.is_empty());
    assert!(evaluation.global_memory_bindings.is_empty());
    assert!(evaluation.image_bindings.is_empty());
}

#[test]
fn exits_an_unconditional_back_edge_after_discovering_the_vector_loop_once() {
    let mut exit = instruction(
        4,
        ShaderEncoding::Sopp,
        "SCbranchExecz",
        Vec::new(),
        Vec::new(),
        None,
    );
    exit.words[0] = 2;
    let mut back_edge = instruction(
        12,
        ShaderEncoding::Sopp,
        "SBranch",
        Vec::new(),
        Vec::new(),
        None,
    );
    back_edge.words[0] = u32::from((-3_i16).cast_unsigned());
    let shader = state(
        vec![
            instruction(
                0,
                ShaderEncoding::Vop1,
                "VMovB32",
                vec![Operand::vector(0)],
                vec![Operand::vector(1)],
                None,
            ),
            exit,
            instruction(
                8,
                ShaderEncoding::Vop1,
                "VMovB32",
                vec![Operand::vector(1)],
                vec![Operand::vector(2)],
                None,
            ),
            back_edge,
            end(16),
        ],
        Vec::new(),
    );

    evaluate_shader(
        &TestMemory::default(),
        &shader,
        EvaluationOptions::default(),
    )
    .expect("unconditional vector loop back edge must remain bounded");
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
