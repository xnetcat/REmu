use super::*;

#[test]
fn classifies_operands_without_conflating_literals_and_inline_constants() {
    assert_eq!(Operand::source(12, None), Operand::scalar(12));
    assert_eq!(Operand::source(256, None), Operand::vector(0));
    assert_eq!(
        Operand::source(249, Some(0xdead_beef)),
        Operand {
            kind: OperandKind::LiteralConstant,
            value: 0xdead_beef,
        }
    );
    assert_eq!(
        Operand::source(242, None),
        Operand {
            kind: OperandKind::EncodedConstant,
            value: 242,
        }
    );
}

#[test]
fn computes_pixel_masks_and_runtime_scalar_registers_once() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sop2,
                opcode: Opcode::from_static("SAddU32"),
                words: vec![0],
                sources: vec![Operand::scalar(4), Operand::vector(1)],
                destinations: vec![Operand::scalar(6)],
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Exp,
                opcode: Opcode::EXPORT,
                words: vec![0, 0],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: Some(InstructionControl::Export(ExportControl {
                    target: 2,
                    enable_mask: 0b1011,
                    compressed: false,
                    done: true,
                    valid_mask: false,
                })),
            },
        ],
    );
    assert_eq!(program.pixel_color_export_masks(), 0x0b00);
    assert!(
        program
            .runtime_scalar_registers()
            .is_superset(&BTreeSet::from([4, 5, 6, 106, 107, 124, 125, 126, 127,]))
    );
    assert!(program.input_scalar_registers().contains(&4));
    assert!(!program.input_scalar_registers().contains(&6));
}

#[test]
fn straight_line_temporaries_do_not_become_shader_inputs() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::from_static("SMovB32"),
                words: vec![0],
                sources: vec![Operand::scalar(4)],
                destinations: vec![Operand::scalar(6)],
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Sop2,
                opcode: Opcode::from_static("SAddU32"),
                words: vec![0],
                sources: vec![Operand::scalar(6), Operand::scalar(8)],
                destinations: vec![Operand::scalar(10)],
                control: None,
            },
        ],
    );
    assert!(program.input_scalar_registers().contains(&4));
    assert!(program.input_scalar_registers().contains(&8));
    assert!(!program.input_scalar_registers().contains(&6));
    assert!(!program.input_scalar_registers().contains(&10));
}

#[test]
fn vector_only_scalar_sources_do_not_become_evaluation_inputs() {
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop2,
            opcode: Opcode::from_static("VAddF32"),
            words: vec![0],
            sources: vec![Operand::scalar(4), Operand::vector(1)],
            destinations: vec![Operand::vector(0)],
            control: None,
        }],
    );
    assert!(program.runtime_scalar_registers().contains(&4));
    assert!(!program.input_scalar_registers().contains(&4));
}

#[test]
fn recurrence_liveness_tracks_only_observable_scalar_load_offsets() {
    let scalar_load = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Smem,
        opcode: Opcode::from_static("SLoadDwordx2"),
        words: vec![0, 0],
        sources: vec![Operand::scalar(4)],
        destinations: vec![Operand::scalar(16), Operand::scalar(17)],
        control: Some(InstructionControl::ScalarMemory(ScalarMemoryControl {
            destination_count: 2,
            immediate_offset_bytes: 0,
            dynamic_offset_register: Some(12),
        })),
    };
    let end = Instruction {
        pc: 16,
        encoding: ShaderEncoding::Sopp,
        opcode: Opcode::END_PROGRAM,
        words: vec![0],
        sources: Vec::new(),
        destinations: Vec::new(),
        control: None,
    };

    let dead = Program::new(0x1000, vec![scalar_load.clone(), end.clone()]);
    let dead_live = dead.evaluation_live_scalar_register_indices(0);
    assert!(dead_live.contains(&4) && dead_live.contains(&5));
    assert!(!dead_live.contains(&12));

    let global = Instruction {
        pc: 8,
        encoding: ShaderEncoding::Flat,
        opcode: Opcode::from_static("GlobalLoadDword"),
        words: vec![0, 0],
        sources: Vec::new(),
        destinations: Vec::new(),
        control: Some(InstructionControl::GlobalMemory(GlobalMemoryControl {
            dword_count: 1,
            vector_address: 0,
            vector_data: 0,
            vector_destination: 0,
            scalar_address: Some(16),
            offset_bytes: 0,
            dlc: false,
            lds: false,
            glc: false,
            slc: false,
        })),
    };
    let live = Program::new(0x1000, vec![scalar_load, global, end]);
    assert!(
        live.evaluation_live_scalar_register_indices(0)
            .contains(&12),
        "offset is observable when the loaded pointer selects a later binding"
    );
}

#[test]
fn branched_programs_keep_conservative_shader_inputs() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::from_static("SMovB32"),
                words: vec![0],
                sources: vec![Operand::scalar(4)],
                destinations: vec![Operand::scalar(6)],
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::from_static("SCbranchScc0"),
                words: vec![0],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Sop2,
                opcode: Opcode::from_static("SAddU32"),
                words: vec![0],
                sources: vec![Operand::scalar(6), Operand::scalar(8)],
                destinations: vec![Operand::scalar(10)],
                control: None,
            },
        ],
    );
    assert!(program.input_scalar_registers().contains(&6));
}

#[test]
fn links_setpc_to_a_separately_allocated_lower_address() {
    let front = Program::new(
        0x0080_7b55_0000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::from_static("SNop"),
                words: vec![0xbf80_0000],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
            Instruction {
                pc: 0x16ac,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::from_static("SSetpcB64"),
                words: vec![0xbe80_2006],
                sources: vec![Operand::scalar(6)],
                destinations: Vec::new(),
                control: None,
            },
        ],
    );
    let continuation = Program::new(
        0x0080_7b3e_0000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::from_static("SGetpcB64"),
                words: vec![0xbe80_1f14],
                sources: Vec::new(),
                destinations: vec![Operand::scalar(20)],
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
        ],
    );

    let linked = link_program_continuation(&front, &continuation).expect("registered continuation");

    assert_eq!(linked.address, front.address);
    assert_eq!(linked.linked_transfer_target(0x16ac), Some(0x16b0));
    assert_eq!(linked.instructions[2].pc, 0x16b0);
    assert_eq!(
        linked.instruction_address(linked.instructions[2].pc),
        continuation.address
    );
    assert_eq!(
        linked.instruction_address(linked.instructions[3].pc),
        continuation.address + 4
    );
    assert_ne!(
        front, linked,
        "a linked program has a distinct cache identity"
    );
}

#[test]
fn refuses_physically_overlapping_linked_allocations() {
    let front = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0,
            encoding: ShaderEncoding::Sop1,
            opcode: Opcode::from_static("SSetpcB64"),
            words: vec![0xbe80_2006],
            sources: vec![Operand::scalar(6)],
            destinations: Vec::new(),
            control: None,
        }],
    );
    let continuation = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0,
            encoding: ShaderEncoding::Sopp,
            opcode: Opcode::END_PROGRAM,
            words: vec![0xbf81_0000],
            sources: Vec::new(),
            destinations: Vec::new(),
            control: None,
        }],
    );

    assert_eq!(
        link_program_continuation(&front, &continuation),
        Err(ProgramLinkError::OverlappingPrograms {
            front_start: 0x1000,
            front_end: 0x1004,
            continuation_start: 0x1000,
            continuation_end: 0x1004,
        })
    );
}

#[test]
fn linked_continuation_operations_are_visible_to_whole_program_classifiers() {
    let front = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0,
            encoding: ShaderEncoding::Sop1,
            opcode: Opcode::from_static("SSetpcB64"),
            words: vec![0xbe80_2006],
            sources: vec![Operand::scalar(6)],
            destinations: Vec::new(),
            control: None,
        }],
    );
    let continuation = Program::new(
        0x4000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::from_static("SSendmsg"),
                words: vec![0],
                sources: vec![Operand {
                    kind: OperandKind::EncodedConstant,
                    value: 9,
                }],
                destinations: Vec::new(),
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Exp,
                opcode: Opcode::EXPORT,
                words: vec![0, 0],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: Some(InstructionControl::Export(ExportControl {
                    target: 20,
                    enable_mask: 1,
                    compressed: false,
                    done: true,
                    valid_mask: false,
                })),
            },
            Instruction {
                pc: 12,
                encoding: ShaderEncoding::Exp,
                opcode: Opcode::EXPORT,
                words: vec![0, 0],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: Some(InstructionControl::Export(ExportControl {
                    target: 12,
                    enable_mask: 0x0f,
                    compressed: false,
                    done: true,
                    valid_mask: false,
                })),
            },
        ],
    );

    let linked = link_program_continuation(&front, &continuation).expect("registered continuation");

    assert!(linked.instructions.iter().any(|instruction| {
        instruction.opcode.is("SSendmsg")
            && instruction
                .sources
                .first()
                .is_some_and(|source| source.value == 9)
    }));
    assert!(linked.instructions.iter().any(|instruction| {
        matches!(
            instruction.control,
            Some(InstructionControl::Export(ExportControl {
                target: 20,
                enable_mask: 1,
                ..
            }))
        )
    }));
    assert!(linked.instructions.iter().any(|instruction| {
        matches!(
            instruction.control,
            Some(InstructionControl::Export(ExportControl {
                target: 12,
                enable_mask: 0x0f,
                ..
            }))
        )
    }));
}

#[test]
fn refuses_to_infer_a_continuation_without_an_explicit_setpc() {
    let front = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0,
            encoding: ShaderEncoding::Sopp,
            opcode: Opcode::END_PROGRAM,
            words: vec![0xbf81_0000],
            sources: Vec::new(),
            destinations: Vec::new(),
            control: None,
        }],
    );
    let continuation = Program::new(
        0x2000,
        vec![Instruction {
            pc: 0,
            encoding: ShaderEncoding::Sopp,
            opcode: Opcode::END_PROGRAM,
            words: vec![0xbf81_0000],
            sources: Vec::new(),
            destinations: Vec::new(),
            control: None,
        }],
    );

    assert_eq!(
        link_program_continuation(&front, &continuation),
        Err(ProgramLinkError::MissingSetpc)
    );
}

fn image_binding(dimension: u32, is_array: bool, opcode: &'static str, word3: u32) -> ImageBinding {
    let mut resource_descriptor = smallvec::SmallVec::from_slice(&[0_u32; 8]);
    resource_descriptor[0] = 0x1000; // non-null base so the descriptor is live
    resource_descriptor[3] = word3;
    ImageBinding {
        pc: 0,
        opcode: Opcode::from_static(opcode),
        control: ImageControl {
            dmask: 0xf,
            vector_address: 0,
            address_registers: vec![0, 1, 2],
            vector_data: 4,
            scalar_resource: 0,
            scalar_sampler: 8,
            dimension,
            is_array,
            glc: false,
            slc: false,
            a16: false,
            d16: false,
            unorm: false,
        },
        resource_descriptor,
        sampler_descriptor: smallvec::SmallVec::new(),
        mip_level: None,
    }
}

#[test]
fn parses_the_descriptor_texture_type_from_dword3() {
    // SDK agc/core/texture.h:249 places `Type m_type : 4` at the top of DWORD3; the values are
    // texture.h:54-63 (k1d=8 .. k2dArrayMsaa=15).
    let cases = [
        (8_u32, TextureDescriptorType::OneD),
        (9, TextureDescriptorType::TwoD),
        (10, TextureDescriptorType::ThreeD),
        (11, TextureDescriptorType::Cube),
        (12, TextureDescriptorType::OneDArray),
        (13, TextureDescriptorType::TwoDArray),
        (14, TextureDescriptorType::TwoDMsaa),
        (15, TextureDescriptorType::TwoDMsaaArray),
    ];
    for (nibble, expected) in cases {
        let binding = image_binding(1, false, "ImageSample", nibble << 28);
        assert_eq!(
            binding.descriptor_type(),
            Some(expected),
            "type nibble {nibble}"
        );
    }
    // Values below k1d are not valid texture descriptors.
    assert_eq!(
        image_binding(1, false, "ImageSample", 0).descriptor_type(),
        None
    );
    assert_eq!(
        image_binding(1, false, "ImageSample", 7 << 28).descriptor_type(),
        None
    );
    // A null descriptor names no resource at all.
    let mut null_binding = image_binding(1, false, "ImageSample", 13 << 28);
    null_binding.resource_descriptor = smallvec::SmallVec::from_slice(&[0_u32; 8]);
    assert_eq!(null_binding.descriptor_type(), None);
}

#[test]
fn classifies_the_instruction_signature_from_the_dim_field() {
    let cases = [
        (0_u32, false, ImageSignatureClass::OneD),
        (1, false, ImageSignatureClass::TwoD),
        (2, false, ImageSignatureClass::ThreeD),
        (3, false, ImageSignatureClass::Cube),
        (4, true, ImageSignatureClass::OneDArray),
        (5, true, ImageSignatureClass::TwoDArray),
        (6, false, ImageSignatureClass::TwoDMsaa),
        (7, true, ImageSignatureClass::TwoDMsaaArray),
    ];
    for (dimension, is_array, expected) in cases {
        let binding = image_binding(dimension, is_array, "ImageSample", 9 << 28);
        assert_eq!(binding.signature_class(), expected, "DIM {dimension}");
    }
}

#[test]
fn reconciles_descriptor_type_against_instruction_dim() {
    // Agreement is direct, including 2D-array storage images (ImageStore with DIM 2DArray on a
    // k2dArray descriptor).
    let direct = image_binding(5, true, "ImageStore", 13 << 28);
    assert_eq!(
        direct.view_reinterpretation(),
        ImageViewReinterpretation::Direct
    );

    // A cubemap addressed with array coordinates is its six-slice 2D-array layout
    // (SDK agc/core/texture.h:321 permits the array view on kCubemap).
    let cube_as_array = image_binding(5, true, "ImageLoad", 11 << 28);
    assert_eq!(
        cube_as_array.view_reinterpretation(),
        ImageViewReinterpretation::CubeFacesAsArray
    );

    // A cube-sampled 2D-array descriptor must be re-viewed as a cube.
    let array_as_cube = image_binding(3, false, "ImageSample", 13 << 28);
    assert_eq!(
        array_as_cube.view_reinterpretation(),
        ImageViewReinterpretation::ArrayAsCube
    );

    // Non-array DIM over an array descriptor selects the base slice; array DIM over a
    // single-layer descriptor is a one-element array.
    assert_eq!(
        image_binding(1, false, "ImageSample", 13 << 28).view_reinterpretation(),
        ImageViewReinterpretation::ArrayBaseSlice
    );
    assert_eq!(
        image_binding(5, true, "ImageSample", 9 << 28).view_reinterpretation(),
        ImageViewReinterpretation::SingleLayerAsArray
    );

    // A null or invalid descriptor follows the instruction shape directly.
    assert_eq!(
        image_binding(5, true, "ImageSample", 0).view_reinterpretation(),
        ImageViewReinterpretation::Direct
    );

    // Anything else is a real mismatch the backend must refuse loudly, never guess.
    assert_eq!(
        image_binding(2, false, "ImageSample", 9 << 28).view_reinterpretation(),
        ImageViewReinterpretation::Incompatible
    );
}
