use super::*;

struct SliceMemory {
    base: u64,
    words: Vec<u32>,
}

impl GuestMemoryReader for SliceMemory {
    fn read(&self, address: u64, output: &mut [u8]) -> bool {
        if output.len() != 4 {
            return false;
        }
        let Some(offset) = address.checked_sub(self.base) else {
            return false;
        };
        if offset & 3 != 0 {
            return false;
        }
        let Ok(index) = usize::try_from(offset / 4) else {
            return false;
        };
        let Some(word) = self.words.get(index) else {
            return false;
        };
        output.copy_from_slice(&word.to_le_bytes());
        true
    }
}

#[test]
fn decodes_known_pixel_program_through_architectural_terminator() {
    let program = decode_program(
        0x1234_0000,
        &[
            0xd52f_0000,
            0x0000_0200,
            0xd52f_0001,
            0x0000_0602,
            0xf800_1c0f,
            0x0000_0100,
            0xbf81_0000,
            0xdead_beef,
        ],
    )
    .expect("well-formed shader");
    assert_eq!(program.instructions.len(), 4);
    assert!(program.instructions[0].opcode.is("VCvtPkrtzF16F32"));
    assert_eq!(program.instructions[0].encoding, ShaderEncoding::Vop3);
    assert_eq!(program.instructions[2].opcode, Opcode::EXPORT);
    assert_eq!(program.instructions[3].opcode, Opcode::END_PROGRAM);
    assert_eq!(program.pixel_color_export_masks(), 0x0f);
}

#[test]
fn decodes_past_an_early_endpgm_needed_by_a_forward_branch() {
    let words = [
        0xbf82_0002, // s_branch from pc 0 to pc 12
        0xbf81_0000, // early fallthrough endpgm at pc 4
        0xbf80_0000, // padding/fallthrough instruction at pc 8
        0xbf80_0000, // forward target at pc 12
        0xbf81_0000, // terminating endpgm at pc 16
        0xdead_beef,
    ];

    let program = decode_program(0x1000, &words).expect("forward branch program");

    assert_eq!(program.instructions.len(), 5);
    assert_eq!(program.instructions[1].opcode, Opcode::END_PROGRAM);
    assert_eq!(program.instructions[3].pc, 12);
    assert_eq!(program.instructions[4].opcode, Opcode::END_PROGRAM);

    let memory = SliceMemory {
        base: 0x1000,
        words: words.to_vec(),
    };
    let program = read_program(&memory, memory.base).expect("memory-backed forward branch");
    assert_eq!(program.instructions.len(), 5);
}

#[test]
fn indirect_setpc_return_terminates_before_embedded_allocation_metadata() {
    let words = [
        0xbf88_0002, // s_cbranch_execz from pc 0 to pc 12
        0xbf80_0000,
        0xbf80_0000,
        0xbf8c_c07f, // branch target: s_waitcnt
        0xbe80_2006, // s_setpc_b64 s[6:7]
        0x3030_6c73, // allocation metadata: "sl00"
        0x0000_023e, // metadata shaped like reserved VOP2 opcode 0
    ];

    let program = decode_program(0x0080_db6c_0000, &words)
        .expect("indirect terminal transfer must stop before allocation metadata");

    assert_eq!(program.instructions.len(), 5);
    assert!(program.instructions.last().unwrap().opcode.is("SSetpcB64"));

    let memory = SliceMemory {
        base: 0x0080_db6c_0000,
        words: words.to_vec(),
    };
    let program = read_program_bounded(
        &memory,
        memory.base,
        memory.words.len() * std::mem::size_of::<u32>(),
    )
    .expect("memory-backed decoder must stop at the indirect terminal transfer");
    assert_eq!(program.instructions.len(), 5);

    let segments = read_program_segments_bounded(
        &memory,
        memory.base,
        memory.words.len() * std::mem::size_of::<u32>(),
    )
    .expect("metadata after an indirect transfer is not an adjacent shader entry point");
    assert_eq!(segments.len(), 1);
    assert!(
        segments[0]
            .instructions
            .last()
            .unwrap()
            .opcode
            .is("SSetpcB64")
    );
}

#[test]
fn indirect_setpc_before_a_known_forward_target_does_not_truncate_the_program() {
    let words = [
        0xbf88_0003, // s_cbranch_execz from pc 0 to pc 16
        0xbe80_2006, // fallthrough path transfers indirectly
        0xbf80_0000,
        0xbf80_0000,
        0xbf8c_c07f, // branch target
        0xbe80_2006, // branch path transfers indirectly
        0x0000_023e,
    ];

    let program = decode_program(0x4000, &words)
        .expect("decoder must retain every statically encoded forward branch target");

    assert_eq!(program.instructions.len(), 6);
    assert_eq!(program.instructions[4].pc, 16);
    assert!(program.instructions.last().unwrap().opcode.is("SSetpcB64"));
}

#[test]
fn decodes_adjacent_merged_shader_entry_points_within_one_allocation() {
    let words = [
        0xbf80_0000, // first entry point
        0xbf81_0000,
        0xbf80_0000, // second entry point
        0xbf80_0000,
        0xbf81_0000,
    ];
    let segments = decode_program_segments(0x2000, &words).expect("adjacent shader entry points");
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].address, 0x2000);
    assert_eq!(segments[0].instructions.len(), 2);
    assert_eq!(segments[1].address, 0x2008);
    assert_eq!(segments[1].instructions.len(), 3);

    let memory = SliceMemory {
        base: 0x2000,
        words: words.to_vec(),
    };
    let segments = read_program_segments_bounded(
        &memory,
        memory.base,
        memory.words.len() * std::mem::size_of::<u32>(),
    )
    .expect("memory-backed adjacent entry points");
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[1].address, 0x2008);
}

#[test]
fn registered_code_extent_allows_a_shader_larger_than_the_fallback_scan() {
    let mut words = vec![0xbf80_0000; FALLBACK_MAX_INSTRUCTIONS + 1];
    words.push(0xbf81_0000);
    let memory = SliceMemory {
        base: 0x4000,
        words,
    };
    assert!(matches!(
        read_program(&memory, memory.base),
        Err(DecodeError::Unterminated {
            inspected_instructions: FALLBACK_MAX_INSTRUCTIONS
        })
    ));

    let program = read_program_bounded(
        &memory,
        memory.base,
        memory.words.len() * std::mem::size_of::<u32>(),
    )
    .expect("AGC's registered shader size is the authoritative decoder boundary");
    assert_eq!(program.instructions.len(), FALLBACK_MAX_INSTRUCTIONS + 2);
    assert_eq!(
        program
            .instructions
            .last()
            .map(|instruction| &instruction.opcode),
        Some(&Opcode::END_PROGRAM)
    );
}

#[test]
fn rejects_a_missing_literal_extension_without_reading_past_the_slice() {
    let error = decode_program(0x1000, &[0x7e00_02ff]).expect_err("missing literal");
    assert_eq!(
        error,
        DecodeError::Truncated {
            pc: 0,
            needed_dwords: 2,
            available_dwords: 1,
        }
    );
}

#[test]
fn retains_framed_unknown_vop3_opcode_without_guessing_semantics() {
    let unknown_vop3 = 0xd000_0000;
    let program = decode_program(0x1000, &[unknown_vop3, 0, 0xbf81_0000])
        .expect("unknown VOP3 has an architecturally defined frame");
    assert_eq!(
        program.instructions[0].opcode,
        Opcode::raw(RawOpcodeFamily::Vop3, 0)
    );
    assert_eq!(program.instructions[1].pc, 8);
}

#[test]
fn decodes_silent_hill_xor_add_and_unsigned_pack_vop3_opcodes() {
    let sources = 257_u32 | (258_u32 << 9) | (259_u32 << 18);
    let program = decode_program(
        0x1000,
        &[0xd345_0008, sources, 0xd36a_0009, sources, 0xbf81_0000],
    )
    .expect("observed RDNA2 VOP3 stream");
    assert!(program.instructions[0].opcode.is("VXadU32"));
    assert_eq!(
        program.instructions[0].sources,
        [Operand::vector(1), Operand::vector(2), Operand::vector(3)]
    );
    assert_eq!(program.instructions[0].destinations, [Operand::vector(8)]);
    assert!(program.instructions[1].opcode.is("VCvtPkU16U32"));
    assert_eq!(
        program.instructions[1].sources,
        [Operand::vector(1), Operand::vector(2), Operand::vector(3)]
    );
    assert_eq!(program.instructions[1].destinations, [Operand::vector(9)]);
}

#[test]
fn decodes_exercised_xor3_and_64_bit_vector_shift_pairs() {
    let xor_sources = 257_u32 | (258_u32 << 9) | (259_u32 << 18);
    let shift_sources = 4_u32 | (258_u32 << 9) | (259_u32 << 18);
    let program = decode_program(
        0x1000,
        &[
            0xd178_0008,
            xor_sources,
            0xd2ff_000a,
            shift_sources,
            0xd300_000c,
            shift_sources,
            0xbf81_0000,
        ],
    )
    .expect("observed RDNA2 XOR3 and B64 shift stream");

    assert!(program.instructions[0].opcode.is("VXor3B32"));
    assert_eq!(program.instructions[0].destinations, [Operand::vector(8)]);
    for (instruction, opcode, destination) in [
        (&program.instructions[1], "VLshlrevB64", 10),
        (&program.instructions[2], "VLshrrevB64", 12),
    ] {
        assert!(instruction.opcode.is(opcode));
        assert_eq!(
            instruction.sources,
            [Operand::scalar(4), Operand::vector(2), Operand::vector(3)]
        );
        assert_eq!(
            instruction.destinations,
            [
                Operand::vector(destination),
                Operand::vector(destination + 1)
            ]
        );
    }
}

#[test]
fn decodes_silent_hill_half_to_u16_and_signed_bitfield_extract() {
    let sources = 257_u32 | (258_u32 << 9) | (259_u32 << 18);
    let program = decode_program(0x1000, &[0x7e06_a4f9, 0, 0xd149_0008, sources, 0xbf81_0000])
        .expect("observed RDNA2 VOP1 SDWA and VOP3 stream");

    assert!(program.instructions[0].opcode.is("VCvtU16F16"));
    assert_eq!(program.instructions[0].sources, [Operand::vector(0)]);
    assert_eq!(program.instructions[0].destinations, [Operand::vector(3)]);
    assert!(matches!(
        program.instructions[0].control,
        Some(InstructionControl::Sdwa(_))
    ));

    assert!(program.instructions[1].opcode.is("VBfeI32"));
    assert_eq!(
        program.instructions[1].sources,
        [Operand::vector(1), Operand::vector(2), Operand::vector(3)]
    );
    assert_eq!(program.instructions[1].destinations, [Operand::vector(8)]);
}

#[test]
fn decodes_silent_hill_half_transcendentals_sdwa() {
    let program = decode_program(
        0x8009_c10000,
        &[
            0x7e0c_aef9, // v_log_f16 v6, src_sdwa
            0,
            0x7e12_b0f9, // v_exp_f16 v9, src_sdwa
            0,
            0xbf81_0000,
        ],
    )
    .expect("observed Silent Hill half-transcendental SDWA stream");

    assert!(program.instructions[0].opcode.is("VLogF16"));
    assert_eq!(program.instructions[0].destinations, [Operand::vector(6)]);
    assert!(matches!(
        program.instructions[0].control,
        Some(InstructionControl::Sdwa(_))
    ));
    assert!(program.instructions[1].opcode.is("VExpF16"));
    assert_eq!(program.instructions[1].destinations, [Operand::vector(9)]);
    assert!(matches!(
        program.instructions[1].control,
        Some(InstructionControl::Sdwa(_))
    ));
    assert_eq!(program.instructions[2].opcode, Opcode::END_PROGRAM);
}

#[test]
fn decodes_silent_hill_single_precision_division_fixup() {
    let sources = 257_u32 | (258_u32 << 9) | (259_u32 << 18);
    let program = decode_program(0x8009_c10000, &[0xd351_0008, sources, 0xbf81_0000])
        .expect("observed Silent Hill V_DIV_FIXUP_F32 stream");

    assert!(program.instructions[0].opcode.is("VDivFixupF32"));
    assert_eq!(
        program.instructions[0].sources,
        [Operand::vector(1), Operand::vector(2), Operand::vector(3)]
    );
    assert_eq!(program.instructions[0].destinations, [Operand::vector(8)]);
    assert_eq!(program.instructions[1].opcode, Opcode::END_PROGRAM);
}

#[test]
fn decodes_rdna2_floor_and_packed_unorm_conversion_encodings() {
    let unary_sources = 257_u32 | (128_u32 << 9) | (128_u32 << 18);
    let packed_sources = 257_u32 | (258_u32 << 9) | (128_u32 << 18);
    let program = decode_program(
        0x1000,
        &[
            // v_cvt_flr_i32_f32_e32 v8, v1
            0x7e10_1b01,
            // v_cvt_flr_i32_f32_e64 v9, -|v1|
            0xd18d_0109,
            unary_sources | (1_u32 << 29),
            // v_cvt_pknorm_u16_f32 v10, |v1|, -v2
            0xd369_010a,
            packed_sources | (2_u32 << 29),
            0xbf81_0000,
        ],
    )
    .expect("RDNA2 conversion stream");

    let floor_e32 = &program.instructions[0];
    assert_eq!(floor_e32.encoding, ShaderEncoding::Vop1);
    assert!(floor_e32.opcode.is("VCvtFlrI32F32"));
    assert_eq!(floor_e32.sources, [Operand::vector(1)]);
    assert_eq!(floor_e32.destinations, [Operand::vector(8)]);

    let floor_e64 = &program.instructions[1];
    assert_eq!(floor_e64.encoding, ShaderEncoding::Vop3);
    assert!(floor_e64.opcode.is("VCvtFlrI32F32"));
    assert_eq!(floor_e64.destinations, [Operand::vector(9)]);
    assert_eq!(
        floor_e64.control,
        Some(InstructionControl::Vop3(Vop3Control {
            absolute_mask: 1,
            negate_mask: 1,
            output_modifier: 0,
            clamp: false,
            operand_select: 0,
            scalar_destination: None,
        }))
    );

    let packed = &program.instructions[2];
    assert_eq!(packed.encoding, ShaderEncoding::Vop3);
    assert!(packed.opcode.is("VCvtPknormU16F32"));
    assert_eq!(
        packed.sources[..2],
        [Operand::vector(1), Operand::vector(2)]
    );
    assert_eq!(packed.destinations, [Operand::vector(10)]);
    assert_eq!(
        packed.control,
        Some(InstructionControl::Vop3(Vop3Control {
            absolute_mask: 1,
            negate_mask: 2,
            output_modifier: 0,
            clamp: false,
            operand_select: 0,
            scalar_destination: None,
        }))
    );
}

#[test]
fn decodes_every_vop3p_opcode_and_every_modifier_bit_pattern() {
    const OPCODES: [(u32, &str); 8] = [
        (0x0e, "VPkFmaF16"),
        (0x0f, "VPkAddF16"),
        (0x10, "VPkMulF16"),
        (0x11, "VPkMinF16"),
        (0x12, "VPkMaxF16"),
        (0x20, "VFmaMixF32"),
        (0x21, "VFmaMixloF16"),
        (0x22, "VFmaMixhiF16"),
    ];
    let sources = 257u32 | (258u32 << 9) | (259u32 << 18);

    for (opcode, name) in OPCODES {
        let word = 0xcc00_0005 | (opcode << 16);
        let frame = decode_frame(0, &[word, sources]).expect("VOP3P frame");
        assert_eq!(frame.encoding, ShaderEncoding::Vop3p);
        assert!(frame.opcode.is(name));
        let instruction = create_instruction(0, frame.encoding, frame.opcode, vec![word, sources]);
        assert_eq!(
            instruction.sources,
            vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)]
        );
        assert_eq!(instruction.destinations, vec![Operand::vector(5)]);
    }

    // Five independent fields occupy thirteen encoding bits. Check all 8192 combinations so
    // a split OP_SEL_HI bit or swapped NEG lane cannot regress unnoticed.
    for bits in 0u32..(1 << 13) {
        let operand_select_mask = bits & 7;
        let operand_select_high_mask = (bits >> 3) & 7;
        let negate_low_mask = (bits >> 6) & 7;
        let negate_high_mask = (bits >> 9) & 7;
        let clamp = bits >> 12 != 0;
        let word = 0xcc0e_0005
            | (operand_select_mask << 11)
            | ((operand_select_high_mask >> 2) << 14)
            | (negate_high_mask << 8)
            | (u32::from(clamp) << 15);
        let extra = sources | ((operand_select_high_mask & 3) << 27) | (negate_low_mask << 29);
        let instruction = create_instruction(
            0,
            ShaderEncoding::Vop3p,
            Opcode::Known("VPkFmaF16"),
            vec![word, extra],
        );
        assert_eq!(
            instruction.control,
            Some(InstructionControl::Vop3p(Vop3pControl {
                operand_select_mask,
                operand_select_high_mask,
                negate_low_mask,
                negate_high_mask,
                clamp,
            })),
            "incorrect VOP3P modifiers for packed bits 0x{bits:04X}"
        );
    }
}

#[test]
fn decodes_every_supported_ds_opcode_and_every_control_bit_pattern() {
    const OPCODES: [(u32, &str, usize, usize); 39] = [
        (0x00, "DsAddU32", 2, 0),
        (0x01, "DsSubU32", 2, 0),
        (0x03, "DsIncU32", 2, 0),
        (0x04, "DsDecU32", 2, 0),
        (0x05, "DsMinI32", 2, 0),
        (0x06, "DsMaxI32", 2, 0),
        (0x07, "DsMinU32", 2, 0),
        (0x08, "DsMaxU32", 2, 0),
        (0x09, "DsAndB32", 2, 0),
        (0x0a, "DsOrB32", 2, 0),
        (0x0b, "DsXorB32", 2, 0),
        (0x0d, "DsWriteB32", 2, 0),
        (0x0e, "DsWrite2B32", 3, 0),
        (0x0f, "DsWrite2St64B32", 3, 0),
        (0x10, "DsCmpstB32", 3, 0),
        (0x20, "DsAddRtnU32", 2, 1),
        (0x21, "DsSubRtnU32", 2, 1),
        (0x23, "DsIncRtnU32", 2, 1),
        (0x24, "DsDecRtnU32", 2, 1),
        (0x25, "DsMinRtnI32", 2, 1),
        (0x26, "DsMaxRtnI32", 2, 1),
        (0x27, "DsMinRtnU32", 2, 1),
        (0x28, "DsMaxRtnU32", 2, 1),
        (0x29, "DsAndRtnB32", 2, 1),
        (0x2a, "DsOrRtnB32", 2, 1),
        (0x2b, "DsXorRtnB32", 2, 1),
        (0x2d, "DsWrxchgRtnB32", 2, 1),
        (0x30, "DsCmpstRtnB32", 3, 1),
        (0x35, "DsSwizzleB32", 1, 1),
        (0x36, "DsReadB32", 1, 1),
        (0x37, "DsRead2B32", 1, 2),
        (0x38, "DsRead2St64B32", 1, 2),
        (0x4d, "DsWriteB64", 3, 0),
        (0xb0, "DsWriteAddtidB32", 1, 0),
        (0xb1, "DsReadAddtidB32", 0, 1),
        (0xde, "DsWriteB96", 4, 0),
        (0xdf, "DsWriteB128", 5, 0),
        (0xfe, "DsReadB96", 1, 3),
        (0xff, "DsReadB128", 1, 4),
    ];
    let extra = 1u32 | (2u32 << 8) | (3u32 << 16) | (4u32 << 24);
    for (opcode, name, source_count, destination_count) in OPCODES {
        let word = 0xd800_0000 | (opcode << 18);
        let frame = decode_frame(0, &[word, extra]).expect("DS frame");
        assert_eq!(frame.encoding, ShaderEncoding::Ds);
        assert!(frame.opcode.is(name));
        let instruction = create_instruction(0, frame.encoding, frame.opcode, vec![word, extra]);
        assert_eq!(instruction.sources.len(), source_count, "{name} sources");
        assert_eq!(
            instruction.destinations.len(),
            destination_count,
            "{name} destinations"
        );
    }

    // OFFSET0, OFFSET1, and GDS are independent fields. Cover all 131072 combinations to
    // prevent a one-bit opcode shift or a truncated concatenated byte offset from regressing.
    for bits in 0u32..(1 << 17) {
        let word = 0xd800_0000 | (bits & 0xffff) | ((bits >> 16) << 17);
        let instruction = create_instruction(
            0,
            ShaderEncoding::Ds,
            Opcode::Known("DsReadB32"),
            vec![word, extra],
        );
        assert_eq!(
            instruction.control,
            Some(InstructionControl::DataShare(DataShareControl {
                offset0: bits & 0xff,
                offset1: (bits >> 8) & 0xff,
                gds: bits >> 16 != 0,
            })),
            "incorrect DS control for packed bits 0x{bits:05X}"
        );
    }
}

#[test]
fn decodes_silent_hill_ds_write_addtid_words_without_an_address_source() {
    let extra = 1u32 | (2u32 << 8) | (3u32 << 16) | (4u32 << 24);
    for (word, expected_offset1) in [
        (0xdac0_0000, 0_u32),
        (0xdac0_0100, 1_u32),
        (0xdac0_0200, 2_u32),
    ] {
        let program = decode_program(0x1000, &[word, extra, 0xbf81_0000])
            .expect("observed Silent Hill DS_WRITE_ADDTID_B32 stream");
        let instruction = &program.instructions[0];

        assert!(instruction.opcode.is("DsWriteAddtidB32"));
        assert_eq!(
            instruction.sources,
            [Operand::vector(2)],
            "ADDR and DATA1 are architecturally ignored"
        );
        assert!(instruction.destinations.is_empty());
        assert_eq!(
            instruction.control,
            Some(InstructionControl::DataShare(DataShareControl {
                offset0: 0,
                offset1: expected_offset1,
                gds: false,
            }))
        );
    }
}

#[test]
fn decodes_poppy_ds_read_b64_into_a_two_register_destination() {
    // Live word from Poppy Playtime's glyph-prep compute shader; rejecting it failed the whole
    // DCB submission, which Unreal treats as fatal.
    let extra = 5u32 | (12u32 << 24);
    let program = decode_program(0x1000, &[0xd9d8_0000, extra, 0xbf81_0000])
        .expect("observed Poppy Playtime DS_READ_B64 stream");
    let instruction = &program.instructions[0];

    assert!(instruction.opcode.is("DsReadB64"));
    assert_eq!(instruction.sources, [Operand::vector(5)]);
    assert_eq!(
        instruction.destinations,
        [Operand::vector(12), Operand::vector(13)]
    );
    assert_eq!(
        instruction.control,
        Some(InstructionControl::DataShare(DataShareControl {
            offset0: 0,
            offset1: 0,
            gds: false,
        }))
    );
}

#[test]
fn decodes_complete_mubuf_and_mtbuf_surfaces_with_all_control_fields() {
    const MUBUF: &[(u32, &str)] = &[
        (0x00, "BufferLoadFormatX"),
        (0x01, "BufferLoadFormatXy"),
        (0x02, "BufferLoadFormatXyz"),
        (0x03, "BufferLoadFormatXyzw"),
        (0x04, "BufferStoreFormatX"),
        (0x05, "BufferStoreFormatXy"),
        (0x06, "BufferStoreFormatXyz"),
        (0x07, "BufferStoreFormatXyzw"),
        (0x08, "BufferLoadUbyte"),
        (0x09, "BufferLoadSbyte"),
        (0x0a, "BufferLoadUshort"),
        (0x0b, "BufferLoadSshort"),
        (0x0c, "BufferLoadDword"),
        (0x0d, "BufferLoadDwordx2"),
        (0x0e, "BufferLoadDwordx4"),
        (0x0f, "BufferLoadDwordx3"),
        (0x18, "BufferStoreByte"),
        (0x19, "BufferStoreByteD16Hi"),
        (0x1a, "BufferStoreShort"),
        (0x1b, "BufferStoreShortD16Hi"),
        (0x1c, "BufferStoreDword"),
        (0x1d, "BufferStoreDwordx2"),
        (0x1e, "BufferStoreDwordx4"),
        (0x1f, "BufferStoreDwordx3"),
        (0x20, "BufferLoadUbyteD16"),
        (0x21, "BufferLoadUbyteD16Hi"),
        (0x22, "BufferLoadSbyteD16"),
        (0x23, "BufferLoadSbyteD16Hi"),
        (0x24, "BufferLoadShortD16"),
        (0x25, "BufferLoadShortD16Hi"),
        (0x26, "BufferLoadFormatD16HiX"),
        (0x27, "BufferStoreFormatD16HiX"),
        (0x30, "BufferAtomicSwap"),
        (0x31, "BufferAtomicCmpswap"),
        (0x32, "BufferAtomicAdd"),
        (0x33, "BufferAtomicSub"),
        (0x35, "BufferAtomicSmin"),
        (0x36, "BufferAtomicUmin"),
        (0x37, "BufferAtomicSmax"),
        (0x38, "BufferAtomicUmax"),
        (0x39, "BufferAtomicAnd"),
        (0x3a, "BufferAtomicOr"),
        (0x3b, "BufferAtomicXor"),
        (0x3c, "BufferAtomicInc"),
        (0x3d, "BufferAtomicDec"),
        (0x80, "BufferLoadFormatD16X"),
        (0x81, "BufferLoadFormatD16Xy"),
        (0x82, "BufferLoadFormatD16Xyz"),
        (0x83, "BufferLoadFormatD16Xyzw"),
        (0x84, "BufferStoreFormatD16X"),
        (0x85, "BufferStoreFormatD16Xy"),
        (0x86, "BufferStoreFormatD16Xyz"),
        (0x87, "BufferStoreFormatD16Xyzw"),
    ];
    const MTBUF: [&str; 16] = [
        "TBufferLoadFormatX",
        "TBufferLoadFormatXy",
        "TBufferLoadFormatXyz",
        "TBufferLoadFormatXyzw",
        "TBufferStoreFormatX",
        "TBufferStoreFormatXy",
        "TBufferStoreFormatXyz",
        "TBufferStoreFormatXyzw",
        "TBufferLoadFormatD16X",
        "TBufferLoadFormatD16Xy",
        "TBufferLoadFormatD16Xyz",
        "TBufferLoadFormatD16Xyzw",
        "TBufferStoreFormatD16X",
        "TBufferStoreFormatD16Xy",
        "TBufferStoreFormatD16Xyz",
        "TBufferStoreFormatD16Xyzw",
    ];
    let extra = 1u32 | (16u32 << 8) | (1u32 << 22) | (1u32 << 23) | (128u32 << 24);
    for &(opcode, name) in MUBUF {
        let word = 0xe000_0000 | (opcode << 18) | 0x0000_fabc;
        let frame = decode_frame(0, &[word, extra]).expect("MUBUF frame");
        assert_eq!(frame.encoding, ShaderEncoding::Mubuf);
        assert!(frame.opcode.is(name), "opcode 0x{opcode:02X}");
    }
    for (opcode, name) in MTBUF.into_iter().enumerate() {
        let opcode = u32::try_from(opcode).expect("small MTBUF opcode");
        let word = 0xe800_0000 | ((opcode & 7) << 16) | (60 << 19) | 0x0000_cabc;
        let extra = extra | ((opcode >> 3) << 21);
        let frame = decode_frame(0, &[word, extra]).expect("MTBUF frame");
        assert_eq!(frame.encoding, ShaderEncoding::Mtbuf);
        assert!(frame.opcode.is(name), "opcode 0x{opcode:02X}");
        let instruction = create_instruction(0, frame.encoding, frame.opcode, vec![word, extra]);
        let Some(InstructionControl::BufferMemory(control)) = instruction.control else {
            panic!("expected MTBUF control");
        };
        assert_eq!(control.instruction_format, Some(60));
        assert!(!control.lds, "MTBUF OP bits must not alias MUBUF LDS");
        assert!(control.dlc && control.glc && control.slc && control.tfe);
    }
}

#[test]
fn decodes_global_vdst_vdata_address_modes_and_twelve_bit_offset() {
    let word = 0xdc00_0000 | (0x32 << 18) | (2 << 14) | (1 << 12) | (1 << 16) | (1 << 17) | 0x800;
    let extra = 1 | (16 << 8) | (0x7f << 16) | (32 << 24);
    let frame = decode_frame(0, &[word, extra]).expect("GLOBAL frame");
    let instruction = create_instruction(0, frame.encoding, frame.opcode, vec![word, extra]);
    assert_eq!(instruction.sources[0], Operand::vector(1));
    assert_eq!(instruction.sources[1], Operand::vector(2));
    assert_eq!(instruction.sources[2], Operand::vector(16));
    assert_eq!(instruction.destinations, [Operand::vector(32)]);
    assert_eq!(
        instruction.control,
        Some(InstructionControl::GlobalMemory(GlobalMemoryControl {
            dword_count: 1,
            vector_address: 1,
            vector_data: 16,
            vector_destination: 32,
            scalar_address: None,
            offset_bytes: -2048,
            dlc: true,
            lds: false,
            glc: true,
            slc: true,
        }))
    );
}

#[test]
fn rejects_reserved_scalar_opcode_instead_of_emitting_a_placeholder() {
    let error = decode_program(0x1000, &[0xbe80_0000]).expect_err("reserved SOP1 opcode");
    assert!(matches!(
        error,
        DecodeError::UnknownOpcode {
            encoding: ShaderEncoding::Sop1,
            ..
        }
    ));
}

#[test]
fn decodes_s_bcnt1_i32_b64_from_the_silent_hill_word() {
    let program =
        decode_program(0x1000, &[0xbe8a_1000, 0xbf81_0000]).expect("well-formed SOP1 program");
    let instruction = &program.instructions[0];

    assert!(instruction.opcode.is("SBcnt1I32B64"));
    assert_eq!(instruction.sources, [Operand::scalar(0)]);
    assert_eq!(instruction.destinations, [Operand::scalar(10)]);
}

#[test]
fn decodes_scalar_b64_mask_and_low_half_pack_sop2_words() {
    let program = decode_program(
        0x1000,
        &[
            0x9294_0c0a, // s_bfm_b64 s[20:21], s10, s12
            0x9914_0c0a, // s_pack_ll_b32_b16 s20, s10, s12
            0xbf81_0000,
        ],
    )
    .expect("well-formed scalar SOP2 program");

    for (instruction, name) in [
        (&program.instructions[0], "SBfmB64"),
        (&program.instructions[1], "SPackLlB32B16"),
    ] {
        assert_eq!(instruction.encoding, ShaderEncoding::Sop2);
        assert!(instruction.opcode.is(name));
        assert_eq!(
            instruction.sources,
            [Operand::scalar(10), Operand::scalar(12)]
        );
        assert_eq!(instruction.destinations, [Operand::scalar(20)]);
    }
}

#[test]
fn records_all_mimg_nsa_address_registers() {
    // IMAGE_LOAD with NSA=2 carries eight additional byte-packed VGPR indices.
    let word = 0xf000_0004;
    let program = decode_program(0x1000, &[word, 0, 0x0403_0201, 0x0807_0605, 0xbf81_0000])
        .expect("MIMG NSA program");
    let InstructionControl::Image(image) = program.instructions[0]
        .control
        .as_ref()
        .expect("image control")
    else {
        panic!("expected image control");
    };
    assert_eq!(image.address_registers, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(program.instructions[1].pc, 16);
}

#[test]
fn image_d16_liveness_uses_one_vgpr_for_each_packed_component_pair() {
    let extra = 1 | (20 << 8) | (1 << 31);
    for (opcode, name) in [(0_u32, "ImageLoad"), (8, "ImageStore")] {
        let word = 0xf000_0000 | (opcode << 18) | (0xf << 8);
        let instruction = create_instruction(
            0,
            ShaderEncoding::Mimg,
            Opcode::Known(name),
            vec![word, extra],
        );
        let Some(InstructionControl::Image(control)) = instruction.control else {
            panic!("expected image control");
        };
        assert!(control.d16);
        if name == "ImageLoad" {
            assert_eq!(
                instruction.destinations,
                [Operand::vector(20), Operand::vector(21)]
            );
        } else {
            assert_eq!(
                &instruction.sources[instruction.sources.len() - 2..],
                [Operand::vector(20), Operand::vector(21)]
            );
            assert!(!instruction.sources.contains(&Operand::vector(22)));
        }
    }
}

#[test]
fn retains_high_mimg_opcodes_without_aliasing_low_operations() {
    for (opcode, nsa_dwords) in [(0x80_u32, 0_u32), (0xa2, 1), (0xe6, 3)] {
        let word = 0xf000_0000 | ((opcode & 0x7f) << 18) | ((opcode >> 7) & 1) | (nsa_dwords << 1);
        let mut words = vec![word, 0];
        words.resize(2 + usize::try_from(nsa_dwords).expect("small NSA count"), 0);
        words.push(0xbf81_0000);

        let program = decode_program(0x1000, &words).expect("framed high MIMG opcode");
        assert_eq!(
            program.instructions[0].opcode,
            Opcode::Raw {
                family: RawOpcodeFamily::Mimg,
                code: u16::try_from(opcode).expect("eight-bit MIMG opcode fits u16"),
            }
        );
        assert_eq!(
            program.instructions[0].words.len(),
            2 + usize::try_from(nsa_dwords).expect("small NSA count")
        );
        assert_eq!(program.instructions[1].pc, (2 + nsa_dwords) * 4);
        assert!(
            !program.instructions[0].opcode.is("ImageLoad")
                && !program.instructions[0].opcode.is("ImageSampleD"),
            "high opcode 0x{opcode:02X} aliased a low operation"
        );
    }
}

#[test]
fn decodes_absdiff_at_the_rdna2_sop2_opcode() {
    let word = 0x8000_0000 | (44 << 23);
    let frame = decode_frame(0, &[word]).expect("S_ABSDIFF_I32 opcode 44");
    assert_eq!(frame.encoding, ShaderEncoding::Sop2);
    assert!(frame.opcode.is("SAbsdiffI32"));

    let reserved = 0x8000_0000 | (45 << 23);
    assert!(matches!(
        decode_frame(0, &[reserved]),
        Err(DecodeError::UnknownOpcode {
            encoding: ShaderEncoding::Sop2,
            opcode: 45,
            ..
        })
    ));
}

#[test]
fn memory_reader_stops_exactly_at_endpgm() {
    let memory = SliceMemory {
        base: 0x8000,
        words: vec![0xbf81_0000],
    };
    let program = read_program(&memory, memory.base).expect("single-word program");
    assert_eq!(program.instructions.len(), 1);
    assert_eq!(program.instructions[0].opcode, Opcode::END_PROGRAM);
}

#[test]
fn rdna2_code_end_terminates_padding_and_half_float_vop2_opcodes_decode() {
    let program = decode_program(
        0x8000,
        &[
            0x6406_0601, // v_add_f16 v3, v1, v3
            0x6600_00f9, // v_sub_f16 v0, src_sdwa, v0
            0,
            0x6812_0202, // v_subrev_f16 v9, v2, v1
            0x6a12_0202, // v_mul_f16 v9, v2, v1
            0x7404_04f9, // v_min_f16 v2, src_sdwa, v2
            0,
            0xd74b_600a, // v_fma_f16 v10, v240, v12, v10 op_sel:[0,0,1,1]
            0x042a_18f0,
            0xbf13_1e6a, // s_cmp_lg_u64 s106:s107, s30:s31
            0x7e12_acf9, // v_rsq_f16 v9, src_dpp
            0,
            0x7d98_04f9, // v_cmp_gt_f16 src_dpp, v2
            0,
            0x7dfa_04f9, // v_cmpx_neq_f16 src_dpp, v2
            0,
            0x7dfc_62f9, // v_cmpx_nlt_f16 src_dpp, v49
            0,
            0xbf9f_0000, // s_code_end
            0,
        ],
    )
    .expect("observed RDNA2 shader subset");

    assert!(program.instructions[0].opcode.is("VAddF16"));
    assert!(program.instructions[1].opcode.is("VSubF16"));
    assert!(program.instructions[2].opcode.is("VSubrevF16"));
    assert!(program.instructions[3].opcode.is("VMulF16"));
    assert!(program.instructions[4].opcode.is("VMinF16"));
    assert!(program.instructions[5].opcode.is("VFmaF16"));
    assert!(program.instructions[6].opcode.is("SCmpLgU64"));
    assert!(program.instructions[7].opcode.is("VRsqF16"));
    assert!(program.instructions[8].opcode.is("VCmpGtF16"));
    assert!(program.instructions[9].opcode.is("VCmpxNeqF16"));
    assert!(program.instructions[10].opcode.is("VCmpxNltF16"));
    assert_eq!(program.instructions[11].opcode, Opcode::END_PROGRAM);
    assert_eq!(program.instructions.len(), 12);
}

#[test]
fn decodes_silent_hill_lane_permute_bitcount_and_u64_mad_shapes() {
    let v_bcnt = (0x22_u32 << 25) | (4_u32 << 17) | (3_u32 << 9) | (0x100_u32 + 2);
    let v_permlane = (0x35_u32 << 26) | (0x377_u32 << 16) | (2_u32 << 11) | 8;
    let v_permlanex = (0x35_u32 << 26) | (0x378_u32 << 16) | (1_u32 << 11) | 9;
    let selectors = (0x100_u32 + 5) | (20_u32 << 9) | (21_u32 << 18);
    let v_mad_u64 = (0x35_u32 << 26) | (0x176_u32 << 16) | (30_u32 << 8) | 40;
    let mad_sources = (0x100_u32 + 6) | ((0x100_u32 + 7) << 9) | ((0x100_u32 + 50) << 18);
    let program = decode_program(
        0xa000,
        &[
            v_bcnt,
            v_permlane,
            selectors,
            v_permlanex,
            selectors,
            v_mad_u64,
            mad_sources,
            0xbf81_0000,
        ],
    )
    .expect("observed Silent Hill compute opcode subset");

    let bcnt = &program.instructions[0];
    assert!(bcnt.opcode.is("VBcntU32B32"));
    assert_eq!(bcnt.sources, [Operand::vector(2), Operand::vector(3)]);
    assert_eq!(bcnt.destinations, [Operand::vector(4)]);

    for (index, opcode, operand_select, destination) in
        [(1, "VPermlane16B32", 2, 8), (2, "VPermlanex16B32", 1, 9)]
    {
        let instruction = &program.instructions[index];
        assert!(instruction.opcode.is(opcode));
        assert_eq!(
            instruction.sources,
            [Operand::vector(5), Operand::scalar(20), Operand::scalar(21)]
        );
        assert_eq!(instruction.destinations, [Operand::vector(destination)]);
        let Some(InstructionControl::Vop3(control)) = instruction.control else {
            panic!("lane permutation must retain VOP3 control");
        };
        assert_eq!(control.operand_select, operand_select);
    }

    let mad = &program.instructions[3];
    assert!(mad.opcode.is("VMadU64U32"));
    assert_eq!(
        mad.sources,
        [Operand::vector(6), Operand::vector(7), Operand::vector(50)]
    );
    assert_eq!(mad.destinations, [Operand::vector(40), Operand::vector(41)]);
    let Some(InstructionControl::Vop3(control)) = mad.control else {
        panic!("64-bit MAD must retain VOP3B control");
    };
    assert_eq!(control.scalar_destination, Some(30));
}

#[test]
fn decodes_clean_room_ps5_extension_regression_envelopes() {
    // These are synthetic raw envelopes from the clean-room Ps5IsaExtensions manifest, not
    // retained guest captures. They are still valuable regressions for the PS5 opcode holes:
    // the two constant MAD forms must consume their unconditional literal or every following
    // instruction in the shader is decoded at the wrong address.
    let vop2 = decode_program(
        0x9000,
        &[
            0x3e00_0501, // v_mac_f32 v0, v1, v2
            0x4000_0501, // v_madmk_f32 v0, v1, 1.0, v2
            0x3f80_0000,
            0x4200_0501, // v_madak_f32 v0, v1, v2, 1.0
            0x3f80_0000,
            0xbf81_0000,
        ],
    )
    .expect("synthetic PS5 VOP2 extension envelopes");
    assert!(vop2.instructions[0].opcode.is("VMacF32"));
    assert_eq!(vop2.instructions[0].pc, 0);
    assert_eq!(vop2.instructions[0].words.len(), 1);
    assert!(vop2.instructions[1].opcode.is("VMadMkF32"));
    assert_eq!(vop2.instructions[1].pc, 4);
    assert_eq!(vop2.instructions[1].words, [0x4000_0501, 0x3f80_0000]);
    assert_eq!(vop2.instructions[1].sources.len(), 3);
    assert!(vop2.instructions[2].opcode.is("VMadAkF32"));
    assert_eq!(vop2.instructions[2].pc, 12);
    assert_eq!(vop2.instructions[2].words, [0x4200_0501, 0x3f80_0000]);
    assert_eq!(vop2.instructions[2].sources.len(), 3);
    assert_eq!(vop2.instructions[3].pc, 20);
    assert_eq!(vop2.instructions[3].opcode, Opcode::END_PROGRAM);

    let vop3 = decode_program(
        0xa000,
        &[
            0xd541_0000, // PS5 VOP3A opcode 321: v_mad_f32 v0, v1, v2, v3
            0x040e_0501,
            0xbf81_0000,
        ],
    )
    .expect("synthetic PS5 VOP3 extension envelope");
    assert!(vop3.instructions[0].opcode.is("VMadF32"));
    assert_eq!(vop3.instructions[0].words, [0xd541_0000, 0x040e_0501]);
    assert_eq!(vop3.instructions[0].sources.len(), 3);
    assert_eq!(vop3.instructions[1].pc, 8);
}

// ---------------------------------------------------------------------------------------------
// Workstream Q1: FLAT scratch + full GLOBAL/FLAT atomics; benign mode-control scalar opcodes.
// Workstream Q2: the full GFX10 IMAGE_SAMPLE/GATHER4 lattice.
// ---------------------------------------------------------------------------------------------

/// The load/store OP numbers shared by the FLAT, SCRATCH and GLOBAL segments, with the name each
/// segment gives them. `decode_flat` must resolve all three — a single spilled register (SCRATCH)
/// or flat-addressed access previously abandoned the whole shader.
const FLAT_SHARED_LOAD_STORE: &[(u32, &str, &str, &str)] = &[
    (0x08, "FlatLoadUbyte", "ScratchLoadUbyte", "GlobalLoadUbyte"),
    (0x0c, "FlatLoadDword", "ScratchLoadDword", "GlobalLoadDword"),
    (
        0x0d,
        "FlatLoadDwordx2",
        "ScratchLoadDwordx2",
        "GlobalLoadDwordx2",
    ),
    (
        0x0e,
        "FlatLoadDwordx4",
        "ScratchLoadDwordx4",
        "GlobalLoadDwordx4",
    ),
    (
        0x0f,
        "FlatLoadDwordx3",
        "ScratchLoadDwordx3",
        "GlobalLoadDwordx3",
    ),
    (
        0x1c,
        "FlatStoreDword",
        "ScratchStoreDword",
        "GlobalStoreDword",
    ),
    (
        0x1d,
        "FlatStoreDwordx2",
        "ScratchStoreDwordx2",
        "GlobalStoreDwordx2",
    ),
    (
        0x24,
        "FlatLoadShortD16",
        "ScratchLoadShortD16",
        "GlobalLoadShortD16",
    ),
];

fn flat_word(opcode: u32, segment: u32) -> u32 {
    0xdc00_0000 | (opcode << 18) | (segment << 14)
}

#[test]
fn decodes_flat_scratch_and_global_load_store_across_all_three_segments() {
    // word1: VADDR=1, VDATA=16, SADDR=NULL(0x7f), VDST=32.
    let extra = 1 | (16 << 8) | (0x7f << 16) | (32 << 24);
    for &(opcode, flat, scratch, global) in FLAT_SHARED_LOAD_STORE {
        for (segment, expected) in [(0, flat), (1, scratch), (2, global)] {
            let word = flat_word(opcode, segment);
            let frame = decode_frame(0, &[word, extra]).expect("FLAT-family frame");
            assert_eq!(frame.encoding, ShaderEncoding::Flat);
            assert!(
                frame.opcode.is(expected),
                "segment {segment} opcode 0x{opcode:02X} decoded as {:?}",
                frame.opcode
            );
            assert_eq!(frame.dwords, 2);
        }
    }
}

/// The integer atomic block exists for FLAT and GLOBAL, mirroring the MUBUF numbering already in
/// the decoder. Only `GlobalAtomicAdd`/`GlobalAtomicUMax` decoded before, so a `GlobalAtomicSwap`
/// or `GlobalAtomicCmpswap` used to abandon the shader.
const FLAT_ATOMICS: &[(u32, &str, &str)] = &[
    (0x30, "FlatAtomicSwap", "GlobalAtomicSwap"),
    (0x31, "FlatAtomicCmpswap", "GlobalAtomicCmpswap"),
    (0x32, "FlatAtomicAdd", "GlobalAtomicAdd"),
    (0x33, "FlatAtomicSub", "GlobalAtomicSub"),
    (0x35, "FlatAtomicSmin", "GlobalAtomicSmin"),
    (0x36, "FlatAtomicUmin", "GlobalAtomicUmin"),
    (0x37, "FlatAtomicSmax", "GlobalAtomicSmax"),
    (0x38, "FlatAtomicUmax", "GlobalAtomicUmax"),
    (0x39, "FlatAtomicAnd", "GlobalAtomicAnd"),
    (0x3a, "FlatAtomicOr", "GlobalAtomicOr"),
    (0x3b, "FlatAtomicXor", "GlobalAtomicXor"),
    (0x3c, "FlatAtomicInc", "GlobalAtomicInc"),
    (0x3d, "FlatAtomicDec", "GlobalAtomicDec"),
];

#[test]
fn decodes_the_full_flat_and_global_integer_atomic_block() {
    let extra = 1 | (16 << 8) | (0x7f << 16) | (32 << 24);
    for &(opcode, flat, global) in FLAT_ATOMICS {
        for (segment, expected) in [(0, flat), (2, global)] {
            let word = flat_word(opcode, segment);
            let frame = decode_frame(0, &[word, extra]).expect("FLAT/GLOBAL atomic frame");
            assert!(
                frame.opcode.is(expected),
                "segment {segment} atomic 0x{opcode:02X} decoded as {:?}",
                frame.opcode
            );
        }
    }
}

#[test]
fn a_global_atomic_returns_to_its_vdst_only_when_glc_requests_it() {
    let extra = 1 | (16 << 8) | (12 << 16) | (7 << 24);
    // GLC (bit 16) set: GLOBAL_ATOMIC_ADD returns the pre-op value to VDST.
    let word = flat_word(0x32, 2) | (1 << 16);
    let instruction = create_instruction(
        0,
        ShaderEncoding::Flat,
        Opcode::Known("GlobalAtomicAdd"),
        vec![word, extra],
    );
    assert_eq!(instruction.destinations, [Operand::vector(7)]);
    // GLC clear: no return value written back.
    let word = flat_word(0x32, 2);
    let instruction = create_instruction(
        0,
        ShaderEncoding::Flat,
        Opcode::Known("GlobalAtomicAdd"),
        vec![word, extra],
    );
    assert!(instruction.destinations.is_empty());
}

#[test]
fn scratch_segment_has_no_atomic_encodings_and_stays_loud() {
    // SCRATCH (segment 1) shares the load/store numbers but has no atomic block; an atomic OP in
    // that segment is genuinely undefined and must remain a loud UnknownOpcode, not a fabrication.
    let word = flat_word(0x32, 1);
    assert!(matches!(
        decode_frame(0, &[word, 0]),
        Err(DecodeError::UnknownOpcode {
            encoding: ShaderEncoding::Flat,
            ..
        })
    ));
}

#[test]
fn reserved_flat_segment_three_is_still_rejected() {
    let word = flat_word(0x0c, 3);
    assert!(matches!(
        decode_frame(0, &[word, 0]),
        Err(DecodeError::FlatSegment { segment: 3, .. })
    ));
}

#[test]
fn decodes_benign_mode_control_and_wait_sopp_opcodes() {
    // These carry no operands that resource discovery models. Decoding them keeps a shader that
    // merely toggled rounding/denormal mode or slept alive instead of dropping the draw.
    const BENIGN_SOPP: &[(u32, &str)] = &[
        (0x03, "SWakeup"),
        (0x0d, "SSethalt"),
        (0x0e, "SSleep"),
        (0x0f, "SSetprio"),
        (0x11, "SSendmsghalt"),
        (0x13, "SIcacheInv"),
        (0x14, "SIncperflevel"),
        (0x15, "SDecperflevel"),
        (0x22, "SWaitIdle"),
        (0x24, "SRoundMode"),
        (0x25, "SDenormMode"),
    ];
    for &(opcode, name) in BENIGN_SOPP {
        let word = 0xbf80_0000 | (opcode << 16);
        let frame = decode_frame(0, &[word]).expect("benign SOPP frame");
        assert_eq!(frame.encoding, ShaderEncoding::Sopp);
        assert!(
            frame.opcode.is(name),
            "SOPP 0x{opcode:02X} decoded as {:?}",
            frame.opcode
        );
        assert_eq!(frame.dwords, 1);
    }
}

#[test]
fn decodes_hardware_register_sopk_opcodes_including_the_immediate_form() {
    // S_GETREG_B32 / S_SETREG_B32 are single dword; S_SETREG_IMM32_B32 carries a trailing literal
    // and must frame as two dwords or every following instruction decodes at the wrong address.
    for (opcode, name, dwords) in [
        (0x11u32, "SGetregB32", 1usize),
        (0x12, "SSetregB32", 1),
        (0x13, "SSetregImm32B32", 2),
    ] {
        let word = 0xb000_0000 | (opcode << 23);
        let frame = decode_frame(0, &[word, 0xdead_beef]).expect("SOPK hwreg frame");
        assert_eq!(frame.encoding, ShaderEncoding::Sopk);
        assert!(
            frame.opcode.is(name),
            "SOPK 0x{opcode:02X} decoded as {:?}",
            frame.opcode
        );
        assert_eq!(frame.dwords, dwords, "{name} dword span");
        let mut words = vec![word];
        if dwords == 2 {
            words.push(0xdead_beef); // the S_SETREG_IMM32_B32 literal
        }
        words.push(0xbf81_0000); // s_endpgm at pc dwords*4
        let program = decode_program(0x1000, &words).expect("hwreg program terminates cleanly");
        assert!(program.instructions[0].opcode.is(name));
        assert_eq!(program.instructions[1].opcode, Opcode::END_PROGRAM);
        assert_eq!(program.instructions[1].pc as usize, dwords * 4);
    }
}

/// The GFX10 IMAGE_SAMPLE/GATHER4 modifier lattice. Real pixel shaders lean on the comparison
/// (…C…, shadow), clamp, bias and gradient variants; only a sparse subset decoded before, so the
/// rest framed as raw MIMG opcodes and dropped the draw at translation.
#[test]
fn decodes_the_full_gfx10_image_sample_gather_lattice() {
    const SAMPLES: &[(u32, &str)] = &[
        (0x20, "ImageSample"),
        (0x21, "ImageSampleCl"),
        (0x23, "ImageSampleDCl"),
        (0x26, "ImageSampleBCl"),
        (0x28, "ImageSampleC"),
        (0x29, "ImageSampleCCl"),
        (0x2a, "ImageSampleCD"),
        (0x2c, "ImageSampleCL"),
        (0x2d, "ImageSampleCB"),
        (0x2f, "ImageSampleCLz"),
        (0x38, "ImageSampleCO"),
        (0x3f, "ImageSampleCLzO"),
        (0x41, "ImageGather4Cl"),
        (0x48, "ImageGather4C"),
        (0x4f, "ImageGather4CLz"),
        (0x60, "ImageGetLod"),
        (0x68, "ImageSampleCd"),
        (0x6b, "ImageSampleCCdCl"),
    ];
    for &(opcode, name) in SAMPLES {
        assert_eq!(opcode >> 7, 0, "lattice opcode fits OP[6:0]");
        let word = 0xf000_0000 | (opcode << 18);
        let frame = decode_frame(0, &[word, 0]).expect("MIMG sample frame");
        assert_eq!(frame.encoding, ShaderEncoding::Mimg);
        assert!(
            frame.opcode.is(name),
            "MIMG 0x{opcode:02X} decoded as {:?}",
            frame.opcode
        );
    }
}

#[test]
fn decodes_benign_sopp_hint_opcodes_as_known_non_branches() {
    // SOPP hints adjust hardware wait/mode/priority state only. Every one of them previously
    // failed the whole shader with UnknownOpcode; a single S_DENORM_MODE killed the draw.
    let words = [
        0xbf83_0000, // s_wakeup
        0xbf8d_0000, // s_sethalt 0
        0xbf8e_0001, // s_sleep 1
        0xbf8f_0003, // s_setprio 3
        0xbf93_0000, // s_icache_inv
        0xbf94_0000, // s_incperflevel 0
        0xbf95_0000, // s_decperflevel 0
        0xbfa2_0000, // s_wait_idle
        0xbfa4_0003, // s_round_mode 3
        0xbfa5_000f, // s_denorm_mode 15
        0xbfa8_0000, // s_ttracedata_imm 0
        0xbf81_0000, // s_endpgm
    ];
    let program = decode_program(0x1000, &words).expect("benign SOPP hints decode");
    let names = [
        "SWakeup",
        "SSethalt",
        "SSleep",
        "SSetprio",
        "SIcacheInv",
        "SIncperflevel",
        "SDecperflevel",
        "SWaitIdle",
        "SRoundMode",
        "SDenormMode",
        "STtracedataImm",
    ];
    for (instruction, name) in program.instructions.iter().zip(names) {
        assert_eq!(instruction.encoding, ShaderEncoding::Sopp);
        assert!(
            instruction.opcode.is(name),
            "expected {name}, got {}",
            instruction.opcode
        );
    }
    assert_eq!(program.instructions[11].opcode, Opcode::END_PROGRAM);
}

#[test]
fn sopp_end_program_variants_terminate_the_stream() {
    for word in [0xbf9b_0000_u32, 0xbf9e_0000] {
        // s_endpgm_saved / s_endpgm_ordered_ps_done end the wave exactly like s_endpgm. Decoding
        // them as the terminator (the S_CODE_END precedent) stops the stream scan correctly.
        let program = decode_program(0x1000, &[word, 0xdead_beef]).expect("end-program variant");
        assert_eq!(program.instructions.len(), 1);
        assert_eq!(program.instructions[0].opcode, Opcode::END_PROGRAM);
    }
}

#[test]
fn decodes_benign_sopk_hints_and_scalar_waitcnt_shapes() {
    let program = decode_program(
        0x1000,
        &[
            0xb080_1234, // s_version 0x1234
            0xb105_0007, // s_cmovk_i32 s5, 7
            0xb885_f801, // s_getreg_b32 s5, hwreg(...)
            0xb905_f801, // s_setreg_b32 hwreg(...), s5
            0xb985_f801, // s_setreg_imm32_b32 hwreg(...), 0x11223344
            0x1122_3344,
            0xbafd_0000, // s_waitcnt_vscnt null, 0
            0xbb05_0001, // s_waitcnt_vmcnt s5, 1
            0xbbfd_0002, // s_waitcnt_expcnt null, 2
            0xbc05_0003, // s_waitcnt_lgkmcnt s5, 3
            0xbf81_0000, // s_endpgm
        ],
    )
    .expect("benign SOPK program");

    let version = &program.instructions[0];
    assert!(version.opcode.is("SVersion"));
    assert!(version.destinations.is_empty(), "s_version writes no SGPR");

    let cmovk = &program.instructions[1];
    assert!(cmovk.opcode.is("SCmovkI32"));
    assert_eq!(cmovk.destinations, [Operand::scalar(5)]);

    let getreg = &program.instructions[2];
    assert!(getreg.opcode.is("SGetregB32"));
    assert_eq!(getreg.destinations, [Operand::scalar(5)]);

    let setreg = &program.instructions[3];
    assert!(setreg.opcode.is("SSetregB32"));
    assert!(setreg.destinations.is_empty(), "s_setreg writes no SGPR");
    assert_eq!(setreg.sources[1], Operand::scalar(5));

    let setreg_imm = &program.instructions[4];
    assert!(setreg_imm.opcode.is("SSetregImm32B32"));
    assert_eq!(
        setreg_imm.words.len(),
        2,
        "s_setreg_imm32_b32 carries a literal"
    );
    assert!(setreg_imm.destinations.is_empty());
    assert_eq!(
        setreg_imm.sources[1],
        Operand {
            kind: OperandKind::LiteralConstant,
            value: 0x1122_3344,
        }
    );

    for (index, name) in [
        (5, "SWaitcntVscnt"),
        (6, "SWaitcntVmcnt"),
        (7, "SWaitcntExpcnt"),
        (8, "SWaitcntLgkmcnt"),
    ] {
        let waitcnt = &program.instructions[index];
        assert!(
            waitcnt.opcode.is(name),
            "expected {name}, got {}",
            waitcnt.opcode
        );
        assert!(
            waitcnt.destinations.is_empty(),
            "{name} reads its SDST field"
        );
    }
    // The register-relative wait count is a read of the SDST field, not a write.
    assert_eq!(program.instructions[6].sources[1], Operand::scalar(5));
}

#[test]
fn decodes_the_sop1_unary_and_relative_move_batch() {
    // SOP1 word: 0xbe80_0000 | sdst<<16 | code<<8 | ssrc.
    let cases: &[(u32, &str)] = &[
        (0xbe85_0503, "SCmovB32"),            // s_cmov_b32 s5, s3
        (0xbe84_0602, "SCmovB64"),            // s_cmov_b64 s[4:5], s[2:3]
        (0xbe85_0903, "SWqmB32"),             // s_wqm_b32 s5, s3
        (0xbe84_0c02, "SBrevB64"),            // s_brev_b64 s[4:5], s[2:3]
        (0xbe85_0d03, "SBcnt0I32B32"),        // s_bcnt0_i32_b32 s5, s3
        (0xbe85_0e02, "SBcnt0I32B64"),        // s_bcnt0_i32_b64 s5, s[2:3]
        (0xbe85_1103, "SFF0I32B32"),          // s_ff0_i32_b32 s5, s3
        (0xbe85_1202, "SFF0I32B64"),          // s_ff0_i32_b64 s5, s[2:3]
        (0xbe85_1402, "SFF1I32B64"),          // s_ff1_i32_b64 s5, s[2:3]
        (0xbe85_1503, "SFlbitI32B32"),        // s_flbit_i32_b32 s5, s3
        (0xbe85_1602, "SFlbitI32B64"),        // s_flbit_i32_b64 s5, s[2:3]
        (0xbe85_1703, "SFlbitI32"),           // s_flbit_i32 s5, s3
        (0xbe85_1802, "SFlbitI32I64"),        // s_flbit_i32_i64 s5, s[2:3]
        (0xbe85_1903, "SSextI32I8"),          // s_sext_i32_i8 s5, s3
        (0xbe85_1a03, "SSextI32I16"),         // s_sext_i32_i16 s5, s3
        (0xbe85_1b03, "SBitset0B32"),         // s_bitset0_b32 s5, s3
        (0xbe84_1c03, "SBitset0B64"),         // s_bitset0_b64 s[4:5], s3
        (0xbe84_1e03, "SBitset1B64"),         // s_bitset1_b64 s[4:5], s3
        (0xbe85_2c03, "SQuadmaskB32"),        // s_quadmask_b32 s5, s3
        (0xbe84_2d02, "SQuadmaskB64"),        // s_quadmask_b64 s[4:5], s[2:3]
        (0xbe85_2e03, "SMovrelsB32"),         // s_movrels_b32 s5, s3
        (0xbe84_2f02, "SMovrelsB64"),         // s_movrels_b64 s[4:5], s[2:3]
        (0xbe85_3003, "SMovreldB32"),         // s_movreld_b32 s5, s3
        (0xbe84_3102, "SMovreldB64"),         // s_movreld_b64 s[4:5], s[2:3]
        (0xbe85_3403, "SAbsI32"),             // s_abs_i32 s5, s3
        (0xbe84_3603, "SBitreplicateB64B32"), // s_bitreplicate_b64_b32 s[4:5], s3
    ];
    for &(word, name) in cases {
        let program =
            decode_program(0x1000, &[word, 0xbf81_0000]).expect("SOP1 batch instruction decodes");
        let instruction = &program.instructions[0];
        assert_eq!(instruction.encoding, ShaderEncoding::Sop1);
        assert!(
            instruction.opcode.is(name),
            "word 0x{word:08X}: expected {name}, got {}",
            instruction.opcode
        );
    }
}

#[test]
fn decodes_s_call_and_scans_past_endpgm_to_its_forward_subroutine() {
    // s_call_b64 s[10:11], +2 jumps over the endpgm to the subroutine; the decoder must keep
    // scanning to the call target exactly as it does for forward branches.
    let words = [
        0xba0a_0001, // s_call_b64 s[10:11], +1 (target pc 8)
        0xbf81_0000, // s_endpgm at pc 4
        0xb005_002a, // s_movk_i32 s5, 42 at pc 8
        0xbf81_0000, // s_endpgm
    ];
    let program = decode_program(0x1000, &words).expect("call target extends the scan");
    assert_eq!(program.instructions.len(), 4);
    assert!(program.instructions[0].opcode.is("SCallB64"));
    assert_eq!(program.instructions[0].destinations, [Operand::scalar(10)]);
    assert!(program.instructions[2].opcode.is("SMovkI32"));
}
