// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    AGC_SHADER_INPUT_SEMANTICS_COUNT_OFFSET, AGC_SHADER_INPUT_SEMANTICS_OFFSET,
    AGC_SHADER_OUTPUT_SEMANTICS_COUNT_OFFSET, AGC_SHADER_OUTPUT_SEMANTICS_OFFSET,
    AGC_SPI_PS_INPUT_CNTL_0, ProbeDispatchState, SHADER_BINARY_TYPE_GS_BACK,
    SHADER_BINARY_TYPE_GS_FRONT, SHADER_HEADER_CODE_OFFSET, SHADER_HEADER_NUM_SH_REGISTERS_OFFSET,
    SHADER_HEADER_SH_REGISTERS_OFFSET, SHADER_HEADER_SIZE, SHADER_HEADER_TYPE_OFFSET,
    SPI_SHADER_PGM_LO_ES, agc_create_interpolant_mapping, agc_fuse_shader_halves,
    agc_get_fused_shader_size, low_u32, read_guest_u64, write_guest_u8, write_guest_u16,
    write_guest_u32, write_guest_u64,
};

/// Engines address content relative to their binaries directory, so parent components inside
/// the game root must resolve rather than be refused.
/// Dumps are frequently stored with normalized names while titles request the original
/// casing, and console lookups do not distinguish case.
/// A merged shader's fused stage keeps the back half's register block, so the reported
/// scratch size must cover exactly those registers; mismatched halves are refused.
#[test]
fn fused_shader_size_covers_the_back_half_registers() {
    let mut front = [0_u8; 0x60];
    let mut back = [0_u8; 0x60];
    front[0x5a] = 4; // GsFront
    back[0x5a] = 6; // GsBack
    back[0x5c] = 7; // seven shader registers
    let mut output = [0_u64; 2];
    assert_eq!(
        agc_get_fused_shader_size([
            output.as_mut_ptr().addr() as u64,
            front.as_ptr().addr() as u64,
            back.as_ptr().addr() as u64,
            0,
            0,
            0,
        ]),
        0
    );
    assert_eq!(output, [7 * 8, 4]);

    // A tessellation back half does not fuse with a geometry front half.
    back[0x5a] = 7; // HsBack
    assert_ne!(
        agc_get_fused_shader_size([
            output.as_mut_ptr().addr() as u64,
            front.as_ptr().addr() as u64,
            back.as_ptr().addr() as u64,
            0,
            0,
            0,
        ]),
        0
    );
}

/// Octopath calls the shader import three times at boot with arguments that cannot describe a
/// shader. Refusing them must not dereference the pointer, and the refusal must stay distinct
/// from a header that is readable but carries the wrong magic.
#[test]
fn create_shader_refuses_null_and_unmapped_headers_without_faulting() {
    const UNMAPPED: u64 = 0x0000_7f00_0000_0000;
    let mut destination = 0_u64;
    let destination_address = (&raw mut destination).addr() as u64;
    let code = [0xbf81_0000_u32; 4];
    let code_address = code.as_ptr().addr() as u64;
    let mut state = ProbeDispatchState::default();

    for arguments in [
        [destination_address, 0, code_address, 0, 0, 0],
        [destination_address, UNMAPPED, 0, 0, 0, 0],
        [destination_address, UNMAPPED, code_address, 0, 0, 0],
    ] {
        assert_eq!(
            crate::agc_create_shader(arguments, &mut state),
            u64::MAX,
            "header={:#x} code={:#x} must be refused",
            arguments[1],
            arguments[2]
        );
    }
    assert_eq!(destination, 0);
}

#[test]
fn create_shader_registers_the_authoritative_agc_code_extent() {
    let mut header = [0_u8; 0x60];
    let mut shader_registers = [0x20c_u32, 0, 0x20d, 0];
    let code = [0xbf81_0000_u32; 0x1001];
    let mut destination = 0_u64;
    let header_address = header.as_mut_ptr().addr() as u64;
    let registers_address = shader_registers.as_mut_ptr().addr() as u64;
    let code_address = code.as_ptr().addr() as u64;
    write_guest_u32(header_address, crate::AGC_SHADER_FILE_HEADER);
    write_guest_u32(header_address + 4, crate::AGC_SHADER_VERSION);
    write_guest_u64(
        header_address + crate::AGC_SHADER_SH_REGISTERS_OFFSET,
        registers_address.wrapping_sub(header_address + crate::AGC_SHADER_SH_REGISTERS_OFFSET),
    );
    write_guest_u32(
        header_address + crate::AGC_SHADER_SIZE_OFFSET,
        u32::try_from(std::mem::size_of_val(&code)).expect("test shader size fits u32"),
    );
    write_guest_u8(header_address + crate::AGC_SHADER_TYPE_OFFSET, 0);
    write_guest_u8(
        header_address + crate::AGC_SHADER_SH_REGISTER_COUNT_OFFSET,
        2,
    );
    let mut state = ProbeDispatchState::default();
    let other_front = code_address.wrapping_add(0x1000);
    state
        .agc_shader_registry
        .fused_continuations
        .entry(code_address)
        .or_default()
        .insert(crate::AgcFusedShaderContinuation {
            front_header_address: header_address,
            back_header_address: 0x1111,
            back_program_address: 0x2222,
            back_code_size: 4,
        });
    state
        .agc_shader_registry
        .fused_continuations
        .entry(other_front)
        .or_default()
        .insert(crate::AgcFusedShaderContinuation {
            front_header_address: 0x3333,
            back_header_address: header_address,
            back_program_address: code_address,
            back_code_size: 4,
        });
    state
        .agc_shader_registry
        .decode_failures
        .insert(code_address);
    state
        .agc_shader_registry
        .decode_failures
        .insert(other_front);

    assert_eq!(
        crate::agc_create_shader(
            [
                (&raw mut destination).addr() as u64,
                header_address,
                code_address,
                0,
                0,
                0,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(destination, header_address);
    assert_eq!(
        state.agc_shader_registry.code_sizes.get(&code_address),
        Some(&u32::try_from(std::mem::size_of_val(&code)).expect("test shader size fits u32"))
    );
    assert_eq!(shader_registers[1], low_u32(code_address >> 8));
    assert_eq!(shader_registers[3], low_u32(code_address >> 40));
    assert!(state.agc_shader_registry.fused_continuations.is_empty());
    assert!(
        !state
            .agc_shader_registry
            .decode_failures
            .contains(&code_address)
    );
    assert!(
        !state
            .agc_shader_registry
            .decode_failures
            .contains(&other_front)
    );
}

#[test]
fn interpolant_mapping_matches_semantics_and_keeps_identity_tail() {
    let mut geometry = [0_u8; 0x60];
    let mut pixel = [0_u8; 0x60];
    let geometry_semantics = [7 | (5 << 8)];
    let pixel_semantics = [7 | (1 << 22) | (2 << 28)];
    let mut registers = [0_u32; 64];
    let geometry_address = geometry.as_mut_ptr().addr() as u64;
    let pixel_address = pixel.as_mut_ptr().addr() as u64;
    write_guest_u64(
        geometry_address + AGC_SHADER_OUTPUT_SEMANTICS_OFFSET,
        geometry_semantics.as_ptr().addr() as u64,
    );
    write_guest_u16(
        geometry_address + AGC_SHADER_OUTPUT_SEMANTICS_COUNT_OFFSET,
        1,
    );
    // This adjacent packed field must not become part of the 16-bit output count.
    write_guest_u16(
        geometry_address + AGC_SHADER_OUTPUT_SEMANTICS_COUNT_OFFSET + 2,
        u16::MAX,
    );
    write_guest_u64(
        pixel_address + AGC_SHADER_INPUT_SEMANTICS_OFFSET,
        pixel_semantics.as_ptr().addr() as u64,
    );
    write_guest_u32(pixel_address + AGC_SHADER_INPUT_SEMANTICS_COUNT_OFFSET, 1);

    assert_eq!(
        agc_create_interpolant_mapping([
            registers.as_mut_ptr().addr() as u64,
            geometry_address,
            pixel_address,
            0,
            0,
            0,
        ]),
        0
    );
    assert_eq!(registers[0], AGC_SPI_PS_INPUT_CNTL_0);
    assert_eq!(registers[1], 0x605);
    assert_eq!(registers[2], AGC_SPI_PS_INPUT_CNTL_0 + 1);
    assert_eq!(registers[3], 1);
    assert_eq!(registers[62], AGC_SPI_PS_INPUT_CNTL_0 + 31);
    assert_eq!(registers[63], 31);

    registers.fill(u32::MAX);
    assert_eq!(
        agc_create_interpolant_mapping([registers.as_mut_ptr().addr() as u64, 0, 0, 0, 0, 0,]),
        0
    );
    assert_eq!(registers[0], AGC_SPI_PS_INPUT_CNTL_0);
    assert_eq!(registers[1], 0);
    assert_eq!(registers[63], 31);
}

#[test]
fn fused_shader_preserves_header_data_and_high_register_flags() {
    let mut front = [0_u8; SHADER_HEADER_SIZE as usize];
    let mut back = [0_u8; SHADER_HEADER_SIZE as usize];
    let mut fused = [0_u8; SHADER_HEADER_SIZE as usize];
    let mut back_registers = [
        SPI_SHADER_PGM_LO_ES,
        0,
        SPI_SHADER_PGM_LO_ES + 1,
        0xa5a5_5a00,
    ];
    let mut scratch = [0_u32; 4];
    let front_address = front.as_mut_ptr().addr() as u64;
    let back_address = back.as_mut_ptr().addr() as u64;
    let code = 0x00ab_cdef_1234_5600_u64;
    let back_code = 0x0012_3456_789a_bc00_u64;
    let user_data = 0x1234_5678_9abc_def0_u64;
    write_guest_u8(
        front_address + SHADER_HEADER_TYPE_OFFSET,
        SHADER_BINARY_TYPE_GS_FRONT,
    );
    write_guest_u8(
        back_address + SHADER_HEADER_TYPE_OFFSET,
        SHADER_BINARY_TYPE_GS_BACK,
    );
    write_guest_u8(back_address + SHADER_HEADER_NUM_SH_REGISTERS_OFFSET, 2);
    write_guest_u64(front_address + SHADER_HEADER_CODE_OFFSET, code);
    write_guest_u64(back_address + SHADER_HEADER_CODE_OFFSET, back_code);
    write_guest_u32(back_address + crate::AGC_SHADER_SIZE_OFFSET, 0x340);
    write_guest_u64(back_address + 8, user_data);
    write_guest_u64(
        back_address + SHADER_HEADER_SH_REGISTERS_OFFSET,
        back_registers.as_mut_ptr().addr() as u64,
    );

    let mut state = crate::ProbeDispatchState::default();
    assert_eq!(
        agc_fuse_shader_halves(
            [
                fused.as_mut_ptr().addr() as u64,
                front_address,
                back_address,
                scratch.as_mut_ptr().addr() as u64,
                0,
                0,
            ],
            &mut state,
        ),
        0
    );
    let fused_address = fused.as_ptr().addr() as u64;
    assert_eq!(read_guest_u64(fused_address + 8), user_data);
    assert_eq!(
        read_guest_u64(fused_address + SHADER_HEADER_SH_REGISTERS_OFFSET),
        scratch.as_ptr().addr() as u64
    );
    assert_eq!(scratch[1], low_u32(code >> 8));
    assert_eq!(scratch[3], 0xa5a5_5a00 | (low_u32(code >> 40) & 0xff));
    assert_eq!(
        state.agc_shader_registry.fused_continuations.get(&code),
        Some(&std::collections::BTreeSet::from([
            crate::AgcFusedShaderContinuation {
                front_header_address: front_address,
                back_header_address: back_address,
                back_program_address: back_code,
                back_code_size: 0x340,
            },
        ]))
    );

    // The ES program register alone cannot distinguish two different continuations for one front
    // allocation. Preserve both fuse-time identities and refuse to guess at snapshot time.
    let mut other_back = back;
    let other_back_address = other_back.as_mut_ptr().addr() as u64;
    let other_back_code = back_code + 0x1000;
    write_guest_u64(
        other_back_address + SHADER_HEADER_CODE_OFFSET,
        other_back_code,
    );
    assert_eq!(
        agc_fuse_shader_halves(
            [
                fused.as_mut_ptr().addr() as u64,
                front_address,
                other_back_address,
                scratch.as_mut_ptr().addr() as u64,
                0,
                0,
            ],
            &mut state,
        ),
        0
    );
    assert_eq!(
        state
            .agc_shader_registry
            .fused_continuations
            .get(&code)
            .map(std::collections::BTreeSet::len),
        Some(2)
    );
    assert!(crate::agc_shader_snapshot(&mut state.agc_shader_registry, code).is_none());
    assert!(state.agc_shader_registry.decode_failures.contains(&code));
}

#[test]
fn fused_shader_snapshot_uses_back_descriptor_and_retains_continuation_program() {
    // The back allocation intentionally precedes the front allocation. Shader heaps recycle
    // independent blocks in either order; their guest addresses do not describe execution order.
    let program_words = [
        0xbf81_0000_u32, // back: s_endpgm
        0,
        0,
        0,
        0xbe80_2006_u32, // front: s_setpc_b64 s[6:7]
    ];
    let mut front_header = [0_u8; SHADER_HEADER_SIZE as usize];
    let mut back_header = [0_u8; SHADER_HEADER_SIZE as usize];
    let mut back_user_data = [0_u8; 0x36];
    let front_program_address = program_words[4..].as_ptr().addr() as u64;
    let back_program_address = program_words.as_ptr().addr() as u64;
    let front_header_address = front_header.as_mut_ptr().addr() as u64;
    let back_header_address = back_header.as_mut_ptr().addr() as u64;
    write_guest_u64(
        back_header_address + 0x08,
        back_user_data.as_mut_ptr().addr() as u64,
    );

    let mut shaders = crate::AgcShaderRegistry::default();
    shaders
        .headers
        .insert(front_program_address, front_header_address);
    shaders
        .headers
        .insert(back_program_address, back_header_address);
    shaders.code_sizes.insert(
        front_program_address,
        u32::try_from(std::mem::size_of::<u32>()).unwrap(),
    );
    shaders.code_sizes.insert(
        back_program_address,
        u32::try_from(std::mem::size_of::<u32>()).unwrap(),
    );
    shaders
        .fused_continuations
        .entry(front_program_address)
        .or_default()
        .insert(crate::AgcFusedShaderContinuation {
            front_header_address,
            back_header_address,
            back_program_address,
            back_code_size: u32::try_from(std::mem::size_of::<u32>()).unwrap(),
        });

    let snapshot = crate::agc_shader_snapshot(&mut shaders, front_program_address)
        .expect("both exact fused programs decode");
    assert_eq!(snapshot.program.address, front_program_address);
    assert_eq!(snapshot.program.instructions.len(), 2);
    assert_eq!(
        snapshot.program.linked_transfer_target(0),
        Some(u32::try_from(std::mem::size_of::<u32>()).unwrap())
    );
    assert_eq!(
        snapshot
            .program
            .instruction_address(u32::try_from(std::mem::size_of::<u32>()).unwrap()),
        back_program_address
    );
    assert_eq!(snapshot.header_address, back_header_address);
    assert!(snapshot.metadata.is_some());
    assert_eq!(snapshot.supplemental_programs.len(), 1);
    assert_eq!(
        snapshot.supplemental_programs[0].address,
        back_program_address
    );
}
