// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Memory lowering tests: MUBUF/MTBUF buffer access, FLAT/GLOBAL access and the
//! range checks applied to raw and formatted elements.

use super::*;

const BUFFER_MEMORY_OPCODES: [&str; 60] = [
    "BufferLoadFormatX",
    "BufferLoadFormatXy",
    "BufferLoadFormatXyz",
    "BufferLoadFormatXyzw",
    "BufferStoreFormatX",
    "BufferStoreFormatXy",
    "BufferStoreFormatXyz",
    "BufferStoreFormatXyzw",
    "BufferLoadUbyte",
    "BufferLoadSbyte",
    "BufferLoadUshort",
    "BufferLoadSshort",
    "BufferLoadDword",
    "BufferLoadDwordx2",
    "BufferLoadDwordx4",
    "BufferLoadDwordx3",
    "BufferStoreByte",
    "BufferStoreByteD16Hi",
    "BufferStoreShort",
    "BufferStoreShortD16Hi",
    "BufferStoreDword",
    "BufferStoreDwordx2",
    "BufferStoreDwordx4",
    "BufferStoreDwordx3",
    "BufferLoadUbyteD16",
    "BufferLoadUbyteD16Hi",
    "BufferLoadSbyteD16",
    "BufferLoadSbyteD16Hi",
    "BufferLoadShortD16",
    "BufferLoadShortD16Hi",
    "BufferLoadFormatD16HiX",
    "BufferStoreFormatD16HiX",
    "BufferAtomicSwap",
    "BufferAtomicCmpswap",
    "BufferAtomicAdd",
    "BufferAtomicSub",
    "BufferAtomicSmin",
    "BufferAtomicUmin",
    "BufferAtomicSmax",
    "BufferAtomicUmax",
    "BufferAtomicAnd",
    "BufferAtomicOr",
    "BufferAtomicXor",
    "BufferAtomicInc",
    "BufferAtomicDec",
    "BufferLoadFormatD16X",
    "BufferLoadFormatD16Xy",
    "BufferLoadFormatD16Xyz",
    "BufferLoadFormatD16Xyzw",
    "BufferStoreFormatD16X",
    "BufferStoreFormatD16Xy",
    "BufferStoreFormatD16Xyz",
    "BufferStoreFormatD16Xyzw",
    "TBufferLoadFormatX",
    "TBufferLoadFormatXy",
    "TBufferLoadFormatXyz",
    "TBufferLoadFormatXyzw",
    "TBufferStoreFormatX",
    "TBufferStoreFormatXy",
    "TBufferStoreFormatXyz",
];

const EXTRA_BUFFER_MEMORY_OPCODE: &str = "TBufferStoreFormatXyzw";

const TYPED_D16_MEMORY_OPCODES: [&str; 8] = [
    "TBufferLoadFormatD16X",
    "TBufferLoadFormatD16Xy",
    "TBufferLoadFormatD16Xyz",
    "TBufferLoadFormatD16Xyzw",
    "TBufferStoreFormatD16X",
    "TBufferStoreFormatD16Xy",
    "TBufferStoreFormatD16Xyz",
    "TBufferStoreFormatD16Xyzw",
];

const GLOBAL_MEMORY_OPCODES: [&str; 24] = [
    "GlobalLoadUbyte",
    "GlobalLoadSbyte",
    "GlobalLoadUshort",
    "GlobalLoadSshort",
    "GlobalLoadDword",
    "GlobalLoadDwordx2",
    "GlobalLoadDwordx4",
    "GlobalLoadDwordx3",
    "GlobalStoreByte",
    "GlobalStoreByteD16Hi",
    "GlobalStoreShort",
    "GlobalStoreShortD16Hi",
    "GlobalStoreDword",
    "GlobalStoreDwordx2",
    "GlobalStoreDwordx4",
    "GlobalStoreDwordx3",
    "GlobalLoadUbyteD16",
    "GlobalLoadUbyteD16Hi",
    "GlobalLoadSbyteD16",
    "GlobalLoadSbyteD16Hi",
    "GlobalLoadShortD16",
    "GlobalLoadShortD16Hi",
    "GlobalAtomicAdd",
    "GlobalAtomicUMax",
];

fn memory_dword_count_for_test(name: &str) -> u32 {
    if name.contains("FormatD16") {
        if name.ends_with("Xyz") || name.ends_with("Xyzw") {
            2
        } else {
            1
        }
    } else if name.ends_with("Xyzw") || name.ends_with("Dwordx4") {
        4
    } else if name.ends_with("Xyz") || name.ends_with("Dwordx3") {
        3
    } else if name.ends_with("Xy") || name.ends_with("Dwordx2") || name.ends_with("Cmpswap") {
        2
    } else {
        1
    }
}

fn complete_memory_program() -> (Program, ShaderEvaluation) {
    let names = BUFFER_MEMORY_OPCODES
        .into_iter()
        .chain([EXTRA_BUFFER_MEMORY_OPCODE])
        .chain(TYPED_D16_MEMORY_OPCODES)
        .chain(GLOBAL_MEMORY_OPCODES);
    let mut pcs = Vec::new();
    let mut instructions = names
        .enumerate()
        .map(|(index, name)| {
            let pc = u32::try_from(index * 8).expect("small memory test program");
            pcs.push(pc);
            let dword_count = memory_dword_count_for_test(name);
            let (encoding, control) = if name.starts_with("Global") {
                (
                    ShaderEncoding::Flat,
                    InstructionControl::GlobalMemory(remu_shader::GlobalMemoryControl {
                        dword_count,
                        vector_address: 1,
                        vector_data: 16,
                        vector_destination: 32,
                        scalar_address: Some(8),
                        offset_bytes: -4,
                        dlc: true,
                        lds: false,
                        glc: true,
                        slc: true,
                    }),
                )
            } else {
                (
                    if name.starts_with("TBuffer") {
                        ShaderEncoding::Mtbuf
                    } else {
                        ShaderEncoding::Mubuf
                    },
                    InstructionControl::BufferMemory(remu_shader::BufferMemoryControl {
                        dword_count,
                        vector_address: 1,
                        vector_data: 16,
                        scalar_resource: 0,
                        instruction_format: name.starts_with("TBuffer").then_some(60),
                        offset_bytes: 4,
                        index_enabled: true,
                        offset_enabled: true,
                        dlc: true,
                        lds: false,
                        glc: true,
                        slc: true,
                        tfe: name.contains("Load") && !name.contains("Atomic"),
                    }),
                )
            };
            Instruction {
                pc,
                encoding,
                opcode: Opcode::Known(name),
                words: vec![0, 0],
                sources: vec![
                    Operand::vector(1),
                    Operand::scalar(0),
                    Operand::source(128, None),
                ],
                destinations: Vec::new(),
                control: Some(control),
            }
        })
        .collect::<Vec<_>>();
    let image_pc = u32::try_from(instructions.len() * 8).expect("small memory test program");
    let image_control = remu_shader::ImageControl {
        dmask: 0xf,
        vector_address: 1,
        address_registers: vec![1, 2],
        vector_data: 64,
        scalar_resource: 16,
        scalar_sampler: 24,
        dimension: 1,
        is_array: false,
        glc: true,
        slc: true,
        a16: false,
        d16: false,
    };
    instructions.push(Instruction {
        pc: image_pc,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageLoad"),
        words: vec![0, 0],
        sources: vec![Operand::vector(1), Operand::vector(2), Operand::scalar(16)],
        destinations: (64..68).map(Operand::vector).collect(),
        control: Some(InstructionControl::Image(image_control.clone())),
    });
    let image_store_pc = image_pc + 8;
    let mut image_store_control = image_control.clone();
    image_store_control.vector_data = 68;
    image_store_control.dimension = 5;
    image_store_control.is_array = true;
    instructions.push(Instruction {
        pc: image_store_pc,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageStore"),
        words: vec![0, 0],
        sources: vec![
            Operand::vector(1),
            Operand::vector(2),
            Operand::scalar(16),
            Operand::vector(68),
            Operand::vector(69),
            Operand::vector(70),
            Operand::vector(71),
        ],
        destinations: Vec::new(),
        control: Some(InstructionControl::Image(image_store_control.clone())),
    });
    let end_pc = u32::try_from(instructions.len() * 8).expect("small memory test program");
    instructions.push(Instruction {
        pc: end_pc,
        encoding: ShaderEncoding::Sopp,
        opcode: Opcode::END_PROGRAM,
        words: vec![0xbf81_0000],
        sources: Vec::new(),
        destinations: Vec::new(),
        control: None,
    });
    let mut evaluation = empty_evaluation();
    let selectors = 4 | (5 << 3) | (6 << 6) | (7 << 9);
    evaluation.initial_scalar_registers[0] = 0x1000;
    evaluation.initial_scalar_registers[1] = 16 << 16;
    evaluation.initial_scalar_registers[2] = 64 * 1024;
    evaluation.initial_scalar_registers[3] = selectors | (60 << 12) | (1 << 24) | (3 << 28);
    evaluation.initial_scalar_registers[8] = 0x1000;
    evaluation.scalar_registers = evaluation.initial_scalar_registers.clone();
    evaluation
        .global_memory_bindings
        .push(remu_shader::GlobalMemoryBinding {
            scalar_address: 0,
            base_address: 0x1000,
            instruction_pcs: pcs.into(),
            byte_count: 64 * 1024,
            data: vec![0; 64 * 1024],
            writable: true,
            write_back_to_guest: true,
        });
    let mut identity_image_descriptor = vec![0; 8];
    identity_image_descriptor[3] = 0xfac;
    evaluation.image_bindings.push(remu_shader::ImageBinding {
        pc: image_pc,
        opcode: Opcode::Known("ImageLoad"),
        control: image_control,
        resource_descriptor: identity_image_descriptor.clone().into(),
        sampler_descriptor: Vec::new().into(),
        mip_level: None,
    });
    let mut swizzled_store_descriptor = identity_image_descriptor;
    swizzled_store_descriptor[3] = 0x9f5;
    evaluation.image_bindings.push(remu_shader::ImageBinding {
        pc: image_store_pc,
        opcode: Opcode::Known("ImageStore"),
        control: image_store_control,
        resource_descriptor: swizzled_store_descriptor.into(),
        sampler_descriptor: Vec::new().into(),
        mip_level: None,
    });
    (Program::new(0x1000, instructions), evaluation)
}

fn format_x_load_program(
    site_count: u32,
    instruction_format: Option<u32>,
) -> (Program, ShaderEvaluation) {
    let instruction_pcs = (0..site_count).map(|site| site * 8).collect::<Vec<_>>();
    let mut instructions = instruction_pcs
        .iter()
        .copied()
        .map(|pc| Instruction {
            pc,
            encoding: if instruction_format.is_some() {
                ShaderEncoding::Mtbuf
            } else {
                ShaderEncoding::Mubuf
            },
            opcode: Opcode::Known(if instruction_format.is_some() {
                "TBufferLoadFormatX"
            } else {
                "BufferLoadFormatX"
            }),
            words: vec![0, 0],
            sources: vec![
                Operand::vector(0),
                Operand::scalar(0),
                Operand::source(128, None),
            ],
            destinations: vec![Operand::vector(0)],
            control: Some(InstructionControl::BufferMemory(
                remu_shader::BufferMemoryControl {
                    dword_count: 1,
                    vector_address: 0,
                    vector_data: 0,
                    scalar_resource: 0,
                    instruction_format,
                    offset_bytes: 0,
                    index_enabled: true,
                    offset_enabled: false,
                    dlc: false,
                    lds: false,
                    glc: false,
                    slc: false,
                    tfe: false,
                },
            )),
        })
        .collect::<Vec<_>>();
    instructions.push(Instruction {
        pc: site_count * 8,
        encoding: ShaderEncoding::Sopp,
        opcode: Opcode::END_PROGRAM,
        words: vec![0xbf81_0000],
        sources: Vec::new(),
        destinations: Vec::new(),
        control: None,
    });

    let mut evaluation = empty_evaluation();
    evaluation.initial_scalar_registers[0] = 0x1000;
    evaluation.initial_scalar_registers[1] = 16 << 16;
    evaluation.initial_scalar_registers[2] = 4096;
    evaluation.initial_scalar_registers[3] =
        4 | (5 << 3) | (6 << 6) | (7 << 9) | (0x16 << 12) | (3 << 28);
    evaluation.scalar_registers = evaluation.initial_scalar_registers.clone();
    evaluation
        .global_memory_bindings
        .push(remu_shader::GlobalMemoryBinding {
            scalar_address: 0,
            base_address: 0x1000,
            instruction_pcs: instruction_pcs.into(),
            byte_count: 4096,
            data: vec![0; 4096],
            writable: false,
            write_back_to_guest: false,
        });
    (Program::new(0x800d_700000, instructions), evaluation)
}

#[test]
fn captured_fixed_format_x_loads_decode_only_observable_components() {
    // Silent Hill's captured 579,896-byte pixel shader contains 72 static
    // TBUFFER_LOAD_FORMAT_X sites with an instruction-selected format. AMD's MTBUF semantics
    // define identity DST_SEL X000 for FORMAT_X, so only canonical X is observable.
    let (program, evaluation) = format_x_load_program(72, Some(0x16));
    let shader = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
        .expect("captured typed format load shape should lower");

    assert_eq!(
        shader.source.matches("remu_format_layout(").count(),
        73,
        "the prelude plus one observable X layout per captured static site"
    );
    assert!(
        shader.source.len() < 170_000,
        "fixed FORMAT_X sites must not restore four-component expansion: {} bytes",
        shader.source.len()
    );
    assert!(
        !shader.source.contains("= remu_s_3;"),
        "instruction-selected loads do not read descriptor format/select fields"
    );
}

#[test]
fn descriptor_selected_format_x_load_keeps_all_canonical_components() {
    let (program, evaluation) = format_x_load_program(1, None);
    let shader = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
        .expect("descriptor-selected format load should lower");

    assert_eq!(
        shader.source.matches("remu_format_layout(").count(),
        5,
        "the descriptor can route any of four canonical components to X"
    );
    assert!(shader.source.contains(">> 0u) & 7u"));
    assert!(shader.source.contains("== 7u ?"));
}

#[test]
fn lowers_complete_decoded_global_mubuf_and_mtbuf_memory_surface() {
    let (program, evaluation) = complete_memory_program();
    let shader = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
        .expect("complete decoded VMEM surface should lower");
    for operation in [
        "remu_atomic_cmpswap",
        "remu_atomic_inc",
        "remu_store_format_bits",
        "remu_load_format_bits",
        "remu_format_to_d16",
        "remu_format_from_d16",
    ] {
        assert!(shader.source.contains(operation), "missing {operation}");
    }
    assert!(shader.source.contains("0xFFFF0000u"));
    assert!(shader.source.contains("0xFFFFFF00u"));
    assert!(shader.source.contains("remu_range_valid"));
    assert!(
        shader
            .source
            .contains("texture2d<float, access::read> tex0")
    );
    assert!(
        shader
            .source
            .contains("texture2d<float, access::read_write> tex1")
    );
    assert!(shader.source.contains("tex0.read(uint2("));
    assert!(shader.source.contains("tex1.write("));
    assert!(
            shader.source.contains(
                "vec<float, 4>(as_type<float>(remu_v_68), as_type<float>(remu_v_69), as_type<float>(remu_v_70), as_type<float>(remu_v_71))"
            ),
            "image stores must map enabled DMASK components directly to physical RGBA channels"
        );
    assert!(
        !shader.source.contains("clamp(as_type<int>"),
        "out-of-range image loads must return zero instead of sampling a clamped edge texel"
    );
    assert!(
        shader.source.contains("= vec<float, 4>(0.0f);"),
        "image loads must initialize an architectural zero result before their bounds-checked read"
    );
    assert!(
        shader.source.contains(" & 3ul"),
        "buffer descriptors must address their dynamically bound, dword-aligned Metal buffer without embedding the evaluated guest base"
    );
}

#[test]
fn range_checks_raw_dword_vectors_per_component() {
    let expression = buffer_oob_expression(
        "null",
        "mode",
        "index",
        "records",
        "offset",
        "stride",
        "swizzled",
        "buffer_offset",
        "reduced_records",
        "4u",
        12,
    );

    assert!(expression.contains("offset + 12u >= stride"));
    assert!(expression.contains("remu_range_valid(offset + 12u, 4u, stride)"));
    assert!(expression.contains("remu_range_valid(buffer_offset + 12u, 4u, reduced_records)"));
}

#[test]
fn formatted_elements_use_record_stride_without_total_size_alignment() {
    let expression =
        format_buffer_element_oob_expression("descriptor_oob", "record_address", "12u", "28u");

    assert_eq!(
        expression,
        "descriptor_oob || 12u == 0u || !remu_range_valid(record_address, 12u, 28u)"
    );
    assert!(
        !expression.contains('%'),
        "a 12-byte float3 at offsets 0, 28, and 56 is legal in a 28-byte record layout"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn generated_memory_msl_compiles_with_apple_metal() {
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

    fn find_metal_compiler() -> Option<PathBuf> {
        if let Some(path) = std::env::var_os("REMU_METAL_COMPILER").map(PathBuf::from)
            && path.is_file()
        {
            return Some(path);
        }
        let bundled = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../.remu/toolchains/Metal.xctoolchain/usr/bin/metal");
        if bundled.is_file() {
            return Some(bundled);
        }
        let output = Command::new("xcrun")
            .args(["--find", "metal"])
            .output()
            .ok()?;
        output
            .status
            .success()
            .then(|| PathBuf::from(String::from_utf8_lossy(&output.stdout).trim().to_owned()))
    }

    let Some(compiler) = find_metal_compiler() else {
        eprintln!("skipping Apple Metal memory compile test: no metal compiler found");
        return;
    };
    let (program, mut evaluation) = complete_memory_program();
    evaluation.compute_system_registers = Some(ComputeSystemRegisters {
        work_group_x: Some(116),
        work_group_y: Some(117),
        work_group_z: Some(118),
        thread_group_size: Some(119),
    });
    let shader = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
        .expect("complete decoded VMEM surface should lower");
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "remu-memory-metal-{}-{sequence}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).expect("create Metal memory test directory");
    let source = directory.join("memory.metal");
    let air = directory.join("memory.air");
    std::fs::write(&source, shader.source).expect("write generated memory MSL");
    let output = Command::new(&compiler)
        .args(["-c", "-target", "air64-apple-macosx15.0"])
        .arg("-fmetal-math-mode=safe")
        .arg("-o")
        .arg(&air)
        .arg(&source)
        .output()
        .expect("run Apple Metal compiler");
    assert!(
        output.status.success(),
        "memory MSL failed to compile:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::remove_dir_all(directory).expect("remove Metal memory test directory");
}
