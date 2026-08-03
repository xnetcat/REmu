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
        unorm: false,
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

fn set_image_texture_type(
    evaluation: &mut ShaderEvaluation,
    binding_index: usize,
    texture_type: u32,
) {
    let binding = &mut evaluation.image_bindings[binding_index];
    let mut descriptor = binding.resource_descriptor.to_vec();
    descriptor[3] = (descriptor[3] & 0x0fff_ffff) | (texture_type << 28);
    binding.resource_descriptor = descriptor.into();
}

fn set_image_address_dimension(
    program: Program,
    evaluation: &mut ShaderEvaluation,
    binding_index: usize,
    dimension: u32,
) -> Program {
    let binding = &mut evaluation.image_bindings[binding_index];
    binding.control.dimension = dimension;
    binding.control.is_array = matches!(dimension, 4 | 5 | 7);
    let pc = binding.pc;
    let mut instructions = program.instructions;
    let instruction = instructions
        .iter_mut()
        .find(|instruction| instruction.pc == pc)
        .expect("image binding instruction must exist");
    let Some(InstructionControl::Image(control)) = instruction.control.as_mut() else {
        panic!("image binding instruction must carry image control");
    };
    control.dimension = dimension;
    control.is_array = binding.control.is_array;
    Program::new(program.address, instructions)
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
fn storage_images_preserve_2d_array_and_3d_resource_dimensions() {
    let (program, mut array_store_evaluation) = complete_memory_program();
    set_image_texture_type(&mut array_store_evaluation, 1, 13);
    let array_store = compile_msl(
        &program,
        &array_store_evaluation,
        &MetalCompileOptions::compute(0),
    )
    .expect("2D-array image store should lower");
    assert!(
        array_store
            .source
            .contains("texture2d_array<float, access::read_write> tex1")
    );
    assert!(array_store.source.contains("tex1.get_array_size()"));
    let array_store_line = array_store
        .source
        .lines()
        .find(|line| line.contains("tex1.write("))
        .expect("2D-array store call");
    assert!(array_store_line.contains("uint2("));
    assert!(array_store_line.contains(", (uint)"));
    assert!(!array_store_line.contains("uint3("));

    let (program, mut array_load_evaluation) = complete_memory_program();
    let program = set_image_address_dimension(program, &mut array_load_evaluation, 0, 5);
    set_image_texture_type(&mut array_load_evaluation, 0, 13);
    let array_load = compile_msl(
        &program,
        &array_load_evaluation,
        &MetalCompileOptions::compute(0),
    )
    .expect("2D-array image load should lower");
    assert!(
        array_load
            .source
            .contains("texture2d_array<float, access::read> tex0")
    );
    assert!(array_load.source.contains("tex0.get_array_size()"));
    let array_load_line = array_load
        .source
        .lines()
        .find(|line| line.contains("tex0.read("))
        .expect("2D-array load call");
    assert!(array_load_line.contains("uint2("));
    assert!(array_load_line.contains(", (uint)"));
    assert!(!array_load_line.contains("uint3("));

    let (program, mut volume_evaluation) = complete_memory_program();
    set_image_texture_type(&mut volume_evaluation, 1, 10);
    let volume = compile_msl(
        &program,
        &volume_evaluation,
        &MetalCompileOptions::compute(0),
    )
    .expect("3D image store should lower");
    assert!(
        volume
            .source
            .contains("texture3d<float, access::read_write> tex1")
    );
    assert!(volume.source.contains("tex1.get_depth()"));
    let volume_store_line = volume
        .source
        .lines()
        .find(|line| line.contains("tex1.write("))
        .expect("3D store call");
    assert!(volume_store_line.contains("uint3("));
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
fn structured_buffer_oob_honors_expanded_metal_binding_span() {
    // Poppy Playtime UI reloads V# from SMEM with guest NUM_RECORDS=4 while the presenter
    // expands the Metal buffer to the instanced draw span. OOB_SELECT=0 must accept those
    // extra records or every glyph instance past the guest count fetches zeroes (tofu).
    let stride = 16_u32;
    let guest_records = 4_u32;
    let draw_records = 256_u32;
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Mubuf,
                opcode: Opcode::Known("BufferLoadFormatXy"),
                words: vec![0, 0],
                sources: vec![
                    Operand::vector(1),
                    Operand::scalar(0),
                    Operand::source(128, None),
                ],
                destinations: Vec::new(),
                control: Some(InstructionControl::BufferMemory(
                    remu_shader::BufferMemoryControl {
                        dword_count: 2,
                        vector_address: 1,
                        vector_data: 16,
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
                    },
                )),
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0xbf81_0000],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    );
    let mut evaluation = empty_evaluation();
    evaluation.initial_scalar_registers[0] = 0x1000;
    evaluation.initial_scalar_registers[1] = stride << 16;
    evaluation.initial_scalar_registers[2] = guest_records;
    evaluation.initial_scalar_registers[3] = 0 << 28; // OOB_SELECT=0
    evaluation.scalar_registers = evaluation.initial_scalar_registers.clone();
    evaluation
        .global_memory_bindings
        .push(remu_shader::GlobalMemoryBinding {
            scalar_address: 0,
            base_address: 0x1000,
            instruction_pcs: vec![0].into(),
            byte_count: usize::try_from(stride * draw_records).unwrap(),
            data: vec![0; usize::try_from(stride * draw_records).unwrap()],
            writable: false,
            write_back_to_guest: false,
        });

    let shader = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
        .expect("expanded structured buffer load should lower");
    assert!(
        shader
            .source
            .contains("max(")
            && shader.source.contains("/ "),
        "structured OOB must lift NUM_RECORDS from the expanded Metal binding length"
    );
    assert!(
        shader.source.contains("<= 1u &&"),
        "record lift must stay limited to OOB_SELECT 0/1"
    );
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

#[test]
fn formatted_srgb_buffers_decode_color_and_preserve_linear_alpha() {
    fn decode(encoded: f32) -> f32 {
        if encoded <= 0.04045 {
            encoded / 12.92
        } else {
            ((encoded + 0.055) / 1.055).powf(2.4)
        }
    }

    fn encode(linear: f32) -> f32 {
        if linear <= 0.003_130_8 {
            linear * 12.92
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        }
    }

    assert!((decode(0.04045) - 0.003_130_8).abs() < 1.0e-7);
    assert!((decode(0.5) - 0.214_041_14).abs() < 1.0e-7);
    assert!((encode(0.003_130_8) - 0.040_449_936).abs() < 1.0e-7);
    assert!((encode(0.214_041_14) - 0.5).abs() < 1.0e-7);

    let (program, evaluation) = complete_memory_program();
    let shader = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
        .expect("formatted buffer shader should lower");
    assert!(
        shader.source.contains(
            "static inline uint remu_format_convert(uint raw, uint bits, uint nfmt, uint dfmt, uint component)"
        )
    );
    assert!(shader.source.contains("case 9u: {"));
    assert!(
        shader
            .source
            .contains("component == 3u ? normalized : remu_srgb_decode(normalized)")
    );
    assert!(
        shader
            .source
            .contains("component == 3u ? clamp(floating, 0.0f, 1.0f)")
    );
    for component in 0..4 {
        assert!(
            shader.source.contains(&format!(", {component}u)")),
            "formatted calls must identify component {component} so only alpha bypasses gamma"
        );
    }
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
    let (array_program, mut array_evaluation) = complete_memory_program();
    let array_program =
        set_image_address_dimension(array_program, &mut array_evaluation, 0, 5);
    set_image_texture_type(&mut array_evaluation, 0, 13);
    set_image_texture_type(&mut array_evaluation, 1, 13);
    let array_shader = compile_msl(
        &array_program,
        &array_evaluation,
        &MetalCompileOptions::compute(0),
    )
    .expect("2D-array image load and store should lower");
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "remu-memory-metal-{}-{sequence}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).expect("create Metal memory test directory");
    for (name, shader) in [("memory", shader), ("storage-array", array_shader)] {
        let source = directory.join(format!("{name}.metal"));
        let air = directory.join(format!("{name}.air"));
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
            "{name} MSL failed to compile:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    std::fs::remove_dir_all(directory).expect("remove Metal memory test directory");
}
