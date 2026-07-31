// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Data share lowering tests: LDS reads, writes and atomics, `DS_SWIZZLE`, the
//! graphics `ADDTID` spill path and NGG mesh primitive export.

use super::*;

#[test]
fn lowers_every_decoded_ds_operation_with_architectural_offsets_and_atomics() {
    let program = decoded_ds_program(0x1234, false);
    for (instruction, (_, name)) in program.instructions.iter().zip(DS_OPCODES) {
        assert!(instruction.opcode.is(name));
    }
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(32 * 1024),
    )
    .expect("every decoded DS operation should lower");
    assert_eq!(shader.bindings.threadgroup_memory_bytes, 32 * 1024);
    assert!(
        shader
            .source
            .contains("threadgroup atomic_uint* remu_lds [[threadgroup(0)]]")
    );
    assert!(shader.source.contains("+ 0x1234u"));
    assert!(shader.source.contains("+ 0xD0u"));
    assert!(shader.source.contains("+ 0x48u"));
    assert!(shader.source.contains("+ 0x3400u"));
    assert!(shader.source.contains("+ 0x1200u"));
    assert!(shader.source.contains("atomic_fetch_add_explicit"));
    assert!(shader.source.contains("atomic_fetch_sub_explicit"));
    assert!(shader.source.contains("atomic_fetch_min_explicit"));
    assert!(shader.source.contains("atomic_fetch_max_explicit"));
    assert!(shader.source.contains("atomic_fetch_and_explicit"));
    assert!(shader.source.contains("atomic_fetch_or_explicit"));
    assert!(shader.source.contains("atomic_fetch_xor_explicit"));
    assert!(shader.source.contains("atomic_exchange_explicit"));
    assert!(shader.source.contains("remu_ds_inc"));
    assert!(
        shader
            .source
            .contains("observed >= ceiling ? 0u : observed + 1u")
    );
    assert!(
        shader
            .source
            .contains("(observed == 0u || observed > ceiling) ? ceiling : observed - 1u")
    );
    assert!(shader.source.contains("remu_ds_cmpst"));
    assert!(shader.source.contains("if (exec &&"));
    assert!(shader.source.contains("< 8192u"));
    assert!(
        shader
            .source
            .contains("uint remu_thread_index [[thread_index_in_threadgroup]]")
    );
    assert!(
        shader.source.contains("(remu_s_124 & 0xFFFFu)")
            && shader.source.contains("remu_thread_index * 4u"),
        "DS_WRITE_ADDTID_B32 must address M0[15:0] + concatenated offset + TID*4"
    );
    assert!(
        shader
            .source
            .contains("threadgroup_barrier(mem_flags::mem_threadgroup)")
    );
}

#[test]
fn ds_rejects_unrepresentable_resources_instead_of_wrapping_or_privatizing_memory() {
    let memory_program = Program::new(
        0x1000,
        vec![decoded_ds_program(0, false).instructions[0].clone()],
    );
    let graphics_error = compile_msl(
        &memory_program,
        &empty_evaluation(),
        &MetalCompileOptions::vertex(0),
    )
    .expect_err("graphics LDS must not become lane-private scratch");
    assert!(graphics_error.to_string().contains("workgroup-shared"));

    let missing_size = compile_msl(
        &memory_program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(0),
    )
    .expect_err("compute LDS requires its architectural allocation size");
    assert!(missing_size.to_string().contains("non-zero dword-aligned"));

    let oversized = compile_msl(
        &memory_program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(64 * 1024),
    )
    .expect_err("64 KiB guest LDS cannot silently alias 32 KiB Metal memory");
    assert!(oversized.to_string().contains("32768-byte"));

    let gds_program = decoded_ds_program(0, true);
    let gds_error = compile_msl(
        &gds_program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(32 * 1024),
    )
    .expect_err("GDS needs a system-wide resource");
    assert!(gds_error.to_string().contains("system-wide"));
}

#[test]
fn lowers_ngg_lds_and_primitive_exports_to_a_mesh_function() {
    let data_share = |pc, opcode, sources, destinations| Instruction {
        pc,
        encoding: ShaderEncoding::Ds,
        opcode: Opcode::Known(opcode),
        words: vec![0, 0],
        sources,
        destinations,
        control: Some(InstructionControl::DataShare(DataShareControl {
            offset0: 0,
            offset1: 0,
            gds: false,
        })),
    };
    let export = |pc, target, sources| Instruction {
        pc,
        encoding: ShaderEncoding::Exp,
        opcode: Opcode::Known("Exp"),
        words: vec![0, 0],
        sources,
        destinations: Vec::new(),
        control: Some(InstructionControl::Export(ExportControl {
            target,
            enable_mask: if target == 20 { 1 } else { 15 },
            compressed: false,
            done: true,
            valid_mask: false,
        })),
    };
    let program = Program::new(
        0x1000,
        vec![
            data_share(
                0,
                "DsWriteB32",
                vec![Operand::vector(0), Operand::vector(1)],
                Vec::new(),
            ),
            data_share(
                8,
                "DsReadB32",
                vec![Operand::vector(0)],
                vec![Operand::vector(2)],
            ),
            export(
                16,
                20,
                vec![
                    Operand::vector(3),
                    Operand::vector(3),
                    Operand::vector(3),
                    Operand::vector(3),
                ],
            ),
            export(
                24,
                12,
                vec![
                    Operand::vector(4),
                    Operand::vector(5),
                    Operand::vector(6),
                    Operand::vector(7),
                ],
            ),
            Instruction {
                pc: 28,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VMovB32"),
                words: vec![0],
                sources: vec![Operand::vector(8)],
                destinations: vec![Operand::vector(9)],
                control: None,
            },
            Instruction {
                pc: 32,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::mesh(2, 16 * 1024, 96, 96, 96),
    )
    .expect("NGG mesh lowering");

    assert_eq!(shader.stage, MetalShaderStage::Mesh);
    assert!(shader.source.contains("[[mesh]] void remu_main"));
    assert!(
        shader
            .source
            .contains("mesh<RemuVsOut, void, 96, 96, topology::triangle>")
    );
    assert!(
        shader
            .source
            .contains("threadgroup atomic_uint remu_lds_storage[4096]")
    );
    assert!(
        shader
            .source
            .contains("threadgroup_barrier(mem_flags::mem_threadgroup)")
    );
    assert!(
        shader.source.contains("remu_mesh_wave_count"),
        "96-thread NGG subgroups must compute a wave32 count for s3[31:28]"
    );
    assert!(
        shader
            .source
            .contains("((remu_thread_index >> 5u) & 0xfu) << 24u"),
        "96-thread NGG subgroups must pack a wave32 index into s3[27:24]"
    );
    assert!(
        shader
            .source
            .contains("((remu_mesh_wave_count & 0xfu) << 28u)"),
        "96-thread NGG subgroups must pack wave count into s3[31:28] for export reduce"
    );
    assert!(
        shader
            .source
            .contains("remu_out.position = float4(0.0f, 0.0f, 0.0f, 0.0f)"),
        "unexported mesh vertices must default to w=0 so they stay clipped"
    );
    assert!(shader.source.contains("remu_mesh.set_index("));
    assert!(shader.source.contains("remu_mesh.set_vertex("));
    assert!(
        !shader.source.contains("remu_mesh.set_vertex_count("),
        "Metal mesh has no set_vertex_count; vertex occupancy is from set_vertex"
    );
    assert!(
        shader
            .source
            .contains("remu_mesh.set_primitive_count(remu_mesh_output_primitives)")
    );
    assert!(
        shader
            .source
            .contains("if (remu_thread_index < remu_mesh_output_vertices) remu_mesh.set_vertex")
    );
    assert!(
        shader.source.contains("remu_v_8 = remu_group_id.y"),
        "mesh instance ID must come from the second dispatch-grid dimension"
    );
    #[cfg(target_os = "macos")]
    if let Some(compiler) = std::env::var_os("REMU_METAL_COMPILER") {
        let directory =
            std::env::temp_dir().join(format!("remu-ngg-mesh-metal-{}", std::process::id()));
        std::fs::create_dir_all(&directory).expect("create mesh Metal test directory");
        let source = directory.join("mesh.metal");
        let air = directory.join("mesh.air");
        std::fs::write(&source, &shader.source).expect("write generated mesh MSL");
        let output = std::process::Command::new(compiler)
            .args(["-c", "-target", "air64-apple-macosx15.0"])
            .arg("-fmetal-math-mode=safe")
            .arg("-o")
            .arg(&air)
            .arg(&source)
            .output()
            .expect("run Apple Metal compiler");
        assert!(
            output.status.success(),
            "mesh MSL failed to compile:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        std::fs::remove_dir_all(directory).expect("remove mesh Metal test directory");
    }
}

#[test]
fn lowers_linked_ngg_front_and_export_continuation_with_silent_hill_lds_allocation() {
    const CAPTURED_GS_RSRC2: u32 = 0x00fb_0018;
    const CAPTURED_LDS_BYTES: u32 = ((CAPTURED_GS_RSRC2 >> 19) & 0xff) * 128 * 4;
    assert_eq!(CAPTURED_LDS_BYTES, 15_872);
    assert!(CAPTURED_LDS_BYTES <= 16 * 1024);

    let export = |pc, target, sources| Instruction {
        pc,
        encoding: ShaderEncoding::Exp,
        opcode: Opcode::Known("Exp"),
        words: vec![0, 0],
        sources,
        destinations: Vec::new(),
        control: Some(InstructionControl::Export(ExportControl {
            target,
            enable_mask: if target == 20 { 1 } else { 15 },
            compressed: false,
            done: true,
            valid_mask: false,
        })),
    };
    let front = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Ds,
                opcode: Opcode::Known("DsWriteB32"),
                words: vec![0, 0],
                sources: vec![Operand::vector(0), Operand::vector(1)],
                destinations: Vec::new(),
                control: Some(InstructionControl::DataShare(DataShareControl {
                    offset0: 0,
                    offset1: 0,
                    gds: false,
                })),
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::Known("SSetpcB64"),
                words: vec![0xbe80_2006],
                sources: vec![Operand::scalar(6)],
                destinations: Vec::new(),
                control: None,
            },
        ],
    );
    let continuation = Program::new(
        0x4000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sop2,
                opcode: Opcode::Known("SLshlB32"),
                words: vec![0],
                sources: vec![
                    Operand::scalar(107),
                    Operand {
                        kind: remu_shader::OperandKind::EncodedConstant,
                        value: 140,
                    },
                ],
                destinations: vec![Operand::scalar(124)],
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Sop2,
                opcode: Opcode::Known("SOrB32"),
                words: vec![0],
                sources: vec![Operand::scalar(0), Operand::scalar(124)],
                destinations: vec![Operand::scalar(124)],
                control: None,
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::Known("SSendmsg"),
                words: vec![0xbf90_0009],
                sources: vec![Operand {
                    kind: remu_shader::OperandKind::EncodedConstant,
                    value: 9,
                }],
                destinations: Vec::new(),
                control: None,
            },
            export(
                12,
                20,
                vec![
                    Operand::vector(2),
                    Operand::vector(2),
                    Operand::vector(2),
                    Operand::vector(2),
                ],
            ),
            export(
                20,
                12,
                vec![
                    Operand::vector(3),
                    Operand::vector(4),
                    Operand::vector(5),
                    Operand::vector(6),
                ],
            ),
            export(
                28,
                32,
                vec![
                    Operand::vector(7),
                    Operand::vector(8),
                    Operand::vector(9),
                    Operand::vector(10),
                ],
            ),
            Instruction {
                pc: 36,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0xbf81_0000],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    );
    let linked = remu_shader::link_program_continuation(&front, &continuation)
        .expect("exact fused front/back pair");

    let shader = compile_msl(
        &linked,
        &empty_evaluation(),
        &MetalCompileOptions::mesh(1, 16 * 1024, 96, 96, 96),
    )
    .expect("linked NGG program should lower as one mesh function");

    assert_eq!(shader.stage, MetalShaderStage::Mesh);
    assert!(shader.register_footprint.direct_control_flow);
    assert!(
        shader
            .source
            .contains("threadgroup atomic_uint remu_lds_storage[4096]")
    );
    assert!(shader.source.contains("remu_mesh.set_index("));
    assert!(
        shader
            .source
            .contains("atomic_store_explicit(&remu_mesh_output_vertex_count")
    );
    assert!(
        shader
            .source
            .contains("atomic_store_explicit(&remu_mesh_output_primitive_count")
    );
    assert!(shader.source.contains("remu_s_124 & 0x7ffu"));
    assert!(shader.source.contains("(remu_s_124 >> 12u) & 0x7ffu"));
    assert!(
        shader
            .source
            .contains("remu_mesh.set_index(remu_thread_index * 3u")
    );
    assert!(
        !shader
            .source
            .contains("remu_mesh.set_index(remu_wave_lane * 3u")
    );
    assert!(
        !shader.source.contains("remu_mesh.set_vertex_count("),
        "Metal mesh has no set_vertex_count; vertex occupancy is from set_vertex"
    );
    assert!(
        shader
            .source
            .contains("remu_mesh.set_primitive_count(remu_mesh_output_primitives)")
    );
    let allocation = shader
        .source
        .find("atomic_store_explicit(&remu_mesh_output_vertex_count")
        .expect("GS_ALLOC_REQ allocation store");
    let publication = shader
        .source
        .rfind("threadgroup_barrier(mem_flags::mem_threadgroup)")
        .expect("converged output-count publication barrier");
    let output_count = shader
        .source
        .find("remu_mesh.set_primitive_count(remu_mesh_output_primitives)")
        .expect("dynamic Metal primitive count");
    assert!(allocation < publication && publication < output_count);
    assert!(
        shader
            .source
            .contains("mesh<RemuVsOut, void, 96, 96, topology::triangle>")
    );
    assert!(shader.source.contains("remu_out.position = float4("));
    assert!(shader.source.contains("remu_out.param0 = float4("));
    assert!(!shader.source.contains("active = false;"));
}

#[test]
fn lowers_graphics_addtid_spills_to_lane_private_records() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0x100,
                encoding: ShaderEncoding::Ds,
                opcode: Opcode::Known("DsWriteAddtidB32"),
                words: vec![0, 0],
                sources: vec![Operand::vector(7)],
                destinations: Vec::new(),
                control: Some(InstructionControl::DataShare(DataShareControl {
                    offset0: 0,
                    offset1: 2,
                    gds: false,
                })),
            },
            Instruction {
                pc: 0x108,
                encoding: ShaderEncoding::Ds,
                opcode: Opcode::Known("DsReadAddtidB32"),
                words: vec![0, 0],
                sources: Vec::new(),
                destinations: vec![Operand::vector(8)],
                control: Some(InstructionControl::DataShare(DataShareControl {
                    offset0: 0,
                    offset1: 2,
                    gds: false,
                })),
            },
        ],
    );
    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::pixel(Vec::new(), 0, 0),
    )
    .expect("lane-private ADDTID spill should lower in a graphics stage");

    assert!(shader.source.contains("uint remu_addtid_address[1] = {};"));
    assert!(shader.source.contains("remu_addtid_address[0] ="));
    assert!(shader.source.contains("remu_addtid_sequence[0] ="));
    assert!(shader.source.contains("remu_addtid_i < 1u"));
    assert!(shader.source.contains("(remu_s_124 & 0xFFFFu) + 0x200u"));
    assert!(!shader.source.contains("threadgroup atomic_uint* remu_lds"));
    assert_eq!(shader.bindings.threadgroup_memory_bytes, 0);
}

#[test]
fn lowers_all_65536_ds_swizzle_controls_and_all_four_isa_modes() {
    let seed = ds_swizzle_instruction(0);
    let program = Program::new(0x1000, vec![seed]);
    let evaluation = empty_evaluation();
    let options = MetalCompileOptions::vertex(0);
    let mut compiler = Compiler::new(&program, &evaluation, &options).expect("compiler");
    for offset in 0u16..=u16::MAX {
        compiler.body.clear();
        compiler.next_temp = 0;
        compiler
            .emit_data_share(&ds_swizzle_instruction(offset))
            .unwrap_or_else(|error| panic!("swizzle 0x{offset:04X} did not lower: {error}"));
        assert!(compiler.body.contains("simd_shuffle(v[2]"));
        assert!(compiler.body.contains("simd_shuffle(exec ? 1u : 0u"));
        assert!(compiler.body.contains("if (exec) v[8]"));
    }

    let basic =
        compile_vector_instruction(ds_swizzle_instruction(0x041f), MetalShaderStage::Vertex);
    assert!(basic.source.contains("remu_lane & 31u"));
    let quad = compile_vector_instruction(ds_swizzle_instruction(0x80f6), MetalShaderStage::Vertex);
    assert!(quad.source.contains("0x80F6u >> ((remu_lane & 3u) * 2u)"));
    let rotate =
        compile_vector_instruction(ds_swizzle_instruction(0xc020), MetalShaderStage::Vertex);
    assert!(rotate.source.contains("remu_lane + 1u"));
    let fft = compile_vector_instruction(ds_swizzle_instruction(0xe000), MetalShaderStage::Vertex);
    assert!(fft.source.contains("reverse_bits(remu_lane & 31u)"));
    assert!(!fft.source.contains("threadgroup atomic_uint* remu_lds"));
}

#[cfg(target_os = "macos")]
#[test]
fn generated_ds_msl_compiles_with_apple_metal_in_every_valid_stage() {
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
        eprintln!("skipping Apple Metal DS compile test: no metal compiler found");
        return;
    };
    let mut swizzles = [0x041f, 0x80f6, 0xc020, 0xe000]
        .into_iter()
        .enumerate()
        .map(|(index, offset)| {
            let mut instruction = ds_swizzle_instruction(offset);
            instruction.pc = u32::try_from(index * 8).expect("small swizzle program");
            instruction
        })
        .collect::<Vec<_>>();
    swizzles.push(Instruction {
        pc: 32,
        encoding: ShaderEncoding::Sopp,
        opcode: Opcode::END_PROGRAM,
        words: vec![0xbf81_0000],
        sources: Vec::new(),
        destinations: Vec::new(),
        control: None,
    });
    let swizzle_program = Program::new(0x1000, swizzles);
    let programs = [
        (
            "compute",
            decoded_ds_program(0x1234, false),
            MetalCompileOptions::compute(32 * 1024),
        ),
        (
            "vertex",
            swizzle_program.clone(),
            MetalCompileOptions::vertex(0),
        ),
        (
            "pixel",
            swizzle_program,
            MetalCompileOptions::pixel(
                vec![PixelOutputBinding {
                    guest_slot: 0,
                    host_location: 0,
                    kind: PixelOutputKind::Float,
                }],
                0,
                0,
            ),
        ),
    ];
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory =
        std::env::temp_dir().join(format!("remu-ds-metal-{}-{sequence}", std::process::id()));
    std::fs::create_dir_all(&directory).expect("create Metal DS test directory");

    for (stem, program, options) in programs {
        let shader = compile_msl(&program, &empty_evaluation(), &options)
            .expect("valid DS program should lower");
        let source = directory.join(format!("{stem}.metal"));
        let air = directory.join(format!("{stem}.air"));
        std::fs::write(&source, shader.source).expect("write generated DS MSL");
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
            "{stem} DS MSL failed to compile:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    std::fs::remove_dir_all(directory).expect("remove Metal DS test directory");
}
