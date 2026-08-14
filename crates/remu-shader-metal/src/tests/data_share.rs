// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Data share lowering tests: LDS reads, writes and atomics, `DS_SWIZZLE`, the
//! graphics `ADDTID` spill path and NGG mesh primitive export.

use super::*;

#[test]
fn no_gs_fixed_allocation_ignores_gs_alloc_req_and_publishes_meshlet_counts() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VMovB32"),
                words: vec![0],
                sources: vec![Operand::vector(1)],
                destinations: vec![Operand::vector(9)],
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VMovB32"),
                words: vec![0],
                sources: vec![Operand::vector(2)],
                destinations: vec![Operand::vector(10)],
                control: None,
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VMovB32"),
                words: vec![0],
                sources: vec![Operand::vector(8)],
                destinations: vec![Operand::vector(11)],
                control: None,
            },
            Instruction {
                pc: 12,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::Known("SMovB32"),
                words: vec![0],
                sources: vec![Operand::scalar(2)],
                destinations: vec![Operand::scalar(80)],
                control: None,
            },
            Instruction {
                pc: 16,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::Known("SMovB32"),
                words: vec![0],
                sources: vec![Operand::scalar(126)],
                destinations: vec![Operand::scalar(81)],
                control: None,
            },
            Instruction {
                pc: 20,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::Known("SMovB32"),
                words: vec![0],
                sources: vec![Operand::scalar(127)],
                destinations: vec![Operand::scalar(82)],
                control: None,
            },
            Instruction {
                pc: 24,
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
            Instruction {
                pc: 28,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0xbf81_0000],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    );
    let mut options = MetalCompileOptions::mesh(0, 0, 256, 256, 256);
    options.enable_fixed_ngg_allocation_fallback();
    let shader = compile_msl(&program, &empty_evaluation(), &options)
        .expect("fixed no-GS mesh ABI should lower");

    assert_eq!(options.mesh_max_vertex_count, 64);
    assert_eq!(options.mesh_max_primitive_count, 64);
    assert_eq!(options.mesh_threadgroup_size, 64);
    assert!(shader.source.contains("remu_group_id.x * 130u"));
    assert!(shader.source.contains("remu_meshlet_base + 66u"));
    assert!(
        shader
            .source
            .contains("remu_out.position = float4(0.0f, 0.0f, 0.0f, 1.0f)")
    );
    assert!(
        !shader
            .source
            .contains("remu_s_2 = ((remu_mesh_input_vertex_count"),
        "the fixed compatibility ABI must preserve the guest's incoming s2"
    );
    assert!(shader.source.contains("simd_ballot(true)"));
    assert!(!shader.source.contains("remu_s_126 = 0xffffffffu;"));
    assert!(
        shader
            .source
            .contains("remu_v_1 = remu_mesh_primitive_base + remu_thread_index")
    );
    assert!(shader.source.contains("remu_v_2 = 0u"));
    assert!(shader.source.contains("remu_v_8 = remu_group_id.y"));
    assert!(!shader.source.contains("(remu_s_124 >> 12u) & 0x7ffu"));
    assert!(
        shader
            .source
            .contains("if (remu_thread_index < remu_mesh_input_vertex_count) remu_mesh.set_vertex")
    );
    assert!(
        shader
            .source
            .contains("remu_mesh.set_primitive_count(remu_mesh_output_primitives)")
    );
}

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
fn lowers_sub_dword_lds_accesses_as_field_inserts_and_extending_extracts() {
    // DS_WRITE_B8/B16 and DS_READ_{I8,U8,I16,U16} address bytes inside the dword-indexed
    // threadgroup array, so lowering must insert and extract the addressed field rather than
    // overwrite or return the whole dword.
    let mut words = Vec::new();
    for opcode in [0x1e, 0x1f, 0x39, 0x3a, 0x3b, 0x3c] {
        words.push(0xd800_0000 | (opcode << 18) | 0x1234);
        words.push(1u32 | (2u32 << 8) | (3u32 << 16) | (8u32 << 24));
    }
    words.extend([0xbf8a_0000, 0xbf81_0000]);
    let program = decode_program(0x1000, &words).expect("sub-dword DS program");
    for (instruction, name) in program.instructions.iter().zip([
        "DsWriteB8",
        "DsWriteB16",
        "DsReadI8",
        "DsReadU8",
        "DsReadI16",
        "DsReadU16",
    ]) {
        assert!(instruction.opcode.is(name), "decoder must name {name}");
    }

    let shader = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::compute(32 * 1024),
    )
    .expect("sub-dword LDS accesses should lower");

    assert!(
        shader.source.contains("remu_ds_insert(remu_lds +"),
        "byte and half writes must insert into the containing dword"
    );
    assert!(
        shader
            .source
            .contains("insert_bits(observed, value, offset, bits)"),
        "the insert must be a compare-exchange so neighbouring lanes keep their bytes"
    );
    assert_eq!(
        shader.source.matches("& 3u) * 8u").count(),
        3,
        "byte accesses select the field from the low two address bits"
    );
    assert_eq!(
        shader.source.matches("& 2u) * 8u").count(),
        3,
        "half accesses ignore the low address bit so the field cannot straddle the dword"
    );
    assert_eq!(
        shader.source.matches("remu_ds_insert(remu_lds +").count(),
        2,
        "one insert per sub-dword write"
    );
    assert!(
        shader.source.contains(", 8u, ") && shader.source.contains(", 16u, "),
        "field width must reach the insert helper"
    );
    assert_eq!(
        shader
            .source
            .matches("as_type<uint>(extract_bits(as_type<int>(t")
            .count(),
        2,
        "DS_READ_I8 and DS_READ_I16 must sign-extend their field"
    );
    assert!(
        shader.source.contains("= extract_bits(t"),
        "unsigned reads must zero-extend the field"
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
            .contains("((remu_mesh_wave_index & 0xfu) << 24u"),
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
    assert!(
        shader.source.contains(
            "for (uint remu_null_primitive = remu_thread_index; remu_null_primitive < 96u;"
        ),
        "GS_ALLOC_REQ slots without EXP PRIM must be initialized as null primitives"
    );
    for component in 0..3 {
        assert!(shader.source.contains(&format!(
            "remu_mesh.set_index(remu_null_primitive * 3u + {component}u, 0u);"
        )));
    }
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
            .contains("if (remu_thread_index < 96u) remu_mesh.set_vertex")
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
fn packs_wave64_merged_wave_info_for_split_ngg_mesh_exports() {
    // Wave64 NGG exports (Poppy Playtime UI/LUT quads) recover their meshlet thread index as
    // `s3[27:24] * 64 + v_mbcnt`, so the merged wave info must count 64-lane guest waves
    // rather than SIMD32 halves once the presenter selects the split-wave64 lowering.
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0,
            encoding: ShaderEncoding::Sopp,
            opcode: Opcode::END_PROGRAM,
            words: vec![0],
            sources: Vec::new(),
            destinations: Vec::new(),
            control: None,
        }],
    );
    let mut options = MetalCompileOptions::mesh(1, 16 * 1024, 128, 128, 128);
    options.wave_lane_count = 64;
    let shader =
        compile_msl(&program, &empty_evaluation(), &options).expect("wave64 NGG mesh lowering");
    assert_eq!(shader.stage, MetalShaderStage::Mesh);
    assert!(
        shader
            .source
            .contains("uint remu_wave_index = remu_thread_index >> 6u;")
    );
    assert!(
        shader
            .source
            .contains("uint remu_mesh_wave_index = remu_thread_index >> 6u;")
    );
    assert!(
        shader
            .source
            .contains("((remu_mesh_wave_index & 0xfu) << 24u"),
        "wave64 NGG subgroups must pack a wave64 index into s3[27:24]"
    );
    assert!(
        shader.source.contains(
            "remu_mesh_input_vertex_count > remu_mesh_wave_base ? remu_mesh_input_vertex_count - remu_mesh_wave_base : 0u, 63u + 1u"
        ),
        "merged-wave byte 0 must carry the current wave's ES invocation count"
    );
    assert!(
        shader.source.contains(
            "remu_mesh_input_primitive_count > remu_mesh_wave_base ? remu_mesh_input_primitive_count - remu_mesh_wave_base : 0u, 63u + 1u"
        ),
        "merged-wave byte 1 must carry the current wave's GS invocation count"
    );
    assert!(
        shader
            .source
            .contains("(remu_mesh_wave_vertex_count & 0xffu)"),
    );
    assert!(
        shader
            .source
            .contains("((remu_mesh_wave_primitive_count & 0xffu) << 8u)"),
    );
    assert!(
        shader.source.contains("+ 63u) >> 6u"),
        "wave64 NGG subgroups must count 64-lane guest waves for s3[31:28]"
    );
    assert!(
        !shader
            .source
            .contains("uint remu_mesh_wave_index = remu_thread_index >> 5u;"),
        "SIMD32 halves must not leak into the guest-visible merged wave info"
    );
}

#[test]
fn empty_workaround_packed_export_sequence_refuses_without_proven_ngg_registers() {
    // J3: the snapshot/restore peephole is deleted. Without Q5-proven NggRegisters the pattern
    // must fail loudly rather than silently rewrite counts.
    let program = empty_workaround_ngg_program();
    let mut options = MetalCompileOptions::mesh(1, 16 * 1024, 192, 64, 192);
    options.wave_lane_count = 64;
    let error = compile_msl(&program, &empty_evaluation(), &options)
        .expect_err("unproven empty-workaround must refuse");
    assert!(
        error.to_string().contains("proven NggRegisters"),
        "loud refuse must name the Q5 proof requirement: {error}"
    );
}

#[test]
fn proven_ngg_registers_lower_empty_workaround_without_the_snapshot_peephole() {
    let program = empty_workaround_ngg_program();
    let mut options = MetalCompileOptions::mesh_from_ngg_registers(
        1,
        192,
        NggRegisters {
            pgm_rsrc1: 0x8,
            // 16 KiB guest LDS (field 0x20) — matches the prior explicit lds_size_bytes argument.
            pgm_rsrc2: 0x20 << 19,
            gs_max_vert_out: 64,
            ge_max_output_per_subgroup: 64,
            esgs_ring_itemsize: 0,
            primgen_passthrough: false,
        },
    );
    options.wave_lane_count = 64;
    let shader = compile_msl(&program, &empty_evaluation(), &options)
        .expect("proven NggRegisters may lower the guest empty-workaround as-is");
    assert!(
        !shader.source.contains("remu_ngg_packed_export_counts"),
        "J3 deleted the snapshot/restore peephole"
    );
    assert!(
        !shader.source.contains("remu_ngg_export_counts_pending"),
        "J3 deleted the snapshot/restore peephole"
    );
}

fn empty_workaround_ngg_program() -> Program {
    Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sopk,
                opcode: Opcode::Known("SCmpkLeU32"),
                words: vec![0xb704_ffff],
                sources: vec![Operand {
                    kind: OperandKind::EncodedConstant,
                    value: 65535,
                }],
                destinations: vec![Operand::scalar(4)],
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Sop2,
                opcode: Opcode::Known("SCselectB32"),
                words: vec![0x8502_6a81],
                sources: vec![
                    Operand {
                        kind: OperandKind::EncodedConstant,
                        value: 129,
                    },
                    Operand::scalar(106),
                ],
                destinations: vec![Operand::scalar(2)],
                control: None,
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Sop2,
                opcode: Opcode::Known("SMaxU32"),
                words: vec![0x8506_6a81],
                sources: vec![
                    Operand {
                        kind: OperandKind::EncodedConstant,
                        value: 129,
                    },
                    Operand::scalar(2),
                ],
                destinations: vec![Operand::scalar(107)],
                control: None,
            },
            Instruction {
                pc: 12,
                encoding: ShaderEncoding::Sop2,
                opcode: Opcode::Known("SOrB32"),
                words: vec![0x887c_6a04],
                sources: vec![Operand::scalar(4), Operand::scalar(106)],
                destinations: vec![Operand::scalar(124)],
                control: None,
            },
            Instruction {
                pc: 16,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::Known("SSendmsg"),
                words: vec![0xbf90_0009],
                sources: vec![Operand {
                    kind: OperandKind::EncodedConstant,
                    value: 9,
                }],
                destinations: Vec::new(),
                control: None,
            },
            Instruction {
                pc: 20,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    )
}

#[test]
fn fixed_ngg_fallback_publishes_compacted_vertex_exports_by_compacted_id() {
    // Superliminal Slate UI: PRIM records name compacted ids built behind
    // `V_CMPX_GT_U32 s4, v18`, so its POS/PARAM records must use v18 as well.
    let program = compacted_vertex_exp_program();
    let mut options = MetalCompileOptions::mesh(2, 16 * 1024, 64, 64, 64);
    options.wave_lane_count = 64;
    options.enable_fixed_ngg_allocation_fallback();
    let shader = compile_msl(&program, &empty_evaluation(), &options)
        .expect("the bounded fixed fallback must lower compacted EXP");
    assert!(
        shader
            .source
            .contains("remu_mesh.set_vertex(remu_v_18, remu_out)"),
        "mesh vertices must publish under the compacted vertex id: {}",
        shader.source
    );
    assert!(
        !shader
            .source
            .contains("remu_mesh.set_vertex(remu_thread_index, remu_out)"),
        "the epilogue must not overwrite compacted records by invocation index: {}",
        shader.source
    );
}

#[test]
fn proven_ngg_registers_publish_vertex_exports_by_compacted_id() {
    let program = compacted_vertex_exp_program();
    let mut options = MetalCompileOptions::mesh_from_ngg_registers(
        2,
        64,
        NggRegisters {
            pgm_rsrc1: 0x8,
            pgm_rsrc2: 0x20 << 19,
            gs_max_vert_out: 64,
            ge_max_output_per_subgroup: 64,
            esgs_ring_itemsize: 0,
            primgen_passthrough: false,
        },
    );
    options.wave_lane_count = 64;
    let shader = compile_msl(&program, &empty_evaluation(), &options)
        .expect("proven NggRegisters lower gated EXP through direct Metal records");
    assert!(
        shader
            .source
            .contains("remu_mesh.set_vertex(remu_v_18, remu_out)"),
        "EXP must publish remu_out under the compacted vertex id: {}",
        shader.source
    );
    assert!(
        !shader
            .source
            .contains("remu_mesh.set_vertex(remu_thread_index, remu_out)"),
        "the epilogue must not overwrite compacted records by invocation index"
    );
}

fn compacted_vertex_exp_program() -> Program {
    let export = |pc, target, sources| Instruction {
        pc,
        encoding: ShaderEncoding::Exp,
        opcode: Opcode::Known("Exp"),
        words: vec![0, 0],
        sources,
        destinations: Vec::new(),
        control: Some(InstructionControl::Export(ExportControl {
            target,
            enable_mask: 15,
            compressed: false,
            done: true,
            valid_mask: false,
        })),
    };
    Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Vopc,
                opcode: Opcode::Known("VCmpxGtU32"),
                words: vec![0],
                sources: vec![Operand::scalar(4), Operand::vector(18)],
                destinations: Vec::new(),
                control: None,
            },
            export(
                4,
                12,
                vec![
                    Operand::vector(0),
                    Operand::vector(1),
                    Operand::vector(2),
                    Operand::vector(3),
                ],
            ),
            export(
                12,
                32,
                vec![
                    Operand::vector(4),
                    Operand::vector(5),
                    Operand::vector(6),
                    Operand::vector(7),
                ],
            ),
            Instruction {
                pc: 20,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    )
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

    let missing_stride = compile_msl(
        &linked,
        &empty_evaluation(),
        &MetalCompileOptions::mesh(1, 16 * 1024, 96, 96, 96),
    )
    .expect_err("linked ES/GS must not invent a vertex-item stride");
    assert!(
        missing_stride
            .to_string()
            .contains("missing VGT_ESGS_RING_ITEMSIZE")
    );

    let mut options = MetalCompileOptions::mesh(1, 16 * 1024, 96, 96, 96);
    // The captured fused stage uses four dwords per ES vertex. This must come from
    // VGT_ESGS_RING_ITEMSIZE rather than being hardcoded in lowering.
    options.ngg_esgs_ring_itemsize = 4;
    let shader = compile_msl(&linked, &empty_evaluation(), &options)
        .expect("linked NGG program should lower as one mesh function");

    assert_eq!(shader.stage, MetalShaderStage::Mesh);
    assert!(shader.register_footprint.direct_control_flow);
    assert!(
        shader.source.contains(
            "uint remu_mesh_gs_vtx_offset_0 = ((((remu_mesh_primitive >> 0u) & 0x1ffu) * 4u)"
        ),
        "a linked ES/GS stage must receive packed 16-bit dword offsets in the GFX9-11 GS ABI"
    );
    assert!(
        shader
            .source
            .contains("remu_v_0 = remu_mesh_gs_vtx_offset_0;")
    );
    assert!(
        shader
            .source
            .contains("remu_v_1 = remu_mesh_gs_vtx_offset_1;")
    );
    assert!(
        shader
            .source
            .contains("remu_v_2 = remu_mesh_primitive_base + remu_thread_index;")
    );
    let front_lds_write = shader
        .source
        .find("atomic_store_explicit(remu_lds")
        .expect("front ES LDS write");
    let continuation_abi = shader
        .source
        .find("// GFX9-11 linked GS continuation VGPR ABI")
        .expect("linked GS continuation must restore its hardware VGPR inputs");
    assert!(
        front_lds_write < continuation_abi,
        "the GS VGPR ABI must be restored at the continuation, after the ES overwrites its inputs"
    );
    let continuation = &shader.source[continuation_abi..];
    assert!(continuation.contains("remu_v_0 = remu_mesh_gs_vtx_offset_0;"));
    assert!(continuation.contains("remu_v_1 = remu_mesh_gs_vtx_offset_1;"));
    assert!(continuation.contains("remu_v_2 = remu_mesh_primitive_base + remu_thread_index;"));
    assert!(continuation.contains("remu_v_3 = 0u;"));
    assert!(continuation.contains("remu_v_4 = 0u;"));
    assert!(
        shader
            .source
            .contains("threadgroup atomic_uint remu_lds_storage[4096]")
    );
    assert!(shader.source.contains("remu_mesh.set_index("));
    assert!(!shader.source.contains("remu_mesh_output_vertex_count"));
    assert!(
        !shader
            .source
            .contains("threadgroup atomic_uint remu_mesh_output")
    );
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
        .find("remu_mesh_output_primitive_count = (remu_s_124 >> 12u)")
        .expect("GS_ALLOC_REQ primitive allocation");
    let output_count = shader
        .source
        .find("remu_mesh.set_primitive_count(remu_mesh_output_primitives)")
        .expect("dynamic Metal primitive count");
    assert!(allocation < output_count);
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
fn selects_the_gfx10_ngg_input_vgpr_layout_from_primgen_passthrough() {
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VMovB32"),
                words: vec![0],
                sources: vec![Operand::vector(0)],
                destinations: vec![Operand::vector(16)],
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VMovB32"),
                words: vec![0],
                sources: vec![Operand::vector(1)],
                destinations: vec![Operand::vector(17)],
                control: None,
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VMovB32"),
                words: vec![0],
                sources: vec![Operand::vector(2)],
                destinations: vec![Operand::vector(18)],
                control: None,
            },
            Instruction {
                pc: 12,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    );

    let ordinary = compile_msl(
        &program,
        &empty_evaluation(),
        &MetalCompileOptions::mesh(0, 0, 64, 64, 64),
    )
    .expect("ordinary NGG ABI");
    assert!(
        ordinary
            .source
            .contains("remu_v_0 = remu_mesh_gs_vtx_offset_0;")
    );
    assert!(
        ordinary
            .source
            .contains("remu_v_1 = remu_mesh_gs_vtx_offset_1;")
    );
    assert!(
        ordinary
            .source
            .contains("remu_v_2 = remu_mesh_primitive_base + remu_thread_index;")
    );

    let mut passthrough_options = MetalCompileOptions::mesh(0, 0, 64, 64, 64);
    passthrough_options.ngg_primgen_passthrough = true;
    let passthrough = compile_msl(&program, &empty_evaluation(), &passthrough_options)
        .expect("passthrough NGG ABI");
    assert!(
        passthrough
            .source
            .contains("remu_v_0 = remu_mesh_primitive;")
    );
    assert!(passthrough.source.contains("remu_v_1 = 0u;"));
    assert!(
        passthrough
            .source
            .contains("remu_v_2 = remu_mesh_primitive_base + remu_thread_index;")
    );
    assert!(!passthrough.source.contains("remu_mesh_gs_vtx_offset_0"));
}

#[test]
fn mesh_layer_helpers_count_against_the_guest_lds_limit() {
    let program = decoded_ds_program(0x1234, false);
    let mut options = MetalCompileOptions::mesh(0, 32 * 1024, 256, 256, 64);
    options.render_target_array_length = 2;

    let error = compile_msl(&program, &empty_evaluation(), &options)
        .expect_err("full guest LDS leaves no room for layered mesh helper arrays");
    assert!(error.to_string().contains("compiler-generated state"));
    assert!(error.to_string().contains("34816 bytes"));
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
                    export_format: PixelExportFormat::Abgr32,
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

#[test]
fn mesh_prologue_seeds_gs_tg_info_workgroup_counts_into_s2() {
    // Merged NGG SGPR s2 is gs_tg_info; its [20:12]/[30:22] fields carry the workgroup input
    // vertex/primitive counts (Mesa ac_nir_lower_intrinsics_to_args.c). Non-culling NGG
    // shaders build the GS_ALLOC_REQ M0 payload straight from them, so an unseeded s2 would
    // publish zero primitives and erase the draw.
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::Known("SMovB32"),
                words: vec![0],
                sources: vec![Operand::scalar(2)],
                destinations: vec![Operand::scalar(80)],
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    );
    let mut options = MetalCompileOptions::mesh(1, 16 * 1024, 128, 128, 128);
    options.wave_lane_count = 64;
    let shader = compile_msl(&program, &empty_evaluation(), &options).expect("mesh lowering");
    assert!(
        shader
            .source
            .contains("= ((remu_mesh_input_vertex_count & 0x1ffu) << 12u)"),
        "s2 must carry the workgroup input vertex count in [20:12]: {}",
        shader.source
    );
    assert!(
        shader
            .source
            .contains("((remu_mesh_input_primitive_count & 0x1ffu) << 22u)"),
        "s2 must carry the workgroup input primitive count in [30:22]: {}",
        shader.source
    );
}

#[test]
fn mesh_prim_export_guards_vertex_indices_against_metal_capacity() {
    // A GFX10 primitive word carries 9-bit vertex fields (0..511) while the Metal mesh
    // declares only mesh_max_vertex_count vertices. Feeding a larger index to set_index is
    // undefined behavior in Metal, so such primitives must keep their degenerate prefill.
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Exp,
                opcode: Opcode::Known("Exp"),
                words: vec![0, 0],
                sources: vec![
                    Operand::vector(0),
                    Operand::vector(0),
                    Operand::vector(0),
                    Operand::vector(0),
                ],
                destinations: Vec::new(),
                control: Some(InstructionControl::Export(ExportControl {
                    target: 20,
                    enable_mask: 1,
                    compressed: false,
                    done: false,
                    valid_mask: false,
                })),
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    );
    let mut options = MetalCompileOptions::mesh(1, 16 * 1024, 64, 64, 64);
    options.wave_lane_count = 64;
    let shader = compile_msl(&program, &empty_evaluation(), &options).expect("mesh lowering");
    assert!(
        shader
            .source
            .contains("if (max(max(remu_prim_i0, remu_prim_i1), remu_prim_i2) < 64u) {"),
        "EXP PRIM must skip primitives whose indices exceed the Metal vertex capacity: {}",
        shader.source
    );
}

#[test]
fn mesh_entry_exec_is_the_full_architectural_wave_mask() {
    // Hardware launches a merged NGG wave with the full architectural EXEC and the shader
    // narrows it against merged_wave_info. A ballot over existing Metal threads would zero
    // the tail of a partial final wave before the guest's own narrowing runs.
    let program = Program::new(
        0x1000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::Known("SMovB32"),
                words: vec![0],
                sources: vec![Operand::scalar(126)],
                destinations: vec![Operand::scalar(80)],
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::Known("SMovB32"),
                words: vec![0],
                sources: vec![Operand::scalar(127)],
                destinations: vec![Operand::scalar(81)],
                control: None,
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    );
    let mut options = MetalCompileOptions::mesh(1, 16 * 1024, 96, 96, 96);
    options.wave_lane_count = 64;
    let shader = compile_msl(&program, &empty_evaluation(), &options).expect("mesh lowering");
    assert!(
        shader.source.contains("remu_s_126 = 0xffffffffu;")
            && shader.source.contains("remu_s_127 = 0xffffffffu;"),
        "split-wave64 mesh entry EXEC must be the full 64-lane mask: {}",
        shader.source
    );
}
