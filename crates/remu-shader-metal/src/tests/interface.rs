// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Stage interface lowering tests: pixel/vertex parameter routing, interpolation
//! modes, built-in rectangle shaders and compute system register state.

use super::*;
use remu_shader::ExportControl;

fn compile_pixel_export(kind: PixelOutputKind, compressed: bool) -> MetalShader {
    compile_pixel_export_mask(kind, compressed, 0x0f)
}

fn compile_pixel_export_mask(
    kind: PixelOutputKind,
    compressed: bool,
    enable_mask: u32,
) -> MetalShader {
    // Decode a real GCN EXP instead of constructing compiler IR by hand. This locks the raw
    // VGPR payload and COMPR/EN bit semantics to the generated Metal source.
    let export_word =
        0xf800_0000 | enable_mask | (u32::from(compressed) << 10) | (1 << 11) | (1 << 12);
    let program = decode_program(0x1000, &[export_word, 0x0302_0100, 0xbf81_0000])
        .expect("pixel export should decode");
    let export = program.instructions.first().expect("decoded EXP");
    assert_eq!(export.opcode, Opcode::EXPORT);
    assert_eq!(
        export.control,
        Some(InstructionControl::Export(ExportControl {
            target: 0,
            enable_mask,
            compressed,
            done: true,
            valid_mask: true,
        }))
    );
    let options = MetalCompileOptions::pixel(
        vec![PixelOutputBinding {
            guest_slot: 0,
            host_location: 0,
            kind,
        }],
        0,
        0,
    );
    compile_msl(&program, &empty_evaluation(), &options).expect("pixel export should lower")
}

#[test]
fn compressed_pixel_export_recovers_coverage_when_guest_packs_zero_alpha() {
    // Poppy glyph PS packs compressed MRT0 as (R,G,B,0). Emit coverage recovery for A==0.
    let shader = compile_pixel_export(PixelOutputKind::Float, true);
    assert!(
        shader.source.contains(
            "if (exec && remu_out.mrt0.a == 0.0f) remu_out.mrt0.a = max(max(fabs(remu_out.mrt0.r), fabs(remu_out.mrt0.g)), fabs(remu_out.mrt0.b));"
        ),
        "expected premul coverage recovery for A=0 compressed export:\n{}",
        shader.source
    );
}

#[test]
fn routes_rdna2_mrtz_component_zero_to_fragment_depth() {
    // AMD RDNA2 EXP target 8 is Z. Silent Hill's captured gameplay shaders use EN=1,
    // uncompressed, with VSRC0 carrying the replacement depth.
    let export_word = 0xf800_0000 | 1 | (8 << 4) | (1 << 11);
    let program = decode_program(0x1000, &[export_word, 0x0302_0100, 0xbf81_0000])
        .expect("MRTZ export should decode");
    let mut options = MetalCompileOptions::pixel(Vec::new(), 0, 0);
    options.pixel_depth_export = true;
    let shader =
        compile_msl(&program, &empty_evaluation(), &options).expect("MRTZ depth should lower");

    assert!(shader.source.contains("float depth [[depth(any)]];"));
    assert!(shader.source.contains("fragment RemuPsOut remu_main"));
    assert!(
        shader
            .source
            .contains("remu_out.depth = remu_in.frag_coord.z;")
    );
    assert!(
        shader
            .source
            .contains("if (exec) remu_out.depth = as_type<float>(remu_v_0);")
    );
    assert!(shader.source.contains("return remu_out;"));
}

#[test]
fn leaves_mrtz_inert_when_depth_export_pipeline_state_is_disabled() {
    let export_word = 0xf800_0000 | 1 | (8 << 4) | (1 << 11);
    let program = decode_program(0x1000, &[export_word, 0x0302_0100, 0xbf81_0000])
        .expect("MRTZ export should decode");
    let options = MetalCompileOptions::pixel(Vec::new(), 0, 0);
    let shader =
        compile_msl(&program, &empty_evaluation(), &options).expect("disabled MRTZ should lower");

    assert!(shader.source.contains("fragment void remu_main"));
    assert!(!shader.source.contains("[[depth("));
}

#[test]
fn masked_pixel_export_components_use_gfx_blend_defaults() {
    let float = compile_pixel_export_mask(PixelOutputKind::Float, false, 0x07);
    assert!(
        float
            .source
            .contains("remu_out.mrt0 = float4(0.0f, 0.0f, 0.0f, 1.0f);")
    );
    assert!(float.source.contains(
        "remu_out.mrt0 = float4(as_type<float>(remu_v_0), as_type<float>(remu_v_1), as_type<float>(remu_v_2), remu_out.mrt0[3]);"
    ));

    let unsigned = compile_pixel_export_mask(PixelOutputKind::Uint, false, 0x07);
    assert!(
        unsigned
            .source
            .contains("remu_out.mrt0 = uint4(0u, 0u, 0u, 1u);")
    );

    let signed = compile_pixel_export_mask(PixelOutputKind::Sint, false, 0x07);
    assert!(signed.source.contains("remu_out.mrt0 = int4(0, 0, 0, 1);"));
}

#[test]
fn preserves_integer_mrt_export_bits_without_float_conversion() {
    let unsigned = compile_pixel_export(PixelOutputKind::Uint, false);
    assert!(
        unsigned
            .source
            .contains("remu_out.mrt0 = uint4(remu_v_0, remu_v_1, remu_v_2, remu_v_3);")
    );
    assert!(!unsigned.source.contains("(uint)(as_type<float>"));

    let signed = compile_pixel_export(PixelOutputKind::Sint, false);
    assert!(signed.source.contains(
        "remu_out.mrt0 = int4(as_type<int>(remu_v_0), as_type<int>(remu_v_1), as_type<int>(remu_v_2), as_type<int>(remu_v_3));"
    ));
    assert!(!signed.source.contains("(int)(as_type<float>"));
}

#[test]
fn unpacks_compressed_integer_mrt_exports_as_integer_halves() {
    // COMPR uses EN[0] for R/G and EN[2] for B/A, so 0b0101 is the canonical all-component mask.
    let unsigned = compile_pixel_export_mask(PixelOutputKind::Uint, true, 0x05);
    assert!(unsigned.source.contains(
        "remu_out.mrt0 = uint4((uint)as_type<ushort2>(remu_v_0)[0], (uint)as_type<ushort2>(remu_v_0)[1], (uint)as_type<ushort2>(remu_v_1)[0], (uint)as_type<ushort2>(remu_v_1)[1]);"
    ));
    assert!(!unsigned.source.contains("as_type<float>(remu_v_"));

    let signed = compile_pixel_export_mask(PixelOutputKind::Sint, true, 0x05);
    assert!(signed.source.contains(
        "remu_out.mrt0 = int4((int)as_type<short2>(remu_v_0)[0], (int)as_type<short2>(remu_v_0)[1], (int)as_type<short2>(remu_v_1)[0], (int)as_type<short2>(remu_v_1)[1]);"
    ));
    assert!(!signed.source.contains("as_type<float>(remu_v_"));
}

#[test]
fn compressed_integer_mrt_pair_enables_preserve_the_other_pair() {
    let rg = compile_pixel_export_mask(PixelOutputKind::Uint, true, 0x01);
    assert!(rg.source.contains(
        "remu_out.mrt0 = uint4((uint)as_type<ushort2>(remu_v_0)[0], (uint)as_type<ushort2>(remu_v_0)[1], remu_out.mrt0[2], remu_out.mrt0[3]);"
    ));

    let ba = compile_pixel_export_mask(PixelOutputKind::Sint, true, 0x04);
    assert!(ba.source.contains(
        "remu_out.mrt0 = int4(remu_out.mrt0[0], remu_out.mrt0[1], (int)as_type<short2>(remu_v_1)[0], (int)as_type<short2>(remu_v_1)[1]);"
    ));
}

#[test]
fn maps_packed_secondary_position_exports_to_metal_clip_distances() {
    let export = |pc, target| Instruction {
        pc,
        encoding: ShaderEncoding::Exp,
        opcode: Opcode::Known("Exp"),
        words: vec![0, 0],
        sources: (0..4).map(Operand::vector).collect(),
        destinations: Vec::new(),
        control: Some(InstructionControl::Export(ExportControl {
            target,
            enable_mask: 0x0f,
            compressed: false,
            done: target == 14,
            valid_mask: false,
        })),
    };
    let program = Program::new(0x1000, vec![export(0, 13), export(8, 14)]);
    let mut options = MetalCompileOptions::vertex(0);
    // Misc occupies POS1; clip/cull-distance vector 0 is therefore packed into POS2. Enable clip
    // distance 0 and cull distance 2. Metal has no cull-distance output, so only the true clip
    // distance is emitted.
    options.vertex_output_control = (1 << 21) | (1 << 22) | (1 << 0) | (1 << (8 + 2));
    let shader =
        compile_msl(&program, &empty_evaluation(), &options).expect("clip exports should lower");

    assert!(
        shader
            .source
            .contains("float clip_distances [[clip_distance]] [8];")
    );
    assert!(
        shader
            .source
            .contains("remu_out.clip_distances[0] = as_type<float>(remu_v_0);")
    );
    assert!(!shader.source.contains("remu_out.clip_distances[2] ="));
    assert!(!shader.source.contains("remu_out.clip_distances[1] ="));
}

#[test]
fn emits_instance_routing_for_layered_render_targets() {
    let program = decode_program(0x1000, &[0xbf81_0000]).expect("minimal vertex program decodes");
    let mut options = MetalCompileOptions::vertex(0);
    options.render_target_array_length = 64;
    let shader =
        compile_msl(&program, &empty_evaluation(), &options).expect("layered vertex lowers");

    assert!(
        shader
            .source
            .contains("uint render_target_array_index [[render_target_array_index]]")
    );
    assert!(
        shader
            .source
            .contains("remu_out.render_target_array_index = min(remu_instance_id, 63u);")
    );
}

#[test]
fn emits_guest_misc_layer_as_mesh_primitive_output() {
    let export = |pc, target, enable_mask| Instruction {
        pc,
        encoding: ShaderEncoding::Exp,
        opcode: Opcode::Known("Exp"),
        words: vec![0, 0],
        sources: (0..4).map(Operand::vector).collect(),
        destinations: Vec::new(),
        control: Some(InstructionControl::Export(ExportControl {
            target,
            enable_mask,
            compressed: false,
            done: target == 20,
            valid_mask: false,
        })),
    };
    let program = Program::new(
        0x1000,
        vec![
            export(0, 13, 1 << 2),
            export(8, 12, 0x0f),
            export(16, 20, 1),
        ],
    );
    let mut options = MetalCompileOptions::mesh(0, 0, 64, 64, 64);
    options.vertex_output_control = (1 << 18) | (1 << 21);
    options.render_target_array_length = 6;
    let shader = compile_msl(&program, &empty_evaluation(), &options).expect("layered mesh lowers");

    assert!(shader.source.contains(
        "struct RemuPrimitiveOut {\n    uint render_target_array_index [[render_target_array_index]];"
    ));
    assert!(
        shader
            .source
            .contains("mesh<RemuVsOut, RemuPrimitiveOut, 64, 64, topology::triangle> remu_mesh")
    );
    assert!(
        shader
            .source
            .contains("threadgroup uint remu_mesh_render_target_indices[64];")
    );
    assert!(
        shader
            .source
            .contains("min(as_type<uint>(as_type<float>(remu_v_2)), 5u)")
    );
    assert!(shader.source.contains(
        "remu_primitive_out.render_target_array_index = remu_mesh_render_target_indices[remu_primitive_source_vertex];"
    ));
    assert!(
        !shader
            .source
            .contains("float4 position [[position]];\n    uint render_target_array_index"),
        "Metal rejects render_target_array_index on a mesh vertex output"
    );

    #[cfg(target_os = "macos")]
    if let Some(compiler) = std::env::var_os("REMU_METAL_COMPILER") {
        let directory =
            std::env::temp_dir().join(format!("remu-layered-mesh-metal-{}", std::process::id()));
        std::fs::create_dir_all(&directory).expect("create layered mesh Metal test directory");
        let source = directory.join("mesh.metal");
        let air = directory.join("mesh.air");
        std::fs::write(&source, &shader.source).expect("write generated layered mesh MSL");
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
            "layered mesh MSL failed to compile:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        std::fs::remove_dir_all(directory).expect("remove layered mesh Metal test directory");
    }
}

#[test]
fn builds_fused_es_gs_rectangle_equivalent_for_volume_generation() {
    let shader = compile_layered_rectangle_vertex(2, 64);

    assert_eq!(shader.stage, MetalShaderStage::Vertex);
    assert_eq!(shader.interpolated_attribute_count, 2);
    assert!(
        shader
            .source
            .contains("uint render_target_array_index [[render_target_array_index]]")
    );
    assert!(
        shader
            .source
            .contains("remu_out.render_target_array_index = min(remu_instance_id, 63u);")
    );
    assert!(
        shader
            .source
            .contains("float remu_layer = (float(remu_instance_id) + 0.5f) / 64.0f;")
    );
    assert!(
        shader
            .source
            .contains("remu_out.param1 = float4(remu_uv.x, 1.0f - remu_uv.y, remu_layer, 1.0f);")
    );
}

#[test]
fn builds_procedural_rectangle_from_pixel_consumer_parameter_roles() {
    let shader = compile_procedural_rectangle_vertex(4, 0b1011);

    assert_eq!(shader.stage, MetalShaderStage::Vertex);
    assert_eq!(shader.interpolated_attribute_count, 4);
    assert!(
        shader
            .source
            .contains("float2(-1.0f, -1.0f)")
            && shader.source.contains("float2( 3.0f, -1.0f)")
            && shader.source.contains("float2(-1.0f,  3.0f)")
    );
    assert!(
        shader
            .source
            .contains("remu_out.position = float4(remu_positions[remu_index], 0.0f, 1.0f);")
    );
    assert!(
        shader
            .source
            .contains("remu_out.param0 = float4(remu_uv.x, 1.0f - remu_uv.y, 1.0f, 1.0f);")
    );
    assert!(
        shader
            .source
            .contains("remu_out.param1 = float4(remu_uv.x, 1.0f - remu_uv.y, 1.0f, 1.0f);")
    );
    assert!(shader.source.contains("remu_out.param2 = float4(1.0f);"));
    assert!(
        shader
            .source
            .contains("remu_out.param3 = float4(remu_uv.x, 1.0f - remu_uv.y, 1.0f, 1.0f);")
    );
}

fn compile_pixel_system_input_shader(
    referenced_vgpr: u32,
    pixel_input_enable: u32,
    pixel_input_address: u32,
    render_target_array_length: u32,
) -> MetalShader {
    let program = Program::new(
        0x1000,
        vec![Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop1,
            opcode: Opcode::Known("VMovB32"),
            words: vec![0],
            sources: vec![Operand::vector(referenced_vgpr)],
            destinations: vec![Operand::vector(8)],
            control: None,
        }],
    );
    let mut options =
        MetalCompileOptions::pixel(Vec::new(), pixel_input_enable, pixel_input_address);
    options.render_target_array_length = render_target_array_length;
    compile_msl(&program, &empty_evaluation(), &options).expect("pixel system input should lower")
}

#[test]
fn routes_layered_render_target_index_into_ancillary_vgpr() {
    let addressed = (1 << 0) | (1 << 13);
    let shader = compile_pixel_system_input_shader(2, addressed, addressed, 32);

    assert!(
        shader
            .source
            .contains("uint render_target_array_index [[render_target_array_index]];")
    );
    assert!(
        shader
            .source
            .contains("remu_v_2 = remu_in.render_target_array_index << 16u;")
    );
}

#[test]
fn disabled_system_inputs_still_reserve_compacted_pixel_vgprs() {
    let address = (1 << 0) | (1 << 12) | (1 << 13);
    let enable = (1 << 0) | (1 << 13);
    let shader = compile_pixel_system_input_shader(3, enable, address, 32);

    assert!(
        shader
            .source
            .contains("remu_v_3 = remu_in.render_target_array_index << 16u;")
    );
    assert!(
        !shader
            .source
            .contains("remu_v_2 = remu_in.render_target_array_index << 16u;")
    );
}

#[test]
fn omits_layer_index_input_for_single_layer_pixel_shader() {
    let addressed = (1 << 0) | (1 << 13);
    let shader = compile_pixel_system_input_shader(2, addressed, addressed, 1);

    assert!(
        !shader
            .source
            .contains("uint render_target_array_index [[render_target_array_index]];")
    );
    assert!(!shader.source.contains("remu_in.render_target_array_index"));
}

#[test]
fn materializes_pixel_position_into_the_named_vgpr() {
    let shader = compile_pixel_system_input_shader(0, 1 << 8, 1 << 8, 1);

    assert!(
        shader
            .source
            .contains("remu_v_0 = as_type<uint>(remu_in.frag_coord.x);")
    );
    assert!(!shader.source.contains("    v[0] ="));
}

fn interpolation_instruction(
    pc: u32,
    attribute: u32,
    channel: u32,
    barycentric_register: u32,
    destination: u32,
) -> Instruction {
    Instruction {
        pc,
        encoding: ShaderEncoding::Vintrp,
        opcode: Opcode::Known("VInterpP2F32"),
        words: vec![0],
        sources: vec![Operand::vector(barycentric_register)],
        destinations: vec![Operand::vector(destination)],
        control: Some(InstructionControl::Interpolation(
            remu_shader::InterpolationControl { attribute, channel },
        )),
    }
}

fn compile_interpolation_shader(
    instructions: Vec<Instruction>,
    pixel_input_address: u32,
    pixel_input_controls: [PixelInputControl; PIXEL_INPUT_CONTROL_COUNT],
) -> MetalShader {
    let mut options =
        MetalCompileOptions::pixel(Vec::new(), pixel_input_address, pixel_input_address);
    options.pixel_input_controls = pixel_input_controls;
    compile_msl(
        &Program::new(0x1000, instructions),
        &empty_evaluation(),
        &options,
    )
    .expect("pixel interpolation should lower")
}

#[test]
fn maps_logical_pixel_input_to_physical_vertex_parameter() {
    let mut controls = identity_pixel_input_controls();
    controls[0] = PixelInputControl::from_register(4);
    assert_eq!(
        controls[0],
        PixelInputControl {
            input_offset: 4,
            use_default: false,
            default_value: 0,
            flat_shade: false,
        }
    );
    assert_eq!(controls[0].register_value(), 4);

    let pixel = compile_interpolation_shader(
        vec![interpolation_instruction(0, 0, 0, 1, 8)],
        1 << 1,
        controls,
    );
    assert!(
        pixel
            .source
            .contains("float4 attr0 [[user(locn4), center_perspective]];")
    );
    assert!(pixel.source.contains("as_type<uint>(remu_in.attr0[0])"));
    assert_eq!(pixel.interpolated_attribute_count, 5);

    let vertex_program =
        decode_program(0x1000, &[0xbf81_0000]).expect("minimal vertex program decodes");
    let vertex = compile_msl(
        &vertex_program,
        &empty_evaluation(),
        &MetalCompileOptions::vertex(pixel.interpolated_attribute_count),
    )
    .expect("mapped vertex interface should lower");
    assert!(vertex.source.contains("float4 param4 [[user(locn4)]];"));
}

#[test]
fn materializes_architectural_pixel_input_default_two() {
    let mut controls = identity_pixel_input_controls();
    controls[0] = PixelInputControl::from_register((1 << 5) | (2 << 8));
    let pixel = compile_interpolation_shader(
        vec![
            interpolation_instruction(0, 0, 2, 1, 8),
            interpolation_instruction(4, 0, 3, 1, 9),
        ],
        1 << 1,
        controls,
    );

    assert!(!pixel.source.contains("float4 attr0"));
    assert!(pixel.source.contains("remu_v_8 = 0x3F800000u;"));
    assert!(pixel.source.contains("remu_v_9 = 0u;"));
    assert_eq!(pixel.interpolated_attribute_count, 0);
}

#[test]
fn emits_flat_interpolation_from_pixel_input_control() {
    let mut controls = identity_pixel_input_controls();
    controls[0] = PixelInputControl::from_register(3 | (1 << 10));
    let pixel = compile_interpolation_shader(
        vec![interpolation_instruction(0, 0, 0, 1, 8)],
        1 << 1,
        controls,
    );

    assert!(pixel.source.contains("float4 attr0 [[user(locn3), flat]];"));
    assert_eq!(pixel.interpolated_attribute_count, 4);
}

#[test]
fn derives_all_metal_interpolation_modes_from_p2_barycentric_vgpr() {
    let pixel = compile_interpolation_shader(
        [(0, 1), (1, 3), (2, 5), (3, 7), (4, 9), (5, 11)]
            .into_iter()
            .enumerate()
            .map(|(index, (attribute, barycentric_register))| {
                interpolation_instruction(
                    u32::try_from(index * 4).expect("test PC fits u32"),
                    attribute,
                    0,
                    barycentric_register,
                    16 + attribute,
                )
            })
            .collect(),
        0x77,
        identity_pixel_input_controls(),
    );

    for (attribute, qualifier) in [
        (0, "sample_perspective"),
        (1, "center_perspective"),
        (2, "centroid_perspective"),
        (3, "sample_no_perspective"),
        (4, "center_no_perspective"),
        (5, "centroid_no_perspective"),
    ] {
        assert!(
            pixel.source.contains(&format!(
                "float4 attr{attribute} [[user(locn{attribute}), {qualifier}]];"
            )),
            "missing {qualifier} for attr{attribute}"
        );
    }
}

#[test]
fn outputless_pixel_shader_lowers_to_void_depth_fragment() {
    let shader = compile_vector_instruction(
        Instruction {
            pc: 0,
            encoding: ShaderEncoding::Vop1,
            opcode: Opcode::Known("VMovB32"),
            words: vec![0],
            sources: vec![Operand::vector(1)],
            destinations: vec![Operand::vector(2)],
            control: None,
        },
        MetalShaderStage::Pixel,
    );

    assert!(shader.source.contains("fragment void remu_main"));
    assert!(!shader.source.contains("RemuPsOut remu_out"));
    assert!(!shader.source.contains("return remu_out"));
}

#[test]
fn initializes_complete_rdna2_compute_system_register_state() {
    let program = Program::new(
        0x1000,
        [4_u32, 5, 6, 7]
            .into_iter()
            .enumerate()
            .map(|(index, source)| Instruction {
                pc: u32::try_from(index * 4).expect("small system-register program"),
                encoding: ShaderEncoding::Sop1,
                opcode: Opcode::Known("SMovB32"),
                words: vec![0],
                sources: vec![Operand::scalar(source)],
                destinations: vec![Operand::scalar(20 + source)],
                control: None,
            })
            .chain([
                Instruction {
                    pc: 16,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![Operand::vector(0)],
                    destinations: vec![Operand::vector(20)],
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
            ])
            .collect(),
    );
    let mut evaluation = empty_evaluation();
    evaluation.compute_system_registers = Some(ComputeSystemRegisters {
        work_group_x: Some(4),
        work_group_y: Some(5),
        work_group_z: Some(6),
        thread_group_size: Some(7),
    });

    let shader = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
        .expect("compute system register lowering");

    assert!(shader.source.contains("remu_v_0 = remu_local_id.x"));
    assert!(shader.source.contains("remu_s_4 = remu_group_id.x;"));
    assert!(shader.source.contains("remu_s_5 = remu_group_id.y;"));
    assert!(shader.source.contains("remu_s_6 = remu_group_id.z;"));
    assert!(
        shader
            .source
            .contains("uint remu_thread_index [[thread_index_in_threadgroup]]")
    );
    assert!(
        shader
            .source
            .contains("uint3 remu_group_size [[threads_per_threadgroup]]")
    );
    assert!(
        shader
            .source
            .contains("(remu_wave_id == 0u ? 0x80000000u : 0u)")
    );
    assert!(shader.source.contains("((remu_wave_id & 31u) << 20u)"));
    assert!(shader.source.contains("(remu_threadgroup_waves & 63u)"));

    let wave64 = compile_msl(
        &program,
        &evaluation,
        &MetalCompileOptions::compute_wave(0, 64),
    )
    .expect("wave64 compute system register lowering");
    assert!(wave64.source.contains("remu_thread_index >> 6u"));
    assert!(
        wave64
            .source
            .contains("(remu_threadgroup_threads + 63u) >> 6u")
    );
    assert!(
        !wave64
            .source
            .contains("uint remu_wave_id = remu_thread_index >> 5u;"),
        "TG_SIZE must count guest wave64s rather than host SIMD32 halves"
    );
}

#[test]
fn rejects_compute_system_registers_outside_compute_and_sgpr_bounds() {
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
    let mut evaluation = empty_evaluation();
    evaluation.compute_system_registers = Some(ComputeSystemRegisters {
        work_group_x: Some(128),
        ..ComputeSystemRegisters::default()
    });
    let error = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
        .expect_err("out-of-range system SGPR must be rejected");
    assert!(
        error
            .to_string()
            .contains("exceeds the scalar register file")
    );

    evaluation.compute_system_registers = Some(ComputeSystemRegisters {
        work_group_x: Some(4),
        ..ComputeSystemRegisters::default()
    });
    let error = compile_msl(&program, &evaluation, &MetalCompileOptions::vertex(0))
        .expect_err("graphics stage must reject compute system SGPRs");
    assert!(error.to_string().contains("require a compute shader stage"));
}
