// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Image lowering tests: sampled and storage textures, descriptor component
//! selection, gathers and depth comparison sampling.

use remu_shader::image_sample_modifiers;

use super::*;

#[test]
fn sampled_images_apply_dynamic_descriptor_component_selection_in_the_shader() {
    let control = ImageControl {
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
        unorm: false,
    };
    let instruction = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageSample"),
        words: vec![0, 0],
        sources: vec![Operand::vector(0), Operand::vector(1), Operand::scalar(0)],
        destinations: (4..8).map(Operand::vector).collect(),
        control: Some(InstructionControl::Image(control.clone())),
    };
    let mut evaluation = empty_evaluation();
    let mut resource_descriptor = vec![0; 8];
    // R32F sampled as R,R,R,R, the live Superliminal descriptor that previously failed
    // because the host tried to materialize the swizzle by changing the texture format.
    resource_descriptor[3] = 0x924;
    evaluation.image_bindings.push(ImageBinding { descriptor_index_is_runtime: false,
        pc: 0,
        opcode: Opcode::Known("ImageSample"),
        control,
        resource_descriptor: resource_descriptor.into(),
        sampler_descriptor: vec![0; 4].into(),
        mip_level: None,
    });
    let shader = compile_msl(
        &Program::new(0x1000, vec![instruction]),
        &evaluation,
        &MetalCompileOptions::compute(0),
    )
    .expect("sampled R32 swizzle should lower");
    assert!(shader.source.contains("remu_s_3 & 0xFFFu"));
    assert!(shader.source.contains("== 0u ? 0xFACu"));
    assert!(shader.source.contains(">= 4u &&"));
    assert!(shader.source.contains("<= 7u ?"));
    assert!(shader.source.contains("tex0.sample("));
}

#[test]
fn a8_font_atlas_select_reads_word_three_of_its_own_descriptor_window() {
    // Poppy Playtime's Slate/UMG font atlas: 1024x1024 R8Unorm data whose T# routes red into
    // ALPHA and zeroes RGB (`destination_select = 0x800`, the PF_A8 pattern). Every other traced
    // Poppy image carries the native 0x204. If this select is lost the shader falls back to the
    // 0xFAC identity, an R8 texture samples as (r, 0, 0, 1), alpha is 1.0 across the quad, and
    // each glyph renders as a solid opaque block. Pin the two things that carry it: the select is
    // read from word three of the descriptor's *own* SGPR window, and the per-channel routing
    // stays dynamic so one MSL function serves every atlas.
    let control = ImageControl {
        dmask: 0xf,
        vector_address: 0,
        address_registers: vec![0, 1],
        vector_data: 4,
        // Deliberately not zero: the emitted register index must follow the binding, not s3.
        scalar_resource: 12,
        scalar_sampler: 8,
        dimension: 1,
        is_array: false,
        glc: false,
        slc: false,
        a16: false,
        d16: false,
        unorm: false,
    };
    let instruction = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageSample"),
        words: vec![0, 0],
        sources: vec![Operand::vector(0), Operand::vector(1), Operand::scalar(12)],
        destinations: (4..8).map(Operand::vector).collect(),
        control: Some(InstructionControl::Image(control.clone())),
    };
    let mut evaluation = empty_evaluation();
    // The live descriptor captured from the guest, verbatim.
    let resource_descriptor = vec![
        0x912f_2a00,
        0xc010_0000,
        0x00ff_c0ff,
        0x9090_0800,
        0,
        0x0070_0000,
        0,
        0,
    ];
    evaluation.initial_scalar_registers[15] = resource_descriptor[3];
    evaluation.scalar_registers[15] = resource_descriptor[3];
    evaluation.image_bindings.push(ImageBinding { descriptor_index_is_runtime: false,
        pc: 0,
        opcode: Opcode::Known("ImageSample"),
        control,
        resource_descriptor: resource_descriptor.into(),
        sampler_descriptor: vec![0; 4].into(),
        mip_level: None,
    });

    let shader = compile_msl(
        &Program::new(0x1000, vec![instruction]),
        &evaluation,
        &MetalCompileOptions::compute(0),
    )
    .expect("A8 font atlas swizzle should lower");

    // `scalar_resource + 3`, seeded from the host's per-draw scalar metadata buffer.
    assert!(
        shader
            .source
            .contains("uint remu_s_15 = remu_initial_s[15];"),
        "descriptor word three must be seeded from the scalar metadata buffer"
    );
    assert!(shader.source.contains("remu_s_15 & 0xFFFu"));
    assert!(shader.source.contains("== 0u ? 0xFACu"));
    // 0x800 decodes to X=0, Y=0, Z=0, W=R, so all four lanes must be extracted from the same
    // word and the alpha lane must be able to reach a texel component.
    for shift in [0_u32, 3, 6, 9] {
        assert!(
            shader.source.contains(&format!(">> {shift}u) & 7u")),
            "missing destination-select lane at shift {shift}"
        );
        assert!(
            shader.source.contains(&format!(">> {shift}u) & 7u) - 4u")),
            "destination-select lane at shift {shift} cannot select a texel component"
        );
    }
    // The swizzle must stay dynamic: baking 0x800 in would recompile per atlas descriptor.
    assert!(!shader.source.contains("0x800"));
}

#[test]
fn zero_lod_gather_returns_four_neighbor_values_from_the_selected_component() {
    let control = ImageControl {
        dmask: 4,
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
        unorm: false,
    };
    let instruction = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageGather4Lz"),
        words: vec![0, 0],
        sources: vec![Operand::vector(0), Operand::vector(1), Operand::scalar(0)],
        destinations: (4..8).map(Operand::vector).collect(),
        control: Some(InstructionControl::Image(control.clone())),
    };
    let mut evaluation = empty_evaluation();
    let mut resource_descriptor = vec![0; 8];
    resource_descriptor[3] = 0xfac | (9 << 28);
    evaluation.image_bindings.push(ImageBinding { descriptor_index_is_runtime: false,
        pc: 0,
        opcode: Opcode::Known("ImageGather4Lz"),
        control,
        resource_descriptor: resource_descriptor.into(),
        sampler_descriptor: vec![0; 4].into(),
        mip_level: Some(0),
    });

    let shader = compile_msl(
        &Program::new(0x1000, vec![instruction]),
        &evaluation,
        &MetalCompileOptions::compute(0),
    )
    .expect("zero-LOD image gather should lower");

    for component in ["x", "y", "z", "w"] {
        assert!(
            shader
                .source
                .contains(&format!("int2(0), component::{component}")),
            "missing physical component gather {component}"
        );
    }
    assert!(shader.source.contains("remu_s_3 & 0xFFFu"));
    assert!(shader.source.contains("remu_v_4 = "));
    assert!(shader.source.contains("remu_v_7 = "));
}

fn compile_offset_sample(name: &'static str, dimension: u32, texture_type: u32) -> MetalShader {
    try_compile_sample(name, dimension, texture_type)
        .unwrap_or_else(|error| panic!("{name} should lower: {error}"))
}

/// Builds a MIMG sample whose VADDR window matches the opcode's exact operand lattice: offset,
/// bias, depth reference, gradients, coordinates, then the trailing LOD-or-clamp dword.
fn try_compile_sample(
    name: &'static str,
    dimension: u32,
    texture_type: u32,
) -> Result<MetalShader, MetalCompileError> {
    let modifiers = image_sample_modifiers(Opcode::Known(name)).expect("known sample opcode");
    let coordinate_count = if matches!(dimension, 2 | 3 | 5) { 3 } else { 2 };
    // Volume and cube gradients carry a third derivative lane; a 2D-array address does not.
    let gradient_count = if modifiers.gradients {
        if matches!(dimension, 2 | 3) { 6 } else { 4 }
    } else {
        0
    };
    let address_count = usize::from(modifiers.offset)
        + usize::from(modifiers.bias)
        + usize::from(modifiers.compare)
        + gradient_count
        + coordinate_count
        + usize::from(modifiers.lod || modifiers.lod_clamp);
    let control = ImageControl {
        dmask: 1,
        vector_address: 0,
        address_registers: (0..u32::try_from(address_count).expect("address count fits u32"))
            .collect(),
        vector_data: 8,
        scalar_resource: 0,
        scalar_sampler: 8,
        dimension,
        is_array: matches!(dimension, 4 | 5 | 7),
        glc: false,
        slc: false,
        a16: false,
        d16: false,
        unorm: false,
    };
    let instruction = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known(name),
        words: vec![0, 0],
        sources: (0..u32::try_from(address_count).expect("address count fits u32"))
            .map(Operand::vector)
            .chain([Operand::scalar(0), Operand::scalar(8)])
            .collect(),
        destinations: vec![Operand::vector(8)],
        control: Some(InstructionControl::Image(control.clone())),
    };
    let mut evaluation = empty_evaluation();
    let mut resource_descriptor = vec![0; 8];
    resource_descriptor[3] = 0x204 | (texture_type << 28);
    evaluation.image_bindings.push(ImageBinding { descriptor_index_is_runtime: false,
        pc: 0,
        opcode: Opcode::Known(name),
        control,
        resource_descriptor: resource_descriptor.into(),
        sampler_descriptor: vec![3 << 12, 0, 0, 0].into(),
        mip_level: None,
    });
    compile_msl(
        &Program::new(0x1000, vec![instruction]),
        &evaluation,
        &MetalCompileOptions::compute(0),
    )
}

#[test]
fn lod_clamp_samples_are_not_depth_comparisons() {
    // IMAGE_SAMPLE_CL (OP 0x21) carries a trailing minimum-LOD clamp, the operand SPIR-V spells
    // MinLod. Reading its mnemonic as a comparison declared a depth texture, consumed the first
    // coordinate dword as a depth reference and emitted sample_compare.
    let shader = compile_offset_sample("ImageSampleCl", 1, 9);

    assert!(shader.source.contains("texture2d<float> tex0"));
    assert!(!shader.source.contains("depth2d"));
    assert!(!shader.source.contains("sample_compare"));
    assert!(shader.source.contains("min_lod_clamp("));
    // Two coordinate dwords then the clamp: the coordinates keep v0/v1.
    assert!(
        shader
            .source
            .contains("float2(as_type<float>(remu_v_0), as_type<float>(remu_v_1))")
    );
    assert!(shader.source.contains("min_lod_clamp(t"));
}

#[test]
fn coarse_derivative_samples_lower_as_gradients_not_comparisons() {
    // IMAGE_SAMPLE_CD (OP 0x68) is a coarse-derivative gradient sample. Only the `CCd` forms
    // compare.
    let ordinary = compile_offset_sample("ImageSampleCd", 1, 9);
    assert!(ordinary.source.contains("texture2d<float> tex0"));
    assert!(!ordinary.source.contains("sample_compare"));
    assert!(ordinary.source.contains("gradient2d("));

    let comparison = compile_offset_sample("ImageSampleCCd", 1, 9);
    assert!(comparison.source.contains("depth2d<float> tex0"));
    assert!(comparison.source.contains("sample_compare("));
    assert!(comparison.source.contains("gradient2d("));
}

#[test]
fn comparison_samples_keep_their_bias_lod_and_gradient_operands() {
    // The compare-plus-modifier forms previously lost their modifier entirely, so each parsed its
    // coordinates out of the bias, LOD or gradient dwords.
    let explicit_lod = compile_offset_sample("ImageSampleCL", 1, 9);
    assert!(explicit_lod.source.contains("sample_compare("));
    // A temporary, not the `level(0.0f)` an `Lz` opcode would emit.
    assert!(explicit_lod.source.contains("level(t"));

    let bias = compile_offset_sample("ImageSampleCB", 1, 9);
    assert!(bias.source.contains("sample_compare("));
    assert!(bias.source.contains("bias(t"));

    let gradients = compile_offset_sample("ImageSampleCD", 1, 9);
    assert!(gradients.source.contains("sample_compare("));
    assert!(gradients.source.contains("gradient2d("));
}

#[test]
fn refuses_the_lod_clamp_forms_metal_cannot_express() {
    // A Metal gather takes no LOD options at all, and sample_compare takes exactly one, so these
    // clamps have no lowering. Refusing names the missing operand instead of silently dropping it.
    for name in ["ImageGather4Cl", "ImageGather4ClO", "ImageGather4CCl"] {
        let error = try_compile_sample(name, 1, 9).expect_err("clamped gather has no lowering");
        assert!(
            format!("{error}").contains("minimum-LOD clamp"),
            "{name} refused with {error}"
        );
    }
    for name in ["ImageSampleCBCl", "ImageSampleCDCl", "ImageSampleCCdClO"] {
        let error = try_compile_sample(name, 1, 9).expect_err("clamped comparison has no lowering");
        assert!(
            format!("{error}").contains("minimum-LOD clamp"),
            "{name} refused with {error}"
        );
    }
}

#[test]
fn zero_lod_sample_offset_uses_the_native_metal_integer_operand() {
    let shader = compile_offset_sample("ImageSampleLzO", 1, 9);

    assert!(shader.source.contains("texture2d<float> tex0"));
    assert!(shader.source.contains(".sample("));
    assert!(shader.source.contains("level(0.0f), int2("));
    assert!(shader.source.contains("extract_bits("));
    assert!(shader.source.contains(", 0u, 6u)"));
    assert!(shader.source.contains(", 8u, 6u)"));
    assert!(!shader.source.contains("tex0.get_width("));
    assert!(!shader.source.contains("tex0.get_height("));
}

#[test]
fn zero_lod_gather_offset_reaches_every_native_component_gather() {
    let shader = compile_offset_sample("ImageGather4LzO", 1, 9);

    for component in ["x", "y", "z", "w"] {
        assert!(
            shader
                .source
                .contains(&format!("), component::{component})")),
            "missing offset gather for component {component}"
        );
    }
    assert!(!shader.source.contains("int2(0), component::"));
    assert!(!shader.source.contains("tex0.get_width("));
}

#[test]
fn zero_lod_comparison_gather_uses_its_packed_native_offset() {
    let shader = compile_offset_sample("ImageGather4CLzO", 1, 9);

    assert!(shader.source.contains("depth2d<float> tex0"));
    assert!(shader.source.contains(".gather_compare("));
    assert!(shader.source.contains(", int2("));
    assert!(!shader.source.contains(", int2(0))"));
    assert!(!shader.source.contains("tex0.get_width("));
}

#[test]
fn native_2d_array_offsets_preserve_the_discrete_layer_operand() {
    let shader = compile_offset_sample("ImageSampleLzO", 5, 13);

    assert!(shader.source.contains("texture2d_array<float> tex0"));
    assert!(shader.source.contains(".sample("));
    assert!(shader.source.contains(".xy"));
    assert!(shader.source.contains("tex0.get_array_size() - 1u"));
    assert!(shader.source.contains("level(0.0f), int2("));
    assert!(!shader.source.contains("tex0.get_width("));
    assert!(!shader.source.contains("tex0.get_height("));
}

#[test]
fn native_2d_array_gather_offsets_use_array_texture_overloads() {
    let ordinary = compile_offset_sample("ImageGather4LzO", 5, 13);
    let comparison = compile_offset_sample("ImageGather4CLzO", 5, 13);

    assert!(ordinary.source.contains("texture2d_array<float> tex0"));
    assert!(ordinary.source.contains(".gather("));
    assert!(ordinary.source.contains("tex0.get_array_size() - 1u"));
    assert!(!ordinary.source.contains("int2(0), component::"));
    assert!(comparison.source.contains("depth2d_array<float> tex0"));
    assert!(comparison.source.contains(".gather_compare("));
    assert!(comparison.source.contains("tex0.get_array_size() - 1u"));
    assert!(!comparison.source.contains(", int2(0))"));
}

#[test]
fn true_3d_offsets_keep_the_three_axis_coordinate_lowering() {
    let shader = compile_offset_sample("ImageSampleLzO", 2, 10);

    assert!(shader.source.contains("texture3d<float> tex0"));
    assert!(shader.source.contains("tex0.get_width("));
    assert!(shader.source.contains("tex0.get_height("));
    assert!(shader.source.contains("tex0.get_depth("));
    assert!(shader.source.contains(".sample("));
    assert!(!shader.source.contains("level(0.0f), int2("));
}

#[test]
fn cube_offsets_remain_a_loud_translation_error() {
    let control = cube_image_control(vec![0, 1, 2, 3], 8);
    let instruction = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageSampleLzO"),
        words: vec![0, 0],
        sources: (0..4)
            .map(Operand::vector)
            .chain([Operand::scalar(0), Operand::scalar(8)])
            .collect(),
        destinations: vec![Operand::vector(8)],
        control: Some(InstructionControl::Image(control.clone())),
    };
    let evaluation = cube_evaluation(Opcode::Known("ImageSampleLzO"), &control);
    let error = compile_msl(
        &Program::new(0x1000, vec![instruction]),
        &evaluation,
        &MetalCompileOptions::compute(0),
    )
    .expect_err("cube integer offsets must not be silently approximated");
    assert!(error.to_string().contains("cube sampling does not support"));
}

#[cfg(target_os = "macos")]
#[test]
fn generated_offset_image_msl_compiles_with_apple_metal() {
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
        eprintln!("skipping Apple Metal image-offset compile test: no metal compiler found");
        return;
    };
    let shaders = [
        ("sample-2d", compile_offset_sample("ImageSampleLzO", 1, 9)),
        ("gather-2d", compile_offset_sample("ImageGather4LzO", 1, 9)),
        (
            "gather-compare-2d",
            compile_offset_sample("ImageGather4CLzO", 1, 9),
        ),
        (
            "sample-2d-array",
            compile_offset_sample("ImageSampleLzO", 5, 13),
        ),
        (
            "gather-2d-array",
            compile_offset_sample("ImageGather4LzO", 5, 13),
        ),
        (
            "gather-compare-2d-array",
            compile_offset_sample("ImageGather4CLzO", 5, 13),
        ),
        // The minimum-LOD clamp overloads, which accept a bias or gradient beside them and sit
        // between the LOD option and the texel offset.
        (
            "sample-lod-clamp-2d",
            compile_offset_sample("ImageSampleClO", 1, 9),
        ),
        (
            "sample-bias-lod-clamp-2d",
            compile_offset_sample("ImageSampleBClO", 1, 9),
        ),
        (
            "sample-gradient-lod-clamp-2d",
            compile_offset_sample("ImageSampleDClO", 1, 9),
        ),
        (
            "sample-compare-lod-clamp-2d",
            compile_offset_sample("ImageSampleCClO", 1, 9),
        ),
        (
            "sample-lod-clamp-2d-array",
            compile_offset_sample("ImageSampleClO", 5, 13),
        ),
    ];
    let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory = std::env::temp_dir().join(format!(
        "remu-image-offset-metal-{}-{sequence}",
        std::process::id()
    ));
    std::fs::create_dir_all(&directory).expect("create Metal image-offset test directory");
    for (name, shader) in shaders {
        let source = directory.join(format!("{name}.metal"));
        let air = directory.join(format!("{name}.air"));
        std::fs::write(&source, shader.source).expect("write generated image-offset MSL");
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
            "{name} image-offset MSL failed to compile:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    std::fs::remove_dir_all(directory).expect("remove Metal image-offset test directory");
}

#[test]
fn canonicalizes_silent_hill_scale_image_uses_below_the_direct_texture_limit() {
    const IMAGE_INSTRUCTION_COUNT: usize = 147;
    let control = ImageControl {
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
        unorm: false,
    };
    let mut instructions = Vec::with_capacity(IMAGE_INSTRUCTION_COUNT);
    let mut evaluation = empty_evaluation();
    let mut resource_descriptor = vec![0; 8];
    resource_descriptor[3] = 0xfac | (9 << 28);

    for index in 0..IMAGE_INSTRUCTION_COUNT {
        let pc = u32::try_from(index * 8).expect("test PC fits u32");
        instructions.push(Instruction {
            pc,
            encoding: ShaderEncoding::Mimg,
            opcode: Opcode::Known("ImageSampleLz"),
            words: vec![0, 0],
            sources: vec![Operand::vector(0), Operand::vector(1), Operand::scalar(0)],
            destinations: (4..8).map(Operand::vector).collect(),
            control: Some(InstructionControl::Image(control.clone())),
        });
        evaluation.image_bindings.push(ImageBinding { descriptor_index_is_runtime: false,
            pc,
            opcode: Opcode::Known("ImageSampleLz"),
            control: control.clone(),
            resource_descriptor: resource_descriptor.clone().into(),
            sampler_descriptor: vec![0; 4].into(),
            mip_level: Some(0),
        });
    }

    let program = Program::new(0x800c_1100_00, instructions);
    let shader = compile_msl(&program, &evaluation, &MetalCompileOptions::compute(0))
        .expect("repeated descriptor uses must share one Metal binding");

    assert_eq!(evaluation.image_bindings.len(), IMAGE_INSTRUCTION_COUNT);
    assert_eq!(shader.bindings.texture_count, 1);
    assert_eq!(shader.bindings.sampler_count, 1);
    assert_eq!(shader.source.matches("[[texture(").count(), 1);
    assert!(shader.source.contains("tex0 [[texture(0)]]"));
    assert!(!shader.source.contains("[[texture(128)]]"));
    assert!(!shader.source.contains("tex128"));

    let mut genuinely_distinct = evaluation.clone();
    for (index, binding) in genuinely_distinct.image_bindings.iter_mut().enumerate() {
        binding.resource_descriptor[0] =
            u32::try_from(index).expect("test descriptor index fits u32");
    }
    let error = compile_msl(
        &program,
        &genuinely_distinct,
        &MetalCompileOptions::compute(0),
    )
    .expect_err("129+ genuinely distinct textures must be rejected, not dropped");
    assert!(error.to_string().contains("distinct image descriptors"));
}

#[test]
fn comparison_gather_uses_distinct_depth_resources_and_gather_compare_codegen() {
    let ordinary_control = ImageControl {
        dmask: 1,
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
        unorm: false,
    };
    let comparison_control = ImageControl {
        address_registers: vec![0, 1, 2],
        vector_data: 8,
        ..ordinary_control.clone()
    };
    let ordinary = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageGather4"),
        words: vec![0, 0],
        sources: vec![Operand::vector(0), Operand::vector(1), Operand::scalar(0)],
        destinations: (4..8).map(Operand::vector).collect(),
        control: Some(InstructionControl::Image(ordinary_control.clone())),
    };
    let comparison = Instruction {
        pc: 8,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageGather4C"),
        words: vec![0, 0],
        sources: vec![
            Operand::vector(0),
            Operand::vector(1),
            Operand::vector(2),
            Operand::scalar(0),
            Operand::scalar(8),
        ],
        destinations: (8..12).map(Operand::vector).collect(),
        control: Some(InstructionControl::Image(comparison_control.clone())),
    };
    let mut evaluation = empty_evaluation();
    let mut resource_descriptor = vec![0; 8];
    resource_descriptor[3] = 0x204 | (9 << 28);
    for (instruction, control) in [
        (&ordinary, ordinary_control),
        (&comparison, comparison_control),
    ] {
        evaluation.image_bindings.push(ImageBinding { descriptor_index_is_runtime: false,
            pc: instruction.pc,
            opcode: instruction.opcode,
            control,
            resource_descriptor: resource_descriptor.clone().into(),
            sampler_descriptor: vec![3 << 12, 0, 0, 0].into(),
            mip_level: None,
        });
    }

    let shader = compile_msl(
        &Program::new(0x1000, vec![ordinary.clone(), comparison.clone()]),
        &evaluation,
        &MetalCompileOptions::compute(0),
    )
    .expect("ordinary and comparison gathers should lower");

    assert_eq!(shader.bindings.texture_count, 2);
    assert_eq!(shader.bindings.sampler_count, 2);
    assert!(
        shader
            .source
            .contains("texture2d<float> tex0 [[texture(0)]]")
    );
    assert!(shader.source.contains("depth2d<float> tex1 [[texture(1)]]"));
    assert!(shader.source.contains("tex0.gather(remu_samplers.s0"));
    assert!(
        shader
            .source
            .contains("tex1.gather_compare(remu_samplers.s1")
    );

    for index in 0..30_u32 {
        evaluation
            .global_memory_bindings
            .push(remu_shader::GlobalMemoryBinding {
                scalar_address: index * 4,
                base_address: 0x10_0000 + u64::from(index) * 0x1000,
                instruction_pcs: Vec::new().into(),
                byte_count: 4,
                data: vec![0; 4],
                writable: false,
                write_back_to_guest: false,
            });
    }
    let direct = compile_msl(
        &Program::new(0x1000, vec![ordinary.clone(), comparison.clone()]),
        &evaluation,
        &MetalCompileOptions::compute(0),
    )
    .expect("samplers should fall back to direct stage bindings at Metal's buffer limit");
    assert_eq!(direct.bindings.initial_scalar_buffer_index, 30);
    assert_eq!(direct.bindings.sampler_argument_buffer_index, None);
    assert!(
        direct
            .source
            .contains("sampler remu_sampler_0 [[sampler(0)]]")
    );
    assert!(direct.source.contains("tex0.gather(remu_sampler_0"));
    assert!(
        !direct
            .source
            .contains("RemuSamplers& remu_samplers [[buffer(31)]]")
    );

    evaluation
        .global_memory_bindings
        .push(remu_shader::GlobalMemoryBinding {
            scalar_address: 30 * 4,
            base_address: 0x20_0000,
            instruction_pcs: Vec::new().into(),
            byte_count: 4,
            data: vec![0; 4],
            writable: false,
            write_back_to_guest: false,
        });
    let packed = compile_msl(
        &Program::new(0x1000, vec![ordinary, comparison]),
        &evaluation,
        &MetalCompileOptions::compute(0),
    )
    .expect("large global interfaces should use one Metal argument buffer");
    assert_eq!(packed.bindings.global_buffer_count, 31);
    assert_eq!(packed.bindings.global_argument_buffer_index, Some(0));
    assert_eq!(packed.bindings.initial_scalar_buffer_index, 1);
    assert_eq!(packed.bindings.sampler_argument_buffer_index, Some(2));
    assert!(
        packed
            .source
            .contains("constant RemuGlobalBuffers& remu_globals [[buffer(0)]]")
    );
    assert!(
        packed
            .source
            .contains("const device uchar* b30 = remu_globals.b30;")
    );
    assert!(packed.source.contains("tex0.gather(remu_samplers.s0"));
    assert!(
        packed
            .source
            .contains("tex1.gather_compare(remu_samplers.s1")
    );
    assert!(!packed.source.contains("tex0.gather(remu_sampler_0"));
    assert!(!packed.source.contains("[[buffer(31)]]"));
}

#[test]
fn comparison_samples_use_depth_texture_typing_and_the_z_reference_operand() {
    let control = ImageControl {
        dmask: 1,
        vector_address: 0,
        address_registers: vec![0, 1, 2],
        vector_data: 4,
        scalar_resource: 0,
        scalar_sampler: 8,
        dimension: 1,
        is_array: false,
        glc: false,
        slc: false,
        a16: false,
        d16: false,
        unorm: false,
    };
    let instruction = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageSampleCLz"),
        words: vec![0, 0],
        sources: vec![
            Operand::vector(0),
            Operand::vector(1),
            Operand::vector(2),
            Operand::scalar(0),
            Operand::scalar(8),
        ],
        destinations: vec![Operand::vector(4)],
        control: Some(InstructionControl::Image(control.clone())),
    };
    let mut evaluation = empty_evaluation();
    let mut resource_descriptor = vec![0; 8];
    resource_descriptor[3] = 0x204 | (9 << 28);
    evaluation.image_bindings.push(ImageBinding { descriptor_index_is_runtime: false,
        pc: 0,
        opcode: Opcode::Known("ImageSampleCLz"),
        control,
        resource_descriptor: resource_descriptor.into(),
        sampler_descriptor: vec![3 << 12, 0, 0, 0].into(),
        mip_level: None,
    });
    let shader = compile_msl(
        &Program::new(0x1000, vec![instruction]),
        &evaluation,
        &MetalCompileOptions::pixel(Vec::new(), 0, 0),
    )
    .expect("depth comparison sample should lower");

    assert!(shader.source.contains("depth2d<float> tex0"));
    assert!(shader.source.contains("tex0.sample_compare("));
    assert!(shader.source.contains("as_type<float>(remu_v_0)"));
    assert!(
        shader
            .source
            .contains("float2(as_type<float>(remu_v_1), as_type<float>(remu_v_2))")
    );
    assert!(shader.source.contains(", level(0.0f))"));
}

fn cube_image_control(address_registers: Vec<u32>, vector_data: u32) -> ImageControl {
    ImageControl {
        dmask: 1,
        vector_address: 0,
        address_registers,
        vector_data,
        scalar_resource: 0,
        scalar_sampler: 8,
        dimension: 3,
        is_array: false,
        glc: false,
        slc: false,
        a16: false,
        d16: false,
        unorm: false,
    }
}

fn cube_evaluation(opcode: Opcode, control: &ImageControl) -> ShaderEvaluation {
    let mut evaluation = empty_evaluation();
    let mut resource_descriptor = vec![0; 8];
    resource_descriptor[3] = 0x204 | (11 << 28);
    evaluation.image_bindings.push(ImageBinding { descriptor_index_is_runtime: false,
        pc: 0,
        opcode,
        control: control.clone(),
        resource_descriptor: resource_descriptor.into(),
        sampler_descriptor: vec![3 << 12, 0, 0, 0].into(),
        mip_level: None,
    });
    evaluation
}

#[test]
fn cube_comparison_samples_bind_a_depth_array_and_invert_the_face_atlas_coordinates() {
    // GCN cube sampling passes V_CUBEID/SC/TC/MA face-atlas coordinates
    // (sc/(2|ma|) + 1.5, tc/(2|ma|) + 1.5, faceId), not a direction vector. The lowering
    // must bind the cube as a 2D array and index (x - 1, y - 1) at layer round(z).
    let control = cube_image_control(vec![0, 1, 2, 3], 4);
    let instruction = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageSampleCLz"),
        words: vec![0, 0],
        sources: vec![
            Operand::vector(0),
            Operand::vector(1),
            Operand::vector(2),
            Operand::vector(3),
            Operand::scalar(0),
            Operand::scalar(8),
        ],
        destinations: vec![Operand::vector(4)],
        control: Some(InstructionControl::Image(control.clone())),
    };
    let evaluation = cube_evaluation(Opcode::Known("ImageSampleCLz"), &control);
    let shader = compile_msl(
        &Program::new(0x1000, vec![instruction]),
        &evaluation,
        &MetalCompileOptions::pixel(Vec::new(), 0, 0),
    )
    .expect("cube depth comparison sample should lower");

    assert!(shader.source.contains("depth2d_array<float> tex0"));
    assert!(!shader.source.contains("depthcube"));
    assert!(shader.source.contains(".x - 1.0f"));
    assert!(shader.source.contains(".y - 1.0f"));
    assert!(shader.source.contains("tex0.get_array_size() - 1u"));
    assert!(shader.source.contains("tex0.sample_compare("));
    assert!(shader.source.contains(", level(0.0f))"));
    // The comparison reference is v0 and the atlas triple is v1..v3.
    assert!(shader.source.contains("as_type<float>(remu_v_0)"));
    assert!(shader.source.contains(
        "float3(as_type<float>(remu_v_1), as_type<float>(remu_v_2), as_type<float>(remu_v_3))"
    ));
}

#[test]
fn cube_samples_bind_a_2d_array_and_index_the_rounded_face_layer() {
    let control = cube_image_control(vec![0, 1, 2], 4);
    let instruction = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageSampleLz"),
        words: vec![0, 0],
        sources: vec![
            Operand::vector(0),
            Operand::vector(1),
            Operand::vector(2),
            Operand::scalar(0),
            Operand::scalar(8),
        ],
        destinations: vec![Operand::vector(4)],
        control: Some(InstructionControl::Image(control.clone())),
    };
    let evaluation = cube_evaluation(Opcode::Known("ImageSampleLz"), &control);
    let shader = compile_msl(
        &Program::new(0x1000, vec![instruction]),
        &evaluation,
        &MetalCompileOptions::pixel(Vec::new(), 0, 0),
    )
    .expect("cube sample should lower");

    assert!(shader.source.contains("texture2d_array<float> tex0"));
    assert!(!shader.source.contains("texturecube"));
    assert!(shader.source.contains(".x - 1.0f"));
    assert!(shader.source.contains(".y - 1.0f"));
    assert!(shader.source.contains("min(uint(rint(max("));
    assert!(shader.source.contains("tex0.get_array_size() - 1u"));
    assert!(shader.source.contains("tex0.sample("));
}

#[test]
fn cube_gradient_samples_pass_2d_face_gradients_to_the_array_sampler() {
    let control = cube_image_control(vec![0, 1, 2, 3, 4, 5, 6, 7, 8], 10);
    let instruction = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageSampleD"),
        words: vec![0, 0],
        sources: (0..9)
            .map(Operand::vector)
            .chain([Operand::scalar(0), Operand::scalar(8)])
            .collect(),
        destinations: vec![Operand::vector(10)],
        control: Some(InstructionControl::Image(control.clone())),
    };
    let evaluation = cube_evaluation(Opcode::Known("ImageSampleD"), &control);
    let shader = compile_msl(
        &Program::new(0x1000, vec![instruction]),
        &evaluation,
        &MetalCompileOptions::pixel(Vec::new(), 0, 0),
    )
    .expect("cube gradient sample should lower");

    assert!(shader.source.contains("texture2d_array<float> tex0"));
    assert!(shader.source.contains("gradient2d("));
    assert!(!shader.source.contains("gradientcube"));
    assert!(shader.source.contains(".xy,"));
}

#[test]
fn image_get_resinfo_queries_mip_relative_dimensions() {
    let control = ImageControl {
        dmask: 0xf,
        vector_address: 0,
        address_registers: vec![0],
        vector_data: 4,
        scalar_resource: 0,
        scalar_sampler: 0,
        dimension: 1,
        is_array: false,
        glc: false,
        slc: false,
        a16: false,
        d16: false,
        unorm: false,
    };
    let instruction = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageGetResinfo"),
        words: vec![0, 0],
        sources: vec![Operand::vector(0), Operand::scalar(0)],
        destinations: (4..8).map(Operand::vector).collect(),
        control: Some(InstructionControl::Image(control.clone())),
    };
    let mut evaluation = empty_evaluation();
    evaluation.image_bindings.push(ImageBinding { descriptor_index_is_runtime: false,
        pc: 0,
        opcode: Opcode::Known("ImageGetResinfo"),
        control: control.clone(),
        resource_descriptor: vec![0; 8].into(),
        sampler_descriptor: Vec::new().into(),
        mip_level: None,
    });
    let shader = compile_msl(
        &Program::new(0x1000, vec![instruction]),
        &evaluation,
        &MetalCompileOptions::pixel(Vec::new(), 0, 0),
    )
    .expect("ImageGetResinfo should lower");
    assert!(shader.source.contains("tex0.get_width("));
    assert!(shader.source.contains("tex0.get_height("));
    assert!(shader.source.contains("tex0.get_num_mip_levels()"));
    assert!(shader.source.contains("if (exec) remu_v_4 ="));
    assert!(shader.source.contains("if (exec) remu_v_5 ="));
    assert!(shader.source.contains("if (exec) remu_v_6 ="));
    assert!(shader.source.contains("if (exec) remu_v_7 ="));
    assert!(!shader.source.contains("as_type<uint>((float)"));
}

#[test]
fn image_get_resinfo_unorm_writes_float_bitcasts() {
    let control = ImageControl {
        dmask: 0x3,
        vector_address: 0,
        address_registers: vec![0],
        vector_data: 2,
        scalar_resource: 0,
        scalar_sampler: 0,
        dimension: 1,
        is_array: false,
        glc: false,
        slc: false,
        a16: false,
        d16: false,
        unorm: true,
    };
    let instruction = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageGetResinfo"),
        words: vec![0, 0],
        sources: vec![Operand::vector(0), Operand::scalar(0)],
        destinations: (2..4).map(Operand::vector).collect(),
        control: Some(InstructionControl::Image(control.clone())),
    };
    let mut evaluation = empty_evaluation();
    evaluation.image_bindings.push(ImageBinding { descriptor_index_is_runtime: false,
        pc: 0,
        opcode: Opcode::Known("ImageGetResinfo"),
        control,
        resource_descriptor: vec![0; 8].into(),
        sampler_descriptor: Vec::new().into(),
        mip_level: None,
    });
    let shader = compile_msl(
        &Program::new(0x1000, vec![instruction]),
        &evaluation,
        &MetalCompileOptions::pixel(Vec::new(), 0, 0),
    )
    .expect("unorm ImageGetResinfo should lower");
    assert!(shader.source.contains("as_type<uint>((float)"));
    assert!(shader.source.contains("tex0.get_width("));
    assert!(shader.source.contains("tex0.get_height("));
}

#[test]
fn rematerializes_dim3_sample_against_type2d_srd_like_ue_slate() {
    // Unreal Slate font atlases: MIMG DIM=3D (control.dimension == 2) against TYPE=2D SRD.
    // Kyty rematerializes image.dimension from the descriptor; without that, Metal binds
    // texture3d and samples with float3 → solid tofu glyphs.
    let control = ImageControl {
        dmask: 0xf,
        vector_address: 0,
        address_registers: vec![0, 1, 2],
        vector_data: 4,
        scalar_resource: 0,
        scalar_sampler: 8,
        dimension: 2,
        is_array: false,
        glc: false,
        slc: false,
        a16: false,
        d16: false,
        unorm: false,
    };
    let instruction = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageSample"),
        words: vec![0, 0],
        sources: vec![Operand::vector(0), Operand::vector(1), Operand::scalar(0)],
        destinations: (4..8).map(Operand::vector).collect(),
        control: Some(InstructionControl::Image(control.clone())),
    };
    let mut evaluation = empty_evaluation();
    let mut resource_descriptor = vec![0; 8];
    resource_descriptor[3] = 0xfac | (9 << 28);
    evaluation.image_bindings.push(ImageBinding { descriptor_index_is_runtime: false,
        pc: 0,
        opcode: Opcode::Known("ImageSample"),
        control,
        resource_descriptor: resource_descriptor.into(),
        sampler_descriptor: vec![0; 4].into(),
        mip_level: None,
    });
    let shader = compile_msl(
        &Program::new(0x1000, vec![instruction]),
        &evaluation,
        &MetalCompileOptions::pixel(Vec::new(), 0, 0),
    )
    .expect("DIM=3D sample against TYPE=2D should rematerialize");
    assert!(
        shader.source.contains("texture2d<float> tex0"),
        "expected texture2d from SRD TYPE, got:\n{}",
        shader.source
    );
    assert!(
        !shader.source.contains("texture3d"),
        "must not bind texture3d for TYPE=2D Slate atlas"
    );
    assert!(
        shader.source.contains(".xy") && shader.source.contains("tex0.sample("),
        "expected .xy sample coords for rematerialized 2D texture:\n{}",
        shader.source
    );
}

fn compile_array_srd_sample(texture_type: u32, dimension: u32) -> String {
    let address_registers = if dimension == 2 {
        vec![0, 1, 2]
    } else {
        vec![0, 1]
    };
    let control = ImageControl {
        dmask: 0xf,
        vector_address: 0,
        address_registers: address_registers.clone(),
        vector_data: 4,
        scalar_resource: 0,
        scalar_sampler: 8,
        dimension,
        is_array: false,
        glc: false,
        slc: false,
        a16: false,
        d16: false,
        unorm: false,
    };
    let instruction = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageSample"),
        words: vec![0, 0],
        sources: address_registers
            .into_iter()
            .map(Operand::vector)
            .chain([Operand::scalar(0), Operand::scalar(8)])
            .collect(),
        destinations: (4..8).map(Operand::vector).collect(),
        control: Some(InstructionControl::Image(control.clone())),
    };
    let mut evaluation = empty_evaluation();
    let mut resource_descriptor = vec![0; 8];
    resource_descriptor[3] = 0x204 | (texture_type << 28);
    evaluation.image_bindings.push(ImageBinding { descriptor_index_is_runtime: false,
        pc: 0,
        opcode: Opcode::Known("ImageSample"),
        control,
        resource_descriptor: resource_descriptor.into(),
        sampler_descriptor: vec![0; 4].into(),
        mip_level: None,
    });
    compile_msl(
        &Program::new(0x1000, vec![instruction]),
        &evaluation,
        &MetalCompileOptions::pixel(Vec::new(), 0, 0),
    )
    .expect("array SRD sample should lower")
    .source
}

#[test]
fn array_srd_types_supply_layer_zero_for_2d_mimg_addresses() {
    for texture_type in [12, 13, 15] {
        let source = compile_array_srd_sample(texture_type, 1);
        assert!(
            source.contains("texture2d_array<float> tex0"),
            "TYPE={texture_type} must bind a Metal array texture:\n{source}"
        );
        let sample = source
            .lines()
            .find(|line| line.contains("tex0.sample("))
            .expect("sample call");
        assert!(
            sample.contains(", 0u)"),
            "TYPE={texture_type} DIM=2D must explicitly select layer zero: {sample}"
        );
    }
}

#[test]
fn array_srd_separates_a_3d_mimg_z_coordinate_into_a_layer() {
    let source = compile_array_srd_sample(12, 2);
    assert!(source.contains("texture2d_array<float> tex0"));
    assert!(source.contains(".xy"));
    assert!(source.contains("tex0.get_array_size() - 1u"));
    let sample = source
        .lines()
        .find(|line| line.contains("tex0.sample("))
        .expect("sample call");
    assert!(
        sample.matches(',').count() >= 2,
        "array sample must pass separate coordinate and layer arguments: {sample}"
    );
}

#[test]
fn three_d_srd_supplies_zero_depth_for_a_2d_mimg_address() {
    let source = compile_array_srd_sample(10, 1);
    assert!(source.contains("texture3d<float> tex0"));
    assert!(
        source.contains("float3(") && source.contains(", 0.0f)"),
        "TYPE=3D DIM=2D must explicitly select depth slice zero:\n{source}"
    );
    let sample = source
        .lines()
        .find(|line| line.contains("tex0.sample("))
        .expect("sample call");
    assert!(
        !sample.contains("float2("),
        "a Metal texture3d sample must not receive float2 coordinates: {sample}"
    );
}
