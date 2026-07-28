// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Image lowering tests: sampled and storage textures, descriptor component
//! selection, gathers and depth comparison sampling.

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
    evaluation.image_bindings.push(ImageBinding {
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
    evaluation.image_bindings.push(ImageBinding {
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
        evaluation.image_bindings.push(ImageBinding {
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
        evaluation.image_bindings.push(ImageBinding {
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
    evaluation.image_bindings.push(ImageBinding {
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
