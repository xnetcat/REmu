// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use metal::CompileOptions;
use remu_gpu::{GpuRegister, ShaderSnapshot, gpu_event_channel};
use remu_shader::{
    GlobalMemoryBinding, ImageBinding, ImageControl, Instruction, InstructionControl,
    InterpolationControl, Opcode, Operand, Program, ShaderEncoding, ShaderEvaluation,
    decode_program,
};
use std::sync::Arc;

use super::{
    MetalPresenter, canonical_sampler_bindings, canonical_texture_bindings, compute_launch,
    graphics_shader_resource_key, image_component_class, missing_vertex_parameter_exports,
    procedural_rectangle_export_stage, rectangle_coordinate_parameter_mask,
    rectangle_draw_has_incomplete_interface, sampler_lod_bias, sampler_uses_normalized_coordinates,
    shader_scalar_metadata, shader_scalar_metadata_for_registers,
    validate_compute_launch_for_metal, wave64_split_blockers,
};

#[test]
fn decodes_force_unnormalized_sampler_coordinates() {
    assert!(sampler_uses_normalized_coordinates(&[0, 0, 0, 0]));
    assert!(!sampler_uses_normalized_coordinates(&[1 << 15, 0, 0, 0,]));
}

#[test]
fn decodes_signed_sampler_lod_bias() {
    assert_eq!(sampler_lod_bias(&[0, 0, 0x03bf, 0]), 959.0 / 256.0);
    assert_eq!(sampler_lod_bias(&[0, 0, 0x3fff, 0]), -1.0 / 256.0);
    assert_eq!(sampler_lod_bias(&[0, 0, 0x2000, 0]), -32.0);
}

#[test]
fn rejects_only_non_default_pixel_inputs_without_vertex_exports() {
    let mut controls =
        std::array::from_fn(|index| u32::try_from(index).expect("input-control index fits u32"));
    controls[0] = 0x320;
    controls[1] = 3;

    assert_eq!(
        missing_vertex_parameter_exports(1 << 3, 0b11, &controls),
        Vec::new(),
        "logical input zero uses architectural default 3 and input one maps to PARAM3"
    );
    assert_eq!(
        missing_vertex_parameter_exports(0, 0b11, &controls),
        vec![(1, 3, 3)],
        "a non-default PARAM3 input cannot be supplied by a position-only stage"
    );
}

#[test]
fn recognizes_the_architectural_procedural_rectangle_export_stage() {
    let words = [
        0xbfa0_0001,
        0x7e00_0000,
        0x7e00_0000,
        0x7e00_0000,
        0x93eb_ff03,
        0x0008_0008,
        0x8f6a_8c6b,
        0x8700_ff03,
        0x0000_00ff,
        0x887c_6a00,
        0xbf90_0009,
        0x81ea_6bc0,
        0x90fe_6ac1,
        0xf800_0941,
        0x0000_0000,
        0x81ea_00c0,
        0xbf8c_ff0f,
        0x90fe_6ac1,
        0x3604_0a81,
        0x2c06_0a81,
        0x7e00_0280,
        0x7e02_02f2,
        0xd746_0002,
        0x0305_0302,
        0xd746_0003,
        0x0305_0303,
        0x7e04_0b02,
        0x7e06_0b03,
        0xf800_08cf,
        0x0100_0302,
        0xbf81_0000,
    ];
    let program = decode_program(0x1000, &words).expect("observed AGC rectangle stage decodes");

    assert!(procedural_rectangle_export_stage(&program));
}

#[test]
fn recognizes_an_incomplete_procedural_rectangle_interface() {
    let export_words = [
        0xbfa0_0001,
        0x7e00_0000,
        0x7e00_0000,
        0x7e00_0000,
        0x93eb_ff03,
        0x0008_0008,
        0x8f6a_8c6b,
        0x8700_ff03,
        0x0000_00ff,
        0x887c_6a00,
        0xbf90_0009,
        0x81ea_6bc0,
        0x90fe_6ac1,
        0xf800_0941,
        0x0000_0000,
        0x81ea_00c0,
        0xbf8c_ff0f,
        0x90fe_6ac1,
        0x3604_0a81,
        0x2c06_0a81,
        0x7e00_0280,
        0x7e02_02f2,
        0xd746_0002,
        0x0305_0302,
        0xd746_0003,
        0x0305_0303,
        0x7e04_0b02,
        0x7e06_0b03,
        0xf800_08cf,
        0x0100_0302,
        0xbf81_0000,
    ];
    let export =
        decode_program(0x1000, &export_words).expect("observed AGC rectangle stage decodes");
    let pixel = Program::new(
        0x2000,
        vec![Instruction {
            pc: 0x2000,
            encoding: ShaderEncoding::Vintrp,
            opcode: Opcode::Known("VInterpMovF32"),
            words: Vec::new(),
            sources: Vec::new(),
            destinations: vec![Operand::vector(0)],
            control: Some(InstructionControl::Interpolation(InterpolationControl {
                attribute: 1,
                channel: 0,
            })),
        }],
    );
    let snapshot = |program: Program| ShaderSnapshot {
        header_address: program.address,
        program: Arc::new(program),
        supplemental_programs: Arc::from([]),
        metadata: None,
    };
    let mut draw = super::indexed_draw(0, 0, 3);
    draw.indexed = false;
    draw.primitive_type = 7;
    draw.export_shader = Some(snapshot(export));
    draw.pixel_shader = Some(snapshot(pixel));

    assert!(
        rectangle_draw_has_incomplete_interface(&draw),
        "the captured interface remains incomplete even though color-disable semantics make its draw a no-op"
    );
}

#[test]
fn derives_procedural_rectangle_coordinates_from_the_first_sample_inputs() {
    let interpolation = |pc, opcode, attribute, destination| Instruction {
        pc,
        encoding: ShaderEncoding::Vintrp,
        opcode: Opcode::Known(opcode),
        words: Vec::new(),
        sources: Vec::new(),
        destinations: vec![Operand::vector(destination)],
        control: Some(InstructionControl::Interpolation(InterpolationControl {
            attribute,
            channel: 0,
        })),
    };
    let sample = Instruction {
        pc: 12,
        encoding: ShaderEncoding::Mimg,
        opcode: Opcode::Known("ImageSample"),
        words: Vec::new(),
        sources: vec![Operand::vector(10)],
        destinations: vec![Operand::vector(0)],
        control: Some(InstructionControl::Image(ImageControl {
            dmask: 0xf,
            vector_address: 10,
            address_registers: vec![10],
            vector_data: 0,
            scalar_resource: 0,
            scalar_sampler: 8,
            dimension: 1,
            is_array: false,
            glc: false,
            slc: false,
            a16: false,
            d16: false,
        })),
    };
    let program = Program::new(
        0x2000,
        vec![
            interpolation(0, "VInterpP1F32", 0, 20),
            interpolation(4, "VInterpP2F32", 1, 30),
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Vop1,
                opcode: Opcode::Known("VMovB32"),
                words: Vec::new(),
                sources: vec![Operand::vector(30)],
                destinations: vec![Operand::vector(10)],
                control: None,
            },
            sample.clone(),
            interpolation(16, "VInterpMovF32", 0, 10),
        ],
    );
    let mut controls =
        std::array::from_fn(|index| u32::try_from(index).expect("control index fits u32"));
    controls[1] = 3;

    assert_eq!(
        rectangle_coordinate_parameter_mask(&program, &controls),
        Some(1 << 3),
        "P1 intermediates and modulation values first materialized after the sample are not UVs"
    );
    assert_eq!(
        rectangle_coordinate_parameter_mask(
            &Program::new(0x3000, vec![interpolation(0, "VInterpP2F32", 0, 10)]),
            &controls,
        ),
        None,
        "sample-free shaders retain the conservative compatibility fallback"
    );

    let mut sample_without_address_provenance = sample;
    let Some(InstructionControl::Image(image)) = sample_without_address_provenance.control.as_mut()
    else {
        panic!("test sample has image control");
    };
    image.address_registers.clear();
    assert_eq!(
        rectangle_coordinate_parameter_mask(
            &Program::new(0x4000, vec![sample_without_address_provenance]),
            &controls,
        ),
        None,
        "an empty dependency slice must use the conservative coordinate fallback"
    );
}

#[test]
fn graphics_shader_cache_key_keeps_dynamic_global_addresses_out_of_pipeline_identity() {
    let evaluation = |base_address| ShaderEvaluation {
        initial_scalar_registers: Vec::new().into(),
        scalar_registers: Vec::new().into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: Vec::new(),
        global_memory_bindings: vec![GlobalMemoryBinding {
            scalar_address: 4,
            base_address,
            instruction_pcs: vec![0x18c].into(),
            byte_count: 64,
            data: vec![0; 64],
            writable: false,
            write_back_to_guest: true,
        }],
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    };

    assert_eq!(
        graphics_shader_resource_key(&evaluation(0x7009_0605_ac), false),
        graphics_shader_resource_key(&evaluation(0x7009_1600_6c), false),
    );
    assert_eq!(
        graphics_shader_resource_key(&evaluation(0x7009_0605_ac), true),
        graphics_shader_resource_key(&evaluation(0x7009_1600_6c), true),
    );
}

#[test]
fn runtime_upload_representatives_follow_canonical_texture_and_sampler_slots() {
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
    let binding = |pc, opcode, descriptor: Vec<u32>| ImageBinding {
        pc,
        opcode: remu_shader::Opcode::Known(opcode),
        control: control.clone(),
        resource_descriptor: descriptor.into(),
        sampler_descriptor: if opcode.starts_with("ImageSample") {
            vec![7, 8, 9, 10].into()
        } else {
            Vec::new().into()
        },
        mip_level: None,
    };
    let descriptor_a = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let descriptor_b = vec![11, 12, 13, 14, 15, 16, 17, 18];
    let evaluation = ShaderEvaluation {
        initial_scalar_registers: Vec::new().into(),
        scalar_registers: Vec::new().into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: vec![
            binding(0, "ImageSample", descriptor_a.clone()),
            binding(8, "ImageSampleLz", descriptor_a.clone()),
            binding(16, "ImageLoad", descriptor_a),
            binding(24, "ImageSample", descriptor_b),
        ],
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    };

    assert_eq!(
        canonical_texture_bindings(&evaluation)
            .iter()
            .map(|binding| binding.pc)
            .collect::<Vec<_>>(),
        [0, 16, 24]
    );
    assert_eq!(
        canonical_sampler_bindings(&evaluation)
            .iter()
            .map(|binding| binding.pc)
            .collect::<Vec<_>>(),
        [0]
    );
}

#[test]
fn graphics_shader_cache_key_normalizes_dynamic_image_formats_and_swizzles() {
    let evaluation = |unified_format: u32, destination_select: u32| ShaderEvaluation {
        initial_scalar_registers: vec![0; 16].into(),
        scalar_registers: vec![0; 16].into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: vec![ImageBinding {
            pc: 0x20,
            opcode: remu_shader::Opcode::Known("ImageSample"),
            control: ImageControl {
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
            },
            resource_descriptor: vec![0, unified_format << 20, 0, destination_select].into(),
            sampler_descriptor: vec![0; 4].into(),
            mip_level: None,
        }],
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    };

    // Both unified formats are float-class interfaces; pixel format and component selection
    // are supplied by the texture and live resource SGPRs at execution time.
    let first = evaluation(56, 0x924); // RGBA8 unorm
    let second = evaluation(58, 0xfac); // RGBA8 unsigned-scaled
    assert_eq!(image_component_class(&first.image_bindings[0]), 0);
    assert_eq!(image_component_class(&second.image_bindings[0]), 0);
    assert_eq!(
        graphics_shader_resource_key(&first, false),
        graphics_shader_resource_key(&second, false),
    );
}

#[test]
fn appends_dynamic_global_base_and_size_metadata_after_guest_scalars() {
    let evaluation = ShaderEvaluation {
        initial_scalar_registers: vec![10, 20].into(),
        scalar_registers: vec![10, 20].into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: Vec::new(),
        global_memory_bindings: vec![GlobalMemoryBinding {
            scalar_address: 4,
            base_address: 0x0070_1234_5678,
            instruction_pcs: vec![0x18c].into(),
            byte_count: 96,
            data: vec![0; 96],
            writable: true,
            write_back_to_guest: true,
        }],
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    };

    assert_eq!(
        shader_scalar_metadata(&evaluation).expect("metadata fits"),
        [10, 20, 0x1234_5678, 0x70, 96, 1]
    );
    assert_eq!(
        shader_scalar_metadata_for_registers(
            &evaluation,
            &[
                GpuRegister {
                    offset: 0x240,
                    value: 30,
                },
                GpuRegister {
                    offset: 0x241,
                    value: 40,
                },
            ],
            0x240,
            0,
        )
        .expect("live scalar metadata fits"),
        [30, 40, 0x1234_5678, 0x70, 96, 1]
    );
}

#[test]
fn live_scalar_metadata_matches_sorted_register_snapshot_values() {
    let evaluation = ShaderEvaluation {
        initial_scalar_registers: vec![u32::MAX; 64].into(),
        scalar_registers: vec![u32::MAX; 64].into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: Vec::new(),
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    };
    let registers = (0..256_u32)
        .filter(|offset| *offset != 0x20)
        .map(|offset| GpuRegister {
            offset: 0x200 + offset,
            value: 0xa500_0000 | offset,
        })
        .collect::<Vec<_>>();
    let expected = std::array::from_fn::<_, 64, _>(|scalar| {
        let offset = 0x200 + u32::try_from(scalar).expect("scalar index fits u32");
        registers
            .iter()
            .find(|register| register.offset == offset)
            .map_or(0, |register| register.value)
    });

    assert_eq!(
        shader_scalar_metadata_for_registers(&evaluation, &registers, 0x200, 0)
            .expect("live scalar metadata fits"),
        expected
    );
    assert_eq!(expected[0], 0xa500_0000);
    assert_eq!(expected[0x20], 0);
    assert_eq!(expected[63], 0xa500_003f);
}

#[test]
fn reuses_immutable_sampler_argument_buffers() {
    let (_sender, receiver) = gpu_event_channel();
    let mut presenter = MetalPresenter::new(receiver).expect("Metal presenter initializes");
    let device = presenter
        .device
        .as_ref()
        .expect("presenter retains its Metal device")
        .clone();
    let source = r"
#include <metal_stdlib>
using namespace metal;
struct RemuSamplers {
    sampler s0 [[id(0)]];
};
kernel void remu_sampler_cache_test(
    constant RemuSamplers& samplers [[buffer(0)]],
    device float4* output [[buffer(1)]],
    texture2d<float> texture [[texture(0)]],
    uint gid [[thread_position_in_grid]]) {
    output[gid] = texture.sample(samplers.s0, float2(0.5f));
}
";
    let library = device
        .new_library_with_source(source, &CompileOptions::new())
        .expect("sampler argument-buffer MSL compiles");
    let function = library
        .get_function("remu_sampler_cache_test", None)
        .expect("sampler cache function exists");
    let evaluation = ShaderEvaluation {
        initial_scalar_registers: Vec::new().into(),
        scalar_registers: Vec::new().into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: vec![ImageBinding {
            pc: 0x100,
            opcode: Opcode::Known("ImageSample"),
            control: ImageControl {
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
            },
            resource_descriptor: vec![0; 8].into(),
            sampler_descriptor: vec![0; 4].into(),
            mip_level: None,
        }],
        global_memory_bindings: Vec::new(),
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    };

    let (first_buffer, first_samplers) = presenter
        .cached_sampler_argument_buffer(&device, &function, Some(0), &evaluation)
        .expect("first sampler argument buffer is created");
    let (second_buffer, second_samplers) = presenter
        .cached_sampler_argument_buffer(&device, &function, Some(0), &evaluation)
        .expect("identical sampler argument buffer is reused");

    assert_eq!(presenter.sampler_argument_cache.len(), 1);
    assert_eq!(first_samplers.len(), 1);
    assert_eq!(second_samplers.len(), 1);
    assert_eq!(
        first_buffer.expect("first buffer").contents(),
        second_buffer.expect("second buffer").contents()
    );
}

#[test]
fn validates_exact_metal_compute_execution_geometry() {
    validate_compute_launch_for_metal(compute_launch(), 32, false, 1_024, 32 * 1024, 0, 16 * 1024)
        .expect("representable compute launch");

    let mut splittable_wave64 = compute_launch();
    splittable_wave64.wave_lane_count = 64;
    validate_compute_launch_for_metal(splittable_wave64, 32, true, 1_024, 32 * 1024, 0, 16 * 1024)
        .expect("wave-local wave64 shader can execute as two SIMD32 subgroups");
    assert!(
        validate_compute_launch_for_metal(
            splittable_wave64,
            32,
            false,
            1_024,
            32 * 1024,
            0,
            16 * 1024,
        )
        .expect_err("cross-lane wave64 shader must not be split")
        .contains("wave64")
    );
}

#[test]
fn permits_split_wave64_readfirstlane_for_uniform_and_varying_values() {
    let uniform_move = Instruction {
        pc: 0,
        encoding: ShaderEncoding::Vop1,
        opcode: Opcode::Known("VMovB32"),
        words: Vec::new(),
        sources: vec![Operand::scalar(18)],
        destinations: vec![Operand::vector(0)],
        control: None,
    };
    let readfirstlane = Instruction {
        pc: 4,
        encoding: ShaderEncoding::Vop1,
        opcode: Opcode::Known("VReadfirstlaneB32"),
        words: Vec::new(),
        sources: vec![Operand::vector(0)],
        destinations: vec![Operand::scalar(107)],
        control: None,
    };
    assert!(
        wave64_split_blockers(&Program::new(
            0x1000,
            vec![uniform_move, readfirstlane.clone()]
        ))
        .is_empty()
    );
    assert!(wave64_split_blockers(&Program::new(0x1000, vec![readfirstlane])).is_empty());
}

#[test]
fn permits_split_wave64_half_local_permlane16_operations() {
    for opcode in ["VPermlane16B32", "VPermlanex16B32"] {
        let instruction = Instruction {
            pc: 0x90,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known(opcode),
            words: Vec::new(),
            sources: vec![Operand::vector(1), Operand::vector(2), Operand::vector(3)],
            destinations: vec![Operand::vector(8)],
            control: None,
        };
        assert!(
            wave64_split_blockers(&Program::new(0x1000, vec![instruction])).is_empty(),
            "{opcode} only permutes within one SIMD32 half"
        );
    }
}

#[test]
fn permits_exercised_split_wave64_lane_dpp_and_predication_operations() {
    let instructions = vec![
        Instruction {
            pc: 0x10,
            encoding: ShaderEncoding::Vop3,
            opcode: Opcode::Known("VReadlaneB32"),
            words: Vec::new(),
            sources: vec![Operand::vector(1), Operand::scalar(2)],
            destinations: vec![Operand::scalar(4)],
            control: None,
        },
        Instruction {
            pc: 0x18,
            encoding: ShaderEncoding::Vop2,
            opcode: Opcode::Known("VMbcntLoU32B32"),
            words: Vec::new(),
            sources: vec![Operand::scalar(2), Operand::vector(1)],
            destinations: vec![Operand::vector(1)],
            control: Some(remu_shader::InstructionControl::Dpp(
                remu_shader::DppControl {
                    control: 0x101,
                    fetch_inactive: false,
                    row_mask: 0xf,
                    bank_mask: 0xf,
                    bound_control: false,
                    absolute_mask: 0,
                    negate_mask: 0,
                },
            )),
        },
        Instruction {
            pc: 0x20,
            encoding: ShaderEncoding::Sop1,
            opcode: Opcode::Known("SOrn2SaveexecB64"),
            words: Vec::new(),
            sources: vec![Operand::scalar(8)],
            destinations: vec![Operand::scalar(10)],
            control: None,
        },
    ];
    assert!(
        wave64_split_blockers(&Program::new(0x1000, instructions)).is_empty(),
        "implemented cross-half lane exchange and half-local DPP/predication are split-safe"
    );
}

#[test]
fn rejects_compute_geometry_metal_cannot_represent_exactly() {
    let mut based = compute_launch();
    based.base_group = [4, 0, 0];
    assert!(
        validate_compute_launch_for_metal(based, 32, false, 1_024, 32 * 1024, 0, 16 * 1024)
            .expect_err("dispatch base is not silently dropped")
            .contains("nonzero base")
    );

    assert!(
        validate_compute_launch_for_metal(
            compute_launch(),
            16,
            false,
            1_024,
            32 * 1024,
            0,
            16 * 1024,
        )
        .expect_err("wave width must match")
        .contains("wave32")
    );

    let mut oversized = compute_launch();
    oversized.threads_per_group = [1_025, 1, 1];
    assert!(
        validate_compute_launch_for_metal(oversized, 32, false, 1_024, 32 * 1024, 0, 16 * 1024,)
            .expect_err("oversized workgroup must fail")
            .contains("1025 threads")
    );

    assert!(
        validate_compute_launch_for_metal(
            compute_launch(),
            32,
            false,
            1_024,
            32 * 1024,
            0,
            8 * 1024,
        )
        .expect_err("LDS binding must equal lowering layout")
        .contains("8192 bytes of LDS")
    );
    assert!(
        validate_compute_launch_for_metal(
            compute_launch(),
            32,
            false,
            1_024,
            32 * 1024,
            20 * 1024,
            16 * 1024,
        )
        .expect_err("Metal LDS limit must be enforced")
        .contains("36864 bytes")
    );
}
