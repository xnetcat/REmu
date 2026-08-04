// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Guest Gen5/RDNA2 shader translation and DXIL compilation.

use remu_shader::{
    GlobalMemoryBinding, GlobalMemoryControl, Instruction, InstructionControl, Opcode, Operand,
    OperandKind, Program, ShaderEncoding, ShaderEvaluation,
};
use remu_shader_metal::{HlslShader, MetalCompileError, MetalCompileOptions};

pub(super) fn translate_guest_shader(
    program: &Program,
    evaluation: &ShaderEvaluation,
    options: &MetalCompileOptions,
    threads_per_group: [u32; 3],
) -> Result<HlslShader, MetalCompileError> {
    remu_shader_metal::compile_hlsl(program, evaluation, options, threads_per_group)
}

pub(super) fn compile_guest_shader(
    shader: &HlslShader,
    source_name: &str,
) -> Result<Vec<u8>, String> {
    super::dxc::compile_hlsl(
        source_name,
        &shader.source,
        &shader.entry_point,
        &shader.target_profile,
    )
}

pub(super) struct GuestComputeSelfTest {
    pub(super) shader: HlslShader,
    pub(super) initial_scalars: Vec<u32>,
}

pub(super) fn build_guest_compute_self_test() -> Result<GuestComputeSelfTest, MetalCompileError> {
    const GUEST_BASE: u64 = 0x1_0000_0000;
    const BYTE_COUNT: usize = 64 * 4;
    let literal = |value| Operand {
        kind: OperandKind::LiteralConstant,
        value,
    };
    let program = Program::new(
        0x4000,
        vec![
            Instruction {
                pc: 0,
                encoding: ShaderEncoding::Vop2,
                opcode: Opcode::Known("VLshlrevB32"),
                words: vec![0],
                sources: vec![literal(2), Operand::vector(0)],
                destinations: vec![Operand::vector(2)],
                control: None,
            },
            Instruction {
                pc: 4,
                encoding: ShaderEncoding::Vop2,
                opcode: Opcode::Known("VAddI32"),
                words: vec![0],
                sources: vec![Operand::vector(0), literal(0x1234_0000)],
                destinations: vec![Operand::vector(1)],
                control: None,
            },
            Instruction {
                pc: 8,
                encoding: ShaderEncoding::Flat,
                opcode: Opcode::Known("GlobalStoreDword"),
                words: vec![0, 0],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: Some(InstructionControl::GlobalMemory(GlobalMemoryControl {
                    dword_count: 1,
                    vector_address: 2,
                    vector_data: 1,
                    vector_destination: 0,
                    scalar_address: Some(0),
                    offset_bytes: 0,
                    dlc: false,
                    lds: false,
                    glc: false,
                    slc: false,
                })),
            },
            Instruction {
                pc: 16,
                encoding: ShaderEncoding::Sopp,
                opcode: Opcode::END_PROGRAM,
                words: vec![0xbf81_0000],
                sources: Vec::new(),
                destinations: Vec::new(),
                control: None,
            },
        ],
    );
    let evaluation = ShaderEvaluation {
        initial_scalar_registers: [GUEST_BASE as u32, (GUEST_BASE >> 32) as u32]
            .into_iter()
            .collect(),
        scalar_registers: Vec::new().into(),
        scalar_memory_reads: Vec::new(),
        image_bindings: Vec::new(),
        global_memory_bindings: vec![GlobalMemoryBinding {
            scalar_address: 0,
            base_address: GUEST_BASE,
            instruction_pcs: [8].into_iter().collect(),
            byte_count: BYTE_COUNT,
            data: Vec::new(),
            writable: true,
            write_back_to_guest: true,
        }],
        compute_system_registers: None,
        vertex_inputs: Vec::new(),
    };
    let shader = translate_guest_shader(
        &program,
        &evaluation,
        &MetalCompileOptions::compute(0),
        [64, 1, 1],
    )?;
    let initial_scalars = vec![
        GUEST_BASE as u32,
        (GUEST_BASE >> 32) as u32,
        GUEST_BASE as u32,
        (GUEST_BASE >> 32) as u32,
        BYTE_COUNT as u32,
        0,
    ];
    Ok(GuestComputeSelfTest {
        shader,
        initial_scalars,
    })
}

#[cfg(test)]
mod tests {
    use remu_shader::{
        ExportControl, ImageBinding, ImageControl, Instruction, InstructionControl, Opcode,
        Operand, PixelExportFormat, PixelOutputBinding, PixelOutputKind, Program, ShaderEncoding,
        ShaderEvaluation, decode_program,
    };
    use remu_shader_metal::MetalCompileOptions;

    use super::{build_guest_compute_self_test, compile_guest_shader, translate_guest_shader};

    fn empty_evaluation(initial: impl IntoIterator<Item = u32>) -> ShaderEvaluation {
        ShaderEvaluation {
            initial_scalar_registers: initial.into_iter().collect(),
            scalar_registers: Vec::new().into(),
            scalar_memory_reads: Vec::new(),
            image_bindings: Vec::new(),
            global_memory_bindings: Vec::new(),
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        }
    }

    #[test]
    fn shared_gen5_lowering_compiles_to_real_sm66_dxil() {
        let program = Program::new(
            0x1000,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Sop1,
                    opcode: Opcode::Known("SMovB32"),
                    words: vec![0],
                    sources: vec![Operand::scalar(12)],
                    destinations: vec![Operand::scalar(80)],
                    control: None,
                },
                Instruction {
                    pc: 4,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VMovB32"),
                    words: vec![0],
                    sources: vec![Operand::scalar(80)],
                    destinations: vec![Operand::vector(3)],
                    control: None,
                },
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Vop2,
                    opcode: Opcode::Known("VAddF32"),
                    words: vec![0],
                    sources: vec![Operand::vector(3), Operand::vector(0)],
                    destinations: vec![Operand::vector(5)],
                    control: None,
                },
                Instruction {
                    pc: 12,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ],
        );
        let mut initial = vec![0; 81];
        initial[12] = 1.25_f32.to_bits();
        let translated = translate_guest_shader(
            &program,
            &empty_evaluation(initial),
            &MetalCompileOptions::compute(0),
            [64, 1, 1],
        )
        .expect("Gen5 program should lower to HLSL");
        assert_eq!(translated.target_profile, "cs_6_6");
        assert!(translated.source.contains("WaveGetLaneIndex"));
        assert!(translated.source.contains("ResourceDescriptorHeap"));

        let dxil = compile_guest_shader(&translated, "remu_guest_compute_test.hlsl")
            .expect("generated HLSL should compile to DXIL");
        assert!(dxil.len() > 128, "DXIL container should not be empty");
    }

    #[test]
    fn isolated_split_wave64_compiles_to_real_sm66_dxil() {
        let program = Program::new(
            0x15700_e5ea00,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Vop1,
                    opcode: Opcode::Known("VReadfirstlaneB32"),
                    words: vec![0],
                    sources: vec![Operand::vector(4)],
                    destinations: vec![Operand::scalar(44)],
                    control: None,
                },
                Instruction {
                    pc: 4,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ],
        );
        let mut options = MetalCompileOptions::compute_wave(0, 64);
        options.hlsl_wave64_group_isolation_axis = Some(0);
        let translated =
            translate_guest_shader(&program, &empty_evaluation([]), &options, [16, 16, 1])
                .expect("isolated split-wave shader should lower to HLSL");

        assert!(translated.source.contains("[WaveSize(32)]"));
        assert!(translated.source.contains("[numthreads(64, 1, 1)]"));
        let dxil = compile_guest_shader(&translated, "remu_guest_wave64_isolation_test.hlsl")
            .expect("isolated split-wave HLSL should compile to DXIL");
        assert!(dxil.len() > 128, "DXIL container should not be empty");
    }

    #[test]
    fn guest_vertex_and_pixel_exports_compile_to_graphics_dxil() {
        let position_export = 0xf800_0000 | 0x0f | (12 << 4) | (1 << 11) | (1 << 12);
        let vertex_program = decode_program(0x2000, &[position_export, 0x0302_0100, 0xbf81_0000])
            .expect("position export should decode");
        let vertex = translate_guest_shader(
            &vertex_program,
            &empty_evaluation([]),
            &MetalCompileOptions::vertex(0),
            [1, 1, 1],
        )
        .expect("vertex export should lower to HLSL");
        let vertex_dxil = compile_guest_shader(&vertex, "remu_guest_vertex_test.hlsl")
            .expect("generated vertex HLSL should compile");
        assert!(vertex_dxil.len() > 128);

        let color_export = 0xf800_0000 | 0x0f | (1 << 11) | (1 << 12);
        let pixel_program = decode_program(0x3000, &[color_export, 0x0302_0100, 0xbf81_0000])
            .expect("color export should decode");
        let pixel_options = MetalCompileOptions::pixel(
            vec![PixelOutputBinding {
                guest_slot: 0,
                host_location: 0,
                kind: PixelOutputKind::Float,
                export_format: PixelExportFormat::Abgr32,
            }],
            0,
            0,
        );
        let pixel = translate_guest_shader(
            &pixel_program,
            &empty_evaluation([]),
            &pixel_options,
            [1, 1, 1],
        )
        .expect("pixel export should lower to HLSL");
        let pixel_dxil = compile_guest_shader(&pixel, "remu_guest_pixel_test.hlsl")
            .expect("generated pixel HLSL should compile");
        assert!(pixel_dxil.len() > 128);
    }

    #[test]
    fn guest_ngg_stage_compiles_to_compute_fallback_dxil() {
        let encoded = |value| Operand {
            kind: remu_shader::OperandKind::EncodedConstant,
            value,
        };
        let program = Program::new(
            0x2800,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Sop1,
                    opcode: Opcode::Known("SMovB32"),
                    words: vec![0],
                    sources: vec![Operand {
                        kind: remu_shader::OperandKind::LiteralConstant,
                        value: 3 | (1 << 12),
                    }],
                    destinations: vec![Operand::scalar(124)],
                    control: None,
                },
                Instruction {
                    pc: 4,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::Known("SSendmsg"),
                    words: vec![0],
                    sources: vec![encoded(9)],
                    destinations: Vec::new(),
                    control: None,
                },
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Exp,
                    opcode: Opcode::Known("Exp"),
                    words: vec![0, 0],
                    sources: vec![Operand::vector(0); 4],
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
                    pc: 16,
                    encoding: ShaderEncoding::Exp,
                    opcode: Opcode::Known("Exp"),
                    words: vec![0, 0],
                    sources: (4..8).map(Operand::vector).collect(),
                    destinations: Vec::new(),
                    control: Some(InstructionControl::Export(ExportControl {
                        target: 12,
                        enable_mask: 15,
                        compressed: false,
                        done: true,
                        valid_mask: false,
                    })),
                },
                Instruction {
                    pc: 24,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ],
        );
        let mut evaluation = empty_evaluation([]);
        evaluation.global_memory_bindings = (0..4)
            .map(|index| remu_shader::GlobalMemoryBinding {
                scalar_address: 0xffff_ff00 + index,
                base_address: 0,
                instruction_pcs: Default::default(),
                byte_count: 4096,
                data: vec![0; 4096],
                writable: index >= 2,
                write_back_to_guest: false,
            })
            .collect();
        let mut options = MetalCompileOptions::mesh(0, 0, 64, 64, 64);
        options.wave_lane_count = 64;
        let shader = translate_guest_shader(&program, &evaluation, &options, [64, 1, 1])
            .expect("NGG stage should lower to a compute fallback");
        assert_eq!(shader.target_profile, "cs_6_6");
        assert!(
            shader
                .source
                .contains("uint remu_wave_half = (remu_thread_index >> 5u) & 1u;")
        );
        let dxil = compile_guest_shader(&shader, "remu_guest_ngg_fallback_test.hlsl")
            .expect("generated NGG compute fallback should compile");
        assert!(dxil.len() > 128);
    }

    #[test]
    fn pixel_private_lane_file_initialization_compiles_to_dxil() {
        let program = Program::new(
            0x3800,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Vop3,
                    opcode: Opcode::Known("VWritelaneB32"),
                    words: vec![0, 0],
                    sources: vec![Operand::scalar(2), Operand::scalar(4)],
                    destinations: vec![Operand::vector(12)],
                    control: None,
                },
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Vop3,
                    opcode: Opcode::Known("VReadlaneB32"),
                    words: vec![0, 0],
                    sources: vec![Operand::vector(12), Operand::scalar(4)],
                    destinations: vec![Operand::scalar(8)],
                    control: None,
                },
                Instruction {
                    pc: 16,
                    encoding: ShaderEncoding::Exp,
                    opcode: Opcode::EXPORT,
                    words: vec![0, 0],
                    sources: (0..4).map(Operand::vector).collect(),
                    destinations: Vec::new(),
                    control: Some(InstructionControl::Export(ExportControl {
                        target: 0,
                        enable_mask: 0xf,
                        compressed: false,
                        done: true,
                        valid_mask: true,
                    })),
                },
            ],
        );
        let options = MetalCompileOptions::pixel(
            vec![PixelOutputBinding {
                guest_slot: 0,
                host_location: 0,
                kind: PixelOutputKind::Float,
                export_format: PixelExportFormat::Abgr32,
            }],
            0,
            0,
        );
        let shader = translate_guest_shader(&program, &empty_evaluation([]), &options, [1, 1, 1])
            .expect("pixel lane-file program should lower");
        assert!(shader.source.contains("uint remu_v_12_lanes[32];"));
        assert!(shader.source.contains("remu_v_12_init < 32u"));
        let dxil = compile_guest_shader(&shader, "remu_guest_pixel_lane_file_test.hlsl")
            .expect("pixel lane-file HLSL should compile");
        assert!(dxil.len() > 128);
    }

    #[test]
    fn guest_global_store_compiles_to_descriptor_heap_uav_dxil() {
        let test = build_guest_compute_self_test().expect("global-store program should lower");
        assert_eq!(test.shader.bindings.global_buffer_count, 1);
        assert_eq!(test.shader.bindings.binding_table_words, 2);
        assert!(test.shader.source.contains("remu_store_u32(b0"));
        let dxil = compile_guest_shader(&test.shader, "remu_guest_global_store_test.hlsl")
            .expect("generated raw-UAV HLSL should compile");
        assert!(dxil.len() > 128);
    }

    #[test]
    fn guest_formatted_buffer_load_compiles_to_dxil() {
        let program = Program::new(
            0x5000,
            vec![
                Instruction {
                    pc: 0,
                    encoding: ShaderEncoding::Mubuf,
                    opcode: Opcode::Known("BufferLoadFormatX"),
                    words: vec![0, 0],
                    sources: vec![
                        Operand::vector(0),
                        Operand::scalar(0),
                        Operand::source(128, None),
                    ],
                    destinations: Vec::new(),
                    control: Some(remu_shader::InstructionControl::BufferMemory(
                        remu_shader::BufferMemoryControl {
                            dword_count: 1,
                            vector_address: 0,
                            vector_data: 4,
                            scalar_resource: 0,
                            instruction_format: None,
                            offset_bytes: 0,
                            index_enabled: false,
                            offset_enabled: true,
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
        let evaluation = ShaderEvaluation {
            initial_scalar_registers: [0_u32; 4].into_iter().collect(),
            scalar_registers: Vec::new().into(),
            scalar_memory_reads: Vec::new(),
            image_bindings: Vec::new(),
            global_memory_bindings: vec![remu_shader::GlobalMemoryBinding {
                scalar_address: 0,
                base_address: 0x2000_0000,
                instruction_pcs: [0].into_iter().collect(),
                byte_count: 256,
                data: Vec::new(),
                writable: false,
                write_back_to_guest: false,
            }],
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        };
        let shader = translate_guest_shader(
            &program,
            &evaluation,
            &MetalCompileOptions::compute(0),
            [64, 1, 1],
        )
        .expect("formatted load should lower");
        let dxil = compile_guest_shader(&shader, "remu_guest_formatted_load_test.hlsl")
            .expect("formatted-buffer HLSL should compile");
        assert!(dxil.len() > 128);
    }

    #[test]
    fn guest_lds_write_barrier_and_read_compile_to_dxil() {
        let ds = |pc, opcode, sources, destinations| Instruction {
            pc,
            encoding: ShaderEncoding::Ds,
            opcode: Opcode::Known(opcode),
            words: vec![0, 0],
            sources,
            destinations,
            control: Some(remu_shader::InstructionControl::DataShare(
                remu_shader::DataShareControl {
                    offset0: 0,
                    offset1: 0,
                    gds: false,
                },
            )),
        };
        let program = Program::new(
            0x6000,
            vec![
                ds(
                    0,
                    "DsWriteB32",
                    vec![Operand::vector(0), Operand::vector(1)],
                    Vec::new(),
                ),
                Instruction {
                    pc: 8,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::Known("SBarrier"),
                    words: vec![0],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
                ds(
                    12,
                    "DsReadB32",
                    vec![Operand::vector(0)],
                    vec![Operand::vector(2)],
                ),
                Instruction {
                    pc: 20,
                    encoding: ShaderEncoding::Sopp,
                    opcode: Opcode::END_PROGRAM,
                    words: vec![0xbf81_0000],
                    sources: Vec::new(),
                    destinations: Vec::new(),
                    control: None,
                },
            ],
        );
        let shader = translate_guest_shader(
            &program,
            &empty_evaluation([]),
            &MetalCompileOptions::compute(4096),
            [64, 1, 1],
        )
        .expect("LDS program should lower");
        assert!(shader.source.contains("groupshared uint remu_lds[1024]"));
        let dxil = compile_guest_shader(&shader, "remu_guest_lds_test.hlsl")
            .expect("LDS HLSL should compile");
        assert!(dxil.len() > 128);
    }

    #[test]
    fn guest_sampled_texture_compiles_to_descriptor_heap_dxil() {
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
            opcode: Opcode::Known("ImageSampleLz"),
            words: vec![0, 0],
            sources: vec![Operand::vector(0), Operand::vector(1), Operand::scalar(0)],
            destinations: (4..8).map(Operand::vector).collect(),
            control: Some(InstructionControl::Image(control.clone())),
        };
        let mut evaluation = empty_evaluation([0_u32; 12]);
        evaluation.image_bindings.push(ImageBinding {
            pc: 0,
            opcode: Opcode::Known("ImageSampleLz"),
            control,
            resource_descriptor: vec![0; 8].into(),
            sampler_descriptor: vec![0; 4].into(),
            mip_level: None,
        });
        let shader = translate_guest_shader(
            &Program::new(0x7000, vec![instruction]),
            &evaluation,
            &MetalCompileOptions::compute(0),
            [64, 1, 1],
        )
        .expect("sampled image should lower");
        assert_eq!(shader.bindings.texture_count, 1);
        assert_eq!(shader.bindings.sampler_count, 1);
        assert!(shader.source.contains("Texture2D<float4> tex0"));
        assert!(shader.source.contains("tex0.SampleLevel(remu_sampler_0"));
        let dxil = compile_guest_shader(&shader, "remu_guest_sample_test.hlsl")
            .expect("sampled-texture HLSL should compile");
        assert!(dxil.len() > 128);
    }

    #[test]
    fn guest_storage_texture_load_and_store_compile_to_dxil() {
        let control = ImageControl {
            dmask: 0xf,
            vector_address: 0,
            address_registers: vec![0, 1],
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
        let image = |pc, opcode, vector_data| Instruction {
            pc,
            encoding: ShaderEncoding::Mimg,
            opcode: Opcode::Known(opcode),
            words: vec![0, 0],
            sources: vec![Operand::vector(0), Operand::vector(1), Operand::scalar(0)],
            destinations: (vector_data..vector_data + 4)
                .map(Operand::vector)
                .collect(),
            control: Some(InstructionControl::Image(ImageControl {
                vector_data,
                ..control.clone()
            })),
        };
        let load = image(0, "ImageLoad", 4);
        let store = image(8, "ImageStore", 4);
        let mut evaluation = empty_evaluation([0_u32; 8]);
        for (pc, opcode) in [(0, "ImageLoad"), (8, "ImageStore")] {
            evaluation.image_bindings.push(ImageBinding {
                pc,
                opcode: Opcode::Known(opcode),
                control: ImageControl {
                    vector_data: 4,
                    ..control.clone()
                },
                resource_descriptor: vec![0; 8].into(),
                sampler_descriptor: Vec::new().into(),
                mip_level: None,
            });
        }
        let shader = translate_guest_shader(
            &Program::new(0x8000, vec![load, store]),
            &evaluation,
            &MetalCompileOptions::compute(0),
            [64, 1, 1],
        )
        .expect("storage images should lower");
        assert_eq!(shader.bindings.texture_count, 2);
        assert!(shader.source.contains("Texture2D<float4> tex0"));
        assert!(shader.source.contains("RWTexture2D<float4> tex1"));
        assert!(shader.source.contains("tex1[uint2"));
        let dxil = compile_guest_shader(&shader, "remu_guest_storage_image_test.hlsl")
            .expect("storage-image HLSL should compile");
        assert!(dxil.len() > 128);
    }
}
