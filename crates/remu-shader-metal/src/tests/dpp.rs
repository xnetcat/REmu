// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! DPP16/DPP8 modifier lowering tests: the accepted control space, lane
//! selection, activity modifiers and row/bank write masks.

use super::*;

#[test]
fn accepts_exactly_the_architectural_dpp16_control_space() {
    for control in 0..=0x1ff {
        let instruction = vector_instruction(
            "VMovB32",
            1,
            InstructionControl::Dpp(DppControl {
                control,
                fetch_inactive: false,
                bound_control: false,
                absolute_mask: 0,
                negate_mask: 0,
                bank_mask: 0xf,
                row_mask: 0xf,
            }),
        );
        let result = compile_msl(
            &Program::new(0x1000, vec![instruction]),
            &empty_evaluation(),
            &MetalCompileOptions::vertex(0),
        );
        assert_eq!(
            result.is_ok(),
            is_supported_dpp_control(control),
            "DPP control 0x{control:03X} acceptance differs from the ISA table: {result:?}"
        );
    }
}

#[test]
fn lowers_dpp_and_dpp8_for_every_otherwise_supported_vector_alu() {
    const ALU: [(&str, usize); 75] = [
        ("VCndmaskB32", 2),
        ("VMovB32", 1),
        ("VAddF32", 2),
        ("VSubF32", 2),
        ("VSubrevF32", 2),
        ("VMulF32", 2),
        ("VAddF16", 2),
        ("VSubF16", 2),
        ("VSubrevF16", 2),
        ("VMulF16", 2),
        ("VFmacF16", 2),
        ("VMaxF16", 2),
        ("VMinF16", 2),
        ("VLdexpF16", 2),
        ("VMinF32", 2),
        ("VMaxF32", 2),
        ("VMin3F32", 3),
        ("VMax3F32", 3),
        ("VMed3F32", 3),
        ("VMin3I32", 3),
        ("VMax3I32", 3),
        ("VMed3I32", 3),
        ("VMin3U32", 3),
        ("VMax3U32", 3),
        ("VMed3U32", 3),
        ("VSadU32", 3),
        ("VFmacF32", 2),
        ("VMacF32", 2),
        ("VFloorF32", 1),
        ("VCeilF32", 1),
        ("VTruncF32", 1),
        ("VRndneF32", 1),
        ("VFractF32", 1),
        ("VSqrtF32", 1),
        ("VRsqF32", 1),
        ("VRcpF32", 1),
        ("VRcpIflagF32", 1),
        ("VLogF32", 1),
        ("VExpF32", 1),
        ("VSinF32", 1),
        ("VCosF32", 1),
        ("VAndB32", 2),
        ("VAddI32", 2),
        ("VLshrrevB32", 2),
        ("VLshlrevB32", 2),
        ("VLshlAddU32", 3),
        ("VBfeU32", 3),
        ("VBfiB32", 3),
        ("VAdd3U32", 3),
        ("VOr3U32", 3),
        ("VAndOrB32", 3),
        ("VLshlOrU32", 3),
        ("VCvtF32I32", 1),
        ("VCvtF32U32", 1),
        ("VCvtF16F32", 1),
        ("VCvtF32F16", 1),
        ("VMinU32", 2),
        ("VMaxU32", 2),
        ("VMulLoU32", 2),
        ("VMulHiU32", 2),
        ("VMulHiI32", 2),
        ("VCvtU32F32", 1),
        ("VCvtI32F32", 1),
        ("VCvtOffF32I4", 1),
        ("VCubeidF32", 3),
        ("VCubescF32", 3),
        ("VCubetcF32", 3),
        ("VCubemaF32", 3),
        ("VCvtPkrtzF16F32", 2),
        ("VMadF32", 3),
        ("VFmaF32", 3),
        ("VMadAkF32", 3),
        ("VMadMkF32", 3),
        ("VFmaAkF32", 3),
        ("VFmaMkF32", 3),
    ];
    const DPP_FAMILIES: [u32; 8] = [0xe4, 0x101, 0x111, 0x121, 0x140, 0x141, 0x150, 0x160];
    for (name, source_count) in ALU {
        for control in DPP_FAMILIES {
            compile_vector_instruction(
                vector_instruction(
                    name,
                    source_count,
                    InstructionControl::Dpp(DppControl {
                        control,
                        fetch_inactive: false,
                        bound_control: false,
                        absolute_mask: 0,
                        negate_mask: 0,
                        bank_mask: 0xf,
                        row_mask: 0xf,
                    }),
                ),
                MetalShaderStage::Compute,
            );
        }
        for fetch_inactive in [false, true] {
            compile_vector_instruction(
                vector_instruction(
                    name,
                    source_count,
                    InstructionControl::Dpp8(Dpp8Control {
                        lane_selectors: 0x00fa_c688,
                        fetch_inactive,
                    }),
                ),
                MetalShaderStage::Compute,
            );
        }
    }
}

#[test]
fn lowers_dpp_and_dpp8_for_all_decoded_vector_compares() {
    const FLOAT_RELATIONS: [&str; 16] = [
        "F", "Lt", "Eq", "Le", "Gt", "Lg", "Ge", "O", "U", "Nge", "Nlg", "Ngt", "Nle", "Neq",
        "Nlt", "Tru",
    ];
    const INTEGER_RELATIONS: [&str; 8] = ["F", "Lt", "Eq", "Le", "Gt", "Ne", "Ge", "T"];
    let mut names = Vec::new();
    for prefix in ["VCmp", "VCmpx"] {
        for relation in FLOAT_RELATIONS {
            names.push(format!("{prefix}{relation}F32"));
        }
        for suffix in ["I32", "U32"] {
            for relation in INTEGER_RELATIONS {
                names.push(format!("{prefix}{relation}{suffix}"));
            }
        }
    }
    names.push("VCmpClassF32".to_owned());
    for name in names {
        let name: &'static str = Box::leak(name.into_boxed_str());
        for control in [
            InstructionControl::Dpp(DppControl {
                control: 0x111,
                fetch_inactive: false,
                bound_control: false,
                absolute_mask: 0,
                negate_mask: 0,
                bank_mask: 0xf,
                row_mask: 0xf,
            }),
            InstructionControl::Dpp8(Dpp8Control {
                lane_selectors: 0x0005_3977,
                fetch_inactive: false,
            }),
        ] {
            let mut instruction = vector_instruction(name, 2, control);
            instruction.encoding = ShaderEncoding::Vopc;
            instruction.destinations.clear();
            compile_vector_instruction(instruction, MetalShaderStage::Compute);
        }
    }
}

#[test]
fn dpp_semantics_cover_lane_selection_activity_modifiers_and_write_masks() {
    let shader = compile_vector_instruction(
        vector_instruction(
            "VAddF32",
            2,
            InstructionControl::Dpp(DppControl {
                control: 0x111,
                fetch_inactive: false,
                bound_control: false,
                absolute_mask: 1,
                negate_mask: 2,
                bank_mask: 0xa,
                row_mask: 5,
            }),
        ),
        MetalShaderStage::Compute,
    );
    assert!(shader.source.contains("simd_shuffle(remu_v_1, (ushort)"));
    assert!(
        shader
            .source
            .contains("simd_shuffle(exec ? 1u : 0u, (ushort)")
    );
    assert!(shader.source.contains("(remu_lane & 15u) >= 1u"));
    assert!(shader.source.contains("fabs(as_type<float>"));
    assert!(shader.source.contains("-(as_type<float>(remu_v_2))"));
    assert!(shader.source.contains("5u >> (remu_lane >> 4)"));
    assert!(shader.source.contains("10u >> (remu_lane & 3u)"));
    assert!(shader.source.contains("? t"));
    assert!(shader.source.contains(": remu_v_8"));

    let fetch_inactive = compile_vector_instruction(
        vector_instruction(
            "VMovB32",
            1,
            InstructionControl::Dpp(DppControl {
                control: 0x101,
                fetch_inactive: true,
                bound_control: true,
                absolute_mask: 0,
                negate_mask: 0,
                bank_mask: 0xf,
                row_mask: 0xf,
            }),
        ),
        MetalShaderStage::Compute,
    );
    assert!(
        !fetch_inactive
            .source
            .contains("simd_shuffle(exec ? 1u : 0u")
    );
    assert!(fetch_inactive.source.contains("&& (true)"));
}

#[test]
fn dpp8_and_compare_masks_preserve_architectural_lane_state() {
    let dpp8 = compile_vector_instruction(
        vector_instruction(
            "VMovB32",
            1,
            InstructionControl::Dpp8(Dpp8Control {
                lane_selectors: 0x0005_3977,
                fetch_inactive: false,
            }),
        ),
        MetalShaderStage::Compute,
    );
    assert!(dpp8.source.contains("342391u >> ((remu_lane & 7u) * 3u)"));
    assert!(dpp8.source.contains("remu_lane & 0xFFFFFFF8u"));
    assert!(dpp8.source.contains("simd_shuffle(remu_v_1, (ushort)"));
    assert!(
        dpp8.source
            .contains("simd_shuffle(exec ? 1u : 0u, (ushort)")
    );

    let mut compare = vector_instruction(
        "VCmpEqF32",
        2,
        InstructionControl::Dpp(DppControl {
            control: 0xe4,
            fetch_inactive: false,
            bound_control: false,
            absolute_mask: 0,
            negate_mask: 0,
            bank_mask: 0,
            row_mask: 0,
        }),
    );
    compare.encoding = ShaderEncoding::Vopc;
    compare.destinations.clear();
    let compare = compile_vector_instruction(compare, MetalShaderStage::Vertex);
    assert!(compare.source.contains(": vcc"));
    assert!(compare.source.contains("remu_s_106 ="));
    assert!(compare.source.contains("remu_s_107 = 0u"));
}

#[test]
fn graphics_dpp_uses_converged_real_wave_operations() {
    let shader = compile_vector_instruction(
        vector_instruction(
            "VMovB32",
            1,
            InstructionControl::Dpp8(Dpp8Control {
                lane_selectors: 0x00fa_c688,
                fetch_inactive: false,
            }),
        ),
        MetalShaderStage::Vertex,
    );
    assert!(
        shader
            .source
            .contains("uint remu_lane = simd_prefix_exclusive_sum(1u)")
    );
    assert!(shader.source.contains("simd_shuffle(remu_v_1, (ushort)"));
    assert_eq!(shader.register_footprint.scalar_values, 0);
    let shuffle = shader
        .source
        .find("simd_shuffle(remu_v_1, (ushort)")
        .expect("DPP8 shuffle");
    let write = shader
        .source
        .find("if (exec) remu_v_8")
        .expect("EXEC-guarded destination write");
    assert!(
        shuffle < write,
        "wave shuffle must be converged before EXEC guard"
    );
}
