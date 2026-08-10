// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Q5: deriving Metal mesh-pipeline capacities from real guest NGG registers.
//!
//! Every derived quantity mirrors an exact SDK register bit layout
//! (`sdk/target/include/agc/registerstructs.h`); the assertions pin those layouts so a future
//! caller in `remu-gpu-metal` can plumb the raw guest dwords without re-deriving the math.

use super::*;

#[test]
fn pgm_rsrc1_vgprs_follows_the_sdk_shshaderresource1_layout() {
    // SDK `ShShaderResource1Cs::getVgprs()` (registerstructs.h:1504-1507):
    // `((VGPRS & 0x3f) + 1) << 3`, and the default field value is 0x8 (registerstructs.h:1427).
    assert_eq!(
        NggRegisters {
            pgm_rsrc1: 0x8,
            pgm_rsrc2: 0,
            gs_max_vert_out: 0,
            ge_max_output_per_subgroup: 0,
            esgs_ring_itemsize: 0,
            primgen_passthrough: false,
        }
        .allocated_vgprs_wave32(),
        72,
        "the SDK default VGPRS field 0x8 allocates (8+1)<<3 = 72 wave32 registers"
    );
    assert_eq!(
        NggRegisters {
            pgm_rsrc1: 0x0,
            pgm_rsrc2: 0,
            gs_max_vert_out: 0,
            ge_max_output_per_subgroup: 0,
            esgs_ring_itemsize: 0,
            primgen_passthrough: false,
        }
        .allocated_vgprs_wave32(),
        8,
        "VGPRS field 0 still allocates one granule of eight registers"
    );
    // Only the low six bits participate; higher PGM_RSRC1 bits (priority, float mode, WGP mode)
    // must not perturb the VGPR count.
    assert_eq!(
        NggRegisters {
            pgm_rsrc1: 0xFFFF_FFC0 | 0x3f,
            pgm_rsrc2: 0,
            gs_max_vert_out: 0,
            ge_max_output_per_subgroup: 0,
            esgs_ring_itemsize: 0,
            primgen_passthrough: false,
        }
        .allocated_vgprs_wave32(),
        512,
        "the maximum 6-bit VGPRS field allocates (63+1)<<3 = 512 wave32 registers"
    );
}

#[test]
fn gs_max_vert_out_is_the_eleven_bit_vgt_field() {
    // SDK `CxGsMaxVertOut::getVertices()` (registerstructs.h:13165-13168): `value & 0x7ff`.
    for (raw, expected) in [
        (0u32, 0u32),
        (64, 64),
        (0x7ff, 2047),
        (0xFFFF_F800 | 64, 64),
    ] {
        assert_eq!(
            NggRegisters {
                pgm_rsrc1: 0,
                pgm_rsrc2: 0,
                gs_max_vert_out: raw,
                ge_max_output_per_subgroup: 0,
                esgs_ring_itemsize: 0,
                primgen_passthrough: false,
            }
            .gs_max_output_vertices(),
            expected,
            "VGT_GS_MAX_VERT_OUT is an 11-bit vertex count; high bits are reserved"
        );
    }
}

#[test]
fn subgroup_vertex_cap_is_the_ten_bit_ge_max_output_field() {
    // SDK `CxGeMaxOutputPerSubgroup::getMaxVertsPerSubgroup()` (registerstructs.h:6241-6246):
    // `value & 0x3ff`.
    for (raw, expected) in [
        (0u32, 0u32),
        (96, 96),
        (0x3ff, 1023),
        (0xFFFF_FC00 | 128, 128),
    ] {
        assert_eq!(
            NggRegisters {
                pgm_rsrc1: 0,
                pgm_rsrc2: 0,
                gs_max_vert_out: 0,
                ge_max_output_per_subgroup: raw,
                esgs_ring_itemsize: 0,
                primgen_passthrough: false,
            }
            .max_verts_per_subgroup(),
            expected,
            "GE_MAX_OUTPUT_PER_SUBGROUP caps per-subgroup output vertices in a 10-bit field"
        );
    }
}

#[test]
fn pgm_rsrc2_lds_size_follows_the_sdk_shshaderresource2gs_layout() {
    // SDK `ShShaderResource2Gs::getLdsSize()` (registerstructs.h:12154-12157):
    // `((m_regs[0].m_value & 0x07f80000u) >> 12)`. The result is a count of dwords,
    // so the byte reservation is four times that value.
    for (field, bytes) in [
        (0u32, 0u32),
        (1, 512),
        (0x11, 8704),
        (0x40, 32768),
        (0x80, 65536),
        (0xff, 130560),
    ] {
        assert_eq!(
            NggRegisters {
                pgm_rsrc1: 0,
                pgm_rsrc2: field << 19,
                gs_max_vert_out: 0,
                ge_max_output_per_subgroup: 0,
                esgs_ring_itemsize: 0,
                primgen_passthrough: false,
            }
            .lds_size_bytes(),
            bytes,
            "GS LDS field {field:#x} must decode to {bytes} bytes"
        );
    }
    assert_eq!(
        NggRegisters {
            pgm_rsrc1: 0,
            pgm_rsrc2: 0xF807_FFFF | (0x40 << 19),
            gs_max_vert_out: 0,
            ge_max_output_per_subgroup: 0,
            esgs_ring_itemsize: 0,
            primgen_passthrough: false,
        }
        .lds_size_bytes(),
        32768,
        "non-LDS RSRC2 bits must not change the decoded LDS byte count"
    );
}

#[test]
fn silent_hill_lut_draw_lds_covers_its_captured_vertex_payload() {
    // The title LUT mesh uses raw GS_RSRC2 0x008b0000 and accesses vertex payload through at
    // least byte 0x1d20. Treating the SDK getter's dword count as bytes truncated this allocation
    // to 2176 bytes, suppressed every guarded payload write/read, and emitted (0, 0, 0, 0)
    // positions for all vertices.
    let bytes = NggRegisters {
        pgm_rsrc1: 0,
        pgm_rsrc2: 0x008b_0000,
        gs_max_vert_out: 0,
        ge_max_output_per_subgroup: 0,
        esgs_ring_itemsize: 0,
        primgen_passthrough: false,
    }
    .lds_size_bytes();
    assert_eq!(bytes, 8704);
    assert!(bytes > 0x1d20);
}

#[test]
fn mesh_options_derive_capacities_and_vgpr_allocation_from_registers() {
    // A subgroup that outputs 96 vertices after amplification, a VGPRS field selecting 128
    // wave32 registers (`((15)+1)<<3`), and an oversized GS_MAX_VERT_OUT the subgroup cap should
    // win over. LDS field 0x08 → 4096 bytes from PGM_RSRC2.
    let registers = NggRegisters {
        pgm_rsrc1: 15,
        pgm_rsrc2: 0x08 << 19,
        gs_max_vert_out: 200,
        ge_max_output_per_subgroup: 96,
        esgs_ring_itemsize: 5,
        primgen_passthrough: false,
    };
    let options = MetalCompileOptions::mesh_from_ngg_registers(4, 64, registers);
    assert_eq!(options.stage, MetalShaderStage::Mesh);
    assert_eq!(options.lds_size_bytes, 4096);
    assert_eq!(
        options.mesh_max_vertex_count, 96,
        "the per-subgroup vertex cap sizes the Metal vertex records"
    );
    assert_eq!(
        options.mesh_max_primitive_count, 96,
        "primitive capacity is bounded by the vertex capacity (no per-subgroup prim field \
         exists in the SDK register structs)"
    );
    assert_eq!(options.mesh_threadgroup_size, 64);
    assert_eq!(options.ngg_esgs_ring_itemsize, 5);
    assert_eq!(
        options.allocated_vector_register_count, 128,
        "PGM_RSRC1.VGPRS = 15 allocates (15+1)<<3 = 128 wave32 registers"
    );
}

#[test]
fn zero_subgroup_cap_falls_back_to_gs_max_vert_out() {
    // A stage that never programmed GE_MAX_OUTPUT_PER_SUBGROUP (field 0) still bounds its Metal records by
    // VGT_GS_MAX_VERT_OUT rather than collapsing to an invalid zero-capacity mesh pipeline.
    let registers = NggRegisters {
        pgm_rsrc1: 0x8,
        pgm_rsrc2: 0,
        gs_max_vert_out: 128,
        ge_max_output_per_subgroup: 0,
        esgs_ring_itemsize: 0,
        primgen_passthrough: false,
    };
    let options = MetalCompileOptions::mesh_from_ngg_registers(0, 32, registers);
    assert_eq!(options.mesh_max_vertex_count, 128);
    assert_eq!(options.mesh_max_primitive_count, 128);
}

#[test]
fn derived_capacities_stay_inside_the_metal_validation_window() {
    // A 512-VGPR allocation and a 1023-vertex subgroup both exceed the Metal-side maxima the
    // driver validates (1..=256). Derivation must clamp so the options always pass
    // `validate_interface`, never produce an out-of-range pipeline the caller cannot use.
    let registers = NggRegisters {
        pgm_rsrc1: 0x3f,
        pgm_rsrc2: 0,
        gs_max_vert_out: 0x7ff,
        ge_max_output_per_subgroup: 0x3ff,
        esgs_ring_itemsize: 0,
        primgen_passthrough: false,
    };
    let options = MetalCompileOptions::mesh_from_ngg_registers(0, 256, registers);
    assert!((1..=256).contains(&options.mesh_max_vertex_count));
    assert!((1..=256).contains(&options.mesh_max_primitive_count));
    assert!((1..=VECTOR_REGISTER_COUNT).contains(&options.allocated_vector_register_count));
    assert_eq!(
        options.allocated_vector_register_count,
        VECTOR_REGISTER_COUNT
    );
}
