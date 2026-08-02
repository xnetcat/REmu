// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Poppy Slate glyph mesh UV fetch: userdata bitfields select FormatXy soffsets into the
//! interleaved vertex buffer. Live draws proved param0 must come from soffset 8 (UV), not 0 (POS).

#[test]
fn poppy_glyph_userdata_soffsets_select_uv_not_pos() {
    // Captured from PPSA20591 NGG glyph mesh draws (GLYPH-BIND attr_sim).
    let s24 = 0x2020_u32;
    let s25 = 0x22020_u32;
    // Live MSL: extract_bits(userdata, 14, 12) after loading vb0.
    let soff_uv = (s25 >> 14) & 0xfff;
    let soff_pos = (s24 >> 14) & 0xfff;
    assert_eq!(soff_uv, 8);
    assert_eq!(soff_pos, 0);

    // Interleaved stride-16 records: [POS.xy, UV.xy].
    let vb: [[f32; 4]; 4] = [
        [1.0, -1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, 0.0],
    ];
    let mut param0 = [(0.0_f32, 0.0_f32); 4];
    let mut pos = [(0.0_f32, 0.0_f32); 4];
    for (i, record) in vb.iter().enumerate() {
        let uv_base = (soff_uv / 4) as usize;
        let pos_base = (soff_pos / 4) as usize;
        param0[i] = (record[uv_base], record[uv_base + 1]);
        pos[i] = (record[pos_base], record[pos_base + 1]);
    }
    assert_eq!(
        param0,
        [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)],
        "param0 must be UV corners; taking xy@0 would export POS and solid-fill tofu"
    );
    assert_eq!(
        pos,
        [(1.0, -1.0), (1.0, 1.0), (-1.0, -1.0), (-1.0, 1.0)]
    );
}
