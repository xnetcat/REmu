// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Compiles the vendored LibAtrac9 decoder (see `vendored/libatrac9/PROVENANCE.md`) into the
//! runtime so the AJM engine can decode ATRAC9 (plan P1). The C is third-party, so its own
//! warnings are silenced and cannot perturb the workspace warning budget; the `cc` crate picks
//! up the active target, so the same sources build for arm64 and x86_64 alike.

use std::path::Path;

fn main() {
    let src = Path::new("vendored/libatrac9/src");
    let sources = [
        "band_extension.c",
        "bit_allocation.c",
        "bit_reader.c",
        "decinit.c",
        "decoder.c",
        "huffCodes.c",
        "imdct.c",
        "libatrac9.c",
        "quantization.c",
        "scale_factors.c",
        "tables.c",
        "unpack.c",
        "utility.c",
    ];

    let mut build = cc::Build::new();
    build
        .include(src)
        .warnings(false)
        .flag_if_supported("-std=c11");
    for source in sources {
        build.file(src.join(source));
    }
    build.compile("atrac9");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=vendored/libatrac9/src");
}
