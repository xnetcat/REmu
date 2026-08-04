// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/metal_exception.m");
    println!("cargo:rerun-if-env-changed=REMU_METAL_RUST_ONLY_CHECK");
    if std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref() != Some("macos") {
        return;
    }
    if std::env::var_os("REMU_METAL_RUST_ONLY_CHECK").is_some_and(|value| value == "1") {
        return;
    }
    cc::Build::new()
        .file("src/metal_exception.m")
        .warnings(false)
        .compile("remu_metal_exception");
}
