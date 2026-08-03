// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Compiles the vendored AJM/AMPR codecs into the runtime:
//! - LibAtrac9 (plan P1) — ATRAC9
//! - minimp3 (plan P6) — MP3
//! - ooz + SIMDe (plan P3) — Kraken/Oodle decompress
//! - libav bridge (plan P5) — in-process AvPlayer/Videodec2 decode against system ffmpeg
//!
//! Third-party C/C++ is compiled with `.warnings(false)` so it cannot perturb the workspace
//! warning budget. The `cc` crate picks up the active target, so the same sources build for
//! arm64 and x86_64 (ooz SSE goes through SIMDe on Apple Silicon).

use std::path::Path;
use std::process::Command;

fn main() {
    build_atrac9();
    build_minimp3();
    build_ooz();
    build_libav_bridge();
    println!("cargo:rerun-if-changed=build.rs");
}

fn build_atrac9() {
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
    println!("cargo:rerun-if-changed=vendored/libatrac9/src");
}

fn build_minimp3() {
    let dir = Path::new("vendored/minimp3");
    cc::Build::new()
        .file(dir.join("minimp3_impl.c"))
        .include(dir)
        .warnings(false)
        .flag_if_supported("-std=c11")
        .compile("minimp3");
    println!("cargo:rerun-if-changed=vendored/minimp3");
}

fn build_ooz() {
    let dir = Path::new("vendored/ooz");
    let simde = Path::new("vendored/simde");
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .include(dir)
        .include(simde)
        .warnings(false)
        .file(dir.join("bitknit.cpp"))
        .file(dir.join("lzna.cpp"))
        .file(dir.join("kraken.cpp"))
        .file(dir.join("ooz_bridge.cpp"));
    // Apple clang needs libc++ explicitly when mixing into a Rust cdylib/staticlib.
    if std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref() == Some("macos") {
        build.cpp_set_stdlib("c++");
    }
    build.compile("ooz");
    println!("cargo:rerun-if-changed=vendored/ooz");
    println!("cargo:rerun-if-changed=vendored/simde");
}

fn build_libav_bridge() {
    let dir = Path::new("vendored/libav_bridge");
    let mut build = cc::Build::new();
    build
        .file(dir.join("remu_libav_bridge.c"))
        .include(dir)
        .warnings(false)
        .flag_if_supported("-std=c11");

    let host = std::env::var("HOST").unwrap_or_default();
    let target = std::env::var("TARGET").unwrap_or_default();
    let cross = host != target;
    let linked = !cross && try_link_system_libav(&mut build);
    if !linked {
        build.define("REMU_LIBAV_STUB", None);
        println!(
            "cargo:warning=libav bridge stubbed (cross-target or system ffmpeg unavailable)"
        );
    }
    build.compile("remu_libav_bridge");
    println!("cargo:rerun-if-changed=vendored/libav_bridge");
}

fn try_link_system_libav(build: &mut cc::Build) -> bool {
    let Ok(output) = Command::new("pkg-config")
        .args([
            "--cflags",
            "--libs",
            "libavformat",
            "libavcodec",
            "libavutil",
            "libswscale",
            "libswresample",
        ])
        .output()
    else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    let Ok(flags) = String::from_utf8(output.stdout) else {
        return false;
    };
    for token in flags.split_whitespace() {
        if let Some(include) = token.strip_prefix("-I") {
            build.include(include);
        } else if let Some(lib_dir) = token.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={lib_dir}");
        } else if let Some(lib) = token.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
    true
}
