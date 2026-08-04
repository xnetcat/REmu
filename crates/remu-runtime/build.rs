// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Compiles the vendored AJM/AMPR codecs into the runtime:
//! - LibAtrac9 (plan P1) — ATRAC9
//! - minimp3 (plan P6) — MP3
//! - ooz + SIMDe (plan P3) — Kraken/Oodle decompress
//! - libav bridge (plan P5) — in-process AvPlayer/Videodec2 decode against pinned FFmpeg
//!
//! Third-party C/C++ is compiled with `.warnings(false)` so it cannot perturb the workspace
//! warning budget. The `cc` crate picks up the active target, so the same sources build for
//! arm64 and x86_64 (ooz SSE goes through SIMDe on Apple Silicon).

use std::path::{Path, PathBuf};
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
    let target = std::env::var("TARGET").expect("Cargo always provides TARGET to build scripts");
    let ffmpeg_root = ensure_ffmpeg(&target);
    let pkgconfig_dir = ffmpeg_root.join("lib/pkgconfig");
    let flags = libav_pkg_config_flags(&pkgconfig_dir, &target);
    let mut build = cc::Build::new();
    build
        .file(dir.join("remu_libav_bridge.c"))
        .include(dir)
        .include(ffmpeg_root.join("include"))
        .warnings(false)
        .flag_if_supported("-std=c11");
    for include in &flags.include_dirs {
        build.include(include);
    }
    for flag in &flags.compiler_flags {
        build.flag(flag);
    }
    build.compile("remu_libav_bridge");

    for search in &flags.library_dirs {
        println!("cargo:rustc-link-search=native={}", search.display());
    }
    for search in &flags.framework_dirs {
        println!("cargo:rustc-link-search=framework={}", search.display());
    }
    for library in &flags.libraries {
        if matches!(
            library.as_str(),
            "avformat" | "avcodec" | "avutil" | "swscale" | "swresample"
        ) {
            println!("cargo:rustc-link-lib=static={library}");
        } else {
            println!("cargo:rustc-link-lib={library}");
        }
    }
    for framework in &flags.frameworks {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
    println!("cargo:rustc-env=REMU_FFMPEG_VERSION=8.1.2");
    println!("cargo:rerun-if-env-changed=REMU_DEPS_DIR");
    println!("cargo:rerun-if-env-changed=REMU_FFMPEG_ROOT");
    println!("cargo:rerun-if-changed=../../scripts/lib/ensure-ffmpeg.sh");
    println!("cargo:rerun-if-changed=vendored/libav_bridge");
}

fn ensure_ffmpeg(target: &str) -> PathBuf {
    if let Some(root) = std::env::var_os("REMU_FFMPEG_ROOT") {
        let root = PathBuf::from(root);
        validate_ffmpeg_root(&root, target);
        return root;
    }

    let manifest_dir = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR")
            .expect("Cargo always provides CARGO_MANIFEST_DIR to build scripts"),
    );
    let script = manifest_dir.join("../../scripts/lib/ensure-ffmpeg.sh");
    let output = Command::new("sh")
        .arg(&script)
        .arg(target)
        .output()
        .unwrap_or_else(|error| {
            panic!(
                "failed to start bundled FFmpeg builder {}: {error}",
                script.display()
            )
        });
    if !output.status.success() {
        panic!(
            "bundled FFmpeg build failed for {target} (status {}):\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8(output.stdout)
        .expect("bundled FFmpeg builder returned a non-UTF-8 install path");
    let root = PathBuf::from(stdout.trim());
    validate_ffmpeg_root(&root, target);
    root
}

fn validate_ffmpeg_root(root: &Path, target: &str) {
    let required = [
        root.join("include/libavcodec/avcodec.h"),
        root.join("lib/libavformat.a"),
        root.join("lib/libavcodec.a"),
        root.join("lib/libavutil.a"),
        root.join("lib/libswscale.a"),
        root.join("lib/libswresample.a"),
    ];
    for path in required {
        assert!(
            path.is_file(),
            "FFmpeg root for {target} is incomplete: {} is missing",
            path.display()
        );
    }
}

#[derive(Default)]
struct PkgConfigFlags {
    include_dirs: Vec<PathBuf>,
    library_dirs: Vec<PathBuf>,
    framework_dirs: Vec<PathBuf>,
    libraries: Vec<String>,
    frameworks: Vec<String>,
    compiler_flags: Vec<String>,
}

fn libav_pkg_config_flags(pkgconfig_dir: &Path, target: &str) -> PkgConfigFlags {
    let output = Command::new("pkg-config")
        .args([
            "--static",
            "--cflags",
            "--libs",
            "libavformat",
            "libavcodec",
            "libavutil",
            "libswscale",
            "libswresample",
        ])
        .env("PKG_CONFIG_ALLOW_CROSS", "1")
        .env("PKG_CONFIG_LIBDIR", pkgconfig_dir)
        .output()
        .unwrap_or_else(|error| panic!("failed to run pkg-config for FFmpeg: {error}"));
    if !output.status.success() {
        panic!(
            "pkg-config rejected bundled FFmpeg for {target}:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let flags = String::from_utf8(output.stdout)
        .expect("pkg-config returned non-UTF-8 flags for bundled FFmpeg");
    parse_pkg_config_flags(&flags)
}

fn parse_pkg_config_flags(flags: &str) -> PkgConfigFlags {
    let mut parsed = PkgConfigFlags::default();
    let mut tokens = flags.split_whitespace();
    while let Some(token) = tokens.next() {
        if let Some(path) = token.strip_prefix("-I") {
            parsed.include_dirs.push(PathBuf::from(path));
        } else if let Some(path) = token.strip_prefix("-L") {
            parsed.library_dirs.push(PathBuf::from(path));
        } else if let Some(path) = token.strip_prefix("-F") {
            parsed.framework_dirs.push(PathBuf::from(path));
        } else if let Some(library) = token.strip_prefix("-l") {
            if !parsed.libraries.iter().any(|existing| existing == library) {
                parsed.libraries.push(library.to_owned());
            }
        } else if token == "-framework" {
            let framework = tokens
                .next()
                .expect("pkg-config emitted -framework without a framework name");
            if !parsed
                .frameworks
                .iter()
                .any(|existing| existing == framework)
            {
                parsed.frameworks.push(framework.to_owned());
            }
        } else if token == "-pthread" || token.starts_with("-D") {
            parsed.compiler_flags.push(token.to_owned());
        } else {
            panic!("unsupported bundled FFmpeg pkg-config flag: {token}");
        }
    }
    parsed
}
