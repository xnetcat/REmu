// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::PIPELINE_COMPILE_FAILED;
use metal::{CompileOptions, Device, Library};
use sha1::{Digest, Sha1};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) const METAL_LIBRARY_CACHE_FORMAT: &[u8] = b"remu-metal-library-v1\0";

pub(crate) fn compile_metal_shader_library(
    device: &Device,
    stage: &str,
    source: &str,
) -> Result<Library, String> {
    if std::env::var_os("REMU_METAL_COMPILER").is_some() {
        return compile_metal_library_with_tools(device, stage, source);
    }

    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(false);
    device
        .new_library_with_source(source, &compile_options)
        .map_err(|error| format!("Metal {stage} compilation: {error}"))
}

pub(crate) fn compile_metal_compute_library(
    device: &Device,
    source: &str,
) -> Result<Library, String> {
    if std::env::var_os("REMU_METAL_COMPILER").is_some() {
        return compile_metal_library_with_tools(device, "compute", source);
    }
    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(false);
    device
        .new_library_with_source(source, &compile_options)
        .map_err(|error| format!("Metal compute compilation: {error}"))
}

pub(crate) fn compile_metal_library_with_tools(
    device: &Device,
    stage: &str,
    source: &str,
) -> Result<Library, String> {
    static NEXT_COMPILE_DIRECTORY: AtomicUsize = AtomicUsize::new(0);
    static CACHE_HITS: AtomicUsize = AtomicUsize::new(0);
    let compiler = canonical_tool_path(
        std::env::var_os("REMU_METAL_COMPILER")
            .map(PathBuf::from)
            .ok_or("REMU_METAL_COMPILER is not configured")?,
    );
    if !compiler.is_file() {
        return Err(format!("{} is not a file", compiler.display()));
    }
    let linker = canonical_tool_path(compiler.with_file_name("metallib"));
    if !linker.is_file() {
        return Err(format!("{} is not a file", linker.display()));
    }
    let cache_path = metal_library_cache_path(source);
    if cache_path.is_file() {
        match device.new_library_with_file(&cache_path) {
            Ok(library) => {
                let hit = CACHE_HITS.fetch_add(1, Ordering::Relaxed) + 1;
                if hit <= 8 || hit.is_power_of_two() {
                    eprintln!(
                        "[REMu][METAL] persistent shader cache hit={hit} stage={stage} path={}",
                        cache_path.display()
                    );
                }
                return Ok(library);
            }
            Err(error) => {
                PIPELINE_COMPILE_FAILED.fetch_add(1, Ordering::Relaxed);
                eprintln!(
                    "[REMu][METAL] discarding unreadable cached {stage} library {}: {error}",
                    cache_path.display()
                );
                let _ = std::fs::remove_file(&cache_path);
            }
        }
    }
    let sequence = NEXT_COMPILE_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    let directory =
        std::env::temp_dir().join(format!("remu-metal-{}-{sequence}", std::process::id()));
    std::fs::create_dir(&directory)
        .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
    let directory = MetalCompileDirectory(directory);
    let source_path = directory.0.join(format!("{stage}.metal"));
    let air_path = directory.0.join(format!("{stage}.air"));
    let library_path = directory.0.join(format!("{stage}.metallib"));
    std::fs::write(&source_path, source)
        .map_err(|error| format!("could not write {}: {error}", source_path.display()))?;
    run_metal_tool(
        Command::new(&compiler)
            .current_dir(&directory.0)
            .arg("-fmetal-math-mode=safe")
            .arg("-c")
            .arg(&source_path)
            .arg("-o")
            .arg(&air_path),
        &format!("{stage} compilation"),
    )?;
    run_metal_tool(
        Command::new(&linker)
            .current_dir(&directory.0)
            .arg(&air_path)
            .arg("-o")
            .arg(&library_path),
        &format!("{stage} library link"),
    )?;
    if let Some(parent) = cache_path.parent()
        && let Err(error) = std::fs::create_dir_all(parent)
    {
        eprintln!(
            "[REMu][METAL] could not create persistent shader cache {}: {error}",
            parent.display()
        );
    } else if let Err(error) = std::fs::rename(&library_path, &cache_path) {
        eprintln!(
            "[REMu][METAL] could not persist {stage} shader library {}: {error}",
            cache_path.display()
        );
    }
    let load_path = if cache_path.is_file() {
        &cache_path
    } else {
        &library_path
    };
    device
        .new_library_with_file(load_path)
        .map_err(|error| format!("could not load {}: {error}", load_path.display()))
}

pub(crate) fn metal_library_cache_path(source: &str) -> PathBuf {
    let mut hasher = Sha1::new();
    hasher.update(METAL_LIBRARY_CACHE_FORMAT);
    hasher.update(source.as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    metal_cache_root().join(format!("{digest}.metallib"))
}

pub(crate) fn metal_cache_root() -> PathBuf {
    std::env::var_os("REMU_METAL_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join("Library/Caches/REMu/metal"))
        })
        .unwrap_or_else(|| std::env::temp_dir().join("remu-metal-cache"))
}

/// Offline Metal tool spawn failures are often transient under concurrent worker load.
/// Treating them as terminal permanently poisons shader pairs that the title submits every frame.
pub(crate) fn metal_compile_failure_is_transient(error: &str) -> bool {
    error.contains("could not start ")
        || error.contains("could not create ")
        || error.contains("could not write ")
}

fn canonical_tool_path(path: PathBuf) -> PathBuf {
    std::fs::canonicalize(&path).unwrap_or(path)
}

pub(crate) fn run_metal_tool(command: &mut Command, operation: &str) -> Result<(), String> {
    let output = command
        .output()
        .map_err(|error| format!("could not start {operation}: {error}"))?;
    if output.status.success() {
        return Ok(());
    }
    let diagnostics = String::from_utf8_lossy(&output.stderr);
    Err(format!(
        "{operation} exited with {}: {}",
        output.status,
        diagnostics.trim()
    ))
}

pub(crate) struct MetalCompileDirectory(PathBuf);

impl Drop for MetalCompileDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(crate) fn dump_shader_pair_msl(
    directory: &Path,
    export_address: u64,
    pixel_address: u64,
    vertex_source: &str,
    pixel_source: &str,
) -> Result<(), String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
    let stem = format!("es-{export_address:x}-ps-{pixel_address:x}");
    let vertex_path = directory.join(format!("{stem}-vertex.metal"));
    let pixel_path = directory.join(format!("{stem}-pixel.metal"));
    std::fs::write(&vertex_path, vertex_source)
        .map_err(|error| format!("could not write {}: {error}", vertex_path.display()))?;
    std::fs::write(&pixel_path, pixel_source)
        .map_err(|error| format!("could not write {}: {error}", pixel_path.display()))?;
    Ok(())
}

pub(crate) fn dump_shader_pair_ir(
    directory: &Path,
    export_address: u64,
    pixel_address: u64,
    export: &[remu_shader::Instruction],
    pixel: &[remu_shader::Instruction],
) -> Result<(), String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
    let stem = format!("es-{export_address:x}-ps-{pixel_address:x}");
    let path = directory.join(format!("{stem}-ir.txt"));
    let contents = format!("export={export:#?}\n\npixel={pixel:#?}\n");
    std::fs::write(&path, contents)
        .map_err(|error| format!("could not write {}: {error}", path.display()))
}

pub(crate) fn dump_compute_shader_msl(
    directory: &Path,
    compute_address: u64,
    instructions: &[remu_shader::Instruction],
    source: &str,
) -> Result<(), String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
    let source_path = directory.join(format!("cs-{compute_address:x}.metal"));
    std::fs::write(&source_path, source)
        .map_err(|error| format!("could not write {}: {error}", source_path.display()))?;
    let ir_path = directory.join(format!("cs-{compute_address:x}-ir.txt"));
    std::fs::write(&ir_path, format!("{instructions:#?}\n"))
        .map_err(|error| format!("could not write {}: {error}", ir_path.display()))
}
