// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use metal::{CompileOptions, Device, Library};
use sha1::{Digest, Sha1};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub(crate) const METAL_LIBRARY_CACHE_FORMAT: &[u8] = b"remu-metal-library-v1\0";

/// Header-only metallibs from failed offline links are ~92 bytes and load successfully but expose
/// no entry points. Reject them before cache hits poison every subsequent compile attempt.
pub(crate) const MIN_VALID_METALLIB_BYTES: u64 = 256;

pub(crate) fn metallib_bytes_are_usable(path: &Path) -> bool {
    path.metadata()
        .map(|metadata| metadata.len() >= MIN_VALID_METALLIB_BYTES)
        .unwrap_or(false)
}

pub(crate) fn invalidate_metal_library_cache(source: &str) {
    let cache_path = metal_library_cache_path(source);
    if cache_path.is_file() {
        let _ = std::fs::remove_file(cache_path);
    }
}

fn compile_metal_library_in_process(
    device: &Device,
    stage: &str,
    source: &str,
) -> Result<Library, String> {
    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(false);
    device
        .new_library_with_source(source, &compile_options)
        .map_err(|error| format!("Metal {stage} compilation: {error}"))
}

/// Preserves MSL that Metal refused to compile, so the compiler's line-numbered error keeps
/// referring to text that still exists. The success-path dump only fires after both lowerings
/// succeed, which is exactly never for the sources that need reading.
fn dump_failed_metal_source(stage: &str, source: &str) {
    let Some(directory) = std::env::var_os("REMU_DUMP_MSL_DIR") else {
        return;
    };
    let mut digest = Sha1::new();
    digest.update(source.as_bytes());
    let digest = digest.finalize();
    let path = PathBuf::from(directory).join(format!(
        "failed-{stage}-{:02x}{:02x}{:02x}{:02x}.metal",
        digest[0], digest[1], digest[2], digest[3]
    ));
    match std::fs::write(&path, source) {
        Ok(()) => eprintln!(
            "[REMu][METAL] wrote failing {stage} MSL to {}",
            path.display()
        ),
        Err(error) => eprintln!(
            "[REMu][METAL] could not write failing {stage} MSL to {}: {error}",
            path.display()
        ),
    }
}

/// Resolves the offline Metal compiler once for the process (plan F1).
///
/// The persistent metallib and pipeline caches are default-on: the compiler is discovered from
/// the executable's ancestry (`.remu/toolchains/…/usr/bin/metal`, the layout `run-game.sh`
/// bundles) without configuration. `REMU_METAL_COMPILER` still overrides the path explicitly,
/// and `REMU_METAL_COMPILER=0` survives one cycle as the kill-switch selecting in-process
/// compilation with no persistent library cache.
pub(crate) fn offline_metal_compiler() -> Option<&'static Path> {
    static COMPILER: OnceLock<Option<PathBuf>> = OnceLock::new();
    COMPILER
        .get_or_init(|| {
            if let Some(value) = std::env::var_os("REMU_METAL_COMPILER") {
                if value == "0" {
                    eprintln!(
                        "[REMu][METAL] offline compiler disabled by kill-switch; persistent metallib cache off"
                    );
                    return None;
                }
                return Some(PathBuf::from(value));
            }
            let discovered = discover_bundled_metal_compiler();
            match &discovered {
                Some(path) => eprintln!(
                    "[REMu][METAL] discovered bundled offline Metal compiler {}",
                    path.display()
                ),
                None => eprintln!(
                    "[REMu][METAL] no offline Metal compiler found; compiling in-process without a persistent metallib cache"
                ),
            }
            discovered
        })
        .as_deref()
}

/// Walks the executable's ancestry for the bundled toolchain layout `run-game.sh` unpacks.
fn discover_bundled_metal_compiler() -> Option<PathBuf> {
    let executable = std::env::current_exe().ok()?;
    let mut directory = executable.parent()?.to_path_buf();
    loop {
        let toolchains = directory.join(".remu/toolchains");
        if toolchains.is_dir() {
            let mut candidates = vec![toolchains.join("Metal.xctoolchain/usr/bin/metal")];
            if let Ok(entries) = std::fs::read_dir(&toolchains) {
                let mut versioned = entries
                    .flatten()
                    .filter(|entry| {
                        entry.file_name().to_string_lossy().starts_with("Metal-")
                    })
                    .map(|entry| entry.path().join("Metal.xctoolchain/usr/bin/metal"))
                    .collect::<Vec<_>>();
                versioned.sort();
                candidates.extend(versioned);
            }
            for candidate in candidates {
                if candidate.is_file() && candidate.with_file_name("metallib").is_file() {
                    return Some(candidate);
                }
            }
        }
        if !directory.pop() {
            return None;
        }
    }
}

pub(crate) fn compile_metal_shader_library(
    device: &Device,
    stage: &str,
    source: &str,
) -> Result<Library, String> {
    let result = if offline_metal_compiler().is_some() {
        match compile_metal_library_with_tools(device, stage, source) {
            Ok(library) => Ok(library),
            Err(error) if metal_compile_failure_is_transient(&error) => Err(error),
            Err(error) => {
                eprintln!(
                    "[REMu][METAL] offline {stage} compile failed ({error}); falling back to in-process"
                );
                compile_metal_library_in_process(device, stage, source)
            }
        }
    } else {
        compile_metal_library_in_process(device, stage, source)
    };
    if result.is_err() {
        dump_failed_metal_source(stage, source);
    }
    result
}

pub(crate) fn compile_metal_compute_library(
    device: &Device,
    source: &str,
) -> Result<Library, String> {
    if offline_metal_compiler().is_some() {
        match compile_metal_library_with_tools(device, "compute", source) {
            Ok(library) => Ok(library),
            Err(error) if metal_compile_failure_is_transient(&error) => Err(error),
            Err(error) => {
                eprintln!(
                    "[REMu][METAL] offline compute compile failed ({error}); falling back to in-process"
                );
                compile_metal_library_in_process(device, "compute", source)
            }
        }
    } else {
        compile_metal_library_in_process(device, "compute", source)
    }
}

pub(crate) fn compile_metal_library_with_tools(
    device: &Device,
    stage: &str,
    source: &str,
) -> Result<Library, String> {
    static NEXT_COMPILE_DIRECTORY: AtomicUsize = AtomicUsize::new(0);
    static CACHE_HITS: AtomicUsize = AtomicUsize::new(0);
    static REPORTED_MULTICALL_LINKER: AtomicBool = AtomicBool::new(false);
    let compiler = canonical_tool_path(
        offline_metal_compiler()
            .map(Path::to_path_buf)
            .ok_or("no offline Metal compiler is configured or bundled")?,
    );
    if !compiler.is_file() {
        return Err(format!("{} is not a file", compiler.display()));
    }
    // `metallib` in Apple's downloadable Metal toolchain is a multicall symlink to `air-lld`.
    // Keep that final path component intact: the linker selects its legacy metallib mode from
    // argv[0]. Canonicalizing the symlink invokes it as `air-lld`, which accepts the AIR input but
    // emits only a 92-byte header with no entry points.
    let linker = metallib_tool_path(&compiler);
    if !linker.is_file() {
        return Err(format!("{} is not a file", linker.display()));
    }
    if !REPORTED_MULTICALL_LINKER.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[REMu][METAL] offline linker selected through metallib multicall entry path={}",
            linker.display()
        );
    }
    let cache_path = metal_library_cache_path(source);
    if cache_path.is_file() {
        if !metallib_bytes_are_usable(&cache_path) {
            let bytes = cache_path
                .metadata()
                .map(|metadata| metadata.len())
                .unwrap_or(0);
            eprintln!(
                "[REMu][METAL] discarding truncated cached {stage} library {} ({bytes} bytes)",
                cache_path.display()
            );
            let _ = std::fs::remove_file(&cache_path);
        } else {
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
                    // Not a compile failure: the eviction falls through to a fresh compile that
                    // usually succeeds, and counting it here made [PERF-PIPELINE] failed:N
                    // disagree with the set of genuinely failing shader pairs.
                    eprintln!(
                        "[REMu][METAL] discarding unreadable cached {stage} library {}: {error}",
                        cache_path.display()
                    );
                    let _ = std::fs::remove_file(&cache_path);
                }
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
    if !metallib_bytes_are_usable(&library_path) {
        let bytes = library_path
            .metadata()
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        return Err(format!(
            "{stage} library link produced an unusable metallib ({bytes} bytes)"
        ));
    }
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

fn metallib_tool_path(compiler: &Path) -> PathBuf {
    compiler.with_file_name("metallib")
}

#[cfg(test)]
mod tests {
    use super::{
        MetalCompileDirectory, canonical_tool_path, discover_bundled_metal_compiler,
        metallib_tool_path,
    };
    use std::ffi::OsStr;
    use std::fs::File;
    use std::os::unix::fs::symlink;
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn bundled_metal_compiler_is_discovered_without_configuration() {
        // Plan F1: the persistent metallib/function caches are default-on, so the offline
        // compiler must be found from the executable's ancestry, not from the environment.
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("workspace root");
        if !repo.join(".remu/toolchains").is_dir() {
            eprintln!("skipped: this checkout carries no bundled Metal toolchain");
            return;
        }
        let compiler = discover_bundled_metal_compiler()
            .expect("a checkout with .remu/toolchains discovers its compiler");
        assert_eq!(compiler.file_name(), Some(OsStr::new("metal")));
        assert!(
            compiler.with_file_name("metallib").is_file(),
            "the discovered compiler must sit beside its metallib linker"
        );
    }

    #[test]
    fn metallib_tool_path_preserves_the_multicall_symlink_name() {
        static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "remu-metallib-path-test-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).expect("create linker path test directory");
        let directory = MetalCompileDirectory(directory);
        let compiler = directory.0.join("metal");
        let linker_target = directory.0.join("air-lld");
        File::create(&compiler).expect("create compiler placeholder");
        File::create(&linker_target).expect("create linker placeholder");
        symlink("air-lld", directory.0.join("metallib")).expect("create multicall linker symlink");

        let compiler = canonical_tool_path(compiler);
        let linker = metallib_tool_path(&compiler);
        assert_eq!(linker.file_name(), Some(OsStr::new("metallib")));
        assert!(
            std::fs::symlink_metadata(&linker)
                .expect("inspect linker path")
                .file_type()
                .is_symlink(),
            "the executable path must retain the multicall symlink"
        );
        assert_eq!(
            std::fs::canonicalize(&linker).expect("resolve linker symlink"),
            std::fs::canonicalize(linker_target).expect("resolve linker target")
        );
    }
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
