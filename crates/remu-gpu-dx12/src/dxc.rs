// SPDX-License-Identifier: GPL-2.0-or-later

//! Runtime DirectX Shader Compiler discovery and Shader Model 6 compilation.

use std::path::{Path, PathBuf};

use hassle_rs::{Dxc, HassleError};
use sha1::{Digest, Sha1};

const DXCOMPILER_FILE: &str = "dxcompiler.dll";
const DXIL_CACHE_SCHEMA: &[u8] = b"remu-dxil-cache-v1";
const DXC_ARGUMENTS: &[&str] = &["-O3", "-Ges"];

pub(super) fn compile_hlsl(
    source_name: &str,
    source: &str,
    entry_point: &str,
    target_profile: &str,
) -> Result<Vec<u8>, String> {
    let (dxc, location) = load_dxc()?;
    let cache_path = dxil_cache_path(
        &location,
        source,
        entry_point,
        target_profile,
        DXC_ARGUMENTS,
    );
    if let Some(path) = &cache_path
        && let Ok(dxil) = std::fs::read(path)
        && valid_dxil_container(&dxil)
    {
        return Ok(dxil);
    }
    let compiler = dxc.create_compiler().map_err(|error| {
        format!(
            "could not create DXC compiler from {}: {error}",
            location.display()
        )
    })?;
    let library = dxc.create_library().map_err(|error| {
        format!(
            "could not create DXC library from {}: {error}",
            location.display()
        )
    })?;
    let source_blob = library
        .create_blob_with_encoding_from_str(source)
        .map_err(|error| format!("could not create DXC source blob: {error}"))?;
    let result = compiler.compile(
        &source_blob,
        source_name,
        entry_point,
        target_profile,
        DXC_ARGUMENTS,
        None,
        &[],
    );
    match result {
        Ok(result) => result
            .get_result()
            .map(|blob| blob.to_vec())
            .map_err(|error| format!("DXC returned no DXIL object for {source_name}: {error}"))
            .map(|dxil| {
                if let Some(path) = cache_path {
                    persist_dxil(&path, &dxil);
                }
                dxil
            }),
        Err((result, hresult)) => {
            let diagnostics = result
                .get_error_buffer()
                .and_then(|blob| library.get_blob_as_string(&blob.into()))
                .unwrap_or_else(|_| "DXC returned no diagnostics".to_owned());
            Err(format!(
                "DXC failed compiling {source_name} ({target_profile}, HRESULT {hresult:?}): {}",
                diagnostics.trim()
            ))
        }
    }
}

fn dxil_cache_path(
    compiler: &Path,
    source: &str,
    entry_point: &str,
    target_profile: &str,
    arguments: &[&str],
) -> Option<PathBuf> {
    if std::env::var_os("REMU_DISABLE_DXIL_CACHE").is_some() {
        return None;
    }
    let root = std::env::var_os("REMU_DXIL_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("LOCALAPPDATA").map(PathBuf::from))
        .map(|root| {
            if std::env::var_os("REMU_DXIL_CACHE_DIR").is_some() {
                root
            } else {
                root.join("REmu").join("cache").join("dxil-v1")
            }
        })?;
    let mut hash = Sha1::new();
    hash.update(DXIL_CACHE_SCHEMA);
    hash.update(source.as_bytes());
    hash.update([0]);
    hash.update(entry_point.as_bytes());
    hash.update([0]);
    hash.update(target_profile.as_bytes());
    for argument in arguments {
        hash.update([0]);
        hash.update(argument.as_bytes());
    }
    if let Ok(metadata) = compiler.metadata() {
        hash.update(metadata.len().to_le_bytes());
        if let Ok(modified) = metadata.modified()
            && let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH)
        {
            hash.update(duration.as_secs().to_le_bytes());
            hash.update(duration.subsec_nanos().to_le_bytes());
        }
    }
    Some(root.join(format!("{:x}.dxil", hash.finalize())))
}

fn valid_dxil_container(bytes: &[u8]) -> bool {
    bytes.len() >= 32 && bytes.starts_with(b"DXBC")
}

fn persist_dxil(path: &Path, dxil: &[u8]) {
    static NEXT_TEMPORARY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let Some(parent) = path.parent() else { return };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let temporary_id = NEXT_TEMPORARY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temporary = path.with_extension(format!("{}-{temporary_id}.tmp", std::process::id()));
    if std::fs::write(&temporary, dxil).is_ok() {
        if std::fs::rename(&temporary, path).is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
    }
}

pub(super) fn find_dxcompiler() -> Result<PathBuf, String> {
    if let Some(configured) = std::env::var_os("REMU_DXCOMPILER") {
        let path = normalize_candidate(PathBuf::from(configured));
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "REMU_DXCOMPILER points to {}, but no dxcompiler.dll exists there",
            path.display()
        ));
    }

    if let Ok(executable) = std::env::current_exe() {
        if let Some(directory) = executable.parent() {
            let adjacent = directory.join(DXCOMPILER_FILE);
            if adjacent.is_file() {
                return Ok(adjacent);
            }
            for ancestor in directory.ancestors() {
                if ancestor.file_name().is_some_and(|name| name == "target") {
                    if let Some(path) = newest_development_dxc(ancestor) {
                        return Ok(path);
                    }
                    break;
                }
            }
        }
    }

    let current = PathBuf::from(DXCOMPILER_FILE);
    if current.is_file() {
        return Ok(current);
    }
    Err(format!(
        "could not find {DXCOMPILER_FILE}; set REMU_DXCOMPILER to the DLL or its directory, or place the matching DXC redistributables beside the REmu executable"
    ))
}

fn load_dxc() -> Result<(Dxc, PathBuf), String> {
    match find_dxcompiler() {
        Ok(path) => Dxc::new(Some(path.clone()))
            .map(|dxc| (dxc, path.clone()))
            .map_err(|error| format_load_error(&path, error)),
        Err(discovery_error) => {
            let fallback = PathBuf::from(DXCOMPILER_FILE);
            Dxc::new(None)
                .map(|dxc| (dxc, fallback.clone()))
                .map_err(|error| format!("{discovery_error}; loader fallback failed: {error}"))
        }
    }
}

fn normalize_candidate(path: PathBuf) -> PathBuf {
    if path.is_dir() {
        path.join(DXCOMPILER_FILE)
    } else {
        path
    }
}

fn newest_development_dxc(target: &Path) -> Option<PathBuf> {
    let tools = target.join("tools");
    let mut candidates = std::fs::read_dir(tools)
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("dxc-"))
        .map(|entry| entry.path().join("bin").join("x64").join(DXCOMPILER_FILE))
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.pop()
}

fn format_load_error(path: &Path, error: HassleError) -> String {
    format!(
        "could not load DirectX Shader Compiler from {}: {error}; keep its companion redistributables in the same directory",
        path.display()
    )
}

#[cfg(test)]
mod tests {
    use super::{normalize_candidate, valid_dxil_container};
    use std::path::PathBuf;

    #[test]
    fn explicit_directory_names_the_compiler_dll() {
        let directory = std::env::temp_dir();
        assert_eq!(
            normalize_candidate(directory.clone()),
            directory.join("dxcompiler.dll")
        );
    }

    #[test]
    fn explicit_file_is_not_modified() {
        let file = PathBuf::from("somewhere/dxcompiler-custom.dll");
        assert_eq!(normalize_candidate(file.clone()), file);
    }

    #[test]
    fn disk_cache_accepts_only_plausible_dxil_containers() {
        let mut valid = vec![0_u8; 32];
        valid[..4].copy_from_slice(b"DXBC");
        assert!(valid_dxil_container(&valid));
        assert!(!valid_dxil_container(b"DXBC"));
        assert!(!valid_dxil_container(&[0_u8; 32]));
    }
}
