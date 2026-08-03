// SPDX-License-Identifier: GPL-2.0-or-later

//! Runtime DirectX Shader Compiler discovery and Shader Model 6 compilation.

use std::path::{Path, PathBuf};

use hassle_rs::{Dxc, HassleError};

const DXCOMPILER_FILE: &str = "dxcompiler.dll";

pub(super) fn compile_hlsl(
    source_name: &str,
    source: &str,
    entry_point: &str,
    target_profile: &str,
) -> Result<Vec<u8>, String> {
    let (dxc, location) = load_dxc()?;
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
        &["-O3", "-Ges"],
        None,
        &[],
    );
    match result {
        Ok(result) => result
            .get_result()
            .map(|blob| blob.to_vec())
            .map_err(|error| format!("DXC returned no DXIL object for {source_name}: {error}")),
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
    use super::normalize_candidate;
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
}
