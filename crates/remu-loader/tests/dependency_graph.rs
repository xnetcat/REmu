// SPDX-License-Identifier: GPL-2.0-or-later

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use remu_loader::{
    ModuleDependencyGraphBuilder, ModuleDependencyGraphError, discover_app_root, inspect_path,
    resolve_module_dependency_graph,
};

const ELF_HEADER_SIZE: usize = 64;
const PROGRAM_HEADER_SIZE: usize = 56;
const DYNAMIC_OFFSET: usize = 0x100;
const STRING_OFFSET: usize = 0x400;
const DT_NULL: i64 = 0;
const DT_NEEDED: i64 = 1;
const DT_STRING_TABLE: i64 = 5;
const DT_STRING_TABLE_SIZE: i64 = 10;
const DT_SCE_ORIGINAL_FILENAME_1: i64 = 0x6100_0041;
const DT_SCE_MODULE_INFO: i64 = 0x6100_000d;
const DT_SCE_NEEDED_MODULE_1: i64 = 0x6100_0045;
const DT_SCE_IMPORT_LIBRARY_1: i64 = 0x6100_0049;

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

struct AppFixture {
    root: PathBuf,
}

impl AppFixture {
    fn new() -> Self {
        let id = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "remu-loader-module-graph-{}-{id}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("sce_sys")).unwrap();
        fs::write(root.join("sce_sys/param.json"), b"{}").unwrap();
        let root = fs::canonicalize(root).unwrap();
        Self { root }
    }

    fn path(&self, relative: impl AsRef<Path>) -> PathBuf {
        self.root.join(relative)
    }

    fn module(&self, relative: &str, metadata: &FixtureMetadata<'_>) -> PathBuf {
        let path = self.path(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, fixture_elf(metadata)).unwrap();
        path
    }
}

impl Drop for AppFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[derive(Default)]
struct FixtureMetadata<'a> {
    needed: &'a [&'a str],
    original_filename: Option<&'a str>,
    required_modules: &'a [(&'a str, u16, u8, u8)],
    exported_modules: &'a [(&'a str, u16, u8, u8)],
    imported_libraries: &'a [(&'a str, u16, u16)],
}

#[test]
fn parses_ps5_module_and_library_dependency_metadata() {
    let fixture = AppFixture::new();
    let path = fixture.module(
        "eboot.bin",
        &FixtureMetadata {
            needed: &["GameCode.prx"],
            original_filename: Some("eboot.bin"),
            required_modules: &[("libkernel", 0x1234, 2, 7)],
            exported_modules: &[("game", 0x2345, 1, 3)],
            imported_libraries: &[("libkernel", 0x3456, 0x0102)],
        },
    );

    let dynamic = inspect_path(path).unwrap().elf.unwrap().dynamic.unwrap();
    assert_eq!(dynamic.needed_libraries, ["GameCode.prx"]);
    assert_eq!(dynamic.original_filename.as_deref(), Some("eboot.bin"));
    assert_eq!(dynamic.required_modules[0].name, "libkernel");
    assert_eq!(dynamic.required_modules[0].id, 0x1234);
    assert_eq!(dynamic.required_modules[0].version_major, 2);
    assert_eq!(dynamic.required_modules[0].version_minor, 7);
    assert_eq!(dynamic.exported_modules[0].name, "game");
    assert_eq!(dynamic.imported_libraries[0].id, 0x3456);
    assert_eq!(dynamic.imported_libraries[0].version, 0x0102);
}

#[test]
fn resolves_overlay_standard_and_nested_modules_in_dependency_order() {
    let fixture = AppFixture::new();
    let entry = fixture.module(
        "decrypted/eboot.bin",
        &FixtureMetadata {
            needed: &["GameCode.prx", "libSystemProvided.prx"],
            ..FixtureMetadata::default()
        },
    );
    fixture.module(
        "decrypted/sce_module/GameCode.prx",
        &FixtureMetadata {
            needed: &["Engine.prx"],
            exported_modules: &[("GameCode", 1, 1, 0)],
            ..FixtureMetadata::default()
        },
    );
    fixture.module(
        "Media/Modules/Engine.prx",
        &FixtureMetadata {
            needed: &["Audio.prx"],
            exported_modules: &[("Engine", 2, 1, 0)],
            ..FixtureMetadata::default()
        },
    );
    fixture.module(
        "engine/plugins/audio/Audio.sprx",
        &FixtureMetadata {
            exported_modules: &[("Audio", 3, 1, 0)],
            ..FixtureMetadata::default()
        },
    );

    assert_eq!(discover_app_root(&entry).unwrap(), fixture.root);
    let graph = resolve_module_dependency_graph(&entry).unwrap();
    let order = graph
        .dependency_order
        .iter()
        .map(|node| node.path.file_name().unwrap().to_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(order, ["Audio.sprx", "Engine.prx", "GameCode.prx"]);
    assert_eq!(
        graph.search_roots,
        [fixture.path("decrypted"), fixture.root.clone()]
    );
    assert_eq!(graph.unresolved_dependencies.len(), 1);
    assert_eq!(
        graph.unresolved_dependencies[0].name,
        "libSystemProvided.prx"
    );
    assert_eq!(graph.unresolved_dependencies[0].requested_by, entry);
}

#[test]
fn decrypted_overlay_shadows_the_packaged_copy() {
    let fixture = AppFixture::new();
    fs::create_dir_all(fixture.path("decrypted/sce_sys")).unwrap();
    fs::write(fixture.path("decrypted/sce_sys/param.json"), b"{}").unwrap();
    let entry = fixture.module(
        "decrypted/eboot.bin",
        &FixtureMetadata {
            needed: &["GameCode.prx"],
            ..FixtureMetadata::default()
        },
    );
    let overlay = fixture.module(
        "decrypted/sce_module/GameCode.prx",
        &FixtureMetadata::default(),
    );
    fixture.module("sce_module/GameCode.prx", &FixtureMetadata::default());

    let graph = resolve_module_dependency_graph(entry).unwrap();
    assert_eq!(graph.dependency_order.len(), 1);
    assert_eq!(graph.dependency_order[0].path, overlay);
}

#[test]
fn rejects_ambiguous_duplicate_candidates_at_the_same_layer() {
    let fixture = AppFixture::new();
    let entry = fixture.module(
        "eboot.bin",
        &FixtureMetadata {
            needed: &["Duplicate.prx"],
            ..FixtureMetadata::default()
        },
    );
    fixture.module("sce_module/Duplicate.prx", &FixtureMetadata::default());
    fixture.module("custom/nested/Duplicate.prx", &FixtureMetadata::default());

    let error = resolve_module_dependency_graph(entry).unwrap_err();
    assert!(matches!(
        error,
        ModuleDependencyGraphError::DuplicateCandidates { ref candidates, .. }
            if candidates.len() == 2
    ));
    assert!(
        error
            .to_string()
            .contains("matches multiple bundled modules")
    );
}

#[test]
fn rejects_dependency_cycles_with_the_full_cycle_chain() {
    let fixture = AppFixture::new();
    let entry = fixture.module(
        "eboot.bin",
        &FixtureMetadata {
            needed: &["A.prx"],
            ..FixtureMetadata::default()
        },
    );
    fixture.module(
        "sce_module/A.prx",
        &FixtureMetadata {
            needed: &["B.prx"],
            ..FixtureMetadata::default()
        },
    );
    fixture.module(
        "Media/Modules/B.prx",
        &FixtureMetadata {
            needed: &["A.prx"],
            ..FixtureMetadata::default()
        },
    );

    let error = resolve_module_dependency_graph(entry).unwrap_err();
    assert!(matches!(
        error,
        ModuleDependencyGraphError::DependencyCycle { ref cycle }
            if cycle.len() == 3
                && cycle[0].ends_with("A.prx")
                && cycle[1].ends_with("B.prx")
                && cycle[2].ends_with("A.prx")
    ));
}

#[test]
fn strict_mode_reports_missing_dependency_and_search_roots() {
    let fixture = AppFixture::new();
    let entry = fixture.module(
        "eboot.bin",
        &FixtureMetadata {
            needed: &["Missing.prx"],
            ..FixtureMetadata::default()
        },
    );

    let error = ModuleDependencyGraphBuilder::new(&entry)
        .deny_unresolved(true)
        .resolve()
        .unwrap_err();
    assert!(matches!(
        error,
        ModuleDependencyGraphError::MissingDependency {
            ref dependency,
            ref searched_roots,
            ..
        } if dependency == "Missing.prx"
            && searched_roots.as_slice() == std::slice::from_ref(&fixture.root)
    ));
}

#[test]
fn rejects_duplicate_dependency_declarations_and_module_identities() {
    let duplicate_fixture = AppFixture::new();
    let duplicate_entry = duplicate_fixture.module(
        "eboot.bin",
        &FixtureMetadata {
            needed: &["A.prx", "a.PRX"],
            ..FixtureMetadata::default()
        },
    );
    duplicate_fixture.module("sce_module/A.prx", &FixtureMetadata::default());
    assert!(matches!(
        resolve_module_dependency_graph(duplicate_entry),
        Err(ModuleDependencyGraphError::DuplicateDependency { .. })
    ));

    let identity_fixture = AppFixture::new();
    let identity_entry = identity_fixture.module(
        "eboot.bin",
        &FixtureMetadata {
            needed: &["A.prx", "B.prx"],
            ..FixtureMetadata::default()
        },
    );
    identity_fixture.module(
        "sce_module/A.prx",
        &FixtureMetadata {
            exported_modules: &[("SharedIdentity", 1, 1, 0)],
            ..FixtureMetadata::default()
        },
    );
    identity_fixture.module(
        "Media/Modules/B.prx",
        &FixtureMetadata {
            exported_modules: &[("sharedidentity", 2, 1, 0)],
            ..FixtureMetadata::default()
        },
    );
    assert!(matches!(
        resolve_module_dependency_graph(identity_entry),
        Err(ModuleDependencyGraphError::DuplicateModuleIdentity { .. })
    ));
}

fn fixture_elf(metadata: &FixtureMetadata<'_>) -> Vec<u8> {
    let mut strings = vec![0_u8];
    let mut entries = Vec::<(i64, u64)>::new();
    for dependency in metadata.needed {
        entries.push((DT_NEEDED, u64::from(push_string(&mut strings, dependency))));
    }
    if let Some(filename) = metadata.original_filename {
        entries.push((
            DT_SCE_ORIGINAL_FILENAME_1,
            u64::from(push_string(&mut strings, filename)),
        ));
    }
    for &(name, id, major, minor) in metadata.required_modules {
        let offset = push_string(&mut strings, name);
        entries.push((
            DT_SCE_NEEDED_MODULE_1,
            pack_module_reference(offset, id, major, minor),
        ));
    }
    for &(name, id, major, minor) in metadata.exported_modules {
        let offset = push_string(&mut strings, name);
        entries.push((
            DT_SCE_MODULE_INFO,
            pack_module_reference(offset, id, major, minor),
        ));
    }
    for &(name, id, version) in metadata.imported_libraries {
        let offset = push_string(&mut strings, name);
        entries.push((
            DT_SCE_IMPORT_LIBRARY_1,
            (u64::from(id) << 48) | (u64::from(version) << 32) | u64::from(offset),
        ));
    }
    entries.push((DT_STRING_TABLE, 0x1300));
    entries.push((DT_STRING_TABLE_SIZE, strings.len() as u64));
    entries.push((DT_NULL, 0));

    let dynamic_size = entries.len() * 16;
    assert!(DYNAMIC_OFFSET + dynamic_size <= STRING_OFFSET);
    let mut bytes = vec![0_u8; STRING_OFFSET + strings.len()];
    let load_size = (bytes.len() - DYNAMIC_OFFSET) as u64;
    write_elf_header(&mut bytes);
    write_program_header(
        &mut bytes,
        0,
        1,
        4,
        DYNAMIC_OFFSET as u64,
        0x1000,
        load_size,
    );
    write_program_header(
        &mut bytes,
        1,
        2,
        4,
        DYNAMIC_OFFSET as u64,
        0x1000,
        dynamic_size as u64,
    );
    for (index, (tag, value)) in entries.into_iter().enumerate() {
        let offset = DYNAMIC_OFFSET + index * 16;
        bytes[offset..offset + 8].copy_from_slice(&tag.to_le_bytes());
        bytes[offset + 8..offset + 16].copy_from_slice(&value.to_le_bytes());
    }
    bytes[STRING_OFFSET..].copy_from_slice(&strings);
    bytes
}

fn push_string(strings: &mut Vec<u8>, value: &str) -> u32 {
    let offset = u32::try_from(strings.len()).unwrap();
    strings.extend_from_slice(value.as_bytes());
    strings.push(0);
    offset
}

const fn pack_module_reference(offset: u32, id: u16, major: u8, minor: u8) -> u64 {
    (id as u64) << 48 | (major as u64) << 40 | (minor as u64) << 32 | offset as u64
}

fn write_elf_header(bytes: &mut [u8]) {
    bytes[..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    bytes[4] = 2;
    bytes[5] = 1;
    bytes[6] = 1;
    bytes[7] = 9;
    bytes[8] = 2;
    bytes[0x10..0x12].copy_from_slice(&0xfe10_u16.to_le_bytes());
    bytes[0x12..0x14].copy_from_slice(&62_u16.to_le_bytes());
    bytes[0x20..0x28].copy_from_slice(&(ELF_HEADER_SIZE as u64).to_le_bytes());
    bytes[0x34..0x36].copy_from_slice(&u16::try_from(ELF_HEADER_SIZE).unwrap().to_le_bytes());
    bytes[0x36..0x38].copy_from_slice(&u16::try_from(PROGRAM_HEADER_SIZE).unwrap().to_le_bytes());
    bytes[0x38..0x3a].copy_from_slice(&2_u16.to_le_bytes());
}

#[allow(clippy::too_many_arguments)]
fn write_program_header(
    bytes: &mut [u8],
    index: usize,
    program_type: u32,
    flags: u32,
    file_offset: u64,
    virtual_address: u64,
    file_size: u64,
) {
    let offset = ELF_HEADER_SIZE + index * PROGRAM_HEADER_SIZE;
    bytes[offset..offset + 4].copy_from_slice(&program_type.to_le_bytes());
    bytes[offset + 4..offset + 8].copy_from_slice(&flags.to_le_bytes());
    bytes[offset + 8..offset + 16].copy_from_slice(&file_offset.to_le_bytes());
    bytes[offset + 16..offset + 24].copy_from_slice(&virtual_address.to_le_bytes());
    bytes[offset + 32..offset + 40].copy_from_slice(&file_size.to_le_bytes());
    bytes[offset + 40..offset + 48].copy_from_slice(&file_size.to_le_bytes());
    bytes[offset + 48..offset + 56].copy_from_slice(&0x1000_u64.to_le_bytes());
}
