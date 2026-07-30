// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::align_up_u64;
use crate::tls::{GUEST_TLS_PREFIX_SIZE, GUEST_TLS_TCB_SIZE, TLS_HELPER_REGION_SIZE};
use remu_core::GuestAddress;
use remu_loader::{DynamicInfo, LoadedExecutable, UnresolvedModuleDependency};
use remu_memory::{GuestImage, MappingError};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::{IMPORT_STUB_SIZE, LinkError, runtime_address_at};

pub(crate) const PT_TLS: u32 = 7;

pub(super) const PT_SCE_PROC_PARAM: u32 = 0x6100_0001;

pub(crate) const MACOS_GUEST_PAGE_ALIGNMENT: u64 = 0x4000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LinkedGuestModule {
    pub(crate) name: String,
    pub(crate) handle: u32,
    pub(crate) exports_by_nid: BTreeMap<String, u64>,
    pub(crate) initializers: Vec<GuestAddress>,
    pub(crate) started: bool,
}

/// Filesystem and dependency information retained from linking a complete guest process image.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModuleLinkReport {
    pub app_root: PathBuf,
    pub entry_path: PathBuf,
    /// Bundled PRX/SPRX objects in the dependency-first order used for mapping and initialization.
    pub loaded_modules: Vec<PathBuf>,
    /// Dependencies that must be supplied by the emulator's system-module implementations.
    pub unresolved_dependencies: Vec<UnresolvedModuleDependency>,
}

pub(crate) fn discover_runtime_module_paths(app_root: &Path) -> Result<Vec<PathBuf>, LinkError> {
    // Unreal plug-ins are commonly unpacked under Media/Plugins, while Wwise titles such as
    // Silent Hill keep load-on-demand effect modules directly in /app0/prx. Both sets must be
    // mapped up front because the process image is finalized before sceKernelLoadStartModule can
    // ask to start one of them.
    let mut pending = [
        app_root.join("decrypted/Media/Plugins"),
        app_root.join("prx"),
    ]
    .into_iter()
    .filter(|root| root.is_dir())
    .collect::<Vec<_>>();
    let mut modules = Vec::new();
    while let Some(directory) = pending.pop() {
        let entries = std::fs::read_dir(&directory)
            .map_err(|source| LinkError::RuntimeModuleDiscovery {
                path: directory.clone(),
                source,
            })?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|source| LinkError::RuntimeModuleDiscovery {
                path: directory.clone(),
                source,
            })?;
        let mut entries = entries;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries.into_iter().rev() {
            let path = entry.path();
            let file_type =
                entry
                    .file_type()
                    .map_err(|source| LinkError::RuntimeModuleDiscovery {
                        path: path.clone(),
                        source,
                    })?;
            if file_type.is_dir() {
                pending.push(path);
                continue;
            }
            if (!file_type.is_file() && !file_type.is_symlink())
                || !path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| {
                        extension.eq_ignore_ascii_case("prx")
                            || extension.eq_ignore_ascii_case("sprx")
                    })
            {
                continue;
            }
            modules.push(std::fs::canonicalize(&path).map_err(|source| {
                LinkError::RuntimeModuleDiscovery {
                    path: path.clone(),
                    source,
                }
            })?);
        }
    }
    modules.sort();
    Ok(modules)
}

pub(super) fn resolve_process_param(
    elf: &remu_loader::Elf64Info,
    image_base: GuestAddress,
) -> Result<Option<GuestAddress>, LinkError> {
    elf.program_headers
        .iter()
        .find(|header| header.program_type == PT_SCE_PROC_PARAM)
        .map(|header| {
            image_base
                .get()
                .checked_add(header.virtual_address.get())
                .map(GuestAddress::new)
                .ok_or(LinkError::AddressOverflow)
        })
        .transpose()
}

pub(super) struct LinkedModuleInput<'a> {
    pub(super) executable: &'a LoadedExecutable,
    pub(super) dynamic: &'a DynamicInfo,
    pub(super) base: GuestAddress,
    pub(super) tls_module_id: u64,
}

pub(super) fn map_bundled_modules<'a>(
    memory: &mut GuestImage,
    modules: &[(&'a str, &'a LoadedExecutable)],
) -> Result<Vec<LinkedModuleInput<'a>>, LinkError> {
    let mut linked_modules = Vec::with_capacity(modules.len());
    for (index, (_name, module)) in modules.iter().enumerate() {
        let dynamic = module
            .info
            .elf
            .as_ref()
            .ok_or(LinkError::MissingElf)?
            .dynamic
            .as_ref()
            .ok_or(LinkError::MissingDynamicMetadata)?;
        let base = memory.map_module(module, MACOS_GUEST_PAGE_ALIGNMENT)?;
        linked_modules.push(LinkedModuleInput {
            executable: module,
            dynamic,
            base,
            tls_module_id: u64::try_from(index + 2).map_err(|_| LinkError::AddressOverflow)?,
        });
    }
    Ok(linked_modules)
}

pub(super) fn calculate_process_auxiliary_reserve(
    main_dynamic: &DynamicInfo,
    modules: &[(&str, &LoadedExecutable)],
    tls_store_helper_region_size: u64,
) -> Result<u64, LinkError> {
    let mut cursor = 0_u64;
    let mut import_count = main_dynamic.imports.len();
    for (_, module) in modules {
        cursor =
            align_up_u64(cursor, MACOS_GUEST_PAGE_ALIGNMENT).ok_or(LinkError::AddressOverflow)?;
        let module_size = module
            .load_segments
            .iter()
            .try_fold(0_u64, |maximum, segment| {
                segment
                    .virtual_address
                    .get()
                    .checked_add(segment.memory_size)
                    .map(|end| maximum.max(end))
                    .ok_or(LinkError::AddressOverflow)
            })?;
        if module_size == 0 {
            return Err(LinkError::Mapping(MappingError::NoLoadSegments));
        }
        cursor = cursor
            .checked_add(module_size)
            .ok_or(LinkError::AddressOverflow)?;
        let dynamic = module
            .info
            .elf
            .as_ref()
            .ok_or(LinkError::MissingElf)?
            .dynamic
            .as_ref()
            .ok_or(LinkError::MissingDynamicMetadata)?;
        import_count = import_count
            .checked_add(dynamic.imports.len())
            .ok_or(LinkError::StubAllocationOverflow)?;
    }

    cursor = align_up_u64(cursor, IMPORT_STUB_SIZE).ok_or(LinkError::AddressOverflow)?;
    let maximum_stub_bytes = u64::try_from(import_count)
        .map_err(|_| LinkError::StubAllocationOverflow)?
        .checked_mul(IMPORT_STUB_SIZE)
        .ok_or(LinkError::StubAllocationOverflow)?;
    cursor = cursor
        .checked_add(maximum_stub_bytes)
        .ok_or(LinkError::AddressOverflow)?;
    cursor = align_up_u64(cursor, 16).ok_or(LinkError::AddressOverflow)?;
    cursor = cursor
        .checked_add(TLS_HELPER_REGION_SIZE)
        .ok_or(LinkError::AddressOverflow)?;
    cursor = cursor
        .checked_add(tls_store_helper_region_size)
        .ok_or(LinkError::AddressOverflow)?;
    cursor = align_up_u64(cursor, MACOS_GUEST_PAGE_ALIGNMENT).ok_or(LinkError::AddressOverflow)?;
    cursor = cursor
        .checked_add(GUEST_TLS_PREFIX_SIZE + GUEST_TLS_TCB_SIZE)
        .ok_or(LinkError::AddressOverflow)?;
    // Keep one guard-sized cushion for alignment changes without hiding unbounded allocations.
    cursor = cursor
        .checked_add(MACOS_GUEST_PAGE_ALIGNMENT)
        .ok_or(LinkError::AddressOverflow)?;
    align_up_u64(cursor, MACOS_GUEST_PAGE_ALIGNMENT).ok_or(LinkError::AddressOverflow)
}

pub(super) fn append_object_initializers(
    output: &mut Vec<GuestAddress>,
    memory: &GuestImage,
    image_base: GuestAddress,
    dynamic: &DynamicInfo,
) -> Result<(), LinkError> {
    if dynamic.init_address != 0 {
        output.push(normalize_process_address(
            memory,
            image_base,
            dynamic.init_address,
        )?);
    }
    output.extend(collect_initializer_array(
        memory,
        image_base,
        dynamic.init_array_address,
        dynamic.init_array_size,
        "DT_INIT_ARRAY",
    )?);
    Ok(())
}

pub(super) fn collect_initializer_array(
    memory: &GuestImage,
    image_base: GuestAddress,
    address: u64,
    size: u64,
    kind: &'static str,
) -> Result<Vec<GuestAddress>, LinkError> {
    if address == 0 || size == 0 {
        return Ok(Vec::new());
    }
    if !size.is_multiple_of(8) {
        return Err(LinkError::InvalidInitializerArraySize { kind, size });
    }
    let count = usize::try_from(size / 8).map_err(|_| LinkError::AddressOverflow)?;
    let byte_length = usize::try_from(size).map_err(|_| LinkError::AddressOverflow)?;
    let array = runtime_address_at(memory, image_base, GuestAddress::new(address), byte_length)?;
    let mut bytes = vec![0_u8; byte_length];
    memory.read(array, &mut bytes)?;
    let mut initializers = Vec::with_capacity(count);
    for entry in bytes.chunks_exact(8) {
        let value = u64::from_le_bytes(entry.try_into().expect("fixed initializer pointer"));
        if value != 0 && value != u64::MAX {
            initializers.push(normalize_process_address(memory, image_base, value)?);
        }
    }
    Ok(initializers)
}

pub(super) fn normalize_process_address(
    memory: &GuestImage,
    image_base: GuestAddress,
    address: u64,
) -> Result<GuestAddress, LinkError> {
    let address = GuestAddress::new(address);
    if memory.contains(address, 1) {
        return Ok(address);
    }
    let rebased = GuestAddress::new(
        image_base
            .get()
            .checked_add(address.get())
            .ok_or(LinkError::AddressOverflow)?,
    );
    if memory.contains(rebased, 1) {
        Ok(rebased)
    } else {
        Err(LinkError::InitializerOutsideImage { address: rebased })
    }
}
