// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use remu_core::GuestAddress;
use remu_loader::{DynamicInfo, Elf64Symbol, LoadedExecutable};
use remu_memory::GuestImage;
use std::collections::BTreeMap;

use super::{LinkError, LinkedModuleInput, defined_symbol_address_at};

#[derive(Clone, Debug)]
pub(super) struct ProcessExport {
    pub(super) address: u64,
    pub(super) weak: bool,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum ProcessSymbolKey {
    Scoped {
        nid: String,
        library: String,
        library_version: u16,
        module: String,
        module_version_major: u8,
        module_version_minor: u8,
        symbol_type: u8,
    },
    Unscoped {
        nid: String,
        symbol_type: u8,
    },
}

impl ProcessSymbolKey {
    pub(super) fn nid(&self) -> &str {
        match self {
            Self::Scoped { nid, .. } | Self::Unscoped { nid, .. } => nid,
        }
    }
}

pub(super) fn main_base_export_input<'a>(
    executable: &'a LoadedExecutable,
    dynamic: &'a DynamicInfo,
    base: GuestAddress,
) -> LinkedModuleInput<'a> {
    LinkedModuleInput {
        executable,
        dynamic,
        base,
        tls_module_id: 1,
    }
}

pub(super) fn collect_process_exports(
    memory: &GuestImage,
    main: &LinkedModuleInput<'_>,
    modules: &[LinkedModuleInput<'_>],
) -> Result<BTreeMap<ProcessSymbolKey, u64>, LinkError> {
    let mut exports = BTreeMap::<ProcessSymbolKey, ProcessExport>::new();
    for module in std::iter::once(main).chain(modules) {
        for symbol in module.dynamic.symbols.iter().filter(|symbol| {
            !symbol.is_undefined() && !symbol.name.is_empty() && matches!(symbol.binding(), 1 | 2)
        }) {
            let key = process_symbol_key(module.dynamic, symbol)?;
            if key.nid().is_empty() {
                continue;
            }
            let address = defined_symbol_address_at(memory, module.base, symbol)?;
            let candidate = ProcessExport {
                address,
                weak: symbol.is_weak(),
            };
            match exports.get_mut(&key) {
                None => {
                    exports.insert(key, candidate);
                }
                Some(current) if current.weak && !candidate.weak => {
                    *current = candidate;
                }
                Some(_) => {}
            }
        }
    }
    Ok(exports
        .into_iter()
        .map(|(nid, export)| (nid, export.address))
        .collect())
}

pub(super) fn module_exports_by_nid(
    memory: &GuestImage,
    module: &LinkedModuleInput<'_>,
) -> Result<BTreeMap<String, u64>, LinkError> {
    let mut exports = BTreeMap::<String, ProcessExport>::new();
    for symbol in module.dynamic.symbols.iter().filter(|symbol| {
        !symbol.is_undefined() && !symbol.name.is_empty() && matches!(symbol.binding(), 1 | 2)
    }) {
        let nid = symbol.name.split('#').next().unwrap_or_default();
        if nid.is_empty() {
            continue;
        }
        let candidate = ProcessExport {
            address: defined_symbol_address_at(memory, module.base, symbol)?,
            weak: symbol.is_weak(),
        };
        match exports.get_mut(nid) {
            None => {
                exports.insert(nid.to_owned(), candidate);
            }
            Some(current) if current.weak && !candidate.weak => *current = candidate,
            Some(_) => {}
        }
    }
    Ok(exports
        .into_iter()
        .map(|(nid, export)| (nid, export.address))
        .collect())
}

pub(super) fn process_symbol_key(
    dynamic: &DynamicInfo,
    symbol: &Elf64Symbol,
) -> Result<ProcessSymbolKey, LinkError> {
    let mut ids = symbol.name.split('#');
    let nid = ids.next().unwrap_or_default();
    let library_id = ids.next();
    let module_id = ids.next();
    if ids.next().is_none()
        && let (Some(library_id), Some(module_id)) = (library_id, module_id)
        && let Some(library) = dynamic
            .imported_libraries
            .iter()
            .chain(&dynamic.exported_libraries)
            .find(|library| encode_sce_id(library.id) == library_id)
        && let Some(module) = dynamic
            .required_modules
            .iter()
            .chain(&dynamic.exported_modules)
            .find(|module| encode_sce_id(module.id) == module_id)
    {
        return Ok(ProcessSymbolKey::Scoped {
            nid: nid.to_owned(),
            library: library.name.clone(),
            library_version: library.version,
            module: module.name.clone(),
            module_version_major: module.version_major,
            module_version_minor: module.version_minor,
            symbol_type: symbol.symbol_type(),
        });
    }
    if library_id.is_some()
        && module_id.is_some()
        && (!dynamic.imported_libraries.is_empty()
            || !dynamic.exported_libraries.is_empty()
            || !dynamic.required_modules.is_empty()
            || !dynamic.exported_modules.is_empty())
    {
        return Err(LinkError::InvalidSymbolNamespace {
            symbol: symbol.name.clone(),
        });
    }
    Ok(ProcessSymbolKey::Unscoped {
        nid: nid.to_owned(),
        symbol_type: symbol.symbol_type(),
    })
}

pub(super) fn encode_sce_id(id: u16) -> String {
    const CODES: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    let mut encoded = String::with_capacity(3);
    if id < 0x40 {
        encoded.push(char::from(CODES[usize::from(id)]));
    } else if id < 0x1000 {
        encoded.push(char::from(CODES[usize::from((id >> 6) & 0x3f)]));
        encoded.push(char::from(CODES[usize::from(id & 0x3f)]));
    } else {
        encoded.push(char::from(CODES[usize::from((id >> 12) & 0x3f)]));
        encoded.push(char::from(CODES[usize::from((id >> 6) & 0x3f)]));
        encoded.push(char::from(CODES[usize::from(id & 0x3f)]));
    }
    encoded
}
