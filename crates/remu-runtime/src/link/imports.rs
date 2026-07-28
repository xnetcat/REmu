// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::import_dispatch_trampoline;
use remu_core::GuestAddress;
use remu_loader::{DynamicInfo, Elf64Symbol, ImportedSymbol};
use remu_memory::{GUEST_EXECUTE, GUEST_READ, GuestImage};
use std::collections::BTreeMap;

use super::{
    LinkError, ProcessSymbolKey, native_import_target, process_symbol_key,
    synthetic_data_import_target,
};

pub(super) const IMPORT_STUB_SIZE: u64 = 32;

pub(crate) const IMPORT_STUB_SIZE_USIZE: usize = 32;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportStub {
    pub nid: String,
    pub address: GuestAddress,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ImportResolutionStats {
    pub bundled_guest: usize,
    pub host_direct: usize,
    pub hle_dispatch: usize,
    pub unresolved_weak: usize,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ImportResolutionKind {
    BundledGuest,
    HostDirect,
    HleDispatch,
    UnresolvedWeak,
}

impl ImportResolutionKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BundledGuest => "bundled-guest",
            Self::HostDirect => "host-direct",
            Self::HleDispatch => "hle-dispatch",
            Self::UnresolvedWeak => "unresolved-weak",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportResolutionRecord {
    pub symbol: String,
    pub nid: String,
    pub weak: bool,
    pub kind: ImportResolutionKind,
}

impl ImportResolutionStats {
    #[must_use]
    pub const fn total(self) -> usize {
        self.bundled_guest + self.host_direct + self.hle_dispatch + self.unresolved_weak
    }
}

pub(super) fn imported_symbol<'a>(
    dynamic: &'a DynamicInfo,
    import: &ImportedSymbol,
) -> Result<&'a Elf64Symbol, LinkError> {
    let index = usize::try_from(import.symbol_index).map_err(|_| LinkError::MissingSymbol {
        index: import.symbol_index,
    })?;
    dynamic.symbols.get(index).ok_or(LinkError::MissingSymbol {
        index: import.symbol_index,
    })
}

pub(super) fn classify_import(
    dynamic: &DynamicInfo,
    import: &ImportedSymbol,
    process_exports: &BTreeMap<ProcessSymbolKey, u64>,
) -> Result<ImportResolutionKind, LinkError> {
    let symbol = imported_symbol(dynamic, import)?;
    let key = process_symbol_key(dynamic, symbol)?;
    if process_exports.contains_key(&key) {
        return Ok(ImportResolutionKind::BundledGuest);
    }
    let nid = key.nid();
    if native_import_target(nid).is_some() || synthetic_data_import_target(nid).is_some() {
        return Ok(ImportResolutionKind::HostDirect);
    }
    if import.weak {
        Ok(ImportResolutionKind::UnresolvedWeak)
    } else {
        Ok(ImportResolutionKind::HleDispatch)
    }
}

pub(super) fn collect_import_resolutions<'a>(
    dynamics: impl IntoIterator<Item = &'a DynamicInfo>,
    process_exports: &BTreeMap<ProcessSymbolKey, u64>,
) -> Result<(ImportResolutionStats, Vec<ImportResolutionRecord>), LinkError> {
    let mut stats = ImportResolutionStats::default();
    let mut records = Vec::new();
    for dynamic in dynamics {
        for import in &dynamic.imports {
            let kind = classify_import(dynamic, import, process_exports)?;
            match kind {
                ImportResolutionKind::BundledGuest => stats.bundled_guest += 1,
                ImportResolutionKind::HostDirect => stats.host_direct += 1,
                ImportResolutionKind::HleDispatch => stats.hle_dispatch += 1,
                ImportResolutionKind::UnresolvedWeak => stats.unresolved_weak += 1,
            }
            let symbol = imported_symbol(dynamic, import)?;
            let key = process_symbol_key(dynamic, symbol)?;
            records.push(ImportResolutionRecord {
                symbol: symbol.name.clone(),
                nid: key.nid().to_owned(),
                weak: import.weak,
                kind,
            });
        }
    }
    records.sort_by(|left, right| {
        (&left.symbol, left.kind, left.weak).cmp(&(&right.symbol, right.kind, right.weak))
    });
    Ok((stats, records))
}

pub(super) fn create_import_stubs(
    memory: &mut GuestImage,
    dynamic: &DynamicInfo,
    first_dispatch_index: usize,
    process_exports: &BTreeMap<ProcessSymbolKey, u64>,
) -> Result<(Vec<ImportStub>, BTreeMap<u32, GuestAddress>), LinkError> {
    let mut dispatch_imports = BTreeMap::<ProcessSymbolKey, Vec<u32>>::new();
    for import in &dynamic.imports {
        if classify_import(dynamic, import, process_exports)? != ImportResolutionKind::HleDispatch {
            continue;
        }
        let symbol = imported_symbol(dynamic, import)?;
        dispatch_imports
            .entry(process_symbol_key(dynamic, symbol)?)
            .or_default()
            .push(import.symbol_index);
    }
    if dispatch_imports.is_empty() {
        return Ok((Vec::new(), BTreeMap::new()));
    }

    let stub_bytes = u64::try_from(dispatch_imports.len())
        .map_err(|_| LinkError::StubAllocationOverflow)?
        .checked_mul(IMPORT_STUB_SIZE)
        .ok_or(LinkError::StubAllocationOverflow)?;
    let stub_base =
        memory.allocate_staging(stub_bytes, IMPORT_STUB_SIZE, GUEST_READ | GUEST_EXECUTE)?;
    let bytes_len = usize::try_from(stub_bytes).map_err(|_| LinkError::StubAllocationOverflow)?;
    let mut bytes = vec![0_u8; bytes_len];
    let mut import_stubs = Vec::with_capacity(dispatch_imports.len());
    let mut symbol_stubs = BTreeMap::new();
    let dispatcher = import_dispatch_trampoline as *const () as usize as u64;
    for (index, (key, symbol_indices)) in dispatch_imports.into_iter().enumerate() {
        let byte_offset = index * IMPORT_STUB_SIZE_USIZE;
        let dispatch_index = first_dispatch_index
            .checked_add(index)
            .ok_or(LinkError::StubAllocationOverflow)?;
        write_import_stub(&mut bytes[byte_offset..], dispatch_index, dispatcher)?;
        let address = GuestAddress::new(
            stub_base
                .get()
                .checked_add(byte_offset as u64)
                .ok_or(LinkError::AddressOverflow)?,
        );
        for symbol_index in symbol_indices {
            symbol_stubs.insert(symbol_index, address);
        }
        import_stubs.push(ImportStub {
            nid: key.nid().to_owned(),
            address,
        });
    }
    memory.write(stub_base, &bytes)?;
    Ok((import_stubs, symbol_stubs))
}

pub(crate) fn write_import_stub(
    slot: &mut [u8],
    index: usize,
    dispatcher: u64,
) -> Result<(), LinkError> {
    let index = i32::try_from(index).map_err(|_| LinkError::StubAllocationOverflow)?;
    // Push the dispatch index without altering the guest's argument registers, then tail-jump via
    // R11 into the shared frame-capture trampoline. The trampoline removes this word before
    // returning to the original guest call site.
    slot[0] = 0x68;
    slot[1..5].copy_from_slice(&index.to_le_bytes());
    slot[5] = 0x49;
    slot[6] = 0xbb;
    slot[7..15].copy_from_slice(&dispatcher.to_le_bytes());
    slot[15..18].copy_from_slice(&[0x41, 0xff, 0xe3]);
    Ok(())
}
