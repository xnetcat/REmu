// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use remu_core::GuestAddress;
use remu_loader::{DynamicInfo, Elf64Rela, Elf64Symbol};
use remu_memory::GuestImage;
use std::collections::BTreeMap;

use super::{
    LinkError, ProcessSymbolKey, native_import_target, process_symbol_key,
    synthetic_data_import_target,
};

pub(super) const R_X86_64_64: u32 = 1;

pub(super) const R_X86_64_GLOB_DAT: u32 = 6;

pub(super) const R_X86_64_JUMP_SLOT: u32 = 7;

pub(super) const R_X86_64_RELATIVE: u32 = 8;

/// x86-64 System V TLS/ifunc relocation types (not defined by the SDK headers, which carry no ELF
/// constants — confirmed against the psABI: `DTPMOD64 = 16`, `DTPOFF64 = 17`, `TPOFF64 = 18`,
/// `IRELATIVE = 37`).
pub(super) const R_X86_64_DTPMOD64: u32 = 16;

pub(super) const R_X86_64_DTPOFF64: u32 = 17;

pub(super) const R_X86_64_TPOFF64: u32 = 18;

pub(super) const R_X86_64_IRELATIVE: u32 = 37;

/// ELF symbol type for thread-local storage objects (`STT_TLS`).
const STT_TLS: u8 = 6;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RelocationStats {
    pub absolute: usize,
    pub global_data: usize,
    pub jump_slots: usize,
    pub relative: usize,
    /// Dynamic-TLS relocations (`DTPMOD64`/`DTPOFF64`/`TPOFF64`).
    pub tls_dynamic: usize,
    /// Ifunc relocations (`IRELATIVE`), resolved by running the guest resolver at process entry.
    pub ifunc: usize,
}

impl RelocationStats {
    #[must_use]
    pub const fn total(self) -> usize {
        self.absolute
            + self.global_data
            + self.jump_slots
            + self.relative
            + self.tls_dynamic
            + self.ifunc
    }
}

pub(super) fn add_relocation_stats(total: &mut RelocationStats, addition: RelocationStats) {
    total.absolute += addition.absolute;
    total.global_data += addition.global_data;
    total.jump_slots += addition.jump_slots;
    total.relative += addition.relative;
    total.tls_dynamic += addition.tls_dynamic;
    total.ifunc += addition.ifunc;
}

/// Where a TLS reference's storage lives, resolved through the module that actually *defines* the
/// symbol (which is not necessarily the module carrying the relocation).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct TlsSymbolResolution {
    /// Dynamic-TLS module id of the defining module (`DTPMOD64`).
    pub(super) module_id: u64,
    /// Offset of the symbol within the defining module's PT_TLS image (`DTPOFF64`).
    pub(super) tls_offset: u64,
    /// The defining module's static-TLS block size below the thread pointer (`TPOFF64`).
    pub(super) static_offset: u64,
}

/// An ifunc relocation whose resolver runs at process entry (like glibc's startup ifunc pass).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PendingIfunc {
    pub(crate) target: GuestAddress,
    pub(crate) resolver: u64,
}

/// Per-module TLS relocation inputs plus the process-wide TLS symbol scope.
pub(super) struct TlsRelocationContext<'a> {
    /// The relocated module's own dynamic-TLS module id.
    pub(super) module_id: u64,
    /// The relocated module's static-TLS block size below the thread pointer.
    pub(super) module_static_offset: u64,
    /// Defined TLS symbols across every mapped module, so a cross-module reference resolves to the
    /// defining module rather than the referencing one.
    pub(super) tls_symbols: &'a BTreeMap<ProcessSymbolKey, TlsSymbolResolution>,
}

pub(super) fn apply_relocations_at(
    memory: &mut GuestImage,
    image_base: GuestAddress,
    tls: &TlsRelocationContext<'_>,
    dynamic: &DynamicInfo,
    symbol_stubs: &BTreeMap<u32, GuestAddress>,
    process_exports: &BTreeMap<ProcessSymbolKey, u64>,
    pending_ifuncs: &mut Vec<PendingIfunc>,
) -> Result<RelocationStats, LinkError> {
    let mut stats = RelocationStats::default();
    for relocation in dynamic.relocations.iter().chain(&dynamic.jump_relocations) {
        let target = runtime_address_at(memory, image_base, relocation.offset, 8)?;
        if std::env::var("REMU_TRACE_GUEST_RELOCATIONS").is_ok_and(|targets| {
            targets.split(',').any(|candidate| {
                let candidate = candidate.trim();
                candidate
                    .strip_prefix("0x")
                    .and_then(|value| u64::from_str_radix(value, 16).ok())
                    .or_else(|| candidate.parse().ok())
                    == Some(target.get())
            })
        }) {
            eprintln!(
                "[REMu][GUEST-RELOCATION] target={:#018x} image_base={:#018x} offset={:#x} type={} symbol={} addend={:#x}",
                target.get(),
                image_base.get(),
                relocation.offset.get(),
                relocation.relocation_type,
                relocation.symbol_index,
                relocation.addend,
            );
        }
        let value = match relocation.relocation_type {
            R_X86_64_RELATIVE => {
                stats.relative += 1;
                image_base.get().wrapping_add_signed(relocation.addend)
            }
            R_X86_64_64 => {
                stats.absolute += 1;
                resolve_symbol_at(
                    memory,
                    image_base,
                    dynamic,
                    symbol_stubs,
                    process_exports,
                    relocation,
                )?
                .wrapping_add_signed(relocation.addend)
            }
            R_X86_64_GLOB_DAT => {
                stats.global_data += 1;
                resolve_symbol_at(
                    memory,
                    image_base,
                    dynamic,
                    symbol_stubs,
                    process_exports,
                    relocation,
                )?
            }
            R_X86_64_JUMP_SLOT => {
                stats.jump_slots += 1;
                resolve_symbol_at(
                    memory,
                    image_base,
                    dynamic,
                    symbol_stubs,
                    process_exports,
                    relocation,
                )?
            }
            R_X86_64_DTPMOD64 | R_X86_64_DTPOFF64 | R_X86_64_TPOFF64 => {
                stats.tls_dynamic += 1;
                let resolution = resolve_tls_reference(dynamic, tls, relocation)?;
                tls_relocation_value(relocation.relocation_type, &resolution, relocation.addend)
            }
            R_X86_64_IRELATIVE => {
                stats.ifunc += 1;
                // The resolver is guest code that cannot run until the process is live, so record
                // it and write the resolver address as a placeholder. The startup ifunc pass
                // overwrites the slot with the resolved target before any initializer runs.
                let resolver = image_base.get().wrapping_add_signed(relocation.addend);
                pending_ifuncs.push(PendingIfunc { target, resolver });
                resolver
            }
            relocation_type => {
                return Err(LinkError::UnsupportedRelocation {
                    relocation_type,
                    offset: relocation.offset,
                });
            }
        };
        memory.write(target, &value.to_le_bytes())?;
    }
    Ok(stats)
}

/// Resolves a TLS relocation to the module that defines its symbol.
///
/// A relocation with no symbol is a local-dynamic reference to the relocated module's own block. A
/// defined symbol lives in that same module. An undefined symbol is a cross-module reference and is
/// resolved through the process-wide TLS symbol scope, so `DTPMOD64` names the defining module —
/// not the referencing one, which the former `tls_module_id` shortcut always assumed.
pub(super) fn resolve_tls_reference(
    dynamic: &DynamicInfo,
    tls: &TlsRelocationContext<'_>,
    relocation: &Elf64Rela,
) -> Result<TlsSymbolResolution, LinkError> {
    let own = TlsSymbolResolution {
        module_id: tls.module_id,
        tls_offset: 0,
        static_offset: tls.module_static_offset,
    };
    if relocation.symbol_index == 0 {
        return Ok(own);
    }
    let index = usize::try_from(relocation.symbol_index).map_err(|_| LinkError::MissingSymbol {
        index: relocation.symbol_index,
    })?;
    let symbol = dynamic.symbols.get(index).ok_or(LinkError::MissingSymbol {
        index: relocation.symbol_index,
    })?;
    if !symbol.is_undefined() {
        return Ok(TlsSymbolResolution {
            tls_offset: symbol.value.get(),
            ..own
        });
    }
    let key = process_symbol_key(dynamic, symbol)?;
    if let Some(resolution) = tls.tls_symbols.get(&key) {
        return Ok(*resolution);
    }
    if symbol.is_weak() {
        // An unresolved weak TLS symbol degrades to this module's block base, matching the way a
        // weak data symbol degrades to zero rather than aborting the link.
        return Ok(own);
    }
    Err(LinkError::MissingSymbol {
        index: relocation.symbol_index,
    })
}

/// Computes the value written for a dynamic-TLS relocation from its resolved storage location.
pub(super) fn tls_relocation_value(
    relocation_type: u32,
    resolution: &TlsSymbolResolution,
    addend: i64,
) -> u64 {
    match relocation_type {
        R_X86_64_DTPMOD64 => resolution.module_id,
        R_X86_64_DTPOFF64 => resolution.tls_offset.wrapping_add_signed(addend),
        // The thread pointer sits at the top of the static-TLS region (variant II), so a symbol at
        // `tls_offset` within a module whose block ends `static_offset` below the pointer is at a
        // negative thread-pointer-relative displacement.
        R_X86_64_TPOFF64 => resolution
            .tls_offset
            .wrapping_add_signed(addend)
            .wrapping_sub(resolution.static_offset),
        _ => 0,
    }
}

/// Builds the process-wide scope of defined TLS symbols, keyed the same way as data exports so a
/// cross-module TLS reference resolves to its defining module.
pub(super) fn collect_tls_symbols(
    modules: &[(u64, u64, &DynamicInfo)],
) -> Result<BTreeMap<ProcessSymbolKey, TlsSymbolResolution>, LinkError> {
    let mut symbols = BTreeMap::<ProcessSymbolKey, (TlsSymbolResolution, bool)>::new();
    for &(module_id, static_offset, dynamic) in modules {
        for symbol in dynamic.symbols.iter().filter(|symbol| {
            symbol.symbol_type() == STT_TLS
                && !symbol.is_undefined()
                && !symbol.name.is_empty()
                && matches!(symbol.binding(), 1 | 2)
        }) {
            let key = process_symbol_key(dynamic, symbol)?;
            if key.nid().is_empty() {
                continue;
            }
            let resolution = TlsSymbolResolution {
                module_id,
                tls_offset: symbol.value.get(),
                static_offset,
            };
            let weak = symbol.is_weak();
            match symbols.get_mut(&key) {
                None => {
                    symbols.insert(key, (resolution, weak));
                }
                Some((current, current_weak)) if *current_weak && !weak => {
                    *current = resolution;
                    *current_weak = weak;
                }
                Some(_) => {}
            }
        }
    }
    Ok(symbols
        .into_iter()
        .map(|(key, (resolution, _))| (key, resolution))
        .collect())
}

pub(super) fn resolve_symbol_at(
    memory: &GuestImage,
    image_base: GuestAddress,
    dynamic: &DynamicInfo,
    symbol_stubs: &BTreeMap<u32, GuestAddress>,
    process_exports: &BTreeMap<ProcessSymbolKey, u64>,
    relocation: &Elf64Rela,
) -> Result<u64, LinkError> {
    let index = usize::try_from(relocation.symbol_index).map_err(|_| LinkError::MissingSymbol {
        index: relocation.symbol_index,
    })?;
    let symbol = dynamic.symbols.get(index).ok_or(LinkError::MissingSymbol {
        index: relocation.symbol_index,
    })?;
    if symbol.is_undefined() {
        let key = process_symbol_key(dynamic, symbol)?;
        let nid = key.nid();
        if let Some(address) = process_exports.get(&key).copied() {
            return Ok(address);
        }
        if let Some(address) = native_import_target(nid) {
            return Ok(address);
        }
        if let Some(address) = synthetic_data_import_target(nid) {
            return Ok(address);
        }
        if symbol.is_weak() {
            return Ok(0);
        }
        return symbol_stubs
            .get(&relocation.symbol_index)
            .map(|address| address.get())
            .ok_or(LinkError::MissingImportStub {
                index: relocation.symbol_index,
            });
    }
    defined_symbol_address_at(memory, image_base, symbol)
}

pub(super) fn defined_symbol_address_at(
    memory: &GuestImage,
    image_base: GuestAddress,
    symbol: &Elf64Symbol,
) -> Result<u64, LinkError> {
    if memory.contains(symbol.value, 1) {
        return Ok(symbol.value.get());
    }
    image_base
        .get()
        .checked_add(symbol.value.get())
        .ok_or(LinkError::AddressOverflow)
}

pub(crate) fn runtime_address_at(
    memory: &GuestImage,
    image_base: GuestAddress,
    value: GuestAddress,
    length: usize,
) -> Result<GuestAddress, LinkError> {
    if memory.contains(value, length) {
        return Ok(value);
    }
    Ok(GuestAddress::new(
        image_base
            .get()
            .checked_add(value.get())
            .ok_or(LinkError::AddressOverflow)?,
    ))
}
