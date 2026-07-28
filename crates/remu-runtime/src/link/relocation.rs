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

pub(super) const R_X86_64_DTPMOD64: u32 = 16;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RelocationStats {
    pub absolute: usize,
    pub global_data: usize,
    pub jump_slots: usize,
    pub relative: usize,
}

impl RelocationStats {
    #[must_use]
    pub const fn total(self) -> usize {
        self.absolute + self.global_data + self.jump_slots + self.relative
    }
}

pub(super) fn add_relocation_stats(total: &mut RelocationStats, addition: RelocationStats) {
    total.absolute += addition.absolute;
    total.global_data += addition.global_data;
    total.jump_slots += addition.jump_slots;
    total.relative += addition.relative;
}

pub(super) fn apply_relocations_at(
    memory: &mut GuestImage,
    image_base: GuestAddress,
    tls_module_id: u64,
    dynamic: &DynamicInfo,
    symbol_stubs: &BTreeMap<u32, GuestAddress>,
    process_exports: &BTreeMap<ProcessSymbolKey, u64>,
) -> Result<RelocationStats, LinkError> {
    let mut stats = RelocationStats::default();
    for relocation in dynamic.relocations.iter().chain(&dynamic.jump_relocations) {
        let target = runtime_address_at(memory, image_base, relocation.offset, 8)?;
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
            R_X86_64_DTPMOD64 => tls_module_id,
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
