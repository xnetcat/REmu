// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::{BTreeMap, BTreeSet};

use remu_core::GuestAddress;
use remu_loader::{DynamicInfo, Elf64Rela, Elf64Symbol, LoadedExecutable};
use remu_memory::{GUEST_EXECUTE, GUEST_READ, GuestImage, MappingError};
use thiserror::Error;

const R_X86_64_64: u32 = 1;
const R_X86_64_GLOB_DAT: u32 = 6;
const R_X86_64_JUMP_SLOT: u32 = 7;
const R_X86_64_RELATIVE: u32 = 8;
const IMPORT_STUB_SIZE: u64 = 16;
const IMPORT_STUB_SIZE_USIZE: usize = 16;
const IMPORT_TRAP_OPCODE: u8 = 0xcc;
const IMPORT_RETURN_OPCODE: u8 = 0xc3;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportStub {
    pub nid: String,
    pub address: GuestAddress,
}

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

#[derive(Debug)]
pub struct LinkedImage {
    pub memory: GuestImage,
    pub import_stubs: Vec<ImportStub>,
    pub relocation_stats: RelocationStats,
}

#[derive(Debug, Error)]
pub enum LinkError {
    #[error(transparent)]
    Mapping(#[from] MappingError),
    #[error("executable does not contain an ELF image")]
    MissingElf,
    #[error("executable does not contain dynamic-linker metadata")]
    MissingDynamicMetadata,
    #[error("import stub allocation size overflow")]
    StubAllocationOverflow,
    #[error("relocation references missing symbol {index}")]
    MissingSymbol { index: u32 },
    #[error("unresolved non-weak symbol {index} has no import stub")]
    MissingImportStub { index: u32 },
    #[error("unsupported x86-64 relocation type {relocation_type} at {offset}")]
    UnsupportedRelocation {
        relocation_type: u32,
        offset: GuestAddress,
    },
    #[error("guest address arithmetic overflow")]
    AddressOverflow,
}

impl LinkedImage {
    pub fn link_main(executable: &LoadedExecutable) -> Result<Self, LinkError> {
        let elf = executable.info.elf.as_ref().ok_or(LinkError::MissingElf)?;
        let dynamic = elf
            .dynamic
            .as_ref()
            .ok_or(LinkError::MissingDynamicMetadata)?;
        let mut memory = GuestImage::map_main(executable)?;
        let (import_stubs, symbol_stubs) = create_import_stubs(&mut memory, dynamic)?;
        let relocation_stats = apply_relocations(&mut memory, dynamic, &symbol_stubs)?;
        memory.finalize_permissions()?;
        Ok(Self {
            memory,
            import_stubs,
            relocation_stats,
        })
    }
}

fn create_import_stubs(
    memory: &mut GuestImage,
    dynamic: &DynamicInfo,
) -> Result<(Vec<ImportStub>, BTreeMap<u32, GuestAddress>), LinkError> {
    let nids = dynamic
        .imports
        .iter()
        .filter(|import| !import.weak)
        .map(|import| extract_nid(&import.name).to_owned())
        .collect::<BTreeSet<_>>();
    if nids.is_empty() {
        return Ok((Vec::new(), BTreeMap::new()));
    }

    let stub_bytes = u64::try_from(nids.len())
        .map_err(|_| LinkError::StubAllocationOverflow)?
        .checked_mul(IMPORT_STUB_SIZE)
        .ok_or(LinkError::StubAllocationOverflow)?;
    let stub_base =
        memory.allocate_staging(stub_bytes, IMPORT_STUB_SIZE, GUEST_READ | GUEST_EXECUTE)?;
    let bytes_len = usize::try_from(stub_bytes).map_err(|_| LinkError::StubAllocationOverflow)?;
    let mut bytes = vec![0_u8; bytes_len];
    let mut import_stubs = Vec::with_capacity(nids.len());
    let mut by_nid = BTreeMap::new();
    for (index, nid) in nids.into_iter().enumerate() {
        let byte_offset = index * IMPORT_STUB_SIZE_USIZE;
        bytes[byte_offset] = IMPORT_TRAP_OPCODE;
        bytes[byte_offset + 1] = IMPORT_RETURN_OPCODE;
        bytes[byte_offset + 8..byte_offset + 16].copy_from_slice(&(index as u64).to_le_bytes());
        let address = GuestAddress::new(
            stub_base
                .get()
                .checked_add(byte_offset as u64)
                .ok_or(LinkError::AddressOverflow)?,
        );
        by_nid.insert(nid.clone(), address);
        import_stubs.push(ImportStub { nid, address });
    }
    memory.write(stub_base, &bytes)?;

    let mut symbol_stubs = BTreeMap::new();
    for import in &dynamic.imports {
        if import.weak {
            continue;
        }
        let nid = extract_nid(&import.name);
        let address = by_nid
            .get(nid)
            .copied()
            .ok_or(LinkError::MissingImportStub {
                index: import.symbol_index,
            })?;
        symbol_stubs.insert(import.symbol_index, address);
    }
    Ok((import_stubs, symbol_stubs))
}

fn apply_relocations(
    memory: &mut GuestImage,
    dynamic: &DynamicInfo,
    symbol_stubs: &BTreeMap<u32, GuestAddress>,
) -> Result<RelocationStats, LinkError> {
    let mut stats = RelocationStats::default();
    for relocation in dynamic.relocations.iter().chain(&dynamic.jump_relocations) {
        let target = runtime_address(memory, relocation.offset, 8)?;
        let value = match relocation.relocation_type {
            R_X86_64_RELATIVE => {
                stats.relative += 1;
                memory
                    .image_base()
                    .get()
                    .wrapping_add_signed(relocation.addend)
            }
            R_X86_64_64 => {
                stats.absolute += 1;
                resolve_symbol(memory, dynamic, symbol_stubs, relocation)?
                    .wrapping_add_signed(relocation.addend)
            }
            R_X86_64_GLOB_DAT => {
                stats.global_data += 1;
                resolve_symbol(memory, dynamic, symbol_stubs, relocation)?
            }
            R_X86_64_JUMP_SLOT => {
                stats.jump_slots += 1;
                resolve_symbol(memory, dynamic, symbol_stubs, relocation)?
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

fn resolve_symbol(
    memory: &GuestImage,
    dynamic: &DynamicInfo,
    symbol_stubs: &BTreeMap<u32, GuestAddress>,
    relocation: &Elf64Rela,
) -> Result<u64, LinkError> {
    let index = usize::try_from(relocation.symbol_index).map_err(|_| LinkError::MissingSymbol {
        index: relocation.symbol_index,
    })?;
    let symbol = dynamic.symbols.get(index).ok_or(LinkError::MissingSymbol {
        index: relocation.symbol_index,
    })?;
    if symbol.is_undefined() {
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
    defined_symbol_address(memory, symbol)
}

fn defined_symbol_address(memory: &GuestImage, symbol: &Elf64Symbol) -> Result<u64, LinkError> {
    if memory.contains(symbol.value, 1) {
        return Ok(symbol.value.get());
    }
    memory
        .image_base()
        .get()
        .checked_add(symbol.value.get())
        .ok_or(LinkError::AddressOverflow)
}

fn runtime_address(
    memory: &GuestImage,
    value: GuestAddress,
    length: usize,
) -> Result<GuestAddress, LinkError> {
    if memory.contains(value, length) {
        return Ok(value);
    }
    Ok(GuestAddress::new(
        memory
            .image_base()
            .get()
            .checked_add(value.get())
            .ok_or(LinkError::AddressOverflow)?,
    ))
}

fn extract_nid(symbol_name: &str) -> &str {
    symbol_name
        .split_once('#')
        .map_or(symbol_name, |(nid, _)| nid)
}

#[cfg(all(test, target_os = "macos", target_arch = "x86_64"))]
mod tests {
    use std::io::Cursor;

    use remu_core::GuestAddress;
    use remu_loader::load_reader;

    use super::LinkedImage;

    #[test]
    fn links_relative_and_import_relocations_into_fixed_memory() {
        let mut bytes = vec![0_u8; 0x300];
        write_elf_header(&mut bytes);
        write_program_header(&mut bytes, 0, 1, 6, 0x100, 0x4000, 0x130, 0x200);
        write_program_header(&mut bytes, 1, 2, 6, 0x100, 0x4000, 0x70, 0x70);
        write_dynamic(&mut bytes[0x100..]);
        bytes[0x180..0x18d].copy_from_slice(b"\0testNid#A#A\0");
        bytes[0x1d8..0x1dc].copy_from_slice(&1_u32.to_le_bytes());
        bytes[0x1dc] = 0x12;
        write_relocation(&mut bytes[0x1f0..], 0, 0x4180, 8, 0x1234);
        write_relocation(&mut bytes[0x1f0..], 1, 0x4188, (1_u64 << 32) | 7, 0);
        let executable = load_reader(&mut Cursor::new(bytes)).unwrap();

        let linked = LinkedImage::link_main(&executable).unwrap();
        assert_eq!(linked.relocation_stats.relative, 1);
        assert_eq!(linked.relocation_stats.jump_slots, 1);
        assert_eq!(linked.import_stubs.len(), 1);
        let base = linked.memory.image_base().get();
        let mut relative = [0_u8; 8];
        linked
            .memory
            .read(GuestAddress::new(base + 0x4180), &mut relative)
            .unwrap();
        assert_eq!(u64::from_le_bytes(relative), base + 0x1234);
        let mut import = [0_u8; 8];
        linked
            .memory
            .read(GuestAddress::new(base + 0x4188), &mut import)
            .unwrap();
        assert_eq!(
            u64::from_le_bytes(import),
            linked.import_stubs[0].address.get()
        );
    }

    fn write_elf_header(bytes: &mut [u8]) {
        bytes[..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[0x12..0x14].copy_from_slice(&62_u16.to_le_bytes());
        bytes[0x18..0x20].copy_from_slice(&0x4000_u64.to_le_bytes());
        bytes[0x20..0x28].copy_from_slice(&64_u64.to_le_bytes());
        bytes[0x34..0x36].copy_from_slice(&64_u16.to_le_bytes());
        bytes[0x36..0x38].copy_from_slice(&56_u16.to_le_bytes());
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
        memory_size: u64,
    ) {
        let offset = 64 + index * 56;
        bytes[offset..offset + 4].copy_from_slice(&program_type.to_le_bytes());
        bytes[offset + 4..offset + 8].copy_from_slice(&flags.to_le_bytes());
        bytes[offset + 8..offset + 16].copy_from_slice(&file_offset.to_le_bytes());
        bytes[offset + 16..offset + 24].copy_from_slice(&virtual_address.to_le_bytes());
        bytes[offset + 32..offset + 40].copy_from_slice(&file_size.to_le_bytes());
        bytes[offset + 40..offset + 48].copy_from_slice(&memory_size.to_le_bytes());
        bytes[offset + 48..offset + 56].copy_from_slice(&0x4000_u64.to_le_bytes());
    }

    fn write_dynamic(bytes: &mut [u8]) {
        let entries = [
            (5_i64, 0x4080_u64),
            (10, 0x40),
            (6, 0x40c0),
            (0x6100_003f, 0x30),
            (7, 0x40f0),
            (8, 0x30),
            (0, 0),
        ];
        for (index, (tag, value)) in entries.into_iter().enumerate() {
            let offset = index * 16;
            bytes[offset..offset + 8].copy_from_slice(&tag.to_le_bytes());
            bytes[offset + 8..offset + 16].copy_from_slice(&value.to_le_bytes());
        }
    }

    fn write_relocation(bytes: &mut [u8], index: usize, offset: u64, info: u64, addend: i64) {
        let entry = index * 24;
        bytes[entry..entry + 8].copy_from_slice(&offset.to_le_bytes());
        bytes[entry + 8..entry + 16].copy_from_slice(&info.to_le_bytes());
        bytes[entry + 16..entry + 24].copy_from_slice(&addend.to_le_bytes());
    }
}
