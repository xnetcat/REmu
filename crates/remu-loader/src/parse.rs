// SPDX-License-Identifier: GPL-2.0-or-later

//! SELF container, ELF64 and Sony dynamic-linker metadata parsing.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek};

use remu_core::GuestAddress;

use crate::bytes::{
    checked_end, read_array_at, read_i64_slice, read_u16, read_u16_slice, read_u32, read_u32_slice,
    read_u64, read_u64_slice, read_vec_at,
};
use crate::{
    CONTAINER_HEADER_SIZE, CONTAINER_SEGMENT_SIZE, ContainerHeader, ContainerInfo,
    ContainerSegment, ContainerSegmentProperties, DT_INIT, DT_INIT_ARRAY, DT_INIT_ARRAY_SIZE,
    DT_JUMP_RELOCATION, DT_NEEDED, DT_NULL, DT_PLT_GOT, DT_PLTREL_SIZE, DT_PREINIT_ARRAY,
    DT_PREINIT_ARRAY_SIZE, DT_RELA, DT_RELA_SIZE, DT_SCE_EXPORT_LIBRARY, DT_SCE_EXPORT_LIBRARY_1,
    DT_SCE_IMPORT_LIBRARY, DT_SCE_IMPORT_LIBRARY_1, DT_SCE_JUMP_RELOCATION, DT_SCE_MODULE_INFO,
    DT_SCE_MODULE_INFO_1, DT_SCE_NEEDED_MODULE, DT_SCE_NEEDED_MODULE_1, DT_SCE_ORIGINAL_FILENAME,
    DT_SCE_ORIGINAL_FILENAME_1, DT_SCE_PLTREL_SIZE, DT_SCE_RELA, DT_SCE_RELA_SIZE,
    DT_SCE_STRING_TABLE, DT_SCE_STRING_TABLE_SIZE, DT_SCE_SYMBOL_TABLE, DT_SCE_SYMBOL_TABLE_SIZE,
    DT_STRING_TABLE, DT_STRING_TABLE_SIZE, DT_SYMBOL_TABLE, DynamicEntry, DynamicInfo,
    ELF64_HEADER_SIZE, ELF64_PROGRAM_HEADER_SIZE, ELF64_RELA_SIZE, ELF64_SYMBOL_SIZE, Elf64Header,
    Elf64Info, Elf64ProgramHeader, Elf64Rela, Elf64Symbol, ImportedSymbol, InspectError,
    MAX_CONTAINER_SEGMENTS, MAX_METADATA_TABLE_SIZE, MAX_PROGRAM_HEADERS, PT_DYNAMIC, PT_LOAD,
    SceLibraryReference, SceModuleReference,
};

pub(crate) fn parse_container(
    reader: &mut (impl Read + Seek),
    file_size: u64,
) -> Result<ContainerInfo, InspectError> {
    let bytes = read_array_at::<CONTAINER_HEADER_SIZE>(reader, 0, file_size)?;
    let segment_count = read_u16(&bytes, 0x18);
    if usize::from(segment_count) > MAX_CONTAINER_SEGMENTS {
        return Err(InspectError::TooManyContainerSegments(segment_count));
    }

    let segment_table_size = u64::from(segment_count)
        .checked_mul(CONTAINER_SEGMENT_SIZE as u64)
        .ok_or(InspectError::OffsetOverflow)?;
    let embedded_elf_offset = (CONTAINER_HEADER_SIZE as u64)
        .checked_add(segment_table_size)
        .ok_or(InspectError::OffsetOverflow)?;
    checked_end(embedded_elf_offset, ELF64_HEADER_SIZE as u64, file_size)?;

    let header = ContainerHeader {
        version: bytes[4],
        mode: bytes[5],
        endian: bytes[6],
        attributes: bytes[7],
        key_type: read_u32(&bytes, 8),
        header_size: read_u16(&bytes, 0x0c),
        metadata_size: read_u16(&bytes, 0x0e),
        declared_file_size: read_u64(&bytes, 0x10),
        segment_count,
        flags: read_u16(&bytes, 0x1a),
        reserved: read_u32(&bytes, 0x1c),
        embedded_elf_offset,
    };

    let mut segments = Vec::with_capacity(usize::from(segment_count));
    for index in 0..usize::from(segment_count) {
        let offset = (CONTAINER_HEADER_SIZE as u64)
            .checked_add(
                (index as u64)
                    .checked_mul(CONTAINER_SEGMENT_SIZE as u64)
                    .ok_or(InspectError::OffsetOverflow)?,
            )
            .ok_or(InspectError::OffsetOverflow)?;
        let segment_bytes = read_array_at::<CONTAINER_SEGMENT_SIZE>(reader, offset, file_size)?;
        let segment = ContainerSegment {
            properties: ContainerSegmentProperties::new(read_u64(&segment_bytes, 0)),
            file_offset: read_u64(&segment_bytes, 8),
            stored_size: read_u64(&segment_bytes, 16),
            expanded_size: read_u64(&segment_bytes, 24),
        };
        let end = segment
            .file_offset
            .checked_add(segment.stored_size)
            .ok_or(InspectError::OffsetOverflow)?;
        if end > file_size {
            return Err(InspectError::ContainerSegmentOutOfBounds {
                index,
                end,
                file_size,
            });
        }
        segments.push(segment);
    }

    Ok(ContainerInfo { header, segments })
}

pub(crate) fn parse_elf(
    reader: &mut (impl Read + Seek),
    file_offset: u64,
    file_size: u64,
) -> Result<Elf64Info, InspectError> {
    let bytes = read_array_at::<ELF64_HEADER_SIZE>(reader, file_offset, file_size)?;
    if bytes[4] != 2 {
        return Err(InspectError::UnsupportedElfClass(bytes[4]));
    }
    if bytes[5] != 1 {
        return Err(InspectError::UnsupportedElfByteOrder(bytes[5]));
    }

    let program_header_entry_size = read_u16(&bytes, 0x36);
    let program_header_count = read_u16(&bytes, 0x38);
    if usize::from(program_header_count) > MAX_PROGRAM_HEADERS {
        return Err(InspectError::TooManyProgramHeaders(program_header_count));
    }
    if program_header_count != 0
        && usize::from(program_header_entry_size) < ELF64_PROGRAM_HEADER_SIZE
    {
        return Err(InspectError::UnsupportedProgramHeaderSize(
            program_header_entry_size,
        ));
    }

    let header = Elf64Header {
        os_abi: bytes[7],
        abi_version: bytes[8],
        object_type: read_u16(&bytes, 0x10),
        machine: read_u16(&bytes, 0x12),
        entry: GuestAddress::new(read_u64(&bytes, 0x18)),
        program_header_offset: read_u64(&bytes, 0x20),
        section_header_offset: read_u64(&bytes, 0x28),
        flags: read_u32(&bytes, 0x30),
        header_size: read_u16(&bytes, 0x34),
        program_header_entry_size,
        program_header_count,
        section_header_entry_size: read_u16(&bytes, 0x3a),
        section_header_count: read_u16(&bytes, 0x3c),
        section_name_index: read_u16(&bytes, 0x3e),
    };

    let table_offset = file_offset
        .checked_add(header.program_header_offset)
        .ok_or(InspectError::OffsetOverflow)?;
    let table_size = u64::from(program_header_entry_size)
        .checked_mul(u64::from(program_header_count))
        .ok_or(InspectError::OffsetOverflow)?;
    checked_end(table_offset, table_size, file_size)?;

    let mut program_headers = Vec::with_capacity(usize::from(program_header_count));
    for index in 0..usize::from(program_header_count) {
        let entry_offset = table_offset
            .checked_add(
                (index as u64)
                    .checked_mul(u64::from(program_header_entry_size))
                    .ok_or(InspectError::OffsetOverflow)?,
            )
            .ok_or(InspectError::OffsetOverflow)?;
        let entry = read_array_at::<ELF64_PROGRAM_HEADER_SIZE>(reader, entry_offset, file_size)?;
        program_headers.push(Elf64ProgramHeader {
            program_type: read_u32(&entry, 0),
            flags: read_u32(&entry, 4),
            file_offset: read_u64(&entry, 8),
            virtual_address: GuestAddress::new(read_u64(&entry, 16)),
            physical_address: GuestAddress::new(read_u64(&entry, 24)),
            file_size: read_u64(&entry, 32),
            memory_size: read_u64(&entry, 40),
            alignment: read_u64(&entry, 48),
        });
    }

    Ok(Elf64Info {
        file_offset,
        header,
        program_headers,
        dynamic: None,
    })
}

pub(crate) fn parse_dynamic(
    reader: &mut (impl Read + Seek),
    file_size: u64,
    container: Option<&ContainerInfo>,
    elf: &Elf64Info,
) -> Result<Option<DynamicInfo>, InspectError> {
    let Some(dynamic_program) = elf
        .program_headers
        .iter()
        .find(|program| program.program_type == PT_DYNAMIC)
    else {
        return Ok(None);
    };
    if dynamic_program.file_size == 0 {
        return Ok(None);
    }

    let dynamic_bytes = read_elf_file_range(
        reader,
        file_size,
        container,
        elf,
        dynamic_program.file_offset,
        dynamic_program.file_size,
        "dynamic",
    )?;
    let mut entries = Vec::with_capacity(dynamic_bytes.len() / 16);
    for entry in dynamic_bytes.chunks_exact(16) {
        let tag = read_i64_slice(entry, 0);
        let value = read_u64_slice(entry, 8);
        entries.push(DynamicEntry { tag, value });
        if tag == DT_NULL {
            break;
        }
    }

    let mut info = summarize_dynamic_entries(entries);

    let mut string_table = Vec::new();
    if info.string_table_address != 0 && info.string_table_size != 0 {
        string_table = read_elf_virtual_range(
            reader,
            file_size,
            container,
            elf,
            info.string_table_address,
            info.string_table_size,
            "dynamic string",
        )?;
        for entry in &info.entries {
            match entry.tag {
                DT_NEEDED => info
                    .needed_libraries
                    .push(read_dynamic_string(&string_table, entry.value)?),
                DT_SCE_ORIGINAL_FILENAME | DT_SCE_ORIGINAL_FILENAME_1 => {
                    info.original_filename = Some(read_dynamic_string(&string_table, entry.value)?);
                }
                DT_SCE_NEEDED_MODULE | DT_SCE_NEEDED_MODULE_1 => info
                    .required_modules
                    .push(read_sce_module_reference(&string_table, entry.value)?),
                DT_SCE_MODULE_INFO | DT_SCE_MODULE_INFO_1 => info
                    .exported_modules
                    .push(read_sce_module_reference(&string_table, entry.value)?),
                DT_SCE_IMPORT_LIBRARY | DT_SCE_IMPORT_LIBRARY_1 => info
                    .imported_libraries
                    .push(read_sce_library_reference(&string_table, entry.value)?),
                DT_SCE_EXPORT_LIBRARY | DT_SCE_EXPORT_LIBRARY_1 => info
                    .exported_libraries
                    .push(read_sce_library_reference(&string_table, entry.value)?),
                _ => {}
            }
        }
    }

    populate_linker_metadata(reader, file_size, container, elf, &string_table, &mut info)?;

    Ok(Some(info))
}

fn summarize_dynamic_entries(entries: Vec<DynamicEntry>) -> DynamicInfo {
    let mut info = DynamicInfo {
        entries,
        needed_libraries: Vec::new(),
        original_filename: None,
        required_modules: Vec::new(),
        exported_modules: Vec::new(),
        imported_libraries: Vec::new(),
        exported_libraries: Vec::new(),
        string_table_address: 0,
        string_table_size: 0,
        symbol_table_address: 0,
        symbol_table_size: 0,
        rela_address: 0,
        rela_size: 0,
        jump_relocation_address: 0,
        jump_relocation_size: 0,
        plt_got_address: 0,
        init_address: 0,
        init_array_address: 0,
        init_array_size: 0,
        preinit_array_address: 0,
        preinit_array_size: 0,
        symbols: Vec::new(),
        relocations: Vec::new(),
        jump_relocations: Vec::new(),
        imports: Vec::new(),
    };

    for entry in &info.entries {
        match entry.tag {
            DT_STRING_TABLE if info.string_table_address == 0 => {
                info.string_table_address = entry.value;
            }
            DT_STRING_TABLE_SIZE if info.string_table_size == 0 => {
                info.string_table_size = entry.value;
            }
            DT_SYMBOL_TABLE if info.symbol_table_address == 0 => {
                info.symbol_table_address = entry.value;
            }
            DT_RELA if info.rela_address == 0 => info.rela_address = entry.value,
            DT_RELA_SIZE if info.rela_size == 0 => info.rela_size = entry.value,
            DT_JUMP_RELOCATION if info.jump_relocation_address == 0 => {
                info.jump_relocation_address = entry.value;
            }
            DT_PLTREL_SIZE if info.jump_relocation_size == 0 => {
                info.jump_relocation_size = entry.value;
            }
            DT_PLT_GOT if info.plt_got_address == 0 => info.plt_got_address = entry.value,
            DT_INIT if info.init_address == 0 => info.init_address = entry.value,
            DT_INIT_ARRAY if info.init_array_address == 0 => {
                info.init_array_address = entry.value;
            }
            DT_INIT_ARRAY_SIZE if info.init_array_size == 0 => {
                info.init_array_size = entry.value;
            }
            DT_PREINIT_ARRAY if info.preinit_array_address == 0 => {
                info.preinit_array_address = entry.value;
            }
            DT_PREINIT_ARRAY_SIZE if info.preinit_array_size == 0 => {
                info.preinit_array_size = entry.value;
            }
            DT_SCE_STRING_TABLE => info.string_table_address = entry.value,
            DT_SCE_STRING_TABLE_SIZE => info.string_table_size = entry.value,
            DT_SCE_SYMBOL_TABLE => info.symbol_table_address = entry.value,
            DT_SCE_SYMBOL_TABLE_SIZE => info.symbol_table_size = entry.value,
            DT_SCE_RELA => info.rela_address = entry.value,
            DT_SCE_RELA_SIZE => info.rela_size = entry.value,
            DT_SCE_JUMP_RELOCATION => info.jump_relocation_address = entry.value,
            DT_SCE_PLTREL_SIZE => info.jump_relocation_size = entry.value,
            _ => {}
        }
    }

    info
}

fn populate_linker_metadata(
    reader: &mut (impl Read + Seek),
    file_size: u64,
    container: Option<&ContainerInfo>,
    elf: &Elf64Info,
    string_table: &[u8],
    info: &mut DynamicInfo,
) -> Result<(), InspectError> {
    info.relocations = read_relocation_table(
        reader,
        file_size,
        container,
        elf,
        info.rela_address,
        info.rela_size,
        "RELA",
    )?;
    info.jump_relocations = read_relocation_table(
        reader,
        file_size,
        container,
        elf,
        info.jump_relocation_address,
        info.jump_relocation_size,
        "jump relocation",
    )?;

    let max_symbol_index = info
        .relocations
        .iter()
        .chain(&info.jump_relocations)
        .map(|relocation| relocation.symbol_index)
        .max()
        .unwrap_or(0);
    let symbol_table_size = if info.symbol_table_size != 0 {
        info.symbol_table_size
    } else if max_symbol_index != 0 {
        u64::from(max_symbol_index + 1)
            .checked_mul(ELF64_SYMBOL_SIZE as u64)
            .ok_or(InspectError::OffsetOverflow)?
    } else {
        0
    };
    if info.symbol_table_address != 0 && symbol_table_size != 0 {
        let symbol_bytes = read_elf_virtual_range(
            reader,
            file_size,
            container,
            elf,
            info.symbol_table_address,
            symbol_table_size,
            "dynamic symbol",
        )?;
        info.symbols = parse_symbol_table(&symbol_bytes, string_table)?;
    }

    info.imports = build_import_inventory(
        &info.symbols,
        info.relocations.iter().chain(&info.jump_relocations),
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn read_relocation_table(
    reader: &mut (impl Read + Seek),
    file_size: u64,
    container: Option<&ContainerInfo>,
    elf: &Elf64Info,
    address: u64,
    size: u64,
    table: &'static str,
) -> Result<Vec<Elf64Rela>, InspectError> {
    if address == 0 || size == 0 {
        return Ok(Vec::new());
    }
    if !size.is_multiple_of(ELF64_RELA_SIZE as u64) {
        return Err(InspectError::InvalidMetadataTableSize {
            table,
            size,
            entry_size: ELF64_RELA_SIZE,
        });
    }
    let bytes = read_elf_virtual_range(reader, file_size, container, elf, address, size, table)?;
    Ok(bytes
        .chunks_exact(ELF64_RELA_SIZE)
        .map(|entry| {
            let info = read_u64_slice(entry, 8);
            let info_bytes = info.to_le_bytes();
            Elf64Rela {
                offset: GuestAddress::new(read_u64_slice(entry, 0)),
                symbol_index: (info >> 32) as u32,
                relocation_type: u32::from_le_bytes([
                    info_bytes[0],
                    info_bytes[1],
                    info_bytes[2],
                    info_bytes[3],
                ]),
                addend: read_i64_slice(entry, 16),
            }
        })
        .collect())
}

fn parse_symbol_table(bytes: &[u8], string_table: &[u8]) -> Result<Vec<Elf64Symbol>, InspectError> {
    if !bytes.len().is_multiple_of(ELF64_SYMBOL_SIZE) {
        return Err(InspectError::InvalidMetadataTableSize {
            table: "dynamic symbol",
            size: bytes.len() as u64,
            entry_size: ELF64_SYMBOL_SIZE,
        });
    }
    bytes
        .chunks_exact(ELF64_SYMBOL_SIZE)
        .map(|entry| {
            let name_offset = read_u32_slice(entry, 0);
            Ok(Elf64Symbol {
                name_offset,
                info: entry[4],
                other: entry[5],
                section_index: read_u16_slice(entry, 6),
                value: GuestAddress::new(read_u64_slice(entry, 8)),
                size: read_u64_slice(entry, 16),
                name: if name_offset == 0 {
                    String::new()
                } else {
                    read_dynamic_string(string_table, u64::from(name_offset))?
                },
            })
        })
        .collect()
}

fn build_import_inventory<'a>(
    symbols: &[Elf64Symbol],
    relocations: impl Iterator<Item = &'a Elf64Rela>,
) -> Result<Vec<ImportedSymbol>, InspectError> {
    let mut imports: BTreeMap<u32, (usize, BTreeSet<u32>)> = BTreeMap::new();
    for relocation in relocations {
        if relocation.symbol_index == 0 {
            continue;
        }
        let symbol_index =
            usize::try_from(relocation.symbol_index).map_err(|_| InspectError::OffsetOverflow)?;
        let Some(symbol) = symbols.get(symbol_index) else {
            return Err(InspectError::RelocationSymbolOutOfBounds {
                index: relocation.symbol_index,
                count: symbols.len(),
            });
        };
        if !symbol.is_undefined() || symbol.name.is_empty() {
            continue;
        }
        let entry = imports
            .entry(relocation.symbol_index)
            .or_insert_with(|| (0, BTreeSet::new()));
        entry.0 += 1;
        entry.1.insert(relocation.relocation_type);
    }

    imports
        .into_iter()
        .map(|(symbol_index, (relocation_count, relocation_types))| {
            let index = usize::try_from(symbol_index).map_err(|_| InspectError::OffsetOverflow)?;
            let symbol = &symbols[index];
            Ok(ImportedSymbol {
                symbol_index,
                name: symbol.name.clone(),
                relocation_count,
                relocation_types: relocation_types.into_iter().collect(),
                weak: symbol.is_weak(),
            })
        })
        .collect()
}

fn read_elf_virtual_range(
    reader: &mut (impl Read + Seek),
    file_size: u64,
    container: Option<&ContainerInfo>,
    elf: &Elf64Info,
    address: u64,
    size: u64,
    table: &'static str,
) -> Result<Vec<u8>, InspectError> {
    let end = address
        .checked_add(size)
        .ok_or(InspectError::OffsetOverflow)?;
    let Some(program) = elf.program_headers.iter().find(|program| {
        if program.program_type != PT_LOAD {
            return false;
        }
        let Some(program_end) = program.virtual_address.get().checked_add(program.file_size) else {
            return false;
        };
        address >= program.virtual_address.get() && end <= program_end
    }) else {
        return Err(InspectError::UnmappedVirtualRange { address, end });
    };
    let delta = address - program.virtual_address.get();
    let logical_offset = program
        .file_offset
        .checked_add(delta)
        .ok_or(InspectError::OffsetOverflow)?;
    read_elf_file_range(
        reader,
        file_size,
        container,
        elf,
        logical_offset,
        size,
        table,
    )
}

fn read_elf_file_range(
    reader: &mut (impl Read + Seek),
    file_size: u64,
    container: Option<&ContainerInfo>,
    elf: &Elf64Info,
    logical_offset: u64,
    size: u64,
    table: &'static str,
) -> Result<Vec<u8>, InspectError> {
    read_elf_file_range_bounded(
        reader,
        file_size,
        container,
        elf,
        logical_offset,
        size,
        table,
        MAX_METADATA_TABLE_SIZE,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn read_elf_file_range_bounded(
    reader: &mut (impl Read + Seek),
    file_size: u64,
    container: Option<&ContainerInfo>,
    elf: &Elf64Info,
    logical_offset: u64,
    size: u64,
    table: &'static str,
    maximum_size: u64,
) -> Result<Vec<u8>, InspectError> {
    if size > maximum_size {
        return Err(InspectError::MetadataTableTooLarge { table, size });
    }
    let logical_end = logical_offset
        .checked_add(size)
        .ok_or(InspectError::OffsetOverflow)?;

    let physical_offset = if let Some(container) = container {
        let mut matched = None;
        for segment in &container.segments {
            if !segment.properties.has_blocks() {
                continue;
            }
            let program_index = segment.properties.program_index();
            let Some(program) = elf.program_headers.get(program_index) else {
                continue;
            };
            let Some(program_end) = program.file_offset.checked_add(program.file_size) else {
                continue;
            };
            if logical_offset < program.file_offset || logical_end > program_end {
                continue;
            }
            let compressed = segment.stored_size != segment.expanded_size;
            if segment.properties.is_encrypted() || compressed {
                return Err(InspectError::UnsupportedContainerPayload {
                    program_index,
                    encrypted: segment.properties.is_encrypted(),
                    compressed,
                });
            }
            let delta = logical_offset - program.file_offset;
            let stored_end = delta
                .checked_add(size)
                .ok_or(InspectError::OffsetOverflow)?;
            if stored_end <= segment.stored_size {
                matched = Some(
                    segment
                        .file_offset
                        .checked_add(delta)
                        .ok_or(InspectError::OffsetOverflow)?,
                );
                break;
            }
        }
        matched.ok_or(InspectError::UnmappedElfFileRange {
            offset: logical_offset,
            end: logical_end,
        })?
    } else {
        elf.file_offset
            .checked_add(logical_offset)
            .ok_or(InspectError::OffsetOverflow)?
    };

    read_vec_at(reader, physical_offset, size, file_size)
}

fn read_dynamic_string(bytes: &[u8], offset: u64) -> Result<String, InspectError> {
    let offset_usize =
        usize::try_from(offset).map_err(|_| InspectError::DynamicStringOutOfBounds { offset })?;
    let suffix = bytes
        .get(offset_usize..)
        .ok_or(InspectError::DynamicStringOutOfBounds { offset })?;
    let length = suffix
        .iter()
        .position(|byte| *byte == 0)
        .ok_or(InspectError::UnterminatedDynamicString { offset })?;
    let value = std::str::from_utf8(&suffix[..length])
        .map_err(|_| InspectError::InvalidDynamicString { offset })?;
    Ok(value.to_owned())
}

fn read_sce_module_reference(
    string_table: &[u8],
    value: u64,
) -> Result<SceModuleReference, InspectError> {
    let bytes = value.to_le_bytes();
    Ok(SceModuleReference {
        name: read_dynamic_string(
            string_table,
            u64::from(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
        )?,
        id: u16::from_le_bytes([bytes[6], bytes[7]]),
        version_major: bytes[5],
        version_minor: bytes[4],
    })
}

fn read_sce_library_reference(
    string_table: &[u8],
    value: u64,
) -> Result<SceLibraryReference, InspectError> {
    let bytes = value.to_le_bytes();
    Ok(SceLibraryReference {
        name: read_dynamic_string(
            string_table,
            u64::from(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
        )?,
        id: u16::from_le_bytes([bytes[6], bytes[7]]),
        version: u16::from_le_bytes([bytes[4], bytes[5]]),
    })
}
