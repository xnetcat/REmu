// SPDX-License-Identifier: GPL-2.0-or-later

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

mod bytes;
use bytes::read_array_at;

mod parse;
use parse::{parse_container, parse_dynamic, parse_elf, read_elf_file_range_bounded};

mod dependency_graph;
pub use dependency_graph::{
    ModuleDependencyGraph, ModuleDependencyGraphBuilder, ModuleDependencyGraphError,
    ModuleDependencyNode, UnresolvedModuleDependency, discover_app_root,
    resolve_module_dependency_graph,
};

mod error;
pub use error::InspectError;

mod model;
pub use model::{
    ContainerHeader, ContainerInfo, ContainerSegment, ContainerSegmentProperties, DynamicEntry,
    DynamicInfo, Elf64Header, Elf64Info, Elf64ProgramHeader, Elf64Rela, Elf64Symbol,
    ExecutableInfo, ExecutableKind, ImportedSymbol, LoadSegmentImage, LoadedExecutable,
    SceLibraryReference, SceModuleReference,
};

const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const PS5_CONTAINER_MAGIC: [u8; 4] = [0x4f, 0x15, 0x3d, 0x1d];
const PS5_RETAIL_CONTAINER_MAGIC: [u8; 4] = [0x54, 0x14, 0xf5, 0xee];
const ELF64_HEADER_SIZE: usize = 64;
const ELF64_PROGRAM_HEADER_SIZE: usize = 56;
const ELF64_SYMBOL_SIZE: usize = 24;
const ELF64_RELA_SIZE: usize = 24;
const CONTAINER_HEADER_SIZE: usize = 32;
const CONTAINER_SEGMENT_SIZE: usize = 32;
const MAX_CONTAINER_SEGMENTS: usize = 4096;
const MAX_PROGRAM_HEADERS: usize = 4096;
const MAX_METADATA_TABLE_SIZE: u64 = 64 * 1024 * 1024;
const MAX_LOAD_SEGMENT_SIZE: u64 = 512 * 1024 * 1024;
const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const DT_NULL: i64 = 0;
const DT_NEEDED: i64 = 1;
const DT_PLTREL_SIZE: i64 = 2;
const DT_PLT_GOT: i64 = 3;
const DT_STRING_TABLE: i64 = 5;
const DT_SYMBOL_TABLE: i64 = 6;
const DT_RELA: i64 = 7;
const DT_RELA_SIZE: i64 = 8;
const DT_STRING_TABLE_SIZE: i64 = 10;
const DT_INIT: i64 = 12;
const DT_JUMP_RELOCATION: i64 = 23;
const DT_INIT_ARRAY: i64 = 25;
const DT_INIT_ARRAY_SIZE: i64 = 27;
const DT_PREINIT_ARRAY: i64 = 32;
const DT_PREINIT_ARRAY_SIZE: i64 = 33;
const DT_SCE_ORIGINAL_FILENAME: i64 = 0x6100_0009;
const DT_SCE_MODULE_INFO: i64 = 0x6100_000d;
const DT_SCE_NEEDED_MODULE: i64 = 0x6100_000f;
const DT_SCE_EXPORT_LIBRARY: i64 = 0x6100_0013;
const DT_SCE_IMPORT_LIBRARY: i64 = 0x6100_0015;
const DT_SCE_JUMP_RELOCATION: i64 = 0x6100_0029;
const DT_SCE_PLTREL_SIZE: i64 = 0x6100_002d;
const DT_SCE_RELA: i64 = 0x6100_002f;
const DT_SCE_RELA_SIZE: i64 = 0x6100_0031;
const DT_SCE_STRING_TABLE: i64 = 0x6100_0035;
const DT_SCE_STRING_TABLE_SIZE: i64 = 0x6100_0037;
const DT_SCE_SYMBOL_TABLE: i64 = 0x6100_0039;
const DT_SCE_SYMBOL_TABLE_SIZE: i64 = 0x6100_003f;
const DT_SCE_ORIGINAL_FILENAME_1: i64 = 0x6100_0041;
const DT_SCE_MODULE_INFO_1: i64 = 0x6100_0043;
const DT_SCE_NEEDED_MODULE_1: i64 = 0x6100_0045;
const DT_SCE_EXPORT_LIBRARY_1: i64 = 0x6100_0047;
const DT_SCE_IMPORT_LIBRARY_1: i64 = 0x6100_0049;

pub fn inspect_path(path: impl AsRef<Path>) -> Result<ExecutableInfo, InspectError> {
    let mut file = open_executable(path.as_ref())?;
    inspect_reader(&mut file)
}

pub fn load_path(path: impl AsRef<Path>) -> Result<LoadedExecutable, InspectError> {
    let mut file = open_executable(path.as_ref())?;
    load_reader(&mut file)
}

fn open_executable(path: &Path) -> Result<File, InspectError> {
    // On Windows, CreateFile on a directory returns ERROR_ACCESS_DENIED (os error 5) rather than
    // "is a directory", which looks like a permissions mystery. Check first so the message is clear.
    if path.is_dir() {
        return Err(InspectError::PathIsDirectory {
            path: path.to_path_buf(),
        });
    }
    File::open(path).map_err(|source| InspectError::Open {
        path: path.to_path_buf(),
        source,
    })
}

pub fn load_reader(reader: &mut (impl Read + Seek)) -> Result<LoadedExecutable, InspectError> {
    let info = inspect_reader(reader)?;
    let Some(elf) = info.elf.as_ref() else {
        return Ok(LoadedExecutable {
            info,
            load_segments: Vec::new(),
        });
    };
    let mut load_segments = Vec::new();
    for (program_index, program) in elf.program_headers.iter().enumerate() {
        if program.program_type != PT_LOAD || program.memory_size == 0 {
            continue;
        }
        if program.file_size > program.memory_size {
            return Err(InspectError::SegmentFileLargerThanMemory {
                program_index,
                file_size: program.file_size,
                memory_size: program.memory_size,
            });
        }
        let data = if program.file_size == 0 {
            Vec::new()
        } else {
            read_elf_file_range_bounded(
                reader,
                info.file_size,
                info.container.as_ref(),
                elf,
                program.file_offset,
                program.file_size,
                "load segment",
                MAX_LOAD_SEGMENT_SIZE,
            )?
        };
        load_segments.push(LoadSegmentImage {
            program_index,
            virtual_address: program.virtual_address,
            memory_size: program.memory_size,
            flags: program.flags,
            alignment: program.alignment,
            data,
        });
    }

    Ok(LoadedExecutable {
        info,
        load_segments,
    })
}

pub fn inspect_reader(reader: &mut (impl Read + Seek)) -> Result<ExecutableInfo, InspectError> {
    let file_size = reader.seek(SeekFrom::End(0))?;
    let magic = read_array_at::<4>(reader, 0, file_size)?;
    let kind = match magic {
        ELF_MAGIC => ExecutableKind::Elf64,
        PS5_CONTAINER_MAGIC => ExecutableKind::Ps5Container,
        PS5_RETAIL_CONTAINER_MAGIC => ExecutableKind::Ps5RetailContainer,
        other => ExecutableKind::Unknown(other),
    };

    match kind {
        ExecutableKind::Elf64 => {
            let mut elf = parse_elf(reader, 0, file_size)?;
            elf.dynamic = parse_dynamic(reader, file_size, None, &elf)?;
            Ok(ExecutableInfo {
                kind,
                file_size,
                container: None,
                elf: Some(elf),
            })
        }
        ExecutableKind::Ps5Container | ExecutableKind::Ps5RetailContainer => {
            let container = parse_container(reader, file_size)?;
            let elf_offset = container.header.embedded_elf_offset;
            let found = read_array_at::<4>(reader, elf_offset, file_size)?;
            if found != ELF_MAGIC {
                return Err(InspectError::MissingEmbeddedElf {
                    offset: elf_offset,
                    found,
                });
            }

            let mut elf = parse_elf(reader, elf_offset, file_size)?;
            elf.dynamic = parse_dynamic(reader, file_size, Some(&container), &elf)?;
            Ok(ExecutableInfo {
                kind,
                file_size,
                container: Some(container),
                elf: Some(elf),
            })
        }
        ExecutableKind::Unknown(_) => Ok(ExecutableInfo {
            kind,
            file_size,
            container: None,
            elf: None,
        }),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::path::Path;

    use super::{
        CONTAINER_HEADER_SIZE, CONTAINER_SEGMENT_SIZE, DT_NEEDED, DT_NULL, DT_RELA, DT_RELA_SIZE,
        DT_SCE_SYMBOL_TABLE_SIZE, DT_STRING_TABLE, DT_STRING_TABLE_SIZE, DT_SYMBOL_TABLE,
        ELF64_HEADER_SIZE, ELF64_PROGRAM_HEADER_SIZE, ExecutableKind, InspectError, inspect_path,
        inspect_reader, load_reader,
    };

    #[test]
    fn opening_a_directory_reports_path_is_directory() {
        let error = inspect_path(Path::new(".")).unwrap_err();
        assert!(
            matches!(error, InspectError::PathIsDirectory { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn identifies_an_observed_ps5_container_and_embedded_elf() {
        let elf_offset = CONTAINER_HEADER_SIZE + CONTAINER_SEGMENT_SIZE;
        let segment_offset = elf_offset + ELF64_HEADER_SIZE;
        let mut bytes = vec![0_u8; segment_offset + 16];
        bytes[..4].copy_from_slice(&[0x4f, 0x15, 0x3d, 0x1d]);
        bytes[0x18..0x1a].copy_from_slice(&1_u16.to_le_bytes());
        bytes[0x20..0x28].copy_from_slice(&0x0010_0004_u64.to_le_bytes());
        bytes[0x28..0x30].copy_from_slice(&(segment_offset as u64).to_le_bytes());
        bytes[0x30..0x38].copy_from_slice(&16_u64.to_le_bytes());
        bytes[0x38..0x40].copy_from_slice(&16_u64.to_le_bytes());
        write_elf_header(&mut bytes[elf_offset..], 0);

        let info = inspect_reader(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(info.kind, ExecutableKind::Ps5Container);
        let container = info.container.unwrap();
        assert_eq!(container.header.embedded_elf_offset, elf_offset as u64);
        assert_eq!(container.segments[0].properties.program_index(), 1);
        assert!(container.segments[0].properties.is_signed());
        assert!(!container.segments[0].properties.is_encrypted());
        assert_eq!(info.elf.unwrap().file_offset, elf_offset as u64);
    }

    #[test]
    fn parses_a_little_endian_elf64_header_and_programs() {
        let mut bytes = vec![0_u8; ELF64_HEADER_SIZE + 2 * ELF64_PROGRAM_HEADER_SIZE];
        write_elf_header(&mut bytes, 2);
        let first = ELF64_HEADER_SIZE;
        bytes[first..first + 4].copy_from_slice(&1_u32.to_le_bytes());
        bytes[first + 4..first + 8].copy_from_slice(&5_u32.to_le_bytes());
        bytes[first + 8..first + 16].copy_from_slice(&0x4000_u64.to_le_bytes());
        bytes[first + 16..first + 24].copy_from_slice(&0x1000_u64.to_le_bytes());
        bytes[first + 32..first + 40].copy_from_slice(&0x2000_u64.to_le_bytes());
        bytes[first + 40..first + 48].copy_from_slice(&0x3000_u64.to_le_bytes());
        bytes[first + 48..first + 56].copy_from_slice(&0x4000_u64.to_le_bytes());

        let info = inspect_reader(&mut Cursor::new(bytes)).unwrap();
        let elf = info.elf.unwrap();
        assert_eq!(elf.header.machine, 62);
        assert_eq!(elf.header.entry.get(), 0x1234_5678);
        assert_eq!(elf.header.program_header_count, 2);
        assert_eq!(elf.program_headers.len(), 2);
        assert_eq!(elf.program_headers[0].program_type, 1);
        assert_eq!(elf.program_headers[0].flags, 5);
        assert_eq!(elf.program_headers[0].file_offset, 0x4000);
        assert_eq!(elf.program_headers[0].virtual_address.get(), 0x1000);
        assert_eq!(elf.program_headers[0].memory_size, 0x3000);
    }

    #[test]
    fn rejects_a_truncated_elf_header() {
        let error = inspect_reader(&mut Cursor::new([0x7f, b'E', b'L', b'F']))
            .expect_err("truncated ELF must fail");
        assert!(matches!(
            error,
            InspectError::Truncated { expected: 64, .. }
        ));
    }

    #[test]
    fn preserves_unknown_magic_without_overreading() {
        let info = inspect_reader(&mut Cursor::new([1_u8, 2, 3, 4])).unwrap();
        assert_eq!(info.kind, ExecutableKind::Unknown([1, 2, 3, 4]));
        assert!(info.elf.is_none());
    }

    #[test]
    fn reads_needed_libraries_from_a_dynamic_table() {
        let mut bytes = vec![0_u8; 0x300];
        write_elf_header(&mut bytes, 2);
        write_program_header(&mut bytes, 0, 1, 4, 0x100, 0x1000, 0x200, 0x200);
        write_program_header(&mut bytes, 1, 2, 4, 0x100, 0x1000, 0x80, 0x80);
        write_dynamic_entry(&mut bytes[0x100..], 0, DT_NEEDED, 1);
        write_dynamic_entry(&mut bytes[0x100..], 1, DT_STRING_TABLE, 0x1080);
        write_dynamic_entry(&mut bytes[0x100..], 2, DT_STRING_TABLE_SIZE, 0x40);
        write_dynamic_entry(&mut bytes[0x100..], 3, DT_SYMBOL_TABLE, 0x10c0);
        write_dynamic_entry(&mut bytes[0x100..], 4, DT_SCE_SYMBOL_TABLE_SIZE, 0x30);
        write_dynamic_entry(&mut bytes[0x100..], 5, DT_RELA, 0x10f0);
        write_dynamic_entry(&mut bytes[0x100..], 6, DT_RELA_SIZE, 0x18);
        write_dynamic_entry(&mut bytes[0x100..], 7, DT_NULL, 0);
        bytes[0x180..0x194].copy_from_slice(b"\0libFoo.prx\0foo#nid\0");
        bytes[0x1d8..0x1dc].copy_from_slice(&12_u32.to_le_bytes());
        bytes[0x1dc] = 0x12;
        bytes[0x1f0..0x1f8].copy_from_slice(&0x2000_u64.to_le_bytes());
        bytes[0x1f8..0x200].copy_from_slice(&((1_u64 << 32) | 7).to_le_bytes());

        let info = inspect_reader(&mut Cursor::new(bytes)).unwrap();
        let dynamic = info.elf.unwrap().dynamic.unwrap();
        assert_eq!(dynamic.string_table_address, 0x1080);
        assert_eq!(dynamic.needed_libraries, ["libFoo.prx"]);
        assert_eq!(dynamic.symbols.len(), 2);
        assert_eq!(dynamic.symbols[1].name, "foo#nid");
        assert_eq!(dynamic.relocations.len(), 1);
        assert_eq!(dynamic.imports.len(), 1);
        assert_eq!(dynamic.imports[0].name, "foo#nid");
        assert_eq!(dynamic.imports[0].relocation_types, [7]);
    }

    #[test]
    fn materializes_load_segments_and_preserves_zero_fill_size() {
        let mut bytes = vec![0_u8; 0x180];
        write_elf_header(&mut bytes, 1);
        write_program_header(&mut bytes, 0, 1, 5, 0x100, 0x4000, 4, 16);
        bytes[0x100..0x104].copy_from_slice(&[1, 2, 3, 4]);

        let loaded = load_reader(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(loaded.load_segments.len(), 1);
        let segment = &loaded.load_segments[0];
        assert_eq!(segment.virtual_address.get(), 0x4000);
        assert_eq!(segment.memory_size, 16);
        assert_eq!(segment.flags, 5);
        assert_eq!(segment.data, [1, 2, 3, 4]);
    }

    fn write_elf_header(bytes: &mut [u8], program_count: u16) {
        bytes[..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[6] = 1;
        bytes[7] = 9;
        bytes[8] = 2;
        bytes[0x10..0x12].copy_from_slice(&0xfe10_u16.to_le_bytes());
        bytes[0x12..0x14].copy_from_slice(&62_u16.to_le_bytes());
        bytes[0x18..0x20].copy_from_slice(&0x1234_5678_u64.to_le_bytes());
        bytes[0x20..0x28].copy_from_slice(&(ELF64_HEADER_SIZE as u64).to_le_bytes());
        bytes[0x34..0x36].copy_from_slice(&64_u16.to_le_bytes());
        bytes[0x36..0x38].copy_from_slice(&56_u16.to_le_bytes());
        bytes[0x38..0x3a].copy_from_slice(&program_count.to_le_bytes());
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
        let offset = ELF64_HEADER_SIZE + index * ELF64_PROGRAM_HEADER_SIZE;
        bytes[offset..offset + 4].copy_from_slice(&program_type.to_le_bytes());
        bytes[offset + 4..offset + 8].copy_from_slice(&flags.to_le_bytes());
        bytes[offset + 8..offset + 16].copy_from_slice(&file_offset.to_le_bytes());
        bytes[offset + 16..offset + 24].copy_from_slice(&virtual_address.to_le_bytes());
        bytes[offset + 32..offset + 40].copy_from_slice(&file_size.to_le_bytes());
        bytes[offset + 40..offset + 48].copy_from_slice(&memory_size.to_le_bytes());
        bytes[offset + 48..offset + 56].copy_from_slice(&0x1000_u64.to_le_bytes());
    }

    fn write_dynamic_entry(bytes: &mut [u8], index: usize, tag: i64, value: u64) {
        let offset = index * 16;
        bytes[offset..offset + 8].copy_from_slice(&tag.to_le_bytes());
        bytes[offset + 8..offset + 16].copy_from_slice(&value.to_le_bytes());
    }
}
