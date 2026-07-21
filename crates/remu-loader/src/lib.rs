// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use remu_core::GuestAddress;
use thiserror::Error;

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
const DT_SCE_JUMP_RELOCATION: i64 = 0x6100_0029;
const DT_SCE_PLTREL_SIZE: i64 = 0x6100_002d;
const DT_SCE_RELA: i64 = 0x6100_002f;
const DT_SCE_RELA_SIZE: i64 = 0x6100_0031;
const DT_SCE_STRING_TABLE: i64 = 0x6100_0035;
const DT_SCE_STRING_TABLE_SIZE: i64 = 0x6100_0037;
const DT_SCE_SYMBOL_TABLE: i64 = 0x6100_0039;
const DT_SCE_SYMBOL_TABLE_SIZE: i64 = 0x6100_003f;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutableKind {
    Elf64,
    Ps5Container,
    Ps5RetailContainer,
    Unknown([u8; 4]),
}

impl fmt::Display for ExecutableKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Elf64 => formatter.write_str("ELF64"),
            Self::Ps5Container => formatter.write_str("PS5 executable container"),
            Self::Ps5RetailContainer => formatter.write_str("PS5 retail executable container"),
            Self::Unknown(magic) => write!(formatter, "unknown ({magic:02x?})"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContainerSegmentProperties(u64);

impl ContainerSegmentProperties {
    const ENCRYPTED: u64 = 0x0002;
    const SIGNED: u64 = 0x0004;
    const HAS_BLOCKS: u64 = 0x0800;

    #[must_use]
    pub const fn new(bits: u64) -> Self {
        Self(bits)
    }

    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }

    #[must_use]
    pub const fn is_encrypted(self) -> bool {
        self.0 & Self::ENCRYPTED != 0
    }

    #[must_use]
    pub const fn is_signed(self) -> bool {
        self.0 & Self::SIGNED != 0
    }

    #[must_use]
    pub const fn has_blocks(self) -> bool {
        self.0 & Self::HAS_BLOCKS != 0
    }

    #[must_use]
    pub const fn program_index(self) -> usize {
        ((self.0 >> 20) & 0x0fff) as usize
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContainerHeader {
    pub version: u8,
    pub mode: u8,
    pub endian: u8,
    pub attributes: u8,
    pub key_type: u32,
    pub header_size: u16,
    pub metadata_size: u16,
    pub declared_file_size: u64,
    pub segment_count: u16,
    pub flags: u16,
    pub reserved: u32,
    pub embedded_elf_offset: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContainerSegment {
    pub properties: ContainerSegmentProperties,
    pub file_offset: u64,
    pub stored_size: u64,
    pub expanded_size: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContainerInfo {
    pub header: ContainerHeader,
    pub segments: Vec<ContainerSegment>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Elf64Header {
    pub os_abi: u8,
    pub abi_version: u8,
    pub object_type: u16,
    pub machine: u16,
    pub entry: GuestAddress,
    pub program_header_offset: u64,
    pub section_header_offset: u64,
    pub flags: u32,
    pub header_size: u16,
    pub program_header_entry_size: u16,
    pub program_header_count: u16,
    pub section_header_entry_size: u16,
    pub section_header_count: u16,
    pub section_name_index: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Elf64ProgramHeader {
    pub program_type: u32,
    pub flags: u32,
    pub file_offset: u64,
    pub virtual_address: GuestAddress,
    pub physical_address: GuestAddress,
    pub file_size: u64,
    pub memory_size: u64,
    pub alignment: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DynamicEntry {
    pub tag: i64,
    pub value: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Elf64Symbol {
    pub name_offset: u32,
    pub info: u8,
    pub other: u8,
    pub section_index: u16,
    pub value: GuestAddress,
    pub size: u64,
    pub name: String,
}

impl Elf64Symbol {
    #[must_use]
    pub const fn binding(&self) -> u8 {
        self.info >> 4
    }

    #[must_use]
    pub const fn symbol_type(&self) -> u8 {
        self.info & 0x0f
    }

    #[must_use]
    pub const fn is_undefined(&self) -> bool {
        self.section_index == 0
    }

    #[must_use]
    pub const fn is_weak(&self) -> bool {
        self.binding() == 2
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Elf64Rela {
    pub offset: GuestAddress,
    pub symbol_index: u32,
    pub relocation_type: u32,
    pub addend: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportedSymbol {
    pub symbol_index: u32,
    pub name: String,
    pub relocation_count: usize,
    pub relocation_types: Vec<u32>,
    pub weak: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DynamicInfo {
    pub entries: Vec<DynamicEntry>,
    pub needed_libraries: Vec<String>,
    pub string_table_address: u64,
    pub string_table_size: u64,
    pub symbol_table_address: u64,
    pub symbol_table_size: u64,
    pub rela_address: u64,
    pub rela_size: u64,
    pub jump_relocation_address: u64,
    pub jump_relocation_size: u64,
    pub plt_got_address: u64,
    pub init_address: u64,
    pub init_array_address: u64,
    pub init_array_size: u64,
    pub preinit_array_address: u64,
    pub preinit_array_size: u64,
    pub symbols: Vec<Elf64Symbol>,
    pub relocations: Vec<Elf64Rela>,
    pub jump_relocations: Vec<Elf64Rela>,
    pub imports: Vec<ImportedSymbol>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Elf64Info {
    pub file_offset: u64,
    pub header: Elf64Header,
    pub program_headers: Vec<Elf64ProgramHeader>,
    pub dynamic: Option<DynamicInfo>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutableInfo {
    pub kind: ExecutableKind,
    pub file_size: u64,
    pub container: Option<ContainerInfo>,
    pub elf: Option<Elf64Info>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoadSegmentImage {
    pub program_index: usize,
    pub virtual_address: GuestAddress,
    pub memory_size: u64,
    pub flags: u32,
    pub alignment: u64,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoadedExecutable {
    pub info: ExecutableInfo,
    pub load_segments: Vec<LoadSegmentImage>,
}

#[derive(Debug, Error)]
pub enum InspectError {
    #[error("failed to read executable: {0}")]
    Io(#[from] io::Error),
    #[error("executable is truncated: expected at least {expected} bytes, found {actual}")]
    Truncated { expected: u64, actual: u64 },
    #[error("unsupported ELF class {0}; REmu currently accepts ELF64 only")]
    UnsupportedElfClass(u8),
    #[error("unsupported ELF byte order {0}; PS5 executables are little-endian")]
    UnsupportedElfByteOrder(u8),
    #[error("container declares {0} segments, exceeding the safety limit")]
    TooManyContainerSegments(u16),
    #[error("ELF declares {0} program headers, exceeding the safety limit")]
    TooManyProgramHeaders(u16),
    #[error("ELF program-header entry size {0} is smaller than ELF64 requires")]
    UnsupportedProgramHeaderSize(u16),
    #[error("container segment {index} ends at 0x{end:x}, beyond file size 0x{file_size:x}")]
    ContainerSegmentOutOfBounds {
        index: usize,
        end: u64,
        file_size: u64,
    },
    #[error("expected an embedded ELF header at 0x{offset:x}, found {found:02x?}")]
    MissingEmbeddedElf { offset: u64, found: [u8; 4] },
    #[error("offset arithmetic overflow while parsing executable")]
    OffsetOverflow,
    #[error("{table} table size 0x{size:x} exceeds the parser safety limit")]
    MetadataTableTooLarge { table: &'static str, size: u64 },
    #[error("ELF file range 0x{offset:x}..0x{end:x} is not backed by a usable input segment")]
    UnmappedElfFileRange { offset: u64, end: u64 },
    #[error("ELF virtual range 0x{address:x}..0x{end:x} is not backed by a file segment")]
    UnmappedVirtualRange { address: u64, end: u64 },
    #[error(
        "container payload for program {program_index} is not directly usable (encrypted={encrypted}, compressed={compressed})"
    )]
    UnsupportedContainerPayload {
        program_index: usize,
        encrypted: bool,
        compressed: bool,
    },
    #[error("dynamic string offset 0x{offset:x} is outside the string table")]
    DynamicStringOutOfBounds { offset: u64 },
    #[error("dynamic string at offset 0x{offset:x} is not NUL terminated")]
    UnterminatedDynamicString { offset: u64 },
    #[error("dynamic string at offset 0x{offset:x} is not valid UTF-8")]
    InvalidDynamicString { offset: u64 },
    #[error("{table} table size 0x{size:x} is not a multiple of entry size {entry_size}")]
    InvalidMetadataTableSize {
        table: &'static str,
        size: u64,
        entry_size: usize,
    },
    #[error("relocation references symbol {index}, but the symbol table has {count} entries")]
    RelocationSymbolOutOfBounds { index: u32, count: usize },
    #[error(
        "load segment {program_index} file size 0x{file_size:x} exceeds memory size 0x{memory_size:x}"
    )]
    SegmentFileLargerThanMemory {
        program_index: usize,
        file_size: u64,
        memory_size: u64,
    },
}

pub fn inspect_path(path: impl AsRef<Path>) -> Result<ExecutableInfo, InspectError> {
    let mut file = File::open(path)?;
    inspect_reader(&mut file)
}

pub fn load_path(path: impl AsRef<Path>) -> Result<LoadedExecutable, InspectError> {
    let mut file = File::open(path)?;
    load_reader(&mut file)
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

fn parse_container(
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

fn parse_elf(
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

fn parse_dynamic(
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
            if entry.tag == DT_NEEDED {
                info.needed_libraries
                    .push(read_dynamic_string(&string_table, entry.value)?);
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
fn read_elf_file_range_bounded(
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

fn read_vec_at(
    reader: &mut (impl Read + Seek),
    offset: u64,
    size: u64,
    file_size: u64,
) -> Result<Vec<u8>, InspectError> {
    checked_end(offset, size, file_size)?;
    let size = usize::try_from(size).map_err(|_| InspectError::OffsetOverflow)?;
    reader.seek(SeekFrom::Start(offset))?;
    let mut bytes = vec![0_u8; size];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn read_array_at<const N: usize>(
    reader: &mut (impl Read + Seek),
    offset: u64,
    file_size: u64,
) -> Result<[u8; N], InspectError> {
    checked_end(offset, N as u64, file_size)?;
    reader.seek(SeekFrom::Start(offset))?;
    let mut bytes = [0_u8; N];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn checked_end(offset: u64, size: u64, file_size: u64) -> Result<u64, InspectError> {
    let end = offset
        .checked_add(size)
        .ok_or(InspectError::OffsetOverflow)?;
    if end > file_size {
        return Err(InspectError::Truncated {
            expected: end,
            actual: file_size,
        });
    }
    Ok(end)
}

fn read_u16<const N: usize>(bytes: &[u8; N], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32<const N: usize>(bytes: &[u8; N], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_u64<const N: usize>(bytes: &[u8; N], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

fn read_i64_slice(bytes: &[u8], offset: usize) -> i64 {
    i64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

fn read_u16_slice(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32_slice(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_u64_slice(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{
        CONTAINER_HEADER_SIZE, CONTAINER_SEGMENT_SIZE, DT_NEEDED, DT_NULL, DT_RELA, DT_RELA_SIZE,
        DT_SCE_SYMBOL_TABLE_SIZE, DT_STRING_TABLE, DT_STRING_TABLE_SIZE, DT_SYMBOL_TABLE,
        ELF64_HEADER_SIZE, ELF64_PROGRAM_HEADER_SIZE, ExecutableKind, InspectError, inspect_reader,
        load_reader,
    };

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
