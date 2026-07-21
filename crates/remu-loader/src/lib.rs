// SPDX-License-Identifier: GPL-2.0-or-later

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
const CONTAINER_HEADER_SIZE: usize = 32;
const CONTAINER_SEGMENT_SIZE: usize = 32;
const MAX_CONTAINER_SEGMENTS: usize = 4096;
const MAX_PROGRAM_HEADERS: usize = 4096;

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Elf64Info {
    pub file_offset: u64,
    pub header: Elf64Header,
    pub program_headers: Vec<Elf64ProgramHeader>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutableInfo {
    pub kind: ExecutableKind,
    pub file_size: u64,
    pub container: Option<ContainerInfo>,
    pub elf: Option<Elf64Info>,
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
}

pub fn inspect_path(path: impl AsRef<Path>) -> Result<ExecutableInfo, InspectError> {
    let mut file = File::open(path)?;
    inspect_reader(&mut file)
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
        ExecutableKind::Elf64 => Ok(ExecutableInfo {
            kind,
            file_size,
            container: None,
            elf: Some(parse_elf(reader, 0, file_size)?),
        }),
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

            Ok(ExecutableInfo {
                kind,
                file_size,
                container: Some(container),
                elf: Some(parse_elf(reader, elf_offset, file_size)?),
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
    })
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

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{
        CONTAINER_HEADER_SIZE, CONTAINER_SEGMENT_SIZE, ELF64_HEADER_SIZE,
        ELF64_PROGRAM_HEADER_SIZE, ExecutableKind, InspectError, inspect_reader,
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
}
