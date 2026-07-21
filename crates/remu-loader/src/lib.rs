// Licensed under the Apache License, Version 2.0.

use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use remu_core::GuestAddress;
use thiserror::Error;

const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const PS5_CONTAINER_MAGIC: [u8; 4] = [0x4f, 0x15, 0x3d, 0x1d];
const PS5_RETAIL_CONTAINER_MAGIC: [u8; 4] = [0x54, 0x14, 0xf5, 0xee];
const ELF64_HEADER_SIZE: usize = 64;

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
pub struct Elf64Header {
    pub os_abi: u8,
    pub abi_version: u8,
    pub object_type: u16,
    pub machine: u16,
    pub entry: GuestAddress,
    pub program_header_offset: u64,
    pub program_header_entry_size: u16,
    pub program_header_count: u16,
    pub section_header_offset: u64,
    pub section_header_entry_size: u16,
    pub section_header_count: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutableInfo {
    pub kind: ExecutableKind,
    pub elf: Option<Elf64Header>,
}

#[derive(Debug, Error)]
pub enum InspectError {
    #[error("failed to read executable: {0}")]
    Io(#[from] io::Error),
    #[error("ELF header is truncated: expected {expected} bytes, found {actual}")]
    Truncated { expected: usize, actual: usize },
    #[error("unsupported ELF class {0}; REmu currently accepts ELF64 only")]
    UnsupportedElfClass(u8),
    #[error("unsupported ELF byte order {0}; PS5 executables are little-endian")]
    UnsupportedElfByteOrder(u8),
}

pub fn inspect_path(path: impl AsRef<Path>) -> Result<ExecutableInfo, InspectError> {
    let mut file = File::open(path)?;
    inspect_reader(&mut file)
}

pub fn inspect_reader(reader: &mut impl Read) -> Result<ExecutableInfo, InspectError> {
    let mut header = [0_u8; ELF64_HEADER_SIZE];
    let mut read = 0;
    while read < header.len() {
        let count = reader.read(&mut header[read..])?;
        if count == 0 {
            break;
        }
        read += count;
    }

    if read < 4 {
        return Err(InspectError::Truncated {
            expected: 4,
            actual: read,
        });
    }

    let magic = [header[0], header[1], header[2], header[3]];
    let kind = match magic {
        ELF_MAGIC => ExecutableKind::Elf64,
        PS5_CONTAINER_MAGIC => ExecutableKind::Ps5Container,
        PS5_RETAIL_CONTAINER_MAGIC => ExecutableKind::Ps5RetailContainer,
        other => ExecutableKind::Unknown(other),
    };

    if kind != ExecutableKind::Elf64 {
        return Ok(ExecutableInfo { kind, elf: None });
    }

    if read < ELF64_HEADER_SIZE {
        return Err(InspectError::Truncated {
            expected: ELF64_HEADER_SIZE,
            actual: read,
        });
    }
    if header[4] != 2 {
        return Err(InspectError::UnsupportedElfClass(header[4]));
    }
    if header[5] != 1 {
        return Err(InspectError::UnsupportedElfByteOrder(header[5]));
    }

    let elf = Elf64Header {
        os_abi: header[7],
        abi_version: header[8],
        object_type: read_u16(&header, 16),
        machine: read_u16(&header, 18),
        entry: GuestAddress::new(read_u64(&header, 24)),
        program_header_offset: read_u64(&header, 32),
        section_header_offset: read_u64(&header, 40),
        program_header_entry_size: read_u16(&header, 54),
        program_header_count: read_u16(&header, 56),
        section_header_entry_size: read_u16(&header, 58),
        section_header_count: read_u16(&header, 60),
    };

    Ok(ExecutableInfo {
        kind,
        elf: Some(elf),
    })
}

fn read_u16(bytes: &[u8; ELF64_HEADER_SIZE], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u64(bytes: &[u8; ELF64_HEADER_SIZE], offset: usize) -> u64 {
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

    use super::{ExecutableKind, InspectError, inspect_reader};

    #[test]
    fn identifies_observed_ps5_container_signatures() {
        let mut standard = Cursor::new([0x4f, 0x15, 0x3d, 0x1d]);
        let mut retail = Cursor::new([0x54, 0x14, 0xf5, 0xee]);

        assert_eq!(
            inspect_reader(&mut standard).unwrap().kind,
            ExecutableKind::Ps5Container
        );
        assert_eq!(
            inspect_reader(&mut retail).unwrap().kind,
            ExecutableKind::Ps5RetailContainer
        );
    }

    #[test]
    fn parses_a_little_endian_elf64_header() {
        let mut bytes = [0_u8; 64];
        bytes[..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[6] = 1;
        bytes[7] = 9;
        bytes[8] = 2;
        bytes[16..18].copy_from_slice(&3_u16.to_le_bytes());
        bytes[18..20].copy_from_slice(&62_u16.to_le_bytes());
        bytes[24..32].copy_from_slice(&0x1234_5678_u64.to_le_bytes());
        bytes[32..40].copy_from_slice(&64_u64.to_le_bytes());
        bytes[54..56].copy_from_slice(&56_u16.to_le_bytes());
        bytes[56..58].copy_from_slice(&7_u16.to_le_bytes());

        let info = inspect_reader(&mut Cursor::new(bytes)).unwrap();
        let elf = info.elf.unwrap();
        assert_eq!(elf.machine, 62);
        assert_eq!(elf.entry.get(), 0x1234_5678);
        assert_eq!(elf.program_header_count, 7);
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
}
