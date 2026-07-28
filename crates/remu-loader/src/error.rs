// SPDX-License-Identifier: GPL-2.0-or-later

//! Failures raised while inspecting or loading a PS5 executable.

use std::io;

use thiserror::Error;

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
