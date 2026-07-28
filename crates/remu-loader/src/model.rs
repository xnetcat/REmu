// SPDX-License-Identifier: GPL-2.0-or-later

//! Data model for PS5 executable containers and the ELF64 images they carry.

use std::fmt;

use remu_core::GuestAddress;

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

/// A module namespace encoded in Sony dynamic-linker metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SceModuleReference {
    pub name: String,
    pub id: u16,
    pub version_major: u8,
    pub version_minor: u8,
}

/// A library namespace encoded in Sony dynamic-linker metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SceLibraryReference {
    pub name: String,
    pub id: u16,
    pub version: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DynamicInfo {
    pub entries: Vec<DynamicEntry>,
    pub needed_libraries: Vec<String>,
    pub original_filename: Option<String>,
    pub required_modules: Vec<SceModuleReference>,
    pub exported_modules: Vec<SceModuleReference>,
    pub imported_libraries: Vec<SceLibraryReference>,
    pub exported_libraries: Vec<SceLibraryReference>,
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
