// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#![allow(clippy::unreadable_literal)]

// The Gen5 register-default layout and values are derived from SharpEmu/Kyty's GPL-compatible
// AGC tables. Keeping the serialization isolated makes the guest-facing pointer graph testable.

use std::collections::BTreeMap;

const DESCRIPTOR_SIZE: usize = 0x40;
const REGISTER_BLOCK_SIZE: usize = 16 * 8;

mod v10;

/// Which AGC register file a default belongs to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RegisterFile {
    Context,
    Shader,
    UserConfig,
}

/// Build a complete shadow register map for one file from the version-10 AGC defaults.
///
/// Absent means default: callers seed on init/reset instead of leaving maps empty and inventing
/// per-site fallbacks. Public table values win over internal supplements when both define an
/// offset.
pub(crate) fn seed_register_defaults(file: RegisterFile) -> BTreeMap<u32, u32> {
    let mut map = BTreeMap::new();
    if file == RegisterFile::Context {
        for &(offset, value) in v10::INTERNAL_CONTEXT_REGISTER_DEFAULTS {
            map.insert(offset, value);
        }
        let (offset, value) = v10::SPI_SHADER_COL_FORMAT_DEFAULT;
        map.insert(offset, value);
    }
    for &(offset, value) in v10::public_register_defaults(file) {
        map.insert(offset, value);
    }
    map
}

/// Look up one version-10 default. Distinguishes known zero from an offset the tables do not
/// represent (so callers can refuse instead of inventing a substitute).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn lookup_register_default(file: RegisterFile, offset: u32) -> Option<u32> {
    if file == RegisterFile::Context && offset == v10::SPI_SHADER_COL_FORMAT_DEFAULT.0 {
        return Some(v10::SPI_SHADER_COL_FORMAT_DEFAULT.1);
    }
    let public = v10::public_register_defaults(file)
        .iter()
        .rev()
        .find(|(candidate, _)| *candidate == offset)
        .map(|(_, value)| *value);
    if public.is_some() {
        return public;
    }
    if file == RegisterFile::Context {
        return v10::INTERNAL_CONTEXT_REGISTER_DEFAULTS
            .iter()
            .rev()
            .find(|(candidate, _)| *candidate == offset)
            .map(|(_, value)| *value);
    }
    None
}

#[derive(Clone, Copy)]
struct CompactRegisterDefaultTable {
    registers: &'static [(u32, u32)],
    pointer_offsets: &'static [u16],
}

impl CompactRegisterDefaultTable {
    const fn new(registers: &'static [(u32, u32)], pointer_offsets: &'static [u16]) -> Self {
        Self {
            registers,
            pointer_offsets,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CompactRegisterDefaults {
    tables: [CompactRegisterDefaultTable; 4],
    /// Exact guest ABI triples: kind, metadata, and reserved.
    types: &'static [u32],
}

impl CompactRegisterDefaults {
    const fn new(tables: [CompactRegisterDefaultTable; 4], types: &'static [u32]) -> Self {
        Self { tables, types }
    }

    fn type_count(self) -> usize {
        self.types.len() / 3
    }

    fn offsets(self) -> CompactRegisterDefaultsOffsets {
        let mut cursor = DESCRIPTOR_SIZE;
        let mut pointer_tables = [0_usize; 4];
        let mut table_index = 0;
        while table_index < self.tables.len() {
            cursor = align_up(cursor, 8);
            pointer_tables[table_index] = cursor;
            cursor += self.tables[table_index].pointer_offsets.len() * 8;
            table_index += 1;
        }

        let types = align_up(cursor, 4);
        cursor = types + self.types.len() * 4;

        let mut register_blocks = [0_usize; 4];
        table_index = 0;
        while table_index < self.tables.len() {
            cursor = align_up(cursor, 8);
            register_blocks[table_index] = cursor;
            cursor += self.tables[table_index].registers.len() * 8;
            table_index += 1;
        }

        CompactRegisterDefaultsOffsets {
            pointer_tables,
            types,
            register_blocks,
            blob_length: cursor,
        }
    }

    fn build(self, address: u64) -> Option<Vec<u8>> {
        if !self.types.len().is_multiple_of(3) {
            return None;
        }
        let offsets = self.offsets();
        let mut blob = vec![0_u8; offsets.blob_length];

        for (table_index, table) in self.tables.iter().enumerate() {
            let pointer_count = table.pointer_offsets.len();
            if pointer_count != 0 {
                write_u64(
                    &mut blob,
                    table_index.checked_mul(8)?,
                    address.checked_add(offsets.pointer_tables[table_index] as u64)?,
                )?;
            }
            write_u32(
                &mut blob,
                0x20_usize.checked_add(table_index.checked_mul(4)?)?,
                u32::try_from(table.registers.len()).ok()?,
            )?;

            for (pointer_index, &register_offset) in table.pointer_offsets.iter().enumerate() {
                let register_offset = usize::from(register_offset);
                if register_offset >= table.registers.len() {
                    return None;
                }
                write_u64(
                    &mut blob,
                    offsets.pointer_tables[table_index]
                        .checked_add(pointer_index.checked_mul(8)?)?,
                    address.checked_add(
                        offsets.register_blocks[table_index]
                            .checked_add(register_offset.checked_mul(8)?)?
                            as u64,
                    )?,
                )?;
            }

            for (register_index, &(register, value)) in table.registers.iter().enumerate() {
                let destination = offsets.register_blocks[table_index]
                    .checked_add(register_index.checked_mul(8)?)?;
                write_u32(&mut blob, destination, register)?;
                write_u32(&mut blob, destination.checked_add(4)?, value)?;
            }
        }

        if !self.types.is_empty() {
            write_u64(&mut blob, 0x30, address.checked_add(offsets.types as u64)?)?;
        }
        write_u32(&mut blob, 0x38, u32::try_from(self.type_count()).ok()?)?;
        for (type_index, &value) in self.types.iter().enumerate() {
            write_u32(
                &mut blob,
                offsets.types.checked_add(type_index.checked_mul(4)?)?,
                value,
            )?;
        }

        Some(blob)
    }
}

#[derive(Clone, Copy)]
struct CompactRegisterDefaultsOffsets {
    pointer_tables: [usize; 4],
    types: usize,
    register_blocks: [usize; 4],
    blob_length: usize,
}

#[derive(Clone, Copy)]
struct RegisterDefaultGroup {
    space: u32,
    index: u32,
    kind: u32,
    registers: &'static [(u32, u32)],
}

impl RegisterDefaultGroup {
    const fn new(space: u32, index: u32, kind: u32, registers: &'static [(u32, u32)]) -> Self {
        Self {
            space,
            index,
            kind,
            registers,
        }
    }

    fn type_metadata(self) -> Option<u32> {
        let register_count = u32::try_from(self.registers.len()).ok()?;
        let class: u32 = match self.kind {
            0x569f7444 | 0x7690af6f | 0xbda98653 => 0x0040_0000,
            0xef550356 | 0x38e92c91 | 0xaaa964b9 | 0x4cc673a0 => 0x0020_0000,
            0x1f2d8149 | 0xb2609506 | 0xb26219ca | 0xb25b6cf9 | 0xb2f86101 | 0x195d37d2
            | 0xf9ec4f85 => 0x0010_0000,
            0x48531062 | 0xbdbd1d0f | 0xbd946fd4 | 0xbdf02a4c => 0x0080_0000,
            0x4626b750 | 0xac0f9e76 => 0x0008_0000,
            0x036ac8a6 => 0x000c_0000,
            _ => 0x0004_0000,
        };
        class
            .checked_add(register_count.checked_shl(10)?)?
            .checked_add(self.index.checked_mul(4)?.checked_add(self.space)?)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct LegacyRegisterDefaultsLayout {
    groups: &'static [RegisterDefaultGroup],
    cx_table_length: usize,
    sh_table_length: usize,
    uc_table_length: usize,
}

impl LegacyRegisterDefaultsLayout {
    const fn primary() -> Self {
        Self {
            groups: PRIMARY_REGISTER_DEFAULTS,
            cx_table_length: 78,
            sh_table_length: 29,
            uc_table_length: 20,
        }
    }

    const fn internal() -> Self {
        Self {
            groups: INTERNAL_REGISTER_DEFAULTS,
            cx_table_length: 4,
            sh_table_length: 15,
            uc_table_length: 3,
        }
    }

    const fn group_count(&self) -> usize {
        self.groups.len()
    }

    pub(crate) fn blob_length(&self) -> usize {
        self.offsets().4
    }

    pub(crate) fn build(&self, address: u64) -> Option<Vec<u8>> {
        let (cx_offset, sh_offset, uc_offset, types_offset, blob_length) = self.offsets();
        let register_blocks_offset =
            blob_length.checked_sub(self.groups.len() * REGISTER_BLOCK_SIZE)?;
        let mut blob = vec![0_u8; blob_length];
        write_u64(&mut blob, 0x00, address.checked_add(cx_offset as u64)?)?;
        write_u64(&mut blob, 0x08, address.checked_add(sh_offset as u64)?)?;
        write_u64(&mut blob, 0x10, address.checked_add(uc_offset as u64)?)?;
        write_u64(&mut blob, 0x30, address.checked_add(types_offset as u64)?)?;
        write_u32(&mut blob, 0x38, u32::try_from(self.groups.len()).ok()?)?;

        for (group_index, group) in self.groups.iter().enumerate() {
            if group.registers.len() > 16 {
                return None;
            }
            let (table_offset, table_length) = match group.space {
                0 => (cx_offset, self.cx_table_length),
                1 => (sh_offset, self.sh_table_length),
                2 => (uc_offset, self.uc_table_length),
                _ => return None,
            };
            let index = usize::try_from(group.index).ok()?;
            if index >= table_length {
                return None;
            }
            let register_block_offset = register_blocks_offset
                .checked_add(group_index.checked_mul(REGISTER_BLOCK_SIZE)?)?;
            write_u64(
                &mut blob,
                table_offset.checked_add(index.checked_mul(8)?)?,
                address.checked_add(register_block_offset as u64)?,
            )?;
            let type_entry_offset = types_offset.checked_add(group_index.checked_mul(12)?)?;
            write_u32(&mut blob, type_entry_offset, group.kind)?;
            write_u32(
                &mut blob,
                type_entry_offset.checked_add(4)?,
                group.type_metadata()?,
            )?;
            for (register_index, &(offset, value)) in group.registers.iter().enumerate() {
                let destination =
                    register_block_offset.checked_add(register_index.checked_mul(8)?)?;
                write_u32(&mut blob, destination, offset)?;
                write_u32(&mut blob, destination.checked_add(4)?, value)?;
            }
        }
        Some(blob)
    }

    fn offsets(&self) -> (usize, usize, usize, usize, usize) {
        let cx_offset = align_up(DESCRIPTOR_SIZE, 8);
        let sh_offset = cx_offset + self.cx_table_length * 8;
        let uc_offset = sh_offset + self.sh_table_length * 8;
        let types_offset = align_up(uc_offset + self.uc_table_length * 8, 4);
        let register_blocks_offset = align_up(types_offset + self.groups.len() * 12, 8);
        let blob_length = register_blocks_offset + self.groups.len() * REGISTER_BLOCK_SIZE;
        (cx_offset, sh_offset, uc_offset, types_offset, blob_length)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum RegisterDefaultsLayout {
    Legacy(LegacyRegisterDefaultsLayout),
    Compact(CompactRegisterDefaults),
}

impl RegisterDefaultsLayout {
    pub(crate) const fn for_version(version: u32, internal: bool) -> Option<Self> {
        match (version, internal) {
            (7 | 8, false) => Some(Self::Legacy(LegacyRegisterDefaultsLayout::primary())),
            (7 | 8, true) => Some(Self::Legacy(LegacyRegisterDefaultsLayout::internal())),
            (10, false) => Some(Self::Compact(v10::PUBLIC)),
            (10, true) => Some(Self::Compact(v10::INTERNAL)),
            _ => None,
        }
    }

    pub(crate) fn group_count(self) -> usize {
        match self {
            Self::Legacy(layout) => layout.group_count(),
            Self::Compact(layout) => layout.type_count(),
        }
    }

    pub(crate) fn blob_length(self) -> usize {
        match self {
            Self::Legacy(layout) => layout.blob_length(),
            Self::Compact(layout) => layout.offsets().blob_length,
        }
    }

    pub(crate) fn build(self, address: u64) -> Option<Vec<u8>> {
        match self {
            Self::Legacy(layout) => layout.build(address),
            Self::Compact(layout) => layout.build(address),
        }
    }
}

const fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

fn write_u32(blob: &mut [u8], offset: usize, value: u32) -> Option<()> {
    blob.get_mut(offset..offset.checked_add(4)?)?
        .copy_from_slice(&value.to_le_bytes());
    Some(())
}

fn write_u64(blob: &mut [u8], offset: usize, value: u64) -> Option<()> {
    blob.get_mut(offset..offset.checked_add(8)?)?
        .copy_from_slice(&value.to_le_bytes());
    Some(())
}

const PRIMARY_REGISTER_DEFAULTS: &[RegisterDefaultGroup] = &[
    RegisterDefaultGroup::new(0, 0, 0xe24f806d, &[(0x202, 0x00cc0010)]),
    RegisterDefaultGroup::new(0, 1, 0xf6c28182, &[(0x109, 0x00000000)]),
    RegisterDefaultGroup::new(0, 2, 0x6f6e55a5, &[(0x104, 0x00000000)]),
    RegisterDefaultGroup::new(0, 3, 0x0bc65da4, &[(0x08f, 0x00000000)]),
    RegisterDefaultGroup::new(0, 4, 0x9e5ad592, &[(0x08e, 0x0000000f)]),
    RegisterDefaultGroup::new(0, 5, 0xbb513b98, &[(0x2dc, 0x0000aa00)]),
    RegisterDefaultGroup::new(0, 6, 0xab64b23b, &[(0x001, 0x00000000)]),
    RegisterDefaultGroup::new(0, 7, 0x53c39964, &[(0x200, 0x00000000)]),
    RegisterDefaultGroup::new(0, 8, 0x01396b11, &[(0x201, 0x00000000)]),
    RegisterDefaultGroup::new(0, 9, 0x7d42019a, &[(0x000, 0x00000000)]),
    RegisterDefaultGroup::new(0, 10, 0x3548f523, &[(0x006, 0x00000000)]),
    RegisterDefaultGroup::new(0, 11, 0xf43ad28a, &[(0x01f, 0x00000000)]),
    RegisterDefaultGroup::new(0, 12, 0x6de4c312, &[(0x203, 0x00000000)]),
    RegisterDefaultGroup::new(0, 13, 0x00a77ae0, &[(0x2b0, 0x00000000)]),
    RegisterDefaultGroup::new(0, 14, 0x00a779b7, &[(0x2b1, 0x00000000)]),
    RegisterDefaultGroup::new(0, 15, 0x5100100c, &[(0x10c, 0x00000000)]),
    RegisterDefaultGroup::new(0, 16, 0x59958bba, &[(0x10d, 0x00000000)]),
    RegisterDefaultGroup::new(0, 17, 0x0c06f17c, &[(0x10b, 0x00000000)]),
    RegisterDefaultGroup::new(0, 18, 0x6f104b72, &[(0x1ff, 0x00000000)]),
    RegisterDefaultGroup::new(0, 19, 0x25c70d9c, &[(0x204, 0x00000000)]),
    RegisterDefaultGroup::new(0, 20, 0x3881201e, &[(0x20d, 0x00000000)]),
    RegisterDefaultGroup::new(0, 21, 0x09afddaf, &[(0x206, 0x0000043f)]),
    RegisterDefaultGroup::new(0, 22, 0x367d63cf, &[(0x2f8, 0x00000000)]),
    RegisterDefaultGroup::new(0, 23, 0x43707db8, &[(0x083, 0x0000ffff)]),
    RegisterDefaultGroup::new(0, 24, 0xf6ae26ba, &[(0x313, 0x00000000)]),
    RegisterDefaultGroup::new(0, 25, 0x1b917652, &[(0x800003fe, 0x00000000)]),
    RegisterDefaultGroup::new(0, 26, 0x94b1e4f7, &[(0x0ea, 0x00000000)]),
    RegisterDefaultGroup::new(0, 27, 0xe3661b6c, &[(0x0e9, 0x00000000)]),
    RegisterDefaultGroup::new(0, 28, 0x1eb8d73a, &[(0x292, 0x00000002)]),
    RegisterDefaultGroup::new(0, 29, 0x15051fa3, &[(0x293, 0x00000000)]),
    RegisterDefaultGroup::new(0, 30, 0x9c51a7f1, &[(0x0e8, 0x00000000)]),
    RegisterDefaultGroup::new(0, 31, 0xa20efc70, &[(0x080, 0x00000000)]),
    RegisterDefaultGroup::new(0, 32, 0x0ec09f6e, &[(0x211, 0x00000000)]),
    RegisterDefaultGroup::new(0, 33, 0x34a7d6d3, &[(0x210, 0x00000000)]),
    RegisterDefaultGroup::new(0, 34, 0xce831b94, &[(0x08d, 0x00000000)]),
    RegisterDefaultGroup::new(0, 35, 0x5cc72a74, &[(0x282, 0x00000008)]),
    RegisterDefaultGroup::new(0, 36, 0x3b77713c, &[(0x281, 0xffff0000)]),
    RegisterDefaultGroup::new(0, 37, 0x40f64410, &[(0x280, 0x00080008)]),
    RegisterDefaultGroup::new(0, 38, 0x69441268, &[(0x2df, 0x00000000)]),
    RegisterDefaultGroup::new(0, 39, 0x2e418b83, &[(0x2de, 0x000001e9)]),
    RegisterDefaultGroup::new(0, 40, 0xa00d0c8d, &[(0x205, 0x00000240)]),
    RegisterDefaultGroup::new(0, 41, 0xb1289fb3, &[(0x20c, 0x00000001)]),
    RegisterDefaultGroup::new(0, 42, 0x144832fb, &[(0x2f9, 0x0000002d)]),
    RegisterDefaultGroup::new(0, 43, 0x9890d9fa, &[(0x1ba, 0x00000000)]),
    RegisterDefaultGroup::new(0, 44, 0x9016faf1, &[(0x2a6, 0x00000000)]),
    RegisterDefaultGroup::new(0, 45, 0x4b73ce27, &[(0x2ce, 0x00000400)]),
    RegisterDefaultGroup::new(0, 46, 0x5f5a3e7b, &[(0x29b, 0x00000002)]),
    RegisterDefaultGroup::new(0, 47, 0xd4af3a51, &[(0x2d6, 0x00000000)]),
    RegisterDefaultGroup::new(0, 48, 0x6cf4f543, &[(0x2a3, 0xffffffff)]),
    RegisterDefaultGroup::new(0, 49, 0x5fb86ccb, &[(0x2a1, 0x00000000)]),
    RegisterDefaultGroup::new(0, 50, 0xedefa188, &[(0x2ad, 0x00000000)]),
    RegisterDefaultGroup::new(0, 51, 0xd0de9ee6, &[(0x2d5, 0x00000000)]),
    RegisterDefaultGroup::new(0, 52, 0xc5831803, &[(0x2d4, 0x88101000)]),
    RegisterDefaultGroup::new(0, 53, 0x8e6de84b, &[(0x2db, 0x00000000)]),
    RegisterDefaultGroup::new(
        0,
        54,
        0xd0771662,
        &[(0x2f5, 0x00000000), (0x2f6, 0x00000000)],
    ),
    RegisterDefaultGroup::new(0, 55, 0x569f7444, &[(0x2fe, 0x00000000)]),
    RegisterDefaultGroup::new(
        0,
        56,
        0x5c6637cd,
        &[(0x30e, 0xffffffff), (0x30f, 0xffffffff)],
    ),
    RegisterDefaultGroup::new(
        0,
        57,
        0xcae3e690,
        &[(0x311, 0x00000002), (0x312, 0x03ff0080)],
    ),
    RegisterDefaultGroup::new(
        0,
        58,
        0x43fbd769,
        &[
            (0x105, 0x00000000),
            (0x107, 0x00000000),
            (0x106, 0x00000000),
            (0x108, 0x00000000),
        ],
    ),
    RegisterDefaultGroup::new(0, 59, 0xef550356, &[(0x1e0, 0x20010001)]),
    RegisterDefaultGroup::new(
        0,
        60,
        0x8f52e279,
        &[(0x020, 0x00000000), (0x021, 0x00000000)],
    ),
    RegisterDefaultGroup::new(
        0,
        61,
        0x1f2d8149,
        &[(0x084, 0x00000000), (0x085, 0x20002000)],
    ),
    RegisterDefaultGroup::new(0, 62, 0x853d0614, &[(0x800003ff, 0x00000000)]),
    RegisterDefaultGroup::new(
        0,
        63,
        0x4413c6f9,
        &[(0x008, 0x00000000), (0x009, 0x00000000)],
    ),
    RegisterDefaultGroup::new(
        0,
        64,
        0x67096014,
        &[
            (0x010, 0x80000000),
            (0x011, 0x20000000),
            (0x012, 0x00000000),
            (0x013, 0x00000000),
            (0x014, 0x00000000),
            (0x015, 0x00000000),
            (0x01a, 0x00000000),
            (0x01b, 0x00000000),
            (0x01c, 0x00000000),
            (0x01d, 0x00000000),
            (0x01e, 0x00000000),
            (0x002, 0x00000000),
            (0x005, 0x00000000),
            (0x007, 0x00000000),
            (0x00b, 0x00000000),
            (0x00a, 0x00000000),
        ],
    ),
    RegisterDefaultGroup::new(
        0,
        65,
        0x88f5e915,
        &[(0x0eb, 0xff00ff00), (0x0ec, 0x00000000)],
    ),
    RegisterDefaultGroup::new(
        0,
        66,
        0x033f1eff,
        &[(0x800003fc, 0x00000000), (0x800003fd, 0x00000000)],
    ),
    RegisterDefaultGroup::new(
        0,
        67,
        0x918106bb,
        &[(0x090, 0x80000000), (0x091, 0x40004000)],
    ),
    RegisterDefaultGroup::new(
        0,
        68,
        0x95f0e7ac,
        &[
            (0x2fa, 0x4e7e0000),
            (0x2fb, 0x4e7e0000),
            (0x2fc, 0x4e7e0000),
            (0x2fd, 0x4e7e0000),
        ],
    ),
    RegisterDefaultGroup::new(
        0,
        69,
        0xb48cbab2,
        &[(0x2e2, 0x00000000), (0x2e3, 0x00000000)],
    ),
    RegisterDefaultGroup::new(
        0,
        70,
        0x05bb3bc6,
        &[(0x2e0, 0x00000000), (0x2e1, 0x00000000)],
    ),
    RegisterDefaultGroup::new(
        0,
        71,
        0x94faba07,
        &[(0x003, 0x00000000), (0x004, 0x00000000)],
    ),
    RegisterDefaultGroup::new(
        0,
        72,
        0x38e92c91,
        &[
            (0x318, 0x00000000),
            (0x31b, 0x00000000),
            (0x31c, 0x00000000),
            (0x31d, 0x00000000),
            (0x31e, 0x00000048),
            (0x31f, 0x00000000),
            (0x321, 0x00000000),
            (0x323, 0x00000000),
            (0x324, 0x00000000),
            (0x325, 0x00000000),
            (0x390, 0x00000000),
            (0x398, 0x00000000),
            (0x3a0, 0x00000000),
            (0x3a8, 0x00000000),
            (0x3b0, 0x00000000),
            (0x3b8, 0x0006c000),
        ],
    ),
    RegisterDefaultGroup::new(
        0,
        73,
        0x0b177b43,
        &[(0x00c, 0x00000000), (0x00d, 0x40004000)],
    ),
    RegisterDefaultGroup::new(0, 74, 0x48531062, &[(0x191, 0x00000000)]),
    RegisterDefaultGroup::new(
        0,
        75,
        0xaaa964b9,
        &[
            (0x16f, 0x00000000),
            (0x170, 0x00000000),
            (0x171, 0x00000000),
            (0x172, 0x00000000),
        ],
    ),
    RegisterDefaultGroup::new(
        0,
        76,
        0x7690af6f,
        &[
            (0x10f, 0x4e7e0000),
            (0x111, 0x4e7e0000),
            (0x113, 0x4e7e0000),
            (0x110, 0x00000000),
            (0x112, 0x00000000),
            (0x114, 0x00000000),
            (0x094, 0x80000000),
            (0x095, 0x40004000),
            (0x0b4, 0x00000000),
            (0x0b5, 0x00000000),
        ],
    ),
    RegisterDefaultGroup::new(
        0,
        77,
        0x078d7060,
        &[(0x081, 0x80000000), (0x082, 0x40004000)],
    ),
    RegisterDefaultGroup::new(1, 0, 0x5d6e3ec7, &[(0x212, 0x00000000)]),
    RegisterDefaultGroup::new(1, 1, 0x57e7079a, &[(0x213, 0x00000000)]),
    RegisterDefaultGroup::new(1, 2, 0x7467fafd, &[(0x228, 0x00000000)]),
    RegisterDefaultGroup::new(1, 3, 0x9e826b50, &[(0x215, 0x00000000)]),
    RegisterDefaultGroup::new(1, 4, 0xdc484f18, &[(0x218, 0x00000000)]),
    RegisterDefaultGroup::new(1, 5, 0x5da8bca3, &[(0x08a, 0x00000000)]),
    RegisterDefaultGroup::new(1, 6, 0x5ca726d8, &[(0x10a, 0x00000000)]),
    RegisterDefaultGroup::new(1, 7, 0x5dd28360, &[(0x00a, 0x00000000)]),
    RegisterDefaultGroup::new(1, 8, 0x57efa0be, &[(0x08b, 0x00000000)]),
    RegisterDefaultGroup::new(1, 9, 0x502363d5, &[(0x10b, 0x00000000)]),
    RegisterDefaultGroup::new(1, 10, 0x506d14bd, &[(0x00b, 0x00000000)]),
    RegisterDefaultGroup::new(1, 11, 0xb2609506, &[(0x224, 0x00000000)]),
    RegisterDefaultGroup::new(
        1,
        12,
        0x9e5cfb8a,
        &[
            (0x107, 0x00000000),
            (0x087, 0x00000000),
            (0x007, 0x00000000),
        ],
    ),
    RegisterDefaultGroup::new(
        1,
        13,
        0xc918df3e,
        &[(0x20c, 0x00000000), (0x20d, 0x00000000)],
    ),
    RegisterDefaultGroup::new(
        1,
        14,
        0xc9751c9c,
        &[(0x0c8, 0x00000000), (0x0c9, 0x00000000)],
    ),
    RegisterDefaultGroup::new(
        1,
        15,
        0xc97ef77a,
        &[(0x088, 0x00000000), (0x089, 0x00000000)],
    ),
    RegisterDefaultGroup::new(
        1,
        16,
        0xc927c6b9,
        &[(0x108, 0x00000000), (0x109, 0x00000000)],
    ),
    RegisterDefaultGroup::new(
        1,
        17,
        0xc92a1ec5,
        &[(0x148, 0x00000000), (0x149, 0x00000000)],
    ),
    RegisterDefaultGroup::new(
        1,
        18,
        0xc9e01b31,
        &[(0x008, 0x00000000), (0x009, 0x00000000)],
    ),
    RegisterDefaultGroup::new(1, 19, 0x50685f29, &[(0x800002ff, 0x00000000)]),
    RegisterDefaultGroup::new(1, 20, 0xb26219ca, &[(0x0b2, 0x00000000)]),
    RegisterDefaultGroup::new(1, 21, 0xb25b6cf9, &[(0x132, 0x00000000)]),
    RegisterDefaultGroup::new(1, 22, 0xb2f86101, &[(0x032, 0x00000000)]),
    RegisterDefaultGroup::new(
        1,
        23,
        0x07e3b155,
        &[(0x082, 0x00000000), (0x083, 0x00000000)],
    ),
    RegisterDefaultGroup::new(
        1,
        24,
        0x07e383c6,
        &[(0x102, 0x00000000), (0x103, 0x00000000)],
    ),
    RegisterDefaultGroup::new(1, 25, 0xbda98653, &[(0x240, 0x00000000)]),
    RegisterDefaultGroup::new(1, 26, 0xbdbd1d0f, &[(0x08c, 0x00000000)]),
    RegisterDefaultGroup::new(1, 27, 0xbd946fd4, &[(0x10c, 0x00000000)]),
    RegisterDefaultGroup::new(1, 28, 0xbdf02a4c, &[(0x00c, 0x00000000)]),
    RegisterDefaultGroup::new(2, 0, 0x19e93e85, &[(0x41f, 0x00000000)]),
    RegisterDefaultGroup::new(2, 1, 0x3b5c2af3, &[(0x41d, 0x00000000)]),
    RegisterDefaultGroup::new(2, 2, 0x47974a35, &[(0x41e, 0x00000000)]),
    RegisterDefaultGroup::new(2, 3, 0x105971c2, &[(0x25b, 0x00000000)]),
    RegisterDefaultGroup::new(2, 4, 0x7d137765, &[(0x24a, 0x00000000)]),
    RegisterDefaultGroup::new(2, 5, 0xd187febc, &[(0x24b, 0x00000000)]),
    RegisterDefaultGroup::new(2, 6, 0x12f854ac, &[(0x25f, 0x00000000)]),
    RegisterDefaultGroup::new(2, 7, 0x40d49ad1, &[(0x262, 0x00000000)]),
    RegisterDefaultGroup::new(2, 8, 0x8c0923da, &[(0x80003ff4, 0x00000000)]),
    RegisterDefaultGroup::new(2, 9, 0xbb8df494, &[(0x80003ffd, 0x00000000)]),
    RegisterDefaultGroup::new(2, 10, 0xf6d8a76e, &[(0x382, 0x40000040)]),
    RegisterDefaultGroup::new(2, 11, 0x7620f1e9, &[(0x248, 0x00000000)]),
    RegisterDefaultGroup::new(2, 12, 0x9ebfab10, &[(0x242, 0x00000000)]),
    RegisterDefaultGroup::new(
        2,
        13,
        0x98a09d0e,
        &[(0x380, 0x00000000), (0x381, 0x00000000)],
    ),
    RegisterDefaultGroup::new(
        2,
        14,
        0x195d37d2,
        &[(0x80003ff5, 0x00000000), (0x80003ff6, 0x00000000)],
    ),
    RegisterDefaultGroup::new(
        2,
        15,
        0xf9ec4f85,
        &[
            (0x80003ff7, 0x00000000),
            (0x80003ff8, 0x00000000),
            (0x80003ff9, 0x00000000),
            (0x80003ffa, 0x00000000),
        ],
    ),
    RegisterDefaultGroup::new(
        2,
        16,
        0x4626b750,
        &[(0x80003ffb, 0x00000000), (0x80003ffc, 0x00000000)],
    ),
    RegisterDefaultGroup::new(2, 17, 0x4cc673a0, &[(0x80003ffe, 0x00000000)]),
    RegisterDefaultGroup::new(2, 18, 0xde5b3431, &[(0x80003fff, 0x00000000)]),
    RegisterDefaultGroup::new(2, 19, 0x036ac8a6, &[(0x25c, 0x00000000)]),
];

const INTERNAL_REGISTER_DEFAULTS: &[RegisterDefaultGroup] = &[
    RegisterDefaultGroup::new(0, 0, 0x8fb4edb5, &[(0x00e, 0)]),
    RegisterDefaultGroup::new(0, 1, 0xb994ad29, &[(0x2af, 0)]),
    RegisterDefaultGroup::new(0, 2, 0xd427322f, &[(0x314, 0)]),
    RegisterDefaultGroup::new(0, 3, 0xf58fea31, &[(0x1b5, 0)]),
    RegisterDefaultGroup::new(1, 0, 0x6ac156ef, &[(0x216, 0)]),
    RegisterDefaultGroup::new(1, 1, 0x6ac15610, &[(0x217, 0)]),
    RegisterDefaultGroup::new(1, 2, 0x6ac15009, &[(0x219, 0)]),
    RegisterDefaultGroup::new(1, 3, 0x6ac153ba, &[(0x21a, 0)]),
    RegisterDefaultGroup::new(1, 4, 0xbe7dcd73, &[(0x27d, 0)]),
    RegisterDefaultGroup::new(1, 5, 0x0c4b1438, &[(0x22a, 0)]),
    RegisterDefaultGroup::new(1, 6, 0xdb00d71a, &[(0x204, 0)]),
    RegisterDefaultGroup::new(1, 7, 0xdb00d249, &[(0x205, 0)]),
    RegisterDefaultGroup::new(1, 8, 0xdb00ec60, &[(0x206, 0)]),
    RegisterDefaultGroup::new(1, 9, 0x0c4d6fe4, &[(0x080, 0)]),
    RegisterDefaultGroup::new(1, 10, 0x0c4a80ef, &[(0x100, 0)]),
    RegisterDefaultGroup::new(1, 11, 0x0dd283e7, &[(0x006, 0)]),
    RegisterDefaultGroup::new(1, 12, 0xc620e68c, &[(0x081, 0)]),
    RegisterDefaultGroup::new(1, 13, 0xc67efacf, &[(0x101, 0)]),
    RegisterDefaultGroup::new(1, 14, 0xd9e6d9f7, &[(0x001, 0)]),
    RegisterDefaultGroup::new(2, 0, 0x31f34b9f, &[(0x24f, 0)]),
    RegisterDefaultGroup::new(2, 1, 0xac0f9e76, &[(0x80003fff, 0)]),
    RegisterDefaultGroup::new(2, 2, 0x929fd95d, &[(0x250, 0)]),
];

#[cfg(test)]
mod tests {
    use super::{
        INTERNAL_REGISTER_DEFAULTS, LegacyRegisterDefaultsLayout, PRIMARY_REGISTER_DEFAULTS,
        RegisterDefaultsLayout, v10,
    };

    const BASE_ADDRESS: u64 = 0x7000_0000;

    fn read_u32(blob: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(blob[offset..offset + 4].try_into().expect("u32"))
    }

    fn read_u64(blob: &[u8], offset: usize) -> u64 {
        u64::from_le_bytes(blob[offset..offset + 8].try_into().expect("u64"))
    }

    fn blob_offset(address: u64) -> usize {
        usize::try_from(address - BASE_ADDRESS).expect("pointer inside blob")
    }

    fn fnv1a64(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    #[test]
    fn seeds_agc_v10_shadow_defaults_for_register_files() {
        use super::{RegisterFile, lookup_register_default, seed_register_defaults};

        let cx = seed_register_defaults(RegisterFile::Context);
        assert_eq!(cx.get(&0x205), Some(&0x240));
        assert_eq!(cx.get(&0x191), Some(&0));
        assert_eq!(cx.get(&0x1b0), Some(&0));
        assert_eq!(cx.get(&0x1c5), Some(&0));
        assert_eq!(cx.get(&0x10f), Some(&0x3f80_0000));
        assert_eq!(cx.get(&0x0b4), Some(&0));
        assert_eq!(cx.get(&0x0b5), Some(&0));
        assert_eq!(cx.get(&0x2d5), Some(&0x2000));
        assert_eq!(cx.get(&0x202), Some(&0x00cc_0010));
        assert_eq!(lookup_register_default(RegisterFile::Context, 0x205), Some(0x240));
        assert_eq!(lookup_register_default(RegisterFile::Context, 0x1c5), Some(0));

        let uc = seed_register_defaults(RegisterFile::UserConfig);
        assert_eq!(uc.get(&0x24a), Some(&0));
        assert_eq!(uc.get(&0x25b), Some(&0));
    }

    #[test]
    fn encodes_group_class_count_and_table_selector() {
        assert_eq!(
            PRIMARY_REGISTER_DEFAULTS[0].type_metadata(),
            Some(0x0004_0400)
        );
        assert_eq!(
            PRIMARY_REGISTER_DEFAULTS[58].type_metadata(),
            Some(0x0004_10e8)
        );
        assert_eq!(
            PRIMARY_REGISTER_DEFAULTS[59].type_metadata(),
            Some(0x0020_04ec)
        );
        assert_eq!(
            PRIMARY_REGISTER_DEFAULTS[72].type_metadata(),
            Some(0x0020_4120)
        );
    }

    #[test]
    fn builds_primary_and_internal_pointer_graphs() {
        for layout in [
            LegacyRegisterDefaultsLayout::primary(),
            LegacyRegisterDefaultsLayout::internal(),
        ] {
            let blob = layout.build(BASE_ADDRESS).expect("valid register defaults");
            assert_eq!(blob.len(), layout.blob_length());
            assert_eq!(
                u32::from_le_bytes(blob[0x38..0x3c].try_into().expect("group count")) as usize,
                layout.group_count()
            );
        }
    }

    #[test]
    fn selects_only_truthful_register_default_versions() {
        assert!(matches!(
            RegisterDefaultsLayout::for_version(7, false),
            Some(RegisterDefaultsLayout::Legacy(_))
        ));
        assert!(matches!(
            RegisterDefaultsLayout::for_version(8, true),
            Some(RegisterDefaultsLayout::Legacy(_))
        ));
        assert!(matches!(
            RegisterDefaultsLayout::for_version(10, false),
            Some(RegisterDefaultsLayout::Compact(_))
        ));
        assert!(RegisterDefaultsLayout::for_version(9, false).is_none());
        assert!(RegisterDefaultsLayout::for_version(13, false).is_none());
    }

    #[test]
    fn serializes_exact_version_ten_public_descriptor_and_blend_group() {
        let layout = v10::PUBLIC;
        assert_eq!(
            layout.tables.map(|table| table.registers.len()),
            [482, 159, 55, 0]
        );
        assert_eq!(
            layout.tables.map(|table| table.pointer_offsets.len()),
            [79, 29, 20, 0]
        );
        assert_eq!(layout.type_count(), 128);

        let offsets = layout.offsets();
        let blob = layout.build(BASE_ADDRESS).expect("valid v10 public layout");
        assert_eq!(blob.len(), offsets.blob_length);
        assert_eq!(
            read_u64(&blob, 0x00),
            BASE_ADDRESS + offsets.pointer_tables[0] as u64
        );
        assert_eq!(
            read_u64(&blob, 0x08),
            BASE_ADDRESS + offsets.pointer_tables[1] as u64
        );
        assert_eq!(
            read_u64(&blob, 0x10),
            BASE_ADDRESS + offsets.pointer_tables[2] as u64
        );
        assert_eq!(read_u64(&blob, 0x18), 0);
        assert_eq!(
            [
                read_u32(&blob, 0x20),
                read_u32(&blob, 0x24),
                read_u32(&blob, 0x28),
                read_u32(&blob, 0x2c),
            ],
            [482, 159, 55, 0]
        );
        assert_eq!(read_u64(&blob, 0x30), BASE_ADDRESS + offsets.types as u64);
        assert_eq!(read_u32(&blob, 0x38), 128);

        let missing_group = offsets.types + 2 * 12;
        assert_eq!(
            [
                read_u32(&blob, missing_group),
                read_u32(&blob, missing_group + 4),
                read_u32(&blob, missing_group + 8),
            ],
            [0xa6d1_2629, 0x0004_0408, 0]
        );

        let blend_group = offsets.types + 60 * 12;
        assert_eq!(read_u32(&blob, blend_group), 0xef55_0356);
        assert_eq!(read_u32(&blob, blend_group + 4), 0x0020_04f0);
        assert_eq!(read_u32(&blob, blend_group + 8), 0);

        let blend_pointer = read_u64(&blob, offsets.pointer_tables[0] + 60 * 8);
        assert_eq!(
            blend_pointer,
            BASE_ADDRESS + offsets.register_blocks[0] as u64 + 0x51 * 8
        );
        let blend_registers = blob_offset(blend_pointer);
        for slot in 0..8 {
            assert_eq!(
                read_u32(&blob, blend_registers + slot * 8),
                0x1e0 + slot as u32
            );
            assert_eq!(read_u32(&blob, blend_registers + slot * 8 + 4), 0x2001_0001);
        }
    }

    #[test]
    fn serializes_exact_version_ten_internal_fourth_table() {
        let layout = v10::INTERNAL;
        assert_eq!(
            layout.tables.map(|table| table.registers.len()),
            [9, 15, 1, 6]
        );
        assert_eq!(
            layout.tables.map(|table| table.pointer_offsets.len()),
            [9, 15, 1, 3]
        );
        assert_eq!(layout.type_count(), 28);

        let offsets = layout.offsets();
        let blob = layout
            .build(BASE_ADDRESS)
            .expect("valid v10 internal layout");
        assert_eq!(
            [
                read_u32(&blob, 0x20),
                read_u32(&blob, 0x24),
                read_u32(&blob, 0x28),
                read_u32(&blob, 0x2c),
            ],
            [9, 15, 1, 6]
        );
        assert_eq!(read_u32(&blob, 0x38), 28);
        assert_eq!(
            read_u64(&blob, 0x18),
            BASE_ADDRESS + offsets.pointer_tables[3] as u64
        );

        let table_three_pointers = (0..3)
            .map(|index| blob_offset(read_u64(&blob, offsets.pointer_tables[3] + index * 8)))
            .collect::<Vec<_>>();
        assert_eq!(
            table_three_pointers,
            [0, 1, 3]
                .map(|index| offsets.register_blocks[3] + index * 8)
                .to_vec()
        );
        assert_eq!(
            (0..6)
                .map(|index| read_u32(&blob, offsets.register_blocks[3] + index * 8))
                .collect::<Vec<_>>(),
            [0x26c, 0x094, 0x095, 0x26e, 0x278, 0x262]
        );

        // The legacy internal table has no fourth pointer table and only 22 types. Keeping this
        // assertion ties the version switch to the ABI discrepancy that motivated it.
        assert_eq!(INTERNAL_REGISTER_DEFAULTS.len(), 22);
    }

    #[test]
    fn version_ten_blobs_match_the_complete_independent_tables() {
        let public = v10::PUBLIC.build(BASE_ADDRESS).expect("v10 public blob");
        let internal = v10::INTERNAL
            .build(BASE_ADDRESS)
            .expect("v10 internal blob");

        assert_eq!(public.len(), 8192);
        assert_eq!(fnv1a64(&public), 0xaa1a_1758_16e7_a905);
        assert_eq!(internal.len(), 872);
        assert_eq!(fnv1a64(&internal), 0xc289_54cb_1176_32d3);
    }
}
