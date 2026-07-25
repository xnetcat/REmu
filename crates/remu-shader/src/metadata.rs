// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::BTreeMap;

use thiserror::Error;

use crate::{ResourceKind, ResourceMapping, ShaderMetadata};

const SHADER_USER_DATA_OFFSET: u64 = 0x08;
const RESOURCE_CLASS_COUNT: usize = 4;
const MAX_METADATA_ENTRIES: u16 = 4096;

pub trait GuestMemoryReader {
    fn read(&self, address: u64, output: &mut [u8]) -> bool;

    /// Returns the readable prefix of `maximum_length` bytes starting at `address` when the
    /// backing implementation can answer without copying the range.
    ///
    /// Evaluating a shader needs the extent of global buffers, but it does not necessarily need
    /// a CPU snapshot of their contents. Readers backed by identity-mapped guest memory can use
    /// this hook to keep large bindings zero-copy. `None` asks the evaluator to fall back to a
    /// validating read, preserving the behavior of simple/test readers.
    fn readable_prefix_length(&self, _address: u64, _maximum_length: usize) -> Option<usize> {
        None
    }
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum MetadataError {
    #[error("shader metadata address arithmetic overflowed")]
    AddressOverflow,
    #[error("shader metadata memory is not readable at {address:#x}")]
    MemoryFault { address: u64 },
    #[error("shader metadata has a null {field} pointer")]
    MissingPointer { field: &'static str },
    #[error("shader metadata {field} count {count} exceeds the supported format maximum")]
    EntryCountExceeded { field: &'static str, count: u16 },
    #[error("shader metadata contains an invalid resource class {0}")]
    InvalidResourceClass(usize),
}

pub fn read_shader_metadata(
    memory: &impl GuestMemoryReader,
    shader_header: u64,
) -> Result<ShaderMetadata, MetadataError> {
    let user_data_address = read_u64(memory, checked_add(shader_header, SHADER_USER_DATA_OFFSET)?)?;
    if user_data_address == 0 {
        return Err(MetadataError::MissingPointer { field: "user-data" });
    }
    let direct_offsets_address = read_u64(memory, user_data_address)?;
    let mut resource_offsets = [0_u64; RESOURCE_CLASS_COUNT];
    for (resource_class, address) in resource_offsets.iter_mut().enumerate() {
        let offset = 0x08_u64
            .checked_add((resource_class as u64) * 8)
            .ok_or(MetadataError::AddressOverflow)?;
        *address = read_u64(memory, checked_add(user_data_address, offset)?)?;
    }

    let extended_user_data_size_dwords =
        u32::from(read_u16(memory, checked_add(user_data_address, 0x28)?)?);
    let shader_resource_table_size_dwords =
        u32::from(read_u16(memory, checked_add(user_data_address, 0x2a)?)?);
    let direct_resource_count = read_count(
        memory,
        checked_add(user_data_address, 0x2c)?,
        "direct-resource",
    )?;
    let mut resource_counts = [0_u16; RESOURCE_CLASS_COUNT];
    for (resource_class, count) in resource_counts.iter_mut().enumerate() {
        let offset = 0x2e_u64
            .checked_add((resource_class as u64) * 2)
            .ok_or(MetadataError::AddressOverflow)?;
        *count = read_count(
            memory,
            checked_add(user_data_address, offset)?,
            "resource-class",
        )?;
    }

    let mut direct_resources = BTreeMap::new();
    if direct_resource_count != 0 {
        if direct_offsets_address == 0 {
            return Err(MetadataError::MissingPointer {
                field: "direct-resource-offsets",
            });
        }
        for resource_type in 0..u32::from(direct_resource_count) {
            let address = checked_add(direct_offsets_address, u64::from(resource_type) * 2)?;
            let offset = read_u16(memory, address)?;
            if offset != u16::MAX {
                direct_resources.insert(resource_type, u32::from(offset));
            }
        }
    }

    let mut resources = Vec::new();
    for resource_class in 0..RESOURCE_CLASS_COUNT {
        let count = resource_counts[resource_class];
        if count == 0 {
            continue;
        }
        let base = resource_offsets[resource_class];
        if base == 0 {
            return Err(MetadataError::MissingPointer {
                field: "resource-offsets",
            });
        }
        let kind = ResourceKind::try_from(resource_class)
            .map_err(|()| MetadataError::InvalidResourceClass(resource_class))?;
        for slot in 0..u32::from(count) {
            let encoded = read_u16(memory, checked_add(base, u64::from(slot) * 2)?)?;
            let offset_dwords = u32::from(encoded & 0x7fff);
            if offset_dwords == 0x7fff {
                continue;
            }
            resources.push(ResourceMapping {
                kind,
                slot,
                offset_dwords,
                size_flag: encoded & 0x8000 != 0,
            });
        }
    }

    Ok(ShaderMetadata {
        extended_user_data_size_dwords,
        shader_resource_table_size_dwords,
        direct_resources,
        resources,
    })
}

fn checked_add(address: u64, offset: u64) -> Result<u64, MetadataError> {
    address
        .checked_add(offset)
        .ok_or(MetadataError::AddressOverflow)
}

fn read_count(
    memory: &impl GuestMemoryReader,
    address: u64,
    field: &'static str,
) -> Result<u16, MetadataError> {
    let count = read_u16(memory, address)?;
    if count > MAX_METADATA_ENTRIES {
        Err(MetadataError::EntryCountExceeded { field, count })
    } else {
        Ok(count)
    }
}

fn read_u16(memory: &impl GuestMemoryReader, address: u64) -> Result<u16, MetadataError> {
    let mut bytes = [0_u8; 2];
    if !memory.read(address, &mut bytes) {
        return Err(MetadataError::MemoryFault { address });
    }
    Ok(u16::from_le_bytes(bytes))
}

fn read_u64(memory: &impl GuestMemoryReader, address: u64) -> Result<u64, MetadataError> {
    let mut bytes = [0_u8; 8];
    if !memory.read(address, &mut bytes) {
        return Err(MetadataError::MemoryFault { address });
    }
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestMemory {
        base: u64,
        bytes: Vec<u8>,
    }

    impl TestMemory {
        fn new(base: u64, length: usize) -> Self {
            Self {
                base,
                bytes: vec![0; length],
            }
        }

        fn write_u16(&mut self, address: u64, value: u16) {
            self.write(address, &value.to_le_bytes());
        }

        fn write_u64(&mut self, address: u64, value: u64) {
            self.write(address, &value.to_le_bytes());
        }

        fn write(&mut self, address: u64, input: &[u8]) {
            let offset = usize::try_from(address - self.base).unwrap();
            self.bytes[offset..offset + input.len()].copy_from_slice(input);
        }
    }

    impl GuestMemoryReader for TestMemory {
        fn read(&self, address: u64, output: &mut [u8]) -> bool {
            let Some(relative) = address.checked_sub(self.base) else {
                return false;
            };
            let Ok(offset) = usize::try_from(relative) else {
                return false;
            };
            let Some(end) = offset.checked_add(output.len()) else {
                return false;
            };
            let Some(source) = self.bytes.get(offset..end) else {
                return false;
            };
            output.copy_from_slice(source);
            true
        }
    }

    #[test]
    fn reads_direct_and_classed_resource_metadata() {
        let mut memory = TestMemory::new(0x1000, 0x1000);
        let header = 0x1000;
        let user_data = 0x1100;
        let direct = 0x1200;
        let textures = 0x1300;
        let samplers = 0x1400;
        memory.write_u64(header + 8, user_data);
        memory.write_u64(user_data, direct);
        memory.write_u64(user_data + 8, textures);
        memory.write_u64(user_data + 0x18, samplers);
        memory.write_u16(user_data + 0x28, 24);
        memory.write_u16(user_data + 0x2a, 48);
        memory.write_u16(user_data + 0x2c, 2);
        memory.write_u16(user_data + 0x2e, 2);
        memory.write_u16(user_data + 0x32, 1);
        memory.write_u16(direct, 7);
        memory.write_u16(direct + 2, u16::MAX);
        memory.write_u16(textures, 0x8004);
        memory.write_u16(textures + 2, 0x7fff);
        memory.write_u16(samplers, 12);

        let metadata = read_shader_metadata(&memory, header).unwrap();
        assert_eq!(metadata.extended_user_data_size_dwords, 24);
        assert_eq!(metadata.shader_resource_table_size_dwords, 48);
        assert_eq!(metadata.direct_resources, BTreeMap::from([(0, 7)]));
        assert_eq!(
            metadata.resources,
            vec![
                ResourceMapping {
                    kind: ResourceKind::ReadOnlyTexture,
                    slot: 0,
                    offset_dwords: 4,
                    size_flag: true,
                },
                ResourceMapping {
                    kind: ResourceKind::Sampler,
                    slot: 0,
                    offset_dwords: 12,
                    size_flag: false,
                },
            ]
        );
    }

    #[test]
    fn rejects_missing_tables_and_unbounded_counts() {
        let mut memory = TestMemory::new(0x2000, 0x400);
        memory.write_u64(0x2008, 0x2100);
        memory.write_u16(0x212c, 4097);
        assert_eq!(
            read_shader_metadata(&memory, 0x2000),
            Err(MetadataError::EntryCountExceeded {
                field: "direct-resource",
                count: 4097,
            })
        );
    }
}
