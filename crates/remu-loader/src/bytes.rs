// SPDX-License-Identifier: GPL-2.0-or-later

//! Bounds-checked primitive readers shared by the container and ELF parsers.

use std::io::{Read, Seek, SeekFrom};

use crate::InspectError;

pub(crate) fn read_vec_at(
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

pub(crate) fn read_array_at<const N: usize>(
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

pub(crate) fn checked_end(offset: u64, size: u64, file_size: u64) -> Result<u64, InspectError> {
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

pub(crate) fn read_u16<const N: usize>(bytes: &[u8; N], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

pub(crate) fn read_u32<const N: usize>(bytes: &[u8; N], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

pub(crate) fn read_u64<const N: usize>(bytes: &[u8; N], offset: usize) -> u64 {
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

pub(crate) fn read_i64_slice(bytes: &[u8], offset: usize) -> i64 {
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

pub(crate) fn read_u16_slice(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

pub(crate) fn read_u32_slice(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

pub(crate) fn read_u64_slice(bytes: &[u8], offset: usize) -> u64 {
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
