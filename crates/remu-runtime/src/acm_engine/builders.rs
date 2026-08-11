// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Checked host implementations of the imported ACM batch builders.

use crate::write_guest_bytes;

use super::abi::{AcmError, SceAcmBatchInfo, read_pod, write_pod};

pub(super) const RECORD_MAGIC: u32 = 0x524d_4143;
pub(super) const RECORD_VERSION: u8 = 1;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RecordKind {
    Test = 1,
}

impl RecordKind {
    pub(super) fn from_byte(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Test),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RecordPayload {
    kind: RecordKind,
    words: Vec<u64>,
}

impl RecordPayload {
    #[cfg(test)]
    fn test(words: usize) -> Self {
        Self {
            kind: RecordKind::Test,
            words: (1..words).map(|word| word as u64).collect(),
        }
    }

    #[cfg(test)]
    fn tagged_test(tag: u64) -> Self {
        Self {
            kind: RecordKind::Test,
            words: vec![tag],
        }
    }
}

fn record_header(kind: RecordKind, words: u16) -> u64 {
    u64::from(RECORD_MAGIC)
        | (u64::from(kind as u8) << 32)
        | (u64::from(RECORD_VERSION) << 40)
        | (u64::from(words) << 48)
}

pub(super) fn append_record(
    info_address: u64,
    words: usize,
    payload: RecordPayload,
) -> Result<(), AcmError> {
    if words == 0 || words > usize::from(u16::MAX) || payload.words.len() > words - 1 {
        return Err(AcmError::InvalidParameter);
    }

    let info = read_pod::<SceAcmBatchInfo>(info_address).map_err(|_| AcmError::InvalidParameter)?;
    if info.buffer == 0 || !info.offset.is_multiple_of(8) || info.offset > info.size {
        return Err(AcmError::InvalidParameter);
    }

    let byte_count = words.checked_mul(8).ok_or(AcmError::InvalidParameter)?;
    let byte_count_u64 = u64::try_from(byte_count).map_err(|_| AcmError::InvalidParameter)?;
    let new_offset = info
        .offset
        .checked_add(byte_count_u64)
        .ok_or(AcmError::InvalidParameter)?;
    if new_offset > info.size {
        return Err(AcmError::BatchBufferFull);
    }
    let destination = info
        .buffer
        .checked_add(info.offset)
        .ok_or(AcmError::InvalidParameter)?;
    super::abi::checked_span(destination, byte_count, 8).map_err(|_| AcmError::InvalidParameter)?;

    let mut encoded = vec![0_u8; byte_count];
    encoded[..8].copy_from_slice(
        &record_header(
            payload.kind,
            u16::try_from(words).expect("validated record word count"),
        )
        .to_le_bytes(),
    );
    for (index, word) in payload.words.iter().enumerate() {
        let start = (index + 1) * 8;
        encoded[start..start + 8].copy_from_slice(&word.to_le_bytes());
    }

    write_guest_bytes(destination, &encoded);
    let committed = SceAcmBatchInfo {
        offset: new_offset,
        ..info
    };
    write_pod(info_address, &committed).map_err(|_| AcmError::InvalidParameter)
}

#[cfg(test)]
pub(super) fn append_test_record(info_address: u64, words: usize) -> Result<(), AcmError> {
    append_record(info_address, words, RecordPayload::test(words))
}

#[cfg(test)]
pub(super) fn append_tagged_test_record(info_address: u64, tag: u64) -> Result<(), AcmError> {
    append_record(info_address, 2, RecordPayload::tagged_test(tag))
}
