// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Immutable ACM batch plans decoded from guest-owned command arenas.

use super::abi::{AcmError, GuestPod, SceAcmBatchInfo, checked_span, read_array};
use super::builders::{RECORD_MAGIC, RECORD_VERSION, RecordKind};

const COMMAND_STATISTICS: u8 = 11;
const COMMAND_NOTIFICATION: u8 = 14;
const STATISTICS_BYTES: usize = 24;
const NOTIFICATION_BYTES: usize = 128;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BatchPlan {
    pub(super) commands: Vec<Command>,
    pub(super) notification: Option<Notification>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum Command {
    /// Test-only private no-op record. It follows the same framing rules as DSP records.
    Activate,
    Barrier {
        test_tag: Option<u64>,
    },
    Transfer(TransferKind),
    Run,
    Statistics(Statistics),
    Inline {
        chained: bool,
        constant: bool,
    },
    Queue,
    Delay,
    #[cfg(test)]
    TestWrite(PendingWrite),
    #[cfg(test)]
    TestFault,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TransferKind {
    ChainInput,
    ChainInputSplit,
    ChainOutput,
    ChainOutputSplit,
    Input,
    InputSplit,
    Output,
    OutputSplit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Statistics {
    pub(super) address: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct AcmStatistics {
    pub(super) flags: u32,
    pub(super) usage_batch: u32,
    pub(super) usage_intervals: [u32; 3],
    pub(super) pad: u32,
}

// SAFETY: the SDK layout is six consecutive `u32` values with no padding.
unsafe impl GuestPod for AcmStatistics {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PendingWrite {
    pub(super) address: u64,
    pub(super) bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Notification {
    pub(super) value: u8,
    pub(super) address: u64,
}

pub(super) fn decode_buffers(infos: &[SceAcmBatchInfo]) -> Result<BatchPlan, AcmError> {
    if infos.is_empty() {
        return Err(AcmError::InvalidParameter);
    }

    let mut plan = BatchPlan {
        commands: Vec::new(),
        notification: None,
    };
    let mut saw_statistics = false;
    let mut chain_active = false;

    for (buffer_index, info) in infos.iter().enumerate() {
        let words = read_buffer_words(info)?;
        let mut cursor = 0_usize;
        while cursor < words.len() {
            let word = words[cursor];
            if word as u32 == RECORD_MAGIC {
                let kind =
                    RecordKind::from_byte((word >> 32) as u8).ok_or(AcmError::MalformedBatch)?;
                let version = (word >> 40) as u8;
                let record_words = usize::from((word >> 48) as u16);
                if version != RECORD_VERSION || record_words == 0 {
                    return Err(AcmError::MalformedBatch);
                }
                let end = cursor
                    .checked_add(record_words)
                    .filter(|end| *end <= words.len())
                    .ok_or(AcmError::MalformedBatch)?;
                match kind {
                    RecordKind::Test => {
                        let tag = words.get(cursor + 1).copied().unwrap_or_default();
                        plan.commands.push(Command::Barrier {
                            test_tag: Some(tag),
                        });
                    }
                }
                cursor = end;
                continue;
            }

            let command = word as u8;
            match command {
                0 => {
                    require_empty_fields(word)?;
                    plan.commands.push(Command::Activate);
                    chain_active = true;
                    cursor += 1;
                }
                1 => {
                    require_empty_fields(word)?;
                    plan.commands.push(Command::Barrier { test_tag: None });
                    cursor += 1;
                }
                2..=9 => {
                    require_empty_fields(word)?;
                    if command <= 5 && !chain_active {
                        return Err(AcmError::MalformedBatch);
                    }
                    let kind = match command {
                        2 => TransferKind::ChainInput,
                        3 => TransferKind::ChainInputSplit,
                        4 => TransferKind::ChainOutput,
                        5 => TransferKind::ChainOutputSplit,
                        6 => TransferKind::Input,
                        7 => TransferKind::InputSplit,
                        8 => TransferKind::Output,
                        9 => TransferKind::OutputSplit,
                        _ => unreachable!(),
                    };
                    plan.commands.push(Command::Transfer(kind));
                    cursor += 1;
                }
                10 => {
                    require_empty_fields(word)?;
                    plan.commands.push(Command::Run);
                    cursor += 1;
                }
                COMMAND_STATISTICS => {
                    if word != u64::from(COMMAND_STATISTICS) || saw_statistics {
                        return Err(AcmError::MalformedBatch);
                    }
                    let address = continuation(&words, cursor)?;
                    checked_span(address, STATISTICS_BYTES, 4)?;
                    plan.commands
                        .push(Command::Statistics(Statistics { address }));
                    saw_statistics = true;
                    cursor += 2;
                }
                12 | 13 | 15 => {
                    require_empty_fields(word)?;
                    let chained = command != 13;
                    if chained && !chain_active {
                        return Err(AcmError::MalformedBatch);
                    }
                    plan.commands.push(Command::Inline {
                        chained,
                        constant: command == 15,
                    });
                    cursor += 1;
                }
                COMMAND_NOTIFICATION => {
                    if word & !0xffff != 0 {
                        return Err(AcmError::MalformedBatch);
                    }
                    let value = (word >> 8) as u8;
                    let address = continuation(&words, cursor)?;
                    if value == 0 || address == 0 {
                        return Err(AcmError::MalformedBatch);
                    }
                    if !address.is_multiple_of(NOTIFICATION_BYTES as u64) {
                        return Err(AcmError::Alignment);
                    }
                    checked_span(address, NOTIFICATION_BYTES, NOTIFICATION_BYTES)?;
                    if plan.notification.is_some()
                        || buffer_index + 1 != infos.len()
                        || cursor + 2 != words.len()
                    {
                        return Err(AcmError::MalformedBatch);
                    }
                    plan.notification = Some(Notification { value, address });
                    cursor += 2;
                }
                16 => {
                    require_empty_fields(word)?;
                    plan.commands.push(Command::Queue);
                    cursor += 1;
                }
                17 => {
                    require_empty_fields(word)?;
                    plan.commands.push(Command::Delay);
                    cursor += 1;
                }
                _ => return Err(AcmError::MalformedBatch),
            }
        }
    }

    Ok(plan)
}

fn require_empty_fields(word: u64) -> Result<(), AcmError> {
    if word >> 8 != 0 {
        return Err(AcmError::MalformedBatch);
    }
    Ok(())
}

fn read_buffer_words(info: &SceAcmBatchInfo) -> Result<Vec<u64>, AcmError> {
    if info.buffer == 0
        || !info.buffer.is_multiple_of(8)
        || !info.offset.is_multiple_of(8)
        || info.offset > info.size
    {
        return Err(AcmError::InvalidParameter);
    }
    let used = usize::try_from(info.offset).map_err(|_| AcmError::InvalidParameter)?;
    let word_count = used / 8;
    read_array(info.buffer, word_count)
}

fn continuation(words: &[u64], cursor: usize) -> Result<u64, AcmError> {
    words
        .get(cursor + 1)
        .copied()
        .ok_or(AcmError::MalformedBatch)
}
