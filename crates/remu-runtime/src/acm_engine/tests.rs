// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::mem::size_of;

use super::AcmEngine;
use super::abi::{AcmError, AcmFormat, SceAcmBatchCommand, SceAcmBatchInfo, return_error};
use super::batch::{Command, decode_buffers};
use super::builders::append_test_record;

const SUCCESS: u64 = 0;

fn args(values: &[u64]) -> [u64; 6] {
    let mut arguments = [0_u64; 6];
    arguments[..values.len()].copy_from_slice(values);
    arguments
}

fn address_mut<T>(value: &mut T) -> u64 {
    std::ptr::from_mut(value) as u64
}

#[repr(align(128))]
struct AlignedBytes([u8; 256]);

fn write_word(bytes: &mut [u8], index: usize, word: u64) {
    let start = index * 8;
    bytes[start..start + 8].copy_from_slice(&word.to_le_bytes());
}

#[test]
fn sdk10_abi_sizes_formats_and_error_words_are_exact() {
    assert_eq!(size_of::<SceAcmBatchInfo>(), 24);
    assert_eq!(size_of::<SceAcmBatchCommand>(), 8);
    assert_eq!(AcmFormat::Fp32 as u32, 0);
    assert_eq!(AcmFormat::Fp16 as u32, 1);
    assert_eq!(
        [
            AcmError::Unknown as u32,
            AcmError::InvalidContext as u32,
            AcmError::InvalidBatch as u32,
            AcmError::OutOfMemory as u32,
            AcmError::OutOfResources as u32,
            AcmError::InvalidParameter as u32,
            AcmError::MalformedBatch as u32,
            AcmError::Busy as u32,
            AcmError::BatchBufferFull as u32,
            AcmError::Alignment as u32,
            AcmError::PageFault as u32,
        ],
        [
            0x8194_0001,
            0x8194_0002,
            0x8194_0003,
            0x8194_0004,
            0x8194_0005,
            0x8194_0006,
            0x8194_0007,
            0x8194_0008,
            0x8194_0009,
            0x8194_000a,
            0x8194_000b,
        ]
    );
    assert_eq!(
        return_error(AcmError::InvalidParameter),
        0xffff_ffff_8194_0006
    );
}

#[test]
fn context_ids_are_checked_and_reset_is_process_local() {
    let engine = AcmEngine::new();
    let mut first = 0_u32;
    let mut second = 0_u32;
    assert_eq!(
        engine.context_create(args(&[address_mut(&mut first)])),
        SUCCESS
    );
    assert_eq!(
        engine.context_create(args(&[address_mut(&mut second)])),
        SUCCESS
    );
    assert_eq!((first, second), (1, 2));
    assert_eq!(
        engine.context_create(args(&[0])),
        return_error(AcmError::InvalidParameter)
    );
    assert_eq!(engine.context_destroy(args(&[u64::from(first)])), SUCCESS);
    assert_eq!(
        engine.context_destroy(args(&[u64::from(first)])),
        return_error(AcmError::InvalidContext)
    );

    engine.reset();
    assert_eq!(
        engine.context_destroy(args(&[u64::from(second)])),
        return_error(AcmError::InvalidContext)
    );
    let mut after_reset = 0_u32;
    assert_eq!(
        engine.context_create(args(&[address_mut(&mut after_reset)])),
        SUCCESS
    );
    assert_eq!(after_reset, 1);
}

#[test]
fn builder_reserves_exact_words_and_does_not_mutate_on_full_or_overflow() {
    let mut arena = AlignedBytes([0; 256]);
    let address = arena.0.as_mut_ptr() as u64;
    let mut info = SceAcmBatchInfo {
        buffer: address,
        offset: 8,
        size: 32,
    };

    let before = info;
    assert_eq!(append_test_record(address_mut(&mut info), 3), Ok(()));
    assert_eq!(info.offset, 32);
    assert_ne!(&arena.0[8..32], &[0; 24]);

    let full = info;
    let arena_before_full = arena.0;
    assert_eq!(
        append_test_record(address_mut(&mut info), 1),
        Err(AcmError::BatchBufferFull)
    );
    assert_eq!(info, full);
    assert_eq!(arena.0, arena_before_full);

    info.offset = u64::MAX - 3;
    let arena_before_overflow = arena.0;
    assert_eq!(
        append_test_record(address_mut(&mut info), 1),
        Err(AcmError::InvalidParameter)
    );
    assert_eq!(info.offset, u64::MAX - 3);
    assert_eq!(arena.0, arena_before_overflow);
    assert_ne!(before.offset, full.offset);
}

#[test]
fn decoded_plan_survives_arena_copy_and_unknown_records_fail_loudly() {
    let mut statistics = [0_u32; 6];
    let mut notification = AlignedBytes([0; 256]);
    let mut original = AlignedBytes([0; 256]);
    write_word(&mut original.0, 0, 11);
    write_word(
        &mut original.0,
        1,
        std::ptr::from_mut(&mut statistics) as u64,
    );
    write_word(&mut original.0, 2, 14 | (7 << 8));
    write_word(&mut original.0, 3, notification.0.as_mut_ptr() as u64);

    let mut copied = AlignedBytes(original.0);
    let copied_info = SceAcmBatchInfo {
        buffer: copied.0.as_mut_ptr() as u64,
        offset: 32,
        size: copied.0.len() as u64,
    };
    let plan = decode_buffers(&[copied_info]).expect("valid copied command stream");
    assert!(matches!(plan.commands.as_slice(), [Command::Statistics(_)]));
    let completion = plan.notification.expect("last notification");
    assert_eq!(completion.value, 7);
    assert_eq!(completion.address, notification.0.as_mut_ptr() as u64);

    let mut unknown = AlignedBytes([0; 256]);
    let mut unknown_info = SceAcmBatchInfo {
        buffer: unknown.0.as_mut_ptr() as u64,
        offset: 0,
        size: unknown.0.len() as u64,
    };
    append_test_record(address_mut(&mut unknown_info), 1).unwrap();
    unknown.0[4] = 0xff;
    assert_eq!(unknown.0[4], 0xff);
    assert_eq!(
        decode_buffers(&[unknown_info]),
        Err(AcmError::MalformedBatch)
    );

    copied.0[2] = 1;
    assert_eq!(
        decode_buffers(&[copied_info]),
        Err(AcmError::MalformedBatch),
        "nonzero reserved fields must not be guessed"
    );

    let truncated = SceAcmBatchInfo {
        buffer: copied.0.as_mut_ptr() as u64,
        offset: 8,
        size: copied.0.len() as u64,
    };
    copied.0[2] = 0;
    assert_eq!(copied.0[2], 0);
    assert_eq!(
        decode_buffers(&[truncated]),
        Err(AcmError::MalformedBatch),
        "an inline continuation cannot cross a buffer boundary"
    );
}
