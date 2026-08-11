// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::mem::size_of;

use super::AcmEngine;
use super::abi::{AcmError, AcmFormat, SceAcmBatchCommand, SceAcmBatchInfo, return_error};
use super::batch::{Command, decode_buffers};
use super::builders::{append_tagged_test_record, append_test_record};

const SUCCESS: u64 = 0;

fn args(values: &[u64]) -> [u64; 6] {
    let mut arguments = [0_u64; 6];
    arguments[..values.len()].copy_from_slice(values);
    arguments
}

fn address_mut<T>(value: &mut T) -> u64 {
    std::ptr::from_mut(value) as u64
}

fn address<T>(value: &T) -> u64 {
    std::ptr::from_ref(value) as u64
}

#[repr(align(128))]
struct AlignedBytes([u8; 256]);

fn write_word(bytes: &mut [u8], index: usize, word: u64) {
    let start = index * 8;
    bytes[start..start + 8].copy_from_slice(&word.to_le_bytes());
}

fn create_context(engine: &AcmEngine) -> u32 {
    let mut context = 0_u32;
    assert_eq!(
        engine.context_create(args(&[address_mut(&mut context)])),
        SUCCESS
    );
    context
}

fn start_empty_batch(engine: &AcmEngine, context: u32, batch_error: u64) -> Result<u32, AcmError> {
    let mut arena = AlignedBytes([0; 256]);
    let info = SceAcmBatchInfo {
        buffer: arena.0.as_mut_ptr() as u64,
        offset: 0,
        size: arena.0.len() as u64,
    };
    let info_pointers = [address(&info)];
    let mut batch = 0_u32;
    let result = engine.batch_start_buffers(args(&[
        u64::from(context),
        1,
        address(&info_pointers),
        batch_error,
        address_mut(&mut batch),
    ]));
    if result == SUCCESS {
        Ok(batch)
    } else {
        Err([
            AcmError::Unknown,
            AcmError::InvalidContext,
            AcmError::InvalidBatch,
            AcmError::OutOfMemory,
            AcmError::OutOfResources,
            AcmError::InvalidParameter,
            AcmError::MalformedBatch,
            AcmError::Busy,
            AcmError::BatchBufferFull,
            AcmError::Alignment,
            AcmError::PageFault,
        ]
        .into_iter()
        .find(|error| return_error(*error) == result)
        .expect("known ACM error"))
    }
}

fn start_tagged(engine: &AcmEngine, context: u32, tag: u64) -> u32 {
    let mut arena = AlignedBytes([0; 256]);
    let mut info = SceAcmBatchInfo {
        buffer: arena.0.as_mut_ptr() as u64,
        offset: 0,
        size: arena.0.len() as u64,
    };
    append_tagged_test_record(address_mut(&mut info), tag).unwrap();
    let info_pointers = [address(&info)];
    let mut batch = 0_u32;
    assert_eq!(
        engine.batch_start_buffers(args(&[
            u64::from(context),
            1,
            address(&info_pointers),
            0,
            address_mut(&mut batch),
        ])),
        SUCCESS
    );
    batch
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

#[test]
fn batch_start_is_async_wait_consumes_retirement_and_poll_reports_busy() {
    let engine = AcmEngine::new();
    let context = create_context(&engine);
    let batch = start_empty_batch(&engine, context, 0).expect("valid batch starts");
    assert_eq!(
        engine.batch_wait(args(&[u64::from(context), u64::from(batch), 0])),
        return_error(AcmError::Busy)
    );
    assert!(engine.execute_next_for_test());
    assert_eq!(
        engine.batch_wait(args(&[u64::from(context), u64::from(batch), 0])),
        SUCCESS
    );
    assert_eq!(
        engine.batch_wait(args(&[u64::from(context), u64::from(batch), 0])),
        return_error(AcmError::InvalidBatch)
    );
    assert_eq!(
        start_empty_batch(&engine, context, 8),
        Err(AcmError::InvalidParameter),
        "SDK 10 only supports a null SceAcmBatchError pointer"
    );
}

#[test]
fn finite_and_infinite_waits_share_one_waiter_slot() {
    let engine = AcmEngine::new();
    let context = create_context(&engine);
    let batch = start_empty_batch(&engine, context, 0).unwrap();
    assert_eq!(
        engine.batch_wait(args(&[u64::from(context), u64::from(batch), 1])),
        return_error(AcmError::Busy)
    );
    engine.set_waiter_active_for_test(batch, true);
    assert_eq!(
        engine.batch_wait(args(&[
            u64::from(context),
            u64::from(batch),
            u64::from(u32::MAX),
        ])),
        return_error(AcmError::Busy)
    );
    engine.set_waiter_active_for_test(batch, false);
    std::thread::scope(|scope| {
        scope.spawn(|| assert!(engine.execute_next_for_test()));
        assert_eq!(
            engine.batch_wait(args(&[
                u64::from(context),
                u64::from(batch),
                u64::from(u32::MAX),
            ])),
            SUCCESS
        );
    });
}

#[test]
fn each_context_is_fifo_while_distinct_contexts_can_run_independently() {
    let engine = AcmEngine::new();
    let a = create_context(&engine);
    let b = create_context(&engine);
    let a1 = start_tagged(&engine, a, 11);
    let a2 = start_tagged(&engine, a, 12);
    let b1 = start_tagged(&engine, b, 21);

    assert_eq!(engine.runnable_for_test(), vec![a1, b1]);
    assert!(engine.execute_batch_for_test(b1));
    assert!(engine.execute_batch_for_test(a1));
    assert_eq!(engine.runnable_for_test(), vec![a2]);
    assert_eq!(engine.execution_tags_for_test(), vec![21, 11]);
}

#[test]
fn destroy_drains_and_reset_cancels() {
    let engine = AcmEngine::new();
    let context = create_context(&engine);
    let batch = start_tagged(&engine, context, 31);

    std::thread::scope(|scope| {
        let destroy = scope.spawn(|| engine.context_destroy(args(&[u64::from(context)])));
        while !engine.context_is_destroying_for_test(context) {
            std::thread::yield_now();
        }
        assert_eq!(
            start_empty_batch(&engine, context, 0),
            Err(AcmError::InvalidContext),
            "destroy must close the context to new submissions before draining"
        );
        assert!(engine.execute_batch_for_test(batch));
        assert_eq!(destroy.join().unwrap(), SUCCESS);
    });

    let context = create_context(&engine);
    let batch = start_tagged(&engine, context, 41);
    std::thread::scope(|scope| {
        let waiter = scope.spawn(|| {
            engine.batch_wait(args(&[
                u64::from(context),
                u64::from(batch),
                u64::from(u32::MAX),
            ]))
        });
        while !engine.waiter_active_for_test(batch) {
            std::thread::yield_now();
        }
        engine.reset();
        assert_eq!(waiter.join().unwrap(), return_error(AcmError::InvalidBatch));
    });
    assert_eq!(create_context(&engine), 1);
}
