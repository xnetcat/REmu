// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::mem::size_of;
use std::time::Duration;

use super::AcmEngine;
use super::abi::{
    AcmError, AcmFormat, ConvChannel, ConvIn, ConvIr, ConvOut, SceAcmBatchCommand, SceAcmBatchInfo,
    return_error,
};
use super::batch::{AcmStatistics, BatchPlan, Command, Notification, PendingWrite, decode_buffers};
use super::builders::{
    append_tagged_test_record, append_test_record, batch_job_notification, convolution, fft, panner,
};

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

#[repr(align(128))]
struct AlignedLarge([u8; 8192]);

fn write_word(bytes: &mut [u8], index: usize, word: u64) {
    let start = index * 8;
    bytes[start..start + 8].copy_from_slice(&word.to_le_bytes());
}

fn decode_inline_words(
    arena: &mut AlignedBytes,
    words: &[u64],
) -> Result<super::batch::BatchPlan, AcmError> {
    arena.0.fill(0);
    for (index, word) in words.iter().copied().enumerate() {
        write_word(&mut arena.0, index, word);
    }
    decode_buffers(&[SceAcmBatchInfo {
        buffer: arena.0.as_mut_ptr() as u64,
        offset: (words.len() * 8) as u64,
        size: arena.0.len() as u64,
    }])
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

#[test]
fn every_public_command_id_has_a_typed_decoder_and_reserved_bits_are_rejected() {
    let mut arena = AlignedBytes([0; 256]);
    let mut statistics = [0_u32; 6];
    let mut notification = AlignedBytes([0; 256]);
    for id in 0_u8..=17 {
        let mut words = if matches!(id, 2..=5 | 12 | 15) {
            vec![0, u64::from(id)]
        } else {
            vec![u64::from(id)]
        };
        match id {
            11 => words.push(std::ptr::from_mut(&mut statistics) as u64),
            14 => {
                words[0] |= 1 << 8;
                words.push(notification.0.as_mut_ptr() as u64);
            }
            _ => {}
        }
        assert!(
            decode_inline_words(&mut arena, &words).is_ok(),
            "command {id} must have a typed decoder"
        );
    }
    assert_eq!(
        decode_inline_words(&mut arena, &[18]),
        Err(AcmError::MalformedBatch)
    );
    assert_eq!(
        decode_inline_words(&mut arena, &[1 | (0xff << 8)]),
        Err(AcmError::MalformedBatch),
        "barrier has no public payload fields"
    );
}

#[test]
fn notification_is_written_last_and_failed_batches_publish_no_partial_output() {
    let mut output = [1_u8, 2, 3, 4];
    let before = output;
    let mut notification = AlignedBytes([0; 256]);
    let plan = BatchPlan {
        commands: vec![
            Command::TestWrite(PendingWrite {
                address: output.as_mut_ptr() as u64,
                bytes: vec![9, 8, 7, 6],
            }),
            Command::TestFault,
        ],
        notification: Some(Notification {
            value: 7,
            address: notification.0.as_mut_ptr() as u64,
        }),
    };

    assert_eq!(
        super::execute_plan_with_elapsed_for_test(&plan, Duration::from_millis(2)),
        Err(AcmError::PageFault)
    );
    assert_eq!(output, before);
    assert_eq!(notification.0[0], 0);
}

#[test]
fn statistics_scale_real_elapsed_time_and_mark_unavailable_history() {
    let mut statistics = AcmStatistics {
        flags: 0,
        usage_batch: 0,
        usage_intervals: [0, 1 << 20, u32::MAX],
        pad: 0,
    };
    let plan = BatchPlan {
        commands: vec![Command::Statistics(super::batch::Statistics {
            address: address_mut(&mut statistics),
        })],
        notification: None,
    };

    assert_eq!(
        super::execute_plan_with_elapsed_for_test(&plan, Duration::from_millis(2)),
        Ok(())
    );
    assert!(statistics.usage_batch > 0);
    assert_eq!(statistics.usage_intervals[0], u32::MAX);
    assert_ne!(statistics.usage_intervals[1], u32::MAX);
}

#[test]
fn notification_builder_is_exact_aligned_and_failure_atomic() {
    let mut arena = AlignedBytes([0; 256]);
    let mut info = SceAcmBatchInfo {
        buffer: arena.0.as_mut_ptr() as u64,
        offset: 0,
        size: 16,
    };
    let mut notification = AlignedBytes([0xaa; 256]);
    assert_eq!(
        batch_job_notification(args(&[
            address_mut(&mut info),
            7,
            notification.0.as_mut_ptr() as u64,
        ])),
        SUCCESS
    );
    assert_eq!(info.offset, 16);
    assert!(notification.0[..128].iter().all(|byte| *byte == 0));
    let plan = decode_buffers(&[info]).unwrap();
    assert_eq!(plan.notification.unwrap().value, 7);

    let full = info;
    notification.0.fill(0xaa);
    assert_eq!(
        batch_job_notification(args(&[
            address_mut(&mut info),
            8,
            notification.0.as_mut_ptr() as u64,
        ])),
        return_error(AcmError::BatchBufferFull)
    );
    assert_eq!(info, full);
    assert!(notification.0.iter().all(|byte| *byte == 0xaa));

    info.offset = 0;
    assert_eq!(
        batch_job_notification(args(&[
            address_mut(&mut info),
            1,
            notification.0.as_mut_ptr() as u64 + 1,
        ])),
        return_error(AcmError::Alignment)
    );
    assert_eq!(info.offset, 0);
}

#[test]
fn execution_history_is_bounded_to_256_samples() {
    let engine = AcmEngine::new();
    let context = create_context(&engine);
    for _ in 0..300 {
        let batch = start_empty_batch(&engine, context, 0).unwrap();
        assert!(engine.execute_next_for_test());
        assert_eq!(
            engine.batch_wait(args(&[u64::from(context), u64::from(batch), 0])),
            SUCCESS
        );
    }
    assert_eq!(engine.history_len_for_test(), 256);
}

fn write_f32_values(address: u64, values: &[f32]) {
    crate::write_guest_bytes(
        address,
        &values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    );
}

fn read_first_f32(address: u64) -> f32 {
    let mut bytes = [0_u8; 4];
    crate::read_guest_bytes(address, &mut bytes);
    f32::from_le_bytes(bytes)
}

#[test]
fn mixed_fft_panner_convolution_batch_retires_with_real_outputs() {
    let mut arena = AlignedLarge([0; 8192]);
    let mut info = SceAcmBatchInfo {
        buffer: arena.0.as_mut_ptr() as u64,
        offset: 0,
        size: arena.0.len() as u64,
    };

    let mut fft_input = AlignedLarge([0; 8192]);
    let mut fft_output = AlignedLarge([0; 8192]);
    write_f32_values(fft_input.0.as_mut_ptr() as u64, &[1.0]);
    let fft_inputs = [fft_input.0.as_mut_ptr() as u64];
    let fft_outputs = [fft_output.0.as_mut_ptr() as u64];
    assert_eq!(
        fft(
            [address_mut(&mut info), 512, 1, 0, address(&fft_inputs), 0,],
            [address(&fft_outputs), 0, 0, 0, 0, 0],
            false,
        ),
        SUCCESS
    );

    let mut panner_input = AlignedLarge([0; 8192]);
    let mut panner_parameter = AlignedLarge([0; 8192]);
    let mut panner_state = AlignedLarge([0; 8192]);
    let mut panner_output = AlignedLarge([0; 8192]);
    write_f32_values(panner_input.0.as_mut_ptr() as u64, &[1.0; 256]);
    panner_parameter.0[4..8].copy_from_slice(&1.0_f32.to_le_bytes());
    panner_state.0[..4].copy_from_slice(&1.0_f32.to_le_bytes());
    let panner_inputs = [panner_input.0.as_mut_ptr() as u64];
    let panner_parameters = [panner_parameter.0.as_mut_ptr() as u64];
    let panner_states = [panner_state.0.as_mut_ptr() as u64];
    let panner_initial = [0_u64];
    let panner_outputs = [panner_output.0.as_mut_ptr() as u64];
    assert_eq!(
        panner(
            [address_mut(&mut info), 1, address(&panner_inputs), 0, 0, 1,],
            [
                address(&panner_parameters),
                address(&panner_states),
                address(&panner_initial),
                address(&panner_outputs),
                0,
                0,
            ],
        ),
        SUCCESS
    );

    let mut convolution_history = AlignedLarge([0; 8192]);
    let mut convolution_input_pcm = AlignedLarge([0; 8192]);
    let mut convolution_ir_blocks = AlignedLarge([0; 8192]);
    let mut convolution_temp = AlignedLarge([0; 8192]);
    let mut convolution_output_pcm = AlignedLarge([0; 8192]);
    write_f32_values(convolution_input_pcm.0.as_mut_ptr() as u64, &[1.0; 256]);
    write_f32_values(convolution_ir_blocks.0.as_mut_ptr() as u64, &[1.0; 512]);
    let input_channel = ConvChannel {
        blocks: convolution_history.0.as_mut_ptr() as u64,
        sizes: 0,
    };
    let input_channels = [address(&input_channel)];
    let input_pcm = [convolution_input_pcm.0.as_mut_ptr() as u64];
    let mut convolution_input = ConvIn {
        block_size: 256,
        block_count: 1,
        channel_count: 1,
        format: 0,
        history_position: 0,
        history_count: 0,
        channels: address(&input_channels),
        pcm: address(&input_pcm),
    };
    let ir_channel = ConvChannel {
        blocks: convolution_ir_blocks.0.as_mut_ptr() as u64,
        sizes: 0,
    };
    let ir_channels = [address(&ir_channel)];
    let convolution_ir = ConvIr {
        block_size: 256,
        block_count: 1,
        channel_count: 1,
        format: 0,
        block_gains: 0,
        channels: address(&ir_channels),
    };
    let convolution_irs = [address(&convolution_ir)];
    let convolution_temps = [convolution_temp.0.as_mut_ptr() as u64];
    let convolution_outputs = [convolution_output_pcm.0.as_mut_ptr() as u64];
    let convolution_output = ConvOut {
        block_size: 256,
        channel_count: 1,
        temps: address(&convolution_temps),
        pcm: address(&convolution_outputs),
    };
    let convolution_output_structs = [address(&convolution_output)];
    assert_eq!(
        convolution(
            [
                address_mut(&mut info),
                1,
                address_mut(&mut convolution_input),
                1,
                address(&convolution_irs),
                0,
            ],
            [address(&convolution_output_structs), 0, 0, 0, 0, 0],
            true,
            false,
        ),
        SUCCESS
    );

    let engine = AcmEngine::new();
    let context = create_context(&engine);
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
    assert!(engine.execute_next_for_test());
    assert_eq!(
        engine.batch_wait(args(&[u64::from(context), u64::from(batch), 0])),
        SUCCESS
    );
    assert!(read_first_f32(fft_output.0.as_mut_ptr() as u64).abs() > 1.0e-6);
    assert!(read_first_f32(panner_output.0.as_mut_ptr() as u64).abs() > 1.0e-6);
    assert!(read_first_f32(convolution_output_pcm.0.as_mut_ptr() as u64).abs() > 1.0e-6);
    let telemetry = engine.telemetry_for_test();
    assert_eq!(telemetry.submitted, 1);
    assert_eq!(telemetry.retired, 1);
    assert_eq!(telemetry.failed, 0);
    assert_eq!(telemetry.fft, 1);
    assert_eq!(telemetry.ifft, 0);
    assert_eq!(telemetry.panner, 1);
    assert_eq!(telemetry.convolution, 1);
    assert_eq!(telemetry.samples, 1024);
    assert!(telemetry.execution_ns > 0);
}
