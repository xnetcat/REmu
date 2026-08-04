// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;
use crate::sce_kernel_error;

const SUCCESS: u64 = 0;

/// Packs a leading argument list into the six-register form a dispatch receives.
fn args(values: &[u64]) -> [u64; 6] {
    let mut arguments = [0_u64; 6];
    arguments[..values.len()].copy_from_slice(values);
    arguments
}

fn error(code: u32) -> u64 {
    sce_kernel_error(code)
}

fn address<T>(value: &T) -> u64 {
    std::ptr::from_ref(value) as u64
}

fn address_mut<T>(value: &mut T) -> u64 {
    std::ptr::from_mut(value) as u64
}

/// Creates a context on `engine` and returns its identifier.
fn create_context(engine: &AjmEngine) -> u32 {
    let mut context = 0_u32;
    assert_eq!(
        engine.initialize(args(&[0, address_mut(&mut context)])),
        SUCCESS
    );
    assert_ne!(context, 0);
    context
}

/// Registers `codec` and creates one instance of it with `flags`.
fn create_instance(engine: &AjmEngine, context: u32, codec: u32, flags: u64) -> u32 {
    assert_eq!(
        engine.module_register(args(&[u64::from(context), u64::from(codec)])),
        SUCCESS
    );
    let mut instance = 0_u32;
    assert_eq!(
        engine.instance_create(args(&[
            u64::from(context),
            u64::from(codec),
            flags,
            address_mut(&mut instance)
        ])),
        SUCCESS
    );
    assert_ne!(instance, 0);
    instance
}

/// An arena plus its cursor, with LPCM initialize parameters the jobs reference.
struct BatchFixture {
    arena: Vec<u8>,
    info: SceAjmBatchInfo,
    lpcm_parameters: [u32; 3],
}

impl BatchFixture {
    fn new() -> Self {
        let mut fixture = Self {
            arena: vec![0_u8; 0x400],
            info: SceAjmBatchInfo::default(),
            lpcm_parameters: [0, 2, 48_000], // S16, stereo, 48 kHz (lpcm_decoder.h:66-71)
        };
        assert_eq!(
            batch_initialize(args(&[
                fixture.arena.as_ptr() as u64,
                fixture.arena.len() as u64,
                address_mut(&mut fixture.info),
            ])),
            SUCCESS
        );
        fixture
    }

    fn info_address(&mut self) -> u64 {
        address_mut(&mut self.info)
    }

    fn append_initialize(&mut self, instance: u32, result: &mut [u8; 8]) {
        let parameters = address(&self.lpcm_parameters);
        assert_eq!(
            batch_job_initialize(args(&[
                self.info_address(),
                u64::from(instance),
                parameters,
                12,
                result.as_mut_ptr() as u64,
            ])),
            SUCCESS
        );
    }

    fn append_decode(
        &mut self,
        instance: u32,
        input: &[u8],
        output: &mut [u8],
        result: &mut [u8; 32],
    ) {
        let info = self.info_address();
        assert_eq!(
            batch_job_decode(
                args(&[
                    info,
                    u64::from(instance),
                    input.as_ptr() as u64,
                    input.len() as u64,
                    output.as_mut_ptr() as u64,
                    output.len() as u64,
                ]),
                args(&[result.as_mut_ptr() as u64]),
            ),
            SUCCESS
        );
    }

    fn start(&mut self, engine: &AjmEngine, context: u32) -> Result<u32, (u64, SceAjmBatchError)> {
        let mut batch = 0_u32;
        let mut report = SceAjmBatchError::default();
        let info = self.info_address();
        let status = engine.batch_start(args(&[
            u64::from(context),
            info,
            40, // SCE_AJM_PRIORITY_GAME_DEFAULT, ajm.h:168
            address_mut(&mut report),
            address_mut(&mut batch),
        ]));
        if status == SUCCESS {
            assert_ne!(batch, 0, "a started batch has a live identifier");
            Ok(batch)
        } else {
            Err((status, report))
        }
    }
}

/// A decode result's stream sideband (`SceAjmDecodeResult`, sideband.h:661-666).
fn decode_result_fields(result: &[u8; 32]) -> (i32, i32, i32, u64, u32) {
    (
        i32::from_le_bytes(result[0..4].try_into().unwrap()),
        i32::from_le_bytes(result[8..12].try_into().unwrap()),
        i32::from_le_bytes(result[12..16].try_into().unwrap()),
        u64::from_le_bytes(result[16..24].try_into().unwrap()),
        u32::from_le_bytes(result[24..28].try_into().unwrap()),
    )
}

#[test]
fn initialize_validates_the_expansion_flag_and_finalize_tears_the_context_down() {
    let engine = AjmEngine::new();
    let mut context = 0_u32;
    // Expansion sizes are a multiple of 2 up to 32 MiB in bits 0-6 (ajm.h:81, ajm.h:75);
    // the high dword carries a revision tag some titles set.
    assert_eq!(
        engine.initialize(args(&[0x3_0000_0008, address_mut(&mut context)])),
        SUCCESS
    );
    assert_ne!(context, 0);
    assert_eq!(
        engine.initialize(args(&[3, address_mut(&mut context)])),
        error(SCE_AJM_ERROR_INVALID_PARAMETER),
        "an odd expansion size is refused"
    );
    assert_eq!(
        engine.initialize(args(&[1 << 8, address_mut(&mut context)])),
        error(SCE_AJM_ERROR_INVALID_PARAMETER),
        "undefined low-dword flag bits are refused"
    );
    assert_eq!(
        engine.initialize(args(&[0, 0])),
        error(SCE_AJM_ERROR_INVALID_PARAMETER)
    );
    assert_eq!(engine.finalize(args(&[u64::from(context)])), SUCCESS);
    assert_eq!(
        engine.finalize(args(&[u64::from(context)])),
        error(SCE_AJM_ERROR_INVALID_CONTEXT)
    );
}

#[test]
fn instance_creation_validates_codec_type_and_registration_and_records_flags() {
    let engine = AjmEngine::new();
    let context = create_context(&engine);
    let mut instance = 0_u32;
    // 7 is not a codec type any SDK header defines.
    assert_eq!(
        engine.module_register(args(&[u64::from(context), 7])),
        error(SCE_AJM_ERROR_INVALID_PARAMETER)
    );
    assert_eq!(
        engine.instance_create(args(&[
            u64::from(context),
            7,
            0,
            address_mut(&mut instance)
        ])),
        error(SCE_AJM_ERROR_INVALID_PARAMETER)
    );
    assert_eq!(
        engine.instance_create(args(&[
            u64::from(context),
            u64::from(AJM_CODEC_DEC_LPCM),
            0,
            address_mut(&mut instance)
        ])),
        error(SCE_AJM_ERROR_CODEC_NOT_REGISTERED)
    );
    // Maximum channels in bits 0-6, output format in bits 7-9 (sideband.h:904-908).
    let flags = 2 | (2 << 7);
    let instance = create_instance(&engine, context, AJM_CODEC_DEC_LPCM, flags);
    {
        let state = engine.lock();
        let recorded = &state.contexts[&context].instances[&instance];
        assert_eq!(recorded.codec, AJM_CODEC_DEC_LPCM);
        assert_eq!(recorded.flags, flags);
        assert_eq!(recorded.output_encoding(), 2);
    }
    assert_eq!(
        engine.instance_destroy(args(&[u64::from(context), u64::from(instance)])),
        SUCCESS
    );
    assert_eq!(
        engine.instance_destroy(args(&[u64::from(context), u64::from(instance)])),
        error(SCE_AJM_ERROR_INVALID_INSTANCE)
    );
    assert_eq!(
        engine.module_register(args(&[u64::from(context), u64::from(AJM_CODEC_DEC_LPCM)])),
        error(SCE_AJM_ERROR_CODEC_ALREADY_REGISTERED)
    );
    assert_eq!(
        engine.module_unregister(args(&[u64::from(context), u64::from(AJM_CODEC_DEC_LPCM)])),
        SUCCESS
    );
    assert_eq!(
        engine.module_unregister(args(&[u64::from(context), u64::from(AJM_CODEC_DEC_LPCM)])),
        error(SCE_AJM_ERROR_CODEC_NOT_REGISTERED)
    );
}

#[test]
fn private_codec_14_has_a_real_module_lifecycle_without_fabricating_decode_support() {
    let engine = AjmEngine::new();
    let context = create_context(&engine);
    assert_eq!(
        engine.module_register(args(&[u64::from(context), u64::from(AJM_CODEC_PRIVATE_14)])),
        SUCCESS,
        "production PS5 middleware registers private AJM codec 14"
    );
    assert!(
        host_codec(AJM_CODEC_PRIVATE_14).is_none(),
        "module registration must not invent a decoder implementation"
    );
    assert_eq!(
        engine.module_unregister(args(&[u64::from(context), u64::from(AJM_CODEC_PRIVATE_14)])),
        SUCCESS
    );
}

#[test]
fn batch_initialize_writes_only_the_five_qword_cursor() {
    let batch = [0x5a_u8; 0x708];
    let original_batch = batch;
    let mut info_storage = [0xa5_u8; 0x48];
    let info_address = info_storage[0x10..0x38].as_mut_ptr() as u64;
    assert_eq!(
        batch_initialize(args(&[
            batch.as_ptr() as u64,
            batch.len() as u64,
            info_address
        ])),
        SUCCESS
    );
    assert_eq!(
        batch, original_batch,
        "the caller-owned batch arena is preserved"
    );
    assert_eq!(&info_storage[..0x10], &[0xa5; 0x10]);
    assert_eq!(&info_storage[0x38..], &[0xa5; 0x10]);
    // SAFETY: info_address points at the initialized 0x28-byte subrange above.
    let info = unsafe { std::ptr::read_unaligned(info_address as *const SceAjmBatchInfo) };
    assert_eq!(
        info,
        SceAjmBatchInfo {
            buffer: batch.as_ptr() as u64,
            offset: 0,
            size: batch.len() as u64,
            last_good_job: 0,
            last_good_job_return_address: 0,
        }
    );
    assert_eq!(
        batch_initialize(args(&[batch.as_ptr() as u64, batch.len() as u64, 0])),
        error(SCE_AJM_ERROR_INVALID_PARAMETER)
    );
}

#[test]
fn an_lpcm_decode_batch_runs_end_to_end_and_reports_real_stream_totals() {
    let engine = AjmEngine::new();
    let context = create_context(&engine);
    let instance = create_instance(&engine, context, AJM_CODEC_DEC_LPCM, 2);

    let input: Vec<u8> = (0..240_i16).flat_map(i16::to_le_bytes).collect();
    let mut output = vec![0_u8; input.len()];
    let mut initialize_result = [0xff_u8; 8];
    let mut decode_result = [0xff_u8; 32];
    let mut fixture = BatchFixture::new();
    fixture.append_initialize(instance, &mut initialize_result);
    fixture.append_decode(instance, &input, &mut output, &mut decode_result);
    let batch = fixture
        .start(&engine, context)
        .expect("a well-formed batch starts");

    assert!(engine.execute_next_queued_batch());
    assert!(!engine.execute_next_queued_batch(), "one batch was queued");
    assert_eq!(
        engine.batch_wait(args(&[
            u64::from(context),
            u64::from(batch),
            u64::from(SCE_AJM_WAIT_INFINITE),
            0
        ])),
        SUCCESS
    );
    assert_eq!(output, input, "LPCM passthrough copies the PCM bitstream");
    assert_eq!(
        i32::from_le_bytes(initialize_result[0..4].try_into().unwrap()),
        0,
        "initialization succeeded"
    );
    let (result, consumed, produced, total_samples, frames) = decode_result_fields(&decode_result);
    assert_eq!(result, 0);
    assert_eq!(consumed, i32::try_from(input.len()).unwrap());
    assert_eq!(produced, i32::try_from(input.len()).unwrap());
    // 240 stereo S16 samples are 120 frames.
    assert_eq!(total_samples, 120);
    assert_eq!(frames, 120);
    assert_eq!(
        engine.batch_wait(args(&[
            u64::from(context),
            u64::from(batch),
            u64::from(SCE_AJM_WAIT_INFINITE),
            0
        ])),
        error(SCE_AJM_ERROR_INVALID_BATCH),
        "a retired batch is disposed by its wait"
    );
}

#[test]
fn lpcm_converts_between_the_declared_input_and_requested_output_encodings() {
    let engine = AjmEngine::new();
    let context = create_context(&engine);
    // Output format FLOAT (sideband.h:522) requested in the format flag bits.
    let instance = create_instance(&engine, context, AJM_CODEC_DEC_LPCM, 1 | (2 << 7));

    let input: Vec<u8> = [i16::MAX, 0, i16::MIN / 2]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let mut output = vec![0_u8; 3 * 4];
    let mut initialize_result = [0_u8; 8];
    let mut decode_result = [0_u8; 32];
    let mut fixture = BatchFixture::new();
    fixture.lpcm_parameters = [0, 1, 48_000]; // S16 mono in
    fixture.append_initialize(instance, &mut initialize_result);
    fixture.append_decode(instance, &input, &mut output, &mut decode_result);
    let batch = fixture
        .start(&engine, context)
        .expect("a well-formed batch starts");
    assert!(engine.execute_next_queued_batch());
    assert_eq!(
        engine.batch_wait(args(&[u64::from(context), u64::from(batch), 0, 0])),
        SUCCESS
    );
    let samples: Vec<f32> = output
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    assert!((samples[0] - 1.0).abs() < 1e-4, "peak S16 maps to 1.0");
    assert!(samples[1].abs() < 1e-6);
    assert!((samples[2] + 0.5).abs() < 1e-3);
    let (result, consumed, produced, ..) = decode_result_fields(&decode_result);
    assert_eq!(result, 0);
    assert_eq!(consumed, 6, "three S16 samples in");
    assert_eq!(produced, 12, "three f32 samples out");
}

#[test]
fn a_job_for_a_codec_without_a_host_implementation_fails_the_whole_batch_loudly() {
    let engine = AjmEngine::new();
    let context = create_context(&engine);
    // Opus (P6 deferred — libopus vendor conflicts with the AJM-only build budget) registers
    // and creates instances as hardware would...
    let instance = create_instance(&engine, context, AJM_CODEC_DEC_OPUS, 2);

    let input = [0_u8; 64];
    let mut output = [0_u8; 256];
    let mut decode_result = [0_u8; 32];
    let mut fixture = BatchFixture::new();
    fixture.append_decode(instance, &input, &mut output, &mut decode_result);
    let job_address = fixture.arena.as_ptr() as u64;
    // ...but a batch that needs the decoder this host does not implement fails whole,
    // with the specific code in the caller's report — never a silently retired batch.
    let (status, report) = fixture
        .start(&engine, context)
        .expect_err("the batch must be refused");
    assert_eq!(status, error(SCE_AJM_ERROR_MALFORMED_BATCH));
    assert_eq!(
        report.error_code,
        SCE_AJM_ERROR_CODEC_NOT_SUPPORTED.cast_signed()
    );
    assert_eq!(report.job_address, job_address);
    assert_eq!(report.command_offset, 0);
    assert_eq!(
        engine.codec_job_count(AJM_CODEC_DEC_OPUS),
        1,
        "demand telemetry counts the refused job"
    );
    assert!(
        output.iter().all(|&byte| byte == 0),
        "no output was invented"
    );
}

#[test]
fn a_scribbled_arena_fails_batch_start_with_invalid_opcode() {
    let engine = AjmEngine::new();
    let context = create_context(&engine);
    let mut fixture = BatchFixture::new();
    fixture.arena[..64].fill(0x5a);
    fixture.info.offset = 64;
    let (status, report) = fixture
        .start(&engine, context)
        .expect_err("a corrupt arena must be refused");
    assert_eq!(status, error(SCE_AJM_ERROR_MALFORMED_BATCH));
    assert_eq!(
        report.error_code,
        SCE_AJM_ERROR_INVALID_OPCODE.cast_signed()
    );
}

#[test]
fn batch_wait_polls_blocks_and_times_out_against_pending_work() {
    let engine = AjmEngine::new();
    let context = create_context(&engine);
    let instance = create_instance(&engine, context, AJM_CODEC_DEC_LPCM, 2);
    let mut initialize_result = [0_u8; 8];
    let mut fixture = BatchFixture::new();
    fixture.append_initialize(instance, &mut initialize_result);
    let batch = fixture.start(&engine, context).expect("the batch starts");

    // No worker has retired the batch: a poll and a finite timeout both report progress.
    assert_eq!(
        engine.batch_wait(args(&[u64::from(context), u64::from(batch), 0, 0])),
        error(SCE_AJM_ERROR_IN_PROGRESS)
    );
    assert_eq!(
        engine.batch_wait(args(&[u64::from(context), u64::from(batch), 20, 0])),
        error(SCE_AJM_ERROR_IN_PROGRESS)
    );
    assert_eq!(
        engine.batch_wait(args(&[u64::from(context), 0x7777, 0, 0])),
        error(SCE_AJM_ERROR_INVALID_BATCH)
    );
    assert_eq!(
        engine.batch_wait(args(&[0x7777, u64::from(batch), 0, 0])),
        error(SCE_AJM_ERROR_INVALID_CONTEXT)
    );
    // An infinite wait parks until another thread retires the batch.
    std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(20));
            assert!(engine.execute_next_queued_batch());
        });
        assert_eq!(
            engine.batch_wait(args(&[
                u64::from(context),
                u64::from(batch),
                u64::from(SCE_AJM_WAIT_INFINITE),
                0
            ])),
            SUCCESS
        );
    });
}

#[test]
fn a_second_waiter_on_one_batch_reports_busy() {
    let engine = AjmEngine::new();
    let context = create_context(&engine);
    let instance = create_instance(&engine, context, AJM_CODEC_DEC_LPCM, 2);
    let mut initialize_result = [0_u8; 8];
    let mut fixture = BatchFixture::new();
    fixture.append_initialize(instance, &mut initialize_result);
    let batch = fixture.start(&engine, context).expect("the batch starts");
    engine
        .lock()
        .batches
        .get_mut(&batch)
        .expect("the batch is live")
        .waiter_active = true;
    assert_eq!(
        engine.batch_wait(args(&[u64::from(context), u64::from(batch), 0, 0])),
        error(SCE_AJM_ERROR_BUSY)
    );
}

#[test]
fn cancelling_a_pending_batch_retires_it_as_cancelled() {
    let engine = AjmEngine::new();
    let context = create_context(&engine);
    let instance = create_instance(&engine, context, AJM_CODEC_DEC_LPCM, 2);
    let mut initialize_result = [0xee_u8; 8];
    let mut fixture = BatchFixture::new();
    fixture.append_initialize(instance, &mut initialize_result);
    let batch = fixture.start(&engine, context).expect("the batch starts");
    assert_eq!(
        engine.batch_cancel(args(&[u64::from(context), u64::from(batch)])),
        SUCCESS
    );
    assert!(
        !engine.execute_next_queued_batch(),
        "a cancelled batch leaves the work queue"
    );
    assert_eq!(
        engine.batch_wait(args(&[u64::from(context), u64::from(batch), 0, 0])),
        error(SCE_AJM_ERROR_CANCELLED)
    );
    assert_eq!(
        initialize_result, [0xee_u8; 8],
        "none of a cancelled batch's output is touched"
    );
    assert_eq!(
        engine.batch_cancel(args(&[u64::from(context), u64::from(batch)])),
        error(SCE_AJM_ERROR_INVALID_BATCH),
        "the wait disposed the cancelled batch"
    );
}

#[test]
fn statistics_jobs_report_engine_accounting_not_invented_decode() {
    let engine = AjmEngine::new();
    let context = create_context(&engine);
    let instance = create_instance(&engine, context, AJM_CODEC_DEC_LPCM, 2);

    let input: Vec<u8> = (0..64_i16).flat_map(i16::to_le_bytes).collect();
    let mut output = vec![0_u8; input.len()];
    let mut initialize_result = [0_u8; 8];
    let mut decode_result = [0_u8; 32];
    let mut statistics = [0xff_u8; 48];
    let mut fixture = BatchFixture::new();
    fixture.append_initialize(instance, &mut initialize_result);
    fixture.append_decode(instance, &input, &mut output, &mut decode_result);
    let info = fixture.info_address();
    assert_eq!(
        batch_job_get_statistics(args(&[info, statistics.as_mut_ptr() as u64])),
        SUCCESS
    );
    assert_eq!(
        statistics, [0_u8; 48],
        "the statistics result is cleared at job creation"
    );
    let arena_bytes = fixture.info.offset;
    let batch = fixture.start(&engine, context).expect("the batch starts");
    assert!(engine.execute_next_queued_batch());
    assert_eq!(
        engine.batch_wait(args(&[u64::from(context), u64::from(batch), 0, 0])),
        SUCCESS
    );
    let field =
        |offset: usize| u32::from_le_bytes(statistics[offset..offset + 4].try_into().unwrap());
    assert_eq!(field(0), 0, "the statistics job itself succeeded");
    let usage_batch = f32::from_le_bytes(statistics[8..12].try_into().unwrap());
    assert!(
        usage_batch >= 0.0 && usage_batch.is_finite(),
        "the batch usage is the real elapsed execution time"
    );
    let instance_free = field(24);
    assert_eq!(
        instance_free,
        AJM_INSTANCE_MEMORY_BUDGET - AJM_INSTANCE_NOMINAL_SIZE,
        "one live instance is charged against the engine's real budget"
    );
    assert_eq!(
        u64::from(field(32)),
        arena_bytes,
        "the batch arena size is real"
    );
    assert_eq!(field(36), u32::try_from(input.len()).unwrap());
    assert_eq!(field(40), u32::try_from(output.len()).unwrap());
}

#[test]
fn a_split_decode_gathers_input_segments_and_scatters_output() {
    let engine = AjmEngine::new();
    let context = create_context(&engine);
    let instance = create_instance(&engine, context, AJM_CODEC_DEC_LPCM, 2);

    let bytes: Vec<u8> = (0..120_i16).flat_map(i16::to_le_bytes).collect();
    let (first, second) = bytes.split_at(100);
    let mut output_front = vec![0_u8; 96];
    let mut output_back = vec![0_u8; bytes.len() - 96];
    let inputs = [
        [first.as_ptr() as u64, first.len() as u64],
        [second.as_ptr() as u64, second.len() as u64],
    ];
    let outputs = [
        [output_front.as_mut_ptr() as u64, output_front.len() as u64],
        [output_back.as_mut_ptr() as u64, output_back.len() as u64],
    ];
    let mut initialize_result = [0_u8; 8];
    let mut decode_result = [0_u8; 32];
    let mut fixture = BatchFixture::new();
    fixture.append_initialize(instance, &mut initialize_result);
    let info = fixture.info_address();
    assert_eq!(
        batch_job_decode_split(
            args(&[
                info,
                u64::from(instance),
                inputs.as_ptr() as u64,
                2,
                outputs.as_ptr() as u64,
                2,
            ]),
            args(&[decode_result.as_mut_ptr() as u64]),
        ),
        SUCCESS
    );
    let batch = fixture.start(&engine, context).expect("the batch starts");
    assert!(engine.execute_next_queued_batch());
    assert_eq!(
        engine.batch_wait(args(&[u64::from(context), u64::from(batch), 0, 0])),
        SUCCESS
    );
    let mut gathered = output_front.clone();
    gathered.extend_from_slice(&output_back);
    assert_eq!(
        gathered, bytes,
        "input segments concatenate across output segments"
    );
    let (result, consumed, produced, ..) = decode_result_fields(&decode_result);
    assert_eq!(result, 0);
    assert_eq!(consumed, i32::try_from(bytes.len()).unwrap());
    assert_eq!(produced, i32::try_from(bytes.len()).unwrap());
}

#[test]
fn batch_job_run_requires_room_for_the_sideband_result() {
    let mut fixture = BatchFixture::new();
    let info = fixture.info_address();
    let sideband = [0_u8; 4];
    assert_eq!(
        batch_job_run(
            args(&[info, 0x4001, 0, 0, 0, 0]),
            args(&[0, sideband.as_ptr() as u64, sideband.len() as u64]),
        ),
        error(SCE_AJM_ERROR_INVALID_PARAMETER),
        "the sideband output must hold at least a SceAjmSidebandResult (ajm.h:1435-1437)"
    );
    assert_eq!(fixture.info.offset, 0, "no job was appended");
}

#[test]
fn an_arena_too_small_for_another_job_reports_job_creation() {
    let arena = [0_u8; 64];
    let mut info = SceAjmBatchInfo::default();
    assert_eq!(
        batch_initialize(args(&[
            arena.as_ptr() as u64,
            arena.len() as u64,
            address_mut(&mut info)
        ])),
        SUCCESS
    );
    let mut result = [0_u8; 8];
    assert_eq!(
        batch_job_clear_context(args(&[
            address_mut(&mut info),
            0x4001,
            result.as_mut_ptr() as u64
        ])),
        SUCCESS
    );
    assert_eq!(
        batch_job_clear_context(args(&[
            address_mut(&mut info),
            0x4001,
            result.as_mut_ptr() as u64
        ])),
        error(SCE_AJM_ERROR_JOB_CREATION),
        "a 64-byte arena holds one 48-byte control job"
    );
    assert_eq!(info.offset, 48);
    assert_eq!(info.last_good_job, arena.as_ptr() as u64);
}

#[test]
fn published_coprocessor_memory_must_be_page_aligned_and_owned_by_a_live_context() {
    let engine = AjmEngine::new();
    let context = create_context(&engine);
    let region = 0x1_0000_0000_u64;
    assert_eq!(
        engine.memory_register(args(&[u64::from(context), region + 1, 1])),
        error(SCE_AJM_ERROR_INVALID_PARAMETER)
    );
    assert_eq!(
        engine.memory_register(args(&[u64::from(context), region, 0])),
        error(SCE_AJM_ERROR_INVALID_PARAMETER)
    );
    assert_eq!(
        engine.memory_register(args(&[0, region, 1])),
        error(SCE_AJM_ERROR_INVALID_CONTEXT)
    );
    assert_eq!(
        engine.memory_register(args(&[u64::from(context), region, 2])),
        SUCCESS
    );
    assert_eq!(
        engine.memory_register(args(&[
            u64::from(context),
            region + AJM_MEMORY_PAGE_SIZE,
            1
        ])),
        error(SCE_AJM_ERROR_INVALID_PARAMETER),
        "a region inside one already published must be refused"
    );
    assert_eq!(
        engine.memory_register(args(&[
            u64::from(context),
            region + 4 * AJM_MEMORY_PAGE_SIZE,
            AJM_MEMORY_PAGE_LIMIT
        ])),
        error(SCE_AJM_ERROR_OUT_OF_RESOURCES)
    );
    assert_eq!(
        engine.memory_unregister(args(&[u64::from(context), region + AJM_MEMORY_PAGE_SIZE])),
        error(SCE_AJM_ERROR_INVALID_PARAMETER)
    );
    assert_eq!(
        engine.memory_unregister(args(&[u64::from(context), region])),
        SUCCESS
    );
    assert_eq!(
        engine.memory_unregister(args(&[u64::from(context), region])),
        error(SCE_AJM_ERROR_INVALID_PARAMETER)
    );
}

#[test]
fn a_process_reset_drops_every_context_and_wakes_a_parked_waiter() {
    let engine = AjmEngine::new();
    let context = create_context(&engine);
    let instance = create_instance(&engine, context, AJM_CODEC_DEC_LPCM, 2);
    let mut initialize_result = [0_u8; 8];
    let mut fixture = BatchFixture::new();
    fixture.append_initialize(instance, &mut initialize_result);
    let batch = fixture.start(&engine, context).expect("the batch starts");
    std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(20));
            engine.reset();
        });
        assert_eq!(
            engine.batch_wait(args(&[
                u64::from(context),
                u64::from(batch),
                u64::from(SCE_AJM_WAIT_INFINITE),
                0
            ])),
            error(SCE_AJM_ERROR_INVALID_BATCH),
            "a waiter parked across the reset reports the batch gone rather than wedging"
        );
    });
    assert_eq!(
        engine.finalize(args(&[u64::from(context)])),
        error(SCE_AJM_ERROR_INVALID_CONTEXT),
        "the reset engine kept no context"
    );
    // The process-wide form is a no-op until a title ever starts the global engine, and safe
    // after: the next boot must not observe this test's contexts either way.
    reset_for_new_process();
}

//----------------------------------------------------------------------------
// ATRAC9 (plan P1)
//----------------------------------------------------------------------------

/// The 4-byte ATRAC9 configuration data of a real Demon's Souls asset,
/// `sound/streams/05_cutscene_sfx/scn000006_sfx.at9` (mono, 48 kHz), taken from the
/// WAVEFORMATEXTENSIBLE ATRAC9 extension. Byte 0 is `SCE_AJM_DEC_AT9_INTERNAL_ID_NORMAL`
/// (0xFE, at9_decoder.h:199).
const AT9_DS_CONFIG: [u8; 4] = [0xFE, 0x70, 0x07, 0xF0];

/// The first eight superframes (2048 bytes) of that asset's raw ATRAC9 bitstream.
const AT9_DS_FRAMES: &[u8] = include_bytes!("testdata/atrac9_demons_souls_scn000006.frames.bin");

/// The reference PCM for those superframes, decoded by **ffmpeg's independent `atrac9`
/// decoder** (8192 mono S16 samples) and sign-negated into ATRAC9/LibAtrac9 output polarity.
/// ffmpeg is a wholly separate implementation, so this is an independent oracle, not a
/// self-referential capture.
const AT9_DS_GOLDEN: &[u8] = include_bytes!("testdata/atrac9_demons_souls_scn000006.s16le.pcm");

/// A real `SceAjmDecAt9InitializeParameters` (config data + reserved word, at9_decoder.h:
/// 268-274) for the Demon's Souls asset. The caller keeps this alive: the initialize job
/// references it until the batch retires.
const AT9_DS_INIT_PARAMETERS: [u8; 8] = [
    AT9_DS_CONFIG[0],
    AT9_DS_CONFIG[1],
    AT9_DS_CONFIG[2],
    AT9_DS_CONFIG[3],
    0,
    0,
    0,
    0,
];

/// Appends an ATRAC9 initialize job referencing the caller-owned `parameters` blob.
fn append_at9_initialize(
    fixture: &mut BatchFixture,
    instance: u32,
    parameters: &[u8],
    result: &mut [u8; 8],
) {
    let info = fixture.info_address();
    assert_eq!(
        batch_job_initialize(args(&[
            info,
            u64::from(instance),
            parameters.as_ptr() as u64,
            parameters.len() as u64,
            result.as_mut_ptr() as u64,
        ])),
        SUCCESS
    );
}

#[test]
fn atrac9_decodes_a_real_superframe_stream_and_matches_the_independent_ffmpeg_golden() {
    let engine = AjmEngine::new();
    let context = create_context(&engine);
    // A default S16-output ATRAC9 instance, exactly as Demon's Souls creates one.
    let instance = create_instance(&engine, context, AJM_CODEC_DEC_AT9, 0);

    let mut output = vec![0_u8; AT9_DS_GOLDEN.len()];
    let mut initialize_result = [0xff_u8; 8];
    let mut decode_result = [0xff_u8; 32];
    let mut fixture = BatchFixture::new();
    append_at9_initialize(
        &mut fixture,
        instance,
        &AT9_DS_INIT_PARAMETERS,
        &mut initialize_result,
    );
    fixture.append_decode(instance, AT9_DS_FRAMES, &mut output, &mut decode_result);
    let batch = fixture
        .start(&engine, context)
        .expect("a well-formed ATRAC9 batch starts");
    assert!(engine.execute_next_queued_batch());
    assert_eq!(
        engine.batch_wait(args(&[
            u64::from(context),
            u64::from(batch),
            u64::from(SCE_AJM_WAIT_INFINITE),
            0
        ])),
        SUCCESS
    );

    assert_eq!(
        i32::from_le_bytes(initialize_result[0..4].try_into().unwrap()),
        0,
        "the real config data initializes the decoder"
    );
    let (result, consumed, produced, total_samples, frames) = decode_result_fields(&decode_result);
    assert_eq!(result, 0, "the decode job succeeded");
    assert_eq!(consumed, 2048, "eight 256-byte superframes were consumed");
    assert_eq!(produced, i32::try_from(AT9_DS_GOLDEN.len()).unwrap());
    // 8 superframes * 4 frames * 256 samples of mono audio.
    assert_eq!(total_samples, 8192);
    assert_eq!(
        frames, 32,
        "four frames per superframe over eight superframes"
    );

    let decoded: Vec<i16> = output
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let reference: Vec<i16> = AT9_DS_GOLDEN
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    assert_eq!(decoded.len(), reference.len());
    let max_abs_diff = decoded
        .iter()
        .zip(&reference)
        .map(|(a, b)| (i32::from(*a) - i32::from(*b)).unsigned_abs())
        .max()
        .unwrap();
    // LibAtrac9 and ffmpeg are independent ATRAC9 implementations; they agree to within one
    // LSB (sub-LSB rounding on the IMDCT), which no integration bug could survive.
    assert!(
        max_abs_diff <= 1,
        "decoded PCM disagrees with the independent ffmpeg golden by {max_abs_diff} > 1 LSB"
    );
    let peak = decoded.iter().map(|s| s.unsigned_abs()).max().unwrap();
    assert!(
        peak > 1000,
        "real audio was produced, not silence (peak {peak})"
    );
}

#[test]
fn atrac9_reports_channel_config_sample_rate_and_codec_info_in_sideband() {
    let engine = AjmEngine::new();
    let context = create_context(&engine);
    let instance = create_instance(&engine, context, AJM_CODEC_DEC_AT9, 0);

    // One superframe in, its PCM out; a run job requesting the stream, format, and ATRAC9
    // codec-info sidebands together (sideband.h:196-202, 187-193, at9_decoder.h:283-293).
    let input = &AT9_DS_FRAMES[..256];
    let mut output = vec![0_u8; 2048];
    let mut initialize_result = [0_u8; 8];
    let mut sideband = [0_u8; 128];
    let mut fixture = BatchFixture::new();
    append_at9_initialize(
        &mut fixture,
        instance,
        &AT9_DS_INIT_PARAMETERS,
        &mut initialize_result,
    );
    let flags = AJM_FLAG_SIDEBAND_STREAM | AJM_FLAG_SIDEBAND_FORMAT | AJM_FLAG_RUN_GET_CODEC_INFO;
    let info = fixture.info_address();
    assert_eq!(
        batch_job_run(
            args(&[
                info,
                u64::from(instance),
                flags,
                input.as_ptr() as u64,
                input.len() as u64,
                output.as_mut_ptr() as u64,
            ]),
            args(&[
                output.len() as u64,
                sideband.as_mut_ptr() as u64,
                sideband.len() as u64,
            ]),
        ),
        SUCCESS
    );
    let batch = fixture.start(&engine, context).expect("the batch starts");
    assert!(engine.execute_next_queued_batch());
    assert_eq!(
        engine.batch_wait(args(&[u64::from(context), u64::from(batch), 0, 0])),
        SUCCESS
    );

    let field =
        |offset: usize| u32::from_le_bytes(sideband[offset..offset + 4].try_into().unwrap());
    // result(8) then stream(16) then format(24) then codec-info(16), in section order.
    assert_eq!(field(0), 0, "the run succeeded");
    // SceAjmSidebandFormat begins at 8 + 16 = 24 (sideband.h:535-543).
    assert_eq!(field(24), 1, "mono channel count");
    assert_eq!(field(24 + 4), 0x4, "mono channel mask (FRONT_CENTER)");
    assert_eq!(field(24 + 8), 48_000, "48 kHz sample rate");
    assert_eq!(field(24 + 12), 0, "default S16 output encoding");
    // SceAjmSidebandDecAt9CodecInfo begins at 24 + 24 = 48 (at9_decoder.h:283-293).
    assert_eq!(field(48), 256, "superframe size in bytes");
    assert_eq!(field(48 + 4), 4, "frames per superframe");
    assert_eq!(field(48 + 12), 256, "samples per frame");
}

#[test]
fn atrac9_rejects_malformed_config_data_then_a_decode_reports_not_initialized() {
    let engine = AjmEngine::new();
    let context = create_context(&engine);
    let instance = create_instance(&engine, context, AJM_CODEC_DEC_AT9, 0);

    // All-zero config data is not a valid ATRAC9 configuration (LibAtrac9 ERR_BAD_CONFIG_DATA).
    let parameters = [0_u8; 8];
    let mut initialize_result = [0_u8; 8];
    let mut output = vec![0_u8; 2048];
    let mut decode_result = [0_u8; 32];
    let mut fixture = BatchFixture::new();
    let info = fixture.info_address();
    assert_eq!(
        batch_job_initialize(args(&[
            info,
            u64::from(instance),
            parameters.as_ptr() as u64,
            parameters.len() as u64,
            initialize_result.as_mut_ptr() as u64,
        ])),
        SUCCESS
    );
    fixture.append_decode(
        instance,
        &AT9_DS_FRAMES[..256],
        &mut output,
        &mut decode_result,
    );
    let batch = fixture.start(&engine, context).expect("the batch starts");
    assert!(engine.execute_next_queued_batch());
    assert_eq!(
        engine.batch_wait(args(&[u64::from(context), u64::from(batch), 0, 0])),
        SUCCESS
    );
    let init_result = i32::from_le_bytes(initialize_result[0..4].try_into().unwrap());
    assert!(
        init_result < 0,
        "a bad ATRAC9 config sets the fatal bit in the initialize result ({init_result:#x})"
    );
    let (decode_status, ..) = decode_result_fields(&decode_result);
    assert!(
        decode_status < 0,
        "a decode with no initialized decoder is fatal, not a fabricated success"
    );
    assert!(
        output.iter().all(|&byte| byte == 0),
        "no PCM was invented for the uninitialized decoder"
    );
}

/// The one serving path: the global engine behind [`dispatch`], with its worker pool
/// retiring batches for a blocking wait.
#[test]
fn dispatch_serves_the_whole_lifecycle_through_the_worker_pool() {
    let call = |nid: &str, arguments: [u64; 6]| {
        dispatch(nid, arguments, [0; 6]).expect("the engine owns this NID")
    };
    let mut context = 0_u32;
    assert_eq!(
        call("dl+4eHSzUu4", args(&[0, address_mut(&mut context)])),
        SUCCESS
    );
    assert_eq!(
        call(
            "Q3dyFuwGn64",
            args(&[u64::from(context), u64::from(AJM_CODEC_DEC_LPCM)])
        ),
        SUCCESS
    );
    let mut instance = 0_u32;
    assert_eq!(
        call(
            "AxoDrINp4J8",
            args(&[
                u64::from(context),
                u64::from(AJM_CODEC_DEC_LPCM),
                2,
                address_mut(&mut instance)
            ])
        ),
        SUCCESS
    );
    let input: Vec<u8> = (0..96_i16).flat_map(i16::to_le_bytes).collect();
    let mut output = vec![0_u8; input.len()];
    let mut initialize_result = [0_u8; 8];
    let mut decode_result = [0_u8; 32];
    let arena = vec![0_u8; 0x200];
    let mut info = SceAjmBatchInfo::default();
    assert_eq!(
        call(
            "MmpF1XsQiHw",
            args(&[
                arena.as_ptr() as u64,
                arena.len() as u64,
                address_mut(&mut info)
            ])
        ),
        SUCCESS
    );
    let parameters = [0_u32, 2, 48_000];
    assert_eq!(
        call(
            "ezM2OhNxzck",
            args(&[
                address_mut(&mut info),
                u64::from(instance),
                address(&parameters),
                12,
                initialize_result.as_mut_ptr() as u64
            ])
        ),
        SUCCESS
    );
    assert_eq!(
        dispatch(
            "39WxhR-ePew",
            args(&[
                address_mut(&mut info),
                u64::from(instance),
                input.as_ptr() as u64,
                input.len() as u64,
                output.as_mut_ptr() as u64,
                output.len() as u64,
            ]),
            args(&[decode_result.as_mut_ptr() as u64]),
        ),
        Some(SUCCESS)
    );
    let mut batch = 0_u32;
    assert_eq!(
        call(
            "5tOfnaClcqM",
            args(&[
                u64::from(context),
                address_mut(&mut info),
                40,
                0,
                address_mut(&mut batch)
            ])
        ),
        SUCCESS
    );
    // The workers retire the batch; the wait blocks until they do.
    assert_eq!(
        call(
            "-qLsfDAywIY",
            args(&[
                u64::from(context),
                u64::from(batch),
                u64::from(SCE_AJM_WAIT_INFINITE),
                0
            ])
        ),
        SUCCESS
    );
    assert_eq!(output, input);
    let (result, .., frames) = decode_result_fields(&decode_result);
    assert_eq!(result, 0);
    assert_eq!(frames, 48);
    assert_eq!(
        call(
            "RbLbuKv8zho",
            args(&[u64::from(context), u64::from(instance)])
        ),
        SUCCESS
    );
    assert_eq!(call("MHur6qCsUus", args(&[u64::from(context)])), SUCCESS);
}

// --- P2 MPEG4-AAC -----------------------------------------------------------------

const M4AAC_ADTS: &[u8] = include_bytes!("testdata/m4aac_sine440.adts");
const M4AAC_GOLDEN: &[u8] = include_bytes!("testdata/m4aac_sine440.s16le.pcm");
/// ADTS initialize parameters: config=ADTS (1), sampling index ignored (m4aac_decoder.h:214-220).
const M4AAC_INIT_PARAMETERS: [u8; 8] = [1, 0, 0, 0, 3, 0, 0, 0];

fn append_m4aac_initialize(
    fixture: &mut BatchFixture,
    instance: u32,
    parameters: &[u8],
    result: &mut [u8; 8],
) {
    let info = fixture.info_address();
    assert_eq!(
        batch_job_initialize(args(&[
            info,
            u64::from(instance),
            parameters.as_ptr() as u64,
            parameters.len() as u64,
            result.as_mut_ptr() as u64,
        ])),
        SUCCESS
    );
}

#[test]
fn m4aac_decodes_an_adts_stream_and_matches_the_independent_ffmpeg_golden() {
    let engine = AjmEngine::new();
    let context = create_context(&engine);
    let instance = create_instance(&engine, context, AJM_CODEC_DEC_M4AAC, 2);

    let mut output = vec![0_u8; M4AAC_GOLDEN.len()];
    let mut initialize_result = [0xff_u8; 8];
    let mut decode_result = [0xff_u8; 32];
    let mut fixture = BatchFixture::new();
    append_m4aac_initialize(
        &mut fixture,
        instance,
        &M4AAC_INIT_PARAMETERS,
        &mut initialize_result,
    );
    fixture.append_decode(instance, M4AAC_ADTS, &mut output, &mut decode_result);
    let batch = fixture
        .start(&engine, context)
        .expect("a well-formed MPEG4-AAC batch starts");
    assert!(engine.execute_next_queued_batch());
    assert_eq!(
        engine.batch_wait(args(&[
            u64::from(context),
            u64::from(batch),
            u64::from(SCE_AJM_WAIT_INFINITE),
            0
        ])),
        SUCCESS
    );

    assert_eq!(
        i32::from_le_bytes(initialize_result[0..4].try_into().unwrap()),
        0
    );
    let (result, consumed, produced, total_samples, frames) = decode_result_fields(&decode_result);
    assert_eq!(result, 0, "the decode job succeeded");
    assert_eq!(consumed, i32::try_from(M4AAC_ADTS.len()).unwrap());
    assert_eq!(produced, i32::try_from(M4AAC_GOLDEN.len()).unwrap());
    assert_eq!(total_samples as usize, M4AAC_GOLDEN.len() / 4); // stereo S16
    assert!(frames > 0);

    let decoded: Vec<i16> = output
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let reference: Vec<i16> = M4AAC_GOLDEN
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    assert_eq!(decoded.len(), reference.len());
    // Symphonia and ffmpeg are independent AAC LC implementations; allow a small absolute
    // error per sample (encoder/decoder rounding), but require a real sine peak.
    let max_abs_diff = decoded
        .iter()
        .zip(&reference)
        .map(|(a, b)| (i32::from(*a) - i32::from(*b)).unsigned_abs())
        .max()
        .unwrap();
    assert!(
        max_abs_diff <= 64,
        "decoded PCM disagrees with the independent ffmpeg golden by {max_abs_diff} > 64"
    );
    let peak = decoded.iter().map(|s| s.unsigned_abs()).max().unwrap();
    assert!(
        peak > 1000,
        "decoded audio must not be silence (peak={peak})"
    );
}

// --- P6 MP3 -----------------------------------------------------------------------

const MP3_STREAM: &[u8] = include_bytes!("testdata/mp3_sine440.mp3");
const MP3_GOLDEN: &[u8] = include_bytes!("testdata/mp3_sine440.s16le.pcm");

#[test]
fn mp3_decodes_a_stream_and_matches_the_independent_ffmpeg_golden() {
    let engine = AjmEngine::new();
    let context = create_context(&engine);
    let instance = create_instance(&engine, context, AJM_CODEC_DEC_MP3, 2);

    let mut output = vec![
        0_u8;
        MP3_GOLDEN
            .len()
            .saturating_mul(2)
            .max(MP3_GOLDEN.len() + 8192)
    ];
    let mut initialize_result = [0xff_u8; 8];
    let mut decode_result = [0xff_u8; 32];
    let mut fixture = BatchFixture::new();
    // MP3 initialize takes no parameter blob (mp3_decoder.h has no InitializeParameters).
    let info = fixture.info_address();
    assert_eq!(
        batch_job_initialize(args(&[
            info,
            u64::from(instance),
            0,
            0,
            initialize_result.as_mut_ptr() as u64,
        ])),
        SUCCESS
    );
    fixture.append_decode(instance, MP3_STREAM, &mut output, &mut decode_result);
    let batch = fixture
        .start(&engine, context)
        .expect("a well-formed MP3 batch starts");
    assert!(engine.execute_next_queued_batch());
    assert_eq!(
        engine.batch_wait(args(&[
            u64::from(context),
            u64::from(batch),
            u64::from(SCE_AJM_WAIT_INFINITE),
            0
        ])),
        SUCCESS
    );

    assert_eq!(
        i32::from_le_bytes(initialize_result[0..4].try_into().unwrap()),
        0
    );
    let (result, consumed, produced, total_samples, frames) = decode_result_fields(&decode_result);
    assert_eq!(result, 0);
    assert!(consumed > 0);
    assert!(produced > 0);
    assert!(total_samples > 0);
    assert!(frames > 0);

    let produced_usize = produced as usize;
    let decoded: Vec<i16> = output[..produced_usize]
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let reference: Vec<i16> = MP3_GOLDEN
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    // Compare the overlapping stream after accounting for decoder-specific MP3 delay.
    let max_offset = 1152 * 2;
    let (decoded_offset, reference_offset, _) = (0..=max_offset)
        .flat_map(|offset| [(offset, 0), (0, offset)])
        .filter_map(|(decoded_offset, reference_offset)| {
            let n = (decoded.len() - decoded_offset)
                .min(reference.len() - reference_offset)
                .saturating_sub(max_offset);
            (n > 1000).then(|| {
                let error = decoded[decoded_offset..decoded_offset + n]
                    .iter()
                    .zip(&reference[reference_offset..reference_offset + n])
                    .map(|(a, b)| u64::from((i32::from(*a) - i32::from(*b)).unsigned_abs()))
                    .sum::<u64>();
                (decoded_offset, reference_offset, error)
            })
        })
        .min_by_key(|(_, _, error)| *error)
        .expect("enough overlapping samples to align");
    let n = (decoded.len() - decoded_offset)
        .min(reference.len() - reference_offset)
        .saturating_sub(max_offset);
    assert!(n > 1000, "enough overlapping samples to compare");
    // The two decoders initialize their synthesis filterbanks differently. Exclude one
    // MPEG frame of startup transient after alignment and compare their steady-state PCM.
    let comparison_start = max_offset.min(n);
    let mut differences: Vec<u32> = decoded[decoded_offset + comparison_start..decoded_offset + n]
        .iter()
        .zip(&reference[reference_offset + comparison_start..reference_offset + n])
        .map(|(a, b)| (i32::from(*a) - i32::from(*b)).unsigned_abs())
        .collect();
    let mean_abs_diff = differences
        .iter()
        .map(|value| u64::from(*value))
        .sum::<u64>()
        / differences.len() as u64;
    differences.sort_unstable();
    let max_abs_diff = differences.last().copied().unwrap_or(0);
    let p99_abs_diff = differences[differences.len() * 99 / 100];
    assert!(
        mean_abs_diff <= 96 && p99_abs_diff <= 3_000,
        "decoded PCM disagrees with the independent ffmpeg golden (max {max_abs_diff}, mean {mean_abs_diff}, p99 {p99_abs_diff}, decoded offset {decoded_offset}, reference offset {reference_offset})"
    );
    let peak = decoded.iter().map(|s| s.unsigned_abs()).max().unwrap();
    assert!(
        peak > 1000,
        "decoded audio must not be silence (peak={peak})"
    );
}

// --- P3 Kraken/Oodle --------------------------------------------------------------

const KRAKEN_SRC: &[u8] = include_bytes!("testdata/kraken_uncompressed.bin");
const KRAKEN_PLAIN: &[u8] = include_bytes!("testdata/kraken_uncompressed.plain");

#[test]
fn ooz_kraken_decompresses_an_uncompressed_quantum() {
    let mut dst = vec![0_u8; KRAKEN_PLAIN.len()];
    let written = super::oodle::kraken_decompress(KRAKEN_SRC, &mut dst)
        .expect("ooz must accept the crafted uncompressed quantum");
    assert_eq!(written, KRAKEN_PLAIN.len());
    assert_eq!(&dst[..], KRAKEN_PLAIN);
}

#[test]
fn ampr_kraken_decode_stage_matches_the_two_phase_closure_signature() {
    use std::io::Write;
    // File framing = 8-byte LE uncompressed size + Kraken bitstream (ooz CLI convention).
    let mut path = std::env::temp_dir();
    path.push(format!("remu-kraken-stage-{}.bin", std::process::id()));
    {
        let mut file = std::fs::File::create(&path).unwrap();
        let size = KRAKEN_PLAIN.len() as u64;
        file.write_all(&size.to_le_bytes()).unwrap();
        file.write_all(KRAKEN_SRC).unwrap();
    }
    let file = std::fs::File::open(&path).unwrap();
    let mut guest = vec![0_u8; KRAKEN_PLAIN.len()];
    let destination = guest.as_mut_ptr() as u64;
    // The closure signature is the seam contract (ampr_api two-phase submit).
    let decode: fn(&std::fs::File, u32, u64, u64, u64, usize) -> Result<usize, u64> =
        super::oodle::ampr_kraken_decode_stage;
    let written = decode(
        &file,
        0x11,
        destination,
        KRAKEN_PLAIN.len() as u64,
        0,
        guest.len(),
    )
    .expect("decode stage succeeds");
    assert_eq!(written, KRAKEN_PLAIN.len());
    assert_eq!(&guest[..], KRAKEN_PLAIN);
    let _ = std::fs::remove_file(&path);
}
