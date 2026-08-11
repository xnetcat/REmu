// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Immutable ACM batch plans decoded from guest-owned command arenas.

use super::abi::{AcmError, AcmFormat, GuestPod, SceAcmBatchInfo, checked_span, read_array};
use super::builders::{
    RECORD_MAGIC, RECORD_VERSION, RecordKind, panner_arrays, panner_parameter_bytes,
    panner_state_bytes,
};

const COMMAND_STATISTICS: u8 = 11;
const COMMAND_NOTIFICATION: u8 = 14;
const STATISTICS_BYTES: usize = 24;
const NOTIFICATION_BYTES: usize = 128;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct BatchPlan {
    pub(super) commands: Vec<Command>,
    pub(super) notification: Option<Notification>,
}

#[derive(Clone, Debug, PartialEq)]
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
    Fft(FftJob),
    Panner(PannerJob),
    Convolution(ConvolutionJob),
    #[cfg(test)]
    TestWrite(PendingWrite),
    #[cfg(test)]
    TestFault,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FftDirection {
    Forward,
    Inverse,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct FftBuffer {
    pub(super) first: u64,
    pub(super) second: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct FftJob {
    pub(super) direction: FftDirection,
    pub(super) size: usize,
    pub(super) input_format: AcmFormat,
    pub(super) output_format: AcmFormat,
    pub(super) inputs: Vec<FftBuffer>,
    pub(super) outputs: Vec<FftBuffer>,
    pub(super) flags: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PannerJob {
    pub(super) biquad_count: usize,
    pub(super) update_log2: usize,
    pub(super) output_count: usize,
    pub(super) inputs: Vec<u64>,
    pub(super) parameters: Vec<u64>,
    pub(super) states: Vec<u64>,
    pub(super) initial_outputs: Vec<Option<u64>>,
    pub(super) outputs: Vec<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ConvolutionSharing {
    SharedInput,
    SharedIr,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ConvolutionInput {
    pub(super) address: u64,
    pub(super) block_size: usize,
    pub(super) block_count: usize,
    pub(super) format: AcmFormat,
    pub(super) history_position: usize,
    pub(super) history_count: usize,
    pub(super) history_null_count: usize,
    pub(super) channels: Vec<super::abi::ConvChannel>,
    pub(super) pcm: Vec<Option<u64>>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ConvolutionIr {
    pub(super) block_size: usize,
    pub(super) block_count: usize,
    pub(super) format: AcmFormat,
    pub(super) channels: Vec<super::abi::ConvChannel>,
    pub(super) sizes: Vec<Option<Vec<super::abi::ConvBlockSize>>>,
    pub(super) block_gains: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ConvolutionOutput {
    pub(super) address: u64,
    pub(super) block_size: usize,
    pub(super) temps: Vec<u64>,
    pub(super) pcm: Vec<u64>,
    pub(super) initial_pcm: Vec<Option<u64>>,
    pub(super) previous_gain: f32,
    pub(super) gain: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ConvolutionJob {
    pub(super) sharing: ConvolutionSharing,
    pub(super) version2: bool,
    pub(super) inputs: Vec<ConvolutionInput>,
    pub(super) irs: Vec<ConvolutionIr>,
    pub(super) gains: Vec<f32>,
    pub(super) outputs: Vec<ConvolutionOutput>,
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
                    RecordKind::Fft | RecordKind::Ifft => {
                        plan.commands.push(Command::Fft(decode_fft_record(
                            kind,
                            &words[cursor + 1..end],
                            record_words,
                        )?));
                    }
                    RecordKind::Panner => plan.commands.push(Command::Panner(
                        decode_panner_record(&words[cursor + 1..end], record_words)?,
                    )),
                    RecordKind::Convolution => {
                        plan.commands
                            .push(Command::Convolution(decode_convolution_record(
                                &words[cursor + 1..end],
                                record_words,
                            )?))
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

fn decode_fft_record(
    kind: RecordKind,
    payload: &[u64],
    record_words: usize,
) -> Result<FftJob, AcmError> {
    let config = *payload.first().ok_or(AcmError::MalformedBatch)?;
    if config >> 56 != 0 {
        return Err(AcmError::MalformedBatch);
    }
    let size = usize::from(config as u16);
    let count = usize::from((config >> 16) as u16);
    let input_format = decode_format((config >> 32) as u8)?;
    let output_format = decode_format((config >> 40) as u8)?;
    let flags = u32::from((config >> 48) as u8);
    if !matches!(size, 512 | 1024 | 2048) || count == 0 || flags & !7 != 0 {
        return Err(AcmError::MalformedBatch);
    }
    let direction = match kind {
        RecordKind::Fft => FftDirection::Forward,
        RecordKind::Ifft => FftDirection::Inverse,
        RecordKind::Test | RecordKind::Panner | RecordKind::Convolution => {
            return Err(AcmError::MalformedBatch);
        }
    };
    let split_input = flags & 2 != 0;
    let split_output = flags & 4 != 0;
    let output_extra = split_output || (direction == FftDirection::Inverse && flags & 1 != 0);
    let words_per_job = 7 + usize::from(split_input) + usize::from(output_extra);
    if record_words != 2 + count * words_per_job || payload.len() != 1 + count * words_per_job {
        return Err(AcmError::MalformedBatch);
    }

    let (input_bytes, output_bytes) =
        fft_buffer_bytes(direction, size, input_format, output_format, flags)?;
    let mut inputs = Vec::with_capacity(count);
    let mut outputs = Vec::with_capacity(count);
    let mut cursor = 1;
    for index in 0..count {
        let fixed = payload
            .get(cursor..cursor + 7)
            .ok_or(AcmError::MalformedBatch)?;
        if fixed[2] != input_bytes as u64
            || fixed[3] != output_bytes as u64
            || fixed[4] != index as u64
            || fixed[5] != 0
            || fixed[6] != 0
        {
            return Err(AcmError::MalformedBatch);
        }
        cursor += 7;
        let input_second = split_input.then(|| {
            let pointer = payload[cursor];
            cursor += 1;
            pointer
        });
        let output_second = if split_output {
            let pointer = payload[cursor];
            cursor += 1;
            Some(pointer)
        } else {
            None
        };
        if direction == FftDirection::Inverse && flags & 1 != 0 && !split_output {
            if payload[cursor] != 0 {
                return Err(AcmError::MalformedBatch);
            }
            cursor += 1;
        }
        validate_audio_buffer(fixed[0], input_bytes)?;
        validate_audio_buffer(fixed[1], output_bytes)?;
        if let Some(pointer) = input_second {
            validate_audio_buffer(pointer, input_bytes)?;
        }
        if let Some(pointer) = output_second {
            validate_audio_buffer(pointer, output_bytes)?;
        }
        inputs.push(FftBuffer {
            first: fixed[0],
            second: input_second,
        });
        outputs.push(FftBuffer {
            first: fixed[1],
            second: output_second,
        });
    }
    Ok(FftJob {
        direction,
        size,
        input_format,
        output_format,
        inputs,
        outputs,
        flags,
    })
}

fn decode_convolution_record(
    payload: &[u64],
    record_words: usize,
) -> Result<ConvolutionJob, AcmError> {
    if payload.len() + 1 != record_words
        || payload.len() < 5
        || payload[5..].iter().any(|word| *word != 0)
    {
        return Err(AcmError::MalformedBatch);
    }
    let config = payload[0];
    let sharing = if config & 1 != 0 {
        ConvolutionSharing::SharedInput
    } else {
        ConvolutionSharing::SharedIr
    };
    let version2 = config & 2 != 0;
    if config & 0xfc != 0 || config >> 56 != 0 {
        return Err(AcmError::MalformedBatch);
    }
    let block_count = (config >> 8) as u32;
    let count = usize::from((config >> 40) as u16);
    let job = super::dsp::snapshot_convolution(
        sharing,
        version2,
        block_count,
        payload[1],
        count,
        payload[2],
        payload[3],
        payload[4],
    )?;
    if super::dsp::convolution_record_words(&job)? != record_words {
        return Err(AcmError::MalformedBatch);
    }
    Ok(job)
}

fn decode_panner_record(payload: &[u64], record_words: usize) -> Result<PannerJob, AcmError> {
    if record_words != 20 || payload.len() != 19 || payload[6..].iter().any(|word| *word != 0) {
        return Err(AcmError::MalformedBatch);
    }
    let config = payload[0];
    if config >> 40 != 0 {
        return Err(AcmError::MalformedBatch);
    }
    let input_count = usize::from(config as u16);
    let biquad_count = usize::from((config >> 16) as u8);
    let update_log2 = usize::from((config >> 24) as u8);
    let output_count = usize::from((config >> 32) as u8);
    if input_count == 0 || biquad_count > 2 || update_log2 > 1 || !(1..=36).contains(&output_count)
    {
        return Err(AcmError::MalformedBatch);
    }
    let arrays = panner_arrays(
        input_count,
        output_count,
        payload[1],
        payload[2],
        payload[3],
        payload[4],
        payload[5],
    )?;
    let parameter_bytes = panner_parameter_bytes(biquad_count, update_log2, output_count)
        .ok_or(AcmError::MalformedBatch)?;
    let state_bytes =
        panner_state_bytes(biquad_count, output_count).ok_or(AcmError::MalformedBatch)?;
    for pointer in &arrays.inputs {
        validate_audio_buffer(*pointer, 256 * 4)?;
    }
    for pointer in &arrays.parameters {
        validate_audio_buffer(*pointer, parameter_bytes)?;
    }
    for pointer in &arrays.states {
        validate_audio_buffer(*pointer, state_bytes)?;
    }
    for pointer in arrays.initial_outputs.iter().flatten() {
        validate_audio_buffer(*pointer, 256 * 4)?;
    }
    for pointer in &arrays.outputs {
        validate_audio_buffer(*pointer, 256 * 4)?;
    }
    Ok(PannerJob {
        biquad_count,
        update_log2,
        output_count,
        inputs: arrays.inputs,
        parameters: arrays.parameters,
        states: arrays.states,
        initial_outputs: arrays.initial_outputs,
        outputs: arrays.outputs,
    })
}

fn decode_format(value: u8) -> Result<AcmFormat, AcmError> {
    match value {
        0 => Ok(AcmFormat::Fp32),
        1 => Ok(AcmFormat::Fp16),
        _ => Err(AcmError::MalformedBatch),
    }
}

fn fft_buffer_bytes(
    direction: FftDirection,
    size: usize,
    input_format: AcmFormat,
    output_format: AcmFormat,
    flags: u32,
) -> Result<(usize, usize), AcmError> {
    let input_parts = 1 + usize::from(flags & 2 != 0);
    let output_parts = 1 + usize::from(flags & 4 != 0);
    let (input_values, output_values) = match direction {
        FftDirection::Forward => (if flags & 1 != 0 { size / 2 } else { size }, size),
        FftDirection::Inverse => (size, if flags & 1 != 0 { size / 2 } else { size }),
    };
    let input_bytes = input_values
        .checked_mul(format_bytes(input_format))
        .and_then(|bytes| bytes.checked_div(input_parts))
        .ok_or(AcmError::MalformedBatch)?;
    let output_bytes = output_values
        .checked_mul(format_bytes(output_format))
        .and_then(|bytes| bytes.checked_div(output_parts))
        .ok_or(AcmError::MalformedBatch)?;
    Ok((input_bytes, output_bytes))
}

fn format_bytes(format: AcmFormat) -> usize {
    match format {
        AcmFormat::Fp32 => 4,
        AcmFormat::Fp16 => 2,
    }
}

fn validate_audio_buffer(address: u64, bytes: usize) -> Result<(), AcmError> {
    if !address.is_multiple_of(128) {
        return Err(AcmError::Alignment);
    }
    checked_span(address, bytes, 128).map(|_| ())
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
