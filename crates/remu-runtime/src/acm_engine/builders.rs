// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Checked host implementations of the imported ACM batch builders.

use crate::write_guest_bytes;

use super::abi::{AcmError, SceAcmBatchInfo, checked_span, read_pod, return_error, write_pod};
use super::batch::FftDirection;

pub(super) const RECORD_MAGIC: u32 = 0x524d_4143;
pub(super) const RECORD_VERSION: u8 = 1;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RecordKind {
    Test = 1,
    Fft = 2,
    Ifft = 3,
    Panner = 4,
}

impl RecordKind {
    pub(super) fn from_byte(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Test),
            2 => Some(Self::Fft),
            3 => Some(Self::Ifft),
            4 => Some(Self::Panner),
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

    let mut encoded = vec![0_u64; words];
    encoded[0] = record_header(
        payload.kind,
        u16::try_from(words).expect("validated record word count"),
    );
    encoded[1..1 + payload.words.len()].copy_from_slice(&payload.words);
    append_words(info_address, &encoded)
}

fn append_words(info_address: u64, words: &[u64]) -> Result<(), AcmError> {
    if words.is_empty() {
        return Err(AcmError::InvalidParameter);
    }

    let info = read_pod::<SceAcmBatchInfo>(info_address).map_err(|_| AcmError::InvalidParameter)?;
    if info.buffer == 0 || !info.offset.is_multiple_of(8) || info.offset > info.size {
        return Err(AcmError::InvalidParameter);
    }

    let byte_count = words
        .len()
        .checked_mul(8)
        .ok_or(AcmError::InvalidParameter)?;
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
    for (index, word) in words.iter().enumerate() {
        let start = index * 8;
        encoded[start..start + 8].copy_from_slice(&word.to_le_bytes());
    }

    write_guest_bytes(destination, &encoded);
    let committed = SceAcmBatchInfo {
        offset: new_offset,
        ..info
    };
    write_pod(info_address, &committed).map_err(|_| AcmError::InvalidParameter)
}

pub(super) fn batch_job_notification(arguments: [u64; 6]) -> u64 {
    let info = arguments[0];
    let value = arguments[1] as u8;
    let notification = arguments[2];
    if info == 0 || value == 0 || notification == 0 {
        return return_error(AcmError::InvalidParameter);
    }
    if !notification.is_multiple_of(128) {
        return return_error(AcmError::Alignment);
    }
    if checked_span(notification, 128, 128).is_err() {
        return return_error(AcmError::InvalidParameter);
    }
    let header = u64::from(14_u8) | (u64::from(value) << 8);
    if let Err(error) = append_words(info, &[header, notification]) {
        return return_error(error);
    }
    write_guest_bytes(notification, &[0; 128]);
    0
}

pub(super) fn fft(arguments: [u64; 6], stack: [u64; 6], inverse: bool) -> u64 {
    let info = arguments[0];
    let size = arguments[1] as usize;
    let count = arguments[2] as usize;
    let input_format = arguments[3] as u32;
    let input_array = arguments[4];
    let output_format = arguments[5] as u32;
    let output_array = stack[0];
    let flags = stack[1] as u32;
    if info == 0
        || !matches!(size, 512 | 1024 | 2048)
        || count == 0
        || count > usize::from(u16::MAX)
        || input_format > 1
        || output_format > 1
        || flags & !7 != 0
    {
        return return_error(AcmError::InvalidParameter);
    }
    let Some(input_count) = count.checked_mul(1 + usize::from(flags & 2 != 0)) else {
        return return_error(AcmError::InvalidParameter);
    };
    let Some(output_count) = count.checked_mul(1 + usize::from(flags & 4 != 0)) else {
        return return_error(AcmError::InvalidParameter);
    };
    let inputs = match super::abi::read_array::<u64>(input_array, input_count) {
        Ok(pointers) => pointers,
        Err(_) => return return_error(AcmError::InvalidParameter),
    };
    let outputs = match super::abi::read_array::<u64>(output_array, output_count) {
        Ok(pointers) => pointers,
        Err(_) => return return_error(AcmError::InvalidParameter),
    };
    let direction = if inverse {
        FftDirection::Inverse
    } else {
        FftDirection::Forward
    };
    let (input_bytes, output_bytes) =
        fft_pointer_bytes(direction, size, input_format, output_format, flags);
    for pointer in &inputs {
        if let Err(error) = validate_audio_pointer(*pointer, input_bytes) {
            return return_error(error);
        }
    }
    for pointer in &outputs {
        if let Err(error) = validate_audio_pointer(*pointer, output_bytes) {
            return return_error(error);
        }
    }

    let output_extra = flags & 4 != 0 || (inverse && flags & 1 != 0);
    let words_per_job = 7 + usize::from(flags & 2 != 0) + usize::from(output_extra);
    let Some(total_words) = count
        .checked_mul(words_per_job)
        .and_then(|words| words.checked_add(2))
    else {
        return return_error(AcmError::InvalidParameter);
    };
    let config = size as u64
        | ((count as u64) << 16)
        | (u64::from(input_format) << 32)
        | (u64::from(output_format) << 40)
        | (u64::from(flags) << 48);
    let mut payload = Vec::with_capacity(total_words - 1);
    payload.push(config);
    let input_stride = 1 + usize::from(flags & 2 != 0);
    let output_stride = 1 + usize::from(flags & 4 != 0);
    for index in 0..count {
        payload.extend_from_slice(&[
            inputs[index * input_stride],
            outputs[index * output_stride],
            input_bytes as u64,
            output_bytes as u64,
            index as u64,
            0,
            0,
        ]);
        if input_stride == 2 {
            payload.push(inputs[index * input_stride + 1]);
        }
        if output_stride == 2 {
            payload.push(outputs[index * output_stride + 1]);
        } else if inverse && flags & 1 != 0 {
            payload.push(0);
        }
    }
    let payload = RecordPayload {
        kind: if inverse {
            RecordKind::Ifft
        } else {
            RecordKind::Fft
        },
        words: payload,
    };
    append_record(info, total_words, payload).map_or_else(return_error, |()| 0)
}

fn fft_pointer_bytes(
    direction: FftDirection,
    size: usize,
    input_format: u32,
    output_format: u32,
    flags: u32,
) -> (usize, usize) {
    let input_values = match direction {
        FftDirection::Forward if flags & 1 != 0 => size / 2,
        _ => size,
    };
    let output_values = match direction {
        FftDirection::Inverse if flags & 1 != 0 => size / 2,
        _ => size,
    };
    let input_parts = 1 + usize::from(flags & 2 != 0);
    let output_parts = 1 + usize::from(flags & 4 != 0);
    (
        input_values * if input_format == 0 { 4 } else { 2 } / input_parts,
        output_values * if output_format == 0 { 4 } else { 2 } / output_parts,
    )
}

fn validate_audio_pointer(address: u64, bytes: usize) -> Result<(), AcmError> {
    if !address.is_multiple_of(128) {
        return Err(AcmError::Alignment);
    }
    checked_span(address, bytes, 128)
        .map(|_| ())
        .map_err(|_| AcmError::InvalidParameter)
}

pub(super) fn panner(arguments: [u64; 6], stack: [u64; 6]) -> u64 {
    let info = arguments[0];
    let input_count = arguments[1] as usize;
    let biquad_count = arguments[3] as usize;
    let update_log2 = arguments[4] as usize;
    let output_count = arguments[5] as usize;
    if info == 0
        || input_count == 0
        || input_count > usize::from(u16::MAX)
        || biquad_count > 2
        || update_log2 > 1
        || !(1..=36).contains(&output_count)
    {
        return return_error(AcmError::InvalidParameter);
    }
    let arrays = match panner_arrays(
        input_count,
        output_count,
        arguments[2],
        stack[0],
        stack[1],
        stack[2],
        stack[3],
    ) {
        Ok(arrays) => arrays,
        Err(error) => return return_error(error),
    };
    let parameter_bytes = match panner_parameter_bytes(biquad_count, update_log2, output_count) {
        Some(bytes) => bytes,
        None => return return_error(AcmError::InvalidParameter),
    };
    let state_bytes = match panner_state_bytes(biquad_count, output_count) {
        Some(bytes) => bytes,
        None => return return_error(AcmError::InvalidParameter),
    };
    for pointer in &arrays.inputs {
        if let Err(error) = validate_audio_pointer(*pointer, 256 * 4) {
            return return_error(error);
        }
    }
    for pointer in &arrays.parameters {
        if let Err(error) = validate_audio_pointer(*pointer, parameter_bytes) {
            return return_error(error);
        }
    }
    for pointer in &arrays.states {
        if let Err(error) = validate_audio_pointer(*pointer, state_bytes) {
            return return_error(error);
        }
    }
    for pointer in arrays.initial_outputs.iter().flatten() {
        if let Err(error) = validate_audio_pointer(*pointer, 256 * 4) {
            return return_error(error);
        }
    }
    for pointer in &arrays.outputs {
        if let Err(error) = validate_audio_pointer(*pointer, 256 * 4) {
            return return_error(error);
        }
    }

    let config = input_count as u64
        | ((biquad_count as u64) << 16)
        | ((update_log2 as u64) << 24)
        | ((output_count as u64) << 32);
    let mut words = vec![config, arguments[2], stack[0], stack[1], stack[2], stack[3]];
    words.resize(19, 0);
    append_record(
        info,
        20,
        RecordPayload {
            kind: RecordKind::Panner,
            words,
        },
    )
    .map_or_else(return_error, |()| 0)
}

pub(super) struct PannerArrays {
    pub(super) inputs: Vec<u64>,
    pub(super) parameters: Vec<u64>,
    pub(super) states: Vec<u64>,
    pub(super) initial_outputs: Vec<Option<u64>>,
    pub(super) outputs: Vec<u64>,
}

pub(super) fn panner_arrays(
    input_count: usize,
    output_count: usize,
    inputs: u64,
    parameters: u64,
    states: u64,
    initial_outputs: u64,
    outputs: u64,
) -> Result<PannerArrays, AcmError> {
    let inputs =
        super::abi::read_array(inputs, input_count).map_err(|_| AcmError::InvalidParameter)?;
    let parameters =
        super::abi::read_array(parameters, input_count).map_err(|_| AcmError::InvalidParameter)?;
    let states =
        super::abi::read_array(states, input_count).map_err(|_| AcmError::InvalidParameter)?;
    let initial_outputs = if initial_outputs == 0 {
        vec![None; output_count]
    } else {
        super::abi::read_array::<u64>(initial_outputs, output_count)
            .map_err(|_| AcmError::InvalidParameter)?
            .into_iter()
            .map(|pointer| (pointer != 0).then_some(pointer))
            .collect()
    };
    let outputs =
        super::abi::read_array(outputs, output_count).map_err(|_| AcmError::InvalidParameter)?;
    Ok(PannerArrays {
        inputs,
        parameters,
        states,
        initial_outputs,
        outputs,
    })
}

pub(super) fn panner_parameter_bytes(
    biquad_count: usize,
    update_log2: usize,
    output_count: usize,
) -> Option<usize> {
    4_usize
        .checked_add(
            biquad_count
                .checked_mul(1 << update_log2)?
                .checked_mul(20)?,
        )?
        .checked_add(output_count.checked_mul(4)?)?
        .checked_add(127)
        .map(|bytes| bytes & !127)
}

pub(super) fn panner_state_bytes(biquad_count: usize, output_count: usize) -> Option<usize> {
    biquad_count
        .checked_mul(8)?
        .checked_add(output_count.checked_mul(4)?)?
        .checked_add(127)
        .map(|bytes| bytes & !127)
}

#[cfg(test)]
pub(super) fn append_test_record(info_address: u64, words: usize) -> Result<(), AcmError> {
    append_record(info_address, words, RecordPayload::test(words))
}

#[cfg(test)]
pub(super) fn append_tagged_test_record(info_address: u64, tag: u64) -> Result<(), AcmError> {
    append_record(info_address, 2, RecordPayload::tagged_test(tag))
}
