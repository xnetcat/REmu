// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Partitioned convolution reverb.

use super::super::abi::{
    AcmError, AcmFormat, ConvBlockSize, ConvChannel, ConvIn, ConvInV2, ConvIr, ConvOut, ConvOutV2,
    checked_span, read_array, read_pod,
};
use super::super::batch::{
    ConvolutionInput, ConvolutionIr, ConvolutionJob, ConvolutionOutput, ConvolutionSharing,
};
use crate::read_guest_bytes;

pub(super) fn validate_channel_graph(
    input_channels: usize,
    ir_channels: usize,
    output_channels: usize,
) -> Result<(), AcmError> {
    if input_channels == 0
        || ir_channels == 0
        || output_channels != input_channels.max(ir_channels)
        || !(input_channels == 1 || ir_channels == 1 || input_channels == ir_channels)
    {
        return Err(AcmError::InvalidParameter);
    }
    Ok(())
}

pub(in crate::acm_engine) fn snapshot_job(
    sharing: ConvolutionSharing,
    version2: bool,
    block_count: u32,
    shared: u64,
    count: usize,
    tuples: u64,
    gains: u64,
    outputs: u64,
) -> Result<ConvolutionJob, AcmError> {
    if block_count != 1 || shared == 0 || count == 0 || count > usize::from(u16::MAX) {
        return Err(AcmError::InvalidParameter);
    }
    let tuple_pointers = read_array::<u64>(tuples, count).map_err(invalid)?;
    let output_pointers = read_array::<u64>(outputs, count).map_err(invalid)?;
    let gains = read_gains(gains, count)?;
    let (inputs, irs) = match sharing {
        ConvolutionSharing::SharedInput => (
            vec![snapshot_input(shared, version2)?],
            tuple_pointers
                .into_iter()
                .map(snapshot_ir)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        ConvolutionSharing::SharedIr => (
            tuple_pointers
                .into_iter()
                .map(|address| snapshot_input(address, version2))
                .collect::<Result<Vec<_>, _>>()?,
            vec![snapshot_ir(shared)?],
        ),
    };
    let outputs = output_pointers
        .into_iter()
        .map(|address| snapshot_output(address, version2))
        .collect::<Result<Vec<_>, _>>()?;
    for index in 0..count {
        let input = match sharing {
            ConvolutionSharing::SharedInput => &inputs[0],
            ConvolutionSharing::SharedIr => &inputs[index],
        };
        let ir = match sharing {
            ConvolutionSharing::SharedInput => &irs[index],
            ConvolutionSharing::SharedIr => &irs[0],
        };
        let output = &outputs[index];
        if input.block_size != ir.block_size || input.block_size != output.block_size {
            return Err(AcmError::InvalidParameter);
        }
        validate_channel_graph(input.channels.len(), ir.channels.len(), output.pcm.len())?;
    }
    Ok(ConvolutionJob {
        sharing,
        version2,
        inputs,
        irs,
        gains,
        outputs,
    })
}

pub(in crate::acm_engine) fn record_words(job: &ConvolutionJob) -> Result<usize, AcmError> {
    let input_channels = job.inputs[0].channels.len();
    let ir_channels = job.irs[0].channels.len();
    let (outer, inner) = match job.sharing {
        ConvolutionSharing::SharedInput => (input_channels, ir_channels),
        ConvolutionSharing::SharedIr => (ir_channels, input_channels),
    };
    let fft_words = 2_usize
        .checked_add(
            input_channels
                .checked_mul(7)
                .ok_or(AcmError::InvalidParameter)?,
        )
        .ok_or(AcmError::InvalidParameter)?;
    let groups = inner.checked_add(7).ok_or(AcmError::InvalidParameter)? / 8;
    fft_words
        .checked_add(2)
        .and_then(|words| words.checked_add(1))
        .and_then(|words| words.checked_add(outer.checked_mul(groups)?.checked_mul(85)?))
        .ok_or(AcmError::InvalidParameter)
}

fn snapshot_input(address: u64, version2: bool) -> Result<ConvolutionInput, AcmError> {
    let (base, null_count) = if version2 {
        let value = read_pod::<ConvInV2>(address).map_err(invalid)?;
        if value.reserved != [0; 6] {
            return Err(AcmError::InvalidParameter);
        }
        (value.base, value.history_null_count as usize)
    } else {
        (read_pod::<ConvIn>(address).map_err(invalid)?, 0)
    };
    let block_size = validate_block_size(base.block_size)?;
    let block_count = base.block_count as usize;
    let channel_count = base.channel_count as usize;
    let format = decode_format(base.format)?;
    if block_count == 0
        || channel_count == 0
        || base.history_count > base.block_count
        || (base.history_count != 0 && base.history_position >= base.block_count)
        || null_count > base.history_count as usize
    {
        return Err(AcmError::InvalidParameter);
    }
    let channels = snapshot_channels(base.channels, channel_count)?;
    let pcm = read_array::<u64>(base.pcm, channel_count)
        .map_err(invalid)?
        .into_iter()
        .map(|pointer| (pointer != 0).then_some(pointer))
        .collect::<Vec<_>>();
    let spectrum_bytes = block_size * 2 * format_bytes(format);
    for channel in &channels {
        if channel.sizes != 0 {
            return Err(AcmError::InvalidParameter);
        }
        validate_audio(channel.blocks, spectrum_bytes * block_count)?;
    }
    for pointer in pcm.iter().flatten() {
        validate_audio(*pointer, block_size * 4)?;
    }
    if !version2 && pcm.iter().any(Option::is_none) {
        return Err(AcmError::InvalidParameter);
    }
    Ok(ConvolutionInput {
        address,
        block_size,
        block_count,
        format,
        history_position: base.history_position as usize,
        history_count: base.history_count as usize,
        history_null_count: null_count,
        channels,
        pcm,
    })
}

fn snapshot_ir(address: u64) -> Result<ConvolutionIr, AcmError> {
    let base = read_pod::<ConvIr>(address).map_err(invalid)?;
    let block_size = validate_block_size(base.block_size)?;
    let block_count = base.block_count as usize;
    let channel_count = base.channel_count as usize;
    let format = decode_format(base.format)?;
    if block_count == 0 || channel_count == 0 {
        return Err(AcmError::InvalidParameter);
    }
    let channels = snapshot_channels(base.channels, channel_count)?;
    let full_block_bytes = block_size * 2 * format_bytes(format);
    let size_alignment = if format == AcmFormat::Fp32 { 256 } else { 128 };
    let mut sizes = Vec::with_capacity(channel_count);
    for channel in &channels {
        validate_audio(channel.blocks, full_block_bytes * block_count)?;
        if channel.sizes == 0 {
            sizes.push(None);
        } else {
            let values =
                read_array::<ConvBlockSize>(channel.sizes, block_count).map_err(invalid)?;
            if values.iter().any(|size| {
                size.size > size.size_memory
                    || usize::from(size.size_memory) > full_block_bytes
                    || !usize::from(size.size_memory).is_multiple_of(size_alignment)
                    || !usize::from(size.size).is_multiple_of(size_alignment)
            }) {
                return Err(AcmError::InvalidParameter);
            }
            sizes.push(Some(values));
        }
    }
    let block_gains = if base.block_gains == 0 {
        vec![1.0; block_count]
    } else {
        read_f32_array(base.block_gains, block_count)?
    };
    Ok(ConvolutionIr {
        block_size,
        block_count,
        format,
        channels,
        sizes,
        block_gains,
    })
}

fn snapshot_output(address: u64, version2: bool) -> Result<ConvolutionOutput, AcmError> {
    let (base, gain, previous_gain, initial_pointer) = if version2 {
        let value = read_pod::<ConvOutV2>(address).map_err(invalid)?;
        if value.reserved != [0; 4] {
            return Err(AcmError::InvalidParameter);
        }
        (value.base, value.gain, value.previous_gain, value.pcm_init)
    } else {
        (read_pod::<ConvOut>(address).map_err(invalid)?, 1.0, 1.0, 0)
    };
    let block_size = validate_block_size(base.block_size)?;
    let channel_count = base.channel_count as usize;
    if channel_count == 0 || !gain.is_finite() || !previous_gain.is_finite() {
        return Err(AcmError::InvalidParameter);
    }
    let temps = read_array::<u64>(base.temps, channel_count).map_err(invalid)?;
    let pcm = read_array::<u64>(base.pcm, channel_count).map_err(invalid)?;
    let initial_pcm = if initial_pointer == 0 {
        vec![None; channel_count]
    } else {
        read_array::<u64>(initial_pointer, channel_count)
            .map_err(invalid)?
            .into_iter()
            .map(|pointer| (pointer != 0).then_some(pointer))
            .collect()
    };
    for pointer in temps.iter().chain(&pcm) {
        validate_audio(*pointer, block_size * 4)?;
    }
    for pointer in initial_pcm.iter().flatten() {
        validate_audio(*pointer, block_size * 4)?;
    }
    Ok(ConvolutionOutput {
        address,
        block_size,
        temps,
        pcm,
        initial_pcm,
        previous_gain,
        gain,
    })
}

fn snapshot_channels(address: u64, count: usize) -> Result<Vec<ConvChannel>, AcmError> {
    read_array::<u64>(address, count)
        .map_err(invalid)?
        .into_iter()
        .map(|address| read_pod::<ConvChannel>(address).map_err(invalid))
        .collect()
}

fn read_gains(address: u64, count: usize) -> Result<Vec<f32>, AcmError> {
    if address == 0 {
        return Ok(vec![1.0; count]);
    }
    read_f32_array(address, count)
}

fn read_f32_array(address: u64, count: usize) -> Result<Vec<f32>, AcmError> {
    let bytes = count.checked_mul(4).ok_or(AcmError::InvalidParameter)?;
    checked_span(address, bytes, 4).map_err(invalid)?;
    let mut data = vec![0_u8; bytes];
    read_guest_bytes(address, &mut data);
    let values = data
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().expect("four-byte gain")))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(AcmError::InvalidParameter);
    }
    Ok(values)
}

fn validate_block_size(value: u32) -> Result<usize, AcmError> {
    let value = value as usize;
    matches!(value, 256 | 512 | 1024)
        .then_some(value)
        .ok_or(AcmError::InvalidParameter)
}

fn decode_format(value: u32) -> Result<AcmFormat, AcmError> {
    match value {
        0 => Ok(AcmFormat::Fp32),
        1 => Ok(AcmFormat::Fp16),
        _ => Err(AcmError::InvalidParameter),
    }
}

fn format_bytes(format: AcmFormat) -> usize {
    match format {
        AcmFormat::Fp32 => 4,
        AcmFormat::Fp16 => 2,
    }
}

fn validate_audio(address: u64, bytes: usize) -> Result<(), AcmError> {
    if !address.is_multiple_of(128) {
        return Err(AcmError::Alignment);
    }
    checked_span(address, bytes, 128)
        .map(|_| ())
        .map_err(invalid)
}

fn invalid(_: AcmError) -> AcmError {
    AcmError::InvalidParameter
}
