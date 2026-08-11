// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Partitioned convolution reverb.

use super::super::abi::{
    AcmError, AcmFormat, ConvBlockSize, ConvChannel, ConvIn, ConvInV2, ConvIr, ConvOut, ConvOutV2,
    checked_span, read_array, read_pod,
};
use super::super::batch::{
    ConvolutionInput, ConvolutionIr, ConvolutionJob, ConvolutionOutput, ConvolutionSharing,
    PendingWrite,
};
use crate::read_guest_bytes;
use half::f16;
#[cfg(test)]
use std::collections::VecDeque;

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

pub(in crate::acm_engine) fn execute(job: &ConvolutionJob) -> Result<Vec<PendingWrite>, AcmError> {
    let mut pending = Vec::new();
    let mut input_states = Vec::with_capacity(job.inputs.len());
    let mut input_null_counts = Vec::with_capacity(job.inputs.len());
    for input in &job.inputs {
        let (history_position, history_count, history_null_count) =
            current_input_state(input, job.version2)?;
        let all_live = input.pcm.iter().all(Option::is_some);
        let all_null = input.pcm.iter().all(Option::is_none);
        if !all_live && !all_null {
            return Err(AcmError::InvalidParameter);
        }
        if (!job.version2 && all_null) || (history_null_count > 0 && all_live) {
            return Err(AcmError::InvalidParameter);
        }
        let position = if history_count == 0 {
            0
        } else {
            (history_position + 1) % input.block_count
        };
        let count = (history_count + 1).min(input.block_count);
        let null_count = if all_null {
            history_null_count.saturating_add(1).min(count)
        } else {
            0
        };
        let mut spectra = Vec::with_capacity(input.channels.len());
        for (channel, pcm) in input.channels.iter().zip(&input.pcm) {
            let spectrum = if let Some(pcm) = pcm {
                let samples = read_f32_block(*pcm, input.block_size)?;
                super::fft::execute_fft_f32(&samples, input.block_size * 2, 1)?
            } else {
                vec![0.0; input.block_size * 2]
            };
            let full_bytes = input.block_size * 2 * format_bytes(input.format);
            pending.push(PendingWrite {
                address: channel
                    .blocks
                    .checked_add((position * full_bytes) as u64)
                    .ok_or(AcmError::InvalidParameter)?,
                bytes: encode_spectrum(&spectrum, input.format),
            });
            spectra.push(spectrum);
        }
        let mut history = Vec::with_capacity(input.channels.len());
        for (channel_index, channel) in input.channels.iter().enumerate() {
            let mut channel_history = vec![spectra[channel_index].clone()];
            for partition in 1..count {
                let slot = (position + input.block_count - partition) % input.block_count;
                channel_history.push(read_spectrum(
                    channel
                        .blocks
                        .checked_add(
                            (slot * input.block_size * 2 * format_bytes(input.format)) as u64,
                        )
                        .ok_or(AcmError::InvalidParameter)?,
                    input.block_size * 2,
                    input.format,
                    input.block_size * 2 * format_bytes(input.format),
                )?);
            }
            history.push(channel_history);
        }
        pending.push(PendingWrite {
            address: input.address + 16,
            bytes: [position as u32, count as u32]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
        });
        if job.version2 {
            pending.push(PendingWrite {
                address: input.address + 40,
                bytes: (null_count as u32).to_le_bytes().to_vec(),
            });
        }
        input_states.push(history);
        input_null_counts.push(null_count);
    }

    for tuple in 0..job.outputs.len() {
        let input_index = match job.sharing {
            ConvolutionSharing::SharedInput => 0,
            ConvolutionSharing::SharedIr => tuple,
        };
        let ir_index = match job.sharing {
            ConvolutionSharing::SharedInput => tuple,
            ConvolutionSharing::SharedIr => 0,
        };
        execute_tuple(
            &job.inputs[input_index],
            &input_states[input_index],
            input_null_counts[input_index],
            &job.irs[ir_index],
            job.gains[tuple],
            &job.outputs[tuple],
            job.version2,
            &mut pending,
        )?;
    }
    Ok(pending)
}

fn execute_tuple(
    input: &ConvolutionInput,
    histories: &[Vec<Vec<f32>>],
    leading_null_count: usize,
    ir: &ConvolutionIr,
    tuple_gain: f32,
    output: &ConvolutionOutput,
    version2: bool,
    pending: &mut Vec<PendingWrite>,
) -> Result<(), AcmError> {
    let previous_gain = if version2 {
        let value = read_pod::<ConvOutV2>(output.address)?;
        if !value.previous_gain.is_finite() {
            return Err(AcmError::InvalidParameter);
        }
        value.previous_gain
    } else {
        1.0
    };
    let channel_count = output.pcm.len();
    for output_channel in 0..channel_count {
        let input_channel = if input.channels.len() == 1 {
            0
        } else {
            output_channel
        };
        let ir_channel = if ir.channels.len() == 1 {
            0
        } else {
            output_channel
        };
        let partitions = histories[input_channel].len().min(ir.block_count);
        let transform_size = input.block_size * 2;
        let mut accumulated = vec![0.0_f32; transform_size];
        let mut ir_offset = 0_usize;
        for partition in 0..partitions {
            let (memory_bytes, runtime_bytes) = ir.sizes[ir_channel].as_ref().map_or(
                (
                    transform_size * format_bytes(ir.format),
                    transform_size * format_bytes(ir.format),
                ),
                |sizes| {
                    (
                        usize::from(sizes[partition].size_memory),
                        usize::from(sizes[partition].size),
                    )
                },
            );
            let ir_address = ir.channels[ir_channel]
                .blocks
                .checked_add(ir_offset as u64)
                .ok_or(AcmError::InvalidParameter)?;
            ir_offset = ir_offset
                .checked_add(memory_bytes)
                .ok_or(AcmError::InvalidParameter)?;
            if partition < leading_null_count {
                continue;
            }
            let ir_spectrum = read_spectrum(ir_address, transform_size, ir.format, runtime_bytes)?;
            multiply_accumulate(
                &mut accumulated,
                &histories[input_channel][partition],
                &ir_spectrum,
                tuple_gain * ir.block_gains[partition],
            );
        }
        let time = super::fft::execute_ifft_f32(&accumulated, transform_size, 0)?;
        let previous_overlap = read_f32_block(output.temps[output_channel], input.block_size)?;
        let initial = match output.initial_pcm[output_channel] {
            Some(address) => read_f32_block(address, input.block_size)?,
            None => vec![0.0; input.block_size],
        };
        let mut pcm = Vec::with_capacity(input.block_size);
        for sample in 0..input.block_size {
            let wet = time[sample] + previous_overlap[sample];
            let gain = if version2 {
                previous_gain
                    + (output.gain - previous_gain)
                        * (sample as f32 / (input.block_size - 1) as f32)
            } else {
                1.0
            };
            pcm.push(initial[sample] + wet * gain);
        }
        pending.push(PendingWrite {
            address: output.pcm[output_channel],
            bytes: encode_f32(&pcm),
        });
        pending.push(PendingWrite {
            address: output.temps[output_channel],
            bytes: encode_f32(&time[input.block_size..]),
        });
    }
    if version2 {
        pending.push(PendingWrite {
            address: output.address + 28,
            bytes: output.gain.to_le_bytes().to_vec(),
        });
    }
    Ok(())
}

fn current_input_state(
    input: &ConvolutionInput,
    version2: bool,
) -> Result<(usize, usize, usize), AcmError> {
    let (base, null_count) = if version2 {
        let value = read_pod::<ConvInV2>(input.address)?;
        (value.base, value.history_null_count as usize)
    } else {
        (read_pod::<ConvIn>(input.address)?, 0)
    };
    let position = base.history_position as usize;
    let count = base.history_count as usize;
    if count > input.block_count
        || (count != 0 && position >= input.block_count)
        || null_count > count
    {
        return Err(AcmError::InvalidParameter);
    }
    Ok((position, count, null_count))
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
        if channel.sizes == 0 {
            validate_audio(channel.blocks, full_block_bytes * block_count)?;
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
            let total_bytes = values.iter().try_fold(0_usize, |total, size| {
                total
                    .checked_add(usize::from(size.size_memory))
                    .ok_or(AcmError::InvalidParameter)
            })?;
            validate_audio(channel.blocks, total_bytes)?;
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

fn read_f32_block(address: u64, count: usize) -> Result<Vec<f32>, AcmError> {
    validate_audio(
        address,
        count.checked_mul(4).ok_or(AcmError::InvalidParameter)?,
    )?;
    read_f32_array(address, count)
}

fn read_spectrum(
    address: u64,
    value_count: usize,
    format: AcmFormat,
    runtime_bytes: usize,
) -> Result<Vec<f32>, AcmError> {
    let bytes_per_value = format_bytes(format);
    let full_bytes = value_count
        .checked_mul(bytes_per_value)
        .ok_or(AcmError::InvalidParameter)?;
    if runtime_bytes > full_bytes || !runtime_bytes.is_multiple_of(bytes_per_value) {
        return Err(AcmError::InvalidParameter);
    }
    validate_audio(address, runtime_bytes)?;
    let mut bytes = vec![0_u8; runtime_bytes];
    read_guest_bytes(address, &mut bytes);
    let mut spectrum = vec![0.0_f32; value_count];
    match format {
        AcmFormat::Fp32 => {
            for (destination, value) in spectrum.iter_mut().zip(bytes.chunks_exact(4)) {
                *destination =
                    f32::from_le_bytes(value.try_into().expect("four-byte spectrum value"));
            }
        }
        AcmFormat::Fp16 => {
            for (destination, value) in spectrum.iter_mut().zip(bytes.chunks_exact(2)) {
                *destination = f16::from_bits(u16::from_le_bytes(
                    value.try_into().expect("two-byte spectrum value"),
                ))
                .to_f32();
            }
        }
    }
    if spectrum.iter().any(|value| !value.is_finite()) {
        return Err(AcmError::InvalidParameter);
    }
    Ok(spectrum)
}

fn encode_spectrum(values: &[f32], format: AcmFormat) -> Vec<u8> {
    match format {
        AcmFormat::Fp32 => encode_f32(values),
        AcmFormat::Fp16 => values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect(),
    }
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
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

fn multiply_accumulate(destination: &mut [f32], left: &[f32], right: &[f32], gain: f32) {
    destination[0] += left[0] * right[0] * gain;
    destination[1] += left[1] * right[1] * gain;
    for frequency in 1..destination.len() / 2 {
        let index = frequency * 2;
        let left_real = left[index];
        let left_imaginary = left[index + 1];
        let right_real = right[index];
        let right_imaginary = right[index + 1];
        destination[index] += (left_real * right_real - left_imaginary * right_imaginary) * gain;
        destination[index + 1] +=
            (left_real * right_imaginary + left_imaginary * right_real) * gain;
    }
}

#[cfg(test)]
pub(super) fn run_partitioned_stream_for_test(
    block_size: usize,
    input: &[f32],
    impulse: &[f32],
) -> Vec<f32> {
    let transform_size = block_size * 2;
    let ir_partitions = impulse
        .chunks(block_size)
        .map(|partition| {
            let mut block = partition.to_vec();
            block.resize(block_size, 0.0);
            super::fft::execute_fft_f32(&block, transform_size, 1).expect("valid IR partition")
        })
        .collect::<Vec<_>>();
    let output_length = input.len() + impulse.len() - 1;
    let output_blocks = output_length.div_ceil(block_size);
    let mut history = VecDeque::<Vec<f32>>::new();
    let mut overlap = vec![0.0_f32; block_size];
    let mut output = Vec::with_capacity(output_blocks * block_size);
    for block_index in 0..output_blocks {
        let start = block_index * block_size;
        let mut block = input
            .get(start..(start + block_size).min(input.len()))
            .unwrap_or_default()
            .to_vec();
        block.resize(block_size, 0.0);
        history.push_front(
            super::fft::execute_fft_f32(&block, transform_size, 1).expect("valid input partition"),
        );
        history.truncate(ir_partitions.len());
        let mut spectrum = vec![0.0_f32; transform_size];
        for (input, ir) in history.iter().zip(&ir_partitions) {
            multiply_accumulate(&mut spectrum, input, ir, 1.0);
        }
        let time = super::fft::execute_ifft_f32(&spectrum, transform_size, 0)
            .expect("valid convolution IFFT");
        output.extend(
            time[..block_size]
                .iter()
                .zip(&overlap)
                .map(|(sample, overlap)| sample + overlap),
        );
        overlap.copy_from_slice(&time[block_size..]);
    }
    output
}
