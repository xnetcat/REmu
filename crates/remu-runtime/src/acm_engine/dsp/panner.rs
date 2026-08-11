// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Panner filtering, persistent state, gain ramps, and output accumulation.

use super::super::abi::{AcmError, checked_span};
use super::super::batch::{PannerJob, PendingWrite};
use crate::read_guest_bytes;

const FRAME_SAMPLES: usize = 256;

pub(in crate::acm_engine) fn execute(job: &PannerJob) -> Result<Vec<PendingWrite>, AcmError> {
    if job.inputs.is_empty()
        || job.inputs.len() != job.parameters.len()
        || job.inputs.len() != job.states.len()
        || job.biquad_count > 2
        || job.update_log2 > 1
        || !(1..=36).contains(&job.output_count)
        || job.initial_outputs.len() != job.output_count
        || job.outputs.len() != job.output_count
    {
        return Err(AcmError::InvalidParameter);
    }
    let mut mixed = Vec::with_capacity(job.output_count);
    for initial in &job.initial_outputs {
        mixed.push(match initial {
            Some(address) => read_f32s(*address, FRAME_SAMPLES)?,
            None => vec![0.0; FRAME_SAMPLES],
        });
    }

    let coefficient_sets = 1_usize << job.update_log2;
    let coefficient_count = job
        .biquad_count
        .checked_mul(coefficient_sets)
        .and_then(|count| count.checked_mul(5))
        .ok_or(AcmError::InvalidParameter)?;
    let parameter_gain_offset = 4 + coefficient_count * 4;
    let state_gain_offset = job.biquad_count * 8;
    let parameter_bytes = padded_128(parameter_gain_offset + job.output_count * 4)?;
    let state_bytes = padded_128(state_gain_offset + job.output_count * 4)?;
    let mut pending_states = Vec::with_capacity(job.inputs.len());
    for voice in 0..job.inputs.len() {
        let input = read_f32s(job.inputs[voice], FRAME_SAMPLES)?;
        let parameter = read_bytes(job.parameters[voice], parameter_bytes)?;
        let flags = u32::from_le_bytes(parameter[..4].try_into().expect("panner flags"));
        if flags & !1 != 0 {
            return Err(AcmError::InvalidParameter);
        }
        let coefficients = decode_f32s(&parameter[4..parameter_gain_offset]);
        let target = decode_f32s(
            &parameter[parameter_gain_offset..parameter_gain_offset + job.output_count * 4],
        );
        let mut state = read_bytes(job.states[voice], state_bytes)?;
        let mut biquad_state = decode_f32s(&state[..state_gain_offset]);
        let previous =
            decode_f32s(&state[state_gain_offset..state_gain_offset + job.output_count * 4]);
        if flags & 1 != 0 {
            biquad_state.fill(0.0);
        }
        let mut filtered = Vec::with_capacity(FRAME_SAMPLES);
        for (sample, mut value) in input.into_iter().enumerate() {
            let coefficient_set = usize::from(job.update_log2 == 1 && sample >= 128);
            for biquad in 0..job.biquad_count {
                let coefficient = (biquad * coefficient_sets + coefficient_set) * 5;
                let [b0, b1, b2, a1, a2] = coefficients[coefficient..coefficient + 5]
                    .try_into()
                    .expect("validated panner coefficient layout");
                let state_offset = biquad * 2;
                let output = b0 * value + biquad_state[state_offset];
                biquad_state[state_offset] =
                    b1 * value - a1 * output + biquad_state[state_offset + 1];
                biquad_state[state_offset + 1] = b2 * value - a2 * output;
                value = output;
            }
            filtered.push(value);
        }
        for output in 0..job.output_count {
            let start = if flags & 1 != 0 {
                target[output]
            } else {
                previous[output]
            };
            for sample in 0..FRAME_SAMPLES {
                let fraction = sample as f32 / (FRAME_SAMPLES - 1) as f32;
                let gain = start + (target[output] - start) * fraction;
                mixed[output][sample] += filtered[sample] * gain;
            }
        }
        encode_f32s(&biquad_state, &mut state[..state_gain_offset]);
        encode_f32s(
            &target,
            &mut state[state_gain_offset..state_gain_offset + job.output_count * 4],
        );
        pending_states.push(PendingWrite {
            address: job.states[voice],
            bytes: state,
        });
    }

    let mut pending = Vec::with_capacity(job.output_count + pending_states.len());
    for (address, output) in job.outputs.iter().copied().zip(mixed) {
        pending.push(PendingWrite {
            address,
            bytes: output.into_iter().flat_map(f32::to_le_bytes).collect(),
        });
    }
    pending.extend(pending_states);
    Ok(pending)
}

fn padded_128(bytes: usize) -> Result<usize, AcmError> {
    bytes
        .checked_add(127)
        .map(|bytes| bytes & !127)
        .ok_or(AcmError::InvalidParameter)
}

fn read_bytes(address: u64, byte_count: usize) -> Result<Vec<u8>, AcmError> {
    checked_span(address, byte_count, 128)?;
    let mut bytes = vec![0_u8; byte_count];
    read_guest_bytes(address, &mut bytes);
    Ok(bytes)
}

fn read_f32s(address: u64, count: usize) -> Result<Vec<f32>, AcmError> {
    let bytes = read_bytes(
        address,
        count.checked_mul(4).ok_or(AcmError::InvalidParameter)?,
    )?;
    Ok(decode_f32s(&bytes))
}

fn decode_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().expect("four-byte float")))
        .collect()
}

fn encode_f32s(values: &[f32], destination: &mut [u8]) {
    for (value, bytes) in values.iter().zip(destination.chunks_exact_mut(4)) {
        bytes.copy_from_slice(&value.to_le_bytes());
    }
}
