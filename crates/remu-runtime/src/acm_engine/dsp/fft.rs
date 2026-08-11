// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Real FFT/IFFT execution and the ACM packed-spectrum representation.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

use half::f16;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use super::super::abi::AcmError;
use super::super::batch::{FftDirection, FftJob, PendingWrite};
use crate::read_guest_bytes;

const HALF_FRAME: u32 = 1;
const VALID_FLAGS: u32 = 1 | 2 | 4;

struct PlanCache {
    planner: FftPlanner<f32>,
    plans: BTreeMap<(usize, bool), Arc<dyn Fft<f32>>>,
}

impl Default for PlanCache {
    fn default() -> Self {
        Self {
            planner: FftPlanner::new(),
            plans: BTreeMap::new(),
        }
    }
}

static PLANS: OnceLock<Mutex<PlanCache>> = OnceLock::new();

pub(in crate::acm_engine) fn execute(job: &FftJob) -> Result<Vec<PendingWrite>, AcmError> {
    validate(job.size, job.flags)?;
    if job.inputs.is_empty() || job.inputs.len() != job.outputs.len() {
        return Err(AcmError::InvalidParameter);
    }
    let split_input = job.flags & 2 != 0;
    let split_output = job.flags & 4 != 0;
    let input_values = match job.direction {
        FftDirection::Forward if job.flags & 1 != 0 => job.size / 2,
        _ => job.size,
    };

    let mut pending = Vec::with_capacity(job.outputs.len() * (1 + usize::from(split_output)));
    for (input, output) in job.inputs.iter().zip(&job.outputs) {
        if input.second.is_some() != split_input || output.second.is_some() != split_output {
            return Err(AcmError::InvalidParameter);
        }
        let input_parts = 1 + usize::from(split_input);
        let mut samples = read_samples(input.first, input_values / input_parts, job.input_format)?;
        if let Some(second) = input.second {
            samples.extend(read_samples(
                second,
                input_values / input_parts,
                job.input_format,
            )?);
        }
        let transformed = match job.direction {
            FftDirection::Forward => execute_fft_f32(&samples, job.size, job.flags)?,
            FftDirection::Inverse => execute_ifft_f32(&samples, job.size, job.flags)?,
        };
        let output_parts = 1 + usize::from(split_output);
        let values_per_output = transformed.len() / output_parts;
        if values_per_output * output_parts != transformed.len() {
            return Err(AcmError::InvalidParameter);
        }
        pending.push(PendingWrite {
            address: output.first,
            bytes: encode_samples(&transformed[..values_per_output], job.output_format),
        });
        if let Some(second) = output.second {
            pending.push(PendingWrite {
                address: second,
                bytes: encode_samples(&transformed[values_per_output..], job.output_format),
            });
        }
    }
    Ok(pending)
}

fn read_samples(
    address: u64,
    count: usize,
    format: super::super::abi::AcmFormat,
) -> Result<Vec<f32>, AcmError> {
    let bytes_per_sample = match format {
        super::super::abi::AcmFormat::Fp32 => 4,
        super::super::abi::AcmFormat::Fp16 => 2,
    };
    let byte_count = count
        .checked_mul(bytes_per_sample)
        .ok_or(AcmError::InvalidParameter)?;
    super::super::abi::checked_span(address, byte_count, 128)?;
    let mut bytes = vec![0_u8; byte_count];
    read_guest_bytes(address, &mut bytes);
    Ok(match format {
        super::super::abi::AcmFormat::Fp32 => bytes
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().expect("four-byte sample")))
            .collect(),
        super::super::abi::AcmFormat::Fp16 => bytes
            .chunks_exact(2)
            .map(|value| {
                f16::from_bits(u16::from_le_bytes(
                    value.try_into().expect("two-byte sample"),
                ))
                .to_f32()
            })
            .collect(),
    })
}

fn encode_samples(samples: &[f32], format: super::super::abi::AcmFormat) -> Vec<u8> {
    match format {
        super::super::abi::AcmFormat::Fp32 => samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect(),
        super::super::abi::AcmFormat::Fp16 => samples
            .iter()
            .flat_map(|sample| f16::from_f32(*sample).to_bits().to_le_bytes())
            .collect(),
    }
}

fn transform(size: usize, inverse: bool) -> Arc<dyn Fft<f32>> {
    let mut cache = PLANS
        .get_or_init(|| Mutex::new(PlanCache::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(plan) = cache.plans.get(&(size, inverse)) {
        return Arc::clone(plan);
    }
    let plan = if inverse {
        cache.planner.plan_fft_inverse(size)
    } else {
        cache.planner.plan_fft_forward(size)
    };
    cache.plans.insert((size, inverse), Arc::clone(&plan));
    plan
}

fn validate(size: usize, flags: u32) -> Result<(), AcmError> {
    if !matches!(size, 512 | 1024 | 2048) || flags & !VALID_FLAGS != 0 {
        return Err(AcmError::InvalidParameter);
    }
    Ok(())
}

pub(super) fn execute_fft_f32(
    input: &[f32],
    size: usize,
    flags: u32,
) -> Result<Vec<f32>, AcmError> {
    validate(size, flags)?;
    let expected = if flags & HALF_FRAME != 0 {
        size / 2
    } else {
        size
    };
    if input.len() != expected {
        return Err(AcmError::InvalidParameter);
    }

    let mut spectrum = vec![Complex32::new(0.0, 0.0); size];
    for (destination, source) in spectrum.iter_mut().zip(input) {
        destination.re = *source;
    }
    transform(size, false).process(&mut spectrum);
    Ok(pack_real_spectrum(&spectrum))
}

pub(super) fn execute_ifft_f32(
    packed: &[f32],
    size: usize,
    flags: u32,
) -> Result<Vec<f32>, AcmError> {
    validate(size, flags)?;
    if packed.len() != size {
        return Err(AcmError::InvalidParameter);
    }

    let positive = unpack_real_spectrum(packed, size);
    let mut spectrum = vec![Complex32::new(0.0, 0.0); size];
    spectrum[..=size / 2].copy_from_slice(&positive);
    for frequency in 1..size / 2 {
        spectrum[size - frequency] = positive[frequency].conj();
    }
    transform(size, true).process(&mut spectrum);
    let scale = 1.0 / size as f32;
    let samples = spectrum
        .into_iter()
        .map(|sample| sample.re * scale)
        .collect::<Vec<_>>();
    if flags & HALF_FRAME != 0 {
        Ok(samples[size / 2..].to_vec())
    } else {
        Ok(samples)
    }
}

fn pack_real_spectrum(spectrum: &[Complex32]) -> Vec<f32> {
    let size = spectrum.len();
    let mut packed = vec![0.0; size];
    packed[0] = spectrum[0].re;
    packed[1] = spectrum[size / 2].re;
    for frequency in 1..size / 2 {
        packed[2 * frequency] = spectrum[frequency].re;
        packed[2 * frequency + 1] = spectrum[frequency].im;
    }
    packed
}

pub(super) fn unpack_real_spectrum(packed: &[f32], size: usize) -> Vec<Complex32> {
    assert_eq!(packed.len(), size, "validated ACM packed spectrum length");
    let mut spectrum = vec![Complex32::new(0.0, 0.0); size / 2 + 1];
    spectrum[0] = Complex32::new(packed[0], 0.0);
    spectrum[size / 2] = Complex32::new(packed[1], 0.0);
    for frequency in 1..size / 2 {
        spectrum[frequency] = Complex32::new(packed[2 * frequency], packed[2 * frequency + 1]);
    }
    spectrum
}
