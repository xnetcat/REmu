// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::f64::consts::TAU;

use rustfft::num_complex::Complex32;

use half::f16;

use super::fft::{execute, execute_fft_f32, execute_ifft_f32, unpack_real_spectrum};
use crate::acm_engine::abi::{AcmError, AcmFormat, SceAcmBatchInfo, return_error};
use crate::acm_engine::batch::{FftBuffer, FftDirection, FftJob};
use crate::acm_engine::builders::fft;

fn scalar_real_dft(signal: &[f32]) -> Vec<Complex32> {
    let size = signal.len();
    (0..=size / 2)
        .map(|frequency| {
            let mut real = 0.0_f64;
            let mut imaginary = 0.0_f64;
            for (sample, value) in signal.iter().copied().enumerate() {
                let angle = -TAU * frequency as f64 * sample as f64 / size as f64;
                real += f64::from(value) * angle.cos();
                imaginary += f64::from(value) * angle.sin();
            }
            Complex32::new(real as f32, imaginary as f32)
        })
        .collect()
}

fn reference_signals(size: usize) -> Vec<Vec<f32>> {
    let mut impulse = vec![0.0; size];
    impulse[0] = 1.0;
    let dc = vec![0.25; size];
    let nyquist = (0..size)
        .map(|sample| if sample % 2 == 0 { 0.5 } else { -0.5 })
        .collect();
    vec![impulse, dc, nyquist]
}

fn assert_complex_close(actual: &[Complex32], expected: &[Complex32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (*actual - *expected).norm();
        let bound = tolerance * (1.0 + expected.norm());
        assert!(
            error <= bound,
            "bin {index}: actual={actual:?} expected={expected:?} error={error} bound={bound}"
        );
    }
}

fn assert_samples_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "sample {index}: actual={actual} expected={expected}"
        );
    }
}

#[test]
fn fft_impulse_dc_nyquist_and_round_trip_match_scalar_reference() {
    for size in [512_usize, 1024, 2048] {
        for signal in reference_signals(size) {
            let expected = scalar_real_dft(&signal);
            let packed = execute_fft_f32(&signal, size, 0).expect("valid FFT");
            assert_complex_close(&unpack_real_spectrum(&packed, size), &expected, 2.0e-4);
            let restored = execute_ifft_f32(&packed, size, 0).expect("valid IFFT");
            assert_samples_close(&restored, &signal, 3.0e-4);
        }
    }
}

#[repr(align(128))]
#[derive(Clone, Copy)]
struct AlignedBuffer([u8; 8192]);

fn fft_builder_result(flags: u32, inverse: bool) -> (u64, usize) {
    let mut arena = AlignedBuffer([0; 8192]);
    let mut inputs = [AlignedBuffer([0; 8192]); 6];
    let mut outputs = [AlignedBuffer([0; 8192]); 6];
    let input_pointers = inputs
        .iter_mut()
        .map(|buffer| buffer.0.as_mut_ptr() as u64)
        .collect::<Vec<_>>();
    let output_pointers = outputs
        .iter_mut()
        .map(|buffer| buffer.0.as_mut_ptr() as u64)
        .collect::<Vec<_>>();
    let mut info = SceAcmBatchInfo {
        buffer: arena.0.as_mut_ptr() as u64,
        offset: 0,
        size: arena.0.len() as u64,
    };
    let result = fft(
        [
            std::ptr::from_mut(&mut info) as u64,
            512,
            3,
            0,
            input_pointers.as_ptr() as u64,
            0,
        ],
        [
            output_pointers.as_ptr() as u64,
            u64::from(flags),
            0,
            0,
            0,
            0,
        ],
        inverse,
    );
    (result, info.offset as usize / 8)
}

#[test]
fn fft_builders_use_sdk_word_counts_and_honor_all_split_flags() {
    for flags in 0_u32..=7 {
        let expected = 2 + 3 * (7 + usize::from(flags & 2 != 0) + usize::from(flags & 4 != 0));
        assert_eq!(
            fft_builder_result(flags, false),
            (0, expected),
            "FFT flags={flags}"
        );
        let inverse_expected =
            2 + 3 * (7 + usize::from(flags & 2 != 0) + usize::from(flags & (1 | 4) != 0));
        assert_eq!(
            fft_builder_result(flags, true),
            (0, inverse_expected),
            "IFFT flags={flags}"
        );
    }
    assert_eq!(
        fft_builder_result(8, false),
        (return_error(AcmError::InvalidParameter), 0)
    );
}

fn write_f16(address: u64, values: &[f32]) {
    let bytes = values
        .iter()
        .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    crate::write_guest_bytes(address, &bytes);
}

fn read_f16(address: u64, count: usize) -> Vec<f32> {
    let mut bytes = vec![0_u8; count * 2];
    crate::read_guest_bytes(address, &mut bytes);
    bytes
        .chunks_exact(2)
        .map(|value| f16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32())
        .collect()
}

fn apply(writes: &[crate::acm_engine::batch::PendingWrite]) {
    for write in writes {
        crate::write_guest_bytes(write.address, &write.bytes);
    }
}

#[test]
fn fp16_and_half_frame_flags_match_reference() {
    let mut input_a = AlignedBuffer([0; 8192]);
    let mut input_b = AlignedBuffer([0; 8192]);
    let mut output_a = AlignedBuffer([0; 8192]);
    let mut output_b = AlignedBuffer([0; 8192]);
    let mut half_signal = vec![0.0_f32; 256];
    half_signal[0] = 1.0;
    write_f16(input_a.0.as_mut_ptr() as u64, &half_signal[..128]);
    write_f16(input_b.0.as_mut_ptr() as u64, &half_signal[128..]);
    let forward = FftJob {
        direction: FftDirection::Forward,
        size: 512,
        input_format: AcmFormat::Fp16,
        output_format: AcmFormat::Fp16,
        inputs: vec![FftBuffer {
            first: input_a.0.as_mut_ptr() as u64,
            second: Some(input_b.0.as_mut_ptr() as u64),
        }],
        outputs: vec![FftBuffer {
            first: output_a.0.as_mut_ptr() as u64,
            second: Some(output_b.0.as_mut_ptr() as u64),
        }],
        flags: 7,
    };
    let writes = execute(&forward).expect("valid fp16 split FFT");
    assert_eq!(writes.len(), 2);
    apply(&writes);
    let mut packed = read_f16(output_a.0.as_mut_ptr() as u64, 256);
    packed.extend(read_f16(output_b.0.as_mut_ptr() as u64, 256));
    let mut full_signal = half_signal;
    full_signal.resize(512, 0.0);
    let expected = execute_fft_f32(&full_signal, 512, 0).unwrap();
    assert_samples_close(&packed, &expected, 2.0e-2);

    let mut inverse_input = AlignedBuffer([0; 8192]);
    let mut inverse_output = AlignedBuffer([0; 8192]);
    let mut impulse = vec![0.0_f32; 512];
    impulse[300] = 1.0;
    let packed = execute_fft_f32(&impulse, 512, 0).unwrap();
    crate::write_guest_bytes(
        inverse_input.0.as_mut_ptr() as u64,
        &packed
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let inverse = FftJob {
        direction: FftDirection::Inverse,
        size: 512,
        input_format: AcmFormat::Fp32,
        output_format: AcmFormat::Fp32,
        inputs: vec![FftBuffer {
            first: inverse_input.0.as_mut_ptr() as u64,
            second: None,
        }],
        outputs: vec![FftBuffer {
            first: inverse_output.0.as_mut_ptr() as u64,
            second: None,
        }],
        flags: 1,
    };
    let writes = execute(&inverse).expect("valid drop-first-half IFFT");
    assert_eq!(writes.len(), 1);
    apply(&writes);
    let restored = inverse_output.0[..256 * 4]
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_samples_close(&restored, &impulse[256..], 3.0e-4);
}
