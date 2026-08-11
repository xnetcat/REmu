// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::panner::execute;
use crate::acm_engine::abi::{AcmError, SceAcmBatchInfo, return_error};
use crate::acm_engine::batch::PannerJob;
use crate::acm_engine::builders::panner;

#[repr(align(128))]
struct AlignedFloats([f32; 256]);

#[repr(align(128))]
struct AlignedBytes([u8; 256]);

fn write_f32s(bytes: &mut [u8], offset: usize, values: &[f32]) {
    for (index, value) in values.iter().enumerate() {
        let start = offset + index * 4;
        bytes[start..start + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn read_output(address: u64) -> Vec<f32> {
    let mut bytes = vec![0_u8; 256 * 4];
    crate::read_guest_bytes(address, &mut bytes);
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
        .collect()
}

fn apply(writes: &[crate::acm_engine::batch::PendingWrite]) {
    for write in writes {
        crate::write_guest_bytes(write.address, &write.bytes);
    }
}

#[test]
fn panner_applies_per_sample_gain_ramp_and_optional_initial_output() {
    let mut input = AlignedFloats([1.0; 256]);
    let mut initial = AlignedFloats([0.25; 256]);
    let mut output_left = AlignedFloats([0.0; 256]);
    let mut output_right = AlignedFloats([0.0; 256]);
    let mut parameters = AlignedBytes([0; 256]);
    write_f32s(&mut parameters.0, 4, &[0.5, 1.0]);
    let mut state = AlignedBytes([0; 256]);
    write_f32s(&mut state.0, 0, &[0.0, 0.5]);
    let job = PannerJob {
        biquad_count: 0,
        update_log2: 0,
        output_count: 2,
        inputs: vec![input.0.as_mut_ptr() as u64],
        parameters: vec![parameters.0.as_mut_ptr() as u64],
        states: vec![state.0.as_mut_ptr() as u64],
        initial_outputs: vec![Some(initial.0.as_mut_ptr() as u64), None],
        outputs: vec![
            output_left.0.as_mut_ptr() as u64,
            output_right.0.as_mut_ptr() as u64,
        ],
    };

    let writes = execute(&job).expect("valid panner job");
    apply(&writes);
    let left = read_output(output_left.0.as_mut_ptr() as u64);
    let right = read_output(output_right.0.as_mut_ptr() as u64);
    assert!((left[0] - 0.25).abs() < 1.0e-6);
    assert!((left[255] - 0.75).abs() < 1.0e-5);
    assert!((right[0] - 0.5).abs() < 1.0e-6);
    assert!((right[255] - 1.0).abs() < 1.0e-5);
}

fn reference_biquad(input: &[f32], coefficient_sets: &[[f32; 5]; 2]) -> Vec<f32> {
    let mut z1 = 0.0_f32;
    let mut z2 = 0.0_f32;
    input
        .iter()
        .enumerate()
        .map(|(sample, input)| {
            let [b0, b1, b2, a1, a2] = coefficient_sets[usize::from(sample >= 128)];
            let output = b0 * input + z1;
            z1 = b1 * input - a1 * output + z2;
            z2 = b2 * input - a2 * output;
            output
        })
        .collect()
}

#[test]
fn panner_biquads_update_at_half_frames_and_key_on_resets_state() {
    let coefficients = [[1.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0]];
    let mut input = AlignedFloats([1.0; 256]);
    let mut output = AlignedFloats([0.0; 256]);
    let mut parameters = AlignedBytes([0; 256]);
    write_f32s(&mut parameters.0, 4, &coefficients.concat());
    write_f32s(&mut parameters.0, 44, &[1.0]);
    let mut state = AlignedBytes([0; 256]);
    write_f32s(&mut state.0, 0, &[0.75, -0.25]);
    write_f32s(&mut state.0, 8, &[1.0]);
    let job = PannerJob {
        biquad_count: 1,
        update_log2: 1,
        output_count: 1,
        inputs: vec![input.0.as_mut_ptr() as u64],
        parameters: vec![parameters.0.as_mut_ptr() as u64],
        states: vec![state.0.as_mut_ptr() as u64],
        initial_outputs: vec![None],
        outputs: vec![output.0.as_mut_ptr() as u64],
    };

    apply(&execute(&job).expect("continued biquad state"));
    let continued = read_output(output.0.as_mut_ptr() as u64);
    parameters.0[..4].copy_from_slice(&1_u32.to_le_bytes());
    output.0.fill(0.0);
    apply(&execute(&job).expect("key-on resets state"));
    let reset = read_output(output.0.as_mut_ptr() as u64);
    assert_ne!(continued[0], reset[0]);
    let expected = reference_biquad(&input.0, &coefficients);
    for (index, (actual, expected)) in reset.iter().zip(expected).enumerate() {
        assert!(
            (*actual - expected).abs() < 1.0e-5,
            "sample {index}: actual={actual} expected={expected}"
        );
    }
}

#[test]
fn panner_builder_uses_twenty_words_and_rejects_invalid_shapes_atomically() {
    let mut arena = AlignedBytes([0; 256]);
    let mut inputs = [AlignedFloats([0.0; 256]), AlignedFloats([0.0; 256])];
    let mut parameters = [AlignedBytes([0; 256]), AlignedBytes([0; 256])];
    let mut states = [AlignedBytes([0; 256]), AlignedBytes([0; 256])];
    let mut outputs = [AlignedFloats([0.0; 256]), AlignedFloats([0.0; 256])];
    let input_pointers = inputs
        .iter_mut()
        .map(|buffer| buffer.0.as_mut_ptr() as u64)
        .collect::<Vec<_>>();
    let parameter_pointers = parameters
        .iter_mut()
        .map(|buffer| buffer.0.as_mut_ptr() as u64)
        .collect::<Vec<_>>();
    let state_pointers = states
        .iter_mut()
        .map(|buffer| buffer.0.as_mut_ptr() as u64)
        .collect::<Vec<_>>();
    let initial_pointers = [0_u64; 2];
    let output_pointers = outputs
        .iter_mut()
        .map(|buffer| buffer.0.as_mut_ptr() as u64)
        .collect::<Vec<_>>();
    let mut info = SceAcmBatchInfo {
        buffer: arena.0.as_mut_ptr() as u64,
        offset: 0,
        size: arena.0.len() as u64,
    };
    let arguments = [
        std::ptr::from_mut(&mut info) as u64,
        2,
        input_pointers.as_ptr() as u64,
        1,
        1,
        2,
    ];
    let stack = [
        parameter_pointers.as_ptr() as u64,
        state_pointers.as_ptr() as u64,
        initial_pointers.as_ptr() as u64,
        output_pointers.as_ptr() as u64,
        0,
        0,
    ];
    assert_eq!(panner(arguments, stack), 0);
    assert_eq!(info.offset, 20 * 8);

    info.offset = 0;
    let mut invalid = arguments;
    invalid[3] = 3;
    assert_eq!(
        panner(invalid, stack),
        return_error(AcmError::InvalidParameter)
    );
    assert_eq!(info.offset, 0);
}
