// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::mem::size_of;

use half::f16;

use super::convolution::{execute, run_partitioned_stream_for_test, validate_channel_graph};
use super::fft::{execute_fft_f32, execute_ifft_f32};
use crate::acm_engine::abi::{
    AcmError, AcmFormat, ConvBlockSize, ConvChannel, ConvIn, ConvInV2, ConvIr, ConvOut, ConvOutV2,
    SceAcmBatchInfo,
};
use crate::acm_engine::batch::{
    Command, ConvolutionInput, ConvolutionIr, ConvolutionJob, ConvolutionOutput,
    ConvolutionSharing, PendingWrite, decode_buffers,
};
use crate::acm_engine::builders::convolution;

#[test]
fn convolution_struct_layouts_match_sdk10_lp64() {
    assert_eq!(size_of::<ConvChannel>(), 16);
    assert_eq!(size_of::<ConvBlockSize>(), 4);
    assert_eq!(size_of::<ConvIn>(), 40);
    assert_eq!(size_of::<ConvInV2>(), 72);
    assert_eq!(size_of::<ConvIr>(), 32);
    assert_eq!(size_of::<ConvOut>(), 24);
    assert_eq!(size_of::<ConvOutV2>(), 56);
}

#[test]
fn convolution_accepts_only_1_to_n_n_to_1_or_n_to_n_channel_graphs() {
    for (input, ir, output) in [(1, 4, 4), (4, 1, 4), (4, 4, 4)] {
        assert!(validate_channel_graph(input, ir, output).is_ok());
    }
    assert_eq!(
        validate_channel_graph(2, 3, 3),
        Err(AcmError::InvalidParameter)
    );
}

#[repr(align(128))]
struct AlignedBytes([u8; 16384]);

#[test]
fn convolution_builder_reserves_public_size_and_preserves_v2_gains() {
    let mut input_blocks = AlignedBytes([0; 16384]);
    let mut input_pcm = AlignedBytes([0; 16384]);
    let input_channel = ConvChannel {
        blocks: input_blocks.0.as_mut_ptr() as u64,
        sizes: 0,
    };
    let input_channel_pointers = [std::ptr::from_ref(&input_channel) as u64];
    let input_pcm_pointers = [input_pcm.0.as_mut_ptr() as u64];
    let mut input = ConvInV2 {
        base: ConvIn {
            block_size: 256,
            block_count: 1,
            channel_count: 1,
            format: 0,
            history_position: 0,
            history_count: 0,
            channels: input_channel_pointers.as_ptr() as u64,
            pcm: input_pcm_pointers.as_ptr() as u64,
        },
        history_null_count: 0,
        reserved: [0; 6],
        tail_padding: 0,
    };

    let mut ir_blocks = AlignedBytes([0; 16384]);
    let ir_channel = ConvChannel {
        blocks: ir_blocks.0.as_mut_ptr() as u64,
        sizes: 0,
    };
    let ir_channel_pointers = [std::ptr::from_ref(&ir_channel) as u64];
    let ir = ConvIr {
        block_size: 256,
        block_count: 1,
        channel_count: 1,
        format: 0,
        block_gains: 0,
        channels: ir_channel_pointers.as_ptr() as u64,
    };
    let ir_pointers = [std::ptr::from_ref(&ir) as u64];

    let mut temp = AlignedBytes([0; 16384]);
    let mut output_pcm = AlignedBytes([0; 16384]);
    let temp_pointers = [temp.0.as_mut_ptr() as u64];
    let output_pcm_pointers = [output_pcm.0.as_mut_ptr() as u64];
    let output = ConvOutV2 {
        base: ConvOut {
            block_size: 256,
            channel_count: 1,
            temps: temp_pointers.as_ptr() as u64,
            pcm: output_pcm_pointers.as_ptr() as u64,
        },
        gain: 0.75,
        previous_gain: 0.25,
        pcm_init: 0,
        reserved: [0; 4],
    };
    let output_pointers = [std::ptr::from_ref(&output) as u64];
    let gains = [1.0_f32];
    let mut arena = AlignedBytes([0; 16384]);
    let mut info = SceAcmBatchInfo {
        buffer: arena.0.as_mut_ptr() as u64,
        offset: 0,
        size: arena.0.len() as u64,
    };
    let arguments = [
        std::ptr::from_mut(&mut info) as u64,
        1,
        std::ptr::from_mut(&mut input) as u64,
        1,
        ir_pointers.as_ptr() as u64,
        gains.as_ptr() as u64,
    ];
    let stack = [output_pointers.as_ptr() as u64, 0, 0, 0, 0, 0];
    assert_eq!(convolution(arguments, stack, true, true), 0);
    assert_eq!(info.offset / 8, 97);
    let plan = decode_buffers(&[info]).expect("valid copied convolution graph");
    let [Command::Convolution(job)] = plan.commands.as_slice() else {
        panic!("expected one convolution command")
    };
    assert_eq!(job.outputs[0].previous_gain, 0.25);
    assert_eq!(job.outputs[0].gain, 0.75);
}

fn seeded_signal(length: usize, mut seed: u32) -> Vec<f32> {
    (0..length)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as i32 as f32) / i32::MAX as f32 * 0.05
        })
        .collect()
}

fn direct_convolution(input: &[f32], impulse: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0_f32; input.len() + impulse.len() - 1];
    for (input_index, input) in input.iter().copied().enumerate() {
        for (ir_index, ir) in impulse.iter().copied().enumerate() {
            output[input_index + ir_index] += input * ir;
        }
    }
    output
}

#[test]
fn partitioned_convolution_matches_direct_time_domain_reference() {
    for block_size in [256_usize, 512, 1024] {
        let input = seeded_signal(block_size * 3, 0x1357_2468);
        let impulse = seeded_signal(block_size * 2 + 17, 0x2468_1357);
        let expected = direct_convolution(&input, &impulse);
        let actual = run_partitioned_stream_for_test(block_size, &input, &impulse);
        assert!(actual.len() >= expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            let error = (actual - expected).abs();
            assert!(
                error < 8.0e-4,
                "block={block_size} sample={index} actual={actual} expected={expected} error={error}"
            );
        }
    }
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

fn read_f32_values(address: u64, count: usize) -> Vec<f32> {
    let mut bytes = vec![0_u8; count * 4];
    crate::read_guest_bytes(address, &mut bytes);
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().expect("four-byte output")))
        .collect()
}

fn apply(writes: &[PendingWrite]) {
    for write in writes {
        crate::write_guest_bytes(write.address, &write.bytes);
    }
}

fn write_f16_values(address: u64, values: &[f32]) {
    crate::write_guest_bytes(
        address,
        &values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
    );
}

fn mapped_fp16_output(input_levels: &[f32], ir_levels: &[f32]) -> Vec<Vec<f32>> {
    const BLOCK: usize = 256;
    const PCM_STRIDE: usize = BLOCK * 4;
    const SPECTRUM_F32_STRIDE: usize = BLOCK * 2 * 4;
    const SPECTRUM_F16_STRIDE: usize = BLOCK * 2 * 2;
    let channels = input_levels.len().max(ir_levels.len());
    let mut history = AlignedBytes([0; 16384]);
    let mut input_pcm = AlignedBytes([0; 16384]);
    let mut ir_blocks = AlignedBytes([0; 16384]);
    let mut temp = AlignedBytes([0; 16384]);
    let mut output_pcm = AlignedBytes([0; 16384]);
    let input_channels = input_levels
        .iter()
        .enumerate()
        .map(|(channel, level)| {
            let pcm = input_pcm.0.as_mut_ptr() as u64 + (channel * PCM_STRIDE) as u64;
            write_f32_values(pcm, &vec![*level; BLOCK]);
            ConvChannel {
                blocks: history.0.as_mut_ptr() as u64 + (channel * SPECTRUM_F32_STRIDE) as u64,
                sizes: 0,
            }
        })
        .collect::<Vec<_>>();
    let pcm = (0..input_levels.len())
        .map(|channel| Some(input_pcm.0.as_mut_ptr() as u64 + (channel * PCM_STRIDE) as u64))
        .collect();
    let ir_channels = ir_levels
        .iter()
        .enumerate()
        .map(|(channel, level)| {
            let mut impulse = vec![0.0_f32; BLOCK];
            impulse[0] = *level;
            let spectrum = execute_fft_f32(&impulse, BLOCK * 2, 1).expect("valid fp16 IR");
            let address = ir_blocks.0.as_mut_ptr() as u64 + (channel * SPECTRUM_F16_STRIDE) as u64;
            write_f16_values(address, &spectrum);
            ConvChannel {
                blocks: address,
                sizes: 0,
            }
        })
        .collect::<Vec<_>>();
    let output_addresses = (0..channels)
        .map(|channel| output_pcm.0.as_mut_ptr() as u64 + (channel * PCM_STRIDE) as u64)
        .collect::<Vec<_>>();
    let job = ConvolutionJob {
        sharing: ConvolutionSharing::SharedInput,
        version2: false,
        inputs: vec![ConvolutionInput {
            address: history.0.as_mut_ptr() as u64,
            block_size: BLOCK,
            block_count: 1,
            format: AcmFormat::Fp32,
            history_position: 0,
            history_count: 0,
            history_null_count: 0,
            channels: input_channels,
            pcm,
        }],
        irs: vec![ConvolutionIr {
            block_size: BLOCK,
            block_count: 1,
            format: AcmFormat::Fp16,
            channels: ir_channels,
            sizes: vec![None; ir_levels.len()],
            block_gains: vec![0.25],
        }],
        gains: vec![0.5],
        outputs: vec![ConvolutionOutput {
            address: temp.0.as_mut_ptr() as u64,
            block_size: BLOCK,
            temps: (0..channels)
                .map(|channel| temp.0.as_mut_ptr() as u64 + (channel * PCM_STRIDE) as u64)
                .collect(),
            pcm: output_addresses.clone(),
            initial_pcm: vec![None; channels],
            previous_gain: 1.0,
            gain: 1.0,
        }],
    };
    apply(&execute(&job).expect("valid channel mapping"));
    output_addresses
        .into_iter()
        .map(|address| read_f32_values(address, BLOCK))
        .collect()
}

#[test]
fn channel_mappings_and_fp16_ir_match_reference() {
    for (inputs, irs, expected) in [
        (vec![2.0], vec![3.0, 4.0], vec![0.75, 1.0]),
        (vec![2.0, 5.0], vec![3.0], vec![0.75, 1.875]),
        (vec![2.0, 5.0], vec![3.0, 4.0], vec![0.75, 2.5]),
    ] {
        let actual = mapped_fp16_output(&inputs, &irs);
        assert_eq!(actual.len(), expected.len());
        for (channel, expected) in actual.iter().zip(expected) {
            for (sample, actual) in channel.iter().enumerate() {
                assert!(
                    (*actual - expected).abs() < 1.0e-3,
                    "sample={sample} actual={actual} expected={expected}"
                );
            }
        }
    }
}

#[test]
fn history_wraparound_preserves_partition_order() {
    const BLOCK: usize = 256;
    let mut history = AlignedBytes([0; 16384]);
    let mut input_pcm = AlignedBytes([0; 16384]);
    let mut ir_blocks = AlignedBytes([0; 16384]);
    let mut temp = AlignedBytes([0; 16384]);
    let mut output_pcm = AlignedBytes([0; 16384]);
    let mut first_ir = vec![0.0_f32; BLOCK];
    first_ir[0] = 1.0;
    let mut second_ir = vec![0.0_f32; BLOCK];
    second_ir[0] = 10.0;
    let first_spectrum = execute_fft_f32(&first_ir, BLOCK * 2, 1).expect("first IR");
    let second_spectrum = execute_fft_f32(&second_ir, BLOCK * 2, 1).expect("second IR");
    write_f32_values(ir_blocks.0.as_mut_ptr() as u64, &first_spectrum);
    write_f32_values(
        ir_blocks.0.as_mut_ptr() as u64 + (BLOCK * 2 * 4) as u64,
        &second_spectrum,
    );
    let mut input_state = ConvIn::default();
    let mut job = ConvolutionJob {
        sharing: ConvolutionSharing::SharedInput,
        version2: false,
        inputs: vec![ConvolutionInput {
            address: std::ptr::from_mut(&mut input_state) as u64,
            block_size: BLOCK,
            block_count: 2,
            format: AcmFormat::Fp32,
            history_position: 0,
            history_count: 0,
            history_null_count: 0,
            channels: vec![ConvChannel {
                blocks: history.0.as_mut_ptr() as u64,
                sizes: 0,
            }],
            pcm: vec![Some(input_pcm.0.as_mut_ptr() as u64)],
        }],
        irs: vec![ConvolutionIr {
            block_size: BLOCK,
            block_count: 2,
            format: AcmFormat::Fp32,
            channels: vec![ConvChannel {
                blocks: ir_blocks.0.as_mut_ptr() as u64,
                sizes: 0,
            }],
            sizes: vec![None],
            block_gains: vec![1.0, 1.0],
        }],
        gains: vec![1.0],
        outputs: vec![ConvolutionOutput {
            address: temp.0.as_mut_ptr() as u64,
            block_size: BLOCK,
            temps: vec![temp.0.as_mut_ptr() as u64],
            pcm: vec![output_pcm.0.as_mut_ptr() as u64],
            initial_pcm: vec![None],
            previous_gain: 1.0,
            gain: 1.0,
        }],
    };

    for (call, (level, expected)) in [(1.0, 1.0), (2.0, 12.0), (3.0, 23.0)]
        .into_iter()
        .enumerate()
    {
        write_f32_values(input_pcm.0.as_mut_ptr() as u64, &vec![level; BLOCK]);
        apply(&execute(&job).expect("valid wrapped history call"));
        let output = read_f32_values(output_pcm.0.as_mut_ptr() as u64, BLOCK);
        for sample in output {
            assert!((sample - expected).abs() < 1.0e-3);
        }
        job.inputs[0].history_position = input_state.history_position as usize;
        job.inputs[0].history_count = input_state.history_count as usize;
        assert_eq!(input_state.history_position as usize, [0, 1, 0][call]);
    }
    assert_eq!(input_state.history_position, 0);
    assert_eq!(input_state.history_count, 2);
}

#[test]
fn compressed_ir_runtime_size_zeros_omitted_high_frequencies() {
    const BLOCK: usize = 256;
    const RUNTIME_VALUES: usize = 64;
    let input = seeded_signal(BLOCK, 0x9911_3377);
    let input_spectrum = execute_fft_f32(&input, BLOCK * 2, 1).expect("valid input");
    let mut impulse = vec![0.0_f32; BLOCK];
    impulse[0] = 1.0;
    let mut ir_spectrum = execute_fft_f32(&impulse, BLOCK * 2, 1).expect("valid IR");
    ir_spectrum[RUNTIME_VALUES..].fill(0.0);
    let mut product = vec![0.0_f32; BLOCK * 2];
    product[0] = input_spectrum[0] * ir_spectrum[0];
    product[1] = input_spectrum[1] * ir_spectrum[1];
    for frequency in 1..BLOCK {
        let index = frequency * 2;
        product[index] = input_spectrum[index] * ir_spectrum[index]
            - input_spectrum[index + 1] * ir_spectrum[index + 1];
        product[index + 1] = input_spectrum[index] * ir_spectrum[index + 1]
            + input_spectrum[index + 1] * ir_spectrum[index];
    }
    let expected = execute_ifft_f32(&product, BLOCK * 2, 0).expect("valid product");

    let mut history = AlignedBytes([0; 16384]);
    let mut input_pcm = AlignedBytes([0; 16384]);
    write_f32_values(input_pcm.0.as_mut_ptr() as u64, &input);
    let mut ir_blocks = AlignedBytes([0; 16384]);
    write_f32_values(
        ir_blocks.0.as_mut_ptr() as u64,
        &ir_spectrum[..RUNTIME_VALUES],
    );
    let mut temp = AlignedBytes([0; 16384]);
    let mut output_pcm = AlignedBytes([0; 16384]);
    let job = ConvolutionJob {
        sharing: ConvolutionSharing::SharedInput,
        version2: false,
        inputs: vec![ConvolutionInput {
            address: history.0.as_mut_ptr() as u64,
            block_size: BLOCK,
            block_count: 1,
            format: AcmFormat::Fp32,
            history_position: 0,
            history_count: 0,
            history_null_count: 0,
            channels: vec![ConvChannel {
                blocks: history.0.as_mut_ptr() as u64,
                sizes: 0,
            }],
            pcm: vec![Some(input_pcm.0.as_mut_ptr() as u64)],
        }],
        irs: vec![ConvolutionIr {
            block_size: BLOCK,
            block_count: 1,
            format: AcmFormat::Fp32,
            channels: vec![ConvChannel {
                blocks: ir_blocks.0.as_mut_ptr() as u64,
                sizes: 0,
            }],
            sizes: vec![Some(vec![ConvBlockSize {
                size_memory: (RUNTIME_VALUES * 4) as u16,
                size: (RUNTIME_VALUES * 4) as u16,
            }])],
            block_gains: vec![1.0],
        }],
        gains: vec![1.0],
        outputs: vec![ConvolutionOutput {
            address: temp.0.as_mut_ptr() as u64,
            block_size: BLOCK,
            temps: vec![temp.0.as_mut_ptr() as u64],
            pcm: vec![output_pcm.0.as_mut_ptr() as u64],
            initial_pcm: vec![None],
            previous_gain: 1.0,
            gain: 1.0,
        }],
    };
    apply(&execute(&job).expect("compressed IR execution"));
    let actual = read_f32_values(output_pcm.0.as_mut_ptr() as u64, BLOCK);
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*actual - expected).abs() < 1.0e-5,
            "sample={index} actual={actual} expected={expected}"
        );
    }
}

#[test]
fn v2_tail_crossfade_and_initial_mix_follow_reference_state() {
    const BLOCK: usize = 256;
    let mut history = AlignedBytes([0; 16384]);
    let mut input_pcm = AlignedBytes([0; 16384]);
    write_f32_values(input_pcm.0.as_mut_ptr() as u64, &[1.0; BLOCK]);

    let mut ir_blocks = AlignedBytes([0; 16384]);
    let mut first_ir = vec![0.0_f32; BLOCK];
    first_ir[0] = 1.0;
    let mut second_ir = vec![0.0_f32; BLOCK];
    second_ir[0] = 0.5;
    let first_spectrum = execute_fft_f32(&first_ir, BLOCK * 2, 1).expect("valid first IR");
    let second_spectrum = execute_fft_f32(&second_ir, BLOCK * 2, 1).expect("valid second IR");
    write_f32_values(ir_blocks.0.as_mut_ptr() as u64, &first_spectrum);
    write_f32_values(
        ir_blocks.0.as_mut_ptr() as u64 + (BLOCK * 2 * 4) as u64,
        &second_spectrum,
    );

    let mut temp = AlignedBytes([0; 16384]);
    let mut output_pcm = AlignedBytes([0; 16384]);
    let mut initial_pcm = AlignedBytes([0; 16384]);
    write_f32_values(initial_pcm.0.as_mut_ptr() as u64, &[0.5; BLOCK]);
    let mut input_state = ConvInV2 {
        base: ConvIn::default(),
        history_null_count: 0,
        reserved: [0; 6],
        tail_padding: 0,
    };
    let mut output_state = ConvOutV2 {
        base: ConvOut::default(),
        gain: 0.75,
        previous_gain: 0.25,
        pcm_init: 0,
        reserved: [0; 4],
    };
    let mut job = ConvolutionJob {
        sharing: ConvolutionSharing::SharedInput,
        version2: true,
        inputs: vec![ConvolutionInput {
            address: std::ptr::from_mut(&mut input_state) as u64,
            block_size: BLOCK,
            block_count: 2,
            format: AcmFormat::Fp32,
            history_position: 0,
            history_count: 0,
            history_null_count: 0,
            channels: vec![ConvChannel {
                blocks: history.0.as_mut_ptr() as u64,
                sizes: 0,
            }],
            pcm: vec![Some(input_pcm.0.as_mut_ptr() as u64)],
        }],
        irs: vec![ConvolutionIr {
            block_size: BLOCK,
            block_count: 2,
            format: AcmFormat::Fp32,
            channels: vec![ConvChannel {
                blocks: ir_blocks.0.as_mut_ptr() as u64,
                sizes: 0,
            }],
            sizes: vec![None],
            block_gains: vec![1.0, 1.0],
        }],
        gains: vec![1.0],
        outputs: vec![ConvolutionOutput {
            address: std::ptr::from_mut(&mut output_state) as u64,
            block_size: BLOCK,
            temps: vec![temp.0.as_mut_ptr() as u64],
            pcm: vec![output_pcm.0.as_mut_ptr() as u64],
            initial_pcm: vec![Some(initial_pcm.0.as_mut_ptr() as u64)],
            previous_gain: 0.25,
            gain: 0.75,
        }],
    };

    apply(&execute(&job).expect("live convolution block"));
    let live = read_f32_values(output_pcm.0.as_mut_ptr() as u64, BLOCK);
    assert!((live[0] - 0.75).abs() < 1.0e-4);
    assert!((live[BLOCK - 1] - 1.25).abs() < 1.0e-4);
    assert_eq!(input_state.base.history_position, 0);
    assert_eq!(input_state.base.history_count, 1);
    assert_eq!(input_state.history_null_count, 0);
    assert_eq!(output_state.previous_gain, 0.75);

    job.inputs[0].history_position = input_state.base.history_position as usize;
    job.inputs[0].history_count = input_state.base.history_count as usize;
    job.inputs[0].history_null_count = input_state.history_null_count as usize;
    job.inputs[0].pcm[0] = None;
    job.outputs[0].previous_gain = output_state.previous_gain;
    apply(&execute(&job).expect("null tail block"));
    let tail = read_f32_values(output_pcm.0.as_mut_ptr() as u64, BLOCK);
    assert!(tail.iter().any(|sample| (*sample - 0.5).abs() > 1.0e-4));
    assert!((tail[0] - 0.875).abs() < 1.0e-4);
    assert_eq!(input_state.base.history_position, 1);
    assert_eq!(input_state.base.history_count, 2);
    assert_eq!(input_state.history_null_count, 1);

    job.inputs[0].history_position = input_state.base.history_position as usize;
    job.inputs[0].history_count = input_state.base.history_count as usize;
    job.inputs[0].history_null_count = input_state.history_null_count as usize;
    apply(&execute(&job).expect("fully null history block"));
    let exhausted_tail = read_f32_values(output_pcm.0.as_mut_ptr() as u64, BLOCK);
    assert!(
        exhausted_tail
            .iter()
            .all(|sample| (*sample - 0.5).abs() < 1.0e-4)
    );
    assert_eq!(input_state.history_null_count, 2);

    job.inputs[0].history_position = input_state.base.history_position as usize;
    job.inputs[0].history_count = input_state.base.history_count as usize;
    job.inputs[0].history_null_count = input_state.history_null_count as usize;
    apply(&execute(&job).expect("null history count saturates"));
    assert_eq!(input_state.history_null_count, 2);

    job.inputs[0].history_position = input_state.base.history_position as usize;
    job.inputs[0].history_count = input_state.base.history_count as usize;
    job.inputs[0].history_null_count = input_state.history_null_count as usize;
    job.inputs[0].pcm[0] = Some(input_pcm.0.as_mut_ptr() as u64);
    assert_eq!(execute(&job), Err(AcmError::InvalidParameter));
}
