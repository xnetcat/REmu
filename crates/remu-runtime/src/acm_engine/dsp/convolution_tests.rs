// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::mem::size_of;

use super::convolution::validate_channel_graph;
use crate::acm_engine::abi::{
    AcmError, ConvBlockSize, ConvChannel, ConvIn, ConvInV2, ConvIr, ConvOut, ConvOutV2,
    SceAcmBatchInfo,
};
use crate::acm_engine::batch::{Command, decode_buffers};
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
