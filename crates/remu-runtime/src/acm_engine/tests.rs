// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::mem::size_of;

use super::AcmEngine;
use super::abi::{AcmError, AcmFormat, SceAcmBatchCommand, SceAcmBatchInfo, return_error};

const SUCCESS: u64 = 0;

fn args(values: &[u64]) -> [u64; 6] {
    let mut arguments = [0_u64; 6];
    arguments[..values.len()].copy_from_slice(values);
    arguments
}

fn address_mut<T>(value: &mut T) -> u64 {
    std::ptr::from_mut(value) as u64
}

#[test]
fn sdk10_abi_sizes_formats_and_error_words_are_exact() {
    assert_eq!(size_of::<SceAcmBatchInfo>(), 24);
    assert_eq!(size_of::<SceAcmBatchCommand>(), 8);
    assert_eq!(AcmFormat::Fp32 as u32, 0);
    assert_eq!(AcmFormat::Fp16 as u32, 1);
    assert_eq!(
        [
            AcmError::Unknown as u32,
            AcmError::InvalidContext as u32,
            AcmError::InvalidBatch as u32,
            AcmError::OutOfMemory as u32,
            AcmError::OutOfResources as u32,
            AcmError::InvalidParameter as u32,
            AcmError::MalformedBatch as u32,
            AcmError::Busy as u32,
            AcmError::BatchBufferFull as u32,
            AcmError::Alignment as u32,
            AcmError::PageFault as u32,
        ],
        [
            0x8194_0001,
            0x8194_0002,
            0x8194_0003,
            0x8194_0004,
            0x8194_0005,
            0x8194_0006,
            0x8194_0007,
            0x8194_0008,
            0x8194_0009,
            0x8194_000a,
            0x8194_000b,
        ]
    );
    assert_eq!(
        return_error(AcmError::InvalidParameter),
        0xffff_ffff_8194_0006
    );
}

#[test]
fn context_ids_are_checked_and_reset_is_process_local() {
    let engine = AcmEngine::new();
    let mut first = 0_u32;
    let mut second = 0_u32;
    assert_eq!(
        engine.context_create(args(&[address_mut(&mut first)])),
        SUCCESS
    );
    assert_eq!(
        engine.context_create(args(&[address_mut(&mut second)])),
        SUCCESS
    );
    assert_eq!((first, second), (1, 2));
    assert_eq!(
        engine.context_create(args(&[0])),
        return_error(AcmError::InvalidParameter)
    );
    assert_eq!(engine.context_destroy(args(&[u64::from(first)])), SUCCESS);
    assert_eq!(
        engine.context_destroy(args(&[u64::from(first)])),
        return_error(AcmError::InvalidContext)
    );

    engine.reset();
    assert_eq!(
        engine.context_destroy(args(&[u64::from(second)])),
        return_error(AcmError::InvalidContext)
    );
    let mut after_reset = 0_u32;
    assert_eq!(
        engine.context_create(args(&[address_mut(&mut after_reset)])),
        SUCCESS
    );
    assert_eq!(after_reset, 1);
}
