// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Public LP64 layouts and error values from SDK 10 `acm.h` and `acm/error.h`.

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SceAcmBatchInfo {
    pub(crate) buffer: u64,
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SceAcmBatchCommand(pub(crate) u64);

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AcmFormat {
    Fp32 = 0,
    Fp16 = 1,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AcmError {
    Unknown = 0x8194_0001,
    InvalidContext = 0x8194_0002,
    InvalidBatch = 0x8194_0003,
    OutOfMemory = 0x8194_0004,
    OutOfResources = 0x8194_0005,
    InvalidParameter = 0x8194_0006,
    MalformedBatch = 0x8194_0007,
    Busy = 0x8194_0008,
    BatchBufferFull = 0x8194_0009,
    Alignment = 0x8194_000a,
    PageFault = 0x8194_000b,
}

pub(crate) fn return_error(error: AcmError) -> u64 {
    crate::sce_kernel_error(error as u32)
}

const _: () = {
    assert!(std::mem::size_of::<SceAcmBatchInfo>() == 24);
    assert!(std::mem::offset_of!(SceAcmBatchInfo, buffer) == 0);
    assert!(std::mem::offset_of!(SceAcmBatchInfo, offset) == 8);
    assert!(std::mem::offset_of!(SceAcmBatchInfo, size) == 16);
    assert!(std::mem::size_of::<SceAcmBatchCommand>() == 8);
};
