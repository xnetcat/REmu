// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Public LP64 layouts and error values from SDK 10 `acm.h` and `acm/error.h`.

use std::mem::{MaybeUninit, align_of, size_of};

use crate::{process_guest_range_has_read_access, read_guest_bytes, write_guest_bytes};

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

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ConvBlockSize {
    pub(crate) size_memory: u16,
    pub(crate) size: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ConvChannel {
    pub(crate) blocks: u64,
    pub(crate) sizes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ConvIn {
    pub(crate) block_size: u32,
    pub(crate) block_count: u32,
    pub(crate) channel_count: u32,
    pub(crate) format: u32,
    pub(crate) history_position: u32,
    pub(crate) history_count: u32,
    pub(crate) channels: u64,
    pub(crate) pcm: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ConvInV2 {
    pub(crate) base: ConvIn,
    pub(crate) history_null_count: u32,
    pub(crate) reserved: [u32; 6],
    pub(crate) tail_padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ConvIr {
    pub(crate) block_size: u32,
    pub(crate) block_count: u32,
    pub(crate) channel_count: u32,
    pub(crate) format: u32,
    pub(crate) block_gains: u64,
    pub(crate) channels: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ConvOut {
    pub(crate) block_size: u32,
    pub(crate) channel_count: u32,
    pub(crate) temps: u64,
    pub(crate) pcm: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ConvOutV2 {
    pub(crate) base: ConvOut,
    pub(crate) gain: f32,
    pub(crate) previous_gain: f32,
    pub(crate) pcm_init: u64,
    pub(crate) reserved: [u32; 4],
}

/// Marker for ABI records whose complete bit pattern can be copied to and from guest memory.
///
/// # Safety
///
/// Implementers must be `Copy`, contain no uninitialized padding, and accept every bit pattern.
pub(crate) unsafe trait GuestPod: Copy {}

// SAFETY: both layouts contain only fixed-width unsigned integers and have no padding.
unsafe impl GuestPod for SceAcmBatchInfo {}
// SAFETY: the transparent wrapper has exactly the validity and layout of `u64`.
unsafe impl GuestPod for SceAcmBatchCommand {}
// SAFETY: fixed-width unsigned integers accept every bit pattern.
unsafe impl GuestPod for u64 {}
// SAFETY: these SDK records contain only fixed-width integers/pointers with initialized padding.
unsafe impl GuestPod for ConvBlockSize {}
// SAFETY: two `u64` fields have no padding and accept every bit pattern.
unsafe impl GuestPod for ConvChannel {}
// SAFETY: the LP64 record contains only integer fields and has no padding.
unsafe impl GuestPod for ConvIn {}
// SAFETY: the v2 tail and alignment padding are explicit initialized integer fields.
unsafe impl GuestPod for ConvInV2 {}
// SAFETY: the LP64 record contains only integer fields and has no padding.
unsafe impl GuestPod for ConvIr {}
// SAFETY: the LP64 record contains only integer fields and has no padding.
unsafe impl GuestPod for ConvOut {}
// SAFETY: every field accepts arbitrary bits; the explicit tail fills the LP64 layout.
unsafe impl GuestPod for ConvOutV2 {}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AcmFormat {
    Fp32 = 0,
    Fp16 = 1,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
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

pub(crate) fn checked_span(address: u64, length: usize, alignment: usize) -> Result<u64, AcmError> {
    if address == 0 || alignment == 0 || !address.is_multiple_of(alignment as u64) {
        return Err(AcmError::InvalidParameter);
    }
    let length_u64 = u64::try_from(length).map_err(|_| AcmError::InvalidParameter)?;
    address
        .checked_add(length_u64)
        .ok_or(AcmError::InvalidParameter)?;
    if !process_guest_range_has_read_access(address, length) {
        return Err(AcmError::PageFault);
    }
    Ok(address)
}

pub(crate) fn read_pod<T: GuestPod>(address: u64) -> Result<T, AcmError> {
    checked_span(address, size_of::<T>(), align_of::<T>())?;
    let mut value = MaybeUninit::<T>::uninit();
    // SAFETY: `GuestPod` guarantees that all bytes and bit patterns of `T` are initialized and
    // valid. `value` provides exactly `size_of::<T>()` writable bytes.
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(value.as_mut_ptr().cast::<u8>(), size_of::<T>()) };
    read_guest_bytes(address, bytes);
    // SAFETY: every byte was initialized above and `GuestPod` accepts every bit pattern.
    Ok(unsafe { value.assume_init() })
}

pub(crate) fn write_pod<T: GuestPod>(address: u64, value: &T) -> Result<(), AcmError> {
    checked_span(address, size_of::<T>(), align_of::<T>())?;
    // SAFETY: `GuestPod` excludes padding, so every byte in the value may be observed.
    let bytes = unsafe {
        std::slice::from_raw_parts(std::ptr::from_ref(value).cast::<u8>(), size_of::<T>())
    };
    write_guest_bytes(address, bytes);
    Ok(())
}

pub(crate) fn read_array<T: GuestPod>(address: u64, count: usize) -> Result<Vec<T>, AcmError> {
    let bytes = size_of::<T>()
        .checked_mul(count)
        .ok_or(AcmError::InvalidParameter)?;
    checked_span(address, bytes, align_of::<T>())?;
    (0..count)
        .map(|index| {
            let offset = size_of::<T>()
                .checked_mul(index)
                .and_then(|offset| u64::try_from(offset).ok())
                .ok_or(AcmError::InvalidParameter)?;
            read_pod(address + offset)
        })
        .collect()
}

const _: () = {
    assert!(std::mem::size_of::<SceAcmBatchInfo>() == 24);
    assert!(std::mem::offset_of!(SceAcmBatchInfo, buffer) == 0);
    assert!(std::mem::offset_of!(SceAcmBatchInfo, offset) == 8);
    assert!(std::mem::offset_of!(SceAcmBatchInfo, size) == 16);
    assert!(std::mem::size_of::<SceAcmBatchCommand>() == 8);
    assert!(std::mem::size_of::<ConvChannel>() == 16);
    assert!(std::mem::size_of::<ConvBlockSize>() == 4);
    assert!(std::mem::size_of::<ConvIn>() == 40);
    assert!(std::mem::size_of::<ConvInV2>() == 72);
    assert!(std::mem::size_of::<ConvIr>() == 32);
    assert!(std::mem::size_of::<ConvOut>() == 24);
    assert!(std::mem::size_of::<ConvOutV2>() == 56);
};
