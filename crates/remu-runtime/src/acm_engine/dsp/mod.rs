// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! CPU implementations of ACM DSP functions.

mod convolution;
mod fft;
mod panner;

pub(super) use convolution::{
    execute as execute_convolution, record_words as convolution_record_words,
    snapshot_job as snapshot_convolution,
};
pub(super) use fft::execute as execute_fft;
pub(super) use panner::execute as execute_panner;

#[cfg(test)]
mod fft_tests;

#[cfg(test)]
mod panner_tests;

#[cfg(test)]
mod convolution_tests;
