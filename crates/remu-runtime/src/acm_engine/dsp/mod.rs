// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! CPU implementations of ACM DSP functions.

mod fft;
mod panner;

pub(super) use fft::execute as execute_fft;
pub(super) use panner::execute as execute_panner;

#[cfg(test)]
mod fft_tests;

#[cfg(test)]
mod panner_tests;
