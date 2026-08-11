// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! CPU implementations of ACM DSP functions.

mod fft;

pub(super) use fft::execute;

#[cfg(test)]
mod fft_tests;
