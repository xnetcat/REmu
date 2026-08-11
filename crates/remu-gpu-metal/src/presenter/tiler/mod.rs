// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Solo S2 GPU tiler (MASTER-PLAN B2 / Phase 1G).
//!
//! Production residency rematerialisation, upload, and guest publication encode through
//! [`msl::mode27_tiler_pipelines`] / [`msl::mode5_tiler_pipelines`] /
//! [`msl::mode9_tiler_pipelines`]. The CPU tiler in [`super::tiling`] remains the
//! differential oracle. MSL equations are ports of the GFX10 within-block address bit
//! tables that `agc_gpu_address/generated_{5,27}.cpp` specialise into SIMD detile loops —
//! never copy those SDK sources into the tree. Mode-1/9 use the SW_256_S / SW_64KB_S
//! tables from `blocks.rs`.
//!
//! Dead scaffolding deleted in Phase 1G close-out:
//! - `deferred.rs` (B3 CPU readback tracker) — GPU publish replaces deferred CPU readback
//! - `absolute_address.rs` (Q4 pool bind) — co-owned with shader-metal; not armed in 1G

mod encode;
mod end_state;
mod equations;
mod msl;

pub(crate) use encode::{
    GpuTilerView, encode_gpu_publish_to_guest_view, encode_gpu_upload_from_guest,
    encode_gpu_upload_from_guest_view, gpu_tiler_owns_descriptor,
};
pub(crate) use end_state::{
    EndStateWriteClass, classify_end_state_write, refuse_bookkeeping_writeback,
};
#[allow(unused_imports)] // Differential-oracle helpers for tiler encode tests.
pub(crate) use equations::{
    GpuTilerMode, mode5_within_block_byte_offset_msl_oracle,
    mode9_within_block_byte_offset_msl_oracle, mode27_within_block_byte_offset_msl_oracle,
    shift_ceil_u32, tiled_physical_byte_offset,
};
pub(crate) use msl::{mode9_tiler_pipelines, mode27_tiler_pipelines};
