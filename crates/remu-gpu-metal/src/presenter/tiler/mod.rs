// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Solo S2 GPU tiler (MASTER-PLAN B2) plus co-owned scaffolding for B3/B5/Q4.
//!
//! Additive module: the CPU tiler in [`super::tiling`] remains the differential oracle.
//! MSL equations are ports of the GFX10 within-block address bit tables that
//! `agc_gpu_address/generated_{5,27}.cpp` specialise into SIMD detile loops — never copy
//! those SDK sources into the tree; cite file:line in NOTES.
//!
//! Gate posture: ship only paths proven byte-exact against the CPU oracle. Unproven
//! production wiring stays a LOUD fallback (see each submodule's seam notes).

mod absolute_address;
mod deferred;
mod end_state;
mod equations;
mod msl;

#[allow(unused_imports)]
// Public surface for residency/readback migration; exercised in submodule tests.
pub(crate) use absolute_address::{
    ABSOLUTE_ADDRESS_POOL_BUFFER_SLOT, AbsoluteAddressPool, AbsoluteAddressPoolBind,
    absolute_address_pool_binding_ready, record_absolute_address_pool_bind,
};
#[allow(unused_imports)]
pub(crate) use deferred::{
    DeferredGuestRead, DeferredGuestReadId, DeferredGuestReadReason, DeferredGuestReadTracker,
    GuestReadbackContinuation,
};
#[allow(unused_imports)]
pub(crate) use end_state::{
    EndStateWriteClass, classify_end_state_write, lease_validated_reconciliation_ready,
    refuse_bookkeeping_writeback,
};
#[allow(unused_imports)]
pub(crate) use equations::{
    GpuTilerMode, mode5_within_block_byte_offset_msl_oracle,
    mode27_within_block_byte_offset_msl_oracle, shift_ceil_u32, tiled_physical_byte_offset,
};
#[allow(unused_imports)]
pub(crate) use msl::{
    MODE5_TILER_MSL, MODE27_TILER_MSL, TilerKernelKind, compile_tiler_library, tiler_kernel_source,
};
