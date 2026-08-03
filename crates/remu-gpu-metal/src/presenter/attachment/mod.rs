// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Batch B5: attachment / MSAA / occlusion seams for remu-gpu-metal.
//!
//! Scope boundaries (NOTES Batch B4/B5):
//! - Owns store-action / memoryless-MSAA policy, attachment zero-copy eligibility, and
//!   `MTLVisibilityResultBuffer` publication for `GpuEvent::OcclusionQuery`.
//! - Does **not** edit texture residency / write_watch / cache (B4) or present/window (K3).
//! - Gate-dependent paths stop at a loud fallback rather than shipping silently-wrong pixels.

mod occlusion;
mod tbdr;
mod zero_copy;

pub(crate) use self::occlusion::{
    arm_visibility_counting, attach_visibility_result_buffer, ensure_visibility_result_buffer,
    occlusion_counter_releases, publish_occlusion_query_to_guest, visibility_result_buffer_byte_count,
};
pub(crate) use self::tbdr::{
    AttachmentStorePolicy, DepthConsumerHint, MsaaMemorylessDecision, MEMORYLESS_MSAA_MIN_DRAWS_PER_PASS,
    color_store_action, depth_store_action, msaa_memoryless_decision,
};
pub(crate) use self::zero_copy::{
    AttachmentZeroCopyDecision, attachment_zero_copy_decision, refuse_unproven_attachment_zero_copy,
};
