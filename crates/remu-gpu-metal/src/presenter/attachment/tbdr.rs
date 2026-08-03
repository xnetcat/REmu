// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! B4 TBDR hygiene for attachment store actions and memoryless MSAA eligibility.
//!
//! Plan (MASTER-PLAN B4 / presenter-render R3):
//! - consumer-derived depth store actions (default DontCare when no later consumer)
//! - memoryless MSAA once draws/pass > 1.2 and the MSAA texture is not re-entered
//!
//! Usage-flag drops (`ShaderWrite`/`PixelFormatView`) and `Private`+blit uploads live in
//! texture/residency (B4 texture batch owns those files) — this module only owns the
//! **attachment** store-action / MSAA policy used by `execute.rs`.
//!
//! Memoryless MSAA allocation is gate-dependent: when the draws/pass ratio is not proven, we
//! keep `StoreAndMultisampleResolve` on a Private MSAA texture and log the seam rather than
//! silently dropping samples across pass boundaries.

use metal::MTLStoreAction;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Measured threshold from the Poppy baseline (presenter-render §3.1 / MASTER-PLAN B4):
/// memoryless MSAA is only sound once draws per pass exceeds this ratio.
pub(crate) const MEMORYLESS_MSAA_MIN_DRAWS_PER_PASS: f64 = 1.2;

/// Whether a later consumer of the depth attachment is known at encode time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // SampledOrCpuRead / None await in-pass consumer lookahead (NOTES seam).
pub(crate) enum DepthConsumerHint {
    /// No proven later sample / CPU-read / HTILE consumer — TBDR can discard.
    None,
    /// Depth will be sampled, CPU-read, or its HTILE base is touched later.
    SampledOrCpuRead,
    /// Lookahead not available; keep Store rather than risk a silent shadow-pass break.
    UnknownKeepStore,
}

/// Color / depth attachment store policy after resolving MSAA and consumer hints.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AttachmentStorePolicy {
    Store,
    DontCare,
    StoreAndMultisampleResolve,
    MultisampleResolve,
}

impl AttachmentStorePolicy {
    pub(crate) fn metal(self) -> MTLStoreAction {
        match self {
            Self::Store => MTLStoreAction::Store,
            Self::DontCare => MTLStoreAction::DontCare,
            Self::StoreAndMultisampleResolve => MTLStoreAction::StoreAndMultisampleResolve,
            Self::MultisampleResolve => MTLStoreAction::MultisampleResolve,
        }
    }
}

/// Memoryless MSAA decision for one pass.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum MsaaMemorylessDecision {
    /// Draws/pass proven above the threshold and MSAA not re-entered — Memoryless + Resolve.
    Eligible { draws_per_pass: f64 },
    /// Keep Private MSAA + StoreAndMultisampleResolve; log the seam.
    Refused {
        reason: &'static str,
        draws_per_pass: f64,
    },
}

/// Depth store action from the consumer hint (plan-B4 consumer-derived depth stores).
pub(crate) fn depth_store_action(
    has_resolve: bool,
    consumer: DepthConsumerHint,
) -> AttachmentStorePolicy {
    if has_resolve {
        // Depth MSAA resolve still needs the resolve destination stored; sample storage follows
        // the memoryless decision separately.
        return AttachmentStorePolicy::StoreAndMultisampleResolve;
    }
    match consumer {
        DepthConsumerHint::None => AttachmentStorePolicy::DontCare,
        DepthConsumerHint::SampledOrCpuRead | DepthConsumerHint::UnknownKeepStore => {
            AttachmentStorePolicy::Store
        }
    }
}

/// Color store action. Resolve path selects MultisampleResolve only when memoryless is eligible.
pub(crate) fn color_store_action(
    has_resolve: bool,
    memoryless: &MsaaMemorylessDecision,
) -> AttachmentStorePolicy {
    if !has_resolve {
        return AttachmentStorePolicy::Store;
    }
    match memoryless {
        MsaaMemorylessDecision::Eligible { .. } => AttachmentStorePolicy::MultisampleResolve,
        MsaaMemorylessDecision::Refused { .. } => AttachmentStorePolicy::StoreAndMultisampleResolve,
    }
}

/// Gate for memoryless MSAA (plan-B4: draws/pass > 1.2 and MSAA texture not re-entered).
pub(crate) fn msaa_memoryless_decision(
    draws_in_pass: u32,
    render_passes: u32,
    msaa_reentered: bool,
) -> MsaaMemorylessDecision {
    if msaa_reentered {
        return refuse(
            "MSAA attachment is re-entered across a pass boundary; Memoryless would drop samples",
            ratio(draws_in_pass, render_passes),
        );
    }
    if render_passes == 0 {
        return refuse(
            "no render-pass denominator for draws/pass gate",
            0.0,
        );
    }
    let draws_per_pass = ratio(draws_in_pass, render_passes);
    if draws_per_pass > MEMORYLESS_MSAA_MIN_DRAWS_PER_PASS {
        MsaaMemorylessDecision::Eligible { draws_per_pass }
    } else {
        refuse(
            "draws/pass at or below 1.2; Memoryless MSAA requires the owner interactive gate to prove pass coalescing",
            draws_per_pass,
        )
    }
}

fn ratio(draws: u32, passes: u32) -> f64 {
    if passes == 0 {
        0.0
    } else {
        f64::from(draws) / f64::from(passes)
    }
}

fn refuse(reason: &'static str, draws_per_pass: f64) -> MsaaMemorylessDecision {
    static REFUSALS: AtomicUsize = AtomicUsize::new(0);
    let sequence = REFUSALS.fetch_add(1, Ordering::Relaxed) + 1;
    if sequence <= 8 || sequence.is_power_of_two() {
        eprintln!(
            "[REMu][METAL][B4] memoryless MSAA refused sequence={sequence} draws_per_pass={draws_per_pass:.3} reason={reason}"
        );
    }
    MsaaMemorylessDecision::Refused {
        reason,
        draws_per_pass,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_without_consumer_is_dont_care() {
        assert_eq!(
            depth_store_action(false, DepthConsumerHint::None),
            AttachmentStorePolicy::DontCare
        );
        // MTLStoreAction is not Eq in this metal crate revision; match the policy enum instead.
        assert!(matches!(
            depth_store_action(false, DepthConsumerHint::None).metal(),
            MTLStoreAction::DontCare
        ));
    }

    #[test]
    fn depth_with_sampled_consumer_stores() {
        assert_eq!(
            depth_store_action(false, DepthConsumerHint::SampledOrCpuRead),
            AttachmentStorePolicy::Store
        );
    }

    #[test]
    fn depth_with_unknown_consumer_keeps_store_loudly() {
        assert_eq!(
            depth_store_action(false, DepthConsumerHint::UnknownKeepStore),
            AttachmentStorePolicy::Store
        );
    }

    #[test]
    fn memoryless_msaa_refuses_below_threshold() {
        // 1.1 draws/pass — the measured TBDR fragmentation regime.
        let decision = msaa_memoryless_decision(11, 10, false);
        assert!(matches!(
            decision,
            MsaaMemorylessDecision::Refused { .. }
        ));
        assert_eq!(
            color_store_action(true, &decision),
            AttachmentStorePolicy::StoreAndMultisampleResolve
        );
    }

    #[test]
    fn memoryless_msaa_eligible_above_threshold_when_not_reentered() {
        let decision = msaa_memoryless_decision(13, 10, false);
        assert!(matches!(
            decision,
            MsaaMemorylessDecision::Eligible { .. }
        ));
        assert_eq!(
            color_store_action(true, &decision),
            AttachmentStorePolicy::MultisampleResolve
        );
    }

    #[test]
    fn memoryless_msaa_refuses_when_reentered() {
        assert!(matches!(
            msaa_memoryless_decision(100, 10, true),
            MsaaMemorylessDecision::Refused { .. }
        ));
    }
}
