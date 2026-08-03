// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Process-relative startup milestones shared by the loader, runtime, and GPU backend.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static STARTUP_EPOCH: OnceLock<Instant> = OnceLock::new();
static RECORDED_MILESTONES: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum StartupMilestone {
    CliRunStart,
    ProcessImageLinked,
    EventSinkInstalled,
    PresenterEnter,
    PresenterResourcesReady,
    WindowReady,
    GuestProbeEntry,
    GuestRuntimeConfigured,
    ModuleInitializersComplete,
    GuestEntryBegin,
    FirstQueueSubmission,
    FirstFlipAccepted,
    FirstEncodedDraw,
    FirstGuestPresentSubmitted,
}

impl StartupMilestone {
    const fn name(self) -> &'static str {
        match self {
            Self::CliRunStart => "cli-run-start",
            Self::ProcessImageLinked => "process-image-linked",
            Self::EventSinkInstalled => "event-sink-installed",
            Self::PresenterEnter => "presenter-enter",
            Self::PresenterResourcesReady => "presenter-resources-ready",
            Self::WindowReady => "window-ready",
            Self::GuestProbeEntry => "guest-probe-entry",
            Self::GuestRuntimeConfigured => "guest-runtime-configured",
            Self::ModuleInitializersComplete => "module-initializers-complete",
            Self::GuestEntryBegin => "guest-entry-begin",
            Self::FirstQueueSubmission => "first-queue-submission",
            Self::FirstFlipAccepted => "first-flip-accepted",
            Self::FirstEncodedDraw => "first-encoded-draw",
            Self::FirstGuestPresentSubmitted => "first-guest-present-submitted",
        }
    }

    const fn mask(self) -> u64 {
        1_u64 << (self as u8)
    }
}

/// Begins the one-shot startup timeline for this process.
pub fn begin_startup_timeline() {
    let epoch = STARTUP_EPOCH.get_or_init(Instant::now);
    RECORDED_MILESTONES.store(0, Ordering::Release);
    record_startup_milestone_at(StartupMilestone::CliRunStart, epoch.elapsed().as_micros());
}

/// Records and prints the first observation of a startup milestone.
pub fn record_startup_milestone(milestone: StartupMilestone) {
    let Some(epoch) = STARTUP_EPOCH.get() else {
        return;
    };
    record_startup_milestone_at(milestone, epoch.elapsed().as_micros());
}

fn record_startup_milestone_at(milestone: StartupMilestone, elapsed_us: u128) {
    let previous = RECORDED_MILESTONES.fetch_or(milestone.mask(), Ordering::AcqRel);
    if previous & milestone.mask() != 0 {
        return;
    }
    eprintln!(
        "[REMu][PERF-STARTUP] milestone={} elapsed_us={elapsed_us}",
        milestone.name(),
    );
}

#[cfg(test)]
mod tests {
    use super::{RECORDED_MILESTONES, StartupMilestone, record_startup_milestone_at};
    use std::sync::atomic::Ordering;

    #[test]
    fn startup_milestones_are_recorded_once_per_kind() {
        RECORDED_MILESTONES.store(0, Ordering::Release);
        record_startup_milestone_at(StartupMilestone::FirstEncodedDraw, 12_345);
        record_startup_milestone_at(StartupMilestone::FirstEncodedDraw, 67_890);
        assert_eq!(
            RECORDED_MILESTONES.load(Ordering::Acquire),
            StartupMilestone::FirstEncodedDraw.mask()
        );
    }
}
