// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! The one display timebase (plan E1).
//!
//! The presenter's display link stamps every hardware vblank here; guest vblank waits, the
//! vblank counter, and presentation pacing all derive from these stamps instead of private
//! synthetic clocks. Without a presenter (headless, tests) no sample ever exists and every
//! consumer keeps its synthetic fallback, so publishing here is strictly additive.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::time::Instant;

static VBLANK_COUNT: AtomicU64 = AtomicU64::new(0);
static LAST_VBLANK_NANOS: AtomicU64 = AtomicU64::new(0);
static VBLANK_PERIOD_NANOS: AtomicU64 = AtomicU64::new(0);

/// Nanoseconds on the process-wide monotonic clock every timebase stamp and query shares.
#[must_use]
pub fn monotonic_nanos() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let epoch = *EPOCH.get_or_init(Instant::now);
    u64::try_from(epoch.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Display pacing source stamped into `run.json` (Phase 0 evidence hygiene).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum DisplayMode {
    /// No link has reported a mode yet.
    Unknown = 0,
    /// CoreVideo `CVDisplayLink` is driving the vblank timebase.
    CvDisplayLink = 1,
    /// Link unavailable or refused; pacing falls back to a synthetic period.
    Synthetic = 2,
}

static DISPLAY_MODE: AtomicU8 = AtomicU8::new(DisplayMode::Unknown as u8);

/// Records how presentation is paced for this process.
pub fn set_display_mode(mode: DisplayMode) {
    DISPLAY_MODE.store(mode as u8, Ordering::Relaxed);
}

/// Returns the current display pacing mode label for `run.json`.
pub fn display_mode_label() -> &'static str {
    match DISPLAY_MODE.load(Ordering::Relaxed) {
        x if x == DisplayMode::CvDisplayLink as u8 => "cvdisplaylink",
        x if x == DisplayMode::Synthetic as u8 => "synthetic",
        _ => "unknown",
    }
}

/// Records one hardware vblank observed by the display link.
///
/// `timestamp_nanos` is [`monotonic_nanos`] at the vblank; `period_nanos` is the display's
/// current refresh period. Ordering makes the count/stamp pair coherent enough for pacing:
/// a reader may observe a torn pair only one vblank apart, which every consumer tolerates.
pub fn record_display_vblank(timestamp_nanos: u64, period_nanos: u64) {
    if period_nanos != 0 {
        VBLANK_PERIOD_NANOS.store(period_nanos, Ordering::Relaxed);
    }
    LAST_VBLANK_NANOS.store(timestamp_nanos, Ordering::Relaxed);
    VBLANK_COUNT.fetch_add(1, Ordering::Release);
}

/// One coherent-enough view of the timebase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DisplayVblankSample {
    /// Hardware vblanks stamped since the presenter started its link.
    pub count: u64,
    /// [`monotonic_nanos`] of the most recent stamped vblank.
    pub last_timestamp_nanos: u64,
    /// The display's refresh period at that vblank.
    pub period_nanos: u64,
}

/// Returns the current timebase, or `None` until a display link has stamped a vblank.
#[must_use]
pub fn display_vblank_sample() -> Option<DisplayVblankSample> {
    let count = VBLANK_COUNT.load(Ordering::Acquire);
    if count == 0 {
        return None;
    }
    let period_nanos = VBLANK_PERIOD_NANOS.load(Ordering::Relaxed);
    if period_nanos == 0 {
        return None;
    }
    Some(DisplayVblankSample {
        count,
        last_timestamp_nanos: LAST_VBLANK_NANOS.load(Ordering::Relaxed),
        period_nanos,
    })
}

impl DisplayVblankSample {
    /// Whether the link is still stamping: a display link pauses when its display sleeps or
    /// the window is fully occluded, and a stale timebase must not serve vblank waits.
    #[must_use]
    pub fn is_live_at(&self, now_nanos: u64) -> bool {
        now_nanos.saturating_sub(self.last_timestamp_nanos) < self.period_nanos.saturating_mul(4)
    }

    /// Nanoseconds from `now_nanos` until the next expected vblank.
    #[must_use]
    pub fn nanos_until_next_vblank(&self, now_nanos: u64) -> u64 {
        let since = now_nanos.saturating_sub(self.last_timestamp_nanos);
        let period = self.period_nanos.max(1);
        period - (since % period)
    }
}

#[cfg(test)]
mod tests {
    use super::DisplayVblankSample;

    #[test]
    fn the_next_vblank_is_predicted_from_the_last_stamp_and_period() {
        let sample = DisplayVblankSample {
            count: 100,
            last_timestamp_nanos: 1_000_000,
            period_nanos: 16_666_667,
        };
        assert_eq!(sample.nanos_until_next_vblank(1_000_000), 16_666_667);
        assert_eq!(sample.nanos_until_next_vblank(9_000_000), 8_666_667);
        // A query later than one period folds into the following interval.
        assert_eq!(
            sample.nanos_until_next_vblank(1_000_000 + 2 * 16_666_667 + 5),
            16_666_662
        );
    }

    #[test]
    fn a_paused_link_reports_stale_after_four_periods() {
        let sample = DisplayVblankSample {
            count: 7,
            last_timestamp_nanos: 0,
            period_nanos: 16_666_667,
        };
        assert!(sample.is_live_at(3 * 16_666_667));
        assert!(!sample.is_live_at(4 * 16_666_667 + 1));
    }
}
