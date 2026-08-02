// SPDX-License-Identifier: GPL-2.0-or-later

//! Allocation-free runtime-dispatch telemetry shared with the presenter sampler.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Upper bounds for the first seven logarithmic latency buckets, in nanoseconds.
///
/// The eighth bucket contains values greater than one second. Keeping the boundaries in the public
/// sample contract lets structured telemetry serialize the histogram without implying units or
/// relying on a log-label convention.
pub const DISPATCH_LATENCY_BUCKET_UPPER_NS: [u64; 7] = [
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
];
pub const DISPATCH_LATENCY_BUCKET_COUNT: usize = DISPATCH_LATENCY_BUCKET_UPPER_NS.len() + 1;

/// Mutually exclusive runtime routes used to serve one guest import.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum DispatchImportFamily {
    /// A link-time binding to a leaf runtime function, bypassing the captured-frame dispatcher.
    Direct = 0,
    /// Yield, clocks, numeric helpers and other lock-free leaf operations.
    Leaf,
    /// Guest mutex, condition, rwlock, semaphore and event-flag operations.
    GuestSync,
    /// `AudioOut` and `AudioOut2` service-local operations.
    Audio,
    /// Controller service-local operations.
    Pad,
    /// The shared stateless dispatch tables.
    Stateless,
    /// File-descriptor operations with their own state lock.
    KernelFd,
    /// Blocking kernel equeue waits served outside the process dispatch lock.
    Equeue,
    /// Dynamic-module loading and initializers.
    Loader,
    /// Ordered APR/AMPR operations.
    Ampr,
    /// Imports served while holding the process-wide dispatch-state lock.
    Stateful,
}

pub const DISPATCH_IMPORT_FAMILY_COUNT: usize = DispatchImportFamily::Stateful as usize + 1;

/// One reset interval of latency observations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DispatchLatencyHistogram {
    pub count: u64,
    pub total_ns: u64,
    pub maximum_ns: u64,
    pub buckets: [u64; DISPATCH_LATENCY_BUCKET_COUNT],
}

/// One reset interval of runtime-dispatch observations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DispatchTelemetrySample {
    pub import_calls: [u64; DISPATCH_IMPORT_FAMILY_COUNT],
    pub dispatch_lock_acquire_wait: DispatchLatencyHistogram,
    pub dispatch_lock_hold: DispatchLatencyHistogram,
    pub gpu_publication_wait: DispatchLatencyHistogram,
}

impl DispatchTelemetrySample {
    #[must_use]
    pub fn import_calls_for(&self, family: DispatchImportFamily) -> u64 {
        self.import_calls[family as usize]
    }

    #[must_use]
    pub fn total_import_calls(&self) -> u64 {
        self.import_calls
            .iter()
            .copied()
            .fold(0, u64::saturating_add)
    }
}

struct AtomicLatencyHistogram {
    count: AtomicU64,
    total_ns: AtomicU64,
    maximum_ns: AtomicU64,
    buckets: [AtomicU64; DISPATCH_LATENCY_BUCKET_COUNT],
}

impl AtomicLatencyHistogram {
    const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_ns: AtomicU64::new(0),
            maximum_ns: AtomicU64::new(0),
            buckets: [const { AtomicU64::new(0) }; DISPATCH_LATENCY_BUCKET_COUNT],
        }
    }

    fn record(&self, duration: Duration) {
        let nanoseconds = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        let bucket = DISPATCH_LATENCY_BUCKET_UPPER_NS
            .iter()
            .position(|upper| nanoseconds <= *upper)
            .unwrap_or(DISPATCH_LATENCY_BUCKET_COUNT - 1);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_ns.fetch_add(nanoseconds, Ordering::Relaxed);
        self.maximum_ns.fetch_max(nanoseconds, Ordering::Relaxed);
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
    }

    fn take(&self) -> DispatchLatencyHistogram {
        DispatchLatencyHistogram {
            count: self.count.swap(0, Ordering::Relaxed),
            total_ns: self.total_ns.swap(0, Ordering::Relaxed),
            maximum_ns: self.maximum_ns.swap(0, Ordering::Relaxed),
            buckets: std::array::from_fn(|index| self.buckets[index].swap(0, Ordering::Relaxed)),
        }
    }
}

struct DispatchTelemetry {
    import_calls: [AtomicU64; DISPATCH_IMPORT_FAMILY_COUNT],
    dispatch_lock_acquire_wait: AtomicLatencyHistogram,
    dispatch_lock_hold: AtomicLatencyHistogram,
    gpu_publication_wait: AtomicLatencyHistogram,
}

impl DispatchTelemetry {
    const fn new() -> Self {
        Self {
            import_calls: [const { AtomicU64::new(0) }; DISPATCH_IMPORT_FAMILY_COUNT],
            dispatch_lock_acquire_wait: AtomicLatencyHistogram::new(),
            dispatch_lock_hold: AtomicLatencyHistogram::new(),
            gpu_publication_wait: AtomicLatencyHistogram::new(),
        }
    }

    fn record_import(&self, family: DispatchImportFamily) {
        self.import_calls[family as usize].fetch_add(1, Ordering::Relaxed);
    }

    fn take(&self) -> DispatchTelemetrySample {
        DispatchTelemetrySample {
            import_calls: std::array::from_fn(|index| {
                self.import_calls[index].swap(0, Ordering::Relaxed)
            }),
            dispatch_lock_acquire_wait: self.dispatch_lock_acquire_wait.take(),
            dispatch_lock_hold: self.dispatch_lock_hold.take(),
            gpu_publication_wait: self.gpu_publication_wait.take(),
        }
    }
}

static DISPATCH_TELEMETRY: DispatchTelemetry = DispatchTelemetry::new();

/// Records the mutually exclusive route which served one guest import.
pub fn record_dispatch_import(family: DispatchImportFamily) {
    DISPATCH_TELEMETRY.record_import(family);
}

/// Records time spent waiting to acquire the process-wide dispatch-state write lock.
pub fn record_dispatch_lock_acquire_wait(duration: Duration) {
    DISPATCH_TELEMETRY
        .dispatch_lock_acquire_wait
        .record(duration);
}

/// Records the complete lifetime of a process-wide dispatch-state write guard.
pub fn record_dispatch_lock_hold(duration: Duration) {
    DISPATCH_TELEMETRY.dispatch_lock_hold.record(duration);
}

/// Records time spent waiting for ordered GPU publication and bounded backend admission.
pub fn record_gpu_publication_wait(duration: Duration) {
    DISPATCH_TELEMETRY.gpu_publication_wait.record(duration);
}

/// Takes and resets every B10 counter as one telemetry interval.
///
/// Individual atomic fields may receive a concurrent observation while the reset is in progress,
/// so a duration exactly on the boundary may split its count, total and bucket between neighboring
/// intervals. No field increment is lost: every field uses an atomic swap and every observation
/// updates fixed storage without allocation.
#[must_use]
pub fn take_dispatch_telemetry_sample() -> DispatchTelemetrySample {
    DISPATCH_TELEMETRY.take()
}

#[cfg(test)]
mod tests {
    use super::{
        AtomicLatencyHistogram, DISPATCH_LATENCY_BUCKET_COUNT, DISPATCH_LATENCY_BUCKET_UPPER_NS,
        DispatchImportFamily, DispatchTelemetry,
    };
    use std::time::Duration;

    #[test]
    fn latency_histogram_uses_declared_boundaries_and_resets() {
        let histogram = AtomicLatencyHistogram::new();
        for nanoseconds in [
            0,
            DISPATCH_LATENCY_BUCKET_UPPER_NS[0],
            DISPATCH_LATENCY_BUCKET_UPPER_NS[0] + 1,
            DISPATCH_LATENCY_BUCKET_UPPER_NS[6],
            DISPATCH_LATENCY_BUCKET_UPPER_NS[6] + 1,
        ] {
            histogram.record(Duration::from_nanos(nanoseconds));
        }
        let sample = histogram.take();
        assert_eq!(sample.count, 5);
        assert_eq!(sample.buckets[0], 2);
        assert_eq!(sample.buckets[1], 1);
        assert_eq!(sample.buckets[6], 1);
        assert_eq!(sample.buckets[DISPATCH_LATENCY_BUCKET_COUNT - 1], 1);
        assert_eq!(sample.maximum_ns, DISPATCH_LATENCY_BUCKET_UPPER_NS[6] + 1);
        assert_eq!(histogram.take(), super::DispatchLatencyHistogram::default());
    }

    #[test]
    fn import_family_counts_are_mutually_exclusive_and_reset() {
        let telemetry = DispatchTelemetry::new();
        telemetry.record_import(DispatchImportFamily::Direct);
        telemetry.record_import(DispatchImportFamily::Direct);
        telemetry.record_import(DispatchImportFamily::Stateful);
        let sample = telemetry.take();
        assert_eq!(sample.total_import_calls(), 3);
        assert_eq!(sample.import_calls_for(DispatchImportFamily::Direct), 2);
        assert_eq!(sample.import_calls_for(DispatchImportFamily::Stateful), 1);
        assert_eq!(telemetry.take(), super::DispatchTelemetrySample::default());
    }
}
