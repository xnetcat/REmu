// SPDX-License-Identifier: GPL-2.0-or-later

//! Allocation-free runtime-dispatch telemetry shared with the presenter sampler.

use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
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

/// Shape of the deferred runtime publication whose exact span is being attributed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum GpuPublicationRoute {
    QueueOnly = 0,
    Mixed,
}

pub const GPU_PUBLICATION_ROUTE_COUNT: usize = GpuPublicationRoute::Mixed as usize + 1;

/// Mutually exclusive serial portions of `flush_deferred_gpu_publication`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum GpuPublicationPhase {
    TicketOrderWait = 0,
    BarrierAcquire,
    ChannelSetup,
    QueueFifoWait,
    BoundedSend,
    Residual,
}

pub const GPU_PUBLICATION_PHASE_COUNT: usize = GpuPublicationPhase::Residual as usize + 1;

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

/// Exact link-time runtime routes which bypass the captured-frame dispatcher.
///
/// Recording the exact route publishes a thread-owned cumulative counter with one relaxed store;
/// observing a high-volume route does not add a contended read/modify/write to the guest call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum DirectDispatchImport {
    ScePthreadSelf = 0,
    PosixPthreadSelf,
    ScePthreadGetThreadId,
    PosixPthreadGetSpecific,
    ScePthreadGetSpecific,
    PosixPthreadSetSpecific,
    ScePthreadSetSpecific,
    ScePthreadMutexLock,
    PosixPthreadMutexLock,
    ScePthreadMutexUnlock,
    PosixPthreadMutexUnlock,
    PosixPthreadRwlockWriteLock,
    PosixPthreadRwlockUnlock,
    KernelGetProcessTimeCounter,
}

pub const DIRECT_DISPATCH_IMPORT_COUNT: usize =
    DirectDispatchImport::KernelGetProcessTimeCounter as usize + 1;
const DIRECT_DISPATCH_STRIPE_COUNT: usize = 256;
#[cfg(test)]
static NEXT_DIRECT_DISPATCH_STRIPE: AtomicUsize = AtomicUsize::new(0);

/// Direct imports are hot enough that even a relaxed read/modify/write can become visible under
/// Rosetta. Each runtime thread therefore owns one publication stripe and writes monotonically
/// increasing counters with ordinary relaxed stores. The sampler differences those published
/// totals, retaining exact interval counts without putting a locked instruction on every import.
/// Threads beyond the fixed stripe budget use the atomic overflow counters.
static NEXT_PUBLISHED_DIRECT_DISPATCH_STRIPE: AtomicUsize = AtomicUsize::new(0);
static REPORTED_DIRECT_DISPATCH_PUBLICATION: AtomicBool = AtomicBool::new(false);
static PUBLISHED_DIRECT_DISPATCH_CALLS: [[AtomicU64; DIRECT_DISPATCH_IMPORT_COUNT];
    DIRECT_DISPATCH_STRIPE_COUNT] =
    [const { [const { AtomicU64::new(0) }; DIRECT_DISPATCH_IMPORT_COUNT] };
        DIRECT_DISPATCH_STRIPE_COUNT];
static SAMPLED_DIRECT_DISPATCH_CALLS: [[AtomicU64; DIRECT_DISPATCH_IMPORT_COUNT];
    DIRECT_DISPATCH_STRIPE_COUNT] =
    [const { [const { AtomicU64::new(0) }; DIRECT_DISPATCH_IMPORT_COUNT] };
        DIRECT_DISPATCH_STRIPE_COUNT];
static OVERFLOW_DIRECT_DISPATCH_CALLS: [AtomicU64; DIRECT_DISPATCH_IMPORT_COUNT] =
    [const { AtomicU64::new(0) }; DIRECT_DISPATCH_IMPORT_COUNT];
static DIRECT_DISPATCH_SAMPLE_LOCK: Mutex<()> = Mutex::new(());

struct PublishedDirectDispatchCounters {
    stripe: Option<usize>,
    calls: [Cell<u64>; DIRECT_DISPATCH_IMPORT_COUNT],
}

impl PublishedDirectDispatchCounters {
    fn new() -> Self {
        if !REPORTED_DIRECT_DISPATCH_PUBLICATION.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[REMu][PERF-DIRECT] thread-owned publication enabled stripes={DIRECT_DISPATCH_STRIPE_COUNT}"
            );
        }
        let stripe = NEXT_PUBLISHED_DIRECT_DISPATCH_STRIPE.fetch_add(1, Ordering::Relaxed);
        Self {
            stripe: (stripe < DIRECT_DISPATCH_STRIPE_COUNT).then_some(stripe),
            calls: [const { Cell::new(0) }; DIRECT_DISPATCH_IMPORT_COUNT],
        }
    }

    fn record(&self, import: DirectDispatchImport) {
        let index = import as usize;
        if let Some(stripe) = self.stripe {
            let calls = self.calls[index].get().wrapping_add(1);
            self.calls[index].set(calls);
            PUBLISHED_DIRECT_DISPATCH_CALLS[stripe][index].store(calls, Ordering::Relaxed);
        } else {
            OVERFLOW_DIRECT_DISPATCH_CALLS[index].fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
thread_local! {
    static DIRECT_DISPATCH_STRIPE: usize =
        NEXT_DIRECT_DISPATCH_STRIPE.fetch_add(1, Ordering::Relaxed) % DIRECT_DISPATCH_STRIPE_COUNT;
}

thread_local! {
    static PUBLISHED_DIRECT_DISPATCH_COUNTERS: PublishedDirectDispatchCounters =
        PublishedDirectDispatchCounters::new();
}

fn take_published_direct_dispatch_calls() -> [u64; DIRECT_DISPATCH_IMPORT_COUNT] {
    let _sample_guard = DIRECT_DISPATCH_SAMPLE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    std::array::from_fn(|index| {
        let striped = PUBLISHED_DIRECT_DISPATCH_CALLS
            .iter()
            .zip(SAMPLED_DIRECT_DISPATCH_CALLS.iter())
            .fold(0_u64, |total, (published, sampled)| {
                let current = published[index].load(Ordering::Relaxed);
                let previous = sampled[index].swap(current, Ordering::Relaxed);
                total.saturating_add(current.wrapping_sub(previous))
            });
        striped.saturating_add(OVERFLOW_DIRECT_DISPATCH_CALLS[index].swap(0, Ordering::Relaxed))
    })
}

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
    pub direct_import_calls: [u64; DIRECT_DISPATCH_IMPORT_COUNT],
    pub dispatch_lock_acquire_wait: DispatchLatencyHistogram,
    pub dispatch_lock_hold: DispatchLatencyHistogram,
    pub gpu_publication_wait: DispatchLatencyHistogram,
    gpu_publication_phases:
        [[DispatchLatencyHistogram; GPU_PUBLICATION_PHASE_COUNT]; GPU_PUBLICATION_ROUTE_COUNT],
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

    #[must_use]
    pub fn direct_import_calls_for(&self, import: DirectDispatchImport) -> u64 {
        self.direct_import_calls[import as usize]
    }

    #[must_use]
    pub fn gpu_publication_phase(
        &self,
        route: GpuPublicationRoute,
        phase: GpuPublicationPhase,
    ) -> DispatchLatencyHistogram {
        self.gpu_publication_phases[route as usize][phase as usize]
    }
}

/// Formats exact outer-publication attribution without producing an unbounded diagnostic line.
#[must_use]
pub fn format_gpu_publication_phase_report(
    window_ms: u128,
    sample: &DispatchTelemetrySample,
) -> [String; 5] {
    let compact = |route, phase| {
        let latency = sample.gpu_publication_phase(route, phase);
        let mean = latency
            .total_ns
            .checked_div(latency.count)
            .unwrap_or_default();
        format!(
            "{}/{}/{}/{}",
            latency.count, latency.total_ns, mean, latency.maximum_ns,
        )
    };
    let total_calls = [GpuPublicationRoute::QueueOnly, GpuPublicationRoute::Mixed]
        .into_iter()
        .map(|route| {
            sample
                .gpu_publication_phase(route, GpuPublicationPhase::TicketOrderWait)
                .count
        })
        .fold(0_u64, u64::saturating_add);
    let route_lines = |name, route| {
        [
            format!(
                "[REMu][PERF-GPU-PUBLICATION] route={name} part=admit order={} barrier={} setup={}",
                compact(route, GpuPublicationPhase::TicketOrderWait),
                compact(route, GpuPublicationPhase::BarrierAcquire),
                compact(route, GpuPublicationPhase::ChannelSetup),
            ),
            format!(
                "[REMu][PERF-GPU-PUBLICATION] route={name} part=channel fifo={} send={} residual={}",
                compact(route, GpuPublicationPhase::QueueFifoWait),
                compact(route, GpuPublicationPhase::BoundedSend),
                compact(route, GpuPublicationPhase::Residual),
            ),
        ]
    };
    let queue = route_lines("queue", GpuPublicationRoute::QueueOnly);
    let mixed = route_lines("mixed", GpuPublicationRoute::Mixed);
    [
        format!(
            "[REMu][PERF-GPU-PUBLICATION] window_ms={window_ms} calls={total_calls} values=count/total_ns/mean_ns/max_ns"
        ),
        queue[0].clone(),
        queue[1].clone(),
        mixed[0].clone(),
        mixed[1].clone(),
    ]
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
    #[cfg(test)]
    direct_import_calls: [[AtomicU64; DIRECT_DISPATCH_IMPORT_COUNT]; DIRECT_DISPATCH_STRIPE_COUNT],
    dispatch_lock_acquire_wait: AtomicLatencyHistogram,
    dispatch_lock_hold: AtomicLatencyHistogram,
    gpu_publication_wait: AtomicLatencyHistogram,
    gpu_publication_phases:
        [[AtomicLatencyHistogram; GPU_PUBLICATION_PHASE_COUNT]; GPU_PUBLICATION_ROUTE_COUNT],
}

impl DispatchTelemetry {
    const fn new() -> Self {
        Self {
            import_calls: [const { AtomicU64::new(0) }; DISPATCH_IMPORT_FAMILY_COUNT],
            #[cfg(test)]
            direct_import_calls: [const {
                [const { AtomicU64::new(0) }; DIRECT_DISPATCH_IMPORT_COUNT]
            }; DIRECT_DISPATCH_STRIPE_COUNT],
            dispatch_lock_acquire_wait: AtomicLatencyHistogram::new(),
            dispatch_lock_hold: AtomicLatencyHistogram::new(),
            gpu_publication_wait: AtomicLatencyHistogram::new(),
            gpu_publication_phases: [const {
                [const { AtomicLatencyHistogram::new() }; GPU_PUBLICATION_PHASE_COUNT]
            }; GPU_PUBLICATION_ROUTE_COUNT],
        }
    }

    fn record_import(&self, family: DispatchImportFamily) {
        self.import_calls[family as usize].fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn record_direct_import(&self, import: DirectDispatchImport) {
        DIRECT_DISPATCH_STRIPE.with(|stripe| {
            self.direct_import_calls[*stripe][import as usize].fetch_add(1, Ordering::Relaxed);
        });
    }

    fn take(&self) -> DispatchTelemetrySample {
        #[cfg(test)]
        let direct_import_calls = std::array::from_fn(|index| {
            self.direct_import_calls.iter().fold(0_u64, |total, stripe| {
                total.saturating_add(stripe[index].swap(0, Ordering::Relaxed))
            })
        });
        #[cfg(not(test))]
        let direct_import_calls = [0; DIRECT_DISPATCH_IMPORT_COUNT];
        let mut import_calls = std::array::from_fn(|index| {
            self.import_calls[index].swap(0, Ordering::Relaxed)
        });
        import_calls[DispatchImportFamily::Direct as usize] = import_calls
            [DispatchImportFamily::Direct as usize]
            .saturating_add(
                direct_import_calls
                    .iter()
                    .copied()
                    .fold(0_u64, u64::saturating_add),
            );
        DispatchTelemetrySample {
            import_calls,
            direct_import_calls,
            dispatch_lock_acquire_wait: self.dispatch_lock_acquire_wait.take(),
            dispatch_lock_hold: self.dispatch_lock_hold.take(),
            gpu_publication_wait: self.gpu_publication_wait.take(),
            gpu_publication_phases: std::array::from_fn(|route| {
                std::array::from_fn(|phase| self.gpu_publication_phases[route][phase].take())
            }),
        }
    }
}

static DISPATCH_TELEMETRY: DispatchTelemetry = DispatchTelemetry::new();

/// Records the mutually exclusive route which served one guest import.
pub fn record_dispatch_import(family: DispatchImportFamily) {
    DISPATCH_TELEMETRY.record_import(family);
}

/// Records the exact link-time runtime route which served one guest import.
pub fn record_direct_dispatch_import(import: DirectDispatchImport) {
    PUBLISHED_DIRECT_DISPATCH_COUNTERS.with(|counters| counters.record(import));
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

static REPORTED_EXACT_GPU_PUBLICATION_ATTRIBUTION: AtomicBool = AtomicBool::new(false);

/// Records one exact outer-publication phase after the complete call has been attributed locally.
pub fn record_gpu_publication_phase(
    route: GpuPublicationRoute,
    phase: GpuPublicationPhase,
    duration: Duration,
) {
    if !REPORTED_EXACT_GPU_PUBLICATION_ATTRIBUTION.load(Ordering::Relaxed)
        && REPORTED_EXACT_GPU_PUBLICATION_ATTRIBUTION
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    {
        eprintln!("[REMu][PERF-GPU-PUBLICATION] exact runtime phase attribution enabled");
    }
    DISPATCH_TELEMETRY.gpu_publication_phases[route as usize][phase as usize].record(duration);
}

/// Takes and resets every B10 counter as one telemetry interval.
///
/// Individual latency fields may receive a concurrent observation while the reset is in progress,
/// so a duration exactly on the boundary may split its count, total and bucket between neighboring
/// intervals. Direct imports publish cumulative thread-owned totals; the sampler serializes and
/// differences them, so a boundary race can move a call into the next interval but cannot lose or
/// double-count it. Every route updates fixed storage without allocation.
#[must_use]
pub fn take_dispatch_telemetry_sample() -> DispatchTelemetrySample {
    let mut sample = DISPATCH_TELEMETRY.take();
    sample.direct_import_calls = take_published_direct_dispatch_calls();
    sample.import_calls[DispatchImportFamily::Direct as usize] =
        sample.import_calls[DispatchImportFamily::Direct as usize].saturating_add(
            sample
                .direct_import_calls
                .iter()
                .copied()
                .fold(0_u64, u64::saturating_add),
        );
    sample
}

#[cfg(test)]
mod tests {
    use super::{
        AtomicLatencyHistogram, DISPATCH_LATENCY_BUCKET_COUNT, DISPATCH_LATENCY_BUCKET_UPPER_NS,
        DirectDispatchImport, DispatchImportFamily, DispatchTelemetry, GpuPublicationPhase,
        GpuPublicationRoute, format_gpu_publication_phase_report, record_direct_dispatch_import,
        take_dispatch_telemetry_sample,
    };
    use std::sync::{Arc, Barrier};
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
        telemetry.record_direct_import(DirectDispatchImport::ScePthreadSelf);
        telemetry.record_direct_import(DirectDispatchImport::KernelGetProcessTimeCounter);
        telemetry.record_import(DispatchImportFamily::Stateful);
        let sample = telemetry.take();
        assert_eq!(sample.total_import_calls(), 3);
        assert_eq!(sample.import_calls_for(DispatchImportFamily::Direct), 2);
        assert_eq!(
            sample.direct_import_calls_for(DirectDispatchImport::ScePthreadSelf),
            1
        );
        assert_eq!(
            sample.direct_import_calls_for(DirectDispatchImport::KernelGetProcessTimeCounter),
            1
        );
        assert_eq!(sample.import_calls_for(DispatchImportFamily::Stateful), 1);
        assert_eq!(telemetry.take(), super::DispatchTelemetrySample::default());
    }

    #[test]
    fn direct_import_stripes_preserve_exact_multithreaded_counts() {
        const THREADS: u64 = 8;
        const CALLS_PER_THREAD: u64 = 10_000;
        let telemetry = DispatchTelemetry::new();
        std::thread::scope(|scope| {
            for _ in 0..THREADS {
                scope.spawn(|| {
                    for _ in 0..CALLS_PER_THREAD {
                        telemetry.record_direct_import(DirectDispatchImport::ScePthreadMutexLock);
                    }
                });
            }
        });
        let sample = telemetry.take();
        assert_eq!(
            sample.direct_import_calls_for(DirectDispatchImport::ScePthreadMutexLock),
            THREADS * CALLS_PER_THREAD
        );
        assert_eq!(
            sample.import_calls_for(DispatchImportFamily::Direct),
            THREADS * CALLS_PER_THREAD
        );
        assert_eq!(telemetry.take(), super::DispatchTelemetrySample::default());
    }

    #[test]
    fn published_direct_import_stores_preserve_counts_across_samples() {
        const THREADS: u64 = 8;
        const CALLS_PER_PHASE: u64 = 10_000;
        let _ = take_dispatch_telemetry_sample();
        let first_phase = Arc::new(Barrier::new(usize::try_from(THREADS).unwrap() + 1));
        let second_phase = Arc::new(Barrier::new(usize::try_from(THREADS).unwrap() + 1));
        std::thread::scope(|scope| {
            for _ in 0..THREADS {
                let first_phase = Arc::clone(&first_phase);
                let second_phase = Arc::clone(&second_phase);
                scope.spawn(move || {
                    for _ in 0..CALLS_PER_PHASE {
                        record_direct_dispatch_import(DirectDispatchImport::ScePthreadGetSpecific);
                    }
                    first_phase.wait();
                    second_phase.wait();
                    for _ in 0..CALLS_PER_PHASE {
                        record_direct_dispatch_import(DirectDispatchImport::ScePthreadGetSpecific);
                    }
                });
            }
            first_phase.wait();
            let first = take_dispatch_telemetry_sample();
            assert_eq!(
                first.direct_import_calls_for(DirectDispatchImport::ScePthreadGetSpecific),
                THREADS * CALLS_PER_PHASE
            );
            second_phase.wait();
        });
        let second = take_dispatch_telemetry_sample();
        assert_eq!(
            second.direct_import_calls_for(DirectDispatchImport::ScePthreadGetSpecific),
            THREADS * CALLS_PER_PHASE
        );
        assert_eq!(
            take_dispatch_telemetry_sample()
                .direct_import_calls_for(DirectDispatchImport::ScePthreadGetSpecific),
            0
        );
    }

    #[test]
    fn gpu_publication_phase_accounting_resets_and_formats_compactly() {
        let telemetry = DispatchTelemetry::new();
        let queue_order = &telemetry.gpu_publication_phases
            [GpuPublicationRoute::QueueOnly as usize]
            [GpuPublicationPhase::TicketOrderWait as usize];
        queue_order.record(Duration::from_nanos(10));
        queue_order.record(Duration::from_nanos(30));
        telemetry.gpu_publication_phases[GpuPublicationRoute::QueueOnly as usize]
            [GpuPublicationPhase::BoundedSend as usize]
            .record(Duration::from_nanos(5));
        telemetry.gpu_publication_phases[GpuPublicationRoute::Mixed as usize]
            [GpuPublicationPhase::Residual as usize]
            .record(Duration::from_nanos(7));

        let sample = telemetry.take();
        let order = sample.gpu_publication_phase(
            GpuPublicationRoute::QueueOnly,
            GpuPublicationPhase::TicketOrderWait,
        );
        assert_eq!(order.count, 2);
        assert_eq!(order.total_ns, 40);
        assert_eq!(order.maximum_ns, 30);
        let report = format_gpu_publication_phase_report(5_000, &sample);
        assert!(report[1].contains("order=2/40/20/30"));
        assert!(report[2].contains("send=1/5/5/5"));
        assert!(report[4].contains("residual=1/7/7/7"));
        assert!(report.iter().all(|line| line.len() < 256));
        assert_eq!(
            telemetry
                .take()
                .gpu_publication_phase(
                    GpuPublicationRoute::QueueOnly,
                    GpuPublicationPhase::TicketOrderWait,
                )
                .count,
            0
        );
    }
}
