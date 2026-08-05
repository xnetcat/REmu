// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    GuestTextureDescriptor, Mode27WorkAggregate, Mode27WorkKey, Mode27WorkOrigin,
    PresenterStatistics, TextureDirtyWorkAggregate, TextureSnapshotWorkKey,
    WRITEBACK_TRIGGER_COUNT,
};
use std::collections::HashMap;
use std::ffi::c_void;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(super) enum SharedEventHostEffectLatencyPhase {
    ConsumerAck = 0,
    ImmutablePrefix,
    HostTerminal,
}

const SHARED_EVENT_HOST_EFFECT_LATENCY_PHASE_COUNT: usize =
    SharedEventHostEffectLatencyPhase::HostTerminal as usize + 1;

/// Presenter-local aggregate for one phase of a truthful SharedEvent host-effect gate.
///
/// The presenter is the sole writer, so these counters need no atomics. A sample is recorded only
/// when a gate crosses a phase boundary; unresolved gates are reported separately as a bounded
/// snapshot of the receiver's queue fronts.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct SharedEventHostEffectLatency {
    pub(super) count: u64,
    pub(super) total_ns: u64,
    pub(super) maximum_ns: u64,
}

impl SharedEventHostEffectLatency {
    pub(super) fn record(&mut self, duration: Duration) {
        let nanoseconds = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        self.count = self.count.saturating_add(1);
        self.total_ns = self.total_ns.saturating_add(nanoseconds);
        self.maximum_ns = self.maximum_ns.max(nanoseconds);
    }

    pub(super) fn mean_ns(self) -> u64 {
        self.total_ns.checked_div(self.count).unwrap_or_default()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct SharedEventHostEffectLatencyStatistics {
    phases: [SharedEventHostEffectLatency; SHARED_EVENT_HOST_EFFECT_LATENCY_PHASE_COUNT],
}

impl SharedEventHostEffectLatencyStatistics {
    pub(super) fn record(&mut self, phase: SharedEventHostEffectLatencyPhase, duration: Duration) {
        self.phases[phase as usize].record(duration);
    }

    pub(super) fn phase(
        self,
        phase: SharedEventHostEffectLatencyPhase,
    ) -> SharedEventHostEffectLatency {
        self.phases[phase as usize]
    }

    pub(super) fn take(&mut self) -> Self {
        std::mem::take(self)
    }
}

/// Point-in-time ages for host-effect gates which have not crossed their current phase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct SharedEventHostEffectPendingStatistics {
    phases: [SharedEventHostEffectLatency; SHARED_EVENT_HOST_EFFECT_LATENCY_PHASE_COUNT],
}

impl SharedEventHostEffectPendingStatistics {
    pub(super) fn record(&mut self, phase: SharedEventHostEffectLatencyPhase, age: Duration) {
        self.phases[phase as usize].record(age);
    }

    pub(super) fn phase(
        self,
        phase: SharedEventHostEffectLatencyPhase,
    ) -> SharedEventHostEffectLatency {
        self.phases[phase as usize]
    }
}

pub(super) fn format_shared_event_host_effect_latency_report(
    window_ms: u128,
    completed: SharedEventHostEffectLatencyStatistics,
    pending: SharedEventHostEffectPendingStatistics,
) -> String {
    let completed_phase = |phase| {
        let latency = completed.phase(phase);
        format!(
            "{}/{}/{}/{}",
            latency.count,
            latency.total_ns,
            latency.mean_ns(),
            latency.maximum_ns,
        )
    };
    let pending_phase = |phase| {
        let latency = pending.phase(phase);
        format!(
            "{}/{}/{}",
            latency.count, latency.total_ns, latency.maximum_ns,
        )
    };
    format!(
        "[REMu][PERF-SHARED-EVENT-LATENCY] window_ms={window_ms} completed=count/total_ns/mean_ns/max_ns ack={} ack-prefix={} prefix-terminal={} pending=count/total_age_ns/max_age_ns ack={} ack-prefix={} prefix-terminal={}",
        completed_phase(SharedEventHostEffectLatencyPhase::ConsumerAck),
        completed_phase(SharedEventHostEffectLatencyPhase::ImmutablePrefix),
        completed_phase(SharedEventHostEffectLatencyPhase::HostTerminal),
        pending_phase(SharedEventHostEffectLatencyPhase::ConsumerAck),
        pending_phase(SharedEventHostEffectLatencyPhase::ImmutablePrefix),
        pending_phase(SharedEventHostEffectLatencyPhase::HostTerminal),
    )
}

/// Bounded opt-in attribution for command buffers which publish a consumer-queue
/// SharedEvent acknowledgement. Every duration belongs to a sampled acknowledgement batch;
/// ordinary batches and disabled runs pay no clock or callback cost.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct SharedEventAckPhaseStatistics {
    pub(super) samples: u64,
    pub(super) draws: u64,
    pub(super) computes: u64,
    pub(super) event_queue: SharedEventHostEffectLatency,
    pub(super) cpu_translation: SharedEventHostEffectLatency,
    pub(super) metal_encoding: SharedEventHostEffectLatency,
    pub(super) ack_encoding: SharedEventHostEffectLatency,
    pub(super) commit_call: SharedEventHostEffectLatency,
    pub(super) commit_to_scheduled: SharedEventHostEffectLatency,
    pub(super) scheduled_to_completed: SharedEventHostEffectLatency,
    pub(super) kernel_to_gpu_start: SharedEventHostEffectLatency,
    pub(super) gpu_execution: SharedEventHostEffectLatency,
    pub(super) gpu_to_kernel_end: SharedEventHostEffectLatency,
    pub(super) commit_to_ack: SharedEventHostEffectLatency,
}

impl SharedEventAckPhaseStatistics {
    pub(super) fn take(&mut self) -> Self {
        std::mem::take(self)
    }
}

pub(super) fn format_shared_event_ack_phase_report(
    window_ms: u128,
    sample: SharedEventAckPhaseStatistics,
) -> [String; 3] {
    let compact = |latency: SharedEventHostEffectLatency| {
        format!(
            "{}/{}/{}/{}",
            latency.count,
            latency.total_ns,
            latency.mean_ns(),
            latency.maximum_ns,
        )
    };
    [
        format!(
            "[REMu][PERF-SHARED-EVENT-ACK-PHASE] window_ms={window_ms} samples={} draws={} computes={} values=count/total_ns/mean_ns/max_ns",
            sample.samples, sample.draws, sample.computes,
        ),
        format!(
            "[REMu][PERF-SHARED-EVENT-ACK-PHASE] part=host event-queue={} cpu-translation={} metal-encoding={} ack-encoding={} commit-call={}",
            compact(sample.event_queue),
            compact(sample.cpu_translation),
            compact(sample.metal_encoding),
            compact(sample.ack_encoding),
            compact(sample.commit_call),
        ),
        format!(
            "[REMu][PERF-SHARED-EVENT-ACK-PHASE] part=native commit-scheduled={} scheduled-completed={} kernel-gpu-start={} gpu-execution={} gpu-kernel-end={} commit-ack={}",
            compact(sample.commit_to_scheduled),
            compact(sample.scheduled_to_completed),
            compact(sample.kernel_to_gpu_start),
            compact(sample.gpu_execution),
            compact(sample.gpu_to_kernel_end),
            compact(sample.commit_to_ack),
        ),
    ]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ProcessMemoryStatistics {
    pub(super) resident_bytes: u64,
    pub(super) physical_footprint_bytes: u64,
    pub(super) peak_physical_footprint_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(super) enum TerminalPublicationPhase {
    MetalTerminal = 0,
    TerminalMetadata,
    ErrorFenceAdvance,
    GlobalWriteback,
    GuestMemoryWrites,
    QueueRetirement,
    GpuTimeQuery,
    CompletionWake,
}

const TERMINAL_PUBLICATION_PHASE_COUNT: usize =
    TerminalPublicationPhase::CompletionWake as usize + 1;

#[cfg(test)]
static TERMINAL_PUBLICATION_TIMING_SAMPLES: AtomicUsize = AtomicUsize::new(0);

fn terminal_publication_now() -> Instant {
    #[cfg(test)]
    TERMINAL_PUBLICATION_TIMING_SAMPLES.fetch_add(1, Ordering::Relaxed);
    Instant::now()
}

/// Presenter-local latency aggregate for one truthful terminal-publication phase.
///
/// Terminal retirement runs only on the presenter thread. Keeping these counters local avoids an
/// atomic read/modify/write for every completed Metal batch while retaining exact periodic sums.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct TerminalPublicationLatency {
    pub(super) count: u64,
    pub(super) total_ns: u64,
    pub(super) maximum_ns: u64,
}

impl TerminalPublicationLatency {
    fn record(&mut self, duration: Duration) {
        let nanoseconds = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
        self.count = self.count.saturating_add(1);
        self.total_ns = self.total_ns.saturating_add(nanoseconds);
        self.maximum_ns = self.maximum_ns.max(nanoseconds);
    }

    pub(super) fn mean_ns(self) -> u64 {
        self.total_ns.checked_div(self.count).unwrap_or_default()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct TerminalPublicationStatistics {
    phases: [TerminalPublicationLatency; TERMINAL_PUBLICATION_PHASE_COUNT],
}

impl TerminalPublicationStatistics {
    fn record(&mut self, phase: TerminalPublicationPhase, duration: Duration) {
        self.phases[phase as usize].record(duration);
    }

    pub(super) fn phase(self, phase: TerminalPublicationPhase) -> TerminalPublicationLatency {
        self.phases[phase as usize]
    }

    /// Takes one periodic sample and resets every phase in the same presenter-thread operation.
    pub(super) fn take(&mut self) -> Self {
        std::mem::take(self)
    }
}

pub(super) fn format_terminal_publication_report(
    window_ms: u128,
    sample: TerminalPublicationStatistics,
) -> [String; 4] {
    let compact = |phase| {
        let latency = sample.phase(phase);
        format!(
            "{}/{}/{}/{}",
            latency.count,
            latency.total_ns,
            latency.mean_ns(),
            latency.maximum_ns,
        )
    };
    let completed = sample.phase(TerminalPublicationPhase::CompletionWake).count;
    [
        format!(
            "[REMu][PERF-GPU-PUBLICATION-PHASE] window_ms={window_ms} terminal_calls={completed} values=count/total_ns/mean_ns/max_ns"
        ),
        format!(
            "[REMu][PERF-GPU-PUBLICATION-PHASE] part=terminal metal={} metadata={} error-fence={}",
            compact(TerminalPublicationPhase::MetalTerminal),
            compact(TerminalPublicationPhase::TerminalMetadata),
            compact(TerminalPublicationPhase::ErrorFenceAdvance),
        ),
        format!(
            "[REMu][PERF-GPU-PUBLICATION-PHASE] part=memory globals={} guest-memory={}",
            compact(TerminalPublicationPhase::GlobalWriteback),
            compact(TerminalPublicationPhase::GuestMemoryWrites),
        ),
        format!(
            "[REMu][PERF-GPU-PUBLICATION-PHASE] part=retire queue={} gpu-time={} completion-wake={}",
            compact(TerminalPublicationPhase::QueueRetirement),
            compact(TerminalPublicationPhase::GpuTimeQuery),
            compact(TerminalPublicationPhase::CompletionWake),
        ),
    ]
}

/// One monotonic boundary sample per serial phase after the initial call timestamp.
pub(super) struct TerminalPublicationTimer {
    last: Option<Instant>,
}

impl TerminalPublicationTimer {
    pub(super) fn start_if(enabled: bool) -> Self {
        Self {
            last: enabled.then(terminal_publication_now),
        }
    }

    pub(super) fn mark(
        &mut self,
        statistics: &mut TerminalPublicationStatistics,
        phase: TerminalPublicationPhase,
    ) {
        let Some(last) = self.last else {
            return;
        };
        let now = terminal_publication_now();
        statistics.record(phase, now.saturating_duration_since(last));
        self.last = Some(now);
    }
}

unsafe extern "C" {
    #[link_name = "proc_pid_rusage"]
    fn process_pid_rusage(
        pid: libc::c_int,
        flavor: libc::c_int,
        buffer: *mut libc::c_void,
    ) -> libc::c_int;
}

pub(super) static TEXTURE_SNAPSHOT_COUNT: AtomicU64 = AtomicU64::new(0);

pub(super) static TEXTURE_SNAPSHOT_BYTES: AtomicU64 = AtomicU64::new(0);

pub(super) static TEXTURE_SNAPSHOT_NANOSECONDS: AtomicU64 = AtomicU64::new(0);

/// Snapshots published while a guest thread was still writing the range.
pub(super) static TEXTURE_SNAPSHOT_UNSTABLE_COUNT: AtomicU64 = AtomicU64::new(0);

pub(super) static TEXTURE_DIRTY_VALIDATION_COUNT: AtomicU64 = AtomicU64::new(0);

pub(super) static TEXTURE_DIRTY_VALIDATION_BYTES: AtomicU64 = AtomicU64::new(0);

pub(super) static TEXTURE_DIRTY_VALIDATION_NANOSECONDS: AtomicU64 = AtomicU64::new(0);

pub(super) static TEXTURE_DIRTY_VALIDATION_UNCHANGED: AtomicU64 = AtomicU64::new(0);

pub(super) static MODE27_DECODE_COUNT: AtomicU64 = AtomicU64::new(0);

pub(super) static MODE27_DECODE_BYTES: AtomicU64 = AtomicU64::new(0);

pub(super) static MODE27_DECODE_NANOSECONDS: AtomicU64 = AtomicU64::new(0);

pub(super) static MIP_UPLOAD_COUNT: AtomicU64 = AtomicU64::new(0);

pub(super) static MIP_UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);

pub(super) static MIP_UPLOAD_NANOSECONDS: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_TEXTURE_CACHE_HITS: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_TEXTURE_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_TEXTURE_CACHE_DIRTY_RELOADS: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_TEXTURE_CACHE_ACQUIRE_INVALIDATIONS: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_TEXTURE_CACHE_CAPACITY_EVICTIONS: AtomicU64 = AtomicU64::new(0);

pub(super) static RESIDENT_TEXTURE_ACQUIRE_INVALIDATIONS: AtomicU64 = AtomicU64::new(0);

/// GPU writes that reached the mark path without the resident the ensure path served.
///
/// Resolving a written target by exact key or shared storage requires an identical pitch, while
/// the ensure path also accepts a pitch-mismatched GPU-sample alias. Any count here is a write
/// whose `gpu_written`, `content_version` and activity never reached a resident that exists.
pub(super) static RESIDENT_WRITE_ATTRIBUTION_MISSES: AtomicU64 = AtomicU64::new(0);

pub(super) static RESIDENT_DEPTH_SAMPLE_HITS: AtomicU64 = AtomicU64::new(0);

/// Times a range set widened its claim instead of tracking every run exactly.
///
/// Widening an authority claim can only cost an extra re-materialisation, never stale pixels, so
/// this is a throughput signal rather than a correctness one. Counted for the `RANGE_SET_MAX_RUNS`
/// collapse and for the layout cases whose rect refinement is deliberately not attempted.
pub(super) static OWNED_RANGE_SET_COARSENED: AtomicU64 = AtomicU64::new(0);

/// Residents admitted with no complete live shared backing for their declared range.
///
/// Their ranges stay in virtual-address space and are never compared against a resident that did
/// resolve, so they are excluded from the cross-resident algebra entirely.
pub(super) static OWNED_UNRESOLVED_SPAN: AtomicU64 = AtomicU64::new(0);

/// Written-region computations that fell back to the writer's whole declared span.
///
/// Every failure inside `written_byte_ranges` lands here, which is exactly the descriptor-level
/// range the code claimed before the range algebra existed.
pub(super) static OWNED_FALLBACK_FULL_RANGE: AtomicU64 = AtomicU64::new(0);

/// Residents an intersecting write reached through the index but did not take a byte from.
///
/// This is the direct measure of pooled tenants surviving a neighbour's write: every count is a
/// resident the descriptor-overlap rule would have destroyed.
pub(super) static OWNED_SUBTRACT_TENANTS_SURVIVED: AtomicU64 = AtomicU64::new(0);

/// Residents that lost their last owned byte and moved to `Stale` in place.
pub(super) static OWNED_DEMOTIONS: AtomicU64 = AtomicU64::new(0);

/// Demoted residents re-taken by a bind that overwrites every texel, at no copy cost.
pub(super) static OWNED_RECLAIM_DISCARD: AtomicU64 = AtomicU64::new(0);

/// Demoted allocations restored by reusing a compatible Metal texture view.
pub(super) static OWNED_REMATERIALISE_VIEW: AtomicU64 = AtomicU64::new(0);

/// Demoted allocations restored by an ordered texture-to-texture copy.
pub(super) static OWNED_REMATERIALISE_BLIT: AtomicU64 = AtomicU64::new(0);

/// Demoted allocations reconstructed by ordered GPU retile/de-tile work.
pub(super) static OWNED_REMATERIALISE_DETILE: AtomicU64 = AtomicU64::new(0);

/// Demoted allocations rebuilt from guest memory after proving RAM is authoritative.
pub(super) static OWNED_REMATERIALISE_CPU: AtomicU64 = AtomicU64::new(0);

/// GPU-current stale reads for which no lossless rematerialisation rung was available.
pub(super) static OWNED_REMATERIALISE_FAILED: AtomicU64 = AtomicU64::new(0);

/// Failure reasons are deliberately separate from the total so a run says which rung to build.
pub(super) static OWNED_REMATERIALISE_FAILED_MAPPING: AtomicU64 = AtomicU64::new(0);
pub(super) static OWNED_REMATERIALISE_FAILED_CPU_GPU: AtomicU64 = AtomicU64::new(0);
pub(super) static OWNED_REMATERIALISE_FAILED_LAYOUT: AtomicU64 = AtomicU64::new(0);
pub(super) static OWNED_REMATERIALISE_FAILED_METAL: AtomicU64 = AtomicU64::new(0);

/// Resident writebacks that reached guest memory, indexed by `WritebackTrigger`.
pub(super) static RESIDENT_WRITEBACK_PUBLISHED: [AtomicU64; WRITEBACK_TRIGGER_COUNT] =
    [const { AtomicU64::new(0) }; WRITEBACK_TRIGGER_COUNT];

/// Resident writebacks refused admission, indexed by `WritebackTrigger`.
///
/// Every count is a host→guest copy REmu declined to make into a range it could not prove still
/// belongs to the texture. The published counters above are the remaining exposure.
pub(super) static RESIDENT_WRITEBACK_DISCARDED: [AtomicU64; WRITEBACK_TRIGGER_COUNT] =
    [const { AtomicU64::new(0) }; WRITEBACK_TRIGGER_COUNT];

pub(super) static RESIDENT_WRITEBACK_PUBLISHED_BYTES: AtomicU64 = AtomicU64::new(0);

pub(super) static RESIDENT_WRITEBACK_DISCARDED_BYTES: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_ALIAS_WRITEBACK_COUNT: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_ALIAS_WRITEBACK_BYTES: AtomicU64 = AtomicU64::new(0);

pub(super) static SAMPLED_ALIAS_WRITEBACK_NANOSECONDS: AtomicU64 = AtomicU64::new(0);

pub(super) static PIPELINE_COMPILE_ENQUEUED: AtomicU64 = AtomicU64::new(0);

pub(super) static PIPELINE_COMPILE_QUEUE_FULL: AtomicU64 = AtomicU64::new(0);

pub(super) static PIPELINE_COMPILE_COMPLETED: AtomicU64 = AtomicU64::new(0);

pub(super) static PIPELINE_COMPILE_FAILED: AtomicU64 = AtomicU64::new(0);

pub(super) static PIPELINE_DRAW_GATES: AtomicU64 = AtomicU64::new(0);
pub(super) static PIPELINE_COMPUTE_GATES: AtomicU64 = AtomicU64::new(0);

/// Draw packets consumed without rasterizing anything, by cause. Every one of these paths
/// reports success, so no drop counter or once-per-key failure log ever sees them; the periodic
/// `[REMu][PERF-NOOP]` line is their only footprint. Most are correct emulation (trigger
/// packets, empty scissors), which is exactly why a misclassification here — real geometry
/// deleted with zero trace — needs a number to falsify against.
pub(super) static NOOP_DRAWS_COVER_PLACEHOLDER: AtomicU64 = AtomicU64::new(0);
pub(super) static NOOP_DRAWS_RASTER_DISABLED: AtomicU64 = AtomicU64::new(0);
pub(super) static NOOP_DRAWS_COLOR_OPERATION: AtomicU64 = AtomicU64::new(0);
pub(super) static NOOP_DRAWS_EMPTY_SCISSOR: AtomicU64 = AtomicU64::new(0);
pub(super) static NOOP_DRAWS_MESH_DISPATCH_SKIPPED: AtomicU64 = AtomicU64::new(0);
pub(super) static NOOP_DRAWS_FAST_CLEAR_NOT_PENDING: AtomicU64 = AtomicU64::new(0);

/// Records a draw that reported success without rasterizing (Group-8 GPU surface).
pub(super) fn note_noop_draw(kind: &'static str) {
    remu_gpu::note_inert(kind);
}

pub(super) fn record_texture_work(
    count: &AtomicU64,
    bytes: &AtomicU64,
    nanoseconds: &AtomicU64,
    byte_count: usize,
    elapsed: Duration,
) {
    count.fetch_add(1, Ordering::Relaxed);
    bytes.fetch_add(
        u64::try_from(byte_count).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    nanoseconds.fetch_add(
        u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
}

pub(super) fn process_memory_statistics() -> Option<ProcessMemoryStatistics> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: `usage` names writable storage for the RUSAGE_INFO_V4 flavor. A zero result
    // guarantees that libproc initialized the complete structure before it is read below.
    let result = unsafe {
        process_pid_rusage(
            i32::try_from(std::process::id()).ok()?,
            libc::RUSAGE_INFO_V4,
            usage.as_mut_ptr().cast::<c_void>(),
        )
    };
    if result != 0 {
        return None;
    }
    // SAFETY: A successful RUSAGE_INFO_V4 query initialized every field of `usage`.
    let usage = unsafe { usage.assume_init() };
    Some(ProcessMemoryStatistics {
        resident_bytes: usage.ri_resident_size,
        physical_footprint_bytes: usage.ri_phys_footprint,
        peak_physical_footprint_bytes: usage.ri_lifetime_max_phys_footprint,
    })
}

#[cfg(test)]
pub(super) fn process_resident_bytes() -> Option<u64> {
    process_memory_statistics().map(|statistics| statistics.resident_bytes)
}

pub(super) static MODE27_WORK_DETAILS: OnceLock<
    Mutex<HashMap<Mode27WorkKey, Mode27WorkAggregate>>,
> = OnceLock::new();

pub(super) static TEXTURE_SNAPSHOT_WORK_DETAILS: OnceLock<
    Mutex<HashMap<TextureSnapshotWorkKey, Mode27WorkAggregate>>,
> = OnceLock::new();

pub(super) static TEXTURE_DIRTY_WORK_DETAILS: OnceLock<
    Mutex<HashMap<TextureSnapshotWorkKey, TextureDirtyWorkAggregate>>,
> = OnceLock::new();

pub(super) fn record_mode27_work_detail(
    origin: Mode27WorkOrigin,
    descriptor: GuestTextureDescriptor,
    mip_level_count: u32,
    byte_count: usize,
    elapsed: Duration,
) {
    let key = Mode27WorkKey {
        origin,
        descriptor,
        mip_level_count,
    };
    let mut details = MODE27_WORK_DETAILS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let aggregate = details.entry(key).or_default();
    aggregate.count = aggregate.count.saturating_add(1);
    aggregate.bytes = aggregate
        .bytes
        .saturating_add(u64::try_from(byte_count).unwrap_or(u64::MAX));
    aggregate.nanoseconds = aggregate
        .nanoseconds
        .saturating_add(u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX));
}

pub(super) fn record_texture_snapshot_work_detail(
    address: u64,
    byte_count: usize,
    caller_line: u32,
    elapsed: Duration,
) {
    let key = TextureSnapshotWorkKey {
        address,
        byte_count,
        caller_line,
    };
    let mut details = TEXTURE_SNAPSHOT_WORK_DETAILS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let aggregate = details.entry(key).or_default();
    aggregate.count = aggregate.count.saturating_add(1);
    aggregate.bytes = aggregate
        .bytes
        .saturating_add(u64::try_from(byte_count).unwrap_or(u64::MAX));
    aggregate.nanoseconds = aggregate
        .nanoseconds
        .saturating_add(u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX));
}

pub(super) fn take_texture_snapshot_work_details()
-> Vec<(TextureSnapshotWorkKey, Mode27WorkAggregate)> {
    let Some(details) = TEXTURE_SNAPSHOT_WORK_DETAILS.get() else {
        return Vec::new();
    };
    let mut details = details
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut snapshot = std::mem::take(&mut *details)
        .into_iter()
        .collect::<Vec<_>>();
    snapshot.sort_unstable_by_key(|(_, aggregate)| {
        std::cmp::Reverse((aggregate.nanoseconds, aggregate.bytes, aggregate.count))
    });
    snapshot
}

pub(super) fn record_texture_dirty_work_detail(
    address: u64,
    byte_count: usize,
    caller_line: u32,
    precise: bool,
    dirty_pages: usize,
    dirty_bytes: u64,
) {
    let key = TextureSnapshotWorkKey {
        address,
        byte_count,
        caller_line,
    };
    let mut details = TEXTURE_DIRTY_WORK_DETAILS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let aggregate = details.entry(key).or_default();
    aggregate.snapshots = aggregate.snapshots.saturating_add(1);
    aggregate.full_bytes = aggregate
        .full_bytes
        .saturating_add(u64::try_from(byte_count).unwrap_or(u64::MAX));
    if !precise {
        aggregate.imprecise_snapshots = aggregate.imprecise_snapshots.saturating_add(1);
    } else if dirty_pages == 0 {
        aggregate.clean_snapshots = aggregate.clean_snapshots.saturating_add(1);
    }
    aggregate.dirty_pages = aggregate
        .dirty_pages
        .saturating_add(u64::try_from(dirty_pages).unwrap_or(u64::MAX));
    aggregate.dirty_bytes = aggregate.dirty_bytes.saturating_add(dirty_bytes);
}

pub(super) fn take_texture_dirty_work_details()
-> Vec<(TextureSnapshotWorkKey, TextureDirtyWorkAggregate)> {
    let Some(details) = TEXTURE_DIRTY_WORK_DETAILS.get() else {
        return Vec::new();
    };
    let mut details = details
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut snapshot = std::mem::take(&mut *details)
        .into_iter()
        .collect::<Vec<_>>();
    snapshot.sort_unstable_by_key(|(_, aggregate)| {
        std::cmp::Reverse((
            aggregate.full_bytes,
            aggregate.dirty_bytes,
            aggregate.snapshots,
        ))
    });
    snapshot
}

pub(super) fn take_mode27_work_details() -> Vec<(Mode27WorkKey, Mode27WorkAggregate)> {
    let Some(details) = MODE27_WORK_DETAILS.get() else {
        return Vec::new();
    };
    let mut details = details
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut snapshot = std::mem::take(&mut *details)
        .into_iter()
        .collect::<Vec<_>>();
    snapshot.sort_unstable_by_key(|(_, aggregate)| {
        std::cmp::Reverse((aggregate.nanoseconds, aggregate.bytes, aggregate.count))
    });
    snapshot
}

impl PresenterStatistics {
    pub(super) fn new() -> Self {
        Self {
            interval_started: Instant::now(),
            last_logged: Instant::now(),
            last_flip_reconciliation: Instant::now(),
            guest_flips: 0,
            completed_presents: 0,
            submitted_guest_batches: 0,
            coalesced_pipeline_barriers: 0,
            encoded_draws: 0,
            encoded_compute_dispatches: 0,
            encoded_render_passes: 0,
            render_pass_ends_for_compute: 0,
            render_pass_ends_for_target_change: 0,
            render_pass_ends_for_color_operation: 0,
            render_pass_ends_for_batch_flush: 0,
            render_pass_key_transitions: Default::default(),
            batch_flushes_from_event_loop: 0,
            batch_flushes_for_guest_memory_write: 0,
            batch_flushes_for_blocked_wait: 0,
            batch_flushes_for_shared_event_host_effect: 0,
            batch_flushes_for_flip_snapshot: 0,
            batch_flushes_for_work_capacity: 0,
            batch_flushes_for_command_processor_barrier: 0,
            batch_flushes_for_cpu_synchronization: 0,
            shared_event_signals: 0,
            shared_event_waits: 0,
            same_queue_wait_elisions: 0,
            shared_event_host_blocks: 0,
            shared_event_host_block_kinds: [0; super::SHARED_HOST_KIND_COUNT],
            shared_event_host_wakes: 0,
            shared_event_acquire_noop_drops: 0,
            shared_event_dma_fill_attempts: 0,
            shared_event_gpu_ordered_dma_fills: 0,
            shared_event_dma_fill_rejections: [0; super::SAME_QUEUE_DMA_FILL_REJECT_COUNT],
            shared_event_gpu_ordered_registered_waits: 0,
            shared_event_gpu_ordered_compute_dispatches: 0,
            shared_event_gpu_ordered_draws: 0,
            shared_event_draw_attempts: 0,
            shared_event_draw_disjoint_candidates: 0,
            shared_event_draw_versioned_global_candidates: 0,
            shared_event_draw_resident_image_candidates: 0,
            shared_event_draw_internal_drains: 0,
            shared_event_draw_rejections: [0; super::SHARED_EVENT_DRAW_REJECT_COUNT],
            shared_event_incomplete_signal_footprints: 0,
            shared_event_error_advances: 0,
            shared_event_fallbacks_no_exact_producer: 0,
            shared_event_fallbacks_timeline_overflow: 0,
            shared_event_host_effect_latency: SharedEventHostEffectLatencyStatistics::default(),
            shared_event_ack_phases: SharedEventAckPhaseStatistics::default(),
            terminal_publication: TerminalPublicationStatistics::default(),
            gpu_execution_time: Duration::ZERO,
            presentation_queue_latency: Duration::ZERO,
            last_audio_output_frames: remu_gpu::audio_output_frames(),
            last_audio_underrun_frames: remu_gpu::audio_underrun_frames(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        SHARED_EVENT_HOST_EFFECT_LATENCY_PHASE_COUNT, SharedEventAckPhaseStatistics,
        SharedEventHostEffectLatencyPhase, SharedEventHostEffectLatencyStatistics,
        SharedEventHostEffectPendingStatistics, TERMINAL_PUBLICATION_TIMING_SAMPLES,
        TerminalPublicationPhase, TerminalPublicationStatistics, TerminalPublicationTimer,
        format_shared_event_ack_phase_report, format_shared_event_host_effect_latency_report,
        format_terminal_publication_report,
    };
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    #[test]
    fn terminal_publication_statistics_count_sum_max_and_reset() {
        let mut statistics = TerminalPublicationStatistics::default();
        statistics.record(
            TerminalPublicationPhase::GlobalWriteback,
            Duration::from_nanos(10),
        );
        statistics.record(
            TerminalPublicationPhase::GlobalWriteback,
            Duration::from_nanos(30),
        );
        statistics.record(
            TerminalPublicationPhase::CompletionWake,
            Duration::from_nanos(7),
        );

        let sample = statistics.take();
        let globals = sample.phase(TerminalPublicationPhase::GlobalWriteback);
        assert_eq!(globals.count, 2);
        assert_eq!(globals.total_ns, 40);
        assert_eq!(globals.mean_ns(), 20);
        assert_eq!(globals.maximum_ns, 30);
        assert_eq!(
            sample.phase(TerminalPublicationPhase::CompletionWake).count,
            1
        );
        assert_eq!(statistics.take(), TerminalPublicationStatistics::default());
    }

    #[test]
    fn terminal_publication_report_is_compact_and_self_describing() {
        let mut statistics = TerminalPublicationStatistics::default();
        statistics.record(
            TerminalPublicationPhase::MetalTerminal,
            Duration::from_nanos(12),
        );
        statistics.record(
            TerminalPublicationPhase::CompletionWake,
            Duration::from_nanos(8),
        );

        let report = format_terminal_publication_report(5_000, statistics.take());
        assert_eq!(
            report[0],
            "[REMu][PERF-GPU-PUBLICATION-PHASE] window_ms=5000 terminal_calls=1 values=count/total_ns/mean_ns/max_ns"
        );
        assert!(report[1].contains("metal=1/12/12/12"));
        assert!(report[2].contains("globals=0/0/0/0"));
        assert!(report[3].contains("completion-wake=1/8/8/8"));
        assert!(report.iter().all(|line| line.len() < 256));
    }

    #[test]
    fn shared_event_host_effect_latency_report_separates_completed_and_open_phases() {
        assert_eq!(SHARED_EVENT_HOST_EFFECT_LATENCY_PHASE_COUNT, 3);
        let mut completed = SharedEventHostEffectLatencyStatistics::default();
        completed.record(
            SharedEventHostEffectLatencyPhase::ConsumerAck,
            Duration::from_nanos(20),
        );
        completed.record(
            SharedEventHostEffectLatencyPhase::ConsumerAck,
            Duration::from_nanos(40),
        );
        completed.record(
            SharedEventHostEffectLatencyPhase::ImmutablePrefix,
            Duration::from_nanos(7),
        );
        let mut pending = SharedEventHostEffectPendingStatistics::default();
        pending.record(
            SharedEventHostEffectLatencyPhase::HostTerminal,
            Duration::from_nanos(90),
        );

        let report = format_shared_event_host_effect_latency_report(5_000, completed, pending);
        assert!(report.starts_with(
            "[REMu][PERF-SHARED-EVENT-LATENCY] window_ms=5000 completed=count/total_ns/mean_ns/max_ns"
        ));
        assert!(report.contains("ack=2/60/30/40"));
        assert!(report.contains("ack-prefix=1/7/7/7"));
        assert!(report.contains("prefix-terminal=0/0/0/0"));
        assert!(report.contains("pending=count/total_age_ns/max_age_ns"));
        assert!(report.ends_with("prefix-terminal=1/90/90"));
    }

    #[test]
    fn disabled_terminal_publication_timer_records_nothing() {
        let before = TERMINAL_PUBLICATION_TIMING_SAMPLES.load(Ordering::Relaxed);
        let mut timer = TerminalPublicationTimer::start_if(false);
        let mut statistics = TerminalPublicationStatistics::default();
        timer.mark(&mut statistics, TerminalPublicationPhase::MetalTerminal);
        timer.mark(&mut statistics, TerminalPublicationPhase::CompletionWake);
        assert_eq!(statistics, TerminalPublicationStatistics::default());
        assert_eq!(
            TERMINAL_PUBLICATION_TIMING_SAMPLES.load(Ordering::Relaxed),
            before,
            "disabled terminal attribution must not sample the monotonic clock"
        );
    }

    #[test]
    fn shared_event_ack_phase_report_names_each_serial_boundary() {
        let mut statistics = SharedEventAckPhaseStatistics {
            samples: 1,
            draws: 2,
            computes: 3,
            ..SharedEventAckPhaseStatistics::default()
        };
        statistics.event_queue.record(Duration::from_nanos(10));
        statistics.cpu_translation.record(Duration::from_nanos(20));
        statistics.metal_encoding.record(Duration::from_nanos(30));
        statistics.ack_encoding.record(Duration::from_nanos(40));
        statistics.commit_call.record(Duration::from_nanos(50));
        statistics
            .commit_to_scheduled
            .record(Duration::from_nanos(60));
        statistics
            .scheduled_to_completed
            .record(Duration::from_nanos(70));
        statistics
            .kernel_to_gpu_start
            .record(Duration::from_nanos(80));
        statistics.gpu_execution.record(Duration::from_nanos(90));
        statistics
            .gpu_to_kernel_end
            .record(Duration::from_nanos(100));
        statistics.commit_to_ack.record(Duration::from_nanos(110));

        let report = format_shared_event_ack_phase_report(5_000, statistics).join("\n");
        assert!(report.contains("samples=1 draws=2 computes=3"));
        assert!(report.contains("event-queue=1/10/10/10"));
        assert!(report.contains("metal-encoding=1/30/30/30"));
        assert!(report.contains("commit-scheduled=1/60/60/60"));
        assert!(report.contains("gpu-execution=1/90/90/90"));
        assert!(report.contains("commit-ack=1/110/110/110"));
    }
}
