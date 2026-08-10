// SPDX-License-Identifier: GPL-2.0-or-later

//! Behavior-neutral provenance for opt-in pixel-pipe query diagnostics.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::{GpuMemoryRelease, GpuMemoryReleaseValue};

/// Identifies one of the sixteen counter-pair writes synthesized for a pixel-pipe statistic dump.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PixelPipeStatReleaseTrace {
    pub dump_sequence: u64,
    pub pair_index: u8,
}

/// Correlates a sampled pair of pixel-pipe dumps with its trailing query-completion release.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PixelPipeQueryCompletionTrace {
    pub first_dump_sequence: u64,
    pub second_dump_sequence: u64,
    pub payload_address: u64,
    pub completion_address: u64,
}

type ReleaseKey = (u64, u64);

static PIXEL_PIPE_STAT_TRACED_RELEASE_COUNT: AtomicUsize = AtomicUsize::new(0);
static PIXEL_PIPE_QUERY_COMPLETION_CANDIDATE_COUNT: AtomicUsize = AtomicUsize::new(0);
static PIXEL_PIPE_QUERY_COMPLETION_TRACED_RELEASE_COUNT: AtomicUsize = AtomicUsize::new(0);

const MAX_QUERY_COMPLETION_CANDIDATES: usize = 64;

#[derive(Clone, Copy)]
struct PixelPipeStatDumpTrace {
    sequence: u64,
    destination: u64,
    sampled: bool,
}

#[derive(Default)]
struct PixelPipeStatTraceRegistry {
    releases: HashMap<ReleaseKey, VecDeque<PixelPipeStatReleaseTrace>>,
    last_dump: Option<PixelPipeStatDumpTrace>,
    completion_candidates: VecDeque<PixelPipeQueryCompletionTrace>,
    completion_releases: HashMap<ReleaseKey, VecDeque<PixelPipeQueryCompletionTrace>>,
}

fn pixel_pipe_stat_trace_registry() -> &'static Mutex<PixelPipeStatTraceRegistry> {
    static REGISTRY: OnceLock<Mutex<PixelPipeStatTraceRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(PixelPipeStatTraceRegistry::default()))
}

fn release_key(release: GpuMemoryRelease) -> Option<ReleaseKey> {
    match release.value {
        GpuMemoryReleaseValue::Immediate64(value) => Some((release.address, value)),
        GpuMemoryReleaseValue::Immediate32(_) | GpuMemoryReleaseValue::Timestamp => None,
    }
}

/// Records diagnostic provenance before an opt-in pixel-pipe result write enters the GPU queue.
///
/// The registry is populated only by the runtime's logarithmically sampled trace path. It neither
/// changes the event payload nor participates in queue scheduling.
pub fn record_pixel_pipe_stat_release_trace(
    release: GpuMemoryRelease,
    trace: PixelPipeStatReleaseTrace,
) {
    let Some(key) = release_key(release) else {
        return;
    };
    let mut registry = pixel_pipe_stat_trace_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    registry.releases.entry(key).or_default().push_back(trace);
    PIXEL_PIPE_STAT_TRACED_RELEASE_COUNT.fetch_add(1, Ordering::Release);
}

/// Takes the oldest opt-in pixel-pipe provenance matching a release at publication time.
pub fn take_pixel_pipe_stat_release_trace(
    release: GpuMemoryRelease,
) -> Option<PixelPipeStatReleaseTrace> {
    if PIXEL_PIPE_STAT_TRACED_RELEASE_COUNT.load(Ordering::Acquire) == 0 {
        return None;
    }
    let key = release_key(release)?;
    let mut registry = pixel_pipe_stat_trace_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (trace, remove_key) = {
        let traces = registry.releases.get_mut(&key)?;
        let trace = traces.pop_front();
        (trace, traces.is_empty())
    };
    if remove_key {
        registry.releases.remove(&key);
    }
    if trace.is_some() {
        PIXEL_PIPE_STAT_TRACED_RELEASE_COUNT.fetch_sub(1, Ordering::Release);
    }
    trace
}

/// Records the address relationship between consecutive pixel-pipe statistic dumps.
///
/// A native query clears a 0x100-byte payload, dumps even and odd backend counters at `payload`
/// and `payload + 8`, then publishes `1` at `payload + 0x100`. Only pairs containing one of the
/// logarithmically sampled dumps become completion candidates, and the candidate queue is bounded.
pub fn record_pixel_pipe_stat_dump_trace(sequence: u64, destination: u64, sampled: bool) {
    let mut registry = pixel_pipe_stat_trace_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let current = PixelPipeStatDumpTrace {
        sequence,
        destination,
        sampled,
    };
    let previous = registry.last_dump.replace(current);
    let Some(previous) = previous else {
        return;
    };
    if previous.sequence.checked_add(1) != Some(sequence)
        || previous.destination.checked_add(8) != Some(destination)
        || (!previous.sampled && !sampled)
    {
        return;
    }
    let Some(completion_address) = previous.destination.checked_add(0x100) else {
        return;
    };
    if registry.completion_candidates.len() == MAX_QUERY_COMPLETION_CANDIDATES {
        registry.completion_candidates.pop_front();
        PIXEL_PIPE_QUERY_COMPLETION_CANDIDATE_COUNT.fetch_sub(1, Ordering::Release);
    }
    registry
        .completion_candidates
        .push_back(PixelPipeQueryCompletionTrace {
            first_dump_sequence: previous.sequence,
            second_dump_sequence: sequence,
            payload_address: previous.destination,
            completion_address,
        });
    PIXEL_PIPE_QUERY_COMPLETION_CANDIDATE_COUNT.fetch_add(1, Ordering::Release);
}

/// Moves the oldest exact query-completion candidate to publication provenance.
///
/// This observes only the diagnostic registry. It does not modify, delay, or synthesize the
/// release itself.
pub fn record_pixel_pipe_query_completion_release_trace(
    release: GpuMemoryRelease,
) -> Option<PixelPipeQueryCompletionTrace> {
    if PIXEL_PIPE_QUERY_COMPLETION_CANDIDATE_COUNT.load(Ordering::Acquire) == 0 {
        return None;
    }
    let GpuMemoryRelease {
        address,
        value: GpuMemoryReleaseValue::Immediate64(1),
        ..
    } = release
    else {
        return None;
    };
    let mut registry = pixel_pipe_stat_trace_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let candidate_index = registry
        .completion_candidates
        .iter()
        .position(|candidate| candidate.completion_address == address)?;
    let trace = registry
        .completion_candidates
        .remove(candidate_index)
        .expect("the located query-completion candidate exists");
    PIXEL_PIPE_QUERY_COMPLETION_CANDIDATE_COUNT.fetch_sub(1, Ordering::Release);
    registry
        .completion_releases
        .entry((address, 1))
        .or_default()
        .push_back(trace);
    PIXEL_PIPE_QUERY_COMPLETION_TRACED_RELEASE_COUNT.fetch_add(1, Ordering::Release);
    Some(trace)
}

/// Takes the oldest query-completion provenance matching a release at publication time.
pub fn take_pixel_pipe_query_completion_release_trace(
    release: GpuMemoryRelease,
) -> Option<PixelPipeQueryCompletionTrace> {
    if PIXEL_PIPE_QUERY_COMPLETION_TRACED_RELEASE_COUNT.load(Ordering::Acquire) == 0 {
        return None;
    }
    let key = release_key(release)?;
    let mut registry = pixel_pipe_stat_trace_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (trace, remove_key) = {
        let traces = registry.completion_releases.get_mut(&key)?;
        let trace = traces.pop_front();
        (trace, traces.is_empty())
    };
    if remove_key {
        registry.completion_releases.remove(&key);
    }
    if trace.is_some() {
        PIXEL_PIPE_QUERY_COMPLETION_TRACED_RELEASE_COUNT.fetch_sub(1, Ordering::Release);
    }
    trace
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_pipe_release_provenance_is_fifo_and_exact() {
        let release =
            GpuMemoryRelease::label(0x7000_1230, GpuMemoryReleaseValue::Immediate64(1_u64 << 63));
        let first = PixelPipeStatReleaseTrace {
            dump_sequence: 3,
            pair_index: 4,
        };
        let second = PixelPipeStatReleaseTrace {
            dump_sequence: 5,
            pair_index: 6,
        };
        record_pixel_pipe_stat_release_trace(release, first);
        record_pixel_pipe_stat_release_trace(release, second);

        assert_eq!(
            take_pixel_pipe_stat_release_trace(GpuMemoryRelease::label(
                release.address + 16,
                release.value
            )),
            None
        );
        assert_eq!(take_pixel_pipe_stat_release_trace(release), Some(first));
        assert_eq!(take_pixel_pipe_stat_release_trace(release), Some(second));
        assert_eq!(take_pixel_pipe_stat_release_trace(release), None);
    }

    #[test]
    fn sampled_dump_pair_traces_only_its_exact_completion_release() {
        let payload_address = 0x7000_5200;
        record_pixel_pipe_stat_dump_trace(1001, payload_address, false);
        record_pixel_pipe_stat_dump_trace(1002, payload_address + 8, true);
        let release = GpuMemoryRelease::label(
            payload_address + 0x100,
            GpuMemoryReleaseValue::Immediate64(1),
        );

        assert_eq!(
            record_pixel_pipe_query_completion_release_trace(GpuMemoryRelease::label(
                release.address,
                GpuMemoryReleaseValue::Immediate64(2)
            )),
            None
        );
        assert_eq!(
            record_pixel_pipe_query_completion_release_trace(GpuMemoryRelease::label(
                release.address + 8,
                release.value
            )),
            None
        );
        let trace = record_pixel_pipe_query_completion_release_trace(release)
            .expect("the exact trailing completion release is traced");
        assert_eq!(
            trace,
            PixelPipeQueryCompletionTrace {
                first_dump_sequence: 1001,
                second_dump_sequence: 1002,
                payload_address,
                completion_address: release.address,
            }
        );
        assert_eq!(
            take_pixel_pipe_query_completion_release_trace(release),
            Some(trace)
        );
        assert_eq!(
            take_pixel_pipe_query_completion_release_trace(release),
            None
        );
    }
}
