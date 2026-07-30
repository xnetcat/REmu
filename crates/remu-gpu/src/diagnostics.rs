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

type ReleaseKey = (u64, u64);

static PIXEL_PIPE_STAT_TRACED_RELEASE_COUNT: AtomicUsize = AtomicUsize::new(0);

#[derive(Default)]
struct PixelPipeStatTraceRegistry {
    releases: HashMap<ReleaseKey, VecDeque<PixelPipeStatReleaseTrace>>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_pipe_release_provenance_is_fifo_and_exact() {
        let release = GpuMemoryRelease {
            address: 0x7000_1230,
            value: GpuMemoryReleaseValue::Immediate64(1_u64 << 63),
        };
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
            take_pixel_pipe_stat_release_trace(GpuMemoryRelease {
                address: release.address + 16,
                value: release.value,
            }),
            None
        );
        assert_eq!(take_pixel_pipe_stat_release_trace(release), Some(first));
        assert_eq!(take_pixel_pipe_stat_release_trace(release), Some(second));
        assert_eq!(take_pixel_pipe_stat_release_trace(release), None);
    }
}
