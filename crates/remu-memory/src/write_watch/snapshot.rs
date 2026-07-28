// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! The immutable, signal-handler-readable snapshot of the watched-page registry.

use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use super::page::host_page_size_infallible;
use super::state::{WatchState, WatchedRange};

#[derive(Clone, Copy)]
pub(super) struct PageOwner {
    pub(super) range: *const WatchedRange,
    pub(super) page_index: usize,
}

#[derive(Clone, Copy)]
struct SnapshotRange {
    start: u64,
    end: u64,
    prefix_max_end: u64,
    range: *const WatchedRange,
}

pub(super) struct WatchSnapshot {
    ranges: Box<[SnapshotRange]>,
    writable_ranges: Box<[(u64, u64)]>,
}

impl WatchSnapshot {
    fn from_state(state: &WatchState) -> Self {
        let page_size = host_page_size_infallible();
        let mut ranges = Vec::with_capacity(state.ranges.len());
        for range in state.ranges.values().filter(|range| range.precise_tracking) {
            let Some(first) = range.pages.first() else {
                continue;
            };
            let Some(last) = range.pages.last() else {
                continue;
            };
            ranges.push(SnapshotRange {
                start: first.address,
                end: last.address.saturating_add(page_size),
                prefix_max_end: 0,
                range: Arc::as_ptr(range),
            });
        }
        ranges.sort_unstable_by_key(|range| range.start);
        let mut prefix_max_end = 0;
        for range in &mut ranges {
            prefix_max_end = prefix_max_end.max(range.end);
            range.prefix_max_end = prefix_max_end;
        }
        Self {
            ranges: ranges.into_boxed_slice(),
            writable_ranges: state
                .writable_ranges
                .iter()
                .map(|(&start, &end)| (start, end))
                .collect(),
        }
    }

    pub(super) fn owners(&self, address: u64) -> SnapshotOwners<'_> {
        SnapshotOwners {
            ranges: &self.ranges,
            address,
            next_index: self.ranges.partition_point(|range| range.start <= address),
        }
    }

    pub(super) fn contains_page(&self, address: u64) -> bool {
        self.owners(address).next().is_some()
    }

    pub(super) fn contains_writable_page(&self, address: u64) -> bool {
        let index = self
            .writable_ranges
            .partition_point(|&(start, _)| start <= address);
        index != 0 && address < self.writable_ranges[index - 1].1
    }
}

pub(super) struct SnapshotOwners<'a> {
    ranges: &'a [SnapshotRange],
    address: u64,
    next_index: usize,
}

impl Iterator for SnapshotOwners<'_> {
    type Item = PageOwner;

    fn next(&mut self) -> Option<Self::Item> {
        while self.next_index != 0 {
            self.next_index -= 1;
            let candidate = self.ranges[self.next_index];
            if candidate.prefix_max_end <= self.address {
                self.next_index = 0;
                return None;
            }
            if self.address < candidate.end {
                let page_index =
                    usize::try_from((self.address - candidate.start) / host_page_size_infallible())
                        .expect("a watched-page index fits usize");
                return Some(PageOwner {
                    range: candidate.range,
                    page_index,
                });
            }
        }
        None
    }
}

pub(super) static WATCH_SNAPSHOT: AtomicPtr<WatchSnapshot> = AtomicPtr::new(std::ptr::null_mut());

pub(super) static ACTIVE_HANDLERS: AtomicUsize = AtomicUsize::new(0);

pub(super) fn publish_snapshot(state: &WatchState) {
    let snapshot = Box::new(WatchSnapshot::from_state(state));
    let previous = WATCH_SNAPSHOT.swap(Box::into_raw(snapshot), Ordering::SeqCst);

    // A handler increments before loading the pointer. With sequential consistency, it therefore
    // either pins the old snapshot before this swap or observes the new snapshot after it.
    while ACTIVE_HANDLERS.load(Ordering::SeqCst) != 0 {
        std::hint::spin_loop();
    }
    if !previous.is_null() {
        // SAFETY: the pointer originated from Box::into_raw, was removed atomically above, and no
        // signal handler can still hold a reference after the quiescence wait.
        unsafe { drop(Box::from_raw(previous)) };
    }
}
