// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! The mutex-protected registry of watched guest ranges and their armed host pages.

use std::collections::BTreeMap;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use super::page::{
    HOST_PAGE_SIZE, align_down, align_up, host_page_size_infallible, protect_page, protect_range,
    protect_range_no_access,
};
use super::snapshot::publish_snapshot;

#[derive(Debug)]
pub(super) struct WatchedPage {
    pub(super) address: u64,
    pub(super) dirty: AtomicBool,
    pub(super) armed: AtomicBool,
    /// Read-watch only: whether the page's GPU-written bytes have been materialized into guest
    /// memory. A freshly armed read-watch page is unmaterialized (stale); the first faulting guest
    /// read materializes it, and a faulting guest write marks it materialized without a copy
    /// because the guest is overwriting the stale bytes anyway. Always `true` for a write-watch
    /// page, whose semantics do not involve materialization.
    pub(super) materialized: AtomicBool,
}

#[derive(Debug)]
pub(super) struct WatchedRange {
    pub(super) id: u64,
    pub(super) address: u64,
    pub(super) byte_count: u64,
    pub(super) precise_tracking: bool,
    /// A read-watch arms its pages with no access so a guest READ faults and materializes the
    /// GPU-written bytes on demand (audit D5). A plain write-watch arms read-only and only writes
    /// fault. Read-watches are always precise and never demote.
    pub(super) read_watch: bool,
    pub(super) generation: AtomicU64,
    pub(super) dirty: AtomicBool,
    pub(super) retirement: AtomicU8,
    /// The `generation` value recorded the last time this range was armed. A rearm whose current
    /// generation differs saw at least one store during the previous armed window, i.e. a
    /// "dirty cycle"; this is how [`crate::CpuWriteWatch::rearm`] detects chronic dirtiness without
    /// touching the hot fault handler.
    pub(super) armed_generation: AtomicU64,
    /// Consecutive dirty rearm cycles. Once it reaches [`DEMOTION_THRESHOLD`] the range demotes.
    pub(super) dirty_rearm_streak: AtomicU64,
    /// A chronically-dirty range stops arming its pages (no more protection faults) and instead
    /// reports itself perpetually dirty, so its consumer revalidates the bytes at acquire time.
    /// This inverts the cost model for a resource the guest rewrites every frame: the per-frame
    /// fault storm on a shared host page (false sharing) is worse than an unconditional reread.
    /// It is the acquire-time-revalidation stage the plan places between coalesced arms and the
    /// D5 read-watch inversion.
    pub(super) demoted: AtomicBool,
    pub(super) pages: Box<[WatchedPage]>,
}

/// Consecutive dirty rearm cycles a precise watch tolerates before demoting to acquire-time
/// revalidation. Sized so a genuinely hot resident (rewritten every frame) demotes within about a
/// second at 60 Hz, while a resource touched occasionally never does.
pub(super) const DEMOTION_THRESHOLD: u64 = 64;

pub(super) const WATCH_ACTIVE: u8 = 0;

pub(super) const WATCH_REPROTECTING: u8 = 1;

pub(super) const WATCH_REPROTECTED: u8 = 2;

pub(super) const WATCH_REPLACED: u8 = 3;

#[derive(Default)]
pub(super) struct WatchState {
    pub(super) next_id: u64,
    pub(super) ranges: BTreeMap<u64, Arc<WatchedRange>>,
    /// Page-aligned live mappings whose last active watch was disarmed and removed.
    ///
    /// A protection fault can already be committed on another CPU when `mprotect(PROT_WRITE)`
    /// returns. Retaining the mapping interval lets its delayed signal recover after the active
    /// owner leaves the immutable snapshot. Guest mapping operations remove these intervals before
    /// replacing or reprotecting the corresponding pages.
    pub(super) writable_ranges: BTreeMap<u64, u64>,
}

static WATCH_STATE: OnceLock<Mutex<WatchState>> = OnceLock::new();

pub(super) fn retire_overlapping(
    address: u64,
    byte_count: u64,
    retirement: u8,
) -> Vec<Arc<WatchedRange>> {
    let Some(end) = address.checked_add(byte_count) else {
        return Vec::new();
    };
    let mut state = state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let page_size = HOST_PAGE_SIZE.load(Ordering::Acquire);
    let (page_start, page_end) = if page_size == 0 {
        (address, end)
    } else {
        (
            align_down(address, page_size),
            align_up(end, page_size).unwrap_or(u64::MAX),
        )
    };
    let forgot_writable_range = forget_writable_range(&mut state, page_start, page_end);
    let retired_ids = state
        .ranges
        .iter()
        .filter_map(|(&id, range)| {
            let range_end = range.address.saturating_add(range.byte_count);
            (range.address < end && address < range_end).then_some(id)
        })
        .collect::<Vec<_>>();
    if retired_ids.is_empty() {
        if forgot_writable_range {
            publish_snapshot(&state);
        }
        return Vec::new();
    }
    let retired = retired_ids
        .into_iter()
        .filter_map(|id| state.ranges.remove(&id))
        .collect::<Vec<_>>();
    for range in &retired {
        range.retirement.store(retirement, Ordering::Release);
        for page in &range.pages {
            page.armed.store(false, Ordering::Release);
        }
    }
    // Keep the old snapshot published until every page is either writable or remains protected
    // by an owner represented in the replacement snapshot. A guest store racing retirement can
    // then be handled through the old snapshot instead of becoming an unowned protection fault.
    for range in &retired {
        restore_unowned_pages(range, &state);
    }
    // `mprotect(PROT_WRITE)` can return while a protection fault for the same page is already
    // committed on another CPU, so the delayed signal arrives after these ranges have left the
    // snapshot. `CpuWriteWatch::drop` retains the mapping interval for exactly that reason;
    // permission-only retirements need the same treatment, because the mapping stays live and the
    // guest is still allowed to write to it. Replacements deliberately do not: their pages are
    // about to be remapped, and recovering a fault there would grant silent write access to
    // memory the mapping change took away.
    if retirement == WATCH_REPROTECTING {
        for range in &retired {
            remember_writable_range(&mut state, range);
        }
    }
    publish_snapshot(&state);
    retired
}

pub(super) fn state() -> &'static Mutex<WatchState> {
    WATCH_STATE.get_or_init(|| Mutex::new(WatchState::default()))
}

pub(super) fn remember_writable_range(state: &mut WatchState, range: &WatchedRange) {
    if !range.precise_tracking {
        return;
    }
    let Some(first) = range.pages.first() else {
        return;
    };
    let Some(last) = range.pages.last() else {
        return;
    };
    let start = first.address;
    let end = last.address.saturating_add(host_page_size_infallible());
    let mut merged_start = start;
    let mut merged_end = end;
    let overlapping = state
        .writable_ranges
        .range(..=end)
        .filter_map(|(&candidate_start, &candidate_end)| {
            (candidate_end >= start).then_some((candidate_start, candidate_end))
        })
        .collect::<Vec<_>>();
    for (candidate_start, candidate_end) in overlapping {
        state.writable_ranges.remove(&candidate_start);
        merged_start = merged_start.min(candidate_start);
        merged_end = merged_end.max(candidate_end);
    }
    state.writable_ranges.insert(merged_start, merged_end);
}

fn forget_writable_range(state: &mut WatchState, start: u64, end: u64) -> bool {
    if start >= end {
        return false;
    }
    let overlapping = state
        .writable_ranges
        .range(..end)
        .filter_map(|(&candidate_start, &candidate_end)| {
            (candidate_end > start).then_some((candidate_start, candidate_end))
        })
        .collect::<Vec<_>>();
    let changed = !overlapping.is_empty();
    for (candidate_start, candidate_end) in overlapping {
        state.writable_ranges.remove(&candidate_start);
        if candidate_start < start {
            state.writable_ranges.insert(candidate_start, start);
        }
        if end < candidate_end {
            state.writable_ranges.insert(end, candidate_end);
        }
    }
    changed
}

pub(super) fn arm_range(range: &WatchedRange, state: &WatchState) -> io::Result<()> {
    if !range.precise_tracking {
        return Ok(());
    }
    // A chronically-dirty range does acquire-time revalidation instead of protection faulting:
    // release write protection on any page it still owns and report itself dirty. It must not be
    // left armed, or the fault handler would keep servicing faults it no longer wants.
    if range.demoted.load(Ordering::Acquire) {
        for page in &range.pages {
            page.dirty.store(true, Ordering::Release);
            page.armed.store(false, Ordering::Release);
        }
        // Restore the now-unowned pages with the same ownership bitmap and contiguous-run
        // coalescing used by teardown. Calling restore_page_if_unowned once per armed tail page
        // made the one-time demotion of a large texture O(pages * all watched pages) plus one
        // mprotect syscall per page.
        restore_unowned_pages(range, state);
        range.dirty.store(true, Ordering::Release);
        return Ok(());
    }
    let page_size = host_page_size_infallible();
    let mut changed = Vec::with_capacity(range.pages.len());
    // First reserve the armed state and clear dirtiness for every page, recording which pages this
    // call newly armed. Protection then follows in maximal contiguous runs: a large resident used
    // to cost one mprotect per host page, which dominated presenter time after an alias writeback.
    for (index, page) in range.pages.iter().enumerate() {
        let was_dirty = page.dirty.swap(false, Ordering::AcqRel);
        // Arming a read-watch re-stales its pages: the GPU may have rewritten the range, so the
        // next guest read must materialize again rather than trust bytes from the previous window.
        if range.read_watch {
            page.materialized.store(false, Ordering::Release);
        }
        let newly_armed = !page.armed.swap(true, Ordering::AcqRel);
        changed.push((index, newly_armed, was_dirty));
    }
    let mut index = 0;
    while index < changed.len() {
        if !changed[index].1 {
            index += 1;
            continue;
        }
        let run_start = index;
        index += 1;
        while index < changed.len() && changed[index].1 {
            index += 1;
        }
        let address = range.pages[run_start].address;
        let byte_count = (index - run_start) as u64 * page_size;
        let protect_result = if range.read_watch {
            protect_range_no_access(address, byte_count)
        } else {
            protect_range(address, byte_count, true)
        };
        if let Err(error) = protect_result {
            rollback_arm(range, state, &changed);
            return Err(error);
        }
        // A write can race the transition when another overlapping watch already protects one of
        // these pages: its handler disarms us before making the page writable. Do not leave any
        // page in the run read-only with no armed owner after our mprotect returns.
        for page in &range.pages[run_start..index] {
            if !page.armed.load(Ordering::Acquire) {
                restore_page_if_unowned(page.address, state);
            }
        }
    }
    range
        .armed_generation
        .store(range.generation.load(Ordering::Acquire), Ordering::Release);
    Ok(())
}

fn rollback_arm(range: &WatchedRange, state: &WatchState, changed: &[(usize, bool, bool)]) {
    for &(index, newly_armed, was_dirty) in changed {
        let page = &range.pages[index];
        if was_dirty {
            page.dirty.store(true, Ordering::Release);
        }
        if newly_armed {
            page.armed.store(false, Ordering::Release);
        }
    }
    for &(index, newly_armed, _) in changed {
        if newly_armed {
            restore_page_if_unowned(range.pages[index].address, state);
        }
    }
}

pub(super) fn restore_unowned_pages(range: &WatchedRange, state: &WatchState) {
    if !range.precise_tracking {
        return;
    }
    let Some(first_page) = range.pages.first() else {
        return;
    };
    let Some(last_page) = range.pages.last() else {
        return;
    };
    let page_size = host_page_size_infallible();
    let retired_start = first_page.address;
    let retired_end = last_page.address.saturating_add(page_size);

    // Teardown used to ask every remaining watch about every page in the retiring range. A large
    // render allocation therefore cost O(retired pages * all watched pages), and Silent Hill
    // spent most of its presenter time here after an alias writeback. Build the ownership bitmap
    // once by intersecting each remaining contiguous watch with this range instead.
    let mut owned = vec![false; range.pages.len()];
    for candidate in state.ranges.values().filter(|range| range.precise_tracking) {
        let Some(candidate_first) = candidate.pages.first() else {
            continue;
        };
        let Some(candidate_last) = candidate.pages.last() else {
            continue;
        };
        let candidate_start = candidate_first.address;
        let candidate_end = candidate_last.address.saturating_add(page_size);
        let overlap_start = retired_start.max(candidate_start);
        let overlap_end = retired_end.min(candidate_end);
        if overlap_start >= overlap_end {
            continue;
        }
        let retired_index =
            usize::try_from((overlap_start - retired_start) / page_size).unwrap_or(usize::MAX);
        let candidate_index =
            usize::try_from((overlap_start - candidate_start) / page_size).unwrap_or(usize::MAX);
        let overlap_pages = usize::try_from((overlap_end - overlap_start) / page_size).unwrap_or(0);
        if retired_index == usize::MAX
            || candidate_index == usize::MAX
            || retired_index.saturating_add(overlap_pages) > owned.len()
            || candidate_index.saturating_add(overlap_pages) > candidate.pages.len()
        {
            continue;
        }
        for offset in 0..overlap_pages {
            if candidate.pages[candidate_index + offset]
                .armed
                .load(Ordering::Acquire)
            {
                owned[retired_index + offset] = true;
            }
        }
    }

    // One mprotect accepts a complete contiguous interval. Coalescing adjacent unowned pages
    // avoids thousands of host syscalls when a large texture has no other live aliases.
    let mut index = 0;
    while index < owned.len() {
        if owned[index] {
            index += 1;
            continue;
        }
        let start = index;
        index += 1;
        while index < owned.len() && !owned[index] {
            index += 1;
        }
        let address = retired_start.saturating_add(start as u64 * page_size);
        let byte_count = (index - start) as u64 * page_size;
        let _ = protect_range(address, byte_count, false);
    }
}

fn restore_page_if_unowned(page_address: u64, state: &WatchState) {
    if state.ranges.values().any(|range| {
        range
            .pages
            .iter()
            .any(|page| page.address == page_address && page.armed.load(Ordering::Acquire))
    }) {
        return;
    }
    let _ = protect_page(page_address, false);
}
