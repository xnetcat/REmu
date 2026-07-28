// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Signal-safe dirty-page tracking for CPU writes to guest unified memory.
//!
//! Metal resources cannot alias arbitrary guest mappings. A backend can use this module to keep
//! a resident Metal resource and refresh only the host pages that the guest CPU has changed.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::io;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Debug)]
struct WatchedPage {
    address: u64,
    dirty: AtomicBool,
    armed: AtomicBool,
}

#[derive(Debug)]
struct WatchedRange {
    id: u64,
    address: u64,
    byte_count: u64,
    precise_tracking: bool,
    generation: AtomicU64,
    dirty: AtomicBool,
    retirement: AtomicU8,
    pages: Box<[WatchedPage]>,
}

const WATCH_ACTIVE: u8 = 0;
const WATCH_REPROTECTING: u8 = 1;
const WATCH_REPROTECTED: u8 = 2;
const WATCH_REPLACED: u8 = 3;

/// Why a CPU write watch stopped observing its original mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuWriteWatchRetirement {
    /// A guest `mprotect` is currently changing the watched pages.
    Reprotecting,
    /// The original mapping remains live after a guest protection change.
    Reprotected,
    /// The original mapping was unmapped or replaced with different storage.
    Replaced,
}

/// Keeps a reprotection distinguishable from a mapping replacement until `mprotect` finishes.
///
/// Dropping the guard publishes completion. This makes early-return error paths safe while letting
/// resource owners preserve GPU-only contents across permission-only changes.
#[must_use]
pub struct CpuWriteWatchReprotection {
    ranges: Vec<Arc<WatchedRange>>,
}

impl Drop for CpuWriteWatchReprotection {
    fn drop(&mut self) {
        for range in &self.ranges {
            let _ = range.retirement.compare_exchange(
                WATCH_REPROTECTING,
                WATCH_REPROTECTED,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }
}

#[derive(Clone, Copy)]
struct PageOwner {
    range: *const WatchedRange,
    page_index: usize,
}

#[derive(Clone, Copy)]
struct SnapshotRange {
    start: u64,
    end: u64,
    prefix_max_end: u64,
    range: *const WatchedRange,
}

struct WatchSnapshot {
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

    fn owners(&self, address: u64) -> SnapshotOwners<'_> {
        SnapshotOwners {
            ranges: &self.ranges,
            address,
            next_index: self.ranges.partition_point(|range| range.start <= address),
        }
    }

    fn contains_page(&self, address: u64) -> bool {
        self.owners(address).next().is_some()
    }

    fn contains_writable_page(&self, address: u64) -> bool {
        let index = self
            .writable_ranges
            .partition_point(|&(start, _)| start <= address);
        index != 0 && address < self.writable_ranges[index - 1].1
    }
}

struct SnapshotOwners<'a> {
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

#[derive(Default)]
struct WatchState {
    next_id: u64,
    ranges: BTreeMap<u64, Arc<WatchedRange>>,
    /// Page-aligned live mappings whose last active watch was disarmed and removed.
    ///
    /// A protection fault can already be committed on another CPU when `mprotect(PROT_WRITE)`
    /// returns. Retaining the mapping interval lets its delayed signal recover after the active
    /// owner leaves the immutable snapshot. Guest mapping operations remove these intervals before
    /// replacing or reprotecting the corresponding pages.
    writable_ranges: BTreeMap<u64, u64>,
}

static WATCH_STATE: OnceLock<Mutex<WatchState>> = OnceLock::new();
static WATCH_SNAPSHOT: AtomicPtr<WatchSnapshot> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_HANDLERS: AtomicUsize = AtomicUsize::new(0);
static HOST_PAGE_SIZE: AtomicU64 = AtomicU64::new(0);
static RECOVERED_DISARMED_FAULTS: AtomicU64 = AtomicU64::new(0);
static HANDLER_RESULT: OnceLock<Result<(), i32>> = OnceLock::new();

struct PreviousAction(std::cell::UnsafeCell<MaybeUninit<libc::sigaction>>);

// SAFETY: installation initializes the slot before a watched page is published and the stored
// action is immutable afterwards. Only the signal handler reads it.
unsafe impl Sync for PreviousAction {}

impl PreviousAction {
    const fn new() -> Self {
        Self(std::cell::UnsafeCell::new(MaybeUninit::uninit()))
    }

    const fn as_mut_ptr(&self) -> *mut libc::sigaction {
        self.0.get().cast::<libc::sigaction>()
    }

    unsafe fn get(&self) -> &libc::sigaction {
        // SAFETY: successful handler installation initializes both previous-action slots.
        unsafe { &*self.as_mut_ptr() }
    }
}

static PREVIOUS_SIGBUS: PreviousAction = PreviousAction::new();
static PREVIOUS_SIGSEGV: PreviousAction = PreviousAction::new();

/// A live CPU write watch over a guest-memory range.
///
/// Protection and dirty tracking operate at host-page granularity. Consequently, disjoint ranges
/// sharing one host page become dirty together. [`Self::dirty_page_ranges`] clips those pages back
/// to the exact byte range requested by the caller.
#[derive(Debug)]
pub struct CpuWriteWatch {
    range: Arc<WatchedRange>,
}

impl CpuWriteWatch {
    /// Starts watching a readable and writable mapped guest range.
    ///
    /// The range may overlap other watches. The mapping must not be unmapped or have its page
    /// permissions independently changed while the returned handle is alive.
    ///
    /// # Safety
    ///
    /// `address..address + byte_count` must belong to a live readable and writable mapping whose
    /// complete host pages can safely alternate between read-only and read/write protection.
    pub unsafe fn new(address: u64, byte_count: u64) -> io::Result<Self> {
        if address == 0 || byte_count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a CPU write watch requires a nonempty mapped range",
            ));
        }
        let range_end = address.checked_add(byte_count).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "CPU write watch range overflows",
            )
        })?;
        let page_size = host_page_size()?;
        let recorded_page_size = HOST_PAGE_SIZE.load(Ordering::Acquire);
        if recorded_page_size == 0 {
            HOST_PAGE_SIZE.store(page_size, Ordering::Release);
        } else if recorded_page_size != page_size {
            return Err(io::Error::other(
                "host page size changed after CPU write-watch initialization",
            ));
        }
        let page_start = align_down(address, page_size);
        let page_end = align_up(range_end, page_size).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "CPU write watch page range overflows",
            )
        })?;
        let page_count = usize::try_from((page_end - page_start) / page_size).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "CPU write watch page count does not fit this host",
            )
        })?;
        let precise_tracking =
            std::env::var_os("REMU_DISABLE_CPU_WRITE_WATCH").is_none_or(|value| value != "1");
        let mut pages = Vec::new();
        pages.try_reserve_exact(page_count).map_err(|error| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                format!("could not allocate CPU write-watch page state: {error}"),
            )
        })?;
        for index in 0..page_count {
            pages.push(WatchedPage {
                address: page_start + index as u64 * page_size,
                dirty: AtomicBool::new(false),
                armed: AtomicBool::new(false),
            });
        }

        if precise_tracking {
            install_fault_handlers()?;
        }
        let mut state = state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.next_id = state
            .next_id
            .checked_add(1)
            .ok_or_else(|| io::Error::other("CPU write-watch identifier space was exhausted"))?;
        let range = Arc::new(WatchedRange {
            id: state.next_id,
            address,
            byte_count,
            precise_tracking,
            generation: AtomicU64::new(0),
            dirty: AtomicBool::new(false),
            retirement: AtomicU8::new(WATCH_ACTIVE),
            pages: pages.into_boxed_slice(),
        });
        state.ranges.insert(range.id, Arc::clone(&range));
        publish_snapshot(&state);
        if let Err(error) = arm_range(&range, &state) {
            state.ranges.remove(&range.id);
            restore_unowned_pages(&range, &state);
            remember_writable_range(&mut state, &range);
            publish_snapshot(&state);
            return Err(error);
        }
        Ok(Self { range })
    }

    /// Returns whether a CPU write was observed since the last call, clearing that indication.
    ///
    /// Call [`Self::rearm`] after consuming or copying the changed bytes. Dirty-page detail remains
    /// available until a successful rearm.
    #[must_use]
    pub fn consume_dirty(&self) -> bool {
        self.range.dirty.swap(false, Ordering::AcqRel)
    }

    /// Returns whether a CPU write has been observed without clearing the indication.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.range.dirty.load(Ordering::Acquire)
    }

    /// Returns whether this watch uses page protection to observe exact CPU writes.
    ///
    /// `REMU_DISABLE_CPU_WRITE_WATCH=1` keeps mapping-lifetime validation active but assumes CPU
    /// texture contents remain unchanged. This intentionally imprecise mode is only for debuggers
    /// that intercept deliberate protection faults before the process handler can service them.
    #[must_use]
    pub fn precise_tracking(&self) -> bool {
        self.range.precise_tracking
    }

    /// Returns the number of protected host pages faulted since this watch was created.
    ///
    /// The counter is monotonically increasing. One guest store can increment multiple watches
    /// that share the same host page, but increments this watch at most once for that page between
    /// rearms.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.range.generation.load(Ordering::Acquire)
    }

    /// Returns dirty host-page intersections as exact `(address, byte_count)` guest ranges.
    ///
    /// Each element corresponds to one host page. The first and last elements are clipped to the
    /// exact watched range and may therefore be smaller than a host page.
    #[must_use]
    pub fn dirty_page_ranges(&self) -> Vec<(u64, u64)> {
        let range_end = self.range.address + self.range.byte_count;
        self.range
            .pages
            .iter()
            .filter(|page| page.dirty.load(Ordering::Acquire))
            .map(|page| {
                let page_end = page.address + host_page_size_infallible();
                let start = page.address.max(self.range.address);
                let end = page_end.min(range_end);
                (start, end - start)
            })
            .collect()
    }

    /// Reinstates read-only protection after the caller has consumed current guest bytes.
    ///
    /// Dirty-page detail is cleared immediately before each page is rearmed. A write after this
    /// method returns is guaranteed to fault and become visible to this watch.
    pub fn rearm(&self) -> io::Result<()> {
        if self.is_retired() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "CPU write watch was retired by a guest mapping change",
            ));
        }
        let state = state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.is_retired() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "CPU write watch was retired by a guest mapping change",
            ));
        }
        arm_range(&self.range, &state)
    }

    /// Runs an operation while the watched guest mapping is pinned against replacement.
    ///
    /// Mapping mutation retires overlapping watches under the same registry lock before changing
    /// host VM state. Holding that lock from the final retirement check through a GPU-to-CPU
    /// writeback therefore prevents an unmap from opening a use-after-unmap window.
    pub fn with_live_mapping<T>(&self, operation: impl FnOnce() -> T) -> io::Result<T> {
        let state = state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.is_retired() || !state.ranges.contains_key(&self.range.id) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "CPU write-watch mapping is no longer live",
            ));
        }
        Ok(operation())
    }

    /// Returns whether the guest mapping underneath this watch was replaced or reprotected.
    #[must_use]
    pub fn is_retired(&self) -> bool {
        self.retirement().is_some()
    }

    /// Returns why this watch no longer observes its original mapping.
    #[must_use]
    pub fn retirement(&self) -> Option<CpuWriteWatchRetirement> {
        match self.range.retirement.load(Ordering::Acquire) {
            WATCH_ACTIVE => None,
            WATCH_REPROTECTING => Some(CpuWriteWatchRetirement::Reprotecting),
            WATCH_REPROTECTED => Some(CpuWriteWatchRetirement::Reprotected),
            WATCH_REPLACED => Some(CpuWriteWatchRetirement::Replaced),
            _ => unreachable!("CPU write-watch retirement has an invalid state"),
        }
    }

    /// Retires every write watch intersecting a guest mapping operation.
    ///
    /// Guest address-space mutation calls this before `mmap`/`mprotect`. A watch must never
    /// restore read/write access to a replacement mapping when its handle is eventually dropped,
    /// nor may its signal handler treat that replacement as the allocation it originally watched.
    pub fn retire_overlapping(address: u64, byte_count: u64) {
        let _ = retire_overlapping(address, byte_count, WATCH_REPLACED);
    }

    /// Temporarily retires watches while permissions change without replacing their mappings.
    ///
    /// The caller must hold the returned guard until its `mprotect` attempt has finished. Resource
    /// owners may then install a fresh watch while retaining GPU-private contents.
    pub fn begin_reprotect_overlapping(address: u64, byte_count: u64) -> CpuWriteWatchReprotection {
        CpuWriteWatchReprotection {
            ranges: retire_overlapping(address, byte_count, WATCH_REPROTECTING),
        }
    }

    /// Returns the exact requested guest range rather than its page-aligned protection span.
    #[must_use]
    pub fn range(&self) -> (u64, u64) {
        (self.range.address, self.range.byte_count)
    }
}

fn retire_overlapping(address: u64, byte_count: u64, retirement: u8) -> Vec<Arc<WatchedRange>> {
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
    publish_snapshot(&state);
    retired
}

impl Drop for CpuWriteWatch {
    fn drop(&mut self) {
        let mut state = state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.ranges.remove(&self.range.id).is_none() {
            return;
        }

        self.range
            .retirement
            .store(WATCH_REPLACED, Ordering::Release);

        for page in &self.range.pages {
            page.armed.store(false, Ordering::Release);
        }
        // Restore access while the old snapshot still pins this range. A racing guest store either
        // observes a writable page or faults through a snapshot that proves the mapping is live.
        // Pages owned by another armed watch stay protected and remain in the new snapshot.
        restore_unowned_pages(&self.range, &state);
        remember_writable_range(&mut state, &self.range);
        publish_snapshot(&state);
    }
}

fn state() -> &'static Mutex<WatchState> {
    WATCH_STATE.get_or_init(|| Mutex::new(WatchState::default()))
}

fn publish_snapshot(state: &WatchState) {
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

fn remember_writable_range(state: &mut WatchState, range: &WatchedRange) {
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

fn arm_range(range: &WatchedRange, state: &WatchState) -> io::Result<()> {
    if !range.precise_tracking {
        return Ok(());
    }
    let mut changed = Vec::with_capacity(range.pages.len());
    for (index, page) in range.pages.iter().enumerate() {
        let was_dirty = page.dirty.swap(false, Ordering::AcqRel);
        if page.armed.swap(true, Ordering::AcqRel) {
            changed.push((index, false, was_dirty));
            continue;
        }
        changed.push((index, true, was_dirty));
        if let Err(error) = protect_page(page.address, true) {
            rollback_arm(range, state, &changed);
            return Err(error);
        }
        // A write can race the transition when another overlapping watch already protects this
        // page. Its handler disarms us before making the page writable; do not leave that page
        // read-only with no armed owner after our mprotect returns.
        if !page.armed.load(Ordering::Acquire) {
            restore_page_if_unowned(page.address, state);
        }
    }
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

fn restore_unowned_pages(range: &WatchedRange, state: &WatchState) {
    if !range.precise_tracking {
        return;
    }
    for page in &range.pages {
        restore_page_if_unowned(page.address, state);
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

fn protect_page(page_address: u64, read_only: bool) -> io::Result<()> {
    protect_range(page_address, host_page_size_infallible(), read_only)
}

fn protect_range(address: u64, byte_count: u64, read_only: bool) -> io::Result<()> {
    let address = usize::try_from(address)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "page address does not fit"))?
        as *mut c_void;
    let byte_count = usize::try_from(byte_count)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "byte count does not fit"))?;
    let protection = libc::PROT_READ | if read_only { 0 } else { libc::PROT_WRITE };
    // SAFETY: CpuWriteWatch's contract keeps every registered page live and permits exactly these
    // read-only/read-write transitions.
    if unsafe { libc::mprotect(address, byte_count, protection) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn host_page_size() -> io::Result<u64> {
    // SAFETY: sysconf has no pointer preconditions.
    let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page_size =
        u64::try_from(value).map_err(|_| io::Error::other("host page size is unavailable"))?;
    if !page_size.is_power_of_two() {
        return Err(io::Error::other("host page size is invalid"));
    }
    Ok(page_size)
}

fn host_page_size_infallible() -> u64 {
    HOST_PAGE_SIZE.load(Ordering::Acquire)
}

const fn align_down(value: u64, alignment: u64) -> u64 {
    value & !(alignment - 1)
}

const fn align_up(value: u64, alignment: u64) -> Option<u64> {
    match value.checked_add(alignment - 1) {
        Some(sum) => Some(sum & !(alignment - 1)),
        None => None,
    }
}

fn install_fault_handlers() -> io::Result<()> {
    let result = HANDLER_RESULT.get_or_init(|| {
        // SAFETY: the static previous-action slots outlive the process and the installed callback
        // has the SA_SIGINFO ABI required by sigaction.
        unsafe { install_fault_handlers_once() }
    });
    result.map_err(io::Error::from_raw_os_error)
}

unsafe fn install_fault_handlers_once() -> Result<(), i32> {
    // SAFETY: a zeroed sigaction is a valid base before its mask, callback and flags are set.
    let mut action = unsafe { std::mem::zeroed::<libc::sigaction>() };
    action.sa_sigaction = write_fault_signal_handler as *const () as usize;
    action.sa_flags = libc::SA_SIGINFO;
    // SAFETY: action.sa_mask is live writable storage.
    unsafe { libc::sigemptyset(&raw mut action.sa_mask) };
    // SAFETY: both actions are valid and PREVIOUS_SIGBUS is uninitialized writable storage.
    if unsafe {
        libc::sigaction(
            libc::SIGBUS,
            &raw const action,
            PREVIOUS_SIGBUS.as_mut_ptr(),
        )
    } != 0
    {
        return Err(last_errno());
    }
    // SAFETY: both actions are valid and PREVIOUS_SIGSEGV is uninitialized writable storage.
    if unsafe {
        libc::sigaction(
            libc::SIGSEGV,
            &raw const action,
            PREVIOUS_SIGSEGV.as_mut_ptr(),
        )
    } != 0
    {
        let error = last_errno();
        // SAFETY: restore the successfully captured SIGBUS action on partial installation.
        unsafe {
            libc::sigaction(
                libc::SIGBUS,
                PREVIOUS_SIGBUS.as_mut_ptr(),
                std::ptr::null_mut(),
            );
        }
        return Err(error);
    }
    Ok(())
}

fn last_errno() -> i32 {
    io::Error::last_os_error().raw_os_error().unwrap_or(1)
}

extern "C" fn write_fault_signal_handler(
    signal: libc::c_int,
    information: *mut libc::siginfo_t,
    context: *mut c_void,
) {
    ACTIVE_HANDLERS.fetch_add(1, Ordering::SeqCst);
    let fault_address = if information.is_null() {
        0
    } else {
        // SAFETY: SA_SIGINFO supplies a live siginfo_t for the duration of this callback.
        unsafe { (*information).si_addr().addr() as u64 }
    };
    let handled = try_handle_write_fault(fault_address);
    ACTIVE_HANDLERS.fetch_sub(1, Ordering::SeqCst);
    if handled {
        return;
    }
    // SAFETY: the previous action was captured before this handler was published.
    unsafe { chain_previous_action(signal, information, context) };
}

fn try_handle_write_fault(fault_address: u64) -> bool {
    if fault_address == 0 {
        write_signal_diagnostic(b"[REMu][WATCH] unhandled fault: null address\n");
        return false;
    }
    let snapshot_pointer = WATCH_SNAPSHOT.load(Ordering::SeqCst);
    if snapshot_pointer.is_null() {
        write_signal_diagnostic(b"[REMu][WATCH] unhandled fault: no watch snapshot\n");
        return false;
    }
    // SAFETY: ACTIVE_HANDLERS pins the immutable snapshot until this handler leaves.
    let snapshot = unsafe { &*snapshot_pointer };
    let page_size = host_page_size_infallible();
    let page_address = align_down(fault_address, page_size);
    let has_page = snapshot.contains_page(page_address);
    if !has_page && !snapshot.contains_writable_page(page_address) {
        // This handler receives every process SIGSEGV/SIGBUS while CPU write watching is active.
        // A fault outside the immutable watch snapshot is not a write-watch failure; forward it
        // unchanged so the system crash report retains the guest RIP and register state.
        write_signal_diagnostic(
            b"[REMu][WATCH] unrelated fault outside watched memory; forwarding\n",
        );
        return false;
    }
    let has_armed_owner = has_page
        && snapshot.owners(page_address).any(|owner| {
            // SAFETY: the snapshot pins all range allocations referenced by its owners.
            let range = unsafe { &*owner.range };
            range.pages[owner.page_index].armed.load(Ordering::Acquire)
        });

    let Ok(address) = usize::try_from(page_address) else {
        write_signal_diagnostic(b"[REMu][WATCH] unhandled fault: address conversion failed\n");
        return false;
    };
    let Ok(length) = usize::try_from(page_size) else {
        write_signal_diagnostic(b"[REMu][WATCH] unhandled fault: length conversion failed\n");
        return false;
    };
    // SAFETY: all page owners registered this live mapping. This is the exact protected host page.
    if unsafe {
        libc::mprotect(
            address as *mut c_void,
            length,
            libc::PROT_READ | libc::PROT_WRITE,
        )
    } != 0
    {
        write_signal_diagnostic(b"[REMu][WATCH] unhandled fault: mprotect failed\n");
        return false;
    }

    // A watched page with no armed owner is required to be read/write: a previous handled store
    // disarmed it. Guest process-image protection and overlapping resource lifetimes can race that
    // invariant under Rosetta. Because the immutable snapshot still proves this is a live unified-
    // memory resource, recover its expected write access and mark every owner dirty instead of
    // forwarding a fatal SIGBUS. Legitimate guest protection changes retire the watches before
    // reaching mprotect, so they do not enter this path.
    if !has_armed_owner {
        // Multiple Unity writers can already be trapped on the same page when the first handler
        // disarms it. The later handlers are expected recovery, not actionable diagnostics;
        // writing one line per collision from signal context caused severe stderr and audio
        // stalls in update-heavy scenes.
        RECOVERED_DISARMED_FAULTS.fetch_add(1, Ordering::Relaxed);
    }

    for owner in snapshot.owners(page_address) {
        // SAFETY: the active snapshot pins the range, and page_index was produced from that range.
        let range = unsafe { &*owner.range };
        let watched_page = &range.pages[owner.page_index];
        if watched_page.armed.swap(false, Ordering::AcqRel) {
            range.generation.fetch_add(1, Ordering::AcqRel);
        }
        watched_page.dirty.store(true, Ordering::Release);
        range.dirty.store(true, Ordering::Release);
    }
    true
}

fn write_signal_diagnostic(message: &'static [u8]) {
    // SAFETY: `write` is async-signal-safe, and the static byte slice remains live throughout the
    // call. Ignore short writes because this is a last-chance crash diagnostic.
    unsafe {
        libc::write(
            libc::STDERR_FILENO,
            message.as_ptr().cast::<c_void>(),
            message.len(),
        );
    }
}

unsafe fn chain_previous_action(
    signal: libc::c_int,
    information: *mut libc::siginfo_t,
    context: *mut c_void,
) {
    let previous = if signal == libc::SIGBUS {
        // SAFETY: installation initialized this slot before the handler became reachable.
        unsafe { PREVIOUS_SIGBUS.get() }
    } else {
        // SAFETY: installation initialized this slot before the handler became reachable.
        unsafe { PREVIOUS_SIGSEGV.get() }
    };
    let callback = previous.sa_sigaction;
    if callback == libc::SIG_DFL {
        // Restoring the default and returning causes the instruction to fault again, preserving
        // the original signal and crash context.
        // SAFETY: previous is the complete action captured by sigaction.
        unsafe { libc::sigaction(signal, previous, std::ptr::null_mut()) };
    } else if callback != libc::SIG_IGN {
        if previous.sa_flags & libc::SA_SIGINFO != 0 {
            // SAFETY: SA_SIGINFO identifies the stored callback's three-argument ABI.
            let callback: unsafe extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut c_void) =
                unsafe { std::mem::transmute(callback) };
            // SAFETY: forward the exact signal arguments supplied by the kernel.
            unsafe { callback(signal, information, context) };
        } else {
            // SAFETY: without SA_SIGINFO the stored callback has the one-argument signal ABI.
            let callback: unsafe extern "C" fn(libc::c_int) =
                unsafe { std::mem::transmute(callback) };
            // SAFETY: forward the signal number supplied by the kernel.
            unsafe { callback(signal) };
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::{CpuWriteWatch, CpuWriteWatchRetirement};

    struct TestMapping {
        pointer: *mut u8,
        length: usize,
    }

    impl TestMapping {
        fn new(page_count: usize) -> Self {
            let length = page_count * page_size();
            // SAFETY: request a private anonymous test mapping and validate the result below.
            let mapping = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    length,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            assert_ne!(mapping, libc::MAP_FAILED);
            Self {
                pointer: mapping.cast(),
                length,
            }
        }

        fn address(&self) -> u64 {
            self.pointer.addr() as u64
        }

        unsafe fn write(&self, offset: usize, value: u8) {
            assert!(offset < self.length);
            // SAFETY: the asserted offset lies in the live mapping owned by self.
            unsafe { self.pointer.add(offset).write_volatile(value) };
        }
    }

    impl Drop for TestMapping {
        fn drop(&mut self) {
            CpuWriteWatch::retire_overlapping(self.address(), self.length as u64);
            // SAFETY: self owns this still-live mapping and all watches are dropped first by each
            // test's declaration/drop order.
            unsafe { libc::munmap(self.pointer.cast(), self.length) };
        }
    }

    fn page_size() -> usize {
        usize::try_from(super::host_page_size().unwrap()).unwrap()
    }

    #[test]
    fn rejects_empty_null_and_overflowing_ranges() {
        // SAFETY: validation rejects these inputs before dereferencing or protecting them.
        assert_eq!(
            unsafe { CpuWriteWatch::new(0, 1) }.unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        // SAFETY: validation rejects these inputs before dereferencing or protecting them.
        assert_eq!(
            unsafe { CpuWriteWatch::new(1, 0) }.unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        // SAFETY: validation rejects these inputs before dereferencing or protecting them.
        assert_eq!(
            unsafe { CpuWriteWatch::new(u64::MAX, 2) }
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
    }

    #[test]
    fn catches_cpu_writes_and_rearms_the_mapping() {
        let mapping = TestMapping::new(1);
        // SAFETY: the mapping remains live until watch is dropped.
        let watch = unsafe { CpuWriteWatch::new(mapping.address(), page_size() as u64).unwrap() };
        assert_eq!(watch.generation(), 0);
        assert!(!watch.is_dirty());
        // SAFETY: the fault handler makes the watched page writable and resumes this store.
        unsafe { mapping.write(0, 0x5a) };
        assert!(watch.consume_dirty());
        assert!(!watch.consume_dirty());
        assert_eq!(
            watch.dirty_page_ranges(),
            vec![(mapping.address(), page_size() as u64)]
        );
        assert_eq!(watch.generation(), 1);
        watch.rearm().unwrap();
        assert!(watch.dirty_page_ranges().is_empty());
        // SAFETY: same live watched mapping.
        unsafe { mapping.write(1, 0xa5) };
        assert!(watch.consume_dirty());
        assert_eq!(watch.generation(), 2);
        drop(watch);
    }

    #[test]
    fn reads_do_not_dirty_a_watch() {
        let mapping = TestMapping::new(1);
        // SAFETY: the mapping remains live until watch is dropped.
        let watch = unsafe { CpuWriteWatch::new(mapping.address(), page_size() as u64).unwrap() };
        // SAFETY: the byte is readable within the live mapping.
        assert_eq!(unsafe { mapping.pointer.read_volatile() }, 0);
        assert!(!watch.is_dirty());
        assert_eq!(watch.generation(), 0);
        drop(watch);
    }

    #[test]
    fn same_page_watches_become_dirty_together() {
        let mapping = TestMapping::new(1);
        // SAFETY: both byte ranges remain within the live mapping until the watches are dropped.
        let left = unsafe { CpuWriteWatch::new(mapping.address() + 8, 64).unwrap() };
        // SAFETY: same live mapping contract as above.
        let right = unsafe { CpuWriteWatch::new(mapping.address() + 128, 64).unwrap() };
        // SAFETY: the store targets their shared protected host page.
        unsafe { mapping.write(32, 1) };
        assert!(left.is_dirty());
        assert!(right.is_dirty());
        assert_eq!(left.dirty_page_ranges(), vec![(mapping.address() + 8, 64)]);
        assert_eq!(
            right.dirty_page_ranges(),
            vec![(mapping.address() + 128, 64)]
        );
        drop(right);
        drop(left);
    }

    #[test]
    fn tracks_multi_page_writes_independently_and_clips_exact_ranges() {
        let page_size = page_size();
        let mapping = TestMapping::new(3);
        let address = mapping.address() + 17;
        let byte_count = page_size as u64 * 2;
        // SAFETY: the unaligned range and its complete pages remain in the live mapping.
        let watch = unsafe { CpuWriteWatch::new(address, byte_count).unwrap() };
        assert_eq!(watch.range(), (address, byte_count));
        // SAFETY: the store is in the first protected page.
        unsafe { mapping.write(32, 1) };
        assert_eq!(
            watch.dirty_page_ranges(),
            vec![(address, page_size as u64 - 17)]
        );
        assert_eq!(watch.generation(), 1);
        // SAFETY: this targets the second protected page, which remained independently armed.
        unsafe { mapping.write(page_size + 32, 2) };
        assert_eq!(
            watch.dirty_page_ranges(),
            vec![
                (address, page_size as u64 - 17),
                (mapping.address() + page_size as u64, page_size as u64),
            ]
        );
        assert_eq!(watch.generation(), 2);
        drop(watch);
    }

    #[test]
    fn page_faults_do_not_expand_through_transitive_overlaps() {
        let page_size = page_size();
        let mapping = TestMapping::new(2);
        // SAFETY: every range lies within the retained mapping.
        let first = unsafe { CpuWriteWatch::new(mapping.address(), page_size as u64).unwrap() };
        // SAFETY: this range deliberately spans both live pages.
        let bridge =
            unsafe { CpuWriteWatch::new(mapping.address(), (page_size * 2) as u64).unwrap() };
        // SAFETY: the second page remains live until this watch is dropped.
        let second = unsafe {
            CpuWriteWatch::new(mapping.address() + page_size as u64, page_size as u64).unwrap()
        };

        // SAFETY: write the first protected page only.
        unsafe { mapping.write(8, 1) };
        assert!(first.is_dirty());
        assert!(bridge.is_dirty());
        assert!(!second.is_dirty());
        assert_eq!(bridge.dirty_page_ranges().len(), 1);

        // SAFETY: the second page is still protected independently.
        unsafe { mapping.write(page_size + 8, 2) };
        assert!(second.is_dirty());
        assert_eq!(bridge.dirty_page_ranges().len(), 2);
        drop(second);
        drop(bridge);
        drop(first);
    }

    #[test]
    fn dropping_an_overlap_preserves_the_surviving_watch() {
        let mapping = TestMapping::new(1);
        // SAFETY: both watches share the retained live mapping.
        let survivor = unsafe { CpuWriteWatch::new(mapping.address(), 128).unwrap() };
        // SAFETY: same live mapping contract.
        let retired = unsafe { CpuWriteWatch::new(mapping.address() + 64, 128).unwrap() };
        drop(retired);
        // SAFETY: the surviving watch keeps the page protected and catches this write.
        unsafe { mapping.write(16, 1) };
        assert!(survivor.is_dirty());
        drop(survivor);
    }

    #[test]
    fn dropping_the_last_watch_restores_write_access() {
        let mapping = TestMapping::new(1);
        // SAFETY: the mapping remains live through the watch lifetime.
        let watch = unsafe { CpuWriteWatch::new(mapping.address(), 128).unwrap() };
        drop(watch);
        // SAFETY: dropping the last owner restores this page to read/write.
        unsafe { mapping.write(0, 0x7f) };
        // SAFETY: the byte remains in the live test mapping.
        assert_eq!(unsafe { mapping.pointer.read_volatile() }, 0x7f);
    }

    #[test]
    fn concurrent_writes_cannot_observe_an_unpublished_protected_page() {
        let mapping = TestMapping::new(1);
        let address = mapping.address() as usize;
        let (write_sender, write_receiver) = std::sync::mpsc::sync_channel::<u8>(0);
        let (done_sender, done_receiver) = std::sync::mpsc::sync_channel::<()>(0);
        let writer = std::thread::spawn(move || {
            while let Ok(value) = write_receiver.recv() {
                // SAFETY: the owning test retains the mapping until this thread is joined. The
                // write deliberately races watch destruction and must either fault through the
                // still-published snapshot or observe restored write permission.
                unsafe { (address as *mut u8).write_volatile(value) };
                done_sender.send(()).unwrap();
            }
        });

        for value in 0..=u8::MAX {
            // SAFETY: the mapping remains live and writable throughout every watch lifecycle.
            let watch = unsafe { CpuWriteWatch::new(mapping.address(), 128).unwrap() };
            write_sender.send(value).unwrap();
            drop(watch);
            done_receiver.recv().unwrap();
        }

        drop(write_sender);
        writer.join().unwrap();
        let state = super::state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(
            state
                .writable_ranges
                .range(..=mapping.address())
                .next_back()
                .is_some_and(|(_, &end)| mapping.address() < end)
        );
    }

    #[test]
    fn live_mapping_access_excludes_concurrent_retirement() {
        let mapping = TestMapping::new(1);
        // SAFETY: the complete page remains mapped until both scoped operations finish.
        let watch = unsafe { CpuWriteWatch::new(mapping.address(), page_size() as u64).unwrap() };
        let (entered_sender, entered_receiver) = std::sync::mpsc::channel();
        let (release_sender, release_receiver) = std::sync::mpsc::channel();
        let (retired_sender, retired_receiver) = std::sync::mpsc::channel();
        let mapping_address = mapping.address();
        let mapping_length = page_size() as u64;

        std::thread::scope(|scope| {
            let watch = &watch;
            let access = scope.spawn(move || {
                watch
                    .with_live_mapping(|| {
                        entered_sender.send(()).unwrap();
                        release_receiver.recv().unwrap();
                    })
                    .unwrap();
            });
            entered_receiver.recv().unwrap();
            let retirement = scope.spawn(move || {
                CpuWriteWatch::retire_overlapping(mapping_address, mapping_length);
                retired_sender.send(()).unwrap();
            });

            assert!(
                retired_receiver
                    .recv_timeout(std::time::Duration::from_millis(20))
                    .is_err()
            );
            release_sender.send(()).unwrap();
            retired_receiver
                .recv_timeout(std::time::Duration::from_secs(1))
                .unwrap();
            access.join().unwrap();
            retirement.join().unwrap();
        });
        assert!(watch.is_retired());
    }

    #[test]
    fn mapping_retirement_forgets_dormant_writable_pages() {
        let mapping = TestMapping::new(1);
        // SAFETY: the mapping remains live and writable through watch destruction.
        let watch = unsafe { CpuWriteWatch::new(mapping.address(), 128).unwrap() };
        drop(watch);
        {
            let state = super::state()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert!(
                state
                    .writable_ranges
                    .range(..=mapping.address())
                    .next_back()
                    .is_some_and(|(_, &end)| mapping.address() < end)
            );
        }

        CpuWriteWatch::retire_overlapping(mapping.address() + 1, 1);
        let state = super::state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(
            !state
                .writable_ranges
                .range(..=mapping.address())
                .next_back()
                .is_some_and(|(_, &end)| mapping.address() < end)
        );
    }

    #[test]
    fn mapping_changes_retire_overlapping_watches_without_late_permission_restore() {
        let mapping = TestMapping::new(2);
        // SAFETY: both watched pages remain mapped for the duration of the test.
        let first = unsafe { CpuWriteWatch::new(mapping.address(), 128).unwrap() };
        // SAFETY: the second page is likewise a live read/write mapping.
        let second =
            unsafe { CpuWriteWatch::new(mapping.address() + page_size() as u64, 128).unwrap() };

        CpuWriteWatch::retire_overlapping(mapping.address(), page_size() as u64);
        assert!(first.is_retired());
        assert_eq!(first.retirement(), Some(CpuWriteWatchRetirement::Replaced));
        assert!(!second.is_retired());
        assert_eq!(first.rearm().unwrap_err().kind(), io::ErrorKind::NotFound);

        // SAFETY: retirement restored the first page, while the active second-page watch still
        // catches its own write independently.
        unsafe {
            mapping.write(0, 0x41);
            mapping.write(page_size(), 0x42);
        }
        assert!(second.is_dirty());
        drop(first);
        drop(second);
    }

    #[test]
    fn reprotection_is_distinct_from_mapping_replacement() {
        let mapping = TestMapping::new(1);
        // SAFETY: the complete host page remains mapped throughout reprotection and rewatching.
        let first = unsafe { CpuWriteWatch::new(mapping.address(), 128).unwrap() };
        let reprotection =
            CpuWriteWatch::begin_reprotect_overlapping(mapping.address(), page_size() as u64);
        assert_eq!(
            first.retirement(),
            Some(CpuWriteWatchRetirement::Reprotecting)
        );
        // SAFETY: emulate the caller's permission-only operation before publishing completion.
        assert_eq!(
            unsafe {
                libc::mprotect(
                    mapping.pointer.cast(),
                    page_size(),
                    libc::PROT_READ | libc::PROT_WRITE,
                )
            },
            0
        );
        drop(reprotection);
        assert_eq!(
            first.retirement(),
            Some(CpuWriteWatchRetirement::Reprotected)
        );

        // SAFETY: the same live read/write mapping can be watched again without replacing it.
        let second = unsafe { CpuWriteWatch::new(mapping.address(), 128).unwrap() };
        // SAFETY: the renewed watch covers this byte.
        unsafe { mapping.write(0, 0x52) };
        assert!(second.is_dirty());
        drop(first);
        drop(second);
    }

    #[test]
    fn recovers_disarmed_page_if_external_protection_becomes_read_only() {
        let mapping = TestMapping::new(1);
        // SAFETY: the complete host page remains a live read/write mapping through the watch.
        let watch = unsafe { CpuWriteWatch::new(mapping.address(), 128).unwrap() };
        // SAFETY: the first store is handled normally and leaves the watched page disarmed/RW.
        unsafe { mapping.write(0, 0x31) };
        assert!(watch.is_dirty());
        // SAFETY: emulate a racing process-image protection update on this exact live host page.
        assert_eq!(
            unsafe { libc::mprotect(mapping.pointer.cast(), page_size(), libc::PROT_READ,) },
            0
        );
        // SAFETY: the signal handler recognizes the page's live watch, restores RW and dirties it.
        unsafe { mapping.write(1, 0x32) };
        assert!(watch.is_dirty());
        drop(watch);
    }

    #[test]
    fn repeated_watch_lifecycles_reclaim_registry_state() {
        let mapping = TestMapping::new(1);
        for value in 0..128_u8 {
            // SAFETY: each watch has a bounded lifetime within the retained mapping.
            let watch = unsafe { CpuWriteWatch::new(mapping.address(), 128).unwrap() };
            let id = watch.range.id;
            // SAFETY: the active watch catches this write.
            unsafe { mapping.write(0, value) };
            assert!(watch.is_dirty());
            drop(watch);
            let state = super::state()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            assert!(!state.ranges.contains_key(&id));
        }
    }
}
