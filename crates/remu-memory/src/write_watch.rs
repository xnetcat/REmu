// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Signal-safe dirty-page tracking for CPU writes to guest unified memory.
//!
//! Metal resources cannot alias arbitrary guest mappings. A backend can use this module to keep
//! a resident Metal resource and refresh only the host pages that the guest CPU has changed.

use std::io;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

mod diagnostic;

mod guest_space;
pub(crate) use guest_space::{forget_guest_reservation, publish_guest_reservation};

mod page;
use page::{HOST_PAGE_SIZE, align_down, align_up, host_page_size, host_page_size_infallible};

mod signal;
pub use signal::{GuestFaultDeliveryHandler, GuestFsFaultHandler, GuestReadMaterializer};
pub use signal::{install_fault_handlers, install_thread_signal_stack};

/// Registers the process-global read-watch materializer used by [`CpuWriteWatch::new_read_materialized`].
///
/// Returns the previously registered materializer, if any. This is the D5 integration seam: the
/// gpu-metal consumer registers a function that copies a resident texture's bytes into a faulting
/// guest page. The materializer runs in signal context and must be async-signal-safe (see
/// [`GuestReadMaterializer`]).
pub fn set_read_watch_materializer(
    materializer: GuestReadMaterializer,
) -> Option<GuestReadMaterializer> {
    signal::set_read_watch_materializer(materializer)
}

/// Registers the process-global FS/`tls` fault backstop used when AOT `fs:` rewriting misses a site.
///
/// Returns the previously registered handler, if any. The handler runs in signal context and must
/// be async-signal-safe (see [`GuestFsFaultHandler`]).
pub fn set_fs_fault_handler(handler: GuestFsFaultHandler) -> Option<GuestFsFaultHandler> {
    signal::set_fs_fault_handler(handler)
}

/// Registers the guest exception-handler delivery hook for unrecovered SIGSEGV/SIGBUS.
pub fn set_guest_fault_delivery_handler(
    handler: GuestFaultDeliveryHandler,
) -> Option<GuestFaultDeliveryHandler> {
    signal::set_guest_fault_delivery_handler(handler)
}

mod snapshot;
use snapshot::publish_snapshot;

mod state;
use state::{
    DEMOTION_THRESHOLD, WATCH_ACTIVE, WATCH_REPLACED, WATCH_REPROTECTED, WATCH_REPROTECTING,
    WatchedPage, WatchedRange, arm_range, remember_writable_range, restore_unowned_pages,
    retire_overlapping, state,
};

/// Publishes bytes written directly by an HLE service into guest memory.
///
/// Host-side copies can bypass the faulting VA watched by a GPU resource when the same physical
/// allocation has multiple guest aliases. Mark every live watch intersecting the original range or
/// any coherent physical alias. Pages remain armed: a later guest store must still fault normally.
#[must_use]
pub fn notify_guest_memory_written(address: u64, byte_count: u64) -> bool {
    notify_guest_memory_written_excluding(address, byte_count, &[])
}

/// Publishes host-written bytes to every live watch except the writer's own.
///
/// A writer that immediately re-baselines its own fingerprints must not see its own store as
/// a foreign modification: dirtying its pages and bumping its generation would ripple through
/// every consumer keyed on that state. Every *other* watch intersecting the write or one of
/// its physical aliases still needs the notification, exactly as for an ordinary HLE write.
#[must_use]
pub fn notify_guest_memory_written_excluding(
    address: u64,
    byte_count: u64,
    excluded: &[&CpuWriteWatch],
) -> bool {
    let aliases = crate::address_space::guest_write_alias_segments(address, byte_count);
    if aliases.is_empty() {
        return false;
    }
    let mut notified = false;
    let state = state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for range in state.ranges.values() {
        if excluded
            .iter()
            .any(|watch| Arc::ptr_eq(&watch.range, range))
        {
            continue;
        }
        let range_end = range.address.saturating_add(range.byte_count);
        let mut range_notified = false;
        for page in &range.pages {
            let page_end = page.address.saturating_add(host_page_size_infallible());
            let watched_start = page.address.max(range.address);
            let watched_end = page_end.min(range_end);
            if watched_start >= watched_end
                || !aliases.iter().any(|segment| {
                    let segment_end = segment.address.saturating_add(segment.byte_count);
                    segment.address < watched_end && watched_start < segment_end
                })
            {
                continue;
            }
            if !page.dirty.swap(true, Ordering::AcqRel) {
                range.generation.fetch_add(1, Ordering::AcqRel);
            }
            range_notified = true;
        }
        if range_notified {
            range.dirty.store(true, Ordering::Release);
            notified = true;
        }
    }
    notified
}

/// Reads `REMU_DISABLE_CPU_WRITE_WATCH` once for the process.
///
/// One watch is constructed per shared-registry record per snapshot, so reading the environment
/// per construction is a linear `environ` scan on the hottest memory path under Rosetta, and lets
/// the flag be observed inconsistently across watches within a single run.
fn precise_tracking_requested() -> bool {
    static PRECISE_TRACKING: OnceLock<bool> = OnceLock::new();
    *PRECISE_TRACKING.get_or_init(|| {
        std::env::var_os("REMU_DISABLE_CPU_WRITE_WATCH").is_none_or(|value| value != "1")
    })
}

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
        // SAFETY: forwarded to the caller's contract on `new`.
        unsafe { Self::new_inner(address, byte_count, false) }
    }

    /// Starts a read-watch that materializes GPU-written bytes on the first guest read (audit D5).
    ///
    /// Pages arm with no access, so a guest READ — not only a write — faults into the handler,
    /// which invokes the process read-watch materializer to fill the page from the producer's
    /// bytes, then leaves it readable. A guest WRITE to an unmaterialized page skips materialization
    /// (the guest is overwriting the stale bytes) and dirties the watch like an ordinary write.
    ///
    /// Fails with [`io::ErrorKind::Unsupported`] if no materializer has been registered through
    /// [`set_read_watch_materializer`]: without one the handler cannot supply the bytes, and arming
    /// anyway would let the first read observe stale memory. This is a loud refusal, not a silent
    /// fallback.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::new`], and additionally the pages must tolerate a no-access
    /// protection window until their first fault.
    pub unsafe fn new_read_materialized(address: u64, byte_count: u64) -> io::Result<Self> {
        if signal::read_watch_materializer().is_none() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "a read-watch requires a registered read-watch materializer",
            ));
        }
        // SAFETY: forwarded to the caller's contract on `new_read_materialized`.
        unsafe { Self::new_inner(address, byte_count, true) }
    }

    /// # Safety
    ///
    /// Same contract as [`Self::new`]: `address..address + byte_count` must belong to a live
    /// readable/writable mapping whose complete host pages tolerate the protection transitions this
    /// watch performs (read-only/read-write, and no-access for a read-watch).
    unsafe fn new_inner(address: u64, byte_count: u64, read_watch: bool) -> io::Result<Self> {
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
        // A read-watch must fault on reads, so it is always precise regardless of the disable flag
        // (which only assumes CPU texture *writes* can be skipped, not reads).
        let precise_tracking = read_watch || precise_tracking_requested();
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
                materialized: AtomicBool::new(!read_watch),
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
            read_watch,
            generation: AtomicU64::new(0),
            dirty: AtomicBool::new(false),
            retirement: AtomicU8::new(WATCH_ACTIVE),
            armed_generation: AtomicU64::new(0),
            dirty_rearm_streak: AtomicU64::new(0),
            demoted: AtomicBool::new(false),
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

    /// Returns whether this watch has demoted itself to acquire-time revalidation.
    ///
    /// A demoted watch no longer arms its pages, so it takes no protection faults; it instead
    /// reports [`Self::is_dirty`] on every armed cycle, and its consumer must revalidate the bytes
    /// each time it acquires them. Demotion is one-way for the life of the watch: a resource hot
    /// enough to trip it stays hot, and re-arming would only reopen the false-sharing fault storm.
    #[must_use]
    pub fn is_demoted(&self) -> bool {
        self.range.demoted.load(Ordering::Acquire)
    }

    /// Returns whether this is a read-watch (materializes GPU bytes on the first guest read).
    #[must_use]
    pub fn is_read_watch(&self) -> bool {
        self.range.read_watch
    }

    /// Returns whether every page of this watch has been materialized.
    ///
    /// A write-watch reports `true` unconditionally; a read-watch reports `true` only once each of
    /// its pages has been materialized by a guest read or overwritten by a guest write.
    #[must_use]
    pub fn all_pages_materialized(&self) -> bool {
        self.range
            .pages
            .iter()
            .all(|page| page.materialized.load(Ordering::Acquire))
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
        self.account_dirty_cycle_before_arming();
        arm_range(&self.range, &state)
    }

    /// Detects chronic dirtiness so [`arm_range`] can demote the range on this rearm.
    ///
    /// A store during the previous armed window advanced `generation` past the value recorded at
    /// arm time, independently of whether the caller has consumed the dirty flag. A run of such
    /// cycles is exactly the false-sharing signature the demotion targets; a clean cycle resets it.
    fn account_dirty_cycle_before_arming(&self) {
        if !self.range.precise_tracking
            || self.range.read_watch
            || self.range.demoted.load(Ordering::Acquire)
        {
            return;
        }
        let armed_generation = self.range.armed_generation.load(Ordering::Acquire);
        if self.range.generation.load(Ordering::Acquire) == armed_generation {
            self.range.dirty_rearm_streak.store(0, Ordering::Release);
            return;
        }
        let streak = self.range.dirty_rearm_streak.fetch_add(1, Ordering::AcqRel) + 1;
        if streak >= DEMOTION_THRESHOLD {
            self.range.demoted.store(true, Ordering::Release);
        }
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

    /// Retires watches while a new VA is added for bytes in the same physical backing.
    ///
    /// Stores through the new alias can occur before a resource owner has rebuilt its complete
    /// alias watch. Marking the old watch dirty forces full content validation after renewal,
    /// preventing an older GPU-owned image from overwriting CPU initialization through that VA.
    pub fn begin_alias_change_overlapping(
        address: u64,
        byte_count: u64,
    ) -> CpuWriteWatchReprotection {
        let ranges = retire_overlapping(address, byte_count, WATCH_REPROTECTING);
        for range in &ranges {
            range.dirty.store(true, Ordering::Release);
        }
        CpuWriteWatchReprotection { ranges }
    }

    /// Returns the exact requested guest range rather than its page-aligned protection span.
    #[must_use]
    pub fn range(&self) -> (u64, u64) {
        (self.range.address, self.range.byte_count)
    }
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

#[cfg(test)]
mod tests;
