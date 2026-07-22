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
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
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
    generation: AtomicU64,
    dirty: AtomicBool,
    pages: Box<[WatchedPage]>,
}

#[derive(Clone, Copy)]
struct PageOwner {
    range: *const WatchedRange,
    page_index: usize,
}

struct SnapshotPage {
    address: u64,
    owners: Box<[PageOwner]>,
}

struct WatchSnapshot {
    pages: Box<[SnapshotPage]>,
}

impl WatchSnapshot {
    fn from_ranges(ranges: impl Iterator<Item = Arc<WatchedRange>>) -> Self {
        let mut pages = BTreeMap::<u64, Vec<PageOwner>>::new();
        for range in ranges {
            let range_pointer = Arc::as_ptr(&range);
            for (page_index, page) in range.pages.iter().enumerate() {
                pages.entry(page.address).or_default().push(PageOwner {
                    range: range_pointer,
                    page_index,
                });
            }
        }
        Self {
            pages: pages
                .into_iter()
                .map(|(address, owners)| SnapshotPage {
                    address,
                    owners: owners.into_boxed_slice(),
                })
                .collect(),
        }
    }

    fn page(&self, address: u64) -> Option<&SnapshotPage> {
        let mut start = 0;
        let mut end = self.pages.len();
        while start < end {
            let middle = start + (end - start) / 2;
            match self.pages[middle].address.cmp(&address) {
                std::cmp::Ordering::Less => start = middle + 1,
                std::cmp::Ordering::Greater => end = middle,
                std::cmp::Ordering::Equal => return Some(&self.pages[middle]),
            }
        }
        None
    }
}

#[derive(Default)]
struct WatchState {
    next_id: u64,
    ranges: BTreeMap<u64, Arc<WatchedRange>>,
}

static WATCH_STATE: OnceLock<Mutex<WatchState>> = OnceLock::new();
static WATCH_SNAPSHOT: AtomicPtr<WatchSnapshot> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_HANDLERS: AtomicUsize = AtomicUsize::new(0);
static HOST_PAGE_SIZE: AtomicU64 = AtomicU64::new(0);
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

        install_fault_handlers()?;
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
            generation: AtomicU64::new(0),
            dirty: AtomicBool::new(false),
            pages: pages.into_boxed_slice(),
        });
        state.ranges.insert(range.id, Arc::clone(&range));
        publish_snapshot(&state);
        if let Err(error) = arm_range(&range, &state) {
            state.ranges.remove(&range.id);
            publish_snapshot(&state);
            restore_unowned_pages(&range, &state);
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
        let state = state()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        arm_range(&self.range, &state)
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

        // Publish first. Once all pre-publication handlers leave, no handler can dereference this
        // range, and the remaining snapshot completely describes who still owns each page.
        publish_snapshot(&state);
        for page in &self.range.pages {
            page.armed.store(false, Ordering::Release);
        }
        restore_unowned_pages(&self.range, &state);
    }
}

fn state() -> &'static Mutex<WatchState> {
    WATCH_STATE.get_or_init(|| Mutex::new(WatchState::default()))
}

fn publish_snapshot(state: &WatchState) {
    let snapshot = Box::new(WatchSnapshot::from_ranges(state.ranges.values().cloned()));
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

fn arm_range(range: &WatchedRange, state: &WatchState) -> io::Result<()> {
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
    let address = usize::try_from(page_address)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "page address does not fit"))?
        as *mut c_void;
    let page_size = usize::try_from(host_page_size_infallible())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "page size does not fit"))?;
    let protection = libc::PROT_READ | if read_only { 0 } else { libc::PROT_WRITE };
    // SAFETY: CpuWriteWatch's contract keeps every registered page live and permits exactly these
    // read-only/read-write transitions.
    if unsafe { libc::mprotect(address, page_size, protection) } != 0 {
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
        return false;
    }
    let snapshot_pointer = WATCH_SNAPSHOT.load(Ordering::SeqCst);
    if snapshot_pointer.is_null() {
        return false;
    }
    // SAFETY: ACTIVE_HANDLERS pins the immutable snapshot until this handler leaves.
    let snapshot = unsafe { &*snapshot_pointer };
    let page_size = host_page_size_infallible();
    let page_address = align_down(fault_address, page_size);
    let Some(page) = snapshot.page(page_address) else {
        return false;
    };
    if !page.owners.iter().any(|owner| {
        // SAFETY: the snapshot pins all range allocations referenced by its owners.
        let range = unsafe { &*owner.range };
        range.pages[owner.page_index].armed.load(Ordering::Acquire)
    }) {
        return false;
    }

    let Ok(address) = usize::try_from(page_address) else {
        return false;
    };
    let Ok(length) = usize::try_from(page_size) else {
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
        return false;
    }

    for owner in &page.owners {
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

    use super::CpuWriteWatch;

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
