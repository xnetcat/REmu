// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::io;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{CpuWriteWatch, CpuWriteWatchRetirement, notify_guest_memory_written};

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
fn explicit_host_write_notification_tracks_pages_and_generations_exactly() {
    let page_size = page_size();
    let mapping = TestMapping::new(3);
    // SAFETY: the retained mapping contains all three watched pages.
    let watch = unsafe { CpuWriteWatch::new(mapping.address(), (page_size * 3) as u64).unwrap() };

    assert!(notify_guest_memory_written(
        mapping.address() + page_size as u64 + 8,
        16
    ));
    assert_eq!(
        watch.dirty_page_ranges(),
        vec![(mapping.address() + page_size as u64, page_size as u64)]
    );
    assert_eq!(watch.generation(), 1);
    assert!(watch.consume_dirty());

    // Repeated HLE publication before rearm reasserts range dirtiness without counting the same
    // dirty page twice.
    assert!(notify_guest_memory_written(
        mapping.address() + page_size as u64 + 16,
        16
    ));
    assert!(watch.is_dirty());
    assert_eq!(watch.generation(), 1);
    watch.rearm().unwrap();

    assert!(notify_guest_memory_written(
        mapping.address() + page_size as u64 + 24,
        16
    ));
    assert_eq!(watch.generation(), 2);
    watch.rearm().unwrap();

    // A direct fault already published this page. The following explicit HLE notification must
    // not double-count it.
    // SAFETY: the write targets the live watched mapping.
    unsafe { mapping.write(page_size + 32, 0x5a) };
    assert_eq!(watch.generation(), 3);
    assert!(notify_guest_memory_written(
        mapping.address() + page_size as u64 + 32,
        1
    ));
    assert_eq!(watch.generation(), 3);
    assert!(!notify_guest_memory_written(u64::MAX, 2));
    assert!(!notify_guest_memory_written(mapping.address(), 0));
    drop(watch);
}

#[test]
fn explicit_host_write_notification_reaches_every_physical_alias() {
    use crate::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };

    let mut memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
    let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).expect("guest physical backing");
    let base = memory.base();
    let first = memory
        .map_shared(
            GuestMapPlacement::Fixed(base),
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("first alias");
    let second_address = base
        .checked_add(4 * GUEST_PAGE_SIZE)
        .expect("second alias address");
    let second = memory
        .map_shared(
            GuestMapPlacement::Fixed(second_address),
            2 * GUEST_PAGE_SIZE,
            GUEST_PAGE_SIZE,
            GuestProtection::READ_WRITE,
            &backing,
            0,
        )
        .expect("second alias");
    // SAFETY: both aliases remain mapped read/write until the watch is dropped.
    let watched_second =
        unsafe { CpuWriteWatch::new(second.get(), 2 * GUEST_PAGE_SIZE).expect("alias watch") };

    assert!(notify_guest_memory_written(
        first.get() + GUEST_PAGE_SIZE + 32,
        64
    ));
    // Dirty tracking is host-page granular: 16 KiB natively, 4 KiB under Rosetta. The 64-byte
    // write lands in the first host page after the guest-page boundary on either size.
    assert_eq!(
        watched_second.dirty_page_ranges(),
        vec![(second.get() + GUEST_PAGE_SIZE, page_size() as u64)]
    );
    assert_eq!(watched_second.generation(), 1);
    drop(watched_second);
}

#[test]
fn a_chronically_dirty_watch_demotes_to_acquire_time_revalidation() {
    let mapping = TestMapping::new(1);
    // SAFETY: the mapping remains live and writable until watch is dropped.
    let watch = unsafe { CpuWriteWatch::new(mapping.address(), page_size() as u64).unwrap() };
    assert!(!watch.is_demoted());

    // Rewrite then rearm every cycle: the false-sharing signature of a per-frame resident.
    for value in 0..super::DEMOTION_THRESHOLD {
        // SAFETY: the store targets the live watched page and faults while armed.
        unsafe { mapping.write(0, value as u8) };
        assert!(watch.consume_dirty());
        watch.rearm().unwrap();
    }
    assert!(
        watch.is_demoted(),
        "a resident rewritten every cycle demotes after the dirty-cycle threshold"
    );

    // Demoted: arming reports the range dirty unconditionally so the consumer revalidates at
    // acquire time, and the page is no longer protected so stores complete without a fault.
    assert!(watch.is_dirty());
    let _ = watch.consume_dirty();
    // SAFETY: the demoted page is writable; this store must not fault nor be lost.
    unsafe { mapping.write(0, 0x5a) };
    watch.rearm().unwrap();
    assert!(
        watch.is_dirty(),
        "a demoted watch re-asserts dirtiness on every rearm"
    );
    drop(watch);
}

#[test]
fn demoting_a_large_dirty_watch_restores_permissions_in_one_batch() {
    let mapping = TestMapping::new(8);
    // SAFETY: all eight pages remain mapped until the watch is dropped.
    let watch = unsafe { CpuWriteWatch::new(mapping.address(), (page_size() * 8) as u64).unwrap() };
    // SAFETY: the final page remains live and deliberately overlaps the large watch. Keeping the
    // overlap at the end also leaves the seven unowned pages as one contiguous restore run.
    let survivor = unsafe {
        CpuWriteWatch::new(
            mapping.address() + (page_size() * 7) as u64,
            page_size() as u64,
        )
        .unwrap()
    };

    for value in 0..(super::DEMOTION_THRESHOLD - 1) {
        // SAFETY: the watched mapping is live and the handler restores this page's write access.
        unsafe { mapping.write(0, value as u8) };
        let _ = watch.consume_dirty();
        watch.rearm().unwrap();
    }
    // The final dirty cycle demotes while the untouched tail pages are still armed.
    unsafe { mapping.write(0, 0xff) };
    let _ = watch.consume_dirty();
    super::page::reset_protect_range_call_count();
    watch.rearm().unwrap();

    assert!(watch.is_demoted());
    assert!(
        super::page::protect_range_call_count() <= 1,
        "demotion should restore one contiguous range instead of one syscall per armed page"
    );
    // SAFETY: the overlapping watch must keep the last page protected after the large watch
    // demotes; this write faults through that survivor and marks it dirty.
    unsafe { mapping.write(page_size() * 7, 0x5a) };
    assert!(survivor.is_dirty());
    drop(survivor);
    drop(watch);
}

#[test]
fn an_occasionally_dirty_watch_never_demotes() {
    let mapping = TestMapping::new(1);
    // SAFETY: the mapping remains live and writable until watch is dropped.
    let watch = unsafe { CpuWriteWatch::new(mapping.address(), page_size() as u64).unwrap() };
    // A clean rearm cycle resets the streak, so interleaving clean cycles prevents demotion even
    // over many more iterations than the threshold.
    for round in 0..(super::DEMOTION_THRESHOLD * 4) {
        if round % 2 == 0 {
            // SAFETY: the store targets the live watched page and faults while armed.
            unsafe { mapping.write(0, round as u8) };
            assert!(watch.consume_dirty());
        }
        watch.rearm().unwrap();
    }
    assert!(
        !watch.is_demoted(),
        "a watch with interleaved clean cycles keeps precise protection tracking"
    );
    drop(watch);
}

static READ_WATCH_MATERIALIZE_CALLS: AtomicU64 = AtomicU64::new(0);

/// A test read-watch materializer: fills the (handler-made-writable) page with a known pattern.
///
/// # Safety
///
/// Invoked from the fault handler with `[page_address, page_address + page_size)` writable.
unsafe extern "C" fn fill_page_with_pattern(page_address: u64, page_size: u64) -> bool {
    READ_WATCH_MATERIALIZE_CALLS.fetch_add(1, Ordering::SeqCst);
    // SAFETY: the handler made the whole page writable before invoking the materializer.
    unsafe {
        std::ptr::write_bytes(page_address as *mut u8, 0xcd, page_size as usize);
    }
    true
}

// The process-global materializer makes these read-watch tests order-sensitive; the crate's
// verification gate runs `--test-threads=1`, under which the shared counter is deterministic.
#[test]
fn a_read_watch_materializes_gpu_bytes_on_the_first_guest_read() {
    super::set_read_watch_materializer(fill_page_with_pattern);
    let mapping = TestMapping::new(1);
    READ_WATCH_MATERIALIZE_CALLS.store(0, Ordering::SeqCst);
    // SAFETY: the mapping stays live and its page tolerates a no-access window until watch drops.
    let watch = unsafe {
        CpuWriteWatch::new_read_materialized(mapping.address(), page_size() as u64).unwrap()
    };
    assert!(watch.is_read_watch());
    assert!(!watch.all_pages_materialized());
    assert!(!watch.is_dirty());

    // A guest READ faults on the no-access page and materializes it.
    // SAFETY: the byte is inside the live read-watched page.
    let observed = unsafe { mapping.pointer.read_volatile() };
    assert_eq!(
        observed, 0xcd,
        "the first read observes materialized producer bytes"
    );
    assert_eq!(READ_WATCH_MATERIALIZE_CALLS.load(Ordering::SeqCst), 1);
    assert!(watch.all_pages_materialized());
    assert!(
        !watch.is_dirty(),
        "a materializing read does not dirty the watch"
    );

    // A second read does not re-materialize: the page is now readable.
    // SAFETY: same live page, now readable.
    assert_eq!(unsafe { mapping.pointer.read_volatile() }, 0xcd);
    assert_eq!(
        READ_WATCH_MATERIALIZE_CALLS.load(Ordering::SeqCst),
        1,
        "materialization happens once per armed window"
    );
    drop(watch);
}

#[test]
fn a_guest_write_to_a_read_watch_skips_materialization_and_dirties() {
    super::set_read_watch_materializer(fill_page_with_pattern);
    let mapping = TestMapping::new(1);
    let calls_before = READ_WATCH_MATERIALIZE_CALLS.load(Ordering::SeqCst);
    // SAFETY: live mapping; no-access window until watch drops.
    let watch = unsafe {
        CpuWriteWatch::new_read_materialized(mapping.address(), page_size() as u64).unwrap()
    };
    // A guest WRITE overwrites the stale bytes: it dirties the watch and skips materialization.
    // SAFETY: the store targets the live read-watched page.
    unsafe { mapping.write(0, 0x42) };
    assert!(watch.is_dirty(), "a write dirties a read-watch");
    assert_eq!(
        READ_WATCH_MATERIALIZE_CALLS.load(Ordering::SeqCst),
        calls_before,
        "a guest write skips materialization (it overwrites the stale bytes)"
    );
    assert!(watch.all_pages_materialized());
    // SAFETY: the byte the guest wrote must stand, not be overwritten by the materializer.
    assert_eq!(unsafe { mapping.pointer.read_volatile() }, 0x42);
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
    let bridge = unsafe { CpuWriteWatch::new(mapping.address(), (page_size * 2) as u64).unwrap() };
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
    let address = usize::try_from(mapping.address()).unwrap();
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
        state
            .writable_ranges
            .range(..=mapping.address())
            .next_back()
            .is_none_or(|(_, &end)| mapping.address() >= end)
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

#[test]
fn a_guest_mapping_change_keeps_recovering_delayed_protection_faults() {
    let mapping = TestMapping::new(1);
    // SAFETY: the retained mapping outlives this watch.
    let watch = unsafe { CpuWriteWatch::new(mapping.address(), watch_bytes()).unwrap() };
    // A permission-only guest change retires the watch and restores write access. A store that
    // already faulted on another CPU can still be delivered afterwards, so the interval must stay
    // recoverable instead of becoming an unowned fault that terminates the process.
    let reprotection = CpuWriteWatch::begin_reprotect_overlapping(mapping.address(), watch_bytes());
    assert_eq!(
        watch.retirement(),
        Some(CpuWriteWatchRetirement::Reprotecting)
    );
    assert!(
        writable_range_covers(mapping.address()),
        "a reprotected mapping must remain recoverable by the fault handler"
    );
    drop(reprotection);
    drop(watch);

    // A replacement is different: the pages are about to be remapped, so silently granting write
    // access would hide a guest access to memory the mapping change took away.
    // SAFETY: the retained mapping outlives this watch.
    let replaced = unsafe { CpuWriteWatch::new(mapping.address(), watch_bytes()).unwrap() };
    CpuWriteWatch::retire_overlapping(mapping.address(), watch_bytes());
    assert_eq!(
        replaced.retirement(),
        Some(CpuWriteWatchRetirement::Replaced)
    );
    assert!(
        !writable_range_covers(mapping.address()),
        "a replaced mapping must not be recoverable by the fault handler"
    );
    drop(replaced);
}

fn writable_range_covers(address: u64) -> bool {
    let state = super::state()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state
        .writable_ranges
        .range(..=address)
        .next_back()
        .is_some_and(|(_, &end)| address < end)
}

fn watch_bytes() -> u64 {
    u64::try_from(page_size()).expect("a host page size fits u64")
}
