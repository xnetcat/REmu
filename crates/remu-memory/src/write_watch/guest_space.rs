// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! A signal-safe registry of the live guest address-space reservations.
//!
//! The fault handler must classify an address it cannot service before forwarding it, and it may
//! not take a lock to do so. Reservations are created once per address space and released at
//! teardown, so a small fixed table of atomics is enough to answer whether a faulting address is
//! guest memory or ordinary host memory.

use std::sync::atomic::{AtomicU64, Ordering};

/// One slot per plausible concurrent address space. Overflowing the table only costs the `space=`
/// classification for the extra reservation, never correctness.
const SLOT_COUNT: usize = 8;

#[expect(
    clippy::declare_interior_mutable_const,
    reason = "the const is only used to initialize the array elements themselves"
)]
const EMPTY_SLOT: AtomicU64 = AtomicU64::new(0);

static RESERVATION_STARTS: [AtomicU64; SLOT_COUNT] = [EMPTY_SLOT; SLOT_COUNT];

static RESERVATION_ENDS: [AtomicU64; SLOT_COUNT] = [EMPTY_SLOT; SLOT_COUNT];

/// Publishes a guest reservation so the fault handler can label addresses inside it.
pub(crate) fn publish_guest_reservation(address: u64, size: u64) {
    let Some(end) = address.checked_add(size) else {
        return;
    };
    if address == 0 || size == 0 {
        return;
    }
    for index in 0..SLOT_COUNT {
        // A zero end marks a free slot. Claiming it before the start is stored would let a
        // concurrent handler read a stale start, so publish the start first.
        if RESERVATION_ENDS[index].load(Ordering::Acquire) != 0 {
            continue;
        }
        RESERVATION_STARTS[index].store(address, Ordering::Release);
        if RESERVATION_ENDS[index]
            .compare_exchange(0, end, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return;
        }
    }
}

/// Retires a reservation published by [`publish_guest_reservation`].
pub(crate) fn forget_guest_reservation(address: u64, size: u64) {
    let Some(end) = address.checked_add(size) else {
        return;
    };
    for index in 0..SLOT_COUNT {
        if RESERVATION_STARTS[index].load(Ordering::Acquire) == address
            && RESERVATION_ENDS[index]
                .compare_exchange(end, 0, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        {
            return;
        }
    }
}

/// Returns whether `address` belongs to a published guest reservation.
pub fn guest_reservation_contains(address: u64) -> bool {
    contains_guest_address(address)
}

/// Returns whether `address` belongs to a published guest reservation.
pub(super) fn contains_guest_address(address: u64) -> bool {
    (0..SLOT_COUNT).any(|index| {
        let end = RESERVATION_ENDS[index].load(Ordering::Acquire);
        let start = RESERVATION_STARTS[index].load(Ordering::Acquire);
        end != 0 && start != 0 && start <= address && address < end
    })
}

/// Pages the watch handler may `mprotect` inside the 8 TiB pin: 0x7X heap banks and
/// 8 TiB-class PoolInfo. GPU 0x80 maps are not stored here — there are too many of them
/// and their `MAP_FIXED` islands already `mprotect` cheaply.
const COMMITTED_SLOT_COUNT: usize = 1 << 18;
const COMMITTED_PROBE: usize = 32;
const COMMITTED_PAGE: u64 = 16 * 1024;

#[expect(
    clippy::declare_interior_mutable_const,
    reason = "the const is only used to initialize the array elements themselves"
)]
const EMPTY_PAGE: AtomicU64 = AtomicU64::new(0);

static COMMITTED_PAGES: [AtomicU64; COMMITTED_SLOT_COUNT] = [EMPTY_PAGE; COMMITTED_SLOT_COUNT];

fn tracks_committed_pin_page(address: u64) -> bool {
    let bank = address >> 32;
    (0x70..=0x7f).contains(&bank) || (address != 0 && (address >> 40) == 0x8)
}

fn committed_slot(page: u64) -> usize {
    let mixed = (page >> 14).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (mixed as usize) & (COMMITTED_SLOT_COUNT - 1)
}

fn insert_committed_page(page: u64) {
    if page == 0 || !tracks_committed_pin_page(page) {
        return;
    }
    let start = committed_slot(page);
    for probe in 0..COMMITTED_PROBE {
        let slot = (start + probe) & (COMMITTED_SLOT_COUNT - 1);
        let current = COMMITTED_PAGES[slot].load(Ordering::Acquire);
        if current == page {
            return;
        }
        if current == 0
            && COMMITTED_PAGES[slot]
                .compare_exchange(0, page, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        {
            return;
        }
    }
}

fn remove_committed_page(page: u64) {
    if page == 0 {
        return;
    }
    let start = committed_slot(page);
    for probe in 0..COMMITTED_PROBE {
        let slot = (start + probe) & (COMMITTED_SLOT_COUNT - 1);
        if COMMITTED_PAGES[slot].load(Ordering::Acquire) == page {
            COMMITTED_PAGES[slot].store(0, Ordering::Release);
            return;
        }
    }
}

/// Records `MAP_FIXED` islands the watch handler may `mprotect` without touching the 8 TiB pin.
pub fn publish_guest_committed_range(address: u64, size: u64) {
    let Some(end) = address.checked_add(size) else {
        return;
    };
    let start = address & !(COMMITTED_PAGE - 1);
    let mut page = start;
    while page < end {
        insert_committed_page(page);
        let Some(next) = page.checked_add(COMMITTED_PAGE) else {
            break;
        };
        page = next;
    }
}

/// Drops committed-page marks for a released or PROT_NONE-restored range.
pub(crate) fn forget_guest_committed_range(address: u64, size: u64) {
    let Some(end) = address.checked_add(size) else {
        return;
    };
    // An 8 TiB reservation drop cannot walk page-by-page. Clear matching slots instead.
    if size >= (1 << 30) {
        for slot in 0..COMMITTED_SLOT_COUNT {
            let page = COMMITTED_PAGES[slot].load(Ordering::Acquire);
            if page >= address && page < end {
                COMMITTED_PAGES[slot].store(0, Ordering::Release);
            }
        }
        return;
    }
    let start = address & !(COMMITTED_PAGE - 1);
    let mut page = start;
    while page < end {
        remove_committed_page(page);
        let Some(next) = page.checked_add(COMMITTED_PAGE) else {
            break;
        };
        page = next;
    }
}

/// True when `address` is a `MAP_FIXED` 0x7X / 8 TiB-class island, not a pin hole.
pub fn guest_page_is_committed(address: u64) -> bool {
    let page = address & !(COMMITTED_PAGE - 1);
    if page == 0 || !tracks_committed_pin_page(page) {
        return false;
    }
    let start = committed_slot(page);
    (0..COMMITTED_PROBE).any(|probe| {
        let slot = (start + probe) & (COMMITTED_SLOT_COUNT - 1);
        COMMITTED_PAGES[slot].load(Ordering::Acquire) == page
    })
}

#[cfg(test)]
mod tests {
    use super::{contains_guest_address, forget_guest_reservation, publish_guest_reservation};

    #[test]
    fn classifies_addresses_only_while_their_reservation_is_published() {
        // The table is process-wide, so use an address range no other test publishes.
        let address = 0x7000_0000_0000;
        let size = 0x1_0000;
        assert!(!contains_guest_address(address));
        publish_guest_reservation(address, size);
        assert!(contains_guest_address(address));
        assert!(contains_guest_address(address + size - 1));
        assert!(!contains_guest_address(address + size));
        assert!(!contains_guest_address(address - 1));
        forget_guest_reservation(address, size);
        assert!(!contains_guest_address(address));
    }

    #[test]
    fn a_real_address_space_publishes_and_retires_its_own_bounds() {
        // Without this wiring every fault would be labelled `host`, and the report could no longer
        // tell a wild guest pointer apart from an emulator access to retired guest memory.
        use crate::{GUEST_PAGE_SIZE, GuestAddressSpace};

        let memory = GuestAddressSpace::new(8 * GUEST_PAGE_SIZE).expect("guest address space");
        let base = memory.base().get();
        let last = memory.end().get() - 1;
        assert!(contains_guest_address(base));
        assert!(contains_guest_address(last));
        drop(memory);
        assert!(!contains_guest_address(base));
    }

    #[test]
    fn rejects_degenerate_and_overflowing_reservations() {
        publish_guest_reservation(0, 0x1000);
        publish_guest_reservation(0x2000, 0);
        publish_guest_reservation(u64::MAX, 2);
        assert!(!contains_guest_address(0));
        assert!(!contains_guest_address(0x2000));
        assert!(!contains_guest_address(u64::MAX));
    }

    #[test]
    fn tracks_only_heap_bank_and_eight_tib_class_islands() {
        use super::{
            forget_guest_committed_range, guest_page_is_committed, publish_guest_committed_range,
        };

        let heap = 0x0075_00e0_0000_u64;
        let gpu = 0x0000_0080_4449_0000_u64;
        publish_guest_committed_range(heap, 0x1_0000);
        publish_guest_committed_range(gpu, 0x1_0000);
        assert!(guest_page_is_committed(heap));
        assert!(guest_page_is_committed(heap + 0x4000));
        assert!(!guest_page_is_committed(gpu), "0x80 GPU pages are not tracked");
        assert!(!guest_page_is_committed(0x0076_a0f4_c000));
        forget_guest_committed_range(heap, 0x1_0000);
        assert!(!guest_page_is_committed(heap));
    }
}
