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
pub(super) fn contains_guest_address(address: u64) -> bool {
    (0..SLOT_COUNT).any(|index| {
        let end = RESERVATION_ENDS[index].load(Ordering::Acquire);
        let start = RESERVATION_STARTS[index].load(Ordering::Acquire);
        end != 0 && start != 0 && start <= address && address < end
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
}
