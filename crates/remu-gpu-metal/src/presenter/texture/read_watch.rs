// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! D5 gpu-metal wiring for `remu_memory::set_read_watch_materializer`.
//!
//! The memory crate's fault handler already carries the read/write bit and invokes a process
//! materializer (async-signal-safe contract). This module owns the gpu-metal side:
//!
//! 1. A lock-free pre-staged page table the materializer may read without taking locks.
//! 2. Registration of the materializer at presenter bring-up.
//! 3. A **loud refusal** when a faulting page has no pre-staged host bytes — Metal readback in
//!    signal context is not async-signal-safe, so we never call Metal from the handler.
//!
//! End-state (NOT armed blind here): when a resident becomes GPU-authored, copy its pixels into a
//! host-visible staging buffer, `prestage_page` each covered host page, and arm
//! `CpuWriteWatch::new_read_materialized` instead of CPU writeback. That swap is the owner-gate
//! seam — shipping it without the interactive/soak gate would turn missing prestages into guest
//! SIGSEGV storms or silent wrong-bytes reads. See NOTES.md Batch B4.

use remu_memory::set_read_watch_materializer;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};

/// Maximum simultaneously pre-staged host pages. Bounded so the signal handler never walks an
/// unbounded table; overflow refuses new prestages loudly rather than reallocating under load.
const PRESTAGE_SLOT_COUNT: usize = 256;

#[repr(C)]
struct PrestageSlot {
    /// Guest/host page address, or 0 when free.
    page_address: AtomicU64,
    /// Pointer to `page_size` producer bytes (host-visible, immutable while published).
    bytes: AtomicPtr<u8>,
    /// Length of `bytes`; must equal the fault page size when published.
    byte_count: AtomicU64,
}

impl PrestageSlot {
    const fn empty() -> Self {
        Self {
            page_address: AtomicU64::new(0),
            bytes: AtomicPtr::new(std::ptr::null_mut()),
            byte_count: AtomicU64::new(0),
        }
    }
}

static PRESTAGE_SLOTS: [PrestageSlot; PRESTAGE_SLOT_COUNT] =
    [const { PrestageSlot::empty() }; PRESTAGE_SLOT_COUNT];
static MATERIALIZER_INSTALLED: AtomicBool = AtomicBool::new(false);
static INSTALL_ONCE: Once = Once::new();
static PRESTAGE_REFUSALS: AtomicUsize = AtomicUsize::new(0);
static MATERIALIZE_MISS: AtomicUsize = AtomicUsize::new(0);

/// Registers the process-global read-watch materializer exactly once.
///
/// Safe to call from presenter init on every launch; subsequent calls are no-ops.
pub(crate) fn ensure_read_watch_materializer_registered() {
    INSTALL_ONCE.call_once(|| {
        set_read_watch_materializer(materialize_prestaged_guest_page);
        MATERIALIZER_INSTALLED.store(true, Ordering::Release);
    });
}

#[must_use]
#[allow(dead_code)] // D5 seam — tests assert installation; production probes may follow.
pub(crate) fn read_watch_materializer_installed() -> bool {
    MATERIALIZER_INSTALLED.load(Ordering::Acquire)
}

/// Publishes host-visible producer bytes for one guest page so a later fault can memcpy them.
///
/// `bytes` must remain valid and immutable until [`clear_prestaged_page`] (or process exit).
/// This is the arm+prestage half of D5; converting residents to `new_read_materialized` is the
/// gated follow-up (NOTES seam).
///
/// # Safety
///
/// `bytes` must point at `byte_count` readable bytes for the lifetime of the prestage entry.
#[allow(dead_code)] // D5 seam API — exercised by unit tests; production arming is gate-blocked.
pub(crate) unsafe fn prestage_page(
    page_address: u64,
    bytes: *const u8,
    byte_count: u64,
) -> Result<(), String> {
    if page_address == 0 || bytes.is_null() || byte_count == 0 {
        return Err("prestage_page requires a nonempty page and host buffer".to_owned());
    }
    ensure_read_watch_materializer_registered();
    // Prefer an existing slot for this page (refresh), else the first free slot.
    for slot in &PRESTAGE_SLOTS {
        if slot.page_address.load(Ordering::Acquire) == page_address {
            slot.bytes.store(bytes as *mut u8, Ordering::Release);
            slot.byte_count.store(byte_count, Ordering::Release);
            return Ok(());
        }
    }
    for slot in &PRESTAGE_SLOTS {
        if slot
            .page_address
            .compare_exchange(0, page_address, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            slot.bytes.store(bytes as *mut u8, Ordering::Release);
            slot.byte_count.store(byte_count, Ordering::Release);
            return Ok(());
        }
    }
    let refused = PRESTAGE_REFUSALS.fetch_add(1, Ordering::Relaxed) + 1;
    if refused <= 16 || refused.is_power_of_two() {
        eprintln!(
            "[REMu][METAL][D5] prestage table full sequence={refused} page={page_address:#x} bytes={byte_count}; refusing rather than reallocating in the materializer path"
        );
    }
    Err(format!(
        "D5 prestage table full ({PRESTAGE_SLOT_COUNT} slots); cannot stage page {page_address:#x}"
    ))
}

/// Drops a previously prestaged page so a later fault refuses loudly instead of copying stale bytes.
#[allow(dead_code)] // D5 seam API — paired with prestage_page; production arming is gate-blocked.
pub(crate) fn clear_prestaged_page(page_address: u64) {
    for slot in &PRESTAGE_SLOTS {
        if slot
            .page_address
            .compare_exchange(page_address, 0, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            slot.bytes.store(std::ptr::null_mut(), Ordering::Release);
            slot.byte_count.store(0, Ordering::Release);
            return;
        }
    }
}

/// Async-signal-safe materializer: bounded memcpy from a prestaged host buffer, or loud refuse.
///
/// # Safety
///
/// Invoked from the SIGSEGV/SIGBUS handler with `[page_address, page_address + page_size)` writable.
unsafe extern "C" fn materialize_prestaged_guest_page(page_address: u64, page_size: u64) -> bool {
    if page_address == 0 || page_size == 0 {
        return false;
    }
    for slot in &PRESTAGE_SLOTS {
        if slot.page_address.load(Ordering::Acquire) != page_address {
            continue;
        }
        let src = slot.bytes.load(Ordering::Acquire);
        let available = slot.byte_count.load(Ordering::Acquire);
        if src.is_null() || available == 0 {
            break;
        }
        let copy = available.min(page_size) as usize;
        // SAFETY: handler made the destination page writable; src is the producer buffer the
        // prestage publisher promised remains readable for this page's lifetime.
        unsafe {
            std::ptr::copy_nonoverlapping(src, page_address as *mut u8, copy);
        }
        return true;
    }
    let miss = MATERIALIZE_MISS.fetch_add(1, Ordering::Relaxed) + 1;
    // eprintln is NOT async-signal-safe; count only. The fault is forwarded (return false) so the
    // failure is loud as a guest SIGSEGV rather than a silent wrong-bytes read.
    let _ = miss;
    false
}

/// Loud refusal helper for call sites that would arm a read-watch without a prestage.
///
/// Returns an error naming the seam; never silently falls back to writeback or Metal-in-handler.
#[allow(dead_code)] // D5 seam — call sites land with resident read-watch arming (gated).
pub(crate) fn refuse_unprestaged_read_watch(page_address: u64) -> Result<(), String> {
    Err(format!(
        "D5 read-watch arm refused for page {page_address:#x}: no host-visible prestage; Metal readback is not async-signal-safe (see texture/read_watch.rs + NOTES Batch B4 seam)"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn materializer_copies_prestaged_page_and_refuses_unstaged() {
        ensure_read_watch_materializer_registered();
        assert!(read_watch_materializer_installed());

        let page_size = 4096_u64;
        let host = vec![0xcd_u8; page_size as usize];
        let mut guest = vec![0_u8; page_size as usize];
        let page_address = guest.as_mut_ptr() as u64;
        // SAFETY: host lives for the prestage lifetime below.
        unsafe {
            prestage_page(page_address, host.as_ptr(), page_size).expect("prestage");
        }
        // SAFETY: destination is our owned guest buffer; mimics the handler-writable page.
        assert!(unsafe { materialize_prestaged_guest_page(page_address, page_size) });
        assert!(guest.iter().all(|&b| b == 0xcd));

        clear_prestaged_page(page_address);
        guest.fill(0);
        assert!(
            !unsafe { materialize_prestaged_guest_page(page_address, page_size) },
            "missing prestage must refuse (loud), never invent bytes"
        );
        assert!(guest.iter().all(|&b| b == 0));
        assert!(refuse_unprestaged_read_watch(page_address).is_err());
    }
}
