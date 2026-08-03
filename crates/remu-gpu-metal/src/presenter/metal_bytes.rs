// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Length-checked access to a Metal buffer's shared CPU-visible `contents()` (M2).
//!
//! Metal's `MTLBuffer::contents()` returns a bare pointer into the shared allocation with no
//! associated length, so every reader across the crate paired it with a hand-written
//! `std::slice::from_raw_parts` / `.add(offset)` / `copy_nonoverlapping`, each its own `unsafe`
//! block that had to re-argue the same bounds invariant. `MetalBytes` captures that pointer with
//! the buffer's byte length and borrows the buffer, so the pointer cannot outlive the mapping.
//! Every accessor bounds-checks against the captured length, collapsing those scattered `unsafe`
//! sites into a single audited boundary here.

use metal::BufferRef;

/// A bounds-checked window over one Metal buffer's shared `contents()` mapping.
///
/// The lifetime `'a` is the borrow of the underlying buffer; every returned slice inherits it, so
/// the shared mapping is provably live for as long as the caller holds the reference.
pub(crate) struct MetalBytes<'a> {
    base: *mut u8,
    len: usize,
    // Ties `base`'s validity to the borrowed buffer: the mapping cannot be freed while the borrow
    // (and therefore any slice derived from it) is live.
    _buffer: &'a BufferRef,
}

impl<'a> MetalBytes<'a> {
    /// Wraps `buffer`'s shared CPU mapping. A null `contents()` (a non-CPU-visible storage mode)
    /// is represented as a zero-length window, so every accessor then fails its bounds check
    /// instead of dereferencing null.
    #[must_use]
    pub(crate) fn new(buffer: &'a BufferRef) -> Self {
        let base = buffer.contents().cast::<u8>();
        let len = if base.is_null() {
            0
        } else {
            usize::try_from(buffer.length()).unwrap_or(usize::MAX)
        };
        Self {
            base,
            len,
            _buffer: buffer,
        }
    }

    fn in_bounds(&self, offset: usize, len: usize) -> bool {
        offset
            .checked_add(len)
            .is_some_and(|end| end <= self.len)
    }

    /// An immutable byte subslice, or `None` if `[offset, offset + len)` leaves the mapping.
    #[must_use]
    pub(crate) fn slice(&self, offset: usize, len: usize) -> Option<&'a [u8]> {
        if !self.in_bounds(offset, len) {
            return None;
        }
        // SAFETY: the bounds check proves `offset + len <= self.len`, and `_buffer` keeps the
        // shared mapping live for `'a`. The GPU aliases this storage; ordering GPU vs. CPU access
        // is the caller's responsibility, exactly as it was at the raw sites this replaces.
        Some(unsafe { std::slice::from_raw_parts(self.base.add(offset), len) })
    }

    /// A mutable byte subslice, or `None` if `[offset, offset + len)` leaves the mapping.
    #[must_use]
    #[allow(clippy::mut_from_ref)]
    pub(crate) fn slice_mut(&self, offset: usize, len: usize) -> Option<&'a mut [u8]> {
        if !self.in_bounds(offset, len) {
            return None;
        }
        // SAFETY: bounds checked above; `_buffer` keeps the shared mapping live for `'a`. Metal
        // shared storage aliases the GPU, so the caller orders access as it did at the raw sites.
        Some(unsafe { std::slice::from_raw_parts_mut(self.base.add(offset), len) })
    }

    /// A typed immutable slice of `count` `T`s starting at byte `offset`, or `None` if the range
    /// leaves the mapping or `offset` is not `T`-aligned.
    #[must_use]
    pub(crate) fn typed_slice<T: Copy>(&self, offset: usize, count: usize) -> Option<&'a [T]> {
        let byte_count = count.checked_mul(std::mem::size_of::<T>())?;
        if !self.in_bounds(offset, byte_count) {
            return None;
        }
        if (self.base as usize).wrapping_add(offset) % std::mem::align_of::<T>() != 0 {
            return None;
        }
        // SAFETY: bounds and `T`-alignment checked above; `_buffer` keeps the mapping live for
        // `'a`. `T: Copy` has no drop glue over the shared storage.
        Some(unsafe { std::slice::from_raw_parts(self.base.add(offset).cast::<T>(), count) })
    }

    /// Copies `src` into the mapping at `offset`. Returns `false` (writing nothing) if the range
    /// would leave the mapping.
    #[must_use]
    pub(crate) fn write(&self, offset: usize, src: &[u8]) -> bool {
        if !self.in_bounds(offset, src.len()) {
            return false;
        }
        // SAFETY: bounds checked above; the destination is the live shared mapping and does not
        // overlap the caller-owned `src`.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.base.add(offset), src.len());
        }
        true
    }

    /// Fills `len` bytes at `offset` with `value`. Returns `false` (writing nothing) if the range
    /// would leave the mapping.
    #[must_use]
    pub(crate) fn fill(&self, offset: usize, value: u8, len: usize) -> bool {
        if !self.in_bounds(offset, len) {
            return false;
        }
        // SAFETY: bounds checked above; the destination is the live shared mapping.
        unsafe {
            std::ptr::write_bytes(self.base.add(offset), value, len);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::MetalBytes;
    use metal::{Device, MTLResourceOptions};

    #[test]
    fn slice_and_write_are_bounds_checked_against_the_mapping() {
        let device = Device::system_default().expect("Metal device");
        let buffer = device.new_buffer(16, MTLResourceOptions::StorageModeShared);
        let bytes = MetalBytes::new(&buffer);
        assert!(bytes.slice(0, 16).is_some());
        assert!(bytes.slice(0, 17).is_none());

        assert!(bytes.fill(0, 0, 16));
        assert!(bytes.write(4, &[1, 2, 3, 4]));
        assert_eq!(bytes.slice(0, 16).expect("full slice")[4..8], [1, 2, 3, 4]);

        // Ranges that leave the mapping refuse instead of reading out of bounds.
        assert!(bytes.slice(12, 8).is_none());
        assert!(bytes.slice(usize::MAX, 1).is_none());
        assert!(!bytes.write(14, &[0, 0, 0, 0]));
        assert!(!bytes.fill(15, 0, 4));
    }

    #[test]
    fn typed_slice_checks_length_and_alignment() {
        let device = Device::system_default().expect("Metal device");
        let buffer = device.new_buffer(16, MTLResourceOptions::StorageModeShared);
        let bytes = MetalBytes::new(&buffer);

        assert!(bytes.write(0, &7_u32.to_ne_bytes()));
        let words = bytes.typed_slice::<u32>(0, 4).expect("four aligned words");
        assert_eq!(words[0], 7);
        // One past the mapping is refused.
        assert!(bytes.typed_slice::<u32>(0, 5).is_none());
        // A one-byte offset is not u32-aligned (base is page aligned), so it is refused rather
        // than producing an unaligned load.
        assert!(bytes.typed_slice::<u32>(1, 1).is_none());
    }
}
