// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Placement, validation and host-mapping internals behind the address-space API.

use remu_core::GuestAddress;

use super::shared::{
    begin_shared_mapping_mutation, existing_shared_backing_alias_segments,
    register_shared_mapping_range, unregister_shared_mapping_range,
};
use super::{
    AddressSpaceError, CpuWriteWatch, GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement,
    GuestPhysicalBacking, GuestProtection, Vma, VmaKind, align_up, checked_end, overlap_error,
    require_aligned,
};
#[cfg(target_os = "macos")]
use super::{host_error, host_pointer, host_size};

impl GuestAddressSpace {
    pub(super) fn validate_allocation(size: u64, alignment: u64) -> Result<(), AddressSpaceError> {
        if size == 0 {
            return Err(AddressSpaceError::ZeroSize);
        }
        require_aligned("mapping size", size, GUEST_PAGE_SIZE)?;
        if alignment < GUEST_PAGE_SIZE
            || !alignment.is_power_of_two()
            || !alignment.is_multiple_of(GUEST_PAGE_SIZE)
        {
            return Err(AddressSpaceError::InvalidAlignment { alignment });
        }
        Ok(())
    }

    pub(super) fn validate_page_range(
        &self,
        address: GuestAddress,
        size: u64,
    ) -> Result<(u64, u64), AddressSpaceError> {
        if size == 0 {
            return Err(AddressSpaceError::ZeroSize);
        }
        require_aligned("range address", address.get(), GUEST_PAGE_SIZE)?;
        require_aligned("range size", size, GUEST_PAGE_SIZE)?;
        let end = checked_end(address.get(), size)?;
        self.validate_bounds(address.get(), end)?;
        Ok((address.get(), end))
    }

    pub(super) fn validate_byte_range(
        &self,
        address: GuestAddress,
        size: usize,
    ) -> Result<(u64, u64), AddressSpaceError> {
        let size_u64 = u64::try_from(size)
            .map_err(|_| AddressSpaceError::HostSizeOverflow { value: u64::MAX })?;
        let end = checked_end(address.get(), size_u64)?;
        if size == 0 && address.get() == self.end {
            return Ok((address.get(), end));
        }
        self.validate_bounds(address.get(), end)?;
        Ok((address.get(), end))
    }

    pub(super) fn validate_bounds(&self, start: u64, end: u64) -> Result<(), AddressSpaceError> {
        if start < self.base().get() || end > self.end {
            return Err(AddressSpaceError::OutOfBounds {
                address: start,
                end,
                space_start: self.base().get(),
                space_end: self.end,
            });
        }
        Ok(())
    }

    pub(super) fn resolve_free_placement(
        &self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
    ) -> Result<u64, AddressSpaceError> {
        match placement {
            GuestMapPlacement::Fixed(address) => {
                let start = address.get();
                require_aligned("fixed address", start, alignment)?;
                let end = checked_end(start, size)?;
                self.validate_bounds(start, end)?;
                self.reject_any_overlap(start, end)?;
                Ok(start)
            }
            GuestMapPlacement::Anywhere => self
                .find_free(self.base().get(), self.end, size, alignment)
                .ok_or(AddressSpaceError::AddressSpaceExhausted { size, alignment }),
            GuestMapPlacement::Hint(hint) => {
                self.validate_hint(hint.get())?;
                self.find_free(hint.get(), self.end, size, alignment)
                    .or_else(|| self.find_free(self.base().get(), hint.get(), size, alignment))
                    .ok_or(AddressSpaceError::AddressSpaceExhausted { size, alignment })
            }
        }
    }

    pub(super) fn resolve_mapping_placement(
        &self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
    ) -> Result<u64, AddressSpaceError> {
        if let GuestMapPlacement::Fixed(address) = placement {
            let start = address.get();
            require_aligned("fixed address", start, alignment)?;
            let end = checked_end(start, size)?;
            self.validate_bounds(start, end)?;
            self.reject_mapped_overlap(start, end)?;
            return Ok(start);
        }
        self.resolve_free_placement(placement, size, alignment)
    }

    pub(super) fn validate_hint(&self, hint: u64) -> Result<(), AddressSpaceError> {
        if hint < self.base().get() || hint >= self.end {
            return Err(AddressSpaceError::OutOfBounds {
                address: hint,
                end: hint.saturating_add(1),
                space_start: self.base().get(),
                space_end: self.end,
            });
        }
        Ok(())
    }

    pub(super) fn find_free(
        &self,
        search_start: u64,
        limit: u64,
        size: u64,
        alignment: u64,
    ) -> Option<u64> {
        let mut candidate = align_up(search_start, alignment).ok()?;
        for vma in self.vmas.values() {
            if vma.end <= candidate {
                continue;
            }
            if vma.start >= limit {
                break;
            }
            if checked_end(candidate, size).ok()? <= vma.start
                && checked_end(candidate, size).ok()? <= limit
            {
                return Some(candidate);
            }
            candidate = align_up(vma.end, alignment).ok()?;
            if candidate >= limit {
                return None;
            }
        }
        (checked_end(candidate, size).ok()? <= limit).then_some(candidate)
    }

    pub(super) fn reject_any_overlap(&self, start: u64, end: u64) -> Result<(), AddressSpaceError> {
        if let Some(vma) = self.first_overlap(start, end) {
            return Err(overlap_error(start, end, vma));
        }
        Ok(())
    }

    pub(super) fn reject_mapped_overlap(
        &self,
        start: u64,
        end: u64,
    ) -> Result<(), AddressSpaceError> {
        if let Some(vma) = self.vmas.values().find(|vma| {
            vma.start < end && vma.end > start && !matches!(vma.kind, VmaKind::Reserved)
        }) {
            return Err(overlap_error(start, end, vma));
        }
        Ok(())
    }

    pub(super) fn first_overlap(&self, start: u64, end: u64) -> Option<&Vma> {
        self.vmas
            .values()
            .find(|vma| vma.start < end && vma.end > start)
    }

    pub(super) fn ensure_covered(
        &self,
        start: u64,
        end: u64,
        require_mapped: bool,
        required: Option<GuestProtection>,
    ) -> Result<(), AddressSpaceError> {
        let mut cursor = start;
        while cursor < end {
            let Some((_, vma)) = self.vmas.range(..=cursor).next_back() else {
                return Err(AddressSpaceError::NotMapped { address: cursor });
            };
            if cursor >= vma.end {
                return Err(AddressSpaceError::NotMapped { address: cursor });
            }
            if require_mapped && matches!(vma.kind, VmaKind::Reserved) {
                return Err(AddressSpaceError::Reserved { address: cursor });
            }
            if let Some(required) = required
                && !vma.protection.contains(required)
            {
                return Err(AddressSpaceError::PermissionDenied {
                    address: cursor,
                    end: end.min(vma.end),
                    required,
                    actual: vma.protection,
                });
            }
            cursor = end.min(vma.end);
        }
        Ok(())
    }

    pub(super) fn map_host_anonymous(
        &self,
        start: u64,
        size: u64,
        protection: GuestProtection,
    ) -> Result<(), AddressSpaceError> {
        #[cfg(windows)]
        let _ = protection;
        debug_assert!(start >= self.base().get());
        debug_assert!(start.saturating_add(size) <= self.end);
        let _mapping_mutation = begin_shared_mapping_mutation();
        CpuWriteWatch::retire_overlapping(start, size);
        #[cfg(target_os = "macos")]
        {
            let pointer = host_pointer(start)?;
            // SAFETY: Placement validation proves this fixed range lies entirely inside our host
            // reservation and does not cover another mapped VMA.
            let result = unsafe {
                libc::mmap(
                    pointer,
                    host_size(size)?,
                    protection.host_flags(),
                    libc::MAP_FIXED | libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            if result == libc::MAP_FAILED {
                return Err(host_error(
                    "anonymous mapping",
                    start,
                    checked_end(start, size)?,
                ));
            }
            debug_assert_eq!(result, pointer);
        }
        #[cfg(windows)]
        self.reservation.restore_placeholder(start, size)?;
        // The fixed anonymous mapping just replaced every byte in the range with zeros without a
        // single byte store; record it so a zero-poisoned crash can name it.
        crate::record_guest_write(crate::GuestWriteSite::MapZero, start, size, &[0], 0);
        unregister_shared_mapping_range(start, size);
        Ok(())
    }

    pub(super) fn map_host_shared(
        &self,
        start: u64,
        size: u64,
        protection: GuestProtection,
        backing: &GuestPhysicalBacking,
        backing_offset: u64,
    ) -> Result<(), AddressSpaceError> {
        debug_assert!(start >= self.base().get());
        debug_assert!(start.saturating_add(size) <= self.end);
        let _mapping_mutation = begin_shared_mapping_mutation();
        let _alias_changes = existing_shared_backing_alias_segments(backing, backing_offset, size)
            .into_iter()
            .map(|segment| {
                CpuWriteWatch::begin_alias_change_overlapping(segment.address, segment.byte_count)
            })
            .collect::<Vec<_>>();
        CpuWriteWatch::retire_overlapping(start, size);
        #[cfg(target_os = "macos")]
        {
            let pointer = host_pointer(start)?;
            let offset = libc::off_t::try_from(backing_offset).map_err(|_| {
                AddressSpaceError::HostSizeOverflow {
                    value: backing_offset,
                }
            })?;
            // SAFETY: Placement and backing validation prove both the fixed VA range and file range.
            let result = unsafe {
                libc::mmap(
                    pointer,
                    host_size(size)?,
                    protection.host_flags(),
                    libc::MAP_FIXED | libc::MAP_SHARED,
                    backing.inner.descriptor,
                    offset,
                )
            };
            if result == libc::MAP_FAILED {
                return Err(host_error(
                    "shared mapping",
                    start,
                    checked_end(start, size)?,
                ));
            }
            debug_assert_eq!(result, pointer);
        }
        #[cfg(windows)]
        self.reservation
            .map_shared(start, size, protection, backing, backing_offset)?;
        register_shared_mapping_range(start, size, backing, backing_offset);
        Ok(())
    }

    pub(super) fn remove_metadata_range(&mut self, start: u64, end: u64) {
        let mut keys = Vec::new();
        if let Some((&key, vma)) = self.vmas.range(..=start).next_back()
            && vma.end > start
        {
            keys.push(key);
        }
        for (&key, _) in self.vmas.range(start..end) {
            if keys.last().copied() != Some(key) {
                keys.push(key);
            }
        }
        for key in keys {
            let original = self.vmas.remove(&key).expect("VMA key was just collected");
            if original.start < start {
                let mut left = original.clone();
                left.end = start;
                self.vmas.insert(left.start, left);
            }
            if original.end > end {
                let mut right = original;
                right.backing_offset += end - right.start;
                right.start = end;
                self.vmas.insert(right.start, right);
            }
        }
    }

    pub(super) fn split_at(&mut self, address: u64) {
        let Some((&key, vma)) = self.vmas.range(..=address).next_back() else {
            return;
        };
        if address <= vma.start || address >= vma.end {
            return;
        }
        let mut left = self.vmas.remove(&key).expect("VMA key exists");
        let mut right = left.clone();
        right.start = address;
        right.backing_offset += address - left.start;
        left.end = address;
        self.vmas.insert(left.start, left);
        self.vmas.insert(right.start, right);
    }

    /// Coalesces VMA boundaries changed by an operation on `start..end`.
    ///
    /// VMAs outside that interval were normalized before the operation, so only its predecessor,
    /// its own records, and its immediate successor can have become mergeable. Keeping the pass
    /// local prevents thousands of small fixed aliases from turning mapping into O(n²) work.
    pub(super) fn normalize_range(&mut self, start: u64, end: u64) {
        let Some(mut current_key) = self
            .vmas
            .range(..start)
            .next_back()
            .or_else(|| self.vmas.range(start..).next())
            .map(|(&key, _)| key)
        else {
            return;
        };
        loop {
            let Some(next_key) = self
                .vmas
                .range((std::ops::Bound::Excluded(current_key), std::ops::Bound::Unbounded))
                .next()
                .map(|(&key, _)| key)
            else {
                break;
            };
            let merge = {
                let current = self.vmas.get(&current_key).expect("current VMA key is live");
                let next = self.vmas.get(&next_key).expect("next VMA key is live");
                current.can_merge(next)
            };
            if merge {
                let next = self.vmas.remove(&next_key).expect("next VMA key is live");
                self.vmas
                    .get_mut(&current_key)
                    .expect("current VMA key is live")
                    .end = next.end;
                continue;
            }
            if next_key > end {
                break;
            }
            current_key = next_key;
        }
    }
}
