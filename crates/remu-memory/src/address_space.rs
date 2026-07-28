// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Sparse, directly addressable guest virtual memory for the PS5's 16 KiB pages.
//!
//! The address space owns one inaccessible host reservation. Committing anonymous or shared
//! memory replaces pages inside that reservation, and unmapping restores inaccessible pages. This
//! keeps guest pointers directly usable by translated code while preventing fixed mappings from
//! overwriting unrelated host memory.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::io;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

use remu_core::GuestAddress;

use super::write_watch::CpuWriteWatch;

mod backing;
pub use backing::GuestPhysicalBacking;

mod internals;

mod error;
pub use error::AddressSpaceError;

mod protection;
pub use protection::{
    GuestMapCoalescing, GuestMapPlacement, GuestProtection, GuestSharedMapOptions, GuestVmaInfo,
    GuestVmaState, RequestedAddress,
};

mod shared;
pub use shared::{
    GuestSharedBackingSegment, GuestSharedMappingIdentity, guest_shared_backing_alias_segments,
    guest_shared_backing_segments, guest_shared_mapping_identity, read_guest_shared_backing,
    write_guest_shared_backing,
};

mod reservation;
use reservation::HostReservation;

mod vma;
use vma::{Vma, VmaKind};

pub const GUEST_PAGE_SIZE: u64 = 16 * 1024;

static NEXT_MAPPING_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub struct GuestAddressSpace {
    reservation: HostReservation,
    end: u64,
    vmas: BTreeMap<u64, Vma>,
}

impl GuestAddressSpace {
    /// Reserves an address space at a host-selected, 16 KiB-aligned address.
    pub fn new(size: u64) -> Result<Self, AddressSpaceError> {
        validate_space_size(size)?;
        let reservation = HostReservation::anywhere(size)?;
        Self::from_reservation(reservation)
    }

    /// Reserves an address space whose guest addresses equal the requested host addresses.
    pub fn new_at(base: GuestAddress, size: u64) -> Result<Self, AddressSpaceError> {
        validate_space_size(size)?;
        require_aligned("address-space base", base.get(), GUEST_PAGE_SIZE)?;
        checked_end(base.get(), size)?;
        let reservation = HostReservation::exact(base.get(), size)?;
        Self::from_reservation(reservation)
    }

    fn from_reservation(reservation: HostReservation) -> Result<Self, AddressSpaceError> {
        let end = checked_end(reservation.address(), reservation.size as u64)?;
        Ok(Self {
            reservation,
            end,
            vmas: BTreeMap::new(),
        })
    }

    #[must_use]
    pub fn base(&self) -> GuestAddress {
        GuestAddress::new(self.reservation.address())
    }

    #[must_use]
    pub fn size(&self) -> u64 {
        self.reservation.size as u64
    }

    #[must_use]
    pub const fn end(&self) -> GuestAddress {
        GuestAddress::new(self.end)
    }

    /// Reserves a guest VMA without granting host access to it.
    pub fn reserve(
        &mut self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
    ) -> Result<GuestAddress, AddressSpaceError> {
        Self::validate_allocation(size, alignment)?;
        let start = self.resolve_free_placement(placement, size, alignment)?;
        let end = checked_end(start, size)?;
        self.vmas.insert(
            start,
            Vma {
                start,
                end,
                protection: GuestProtection::NONE,
                kind: VmaKind::Reserved,
                backing_offset: 0,
                metadata: 0,
                mapping_id: 0,
            },
        );
        self.normalize();
        Ok(GuestAddress::new(start))
    }

    /// Maps zero-filled private memory. A fixed map may commit free or reserved pages, but never
    /// replaces an existing anonymous or shared mapping.
    pub fn map_anonymous(
        &mut self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
        protection: GuestProtection,
    ) -> Result<GuestAddress, AddressSpaceError> {
        Self::validate_allocation(size, alignment)?;
        let start = self.resolve_mapping_placement(placement, size, alignment)?;
        let end = checked_end(start, size)?;
        self.map_host_anonymous(start, size, protection)?;
        self.remove_metadata_range(start, end);
        self.vmas.insert(
            start,
            Vma {
                start,
                end,
                protection,
                kind: VmaKind::Anonymous,
                backing_offset: 0,
                metadata: 0,
                mapping_id: 0,
            },
        );
        self.normalize();
        Ok(GuestAddress::new(start))
    }

    /// Maps a slice of physical backing. Mapping the same slice more than once creates coherent
    /// aliases without copying.
    pub fn map_shared(
        &mut self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
        protection: GuestProtection,
        backing: &GuestPhysicalBacking,
        backing_offset: u64,
    ) -> Result<GuestAddress, AddressSpaceError> {
        self.map_shared_with_coalescing(
            placement,
            size,
            alignment,
            protection,
            backing,
            GuestSharedMapOptions {
                backing_offset,
                coalescing: GuestMapCoalescing::Allowed,
            },
        )
    }

    /// Maps a shared backing slice while preserving independent mapping-call boundaries when
    /// coalescing is disabled by the guest ABI.
    pub fn map_shared_with_coalescing(
        &mut self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
        protection: GuestProtection,
        backing: &GuestPhysicalBacking,
        options: GuestSharedMapOptions,
    ) -> Result<GuestAddress, AddressSpaceError> {
        let GuestSharedMapOptions {
            backing_offset,
            coalescing,
        } = options;
        Self::validate_allocation(size, alignment)?;
        require_aligned("physical backing offset", backing_offset, GUEST_PAGE_SIZE)?;
        let backing_end = checked_end(backing_offset, size)?;
        if backing_end > backing.size() {
            return Err(AddressSpaceError::BackingOutOfBounds {
                backing_id: backing.id(),
                offset: backing_offset,
                end: backing_end,
                backing_size: backing.size(),
            });
        }
        let start = self.resolve_mapping_placement(placement, size, alignment)?;
        let end = checked_end(start, size)?;
        self.map_host_shared(start, size, protection, backing, backing_offset)?;
        self.remove_metadata_range(start, end);
        self.vmas.insert(
            start,
            Vma {
                start,
                end,
                protection,
                kind: VmaKind::Shared(backing.clone()),
                backing_offset,
                metadata: 0,
                mapping_id: match coalescing {
                    GuestMapCoalescing::Allowed => 0,
                    GuestMapCoalescing::Disallowed => next_mapping_id(),
                },
            },
        );
        self.normalize();
        Ok(GuestAddress::new(start))
    }

    /// Releases mapped or reserved pages and restores the host guard mapping.
    pub fn unmap(&mut self, address: GuestAddress, size: u64) -> Result<(), AddressSpaceError> {
        let (start, end) = self.validate_page_range(address, size)?;
        self.ensure_covered(start, end, false, None)?;
        self.map_host_anonymous(start, size, GuestProtection::NONE)?;
        self.remove_metadata_range(start, end);
        self.normalize();
        Ok(())
    }

    /// Changes protection on claimed pages. Granting access to reserved pages commits anonymous
    /// memory there; partial ranges split VMAs and equal adjacent pieces are merged again
    /// automatically.
    pub fn protect(
        &mut self,
        address: GuestAddress,
        size: u64,
        protection: GuestProtection,
    ) -> Result<(), AddressSpaceError> {
        let (start, end) = self.validate_page_range(address, size)?;
        self.ensure_covered(start, end, false, None)?;
        let reprotection = CpuWriteWatch::begin_reprotect_overlapping(start, size);
        let address_pointer = host_pointer(start)?;
        // SAFETY: Validation proves the complete range is claimed inside our host reservation.
        let result =
            unsafe { libc::mprotect(address_pointer, host_size(size)?, protection.host_flags()) };
        drop(reprotection);
        if result != 0 {
            return Err(host_error("protection change", start, end));
        }
        self.split_at(start);
        self.split_at(end);
        for (_, vma) in self.vmas.range_mut(start..end) {
            vma.protection = protection;
            if protection != GuestProtection::NONE && matches!(vma.kind, VmaKind::Reserved) {
                vma.kind = VmaKind::Anonymous;
            }
        }
        self.normalize();
        Ok(())
    }

    /// Associates opaque subsystem metadata with mapped pages. The value participates in VMA
    /// merging and is retained by partial unmap and protection operations.
    pub fn set_metadata(
        &mut self,
        address: GuestAddress,
        size: u64,
        metadata: u64,
    ) -> Result<(), AddressSpaceError> {
        let (start, end) = self.validate_page_range(address, size)?;
        self.ensure_covered(start, end, true, None)?;
        self.split_at(start);
        self.split_at(end);
        for (_, vma) in self.vmas.range_mut(start..end) {
            vma.metadata = metadata;
        }
        self.normalize();
        Ok(())
    }

    /// Returns either the tracked VMA or the complete free gap containing `address`. Protection and
    /// metadata surgery may split one mapping internally; guest mappings created with coalescing
    /// disabled retain their original query boundaries.
    pub fn query(&self, address: GuestAddress) -> Result<GuestVmaInfo, AddressSpaceError> {
        let value = address.get();
        if value < self.base().get() || value >= self.end {
            return Err(AddressSpaceError::OutOfBounds {
                address: value,
                end: value.saturating_add(1),
                space_start: self.base().get(),
                space_end: self.end,
            });
        }
        if let Some((_, vma)) = self.vmas.range(..=value).next_back()
            && value < vma.end
        {
            return Ok(vma.info());
        }
        let free_start = self
            .vmas
            .range(..=value)
            .next_back()
            .map_or(self.base().get(), |(_, vma)| vma.end);
        let free_end = self
            .vmas
            .range(value..)
            .next()
            .map_or(self.end, |(_, vma)| vma.start);
        Ok(GuestVmaInfo {
            start: GuestAddress::new(free_start),
            size: free_end - free_start,
            state: GuestVmaState::Free,
            protection: GuestProtection::NONE,
            backing_id: None,
            backing_offset: 0,
            metadata: 0,
        })
    }

    /// Returns explicit reserved and mapped VMAs in ascending address order. Free gaps are obtained
    /// with [`Self::query`].
    #[must_use]
    pub fn vmas(&self) -> Vec<GuestVmaInfo> {
        self.vmas.values().map(Vma::info).collect()
    }

    pub fn read(
        &self,
        address: GuestAddress,
        destination: &mut [u8],
    ) -> Result<(), AddressSpaceError> {
        let (start, end) = self.validate_byte_range(address, destination.len())?;
        if destination.is_empty() {
            return Ok(());
        }
        self.ensure_covered(start, end, true, Some(GuestProtection::READ))?;
        // SAFETY: Permission and coverage validation prove the source is readable for this length.
        unsafe {
            std::ptr::copy(
                host_pointer(start)?.cast_const().cast::<u8>(),
                destination.as_mut_ptr(),
                destination.len(),
            );
        }
        Ok(())
    }

    pub fn write(&mut self, address: GuestAddress, source: &[u8]) -> Result<(), AddressSpaceError> {
        let (start, end) = self.validate_byte_range(address, source.len())?;
        if source.is_empty() {
            return Ok(());
        }
        self.ensure_covered(start, end, true, Some(GuestProtection::WRITE))?;
        // SAFETY: Permission and coverage validation prove the destination is writable for this
        // length and `source` provides readable bytes of the same length.
        unsafe {
            std::ptr::copy(
                source.as_ptr(),
                host_pointer(start)?.cast::<u8>(),
                source.len(),
            );
        }
        Ok(())
    }

    /// Returns the identity-mapped host pointer after verifying that the byte is mapped.
    pub fn host_pointer(&self, address: GuestAddress) -> Result<NonNull<u8>, AddressSpaceError> {
        let value = address.get();
        let end = checked_end(value, 1)?;
        self.validate_bounds(value, end)?;
        self.ensure_covered(value, end, true, None)?;
        NonNull::new(host_pointer(value)?.cast::<u8>())
            .ok_or(AddressSpaceError::HostSizeOverflow { value })
    }
}

fn validate_space_size(size: u64) -> Result<(), AddressSpaceError> {
    if size == 0 {
        return Err(AddressSpaceError::ZeroSize);
    }
    require_aligned("address-space size", size, GUEST_PAGE_SIZE)?;
    let host_page_size = host_page_size()?;
    if host_page_size > GUEST_PAGE_SIZE || !GUEST_PAGE_SIZE.is_multiple_of(host_page_size) {
        return Err(AddressSpaceError::IncompatibleHostPageSize { host_page_size });
    }
    host_size(size)?;
    Ok(())
}

fn host_page_size() -> Result<u64, AddressSpaceError> {
    // SAFETY: sysconf has no pointer preconditions.
    let raw = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let size = u64::try_from(raw).map_err(|_| AddressSpaceError::InvalidHostPageSize)?;
    if !size.is_power_of_two() {
        return Err(AddressSpaceError::InvalidHostPageSize);
    }
    Ok(size)
}

fn next_mapping_id() -> u64 {
    loop {
        let id = NEXT_MAPPING_ID.fetch_add(1, Ordering::Relaxed);
        if id != 0 {
            return id;
        }
    }
}

fn require_aligned(
    field: &'static str,
    value: u64,
    alignment: u64,
) -> Result<(), AddressSpaceError> {
    if !value.is_multiple_of(alignment) {
        return Err(AddressSpaceError::Unaligned {
            field,
            value,
            alignment,
        });
    }
    Ok(())
}

const fn checked_end(address: u64, size: u64) -> Result<u64, AddressSpaceError> {
    match address.checked_add(size) {
        Some(end) => Ok(end),
        None => Err(AddressSpaceError::AddressOverflow { address, size }),
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, AddressSpaceError> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .ok_or(AddressSpaceError::AddressOverflow {
            address: value,
            size: alignment - 1,
        })
}

fn host_size(value: u64) -> Result<usize, AddressSpaceError> {
    usize::try_from(value).map_err(|_| AddressSpaceError::HostSizeOverflow { value })
}

fn host_pointer(value: u64) -> Result<*mut c_void, AddressSpaceError> {
    Ok(host_size(value)? as *mut c_void)
}

fn overlap_error(start: u64, end: u64, vma: &Vma) -> AddressSpaceError {
    AddressSpaceError::Overlap {
        address: start,
        end,
        existing_start: vma.start,
        existing_end: vma.end,
        state: vma.kind.state(),
    }
}

fn host_error(operation: &'static str, start: u64, end: u64) -> AddressSpaceError {
    AddressSpaceError::HostOperationFailed {
        operation,
        address: start,
        end,
        source: io::Error::last_os_error(),
    }
}

#[cfg(test)]
mod tests;
