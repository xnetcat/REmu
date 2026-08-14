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
pub use backing::GuestSharedBackingView;

mod internals;

mod error;
pub use error::AddressSpaceError;

mod protection;
pub use protection::{
    GuestMapCoalescing, GuestMapPlacement, GuestProtection, GuestSharedMapOptions, GuestVmaInfo,
    GuestVmaState, RequestedAddress,
};

mod shared;
pub(crate) use shared::guest_write_alias_segments;
pub use shared::{
    GuardedBackingWrite, GuestRangeLease, GuestSharedBackingRange, GuestSharedBackingSegment,
    GuestSharedMappingIdentity, LeasePublishOutcome, guest_publish_release_fence,
    guest_shared_backing_alias_segments, guest_shared_backing_identity, guest_shared_backing_range,
    take_guest_shared_identity_statistics,
    guest_shared_backing_any_range_overlaps, guest_shared_backing_any_range_overlaps_resolved,
    guest_shared_backing_ranges_overlap, guest_shared_backing_segments, guest_shared_mapping_epoch,
    guest_shared_mapping_identity, guest_shared_mapping_table, load_guest_shared_u32,
    load_guest_shared_u64,
    load_guest_visible_u32, load_guest_visible_u64, publish_guest_shared_u32_if_identity,
    publish_guest_shared_u64_if_identity, read_guest_shared_backing, store_guest_visible_u32,
    store_guest_visible_u64, write_guest_shared_backing, write_guest_shared_backing_at_identity,
    write_guest_shared_backing_at_identity_with_baseline, write_guest_shared_backing_if_identity,
    write_guest_shared_backing_if_identity_and_baseline, write_guest_shared_backing_scattered,
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
    /// Lazily created backing for committed memory, spanning the whole space.
    ///
    /// Reserve-then-commit and flexible allocations have no guest-visible physical address, so
    /// their physical identity is their virtual placement: byte `address` of the space is byte
    /// `address - base` of this sparse object. Committing adjacent ranges in separate operations
    /// therefore yields contiguous backing slices, and a GPU resource spanning that seam still
    /// resolves as one physical run — mirroring hardware, where the GPU reads such memory through
    /// per-process page tables by virtual address.
    commit_backing: Option<GuestPhysicalBacking>,
    /// Opt-in first-write probes installed as soon as their guest mapping exists.
    ///
    /// Texture write watches are normally created only when the GPU first samples a resource.
    /// `REMU_TRACE_GUEST_CPU_WRITE_RANGE` needs to observe initialization which can happen before
    /// that point, so the address space retains this diagnostic watch for the mapping's lifetime.
    diagnostic_write_watches: Vec<CpuWriteWatch>,
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
        reservation.publish_bounds();
        Ok(Self {
            reservation,
            end,
            vmas: BTreeMap::new(),
            commit_backing: None,
            diagnostic_write_watches: Vec::new(),
        })
    }

    /// Returns the space-wide committed-memory backing, creating it on first commit.
    ///
    /// The object is sparse: pages materialize when first touched, so its full-space size costs
    /// address range only. Decommit zeroes pages before releasing them (see [`Self::unmap`]),
    /// which keeps retained object pages trivially compressible and guarantees that recommitting
    /// the same addresses observes fresh zeroed memory.
    fn committed_backing(&mut self) -> Result<GuestPhysicalBacking, AddressSpaceError> {
        if let Some(backing) = &self.commit_backing {
            return Ok(backing.clone());
        }
        let backing = GuestPhysicalBacking::new_virtually_keyed(self.size())?;
        self.commit_backing = Some(backing.clone());
        Ok(backing)
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

    /// Maps zero-filled memory that behaves like anonymous for the CPU while remaining
    /// GPU-referencable.
    ///
    /// Phase 1G: host-private anonymous pages cannot back Metal (`newBufferWithBytesNoCopy`
    /// requires a shared physical identity). Route every former anonymous map through the
    /// committed-backing path so residency and the GPU tiler resolve the same way as flexible
    /// / commit-on-protect memory. A fixed map may commit free or reserved pages, but never
    /// replaces an existing committed or shared mapping.
    pub fn map_anonymous(
        &mut self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
        protection: GuestProtection,
    ) -> Result<GuestAddress, AddressSpaceError> {
        self.map_committed(placement, size, alignment, protection)
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

    /// Maps zero-filled committed memory backed by the space's virtually-keyed backing object.
    ///
    /// This is the mapping form for guest allocations that never see a physical address —
    /// flexible memory and commit-on-protect ranges. The result behaves like anonymous memory for
    /// the CPU while remaining resolvable by GPU residency and write-watching through the shared
    /// backing registry. A fixed placement may commit free or reserved pages, but never replaces
    /// an existing anonymous, committed, or shared mapping.
    pub fn map_committed(
        &mut self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
        protection: GuestProtection,
    ) -> Result<GuestAddress, AddressSpaceError> {
        Self::validate_allocation(size, alignment)?;
        let start = self.resolve_mapping_placement(placement, size, alignment)?;
        let end = checked_end(start, size)?;
        let backing = self.committed_backing()?;
        let backing_offset = start - self.base().get();
        self.map_host_shared(start, size, protection, &backing, backing_offset)?;
        self.remove_metadata_range(start, end);
        self.vmas.insert(
            start,
            Vma {
                start,
                end,
                protection,
                kind: VmaKind::Committed(backing),
                backing_offset,
                metadata: 0,
                mapping_id: 0,
            },
        );
        self.normalize();
        Ok(GuestAddress::new(start))
    }

    /// Releases mapped or reserved pages and restores the host guard mapping.
    ///
    /// Committed pages are zeroed in their backing object: decommitted contents must never
    /// resurface when the guest later recommits the same addresses, and zeroed retained pages
    /// cost the host nothing once compressed.
    pub fn unmap(&mut self, address: GuestAddress, size: u64) -> Result<(), AddressSpaceError> {
        let (start, end) = self.validate_page_range(address, size)?;
        self.ensure_covered(start, end, false, None)?;
        self.map_host_anonymous(start, size, GuestProtection::NONE)?;
        // Zero only after the host remap above has retired overlapping write watches. A GPU
        // writeback pinned by a still-live watch could otherwise land between zeroing and
        // retirement and leave its bytes in the retained backing, so a later recommit of these
        // addresses would observe them instead of fresh zeros.
        self.zero_committed_backing(start, end);
        self.remove_metadata_range(start, end);
        self.normalize();
        Ok(())
    }

    /// Zeroes the committed-backing bytes behind every committed VMA intersecting the range.
    fn zero_committed_backing(&self, start: u64, end: u64) {
        const ZERO_CHUNK: usize = 1 << 20;
        let zeroes = vec![0_u8; ZERO_CHUNK];
        for vma in self.vmas.values() {
            let VmaKind::Committed(backing) = &vma.kind else {
                continue;
            };
            let intersection_start = vma.start.max(start);
            let intersection_end = vma.end.min(end);
            if intersection_start >= intersection_end {
                continue;
            }
            let mut offset = vma.backing_offset + (intersection_start - vma.start);
            let mut remaining = intersection_end - intersection_start;
            while remaining > 0 {
                let chunk = usize::try_from(remaining.min(ZERO_CHUNK as u64))
                    .expect("chunk length is bounded by ZERO_CHUNK");
                backing.write_all_at(offset, &zeroes[..chunk]);
                offset += chunk as u64;
                remaining -= chunk as u64;
            }
        }
    }

    /// Changes protection on claimed pages and returns how many reserved bytes were committed.
    ///
    /// Granting access to reserved pages commits them: on the console, `sceKernelMprotect` over a
    /// reserved range is the commit step of the reserve-then-commit allocation pattern, and the
    /// resulting memory is real, zero-filled, and GPU-referencable. Committed pages join the
    /// space's virtually-keyed backing so the shared-backing registry resolves them. Partial
    /// ranges split VMAs and equal adjacent pieces are merged again automatically; protection
    /// changes on already-committed pages never move or clear their contents.
    pub fn protect(
        &mut self,
        address: GuestAddress,
        size: u64,
        protection: GuestProtection,
    ) -> Result<u64, AddressSpaceError> {
        let (start, end) = self.validate_page_range(address, size)?;
        self.ensure_covered(start, end, false, None)?;
        self.split_at(start);
        self.split_at(end);
        let subranges: Vec<(u64, u64, bool)> = self
            .vmas
            .range(start..end)
            .map(|(&sub_start, vma)| {
                (
                    sub_start,
                    vma.end,
                    matches!(vma.kind, VmaKind::Reserved) && protection != GuestProtection::NONE,
                )
            })
            .collect();
        let mut committed_bytes = 0;
        for (sub_start, sub_end, commit) in subranges {
            let sub_size = sub_end - sub_start;
            if commit {
                let backing = self.committed_backing()?;
                let backing_offset = sub_start - self.base().get();
                self.map_host_shared(sub_start, sub_size, protection, &backing, backing_offset)?;
                // Committing a Reserved range replaces whatever the VA showed with the backing's
                // pages. If a VMA-kind desync ever marks a populated range Reserved, this is the
                // silent bulk zero that destroys it — record it so a crash can say so.
                crate::record_guest_write(
                    crate::GuestWriteSite::MapCommit,
                    sub_start,
                    sub_size,
                    &[0],
                    0,
                );
                if let Some(vma) = self.vmas.get_mut(&sub_start) {
                    vma.kind = VmaKind::Committed(backing);
                    vma.backing_offset = backing_offset;
                    vma.protection = protection;
                }
                committed_bytes += sub_size;
                continue;
            }
            let _mapping_mutation = shared::begin_shared_mapping_mutation();
            let reprotection = CpuWriteWatch::begin_reprotect_overlapping(sub_start, sub_size);
            // SAFETY: Validation proves this subrange is claimed inside our host reservation.
            let result = unsafe {
                libc::mprotect(
                    host_pointer(sub_start)?,
                    host_size(sub_size)?,
                    protection.host_flags(),
                )
            };
            drop(reprotection);
            if result != 0 {
                self.normalize();
                return Err(host_error("protection change", sub_start, sub_end));
            }
            if let Some(vma) = self.vmas.get_mut(&sub_start) {
                vma.protection = protection;
            }
        }
        self.normalize();
        Ok(committed_bytes)
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
