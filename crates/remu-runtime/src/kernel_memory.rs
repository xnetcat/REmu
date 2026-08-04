// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::BTreeMap;

use remu_core::GuestAddress;
#[cfg(test)]
use remu_memory::GUEST_PAGE_SIZE;
use remu_memory::{
    AddressSpaceError, CpuWriteWatch, GuestAddressSpace, GuestMapCoalescing, GuestMapPlacement,
    GuestPhysicalBacking, GuestProtection, GuestSharedMapOptions, GuestVmaInfo, GuestVmaState,
};
use thiserror::Error;

pub(super) const DIRECT_MEMORY_SIZE: u64 = 16 * 1024 * 1024 * 1024;
/// Size of the flexible-memory pool the guest may map without owning direct memory.
///
/// Flexible memory is a separate, much smaller budget than direct memory: the kernel commits it
/// on demand and charges every flexible mapping against this total. The figure pairs with the
/// 16 GiB direct-memory total above and matches SharpEmu, whose PS5 kernel surface uses the same
/// pair. A title may override its own budget through `mem_param->flexible_memory_size` in its
/// process parameters; that override is not read yet, and no title exercised here sets it.
pub(super) const FLEXIBLE_MEMORY_SIZE: u64 = 448 * 1024 * 1024;
/// Size of the managed guest virtual address space.
///
/// This is address space, not memory: the host reservation is one `PROT_NONE` mapping and pages
/// are committed only when the guest maps them. Console user address space spans roughly a
/// terabyte, and engines reserve very large sparse ranges up front — Unreal reserves hundreds of
/// gigabytes for its virtual-memory allocator before committing any of it — so a space sized for
/// expected residency instead of the architectural range fails those reservations.
const VIRTUAL_MEMORY_SIZE: u64 = 1024 * 1024 * 1024 * 1024;

#[derive(Debug, Error)]
pub(super) enum KernelMemoryError {
    #[error(transparent)]
    AddressSpace(#[from] AddressSpaceError),
    #[error("direct-memory range 0x{start:x}..0x{end:x} contains unallocated physical pages")]
    DirectMemoryNotAllocated { start: u64, end: u64 },
    #[error("no direct-memory range can satisfy size 0x{size:x} aligned to 0x{alignment:x}")]
    DirectMemoryExhausted { size: u64, alignment: u64 },
    #[error("direct-memory range beginning at 0x{start:x} with size 0x{size:x} overflows")]
    DirectMemoryOverflow { start: u64, size: u64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct KernelVirtualMapping {
    pub address: GuestAddress,
    pub size: u64,
    pub direct_start: u64,
    pub protection: i32,
    pub memory_type: i32,
    pub kind: KernelVirtualMappingKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum KernelVirtualMappingKind {
    Reserved,
    Flexible,
    Direct,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct KernelDirectMapOptions {
    pub raw_protection: i32,
    pub direct_start: u64,
    pub memory_type: Option<i32>,
    pub coalescing: GuestMapCoalescing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct KernelDirectAllocation {
    pub start: u64,
    pub end: u64,
    pub memory_type: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DirectAllocation {
    start: u64,
    end: u64,
    memory_type: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DirectSegment {
    start: u64,
    end: u64,
    memory_type: i32,
}

#[derive(Debug)]
struct DirectMemory {
    backing: GuestPhysicalBacking,
    allocations: BTreeMap<u64, DirectAllocation>,
}

impl DirectMemory {
    fn new() -> Result<Self, AddressSpaceError> {
        Ok(Self {
            backing: GuestPhysicalBacking::new(DIRECT_MEMORY_SIZE)?,
            allocations: BTreeMap::new(),
        })
    }

    fn allocate(
        &mut self,
        search_start: u64,
        search_end: u64,
        size: u64,
        alignment: u64,
        memory_type: i32,
    ) -> Result<u64, KernelMemoryError> {
        let mut candidate =
            align_up(search_start, alignment).ok_or(KernelMemoryError::DirectMemoryOverflow {
                start: search_start,
                size: alignment,
            })?;
        for allocation in self.allocations.values() {
            if allocation.end <= candidate {
                continue;
            }
            let candidate_end =
                candidate
                    .checked_add(size)
                    .ok_or(KernelMemoryError::DirectMemoryOverflow {
                        start: candidate,
                        size,
                    })?;
            if candidate_end <= allocation.start.min(search_end) {
                return Ok(self.insert(candidate, candidate_end, memory_type));
            }
            if allocation.start >= search_end {
                break;
            }
            candidate = align_up(candidate.max(allocation.end), alignment).ok_or(
                KernelMemoryError::DirectMemoryOverflow {
                    start: candidate,
                    size: alignment,
                },
            )?;
        }
        let end = candidate
            .checked_add(size)
            .ok_or(KernelMemoryError::DirectMemoryOverflow {
                start: candidate,
                size,
            })?;
        if end <= search_end {
            return Ok(self.insert(candidate, end, memory_type));
        }
        Err(KernelMemoryError::DirectMemoryExhausted { size, alignment })
    }

    fn insert(&mut self, start: u64, end: u64, memory_type: i32) -> u64 {
        self.allocations.insert(
            start,
            DirectAllocation {
                start,
                end,
                memory_type,
            },
        );
        start
    }

    fn release(&mut self, start: u64, size: u64) -> Result<bool, KernelMemoryError> {
        let Some(owner) = self.containing_allocation(start, size)? else {
            return Ok(false);
        };
        let end = start + size;
        self.allocations.remove(&owner.start);
        if owner.start < start {
            self.insert(owner.start, start, owner.memory_type);
        }
        if end < owner.end {
            self.insert(end, owner.end, owner.memory_type);
        }
        Ok(true)
    }

    fn containing_allocation(
        &self,
        start: u64,
        size: u64,
    ) -> Result<Option<DirectAllocation>, KernelMemoryError> {
        let end = start
            .checked_add(size)
            .ok_or(KernelMemoryError::DirectMemoryOverflow { start, size })?;
        Ok(self
            .allocations
            .range(..=start)
            .next_back()
            .map(|(_, allocation)| *allocation)
            .filter(|allocation| start >= allocation.start && end <= allocation.end))
    }

    fn segments(&self, start: u64, size: u64) -> Result<Vec<DirectSegment>, KernelMemoryError> {
        let end = start
            .checked_add(size)
            .ok_or(KernelMemoryError::DirectMemoryOverflow { start, size })?;
        if end > DIRECT_MEMORY_SIZE {
            return Err(KernelMemoryError::DirectMemoryNotAllocated { start, end });
        }
        let mut cursor = start;
        let mut segments = Vec::new();
        while cursor < end {
            let allocation = self
                .allocations
                .range(..=cursor)
                .next_back()
                .map(|(_, allocation)| *allocation)
                .filter(|allocation| cursor >= allocation.start && cursor < allocation.end)
                .ok_or(KernelMemoryError::DirectMemoryNotAllocated { start: cursor, end })?;
            let segment_end = end.min(allocation.end);
            segments.push(DirectSegment {
                start: cursor,
                end: segment_end,
                memory_type: allocation.memory_type,
            });
            cursor = segment_end;
        }
        Ok(segments)
    }

    fn query(&self, offset: u64, find_next: bool) -> Option<DirectAllocation> {
        if let Some(allocation) = self
            .allocations
            .range(..=offset)
            .next_back()
            .map(|(_, allocation)| *allocation)
            .filter(|allocation| offset < allocation.end)
        {
            return Some(allocation);
        }
        find_next
            .then(|| {
                self.allocations
                    .range(offset..)
                    .next()
                    .map(|(_, value)| *value)
            })
            .flatten()
    }

    /// Reports the largest unallocated span inside `[search_start, search_end)` whose base meets
    /// `alignment`, as `(start, size)`.
    ///
    /// This is the query side of [`Self::allocate`] and must stay consistent with it: a guest
    /// asks how much direct memory is left and then allocates exactly that, so a span reported
    /// here that `allocate` would refuse corrupts the guest's own accounting.
    fn available(&self, search_start: u64, search_end: u64, alignment: u64) -> Option<(u64, u64)> {
        let mut best: Option<(u64, u64)> = None;
        let mut cursor = search_start;
        for allocation in self.allocations.values() {
            if allocation.end <= cursor {
                continue;
            }
            if allocation.start >= search_end {
                break;
            }
            if cursor < allocation.start {
                best = larger_aligned_span(best, cursor, allocation.start, alignment);
            }
            cursor = allocation.end;
            if cursor >= search_end {
                return best;
            }
        }
        larger_aligned_span(best, cursor, search_end, alignment)
    }
}

/// Keeps whichever is larger of `best` and the aligned remainder of `[hole_start, hole_end)`.
fn larger_aligned_span(
    best: Option<(u64, u64)>,
    hole_start: u64,
    hole_end: u64,
    alignment: u64,
) -> Option<(u64, u64)> {
    // Alignment can consume the whole hole, and an alignment near u64::MAX can overflow the
    // rounding itself. Neither is an error; the hole simply offers nothing, and an earlier hole
    // that did offer something must survive.
    let Some(base) = align_up(hole_start, alignment) else {
        return best;
    };
    if base >= hole_end {
        return best;
    }
    let size = hole_end - base;
    match best {
        Some((_, best_size)) if best_size >= size => best,
        _ => Some((base, size)),
    }
}

#[derive(Debug)]
pub(super) struct KernelMemory {
    virtual_memory: GuestAddressSpace,
    direct_memory: DirectMemory,
}

impl KernelMemory {
    pub fn new() -> Result<Self, KernelMemoryError> {
        Ok(Self {
            virtual_memory: GuestAddressSpace::new(VIRTUAL_MEMORY_SIZE)?,
            direct_memory: DirectMemory::new()?,
        })
    }

    #[cfg(test)]
    pub fn virtual_base(&self) -> GuestAddress {
        self.virtual_memory.base()
    }

    pub fn contains_virtual_address(&self, address: GuestAddress) -> bool {
        address >= self.virtual_memory.base() && address < self.virtual_memory.end()
    }

    /// Reserves an inaccessible virtual range for the Gen5 virtual-range ABI.
    ///
    /// A reservation claims addresses without committing pages: the range reports protection zero
    /// through `sceKernelVirtualQuery` and grants no CPU access. Its purpose is to let the guest
    /// place later fixed mappings, so the range must stay replaceable — committing it as an
    /// ordinary anonymous mapping makes every subsequent `MAP_FIXED` into the reservation fail as
    /// an overlap.
    pub fn reserve_virtual(
        &mut self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
    ) -> Result<GuestAddress, KernelMemoryError> {
        Ok(self.virtual_memory.reserve(placement, size, alignment)?)
    }

    /// Maps zero-filled flexible memory with the guest-visible protection metadata intact.
    ///
    /// Flexible memory is committed guest memory on a unified-memory console: the GPU may read
    /// it through resource descriptors, so it maps through the committed-backing path rather
    /// than as host-private anonymous pages.
    pub fn map_flexible(
        &mut self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
        raw_protection: i32,
    ) -> Result<GuestAddress, KernelMemoryError> {
        let mapped = self.virtual_memory.map_committed(
            placement,
            size,
            alignment,
            translate_protection(raw_protection),
        )?;
        if let Err(error) = self.virtual_memory.set_metadata(
            mapped,
            size,
            encode_metadata(raw_protection, 0) | FLEXIBLE_METADATA_BIT,
        ) {
            let _ = self.virtual_memory.unmap(mapped, size);
            return Err(error.into());
        }
        Ok(mapped)
    }

    /// Returns the currently readable identity-mapped virtual ranges.
    ///
    /// GPU command decoding uses this snapshot to validate process-local copies without issuing a
    /// Mach VM read syscall for every small shader resource. Callers must refresh their snapshot
    /// after changing a mapping or its protection. Adjacent readable VMAs are one continuous host
    /// virtual range for that purpose even when their guest query metadata or physical backing
    /// offsets differ.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub fn readable_ranges(&self) -> Vec<(u64, u64)> {
        let mut ranges: Vec<(u64, u64)> = Vec::new();
        for mapping in self.virtual_memory.vmas().into_iter().filter(|mapping| {
            matches!(
                mapping.state,
                GuestVmaState::Anonymous | GuestVmaState::Committed | GuestVmaState::Shared
            ) && mapping.protection.contains(GuestProtection::READ)
        }) {
            let Some(end) = mapping.start.get().checked_add(mapping.size) else {
                continue;
            };
            let start = mapping.start.get();
            if let Some((_, previous_end)) = ranges.last_mut()
                && *previous_end == start
            {
                *previous_end = end;
            } else {
                ranges.push((start, end));
            }
        }
        ranges
    }

    /// Returns the number of bytes currently mapped as flexible memory.
    ///
    /// This is derived from the live mappings rather than kept in a running counter, so every
    /// unmap, partial unmap, replacement, and failed-mapping rollback is accounted for by
    /// construction instead of by remembering to adjust a total on each path.
    pub fn flexible_usage(&self) -> u64 {
        self.virtual_memory
            .vmas()
            .into_iter()
            .filter(|mapping| mapping.metadata & FLEXIBLE_METADATA_BIT != 0)
            .map(|mapping| mapping.size)
            .sum()
    }

    /// Returns the flexible-memory bytes the guest may still map.
    pub fn available_flexible(&self) -> u64 {
        FLEXIBLE_MEMORY_SIZE.saturating_sub(self.flexible_usage())
    }

    pub fn allocate_direct(
        &mut self,
        search_start: u64,
        search_end: u64,
        size: u64,
        alignment: u64,
        memory_type: i32,
    ) -> Result<u64, KernelMemoryError> {
        self.direct_memory
            .allocate(search_start, search_end, size, alignment, memory_type)
    }

    pub fn available_direct(
        &self,
        search_start: u64,
        search_end: u64,
        alignment: u64,
    ) -> Option<(u64, u64)> {
        self.direct_memory
            .available(search_start, search_end, alignment)
    }

    pub fn query_direct(&self, offset: u64, find_next: bool) -> Option<KernelDirectAllocation> {
        self.direct_memory
            .query(offset, find_next)
            .map(|allocation| KernelDirectAllocation {
                start: allocation.start,
                end: allocation.end,
                memory_type: allocation.memory_type,
            })
    }

    pub fn release_direct(&mut self, start: u64, size: u64) -> Result<bool, KernelMemoryError> {
        // Physical allocation ownership and virtual alias lifetime are separate in the kernel
        // ABI. Unity releases the direct range first and then explicitly munmaps its virtual view;
        // tearing the alias down here makes that legal sequence fault before sceKernelMunmap.
        let alias_ranges = self.direct_alias_ranges(start, size)?;
        let released = self.direct_memory.release(start, size)?;
        if released {
            // A virtual alias may legally outlive physical ownership. It must not, however, keep
            // a GPU resident alive after the same direct-memory pages are returned to the guest
            // allocator. Otherwise a delayed texture readback can publish the old allocation
            // through that alias after the physical range has been reused for ordinary objects.
            //
            // Retiring watches preserves every virtual mapping while making resident/sample
            // caches observe `Replaced` before they can read back into the recycled backing.
            for (address, byte_count) in alias_ranges {
                CpuWriteWatch::retire_overlapping(address, byte_count);
            }
        }
        Ok(released)
    }

    fn direct_alias_ranges(
        &self,
        start: u64,
        size: u64,
    ) -> Result<Vec<(u64, u64)>, KernelMemoryError> {
        let end = start
            .checked_add(size)
            .ok_or(KernelMemoryError::DirectMemoryOverflow { start, size })?;
        let backing_id = self.direct_memory.backing.id();
        Ok(self
            .virtual_memory
            .vmas()
            .into_iter()
            .filter(|mapping| {
                mapping.state == GuestVmaState::Shared && mapping.backing_id == Some(backing_id)
            })
            .filter_map(|mapping| {
                let mapping_physical_end = mapping.backing_offset.checked_add(mapping.size)?;
                let intersection_start = mapping.backing_offset.max(start);
                let intersection_end = mapping_physical_end.min(end);
                (intersection_start < intersection_end).then(|| {
                    (
                        mapping.start.get() + (intersection_start - mapping.backing_offset),
                        intersection_end - intersection_start,
                    )
                })
            })
            .collect())
    }

    #[cfg(test)]
    fn map_direct(
        &mut self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
        raw_protection: i32,
        direct_start: u64,
    ) -> Result<GuestAddress, KernelMemoryError> {
        self.map_direct_with_options(
            placement,
            size,
            alignment,
            KernelDirectMapOptions {
                raw_protection,
                direct_start,
                memory_type: None,
                coalescing: GuestMapCoalescing::Allowed,
            },
        )
    }

    pub fn map_direct_with_options(
        &mut self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
        options: KernelDirectMapOptions,
    ) -> Result<GuestAddress, KernelMemoryError> {
        let KernelDirectMapOptions {
            raw_protection,
            direct_start,
            memory_type,
            coalescing,
        } = options;
        let segments = self.direct_memory.segments(direct_start, size)?;
        let protection = translate_protection(raw_protection);
        let mapped = self.virtual_memory.map_shared_with_coalescing(
            placement,
            size,
            alignment,
            protection,
            &self.direct_memory.backing,
            GuestSharedMapOptions {
                backing_offset: direct_start,
                coalescing,
            },
        )?;
        for segment in segments {
            let virtual_start = mapped
                .get()
                .checked_add(segment.start - direct_start)
                .ok_or_else(|| KernelMemoryError::DirectMemoryOverflow {
                    start: mapped.get(),
                    size: segment.start - direct_start,
                })?;
            if let Err(error) = self.virtual_memory.set_metadata(
                GuestAddress::new(virtual_start),
                segment.end - segment.start,
                encode_metadata(raw_protection, memory_type.unwrap_or(segment.memory_type)),
            ) {
                let _ = self.virtual_memory.unmap(mapped, size);
                return Err(error.into());
            }
        }
        Ok(mapped)
    }

    pub fn protect(
        &mut self,
        address: GuestAddress,
        size: u64,
        raw_protection: i32,
    ) -> Result<u64, KernelMemoryError> {
        self.protect_with_type(address, size, raw_protection, None)
    }

    /// Changes protection and, when `memory_type` is supplied, the guest-visible memory type.
    ///
    /// `sceKernelMtypeprotect` retypes a range in the same operation that reprotects it, while
    /// plain `sceKernelMprotect` must preserve each affected mapping's existing type. Returns the
    /// number of previously reserved bytes the operation committed.
    pub fn protect_with_type(
        &mut self,
        address: GuestAddress,
        size: u64,
        raw_protection: i32,
        memory_type: Option<i32>,
    ) -> Result<u64, KernelMemoryError> {
        let committed_bytes =
            self.virtual_memory
                .protect(address, size, translate_protection(raw_protection))?;
        let end = address.get().checked_add(size).ok_or_else(|| {
            KernelMemoryError::DirectMemoryOverflow {
                start: address.get(),
                size,
            }
        })?;
        let affected: Vec<_> = self
            .virtual_memory
            .vmas()
            .into_iter()
            .filter(|vma| {
                vma.start.get() < end && vma.start.get().saturating_add(vma.size) > address.get()
            })
            .collect();
        for vma in affected {
            self.virtual_memory.set_metadata(
                vma.start,
                vma.size,
                encode_metadata(
                    raw_protection,
                    memory_type.unwrap_or_else(|| decode_memory_type(vma.metadata)),
                ) | (vma.metadata & FLEXIBLE_METADATA_BIT),
            )?;
        }
        Ok(committed_bytes)
    }

    pub fn unmap(&mut self, address: GuestAddress, size: u64) -> Result<(), KernelMemoryError> {
        self.virtual_memory.unmap(address, size)?;
        Ok(())
    }

    pub fn query(&self, address: GuestAddress, find_next: bool) -> Option<KernelVirtualMapping> {
        let exact = self
            .virtual_memory
            .query(address)
            .ok()
            .filter(|vma| vma.state != GuestVmaState::Free);
        let info = exact.or_else(|| {
            if !find_next {
                return None;
            }
            self.virtual_memory
                .vmas()
                .into_iter()
                .find(|vma| vma.start >= address)
        })?;
        Some(mapping_from_vma(info))
    }
}

fn mapping_from_vma(info: GuestVmaInfo) -> KernelVirtualMapping {
    // Committed memory reports through the same guest-visible categories as before it carried a
    // backing: flexible mappings stay flexible, and commit-on-protect ranges keep reporting as
    // the reservation the guest sees. Only direct-memory views expose a physical offset.
    let kind = if info.state == GuestVmaState::Shared {
        KernelVirtualMappingKind::Direct
    } else if info.metadata & FLEXIBLE_METADATA_BIT != 0 {
        KernelVirtualMappingKind::Flexible
    } else {
        KernelVirtualMappingKind::Reserved
    };
    KernelVirtualMapping {
        address: info.start,
        size: info.size,
        direct_start: if info.state == GuestVmaState::Shared {
            info.backing_offset
        } else {
            0
        },
        protection: decode_protection(info.metadata),
        memory_type: decode_memory_type(info.metadata),
        kind,
    }
}

const fn translate_protection(raw: i32) -> GuestProtection {
    let mut protection = GuestProtection::NONE;
    // CPU_READ_WRITE is encoded as bit 1, not as CPU_READ | CPU_WRITE. Treat it as readable as
    // well as writable so protection 2 matches the console ABI.
    if raw & (0x01 | 0x02 | 0x10) != 0 {
        protection = protection.union(GuestProtection::READ);
    }
    if raw & (0x02 | 0x20) != 0 {
        protection = protection.union(GuestProtection::WRITE);
    }
    if raw & 0x04 != 0 {
        protection = protection.union(GuestProtection::EXECUTE);
    }
    protection
}

const FLEXIBLE_METADATA_BIT: u64 = 1 << 63;

fn encode_metadata(protection: i32, memory_type: i32) -> u64 {
    u64::from(protection.cast_unsigned()) | (u64::from(memory_type.cast_unsigned()) << 32)
}

fn decode_protection(metadata: u64) -> i32 {
    u32::try_from(metadata & u64::from(u32::MAX))
        .expect("masked protection metadata fits u32")
        .cast_signed()
}

fn decode_memory_type(metadata: u64) -> i32 {
    u32::try_from((metadata >> 32) & 0x7fff_ffff)
        .expect("shifted memory-type metadata fits u32")
        .cast_signed()
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment.checked_sub(1)?)
        .map(|sum| sum & !(alignment - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page(address: GuestAddress, pages: u64) -> GuestAddress {
        GuestAddress::new(address.get() + pages * GUEST_PAGE_SIZE)
    }

    #[test]
    fn physical_allocator_is_deterministic_and_release_splits_ranges() {
        let mut memory = KernelMemory::new().unwrap();
        let first = memory
            .allocate_direct(
                0,
                8 * GUEST_PAGE_SIZE,
                3 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                7,
            )
            .unwrap();
        let second = memory
            .allocate_direct(0, 8 * GUEST_PAGE_SIZE, GUEST_PAGE_SIZE, GUEST_PAGE_SIZE, 9)
            .unwrap();
        assert_eq!(first, 0);
        assert_eq!(second, 3 * GUEST_PAGE_SIZE);
        assert!(
            memory
                .release_direct(GUEST_PAGE_SIZE, GUEST_PAGE_SIZE)
                .unwrap()
        );
        let reused = memory
            .allocate_direct(0, 8 * GUEST_PAGE_SIZE, GUEST_PAGE_SIZE, GUEST_PAGE_SIZE, 11)
            .unwrap();
        assert_eq!(reused, GUEST_PAGE_SIZE);
    }

    #[test]
    fn available_direct_reports_the_largest_free_aligned_span() {
        let mut memory = KernelMemory::new().unwrap();
        // An untouched pool offers all of itself.
        assert_eq!(
            memory.available_direct(0, DIRECT_MEMORY_SIZE, GUEST_PAGE_SIZE),
            Some((0, DIRECT_MEMORY_SIZE))
        );
        let carved = memory
            .allocate_direct(
                2 * GUEST_PAGE_SIZE,
                3 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                12,
            )
            .unwrap();
        assert_eq!(carved, 2 * GUEST_PAGE_SIZE);
        // Of the two holes the allocation leaves, the larger one is reported.
        assert_eq!(
            memory.available_direct(0, 8 * GUEST_PAGE_SIZE, GUEST_PAGE_SIZE),
            Some((3 * GUEST_PAGE_SIZE, 5 * GUEST_PAGE_SIZE))
        );
        // The search window clips a hole at both ends.
        assert_eq!(
            memory.available_direct(0, 2 * GUEST_PAGE_SIZE, GUEST_PAGE_SIZE),
            Some((0, 2 * GUEST_PAGE_SIZE))
        );
        // Alignment moves a hole's base up and shortens what it can offer, which can change
        // which hole wins.
        assert_eq!(
            memory.available_direct(0, 8 * GUEST_PAGE_SIZE, 4 * GUEST_PAGE_SIZE),
            Some((4 * GUEST_PAGE_SIZE, 4 * GUEST_PAGE_SIZE))
        );
        // A window covered entirely by one allocation has nothing to offer.
        assert_eq!(
            memory.available_direct(2 * GUEST_PAGE_SIZE, 3 * GUEST_PAGE_SIZE, GUEST_PAGE_SIZE),
            None
        );
    }

    #[test]
    fn an_available_span_is_allocatable_in_full() {
        let mut memory = KernelMemory::new().unwrap();
        // The shape the failing title uses: reserve its tracking overhead at the top of the pool,
        // ask what is left, then pre-allocate exactly that.
        const OVERHEAD: u64 = 0x26_0000;
        const ALIGNMENT: u64 = 0x1_0000;
        let tracking = memory
            .allocate_direct(
                DIRECT_MEMORY_SIZE - OVERHEAD,
                DIRECT_MEMORY_SIZE,
                OVERHEAD,
                ALIGNMENT,
                12,
            )
            .unwrap();
        assert_eq!(tracking, DIRECT_MEMORY_SIZE - OVERHEAD);
        let (start, size) = memory
            .available_direct(0, DIRECT_MEMORY_SIZE, ALIGNMENT)
            .unwrap();
        assert_eq!((start, size), (0, DIRECT_MEMORY_SIZE - OVERHEAD));
        // A reported span the allocator then refuses would put the guest's own accounting into
        // exactly the negative-size state this query exists to prevent.
        assert_eq!(
            memory
                .allocate_direct(start, start + size, size, ALIGNMENT, 12)
                .unwrap(),
            start
        );
    }

    #[test]
    fn flexible_usage_follows_live_mappings_in_both_directions() {
        let mut memory = KernelMemory::new().unwrap();
        assert_eq!(memory.flexible_usage(), 0);
        let first = memory
            .map_flexible(
                GuestMapPlacement::Anywhere,
                4 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                0x03,
            )
            .unwrap();
        assert_eq!(memory.flexible_usage(), 4 * GUEST_PAGE_SIZE);
        memory
            .map_flexible(
                GuestMapPlacement::Anywhere,
                2 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                0x03,
            )
            .unwrap();
        assert_eq!(memory.flexible_usage(), 6 * GUEST_PAGE_SIZE);
        // Usage is derived from the mappings themselves, so a partial unmap is accounted for
        // without any separate counter to keep in step.
        memory.unmap(first, 2 * GUEST_PAGE_SIZE).unwrap();
        assert_eq!(memory.flexible_usage(), 4 * GUEST_PAGE_SIZE);
        // Direct memory is a different pool and must not be counted against the flexible budget.
        memory
            .allocate_direct(0, DIRECT_MEMORY_SIZE, GUEST_PAGE_SIZE, GUEST_PAGE_SIZE, 12)
            .unwrap();
        memory
            .map_direct(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                0x33,
                0,
            )
            .unwrap();
        assert_eq!(memory.flexible_usage(), 4 * GUEST_PAGE_SIZE);
    }

    #[test]
    fn aliases_share_bytes_and_query_preserves_ps5_metadata() {
        let mut memory = KernelMemory::new().unwrap();
        memory
            .allocate_direct(0, DIRECT_MEMORY_SIZE, GUEST_PAGE_SIZE, GUEST_PAGE_SIZE, 12)
            .unwrap();
        let first = memory
            .map_direct(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                0x33,
                0,
            )
            .unwrap();
        let second = memory
            .map_direct(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                0x33,
                0,
            )
            .unwrap();
        // SAFETY: Both addresses are live read/write aliases returned by the subsystem.
        unsafe {
            (first.get() as *mut u8).write_volatile(0x5a);
            assert_eq!((second.get() as *const u8).read_volatile(), 0x5a);
        }
        assert_eq!(
            memory.query(second, false),
            Some(KernelVirtualMapping {
                address: second,
                size: GUEST_PAGE_SIZE,
                direct_start: 0,
                protection: 0x33,
                memory_type: 12,
                kind: KernelVirtualMappingKind::Direct,
            })
        );
    }

    #[test]
    fn flexible_mapping_is_read_write_and_preserves_query_metadata() {
        let mut memory = KernelMemory::new().unwrap();
        let mapped = memory
            .map_flexible(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                0x02,
            )
            .unwrap();

        // SAFETY: CPU_READ_WRITE protection makes the returned anonymous mapping accessible.
        unsafe {
            (mapped.get() as *mut u8).write_volatile(0x5a);
            assert_eq!((mapped.get() as *const u8).read_volatile(), 0x5a);
        }
        assert!(
            memory
                .readable_ranges()
                .iter()
                .any(|&(start, end)| start <= mapped.get() && end > mapped.get())
        );
        assert_eq!(
            memory.query(mapped, false),
            Some(KernelVirtualMapping {
                address: mapped,
                size: GUEST_PAGE_SIZE,
                direct_start: 0,
                protection: 0x02,
                memory_type: 0,
                kind: KernelVirtualMappingKind::Flexible,
            })
        );
    }

    #[test]
    fn reserved_virtual_range_reports_no_protection_and_is_queryable() {
        let mut memory = KernelMemory::new().unwrap();
        let reserved = memory
            .reserve_virtual(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
            )
            .unwrap();
        let mapping = memory.query(reserved, false).unwrap();
        assert_eq!(mapping.address, reserved);
        assert_eq!(mapping.size, GUEST_PAGE_SIZE);
        assert_eq!(mapping.protection, 0);
        assert_eq!(mapping.kind, KernelVirtualMappingKind::Reserved);
    }

    #[test]
    fn protecting_a_reserved_virtual_range_commits_read_write_pages() {
        let mut memory = KernelMemory::new().unwrap();
        let reserved = memory
            .reserve_virtual(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
            )
            .unwrap();

        memory.protect(reserved, GUEST_PAGE_SIZE, 0xf2).unwrap();
        // SAFETY: protection 0xf2 includes the guest CPU read/write mode.
        unsafe {
            (reserved.get() as *mut u8).write_volatile(0x5a);
            assert_eq!((reserved.get() as *const u8).read_volatile(), 0x5a);
        }

        assert_eq!(
            memory.query(reserved, false),
            Some(KernelVirtualMapping {
                address: reserved,
                size: GUEST_PAGE_SIZE,
                direct_start: 0,
                protection: 0xf2,
                memory_type: 0,
                kind: KernelVirtualMappingKind::Reserved,
            })
        );
        assert_eq!(
            memory.readable_ranges(),
            vec![(reserved.get(), reserved.get() + GUEST_PAGE_SIZE)]
        );
    }

    /// Unity's PS5 allocator never maps direct memory for most allocations: it reserves virtual
    /// ranges and commits them with `sceKernelMprotect` carrying CPU, GPU, and AMPR protection
    /// bits. Texture descriptors then point straight into that committed memory, so the GPU
    /// residency layer must resolve it exactly like a direct-memory mapping.
    #[test]
    fn mprotect_committed_reservation_is_gpu_resolvable() {
        let mut memory = KernelMemory::new().unwrap();
        let reserved = memory
            .reserve_virtual(
                GuestMapPlacement::Anywhere,
                32 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
            )
            .unwrap();
        memory
            .protect(reserved, 32 * GUEST_PAGE_SIZE, 0xf2)
            .unwrap();

        // A texture that starts inside the committed block, exactly like Superliminal's
        // rejected draw at reserve+0x1000.
        let texture = page(reserved, 1);
        let texture_bytes = 30 * GUEST_PAGE_SIZE;
        assert!(
            remu_memory::guest_shared_mapping_identity(texture.get(), texture_bytes).is_some(),
            "committed pages must expose a physical backing identity"
        );
        let segments =
            remu_memory::guest_shared_backing_alias_segments(texture.get(), texture_bytes)
                .expect("committed pages enumerate their coherent alias");
        assert_eq!(
            segments
                .iter()
                .map(|segment| segment.byte_count)
                .sum::<u64>(),
            texture_bytes
        );
        let mut probe = vec![0_u8; 16];
        assert!(remu_memory::read_guest_shared_backing(
            texture.get(),
            &mut probe
        ));
    }

    #[test]
    fn flexible_mapping_is_gpu_resolvable() {
        let mut memory = KernelMemory::new().unwrap();
        let mapped = memory
            .map_flexible(
                GuestMapPlacement::Anywhere,
                2 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                0x02,
            )
            .unwrap();
        assert!(
            remu_memory::guest_shared_mapping_identity(mapped.get(), 2 * GUEST_PAGE_SIZE).is_some(),
            "flexible memory is committed guest memory and must resolve for the GPU"
        );
        // The flexible query ABI observed by guests must be unchanged by the backing model.
        assert_eq!(
            memory.query(mapped, false).map(|mapping| mapping.kind),
            Some(KernelVirtualMappingKind::Flexible)
        );
        assert_eq!(
            memory
                .query(mapped, false)
                .map(|mapping| mapping.direct_start),
            Some(0)
        );
    }

    /// A reservation exists so the guest can choose where a later mapping lands. Unreal's PS5
    /// allocator reserves one arena up front and then commits direct memory into it with
    /// `MAP_FIXED`; committing the reservation eagerly turns that second call into an overlap
    /// rejection and deadlocks the guest allocator.
    #[test]
    fn fixed_direct_mapping_commits_inside_a_reserved_range() {
        let mut memory = KernelMemory::new().unwrap();
        let reserved = memory
            .reserve_virtual(
                GuestMapPlacement::Anywhere,
                4 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
            )
            .unwrap();
        memory
            .allocate_direct(
                0,
                DIRECT_MEMORY_SIZE,
                2 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                12,
            )
            .unwrap();
        let mapped = memory
            .map_direct(
                GuestMapPlacement::Fixed(reserved),
                2 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                0x33,
                0,
            )
            .expect("a fixed mapping may commit reserved pages");
        assert_eq!(mapped, reserved);
        let mapping = memory.query(mapped, false).unwrap();
        assert_eq!(mapping.kind, KernelVirtualMappingKind::Direct);
        assert_eq!(mapping.size, 2 * GUEST_PAGE_SIZE);
        // The untouched tail of the reservation stays reserved and independently mappable.
        let tail = page(reserved, 2);
        assert_eq!(
            memory.query(tail, false).map(|mapping| mapping.kind),
            Some(KernelVirtualMappingKind::Reserved)
        );
    }

    #[test]
    fn direct_release_preserves_virtual_aliases_until_explicit_unmap() {
        let mut memory = KernelMemory::new().unwrap();
        memory
            .allocate_direct(
                0,
                DIRECT_MEMORY_SIZE,
                3 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                12,
            )
            .unwrap();
        let first = memory
            .map_direct(
                GuestMapPlacement::Anywhere,
                3 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                0x33,
                0,
            )
            .unwrap();
        let second = memory
            .map_direct(
                GuestMapPlacement::Anywhere,
                3 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                0x33,
                0,
            )
            .unwrap();
        // SAFETY: both direct aliases remain mapped until the explicit unmap below.
        let first_watch = unsafe {
            CpuWriteWatch::new(page(first, 1).get(), GUEST_PAGE_SIZE).expect("first alias watch")
        };
        // SAFETY: see first_watch.
        let second_watch = unsafe {
            CpuWriteWatch::new(page(second, 1).get(), GUEST_PAGE_SIZE).expect("second alias watch")
        };
        // SAFETY: the adjacent direct page also remains mapped throughout the assertion.
        let unreleased_watch = unsafe {
            CpuWriteWatch::new(page(first, 2).get(), GUEST_PAGE_SIZE)
                .expect("unreleased-page watch")
        };

        assert!(
            memory
                .release_direct(GUEST_PAGE_SIZE, GUEST_PAGE_SIZE)
                .unwrap()
        );
        assert_eq!(
            first_watch.retirement(),
            Some(remu_memory::CpuWriteWatchRetirement::Replaced)
        );
        assert_eq!(
            second_watch.retirement(),
            Some(remu_memory::CpuWriteWatchRetirement::Replaced)
        );
        assert_eq!(unreleased_watch.retirement(), None);
        for alias in [first, second] {
            assert!(memory.query(alias, false).is_some());
            assert!(memory.query(page(alias, 1), false).is_some());
            assert!(memory.query(page(alias, 2), false).is_some());
        }
        memory.unmap(page(first, 1), GUEST_PAGE_SIZE).unwrap();
        assert!(memory.query(page(first, 1), false).is_none());
        assert!(memory.query(page(second, 1), false).is_some());
        assert!(
            !memory
                .release_direct(GUEST_PAGE_SIZE, GUEST_PAGE_SIZE)
                .unwrap()
        );
        assert!(matches!(
            memory.map_direct(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                0x33,
                GUEST_PAGE_SIZE,
            ),
            Err(KernelMemoryError::DirectMemoryNotAllocated { .. })
        ));
    }

    #[test]
    fn mapping_across_physical_allocations_retains_each_memory_type() {
        let mut memory = KernelMemory::new().unwrap();
        memory
            .allocate_direct(0, DIRECT_MEMORY_SIZE, GUEST_PAGE_SIZE, GUEST_PAGE_SIZE, 4)
            .unwrap();
        memory
            .allocate_direct(
                GUEST_PAGE_SIZE,
                DIRECT_MEMORY_SIZE,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                9,
            )
            .unwrap();
        let mapped = memory
            .map_direct(
                GuestMapPlacement::Anywhere,
                2 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                0x33,
                0,
            )
            .unwrap();
        assert_eq!(memory.query(mapped, false).unwrap().memory_type, 4);
        assert_eq!(memory.query(page(mapped, 1), false).unwrap().memory_type, 9);
    }

    #[test]
    fn independent_adjacent_direct_maps_retain_virtual_query_boundaries() {
        let mut memory = KernelMemory::new().unwrap();
        memory
            .allocate_direct(0, DIRECT_MEMORY_SIZE, GUEST_PAGE_SIZE, GUEST_PAGE_SIZE, 12)
            .unwrap();
        memory
            .allocate_direct(
                GUEST_PAGE_SIZE,
                DIRECT_MEMORY_SIZE,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                12,
            )
            .unwrap();
        let first = memory
            .map_direct_with_options(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                KernelDirectMapOptions {
                    raw_protection: 0x33,
                    direct_start: 0,
                    memory_type: None,
                    coalescing: GuestMapCoalescing::Disallowed,
                },
            )
            .unwrap();
        let second = memory
            .map_direct_with_options(
                GuestMapPlacement::Fixed(page(first, 1)),
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                KernelDirectMapOptions {
                    raw_protection: 0x33,
                    direct_start: GUEST_PAGE_SIZE,
                    memory_type: None,
                    coalescing: GuestMapCoalescing::Disallowed,
                },
            )
            .unwrap();

        assert_eq!(second, page(first, 1));
        assert_eq!(memory.query(first, false).unwrap().size, GUEST_PAGE_SIZE);
        let second_query = memory.query(second, false).unwrap();
        assert_eq!(second_query.address, second);
        assert_eq!(second_query.size, GUEST_PAGE_SIZE);
        assert_eq!(second_query.direct_start, GUEST_PAGE_SIZE);
        assert_eq!(
            memory.readable_ranges(),
            vec![(first.get(), second.get() + GUEST_PAGE_SIZE)]
        );
    }

    #[test]
    fn virtual_query_prefers_containment_then_the_smallest_later_mapping() {
        let mut memory = KernelMemory::new().unwrap();
        memory
            .allocate_direct(
                0,
                DIRECT_MEMORY_SIZE,
                2 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                12,
            )
            .unwrap();
        let base = memory.virtual_base();
        let first = page(base, 2);
        let second = page(base, 5);
        for (address, direct_start) in [(first, 0), (second, GUEST_PAGE_SIZE)] {
            memory
                .map_direct(
                    GuestMapPlacement::Fixed(address),
                    GUEST_PAGE_SIZE,
                    GUEST_PAGE_SIZE,
                    0x33,
                    direct_start,
                )
                .unwrap();
        }

        assert_eq!(memory.query(first, true).unwrap().address, first);
        assert!(memory.query(page(base, 3), false).is_none());
        assert_eq!(memory.query(page(base, 3), true).unwrap().address, second);
    }

    #[test]
    fn fixed_and_hinted_placements_are_exact_inside_the_owned_space() {
        let mut memory = KernelMemory::new().unwrap();
        memory
            .allocate_direct(
                0,
                DIRECT_MEMORY_SIZE,
                3 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                1,
            )
            .unwrap();
        let base = memory.virtual_base();
        let fixed = page(base, 4);
        assert_eq!(
            memory
                .map_direct(
                    GuestMapPlacement::Fixed(fixed),
                    GUEST_PAGE_SIZE,
                    GUEST_PAGE_SIZE,
                    3,
                    0,
                )
                .unwrap(),
            fixed
        );
        let hinted = page(base, 8);
        assert_eq!(
            memory
                .map_direct(
                    GuestMapPlacement::Hint(hinted),
                    GUEST_PAGE_SIZE,
                    GUEST_PAGE_SIZE,
                    3,
                    GUEST_PAGE_SIZE,
                )
                .unwrap(),
            hinted
        );
        assert!(matches!(
            memory.map_direct(
                GuestMapPlacement::Fixed(fixed),
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                3,
                2 * GUEST_PAGE_SIZE,
            ),
            Err(KernelMemoryError::AddressSpace(
                AddressSpaceError::Overlap { .. }
            ))
        ));
    }

    #[test]
    fn protect_and_unmap_use_strict_split_vma_semantics() {
        let mut memory = KernelMemory::new().unwrap();
        memory
            .allocate_direct(
                0,
                DIRECT_MEMORY_SIZE,
                3 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                4,
            )
            .unwrap();
        let mapped = memory
            .map_direct(
                GuestMapPlacement::Anywhere,
                3 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                0x33,
                0,
            )
            .unwrap();
        memory
            .protect(page(mapped, 1), GUEST_PAGE_SIZE, 0x11)
            .unwrap();
        assert_eq!(
            memory.query(page(mapped, 1), false).unwrap().protection,
            0x11
        );
        memory.unmap(page(mapped, 1), GUEST_PAGE_SIZE).unwrap();
        assert!(memory.query(page(mapped, 1), false).is_none());
        assert!(matches!(
            memory.unmap(mapped, 3 * GUEST_PAGE_SIZE),
            Err(KernelMemoryError::AddressSpace(
                AddressSpaceError::NotMapped { .. }
            ))
        ));
        assert!(memory.query(mapped, false).is_some());
        assert!(memory.query(page(mapped, 2), false).is_some());
    }
}
