// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::BTreeMap;

use remu_core::GuestAddress;
#[cfg(test)]
use remu_memory::GUEST_PAGE_SIZE;
use remu_memory::{
    AddressSpaceError, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking, GuestProtection,
    GuestVmaInfo, GuestVmaState,
};
use thiserror::Error;

pub(super) const DIRECT_MEMORY_SIZE: u64 = 16 * 1024 * 1024 * 1024;
const VIRTUAL_MEMORY_SIZE: u64 = 64 * 1024 * 1024 * 1024;

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
        let end = start
            .checked_add(size)
            .ok_or(KernelMemoryError::DirectMemoryOverflow { start, size })?;
        let owner = self
            .allocations
            .range(..=start)
            .next_back()
            .map(|(_, allocation)| *allocation)
            .filter(|allocation| start >= allocation.start && end <= allocation.end);
        let Some(owner) = owner else {
            return Ok(false);
        };
        self.allocations.remove(&owner.start);
        if owner.start < start {
            self.insert(owner.start, start, owner.memory_type);
        }
        if end < owner.end {
            self.insert(end, owner.end, owner.memory_type);
        }
        Ok(true)
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

    pub fn release_direct(&mut self, start: u64, size: u64) -> Result<bool, KernelMemoryError> {
        self.direct_memory.release(start, size)
    }

    pub fn map_direct(
        &mut self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
        raw_protection: i32,
        direct_start: u64,
    ) -> Result<GuestAddress, KernelMemoryError> {
        self.map_direct_with_memory_type(
            placement,
            size,
            alignment,
            raw_protection,
            direct_start,
            None,
        )
    }

    pub fn map_direct_with_memory_type(
        &mut self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
        raw_protection: i32,
        direct_start: u64,
        memory_type: Option<i32>,
    ) -> Result<GuestAddress, KernelMemoryError> {
        let segments = self.direct_memory.segments(direct_start, size)?;
        let protection = translate_protection(raw_protection);
        let mapped = self.virtual_memory.map_shared(
            placement,
            size,
            alignment,
            protection,
            &self.direct_memory.backing,
            direct_start,
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
    ) -> Result<(), KernelMemoryError> {
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
                encode_metadata(raw_protection, decode_memory_type(vma.metadata)),
            )?;
        }
        Ok(())
    }

    pub fn unmap(&mut self, address: GuestAddress, size: u64) -> Result<(), KernelMemoryError> {
        self.virtual_memory.unmap(address, size)?;
        Ok(())
    }

    pub fn query(&self, address: GuestAddress, find_next: bool) -> Option<KernelVirtualMapping> {
        let exact = self.virtual_memory.query(address).ok().and_then(|vma| {
            (vma.state == GuestVmaState::Shared
                && vma.backing_id == Some(self.direct_memory.backing.id()))
            .then_some(vma)
        });
        let info = exact.or_else(|| {
            if !find_next {
                return None;
            }
            self.virtual_memory.vmas().into_iter().find(|vma| {
                vma.start >= address
                    && vma.state == GuestVmaState::Shared
                    && vma.backing_id == Some(self.direct_memory.backing.id())
            })
        })?;
        Some(mapping_from_vma(info))
    }
}

fn mapping_from_vma(info: GuestVmaInfo) -> KernelVirtualMapping {
    KernelVirtualMapping {
        address: info.start,
        size: info.size,
        direct_start: info.backing_offset,
        protection: decode_protection(info.metadata),
        memory_type: decode_memory_type(info.metadata),
    }
}

const fn translate_protection(raw: i32) -> GuestProtection {
    let mut protection = GuestProtection::NONE;
    if raw & (0x01 | 0x10) != 0 {
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

fn encode_metadata(protection: i32, memory_type: i32) -> u64 {
    u64::from(protection.cast_unsigned()) | (u64::from(memory_type.cast_unsigned()) << 32)
}

fn decode_protection(metadata: u64) -> i32 {
    u32::try_from(metadata & u64::from(u32::MAX))
        .expect("masked protection metadata fits u32")
        .cast_signed()
}

fn decode_memory_type(metadata: u64) -> i32 {
    u32::try_from(metadata >> 32)
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
            })
        );
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
