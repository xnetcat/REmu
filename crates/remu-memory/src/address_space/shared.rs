// SPDX-License-Identifier: GPL-2.0-or-later

//! Process-global registry mapping live shared guest ranges back to their physical backing.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use super::GuestPhysicalBacking;

#[derive(Clone, Debug)]
struct SharedMappingRecord {
    end: u64,
    backing: GuestPhysicalBacking,
    backing_offset: u64,
}

/// Stable physical-backing identity for a byte range in a shared guest mapping.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GuestSharedMappingIdentity {
    pub backing_id: u64,
    pub backing_offset: u64,
}

/// One live CPU mapping that covers a contiguous portion of a GPU-physical backing range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuestSharedBackingSegment {
    pub address: u64,
    pub byte_count: u64,
}

static SHARED_MAPPINGS: OnceLock<Mutex<BTreeMap<u64, SharedMappingRecord>>> = OnceLock::new();

fn shared_mappings() -> &'static Mutex<BTreeMap<u64, SharedMappingRecord>> {
    SHARED_MAPPINGS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn resolve_shared_backing_range(
    mappings: &BTreeMap<u64, SharedMappingRecord>,
    address: u64,
    byte_count: u64,
) -> Option<(GuestPhysicalBacking, u64, u64)> {
    if byte_count == 0 {
        return None;
    }
    let (&first_start, first) = mappings.range(..=address).next_back()?;
    if address >= first.end {
        return None;
    }
    let backing_id = first.backing.id();
    let physical_start = first.backing_offset.checked_add(address - first_start)?;
    let physical_end = physical_start.checked_add(byte_count)?;
    if physical_end > first.backing.size() {
        return None;
    }
    let mut physical_cursor = physical_start;
    while physical_cursor < physical_end {
        let covered_end = mappings
            .iter()
            .filter(|(_, record)| record.backing.id() == backing_id)
            .filter_map(|(&start, record)| {
                let mapping_length = record.end.checked_sub(start)?;
                let backing_end = record.backing_offset.checked_add(mapping_length)?;
                (record.backing_offset <= physical_cursor && physical_cursor < backing_end)
                    .then_some(backing_end)
            })
            .max()?
            .min(physical_end);
        if covered_end == physical_cursor {
            return None;
        }
        physical_cursor = covered_end;
    }
    Some((first.backing.clone(), physical_start, physical_end))
}

/// Resolves a live shared guest range to its backing object and byte offset.
///
/// Adjacent virtual mappings are accepted when they name contiguous slices of the same backing.
/// Anonymous and guard mappings intentionally return `None` because they cannot alias another VA.
#[must_use]
pub fn guest_shared_mapping_identity(
    address: u64,
    byte_count: u64,
) -> Option<GuestSharedMappingIdentity> {
    if byte_count == 0 {
        return None;
    }
    let end = address.checked_add(byte_count)?;
    let mappings = shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (&start, first) = mappings.range(..=address).next_back()?;
    if address >= first.end {
        return None;
    }
    let identity = GuestSharedMappingIdentity {
        backing_id: first.backing.id(),
        backing_offset: first.backing_offset.checked_add(address - start)?,
    };
    let mut cursor = first.end.min(end);
    let mut expected_backing_offset = first
        .backing_offset
        .checked_add(first.end.checked_sub(start)?)?;
    while cursor < end {
        let next = mappings.get(&cursor)?;
        if next.backing.id() != identity.backing_id
            || next.backing_offset != expected_backing_offset
        {
            return None;
        }
        expected_backing_offset = expected_backing_offset.checked_add(next.end - cursor)?;
        cursor = next.end.min(end);
    }
    Some(identity)
}

/// Resolves a GPU resource range through the shared physical backing behind its first VA.
///
/// Prospero can map adjacent direct-memory pages at unrelated CPU virtual addresses while the GPU
/// resource remains physically contiguous. This returns live CPU aliases which, together, cover
/// that exact physical byte interval. Every byte must still have a live mapping; holes in physical
/// backing coverage are rejected.
#[must_use]
pub fn guest_shared_backing_segments(
    address: u64,
    byte_count: u64,
) -> Option<Vec<GuestSharedBackingSegment>> {
    let mappings = shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (backing, physical_start, physical_end) =
        resolve_shared_backing_range(&mappings, address, byte_count)?;
    let backing_id = backing.id();

    let mut physical_cursor = physical_start;
    let mut segments = Vec::new();
    while physical_cursor < physical_end {
        let candidate = mappings
            .iter()
            .filter(|(_, record)| record.backing.id() == backing_id)
            .filter_map(|(&start, record)| {
                let length = record.end.checked_sub(start)?;
                let backing_end = record.backing_offset.checked_add(length)?;
                (record.backing_offset <= physical_cursor && physical_cursor < backing_end)
                    .then_some((start, record, backing_end))
            })
            // Prefer the alias that covers the most remaining bytes. This avoids needlessly
            // splitting a range when several coherent aliases name the same physical pages.
            .max_by_key(|(_, _, backing_end)| *backing_end)?;
        let (mapping_start, record, mapping_backing_end) = candidate;
        let covered_end = mapping_backing_end.min(physical_end);
        let guest_address = mapping_start.checked_add(physical_cursor - record.backing_offset)?;
        let covered_bytes = covered_end.checked_sub(physical_cursor)?;
        if covered_bytes == 0 {
            return None;
        }
        segments.push(GuestSharedBackingSegment {
            address: guest_address,
            byte_count: covered_bytes,
        });
        physical_cursor = covered_end;
    }
    Some(segments)
}

/// Returns every live CPU alias intersecting a GPU-physical backing range.
///
/// This is intentionally broader than [`guest_shared_backing_segments`]: write watching must
/// observe stores through every coherent alias, not just one representative that covers each
/// physical byte.
#[must_use]
pub fn guest_shared_backing_alias_segments(
    address: u64,
    byte_count: u64,
) -> Option<Vec<GuestSharedBackingSegment>> {
    let mappings = shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (backing, physical_start, physical_end) =
        resolve_shared_backing_range(&mappings, address, byte_count)?;
    let backing_id = backing.id();
    let mut aliases = mappings
        .iter()
        .filter(|(_, record)| record.backing.id() == backing_id)
        .filter_map(|(&mapping_start, record)| {
            let mapping_length = record.end.checked_sub(mapping_start)?;
            let mapping_backing_end = record.backing_offset.checked_add(mapping_length)?;
            let intersection_start = record.backing_offset.max(physical_start);
            let intersection_end = mapping_backing_end.min(physical_end);
            (intersection_start < intersection_end).then(|| GuestSharedBackingSegment {
                address: mapping_start + (intersection_start - record.backing_offset),
                byte_count: intersection_end - intersection_start,
            })
        })
        .collect::<Vec<_>>();
    aliases.sort_unstable_by_key(|segment| (segment.address, segment.byte_count));
    aliases.dedup();
    (!aliases.is_empty()).then_some(aliases)
}

/// Copies a GPU-physical resource range out of its shared direct-memory backing.
///
/// Unlike dereferencing the resource VA, this remains valid when consecutive backing pages are
/// exposed through disjoint CPU aliases. Live-mapping coverage is validated before the transfer.
pub fn read_guest_shared_backing(address: u64, destination: &mut [u8]) -> bool {
    let Ok(byte_count) = u64::try_from(destination.len()) else {
        return false;
    };
    let Some((backing, offset, _)) = ({
        let mappings = shared_mappings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        resolve_shared_backing_range(&mappings, address, byte_count)
    }) else {
        return false;
    };
    backing.read_exact_at(offset, destination)
}

/// Copies bytes into a GPU-physical resource range in shared direct-memory backing.
///
/// CPU aliases observe the write coherently because they map this same shared backing object.
pub fn write_guest_shared_backing(address: u64, source: &[u8]) -> bool {
    let Ok(byte_count) = u64::try_from(source.len()) else {
        return false;
    };
    let Some((backing, offset, _)) = ({
        let mappings = shared_mappings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        resolve_shared_backing_range(&mappings, address, byte_count)
    }) else {
        return false;
    };
    backing.write_all_at(offset, source)
}

pub(super) fn unregister_shared_mapping_range(address: u64, byte_count: u64) {
    let Some(end) = address.checked_add(byte_count) else {
        return;
    };
    let mut mappings = shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let overlaps = mappings
        .range(..end)
        .filter(|(start, record)| record.end > address && **start < end)
        .map(|(&start, record)| (start, record.clone()))
        .collect::<Vec<_>>();
    for (start, record) in overlaps {
        mappings.remove(&start);
        if start < address {
            mappings.insert(
                start,
                SharedMappingRecord {
                    end: address,
                    ..record.clone()
                },
            );
        }
        if record.end > end {
            mappings.insert(
                end,
                SharedMappingRecord {
                    end: record.end,
                    backing: record.backing,
                    backing_offset: record.backing_offset + (end - start),
                },
            );
        }
    }
}

pub(super) fn register_shared_mapping_range(
    address: u64,
    byte_count: u64,
    backing: &GuestPhysicalBacking,
    backing_offset: u64,
) {
    let Some(end) = address.checked_add(byte_count) else {
        return;
    };
    unregister_shared_mapping_range(address, byte_count);
    shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(
            address,
            SharedMappingRecord {
                end,
                backing: backing.clone(),
                backing_offset,
            },
        );
}
