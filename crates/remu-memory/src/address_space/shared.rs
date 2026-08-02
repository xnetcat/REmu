// SPDX-License-Identifier: GPL-2.0-or-later

//! Process-global registry mapping live shared guest ranges back to their physical backing.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use super::{GuestPhysicalBacking, GuestSharedBackingView, align_up, host_page_size, host_size};

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

/// A live guest range resolved to one page-aligned physical-backing view.
///
/// The range carries a backing clone but does not map it until [`Self::map_view`] is called. This
/// lets GPU caches look up an existing identity view without paying for a redundant mmap.
#[derive(Clone, Debug)]
pub struct GuestSharedBackingRange {
    backing: GuestPhysicalBacking,
    view_identity: GuestSharedMappingIdentity,
    view_byte_count: usize,
    resource_offset: u64,
}

impl GuestSharedBackingRange {
    /// Identity of the first byte in the page-aligned backing view.
    #[must_use]
    pub const fn view_identity(&self) -> GuestSharedMappingIdentity {
        self.view_identity
    }

    #[must_use]
    pub const fn view_byte_count(&self) -> usize {
        self.view_byte_count
    }

    /// Offset of the requested guest byte range inside the page-aligned view.
    #[must_use]
    pub const fn resource_offset(&self) -> u64 {
        self.resource_offset
    }

    /// Maps the resolved backing interval read-write for a coherent CPU/GPU shared view.
    #[must_use]
    pub fn map_view(&self) -> Option<GuestSharedBackingView> {
        self.backing
            .map_shared_view(self.view_identity.backing_offset, self.view_byte_count)
    }
}

static SHARED_MAPPINGS: OnceLock<Mutex<BTreeMap<u64, SharedMappingRecord>>> = OnceLock::new();

fn shared_mappings() -> &'static Mutex<BTreeMap<u64, SharedMappingRecord>> {
    SHARED_MAPPINGS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

static SHARED_MAPPING_MUTATION: OnceLock<Mutex<()>> = OnceLock::new();

static SHARED_MAPPING_EPOCH: AtomicU64 = AtomicU64::new(0);

pub(super) struct GuestSharedMappingMutation {
    _guard: MutexGuard<'static, ()>,
}

impl Drop for GuestSharedMappingMutation {
    fn drop(&mut self) {
        let previous = SHARED_MAPPING_EPOCH.fetch_add(1, Ordering::Release);
        debug_assert_eq!(
            previous & 1,
            1,
            "shared-mapping mutation epoch must be odd while changing"
        );
    }
}

/// Serializes a guest mapping transition and makes incomplete alias snapshots detectable.
pub(super) fn begin_shared_mapping_mutation() -> GuestSharedMappingMutation {
    let guard = SHARED_MAPPING_MUTATION
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let previous = SHARED_MAPPING_EPOCH.fetch_add(1, Ordering::AcqRel);
    debug_assert_eq!(
        previous & 1,
        0,
        "shared-mapping mutation epoch must be even before changing"
    );
    GuestSharedMappingMutation { _guard: guard }
}

/// Returns an even value while the shared-mapping alias set is stable and an odd value otherwise.
#[must_use]
pub fn guest_shared_mapping_epoch() -> u64 {
    SHARED_MAPPING_EPOCH.load(Ordering::Acquire)
}

/// Returns every current VA intersecting a supplied physical-backing interval.
pub(super) fn existing_shared_backing_alias_segments(
    backing: &GuestPhysicalBacking,
    backing_offset: u64,
    byte_count: u64,
) -> Vec<GuestSharedBackingSegment> {
    let Some(backing_end) = backing_offset.checked_add(byte_count) else {
        return Vec::new();
    };
    let mut aliases = shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .filter(|(_, record)| record.backing.id() == backing.id())
        .filter_map(|(&mapping_start, record)| {
            let mapping_length = record.end.checked_sub(mapping_start)?;
            let mapping_backing_end = record.backing_offset.checked_add(mapping_length)?;
            let intersection_start = record.backing_offset.max(backing_offset);
            let intersection_end = mapping_backing_end.min(backing_end);
            (intersection_start < intersection_end).then(|| GuestSharedBackingSegment {
                address: mapping_start + (intersection_start - record.backing_offset),
                byte_count: intersection_end - intersection_start,
            })
        })
        .collect::<Vec<_>>();
    aliases.sort_unstable_by_key(|segment| (segment.address, segment.byte_count));
    aliases.dedup();
    aliases
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
    let first_mapping_length = first.end.checked_sub(first_start)?;
    let first_mapping_physical_end = first.backing_offset.checked_add(first_mapping_length)?;
    if physical_end <= first_mapping_physical_end {
        return Some((first.backing.clone(), physical_start, physical_end));
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

/// Resolves a GPU-physical guest range to its backing object identity and byte offset.
///
/// Unlike [`guest_shared_mapping_identity`], this accepts a physically contiguous range exposed
/// through disjoint virtual mappings. The full physical interval must still have live CPU mapping
/// coverage.
#[must_use]
pub fn guest_shared_backing_identity(
    address: u64,
    byte_count: u64,
) -> Option<GuestSharedMappingIdentity> {
    let mappings = shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    resolve_shared_backing_range(&mappings, address, byte_count).map(
        |(backing, physical_start, _)| GuestSharedMappingIdentity {
            backing_id: backing.id(),
            backing_offset: physical_start,
        },
    )
}

/// Resolves an exact live guest range to a lazily mapped, page-aligned backing view.
///
/// Unlike a no-copy view over the guest VA, this remains contiguous when consecutive physical
/// pages are exposed through unrelated CPU aliases. Only the requested physical bytes require
/// live guest mapping coverage; page padding is mapped directly from the same backing object.
#[must_use]
pub fn guest_shared_backing_range(
    address: u64,
    byte_count: u64,
) -> Option<GuestSharedBackingRange> {
    let mappings = shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (backing, physical_start, physical_end) =
        resolve_shared_backing_range(&mappings, address, byte_count)?;
    drop(mappings);

    let page_size = host_page_size().ok()?;
    let view_start = physical_start - physical_start % page_size;
    let resource_offset = physical_start.checked_sub(view_start)?;
    let requested_bytes = physical_end.checked_sub(physical_start)?;
    let view_byte_count = resource_offset
        .checked_add(requested_bytes)
        .and_then(|length| align_up(length, page_size).ok())
        .and_then(|length| host_size(length).ok())?;
    view_start
        .checked_add(u64::try_from(view_byte_count).ok()?)
        .filter(|end| *end <= backing.size())?;

    Some(GuestSharedBackingRange {
        view_identity: GuestSharedMappingIdentity {
            backing_id: backing.id(),
            backing_offset: view_start,
        },
        backing,
        view_byte_count,
        resource_offset,
    })
}

fn ranges_overlap(left: u64, left_end: u64, right: u64, right_end: u64) -> bool {
    left < right_end && right < left_end
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

/// Returns the written virtual range plus every live VA naming any shared-backing portion of it.
///
/// Unlike [`guest_shared_backing_alias_segments`], this accepts a range spanning independent
/// shared mappings and anonymous gaps. The original VA remains present for gaps, while each shared
/// subrange expands to all coherent physical aliases.
pub(crate) fn guest_write_alias_segments(
    address: u64,
    byte_count: u64,
) -> Vec<GuestSharedBackingSegment> {
    let Some(end) = address.checked_add(byte_count) else {
        return Vec::new();
    };
    if byte_count == 0 {
        return Vec::new();
    }
    let mappings = shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let written_shared_intervals = mappings
        .range(..end)
        .filter_map(|(&mapping_start, record)| {
            let intersection_start = mapping_start.max(address);
            let intersection_end = record.end.min(end);
            if intersection_start >= intersection_end {
                return None;
            }
            let physical_start = record
                .backing_offset
                .checked_add(intersection_start.checked_sub(mapping_start)?)?;
            let physical_end =
                physical_start.checked_add(intersection_end.checked_sub(intersection_start)?)?;
            Some((record.backing.id(), physical_start, physical_end))
        })
        .collect::<Vec<_>>();

    let mut aliases = vec![GuestSharedBackingSegment {
        address,
        byte_count,
    }];
    for (backing_id, physical_start, physical_end) in written_shared_intervals {
        aliases.extend(
            mappings
                .iter()
                .filter(|(_, record)| record.backing.id() == backing_id)
                .filter_map(|(&mapping_start, record)| {
                    let mapping_length = record.end.checked_sub(mapping_start)?;
                    let mapping_physical_end = record.backing_offset.checked_add(mapping_length)?;
                    let intersection_start = record.backing_offset.max(physical_start);
                    let intersection_end = mapping_physical_end.min(physical_end);
                    (intersection_start < intersection_end).then(|| GuestSharedBackingSegment {
                        address: mapping_start + (intersection_start - record.backing_offset),
                        byte_count: intersection_end - intersection_start,
                    })
                }),
        );
    }
    aliases.sort_unstable_by_key(|segment| (segment.address, segment.byte_count));
    aliases.dedup();
    aliases
}

/// Reports whether two guest ranges name overlapping bytes of one shared physical backing.
///
/// The common case compares physical offsets without allocating an alias list. If the right-hand
/// range is only partially mapped, the fallback retains the alias-intersection behavior used by
/// GPU hazard tracking.
#[must_use]
pub fn guest_shared_backing_ranges_overlap(
    left: u64,
    left_byte_count: u64,
    right: u64,
    right_byte_count: u64,
) -> bool {
    if left_byte_count == 0 || right_byte_count == 0 {
        return false;
    }
    let mappings = shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some((left_backing, left_physical_start, left_physical_end)) =
        resolve_shared_backing_range(&mappings, left, left_byte_count)
    else {
        return false;
    };
    if let Some((right_backing, right_physical_start, right_physical_end)) =
        resolve_shared_backing_range(&mappings, right, right_byte_count)
    {
        return left_backing.id() == right_backing.id()
            && ranges_overlap(
                left_physical_start,
                left_physical_end,
                right_physical_start,
                right_physical_end,
            );
    }

    let Some(right_virtual_end) = right.checked_add(right_byte_count) else {
        return false;
    };
    mappings
        .iter()
        .filter(|(_, record)| record.backing.id() == left_backing.id())
        .any(|(&mapping_start, record)| {
            let Some(mapping_length) = record.end.checked_sub(mapping_start) else {
                return false;
            };
            let Some(mapping_physical_end) = record.backing_offset.checked_add(mapping_length)
            else {
                return false;
            };
            let intersection_start = record.backing_offset.max(left_physical_start);
            let intersection_end = mapping_physical_end.min(left_physical_end);
            if intersection_start >= intersection_end {
                return false;
            }
            let Some(alias_start) =
                mapping_start.checked_add(intersection_start - record.backing_offset)
            else {
                return false;
            };
            let Some(alias_end) = alias_start.checked_add(intersection_end - intersection_start)
            else {
                return false;
            };
            ranges_overlap(alias_start, alias_end, right, right_virtual_end)
        })
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

/// Copies bytes only while the guest range still names the expected physical backing.
///
/// Mapping validation and the backing write share the registry lock, so an unmap/remap cannot
/// replace the guest VA between the identity check and the transfer.
pub fn write_guest_shared_backing_if_identity(
    address: u64,
    expected: GuestSharedMappingIdentity,
    source: &[u8],
) -> bool {
    let Ok(byte_count) = u64::try_from(source.len()) else {
        return false;
    };
    let mappings = shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some((backing, offset, _)) = resolve_shared_backing_range(&mappings, address, byte_count)
    else {
        return false;
    };
    if backing.id() != expected.backing_id || offset != expected.backing_offset {
        return false;
    }
    backing.write_all_at(offset, source)
}

/// Copies bytes into a captured physical backing identity without requiring the original VA.
///
/// Prospero recycles direct-memory VAs while GPU work is still in flight. Metal writebacks that
/// recorded the producer identity must still land on those physical pages so any remaining alias
/// (and later remaps back onto that interval) observe the GPU results.
///
/// Two conditions bound that licence, because a recorded identity is not by itself a claim on
/// memory. The backing must be physically keyed: in a virtually keyed object, `backing_offset` is
/// just `guest_va - base`, so publishing "by producer identity" would rewrite the very address
/// whose liveness check has already failed, straight into whatever the guest allocator has since
/// put there. And live mappings must cover the whole of `[backing_offset, backing_offset + len)`;
/// naming the backing id somewhere else in the object proves nothing about these bytes. Coverage
/// and the transfer share the registry lock so an unmap cannot intervene between them.
///
/// Refusals are reported: a silently dropped writeback is indistinguishable from a correct one.
pub fn write_guest_shared_backing_at_identity(
    identity: GuestSharedMappingIdentity,
    source: &[u8],
) -> bool {
    let Ok(byte_count) = u64::try_from(source.len()) else {
        return false;
    };
    // An empty range proves no coverage, so it cannot be published; `write_guest_shared_backing`
    // rejects zero-length writes for the same reason.
    if byte_count == 0 {
        return false;
    }
    let Some(end) = identity.backing_offset.checked_add(byte_count) else {
        return false;
    };
    let mappings = shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(backing) = mappings
        .values()
        .find(|record| record.backing.id() == identity.backing_id)
        .map(|record| record.backing.clone())
    else {
        trace_refused_identity_write(&mappings, identity, byte_count, "backing has no live mapping");
        return false;
    };
    if backing.is_virtually_keyed() {
        trace_refused_identity_write(&mappings, identity, byte_count, "virtually keyed backing");
        return false;
    }
    if end > backing.size() {
        trace_refused_identity_write(&mappings, identity, byte_count, "range exceeds backing");
        return false;
    }
    if !identity_range_is_live(&mappings, identity.backing_id, identity.backing_offset, end) {
        trace_refused_identity_write(
            &mappings,
            identity,
            byte_count,
            "no live mapping covers the range",
        );
        return false;
    }
    backing.write_all_at(identity.backing_offset, source)
}

/// Reports whether live records cover every byte of a physical interval in one backing.
fn identity_range_is_live(
    mappings: &BTreeMap<u64, SharedMappingRecord>,
    backing_id: u64,
    start: u64,
    end: u64,
) -> bool {
    let mut cursor = start;
    while cursor < end {
        let covered_end = mappings
            .iter()
            .filter(|(_, record)| record.backing.id() == backing_id)
            .filter_map(|(&mapping_start, record)| {
                let mapping_length = record.end.checked_sub(mapping_start)?;
                let mapping_backing_end = record.backing_offset.checked_add(mapping_length)?;
                (record.backing_offset <= cursor && cursor < mapping_backing_end)
                    .then_some(mapping_backing_end)
            })
            .max();
        let Some(covered_end) = covered_end.filter(|covered_end| *covered_end > cursor) else {
            return false;
        };
        cursor = covered_end.min(end);
    }
    true
}

static REFUSED_IDENTITY_WRITES: AtomicU64 = AtomicU64::new(0);

fn trace_refused_identity_write(
    mappings: &BTreeMap<u64, SharedMappingRecord>,
    identity: GuestSharedMappingIdentity,
    byte_count: u64,
    reason: &str,
) {
    let sequence = REFUSED_IDENTITY_WRITES.fetch_add(1, Ordering::Relaxed) + 1;
    if sequence > 16 && !sequence.is_power_of_two() {
        return;
    }
    let alias = mappings
        .iter()
        .filter(|(_, record)| record.backing.id() == identity.backing_id)
        .find_map(|(&mapping_start, record)| {
            let mapping_length = record.end.checked_sub(mapping_start)?;
            let mapping_backing_end = record.backing_offset.checked_add(mapping_length)?;
            (record.backing_offset <= identity.backing_offset
                && identity.backing_offset < mapping_backing_end)
                .then(|| mapping_start + (identity.backing_offset - record.backing_offset))
        })
        .map_or_else(|| "none".to_owned(), |address| format!("{address:#x}"));
    eprintln!(
        "[REMu][VM] refused producer-identity writeback sequence={sequence} reason={reason} address={alias} bytes={byte_count:#x} backing={} offset={:#x}",
        identity.backing_id, identity.backing_offset,
    );
}

fn unregister_shared_mapping_range_locked(
    mappings: &mut BTreeMap<u64, SharedMappingRecord>,
    address: u64,
    byte_count: u64,
) {
    let Some(end) = address.checked_add(byte_count) else {
        return;
    };
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

pub(super) fn unregister_shared_mapping_range(address: u64, byte_count: u64) {
    let mut mappings = shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    unregister_shared_mapping_range_locked(&mut mappings, address, byte_count);
}

fn records_are_physically_contiguous(
    left_start: u64,
    left: &SharedMappingRecord,
    right_start: u64,
    right: &SharedMappingRecord,
) -> bool {
    left.end == right_start
        && left.backing.id() == right.backing.id()
        && left
            .backing_offset
            .checked_add(left.end - left_start)
            .is_some_and(|left_backing_end| left_backing_end == right.backing_offset)
}

fn coalesce_shared_mapping_record(mappings: &mut BTreeMap<u64, SharedMappingRecord>, address: u64) {
    let mut merged_start = address;
    let Some(current) = mappings.get(&merged_start).cloned() else {
        return;
    };
    if let Some((&previous_start, previous)) = mappings.range(..merged_start).next_back()
        && records_are_physically_contiguous(previous_start, previous, merged_start, &current)
    {
        let previous = previous.clone();
        mappings.remove(&previous_start);
        mappings.remove(&merged_start);
        mappings.insert(
            previous_start,
            SharedMappingRecord {
                end: current.end,
                backing: previous.backing,
                backing_offset: previous.backing_offset,
            },
        );
        merged_start = previous_start;
    }

    while let Some(current) = mappings.get(&merged_start).cloned() {
        let Some(next) = mappings.get(&current.end).cloned() else {
            break;
        };
        if !records_are_physically_contiguous(merged_start, &current, current.end, &next) {
            break;
        }
        mappings.remove(&merged_start);
        mappings.remove(&current.end);
        mappings.insert(
            merged_start,
            SharedMappingRecord {
                end: next.end,
                backing: current.backing,
                backing_offset: current.backing_offset,
            },
        );
    }
}

pub(super) fn register_shared_mapping_range(
    address: u64,
    byte_count: u64,
    backing: &GuestPhysicalBacking,
    backing_offset: u64,
) {
    if byte_count == 0 {
        return;
    }
    let Some(end) = address.checked_add(byte_count) else {
        return;
    };
    let mut mappings = shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    unregister_shared_mapping_range_locked(&mut mappings, address, byte_count);
    mappings.insert(
        address,
        SharedMappingRecord {
            end,
            backing: backing.clone(),
            backing_offset,
        },
    );
    // The VMA layer retains mapping-call boundaries when the guest requests them. The physical
    // registry does not need those virtual bookkeeping seams: texture snapshots and CPU write
    // watches care only that adjacent addresses name adjacent bytes of one backing. Coalescing
    // here turns separately committed pages back into one watchable/copyable physical run.
    coalesce_shared_mapping_record(&mut mappings, address);
}
