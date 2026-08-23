// SPDX-License-Identifier: GPL-2.0-or-later

//! Process-global registry mapping live shared guest ranges back to their physical backing.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, RwLock};

use super::{GuestPhysicalBacking, GuestSharedBackingView, align_up, host_page_size, host_size};

#[derive(Clone, Debug)]
struct SharedMappingRecord {
    end: u64,
    backing: GuestPhysicalBacking,
    backing_offset: u64,
    generation: u64,
}

static NEXT_SHARED_MAPPING_RECORD_GENERATION: AtomicU64 = AtomicU64::new(1);

fn next_shared_mapping_record_generation() -> u64 {
    NEXT_SHARED_MAPPING_RECORD_GENERATION
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |generation| {
            generation.checked_add(1)
        })
        .expect("shared-mapping record generation exhausted")
}

/// Stable physical-backing identity for a byte range in a shared guest mapping.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GuestSharedMappingIdentity {
    pub backing_id: u64,
    pub backing_offset: u64,
}

/// Opaque proof naming the exact registry records used to resolve a guest range.
///
/// Record generations are process-unique. Unmapping and mapping the same virtual range to the
/// same numeric backing offset therefore produces a different token and cannot pass an ABA check.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct GuestSharedMappingRangeToken {
    record_generations: Box<[u64]>,
}

impl GuestSharedMappingRangeToken {
    #[must_use]
    pub fn record_count(&self) -> usize {
        self.record_generations.len()
    }
}

/// A full-range physical resolution stamped with the mapping epoch and ABA token.
#[derive(Clone, Debug)]
pub struct GuestSharedRangeResolution {
    guest_address: u64,
    identity: GuestSharedMappingIdentity,
    byte_count: u64,
    mapping_epoch: u64,
    mapping_token: GuestSharedMappingRangeToken,
    backing_range: GuestSharedBackingRange,
}

impl GuestSharedRangeResolution {
    #[must_use]
    pub const fn identity(&self) -> GuestSharedMappingIdentity {
        self.identity
    }

    #[must_use]
    pub const fn guest_address(&self) -> u64 {
        self.guest_address
    }

    #[must_use]
    pub const fn byte_count(&self) -> u64 {
        self.byte_count
    }

    #[must_use]
    pub const fn mapping_epoch(&self) -> u64 {
        self.mapping_epoch
    }

    #[must_use]
    pub const fn mapping_token(&self) -> &GuestSharedMappingRangeToken {
        &self.mapping_token
    }

    #[must_use]
    pub fn is_virtually_keyed(&self) -> bool {
        self.backing_range.is_virtually_keyed()
    }

    #[must_use]
    pub fn backing_range(&self) -> &GuestSharedBackingRange {
        &self.backing_range
    }

    /// Cheap validity hint. False requires full re-resolution; true proves no mapping transition
    /// began since this range was resolved.
    #[must_use]
    pub fn epoch_is_current(&self) -> bool {
        guest_shared_mapping_epoch() == self.mapping_epoch
    }

    /// Re-resolves this binding after an epoch drift and enforces the backing's ABA policy.
    #[must_use]
    pub fn revalidate(&self) -> GuestSharedRangeRevalidation {
        if self.epoch_is_current() {
            return GuestSharedRangeRevalidation::Current(self.clone());
        }
        let Some(current) = guest_shared_resolve_stamped_range(self.guest_address, self.byte_count)
        else {
            return GuestSharedRangeRevalidation::Unmapped;
        };
        let same_physical = current.identity == self.identity;
        let same_token = current.mapping_token == self.mapping_token;
        if same_physical && (!self.is_virtually_keyed() || same_token) {
            GuestSharedRangeRevalidation::Current(current)
        } else {
            GuestSharedRangeRevalidation::Replaced
        }
    }
}

/// Result of re-resolving a stamped guest range after its epoch changed.
#[derive(Clone, Debug)]
pub enum GuestSharedRangeRevalidation {
    Current(GuestSharedRangeResolution),
    Replaced,
    Unmapped,
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

    /// Whether offsets in this backing name guest virtual placement rather than durable
    /// physical bytes.
    ///
    /// Async GPU consumers must not retain a virtually keyed range across allocator reuse: the
    /// same `(backing, offset)` can then describe a different committed allocation. Physically
    /// keyed direct-memory ranges remain stable while a retained backing clone exists.
    #[must_use]
    pub fn is_virtually_keyed(&self) -> bool {
        self.backing.is_virtually_keyed()
    }

    /// Maps the resolved backing interval read-write for a coherent CPU/GPU shared view.
    #[must_use]
    pub fn map_view(&self) -> Option<GuestSharedBackingView> {
        self.backing
            .map_shared_view(self.view_identity.backing_offset, self.view_byte_count)
    }
}

/// The live shared-mapping records, indexed both by guest address and by physical backing.
///
/// The `by_address` map is the source of truth and supports the ordered lookups resolution needs
/// (`range(..=address)`, `get(&cursor)`). `by_backing` is a secondary index: commit-heavy titles
/// register hundreds of records against one direct-memory object, and every resolution scan filters
/// by backing id. Scanning the whole address map for one backing was `O(all live records)` per
/// watch; `by_backing` bounds those scans to the records that actually name the backing (audit N2).
///
/// The two indices are kept consistent by routing every mutation through [`Self::insert`] and
/// [`Self::remove`]; `Deref` exposes only the immutable `by_address` map so no caller can insert or
/// remove behind the index's back.
#[derive(Default)]
struct SharedRegistry {
    by_address: BTreeMap<u64, SharedMappingRecord>,
    by_backing: BTreeMap<u64, BTreeSet<u64>>,
}

impl std::ops::Deref for SharedRegistry {
    type Target = BTreeMap<u64, SharedMappingRecord>;

    fn deref(&self) -> &Self::Target {
        &self.by_address
    }
}

impl SharedRegistry {
    fn insert(&mut self, address: u64, mut record: SharedMappingRecord) {
        record.generation = next_shared_mapping_record_generation();
        let backing_id = record.backing.id();
        if let Some(previous) = self.by_address.insert(address, record)
            && previous.backing.id() != backing_id
        {
            self.forget_backing_address(previous.backing.id(), address);
        }
        self.by_backing
            .entry(backing_id)
            .or_default()
            .insert(address);
    }

    fn remove(&mut self, address: u64) -> Option<SharedMappingRecord> {
        let record = self.by_address.remove(&address)?;
        self.forget_backing_address(record.backing.id(), address);
        Some(record)
    }

    fn forget_backing_address(&mut self, backing_id: u64, address: u64) {
        if let Some(addresses) = self.by_backing.get_mut(&backing_id) {
            addresses.remove(&address);
            if addresses.is_empty() {
                self.by_backing.remove(&backing_id);
            }
        }
    }

    /// Live records naming `backing_id`, yielded `(mapping_start, record)` in address order.
    ///
    /// This is the secondary-index equivalent of `self.iter().filter(|(_, r)| r.backing.id() ==
    /// backing_id)`; the `by_backing` set holds exactly those addresses, so the filter is gone.
    fn backing_records(
        &self,
        backing_id: u64,
    ) -> impl Iterator<Item = (u64, &SharedMappingRecord)> {
        self.by_backing
            .get(&backing_id)
            .into_iter()
            .flatten()
            .filter_map(move |&address| {
                self.by_address
                    .get(&address)
                    .map(|record| (address, record))
            })
    }

    /// A backing clone for any live record naming `backing_id`, or `None` if none is mapped.
    fn any_backing(&self, backing_id: u64) -> Option<GuestPhysicalBacking> {
        self.backing_records(backing_id)
            .next()
            .map(|(_, record)| record.backing.clone())
    }

    #[cfg(test)]
    fn backing_record_count(&self, backing_id: u64) -> usize {
        self.by_backing
            .get(&backing_id)
            .map_or(0, std::collections::BTreeSet::len)
    }
}

static SHARED_MAPPINGS: OnceLock<RwLock<SharedRegistry>> = OnceLock::new();

fn shared_mappings() -> &'static RwLock<SharedRegistry> {
    SHARED_MAPPINGS.get_or_init(|| RwLock::new(SharedRegistry::default()))
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
    let mappings = shared_mappings()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut aliases = mappings
        .backing_records(backing.id())
        .filter_map(|(mapping_start, record)| {
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
    mappings: &SharedRegistry,
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
            .backing_records(backing_id)
            .filter_map(|(start, record)| {
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

fn resolved_range_token(
    mappings: &SharedRegistry,
    address: u64,
    physical_start: u64,
    physical_end: u64,
    backing_id: u64,
) -> Option<GuestSharedMappingRangeToken> {
    let (_, first) = mappings.range(..=address).next_back()?;
    let mut generations = vec![first.generation];
    let mut cursor = physical_start;
    while cursor < physical_end {
        let (_, record, covered_end) = mappings
            .backing_records(backing_id)
            .filter_map(|(mapping_start, record)| {
                let mapping_length = record.end.checked_sub(mapping_start)?;
                let backing_end = record.backing_offset.checked_add(mapping_length)?;
                (record.backing_offset <= cursor && cursor < backing_end).then_some((
                    mapping_start,
                    record,
                    backing_end.min(physical_end),
                ))
            })
            .max_by_key(|(mapping_start, _, covered_end)| (*covered_end, *mapping_start))?;
        if covered_end <= cursor {
            return None;
        }
        if generations.last().copied() != Some(record.generation)
            && !generations.contains(&record.generation)
        {
            generations.push(record.generation);
        }
        cursor = covered_end;
    }
    Some(GuestSharedMappingRangeToken {
        record_generations: generations.into_boxed_slice(),
    })
}

fn retained_backing_range(
    backing: GuestPhysicalBacking,
    physical_start: u64,
    physical_end: u64,
) -> Option<GuestSharedBackingRange> {
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

/// Resolves a complete guest range under an even mapping epoch and returns its retained backing,
/// physical identity, and mapping-record ABA token as one coherent snapshot.
#[must_use]
pub fn guest_shared_resolve_stamped_range(
    address: u64,
    byte_count: u64,
) -> Option<GuestSharedRangeResolution> {
    loop {
        let before = guest_shared_mapping_epoch();
        if before & 1 != 0 {
            std::hint::spin_loop();
            continue;
        }
        let mappings = shared_mappings()
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let resolved = resolve_shared_backing_range(&mappings, address, byte_count).and_then(
            |(backing, physical_start, physical_end)| {
                let token = resolved_range_token(
                    &mappings,
                    address,
                    physical_start,
                    physical_end,
                    backing.id(),
                )?;
                let backing_range = retained_backing_range(backing, physical_start, physical_end)?;
                Some(GuestSharedRangeResolution {
                    guest_address: address,
                    identity: GuestSharedMappingIdentity {
                        backing_id: backing_range.view_identity().backing_id,
                        backing_offset: physical_start,
                    },
                    byte_count,
                    mapping_epoch: before,
                    mapping_token: token,
                    backing_range,
                })
            },
        );
        drop(mappings);
        let after = guest_shared_mapping_epoch();
        if before == after && after & 1 == 0 {
            return resolved;
        }
    }
}

/// Resolves a live shared guest range to its backing object and byte offset.
///
/// Adjacent virtual mappings are accepted when they name contiguous slices of the same backing.
/// Reserved/guard mappings intentionally return `None` because they have no physical backing.
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
        .read()
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
    let started = std::time::Instant::now();
    let mappings = shared_mappings()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let identity = resolve_shared_backing_range(&mappings, address, byte_count).map(
        |(backing, physical_start, _)| GuestSharedMappingIdentity {
            backing_id: backing.id(),
            backing_offset: physical_start,
        },
    );
    drop(mappings);
    IDENTITY_RESOLVE_CALLS.fetch_add(1, Ordering::Relaxed);
    IDENTITY_RESOLVE_NANOS.fetch_add(
        u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
    identity
}

static IDENTITY_RESOLVE_CALLS: AtomicU64 = AtomicU64::new(0);
static IDENTITY_RESOLVE_NANOS: AtomicU64 = AtomicU64::new(0);

/// Takes and resets the process-wide call count and total wall time spent resolving shared
/// backing identities. The per-draw translation cost attribution reads this each perf window;
/// lock acquisition is included deliberately since contention is part of the cost under study.
#[must_use]
pub fn take_guest_shared_identity_statistics() -> (u64, u64) {
    (
        IDENTITY_RESOLVE_CALLS.swap(0, Ordering::Relaxed),
        IDENTITY_RESOLVE_NANOS.swap(0, Ordering::Relaxed),
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
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (backing, physical_start, physical_end) =
        resolve_shared_backing_range(&mappings, address, byte_count)?;
    drop(mappings);
    retained_backing_range(backing, physical_start, physical_end)
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
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (backing, physical_start, physical_end) =
        resolve_shared_backing_range(&mappings, address, byte_count)?;
    let backing_id = backing.id();

    let mut physical_cursor = physical_start;
    let mut segments = Vec::new();
    while physical_cursor < physical_end {
        let candidate = mappings
            .backing_records(backing_id)
            .filter_map(|(start, record)| {
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
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (backing, physical_start, physical_end) =
        resolve_shared_backing_range(&mappings, address, byte_count)?;
    let backing_id = backing.id();
    let mut aliases = mappings
        .backing_records(backing_id)
        .filter_map(|(mapping_start, record)| {
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
        .read()
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
        aliases.extend(mappings.backing_records(backing_id).filter_map(
            |(mapping_start, record)| {
                let mapping_length = record.end.checked_sub(mapping_start)?;
                let mapping_physical_end = record.backing_offset.checked_add(mapping_length)?;
                let intersection_start = record.backing_offset.max(physical_start);
                let intersection_end = mapping_physical_end.min(physical_end);
                (intersection_start < intersection_end).then(|| GuestSharedBackingSegment {
                    address: mapping_start + (intersection_start - record.backing_offset),
                    byte_count: intersection_end - intersection_start,
                })
            },
        ));
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
/// Tests many candidate ranges against one fixed range under a single registry read.
///
/// [`guest_shared_backing_ranges_overlap`] takes the registry lock and resolves BOTH sides on every
/// call, so scanning N candidates costs N lock acquisitions and 2N resolutions. The presenter runs
/// exactly that scan per binding against every queued batch's write set, which makes the cost grow
/// with submission backlog — deep enough and the presenter spends its budget in these scans instead
/// of retiring work, starving the guest's fences.
///
/// The argument order matches `guest_shared_backing_ranges_overlap(candidate, .., fixed, ..)` and
/// must stay that way: the relation is NOT symmetric. When only one side resolves to a backing, the
/// resolved side is walked against the other side's VIRTUAL range, so swapping them changes the
/// answer. Only the fixed side's resolution and the lock can be hoisted; each candidate still
/// resolves individually.
pub fn guest_shared_backing_any_range_overlaps(
    candidates: impl IntoIterator<Item = (u64, u64)>,
    fixed: u64,
    fixed_byte_count: u64,
) -> bool {
    if fixed_byte_count == 0 {
        return false;
    }
    let mappings = shared_mappings()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixed_resolved = resolve_shared_backing_range(&mappings, fixed, fixed_byte_count);
    let fixed_virtual_end = fixed.checked_add(fixed_byte_count);
    candidates
        .into_iter()
        .any(|(candidate, candidate_byte_count)| {
            if candidate_byte_count == 0 {
                return false;
            }
            let Some((candidate_backing, candidate_start, candidate_end)) =
                resolve_shared_backing_range(&mappings, candidate, candidate_byte_count)
            else {
                return false;
            };
            if let Some((fixed_backing, fixed_start, fixed_end)) = &fixed_resolved {
                return candidate_backing.id() == fixed_backing.id()
                    && ranges_overlap(candidate_start, candidate_end, *fixed_start, *fixed_end);
            }
            let Some(fixed_virtual_end) = fixed_virtual_end else {
                return false;
            };
            mappings
                .backing_records(candidate_backing.id())
                .any(|(mapping_start, record)| {
                    let Some(mapping_length) = record.end.checked_sub(mapping_start) else {
                        return false;
                    };
                    let Some(mapping_physical_end) =
                        record.backing_offset.checked_add(mapping_length)
                    else {
                        return false;
                    };
                    let intersection_start = record.backing_offset.max(candidate_start);
                    let intersection_end = mapping_physical_end.min(candidate_end);
                    if intersection_start >= intersection_end {
                        return false;
                    }
                    let Some(alias_start) =
                        mapping_start.checked_add(intersection_start - record.backing_offset)
                    else {
                        return false;
                    };
                    let Some(alias_end) =
                        alias_start.checked_add(intersection_end - intersection_start)
                    else {
                        return false;
                    };
                    ranges_overlap(alias_start, alias_end, fixed, fixed_virtual_end)
                })
        })
}

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
        .read()
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
        .backing_records(left_backing.id())
        .any(|(mapping_start, record)| {
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
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        resolve_shared_backing_range(&mappings, address, byte_count)
    }) else {
        return false;
    };
    backing.read_exact_at(offset, destination)
}

/// Copies disjoint byte segments into one guest range's shared backing.
///
/// A texture writeback authors only the bytes the GPU owns — for a linear image, the visible
/// row bytes, never the pitch padding between rows. Publishing through one contiguous write
/// required reading the whole span first and writing foreign padding bytes back, and a guest
/// store landing between that read and write was silently reverted: allocator canaries in a
/// pooled heap were the observed casualty. Segments are offsets into `address`'s range; the
/// range is resolved once and every segment is written under that single resolution.
pub fn write_guest_shared_backing_scattered(
    address: u64,
    byte_count: u64,
    segments: &[(u64, &[u8])],
) -> bool {
    let Some((backing, base_offset, _)) = ({
        let mappings = shared_mappings()
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        resolve_shared_backing_range(&mappings, address, byte_count)
    }) else {
        return false;
    };
    for (segment_offset, bytes) in segments {
        let in_range = u64::try_from(bytes.len())
            .ok()
            .and_then(|len| segment_offset.checked_add(len))
            .is_some_and(|end| end <= byte_count);
        if !in_range {
            return false;
        }
        let Some(offset) = base_offset.checked_add(*segment_offset) else {
            return false;
        };
        if !backing.write_all_at(offset, bytes) {
            return false;
        }
    }
    true
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
            .read()
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
        .read()
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
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(backing) = mappings.any_backing(identity.backing_id) else {
        trace_refused_identity_write(
            &mappings,
            identity,
            byte_count,
            "backing has no live mapping",
        );
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

/// Outcome of a baseline-guarded backing publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuardedBackingWrite {
    /// The backing bytes still matched the capture-time baseline and the source was published.
    Applied,
    /// The guest wrote inside the range after the baseline was captured; nothing was written.
    GuestModified,
    /// The range could not be resolved: identity moved, backing gone, or coverage lost.
    Unavailable,
}

/// Publishes `source` only while the backing bytes still equal `baseline`.
///
/// The backing holds the truth for every alias, so a byte that no longer matches the
/// capture-time baseline was authored by the guest (or an ordered command-processor effect)
/// after the capture. Overwriting it would revert a store the owner has already relied on —
/// the recorded MallocBinned3 0x40-poison corruption. The caller holds the registry lock, so
/// the comparison and the write cannot interleave with a remap.
///
/// Aligned 4- and 8-byte runs use a single compare-exchange against the baseline word so a
/// concurrent guest store cannot be silently reverted in the read-compare-write window that a
/// bulk memcpy publication still has for larger texture writebacks.
fn guarded_backing_write(
    backing: &GuestPhysicalBacking,
    offset: u64,
    baseline: &[u8],
    source: &[u8],
) -> GuardedBackingWrite {
    debug_assert_eq!(baseline.len(), source.len());
    if baseline.len() == 8 && offset.is_multiple_of(8) {
        let expected = u64::from_le_bytes(baseline.try_into().expect("8-byte baseline"));
        let desired = u64::from_le_bytes(source.try_into().expect("8-byte source"));
        return match backing.compare_exchange_u64_at(offset, expected, desired) {
            Some(true) => GuardedBackingWrite::Applied,
            Some(false) => GuardedBackingWrite::GuestModified,
            None => GuardedBackingWrite::Unavailable,
        };
    }
    if baseline.len() == 4 && offset.is_multiple_of(4) {
        let expected = u32::from_le_bytes(baseline.try_into().expect("4-byte baseline"));
        let desired = u32::from_le_bytes(source.try_into().expect("4-byte source"));
        return match backing.compare_exchange_u32_at(offset, expected, desired) {
            Some(true) => GuardedBackingWrite::Applied,
            Some(false) => GuardedBackingWrite::GuestModified,
            None => GuardedBackingWrite::Unavailable,
        };
    }
    let mut current = vec![0_u8; baseline.len()];
    if !backing.read_exact_at(offset, &mut current) {
        return GuardedBackingWrite::Unavailable;
    }
    if current != baseline {
        return GuardedBackingWrite::GuestModified;
    }
    if backing.write_all_at(offset, source) {
        GuardedBackingWrite::Applied
    } else {
        GuardedBackingWrite::Unavailable
    }
}

/// Release fence separating unordered payload stores from a subsequent label publication.
///
/// Call after writing DMA/fill payload bytes and before publishing the guest-visible label that
/// makes those bytes architecturally observable. Label stores themselves also use release
/// ordering; this fence covers the large memcpy path that cannot carry a per-store order.
#[inline]
pub fn guest_publish_release_fence() {
    std::sync::atomic::fence(Ordering::Release);
}

/// Release-store an aligned 32-bit word through a live guest VA.
///
/// Shared by HLE and the direct (write-watch-pinned) GPU publication path. The guest address space
/// is identity-mapped, so the VA names the same physical bytes as every shared alias.
pub fn store_guest_visible_u32(address: u64, value: u32) {
    debug_assert!(address.is_multiple_of(4));
    // SAFETY: callers prove `address` is a live, 4-byte-aligned guest mapping retained for the
    // store. Release ordering publishes prior host payload writes to guest observers.
    unsafe {
        std::sync::atomic::AtomicU32::from_ptr(address as *mut u32).store(value, Ordering::Release);
    }
}

/// Release-store an aligned 64-bit word through a live guest VA.
pub fn store_guest_visible_u64(address: u64, value: u64) {
    debug_assert!(address.is_multiple_of(8));
    // SAFETY: callers prove `address` is a live, 8-byte-aligned guest mapping retained for the
    // store. Release ordering publishes prior host payload writes to guest observers.
    unsafe {
        AtomicU64::from_ptr(address as *mut u64).store(value, Ordering::Release);
    }
}

/// Acquire-load an aligned 32-bit word through a live guest VA.
#[must_use]
pub fn load_guest_visible_u32(address: u64) -> u32 {
    debug_assert!(address.is_multiple_of(4));
    // SAFETY: callers prove `address` is a live, 4-byte-aligned guest mapping retained for the load.
    unsafe { std::sync::atomic::AtomicU32::from_ptr(address as *mut u32).load(Ordering::Acquire) }
}

/// Acquire-load an aligned 64-bit word through a live guest VA.
#[must_use]
pub fn load_guest_visible_u64(address: u64) -> u64 {
    debug_assert!(address.is_multiple_of(8));
    // SAFETY: callers prove `address` is a live, 8-byte-aligned guest mapping retained for the load.
    unsafe { AtomicU64::from_ptr(address as *mut u64).load(Ordering::Acquire) }
}

/// Publishes one aligned 32-bit label through shared backing while the identity still holds.
pub fn publish_guest_shared_u32_if_identity(
    address: u64,
    expected: GuestSharedMappingIdentity,
    value: u32,
) -> bool {
    if !address.is_multiple_of(4) {
        return false;
    }
    let mappings = shared_mappings()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some((backing, offset, _)) = resolve_shared_backing_range(&mappings, address, 4) else {
        return false;
    };
    if backing.id() != expected.backing_id || offset != expected.backing_offset {
        return false;
    }
    backing.store_u32_release_at(offset, value)
}

/// Publishes one aligned 64-bit label through shared backing while the identity still holds.
///
/// This is the ordered replacement for publishing fence labels as a runtime-length memcpy: one
/// aligned release store, not eight byte copies that can tear under a concurrent reader.
pub fn publish_guest_shared_u64_if_identity(
    address: u64,
    expected: GuestSharedMappingIdentity,
    value: u64,
) -> bool {
    if !address.is_multiple_of(8) {
        return false;
    }
    let mappings = shared_mappings()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some((backing, offset, _)) = resolve_shared_backing_range(&mappings, address, 8) else {
        return false;
    };
    if backing.id() != expected.backing_id || offset != expected.backing_offset {
        return false;
    }
    backing.store_u64_release_at(offset, value)
}

/// Acquire-loads one aligned 32-bit word from shared backing at `address`.
#[must_use]
pub fn load_guest_shared_u32(address: u64) -> Option<u32> {
    if !address.is_multiple_of(4) {
        return None;
    }
    let mappings = shared_mappings()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (backing, offset, _) = resolve_shared_backing_range(&mappings, address, 4)?;
    backing.load_u32_acquire_at(offset)
}

/// Acquire-loads one aligned 64-bit word from shared backing at `address`.
#[must_use]
pub fn load_guest_shared_u64(address: u64) -> Option<u64> {
    if !address.is_multiple_of(8) {
        return None;
    }
    let mappings = shared_mappings()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (backing, offset, _) = resolve_shared_backing_range(&mappings, address, 8)?;
    backing.load_u64_acquire_at(offset)
}

/// Baseline-guarded form of [`write_guest_shared_backing_if_identity`].
pub fn write_guest_shared_backing_if_identity_and_baseline(
    address: u64,
    expected: GuestSharedMappingIdentity,
    baseline: &[u8],
    source: &[u8],
) -> GuardedBackingWrite {
    let Ok(byte_count) = u64::try_from(source.len()) else {
        return GuardedBackingWrite::Unavailable;
    };
    let mappings = shared_mappings()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some((backing, offset, _)) = resolve_shared_backing_range(&mappings, address, byte_count)
    else {
        return GuardedBackingWrite::Unavailable;
    };
    if backing.id() != expected.backing_id || offset != expected.backing_offset {
        return GuardedBackingWrite::Unavailable;
    }
    guarded_backing_write(&backing, offset, baseline, source)
}

/// Baseline-guarded form of [`write_guest_shared_backing_at_identity`].
///
/// The same two licence bounds apply — a physically keyed backing and full live-mapping
/// coverage — and on top of them the capture-time baseline must still match, so a recycled
/// allocation the guest has since written is left exactly as the guest left it.
pub fn write_guest_shared_backing_at_identity_with_baseline(
    identity: GuestSharedMappingIdentity,
    baseline: &[u8],
    source: &[u8],
) -> GuardedBackingWrite {
    let Ok(byte_count) = u64::try_from(source.len()) else {
        return GuardedBackingWrite::Unavailable;
    };
    if byte_count == 0 {
        return GuardedBackingWrite::Unavailable;
    }
    let Some(end) = identity.backing_offset.checked_add(byte_count) else {
        return GuardedBackingWrite::Unavailable;
    };
    let mappings = shared_mappings()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(backing) = mappings.any_backing(identity.backing_id) else {
        trace_refused_identity_write(
            &mappings,
            identity,
            byte_count,
            "backing has no live mapping",
        );
        return GuardedBackingWrite::Unavailable;
    };
    if backing.is_virtually_keyed() {
        trace_refused_identity_write(&mappings, identity, byte_count, "virtually keyed backing");
        return GuardedBackingWrite::Unavailable;
    }
    if end > backing.size() {
        trace_refused_identity_write(&mappings, identity, byte_count, "range exceeds backing");
        return GuardedBackingWrite::Unavailable;
    }
    if !identity_range_is_live(&mappings, identity.backing_id, identity.backing_offset, end) {
        trace_refused_identity_write(
            &mappings,
            identity,
            byte_count,
            "no live mapping covers the range",
        );
        return GuardedBackingWrite::Unavailable;
    }
    guarded_backing_write(&backing, identity.backing_offset, baseline, source)
}

/// A capture-time claim on a guest range's physical backing, held by a host-side producer —
/// today a GPU writeback — from capture until publication.
///
/// The lease is the one concept that answers "may this producer still publish into guest
/// memory?" (audit B21). This first stage is a behaviour-preserving wrapper: it records the
/// resolved identity and the shared-mapping epoch at acquisition, and its publish method
/// performs, under a single registry-lock pass, exactly the dance its call sites previously
/// spelled out — live-VA publish, then the physically-keyed producer-identity licence — with
/// the capture-baseline guard applied in every mode. Later stages tighten invalidation
/// (release, decommit, address-space drop) behind the same method.
pub struct GuestRangeLease {
    address: u64,
    byte_count: u64,
    identity: GuestSharedMappingIdentity,
    /// Keeps the backing object alive across the lease so later invalidation stages can
    /// compare object identity; it deliberately grants no publication licence — every publish
    /// re-proves live mapping coverage under the registry lock.
    _backing: Option<GuestPhysicalBacking>,
    acquisition_epoch: u64,
}

/// Outcome of a leased baseline-guarded publication of one changed run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LeasePublishOutcome {
    /// The guest VA still resolves to the leased identity and the bytes were published there.
    AppliedAtLiveVa,
    /// The VA moved; the bytes were published through the producer-identity licence.
    AppliedByIdentity,
    /// The guest wrote inside the run after capture; nothing was written, the guest bytes win.
    GuestModified,
    /// Neither the live VA nor the identity licence admits the publication.
    Revoked,
    /// The VA resolves to the leased identity but the backing write failed — an I/O defect,
    /// not an ownership change; callers should surface it as an error.
    WriteFailed,
}

impl GuestRangeLease {
    /// Resolves and claims `address..address+byte_count` now.
    #[must_use]
    pub fn acquire(address: u64, byte_count: u64) -> Option<Self> {
        let epoch = guest_shared_mapping_epoch();
        let mappings = shared_mappings()
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (backing, offset, _) = resolve_shared_backing_range(&mappings, address, byte_count)?;
        let identity = GuestSharedMappingIdentity {
            backing_id: backing.id(),
            backing_offset: offset,
        };
        Some(Self {
            address,
            byte_count,
            identity,
            _backing: Some(backing),
            acquisition_epoch: epoch,
        })
    }

    /// Adopts an identity resolved earlier — an upload-time capture — without re-resolving.
    ///
    /// The backing object is pinned when it still has a live mapping; a range whose backing is
    /// already gone still constructs, and every publish through it then reports [`Revoked`]
    /// exactly as the unadopted path would.
    ///
    /// [`Revoked`]: LeasePublishOutcome::Revoked
    #[must_use]
    pub fn adopt(address: u64, byte_count: u64, identity: GuestSharedMappingIdentity) -> Self {
        let backing = {
            let mappings = shared_mappings()
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            mappings.any_backing(identity.backing_id)
        };
        Self {
            address,
            byte_count,
            identity,
            _backing: backing,
            acquisition_epoch: guest_shared_mapping_epoch(),
        }
    }

    #[must_use]
    pub fn identity(&self) -> GuestSharedMappingIdentity {
        self.identity
    }

    #[must_use]
    pub fn address(&self) -> u64 {
        self.address
    }

    /// Reports whether the guest VA still resolves to the leased identity.
    #[must_use]
    pub fn live_va_matches(&self) -> bool {
        guest_shared_backing_identity(self.address, self.byte_count) == Some(self.identity)
    }

    /// Publishes one changed run only while its capture-time VA still names the leased backing.
    ///
    /// This is the publication rule for resources whose ownership is expressed by a guest
    /// virtual address, such as copied writable global buffers. Unlike
    /// [`Self::publish_run_if_baseline`], it never follows the old physical backing through a
    /// surviving alias after the guest recycles that VA.
    #[must_use]
    pub fn publish_run_at_live_va_if_baseline(
        &self,
        run_offset: u64,
        baseline: &[u8],
        source: &[u8],
    ) -> LeasePublishOutcome {
        debug_assert_eq!(baseline.len(), source.len());
        let Ok(run_bytes) = u64::try_from(source.len()) else {
            return LeasePublishOutcome::Revoked;
        };
        if run_bytes == 0 {
            return LeasePublishOutcome::Revoked;
        }
        let Some(run_address) = self.address.checked_add(run_offset) else {
            return LeasePublishOutcome::Revoked;
        };
        let Some(run_backing_offset) = self.identity.backing_offset.checked_add(run_offset) else {
            return LeasePublishOutcome::Revoked;
        };
        let mappings = shared_mappings()
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some((backing, offset, _)) =
            resolve_shared_backing_range(&mappings, run_address, run_bytes)
        else {
            return LeasePublishOutcome::Revoked;
        };
        if backing.id() != self.identity.backing_id || offset != run_backing_offset {
            return LeasePublishOutcome::Revoked;
        }
        match guarded_backing_write(&backing, offset, baseline, source) {
            GuardedBackingWrite::Applied => LeasePublishOutcome::AppliedAtLiveVa,
            GuardedBackingWrite::GuestModified => LeasePublishOutcome::GuestModified,
            GuardedBackingWrite::Unavailable => LeasePublishOutcome::WriteFailed,
        }
    }

    /// Publishes one changed run while the capture-time baseline still holds.
    ///
    /// One registry-lock pass: if the run's VA still resolves to the leased identity, the
    /// baseline-guarded write lands there; otherwise the producer-identity licence applies —
    /// physically keyed backing, full live coverage — with the same baseline guard. A run the
    /// guest has since written is never overwritten in any mode.
    #[must_use]
    pub fn publish_run_if_baseline(
        &self,
        run_offset: u64,
        baseline: &[u8],
        source: &[u8],
    ) -> LeasePublishOutcome {
        debug_assert_eq!(baseline.len(), source.len());
        let Ok(run_bytes) = u64::try_from(source.len()) else {
            return LeasePublishOutcome::Revoked;
        };
        if run_bytes == 0 {
            return LeasePublishOutcome::Revoked;
        }
        let Some(run_address) = self.address.checked_add(run_offset) else {
            return LeasePublishOutcome::Revoked;
        };
        let Some(run_backing_offset) = self.identity.backing_offset.checked_add(run_offset) else {
            return LeasePublishOutcome::Revoked;
        };
        let Some(run_backing_end) = run_backing_offset.checked_add(run_bytes) else {
            return LeasePublishOutcome::Revoked;
        };
        let mappings = shared_mappings()
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some((backing, offset, _)) =
            resolve_shared_backing_range(&mappings, run_address, run_bytes)
            && backing.id() == self.identity.backing_id
            && offset == run_backing_offset
        {
            return match guarded_backing_write(&backing, offset, baseline, source) {
                GuardedBackingWrite::Applied => LeasePublishOutcome::AppliedAtLiveVa,
                GuardedBackingWrite::GuestModified => LeasePublishOutcome::GuestModified,
                GuardedBackingWrite::Unavailable => LeasePublishOutcome::WriteFailed,
            };
        }
        // The VA moved. Publish through the producer-identity licence, under its two bounds.
        let Some(backing) = mappings.any_backing(self.identity.backing_id) else {
            trace_refused_identity_write(
                &mappings,
                self.identity,
                run_bytes,
                "backing has no live mapping",
            );
            return LeasePublishOutcome::Revoked;
        };
        if backing.is_virtually_keyed() {
            trace_refused_identity_write(
                &mappings,
                self.identity,
                run_bytes,
                "virtually keyed backing",
            );
            return LeasePublishOutcome::Revoked;
        }
        if run_backing_end > backing.size() {
            trace_refused_identity_write(
                &mappings,
                self.identity,
                run_bytes,
                "range exceeds backing",
            );
            return LeasePublishOutcome::Revoked;
        }
        if !identity_range_is_live(
            &mappings,
            self.identity.backing_id,
            run_backing_offset,
            run_backing_end,
        ) {
            trace_refused_identity_write(
                &mappings,
                self.identity,
                run_bytes,
                "no live mapping covers the range",
            );
            return LeasePublishOutcome::Revoked;
        }
        match guarded_backing_write(&backing, run_backing_offset, baseline, source) {
            GuardedBackingWrite::Applied => LeasePublishOutcome::AppliedByIdentity,
            GuardedBackingWrite::GuestModified => LeasePublishOutcome::GuestModified,
            GuardedBackingWrite::Unavailable => LeasePublishOutcome::Revoked,
        }
    }

    /// The shared-mapping epoch recorded at acquisition; later invalidation stages compare it.
    #[must_use]
    pub fn acquisition_epoch(&self) -> u64 {
        self.acquisition_epoch
    }
}

/// Reports whether live records cover every byte of a physical interval in one backing.
fn identity_range_is_live(
    mappings: &SharedRegistry,
    backing_id: u64,
    start: u64,
    end: u64,
) -> bool {
    let mut cursor = start;
    while cursor < end {
        let covered_end = mappings
            .backing_records(backing_id)
            .filter_map(|(mapping_start, record)| {
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
    mappings: &SharedRegistry,
    identity: GuestSharedMappingIdentity,
    byte_count: u64,
    reason: &str,
) {
    let sequence = REFUSED_IDENTITY_WRITES.fetch_add(1, Ordering::Relaxed) + 1;
    if sequence > 16 && !sequence.is_power_of_two() {
        return;
    }
    let alias = mappings
        .backing_records(identity.backing_id)
        .find_map(|(mapping_start, record)| {
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
    mappings: &mut SharedRegistry,
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
        mappings.remove(start);
        if start < address {
            mappings.insert(
                start,
                SharedMappingRecord {
                    end: address,
                    generation: 0,
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
                    generation: 0,
                },
            );
        }
    }
}

pub(super) fn unregister_shared_mapping_range(address: u64, byte_count: u64) {
    let mut mappings = shared_mappings()
        .write()
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

fn coalesce_shared_mapping_record(mappings: &mut SharedRegistry, address: u64) {
    let mut merged_start = address;
    let Some(current) = mappings.get(&merged_start).cloned() else {
        return;
    };
    if let Some((&previous_start, previous)) = mappings.range(..merged_start).next_back()
        && records_are_physically_contiguous(previous_start, previous, merged_start, &current)
    {
        let previous = previous.clone();
        mappings.remove(previous_start);
        mappings.remove(merged_start);
        mappings.insert(
            previous_start,
            SharedMappingRecord {
                end: current.end,
                backing: previous.backing,
                backing_offset: previous.backing_offset,
                generation: 0,
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
        mappings.remove(merged_start);
        mappings.remove(current.end);
        mappings.insert(
            merged_start,
            SharedMappingRecord {
                end: next.end,
                backing: current.backing,
                backing_offset: current.backing_offset,
                generation: 0,
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
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    unregister_shared_mapping_range_locked(&mut mappings, address, byte_count);
    mappings.insert(
        address,
        SharedMappingRecord {
            end,
            backing: backing.clone(),
            backing_offset,
            generation: 0,
        },
    );
    // The VMA layer retains mapping-call boundaries when the guest requests them. The physical
    // registry does not need those virtual bookkeeping seams: texture snapshots and CPU write
    // watches care only that adjacent addresses name adjacent bytes of one backing. Coalescing
    // here turns separately committed pages back into one watchable/copyable physical run.
    coalesce_shared_mapping_record(&mut mappings, address);
}

/// Number of `by_address` records naming `backing_id` (N1 coalescing / N2 index regression check).
///
/// Backing ids are allocated monotonically, so a test that creates its own backing observes only
/// its own records here regardless of what other process-global tests are doing concurrently.
#[cfg(test)]
pub(super) fn shared_backing_record_count(backing_id: u64) -> usize {
    shared_mappings()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .backing_record_count(backing_id)
}

/// Snapshot of every live shared guest mapping: (guest address, byte length, backing id,
/// backing offset). The table view a whole-memory dump walks.
pub fn guest_shared_mapping_table() -> Vec<(u64, u64, u64, u64)> {
    shared_mappings()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .map(|(address, record)| {
            (
                *address,
                record.end.saturating_sub(*address),
                record.backing.id(),
                record.backing_offset,
            )
        })
        .collect()
}
