// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    GuestTextureDescriptor, ResidentGuestTexture, ResidentTextureMap, ResourceSpan, SpanQueryOrder,
};
use remu_memory::{CpuWriteWatchRetirement, guest_shared_backing_identity};
use std::collections::HashSet;

pub(crate) const RESIDENT_WORK_AGE_LABELS: [&str; 6] = [
    "0-63",
    "64-511",
    "512-4095",
    "4096-16383",
    "16384+",
    "never",
];
pub(crate) const RESIDENT_FLIP_AGE_LABELS: [&str; 6] = ["0", "1", "2-7", "8-31", "32+", "never"];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ResidentTextureAgeBucket {
    pub(crate) resident_count: usize,
    pub(crate) resident_bytes: u64,
    pub(crate) gpu_owned_count: usize,
    pub(crate) gpu_owned_bytes: u64,
    pub(crate) cpu_owned_count: usize,
    pub(crate) cpu_owned_bytes: u64,
}

impl ResidentTextureAgeBucket {
    fn record(&mut self, resident: &ResidentGuestTexture) {
        let bytes = u64::try_from(resident.texture.allocated_size()).unwrap_or(u64::MAX);
        self.resident_count += 1;
        self.resident_bytes = self.resident_bytes.saturating_add(bytes);
        if resident.gpu_written() {
            self.gpu_owned_count += 1;
            self.gpu_owned_bytes = self.gpu_owned_bytes.saturating_add(bytes);
        } else {
            self.cpu_owned_count += 1;
            self.cpu_owned_bytes = self.cpu_owned_bytes.saturating_add(bytes);
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ResidentTextureActivityStats {
    pub(crate) read_work: [ResidentTextureAgeBucket; 6],
    pub(crate) write_work: [ResidentTextureAgeBucket; 6],
    pub(crate) read_flip: [ResidentTextureAgeBucket; 6],
    pub(crate) write_flip: [ResidentTextureAgeBucket; 6],
}

pub(crate) fn resident_texture_activity_stats(
    residents: &ResidentTextureMap,
    current_work: u64,
    current_flip: u64,
) -> ResidentTextureActivityStats {
    let mut stats = ResidentTextureActivityStats::default();
    for resident in residents.values() {
        let activity = resident.activity.unwrap_or_default();
        stats.read_work[work_age_bucket(activity.last_read_work, current_work)].record(resident);
        stats.write_work[work_age_bucket(activity.last_write_work, current_work)].record(resident);
        stats.read_flip[flip_age_bucket(activity.last_read_flip, current_flip)].record(resident);
        stats.write_flip[flip_age_bucket(activity.last_write_flip, current_flip)].record(resident);
    }
    stats
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ResidentTextureEvictionPlan {
    pub(crate) resident_bytes: u64,
    pub(crate) planned_bytes: u64,
    pub(crate) descriptors: Vec<GuestTextureDescriptor>,
}

/// Select completed-frame residents for pressure eviction without discarding architectural state.
///
/// The caller is responsible for downloading GPU-owned candidates before removal. Residents used
/// by `current_flip` are never selected: their Metal allocation can still be referenced by the
/// open frame even if a nominal byte budget is exceeded temporarily. Clean (CPU-owned) residents
/// are preferred because guest RAM is already authoritative, followed by the oldest GPU-owned
/// allocations.
pub(crate) fn resident_texture_eviction_plan(
    residents: &ResidentTextureMap,
    byte_capacity: u64,
    current_flip: u64,
    maximum_evictions: usize,
) -> ResidentTextureEvictionPlan {
    let resident_bytes = residents.values().fold(0_u64, |total, resident| {
        total.saturating_add(u64::try_from(resident.texture.allocated_size()).unwrap_or(u64::MAX))
    });
    if resident_bytes <= byte_capacity || maximum_evictions == 0 {
        return ResidentTextureEvictionPlan {
            resident_bytes,
            ..ResidentTextureEvictionPlan::default()
        };
    }

    let mut candidates = residents
        .values()
        .filter(|resident| resident.last_use.flip < current_flip)
        .filter(|resident| {
            // The pairwise scan this replaces was quadratic in the resident count and took the
            // shared-mapping lock once per pair.
            !resident.gpu_written()
                || !ResourceSpan::for_texture(resident.descriptor).is_some_and(|span| {
                    residents
                        .overlapping(&span, SpanQueryOrder::QueryFirst)
                        .into_iter()
                        .any(|other| {
                            other != resident.descriptor
                                && residents
                                    .get(&other)
                                    .is_some_and(|other| other.gpu_written())
                        })
                })
        })
        .map(|resident| {
            let mapping_replaced =
                resident.write_watch.retirement() == Some(CpuWriteWatchRetirement::Replaced);
            (
                // Replaced mappings carry no valid destination for writeback, and a demoted
                // allocation answers for no byte at all. Clean residents need no writeback.
                // Retire all three before GPU-owned live mappings.
                !mapping_replaced && resident.is_live(),
                resident.gpu_written(),
                resident.last_use.flip,
                resident.last_use.work,
                resident.content_version,
                resident.descriptor,
                u64::try_from(resident.texture.allocated_size()).unwrap_or(u64::MAX),
            )
        })
        .collect::<Vec<_>>();
    candidates.sort_unstable_by_key(
        |(reclaim_last, gpu_written, flip, work, version, descriptor, _)| {
            (
                *reclaim_last,
                *gpu_written,
                *flip,
                *work,
                *version,
                descriptor.address,
            )
        },
    );

    let mut planned_bytes = 0_u64;
    let mut descriptors = Vec::new();
    for (_, _, _, _, _, descriptor, bytes) in candidates {
        if resident_bytes.saturating_sub(planned_bytes) <= byte_capacity
            || descriptors.len() == maximum_evictions
        {
            break;
        }
        planned_bytes = planned_bytes.saturating_add(bytes);
        descriptors.push(descriptor);
    }
    ResidentTextureEvictionPlan {
        resident_bytes,
        planned_bytes,
        descriptors,
    }
}

fn work_age_bucket(last: Option<u64>, current: u64) -> usize {
    let Some(last) = last else {
        return 5;
    };
    match current.saturating_sub(last) {
        0..=63 => 0,
        64..=511 => 1,
        512..=4_095 => 2,
        4_096..=16_383 => 3,
        _ => 4,
    }
}

fn flip_age_bucket(last: Option<u64>, current: u64) -> usize {
    let Some(last) = last else {
        return 5;
    };
    match current.saturating_sub(last) {
        0 => 0,
        1 => 1,
        2..=7 => 2,
        8..=31 => 3,
        _ => 4,
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ResidentTextureBackingOverlapStats {
    pub(crate) pair_count: usize,
    /// Sum of every pairwise physical intersection. This intentionally exposes nested aliases:
    /// their duplicated coverage is the quantity whose Metal residency needs attribution.
    pub(crate) pair_overlap_bytes: u64,
    pub(crate) resident_count: usize,
    pub(crate) resident_bytes: u64,
    pub(crate) gpu_owned_count: usize,
    pub(crate) gpu_owned_bytes: u64,
    pub(crate) cpu_owned_count: usize,
    pub(crate) cpu_owned_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
struct BackingInterval {
    descriptor: GuestTextureDescriptor,
    backing_id: u64,
    start: u64,
    end: u64,
}

pub(crate) fn resident_texture_backing_overlap_stats(
    residents: &ResidentTextureMap,
) -> ResidentTextureBackingOverlapStats {
    let mut intervals = residents
        .values()
        .filter(|resident| {
            resident.write_watch.retirement() != Some(CpuWriteWatchRetirement::Replaced)
        })
        .filter_map(|resident| {
            let (address, byte_count) = resident.write_watch.range();
            let identity = guest_shared_backing_identity(address, byte_count)?;
            Some(BackingInterval {
                descriptor: resident.descriptor,
                backing_id: identity.backing_id,
                start: identity.backing_offset,
                end: identity.backing_offset.checked_add(byte_count)?,
            })
        })
        .collect::<Vec<_>>();
    intervals.sort_unstable_by_key(|interval| (interval.backing_id, interval.start, interval.end));

    let mut stats = ResidentTextureBackingOverlapStats::default();
    let mut participants = HashSet::new();
    for (left_index, left) in intervals.iter().enumerate() {
        for right in intervals.iter().skip(left_index + 1) {
            if right.backing_id != left.backing_id || right.start >= left.end {
                break;
            }
            let overlap_start = left.start.max(right.start);
            let overlap_end = left.end.min(right.end);
            if overlap_start >= overlap_end {
                continue;
            }
            stats.pair_count += 1;
            stats.pair_overlap_bytes = stats
                .pair_overlap_bytes
                .saturating_add(overlap_end - overlap_start);
            participants.insert(left.descriptor);
            participants.insert(right.descriptor);
        }
    }

    for descriptor in participants {
        let resident = residents
            .get(&descriptor)
            .expect("overlap participant belongs to the resident map");
        let bytes = u64::try_from(resident.texture.allocated_size()).unwrap_or(u64::MAX);
        stats.resident_count += 1;
        stats.resident_bytes = stats.resident_bytes.saturating_add(bytes);
        if resident.gpu_written() {
            stats.gpu_owned_count += 1;
            stats.gpu_owned_bytes = stats.gpu_owned_bytes.saturating_add(bytes);
        } else {
            stats.cpu_owned_count += 1;
            stats.cpu_owned_bytes = stats.cpu_owned_bytes.saturating_add(bytes);
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::{flip_age_bucket, work_age_bucket};

    #[test]
    fn activity_age_buckets_keep_recent_old_and_unseen_resources_distinct() {
        assert_eq!(work_age_bucket(Some(10_000), 10_000), 0);
        assert_eq!(work_age_bucket(Some(9_937), 10_000), 0);
        assert_eq!(work_age_bucket(Some(9_936), 10_000), 1);
        assert_eq!(work_age_bucket(Some(5_904), 10_000), 3);
        assert_eq!(work_age_bucket(Some(0), 20_000), 4);
        assert_eq!(work_age_bucket(None, 10_000), 5);

        assert_eq!(flip_age_bucket(Some(20), 20), 0);
        assert_eq!(flip_age_bucket(Some(19), 20), 1);
        assert_eq!(flip_age_bucket(Some(13), 20), 2);
        assert_eq!(flip_age_bucket(Some(0), 20), 3);
        assert_eq!(flip_age_bucket(Some(0), 40), 4);
        assert_eq!(flip_age_bucket(None, 20), 5);
    }
}
