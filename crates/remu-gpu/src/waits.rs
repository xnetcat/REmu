// SPDX-License-Identifier: GPL-2.0-or-later

//! Process-global registry of submitted guest-memory waits and their producer latches.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};

use crate::{GpuEvent, GpuMemoryWait, GpuWaitId};

pub(crate) const GPU_WAIT_CAPACITY: usize = 65_536;

#[derive(Clone, Copy, Debug)]
pub(crate) struct RegisteredGpuWait {
    condition: GpuMemoryWait,
    latched_value: Option<u64>,
}

#[derive(Default)]
pub(crate) struct GpuWaitRegistry {
    next_id: u64,
    pub(crate) waits: HashMap<GpuWaitId, RegisteredGpuWait>,
    by_label: HashMap<(u64, bool), Vec<GpuWaitId>>,
    by_address: BTreeMap<u64, Vec<GpuWaitId>>,
}

impl GpuWaitRegistry {
    fn register_many(&mut self, conditions: &[GpuMemoryWait]) -> Result<Vec<GpuWaitId>, ()> {
        if self
            .waits
            .len()
            .checked_add(conditions.len())
            .is_none_or(|count| count > GPU_WAIT_CAPACITY)
        {
            return Err(());
        }
        let condition_count = u64::try_from(conditions.len()).map_err(|_| ())?;
        let next_id = self.next_id.checked_add(condition_count).ok_or(())?;

        let mut ids = Vec::with_capacity(conditions.len());
        for (index, &condition) in conditions.iter().enumerate() {
            let id = GpuWaitId(
                self.next_id
                    + u64::try_from(index)
                        .expect("a registered wait index fits u64 after length validation")
                    + 1,
            );
            self.waits.insert(
                id,
                RegisteredGpuWait {
                    condition,
                    latched_value: None,
                },
            );
            self.by_label
                .entry((condition.address, condition.is_64_bit))
                .or_default()
                .push(id);
            self.by_address
                .entry(condition.address)
                .or_default()
                .push(id);
            ids.push(id);
        }
        self.next_id = next_id;
        Ok(ids)
    }

    fn retire(&mut self, id: GpuWaitId) {
        let Some(registered) = self.waits.remove(&id) else {
            return;
        };
        let key = (registered.condition.address, registered.condition.is_64_bit);
        let remove_key = self.by_label.get_mut(&key).is_some_and(|ids| {
            ids.retain(|candidate| *candidate != id);
            ids.is_empty()
        });
        if remove_key {
            self.by_label.remove(&key);
        }
        let remove_address = self
            .by_address
            .get_mut(&registered.condition.address)
            .is_some_and(|ids| {
                ids.retain(|candidate| *candidate != id);
                ids.is_empty()
            });
        if remove_address {
            self.by_address.remove(&registered.condition.address);
        }
    }

    pub(crate) fn cancel(&mut self, ids: &[GpuWaitId]) {
        for &id in ids {
            self.retire(id);
        }
    }

    fn take_latched(&mut self, id: GpuWaitId) -> Option<u64> {
        let value = self.waits.get(&id)?.latched_value?;
        self.retire(id);
        Some(value)
    }

    fn record_producer(&mut self, address: u64, is_64_bit: bool, value: u64) {
        let Some(ids) = self.by_label.get(&(address, is_64_bit)).cloned() else {
            return;
        };
        for id in ids {
            let Some(wait) = self.waits.get_mut(&id) else {
                continue;
            };
            if wait.latched_value.is_none() && wait.condition.is_satisfied_by(value) {
                wait.latched_value = Some(value);
            }
        }
    }

    fn pending_overlapping(&self, start: u64, byte_count: u64) -> Vec<(GpuWaitId, GpuMemoryWait)> {
        if byte_count == 0 {
            return Vec::new();
        }
        let end = start.saturating_add(byte_count);
        // A producer can begin up to seven bytes after the start of a pending 64-bit wait while
        // still overlapping it. Indexing by wait address turns the common label/fill lookup from
        // O(all live waits) into O(log n + waits near the produced range), without losing those
        // left-edge overlaps.
        let first_possible_wait = start.saturating_sub(7);
        self.by_address
            .range(first_possible_wait..end)
            .flat_map(|(_, ids)| ids)
            .filter_map(|&id| {
                let registered = self
                    .waits
                    .get(&id)
                    .expect("wait address index references a live wait");
                if registered.latched_value.is_some() {
                    return None;
                }
                let width = if registered.condition.is_64_bit { 8 } else { 4 };
                let wait_end = registered.condition.address.saturating_add(width);
                (start < wait_end && registered.condition.address < end)
                    .then_some((id, registered.condition))
            })
            .collect()
    }

    pub(crate) fn clear(&mut self) {
        self.waits.clear();
        self.by_label.clear();
        self.by_address.clear();
    }
}

static GPU_WAITS: OnceLock<Mutex<GpuWaitRegistry>> = OnceLock::new();

pub(crate) fn gpu_wait_registry() -> &'static Mutex<GpuWaitRegistry> {
    GPU_WAITS.get_or_init(|| Mutex::new(GpuWaitRegistry::default()))
}

pub(crate) fn register_queue_submission_waits(event: &mut GpuEvent) -> Result<Vec<GpuWaitId>, ()> {
    let GpuEvent::QueueSubmission { events, .. } = event else {
        return Ok(Vec::new());
    };
    let waits = events
        .iter()
        .filter_map(|event| match event {
            GpuEvent::MemoryWait(wait) => Some(*wait),
            _ => None,
        })
        .collect::<Vec<_>>();
    let registered_ids = gpu_wait_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .register_many(&waits)?;
    let mut ids = registered_ids.iter().copied();
    for event in events.iter_mut() {
        if let GpuEvent::MemoryWait(wait) = event {
            let wait = *wait;
            *event = GpuEvent::RegisteredMemoryWait {
                id: ids
                    .next()
                    .expect("every submitted memory wait received a ticket"),
                wait,
            };
        }
    }
    debug_assert!(ids.next().is_none());
    Ok(registered_ids)
}

/// Takes a satisfying producer value latched for one submitted wait.
///
/// A successful take retires the ticket. A pending, cancelled, or already-consumed ticket returns
/// `None`.
pub fn take_gpu_wait_latch(id: GpuWaitId) -> Option<u64> {
    gpu_wait_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take_latched(id)
}

/// Retires one satisfied, rejected, or otherwise consumed wait ticket.
pub fn retire_gpu_wait(id: GpuWaitId) {
    gpu_wait_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .retire(id);
}

/// Latches a concrete ordered producer value into matching waits already submitted for this label.
///
/// The registry intentionally retains no historical producer value. A later reuse of the same
/// guest address therefore cannot inherit satisfaction from an older label generation.
pub fn record_gpu_wait_producer(address: u64, is_64_bit: bool, value: u64) {
    gpu_wait_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .record_producer(address, is_64_bit, value);
}

/// Returns pending submitted waits whose watched bytes overlap an ordered producer range.
///
/// Backends use this for DMA fills/copies and proven shader writebacks: after applying the complete
/// producer packet, they read only these 4- or 8-byte values and pass them to
/// [`record_gpu_wait_producer`].
pub fn pending_gpu_waits_overlapping(
    start: u64,
    byte_count: u64,
) -> Vec<(GpuWaitId, GpuMemoryWait)> {
    gpu_wait_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .pending_overlapping(start, byte_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wait(address: u64, is_64_bit: bool) -> GpuMemoryWait {
        GpuMemoryWait {
            address,
            reference: 1,
            mask: u64::MAX,
            compare_function: 3,
            is_64_bit,
        }
    }

    #[test]
    fn overlap_index_includes_waits_beginning_before_the_producer() {
        let mut registry = GpuWaitRegistry::default();
        let conditions = [wait(0x1000, true), wait(0x1004, false), wait(0x2000, false)];
        let ids = registry
            .register_many(&conditions)
            .expect("register test waits");

        let overlapping = registry.pending_overlapping(0x1004, 4);
        assert_eq!(overlapping.len(), 2);
        assert!(overlapping.contains(&(ids[0], conditions[0])));
        assert!(overlapping.contains(&(ids[1], conditions[1])));
    }

    #[test]
    fn overlap_index_removes_retired_and_latched_waits() {
        let mut registry = GpuWaitRegistry::default();
        let conditions = [
            wait(0x1000, false),
            wait(0x1004, false),
            wait(0x1008, false),
        ];
        let ids = registry
            .register_many(&conditions)
            .expect("register test waits");
        registry.retire(ids[0]);
        registry.record_producer(0x1004, false, 1);

        assert_eq!(
            registry.pending_overlapping(0x1000, 12),
            vec![(ids[2], conditions[2])]
        );
        assert!(!registry.by_address.contains_key(&0x1000));
    }
}
