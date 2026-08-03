// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{GuestTextureDescriptor, ResidentGuestTexture, ResourceSpan};
use std::collections::HashMap;

/// `resident_guest_textures` together with an interval index over the residents' spans.
///
/// The map is private: the index is only correct while it mirrors the map exactly, so every
/// insertion and removal has to pass through this type rather than touching a `HashMap` directly.
pub(crate) struct ResidentTextureMap {
    textures: HashMap<GuestTextureDescriptor, ResidentGuestTexture>,
    index: ResidentSpanIndex,
}

impl ResidentTextureMap {
    pub(crate) fn new() -> Self {
        Self {
            textures: HashMap::new(),
            index: ResidentSpanIndex::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.textures.len()
    }

    pub(crate) fn contains_key(&self, descriptor: &GuestTextureDescriptor) -> bool {
        self.textures.contains_key(descriptor)
    }

    pub(crate) fn get(&self, descriptor: &GuestTextureDescriptor) -> Option<&ResidentGuestTexture> {
        self.textures.get(descriptor)
    }

    pub(crate) fn get_mut(
        &mut self,
        descriptor: &GuestTextureDescriptor,
    ) -> Option<&mut ResidentGuestTexture> {
        self.textures.get_mut(descriptor)
    }

    /// Files `resident` under `descriptor` and indexes its span.
    ///
    /// Callers must keep `resident.descriptor == descriptor`: overlap scans read the descriptor
    /// off the value while the index is built from the key, and they have to agree.
    pub(crate) fn insert(
        &mut self,
        descriptor: GuestTextureDescriptor,
        resident: ResidentGuestTexture,
    ) -> Option<ResidentGuestTexture> {
        self.index.insert(descriptor);
        self.textures.insert(descriptor, resident)
    }

    pub(crate) fn remove(
        &mut self,
        descriptor: &GuestTextureDescriptor,
    ) -> Option<ResidentGuestTexture> {
        self.index.remove(descriptor);
        self.textures.remove(descriptor)
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = &GuestTextureDescriptor> {
        self.textures.keys()
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &ResidentGuestTexture> {
        self.textures.values()
    }

    pub(crate) fn iter(
        &self,
    ) -> impl Iterator<Item = (&GuestTextureDescriptor, &ResidentGuestTexture)> {
        self.textures.iter()
    }

    /// Residency keys whose span overlaps `query` under the predicate's operand `order`.
    pub(crate) fn overlapping(
        &self,
        query: &ResourceSpan,
        order: SpanQueryOrder,
    ) -> Vec<GuestTextureDescriptor> {
        self.index.overlapping(query, order)
    }
}

#[cfg(test)]
impl std::ops::Index<&GuestTextureDescriptor> for ResidentTextureMap {
    type Output = ResidentGuestTexture;

    fn index(&self, descriptor: &GuestTextureDescriptor) -> &Self::Output {
        &self.textures[descriptor]
    }
}

/// Which operand the overlap predicate's asymmetric registry fallback resolves first.
///
/// The fallback walks the aliases of its *left* operand's backing, so it can answer true for
/// `left.overlaps(right)` and false for `right.overlaps(left)` when only one of them resolves.
/// Every converted scan must therefore keep the operand order it used as a linear filter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SpanQueryOrder {
    /// The linear scan tested `resident.overlaps(query)`.
    ResidentFirst,
    /// The linear scan tested `query.overlaps(resident)`.
    QueryFirst,
}

/// Sorted interval index over the spans of a resident set.
///
/// Two interval lists are needed because the overlap predicate answers true on a virtual-address
/// intersection even when the operands resolve to unrelated physical backings, and true on a
/// physical intersection even when the virtual ranges are disjoint. Residents whose physical
/// identity did not resolve stay in a linear list: the predicate answers those from the registry,
/// which is exactly what the index exists to avoid, but they are the rare case.
pub(crate) struct ResidentSpanIndex {
    spans: HashMap<GuestTextureDescriptor, ResourceSpan>,
    virtual_intervals: IntervalList,
    physical_intervals: IntervalList,
    unresolved: Vec<GuestTextureDescriptor>,
}

impl ResidentSpanIndex {
    pub(crate) fn new() -> Self {
        Self {
            spans: HashMap::new(),
            virtual_intervals: IntervalList::default(),
            physical_intervals: IntervalList::default(),
            unresolved: Vec::new(),
        }
    }

    pub(crate) fn insert(&mut self, descriptor: GuestTextureDescriptor) {
        self.remove(&descriptor);
        // A descriptor whose byte count overflows can never overlap anything, so it is left out
        // of the index entirely rather than admitted with a degenerate span.
        let Some(span) = ResourceSpan::for_texture(descriptor) else {
            return;
        };
        self.virtual_intervals
            .insert(0, span.address(), span.end(), descriptor);
        match span.backing() {
            Some(backing) => self.physical_intervals.insert(
                backing.backing_id,
                backing.backing_offset,
                backing.backing_offset.saturating_add(span.len()),
                descriptor,
            ),
            None => self.unresolved.push(descriptor),
        }
        self.spans.insert(descriptor, span);
    }

    pub(crate) fn remove(&mut self, descriptor: &GuestTextureDescriptor) {
        // Removal uses the admission-time span, never a fresh resolution: a mapping that changed
        // underneath a resident must not be able to strand its interval in the index.
        let Some(span) = self.spans.remove(descriptor) else {
            return;
        };
        self.virtual_intervals.remove(0, span.address(), descriptor);
        match span.backing() {
            Some(backing) => {
                self.physical_intervals.remove(
                    backing.backing_id,
                    backing.backing_offset,
                    descriptor,
                );
            }
            None => self.unresolved.retain(|candidate| candidate != descriptor),
        }
    }

    pub(crate) fn overlapping(
        &self,
        query: &ResourceSpan,
        order: SpanQueryOrder,
    ) -> Vec<GuestTextureDescriptor> {
        let overlaps = |resident: &ResourceSpan| match order {
            SpanQueryOrder::ResidentFirst => resident.overlaps(query),
            SpanQueryOrder::QueryFirst => query.overlaps(resident),
        };
        let mut matches = Vec::new();
        let Some(backing) = query.backing() else {
            // An unresolved query reaches the registry fallback for every candidate, so neither
            // interval list can answer it. This is the cost the linear scan always paid.
            for (descriptor, span) in &self.spans {
                if overlaps(span) {
                    push_unique(&mut matches, *descriptor);
                }
            }
            return matches;
        };
        // Both interval lists answer the branches the predicate decides symmetrically: a virtual
        // intersection, and a physical intersection between two resolved identities.
        self.virtual_intervals
            .extend_overlapping(0, query.address(), query.end(), &mut matches);
        self.physical_intervals.extend_overlapping(
            backing.backing_id,
            backing.backing_offset,
            backing.backing_offset.saturating_add(query.len()),
            &mut matches,
        );
        for descriptor in &self.unresolved {
            if overlaps(&self.spans[descriptor]) {
                push_unique(&mut matches, *descriptor);
            }
        }
        matches
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.spans.len()
    }
}

fn push_unique(matches: &mut Vec<GuestTextureDescriptor>, descriptor: GuestTextureDescriptor) {
    // A resident can be reached through both interval lists. The overlapping set is a handful of
    // entries even on pooled heaps, so a scan beats a hash set here.
    if !matches.contains(&descriptor) {
        matches.push(descriptor);
    }
}

/// Half-open intervals grouped by a key, sorted by start, augmented with a running maximum end.
#[derive(Default)]
struct IntervalList {
    entries: Vec<IntervalEntry>,
}

#[derive(Clone, Copy)]
struct IntervalEntry {
    group: u64,
    start: u64,
    end: u64,
    /// Largest `end` among this group's entries up to and including this one.
    ///
    /// Sorting by start alone does not bound a backwards walk, because one pooled allocation can
    /// precede and outlive dozens of small tenants. The running maximum terminates that walk.
    group_max_end: u64,
    descriptor: GuestTextureDescriptor,
}

impl IntervalList {
    fn insert(&mut self, group: u64, start: u64, end: u64, descriptor: GuestTextureDescriptor) {
        let position = self
            .entries
            .partition_point(|entry| (entry.group, entry.start) < (group, start));
        self.entries.insert(
            position,
            IntervalEntry {
                group,
                start,
                end,
                group_max_end: 0,
                descriptor,
            },
        );
        self.refresh_group(group);
    }

    fn remove(&mut self, group: u64, start: u64, descriptor: &GuestTextureDescriptor) {
        let range = self.group_range(group);
        let position = self.entries[range.clone()]
            .iter()
            .position(|entry| entry.start == start && entry.descriptor == *descriptor);
        let Some(position) = position else {
            return;
        };
        self.entries.remove(range.start + position);
        self.refresh_group(group);
    }

    fn extend_overlapping(
        &self,
        group: u64,
        start: u64,
        end: u64,
        matches: &mut Vec<GuestTextureDescriptor>,
    ) {
        if start >= end {
            return;
        }
        let range = self.group_range(group);
        let upper =
            range.start + self.entries[range.clone()].partition_point(|entry| entry.start < end);
        for index in (range.start..upper).rev() {
            let entry = self.entries[index];
            if entry.group_max_end <= start {
                break;
            }
            if entry.end > start {
                push_unique(matches, entry.descriptor);
            }
        }
    }

    fn group_range(&self, group: u64) -> std::ops::Range<usize> {
        let start = self.entries.partition_point(|entry| entry.group < group);
        let end = self.entries.partition_point(|entry| entry.group <= group);
        start..end
    }

    fn refresh_group(&mut self, group: u64) {
        let range = self.group_range(group);
        let mut maximum_end = 0_u64;
        for entry in &mut self.entries[range] {
            maximum_end = maximum_end.max(entry.end);
            entry.group_max_end = maximum_end;
        }
    }
}
