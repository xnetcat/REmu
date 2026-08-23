// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::stats::{
    RESIDENT_RETIRE_CLEAN, RESIDENT_RETIRE_DIRTY_REFUSED, RESIDENT_RETIRE_PUBLISHED,
    RESIDENT_RETIRE_TRANSFERRED,
};
use super::{GpuDebtSet, GuestTextureDescriptor, ResidentGuestTexture, ResourceSpan};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ResidentRetireReason(pub(crate) &'static str);

pub(crate) struct PublishReceipt {
    source: GuestTextureDescriptor,
    source_span: Option<ResourceSpan>,
    covered: GpuDebtSet,
}

pub(crate) struct TransferReceipt {
    source: GuestTextureDescriptor,
    source_span: Option<ResourceSpan>,
    target: GuestTextureDescriptor,
    covered: GpuDebtSet,
    ordered_content_version: u64,
}

pub(crate) enum RetireDisposition {
    Clean { reason: ResidentRetireReason },
    Published { receipt: PublishReceipt },
    TransferredTo { receipt: TransferReceipt },
}

#[derive(Debug)]
pub(crate) enum ResidentRetireError {
    Dirty {
        descriptor: GuestTextureDescriptor,
        reason: &'static str,
        debt_bytes: u64,
        debt_runs: usize,
    },
    InvalidReceipt(&'static str),
}

impl fmt::Display for ResidentRetireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dirty {
                descriptor,
                reason,
                debt_bytes,
                debt_runs,
            } => write!(
                formatter,
                "refused dirty resident retirement address={:#x} reason={reason} debt_bytes={debt_bytes:#x} debt_runs={debt_runs}",
                descriptor.address
            ),
            Self::InvalidReceipt(reason) => {
                write!(formatter, "invalid resident retirement receipt: {reason}")
            }
        }
    }
}

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
        assert!(
            !self.textures.contains_key(&descriptor),
            "resident replacement must retire or take the previous allocation first"
        );
        // `resident.span` is the admission-time extent of the complete allocation (all mip
        // levels and array layers). Recomputing a span from the descriptor here would index only
        // level zero and could also bind the interval to a mapping installed after admission.
        self.index.insert(descriptor, resident.span);
        self.textures.insert(descriptor, resident)
    }

    /// Builds and atomically installs a replacement while the original resident and index entry
    /// remain live. A failed builder leaves both source-of-truth structures unchanged.
    pub(crate) fn try_replace<E>(
        &mut self,
        descriptor: &GuestTextureDescriptor,
        build: impl FnOnce(&ResidentGuestTexture) -> Result<ResidentGuestTexture, E>,
    ) -> Result<bool, E> {
        let Some(current) = self.textures.get(descriptor) else {
            return Ok(false);
        };
        let replacement = build(current)?;
        assert_eq!(
            replacement.descriptor, *descriptor,
            "transactional replacement cannot change the descriptor key"
        );
        self.index.remove(descriptor);
        self.index.insert(*descriptor, replacement.span);
        self.textures.insert(*descriptor, replacement);
        Ok(true)
    }

    /// The only operation allowed to permanently delete a resident allocation.
    pub(crate) fn retire(
        &mut self,
        descriptor: &GuestTextureDescriptor,
        disposition: RetireDisposition,
    ) -> Result<Option<ResidentGuestTexture>, ResidentRetireError> {
        let Some(resident) = self.textures.get(descriptor) else {
            return Ok(None);
        };
        match &disposition {
            RetireDisposition::Clean { reason } => {
                if !resident.gpu_authored.is_empty() {
                    RESIDENT_RETIRE_DIRTY_REFUSED.fetch_add(1, Ordering::Relaxed);
                    return Err(ResidentRetireError::Dirty {
                        descriptor: *descriptor,
                        reason: reason.0,
                        debt_bytes: resident.gpu_authored.byte_count(),
                        debt_runs: resident.gpu_authored.runs().len(),
                    });
                }
            }
            RetireDisposition::Published { receipt } => {
                if receipt.source != *descriptor
                    || receipt.source_span != resident.span
                    || !receipt.covered.contains_all(&resident.gpu_authored)
                {
                    return Err(ResidentRetireError::InvalidReceipt(
                        "publication receipt does not cover the source debt",
                    ));
                }
            }
            RetireDisposition::TransferredTo { receipt } => {
                if receipt.source != *descriptor || receipt.target == *descriptor {
                    return Err(ResidentRetireError::InvalidReceipt(
                        "transfer receipt names an invalid source or target",
                    ));
                }
                let Some(target) = self.textures.get(&receipt.target) else {
                    return Err(ResidentRetireError::InvalidReceipt(
                        "transfer target is not live",
                    ));
                };
                if receipt.source_span != resident.span
                    || !resident
                        .span
                        .zip(target.span)
                        .is_some_and(|(source, target)| source.shares_offset_space(&target))
                    || target.content_version < receipt.ordered_content_version
                    || !receipt.covered.contains_all(&resident.gpu_authored)
                    || !target.gpu_authored.contains_all(&resident.gpu_authored)
                    || !target.owned.contains_all(resident.gpu_authored.as_ranges())
                {
                    return Err(ResidentRetireError::InvalidReceipt(
                        "transfer target does not hold ordered authority and debt",
                    ));
                }
            }
        }
        let retired = self.remove(descriptor);
        match disposition {
            RetireDisposition::Clean { .. } => {
                RESIDENT_RETIRE_CLEAN.fetch_add(1, Ordering::Relaxed);
            }
            RetireDisposition::Published { .. } => {
                RESIDENT_RETIRE_PUBLISHED.fetch_add(1, Ordering::Relaxed);
            }
            RetireDisposition::TransferredTo { .. } => {
                RESIDENT_RETIRE_TRANSFERRED.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(retired)
    }

    pub(crate) fn retire_clean(
        &mut self,
        descriptor: &GuestTextureDescriptor,
        reason: &'static str,
    ) -> Result<Option<ResidentGuestTexture>, String> {
        self.retire(
            descriptor,
            RetireDisposition::Clean {
                reason: ResidentRetireReason(reason),
            },
        )
        .map_err(|error| error.to_string())
    }

    #[cfg(test)]
    pub(crate) fn test_published_disposition(
        &self,
        source: GuestTextureDescriptor,
        covered: GpuDebtSet,
    ) -> RetireDisposition {
        RetireDisposition::Published {
            receipt: PublishReceipt {
                source,
                source_span: self.get(&source).and_then(|resident| resident.span),
                covered,
            },
        }
    }

    #[cfg(test)]
    pub(crate) fn test_transferred_disposition(
        &self,
        source: GuestTextureDescriptor,
        target: GuestTextureDescriptor,
        covered: GpuDebtSet,
        ordered_content_version: u64,
    ) -> RetireDisposition {
        RetireDisposition::TransferredTo {
            receipt: TransferReceipt {
                source,
                source_span: self.get(&source).and_then(|resident| resident.span),
                target,
                covered,
                ordered_content_version,
            },
        }
    }

    fn remove(
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

    pub(crate) fn insert(
        &mut self,
        descriptor: GuestTextureDescriptor,
        span: Option<ResourceSpan>,
    ) {
        self.remove(&descriptor);
        // A resident without an admission-time span can never participate in an indexed query.
        // Do not re-resolve it here: doing so would associate an old resident with a new mapping.
        let Some(span) = span else {
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
