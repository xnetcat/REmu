// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{OWNED_RANGE_SET_COARSENED, ResourceSpan};
use std::sync::atomic::Ordering;

/// Runs one set holds before `insert` collapses it to its bounding run.
///
/// A full-surface owner is one run and a partially superseded tenant is at most one run per block
/// row, so the cap is only reached by pathological write patterns.
pub(crate) const RANGE_SET_MAX_RUNS: usize = 256;

/// A set of guest byte intervals, kept sorted, disjoint and coalesced.
///
/// Intervals are `ResourceSpan::range_start`-based offsets, never guest virtual addresses: one
/// physical resource can be mapped at several unrelated VAs, and the algebra must treat those as
/// one set of bytes.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct RangeSet {
    /// `[start, end)`, sorted by start, non-empty, non-adjacent.
    runs: Vec<(u64, u64)>,
}

impl RangeSet {
    pub(crate) fn from_run(start: u64, end: u64) -> Self {
        let mut set = Self::default();
        set.insert_run(start, end);
        set
    }

    /// The whole of `span`, which is what a freshly created resident is the authority for.
    pub(crate) fn for_span(span: &ResourceSpan) -> Self {
        Self::from_run(span.range_start(), span.range_end())
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    pub(crate) fn byte_count(&self) -> u64 {
        self.runs.iter().fold(0_u64, |total, (start, end)| {
            total.saturating_add(end - start)
        })
    }

    pub(crate) fn runs(&self) -> &[(u64, u64)] {
        &self.runs
    }

    /// The single run spanning every run in the set, or `None` when the set is empty.
    pub(crate) fn bounding_run(&self) -> Option<(u64, u64)> {
        Some((self.runs.first()?.0, self.runs.last()?.1))
    }

    /// Union in place.
    ///
    /// This is the only mutation allowed to coarsen: a union widens the claim, and a wider
    /// authority claim can only cost an extra re-materialisation, never stale pixels.
    pub(crate) fn insert(&mut self, other: &Self) {
        self.insert_exact(other);
        self.coarsen_if_fragmented();
    }

    /// Union in place without the authority set's fragmentation coarsening.
    ///
    /// Publication debt uses this operation: widening debt would publish bytes which the GPU did
    /// not author and can overwrite a newer guest write in one of the holes.
    fn insert_exact(&mut self, other: &Self) {
        for &(start, end) in &other.runs {
            self.insert_run(start, end);
        }
    }

    fn coarsen_if_fragmented(&mut self) {
        if self.runs.len() > RANGE_SET_MAX_RUNS
            && let Some((start, end)) = self.bounding_run()
        {
            self.runs.clear();
            self.runs.push((start, end));
            OWNED_RANGE_SET_COARSENED.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn insert_run(&mut self, start: u64, end: u64) {
        if start >= end {
            return;
        }
        // Runs are disjoint and non-adjacent, so both predicates are monotone over the vector.
        let first = self.runs.partition_point(|(_, run_end)| *run_end < start);
        let last = self
            .runs
            .partition_point(|(run_start, _)| *run_start <= end);
        let (mut merged_start, mut merged_end) = (start, end);
        if first < last {
            merged_start = merged_start.min(self.runs[first].0);
            merged_end = merged_end.max(self.runs[last - 1].1);
        }
        self.runs.splice(first..last, [(merged_start, merged_end)]);
    }

    /// Difference in place.
    ///
    /// Never coarsens, in either direction: widening the removal would drop bytes this set still
    /// holds, and widening the result would keep bytes another writer has taken.
    pub(crate) fn subtract(&mut self, other: &Self) {
        for &(start, end) in &other.runs {
            self.subtract_run(start, end);
        }
    }

    fn subtract_run(&mut self, start: u64, end: u64) {
        if start >= end {
            return;
        }
        let first = self.runs.partition_point(|(_, run_end)| *run_end <= start);
        let last = self.runs.partition_point(|(run_start, _)| *run_start < end);
        if first >= last {
            return;
        }
        let mut replacement = Vec::new();
        if self.runs[first].0 < start {
            replacement.push((self.runs[first].0, start));
        }
        if self.runs[last - 1].1 > end {
            replacement.push((end, self.runs[last - 1].1));
        }
        self.runs.splice(first..last, replacement);
    }

    /// Intersection. PR3's publication debt transfer is its only non-test consumer.
    #[allow(dead_code)]
    pub(crate) fn intersect(&self, other: &Self) -> Self {
        let mut runs = Vec::new();
        let (mut left, mut right) = (0_usize, 0_usize);
        while left < self.runs.len() && right < other.runs.len() {
            let start = self.runs[left].0.max(other.runs[right].0);
            let end = self.runs[left].1.min(other.runs[right].1);
            if start < end {
                runs.push((start, end));
            }
            if self.runs[left].1 < other.runs[right].1 {
                left += 1;
            } else {
                right += 1;
            }
        }
        Self { runs }
    }

    pub(crate) fn intersects(&self, other: &Self) -> bool {
        let (mut left, mut right) = (0_usize, 0_usize);
        while left < self.runs.len() && right < other.runs.len() {
            if self.runs[left].0.max(other.runs[right].0)
                < self.runs[left].1.min(other.runs[right].1)
            {
                return true;
            }
            if self.runs[left].1 < other.runs[right].1 {
                left += 1;
            } else {
                right += 1;
            }
        }
        false
    }

    /// True when a single run of this set contains all of `[start, end)`.
    pub(crate) fn contains_run(&self, start: u64, end: u64) -> bool {
        if start >= end {
            return true;
        }
        let index = self.runs.partition_point(|(_, run_end)| *run_end < end);
        self.runs
            .get(index)
            .is_some_and(|(run_start, run_end)| *run_start <= start && end <= *run_end)
    }

    pub(crate) fn covers(&self, span: &ResourceSpan) -> bool {
        self.contains_run(span.range_start(), span.range_end())
    }

    /// True when every run of `other` is contained in this set. Invariants I1 and I2.
    pub(crate) fn contains_all(&self, other: &Self) -> bool {
        other
            .runs
            .iter()
            .all(|(start, end)| self.contains_run(*start, *end))
    }
}

/// Exact GPU publication debt.
///
/// This is deliberately a distinct type from `RangeSet`, which represents authority and may
/// conservatively coarsen after extreme fragmentation. Debt never coarsens: separated runs remain
/// separated regardless of count.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct GpuDebtSet(RangeSet);

impl GpuDebtSet {
    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(crate) fn byte_count(&self) -> u64 {
        self.0.byte_count()
    }

    pub(crate) fn runs(&self) -> &[(u64, u64)] {
        self.0.runs()
    }

    pub(crate) fn insert(&mut self, other: &RangeSet) {
        self.0.insert_exact(other);
    }

    pub(crate) fn subtract(&mut self, other: &RangeSet) {
        self.0.subtract(other);
    }

    pub(crate) fn intersect(&self, other: &RangeSet) -> RangeSet {
        self.0.intersect(other)
    }

    pub(crate) fn as_ranges(&self) -> &RangeSet {
        &self.0
    }

    pub(crate) fn contains_all(&self, other: &Self) -> bool {
        self.0.contains_all(&other.0)
    }

    /// Rebase a whole-allocation debt when a layout migration moves every byte.
    ///
    /// Sparse debt cannot be translated without a texel-to-byte proof for both layouts. Refuse it
    /// rather than widening the holes to the new allocation span.
    pub(crate) fn rebase_whole_span(
        &self,
        old_span: &RangeSet,
        new_span: RangeSet,
    ) -> Result<Self, &'static str> {
        if self.is_empty() {
            return Ok(Self::default());
        }
        if self.as_ranges() != old_span {
            return Err("sparse GPU debt cannot be rebased across a layout change");
        }
        Ok(Self(new_span))
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuDebtSet, RANGE_SET_MAX_RUNS, RangeSet};

    fn set(runs: &[(u64, u64)]) -> RangeSet {
        let mut set = RangeSet::default();
        for &(start, end) in runs {
            set.insert_run(start, end);
        }
        set
    }

    #[test]
    fn an_empty_or_inverted_run_is_not_a_member() {
        assert!(RangeSet::default().is_empty());
        assert!(RangeSet::from_run(8, 8).is_empty());
        assert!(RangeSet::from_run(16, 8).is_empty());
        assert_eq!(RangeSet::default().bounding_run(), None);
        assert_eq!(RangeSet::default().byte_count(), 0);
    }

    #[test]
    fn insertion_coalesces_overlapping_touching_and_nested_runs() {
        // Disjoint runs stay separate, in start order regardless of insertion order.
        assert_eq!(set(&[(20, 30), (0, 10)]).runs(), [(0, 10), (20, 30)]);
        // Adjacent runs coalesce: [0,10) ∪ [10,20) is one interval, not two.
        assert_eq!(set(&[(0, 10), (10, 20)]).runs(), [(0, 20)]);
        // Overlapping runs coalesce to their union.
        assert_eq!(set(&[(0, 10), (5, 20)]).runs(), [(0, 20)]);
        // A nested run changes nothing.
        assert_eq!(set(&[(0, 100), (40, 50)]).runs(), [(0, 100)]);
        // One insertion can bridge several existing runs.
        assert_eq!(
            set(&[(0, 10), (20, 30), (40, 50), (5, 45)]).runs(),
            [(0, 50)]
        );
        assert_eq!(set(&[(0, 10), (20, 30)]).byte_count(), 20);
        assert_eq!(set(&[(0, 10), (20, 30)]).bounding_run(), Some((0, 30)));
    }

    #[test]
    fn union_of_two_sets_is_the_coalesced_union_of_their_runs() {
        let mut left = set(&[(0, 10), (30, 40)]);
        left.insert(&set(&[(10, 20), (100, 110)]));
        assert_eq!(left.runs(), [(0, 20), (30, 40), (100, 110)]);
    }

    #[test]
    fn subtraction_trims_splits_and_empties_runs() {
        // A subtraction inside a run splits it.
        let mut middle = set(&[(0, 100)]);
        middle.subtract(&set(&[(40, 60)]));
        assert_eq!(middle.runs(), [(0, 40), (60, 100)]);
        // Leading and trailing subtractions trim.
        let mut edges = set(&[(0, 100)]);
        edges.subtract(&set(&[(0, 10), (90, 100)]));
        assert_eq!(edges.runs(), [(10, 90)]);
        // A disjoint subtraction is a no-op, which is the pooled-neighbour case.
        let mut disjoint = set(&[(0, 100)]);
        disjoint.subtract(&set(&[(200, 300)]));
        assert_eq!(disjoint.runs(), [(0, 100)]);
        // A covering subtraction empties the set, which is what demotion keys on.
        let mut covered = set(&[(10, 20), (30, 40)]);
        covered.subtract(&set(&[(0, 100)]));
        assert!(covered.is_empty());
        // Touching but not overlapping removes nothing.
        let mut touching = set(&[(10, 20)]);
        touching.subtract(&set(&[(0, 10), (20, 30)]));
        assert_eq!(touching.runs(), [(10, 20)]);
    }

    #[test]
    fn intersection_keeps_only_shared_bytes() {
        let left = set(&[(0, 10), (20, 30), (40, 50)]);
        let right = set(&[(5, 25), (45, 100)]);
        assert_eq!(left.intersect(&right).runs(), [(5, 10), (20, 25), (45, 50)]);
        assert!(left.intersects(&right));
        assert!(right.intersects(&left));

        let apart = set(&[(10, 20), (30, 40)]);
        assert!(left.intersect(&apart).is_empty());
        assert!(!left.intersects(&apart));
        // Adjacency is not intersection: [0,10) and [10,20) share no byte.
        assert!(!set(&[(0, 10)]).intersects(&set(&[(10, 20)])));
        assert!(!RangeSet::default().intersects(&left));
    }

    #[test]
    fn containment_requires_a_single_covering_run() {
        let set = set(&[(0, 10), (10, 20), (40, 50)]);
        assert!(set.contains_run(0, 20));
        assert!(set.contains_run(5, 15));
        assert!(set.contains_run(40, 50));
        assert!(!set.contains_run(0, 21));
        assert!(!set.contains_run(15, 45));
        assert!(!set.contains_run(60, 70));
        // A degenerate query is vacuously contained.
        assert!(set.contains_run(1_000, 1_000));
        assert!(set.contains_all(&RangeSet::from_run(5, 15)));
        assert!(!set.contains_all(&RangeSet::from_run(15, 45)));
    }

    #[test]
    fn exceeding_the_run_cap_widens_the_claim_to_its_bounding_run() {
        let mut set = RangeSet::default();
        // Alternating runs cannot coalesce, so the cap is the only thing that bounds them.
        for index in 0..u64::try_from(RANGE_SET_MAX_RUNS).expect("cap fits u64") {
            set.insert(&RangeSet::from_run(index * 4, index * 4 + 2));
        }
        assert_eq!(set.runs().len(), RANGE_SET_MAX_RUNS);
        let before = set.byte_count();

        let last = u64::try_from(RANGE_SET_MAX_RUNS).expect("cap fits u64") * 4;
        set.insert(&RangeSet::from_run(last, last + 2));
        assert_eq!(set.runs(), [(0, last + 2)]);
        // Collapsing must widen: a narrower claim would leave superseded bytes readable.
        assert!(set.byte_count() > before);
        assert!(set.contains_run(2, 4), "the gaps are claimed, not dropped");
    }

    #[test]
    fn subtraction_never_coarsens_even_past_the_run_cap() {
        // Widening a difference would keep bytes another writer took, so `subtract` has no cap.
        let cap = u64::try_from(RANGE_SET_MAX_RUNS).expect("cap fits u64");
        let mut set = RangeSet::from_run(0, (cap + 8) * 4);
        for index in 0..=cap {
            set.subtract(&RangeSet::from_run(index * 4 + 2, index * 4 + 4));
        }
        assert!(set.runs().len() > RANGE_SET_MAX_RUNS);
        assert!(!set.contains_run(2, 4));
    }

    #[test]
    fn gpu_debt_never_coarsens_past_the_authority_run_cap() {
        let mut debt = GpuDebtSet::default();
        let run_count = RANGE_SET_MAX_RUNS + 65;
        for index in 0..u64::try_from(run_count).expect("run count fits u64") {
            debt.insert(&RangeSet::from_run(index * 4, index * 4 + 2));
        }

        assert_eq!(debt.runs().len(), run_count);
        assert_eq!(debt.byte_count(), u64::try_from(run_count).unwrap() * 2);
        for index in 0..u64::try_from(run_count).unwrap() {
            assert!(debt.as_ranges().contains_run(index * 4, index * 4 + 2));
            assert!(!debt.as_ranges().contains_run(index * 4 + 2, index * 4 + 4));
        }
    }

    #[test]
    fn sparse_gpu_debt_refuses_whole_span_layout_rebase_without_widening() {
        let mut debt = GpuDebtSet::default();
        let run_count = RANGE_SET_MAX_RUNS + 65;
        for index in 0..u64::try_from(run_count).unwrap() {
            debt.insert(&RangeSet::from_run(index * 4, index * 4 + 2));
        }
        let before = debt.clone();
        let old_span = RangeSet::from_run(0, u64::try_from(run_count).unwrap() * 4);
        let new_span = RangeSet::from_run(0, u64::try_from(run_count + 128).unwrap() * 4);

        assert!(debt.rebase_whole_span(&old_span, new_span).is_err());
        assert_eq!(debt, before, "a refused promotion leaves exact debt unchanged");
        assert!(!debt.as_ranges().contains_run(2, 4));
    }
}
