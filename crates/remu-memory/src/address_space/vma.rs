// SPDX-License-Identifier: GPL-2.0-or-later

//! Virtual memory areas: the address space's per-range reservation and mapping records.

use remu_core::GuestAddress;

use super::{GuestPhysicalBacking, GuestProtection, GuestVmaInfo, GuestVmaState};

#[derive(Clone, Debug)]
pub(super) enum VmaKind {
    Reserved,
    /// Commit-on-protect, flexible, or former-anonymous memory backed by the space's
    /// virtually-keyed backing (Phase 1G: every CPU-private map must remain Metal-backable).
    Committed(GuestPhysicalBacking),
    Shared(GuestPhysicalBacking),
}

impl VmaKind {
    pub(super) const fn state(&self) -> GuestVmaState {
        match self {
            Self::Reserved => GuestVmaState::Reserved,
            Self::Committed(_) => GuestVmaState::Committed,
            Self::Shared(_) => GuestVmaState::Shared,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct Vma {
    pub(super) start: u64,
    pub(super) end: u64,
    pub(super) protection: GuestProtection,
    pub(super) kind: VmaKind,
    pub(super) backing_offset: u64,
    pub(super) metadata: u64,
    pub(super) mapping_id: u64,
}

impl Vma {
    pub(super) fn info(&self) -> GuestVmaInfo {
        let (backing_id, backing_offset) = match &self.kind {
            VmaKind::Shared(backing) | VmaKind::Committed(backing) => {
                (Some(backing.inner.id), self.backing_offset)
            }
            VmaKind::Reserved => (None, 0),
        };
        GuestVmaInfo {
            start: GuestAddress::new(self.start),
            size: self.end - self.start,
            state: self.kind.state(),
            protection: self.protection,
            backing_id,
            backing_offset,
            metadata: self.metadata,
        }
    }

    pub(super) fn can_merge(&self, next: &Self) -> bool {
        if self.end != next.start
            || self.protection != next.protection
            || self.metadata != next.metadata
            || self.mapping_id != next.mapping_id
        {
            return false;
        }
        match (&self.kind, &next.kind) {
            (VmaKind::Reserved, VmaKind::Reserved) => true,
            (VmaKind::Shared(left), VmaKind::Shared(right))
            | (VmaKind::Committed(left), VmaKind::Committed(right)) => {
                left.id() == right.id()
                    && self.backing_offset + (self.end - self.start) == next.backing_offset
            }
            _ => false,
        }
    }
}
