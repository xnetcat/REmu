//! Per-dispatch-index route table (plan C1 / B13 second PR).
//!
//! The dispatch index is dense and stable for the process lifetime. On the first call the NID is
//! resolved through today's cascade and the winning family is recorded; every subsequent call
//! switches on the cached `u16` and hands the NID only to that family's handler.
//!
//! Routes are recorded on the *return* path after a family claims the NID — several families are
//! entered conditionally (`Option`-returning dispatchers).

#![allow(dead_code)]

use std::sync::atomic::{AtomicU16, Ordering};

/// Family that claimed a dispatch index. Zero means "unresolved".
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub(crate) enum ImportRoute {
    Unresolved = 0,
    Leaf = 1,
    Sync = 2,
    Timer = 3,
    Audio = 4,
    Pad = 5,
    Stateless = 6,
    Fd = 7,
    Path = 8,
    AmprOrdered = 9,
    AmprSerialized = 10,
    Stateful = 11,
    Fiber = 12,
    Platform = 13,
    Ajm = 14,
    Unimplemented = 15,
}

impl ImportRoute {
    pub(crate) const fn from_u16(value: u16) -> Option<Self> {
        match value {
            0 => Some(Self::Unresolved),
            1 => Some(Self::Leaf),
            2 => Some(Self::Sync),
            3 => Some(Self::Timer),
            4 => Some(Self::Audio),
            5 => Some(Self::Pad),
            6 => Some(Self::Stateless),
            7 => Some(Self::Fd),
            8 => Some(Self::Path),
            9 => Some(Self::AmprOrdered),
            10 => Some(Self::AmprSerialized),
            11 => Some(Self::Stateful),
            12 => Some(Self::Fiber),
            13 => Some(Self::Platform),
            14 => Some(Self::Ajm),
            15 => Some(Self::Unimplemented),
            _ => None,
        }
    }

    pub(crate) const fn as_u16(self) -> u16 {
        self as u16
    }
}

/// Dense route cache parallel to [`crate::ImportTelemetry::nids`].
#[derive(Debug)]
pub(crate) struct ImportRouteTable {
    routes: Box<[AtomicU16]>,
}

impl ImportRouteTable {
    pub(crate) fn new(count: usize) -> Self {
        Self {
            routes: (0..count)
                .map(|_| AtomicU16::new(ImportRoute::Unresolved.as_u16()))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    pub(crate) fn load(&self, index: usize) -> ImportRoute {
        let raw = self
            .routes
            .get(index)
            .map(|slot| slot.load(Ordering::Acquire))
            .unwrap_or(0);
        ImportRoute::from_u16(raw).unwrap_or(ImportRoute::Unresolved)
    }

    /// Records the winning family after the handler claimed the NID.
    ///
    /// Under `debug_assertions`, a second claim that disagrees with the cached route panics —
    /// that is the alias-collapse / single-owner invariant for a dispatch index.
    pub(crate) fn store(&self, index: usize, route: ImportRoute) {
        let Some(slot) = self.routes.get(index) else {
            return;
        };
        let previous = slot.swap(route.as_u16(), Ordering::AcqRel);
        if cfg!(debug_assertions)
            && previous != ImportRoute::Unresolved.as_u16()
            && previous != route.as_u16()
        {
            panic!(
                "import route for dispatch index {index} changed from {previous} to {}",
                route.as_u16()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ImportRoute, ImportRouteTable};

    #[test]
    fn route_table_starts_unresolved_and_records_the_winning_family() {
        let table = ImportRouteTable::new(4);
        assert_eq!(table.load(2), ImportRoute::Unresolved);
        table.store(2, ImportRoute::Path);
        assert_eq!(table.load(2), ImportRoute::Path);
        // Idempotent re-store of the same family is fine.
        table.store(2, ImportRoute::Path);
        assert_eq!(table.load(2), ImportRoute::Path);
    }

    #[test]
    #[should_panic(expected = "import route for dispatch index")]
    fn route_table_rejects_a_conflicting_family_under_debug_assertions() {
        let table = ImportRouteTable::new(1);
        table.store(0, ImportRoute::Fd);
        table.store(0, ImportRoute::Path);
    }
}
