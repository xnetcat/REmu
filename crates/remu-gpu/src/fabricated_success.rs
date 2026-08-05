// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Process-wide ledger of handlers that returned success without doing the work.
//!
//! The ABI-gap reporter can only see unimplemented imports, failed imports, declined shaders,
//! and SIGILL recoveries. A handler that returns `0` matches none of those by construction.
//! Every inert / fabricated-success arm calls [`note_inert`] so the end-of-run `[REMu][REPORT]`
//! block can name them. Keys are usually NIDs; internal paths use a stable `subsystem:name` label.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

static FABRICATED_SUCCESS: OnceLock<Mutex<BTreeMap<String, u64>>> = OnceLock::new();

/// Records one fabricated-success / inert return for `key`.
///
/// `key` is usually an 11-character NID, or a `subsystem:name` label for non-import paths.
pub fn note_inert(key: &str) {
    *FABRICATED_SUCCESS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .entry(key.to_owned())
        .or_insert(0) += 1;
}

/// Snapshot of every key that has been noted at least once, ordered by key.
pub fn fabricated_success_counts() -> Vec<(String, u64)> {
    let Some(counts) = FABRICATED_SUCCESS.get() else {
        return Vec::new();
    };
    let counts = counts
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    counts
        .iter()
        .map(|(key, count)| (key.clone(), *count))
        .collect()
}

/// Sum of all fabricated-success observations (for the ABI-gap fingerprint).
pub fn fabricated_success_total() -> u64 {
    fabricated_success_counts()
        .into_iter()
        .map(|(_, count)| count)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_inert_accumulates_per_key() {
        // Keys unique to this test so parallel crates don't collide on the process-wide map.
        note_inert("test:fabricated-success-a");
        note_inert("test:fabricated-success-a");
        note_inert("test:fabricated-success-b");
        let counts = fabricated_success_counts();
        let a = counts
            .iter()
            .find(|(key, _)| key == "test:fabricated-success-a")
            .map(|(_, count)| *count)
            .unwrap_or(0);
        let b = counts
            .iter()
            .find(|(key, _)| key == "test:fabricated-success-b")
            .map(|(_, count)| *count)
            .unwrap_or(0);
        assert!(a >= 2, "expected at least two notes for key a, got {a}");
        assert!(b >= 1, "expected at least one note for key b, got {b}");
    }
}
