// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! B5 end state: GPU writes land in guest RAM at execution time; "writeback" shrinks to
//! lease-validated residency reconciliation.
//!
//! Today most colour/depth traffic still lives in Metal-private storage and is published
//! through the artefact writeback path (`remu-writeback-is-an-artefact`). This module
//! classifies each proposed publish so new call sites cannot silently grow the wrong class.
//! Full zero-copy execution-time landing waits on B2 tiler + shared-backing proof (PROT_READ
//! watch precondition) — until then reconciliation stays LOUD when the lease is missing.

#![allow(dead_code)] // Classification helpers until writeback call sites migrate.

use remu_memory::GuestRangeLease;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EndStateWriteClass {
    /// Guest named the address and will read it (flip target, DMA source). Legitimate.
    GuestObservable,
    /// REmu bookkeeping wants bytes in RAM with no guest request — must not publish.
    RemuBookkeeping,
    /// Lease-validated residency reconciliation after GPU execution touched shared backing.
    LeaseValidatedReconciliation,
}

/// Classifies a proposed GPU→guest publish. Bookkeeping publishes must stay deleted.
pub(crate) fn classify_end_state_write(
    guest_named_address: bool,
    has_live_lease: bool,
    guest_will_read_now: bool,
) -> EndStateWriteClass {
    if guest_named_address && guest_will_read_now {
        return EndStateWriteClass::GuestObservable;
    }
    if has_live_lease && guest_named_address {
        return EndStateWriteClass::LeaseValidatedReconciliation;
    }
    EndStateWriteClass::RemuBookkeeping
}

/// True when a lease proves the range is still the guest's and shared backing is the
/// execution-time destination — the B5 end-state publish path.
pub(crate) fn lease_validated_reconciliation_ready(lease: Option<&GuestRangeLease>) -> bool {
    // `live_va_matches` re-proves coverage under the registry; a missing lease is never ready.
    lease.is_some_and(GuestRangeLease::live_va_matches)
}

/// LOUD refusal for bookkeeping publishes. Callers must not invent a success path.
pub(crate) fn refuse_bookkeeping_writeback(class: EndStateWriteClass) -> Result<(), String> {
    match class {
        EndStateWriteClass::RemuBookkeeping => Err(
            "B5 end-state: refusing RemuBookkeeping writeback — GPU writes must land in guest RAM at execution or publish only lease-validated / guest-observable ranges".to_owned(),
        ),
        EndStateWriteClass::GuestObservable | EndStateWriteClass::LeaseValidatedReconciliation => {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guest_observable_flip_target_is_allowed() {
        assert_eq!(
            classify_end_state_write(true, false, true),
            EndStateWriteClass::GuestObservable
        );
        refuse_bookkeeping_writeback(EndStateWriteClass::GuestObservable).unwrap();
    }

    #[test]
    fn bookkeeping_without_lease_stays_loud() {
        let class = classify_end_state_write(false, false, false);
        assert_eq!(class, EndStateWriteClass::RemuBookkeeping);
        let err = refuse_bookkeeping_writeback(class).expect_err("bookkeeping must fail loud");
        assert!(err.contains("refusing RemuBookkeeping"), "{err}");
    }

    #[test]
    fn lease_validated_reconciliation_requires_a_named_range() {
        assert_eq!(
            classify_end_state_write(true, true, false),
            EndStateWriteClass::LeaseValidatedReconciliation
        );
        assert!(!lease_validated_reconciliation_ready(None));
    }
}
