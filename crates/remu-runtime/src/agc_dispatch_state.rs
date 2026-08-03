//! AGC lock-hierarchy level and publication-ticket rework (plan C2 step 3 / B20).
//!
//! Full ownership extraction of `AgcGraphicsState` out of [`crate::ProbeDispatchState`] is a
//! follow-up seam (the type is not `Clone` and has ~70 call sites). Solo S1 lands:
//! 1. the [`LockLevel::AgcState`] hierarchy token used around AGC submission critical sections
//! 2. retirement of the runtime-side GPU publication ticket FIFO for queue-only events, so
//!    publication order no longer silently depends on the process-wide dispatch lock
//!
//! `remu-gpu`'s `PublicationCoordinator` remains the per-queue FIFO authority.

use crate::dispatch_locks::{LockHierarchyGuard, LockLevel};

/// Enters the AGC lock-hierarchy level. Pair with the existing dispatch or submission critical
/// section so debug builds reject lower-then-higher acquisitions.
#[must_use]
pub(crate) fn enter_agc_lock_level() -> LockHierarchyGuard {
    LockHierarchyGuard::enter(LockLevel::AgcState)
}

#[cfg(test)]
mod tests {
    use super::enter_agc_lock_level;
    use crate::dispatch_locks::{LockHierarchyGuard, LockLevel, held_lock_level};

    #[test]
    fn agc_level_sits_below_probe_dispatch_in_the_hierarchy() {
        let _agc = enter_agc_lock_level();
        assert_eq!(held_lock_level(), Some(LockLevel::AgcState));
        let _dispatch = LockHierarchyGuard::enter(LockLevel::ProbeDispatch);
        assert_eq!(held_lock_level(), Some(LockLevel::ProbeDispatch));
    }
}
