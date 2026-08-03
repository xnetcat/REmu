//! AGC graphics/compute/shader registry behind its own lock (plan C2 step 3 / B20).
//!
//! Sole owner of [`AgcGraphicsState`], compute-queue map, and [`AgcShaderRegistry`]. Call sites that
//! already hold [`LockLevel::ProbeDispatch`] may nest this lock; AgcState critical sections must
//! never acquire ProbeDispatch. Publication ticket FIFO was retired earlier — `remu-gpu`'s
//! `PublicationCoordinator` remains the per-queue FIFO authority.

#![allow(dead_code)] // Read/with/reset helpers land with call-site migration off ProbeDispatch.
use std::collections::BTreeMap;
use std::sync::{OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::dispatch_locks::{LockHierarchyGuard, LockLevel, held_lock_level};
use crate::{AgcGraphicsState, AgcShaderRegistry};

#[derive(Debug, Default)]
pub(crate) struct AgcState {
    pub(crate) graphics: AgcGraphicsState,
    pub(crate) compute_queues: BTreeMap<u32, AgcGraphicsState>,
    pub(crate) shader_registry: AgcShaderRegistry,
}

static AGC_STATE: OnceLock<RwLock<AgcState>> = OnceLock::new();

fn cell() -> &'static RwLock<AgcState> {
    AGC_STATE.get_or_init(|| RwLock::new(AgcState::default()))
}

/// Write lock for AgcState. Nests without hierarchy re-entry when ProbeDispatch is already held.
pub(crate) fn lock_agc_for_write() -> (Option<LockHierarchyGuard>, RwLockWriteGuard<'static, AgcState>) {
    let hierarchy = match held_lock_level() {
        Some(LockLevel::ProbeDispatch) => None,
        _ => Some(LockHierarchyGuard::enter(LockLevel::AgcState)),
    };
    let guard = cell()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    (hierarchy, guard)
}

pub(crate) fn lock_agc_for_read() -> (Option<LockHierarchyGuard>, RwLockReadGuard<'static, AgcState>) {
    let hierarchy = match held_lock_level() {
        Some(LockLevel::ProbeDispatch) => None,
        _ => Some(LockHierarchyGuard::enter(LockLevel::AgcState)),
    };
    let guard = cell()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    (hierarchy, guard)
}

pub(crate) fn reset_agc_state() {
    let mut state = cell()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *state = AgcState::default();
}

pub(crate) fn with_agc_mut<R>(f: impl FnOnce(&mut AgcState) -> R) -> R {
    let (_h, mut agc) = lock_agc_for_write();
    f(&mut agc)
}

pub(crate) fn with_agc<R>(f: impl FnOnce(&AgcState) -> R) -> R {
    let (_h, agc) = lock_agc_for_read();
    f(&agc)
}

/// Enters the AGC lock-hierarchy level without taking the AgcState RwLock (tests / publication).
#[must_use]
pub(crate) fn enter_agc_lock_level() -> LockHierarchyGuard {
    LockHierarchyGuard::enter(LockLevel::AgcState)
}

#[cfg(test)]
mod tests {
    use super::{enter_agc_lock_level, lock_agc_for_write, reset_agc_state};
    use crate::dispatch_locks::{LockHierarchyGuard, LockLevel, held_lock_level};

    #[test]
    fn agc_level_sits_below_probe_dispatch_in_the_hierarchy() {
        let _agc = enter_agc_lock_level();
        assert_eq!(held_lock_level(), Some(LockLevel::AgcState));
        let _dispatch = LockHierarchyGuard::enter(LockLevel::ProbeDispatch);
        assert_eq!(held_lock_level(), Some(LockLevel::ProbeDispatch));
    }

    #[test]
    fn agc_state_may_nest_under_probe_dispatch() {
        reset_agc_state();
        let _dispatch = LockHierarchyGuard::enter(LockLevel::ProbeDispatch);
        let (_h, mut agc) = lock_agc_for_write();
        agc.compute_queues.insert(1, crate::AgcGraphicsState::default());
        assert!(agc.compute_queues.contains_key(&1));
    }
}
