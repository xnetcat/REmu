//! Path roots behind a snapshot cell (plan C2 step 1 / B20).
//!
//! `game_root`, `save_root`, `save_mounts`, and the static [`VfsState`] mounts are read by
//! thousands of failing path probes per second on Unreal titles. Keeping them behind the
//! process-wide dispatch write lock excludes AGC submission for each probe. An `Arc` snapshot
//! published through a short-held `RwLock` lets path syscalls run in the pre-lock tier (same role
//! as `ArcSwap` in the plan). The VFS table is included because `/app0`/`/download0`/`/temp0`
//! resolution is part of every `open`/`stat`/`mkdir`/`rmdir`.

#![allow(dead_code)] // snapshot helpers land ahead of full call-site migration

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

use crate::SaveDataMount;
use crate::vfs::VfsState;

/// Immutable path-resolution roots for one published generation.
#[derive(Clone, Debug, Default)]
pub(crate) struct PathRoots {
    pub(crate) game_root: PathBuf,
    pub(crate) save_root: PathBuf,
    pub(crate) save_mounts: BTreeMap<String, SaveDataMount>,
    pub(crate) vfs: VfsState,
    /// Save-data transaction resources that have completed Prepare (writability gate).
    pub(crate) prepared_save_resources: BTreeSet<u32>,
}

impl PathRoots {
    pub(crate) fn with_game_root(game_root: impl Into<PathBuf>, save_root: impl Into<PathBuf>) -> Self {
        Self {
            game_root: game_root.into(),
            save_root: save_root.into(),
            save_mounts: BTreeMap::new(),
            vfs: VfsState::default(),
            prepared_save_resources: BTreeSet::new(),
        }
    }
}

static PATH_ROOTS: OnceLock<RwLock<Arc<PathRoots>>> = OnceLock::new();

fn cell() -> &'static RwLock<Arc<PathRoots>> {
    PATH_ROOTS.get_or_init(|| RwLock::new(Arc::new(PathRoots::default())))
}

/// Publishes a new path-roots snapshot. Callers that mutate mounts under the dispatch lock must
/// republish after the mutation so pre-lock path syscalls observe the change.
pub(crate) fn publish_path_roots(roots: PathRoots) {
    let _hierarchy = crate::dispatch_locks::LockHierarchyGuard::enter(
        crate::dispatch_locks::LockLevel::PathRoots,
    );
    let mut guard = cell()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *guard = Arc::new(roots);
}

/// Loads the current path-roots snapshot without taking the dispatch lock.
#[must_use]
pub(crate) fn load_path_roots() -> Arc<PathRoots> {
    cell()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// Convenience for diagnostics and tests.
#[must_use]
pub(crate) fn path_roots_game_root() -> PathBuf {
    load_path_roots().game_root.clone()
}

/// True when `candidate` is under the published game root (prefix check only).
#[must_use]
pub(crate) fn path_is_under_game_root(candidate: &Path) -> bool {
    let roots = load_path_roots();
    candidate.starts_with(&roots.game_root)
}

/// Republishes path roots from a live [`crate::ProbeDispatchState`] after a mount mutation.
pub(crate) fn republish_from_dispatch(state: &crate::ProbeDispatchState) {
    publish_path_roots(PathRoots {
        game_root: state.game_root.clone(),
        save_root: state.save_root.clone(),
        save_mounts: state.save_mounts.clone(),
        vfs: state.vfs.clone(),
        prepared_save_resources: state.prepared_save_resources.clone(),
    });
}

#[cfg(test)]
mod tests {
    use super::{PathRoots, load_path_roots, publish_path_roots};
    use std::path::PathBuf;

    #[test]
    fn published_path_roots_are_visible_without_the_dispatch_lock() {
        publish_path_roots(PathRoots::with_game_root(
            PathBuf::from("/tmp/remu-c2-game"),
            PathBuf::from("/tmp/remu-c2-save"),
        ));
        let snap = load_path_roots();
        assert_eq!(snap.game_root, PathBuf::from("/tmp/remu-c2-game"));
        assert_eq!(snap.save_root, PathBuf::from("/tmp/remu-c2-save"));
        assert!(snap.save_mounts.is_empty());
    }
}
