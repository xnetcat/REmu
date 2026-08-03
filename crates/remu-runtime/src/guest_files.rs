// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! The guest filesystem: console path resolution, the descriptor table and the file syscalls.
//!
//! A dump is a case-preserving copy of a case-insensitive console filesystem, so every guest path
//! is resolved through a cache that remembers the one host name a console path maps to. The
//! descriptor table and the `stat` layouts below are what the guest sees once a path resolves.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::ffi::CString;
use std::io::{Read, Seek, Write};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, RwLock};

use crate::{
    ProbeDispatchState, SCE_KERNEL_ERROR_ALREADY_EXISTS, SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR,
    SCE_KERNEL_ERROR_INVALID_ARGUMENT, SCE_KERNEL_ERROR_MEMORY_FAULT,
    SCE_KERNEL_ERROR_NOT_DIRECTORY, SCE_KERNEL_ERROR_NOT_FOUND, SCE_KERNEL_ERROR_PERMISSION_DENIED,
    SCE_KERNEL_STAT_SIZE, low_i32, low_u32, read_guest_cstring, sce_kernel_error, sce_kernel_io_error,
    signed_i32_return, trace_guest_vfs, trace_guest_vfs_miss, write_guest_bytes, write_guest_u64,
};

pub(crate) static KERNEL_FILES: OnceLock<Mutex<KernelFileState>> = OnceLock::new();

#[derive(Debug)]
pub(crate) struct KernelFileState {
    pub(crate) next_fd: i32,
    pub(crate) files: BTreeMap<i32, std::fs::File>,
    pub(crate) directories: BTreeMap<i32, KernelDirectory>,
    pub(crate) next_file_number: u32,
}

impl Default for KernelFileState {
    fn default() -> Self {
        Self {
            next_fd: 3,
            files: BTreeMap::new(),
            directories: BTreeMap::new(),
            next_file_number: 1,
        }
    }
}

/// A directory opened by the guest, holding its entries already packed as console dirent records.
///
/// The listing is captured when the directory is opened so repeated reads see one stable
/// enumeration, matching a guest that walks a directory while writing files elsewhere.
#[derive(Debug, Default)]
pub(crate) struct KernelDirectory {
    pub(crate) records: Vec<u8>,
    pub(crate) offset: usize,
}

/// Bytes of fixed dirent header before the name: `d_fileno`, `d_reclen`, `d_type`, `d_namlen`.
pub(crate) const DIRENT_HEADER_SIZE: usize = 8;

/// Console dirent records never straddle this boundary, so a reader can consume whole blocks.
pub(crate) const DIRENT_BLOCK_SIZE: usize = 512;

pub(crate) const DIRENT_TYPE_DIRECTORY: u8 = 4;

pub(crate) const DIRENT_TYPE_FILE: u8 = 8;

impl KernelDirectory {
    /// Packs one directory listing into console dirent records.
    ///
    /// Each record is four-byte aligned, and a record that would cross a 512-byte block is pushed
    /// to the next block by extending the previous record's length. Readers rely on that
    /// invariant to consume a block at a time without partially decoding a trailing record.
    pub(crate) fn new(entries: &[(String, bool)], next_file_number: &mut u32) -> Self {
        let mut records: Vec<u8> = Vec::new();
        let mut last_record_start: Option<usize> = None;
        for (name, is_directory) in entries {
            let name_bytes = name.as_bytes();
            let Ok(name_length) = u8::try_from(name_bytes.len()) else {
                // The console's dirent name length is a single byte; longer names cannot be
                // represented and are skipped rather than truncated into a different name.
                continue;
            };
            let record_length = (DIRENT_HEADER_SIZE + name_bytes.len() + 1).next_multiple_of(4);
            let block_remaining = DIRENT_BLOCK_SIZE - records.len() % DIRENT_BLOCK_SIZE;
            if record_length > block_remaining {
                // Extend the previous record over the padding so the block stays fully described.
                if let Some(start) = last_record_start {
                    let previous = u16::from_le_bytes([records[start + 4], records[start + 5]]);
                    let extended =
                        previous.saturating_add(u16::try_from(block_remaining).unwrap_or(u16::MAX));
                    records[start + 4..start + 6].copy_from_slice(&extended.to_le_bytes());
                }
                records.resize(records.len() + block_remaining, 0);
            }
            let file_number = *next_file_number;
            *next_file_number = next_file_number.wrapping_add(1).max(1);
            last_record_start = Some(records.len());
            records.extend_from_slice(&file_number.to_le_bytes());
            records.extend_from_slice(
                &u16::try_from(record_length)
                    .unwrap_or(u16::MAX)
                    .to_le_bytes(),
            );
            records.push(if *is_directory {
                DIRENT_TYPE_DIRECTORY
            } else {
                DIRENT_TYPE_FILE
            });
            records.push(name_length);
            records.extend_from_slice(name_bytes);
            records.resize(last_record_start.unwrap_or(0) + record_length, 0);
        }
        if !records.is_empty() {
            // Pad the final block so every returned read is a whole number of blocks.
            let tail = records.len() % DIRENT_BLOCK_SIZE;
            if tail != 0 {
                let padding = DIRENT_BLOCK_SIZE - tail;
                if let Some(start) = last_record_start {
                    let previous = u16::from_le_bytes([records[start + 4], records[start + 5]]);
                    let extended =
                        previous.saturating_add(u16::try_from(padding).unwrap_or(u16::MAX));
                    records[start + 4..start + 6].copy_from_slice(&extended.to_le_bytes());
                }
                records.resize(records.len() + padding, 0);
            }
        }
        Self { records, offset: 0 }
    }
}

pub(crate) fn guest_kernel_rmdir(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    guest_kernel_rmdir_resolved(arguments, |path| resolve_probe_guest_path(state, path))
}

pub(crate) fn guest_kernel_rmdir_prelock(arguments: [u64; 6]) -> u64 {
    let roots = crate::path_roots::load_path_roots();
    guest_kernel_rmdir_resolved(arguments, |path| {
        resolve_probe_guest_path_from_roots(&roots, path)
    })
}

fn guest_kernel_rmdir_resolved(
    arguments: [u64; 6],
    resolve: impl FnOnce(&str) -> Option<(PathBuf, bool)>,
) -> u64 {
    let Some(guest_path) = read_guest_cstring(arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    let Some((host_path, writable)) = resolve(&guest_path) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if !writable {
        report_read_only_directory_write("rmdir", &guest_path);
        return sce_kernel_error(SCE_KERNEL_ERROR_PERMISSION_DENIED);
    }
    if !host_path.exists() {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    if !host_path.is_dir() {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_DIRECTORY);
    }
    // Only an empty directory may be removed; the guest deletes contents explicitly first.
    match std::fs::remove_dir(&host_path) {
        Ok(()) => 0,
        Err(error) => sce_kernel_io_error(&error),
    }
}

/// Names the path a directory write was denied on. The read-only game mount answers every
/// mkdir/rmdir under /app0 with EACCES by design; the path is what tells a title probing for
/// scratch space (harmless) from one that genuinely needs a writable overlay REmu lacks.
fn report_read_only_directory_write(operation: &str, guest_path: &str) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static REPORTED: AtomicU64 = AtomicU64::new(0);
    let sequence = REPORTED.fetch_add(1, Ordering::Relaxed) + 1;
    if crate::trace_logarithmic_sequence(sequence, 16) {
        eprintln!(
            "[REMu][FILES] denied {operation} on read-only mount sequence={sequence} \
             path={guest_path:?}"
        );
    }
}

pub(crate) fn guest_kernel_mkdir(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    guest_kernel_mkdir_resolved(arguments, |path| resolve_probe_guest_path(state, path))
}

pub(crate) fn guest_kernel_mkdir_prelock(arguments: [u64; 6]) -> u64 {
    let roots = crate::path_roots::load_path_roots();
    guest_kernel_mkdir_resolved(arguments, |path| {
        resolve_probe_guest_path_from_roots(&roots, path)
    })
}

fn guest_kernel_mkdir_resolved(
    arguments: [u64; 6],
    resolve: impl FnOnce(&str) -> Option<(PathBuf, bool)>,
) -> u64 {
    let Some(guest_path) = read_guest_cstring(arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    let Some((host_path, writable)) = resolve(&guest_path) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if !writable {
        report_read_only_directory_write("mkdir", &guest_path);
        return sce_kernel_error(SCE_KERNEL_ERROR_PERMISSION_DENIED);
    }
    if host_path.exists() {
        return sce_kernel_error(SCE_KERNEL_ERROR_ALREADY_EXISTS);
    }
    let Some(parent) = host_path.parent() else {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if !parent.is_dir() {
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    match std::fs::create_dir(&host_path) {
        Ok(()) => 0,
        Err(error) => sce_kernel_io_error(&error),
    }
}

pub(crate) fn guest_fopen(path_address: u64, mode_address: u64, state: &ProbeDispatchState) -> u64 {
    let (Some(guest_path), Some(mode)) = (
        read_guest_cstring(path_address),
        read_guest_cstring(mode_address),
    ) else {
        return 0;
    };
    let Some((host_path, writable)) = resolve_probe_guest_path(state, &guest_path) else {
        if let Some(miss) = trace_guest_vfs_miss() {
            eprintln!("[REMu][VFS] miss={miss} rejected path {guest_path:?}");
        }
        return 0;
    };
    if !mode.starts_with('r') && !writable {
        if let Some(miss) = trace_guest_vfs_miss() {
            eprintln!("[REMu][VFS] miss={miss} rejected write mode {mode:?} for {guest_path:?}");
        }
        return 0;
    }
    if writable
        && !mode.starts_with('r')
        && let Some(parent) = host_path.parent()
        && std::fs::create_dir_all(parent).is_err()
    {
        return 0;
    }
    let (Ok(host_path), Ok(mode)) = (
        CString::new(host_path.to_string_lossy().as_bytes()),
        CString::new(mode),
    ) else {
        return 0;
    };
    // SAFETY: Both strings are NUL-terminated and host libc owns the returned FILE object.
    let file = unsafe { libc::fopen(host_path.as_ptr(), mode.as_ptr()) };
    if trace_guest_vfs() {
        eprintln!(
            "[REMu][VFS] fopen guest={guest_path:?} host={:?} result={file:p}",
            host_path.to_string_lossy()
        );
    }
    file as usize as u64
}

pub(crate) const APP_PATH_CACHE_CAPACITY: usize = 65_536;

#[derive(Debug, Default)]
pub(crate) struct ImmutableDirectoryIndex {
    pub(crate) exact: HashMap<String, PathBuf>,
    pub(crate) folded: HashMap<String, PathBuf>,
}

impl ImmutableDirectoryIndex {
    pub(crate) fn read(parent: &Path) -> Option<Self> {
        let mut index = Self::default();
        let entries = std::fs::read_dir(parent).ok()?;
        for entry in entries {
            let entry = entry.ok()?;
            let Some(name) = entry.file_name().to_str().map(ToOwned::to_owned) else {
                continue;
            };
            let path = entry.path();
            index
                .folded
                .entry(name.to_ascii_lowercase())
                .or_insert_with(|| path.clone());
            index.exact.insert(name, path);
        }
        Some(index)
    }

    pub(crate) fn get(&self, component: &str) -> Option<PathBuf> {
        self.exact
            .get(component)
            .or_else(|| self.folded.get(&component.to_ascii_lowercase()))
            .cloned()
    }

    pub(crate) fn entry_count(&self) -> usize {
        self.exact.len().saturating_add(self.folded.len())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ImmutableAppPathResolution {
    pub(crate) path: PathBuf,
    pub(crate) known_absent: bool,
}

#[derive(Debug, Default)]
pub(crate) struct ImmutableAppPathCache {
    pub(crate) roots: HashMap<PathBuf, HashMap<String, ImmutableAppPathResolution>>,
    pub(crate) directories: HashMap<PathBuf, ImmutableDirectoryIndex>,
    pub(crate) entries: usize,
}

impl ImmutableAppPathCache {
    #[cfg(test)]
    pub(crate) fn get(&self, game_root: &Path, relative: &str) -> Option<PathBuf> {
        self.get_resolution(game_root, relative)
            .map(|resolved| resolved.path)
    }

    pub(crate) fn get_resolution(
        &self,
        game_root: &Path,
        relative: &str,
    ) -> Option<ImmutableAppPathResolution> {
        self.roots.get(game_root)?.get(relative).cloned()
    }

    #[cfg(test)]
    pub(crate) fn insert(&mut self, game_root: &Path, relative: &str, resolved: PathBuf) {
        self.insert_resolution(
            game_root,
            relative,
            ImmutableAppPathResolution {
                path: resolved,
                known_absent: false,
            },
        );
    }

    pub(crate) fn insert_resolution(
        &mut self,
        game_root: &Path,
        relative: &str,
        resolved: ImmutableAppPathResolution,
    ) {
        if self.entries >= APP_PATH_CACHE_CAPACITY {
            self.roots.clear();
            self.entries = 0;
        }
        let previous = self
            .roots
            .entry(game_root.to_owned())
            .or_default()
            .insert(relative.to_owned(), resolved);
        if previous.is_none() {
            self.entries += 1;
        }
    }

    pub(crate) fn directory_entry(
        &self,
        parent: &Path,
        component: &str,
    ) -> Option<Option<PathBuf>> {
        self.directories
            .get(parent)
            .map(|directory| directory.get(component))
    }

    pub(crate) fn insert_directory(&mut self, parent: PathBuf, directory: ImmutableDirectoryIndex) {
        let new_entries = directory.entry_count();
        if self.entries.saturating_add(new_entries) >= APP_PATH_CACHE_CAPACITY {
            self.roots.clear();
            self.directories.clear();
            self.entries = 0;
        }
        if let Some(previous) = self.directories.insert(parent, directory) {
            self.entries = self.entries.saturating_sub(previous.entry_count());
        }
        self.entries = self.entries.saturating_add(new_entries);
    }
}

#[cfg(test)]
pub(crate) fn resolve_immutable_app_path(
    root: &Path,
    relative: &str,
    cache: &RwLock<ImmutableAppPathCache>,
) -> Option<PathBuf> {
    resolve_immutable_app_path_with_presence(root, relative, cache).map(|resolved| resolved.path)
}

pub(crate) fn resolve_immutable_app_path_with_presence(
    root: &Path,
    relative: &str,
    cache: &RwLock<ImmutableAppPathCache>,
) -> Option<ImmutableAppPathResolution> {
    let mut components: Vec<&str> = Vec::new();
    for component in relative.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                components.pop()?;
            }
            component => components.push(component),
        }
    }

    let mut host = root.to_owned();
    let mut known_absent = false;
    for component in components {
        if known_absent {
            host.push(component);
            continue;
        }
        let cached = cache
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .directory_entry(&host, component);
        match cached {
            Some(Some(matched)) => {
                host = matched;
            }
            Some(None) => {
                host.push(component);
                known_absent = true;
            }
            None => {
                let Some(directory) = ImmutableDirectoryIndex::read(&host) else {
                    // A transient I/O or permission failure is not proof that this component is
                    // absent. Preserve the requested spelling and let the caller's final host
                    // operation report the authoritative error.
                    host.push(component);
                    continue;
                };
                let matched = directory.get(component);
                cache
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert_directory(host.clone(), directory);
                match matched {
                    Some(matched) => host = matched,
                    None => {
                        host.push(component);
                        known_absent = true;
                    }
                }
            }
        }
    }
    Some(ImmutableAppPathResolution {
        path: host,
        known_absent,
    })
}

fn normalized_guest_path(guest_path: &str) -> Cow<'_, str> {
    if guest_path.as_bytes().contains(&b'\\') {
        Cow::Owned(guest_path.replace('\\', "/"))
    } else {
        Cow::Borrowed(guest_path)
    }
}

#[cfg(test)]
mod path_normalization_tests {
    use std::borrow::Cow;

    use super::normalized_guest_path;

    #[test]
    fn forward_slash_guest_paths_keep_borrowed_storage() {
        let path = "/app0/majesty/content/paks/majesty-ps5.pak";
        let normalized = normalized_guest_path(path);

        assert_eq!(normalized, path);
        assert!(matches!(normalized, Cow::Borrowed(_)));
    }

    #[test]
    fn backslash_guest_paths_are_normalized_once() {
        let normalized = normalized_guest_path(r"\app0\majesty\content\paks");

        assert_eq!(normalized, "/app0/majesty/content/paks");
        assert!(matches!(normalized, Cow::Owned(_)));
    }
}

pub(crate) fn resolve_normalized_app_path_with_presence(
    game_root: &Path,
    normalized: &str,
) -> Option<ImmutableAppPathResolution> {
    let relative = if let Some(path) = normalized.strip_prefix("/app0/") {
        path
    } else if normalized == "/app0" {
        ""
    } else if let Some((_, path)) = normalized.split_once("/app0/") {
        path
    } else if normalized.starts_with('/') {
        return None;
    } else {
        normalized
    };
    // `/app0` is immutable for the lifetime of a title. Unreal repeatedly probes the same
    // optional configuration and localization paths, and resolving each miss used to repeat an
    // `exists` syscall for every component plus a `read_dir` scan for every case-insensitive
    // fallback. Cache the resolved host spelling (including paths that ultimately do not exist)
    // so subsequent operations do not repeat path-component I/O. A successful immutable parent
    // scan also proves a missing child cannot exist, allowing metadata-only callers to avoid a
    // redundant final syscall without changing open/read behavior.
    static APP_PATH_CACHE: OnceLock<RwLock<ImmutableAppPathCache>> = OnceLock::new();
    let cache = APP_PATH_CACHE.get_or_init(|| RwLock::new(ImmutableAppPathCache::default()));
    if let Some(resolved) = cache
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get_resolution(game_root, relative)
    {
        return Some(resolved);
    }
    let resolved = resolve_immutable_app_path_with_presence(game_root, relative, cache)?;
    let mut cache = cache
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    cache.insert_resolution(game_root, relative, resolved.clone());
    Some(resolved)
}

fn resolve_guest_path_with_presence(
    game_root: &Path,
    guest_path: &str,
) -> Option<ImmutableAppPathResolution> {
    let normalized = normalized_guest_path(guest_path);
    resolve_normalized_app_path_with_presence(game_root, &normalized)
}

pub(crate) fn resolve_guest_path(game_root: &Path, guest_path: &str) -> Option<PathBuf> {
    resolve_guest_path_with_presence(game_root, guest_path).map(|resolved| resolved.path)
}

/// Returns a host file that stands in for a guest path, if an overlay directory supplies one.
///
/// A dump is read-only and stays that way, so replacing one file in it -- a command line, a
/// configuration file -- otherwise means editing the dump itself. An overlay directory mirrors
/// the guest tree and is consulted first, which keeps experiments out of the dump entirely.
///
/// Set `REMU_OVERLAY_DIR` to a directory whose layout mirrors the guest filesystem, so
/// `/app0/ue4commandline.txt` is served from `$REMU_OVERLAY_DIR/app0/ue4commandline.txt`.
pub(crate) fn overlay_guest_path(normalized: &str) -> Option<PathBuf> {
    static OVERLAY: OnceLock<Option<PathBuf>> = OnceLock::new();
    let root = OVERLAY
        .get_or_init(|| std::env::var_os("REMU_OVERLAY_DIR").map(PathBuf::from))
        .as_ref()?;
    let candidate = resolve_path_beneath(root, normalized.trim_start_matches('/'))?;
    candidate.is_file().then_some(candidate)
}

pub(crate) fn resolve_probe_guest_path(
    state: &ProbeDispatchState,
    guest_path: &str,
) -> Option<(PathBuf, bool)> {
    resolve_probe_guest_path_with_presence(state, guest_path)
        .map(|resolved| (resolved.path, resolved.writable))
}

/// Path resolution against a published [`crate::path_roots::PathRoots`] snapshot (C2 pre-lock).
pub(crate) fn resolve_probe_guest_path_from_roots(
    roots: &crate::path_roots::PathRoots,
    guest_path: &str,
) -> Option<(PathBuf, bool)> {
    resolve_probe_guest_path_with_presence_from_roots(roots, guest_path)
        .map(|resolved| (resolved.path, resolved.writable))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedProbeGuestPath {
    pub(crate) path: PathBuf,
    pub(crate) writable: bool,
    pub(crate) known_absent: bool,
}

pub(crate) fn resolve_probe_guest_path_with_presence(
    state: &ProbeDispatchState,
    guest_path: &str,
) -> Option<ResolvedProbeGuestPath> {
    resolve_probe_guest_path_with_presence_from_parts(
        &state.game_root,
        &state.save_mounts,
        &state.vfs,
        &state.prepared_save_resources,
        guest_path,
    )
}

pub(crate) fn resolve_probe_guest_path_with_presence_from_roots(
    roots: &crate::path_roots::PathRoots,
    guest_path: &str,
) -> Option<ResolvedProbeGuestPath> {
    resolve_probe_guest_path_with_presence_from_parts(
        &roots.game_root,
        &roots.save_mounts,
        &roots.vfs,
        &roots.prepared_save_resources,
        guest_path,
    )
}

fn resolve_probe_guest_path_with_presence_from_parts(
    game_root: &Path,
    save_mounts: &std::collections::BTreeMap<String, crate::SaveDataMount>,
    vfs: &crate::vfs::VfsState,
    prepared_save_resources: &std::collections::BTreeSet<u32>,
    guest_path: &str,
) -> Option<ResolvedProbeGuestPath> {
    let normalized = normalized_guest_path(guest_path);
    if let Some(overlaid) = overlay_guest_path(&normalized) {
        return Some(ResolvedProbeGuestPath {
            path: overlaid,
            writable: false,
            known_absent: false,
        });
    }
    // Static namespaces (`/app0`, `/download0`, `/temp0`) live in the VfsMount table (G2).
    // Save-data mounts stay on `save_mounts` because Prepare retargets their host root and
    // writability depends on the transaction resource — folding that into VfsMount is a later seam.
    if let Some(mount) = vfs.find_mount(&normalized)
        && !matches!(mount.policy.kind, crate::vfs::VfsMountKind::SaveData)
    {
        return crate::vfs::resolve_via_vfs_mount(mount, &normalized);
    }
    for (mount_point, mounted) in save_mounts {
        let writable = !mounted.read_only
            && mounted.resource != u32::MAX
            && prepared_save_resources.contains(&mounted.resource);
        if normalized == *mount_point {
            return Some(ResolvedProbeGuestPath {
                path: mounted.path.clone(),
                writable,
                known_absent: false,
            });
        }
        if let Some(relative) = normalized
            .strip_prefix(mount_point)
            .and_then(|relative| relative.strip_prefix('/'))
        {
            return resolve_path_beneath(&mounted.path, relative).map(|path| {
                ResolvedProbeGuestPath {
                    path,
                    writable,
                    known_absent: false,
                }
            });
        }
    }
    resolve_normalized_app_path_with_presence(game_root, &normalized).map(|resolved| {
        ResolvedProbeGuestPath {
            path: resolved.path,
            writable: false,
            known_absent: resolved.known_absent,
        }
    })
}

/// Finds a sibling of `path` whose name differs only by case.
///
/// Returns `None` when the parent cannot be listed or nothing matches, leaving the caller's
/// original path so a genuine miss still reports the name the guest asked for.
pub(crate) fn case_insensitive_child(path: &Path) -> Option<PathBuf> {
    let name = path.file_name()?.to_str()?;
    let parent = path.parent()?;
    let mut matched = None;
    for entry in std::fs::read_dir(parent).ok()? {
        let entry = entry.ok()?;
        let candidate = entry.file_name();
        let candidate = candidate.to_str()?;
        if candidate.eq_ignore_ascii_case(name) {
            // An exact match always wins; otherwise keep the first case-insensitive hit so the
            // result stays stable regardless of directory order.
            if candidate == name {
                return Some(entry.path());
            }
            matched.get_or_insert_with(|| entry.path());
        }
    }
    matched
}

pub(crate) fn resolve_path_beneath(root: &Path, relative: &str) -> Option<PathBuf> {
    // Console paths routinely contain duplicate separators (for example
    // `/app0//shaders/presenter.vert`) and parent components: engines address their content
    // relative to a binaries directory, so `../../../game/config.ini` is an ordinary request, not
    // an attack. Resolve `..` by popping the component it cancels, and refuse only a path that
    // would leave the game root entirely.
    let mut components: Vec<&str> = Vec::new();
    for component in relative.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                components.pop()?;
            }
            component => components.push(component),
        }
    }
    let mut host = root.to_owned();
    for component in components {
        host.push(component);
        if !host.exists() {
            // Console lookups do not distinguish case, and dumps are frequently stored with
            // normalized names. Fall back to a case-insensitive match within the parent so a
            // title that asks for `AkRoomVerb.prx` still finds `akroomverb.prx`.
            if let Some(matched) = case_insensitive_child(&host) {
                host = matched;
            }
        }
    }
    Some(host)
}

pub(crate) fn guest_fseek(handle: u64, offset: u64, whence: u64) -> u64 {
    if handle == 0 {
        return u64::MAX;
    }
    let offset = i64::from_ne_bytes(offset.to_ne_bytes());
    let whence = low_i32(whence);
    // SAFETY: fopen returned the handle, and libc validates offset/whence.
    #[cfg(windows)]
    let status = unsafe { windows_fseeki64(handle as *mut libc::FILE, offset, whence) };
    #[cfg(not(windows))]
    let status = unsafe { libc::fseek(handle as *mut libc::FILE, offset as libc::c_long, whence) };
    signed_i32_return(status)
}

pub(crate) fn guest_ftell(handle: u64) -> u64 {
    if handle == 0 {
        return u64::MAX;
    }
    // SAFETY: fopen returned the live FILE handle.
    #[cfg(windows)]
    let position = unsafe { windows_ftelli64(handle as *mut libc::FILE) };
    #[cfg(not(windows))]
    let position = unsafe { i64::from(libc::ftell(handle as *mut libc::FILE)) };
    if position < 0 {
        u64::MAX
    } else {
        position.cast_unsigned()
    }
}

#[cfg(windows)]
#[link(name = "ucrt")]
unsafe extern "C" {
    #[link_name = "_fseeki64"]
    fn windows_fseeki64(stream: *mut libc::FILE, offset: i64, origin: i32) -> i32;
    #[link_name = "_ftelli64"]
    fn windows_ftelli64(stream: *mut libc::FILE) -> i64;
}

pub(crate) fn guest_fclose(handle: u64) -> u64 {
    if handle == 0 {
        return u64::MAX;
    }
    // SAFETY: fopen returned the live FILE handle and ownership ends here.
    signed_i32_return(unsafe { libc::fclose(handle as *mut libc::FILE) })
}

pub(crate) fn guest_fread(destination: u64, size: u64, count: u64, handle: u64) -> u64 {
    if destination == 0 || handle == 0 || size == 0 || count == 0 {
        return 0;
    }
    let (Ok(size), Ok(count)) = (usize::try_from(size), usize::try_from(count)) else {
        return 0;
    };
    // SAFETY: The guest supplies a destination sized for size*count and fopen returned the handle.
    unsafe {
        libc::fread(
            destination as *mut libc::c_void,
            size,
            count,
            handle as *mut libc::FILE,
        ) as u64
    }
}

pub(crate) fn guest_kernel_open(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    guest_kernel_open_with_resolver(arguments, |path| resolve_probe_guest_path(state, path))
}

/// Pre-lock `sceKernelOpen` against the published path-roots snapshot (plan C2).
pub(crate) fn guest_kernel_open_prelock(arguments: [u64; 6]) -> u64 {
    let roots = crate::path_roots::load_path_roots();
    guest_kernel_open_with_resolver(arguments, |path| {
        resolve_probe_guest_path_from_roots(&roots, path)
    })
}

fn guest_kernel_open_with_resolver(
    arguments: [u64; 6],
    resolve: impl FnOnce(&str) -> Option<(PathBuf, bool)>,
) -> u64 {
    let Some(guest_path) = read_guest_cstring(arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    let flags = low_i32(arguments[1]);
    let access = flags & 0x3;
    if access == 0x3 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let Some((host_path, writable)) = resolve(&guest_path) else {
        if let Some(miss) = trace_guest_vfs_miss() {
            eprintln!("[REMu][VFS] miss={miss} sceKernelOpen rejected guest={guest_path:?}");
        }
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    if access != 0 && !writable {
        if let Some(miss) = trace_guest_vfs_miss() {
            eprintln!(
                "[REMu][VFS] miss={miss} sceKernelOpen rejected write access guest={guest_path:?} flags={flags:#x}"
            );
        }
        return sce_kernel_error(SCE_KERNEL_ERROR_PERMISSION_DENIED);
    }
    let create = writable && flags & 0x200 != 0;
    let exclusive = create && flags & 0x800 != 0;
    let mut options = std::fs::OpenOptions::new();
    options
        .read(access != 1)
        .write(access != 0)
        .append(access != 0 && flags & 0x8 != 0)
        .create(create && !exclusive)
        .create_new(exclusive)
        .truncate(writable && access != 0 && flags & 0x400 != 0);
    #[cfg(unix)]
    {
        options.mode(low_u32(arguments[2]));
    }
    let file = match options.open(&host_path) {
        Ok(file) => file,
        Err(error) => {
            let miss = trace_guest_vfs_miss();
            if error.kind() != std::io::ErrorKind::NotFound || miss.is_some() {
                eprintln!(
                    "[REMu][VFS] miss={miss:?} sceKernelOpen failed guest={guest_path:?} host={:?}: {error}",
                    host_path.display()
                );
            }
            return sce_kernel_io_error(&error);
        }
    };
    if flags & 0x0002_0000 != 0 && file.metadata().is_ok_and(|metadata| !metadata.is_dir()) {
        if let Some(miss) = trace_guest_vfs_miss() {
            eprintln!(
                "[REMu][VFS] miss={miss} sceKernelOpen expected directory guest={guest_path:?} host={:?}",
                host_path.display()
            );
        }
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_DIRECTORY);
    }
    let is_directory = file.metadata().is_ok_and(|metadata| metadata.is_dir());
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fd = files.next_fd.max(3);
    files.next_fd = fd.saturating_add(1);
    if is_directory {
        let mut entries: Vec<(String, bool)> = match std::fs::read_dir(&host_path) {
            Ok(listing) => listing
                .filter_map(|entry| {
                    let entry = entry.ok()?;
                    Some((
                        entry.file_name().to_string_lossy().into_owned(),
                        entry.file_type().is_ok_and(|kind| kind.is_dir()),
                    ))
                })
                .collect(),
            Err(error) => {
                let miss = trace_guest_vfs_miss();
                if error.kind() != std::io::ErrorKind::NotFound || miss.is_some() {
                    eprintln!(
                        "[REMu][VFS] miss={miss:?} sceKernelOpen could not list guest={guest_path:?} host={:?}: {error}",
                        host_path.display()
                    );
                }
                return sce_kernel_io_error(&error);
            }
        };
        // A stable order keeps repeated enumerations reproducible across runs and hosts.
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        let mut next_file_number = files.next_file_number;
        let directory = KernelDirectory::new(&entries, &mut next_file_number);
        files.next_file_number = next_file_number;
        files.directories.insert(fd, directory);
        if trace_guest_vfs() {
            eprintln!(
                "[REMu][VFS] sceKernelOpen directory guest={guest_path:?} entries={} fd={fd}",
                entries.len()
            );
        }
    }
    files.files.insert(fd, file);
    if trace_guest_vfs() {
        eprintln!(
            "[REMu][VFS] sceKernelOpen guest={guest_path:?} host={:?} flags={flags:#x} fd={fd}",
            host_path.display()
        );
    }
    signed_i32_return(fd)
}

/// Reads packed directory entries into the guest buffer.
///
/// Returns the number of bytes written and zero once the enumeration is exhausted. `basep`, when
/// supplied, receives the enumeration position before the read so the guest can resume there.
pub(crate) fn guest_kernel_getdents(arguments: [u64; 6], report_position: bool) -> u64 {
    let (fd, buffer, capacity) = (low_i32(arguments[0]), arguments[1], arguments[2]);
    if buffer == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    if capacity < DIRENT_BLOCK_SIZE as u64 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !files.files.contains_key(&fd) {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    }
    let Some(directory) = files.directories.get_mut(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if report_position && arguments[3] != 0 {
        write_guest_u64(arguments[3], directory.offset as u64);
    }
    let remaining = directory.records.len().saturating_sub(directory.offset);
    if remaining == 0 {
        return 0;
    }
    // Only whole blocks are transferred, so a record is never split across two reads.
    let transfer = remaining.min((capacity as usize / DIRENT_BLOCK_SIZE) * DIRENT_BLOCK_SIZE);
    if transfer == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let end = directory.offset + transfer;
    write_guest_bytes(buffer, &directory.records[directory.offset..end]);
    directory.offset = end;
    transfer as u64
}

pub(crate) fn guest_kernel_stat(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    guest_kernel_stat_resolved(arguments, |path| {
        resolve_probe_guest_path_with_presence(state, path)
    })
}

pub(crate) fn guest_kernel_stat_prelock(arguments: [u64; 6]) -> u64 {
    let roots = crate::path_roots::load_path_roots();
    guest_kernel_stat_resolved(arguments, |path| {
        resolve_probe_guest_path_with_presence_from_roots(&roots, path)
    })
}

fn guest_kernel_stat_resolved(
    arguments: [u64; 6],
    resolve: impl FnOnce(&str) -> Option<ResolvedProbeGuestPath>,
) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let Some(guest_path) = read_guest_cstring(arguments[0]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    };
    let Some(resolved) = resolve(&guest_path) else {
        if let Some(miss) = trace_guest_vfs_miss() {
            eprintln!("[REMu][VFS] miss={miss} sceKernelStat rejected guest={guest_path:?}");
        }
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    };
    let host_path = resolved.path;
    if resolved.known_absent {
        if let Some(miss) = trace_guest_vfs_miss() {
            eprintln!(
                "[REMu][VFS] miss={miss} sceKernelStat known absent guest={guest_path:?} host={:?}",
                host_path.display()
            );
        }
        return sce_kernel_error(SCE_KERNEL_ERROR_NOT_FOUND);
    }
    let metadata = match std::fs::metadata(&host_path) {
        Ok(metadata) => metadata,
        Err(error) => {
            let miss = trace_guest_vfs_miss();
            if error.kind() != std::io::ErrorKind::NotFound || miss.is_some() {
                eprintln!(
                    "[REMu][VFS] miss={miss:?} sceKernelStat failed guest={guest_path:?} host={:?}: {error}",
                    host_path.display()
                );
            }
            return sce_kernel_io_error(&error);
        }
    };
    let payload = kernel_stat_payload_from_metadata(&metadata, guest_path.as_bytes());
    // SAFETY: sceKernelStat supplies a writable 120-byte Gen5 stat object in live guest memory.
    unsafe {
        std::ptr::copy_nonoverlapping(payload.as_ptr(), arguments[1] as *mut u8, payload.len());
    }
    if trace_guest_vfs() {
        eprintln!(
            "[REMu][VFS] sceKernelStat guest={guest_path:?} host={:?} size={}",
            host_path.display(),
            metadata.len()
        );
    }
    0
}

pub(crate) fn guest_kernel_read(arguments: [u64; 6]) -> u64 {
    let fd = low_i32(arguments[0]);
    let destination = arguments[1];
    let requested = arguments[2];
    if requested == 0 {
        return 0;
    }
    if destination == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let Ok(requested) = usize::try_from(requested) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(file) = files.files.get_mut(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    };
    // SAFETY: sceKernelRead supplies a writable range of exactly `requested` bytes in live guest
    // memory, and the global import dispatcher serializes access to the associated File.
    let destination = unsafe { std::slice::from_raw_parts_mut(destination as *mut u8, requested) };
    let read = match file.read(destination) {
        Ok(read) => read,
        Err(error) => return sce_kernel_io_error(&error),
    };
    if trace_guest_vfs() {
        eprintln!("[REMu][VFS] sceKernelRead fd={fd} requested={requested} read={read}");
    }
    u64::try_from(read).expect("read byte count fits u64")
}

pub(crate) fn guest_kernel_write(arguments: [u64; 6]) -> u64 {
    let fd = low_i32(arguments[0]);
    let Ok(length) = usize::try_from(arguments[2]) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    };
    if length != 0 && arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let payload = if length == 0 {
        &[]
    } else {
        // SAFETY: `_write` supplies a readable guest buffer of exactly `length` bytes.
        unsafe { std::slice::from_raw_parts(arguments[1] as *const u8, length) }
    };
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let result = match fd {
        1 => std::io::stdout().write(payload),
        2 => std::io::stderr().write(payload),
        _ => files
            .files
            .get_mut(&fd)
            .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::NotFound))
            .and_then(|file| file.write(payload)),
    };
    match result {
        Ok(written) => u64::try_from(written).expect("written byte count fits u64"),
        Err(error) if fd != 1 && fd != 2 && !files.files.contains_key(&fd) => {
            sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR)
        }
        Err(error) => sce_kernel_io_error(&error),
    }
}

pub(crate) fn guest_kernel_close(arguments: [u64; 6]) -> u64 {
    let fd = low_i32(arguments[0]);
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    files.directories.remove(&fd);
    if files.files.remove(&fd).is_none() {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    }
    drop(files);
    if trace_guest_vfs() {
        eprintln!("[REMu][VFS] sceKernelClose fd={fd}");
    }
    0
}

pub(crate) fn guest_kernel_lseek(arguments: [u64; 6]) -> u64 {
    let fd = low_i32(arguments[0]);
    let offset = i64::from_ne_bytes(arguments[1].to_ne_bytes());
    let seek_from = match low_i32(arguments[2]) {
        0 if offset >= 0 => std::io::SeekFrom::Start(offset.cast_unsigned()),
        1 => std::io::SeekFrom::Current(offset),
        2 => std::io::SeekFrom::End(offset),
        _ => return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT),
    };
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(file) = files.files.get_mut(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    };
    match file.seek(seek_from) {
        Ok(position) => position,
        Err(error) => sce_kernel_io_error(&error),
    }
}

pub(crate) fn guest_kernel_fstat(arguments: [u64; 6]) -> u64 {
    if arguments[1] == 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_MEMORY_FAULT);
    }
    let fd = low_i32(arguments[0]);
    let files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(file) = files.files.get(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    };
    let metadata = match file.metadata() {
        Ok(metadata) => metadata,
        Err(error) => return sce_kernel_io_error(&error),
    };
    let payload = kernel_stat_payload_from_metadata(&metadata, &fd.to_le_bytes());
    // SAFETY: the guest ABI supplies writable storage for one complete 120-byte stat object.
    unsafe {
        std::ptr::copy_nonoverlapping(payload.as_ptr(), arguments[1] as *mut u8, payload.len());
    }
    0
}

pub(crate) fn guest_kernel_ftruncate(arguments: [u64; 6]) -> u64 {
    let fd = low_i32(arguments[0]);
    let length = i64::from_ne_bytes(arguments[1].to_ne_bytes());
    if length < 0 {
        return sce_kernel_error(SCE_KERNEL_ERROR_INVALID_ARGUMENT);
    }
    let mut files = KERNEL_FILES
        .get_or_init(|| Mutex::new(KernelFileState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(file) = files.files.get_mut(&fd) else {
        return sce_kernel_error(SCE_KERNEL_ERROR_BAD_FILE_DESCRIPTOR);
    };
    file.set_len(length.cast_unsigned())
        .map_or_else(|error| sce_kernel_io_error(&error), |()| 0)
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct GuestKernelStat {
    pub(crate) dev: u32,
    pub(crate) ino: u32,
    pub(crate) mode: u16,
    pub(crate) nlink: u16,
    pub(crate) uid: u32,
    pub(crate) gid: u32,
    pub(crate) rdev: u32,
    pub(crate) atime_sec: i64,
    pub(crate) atime_nsec: i64,
    pub(crate) mtime_sec: i64,
    pub(crate) mtime_nsec: i64,
    pub(crate) ctime_sec: i64,
    pub(crate) ctime_nsec: i64,
    pub(crate) size: i64,
    pub(crate) blocks: i64,
    pub(crate) block_size: u32,
    pub(crate) flags: u32,
    pub(crate) generation: u32,
    pub(crate) spare: i32,
    pub(crate) birthtime_sec: i64,
    pub(crate) birthtime_nsec: i64,
}

impl GuestKernelStat {
    pub(crate) fn encode(self) -> [u8; SCE_KERNEL_STAT_SIZE] {
        let mut payload = [0_u8; SCE_KERNEL_STAT_SIZE];
        payload[0..4].copy_from_slice(&self.dev.to_le_bytes());
        payload[4..8].copy_from_slice(&self.ino.to_le_bytes());
        payload[8..10].copy_from_slice(&self.mode.to_le_bytes());
        payload[10..12].copy_from_slice(&self.nlink.to_le_bytes());
        payload[12..16].copy_from_slice(&self.uid.to_le_bytes());
        payload[16..20].copy_from_slice(&self.gid.to_le_bytes());
        payload[20..24].copy_from_slice(&self.rdev.to_le_bytes());
        payload[24..32].copy_from_slice(&self.atime_sec.to_le_bytes());
        payload[32..40].copy_from_slice(&self.atime_nsec.to_le_bytes());
        payload[40..48].copy_from_slice(&self.mtime_sec.to_le_bytes());
        payload[48..56].copy_from_slice(&self.mtime_nsec.to_le_bytes());
        payload[56..64].copy_from_slice(&self.ctime_sec.to_le_bytes());
        payload[64..72].copy_from_slice(&self.ctime_nsec.to_le_bytes());
        payload[72..80].copy_from_slice(&self.size.to_le_bytes());
        payload[80..88].copy_from_slice(&self.blocks.to_le_bytes());
        payload[88..92].copy_from_slice(&self.block_size.to_le_bytes());
        payload[92..96].copy_from_slice(&self.flags.to_le_bytes());
        payload[96..100].copy_from_slice(&self.generation.to_le_bytes());
        payload[100..104].copy_from_slice(&self.spare.to_le_bytes());
        payload[104..112].copy_from_slice(&self.birthtime_sec.to_le_bytes());
        payload[112..120].copy_from_slice(&self.birthtime_nsec.to_le_bytes());
        payload
    }
}

pub(crate) fn fallback_inode(inode_seed: &[u8]) -> u32 {
    inode_seed.iter().fold(0x811c_9dc5_u32, |hash, &byte| {
        (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
    })
}

#[cfg(any(test, not(unix)))]
pub(crate) fn kernel_stat_payload(size: u64, is_directory: bool, inode_seed: &[u8]) -> [u8; 120] {
    let size = i64::try_from(size).unwrap_or(i64::MAX);
    GuestKernelStat {
        ino: fallback_inode(inode_seed),
        mode: if is_directory { 0x41ff } else { 0x81ff },
        nlink: 1,
        size,
        blocks: if is_directory {
            128
        } else {
            (size + 511) / 512
        },
        block_size: if is_directory { 65_536 } else { 512 },
        ..GuestKernelStat::default()
    }
    .encode()
}

pub(crate) fn kernel_stat_payload_from_metadata(
    metadata: &std::fs::Metadata,
    inode_seed: &[u8],
) -> [u8; SCE_KERNEL_STAT_SIZE] {
    #[cfg(unix)]
    {
        let (birth_seconds, birth_nanoseconds) = metadata
            .created()
            .ok()
            .and_then(|created| created.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or((metadata.ctime(), metadata.ctime_nsec()), |duration| {
                (
                    i64::try_from(duration.as_secs()).unwrap_or(i64::MAX),
                    i64::from(duration.subsec_nanos()),
                )
            });
        GuestKernelStat {
            dev: u32::try_from(metadata.dev()).unwrap_or(u32::MAX),
            ino: u32::try_from(metadata.ino()).unwrap_or_else(|_| fallback_inode(inode_seed)),
            mode: u16::try_from(metadata.mode() & u32::from(u16::MAX)).expect("masked mode fits"),
            nlink: u16::try_from(metadata.nlink()).unwrap_or(u16::MAX),
            uid: metadata.uid(),
            gid: metadata.gid(),
            rdev: u32::try_from(metadata.rdev()).unwrap_or(u32::MAX),
            atime_sec: metadata.atime(),
            atime_nsec: metadata.atime_nsec(),
            mtime_sec: metadata.mtime(),
            mtime_nsec: metadata.mtime_nsec(),
            ctime_sec: metadata.ctime(),
            ctime_nsec: metadata.ctime_nsec(),
            size: i64::try_from(metadata.size()).unwrap_or(i64::MAX),
            blocks: i64::try_from(metadata.blocks()).unwrap_or(i64::MAX),
            block_size: u32::try_from(metadata.blksize()).unwrap_or(u32::MAX),
            flags: 0,
            generation: 0,
            spare: 0,
            birthtime_sec: birth_seconds,
            birthtime_nsec: birth_nanoseconds,
        }
        .encode()
    }
    #[cfg(not(unix))]
    {
        kernel_stat_payload(metadata.len(), metadata.is_dir(), inode_seed)
    }
}
