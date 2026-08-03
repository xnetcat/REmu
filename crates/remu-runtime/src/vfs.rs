//! Guest VFS mount table (plan G2 / audit B27 R1).
//!
//! Path resolution matches the longest registered guest prefix and applies that mount's policy.
//! `/download0` is a `Download` kind (never enumerable as savedata). `/temp0` is a writable
//! `Scratch` mount registered by `sceAppContentTemporaryDataMount2`
//! (`sdk/target/include/app_content.h:328-334`, mount-point size
//! `SCE_APP_CONTENT_MOUNTPOINT_DATA_MAXSIZE` = 16 at `:215`).

use std::path::{Path, PathBuf};

use crate::guest_files::{
    ResolvedProbeGuestPath, resolve_normalized_app_path_with_presence, resolve_path_beneath,
};

/// Class of a registered guest mount. Policy fields that used to be string filters
/// (`download0`, `temp0`) become variants here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VfsMountKind {
    /// Read-only package content (`/app0`).
    Immutable,
    /// Per-title temporary data (`/temp0`).
    Scratch,
    /// Persistent download / cache namespace (`/download0`).
    Download,
    /// A live save-data slot (`/savedataN`).
    SaveData,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MountPolicy {
    pub(crate) writable: bool,
    pub(crate) case_insensitive: bool,
    /// When true, the host directory under this mount may appear in `sceSaveDataDirNameSearch`.
    /// Download and scratch mounts are never enumerable as save slots (closes F-01 structurally).
    pub(crate) enumerable_as_savedata: bool,
    pub(crate) quota: Option<u64>,
    pub(crate) kind: VfsMountKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct VfsMount {
    pub(crate) guest_prefix: String,
    pub(crate) host_root: PathBuf,
    pub(crate) policy: MountPolicy,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct VfsState {
    /// Sorted by descending guest-prefix length so the first match is the longest prefix.
    mounts: Vec<VfsMount>,
}

impl VfsState {
    pub(crate) fn mounts(&self) -> &[VfsMount] {
        &self.mounts
    }

    /// Insert or replace a mount with the same guest prefix, then re-sort by prefix length.
    pub(crate) fn register(&mut self, mount: VfsMount) {
        if let Some(existing) = self
            .mounts
            .iter_mut()
            .find(|entry| entry.guest_prefix == mount.guest_prefix)
        {
            *existing = mount;
        } else {
            self.mounts.push(mount);
        }
        self.mounts
            .sort_by_key(|entry| std::cmp::Reverse(entry.guest_prefix.len()));
    }

    pub(crate) fn unregister(&mut self, guest_prefix: &str) -> bool {
        let before = self.mounts.len();
        self.mounts
            .retain(|entry| entry.guest_prefix != guest_prefix);
        before != self.mounts.len()
    }

    pub(crate) fn clear(&mut self) {
        self.mounts.clear();
    }

    /// Longest registered prefix that equals `normalized` or is a parent path of it.
    pub(crate) fn find_mount(&self, normalized: &str) -> Option<&VfsMount> {
        self.mounts.iter().find(|mount| {
            normalized == mount.guest_prefix
                || normalized
                    .strip_prefix(&mount.guest_prefix)
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
        })
    }

    pub(crate) fn find_mount_mut(&mut self, guest_prefix: &str) -> Option<&mut VfsMount> {
        self.mounts
            .iter_mut()
            .find(|mount| mount.guest_prefix == guest_prefix)
    }

    /// Rebuild the static mounts that exist for every title: `/app0` and `/download0`.
    ///
    /// `/temp0` is registered only when `sceAppContentTemporaryDataMount2` creates it — titles that
    /// never ask for temporary data must not see a phantom writable root.
    pub(crate) fn rebuild_static_mounts(&mut self, game_root: &Path, save_root: &Path) {
        self.mounts
            .retain(|mount| matches!(mount.policy.kind, VfsMountKind::SaveData));
        self.register(VfsMount {
            guest_prefix: "/app0".to_owned(),
            host_root: game_root.to_owned(),
            policy: MountPolicy {
                writable: false,
                case_insensitive: true,
                enumerable_as_savedata: false,
                quota: None,
                kind: VfsMountKind::Immutable,
            },
        });
        self.register(download_mount(save_root));
    }

    pub(crate) fn register_temp0(&mut self, save_root: &Path) -> PathBuf {
        let host_root = save_root.join("temp0");
        self.register(VfsMount {
            guest_prefix: "/temp0".to_owned(),
            host_root: host_root.clone(),
            policy: MountPolicy {
                writable: true,
                case_insensitive: true,
                enumerable_as_savedata: false,
                quota: None,
                kind: VfsMountKind::Scratch,
            },
        });
        host_root
    }

    pub(crate) fn register_save_data_mount(
        &mut self,
        guest_prefix: &str,
        host_root: PathBuf,
    ) -> Result<(), &'static str> {
        if !guest_prefix.starts_with("/savedata") {
            return Err("save-data mount prefix must be /savedataN");
        }
        self.register(VfsMount {
            guest_prefix: guest_prefix.to_owned(),
            host_root,
            policy: MountPolicy {
                writable: false,
                case_insensitive: true,
                enumerable_as_savedata: true,
                quota: None,
                kind: VfsMountKind::SaveData,
            },
        });
        Ok(())
    }

    pub(crate) fn set_save_data_host_root(&mut self, guest_prefix: &str, host_root: PathBuf) {
        if let Some(mount) = self.find_mount_mut(guest_prefix) {
            mount.host_root = host_root;
        }
    }
}

fn download_mount(save_root: &Path) -> VfsMount {
    VfsMount {
        guest_prefix: "/download0".to_owned(),
        host_root: save_root.join("download0"),
        policy: MountPolicy {
            writable: true,
            case_insensitive: true,
            enumerable_as_savedata: false,
            quota: None,
            kind: VfsMountKind::Download,
        },
    }
}

/// Resolve a normalized guest path against the mount table.
///
/// Immutable `/app0` keeps the specialized presence-aware resolver (cache + known-absent). Other
/// kinds walk beneath `host_root` and refuse `..` escapes via [`resolve_path_beneath`].
pub(crate) fn resolve_via_vfs_mount(
    mount: &VfsMount,
    normalized: &str,
) -> Option<ResolvedProbeGuestPath> {
    match mount.policy.kind {
        VfsMountKind::Immutable => {
            resolve_normalized_app_path_with_presence(&mount.host_root, normalized).map(
                |resolved| ResolvedProbeGuestPath {
                    path: resolved.path,
                    writable: false,
                    known_absent: resolved.known_absent,
                },
            )
        }
        VfsMountKind::Scratch | VfsMountKind::Download | VfsMountKind::SaveData => {
            if normalized == mount.guest_prefix {
                return Some(ResolvedProbeGuestPath {
                    path: mount.host_root.clone(),
                    writable: mount.policy.writable,
                    known_absent: false,
                });
            }
            let relative = normalized
                .strip_prefix(&mount.guest_prefix)?
                .strip_prefix('/')?;
            resolve_path_beneath(&mount.host_root, relative).map(|path| ResolvedProbeGuestPath {
                path,
                writable: mount.policy.writable,
                known_absent: false,
            })
        }
    }
}

/// Host directory that holds committed save-data slots (`save_root/savedata/`).
///
/// Keeping slots out of the flat `save_root` removes the ambiguity that made `download0` look like
/// a save directory to `DirNameSearch` (audit F-01).
pub(crate) fn save_data_slots_root(save_root: &Path) -> PathBuf {
    save_root.join("savedata")
}

/// One-time migration: move legacy flat save slots into `save_root/savedata/`.
///
/// Leaves `download0`, `temp0`, `savedata`, and any `sce_*` bookkeeping directory alone.
pub(crate) fn migrate_legacy_save_slots(save_root: &Path) -> std::io::Result<()> {
    let slots_root = save_data_slots_root(save_root);
    std::fs::create_dir_all(&slots_root)?;
    if !save_root.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(save_root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let Some(name) = entry.file_name().to_str().map(ToOwned::to_owned) else {
            continue;
        };
        if name.eq_ignore_ascii_case("download0")
            || name.eq_ignore_ascii_case("temp0")
            || name.eq_ignore_ascii_case("savedata")
            || name.to_ascii_lowercase().starts_with("sce_")
        {
            continue;
        }
        let destination = slots_root.join(&name);
        if destination.exists() {
            eprintln!(
                "[REMu][VFS] legacy save slot {:?} already present under savedata/; leaving {:?}",
                name,
                entry.path().display()
            );
            continue;
        }
        std::fs::rename(entry.path(), &destination)?;
        eprintln!(
            "[REMu][VFS] migrated legacy save slot {:?} → {:?}",
            name,
            destination.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "remu-vfs-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn mounts_are_matched_by_longest_guest_prefix() {
        let mut vfs = VfsState::default();
        vfs.register(VfsMount {
            guest_prefix: "/download0".to_owned(),
            host_root: PathBuf::from("/host/download0"),
            policy: MountPolicy {
                writable: true,
                case_insensitive: true,
                enumerable_as_savedata: false,
                quota: None,
                kind: VfsMountKind::Download,
            },
        });
        vfs.register(VfsMount {
            guest_prefix: "/download0/deep".to_owned(),
            host_root: PathBuf::from("/host/deep"),
            policy: MountPolicy {
                writable: true,
                case_insensitive: true,
                enumerable_as_savedata: false,
                quota: None,
                kind: VfsMountKind::Download,
            },
        });
        assert_eq!(
            vfs.find_mount("/download0/deep/file.bin")
                .map(|mount| mount.host_root.as_path()),
            Some(Path::new("/host/deep"))
        );
        assert_eq!(
            vfs.find_mount("/download0/other")
                .map(|mount| mount.host_root.as_path()),
            Some(Path::new("/host/download0"))
        );
    }

    #[test]
    fn download_mount_is_not_enumerable_as_savedata() {
        let temporary = scratch_root("download-policy");
        let mut vfs = VfsState::default();
        vfs.rebuild_static_mounts(Path::new("/game"), &temporary);
        let download = vfs
            .find_mount("/download0")
            .expect("download0 must be registered");
        assert_eq!(download.policy.kind, VfsMountKind::Download);
        assert!(!download.policy.enumerable_as_savedata);
        assert!(download.policy.writable);
        assert_eq!(download.host_root, temporary.join("download0"));
        let _ = std::fs::remove_dir_all(temporary);
    }

    #[test]
    fn temp0_registers_as_writable_scratch_and_resolves() {
        let temporary = scratch_root("temp0");
        let mut vfs = VfsState::default();
        vfs.rebuild_static_mounts(Path::new("/game"), &temporary);
        assert!(vfs.find_mount("/temp0").is_none());
        let host = vfs.register_temp0(&temporary);
        std::fs::create_dir_all(&host).unwrap();
        std::fs::write(host.join("scratch.bin"), b"tmp").unwrap();
        let mount = vfs.find_mount("/temp0/scratch.bin").expect("temp0 mount");
        assert_eq!(mount.policy.kind, VfsMountKind::Scratch);
        assert!(mount.policy.writable);
        assert!(!mount.policy.enumerable_as_savedata);
        let resolved = resolve_via_vfs_mount(mount, "/temp0/scratch.bin").unwrap();
        assert_eq!(resolved.path, host.join("scratch.bin"));
        assert!(resolved.writable);
        assert_eq!(
            resolve_via_vfs_mount(mount, "/temp0/../../outside"),
            None,
            "temp0 must refuse parent escapes"
        );
        let _ = std::fs::remove_dir_all(temporary);
    }

    #[test]
    fn migrate_moves_legacy_slots_under_savedata_without_touching_download_or_temp() {
        let save_root = scratch_root("migrate");
        std::fs::create_dir_all(save_root.join("SAVEDATA00")).unwrap();
        std::fs::write(save_root.join("SAVEDATA00/save.bin"), b"slot").unwrap();
        std::fs::create_dir_all(save_root.join("download0")).unwrap();
        std::fs::create_dir_all(save_root.join("temp0")).unwrap();
        std::fs::create_dir_all(save_root.join("sce_transaction_1_1")).unwrap();

        migrate_legacy_save_slots(&save_root).unwrap();

        assert!(save_root.join("savedata/SAVEDATA00/save.bin").is_file());
        assert!(!save_root.join("SAVEDATA00").exists());
        assert!(save_root.join("download0").is_dir());
        assert!(save_root.join("temp0").is_dir());
        assert!(save_root.join("sce_transaction_1_1").is_dir());
        let _ = std::fs::remove_dir_all(save_root);
    }
}
