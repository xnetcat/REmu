//! Journaled save-data commit + APFS clone for Prepare (plan G3 / audit B27 R3).
//!
//! Commit sequence (each step is named so a test can stop between any two):
//! 1. sync staging tree
//! 2. rename staging → `sce_commit_new_<txn>`
//! 3. fsync parent; write `sce_commit_journal_<txn>` naming (slot, new, backup); fsync journal
//! 4. rename committed → `sce_commit_backup_<txn>`
//! 5. rename new → committed
//! 6. fsync parent; unlink journal; remove backup
//!
//! A crash between the two renames previously left an invisible save (hidden by the `sce_`
//! enumerate filter) with a complete copy beside it. Recovery in `save_data_initialize` rolls
//! the journal forward or back.
//!
//! Prepare uses `clonefile(2)` on macOS (APFS copy-on-write) with a recursive-copy fallback.
//! SDK: `SCE_SAVE_DATA_ERROR_NO_COWBLK` (`save_data_error.h:179`) documents the console's COW
//! block budget; REmu's host clone is the analogous fast path, not a claim about console blocks.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::SaveDataTransaction;

/// Flushes directory namespace changes where the host exposes that primitive.
///
/// Unix provides directory `fsync`. Windows does not expose an equivalent durable directory flush
/// through ordinary unprivileged file handles; file payloads are still flushed before renames and
/// NTFS journals the namespace operations. Opening the directory with backup semantics verifies
/// that it remains reachable without turning the unsupported flush into a save-data failure.
pub(crate) fn sync_directory(directory: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        return File::open(directory)?.sync_all();
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;

        const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
        const FILE_SHARE_READ_WRITE_DELETE: u32 = 0x0000_0007;
        std::fs::OpenOptions::new()
            .access_mode(0)
            .share_mode(FILE_SHARE_READ_WRITE_DELETE)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(directory)?;
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = directory;
        Ok(())
    }
}

fn sync_file(path: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        // FlushFileBuffers, which backs File::sync_all on Windows, requires a handle opened with
        // write access even when no bytes are changed by this operation.
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?
            .sync_all()
    }
    #[cfg(not(windows))]
    {
        File::open(path)?.sync_all()
    }
}

/// Named steps of a journaled commit. Tests drive the sequence one step at a time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CommitStep {
    SyncStaging,
    PromoteStagingToNew,
    WriteJournal,
    RenameCommittedToBackup,
    RenameNewToCommitted,
    SyncParent,
    UnlinkJournal,
    RemoveBackup,
}

impl CommitStep {
    pub(crate) const ALL: [CommitStep; 8] = [
        CommitStep::SyncStaging,
        CommitStep::PromoteStagingToNew,
        CommitStep::WriteJournal,
        CommitStep::RenameCommittedToBackup,
        CommitStep::RenameNewToCommitted,
        CommitStep::SyncParent,
        CommitStep::UnlinkJournal,
        CommitStep::RemoveBackup,
    ];
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CommitPaths {
    pub(crate) committed: PathBuf,
    pub(crate) staging: PathBuf,
    pub(crate) new_path: PathBuf,
    pub(crate) backup_path: PathBuf,
    pub(crate) journal_path: PathBuf,
}

impl CommitPaths {
    pub(crate) fn for_transaction(transaction: &SaveDataTransaction) -> Self {
        let parent = transaction
            .committed_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let file_name = transaction
            .committed_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("save");
        let txn_tag = transaction
            .staging_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("transaction");
        let pid = std::process::id();
        Self {
            committed: transaction.committed_path.clone(),
            staging: transaction.staging_path.clone(),
            new_path: parent.join(format!("sce_commit_new_{pid}_{file_name}_{txn_tag}")),
            backup_path: parent.join(format!("sce_commit_backup_{pid}_{file_name}_{txn_tag}")),
            journal_path: parent.join(format!("sce_commit_journal_{pid}_{file_name}_{txn_tag}")),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct JournalRecord {
    committed: PathBuf,
    new_path: PathBuf,
    backup_path: PathBuf,
}

/// Run the full journaled commit.
pub(crate) fn commit_save_data_transaction_journaled(
    transaction: &SaveDataTransaction,
) -> std::io::Result<()> {
    let paths = CommitPaths::for_transaction(transaction);
    let mut state = CommitInFlight {
        paths,
        staging_promoted: false,
        journal_written: false,
        committed_backed_up: false,
        new_installed: false,
    };
    for step in CommitStep::ALL {
        run_commit_step(step, &mut state)?;
    }
    Ok(())
}

struct CommitInFlight {
    paths: CommitPaths,
    staging_promoted: bool,
    journal_written: bool,
    committed_backed_up: bool,
    new_installed: bool,
}

fn run_commit_step(step: CommitStep, state: &mut CommitInFlight) -> std::io::Result<()> {
    match step {
        CommitStep::SyncStaging => {
            let source = if state.staging_promoted {
                &state.paths.new_path
            } else {
                &state.paths.staging
            };
            sync_tree(source)
        }
        CommitStep::PromoteStagingToNew => {
            if state.paths.new_path.exists() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!(
                        "commit new path already exists: {}",
                        state.paths.new_path.display()
                    ),
                ));
            }
            std::fs::rename(&state.paths.staging, &state.paths.new_path)?;
            state.staging_promoted = true;
            Ok(())
        }
        CommitStep::WriteJournal => {
            if let Some(parent) = state.paths.journal_path.parent() {
                sync_directory(parent)?;
            }
            write_journal(
                &state.paths.journal_path,
                &JournalRecord {
                    committed: state.paths.committed.clone(),
                    new_path: state.paths.new_path.clone(),
                    backup_path: state.paths.backup_path.clone(),
                },
            )?;
            state.journal_written = true;
            Ok(())
        }
        CommitStep::RenameCommittedToBackup => {
            if state.paths.backup_path.exists() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!(
                        "commit backup path already exists: {}",
                        state.paths.backup_path.display()
                    ),
                ));
            }
            std::fs::rename(&state.paths.committed, &state.paths.backup_path)?;
            state.committed_backed_up = true;
            Ok(())
        }
        CommitStep::RenameNewToCommitted => {
            if let Err(error) = std::fs::rename(&state.paths.new_path, &state.paths.committed) {
                // Roll the first rename back so the slot is not left invisible.
                let _ = std::fs::rename(&state.paths.backup_path, &state.paths.committed);
                return Err(error);
            }
            state.new_installed = true;
            Ok(())
        }
        CommitStep::SyncParent => {
            if let Some(parent) = state.paths.committed.parent() {
                sync_directory(parent)?;
            }
            Ok(())
        }
        CommitStep::UnlinkJournal => {
            if state.paths.journal_path.exists() {
                std::fs::remove_file(&state.paths.journal_path)?;
            }
            Ok(())
        }
        CommitStep::RemoveBackup => {
            if state.paths.backup_path.exists() {
                std::fs::remove_dir_all(&state.paths.backup_path)?;
            }
            Ok(())
        }
    }
}

fn write_journal(path: &Path, record: &JournalRecord) -> std::io::Result<()> {
    let body = format!(
        "version=1\ncommitted={}\nnew={}\nbackup={}\n",
        record.committed.display(),
        record.new_path.display(),
        record.backup_path.display()
    );
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    file.write_all(body.as_bytes())?;
    file.sync_all()
}

fn read_journal(path: &Path) -> std::io::Result<JournalRecord> {
    let mut body = String::new();
    File::open(path)?.read_to_string(&mut body)?;
    let mut committed = None;
    let mut new_path = None;
    let mut backup_path = None;
    let mut version_ok = false;
    for line in body.lines() {
        if let Some(value) = line.strip_prefix("version=") {
            version_ok = value.trim() == "1";
        } else if let Some(value) = line.strip_prefix("committed=") {
            committed = Some(PathBuf::from(value));
        } else if let Some(value) = line.strip_prefix("new=") {
            new_path = Some(PathBuf::from(value));
        } else if let Some(value) = line.strip_prefix("backup=") {
            backup_path = Some(PathBuf::from(value));
        }
    }
    if !version_ok {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unsupported save commit journal at {}", path.display()),
        ));
    }
    match (committed, new_path, backup_path) {
        (Some(committed), Some(new_path), Some(backup_path)) => Ok(JournalRecord {
            committed,
            new_path,
            backup_path,
        }),
        _ => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("incomplete save commit journal at {}", path.display()),
        )),
    }
}

/// Scan `root` (and its `savedata/` child) for leftover journals and orphaned staging trees.
pub(crate) fn recover_save_data_root(root: &Path) -> std::io::Result<()> {
    if !root.is_dir() {
        return Ok(());
    }
    recover_journals_in(root)?;
    let slots = root.join("savedata");
    if slots.is_dir() {
        recover_journals_in(&slots)?;
    }
    delete_orphaned_transactions(root)?;
    if slots.is_dir() {
        delete_orphaned_transactions(&slots)?;
    }
    Ok(())
}

fn recover_journals_in(directory: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !name.starts_with("sce_commit_journal_") || !entry.file_type()?.is_file() {
            continue;
        }
        let journal_path = entry.path();
        match read_journal(&journal_path) {
            Ok(record) => {
                if let Err(error) = recover_journal_record(&record) {
                    eprintln!(
                        "[REMu][SAVEDATA] journal recovery failed journal={:?}: {error}",
                        journal_path.display()
                    );
                    continue;
                }
                let _ = std::fs::remove_file(&journal_path);
                eprintln!(
                    "[REMu][SAVEDATA] recovered journal {:?}",
                    journal_path.display()
                );
            }
            Err(error) => {
                eprintln!(
                    "[REMu][SAVEDATA] unreadable commit journal {:?}: {error}",
                    journal_path.display()
                );
            }
        }
    }
    Ok(())
}

fn recover_journal_record(record: &JournalRecord) -> std::io::Result<()> {
    // Committed present → prior commit finished the critical renames; discard leftovers.
    if record.committed.is_dir() {
        if record.new_path.exists() {
            std::fs::remove_dir_all(&record.new_path)?;
        }
        if record.backup_path.exists() {
            std::fs::remove_dir_all(&record.backup_path)?;
        }
        return Ok(());
    }
    // Committed missing, new present → finish installing the new tree.
    if record.new_path.is_dir() {
        std::fs::rename(&record.new_path, &record.committed)?;
        if record.backup_path.exists() {
            std::fs::remove_dir_all(&record.backup_path)?;
        }
        if let Some(parent) = record.committed.parent() {
            sync_directory(parent)?;
        }
        return Ok(());
    }
    // Only backup remains → restore the previous committed tree.
    if record.backup_path.is_dir() {
        std::fs::rename(&record.backup_path, &record.committed)?;
        if let Some(parent) = record.committed.parent() {
            sync_directory(parent)?;
        }
        return Ok(());
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!(
            "journal names no recoverable tree for {}",
            record.committed.display()
        ),
    ))
}

fn delete_orphaned_transactions(directory: &Path) -> std::io::Result<()> {
    let live_pid = std::process::id();
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(rest) = name.strip_prefix("sce_transaction_") else {
            continue;
        };
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let Some((pid_text, _)) = rest.split_once('_') else {
            continue;
        };
        let Ok(pid) = pid_text.parse::<u32>() else {
            continue;
        };
        if pid == live_pid || process_exists(pid) {
            continue;
        }
        match std::fs::remove_dir_all(entry.path()) {
            Ok(()) => eprintln!(
                "[REMu][SAVEDATA] removed orphaned staging {:?} from dead pid={pid}",
                entry.path().display()
            ),
            Err(error) => eprintln!(
                "[REMu][SAVEDATA] could not remove orphaned staging {:?}: {error}",
                entry.path().display()
            ),
        }
    }
    Ok(())
}

#[cfg(unix)]
fn process_exists(pid: u32) -> bool {
    // SAFETY: kill(pid, 0) is the POSIX existence probe; it does not signal the target.
    let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
    result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(windows)]
fn process_exists(pid: u32) -> bool {
    use windows::Win32::Foundation::{CloseHandle, E_ACCESSDENIED};
    use windows::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};

    // SAFETY: this requests a query-only handle and never signals or mutates the target process.
    match unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) } {
        Ok(handle) => {
            // SAFETY: OpenProcess returned one owned kernel handle, released exactly once here.
            let _ = unsafe { CloseHandle(handle) };
            true
        }
        Err(error) => error.code() == E_ACCESSDENIED,
    }
}

#[cfg(not(any(unix, windows)))]
fn process_exists(_pid: u32) -> bool {
    false
}

/// Clone `source` onto `destination` via `clonefile(2)` on macOS, else recursive copy.
pub(crate) fn clone_or_copy_tree(source: &Path, destination: &Path) -> std::io::Result<()> {
    if destination.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!(
                "prepare destination already exists: {}",
                destination.display()
            ),
        ));
    }
    #[cfg(target_os = "macos")]
    {
        match clonefile_tree(source, destination) {
            Ok(()) => return Ok(()),
            Err(error) => {
                eprintln!(
                    "[REMu][SAVEDATA] clonefile(2) unavailable for {:?} → {:?}: {error}; falling back to copy",
                    source.display(),
                    destination.display()
                );
                let _ = std::fs::remove_dir_all(destination);
            }
        }
    }
    copy_tree(source, destination)
}

#[cfg(target_os = "macos")]
fn clonefile_tree(source: &Path, destination: &Path) -> std::io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let source_c = CString::new(source.as_os_str().as_bytes()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "save path contains interior NUL",
        )
    })?;
    let destination_c = CString::new(destination.as_os_str().as_bytes()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "save path contains interior NUL",
        )
    })?;
    // SAFETY: both paths are owned CStrings; clonefile does not retain the pointers.
    // Darwin declares clonefile(2) in <sys/clonefile.h>; libc may not always bind it.
    let status = unsafe { remu_clonefile(source_c.as_ptr(), destination_c.as_ptr(), 0) };
    if status == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    #[link_name = "clonefile"]
    fn remu_clonefile(
        src: *const libc::c_char,
        dst: *const libc::c_char,
        flags: u32,
    ) -> libc::c_int;
}

fn copy_tree(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::fs::create_dir(destination)?;
    let mut pending = vec![(source.to_owned(), destination.to_owned())];
    while let Some((source_directory, destination_directory)) = pending.pop() {
        for entry in std::fs::read_dir(source_directory)? {
            let entry = entry?;
            let source_path = entry.path();
            let destination_path = destination_directory.join(entry.file_name());
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                std::fs::create_dir(&destination_path)?;
                pending.push((source_path, destination_path));
            } else if file_type.is_file() {
                std::fs::copy(source_path, destination_path)?;
            }
        }
    }
    Ok(())
}

fn sync_tree(root: &Path) -> std::io::Result<()> {
    let mut directories = vec![root.to_owned()];
    let mut index = 0;
    while index < directories.len() {
        for entry in std::fs::read_dir(&directories[index])? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                directories.push(entry.path());
            } else if file_type.is_file() {
                sync_file(&entry.path())?;
            }
        }
        index += 1;
    }
    for directory in directories.into_iter().rev() {
        sync_directory(&directory)?;
    }
    Ok(())
}

/// Test helper: run commit steps up to and including `stop_after`, then leave the in-flight state.
#[cfg(test)]
pub(crate) fn commit_until(
    transaction: &SaveDataTransaction,
    stop_after: CommitStep,
) -> std::io::Result<CommitPaths> {
    let paths = CommitPaths::for_transaction(transaction);
    let mut state = CommitInFlight {
        paths: paths.clone(),
        staging_promoted: false,
        journal_written: false,
        committed_backed_up: false,
        new_installed: false,
    };
    for step in CommitStep::ALL {
        run_commit_step(step, &mut state)?;
        if step == stop_after {
            break;
        }
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SaveDataTransaction;

    fn scratch(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "remu-save-journal-{label}-{}-{}",
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

    fn seed_slot(root: &Path, name: &str, payload: &[u8]) -> (PathBuf, PathBuf) {
        let committed = root.join(name);
        std::fs::create_dir_all(committed.join("sce_sys")).unwrap();
        std::fs::write(committed.join("save.bin"), payload).unwrap();
        let staging = root.join(format!("sce_transaction_{}_1", std::process::id()));
        copy_tree(&committed, &staging).unwrap();
        std::fs::write(staging.join("save.bin"), b"staged-new").unwrap();
        (committed, staging)
    }

    #[test]
    fn a_crash_between_renames_is_recovered_by_the_startup_scan() {
        let root = scratch("between-renames");
        let (committed, staging) = seed_slot(&root, "SAVEDATA00", b"old");
        let transaction = SaveDataTransaction {
            mount_point: "/savedata0".to_owned(),
            committed_path: committed.clone(),
            staging_path: staging,
        };
        let paths = commit_until(&transaction, CommitStep::RenameCommittedToBackup).unwrap();
        assert!(!paths.committed.exists());
        assert!(paths.backup_path.is_dir());
        assert!(paths.new_path.is_dir());
        assert!(paths.journal_path.is_file());

        recover_save_data_root(&root).unwrap();
        assert!(paths.committed.is_dir());
        assert_eq!(
            std::fs::read(paths.committed.join("save.bin")).unwrap(),
            b"staged-new"
        );
        assert!(!paths.journal_path.exists());
        assert!(!paths.backup_path.exists());
        assert!(!paths.new_path.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_journal_with_only_backup_restores_the_previous_committed_tree() {
        let root = scratch("backup-only");
        let (committed, staging) = seed_slot(&root, "SAVEDATA01", b"keep-me");
        let transaction = SaveDataTransaction {
            mount_point: "/savedata0".to_owned(),
            committed_path: committed.clone(),
            staging_path: staging.clone(),
        };
        let paths = commit_until(&transaction, CommitStep::RenameCommittedToBackup).unwrap();
        std::fs::remove_dir_all(&paths.new_path).unwrap();
        assert!(!paths.committed.exists());
        assert!(paths.backup_path.is_dir());

        recover_save_data_root(&root).unwrap();
        assert_eq!(
            std::fs::read(committed.join("save.bin")).unwrap(),
            b"keep-me"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn prepare_clone_or_copy_produces_an_independent_tree() {
        let root = scratch("clone-prepare");
        let source = root.join("slot");
        std::fs::create_dir_all(source.join("sce_sys")).unwrap();
        std::fs::write(source.join("save.bin"), b"payload").unwrap();
        let destination = root.join("sce_transaction_clone");
        clone_or_copy_tree(&source, &destination).unwrap();
        assert_eq!(
            std::fs::read(destination.join("save.bin")).unwrap(),
            b"payload"
        );
        std::fs::write(destination.join("save.bin"), b"mutated").unwrap();
        assert_eq!(std::fs::read(source.join("save.bin")).unwrap(), b"payload");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn orphaned_staging_from_a_dead_pid_is_removed() {
        let root = scratch("orphan-staging");
        let live_pid = std::process::id();
        let live = root.join(format!("sce_transaction_{live_pid}_9"));
        std::fs::create_dir_all(&live).unwrap();
        std::fs::write(live.join("x.bin"), b"x").unwrap();
        // Use a synthetic high PID unlikely to exist on any supported host.
        let dead = root.join("sce_transaction_2147483646_9");
        std::fs::create_dir_all(&dead).unwrap();
        recover_save_data_root(&root).unwrap();
        assert!(
            live.exists(),
            "the current process is live; its staging must not be deleted"
        );
        assert!(
            !dead.exists(),
            "staging from a dead pid must be deleted on recovery"
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
