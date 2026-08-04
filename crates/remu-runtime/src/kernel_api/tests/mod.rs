mod aio;
mod event;
mod filesystem;
mod net;
mod thread;

use std::io::Seek;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

use super::*;
use crate::{
    DEFAULT_THREAD_AFFINITY_MASK, DEFAULT_THREAD_PRIORITY, KernelQueuedEvent, PthreadAttrState,
    PthreadThreadObject, SaveDataMount,
};

const SUCCESS: u64 = 0;
const KERNEL_NOT_FOUND: u64 = 0xffff_ffff_8002_0002;
const KERNEL_BAD_FILE_DESCRIPTOR: u64 = 0xffff_ffff_8002_0009;
const KERNEL_PERMISSION_DENIED: u64 = 0xffff_ffff_8002_000d;
const KERNEL_MEMORY_FAULT: u64 = 0xffff_ffff_8002_000e;
const KERNEL_IS_DIRECTORY: u64 = 0xffff_ffff_8002_0015;
const KERNEL_INVALID_ARGUMENT: u64 = 0xffff_ffff_8002_0016;
const POSIX_FAILURE: u64 = u64::MAX;

/// A directory removed when the test that made it ends.
struct TemporaryDirectory {
    path: std::path::PathBuf,
}

impl TemporaryDirectory {
    fn new(label: &str) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "remu-kernel-api-{label}-{}-{id}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("the test directory can be created");
        Self { path }
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// A writable guest mount backed by a temporary host directory.
struct MountFixture {
    directory: TemporaryDirectory,
    state: ProbeDispatchState,
}

impl MountFixture {
    fn new(label: &str) -> Self {
        Self::with_read_only(label, false)
    }

    fn with_read_only(label: &str, read_only: bool) -> Self {
        let directory = TemporaryDirectory::new(label);
        let mut state = ProbeDispatchState::default();
        state.save_mounts.insert(
            "/savedata0".to_owned(),
            SaveDataMount {
                path: directory.path.clone(),
                committed_path: directory.path.clone(),
                dir_name: "test".to_owned(),
                user_id: 1,
                resource: 1,
                read_only,
            },
        );
        if !read_only {
            state.prepared_save_resources.insert(1);
        }
        Self { directory, state }
    }

    fn write(&self, name: &str, contents: &[u8]) -> std::path::PathBuf {
        let path = self.directory.path.join(name);
        std::fs::write(&path, contents).expect("the test file can be written");
        path
    }

    fn call(&mut self, nid: &str, arguments: [u64; 6]) -> u64 {
        dispatch(nid, arguments, &mut self.state).expect("the test names an owned NID")
    }
}

fn args(values: &[u64]) -> [u64; 6] {
    let mut arguments = [0_u64; 6];
    arguments[..values.len()].copy_from_slice(values);
    arguments
}

fn guest_string(text: &str) -> Vec<u8> {
    let mut bytes = text.as_bytes().to_vec();
    bytes.push(0);
    bytes
}

fn address(bytes: &[u8]) -> u64 {
    bytes.as_ptr().addr() as u64
}

/// Registers an open host file in the shared descriptor table and reports its guest fd.
fn register_descriptor(file: std::fs::File) -> i32 {
    let mut files = kernel_files();
    let fd = files.next_fd;
    files.next_fd = fd.wrapping_add(1);
    files.files.insert(fd, file);
    fd
}

fn release_descriptor(fd: i32) {
    kernel_files().files.remove(&fd);
}
