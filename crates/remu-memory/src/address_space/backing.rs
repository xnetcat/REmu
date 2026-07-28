// SPDX-License-Identifier: GPL-2.0-or-later

//! Shareable guest physical backing objects built on POSIX shared memory.

use std::ffi::CString;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{AddressSpaceError, GUEST_PAGE_SIZE, host_page_size, host_size, require_aligned};

static NEXT_BACKING_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub(super) struct BackingInner {
    pub(super) id: u64,
    size: u64,
    pub(super) descriptor: i32,
}

impl Drop for BackingInner {
    fn drop(&mut self) {
        // SAFETY: The descriptor is owned by this object and closed exactly once.
        unsafe {
            libc::close(self.descriptor);
        }
    }
}

#[derive(Clone, Debug)]
pub struct GuestPhysicalBacking {
    pub(super) inner: Arc<BackingInner>,
}

impl GuestPhysicalBacking {
    pub fn new(size: u64) -> Result<Self, AddressSpaceError> {
        if size == 0 {
            return Err(AddressSpaceError::ZeroBackingSize);
        }
        require_aligned("backing size", size, GUEST_PAGE_SIZE)?;
        let truncate_size = libc::off_t::try_from(size)
            .map_err(|_| AddressSpaceError::HostSizeOverflow { value: size })?;
        let id = NEXT_BACKING_ID.fetch_add(1, Ordering::Relaxed);
        let name = CString::new(format!("/remu-{}-{id}", std::process::id())).map_err(|error| {
            AddressSpaceError::BackingOperationFailed {
                operation: "name generation",
                source: io::Error::new(io::ErrorKind::InvalidInput, error),
            }
        })?;

        // SAFETY: `name` is a live C string and the flags/mode are valid for shm_open.
        let descriptor = unsafe {
            libc::shm_open(
                name.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
                0o600,
            )
        };
        if descriptor < 0 {
            return Err(AddressSpaceError::BackingOperationFailed {
                operation: "creation",
                source: io::Error::last_os_error(),
            });
        }
        // The descriptor keeps the object alive; unlinking immediately prevents leaked names.
        // SAFETY: `name` remains valid for this call.
        unsafe {
            libc::shm_unlink(name.as_ptr());
        }
        // SAFETY: `descriptor` is live and `truncate_size` is representable by off_t.
        if unsafe { libc::ftruncate(descriptor, truncate_size) } != 0 {
            let source = io::Error::last_os_error();
            // SAFETY: The descriptor is owned locally because BackingInner was not constructed.
            unsafe {
                libc::close(descriptor);
            }
            return Err(AddressSpaceError::BackingOperationFailed {
                operation: "resize",
                source,
            });
        }

        Ok(Self {
            inner: Arc::new(BackingInner {
                id,
                size,
                descriptor,
            }),
        })
    }

    #[must_use]
    pub fn id(&self) -> u64 {
        self.inner.id
    }

    #[must_use]
    pub fn size(&self) -> u64 {
        self.inner.size
    }

    pub(super) fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> bool {
        let Some(end) = offset.checked_add(u64::try_from(destination.len()).unwrap_or(u64::MAX))
        else {
            return false;
        };
        if end > self.size() {
            return false;
        }
        let Ok(page_size) = host_page_size() else {
            return false;
        };
        let aligned_offset = offset - offset % page_size;
        let in_mapping_offset = offset - aligned_offset;
        let Some(mapping_length) = in_mapping_offset
            .checked_add(u64::try_from(destination.len()).unwrap_or(u64::MAX))
            .and_then(|length| host_size(length).ok())
        else {
            return false;
        };
        let Ok(file_offset) = libc::off_t::try_from(aligned_offset) else {
            return false;
        };
        // macOS POSIX shared-memory descriptors do not support pread/pwrite. Map only the
        // requested physical span and copy from that stable direct-backing view instead.
        // SAFETY: the aligned file range was validated against the live backing object.
        let mapping = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapping_length,
                libc::PROT_READ,
                libc::MAP_SHARED,
                self.inner.descriptor,
                file_offset,
            )
        };
        if mapping == libc::MAP_FAILED {
            return false;
        }
        // SAFETY: `mapping` covers the requested offset and destination length.
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapping.cast::<u8>().add(
                    usize::try_from(in_mapping_offset)
                        .expect("validated mapping offset fits usize"),
                ),
                destination.as_mut_ptr(),
                destination.len(),
            );
            libc::munmap(mapping, mapping_length);
        }
        true
    }

    pub(super) fn write_all_at(&self, offset: u64, source: &[u8]) -> bool {
        let Some(end) = offset.checked_add(u64::try_from(source.len()).unwrap_or(u64::MAX)) else {
            return false;
        };
        if end > self.size() {
            return false;
        }
        let Ok(page_size) = host_page_size() else {
            return false;
        };
        let aligned_offset = offset - offset % page_size;
        let in_mapping_offset = offset - aligned_offset;
        let Some(mapping_length) = in_mapping_offset
            .checked_add(u64::try_from(source.len()).unwrap_or(u64::MAX))
            .and_then(|length| host_size(length).ok())
        else {
            return false;
        };
        let Ok(file_offset) = libc::off_t::try_from(aligned_offset) else {
            return false;
        };
        // SAFETY: the aligned file range was validated against the live backing object.
        let mapping = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapping_length,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                self.inner.descriptor,
                file_offset,
            )
        };
        if mapping == libc::MAP_FAILED {
            return false;
        }
        // SAFETY: `mapping` covers the requested offset and source length.
        unsafe {
            std::ptr::copy_nonoverlapping(
                source.as_ptr(),
                mapping.cast::<u8>().add(
                    usize::try_from(in_mapping_offset)
                        .expect("validated mapping offset fits usize"),
                ),
                source.len(),
            );
            libc::munmap(mapping, mapping_length);
        }
        true
    }
}
