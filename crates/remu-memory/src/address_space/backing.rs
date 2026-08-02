// SPDX-License-Identifier: GPL-2.0-or-later

//! Shareable guest physical backing objects built on POSIX shared memory.

use std::ffi::CString;
use std::io;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{
    AddressSpaceError, GUEST_PAGE_SIZE, align_up, host_page_size, host_size, require_aligned,
};

static NEXT_BACKING_ID: AtomicU64 = AtomicU64::new(1);

// Runtime backing objects can span 16 GiB of direct memory or a 1 TiB sparse virtual arena, so
// mapping an entire object permanently would distort VM accounting. Retain only the hot physical
// windows used by repeated GPU reads and let the fixed limits bound both aliases and residency.
const READ_MAPPING_CACHE_CAPACITY: usize = 256;
const READ_MAPPING_CACHE_BYTE_CAPACITY: usize = 16 * 1024 * 1024;

#[derive(Debug)]
struct CachedReadMapping {
    address: usize,
    backing_offset: u64,
    byte_count: usize,
    last_use: u64,
}

impl CachedReadMapping {
    fn new(descriptor: i32, backing_offset: u64, byte_count: usize) -> Option<Self> {
        let file_offset = libc::off_t::try_from(backing_offset).ok()?;
        // SAFETY: the caller validated that this page-aligned file interval belongs to the live
        // shared-memory object. The mapping remains owned by this value until Drop.
        let mapping = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                byte_count,
                libc::PROT_READ,
                libc::MAP_SHARED,
                descriptor,
                file_offset,
            )
        };
        if mapping == libc::MAP_FAILED {
            return None;
        }
        Some(Self {
            address: mapping as usize,
            backing_offset,
            byte_count,
            last_use: 0,
        })
    }

    fn contains(&self, offset: u64, byte_count: usize) -> bool {
        let Ok(byte_count) = u64::try_from(byte_count) else {
            return false;
        };
        let Some(request_end) = offset.checked_add(byte_count) else {
            return false;
        };
        let Some(mapping_end) = self
            .backing_offset
            .checked_add(u64::try_from(self.byte_count).unwrap_or(u64::MAX))
        else {
            return false;
        };
        self.backing_offset <= offset && request_end <= mapping_end
    }

    fn copy_to(&self, offset: u64, destination: &mut [u8]) {
        debug_assert!(self.contains(offset, destination.len()));
        let in_mapping_offset =
            usize::try_from(offset - self.backing_offset).expect("contained offset fits usize");
        // SAFETY: `contains` proves this immutable mapping covers the requested physical range,
        // and `destination` provides an independently owned writable range of the same length.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (self.address as *const u8).add(in_mapping_offset),
                destination.as_mut_ptr(),
                destination.len(),
            );
        }
    }
}

impl Drop for CachedReadMapping {
    fn drop(&mut self) {
        // SAFETY: this value owns the complete mmap interval and releases it exactly once.
        unsafe {
            libc::munmap(self.address as *mut libc::c_void, self.byte_count);
        }
    }
}

#[derive(Debug, Default)]
struct ReadMappingCache {
    mappings: Vec<CachedReadMapping>,
    mapped_bytes: usize,
    next_use: u64,
    #[cfg(test)]
    hits: u64,
    #[cfg(test)]
    misses: u64,
}

impl ReadMappingCache {
    fn copy_if_present(&mut self, offset: u64, destination: &mut [u8]) -> bool {
        let Some(index) = self
            .mappings
            .iter()
            .position(|mapping| mapping.contains(offset, destination.len()))
        else {
            #[cfg(test)]
            {
                self.misses = self.misses.saturating_add(1);
            }
            return false;
        };
        self.next_use = self.next_use.wrapping_add(1);
        self.mappings[index].last_use = self.next_use;
        self.mappings[index].copy_to(offset, destination);
        #[cfg(test)]
        {
            self.hits = self.hits.saturating_add(1);
        }
        true
    }

    fn insert(&mut self, mut mapping: CachedReadMapping) {
        while !self.mappings.is_empty()
            && (self.mappings.len() >= READ_MAPPING_CACHE_CAPACITY
                || self.mapped_bytes.saturating_add(mapping.byte_count)
                    > READ_MAPPING_CACHE_BYTE_CAPACITY)
        {
            let oldest = self
                .mappings
                .iter()
                .enumerate()
                .min_by_key(|(_, mapping)| mapping.last_use)
                .map(|(index, _)| index)
                .expect("non-empty mapping cache has an oldest entry");
            let removed = self.mappings.swap_remove(oldest);
            self.mapped_bytes = self.mapped_bytes.saturating_sub(removed.byte_count);
        }
        self.next_use = self.next_use.wrapping_add(1);
        mapping.last_use = self.next_use;
        self.mapped_bytes = self.mapped_bytes.saturating_add(mapping.byte_count);
        self.mappings.push(mapping);
    }
}

#[derive(Debug)]
pub(super) struct BackingInner {
    pub(super) id: u64,
    size: u64,
    virtually_keyed: bool,
    pub(super) descriptor: i32,
    read_mappings: Mutex<ReadMappingCache>,
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

/// One page-aligned writable mapping of a shared guest physical backing.
///
/// This mapping is independent of the guest virtual aliases which exposed the backing range.
/// Consumers may retain it while asynchronous GPU work is in flight; dropping it releases the
/// host virtual mapping without changing the backing object or any guest alias.
#[derive(Debug)]
pub struct GuestSharedBackingView {
    address: usize,
    byte_count: usize,
}

impl GuestSharedBackingView {
    #[must_use]
    pub const fn as_ptr(&self) -> *const std::ffi::c_void {
        self.address as *const std::ffi::c_void
    }

    #[must_use]
    pub const fn byte_count(&self) -> usize {
        self.byte_count
    }
}

impl Drop for GuestSharedBackingView {
    fn drop(&mut self) {
        // SAFETY: this object owns the complete mmap interval and releases it exactly once.
        unsafe {
            libc::munmap(self.address as *mut libc::c_void, self.byte_count);
        }
    }
}

impl GuestPhysicalBacking {
    /// Creates a physically keyed backing: its offsets name stable guest physical bytes.
    pub fn new(size: u64) -> Result<Self, AddressSpaceError> {
        Self::create(size, false)
    }

    /// Creates a virtually keyed backing, whose offsets are guest virtual placements.
    ///
    /// Only an address space's committed-memory object may use this: byte `offset` of the object
    /// is guest address `base + offset`, so a recorded `(backing, offset)` pair names a *virtual*
    /// address whose contents the allocator may already have handed to another object.
    pub(super) fn new_virtually_keyed(size: u64) -> Result<Self, AddressSpaceError> {
        Self::create(size, true)
    }

    fn create(size: u64, virtually_keyed: bool) -> Result<Self, AddressSpaceError> {
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
                virtually_keyed,
                descriptor,
                read_mappings: Mutex::new(ReadMappingCache::default()),
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

    /// Reports whether this object is keyed by guest virtual placement rather than physical bytes.
    ///
    /// A recorded offset into such an object is not a durable physical identity, so no writer may
    /// republish captured bytes there once the naming mapping is gone.
    #[must_use]
    pub fn is_virtually_keyed(&self) -> bool {
        self.inner.virtually_keyed
    }

    pub(super) fn map_shared_view(
        &self,
        backing_offset: u64,
        byte_count: usize,
    ) -> Option<GuestSharedBackingView> {
        if byte_count == 0
            || backing_offset.checked_add(u64::try_from(byte_count).ok()?)? > self.size()
        {
            return None;
        }
        let file_offset = libc::off_t::try_from(backing_offset).ok()?;
        // SAFETY: the shared-range resolver validated the interval against this live backing.
        // The returned owner keeps the mapping alive until its eventual Drop.
        let mapping = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                byte_count,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                self.inner.descriptor,
                file_offset,
            )
        };
        if mapping == libc::MAP_FAILED {
            return None;
        }
        Some(GuestSharedBackingView {
            address: mapping as usize,
            byte_count,
        })
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
            .and_then(|length| align_up(length, page_size).ok())
            .and_then(|length| host_size(length).ok())
        else {
            return false;
        };
        if mapping_length <= READ_MAPPING_CACHE_BYTE_CAPACITY {
            let mut cache = self
                .inner
                .read_mappings
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if cache.copy_if_present(offset, destination) {
                return true;
            }
            let Some(mapping) =
                CachedReadMapping::new(self.inner.descriptor, aligned_offset, mapping_length)
            else {
                return false;
            };
            mapping.copy_to(offset, destination);
            cache.insert(mapping);
            return true;
        }
        let Some(mapping) =
            CachedReadMapping::new(self.inner.descriptor, aligned_offset, mapping_length)
        else {
            return false;
        };
        mapping.copy_to(offset, destination);
        true
    }

    #[cfg(test)]
    pub(super) fn read_mapping_cache_stats(&self) -> (usize, u64, u64) {
        let cache = self
            .inner
            .read_mappings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (cache.mappings.len(), cache.hits, cache.misses)
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
