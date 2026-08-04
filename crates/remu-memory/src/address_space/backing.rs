// SPDX-License-Identifier: GPL-2.0-or-later

//! Sparse, shareable guest physical backing objects.

#[cfg(target_os = "macos")]
use std::ffi::CString;
use std::io;
#[cfg(windows)]
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(windows)]
use windows::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
#[cfg(windows)]
use windows::Win32::System::Memory::{
    CreateFileMappingW, MEM_COMMIT, MapViewOfFile3, PAGE_EXECUTE_READWRITE, PAGE_PROTECTION_FLAGS,
    PAGE_READWRITE, SEC_RESERVE, UnmapViewOfFile, VIRTUAL_ALLOCATION_TYPE, VirtualAlloc,
};

use super::{
    AddressSpaceError, GUEST_PAGE_SIZE, align_up, host_page_size, host_size, require_aligned,
};

static NEXT_BACKING_ID: AtomicU64 = AtomicU64::new(1);

#[cfg(windows)]
#[derive(Clone, Copy, Debug)]
struct DiagnosticBackingWriteWatch {
    backing_id: u64,
    offset: u64,
    byte_count: u64,
}

#[cfg(windows)]
fn diagnostic_backing_write_watch() -> &'static Mutex<Option<DiagnosticBackingWriteWatch>> {
    static WATCH: OnceLock<Mutex<Option<DiagnosticBackingWriteWatch>>> = OnceLock::new();
    WATCH.get_or_init(|| Mutex::new(None))
}

#[cfg(windows)]
fn trace_diagnostic_backing_write(backing_id: u64, offset: u64, source: &[u8]) {
    let watch = *diagnostic_backing_write_watch()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(watch) = watch else {
        return;
    };
    let Some(end) = offset.checked_add(source.len() as u64) else {
        return;
    };
    let watch_end = watch.offset + watch.byte_count;
    if backing_id != watch.backing_id || offset >= watch_end || watch.offset >= end {
        return;
    }
    let mut prefix = [0_u8; 8];
    let copied = prefix.len().min(source.len());
    prefix[..copied].copy_from_slice(&source[..copied]);
    eprintln!(
        "[REMu][JOURNAL][BACKING-WRITE] backing={backing_id} offset={offset:#x} bytes={:#x} payload={:#x} thread={:?}\n{}",
        source.len(),
        u64::from_le_bytes(prefix),
        std::thread::current().name(),
        std::backtrace::Backtrace::force_capture(),
    );
}

const READ_MAPPING_CACHE_CAPACITY: usize = 256;
const READ_MAPPING_CACHE_BYTE_CAPACITY: usize = 16 * 1024 * 1024;
#[cfg(windows)]
const WINDOWS_ALLOCATION_GRANULARITY: u64 = 64 * 1024;

#[derive(Debug)]
struct HostBackingMapping {
    mapping_address: usize,
    #[cfg(target_os = "macos")]
    mapping_byte_count: usize,
    address: usize,
}

impl HostBackingMapping {
    fn new(inner: &BackingInner, backing_offset: u64, byte_count: usize) -> Option<Self> {
        #[cfg(target_os = "macos")]
        {
            let file_offset = libc::off_t::try_from(backing_offset).ok()?;
            // SAFETY: The caller validates the page-aligned section interval.
            let mapping = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    byte_count,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    inner.descriptor,
                    file_offset,
                )
            };
            if mapping == libc::MAP_FAILED {
                return None;
            }
            return Some(Self {
                mapping_address: mapping as usize,
                mapping_byte_count: byte_count,
                address: mapping as usize,
            });
        }

        #[cfg(windows)]
        {
            let aligned_offset = backing_offset - backing_offset % WINDOWS_ALLOCATION_GRANULARITY;
            let in_mapping_offset = usize::try_from(backing_offset - aligned_offset).ok()?;
            let page_size = host_page_size().ok()?;
            let mapping_byte_count = u64::try_from(in_mapping_offset)
                .ok()?
                .checked_add(u64::try_from(byte_count).ok()?)
                .and_then(|size| align_up(size, page_size).ok())
                .and_then(|size| host_size(size).ok())?;
            // SAFETY: The range is within a live sparse pagefile-backed section. A null base lets
            // Windows choose a suitably aligned temporary address.
            let mapping = unsafe {
                MapViewOfFile3(
                    inner.windows_handle(),
                    None,
                    None,
                    aligned_offset,
                    mapping_byte_count,
                    VIRTUAL_ALLOCATION_TYPE(0),
                    PAGE_READWRITE.0,
                    None,
                )
            };
            if mapping.Value.is_null() {
                return None;
            }
            let committed = unsafe {
                VirtualAlloc(
                    Some(mapping.Value.cast_const()),
                    mapping_byte_count,
                    MEM_COMMIT,
                    PAGE_READWRITE,
                )
            };
            if committed != mapping.Value {
                let _ = unsafe { UnmapViewOfFile(mapping) };
                return None;
            }
            Some(Self {
                mapping_address: mapping.Value as usize,
                address: mapping.Value as usize + in_mapping_offset,
            })
        }
    }
}

impl Drop for HostBackingMapping {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        unsafe {
            libc::munmap(
                self.mapping_address as *mut libc::c_void,
                self.mapping_byte_count,
            );
        }
        #[cfg(windows)]
        {
            let _ = unsafe {
                UnmapViewOfFile(windows::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: self.mapping_address as *mut std::ffi::c_void,
                })
            };
        }
    }
}

#[derive(Debug)]
struct CachedReadMapping {
    host: HostBackingMapping,
    backing_offset: u64,
    byte_count: usize,
    last_use: u64,
}

impl CachedReadMapping {
    fn new(inner: &BackingInner, backing_offset: u64, byte_count: usize) -> Option<Self> {
        Some(Self {
            host: HostBackingMapping::new(inner, backing_offset, byte_count)?,
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
        // SAFETY: `contains` proves the temporary mapping covers the requested range.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (self.host.address as *const u8).add(in_mapping_offset),
                destination.as_mut_ptr(),
                destination.len(),
            );
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

/// One page-aligned writable window retained across writebacks to the same physical range.
///
/// It mirrors [`CachedReadMapping`] but maps read/write so publications reuse it. Being a
/// `MAP_SHARED` window over the same object, its stores are coherent with every read mapping and
/// guest alias of that object.
#[derive(Debug)]
struct CachedWriteMapping {
    host: HostBackingMapping,
    backing_offset: u64,
    byte_count: usize,
    last_use: u64,
}

impl CachedWriteMapping {
    fn new(inner: &BackingInner, backing_offset: u64, byte_count: usize) -> Option<Self> {
        Some(Self {
            host: HostBackingMapping::new(inner, backing_offset, byte_count)?,
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

    fn copy_from(&self, offset: u64, source: &[u8]) {
        debug_assert!(self.contains(offset, source.len()));
        let in_mapping_offset =
            usize::try_from(offset - self.backing_offset).expect("contained offset fits usize");
        // SAFETY: `contains` proves this writable mapping covers the requested physical range, and
        // `source` provides an independently owned readable range of the same length.
        unsafe {
            std::ptr::copy_nonoverlapping(
                source.as_ptr(),
                (self.host.address as *mut u8).add(in_mapping_offset),
                source.len(),
            );
        }
    }
}

#[derive(Debug, Default)]
struct WriteMappingCache {
    mappings: Vec<CachedWriteMapping>,
    mapped_bytes: usize,
    next_use: u64,
    #[cfg(test)]
    hits: u64,
    #[cfg(test)]
    misses: u64,
}

impl WriteMappingCache {
    fn copy_if_present(&mut self, offset: u64, source: &[u8]) -> bool {
        let Some(index) = self
            .mappings
            .iter()
            .position(|mapping| mapping.contains(offset, source.len()))
        else {
            #[cfg(test)]
            {
                self.misses = self.misses.saturating_add(1);
            }
            return false;
        };
        self.next_use = self.next_use.wrapping_add(1);
        self.mappings[index].last_use = self.next_use;
        self.mappings[index].copy_from(offset, source);
        #[cfg(test)]
        {
            self.hits = self.hits.saturating_add(1);
        }
        true
    }

    fn insert(&mut self, mut mapping: CachedWriteMapping) {
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
    #[cfg(target_os = "macos")]
    pub(super) descriptor: i32,
    #[cfg(windows)]
    handle: isize,
    read_mappings: Mutex<ReadMappingCache>,
    write_mappings: Mutex<WriteMappingCache>,
}

impl BackingInner {
    #[cfg(windows)]
    fn windows_handle(&self) -> HANDLE {
        HANDLE(self.handle as *mut std::ffi::c_void)
    }
}

impl Drop for BackingInner {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        unsafe {
            libc::close(self.descriptor);
        }
        #[cfg(windows)]
        {
            let _ = unsafe { CloseHandle(self.windows_handle()) };
        }
    }
}

#[derive(Clone, Debug)]
pub struct GuestPhysicalBacking {
    pub(super) inner: Arc<BackingInner>,
}

#[derive(Debug)]
pub struct GuestSharedBackingView {
    host: HostBackingMapping,
    byte_count: usize,
}

impl GuestSharedBackingView {
    #[must_use]
    pub const fn as_ptr(&self) -> *const std::ffi::c_void {
        self.host.address as *const std::ffi::c_void
    }

    #[must_use]
    pub const fn byte_count(&self) -> usize {
        self.byte_count
    }
}

impl GuestPhysicalBacking {
    pub fn new(size: u64) -> Result<Self, AddressSpaceError> {
        Self::create(size, false)
    }

    pub(super) fn new_virtually_keyed(size: u64) -> Result<Self, AddressSpaceError> {
        Self::create(size, true)
    }

    fn create(size: u64, virtually_keyed: bool) -> Result<Self, AddressSpaceError> {
        if size == 0 {
            return Err(AddressSpaceError::ZeroBackingSize);
        }
        require_aligned("backing size", size, GUEST_PAGE_SIZE)?;
        let id = NEXT_BACKING_ID.fetch_add(1, Ordering::Relaxed);

        #[cfg(target_os = "macos")]
        let descriptor = {
            let truncate_size = libc::off_t::try_from(size)
                .map_err(|_| AddressSpaceError::HostSizeOverflow { value: size })?;
            let name =
                CString::new(format!("/remu-{}-{id}", std::process::id())).map_err(|error| {
                    AddressSpaceError::BackingOperationFailed {
                        operation: "name generation",
                        source: io::Error::new(io::ErrorKind::InvalidInput, error),
                    }
                })?;
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
            unsafe {
                libc::shm_unlink(name.as_ptr());
            }
            if unsafe { libc::ftruncate(descriptor, truncate_size) } != 0 {
                let source = io::Error::last_os_error();
                unsafe {
                    libc::close(descriptor);
                }
                return Err(AddressSpaceError::BackingOperationFailed {
                    operation: "resize",
                    source,
                });
            }
            descriptor
        };

        #[cfg(windows)]
        let handle = {
            let protection = PAGE_PROTECTION_FLAGS(PAGE_EXECUTE_READWRITE.0 | SEC_RESERVE.0);
            let result = unsafe {
                CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    None,
                    protection,
                    (size >> 32) as u32,
                    size as u32,
                    windows::core::PCWSTR::null(),
                )
            };
            let handle = result.map_err(|_| AddressSpaceError::BackingOperationFailed {
                operation: "creation",
                source: io::Error::last_os_error(),
            })?;
            handle.0 as isize
        };

        Ok(Self {
            inner: Arc::new(BackingInner {
                id,
                size,
                virtually_keyed,
                #[cfg(target_os = "macos")]
                descriptor,
                #[cfg(windows)]
                handle,
                read_mappings: Mutex::new(ReadMappingCache::default()),
                write_mappings: Mutex::new(WriteMappingCache::default()),
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

    #[must_use]
    pub fn is_virtually_keyed(&self) -> bool {
        self.inner.virtually_keyed
    }

    #[cfg(windows)]
    pub(super) fn install_write_diagnostic(&self, offset: u64, byte_count: u64) {
        *diagnostic_backing_write_watch()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some(DiagnosticBackingWriteWatch {
                backing_id: self.id(),
                offset,
                byte_count,
            });
        eprintln!(
            "[REMu][JOURNAL] watching backing={} offset={offset:#x} length={byte_count:#x}",
            self.id(),
        );
    }

    #[cfg(windows)]
    pub(super) fn windows_handle(&self) -> HANDLE {
        self.inner.windows_handle()
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
        Some(GuestSharedBackingView {
            host: HostBackingMapping::new(&self.inner, backing_offset, byte_count)?,
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
            let Some(mapping) = CachedReadMapping::new(&self.inner, aligned_offset, mapping_length)
            else {
                return false;
            };
            mapping.copy_to(offset, destination);
            cache.insert(mapping);
            return true;
        }
        let Some(mapping) = CachedReadMapping::new(&self.inner, aligned_offset, mapping_length)
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
        #[cfg(windows)]
        trace_diagnostic_backing_write(self.id(), offset, source);
        let Ok(page_size) = host_page_size() else {
            return false;
        };
        let aligned_offset = offset - offset % page_size;
        let in_mapping_offset = offset - aligned_offset;
        let Some(mapping_length) = in_mapping_offset
            .checked_add(u64::try_from(source.len()).unwrap_or(u64::MAX))
            .and_then(|length| align_up(length, page_size).ok())
            .and_then(|length| host_size(length).ok())
        else {
            return false;
        };
        // Aligning the window end past the object is legal for mmap only up to the file size;
        // clamp so a final partial page still maps. `end <= size` and page-aligned object sizes
        // make the clamp a no-op except for objects smaller than one page window.
        let mapping_length = match usize::try_from(self.size() - aligned_offset) {
            Ok(available) => mapping_length.min(available),
            Err(_) => mapping_length,
        };
        if mapping_length <= READ_MAPPING_CACHE_BYTE_CAPACITY {
            let mut cache = self
                .inner
                .write_mappings
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if cache.copy_if_present(offset, source) {
                return true;
            }
            let Some(mapping) =
                CachedWriteMapping::new(&self.inner, aligned_offset, mapping_length)
            else {
                return false;
            };
            mapping.copy_from(offset, source);
            cache.insert(mapping);
            return true;
        }
        let Some(mapping) =
            CachedWriteMapping::new(&self.inner, aligned_offset, mapping_length)
        else {
            return false;
        };
        mapping.copy_from(offset, source);
        true
    }

    #[cfg(test)]
    pub(super) fn write_mapping_cache_stats(&self) -> (usize, u64, u64) {
        let cache = self
            .inner
            .write_mappings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (cache.mappings.len(), cache.hits, cache.misses)
    }
}
