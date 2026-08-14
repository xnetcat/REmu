// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::io;
use std::ptr::NonNull;

use remu_core::GuestAddress;
use remu_loader::LoadedExecutable;
use thiserror::Error;

mod async_debug_log;
pub use async_debug_log::{DEBUG_LOG_VARIABLE, debug_log_enabled, debug_log_line};

mod async_stderr;
pub use async_stderr::{
    SYNC_STDERR_VARIABLE, crash_output_fd, flush_async_stderr, install_async_stderr,
};

mod guest_write_journal;
pub use guest_write_journal::{
    GUEST_WRITE_JOURNAL_VARIABLE, GuestWriteSite, address_looks_poisoned, dump_guest_write_journal,
    guest_write_journal_enabled, record_guest_write, report_unrecovered_signal,
};

mod signal_text;

#[cfg(target_os = "macos")]
mod write_watch;
#[cfg(target_os = "macos")]
pub use write_watch::notify_guest_memory_written;
#[cfg(target_os = "macos")]
pub use write_watch::notify_guest_memory_written_excluding;
#[cfg(target_os = "macos")]
pub use write_watch::{
    CpuWriteWatch, CpuWriteWatchReprotection, CpuWriteWatchRetirement, GuestFaultDeliveryHandler,
    GuestFsFaultHandler, GuestReadMaterializer, install_fault_handlers,
    install_thread_signal_stack, set_fs_fault_handler, set_guest_fault_delivery_handler,
    set_read_watch_materializer,
};

/// Non-macOS builds have no unified-memory write-watch backend to notify.
#[cfg(not(target_os = "macos"))]
#[must_use]
pub const fn notify_guest_memory_written(_address: u64, _byte_count: u64) -> bool {
    false
}

#[cfg(target_os = "macos")]
mod address_space;
#[cfg(target_os = "macos")]
pub use address_space::{
    AddressSpaceError, GUEST_PAGE_SIZE, GuardedBackingWrite, GuestAddressSpace, GuestMapCoalescing,
    GuestMapPlacement, GuestPhysicalBacking, GuestProtection, GuestRangeLease,
    GuestSharedBackingRange, GuestSharedBackingSegment, GuestSharedBackingView,
    GuestSharedMapOptions, GuestSharedMappingIdentity, GuestVmaInfo, GuestVmaState,
    LeasePublishOutcome, RequestedAddress, guest_publish_release_fence,
    guest_shared_backing_alias_segments, guest_shared_backing_identity, guest_shared_backing_range,
    take_guest_shared_identity_statistics,
    guest_shared_backing_any_range_overlaps, guest_shared_backing_ranges_overlap,
    guest_shared_backing_segments, guest_shared_mapping_epoch,
    guest_shared_mapping_identity, guest_shared_mapping_table, load_guest_shared_u32,
    load_guest_shared_u64,
    load_guest_visible_u32, load_guest_visible_u64, publish_guest_shared_u32_if_identity,
    publish_guest_shared_u64_if_identity, read_guest_shared_backing, store_guest_visible_u32,
    store_guest_visible_u64, write_guest_shared_backing, write_guest_shared_backing_at_identity,
    write_guest_shared_backing_at_identity_with_baseline, write_guest_shared_backing_if_identity,
    write_guest_shared_backing_if_identity_and_baseline, write_guest_shared_backing_scattered,
};

/// Validated `GuestPtr` / `GuestSlice` access layer (plan M1). Depends on the shared-mapping
/// registry, so it is available on the same hosts as [`GuestRangeLease`].
#[cfg(target_os = "macos")]
mod guest_ptr;
#[cfg(target_os = "macos")]
pub use guest_ptr::{
    GuestAccessError, GuestAccessKind, GuestAccessPin, GuestPtr, GuestPtrOwned, GuestSlice,
    GuestSliceOwned,
};

/// Guest page size is 16 KiB on every host. The sparse address-space backend that uses it is
/// macOS-only today; analysis and page-alignment helpers still need the constant everywhere.
#[cfg(not(target_os = "macos"))]
pub const GUEST_PAGE_SIZE: u64 = 16 * 1024;

pub const PS5_MAIN_IMAGE_BASE: GuestAddress = GuestAddress::new(0x0000_0008_0000_0000);

pub const GUEST_EXECUTE: u32 = 0x1;
pub const GUEST_WRITE: u32 = 0x2;
pub const GUEST_READ: u32 = 0x4;
const AUXILIARY_RESERVE_SIZE: u64 = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuestRegion {
    pub start: GuestAddress,
    pub memory_size: u64,
    pub file_size: u64,
    pub flags: u32,
}

#[derive(Debug, Error)]
pub enum MappingError {
    #[error("executable has no loadable segments")]
    NoLoadSegments,
    #[error("host page size is unavailable or invalid")]
    InvalidHostPageSize,
    #[error("guest address arithmetic overflow")]
    AddressOverflow,
    #[error("guest mapping size cannot be represented by this host")]
    SizeNotRepresentable,
    #[error("fixed guest mapping at 0x{address:x} with size 0x{size:x} failed: {source}")]
    AllocationFailed {
        address: u64,
        size: u64,
        #[source]
        source: io::Error,
    },
    #[error("Mach fixed allocation at 0x{address:x} with size 0x{size:x} failed ({code})")]
    MachAllocationFailed { address: u64, size: u64, code: i32 },
    #[error("mprotect failed for 0x{address:x}..0x{end:x}: {source}")]
    ProtectionFailed {
        address: u64,
        end: u64,
        #[source]
        source: io::Error,
    },
    #[error("guest range 0x{address:x}..0x{end:x} is outside 0x{start:x}..0x{limit:x}")]
    OutOfBounds {
        address: u64,
        end: u64,
        start: u64,
        limit: u64,
    },
    #[error("guest memory permissions have already been finalized")]
    AlreadyFinalized,
    #[error("allocation alignment 0x{0:x} is not a nonzero power of two")]
    InvalidAlignment(u64),
    #[error("guest auxiliary allocation of 0x{size:x} bytes exceeds the reserved range")]
    AuxiliaryRangeExhausted { size: u64 },
    #[error("fixed guest mappings are not supported on this host")]
    UnsupportedPlatform,
}

#[derive(Debug)]
pub struct GuestImage {
    allocation: RawAllocation,
    image_base: GuestAddress,
    entry_address: GuestAddress,
    page_size: u64,
    page_flags: BTreeMap<u64, u32>,
    regions: Vec<GuestRegion>,
    auxiliary_cursor: u64,
    auxiliary_end: u64,
    finalized: bool,
}

impl GuestImage {
    pub fn map_main(executable: &LoadedExecutable) -> Result<Self, MappingError> {
        Self::map_at(executable, PS5_MAIN_IMAGE_BASE)
    }

    /// Maps a main executable while reserving an explicitly preflighted amount of contiguous
    /// address space for PRX images, import trampolines, and process TLS.
    pub fn map_main_with_auxiliary_reserve(
        executable: &LoadedExecutable,
        auxiliary_reserve_size: u64,
    ) -> Result<Self, MappingError> {
        Self::map_at_with_auxiliary_reserve(executable, PS5_MAIN_IMAGE_BASE, auxiliary_reserve_size)
    }

    pub fn map_at(
        executable: &LoadedExecutable,
        image_base: GuestAddress,
    ) -> Result<Self, MappingError> {
        Self::map_at_with_auxiliary_reserve(executable, image_base, AUXILIARY_RESERVE_SIZE)
    }

    pub fn map_at_with_auxiliary_reserve(
        executable: &LoadedExecutable,
        image_base: GuestAddress,
        auxiliary_reserve_size: u64,
    ) -> Result<Self, MappingError> {
        let page_size = host_page_size()?;
        let mut minimum = u64::MAX;
        let mut maximum = 0_u64;
        for segment in &executable.load_segments {
            let start = image_base
                .get()
                .checked_add(segment.virtual_address.get())
                .ok_or(MappingError::AddressOverflow)?;
            let end = start
                .checked_add(segment.memory_size)
                .ok_or(MappingError::AddressOverflow)?;
            minimum = minimum.min(start);
            maximum = maximum.max(end);
        }
        if minimum == u64::MAX {
            return Err(MappingError::NoLoadSegments);
        }

        let mapping_start = align_down(minimum, page_size);
        let image_end = align_up(maximum, page_size)?;
        let mapping_end = image_end
            .checked_add(auxiliary_reserve_size)
            .ok_or(MappingError::AddressOverflow)?;
        let mapping_size = mapping_end
            .checked_sub(mapping_start)
            .ok_or(MappingError::AddressOverflow)?;
        let allocation = RawAllocation::allocate_exact(mapping_start, mapping_size)?;

        let entry = executable
            .info
            .elf
            .as_ref()
            .map_or(0, |elf| elf.header.entry.get());
        let entry_address = GuestAddress::new(
            image_base
                .get()
                .checked_add(entry)
                .ok_or(MappingError::AddressOverflow)?,
        );
        let mut image = Self {
            allocation,
            image_base,
            entry_address,
            page_size,
            page_flags: BTreeMap::new(),
            regions: Vec::with_capacity(executable.load_segments.len()),
            auxiliary_cursor: image_end,
            auxiliary_end: mapping_end,
            finalized: false,
        };

        for segment in &executable.load_segments {
            let start = image_base
                .get()
                .checked_add(segment.virtual_address.get())
                .ok_or(MappingError::AddressOverflow)?;
            image.write(GuestAddress::new(start), &segment.data)?;
            image.regions.push(GuestRegion {
                start: GuestAddress::new(start),
                memory_size: segment.memory_size,
                file_size: segment.data.len() as u64,
                flags: segment.flags,
            });

            let page_start = align_down(start, page_size);
            let page_end = align_up(
                start
                    .checked_add(segment.memory_size)
                    .ok_or(MappingError::AddressOverflow)?,
                page_size,
            )?;
            let mut page = page_start;
            while page < page_end {
                *image.page_flags.entry(page).or_default() |= segment.flags;
                page = page
                    .checked_add(page_size)
                    .ok_or(MappingError::AddressOverflow)?;
            }
        }

        Ok(image)
    }

    /// Maps a secondary PRX into this image's reserved auxiliary range.
    ///
    /// The returned address is the module's zero virtual-address base, so callers can apply the
    /// module's ELF relocations with `base + virtual_address`. Secondary modules must be installed
    /// before permissions are finalized.
    pub fn map_module(
        &mut self,
        executable: &LoadedExecutable,
        alignment: u64,
    ) -> Result<GuestAddress, MappingError> {
        if self.finalized {
            return Err(MappingError::AlreadyFinalized);
        }
        let maximum = executable
            .load_segments
            .iter()
            .try_fold(0_u64, |maximum, segment| {
                segment
                    .virtual_address
                    .get()
                    .checked_add(segment.memory_size)
                    .map(|end| maximum.max(end))
                    .ok_or(MappingError::AddressOverflow)
            })?;
        if maximum == 0 {
            return Err(MappingError::NoLoadSegments);
        }

        let module_base = self.allocate_staging(maximum, alignment, 0)?;
        for segment in &executable.load_segments {
            let start = module_base
                .get()
                .checked_add(segment.virtual_address.get())
                .ok_or(MappingError::AddressOverflow)?;
            self.write(GuestAddress::new(start), &segment.data)?;
            self.regions.push(GuestRegion {
                start: GuestAddress::new(start),
                memory_size: segment.memory_size,
                file_size: segment.data.len() as u64,
                flags: segment.flags,
            });

            let page_start = align_down(start, self.page_size);
            let page_end = align_up(
                start
                    .checked_add(segment.memory_size)
                    .ok_or(MappingError::AddressOverflow)?,
                self.page_size,
            )?;
            let mut page = page_start;
            while page < page_end {
                *self.page_flags.entry(page).or_default() |= segment.flags;
                page = page
                    .checked_add(self.page_size)
                    .ok_or(MappingError::AddressOverflow)?;
            }
        }
        Ok(module_base)
    }

    #[must_use]
    pub const fn image_base(&self) -> GuestAddress {
        self.image_base
    }

    #[must_use]
    pub const fn entry_address(&self) -> GuestAddress {
        self.entry_address
    }

    #[must_use]
    pub fn mapping_start(&self) -> GuestAddress {
        GuestAddress::new(self.allocation.address_u64())
    }

    #[must_use]
    pub const fn mapping_size(&self) -> u64 {
        self.allocation.size as u64
    }

    #[must_use]
    pub fn regions(&self) -> &[GuestRegion] {
        &self.regions
    }

    #[must_use]
    pub fn contains(&self, address: GuestAddress, length: usize) -> bool {
        self.checked_offset(address, length).is_ok()
    }

    pub fn allocate_staging(
        &mut self,
        size: u64,
        alignment: u64,
        flags: u32,
    ) -> Result<GuestAddress, MappingError> {
        if self.finalized {
            return Err(MappingError::AlreadyFinalized);
        }
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(MappingError::InvalidAlignment(alignment));
        }
        let start = align_up(self.auxiliary_cursor, alignment)?;
        let end = start
            .checked_add(size)
            .ok_or(MappingError::AddressOverflow)?;
        if end > self.auxiliary_end {
            return Err(MappingError::AuxiliaryRangeExhausted { size });
        }
        let page_start = align_down(start, self.page_size);
        let page_end = align_up(end, self.page_size)?;
        let mut page = page_start;
        while page < page_end {
            *self.page_flags.entry(page).or_default() |= flags;
            page = page
                .checked_add(self.page_size)
                .ok_or(MappingError::AddressOverflow)?;
        }
        self.auxiliary_cursor = end;
        Ok(GuestAddress::new(start))
    }

    pub fn read(&self, address: GuestAddress, destination: &mut [u8]) -> Result<(), MappingError> {
        let offset = self.checked_offset(address, destination.len())?;
        // SAFETY: `checked_offset` proves the entire source range belongs to this live allocation,
        // and `destination` supplies valid writable storage of the same length.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.allocation.address.as_ptr().cast::<u8>().add(offset),
                destination.as_mut_ptr(),
                destination.len(),
            );
        }
        Ok(())
    }

    pub fn write(&mut self, address: GuestAddress, source: &[u8]) -> Result<(), MappingError> {
        if self.finalized {
            return Err(MappingError::AlreadyFinalized);
        }
        let offset = self.checked_offset(address, source.len())?;
        // SAFETY: `checked_offset` proves the entire destination range belongs to this live,
        // writable staging allocation, and `source` is valid for `source.len()` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                source.as_ptr(),
                self.allocation.address.as_ptr().cast::<u8>().add(offset),
                source.len(),
            );
        }
        Ok(())
    }

    pub fn finalize_permissions(&mut self) -> Result<(), MappingError> {
        if self.finalized {
            return Err(MappingError::AlreadyFinalized);
        }
        let mapping_start = self.allocation.address_u64();
        let mapping_end = mapping_start
            .checked_add(self.allocation.size as u64)
            .ok_or(MappingError::AddressOverflow)?;
        let mut run_start = mapping_start;
        let mut run_flags = self.page_flags.get(&mapping_start).copied().unwrap_or(0);
        let mut page = mapping_start
            .checked_add(self.page_size)
            .ok_or(MappingError::AddressOverflow)?;
        while page < mapping_end {
            let flags = self.page_flags.get(&page).copied().unwrap_or(0);
            if flags != run_flags {
                protect(run_start, page - run_start, run_flags)?;
                run_start = page;
                run_flags = flags;
            }
            page = page
                .checked_add(self.page_size)
                .ok_or(MappingError::AddressOverflow)?;
        }
        protect(run_start, mapping_end - run_start, run_flags)?;
        self.finalized = true;
        Ok(())
    }

    fn checked_offset(&self, address: GuestAddress, length: usize) -> Result<usize, MappingError> {
        let start = self.allocation.address_u64();
        let limit = start
            .checked_add(self.allocation.size as u64)
            .ok_or(MappingError::AddressOverflow)?;
        let end = address
            .get()
            .checked_add(length as u64)
            .ok_or(MappingError::AddressOverflow)?;
        if address.get() < start || end > limit {
            return Err(MappingError::OutOfBounds {
                address: address.get(),
                end,
                start,
                limit,
            });
        }
        usize::try_from(address.get() - start).map_err(|_| MappingError::SizeNotRepresentable)
    }
}

#[derive(Clone, Copy, Debug)]
enum AllocationKind {
    #[cfg(unix)]
    Mmap,
    #[cfg(target_os = "macos")]
    Mach,
    #[cfg(windows)]
    VirtualAlloc,
}

#[derive(Debug)]
struct RawAllocation {
    address: NonNull<c_void>,
    size: usize,
    kind: AllocationKind,
}

// SAFETY: RawAllocation is an owning handle to a process-wide VM mapping. Moving that sole owner
// to another thread does not create aliases; Drop releases the same mapping from either thread.
unsafe impl Send for RawAllocation {}

impl RawAllocation {
    #[cfg(target_os = "macos")]
    fn allocate_exact(address: u64, size: u64) -> Result<Self, MappingError> {
        let size_usize = usize::try_from(size).map_err(|_| MappingError::SizeNotRepresentable)?;
        let requested = usize::try_from(address).map_err(|_| MappingError::SizeNotRepresentable)?
            as *mut c_void;
        // SAFETY: This requests a new anonymous mapping. No Rust references cover the requested
        // guest range, and a non-exact hint result is immediately released.
        let mapped = unsafe {
            libc::mmap(
                requested,
                size_usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if mapped != libc::MAP_FAILED {
            if mapped == requested {
                return Ok(Self {
                    address: NonNull::new(mapped).ok_or(MappingError::AddressOverflow)?,
                    size: size_usize,
                    kind: AllocationKind::Mmap,
                });
            }
            // SAFETY: `mapped` and `size_usize` are the mapping returned immediately above.
            unsafe {
                libc::munmap(mapped, size_usize);
            }
        }

        let mut mach_address = address;
        // SAFETY: VM_FLAGS_FIXED without overwrite atomically succeeds only if the range is free.
        let code = unsafe {
            mach_vm_allocate(
                mach_task_self(),
                &raw mut mach_address,
                size,
                VM_FLAGS_FIXED,
            )
        };
        if code != KERN_SUCCESS || mach_address != address {
            if code == KERN_SUCCESS {
                // SAFETY: The successful allocation is owned by this function.
                unsafe {
                    mach_vm_deallocate(mach_task_self(), mach_address, size);
                }
            }
            return Err(MappingError::MachAllocationFailed {
                address,
                size,
                code,
            });
        }
        // SAFETY: The Mach allocation covers this exact address and size.
        let protect_result =
            unsafe { libc::mprotect(requested, size_usize, libc::PROT_READ | libc::PROT_WRITE) };
        if protect_result != 0 {
            let source = io::Error::last_os_error();
            // SAFETY: The Mach allocation is still owned by this function.
            unsafe {
                mach_vm_deallocate(mach_task_self(), address, size);
            }
            return Err(MappingError::AllocationFailed {
                address,
                size,
                source,
            });
        }

        Ok(Self {
            address: NonNull::new(requested).ok_or(MappingError::AddressOverflow)?,
            size: size_usize,
            kind: AllocationKind::Mach,
        })
    }

    #[cfg(windows)]
    fn allocate_exact(address: u64, size: u64) -> Result<Self, MappingError> {
        let size_usize = usize::try_from(size).map_err(|_| MappingError::SizeNotRepresentable)?;
        let requested = usize::try_from(address).map_err(|_| MappingError::SizeNotRepresentable)?
            as *mut c_void;
        // SAFETY: Reserves and commits an exact free range; failure leaves no mapping owned.
        let mapped = unsafe {
            VirtualAlloc(
                requested,
                size_usize,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        if mapped.is_null() {
            return Err(MappingError::AllocationFailed {
                address,
                size,
                source: io::Error::last_os_error(),
            });
        }
        if mapped != requested {
            // SAFETY: Owns the unexpected mapping returned above.
            unsafe {
                VirtualFree(mapped, 0, MEM_RELEASE);
            }
            return Err(MappingError::AllocationFailed {
                address,
                size,
                source: io::Error::from_raw_os_error(ERROR_INVALID_ADDRESS),
            });
        }
        Ok(Self {
            address: NonNull::new(mapped).ok_or(MappingError::AddressOverflow)?,
            size: size_usize,
            kind: AllocationKind::VirtualAlloc,
        })
    }

    #[cfg(not(any(target_os = "macos", windows)))]
    const fn allocate_exact(_address: u64, _size: u64) -> Result<Self, MappingError> {
        Err(MappingError::UnsupportedPlatform)
    }

    fn address_u64(&self) -> u64 {
        self.address.as_ptr() as usize as u64
    }
}

impl Drop for RawAllocation {
    fn drop(&mut self) {
        match self.kind {
            #[cfg(unix)]
            AllocationKind::Mmap => {
                // SAFETY: This allocation is live and owned exclusively by `self`.
                unsafe {
                    libc::munmap(self.address.as_ptr(), self.size);
                }
            }
            #[cfg(target_os = "macos")]
            AllocationKind::Mach => {
                // SAFETY: This Mach allocation is live and owned exclusively by `self`.
                unsafe {
                    mach_vm_deallocate(mach_task_self(), self.address_u64(), self.size as u64);
                }
            }
            #[cfg(windows)]
            AllocationKind::VirtualAlloc => {
                // SAFETY: This VirtualAlloc mapping is live and owned exclusively by `self`.
                unsafe {
                    VirtualFree(self.address.as_ptr(), 0, MEM_RELEASE);
                }
            }
        }
    }
}

#[cfg(unix)]
fn host_page_size() -> Result<u64, MappingError> {
    // SAFETY: `sysconf` has no pointer preconditions.
    let value = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let page_size = u64::try_from(value).map_err(|_| MappingError::InvalidHostPageSize)?;
    if !page_size.is_power_of_two() {
        return Err(MappingError::InvalidHostPageSize);
    }
    Ok(page_size)
}

#[cfg(windows)]
fn host_page_size() -> Result<u64, MappingError> {
    let mut info = SystemInfo::zeroed();
    // SAFETY: `info` is a valid SYSTEM_INFO buffer for GetSystemInfo.
    unsafe {
        GetSystemInfo(&raw mut info);
    }
    let page_size = u64::from(info.dw_page_size);
    if page_size == 0 || !page_size.is_power_of_two() {
        return Err(MappingError::InvalidHostPageSize);
    }
    Ok(page_size)
}

#[cfg(not(any(unix, windows)))]
fn host_page_size() -> Result<u64, MappingError> {
    Err(MappingError::UnsupportedPlatform)
}

fn align_down(value: u64, alignment: u64) -> u64 {
    value & !(alignment - 1)
}

fn align_up(value: u64, alignment: u64) -> Result<u64, MappingError> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .ok_or(MappingError::AddressOverflow)
}

#[cfg(unix)]
fn protect(address: u64, size: u64, flags: u32) -> Result<(), MappingError> {
    let mut protection = libc::PROT_NONE;
    if flags & GUEST_READ != 0 || flags & GUEST_EXECUTE != 0 {
        protection |= libc::PROT_READ;
    }
    if flags & GUEST_WRITE != 0 {
        protection |= libc::PROT_WRITE;
    }
    if flags & GUEST_EXECUTE != 0 {
        protection |= libc::PROT_EXEC;
    }
    let address_ptr =
        usize::try_from(address).map_err(|_| MappingError::SizeNotRepresentable)? as *mut c_void;
    let size_usize = usize::try_from(size).map_err(|_| MappingError::SizeNotRepresentable)?;
    // SAFETY: The caller passes page-aligned ranges belonging to the live guest allocation.
    if unsafe { libc::mprotect(address_ptr, size_usize, protection) } != 0 {
        return Err(MappingError::ProtectionFailed {
            address,
            end: address
                .checked_add(size)
                .ok_or(MappingError::AddressOverflow)?,
            source: io::Error::last_os_error(),
        });
    }
    Ok(())
}

#[cfg(windows)]
fn protect(address: u64, size: u64, flags: u32) -> Result<(), MappingError> {
    let protection = windows_page_protection(flags);
    let address_ptr =
        usize::try_from(address).map_err(|_| MappingError::SizeNotRepresentable)? as *mut c_void;
    let size_usize = usize::try_from(size).map_err(|_| MappingError::SizeNotRepresentable)?;
    let mut old_protection = 0_u32;
    // SAFETY: The caller passes page-aligned ranges belonging to the live guest allocation.
    if unsafe { VirtualProtect(address_ptr, size_usize, protection, &raw mut old_protection) } == 0
    {
        return Err(MappingError::ProtectionFailed {
            address,
            end: address
                .checked_add(size)
                .ok_or(MappingError::AddressOverflow)?,
            source: io::Error::last_os_error(),
        });
    }
    Ok(())
}

#[cfg(windows)]
const fn windows_page_protection(flags: u32) -> u32 {
    let readable = flags & GUEST_READ != 0 || flags & GUEST_EXECUTE != 0;
    let writable = flags & GUEST_WRITE != 0;
    let executable = flags & GUEST_EXECUTE != 0;
    match (readable, writable, executable) {
        (_, true, true) => PAGE_EXECUTE_READWRITE,
        (_, false, true) => PAGE_EXECUTE_READ,
        (_, true, false) => PAGE_READWRITE,
        (true, false, false) => PAGE_READONLY,
        (false, false, false) => PAGE_NOACCESS,
    }
}

#[cfg(not(any(unix, windows)))]
fn protect(_address: u64, _size: u64, _flags: u32) -> Result<(), MappingError> {
    Err(MappingError::UnsupportedPlatform)
}

#[cfg(target_os = "macos")]
const KERN_SUCCESS: i32 = 0;
#[cfg(target_os = "macos")]
const VM_FLAGS_FIXED: i32 = 0;

// SAFETY (declaration): these are the libSystem Mach VM entry points with their documented ABI;
// each call site upholds the per-call preconditions (live task, owned address range, sizes fitting
// the platform types) noted at that site.
#[cfg(target_os = "macos")]
#[link(name = "System")]
unsafe extern "C" {
    fn mach_task_self() -> u32;
    fn mach_vm_allocate(target: u32, address: *mut u64, size: u64, flags: i32) -> i32;
    fn mach_vm_deallocate(target: u32, address: u64, size: u64) -> i32;
}

#[cfg(windows)]
const MEM_COMMIT: u32 = 0x1000;
#[cfg(windows)]
const MEM_RESERVE: u32 = 0x2000;
#[cfg(windows)]
const MEM_RELEASE: u32 = 0x8000;
#[cfg(windows)]
const PAGE_NOACCESS: u32 = 0x01;
#[cfg(windows)]
const PAGE_READONLY: u32 = 0x02;
#[cfg(windows)]
const PAGE_READWRITE: u32 = 0x04;
#[cfg(windows)]
const PAGE_EXECUTE_READ: u32 = 0x20;
#[cfg(windows)]
const PAGE_EXECUTE_READWRITE: u32 = 0x40;
#[cfg(windows)]
const ERROR_INVALID_ADDRESS: i32 = 487;

#[cfg(windows)]
#[repr(C)]
struct SystemInfo {
    w_processor_architecture: u16,
    w_reserved: u16,
    dw_page_size: u32,
    lp_minimum_application_address: *mut c_void,
    lp_maximum_application_address: *mut c_void,
    dw_active_processor_mask: usize,
    dw_number_of_processors: u32,
    dw_processor_type: u32,
    dw_allocation_granularity: u32,
    w_processor_level: u16,
    w_processor_revision: u16,
}

#[cfg(windows)]
impl SystemInfo {
    const fn zeroed() -> Self {
        Self {
            w_processor_architecture: 0,
            w_reserved: 0,
            dw_page_size: 0,
            lp_minimum_application_address: std::ptr::null_mut(),
            lp_maximum_application_address: std::ptr::null_mut(),
            dw_active_processor_mask: 0,
            dw_number_of_processors: 0,
            dw_processor_type: 0,
            dw_allocation_granularity: 0,
            w_processor_level: 0,
            w_processor_revision: 0,
        }
    }
}

// SAFETY (declaration): these are the kernel32 VM entry points with their documented ABI; each
// call site upholds the per-call preconditions noted at that site.
#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn VirtualAlloc(
        lp_address: *mut c_void,
        dw_size: usize,
        fl_allocation_type: u32,
        fl_protect: u32,
    ) -> *mut c_void;
    fn VirtualFree(lp_address: *mut c_void, dw_size: usize, dw_free_type: u32) -> i32;
    fn VirtualProtect(
        lp_address: *mut c_void,
        dw_size: usize,
        fl_new_protect: u32,
        lpfl_old_protect: *mut u32,
    ) -> i32;
    fn GetSystemInfo(lp_system_info: *mut SystemInfo);
}

#[cfg(all(test, target_os = "macos", target_arch = "x86_64"))]
mod tests {
    use std::io::Cursor;

    use remu_core::GuestAddress;
    use remu_loader::load_reader;

    use super::GuestImage;

    #[test]
    fn maps_load_bytes_and_zero_filled_memory_at_a_fixed_guest_address() {
        let mut bytes = vec![0_u8; 0x180];
        bytes[..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[0x12..0x14].copy_from_slice(&62_u16.to_le_bytes());
        bytes[0x20..0x28].copy_from_slice(&64_u64.to_le_bytes());
        bytes[0x34..0x36].copy_from_slice(&64_u16.to_le_bytes());
        bytes[0x36..0x38].copy_from_slice(&56_u16.to_le_bytes());
        bytes[0x38..0x3a].copy_from_slice(&1_u16.to_le_bytes());
        bytes[0x40..0x44].copy_from_slice(&1_u32.to_le_bytes());
        bytes[0x44..0x48].copy_from_slice(&6_u32.to_le_bytes());
        bytes[0x48..0x50].copy_from_slice(&0x100_u64.to_le_bytes());
        bytes[0x50..0x58].copy_from_slice(&0x4000_u64.to_le_bytes());
        bytes[0x60..0x68].copy_from_slice(&4_u64.to_le_bytes());
        bytes[0x68..0x70].copy_from_slice(&16_u64.to_le_bytes());
        bytes[0x70..0x78].copy_from_slice(&0x4000_u64.to_le_bytes());
        bytes[0x100..0x104].copy_from_slice(&[1, 2, 3, 4]);
        let loaded = load_reader(&mut Cursor::new(bytes)).unwrap();

        let image_base = GuestAddress::new(0x0000_000a_0000_0000);
        let mut image = GuestImage::map_at(&loaded, image_base).unwrap();
        let address = GuestAddress::new(image_base.get() + 0x4000);
        let mut actual = [0xff_u8; 16];
        image.read(address, &mut actual).unwrap();
        assert_eq!(&actual[..4], &[1, 2, 3, 4]);
        assert_eq!(&actual[4..], &[0; 12]);
        image.write(address, &[9, 8, 7, 6]).unwrap();
        image.finalize_permissions().unwrap();
    }
}
