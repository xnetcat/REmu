// SPDX-License-Identifier: GPL-2.0-or-later

//! The single inaccessible host reservation that backs one guest address space.

use std::ffi::c_void;
use std::io;
use std::ptr::NonNull;

#[cfg(windows)]
use std::collections::BTreeMap;
#[cfg(windows)]
use std::sync::Mutex;
#[cfg(windows)]
use windows::Win32::System::Memory::{
    MEM_COMMIT, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3, PAGE_NOACCESS,
    UnmapViewOfFileEx, VIRTUAL_ALLOCATION_TYPE, VIRTUAL_FREE_TYPE, VirtualAlloc, VirtualAlloc2,
    VirtualFree,
};

#[cfg(target_os = "macos")]
use super::align_up;
use super::shared::{begin_shared_mapping_mutation, unregister_shared_mapping_range};
use super::{AddressSpaceError, GUEST_PAGE_SIZE, RequestedAddress, host_pointer, host_size};
#[cfg(windows)]
use super::{GuestPhysicalBacking, GuestProtection, checked_end};
use crate::write_watch::{CpuWriteWatch, forget_guest_reservation, publish_guest_reservation};

#[derive(Clone, Copy, Debug)]
pub(super) enum ReservationKind {
    #[cfg(target_os = "macos")]
    Mmap,
    #[cfg(target_os = "macos")]
    Mach,
    #[cfg(windows)]
    Placeholder,
}

#[cfg(windows)]
#[derive(Clone, Debug)]
struct WindowsMappedView {
    start: u64,
    end: u64,
    backing: GuestPhysicalBacking,
    backing_offset: u64,
    protection: GuestProtection,
}

#[cfg(windows)]
#[derive(Debug)]
struct WindowsReservationState {
    placeholders: BTreeMap<u64, u64>,
    views: BTreeMap<u64, WindowsMappedView>,
}

#[derive(Debug)]
pub(super) struct HostReservation {
    address: NonNull<c_void>,
    pub(super) size: usize,
    kind: ReservationKind,
    #[cfg(windows)]
    state: Mutex<WindowsReservationState>,
}

// SAFETY: This is the unique owner of process VM pages. Moving it between threads does not create
// an additional owner or invalidate the mapping.
unsafe impl Send for HostReservation {}
// SAFETY: HostReservation contains no Rust references and Windows host-view state is serialized.
unsafe impl Sync for HostReservation {}

impl HostReservation {
    #[cfg(target_os = "macos")]
    pub(super) fn anywhere(size: u64) -> Result<Self, AddressSpaceError> {
        let padded = size
            .checked_add(GUEST_PAGE_SIZE)
            .ok_or(AddressSpaceError::AddressOverflow { address: 0, size })?;
        let padded_usize = host_size(padded)?;
        // SAFETY: This creates a new inaccessible anonymous mapping and does not dereference it.
        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                padded_usize,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            return Err(AddressSpaceError::ReservationFailed {
                requested: RequestedAddress::Anywhere,
                size,
                source: io::Error::last_os_error(),
            });
        }

        let raw_address = raw as usize as u64;
        let aligned = align_up(raw_address, GUEST_PAGE_SIZE)?;
        let prefix = aligned - raw_address;
        let suffix = padded - prefix - size;
        if prefix != 0 {
            // SAFETY: This is the leading subrange of the fresh mapping.
            unsafe {
                libc::munmap(raw, host_size(prefix)?);
            }
        }
        if suffix != 0 {
            let suffix_address =
                aligned
                    .checked_add(size)
                    .ok_or(AddressSpaceError::AddressOverflow {
                        address: aligned,
                        size,
                    })?;
            // SAFETY: This is the trailing subrange of the fresh mapping.
            unsafe {
                libc::munmap(host_pointer(suffix_address)?, host_size(suffix)?);
            }
        }
        Ok(Self {
            address: NonNull::new(host_pointer(aligned)?)
                .ok_or(AddressSpaceError::HostSizeOverflow { value: aligned })?,
            size: host_size(size)?,
            kind: ReservationKind::Mmap,
        })
    }

    #[cfg(windows)]
    pub(super) fn anywhere(size: u64) -> Result<Self, AddressSpaceError> {
        let size_usize = host_size(size)?;
        // SAFETY: The call reserves an inaccessible placeholder and does not dereference it.
        let raw = unsafe {
            VirtualAlloc2(
                None,
                None,
                size_usize,
                VIRTUAL_ALLOCATION_TYPE(MEM_RESERVE.0 | MEM_RESERVE_PLACEHOLDER.0),
                PAGE_NOACCESS.0,
                None,
            )
        };
        let address = NonNull::new(raw).ok_or_else(|| AddressSpaceError::ReservationFailed {
            requested: RequestedAddress::Anywhere,
            size,
            source: io::Error::last_os_error(),
        })?;
        let base = address.as_ptr() as usize as u64;
        debug_assert!(base.is_multiple_of(GUEST_PAGE_SIZE));
        Ok(Self {
            address,
            size: size_usize,
            kind: ReservationKind::Placeholder,
            state: Mutex::new(WindowsReservationState {
                placeholders: BTreeMap::from([(base, base + size)]),
                views: BTreeMap::new(),
            }),
        })
    }

    #[cfg(target_os = "macos")]
    pub(super) fn exact(address: u64, size: u64) -> Result<Self, AddressSpaceError> {
        let size_usize = host_size(size)?;
        let requested = host_pointer(address)?;
        // A non-fixed mmap is a safe exact-address probe: a different result is released.
        // SAFETY: No Rust references cover the requested guest range.
        let raw = unsafe {
            libc::mmap(
                requested,
                size_usize,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if raw != libc::MAP_FAILED {
            if raw == requested {
                return Ok(Self {
                    address: NonNull::new(raw)
                        .ok_or(AddressSpaceError::HostSizeOverflow { value: address })?,
                    size: size_usize,
                    kind: ReservationKind::Mmap,
                });
            }
            // SAFETY: `raw` is exactly the mapping returned above.
            unsafe {
                libc::munmap(raw, size_usize);
            }
        }

        let mut actual = address;
        // SAFETY: VM_FLAGS_FIXED does not overwrite an existing mapping and is an atomic exact
        // allocation attempt on macOS.
        let code =
            unsafe { mach_vm_allocate(mach_task_self(), &raw mut actual, size, VM_FLAGS_FIXED) };
        if code != KERN_SUCCESS || actual != address {
            if code == KERN_SUCCESS {
                // SAFETY: This function owns the unexpected successful allocation.
                unsafe {
                    mach_vm_deallocate(mach_task_self(), actual, size);
                }
            }
            return Err(AddressSpaceError::MachReservationFailed {
                address,
                size,
                code,
            });
        }
        // Mach allocation initially grants access; convert the whole reservation to a guard range.
        // SAFETY: `requested`/`size_usize` name the exact Mach reservation this function just
        // allocated (the `mach_vm_allocate` above returned KERN_SUCCESS with `actual == address`),
        // so the range is live and owned here; the mprotect only narrows its own pages' access.
        if unsafe { libc::mprotect(requested, size_usize, libc::PROT_NONE) } != 0 {
            let source = io::Error::last_os_error();
            // SAFETY: This function still owns the Mach allocation.
            unsafe {
                mach_vm_deallocate(mach_task_self(), address, size);
            }
            return Err(AddressSpaceError::ReservationFailed {
                requested: RequestedAddress::Exact(address),
                size,
                source,
            });
        }
        Ok(Self {
            address: NonNull::new(requested)
                .ok_or(AddressSpaceError::HostSizeOverflow { value: address })?,
            size: size_usize,
            kind: ReservationKind::Mach,
        })
    }

    #[cfg(windows)]
    pub(super) fn exact(address: u64, size: u64) -> Result<Self, AddressSpaceError> {
        let size_usize = host_size(size)?;
        let requested = host_pointer(address)?.cast_const();
        // SAFETY: This is an exact placeholder reservation attempt; failure leaves existing VM
        // untouched, and success returns the only owner of the range.
        let raw = unsafe {
            VirtualAlloc2(
                None,
                Some(requested),
                size_usize,
                VIRTUAL_ALLOCATION_TYPE(MEM_RESERVE.0 | MEM_RESERVE_PLACEHOLDER.0),
                PAGE_NOACCESS.0,
                None,
            )
        };
        if raw != requested.cast_mut() {
            if !raw.is_null() {
                // SAFETY: This function owns an unexpected successful allocation.
                let _ = unsafe { VirtualFree(raw, 0, MEM_RELEASE) };
            }
            return Err(AddressSpaceError::ReservationFailed {
                requested: RequestedAddress::Exact(address),
                size,
                source: io::Error::last_os_error(),
            });
        }
        Ok(Self {
            address: NonNull::new(raw)
                .ok_or(AddressSpaceError::HostSizeOverflow { value: address })?,
            size: size_usize,
            kind: ReservationKind::Placeholder,
            state: Mutex::new(WindowsReservationState {
                placeholders: BTreeMap::from([(address, address + size)]),
                views: BTreeMap::new(),
            }),
        })
    }

    pub(super) fn address(&self) -> u64 {
        self.address.as_ptr() as usize as u64
    }

    pub(super) fn publish_bounds(&self) {
        publish_guest_reservation(self.address(), self.size as u64);
    }

    #[cfg(windows)]
    pub(super) fn map_shared(
        &self,
        start: u64,
        size: u64,
        protection: GuestProtection,
        backing: &GuestPhysicalBacking,
        backing_offset: u64,
    ) -> Result<(), AddressSpaceError> {
        let end = checked_end(start, size)?;
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.map_view(WindowsMappedView {
            start,
            end,
            backing: backing.clone(),
            backing_offset,
            protection,
        })
    }

    #[cfg(windows)]
    pub(super) fn restore_placeholder(
        &self,
        start: u64,
        size: u64,
    ) -> Result<(), AddressSpaceError> {
        let end = checked_end(start, size)?;
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.replace_views_with_placeholder(start, end)
    }

    #[cfg(windows)]
    pub(super) fn protect_mapped(
        &self,
        start: u64,
        size: u64,
        protection: GuestProtection,
    ) -> Result<(), AddressSpaceError> {
        let end = checked_end(start, size)?;
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.reprotect_views(start, end, protection)
    }
}

#[cfg(windows)]
impl WindowsReservationState {
    fn split_placeholder_at(&mut self, address: u64) -> Result<(), AddressSpaceError> {
        let Some((&start, &end)) = self.placeholders.range(..=address).next_back() else {
            return Ok(());
        };
        if address <= start || address >= end {
            return Ok(());
        }
        // SAFETY: State proves `start..end` is one live placeholder. Preserving its prefix splits
        // it into two placeholders without exposing the address range.
        unsafe {
            VirtualFree(
                host_pointer(start)?,
                host_size(address - start)?,
                VIRTUAL_FREE_TYPE(MEM_RELEASE.0 | MEM_PRESERVE_PLACEHOLDER.0),
            )
        }
        .map_err(|_| host_operation_error("placeholder split", start, end))?;
        self.placeholders.remove(&start);
        self.placeholders.insert(start, address);
        self.placeholders.insert(address, end);
        Ok(())
    }

    fn isolate_placeholder(&mut self, start: u64, end: u64) -> Result<(), AddressSpaceError> {
        self.split_placeholder_at(start)?;
        self.split_placeholder_at(end)?;
        let pieces: Vec<(u64, u64)> = self
            .placeholders
            .range(start..end)
            .map(|(&piece_start, &piece_end)| (piece_start, piece_end))
            .collect();
        let mut cursor = start;
        for &(piece_start, piece_end) in &pieces {
            if piece_start != cursor || piece_end > end {
                return Err(host_operation_error("placeholder isolation", start, end));
            }
            cursor = piece_end;
        }
        if cursor != end {
            return Err(host_operation_error("placeholder isolation", start, end));
        }
        if pieces.len() > 1 {
            // SAFETY: The collected placeholders are adjacent and exactly cover this range.
            unsafe {
                VirtualFree(
                    host_pointer(start)?,
                    host_size(end - start)?,
                    VIRTUAL_FREE_TYPE(MEM_RELEASE.0 | 1),
                )
            }
            .map_err(|_| host_operation_error("placeholder coalescing", start, end))?;
            for (piece_start, _) in pieces {
                self.placeholders.remove(&piece_start);
            }
            self.placeholders.insert(start, end);
        }
        Ok(())
    }

    fn map_view(&mut self, view: WindowsMappedView) -> Result<(), AddressSpaceError> {
        self.isolate_placeholder(view.start, view.end)?;
        let size = host_size(view.end - view.start)?;
        let pointer = host_pointer(view.start)?.cast_const();
        // SAFETY: The target is an isolated placeholder exactly matching the requested view and
        // the caller validated the section interval.
        let mapped = unsafe {
            MapViewOfFile3(
                view.backing.windows_handle(),
                None,
                Some(pointer),
                view.backing_offset,
                size,
                MEM_REPLACE_PLACEHOLDER,
                view.protection.host_flags().0,
                None,
            )
        };
        if mapped.Value != pointer.cast_mut() {
            return Err(host_operation_error(
                "shared placeholder mapping",
                view.start,
                view.end,
            ));
        }
        // Backings use SEC_RESERVE so a 16 GiB direct section and 1 TiB virtually-keyed section
        // consume commit only for mapped guest pages.
        let committed = unsafe {
            VirtualAlloc(
                Some(pointer),
                size,
                MEM_COMMIT,
                view.protection.host_flags(),
            )
        };
        if committed != pointer.cast_mut() {
            let _ = unsafe { UnmapViewOfFileEx(mapped, MEM_PRESERVE_PLACEHOLDER) };
            return Err(host_operation_error(
                "shared page commit",
                view.start,
                view.end,
            ));
        }
        self.placeholders.remove(&view.start);
        self.views.insert(view.start, view);
        Ok(())
    }

    fn unmap_view(&mut self, start: u64) -> Result<WindowsMappedView, AddressSpaceError> {
        let view = self
            .views
            .remove(&start)
            .expect("mapped-view key was collected immediately before unmapping");
        let address = MEMORY_MAPPED_VIEW_ADDRESS {
            Value: host_pointer(view.start)?,
        };
        // SAFETY: The state entry owns this exact complete mapped view. The preserve flag returns
        // its address range to an inaccessible placeholder.
        unsafe { UnmapViewOfFileEx(address, MEM_PRESERVE_PLACEHOLDER) }
            .map_err(|_| host_operation_error("shared view unmap", view.start, view.end))?;
        self.placeholders.insert(view.start, view.end);
        Ok(view)
    }

    fn overlapping_views(&self, start: u64, end: u64) -> Vec<u64> {
        self.views
            .values()
            .filter(|view| view.start < end && view.end > start)
            .map(|view| view.start)
            .collect()
    }

    fn replace_views_with_placeholder(
        &mut self,
        start: u64,
        end: u64,
    ) -> Result<(), AddressSpaceError> {
        let keys = self.overlapping_views(start, end);
        let views = keys
            .into_iter()
            .map(|key| self.unmap_view(key))
            .collect::<Result<Vec<_>, _>>()?;
        for view in views {
            if view.start < start {
                self.map_view(WindowsMappedView {
                    start: view.start,
                    end: start.min(view.end),
                    backing: view.backing.clone(),
                    backing_offset: view.backing_offset,
                    protection: view.protection,
                })?;
            }
            if view.end > end {
                let right_start = end.max(view.start);
                self.map_view(WindowsMappedView {
                    start: right_start,
                    end: view.end,
                    backing: view.backing,
                    backing_offset: view.backing_offset + (right_start - view.start),
                    protection: view.protection,
                })?;
            }
        }
        Ok(())
    }

    fn reprotect_views(
        &mut self,
        start: u64,
        end: u64,
        protection: GuestProtection,
    ) -> Result<(), AddressSpaceError> {
        let keys = self.overlapping_views(start, end);
        let views = keys
            .into_iter()
            .map(|key| self.unmap_view(key))
            .collect::<Result<Vec<_>, _>>()?;
        for view in views {
            let middle_start = view.start.max(start);
            let middle_end = view.end.min(end);
            if view.start < middle_start {
                self.map_view(WindowsMappedView {
                    start: view.start,
                    end: middle_start,
                    backing: view.backing.clone(),
                    backing_offset: view.backing_offset,
                    protection: view.protection,
                })?;
            }
            self.map_view(WindowsMappedView {
                start: middle_start,
                end: middle_end,
                backing: view.backing.clone(),
                backing_offset: view.backing_offset + (middle_start - view.start),
                protection,
            })?;
            if middle_end < view.end {
                self.map_view(WindowsMappedView {
                    start: middle_end,
                    end: view.end,
                    backing: view.backing,
                    backing_offset: view.backing_offset + (middle_end - view.start),
                    protection: view.protection,
                })?;
            }
        }
        Ok(())
    }
}

#[cfg(windows)]
fn host_operation_error(operation: &'static str, start: u64, end: u64) -> AddressSpaceError {
    AddressSpaceError::HostOperationFailed {
        operation,
        address: start,
        end,
        source: io::Error::last_os_error(),
    }
}

impl Drop for HostReservation {
    fn drop(&mut self) {
        forget_guest_reservation(self.address(), self.size as u64);
        let _mapping_mutation = begin_shared_mapping_mutation();
        CpuWriteWatch::retire_overlapping(self.address(), self.size as u64);
        unregister_shared_mapping_range(self.address(), self.size as u64);
        match self.kind {
            #[cfg(target_os = "macos")]
            ReservationKind::Mmap => {
                // SAFETY: This object owns the complete reservation, including replacement maps.
                unsafe {
                    libc::munmap(self.address.as_ptr(), self.size);
                }
            }
            #[cfg(target_os = "macos")]
            ReservationKind::Mach => {
                // SAFETY: This object owns this exact Mach address range.
                unsafe {
                    mach_vm_deallocate(mach_task_self(), self.address(), self.size as u64);
                }
            }
            #[cfg(windows)]
            ReservationKind::Placeholder => {
                let mut state = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let keys: Vec<u64> = state.views.keys().copied().collect();
                for key in keys {
                    let _ = state.unmap_view(key);
                }
                let base = self.address();
                let end = base + self.size as u64;
                let _ = state.isolate_placeholder(base, end);
                // SAFETY: All mapped views were returned to placeholders and coalesced above.
                let _ = unsafe { VirtualFree(self.address.as_ptr(), 0, MEM_RELEASE) };
            }
        }
    }
}

#[cfg(target_os = "macos")]
const KERN_SUCCESS: i32 = 0;
#[cfg(target_os = "macos")]
const VM_FLAGS_FIXED: i32 = 0;

// SAFETY (declaration): the libSystem Mach VM entry points with their documented ABI; each call
// site upholds the per-call preconditions (live task, owned range, representable sizes).
#[link(name = "System")]
unsafe extern "C" {
    fn mach_task_self() -> u32;
    fn mach_vm_allocate(target: u32, address: *mut u64, size: u64, flags: i32) -> i32;
    fn mach_vm_deallocate(target: u32, address: u64, size: u64) -> i32;
}
