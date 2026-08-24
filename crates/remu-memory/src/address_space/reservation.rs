// SPDX-License-Identifier: GPL-2.0-or-later

//! The single inaccessible host reservation that backs one guest address space.

use std::ffi::c_void;
use std::io;
use std::ptr::NonNull;

use super::shared::{begin_shared_mapping_mutation, unregister_shared_mapping_range};
use super::{
    AddressSpaceError, GUEST_PAGE_SIZE, RequestedAddress, align_up, host_pointer, host_size,
};
use crate::write_watch::{
    CpuWriteWatch, forget_guest_committed_range, forget_guest_reservation, publish_guest_reservation,
};

#[derive(Clone, Copy, Debug)]
pub(super) enum ReservationKind {
    Mmap,
    Mach,
}

#[derive(Debug)]
pub(super) struct HostReservation {
    address: NonNull<c_void>,
    pub(super) size: usize,
    kind: ReservationKind,
}

// SAFETY: This is the unique owner of process VM pages. Moving it between threads does not create
// an additional owner or invalidate the mapping.
unsafe impl Send for HostReservation {}
// SAFETY: HostReservation contains no Rust references and immutable access only observes its
// address and size. VMA mutation still requires `&mut GuestAddressSpace`; callers that execute or
// expose raw guest pointers must synchronize mapping changes with those users at the runtime
// boundary, just as they must for mmap/munmap themselves.
unsafe impl Sync for HostReservation {}

impl HostReservation {
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

    pub(super) fn address(&self) -> u64 {
        self.address.as_ptr() as usize as u64
    }

    /// Lets the fault handler tell guest-memory faults apart from ordinary host-memory faults.
    ///
    /// The handler observes every process SIGSEGV/SIGBUS and cannot take a lock, so the reservation
    /// bounds are published to a small signal-safe table as soon as they are known.
    pub(super) fn publish_bounds(&self) {
        publish_guest_reservation(self.address(), self.size as u64);
    }
}

impl Drop for HostReservation {
    fn drop(&mut self) {
        forget_guest_reservation(self.address(), self.size as u64);
        forget_guest_committed_range(self.address(), self.size as u64);
        let _mapping_mutation = begin_shared_mapping_mutation();
        CpuWriteWatch::retire_overlapping(self.address(), self.size as u64);
        unregister_shared_mapping_range(self.address(), self.size as u64);
        match self.kind {
            ReservationKind::Mmap => {
                // SAFETY: This object owns the complete reservation, including replacement maps.
                unsafe {
                    libc::munmap(self.address.as_ptr(), self.size);
                }
            }
            ReservationKind::Mach => {
                // SAFETY: This object owns this exact Mach address range.
                unsafe {
                    mach_vm_deallocate(mach_task_self(), self.address(), self.size as u64);
                }
            }
        }
    }
}

const KERN_SUCCESS: i32 = 0;
const VM_FLAGS_FIXED: i32 = 0;

// SAFETY (declaration): the libSystem Mach VM entry points with their documented ABI; each call
// site upholds the per-call preconditions (live task, owned range, representable sizes).
#[link(name = "System")]
unsafe extern "C" {
    fn mach_task_self() -> u32;
    fn mach_vm_allocate(target: u32, address: *mut u64, size: u64, flags: i32) -> i32;
    fn mach_vm_deallocate(target: u32, address: u64, size: u64) -> i32;
}
