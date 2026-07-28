// SPDX-License-Identifier: GPL-2.0-or-later

//! Failures raised by guest address-space reservation, mapping and access operations.

use std::io;

use thiserror::Error;

use super::{GuestProtection, GuestVmaState, RequestedAddress};

#[derive(Debug, Error)]
pub enum AddressSpaceError {
    #[error("guest address space size must not be zero")]
    ZeroSize,
    #[error("{field} 0x{value:x} is not aligned to 0x{alignment:x}")]
    Unaligned {
        field: &'static str,
        value: u64,
        alignment: u64,
    },
    #[error("alignment 0x{alignment:x} must be a power of two and at least one 16 KiB page")]
    InvalidAlignment { alignment: u64 },
    #[error("address range beginning at 0x{address:x} with size 0x{size:x} overflows")]
    AddressOverflow { address: u64, size: u64 },
    #[error("value 0x{value:x} cannot be represented by the host")]
    HostSizeOverflow { value: u64 },
    #[error("host page size is unavailable or invalid")]
    InvalidHostPageSize,
    #[error("host page size 0x{host_page_size:x} cannot represent 16 KiB guest pages")]
    IncompatibleHostPageSize { host_page_size: u64 },
    #[error(
        "range 0x{address:x}..0x{end:x} is outside address space 0x{space_start:x}..0x{space_end:x}"
    )]
    OutOfBounds {
        address: u64,
        end: u64,
        space_start: u64,
        space_end: u64,
    },
    #[error("no aligned gap can satisfy a mapping of size 0x{size:x}")]
    AddressSpaceExhausted { size: u64, alignment: u64 },
    #[error(
        "range 0x{address:x}..0x{end:x} overlaps {state:?} VMA 0x{existing_start:x}..0x{existing_end:x}"
    )]
    Overlap {
        address: u64,
        end: u64,
        existing_start: u64,
        existing_end: u64,
        state: GuestVmaState,
    },
    #[error("address 0x{address:x} is not mapped")]
    NotMapped { address: u64 },
    #[error("address 0x{address:x} is reserved but not mapped")]
    Reserved { address: u64 },
    #[error("range 0x{address:x}..0x{end:x} requires {required} access, but VMA has {actual}")]
    PermissionDenied {
        address: u64,
        end: u64,
        required: GuestProtection,
        actual: GuestProtection,
    },
    #[error("physical backing size must not be zero")]
    ZeroBackingSize,
    #[error(
        "physical backing range 0x{offset:x}..0x{end:x} exceeds backing {backing_id} size 0x{backing_size:x}"
    )]
    BackingOutOfBounds {
        backing_id: u64,
        offset: u64,
        end: u64,
        backing_size: u64,
    },
    #[error("physical backing {operation} failed: {source}")]
    BackingOperationFailed {
        operation: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("host address-space reservation at {requested} with size 0x{size:x} failed: {source}")]
    ReservationFailed {
        requested: RequestedAddress,
        size: u64,
        #[source]
        source: io::Error,
    },
    #[error("Mach fixed reservation at 0x{address:x} with size 0x{size:x} failed ({code})")]
    MachReservationFailed { address: u64, size: u64, code: i32 },
    #[error("host {operation} failed for 0x{address:x}..0x{end:x}: {source}")]
    HostOperationFailed {
        operation: &'static str,
        address: u64,
        end: u64,
        #[source]
        source: io::Error,
    },
}
