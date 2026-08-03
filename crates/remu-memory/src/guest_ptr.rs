// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Validated guest-pointer access (plan M1).
//!
//! HLE handlers historically reached into guest memory with bare `slice::from_raw_parts` after a
//! null check only. That turns a bad guest pointer into a host SIGSEGV instead of a guest-visible
//! `MEMORY_FAULT`. This module is the remu-memory boundary that retires those sites:
//!
//! 1. Acquire a [`GuestAccessPin`] over a guest byte range. Acquisition proves the range is covered
//!    by a live shared mapping and keeps the backing object alive for the pin's lifetime (the same
//!    lease/watch pin contract plan M1 requires).
//! 2. Build a [`GuestPtr`] or [`GuestSlice`] from that pin. Typed helpers re-check length and
//!    alignment against the pinned range.
//! 3. Prefer the copy APIs ([`GuestAccessPin::read_bytes`], [`GuestPtr::read`], …). The
//!    identity-mapped views ([`GuestAccessPin::as_bytes`], [`GuestSlice::as_bytes`]) exist for
//!    zero-copy call sites that already hold the process dispatch lock across the borrow; their
//!    lifetime is tied to the pin so the raw `from_raw_parts` cannot outlive the proof.
//!
//! Watch-backed call sites should acquire the pin inside [`crate::CpuWriteWatch::with_live_mapping`]
//! so remapping cannot retire the VA between the proof and the access.

use std::marker::PhantomData;
use std::mem::{align_of, size_of};

use thiserror::Error;

use crate::GuestRangeLease;

/// Why a guest-pointer acquire or access was refused.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum GuestAccessError {
    #[error("guest address is null")]
    Null,
    #[error("guest range length overflows address space")]
    Overflow,
    #[error("guest range 0x{address:x}+{byte_count:#x} is not covered by a live shared mapping")]
    Unmapped { address: u64, byte_count: u64 },
    #[error("guest access pin is read-only")]
    ReadOnly,
    #[error("guest address 0x{address:x} is not aligned for {type_name}")]
    Misaligned {
        address: u64,
        type_name: &'static str,
    },
    #[error(
        "guest typed range 0x{address:x}+{element_count}×{type_name} does not fit the pinned bytes"
    )]
    TooShort {
        address: u64,
        element_count: usize,
        type_name: &'static str,
    },
    #[error("guest destination length {destination} does not match pinned length {pinned}")]
    LengthMismatch { destination: usize, pinned: usize },
}

/// Required access when pinning a guest range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuestAccessKind {
    /// Reads (and immutable identity views) only.
    Read,
    /// Reads and writes.
    ReadWrite,
}

/// A lease-backed pin over a guest byte range.
///
/// Holding this value keeps the resolved shared backing alive. Every identity-mapped view borrows
/// `self`, so a raw guest slice cannot outlive the mapping proof. A zero-length pin stores no
/// lease — empty buffers never touch guest memory.
pub struct GuestAccessPin {
    address: u64,
    byte_count: u64,
    access: GuestAccessKind,
    lease: Option<GuestRangeLease>,
}

impl std::fmt::Debug for GuestAccessPin {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GuestAccessPin")
            .field("address", &format_args!("{:#x}", self.address))
            .field("byte_count", &self.byte_count)
            .field("access", &self.access)
            .field("leased", &self.lease.is_some())
            .finish()
    }
}

impl GuestAccessPin {
    /// Pins `address..address+byte_count` after proving a live shared mapping covers it.
    ///
    /// A zero-length range is accepted (null still refuses) so callers can model empty buffers
    /// without special-casing the acquire path.
    pub fn try_acquire(
        address: u64,
        byte_count: u64,
        access: GuestAccessKind,
    ) -> Result<Self, GuestAccessError> {
        if address == 0 {
            return Err(GuestAccessError::Null);
        }
        if byte_count == 0 {
            return Ok(Self {
                address,
                byte_count: 0,
                access,
                lease: None,
            });
        }
        if address.checked_add(byte_count).is_none() {
            return Err(GuestAccessError::Overflow);
        }
        let Some(lease) = GuestRangeLease::acquire(address, byte_count) else {
            return Err(GuestAccessError::Unmapped {
                address,
                byte_count,
            });
        };
        Ok(Self {
            address,
            byte_count,
            access,
            lease: Some(lease),
        })
    }

    /// Adopts an already-acquired lease as the pin. The caller names the access kind.
    #[must_use]
    pub fn from_lease(lease: GuestRangeLease, byte_count: u64, access: GuestAccessKind) -> Self {
        Self {
            address: lease.address(),
            byte_count,
            access,
            lease: Some(lease),
        }
    }

    #[must_use]
    pub fn address(&self) -> u64 {
        self.address
    }

    #[must_use]
    pub fn byte_count(&self) -> u64 {
        self.byte_count
    }

    #[must_use]
    pub fn access(&self) -> GuestAccessKind {
        self.access
    }

    #[must_use]
    pub fn lease(&self) -> Option<&GuestRangeLease> {
        self.lease.as_ref()
    }

    /// Reports whether the guest VA still resolves to the leased identity.
    #[must_use]
    pub fn live_va_matches(&self) -> bool {
        match self.lease.as_ref() {
            None => self.byte_count == 0,
            Some(lease) => lease.live_va_matches(),
        }
    }

    /// Copies the pinned range into `destination` through the shared backing.
    pub fn read_bytes(&self, destination: &mut [u8]) -> Result<(), GuestAccessError> {
        let Ok(expected) = usize::try_from(self.byte_count) else {
            return Err(GuestAccessError::Overflow);
        };
        if destination.len() != expected {
            return Err(GuestAccessError::LengthMismatch {
                destination: destination.len(),
                pinned: expected,
            });
        }
        if destination.is_empty() {
            return Ok(());
        }
        if !crate::read_guest_shared_backing(self.address(), destination) {
            return Err(GuestAccessError::Unmapped {
                address: self.address(),
                byte_count: self.byte_count,
            });
        }
        Ok(())
    }

    /// Copies `source` into the pinned range through the shared backing.
    pub fn write_bytes(&self, source: &[u8]) -> Result<(), GuestAccessError> {
        if self.access != GuestAccessKind::ReadWrite {
            return Err(GuestAccessError::ReadOnly);
        }
        let Ok(expected) = usize::try_from(self.byte_count) else {
            return Err(GuestAccessError::Overflow);
        };
        if source.len() != expected {
            return Err(GuestAccessError::LengthMismatch {
                destination: source.len(),
                pinned: expected,
            });
        }
        if source.is_empty() {
            return Ok(());
        }
        if !crate::write_guest_shared_backing(self.address(), source) {
            return Err(GuestAccessError::Unmapped {
                address: self.address(),
                byte_count: self.byte_count,
            });
        }
        Ok(())
    }

    /// Identity-mapped immutable view of the pinned bytes.
    ///
    /// The returned slice borrows `self`, so it cannot outlive the lease pin. Callers that do not
    /// hold the process dispatch lock across the borrow should prefer [`Self::read_bytes`].
    pub fn as_bytes(&self) -> Result<&[u8], GuestAccessError> {
        let len = self.byte_len()?;
        if len != 0 && !self.live_va_matches() {
            return Err(GuestAccessError::Unmapped {
                address: self.address(),
                byte_count: self.byte_count,
            });
        }
        // SAFETY: a zero-length slice may be formed from any address. For `len > 0`, acquisition
        // proved a live shared mapping covers `address..+byte_count`, the lease keeps the backing
        // object alive, and `live_va_matches` re-proves the VA still resolves to that identity.
        // The slice lifetime is bound to `self`.
        Ok(unsafe { std::slice::from_raw_parts(self.address() as *const u8, len) })
    }

    /// Identity-mapped mutable view of the pinned bytes.
    #[allow(clippy::mut_from_ref)]
    pub fn as_bytes_mut(&self) -> Result<&mut [u8], GuestAccessError> {
        if self.access != GuestAccessKind::ReadWrite {
            return Err(GuestAccessError::ReadOnly);
        }
        let len = self.byte_len()?;
        if len != 0 && !self.live_va_matches() {
            return Err(GuestAccessError::Unmapped {
                address: self.address(),
                byte_count: self.byte_count,
            });
        }
        // SAFETY: same proof as [`Self::as_bytes`], plus the pin was acquired for ReadWrite.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.address() as *mut u8, len) })
    }

    fn byte_len(&self) -> Result<usize, GuestAccessError> {
        usize::try_from(self.byte_count).map_err(|_| GuestAccessError::Overflow)
    }
}

/// A validated pointer to one guest `T`, lifetime-bound to a [`GuestAccessPin`].
#[derive(Debug)]
pub struct GuestPtr<'pin, T> {
    pin: &'pin GuestAccessPin,
    _marker: PhantomData<*const T>,
}

impl<'pin, T> GuestPtr<'pin, T> {
    /// Views the pin as a single `T`. The pin must cover at least `size_of::<T>()` bytes and the
    /// guest address must be `T`-aligned.
    pub fn try_from_pin(pin: &'pin GuestAccessPin) -> Result<Self, GuestAccessError> {
        require_fit::<T>(pin, 1)?;
        Ok(Self {
            pin,
            _marker: PhantomData,
        })
    }

    /// Acquires a read pin over one `T` and returns an owning wrapper.
    pub fn try_acquire(address: u64) -> Result<GuestPtrOwned<T>, GuestAccessError> {
        GuestPtrOwned::try_acquire(address, GuestAccessKind::Read)
    }

    /// Acquires a read/write pin over one `T` and returns an owning wrapper.
    pub fn try_acquire_mut(address: u64) -> Result<GuestPtrOwned<T>, GuestAccessError> {
        GuestPtrOwned::try_acquire(address, GuestAccessKind::ReadWrite)
    }

    #[must_use]
    pub fn address(&self) -> u64 {
        self.pin.address()
    }

    #[must_use]
    pub fn pin(&self) -> &'pin GuestAccessPin {
        self.pin
    }
}

impl<T: Copy> GuestPtr<'_, T> {
    /// Reads one `T` through the shared backing.
    pub fn read(&self) -> Result<T, GuestAccessError> {
        let mut bytes = vec![0_u8; size_of::<T>()];
        self.read_prefix_bytes(&mut bytes)?;
        // SAFETY: `T: Copy` and `bytes` holds exactly `size_of::<T>()` guest bytes.
        Ok(unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<T>()) })
    }

    /// Writes one `T` through the shared backing. Refuses a read-only pin.
    pub fn write(&self, value: T) -> Result<(), GuestAccessError> {
        if self.pin.access() != GuestAccessKind::ReadWrite {
            return Err(GuestAccessError::ReadOnly);
        }
        let mut bytes = vec![0_u8; size_of::<T>()];
        // SAFETY: writing a Copy value into a local byte buffer of exact size.
        unsafe {
            std::ptr::write_unaligned(bytes.as_mut_ptr().cast::<T>(), value);
        }
        let typed = u64::try_from(size_of::<T>()).map_err(|_| GuestAccessError::Overflow)?;
        if self.pin.byte_count() == typed {
            return self.pin.write_bytes(&bytes);
        }
        // Pin is larger than T: read-modify-write the typed prefix through the backing.
        let mut whole = vec![0_u8; self.pin.byte_len()?];
        self.pin.read_bytes(&mut whole)?;
        whole[..size_of::<T>()].copy_from_slice(&bytes);
        self.pin.write_bytes(&whole)
    }

    fn read_prefix_bytes(&self, bytes: &mut [u8]) -> Result<(), GuestAccessError> {
        debug_assert_eq!(bytes.len(), size_of::<T>());
        let typed = u64::try_from(size_of::<T>()).map_err(|_| GuestAccessError::Overflow)?;
        if self.pin.byte_count() == typed {
            return self.pin.read_bytes(bytes);
        }
        let mut whole = vec![0_u8; self.pin.byte_len()?];
        self.pin.read_bytes(&mut whole)?;
        bytes.copy_from_slice(&whole[..size_of::<T>()]);
        Ok(())
    }
}

/// Owning [`GuestPtr`] that keeps its pin alive.
#[derive(Debug)]
pub struct GuestPtrOwned<T> {
    pin: GuestAccessPin,
    _marker: PhantomData<*const T>,
}

impl<T> GuestPtrOwned<T> {
    fn try_acquire(address: u64, access: GuestAccessKind) -> Result<Self, GuestAccessError> {
        let byte_count = u64::try_from(size_of::<T>()).map_err(|_| GuestAccessError::Overflow)?;
        let pin = GuestAccessPin::try_acquire(address, byte_count, access)?;
        require_fit::<T>(&pin, 1)?;
        Ok(Self {
            pin,
            _marker: PhantomData,
        })
    }

    #[must_use]
    pub fn as_ref(&self) -> GuestPtr<'_, T> {
        GuestPtr {
            pin: &self.pin,
            _marker: PhantomData,
        }
    }

    #[must_use]
    pub fn pin(&self) -> &GuestAccessPin {
        &self.pin
    }

    #[must_use]
    pub fn address(&self) -> u64 {
        self.pin.address()
    }
}

impl<T: Copy> GuestPtrOwned<T> {
    pub fn read(&self) -> Result<T, GuestAccessError> {
        self.as_ref().read()
    }

    pub fn write(&self, value: T) -> Result<(), GuestAccessError> {
        self.as_ref().write(value)
    }
}

/// A validated guest slice of `len` `T`s, lifetime-bound to a [`GuestAccessPin`].
#[derive(Debug)]
pub struct GuestSlice<'pin, T> {
    pin: &'pin GuestAccessPin,
    len: usize,
    _marker: PhantomData<*const T>,
}

impl<'pin, T> GuestSlice<'pin, T> {
    /// Views the whole pin as `len` elements of `T`.
    pub fn try_from_pin(
        pin: &'pin GuestAccessPin,
        len: usize,
    ) -> Result<Self, GuestAccessError> {
        require_fit::<T>(pin, len)?;
        Ok(Self {
            pin,
            len,
            _marker: PhantomData,
        })
    }

    /// Acquires a read pin over `len` elements and returns an owning wrapper.
    pub fn try_acquire(address: u64, len: usize) -> Result<GuestSliceOwned<T>, GuestAccessError> {
        GuestSliceOwned::try_acquire(address, len, GuestAccessKind::Read)
    }

    /// Acquires a read/write pin over `len` elements and returns an owning wrapper.
    pub fn try_acquire_mut(
        address: u64,
        len: usize,
    ) -> Result<GuestSliceOwned<T>, GuestAccessError> {
        GuestSliceOwned::try_acquire(address, len, GuestAccessKind::ReadWrite)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub fn address(&self) -> u64 {
        self.pin.address()
    }

    #[must_use]
    pub fn pin(&self) -> &'pin GuestAccessPin {
        self.pin
    }

    #[must_use]
    pub fn byte_count(&self) -> usize {
        self.len.saturating_mul(size_of::<T>())
    }
}

impl GuestSlice<'_, u8> {
    /// Identity-mapped byte view. Prefer [`GuestAccessPin::read_bytes`] when the dispatch lock is
    /// not held across the borrow.
    pub fn as_bytes(&self) -> Result<&[u8], GuestAccessError> {
        let bytes = self.pin.as_bytes()?;
        Ok(&bytes[..self.len])
    }

    /// Identity-mapped mutable byte view.
    #[allow(clippy::mut_from_ref)]
    pub fn as_bytes_mut(&self) -> Result<&mut [u8], GuestAccessError> {
        let bytes = self.pin.as_bytes_mut()?;
        Ok(&mut bytes[..self.len])
    }

    /// Copies the slice into a host `Vec`.
    pub fn to_vec(&self) -> Result<Vec<u8>, GuestAccessError> {
        let mut bytes = vec![0_u8; self.len];
        if self.len == self.pin.byte_len()? {
            self.pin.read_bytes(&mut bytes)?;
        } else {
            let mut whole = vec![0_u8; self.pin.byte_len()?];
            self.pin.read_bytes(&mut whole)?;
            bytes.copy_from_slice(&whole[..self.len]);
        }
        Ok(bytes)
    }
}

/// Owning [`GuestSlice`] that keeps its pin alive.
#[derive(Debug)]
pub struct GuestSliceOwned<T> {
    pin: GuestAccessPin,
    len: usize,
    _marker: PhantomData<*const T>,
}

impl<T> GuestSliceOwned<T> {
    fn try_acquire(
        address: u64,
        len: usize,
        access: GuestAccessKind,
    ) -> Result<Self, GuestAccessError> {
        let element = u64::try_from(size_of::<T>()).map_err(|_| GuestAccessError::Overflow)?;
        let byte_count = element
            .checked_mul(u64::try_from(len).map_err(|_| GuestAccessError::Overflow)?)
            .ok_or(GuestAccessError::Overflow)?;
        let pin = GuestAccessPin::try_acquire(address, byte_count, access)?;
        require_fit::<T>(&pin, len)?;
        Ok(Self {
            pin,
            len,
            _marker: PhantomData,
        })
    }

    #[must_use]
    pub fn as_ref(&self) -> GuestSlice<'_, T> {
        GuestSlice {
            pin: &self.pin,
            len: self.len,
            _marker: PhantomData,
        }
    }

    #[must_use]
    pub fn pin(&self) -> &GuestAccessPin {
        &self.pin
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub fn address(&self) -> u64 {
        self.pin.address()
    }
}

impl GuestSliceOwned<u8> {
    pub fn as_bytes(&self) -> Result<&[u8], GuestAccessError> {
        let bytes = self.pin.as_bytes()?;
        Ok(&bytes[..self.len])
    }

    #[allow(clippy::mut_from_ref)]
    pub fn as_bytes_mut(&self) -> Result<&mut [u8], GuestAccessError> {
        let bytes = self.pin.as_bytes_mut()?;
        Ok(&mut bytes[..self.len])
    }

    pub fn to_vec(&self) -> Result<Vec<u8>, GuestAccessError> {
        self.as_ref().to_vec()
    }

    pub fn write_bytes(&self, source: &[u8]) -> Result<(), GuestAccessError> {
        self.pin.write_bytes(source)
    }
}

fn require_fit<T>(pin: &GuestAccessPin, element_count: usize) -> Result<(), GuestAccessError> {
    let address = pin.address();
    if element_count == 0 {
        return Ok(());
    }
    if !address.is_multiple_of(align_of::<T>() as u64) {
        return Err(GuestAccessError::Misaligned {
            address,
            type_name: std::any::type_name::<T>(),
        });
    }
    let needed = size_of::<T>()
        .checked_mul(element_count)
        .ok_or(GuestAccessError::Overflow)?;
    let Ok(needed_u64) = u64::try_from(needed) else {
        return Err(GuestAccessError::Overflow);
    };
    if pin.byte_count() < needed_u64 {
        return Err(GuestAccessError::TooShort {
            address,
            element_count,
            type_name: std::any::type_name::<T>(),
        });
    }
    Ok(())
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::{
        GuestAccessError, GuestAccessKind, GuestAccessPin, GuestPtr, GuestSlice,
    };
    use crate::{
        GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
        GuestProtection,
    };
    use remu_core::GuestAddress;

    const SPACE_SIZE: u64 = 16 * GUEST_PAGE_SIZE;

    fn map_rw_page() -> (GuestAddressSpace, u64) {
        let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
        let mapped = memory
            .map_shared(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .unwrap();
        (memory, mapped.get())
    }

    #[test]
    fn null_and_unmapped_acquires_refuse_instead_of_dereferencing() {
        assert_eq!(
            GuestAccessPin::try_acquire(0, 8, GuestAccessKind::Read).map(|_| ()),
            Err(GuestAccessError::Null)
        );
        assert!(matches!(
            GuestAccessPin::try_acquire(0x1000, 8, GuestAccessKind::Read),
            Err(GuestAccessError::Unmapped { .. })
        ));
        assert!(matches!(
            GuestPtr::<u32>::try_acquire(0),
            Err(GuestAccessError::Null)
        ));
        assert!(matches!(
            GuestSlice::<u8>::try_acquire(0x2000, 16),
            Err(GuestAccessError::Unmapped { .. })
        ));
    }

    #[test]
    fn pin_read_write_round_trips_through_shared_backing() {
        let (mut memory, address) = map_rw_page();
        memory
            .write(GuestAddress::new(address), &[0x11; 8])
            .unwrap();

        let pin = GuestAccessPin::try_acquire(address, 8, GuestAccessKind::ReadWrite)
            .expect("live shared range pins");
        let mut observed = [0_u8; 8];
        pin.read_bytes(&mut observed).unwrap();
        assert_eq!(observed, [0x11; 8]);

        pin.write_bytes(&[0x5a; 8]).unwrap();
        memory
            .read(GuestAddress::new(address), &mut observed)
            .unwrap();
        assert_eq!(observed, [0x5a; 8]);
    }

    #[test]
    fn identity_mapped_slice_is_bound_to_the_pin() {
        let (mut memory, address) = map_rw_page();
        memory
            .write(GuestAddress::new(address), b"validated")
            .unwrap();

        let slice = GuestSlice::<u8>::try_acquire_mut(address, 9).expect("byte slice");
        assert_eq!(slice.as_bytes().unwrap(), b"validated");
        slice.as_bytes_mut().unwrap().copy_from_slice(b"GUEST-PIN");
        assert_eq!(slice.to_vec().unwrap(), b"GUEST-PIN");
    }

    #[test]
    fn typed_ptr_checks_alignment_and_refuses_read_only_writes() {
        let (mut memory, address) = map_rw_page();
        memory
            .write(
                GuestAddress::new(address),
                &0x1122_3344_u32.to_le_bytes(),
            )
            .unwrap();

        let ptr = GuestPtr::<u32>::try_acquire(address).expect("aligned u32");
        assert_eq!(ptr.read().unwrap(), 0x1122_3344);
        assert_eq!(ptr.write(1), Err(GuestAccessError::ReadOnly));

        let mut_ptr = GuestPtr::<u32>::try_acquire_mut(address).expect("writable u32");
        mut_ptr.write(0xaabb_ccdd).unwrap();
        assert_eq!(mut_ptr.read().unwrap(), 0xaabb_ccdd);

        assert!(matches!(
            GuestPtr::<u32>::try_acquire(address + 1),
            Err(GuestAccessError::Misaligned { .. })
        ));
    }

    #[test]
    fn typed_slice_refuses_a_range_that_exceeds_the_pin() {
        let (_memory, address) = map_rw_page();
        let pin = GuestAccessPin::try_acquire(address, 8, GuestAccessKind::Read).unwrap();
        assert!(matches!(
            GuestSlice::<u32>::try_from_pin(&pin, 3),
            Err(GuestAccessError::TooShort { .. })
        ));
        let words = GuestSlice::<u32>::try_from_pin(&pin, 2).expect("two words fit");
        assert_eq!(words.len(), 2);
        assert_eq!(words.byte_count(), 8);
    }

    #[test]
    fn empty_slice_acquires_without_touching_guest_memory() {
        let slice = GuestSlice::<u8>::try_acquire(0x8000_0000_1000, 0).expect("empty");
        assert!(slice.is_empty());
        assert_eq!(slice.to_vec().unwrap(), Vec::<u8>::new());
        assert_eq!(
            GuestSlice::<u8>::try_acquire(0, 0).map(|_| ()),
            Err(GuestAccessError::Null)
        );
    }

    #[test]
    fn unmapped_after_unmap_revokes_identity_views() {
        let (mut memory, address) = map_rw_page();
        let pin = GuestAccessPin::try_acquire(address, 4, GuestAccessKind::ReadWrite).unwrap();
        assert!(pin.as_bytes().is_ok());

        memory
            .unmap(GuestAddress::new(address), GUEST_PAGE_SIZE)
            .unwrap();
        assert!(!pin.live_va_matches());
        assert!(matches!(
            pin.as_bytes(),
            Err(GuestAccessError::Unmapped { .. })
        ));
    }
}
