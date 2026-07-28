// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Sparse, directly addressable guest virtual memory for the PS5's 16 KiB pages.
//!
//! The address space owns one inaccessible host reservation. Committing anonymous or shared
//! memory replaces pages inside that reservation, and unmapping restores inaccessible pages. This
//! keeps guest pointers directly usable by translated code while preventing fixed mappings from
//! overwriting unrelated host memory.

use std::collections::BTreeMap;
use std::ffi::{CString, c_void};
use std::fmt;
use std::io;
use std::ops::BitOr;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use remu_core::GuestAddress;
use thiserror::Error;

use super::write_watch::CpuWriteWatch;

pub const GUEST_PAGE_SIZE: u64 = 16 * 1024;

static NEXT_BACKING_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_MAPPING_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug)]
struct SharedMappingRecord {
    end: u64,
    backing: GuestPhysicalBacking,
    backing_offset: u64,
}

/// Stable physical-backing identity for a byte range in a shared guest mapping.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GuestSharedMappingIdentity {
    pub backing_id: u64,
    pub backing_offset: u64,
}

/// One live CPU mapping that covers a contiguous portion of a GPU-physical backing range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuestSharedBackingSegment {
    pub address: u64,
    pub byte_count: u64,
}

static SHARED_MAPPINGS: OnceLock<Mutex<BTreeMap<u64, SharedMappingRecord>>> = OnceLock::new();

fn shared_mappings() -> &'static Mutex<BTreeMap<u64, SharedMappingRecord>> {
    SHARED_MAPPINGS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn resolve_shared_backing_range(
    mappings: &BTreeMap<u64, SharedMappingRecord>,
    address: u64,
    byte_count: u64,
) -> Option<(GuestPhysicalBacking, u64, u64)> {
    if byte_count == 0 {
        return None;
    }
    let (&first_start, first) = mappings.range(..=address).next_back()?;
    if address >= first.end {
        return None;
    }
    let backing_id = first.backing.id();
    let physical_start = first.backing_offset.checked_add(address - first_start)?;
    let physical_end = physical_start.checked_add(byte_count)?;
    if physical_end > first.backing.size() {
        return None;
    }
    let mut physical_cursor = physical_start;
    while physical_cursor < physical_end {
        let covered_end = mappings
            .iter()
            .filter(|(_, record)| record.backing.id() == backing_id)
            .filter_map(|(&start, record)| {
                let mapping_length = record.end.checked_sub(start)?;
                let backing_end = record.backing_offset.checked_add(mapping_length)?;
                (record.backing_offset <= physical_cursor && physical_cursor < backing_end)
                    .then_some(backing_end)
            })
            .max()?
            .min(physical_end);
        if covered_end == physical_cursor {
            return None;
        }
        physical_cursor = covered_end;
    }
    Some((first.backing.clone(), physical_start, physical_end))
}

/// Resolves a live shared guest range to its backing object and byte offset.
///
/// Adjacent virtual mappings are accepted when they name contiguous slices of the same backing.
/// Anonymous and guard mappings intentionally return `None` because they cannot alias another VA.
#[must_use]
pub fn guest_shared_mapping_identity(
    address: u64,
    byte_count: u64,
) -> Option<GuestSharedMappingIdentity> {
    if byte_count == 0 {
        return None;
    }
    let end = address.checked_add(byte_count)?;
    let mappings = shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (&start, first) = mappings.range(..=address).next_back()?;
    if address >= first.end {
        return None;
    }
    let identity = GuestSharedMappingIdentity {
        backing_id: first.backing.id(),
        backing_offset: first.backing_offset.checked_add(address - start)?,
    };
    let mut cursor = first.end.min(end);
    let mut expected_backing_offset = first
        .backing_offset
        .checked_add(first.end.checked_sub(start)?)?;
    while cursor < end {
        let next = mappings.get(&cursor)?;
        if next.backing.id() != identity.backing_id
            || next.backing_offset != expected_backing_offset
        {
            return None;
        }
        expected_backing_offset = expected_backing_offset.checked_add(next.end - cursor)?;
        cursor = next.end.min(end);
    }
    Some(identity)
}

/// Resolves a GPU resource range through the shared physical backing behind its first VA.
///
/// Prospero can map adjacent direct-memory pages at unrelated CPU virtual addresses while the GPU
/// resource remains physically contiguous. This returns live CPU aliases which, together, cover
/// that exact physical byte interval. Every byte must still have a live mapping; holes in physical
/// backing coverage are rejected.
#[must_use]
pub fn guest_shared_backing_segments(
    address: u64,
    byte_count: u64,
) -> Option<Vec<GuestSharedBackingSegment>> {
    let mappings = shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (backing, physical_start, physical_end) =
        resolve_shared_backing_range(&mappings, address, byte_count)?;
    let backing_id = backing.id();

    let mut physical_cursor = physical_start;
    let mut segments = Vec::new();
    while physical_cursor < physical_end {
        let candidate = mappings
            .iter()
            .filter(|(_, record)| record.backing.id() == backing_id)
            .filter_map(|(&start, record)| {
                let length = record.end.checked_sub(start)?;
                let backing_end = record.backing_offset.checked_add(length)?;
                (record.backing_offset <= physical_cursor && physical_cursor < backing_end)
                    .then_some((start, record, backing_end))
            })
            // Prefer the alias that covers the most remaining bytes. This avoids needlessly
            // splitting a range when several coherent aliases name the same physical pages.
            .max_by_key(|(_, _, backing_end)| *backing_end)?;
        let (mapping_start, record, mapping_backing_end) = candidate;
        let covered_end = mapping_backing_end.min(physical_end);
        let guest_address = mapping_start.checked_add(physical_cursor - record.backing_offset)?;
        let covered_bytes = covered_end.checked_sub(physical_cursor)?;
        if covered_bytes == 0 {
            return None;
        }
        segments.push(GuestSharedBackingSegment {
            address: guest_address,
            byte_count: covered_bytes,
        });
        physical_cursor = covered_end;
    }
    Some(segments)
}

/// Returns every live CPU alias intersecting a GPU-physical backing range.
///
/// This is intentionally broader than [`guest_shared_backing_segments`]: write watching must
/// observe stores through every coherent alias, not just one representative that covers each
/// physical byte.
#[must_use]
pub fn guest_shared_backing_alias_segments(
    address: u64,
    byte_count: u64,
) -> Option<Vec<GuestSharedBackingSegment>> {
    let mappings = shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (backing, physical_start, physical_end) =
        resolve_shared_backing_range(&mappings, address, byte_count)?;
    let backing_id = backing.id();
    let mut aliases = mappings
        .iter()
        .filter(|(_, record)| record.backing.id() == backing_id)
        .filter_map(|(&mapping_start, record)| {
            let mapping_length = record.end.checked_sub(mapping_start)?;
            let mapping_backing_end = record.backing_offset.checked_add(mapping_length)?;
            let intersection_start = record.backing_offset.max(physical_start);
            let intersection_end = mapping_backing_end.min(physical_end);
            (intersection_start < intersection_end).then(|| GuestSharedBackingSegment {
                address: mapping_start + (intersection_start - record.backing_offset),
                byte_count: intersection_end - intersection_start,
            })
        })
        .collect::<Vec<_>>();
    aliases.sort_unstable_by_key(|segment| (segment.address, segment.byte_count));
    aliases.dedup();
    (!aliases.is_empty()).then_some(aliases)
}

/// Copies a GPU-physical resource range out of its shared direct-memory backing.
///
/// Unlike dereferencing the resource VA, this remains valid when consecutive backing pages are
/// exposed through disjoint CPU aliases. Live-mapping coverage is validated before the transfer.
pub fn read_guest_shared_backing(address: u64, destination: &mut [u8]) -> bool {
    let Ok(byte_count) = u64::try_from(destination.len()) else {
        return false;
    };
    let Some((backing, offset, _)) = ({
        let mappings = shared_mappings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        resolve_shared_backing_range(&mappings, address, byte_count)
    }) else {
        return false;
    };
    backing.read_exact_at(offset, destination)
}

/// Copies bytes into a GPU-physical resource range in shared direct-memory backing.
///
/// CPU aliases observe the write coherently because they map this same shared backing object.
pub fn write_guest_shared_backing(address: u64, source: &[u8]) -> bool {
    let Ok(byte_count) = u64::try_from(source.len()) else {
        return false;
    };
    let Some((backing, offset, _)) = ({
        let mappings = shared_mappings()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        resolve_shared_backing_range(&mappings, address, byte_count)
    }) else {
        return false;
    };
    backing.write_all_at(offset, source)
}

fn unregister_shared_mapping_range(address: u64, byte_count: u64) {
    let Some(end) = address.checked_add(byte_count) else {
        return;
    };
    let mut mappings = shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let overlaps = mappings
        .range(..end)
        .filter(|(start, record)| record.end > address && **start < end)
        .map(|(&start, record)| (start, record.clone()))
        .collect::<Vec<_>>();
    for (start, record) in overlaps {
        mappings.remove(&start);
        if start < address {
            mappings.insert(
                start,
                SharedMappingRecord {
                    end: address,
                    ..record.clone()
                },
            );
        }
        if record.end > end {
            mappings.insert(
                end,
                SharedMappingRecord {
                    end: record.end,
                    backing: record.backing,
                    backing_offset: record.backing_offset + (end - start),
                },
            );
        }
    }
}

fn register_shared_mapping_range(
    address: u64,
    byte_count: u64,
    backing: &GuestPhysicalBacking,
    backing_offset: u64,
) {
    let Some(end) = address.checked_add(byte_count) else {
        return;
    };
    unregister_shared_mapping_range(address, byte_count);
    shared_mappings()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(
            address,
            SharedMappingRecord {
                end,
                backing: backing.clone(),
                backing_offset,
            },
        );
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GuestProtection(u8);

impl GuestProtection {
    pub const NONE: Self = Self(0);
    pub const READ: Self = Self(1 << 0);
    pub const WRITE: Self = Self(1 << 1);
    pub const EXECUTE: Self = Self(1 << 2);
    pub const READ_WRITE: Self = Self(Self::READ.0 | Self::WRITE.0);
    pub const READ_EXECUTE: Self = Self(Self::READ.0 | Self::EXECUTE.0);
    pub const READ_WRITE_EXECUTE: Self = Self(Self::READ.0 | Self::WRITE.0 | Self::EXECUTE.0);

    /// Converts ELF `PF_R`, `PF_W`, and `PF_X` flags used by loaded executable segments.
    #[must_use]
    pub const fn from_elf_flags(flags: u32) -> Self {
        let mut protection = Self::NONE;
        if flags & super::GUEST_READ != 0 {
            protection = protection.union(Self::READ);
        }
        if flags & super::GUEST_WRITE != 0 {
            protection = protection.union(Self::WRITE);
        }
        if flags & super::GUEST_EXECUTE != 0 {
            protection = protection.union(Self::EXECUTE);
        }
        protection
    }

    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn contains(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    const fn host_flags(self) -> i32 {
        let mut flags = libc::PROT_NONE;
        if self.contains(Self::READ) || self.contains(Self::EXECUTE) {
            flags |= libc::PROT_READ;
        }
        if self.contains(Self::WRITE) {
            flags |= libc::PROT_WRITE;
        }
        if self.contains(Self::EXECUTE) {
            flags |= libc::PROT_EXEC;
        }
        flags
    }
}

impl BitOr for GuestProtection {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl fmt::Display for GuestProtection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let read = if self.contains(Self::READ) { 'r' } else { '-' };
        let write = if self.contains(Self::WRITE) { 'w' } else { '-' };
        let execute = if self.contains(Self::EXECUTE) {
            'x'
        } else {
            '-'
        };
        write!(formatter, "{read}{write}{execute}")
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuestMapPlacement {
    Anywhere,
    Hint(GuestAddress),
    Fixed(GuestAddress),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuestMapCoalescing {
    Allowed,
    Disallowed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuestSharedMapOptions {
    pub backing_offset: u64,
    pub coalescing: GuestMapCoalescing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuestVmaState {
    Free,
    Reserved,
    Anonymous,
    Shared,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuestVmaInfo {
    pub start: GuestAddress,
    pub size: u64,
    pub state: GuestVmaState,
    pub protection: GuestProtection,
    pub backing_id: Option<u64>,
    pub backing_offset: u64,
    /// Opaque subsystem-owned metadata preserved across VMA splits.
    pub metadata: u64,
}

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestedAddress {
    Anywhere,
    Exact(u64),
}

impl fmt::Display for RequestedAddress {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Anywhere => formatter.write_str("anywhere"),
            Self::Exact(address) => write!(formatter, "0x{address:x}"),
        }
    }
}

#[derive(Debug)]
struct BackingInner {
    id: u64,
    size: u64,
    descriptor: i32,
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
    inner: Arc<BackingInner>,
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

    fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> bool {
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

    fn write_all_at(&self, offset: u64, source: &[u8]) -> bool {
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

#[derive(Clone, Debug)]
enum VmaKind {
    Reserved,
    Anonymous,
    Shared(GuestPhysicalBacking),
}

impl VmaKind {
    const fn state(&self) -> GuestVmaState {
        match self {
            Self::Reserved => GuestVmaState::Reserved,
            Self::Anonymous => GuestVmaState::Anonymous,
            Self::Shared(_) => GuestVmaState::Shared,
        }
    }
}

#[derive(Clone, Debug)]
struct Vma {
    start: u64,
    end: u64,
    protection: GuestProtection,
    kind: VmaKind,
    backing_offset: u64,
    metadata: u64,
    mapping_id: u64,
}

impl Vma {
    fn info(&self) -> GuestVmaInfo {
        let (backing_id, backing_offset) = match &self.kind {
            VmaKind::Shared(backing) => (Some(backing.inner.id), self.backing_offset),
            VmaKind::Reserved | VmaKind::Anonymous => (None, 0),
        };
        GuestVmaInfo {
            start: GuestAddress::new(self.start),
            size: self.end - self.start,
            state: self.kind.state(),
            protection: self.protection,
            backing_id,
            backing_offset,
            metadata: self.metadata,
        }
    }

    fn can_merge(&self, next: &Self) -> bool {
        if self.end != next.start
            || self.protection != next.protection
            || self.metadata != next.metadata
            || self.mapping_id != next.mapping_id
        {
            return false;
        }
        match (&self.kind, &next.kind) {
            (VmaKind::Reserved, VmaKind::Reserved) | (VmaKind::Anonymous, VmaKind::Anonymous) => {
                true
            }
            (VmaKind::Shared(left), VmaKind::Shared(right)) => {
                left.id() == right.id()
                    && self.backing_offset + (self.end - self.start) == next.backing_offset
            }
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ReservationKind {
    Mmap,
    Mach,
}

#[derive(Debug)]
struct HostReservation {
    address: NonNull<c_void>,
    size: usize,
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
    fn anywhere(size: u64) -> Result<Self, AddressSpaceError> {
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

    fn exact(address: u64, size: u64) -> Result<Self, AddressSpaceError> {
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

    fn address(&self) -> u64 {
        self.address.as_ptr() as usize as u64
    }
}

impl Drop for HostReservation {
    fn drop(&mut self) {
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

#[derive(Debug)]
pub struct GuestAddressSpace {
    reservation: HostReservation,
    end: u64,
    vmas: BTreeMap<u64, Vma>,
}

impl GuestAddressSpace {
    /// Reserves an address space at a host-selected, 16 KiB-aligned address.
    pub fn new(size: u64) -> Result<Self, AddressSpaceError> {
        validate_space_size(size)?;
        let reservation = HostReservation::anywhere(size)?;
        Self::from_reservation(reservation)
    }

    /// Reserves an address space whose guest addresses equal the requested host addresses.
    pub fn new_at(base: GuestAddress, size: u64) -> Result<Self, AddressSpaceError> {
        validate_space_size(size)?;
        require_aligned("address-space base", base.get(), GUEST_PAGE_SIZE)?;
        checked_end(base.get(), size)?;
        let reservation = HostReservation::exact(base.get(), size)?;
        Self::from_reservation(reservation)
    }

    fn from_reservation(reservation: HostReservation) -> Result<Self, AddressSpaceError> {
        let end = checked_end(reservation.address(), reservation.size as u64)?;
        Ok(Self {
            reservation,
            end,
            vmas: BTreeMap::new(),
        })
    }

    #[must_use]
    pub fn base(&self) -> GuestAddress {
        GuestAddress::new(self.reservation.address())
    }

    #[must_use]
    pub fn size(&self) -> u64 {
        self.reservation.size as u64
    }

    #[must_use]
    pub const fn end(&self) -> GuestAddress {
        GuestAddress::new(self.end)
    }

    /// Reserves a guest VMA without granting host access to it.
    pub fn reserve(
        &mut self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
    ) -> Result<GuestAddress, AddressSpaceError> {
        Self::validate_allocation(size, alignment)?;
        let start = self.resolve_free_placement(placement, size, alignment)?;
        let end = checked_end(start, size)?;
        self.vmas.insert(
            start,
            Vma {
                start,
                end,
                protection: GuestProtection::NONE,
                kind: VmaKind::Reserved,
                backing_offset: 0,
                metadata: 0,
                mapping_id: 0,
            },
        );
        self.normalize();
        Ok(GuestAddress::new(start))
    }

    /// Maps zero-filled private memory. A fixed map may commit free or reserved pages, but never
    /// replaces an existing anonymous or shared mapping.
    pub fn map_anonymous(
        &mut self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
        protection: GuestProtection,
    ) -> Result<GuestAddress, AddressSpaceError> {
        Self::validate_allocation(size, alignment)?;
        let start = self.resolve_mapping_placement(placement, size, alignment)?;
        let end = checked_end(start, size)?;
        self.map_host_anonymous(start, size, protection)?;
        self.remove_metadata_range(start, end);
        self.vmas.insert(
            start,
            Vma {
                start,
                end,
                protection,
                kind: VmaKind::Anonymous,
                backing_offset: 0,
                metadata: 0,
                mapping_id: 0,
            },
        );
        self.normalize();
        Ok(GuestAddress::new(start))
    }

    /// Maps a slice of physical backing. Mapping the same slice more than once creates coherent
    /// aliases without copying.
    pub fn map_shared(
        &mut self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
        protection: GuestProtection,
        backing: &GuestPhysicalBacking,
        backing_offset: u64,
    ) -> Result<GuestAddress, AddressSpaceError> {
        self.map_shared_with_coalescing(
            placement,
            size,
            alignment,
            protection,
            backing,
            GuestSharedMapOptions {
                backing_offset,
                coalescing: GuestMapCoalescing::Allowed,
            },
        )
    }

    /// Maps a shared backing slice while preserving independent mapping-call boundaries when
    /// coalescing is disabled by the guest ABI.
    pub fn map_shared_with_coalescing(
        &mut self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
        protection: GuestProtection,
        backing: &GuestPhysicalBacking,
        options: GuestSharedMapOptions,
    ) -> Result<GuestAddress, AddressSpaceError> {
        let GuestSharedMapOptions {
            backing_offset,
            coalescing,
        } = options;
        Self::validate_allocation(size, alignment)?;
        require_aligned("physical backing offset", backing_offset, GUEST_PAGE_SIZE)?;
        let backing_end = checked_end(backing_offset, size)?;
        if backing_end > backing.size() {
            return Err(AddressSpaceError::BackingOutOfBounds {
                backing_id: backing.id(),
                offset: backing_offset,
                end: backing_end,
                backing_size: backing.size(),
            });
        }
        let start = self.resolve_mapping_placement(placement, size, alignment)?;
        let end = checked_end(start, size)?;
        self.map_host_shared(start, size, protection, backing, backing_offset)?;
        self.remove_metadata_range(start, end);
        self.vmas.insert(
            start,
            Vma {
                start,
                end,
                protection,
                kind: VmaKind::Shared(backing.clone()),
                backing_offset,
                metadata: 0,
                mapping_id: match coalescing {
                    GuestMapCoalescing::Allowed => 0,
                    GuestMapCoalescing::Disallowed => next_mapping_id(),
                },
            },
        );
        self.normalize();
        Ok(GuestAddress::new(start))
    }

    /// Releases mapped or reserved pages and restores the host guard mapping.
    pub fn unmap(&mut self, address: GuestAddress, size: u64) -> Result<(), AddressSpaceError> {
        let (start, end) = self.validate_page_range(address, size)?;
        self.ensure_covered(start, end, false, None)?;
        self.map_host_anonymous(start, size, GuestProtection::NONE)?;
        self.remove_metadata_range(start, end);
        self.normalize();
        Ok(())
    }

    /// Changes protection on claimed pages. Granting access to reserved pages commits anonymous
    /// memory there; partial ranges split VMAs and equal adjacent pieces are merged again
    /// automatically.
    pub fn protect(
        &mut self,
        address: GuestAddress,
        size: u64,
        protection: GuestProtection,
    ) -> Result<(), AddressSpaceError> {
        let (start, end) = self.validate_page_range(address, size)?;
        self.ensure_covered(start, end, false, None)?;
        let reprotection = CpuWriteWatch::begin_reprotect_overlapping(start, size);
        let address_pointer = host_pointer(start)?;
        // SAFETY: Validation proves the complete range is claimed inside our host reservation.
        let result =
            unsafe { libc::mprotect(address_pointer, host_size(size)?, protection.host_flags()) };
        drop(reprotection);
        if result != 0 {
            return Err(host_error("protection change", start, end));
        }
        self.split_at(start);
        self.split_at(end);
        for (_, vma) in self.vmas.range_mut(start..end) {
            vma.protection = protection;
            if protection != GuestProtection::NONE && matches!(vma.kind, VmaKind::Reserved) {
                vma.kind = VmaKind::Anonymous;
            }
        }
        self.normalize();
        Ok(())
    }

    /// Associates opaque subsystem metadata with mapped pages. The value participates in VMA
    /// merging and is retained by partial unmap and protection operations.
    pub fn set_metadata(
        &mut self,
        address: GuestAddress,
        size: u64,
        metadata: u64,
    ) -> Result<(), AddressSpaceError> {
        let (start, end) = self.validate_page_range(address, size)?;
        self.ensure_covered(start, end, true, None)?;
        self.split_at(start);
        self.split_at(end);
        for (_, vma) in self.vmas.range_mut(start..end) {
            vma.metadata = metadata;
        }
        self.normalize();
        Ok(())
    }

    /// Returns either the tracked VMA or the complete free gap containing `address`. Protection and
    /// metadata surgery may split one mapping internally; guest mappings created with coalescing
    /// disabled retain their original query boundaries.
    pub fn query(&self, address: GuestAddress) -> Result<GuestVmaInfo, AddressSpaceError> {
        let value = address.get();
        if value < self.base().get() || value >= self.end {
            return Err(AddressSpaceError::OutOfBounds {
                address: value,
                end: value.saturating_add(1),
                space_start: self.base().get(),
                space_end: self.end,
            });
        }
        if let Some((_, vma)) = self.vmas.range(..=value).next_back()
            && value < vma.end
        {
            return Ok(vma.info());
        }
        let free_start = self
            .vmas
            .range(..=value)
            .next_back()
            .map_or(self.base().get(), |(_, vma)| vma.end);
        let free_end = self
            .vmas
            .range(value..)
            .next()
            .map_or(self.end, |(_, vma)| vma.start);
        Ok(GuestVmaInfo {
            start: GuestAddress::new(free_start),
            size: free_end - free_start,
            state: GuestVmaState::Free,
            protection: GuestProtection::NONE,
            backing_id: None,
            backing_offset: 0,
            metadata: 0,
        })
    }

    /// Returns explicit reserved and mapped VMAs in ascending address order. Free gaps are obtained
    /// with [`Self::query`].
    #[must_use]
    pub fn vmas(&self) -> Vec<GuestVmaInfo> {
        self.vmas.values().map(Vma::info).collect()
    }

    pub fn read(
        &self,
        address: GuestAddress,
        destination: &mut [u8],
    ) -> Result<(), AddressSpaceError> {
        let (start, end) = self.validate_byte_range(address, destination.len())?;
        if destination.is_empty() {
            return Ok(());
        }
        self.ensure_covered(start, end, true, Some(GuestProtection::READ))?;
        // SAFETY: Permission and coverage validation prove the source is readable for this length.
        unsafe {
            std::ptr::copy(
                host_pointer(start)?.cast_const().cast::<u8>(),
                destination.as_mut_ptr(),
                destination.len(),
            );
        }
        Ok(())
    }

    pub fn write(&mut self, address: GuestAddress, source: &[u8]) -> Result<(), AddressSpaceError> {
        let (start, end) = self.validate_byte_range(address, source.len())?;
        if source.is_empty() {
            return Ok(());
        }
        self.ensure_covered(start, end, true, Some(GuestProtection::WRITE))?;
        // SAFETY: Permission and coverage validation prove the destination is writable for this
        // length and `source` provides readable bytes of the same length.
        unsafe {
            std::ptr::copy(
                source.as_ptr(),
                host_pointer(start)?.cast::<u8>(),
                source.len(),
            );
        }
        Ok(())
    }

    /// Returns the identity-mapped host pointer after verifying that the byte is mapped.
    pub fn host_pointer(&self, address: GuestAddress) -> Result<NonNull<u8>, AddressSpaceError> {
        let value = address.get();
        let end = checked_end(value, 1)?;
        self.validate_bounds(value, end)?;
        self.ensure_covered(value, end, true, None)?;
        NonNull::new(host_pointer(value)?.cast::<u8>())
            .ok_or(AddressSpaceError::HostSizeOverflow { value })
    }

    fn validate_allocation(size: u64, alignment: u64) -> Result<(), AddressSpaceError> {
        if size == 0 {
            return Err(AddressSpaceError::ZeroSize);
        }
        require_aligned("mapping size", size, GUEST_PAGE_SIZE)?;
        if alignment < GUEST_PAGE_SIZE
            || !alignment.is_power_of_two()
            || !alignment.is_multiple_of(GUEST_PAGE_SIZE)
        {
            return Err(AddressSpaceError::InvalidAlignment { alignment });
        }
        Ok(())
    }

    fn validate_page_range(
        &self,
        address: GuestAddress,
        size: u64,
    ) -> Result<(u64, u64), AddressSpaceError> {
        if size == 0 {
            return Err(AddressSpaceError::ZeroSize);
        }
        require_aligned("range address", address.get(), GUEST_PAGE_SIZE)?;
        require_aligned("range size", size, GUEST_PAGE_SIZE)?;
        let end = checked_end(address.get(), size)?;
        self.validate_bounds(address.get(), end)?;
        Ok((address.get(), end))
    }

    fn validate_byte_range(
        &self,
        address: GuestAddress,
        size: usize,
    ) -> Result<(u64, u64), AddressSpaceError> {
        let size_u64 = u64::try_from(size)
            .map_err(|_| AddressSpaceError::HostSizeOverflow { value: u64::MAX })?;
        let end = checked_end(address.get(), size_u64)?;
        if size == 0 && address.get() == self.end {
            return Ok((address.get(), end));
        }
        self.validate_bounds(address.get(), end)?;
        Ok((address.get(), end))
    }

    fn validate_bounds(&self, start: u64, end: u64) -> Result<(), AddressSpaceError> {
        if start < self.base().get() || end > self.end {
            return Err(AddressSpaceError::OutOfBounds {
                address: start,
                end,
                space_start: self.base().get(),
                space_end: self.end,
            });
        }
        Ok(())
    }

    fn resolve_free_placement(
        &self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
    ) -> Result<u64, AddressSpaceError> {
        match placement {
            GuestMapPlacement::Fixed(address) => {
                let start = address.get();
                require_aligned("fixed address", start, alignment)?;
                let end = checked_end(start, size)?;
                self.validate_bounds(start, end)?;
                self.reject_any_overlap(start, end)?;
                Ok(start)
            }
            GuestMapPlacement::Anywhere => self
                .find_free(self.base().get(), self.end, size, alignment)
                .ok_or(AddressSpaceError::AddressSpaceExhausted { size, alignment }),
            GuestMapPlacement::Hint(hint) => {
                self.validate_hint(hint.get())?;
                self.find_free(hint.get(), self.end, size, alignment)
                    .or_else(|| self.find_free(self.base().get(), hint.get(), size, alignment))
                    .ok_or(AddressSpaceError::AddressSpaceExhausted { size, alignment })
            }
        }
    }

    fn resolve_mapping_placement(
        &self,
        placement: GuestMapPlacement,
        size: u64,
        alignment: u64,
    ) -> Result<u64, AddressSpaceError> {
        if let GuestMapPlacement::Fixed(address) = placement {
            let start = address.get();
            require_aligned("fixed address", start, alignment)?;
            let end = checked_end(start, size)?;
            self.validate_bounds(start, end)?;
            self.reject_mapped_overlap(start, end)?;
            return Ok(start);
        }
        self.resolve_free_placement(placement, size, alignment)
    }

    fn validate_hint(&self, hint: u64) -> Result<(), AddressSpaceError> {
        if hint < self.base().get() || hint >= self.end {
            return Err(AddressSpaceError::OutOfBounds {
                address: hint,
                end: hint.saturating_add(1),
                space_start: self.base().get(),
                space_end: self.end,
            });
        }
        Ok(())
    }

    fn find_free(&self, search_start: u64, limit: u64, size: u64, alignment: u64) -> Option<u64> {
        let mut candidate = align_up(search_start, alignment).ok()?;
        for vma in self.vmas.values() {
            if vma.end <= candidate {
                continue;
            }
            if vma.start >= limit {
                break;
            }
            if checked_end(candidate, size).ok()? <= vma.start
                && checked_end(candidate, size).ok()? <= limit
            {
                return Some(candidate);
            }
            candidate = align_up(vma.end, alignment).ok()?;
            if candidate >= limit {
                return None;
            }
        }
        (checked_end(candidate, size).ok()? <= limit).then_some(candidate)
    }

    fn reject_any_overlap(&self, start: u64, end: u64) -> Result<(), AddressSpaceError> {
        if let Some(vma) = self.first_overlap(start, end) {
            return Err(overlap_error(start, end, vma));
        }
        Ok(())
    }

    fn reject_mapped_overlap(&self, start: u64, end: u64) -> Result<(), AddressSpaceError> {
        if let Some(vma) = self.vmas.values().find(|vma| {
            vma.start < end && vma.end > start && !matches!(vma.kind, VmaKind::Reserved)
        }) {
            return Err(overlap_error(start, end, vma));
        }
        Ok(())
    }

    fn first_overlap(&self, start: u64, end: u64) -> Option<&Vma> {
        self.vmas
            .values()
            .find(|vma| vma.start < end && vma.end > start)
    }

    fn ensure_covered(
        &self,
        start: u64,
        end: u64,
        require_mapped: bool,
        required: Option<GuestProtection>,
    ) -> Result<(), AddressSpaceError> {
        let mut cursor = start;
        while cursor < end {
            let Some((_, vma)) = self.vmas.range(..=cursor).next_back() else {
                return Err(AddressSpaceError::NotMapped { address: cursor });
            };
            if cursor >= vma.end {
                return Err(AddressSpaceError::NotMapped { address: cursor });
            }
            if require_mapped && matches!(vma.kind, VmaKind::Reserved) {
                return Err(AddressSpaceError::Reserved { address: cursor });
            }
            if let Some(required) = required
                && !vma.protection.contains(required)
            {
                return Err(AddressSpaceError::PermissionDenied {
                    address: cursor,
                    end: end.min(vma.end),
                    required,
                    actual: vma.protection,
                });
            }
            cursor = end.min(vma.end);
        }
        Ok(())
    }

    fn map_host_anonymous(
        &self,
        start: u64,
        size: u64,
        protection: GuestProtection,
    ) -> Result<(), AddressSpaceError> {
        debug_assert!(start >= self.base().get());
        debug_assert!(start.saturating_add(size) <= self.end);
        CpuWriteWatch::retire_overlapping(start, size);
        let pointer = host_pointer(start)?;
        // SAFETY: Placement validation proves this fixed range lies entirely inside our host
        // reservation and does not cover another mapped VMA.
        let result = unsafe {
            libc::mmap(
                pointer,
                host_size(size)?,
                protection.host_flags(),
                libc::MAP_FIXED | libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if result == libc::MAP_FAILED {
            return Err(host_error(
                "anonymous mapping",
                start,
                checked_end(start, size)?,
            ));
        }
        debug_assert_eq!(result, pointer);
        unregister_shared_mapping_range(start, size);
        Ok(())
    }

    fn map_host_shared(
        &self,
        start: u64,
        size: u64,
        protection: GuestProtection,
        backing: &GuestPhysicalBacking,
        backing_offset: u64,
    ) -> Result<(), AddressSpaceError> {
        debug_assert!(start >= self.base().get());
        debug_assert!(start.saturating_add(size) <= self.end);
        CpuWriteWatch::retire_overlapping(start, size);
        let pointer = host_pointer(start)?;
        let offset = libc::off_t::try_from(backing_offset).map_err(|_| {
            AddressSpaceError::HostSizeOverflow {
                value: backing_offset,
            }
        })?;
        // SAFETY: Placement and backing validation prove both the fixed VA range and file range.
        let result = unsafe {
            libc::mmap(
                pointer,
                host_size(size)?,
                protection.host_flags(),
                libc::MAP_FIXED | libc::MAP_SHARED,
                backing.inner.descriptor,
                offset,
            )
        };
        if result == libc::MAP_FAILED {
            return Err(host_error(
                "shared mapping",
                start,
                checked_end(start, size)?,
            ));
        }
        debug_assert_eq!(result, pointer);
        register_shared_mapping_range(start, size, backing, backing_offset);
        Ok(())
    }

    fn remove_metadata_range(&mut self, start: u64, end: u64) {
        let keys: Vec<_> = self
            .vmas
            .iter()
            .filter_map(|(&key, vma)| (vma.start < end && vma.end > start).then_some(key))
            .collect();
        for key in keys {
            let original = self.vmas.remove(&key).expect("VMA key was just collected");
            if original.start < start {
                let mut left = original.clone();
                left.end = start;
                self.vmas.insert(left.start, left);
            }
            if original.end > end {
                let mut right = original;
                right.backing_offset += end - right.start;
                right.start = end;
                self.vmas.insert(right.start, right);
            }
        }
    }

    fn split_at(&mut self, address: u64) {
        let Some((&key, vma)) = self.vmas.range(..=address).next_back() else {
            return;
        };
        if address <= vma.start || address >= vma.end {
            return;
        }
        let mut left = self.vmas.remove(&key).expect("VMA key exists");
        let mut right = left.clone();
        right.start = address;
        right.backing_offset += address - left.start;
        left.end = address;
        self.vmas.insert(left.start, left);
        self.vmas.insert(right.start, right);
    }

    fn normalize(&mut self) {
        let old = std::mem::take(&mut self.vmas);
        for (_, vma) in old {
            if let Some(previous_key) = self.vmas.last_key_value().map(|(&key, _)| key) {
                let previous = self
                    .vmas
                    .get_mut(&previous_key)
                    .expect("last VMA key was just observed");
                if previous.can_merge(&vma) {
                    previous.end = vma.end;
                    continue;
                }
            }
            self.vmas.insert(vma.start, vma);
        }
    }
}

fn validate_space_size(size: u64) -> Result<(), AddressSpaceError> {
    if size == 0 {
        return Err(AddressSpaceError::ZeroSize);
    }
    require_aligned("address-space size", size, GUEST_PAGE_SIZE)?;
    let host_page_size = host_page_size()?;
    if host_page_size > GUEST_PAGE_SIZE || !GUEST_PAGE_SIZE.is_multiple_of(host_page_size) {
        return Err(AddressSpaceError::IncompatibleHostPageSize { host_page_size });
    }
    host_size(size)?;
    Ok(())
}

fn host_page_size() -> Result<u64, AddressSpaceError> {
    // SAFETY: sysconf has no pointer preconditions.
    let raw = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    let size = u64::try_from(raw).map_err(|_| AddressSpaceError::InvalidHostPageSize)?;
    if !size.is_power_of_two() {
        return Err(AddressSpaceError::InvalidHostPageSize);
    }
    Ok(size)
}

fn next_mapping_id() -> u64 {
    loop {
        let id = NEXT_MAPPING_ID.fetch_add(1, Ordering::Relaxed);
        if id != 0 {
            return id;
        }
    }
}

fn require_aligned(
    field: &'static str,
    value: u64,
    alignment: u64,
) -> Result<(), AddressSpaceError> {
    if !value.is_multiple_of(alignment) {
        return Err(AddressSpaceError::Unaligned {
            field,
            value,
            alignment,
        });
    }
    Ok(())
}

const fn checked_end(address: u64, size: u64) -> Result<u64, AddressSpaceError> {
    match address.checked_add(size) {
        Some(end) => Ok(end),
        None => Err(AddressSpaceError::AddressOverflow { address, size }),
    }
}

fn align_up(value: u64, alignment: u64) -> Result<u64, AddressSpaceError> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum & !(alignment - 1))
        .ok_or(AddressSpaceError::AddressOverflow {
            address: value,
            size: alignment - 1,
        })
}

fn host_size(value: u64) -> Result<usize, AddressSpaceError> {
    usize::try_from(value).map_err(|_| AddressSpaceError::HostSizeOverflow { value })
}

fn host_pointer(value: u64) -> Result<*mut c_void, AddressSpaceError> {
    Ok(host_size(value)? as *mut c_void)
}

fn overlap_error(start: u64, end: u64, vma: &Vma) -> AddressSpaceError {
    AddressSpaceError::Overlap {
        address: start,
        end,
        existing_start: vma.start,
        existing_end: vma.end,
        state: vma.kind.state(),
    }
}

fn host_error(operation: &'static str, start: u64, end: u64) -> AddressSpaceError {
    AddressSpaceError::HostOperationFailed {
        operation,
        address: start,
        end,
        source: io::Error::last_os_error(),
    }
}

const KERN_SUCCESS: i32 = 0;
const VM_FLAGS_FIXED: i32 = 0;

#[link(name = "System")]
unsafe extern "C" {
    fn mach_task_self() -> u32;
    fn mach_vm_allocate(target: u32, address: *mut u64, size: u64, flags: i32) -> i32;
    fn mach_vm_deallocate(target: u32, address: u64, size: u64) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPACE_SIZE: u64 = 16 * GUEST_PAGE_SIZE;

    fn page(base: GuestAddress, index: u64) -> GuestAddress {
        GuestAddress::new(base.get() + index * GUEST_PAGE_SIZE)
    }

    #[test]
    fn creates_aligned_guarded_address_space_and_queries_free_gap() {
        let memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        assert_eq!(memory.base().get() % GUEST_PAGE_SIZE, 0);
        assert_eq!(memory.size(), SPACE_SIZE);
        assert_eq!(
            memory.query(memory.base()).unwrap(),
            GuestVmaInfo {
                start: memory.base(),
                size: SPACE_SIZE,
                state: GuestVmaState::Free,
                protection: GuestProtection::NONE,
                backing_id: None,
                backing_offset: 0,
                metadata: 0,
            }
        );
    }

    #[test]
    fn reservations_and_hints_have_deterministic_first_fit_order() {
        let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        let base = memory.base();
        let middle = memory
            .reserve(
                GuestMapPlacement::Fixed(page(base, 4)),
                2 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
            )
            .unwrap();
        assert_eq!(middle, page(base, 4));
        let after = memory
            .reserve(
                GuestMapPlacement::Hint(page(base, 4)),
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
            )
            .unwrap();
        assert_eq!(after, page(base, 6));
        let first = memory
            .reserve(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
            )
            .unwrap();
        assert_eq!(first, base);
    }

    #[test]
    fn fixed_mapping_commits_a_reservation_but_rejects_live_mapping_overlap() {
        let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        let target = page(memory.base(), 2);
        memory
            .reserve(
                GuestMapPlacement::Fixed(target),
                3 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
            )
            .unwrap();
        memory
            .map_anonymous(
                GuestMapPlacement::Fixed(page(target, 1)),
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
            )
            .unwrap();
        assert_eq!(memory.vmas().len(), 3);
        let error = memory
            .map_anonymous(
                GuestMapPlacement::Fixed(page(target, 1)),
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            AddressSpaceError::Overlap {
                state: GuestVmaState::Anonymous,
                ..
            }
        ));
    }

    #[test]
    fn protect_splits_and_then_merges_anonymous_vmas() {
        let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        let base = memory
            .map_anonymous(
                GuestMapPlacement::Anywhere,
                3 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
            )
            .unwrap();
        memory
            .protect(page(base, 1), GUEST_PAGE_SIZE, GuestProtection::READ)
            .unwrap();
        let vmas = memory.vmas();
        assert_eq!(vmas.len(), 3);
        assert_eq!(vmas[1].protection, GuestProtection::READ);
        memory
            .protect(page(base, 1), GUEST_PAGE_SIZE, GuestProtection::READ_WRITE)
            .unwrap();
        assert_eq!(memory.vmas().len(), 1);
    }

    #[test]
    fn protection_commits_only_the_selected_reserved_pages() {
        let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        let reserved = memory
            .reserve(
                GuestMapPlacement::Anywhere,
                3 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
            )
            .unwrap();

        let committed = page(reserved, 1);
        memory
            .protect(committed, GUEST_PAGE_SIZE, GuestProtection::READ_WRITE)
            .unwrap();
        memory.write(committed, b"committed").unwrap();

        assert_eq!(
            memory.query(reserved).unwrap().state,
            GuestVmaState::Reserved
        );
        assert_eq!(
            memory.query(committed).unwrap().state,
            GuestVmaState::Anonymous
        );
        assert_eq!(
            memory.query(page(reserved, 2)).unwrap().state,
            GuestVmaState::Reserved
        );
        assert!(matches!(
            memory.map_anonymous(
                GuestMapPlacement::Fixed(committed),
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
            ),
            Err(AddressSpaceError::Overlap {
                state: GuestVmaState::Anonymous,
                ..
            })
        ));
    }

    #[test]
    fn opaque_metadata_follows_split_and_merge_range_surgery() {
        let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        let base = memory
            .map_anonymous(
                GuestMapPlacement::Anywhere,
                3 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
            )
            .unwrap();
        memory
            .set_metadata(page(base, 1), GUEST_PAGE_SIZE, 42)
            .unwrap();
        let split = memory.vmas();
        assert_eq!(split.len(), 3);
        assert_eq!(split[1].metadata, 42);
        memory.set_metadata(base, 3 * GUEST_PAGE_SIZE, 7).unwrap();
        let merged = memory.vmas();
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].metadata, 7);
    }

    #[test]
    fn unmap_splits_vma_and_freed_hole_is_reused() {
        let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        let base = memory
            .map_anonymous(
                GuestMapPlacement::Anywhere,
                4 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
            )
            .unwrap();
        memory.unmap(page(base, 1), 2 * GUEST_PAGE_SIZE).unwrap();
        assert_eq!(memory.vmas().len(), 2);
        assert_eq!(
            memory.query(page(base, 1)).unwrap().state,
            GuestVmaState::Free
        );
        let reused = memory
            .map_anonymous(
                GuestMapPlacement::Hint(page(base, 1)),
                2 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
            )
            .unwrap();
        assert_eq!(reused, page(base, 1));
        assert_eq!(memory.vmas().len(), 1);
    }

    #[test]
    fn shared_backing_aliases_are_coherent() {
        let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).unwrap();
        let first = memory
            .map_shared(
                GuestMapPlacement::Anywhere,
                2 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .unwrap();
        let second = memory
            .map_shared(
                GuestMapPlacement::Anywhere,
                2 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .unwrap();
        let backing_id = backing.id();
        drop(backing);
        memory.write(page(first, 1), b"alias").unwrap();
        let mut bytes = [0_u8; 5];
        memory.read(page(second, 1), &mut bytes).unwrap();
        assert_eq!(&bytes, b"alias");
        assert_eq!(memory.query(second).unwrap().backing_id, Some(backing_id));
        assert_eq!(
            guest_shared_mapping_identity(first.get(), 2 * GUEST_PAGE_SIZE),
            Some(GuestSharedMappingIdentity {
                backing_id,
                backing_offset: 0,
            })
        );
        assert_eq!(
            guest_shared_mapping_identity(page(second, 1).get(), GUEST_PAGE_SIZE),
            Some(GuestSharedMappingIdentity {
                backing_id,
                backing_offset: GUEST_PAGE_SIZE,
            })
        );
    }

    #[test]
    fn private_backing_write_is_coherent_without_falsely_dirtying_cpu_watches() {
        let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
        let first = memory
            .map_shared(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .unwrap();
        let second = memory
            .map_shared(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .unwrap();
        // SAFETY: both mappings remain live until the watches are dropped at the end of the test.
        let first_watch =
            unsafe { CpuWriteWatch::new(first.get(), GUEST_PAGE_SIZE).expect("first watch") };
        // SAFETY: same backing and lifetime guarantee as the first alias.
        let second_watch =
            unsafe { CpuWriteWatch::new(second.get(), GUEST_PAGE_SIZE).expect("second watch") };

        assert!(write_guest_shared_backing(first.get(), b"gpu"));
        let mut coherent = [0_u8; 3];
        memory.read(second, &mut coherent).unwrap();
        assert_eq!(&coherent, b"gpu");
        assert_eq!(first_watch.generation(), 0);
        assert_eq!(second_watch.generation(), 0);
        assert!(!first_watch.is_dirty());
        assert!(!second_watch.is_dirty());

        // Fault ownership is virtual-address-specific. A texture watcher covers every physical
        // alias and therefore observes this store through `first_watch`; the untouched alias does
        // not need a duplicate dirty transition.
        memory.write(first, b"cpu").unwrap();
        assert!(first_watch.is_dirty());
        assert!(!second_watch.is_dirty());
        assert_eq!(first_watch.generation(), 1);
        assert_eq!(second_watch.generation(), 0);

        assert!(write_guest_shared_backing(first.get(), b"gpu"));
        assert!(first_watch.is_dirty());
        assert_eq!(first_watch.generation(), 1);
    }

    #[test]
    fn replacing_shared_pages_removes_their_backing_identity() {
        let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).unwrap();
        let mapped = memory
            .map_shared(
                GuestMapPlacement::Anywhere,
                2 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .unwrap();
        assert!(guest_shared_mapping_identity(mapped.get(), GUEST_PAGE_SIZE).is_some());

        memory.unmap(mapped, 2 * GUEST_PAGE_SIZE).unwrap();

        assert_eq!(
            guest_shared_mapping_identity(mapped.get(), GUEST_PAGE_SIZE),
            None
        );
    }

    #[test]
    fn contiguous_shared_slices_merge_but_discontinuous_slices_do_not() {
        let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        let backing = GuestPhysicalBacking::new(4 * GUEST_PAGE_SIZE).unwrap();
        let base = memory.base();
        for (virtual_page, physical_page) in [(0, 0), (1, 1), (2, 3)] {
            memory
                .map_shared(
                    GuestMapPlacement::Fixed(page(base, virtual_page)),
                    GUEST_PAGE_SIZE,
                    GUEST_PAGE_SIZE,
                    GuestProtection::READ_WRITE,
                    &backing,
                    physical_page * GUEST_PAGE_SIZE,
                )
                .unwrap();
        }
        let vmas = memory.vmas();
        assert_eq!(vmas.len(), 2);
        assert_eq!(vmas[0].size, 2 * GUEST_PAGE_SIZE);
        assert_eq!(vmas[1].backing_offset, 3 * GUEST_PAGE_SIZE);
    }

    #[test]
    fn gpu_backing_range_spans_disjoint_virtual_aliases() {
        let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        let backing = GuestPhysicalBacking::new(4 * GUEST_PAGE_SIZE).unwrap();
        let base = memory.base();
        let first = memory
            .map_shared(
                GuestMapPlacement::Fixed(page(base, 0)),
                2 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .unwrap();
        let second = memory
            .map_shared(
                GuestMapPlacement::Fixed(page(base, 8)),
                2 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                2 * GUEST_PAGE_SIZE,
            )
            .unwrap();
        let source = (0..4 * usize::try_from(GUEST_PAGE_SIZE).unwrap())
            .map(|index| u8::try_from(index % 251).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(
            guest_shared_mapping_identity(first.get(), 4 * GUEST_PAGE_SIZE),
            None,
            "the CPU virtual interval contains a guard hole"
        );
        assert_eq!(
            guest_shared_backing_segments(first.get(), 4 * GUEST_PAGE_SIZE),
            Some(vec![
                GuestSharedBackingSegment {
                    address: first.get(),
                    byte_count: 2 * GUEST_PAGE_SIZE,
                },
                GuestSharedBackingSegment {
                    address: second.get(),
                    byte_count: 2 * GUEST_PAGE_SIZE,
                },
            ])
        );
        assert!(write_guest_shared_backing(first.get(), &source));
        let mut second_half = vec![0; usize::try_from(2 * GUEST_PAGE_SIZE).unwrap()];
        memory.read(second, &mut second_half).unwrap();
        assert_eq!(
            second_half,
            source[usize::try_from(2 * GUEST_PAGE_SIZE).unwrap()..]
        );
        let mut round_trip = vec![0; source.len()];
        assert!(read_guest_shared_backing(first.get(), &mut round_trip));
        assert_eq!(round_trip, source);
    }

    #[test]
    fn gpu_backing_range_rejects_unmapped_physical_holes() {
        let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        let backing = GuestPhysicalBacking::new(4 * GUEST_PAGE_SIZE).unwrap();
        let base = memory.base();
        let first = memory
            .map_shared(
                GuestMapPlacement::Fixed(page(base, 0)),
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .unwrap();
        memory
            .map_shared(
                GuestMapPlacement::Fixed(page(base, 8)),
                2 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                2 * GUEST_PAGE_SIZE,
            )
            .unwrap();

        assert_eq!(
            guest_shared_backing_segments(first.get(), 4 * GUEST_PAGE_SIZE),
            None
        );
        assert!(!read_guest_shared_backing(
            first.get(),
            &mut vec![0; usize::try_from(4 * GUEST_PAGE_SIZE).unwrap()]
        ));
    }

    #[test]
    fn gpu_backing_alias_enumeration_tracks_aliases_and_remaps() {
        let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).unwrap();
        let replacement = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
        let base = memory.base();
        let first = memory
            .map_shared(
                GuestMapPlacement::Fixed(page(base, 0)),
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .unwrap();
        let continuation = memory
            .map_shared(
                GuestMapPlacement::Fixed(page(base, 4)),
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                GUEST_PAGE_SIZE,
            )
            .unwrap();
        let alias = memory
            .map_shared(
                GuestMapPlacement::Fixed(page(base, 8)),
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .unwrap();

        assert_eq!(
            guest_shared_backing_alias_segments(first.get(), 2 * GUEST_PAGE_SIZE),
            Some(vec![
                GuestSharedBackingSegment {
                    address: first.get(),
                    byte_count: GUEST_PAGE_SIZE,
                },
                GuestSharedBackingSegment {
                    address: continuation.get(),
                    byte_count: GUEST_PAGE_SIZE,
                },
                GuestSharedBackingSegment {
                    address: alias.get(),
                    byte_count: GUEST_PAGE_SIZE,
                },
            ])
        );

        memory.unmap(continuation, GUEST_PAGE_SIZE).unwrap();
        assert_eq!(
            guest_shared_backing_alias_segments(first.get(), 2 * GUEST_PAGE_SIZE),
            None,
            "removing the only continuation invalidates complete backing coverage"
        );
        memory
            .map_shared(
                GuestMapPlacement::Fixed(continuation),
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &replacement,
                0,
            )
            .unwrap();
        assert_eq!(
            guest_shared_backing_alias_segments(first.get(), 2 * GUEST_PAGE_SIZE),
            None,
            "a different physical backing cannot repair the resource span"
        );
    }

    #[test]
    fn no_coalesce_shared_maps_retain_independent_query_boundaries() {
        let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        let backing = GuestPhysicalBacking::new(2 * GUEST_PAGE_SIZE).unwrap();
        let base = memory.base();
        for page_index in 0..2 {
            memory
                .map_shared_with_coalescing(
                    GuestMapPlacement::Fixed(page(base, page_index)),
                    GUEST_PAGE_SIZE,
                    GUEST_PAGE_SIZE,
                    GuestProtection::READ_WRITE,
                    &backing,
                    GuestSharedMapOptions {
                        backing_offset: page_index * GUEST_PAGE_SIZE,
                        coalescing: GuestMapCoalescing::Disallowed,
                    },
                )
                .unwrap();
        }
        let vmas = memory.vmas();
        assert_eq!(vmas.len(), 2);
        assert!(vmas.iter().all(|vma| vma.size == GUEST_PAGE_SIZE));
        assert_eq!(memory.query(base).unwrap().start, base);
        assert_eq!(memory.query(page(base, 1)).unwrap().start, page(base, 1));
        memory.unmap(base, 2 * GUEST_PAGE_SIZE).unwrap();
        assert!(memory.vmas().is_empty());
    }

    #[test]
    fn read_write_helpers_enforce_permissions_and_mapping_coverage() {
        let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        let base = memory
            .map_anonymous(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ,
            )
            .unwrap();
        assert!(matches!(
            memory.write(base, b"denied").unwrap_err(),
            AddressSpaceError::PermissionDenied { .. }
        ));
        assert!(matches!(
            memory.read(page(base, 1), &mut [0_u8; 1]).unwrap_err(),
            AddressSpaceError::NotMapped { .. }
        ));
    }

    #[test]
    fn rejects_alignment_bounds_backing_and_overflow_errors() {
        let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        assert!(matches!(
            memory.reserve(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE - 1,
                GUEST_PAGE_SIZE
            ),
            Err(AddressSpaceError::Unaligned { .. })
        ));
        assert!(matches!(
            memory.reserve(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE / 2
            ),
            Err(AddressSpaceError::InvalidAlignment { .. })
        ));
        assert!(matches!(
            memory.map_anonymous(
                GuestMapPlacement::Fixed(GuestAddress::new(!(GUEST_PAGE_SIZE - 1))),
                2 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
            ),
            Err(AddressSpaceError::AddressOverflow { .. })
        ));
        let backing = GuestPhysicalBacking::new(GUEST_PAGE_SIZE).unwrap();
        assert!(matches!(
            memory.map_shared(
                GuestMapPlacement::Anywhere,
                2 * GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            ),
            Err(AddressSpaceError::BackingOutOfBounds { .. })
        ));
    }

    #[test]
    fn unmap_is_atomic_on_unmapped_gap_validation_failure() {
        let mut memory = GuestAddressSpace::new(SPACE_SIZE).unwrap();
        let base = memory
            .map_anonymous(
                GuestMapPlacement::Anywhere,
                GUEST_PAGE_SIZE,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
            )
            .unwrap();
        let before = memory.vmas();
        assert!(matches!(
            memory.unmap(base, 2 * GUEST_PAGE_SIZE),
            Err(AddressSpaceError::NotMapped { .. })
        ));
        assert_eq!(memory.vmas(), before);
        memory.write(base, b"still mapped").unwrap();
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dropping_address_space_releases_exact_range_for_reuse() {
        // Keep this away from host-selected mappings used by concurrently executing tests. Using
        // the address of a dropped anywhere mapping would race another test's mmap allocation.
        let base = GuestAddress::new(0x0000_000a_1000_0000);
        drop(GuestAddressSpace::new_at(base, SPACE_SIZE).unwrap());
        let memory = GuestAddressSpace::new_at(base, SPACE_SIZE).unwrap();
        assert_eq!(memory.base(), base);
    }
}
