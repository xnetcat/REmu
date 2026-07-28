// SPDX-License-Identifier: GPL-2.0-or-later

//! Guest page protection, mapping placement options and VMA query descriptions.

use std::fmt;
use std::ops::BitOr;

use remu_core::GuestAddress;

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
        if flags & crate::GUEST_READ != 0 {
            protection = protection.union(Self::READ);
        }
        if flags & crate::GUEST_WRITE != 0 {
            protection = protection.union(Self::WRITE);
        }
        if flags & crate::GUEST_EXECUTE != 0 {
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

    pub(super) const fn host_flags(self) -> i32 {
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
    /// Committed guest memory whose only physical identity is its virtual placement.
    ///
    /// Reserve-then-commit allocations never expose a physical address to the guest, so unlike
    /// [`Self::Shared`] direct-memory views they cannot be aliased at another VA. They still
    /// carry a shared physical backing so GPU residency and write-watching resolve them.
    Committed,
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
