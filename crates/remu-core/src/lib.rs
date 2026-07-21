// Licensed under the Apache License, Version 2.0.

use core::fmt;

/// An address in the emulated process address space.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct GuestAddress(u64);

impl GuestAddress {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    #[must_use]
    pub const fn checked_add(self, offset: u64) -> Option<Self> {
        match self.0.checked_add(offset) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
}

impl fmt::Display for GuestAddress {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "0x{:016x}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::GuestAddress;

    #[test]
    fn checked_add_rejects_wraparound() {
        assert_eq!(GuestAddress::new(u64::MAX).checked_add(1), None);
    }
}
