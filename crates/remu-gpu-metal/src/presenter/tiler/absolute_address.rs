// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Q4 absolute-address shader memory model — gpu-metal half of the B2 co-design.
//!
//! Contract (NOTES / MASTER-PLAN Q4): bind one `device uchar*` per memory pool at a fixed
//! buffer slot and pass the pool base VA. Shader-metal lowers V# access to
//! `*(device T*)(pool_base + (va - pool_base))` with NUM_RECORDS folded into one range check.
//!
//! This module owns only the gpu-metal bind surface. Changing shader-metal's binding model
//! without a live pool buffer here would break every existing path — so readiness stays
//! explicit and LOUD until both sides prove a draw.

#![allow(dead_code)] // Bind surface until shader-metal agrees on the slot + lowering.

use metal::{Buffer, Device, MTLResourceOptions};
use remu_memory::GuestSharedBackingRange;

/// Fixed compute/vertex buffer slot reserved for the absolute-address pool (Q4).
/// Shader-metal must agree before any binding-model flip.
pub(crate) const ABSOLUTE_ADDRESS_POOL_BUFFER_SLOT: u64 = 30;

#[derive(Clone, Debug)]
pub(crate) struct AbsoluteAddressPool {
    pub(crate) base_va: u64,
    pub(crate) byte_count: u64,
    /// Metal no-copy buffer over unified guest backing when available.
    pub(crate) buffer: Option<Buffer>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AbsoluteAddressPoolBind {
    pub(crate) base_va: u64,
    pub(crate) byte_count: u64,
    pub(crate) buffer_slot: u64,
}

impl AbsoluteAddressPool {
    pub(crate) fn from_backing(
        device: &Device,
        base_va: u64,
        backing: &GuestSharedBackingRange,
    ) -> Result<Self, String> {
        let byte_count = u64::try_from(backing.view_byte_count())
            .map_err(|_| "absolute-address pool byte count does not fit u64")?;
        if byte_count == 0 {
            return Err(
                "Q4 absolute-address pool: empty backing — refuse fabricated pool bind".to_owned(),
            );
        }
        let buffer = super::super::cache::metal_buffer_with_guest_backing_range(device, backing);
        if buffer.is_none() {
            // PROT_READ-watch precondition (B2): map_view must remain independently RW. A
            // missing no-copy buffer is a loud seam, not a CPU snapshot fallback for Q4.
            return Err(
                "Q4 absolute-address pool: newBufferWithBytesNoCopy unavailable for guest backing (B2 PROT_READ-watch precondition unmet) — refuse silently-wrong pool".to_owned(),
            );
        }
        Ok(Self {
            base_va,
            byte_count,
            buffer,
        })
    }

    /// Host-only / test helper when a real shared backing is unavailable.
    pub(crate) fn host_placeholder(device: &Device, base_va: u64, byte_count: u64) -> Result<Self, String> {
        if byte_count == 0 {
            return Err("Q4 absolute-address pool: zero-sized placeholder refused".to_owned());
        }
        let buffer = device.new_buffer(byte_count, MTLResourceOptions::StorageModeShared);
        buffer.set_label("REMu Q4 absolute-address pool placeholder");
        Ok(Self {
            base_va,
            byte_count,
            buffer: Some(buffer),
        })
    }

    pub(crate) fn bind_descriptor(&self) -> AbsoluteAddressPoolBind {
        AbsoluteAddressPoolBind {
            base_va: self.base_va,
            byte_count: self.byte_count,
            buffer_slot: ABSOLUTE_ADDRESS_POOL_BUFFER_SLOT,
        }
    }
}

pub(crate) fn absolute_address_pool_binding_ready(pool: Option<&AbsoluteAddressPool>) -> bool {
    pool.is_some_and(|pool| pool.buffer.is_some() && pool.byte_count > 0)
}

/// Records a bind for diagnostics. Does not change shader-metal's binding model.
pub(crate) fn record_absolute_address_pool_bind(
    pool: &AbsoluteAddressPool,
) -> Result<AbsoluteAddressPoolBind, String> {
    if !absolute_address_pool_binding_ready(Some(pool)) {
        return Err(
            "Q4 absolute-address: pool not ready — shader-metal must keep per-binding buffers until gpu-metal proves a live device pool".to_owned(),
        );
    }
    Ok(pool.bind_descriptor())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_bind_uses_the_reserved_slot() {
        let Some(device) = Device::system_default() else {
            eprintln!("[REMu][Q4] no Metal device; skipping pool bind test");
            return;
        };
        let pool = AbsoluteAddressPool::host_placeholder(&device, 0x10_0000_0000, 4096)
            .expect("placeholder");
        let bind = record_absolute_address_pool_bind(&pool).expect("bind");
        assert_eq!(bind.buffer_slot, ABSOLUTE_ADDRESS_POOL_BUFFER_SLOT);
        assert_eq!(bind.base_va, 0x10_0000_0000);
        assert!(absolute_address_pool_binding_ready(Some(&pool)));
    }

    #[test]
    fn missing_pool_stays_loud() {
        assert!(!absolute_address_pool_binding_ready(None));
        let err = AbsoluteAddressPool {
            base_va: 0,
            byte_count: 0,
            buffer: None,
        };
        let refused = record_absolute_address_pool_bind(&err).expect_err("empty pool");
        assert!(refused.contains("not ready"), "{refused}");
    }
}
