// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! B1: zero-copy eligibility for linear, page-coverable attachment bindings.
//!
//! Plumbing for `newBufferWithBytesNoCopy` over shared backing already exists
//! (`cache::metal_buffer_with_guest_backing_range`). This module decides when an **attachment**
//! may safely use that path. Tiled / non-page-aligned / unbacked ranges refuse loudly rather
//! than silently snapshotting — silent snapshots are the CPU-timeline round-trip B1 deletes.
//!
//! End-state wiring (owner gate): when `Eligible`, execute/residency may bind the no-copy buffer
//! as the attachment upload source. That bind is not flipped blind here because Private-storage
//! residents + blit uploads (B4) and live title soak must prove aliasing; until then callers use
//! [`refuse_unproven_attachment_zero_copy`] / keep the existing resident path after logging.

use remu_memory::{GUEST_PAGE_SIZE, guest_shared_backing_identity};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::super::GuestTextureDescriptor;

/// Plan-B1 decision for one attachment descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum AttachmentZeroCopyDecision {
    /// Linear tile mode, page-coverable live shared backing — safe to attempt no-copy.
    Eligible {
        address: u64,
        byte_count: u64,
    },
    /// Not proven safe; caller must not invent a snapshot "success".
    Refused {
        reason: &'static str,
        address: u64,
        tile_mode: u32,
    },
}

/// Returns whether `descriptor` is a linear page-coverable attachment candidate for zero-copy.
///
/// `byte_count` is the declared attachment footprint the caller would bind. GFX10 linear color
/// targets use `tile_mode == 0` (`decode/targets.rs`); depth is never linear in this presenter
/// (fixed depth tile mode 0x18) and therefore always refuses.
///
/// # Seam
///
/// Production bind of an [`AttachmentZeroCopyDecision::Eligible`] range onto a no-copy Metal
/// buffer is deferred until Private-storage residents + blit uploads (texture/residency, B4) are
/// soak-proven. Callers should invoke [`refuse_unproven_attachment_zero_copy`] for `Refused` and
/// must not silently snapshot.
#[allow(dead_code)] // B1 seam API — unit-tested; production bind is owner-gate blocked (NOTES).
pub(crate) fn attachment_zero_copy_decision(
    descriptor: GuestTextureDescriptor,
    byte_count: u64,
) -> AttachmentZeroCopyDecision {
    if descriptor.tile_mode != 0 {
        return AttachmentZeroCopyDecision::Refused {
            reason: "attachment tile mode is not linear (tile_mode==0 required for B1 zero-copy)",
            address: descriptor.address,
            tile_mode: descriptor.tile_mode,
        };
    }
    if byte_count == 0 {
        return AttachmentZeroCopyDecision::Refused {
            reason: "attachment byte count is zero",
            address: descriptor.address,
            tile_mode: descriptor.tile_mode,
        };
    }
    if !descriptor.address.is_multiple_of(GUEST_PAGE_SIZE) {
        return AttachmentZeroCopyDecision::Refused {
            reason: "attachment base is not guest-page aligned",
            address: descriptor.address,
            tile_mode: descriptor.tile_mode,
        };
    }
    if !byte_count.is_multiple_of(GUEST_PAGE_SIZE) {
        return AttachmentZeroCopyDecision::Refused {
            reason: "attachment byte count is not guest-page aligned",
            address: descriptor.address,
            tile_mode: descriptor.tile_mode,
        };
    }
    let Some(_identity) = guest_shared_backing_identity(descriptor.address, byte_count) else {
        return AttachmentZeroCopyDecision::Refused {
            reason: "attachment range has no complete live shared-backing identity",
            address: descriptor.address,
            tile_mode: descriptor.tile_mode,
        };
    };
    AttachmentZeroCopyDecision::Eligible {
        address: descriptor.address,
        byte_count,
    }
}

/// Loud refusal helper when a caller would otherwise silently copy attachment bytes.
///
/// Returns `false` after rate-limited diagnostics so the existing resident path can continue
/// without claiming a zero-copy bind.
#[allow(dead_code)] // B1 seam API — unit-tested; production refuse sites land with the bind gate.
pub(crate) fn refuse_unproven_attachment_zero_copy(
    decision: &AttachmentZeroCopyDecision,
) -> bool {
    let AttachmentZeroCopyDecision::Refused {
        reason,
        address,
        tile_mode,
    } = decision
    else {
        return false;
    };
    static REFUSALS: AtomicUsize = AtomicUsize::new(0);
    let sequence = REFUSALS.fetch_add(1, Ordering::Relaxed) + 1;
    if sequence <= 16 || sequence.is_power_of_two() {
        eprintln!(
            "[REMu][METAL][B1] refusing attachment zero-copy sequence={sequence} address={address:#x} tile={tile_mode} reason={reason}"
        );
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::MTLPixelFormat;

    fn linear_descriptor(address: u64) -> GuestTextureDescriptor {
        GuestTextureDescriptor {
            address,
            width: 64,
            height: 64,
            pitch: 64,
            tile_mode: 0,
            pixel_format: MTLPixelFormat::BGRA8Unorm,
            bytes_per_pixel: 4,
        }
    }

    #[test]
    fn tiled_attachment_refuses_zero_copy() {
        let mut descriptor = linear_descriptor(GUEST_PAGE_SIZE);
        descriptor.tile_mode = 27;
        let decision = attachment_zero_copy_decision(descriptor, GUEST_PAGE_SIZE);
        assert!(matches!(
            decision,
            AttachmentZeroCopyDecision::Refused { tile_mode: 27, .. }
        ));
        assert!(refuse_unproven_attachment_zero_copy(&decision));
    }

    #[test]
    fn unaligned_attachment_refuses_zero_copy() {
        let decision =
            attachment_zero_copy_decision(linear_descriptor(GUEST_PAGE_SIZE + 0x100), GUEST_PAGE_SIZE);
        assert!(matches!(
            decision,
            AttachmentZeroCopyDecision::Refused {
                reason: "attachment base is not guest-page aligned",
                ..
            }
        ));
    }

    #[test]
    fn linear_without_shared_backing_refuses_rather_than_snapshotting() {
        // Synthetic address with no installed guest mapping — must refuse, never invent bytes.
        let decision =
            attachment_zero_copy_decision(linear_descriptor(0x7000_0000_0000), GUEST_PAGE_SIZE);
        assert!(matches!(
            decision,
            AttachmentZeroCopyDecision::Refused {
                reason: "attachment range has no complete live shared-backing identity",
                ..
            }
        ));
    }
}
