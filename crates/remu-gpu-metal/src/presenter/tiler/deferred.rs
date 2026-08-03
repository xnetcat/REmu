// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! B3 deferred in-batch readbacks: encoding records work; it never commits for the value.
//!
//! Seam: production `PendingTextureReadback::encode_*` still commits today. Callers that
//! need deferred semantics must go through [`DeferredGuestReadTracker`]. Resolving a
//! continuation into a guest-queue block is wired via `QueueBlockReason::GuestReadback`;
//! encoding the blit into an open guest batch (instead of a private commit) is the next
//! ownership step for the texture/readback modules — left LOUD until proven.

#![allow(dead_code)] // Scaffolding surface until open-batch blit encode owns the path.

use crate::presenter::GuestTextureDescriptor;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DeferredGuestReadId(u64);

impl DeferredGuestReadId {
    pub(crate) fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DeferredGuestReadReason {
    /// Guest-observable read of a Metal-private resident (flip / DMA / sampled recycle).
    GuestObservable,
    /// Coherence layer needs texels before a later encode step in the same batch.
    InBatchCoherence,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DeferredGuestRead {
    pub(crate) id: DeferredGuestReadId,
    pub(crate) descriptor: GuestTextureDescriptor,
    pub(crate) reason: DeferredGuestReadReason,
    /// Guest queue that must not advance past this read until the continuation lands.
    pub(crate) needed_by_queue: Option<u64>,
    pub(crate) mip_level: u32,
    pub(crate) slice: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct GuestReadbackContinuation {
    pub(crate) id: DeferredGuestReadId,
    /// Staging bytes become available only after the batch that encoded the blit retires.
    pub(crate) pending: bool,
    pub(crate) bytes: Option<Vec<u8>>,
}

#[derive(Default)]
pub(crate) struct DeferredGuestReadTracker {
    next_id: AtomicU64,
    pending: HashMap<DeferredGuestReadId, DeferredGuestRead>,
    continuations: HashMap<DeferredGuestReadId, GuestReadbackContinuation>,
}

impl DeferredGuestReadTracker {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Records a deferred read. Encoding must not commit; the coherence layer resolves later.
    pub(crate) fn record(
        &mut self,
        descriptor: GuestTextureDescriptor,
        reason: DeferredGuestReadReason,
        needed_by_queue: Option<u64>,
        mip_level: u32,
        slice: u32,
    ) -> DeferredGuestReadId {
        let id = DeferredGuestReadId(self.next_id.fetch_add(1, Ordering::Relaxed) + 1);
        self.pending.insert(
            id,
            DeferredGuestRead {
                id,
                descriptor,
                reason,
                needed_by_queue,
                mip_level,
                slice,
            },
        );
        self.continuations.insert(
            id,
            GuestReadbackContinuation {
                id,
                pending: true,
                bytes: None,
            },
        );
        id
    }

    pub(crate) fn take_pending(&mut self, id: DeferredGuestReadId) -> Option<DeferredGuestRead> {
        self.pending.remove(&id)
    }

    pub(crate) fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Completes a continuation after the batch-local blit retires. Until the open-batch
    /// encode path lands, callers must treat unresolved ids as LOUD blocks — never invent bytes.
    pub(crate) fn complete_with_bytes(
        &mut self,
        id: DeferredGuestReadId,
        bytes: Vec<u8>,
    ) -> Result<(), String> {
        let continuation = self.continuations.get_mut(&id).ok_or_else(|| {
            format!(
                "DeferredGuestRead {}: no continuation (encoding must not invent readback bytes)",
                id.get()
            )
        })?;
        if !continuation.pending {
            return Err(format!(
                "DeferredGuestRead {}: continuation already completed",
                id.get()
            ));
        }
        continuation.pending = false;
        continuation.bytes = Some(bytes);
        self.pending.remove(&id);
        Ok(())
    }

    pub(crate) fn continuation(
        &self,
        id: DeferredGuestReadId,
    ) -> Option<&GuestReadbackContinuation> {
        self.continuations.get(&id)
    }

    /// LOUD fallback: consuming unresolved bytes is a programming error, not a silent zero fill.
    pub(crate) fn require_bytes(&self, id: DeferredGuestReadId) -> Result<&[u8], String> {
        let continuation = self.continuation(id).ok_or_else(|| {
            format!(
                "DeferredGuestRead {}: unknown id — refuse fabricated readback",
                id.get()
            )
        })?;
        if continuation.pending {
            return Err(format!(
                "DeferredGuestRead {}: still pending — encoding must not commit; wait for continuation (B3 seam: open-batch blit encode not yet owned by texture/readback)",
                id.get()
            ));
        }
        continuation.bytes.as_deref().ok_or_else(|| {
            format!(
                "DeferredGuestRead {}: completed without bytes — refuse fabricated success",
                id.get()
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::MTLPixelFormat;

    fn descriptor() -> GuestTextureDescriptor {
        GuestTextureDescriptor {
            address: 0x70_0000_1000,
            width: 64,
            height: 64,
            pitch: 64,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        }
    }

    #[test]
    fn recording_a_deferred_read_does_not_invent_bytes() {
        let mut tracker = DeferredGuestReadTracker::new();
        let id = tracker.record(
            descriptor(),
            DeferredGuestReadReason::GuestObservable,
            Some(7),
            0,
            0,
        );
        assert_eq!(tracker.pending_count(), 1);
        let err = tracker
            .require_bytes(id)
            .expect_err("unresolved read must stay loud");
        assert!(err.contains("still pending"), "{err}");
        assert!(err.contains("must not commit"), "{err}");
    }

    #[test]
    fn completing_a_continuation_publishes_bytes_exactly_once() {
        let mut tracker = DeferredGuestReadTracker::new();
        let id = tracker.record(
            descriptor(),
            DeferredGuestReadReason::InBatchCoherence,
            None,
            0,
            0,
        );
        tracker
            .complete_with_bytes(id, vec![1, 2, 3, 4])
            .expect("complete");
        assert_eq!(tracker.require_bytes(id).expect("bytes"), &[1, 2, 3, 4]);
        assert_eq!(tracker.pending_count(), 0);
        let err = tracker
            .complete_with_bytes(id, vec![9])
            .expect_err("double complete");
        assert!(err.contains("already completed"), "{err}");
    }
}
