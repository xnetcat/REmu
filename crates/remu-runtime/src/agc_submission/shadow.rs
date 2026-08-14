// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock};

use remu_shader::GuestMemoryReader;

use crate::{ProcessGuestMemory, process_guest_range_has_read_access};

static AGC_MEMORY_SHADOW_WRITE_COUNT: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
struct AgcMemoryShadowWrite {
    address: u64,
    bytes: Arc<[u8]>,
}

#[derive(Default, Debug)]
struct AgcMemoryShadow {
    writes: Vec<AgcMemoryShadowWrite>,
    retained_bytes: usize,
}

fn agc_memory_shadow() -> &'static RwLock<Option<AgcMemoryShadow>> {
    static SHADOW: OnceLock<RwLock<Option<AgcMemoryShadow>>> = OnceLock::new();
    SHADOW.get_or_init(|| RwLock::new(None))
}

fn agc_memory_shadow_exclusive() -> &'static Mutex<()> {
    static EXCLUSIVE: Mutex<()> = Mutex::new(());
    &EXCLUSIVE
}

pub(crate) struct AgcMemoryShadowScope {
    _exclusive: MutexGuard<'static, ()>,
}

impl Drop for AgcMemoryShadowScope {
    fn drop(&mut self) {
        let mut shadow = agc_memory_shadow()
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *shadow = None;
        AGC_MEMORY_SHADOW_WRITE_COUNT.store(0, Ordering::Release);
    }
}

pub(crate) fn begin_agc_memory_shadow() -> AgcMemoryShadowScope {
    // Production AGC parsing is serialized by the dispatch-state lock. Unit tests can run
    // independent captures concurrently, so preserve that production invariant here too; shader
    // evaluator worker threads still read the active shadow through the separate RwLock below.
    let exclusive = agc_memory_shadow_exclusive()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut shadow = agc_memory_shadow()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert!(
        shadow.is_none(),
        "nested AGC memory shadow would lose command ordering"
    );
    *shadow = Some(AgcMemoryShadow::default());
    AGC_MEMORY_SHADOW_WRITE_COUNT.store(0, Ordering::Release);
    drop(shadow);
    AgcMemoryShadowScope {
        _exclusive: exclusive,
    }
}

pub(crate) fn agc_memory_shadow_is_active() -> bool {
    agc_memory_shadow()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .is_some()
}

pub(crate) fn agc_memory_shadow_overlaps(address: u64, byte_count: usize) -> bool {
    if AGC_MEMORY_SHADOW_WRITE_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    let Ok(byte_count) = u64::try_from(byte_count) else {
        return false;
    };
    let Some(end) = address.checked_add(byte_count) else {
        return false;
    };
    agc_memory_shadow()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .is_some_and(|shadow| {
            shadow.writes.iter().any(|write| {
                u64::try_from(write.bytes.len())
                    .ok()
                    .and_then(|length| write.address.checked_add(length))
                    .is_some_and(|write_end| address < write_end && write.address < end)
            })
        })
}

/// Applies command-processor writes staged earlier in the command stream to a validated guest
/// memory read. The underlying guest allocation remains unchanged until the backend executes the
/// corresponding ordered event.
pub(crate) fn apply_agc_memory_shadow(address: u64, output: &mut [u8]) {
    if AGC_MEMORY_SHADOW_WRITE_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let Ok(output_length) = u64::try_from(output.len()) else {
        return;
    };
    let Some(output_end) = address.checked_add(output_length) else {
        return;
    };
    let shadow = agc_memory_shadow()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(shadow) = shadow.as_ref() else {
        return;
    };
    for write in &shadow.writes {
        let Ok(write_length) = u64::try_from(write.bytes.len()) else {
            continue;
        };
        let Some(write_end) = write.address.checked_add(write_length) else {
            continue;
        };
        let overlap_start = address.max(write.address);
        let overlap_end = output_end.min(write_end);
        if overlap_start >= overlap_end {
            continue;
        }
        let output_start =
            usize::try_from(overlap_start - address).expect("output overlap offset fits usize");
        let write_start = usize::try_from(overlap_start - write.address)
            .expect("write overlap offset fits usize");
        let length =
            usize::try_from(overlap_end - overlap_start).expect("overlap length fits usize");
        output[output_start..output_start + length]
            .copy_from_slice(&write.bytes[write_start..write_start + length]);
    }
}

/// Stages the CPU-readable effect of a DMA packet for command decoding only.
///
/// Draw and dispatch descriptors are decoded synchronously, while the real DMA executes later on
/// the backend queue. A shadow lets following packets observe that copy without changing guest
/// memory early; preceding draw events therefore still see their original resources when the
/// backend consumes them. Large bulk transfers remain backend-only because descriptor and
/// indirect-argument uploads are small and retaining hundreds of MiB while parsing would be
/// counterproductive.
#[derive(Default)]
pub(crate) struct StagedAgcMemoryCopy {
    pub(crate) shadowed: bool,
    pub(crate) source_snapshot: Option<remu_gpu::GpuMemoryCopySnapshot>,
    pub(crate) budget_exceeded: Option<AgcMemorySnapshotBudgetExceeded>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AgcMemorySnapshotBudgetExceeded {
    pub(crate) requested_bytes: usize,
    pub(crate) limit_bytes: usize,
}

pub(crate) fn stage_agc_memory_copy(copy: &remu_gpu::GpuMemoryCopy) -> StagedAgcMemoryCopy {
    static RETAIN_EVENT_SNAPSHOTS: OnceLock<bool> = OnceLock::new();
    let retain_event_snapshots = *RETAIN_EVENT_SNAPSHOTS.get_or_init(|| {
        std::env::var_os("REMU_AGC_DMA_EVENT_SNAPSHOTS").is_none_or(|value| value != "0")
    });
    stage_agc_memory_copy_with_policy(copy, 64 * 1024 * 1024, retain_event_snapshots)
}

fn stage_agc_memory_copy_with_limit(
    copy: &remu_gpu::GpuMemoryCopy,
    max_shadow_total_bytes: usize,
) -> StagedAgcMemoryCopy {
    stage_agc_memory_copy_with_policy(copy, max_shadow_total_bytes, true)
}

fn stage_agc_memory_copy_with_policy(
    copy: &remu_gpu::GpuMemoryCopy,
    max_shadow_total_bytes: usize,
    retain_event_snapshot: bool,
) -> StagedAgcMemoryCopy {
    const MAX_SHADOW_COPY_BYTES: u64 = 1024 * 1024;

    if !agc_memory_shadow_is_active()
        || !matches!(copy.destination_selection, 0 | 3)
        || copy.byte_count == 0
        || copy.byte_count > MAX_SHADOW_COPY_BYTES
    {
        return StagedAgcMemoryCopy::default();
    }
    let Ok(byte_count) = usize::try_from(copy.byte_count) else {
        return StagedAgcMemoryCopy::default();
    };
    if !process_guest_range_has_read_access(copy.destination_address, byte_count) {
        return StagedAgcMemoryCopy::default();
    }
    let requested_bytes = agc_memory_shadow()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .as_ref()
        .map_or(0, |shadow| shadow.retained_bytes)
        .saturating_add(byte_count);
    if requested_bytes > max_shadow_total_bytes {
        return StagedAgcMemoryCopy {
            budget_exceeded: Some(AgcMemorySnapshotBudgetExceeded {
                requested_bytes,
                limit_bytes: max_shadow_total_bytes,
            }),
            ..StagedAgcMemoryCopy::default()
        };
    }

    let (bytes, source_snapshot) = match copy.source_selection {
        0 | 3 => {
            let mut bytes = vec![0_u8; byte_count];
            if !ProcessGuestMemory.read(copy.source_address, &mut bytes) {
                return StagedAgcMemoryCopy::default();
            }
            // DMA packets are decoded before the asynchronous backend consumes them. A source
            // may therefore name bytes written by an earlier DMA in this same command stream,
            // while the underlying guest mapping still contains its old allocation contents.
            // Snapshot the ordered shadow view, not that stale backing allocation.
            apply_agc_memory_shadow(copy.source_address, &mut bytes);
            let bytes = Arc::<[u8]>::from(bytes);
            let source_snapshot = if retain_event_snapshot {
                match remu_gpu::GpuMemoryCopySnapshot::try_new(Arc::clone(&bytes)) {
                    Ok(snapshot) => Some(snapshot),
                    Err(error) => {
                        return StagedAgcMemoryCopy {
                            budget_exceeded: Some(AgcMemorySnapshotBudgetExceeded {
                                requested_bytes: error.requested_bytes,
                                limit_bytes: error.limit_bytes,
                            }),
                            ..StagedAgcMemoryCopy::default()
                        };
                    }
                }
            } else {
                None
            };
            (bytes, source_snapshot)
        }
        2 if copy.destination_address.is_multiple_of(4) && copy.byte_count.is_multiple_of(4) => {
            let value = (copy.source_address as u32).to_le_bytes();
            let mut bytes = vec![0_u8; byte_count];
            for dword in bytes.chunks_exact_mut(4) {
                dword.copy_from_slice(&value);
            }
            (Arc::<[u8]>::from(bytes), None)
        }
        _ => return StagedAgcMemoryCopy::default(),
    };

    let mut shadow = agc_memory_shadow()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(shadow) = shadow.as_mut() else {
        return StagedAgcMemoryCopy::default();
    };
    shadow.writes.push(AgcMemoryShadowWrite {
        address: copy.destination_address,
        bytes,
    });
    shadow.retained_bytes = shadow.retained_bytes.saturating_add(byte_count);
    AGC_MEMORY_SHADOW_WRITE_COUNT.store(shadow.writes.len(), Ordering::Release);
    StagedAgcMemoryCopy {
        shadowed: true,
        source_snapshot,
        budget_exceeded: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_agc_memory_shadow, begin_agc_memory_shadow, stage_agc_memory_copy,
        stage_agc_memory_copy_with_limit, stage_agc_memory_copy_with_policy,
    };

    #[test]
    fn dma_snapshot_retention_has_an_exact_one_mibibyte_boundary() {
        const MIB: usize = 1024 * 1024;
        let source = vec![0x5a_u8; MIB + 1];
        let destination = vec![0_u8; MIB + 1];
        let _shadow = begin_agc_memory_shadow();
        let exact = stage_agc_memory_copy(&remu_gpu::GpuMemoryCopy {
            source_address: source.as_ptr().addr() as u64,
            destination_address: destination.as_ptr().addr() as u64,
            byte_count: MIB as u64,
            source_selection: 3,
            destination_selection: 3,
            source_snapshot: None,
        });
        assert!(exact.shadowed);
        assert_eq!(exact.source_snapshot.as_deref().map(<[u8]>::len), Some(MIB));

        let oversized = stage_agc_memory_copy(&remu_gpu::GpuMemoryCopy {
            source_address: source.as_ptr().addr() as u64,
            destination_address: destination.as_ptr().addr() as u64,
            byte_count: (MIB + 1) as u64,
            source_selection: 3,
            destination_selection: 3,
            source_snapshot: None,
        });
        assert!(!oversized.shadowed);
        assert!(oversized.source_snapshot.is_none());
    }

    #[test]
    fn immediate_dma_shadow_retains_no_event_snapshot() {
        let destination = [0_u32; 4];
        let _shadow = begin_agc_memory_shadow();
        let staged = stage_agc_memory_copy(&remu_gpu::GpuMemoryCopy {
            source_address: 0x1122_3344,
            destination_address: destination.as_ptr().addr() as u64,
            byte_count: std::mem::size_of_val(&destination) as u64,
            source_selection: 2,
            destination_selection: 3,
            source_snapshot: None,
        });
        assert!(staged.shadowed);
        assert!(staged.source_snapshot.is_none());
    }

    #[test]
    fn chained_dma_snapshot_observes_the_prior_shadowed_copy() {
        let source = [0x11_u8, 0x22, 0x33, 0x44];
        let intermediate = [0xcd_u8; 4];
        let destination = [0xcd_u8; 4];
        let _shadow = begin_agc_memory_shadow();

        stage_agc_memory_copy(&remu_gpu::GpuMemoryCopy {
            source_address: source.as_ptr().addr() as u64,
            destination_address: intermediate.as_ptr().addr() as u64,
            byte_count: source.len() as u64,
            source_selection: 3,
            destination_selection: 3,
            source_snapshot: None,
        });
        let chained = stage_agc_memory_copy(&remu_gpu::GpuMemoryCopy {
            source_address: intermediate.as_ptr().addr() as u64,
            destination_address: destination.as_ptr().addr() as u64,
            byte_count: intermediate.len() as u64,
            source_selection: 3,
            destination_selection: 3,
            source_snapshot: None,
        });

        assert_eq!(chained.source_snapshot.as_deref(), Some(source.as_slice()));
        let mut observed = destination;
        apply_agc_memory_shadow(destination.as_ptr().addr() as u64, &mut observed);
        assert_eq!(observed, source);
        assert_eq!(
            intermediate, [0xcd; 4],
            "decode shadow must not mutate guest RAM"
        );
        assert_eq!(
            destination, [0xcd; 4],
            "decode shadow must not mutate guest RAM"
        );
    }

    #[test]
    fn snapshot_budget_exhaustion_preserves_the_ordered_backend_fallback() {
        let source = [0x11_u8, 0x22, 0x33, 0x44];
        let destination = [0xcd_u8; 4];
        let _shadow = begin_agc_memory_shadow();
        let staged = stage_agc_memory_copy_with_limit(
            &remu_gpu::GpuMemoryCopy {
                source_address: source.as_ptr().addr() as u64,
                destination_address: destination.as_ptr().addr() as u64,
                byte_count: source.len() as u64,
                source_selection: 3,
                destination_selection: 3,
                source_snapshot: None,
            },
            0,
        );

        assert!(!staged.shadowed);
        assert!(staged.source_snapshot.is_none());
        assert_eq!(
            staged.budget_exceeded,
            Some(super::AgcMemorySnapshotBudgetExceeded {
                requested_bytes: source.len(),
                limit_bytes: 0,
            })
        );
        assert_eq!(destination, [0xcd; 4]);
        let mut observed = destination;
        apply_agc_memory_shadow(destination.as_ptr().addr() as u64, &mut observed);
        assert_eq!(
            observed, destination,
            "fallback must not invent an early write"
        );
    }

    #[test]
    fn event_snapshot_rollback_keeps_the_decode_order_shadow() {
        let source = [0x19_u8, 0x28, 0x37, 0x46];
        let destination = [0xcd_u8; 4];
        let _shadow = begin_agc_memory_shadow();
        let staged = stage_agc_memory_copy_with_policy(
            &remu_gpu::GpuMemoryCopy {
                source_address: source.as_ptr().addr() as u64,
                destination_address: destination.as_ptr().addr() as u64,
                byte_count: source.len() as u64,
                source_selection: 3,
                destination_selection: 3,
                source_snapshot: None,
            },
            source.len(),
            false,
        );

        assert!(staged.shadowed);
        assert!(staged.source_snapshot.is_none());
        let mut observed = destination;
        apply_agc_memory_shadow(destination.as_ptr().addr() as u64, &mut observed);
        assert_eq!(observed, source);
        assert_eq!(destination, [0xcd; 4]);
    }
}
