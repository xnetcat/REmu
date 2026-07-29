// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, RwLock};

use remu_shader::GuestMemoryReader;

use crate::{ProcessGuestMemory, process_guest_range_has_read_access};

static AGC_MEMORY_SHADOW_WRITE_COUNT: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
struct AgcMemoryShadowWrite {
    address: u64,
    bytes: Vec<u8>,
}

#[derive(Default, Debug)]
struct AgcMemoryShadow {
    writes: Vec<AgcMemoryShadowWrite>,
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
pub(crate) fn stage_agc_memory_copy(copy: &remu_gpu::GpuMemoryCopy) -> bool {
    const MAX_SHADOW_COPY_BYTES: u64 = 1024 * 1024;

    if !agc_memory_shadow_is_active()
        || !matches!(copy.destination_selection, 0 | 3)
        || copy.byte_count == 0
        || copy.byte_count > MAX_SHADOW_COPY_BYTES
    {
        return false;
    }
    let Ok(byte_count) = usize::try_from(copy.byte_count) else {
        return false;
    };
    if !process_guest_range_has_read_access(copy.destination_address, byte_count) {
        return false;
    }

    let bytes = match copy.source_selection {
        0 | 3 => {
            let mut bytes = vec![0_u8; byte_count];
            if !ProcessGuestMemory.read(copy.source_address, &mut bytes) {
                return false;
            }
            bytes
        }
        2 if copy.destination_address.is_multiple_of(4) && copy.byte_count.is_multiple_of(4) => {
            let value = (copy.source_address as u32).to_le_bytes();
            let mut bytes = vec![0_u8; byte_count];
            for dword in bytes.chunks_exact_mut(4) {
                dword.copy_from_slice(&value);
            }
            bytes
        }
        _ => return false,
    };

    let mut shadow = agc_memory_shadow()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(shadow) = shadow.as_mut() else {
        return false;
    };
    shadow.writes.push(AgcMemoryShadowWrite {
        address: copy.destination_address,
        bytes,
    });
    AGC_MEMORY_SHADOW_WRITE_COUNT.store(shadow.writes.len(), Ordering::Release);
    true
}
