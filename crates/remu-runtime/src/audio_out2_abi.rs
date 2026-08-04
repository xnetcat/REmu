// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! `AudioOut2` objects the guest reads and writes directly.
//!
//! Every layout here is the console's, checked against the objects a title hands the library, and
//! the queue bookkeeping is what makes `sceAudioOut2ContextPush` block for one grain the way the
//! hardware mixer does.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::{process_guest_range_is_readable, read_guest_u32, read_guest_u64};

#[derive(Debug, Default)]
pub(crate) struct AudioOut2State {
    pub(crate) next_handle: u64,
    pub(crate) contexts: BTreeMap<u64, AudioOut2ContextState>,
    pub(crate) users: BTreeSet<u64>,
    pub(crate) ports: BTreeMap<u64, AudioOut2PortRuntimeState>,
    pub(crate) mastering_parameters: u64,
    pub(crate) user_creates: u64,
    pub(crate) user_destroys: u64,
    pub(crate) port_creates: u64,
    pub(crate) port_destroys: u64,
    pub(crate) queue_level_queries: u64,
    pub(crate) speaker_queries: u64,
    pub(crate) attribute_updates: u64,
    pub(crate) port_state_queries: u64,
    pub(crate) context_advances: u64,
    pub(crate) context_pushes: u64,
    pub(crate) failed_context_pushes: u64,
    pub(crate) audible_context_pushes: u64,
    #[cfg(test)]
    pub(crate) submitted_pcm: Vec<(u64, Vec<u8>)>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AudioOut2ContextState {
    pub(crate) queue_depth: u32,
    pub(crate) grain_frames: u32,
    pub(crate) queued: u32,
    pub(crate) last_queue_update: Instant,
    /// Device-consumed frames observed at the last queue drain (plan I3).
    pub(crate) last_device_frames: u64,
    pub(crate) memory_address: u64,
    pub(crate) memory_size: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AudioOut2PortRuntimeState {
    pub(crate) context: u64,
    pub(crate) backend_handle: Option<u64>,
    pub(crate) buffer_address: u64,
    pub(crate) channel_count: u8,
    pub(crate) gains: [f32; 8],
    pub(crate) parameter: SceAudioOut2PortParam,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SceAudioOut2ContextParam {
    pub(crate) max_ports: u32,
    pub(crate) max_object_ports: u32,
    pub(crate) guarantee_object_ports: u32,
    pub(crate) queue_depth: u32,
    pub(crate) num_grains: u32,
    pub(crate) flags: u32,
    pub(crate) reserved: [u32; 10],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SceAudioOut2PortParam {
    pub(crate) port_type: u16,
    pub(crate) pad: u16,
    pub(crate) data_format: u32,
    pub(crate) sampling_frequency: u32,
    pub(crate) flags: u32,
    pub(crate) user_handle: u64,
    pub(crate) reserved: [u32; 10],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SceAudioOut2Attribute {
    pub(crate) attribute_id: u32,
    pub(crate) reserved: i32,
    pub(crate) value: u64,
    pub(crate) value_size: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SceAudioOut2Pcm {
    pub(crate) data: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SceAudioOut2PortState {
    pub(crate) output: u16,
    pub(crate) num_channels: u8,
    pub(crate) pad1: u8,
    pub(crate) volume: i16,
    pub(crate) reroute_counter: u16,
    pub(crate) flags: u32,
    pub(crate) pad2: u32,
    pub(crate) reserved: [u64; 6],
}

pub(crate) const _: () = {
    assert!(std::mem::size_of::<SceAudioOut2ContextParam>() == 0x40);
    assert!(std::mem::offset_of!(SceAudioOut2ContextParam, max_ports) == 0x00);
    assert!(std::mem::offset_of!(SceAudioOut2ContextParam, max_object_ports) == 0x04);
    assert!(std::mem::offset_of!(SceAudioOut2ContextParam, guarantee_object_ports) == 0x08);
    assert!(std::mem::offset_of!(SceAudioOut2ContextParam, queue_depth) == 0x0c);
    assert!(std::mem::offset_of!(SceAudioOut2ContextParam, num_grains) == 0x10);
    assert!(std::mem::offset_of!(SceAudioOut2ContextParam, flags) == 0x14);

    assert!(std::mem::size_of::<SceAudioOut2PortParam>() == 0x40);
    assert!(std::mem::offset_of!(SceAudioOut2PortParam, port_type) == 0x00);
    assert!(std::mem::offset_of!(SceAudioOut2PortParam, data_format) == 0x04);
    assert!(std::mem::offset_of!(SceAudioOut2PortParam, sampling_frequency) == 0x08);
    assert!(std::mem::offset_of!(SceAudioOut2PortParam, flags) == 0x0c);
    assert!(std::mem::offset_of!(SceAudioOut2PortParam, user_handle) == 0x10);

    assert!(std::mem::size_of::<SceAudioOut2Attribute>() == 0x18);
    assert!(std::mem::offset_of!(SceAudioOut2Attribute, attribute_id) == 0x00);
    assert!(std::mem::offset_of!(SceAudioOut2Attribute, reserved) == 0x04);
    assert!(std::mem::offset_of!(SceAudioOut2Attribute, value) == 0x08);
    assert!(std::mem::offset_of!(SceAudioOut2Attribute, value_size) == 0x10);

    assert!(std::mem::size_of::<SceAudioOut2Pcm>() == 0x08);
    assert!(std::mem::offset_of!(SceAudioOut2Pcm, data) == 0x00);

    assert!(std::mem::size_of::<SceAudioOut2PortState>() == 0x40);
    assert!(std::mem::offset_of!(SceAudioOut2PortState, output) == 0x00);
    assert!(std::mem::offset_of!(SceAudioOut2PortState, num_channels) == 0x02);
    assert!(std::mem::offset_of!(SceAudioOut2PortState, volume) == 0x04);
    assert!(std::mem::offset_of!(SceAudioOut2PortState, reroute_counter) == 0x06);
    assert!(std::mem::offset_of!(SceAudioOut2PortState, flags) == 0x08);
    assert!(std::mem::offset_of!(SceAudioOut2PortState, reserved) == 0x10);
};

pub(crate) const AUDIO_OUT2_CONTEXT_PARAM_SIZE: u64 =
    std::mem::size_of::<SceAudioOut2ContextParam>() as u64;
// `sceAudioOut2ContextQueryMemory` writes one `size_t`. Silent Hill's Wwise wrapper places that
// qword immediately before its 0x40-byte context parameter and reads it back as the allocation
// size after the call. SharpEmu currently writes a larger private memory-info guess here, but doing
// that overwrites the live parameter in this shipped caller.
pub(crate) const AUDIO_OUT2_CONTEXT_MEMORY_BASE_SIZE: u64 = 0x1_0000;
pub(crate) const AUDIO_OUT2_CONTEXT_MEMORY_QUEUE_STRIDE: u64 = 0x590;
pub(crate) const AUDIO_OUT2_DEFAULT_QUEUE_DEPTH: u32 = 4;
pub(crate) const AUDIO_OUT2_DEFAULT_GRAIN_FRAMES: u32 = 0x200;
pub(crate) const AUDIO_OUT2_MAX_QUEUE_DEPTH: u32 = 64;
/// Bytes one `SceAudioOut2PortState` occupies.
///
/// Silent Hill has two call sites that reserve and consume the full 0x40 bytes. One copies the
/// result into a long-lived Wwise object with two 32-byte vector loads, so leaving the upper half
/// untouched launders stale stack data into live device state.
pub(crate) const AUDIO_OUT2_PORT_STATE_SIZE: u64 =
    std::mem::size_of::<SceAudioOut2PortState>() as u64;
/// Bytes one `SceAudioOut2SpeakerInfo` occupies.
///
/// The Wwise caller reserves exactly 0x40 bytes at `rbp - 0x80`. Clearing the previous 0x50-byte
/// guess overwrote the next sixteen bytes of its stack frame.
pub(crate) const AUDIO_OUT2_SPEAKER_INFO_SIZE: u64 = 0x40;
pub(crate) const AUDIO_OUT2_PORT_TYPE_VIBRATION: u16 = 6;
pub(crate) const SCE_AUDIO_OUT2_ERROR_NOT_READY: u32 = 0x8026_8008;

pub(crate) const fn decode_audio_out2_channel_format(format: u32) -> Option<(u8, u64)> {
    // The AudioOut2 format word carries a data type in bit 0 (0 float, 1 i16), the
    // standard-channel-order attribute in bit 7 (defined only for the eight-channel layouts),
    // and the channel count in bits 8 and up. Each defined combination maps to the ordinary
    // AudioOut format the host mixer consumes; both eight-channel orders keep the center and
    // low-frequency lanes in the same slots, so the stereo downmix treats them alike. The
    // twelve-channel layouts exist in the ABI but stay rejected until the mixer models twelve
    // lanes, and anything outside the defined lattice is invalid input, never a nearest match.
    match format {
        0x100 => Some((1, 3)),
        0x101 => Some((1, 0)),
        0x200 => Some((2, 4)),
        0x201 => Some((2, 1)),
        0x800 => Some((8, 5)),
        0x801 => Some((8, 2)),
        0x880 => Some((8, 7)),
        0x881 => Some((8, 6)),
        _ => None,
    }
}

pub(crate) fn read_audio_out2_context_param(address: u64) -> SceAudioOut2ContextParam {
    // SAFETY: callers validate a complete readable 0x40-byte parameter object first.
    unsafe { std::ptr::read_unaligned(address as *const SceAudioOut2ContextParam) }
}

pub(crate) fn read_audio_out2_port_param(address: u64) -> SceAudioOut2PortParam {
    // SAFETY: callers validate a complete readable 0x40-byte parameter object first.
    unsafe { std::ptr::read_unaligned(address as *const SceAudioOut2PortParam) }
}

pub(crate) fn decode_audio_out2_attribute(address: u64) -> SceAudioOut2Attribute {
    // SAFETY: callers validate the complete attribute array before decoding an entry.
    unsafe { std::ptr::read_unaligned(address as *const SceAudioOut2Attribute) }
}

pub(crate) fn write_audio_out2_context_param(address: u64, parameter: SceAudioOut2ContextParam) {
    // SAFETY: callers validate one complete writable context-parameter object first.
    unsafe {
        std::ptr::write_unaligned(address as *mut SceAudioOut2ContextParam, parameter);
    }
}

pub(crate) fn write_audio_out2_port_state(address: u64, port_state: SceAudioOut2PortState) {
    // SAFETY: callers validate one complete writable port-state object first.
    unsafe {
        std::ptr::write_unaligned(address as *mut SceAudioOut2PortState, port_state);
    }
}

pub(crate) fn audio_out2_grain_duration(grain_frames: u32) -> Duration {
    Duration::from_nanos(u64::from(grain_frames) * 1_000_000_000 / 48_000)
}

pub(crate) fn audio_out2_update_context_queue(context: &mut AudioOut2ContextState, now: Instant) {
    if let Some(device_frames) = crate::audio::device_clock_frames() {
        audio_out2_update_context_queue_from_device(context, device_frames);
        return;
    }
    audio_out2_update_context_queue_from_instant(context, now);
}

fn audio_out2_update_context_queue_from_device(
    context: &mut AudioOut2ContextState,
    device_frames: u64,
) {
    let grain = u64::from(context.grain_frames.max(1));
    if context.queued == 0 {
        // Preserve short producer lateness as catch-up credit against the device clock. A
        // genuinely stopped stream re-anchors after one complete queue depth of idle device time.
        let idle_limit = grain.saturating_mul(u64::from(context.queue_depth.max(1)));
        if device_frames.saturating_sub(context.last_device_frames) > idle_limit {
            context.last_device_frames = device_frames;
        }
        return;
    }
    let elapsed = device_frames.saturating_sub(context.last_device_frames);
    let drained = (elapsed / grain).min(u64::from(context.queued));
    let drained = u32::try_from(drained)
        .unwrap_or(context.queued)
        .min(context.queued);
    if drained == 0 {
        return;
    }
    context.queued -= drained;
    context.last_device_frames = context
        .last_device_frames
        .saturating_add(u64::from(drained).saturating_mul(grain));
}

fn audio_out2_update_context_queue_from_instant(context: &mut AudioOut2ContextState, now: Instant) {
    let grain_duration = audio_out2_grain_duration(context.grain_frames);
    if context.queued == 0 {
        // Preserve a short amount of producer lateness as catch-up credit. Re-anchoring every
        // empty queue to `now` permanently reduces the guest's effective sample rate by its
        // scheduling latency. A genuinely stopped stream is re-anchored after one complete queue
        // depth so it does not return from a long pause with an unbounded burst.
        let idle_limit = grain_duration
            .checked_mul(context.queue_depth.max(1))
            .expect("a bounded AudioOut2 queue duration cannot overflow");
        if now.saturating_duration_since(context.last_queue_update) > idle_limit {
            context.last_queue_update = now;
        }
        return;
    }
    let elapsed = now.saturating_duration_since(context.last_queue_update);
    let drained = u32::try_from(
        (elapsed.as_nanos() / grain_duration.as_nanos()).min(u128::from(context.queued)),
    )
    .expect("the drained AudioOut2 queue count is bounded by u32");
    if drained == 0 {
        return;
    }
    context.queued -= drained;
    context.last_queue_update += grain_duration
        .checked_mul(drained)
        .expect("a bounded AudioOut2 queue duration cannot overflow");
}

pub(crate) fn audio_out2_queue_wait_duration(
    context: &AudioOut2ContextState,
    now: Instant,
) -> Duration {
    let grain_duration = audio_out2_grain_duration(context.grain_frames);
    grain_duration.saturating_sub(now.saturating_duration_since(context.last_queue_update))
}

pub(crate) fn trace_audio_out2_memory() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_AUDIO2_MEMORY").is_some_and(|value| value == "1")
    })
}

pub(crate) fn trace_audio_out2_pcm(
    push: u64,
    audible_push: u64,
    port: u64,
    address: u64,
    channels: u8,
    snapshot: &[u8],
) {
    if !trace_audio_out2_memory()
        || (!crate::trace_logarithmic_sequence(push, 8)
            && !crate::trace_logarithmic_sequence(audible_push, 8))
    {
        return;
    }
    let mut live_samples = 0_u64;
    let mut peak = 0.0_f32;
    let mut sum_squares = 0.0_f64;
    let mut finite_samples = 0_u64;
    for bytes in snapshot.chunks_exact(size_of::<f32>()) {
        let sample = f32::from_le_bytes(bytes.try_into().expect("one f32 sample"));
        if !sample.is_finite() {
            continue;
        }
        finite_samples = finite_samples.saturating_add(1);
        if sample.abs() >= 1.0e-8 {
            live_samples = live_samples.saturating_add(1);
        }
        peak = peak.max(sample.abs());
        sum_squares += f64::from(sample) * f64::from(sample);
    }
    let rms = if finite_samples == 0 {
        0.0
    } else {
        (sum_squares / finite_samples as f64).sqrt()
    };
    eprintln!(
        "[REMu][AUDIO2-PCM] push={push} audible={audible_push} port={port} address={address:#x} channels={channels} bytes={} finite={finite_samples} live={live_samples} peak={peak:.6} rms={rms:.6}",
        snapshot.len()
    );
}

pub(crate) fn trace_audio_out2_context_memory(
    context: AudioOut2ContextState,
    push: u64,
    channels: &[u8],
) {
    const PAGE_SIZE: u64 = 0x1000;
    const MAX_POINTERS: usize = 24;
    let Ok(memory_size) = usize::try_from(context.memory_size) else {
        return;
    };
    if memory_size == 0
        || memory_size > 0x20_0000
        || !process_guest_range_is_readable(context.memory_address, memory_size)
    {
        eprintln!(
            "[REMu][AUDIO2-MEM] push={push} unreadable context={:#x}+{:#x}",
            context.memory_address, context.memory_size
        );
        return;
    }

    let mut pages = Vec::new();
    let mut pointers = Vec::new();
    let mut page_offset = 0;
    while page_offset < context.memory_size {
        let page_bytes = (context.memory_size - page_offset).min(PAGE_SIZE);
        let qwords = page_bytes / 8;
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        let mut nonzero = 0_u64;
        let mut float_samples = 0_u64;
        let mut live_float_samples = 0_u64;
        for index in 0..qwords {
            let offset = page_offset + index * 8;
            let value = read_guest_u64(context.memory_address + offset);
            hash ^= value;
            hash = hash.wrapping_mul(0x1000_0000_01b3);
            nonzero += u64::from(value != 0);
            for bits in [value as u32, (value >> 32) as u32] {
                let sample = f32::from_bits(bits);
                if sample.is_finite() && sample.abs() <= 2.0 {
                    float_samples += 1;
                    live_float_samples += u64::from(sample.abs() >= f32::MIN_POSITIVE);
                }
            }
            if pointers.len() < MAX_POINTERS
                && value & 3 == 0
                && value != context.memory_address
                && process_guest_range_is_readable(value, 128)
                && !pointers
                    .iter()
                    .any(|(_, candidate): &(u64, u64)| *candidate == value)
            {
                pointers.push((offset, value));
            }
        }
        if nonzero != 0 {
            pages.push(format!(
                "{page_offset:#x}:nz={nonzero}/{qwords},f={live_float_samples}/{float_samples},h={hash:016x}"
            ));
        }
        page_offset += page_bytes;
    }

    let pointers = pointers
        .into_iter()
        .map(|(offset, address)| {
            let mut live_float_samples = 0_u32;
            let mut peak = 0.0_f32;
            for index in 0..16_u64 {
                let value = read_guest_u64(address + index * 8);
                for bits in [value as u32, (value >> 32) as u32] {
                    let sample = f32::from_bits(bits);
                    if sample.is_finite() && sample.abs() <= 2.0 {
                        live_float_samples += u32::from(sample.abs() >= f32::MIN_POSITIVE);
                        peak = peak.max(sample.abs());
                    }
                }
            }
            format!("{offset:#x}->{address:#x}(live-f32={live_float_samples}/32,peak={peak:.3})")
        })
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "[REMu][AUDIO2-MEM] push={push} context={:#x}+{:#x} channels={channels:?} pages=[{}] pointers=[{pointers}]",
        context.memory_address,
        context.memory_size,
        pages.join(","),
    );
}

pub(crate) fn trace_audio_out2_mastering_memory(parameters: u64, push: u64) {
    const MASTERING_OBJECT_PREFIX: u64 = 0x5b8;
    const MASTERING_OBJECT_SIZE: usize = 0x800;
    const TARGET_SAMPLE_SIZES: [usize; 3] = [0x4000, 0x2000, 0x1000];
    const MAX_TARGETS: usize = 48;
    const SECOND_LEVEL_SCAN_SIZES: [usize; 3] = [0x1_0000, 0x8000, 0x4000];
    const MAX_SECOND_LEVEL_TARGETS: usize = 64;

    let Some(object) = parameters.checked_sub(MASTERING_OBJECT_PREFIX) else {
        return;
    };
    if parameters == 0 || !process_guest_range_is_readable(object, MASTERING_OBJECT_SIZE) {
        eprintln!(
            "[REMu][AUDIO2-MASTER] push={push} unreadable parameters={parameters:#x} object={object:#x}"
        );
        return;
    }

    let mut targets = Vec::new();
    for offset in (0..MASTERING_OBJECT_SIZE).step_by(size_of::<u64>()) {
        let address = read_guest_u64(object + offset as u64);
        let sample_bytes = TARGET_SAMPLE_SIZES
            .into_iter()
            .find(|sample_bytes| process_guest_range_is_readable(address, *sample_bytes));
        if address & 3 != 0
            || sample_bytes.is_none()
            || targets
                .iter()
                .any(|(_, candidate, _, _, _, _)| *candidate == address)
        {
            continue;
        }
        let sample_bytes = sample_bytes.unwrap();
        let mut live_samples = 0_u32;
        let mut peak = 0.0_f32;
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for index in 0..sample_bytes / size_of::<u32>() {
            let bits = read_guest_u32(address + (index * size_of::<u32>()) as u64);
            hash ^= u64::from(bits);
            hash = hash.wrapping_mul(0x1000_0000_01b3);
            let sample = f32::from_bits(bits);
            if sample.is_finite() && (1.0e-8..=8.0).contains(&sample.abs()) {
                live_samples = live_samples.saturating_add(1);
                peak = peak.max(sample.abs());
            }
        }
        targets.push((offset, address, sample_bytes, live_samples, peak, hash));
        if targets.len() == MAX_TARGETS {
            break;
        }
    }
    let formatted_targets = targets
        .iter()
        .map(|(offset, address, sample_bytes, live_samples, peak, hash)| {
            format!(
                "{offset:#x}->{address:#x}(bytes={sample_bytes:#x},live-f32={live_samples}/{},peak={peak:.3},h={hash:016x})",
                sample_bytes / size_of::<u32>(),
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "[REMu][AUDIO2-MASTER] push={push} parameters={parameters:#x} object={object:#x} targets=[{formatted_targets}]"
    );

    let mut second_level_targets = Vec::new();
    for (parent_offset, parent, _, _, _, _) in &targets {
        let Some(scan_bytes) = SECOND_LEVEL_SCAN_SIZES
            .into_iter()
            .find(|scan_bytes| process_guest_range_is_readable(*parent, *scan_bytes))
        else {
            continue;
        };
        for child_offset in (0..scan_bytes).step_by(size_of::<u64>()) {
            let address = read_guest_u64(*parent + child_offset as u64);
            let sample_bytes = TARGET_SAMPLE_SIZES
                .into_iter()
                .find(|sample_bytes| process_guest_range_is_readable(address, *sample_bytes));
            if address & 3 != 0
                || address == object
                || address == *parent
                || sample_bytes.is_none()
                || second_level_targets
                    .iter()
                    .any(|(_, _, candidate, _, _, _, _)| *candidate == address)
            {
                continue;
            }
            let sample_bytes = sample_bytes.unwrap();
            let mut live_samples = 0_u32;
            let mut peak = 0.0_f32;
            let mut hash = 0xcbf2_9ce4_8422_2325_u64;
            for index in 0..sample_bytes / size_of::<u32>() {
                let bits = read_guest_u32(address + (index * size_of::<u32>()) as u64);
                hash ^= u64::from(bits);
                hash = hash.wrapping_mul(0x1000_0000_01b3);
                let sample = f32::from_bits(bits);
                if sample.is_finite() && (1.0e-8..=8.0).contains(&sample.abs()) {
                    live_samples = live_samples.saturating_add(1);
                    peak = peak.max(sample.abs());
                }
            }
            second_level_targets.push((
                *parent_offset,
                child_offset,
                address,
                sample_bytes,
                live_samples,
                peak,
                hash,
            ));
        }
    }
    second_level_targets
        .sort_unstable_by_key(|(_, _, _, _, live_samples, _, _)| std::cmp::Reverse(*live_samples));
    second_level_targets.truncate(MAX_SECOND_LEVEL_TARGETS);
    let second_level_targets = second_level_targets
        .into_iter()
        .map(
            |(parent_offset, child_offset, address, sample_bytes, live_samples, peak, hash)| {
                format!(
                    "{parent_offset:#x}+{child_offset:#x}->{address:#x}(bytes={sample_bytes:#x},live-f32={live_samples}/{},peak={peak:.3},h={hash:016x})",
                    sample_bytes / size_of::<u32>(),
                )
            },
        )
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "[REMu][AUDIO2-MASTER2] push={push} parameters={parameters:#x} object={object:#x} targets=[{second_level_targets}]"
    );
}

impl AudioOut2State {
    pub(crate) fn allocate_handle(&mut self) -> u64 {
        let handle = self.next_handle.max(1);
        self.next_handle = handle.saturating_add(1);
        handle
    }
}
