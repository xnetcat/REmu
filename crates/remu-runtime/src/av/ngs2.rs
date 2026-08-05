// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::{
    guest_memset, low_u32, process_guest_range_is_readable, read_guest_u32, read_guest_u64,
    sce_kernel_error, write_guest_u32, write_guest_u64,
};
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use super::{GUEST_HANDLE_SIZE, GUEST_STATUS_WORD_SIZE, guest_object_is_reachable};

/// `NGS2` error codes this module reports.
pub(super) const NGS2_ERROR_INVALID_OUT_ADDRESS: u32 = 0x804a_8010;
pub(super) const NGS2_ERROR_INVALID_OPTION_ADDRESS: u32 = 0x804a_8012;
pub(super) const NGS2_ERROR_INVALID_OPTION_SIZE: u32 = 0x804a_8013;
pub(super) const NGS2_ERROR_INVALID_CONTEXT_ALLOCATOR: u32 = 0x804a_8023;
pub(super) const NGS2_ERROR_INVALID_PARAM_ADDRESS: u32 = 0x804a_8030;
pub(super) const NGS2_ERROR_INVALID_HANDLER_ADDRESS: u32 = 0x804a_8040;
pub(super) const NGS2_ERROR_INVALID_WAVEFORM_TYPE: u32 = 0x804a_8050;
pub(super) const NGS2_ERROR_INVALID_SAMPLE_RATE: u32 = 0x804a_8051;
pub(super) const NGS2_ERROR_INVALID_NUM_CHANNELS: u32 = 0x804a_8052;
pub(super) const NGS2_ERROR_INVALID_WAVEFORM_ADDRESS: u32 = 0x804a_8055;
pub(super) const NGS2_ERROR_INVALID_SYSTEM_HANDLE: u32 = 0x804a_8201;
pub(super) const NGS2_ERROR_INVALID_RACK_HANDLE: u32 = 0x804a_8202;
pub(super) const NGS2_ERROR_INVALID_VOICE_HANDLE: u32 = 0x804a_8203;
pub(super) const NGS2_ERROR_INVALID_HANDLE: u32 = 0x804a_8205;
pub(super) const NGS2_ERROR_INVALID_MODULE_INDEX: u32 = 0x804a_8256;
pub(super) const NGS2_ERROR_INVALID_RACK_ID: u32 = 0x804a_8300;
pub(super) const NGS2_ERROR_INVALID_STATE_SIZE: u32 = 0x804a_8302;
pub(super) const NGS2_ERROR_INVALID_VOICE_INDEX: u32 = 0x804a_8305;
pub(super) const NGS2_ERROR_INVALID_MODULE_LOCATION: u32 = 0x804a_8308;
pub(super) const NGS2_ERROR_INVALID_NUM_VOICES: u32 = 0x804a_8309;
pub(super) const NGS2_ERROR_INVALID_NUM_OUTPUTS: u32 = 0x804a_830e;
pub(super) const NGS2_ERROR_INVALID_NUM_MATRICES: u32 = 0x804a_830f;
pub(super) const NGS2_ERROR_INVALID_NUM_GRAIN_SAMPLES: u32 = 0x804a_8310;
pub(super) const NGS2_ERROR_INVALID_MAX_GRAIN_SAMPLES: u32 = 0x804a_8311;
pub(super) const NGS2_ERROR_INVALID_MAX_OUTPUT_CHANNELS: u32 = 0x804a_8312;
pub(super) const NGS2_ERROR_INVALID_MAX_VOICE_CHANNELS: u32 = 0x804a_8313;
pub(super) const NGS2_ERROR_INVALID_VOICE_SETUP_FLAG: u32 = 0x804a_8352;
pub(super) const NGS2_ERROR_INVALID_VOICE_EVENT_TYPE: u32 = 0x804a_8353;
pub(super) const NGS2_ERROR_INVALID_VOICE_MATRIX_INDEX: u32 = 0x804a_8355;
pub(super) const NGS2_ERROR_INVALID_VOICE_OUTPUT_INDEX: u32 = 0x804a_8357;
pub(super) const NGS2_ERROR_INVALID_VOICE_PATCH_TYPE: u32 = 0x804a_8359;
pub(super) const NGS2_ERROR_INVALID_VOICE_PARAM_ID: u32 = 0x804a_835a;
pub(super) const NGS2_ERROR_INVALID_MATRIX_LEVEL_ADDRESS: u32 = 0x804a_835e;
pub(super) const NGS2_ERROR_INVALID_VOICE_CONTROL_ADDRESS: u32 = 0x804a_8360;
pub(super) const NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE: u32 = 0x804a_8361;
#[allow(dead_code)] // SDK error; returned when a control id enum is validated end-to-end.
pub(super) const NGS2_ERROR_INVALID_VOICE_CONTROL_ID: u32 = 0x804a_8362;
pub(super) const NGS2_ERROR_INVALID_WAVEFORM_DATA: u32 = 0x804a_8430;
pub(super) const NGS2_ERROR_INVALID_WAVEFORM_FORMAT: u32 = 0x804a_8431;
pub(super) const NGS2_ERROR_UNKNOWN_WAVEFORM_FORMAT: u32 = 0x804a_8432;

/// Bytes one `SceNgs2SystemOption` occupies, and the least it may declare.
///
/// A caller built against an older revision may declare a shorter option as long as it still
/// carries every field this build reads.
pub(super) const NGS2_SYSTEM_OPTION_SIZE: u64 = 0x90;
pub(super) const NGS2_SYSTEM_OPTION_MINIMUM_SIZE: u64 = 0x7c;
/// Byte offsets within that option.
pub(super) const NGS2_SYSTEM_OPTION_FLAGS_OFFSET: u64 = 104;
pub(super) const NGS2_SYSTEM_OPTION_MAX_GRAIN_SAMPLES_OFFSET: u64 = 108;
pub(super) const NGS2_SYSTEM_OPTION_NUM_GRAIN_SAMPLES_OFFSET: u64 = 112;
pub(super) const NGS2_SYSTEM_OPTION_SAMPLE_RATE_OFFSET: u64 = 116;
pub(super) const NGS2_SYSTEM_OPTION_MAX_VOICE_CHANNELS_OFFSET: u64 = 120;

/// Bytes one `SceNgs2RackOption` occupies in the SDK revision this build was written against.
///
/// Newer SDKs may declare a larger option; [`ngs2_rack_option_size`] accepts any size at least
/// [`NGS2_RACK_OPTION_MINIMUM_SIZE`] and only reads the field offsets defined here.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) const NGS2_RACK_OPTION_SIZE: u64 = 176;
pub(super) const NGS2_RACK_OPTION_MINIMUM_SIZE: u64 = 112;
/// Upper bound for a forward-compatible rack option. Large enough for observed PS5 SDK growth
/// (Unity/FMOD has declared sizes up to 0x508).
pub(super) const NGS2_RACK_OPTION_MAXIMUM_SIZE: u64 = 0x800;
/// Byte offsets within that option.
pub(super) const NGS2_RACK_OPTION_FLAGS_OFFSET: u64 = 72;
pub(super) const NGS2_RACK_OPTION_MAX_GRAIN_SAMPLES_OFFSET: u64 = 76;
pub(super) const NGS2_RACK_OPTION_MAX_VOICES_OFFSET: u64 = 80;
pub(super) const NGS2_RACK_OPTION_MAX_MATRICES_OFFSET: u64 = 88;
pub(super) const NGS2_RACK_OPTION_MAX_PORTS_OFFSET: u64 = 92;
pub(super) const NGS2_RACK_OPTION_MAX_VOICE_CHANNELS_OFFSET: u64 = 96;
pub(super) const NGS2_RACK_OPTION_MAX_OUTPUT_CHANNELS_OFFSET: u64 = 100;

/// Bytes one `SceNgs2ContextBufferInfo` occupies, and the offset of its user data.
pub(super) const NGS2_CONTEXT_BUFFER_INFO_SIZE: u64 = 64;
pub(super) const NGS2_CONTEXT_BUFFER_INFO_USER_DATA_OFFSET: u64 = 56;
/// Bytes one `SceNgs2BufferAllocator` occupies.
pub(super) const NGS2_BUFFER_ALLOCATOR_SIZE: usize = 24;
pub(super) const NGS2_BUFFER_ALLOCATOR_FREE_HANDLER_OFFSET: u64 = 8;
pub(super) const NGS2_BUFFER_ALLOCATOR_USER_DATA_OFFSET: u64 = 16;

/// Bytes one `SceNgs2VoiceState` occupies.
pub(super) const NGS2_VOICE_STATE_SIZE: u64 = 8;

/// Grain, channel and rack limits the library enforces.
pub(super) const NGS2_MIN_GRAIN_SAMPLES: u32 = 64;
pub(super) const NGS2_MAX_GRAIN_SAMPLES: u32 = 2048;
pub(super) const NGS2_GRAIN_SAMPLE_STEP: u32 = 64;
pub(super) const NGS2_MAX_VOICE_CHANNELS: u32 = 37;
pub(super) const NGS2_MIN_SAMPLE_RATE: u32 = 11_025;
pub(super) const NGS2_MAX_SAMPLE_RATE: u32 = 192_000;
pub(super) const NGS2_RACK_MAX_VOICES: u32 = 32_768;
pub(super) const NGS2_RACK_MAX_PORTS: u32 = 32;
pub(super) const NGS2_RACK_MAX_MATRICES: u32 = 32;
pub(super) const NGS2_RACK_MAX_MODULES: u32 = 32;
pub(super) const NGS2_MODULE_MAX_LOCATIONS: u32 = 34;

/// Rack identifiers the library recognizes.
pub(super) const NGS2_RACK_IDS: [u32; 9] = [
    0x1000, 0x2000, 0x2001, 0x3000, 0x4000, 0x4001, 0x4002, 0x4003, 0x4004,
];

/// Voice state flags `sceNgs2VoiceGetStateFlags` reports.
pub(super) const NGS2_VOICE_STATE_FLAG_IN_USE: u32 = 1 << 0;
pub(super) const NGS2_VOICE_STATE_FLAG_PLAYING: u32 = 1 << 1;
pub(super) const NGS2_VOICE_STATE_FLAG_PAUSED: u32 = 1 << 2;
pub(super) const NGS2_VOICE_STATE_FLAG_STOPPED: u32 = 1 << 3;

/// Voice events a control entry or a command may kick.
pub(super) const NGS2_VOICE_EVENT_PLAY: u32 = 0x0001;
pub(super) const NGS2_VOICE_EVENT_STOP: u32 = 0x0002;
pub(super) const NGS2_VOICE_EVENT_STOP_IMMEDIATE: u32 = 0x0004;
pub(super) const NGS2_VOICE_EVENT_KILL: u32 = 0x0008;
pub(super) const NGS2_VOICE_EVENT_PAUSE: u32 = 0x0010;
pub(super) const NGS2_VOICE_EVENT_RESUME: u32 = 0x0020;

/// First handle the object table hands out.
///
/// The value sits far outside anything the guest maps, so a title that mistakes a handle for a
/// pointer faults at once instead of writing over live state.
pub(super) const NGS2_HANDLE_BASE: u64 = 0x4e47_5332_0000;
/// Distance between consecutive handles.
pub(super) const NGS2_HANDLE_STRIDE: u64 = 0x40;

/// One system object and the option it was created from.
#[derive(Debug)]
pub(super) struct Ngs2System {
    pub(super) sample_rate: u32,
    pub(super) num_grain_samples: u32,
    pub(super) max_grain_samples: u32,
    pub(super) max_voice_channels: u32,
    pub(super) flags: u32,
    pub(super) allocator_user_data: u64,
    pub(super) racks: Vec<u64>,
}

/// One rack object, its sizing, and the voices a title has resolved from it.
#[derive(Debug)]
pub(super) struct Ngs2Rack {
    pub(super) system: u64,
    pub(super) rack_id: u32,
    pub(super) max_voices: u32,
    pub(super) max_ports: u32,
    pub(super) max_matrices: u32,
    pub(super) max_voice_channels: u32,
    pub(super) max_output_channels: u32,
    pub(super) allocator_user_data: u64,
    pub(super) voices: BTreeMap<u32, u64>,
}

/// One voice, and everything a title can observe about it without a render engine.
#[derive(Debug, Default)]
pub(super) struct Ngs2Voice {
    pub(super) rack: u64,
    pub(super) state_flags: u32,
    pub(super) error_code: i32,
    pub(super) user_data: u64,
    pub(super) num_channels: u32,
    pub(super) setup_flags: u32,
    pub(super) callback_handler: u64,
}

/// Every live `NGS2` object, keyed by the handle the guest holds.
#[derive(Debug, Default)]
pub(super) struct Ngs2State {
    pub(super) next_handle: u64,
    pub(super) systems: BTreeMap<u64, Ngs2System>,
    pub(super) racks: BTreeMap<u64, Ngs2Rack>,
    pub(super) voices: BTreeMap<u64, Ngs2Voice>,
}

impl Ngs2State {
    /// Returns a handle no live object holds.
    pub(super) fn allocate_handle(&mut self) -> u64 {
        let index = self.next_handle;
        self.next_handle = index.saturating_add(1);
        NGS2_HANDLE_BASE.saturating_add(index.saturating_mul(NGS2_HANDLE_STRIDE))
    }

    /// Discards a rack and every voice resolved from it.
    pub(super) fn remove_rack(&mut self, rack_handle: u64) -> Option<Ngs2Rack> {
        let rack = self.racks.remove(&rack_handle)?;
        for voice in rack.voices.values() {
            self.voices.remove(voice);
        }
        Some(rack)
    }
}

/// Returns the process-wide `NGS2` object table.
pub(super) fn ngs2_state() -> std::sync::MutexGuard<'static, Ngs2State> {
    static STATE: OnceLock<Mutex<Ngs2State>> = OnceLock::new();
    STATE
        .get_or_init(|| Mutex::new(Ngs2State::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Reports the size a caller declared for an option block, or the error it violates.
pub(super) fn ngs2_option_size(option: u64, minimum: u64, maximum: u64) -> Result<u64, u32> {
    if !process_guest_range_is_readable(option, std::mem::size_of::<u64>()) {
        return Err(NGS2_ERROR_INVALID_OPTION_ADDRESS);
    }
    let size = read_guest_u64(option);
    if !(minimum..=maximum).contains(&size)
        || !process_guest_range_is_readable(option, usize::try_from(size).unwrap_or(usize::MAX))
    {
        eprintln!(
            "[REMu][NGS2] invalid option size={size} expected={minimum}..={maximum} address={option:#x}"
        );
        return Err(NGS2_ERROR_INVALID_OPTION_SIZE);
    }
    Ok(size)
}

/// Accepts a rack option whose size is at least the fields this build reads.
///
/// SDK revisions grow the option with trailing reserved fields; rejecting anything larger than
/// the revision this file was written against breaks titles that ship a newer header. Some
/// Unity/FMOD builds also write the size as a 32-bit field; when the upper half is non-zero
/// garbage from an adjacent name, the low 32 bits are accepted when they fall in range.
pub(super) fn ngs2_rack_option_size(option: u64) -> Result<u64, u32> {
    if !process_guest_range_is_readable(option, std::mem::size_of::<u64>()) {
        return Err(NGS2_ERROR_INVALID_OPTION_ADDRESS);
    }
    let raw = read_guest_u64(option);
    let size = if (NGS2_RACK_OPTION_MINIMUM_SIZE..=NGS2_RACK_OPTION_MAXIMUM_SIZE).contains(&raw) {
        raw
    } else {
        let low = raw & 0xffff_ffff;
        if (NGS2_RACK_OPTION_MINIMUM_SIZE..=NGS2_RACK_OPTION_MAXIMUM_SIZE).contains(&low) {
            low
        } else {
            eprintln!(
                "[REMu][NGS2] invalid rack option size raw={raw:#x} expected={NGS2_RACK_OPTION_MINIMUM_SIZE}..={NGS2_RACK_OPTION_MAXIMUM_SIZE} address={option:#x}"
            );
            return Err(NGS2_ERROR_INVALID_OPTION_SIZE);
        }
    };
    if !process_guest_range_is_readable(option, usize::try_from(size).unwrap_or(usize::MAX)) {
        return Err(NGS2_ERROR_INVALID_OPTION_SIZE);
    }
    Ok(size)
}

/// Renders one grain of silence into the caller's render buffers.
///
/// There is no DSP graph behind the NGS2 object table, so the honest output of a render is
/// cleared float samples for the grain length and channel count the system was created with.
/// Fabricating tone or copying waveform data would claim a mixer that does not exist.
pub(super) fn ngs2_system_render(arguments: [u64; 6]) -> u64 {
    remu_gpu::note_inert("ngs2:SystemRender");
    let (system_handle, buffers, buffer_count) =
        (arguments[0], arguments[1], low_u32(arguments[2]));
    if buffer_count == 0 {
        return sce_kernel_error(NGS2_ERROR_INVALID_OUT_ADDRESS);
    }
    let state = ngs2_state();
    let Some(system) = state.systems.get(&system_handle) else {
        return sce_kernel_error(NGS2_ERROR_INVALID_SYSTEM_HANDLE);
    };
    let grain = system.num_grain_samples;
    drop(state);
    const RENDER_BUFFER_INFO_SIZE: u64 = 24;
    const BUFFER_PTR_OFFSET: u64 = 0;
    const BUFFER_SIZE_OFFSET: u64 = 8;
    const NUM_CHANNELS_OFFSET: u64 = 20;
    for index in 0..buffer_count {
        let info = buffers.saturating_add(u64::from(index).saturating_mul(RENDER_BUFFER_INFO_SIZE));
        if !process_guest_range_is_readable(
            info,
            usize::try_from(RENDER_BUFFER_INFO_SIZE).unwrap_or(usize::MAX),
        ) {
            return sce_kernel_error(NGS2_ERROR_INVALID_OUT_ADDRESS);
        }
        let buffer = read_guest_u64(info + BUFFER_PTR_OFFSET);
        let buffer_size = read_guest_u64(info + BUFFER_SIZE_OFFSET);
        let channels = read_guest_u32(info + NUM_CHANNELS_OFFSET).max(1);
        let samples = u64::from(grain).saturating_mul(u64::from(channels));
        let bytes = samples.saturating_mul(4).min(buffer_size);
        if buffer == 0 || bytes == 0 {
            return sce_kernel_error(NGS2_ERROR_INVALID_OUT_ADDRESS);
        }
        let Ok(byte_len) = usize::try_from(bytes) else {
            return sce_kernel_error(NGS2_ERROR_INVALID_OUT_ADDRESS);
        };
        if !process_guest_range_is_readable(buffer, byte_len) {
            eprintln!(
                "[REMu][NGS2] sceNgs2SystemRender rejected unreadable buffer at {buffer:#x} bytes={bytes}"
            );
            return sce_kernel_error(NGS2_ERROR_INVALID_OUT_ADDRESS);
        }
        guest_memset(buffer, 0, bytes);
    }
    0
}

/// Reports the user data a buffer allocator carries once its handlers are known good.
///
/// The handlers are never called: an import runs under the process-wide dispatch lock, and a
/// handler that reached for kernel memory would re-enter it. Emulated objects therefore own their
/// storage host-side and the allocator is validated for its errors alone.
pub(super) fn ngs2_allocator_user_data(allocator: u64) -> Result<u64, u32> {
    if allocator == 0 || !process_guest_range_is_readable(allocator, NGS2_BUFFER_ALLOCATOR_SIZE) {
        return Err(NGS2_ERROR_INVALID_CONTEXT_ALLOCATOR);
    }
    let allocate = read_guest_u64(allocator);
    let free = read_guest_u64(allocator + NGS2_BUFFER_ALLOCATOR_FREE_HANDLER_OFFSET);
    if allocate == 0 || free == 0 {
        return Err(NGS2_ERROR_INVALID_CONTEXT_ALLOCATOR);
    }
    Ok(read_guest_u64(
        allocator + NGS2_BUFFER_ALLOCATOR_USER_DATA_OFFSET,
    ))
}

/// Writes the context buffer a destroyed object reports, carrying its allocator user data back.
pub(super) fn ngs2_write_context_buffer_info(buffer_info: u64, user_data: u64) -> u64 {
    // The record is optional: a caller that frees nothing of its own passes no address for it.
    if !guest_object_is_reachable(buffer_info, NGS2_CONTEXT_BUFFER_INFO_SIZE) {
        return 0;
    }
    guest_memset(buffer_info, 0, NGS2_CONTEXT_BUFFER_INFO_SIZE);
    write_guest_u64(
        buffer_info + NGS2_CONTEXT_BUFFER_INFO_USER_DATA_OFFSET,
        user_data,
    );
    0
}

/// Reports a grain-sample count as accepted, defaulted, or rejected with `error`.
pub(super) fn ngs2_grain_samples(value: u32, default: u32, error: u32) -> Result<u32, u32> {
    if value == 0 {
        return Ok(default);
    }
    if !(NGS2_MIN_GRAIN_SAMPLES..=NGS2_MAX_GRAIN_SAMPLES).contains(&value)
        || !value.is_multiple_of(NGS2_GRAIN_SAMPLE_STEP)
    {
        return Err(error);
    }
    Ok(value)
}

/// Fills an option block with the library's defaults, which are all sentinel zeroes.
pub(super) fn ngs2_system_reset_option(option: u64) -> u64 {
    if !guest_object_is_reachable(option, NGS2_SYSTEM_OPTION_SIZE) {
        return sce_kernel_error(NGS2_ERROR_INVALID_OUT_ADDRESS);
    }
    guest_memset(option, 0, NGS2_SYSTEM_OPTION_SIZE);
    write_guest_u64(option, NGS2_SYSTEM_OPTION_SIZE);
    0
}

/// Default grain and channel sizing a system takes when its option leaves a field at zero.
pub(super) const NGS2_DEFAULT_MAX_GRAIN_SAMPLES: u32 = 512;
pub(super) const NGS2_DEFAULT_NUM_GRAIN_SAMPLES: u32 = 256;
pub(super) const NGS2_DEFAULT_SAMPLE_RATE: u32 = 48_000;
pub(super) const NGS2_DEFAULT_VOICE_CHANNELS: u32 = 8;

/// Creates the root of a mixing graph.
pub(super) fn ngs2_system_create_with_allocator(arguments: [u64; 6]) -> u64 {
    let (option, allocator, out_handle) = (arguments[0], arguments[1], arguments[2]);
    if !guest_object_is_reachable(out_handle, GUEST_HANDLE_SIZE) {
        return sce_kernel_error(NGS2_ERROR_INVALID_OUT_ADDRESS);
    }
    let allocator_user_data = match ngs2_allocator_user_data(allocator) {
        Ok(user_data) => user_data,
        Err(error) => return sce_kernel_error(error),
    };
    let system = match ngs2_system_from_option(option, allocator_user_data) {
        Ok(system) => system,
        Err(error) => return sce_kernel_error(error),
    };
    let mut state = ngs2_state();
    let handle = state.allocate_handle();
    eprintln!(
        "[REMu][NGS2] created system={handle:#x} rate={} grain={}",
        system.sample_rate, system.num_grain_samples
    );
    state.systems.insert(handle, system);
    write_guest_u64(out_handle, handle);
    0
}

/// Reads and validates the option a system is created from.
pub(super) fn ngs2_system_from_option(
    option: u64,
    allocator_user_data: u64,
) -> Result<Ngs2System, u32> {
    let mut system = Ngs2System {
        sample_rate: NGS2_DEFAULT_SAMPLE_RATE,
        num_grain_samples: NGS2_DEFAULT_NUM_GRAIN_SAMPLES,
        max_grain_samples: NGS2_DEFAULT_MAX_GRAIN_SAMPLES,
        max_voice_channels: NGS2_DEFAULT_VOICE_CHANNELS,
        flags: 0,
        allocator_user_data,
        racks: Vec::new(),
    };
    if option == 0 {
        return Ok(system);
    }
    ngs2_option_size(
        option,
        NGS2_SYSTEM_OPTION_MINIMUM_SIZE,
        NGS2_SYSTEM_OPTION_SIZE,
    )?;
    system.flags = read_guest_u32(option + NGS2_SYSTEM_OPTION_FLAGS_OFFSET);
    system.num_grain_samples = ngs2_grain_samples(
        read_guest_u32(option + NGS2_SYSTEM_OPTION_NUM_GRAIN_SAMPLES_OFFSET),
        NGS2_DEFAULT_NUM_GRAIN_SAMPLES,
        NGS2_ERROR_INVALID_NUM_GRAIN_SAMPLES,
    )?;
    system.max_grain_samples = ngs2_grain_samples(
        read_guest_u32(option + NGS2_SYSTEM_OPTION_MAX_GRAIN_SAMPLES_OFFSET),
        NGS2_DEFAULT_MAX_GRAIN_SAMPLES,
        NGS2_ERROR_INVALID_MAX_GRAIN_SAMPLES,
    )?;
    if system.max_grain_samples < system.num_grain_samples {
        return Err(NGS2_ERROR_INVALID_MAX_GRAIN_SAMPLES);
    }
    system.sample_rate = match read_guest_u32(option + NGS2_SYSTEM_OPTION_SAMPLE_RATE_OFFSET) {
        0 => NGS2_DEFAULT_SAMPLE_RATE,
        rate if (NGS2_MIN_SAMPLE_RATE..=NGS2_MAX_SAMPLE_RATE).contains(&rate) => rate,
        _ => return Err(NGS2_ERROR_INVALID_SAMPLE_RATE),
    };
    system.max_voice_channels =
        match read_guest_u32(option + NGS2_SYSTEM_OPTION_MAX_VOICE_CHANNELS_OFFSET) {
            0 => NGS2_DEFAULT_VOICE_CHANNELS,
            channels if channels <= NGS2_MAX_VOICE_CHANNELS => channels,
            _ => return Err(NGS2_ERROR_INVALID_MAX_VOICE_CHANNELS),
        };
    Ok(system)
}

/// Tears down a system, the racks it owns, and their voices.
pub(super) fn ngs2_system_destroy(arguments: [u64; 6]) -> u64 {
    let mut state = ngs2_state();
    let Some(system) = state.systems.remove(&arguments[0]) else {
        return sce_kernel_error(NGS2_ERROR_INVALID_SYSTEM_HANDLE);
    };
    for rack in system.racks {
        state.remove_rack(rack);
    }
    drop(state);
    ngs2_write_context_buffer_info(arguments[1], system.allocator_user_data)
}

/// Creates a rack inside a system.
pub(super) fn ngs2_rack_create_with_allocator(arguments: [u64; 6]) -> u64 {
    let (system_handle, rack_id, option) = (arguments[0], low_u32(arguments[1]), arguments[2]);
    let (allocator, out_handle) = (arguments[3], arguments[4]);
    if !guest_object_is_reachable(out_handle, GUEST_HANDLE_SIZE) {
        return sce_kernel_error(NGS2_ERROR_INVALID_OUT_ADDRESS);
    }
    if !NGS2_RACK_IDS.contains(&rack_id) {
        return sce_kernel_error(NGS2_ERROR_INVALID_RACK_ID);
    }
    let allocator_user_data = match ngs2_allocator_user_data(allocator) {
        Ok(user_data) => user_data,
        Err(error) => return sce_kernel_error(error),
    };
    let mut rack = match ngs2_rack_from_option(option, allocator_user_data) {
        Ok(rack) => rack,
        Err(error) => return sce_kernel_error(error),
    };
    rack.system = system_handle;
    rack.rack_id = rack_id;
    let mut state = ngs2_state();
    if !state.systems.contains_key(&system_handle) {
        return sce_kernel_error(NGS2_ERROR_INVALID_SYSTEM_HANDLE);
    }
    let handle = state.allocate_handle();
    eprintln!(
        "[REMu][NGS2] created rack={handle:#x} id={rack_id:#x} system={system_handle:#x} voices={}",
        rack.max_voices
    );
    state.racks.insert(handle, rack);
    if let Some(system) = state.systems.get_mut(&system_handle) {
        system.racks.push(handle);
    }
    write_guest_u64(out_handle, handle);
    0
}

/// Reads and validates the option a rack is created from.
pub(super) fn ngs2_rack_from_option(
    option: u64,
    allocator_user_data: u64,
) -> Result<Ngs2Rack, u32> {
    let mut rack = Ngs2Rack {
        system: 0,
        rack_id: 0,
        max_voices: 1,
        max_ports: 1,
        max_matrices: 1,
        max_voice_channels: NGS2_DEFAULT_VOICE_CHANNELS,
        max_output_channels: NGS2_DEFAULT_VOICE_CHANNELS,
        allocator_user_data,
        voices: BTreeMap::new(),
    };
    if option == 0 {
        return Ok(rack);
    }
    ngs2_rack_option_size(option)?;
    let _ = read_guest_u32(option + NGS2_RACK_OPTION_FLAGS_OFFSET);
    ngs2_grain_samples(
        read_guest_u32(option + NGS2_RACK_OPTION_MAX_GRAIN_SAMPLES_OFFSET),
        NGS2_DEFAULT_MAX_GRAIN_SAMPLES,
        NGS2_ERROR_INVALID_MAX_GRAIN_SAMPLES,
    )?;
    rack.max_voices = match read_guest_u32(option + NGS2_RACK_OPTION_MAX_VOICES_OFFSET) {
        voices if (1..=NGS2_RACK_MAX_VOICES).contains(&voices) => voices,
        _ => return Err(NGS2_ERROR_INVALID_NUM_VOICES),
    };
    rack.max_matrices = match read_guest_u32(option + NGS2_RACK_OPTION_MAX_MATRICES_OFFSET) {
        matrices if matrices <= NGS2_RACK_MAX_MATRICES => matrices,
        _ => return Err(NGS2_ERROR_INVALID_NUM_MATRICES),
    };
    rack.max_ports = match read_guest_u32(option + NGS2_RACK_OPTION_MAX_PORTS_OFFSET) {
        ports if ports <= NGS2_RACK_MAX_PORTS => ports,
        _ => return Err(NGS2_ERROR_INVALID_NUM_OUTPUTS),
    };
    rack.max_voice_channels =
        match read_guest_u32(option + NGS2_RACK_OPTION_MAX_VOICE_CHANNELS_OFFSET) {
            0 => NGS2_DEFAULT_VOICE_CHANNELS,
            channels if channels <= NGS2_MAX_VOICE_CHANNELS => channels,
            _ => return Err(NGS2_ERROR_INVALID_MAX_VOICE_CHANNELS),
        };
    rack.max_output_channels =
        match read_guest_u32(option + NGS2_RACK_OPTION_MAX_OUTPUT_CHANNELS_OFFSET) {
            0 => NGS2_DEFAULT_VOICE_CHANNELS,
            channels if channels <= NGS2_MAX_VOICE_CHANNELS => channels,
            _ => return Err(NGS2_ERROR_INVALID_MAX_OUTPUT_CHANNELS),
        };
    Ok(rack)
}

/// Tears down a rack and the voices resolved from it.
pub(super) fn ngs2_rack_destroy(arguments: [u64; 6]) -> u64 {
    let mut state = ngs2_state();
    let Some(rack) = state.remove_rack(arguments[0]) else {
        return sce_kernel_error(NGS2_ERROR_INVALID_RACK_HANDLE);
    };
    if let Some(system) = state.systems.get_mut(&rack.system) {
        system.racks.retain(|&handle| handle != arguments[0]);
    }
    drop(state);
    ngs2_write_context_buffer_info(arguments[1], rack.allocator_user_data)
}

/// Resolves one of a rack's pre-allocated voices to its handle.
pub(super) fn ngs2_rack_get_voice_handle(arguments: [u64; 6]) -> u64 {
    let (rack_handle, voice_index, out_handle) =
        (arguments[0], low_u32(arguments[1]), arguments[2]);
    if !guest_object_is_reachable(out_handle, GUEST_HANDLE_SIZE) {
        return sce_kernel_error(NGS2_ERROR_INVALID_OUT_ADDRESS);
    }
    let mut state = ngs2_state();
    let Some(rack) = state.racks.get(&rack_handle) else {
        return sce_kernel_error(NGS2_ERROR_INVALID_RACK_HANDLE);
    };
    if voice_index >= rack.max_voices {
        return sce_kernel_error(NGS2_ERROR_INVALID_VOICE_INDEX);
    }
    // A rack may be sized for thousands of voices. Only the ones a title actually asks for cost
    // anything, and the handle it is given stays the same for every later query.
    let handle = if let Some(&handle) = rack.voices.get(&voice_index) {
        handle
    } else {
        let handle = state.allocate_handle();
        state.voices.insert(
            handle,
            Ngs2Voice {
                rack: rack_handle,
                ..Ngs2Voice::default()
            },
        );
        if let Some(rack) = state.racks.get_mut(&rack_handle) {
            rack.voices.insert(voice_index, handle);
        }
        handle
    };
    write_guest_u64(out_handle, handle);
    0
}

/// Reports the flags and error code a voice currently holds.
pub(super) fn ngs2_voice_get_state(arguments: [u64; 6]) -> u64 {
    let (voice_handle, out_state, state_size) = (arguments[0], arguments[1], arguments[2]);
    if !guest_object_is_reachable(out_state, NGS2_VOICE_STATE_SIZE) {
        return sce_kernel_error(NGS2_ERROR_INVALID_OUT_ADDRESS);
    }
    if state_size != NGS2_VOICE_STATE_SIZE {
        return sce_kernel_error(NGS2_ERROR_INVALID_STATE_SIZE);
    }
    let state = ngs2_state();
    let Some(voice) = state.voices.get(&voice_handle) else {
        return sce_kernel_error(NGS2_ERROR_INVALID_VOICE_HANDLE);
    };
    write_guest_u32(out_state, voice.state_flags);
    write_guest_u32(
        out_state + GUEST_STATUS_WORD_SIZE,
        voice.error_code.cast_unsigned(),
    );
    0
}

/// Reports just the flags a voice currently holds.
pub(super) fn ngs2_voice_get_state_flags(arguments: [u64; 6]) -> u64 {
    let (voice_handle, out_flags) = (arguments[0], arguments[1]);
    if !guest_object_is_reachable(out_flags, GUEST_STATUS_WORD_SIZE) {
        return sce_kernel_error(NGS2_ERROR_INVALID_OUT_ADDRESS);
    }
    let state = ngs2_state();
    let Some(voice) = state.voices.get(&voice_handle) else {
        return sce_kernel_error(NGS2_ERROR_INVALID_VOICE_HANDLE);
    };
    write_guest_u32(out_flags, voice.state_flags);
    0
}

/// Applies one kicked event to the flags a voice reports.
pub(super) fn ngs2_apply_voice_event(voice: &mut Ngs2Voice, event: u32) -> Result<(), u32> {
    voice.state_flags = match event {
        NGS2_VOICE_EVENT_PLAY | NGS2_VOICE_EVENT_RESUME => {
            // PLAYING never clears on its own (no mixer progress) — Group-8 stall-shaped lie.
            remu_gpu::note_inert("ngs2:PLAY-without-auto-complete");
            NGS2_VOICE_STATE_FLAG_IN_USE | NGS2_VOICE_STATE_FLAG_PLAYING
        }
        NGS2_VOICE_EVENT_STOP | NGS2_VOICE_EVENT_STOP_IMMEDIATE => {
            NGS2_VOICE_STATE_FLAG_IN_USE | NGS2_VOICE_STATE_FLAG_STOPPED
        }
        NGS2_VOICE_EVENT_PAUSE => NGS2_VOICE_STATE_FLAG_IN_USE | NGS2_VOICE_STATE_FLAG_PAUSED,
        NGS2_VOICE_EVENT_KILL => 0,
        _ => return Err(NGS2_ERROR_INVALID_VOICE_EVENT_TYPE),
    };
    if event == NGS2_VOICE_EVENT_KILL {
        voice.error_code = 0;
    }
    Ok(())
}
