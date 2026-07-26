// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Audio and video output entry points.
//!
//! This module implements the remaining `libSceNgs2`, `libSceAjm`, `libSceAudioOut` and
//! `libSceVideoOut` entry points.
//!
//! The `NGS2` entry points own the object graph -- systems, racks, voices and the state a title
//! polls for -- but no mixing engine stands behind them, so a voice never consumes waveform data
//! and `sceNgs2SystemRender` stays unimplemented rather than producing silence that looks like a
//! working render path.
//!
//! [`dispatch`] is consulted by the runtime's import dispatcher only after the main table
//! misses, and returns `None` for any NID it does not own so the caller can keep looking.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::{
    AUDIO_OUT, ProbeDispatchState, SCE_AJM_ERROR_INVALID_CONTEXT, SCE_AJM_ERROR_INVALID_PARAMETER,
    VIDEO_OUT_ERROR_INVALID_ADDRESS, VIDEO_OUT_ERROR_INVALID_HANDLE, VIDEO_OUT_ERROR_INVALID_INDEX,
    VIDEO_OUT_ERROR_INVALID_VALUE, VideoOutBufferAttributes, audio, audio_out_error, guest_malloc,
    guest_memset, kernel_process_time, low_i32, low_u32, process_guest_range_is_readable,
    read_guest_bytes, read_guest_u16, read_guest_u32, read_guest_u64, sce_kernel_error,
    signed_i32_return, write_guest_bytes, write_guest_u32, write_guest_u64,
};

/// Handles a NID belonging to this module, or returns `None` if it owns no such NID.
pub(crate) fn dispatch(
    nid: &str,
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> Option<u64> {
    if let Some(result) = dispatch_video_out(nid, arguments, state) {
        return Some(result);
    }
    if let Some(result) = dispatch_ngs2(nid, arguments) {
        return Some(result);
    }
    if let Some(result) = dispatch_ajm(nid, arguments, state) {
        return Some(result);
    }
    dispatch_audio_out(nid, arguments)
}

/// Handles the display-port, buffer-set, vblank and colour entry points.
fn dispatch_video_out(
    nid: &str,
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> Option<u64> {
    Some(match nid {
        "uquVH4-Du78" => video_out_close(arguments, state),
        "N5KDtkIjjJ4" => video_out_unregister_buffers(arguments, state),
        "HuViW4HnrOw" => video_out_submit_change_buffer_attribute2(arguments, state),
        "1FZBKy8HeNU" => video_out_get_vblank_status(arguments, state),
        "j6RaAUlaLv0" => video_out_wait_vblank(arguments, state),
        "U2JJtSqNKZI" => video_out_get_event_id(arguments[0]),
        "+I4K03i3EL0" => video_out_initialize_output_options(arguments[0]),
        "Nv8c-Kb+DUM" => video_out_is_output_supported(arguments, state),
        "T4ucGB8CsnM" => video_out_vrr_unpeg_from_fixed_rate(arguments, state),
        "pv9CI5VC+R0" => video_out_adjust_color(arguments, state),
        _ => return None,
    })
}

/// Handles the `NGS2` system, rack, voice and waveform entry points.
fn dispatch_ngs2(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "AQkj7C0f3PY" => ngs2_system_reset_option(arguments[0]),
        "mPYgU4oYpuY" => ngs2_system_create_with_allocator(arguments),
        "u-WrYDaJA3k" => ngs2_system_destroy(arguments),
        "U546k6orxQo" => ngs2_rack_create_with_allocator(arguments),
        "lCqD7oycmIM" => ngs2_rack_destroy(arguments),
        "MwmHz8pAdAo" => ngs2_rack_get_voice_handle(arguments),
        "uu94irFOGpA" => ngs2_voice_control(arguments),
        "AbYvTOZ8Pts" => ngs2_voice_run_commands(arguments),
        "-TOuuAQ-buE" => ngs2_voice_get_state(arguments),
        "rEh728kXk3w" => ngs2_voice_get_state_flags(arguments),
        "3pCNbVM11UA" => ngs2_calc_waveform_block(arguments),
        "hyVLT2VlOYk" => ngs2_parse_waveform_data(arguments),
        _ => return None,
    })
}

/// Handles the audio-job-manager entry points this module owns.
fn dispatch_ajm(nid: &str, arguments: [u64; 6], state: &ProbeDispatchState) -> Option<u64> {
    Some(match nid {
        "bkRHEYG6lEM" => ajm_memory_register(arguments, state),
        "pIpGiaYkHkM" => ajm_memory_unregister(arguments, state),
        "AxhcqVv5AYU" => ajm_str_error(arguments[0]),
        "eDFeTyi+G3Y" => ajm_dec_mp3_parse_frame(arguments),
        _ => return None,
    })
}

/// Handles the audio-output entry points this module owns.
fn dispatch_audio_out(nid: &str, arguments: [u64; 6]) -> Option<u64> {
    Some(match nid {
        "w3PdaSTSwGE" => audio_out_outputs(arguments),
        "wVwPU50pS1c" => audio_out_set_mix_level_pad_spk(arguments),
        _ => return None,
    })
}

/// Handle the console assigns to the one main display port `sceVideoOutOpen` hands out.
const VIDEO_OUT_MAIN_HANDLE: i32 = 1;
/// Buffer set the runtime keeps. Titles that use more than one swap chain are not served here.
const VIDEO_OUT_PRIMARY_BUFFER_SET: i32 = 0;
/// Bytes one `SceVideoOutVblankStatus` occupies.
const VIDEO_OUT_VBLANK_STATUS_SIZE: u64 = 0x28;
/// Byte offsets within that block.
const VIDEO_OUT_VBLANK_PROCESS_TIME_OFFSET: u64 = 0x08;
const VIDEO_OUT_VBLANK_PROCESS_TIME_COUNTER_OFFSET: u64 = 0x18;
/// Bytes one `SceVideoOutOutputOptions` occupies.
const VIDEO_OUT_OUTPUT_OPTIONS_SIZE: u64 = 0x40;
/// Bytes one `SceVideoOutColorSettings` occupies, and the size every caller must declare.
const VIDEO_OUT_COLOR_SETTINGS_SIZE: u32 = 0x10;
/// Bytes one `SceVideoOutBufferAttribute2` occupies.
const VIDEO_OUT_BUFFER_ATTRIBUTE2_SIZE: usize = 0x50;
/// Byte offsets within that attribute block.
const VIDEO_OUT_ATTRIBUTE_TILING_MODE_OFFSET: u64 = 0x04;
const VIDEO_OUT_ATTRIBUTE_WIDTH_OFFSET: u64 = 0x0c;
const VIDEO_OUT_ATTRIBUTE_HEIGHT_OFFSET: u64 = 0x10;
const VIDEO_OUT_ATTRIBUTE_PITCH_OFFSET: u64 = 0x14;
const VIDEO_OUT_ATTRIBUTE_PIXEL_FORMAT_OFFSET: u64 = 0x20;
/// Tiling modes the display accepts.
const VIDEO_OUT_TILING_MODE_TILE: u32 = 0;
const VIDEO_OUT_TILING_MODE_LINEAR: u32 = 1;
/// Errors the buffer-attribute change reports for a geometry the display cannot scan out.
const VIDEO_OUT_ERROR_INVALID_RESOLUTION: u32 = 0x8029_0005;
const VIDEO_OUT_ERROR_INVALID_TILING_MODE: u32 = 0x8029_0007;
/// Bytes one delivered kernel event occupies, and the offset of its filter-specific flags.
const KERNEL_EVENT_SIZE: usize = 0x20;
const KERNEL_EVENT_FFLAGS_OFFSET: u64 = 0x0c;
/// Output mode the display always presents at.
const VIDEO_OUT_OUTPUT_MODE_DEFAULT: u64 = 1;
/// Answers `sceVideoOutIsOutputSupported` reports.
const VIDEO_OUT_TRUE: u64 = 1;
const VIDEO_OUT_FALSE: u64 = 0;
/// Period of the display's vertical blank, in microseconds.
///
/// The presenter runs at one fixed mode, so this rate is a property of the emulated display
/// rather than of the flip cadence a title selects.
const VIDEO_OUT_VBLANK_INTERVAL_MICROSECONDS: u64 = 1_000_000 / 60;
/// Nanoseconds in one microsecond, the scale between the two process-time clocks.
const NANOSECONDS_PER_MICROSECOND: u64 = 1_000;

/// Instant the vblank counter `sceVideoOutGetVblankStatus` reports counts from.
static VIDEO_OUT_DISPLAY_CLOCK: OnceLock<Mutex<Option<Instant>>> = OnceLock::new();

/// Gamma the guest last asked the display to apply, retained for a colour pipeline to consume.
static VIDEO_OUT_COLOR_GAMMA: OnceLock<Mutex<[f32; 3]>> = OnceLock::new();

/// Reports whether a guest object of `size` bytes can be reached at `address`.
///
/// A call that reports its result through a guest pointer checks it here first, so a null or
/// unreachable address is refused rather than faulting or wrapping the address space.
fn guest_object_is_reachable(address: u64, size: u64) -> bool {
    address != 0
        && process_guest_range_is_readable(address, usize::try_from(size).unwrap_or(usize::MAX))
}

/// Bytes a handle and a status word written back to the guest occupy.
const GUEST_HANDLE_SIZE: u64 = 8;
const GUEST_STATUS_WORD_SIZE: u64 = 4;

/// Reports whether `handle` names the open main display port.
fn video_out_port_is_open(handle: u64, state: &ProbeDispatchState) -> bool {
    low_i32(handle) == VIDEO_OUT_MAIN_HANDLE && state.video_out.is_open
}

/// Returns the instant the display's vblank counter starts from, starting it on first use.
fn video_out_display_clock_start() -> Instant {
    let clock = VIDEO_OUT_DISPLAY_CLOCK.get_or_init(|| Mutex::new(None));
    let mut clock = clock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *clock.get_or_insert_with(Instant::now)
}

/// Discards the vblank counter so the next port to open starts one from zero.
fn video_out_reset_display_clock() {
    let clock = VIDEO_OUT_DISPLAY_CLOCK.get_or_init(|| Mutex::new(None));
    *clock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
}

/// Returns the vblanks elapsed since the counter started and the microseconds since the last one.
fn video_out_vblank_position() -> (u64, u64) {
    let elapsed =
        u64::try_from(video_out_display_clock_start().elapsed().as_micros()).unwrap_or(u64::MAX);
    (
        elapsed / VIDEO_OUT_VBLANK_INTERVAL_MICROSECONDS,
        elapsed % VIDEO_OUT_VBLANK_INTERVAL_MICROSECONDS,
    )
}

/// Releases the display port and everything registered against it.
fn video_out_close(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if !video_out_port_is_open(arguments[0], state) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    state.video_out.is_open = false;
    state.video_out.flip_events.clear();
    state.video_out.display_buffers.clear();
    state.video_out.display_buffer_attributes.clear();
    video_out_reset_display_clock();
    0
}

/// Frees the display-buffer set registered at `setIndex`.
fn video_out_unregister_buffers(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if !video_out_port_is_open(arguments[0], state) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    if low_i32(arguments[1]) != VIDEO_OUT_PRIMARY_BUFFER_SET
        || state.video_out.display_buffers.is_empty()
    {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_INDEX);
    }
    state.video_out.display_buffers.clear();
    state.video_out.display_buffer_attributes.clear();
    0
}

/// Changes the pixel format, tiling and geometry of an already-registered buffer set.
///
/// The console defers the change to the set's next flip. `REmu` applies it at once, matching the
/// registration path it shares state with, so a title that changes attributes and flips in the
/// same frame sees the format it asked for.
fn video_out_submit_change_buffer_attribute2(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    if !video_out_port_is_open(arguments[0], state) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    if low_i32(arguments[1]) != VIDEO_OUT_PRIMARY_BUFFER_SET
        || state.video_out.display_buffers.is_empty()
    {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_INDEX);
    }
    let attribute = arguments[2];
    if attribute == 0
        || !process_guest_range_is_readable(attribute, VIDEO_OUT_BUFFER_ATTRIBUTE2_SIZE)
    {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_ADDRESS);
    }
    let tiling_mode = read_guest_u32(attribute + VIDEO_OUT_ATTRIBUTE_TILING_MODE_OFFSET);
    if !matches!(
        tiling_mode,
        VIDEO_OUT_TILING_MODE_TILE | VIDEO_OUT_TILING_MODE_LINEAR
    ) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_TILING_MODE);
    }
    let width = read_guest_u32(attribute + VIDEO_OUT_ATTRIBUTE_WIDTH_OFFSET);
    let height = read_guest_u32(attribute + VIDEO_OUT_ATTRIBUTE_HEIGHT_OFFSET);
    if width == 0 || height == 0 {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_RESOLUTION);
    }
    let pitch = match read_guest_u32(attribute + VIDEO_OUT_ATTRIBUTE_PITCH_OFFSET) {
        0 => width,
        pitch => pitch,
    };
    let attributes = VideoOutBufferAttributes {
        width,
        height,
        pitch,
        pixel_format: read_guest_u64(attribute + VIDEO_OUT_ATTRIBUTE_PIXEL_FORMAT_OFFSET),
        tiling_mode,
    };
    apply_video_out_attributes(state, attributes);
    0
}

/// Retargets every registered display buffer at `attributes` and republishes the display.
fn apply_video_out_attributes(
    state: &mut ProbeDispatchState,
    attributes: VideoOutBufferAttributes,
) {
    let unchanged = state.video_out.width == attributes.width
        && state.video_out.height == attributes.height
        && state.video_out.pixel_format == attributes.pixel_format
        && state.video_out.tiling_mode == attributes.tiling_mode;
    state.video_out.width = attributes.width;
    state.video_out.height = attributes.height;
    state.video_out.pixel_format = attributes.pixel_format;
    state.video_out.tiling_mode = attributes.tiling_mode;
    for entry in &mut state.video_out.display_buffer_attributes {
        *entry = Some(attributes);
    }
    if unchanged {
        return;
    }
    eprintln!(
        "[REMu][VIDEO] changed buffer attributes to {}x{} format={:#x} tiling={}",
        attributes.width, attributes.height, attributes.pixel_format, attributes.tiling_mode
    );
    remu_gpu::emit_event(remu_gpu::GpuEvent::DisplayRegistered {
        width: attributes.width,
        height: attributes.height,
        pitch: attributes.pitch,
        pixel_format: attributes.pixel_format,
        tiling_mode: attributes.tiling_mode,
        buffers: state.video_out.display_buffers.clone(),
    });
}

/// Reports the display's vblank counter and the process time of the most recent vblank.
fn video_out_get_vblank_status(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if !video_out_port_is_open(arguments[0], state) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    let status = arguments[1];
    if !guest_object_is_reachable(status, VIDEO_OUT_VBLANK_STATUS_SIZE) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_ADDRESS);
    }
    let (count, since_vblank_microseconds) = video_out_vblank_position();
    guest_memset(status, 0, VIDEO_OUT_VBLANK_STATUS_SIZE);
    write_guest_u64(status, count);
    write_guest_u64(
        status + VIDEO_OUT_VBLANK_PROCESS_TIME_OFFSET,
        kernel_process_time(false).saturating_sub(since_vblank_microseconds),
    );
    write_guest_u64(
        status + VIDEO_OUT_VBLANK_PROCESS_TIME_COUNTER_OFFSET,
        kernel_process_time(true)
            .saturating_sub(since_vblank_microseconds.saturating_mul(NANOSECONDS_PER_MICROSECOND)),
    );
    0
}

/// Blocks the caller until the display's next vertical blank.
fn video_out_wait_vblank(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if !video_out_port_is_open(arguments[0], state) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    let (_, since_vblank_microseconds) = video_out_vblank_position();
    std::thread::sleep(Duration::from_micros(
        VIDEO_OUT_VBLANK_INTERVAL_MICROSECONDS - since_vblank_microseconds,
    ));
    0
}

/// Reports which display event a delivered kernel event carries.
fn video_out_get_event_id(event: u64) -> u64 {
    if event == 0 || !process_guest_range_is_readable(event, KERNEL_EVENT_SIZE) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_ADDRESS);
    }
    signed_i32_return(read_guest_u32(event + KERNEL_EVENT_FFLAGS_OFFSET).cast_signed())
}

/// Clears an output-options block so a caller can select the fields it cares about.
fn video_out_initialize_output_options(options: u64) -> u64 {
    if !guest_object_is_reachable(options, VIDEO_OUT_OUTPUT_OPTIONS_SIZE) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_ADDRESS);
    }
    guest_memset(options, 0, VIDEO_OUT_OUTPUT_OPTIONS_SIZE);
    0
}

/// Reports whether the display could later be configured to `mode`.
fn video_out_is_output_supported(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if !video_out_port_is_open(arguments[0], state) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    if arguments[3] != 0 || arguments[4] != 0 {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_VALUE);
    }
    let options = arguments[2];
    if options != 0 && !guest_object_is_reachable(options, VIDEO_OUT_OUTPUT_OPTIONS_SIZE) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_ADDRESS);
    }
    // The presenter owns one fixed mode, so no coded refresh rate or resolution can be reached.
    if arguments[1] == VIDEO_OUT_OUTPUT_MODE_DEFAULT {
        VIDEO_OUT_TRUE
    } else {
        VIDEO_OUT_FALSE
    }
}

/// Returns the display to variable-refresh operation.
fn video_out_vrr_unpeg_from_fixed_rate(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if !video_out_port_is_open(arguments[0], state) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    // The presenter has no variable-refresh path to leave; the display is already free-running
    // with respect to the rate a title pegged it to.
    0
}

/// Applies a gamma adjustment to the display's output.
fn video_out_adjust_color(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    if !video_out_port_is_open(arguments[0], state) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    if low_u32(arguments[2]) != VIDEO_OUT_COLOR_SETTINGS_SIZE {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_VALUE);
    }
    let settings = arguments[1];
    if !guest_object_is_reachable(settings, u64::from(VIDEO_OUT_COLOR_SETTINGS_SIZE)) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_ADDRESS);
    }
    let mut gamma = [0.0_f32; 3];
    for (channel, value) in gamma.iter_mut().enumerate() {
        let offset = u64::try_from(channel * std::mem::size_of::<f32>()).unwrap_or(0);
        *value = f32::from_bits(read_guest_u32(settings + offset));
        if !value.is_finite() || *value < 0.0 {
            return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_VALUE);
        }
    }
    *VIDEO_OUT_COLOR_GAMMA
        .get_or_init(|| Mutex::new([1.0; 3]))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = gamma;
    0
}

/// `NGS2` error codes this module reports.
const NGS2_ERROR_INVALID_OUT_ADDRESS: u32 = 0x804a_8010;
const NGS2_ERROR_INVALID_OPTION_ADDRESS: u32 = 0x804a_8012;
const NGS2_ERROR_INVALID_OPTION_SIZE: u32 = 0x804a_8013;
const NGS2_ERROR_INVALID_CONTEXT_ALLOCATOR: u32 = 0x804a_8023;
const NGS2_ERROR_INVALID_PARAM_ADDRESS: u32 = 0x804a_8030;
const NGS2_ERROR_INVALID_HANDLER_ADDRESS: u32 = 0x804a_8040;
const NGS2_ERROR_INVALID_WAVEFORM_TYPE: u32 = 0x804a_8050;
const NGS2_ERROR_INVALID_SAMPLE_RATE: u32 = 0x804a_8051;
const NGS2_ERROR_INVALID_NUM_CHANNELS: u32 = 0x804a_8052;
const NGS2_ERROR_INVALID_WAVEFORM_ADDRESS: u32 = 0x804a_8055;
const NGS2_ERROR_INVALID_SYSTEM_HANDLE: u32 = 0x804a_8201;
const NGS2_ERROR_INVALID_RACK_HANDLE: u32 = 0x804a_8202;
const NGS2_ERROR_INVALID_VOICE_HANDLE: u32 = 0x804a_8203;
const NGS2_ERROR_INVALID_HANDLE: u32 = 0x804a_8205;
const NGS2_ERROR_INVALID_MODULE_INDEX: u32 = 0x804a_8256;
const NGS2_ERROR_INVALID_RACK_ID: u32 = 0x804a_8300;
const NGS2_ERROR_INVALID_STATE_SIZE: u32 = 0x804a_8302;
const NGS2_ERROR_INVALID_VOICE_INDEX: u32 = 0x804a_8305;
const NGS2_ERROR_INVALID_MODULE_LOCATION: u32 = 0x804a_8308;
const NGS2_ERROR_INVALID_NUM_VOICES: u32 = 0x804a_8309;
const NGS2_ERROR_INVALID_NUM_OUTPUTS: u32 = 0x804a_830e;
const NGS2_ERROR_INVALID_NUM_MATRICES: u32 = 0x804a_830f;
const NGS2_ERROR_INVALID_NUM_GRAIN_SAMPLES: u32 = 0x804a_8310;
const NGS2_ERROR_INVALID_MAX_GRAIN_SAMPLES: u32 = 0x804a_8311;
const NGS2_ERROR_INVALID_MAX_OUTPUT_CHANNELS: u32 = 0x804a_8312;
const NGS2_ERROR_INVALID_MAX_VOICE_CHANNELS: u32 = 0x804a_8313;
const NGS2_ERROR_INVALID_VOICE_SETUP_FLAG: u32 = 0x804a_8352;
const NGS2_ERROR_INVALID_VOICE_EVENT_TYPE: u32 = 0x804a_8353;
const NGS2_ERROR_INVALID_VOICE_MATRIX_INDEX: u32 = 0x804a_8355;
const NGS2_ERROR_INVALID_VOICE_OUTPUT_INDEX: u32 = 0x804a_8357;
const NGS2_ERROR_INVALID_VOICE_PATCH_TYPE: u32 = 0x804a_8359;
const NGS2_ERROR_INVALID_VOICE_PARAM_ID: u32 = 0x804a_835a;
const NGS2_ERROR_INVALID_MATRIX_LEVEL_ADDRESS: u32 = 0x804a_835e;
const NGS2_ERROR_INVALID_VOICE_CONTROL_ADDRESS: u32 = 0x804a_8360;
const NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE: u32 = 0x804a_8361;
const NGS2_ERROR_INVALID_VOICE_CONTROL_ID: u32 = 0x804a_8362;
const NGS2_ERROR_INVALID_WAVEFORM_DATA: u32 = 0x804a_8430;
const NGS2_ERROR_INVALID_WAVEFORM_FORMAT: u32 = 0x804a_8431;
const NGS2_ERROR_UNKNOWN_WAVEFORM_FORMAT: u32 = 0x804a_8432;

/// Bytes one `SceNgs2SystemOption` occupies, and the least it may declare.
///
/// A caller built against an older revision may declare a shorter option as long as it still
/// carries every field this build reads.
const NGS2_SYSTEM_OPTION_SIZE: u64 = 0x90;
const NGS2_SYSTEM_OPTION_MINIMUM_SIZE: u64 = 0x7c;
/// Byte offsets within that option.
const NGS2_SYSTEM_OPTION_FLAGS_OFFSET: u64 = 104;
const NGS2_SYSTEM_OPTION_MAX_GRAIN_SAMPLES_OFFSET: u64 = 108;
const NGS2_SYSTEM_OPTION_NUM_GRAIN_SAMPLES_OFFSET: u64 = 112;
const NGS2_SYSTEM_OPTION_SAMPLE_RATE_OFFSET: u64 = 116;
const NGS2_SYSTEM_OPTION_MAX_VOICE_CHANNELS_OFFSET: u64 = 120;

/// Bytes one `SceNgs2RackOption` occupies, and the least it may declare.
const NGS2_RACK_OPTION_SIZE: u64 = 176;
const NGS2_RACK_OPTION_MINIMUM_SIZE: u64 = 112;
/// Byte offsets within that option.
const NGS2_RACK_OPTION_FLAGS_OFFSET: u64 = 72;
const NGS2_RACK_OPTION_MAX_GRAIN_SAMPLES_OFFSET: u64 = 76;
const NGS2_RACK_OPTION_MAX_VOICES_OFFSET: u64 = 80;
const NGS2_RACK_OPTION_MAX_MATRICES_OFFSET: u64 = 88;
const NGS2_RACK_OPTION_MAX_PORTS_OFFSET: u64 = 92;
const NGS2_RACK_OPTION_MAX_VOICE_CHANNELS_OFFSET: u64 = 96;
const NGS2_RACK_OPTION_MAX_OUTPUT_CHANNELS_OFFSET: u64 = 100;

/// Bytes one `SceNgs2ContextBufferInfo` occupies, and the offset of its user data.
const NGS2_CONTEXT_BUFFER_INFO_SIZE: u64 = 64;
const NGS2_CONTEXT_BUFFER_INFO_USER_DATA_OFFSET: u64 = 56;
/// Bytes one `SceNgs2BufferAllocator` occupies.
const NGS2_BUFFER_ALLOCATOR_SIZE: usize = 24;
const NGS2_BUFFER_ALLOCATOR_FREE_HANDLER_OFFSET: u64 = 8;
const NGS2_BUFFER_ALLOCATOR_USER_DATA_OFFSET: u64 = 16;

/// Bytes one `SceNgs2VoiceState` occupies.
const NGS2_VOICE_STATE_SIZE: u64 = 8;

/// Grain, channel and rack limits the library enforces.
const NGS2_MIN_GRAIN_SAMPLES: u32 = 64;
const NGS2_MAX_GRAIN_SAMPLES: u32 = 2048;
const NGS2_GRAIN_SAMPLE_STEP: u32 = 64;
const NGS2_MAX_VOICE_CHANNELS: u32 = 37;
const NGS2_MIN_SAMPLE_RATE: u32 = 11_025;
const NGS2_MAX_SAMPLE_RATE: u32 = 192_000;
const NGS2_RACK_MAX_VOICES: u32 = 32_768;
const NGS2_RACK_MAX_PORTS: u32 = 32;
const NGS2_RACK_MAX_MATRICES: u32 = 32;
const NGS2_RACK_MAX_MODULES: u32 = 32;
const NGS2_MODULE_MAX_LOCATIONS: u32 = 34;

/// Rack identifiers the library recognizes.
const NGS2_RACK_IDS: [u32; 9] = [
    0x1000, 0x2000, 0x2001, 0x3000, 0x4000, 0x4001, 0x4002, 0x4003, 0x4004,
];

/// Voice state flags `sceNgs2VoiceGetStateFlags` reports.
const NGS2_VOICE_STATE_FLAG_IN_USE: u32 = 1 << 0;
const NGS2_VOICE_STATE_FLAG_PLAYING: u32 = 1 << 1;
const NGS2_VOICE_STATE_FLAG_PAUSED: u32 = 1 << 2;
const NGS2_VOICE_STATE_FLAG_STOPPED: u32 = 1 << 3;

/// Voice events a control entry or a command may kick.
const NGS2_VOICE_EVENT_PLAY: u32 = 0x0001;
const NGS2_VOICE_EVENT_STOP: u32 = 0x0002;
const NGS2_VOICE_EVENT_STOP_IMMEDIATE: u32 = 0x0004;
const NGS2_VOICE_EVENT_KILL: u32 = 0x0008;
const NGS2_VOICE_EVENT_PAUSE: u32 = 0x0010;
const NGS2_VOICE_EVENT_RESUME: u32 = 0x0020;

/// First handle the object table hands out.
///
/// The value sits far outside anything the guest maps, so a title that mistakes a handle for a
/// pointer faults at once instead of writing over live state.
const NGS2_HANDLE_BASE: u64 = 0x4e47_5332_0000;
/// Distance between consecutive handles.
const NGS2_HANDLE_STRIDE: u64 = 0x40;

/// One system object and the option it was created from.
#[derive(Debug)]
struct Ngs2System {
    sample_rate: u32,
    num_grain_samples: u32,
    max_grain_samples: u32,
    max_voice_channels: u32,
    flags: u32,
    allocator_user_data: u64,
    racks: Vec<u64>,
}

/// One rack object, its sizing, and the voices a title has resolved from it.
#[derive(Debug)]
struct Ngs2Rack {
    system: u64,
    rack_id: u32,
    max_voices: u32,
    max_ports: u32,
    max_matrices: u32,
    max_voice_channels: u32,
    max_output_channels: u32,
    allocator_user_data: u64,
    voices: BTreeMap<u32, u64>,
}

/// One voice, and everything a title can observe about it without a render engine.
#[derive(Debug, Default)]
struct Ngs2Voice {
    rack: u64,
    state_flags: u32,
    error_code: i32,
    user_data: u64,
    num_channels: u32,
    setup_flags: u32,
    callback_handler: u64,
}

/// Every live `NGS2` object, keyed by the handle the guest holds.
#[derive(Debug, Default)]
struct Ngs2State {
    next_handle: u64,
    systems: BTreeMap<u64, Ngs2System>,
    racks: BTreeMap<u64, Ngs2Rack>,
    voices: BTreeMap<u64, Ngs2Voice>,
}

impl Ngs2State {
    /// Returns a handle no live object holds.
    fn allocate_handle(&mut self) -> u64 {
        let index = self.next_handle;
        self.next_handle = index.saturating_add(1);
        NGS2_HANDLE_BASE.saturating_add(index.saturating_mul(NGS2_HANDLE_STRIDE))
    }

    /// Discards a rack and every voice resolved from it.
    fn remove_rack(&mut self, rack_handle: u64) -> Option<Ngs2Rack> {
        let rack = self.racks.remove(&rack_handle)?;
        for voice in rack.voices.values() {
            self.voices.remove(voice);
        }
        Some(rack)
    }
}

/// Returns the process-wide `NGS2` object table.
fn ngs2_state() -> std::sync::MutexGuard<'static, Ngs2State> {
    static STATE: OnceLock<Mutex<Ngs2State>> = OnceLock::new();
    STATE
        .get_or_init(|| Mutex::new(Ngs2State::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Reports the size a caller declared for an option block, or the error it violates.
fn ngs2_option_size(option: u64, minimum: u64, maximum: u64) -> Result<u64, u32> {
    if !process_guest_range_is_readable(option, std::mem::size_of::<u64>()) {
        return Err(NGS2_ERROR_INVALID_OPTION_ADDRESS);
    }
    let size = read_guest_u64(option);
    if !(minimum..=maximum).contains(&size)
        || !process_guest_range_is_readable(option, usize::try_from(size).unwrap_or(usize::MAX))
    {
        return Err(NGS2_ERROR_INVALID_OPTION_SIZE);
    }
    Ok(size)
}

/// Reports the user data a buffer allocator carries once its handlers are known good.
///
/// The handlers are never called: an import runs under the process-wide dispatch lock, and a
/// handler that reached for kernel memory would re-enter it. Emulated objects therefore own their
/// storage host-side and the allocator is validated for its errors alone.
fn ngs2_allocator_user_data(allocator: u64) -> Result<u64, u32> {
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
fn ngs2_write_context_buffer_info(buffer_info: u64, user_data: u64) -> u64 {
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
fn ngs2_grain_samples(value: u32, default: u32, error: u32) -> Result<u32, u32> {
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
fn ngs2_system_reset_option(option: u64) -> u64 {
    if !guest_object_is_reachable(option, NGS2_SYSTEM_OPTION_SIZE) {
        return sce_kernel_error(NGS2_ERROR_INVALID_OUT_ADDRESS);
    }
    guest_memset(option, 0, NGS2_SYSTEM_OPTION_SIZE);
    write_guest_u64(option, NGS2_SYSTEM_OPTION_SIZE);
    0
}

/// Default grain and channel sizing a system takes when its option leaves a field at zero.
const NGS2_DEFAULT_MAX_GRAIN_SAMPLES: u32 = 512;
const NGS2_DEFAULT_NUM_GRAIN_SAMPLES: u32 = 256;
const NGS2_DEFAULT_SAMPLE_RATE: u32 = 48_000;
const NGS2_DEFAULT_VOICE_CHANNELS: u32 = 8;

/// Creates the root of a mixing graph.
fn ngs2_system_create_with_allocator(arguments: [u64; 6]) -> u64 {
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
fn ngs2_system_from_option(option: u64, allocator_user_data: u64) -> Result<Ngs2System, u32> {
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
fn ngs2_system_destroy(arguments: [u64; 6]) -> u64 {
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
fn ngs2_rack_create_with_allocator(arguments: [u64; 6]) -> u64 {
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
fn ngs2_rack_from_option(option: u64, allocator_user_data: u64) -> Result<Ngs2Rack, u32> {
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
    ngs2_option_size(option, NGS2_RACK_OPTION_MINIMUM_SIZE, NGS2_RACK_OPTION_SIZE)?;
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
fn ngs2_rack_destroy(arguments: [u64; 6]) -> u64 {
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
fn ngs2_rack_get_voice_handle(arguments: [u64; 6]) -> u64 {
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
fn ngs2_voice_get_state(arguments: [u64; 6]) -> u64 {
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
fn ngs2_voice_get_state_flags(arguments: [u64; 6]) -> u64 {
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
fn ngs2_apply_voice_event(voice: &mut Ngs2Voice, event: u32) -> Result<(), u32> {
    voice.state_flags = match event {
        NGS2_VOICE_EVENT_PLAY | NGS2_VOICE_EVENT_RESUME => {
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

/// Voice control identifiers a parameter chain may carry.
const NGS2_VOICE_SET_MATRIX_LEVELS: u32 = 0x0001;
const NGS2_VOICE_SET_PORT_VOLUME: u32 = 0x0002;
const NGS2_VOICE_SET_PORT_MATRIX: u32 = 0x0003;
const NGS2_VOICE_SET_PORT_DELAY: u32 = 0x0004;
const NGS2_VOICE_PATCH: u32 = 0x0005;
const NGS2_VOICE_KICK_EVENT: u32 = 0x0006;
const NGS2_VOICE_SET_CALLBACK: u32 = 0x0007;

/// Bytes a voice control parameter header occupies.
const NGS2_VOICE_PARAM_HEADER_SIZE: u16 = 8;
/// Entries one chain may carry before it is treated as malformed.
///
/// A chain links its entries by a signed offset, so a broken one can close a cycle rather than
/// terminate; this bounds the walk without depending on the guest's own bookkeeping.
const NGS2_VOICE_CONTROL_CHAIN_LIMIT: usize = 256;
/// Levels one matrix may carry.
const NGS2_MAX_MATRIX_LEVELS: u32 = NGS2_MAX_VOICE_CHANNELS * NGS2_MAX_VOICE_CHANNELS;
/// Error reported for a matrix level count outside that bound.
const NGS2_ERROR_INVALID_NUM_MATRIX_VOLUMES: u32 = 0x804a_835d;
/// Delay one port may hold, in samples.
const NGS2_MAX_PORT_DELAY_SAMPLES: u32 = 65_535;
/// Highest patch type a patch parameter may name.
const NGS2_VOICE_PATCH_MAX_TYPE: u32 = 3;
/// Highest input identifier a patch destination may name.
const NGS2_VOICE_MAX_INPUT_ID: u32 = 1;

/// Rack sizing a voice's parameters are checked against.
#[derive(Clone, Copy, Debug)]
struct Ngs2RackLimits {
    max_ports: u32,
    max_matrices: u32,
}

/// Returns the sizing of the rack that owns `voice_handle`.
fn ngs2_voice_limits(state: &Ngs2State, voice_handle: u64) -> Option<Ngs2RackLimits> {
    let voice = state.voices.get(&voice_handle)?;
    let rack = state.racks.get(&voice.rack)?;
    Some(Ngs2RackLimits {
        max_ports: rack.max_ports,
        max_matrices: rack.max_matrices,
    })
}

/// Rejects a control entry that is too short for the parameters its identifier implies.
fn ngs2_require_control_size(size: u16, required: u16) -> Result<(), u32> {
    if size < required {
        return Err(NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE);
    }
    Ok(())
}

/// Executes a chain of voice control parameters against one voice.
fn ngs2_voice_control(arguments: [u64; 6]) -> u64 {
    let (voice_handle, param_list) = (arguments[0], arguments[1]);
    if param_list == 0 {
        return sce_kernel_error(NGS2_ERROR_INVALID_VOICE_CONTROL_ADDRESS);
    }
    let mut state = ngs2_state();
    let Some(limits) = ngs2_voice_limits(&state, voice_handle) else {
        return sce_kernel_error(NGS2_ERROR_INVALID_VOICE_HANDLE);
    };
    let mut address = param_list;
    for _ in 0..NGS2_VOICE_CONTROL_CHAIN_LIMIT {
        if !process_guest_range_is_readable(address, usize::from(NGS2_VOICE_PARAM_HEADER_SIZE)) {
            return sce_kernel_error(NGS2_ERROR_INVALID_VOICE_CONTROL_ADDRESS);
        }
        let size = read_guest_u16(address);
        let next = read_guest_u16(address + 2).cast_signed();
        let id = read_guest_u32(address + 4);
        if size < NGS2_VOICE_PARAM_HEADER_SIZE
            || !process_guest_range_is_readable(address, usize::from(size))
        {
            return sce_kernel_error(NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE);
        }
        if let Err(error) =
            ngs2_apply_voice_control(&mut state, voice_handle, limits, address, size, id)
        {
            return sce_kernel_error(error);
        }
        if next == 0 {
            return 0;
        }
        address = address.wrapping_add_signed(i64::from(next));
    }
    sce_kernel_error(NGS2_ERROR_INVALID_VOICE_CONTROL_ADDRESS)
}

/// Executes one control entry.
///
/// Levels, volumes and patches are validated and then discarded: nothing downstream mixes, and a
/// level this build cannot apply is worth less than the error a bad one still reports.
fn ngs2_apply_voice_control(
    state: &mut Ngs2State,
    voice_handle: u64,
    limits: Ngs2RackLimits,
    address: u64,
    size: u16,
    id: u32,
) -> Result<(), u32> {
    match id {
        NGS2_VOICE_SET_MATRIX_LEVELS => {
            ngs2_require_control_size(size, 24)?;
            if read_guest_u32(address + 8) >= limits.max_matrices {
                return Err(NGS2_ERROR_INVALID_VOICE_MATRIX_INDEX);
            }
            let levels = read_guest_u32(address + 12);
            if !(1..=NGS2_MAX_MATRIX_LEVELS).contains(&levels) {
                return Err(NGS2_ERROR_INVALID_NUM_MATRIX_VOLUMES);
            }
            let level_array = read_guest_u64(address + 16);
            let span = usize::try_from(levels)
                .unwrap_or(usize::MAX)
                .saturating_mul(std::mem::size_of::<f32>());
            if level_array == 0 || !process_guest_range_is_readable(level_array, span) {
                return Err(NGS2_ERROR_INVALID_MATRIX_LEVEL_ADDRESS);
            }
            Ok(())
        }
        NGS2_VOICE_SET_PORT_VOLUME | NGS2_VOICE_SET_PORT_MATRIX | NGS2_VOICE_SET_PORT_DELAY => {
            ngs2_require_control_size(size, 16)?;
            if read_guest_u32(address + 8) >= limits.max_ports {
                return Err(NGS2_ERROR_INVALID_VOICE_OUTPUT_INDEX);
            }
            if id == NGS2_VOICE_SET_PORT_DELAY
                && read_guest_u32(address + 12) > NGS2_MAX_PORT_DELAY_SAMPLES
            {
                return Err(NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE);
            }
            Ok(())
        }
        NGS2_VOICE_PATCH => {
            ngs2_require_control_size(size, 24)?;
            if read_guest_u32(address + 8) >= limits.max_ports {
                return Err(NGS2_ERROR_INVALID_VOICE_OUTPUT_INDEX);
            }
            if read_guest_u32(address + 12) > NGS2_VOICE_MAX_INPUT_ID {
                return Err(NGS2_ERROR_INVALID_VOICE_PATCH_TYPE);
            }
            let destination = read_guest_u64(address + 16);
            if destination != 0
                && !state.racks.contains_key(&destination)
                && !state.voices.contains_key(&destination)
            {
                return Err(NGS2_ERROR_INVALID_HANDLE);
            }
            Ok(())
        }
        NGS2_VOICE_KICK_EVENT => {
            ngs2_require_control_size(size, 12)?;
            let event = read_guest_u32(address + 8);
            let voice = state
                .voices
                .get_mut(&voice_handle)
                .ok_or(NGS2_ERROR_INVALID_VOICE_HANDLE)?;
            ngs2_apply_voice_event(voice, event)
        }
        NGS2_VOICE_SET_CALLBACK => {
            ngs2_require_control_size(size, 32)?;
            let handler = read_guest_u64(address + 8);
            if handler == 0 {
                return Err(NGS2_ERROR_INVALID_HANDLER_ADDRESS);
            }
            let voice = state
                .voices
                .get_mut(&voice_handle)
                .ok_or(NGS2_ERROR_INVALID_VOICE_HANDLE)?;
            voice.callback_handler = handler;
            Ok(())
        }
        _ => Err(NGS2_ERROR_INVALID_VOICE_CONTROL_ID),
    }
}

/// Bytes one `SceNgs2CommandParam` occupies, and the offset of its value union.
const NGS2_COMMAND_PARAM_SIZE: u64 = 16;
const NGS2_COMMAND_PARAM_VALUE_OFFSET: u64 = 8;
/// Entries one command array may carry before it is treated as malformed.
const NGS2_COMMAND_PARAM_LIMIT: u64 = 4096;
/// Component that addresses the voice itself; everything below the reserved range is a module.
const NGS2_COMPONENT_ID_VOICE: u32 = 0x00;
const NGS2_COMPONENT_ID_SYSTEM: u32 = 0xf0;
/// Parameter identifier field that selects a command rather than a query or a callback.
const NGS2_PARAM_ID_TYPE_COMMAND: u32 = 0;

/// Voice command interface identifiers.
const NGS2_VOICE_COMMAND_SET_USER_DATA: u32 = 0x00;
const NGS2_VOICE_COMMAND_SETUP: u32 = 0x01;
const NGS2_VOICE_COMMAND_KICK_EVENT: u32 = 0x02;
const NGS2_VOICE_COMMAND_SET_AUTO_SUSPEND: u32 = 0x03;
const NGS2_VOICE_COMMAND_SET_MODULE_LOCATION: u32 = 0x04;
const NGS2_VOICE_COMMAND_SET_MATRIX_LEVELS: u32 = 0x05;
const NGS2_VOICE_COMMAND_SET_PORT_VOLUME: u32 = 0x06;
const NGS2_VOICE_COMMAND_SET_PORT_MATRIX: u32 = 0x07;
const NGS2_VOICE_COMMAND_SET_PORT_PANNER: u32 = 0x08;
const NGS2_VOICE_COMMAND_PATCH: u32 = 0x09;
const NGS2_VOICE_COMMAND_SET_STRING_PARAM: u32 = 0x0a;
const NGS2_VOICE_COMMAND_BYPASS_MODULE: u32 = 0x0b;
const NGS2_VOICE_COMMAND_SET_CALLBACK: u32 = 0x0c;
const NGS2_VOICE_COMMAND_SET_MODULE_ORDER: u32 = 0x0d;

/// One decoded command parameter.
#[derive(Clone, Copy, Debug)]
struct Ngs2Command {
    component: u32,
    kind: u32,
    interface: u32,
    unit: u32,
    values: u32,
    value: u64,
}

/// Splits a command parameter into the fields its identifier packs together.
fn ngs2_decode_command(entry: u64) -> Ngs2Command {
    let param_id = read_guest_u32(entry);
    Ngs2Command {
        component: (param_id >> 16) & 0xff,
        kind: (param_id >> 14) & 0x3,
        interface: param_id & 0x3fff,
        unit: param_id >> 24,
        values: u32::from(read_guest_u16(entry + 6)),
        value: read_guest_u64(entry + NGS2_COMMAND_PARAM_VALUE_OFFSET),
    }
}

/// Applies an array of command parameters to one voice in order.
fn ngs2_voice_run_commands(arguments: [u64; 6]) -> u64 {
    let (voice_handle, params, count) = (arguments[0], arguments[1], arguments[2]);
    if params == 0 {
        return sce_kernel_error(NGS2_ERROR_INVALID_PARAM_ADDRESS);
    }
    if count > NGS2_COMMAND_PARAM_LIMIT {
        return sce_kernel_error(NGS2_ERROR_INVALID_PARAM_ADDRESS);
    }
    let span = usize::try_from(count.saturating_mul(NGS2_COMMAND_PARAM_SIZE)).unwrap_or(usize::MAX);
    if !process_guest_range_is_readable(params, span) {
        return sce_kernel_error(NGS2_ERROR_INVALID_PARAM_ADDRESS);
    }
    let mut state = ngs2_state();
    let Some(limits) = ngs2_voice_limits(&state, voice_handle) else {
        return sce_kernel_error(NGS2_ERROR_INVALID_VOICE_HANDLE);
    };
    for index in 0..count {
        let entry = params + index * NGS2_COMMAND_PARAM_SIZE;
        let command = ngs2_decode_command(entry);
        if let Err(error) = ngs2_apply_voice_command(&mut state, voice_handle, limits, command) {
            return sce_kernel_error(error);
        }
    }
    0
}

/// Executes one decoded command parameter.
fn ngs2_apply_voice_command(
    state: &mut Ngs2State,
    voice_handle: u64,
    limits: Ngs2RackLimits,
    command: Ngs2Command,
) -> Result<(), u32> {
    if command.kind != NGS2_PARAM_ID_TYPE_COMMAND || command.component >= NGS2_COMPONENT_ID_SYSTEM {
        return Err(NGS2_ERROR_INVALID_VOICE_PARAM_ID);
    }
    // A component below the reserved range names one of the rack's modules. This build has no
    // module graph to route such a parameter into, so it is accepted and dropped.
    if command.component != NGS2_COMPONENT_ID_VOICE {
        return Ok(());
    }
    match command.interface {
        NGS2_VOICE_COMMAND_SET_USER_DATA => {
            let voice = ngs2_voice_mut(state, voice_handle)?;
            voice.user_data = command.value;
            Ok(())
        }
        NGS2_VOICE_COMMAND_SETUP => ngs2_apply_voice_setup(state, voice_handle, command),
        NGS2_VOICE_COMMAND_KICK_EVENT => {
            let event = low_u32(command.value);
            ngs2_apply_voice_event(ngs2_voice_mut(state, voice_handle)?, event)
        }
        NGS2_VOICE_COMMAND_SET_MODULE_LOCATION => {
            if command.unit >= NGS2_RACK_MAX_MODULES {
                return Err(NGS2_ERROR_INVALID_MODULE_INDEX);
            }
            if low_u32(command.value) >= NGS2_MODULE_MAX_LOCATIONS {
                return Err(NGS2_ERROR_INVALID_MODULE_LOCATION);
            }
            Ok(())
        }
        NGS2_VOICE_COMMAND_SET_MATRIX_LEVELS => {
            if command.unit >= limits.max_matrices {
                return Err(NGS2_ERROR_INVALID_VOICE_MATRIX_INDEX);
            }
            let span = usize::try_from(command.values)
                .unwrap_or(usize::MAX)
                .saturating_mul(std::mem::size_of::<f32>());
            if command.value == 0 || !process_guest_range_is_readable(command.value, span) {
                return Err(NGS2_ERROR_INVALID_MATRIX_LEVEL_ADDRESS);
            }
            Ok(())
        }
        NGS2_VOICE_COMMAND_SET_PORT_VOLUME
        | NGS2_VOICE_COMMAND_SET_PORT_MATRIX
        | NGS2_VOICE_COMMAND_SET_PORT_PANNER => {
            if command.unit >= limits.max_ports {
                return Err(NGS2_ERROR_INVALID_VOICE_OUTPUT_INDEX);
            }
            Ok(())
        }
        NGS2_VOICE_COMMAND_PATCH => {
            if command.unit >= limits.max_ports {
                return Err(NGS2_ERROR_INVALID_VOICE_OUTPUT_INDEX);
            }
            if command.value == 0
                || !process_guest_range_is_readable(command.value, std::mem::size_of::<u32>())
                || read_guest_u32(command.value) > NGS2_VOICE_PATCH_MAX_TYPE
            {
                return Err(NGS2_ERROR_INVALID_VOICE_PATCH_TYPE);
            }
            Ok(())
        }
        NGS2_VOICE_COMMAND_BYPASS_MODULE => {
            if command.unit >= NGS2_RACK_MAX_MODULES {
                return Err(NGS2_ERROR_INVALID_MODULE_INDEX);
            }
            Ok(())
        }
        NGS2_VOICE_COMMAND_SET_CALLBACK => {
            if command.value == 0 {
                return Err(NGS2_ERROR_INVALID_HANDLER_ADDRESS);
            }
            Ok(())
        }
        NGS2_VOICE_COMMAND_SET_AUTO_SUSPEND
        | NGS2_VOICE_COMMAND_SET_STRING_PARAM
        | NGS2_VOICE_COMMAND_SET_MODULE_ORDER => {
            if command.value == 0 {
                return Err(NGS2_ERROR_INVALID_PARAM_ADDRESS);
            }
            Ok(())
        }
        _ => Err(NGS2_ERROR_INVALID_VOICE_PARAM_ID),
    }
}

/// Returns a voice for mutation, or the error a stale handle reports.
fn ngs2_voice_mut(state: &mut Ngs2State, voice_handle: u64) -> Result<&mut Ngs2Voice, u32> {
    state
        .voices
        .get_mut(&voice_handle)
        .ok_or(NGS2_ERROR_INVALID_VOICE_HANDLE)
}

/// Records the channel layout and sampling rate a voice is set up for.
fn ngs2_apply_voice_setup(
    state: &mut Ngs2State,
    voice_handle: u64,
    command: Ngs2Command,
) -> Result<(), u32> {
    const SETUP_PARAM_SIZE: usize = 8;
    /// Field of the setup flags that selects the voice's base sampling rate.
    const SAMPLE_RATE_MASK: u32 = 0x0000_000f;
    const MAX_SAMPLE_RATE_SELECTOR: u32 = 5;
    /// Channel counts above one byte pack an input count above an output count.
    const CHANNEL_PAIR_SHIFT: u32 = 8;
    const CHANNEL_MASK: u32 = 0xff;

    if command.value == 0 || !process_guest_range_is_readable(command.value, SETUP_PARAM_SIZE) {
        return Err(NGS2_ERROR_INVALID_PARAM_ADDRESS);
    }
    let channels = read_guest_u32(command.value);
    let flags = read_guest_u32(command.value + 4);
    let (inputs, outputs) = if channels > CHANNEL_MASK {
        (channels >> CHANNEL_PAIR_SHIFT, channels & CHANNEL_MASK)
    } else {
        (channels, channels)
    };
    if inputs > NGS2_MAX_VOICE_CHANNELS || outputs > NGS2_MAX_VOICE_CHANNELS {
        return Err(NGS2_ERROR_INVALID_NUM_CHANNELS);
    }
    if flags & SAMPLE_RATE_MASK > MAX_SAMPLE_RATE_SELECTOR {
        return Err(NGS2_ERROR_INVALID_VOICE_SETUP_FLAG);
    }
    let voice = ngs2_voice_mut(state, voice_handle)?;
    voice.num_channels = channels;
    voice.setup_flags = flags;
    voice.state_flags |= NGS2_VOICE_STATE_FLAG_IN_USE;
    Ok(())
}

/// Bytes one `SceNgs2WaveformBlock` occupies.
const NGS2_WAVEFORM_BLOCK_SIZE: usize = 40;
/// Bytes one `SceNgs2WaveformInfo` occupies, and one `SceNgs2WaveformFormat` inside it.
const NGS2_WAVEFORM_INFO_SIZE: usize = 232;
const NGS2_WAVEFORM_FORMAT_SIZE: usize = 24;
/// Byte offset of the first block inside a waveform information block.
const NGS2_WAVEFORM_INFO_BLOCK_OFFSET: usize = 72;

/// Waveform types this build can address by sample position.
const NGS2_WAVEFORM_TYPE_PCM_I8: u32 = 0x10;
const NGS2_WAVEFORM_TYPE_PCM_U8: u32 = 0x11;
const NGS2_WAVEFORM_TYPE_PCM_I16L: u32 = 0x12;
const NGS2_WAVEFORM_TYPE_PCM_I16B: u32 = 0x13;
const NGS2_WAVEFORM_TYPE_PCM_I24L: u32 = 0x14;
const NGS2_WAVEFORM_TYPE_PCM_I24B: u32 = 0x15;
const NGS2_WAVEFORM_TYPE_PCM_I32L: u32 = 0x16;
const NGS2_WAVEFORM_TYPE_PCM_I32B: u32 = 0x17;
const NGS2_WAVEFORM_TYPE_PCM_F32L: u32 = 0x18;
const NGS2_WAVEFORM_TYPE_PCM_F32B: u32 = 0x19;
const NGS2_WAVEFORM_TYPE_PCM_F64L: u32 = 0x1a;
const NGS2_WAVEFORM_TYPE_PCM_F64B: u32 = 0x1b;
const NGS2_WAVEFORM_TYPE_VAG: u32 = 0x1c;
/// Samples one compressed `VAG` unit carries, and the bytes it occupies per channel.
const NGS2_VAG_UNIT_SAMPLES: u32 = 28;
const NGS2_VAG_UNIT_BYTES: u32 = 16;

/// Returns the bytes and samples one audio unit of a sample-addressed waveform carries.
///
/// Codec waveforms carry their frame geometry in the format's configuration word, which this
/// build does not decode; they are refused rather than answered with an offset into the wrong
/// frame.
fn ngs2_audio_unit(waveform_type: u32, channels: u32) -> Option<(u32, u32)> {
    let bytes_per_channel = match waveform_type {
        NGS2_WAVEFORM_TYPE_PCM_I8 | NGS2_WAVEFORM_TYPE_PCM_U8 => 1,
        NGS2_WAVEFORM_TYPE_PCM_I16L | NGS2_WAVEFORM_TYPE_PCM_I16B => 2,
        NGS2_WAVEFORM_TYPE_PCM_I24L | NGS2_WAVEFORM_TYPE_PCM_I24B => 3,
        NGS2_WAVEFORM_TYPE_PCM_I32L
        | NGS2_WAVEFORM_TYPE_PCM_I32B
        | NGS2_WAVEFORM_TYPE_PCM_F32L
        | NGS2_WAVEFORM_TYPE_PCM_F32B => 4,
        NGS2_WAVEFORM_TYPE_PCM_F64L | NGS2_WAVEFORM_TYPE_PCM_F64B => 8,
        NGS2_WAVEFORM_TYPE_VAG => {
            return Some((
                NGS2_VAG_UNIT_BYTES.checked_mul(channels)?,
                NGS2_VAG_UNIT_SAMPLES,
            ));
        }
        _ => return None,
    };
    Some((bytes_per_channel * channels, 1))
}

/// Serializes one waveform block.
fn ngs2_waveform_block_bytes(
    data_offset: u64,
    data_size: u64,
    skip_samples: u32,
    samples: u32,
) -> [u8; NGS2_WAVEFORM_BLOCK_SIZE] {
    let mut block = [0_u8; NGS2_WAVEFORM_BLOCK_SIZE];
    block[..8].copy_from_slice(&data_offset.to_le_bytes());
    block[8..16].copy_from_slice(&data_size.to_le_bytes());
    block[16..20].copy_from_slice(&1_u32.to_le_bytes());
    block[20..24].copy_from_slice(&skip_samples.to_le_bytes());
    block[24..28].copy_from_slice(&samples.to_le_bytes());
    block
}

/// Converts a sample range into the block that addresses it.
fn ngs2_calc_waveform_block(arguments: [u64; 6]) -> u64 {
    let (format, sample_position, samples, out_block) = (
        arguments[0],
        low_u32(arguments[1]),
        low_u32(arguments[2]),
        arguments[3],
    );
    if !guest_object_is_reachable(out_block, NGS2_WAVEFORM_BLOCK_SIZE as u64) {
        return sce_kernel_error(NGS2_ERROR_INVALID_OUT_ADDRESS);
    }
    if format == 0 || !process_guest_range_is_readable(format, NGS2_WAVEFORM_FORMAT_SIZE) {
        return sce_kernel_error(NGS2_ERROR_INVALID_WAVEFORM_ADDRESS);
    }
    let waveform_type = read_guest_u32(format);
    let channels = read_guest_u32(format + 4);
    if channels == 0 || channels > NGS2_MAX_VOICE_CHANNELS {
        return sce_kernel_error(NGS2_ERROR_INVALID_NUM_CHANNELS);
    }
    let Some((unit_size, unit_samples)) = ngs2_audio_unit(waveform_type, channels) else {
        return sce_kernel_error(NGS2_ERROR_INVALID_WAVEFORM_TYPE);
    };
    let unit_index = u64::from(sample_position / unit_samples);
    let skip_samples = sample_position % unit_samples;
    let units = u64::from(skip_samples)
        .saturating_add(u64::from(samples))
        .div_ceil(u64::from(unit_samples));
    let block = ngs2_waveform_block_bytes(
        unit_index.saturating_mul(u64::from(unit_size)),
        units.saturating_mul(u64::from(unit_size)),
        skip_samples,
        samples,
    );
    write_guest_bytes(out_block, &block);
    0
}

/// Chunk identifiers of the container this build recognizes, little-endian as they appear.
const RIFF_MAGIC: u32 = 0x4646_4952;
const RIFF_WAVE_MAGIC: u32 = 0x4556_4157;
const RIFF_FORMAT_CHUNK: u32 = 0x2074_6d66;
const RIFF_DATA_CHUNK: u32 = 0x6174_6164;
/// Bytes the container's leading identifiers occupy.
const RIFF_HEADER_SIZE: u64 = 12;
/// Chunks the walk visits before the container is treated as malformed.
const RIFF_CHUNK_LIMIT: usize = 64;
/// Sample encodings a format chunk may name.
const RIFF_FORMAT_PCM: u16 = 1;
const RIFF_FORMAT_FLOAT: u16 = 3;
const RIFF_FORMAT_EXTENSIBLE: u16 = 0xfffe;

/// One recognized container, reduced to what a waveform information block needs.
#[derive(Clone, Copy, Debug)]
struct RiffWaveform {
    waveform_type: u32,
    channels: u32,
    sample_rate: u32,
    unit_size: u32,
    data_offset: u32,
    data_size: u32,
}

/// Reads a bounded word from a guest blob, or `None` if it lies outside the blob.
fn blob_u32(data: u64, size: u64, offset: u64) -> Option<u32> {
    let end = offset.checked_add(4)?;
    let address = data.checked_add(offset)?;
    (end <= size && process_guest_range_is_readable(address, 4)).then(|| read_guest_u32(address))
}

/// Reads a bounded half-word from a guest blob, or `None` if it lies outside the blob.
fn blob_u16(data: u64, size: u64, offset: u64) -> Option<u16> {
    let end = offset.checked_add(2)?;
    let address = data.checked_add(offset)?;
    (end <= size && process_guest_range_is_readable(address, 2)).then(|| read_guest_u16(address))
}

/// Returns the waveform type a sample encoding and width name.
fn riff_waveform_type(format_tag: u16, bits: u16) -> Option<u32> {
    match (format_tag, bits) {
        (RIFF_FORMAT_PCM, 8) => Some(NGS2_WAVEFORM_TYPE_PCM_U8),
        (RIFF_FORMAT_PCM, 16) => Some(NGS2_WAVEFORM_TYPE_PCM_I16L),
        (RIFF_FORMAT_PCM, 24) => Some(NGS2_WAVEFORM_TYPE_PCM_I24L),
        (RIFF_FORMAT_PCM, 32) => Some(NGS2_WAVEFORM_TYPE_PCM_I32L),
        (RIFF_FORMAT_FLOAT, 32) => Some(NGS2_WAVEFORM_TYPE_PCM_F32L),
        (RIFF_FORMAT_FLOAT, 64) => Some(NGS2_WAVEFORM_TYPE_PCM_F64L),
        _ => None,
    }
}

/// Reads the format chunk at `offset`, whose payload runs for `length` bytes.
fn riff_format_chunk(
    data: u64,
    size: u64,
    offset: u64,
    length: u64,
) -> Result<(u32, u32, u32, u32), u32> {
    const EXTENSIBLE_SUBFORMAT_OFFSET: u64 = 24;
    const MINIMUM_FORMAT_CHUNK: u64 = 16;

    if length < MINIMUM_FORMAT_CHUNK {
        return Err(NGS2_ERROR_INVALID_WAVEFORM_FORMAT);
    }
    let (Some(tag), Some(channels), Some(rate), Some(bits)) = (
        blob_u16(data, size, offset),
        blob_u16(data, size, offset + 2),
        blob_u32(data, size, offset + 4),
        blob_u16(data, size, offset + 14),
    ) else {
        return Err(NGS2_ERROR_INVALID_WAVEFORM_DATA);
    };
    let tag = if tag == RIFF_FORMAT_EXTENSIBLE {
        blob_u16(data, size, offset + EXTENSIBLE_SUBFORMAT_OFFSET)
            .ok_or(NGS2_ERROR_INVALID_WAVEFORM_DATA)?
    } else {
        tag
    };
    let waveform_type = riff_waveform_type(tag, bits).ok_or(NGS2_ERROR_UNKNOWN_WAVEFORM_FORMAT)?;
    let channels = u32::from(channels);
    if channels == 0
        || channels > NGS2_MAX_VOICE_CHANNELS
        || rate == 0
        || rate > NGS2_MAX_SAMPLE_RATE
    {
        return Err(NGS2_ERROR_INVALID_WAVEFORM_FORMAT);
    }
    let unit_size = u32::from(bits / 8) * channels;
    if unit_size == 0 {
        return Err(NGS2_ERROR_INVALID_WAVEFORM_FORMAT);
    }
    Ok((waveform_type, channels, rate, unit_size))
}

/// Recognizes a container and reduces it to the fields a waveform information block reports.
fn riff_parse_waveform(data: u64, size: u64) -> Result<RiffWaveform, u32> {
    if blob_u32(data, size, 0) != Some(RIFF_MAGIC)
        || blob_u32(data, size, 8) != Some(RIFF_WAVE_MAGIC)
    {
        return Err(NGS2_ERROR_UNKNOWN_WAVEFORM_FORMAT);
    }
    let mut format = None;
    let mut payload = None;
    let mut offset = RIFF_HEADER_SIZE;
    for _ in 0..RIFF_CHUNK_LIMIT {
        let (Some(id), Some(length)) = (
            blob_u32(data, size, offset),
            blob_u32(data, size, offset + 4),
        ) else {
            break;
        };
        let body = offset + 8;
        let length = u64::from(length);
        match id {
            RIFF_FORMAT_CHUNK => format = Some(riff_format_chunk(data, size, body, length)?),
            RIFF_DATA_CHUNK => payload = Some((body, length.min(size.saturating_sub(body)))),
            _ => {}
        }
        offset = body.saturating_add(length).saturating_add(length & 1);
        if offset >= size {
            break;
        }
    }
    let (Some((waveform_type, channels, sample_rate, unit_size)), Some((body, length))) =
        (format, payload)
    else {
        return Err(NGS2_ERROR_INVALID_WAVEFORM_DATA);
    };
    let (Ok(data_offset), Ok(data_size)) = (u32::try_from(body), u32::try_from(length)) else {
        return Err(NGS2_ERROR_INVALID_WAVEFORM_DATA);
    };
    Ok(RiffWaveform {
        waveform_type,
        channels,
        sample_rate,
        unit_size,
        data_offset,
        data_size,
    })
}

/// Serializes a recognized container into a waveform information block.
fn ngs2_waveform_info_bytes(waveform: RiffWaveform) -> [u8; NGS2_WAVEFORM_INFO_SIZE] {
    let samples = waveform.data_size / waveform.unit_size;
    let mut info = [0_u8; NGS2_WAVEFORM_INFO_SIZE];
    let mut field = |offset: usize, value: u32| {
        info[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    };
    field(0, waveform.waveform_type);
    field(4, waveform.channels);
    field(8, waveform.sample_rate);
    field(24, waveform.data_offset);
    field(28, waveform.data_size);
    field(40, samples);
    field(44, waveform.unit_size);
    field(48, 1);
    field(52, 1);
    field(56, waveform.unit_size);
    field(60, 1);
    field(68, 1);
    let block = ngs2_waveform_block_bytes(0, u64::from(waveform.data_size), 0, samples);
    info[NGS2_WAVEFORM_INFO_BLOCK_OFFSET..NGS2_WAVEFORM_INFO_BLOCK_OFFSET + block.len()]
        .copy_from_slice(&block);
    info
}

/// Recognizes an in-memory waveform container and reports what it holds.
fn ngs2_parse_waveform_data(arguments: [u64; 6]) -> u64 {
    let (data, size, out_info) = (arguments[0], arguments[1], arguments[2]);
    if !guest_object_is_reachable(out_info, NGS2_WAVEFORM_INFO_SIZE as u64) {
        return sce_kernel_error(NGS2_ERROR_INVALID_OUT_ADDRESS);
    }
    if data == 0 || size < RIFF_HEADER_SIZE {
        return sce_kernel_error(NGS2_ERROR_INVALID_WAVEFORM_DATA);
    }
    match riff_parse_waveform(data, size) {
        Ok(waveform) => {
            write_guest_bytes(out_info, &ngs2_waveform_info_bytes(waveform));
            0
        }
        Err(error) => sce_kernel_error(error),
    }
}

/// Error the audio-job-manager entry points here add to the ones the main table already reports.
const SCE_AJM_ERROR_OUT_OF_RESOURCES: u32 = 0x8093_0007;

/// Bytes one page of memory published to the audio coprocessor covers.
const AJM_MEMORY_PAGE_SIZE: u64 = 2 * 1024 * 1024;
/// Pages one context may publish.
///
/// No console ceiling is published for this. The bound keeps a runaway caller from growing the
/// region table without limit, and reports the same exhaustion error the hardware would.
const AJM_MEMORY_PAGE_LIMIT: u64 = 512;

/// Regions each context has published, keyed by context and starting address.
static AJM_REGIONS: OnceLock<Mutex<BTreeMap<(u32, u64), u64>>> = OnceLock::new();

/// Returns the process-wide table of published regions.
fn ajm_regions() -> std::sync::MutexGuard<'static, BTreeMap<(u32, u64), u64>> {
    AJM_REGIONS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Declares a region of guest memory available to the audio coprocessor.
fn ajm_memory_register(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (context, region, pages) = (low_u32(arguments[0]), arguments[1], arguments[2]);
    if !state.ajm_contexts.contains_key(&context) {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
    }
    let Some(length) = pages.checked_mul(AJM_MEMORY_PAGE_SIZE) else {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    };
    if region == 0
        || pages == 0
        || !region.is_multiple_of(AJM_MEMORY_PAGE_SIZE)
        || region.checked_add(length).is_none()
    {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    let mut regions = ajm_regions();
    // A finalized context takes its regions with it; the table is only consulted here.
    regions.retain(|&(owner, _), _| state.ajm_contexts.contains_key(&owner));
    let mut published = 0_u64;
    for (&(owner, start), &count) in regions.iter() {
        if owner != context {
            continue;
        }
        published = published.saturating_add(count);
        let end = start.saturating_add(count.saturating_mul(AJM_MEMORY_PAGE_SIZE));
        if region < end && start < region.saturating_add(length) {
            return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
        }
    }
    if published.saturating_add(pages) > AJM_MEMORY_PAGE_LIMIT {
        return sce_kernel_error(SCE_AJM_ERROR_OUT_OF_RESOURCES);
    }
    regions.insert((context, region), pages);
    0
}

/// Withdraws a region previously published to the audio coprocessor.
fn ajm_memory_unregister(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
    let (context, region) = (low_u32(arguments[0]), arguments[1]);
    if !state.ajm_contexts.contains_key(&context) {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_CONTEXT);
    }
    if region == 0 {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    if ajm_regions().remove(&(context, region)).is_none() {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    0
}

/// Returns the description a code carries.
fn ajm_error_text(code: i32) -> &'static str {
    match code.cast_unsigned() {
        0 => "no error",
        0x8093_0001 => "unknown error",
        0x8093_0002 => "invalid context",
        0x8093_0003 => "invalid instance",
        0x8093_0004 => "invalid batch",
        0x8093_0005 => "invalid parameter",
        0x8093_0006 => "out of memory",
        0x8093_0007 => "out of resources",
        0x8093_0008 => "codec not supported",
        0x8093_0009 => "codec already registered",
        0x8093_000a => "codec not registered",
        0x8093_000b => "wrong revision flag",
        0x8093_000c => "flag not supported",
        0x8093_000d => "busy",
        0x8093_000e => "bad priority",
        0x8093_000f => "in progress",
        0x8093_0010 => "retry",
        0x8093_0011 => "malformed batch",
        0x8093_0012 => "job creation failed",
        0x8093_0013 => "invalid opcode",
        0x8093_0014 => "priority violation",
        0x8093_0015 => "buffer too big",
        0x8093_0016 => "invalid address",
        0x8093_0017 => "cancelled",
        0x8093_0018 => "memory already registered",
        0x8093_0019 => "memory not registered",
        0x8093_001a => "instance memory already extended",
        _ => "unrecognized error code",
    }
}

/// Descriptions already published to the guest, so one code always reports one address.
static AJM_ERROR_STRINGS: OnceLock<Mutex<BTreeMap<i32, u64>>> = OnceLock::new();

/// Reports a readable description of an error code.
fn ajm_str_error(code: u64) -> u64 {
    let code = low_i32(code);
    let mut published = AJM_ERROR_STRINGS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(&address) = published.get(&code) {
        return address;
    }
    let text = ajm_error_text(code);
    let length = u64::try_from(text.len()).unwrap_or(0) + 1;
    let address = guest_malloc(length);
    if address == 0 {
        return 0;
    }
    let mut bytes = text.as_bytes().to_vec();
    bytes.push(0);
    write_guest_bytes(address, &bytes);
    published.insert(code, address);
    address
}

/// Bytes one `SceAjmDecMp3ParseFrame` occupies.
const MP3_PARSE_FRAME_SIZE: usize = 40;
/// Bytes of a caller's bitstream this build searches for a frame header.
const MP3_SEARCH_WINDOW: u64 = 8192;
/// Bitrates, in kilobits per second, indexed by the header's bitrate field.
const MP3_BITRATES_VERSION1: [u32; 16] = [
    0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0,
];
const MP3_BITRATES_VERSION2: [u32; 16] = [
    0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0,
];
/// Sampling rates of the first version, which the later ones halve and quarter.
const MP3_SAMPLE_RATES: [u32; 3] = [44_100, 48_000, 32_000];
/// Samples one frame carries in each version.
const MP3_VERSION1_FRAME_SAMPLES: u32 = 1152;
const MP3_VERSION2_FRAME_SAMPLES: u32 = 576;
/// Layer field value that selects layer III; this entry point parses no other layer.
const MP3_LAYER_III: u8 = 1;
/// Channel mode that selects a single channel.
const MP3_MODE_MONO: u8 = 3;
/// Original-file-length header kinds a frame may carry.
const MP3_OFL_TYPE_NONE: u32 = 0;
const MP3_OFL_TYPE_LAME: u32 = 1;
const MP3_OFL_TYPE_VBRI: u32 = 2;
/// Byte offset of a `VBRI` header from the start of the frame.
const MP3_VBRI_OFFSET: usize = 36;

/// One parsed frame header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Mp3Frame {
    size: u32,
    channels: u32,
    samples: u32,
    bitrate: u32,
    sample_rate: u32,
    side_info: usize,
}

/// Decodes a frame header, or returns `None` if the four bytes do not open a layer III frame.
fn mp3_parse_header(header: &[u8]) -> Option<Mp3Frame> {
    const SYNC_MASK: u8 = 0xe0;
    const VERSION_RESERVED: u8 = 1;
    const VERSION_1: u8 = 3;
    const VERSION_2: u8 = 2;

    if header.len() < 4 || header[0] != 0xff || header[1] & SYNC_MASK != SYNC_MASK {
        return None;
    }
    let version = (header[1] >> 3) & 0x3;
    let layer = (header[1] >> 1) & 0x3;
    if version == VERSION_RESERVED || layer != MP3_LAYER_III {
        return None;
    }
    let bitrate_index = usize::from((header[2] >> 4) & 0xf);
    let rate_index = usize::from((header[2] >> 2) & 0x3);
    let padding = u32::from((header[2] >> 1) & 0x1);
    let mono = (header[3] >> 6) & 0x3 == MP3_MODE_MONO;
    if rate_index >= MP3_SAMPLE_RATES.len() {
        return None;
    }
    let first_version = version == VERSION_1;
    let bitrate = if first_version {
        MP3_BITRATES_VERSION1[bitrate_index]
    } else {
        MP3_BITRATES_VERSION2[bitrate_index]
    };
    if bitrate == 0 {
        return None;
    }
    let sample_rate = MP3_SAMPLE_RATES[rate_index]
        / match version {
            VERSION_1 => 1,
            VERSION_2 => 2,
            _ => 4,
        };
    let samples = if first_version {
        MP3_VERSION1_FRAME_SAMPLES
    } else {
        MP3_VERSION2_FRAME_SAMPLES
    };
    let bitrate = bitrate * 1000;
    let side_info = match (first_version, mono) {
        (true, false) => 32,
        (true, true) | (false, false) => 17,
        (false, true) => 9,
    };
    Some(Mp3Frame {
        size: samples / 8 * bitrate / sample_rate + padding,
        channels: if mono { 1 } else { 2 },
        samples,
        bitrate,
        sample_rate,
        side_info,
    })
}

/// Original-file-length data recovered from a frame's ancillary bytes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Mp3OriginalFileLength {
    kind: u32,
    encoder_delay: u32,
    frames: u32,
}

/// Reads a big-endian word from a frame at `offset`, if the frame carries one there.
fn mp3_be_u32(frame: &[u8], offset: usize) -> Option<u32> {
    let bytes = frame.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_be_bytes(bytes.try_into().ok()?))
}

/// Recovers the original-file-length header a frame carries, if any.
fn mp3_parse_original_file_length(frame: &[u8], header: Mp3Frame) -> Mp3OriginalFileLength {
    const XING_FRAMES_FLAG: u32 = 1 << 0;
    const XING_BYTES_FLAG: u32 = 1 << 1;
    const XING_TOC_FLAG: u32 = 1 << 2;
    const XING_QUALITY_FLAG: u32 = 1 << 3;
    const XING_TOC_SIZE: usize = 100;
    /// Distance from the encoder tag's magic to the twelve-bit encoder delay.
    const LAME_DELAY_OFFSET: usize = 21;
    const VBRI_DELAY_OFFSET: usize = 6;
    const VBRI_FRAMES_OFFSET: usize = 14;

    let xing = 4 + header.side_info;
    if matches!(frame.get(xing..xing + 4), Some(b"Xing" | b"Info")) {
        let flags = mp3_be_u32(frame, xing + 4).unwrap_or(0);
        let mut cursor = xing + 8;
        let mut frames = 0;
        if flags & XING_FRAMES_FLAG != 0 {
            frames = mp3_be_u32(frame, cursor).unwrap_or(0);
            cursor += 4;
        }
        for (flag, length) in [
            (XING_BYTES_FLAG, 4),
            (XING_TOC_FLAG, XING_TOC_SIZE),
            (XING_QUALITY_FLAG, 4),
        ] {
            if flags & flag != 0 {
                cursor += length;
            }
        }
        let encoder_delay = frame
            .get(cursor + LAME_DELAY_OFFSET..cursor + LAME_DELAY_OFFSET + 2)
            .map_or(0, |bytes| {
                (u32::from(bytes[0]) << 4) | u32::from(bytes[1] >> 4)
            });
        return Mp3OriginalFileLength {
            kind: MP3_OFL_TYPE_LAME,
            encoder_delay,
            frames,
        };
    }
    if frame.get(MP3_VBRI_OFFSET..MP3_VBRI_OFFSET + 4) == Some(b"VBRI") {
        return Mp3OriginalFileLength {
            kind: MP3_OFL_TYPE_VBRI,
            encoder_delay: frame
                .get(MP3_VBRI_OFFSET + VBRI_DELAY_OFFSET..MP3_VBRI_OFFSET + VBRI_DELAY_OFFSET + 2)
                .map_or(0, |bytes| {
                    u32::from(u16::from_be_bytes([bytes[0], bytes[1]]))
                }),
            frames: mp3_be_u32(frame, MP3_VBRI_OFFSET + VBRI_FRAMES_OFFSET).unwrap_or(0),
        };
    }
    Mp3OriginalFileLength::default()
}

/// Serializes one parsed frame.
fn mp3_frame_bytes(
    header: Mp3Frame,
    original: Mp3OriginalFileLength,
) -> [u8; MP3_PARSE_FRAME_SIZE] {
    let mut parsed = [0_u8; MP3_PARSE_FRAME_SIZE];
    parsed[..8].copy_from_slice(&u64::from(header.size).to_le_bytes());
    parsed[8..12].copy_from_slice(&header.channels.to_le_bytes());
    parsed[12..16].copy_from_slice(&header.samples.to_le_bytes());
    parsed[16..20].copy_from_slice(&header.bitrate.to_le_bytes());
    parsed[20..24].copy_from_slice(&header.sample_rate.to_le_bytes());
    parsed[24..28].copy_from_slice(&original.encoder_delay.to_le_bytes());
    parsed[28..32].copy_from_slice(&original.frames.to_le_bytes());
    parsed[32..36].copy_from_slice(&original.frames.saturating_mul(header.samples).to_le_bytes());
    parsed[36..40].copy_from_slice(&original.kind.to_le_bytes());
    parsed
}

/// Reports the geometry of the first frame a bitstream carries.
fn ajm_dec_mp3_parse_frame(arguments: [u64; 6]) -> u64 {
    let (bitstream, length, parse_original, out_frame) =
        (arguments[0], arguments[1], arguments[2], arguments[3]);
    if bitstream == 0
        || length == 0
        || !guest_object_is_reachable(out_frame, MP3_PARSE_FRAME_SIZE as u64)
    {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    let window = usize::try_from(length.min(MP3_SEARCH_WINDOW)).unwrap_or(0);
    if !process_guest_range_is_readable(bitstream, window) {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    }
    let mut bytes = vec![0_u8; window];
    read_guest_bytes(bitstream, &mut bytes);
    let Some((start, header)) = (0..window.saturating_sub(3))
        .find_map(|offset| mp3_parse_header(&bytes[offset..]).map(|header| (offset, header)))
    else {
        return sce_kernel_error(SCE_AJM_ERROR_INVALID_PARAMETER);
    };
    let original = if parse_original == 0 {
        Mp3OriginalFileLength {
            kind: MP3_OFL_TYPE_NONE,
            ..Mp3OriginalFileLength::default()
        }
    } else {
        mp3_parse_original_file_length(&bytes[start..], header)
    };
    write_guest_bytes(out_frame, &mp3_frame_bytes(header, original));
    0
}

/// Ports one synchronized output may name.
const AUDIO_OUT_OUTPUTS_MAX: u64 = 8;
/// Bytes one `SceAudioOutOutputParam` occupies, and the offset of its buffer pointer.
const AUDIO_OUT_OUTPUT_PARAM_SIZE: u64 = 16;
const AUDIO_OUT_OUTPUT_PARAM_BUFFER_OFFSET: u64 = 8;
/// Bytes one `SceAudioOutPortState` occupies.
const AUDIO_OUT_PORT_STATE_SIZE: usize = 16;
/// Error reported for a port count outside the range a synchronized output accepts.
const AUDIO_OUT_ERROR_INVALID_ARG: u32 = 0x8026_0015;
/// Error reported for a mix level outside the range the pad speaker accepts.
const AUDIO_OUT_ERROR_INVALID_MIXLEVEL: u32 = 0x8026_0014;
/// Loudest mix level the pad speaker accepts; unity gain is half of it.
const AUDIO_OUT_MIXLEVEL_MAXIMUM: i32 = 32_768;

/// Mix level each port applies to the pad speaker it is additionally routed to.
static AUDIO_OUT_PAD_SPEAKER_MIX: OnceLock<Mutex<BTreeMap<i32, i32>>> = OnceLock::new();

/// Returns the process-wide audio-output state.
fn audio_out_state() -> std::sync::MutexGuard<'static, audio::AudioOutState> {
    AUDIO_OUT
        .get_or_init(|| Mutex::new(audio::AudioOutState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Reports whether a port handle names an open port.
///
/// The mixer owns its port table privately, so the port-state query is what answers this.
fn audio_out_port_is_open(state: &audio::AudioOutState, handle: i32) -> bool {
    let mut probe = [0_u8; AUDIO_OUT_PORT_STATE_SIZE];
    let address = probe.as_mut_ptr() as u64;
    state
        .get_port_state([u64::from(handle.cast_unsigned()), address, 0, 0, 0, 0])
        .is_ok()
}

/// Outputs one configured buffer on each of several ports at once.
///
/// Like the single-port output this blocks until the ports are ready for their next buffer, and
/// waits once for the longest of them rather than serially.
fn audio_out_outputs(arguments: [u64; 6]) -> u64 {
    let (params, count) = (arguments[0], arguments[1]);
    if params == 0 {
        return audio_out_error(audio::AudioOutError::InvalidPointer);
    }
    if count == 0 || count > AUDIO_OUT_OUTPUTS_MAX {
        return sce_kernel_error(AUDIO_OUT_ERROR_INVALID_ARG);
    }
    let span = usize::try_from(count * AUDIO_OUT_OUTPUT_PARAM_SIZE).unwrap_or(usize::MAX);
    if !process_guest_range_is_readable(params, span) {
        return audio_out_error(audio::AudioOutError::InvalidPointer);
    }
    let entries = (0..count)
        .map(|index| {
            let entry = params + index * AUDIO_OUT_OUTPUT_PARAM_SIZE;
            (
                read_guest_u32(entry).cast_signed(),
                read_guest_u64(entry + AUDIO_OUT_OUTPUT_PARAM_BUFFER_OFFSET),
            )
        })
        .collect::<Vec<_>>();
    let mut state = audio_out_state();
    if !entries
        .iter()
        .all(|&(handle, _)| audio_out_port_is_open(&state, handle))
    {
        return audio_out_error(audio::AudioOutError::InvalidPort);
    }
    let mut delay = Duration::ZERO;
    for (handle, buffer) in entries {
        match state.output([u64::from(handle.cast_unsigned()), buffer, 0, 0, 0, 0]) {
            Ok(port_delay) => delay = delay.max(port_delay),
            Err(error) => return audio_out_error(error),
        }
    }
    drop(state);
    if !delay.is_zero() {
        std::thread::sleep(delay);
    }
    0
}

/// Sets the level at which a pad-speaker port is mixed into a connected headset.
///
/// `REmu` mixes every port to one host device and emits no separate controller-speaker route, so
/// the level is recorded for the port and changes nothing a listener hears.
fn audio_out_set_mix_level_pad_spk(arguments: [u64; 6]) -> u64 {
    let (handle, mix_level) = (low_i32(arguments[0]), low_i32(arguments[1]));
    if !(0..=AUDIO_OUT_MIXLEVEL_MAXIMUM).contains(&mix_level) {
        return sce_kernel_error(AUDIO_OUT_ERROR_INVALID_MIXLEVEL);
    }
    if !audio_out_port_is_open(&audio_out_state(), handle) {
        return audio_out_error(audio::AudioOutError::InvalidPort);
    }
    AUDIO_OUT_PAD_SPEAKER_MIX
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(handle, mix_level);
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AjmContextState;

    const SUCCESS: u64 = 0;
    const VIDEO_OUT_HANDLE: u64 = 1;

    /// Packs a leading argument list into the six-register form a dispatch receives.
    fn args(values: &[u64]) -> [u64; 6] {
        let mut arguments = [0_u64; 6];
        arguments[..values.len()].copy_from_slice(values);
        arguments
    }

    /// Reports the address of a guest object a test owns.
    fn address<T>(value: &T) -> u64 {
        std::ptr::from_ref(value) as u64
    }

    /// Reports the address of a guest object a call under test writes through.
    fn address_mut<T>(value: &mut T) -> u64 {
        std::ptr::from_mut(value) as u64
    }

    /// Reports the error result a code produces when the guest reads it as a signed word.
    fn error(code: u32) -> u64 {
        sce_kernel_error(code)
    }

    /// Returns a state whose display port is open with one registered buffer.
    fn open_display() -> ProbeDispatchState {
        let mut state = ProbeDispatchState::default();
        state.video_out.is_open = true;
        state.video_out.width = 1280;
        state.video_out.height = 720;
        state.video_out.display_buffers = vec![0x1000, 0x2000];
        state.video_out.display_buffer_attributes = vec![None, None];
        state
    }

    /// Builds a buffer attribute block.
    fn buffer_attribute(
        tiling: u32,
        width: u32,
        height: u32,
    ) -> [u8; VIDEO_OUT_BUFFER_ATTRIBUTE2_SIZE] {
        let mut attribute = [0_u8; VIDEO_OUT_BUFFER_ATTRIBUTE2_SIZE];
        attribute[4..8].copy_from_slice(&tiling.to_le_bytes());
        attribute[12..16].copy_from_slice(&width.to_le_bytes());
        attribute[16..20].copy_from_slice(&height.to_le_bytes());
        attribute[32..40].copy_from_slice(&0x8000_0000_2200_0000_u64.to_le_bytes());
        attribute
    }

    #[test]
    fn closing_the_display_port_releases_its_buffers_and_refuses_a_second_close() {
        let mut state = open_display();
        assert_eq!(
            dispatch("uquVH4-Du78", args(&[VIDEO_OUT_HANDLE]), &mut state),
            Some(SUCCESS)
        );
        assert!(!state.video_out.is_open);
        assert!(state.video_out.display_buffers.is_empty());
        assert!(state.video_out.display_buffer_attributes.is_empty());
        assert_eq!(
            dispatch("uquVH4-Du78", args(&[VIDEO_OUT_HANDLE]), &mut state),
            Some(error(VIDEO_OUT_ERROR_INVALID_HANDLE))
        );
    }

    #[test]
    fn a_display_call_on_a_handle_no_port_owns_is_rejected() {
        let mut state = open_display();
        for nid in [
            "uquVH4-Du78",
            "N5KDtkIjjJ4",
            "1FZBKy8HeNU",
            "j6RaAUlaLv0",
            "T4ucGB8CsnM",
        ] {
            assert_eq!(
                dispatch(nid, args(&[2, 0]), &mut state),
                Some(error(VIDEO_OUT_ERROR_INVALID_HANDLE)),
                "{nid} accepted a handle no port owns"
            );
        }
    }

    #[test]
    fn unregistering_the_buffer_set_clears_it_once_and_refuses_an_unknown_set() {
        let mut state = open_display();
        assert_eq!(
            dispatch("N5KDtkIjjJ4", args(&[VIDEO_OUT_HANDLE, 1]), &mut state),
            Some(error(VIDEO_OUT_ERROR_INVALID_INDEX))
        );
        assert_eq!(
            dispatch("N5KDtkIjjJ4", args(&[VIDEO_OUT_HANDLE, 0]), &mut state),
            Some(SUCCESS)
        );
        assert!(state.video_out.display_buffers.is_empty());
        assert_eq!(
            dispatch("N5KDtkIjjJ4", args(&[VIDEO_OUT_HANDLE, 0]), &mut state),
            Some(error(VIDEO_OUT_ERROR_INVALID_INDEX))
        );
    }

    #[test]
    fn changing_the_buffer_attributes_retargets_every_registered_buffer() {
        let mut state = open_display();
        let attribute = buffer_attribute(VIDEO_OUT_TILING_MODE_LINEAR, 1920, 1080);
        assert_eq!(
            dispatch(
                "HuViW4HnrOw",
                args(&[VIDEO_OUT_HANDLE, 0, address(&attribute), 0]),
                &mut state
            ),
            Some(SUCCESS)
        );
        assert_eq!(state.video_out.width, 1920);
        assert_eq!(state.video_out.height, 1080);
        assert_eq!(state.video_out.tiling_mode, VIDEO_OUT_TILING_MODE_LINEAR);
        assert_eq!(state.video_out.pixel_format, 0x8000_0000_2200_0000);
        assert_eq!(
            state.video_out.display_buffer_attributes,
            vec![
                Some(VideoOutBufferAttributes {
                    width: 1920,
                    height: 1080,
                    pitch: 1920,
                    pixel_format: 0x8000_0000_2200_0000,
                    tiling_mode: VIDEO_OUT_TILING_MODE_LINEAR,
                });
                2
            ]
        );
    }

    #[test]
    fn a_malformed_buffer_attribute_change_is_rejected_rather_than_applied() {
        let mut state = open_display();
        let unusable_tiling = buffer_attribute(2, 1920, 1080);
        let no_resolution = buffer_attribute(VIDEO_OUT_TILING_MODE_TILE, 0, 1080);
        for (attribute, expected) in [
            (
                address(&unusable_tiling),
                VIDEO_OUT_ERROR_INVALID_TILING_MODE,
            ),
            (address(&no_resolution), VIDEO_OUT_ERROR_INVALID_RESOLUTION),
            (0, VIDEO_OUT_ERROR_INVALID_ADDRESS),
        ] {
            assert_eq!(
                dispatch(
                    "HuViW4HnrOw",
                    args(&[VIDEO_OUT_HANDLE, 0, attribute, 0]),
                    &mut state
                ),
                Some(error(expected))
            );
        }
        assert_eq!(state.video_out.width, 1280);
        assert_eq!(
            dispatch(
                "HuViW4HnrOw",
                args(&[VIDEO_OUT_HANDLE, 1, address(&no_resolution), 0]),
                &mut state
            ),
            Some(error(VIDEO_OUT_ERROR_INVALID_INDEX))
        );
    }

    #[test]
    fn the_vblank_status_reports_a_free_running_counter_in_its_documented_layout() {
        let mut state = open_display();
        let mut status = [0xff_u8; 0x28];
        let status_address = address_mut(&mut status);
        assert_eq!(
            dispatch(
                "1FZBKy8HeNU",
                args(&[VIDEO_OUT_HANDLE, status_address]),
                &mut state
            ),
            Some(SUCCESS)
        );
        let count = u64::from_le_bytes(status[..8].try_into().expect("the count is eight bytes"));
        let process_time =
            u64::from_le_bytes(status[8..16].try_into().expect("the time is eight bytes"));
        assert!(
            process_time <= kernel_process_time(false),
            "the reported vblank must already have happened"
        );
        assert_eq!(status[16..24], [0; 8], "the reserved word must be cleared");
        assert_eq!(
            status[32..40],
            [0; 8],
            "flags, phase and padding must be zero"
        );

        assert_eq!(
            dispatch("j6RaAUlaLv0", args(&[VIDEO_OUT_HANDLE]), &mut state),
            Some(SUCCESS)
        );
        assert_eq!(
            dispatch(
                "1FZBKy8HeNU",
                args(&[VIDEO_OUT_HANDLE, status_address]),
                &mut state
            ),
            Some(SUCCESS)
        );
        let waited = u64::from_le_bytes(status[..8].try_into().expect("the count is eight bytes"));
        assert!(
            waited > count,
            "a completed vblank wait must advance the counter: {count} then {waited}"
        );
        assert_eq!(
            dispatch("1FZBKy8HeNU", args(&[VIDEO_OUT_HANDLE, 0]), &mut state),
            Some(error(VIDEO_OUT_ERROR_INVALID_ADDRESS))
        );
    }

    #[test]
    fn a_delivered_event_reports_the_display_event_it_carries() {
        let mut state = open_display();
        let mut event = [0_u8; KERNEL_EVENT_SIZE];
        event[12..16].copy_from_slice(&1_u32.to_le_bytes());
        assert_eq!(
            dispatch("U2JJtSqNKZI", args(&[address(&event)]), &mut state),
            Some(1)
        );
        event[12..16].copy_from_slice(&0_u32.to_le_bytes());
        assert_eq!(
            dispatch("U2JJtSqNKZI", args(&[address(&event)]), &mut state),
            Some(SUCCESS)
        );
        assert_eq!(
            dispatch("U2JJtSqNKZI", args(&[0]), &mut state),
            Some(error(VIDEO_OUT_ERROR_INVALID_ADDRESS))
        );
    }

    #[test]
    fn output_options_are_cleared_and_a_null_options_block_is_rejected() {
        let mut state = open_display();
        let mut options = [0xff_u8; 0x40];
        assert_eq!(
            dispatch(
                "+I4K03i3EL0",
                args(&[address_mut(&mut options)]),
                &mut state
            ),
            Some(SUCCESS)
        );
        assert_eq!(options, [0; 0x40]);
        assert_eq!(
            dispatch("+I4K03i3EL0", args(&[0]), &mut state),
            Some(error(VIDEO_OUT_ERROR_INVALID_ADDRESS))
        );
    }

    #[test]
    fn only_the_default_output_mode_is_reported_as_available() {
        let mut state = open_display();
        assert_eq!(
            dispatch(
                "Nv8c-Kb+DUM",
                args(&[VIDEO_OUT_HANDLE, 1, 0, 0, 0]),
                &mut state
            ),
            Some(VIDEO_OUT_TRUE)
        );
        assert_eq!(
            dispatch(
                "Nv8c-Kb+DUM",
                args(&[VIDEO_OUT_HANDLE, 0xd000_000a, 0, 0, 0]),
                &mut state
            ),
            Some(VIDEO_OUT_FALSE)
        );
        assert_eq!(
            dispatch(
                "Nv8c-Kb+DUM",
                args(&[VIDEO_OUT_HANDLE, 1, 0, 8, 0]),
                &mut state
            ),
            Some(error(VIDEO_OUT_ERROR_INVALID_VALUE))
        );
        assert_eq!(
            dispatch(
                "Nv8c-Kb+DUM",
                args(&[VIDEO_OUT_HANDLE, 1, 0, 0, 1]),
                &mut state
            ),
            Some(error(VIDEO_OUT_ERROR_INVALID_VALUE))
        );
    }

    #[test]
    fn colour_adjustment_requires_the_declared_size_and_a_usable_gamma() {
        let mut state = open_display();
        let settings = [1.0_f32, 0.5, 2.0, f32::from_bits(0)];
        let broken = [f32::NAN, 1.0, 1.0, 0.0];
        assert_eq!(
            dispatch(
                "pv9CI5VC+R0",
                args(&[VIDEO_OUT_HANDLE, address(&settings), 16]),
                &mut state
            ),
            Some(SUCCESS)
        );
        assert_eq!(
            dispatch(
                "pv9CI5VC+R0",
                args(&[VIDEO_OUT_HANDLE, address(&settings), 8]),
                &mut state
            ),
            Some(error(VIDEO_OUT_ERROR_INVALID_VALUE))
        );
        assert_eq!(
            dispatch(
                "pv9CI5VC+R0",
                args(&[VIDEO_OUT_HANDLE, address(&broken), 16]),
                &mut state
            ),
            Some(error(VIDEO_OUT_ERROR_INVALID_VALUE))
        );
        assert_eq!(
            dispatch("pv9CI5VC+R0", args(&[VIDEO_OUT_HANDLE, 0, 16]), &mut state),
            Some(error(VIDEO_OUT_ERROR_INVALID_ADDRESS))
        );
    }

    #[test]
    fn releasing_a_pegged_refresh_rate_needs_the_open_port() {
        let mut state = open_display();
        assert_eq!(
            dispatch("T4ucGB8CsnM", args(&[VIDEO_OUT_HANDLE]), &mut state),
            Some(SUCCESS)
        );
        state.video_out.is_open = false;
        assert_eq!(
            dispatch("T4ucGB8CsnM", args(&[VIDEO_OUT_HANDLE]), &mut state),
            Some(error(VIDEO_OUT_ERROR_INVALID_HANDLE))
        );
    }

    /// Builds an allocator whose handlers are addresses the library records but never calls.
    fn buffer_allocator(user_data: u64) -> [u64; 3] {
        [0x1000_0000, 0x2000_0000, user_data]
    }

    /// Builds a system option declaring its own size.
    fn system_option(sample_rate: u32, grain_samples: u32) -> [u8; 0x90] {
        let mut option = [0_u8; 0x90];
        option[..8].copy_from_slice(&NGS2_SYSTEM_OPTION_SIZE.to_le_bytes());
        option[112..116].copy_from_slice(&grain_samples.to_le_bytes());
        option[116..120].copy_from_slice(&sample_rate.to_le_bytes());
        option
    }

    /// Builds a rack option declaring its own size.
    fn rack_option(voices: u32, ports: u32, matrices: u32) -> [u8; 176] {
        let mut option = [0_u8; 176];
        option[..8].copy_from_slice(&NGS2_RACK_OPTION_SIZE.to_le_bytes());
        option[80..84].copy_from_slice(&voices.to_le_bytes());
        option[88..92].copy_from_slice(&matrices.to_le_bytes());
        option[92..96].copy_from_slice(&ports.to_le_bytes());
        option
    }

    /// A system, a rack inside it, and the first voice of that rack.
    struct VoiceFixture {
        state: ProbeDispatchState,
        system: u64,
        rack: u64,
        voice: u64,
    }

    impl VoiceFixture {
        fn new(voices: u32, ports: u32, matrices: u32) -> Self {
            let mut state = ProbeDispatchState::default();
            let allocator = buffer_allocator(0);
            let option = system_option(48_000, 256);
            let mut system = 0_u64;
            assert_eq!(
                dispatch(
                    "mPYgU4oYpuY",
                    args(&[
                        address(&option),
                        address(&allocator),
                        address_mut(&mut system)
                    ]),
                    &mut state
                ),
                Some(SUCCESS)
            );
            let option = rack_option(voices, ports, matrices);
            let mut rack = 0_u64;
            assert_eq!(
                dispatch(
                    "U546k6orxQo",
                    args(&[
                        system,
                        0x1000,
                        address(&option),
                        address(&allocator),
                        address_mut(&mut rack)
                    ]),
                    &mut state
                ),
                Some(SUCCESS)
            );
            let mut voice = 0_u64;
            assert_eq!(
                dispatch(
                    "MwmHz8pAdAo",
                    args(&[rack, 0, address_mut(&mut voice)]),
                    &mut state
                ),
                Some(SUCCESS)
            );
            Self {
                state,
                system,
                rack,
                voice,
            }
        }

        fn state_flags(&mut self) -> u32 {
            let mut flags = 0_u32;
            let voice = self.voice;
            assert_eq!(
                dispatch(
                    "rEh728kXk3w",
                    args(&[voice, address_mut(&mut flags)]),
                    &mut self.state
                ),
                Some(SUCCESS)
            );
            flags
        }
    }

    #[test]
    fn a_created_rack_hands_out_one_stable_handle_per_voice_it_was_sized_for() {
        let mut fixture = VoiceFixture::new(4, 2, 2);
        let mut again = 0_u64;
        assert_eq!(
            dispatch(
                "MwmHz8pAdAo",
                args(&[fixture.rack, 0, address_mut(&mut again)]),
                &mut fixture.state
            ),
            Some(SUCCESS)
        );
        assert_eq!(again, fixture.voice);
        let mut last = 0_u64;
        assert_eq!(
            dispatch(
                "MwmHz8pAdAo",
                args(&[fixture.rack, 3, address_mut(&mut last)]),
                &mut fixture.state
            ),
            Some(SUCCESS)
        );
        assert_ne!(last, fixture.voice);
        assert_eq!(
            dispatch(
                "MwmHz8pAdAo",
                args(&[fixture.rack, 4, address_mut(&mut last)]),
                &mut fixture.state
            ),
            Some(error(NGS2_ERROR_INVALID_VOICE_INDEX))
        );
        assert_eq!(
            dispatch(
                "MwmHz8pAdAo",
                args(&[fixture.rack, 0, 0]),
                &mut fixture.state
            ),
            Some(error(NGS2_ERROR_INVALID_OUT_ADDRESS))
        );
    }

    #[test]
    fn destroying_a_rack_reports_its_allocator_data_and_retires_its_voices() {
        let mut state = ProbeDispatchState::default();
        let allocator = buffer_allocator(0xabcd);
        let option = system_option(48_000, 256);
        let mut system = 0_u64;
        assert_eq!(
            dispatch(
                "mPYgU4oYpuY",
                args(&[
                    address(&option),
                    address(&allocator),
                    address_mut(&mut system)
                ]),
                &mut state
            ),
            Some(SUCCESS)
        );
        let option = rack_option(2, 1, 1);
        let mut rack = 0_u64;
        assert_eq!(
            dispatch(
                "U546k6orxQo",
                args(&[
                    system,
                    0x2000,
                    address(&option),
                    address(&allocator),
                    address_mut(&mut rack)
                ]),
                &mut state
            ),
            Some(SUCCESS)
        );
        let mut voice = 0_u64;
        assert_eq!(
            dispatch(
                "MwmHz8pAdAo",
                args(&[rack, 1, address_mut(&mut voice)]),
                &mut state
            ),
            Some(SUCCESS)
        );
        let mut buffer_info = [0xff_u8; 64];
        assert_eq!(
            dispatch(
                "lCqD7oycmIM",
                args(&[rack, address_mut(&mut buffer_info)]),
                &mut state
            ),
            Some(SUCCESS)
        );
        assert_eq!(
            buffer_info[..56],
            [0; 56],
            "the buffer record must be cleared"
        );
        assert_eq!(
            u64::from_le_bytes(
                buffer_info[56..]
                    .try_into()
                    .expect("user data is eight bytes")
            ),
            0xabcd
        );
        let mut flags = 0_u32;
        assert_eq!(
            dispatch(
                "rEh728kXk3w",
                args(&[voice, address_mut(&mut flags)]),
                &mut state
            ),
            Some(error(NGS2_ERROR_INVALID_VOICE_HANDLE))
        );
        assert_eq!(
            dispatch("lCqD7oycmIM", args(&[rack, 0]), &mut state),
            Some(error(NGS2_ERROR_INVALID_RACK_HANDLE))
        );
        assert_eq!(
            dispatch("u-WrYDaJA3k", args(&[system, 0]), &mut state),
            Some(SUCCESS)
        );
        assert_eq!(
            dispatch("u-WrYDaJA3k", args(&[system, 0]), &mut state),
            Some(error(NGS2_ERROR_INVALID_SYSTEM_HANDLE))
        );
    }

    #[test]
    fn creating_a_system_rejects_a_missing_allocator_and_an_option_it_cannot_honour() {
        let mut state = ProbeDispatchState::default();
        let allocator = buffer_allocator(0);
        let option = system_option(48_000, 256);
        let mut handle = 0_u64;
        assert_eq!(
            dispatch(
                "mPYgU4oYpuY",
                args(&[address(&option), address(&allocator), 0]),
                &mut state
            ),
            Some(error(NGS2_ERROR_INVALID_OUT_ADDRESS))
        );
        assert_eq!(
            dispatch(
                "mPYgU4oYpuY",
                args(&[address(&option), 0, address_mut(&mut handle)]),
                &mut state
            ),
            Some(error(NGS2_ERROR_INVALID_CONTEXT_ALLOCATOR))
        );
        let no_handlers = buffer_allocator(0).map(|_| 0);
        assert_eq!(
            dispatch(
                "mPYgU4oYpuY",
                args(&[
                    address(&option),
                    address(&no_handlers),
                    address_mut(&mut handle)
                ]),
                &mut state
            ),
            Some(error(NGS2_ERROR_INVALID_CONTEXT_ALLOCATOR))
        );
        let unusable_rate = system_option(1_000, 256);
        assert_eq!(
            dispatch(
                "mPYgU4oYpuY",
                args(&[
                    address(&unusable_rate),
                    address(&allocator),
                    address_mut(&mut handle)
                ]),
                &mut state
            ),
            Some(error(NGS2_ERROR_INVALID_SAMPLE_RATE))
        );
        let unusable_grain = system_option(48_000, 100);
        assert_eq!(
            dispatch(
                "mPYgU4oYpuY",
                args(&[
                    address(&unusable_grain),
                    address(&allocator),
                    address_mut(&mut handle)
                ]),
                &mut state
            ),
            Some(error(NGS2_ERROR_INVALID_NUM_GRAIN_SAMPLES))
        );
        let mut short_option = system_option(48_000, 256);
        short_option[..8].copy_from_slice(&8_u64.to_le_bytes());
        assert_eq!(
            dispatch(
                "mPYgU4oYpuY",
                args(&[
                    address(&short_option),
                    address(&allocator),
                    address_mut(&mut handle)
                ]),
                &mut state
            ),
            Some(error(NGS2_ERROR_INVALID_OPTION_SIZE))
        );
    }

    #[test]
    fn creating_a_rack_rejects_an_unknown_identifier_system_or_voice_count() {
        let mut fixture = VoiceFixture::new(1, 1, 1);
        let allocator = buffer_allocator(0);
        let option = rack_option(1, 1, 1);
        let mut rack = 0_u64;
        assert_eq!(
            dispatch(
                "U546k6orxQo",
                args(&[
                    fixture.system,
                    0x1234,
                    address(&option),
                    address(&allocator),
                    address_mut(&mut rack)
                ]),
                &mut fixture.state
            ),
            Some(error(NGS2_ERROR_INVALID_RACK_ID))
        );
        assert_eq!(
            dispatch(
                "U546k6orxQo",
                args(&[
                    0,
                    0x1000,
                    address(&option),
                    address(&allocator),
                    address_mut(&mut rack)
                ]),
                &mut fixture.state
            ),
            Some(error(NGS2_ERROR_INVALID_SYSTEM_HANDLE))
        );
        let no_voices = rack_option(0, 1, 1);
        assert_eq!(
            dispatch(
                "U546k6orxQo",
                args(&[
                    fixture.system,
                    0x1000,
                    address(&no_voices),
                    address(&allocator),
                    address_mut(&mut rack)
                ]),
                &mut fixture.state
            ),
            Some(error(NGS2_ERROR_INVALID_NUM_VOICES))
        );
    }

    #[test]
    fn the_system_option_defaults_to_zero_with_its_own_size() {
        let mut state = ProbeDispatchState::default();
        let mut option = [0xff_u8; 0x90];
        assert_eq!(
            dispatch("AQkj7C0f3PY", args(&[address_mut(&mut option)]), &mut state),
            Some(SUCCESS)
        );
        assert_eq!(
            u64::from_le_bytes(option[..8].try_into().expect("the size is eight bytes")),
            NGS2_SYSTEM_OPTION_SIZE
        );
        assert_eq!(option[8..], [0; 0x88]);
        assert_eq!(
            dispatch("AQkj7C0f3PY", args(&[0]), &mut state),
            Some(error(NGS2_ERROR_INVALID_OUT_ADDRESS))
        );
    }

    /// Builds one voice control entry at `offset` of a chain buffer.
    fn control_entry(chain: &mut [u8], offset: usize, size: u16, next: i16, id: u32) {
        chain[offset..offset + 2].copy_from_slice(&size.to_le_bytes());
        chain[offset + 2..offset + 4].copy_from_slice(&next.to_le_bytes());
        chain[offset + 4..offset + 8].copy_from_slice(&id.to_le_bytes());
    }

    /// Builds one command parameter at `index` of a command array.
    fn command_entry(commands: &mut [u8], index: usize, param_id: u32, value: u64) {
        let entry = index * 16;
        commands[entry..entry + 4].copy_from_slice(&param_id.to_le_bytes());
        commands[entry + 6..entry + 8].copy_from_slice(&1_u16.to_le_bytes());
        commands[entry + 8..entry + 16].copy_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn a_control_chain_kicks_every_entry_it_links_and_leaves_the_voice_playing() {
        let mut fixture = VoiceFixture::new(2, 2, 2);
        let mut chain = [0_u8; 28];
        control_entry(&mut chain, 0, 12, 12, NGS2_VOICE_KICK_EVENT);
        chain[8..12].copy_from_slice(&NGS2_VOICE_EVENT_PLAY.to_le_bytes());
        control_entry(&mut chain, 12, 16, 0, NGS2_VOICE_SET_PORT_VOLUME);
        chain[24..28].copy_from_slice(&1.0_f32.to_bits().to_le_bytes());
        let voice = fixture.voice;
        assert_eq!(
            dispatch(
                "uu94irFOGpA",
                args(&[voice, address(&chain)]),
                &mut fixture.state
            ),
            Some(SUCCESS)
        );
        assert_eq!(
            fixture.state_flags(),
            NGS2_VOICE_STATE_FLAG_IN_USE | NGS2_VOICE_STATE_FLAG_PLAYING
        );

        let mut stop = [0_u8; 12];
        control_entry(&mut stop, 0, 12, 0, NGS2_VOICE_KICK_EVENT);
        stop[8..12].copy_from_slice(&NGS2_VOICE_EVENT_STOP.to_le_bytes());
        assert_eq!(
            dispatch(
                "uu94irFOGpA",
                args(&[voice, address(&stop)]),
                &mut fixture.state
            ),
            Some(SUCCESS)
        );
        assert_eq!(
            fixture.state_flags(),
            NGS2_VOICE_STATE_FLAG_IN_USE | NGS2_VOICE_STATE_FLAG_STOPPED
        );
    }

    #[test]
    fn a_malformed_control_chain_is_rejected_rather_than_walked() {
        let mut fixture = VoiceFixture::new(2, 2, 2);
        let voice = fixture.voice;
        assert_eq!(
            dispatch("uu94irFOGpA", args(&[voice, 0]), &mut fixture.state),
            Some(error(NGS2_ERROR_INVALID_VOICE_CONTROL_ADDRESS))
        );

        let mut unknown = [0_u8; 12];
        control_entry(&mut unknown, 0, 12, 0, 0x99);
        assert_eq!(
            dispatch(
                "uu94irFOGpA",
                args(&[voice, address(&unknown)]),
                &mut fixture.state
            ),
            Some(error(NGS2_ERROR_INVALID_VOICE_CONTROL_ID))
        );

        let mut truncated = [0_u8; 12];
        control_entry(&mut truncated, 0, 8, 0, NGS2_VOICE_KICK_EVENT);
        assert_eq!(
            dispatch(
                "uu94irFOGpA",
                args(&[voice, address(&truncated)]),
                &mut fixture.state
            ),
            Some(error(NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE))
        );

        let mut unknown_port = [0_u8; 16];
        control_entry(&mut unknown_port, 0, 16, 0, NGS2_VOICE_SET_PORT_VOLUME);
        unknown_port[8..12].copy_from_slice(&5_u32.to_le_bytes());
        assert_eq!(
            dispatch(
                "uu94irFOGpA",
                args(&[voice, address(&unknown_port)]),
                &mut fixture.state
            ),
            Some(error(NGS2_ERROR_INVALID_VOICE_OUTPUT_INDEX))
        );

        let mut unknown_event = [0_u8; 12];
        control_entry(&mut unknown_event, 0, 12, 0, NGS2_VOICE_KICK_EVENT);
        unknown_event[8..12].copy_from_slice(&0x40_u32.to_le_bytes());
        assert_eq!(
            dispatch(
                "uu94irFOGpA",
                args(&[voice, address(&unknown_event)]),
                &mut fixture.state
            ),
            Some(error(NGS2_ERROR_INVALID_VOICE_EVENT_TYPE))
        );
        assert_eq!(fixture.state_flags(), 0);
    }

    #[test]
    fn a_control_chain_that_links_back_to_itself_is_refused_rather_than_walked_forever() {
        let mut fixture = VoiceFixture::new(2, 2, 2);
        let mut chain = [0_u8; 24];
        control_entry(&mut chain, 0, 12, 12, NGS2_VOICE_KICK_EVENT);
        chain[8..12].copy_from_slice(&NGS2_VOICE_EVENT_PLAY.to_le_bytes());
        control_entry(&mut chain, 12, 12, -12, NGS2_VOICE_KICK_EVENT);
        chain[20..24].copy_from_slice(&NGS2_VOICE_EVENT_PLAY.to_le_bytes());
        let voice = fixture.voice;
        assert_eq!(
            dispatch(
                "uu94irFOGpA",
                args(&[voice, address(&chain)]),
                &mut fixture.state
            ),
            Some(error(NGS2_ERROR_INVALID_VOICE_CONTROL_ADDRESS))
        );
    }

    #[test]
    fn a_command_array_sets_a_voice_up_and_kicks_it_in_order() {
        let mut fixture = VoiceFixture::new(2, 2, 2);
        let setup = [2_u32, 3];
        let mut commands = [0_u8; 32];
        command_entry(&mut commands, 0, NGS2_VOICE_COMMAND_SETUP, address(&setup));
        command_entry(
            &mut commands,
            1,
            NGS2_VOICE_COMMAND_KICK_EVENT,
            u64::from(NGS2_VOICE_EVENT_PLAY),
        );
        let voice = fixture.voice;
        assert_eq!(
            dispatch(
                "AbYvTOZ8Pts",
                args(&[voice, address(&commands), 2]),
                &mut fixture.state
            ),
            Some(SUCCESS)
        );
        assert_eq!(
            fixture.state_flags(),
            NGS2_VOICE_STATE_FLAG_IN_USE | NGS2_VOICE_STATE_FLAG_PLAYING
        );

        let mut state = [0_u8; 8];
        assert_eq!(
            dispatch(
                "-TOuuAQ-buE",
                args(&[voice, address_mut(&mut state), 8]),
                &mut fixture.state
            ),
            Some(SUCCESS)
        );
        let mut expected = [0_u8; 8];
        expected[..4].copy_from_slice(
            &(NGS2_VOICE_STATE_FLAG_IN_USE | NGS2_VOICE_STATE_FLAG_PLAYING).to_le_bytes(),
        );
        assert_eq!(
            state, expected,
            "the flags lead the state and no error follows"
        );
        assert_eq!(
            dispatch(
                "-TOuuAQ-buE",
                args(&[voice, address_mut(&mut state), 4]),
                &mut fixture.state
            ),
            Some(error(NGS2_ERROR_INVALID_STATE_SIZE))
        );
        assert_eq!(
            dispatch("-TOuuAQ-buE", args(&[voice, 0, 8]), &mut fixture.state),
            Some(error(NGS2_ERROR_INVALID_OUT_ADDRESS))
        );
    }

    #[test]
    fn a_command_array_refuses_a_parameter_that_does_not_address_this_voice() {
        let mut fixture = VoiceFixture::new(2, 2, 2);
        let voice = fixture.voice;
        let mut commands = [0_u8; 16];
        command_entry(&mut commands, 0, 0x00f0_0000, 0);
        assert_eq!(
            dispatch(
                "AbYvTOZ8Pts",
                args(&[voice, address(&commands), 1]),
                &mut fixture.state
            ),
            Some(error(NGS2_ERROR_INVALID_VOICE_PARAM_ID))
        );
        command_entry(&mut commands, 0, 0x4000, 0);
        assert_eq!(
            dispatch(
                "AbYvTOZ8Pts",
                args(&[voice, address(&commands), 1]),
                &mut fixture.state
            ),
            Some(error(NGS2_ERROR_INVALID_VOICE_PARAM_ID))
        );
        command_entry(&mut commands, 0, 0x000e, 0);
        assert_eq!(
            dispatch(
                "AbYvTOZ8Pts",
                args(&[voice, address(&commands), 1]),
                &mut fixture.state
            ),
            Some(error(NGS2_ERROR_INVALID_VOICE_PARAM_ID))
        );
        // A module component carries a parameter for a module the rack would own.
        command_entry(&mut commands, 0, 0x0001_0002, 0);
        assert_eq!(
            dispatch(
                "AbYvTOZ8Pts",
                args(&[voice, address(&commands), 1]),
                &mut fixture.state
            ),
            Some(SUCCESS)
        );
        assert_eq!(
            dispatch("AbYvTOZ8Pts", args(&[voice, 0, 1]), &mut fixture.state),
            Some(error(NGS2_ERROR_INVALID_PARAM_ADDRESS))
        );
        let too_many = [0_u8; 16];
        assert_eq!(
            dispatch(
                "AbYvTOZ8Pts",
                args(&[voice, address(&too_many), NGS2_COMMAND_PARAM_LIMIT + 1]),
                &mut fixture.state
            ),
            Some(error(NGS2_ERROR_INVALID_PARAM_ADDRESS))
        );
    }

    #[test]
    fn a_voice_setup_command_bounds_the_channel_count_it_accepts() {
        let mut fixture = VoiceFixture::new(2, 2, 2);
        let voice = fixture.voice;
        let setup = [64_u32, 3];
        let mut commands = [0_u8; 16];
        command_entry(&mut commands, 0, NGS2_VOICE_COMMAND_SETUP, address(&setup));
        assert_eq!(
            dispatch(
                "AbYvTOZ8Pts",
                args(&[voice, address(&commands), 1]),
                &mut fixture.state
            ),
            Some(error(NGS2_ERROR_INVALID_NUM_CHANNELS))
        );
        let setup = [2_u32, 9];
        command_entry(&mut commands, 0, NGS2_VOICE_COMMAND_SETUP, address(&setup));
        assert_eq!(
            dispatch(
                "AbYvTOZ8Pts",
                args(&[voice, address(&commands), 1]),
                &mut fixture.state
            ),
            Some(error(NGS2_ERROR_INVALID_VOICE_SETUP_FLAG))
        );
        command_entry(&mut commands, 0, NGS2_VOICE_COMMAND_SETUP, 0);
        assert_eq!(
            dispatch(
                "AbYvTOZ8Pts",
                args(&[voice, address(&commands), 1]),
                &mut fixture.state
            ),
            Some(error(NGS2_ERROR_INVALID_PARAM_ADDRESS))
        );
    }

    /// Builds a waveform format block.
    fn waveform_format(waveform_type: u32, channels: u32) -> [u32; 6] {
        [waveform_type, channels, 48_000, 0, 0, 0]
    }

    #[test]
    fn a_sample_range_becomes_the_block_that_addresses_it() {
        let mut state = ProbeDispatchState::default();
        let format = waveform_format(NGS2_WAVEFORM_TYPE_PCM_I16L, 2);
        let mut block = [0xff_u8; NGS2_WAVEFORM_BLOCK_SIZE];
        assert_eq!(
            dispatch(
                "3pCNbVM11UA",
                args(&[address(&format), 100, 50, address_mut(&mut block)]),
                &mut state
            ),
            Some(SUCCESS)
        );
        let mut expected = [0_u8; NGS2_WAVEFORM_BLOCK_SIZE];
        expected[..8].copy_from_slice(&400_u64.to_le_bytes());
        expected[8..16].copy_from_slice(&200_u64.to_le_bytes());
        expected[16..20].copy_from_slice(&1_u32.to_le_bytes());
        expected[24..28].copy_from_slice(&50_u32.to_le_bytes());
        assert_eq!(block, expected);
    }

    #[test]
    fn a_sample_range_inside_a_compressed_unit_reports_the_samples_to_skip() {
        let mut state = ProbeDispatchState::default();
        let format = waveform_format(NGS2_WAVEFORM_TYPE_VAG, 1);
        let mut block = [0_u8; NGS2_WAVEFORM_BLOCK_SIZE];
        assert_eq!(
            dispatch(
                "3pCNbVM11UA",
                args(&[address(&format), 30, 28, address_mut(&mut block)]),
                &mut state
            ),
            Some(SUCCESS)
        );
        assert_eq!(
            u64::from_le_bytes(block[..8].try_into().expect("the offset is eight bytes")),
            u64::from(NGS2_VAG_UNIT_BYTES)
        );
        assert_eq!(
            u64::from_le_bytes(block[8..16].try_into().expect("the size is eight bytes")),
            u64::from(NGS2_VAG_UNIT_BYTES) * 2
        );
        assert_eq!(
            u32::from_le_bytes(block[20..24].try_into().expect("the skip is four bytes")),
            2
        );
    }

    #[test]
    fn a_waveform_this_build_cannot_address_by_sample_is_refused() {
        let mut state = ProbeDispatchState::default();
        let codec = waveform_format(0x40, 2);
        let no_channels = waveform_format(NGS2_WAVEFORM_TYPE_PCM_I16L, 0);
        let mut block = [0_u8; NGS2_WAVEFORM_BLOCK_SIZE];
        let output = address_mut(&mut block);
        assert_eq!(
            dispatch(
                "3pCNbVM11UA",
                args(&[address(&codec), 0, 1, output]),
                &mut state
            ),
            Some(error(NGS2_ERROR_INVALID_WAVEFORM_TYPE))
        );
        assert_eq!(
            dispatch(
                "3pCNbVM11UA",
                args(&[address(&no_channels), 0, 1, output]),
                &mut state
            ),
            Some(error(NGS2_ERROR_INVALID_NUM_CHANNELS))
        );
        assert_eq!(
            dispatch("3pCNbVM11UA", args(&[address(&codec), 0, 1, 0]), &mut state),
            Some(error(NGS2_ERROR_INVALID_OUT_ADDRESS))
        );
        assert_eq!(
            dispatch("3pCNbVM11UA", args(&[0, 0, 1, output]), &mut state),
            Some(error(NGS2_ERROR_INVALID_WAVEFORM_ADDRESS))
        );
    }

    /// Builds a container holding `samples` stereo sixteen-bit frames.
    fn riff_container(samples: u32) -> Vec<u8> {
        let payload = samples * 4;
        let mut data = Vec::new();
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&(36 + payload).to_le_bytes());
        data.extend_from_slice(b"WAVE");
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&16_u32.to_le_bytes());
        data.extend_from_slice(&1_u16.to_le_bytes());
        data.extend_from_slice(&2_u16.to_le_bytes());
        data.extend_from_slice(&48_000_u32.to_le_bytes());
        data.extend_from_slice(&192_000_u32.to_le_bytes());
        data.extend_from_slice(&4_u16.to_le_bytes());
        data.extend_from_slice(&16_u16.to_le_bytes());
        data.extend_from_slice(b"data");
        data.extend_from_slice(&payload.to_le_bytes());
        data.resize(44 + payload as usize, 0);
        data
    }

    #[test]
    fn a_recognized_container_reports_its_format_data_and_first_block() {
        let mut state = ProbeDispatchState::default();
        let container = riff_container(2);
        let mut info = [0xff_u8; NGS2_WAVEFORM_INFO_SIZE];
        assert_eq!(
            dispatch(
                "hyVLT2VlOYk",
                args(&[
                    container.as_ptr() as u64,
                    container.len() as u64,
                    address_mut(&mut info)
                ]),
                &mut state
            ),
            Some(SUCCESS)
        );
        let field = |offset: usize| {
            u32::from_le_bytes(
                info[offset..offset + 4]
                    .try_into()
                    .expect("a field is four bytes"),
            )
        };
        assert_eq!(field(0), NGS2_WAVEFORM_TYPE_PCM_I16L);
        assert_eq!(field(4), 2);
        assert_eq!(field(8), 48_000);
        assert_eq!(info[12..24], [0; 12], "no codec configuration is reported");
        assert_eq!(field(24), 44);
        assert_eq!(field(28), 8);
        assert_eq!(info[32..40], [0; 8], "an unlooped waveform reports no loop");
        assert_eq!(field(40), 2);
        assert_eq!(field(44), 4);
        assert_eq!(field(48), 1);
        assert_eq!(field(52), 1);
        assert_eq!(field(56), 4);
        assert_eq!(field(60), 1);
        assert_eq!(field(64), 0);
        assert_eq!(field(68), 1);
        let mut block = [0_u8; NGS2_WAVEFORM_BLOCK_SIZE];
        block[8..16].copy_from_slice(&8_u64.to_le_bytes());
        block[16..20].copy_from_slice(&1_u32.to_le_bytes());
        block[24..28].copy_from_slice(&2_u32.to_le_bytes());
        assert_eq!(info[72..112], block);
        assert_eq!(info[112..], [0; 120], "only one block is described");
    }

    #[test]
    fn a_container_this_build_does_not_recognize_is_reported_as_unknown() {
        let mut state = ProbeDispatchState::default();
        let mut info = [0_u8; NGS2_WAVEFORM_INFO_SIZE];
        let output = address_mut(&mut info);
        let unknown = *b"OggS____________";
        assert_eq!(
            dispatch(
                "hyVLT2VlOYk",
                args(&[address(&unknown), unknown.len() as u64, output]),
                &mut state
            ),
            Some(error(NGS2_ERROR_UNKNOWN_WAVEFORM_FORMAT))
        );
        let container = riff_container(2);
        assert_eq!(
            dispatch(
                "hyVLT2VlOYk",
                args(&[container.as_ptr() as u64, container.len() as u64, 0]),
                &mut state
            ),
            Some(error(NGS2_ERROR_INVALID_OUT_ADDRESS))
        );
        assert_eq!(
            dispatch("hyVLT2VlOYk", args(&[0, 64, output]), &mut state),
            Some(error(NGS2_ERROR_INVALID_WAVEFORM_DATA))
        );
        assert_eq!(
            dispatch(
                "hyVLT2VlOYk",
                args(&[container.as_ptr() as u64, 4, output]),
                &mut state
            ),
            Some(error(NGS2_ERROR_INVALID_WAVEFORM_DATA))
        );
    }

    /// Context the memory-registration test publishes its regions under.
    const AJM_TEST_CONTEXT: u32 = 0x4176;

    /// Builds one padded layer III frame at 128 kbit/s, 44100 Hz, two channels.
    fn mp3_frame() -> Vec<u8> {
        let mut frame = vec![0_u8; 417];
        frame[0] = 0xff;
        frame[1] = 0xfb;
        frame[2] = 0x90;
        frame
    }

    #[test]
    fn a_frame_header_reports_the_geometry_the_decoder_needs() {
        let mut state = ProbeDispatchState::default();
        let frame = mp3_frame();
        let mut parsed = [0xff_u8; MP3_PARSE_FRAME_SIZE];
        assert_eq!(
            dispatch(
                "eDFeTyi+G3Y",
                args(&[
                    frame.as_ptr() as u64,
                    frame.len() as u64,
                    0,
                    address_mut(&mut parsed)
                ]),
                &mut state
            ),
            Some(SUCCESS)
        );
        let mut expected = [0_u8; MP3_PARSE_FRAME_SIZE];
        expected[..8].copy_from_slice(&417_u64.to_le_bytes());
        expected[8..12].copy_from_slice(&2_u32.to_le_bytes());
        expected[12..16].copy_from_slice(&1152_u32.to_le_bytes());
        expected[16..20].copy_from_slice(&128_000_u32.to_le_bytes());
        expected[20..24].copy_from_slice(&44_100_u32.to_le_bytes());
        assert_eq!(parsed, expected);
    }

    #[test]
    fn an_original_file_length_header_is_read_only_when_it_is_asked_for() {
        let mut state = ProbeDispatchState::default();
        let mut frame = mp3_frame();
        frame[36..40].copy_from_slice(b"Xing");
        frame[40..44].copy_from_slice(&1_u32.to_be_bytes());
        frame[44..48].copy_from_slice(&1000_u32.to_be_bytes());
        frame[48..52].copy_from_slice(b"LAME");
        frame[69] = 0x02;
        frame[70] = 0x10;
        let mut parsed = [0_u8; MP3_PARSE_FRAME_SIZE];
        let output = address_mut(&mut parsed);
        assert_eq!(
            dispatch(
                "eDFeTyi+G3Y",
                args(&[frame.as_ptr() as u64, frame.len() as u64, 1, output]),
                &mut state
            ),
            Some(SUCCESS)
        );
        let field = |offset: usize| {
            u32::from_le_bytes(
                parsed[offset..offset + 4]
                    .try_into()
                    .expect("a field is four bytes"),
            )
        };
        assert_eq!(
            field(24),
            33,
            "the encoder delay comes from the encoder tag"
        );
        assert_eq!(field(28), 1000);
        assert_eq!(field(32), 1000 * 1152);
        assert_eq!(field(36), MP3_OFL_TYPE_LAME);

        assert_eq!(
            dispatch(
                "eDFeTyi+G3Y",
                args(&[frame.as_ptr() as u64, frame.len() as u64, 0, output]),
                &mut state
            ),
            Some(SUCCESS)
        );
        assert_eq!(parsed[24..40], [0; 16], "an unrequested header is not read");
    }

    #[test]
    fn a_bitstream_carrying_no_frame_is_rejected_rather_than_guessed_at() {
        let mut state = ProbeDispatchState::default();
        let silence = [0_u8; 64];
        let mut parsed = [0_u8; MP3_PARSE_FRAME_SIZE];
        let output = address_mut(&mut parsed);
        for arguments in [
            args(&[address(&silence), silence.len() as u64, 0, output]),
            args(&[0, 64, 0, output]),
            args(&[address(&silence), 0, 0, output]),
            args(&[address(&silence), silence.len() as u64, 0, 0]),
        ] {
            assert_eq!(
                dispatch("eDFeTyi+G3Y", arguments, &mut state),
                Some(error(SCE_AJM_ERROR_INVALID_PARAMETER))
            );
        }
        assert_eq!(parsed, [0; MP3_PARSE_FRAME_SIZE]);
    }

    #[test]
    fn published_coprocessor_memory_must_be_page_aligned_and_owned_by_a_live_context() {
        let mut state = ProbeDispatchState::default();
        state
            .ajm_contexts
            .insert(AJM_TEST_CONTEXT, AjmContextState::default());
        let region = 0x1_0000_0000_u64;
        assert_eq!(
            dispatch(
                "bkRHEYG6lEM",
                args(&[u64::from(AJM_TEST_CONTEXT), region + 1, 1]),
                &mut state
            ),
            Some(error(SCE_AJM_ERROR_INVALID_PARAMETER))
        );
        assert_eq!(
            dispatch(
                "bkRHEYG6lEM",
                args(&[u64::from(AJM_TEST_CONTEXT), region, 0]),
                &mut state
            ),
            Some(error(SCE_AJM_ERROR_INVALID_PARAMETER))
        );
        assert_eq!(
            dispatch("bkRHEYG6lEM", args(&[0, region, 1]), &mut state),
            Some(error(SCE_AJM_ERROR_INVALID_CONTEXT))
        );
        assert_eq!(
            dispatch(
                "bkRHEYG6lEM",
                args(&[u64::from(AJM_TEST_CONTEXT), region, 2]),
                &mut state
            ),
            Some(SUCCESS)
        );
        assert_eq!(
            dispatch(
                "bkRHEYG6lEM",
                args(&[
                    u64::from(AJM_TEST_CONTEXT),
                    region + AJM_MEMORY_PAGE_SIZE,
                    1
                ]),
                &mut state
            ),
            Some(error(SCE_AJM_ERROR_INVALID_PARAMETER)),
            "a region inside one already published must be refused"
        );
        assert_eq!(
            dispatch(
                "bkRHEYG6lEM",
                args(&[u64::from(AJM_TEST_CONTEXT), region, AJM_MEMORY_PAGE_LIMIT]),
                &mut state
            ),
            Some(error(SCE_AJM_ERROR_INVALID_PARAMETER))
        );
        assert_eq!(
            dispatch(
                "pIpGiaYkHkM",
                args(&[u64::from(AJM_TEST_CONTEXT), region + AJM_MEMORY_PAGE_SIZE]),
                &mut state
            ),
            Some(error(SCE_AJM_ERROR_INVALID_PARAMETER))
        );
        assert_eq!(
            dispatch(
                "pIpGiaYkHkM",
                args(&[u64::from(AJM_TEST_CONTEXT), region]),
                &mut state
            ),
            Some(SUCCESS)
        );
        assert_eq!(
            dispatch(
                "pIpGiaYkHkM",
                args(&[u64::from(AJM_TEST_CONTEXT), region]),
                &mut state
            ),
            Some(error(SCE_AJM_ERROR_INVALID_PARAMETER))
        );
    }

    #[test]
    fn an_error_code_is_described_at_one_address_the_guest_can_read() {
        let mut state = ProbeDispatchState::default();
        let Some(address) = dispatch(
            "AxhcqVv5AYU",
            args(&[u64::from(SCE_AJM_ERROR_INVALID_PARAMETER)]),
            &mut state,
        ) else {
            panic!("the module owns this NID");
        };
        assert_ne!(address, 0);
        assert_eq!(
            crate::read_guest_cstring(address).as_deref(),
            Some("invalid parameter")
        );
        assert_eq!(
            dispatch(
                "AxhcqVv5AYU",
                args(&[u64::from(SCE_AJM_ERROR_INVALID_PARAMETER)]),
                &mut state
            ),
            Some(address),
            "one code must always describe itself at one address"
        );
        let Some(unknown) = dispatch("AxhcqVv5AYU", args(&[0x8093_7fff]), &mut state) else {
            panic!("the module owns this NID");
        };
        assert_eq!(
            crate::read_guest_cstring(unknown).as_deref(),
            Some("unrecognized error code")
        );
    }

    #[test]
    fn a_synchronized_output_bounds_its_port_array_before_it_touches_a_port() {
        let mut state = ProbeDispatchState::default();
        let mut entries = [0_u8; 32];
        entries[..4].copy_from_slice(&0x7fff_u32.to_le_bytes());
        entries[16..20].copy_from_slice(&0x7ffe_u32.to_le_bytes());
        assert_eq!(
            dispatch("w3PdaSTSwGE", args(&[0, 1]), &mut state),
            Some(audio_out_error(audio::AudioOutError::InvalidPointer))
        );
        assert_eq!(
            dispatch("w3PdaSTSwGE", args(&[address(&entries), 0]), &mut state),
            Some(error(AUDIO_OUT_ERROR_INVALID_ARG))
        );
        assert_eq!(
            dispatch(
                "w3PdaSTSwGE",
                args(&[address(&entries), AUDIO_OUT_OUTPUTS_MAX + 1]),
                &mut state
            ),
            Some(error(AUDIO_OUT_ERROR_INVALID_ARG))
        );
        assert_eq!(
            dispatch("w3PdaSTSwGE", args(&[address(&entries), 2]), &mut state),
            Some(audio_out_error(audio::AudioOutError::InvalidPort))
        );
    }

    #[test]
    fn a_pad_speaker_mix_level_outside_its_range_is_rejected_before_the_port_is_looked_up() {
        let mut state = ProbeDispatchState::default();
        for level in [
            u64::MAX,
            u64::try_from(AUDIO_OUT_MIXLEVEL_MAXIMUM + 1).unwrap_or(0),
        ] {
            assert_eq!(
                dispatch("wVwPU50pS1c", args(&[0x7fff, level]), &mut state),
                Some(error(AUDIO_OUT_ERROR_INVALID_MIXLEVEL))
            );
        }
        assert_eq!(
            dispatch(
                "wVwPU50pS1c",
                args(&[
                    0x7fff,
                    u64::try_from(AUDIO_OUT_MIXLEVEL_MAXIMUM).unwrap_or(0)
                ]),
                &mut state
            ),
            Some(audio_out_error(audio::AudioOutError::InvalidPort))
        );
    }

    #[test]
    fn a_nid_no_module_here_owns_is_left_for_the_next_dispatcher() {
        let mut state = open_display();
        assert_eq!(dispatch("AAAAAAAAAAA", args(&[0]), &mut state), None);
    }
}
