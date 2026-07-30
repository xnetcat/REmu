// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::{
    ProbeDispatchState, VIDEO_OUT_ERROR_INVALID_ADDRESS, VIDEO_OUT_ERROR_INVALID_HANDLE,
    VIDEO_OUT_ERROR_INVALID_INDEX, VIDEO_OUT_ERROR_INVALID_VALUE, VideoOutBufferAttributes,
    guest_memset, kernel_process_time, low_i32, low_u32, process_guest_range_is_readable,
    publish_gpu_event, read_guest_u32, read_guest_u64, sce_kernel_error, signed_i32_return,
    write_guest_u64,
};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::guest_object_is_reachable;

/// Handle the console assigns to the one main display port `sceVideoOutOpen` hands out.
pub(super) const VIDEO_OUT_MAIN_HANDLE: i32 = 1;

/// Buffer set the runtime keeps. Titles that use more than one swap chain are not served here.
pub(super) const VIDEO_OUT_PRIMARY_BUFFER_SET: i32 = 0;

/// Bytes one `SceVideoOutVblankStatus` occupies.
pub(super) const VIDEO_OUT_VBLANK_STATUS_SIZE: u64 = 0x28;

/// Byte offsets within that block.
pub(super) const VIDEO_OUT_VBLANK_PROCESS_TIME_OFFSET: u64 = 0x08;

pub(super) const VIDEO_OUT_VBLANK_PROCESS_TIME_COUNTER_OFFSET: u64 = 0x18;

/// Bytes one `SceVideoOutOutputOptions` occupies.
pub(super) const VIDEO_OUT_OUTPUT_OPTIONS_SIZE: u64 = 0x40;

/// Bytes one `SceVideoOutColorSettings` occupies, and the size every caller must declare.
pub(super) const VIDEO_OUT_COLOR_SETTINGS_SIZE: u32 = 0x10;

/// Bytes one `SceVideoOutBufferAttribute2` occupies.
pub(super) const VIDEO_OUT_BUFFER_ATTRIBUTE2_SIZE: usize = 0x50;

/// Byte offsets within that attribute block.
pub(super) const VIDEO_OUT_ATTRIBUTE_TILING_MODE_OFFSET: u64 = 0x04;

pub(super) const VIDEO_OUT_ATTRIBUTE_WIDTH_OFFSET: u64 = 0x0c;

pub(super) const VIDEO_OUT_ATTRIBUTE_HEIGHT_OFFSET: u64 = 0x10;

pub(super) const VIDEO_OUT_ATTRIBUTE_PITCH_OFFSET: u64 = 0x14;

pub(super) const VIDEO_OUT_ATTRIBUTE_PIXEL_FORMAT_OFFSET: u64 = 0x20;

/// Tiling modes the display accepts.
pub(super) const VIDEO_OUT_TILING_MODE_TILE: u32 = 0;

pub(super) const VIDEO_OUT_TILING_MODE_LINEAR: u32 = 1;

/// Errors the buffer-attribute change reports for a geometry the display cannot scan out.
pub(super) const VIDEO_OUT_ERROR_INVALID_RESOLUTION: u32 = 0x8029_0005;

pub(super) const VIDEO_OUT_ERROR_INVALID_TILING_MODE: u32 = 0x8029_0007;

/// Bytes one delivered kernel event occupies, and the offset of its filter-specific flags.
pub(super) const KERNEL_EVENT_SIZE: usize = 0x20;

pub(super) const KERNEL_EVENT_FFLAGS_OFFSET: u64 = 0x0c;

/// Output mode the display always presents at.
pub(super) const VIDEO_OUT_OUTPUT_MODE_DEFAULT: u64 = 1;

/// Answers `sceVideoOutIsOutputSupported` reports.
pub(super) const VIDEO_OUT_TRUE: u64 = 1;

pub(super) const VIDEO_OUT_FALSE: u64 = 0;

/// Period of the display's vertical blank, in microseconds.
///
/// The presenter runs at one fixed mode, so this rate is a property of the emulated display
/// rather than of the flip cadence a title selects.
pub(super) const VIDEO_OUT_VBLANK_INTERVAL_MICROSECONDS: u64 = 1_000_000 / 60;

/// Nanoseconds in one microsecond, the scale between the two process-time clocks.
pub(super) const NANOSECONDS_PER_MICROSECOND: u64 = 1_000;

/// Instant the vblank counter `sceVideoOutGetVblankStatus` reports counts from.
pub(super) static VIDEO_OUT_DISPLAY_CLOCK: OnceLock<Mutex<Option<Instant>>> = OnceLock::new();

/// Gamma the guest last asked the display to apply, retained for a colour pipeline to consume.
pub(super) static VIDEO_OUT_COLOR_GAMMA: OnceLock<Mutex<[f32; 3]>> = OnceLock::new();

/// Reports whether `handle` names the open main display port.
pub(super) fn video_out_port_is_open(handle: u64, state: &ProbeDispatchState) -> bool {
    low_i32(handle) == VIDEO_OUT_MAIN_HANDLE && state.video_out.is_open
}

/// Returns the instant the display's vblank counter starts from, starting it on first use.
pub(super) fn video_out_display_clock_start() -> Instant {
    let clock = VIDEO_OUT_DISPLAY_CLOCK.get_or_init(|| Mutex::new(None));
    let mut clock = clock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *clock.get_or_insert_with(Instant::now)
}

/// Discards the vblank counter so the next port to open starts one from zero.
pub(super) fn video_out_reset_display_clock() {
    let clock = VIDEO_OUT_DISPLAY_CLOCK.get_or_init(|| Mutex::new(None));
    *clock
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
}

/// Returns the vblanks elapsed since the counter started and the microseconds since the last one.
pub(super) fn video_out_vblank_position() -> (u64, u64) {
    let elapsed =
        u64::try_from(video_out_display_clock_start().elapsed().as_micros()).unwrap_or(u64::MAX);
    (
        elapsed / VIDEO_OUT_VBLANK_INTERVAL_MICROSECONDS,
        elapsed % VIDEO_OUT_VBLANK_INTERVAL_MICROSECONDS,
    )
}

/// Releases the display port and everything registered against it.
pub(super) fn video_out_close(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if !video_out_port_is_open(arguments[0], state) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    state.video_out.is_open = false;
    state.video_out.flip_events.clear();
    state.video_out.vblank_events.clear();
    state.video_out.display_buffers.clear();
    state.video_out.display_buffer_attributes.clear();
    video_out_reset_display_clock();
    0
}

/// Frees the display-buffer set registered at `setIndex`.
pub(super) fn video_out_unregister_buffers(
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
    state.video_out.display_buffers.clear();
    state.video_out.display_buffer_attributes.clear();
    0
}

/// Changes the pixel format, tiling and geometry of an already-registered buffer set.
///
/// The console defers the change to the set's next flip. `REmu` applies it at once, matching the
/// registration path it shares state with, so a title that changes attributes and flips in the
/// same frame sees the format it asked for.
pub(super) fn video_out_submit_change_buffer_attribute2(
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
pub(super) fn apply_video_out_attributes(
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
    publish_gpu_event(remu_gpu::GpuEvent::DisplayRegistered {
        width: attributes.width,
        height: attributes.height,
        pitch: attributes.pitch,
        pixel_format: attributes.pixel_format,
        tiling_mode: attributes.tiling_mode,
        buffers: state.video_out.display_buffers.clone(),
    });
}

/// Reports the display's vblank counter and the process time of the most recent vblank.
pub(super) fn video_out_get_vblank_status(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
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
pub(super) fn video_out_wait_vblank(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
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
pub(super) fn video_out_get_event_id(event: u64) -> u64 {
    if event == 0 || !process_guest_range_is_readable(event, KERNEL_EVENT_SIZE) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_ADDRESS);
    }
    signed_i32_return(read_guest_u32(event + KERNEL_EVENT_FFLAGS_OFFSET).cast_signed())
}

/// Clears an output-options block so a caller can select the fields it cares about.
pub(super) fn video_out_initialize_output_options(options: u64) -> u64 {
    if !guest_object_is_reachable(options, VIDEO_OUT_OUTPUT_OPTIONS_SIZE) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_ADDRESS);
    }
    guest_memset(options, 0, VIDEO_OUT_OUTPUT_OPTIONS_SIZE);
    0
}

/// Reports whether the display could later be configured to `mode`.
pub(super) fn video_out_is_output_supported(
    arguments: [u64; 6],
    state: &ProbeDispatchState,
) -> u64 {
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
pub(super) fn video_out_vrr_unpeg_from_fixed_rate(
    arguments: [u64; 6],
    state: &ProbeDispatchState,
) -> u64 {
    if !video_out_port_is_open(arguments[0], state) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    // The presenter has no variable-refresh path to leave; the display is already free-running
    // with respect to the rate a title pegged it to.
    0
}

/// Applies a gamma adjustment to the display's output.
pub(super) fn video_out_adjust_color(arguments: [u64; 6], state: &ProbeDispatchState) -> u64 {
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
