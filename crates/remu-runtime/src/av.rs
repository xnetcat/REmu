// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Audio and video output entry points.
//!
//! This module implements the remaining `libSceNgs2`, `libSceAjm`, `libSceAudioOut` and
//! `libSceVideoOut` entry points.
//!
//! The `NGS2` entry points own the object graph -- systems, racks, voices and the state a title
//! polls for. There is no DSP mixer behind them, so `sceNgs2SystemRender` clears the caller's
//! render buffers for one grain of silence rather than inventing mixed audio.
//!
//! [`dispatch`] is consulted by the runtime's import dispatcher only after the main table
//! misses, and returns `None` for any NID it does not own so the caller can keep looking.

use crate::{ProbeDispatchState, process_guest_range_is_readable};
mod ajm;
mod audio_out;
mod mp3;
mod ngs2;
mod ngs2_voice;
mod ngs2_waveform;
mod video_out;
#[cfg(test)]
use self::ajm::{AJM_MEMORY_PAGE_LIMIT, AJM_MEMORY_PAGE_SIZE};
use self::ajm::{ajm_memory_register, ajm_memory_unregister, ajm_str_error};
#[cfg(test)]
use self::audio_out::{
    AUDIO_OUT_ERROR_INVALID_ARG, AUDIO_OUT_ERROR_INVALID_MIXLEVEL, AUDIO_OUT_MIXLEVEL_MAXIMUM,
    AUDIO_OUT_OUTPUTS_MAX,
};
use self::audio_out::{audio_out_outputs, audio_out_set_mix_level_pad_spk};
use self::mp3::ajm_dec_mp3_parse_frame;
#[cfg(test)]
use self::mp3::{MP3_OFL_TYPE_LAME, MP3_PARSE_FRAME_SIZE};
#[cfg(test)]
use self::ngs2::{
    NGS2_ERROR_INVALID_CONTEXT_ALLOCATOR, NGS2_ERROR_INVALID_NUM_GRAIN_SAMPLES,
    NGS2_ERROR_INVALID_NUM_VOICES, NGS2_ERROR_INVALID_OPTION_SIZE, NGS2_ERROR_INVALID_RACK_HANDLE,
    NGS2_ERROR_INVALID_RACK_ID, NGS2_ERROR_INVALID_SAMPLE_RATE, NGS2_ERROR_INVALID_STATE_SIZE,
    NGS2_ERROR_INVALID_SYSTEM_HANDLE, NGS2_ERROR_INVALID_VOICE_EVENT_TYPE,
    NGS2_ERROR_INVALID_VOICE_INDEX, NGS2_RACK_OPTION_SIZE, NGS2_SYSTEM_OPTION_SIZE,
    NGS2_VOICE_EVENT_PLAY, NGS2_VOICE_EVENT_STOP, NGS2_VOICE_STATE_FLAG_PLAYING,
    NGS2_VOICE_STATE_FLAG_STOPPED,
};
use self::ngs2::{
    NGS2_ERROR_INVALID_HANDLE, NGS2_ERROR_INVALID_HANDLER_ADDRESS,
    NGS2_ERROR_INVALID_MATRIX_LEVEL_ADDRESS, NGS2_ERROR_INVALID_MODULE_INDEX,
    NGS2_ERROR_INVALID_MODULE_LOCATION, NGS2_ERROR_INVALID_NUM_CHANNELS,
    NGS2_ERROR_INVALID_OUT_ADDRESS, NGS2_ERROR_INVALID_PARAM_ADDRESS,
    NGS2_ERROR_INVALID_VOICE_CONTROL_ADDRESS, NGS2_ERROR_INVALID_VOICE_CONTROL_ID,
    NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE, NGS2_ERROR_INVALID_VOICE_HANDLE,
    NGS2_ERROR_INVALID_VOICE_MATRIX_INDEX, NGS2_ERROR_INVALID_VOICE_OUTPUT_INDEX,
    NGS2_ERROR_INVALID_VOICE_PARAM_ID, NGS2_ERROR_INVALID_VOICE_PATCH_TYPE,
    NGS2_ERROR_INVALID_VOICE_SETUP_FLAG, NGS2_ERROR_INVALID_WAVEFORM_ADDRESS,
    NGS2_ERROR_INVALID_WAVEFORM_DATA, NGS2_ERROR_INVALID_WAVEFORM_FORMAT,
    NGS2_ERROR_INVALID_WAVEFORM_TYPE, NGS2_ERROR_UNKNOWN_WAVEFORM_FORMAT, NGS2_MAX_SAMPLE_RATE,
    NGS2_MAX_VOICE_CHANNELS, NGS2_MODULE_MAX_LOCATIONS, NGS2_RACK_MAX_MODULES,
    NGS2_VOICE_STATE_FLAG_IN_USE, Ngs2State, Ngs2Voice, ngs2_apply_voice_event,
    ngs2_rack_create_with_allocator, ngs2_rack_destroy, ngs2_rack_get_voice_handle, ngs2_state,
    ngs2_system_create_with_allocator, ngs2_system_destroy, ngs2_system_render,
    ngs2_system_reset_option, ngs2_voice_get_state, ngs2_voice_get_state_flags,
};
#[cfg(test)]
use self::ngs2_voice::{
    NGS2_COMMAND_PARAM_LIMIT, NGS2_VOICE_COMMAND_KICK_EVENT, NGS2_VOICE_COMMAND_SETUP,
    NGS2_VOICE_KICK_EVENT, NGS2_VOICE_SET_PORT_VOLUME,
};
use self::ngs2_voice::{ngs2_voice_control, ngs2_voice_run_commands};
#[cfg(test)]
use self::ngs2_waveform::{
    NGS2_VAG_UNIT_BYTES, NGS2_WAVEFORM_BLOCK_SIZE, NGS2_WAVEFORM_INFO_SIZE,
    NGS2_WAVEFORM_TYPE_PCM_I16L, NGS2_WAVEFORM_TYPE_VAG,
};
use self::ngs2_waveform::{ngs2_calc_waveform_block, ngs2_parse_waveform_data};
#[cfg(test)]
use self::video_out::{
    KERNEL_EVENT_SIZE, VIDEO_OUT_BUFFER_ATTRIBUTE2_SIZE, VIDEO_OUT_ERROR_INVALID_RESOLUTION,
    VIDEO_OUT_ERROR_INVALID_TILING_MODE, VIDEO_OUT_FALSE, VIDEO_OUT_TILING_MODE_LINEAR,
    VIDEO_OUT_TILING_MODE_TILE, VIDEO_OUT_TRUE,
};
use self::video_out::{
    video_out_adjust_color, video_out_close, video_out_get_event_id, video_out_get_flip_status,
    video_out_get_vblank_status, video_out_initialize_output_options,
    video_out_is_output_supported, video_out_submit_change_buffer_attribute2,
    video_out_unregister_buffers, video_out_vrr_unpeg_from_fixed_rate, video_out_wait_vblank,
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
        "SbU3dwp80lQ" => video_out_get_flip_status(arguments, state),
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
        "i0VnXM-C9fc" => ngs2_system_render(arguments),
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

#[cfg(test)]
mod tests;
