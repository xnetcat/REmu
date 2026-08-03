// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::{
    AUDIO_OUT, audio, audio_out_error, low_i32, process_guest_range_is_readable, read_guest_u32,
    read_guest_u64, sce_kernel_error,
};
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// Ports one synchronized output may name.
pub(super) const AUDIO_OUT_OUTPUTS_MAX: u64 = 8;
/// Bytes one `SceAudioOutOutputParam` occupies, and the offset of its buffer pointer.
pub(super) const AUDIO_OUT_OUTPUT_PARAM_SIZE: u64 = 16;
pub(super) const AUDIO_OUT_OUTPUT_PARAM_BUFFER_OFFSET: u64 = 8;
/// Bytes one `SceAudioOutPortState` occupies.
pub(super) const AUDIO_OUT_PORT_STATE_SIZE: usize = 16;
/// Error reported for a port count outside the range a synchronized output accepts.
pub(super) const AUDIO_OUT_ERROR_INVALID_ARG: u32 = 0x8026_0015;
/// Error reported for a mix level outside the range the pad speaker accepts.
pub(super) const AUDIO_OUT_ERROR_INVALID_MIXLEVEL: u32 = 0x8026_0014;
/// Loudest mix level the pad speaker accepts; unity gain is half of it.
pub(super) const AUDIO_OUT_MIXLEVEL_MAXIMUM: i32 = 32_768;

/// Mix level each port applies to the pad speaker it is additionally routed to.
pub(super) static AUDIO_OUT_PAD_SPEAKER_MIX: OnceLock<Mutex<BTreeMap<i32, i32>>> = OnceLock::new();

/// Returns the process-wide audio-output state.
pub(super) fn audio_out_state() -> std::sync::MutexGuard<'static, audio::AudioOutState> {
    AUDIO_OUT
        .get_or_init(|| Mutex::new(audio::AudioOutState::default()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Reports whether a port handle names an open port.
///
/// The mixer owns its port table privately, so the port-state query is what answers this.
pub(super) fn audio_out_port_is_open(state: &audio::AudioOutState, handle: i32) -> bool {
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
pub(crate) fn audio_out_outputs(arguments: [u64; 6]) -> u64 {
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
        // Dispatched before the process-wide dispatch lock (plan C5): the ready-wait is a
        // sleep, and holding the write lock across it would stall every stateful import.
        crate::assert_wait_is_off_the_dispatch_lock("sceAudioOutOutputs");
        std::thread::sleep(delay);
    }
    0
}

/// Sets the level at which a pad-speaker port is mixed into a connected headset.
///
/// `REmu` mixes every port to one host device and emits no separate controller-speaker route, so
/// the level is recorded for the port and changes nothing a listener hears.
pub(super) fn audio_out_set_mix_level_pad_spk(arguments: [u64; 6]) -> u64 {
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
