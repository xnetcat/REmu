// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use crate::{
    low_u32, process_guest_range_is_readable, read_guest_u16, read_guest_u32, read_guest_u64,
    sce_kernel_error,
};

use super::{
    NGS2_ERROR_INVALID_HANDLE, NGS2_ERROR_INVALID_HANDLER_ADDRESS,
    NGS2_ERROR_INVALID_MATRIX_LEVEL_ADDRESS, NGS2_ERROR_INVALID_MODULE_INDEX,
    NGS2_ERROR_INVALID_MODULE_LOCATION, NGS2_ERROR_INVALID_NUM_CHANNELS,
    NGS2_ERROR_INVALID_PARAM_ADDRESS, NGS2_ERROR_INVALID_VOICE_CONTROL_ADDRESS,
    NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE, NGS2_ERROR_INVALID_VOICE_HANDLE,
    NGS2_ERROR_INVALID_VOICE_MATRIX_INDEX, NGS2_ERROR_INVALID_VOICE_OUTPUT_INDEX,
    NGS2_ERROR_INVALID_VOICE_PARAM_ID, NGS2_ERROR_INVALID_VOICE_PATCH_TYPE,
    NGS2_ERROR_INVALID_VOICE_SETUP_FLAG, NGS2_MAX_VOICE_CHANNELS, NGS2_MODULE_MAX_LOCATIONS,
    NGS2_RACK_MAX_MODULES, NGS2_VOICE_STATE_FLAG_IN_USE, Ngs2State, Ngs2Voice,
    ngs2_apply_voice_event, ngs2_state,
};

/// Voice control identifiers a parameter chain may carry.
pub(super) const NGS2_VOICE_SET_MATRIX_LEVELS: u32 = 0x0001;
pub(super) const NGS2_VOICE_SET_PORT_VOLUME: u32 = 0x0002;
pub(super) const NGS2_VOICE_SET_PORT_MATRIX: u32 = 0x0003;
pub(super) const NGS2_VOICE_SET_PORT_DELAY: u32 = 0x0004;
pub(super) const NGS2_VOICE_PATCH: u32 = 0x0005;
pub(super) const NGS2_VOICE_KICK_EVENT: u32 = 0x0006;
pub(super) const NGS2_VOICE_SET_CALLBACK: u32 = 0x0007;

/// Bytes a voice control parameter header occupies.
pub(super) const NGS2_VOICE_PARAM_HEADER_SIZE: u16 = 8;
/// Entries one chain may carry before it is treated as malformed.
///
/// A chain links its entries by a signed offset, so a broken one can close a cycle rather than
/// terminate; this bounds the walk without depending on the guest's own bookkeeping.
pub(super) const NGS2_VOICE_CONTROL_CHAIN_LIMIT: usize = 256;
/// Levels one matrix may carry.
pub(super) const NGS2_MAX_MATRIX_LEVELS: u32 = NGS2_MAX_VOICE_CHANNELS * NGS2_MAX_VOICE_CHANNELS;
/// Error reported for a matrix level count outside that bound.
pub(super) const NGS2_ERROR_INVALID_NUM_MATRIX_VOLUMES: u32 = 0x804a_835d;
/// Delay one port may hold, in samples.
pub(super) const NGS2_MAX_PORT_DELAY_SAMPLES: u32 = 65_535;
/// Highest patch type a patch parameter may name.
pub(super) const NGS2_VOICE_PATCH_MAX_TYPE: u32 = 3;
/// Highest input identifier a patch destination may name.
pub(super) const NGS2_VOICE_MAX_INPUT_ID: u32 = 1;

/// Rack sizing a voice's parameters are checked against.
#[derive(Clone, Copy, Debug)]
pub(super) struct Ngs2RackLimits {
    pub(super) max_ports: u32,
    pub(super) max_matrices: u32,
}

/// Returns the sizing of the rack that owns `voice_handle`.
pub(super) fn ngs2_voice_limits(state: &Ngs2State, voice_handle: u64) -> Option<Ngs2RackLimits> {
    let voice = state.voices.get(&voice_handle)?;
    let rack = state.racks.get(&voice.rack)?;
    Some(Ngs2RackLimits {
        max_ports: rack.max_ports,
        max_matrices: rack.max_matrices,
    })
}

/// Rejects a control entry that is too short for the parameters its identifier implies.
pub(super) fn ngs2_require_control_size(size: u16, required: u16) -> Result<(), u32> {
    if size < required {
        return Err(NGS2_ERROR_INVALID_VOICE_CONTROL_SIZE);
    }
    Ok(())
}

/// Executes a chain of voice control parameters against one voice.
pub(super) fn ngs2_voice_control(arguments: [u64; 6]) -> u64 {
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
pub(super) fn ngs2_apply_voice_control(
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
        unknown => {
            // Module-specific control identifiers arrive from FMOD/Unity racks. They are
            // validated for a minimum header and discarded the same way levels and patches are:
            // nothing mixes, so accepting an unknown id as a no-op is honest once the chain is
            // well-formed. Reject only when the entry is too short to be a real control header.
            ngs2_require_control_size(size, NGS2_VOICE_PARAM_HEADER_SIZE)?;
            static LOGGED: std::sync::Mutex<std::collections::BTreeSet<u32>> =
                std::sync::Mutex::new(std::collections::BTreeSet::new());
            let mut logged = LOGGED
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if logged.insert(unknown) {
                eprintln!(
                    "[REMu][NGS2] accepting unhandled voice control id={unknown:#x} size={size} (no mixer)"
                );
            }
            Ok(())
        }
    }
}

/// Bytes one `SceNgs2CommandParam` occupies, and the offset of its value union.
pub(super) const NGS2_COMMAND_PARAM_SIZE: u64 = 16;
pub(super) const NGS2_COMMAND_PARAM_VALUE_OFFSET: u64 = 8;
/// Entries one command array may carry before it is treated as malformed.
pub(super) const NGS2_COMMAND_PARAM_LIMIT: u64 = 4096;
/// Component that addresses the voice itself; everything below the reserved range is a module.
pub(super) const NGS2_COMPONENT_ID_VOICE: u32 = 0x00;
pub(super) const NGS2_COMPONENT_ID_SYSTEM: u32 = 0xf0;
/// Parameter identifier field that selects a command rather than a query or a callback.
pub(super) const NGS2_PARAM_ID_TYPE_COMMAND: u32 = 0;

/// Voice command interface identifiers.
pub(super) const NGS2_VOICE_COMMAND_SET_USER_DATA: u32 = 0x00;
pub(super) const NGS2_VOICE_COMMAND_SETUP: u32 = 0x01;
pub(super) const NGS2_VOICE_COMMAND_KICK_EVENT: u32 = 0x02;
pub(super) const NGS2_VOICE_COMMAND_SET_AUTO_SUSPEND: u32 = 0x03;
pub(super) const NGS2_VOICE_COMMAND_SET_MODULE_LOCATION: u32 = 0x04;
pub(super) const NGS2_VOICE_COMMAND_SET_MATRIX_LEVELS: u32 = 0x05;
pub(super) const NGS2_VOICE_COMMAND_SET_PORT_VOLUME: u32 = 0x06;
pub(super) const NGS2_VOICE_COMMAND_SET_PORT_MATRIX: u32 = 0x07;
pub(super) const NGS2_VOICE_COMMAND_SET_PORT_PANNER: u32 = 0x08;
pub(super) const NGS2_VOICE_COMMAND_PATCH: u32 = 0x09;
pub(super) const NGS2_VOICE_COMMAND_SET_STRING_PARAM: u32 = 0x0a;
pub(super) const NGS2_VOICE_COMMAND_BYPASS_MODULE: u32 = 0x0b;
pub(super) const NGS2_VOICE_COMMAND_SET_CALLBACK: u32 = 0x0c;
pub(super) const NGS2_VOICE_COMMAND_SET_MODULE_ORDER: u32 = 0x0d;

/// One decoded command parameter.
#[derive(Clone, Copy, Debug)]
pub(super) struct Ngs2Command {
    pub(super) component: u32,
    pub(super) kind: u32,
    pub(super) interface: u32,
    pub(super) unit: u32,
    pub(super) values: u32,
    pub(super) value: u64,
}

/// Splits a command parameter into the fields its identifier packs together.
pub(super) fn ngs2_decode_command(entry: u64) -> Ngs2Command {
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
pub(super) fn ngs2_voice_run_commands(arguments: [u64; 6]) -> u64 {
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
pub(super) fn ngs2_apply_voice_command(
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
pub(super) fn ngs2_voice_mut(
    state: &mut Ngs2State,
    voice_handle: u64,
) -> Result<&mut Ngs2Voice, u32> {
    state
        .voices
        .get_mut(&voice_handle)
        .ok_or(NGS2_ERROR_INVALID_VOICE_HANDLE)
}

/// Records the channel layout and sampling rate a voice is set up for.
pub(super) fn ngs2_apply_voice_setup(
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
