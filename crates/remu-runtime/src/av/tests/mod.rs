mod ajm;
mod audio_out;
mod mp3;
mod ngs2;
mod ngs2_waveform;
mod video_out;

use super::*;
use crate::ajm_engine::SCE_AJM_ERROR_INVALID_PARAMETER;
use crate::{
    CompletedVideoOutFlip, PendingVideoOutFlip, VIDEO_OUT_ERROR_INVALID_ADDRESS,
    VIDEO_OUT_ERROR_INVALID_HANDLE, VIDEO_OUT_ERROR_INVALID_INDEX, VIDEO_OUT_ERROR_INVALID_VALUE,
    VideoOutBufferAttributes, audio, audio_out_error, kernel_process_time, sce_kernel_error,
};

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

/// Builds a waveform format block.
fn waveform_format(waveform_type: u32, channels: u32) -> [u32; 6] {
    [waveform_type, channels, 48_000, 0, 0, 0]
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

/// Builds one padded layer III frame at 128 kbit/s, 44100 Hz, two channels.
fn mp3_frame() -> Vec<u8> {
    let mut frame = vec![0_u8; 417];
    frame[0] = 0xff;
    frame[1] = 0xfb;
    frame[2] = 0x90;
    frame
}
