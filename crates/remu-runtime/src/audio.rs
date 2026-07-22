// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_QUEUED_AUDIO_SECONDS: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AudioOutError {
    NotOpened,
    Busy,
    InvalidPort,
    InvalidPointer,
    PortFull,
    InvalidSize,
    InvalidFormat,
    InvalidSampleFrequency,
    InvalidVolume,
    InvalidPortType,
    OutOfMemory,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SampleEncoding {
    Signed16,
    Float32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AudioFormat {
    channels: usize,
    encoding: SampleEncoding,
}

impl AudioFormat {
    fn decode(raw: i32) -> Option<Self> {
        let format = raw & 0xff;
        let channels = match format {
            0 | 3 => 1,
            1 | 4 => 2,
            2 | 5 | 6 | 7 => 8,
            _ => return None,
        };
        let encoding = if matches!(format, 3..=5 | 7) {
            SampleEncoding::Float32
        } else {
            SampleEncoding::Signed16
        };
        Some(Self { channels, encoding })
    }

    const fn bytes_per_sample(self) -> usize {
        match self.encoding {
            SampleEncoding::Signed16 => 2,
            SampleEncoding::Float32 => 4,
        }
    }

    fn buffer_bytes(self, frames: u32) -> Option<usize> {
        usize::try_from(frames)
            .ok()?
            .checked_mul(self.channels)?
            .checked_mul(self.bytes_per_sample())
    }
}

#[derive(Debug)]
struct AudioPort {
    user_id: i32,
    port_type: i32,
    buffer_frames: u32,
    frequency: u32,
    raw_format: i32,
    format: AudioFormat,
    channel_volume: [f32; 8],
    next_deadline: Option<Instant>,
}

impl AudioPort {
    fn output_deadline(&mut self) -> Duration {
        let now = Instant::now();
        let deadline = self.next_deadline.unwrap_or(now).max(now);
        let frame_duration =
            Duration::from_secs_f64(f64::from(self.buffer_frames) / f64::from(self.frequency));
        self.next_deadline = Some(deadline + frame_duration);
        deadline.saturating_duration_since(now)
    }
}

#[derive(Debug, Default)]
struct AudioMixer {
    queues: BTreeMap<i32, VecDeque<[f32; 2]>>,
}

impl AudioMixer {
    fn add_port(&mut self, handle: i32) {
        self.queues.entry(handle).or_default();
    }

    fn remove_port(&mut self, handle: i32) {
        self.queues.remove(&handle);
    }

    fn push(&mut self, handle: i32, frames: impl IntoIterator<Item = [f32; 2]>) -> bool {
        let Some(queue) = self.queues.get_mut(&handle) else {
            return false;
        };
        queue.extend(frames);
        true
    }

    fn write_interleaved(&mut self, output: &mut [f32], channels: usize) {
        if channels == 0 {
            output.fill(0.0);
            return;
        }
        let has_open_port = !self.queues.is_empty();
        let mut output_frames = 0_u64;
        let mut underrun_frames = 0_u64;
        for frame in output.chunks_mut(channels) {
            output_frames = output_frames.saturating_add(1);
            let mut mixed = [0.0_f32; 2];
            let mut missing_port_sample = false;
            for queue in self.queues.values_mut() {
                if let Some(sample) = queue.pop_front() {
                    mixed[0] += sample[0];
                    mixed[1] += sample[1];
                } else {
                    missing_port_sample = true;
                }
            }
            if has_open_port && missing_port_sample {
                underrun_frames = underrun_frames.saturating_add(1);
            }
            frame.fill(0.0);
            if frame.len() == 1 {
                frame[0] = ((mixed[0] + mixed[1]) * 0.5).clamp(-1.0, 1.0);
            } else {
                frame[0] = mixed[0].clamp(-1.0, 1.0);
                frame[1] = mixed[1].clamp(-1.0, 1.0);
            }
        }
        if has_open_port {
            remu_gpu::report_audio_frames(output_frames, underrun_frames);
        }
    }
}

struct HostAudio {
    sample_rate: u32,
    channels: usize,
    mixer: Arc<Mutex<AudioMixer>>,
    #[cfg(target_os = "macos")]
    _stream: cpal::Stream,
}

impl fmt::Debug for HostAudio {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HostAudio")
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.channels)
            .finish_non_exhaustive()
    }
}

#[cfg(target_os = "macos")]
impl HostAudio {
    fn open() -> Result<Self, String> {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "no default output device".to_owned())?;
        let mut configs = device
            .supported_output_configs()
            .map_err(|error| error.to_string())?
            .filter(|config| config.sample_format() == cpal::SampleFormat::F32)
            .collect::<Vec<_>>();
        configs.sort_unstable_by_key(|config| (config.channels() != 2, config.channels()));
        let range = configs
            .into_iter()
            .next()
            .ok_or_else(|| "default output device has no f32 stream".to_owned())?;
        let preferred_rate: cpal::SampleRate = 48_000;
        let sample_rate = if range.min_sample_rate() <= preferred_rate
            && preferred_rate <= range.max_sample_rate()
        {
            preferred_rate
        } else {
            range.max_sample_rate()
        };
        let supported = range.with_sample_rate(sample_rate);
        let config = supported.config();
        let channels = usize::from(config.channels);
        let mixer = Arc::new(Mutex::new(AudioMixer::default()));
        let callback_mixer = Arc::clone(&mixer);
        let stream = device
            .build_output_stream(
                config,
                move |output: &mut [f32], _| {
                    callback_mixer
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .write_interleaved(output, channels);
                },
                |error| eprintln!("[REMu][AUDIO] host stream error: {error}"),
                None,
            )
            .map_err(|error| error.to_string())?;
        stream.play().map_err(|error| error.to_string())?;
        Ok(Self {
            sample_rate,
            channels,
            mixer,
            _stream: stream,
        })
    }
}

#[cfg(not(target_os = "macos"))]
impl HostAudio {
    fn open() -> Result<Self, String> {
        Err("host audio backend is not implemented on this platform".to_owned())
    }
}

#[derive(Debug, Default)]
pub(crate) struct AudioOutState {
    initialized: bool,
    next_handle: i32,
    ports: BTreeMap<i32, AudioPort>,
    host: Option<HostAudio>,
}

impl AudioOutState {
    pub(crate) fn initialize(&mut self) {
        if self.initialized {
            return;
        }
        self.initialized = true;
        match HostAudio::open() {
            Ok(host) => {
                eprintln!(
                    "[REMu][AUDIO] opened host output at {} Hz with {} channels",
                    host.sample_rate, host.channels
                );
                self.host = Some(host);
            }
            Err(error) => {
                eprintln!("[REMu][AUDIO] host output unavailable; using timed sink: {error}");
            }
        }
    }

    pub(crate) fn open(&mut self, arguments: [u64; 6]) -> Result<u64, AudioOutError> {
        if !self.initialized {
            return Err(AudioOutError::NotOpened);
        }
        let user_id = low_i32(arguments[0]);
        let port_type = low_i32(arguments[1]);
        let buffer_frames = low_u32(arguments[3]);
        let frequency = low_u32(arguments[4]);
        let raw_format = low_i32(arguments[5]);
        if !(0..=4).contains(&port_type) {
            return Err(AudioOutError::InvalidPortType);
        }
        let format = AudioFormat::decode(raw_format).ok_or(AudioOutError::InvalidFormat)?;
        if !(64..=8192).contains(&buffer_frames) || !buffer_frames.is_power_of_two() {
            return Err(AudioOutError::InvalidSize);
        }
        if !(8_000..=192_000).contains(&frequency) {
            return Err(AudioOutError::InvalidSampleFrequency);
        }
        if format.buffer_bytes(buffer_frames).is_none() {
            return Err(AudioOutError::OutOfMemory);
        }
        if self.ports.len() >= 8 {
            return Err(AudioOutError::PortFull);
        }
        self.next_handle = self.next_handle.saturating_add(1).max(1);
        let handle = self.next_handle;
        self.ports.insert(
            handle,
            AudioPort {
                user_id,
                port_type,
                buffer_frames,
                frequency,
                raw_format,
                format,
                channel_volume: [1.0; 8],
                next_deadline: None,
            },
        );
        if let Some(host) = self.host.as_ref() {
            host.mixer
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .add_port(handle);
        }
        eprintln!(
            "[REMu][AUDIO] opened port={handle} user={user_id:#x} type={port_type} frames={buffer_frames} rate={frequency} format={raw_format:#x} channels={} encoding={:?}",
            format.channels, format.encoding
        );
        Ok(handle.cast_unsigned().into())
    }

    pub(crate) fn close(&mut self, handle: u64) -> Result<(), AudioOutError> {
        let handle = low_i32(handle);
        let port = self
            .ports
            .remove(&handle)
            .ok_or(AudioOutError::InvalidPort)?;
        if let Some(host) = self.host.as_ref() {
            host.mixer
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove_port(handle);
        }
        eprintln!(
            "[REMu][AUDIO] closed port={handle} user={:#x} type={} format={:#x}",
            port.user_id, port.port_type, port.raw_format
        );
        Ok(())
    }

    pub(crate) fn set_volume(&mut self, arguments: [u64; 6]) -> Result<(), AudioOutError> {
        let handle = low_i32(arguments[0]);
        let flags = low_u32(arguments[1]);
        let volumes = arguments[2];
        let port = self
            .ports
            .get_mut(&handle)
            .ok_or(AudioOutError::InvalidPort)?;
        if flags != 0 && volumes == 0 {
            return Err(AudioOutError::InvalidPointer);
        }
        for channel in 0..8_u32 {
            if flags & (1 << channel) == 0 {
                continue;
            }
            let address = volumes + u64::from(channel) * 4;
            // SAFETY: sceAudioOutSetVolume supplies one readable i32 for every selected channel.
            let raw = unsafe { std::ptr::read_unaligned(address as *const i32) };
            if !(0..=32_768).contains(&raw) {
                return Err(AudioOutError::InvalidVolume);
            }
            let raw = u16::try_from(raw).map_err(|_| AudioOutError::InvalidVolume)?;
            port.channel_volume[usize::try_from(channel).expect("channel is below eight")] =
                f32::from(raw) / 32_768.0;
        }
        Ok(())
    }

    pub(crate) fn get_port_state(&self, arguments: [u64; 6]) -> Result<(), AudioOutError> {
        let handle = low_i32(arguments[0]);
        let output = arguments[1];
        let port = self.ports.get(&handle).ok_or(AudioOutError::InvalidPort)?;
        if output == 0 {
            return Err(AudioOutError::InvalidPointer);
        }
        // SceAudioOutPortState is 16 bytes: output count, channel count, reserved/reroute fields,
        // volume, and flags. A live primary device reports one output and the port channel count.
        // SAFETY: The ABI supplies a writable 16-byte port-state object.
        unsafe {
            std::ptr::write_bytes(output as *mut u8, 0, 16);
            std::ptr::write_unaligned(output as *mut u16, 1);
            std::ptr::write_unaligned(
                (output + 2) as *mut u16,
                u16::try_from(port.format.channels).expect("Gen5 audio has at most eight channels"),
            );
            std::ptr::write_unaligned((output + 7) as *mut u8, 127);
        }
        Ok(())
    }

    pub(crate) fn output(&mut self, arguments: [u64; 6]) -> Result<Duration, AudioOutError> {
        let handle = low_i32(arguments[0]);
        let source_address = arguments[1];
        let port = self
            .ports
            .get_mut(&handle)
            .ok_or(AudioOutError::InvalidPort)?;
        let delay = port.output_deadline();
        if source_address == 0 {
            return Ok(delay);
        }
        let byte_length = port
            .format
            .buffer_bytes(port.buffer_frames)
            .ok_or(AudioOutError::OutOfMemory)?;
        // SAFETY: The guest port contract supplies exactly one complete configured audio buffer.
        let source =
            unsafe { std::slice::from_raw_parts(source_address as *const u8, byte_length) };
        let stereo = decode_stereo_frames(source, port);
        if let Some(host) = self.host.as_ref() {
            let resampled = resample_stereo(&stereo, port.frequency, host.sample_rate);
            let mut mixer = host
                .mixer
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let queued = mixer.queues.get(&handle).map_or(0, VecDeque::len);
            let maximum =
                usize::try_from(host.sample_rate).unwrap_or(48_000) * MAX_QUEUED_AUDIO_SECONDS;
            if queued.saturating_add(resampled.len()) > maximum || !mixer.push(handle, resampled) {
                return Err(AudioOutError::Busy);
            }
        }
        Ok(delay)
    }
}

fn decode_stereo_frames(source: &[u8], port: &AudioPort) -> Vec<[f32; 2]> {
    let frame_bytes = port.format.channels * port.format.bytes_per_sample();
    source
        .chunks_exact(frame_bytes)
        .map(|frame| {
            let mut samples = [0.0_f32; 8];
            for (channel, sample) in samples.iter_mut().enumerate().take(port.format.channels) {
                *sample = read_sample(frame, channel, port.format) * port.channel_volume[channel];
            }
            match port.format.channels {
                1 => [samples[0], samples[0]],
                2 => [samples[0], samples[1]],
                8 => [
                    samples[0]
                        + samples[2] * 0.707
                        + samples[3] * 0.5
                        + samples[4] * 0.707
                        + samples[6] * 0.707,
                    samples[1]
                        + samples[2] * 0.707
                        + samples[3] * 0.5
                        + samples[5] * 0.707
                        + samples[7] * 0.707,
                ],
                _ => unreachable!("validated Gen5 audio format channel count"),
            }
        })
        .collect()
}

fn read_sample(frame: &[u8], channel: usize, format: AudioFormat) -> f32 {
    let offset = channel * format.bytes_per_sample();
    match format.encoding {
        SampleEncoding::Signed16 => {
            let bytes = [frame[offset], frame[offset + 1]];
            f32::from(i16::from_le_bytes(bytes)) / 32_768.0
        }
        SampleEncoding::Float32 => {
            let bytes = frame[offset..offset + 4]
                .try_into()
                .expect("a float sample is exactly four bytes");
            f32::from_le_bytes(bytes).clamp(-1.0, 1.0)
        }
    }
}

fn resample_stereo(source: &[[f32; 2]], source_rate: u32, destination_rate: u32) -> Vec<[f32; 2]> {
    if source.is_empty() || source_rate == destination_rate {
        return source.to_vec();
    }
    let destination_frames = source
        .len()
        .saturating_mul(usize::try_from(destination_rate).unwrap_or(0))
        .div_ceil(usize::try_from(source_rate).unwrap_or(1));
    (0..destination_frames)
        .map(|index| {
            let numerator = u64::try_from(index)
                .unwrap_or(u64::MAX)
                .saturating_mul(u64::from(source_rate));
            let left_index = usize::try_from(numerator / u64::from(destination_rate))
                .unwrap_or(source.len() - 1)
                .min(source.len() - 1);
            let right_index = (left_index + 1).min(source.len() - 1);
            let remainder = numerator % u64::from(destination_rate);
            let fraction_q16 = remainder.saturating_mul(65_536) / u64::from(destination_rate);
            let fraction = f32::from(u16::try_from(fraction_q16).unwrap_or(u16::MAX)) / 65_536.0;
            [
                source[left_index][0] + (source[right_index][0] - source[left_index][0]) * fraction,
                source[left_index][1] + (source[right_index][1] - source[left_index][1]) * fraction,
            ]
        })
        .collect()
}

const fn low_u32(value: u64) -> u32 {
    let bytes = value.to_le_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

const fn low_i32(value: u64) -> i32 {
    let bytes = value.to_le_bytes();
    i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[cfg(test)]
mod tests {
    use super::{AudioFormat, AudioPort, SampleEncoding, decode_stereo_frames, resample_stereo};

    #[test]
    fn decodes_and_downmixes_float_surround_audio() {
        let samples = [1.0_f32, -1.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0];
        let bytes = samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>();
        let port = AudioPort {
            user_id: 0,
            port_type: 0,
            buffer_frames: 1,
            frequency: 48_000,
            raw_format: 5,
            format: AudioFormat {
                channels: 8,
                encoding: SampleEncoding::Float32,
            },
            channel_volume: [1.0; 8],
            next_deadline: None,
        };
        let output = decode_stereo_frames(&bytes, &port);
        assert_eq!(output.len(), 1);
        assert!((output[0][0] - 1.4785).abs() < 0.0001);
        assert!((output[0][1] - -0.5215).abs() < 0.0001);
    }

    #[test]
    fn linearly_resamples_stereo_frames() {
        let source = [[0.0, 1.0], [1.0, 0.0]];
        let output = resample_stereo(&source, 24_000, 48_000);
        assert_eq!(output, [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [1.0, 0.0]]);
    }
}
