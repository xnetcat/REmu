// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use std::cell::UnsafeCell;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::process_guest_range_has_read_access;

const MAX_QUEUED_AUDIO_SECONDS: usize = 2;
// Thirty-two 256-frame grains cover roughly 170 ms at 48 kHz, enough to ride
// through the 80-100 ms render stalls seen while Gen5 pipelines compile.
const AUDIO_PREBUFFER_FRAMES: usize = 8_192;

/// Bytes one `SceAudioOutPortState` occupies (audioout.h:453-461).
pub(crate) const AUDIO_OUT_PORT_STATE_SIZE: usize = 0x20;

/// Process-wide frames the host device has consumed since the stream opened.
///
/// AudioOut2 drains its virtual queue from this counter (plan I3) rather than wall time when a
/// live host stream is advancing it.
static DEVICE_CONSUMED_FRAMES: AtomicU64 = AtomicU64::new(0);
static DEVICE_CLOCK_LIVE: AtomicBool = AtomicBool::new(false);

/// Returns the host device's consumed-frame clock when a live output stream is advancing it.
pub(crate) fn device_clock_frames() -> Option<u64> {
    DEVICE_CLOCK_LIVE
        .load(Ordering::Acquire)
        .then(|| DEVICE_CONSUMED_FRAMES.load(Ordering::Acquire))
}

#[cfg(test)]
pub(crate) fn reset_device_clock_for_test() {
    DEVICE_CLOCK_LIVE.store(false, Ordering::Release);
    DEVICE_CONSUMED_FRAMES.store(0, Ordering::Release);
}

#[cfg(test)]
pub(crate) fn advance_device_clock_for_test(frames: u64) {
    DEVICE_CLOCK_LIVE.store(true, Ordering::Release);
    DEVICE_CONSUMED_FRAMES.fetch_add(frames, Ordering::AcqRel);
}

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
    resampler: Option<StreamingStereoResampler>,
    /// Host-side ring for this port when a device stream is open.
    queue: Option<Arc<AudioQueue>>,
}

impl AudioPort {
    fn output_deadline(&mut self) -> Duration {
        self.output_deadline_at(Instant::now())
    }

    fn output_deadline_at(&mut self, now: Instant) -> Duration {
        let frame_duration =
            Duration::from_secs_f64(f64::from(self.buffer_frames) / f64::from(self.frequency));
        let deadline = self.next_deadline.map_or(now, |scheduled| {
            // Preserve up to one grain of scheduling lateness as catch-up credit. Clamping every
            // late deadline to `now` permanently lowers the effective sample rate by the thread's
            // wake-up latency and turns ordinary AudioOut into the same crackle-producing clock
            // that AudioOut2 used to have. A longer gap is a real stream stop, so re-anchor then
            // rather than submitting an unbounded burst.
            if now.saturating_duration_since(scheduled) > frame_duration {
                now
            } else {
                scheduled
            }
        });
        self.next_deadline = Some(deadline + frame_duration);
        deadline.saturating_duration_since(now)
    }

    /// Device-referenced ready wait: sleep only while the host backlog exceeds one prebuffer.
    fn output_deadline_from_consumed(&self, sample_rate: u32) -> Duration {
        let Some(queue) = self.queue.as_ref() else {
            return Duration::ZERO;
        };
        let submitted = queue.submitted.load(Ordering::Acquire);
        let consumed = queue.consumed.load(Ordering::Acquire);
        let backlog = submitted.saturating_sub(consumed);
        let target = u64::try_from(AUDIO_PREBUFFER_FRAMES).unwrap_or(u64::MAX);
        if backlog <= target || sample_rate == 0 {
            return Duration::ZERO;
        }
        let wait_frames = backlog - target;
        Duration::from_secs_f64(wait_frames as f64 / f64::from(sample_rate))
    }
}

/// Playback phase for one mixer voice (plan I3 / F5).
const QUEUE_PRIMING: u8 = 0;
const QUEUE_RUNNING: u8 = 1;
const QUEUE_REBUFFERING: u8 = 2;

/// Single-producer ring; the render callback is the consumer, and the DiscardStale path may also
/// reclaim from the head under CAS so AudioOut2 can drop overdue frames without taking the mixer
/// mutex in the callback.
struct SpscStereoRing {
    slots: Box<[UnsafeCell<[f32; 2]>]>,
    head: AtomicUsize,
    tail: AtomicUsize,
}

// SAFETY: only the AudioOut guest path produces into a given ring, and consumers coordinate head
// advances with compare-exchange. Slots between head and tail are exclusively owned by one side.
unsafe impl Sync for SpscStereoRing {}
unsafe impl Send for SpscStereoRing {}

impl SpscStereoRing {
    fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let mut slots = Vec::with_capacity(capacity);
        slots.resize_with(capacity, || UnsafeCell::new([0.0, 0.0]));
        Self {
            slots: slots.into_boxed_slice(),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    fn capacity(&self) -> usize {
        self.slots.len()
    }

    fn len(&self) -> usize {
        self.tail
            .load(Ordering::Acquire)
            .wrapping_sub(self.head.load(Ordering::Acquire))
    }

    fn try_push(&self, sample: [f32; 2]) -> bool {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        if tail.wrapping_sub(head) >= self.slots.len() {
            return false;
        }
        let index = tail % self.slots.len();
        // SAFETY: the slot is past the consumer head; only this producer writes the tail side.
        unsafe {
            *self.slots[index].get() = sample;
        }
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        true
    }

    fn try_pop(&self) -> Option<[f32; 2]> {
        loop {
            let head = self.head.load(Ordering::Relaxed);
            let tail = self.tail.load(Ordering::Acquire);
            if head == tail {
                return None;
            }
            let index = head % self.slots.len();
            // SAFETY: this consumer exclusively claims `head` via the CAS below before any reuse.
            let sample = unsafe { *self.slots[index].get() };
            if self
                .head
                .compare_exchange(head, head.wrapping_add(1), Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                return Some(sample);
            }
        }
    }

    fn push_realtime_bounded(
        &self,
        frames: impl IntoIterator<Item = [f32; 2]>,
        maximum: usize,
    ) -> usize {
        let maximum = maximum.min(self.capacity()).max(1);
        let mut discarded = 0_usize;
        for frame in frames {
            while self.len() >= maximum {
                if self.try_pop().is_none() {
                    break;
                }
                discarded = discarded.saturating_add(1);
            }
            if !self.try_push(frame) {
                discarded = discarded.saturating_add(1);
            }
        }
        discarded
    }
}

impl fmt::Debug for SpscStereoRing {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SpscStereoRing")
            .field("capacity", &self.capacity())
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct AudioQueue {
    ring: SpscStereoRing,
    consumed: AtomicU64,
    submitted: AtomicU64,
    state: AtomicU8,
}

impl AudioQueue {
    fn with_capacity(maximum_queued_frames: usize) -> Self {
        Self {
            ring: SpscStereoRing::with_capacity(
                maximum_queued_frames.max(AUDIO_PREBUFFER_FRAMES),
            ),
            consumed: AtomicU64::new(0),
            submitted: AtomicU64::new(0),
            state: AtomicU8::new(QUEUE_PRIMING),
        }
    }

    fn queued_frames(&self) -> usize {
        self.ring.len()
    }

    #[cfg(test)]
    fn try_pop_for_test(&self) -> Option<[f32; 2]> {
        self.ring.try_pop()
    }
}

#[derive(Debug, Default)]
struct AudioMixer {
    queues: BTreeMap<i32, Arc<AudioQueue>>,
}

impl AudioMixer {
    fn add_port(&mut self, handle: i32, queue: Arc<AudioQueue>) {
        self.queues.insert(handle, queue);
    }

    fn remove_port(&mut self, handle: i32) {
        self.queues.remove(&handle);
    }

    fn push(&self, handle: i32, frames: impl IntoIterator<Item = [f32; 2]>) -> bool {
        let Some(queue) = self.queues.get(&handle) else {
            return false;
        };
        let mut count = 0_u64;
        for frame in frames {
            if !queue.ring.try_push(frame) {
                return false;
            }
            count = count.saturating_add(1);
        }
        queue.submitted.fetch_add(count, Ordering::AcqRel);
        true
    }

    /// Appends a real-time stream while bounding host latency.
    ///
    /// AudioOut2 already models the console hardware queue before reaching the host mixer. If
    /// CoreAudio falls behind, propagating this implementation queue's capacity as a guest BUSY
    /// error creates a second, unrelated backpressure domain and can deadlock the title. Samples
    /// beyond the host latency budget have already missed their presentation time, so retain the
    /// newest frames and report how many stale frames were discarded.
    fn push_realtime_bounded(
        &self,
        handle: i32,
        frames: Vec<[f32; 2]>,
        maximum: usize,
    ) -> Option<usize> {
        let queue = self.queues.get(&handle)?;
        let submitted = u64::try_from(frames.len()).unwrap_or(u64::MAX);
        let discarded = queue.ring.push_realtime_bounded(frames, maximum);
        queue.submitted.fetch_add(submitted, Ordering::AcqRel);
        Some(discarded)
    }

    fn snapshot_queues(&self) -> Vec<Arc<AudioQueue>> {
        self.queues.values().cloned().collect()
    }

    fn write_interleaved(queues: &[Arc<AudioQueue>], output: &mut [f32], channels: usize) {
        if channels == 0 {
            output.fill(0.0);
            return;
        }
        let requested_frames = output.len() / channels;
        let output_frames = u64::try_from(requested_frames).unwrap_or(u64::MAX);
        DEVICE_CONSUMED_FRAMES.fetch_add(output_frames, Ordering::Relaxed);

        let mut underrun_frames = 0_u64;
        let mut has_audio_port = false;
        for queue in queues {
            let available = queue.ring.len();
            if available != 0
                || queue.submitted.load(Ordering::Acquire) != 0
                || queue.state.load(Ordering::Acquire) != QUEUE_PRIMING
            {
                has_audio_port = true;
            }
            let state = queue.state.load(Ordering::Acquire);
            if state == QUEUE_RUNNING && available < requested_frames {
                // Do not consume a partial callback. Rebuffering produces a short clean silence
                // instead of a sharp audio-to-zero edge in the middle of CoreAudio's buffer.
                queue.state.store(QUEUE_REBUFFERING, Ordering::Release);
            }
            let state = queue.state.load(Ordering::Acquire);
            if state != QUEUE_RUNNING && available >= AUDIO_PREBUFFER_FRAMES {
                queue.state.store(QUEUE_RUNNING, Ordering::Release);
            }
        }

        for frame in output.chunks_mut(channels) {
            let mut mixed = [0.0_f32; 2];
            let mut missing_port_sample = false;
            let mut rebuffering_silence = false;
            for queue in queues {
                let state = queue.state.load(Ordering::Acquire);
                if state == QUEUE_REBUFFERING {
                    rebuffering_silence = true;
                    continue;
                }
                if state != QUEUE_RUNNING {
                    continue;
                }
                if let Some(sample) = queue.ring.try_pop() {
                    queue.consumed.fetch_add(1, Ordering::Relaxed);
                    mixed[0] += sample[0];
                    mixed[1] += sample[1];
                } else {
                    missing_port_sample = true;
                }
            }
            // Rebuffering silence is real underrun time (plan I3 / F5). Startup priming is not.
            if rebuffering_silence || missing_port_sample {
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
        if has_audio_port {
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
        let preferred_rate: cpal::SampleRate = 48_000;
        configs.sort_unstable_by_key(|config| {
            (
                config.channels() != 2,
                !(config.min_sample_rate() <= preferred_rate
                    && preferred_rate <= config.max_sample_rate()),
                config.channels(),
            )
        });
        let range = configs
            .into_iter()
            .next()
            .ok_or_else(|| "default output device has no f32 stream".to_owned())?;
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
        DEVICE_CLOCK_LIVE.store(true, Ordering::Release);
        let stream = device
            .build_output_stream(
                config,
                move |output: &mut [f32], _| {
                    // Snapshot the port rings under a brief lock, then mix without holding it so a
                    // guest decode/resample critical section cannot stall the real-time callback
                    // for the whole buffer (plan I3 / R4).
                    let queues = callback_mixer
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .snapshot_queues();
                    AudioMixer::write_interleaved(&queues, output, channels);
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
    successful_opens: u64,
    successful_closes: u64,
    ports: BTreeMap<i32, AudioPort>,
    host: Option<HostAudio>,
}

impl AudioOutState {
    #[cfg(test)]
    pub(crate) fn use_timed_sink_for_test(&mut self) {
        self.initialized = true;
        self.host = None;
    }

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

    /// Ports one process may hold through `sceAudioOutOpen`.
    ///
    /// This is that API's own budget: its ports are hardware outputs and the console bounds them.
    pub(crate) const HARDWARE_PORT_BUDGET: usize = 8;

    /// Backend voices AudioOut2 may hold.
    ///
    /// AudioOut2 is a software mixer whose ports are voices rather than hardware outputs, so the
    /// hardware budget above does not apply to it — an observed title holds more than forty at
    /// once, and charging those against eight made its audio engine fail to build and tear its
    /// whole stack down. This bound exists only so a runaway caller cannot grow the table without
    /// limit; it is not an ABI limit and nothing reports it to the guest.
    pub(crate) const MIXER_VOICE_BUDGET: usize = 256;

    pub(crate) fn open(&mut self, arguments: [u64; 6]) -> Result<u64, AudioOutError> {
        self.open_within(arguments, Self::HARDWARE_PORT_BUDGET)
    }

    pub(crate) fn open_within(
        &mut self,
        arguments: [u64; 6],
        budget: usize,
    ) -> Result<u64, AudioOutError> {
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
        if self.ports.len() >= budget {
            return Err(AudioOutError::PortFull);
        }
        self.next_handle = self.next_handle.saturating_add(1).max(1);
        let handle = self.next_handle;
        let queue = self.host.as_ref().map(|host| {
            let maximum_queued_frames =
                usize::try_from(host.sample_rate).unwrap_or(48_000) * MAX_QUEUED_AUDIO_SECONDS;
            let queue = Arc::new(AudioQueue::with_capacity(maximum_queued_frames));
            host.mixer
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .add_port(handle, Arc::clone(&queue));
            queue
        });
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
                resampler: self.host.as_ref().and_then(|host| {
                    (frequency != host.sample_rate)
                        .then(|| StreamingStereoResampler::new(frequency, host.sample_rate))
                }),
                queue,
            },
        );
        self.successful_opens = self.successful_opens.saturating_add(1);
        if self.successful_opens <= 8 || self.successful_opens.is_power_of_two() {
            eprintln!(
                "[REMu][AUDIO] opened sequence={} port={handle} user={user_id:#x} type={port_type} frames={buffer_frames} rate={frequency} format={raw_format:#x} channels={} encoding={:?}",
                self.successful_opens, format.channels, format.encoding
            );
        }
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
        self.successful_closes = self.successful_closes.saturating_add(1);
        if self.successful_closes <= 8 || self.successful_closes.is_power_of_two() {
            eprintln!(
                "[REMu][AUDIO] closed sequence={} port={handle} user={:#x} type={} format={:#x}",
                self.successful_closes, port.user_id, port.port_type, port.raw_format
            );
        }
        Ok(())
    }

    /// Reports whether `handle` names an open ordinary AudioOut port.
    pub(crate) fn contains_port(&self, handle: i32) -> bool {
        self.ports.contains_key(&handle)
    }

    /// Returns the configured PCM byte length for an open port.
    pub(crate) fn port_buffer_bytes(&self, handle: i32) -> Option<usize> {
        self.ports
            .get(&handle)
            .and_then(|port| port.format.buffer_bytes(port.buffer_frames))
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
        if flags != 0 {
            let mut highest = 0_u32;
            for channel in 0..8_u32 {
                if flags & (1 << channel) != 0 {
                    highest = channel;
                }
            }
            let span = usize::try_from(highest.saturating_add(1).saturating_mul(4)).unwrap_or(usize::MAX);
            if !process_guest_range_has_read_access(volumes, span) {
                eprintln!(
                    "[REMu][AUDIO] sceAudioOutSetVolume rejected unreadable volume array at {volumes:#x} span={span}"
                );
                return Err(AudioOutError::InvalidPointer);
            }
        }
        for channel in 0..8_u32 {
            if flags & (1 << channel) == 0 {
                continue;
            }
            let address = volumes + u64::from(channel) * 4;
            // SAFETY: the volume array range was proven readable above for every selected channel.
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

    pub(crate) fn set_channel_gains(
        &mut self,
        handle: u64,
        gains: [f32; 8],
    ) -> Result<(), AudioOutError> {
        if gains.iter().any(|gain| !gain.is_finite()) {
            return Err(AudioOutError::InvalidVolume);
        }
        let handle = low_i32(handle);
        let port = self
            .ports
            .get_mut(&handle)
            .ok_or(AudioOutError::InvalidPort)?;
        port.channel_volume = gains;
        Ok(())
    }

    pub(crate) fn get_port_state(&self, arguments: [u64; 6]) -> Result<(), AudioOutError> {
        let handle = low_i32(arguments[0]);
        let output = arguments[1];
        if !self.ports.contains_key(&handle) {
            return Err(AudioOutError::InvalidPort);
        }
        if output == 0 || !process_guest_range_has_read_access(output, AUDIO_OUT_PORT_STATE_SIZE) {
            eprintln!(
                "[REMu][AUDIO] sceAudioOutGetPortState rejected invalid state pointer {output:#x}"
            );
            return Err(AudioOutError::InvalidPointer);
        }
        // SceAudioOutPortState (audioout.h:453-461): u16 output route, u8 channel count, reserved
        // byte, i16 volume, u16 reroute counter, u64 flags, two reserved u64s — 0x20 bytes total.
        // The emulated host is a stereo primary route; volume is invalid for primary outputs and
        // the reroute counter remains zero until the host device actually changes.
        // SAFETY: the ABI supplies one writable 0x20-byte SceAudioOutPortState object.
        unsafe {
            std::ptr::write_bytes(output as *mut u8, 0, AUDIO_OUT_PORT_STATE_SIZE);
            std::ptr::write_unaligned(output as *mut u16, 1);
            std::ptr::write_unaligned((output + 2) as *mut u8, 2);
            std::ptr::write_unaligned((output + 4) as *mut i16, -1);
        }
        Ok(())
    }

    /// Publishes the device time for the last frame consumed on `handle` (microseconds).
    ///
    /// Derived from the port's consumed-frame counter (plan I3), so A/V sync can couple to the
    /// same clock the mixer uses for backlog pacing.
    pub(crate) fn get_last_output_time(&self, arguments: [u64; 6]) -> Result<(), AudioOutError> {
        let handle = low_i32(arguments[0]);
        let output = arguments[1];
        let port = self
            .ports
            .get(&handle)
            .ok_or(AudioOutError::InvalidPort)?;
        if output == 0 || !process_guest_range_has_read_access(output, std::mem::size_of::<u64>()) {
            eprintln!(
                "[REMu][AUDIO] sceAudioOutGetLastOutputTime rejected invalid pointer {output:#x}"
            );
            return Err(AudioOutError::InvalidPointer);
        }
        let sample_rate = self
            .host
            .as_ref()
            .map(|host| host.sample_rate)
            .unwrap_or(port.frequency)
            .max(1);
        let consumed = port
            .queue
            .as_ref()
            .map(|queue| queue.consumed.load(Ordering::Acquire))
            .unwrap_or(0);
        let micros = consumed.saturating_mul(1_000_000) / u64::from(sample_rate);
        // SAFETY: the output pointer was validated as an 8-byte writable guest object above.
        unsafe {
            std::ptr::write_unaligned(output as *mut u64, micros);
        }
        Ok(())
    }

    pub(crate) fn output(&mut self, arguments: [u64; 6]) -> Result<Duration, AudioOutError> {
        self.output_with_host_overflow_policy(arguments, HostOverflowPolicy::Reject)
    }

    /// Submits an AudioOut2 grain to the host sink.
    ///
    /// The AudioOut2 context queue has already accepted and paced this grain. Host callback
    /// lateness must therefore not surface as a second console-facing BUSY condition.
    pub(crate) fn output_realtime(
        &mut self,
        arguments: [u64; 6],
    ) -> Result<Duration, AudioOutError> {
        self.output_with_host_overflow_policy(arguments, HostOverflowPolicy::DiscardStale)
    }

    fn output_with_host_overflow_policy(
        &mut self,
        arguments: [u64; 6],
        overflow_policy: HostOverflowPolicy,
    ) -> Result<Duration, AudioOutError> {
        let handle = low_i32(arguments[0]);
        let source_address = arguments[1];
        if !self.ports.contains_key(&handle) {
            return Err(AudioOutError::InvalidPort);
        }
        if source_address == 0 {
            let sample_rate = self.host.as_ref().map(|host| host.sample_rate);
            let port = self
                .ports
                .get_mut(&handle)
                .expect("port was present above");
            return Ok(match sample_rate {
                Some(rate) if port.queue.is_some() => port.output_deadline_from_consumed(rate),
                _ => port.output_deadline(),
            });
        }
        let byte_length = {
            let port = self
                .ports
                .get(&handle)
                .expect("port was present above");
            port.format
                .buffer_bytes(port.buffer_frames)
                .ok_or(AudioOutError::OutOfMemory)?
        };
        if !process_guest_range_has_read_access(source_address, byte_length) {
            eprintln!(
                "[REMu][AUDIO] sceAudioOutOutput rejected unreadable PCM at {source_address:#x} bytes={byte_length}"
            );
            return Err(AudioOutError::InvalidPointer);
        }
        // SAFETY: the guest port contract supplies exactly one complete configured audio buffer,
        // and the range was proven readable immediately above.
        let source =
            unsafe { std::slice::from_raw_parts(source_address as *const u8, byte_length) };
        let (resampled, uses_device_clock) = {
            let port = self
                .ports
                .get_mut(&handle)
                .expect("port was present above");
            let stereo = decode_stereo_frames(source, port);
            let resampled = if let Some(resampler) = port.resampler.as_mut() {
                resampler.process(&stereo)
            } else {
                stereo
            };
            (resampled, port.queue.is_some())
        };
        let host_rate = self.host.as_ref().map(|host| host.sample_rate);
        if let Some(host) = self.host.as_ref() {
            let mixer = host
                .mixer
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let queued = mixer
                .queues
                .get(&handle)
                .map_or(0, |queue| queue.queued_frames());
            let maximum =
                usize::try_from(host.sample_rate).unwrap_or(48_000) * MAX_QUEUED_AUDIO_SECONDS;
            match overflow_policy {
                HostOverflowPolicy::Reject => {
                    if queued.saturating_add(resampled.len()) > maximum
                        || !mixer.push(handle, resampled)
                    {
                        return Err(AudioOutError::Busy);
                    }
                }
                HostOverflowPolicy::DiscardStale => {
                    let Some(discarded) = mixer.push_realtime_bounded(handle, resampled, maximum)
                    else {
                        return Err(AudioOutError::InvalidPort);
                    };
                    if discarded != 0 {
                        static DISCARD_EVENTS: AtomicU64 = AtomicU64::new(0);
                        static DISCARDED_FRAMES: AtomicU64 = AtomicU64::new(0);
                        let event = DISCARD_EVENTS.fetch_add(1, Ordering::Relaxed) + 1;
                        let discarded = u64::try_from(discarded).unwrap_or(u64::MAX);
                        let total = DISCARDED_FRAMES
                            .fetch_add(discarded, Ordering::Relaxed)
                            .saturating_add(discarded);
                        if event <= 8 || event.is_power_of_two() {
                            eprintln!(
                                "[REMu][AUDIO] host callback fell behind; discarded {discarded} stale frames from port {handle} (event={event}, total={total})"
                            );
                        }
                    }
                }
            }
        }
        let port = self
            .ports
            .get_mut(&handle)
            .expect("port was present above");
        Ok(match host_rate {
            Some(rate) if uses_device_clock => port.output_deadline_from_consumed(rate),
            _ => port.output_deadline(),
        })
    }
}

#[derive(Clone, Copy)]
enum HostOverflowPolicy {
    Reject,
    DiscardStale,
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

#[derive(Debug)]
struct StreamingStereoResampler {
    source_rate: u32,
    destination_rate: u32,
    source_frames_seen: u64,
    next_output_frame: u64,
    previous: Option<[f32; 2]>,
}

impl StreamingStereoResampler {
    const fn new(source_rate: u32, destination_rate: u32) -> Self {
        Self {
            source_rate,
            destination_rate,
            source_frames_seen: 0,
            next_output_frame: 0,
            previous: None,
        }
    }

    fn process(&mut self, source: &[[f32; 2]]) -> Vec<[f32; 2]> {
        if source.is_empty() {
            return Vec::new();
        }
        debug_assert_ne!(self.source_rate, self.destination_rate);
        let block_start = self.source_frames_seen;
        let block_end = block_start.saturating_add(u64::try_from(source.len()).unwrap_or(u64::MAX));
        let mut output = Vec::new();
        loop {
            let position_numerator = self
                .next_output_frame
                .saturating_mul(u64::from(self.source_rate));
            let left_index = position_numerator / u64::from(self.destination_rate);
            let right_index = left_index.saturating_add(1);
            if right_index >= block_end {
                break;
            }
            let left = if left_index < block_start {
                self.previous
                    .expect("only the immediately preceding streaming sample is retained")
            } else {
                source[usize::try_from(left_index - block_start)
                    .expect("current resampler block index fits usize")]
            };
            let right = source[usize::try_from(right_index - block_start)
                .expect("current resampler block index fits usize")];
            let remainder = position_numerator % u64::from(self.destination_rate);
            let fraction = remainder as f64 / f64::from(self.destination_rate);
            output.push([
                (f64::from(left[0]) + f64::from(right[0] - left[0]) * fraction) as f32,
                (f64::from(left[1]) + f64::from(right[1] - left[1]) * fraction) as f32,
            ]);
            self.next_output_frame = self.next_output_frame.saturating_add(1);
        }
        self.source_frames_seen = block_end;
        self.previous = source.last().copied();
        output
    }
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
    use super::{
        AUDIO_OUT_PORT_STATE_SIZE, AUDIO_PREBUFFER_FRAMES, AudioFormat, AudioMixer, AudioOutError,
        AudioOutState, AudioPort, AudioQueue, MAX_QUEUED_AUDIO_SECONDS, QUEUE_REBUFFERING,
        QUEUE_RUNNING, SampleEncoding, StreamingStereoResampler, advance_device_clock_for_test,
        decode_stereo_frames, reset_device_clock_for_test,
    };
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    fn stereo_audio_port() -> AudioPort {
        AudioPort {
            user_id: 0,
            port_type: 0,
            buffer_frames: 256,
            frequency: 48_000,
            raw_format: 1,
            format: AudioFormat {
                channels: 2,
                encoding: SampleEncoding::Signed16,
            },
            channel_volume: [1.0; 8],
            next_deadline: None,
            resampler: None,
            queue: None,
        }
    }

    fn open_timed_stereo_port() -> (AudioOutState, i32) {
        let mut state = AudioOutState::default();
        state.use_timed_sink_for_test();
        let handle = state
            .open([0, 0, 0, 256, 48_000, 1])
            .expect("timed sink accepts a stereo port") as i32;
        (state, handle)
    }

    #[test]
    fn ordinary_audio_deadline_preserves_short_lateness_as_catch_up_credit() {
        let mut port = stereo_audio_port();
        let grain = Duration::from_secs_f64(256.0 / 48_000.0);
        let start = Instant::now();

        assert_eq!(port.output_deadline_at(start), Duration::ZERO);
        let late = start + grain + Duration::from_millis(1);
        assert_eq!(
            port.output_deadline_at(late),
            Duration::ZERO,
            "a late call must return immediately to catch the stream back up"
        );
        assert_eq!(
            port.next_deadline,
            Some(start + grain + grain),
            "short producer lateness must not shift the device timeline"
        );
        assert_eq!(
            port.output_deadline_at(late),
            grain - Duration::from_millis(1),
            "the following call must reclaim the sub-grain scheduling lateness"
        );
    }

    #[test]
    fn ordinary_audio_deadline_reanchors_after_a_real_stream_stop() {
        let mut port = stereo_audio_port();
        let grain = Duration::from_secs_f64(256.0 / 48_000.0);
        let start = Instant::now();

        assert_eq!(port.output_deadline_at(start), Duration::ZERO);
        let resumed = start + grain + grain + Duration::from_millis(1);
        assert_eq!(port.output_deadline_at(resumed), Duration::ZERO);
        assert_eq!(
            port.next_deadline,
            Some(resumed + grain),
            "a stopped stream must resume from the current device time"
        );
    }

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
            resampler: None,
            queue: None,
        };
        let output = decode_stereo_frames(&bytes, &port);
        assert_eq!(output.len(), 1);
        assert!((output[0][0] - 1.4785).abs() < 0.0001);
        assert!((output[0][1] - -0.5215).abs() < 0.0001);
    }

    #[test]
    fn continuously_resamples_across_guest_buffer_boundaries() {
        let source = (0..512)
            .map(|index| [index as f32, -(index as f32)])
            .collect::<Vec<_>>();
        let mut whole = StreamingStereoResampler::new(48_000, 44_100);
        let expected = whole.process(&source);

        let mut streamed = StreamingStereoResampler::new(48_000, 44_100);
        let mut actual = streamed.process(&source[..256]);
        actual.extend(streamed.process(&source[256..]));

        assert_eq!(actual, expected);
        assert_eq!(actual.len(), 470);
        assert!(actual.windows(2).all(|pair| pair[1][0] > pair[0][0]));
    }

    #[test]
    fn port_state_writes_exactly_one_thirty_two_byte_abi_object() {
        let mut ports = BTreeMap::new();
        ports.insert(1, stereo_audio_port());
        let state = AudioOutState {
            initialized: true,
            next_handle: 1,
            successful_opens: 1,
            successful_closes: 0,
            ports,
            host: None,
        };
        let mut output = [0xa5_u8; 48];

        state
            .get_port_state([1, output.as_mut_ptr() as u64, 0, 0, 0, 0])
            .expect("an open port has a queryable state");

        assert_eq!(AUDIO_OUT_PORT_STATE_SIZE, 0x20);
        assert_eq!(u16::from_le_bytes(output[0..2].try_into().unwrap()), 1);
        assert_eq!(output[2], 2);
        assert_eq!(i16::from_le_bytes(output[4..6].try_into().unwrap()), -1);
        assert_eq!(&output[6..AUDIO_OUT_PORT_STATE_SIZE], &[0; 26]);
        assert_eq!(&output[AUDIO_OUT_PORT_STATE_SIZE..], &[0xa5; 16]);
    }

    #[test]
    fn contains_port_answers_without_writing_guest_abi_state() {
        let (state, handle) = open_timed_stereo_port();
        assert!(state.contains_port(handle));
        assert!(!state.contains_port(handle.wrapping_add(1)));
    }

    #[test]
    fn output_rejects_an_unreadable_pcm_pointer_without_faulting() {
        let (mut state, handle) = open_timed_stereo_port();
        assert_eq!(
            state.output([u64::from(handle.cast_unsigned()), 0x10, 0, 0, 0, 0]),
            Err(AudioOutError::InvalidPointer)
        );
    }

    #[test]
    fn set_volume_rejects_an_unreadable_volume_array_without_faulting() {
        let (mut state, handle) = open_timed_stereo_port();
        assert_eq!(
            state.set_volume([u64::from(handle.cast_unsigned()), 0x3, 0x10, 0, 0, 0]),
            Err(AudioOutError::InvalidPointer)
        );
    }

    #[test]
    fn get_port_state_rejects_an_unreadable_output_pointer_without_faulting() {
        let (state, handle) = open_timed_stereo_port();
        assert_eq!(
            state.get_port_state([u64::from(handle.cast_unsigned()), 0x10, 0, 0, 0, 0]),
            Err(AudioOutError::InvalidPointer)
        );
    }

    #[test]
    fn get_last_output_time_publishes_consumed_microseconds() {
        let mut state = AudioOutState::default();
        state.use_timed_sink_for_test();
        let handle = state
            .open([0, 0, 0, 256, 48_000, 1])
            .expect("timed open") as i32;
        // Timed sinks have no device queue; consumed stays zero.
        let mut output = 0xdead_u64;
        state
            .get_last_output_time([
                u64::from(handle.cast_unsigned()),
                std::ptr::from_mut(&mut output) as u64,
                0,
                0,
                0,
                0,
            ])
            .expect("open port reports a last-output time");
        assert_eq!(output, 0);

        // Device-backed ports publish consumed * 1e6 / rate.
        let queue = Arc::new(AudioQueue::with_capacity(AUDIO_PREBUFFER_FRAMES));
        queue.consumed.store(48_000, Ordering::Release);
        state.ports.get_mut(&handle).unwrap().queue = Some(queue);
        state
            .get_last_output_time([
                u64::from(handle.cast_unsigned()),
                std::ptr::from_mut(&mut output) as u64,
                0,
                0,
                0,
                0,
            ])
            .expect("consumed clock is readable");
        assert_eq!(output, 1_000_000);
    }

    #[test]
    fn audio_out2_float_gains_replace_the_backend_channel_gains() {
        let mut ports = BTreeMap::new();
        ports.insert(1, stereo_audio_port());
        let mut state = AudioOutState {
            initialized: true,
            next_handle: 1,
            successful_opens: 1,
            successful_closes: 0,
            ports,
            host: None,
        };
        let gains = [0.125_f32, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0];

        state
            .set_channel_gains(1, gains)
            .expect("the AudioOut2 port owns a live backend port");

        assert_eq!(state.ports.get(&1).unwrap().channel_volume, gains);
    }

    #[test]
    fn mixer_prebuffers_and_rebuffers_without_consuming_partial_callbacks() {
        reset_device_clock_for_test();
        let mut mixer = AudioMixer::default();
        let queue = Arc::new(AudioQueue::with_capacity(AUDIO_PREBUFFER_FRAMES));
        mixer.add_port(1, Arc::clone(&queue));
        assert!(mixer.push(
            1,
            std::iter::repeat_n([0.25, -0.25], AUDIO_PREBUFFER_FRAMES / 2)
        ));
        let mut output = vec![1.0_f32; 256 * 2];
        let snapshot = mixer.snapshot_queues();
        AudioMixer::write_interleaved(&snapshot, &mut output, 2);
        assert_eq!(output, vec![0.0; 256 * 2]);
        assert_eq!(
            queue.queued_frames(),
            AUDIO_PREBUFFER_FRAMES / 2,
            "startup silence must retain the partial prebuffer"
        );

        assert!(mixer.push(
            1,
            std::iter::repeat_n([0.25, -0.25], AUDIO_PREBUFFER_FRAMES / 2)
        ));
        let snapshot = mixer.snapshot_queues();
        AudioMixer::write_interleaved(&snapshot, &mut output, 2);
        assert!(output.chunks_exact(2).all(|frame| frame == [0.25, -0.25]));

        let queued = queue.queued_frames();
        let mut oversized_callback = vec![1.0_f32; (queued + 1) * 2];
        let snapshot = mixer.snapshot_queues();
        AudioMixer::write_interleaved(&snapshot, &mut oversized_callback, 2);
        assert_eq!(oversized_callback, vec![0.0; (queued + 1) * 2]);
        assert_eq!(
            queue.queued_frames(),
            queued,
            "an underrun must rebuffer instead of emitting a partial callback"
        );
        assert_eq!(queue.state.load(Ordering::Acquire), QUEUE_REBUFFERING);
    }

    #[test]
    fn rebuffering_silence_counts_as_underrun_frames() {
        reset_device_clock_for_test();
        let before = remu_gpu::audio_underrun_frames();
        let mut mixer = AudioMixer::default();
        let queue = Arc::new(AudioQueue::with_capacity(AUDIO_PREBUFFER_FRAMES));
        mixer.add_port(1, Arc::clone(&queue));
        assert!(mixer.push(
            1,
            std::iter::repeat_n([0.5, -0.5], AUDIO_PREBUFFER_FRAMES)
        ));
        let mut output = vec![0.0_f32; 256 * 2];
        let snapshot = mixer.snapshot_queues();
        AudioMixer::write_interleaved(&snapshot, &mut output, 2);
        assert_eq!(queue.state.load(Ordering::Acquire), QUEUE_RUNNING);

        // Starve the port so the next callback enters rebuffering, then count the silent frames.
        while queue.try_pop_for_test().is_some() {}
        queue.state.store(QUEUE_RUNNING, Ordering::Release);
        let snapshot = mixer.snapshot_queues();
        AudioMixer::write_interleaved(&snapshot, &mut output, 2);
        assert_eq!(queue.state.load(Ordering::Acquire), QUEUE_REBUFFERING);
        let after_starve = remu_gpu::audio_underrun_frames();
        assert!(
            after_starve >= before + 256,
            "the starvation callback itself reports underrun frames"
        );

        // While rebuffering for another full callback, every silent frame is an underrun.
        let before_rebuffer = remu_gpu::audio_underrun_frames();
        let snapshot = mixer.snapshot_queues();
        AudioMixer::write_interleaved(&snapshot, &mut output, 2);
        assert_eq!(
            remu_gpu::audio_underrun_frames(),
            before_rebuffer + 256,
            "rebuffering silence must not be invisible to U%"
        );
    }

    #[test]
    fn realtime_queue_retains_newest_frames_within_latency_budget() {
        let mut mixer = AudioMixer::default();
        let queue = Arc::new(AudioQueue::with_capacity(4));
        mixer.add_port(1, Arc::clone(&queue));
        assert_eq!(
            mixer.push_realtime_bounded(1, (0..6).map(|value| [value as f32, 0.0]).collect(), 4,),
            Some(2)
        );
        assert_eq!(
            (0..queue.queued_frames())
                .map(|_| queue.ring.try_pop().unwrap()[0])
                .collect::<Vec<_>>(),
            [2.0, 3.0, 4.0, 5.0]
        );

        // Refill after the pops above emptied the ring.
        assert_eq!(
            mixer.push_realtime_bounded(1, (2..6).map(|value| [value as f32, 0.0]).collect(), 4,),
            Some(0)
        );
        assert_eq!(
            mixer.push_realtime_bounded(1, (6..9).map(|value| [value as f32, 0.0]).collect(), 4,),
            Some(3)
        );
        assert_eq!(
            (0..queue.queued_frames())
                .map(|_| queue.ring.try_pop().unwrap()[0])
                .collect::<Vec<_>>(),
            [5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn mixer_reserves_the_complete_realtime_latency_window() {
        let mut mixer = AudioMixer::default();
        let maximum = 48_000 * MAX_QUEUED_AUDIO_SECONDS;
        let queue = Arc::new(AudioQueue::with_capacity(maximum));
        mixer.add_port(1, Arc::clone(&queue));

        assert!(
            queue.ring.capacity() >= maximum,
            "a bounded realtime push must not allocate while holding the callback mutex"
        );
    }

    #[test]
    fn device_clock_advances_with_mixer_callbacks() {
        reset_device_clock_for_test();
        advance_device_clock_for_test(480);
        assert_eq!(super::device_clock_frames(), Some(480));
        reset_device_clock_for_test();
        assert_eq!(super::device_clock_frames(), None);
    }
}
