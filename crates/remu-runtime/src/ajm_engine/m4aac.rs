// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! MPEG4-AAC decoder for the AJM engine (plan P2), driven by Symphonia's AAC codec.
//!
//! SDK: `SCE_AJM_CODEC_M4AAC_DEC = 2` (`m4aac_decoder.h:25`). Initialize parameters are
//! `SceAjmDecM4aacInitializeParameters` (`m4aac_decoder.h:214-220`): header type in
//! `uiConfigNumber` (ADTS=1, RAW=2/SAF=3 per `m4aac_decoder.h:145-148`) and a sampling-
//! frequency index used only for RAW (`m4aac_decoder.h:159-172`). HE-AAC/SBR is enabled by
//! the instance create flag `SCE_AJM_INSTANCE_FLAG_DEC_M4AAC_ENABLE_SBR_DECODE`
//! (`m4aac_decoder.h:49`); Symphonia's AAC path is LC — SBR streams still decode the LC
//! core when present, and a title that depends on HE reconstruction will surface that as
//! audible quality rather than a fabricated success.

use std::io::{Cursor, Read, Seek, SeekFrom};

use symphonia::core::audio::{AudioBufferRef, SampleBuffer};
use symphonia::core::codecs::{Decoder, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::{MediaSource, MediaSourceStream};
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// ADTS header type (`SCE_AJM_DEC_M4AAC_CONFIG_NUMBER_ADTS`, m4aac_decoder.h:145).
pub(crate) const M4AAC_CONFIG_ADTS: u32 = 1;
/// RAW / SAF header types (m4aac_decoder.h:146-147).
pub(crate) const M4AAC_CONFIG_RAW: u32 = 2;

/// Sampling-frequency index → Hz (m4aac_decoder.h:159-172).
pub(crate) fn sampling_freq_hz(index: u32) -> Option<u32> {
    Some(match index {
        0 => 96_000,
        1 => 88_200,
        2 => 64_000,
        3 => 48_000,
        4 => 44_100,
        5 => 32_000,
        6 => 24_000,
        7 => 22_050,
        8 => 16_000,
        9 => 12_000,
        10 => 11_025,
        11 => 8_000,
        _ => return None,
    })
}

/// One initialized MPEG4-AAC decoder instance.
pub(crate) struct M4aacDecoder {
    pub(crate) config_number: u32,
    #[allow(dead_code)] // retained for RAW-mode ASC construction when a title needs it
    pub(crate) sampling_freq_index: u32,
    pub(crate) channels: u32,
    pub(crate) sample_rate: u32,
    /// Set once HE-AAC is observed (`SceAjmSidebandDecM4aacCodecInfo.uiHeaac`).
    heaac: u32,
}

impl M4aacDecoder {
    pub(crate) fn new(config_number: u32, sampling_freq_index: u32, max_channels: u32) -> Option<Self> {
        if config_number != M4AAC_CONFIG_ADTS
            && config_number != M4AAC_CONFIG_RAW
            && config_number != 3
        {
            return None;
        }
        if config_number != M4AAC_CONFIG_ADTS && sampling_freq_hz(sampling_freq_index).is_none() {
            return None;
        }
        let sample_rate = if config_number == M4AAC_CONFIG_ADTS {
            0
        } else {
            sampling_freq_hz(sampling_freq_index)?
        };
        let channels = if max_channels == 0 { 2 } else { max_channels.min(8) };
        Some(Self {
            config_number,
            sampling_freq_index,
            channels,
            sample_rate,
            heaac: 0,
        })
    }

    /// Codec-info sideband (`SceAjmSidebandDecM4aacCodecInfo`, m4aac_decoder.h:229-235).
    pub(crate) fn codec_info_bytes(&self) -> [u8; 8] {
        let mut out = [0_u8; 8];
        out[0..4].copy_from_slice(&self.heaac.to_le_bytes());
        out
    }

    /// Decodes ADTS AAC from `input` into interleaved S16 PCM, stopping before exceeding
    /// `max_samples` interleaved samples. Returns produced samples, bytes consumed from
    /// `input`, and the sample-frame count.
    pub(crate) fn decode_s16(
        &mut self,
        input: &[u8],
        max_samples: usize,
    ) -> Option<(Vec<i16>, usize, u64)> {
        if input.is_empty() || max_samples == 0 {
            return Some((Vec::new(), 0, 0));
        }
        // ADTS is what titles feed AJM for MPEG4-AAC in practice; RAW without an ASC is
        // accepted at initialize but still needs framed access units — treat the buffer as
        // ADTS when it syncs, otherwise refuse rather than inventing an ASC.
        if self.config_number != M4AAC_CONFIG_ADTS && !looks_like_adts(input) {
            return None;
        }
        if !looks_like_adts(input) {
            return None;
        }

        let source = CursorSource::new(input.to_vec());
        let mss = MediaSourceStream::new(Box::new(source), Default::default());
        let mut hint = Hint::new();
        hint.with_extension("aac");
        let probed = symphonia::default::get_probe()
            .format(
                &hint,
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .ok()?;
        let mut format: Box<dyn FormatReader> = probed.format;
        let track = format.tracks().first()?.clone();
        let track_id = track.id;
        if let Some(rate) = track.codec_params.sample_rate {
            self.sample_rate = rate;
        }
        if let Some(channels) = track.codec_params.channels {
            self.channels = channels.count() as u32;
        }
        let mut decoder: Box<dyn Decoder> = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .ok()?;

        let mut produced: Vec<i16> = Vec::new();
        let mut frames = 0_u64;
        let mut packets_ok = 0_u64;
        loop {
            if produced.len() >= max_samples {
                break;
            }
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(SymphoniaError::ResetRequired) => continue,
                Err(_) => break,
            };
            if packet.track_id() != track_id {
                continue;
            }
            match decoder.decode(&packet) {
                Ok(audio) => {
                    let samples = interleaved_i16(&audio)?;
                    let room = max_samples.saturating_sub(produced.len());
                    if samples.len() > room {
                        // A frame that would overrun the guest output is left unconsumed so
                        // the next run can retry with a larger buffer.
                        break;
                    }
                    let channels = self.channels.max(1) as u64;
                    frames += samples.len() as u64 / channels;
                    produced.extend_from_slice(&samples);
                    packets_ok += 1;
                }
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(_) => break,
            }
        }
        let consumed = adts_bytes_for_packets(input, packets_ok);
        Some((produced, consumed, frames))
    }
}

fn looks_like_adts(input: &[u8]) -> bool {
    input.len() >= 7 && input[0] == 0xff && (input[1] & 0xf0) == 0xf0
}

/// Sums the ADTS frame lengths of the first `packet_count` frames in `input`.
fn adts_bytes_for_packets(input: &[u8], packet_count: u64) -> usize {
    let mut offset = 0_usize;
    let mut seen = 0_u64;
    while seen < packet_count {
        let Some(frame_len) = adts_frame_len(&input[offset..]) else {
            break;
        };
        offset += frame_len;
        seen += 1;
    }
    offset
}

fn adts_frame_len(data: &[u8]) -> Option<usize> {
    if data.len() < 7 || data[0] != 0xff || (data[1] & 0xf0) != 0xf0 {
        return None;
    }
    let len = (((data[3] & 0x03) as usize) << 11)
        | ((data[4] as usize) << 3)
        | ((data[5] as usize) >> 5);
    if len < 7 || len > data.len() {
        None
    } else {
        Some(len)
    }
}

fn interleaved_i16(audio: &AudioBufferRef<'_>) -> Option<Vec<i16>> {
    let mut buf = SampleBuffer::<i16>::new(audio.capacity() as u64, *audio.spec());
    buf.copy_interleaved_ref(audio.clone());
    Some(buf.samples().to_vec())
}

/// Seekable in-memory media source for Symphonia.
struct CursorSource {
    inner: Cursor<Vec<u8>>,
}

impl CursorSource {
    fn new(data: Vec<u8>) -> Self {
        Self {
            inner: Cursor::new(data),
        }
    }
}

impl Read for CursorSource {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buf)
    }
}

impl Seek for CursorSource {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(pos)
    }
}

impl MediaSource for CursorSource {
    fn is_seekable(&self) -> bool {
        true
    }

    fn byte_len(&self) -> Option<u64> {
        Some(self.inner.get_ref().len() as u64)
    }
}
