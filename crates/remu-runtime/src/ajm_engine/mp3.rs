// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! MP3 decoder for the AJM engine (plan P6), backed by the vendored minimp3 single-header
//! decoder (`vendored/minimp3/PROVENANCE.md`).
//!
//! SDK: `SCE_AJM_CODEC_MP3_DEC = 0` (`mp3_decoder.h:30`). MP3 needs no initialize-parameter
//! blob — geometry comes from each frame header — so `initialize` is a no-op success and the
//! first `run` that sees a valid frame fills channels/rate. Codec-info sideband is
//! `SceAjmSidebandDecMp3CodecInfo` (`mp3_decoder.h:88-106`).

use std::os::raw::c_int;

/// `mp3dec_frame_info_t` (minimp3.h) — field order matches the header.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Mp3decFrameInfo {
    frame_bytes: c_int,
    frame_offset: c_int,
    channels: c_int,
    hz: c_int,
    layer: c_int,
    bitrate_kbps: c_int,
}

/// `mp3dec_t` (minimp3.h:18-23). Sized to match `sizeof(mp3dec_t) == 6668` on this host.
#[repr(C)]
struct Mp3dec {
    mdct_overlap: [[f32; 9 * 32]; 2],
    qmf_state: [f32; 15 * 2 * 32],
    reserv: c_int,
    free_format_bytes: c_int,
    header: [u8; 4],
    reserv_buf: [u8; 511],
}

unsafe extern "C" {
    fn mp3dec_init(dec: *mut Mp3dec);
    fn mp3dec_decode_frame(
        dec: *mut Mp3dec,
        mp3: *const u8,
        mp3_bytes: c_int,
        pcm: *mut i16,
        info: *mut Mp3decFrameInfo,
    ) -> c_int;
}

/// One initialized MP3 decoder instance.
pub(crate) struct Mp3Decoder {
    dec: Box<Mp3dec>,
    pub(crate) channels: u32,
    pub(crate) sample_rate: u32,
    /// Last frame's MPEG header word for the codec-info sideband.
    last_header: u32,
    last_crc: u8,
    last_mode: u8,
}

// Touched only under the engine mutex, same Send story as ATRAC9.
unsafe impl Send for Mp3Decoder {}

impl Mp3Decoder {
    pub(crate) fn new() -> Self {
        let mut dec = Box::new(Mp3dec {
            mdct_overlap: [[0.0; 9 * 32]; 2],
            qmf_state: [0.0; 15 * 2 * 32],
            reserv: 0,
            free_format_bytes: 0,
            header: [0; 4],
            reserv_buf: [0; 511],
        });
        // SAFETY: `dec` matches `mp3dec_t`; minimp3 only writes its own fields.
        unsafe { mp3dec_init(dec.as_mut()) };
        Self {
            dec,
            channels: 0,
            sample_rate: 0,
            last_header: 0,
            last_crc: 0,
            last_mode: 0,
        }
    }

    /// Decodes successive MP3 frames from `input` into interleaved S16, stopping before
    /// exceeding `max_samples`. Returns produced samples, bytes consumed, and frame count
    /// (MPEG frames, not sample frames).
    pub(crate) fn decode_s16(
        &mut self,
        input: &[u8],
        max_samples: usize,
    ) -> Option<(Vec<i16>, usize, u64)> {
        if input.is_empty() || max_samples == 0 {
            return Some((Vec::new(), 0, 0));
        }
        let mut produced = Vec::new();
        let mut consumed = 0_usize;
        let mut frames = 0_u64;
        // minimp3 writes at most 1152 * 2 samples per call (mp3_decoder.h:41).
        let mut pcm = vec![0_i16; 1152 * 2];
        while consumed < input.len() && produced.len() < max_samples {
            let remain = &input[consumed..];
            let mut info = Mp3decFrameInfo::default();
            // SAFETY: decoder is live; `remain` is readable; `pcm` holds the max frame;
            // `info` is a valid out-parameter.
            let samples = unsafe {
                mp3dec_decode_frame(
                    self.dec.as_mut(),
                    remain.as_ptr(),
                    c_int::try_from(remain.len()).unwrap_or(c_int::MAX),
                    pcm.as_mut_ptr(),
                    &mut info,
                )
            };
            if info.frame_bytes <= 0 {
                break;
            }
            let frame_bytes = info.frame_bytes as usize;
            if samples > 0 {
                let channels = info.channels.max(1) as usize;
                let sample_count = (samples as usize) * channels;
                if produced.len() + sample_count > max_samples {
                    break;
                }
                produced.extend_from_slice(&pcm[..sample_count]);
                self.channels = info.channels.unsigned_abs();
                self.sample_rate = info.hz.unsigned_abs();
                // Reconstruct a plausible 32-bit header word for the sideband from the
                // bytes minimp3 just accepted (big-endian sync word at the frame start).
                if remain.len() >= 4 {
                    self.last_header = u32::from_be_bytes([remain[0], remain[1], remain[2], remain[3]]);
                    self.last_crc = if (remain[1] & 0x01) == 0 { 0 } else { 1 };
                    self.last_mode = (remain[3] >> 6) & 0x3;
                }
                frames += 1;
            }
            consumed += frame_bytes;
        }
        Some((produced, consumed, frames))
    }

    /// `SceAjmSidebandDecMp3CodecInfo` (mp3_decoder.h:88-106) — 16 bytes.
    pub(crate) fn codec_info_bytes(&self) -> [u8; 16] {
        let mut out = [0_u8; 16];
        out[0..4].copy_from_slice(&self.last_header.to_le_bytes());
        out[4] = self.last_crc;
        out[5] = self.last_mode;
        // mode_extension / copyright / original / emphasis left zero when unknown.
        out
    }
}
