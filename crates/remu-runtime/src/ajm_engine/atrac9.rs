// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! A thin safe wrapper over the vendored LibAtrac9 decoder (see
//! `vendored/libatrac9/PROVENANCE.md`), bound for the AJM engine's ATRAC9 codec (plan P1).
//!
//! Only the five public entry points from `libatrac9.h` are used: get/release a decoder
//! handle, initialize it from the 4-byte ATRAC9 configuration data, decode one frame, and
//! query the stream geometry. The handle is opaque C state (a `calloc`'d `Atrac9Handle` that
//! carries the per-channel IMDCT overlap across frames), so a decoder instance must decode a
//! stream in order and must be released exactly once.

use std::os::raw::{c_int, c_uchar, c_void};

/// `Atrac9CodecInfo` (libatrac9.h:18-27). Mirrors the C layout field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Atrac9CodecInfo {
    channels: c_int,
    channel_config_index: c_int,
    sampling_rate: c_int,
    superframe_size: c_int,
    frames_in_superframe: c_int,
    frame_samples: c_int,
    wlength: c_int,
    config_data: [c_uchar; 4],
}

// Vendored LibAtrac9 (libatrac9.h:29-35), compiled by build.rs.
unsafe extern "C" {
    fn Atrac9GetHandle() -> *mut c_void;
    fn Atrac9ReleaseHandle(handle: *mut c_void);
    fn Atrac9InitDecoder(handle: *mut c_void, config: *mut c_uchar) -> c_int;
    fn Atrac9Decode(
        handle: *mut c_void,
        input: *const c_uchar,
        pcm: *mut i16,
        bytes_used: *mut c_int,
    ) -> c_int;
    fn Atrac9GetCodecInfo(handle: *mut c_void, info: *mut Atrac9CodecInfo) -> c_int;
}

/// One decoded ATRAC9 frame: interleaved S16 samples and the bytes the frame consumed.
pub(crate) struct Atrac9Frame {
    pub(crate) samples: Vec<i16>,
    pub(crate) bytes_used: usize,
}

/// An initialized ATRAC9 decoder instance owning its LibAtrac9 handle.
pub(crate) struct Atrac9Decoder {
    handle: *mut c_void,
    pub(crate) channels: u32,
    pub(crate) sample_rate: u32,
    pub(crate) superframe_size: u32,
    pub(crate) frames_in_superframe: u32,
    pub(crate) frame_samples: u32,
}

// The handle is touched only while the engine's own mutex is held (one job at a time against
// an instance), never concurrently, so the decoder may move between the worker threads that
// drain the batch queue. It is deliberately not `Sync`: nothing shares one by reference.
unsafe impl Send for Atrac9Decoder {}

impl Atrac9Decoder {
    /// Initializes a decoder from the 4-byte ATRAC9 configuration data delivered in the AJM
    /// `SceAjmDecAt9InitializeParameters.uiConfigData` (at9_decoder.h:268-274), returning
    /// `None` if LibAtrac9 rejects the configuration (`ERR_BAD_CONFIG_DATA`).
    pub(crate) fn new(config: [u8; 4]) -> Option<Self> {
        // SAFETY: `Atrac9GetHandle` returns an owned `calloc`'d handle or null.
        let handle = unsafe { Atrac9GetHandle() };
        if handle.is_null() {
            return None;
        }
        let mut config = config;
        // SAFETY: `handle` is a live decoder; LibAtrac9 reads 4 bytes of `config`.
        let init = unsafe { Atrac9InitDecoder(handle, config.as_mut_ptr()) };
        let mut info = Atrac9CodecInfo::default();
        // SAFETY: `handle` is live; `info` is a valid out-parameter of the right type.
        let query = unsafe { Atrac9GetCodecInfo(handle, &mut info) };
        if init != 0 || query != 0 || info.channels <= 0 || info.frame_samples <= 0 {
            // SAFETY: releasing the handle we just acquired; not used afterwards.
            unsafe { Atrac9ReleaseHandle(handle) };
            return None;
        }
        Some(Self {
            handle,
            channels: info.channels.unsigned_abs(),
            sample_rate: info.sampling_rate.unsigned_abs(),
            superframe_size: info.superframe_size.unsigned_abs(),
            frames_in_superframe: info.frames_in_superframe.unsigned_abs(),
            frame_samples: info.frame_samples.unsigned_abs(),
        })
    }

    /// Decodes one ATRAC9 frame from the front of `input`, advancing the instance's decode
    /// state. `input` must span at least the remaining bytes of the current superframe so the
    /// bit reader never runs past the buffer. Returns `None` on a stream error.
    pub(crate) fn decode_frame(&self, input: &[u8]) -> Option<Atrac9Frame> {
        let sample_count = (self.frame_samples * self.channels) as usize;
        if input.is_empty() || sample_count == 0 {
            return None;
        }
        let mut samples = vec![0_i16; sample_count];
        let mut bytes_used: c_int = 0;
        // SAFETY: `handle` is live; `input` is a readable host buffer at least one frame long;
        // `samples` holds exactly `frame_samples * channels` shorts, the fixed output of one
        // frame (decoder.c:46-60); `bytes_used` is a valid out-parameter.
        let status = unsafe {
            Atrac9Decode(
                self.handle,
                input.as_ptr(),
                samples.as_mut_ptr(),
                &mut bytes_used,
            )
        };
        if status != 0 || bytes_used <= 0 {
            return None;
        }
        Some(Atrac9Frame {
            samples,
            bytes_used: bytes_used.unsigned_abs() as usize,
        })
    }
}

impl Drop for Atrac9Decoder {
    fn drop(&mut self) {
        // SAFETY: the handle was acquired in `new`, is released exactly once here, and is not
        // used afterwards.
        unsafe { Atrac9ReleaseHandle(self.handle) };
    }
}
