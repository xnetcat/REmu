// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! FFI to the in-process libav bridge (plan P5).
//!
//! Production AvPlayer/Videodec2 decode goes through these symbols. Cargo builds and statically
//! links the pinned FFmpeg subset for the active macOS target.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::ptr;
use std::time::Duration;

#[repr(C)]
pub(crate) struct RemuLibavDemux {
    _private: [u8; 0],
}

#[repr(C)]
pub(crate) struct RemuLibavCodec {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn remu_libav_available() -> c_int;
    fn remu_libav_probe_video(
        path: *const c_char,
        width_out: *mut i32,
        height_out: *mut i32,
        fps_out: *mut f64,
        duration_out: *mut f64,
    ) -> c_int;
    fn remu_libav_probe_has_audio(path: *const c_char) -> c_int;
    fn remu_libav_demux_open(
        path: *const c_char,
        audio: c_int,
        seek_seconds: f64,
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> *mut RemuLibavDemux;
    fn remu_libav_demux_read_frame(
        session: *mut RemuLibavDemux,
        out: *mut u8,
        out_len: usize,
    ) -> c_int;
    fn remu_libav_demux_close(session: *mut RemuLibavDemux);
    fn remu_libav_codec_open(
        codec_id: u32,
        err_buf: *mut c_char,
        err_buf_len: usize,
    ) -> *mut RemuLibavCodec;
    fn remu_libav_codec_send(
        codec: *mut RemuLibavCodec,
        au: *const u8,
        au_size: usize,
        pts: i64,
    ) -> c_int;
    fn remu_libav_codec_receive_nv12(
        codec: *mut RemuLibavCodec,
        out: *mut u8,
        out_len: usize,
        width_out: *mut i32,
        height_out: *mut i32,
        pitch_out: *mut i32,
    ) -> c_int;
    fn remu_libav_codec_flush(codec: *mut RemuLibavCodec) -> c_int;
    fn remu_libav_codec_close(codec: *mut RemuLibavCodec);
}

pub(crate) fn libav_available() -> bool {
    // SAFETY: remu_libav_available is a pure status query with no pointer arguments.
    unsafe { remu_libav_available() != 0 }
}

pub(crate) fn probe_video(path: &Path) -> Option<(u32, u32, f64, Duration)> {
    let path = CString::new(path.to_string_lossy().as_bytes()).ok()?;
    let mut width = 0_i32;
    let mut height = 0_i32;
    let mut fps = 0.0_f64;
    let mut duration = 0.0_f64;
    // SAFETY: path is a NUL-terminated CString; outputs are stack locals.
    let ok = unsafe {
        remu_libav_probe_video(
            path.as_ptr(),
            &mut width,
            &mut height,
            &mut fps,
            &mut duration,
        )
    };
    if ok == 0 || width <= 0 || height <= 0 || !fps.is_finite() || fps <= 0.0 {
        return None;
    }
    Some((
        width as u32,
        height as u32,
        fps,
        if duration.is_finite() && duration > 0.0 {
            Duration::from_secs_f64(duration)
        } else {
            Duration::ZERO
        },
    ))
}

pub(crate) fn probe_has_audio(path: &Path) -> bool {
    let Ok(path) = CString::new(path.to_string_lossy().as_bytes()) else {
        return false;
    };
    // SAFETY: path is a NUL-terminated CString.
    unsafe { remu_libav_probe_has_audio(path.as_ptr()) != 0 }
}

pub(crate) struct LibavDemuxSession {
    raw: *mut RemuLibavDemux,
}

impl std::fmt::Debug for LibavDemuxSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LibavDemuxSession")
            .field("raw", &self.raw)
            .finish()
    }
}

// The C session is confined to one worker thread; Sync is required because
// ProbeDispatchState lives behind an RwLock shared across guest threads.
unsafe impl Send for LibavDemuxSession {}
unsafe impl Sync for LibavDemuxSession {}

impl LibavDemuxSession {
    pub(crate) fn open(path: &Path, audio: bool, seek: Duration) -> Result<Self, String> {
        if !libav_available() {
            return Err(
                "in-process libavcodec failed its build-time availability contract".to_owned(),
            );
        }
        let path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| "media path contains interior NUL".to_owned())?;
        let mut err = vec![0_u8; 256];
        // SAFETY: path/err are valid buffers for the duration of the call.
        let raw = unsafe {
            remu_libav_demux_open(
                path.as_ptr(),
                i32::from(audio),
                seek.as_secs_f64(),
                err.as_mut_ptr().cast(),
                err.len(),
            )
        };
        if raw.is_null() {
            let message = CStr::from_bytes_until_nul(&err)
                .ok()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "libav demux open failed".to_owned());
            return Err(message);
        }
        Ok(Self { raw })
    }

    /// 0 = frame, 1 = EOF, -1 = error.
    pub(crate) fn read_frame(&mut self, out: &mut [u8]) -> i32 {
        // SAFETY: `raw` is a live session owned by this wrapper; `out` is a writable host buffer.
        unsafe { remu_libav_demux_read_frame(self.raw, out.as_mut_ptr(), out.len()) }
    }
}

impl Drop for LibavDemuxSession {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: closes the session exactly once.
            unsafe { remu_libav_demux_close(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

pub(crate) struct LibavCodecSession {
    raw: *mut RemuLibavCodec,
}

impl std::fmt::Debug for LibavCodecSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LibavCodecSession")
            .field("raw", &self.raw)
            .finish()
    }
}

unsafe impl Send for LibavCodecSession {}
unsafe impl Sync for LibavCodecSession {}

impl LibavCodecSession {
    pub(crate) fn open(codec_id: u32) -> Result<Self, String> {
        if !libav_available() {
            return Err(
                "in-process libavcodec failed its build-time availability contract".to_owned(),
            );
        }
        let mut err = vec![0_u8; 256];
        // SAFETY: err is a writable host buffer.
        let raw = unsafe { remu_libav_codec_open(codec_id, err.as_mut_ptr().cast(), err.len()) };
        if raw.is_null() {
            let message = CStr::from_bytes_until_nul(&err)
                .ok()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "libav codec open failed".to_owned());
            return Err(message);
        }
        Ok(Self { raw })
    }

    pub(crate) fn send(&mut self, au: &[u8], pts: i64) -> Result<(), ()> {
        // SAFETY: live codec; au is readable for au.len().
        let rc = unsafe {
            remu_libav_codec_send(
                self.raw,
                if au.is_empty() {
                    ptr::null()
                } else {
                    au.as_ptr()
                },
                au.len(),
                pts,
            )
        };
        if rc == 0 { Ok(()) } else { Err(()) }
    }

    /// Ok(Some) = frame, Ok(None) = need more input / drained, Err = hard failure.
    pub(crate) fn receive_nv12(&mut self, out: &mut [u8]) -> Result<Option<(u32, u32, u32)>, ()> {
        let mut width = 0_i32;
        let mut height = 0_i32;
        let mut pitch = 0_i32;
        // SAFETY: live codec; out/locals are valid.
        let rc = unsafe {
            remu_libav_codec_receive_nv12(
                self.raw,
                out.as_mut_ptr(),
                out.len(),
                &mut width,
                &mut height,
                &mut pitch,
            )
        };
        match rc {
            0 => Ok(Some((width as u32, height as u32, pitch as u32))),
            1 => Ok(None),
            _ => Err(()),
        }
    }

    pub(crate) fn flush(&mut self) -> Result<(), ()> {
        // SAFETY: live codec.
        if unsafe { remu_libav_codec_flush(self.raw) } == 0 {
            Ok(())
        } else {
            Err(())
        }
    }
}

impl Drop for LibavCodecSession {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: closes the codec exactly once.
            unsafe { remu_libav_codec_close(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}
