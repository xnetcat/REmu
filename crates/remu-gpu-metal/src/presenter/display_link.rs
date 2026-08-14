// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! A CoreVideo display link stamping the process display timebase (plan E1).
//!
//! Every hardware vblank stamps [`remu_gpu::record_display_vblank`] and wakes the winit event
//! loop, so presentation submits phase-aligned right after a vblank — with a whole refresh
//! period of margin to the next deadline — instead of drifting into the sticky half-rate mode
//! where every present misses its vsync by a hair and slips a full frame.

use std::ffi::c_void;
use std::time::{Duration, Instant};

/// PS5's fixed display-vblank period. The fallback must keep stamping one shared clock rather
/// than leaving every presentation/wait caller to free-run when CoreVideo temporarily has no
/// active display (for example after a display sleep or on newer macOS releases where the
/// deprecated process display-link API can return `kCVReturnInvalidArgument`).
const SYNTHETIC_VBLANK_PERIOD: Duration = Duration::from_nanos(1_000_000_000 / 60);

// CoreVideo's C display-link API. The macOS 15 SDK marks it deprecated in favour of the
// view-owned CADisplayLink, but it remains the only variant reachable without an Objective-C
// target object, and these raw declarations carry no deprecation attribute to warn on.
#[repr(C)]
struct CVTimeStampRaw {
    version: u32,
    video_time_scale: i32,
    video_time: i64,
    host_time: u64,
    rate_scalar: f64,
    video_refresh_period: i64,
    smpte_time: [u8; 32],
    flags: u64,
    reserved: u64,
}

type CVDisplayLinkOutputCallback = extern "C" fn(
    display_link: *mut c_void,
    in_now: *const CVTimeStampRaw,
    in_output_time: *const CVTimeStampRaw,
    flags_in: u64,
    flags_out: *mut u64,
    context: *mut c_void,
) -> i32;

#[link(name = "CoreVideo", kind = "framework")]
unsafe extern "C" {
    fn CVDisplayLinkCreateWithActiveCGDisplays(link_out: *mut *mut c_void) -> i32;
    fn CVDisplayLinkSetOutputCallback(
        link: *mut c_void,
        callback: CVDisplayLinkOutputCallback,
        context: *mut c_void,
    ) -> i32;
    fn CVDisplayLinkStart(link: *mut c_void) -> i32;
    fn CVDisplayLinkGetActualOutputVideoRefreshPeriod(link: *mut c_void) -> f64;
}

extern "C" fn display_link_fired(
    link: *mut c_void,
    _in_now: *const CVTimeStampRaw,
    _in_output_time: *const CVTimeStampRaw,
    _flags_in: u64,
    _flags_out: *mut u64,
    context: *mut c_void,
) -> i32 {
    // SAFETY: `start_display_link` leaks the waker box for the process lifetime, and the link
    // is never stopped, so the context stays valid for every callback.
    let waker = unsafe { &*context.cast::<winit::event_loop::EventLoopProxy<()>>() };
    // SAFETY: the link handle the callback receives is the live link this callback registered on.
    let seconds_per_frame = unsafe { CVDisplayLinkGetActualOutputVideoRefreshPeriod(link) };
    let period_nanos = if seconds_per_frame.is_finite() && seconds_per_frame > 0.0 {
        (seconds_per_frame * 1_000_000_000.0) as u64
    } else {
        0
    };
    remu_gpu::record_display_vblank(remu_gpu::monotonic_nanos(), period_nanos);
    let _ = waker.send_event(());
    0
}

fn start_synthetic_display_link(waker: winit::event_loop::EventLoopProxy<()>, reason: &str) {
    remu_gpu::set_display_mode(remu_gpu::DisplayMode::Synthetic);
    eprintln!("[REMu][METAL] {reason}; pacing on the process-wide 60 Hz synthetic vblank timebase");
    std::thread::Builder::new()
        .name("remu-display-clock".to_owned())
        .spawn(move || {
            let mut next_vblank = Instant::now();
            loop {
                next_vblank += SYNTHETIC_VBLANK_PERIOD;
                let now = Instant::now();
                if next_vblank > now {
                    std::thread::sleep(next_vblank - now);
                } else if now.duration_since(next_vblank) >= SYNTHETIC_VBLANK_PERIOD {
                    // A suspended process must not replay accumulated vblanks in a tight loop.
                    next_vblank = now;
                }
                remu_gpu::record_display_vblank(
                    remu_gpu::monotonic_nanos(),
                    SYNTHETIC_VBLANK_PERIOD.as_nanos() as u64,
                );
                if waker.send_event(()).is_err() {
                    break;
                }
            }
        })
        .expect("the synthetic display-clock thread must start");
}

/// Starts the process display link, or reports why the timebase will stay synthetic.
///
/// The link runs for the process lifetime on CoreVideo's own thread; there is deliberately no
/// stop path, matching the presenter's lifetime.
pub(super) fn start_display_link(waker: winit::event_loop::EventLoopProxy<()>) {
    let mut link: *mut c_void = std::ptr::null_mut();
    // SAFETY: plain C calls with a link-out parameter, a callback with a leaked 'static
    // context, and a start on the created link, all checked for CoreVideo's kCVReturnSuccess.
    unsafe {
        let create_status = CVDisplayLinkCreateWithActiveCGDisplays(&raw mut link);
        if create_status != 0 || link.is_null() {
            start_synthetic_display_link(
                waker,
                &format!("display link unavailable (CoreVideo status={create_status})"),
            );
            return;
        }
        let fallback_waker = waker.clone();
        let context = Box::into_raw(Box::new(waker)).cast::<c_void>();
        let callback_status = CVDisplayLinkSetOutputCallback(link, display_link_fired, context);
        let start_status = if callback_status == 0 {
            CVDisplayLinkStart(link)
        } else {
            callback_status
        };
        if callback_status != 0 || start_status != 0 {
            start_synthetic_display_link(
                fallback_waker,
                &format!(
                    "display link refused to start (callback status={callback_status}, start status={start_status})"
                ),
            );
            return;
        }
    }
    remu_gpu::set_display_mode(remu_gpu::DisplayMode::CvDisplayLink);
    eprintln!("[REMu][METAL] display link started; presentation paces on the vblank timebase");
}
