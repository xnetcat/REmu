use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::{
    ProbeDispatchState, agc_submit_ordered_video_out_flip, guest_memset, low_i32, low_u32,
    queue_video_out_flip, sce_kernel_error, write_guest_u32, write_guest_u64,
};

pub(crate) fn video_out_set_buffer_attribute2(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    let attribute = arguments[0];
    if attribute == 0 {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_ADDRESS);
    }
    guest_memset(attribute, 0, 0x50);
    write_guest_u32(attribute + 0x04, low_u32(arguments[2]));
    write_guest_u32(attribute + 0x0c, low_u32(arguments[3]));
    write_guest_u32(attribute + 0x10, low_u32(arguments[4]));
    write_guest_u64(attribute + 0x18, arguments[5]);
    write_guest_u64(attribute + 0x20, arguments[1]);

    state.video_out.pixel_format = arguments[1];
    state.video_out.tiling_mode = low_u32(arguments[2]);
    state.video_out.width = low_u32(arguments[3]);
    state.video_out.height = low_u32(arguments[4]);
    0
}

pub(crate) const VIDEO_OUT_ERROR_INVALID_VALUE: u32 = 0x8029_0001;
pub(crate) const VIDEO_OUT_ERROR_INVALID_ADDRESS: u32 = 0x8029_0002;
pub(crate) const VIDEO_OUT_ERROR_RESOURCE_BUSY: u32 = 0x8029_0009;
pub(crate) const VIDEO_OUT_ERROR_INVALID_INDEX: u32 = 0x8029_000a;
pub(crate) const VIDEO_OUT_ERROR_INVALID_HANDLE: u32 = 0x8029_000b;

pub(crate) fn video_out_open(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    let user_id = low_i32(arguments[0]);
    let bus_type = low_i32(arguments[1]);
    let index = low_i32(arguments[2]);
    if bus_type != 0 || index != 0 {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_VALUE);
    }
    if state.video_out.is_open {
        return sce_kernel_error(VIDEO_OUT_ERROR_RESOURCE_BUSY);
    }
    state.video_out.is_open = true;
    state.video_out.user_id = user_id;
    state.video_out.bus_type = bus_type;
    state.video_out.index = index;
    1
}

pub(crate) fn video_out_get_output_status(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    if low_i32(arguments[0]) != 1 || !state.video_out.is_open {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    let output = arguments[1];
    if output == 0 {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_ADDRESS);
    }
    guest_memset(output, 0, 0x30);
    let resolution_class =
        u32::from(state.video_out.width >= 3840 || state.video_out.height >= 2160)
            .saturating_add(1);
    write_guest_u32(output, resolution_class);
    write_guest_u32(output + 4, 1);
    write_guest_u64(output + 8, 60);
    0
}

/// Interval the display holds each flip, in microseconds. Zero disables pacing entirely.
pub(crate) static VIDEO_OUT_FLIP_INTERVAL: AtomicU64 =
    AtomicU64::new(DEFAULT_FLIP_INTERVAL_MICROSECONDS);
/// Earliest instant the next flip may complete.
static VIDEO_OUT_NEXT_FLIP: OnceLock<Mutex<Option<Instant>>> = OnceLock::new();
/// The console's default flip rate is one flip per 60 Hz vblank.
pub(crate) const DEFAULT_FLIP_INTERVAL_MICROSECONDS: u64 = 1_000_000 / 60;

/// Microseconds per flip for a `sceVideoOutSetFlipRate` selector.
pub(crate) const fn flip_interval_microseconds(rate: i32) -> u64 {
    match rate {
        1 => 1_000_000 / 30,
        2 => 1_000_000 / 20,
        _ => DEFAULT_FLIP_INTERVAL_MICROSECONDS,
    }
}

/// Overrides the paced flip rate, in Hz. `0` presents as fast as the guest submits.
fn flip_rate_override() -> Option<u64> {
    static OVERRIDE: OnceLock<Option<u64>> = OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        std::env::var("REMU_FLIP_RATE_HZ")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
    })
}

/// Maximum accepted VideoOut flips allowed to wait for host presentation.
///
/// Three retains normal double/triple buffering while preventing movie and transition frames from
/// advancing minutes ahead of a slow translator. `0` disables this diagnostic safety valve.
fn flip_presentation_backlog_limit() -> Option<usize> {
    static LIMIT: OnceLock<Option<usize>> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("REMU_FLIP_PRESENT_BACKLOG")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .map_or(Some(3), |value| (value != 0).then_some(value))
    })
}

fn effective_flip_interval_microseconds() -> u64 {
    match flip_rate_override() {
        Some(0) => 0,
        Some(hertz) => 1_000_000 / hertz.max(1),
        None => VIDEO_OUT_FLIP_INTERVAL.load(Ordering::Relaxed),
    }
}

/// Tracks whether a backlogged presenter is making net progress.
///
/// A single Metal command buffer can legitimately take several seconds while a large scene is
/// being compiled and uploaded. An absolute timeout lets the guest resume after that interval
/// even when the presenter is draining normally, which immediately rebuilds the queue and lets
/// movie/audio state run ahead of the visible frame. Only a new low-water mark resets the stall
/// timer; queue churn above that mark is not progress toward the configured limit.
struct BackpressureProgress {
    lowest_pending: usize,
    last_progress: Instant,
}

impl BackpressureProgress {
    fn new(pending: usize, now: Instant) -> Self {
        Self {
            lowest_pending: pending,
            last_progress: now,
        }
    }

    fn stalled(&mut self, pending: usize, now: Instant, timeout: Duration) -> bool {
        if pending < self.lowest_pending {
            self.lowest_pending = pending;
            self.last_progress = now;
        }
        now.saturating_duration_since(self.last_progress) >= timeout
    }
}

/// Holds the submitting thread until its flip is due.
///
/// Without this the guest's frame loop is never gated: flips complete the instant they are
/// submitted, so a title that advances simulation once per displayed frame runs as fast as the
/// host can retire draws instead of at the rate it selected.
pub(crate) fn pace_video_out_flip() {
    // An accepted flip only depends on work already submitted before it, so this finite queue is
    // safe to pace. Do not apply the same policy to the aggregate GPU-event count: an async queue
    // may legally wait for a label written by a later graphics submission, and pausing the guest
    // before that producer is submitted creates a host-only dependency cycle.
    const MAX_BACKPRESSURE_STALL: Duration = Duration::from_secs(5);
    let interval = effective_flip_interval_microseconds();
    if interval == 0 {
        return;
    }
    let interval = Duration::from_micros(interval);
    let now = Instant::now();
    let mut next = VIDEO_OUT_NEXT_FLIP
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // A guest that falls behind resumes from the present instant rather than accumulating debt
    // and then running unbounded to catch up.
    let due = (*next).filter(|due| *due > now).unwrap_or(now);
    *next = Some(due + interval);
    drop(next);
    let delay = due.saturating_duration_since(now);
    if !delay.is_zero() {
        std::thread::sleep(delay);
    }
    if let Some(backlog_limit) = flip_presentation_backlog_limit() {
        let mut progress = BackpressureProgress::new(remu_gpu::gpu_flips_pending(), Instant::now());
        let mut logged = false;
        while remu_gpu::gpu_flips_pending() > backlog_limit {
            if !logged {
                static PRESENT_BACKPRESSURE_COUNT: AtomicU64 = AtomicU64::new(0);
                let sequence = PRESENT_BACKPRESSURE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                if sequence <= 16 || sequence.is_power_of_two() {
                    eprintln!(
                        "[REMu][VIDEO] presentation backpressure sequence={sequence} queued={} target={backlog_limit}",
                        remu_gpu::gpu_flips_pending(),
                    );
                }
                logged = true;
            }
            let pending = remu_gpu::gpu_flips_pending();
            if progress.stalled(pending, Instant::now(), MAX_BACKPRESSURE_STALL) {
                eprintln!(
                    "[REMu][VIDEO] presentation backpressure made no progress for {} ms; continuing queued={} target={backlog_limit}",
                    MAX_BACKPRESSURE_STALL.as_millis(),
                    pending,
                );
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

pub(crate) fn video_out_set_flip_rate(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if low_i32(arguments[0]) != 1 || !state.video_out.is_open {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    let rate = low_i32(arguments[1]);
    if !(0..=2).contains(&rate) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_VALUE);
    }
    VIDEO_OUT_FLIP_INTERVAL.store(flip_interval_microseconds(rate), Ordering::Relaxed);
    0
}

pub(crate) fn video_out_is_flip_pending(
    arguments: [u64; 6],
    state: &mut ProbeDispatchState,
) -> u64 {
    if low_i32(arguments[0]) != 1 || !state.video_out.is_open {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    u64::from(!state.pending_video_out_flips.is_empty())
}

pub(crate) fn video_out_submit_flip(arguments: [u64; 6], state: &mut ProbeDispatchState) -> u64 {
    if low_i32(arguments[0]) != 1 || !state.video_out.is_open {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_HANDLE);
    }
    let buffer_index = low_u32(arguments[1]);
    // UINT32_MAX requests the initial blank scanout before display buffers are registered.
    if buffer_index == u32::MAX {
        return 0;
    }
    let Ok(index) = usize::try_from(buffer_index) else {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_INDEX);
    };
    if index >= state.video_out.display_buffers.len() {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_INDEX);
    }
    // Standalone VideoOut clients (including splash and movie presentation) submit flips
    // directly instead of embedding a flip marker in an AGC command buffer. Publish the same
    // presenter event used by the ordered AGC path before completing the guest-visible flip.
    // Without it the guest advances and reuses the buffer while the host window never sees it.
    if !agc_submit_ordered_video_out_flip(&state.video_out, 1, buffer_index, arguments[3]) {
        return sce_kernel_error(VIDEO_OUT_ERROR_INVALID_INDEX);
    }
    queue_video_out_flip(state, 1, buffer_index, arguments[3]);
    0
}

#[cfg(test)]
mod tests {
    use super::BackpressureProgress;
    use std::time::{Duration, Instant};

    #[test]
    fn backpressure_stall_timer_resets_at_each_low_water_mark() {
        let start = Instant::now();
        let timeout = Duration::from_secs(5);
        let mut progress = BackpressureProgress::new(500, start);

        assert!(!progress.stalled(400, start + Duration::from_secs(4), timeout));
        assert!(!progress.stalled(350, start + Duration::from_secs(8), timeout));
        assert!(!progress.stalled(350, start + Duration::from_secs(12), timeout));
        assert!(progress.stalled(350, start + Duration::from_secs(13), timeout));
    }

    #[test]
    fn backpressure_queue_churn_does_not_count_as_progress() {
        let start = Instant::now();
        let timeout = Duration::from_secs(5);
        let mut progress = BackpressureProgress::new(100, start);

        assert!(!progress.stalled(120, start + Duration::from_secs(2), timeout));
        assert!(!progress.stalled(100, start + Duration::from_secs(4), timeout));
        assert!(progress.stalled(110, start + Duration::from_secs(5), timeout));
    }
}
