// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Non-blocking, env-gated debug log shared by every emulator subsystem.
//!
//! `stderr` diagnostics serialize writers behind one pipe lock, so a hot path cannot afford to
//! narrate itself there. This sink decouples authorship from I/O: callers format only when the
//! log is armed, hand the line to a bounded channel without waiting, and a single background
//! thread appends to the log file. When the channel is full the line is dropped and counted —
//! timing is never distorted by a slow disk, and the drop count itself is published so a gap in
//! the record reads as "overloaded", not "nothing happened".
//!
//! Armed with `REMU_DEBUG_LOG=1`. The destination is `REMU_DEBUG_LOG_FILE` when set, otherwise
//! `debug.log` beside the run's capture directory (`REMU_CAPTURE_DIR`), otherwise
//! `$TMPDIR/remu-debug.log`.

use std::io::Write as _;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::time::{Duration, Instant};

/// Set to `1` to arm the asynchronous debug log.
pub const DEBUG_LOG_VARIABLE: &str = "REMU_DEBUG_LOG";

/// Lines buffered between the authoring threads and the writer thread.
///
/// At ~120 bytes per line this bounds the queue near 16 MiB. The writer drains a saturated
/// channel in well under a second, so sustained drops indicate genuinely unloggable volume.
const CHANNEL_CAPACITY: usize = 128 * 1024;

/// Lines the writer batches between explicit flushes.
const FLUSH_EVERY_LINES: u64 = 512;

/// How long the writer waits for the next line before flushing a partial batch.
///
/// This replaces a `try_recv` drain probe that answered "is the queue empty?" by *consuming* a
/// line. The wait only runs with a batch already buffered; an idle writer parks in `recv` and
/// costs nothing.
const IDLE_FLUSH_WAIT: Duration = Duration::from_millis(20);

static ENABLED: AtomicU8 = AtomicU8::new(0);
const ENABLED_UNRESOLVED: u8 = 0;
const ENABLED_YES: u8 = 1;
const ENABLED_NO: u8 = 2;

static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Lines accepted by the channel over the process lifetime.
static SUBMITTED: AtomicU64 = AtomicU64::new(0);
/// Lines the writer has both written and flushed to the file, so a reader of the file would see
/// them. [`flush_async_debug_log`] waits for this to catch up with [`SUBMITTED`].
static FLUSHED: AtomicU64 = AtomicU64::new(0);

struct DebugLogState {
    sender: SyncSender<String>,
    start: Instant,
}

static STATE: OnceLock<Option<DebugLogState>> = OnceLock::new();

/// True when `REMU_DEBUG_LOG=1` armed the sink; callers gate their formatting on this.
#[inline]
pub fn debug_log_enabled() -> bool {
    match ENABLED.load(Ordering::Relaxed) {
        ENABLED_YES => true,
        ENABLED_NO => false,
        _ => {
            let armed = std::env::var_os(DEBUG_LOG_VARIABLE)
                .is_some_and(|value| value == "1" || value == "true")
                && state().is_some();
            ENABLED.store(
                if armed { ENABLED_YES } else { ENABLED_NO },
                Ordering::Relaxed,
            );
            armed
        }
    }
}

fn log_file_path() -> std::path::PathBuf {
    if let Some(path) = std::env::var_os("REMU_DEBUG_LOG_FILE") {
        return path.into();
    }
    if let Some(capture) = std::env::var_os("REMU_CAPTURE_DIR") {
        return std::path::Path::new(&capture).join("debug.log");
    }
    std::env::temp_dir().join("remu-debug.log")
}

fn state() -> Option<&'static DebugLogState> {
    STATE
        .get_or_init(|| {
            let path = log_file_path();
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let file = match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                Ok(file) => file,
                Err(error) => {
                    eprintln!("[REMu][DEBUG-LOG] cannot open {}: {error}", path.display());
                    return None;
                }
            };
            let (sender, receiver) = sync_channel::<String>(CHANNEL_CAPACITY);
            let spawned = std::thread::Builder::new()
                .name("remu-debug-log".into())
                .spawn(move || writer_thread(file, receiver));
            if let Err(error) = spawned {
                eprintln!("[REMu][DEBUG-LOG] cannot spawn writer thread: {error}");
                return None;
            }
            eprintln!("[REMu][DEBUG-LOG] armed; writing {}", path.display());
            Some(DebugLogState {
                sender,
                start: Instant::now(),
            })
        })
        .as_ref()
}

/// Drains the channel into `sink` until every sender is gone.
///
/// Every line this receives is written. The loop deliberately has no "peek" step: the previous
/// shape ended each iteration with `receiver.try_recv().is_err()` to ask whether the queue had
/// gone quiet, which *removed* a line from the channel whenever the answer was "no" and then
/// dropped it on the floor — silently, and without reaching the [`DROPPED`] counter that exists
/// to make gaps legible. Idleness is now detected by waiting for the next line instead of by
/// taking it.
fn writer_thread<W: std::io::Write>(sink: W, receiver: Receiver<String>) {
    let mut writer = std::io::BufWriter::with_capacity(1 << 20, sink);
    let mut since_flush = 0_u64;
    let mut written = 0_u64;
    let mut reported_drops = 0_u64;
    loop {
        // With nothing buffered there is no partial batch to time out, so park indefinitely
        // rather than waking on a timer the whole run.
        let received = if since_flush == 0 {
            receiver.recv().map_err(|_| RecvTimeoutError::Disconnected)
        } else {
            receiver.recv_timeout(IDLE_FLUSH_WAIT)
        };
        match received {
            Ok(line) => {
                let _ = writer.write_all(line.as_bytes());
                let _ = writer.write_all(b"\n");
                since_flush += 1;
                written += 1;
                if since_flush >= FLUSH_EVERY_LINES {
                    flush_batch(&mut writer, written, &mut reported_drops);
                    since_flush = 0;
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                flush_batch(&mut writer, written, &mut reported_drops);
                since_flush = 0;
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    flush_batch(&mut writer, written, &mut reported_drops);
}

/// Flushes the buffered batch and publishes how much of the record is now on disk.
///
/// Lines dropped for capacity since the last flush are noted in the file itself, so a gap in the
/// record reads as "overloaded" rather than "nothing happened".
fn flush_batch<W: std::io::Write>(
    writer: &mut std::io::BufWriter<W>,
    written: u64,
    reported_drops: &mut u64,
) {
    let dropped = DROPPED.load(Ordering::Relaxed);
    if dropped > *reported_drops {
        let _ = writeln!(writer, "!dropped {} lines", dropped - *reported_drops);
        *reported_drops = dropped;
    }
    let _ = writer.flush();
    FLUSHED.store(written, Ordering::Release);
}

/// Waits (bounded) for queued debug lines to reach the file, for use on exit paths.
///
/// Best-effort by contract, in two ways: it gives up on a writer that has stopped making
/// progress, and it gives up unconditionally at the deadline. Its callers are terminating, so a
/// sink that cannot drain must cost a fraction of a second and no more.
///
/// It takes no lock and allocates nothing — only atomic loads, `clock_gettime`, and `nanosleep`
/// — which is what makes it usable on the exit paths a guest signal handler can reach.
pub fn flush_async_debug_log() {
    /// Ceiling on the whole wait.
    const FLUSH_DEADLINE: std::time::Duration = Duration::from_millis(500);
    /// Consecutive idle steps before concluding the writer is not draining at all.
    const FLUSH_STALL_STEPS: u32 = 100;

    // `STATE.get()`, never `state()`: an exit path must not open the log file and spawn a writer
    // thread for a sink this run never armed.
    let Some(Some(_)) = STATE.get() else { return };
    let target = SUBMITTED.load(Ordering::Acquire);
    let deadline = Instant::now() + FLUSH_DEADLINE;
    let mut last_seen = FLUSHED.load(Ordering::Acquire);
    let mut stalled_steps = 0_u32;
    loop {
        let flushed = FLUSHED.load(Ordering::Acquire);
        if flushed >= target {
            return;
        }
        if flushed == last_seen {
            stalled_steps += 1;
            if stalled_steps >= FLUSH_STALL_STEPS {
                return;
            }
        } else {
            last_seen = flushed;
            stalled_steps = 0;
        }
        if Instant::now() >= deadline {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Queues one pre-formatted line; never blocks. Call only after [`debug_log_enabled`].
pub fn debug_log_line(module: &'static str, message: std::fmt::Arguments<'_>) {
    let Some(state) = state() else { return };
    let micros = state.start.elapsed().as_micros();
    let thread = std::thread::current();
    let line = format!(
        "{micros:>10} {:<12} {module}: {message}",
        thread.name().unwrap_or("?"),
    );
    match state.sender.try_send(line) {
        Ok(()) => {
            SUBMITTED.fetch_add(1, Ordering::Release);
        }
        Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
            DROPPED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Formats and queues one debug line iff `REMU_DEBUG_LOG=1`; the format arguments are not
/// evaluated when the log is disarmed.
#[macro_export]
macro_rules! debug_log {
    ($module:expr, $($arg:tt)*) => {
        if $crate::debug_log_enabled() {
            $crate::debug_log_line($module, format_args!($($arg)*));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::{CHANNEL_CAPACITY, FLUSH_EVERY_LINES, writer_thread};
    use std::sync::mpsc::sync_channel;
    use std::sync::{Arc, Mutex};

    /// A sink the test can still read after the writer has consumed it.
    #[derive(Clone, Default)]
    struct SharedSink(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for SharedSink {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("sink mutex").extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// The bug this pins: the writer used to end each iteration with `receiver.try_recv()` to
    /// test whether the queue had drained, which consumed and discarded a line every time the
    /// queue was *not* empty — roughly every other line once past the first batch, and never
    /// counted as a drop. Queueing more than one batch before the writer starts reproduces the
    /// backlog that triggered it.
    #[test]
    fn writer_records_every_queued_line_even_with_a_backlog() {
        const LINES: usize = FLUSH_EVERY_LINES as usize * 4;

        let (sender, receiver) = sync_channel::<String>(CHANNEL_CAPACITY);
        for index in 0..LINES {
            sender.send(format!("line {index}")).expect("queue line");
        }
        // Ends the writer once the backlog is drained.
        drop(sender);

        let sink = SharedSink::default();
        writer_thread(sink.clone(), receiver);

        let written = sink.0.lock().expect("sink mutex").clone();
        let text = String::from_utf8(written).expect("sink holds utf-8");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), LINES, "every queued line must reach the sink");
        for (index, line) in lines.iter().enumerate() {
            assert_eq!(*line, format!("line {index}"), "in order, none skipped");
        }
    }
}
