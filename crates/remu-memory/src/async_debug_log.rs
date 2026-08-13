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
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::time::Instant;

/// Set to `1` to arm the asynchronous debug log.
pub const DEBUG_LOG_VARIABLE: &str = "REMU_DEBUG_LOG";

/// Lines buffered between the authoring threads and the writer thread.
///
/// At ~120 bytes per line this bounds the queue near 16 MiB. The writer drains a saturated
/// channel in well under a second, so sustained drops indicate genuinely unloggable volume.
const CHANNEL_CAPACITY: usize = 128 * 1024;

/// Lines the writer batches between explicit flushes.
const FLUSH_EVERY_LINES: u64 = 512;

static ENABLED: AtomicU8 = AtomicU8::new(0);
const ENABLED_UNRESOLVED: u8 = 0;
const ENABLED_YES: u8 = 1;
const ENABLED_NO: u8 = 2;

static DROPPED: AtomicU64 = AtomicU64::new(0);

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

fn writer_thread(file: std::fs::File, receiver: Receiver<String>) {
    let mut writer = std::io::BufWriter::with_capacity(1 << 20, file);
    let mut since_flush = 0_u64;
    let mut reported_drops = 0_u64;
    while let Ok(line) = receiver.recv() {
        let _ = writer.write_all(line.as_bytes());
        let _ = writer.write_all(b"\n");
        since_flush += 1;
        if since_flush >= FLUSH_EVERY_LINES || receiver.try_recv().is_err() {
            // Also surface any lines dropped since the last flush so the file itself
            // records its own gaps.
            let dropped = DROPPED.load(Ordering::Relaxed);
            if dropped > reported_drops {
                let _ = writeln!(writer, "!dropped {} lines", dropped - reported_drops);
                reported_drops = dropped;
            }
            let _ = writer.flush();
            since_flush = 0;
        }
    }
    let _ = writer.flush();
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
        Ok(()) => {}
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
