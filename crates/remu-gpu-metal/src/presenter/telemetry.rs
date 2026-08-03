// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Structured per-window performance records (plan E4 + K3).
//!
//! One JSON object per declared measurement window is appended to `telemetry.jsonl` in the run
//! directory, and `run.json` describes the run once. Every number comes from the same sampler
//! the window title and `[REMu][PERF*]` lines display, so the two can never disagree. The JSON
//! is hand-assembled from a flat field list: values are numbers and pre-escaped identifiers
//! only, which keeps the workspace free of a serialization dependency for data this simple.
//!
//! Plan K3 folds the nineteen audit PERF channels (previously eprintln!-only) into the same
//! record. The title `PERF` rates stay in the E4 core fields; the other eighteen channels land
//! as the `PerfChannelSample` fields below. Free-form diagnostic strings (queue dumps, top-N
//! lists) are stored as escaped text so `jq` can read them without grepping logs.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) const TELEMETRY_SCHEMA: &str = "remu-telemetry-v1";
pub(super) const RUN_SCHEMA: &str = "remu-run-v1";

/// The nineteen `[REMu][PERF*]` channels the testing/observability audit counted as parsed by
/// nothing (`architecture-audit-20260802/testing-observability.md`). Title `PERF` is covered by
/// the E4 core window fields; K3 emits the rest into the same JSON line.
pub(super) const AUDIT_PERF_CHANNELS: &[&str] = &[
    "PERF",
    "PERF-CAUSES",
    "PERF-RP-KEY",
    "PERF-PIPELINE",
    "PERF-QUEUES",
    "PERF-QUEUE-WAITS",
    "PERF-BATCHES",
    "PERF-DROPPED",
    "PERF-TEXTURE",
    "PERF-TEXTURE-CACHE",
    "PERF-METAL-RESIDENCY",
    "PERF-METAL-TARGETS",
    "PERF-METAL-TARGET-ACTIVITY",
    "PERF-METAL-TARGET-OVERLAP",
    "PERF-METAL-TARGET-IDLE",
    "PERF-GLOBAL-CACHE",
    "PERF-SNAPSHOT-TOP",
    "PERF-TEXTURE-DIRTY-TOP",
    "PERF-TEXTURE-TOP",
];

/// One marker field name per non-title audit channel. Present in every K3 record so a jq query
/// can prove coverage without scraping eprintln! text.
pub(super) const K3_CHANNEL_MARKER_FIELDS: &[(&str, &str)] = &[
    ("PERF-CAUSES", "rp_end_compute_per_s"),
    ("PERF-RP-KEY", "rp_key_target_addr_per_s"),
    ("PERF-PIPELINE", "pipeline_ready_graphics"),
    ("PERF-QUEUES", "queues"),
    ("PERF-QUEUE-WAITS", "queue_waits"),
    ("PERF-BATCHES", "batches"),
    ("PERF-DROPPED", "dropped_draws"),
    ("PERF-TEXTURE", "texture_snapshot_count"),
    ("PERF-TEXTURE-CACHE", "texcache_hits"),
    ("PERF-METAL-RESIDENCY", "metal_res_sampled_count"),
    ("PERF-METAL-TARGETS", "metal_tgt_gpu_owned_count"),
    ("PERF-METAL-TARGET-ACTIVITY", "metal_tgt_activity"),
    ("PERF-METAL-TARGET-OVERLAP", "metal_tgt_overlap_pairs"),
    ("PERF-METAL-TARGET-IDLE", "metal_tgt_idle"),
    ("PERF-GLOBAL-CACHE", "global_cache_captured_count"),
    ("PERF-SNAPSHOT-TOP", "snapshot_top"),
    ("PERF-TEXTURE-DIRTY-TOP", "texture_dirty_top"),
    ("PERF-TEXTURE-TOP", "texture_top"),
];

/// A field value in a telemetry record.
pub(super) enum TelemetryValue {
    Unsigned(u64),
    Float(f64),
    Text(String),
}

impl From<u64> for TelemetryValue {
    fn from(value: u64) -> Self {
        Self::Unsigned(value)
    }
}

impl From<f64> for TelemetryValue {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<&str> for TelemetryValue {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<String> for TelemetryValue {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

/// Serializes one flat record as a single JSON line.
///
/// Field names are compile-time identifiers and never need escaping; text values are escaped.
/// Non-finite floats serialize as `0` — a JSON document must stay parseable even when a
/// division produced a NaN in a zero-length window.
pub(super) fn telemetry_record_json(fields: &[(&str, TelemetryValue)]) -> String {
    let mut line = String::with_capacity(fields.len() * 24);
    line.push('{');
    for (index, (name, value)) in fields.iter().enumerate() {
        if index > 0 {
            line.push(',');
        }
        line.push('"');
        line.push_str(name);
        line.push_str("\":");
        match value {
            TelemetryValue::Unsigned(value) => line.push_str(&value.to_string()),
            TelemetryValue::Float(value) => {
                if value.is_finite() {
                    line.push_str(&format!("{value:.3}"));
                } else {
                    line.push('0');
                }
            }
            TelemetryValue::Text(value) => {
                line.push('"');
                for character in value.chars() {
                    match character {
                        '"' => line.push_str("\\\""),
                        '\\' => line.push_str("\\\\"),
                        control if (control as u32) < 0x20 => {
                            line.push_str(&format!("\\u{:04x}", control as u32));
                        }
                        other => line.push(other),
                    }
                }
                line.push('"');
            }
        }
    }
    line.push('}');
    line
}

pub(super) fn unix_time_milliseconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// The run directory `run-game.sh` exports; absent for bare launches, which keeps every write
/// here optional.
fn telemetry_run_directory() -> Option<PathBuf> {
    let directory = PathBuf::from(std::env::var_os("REMU_RUN_DIR")?);
    directory.is_dir().then_some(directory)
}

/// The open per-window record stream, plus the sequence number stamped into each record.
pub(super) struct TelemetryLog {
    file: File,
    sequence: u64,
}

impl TelemetryLog {
    /// Opens the stream in the exported run directory and writes `run.json` beside it.
    pub(super) fn open_in_run_directory(device_name: &str) -> Option<Self> {
        let directory = telemetry_run_directory()?;
        if let Err(error) = write_run_json(&directory, device_name) {
            eprintln!("[REMu][TELEMETRY] could not write run.json: {error}");
        }
        let path = directory.join("telemetry.jsonl");
        match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(file) => {
                eprintln!("[REMu][TELEMETRY] recording windows to {}", path.display());
                Some(Self { file, sequence: 0 })
            }
            Err(error) => {
                eprintln!(
                    "[REMu][TELEMETRY] could not open {}: {error}",
                    path.display()
                );
                None
            }
        }
    }

    /// Appends one record, stamping schema, sequence and wall clock.
    pub(super) fn record(&mut self, fields: Vec<(&str, TelemetryValue)>) {
        let mut stamped = vec![
            ("schema", TelemetryValue::from(TELEMETRY_SCHEMA)),
            ("seq", TelemetryValue::from(self.sequence)),
            ("unix_ms", TelemetryValue::from(unix_time_milliseconds())),
        ];
        stamped.extend(fields);
        let line = telemetry_record_json(&stamped);
        self.sequence = self.sequence.saturating_add(1);
        if let Err(error) = writeln!(self.file, "{line}") {
            eprintln!("[REMu][TELEMETRY] could not append record: {error}");
        }
    }
}

fn write_run_json(directory: &Path, device_name: &str) -> std::io::Result<()> {
    let revision = std::env::var("REMU_RUN_REVISION").unwrap_or_else(|_| "unknown".to_owned());
    let executable = std::env::current_exe()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|_| "unknown".to_owned());
    let record = telemetry_record_json(&[
        ("schema", TelemetryValue::from(RUN_SCHEMA)),
        ("pid", TelemetryValue::from(u64::from(std::process::id()))),
        (
            "start_unix_ms",
            TelemetryValue::from(unix_time_milliseconds()),
        ),
        ("revision", TelemetryValue::from(revision.as_str())),
        ("executable", TelemetryValue::from(executable.as_str())),
        ("device", TelemetryValue::from(device_name)),
    ]);
    std::fs::write(directory.join("run.json"), format!("{record}\n"))
}

/// Numeric and text payloads for the eighteen non-title audit PERF channels (plan K3).
///
/// Built from the same locals `update_window_title` prints, so a field here and its eprintln!
/// counterpart cannot disagree. Empty strings mean the channel had nothing to report that
/// window (queues idle, top-N empty, lifetime diagnostics off).
#[derive(Clone, Debug, Default)]
pub(super) struct PerfChannelSample {
    // PERF-CAUSES
    pub rp_end_compute_per_s: f64,
    pub rp_end_key_per_s: f64,
    pub rp_end_color_op_per_s: f64,
    pub rp_end_flush_per_s: f64,
    pub cb_flush_tick_per_s: f64,
    pub cb_flush_memory_per_s: f64,
    pub cb_flush_wait_per_s: f64,
    pub cb_flush_shared_host_per_s: f64,
    pub cb_flush_flip_per_s: f64,
    pub cb_flush_capacity_per_s: f64,
    pub cb_flush_pfp_per_s: f64,
    pub cb_flush_sync_per_s: f64,

    // PERF-RP-KEY (headline addr rate + presence; full field dump stays in the log line)
    pub rp_key_target_count_per_s: f64,
    pub rp_key_target_addr_per_s: f64,
    pub rp_key_depth_presence_per_s: f64,
    pub rp_key_depth_addr_per_s: f64,

    // PERF-PIPELINE
    pub pipeline_ready_graphics: u64,
    pub pipeline_ready_compute: u64,
    pub pipeline_active_graphics: u64,
    pub pipeline_active_compute: u64,
    pub pipeline_failed_graphics: u64,
    pub pipeline_failed_compute: u64,
    pub pipeline_enqueued: u64,
    pub pipeline_queue_full: u64,
    pub pipeline_completed: u64,
    pub pipeline_failed_interval: u64,
    pub pipeline_draw_gates: u64,
    pub pipeline_compute_gates: u64,
    pub pipeline_blocks: u64,
    pub pipeline_memory_blocks: u64,
    pub pipeline_flip_blocks: u64,

    // PERF-QUEUES / PERF-QUEUE-WAITS / PERF-BATCHES (free-form dumps)
    pub queues: String,
    pub queue_waits: String,
    pub batches: String,
    pub batch_diag_count: u64,

    // PERF-DROPPED
    pub dropped_draws: u64,
    pub dropped_draw_keys: u64,
    pub dropped_compute: u64,
    pub dropped_compute_keys: u64,

    // PERF-TEXTURE
    pub texture_snapshot_count: u64,
    pub texture_snapshot_mib: f64,
    pub texture_snapshot_ms: f64,
    pub texture_snapshot_unstable: u64,
    pub texture_dirty_count: u64,
    pub texture_dirty_mib: f64,
    pub texture_dirty_ms: f64,
    pub texture_dirty_clean: u64,
    pub texture_mode27_count: u64,
    pub texture_mode27_mib: f64,
    pub texture_mode27_ms: f64,
    pub texture_upload_count: u64,
    pub texture_upload_mib: f64,
    pub texture_upload_ms: f64,

    // PERF-TEXTURE-CACHE
    pub texcache_resident_count: u64,
    pub texcache_resident_mib: u64,
    pub texcache_guest_mib: u64,
    pub texcache_metal_allocated_mib: u64,
    pub texcache_metal_recommended_mib: u64,
    pub texcache_watches: u64,
    pub texcache_hits: u64,
    pub texcache_misses: u64,
    pub texcache_dirty_reloads: u64,
    pub texcache_acquire_invalidate: u64,
    pub texcache_resident_acquire_invalidate: u64,
    pub texcache_capacity_evict: u64,
    pub texcache_depth_native: u64,
    pub texcache_alias_writeback_count: u64,
    pub texcache_alias_writeback_mib: f64,
    pub texcache_alias_writeback_ms: f64,
    pub texcache_write_attribution_miss: u64,

    // PERF-METAL-RESIDENCY
    pub metal_res_sampled_count: u64,
    pub metal_res_sampled_mib: f64,
    pub metal_res_targets_count: u64,
    pub metal_res_targets_mib: f64,
    pub metal_res_depth_arrays_count: u64,
    pub metal_res_depth_arrays_mib: f64,
    pub metal_res_direct_scanout_count: u64,
    pub metal_res_direct_scanout_mib: f64,
    pub metal_res_scanout_snapshots_count: u64,
    pub metal_res_scanout_snapshots_mib: f64,
    pub metal_res_msaa_count: u64,
    pub metal_res_msaa_mib: f64,
    pub metal_res_fallback_count: u64,
    pub metal_res_fallback_mib: f64,
    pub metal_res_accounted_mib: f64,
    pub metal_res_other_mib: f64,

    // PERF-METAL-TARGETS
    pub metal_tgt_gpu_owned_count: u64,
    pub metal_tgt_gpu_owned_mib: f64,
    pub metal_tgt_cpu_owned_count: u64,
    pub metal_tgt_cpu_owned_mib: f64,
    pub metal_tgt_active_mapping_count: u64,
    pub metal_tgt_active_mapping_mib: f64,
    pub metal_tgt_unresolved_count: u64,
    pub metal_tgt_unresolved_mib: f64,
    pub metal_tgt_top: String,

    // PERF-METAL-TARGET-ACTIVITY / OVERLAP / IDLE
    pub metal_tgt_activity: String,
    pub metal_tgt_overlap_pairs: u64,
    pub metal_tgt_overlap_pair_mib: f64,
    pub metal_tgt_overlap_residents: u64,
    pub metal_tgt_overlap_resident_mib: f64,
    pub metal_tgt_idle: String,

    // PERF-GLOBAL-CACHE
    pub global_cache_captured_count: u64,
    pub global_cache_captured_mib: f64,
    pub global_cache_identity_count: u64,
    pub global_cache_identity_mib: f64,

    // PERF-*-TOP
    pub snapshot_top: String,
    pub texture_dirty_top: String,
    pub texture_top: String,
}

impl PerfChannelSample {
    /// Flat field list for the K3 channels. Marker fields from [`K3_CHANNEL_MARKER_FIELDS`]
    /// are always present (zeros or empty strings when the window had nothing to report).
    pub(super) fn into_fields(self) -> Vec<(&'static str, TelemetryValue)> {
        debug_assert_eq!(AUDIT_PERF_CHANNELS.len(), 19);
        debug_assert_eq!(K3_CHANNEL_MARKER_FIELDS.len(), 18);
        vec![
            ("perf_channel_count", TelemetryValue::from(19_u64)),
            (
                "perf_channels",
                TelemetryValue::from(AUDIT_PERF_CHANNELS.join(",")),
            ),
            // PERF-CAUSES
            (
                "rp_end_compute_per_s",
                TelemetryValue::from(self.rp_end_compute_per_s),
            ),
            (
                "rp_end_key_per_s",
                TelemetryValue::from(self.rp_end_key_per_s),
            ),
            (
                "rp_end_color_op_per_s",
                TelemetryValue::from(self.rp_end_color_op_per_s),
            ),
            (
                "rp_end_flush_per_s",
                TelemetryValue::from(self.rp_end_flush_per_s),
            ),
            (
                "cb_flush_tick_per_s",
                TelemetryValue::from(self.cb_flush_tick_per_s),
            ),
            (
                "cb_flush_memory_per_s",
                TelemetryValue::from(self.cb_flush_memory_per_s),
            ),
            (
                "cb_flush_wait_per_s",
                TelemetryValue::from(self.cb_flush_wait_per_s),
            ),
            (
                "cb_flush_shared_host_per_s",
                TelemetryValue::from(self.cb_flush_shared_host_per_s),
            ),
            (
                "cb_flush_flip_per_s",
                TelemetryValue::from(self.cb_flush_flip_per_s),
            ),
            (
                "cb_flush_capacity_per_s",
                TelemetryValue::from(self.cb_flush_capacity_per_s),
            ),
            (
                "cb_flush_pfp_per_s",
                TelemetryValue::from(self.cb_flush_pfp_per_s),
            ),
            (
                "cb_flush_sync_per_s",
                TelemetryValue::from(self.cb_flush_sync_per_s),
            ),
            // PERF-RP-KEY
            (
                "rp_key_target_count_per_s",
                TelemetryValue::from(self.rp_key_target_count_per_s),
            ),
            (
                "rp_key_target_addr_per_s",
                TelemetryValue::from(self.rp_key_target_addr_per_s),
            ),
            (
                "rp_key_depth_presence_per_s",
                TelemetryValue::from(self.rp_key_depth_presence_per_s),
            ),
            (
                "rp_key_depth_addr_per_s",
                TelemetryValue::from(self.rp_key_depth_addr_per_s),
            ),
            // PERF-PIPELINE
            (
                "pipeline_ready_graphics",
                TelemetryValue::from(self.pipeline_ready_graphics),
            ),
            (
                "pipeline_ready_compute",
                TelemetryValue::from(self.pipeline_ready_compute),
            ),
            (
                "pipeline_active_graphics",
                TelemetryValue::from(self.pipeline_active_graphics),
            ),
            (
                "pipeline_active_compute",
                TelemetryValue::from(self.pipeline_active_compute),
            ),
            (
                "pipeline_failed_graphics",
                TelemetryValue::from(self.pipeline_failed_graphics),
            ),
            (
                "pipeline_failed_compute",
                TelemetryValue::from(self.pipeline_failed_compute),
            ),
            (
                "pipeline_enqueued",
                TelemetryValue::from(self.pipeline_enqueued),
            ),
            (
                "pipeline_queue_full",
                TelemetryValue::from(self.pipeline_queue_full),
            ),
            (
                "pipeline_completed",
                TelemetryValue::from(self.pipeline_completed),
            ),
            (
                "pipeline_failed_interval",
                TelemetryValue::from(self.pipeline_failed_interval),
            ),
            (
                "pipeline_draw_gates",
                TelemetryValue::from(self.pipeline_draw_gates),
            ),
            (
                "pipeline_compute_gates",
                TelemetryValue::from(self.pipeline_compute_gates),
            ),
            ("pipeline_blocks", TelemetryValue::from(self.pipeline_blocks)),
            (
                "pipeline_memory_blocks",
                TelemetryValue::from(self.pipeline_memory_blocks),
            ),
            (
                "pipeline_flip_blocks",
                TelemetryValue::from(self.pipeline_flip_blocks),
            ),
            // PERF-QUEUES / QUEUE-WAITS / BATCHES
            ("queues", TelemetryValue::from(self.queues)),
            ("queue_waits", TelemetryValue::from(self.queue_waits)),
            ("batches", TelemetryValue::from(self.batches)),
            (
                "batch_diag_count",
                TelemetryValue::from(self.batch_diag_count),
            ),
            // PERF-DROPPED
            ("dropped_draws", TelemetryValue::from(self.dropped_draws)),
            (
                "dropped_draw_keys",
                TelemetryValue::from(self.dropped_draw_keys),
            ),
            ("dropped_compute", TelemetryValue::from(self.dropped_compute)),
            (
                "dropped_compute_keys",
                TelemetryValue::from(self.dropped_compute_keys),
            ),
            // PERF-TEXTURE
            (
                "texture_snapshot_count",
                TelemetryValue::from(self.texture_snapshot_count),
            ),
            (
                "texture_snapshot_mib",
                TelemetryValue::from(self.texture_snapshot_mib),
            ),
            (
                "texture_snapshot_ms",
                TelemetryValue::from(self.texture_snapshot_ms),
            ),
            (
                "texture_snapshot_unstable",
                TelemetryValue::from(self.texture_snapshot_unstable),
            ),
            (
                "texture_dirty_count",
                TelemetryValue::from(self.texture_dirty_count),
            ),
            (
                "texture_dirty_mib",
                TelemetryValue::from(self.texture_dirty_mib),
            ),
            (
                "texture_dirty_ms",
                TelemetryValue::from(self.texture_dirty_ms),
            ),
            (
                "texture_dirty_clean",
                TelemetryValue::from(self.texture_dirty_clean),
            ),
            (
                "texture_mode27_count",
                TelemetryValue::from(self.texture_mode27_count),
            ),
            (
                "texture_mode27_mib",
                TelemetryValue::from(self.texture_mode27_mib),
            ),
            (
                "texture_mode27_ms",
                TelemetryValue::from(self.texture_mode27_ms),
            ),
            (
                "texture_upload_count",
                TelemetryValue::from(self.texture_upload_count),
            ),
            (
                "texture_upload_mib",
                TelemetryValue::from(self.texture_upload_mib),
            ),
            (
                "texture_upload_ms",
                TelemetryValue::from(self.texture_upload_ms),
            ),
            // PERF-TEXTURE-CACHE
            (
                "texcache_resident_count",
                TelemetryValue::from(self.texcache_resident_count),
            ),
            (
                "texcache_resident_mib",
                TelemetryValue::from(self.texcache_resident_mib),
            ),
            (
                "texcache_guest_mib",
                TelemetryValue::from(self.texcache_guest_mib),
            ),
            (
                "texcache_metal_allocated_mib",
                TelemetryValue::from(self.texcache_metal_allocated_mib),
            ),
            (
                "texcache_metal_recommended_mib",
                TelemetryValue::from(self.texcache_metal_recommended_mib),
            ),
            (
                "texcache_watches",
                TelemetryValue::from(self.texcache_watches),
            ),
            ("texcache_hits", TelemetryValue::from(self.texcache_hits)),
            ("texcache_misses", TelemetryValue::from(self.texcache_misses)),
            (
                "texcache_dirty_reloads",
                TelemetryValue::from(self.texcache_dirty_reloads),
            ),
            (
                "texcache_acquire_invalidate",
                TelemetryValue::from(self.texcache_acquire_invalidate),
            ),
            (
                "texcache_resident_acquire_invalidate",
                TelemetryValue::from(self.texcache_resident_acquire_invalidate),
            ),
            (
                "texcache_capacity_evict",
                TelemetryValue::from(self.texcache_capacity_evict),
            ),
            (
                "texcache_depth_native",
                TelemetryValue::from(self.texcache_depth_native),
            ),
            (
                "texcache_alias_writeback_count",
                TelemetryValue::from(self.texcache_alias_writeback_count),
            ),
            (
                "texcache_alias_writeback_mib",
                TelemetryValue::from(self.texcache_alias_writeback_mib),
            ),
            (
                "texcache_alias_writeback_ms",
                TelemetryValue::from(self.texcache_alias_writeback_ms),
            ),
            (
                "texcache_write_attribution_miss",
                TelemetryValue::from(self.texcache_write_attribution_miss),
            ),
            // PERF-METAL-RESIDENCY
            (
                "metal_res_sampled_count",
                TelemetryValue::from(self.metal_res_sampled_count),
            ),
            (
                "metal_res_sampled_mib",
                TelemetryValue::from(self.metal_res_sampled_mib),
            ),
            (
                "metal_res_targets_count",
                TelemetryValue::from(self.metal_res_targets_count),
            ),
            (
                "metal_res_targets_mib",
                TelemetryValue::from(self.metal_res_targets_mib),
            ),
            (
                "metal_res_depth_arrays_count",
                TelemetryValue::from(self.metal_res_depth_arrays_count),
            ),
            (
                "metal_res_depth_arrays_mib",
                TelemetryValue::from(self.metal_res_depth_arrays_mib),
            ),
            (
                "metal_res_direct_scanout_count",
                TelemetryValue::from(self.metal_res_direct_scanout_count),
            ),
            (
                "metal_res_direct_scanout_mib",
                TelemetryValue::from(self.metal_res_direct_scanout_mib),
            ),
            (
                "metal_res_scanout_snapshots_count",
                TelemetryValue::from(self.metal_res_scanout_snapshots_count),
            ),
            (
                "metal_res_scanout_snapshots_mib",
                TelemetryValue::from(self.metal_res_scanout_snapshots_mib),
            ),
            (
                "metal_res_msaa_count",
                TelemetryValue::from(self.metal_res_msaa_count),
            ),
            (
                "metal_res_msaa_mib",
                TelemetryValue::from(self.metal_res_msaa_mib),
            ),
            (
                "metal_res_fallback_count",
                TelemetryValue::from(self.metal_res_fallback_count),
            ),
            (
                "metal_res_fallback_mib",
                TelemetryValue::from(self.metal_res_fallback_mib),
            ),
            (
                "metal_res_accounted_mib",
                TelemetryValue::from(self.metal_res_accounted_mib),
            ),
            (
                "metal_res_other_mib",
                TelemetryValue::from(self.metal_res_other_mib),
            ),
            // PERF-METAL-TARGETS
            (
                "metal_tgt_gpu_owned_count",
                TelemetryValue::from(self.metal_tgt_gpu_owned_count),
            ),
            (
                "metal_tgt_gpu_owned_mib",
                TelemetryValue::from(self.metal_tgt_gpu_owned_mib),
            ),
            (
                "metal_tgt_cpu_owned_count",
                TelemetryValue::from(self.metal_tgt_cpu_owned_count),
            ),
            (
                "metal_tgt_cpu_owned_mib",
                TelemetryValue::from(self.metal_tgt_cpu_owned_mib),
            ),
            (
                "metal_tgt_active_mapping_count",
                TelemetryValue::from(self.metal_tgt_active_mapping_count),
            ),
            (
                "metal_tgt_active_mapping_mib",
                TelemetryValue::from(self.metal_tgt_active_mapping_mib),
            ),
            (
                "metal_tgt_unresolved_count",
                TelemetryValue::from(self.metal_tgt_unresolved_count),
            ),
            (
                "metal_tgt_unresolved_mib",
                TelemetryValue::from(self.metal_tgt_unresolved_mib),
            ),
            ("metal_tgt_top", TelemetryValue::from(self.metal_tgt_top)),
            // PERF-METAL-TARGET-ACTIVITY / OVERLAP / IDLE
            (
                "metal_tgt_activity",
                TelemetryValue::from(self.metal_tgt_activity),
            ),
            (
                "metal_tgt_overlap_pairs",
                TelemetryValue::from(self.metal_tgt_overlap_pairs),
            ),
            (
                "metal_tgt_overlap_pair_mib",
                TelemetryValue::from(self.metal_tgt_overlap_pair_mib),
            ),
            (
                "metal_tgt_overlap_residents",
                TelemetryValue::from(self.metal_tgt_overlap_residents),
            ),
            (
                "metal_tgt_overlap_resident_mib",
                TelemetryValue::from(self.metal_tgt_overlap_resident_mib),
            ),
            ("metal_tgt_idle", TelemetryValue::from(self.metal_tgt_idle)),
            // PERF-GLOBAL-CACHE
            (
                "global_cache_captured_count",
                TelemetryValue::from(self.global_cache_captured_count),
            ),
            (
                "global_cache_captured_mib",
                TelemetryValue::from(self.global_cache_captured_mib),
            ),
            (
                "global_cache_identity_count",
                TelemetryValue::from(self.global_cache_identity_count),
            ),
            (
                "global_cache_identity_mib",
                TelemetryValue::from(self.global_cache_identity_mib),
            ),
            // PERF-*-TOP
            ("snapshot_top", TelemetryValue::from(self.snapshot_top)),
            (
                "texture_dirty_top",
                TelemetryValue::from(self.texture_dirty_top),
            ),
            ("texture_top", TelemetryValue::from(self.texture_top)),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AUDIT_PERF_CHANNELS, K3_CHANNEL_MARKER_FIELDS, PerfChannelSample, TelemetryValue,
        telemetry_record_json,
    };
    use std::collections::HashSet;

    #[test]
    fn a_record_serializes_every_value_kind_on_one_line() {
        let line = telemetry_record_json(&[
            ("seq", TelemetryValue::from(7_u64)),
            ("fps", TelemetryValue::from(59.943_f64)),
            ("device", TelemetryValue::from("Apple M2 \"Pro\"")),
        ]);
        assert_eq!(
            line,
            "{\"seq\":7,\"fps\":59.943,\"device\":\"Apple M2 \\\"Pro\\\"\"}"
        );
        assert!(!line.contains('\n'));
    }

    #[test]
    fn a_non_finite_float_stays_parseable_json() {
        let line = telemetry_record_json(&[("mean", TelemetryValue::from(f64::NAN))]);
        assert_eq!(line, "{\"mean\":0}");
    }

    #[test]
    fn k3_audit_lists_exactly_nineteen_perf_channels() {
        assert_eq!(AUDIT_PERF_CHANNELS.len(), 19);
        assert_eq!(K3_CHANNEL_MARKER_FIELDS.len(), 18);
        let mut seen = HashSet::new();
        for channel in AUDIT_PERF_CHANNELS {
            assert!(seen.insert(*channel), "duplicate channel {channel}");
        }
    }

    #[test]
    fn k3_sample_emits_a_marker_field_for_every_non_title_channel() {
        let fields = PerfChannelSample {
            rp_end_compute_per_s: 4.5,
            rp_key_target_addr_per_s: 9.0,
            pipeline_ready_graphics: 30,
            queues: "q0:busy".into(),
            queue_waits: "wait:flip".into(),
            batches: "0:Completed".into(),
            dropped_draws: 2,
            texture_snapshot_count: 62,
            texcache_hits: 100,
            metal_res_sampled_count: 44,
            metal_tgt_gpu_owned_count: 12,
            metal_tgt_activity: "work:7".into(),
            metal_tgt_overlap_pairs: 1,
            metal_tgt_idle: "oldest".into(),
            global_cache_captured_count: 3,
            snapshot_top: "0x1".into(),
            texture_dirty_top: "0x2".into(),
            texture_top: "0x3".into(),
            ..PerfChannelSample::default()
        }
        .into_fields();
        let names: HashSet<&str> = fields.iter().map(|(name, _)| *name).collect();
        for (channel, marker) in K3_CHANNEL_MARKER_FIELDS {
            assert!(
                names.contains(marker),
                "K3 marker for {channel} missing: {marker}"
            );
        }
        let count = fields
            .iter()
            .find_map(|(name, value)| {
                (*name == "perf_channel_count").then_some(match value {
                    TelemetryValue::Unsigned(value) => *value,
                    _ => 0,
                })
            })
            .expect("perf_channel_count");
        assert_eq!(count, 19);
        let line = telemetry_record_json(&fields);
        assert!(line.contains("\"rp_end_compute_per_s\":4.500"));
        assert!(line.contains("\"texture_snapshot_count\":62"));
        assert!(line.contains("\"queues\":\"q0:busy\""));
        assert!(line.contains("\"perf_channels\":\"PERF,PERF-CAUSES,"));
        assert!(!line.contains('\n'));
        // Every marker field is named in K3_CHANNEL_MARKER_FIELDS for the non-title channels.
        for (channel, marker) in K3_CHANNEL_MARKER_FIELDS {
            assert!(
                AUDIT_PERF_CHANNELS.contains(channel),
                "{channel} missing from AUDIT_PERF_CHANNELS"
            );
            let _ = marker;
        }
    }

    #[test]
    fn k3_default_sample_still_emits_every_marker_as_zero_or_empty() {
        let fields = PerfChannelSample::default().into_fields();
        let names: HashSet<&str> = fields.iter().map(|(name, _)| *name).collect();
        for (channel, marker) in K3_CHANNEL_MARKER_FIELDS {
            assert!(
                names.contains(marker),
                "default sample must still cover {channel} via {marker}"
            );
        }
    }
}
