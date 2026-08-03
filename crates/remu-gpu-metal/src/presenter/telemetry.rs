// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Structured per-window performance records (plan E4).
//!
//! One JSON object per declared measurement window is appended to `telemetry.jsonl` in the run
//! directory, and `run.json` describes the run once. Every number comes from the same sampler
//! the window title displays, so the two can never disagree. The JSON is hand-assembled from a
//! flat field list: values are numbers and pre-escaped identifiers only, which keeps the
//! workspace free of a serialization dependency for data this simple.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) const TELEMETRY_SCHEMA: &str = "remu-telemetry-v1";
pub(super) const RUN_SCHEMA: &str = "remu-run-v1";

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

#[cfg(test)]
mod tests {
    use super::{TelemetryValue, telemetry_record_json};

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
}
