// SPDX-License-Identifier: GPL-2.0-or-later
//! E3 criterion stub: PM4 decode throughput (per MiB).
//!
//! DEFERRED SEAM: add `[[bench]] name = "pm4_decode_per_mib"` + criterion to the owning crate's
//! Cargo.toml in a later solo pass (see SEAMS.md). Do not treat this file as a live bench.

compile_error!(
    "E3 stub: pm4_decode_per_mib is scaffolding under scripts/scaffolding/criterion-benches/; \
     wire criterion + [[bench]] in the owning crate Cargo.toml before compiling this as a bench"
);

fn main() {}
