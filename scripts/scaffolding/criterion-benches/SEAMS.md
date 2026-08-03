# Criterion micro-bench scaffolding (E3 / testing R12)

Status: **source stubs only under `scripts/scaffolding/`**. No crate `Cargo.toml` was edited in
Batch A6 — that is a deferred solo seam so parallel crate workers are not fought.

## Intended benches (plan E3)

| Bench id | Measures | Likely crate home |
| --- | --- | --- |
| `pm4_decode_per_mib` | PM4 / DCB decode throughput | `remu-runtime` or gpu PM4 decoder |
| `msl_translate_per_shader` | RDNA2 → MSL translation | `remu-shader-metal` |
| `write_watch_arm_fault_per_page` | watch arm + fault path | `remu-memory` |
| `tiled_texture_decode_per_mib` | tiled texture decode | `remu-gpu-metal` |

## Deferred Cargo.toml seam (later solo pass)

For each target crate, a future commit should add something shaped like:

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "pm4_decode_per_mib"
harness = false
```

…and move/copy the matching stub from this directory into `crates/<crate>/benches/`, replacing the
`unimplemented!` body with a real Criterion group. Until then, `cargo bench` will not see these
files — by design.

## Loud stub contract

Each `*.rs` stub compiles only as documentation of intent. They use `compile_error!` so they cannot
be silently compiled into a fake green bench if someone adds a `[[bench]]` without implementing the
body.
