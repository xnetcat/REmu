# E3 / B23 scaffolding — submission recording → `remu replay` → frame goldens

Status: **loud stubs only**. Full replay needs remu-cli + gpu-metal work outside Batch A6
(`scripts/` + `.github/` scope). Do not treat a green CI python job as proof that replay works.

## Intended pipeline (plan E3 / testing-observability L4)

1. **Recording** (`DeterminismMode::Recording`): each `agc_driver_submit_*` appends a record to
   `$run_dir/submissions.bin` — see `../submission-record/FORMAT.md`. Off by default; additive.
2. **Replay**: `remu replay <run-dir> [--frames N] [--dump-frames out/]` feeds the corpus through
   the real decode + translate + encode path against a synthetic guest memory image, with async
   compilation forced synchronous and `GPU_EVENTS_PER_TICK=1`. No guest CPU, timers, audio, or
   window.
3. **Goldens**: `remu replay --check <dir>` compares FNV fingerprints (existing readback path) and
   optional perceptual deltas; `--bless` regenerates. Corpus lives **outside** the repo, referenced
   by hash (`docs/source-policy.md`).

## Scripts-side entry points (this batch)

| Path | Role |
| --- | --- |
| `scripts/remu-replay.py` / `.sh` | Loud EX_CONFIG (78) stub with deferred-seam list |
| `scripts/scaffolding/submission-record/` | Format draft + Python encode/decode stub |
| `scripts/scaffolding/criterion-benches/` | Criterion bench *source* stubs; Cargo.toml seams deferred |

## Deferred crate seams (later solo pass — do not fight other workers)

- `crates/remu-cli`: add `Command::Replay { … }`; wire to a headless gpu path.
- `crates/remu-runtime` / `remu-gpu-metal`: `DeterminismMode::Recording`, submissions writer,
  synthetic memory image loader.
- Per-crate `[[bench]]` + `criterion` in `Cargo.toml` for PM4 / MSL / watch / tile (see
  `criterion-benches/SEAMS.md`) — **not** edited in this batch.

## CI hook

`.github/workflows/ci.yml` runs the stub self-test so the entry point cannot silently disappear.
Tier B golden `--check` stays commented until a corpus + real replay exist (plan H1 after E2).
