# Silent Hill present / UI / 3D — sampled tiler ownership (2026-08-06)

**Branch:** `sh-present-mode19-tiler` @ `1d5d4b3`  
**Base:** `main` @ `c69bf0c` (Phase 1 A–H closed)

## Root cause of `captured_frames=0`

Present flipped and snapshotted translated scanout (`0xefc0…` / `0xefc1…`), but PPM capture
skips fully-black readbacks. The VideoOut compositor draw

`es=0x800d550000` / `ps=0x8008020000` → `target=0xefc…`

was failing every frame with:

```text
Phase 1G: sampled GPU upload refuses tile=T format=F bpe=B — no tiler ownership
```

Phase 1G deleted CPU snapshot/`replaceRegion` for tiler-owned sampled uploads and only owned
uncompressed `tile ∈ {0,5,24,27}`. Silent Hill’s compositor samples:

| Step | Refusal that blocked the pair |
| --- | --- |
| pre-fix (1H tip) | `tile=1` `RGBA8Unorm` |
| after mode-1/9 uncompressed | `tile=5` `BC7_RGBAUnorm_sRGB` |
| after mode-5 BC | `tile=9` `BC7_RGBAUnorm_sRGB` |
| after mode-9 BC | `tile=1` `BC7_RGBAUnorm_sRGB` |

So the architectural hole was **GPU-tiler ownership for SW_256_S / SW_64KB_S / SW_4KB_S
sampled uploads, including the BC element grid** — not present/flip plumbing.

## What landed

| Commit | Concern |
| --- | --- |
| `b3c5b5b` | GPU-tiler MSL + encode for mode-1/9 uncompressed (`SW_256_S` / `SW_64KB_S`) |
| `0ef94d6` | Mode-5 BC element-grid ownership; plan detile in elements, blit Metal in pixels |
| `d8706d1` | Mode-9 BC ownership |
| `1d5d4b3` | Mode-1 BC ownership |

Compressed mip chains and mode-27 BC stay loud-refused.

## Live Silent Hill @ `1d5d4b3` (90s, clean tree)

Artifacts: `runs/silent-hill-mode1bc/`.

| Metric | 1H tip (`f8bdad5`) | This tip |
| --- | --- | --- |
| outcome | watchdog 124 | watchdog 124 |
| `captured_frames` | 0 | **0** (still fully black) |
| worst PERF-DROPPED | `800d55/800802` ~9k (tiler refuse) | `8007a8/800802` ~1.7k (no CB target) |
| tiler refuse on compositor | yes | **gone** |
| encoded compositor to VideoOut | no | **yes** (`encoded … es=0x800d550000 ps=0x8008040000` → `0xefc1…`) |
| pipeline failed | 0 | 0 |
| launch `vm.loadavg` | (overloaded earlier) | `{ 3.x … }` — no FPS claim |

**Honest UI/3D assessment:** tiler ownership no longer blocks the compositor path, and at least
one mesh compositor pair encodes into the flip targets. Present readback is still fully black, so
PPM UI/3D remains **unassessable**. This is **not** title-UI/3D closure (“success is not the
hallway”).

## Anti-claims

- Do not claim UI or 3D from flip count, watchdog timeout, or “no tiler refuse”.
- Do not claim FPS.
- Alias/sampled publish stays default-off (`REMU_PUBLISH_INTERNAL_WRITEBACKS=1` opt-in).
- Remaining `8007a8/800802` drops with `target=0x0` / `target_mask=0x0` are genuine no-attachment
  draws (pipeline key), not the efc compositor path.

## Follow-up diagnostic (main @ tip, `runs/silent-hill-scanout-diag/`)

With `REMU_TRACE_SCANOUT_WRITES=1` + `REMU_DIAGNOSTIC_SCANOUT_WRITERS=targets:0xefc0…,0xefc1…`:

| Observation | Evidence |
| --- | --- |
| Compositor writes VideoOut | **273** `SCANOUT-WRITE` for `es=0x800d55` / `ps=0x800802\|800804` |
| Residents advance | `content_version` → ~4760, `gpu_written=true` |
| Present snapshots translated | every flip `activation=translated` |
| Flip write coalescing | **all** `target_written=false` under TRACE (mid-frame snapshot sets `uploaded_flip_sequence` → `note_pending_flip_target_write` early-return); not the sole black cause |
| Bound samples | mode-9 **BC7** 3840×2160, mode-9 **R8** 1024², mode-1 **RGBA8** 1×1 |
| Alias publish | still default-off (`sampled-alias` / `alias-rebuild` RemuBookkeeping) |
| Outcome | still `captured_frames=0` — Metal readback of efc is fully black |

So the tiler-ownership gate is closed; the remaining gate is **compositor color** (sample
bytes / mesh+PS exports), not “no encode into VideoOut”.

## Next recommended blocker

**Why encoded VideoOut compositor output is still fully black**, in order:

1. Prove BC/R8 mode-1/5/9 GPU detile bytes against the CPU `blocks.rs` oracle on a live SH SRD
   (`REMU_CAPTURE_DIR` + `REMU_CAPTURE_TARGET_SAMPLES=1`, or GPU↔CPU linear buffer compare).
2. Trace `ps=0x8008020000` / `0x8008040000` after upload (SRD, resident `content_version`,
   nonzero sample fingerprints) — dump scanout MSL if needed.
3. Keep LDS+missing-EXP loud refuse (no INTERFACE-FILL / UV invent).
4. If SH present stays black with proven uploads, pivot: DeS `sceAcmBatchStartBuffers`, or
   Dreaming Sarah / Void null-SIGSEGV.

## Checks

- `cargo test -p remu-gpu-metal --lib presenter::tiler -- --test-threads=1` — green
- `cargo check --workspace --all-targets` — arm64 + `x86_64-apple-darwin` green
