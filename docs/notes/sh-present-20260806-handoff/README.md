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

## Live Silent Hill

### Without `REMU_CAPTURE_FRAMES` (matrix default) — `@1d5d4b3` / `@119f798`

Artifacts: `runs/silent-hill-mode1bc/`. `captured_frames=0` because **PPM capture is off**, not
because present never wrote. Compositor tiler refuses gone; mesh encodes to `0xefc…`.

### With `REMU_CAPTURE_FRAMES=1` — `@3f261a8` (60s, clean tree)

Artifacts: `runs/silent-hill-black-skip2/` (loadavg launch `{ 4.58 … }` — no FPS claim).

| Metric | Value |
| --- | --- |
| outcome | external-watchdog-timeout / 124 |
| `captured_frames` | **13** |
| early efc readback | `skip capture: fully black … nonzero=0` (first presents) |
| assessable splash | **HEXA DRIVE** logo; **KONAMI** on magenta |
| other PPMs | solid brand-color fades; at least one dense noise/garbage frame |
| tiler refuse on compositor | gone |

**Honest UI/3D assessment:** boot/splash 2D UI is assessable again after the tiler-ownership fix
when capture is enabled. This is **not** title-menu 3D / ruined-hallway closure (“success is not
the hallway”). Garbage frames after splash remain an open present/sample path.

## Anti-claims

- Do not treat matrix `captured_frames=0` as proof of black present when `REMU_CAPTURE_FRAMES`
  was unset (default).
- Do not claim title 3D or hallway from splash logos or flip count.
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

**Post-splash / title 3D (and garbage PPMs)**, in order:

1. Keep `REMU_CAPTURE_FRAMES=1` on SH UI evidence runs; correlate noise frames with sample
   uploads (`REMU_CAPTURE_TARGET_SAMPLES=1`) for mode-9 BC7/R8 compositor SRDs.
2. Trace `ps=0x8008020000` / `0x8008040000` after splash (title background path).
3. Keep LDS+missing-EXP loud refuse (no INTERFACE-FILL / UV invent).
4. Parallel high blockers if SH 3D stalls: DeS `sceAcmBatchStartBuffers`, Dreaming Sarah / Void
   null-SIGSEGV.

## Checks

- `cargo test -p remu-gpu-metal --lib presenter::tiler -- --test-threads=1` — green
- `cargo check --workspace --all-targets` — arm64 + `x86_64-apple-darwin` green
