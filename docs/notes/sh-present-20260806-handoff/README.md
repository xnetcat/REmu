# Silent Hill present / UI / 3D — tiler + mesh LDS + mode-5 BC mips + stencil latch (2026-08-06)

**Tip:** `main` @ `0a22e15` (docs tip may be ahead).

## Closed gates

| Gate | Evidence |
| --- | --- |
| Compositor tiler ownership (mode 1/5/9 + BC single-level) | Splash HEXA DRIVE / KONAMI with `REMU_CAPTURE_FRAMES=1` |
| Matrix `captured_frames=0` misconception | Often capture-off — not present failure |
| NGG mesh LDS over-allocation (`f28cee1`) | SDK `ShShaderResource2Gs::getLdsSize`; title mesh TG overflow gone |
| Mode-5 compressed mip-chain GPU tiler (`8e4ed20`) | Live refuse count **0** (was ~10 @ `f28cee1`) |
| Latched `STENCIL_ENABLE` aborting color MRTs (`0a22e15`) | Prior “stencil-only” refuses on CB-backed posts gone; depth decode returns `Ok(None)` |

## Live progression @ `0a22e15` (180s, Cross, capture on)

Artifacts: `runs/silent-hill-title-bg/` (launch loadavg ~3.75 — no FPS claim).

| Stage | Assessable? |
| --- | --- |
| Splash logos | Yes |
| Content-warning + **ruined hallway 3D** | Yes (`cecb5b924b9bc9db`) |
| Title UI (NEW GAME / OPTIONS / CONTROLS) | Yes |
| Title **3D background** | **No — still solid black** (`e1027392f3b4632e`, same pixel hash as pre-fix) |

Metrics: `captured_frames=26`; `pipeline_failed_graphics=0`; dropped **603/20** keys (was 684/22 @ `8e4ed20` — stencil keys removed). Worst remains `es:0x8007a80000/ps:0x8008020000` no-attachment (290). Zero compressed-mip refuses. Zero `stencil-only targets are not implemented` refuses; latched-stencil diagnostics only.

## Root causes (historical → current)

1. **Splash black / `captured_frames=0`:** Phase 1G refused sampled upload for
   `tile ∈ {1,5,9}` — fixed (`b3c5b5b`…`1d5d4b3`). Capture-off also zeroed matrix PPMs.
2. **Title mesh TG overflow:** hardcoded 16 KiB LDS → Metal 32 KiB limit — fixed (`f28cee1`).
3. **Mode-5 BC mip upload refuse:** exact mode-5 sampled views hit
   `refuses compressed mip chains (single-level only)` — fixed (`8e4ed20`).
4. **Color MRT posts aborted by latched STENCIL_ENABLE:** `decode_depth_target` returned `Err`
   when `STENCIL_ENABLE` was set with Z ops off, even with valid `CB_TARGET_MASK` /
   `target=0x8037bd0000` / `0x8025bb0000`. Fixed (`0a22e15`): no depth attachment → `Ok(None)`;
   true stencil-only (no color either) still loud-refused at `graphics_pipeline_key_from_targets`.
5. **Title background still black after (4):** not explained by stencil alone (proven @ `0a22e15`,
   identical title PPM hash). Remaining signals: no-attachment compositor
   `es=0x8007a80000/ps=0x8008020000` (`target_mask=0` with live `ps_scalars` SRDs), null /
   type-0 IMAGE-FALLBACKS (opaque black), NGG topology `0x7` (1 empty draw).

## Anti-claims

- Do not treat matrix `captured_frames=0` as black present when capture was unset.
- Hallway on the content-warning screen and title UI are **not** title-background 3D closure
  (“success is not the hallway”).
- Closing mode-5 BC mip refuses ≠ title-background 3D closure (proven @ `8e4ed20`).
- Closing latched-stencil color-MRT aborts ≠ title-background 3D closure (proven @ `0a22e15`).
- Do not claim FPS unless launch loadavg 1-minute < 5 and a timed interactive measurement is taken.
- Alias/sampled publish stays default-off (`REMU_PUBLISH_INTERNAL_WRITEBACKS=1` opt-in).

## Next recommended blockers

1. **Title compositor sample path:** when scanout shows title UI, log
   `REMU_CAPTURE_TARGET_SAMPLES=1` SRDs for the fullscreen pass that should supply the BG layer —
   resident `gpu_written` vs IMAGE-FALLBACK black vs CPU upload of unpublished guest bytes.
2. **No-attachment `ps=0x8008020000` / friends:** many drops keep `target_mask=0` while
   `ps_scalars` hold live image SRDs — determine UAV/`ImageStore`-only vs truly discarded covers;
   do not silently invent CB targets.
3. **Type-0 / null IMAGE-FALLBACK:** SDK buffer V# uses TYPE nibble 0; opaque-black fallback
   masks bad binds — prefer loud attribution over silent black modulation.
4. Mode-1/9 compressed mips + NGG rect-list `0x7` if proven on the title-BG critical path.
5. Parallel if SH stalls: DeS `sceAcmBatchStartBuffers`; Dreaming Sarah / Void null-SIGSEGV.

## Commits (this continuation)

| Commit | Concern |
| --- | --- |
| `b3c5b5b`…`1d5d4b3` | GPU-tiler mode-1/5/9 (+ BC) sampled ownership |
| `f28cee1` | NGG mesh LDS from `ShShaderResource2Gs::getLdsSize` |
| `8e4ed20` | Mode-5 compressed mip-chain GPU-tiler ownership |
| `0a22e15` | Latched `STENCIL_ENABLE` must not abort color MRTs |

## Checks (`0a22e15`)

- `cargo test -p remu-gpu-metal --lib presenter::tests::decode` — green
- `cargo check --workspace --all-targets` — arm64 + `x86_64-apple-darwin` green
- Live SH 180s Cross+capture — hallway + title UI; title 3D black; stencil-only refuses **gone**
