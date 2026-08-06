# Silent Hill present / UI / 3D — tiler + mesh LDS + mode-5 BC mips (2026-08-06)

**Tip:** `main` @ `8e4ed20`.

## Closed gates

| Gate | Evidence |
| --- | --- |
| Compositor tiler ownership (mode 1/5/9 + BC single-level) | Splash HEXA DRIVE / KONAMI with `REMU_CAPTURE_FRAMES=1` |
| Matrix `captured_frames=0` misconception | Often capture-off — not present failure |
| NGG mesh LDS over-allocation (`f28cee1`) | SDK `ShShaderResource2Gs::getLdsSize`; title mesh TG overflow gone |
| Mode-5 compressed mip-chain GPU tiler (`8e4ed20`) | Live refuse count **0** (was ~10 @ `f28cee1`) |

## Live progression @ `8e4ed20` (180s, Cross, capture on)

Artifacts: `runs/silent-hill-bc-mips/` (launch loadavg ~6.2 — no FPS claim).

| Stage | Assessable? |
| --- | --- |
| Splash logos | Yes |
| Content-warning + **ruined hallway 3D** | Yes |
| Title UI (NEW GAME / OPTIONS / CONTROLS) | Yes |
| Title **3D background** | **No — still solid black** |

Metrics: `captured_frames=19`; `failed=0/0`; ready≈379; **zero**
`compressed mip chains` refuses; **zero** `Threadgroup memory` / LDS-attribute-ring refuses.
Dropped draws (22 keys) are now: no-attachment compositor (18), stencil-only (3), NGG
topology `0x7` (1). Worst draw remains `es=0x8007a80000/ps:0x8008020000` (pipeline key /
no-attachment), not image-upload fails.

## Root causes (historical → current)

1. **Splash black / `captured_frames=0`:** Phase 1G refused sampled upload for
   `tile ∈ {1,5,9}` — fixed (`b3c5b5b`…`1d5d4b3`). Capture-off also zeroed matrix PPMs.
2. **Title mesh TG overflow:** hardcoded 16 KiB LDS → Metal 32 KiB limit — fixed (`f28cee1`).
3. **Mode-5 BC mip upload refuse:** exact mode-5 sampled views hit
   `refuses compressed mip chains (single-level only)` — fixed (`8e4ed20`) by planning
   AddrLib mode-5 chains in element space and blitting floor-shifted pixel extents.
   Mode-1/9 compressed mips still loud-refuse; mode-27 BC still unowned.
4. **Title background still black after (3):** not explained by the closed mip refuse.
   Remaining signals: compositor pair `ps=0x8008020000` (incl. `target=0x0` / no-attachment
   drops), texture-type-0 / null SRD IMAGE-FALLBACKS, stencil-only, NGG rect-list `0x7`,
   sampled-alias publish default-off (debt kept — by design).

## Anti-claims

- Do not treat matrix `captured_frames=0` as black present when capture was unset.
- Hallway on the content-warning screen and title UI are **not** title-background 3D closure
  (“success is not the hallway”).
- Closing mode-5 BC mip refuses ≠ title-background 3D closure (proven @ `8e4ed20`).
- Do not claim FPS (launch loadavg was >5).
- Alias/sampled publish stays default-off (`REMU_PUBLISH_INTERNAL_WRITEBACKS=1` opt-in).

## Next recommended blockers

1. Trace compositor / title-background path after splash: `ps=0x8008020000` /
   `0x8008040000`, SRD binds (`REMU_CAPTURE_TARGET_SAMPLES=1`), and why many compositor
   draws still hit no-attachment `target=0x0`.
2. Mode-1/9 compressed mip layouts (still refused) if title assets use those tiles with
   `mips>1` — keep loud until AddrLib-backed.
3. NGG topology `0x7` (rect list) + stencil-only targets if those draws are on the title path.
4. Parallel if SH stalls: DeS `sceAcmBatchStartBuffers`; Dreaming Sarah / Void null-SIGSEGV.

## Commits (this continuation)

| Commit | Concern |
| --- | --- |
| `b3c5b5b`…`1d5d4b3` | GPU-tiler mode-1/5/9 (+ BC) sampled ownership |
| `f28cee1` | NGG mesh LDS from `ShShaderResource2Gs::getLdsSize` |
| `8e4ed20` | Mode-5 compressed mip-chain GPU-tiler ownership |

## Checks (`8e4ed20`)

- `cargo test -p remu-gpu-metal --lib tiler::encode::tests` — green
- `cargo check --workspace --all-targets` — arm64 + `x86_64-apple-darwin` green
- Live SH 180s Cross+capture — hallway + title UI; title 3D black; mip-chain refuses **gone**
