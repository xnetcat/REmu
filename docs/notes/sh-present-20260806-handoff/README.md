# Silent Hill present / UI / 3D — tiler + mesh LDS + mode-5 BC mips + stencil latch + sample-path (2026-08-06)

**Tip:** `main` @ (see git); code tip after this continuation lands type-0 refuse + BGRA↔RGBA authority.

## Closed gates

| Gate | Evidence |
| --- | --- |
| Compositor tiler ownership (mode 1/5/9 + BC single-level) | Splash HEXA DRIVE / KONAMI with `REMU_CAPTURE_FRAMES=1` |
| Matrix `captured_frames=0` misconception | Often capture-off — not present failure |
| NGG mesh LDS over-allocation (`f28cee1`) | SDK `ShShaderResource2Gs::getLdsSize`; title mesh TG overflow gone |
| Mode-5 compressed mip-chain GPU tiler (`8e4ed20`) | Live refuse count **0** (was ~10 @ `f28cee1`) |
| Latched `STENCIL_ENABLE` aborting color MRTs (`0a22e15`) | Prior “stencil-only” refuses on CB-backed posts gone; depth decode returns `Ok(None)` |
| Non-null TYPE-0 IMAGE-FALLBACK → opaque black | Loud `[IMAGE-REFUSE] reason=non-null-type-0-buffer-srd` instead of silent black 1×1 |
| False shared-heap SampledAlias / authority theft on title intermediate | BGRA→RGBA `ALIAS-FLUSH` **0**; `ALIAS-VIEW` binds gpu-written BGRA; VA-only authority subtract |

## Live progression (capture on)

| Stage | Assessable? |
| --- | --- |
| Splash logos | Yes |
| Content-warning + **ruined hallway 3D** | Yes |
| Title UI (NEW GAME / OPTIONS / CONTROLS) | Yes |
| Title **3D background** | **No — still solid black** (hash moved off `e1027392f3b4632e` only because UI chrome differed; BG still void) |

Dirty tagged probe @ alias fix (`runs/silent-hill-title-bg-alias/`, `REMU_ALLOW_DIRTY_EVIDENCE=1`, launch loadavg ~4.1→6): `captured_frames=26`; `pipeline_failed=0`; `IMAGE-REFUSE` live; `ALIAS-VIEW` for title intermediate `0x8044550000` BGRA→RGBA; SAMPLE shows gpu-written BGRA alias. **Title BG still black** — color-view path is no longer the sole blocker.

## Root causes (historical → current)

1. **Splash black / `captured_frames=0`:** Phase 1G refused sampled upload for
   `tile ∈ {1,5,9}` — fixed (`b3c5b5b`…`1d5d4b3`). Capture-off also zeroed matrix PPMs.
2. **Title mesh TG overflow:** hardcoded 16 KiB LDS → Metal 32 KiB limit — fixed (`f28cee1`).
3. **Mode-5 BC mip upload refuse:** exact mode-5 sampled views hit
   `refuses compressed mip chains (single-level only)` — fixed (`8e4ed20`).
4. **Color MRT posts aborted by latched STENCIL_ENABLE:** fixed (`0a22e15`).
5. **Type-0 / null IMAGE-FALLBACK painting opaque black:** non-null TYPE nibble 0 now
   loud-refuses; null SRDs still optional-bind fallback.
6. **Title intermediate BGRA8 CB sampled as RGBA8:** Metal color-view existed, but shared-heap
   `ResourceSpan` overlap with neighbours (e.g. RGBA16Float) stole owned-byte authority
   (`!serves_reads` while `gpu_written`), forcing SampledAlias flush → RemuBookkeeping refuse →
   CPU upload of unpublished guest RAM (black). Fixed: VA/share_byte_storage authority guard +
   same-address same-extent color-view eligibility without full `serves_reads`.
7. **Title background still black after (6):** `800e73`→intermediate→`800da3`→scanout still
   produces black BG while ALIAS-VIEW binds. Next: prove whether intermediate texels are black
   (upstream title scene / HDR sample) vs composite/blend after a correct view.

## Anti-claims

- Do not treat matrix `captured_frames=0` as black present when capture was unset.
- Hallway on the content-warning screen and title UI are **not** title-background 3D closure
  (“success is not the hallway”).
- Closing mode-5 BC mip refuses ≠ title-background 3D closure (proven @ `8e4ed20`).
- Closing latched-stencil color-MRT aborts ≠ title-background 3D closure (proven @ `0a22e15`).
- Closing type-0 refuse + BGRA↔RGBA authority ≠ title-background 3D closure (proven live: BG still black with ALIAS-VIEW).
- Dirty-tree runs are not evidence.
- Do not claim FPS unless launch loadavg 1-minute < 5 and a timed interactive measurement is taken.
- Alias/sampled publish stays default-off (`REMU_PUBLISH_INTERNAL_WRITEBACKS=1` opt-in).

## Next recommended blockers

1. **Dump/prove `800e73` intermediate** (`0x8044550000` / run-local twin) occupancy at title —
   gpu-written Metal texels vs clear-black; HDR input sample path (`RG11B10` / scene target).
2. **No-attachment `ps=0x8008020000` / friends:** many drops keep `target_mask=0` while
   `ps_scalars` hold live image SRDs — UAV/`ImageStore`-only vs discarded covers; do not invent CB targets.
3. Null IMAGE-FALLBACK policy (optional binds) if still modulating covers.
4. Mode-1/9 compressed mips + NGG rect-list `0x7` if proven on the title-BG critical path.
5. Parallel if SH stalls: DeS `sceAcmBatchStartBuffers`; Dreaming Sarah / Void null-SIGSEGV.

## Commits (this continuation)

| Commit | Concern |
| --- | --- |
| `b3c5b5b`…`1d5d4b3` | GPU-tiler mode-1/5/9 (+ BC) sampled ownership |
| `f28cee1` | NGG mesh LDS from `ShShaderResource2Gs::getLdsSize` |
| `8e4ed20` | Mode-5 compressed mip-chain GPU-tiler ownership |
| `0a22e15` | Latched `STENCIL_ENABLE` must not abort color MRTs |
| (pending) | Type-0 refuse + BGRA↔RGBA sample authority |

## Checks

- `cargo test -p remu-gpu-metal --lib same_address_color_view` — green
- `cargo check --workspace --all-targets` — arm64 + `x86_64-apple-darwin` green
- Live SH dirty probe — hallway + title UI; title 3D still black; IMAGE-REFUSE live; BGRA→RGBA ALIAS-FLUSH 0
