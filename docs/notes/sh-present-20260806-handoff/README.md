# Silent Hill present / UI / 3D — tiler + mesh LDS + mode-5 BC mips + stencil latch + sample-path (2026-08-06)

**Tip:** `main` @ `32b445d` (`COMPUTE-IMAGES` + ImageLoad writer lineage) on top of
`391f3fc` / `bbe413d` (`magnitude_nz`) / `e75d19c` (`VTX_W0_FMT`).

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
| Attribution: black intermediate vs composite discard | Clean `runs/silent-hill-title-bg-occ/` — intermediate **RGB black**; not a later blend discard |
| Missing `PA_CL_VTE_CNTL.VTX_W0_FMT` → Metal clip (no coverage) | Clean `runs/silent-hill-title-bg-vte-w0/` @ `e75d19c`: `800bf4` now has `pixel_outputs=Some((2073600,…))` vs pre-fix `None` |
| `800bf4` zero exports after coverage — cause class | Clean vte-w0: PS exports `±0.0` because tonemap **tex0 is float-±0**; AbgrFp16→RG11B10 packing not implicated; prior “nz≠0” was sign-bit byte occupancy |
| `800c7e` float-±0 scene color — not missing color SRD | Clean `runs/silent-hill-title-bg-cs800c7e/` @ `32b445d`: CS `ImageLoad`s full-res `803d6e` (all-zero) + depth; writes ±0. Sampled 1×1 dummies are optional. Upstream `800a69` full coverage + zero exports; GBuffer `8023b9` Metal+guest empty |

## Live progression (capture on)

| Stage | Assessable? |
| --- | --- |
| Splash logos | Yes |
| Content-warning + **ruined hallway 3D** | Yes |
| Title UI (NEW GAME / OPTIONS / CONTROLS) | Yes |
| Title **3D background** | **No — still solid black** |

Clean post-`COMPUTE-IMAGES` @ `32b445d` (`runs/silent-hill-title-bg-cs800c7e/`,
`dirty_tree=0`, launch loadavg ~2.2→4.1): `captured_frames=33`; title UI present on black
void; hallway 3D still assessable. `800c7e` ImageLoads empty `803d6e`; `800a69` exports
zeros over full coverage; GBuffer `8023b9` empty.

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
   `ResourceSpan` overlap stole owned-byte authority — fixed (`3c025bd`).
7. **Title intermediate Metal texels are clear-black at title** (proven clean @ `a86c225`):
   failure upstream of `800da3`/ALIAS — content never arrived.
8. **HDR producer lacked mesh coverage:** ES float4 VB left **W=0**; AGC
   `PA_CL_VTE_CNTL=0x43f` (`VTX_W0_FMT`) was not applied → Metal clipped the triangle.
   Fixed (`e75d19c`); live proves coverage restored.
9. **`800bf4` exports float-zero after coverage** (clean `vte-w0`): PS runs full-screen;
   exports match float-±0 scene tex0. **Not** an AbgrFp16→RG11B10 CB packing bug. Prior
   “sampled natives nz≠0” counted IEEE sign bits (`0x80` / `0x80000000`). Writer of tex0 is
   compute `800c7e`.
10. **`800c7e` writes ±0 because upstream GBuffers are empty** (clean `cs800c7e` @ `32b445d`):
    `COMPUTE-IMAGES` shows `ImageLoad` of full-res `803d6e` RGBA16Float + depth + R8Uint
    (plus sampled 1×1 optionals). Readback bins: color+depth magnitude-zero; store is fp16
    `0x8000` only. Lighting `800a69` has full coverage but zero exports; its sampled
    `8023b9` is all-zero in Metal and guest. Sampled-only writer lineage previously hid
    the ImageLoads — not “CS only binds dummies”.

## Anti-claims

- Do not treat matrix `captured_frames=0` as black present when capture was unset.
- Hallway on the content-warning screen and title UI are **not** title-background 3D closure
  (“success is not the hallway”).
- Closing mode-5 BC mip refuses ≠ title-background 3D closure (proven @ `8e4ed20`).
- Closing latched-stencil color-MRT aborts ≠ title-background 3D closure (proven @ `0a22e15`).
- Closing type-0 refuse + BGRA↔RGBA authority ≠ title-background 3D closure (proven live: BG still black with ALIAS-VIEW).
- Proving intermediate texels black ≠ title-background 3D closure — only attributes the failure upstream of `800da3`.
- Restoring `VTX_W0_FMT` coverage ≠ title-background 3D closure — clean live @ `e75d19c` still
  shows title UI on black void with `800bf4` float-zero exports.
- Byte/`bit` nonzero on HDR textures that are float-±0 ≠ “non-zero sampled natives”.
- Sampled-only diagnostic writer lists for compute ≠ complete image inputs (`ImageLoad` counts).
- Proving `800c7e` binds real (but empty) GBuffers ≠ title-background 3D closure.
- Dirty-tree runs are not evidence.
- Do not claim FPS unless launch loadavg 1-minute < 5 and a timed interactive measurement is taken.
- Alias/sampled publish stays default-off (`REMU_PUBLISH_INTERNAL_WRITEBACKS=1` opt-in).

## Next recommended blockers

1. **Empty GBuffer sources at title** — why `8023b9` / `8025ba` / depth stay magnitude-zero
   before lighting `800a69`/`8009c4`/`800ae7` (missing geometry writer, dropped draws, clear,
   or unpublished resident). Clean evidence: `runs/silent-hill-title-bg-cs800c7e/`.
2. **No-attachment `ps=0x8008020000` / friends:** many drops keep `target_mask=0` while
   `ps_scalars` hold live image SRDs — only if proven on the HDR/title-BG path; do not invent CB targets.
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
| `3c025bd` | Type-0 refuse + BGRA↔RGBA sample authority |
| `a86c225` | BGRA occupancy dump + `ADDRESS=auto` skips scanout covers |
| `e75d19c` | Honor `PA_CL_VTE_CNTL.VTX_W0_FMT` in Metal position export |
| `bbe413d` | TARGET/PS occupancy: float ±0 is not magnitude-nonzero |
| `32b445d` | Compute `COMPUTE-IMAGES` + ImageLoad inputs in writer lineage |

## Checks

- `cargo test -p remu-shader-metal --lib forces_clip_space_w_to_one` — green
- `cargo test -p remu-gpu-metal --lib carries_pa_cl_vte_cntl` — green
- `cargo test -p remu-gpu-metal --lib magnitude_nonzero` — green
- `cargo check -p remu-gpu-metal` — arm64 + `x86_64-apple-darwin` green @ `32b445d`
- Live SH clean @ `32b445d` (`runs/silent-hill-title-bg-cs800c7e/`) — hallway + title UI;
  title 3D still black; `800c7e` ImageLoads empty GBuffers; `800a69` zero exports
