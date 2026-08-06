# Silent Hill present / UI / 3D — tiler + mesh LDS + mode-5 BC mips + stencil latch + sample-path (2026-08-06)

**Tip:** `main` @ `de613d3` (MRT-min TARGET dump + late `COMPUTE-IMAGES`) on top of
`be21eb9` / `9cc9f2f` / `736d540`.

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
| Empty GBuffers — not Metal-dropped BasePass | Clean `runs/silent-hill-title-bg-gbuffer/` @ `736d540`: content-warning-only (no pad) had `total_mask_mrt3plus=0`; multi-MRT emits only 2-slot shadows; `COLOR-MALFORMED=0`; `target=0x8023b9…` never appears. Depth `801ac2` stays `gpufalse/v0` |
| Title BasePass **does** emit (≥3 MRT) | Clean `runs/silent-hill-title-bg-basepass/` @ `9cc9f2f` + AUTO_CROSS: title NEW GAME confirmed; `total_mask_mrt3plus≈8591`; 7×1920×1080 MRT to `8068e0…` (Metal `gputrue`); also mrt=3 incl. `801c44`. Prior “never emits” was pad-not-advanced artifact. Title BG still black; `8023b9` still never COLOR; depth `801ac2` still `v0` |
| Emitted title MRT0 **not** clear-only | Clean `runs/silent-hill-title-bg-mrtocc2/` @ `de613d3` + AUTO_CROSS: `REMU_DUMP_TARGET_MRT_MIN=7` → `806aa4` `magnitude_nz=1327104` (≈64% texels). MRT0 VA remaps (`8068e0`/`8068f4`/`806aa4`). `800a69` samples remapped `8068*` set; `8009c4` still samples legacy `8023b9`. Title BG still black |
| Post-BasePass lighting sample VA | Clean `runs/silent-hill-title-bg-lightpost2/` @ `de613d3`: `800b22` @ draw 22072 binds remapped `80688f` (BasePass BGRA8 slot), **not** empty `8023b9`. Early `800a69` still legacy-empty (pre-BasePass only). |
| Title present-chain gap (attribution) | Clean `lightpost2`: after BasePass `8070b3`, `800a69`/`800c7e` **never** re-run; only `800b22` touches `806990`/`80688f` (RGB-preserving blend no-op); `800bf4`/efc stay on stale ±0 scene-color path. MRT0 never sampled into present. |

## Live progression (capture on)

| Stage | Assessable? |
| --- | --- |
| Splash logos | Yes |
| Content-warning + **ruined hallway 3D** | Yes |
| Title UI (NEW GAME / OPTIONS / CONTROLS) | Yes (prior clean runs) |
| Title **3D background** | **No — still solid black** (title UI yes; BasePass-class MRT emits yes) |

Clean post-`COLOR-EMIT` @ `736d540` (`runs/silent-hill-title-bg-gbuffer/`,
`dirty_tree=0`, launch loadavg ~2.2→4.1): `captured_frames=33`; this particular 120s run
stayed on content-warning hallway (pad not advanced). Emit counters there cover only the
pre-title deferred stream.

Clean title-frame evidence @ `9cc9f2f` (`runs/silent-hill-title-bg-basepass/`, AUTO_CROSS,
`dirty_tree=0`, loadavg ~4.0): NEW GAME frame confirmed; `total_mask_mrt3plus≈8591` with
full-res 7-MRT writes to `8068e0…` (PERF `gputrue`); BG still void.

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
11. **GBuffers empty because deferred BasePass is never submitted** — **narrowed**: true for
    content-warning-only (clean `gbuffer` @ `736d540`, no pad). **False at title**: clean
    `basepass` @ `9cc9f2f` + AUTO_CROSS reaches NEW GAME with `total_mask_mrt3plus≈8591`
    (7-slot 1920×1080 to `8068e0…`, Metal `gputrue`). `8023b9` still never a COLOR target;
    depth `801ac2` still `gpufalse/v0`; title BG still black. COLOR-EMIT previously mis-logged
    AGC ATTRIB2 extents as GFX10 packing (`c438×1e0`); real size is 1920×1080.
12. **Title BG black despite filled BasePass MRTs** (clean `mrtocc2` + `lightpost2` @ `de613d3`):
    occupancy is real geometry magnitude, not clear-only version bumps. Post-BasePass
    `800b22` samples remapped `80688f` (RGB-black/alpha-occupied PPM; blend preserves MRT0).
    Deferred lighting `800a69` and tonemap CS `800c7e` stop before BasePass and do not
    resume; tonemap/present never consume filled `806990`. Early “empty `8023b9`” remains
    true only for the pre-BasePass lighting path.

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
- Proving BasePass never reaches AGC emit on a **content-warning-only** run ≠ title-BG closure
  and ≠ “BasePass absent at title” (clean title `basepass` run emits ≥3-slot MRTs).
- Emitting full-res multi-MRT BasePass-class draws ≠ title-BG 3D closure while BG stays black.
- Post-BasePass `800b22` binding remapped `80688f` ≠ lighting/tonemap consuming BasePass MRT0.
- Dirty-tree runs are not evidence.
- Do not claim FPS unless launch loadavg 1-minute < 5 and a timed interactive measurement is taken.
- Alias/sampled publish stays default-off (`REMU_PUBLISH_INTERNAL_WRITEBACKS=1` opt-in).

## Next recommended blockers

1. **Why `800a69` / `800c7e` never re-run after title BasePass** (clean `lightpost2`):
   never submitted vs PERF-dropped vs fence/stall. Without them, filled `8068*` MRTs never
   reach scene color / efc. Do not re-litigate empty MRT0 or “always samples `8023b9`” for
   the post-BasePass path.
2. **Clean-env dump `800bf4` with `MIN_DRAW` after BasePass geometry** (unset sticky
   `REMU_DUMP_*` / `REMU_RUN_ROOT`; loadavg 1-min <5): prove post-BP tonemap sample VAs +
   exports. `tonemap2` refused (no BasePass under hot load).
3. **Depth `801ac2` occupancy after `8007f4` prepass** (`REMU_DUMP_DEPTH_ES`) — prepass
   `write_enabled=true` but historical `gpufalse/v0` suspicion remains.
4. **No-attachment `ps=0x8008020000` / friends** — only if proven on the title-BG path; do
   not invent CB targets.
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
| `736d540` | Emit-side `COLOR-EMIT` / `COLOR-MALFORMED` + `GBUFFER-UNWRITTEN` |
| `9cc9f2f` | docs: SH GBuffers empty because BasePass never emits (content-warning scope) |
| `be21eb9` | AGC ATTRIB2 extent in `COLOR-EMIT` + title BasePass evidence correction |

## Checks

- `cargo check -p remu-runtime -p remu-gpu-metal -p remu-gpu` — arm64 + `x86_64-apple-darwin` green @ `be21eb9`
- Live SH clean title @ `9cc9f2f` (`runs/silent-hill-title-bg-basepass/`) — NEW GAME;
  `total_mask_mrt3plus≈8591`; title 3D still not closed
- Prior content-warning-only @ `736d540` (`runs/silent-hill-title-bg-gbuffer/`) —
  `total_mask_mrt3plus=0` (not title evidence)
