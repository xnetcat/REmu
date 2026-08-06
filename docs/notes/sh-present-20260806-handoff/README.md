# Silent Hill present / UI / 3D — tiler + mesh LDS + mode-5 BC mips + stencil latch + sample-path (2026-08-06)

**Tip:** `main` @ SLoad heap-span + V# attribution (`VSHARP-ATTR.txt`). Depth fix `3f9e559`.
Live post-BP tonemap: clean `runs/silent-hill-title-bg-tonemap4/` + `UV-ATTR-ANALYSIS.txt` + `VSHARP-ATTR.txt`.

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
| R32 depth-sample view stealing Depth DB authority | `3f9e559` + live `runs/silent-hill-title-bg-depthfix/`: `depth_native` max **469** (was **0** forever on `lightpost2`). Post-BP `800c7e` writes non-zero scene color to `804345` (`magnitude_nz≈0x13b220`). |
| `800bf4` empty TARGET with lit tex0 — not blend/coverage/bind | Clean `runs/silent-hill-title-bg-tonemap4/` @ `267d256` + AUTO_CROSS, `MIN_DRAW=47000`: draw **47402** after BasePass peak (`mrt3plus≈8192`). tex0 `804345` Metal+guest `magnitude_nz≃0xf5640`; blend off; TARGET + exports true 0.0. Packing not implicated. |
| `800bf4` zero exports — UV/attr0 class (revised) | Heap V# `0x00005204` is SDK **RegularBuffer** (not word3 mis-decode); meshlet patches format 64/77. VB at `7120e7d8c0` has real float ±1 UV + UTF-16 padding. See `VSHARP-ATTR.txt`. Remaining: UV scale-bias / TL sample + `804345` tiler garbage. |

## Live progression (capture on)

| Stage | Assessable? |
| --- | --- |
| Splash logos | Yes |
| Content-warning + **ruined hallway 3D** | Yes |
| Title UI (NEW GAME / OPTIONS / CONTROLS) | Yes (prior clean runs + `depthfix` efc) |
| Title **3D background** | **No — still solid black** (UI yes; BasePass + lit tex0; `800bf4` true-zero; V# decode cleared — RegularBuffer+meshlet patch; UV transform / TL sample open) |

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
    Deferred lighting `800a69` is **not** re-submitted after title BasePass (title uses
    `8009c1` onto MRT0 instead). **`800c7e` does re-run after BasePass** (PERF-BATCHES;
    prior “never after BasePass” overstated from early-only `COMPUTE-IMAGES` dumps).
13. **Empty R32Float depth shadow stole Depth32Float DB authority** (fix `3f9e559`):
    post-prepass samples at `801ac2`/`801abe` bound a gpu-authored magnitude-0 R32Float
    resident while Depth32Float was demoted (`gpufalse`) → `depth_native=0`. Live after fix:
    `depth_native` hits hundreds; post-BP `800c7e` writes non-zero `804345`.
14. **`800bf4` true-zero exports with lit tex0** (clean `tonemap4` @ `267d256`):
    post-BasePass draw 47402, blend off, tex0 `804345` Metal+guest magnitude ≫0, TARGET
    and exports true float 0.0. Not packing / empty bind / pre-BP stale SRD (`tonemap3`).
15. **UV/attr0 via mesh BufferLoad V#** (tonemap4 TARGET globals + SDK AGC):
    Heap `8054eb9354` holds **RegularBuffer** `0x00005204` (= `initializeRegularBuffer`
    k8UInt+X001) — **not** a BUF_RSRC_WORD3 mis-decode. UE patches FORMAT from meshlet
    (`s21` format 64 float2, soff 16). VB `7120e7d8c0` is float ±1 pos/uv with UTF-16
    padding (not material ASCII). Sticky pixel diagnostic fixed (`d23a898`). Remaining:
    UV scale-bias maps NDC±1 → ~±0.6 (samples empty TL of lit `804345`); tiler garbage
    on `804345` once UVs forward; SLoad heap bind floor `0x200` for multi-slot.

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
- Claiming `800c7e` “never after BasePass” from early-only `COMPUTE-IMAGES` ≠ absence (PERF-BATCHES
  shows post-BP `cs=0x800c7e`).
- `depth_native>0` + non-zero `800c7e` scene color ≠ title-BG closure while `800bf4`/efc stay empty/black.
- Dumping `800bf4` before BasePass peak (`MIN_DRAW≈22000`) can bind empty `803e70` and must not
  be cited as the lit-tex0 failure mode (`tonemap3` vs `tonemap4`).
- Non-zero tex0 magnitude alone ≠ tonemap forwards it (`tonemap4` pixel_outputs all 0.0).
- Sticky `REMU_DIAGNOSTIC_CAPTURE_PIXEL_OUTPUT` buffer (no per-draw clear) ≠ per-draw coverage.
- Heap RegularBuffer `0x00005204` ≠ BUF_RSRC_WORD3 mis-decode (matches AGC `initializeRegularBuffer`).
- UTF-16 in stride-32 VB padding ≠ “VB is material/debug ASCII only”.
- Dirty-tree runs are not evidence.
- Do not claim FPS unless launch loadavg 1-minute < 5 and a timed interactive measurement is taken.
- Alias/sampled publish stays default-off (`REMU_PUBLISH_INTERNAL_WRITEBACKS=1` opt-in).

## Next recommended blockers

1. **Live confirm meshlet FORMAT patch + PARAM0 UV range** on post-BP `800bf4` (MESH-V#
   diagnostic; per-draw pixel clear). If patch applies, fix ES/PS UV scale-bias
   (1152/1920 vs missing +0.5) so samples leave empty TL of `804345`.
2. **Why `804345` Metal texels are quadrant-garbage** (tiler/store) once tonemap forwards
   nonzero samples — otherwise BG still will not be assessable scene content.
3. **Optional:** dump `8009c1` TARGET post-BP for MRT0 lighting exports (title substitute for
   absent post-BP `800a69`).
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
| `3f9e559` | Keep Depth32Float authority over same-address R32 depth sample views |
| `267d256` | docs: depth-authority fix live; 800bf4 still empty after lit tex0 |
| `4c819b9` | docs: tonemap4 — true-zero exports with lit tex0 post-BasePass |
| `d23a898` | Clear TARGET diagnostic buffers each draw under capture |
| `38a9747` | docs: 800bf4 UV collapse via mesh V# DST_SEL_Y=0 / format 5 |
| `76e6828` | docs: point handoff commits table at UV-attr tip |
| `37589c3` | Expand dynamic-offset S_LOAD binds to descriptor-heap span + MESH-V# warn |
| *(pending)* | docs: V# RegularBuffer+meshlet patch attribution (`VSHARP-ATTR`) |

## Checks

- `cargo check -p remu-runtime -p remu-gpu-metal -p remu-gpu` — arm64 + `x86_64-apple-darwin` green @ `3f9e559`
- Unit: `r32_depth_sample_*` (prefer Depth over clean R32; keep Depth after later R32 mark_written)
