# Silent Hill present / UI / 3D — compositor investigation (updated 2026-08-10)

**Tip:** `main` @ `9f303f98` plus the uncommitted 2026-08-10 architecture-audit changes. The title UI is assessable, but its
3D background is black. The content-warning hallway is 2D and is not a 3D checkpoint.
Silent Hill 2 is an Unreal Engine 5 title, not Unity; the run overlay also contains
`ps5engine.ini` and `ue4commandline.txt`. Treat it as a renderer test case, not proof of a
Unity-specific shared failure. See the [official PlayStation/Bloober announcement](https://blog.playstation.com/2022/10/19/silent-hill-2-remake-revealed-first-gameplay-details-and-design-changes-announced/).

## 2026-08-10 log-only forensic correction (no new game run)

- The recognizable but spatially decomposed 3D scene was restored by clean commit `3f9e559`, not
  by the later compositor experiments. An empty same-address R32Float sampling shadow had stolen
  authority from the native Depth32Float DB at `801ac2`. Keeping Depth32Float authoritative made
  `depth_native` rise from 0 to 469 and made post-BasePass `800c7e` scene color nonzero
  (`804345`, `magnitude_nz≈0x13b220`). See `runs/silent-hill-title-bg-depthfix/EVIDENCE.txt`.
- Re-reading archived clean `composite4` draw 30370 changes the remaining attribution. Its
  `800dbb`/`800e73` pixel shader executed on 746,496 pixels and exported only the expected
  `±0.000976` dither while the BGRA8 target remained `(0,0,0,1)`. The shader is straight-line at
  this stage; this is not an `800e73` EXEC/control-flow or target-conversion loss.
- `800e73`'s final color lookup samples tex3, a 32×32×32 RGB10A2Unorm 3D color-grading LUT at
  `803b680000`. The preserved native capture proves every one of its 32 depth planes has RGB
  min=max=0 and A min=max=1. A valid HDR scene therefore becomes black at the final lookup.
- The immediately preceding draw generates that LUT: sequence 24345, `800aed`/`800d700`, four
  vertices, 32 instances/layers, target `803b680000`. Its archived generated mesh shader allocated
  `threadgroup atomic_uint remu_lds_storage[544]` (2,176 bytes), but guest GS_RSRC2 `0x008b0000`
  requests 2,176 **dwords** (8,704 bytes) and the shader accesses through byte `0x1d20`. Bounds
  guards therefore suppressed the relevant LDS payload writes/reads and left mesh positions zero,
  so the LUT retained its opaque-black clear value.
- The factor-of-four fix is architectural: SDK `ShShaderResource2Gs::getLdsSize()` returns a dword
  count, and AMD documents shader LDS allocation granularity in dwords. The current dirty tree also
  corrects downstream NGG issues exposed after restoring the payload: unsigned SOPK immediate
  zero-extension, per-guest-wave ES/GS invocation counts, linked-continuation GS input-VGPR
  reseeding, passthrough PrimitiveID placement, `UcGeCntl` amplification, and dynamic linked-ES/GS
  vertex stride. Focused offline tests pass; because no clean game run was made, these changes are
  not yet compatibility evidence.
- This supersedes the recommendation to spend the next run on EyeAdapt or `remu_s_31`. The first
  failing resource with direct clean evidence is the all-black generated LUT and its NGG writer.
  Do not add a title/address workaround and do not apply/pop the GitHub Desktop stash; inspect it
  read-only as `stash@{0}`.

The follow-up whole-renderer audit is recorded in
`../gpu-3d-architecture-audit-20260810.md`. It additionally corrected the passthrough PrimitiveID
VGPR, the `UcGeCntl` amplification alias, dynamic linked-ES/GS item stride, threadgroup-memory
accounting, GFX10 scalar opcode tables, and `S_ADDK_I32` SCC. It also records the remaining
split-wave64, GS-instancing, and layered texture/target gaps. No game was run for that audit.

AMD's public register reference describes `LDS_SIZE` allocations in dwords:
[CIK 3D registers](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/programmer-references/CIK_3D_registers_v2.pdf).

## 2026-08-09 correction and continuation

- The correct live checkpoint is the NEW GAME screen's 3D background. Eight finite Cross pulses
  at 8, 12, 16, 20, 24, 28, 32, and 36 seconds reached it without entering the game.
  Reproduce with `REMU_AUTO_CROSS=8,12,16,20,24,28,32,36` and
  `REMU_AUTO_CROSS_REPEAT_MS=0`; do not use the default repeating pulse.
- Draw sequence and shader occurrence are not stable selectors: clean runs placed the relevant
  compositor at draws 17289, 20057, and 24492. An address-range selector was added so internal
  title targets can be selected without accidentally capturing startup/scanout occurrences.
- A bounded dirty-tree diagnostic run selected `800dbb`/`800e73` draw 22555. Its export-stage
  EyeAdapt texture at `8043330000` was a live, readable, GPU-written RGBA32Float resident. Native
  texel X was `0x3eb851f6` (`0.3600003`); mesh register `v6`, which supplies `param1.x`, was the
  identical `0x3eb851f6`. The guest-RAM shadow was zero, as expected for a GPU-owned internal image.
- Therefore shader compilation, mesh texture binding, native residency, and this `ImageLoad` are
  working. The earlier `attr1≈0`/history-address theory was inferred from the final epsilon and is
  superseded by direct register evidence; history address rotation alone is not a root cause.
- That selected compositor still wrote opaque RGB black because its main HDR tex0 was all zero.
  It occurred before the later title BasePass peak (for example draw 29158), so it is transitional
  and does not supersede the clean later `composite3` evidence that tex0 can be nonzero.
- This EyeAdapt result remains valid, but the 2026-08-10 archived-LUT evidence supersedes the next
  step below: diagnose/fix the earlier LUT writer before requesting another compositor capture.
  Do not add a title-, address-, or game-specific workaround.

Engine/platform references: Epic documents that auto exposure uses prior-frame exposure history;
Apple requires shader-read usage and correct stage resource preparation, both satisfied by the
captured resident; AMD's RDNA ISA remains the instruction-level source for `ImageLoad` semantics.
See [Epic auto exposure](https://dev.epicgames.com/documentation/unreal-engine/auto-exposure-eye-adaptation?application_version=4.27),
[Apple resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization),
[Apple shader-read usage](https://developer.apple.com/documentation/metal/mtltextureusage/shaderread),
[Apple mesh resource preparation](https://developer.apple.com/documentation/metal/mesh-and-object-shader-resource-preparation-commands),
and the [AMD RDNA ISA reference](https://gpuopen.com/wp-content/uploads/2019/08/RDNA_Shader_ISA_7July2019.pdf).

## Closed gates

| Gate | Evidence |
| --- | --- |
| Compositor tiler ownership (mode 1/5/9 + BC single-level) | Splash HEXA DRIVE / KONAMI with `REMU_CAPTURE_FRAMES=1` |
| Matrix `captured_frames=0` misconception | Often capture-off — not present failure |
| Hardcoded NGG mesh LDS over-allocation (`f28cee1`) | Register-derived allocation removed the original Metal TG overflow, but the getter's dword result was mislabeled as bytes; the LUT writer was consequently under-allocated by 4×. |
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
| `800bf4` zero exports — UV/attr0 class (revised) | Heap V# `0x00005204` is SDK **RegularBuffer** (not word3 mis-decode); meshlet patches format 64/77. VB has float ±1 UV. SLoad heap span `0x200` (`37589c3`). Live `tonemap5c`: TARGET **magnitude_nz=109231** (was 0 @ tonemap4). |
| ES UV ±0.6 / “missing +0.5” | **Guest intent** — ViewSize 1152×648 / BufferSize 1920×1080, bias 0; oversized tri → screen UV [0,0.6]; PS samples raw attr0 (ISA). Not CB scramble / sample-VGPR bug. `UV-TEX0-ATTR.txt`. |
| tex0 “780×438” vs 1920×1080 | **Diagnostic misread** — `TARGET-DRAW-TEXTURES` used `{:x?}` (1920→`780`, 1080→`438`). SRD+PPM are 1920×1080; `800c7e` fills TL 1152×648. |
| Nonzero tonemap reaches composite sample VA | Clean `composite3`: title `800e73` tex0 `80c2d4` RG11B10 **mag≈685651**; TARGET still opaque black. |
| `800e73` near-zero exports (historical inference) | Filmic path uses `attr1[0]` (ES `800dbb` EyeAdapt read) and `remu_s_31`; epsilon suggested **attr1≈0**, but direct 2026-08-09 `v6` capture supersedes that inference for the measured draw. |
| `800b44` EyeAdapt writer itself | Clean `eyeadapt`: draw **34617** → 1×1 RGBA32Float exports **`[0.18, 0.18, 1, 1]`**. |
| Same-format sample after owned-byte supersession | `884a372` + unit test; live `composite4` still epsilon — **not** the sole closer. |
| `800dbb` EyeAdapt native load | 2026-08-09 diagnostic draw 22555: native texel X and captured mesh `v6` both **`0x3eb851f6`**; resource is `Live`/readable/GPU-written. Prior attr1/history theory superseded. |
| Final composite's 3D LUT | Archived clean `composite4` draw 30370: tex3 `803b680` is D3 32³ RGB10A2; all 32 planes are exactly **`[0,0,0,1]`**. `800e73` samples it last, so its dither-only black export is expected. |

## Live progression (capture on)

| Stage | Assessable? |
| --- | --- |
| Splash logos | Yes |
| Content-warning + ruined hallway | Yes, but **2D** — not a 3D checkpoint |
| Title UI (NEW GAME / OPTIONS / CONTROLS) | Yes (prior clean runs + `depthfix` efc) |
| Title **3D background** | **No** — `composite4` @ `884a372`: `800e73` TARGET `rgb_nonzero=0`, pixel_outputs ±1e-3; efc center black |

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
   exports match float-±0 scene tex0. **Not** an AbgrFp16→RG11B10 CB packing bug.
10. **`800c7e` writes ±0 because upstream GBuffers are empty** (clean `cs800c7e` @ `32b445d`).
11. **GBuffers empty on content-warning-only; BasePass emits at title** (`basepass` @ `9cc9f2f`).
12. **Title BG black despite filled BasePass MRTs** (`mrtocc2` / `lightpost2`).
13. **Empty R32Float depth shadow stole Depth32Float DB authority** (`3f9e559`).
14. **`800bf4` true-zero exports with lit tex0** (`tonemap4`).
15. **UV/attr0 via mesh BufferLoad V#** — SLoad heap span `0x200` (`37589c3`); tonemap5c mag≠0.
16. **UV ±0.6 is intentional ViewSize/BufferSize** (`UV-TEX0-ATTR.txt`).
17. **tex0 “780×438” was `{:x?}` Debug** on 1920×1080 descriptors.
18. **`800bf4`→`800e73`→`800da3`→efc remains black despite lit tonemap** (clean
    `composite3`/`4`, `eyeadapt`). EyeAdapt residency is working for the directly measured draw.
19. **Clean `composite4` identifies the first decisive downstream zero:** `800e73` samples an
    all-black 32³ color LUT generated immediately beforehand by `800aed`/`800d700`. Its mesh LDS
    was allocated as 2,176 bytes although GS_RSRC2 requests 2,176 dwords; guest accesses through
    byte `0x1d20` were guarded out. The old generated MSL therefore could not publish valid LUT
    geometry. Unsigned SOPK, per-wave merged counts, and the linked GS VGPR ABI are additional
    general NGG corrections in the current uncommitted work, not clean live evidence yet.

## Anti-claims

- Do not treat matrix `captured_frames=0` as black present when capture was unset.
- The content-warning hallway is **2D**. It and the title UI are not title-background 3D closure.
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
- ES UV PARAM0 ±0.6 ≠ missing-+0.5 CB bug (ViewSize/BufferSize + oversized tri; ISA samples raw attr0).
- TARGET dump `width: 780` under `{:x?}` ≠ half-res tex0 (decimal 1920; PPM 1920×1080).
- Lit composite tex0 + `800b44` writing 0.18 ≠ ES EyeAdapt T# is that write VA (`composite4`).
- Different EyeAdapt history VAs ≠ a broken load: direct native texel→mesh-register evidence wins.
- An early `800e73` whose HDR tex0 is zero ≠ the stable post-BasePass compositor.
- A nonzero HDR tex0 ≠ a nonblack final composite when the last 3D LUT is `[0,0,0,1]` everywhere.
- The old `f28cee1` `getLdsSize` decode removed one over-allocation but did **not** finish LDS
  correctness: its returned count is dwords, not bytes.
- `884a372` same-format owned-supersession bind ≠ title-BG closure (live `composite4` still ε).
- Dirty-tree runs are not evidence.
- Do not claim FPS unless launch loadavg 1-minute < 5 and a timed interactive measurement is taken.
- Alias/sampled publish stays default-off (`REMU_PUBLISH_INTERNAL_WRITEBACKS=1` opt-in).

## Next recommended blockers

1. **Review the general LUT-writer patch as one architectural unit** — LDS dword→byte conversion,
   per-wave merged counts, fixed/dynamic GS input ABI, correct GFX10 scalar decode, and exact Metal
   threadgroup-memory accounting now have focused offline coverage.
2. **Regenerate/replay the captured `800aed` translation offline** — require 8,704 bytes of LDS,
   six vertices/two primitives for the small quad, nonzero positions/param0, and all 32 layers.
3. **Resolve or reject unsafe split-wave64 shapes** — mask branches, SCC, cross-half LDS ordering,
   and dynamically divergent whole-threadgroup barriers remain open. The pairwise rendezvous
   attempt caused a Metal command-buffer Error and is not evidence for this mesh writer.
4. **Only when game testing is allowed again:** validate one clean title run by capturing the LUT
   writer target first, then the following `800e73` target. Do not return to EyeAdapt/remu_s31
   unless the LUT is nonblack and the composite is still black.

## Commits (this continuation)

| Commit | Concern |
| --- | --- |
| `b3c5b5b`…`1d5d4b3` | GPU-tiler mode-1/5/9 (+ BC) sampled ownership |
| `f28cee1` | Replaced hardcoded NGG mesh LDS; its first `getLdsSize` decode mislabeled dwords as bytes. |
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
| `6cf8932` | docs: V# RegularBuffer+meshlet patch attribution (`VSHARP-ATTR`) |
| `b526ed2` | docs: record 6cf8932 and dual-arch check at SLoad tip |
| `b9a6bc24` | docs: tonemap5c live — TARGET mag≠0 after SLoad span; UV±0.6 next |
| `9b17b54` | docs: point handoff tip at tonemap5c HEAD |
| `a82f031` | TARGET dump: decimal extents (no `{:x?}` width/height) |
| `a363cde` / `8e75b620` | docs: UV/tex0 attribution closed |
| `884a372` | Same-format GPU sample after owned-byte supersession (EyeAdapt-class) |

## Checks

- `cargo check -p remu-gpu-metal --all-targets` — arm64 + `x86_64-apple-darwin` green @ `884a372`
- Unit: `same_address_same_format_sample_survives_partial_owned_supersession` + BGRA color-view sibling
- Live clean `composite4` @ `884a372`: `800e73` still ε / `rgb_nonzero=0`; ES T# `804327` ≠ writer `803b67`
- Live clean `eyeadapt`: `800b44` exports `[0.18,0.18,1,1]`
- 2026-08-09 bounded dirty diagnostic: draw 22555 native EyeAdapt X = mesh `v6` = `0x3eb851f6`;
  target opaque RGB black; HDR tex0 zero; later BasePass-class peak at draw 29158. Diagnostic only,
  not regression/compatibility evidence.
- Focused unit tests: mesh diagnostic capture, address-range target selection, and same-address
  same-format resident sampling all pass.
- 2026-08-10 log-only/offline checks (no game process):
  `silent_hill_lut_draw_lds_covers_its_captured_vertex_payload`,
  `unsigned_sopk_compare_zero_extends_its_immediate`,
  `packs_wave64_merged_wave_info_for_split_ngg_mesh_exports`,
  `proven_ngg_registers_lower_empty_workaround_without_the_snapshot_peephole`,
  `lowers_linked_ngg_front_and_export_continuation_with_silent_hill_lds_allocation`, and
  `selects_the_gfx10_ngg_input_vgpr_layout_from_primgen_passthrough` pass on the dirty tree.
- Full `cargo test -p remu-shader-metal`: **177/179 pass**. The two known irreducible-control-flow
  dispatcher tests still fail; this is not a green package gate and is unrelated to the focused
  LUT checks above.

New gated diagnostic controls are `REMU_DUMP_TARGET_ADDRESS_MIN` /
`REMU_DUMP_TARGET_ADDRESS_MAX` (inclusive hexadecimal bounds) and
`REMU_DIAGNOSTIC_CAPTURE_VERTEX_POSITION=1`. Mesh captures bind buffer 28 and record position plus
`v0..v19` by Metal mesh-local thread; non-mesh captures retain the prior guest-index behavior.
