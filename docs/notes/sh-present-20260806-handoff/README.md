# Silent Hill present / UI / 3D — tiler + mesh LDS + mode-5 BC mips + stencil latch + sample-path (2026-08-06)

**Tip:** `main` @ `e75d19c` (`VTX_W0_FMT`) on top of `ef55a61` / `a86c225` attribution.

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

## Live progression (capture on)

| Stage | Assessable? |
| --- | --- |
| Splash logos | Yes |
| Content-warning + **ruined hallway 3D** | Yes |
| Title UI (NEW GAME / OPTIONS / CONTROLS) | Yes |
| Title **3D background** | **No — still solid black** |

Clean post-`VTX_W0_FMT` @ `e75d19c` (`runs/silent-hill-title-bg-vte-w0/`, `dirty_tree=0`,
launch loadavg ~3.1→4.8): `captured_frames=19`; title UI present on black void; hallway 3D
still assessable. First `800bf4` `[TARGET]`: `PA_CL_VTE_CNTL=0x43f`, full-screen pixel
coverage (`2073600`), but `nonzero=0` `fp=97d30483e5dd6325` and PS exports `±0.0` despite
sampled natives with `nz≠0`.

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
9. **`800bf4` still writes clear-black after coverage** (clean `vte-w0` run): PS runs
   full-screen but exports `±0.0`; RT `nonzero=0` same fingerprint as pre-fix. **Next
   architectural concern:** why tonemap/HDR pixel math stays zero given non-zero samples
   (`AbgrFp16`→`RG11B10` / sample path / inputs) — not VTE clip.

## Anti-claims

- Do not treat matrix `captured_frames=0` as black present when capture was unset.
- Hallway on the content-warning screen and title UI are **not** title-background 3D closure
  (“success is not the hallway”).
- Closing mode-5 BC mip refuses ≠ title-background 3D closure (proven @ `8e4ed20`).
- Closing latched-stencil color-MRT aborts ≠ title-background 3D closure (proven @ `0a22e15`).
- Closing type-0 refuse + BGRA↔RGBA authority ≠ title-background 3D closure (proven live: BG still black with ALIAS-VIEW).
- Proving intermediate texels black ≠ title-background 3D closure — only attributes the failure upstream of `800da3`.
- Restoring `VTX_W0_FMT` coverage ≠ title-background 3D closure — clean live @ `e75d19c` still
  shows title UI on black void with `800bf4` `nonzero=0`.
- Dirty-tree runs are not evidence.
- Do not claim FPS unless launch loadavg 1-minute < 5 and a timed interactive measurement is taken.
- Alias/sampled publish stays default-off (`REMU_PUBLISH_INTERNAL_WRITEBACKS=1` opt-in).

## Next recommended blockers

1. **`800bf4` PS exports zero after coverage** — dump MSL/`AbgrFp16`→`RG11B10` / sample
   fetches for `es=0x800d690000 ps=0x800bf40000` with live `pixel_outputs` proving coverage.
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

## Checks

- `cargo test -p remu-shader-metal --lib forces_clip_space_w_to_one` — green
- `cargo test -p remu-gpu-metal --lib carries_pa_cl_vte_cntl` — green
- `cargo check -p remu-gpu-metal -p remu-shader-metal` — arm64 + `x86_64-apple-darwin` green
- Live SH clean @ `e75d19c` (`runs/silent-hill-title-bg-vte-w0/`) — hallway + title UI;
  title 3D still black; `800bf4` coverage restored but `nonzero=0` / PS `±0.0`
