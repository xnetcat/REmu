# Technical findings

## The exact visible failure

Silent Hill reaches the actual title menu. New Game / Continue / Settings and related UI are
visible, but the intended 3D scene behind that UI is black. The ruined hallway seen before this
menu is not the missing scene. The brightness page accepts Cross and advances; it is grayscale
after the current fixes.

Several other titles reportedly turn black at the point where substantive 3D should begin, so a
generic shader/geometry defect is more plausible than a Silent-Hill-specific media problem.

## FFmpeg and the title movie are healthy, but not the root cause

The in-process AvPlayer path decoded Silent Hill's `UI_TitleMovie01.mp4` as 378x150 NV12. Both
planes contained non-zero data. REmu's NV12 conversion draw produced a correct, colorful 244x150
BGRA image that visually matched an FFmpeg-decoded reference frame.

The conversion target's 244-pixel width is genuine guest CB state; no matching 378x150 sampled
descriptor was observed. There is not enough evidence to call this an extent bug.

The first video-chain diagnostic reported downstream consumers only because it treated any raw
physical-range overlap as resource identity. One alleged consumer was a 72x41 RGBA16Float,
96-layer/volume-like allocation whose multi-megabyte footprint merely overlapped the 244x150
surface in a recycled heap. Matching was narrowed to exact descriptor identity or proven shared
storage. With that correction, the run emitted zero legitimate downstream consumers. The movie
conversion appears to be an auxiliary media update, not the actual title background.

## Broad sampled-alias publication is wrong

`REMU_PUBLISH_INTERNAL_WRITEBACKS=1` was tested as a diagnostic. Publishing `SampledAlias`
writebacks either crashed early or generated hundreds of MiB of readback traffic, drove the game
to 0 FPS, and eventually timed out its game thread. The reason is the same false physical-overlap
classification described above. The temporary policy change and its test changes were fully
reverted. Do not re-enable it.

The remaining video-chain diagnostic refinement is in
`crates/remu-gpu-metal/src/presenter/diagnostics.rs`. It is diagnostic-only and should be reviewed
or removed during final cleanup.

## Culling and merged counts were ruled out

Live runs with the relevant culling changes and merged-count experiments still reached the black
title background. They changed neither the decisive visual symptom nor the missing export
interface. The preserved screenshots and older logs remain useful as negative evidence, but these
are no longer the primary line of work.

## Strongest current lead: fused NGG was lowered as a vertex shader

The targeted dump used pixel shader `0x800da30000`. A representative export shader at
`0x8066760000` has the fused GFX10 NGG shape:

- reads packed merged-wave data from `s3`;
- sends `S_SENDMSG 9` (`GS_ALLOC_REQ`);
- emits `EXP target 20` for primitive connectivity;
- later emits `EXP target 12` for position and target 13 for misc output;
- performs real vertex-buffer loads;
- contains no conventional `EXP PARAM 32..63` in its export half.

Before the last change, the compile log nevertheless said `stage=vertex`. The generated
`vertex.metal` ignored the primitive/allocation contract and only returned one local
`RemuVsOut`. That cannot preserve merged-wave/primitive semantics.

The selection error was in
`crates/remu-gpu-metal/src/presenter/shader/program.rs::ngg_mesh_stage_for_draw`: it recognized the
architectural NGG contract but returned `false` whenever the shader used no general LDS. The live
program proves general LDS is not a valid discriminator: pass-through NGG can publish connectivity
and consume merged-wave values without staging records through LDS.

Commit `6a082b1` removes that LDS gate. Proven memoryless fullscreen covers are still selected
first by `ngg_fullscreen_cover_draw` / `procedural_rectangle_stage_for_draw`; supported
non-procedural triangle/strip/point draws use the mesh ABI.

## First validation of the stage correction

The 75-second run at `20260804T110724Z` compiled the formerly vertex-lowered families as mesh
stages. Examples include:

- `es=0x800bd30000 ps=0x800bd40000 stage=mesh`
- `es=0x800bff0000 ps=0x800b220000 stage=mesh`
- `es=0x800d690000 ps=0x800da30000 stage=mesh`
- later large scene programs at `0x8091430000`, `0x8091490000`, and `0x8091550000`

`[REMu][METAL][NGG-MESH]` records show ordered source geometry such as the indexed quad
`[0,1,2,2,1,3]`, packed unique vertices, primitive counts, and mesh group counts. No Metal pipeline
failed in that run.

The change dramatically enlarged the cold set of mesh pipeline compilations. By 80 seconds the
game had queued the title/scene workload but had not drained it; 13 changed captures reached only
the pre-menu hallway. The run ended at the external watchdog. A later normal-write-watch run was
manually stopped at 40 seconds. Therefore the stage correction is mechanically active but its
pixel result is unverified.

## The next semantic gap: parameter exports are still missing

Even on the mesh path, many real 3D pairs emit `[REMu][METAL][INTERFACE-FILL]`: the pixel shader
interpolates one to five attributes while the decoded export program reports no conventional
parameter exports. The current backend declares the required Metal interface and leaves the
zero-initialized `RemuVsOut` fields as deterministic zero. A source comment explicitly describes
this as temporary until NGG staging is lowered. This is a genuine unfinished seam and can produce
black or constant-texture pixels.

Do not blindly synthesize UVs for real 3D. Determine from SDK-defined NGG semantics where these
attributes come from: general LDS staging, primitive attributes, compacted vertex export records,
or the pixel-input control mapping. Some logged `target=0x0` draws can be depth/internal passes,
so an interface-fill log alone is not proof that a particular draw should color the title.

Useful source locations:

- `crates/remu-gpu-metal/src/presenter/shader.rs` — interface matching and mesh compile options
- `crates/remu-gpu-metal/src/presenter/shader/program.rs` — NGG recognition/stage selection
- `crates/remu-gpu-metal/src/presenter/decode/pipeline.rs` — subgroup/register decode and key
- `crates/remu-gpu-metal/src/presenter/execute.rs` — meshlet preparation and dispatch
- `crates/remu-shader-metal/src/compiler/module.rs` — mesh function ABI/output arrays
- `crates/remu-shader-metal/src/compiler/driver.rs` — `S_SENDMSG` and NGG peepholes
- `crates/remu-shader-metal/src/compiler/export.rs` — primitive, position, and PARAM export

## Additional observations that may matter later

- The first corrected-stage run logged an image fallback at pixel PC `0xa0c` for an all-zero image
  descriptor (`GFX10 format 0/0`). This may be a legitimately null optional binding; establish
  provenance before treating it as a scene blocker.
- Large first-use MSL pipelines took up to several seconds each (`0x8091550000` was roughly four
  seconds in the pipeline phase), so a cold validation gate needs a generous timeout or a reusable
  pipeline cache. Do not confuse queued compilation with a deadlock.
- `ngg_export_uses_general_lds` is now test-only. Native and Rosetta-target workspace checks pass
  without the former unused-function warning.
- The thread-affinity work is preserved in runtime commit `8d899ba`. A user's external tip noted
  that games may expect pinned cores; validate it, but no current evidence ties title-menu
  blackness to affinity.
