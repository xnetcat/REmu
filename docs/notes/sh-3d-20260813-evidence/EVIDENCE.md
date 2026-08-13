# First coherent title-scene 3D geometry — 2026-08-13

`title-basepass-30612.png` is the in-progress BasePass color target at draw 30612 of run
`20260813T051318Z-PPSA10112-app` (clean tree at `ed497d47`; scanout-writer diagnostic
`auto1080`, `MIN_DRAW=30000`, slot 149). It shows the NEW GAME rooftop cityscape as
spatially coherent 3D: buildings with lit windows, streets, correct perspective. Dark is
expected — the dump samples ~2% of the frame's ~8.6k BasePass draws, before deferred
lighting; the growth series in the same run (slots 000→149, nonzero sampled pixels
0→48/4156) shows monotone accumulation with no horizontal striping and no collapsed
one-pixel slivers.

Contrast: every pre-`6f19764e` capture of this scene showed either black or green/magenta
horizontal bands attributed to "thousands of one-pixel horizontal triangle fragments"
(docs/status.md 2026-08-10 entry, sh-present handoff). The mesh export contract change
(`6f19764e`: publication at the invocation index, degenerate default fill, gs_tg_info seed)
is the only geometry-affecting change between those captures and this one.

Not yet closed: this is a mid-chain diagnostic readback, not a presented frame. The
presented NEW GAME screen still hinges on the presentation-cadence work (docs/status.md
2026-08-13 entry: PSO recreation per run, executor throughput, intermittent
RenderThread-starve wedge).

## Stage-by-stage verdict (run 20260813T052050Z, chain-filtered writer dumps)

Filter `auto1080:0x800e730000,0x800bf40000,0x800a690000,0x800ae70000,0x800da30000`,
`MIN_DRAW=25000`, 60 slots spanning draws 26624-44713 (title screen, ~10 frames):

| Stage | Observation |
| --- | --- |
| BasePass materials (8091xx) | Coherent rooftop cityscape (previous section) — geometry FIXED |
| Deferred direct light `800a69` | All-zero in every captured frame (0/4156 sampled) |
| Combine `800ae7` | All-zero |
| Tonemap `800bf4` | Mostly zero; 1497/4156 during a white flash |
| Composite `800e73` | ~800/4156 recurring (partial content from non-scene inputs) |
| Scanout cover `800da3` → efc | `title-scanout-striped-27756.png` — the green/magenta stripes |

The two remaining corruption stages line up with the split-wave64 audit's ranked hazards
(wave64-split-audit-20260812.md): the zero lighting with the wave64 pixel mask classes
(S_BCNT1_I32_B64 / readlane selector; phase 3 of the plan), and the striping with the
compute compaction off-by-32 (V_MBCNT on a half-local mask; phases 1-2, shaders
0x800c7e / 0x80095b). Next implementation step is Phase 1: lane-doubled compute for
single-wave (64-thread) groups.

## Wave64 hazard census (live, 2026-08-13 runs)

`[REMu][WAVE64-CENSUS]` inventory of the title's wave64 compute programs (once per address):

- FFT/bloom family `0x800959/0x80095b/0x800973`: scc_branch 7-8, exec_branch 6-24,
  barriers 80-160, **mbcnt_mask=0, bcnt1_b64=0** — the split-audit §1.3 compaction
  hypothesis does NOT apply to these; their hazard classes are half-local SCC/EXEC
  branch decisions and whole-group barriers under 16 guest waves (the Metal-illegal
  class), plus the epoch-desync deadlock exposure of the pair rendezvous.
- Shadow-mask kernels `0x800838/0x80085d`: mask64_ops=26 each with exec_branch=2 —
  B64 mask arithmetic is half-local today; a wrong mask here zeroes downstream
  lighting, which matches the all-zero `800a69` output.
- `0x800a13`: readlane=67 — the 4-KiB readlane table + its two whole-group barriers
  per site; illegal in multi-wave groups when reached inside structured control flow.
- No wave64 compute program in the census uses V_MBCNT on a mask SGPR or
  S_BCNT1_I32_B64, so lane-doubling Phase 1/2 should be validated against the branch
  and barrier classes first.

Lane-doubling (split-audit §3b) remains the strategy; this census fixes the
validation targets: shadow-mask kernels first (small, mask64-heavy, feeds lighting),
then the FFT family.

## Live attribution: the stripes ARE the FFT bloom (2026-08-13, run 20260813T074744Z)

`title-menu-clean-nobloom.png`: the presented NEW GAME screen with a UE config overlay
(`REMU_OVERLAY_DIR` ps5engine.ini `[SystemSettings]`) setting `r.BloomQuality=0`
(+`r.PostProcessAAQuality=0`, motion blur/lens flare/fringe off). The FFT family
(`0x80095b` et al.) never dispatches (zero log references), and the title screen presents
with the logo, menu, prompts and copyright pixel-clean — zero green/magenta striping.
Contrast the user's live capture of the same screen without the overlay (full-frame
striping over everything but the logo). This closes the striping attribution: the split
wave64 FFT bloom convolution is the sole stripe source in the presented title frame.

The background is black rather than the reference's foggy rooftop: deferred lighting
still outputs zero (see the stage table above). The overlay is a validation instrument
and an interim visual improvement, not a fix — the renderer fix remains the lane-doubled
wave64 lowering, after which bloom/TAA get re-enabled.

## Negative result: shadows/AO/SSR/fog off does NOT light the scene (run 20260813T075925Z)

Adding `r.ShadowQuality=0, r.CapsuleShadows=0, r.DistanceFieldShadowing=0,
r.DistanceFieldAO=0, r.AmbientOcclusionLevels=0, r.SSR.Quality=0, r.SSGI.Quality=0,
r.VolumetricFog=0, r.VolumetricCloud=0, r.LightShaftQuality=0` to the overlay keeps the
menu pixel-clean but the 3D background stays exactly black
(`title-menu-noshadows-still-black.png`, gamma-boosted 0.45). The zero-lighting cause is
therefore inside the deferred lighting pixel shaders (800a69 direct / 8009c4 sky) or their
non-occlusion inputs — the split-wave64 audit's pixel mask classes — not the occlusion
kernels they sample.

## Lighting zero root cause: lost resident, not shader math (run 20260813T080939Z)

Targeted dump of direct-light draw 32974 (es=800bff/ps=800a69, fullscreen tri, bloom-only
overlay): all 746,496 covered pixels export exactly 0.0 (pixel_outputs min=max=mean=0), no
non-finite values, target ends (0,0,0,α). The shader's screen-space input at 0x801e8f0000
(sampled 1920x1080 RGBA8, tile 27) has **resident=None, writer=None** — served from guest
RAM, which is zero — while overlap_writers records a GPU draw that wrote the same address
as a BGRA8 1080p target (its own sampled input an 80x40 BC7). Light x attenuation(0) = 0.
The zero-lighting class is therefore lost sampled residents falling back to unpublished
guest RAM (the B15 residency/alias-lifetime class), NOT split-wave64 pixel math. The
shadows-off permutation stayed black too — consistent with the same lost-resident fallback
on that permutation's other utility inputs.

Discriminator queued: dump the 0x801e8f0000 writer at write time (nonzero at write =>
residency loss confirmed; zero at write => fault moves up the writer's chain).

## Residency loss confirmed (run 20260813T082409Z, REMU_DUMP_SHADOW_CONSUMER=1)

The attenuation writer (draw 24427, ps=0x8007c5, 1296-index grid, BGRA8 1080p target
0x8026bb0000) produces a fully-populated image at write time: every pixel nonzero
(magnitude_nz=2,073,600), first pixel [ff,ff,ff,ff], range 0..1, fingerprint
def1793351704b43. Combined with the light-pass record (its sample of the same-class buffer:
resident=None → zero guest RAM), the zero-lighting mechanism is closed: the GPU-written
attenuation resident is retired between writer and consumer and the sampled upload serves
unpublished guest RAM. Fix territory: sampled-cache upload must not serve zero RAM for
addresses whose bytes are GPU-owned (PERF-TEXTURE-OWNED spans) — retain, rebind
(BGRA<->RGBA view), or rematerialize instead. The existing rematerialise path reports 0
attempts at these samples. This is the B15 residency/alias-lifetime class; no wave64 work
is needed for lighting.

Toward the reference NEW GAME screen: menu/UI pixel-clean (bloom off), geometry coherent,
lighting = this one residency fix, stripes = wave64 FFT (re-enable bloom after
lane-doubling).

## Lighting reclaim iteration trail (2026-08-13, commits 2ee64123..b817679b)

Six instrumented iterations on the light-attenuation lifetime, each eliminating one gate:
1. Sticky `gpu_content_executed` + read-only clean check — misses persisted
   (`executed=true cpu_clean=false`: generations drift on renewal).
2. Fingerprint verification in reclaim_for_bind — still zero reclaims.
3. RECLAIM-BLOCKED probe: named `acquire-invalidated` as a gate on 0x8041db0000
   (RGBA16F 1080p, every frame).
4. ACQUIRE fingerprint verification — still zero reclaims.
5. ENSURE-REPLACE probe: at removal, `acquire_invalidated=false, retirement=None` —
   the executed arm itself declines.
6. Watch-oracle (`is_dirty` primary, fingerprint fallback) — still zero reclaims and
   ENSURE-REPLACE still fires: the watch is genuinely dirty AND content genuinely
   changed. The guest CPU really writes these pages between GPU uses (transient-pool
   reuse), so no local predicate can decide whether Metal or RAM holds the logical
   resource's current bytes.

Conclusion: content identity across transient-pool reuse requires the B15
versioned-ownership rewrite (owned range sets + Stale-as-versioning per
MASTER-REWORK-PLAN B15), with the mapping/backing identity
(GuestSharedMappingIdentity) distinguishing "same allocation reused for the same
logical target" from "reused for something else". All six iterations' instruments
(SAMPLED-MISS with self-describing state, ENSURE-REPLACE, RECLAIM-BLOCKED,
reclaim_executed counter) are in tree and give B15 its acceptance signals: at the SH
title, reclaim_executed > 0, SAMPLED-MISS = 0, 800a69 exports nonzero, rooftop visible.
All changes are conservative (verification-gated) and the full suite stays green.

## Final probe round: the families separate (runs 20260813T0942..0955)

ACQUIRE-REJECT sampling shows the per-frame-replaced RGBA16Float 1080p targets carry REAL
guest-RAM data (zero_head=0/64, watch genuinely dirty) — these are legitimately republished
ping-pong buffers (GPU write → writeback publish → RAM → re-upload), a slow-but-correct
cycle, NOT the black-lighting chain. The black chain remains the sampled-side class
(attenuation RGBA8/BGRA8, resident=None, RAM zeros). At least three distinct buffer
lifecycles interleave at the same sizes; point probes keep landing on the wrong family.
This is the strongest confirmation yet that content authority needs B15's per-range version
ledger rather than further predicate patches. All probes in tree are gated and cheap; they
are B15's live acceptance instrumentation.

## Residency fixes take hold (runs 20260813T1009..1027, commits cc4ccdb1..a1f62f37)

After the reclaim + resolve-stage-oracle changes: SAMPLED-MISS residents now show
live=true serves_reads=true gpu_written=true (the executed content SURVIVES — previously
stripped to nothing every frame), the residual miss classes changed composition (first
enumerated: BC3 block-view pairs, sampled=BC3_RGBA/7800x7800/tile9 vs
resident=RGBA32Uint/1950x1950/tile9 — the compressed-storage-view compatibility gap,
audit finding #10 territory), and the direct-light dump moved for the first time:
pixel_outputs covered=746496 nonzero=513280 (alpha writes land; previously zero), RGB
radiance still 0.0 — one remaining zeroed input upstream (enumerate via the now-healthy
SAMPLED-MISS pairings and the same 800a69 dump).

Note: guest crashes (the UE-watchdog family) now occur most runs between 46-350s and are
the main iteration obstacle; the title menu presents continuously when runs survive.

## First presented title frame with 3D scene content (run 20260813T103355Z, ce0f0cb3)

`title-scene-first-presented-boosted.png` (gamma 0.45 boost of the raw efc capture at
t=367s): the reference composition — foggy bright region right-of-center, rooftop
structure lower-left, menu/logo on top — is present in a PRESENTED frame for the first
time. The compatible-arm tentative acceptance + resolve-stage fingerprint oracle
(cc4ccdb1..ce0f0cb3) connected the lighting chain through to the screen. The run also
survived its full 425s window (first full survival of the day) and presented 29 distinct
frames. Remaining darkness: the scene renders several stops too dark; the last known
zeroed input is the 1024x512 R16Float sky-transmittance LUT (0x8051d1-class) — zero
transmittance suppresses sky/ambient radiance. Next: serve that LUT (same residency
class) and re-measure exposure.

## Darkness suspect: the transmittance LUT rides the mip-promotion path

The 1024x512 R16Float sky-transmittance LUT (0x803fb4/0x8051d1 per run) is sampled with
explicit LODs and REmu promotes it through "ordered mip promotion levels 1->5"
(gpu_current=false for 1-4): if the base level is zero at promotion time, every
synthesized mip is zero — audit finding #12 (authored-mip synthesis) meeting the same
residency timing. Discriminating dump prepared but not yet landed (needs
REMU_DUMP_COMPUTE_TEXTURE_DIR + EXTENT=1024x512; runs currently die early to the
intermittent guest-crash family before the LUT phase). If the compute writer's output is
nonzero, extend the residency serving to the mip-promotion source; if zero, the writer's
WAVE64-CENSUS entry names the hazard.

## Stability note (runs 20260813T1043..1113)

The scene-behind-menu visibility is not yet stable run-to-run: the ce0f0cb3 survivor
showed the composition at t=367s; two 9ac36970 survivors presented the menu continuously
(8 live frames, t=261-407) but with the fog region at exact (0,0,0) in sampled captures.
Candidates: transient fade windows vs residency race nondeterminism vs the 9ac36970
promotion widening; discrimination needs either the intermittent guest-crash fix (task 9,
UE-watchdog family — now the top iteration gate) or continuous interactive observation
rather than 8-10s capture sampling. All lighting-chain commits remain conservative
(verification-gated) and suite-green.

## 2026-08-13 (later): black-background root cause — CLOSED

Chain walked with post-draw images: composite pass wrote black into the pre-scanout LDR buffer
while upstream scene targets were 30-35% lit. Instrumented broadly (async debug log
`REMU_DEBUG_LOG=1`; journal range dump at exit). Eliminated: eye adaptation, 64KB base offset
(SAMPLED-NEAR probe, 0 hits), DMA/global-writeback/memcpy writers (journal, 0 entries).

Root cause per `[SAMPLED-MISS]`: layer rendered **BGRA8**, sampled **RGBA8**, resident
`gpu_written=false executed=true serves_reads=true` — the compatible-arm *selection group*
lacked the executed-content colour-view clause (it existed only in the serve-family gate), so
the sampler used guest RAM (black, never written back). Fix in upload.rs selection group.

Verification: SAMPLED-MISS 40+→0; content-warning screen now presents its full 3D corridor
background (previously text-over-black). New blockers: crash family kills runs pre-menu
(SIGSEGV null pc=0x802816015 @40s; SIGBUS texture read 0x805b2a0000 @70s, no journal writer —
watch materialization suspect); font-atlas glyphs partially missing.

## W3 wedge capture notes (2026-08-13 late)
- Wedge reproduces in the in-game load window; silent variant exists (no UE fatal — presents
  stop, process idles to external shutdown). Presenter provably idle (queues empty) during it.
- RenderThread's last acts in fatal variants: sceAvPlayerStop → STOP delivered → sceAvPlayerClose
  never returns. Lock-free imports (RTC, event flags) keep flowing; state-locked ones stop.
- Host-side `sample` at wedge time is the decisive instrument; sample `pgrep -x remu` — the
  `-f` pattern matches the run-game.sh bash wrapper instead.

## NEW GAME screen reached (run 20260813T134345Z)
Parked at the title menu t=198-353s: logo, menu entries, prompts, copyright all correct and
animating. Background is the static-glitch overlay at full strength; the rooftop scene behind
it is hidden. Lead: PS 0x809ca50000 pc=0x84 ImageSampleB refused 51x — evaluated SRD is two
consecutive buffer V#s (descriptor-table head), i.e. dynamically-indexed bindless T# fetch that
static evaluation resolves to the wrong entry. The same run wedged silently t=23-175 (W3
variant that RECOVERS; UE's 120s watchdog converts the longer ones into the pc=0x809044770
crash-handler fatal). Persistent metallib + render-pipeline archives exist under
~/Library/Caches/REMu/metal; warm runs shrink the storm.
