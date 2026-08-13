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
