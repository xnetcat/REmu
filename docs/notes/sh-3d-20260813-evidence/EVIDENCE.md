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
