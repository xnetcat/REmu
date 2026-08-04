# REmu / Silent Hill handoff — stopped 2026-08-04

## Honest current state

Work was stopped at the owner's request. No REmu or game process remains running.

The stopped state has been preserved on `main` as a logical checkpoint series rather than one
large commit:

- `6a082b1` — GPU, shader, texture, and NGG execution work
- `8d899ba` — runtime services, synchronization, media, and guest support
- `ed6e32a` — stall-inspection tooling and analysis examples
- `2345cb6` — historical Silent Hill bring-up evidence
- the commit containing this file — durable handoff notes, copied logs, captures, and shader dumps

The series starts after `30bd087`. It records an incomplete compatibility investigation; commit
boundaries are not claims that Silent Hill or the master plan is complete.

What is proven:

- The Rosetta build no longer falls back to a libav stub. Pinned static FFmpeg 8.1.2 is built for
  both macOS architectures and linked into the runtime. App packaging calls the real repository
  script, `scripts/build-macos-app.sh`.
- Silent Hill boots, renders startup/setup UI, accepts automatic Cross input, and reaches its title
  menu without the earlier startup crashes.
- The brightness page is grayscale now; the red calibration page defect is fixed.
- The pre-menu ruined hallway/content-warning backdrop renders, but **it is not the title menu's
  real 3D scene and must not be used as evidence that 3D rendering is fixed**.
- At the actual New Game / Continue / Settings title menu, the UI renders and the intended scene
  behind it is still entirely black. Cinematic/gameplay rendering and crash-free sustained play
  are therefore not proven.
- Poppy Playtime's one permitted observation was unhealthy: black screen, 0 FPS. It has not been
  relaunched and remains unfixed.

What was in progress at the stop:

- A captured Silent Hill export program contained the GFX10 fused-NGG contract (`S_SENDMSG 9`,
  primitive `EXP target 20`, and position `EXP target 12`) but was compiled as an ordinary Metal
  vertex shader because it used no general LDS.
- Commit `6a082b1` makes every supported non-procedural fused-NGG draw use the Metal mesh ABI even
  when it is LDS-free. The focused `remu-gpu-metal` shader suite passes 34/34.
- A 75-second cold game probe showed those programs compiling and dispatching as mesh stages with
  no pipeline failures or host crash. It captured only startup and the pre-menu hallway before the
  external watchdog; the large first-use scene shader set was still compiling. This does **not**
  prove the title background is fixed.
- A 220-second follow-up with normal CPU write watches was manually interrupted after 40 seconds
  because the owner asked to stop. Its exit 130 is the intentional Ctrl-C, not a guest crash. It
  had captured startup/autosave frames but had not reached the title-menu proof point.

The compile gates are green, but the broader serial test runs are not. See
`known-test-failures.md` for the exact failures and the runtime-suite hang that was terminated when
work stopped.

## Important correction to the older progress note

`docs/progress/silent-hill-20260804/observations.md` previously called the ruined hallway a “fully
rendered 3D” scene. The owner clarified that this is pre-menu presentation artwork/content and is
not the actual scene behind the New Game menu. The compatibility gate remains the true title
background, then cinematic and interactive gameplay. This handoff and the correction added to the
older note are authoritative.

## Master-plan status

The canonical master plan is **not complete**. The historical implementation log still records
multiple `partial-with-seam` and `loud-refused` items (C1/C2/C4, P3, S2 production encoder and
open-batch work, Q shader closure, pointer migration, codec tails, and compatibility gates). The
present worktree adds substantial implementation beyond that checkpoint, but it has not completed
the definition of done: all shaders do not yet compile with correct pixels, all titles have not
passed the required interaction gates, and the latest major NGG change has not had an all-game
matrix.

The canonical sources remain:

- `/Users/xnetcat/Documents/REmu-research/architecture-audit-20260802/MASTER-PLAN.md`
- `/Users/xnetcat/Documents/REmu-research/implementation-notes/NOTES.md`
- `/Users/xnetcat/Documents/REmu/docs/status.md`

## Resume point

Read these files in order:

1. `technical-findings.md`
2. `experiments.md`
3. `workspace-state.md`
4. `known-test-failures.md`
5. `next-steps.md`
6. `artifact-index.md`

Do not resume from the media/video hypothesis or publish broad sampled aliases. The strongest live
lead is the incomplete fused-NGG export/interface lowering described in `technical-findings.md`.
