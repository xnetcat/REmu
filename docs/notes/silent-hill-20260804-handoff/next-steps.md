# Exact resume plan

## 1. Re-establish the checkpoint without changing behavior

1. Confirm no game process remains.
2. Read this handoff plus the canonical master plan, implementation notes, and `docs/status.md`.
3. Inspect checkpoint commits `6a082b1`, `8d899ba`, `ed6e32a`, and `2345cb6`; do not discard or
   rewrite the preserved investigation state casually.
4. Read `known-test-failures.md`, then reproduce/narrow its serial failures before treating broad
   test gates as green.
5. Re-run the focused shader tests and both-target compile checks before new behavior changes.

## 2. Validate the LDS-free mesh decision to the real title gate

Use automatic dialog dismissal and Cross pulses. Keep normal CPU write watches for the qualifying
run. Allow enough time for the cold first-use mesh pipeline set; the 75-second run was too short.
Capture changed guest frames and a synchronized window screenshot at the actual menu.

Success is not the hallway. It is all of the following:

- the actual New Game / Continue / Settings title UI;
- a visibly non-black, spatially correct scene behind that UI;
- stable animation/presentation cadence rather than a single stale frame;
- no Metal pipeline failures, host panic, guest fatal, or spontaneous crash.

If the corrected mesh selection renders the scene, retain it and add a regression from the
captured LDS-free NGG program shape. If it does not, keep the evidence but do not call the stage
change complete.

## 3. Finish real NGG vertex/primitive attribute lowering

**Phase 1E follow-up (`5853fa7` on `main`):** `INTERFACE-FILL` zero-fill is deleted. Unmatched
`SPI_PS_INPUT_CNTL` slots without EXP PARAM and without general LDS take SDK `UseDefault0`
(identity PARAM offsets from null-PS `createInterpolantMapping` no longer refuse the pair).
General LDS + missing EXP PARAM remains a loud refuse — attribute-ring producer not lowered.
Silent Hill missing-PARAM Gen5 fails dropped **31→0**; still **not** a title-UI/3D closure
(“success is not the hallway”).

Remaining diagnostic sequence for real attribute producers (LDS / primitive-attr when EXP is
absent):

1. Dump mesh MSL for LDS-bearing ES halves paired with VInterp PS and compare with pass-through
   PRIM+POS-only exports already covered by `UseDefault0`.
2. Capture actual mesh vertex positions and attributes on the title-scene color-producing passes;
   separate depth-only/internal passes from final G-buffer/color passes.
3. Trace whether parameters are staged through LDS attribute rings or primitive attributes, then
   implement that architectural data flow in `remu-shader-metal` (not RemuVsOut invent).
4. Keep a loud, specific refusal for genuinely unsupported forms; do not fabricate UVs/colors or
   restore INTERFACE-FILL.
5. Add decoder/compiler/presenter tests derived from the generic instruction/register contract.

## 4. Continue Silent Hill beyond the title

After the title background is correct:

1. Select New Game automatically.
2. Capture and inspect the opening cinematic.
3. Reach interactive gameplay and exercise movement/input.
4. Sustain play long enough to cover the five historical stale-storage fault sites and scene
   transitions.
5. Verify audio/video sync, frame progression, non-black 3D, stable telemetry, and clean shutdown.

Do not claim Silent Hill working from a splash, setup page, hallway, title UI alone, timeout, or
absence of unimplemented NIDs.

## 5. Run the compatibility matrix after each major GPU/shader change

Use `REMU_AUTO_DISMISS_DIALOGS=1` (the harness defaults to it) and Cross every few seconds where
appropriate. Capture screenshots and notes only; keep fixes focused on Silent Hill until its gate
passes.

Matrix:

- Silent Hill (primary gate)
- Octopath Traveler II
- Superliminal
- Demon's Souls
- Dreaming Sarah
- Void Terrarium
- Oregon Trail
- any other games listed by the repository's matrix script/notes

For every title record launch outcome, last visible frame, FPS/progress trend, shader failures,
unimplemented imports, crash signature, and exact run directory. A black/stale screen is a failure
even if the process remains alive.

## 6. Poppy Playtime only after Silent Hill

Poppy's prior run was black at 0 FPS. Once Silent Hill's real 3D path works, bring Poppy back as the
next implementation target, progressing through menu to gameplay and soak-testing n>=5 if the
canonical gate still requires it. Do not describe the old run as healthy evidence.

## 7. Finish the canonical master plan and clean the checkpoint

Reconcile every historical `partial-with-seam`, `loud-refused`, and newly exposed interface
fallback against the master plan. The owner's current requirement is no stubs, seams, mocks,
deferred work, or leftover investigation code in the finished result.

Before a compatibility-complete release/follow-up commit:

- remove investigation-only diagnostics, broad tracing, example binaries, and stale screenshots;
- keep useful opt-in diagnostics only when tested and documented;
- correct old compatibility claims using the real title gate;
- run focused tests, then workspace tests serially, both-target checks, clippy/warning gates, and
  all required game gates;
- ensure FFmpeg cold/warm builds and app packaging work for both architectures;
- update canonical notes and per-title evidence;
- review the entire checkpoint series and any follow-up diff for title-keyed behavior, fabricated
  success, unsafe pointers, and silent fallbacks;
- commit follow-up work intentionally only after its implementation and claimed gates are
  genuinely done.
