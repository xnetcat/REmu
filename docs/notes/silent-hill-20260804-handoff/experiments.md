# Experiment log and rejected paths

This records both successful work and negative experiments so the next session does not repeat
them. Timestamps are run-directory timestamps in Europe/Warsaw on 2026-08-04.

## Build/runtime bring-up already completed

1. Assessed the original `libav bridge stubbed` warning and missing
   `scripts/lib/build-macos-app.sh` failure separately.
2. Added a build-owned, checksum-pinned static FFmpeg 8.1.2 dependency for both
   `aarch64-apple-darwin` and `x86_64-apple-darwin`; removed the functional stub fallback; linked
   AvPlayer/Videodec2 to the in-process libraries.
3. Fixed the Rosetta packager path to `scripts/build-macos-app.sh`.
4. Fixed prefixed `mov r64, fs:[0]` TLS rewriting under Rosetta.
5. Fixed stale copied-global publication following a recycled VA onto a new physical allocation.
6. Bounded dirty-content fingerprint retries so a continuously written 15.9 MiB texture cannot
   starve the presenter forever.
7. Fixed a condition timeout/notify double-winner race.
8. Added/extended indirect NGG meshlet preparation, BC1 comparison/depth uploads, exact mode-5 and
   mode-27 mip handling, DS64 pairs, half-float FMA literal decoding, output/culling/format work,
   shared-event ordering, scratch/global shader work, and thread-affinity support in the current
   dirty tree.

These items are not all independently final-gated. The existing historical note has the detailed
test provenance: `docs/progress/silent-hill-20260804/observations.md`.

## Visual progression and the corrected gate

- Startup logos, autosave UI, content-warning UI, and the grayscale brightness page render.
- Cross-only input advances the brightness page.
- Silent Hill can reach the real title menu and New Game flow.
- The pre-menu ruined hallway is **not** the real title scene.
- At the actual title menu, the 3D background is black. That is the unresolved visual gate.

## Culling probe

Runs/screenshots around `20260804T094340Z` exercised culling changes and reached a steady menu.
They did not restore the actual title background. Preserved pictures are under the older progress
directory; representative hallway/menu evidence is copied into this handoff.

Conclusion: do not resume by toggling culling globally.

## Merged-count probe

Runs/screenshots around `20260804T095803Z` tested merged export/count handling. Multiple stages were
captured, but the true title background remained black.

Conclusion: count handling alone is not the missing title scene.

## Media/video-chain probe

### `20260804T103903Z`

Captured the black actual title menu while tracing the title-video path. Independent decode and
conversion checks showed healthy NV12 planes and a correct colorful BGRA conversion.

### `20260804T104431Z`

Initial video-chain instrumentation reported compute consumers with `root=false`. The match was
based on broad physical overlap and was later proven false: a different layered allocation merely
covered the same heap range.

### `20260804T104752Z`, `20260804T104941Z`, `20260804T105032Z`

Tried publishing internal `SampledAlias` writebacks (`REMU_PUBLISH_INTERNAL_WRITEBACKS=1` and a
temporary equivalent policy patch). Outcomes included early SIGSEGVs and, with normal watches, a
readback flood of hundreds of MiB, 0 FPS, and a game-thread timeout. The policy patch and tests
were reverted completely.

### `20260804T105540Z` / `20260804T105913Z`

Narrowed chain matching to exact descriptor identity or proven shared storage. No legitimate
downstream consumer of the converted title movie appeared. Diagnostics now keep root and chain
log counters separate so root traffic cannot suppress the first chain records.

Conclusion: decoder, NV12 planes, and conversion are healthy; the converted movie is not the
missing 3D background; broad physical alias publication is unsafe.

## Targeted NGG shader capture

### `20260804T110142Z`

Command shape:

```sh
REMU_GAME_TIMEOUT_SECONDS=100 \
REMU_AUTO_CROSS=8 \
REMU_AUTO_CROSS_REPEAT_MS=5000 \
REMU_AUTO_CROSS_HOLD_MS=250 \
REMU_DISABLE_CPU_WRITE_WATCH=1 \
REMU_DUMP_MSL_DIR=/tmp/remu-silent-ngg-early \
REMU_DUMP_MSL_PS=800da30000 \
./scripts/run-game.sh '/Volumes/Untitled/sharpemu/games/SILENT HILL The Short Message/PPSA10112-app/eboot.bin'
```

The run ended at the external watchdog. Its dump proved that `0x8066760000` has allocation,
primitive, and position exports yet was compiled as `stage=vertex`. The generated vertex MSL had
no primitive-stage behavior and no `param0` assignment. Raw IR/MSL are saved under
`shader-dumps/`.

## LDS-free NGG stage-selection change

Changed `ngg_mesh_stage_for_draw` so supported fused NGG uses the mesh ABI whether or not it uses
general LDS. Memoryless covers remain intercepted by the procedural-cover path. Updated focused
tests.

Validation command:

```sh
cargo test -p remu-gpu-metal --lib presenter::tests::shader --no-default-features
```

Result: 34 passed, 0 failed. `git diff --check` also passed at that point.

The earlier diagnostic suite also passed:

```text
cargo test -p remu-gpu-metal --lib diagnostics --no-default-features
19 passed

cargo test -p remu-gpu-metal --lib \
  residency_trigger_keeps_the_debt_until_a_guest_read_asks_for_the_pixels \
  --no-default-features
1 passed
```

## Corrected-stage game probes

### `20260804T110724Z` — 75-second cold probe

```sh
REMU_GAME_TIMEOUT_SECONDS=75 \
REMU_AUTO_CROSS=8 \
REMU_AUTO_CROSS_REPEAT_MS=4500 \
REMU_AUTO_CROSS_HOLD_MS=250 \
REMU_CAPTURE_FRAMES=1 \
REMU_DISABLE_CPU_WRITE_WATCH=1 \
./scripts/run-game.sh '/Volumes/Untitled/sharpemu/games/SILENT HILL The Short Message/PPSA10112-app/eboot.bin'
```

Observed:

- former vertex programs compiled as `stage=mesh`;
- ordered `[REMu][METAL][NGG-MESH]` records were emitted;
- 13 changed 1920x1080 captures, reaching the pre-menu hallway. Twelve are in the durable contact
  sheet; one temporary later hallway-period PPM was accidentally overwritten while arranging the
  contact sheet and is documented in `artifact-index.md`;
- 98/38 pipeline readiness at the end, three graphics compilations active, no failed pipeline;
- no host panic or guest fatal in the saved log;
- external watchdog status 124 after 80 seconds, while the queue was waiting for further scene
  pipeline compilation.

This is activation evidence, not visual proof of the missing title scene.

### `20260804T111022Z` — normal-watch longer probe, stopped by owner

```sh
REMU_GAME_TIMEOUT_SECONDS=220 \
REMU_AUTO_CROSS=8 \
REMU_AUTO_CROSS_REPEAT_MS=5000 \
REMU_AUTO_CROSS_HOLD_MS=250 \
REMU_CAPTURE_FRAMES=1 \
./scripts/run-game.sh '/Volumes/Untitled/sharpemu/games/SILENT HILL The Short Message/PPSA10112-app/eboot.bin'
```

The owner requested a stop after 40 seconds. Ctrl-C produced exit status 130 and the harness label
`guest-failure`; this is an orchestration stop, not a spontaneous game crash. Eight frames reached
logos/autosave. The true title scene was not yet observed.

## Compatibility observations outside Silent Hill

No title-specific fixes were made from these observations:

- Poppy Playtime: the one launch was black at 0 FPS and unhealthy. It has not been relaunched.
- Octopath Traveler II: earlier observation reached startup/setup/dialogue/3D at about 7–10 FPS,
  but the owner later hit a large stall and reported that actual 3D commonly turns black. Preserve
  `inputs/octopath-stall-log.txt`; do not claim Octopath healthy.
- Superliminal: one hallway frame rendered, then the title stayed static at 0 FPS with no further
  submissions.
- Demon's Souls: aborted in its own audio assertion because `sceAcmBatchStartBuffers` is not yet
  fully served with real processing.
- Dreaming Sarah: missing packaged `shaders/*.vert` files and a startup segfault in the observed
  run.
- Void Terrarium: startup/import/thread-initialization segfault in the observed run.
- Oregon Trail: duplicate `libc.prx` module paths prevented launch.

An all-game matrix was **not** run after the new LDS-free NGG stage change because work was stopped.
