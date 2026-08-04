# Artifact index

All durable handoff material is rooted at:

`/Users/xnetcat/Documents/REMu/docs/notes/silent-hill-20260804-handoff`

The checkpoint contains 101 files (about 28 MiB). Raw guest logs and generated Metal shader dumps
are copied verbatim, including their original trailing whitespace; `git diff --check` reports those
artifact lines, not whitespace introduced in the handwritten Markdown notes or source changes.

## Written notes

- `README.md` — executive state, honest compatibility result, master-plan status, resume order
- `technical-findings.md` — media/alias/NGG evidence and current root-cause lead
- `experiments.md` — chronological trials, commands, outcomes, and rejected hypotheses
- `workspace-state.md` — branch/diff/test state and key dirty areas
- `known-test-failures.md` — exact serial-suite failures, invalid parallel run, and runtime hang
- `next-steps.md` — exact resume and completion sequence
- `artifact-index.md` — this location map

## Curated screenshots

Directory:

`/Users/xnetcat/Documents/REMu/docs/notes/silent-hill-20260804-handoff/screenshots`

- `brightness-grayscale-correct.png` — corrected grayscale brightness calibration
- `title-menu-black-background.png` — actual title UI with the intended scene black
- `video-chain-probe-black-title.png` — black-title media-chain diagnostic window
- `pre-menu-hallway-not-title-3d.png` — hallway/content-warning backdrop; explicitly not title 3D
- `first-mesh-stage-run-contact-sheet.png` — 12 of the 13 changed frames from the 75-second
  corrected-stage run. One temporary PPM was accidentally overwritten by an incorrectly ordered
  ImageMagick montage command during handoff creation; it was a later hallway-period capture and
  was excluded after its dimensions exposed the overwrite. The harness summary still records the
  original count of 13, and the remaining hallway frames preserve the same visible state.
- `interrupted-long-mesh-run-contact-sheet.png` — eight frames before the owner-requested stop
- `interrupted-long-mesh-run-last-frame.png` — final captured frame before that stop
- `octopath-observation.png` — earlier compatibility observation, not a health claim
- `superliminal-observation.png` — earlier static hallway observation, not a health claim

The larger pre-existing screenshot collection remains at:

`/Users/xnetcat/Documents/REmu/docs/progress/silent-hill-20260804/screenshots`

Every usable recent guest-frame capture is also preserved individually as PNG under:

`/Users/xnetcat/Documents/REMu/docs/notes/silent-hill-20260804-handoff/screenshots/runs`

- `20260804T103903Z/` — 13 full-screen captures plus the diagnostic 244x150 converted video target
- `20260804T104941Z/` — two captures from the sampled-alias crash probe
- `20260804T105032Z/` — seven captures from the sampled-alias flood probe
- `20260804T110724Z/` — twelve usable captures from the first mesh-stage run
- `20260804T111022Z/` — eight captures from the owner-interrupted long run

## Preserved recent logs

Directory:

`/Users/xnetcat/Documents/REmu/docs/notes/silent-hill-20260804-handoff/logs`

Each timestamp has its available `guest.log`, `summary.txt`, and `telemetry.jsonl` copied with a
timestamp prefix:

- `20260804T103903Z-*` — actual black title/video probe
- `20260804T104431Z-*` — broad-overlap video-chain trace
- `20260804T104752Z-*` — internal-writeback publication crash probe
- `20260804T104941Z-*` — sampled-alias/disabled-watch crash probe
- `20260804T105032Z-*` — sampled-alias readback-flood/timeout probe
- `20260804T105540Z-*` — narrowed exact/shared-storage chain trace
- `20260804T105913Z-*` — follow-up chain/stability trace
- `20260804T110142Z-*` — targeted NGG shader dump run
- `20260804T110724Z-*` — first LDS-free mesh-stage validation run
- `20260804T111022Z-*` — owner-interrupted normal-watch validation run

Original run directories, while the macOS temporary directory survives:

`/var/folders/hq/mpk08tmn5r1_3dqbd58mff380000gn/T/remu-runs/<timestamp>-PPSA10112-app`

The durable copied logs are the preferred references.

## Raw shader evidence

Directory:

`/Users/xnetcat/Documents/REmu/docs/notes/silent-hill-20260804-handoff/shader-dumps`

- `es-8066760000-ps-800da30000-ir.txt`
- `es-8066760000-ps-800da30000-vertex.metal`
- `es-8066760000-ps-800da30000-pixel.metal`
- `es-800c630000-ps-800da30000-ir.txt`
- `es-800c630000-ps-800da30000-vertex.metal`
- `es-800c630000-ps-800da30000-pixel.metal`
- `es-800d690000-ps-800da30000-ir.txt`
- `es-800d690000-ps-800da30000-vertex.metal`
- `es-800d690000-ps-800da30000-pixel.metal`
- `0000008066760000-000000808c510000-ir.txt` — larger later pair dump

## Original user inputs copied for continuity

Directory:

`/Users/xnetcat/Documents/REmu/docs/notes/silent-hill-20260804-handoff/inputs`

- `initial-master-plan-and-context.txt`
- `initial-direct-run-log.txt`
- `octopath-stall-log.txt`
- `thread-affinity-tip.png`

## Older durable progress material

- Main historical observation note:
  `/Users/xnetcat/Documents/REmu/docs/progress/silent-hill-20260804/observations.md`
- Older copied diagnostics:
  `/Users/xnetcat/Documents/REmu/docs/progress/silent-hill-20260804/diagnostics`
- Subsystem truth/anti-claim contract:
  `/Users/xnetcat/Documents/REmu/docs/status.md`
- Canonical master plan:
  `/Users/xnetcat/Documents/REmu-research/architecture-audit-20260802/MASTER-PLAN.md`
- Canonical long-form implementation log:
  `/Users/xnetcat/Documents/REmu-research/implementation-notes/NOTES.md`

## Key source locations at the active seam

- `/Users/xnetcat/Documents/REmu/crates/remu-gpu-metal/src/presenter/shader/program.rs`
- `/Users/xnetcat/Documents/REmu/crates/remu-gpu-metal/src/presenter/shader.rs`
- `/Users/xnetcat/Documents/REmu/crates/remu-gpu-metal/src/presenter/decode/pipeline.rs`
- `/Users/xnetcat/Documents/REmu/crates/remu-gpu-metal/src/presenter/execute.rs`
- `/Users/xnetcat/Documents/REmu/crates/remu-gpu-metal/src/presenter/diagnostics.rs`
- `/Users/xnetcat/Documents/REmu/crates/remu-gpu-metal/src/presenter/tests/shader.rs`
- `/Users/xnetcat/Documents/REmu/crates/remu-shader-metal/src/compiler/module.rs`
- `/Users/xnetcat/Documents/REmu/crates/remu-shader-metal/src/compiler/driver.rs`
- `/Users/xnetcat/Documents/REmu/crates/remu-shader-metal/src/compiler/export.rs`
