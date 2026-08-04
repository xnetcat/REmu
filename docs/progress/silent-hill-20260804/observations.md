# Silent Hill bring-up — 2026-08-04

> **Correction / current gate:** the ruined hallway/content-warning backdrop described below is a
> pre-menu presentation and does not prove the actual title-menu 3D scene renders. The real New
> Game / Continue / Settings menu is reached, but its intended scene background is entirely black.
> Silent Hill cinematic/gameplay rendering and crash-free sustained play remain open. The durable
> current handoff is `docs/notes/silent-hill-20260804-handoff/README.md`.

## Outcome

`SILENT HILL: The Short Message` now boots through the REmu Rosetta app runner, renders its
1920×1080 startup and setup UI, responds to automated Cross input, and remains alive until the
150-second test watchdog deliberately ends the latest-code run. A separate uninterrupted
240-second endurance run reaches the interactive brightness screen. No guest fatal, unhandled
memory fault, host panic, or process crash appears in the final preserved run log.

The final run is evidence for stable boot and rendered interactive setup, not a claim that every
later gameplay path has been exercised.

## Initial assessment

The reported launch had two independent build defects and four later runtime defects:

1. Cross-target builds silently compiled a non-functional libav bridge when system FFmpeg was not
   visible.
2. `prepare-rosetta-runner.sh` called `scripts/lib/build-macos-app.sh`, which does not exist; the
   packager actually lives at `scripts/build-macos-app.sh`.
3. Silent Hill's redundant x86 operand-size prefixes hid a valid `mov r64, fs:[0]` from the TLS
   patcher and left a raw FS access for Rosetta to execute.
4. A queued copied GPU writeback was forwarded after its guest VA had been recycled onto a new
   physical allocation. This corrupted Unreal Engine state and produced the missing
   `BlueprintUpdateCamera` fatal. A game-backed bisect identified commit
   `561a83a8d22f4eba0c5074d351242d2f829e77bb` as the first bad change.
5. After rendering was restored, texture content validation could retry forever while a guest CPU
   thread continuously wrote a watched 15.9 MiB surface. A live process sample caught every
   presenter-thread sample in the dirty-content fingerprint loop.
6. Repeated game-backed validation exposed a condition-variable boundary race: a timeout could be
   decided under the condition lock, then a notify could select that expired ticket before waiter
   cleanup acquired the registry lock. The resulting double-winner assertion aborted the host.

## Implemented fixes

### FFmpeg and build integration

- Added `scripts/lib/ensure-ffmpeg.sh` as the build-owned dependency path.
- Pinned the official FFmpeg 8.1.2 source archive and its SHA-256
  `464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c`.
- The first Cargo build compiles and caches static target-specific libraries under
  `.remu/deps/ffmpeg/8.1.2-remu4/{aarch64-apple-darwin,x86_64-apple-darwin}`. Builds are locked,
  staged, architecture-checked, and atomically published.
- Enabled the MOV/MP4 demuxer plus H.264, HEVC, and AAC decoding required by REmu's current
  AvPlayer/Videodec2 bridge.
- `remu-runtime/build.rs` now requires and statically links the pinned libraries for both native
  arm64 and Rosetta x86_64. It no longer falls back to `REMU_LIBAV_STUB`; the C stub implementation
  was removed.
- Kept the builder's stdout contract to one install path so Cargo can consume it on a cold cache.
- Corrected Rosetta app packaging to invoke the repository-level `scripts/build-macos-app.sh`.

### TLS startup

- Replaced byte-length assumptions with semantic iced-x86 matching for `mov r64, fs:[0]`.
- Direct calls are traversed during seeded decoding.
- The complete decoded instruction, including redundant prefixes, is replaced by the helper call.
- TLS load/store helpers preserve the flags, GPR, stack/red-zone, and vector state that the replaced
  MOV instruction leaves unchanged.

### GPU allocation lifetime

- Copied writable globals are now published only while the captured VA still maps the expected
  backing. A recycled VA revokes the delayed writeback instead of following the old physical
  allocation through an alias.
- A later read of a recycled VA synchronizes and snapshots its replacement mapping rather than
  forwarding the previous owner's Metal buffer.
- Physically owned resource paths retain their explicit producer-identity lease semantics; this
  change is scoped to VA-owned copied globals.

### Presenter liveness

- Dirty-content fingerprinting now makes at most four stability attempts.
- If the guest continues writing, validation conservatively reports changed and returns control to
  the presenter. The active write watch remains dirty so the normal reload/snapshot path retains
  correctness.

### Condition timeout/notify race

- A timeout, cooperative interrupt, or diagnostic spurious release now cancels its waiter ticket
  while still holding the condition's state lock.
- A late notifier can no longer consume its signal on an expired ticket or race waiter cleanup into
  the exactly-one-winner assertion.

## Validation

### Media bridge

The H.264/AAC two-second AvPlayer smoke test passed with actual in-process video and audio decoder
startup on both targets:

- native `aarch64-apple-darwin`
- Rosetta `x86_64-apple-darwin`

The Silent Hill startup segment did not invoke AvPlayer (`controller=not-started`), so these decoder
tests independently verify that media support is real rather than inferred from the game boot.

### Focused regressions

- FFmpeg/build integration: 4 Python tests passed; both shell scripts pass `sh -n`.
- FS decode: 6 tests passed.
- TLS helper/patching: 6 tests passed.
- Live-VA lease revocation: passed.
- Recycled copied/global writeback tests: passed.
- Valid shifted physical-alias and ordinary queued-buffer forwarding tests: passed.
- Continuous guest writer snapshot and content-validation tests: passed.
- Clean texture validation and live-baseline fast-path tests: passed.
- Condition timeout-ticket retirement plus four existing timeout/signal/broadcast/destruction
  semantics tests: passed.
- `git diff --check`: passed.

### Final Silent Hill run

Command shape:

```sh
REMU_WATCHDOG_SLACK_SECONDS=30 \
REMU_CAPTURE_FRAMES=1 \
REMU_CAPTURE_INTERVAL_SECONDS=10 \
REMU_AUTO_CROSS='<every 4 seconds from 4 through 140>' \
./scripts/run-game.sh '/Volumes/Untitled/sharpemu/games/SILENT HILL The Short Message/PPSA10112-app/eboot.bin' 150
```

Observed result:

- outcome `timeout`, exit status 142: the configured watchdog ended the still-running title
- 162 seconds wall time, including startup/teardown overhead
- 4 captured changed frames, 1920×1080
- 0 unimplemented NIDs
- no condition double-winner, `BlueprintUpdateCamera`, `LowLevelFatal`, unhandled fault, panic, or
  crash signature
- 28 telemetry windows
- final stall attribution: `gpu=outstanding-active`, completed submission 1845, progress age 6 ms
- final AGC state: 1850 submissions with active graphics/compute work

The preceding 240-second endurance run also ended only at its watchdog, emitted 46 telemetry
windows, completed submission 2326 with a 3 ms progress age, and remained on the rendered
brightness calibration page. This was later traced to the old automation alternating Cross and
Square. Cross-only edge pulses do accept the brightness value and advance the title without manual
input.

### Ten-minute Cross-only progression run

A later 600-second run used Cross-only release/press edges and reached rendered opening content
beyond calibration. It loaded 474 graphics and 132 compute pipelines and stayed alive until the
configured watchdog. The captured frame `screenshots/opening-room-content-warning.png` contains the
ruined-room pre-menu backdrop, the crisis-support content warning, and the Continue prompt. It must
not be counted as proof of the actual title-menu 3D scene.

That run exposed four concrete remaining translation gaps which were fixed before the next run:

- indirect NGG draws now build ordered meshlet records and dispatch arguments on the GPU from the
  live indirect/index buffers;
- comparison-sampled BC1 color resources and exact mode-5/mode-27 mip hierarchies now upload as
  native Metal depth textures;
- the complete 64-bit LDS pair family (`DS_WRITE2_B64`, `DS_WRITE2ST64_B64`, `DS_READ2_B64`, and
  `DS_READ2ST64_B64`) now decodes and lowers with architectural element strides;
- half-float `V_FMAMK_F16`/`V_FMAAK_F16` now consume their unconditional literal dword, preventing
  the literal from being misdecoded as a following raw VOP2 instruction.

### Latest title-menu progression

The next 600-second Cross-only run remained alive until its external watchdog and reached the
actual title menu. Automated Cross selected New Game and progressed through the content warning to
the opening quote. At termination 475 graphics and 133 compute pipelines were ready, none had
failed, no draw or compute was dropped, and no NID was unimplemented. This is the first preserved
run proving autonomous progression from process launch through brightness calibration and into a
new-game cinematic sequence.

The diagnostic PPMs from this run contained partial horizontal/vertical regions and damaged text.
The pattern did not resemble a font-atlas-only failure: it crossed the logo, background, and UI at
shared scan lines. Inspection found that periodic readback was submitted on the host queue before
the translated scanout's guest-queue SharedEvent fence. The drawable present already waits on that
fence; diagnostic readback now encodes the same wait. A synchronized desktop capture will be used
as the visual authority for the next run.

### Post-change compatibility observations

These are observations only. No title-specific compatibility work was done outside Silent Hill.
Every launch enabled automatic dialog dismissal and Cross pulses.

- Octopath Traveler II ran for 185 seconds until the external watchdog, rendered startup, setup,
  dialogue, and 3D scenes at roughly 7–10 FPS, and did not crash. Two shader pairs were rejected at
  `V_WRITELANE_B32`/`V_READLANE_B32` because their Metal lane occupancy is not yet proven. Evidence:
  `screenshots/octopath-observation.png`.
- Superliminal remained alive for 185 seconds and produced one clean rendered 3D hallway frame,
  then stayed static with no further GPU submissions and a 0 FPS title. Evidence:
  `screenshots/superliminal-observation.png`.
- Demon's Souls aborted after 15 seconds in its own audio assertion because
  `sceAcmBatchStartBuffers` is unimplemented; it produced no captured frame.
- Dreaming Sarah logged missing packaged `shaders/*.vert` files and segfaulted before a frame.
- Void Terrarium segfaulted during startup/import/thread initialization before a frame.
- The Oregon Trail did not launch because the dump contains duplicate `libc.prx` module paths.

Poppy was deliberately not relaunched during this matrix; its requested second bring-up remains
behind the Silent Hill gameplay gate.

The summary does list normal negative results from implemented APIs—for example absent host files,
no connected controller state, and no pending user-service event. These are not unimplemented NIDs
and did not prevent the title from rendering or advancing through setup.

## Evidence

### Deferred Poppy result

- The one Poppy launch was not healthy: it remained on a black screen at 0 FPS.
- That launch is not evidence of rendering, performance, or title compatibility. Poppy remains
  excluded until Silent Hill passes the title gate; it will be revisited afterward as a separate
  compatibility target.

- `screenshots/stable-120s-05.png`: rendered autosave screen
- `screenshots/stable-120s-06.png`: fully composed content-warning scene
- `screenshots/stable-120s-07.png`: interactive Continue prompt
- `screenshots/stable-120s-08.png`: interactive brightness calibration
- `screenshots/stable-240s-04.png`: later-run fully composed content-warning scene
- `screenshots/stable-240s-05.png`: 240-second endurance-run brightness calibration
- `screenshots/post-race-150s-*.png`: final latest-code rendered progression after the condition
  race fix
- `screenshots/opening-room-content-warning.png`: Cross-only progression into the ruined-room
  pre-menu presentation beyond brightness calibration; not title-menu 3D evidence
- `screenshots/latest-title-menu.png`: latest run at the actual title menu (pre-fix asynchronous
  diagnostic readback; retained as evidence of progression, not pixel fidelity)
- `screenshots/latest-opening-quote.png`: latest run after selecting New Game (same pre-fix
  diagnostic readback limitation)
- `screenshots/octopath-observation.png`: isolated post-change Octopath observation
- `screenshots/superliminal-observation.png`: synchronized isolated Superliminal observation
- `screenshots/run1-*.png`: earlier bring-up progression, retained for comparison
- `diagnostics/post-race-150s-summary.txt`: final latest-code harness result and ABI summary
- `diagnostics/post-race-150s-telemetry.jsonl`: all telemetry windows from the final latest-code run
- `diagnostics/post-race-150s-guest.log`: complete final latest-code guest/host log
- `diagnostics/stable-240s-*`: extended endurance evidence
- `diagnostics/condition-race-repro-*`: preserved 26-second abort that authenticated the final
  timeout/notify fix
- `diagnostics/stalled-process-sample.txt`: process sample that authenticated the pre-fix presenter
  starvation loop
