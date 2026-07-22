# REmu next-agent handoff prompt

Copy everything below this line into the next primary agent's task.

---

You are the primary implementation agent for REmu, a GPL-2.0-or-later PlayStation 5 userspace
compatibility layer written in Rust. Work continuously toward three ordered game milestones:

1. Dreaming Sarah (`PPSA02929`) is visibly playable with correct rendering, input, audio, videos,
   storage, and stable frame pacing.
2. Superliminal (`PPSA06084`) reaches the same standard.
3. Demon's Souls reaches the same standard, including the 3D, compute, synchronization, and
   advanced rendering behavior it exercises.

Do not implement the PS5 shell. Test only the user's lawful local dumps. Do not put game files,
firmware, keys, proprietary SDK material, copyrighted assets, or decompiler output in Git.

## Non-negotiable product direction

- Apple Silicon macOS is the only host target for the current milestone. Run the x86-64 guest and
  emulator process through Rosetta 2 for now. Keep a clean CPU-backend boundary for a future custom
  x86-64-to-AArch64 translator; do not start that translator before the game milestones require it.
- macOS uses direct Metal. The future Windows backend is direct D3D12 and Linux is Vulkan. Keep AGC,
  PM4, shader IR, memory, synchronization, and resources backend-neutral so those backends do not
  require a redesign.
- Build a general emulator/translation layer, not per-title patches. Every fix must follow PS5 ABI,
  AGC/RDNA2, kernel, library, or file-format semantics and be covered by a synthetic regression test.
- Never add a stub, silent no-op, guessed success result, dummy resource, fake shader output, fallback
  frame, or title/product-ID special case. If behavior is not yet implemented, fail explicitly with
  enough structured state to implement it correctly next.
- If an unimplemented ABI, NID, PM4 packet, shader instruction, format, synchronization operation, or
  resource behavior is encountered, research and implement its full reusable semantics. Do not only
  implement the narrow values seen in one trace.
- Rendering correctness and performance are co-equal requirements. A frame is not accepted because
  it merely appears; validate its resource lifetime, ordering, formats, shader semantics, blending,
  depth/stencil, synchronization, and readback path. A correct path is not accepted if avoidable
  copies, synchronous compilation, unbounded queues, or per-draw large allocations make it unusable.
- Audio, Bink/video playback, storage/save data, configuration files, controller input, and error
  handling are day-one workstreams. Implement them alongside graphics/runtime work when the games
  first exercise them. Do not defer them to a hypothetical polish phase.
- Commit completed, tested increments frequently. Keep commits focused and explain the hardware/ABI
  source and invariant in the commit message or provenance documentation.

## Absolute execution and UI rules

- Never use cmux, Computer Use, browser UI automation, AppleScript, GUI terminal automation, or any
  indirect UI launcher. This applies to the primary agent and every subagent.
- Visible REmu launches are allowed only from a direct shell command.
- Only one emulator process/window may exist at a time. Before a launch, use a read-only process
  check scoped to REmu. Never kill unrelated processes. Always stop the exact process you started.
- Do not launch the emulator as a routine debugging loop. Launch only after a concrete fix or a
  substantial feature has passed unit/integration tests and static checks.
- Prefer deterministic frame dumps, command-buffer/PM4 dumps, shader dumps, structured log diffs,
  import audits, executable inspection, and local decompilation/disassembly for normal development.
  Keep proprietary/decompiled output outside Git.
- Headless Metal is the default validation mode when a real GPU run is required. Use a visible app
  bundle only when validating AppKit/Dock/window/input/presentation behavior.
- Do not leave a background emulator running. Bounded diagnostic runs must end at their deliberate
  timeout. Interactive runs are only for an explicit human playability check.

## Use subagents deliberately

Use subagents in parallel for independent, bounded work, with isolated Git worktrees/branches when
they edit. At the start of every subagent task, repeat all of these rules: direct shell only, no cmux,
no Computer Use or UI automation, do not launch unless explicitly authorized, no stubs or title hacks,
write tests, commit the result, and report exact evidence. The primary agent owns architecture,
reviews every diff, runs integration validation, and cherry-picks only understood commits.

Maintain at most these parallel tracks:

1. GPU/Metal/PM4/shader correctness and performance.
2. ABI/NID/kernel/runtime/audio/storage/video implementation.
3. Research and differential audits across emulator sources, drivers, public AMD documentation, and
   the local game binaries/traces.

Do not have multiple agents edit the same files. Do not let an agent launch REmu while another launch
is active. Stop or redirect agents that start speculative/title-specific work.

## Repository and exact starting state

- Repository: `/Users/xnetcat/Documents/REmu`
- Branch: `main`
- Starting commit: `82d7faf9d87f91d6a714bacbe2af1f13a018ef01`
- The main worktree was clean when this handoff was written.
- License: GPL-2.0-or-later. Preserve provenance and upstream notices according to
  `docs/source-policy.md` and `docs/provenance.md`.
- Read `README.md`, `docs/architecture.md`, `docs/source-policy.md`, `docs/reference-audit.md`,
  `docs/provenance.md`, and `crates/remu-shader-metal/OPTIMIZATION.md` before changing architecture.

The last main commits, newest first, are:

- `82d7faf` preserve GPU order across shader compilation
- `a17a641` implement exact AGC packet predication ABI
- `ad58da0` apply native AGC clear-state semantics
- `8f9cc24` launch interactive runs as a macOS app bundle
- `c5f6f6e` add opt-in headless Metal presenter
- `6181a38` make import audits understand native AGC dispatch
- `321c615` route exact AGC type-two packet ABI
- `1f679f1` present targetless blits through direct scanout
- `32e9861` make interactive runner lifecycle explicit
- `f6ca0db` associate targetless final blits with scanout flips
- `564d18d` implement GFX10 Metal render targets
- `bf94e13` lower single-path shaders directly to Metal
- `7c9590c` route native Gen5 AGC command ABI
- `015b7c0` wire Gen5 compute submissions into Metal

Do not assume all detached/old-agent work belongs on main. Most is superseded or deliberately
rejected. The only known high-value unmerged item is the general blend-state work represented by
old commit `a0388df`; port and adapt its semantics to the current multi-render-target architecture
after review. Do not cherry-pick it wholesale. An interrupted fresh blend worktree had no changes.

Explicitly reject these old approaches:

- `e7906d9`, `21d4f30`, `cb398c3`: truncating/zero-padding shader buffers.
- `5170f40`, `b776a2e`: ignoring SIGTERM.
- `2dd839d`: hiding REmu from the Dock/accessory activation.
- `2d3f0c9`: discarding GPU events while a shader compiles.
- `e85cdaf`: clock regression.
- `1094085`: unverified title-specific clock behavior.
- `c740fcf`: obsolete sampler fallback.

Potential lower-priority old work to audit, not blindly merge:

- `3fb8476`: deterministic pad sequences.
- `785b5ea`: delayed/count-limited capture.
- `f77a497`, `99913ef`, `0062524`, `b0d2767`, `f0b44e8`: diagnostics.

## Local lawful test inputs

- Dreaming Sarah executable:
  `/Volumes/Untitled/sharpemu/games/test-game/PPSA02929-app0/eboot.bin`
- Superliminal executable:
  `/Volumes/Untitled/sharpemu/games/PPSA06084-app/PPSA06084-app0/decrypted/eboot.bin`
- Search the external drive read-only for the Demon's Souls dump and record its selected executable
  through a local ignored configuration/environment variable. Do not hard-code a guessed path.
- `scripts/run-game.sh` already supports `DREAMING`, `SUPERLIMINAL`, or an explicit executable path.
- The bundled offline Metal compiler, when present, is
  `.remu/toolchains/Metal.xctoolchain/usr/bin/metal` and is opt-in through
  `REMU_USE_BUNDLED_METAL_COMPILER=1` or `REMU_METAL_COMPILER`.

## Proven working evidence

Before the latest ordered-event integration, headless Metal translated and rendered a real Dreaming
Sarah Ratalaika Games splash. This was not a synthetic/title fallback:

- Capture:
  `/private/tmp/remu-headless-capture.PM6DwP/remu-7003240000-3840x2160-tile27-7238bf0c5c979715.ppm`
- SHA-256:
  `d2f64c7e5ed97bc2245f7427fffc9d7def7f7baf5cfd7947ae2fdf74fb4b96a3`
- It was visually checked for correct color, orientation, and layout.

Interactive app-bundle validation also established:

- LaunchServices registered `REmu` with bundle ID `org.remu.emulator`.
- REmu appeared in the Dock and stayed alive for deliberate 12-second and 20-second bounded runs.
- No window-close, channel-disconnect, or AppKit-loop-stop message occurred during those runs.
- The drawable was 1280x720 logical-window resolution, not forced to 4K. The guest independently
  requested a 3840x2160 display buffer. `REMU_PRESENTATION_SIZE` configures presentation only.

## Immediate P0 regression: fix this before another milestone run

Current `main` crashes in the headless capture path immediately after replaying a draw whose shaders
finished compiling. Do not blindly rerun. First make the capture/readback path valid for the exact
Metal texture storage/layout and preserve event ordering.

Reproduction that produced the crash (do not repeat until code inspection and a concrete fix):

```sh
env REMU_SKIP_BUILD=1 \
    REMU_HEADLESS_METAL=1 \
    REMU_CAPTURE_FRAMES=1 \
    REMU_METAL_COMPILER=/Users/xnetcat/Documents/REmu/.remu/toolchains/Metal.xctoolchain/usr/bin/metal \
    REMU_PRESENTATION_SIZE=window \
    scripts/run-game.sh DREAMING 15
```

Artifacts:

- Run: `.remu/runs/20260722T055628Z-dreaming-sarah`
- Crash report:
  `/Users/xnetcat/Library/Logs/DiagnosticReports/remu-2026-07-22-075633.000.ips`
- Exception: `EXC_BAD_ACCESS / SIGSEGV`, invalid address `0x1fe0000`.
- Faulting thread: main / `com.apple.main-thread`.
- Native stack:
  `AGX readCompressedRegion2D` -> `AGX Texture::readRegion` ->
  `-[AGXG16GFamilyTexture getBytes:...]` -> `metal::TextureRef::get_bytes` ->
  `remu_gpu_metal::presenter::read_texture_linear` ->
  `MetalPresenter::receive_events`.
- Last important log:
  `[REMu][METAL] compiled Gen5 shader pair es=0x7005609300 ps=0x7005609d00 MSL=27575/5717 bytes`
- No frame was captured on current main.

Inspect `crates/remu-gpu-metal/src/presenter.rs`, especially `OrderedGpuEventReceiver`,
`receive_events`, compilation-result replay, direct scanout, `read_texture_linear`, texture storage
modes, and command-buffer completion. The likely class of bug is illegal synchronous `getBytes` from
a private/compressed render target or a missing resolve/blit/synchronization step exposed by replay;
prove the cause from descriptors and Metal rules before editing. A correct readback path normally
uses a shared/managed staging buffer, a Metal blit/resolve, command-buffer completion, and exact row
pitch/format handling. It must work for compressed/tiled guest images, MSAA resolve targets, depth
formats where supported, and future non-capture consumers—not just this splash.

The ordered gate itself has important invariants that must survive the fix:

- A draw blocked on shader compilation replays exactly once.
- Later draw, compute, acquire, flip, capture, and shutdown events cannot overtake it.
- GPU event capacity is bounded (currently 64); compile queue capacity is one.
- Compile failure resolves the gate without a deadlock and reports the real error.

Add unit/integration coverage for ordered replay plus readback resource lifetime. Then run native and
Rosetta tests and Clippy before one bounded headless capture. Compare the resulting frame hash/image
to the known-good splash and keep both logs.

## Current audits and prioritized implementation work

### GPU, PM4, memory, and shaders

After the readback crash, address these as general architectural work:

1. Complete GFX10/RDNA2 blend, color-write-mask, depth/stencil, rasterization, culling, front-face,
   viewport/scissor, multisampling, resolve, and multi-render-target pipeline state. Include every
   relevant field in structural pipeline-cache identity. Port verified semantics from `a0388df` only
   after comparing it with current state and public AMD/driver definitions.
2. Implement missing PM4/AGC packet semantics and synchronization. Dreaming traces include event
   write opcode `0x46`; it is currently ignored. Do not turn it into a no-op or globally hard-error
   without implementing the associated graphics event-queue registration, ordered trigger delivery,
   release/wait semantics, cache visibility, and fence behavior.
3. Replace per-draw large CPU shader snapshots (historically about 512 KiB) with bounded exact shader
   spans/content-addressed immutable objects. Avoid fresh buffers per draw. Cache decoded IR, compiled
   MSL, Metal libraries/functions, render/compute pipelines, resource views, and samplers by complete
   structural keys. Instrument hit rates, compile latency, allocation volume, queue depth, CPU/GPU
   time, and copy bytes.
4. Model GPU-to-GPU dependencies and guest memory visibility explicitly. Do not use stale CPU
   snapshots as synchronization. Track resource ownership, aliasing, dirty ranges, cache operations,
   barriers, event writes, waits, and display-buffer acquisition/flip ordering.
5. Complete shader decoding/lowering using PS5 GFX10/RDNA2 semantics: scalar/vector ALU, control flow,
   EXEC/VCC/SCC, wave32/wave64, interpolation, exports, buffer/image/sampler operations, atomics, LDS,
   compute dispatch, derivatives, texture LOD/gradients, gather/compare, cube/array/MSAA textures,
   DCC, format conversion, and robust bounds/alignment behavior. Reject unsupported instructions with
   exact decoded fields and preserve dumps for implementation.
6. Implement ray tracing as real backend-neutral acceleration-structure and shader semantics when
   exercised. Research PS5/RDNA2 behavior and map it to Metal acceleration structures without
   contaminating guest IR with Metal types. Never fake an RT result.
7. Implement guest tiling/detiling, swizzles, mip chains, arrays, pitch, compression metadata, depth,
   stencil, MSAA, and scanout for all formats exercised by the milestones. Validate conversions with
   independent CPU reference tests and driver documentation.
8. Resolve `resolve_vertex_inputs = false` and complete vertex fetch/index/base/instance/topology
   semantics, including triangle-fan expansion where Metal requires it.

### ABI, libraries, and kernel services

A whole-process Dreaming Sarah import audit found 511 unique callable imports:

- 379 route to HLE.
- 130 resolve to bundled guest libc.
- 1 resolves directly to a host function.
- 1 weak import remains unresolved: `kbw4UHHSYy0` (`__pthread_cxa_finalize`).
- 233/379 HLE imports had concrete REmu paths at audit time.
- 146 were static dispatch gaps; 73 appeared in reference emulators, including 54 in SharpEmu.
- No observed call in the sampled run reached a static gap, so do not misdiagnose every graphics bug
  as a missing NID. Still implement source-backed gaps systematically and test exact ABI behavior.

Centralized AGC dispatch contains 25 handlers duplicated by unreachable legacy arms in
`crates/remu-runtime/src/lib.rs`; remove those only after equivalence tests. The audit/tooling commit
`6181a38` already teaches `scripts/audit-import-coverage.py` about centralized AGC handlers.

High-value source-backed missing work:

- `dolOmWH+huQ` and `fd5Bp5tGTgo`: fused shader-half sizing/fusion. Compare KytyPS5 `agc.cpp` near
  its shader-fusion implementation and implement full stage validation, aligned scratch/register
  sizing, type promotion, register copying, address/checksum patching, and user-data semantics.
- `w2rJhmD+dsE` / `DL2RXaXOy88`: AGC graphics event-queue add/delete. Compare SharpEmu AGC exports
  and implement queue/filter/user-data/error behavior plus ordered PM4 event delivery.
- Save-data dialog lifecycle/result APIs: implement the complete state machine and output layouts.
- `12wOHk8ywb0`: semaphore poll with atomic validate-and-consume/busy semantics.
- `AUXVxWeJU-A`: unlink with guest path resolution, read-only enforcement, file type/existence
  checks, error mapping, and cache invalidation.
- Complete stateful pad close, pthread exit, save deletion, sysmodule query/unload, and the remaining
  source-backed imported APIs.
- Anonymous NIDs still requiring binary/reference research: `gQkqkLttcpw`, `k0E7vkgqAuE`,
  `zARR5aCmkoY`.

Already implemented on main: exact AGC packet predication/range walk, clear-state reset, exact type-2
packet routing, native Gen5 command dispatch, compute submission, targetless scanout association, and
the name-table fixes for the previously misleading 22 runtime `name=unknown` calls.

### Audio, video/Bink, storage, input, and configuration

Treat these as continuous compatibility layers, not later polish:

- Audio: preserve guest port/config/channel/sample-format/timing semantics; implement buffering,
  backpressure, resampling, downmix only where the guest API requires it, device changes, pause,
  teardown, and A/V clock synchronization. Never drop audio silently to improve frame rate. Add
  deterministic offline mixer tests and bounded host-backend tests. Current host output is in
  `crates/remu-runtime/src/audio.rs`; audit it against game usage and reference implementations.
- Bink/video: first determine whether each title uses a bundled decoder, Bink library import, or
  system media API. Support the actual ABI and demux/decode/presentation pipeline. Prefer a compatible
  host codec library only when its bitstream, pixel-format, color-space, alpha, audio-track, seek,
  threading, and timing behavior match. Upload decoded planes efficiently and synchronize video to
  the guest clock/audio. Add legal synthetic or generated tiny media fixtures; do not commit game
  video data.
- Storage/save data: implement mount namespaces, path normalization, case behavior, permissions,
  directory iteration, stat metadata, errno translation, atomic rename/write, fsync/durability,
  quotas, save slots, dialogs, deletion, and concurrent access. Store user data under a configurable
  per-title directory; never modify the dump. Add failure/crash-consistency tests.
- Configuration: introduce a versioned, documented config file plus CLI/environment overrides with
  deterministic precedence. Include presentation size/fullscreen, frame cap/vsync, audio device and
  latency, controller mapping, save/cache/log/capture directories, shader-cache controls, debug
  overlays, and per-title overrides. Defaults must be safe and performant. Reject invalid keys/values
  with actionable errors; migrate versions explicitly; never silently change guest semantics.
- Input: implement accurate pad handles, connect/disconnect, timestamps, dead zones, motion/touchpad
  where exercised, deterministic replay sequences, and focus behavior. Automated Cross presses stay
  opt-in (`REMU_AUTO_CROSS`), never part of normal interactive behavior.

## Required research method

Do not rediscover work already established in emulators or open drivers. For every subsystem, build a
behavior matrix from multiple independent sources before coding, then implement the shared hardware
or ABI invariant in REmu's architecture.

Local research trees already exist under `research/upstreams/` and include Kyty, KytyPS5, SharpEmu,
SharpEmu's Metal branch, SharpEmu upstream, shadPS4, RPCSX, fpPS4, and Obliteration. The user's linked
checkout is `/Volumes/Untitled/sharpemu/`. Record the exact upstream commit/file/behavior consulted in
`docs/provenance.md` for materially adapted work.

Also consult primary/public sources directly:

- AMD GFX10/RDNA/RDNA2 ISA documentation, shader register references, PM4 packet definitions where
  public, GPUOpen samples/tools, format/tiling information, synchronization/cache guidance, and
  performance material.
- Mesa AMDGPU/RADV and Linux kernel AMDGPU driver code for public packet/register/format/barrier,
  command submission, memory, tiling, DCC, depth, MSAA, and synchronization behavior. Distinguish
  hardware facts from driver policy and PS5-specific differences.
- LLVM AMDGPU backend for instruction encodings and semantic cross-checks.
- Apple's current Metal, Metal Shading Language, resource storage/hazard tracking, synchronization,
  binary archive, counter sampling, acceleration structure, texture readback, and performance HUD
  documentation. Prefer Metal API capture/counters and offline compilation for evidence.
- Microsoft D3D12/DXIL and Khronos Vulkan/SPIR-V specifications when defining backend-neutral
  contracts, even though those backends are not current launch targets.
- Public ELF/System V AMD64 ABI, FreeBSD/POSIX behavior, and compatible reference-emulator HLE for
  loader/kernel/library work. Confirm PS5 deviations from binaries/traces rather than assuming PS4.
- Public Bink SDK/API documentation that is legally available, FFmpeg/libavcodec documentation/source,
  and the actual imported ABI/call sites for media work.

Use references as research, not unquestioned truth. Kyty/KytyPS5 may contain obsolete or incomplete
assumptions; SharpEmu may have placeholders; PS4 emulators do not define PS5 behavior; Mesa targets PC
hardware. Triangulate, write synthetic tests, and state uncertainty. Never copy proprietary headers or
paste decompiler output into source.

For game-side analysis, prefer:

- `scripts/audit-import-coverage.py`, loader/import reports, symbol/NID cross-reference tables.
- `llvm-objdump`, `otool`, `readelf`/equivalents, `nm`, `strings`, and a local decompiler/disassembler.
- Bounded call-site disassembly around imports, argument/register traces, PM4 packet dumps, shader
  binary/IR/MSL dumps, resource descriptors, and frame hashes/diffs.
- One trace before a change and one trace after it. Convert observations into title-independent unit
  tests, then delete or keep proprietary artifacts only in ignored local directories.

## Performance engineering requirements

Measure before and after substantial changes. Keep a structured per-run summary containing revision,
game, launch settings, elapsed time, frame count, frame hashes, unimplemented NIDs, shader/pipeline
cache hits/misses, compile time, queue high-water marks, CPU/RAM usage, GPU time/utilization, allocated
bytes, copied/uploaded/read-back bytes, audio underruns, and dropped/late frames.

Prioritize removal of known systemic costs:

- Synchronous offline shader compilation on the presentation thread.
- Duplicate translation/compilation and incomplete pipeline cache keys.
- Per-draw 512 KiB shader snapshots and fresh Metal buffers.
- CPU round trips for GPU resources, direct scanout, capture, synchronization, and aliasing.
- 4K capture/readback during normal play. The guest may render 4K; the host drawable and diagnostics
  remain independently configurable. Capture stays off unless required.
- Unbounded event growth or blocking producer/consumer cycles.
- Coarse guest-memory locks, repeated full-range copies/hashes, busy waits, and excessive logging.

Do not trade correctness for speed. Use immutable/content-addressed shader objects, bounded queues,
asynchronous compilation with strict event ordering, persistent caches, staging/upload rings, heaps,
argument buffers where appropriate, dirty-range tracking, explicit fences/events, and zero-copy paths
only when their ownership and visibility invariants are proven.

## Development and validation loop

For each substantial issue:

1. Reproduce or identify it from an existing bounded log/dump/crash report. Avoid a new launch when
   current evidence is sufficient.
2. Ask parallel research agents for exact source/driver/documentation comparisons while the primary
   agent traces REmu's current implementation.
3. Write down the hardware/ABI invariant and decide the correct backend-neutral boundary.
4. Add focused synthetic tests that fail for the real semantic gap.
5. Implement the full general behavior. No stubs, ignored errors, or title checks.
6. Run formatting, relevant tests, then native and Rosetta workspace tests and strict Clippy.
7. Commit the coherent change.
8. Only after concrete fixes/features accumulate, run one bounded headless emulator validation. Use
   frame/log/shader/PM4 dumps and compare to prior artifacts.
9. Use one direct-shell visible `REmu.app` launch only for a milestone human validation that actually
   needs a window/Dock/input/Metal HUD. Ensure no other emulator is running and clean it up afterward.

Baseline checks:

```sh
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --target x86_64-apple-darwin
cargo clippy --workspace --all-targets --target x86_64-apple-darwin -- -D warnings
python3 -m unittest scripts/tests/test_audit_import_coverage.py
```

Build the Rosetta release only when a GPU/game run is justified:

```sh
arch -x86_64 cargo build --release --target x86_64-apple-darwin -p remu-cli
```

Headless validation, only after a concrete fix and process check:

```sh
env REMU_SKIP_BUILD=1 \
    REMU_HEADLESS_METAL=1 \
    REMU_CAPTURE_FRAMES=1 \
    REMU_USE_BUNDLED_METAL_COMPILER=1 \
    scripts/run-game.sh DREAMING 15
```

Interactive validation is direct shell only:

```sh
env REMU_SKIP_BUILD=1 REMU_CAPTURE_FRAMES=0 REMU_PRESENTATION_SIZE=window \
    scripts/run-game.sh DREAMING
```

Never use that interactive command as an automated loop and never run Dreaming Sarah, Superliminal,
or Demon's Souls concurrently.

## Milestone acceptance gates

### Dreaming Sarah

- Current ordered-readback crash is fixed with a general Metal resource/synchronization design.
- The known Ratalaika splash reproduces without corruption and later frames progress deterministically.
- Menus and gameplay render correctly: textures, blending, color, alpha, depth, viewport/scissor,
  presentation size, orientation, and frame order.
- Input, audio, Bink/video paths used by the title, save/storage, and config behavior work without
  silent fallbacks.
- No unimplemented API/packet/instruction reached during the accepted play path.
- No window disappearance, background orphan, unbounded queue, continual shader recompilation,
  pathological copy volume, audio underrun, or unacceptable frame pacing.
- A bounded trace and a human direct-shell playability check are retained as local evidence.

### Superliminal

- Loader/module/import coverage is complete for its executable and bundled modules.
- 3D rendering, render targets, depth/stencil, compute, textures, synchronization, post-processing,
  audio/video/input/storage/config, and timing are correct through real gameplay.
- Fixes remain general and do not regress the Dreaming Sarah corpus.
- Shader/pipeline caches stabilize after warm-up and performance telemetry identifies no systemic
  blocking/copy/allocation path.

### Demon's Souls

- Locate and audit the lawful dump without committing its content/path-derived artifacts.
- Complete the broader 3D, compute, memory, synchronization, streaming, audio/video, storage, input,
  shader, format/compression/MSAA, and any ray-tracing behavior it exercises.
- Establish cold/warm boot and shader-cache behavior, sustained frame pacing, memory bounds, and
  crash-free gameplay checkpoints.
- Run the full earlier-game regression corpus after every major subsystem completion.

Do not declare a milestone complete from a splash screen, a single frame, a timeout without a crash,
or the absence of an `unimplemented` log line. Completion means stable interactive game progress with
correct output and measured performance.

## Reporting and continuity

Keep a living checked-in engineering status document that records, for each subsystem, implemented
semantics, tests, known concrete gaps, performance counters, reference sources, and the next evidence-
based task. Update it with every major commit so another agent can resume without replaying launches or
rediscovering prior work.

In user updates, lead with concrete shipped behavior, commit IDs, test evidence, and the current
blocker. Do not report activity as progress. If a launch is necessary, say exactly why it is justified
and confirm direct-shell/no-cmux/one-window compliance.

Begin by reading the current crash report and Metal presenter code, assigning read-only research
agents to Metal readback/resource-storage rules and comparable emulator/driver implementations, and
fixing the ordered headless readback regression with tests. Do not launch REmu again until that fix has
passed the non-game validation gates.
