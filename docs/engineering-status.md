# Engineering status

This is the resumable implementation ledger for the ordered game milestones. It records proven
semantics and concrete gaps; a splash, timeout, or lack of `unimplemented` messages is not a
milestone completion claim.

## Current milestone evidence

- Target: Dreaming Sarah, then Superliminal, then Demon's Souls.
- Last known-good pre-regression evidence: a real 3840x2160 Ratalaika splash was captured through the
  translated Metal path and visually checked for color, orientation, and layout.
- The later ordered shader-compilation gate exposed a P0 crash when capture directly called Metal
  `getBytes` on a private, internally compressed direct-scanout render target.
- Current fix: every supported texture-to-CPU path uses a checked texture-to-shared-buffer blit
  ticket, waits for exact command-buffer completion, retains source/staging lifetime, strips aligned
  row padding, and rejects unresolved MSAA, memoryless, framebuffer-only, mismatched, and unknown
  layouts. No title-specific behavior is involved.
- Synthetic Metal coverage includes private BGRA render-target readback, odd-width packed rows,
  source lifetime after the caller drops its texture handle, four-sample resolve-target readback,
  private `Depth32Float` readback, and the ordered one-shot draw-before-flip gate.
- Post-fix bounded validation passed by deliberate timeout after 15 seconds: the ordered draw replayed
  once, two identical 3840x2160 frames were captured, later draws and AGC work progressed, no
  unimplemented NID was reached, and no REmu process remained. The two raw sRGB-storage PPM files have
  SHA-256 `3b5036346a18963e5161ee8fdce40b46002bc006bba6e48136825f5162e4c553` and were visually confirmed as
  the Ratalaika splash with correct orientation and layout.
- The raw hash differs from the older linear-UNORM reference because the direct scanout target is an
  sRGB Metal texture. Applying the standard sRGB transfer leaves only one-code-value quantization
  differences in 204,300 of 24,883,200 color channels (maximum delta 1). Capture now performs this
  format conversion through a 256-entry lookup table; the real P0 crash is closed, while later-frame
  progress and full Dreaming Sarah playability remain open milestones.
- Retained post-splash evidence identifies the next ordered stream as a constant-color fullscreen
  draw with no programmed target, followed by pixel partial flush, memory acquire, and VideoOut
  flip. Both shaders already lower to valid MSL; Metal rejected the draw before compilation because
  it had no attachment.
- The current increment resolves that draw against the nearest valid later flip during a validated
  two-pass submission parse, then emits `Draw -> PipelineBarrier(Pixel) -> MemoryAcquire -> Flip`
  without reordering. Inference requires complete shader resource evaluation, one effective color
  export, no depth, and no image writes, and it never crosses clear/reset, predication, conditional,
  or indirect-control boundaries. Synthetic runtime and ordered Metal compile-gate tests pass.
- A single direct-shell, process-checked, 30-second headless run at `8ac72bd` crossed the former
  rejection: the resolved shader pair `0x80049c200/0x80049e100` compiled, both scanout buffers were
  captured after execution, no unimplemented NID was reached, and no emulator process remained.
  Two splash frames have SHA-256
  `e40b7edcc73511a9b535953e3430fd37d5bfd87b4b276cf11205f15ef9ffe7a7`; two later frames have
  SHA-256 `2d02539a60a0d952d92b30e455ac5aeb2ab13510993224e7c94cdd4e29fbb581`. Visual inspection shows
  the correct Ratalaika splash followed by opaque black, matching the later pixel shader's constant
  `[0, 0, 0, 1]` export. This closes the retained blocker, not the Dreaming Sarah milestone.
- The 30-second run ends before a new failure: historical retained runs first reached the dynamic
  1280x720 tile-27 phase after roughly 45-51 seconds. Runtime submission/draw/compute/flip logs and
  Metal encoded-draw logs now retain their small initial prefix plus power-of-two checkpoints, so a
  longer bounded run can prove progress or a stall with logarithmic rather than unbounded output.
- A single direct-shell, process-checked, 70-second run at `f9e476b` reached AGC submission and draw
  checkpoint 1024 and completed the retained data-payload parse, then exited from a guest `SIGBUS`
  after about 67 seconds. The retained run is
  `.remu/runs/20260722T124738Z-dreaming-sarah`; the matching macOS report is
  `~/Library/Logs/DiagnosticReports/remu-2026-07-22-144902.ips`. It faulted BackgroundLoader worker
  0 at guest RIP `0x800104f7d` while dereferencing an inaccessible allocator-list pointer. Other
  loader workers were active or blocked on the same work-pool synchronization, and the last import
  hit was `kDh-NfxgMtE` (`scePthreadCondSignal`) at the producer-side scheduling call. This is a
  temporal correlation, not a causal conclusion: the exact consumer loops on a mutex-protected work
  predicate, so an extra awakened worker rechecks false and waits again. The immediate failure is a
  stale per-thread allocator freelist pointer into a range the guest had explicitly unmapped.
- The runtime had routed signal and broadcast through one condition-wide generation plus
  `notify_all`, incorrectly selecting every guest waiter. The current source-backed increment uses
  FIFO per-wait tickets: signal selects exactly one current waiter, broadcast snapshots all current
  waiters, empty signals are not retained, timeout and notification have one atomic winner, waits
  release/reacquire the mutex atomically while preserving recursive depth, and selected completion
  survives legal condition destruction. SCE calls return kernel errors and POSIX calls return errno.
  Threaded tests cover exact one-waiter selection, mutex reacquisition, timeout, dynamic mutex
  binding, busy destruction with pending waiters, destruction after selection, and a three-worker
  predicate-guarded pool in which one signal permits exactly one task consumption. Bounded
  power-of-two condition ticket/selection telemetry complements the existing opt-in all-call VM
  map/protect/query/unmap trace. Semaphore waits and signals use the same guest-call sequence policy:
  a blocked wait is logged once rather than after every unrelated wake of the shared host condition
  variable. The retained run below contained 12,609 semaphore lines before this bound. Native and
  Rosetta workspace tests and strict Clippy passed before the process-checked run below.
- The single post-fix run at `c883eff` used direct shell, headless Metal, an exact 80-second bound,
  and opt-in VM/synchronization traces. It ended by deliberate timeout with four frames, zero
  unimplemented NIDs, and no remaining process; artifacts are retained under
  `.remu/runs/20260722T131219Z-dreaming-sarah`. It crossed the former 67-second SIGBUS, completed
  the data-payload parse, retained AGC/Metal draw checkpoint 1024, and continued loading launcher,
  menu, image, FLAC, and OGG resources through the timeout. Splash and opaque-black transition
  hashes remain `e40b7edcc73511a9b535953e3430fd37d5bfd87b4b276cf11205f15ef9ffe7a7` and
  `2d02539a60a0d952d92b30e455ac5aeb2ab13510993224e7c94cdd4e29fbb581`. This closes that crash
  checkpoint, not the title milestone or post-transition rendering.
- The VM trace exposed a more direct allocator-corruption mechanism. Independent adjacent direct
  mappings were normalized into one VMA, so a query at `0x7005530000` reported the merged range
  `0x7000230000..0x7005560000`. The guest consequently requested a `0x5330000`-byte unmap from the
  interior address; strict coverage rejected it, but the following physical release succeeded and
  allowed live aliases to be reused. The old fault page `0x70093c39a0` lay in the same incorrectly
  coalesced/released span. The guest explicitly supplied `MAP_NO_COALESCE` (`0x400000`). The current
  increment propagates that public flag into backend-neutral VMA policy: fragments from one mapping
  may recombine, compatible default mappings may coalesce, and independent no-coalesce mappings
  retain exact `VirtualQuery` boundaries. Releasing a direct-memory subrange now invalidates every
  overlapping virtual alias while preserving adjusted tails before physical ownership becomes
  reusable. Synthetic address-space and kernel-memory tests reproduce these invariants without
  title data.

## Subsystem ledger

| Subsystem | Implemented invariant / evidence | Concrete next gap |
| --- | --- | --- |
| CPU, loader, linker | x86-64 guest execution remains behind the native CPU backend and runs under Rosetta; ELF/SELF mapping, observed relocations, TLS, and import trampolines have synthetic coverage. | Continue source-backed HLE import closure; do not begin the AArch64 translator before a milestone requires it. |
| AGC / PM4 ordering | Native Gen5 command routing, type-2 packets, clear state, ABI-side packet-predication patching, submission-bounded targetless scanout association, compute submission, a backpressured 64-event transport channel, and a one-entry shader compile gate are implemented. Partial-flush `EVENT_WRITE` types 7/15/16 become ordered compute/vertex/pixel GPU barriers; a blocked draw replays once before barrier/acquire/flip successors. | Implement execution-side `SET_PREDICATION`/header-bit semantics, then source-backed release/wait, cache visibility, fence writes, and graphics-equeue completion. Keep those operations distinct; a partial flush alone is not a CPU notification. |
| Metal resources / readback | Persistent guest textures, direct private scanout targets, command-batch completion, shared staging-buffer readback, exact packed rows, MSAA resolve targets, and supported depth readback are covered by GPU tests. Internal Metal compression is handled by the blit and remains separate from guest tiling. | Move diagnostic capture to bounded asynchronous tickets and a reusable staging pool after correctness evidence; add explicit block-compressed and combined depth/stencil aspect layouts when guest formats exercise them. |
| Guest virtual/direct memory | Fixed/hinted mappings, physical allocation splits, coherent aliases, strict range unmap/protect, and guest metadata are covered. `MAP_NO_COALESCE` mappings retain private identities for exact `VirtualQuery` boundaries; direct release invalidates all overlapping aliases and preserves unaffected tails. Containment-first/find-next query behavior and cross-VMA atomic unmap have synthetic coverage. | The full native/Rosetta gate is green. Validate the corrected query/unmap/release sequence in the next single controlled title run, then complete mapping lifecycle counters and exact query error/flag behavior for all exercised cases. |
| Render state / pipelines | GFX10 render-target decoding, MRT attachment layout, write masks, viewport/scissor, formats, and translated shader pipelines are present. | Complete reusable blend, depth/stencil, rasterization, culling, front-face, multisampling, resolve, and full structural cache identity; audit old blend commit `a0388df` without wholesale cherry-picking. |
| Shader translation | Typed backend-neutral Gen5 IR, direct single-path MSL lowering, scalar/vector state, exports, sampled/storage images, buffers, and observed compute geometry are implemented with offline Metal compilation tests. | Complete the unimplemented RDNA2 instruction/resource/CFG surface and replace large per-draw shader snapshots with bounded content-addressed immutable spans. |
| Presentation / frame capture | Guest display size is independent of host drawable size; headless Metal is opt-in; interactive runs use a regular app bundle; direct scanout avoids routine GPU-to-CPU presentation round trips. The bounded scanout/barrier run captured the correct splash and the expected later opaque-black transition from both alternating buffers. | Reach and capture the first post-transition scene after another concrete fix or diagnostic feature, then measure frame pacing. Do not infer playability from splash/transition frames. |
| Runtime / ABI | Whole-process import auditing reports 379 HLE routes, 130 guest-libc routes, one host route, and one unresolved weak `__pthread_cxa_finalize`. Condition waits now use explicit FIFO tickets with distinct signal/broadcast cardinality, atomic mutex release/reacquisition, recursive-depth preservation, dynamic mutex binding, timeout arbitration, and SCE/POSIX error encodings. | Validate the BackgroundLoader crash correlation, then complete opaque guest condition-object lifecycle, guest-thread identity, cancellation, condition clock attributes, and POSIX absolute timed waits. Continue source-backed AGC fusion/event queues, save-dialog lifecycle, semaphore poll, unlink, pad/thread/save/sysmodule lifecycle, and researched anonymous NIDs. |
| Audio | Host output, guest port/config validation, bounded queues, resampling, and stereo downmixing exist. | Audit timing/backpressure/device change/pause/teardown/A-V synchronization against actual title use and add deterministic offline mixer evidence. |
| Video / Bink | No compatibility claim yet. | Identify each title's actual decoder/API path, then implement exact demux/decode/color/audio/timing semantics with legal generated fixtures. |
| Storage / save data | Read-only app filesystem access and initial save transactions exist; dumps are not modified. | Complete namespace/path/case/permission/stat/iteration/atomicity/fsync/quota/slot/dialog/delete/concurrency semantics with crash-consistency tests. |
| Input | Keyboard-backed pad state and opt-in deterministic Cross sequences exist. | Complete handle lifetime, disconnect/timestamps/dead zones/motion/touchpad/controller mapping/focus semantics as exercised. |
| Configuration / telemetry | Environment-controlled presentation, capture, Metal compiler, HUD, and input diagnostics exist; run summaries retain revision/outcome/frame counts. AGC submissions, draws, compute dispatches, flips, Metal draw encodes, and condition wait/notify selections keep bounded logarithmic checkpoints; opt-in VM tracing records every low-volume map/protect/query/unmap transition. | Add a versioned documented config with strict precedence/validation and structured shader/pipeline/queue/copy/CPU/GPU/audio/frame-pacing counters. |

## Required validation for the current P0 increment

1. `cargo fmt --all -- --check`
2. `cargo test --workspace`
3. `cargo clippy --workspace --all-targets -- -D warnings`
4. `cargo test --workspace --target x86_64-apple-darwin`
5. `cargo clippy --workspace --all-targets --target x86_64-apple-darwin -- -D warnings`
6. `python3 -m unittest scripts/tests/test_audit_import_coverage.py`
7. Complete at `c883eff`: one direct-shell, process-checked, 80-second headless Dreaming Sarah run
   crossed the old SIGBUS and reached sustained background asset loading. Its VM trace exposed the
   adjacent-map query-boundary defect now covered synthetically. The mapping-identity increment has
   passed checks 1-6; its next evidence step is one controlled direct-shell run, not an interactive
   or routine launch loop.

Reference implementation provenance is recorded in [provenance](provenance.md); architectural and
licensing constraints remain in [architecture](architecture.md) and [source policy](source-policy.md).
