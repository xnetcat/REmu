# Subsystem status

This records what each subsystem actually implements and its next concrete gap. A splash screen,
a timeout, or the absence of `unimplemented` log lines is never treated as a compatibility claim;
completion means stable interactive game progress with correct output and measured performance.

Current overall state: the first small test title boots through resource loading, submits AGC
draw streams, compiles observed RDNA2 shaders to Metal, and flips its display buffers. Early
splash frames render correctly; full scene rendering and playability are still open.

## Import coverage

A PS5 import identifies its function only by an 11-character NID, so measuring coverage means
first recovering names. `scripts/generate-nid-names.py` hashes every symbol exported by a
directory of PS5 modules and keeps the matches; `scripts/audit-import-coverage.py` then reports,
per title, how each imported NID is served.

For the larger of the two test titles, of 2028 callable imports: 1140 resolve inside modules the
game itself ships, 436 are implemented here, and 443 remain unserved. All but 12 have a recovered
name, which is what makes the remaining gap a work list rather than a list of hashes. The unserved
majority is concentrated in the graphics and asynchronous-file command-buffer builders.

Naming a NID is not implementing it. The audit deliberately separates the two, and a handler that
claimed a NID while returning a fabricated success would be worse than no handler at all: it turns
a loud gap into a silent one. Unimplemented imports are expected to stay visible in the log.

| Subsystem | Implemented | Next concrete gap |
| --- | --- | --- |
| CPU, loader, linker | x86-64 guest execution behind a native CPU backend under Rosetta; SELF/ELF mapping, observed relocations, AMD64 DTV slots, per-thread `__tls_get_addr`, TLS clone rebasing, and import trampolines with synthetic coverage. | Continue source-backed HLE import closure. The AArch64 translator stays deferred until a milestone requires it. |
| AGC / PM4 ordering | Native Gen5 command routing, type-2 packets, clear state, packet predication, targetless scanout association, compute submission, ordered memory copy/write/release/wait events, submission boundaries, and a bounded backpressured event channel with an ordered shader-compile gate. | Execution-side `SET_PREDICATION` semantics, then source-backed release/wait, cache visibility, fence writes, and graphics-equeue completion. |
| Metal resources / readback | Persistent guest textures, direct private scanout targets, command-batch completion, shared staging-buffer readback, exact packed rows, MSAA resolve targets, and supported depth readback, covered by GPU tests. | Bounded asynchronous capture tickets and a reusable staging pool; explicit block-compressed and combined depth/stencil layouts when guest formats exercise them. |
| Guest virtual/direct memory | Fixed/hinted mappings, physical allocation splits, coherent aliases, strict range unmap/protect, `MAP_NO_COALESCE` identity preservation, release-time alias invalidation, shared-mapping identity tracking, and CPU write-watch retirement across remaps, all with synthetic coverage. | Mapping lifecycle counters and exact query error/flag behavior for all exercised cases. |
| Render state / pipelines | GFX10 render-target decoding, MRT attachment layout, write masks, viewport/scissor, formats, depth-target validity requirements, and translated shader pipelines with cached construction. | Complete reusable blend, depth/stencil, rasterization, culling, multisampling, resolve, and full structural pipeline-cache identity. |
| Shader translation | Typed backend-neutral Gen5 IR, direct single-path MSL lowering, scalar/vector state, exports, sampled/storage images, buffers, zero-copy global bindings, wave64 ballot handling on SIMD32, and observed compute geometry, with offline Metal compilation tests. | Complete the remaining RDNA2 instruction/resource/control-flow surface and eliminate residual per-draw Metal buffer churn without weakening guest-memory visibility. |
| Presentation / frame capture | Guest display size independent of host drawable size; opt-in headless Metal; interactive runs as a regular app bundle; direct scanout without routine GPU-to-CPU round trips; asynchronous shader compilation with strict event ordering; on-disk Metal library and pipeline caches; live title-bar performance statistics. | Resolve early post-boot output into the first real scene and use completed-presentation and guest-flip rates to measure pacing. |
| Runtime / ABI | Whole-process import auditing across HLE, guest-libc, and host routes; FIFO-ticket condition variables with exact signal/broadcast cardinality; guest mutexes and semaphores; per-thread TLS resolution; video-out attributes; save-data basics; NP service objects. | Complete opaque sync-object lifecycle, cancellation, clock attributes, and absolute timed waits; continue source-backed AGC event queues, save dialogs, and remaining researched NIDs. |
| Audio | Host output, guest port/config validation, bounded queues, resampling, and stereo downmixing. | Audit timing, backpressure, device change, pause, teardown, and A/V synchronization against real title use with deterministic offline mixer evidence. |
| Video / Bink | No compatibility claim yet. | Identify each title's actual decoder/API path, then implement exact demux/decode/color/timing semantics with legal generated fixtures. |
| Storage / save data | Read-only app filesystem access and initial save transactions; dumps are never modified. | Complete namespace, path, permission, iteration, atomicity, durability, quota, slot, dialog, and concurrency semantics with crash-consistency tests. |
| Input | Keyboard-backed pad state and opt-in deterministic input sequences. | Complete handle lifetime, disconnect, timestamps, dead zones, motion/touchpad, controller mapping, and focus semantics as titles exercise them. |
| Configuration / telemetry | Environment-controlled presentation, capture, Metal compiler, and diagnostics; bounded logarithmic log checkpoints; opt-in VM tracing; live window-title performance statistics; per-run summaries. | A versioned, documented configuration file with strict precedence and validation, plus structured persistence of the live counters. |
