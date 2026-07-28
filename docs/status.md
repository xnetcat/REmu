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

## Open defects

These are known failures with enough recorded state to be worked on. A defect is listed here
rather than worked around; a title-specific patch that hides one is not an acceptable substitute.

### Guest object storage goes stale mid-run

A guest object's storage is observed zeroed while the guest still holds a live reference to it.
Five distinct consumer sites were recorded faulting on the result, all in one Unreal-based title:

- A graphics pipeline state object is stored and referenced, then its mandatory `+0x18` child
  reads back as null before submission. The producing constructor guarantees that child is
  non-null, so this is not an object that merely needs more construction time.
- A queued render task reads back with a zero vtable pointer.
- A container slot that the guest's own iteration treats as populated reads back null.
- A per-field handler pointer reads back null at a virtual call site.
- A parent object is destroyed by a nested update, after which the caller reloads its vtable
  pointer and finds it zeroed.

All five share one shape: memory the guest wrote and still owns does not read back with the value
it wrote. That points at a REmu defect in memory lifetime, aliasing, write visibility, or an HLE
call returning a value that makes the guest free storage early. It is not a guest defect, and the
same five sites are the evidence for whichever of those turns out to be the cause.

`scripts/lldb_allocator_map_watch.py` and `scripts/lldb_allocator_pop_watch.py` observe the
allocator during this failure and are the current starting point.

This was previously masked by a compatibility guard that patched the five consumer sites in the
guest executable so they tested for the bad value and skipped their work. The guard was removed
because it was locked to hardcoded addresses and byte signatures from one build of one title, and
because suppressing the symptom removed the pressure to find the cause. The affected title is
expected to fail at startup until the defect is fixed.

### A run intermittently renders nothing

Roughly every other launch of the working test title produces `FPS 0.0`, `Draw 0/s` and
`Tex 1` for the whole run while audio continues at its normal rate, then exits on the
diagnostic timer with no error. The next launch of the same binary renders normally.

This is not caused by any recent change: it reproduces at `e57a6f1`, before the per-game
fixes were removed and before any file was split. Audio running normally while no draw is
ever submitted points at display or scanout setup losing a race during startup, not at the
guest failing.

Frame rate is also not stable across successful runs of one binary: 59.9, 61.9, 87.9,
91.9, 106.0 and 119.9 FPS have all been observed at the same revision, depending on window
and display state. Neither frame rate nor draw rate can therefore be used to detect a
performance regression. There is no CPU-throughput benchmark in the tree, which is the gap
to close before any performance claim can be made.

### The runtime test suite is order-dependent

`cargo test -p remu-runtime --lib` fails intermittently, at roughly a one-in-three rate, in
`agc_driver::tests::a_graphics_submission_parses_its_stream_and_requests_a_completion_event`. It
passes reliably with `--test-threads=1`.

`remu_gpu::install_event_sink` writes a process-global sink. Tests that assert on emitted GPU
events take `GPU_EVENT_SINK_TEST` to serialize, but 121 tests reach the import dispatcher without
holding it, so an event emitted by one test can be delivered to another test's receiver.

Until this is fixed, verification runs must pass `--test-threads=1`, which makes the suite
deterministic. The fix is either to make every test that can emit take the guard, or to route test
events through the existing thread-local capture instead of the process-global sink.

## Subsystem table

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
