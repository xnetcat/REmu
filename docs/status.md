# Subsystem status

This records what each subsystem actually implements and its next concrete gap. A splash screen,
a timeout, or the absence of `unimplemented` log lines is never treated as a compatibility claim;
completion means stable interactive game progress with correct output and measured performance.

Current overall state: the first small test title boots through resource loading, submits AGC
draw streams, compiles observed RDNA2 shaders to Metal, and flips its display buffers. Early
splash frames render correctly; full scene rendering and playability are still open. A second,
Unity/IL2CPP title (which commits nearly all of its memory by reserving virtual ranges and
`sceKernelMprotect`-committing them rather than mapping direct memory) now gets resident,
write-watched textures; its environment backdrop, lighting, and menu UI now render with the
correct orientation. Lighting and post-processing are improved but not yet visually correct.

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
because suppressing the symptom removed the pressure to find the cause.

After commit-on-protect landed (2026-07-28), the startup crash no longer reproduces: two
consecutive 45-second runs render real scenes (22–26 FPS, ~40 resident textures, heavy
GPU-to-guest writeback traffic) and end only on the diagnostic timer, with no faults logged.
That is consistent with this defect living in the same layer the fix repaired — guest objects
in reserve-then-`mprotect`-committed memory were invisible to every shared-backing path, so
GPU-to-guest writebacks into such storage could not land — but it is not proof. The five
recorded consumer sites remain the checklist: this entry closes when sustained interactive
progress shows none of them faulting, not because the crash stopped appearing in short runs.

### The Bluepoint title idles at startup without its fiber ABI

A third title (Bluepoint's BPE engine) previously aborted about three seconds into startup, inside
its own `PS5PlatformPageAllocator`. Three unserved imports caused it, each one a call whose result
the engine consumes directly:

- `sceKernelAvailableDirectMemorySize` (`C0f7TJcbfac`) left both output parameters untouched, so
  the engine read an available size of zero, computed `0 - trackingOverhead` as its pre-allocation
  size, and failed its own allocation verify at `PS5PlatformPageAllocator.cpp:383`.
- `sceKernelAvailableFlexibleMemorySize` (`aNz11fnnzi4`) failed the same way at line 450.
- `scePthreadRwlockRdlock`/`Wrlock` (`Ox9i0c7L5w0`, `mqdNorrB+gI`) were the only members of the
  console-named reader/writer lock family not routed to the implementation the POSIX-named
  spellings already used, so Dantelion2's `DLReadWriteLock` took a not-implemented return as a
  fatal error rather than as contention.

With those served, the title reaches its own subsystem initialization, opens the presenter, and
runs to the diagnostic timer without faulting. Its job system then stalled because it is built on
the fiber ABI, which was unserved; `sceFiberRun` returning not-implemented meant the job driving
startup never ran and the title blocked in `sceKernelWaitEventFlag` for the rest of the run.

The fiber entry points are now implemented (`fiber.rs`), and a live run exercises the whole
lifecycle without error: `sceFiberOptParamInitialize`, `_sceFiberInitializeImpl`, `sceFiberGetInfo`,
`sceFiberRename`, `sceFiberRun`, `sceFiberReturnToThread`, and later `sceFiberSwitch`. The job
system now runs and reaches asynchronous asset loading, resolving and reading
`/app0/playgo-chunkdefs.xml` through the AMPR command-buffer path.

Two further defects were then found and fixed, both by tracing rather than inspection:

- `sceAmprCommandBufferWriteAddressOnCompletion` was unserved. It is the polled half of the APR
  completion contract — a guest that registers no event queue watches a completion word instead —
  so the buffer was submitted and its transfer performed while the word the guest spun on was
  never written. It is now recorded like the queue-signal command and published on submission,
  before any event is announced, so a guest woken by the queue already observes the word.
- The fiber implementation resolved its per-thread current-fiber slot in a way the compiler could
  hoist across a context switch. That is sound for ordinary code, where a function cannot change
  threads midway, but a fiber can: this title parks a fiber on one worker and lets another resume
  it. The resumed frame then read the *suspending* thread's slot, which by then was cleared, and
  reported `SCE_FIBER_ERROR_PERMISSION` back into the job system. Forcing the resolution to happen
  after the resume fixed it, and a test that resumes a fiber on a second thread covers it.

With those, fiber traffic runs clean — over two thousand transitions in a 45-second run with no
refused call — and the title reaches its Bink video threads and its `Core.Res` decompressor,
file-loader, and task-manager pipeline.

The title now runs to roughly ninety thousand imports. It configures its display — 1920x1080
through `sceVideoOutSetBufferAttribute2` and `sceVideoOutRegisterBuffers2` — and brings up audio.

`sceAcmContextCreate`/`Destroy` are served: an ACM context is an opaque handle the guest hands back
to the batch entry points, so issuing a distinct live id is that call's whole contract. The
signal-processing batch entry points stay unimplemented. This removed a spurious failure but was
not a blocker: the engine creates a context, immediately destroys it, unloads the module, and
finalizes Ajm either way, which reads as a capability probe rather than a failed path.

It is still not playable and no compatibility is claimed: it submits no draws, and its `Core.Res`
threads sit in condition waits having never been given work. Why they are never fed is not
established. The remaining unserved imports it reaches are `sceSystemGestureOpen` and
`sceSystemGestureInitializePrimitiveTouchRecognizer`, `sceVideoOutColorSettingsSetGamma`,
`sceKernelSetPrtAperture`, `sceNetCtlInit`, and one NID (`D-CzAxQL0XI`) with no recovered name.
None of them has been shown to be what the resource threads are waiting on, so serving them is not
known to be the next step; tracing what feeds those threads is.

The fiber implementation is x86-64 only, because it saves and restores x86-64 register state. On
any other host the NIDs fall through to the unimplemented path rather than to a stub, so its tests
run under `--target x86_64-apple-darwin`. The attach-and-run/switch variants and the
context-size-check pair stay unserved until a title needs them.

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

### Write-watch arming assumes plain read/write data pages

`CpuWriteWatch` arming and disarming reconstruct protection as read-only and read/write
respectively (`write_watch/page.rs`), never consulting the mapping's real protection. Watching a
read-only range therefore grants writes after the first fault instead of restoring read-only,
and arming a watch whose page-aligned span touches executable pages strips execute permission
permanently. This predates the committed-memory work — read-only direct-memory views were
already registrable — but commit-on-protect widens the reachable surface to any committed
protection. Neither regression title exercises it: every range the texture cache watches today
is plain read/write. The fix shape is to carry the mapping's original protection into
arm/disarm (write-watch arming becomes "remove write, keep the rest"), or to refuse watch
registration for non-read/write ranges explicitly.

### The Unity title's menu runs at 14–22 FPS

With FMOD's AudioOut2 port accepted, the Unity title reaches its interactive menu. Three
successive bottlenecks were profiled on its ~600-draw frames; two are fixed and one remains:

- Fixed: `scan_valid_command_stream_end` proved readability of every candidate command prefix by
  allocating and copying the whole span (up to 4 MiB per candidate per submission). It now uses
  the non-copying mapped-range check; this alone took the title from 4 to ~14 FPS.
- Fixed: per-face shadow-depth writebacks (see the Metal resources row) — `alias_writeback`
  dropped from 36/s (72 MiB/s) to zero.
- Remaining: a mach-exception storm keeps the Rosetta exception server saturated. Disabling CPU
  write watches lifts the title to ~20 FPS, so watch faults are roughly 40% of the remaining
  frame time — Unity co-locates hot mutable data on the same 16 KiB pages as watched texture
  bytes, so unrelated guest stores fault continuously. The fix direction is fault-rate reduction:
  registry/watch coalescing, and demoting chronically dirty textures from fault-based watching to
  acquire-time re-validation. A further CPU-side layer remains beneath that (~20 FPS with watches
  off, GPU under 30%).

The scene-orientation and white-overlay defects are fixed:

- GFX10 `SW_4KB_S` textures now use AddrLib-compatible packed mip chains: the mip-tail occupies
  the first 4 KiB, non-tail levels follow in reverse level order, and array/cube slices use the
  full chain stride. The previous linear mip offset treated tail bytes as mip zero, producing
  mirrored/upside-down scene regions and duplicate light fixtures.
- `CB_COLOR_CONTROL` special modes are decoded. Resolve, eliminate-fast-clear, and FMASK
  decompression packets end the current Metal render pass without rasterizing their attached
  dummy shader. Rasterizing Superliminal's resolve trigger as an ordinary constant-white draw
  caused the washed-out frame.
- GCN cube-resource face-atlas coordinates are lowered to Metal 2D-array sampling rather than
  being passed to direction-based cube samplers.

A live run without diagnostic bypasses now presents the warehouse with the correct orientation,
one physical light fixture, and readable menu text. Lighting, shadows, and post-processing still
do not fully match the intended scene. Performance also remains defective, particularly CPU
write-watch fault volume.

### The Unity title's intro video remains paused after its first frame

The retro-TV intro (60 fps 1440x1080 H.264 inside `sharedassets2.resource`) decodes through the
ffmpeg pipe backend. AvPlayer now follows shadPS4's source-backed pause contract: Pause preserves
an active session but freezes `CurrentTime`, and both data exports return no frame until Resume.
The earlier bounded paused-preroll experiment was removed after live tracing showed that it
advanced stream positions while Unity was paused and prevented the expected state transition.
Start and Resume now use separate state transitions; JumpToTime restarts the decoders with an
input-side `-ss` seek, and `sceAvPlayerChangeStream`/`sceAvPlayerSetTrickSpeed` cover the observed
single-representation and forward-rate cases. `REMU_TRACE_AVPLAYER=1` counts every zero-return by
reason. A live automated pass now reaches Ready, Start, and the first decoded video frame, after
which Unity calls Pause and issues no later Resume in that input path. No intro frame is presented
to the user, so the event/state transition between preparation, pause, and visible playback remains
an open defect.

### Gameplay after NEW GAME remains black

The loading screen completes, but the following in-game view remains black. UI elements can still
appear and the renderer continues submitting work with no failed pipelines, so this remains a
scene-content, synchronization, or post-processing defect rather than a completed compatibility
milestone.

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
| Metal resources / readback | Persistent guest textures backed by either direct-memory views or committed (reserve-then-commit / flexible) guest memory, direct private scanout targets, command-batch completion, shared staging-buffer readback, exact packed rows, AddrLib-compatible GFX10 `SW_4KB_S` packed mip/array/cube upload, MSAA resolve targets, supported depth readback, and GPU-side assembly of layered depth-sample views (shadow cubemaps/cascades rendered as per-face 2D depth targets and bound through one layered SRD are blit-assembled into a cached depth-format array inside the guest batch instead of round-tripping every face through guest memory each frame). Covered by GPU tests. | Bounded asynchronous capture tickets and a reusable staging pool; explicit block-compressed and combined depth/stencil layouts when guest formats exercise them. |
| Guest virtual/direct memory | Fixed/hinted mappings, physical allocation splits, coherent aliases, strict range unmap/protect, `MAP_NO_COALESCE` identity preservation, release-time alias invalidation, shared-mapping identity tracking, CPU write-watch retirement across remaps, and commit-on-protect reserve+commit semantics: `sceKernelMprotect` over a reserved range and flexible mappings commit zero-filled memory into a virtually-keyed shared backing, so GPU residency and write-watching resolve them exactly like direct-memory views, decommit zeroes pages before release, and guest-visible `VirtualQuery` categories are unchanged. The backing is keyed by virtual placement — a deliberate scope decision: this memory class never exposes a physical address to the guest, so physical aliasing of it is unrepresentable by construction, unlike direct memory, which keeps its own physically-keyed backing object. All with synthetic coverage. | Coalesce adjacent contiguous shared-registry records at registration: commit-heavy titles insert one record per `mprotect`, and resolution currently scans records linearly per watch creation and arms one watch segment per record. Mapping lifecycle counters and exact query error/flag behavior for all exercised cases. Memory-pool NIDs (`sceKernelMemoryPoolReserve`/`Commit`/`Expand`/`Move`/…) stay loudly unimplemented until designed: pools deal in guest-physical block addresses (`Expand` returns one, `Move` re-points committed blocks without copying), so they need physically-keyed backing and must not silently reuse the virtually-keyed commit path. |
| Render state / pipelines | GFX10 render-target decoding, MRT attachment layout, write masks, viewport/scissor, formats, depth-target validity requirements, `CB_COLOR_CONTROL` resolve/eliminate-fast-clear/FMASK-decompression command handling, and translated shader pipelines with cached construction. | Complete reusable blend, depth/stencil, rasterization, culling, remaining multisampling behavior, and full structural pipeline-cache identity. |
| Shader translation | Typed backend-neutral Gen5 IR, direct single-path MSL lowering, scalar/vector state, exports, sampled/storage images, buffers, zero-copy global bindings, wave64 ballot handling on SIMD32, and observed compute geometry, with offline Metal compilation tests. | Complete the remaining RDNA2 instruction/resource/control-flow surface and eliminate residual per-draw Metal buffer churn without weakening guest-memory visibility. |
| Presentation / frame capture | Guest display size independent of host drawable size; opt-in headless Metal; interactive runs as a regular app bundle; direct scanout without routine GPU-to-CPU round trips; asynchronous shader compilation with strict event ordering; on-disk Metal library and pipeline caches; live title-bar performance statistics. | Resolve early post-boot output into the first real scene and use completed-presentation and guest-flip rates to measure pacing. |
| Runtime / ABI | Whole-process import auditing across HLE, guest-libc, and host routes; FIFO-ticket condition variables with exact signal/broadcast cardinality; guest mutexes and semaphores; per-thread TLS resolution; video-out attributes; save-data basics; NP service objects. | Complete opaque sync-object lifecycle, cancellation, clock attributes, and absolute timed waits; continue source-backed AGC event queues, save dialogs, and remaining researched NIDs. |
| Audio | Host output, guest port/config validation, bounded queues, resampling, stereo downmixing, and the complete defined AudioOut2 format lattice (float/i16 × mono/stereo/7.1 × standard order), which is what FMOD titles open. Twelve-channel layouts stay explicitly rejected until the mixer models them. | Crackling from 15–17% audio underruns under heavy import-thread contention on command-heavy titles; audit timing, backpressure, device change, pause, teardown, and A/V synchronization with deterministic offline mixer evidence. |
| Video / Bink | Initial AvPlayer MP4/H.264 and audio decode through ffmpeg pipes, stream enumeration, play/pause/resume/seek/loop/trick-speed state, media-clock frame gating, NV12 buffers, and deferred event callbacks, covered by generated fixtures. | Present the Unity title's intro frames and resolve its preparation/pause transition; Bink has no compatibility claim yet. |
| Storage / save data | Read-only app filesystem access and initial save transactions; dumps are never modified. | Complete namespace, path, permission, iteration, atomicity, durability, quota, slot, dialog, and concurrency semantics with crash-consistency tests. |
| Input | Keyboard-backed pad state and opt-in deterministic input sequences. | Complete handle lifetime, disconnect, timestamps, dead zones, motion/touchpad, controller mapping, and focus semantics as titles exercise them. |
| Configuration / telemetry | Environment-controlled presentation, capture, Metal compiler, and diagnostics; bounded logarithmic log checkpoints; opt-in VM tracing; live window-title performance statistics; per-run summaries. | A versioned, documented configuration file with strict precedence and validation, plus structured persistence of the live counters. |
