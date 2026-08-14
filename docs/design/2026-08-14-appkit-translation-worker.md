# AppKit shell and Metal translation worker

Status: design only, 2026-08-14. Implementation is deferred until the Phase 2 integration
and evidence gate is complete and an updated base revision is approved.

This design addresses sync audit finding #1 and async audit finding #2 in
`docs/notes/architecture-audit-20260814/`. Today `MetalPresenter` is both the winit
`ApplicationHandler` and the sole owner of ordered guest translation. Consequently,
`about_to_wait` calls `receive_events`, batch flush/retirement, scanout upload, telemetry,
host UI, and `render` on AppKit's main thread. A 1024/2048-event callback can therefore
occupy that thread for seconds. The target is a dedicated worker that owns the complete
guest command/residency state, while AppKit owns only window lifecycle, input, drawable
acquisition, and the final texture-to-drawable composite.

## Invariants and non-goals

The split must preserve these existing invariants:

- `OrderedGpuEventReceiver` has exactly one consumer and every guest event is finished in
  its existing per-queue order.
- Guest submission completion is published only after all terminal CPU-visible effects of
  that exact batch have been applied.
- A flip completion is published only after both its exact guest submission and one terminal
  disposition (`Presentation`, `Superseded`, or `SurfaceLess`) are known, in flip order.
- A windowed `Presentation` is not terminal at command-buffer completion; Core Animation's
  drawable-presented callback must also have fired.
- A texture offered to AppKit is immutable until AppKit rejects it or its presentation is
  terminal. Its producer dependency is an explicit Metal shared-event fence.
- No AppKit object (`Window`, `MetalLayer`, drawable, or `ActiveEventLoop`) crosses to the
  worker. The bridge may retain the documented thread-safe `EventLoopProxy` solely to wake
  AppKit; it conveys no AppKit state. No guest command encoder, residency map, write watch,
  or ordered receiver crosses to AppKit.

This is not a second GPU scheduler, a relaxation of guest ordering, or an upload thread that
shares mutable presenter state through one large mutex. Shader compiler workers remain
independent. The headless loop already has no AppKit responsiveness problem and remains on
its current path until the windowed split is proven.

## Ownership boundary

Replace the monolithic `MetalPresenter` with two state owners and a small shared bridge.

| AppKit main thread: `AppKitShell` | Translation thread: `TranslationWorker` |
| --- | --- |
| `Window`, `MetalLayer`, winit `ApplicationHandler` and `ActiveEventLoop` use | `OrderedGpuEventReceiver` and all queue block/order state |
| presentation `MTLCommandQueue`, presentation pipelines, sampler, overlay texture | guest/per-equeue command queues, open/committed batches, shared-event timelines |
| drawable acquisition, resize/scale/vblank pacing, final composite encoding | draw/compute decode, pipeline gates, texture/global/index upload, residency and write watches |
| keyboard/focus handling, pad publication, host-UI polling and overlay upload | flip acceptance/finalization, snapshot production, buffer safety and completion joins |
| presentation receipts and AppKit lifecycle state | shader-result ingestion, guest retirement/publication, telemetry aggregation and archives |

Both sides retain a clone of the same `metal::Device`. The `metal` crate declares its owned
Objective-C wrappers `Send + Sync`, and Metal permits resource and command-queue use from
multiple CPU threads. That permission is not treated as permission to share mutable encoding
objects: every command queue, command buffer before commit, encoder, cache, and state machine
has one thread owner. Cross-thread objects are restricted to retained immutable textures,
shared events, pipeline-independent value data, and atomic presentation receipts. Add compile
time `Send` checks for every bridge payload and owner-thread assertions at the public entry
points of both state owners.

The worker's presentation product is an immutable `PresentationFrame`:

```text
frame generation and optional flip sequence
source kind, format, and source dimensions
retained private MTLTexture
optional (retained MTLSharedEvent, required value)
Arc<PresentationReceipt>
```

The texture is a private frame snapshot, not a mutable resident texture or the reusable raw
fallback texture. A bounded snapshot pool owns its allocations. A lease returns to that pool
only after the receipt is rejected/superseded or both presentation callbacks are terminal.
This makes later guest encoding unable to overwrite pixels still sampled by a drawable.

## Bounded mailboxes and backpressure

The bridge has bounded, non-blocking mailboxes; AppKit never waits for the worker and a Metal
callback never sends on a blocking channel.

- Existing guest ingress remains bounded by `remu_gpu`'s event channel. Only the worker
  consumes it.
- `AppControlMailbox` is a 32-entry queue for lifecycle commands such as `BeginShutdown`,
  `SurfaceAvailable`, and `SurfaceUnavailable`. Resize and input stay AppKit-local. One slot
  is reserved for shutdown; any other overflow is a loud fatal protocol error.
- `LatestFrameMailbox` contains at most one unsubmitted `PresentationFrame`. Publishing a
  newer frame atomically returns the older offer to the worker, which gives that flip the
  ordinary ordered `Superseded` disposition. AppKit atomically accepts an offer before it
  commits a presentation; an offer cannot be accepted and superseded by the competing side.
- Title text, redraw-needed, overlay-dirty, worker-stopped, and fatal error are coalescing
  single-value result slots, not an unbounded notification stream. Publishing any result
  sends the winit proxy one wake event.
- The private frame pool is capped initially at four textures: one latest offer and at most
  three AppKit-submitted frames. The cap is a measured constant and is reported in telemetry.

If AppKit has no drawable, it keeps only the latest offer. Later offers replace and retire
older unaccepted flips; memory does not grow with occlusion. If all frame leases are accepted
and in flight, the worker stops before the next operation that requires a snapshot and parks
on the presentation-receipt waker. It may retire unrelated completed guest work first, but it
must not busy-spin or let a later flip overtake the blocked snapshot. This is explicit
presentation backpressure, not a reason to run translation on AppKit.

Mailbox implementations use `Mutex` only for the short swap/pop operation and a monotonic
generation plus `Condvar` for the worker. They never hold a bridge lock while calling AppKit,
Metal, guest memory, or completion publication. Every capacity and replacement is covered by
a deterministic test.

## Worker loop and wakeups

The translation thread is named `remu-metal-translation`. Each iteration, inside an
autorelease pool, performs bounded work in this order:

1. drain AppKit control messages and presentation receipt transitions;
2. ingest shader results and retire terminal guest batches/snapshots;
3. publish any newly joined guest submissions and flips;
4. process ordered GPU events until its event/time budget, a queue block, or presentation
   backpressure; and
5. opportunistically flush guest command batches and publish coalesced telemetry/results.

The budget remains to prevent completion/control starvation, but no longer limits AppKit
callback duration. If work is immediately runnable the worker loops. Otherwise it captures a
wake generation, rechecks all predicates, and parks on its condition variable. Guest batch
`addCompletedHandler`, shader-result publication, and presentation receipt transitions signal
that waker. Existing guest ingress has no compatible notifier, so the first migration keeps a
short timed poll only for new GPU events; it happens on the worker, never AppKit. A later
channel change may signal the same waker without changing this design.

The existing command-buffer completion handler must stop sending AppKit's `EventLoopProxy`.
It signals the worker waker. AppKit is woken only for a new/coalesced result, vblank, input,
or lifecycle event. `about_to_wait` therefore selects `Wait`/`WaitUntil` from redraw, UI expiry,
and shutdown deadlines; it never calls `receive_events`, flushes a guest batch, or examines
guest backpressure.

## Frame and flip protocol

The worker remains the sole owner of `flip_completion_joins`, safe-buffer sequencing, guest
submission publication, and `record_gpu_flip_completed`.

1. The worker accepts a flip and records its exact submission join as today.
2. At the same ordered boundary as today, it copies the selected translated or raw scanout
   into a leased private frame texture. The frame carries the producer queue's shared event
   and value. Raw guest bytes are decoded and uploaded on the worker; AppKit performs no
   guest-memory reads or 8.3 MB `replace_region` copy.
3. The worker publishes the offer only after all metadata is fixed. Publication does not
   claim the flip's terminal side.
4. Immediately before encoding, AppKit changes the receipt atomically from `Offered` to
   `Submitted`. If a worker replacement already changed it to `Superseded`, AppKit drops the
   stale offer without presenting it. If AppKit wins, the worker observes `Submitted`, calls
   the existing terminal claim exactly once, and will never supersede that receipt.
5. The presentation command buffer waits on the frame's shared event, composites into the
   drawable on AppKit's presentation queue, and installs both a command-buffer completion
   handler and drawable-presented handler. The handlers update atomic fields in the receipt
   and wake both the worker and AppKit as needed; neither mutates presenter state.
6. The worker marks `Presentation` terminal only when the command buffer completed and the
   drawable-presented bit is set. A command-buffer error is logged and becomes terminal under
   the current recovery policy. Submission-first and presentation-first races still converge
   through the existing ordered join.

No-drawable is not a terminal result; the offer remains the latest retryable frame. A newer
offer may supersede it. A truly surface-less configuration uses a worker-owned tail command
buffer and the existing `SurfaceLess` disposition, never a fabricated AppKit acknowledgement.

## AppKit, UI, and lifecycle

`resumed` creates the window, layer, presentation pipelines, and presentation queue before it
starts the translation worker. Startup failure before that point returns directly without a
worker. `window_event` keeps resize, scale-factor, focus, and keyboard handling. It publishes
pad state directly and requests composites from the latest frame. `poll_host_ui` and overlay
texture upload remain AppKit-local because they feed only the final composite.

Guest dialog/toast events are processed in ordered worker ingress using the existing
thread-safe host-UI state, then set `overlay-dirty` and wake AppKit. Splash image texture
creation and upload move to the worker and are offered through the same immutable frame
protocol. Window titles become worker-produced telemetry snapshots: logging and counter
reset occur on the worker, while AppKit only applies the latest title string.

Introduce an explicit lifecycle state machine:

```text
Starting -> Running -> ClosingGuest -> DrainingPresentation -> Stopped
                    \-> Fatal -------------------------------/
```

Window close, guest `Shutdown`, event-channel disconnect, and a worker fatal all enter the
same shutdown path. `ClosingGuest` stops new ingress, rejects an unaccepted frame, flushes and
retires guest batches, and persists the pipeline archive. `DrainingPresentation` lets AppKit
finish already-submitted receipts and return their private leases. AppKit exits the winit loop
only after the worker publishes `Stopped`; `run_presenter` then joins the worker thread.

A worker panic is caught at the thread boundary, stored in the non-droppable fatal slot, and
wakes AppKit. An AppKit-side protocol or presentation error sends `BeginShutdown`. Shutdown
has a logged deadline and prints every outstanding flip/receipt/batch if exceeded; it never
silently abandons a pacing token. Process termination may then fail loudly rather than hang.

## Instrumentation

Keep the existing `[REMu][PERF-EVENT-LOOP]`, translation-term, slow-command-buffer, and
completion-order records, but annotate presenter-loop records with `thread=translation`.
Announce the mode and both thread IDs once at startup.

Add a five-second `[REMu][PERF-PRESENTER-SPLIT]` record containing:

- worker wall/busy/park time and decode, encode, upload, retirement, and publication time;
- AppKit callback/composite CPU time, maximum callback time, drawable misses, and vblank wakes;
- guest ingress depth, runnable/blocked queues, command batches, control depth, frame-slot
  occupancy, frame-pool in-flight/high-water, and receipt depth;
- frames produced, replaced, accepted, submitted, command-completed, presented, errored, and
  surface-less; and
- accept-to-ready, ready-to-submit, submit-to-command-complete, and submit-to-present latency.

The worker progress sequence and AppKit heartbeat are sampled by the existing cheap in-tree
telemetry. A stalled sequence prints the last ordered event class, queue, frame generation,
and outstanding dependency. This preserves visibility inside the Rosetta process without
requiring a host profiler.

## Tests and acceptance gates

Protocol tests use fake frame resources and deterministic barriers to cover:

- AppKit acceptance racing worker replacement, with exactly one winning disposition;
- a full latest-frame slot, four leased frames, occlusion, and wake-without-busy-spin;
- command completion before drawable presentation and the reverse order;
- presentation error, surface loss, shutdown at every state, panic propagation, and mailbox
  overflow diagnostics;
- submission-first/presentation-first flip joins, superseded flips behind outstanding guest
  work, and no completion before exact batch publication; and
- immutable frame reuse: a later worker snapshot cannot alter pixels held by an in-flight
  presentation.

Metal integration tests encode on a worker-owned queue and sample on the AppKit-owned queue
through a shared-event wait, compare pixels, and overlap at least two frames to prove queue
and lease ordering. Thread-affinity tests record callback thread IDs and prove decode/encode,
raw upload, retirement, and telemetry never execute on the AppKit thread. Existing headless,
present, submission, texture, and completion tests remain valid.

Every implementation commit must pass, serially in release mode:

```text
cargo test -p remu-gpu --release -- --test-threads=1
cargo test -p remu-gpu-metal --release -- --test-threads=1
cargo test -p remu-memory --release -- --test-threads=1
cargo test -p remu-runtime --release -- --test-threads=1
```

`remu-gpu-metal` and `remu-memory` must have zero failures. `remu-runtime` may have only the
12 named deterministic baseline failures; no new failure or changed failure is accepted.
The audit baseline is `remu-gpu-metal` 504/0, `remu-memory` 93/0, and `remu-runtime`
665 passing/12 failing. The exact, higher pass counts are recorded again from the Phase 2
integration base before implementation and may not decrease.

After the suites pass on a clean tree, run the prescribed 280-second PPSA10112 harness with
`REMU_SHARED_EVENT_ACK_PHASES=1`, captures, and the fixed auto-Cross schedule. Record
`vm.loadavg`, compare the split disabled/enabled at the same revision, and inspect guest log,
telemetry, and frames. Acceptance requires:

```text
REMU_SLOW_CB_MS=150 \
REMU_CAPTURE_FRAMES=1 \
REMU_CAPTURE_INTERVAL_SECONDS=2 \
REMU_AUTO_CROSS=12,20,28,0,108,116 \
REMU_AUTO_CROSS_HOLD_MS=1000 \
REMU_AUTO_CROSS_REPEAT_MS=0 \
REMU_SHARED_EVENT_ACK_PHASES=1 \
scripts/run-game.sh \
  "/Volumes/Untitled/sharpemu/games/SILENT HILL The Short Message/PPSA10112-app/eboot.bin" \
  280
```

- the crawl corridor's `[REMu][PERF-EVENT-LOOP]` draw-event-seen rate to sustain the requested
  10+ fps rather than the prior roughly 1.4 fps;
- a captured rooftop NEW GAME menu frame with the expected SILENT HILL/logo/menu content;
- no AppKit callback containing guest decode/translation/encode work and no long main-thread
  callback attributable to the worker path;
- no flip-completion, presentation-order, frame-pool, or mailbox invariant error; and
- the PPSA02929 regression title reaching its integration-base known-good state.

Evidence runs obey the repository rule: no edits while the harness is in flight and no
interpretation of fps under a busy host without recording the load.

## Migration, rollout, and rollback

Implement as separately revertible commits after the Phase 2 gate:

1. add the protocol types, bounded mailboxes, receipts, fake-resource tests, and telemetry;
2. split `MetalPresenter` into `TranslationCore` and `AppKitShell` while invoking both on the
   main thread, proving ownership without changing scheduling;
3. make scanout products private immutable frame leases and add cross-queue Metal tests;
4. start `TranslationWorker` behind `REMU_TRANSLATION_WORKER=1`, move guest completion and
   compiler wakeups, and keep legacy as the default;
5. move raw upload, flip joins, telemetry/title production, and host-UI invalidation across
   the bridge; add the unified shutdown state machine;
6. run release suites and A/B title evidence, then enable the worker by default in its own
   commit while retaining `REMU_TRANSLATION_WORKER=0` for one release cycle; and
7. delete the legacy main-thread translation path only after the rollback window and both
   title gates remain green.

The flag is resolved once before either owner starts and is announced. It is not read from a
hot path. Rollback changes scheduling only; frame/flip ordering tests and the private frame
lease remain shared by both modes so disabling the worker cannot restore an unsafe mutable
scanout handoff.
