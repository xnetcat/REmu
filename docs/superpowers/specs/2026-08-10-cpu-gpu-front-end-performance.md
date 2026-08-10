# CPU/GPU Front-End Performance Design

## Goal

Raise steady-state frame throughput by removing repeated CPU work and avoidable synchronization
from the path that translates AGC submissions into Metal work. Preserve guest ordering and GPU
resource correctness. Do not replace native translated Metal shaders with a permanent GPU ISA
interpreter.

## Evidence and constraints

- Existing telemetry often reports only 8–30% Metal utilization while the guest produces low FPS.
- Same-queue `WAIT_REG_MEM` is already elided frequently, but a remaining cross-queue wait can be
  dequeued before its queued producer and force a command-buffer flush.
- AGC command parsing and shader resource evaluation currently run while the process-wide import
  dispatch lock is held, delaying unrelated guest services.
- Draw and compute events own three `Vec<GpuRegister>` snapshots. Cloning events and pipeline
  candidates repeatedly copies those snapshots.
- Shader evaluation's two-hit admission queue retains complete 64-word cache keys for states that
  are usually seen once.
- Chronic write watches already demote to revalidation, but demoting a multi-page range restores
  pages one at a time through a registry-wide ownership scan.
- No game launch is permitted for this work. Correctness is verified by focused tests and the
  existing deterministic test suites.

## Design

### 1. Short process-wide lock ownership

Route the six live graphics/compute submission imports before the process-wide dispatch lock.
Serialize submission translation with its own mutex, snapshot VideoOut state and the next sequence
under a short dispatch-lock section, parse under the existing AGC lock, then publish completion
bookkeeping under another short section. Event queues already have their own lock; publication
helpers will accept an explicit submission sequence and a VideoOut snapshot instead of requiring a
mutable `ProbeDispatchState` for the entire parse.

The old stateful entry points remain as test/compatibility wrappers. There is one parser and one
completion implementation, not parallel semantics.

### 2. Producer-first cross-queue scheduling

Index queued small writes by exact four-byte cell and guest queue. Before returning a front
`WAIT_REG_MEM`, the ordered receiver checks whether a different runnable queue already contains a
matching producer. It emits producer-queue events first while preserving each queue's FIFO. Once
the matching release/write is emitted, the consumer wait resumes and uses the exact shared-event
checkpoint without first flushing its open Metal command buffer.

This optimization never reorders work within a guest queue and never treats a queued value as
completion. Missing, same-queue, wide, blocked, or host-produced dependencies retain the existing
blocking fallback.

### 3. Shared immutable register snapshots

Change draw and compute register snapshots to `Arc<[GpuRegister]>`. Track register-file generations
in `AgcGraphicsState` and reuse an existing immutable snapshot until that file changes. Draw/event
clones and speculative pipeline candidates then clone three `Arc`s instead of copying register
vectors. Register lookup over these sorted snapshots uses binary search.

### 4. Compact shader-evaluation admission

Compute one content fingerprint when an evaluation key is constructed. Hash-map lookups hash that
fingerprint and retain full equality for admitted entries. The two-hit ghost/admission queue stores
only `(fingerprint, admission_id)`, not duplicate complete keys. A collision can at most admit an
extra full key; it cannot return the wrong evaluation because the main cache still compares the
complete key.

The evaluator's existing program-level jump table and liveness data remain the static resource
access plan. This change removes redundant identity work around it without introducing a second
shader interpreter.

### 5. Batched write-watch demotion

When a chronic multi-page watch demotes, disarm all of its pages and reuse the existing coalesced
`restore_unowned_pages` operation. This replaces one registry scan and `mprotect` per page with one
ownership pass and maximal contiguous protection ranges. Read watches and non-demoted precise
watches are unchanged.

### 6. Verification and observability

Add focused behavioral/performance-contract tests for producer-first ordering, exact producer
indexing, compact two-hit admission, shared register snapshot reuse/invalidation, sorted snapshot
lookup, and bounded demotion protection calls. Run runtime tests single-threaded because the
existing global event-sink tests are documented as order-dependent. Run the Metal and shader suites
without launching a title.

## Non-goals

- No permanent GPU RDNA2 interpreter.
- No removal or speculative satisfaction of guest synchronization.
- No Metal indirect-command-buffer migration in this patch; arbitrary pipeline state must still
  be created by the CPU, and the measured front-end waste can be removed without that risk.
- No visual/FPS claim without a later clean, low-load game benchmark.

