# CPU/GPU Front-End Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove avoidable CPU serialization, copying, hashing, and page-protection work from the AGC-to-Metal hot path.

**Architecture:** Keep the native MSL renderer and exact guest ordering. Shorten process-wide lock ownership, schedule already-queued cross-queue producers before their consumers, share immutable register snapshots, compact shader-cache admission, and batch write-watch demotion.

**Tech Stack:** Rust 2024, Metal-rs, `Arc<[T]>`, standard-library synchronization and collections, Cargo tests.

## Global Constraints

- Do not launch a game or emulator title.
- Do not remove correctness waits; every fast path needs the existing conservative fallback.
- Preserve all unrelated dirty workspace changes.
- Use test-first red/green cycles for every production change.
- Run `remu-runtime` library tests with `--test-threads=1`.

---

### Task 1: Route AGC submission parsing outside the process-wide dispatch lock

**Files:**
- Modify: `crates/remu-runtime/src/lib.rs`
- Modify: `crates/remu-runtime/src/agc_submission/driver.rs`
- Modify: `crates/remu-runtime/src/agc_driver.rs`
- Modify: `crates/remu-runtime/src/agc_flip.rs`
- Test: `crates/remu-runtime/src/platform_independent_tests/agc_submission.rs`

**Interfaces:**
- Produces: `dispatch_unlocked_agc_submission_import(nid, arguments) -> Option<u64>`.
- Produces: completion helpers accepting an explicit `sequence: u64`.
- Consumes: existing AGC-state and EventState locks.

- [x] Add a failing route/ordering test proving all six submission NIDs use the unlocked handler and an unrelated NID does not.
- [x] Run the focused runtime test and confirm it fails because the unlocked route is absent.
- [x] Add a submission-serialization mutex, cloneable VideoOut snapshot, short dispatch-lock snapshot/update sections, and explicit-sequence publication helpers.
- [x] Route the unlocked handler before the process-wide write lock in the live import path.
- [x] Run the focused tests and the existing AGC submission tests.

### Task 2: Schedule an ingested cross-queue producer before a front wait

**Files:**
- Modify: `crates/remu-gpu-metal/src/presenter/submission/events.rs`
- Test: `crates/remu-gpu-metal/src/presenter/tests/submission.rs`

**Interfaces:**
- Produces: exact `(four_byte_cell, queue_id)` producer counts.
- Produces: a receiver decision returning one runnable producer queue for a front wait.

- [x] Add a failing receiver test with a consumer wait queued before a producer release; require the release to be returned first without blocking the consumer.
- [x] Add a failing test proving a write to the other half of the old eight-byte cell is not a producer.
- [x] Run both focused tests and confirm the old consumer-first/cell-alias behavior.
- [x] Replace the count-only eight-byte index with exact four-byte, per-queue counts.
- [x] Select one different runnable producer queue before popping a front wait, preserving both queue FIFOs and the existing fallback.
- [x] Run the focused and full gpu-metal submission tests.

### Task 3: Share and cache immutable register snapshots

**Files:**
- Modify: `crates/remu-gpu/src/resource.rs`
- Modify: `crates/remu-gpu/src/compute.rs`
- Modify: `crates/remu-runtime/src/lib.rs`
- Modify: `crates/remu-runtime/src/agc_registers.rs`
- Modify: `crates/remu-runtime/src/agc_draw.rs`
- Modify: `crates/remu-runtime/src/agc_submission.rs`
- Modify: affected test fixtures under `crates/remu-runtime` and `crates/remu-gpu-metal`
- Test: `crates/remu-runtime/src/tests.rs`

**Interfaces:**
- Produces: draw/compute fields of type `Arc<[GpuRegister]>`.
- Produces: generation-keyed `AgcGraphicsState` snapshot methods.

- [x] Add failing tests asserting two snapshots without a register mutation are `Arc::ptr_eq` and a real mutation creates a new snapshot.
- [x] Add a failing sorted-lookup test that distinguishes present and absent offsets.
- [x] Convert event payloads and fixtures from `Vec<GpuRegister>` to `Arc<[GpuRegister]>`.
- [x] Add per-register-file generations and cached snapshots; invalidate only when a value really changes.
- [x] Use binary search for runtime lookup over sorted snapshots.
- [x] Run `remu-gpu`, focused runtime, and gpu-metal compile/tests.

### Task 4: Compact shader-evaluation cache admission

**Files:**
- Modify: `crates/remu-runtime/src/shader_cache.rs`
- Test: `crates/remu-runtime/src/tests.rs`

**Interfaces:**
- Produces: `ShaderEvaluationAdmissionFilter::observe(fingerprint) -> bool`.
- Produces: one precomputed fingerprint per full evaluation key.

- [x] Add failing tests for first-observation rejection, second-observation admission, and bounded oldest-fingerprint retirement.
- [x] Extract the admission filter using only `u64` fingerprints and admission ids.
- [x] Precompute the full key fingerprint once and make the admitted map hash only that fingerprint while retaining full key equality.
- [x] Run focused cache tests and the deterministic runtime library suite.

### Task 5: Batch chronic write-watch demotion

**Files:**
- Modify: `crates/remu-memory/src/write_watch/page.rs`
- Modify: `crates/remu-memory/src/write_watch/state.rs`
- Test: `crates/remu-memory/src/write_watch/tests.rs`

**Interfaces:**
- Produces: test-only protection-call counter.
- Consumes: existing `restore_unowned_pages` coalescing.

- [x] Add a failing multi-page demotion test that asserts bounded protection calls and preserves an overlapping armed owner.
- [x] Count protection calls under `cfg(test)` without changing release behavior.
- [x] Disarm the whole demoted range and restore it once through `restore_unowned_pages`.
- [x] Run the focused and full remu-memory tests single-threaded.

### Task 6: Release build and documentation verification

**Files:**
- Modify: `Cargo.toml`
- Modify: `docs/status.md`

**Interfaces:**
- Produces: one release codegen unit with ThinLTO.

- [x] Set release `codegen-units = 1` so the Rosetta-hosted hot path is optimized as one unit.
- [x] Update status documentation with implemented fast paths and the no-live-benchmark limitation.
- [x] Run formatting, focused suites, workspace check, and clippy where the existing dirty tree permits.
- [x] Inspect the final diff for unrelated edits and report any pre-existing failures separately.
