# Phase 1H handoff — GPU SYNC COMPLETENESS (2026-08-06) — **CLOSED**

**Branch:** `phase1h-gpu-sync` @ tip before merge (worktree `/Users/xnetcat/Documents/REmu-phase1h`)  
**Base:** `main` @ `8ed20de`  
**Code tip for live matrix:** `f8bdad5`  
**Merged to main** @ `6f503df`. Phase 1 A–H complete.

## CLOSED — what “closed” means (loud anti-claims)

SDK `sync.cpp` data flow for RELEASE_MEM / WAIT_REG_MEM / CB·DB writeback / `gpuSyncDma`,
plus W712 cross-queue fence DMA on the GPU timeline and consumer-aware ACQUIRE demotion, is
landed. Phase 1 A–H architectural invariants are complete on `main` after merge.

**Anti-claims (do not invent success):**

- Silent Hill `captured_frames=0` — present path runs (flips ≥2048) but PPM UI-garbage /
  3D-scene closure remain **unassessable**. No FPS claims (SH launch `vm.loadavg` 1m was 5.17).
- Live SH `PERF-SHARED-EVENT-DMA-FILL` stayed at `attempts=0/s` this revision — the rate meter
  did **not** sample accepted cross-queue fills. W712 closure is architectural + unit-tested
  (foreign missing checkpoint → `IncompleteDependency`; `CrossQueue` reject **deleted**), not a
  claim that SH’s historic Q W712 convoy counter was re-measured live.
- Dreaming Sarah / Void Terrarium null-SIGSEGV before `InstallExceptionHandler` — known, not 1H.
- Demon’s Souls still ACM-blocked (`sceAcmBatchStartBuffers`); this tip also exited 139 after
  ACM traffic with host SIGSEGV at nonsense addresses — **not** claimed as a 1H regression or fix.

## Design (SDK `sync.cpp`)

| Piece | Behavior |
| --- | --- |
| **RELEASE_MEM** | Decodes action / GCR / cache policy / interrupt; `GpuMemoryRelease::label` + Phase 1C release publication; CB/DB end-of-pipe actions close the Metal render pass before the label write. |
| **CB/DB writeback** | EventType `0x16/0x2A/0x2C/0x2E/0x31` → `GpuEvent::RenderBackendFlush` (no longer dropped). |
| **WAIT_REG_MEM** | WaitOperation + cache policy; PFP stalls close open encoders; ME may leave the parser ahead. |
| **gpuSyncDma** | `DmaDataDst::kNowhere` + size 0 → `DmaSyncBarrier` (close encoders + CP flush). |
| **Cache / label visibility** | Builds on Phase 1C release stores; RELEASE cache actions flush RB before publication. |
| **W712 DMA fences** | Cross-queue immediate fence fills look up producer signal checkpoints and encode Metal waits; fence-shaped zero fills signal shared events like RELEASE_MEM. `SameQueueDmaFillReject::CrossQueue` deleted. |
| **ACQUIRE demotion** | Fingerprints already gone (1G). ACQUIRE marks `acquire_invalidated`; demote at consumer `reclaim_for_bind` (sampled still evicts). |

## Commits

| Commit | Concern |
| --- | --- |
| `a0442a0` | gpu: SDK sync event types (RELEASE/WAIT/CB-DB/DmaSync) |
| `a4508af` | runtime: decode + emit those packets |
| `f8bdad5` | gpu-metal: CB/DB flush, WAIT engine, W712 DMA, ACQUIRE demotion, delete CrossQueue |
| *(docs)* | this handoff + status/MEMORY |

## Safeguards deleted

- `SameQueueDmaFillReject::CrossQueue` (and its PERF `reject=cross-queue:` bucket) — foreign
  producers are GPU-timeline waits or `IncompleteDependency` when the checkpoint is missing.
- Eager ACQUIRE resident demotion that walked unused textures (REPORT fingerprint waste loop);
  demotion is consumer-bind deferred.

## Checks

- `cargo test -p remu-gpu --lib -- --test-threads=1` — 47 passed
- `cargo test -p remu-runtime --lib platform_independent_tests::agc_packets::decodes_ -- --test-threads=1` — 6 passed
- `cargo test -p remu-gpu-metal --lib -- --test-threads=1` — 476 passed, 5 ignored
- `cargo check --workspace --all-targets` — arm64 + `x86_64-apple-darwin` green

## Live matrix @ `f8bdad5` (clean tree)

| Title | Outcome | Notes |
| --- | --- | --- |
| Silent Hill (primary, 90s) | external-watchdog-timeout / 124 | flips≥2048; frames=0; DMA-FILL attempts=0; host dma=0; loadavg launch 5.17 — no FPS |
| Demon’s Souls | guest-failure / 139 | ACM unimplemented + SIGSEGV; out of 1H scope |
| Dreaming Sarah | guest-failure / 139 | null-SIGSEGV (pre-existing) |
| Void Terrarium | guest-failure / 139 | null-SIGSEGV (pre-existing) |
| Oregon Trail | external-watchdog-timeout / 124 | |
| Poppy | external-watchdog-timeout / 124 | loadavg &lt;5 at launch |

Artifacts under `runs/`.

## Phase 1 A–H

**Complete** after merge to `main`. Remaining loud refusals are post-Phase-1 product gaps
(UI/3D, ACM, Unity GC STW, sampled alias publish opt-in), not open Phase 1 invariants.
