# Phase 1C handoff — ordered publication (2026-08-05)

Architectural data flow only. Phase 1D–H not started.

## Design

Guest-visible labels and HLE sync words publish as aligned release stores. Payload bytes that
precede a label in the same terminal publication batch are ordered with an explicit release fence
before the label store. Word-sized baseline-guarded writebacks use compare-exchange so a concurrent
guest store cannot be silently reverted in a read-compare-write window.

### Shared accessors (`remu-memory`)

| Helper | Role |
| --- | --- |
| `store_guest_visible_u32/u64` | Release-store through live guest VA (HLE + direct GPU path) |
| `load_guest_visible_u32/u64` | Acquire-load through live guest VA (waits / HLE) |
| `publish_guest_shared_u32/u64_if_identity` | Release-store through shared backing while identity holds |
| `load_guest_shared_u32/u64` | Acquire-load through shared backing |
| `guest_publish_release_fence` | Orders unordered payload memcpy before the following label |

`guarded_backing_write` uses CAS for aligned 4- and 8-byte runs; larger texture writebacks keep the
bulk baseline compare (deleted later by Phase 1G's one-copy model).

### GPU publication (`submission/memory.rs`)

- `RELEASE_MEM` / `WRITE_DATA` / fence-shaped 4- or 8-byte DMA copies → word release stores (shared
  or direct), never runtime-length `memcpy` of the label.
- Large `Fill` / `Copy` remain memcpy payload; `apply_pending_guest_memory_writes` fences before the
  next label-shaped publish in the batch.
- Wait observation uses acquire loads (`read_guest_memory_wait_value`).

### HLE

Aligned `write_guest_u32/u64` / `read_guest_u32/u64` route through the same visible-word helpers.

### Docs

`docs/review-unsafe.md` SAFETY checklist now requires an ordering clause for guest-visible shared
state and prefers the shared word accessors.

## Safeguards deleted

- Shared-path label publication via `to_le_bytes()` + runtime-length `write_guest_shared_backing`
  memcpy (torn 64-bit labels).
- Reliance on the write-mapping cache mutex as incidental payload→label ordering for large
  payloads (replaced by `guest_publish_release_fence` + release stores).
- Non-atomic read-compare-write for word-sized baseline-guarded publishes (replaced by CAS).
- Cached write-mapping `copy_from` / `copy_if_present` helpers superseded by
  `with_write_pointer_at` (single path for memcpy payload and atomic word ops).

## Live evidence (`1aa4d5b`, clean tree)

| Title | Outcome | Publication note |
| --- | --- | --- |
| Silent Hill 20s | external-watchdog-timeout; `runs/silent-hill/` (`run.json` clean) | Sustains AGC; no new publication refusal class |
| Demon's Souls 45s | guest-failure ~26s; `runs/demons-souls/` (`run.json` clean) | Scream mutex still passes; next blocker still `sceAcmBatchStartBuffers` (1794) then SIGSEGV `addr=0x303030303030303` |

Loadavg at SH launch 1-minute ~3.94; DeS ~4.01 (no FPS/timing claim).

## Checks

- `cargo test -p remu-memory --lib -- --test-threads=1` — green arm64 and `x86_64-apple-darwin`.
- `cargo test -p remu-gpu-metal --lib -- --test-threads=1` filters
  `payload_then_label` / `ordered_guest_label` — both arches green.
- `cargo test -p remu-runtime --lib -- --test-threads=1` filters `guest_sync::` /
  `platform_independent_tests::memory` / `kernel_memory` — both arches green.
- `cargo check --workspace --all-targets` — both arches green.

## Commits

- `f268167` memory: publish guest words with release stores and CAS
- `db7c8fe` gpu-metal: order payload stores before label publication
- `fa16e4c` runtime: publish HLE guest words through shared accessors
- `1aa4d5b` docs: require ordering proofs in unsafe SAFETY comments
- (this commit) docs: Phase 1C ordered-publication handoff

## Loud refusals / refutations

- `sceAcmBatchStartBuffers` remains unimplemented — Demon's Souls audio blocker after mutex; **not**
  Phase 1C.
- Bulk texture writeback baseline compare is **not** deleted here — Phase 1G owns the one-copy
  model that removes host→guest writebacks.
- AJM sideband / pad / audio ring ordering called out in CORE-MECHANISMS are outside this phase's
  listed publication data flow.
- Phase 1D–H not started.
