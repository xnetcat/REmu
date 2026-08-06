# Phase 1G handoff — MEMORY MODEL (2026-08-06) — **MID-STOP / LOUD REFUSAL**

**Branch:** `phase1g-memory-model` @ `a1618a1` (worktree `/Users/xnetcat/Documents/REmu-phase1g`)  
**Base:** `main` @ `9397f11`  
**NOT merged to main.** Phase 1H **not** started.

## LOUD REFUSAL — Phase 1G is NOT closed

Architectural data flow is only **partially** landed. The following remain before
definition-of-done:

1. **Sampled-texture CPU upload still live** — `upload.rs` / `create_sampled_guest_texture`
   still snapshot + CPU-detile + `replaceRegion`. Must route through
   `tiler::encode_gpu_upload_from_guest` (or equivalent) with an open/settle CB.
2. **`ensure_resident_texture_in_map` free function** still refuses CPU upload but is
   called from `upload.rs` storage path — must take presenter/batch context.
3. **Fingerprints** (`texture/watch.rs` content fingerprints / dirty chunk compare) not
   deleted — still used by sampled-cache dirty reload.
4. **Persistent `gpu_authored` debt ledger** still exists (used as ephemeral run list for
   GPU publish + rematerialisation overlay). Not collapsed to encode-then-clear-only.
5. **`gpu_write_watch_generations` / self-readback generations** not deleted.
6. **WritebackTrigger refusal policy** still gates bookkeeping triggers (PressureEviction
   etc.) — keep or delete only after GPU publish is default for all guest-observable
   paths and debt is gone.
7. **D5 prestaging materializer** (`texture/read_watch.rs`) still installed / dead-armed —
   must **delete** (GPU→guest retile replaces prestaged Metal→fault materialize).
8. **Dead tiler scaffolding** (`tiler/deferred.rs`, `tiler/end_state.rs`,
   `tiler/absolute_address.rs` Q4) still `dead_code` seams — wire or delete.
9. **Mip-chain / layered / mode-5 sampled** GPU upload+publish coverage incomplete
   (single-level 2D tiler-owned path is what landed).
10. **Per-hit byte compares** in global/binding cache not audited/deleted.
11. **Live title matrix** not run (Silent Hill UI garbage gate, DeS, Dreaming Sarah,
    Void Terrarium, Oregon, Poppy). No FPS claims.
12. **Full package tests** not green end-to-end after remaining deletions; only targeted
    rematerialisation + writeback-settle + tiler + `remu-memory` + dual-arch
    `cargo check` proven so far.
13. **docs/status.md / MEMORY.md / anti-claim** not updated to CLOSED (correct — not closed).

Do **not** treat this branch as Phase 1G complete. Do **not** merge to main until the
list above is empty without stubs.

## What landed (commits on `phase1g-memory-model`)

| Commit | Concern |
| --- | --- |
| `53b97ba` | memory: route `map_anonymous` through committed backing; delete `VmaKind::Anonymous` |
| `7d691ce` | gpu-metal: wire S2 tiler MSL into mode-27 rematerialisation; delete inline duplicate |
| `b312be0` | gpu-metal: GPU-timeline upload + guest publish encode helpers; resident create upload; settle via GPU retile |
| `1753158` | gpu-metal: standalone publish CB when no active guest queue |
| `a1618a1` | gpu-metal: delete CPU `write_resident_texture_contents_to_guest` publish path |

## Wired vs deleted

**Wired**
- Anonymous → committed shared backing (Metal-backable)
- S2 `tiler/msl.rs` is the rematerialisation library (mode-24/27); mode-5 pipelines ready
- `tiler/encode.rs`: `encode_gpu_upload_from_guest` / `encode_gpu_publish_to_guest`
- Resident exact-key create (presenter path) uses GPU upload for tiler-owned descriptors
- `settle_resident_texture_writeback` publishes via GPU retile (not CPU readback)

**Deleted**
- Host-private anonymous VMA kind
- Duplicate inline rematerialisation MSL
- CPU `write_resident_texture_contents_to_guest` stage

**Kept (still present; some still required)**
- Leases (lifetime)
- Coarse write-watches
- Alias canonicalization
- CP-effect writes
- `gpu_authored` (still ledger-shaped)
- Fingerprints / self-readback generations
- D5 prestaging (dead seam — delete pending)
- WritebackTrigger refusal policy

## Checks proven at tip

- `cargo test -p remu-memory --lib -- --test-threads=1` — green (after anonymous commit)
- Targeted `remu-gpu-metal` rematerialisation + GPU publish settle tests — green
- `cargo check --workspace --all-targets` — arm64 + `x86_64-apple-darwin` green @ `a1618a1`
- Live titles — **not run**
- `vm.loadavg` at check time ~`{ 3.94 2.95 2.84 }` (no FPS claim)

## Phase 1H

**Not started.**
