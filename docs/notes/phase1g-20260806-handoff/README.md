# Phase 1G handoff — MEMORY MODEL (2026-08-06) — **CLOSED**

**Branch:** `phase1g-memory-model` @ `c197935` (worktree `/Users/xnetcat/Documents/REmu-phase1g`)  
**Base:** `main` @ `9397f11`  
**Merged to main** @ `b4a82f5`. Phase 1H **not** started.

## CLOSED — what “closed” means (loud anti-claims)

Architectural guest-backed + GPU-tiler data flow for tiler-owned textures is landed and on
`main`. Silent Hill’s **1G-introduced** guest SIGSEGV is fixed; SH/Oregon/Poppy reach
watchdog-timeout clean again (Phase 1F-shaped acceptance).

**Anti-claims (do not invent success):**

- `captured_frames=0` on tip **and** main control same hour — PPM UI-garbage gate remains
  **unassessable**. Present path runs (SH flips ≥1024 with scanout snapshots); do **not** claim
  UI/3D scene closure from timeout or flip count alone.
- Dreaming Sarah / Void Terrarium null-SIGSEGV before `InstallExceptionHandler` — known, not 1G.
- Demon’s Souls still ACM-blocked (`timeout` / unimplemented ACM NIDs).
- No FPS claims.

### Silent Hill fault (root cause)

| | |
| --- | --- |
| Symptom | `SIGSEGV` `addr=0x8000000080b0` `pc=0x8016b7c09` ~14s exit 139 (n≥5 @ `bf4130b`/`f21484c`) |
| Main control | `9397f11` same hour: watchdog 124, **no** SIGSEGV, flips≥512, `captured_frames=0` |
| Bisect | `f9c11d3` survives; first bad `56f7737`/`e731ce5` |
| Cause | Alias/SampledAlias classified lease-validated → GPU publish into recycled guest heaps |
| Fixes | `f21484c` keep debt on bookkeeping; `f07da85` refuse alias/sampled publish by default |
| Post-fix | `f07da85` watchdog 124; flips to 1024; no SIGSEGV |

Evidence: `runs/silent-hill-fault-diagnosis.txt`, `runs/silent-hill-main-control/`,
`runs/silent-hill/` + `runs/silent-hill-artifacts/`.

### Live matrix @ tip (post-fix)

| Title | Outcome | Notes |
| --- | --- | --- |
| Silent Hill | external-watchdog-timeout / 124 | flips≥1024; frames=0 |
| Demon’s Souls | timeout / 142 | ACM shape unchanged |
| Dreaming Sarah | guest-failure / 139 | null-SIGSEGV (pre-existing) |
| Void Terrarium | guest-failure / 139 | null-SIGSEGV (pre-existing) |
| Oregon Trail | external-watchdog-timeout / 124 | |
| Poppy | external-watchdog-timeout / 124 | loadavg &lt;5 |

### Loud refusals that remain by design

- Sampled comparison uploads refuse formats other than R16Unorm/R32Float reinterpret-as-depth
- Compressed / non-tiler-owned sampled uploads refuse (no GPU tiler ownership)
- Mode-24 mip chains refuse (no SDK mip layout wired)
- `WritebackTrigger::PressureEviction` / Alias / SampledAlias are RemuBookkeeping by default —
  refuse publish and **keep** Metal debt (opt-in: `REMU_PUBLISH_INTERNAL_WRITEBACKS=1`)

## What landed

| Commit | Concern |
| --- | --- |
| `53b97ba` | memory: anonymous → committed backing |
| `7d691ce`…`a1618a1` | GPU tiler upload/publish; delete CPU writeback path |
| `f9c11d3` | sampled/storage GPU upload via tiler |
| `56f7737`/`e731ce5` | delete fingerprints / dead seams (introduced SH fault) |
| `5745cf1` | packet globals without byte-compare cache |
| `5b920ab` | remaps must not obsolete GPU-authored residents |
| `f21484c` | keep Metal debt on bookkeeping settle; restore generations |
| `f07da85` | refuse alias/sampled publish by default |

## Wired vs deleted

**Wired**
- Anonymous → committed shared backing
- S2 tiler MSL for rematerialisation + `encode_gpu_upload_from_guest(_view)` /
  `encode_gpu_publish_to_guest(_view)` (mode 0/5/24/27; mode-5 layered; mode-27 mips)
- Sampled cache create/reload via settle GPU upload (no CPU snapshot/detile/`replaceRegion`)
- Storage `ensure_resident_texture_in_map` / mipmapped storage via GPU upload + queue
- End-state classification: DMA/flip guest-observable publish; bookkeeping refuses + keeps debt
- Packet global uploads without per-hit full byte compares

**Deleted**
- Host-private anonymous VMA kind
- Duplicate inline rematerialisation MSL
- CPU `write_resident_texture_contents_to_guest` stage
- Content fingerprints / dirty chunk compares
- Self-readback generation *filtering fiction without keep-debt* (generations restored for Active path)
- D5 `texture/read_watch.rs` prestaging materializer
- Dead tiler `deferred.rs` / `absolute_address.rs` seams
- VA-keyed host byte-compare global buffer cache

**Kept**
- Leases (lifetime-only)
- Coarse write-watches for derived invalidation
- `gpu_write_watch_generations` for Active false-positive filtering after fingerprint deletion
- Alias canonicalization
- CP-effect writes
- Phase 1C ordered publication
- `gpu_authored` as publication debt until a real guest-observable publish

## Checks proven at tip

- `cargo test -p remu-gpu-metal --lib -- --test-threads=1` — 476 passed, 5 ignored
- `cargo test -p remu-memory --lib -- --test-threads=1` — green
- `cargo check --workspace --all-targets` — arm64 + `x86_64-apple-darwin` green
- Live matrix — run (table above); SH fault closed vs main control
- No FPS claims

## Phase 1H

**Not started.**
