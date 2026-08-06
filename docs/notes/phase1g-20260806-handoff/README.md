# Phase 1G handoff — MEMORY MODEL (2026-08-06) — **MID-STOP / LOUD REFUSAL**

**Branch:** `phase1g-memory-model` @ `8ef3057` (worktree `/Users/xnetcat/Documents/REmu-phase1g`)  
**Base:** `main` @ `9397f11`  
**NOT merged to main.** Phase 1H **not** started.

## LOUD REFUSAL — Phase 1G is NOT closed

Architectural guest-backed + GPU-tiler data flow for tiler-owned textures is landed.
The **live title matrix gate is not closed** — every matrix title finished with
`captured_frames=0` on tip `5b920ab`. Silent Hill (primary UI-garbage gate) guest-faults
before a presentable frame; UI garbage cannot be assessed. Do **not** merge to main.

### Exact remainder

1. **Live title matrix** — Silent Hill UI garbage/corruption primary gate (need frames);
   then DeS, Dreaming Sarah, Void Terrarium, Oregon Trail; Poppy last; crash claims n≥5.
   Current tip evidence under `runs/` (all `captured_frames=0`):
   - Silent Hill: `guest-failure` / exit 139 (~14–15s); IMAGE-FALLBACK + SIGSEGV
   - Dreaming Sarah: null-SIGSEGV (~1s) — known pre-`InstallExceptionHandler` shape
   - Void Terrarium: `guest-failure` / 139 (~2s)
   - Oregon Trail: external-watchdog (~36s)
   - Demon's Souls: timeout (~34s) — ACM-blocked shape unchanged
   - Poppy: external-watchdog (~35s); loadavg rose to ~6.8 during run
2. **Optional cleanup:** delete or rewrite ignored fingerprint-era unit tests (5 ignored)
   once the live matrix proves coarse-watch invalidation is enough.

### Loud refusals that remain by design (not silent seams)

- Sampled comparison uploads refuse formats other than R16Unorm/R32Float reinterpret-as-depth
- Compressed / non-tiler-owned sampled uploads refuse (no GPU tiler ownership)
- Mode-24 mip chains refuse (no SDK mip layout wired)
- `WritebackTrigger::PressureEviction` is RemuBookkeeping — clears authored runs without publish

## What landed (commits on `phase1g-memory-model` after mid-stop `e0c582c`)

| Commit | Concern |
| --- | --- |
| `f9c11d3` | gpu-metal: sampled/storage GPU upload via tiler (mips/layers/mode-5); coarse dirty reload |
| `56f7737` | gpu-metal: delete fingerprints, self-readback gens, D5 prestaging, dead tiler scaffolding; end-state settle |
| `e731ce5` | gpu-metal: drop orphan derive after fingerprint deletion |
| `5745cf1` | gpu-metal: packet globals without byte-compare cache; ignore obsolete fingerprint tests |
| `5b920ab` | gpu-metal: remaps must not obsolete GPU-authored residents |

Earlier (pre mid-stop handoff):

| Commit | Concern |
| --- | --- |
| `53b97ba` | memory: anonymous → committed backing |
| `7d691ce` | gpu-metal: S2 tiler MSL into mode-27 rematerialisation |
| `b312be0` | gpu-metal: GPU upload + guest publish helpers; resident create |
| `1753158` | gpu-metal: standalone publish without active guest queue |
| `a1618a1` | gpu-metal: delete CPU resident writeback publish path |

## Wired vs deleted

**Wired**
- Anonymous → committed shared backing
- S2 tiler MSL for rematerialisation + `encode_gpu_upload_from_guest(_view)` /
  `encode_gpu_publish_to_guest(_view)` (mode 0/5/24/27; mode-5 layered; mode-27 mips)
- Sampled cache create/reload via settle GPU upload (no CPU snapshot/detile/`replaceRegion`)
- Storage `ensure_resident_texture_in_map` / mipmapped storage via GPU upload + queue
- End-state classification drives settle (guest-observable / lease-validated publish; bookkeeping clear)
- Packet global uploads without per-hit full byte compares

**Deleted**
- Host-private anonymous VMA kind
- Duplicate inline rematerialisation MSL
- CPU `write_resident_texture_contents_to_guest` stage
- Content fingerprints / dirty chunk compares
- Self-readback `gpu_write_watch_generations`
- D5 `texture/read_watch.rs` prestaging materializer
- Dead tiler `deferred.rs` / `absolute_address.rs` seams + presenter fields
- Persistent debt fiction on bookkeeping triggers (encode-then-clear / clear-without-publish)
- VA-keyed host byte-compare global buffer cache

**Kept**
- Leases (lifetime-only)
- Coarse write-watches for derived invalidation
- Alias canonicalization
- CP-effect writes
- Phase 1C ordered publication
- `gpu_authored` as ephemeral authored-run list for GPU publish encode

## Checks proven at tip

- `cargo test -p remu-gpu-metal --lib -- --test-threads=1` — 476 passed, 5 ignored
- `cargo test -p remu-memory --lib -- --test-threads=1` — green
- `cargo check --workspace --all-targets` — arm64 + `x86_64-apple-darwin` green @ `5b920ab`
- Live titles — **run; all `captured_frames=0`** (see remainder)
- No FPS claims

## Phase 1H

**Not started.**
