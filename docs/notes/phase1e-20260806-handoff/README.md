# Phase 1E handoff — SHADER CORE (2026-08-06)

Branch `phase1e-shader-core` tip `71d259e` (base `0a8212a`). Phase 1F–H **not** started.

## Design (what landed)

### Opcode / lattice

| Item | Status |
| --- | --- |
| Exact-match `image_is_comparison` + `ImageSampleModifiers` lattice | Landed (`5d10df6`) |
| `V_BFE_I32` sign-extend; integer CLAMP saturation; SCC both layers | Landed (`aab4955`) |
| `V_DIV_SCALE_F32` / `V_DIV_FMAS_F32` (scale ±32 pairs with FMAS `*2^32`) | Landed (`2160f55`, `9d01508`) |
| `VMulLoI32`, clz/ctz, `V_MUL_LEGACY_F32`, SOPP mode no-ops | Landed (`0213f71`) |
| Sub-dword LDS `DS_WRITE/READ` B8/B16 | Landed (`f104cdf`) |
| Graphics `SPI_TMPRING_SIZE` scratch + `GsWaveSize` wave width | Landed (`a2ebe85`); PS wave register still absent in SDK → scratch/32 documented |
| Stop fabricating raw VOP3/VOP3P/DS/MUBUF/MIMG operands | Landed (`502ca69`) — loud miss like SMRD/SMEM/FLAT |
| Opcode `0x351` vs libSceShader | Settled: `0x351` = `V_MIN3_F16`; `V_DIV_FIXUP_F32` = `0x15f` (`9d01508`) |
| Converged CF: graphics Dispatcher ballots → single-lane; `S_BARRIER` refuses **Dispatcher only** | Landed (`c1ed16d`, corrected `71d259e`) |
| `S_GETREG_B32` | Stays loud refuse; evaluator test updated |

### NGG / interface

| Item | Status |
| --- | --- |
| Compacted-vertex EXP → threadgroup staged records → `set_vertex` by compacted id | Landed under proven `NggRegisters` (`19325ba`) |
| `INTERFACE-FILL` zero-fill | **Deleted** (`bf5ceb6`) |
| Missing `SPI_PS_INPUT_CNTL` ↔ `EXP PARAM` (non-procedural) | **Loud refuse** (no UV invent) |
| PARAM-less NGG attribute producers (LDS / primitive-attr when no EXP PARAM) | **Loud refuse remains** — SH dumps `es=0x806676…` / `0x800c630…` are PRIM+POS(+POS1) only |

## Safeguards deleted

- `[INTERFACE-FILL]` zero-init of missing `RemuVsOut` params
- Operand fabrication for raw VOP3/VOP3P/DS/MUBUF/MIMG
- Underscore/prefix fuzzy `image_is_comparison`
- Wrong `0x351` = `VDivFixupF32` alias
- Padding `vertex_outputs` with zeros for missing NGG EXP PARAMs (procedural rectangle synthesis kept)

## Loud refusals / remaining seams

1. **PARAM-less NGG + VInterp** — refuse pipeline; architectural producer still unknown (no DS attr staging in captured SH ES halves). Do not restore INTERFACE-FILL.
2. **`S_BARRIER` under Dispatcher** — refuse (SharpEmu divergent-barrier class).
3. **`S_GETREG_B32`** — refuse (unmodeled HW state).
4. **F64 div forms** — decode; lowering still refused with F64 policy.
5. **Pixel wave-size register** — not in SDK; scratch divisor documents 32.
6. **Pre-existing** `remu-shader-metal` irreducible CF tests fail on `0a8212a` and tip (2 tests).
7. **Dreaming Sarah / Void Terrarium host SIGSEGV `addr=0x0`** — reproduces on Phase 1D tip `0a8212a` as well as `71d259e`; **not introduced by 1E**. `bf50225` control still reaches timeout. Already noted under Phase 1B status.

## Live evidence (`71d259e`, clean tree)

| Title | Outcome | Notes |
| --- | --- | --- |
| bf50225 DS control (same hour) | timeout 45s / 30s recheck | PERF ~24–40 FPS, **not 60** → machine ≠ sentinel; **no FPS claims** |
| Silent Hill 150s | external-watchdog-timeout | `INTERFACE-FILL=0`; ~31 loud missing-PARAM refuses; `SBarrier` refuse=0 after `71d259e`; NGG-MESH active; **not** title-UI/3D closure (“success is not the hallway”) |
| Demon's Souls 45s | external-watchdog-timeout | PARAM refuse=0; reaches ACM unimplemented (unchanged class) |
| Dreaming Sarah 45s | guest-failure SIGSEGV | Same as `0a8212a`; control `bf50225` OK |
| Void Terrarium 45s | guest-failure SIGSEGV | Same early null-fault class as status.md Phase 1B note |
| Oregon Trail 45s | external-watchdog-timeout | Sustains |
| Poppy 45s | external-watchdog-timeout | Single run; no crash claim (n≥5 not gathered) |

Loadavg at SH launch: `{ 14.73 9.56 8.69 }` / `{ 5.83 … }` earlier — above FPS gate; functional outcomes only.

Artifacts under `runs/` (symlinks into `$TMPDIR/remu-runs-phase1e/`).

## Checks

- `cargo test -p remu-shader --lib -- --test-threads=1` — 154 pass
- `cargo test -p remu-gpu-metal --lib -- --test-threads=1` — 485 pass
- `cargo test -p remu-shader-metal --lib -- --test-threads=1` — 166 pass + 2 pre-existing irreducible CF fails (also on `0a8212a`)
- `cargo check --workspace --all-targets` — arm64 + `x86_64-apple-darwin` green

## Commits (one concern each)

```
f104cdf shader-metal: lower sub-dword LDS reads and writes
502ca69 shader: stop fabricating operands for raw VOP3/VOP3P/DS/MUBUF/MIMG
aab4955 shader-metal: honor the sign, saturation and SCC the VALU actually defines
5d10df6 shader: exact-match image compare lattice
9d01508 shader: read the precise-division sequence at its real VOP3 opcodes
0213f71 shader-metal: lower the ALU and wave-state opcodes that dropped whole draws
a2ebe85 gpu-metal: read graphics scratch and export wave width from their registers
2160f55 shader-metal: lower V_DIV_SCALE_F32 with the precise-division VCC lane
bf5ceb6 gpu-metal: delete INTERFACE-FILL zero-fill for missing NGG params
c1ed16d shader-metal: keep barriers and ballots on converged control flow
19325ba shader-metal: stage NGG EXP payloads by compacted vertex id
71d259e shader-metal: allow S_BARRIER on structured CF; refuse only Dispatcher
(+ docs commit)
```

## Phase 1F–H

**Not started.**
