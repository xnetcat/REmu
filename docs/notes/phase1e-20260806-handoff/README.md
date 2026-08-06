# Phase 1E handoff — SHADER CORE (2026-08-06)

Merged onto `main` at `5853fa7` (Phase 1E tip `2e82fd2` + PARAM follow-up). Phase 1F–H **not**
started.

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
| `SPI_PS_INPUT_CNTL` ↔ EXP PARAM producer reconcile | Landed (`5853fa7`) — unmatched slots without general LDS → SDK `UseDefault0` (`0x20`); general LDS + missing EXP still loud-refuse |
| Fabricated UV / RemuVsOut zero-fill seams | **Gone** — no title-keyed hacks |

Root cause of the prior ~31 Silent Hill “missing PARAM” refuses: live HLE
`sceAgcCreateInterpolantMapping` with null PS wrote identity `SPI_PS_INPUT_CNTL`, while captured
ES halves are PRIM+POS(+POS1) only (`exports=0`). Compacted EXP staging only relocates payloads EXP
already wrote. SDK/HLE unmatched-semantic path is `UseDefault0`, not identity PARAM offsets and not
INTERFACE-FILL.

## Safeguards deleted

- `[INTERFACE-FILL]` zero-init of missing `RemuVsOut` params
- Operand fabrication for raw VOP3/VOP3P/DS/MUBUF/MIMG
- Underscore/prefix fuzzy `image_is_comparison`
- Wrong `0x351` = `VDivFixupF32` alias
- Padding `vertex_outputs` with zeros for missing NGG EXP PARAMs (procedural rectangle synthesis kept)

## Loud refusals / remaining seams

1. **General LDS + missing EXP PARAM** — loud refuse (attribute-ring producer not lowered). Do not
   restore INTERFACE-FILL; do not invent UVs.
2. **`S_BARRIER` under Dispatcher** — refuse (SharpEmu divergent-barrier class).
3. **`S_GETREG_B32`** — refuse (unmodeled HW state).
4. **F64 div forms** — decode; lowering still refused with F64 policy.
5. **Pixel wave-size register** — not in SDK; scratch divisor documents 32.
6. **Pre-existing** `remu-shader-metal` irreducible CF tests fail on `0a8212a` and tip (2 tests).
7. **Dreaming Sarah / Void Terrarium host SIGSEGV `addr=0x0`** — reproduces on Phase 1D tip
   `0a8212a` as well as 1E tips; **not introduced by 1E**. `bf50225` control still reaches timeout.
   Already noted under Phase 1B status.

## Live evidence

| Title | Revision | Outcome | Notes |
| --- | --- | --- | --- |
| bf50225 DS control (1E close hour) | `bf50225` | timeout 45s / 30s recheck | PERF ~24–40 FPS, **not 60** → machine ≠ sentinel; **no FPS claims** |
| Silent Hill 150s (1E close) | `71d259e` | external-watchdog-timeout | `INTERFACE-FILL=0`; **31** loud missing-PARAM pair fails; NGG-MESH active; **not** 3D closure |
| Silent Hill 150s (PARAM follow-up) | `5853fa7` | external-watchdog-timeout | missing-PARAM pair fails **31→0**; Gen5 compiles 118; `captured_frames=0` this run (launch `vm.loadavg` `{ 16.87 9.78 9.43 }` — overloaded); **not** title-UI/3D closure (“success is not the hallway”) |
| Demon's Souls 45s | `71d259e` | external-watchdog-timeout | PARAM refuse=0; ACM unimplemented |
| Dreaming Sarah 45s | `71d259e` | guest-failure SIGSEGV | Same as `0a8212a`; control `bf50225` OK |
| Void Terrarium 45s | `71d259e` | guest-failure SIGSEGV | Same early null-fault class |
| Oregon Trail / Poppy 45s | `71d259e` | external-watchdog-timeout | Sustains / single run |

Follow-up artifacts: `runs/silent-hill-param-followup/` + `silent-hill-param-followup-run.log`.

## Checks (follow-up `5853fa7`)

- `cargo test -p remu-shader --lib -- --test-threads=1` — 154 pass
- `cargo test -p remu-gpu-metal --lib -- --test-threads=1` — 487 pass
- `cargo test -p remu-shader-metal --lib -- --test-threads=1` — 167 pass + 2 pre-existing irreducible CF fails (also on `0a8212a`)
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
2e82fd2 docs: Phase 1E shader-core handoff and status close
5853fa7 gpu-metal: reconcile SPI_PS_INPUT_CNTL with EXP PARAM producers
(+ this docs follow-up commit)
```

## Phase 1F–H

**Not started.**
