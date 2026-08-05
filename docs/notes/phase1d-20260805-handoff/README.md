# Phase 1D handoff — register model (2026-08-05)

Architectural data flow only. Phase 1E–H not started.

## Design

Absent means AGC/SDK default, never unknown. `AgcGraphicsState` seeds Cx/Sh/Uc from
`agc_register_defaults` on construction. `CLEAR_STATE` / `resetQueue` apply the SDK 12-bit
`ResetQueueOp` mask and reseed selected files (ops `0` = `kNone`, no-op). Draw snapshots carry
the seeded maps; decoder fallbacks that invented CW winding, identity `SPI_PS_INPUT_CNTL`,
full-target viewports, or non-zero `SPI_SHADER_COL_FORMAT` writemasks are deleted.

### Runtime

| Piece | Behavior |
| --- | --- |
| Seed | `seed_register_defaults` over public v10 tables + internal Cx supplements + `SPI_SHADER_COL_FORMAT=0` |
| `resetQueue` | Selective bits (`CX_STATE`…`MEMORY_MAPPING_MASK`); never empty maps |
| `UcIndexOffset` | Base vertex from Uc `0x24a` (not Cx `0x102`) |
| Mask-zero draws | Still emit CP-ordered draw events; skip only shader evaluation when mask=0, no depth, no memory side effects |

### GPU Metal decode

| Register / state | Decode |
| --- | --- |
| `PA_SU_SC_MODE_CNTL` | Default `0x240` (CCW); no CW invent |
| `SPI_PS_INPUT_CNTL_n` | Default `0` (not logical index) |
| Viewport | Scale `1.0`, Z `0/0`; no full-target invent |
| `SPI_SHADER_COL_FORMAT` | Missing/default `0` → Zero export → write mask 0 |
| Stencil | Enable/ops/refs → Metal stencil; stencil-only without surface loud-refuses |
| Depth write disable | `DB_Z_INFO` bit 24 |
| Depth bias | Polygon-offset regs → `set_depth_bias` (scale/16); front≠back refuses |
| `CB_COLOR_INFO` | Blend bypass (bit 16) disables Metal blend |
| `CxVgtShaderStagesEn` / `UcGeCntl` | NGG sizing when Cx `0x291`/`0x2d3` absent |

### 6a082b1 viewport-Y / winding

Re-verified: Metal positive viewport height is kept; `PA_SU_SC_MODE_CNTL` front-face is passed
through unchanged. The deleted `guest_viewport_y_scale_is_negative` winding flip is **not**
reintroduced.

## Safeguards deleted

- Empty register maps after reset (reseed instead).
- Hard-coded full `resetQueue` / payload-zero CLEAR that ignored the selective mask.
- CW `PA_SU_SC_MODE_CNTL` fallback (`1<<2`).
- Identity `SPI_PS_INPUT_CNTL` absence fallback.
- Full-target viewport invent when scales absent.
- Dropping mask-zero draws entirely (now emit + skip eval only).
- Fake Cx `0x102` base-vertex source.

## Live evidence (`1273aab`, clean tree)

| Title | Outcome | Note |
| --- | --- | --- |
| Silent Hill 20s | `external-watchdog-timeout`; `runs/silent-hill/` (`run.json` clean) | Sustains AGC; NGG mesh logs show `winding=CounterClockwise`. **Not** a 3D-scene closure claim. |
| Matrix | not run | Title not present under local game library. |

Loadavg at SH launch (from `run.json`): 1-minute ~5.35 — no FPS/timing claim.

## Checks

- `cargo test -p remu-runtime --lib -- --test-threads=1` filters
  `agc_register_defaults::` / `platform_independent_tests::agc_packets::` /
  `platform_independent_tests::agc_draw::` — green arm64 and `x86_64-apple-darwin`.
- `cargo test -p remu-gpu-metal --lib -- --test-threads=1` — green both arches (484 tests).
- `cargo check --workspace --all-targets` — both arches green.

Full `remu-runtime` lib suite was not used as a gate: `event_state_lock_is_usable_without_probe_dispatch`
deadlocks (pre-existing; untouched here).

## Commits

- `37101fb` runtime: seed AGC register defaults and selective resetQueue
- `b31b3d7` runtime: read UcIndexOffset and keep mask-zero draws ordered
- `1273aab` gpu-metal: decode register defaults instead of wrong fallbacks
- (this commit) docs: Phase 1D register-model handoff and status close

## Loud refusals / refutations

- Stencil-only without a depth/stencil surface still refuses loudly (no fabricated stencil target).
- Front/back polygon-offset mismatch refuses (no Metal invent).
- Combined depth/stencil pixel formats remain future work (Metal resources row).
- Phase 1E–H not started (shader-core backlog untouched).
- Success is not the hallway: SH watchdog timeout is sustain evidence, not interactive 3D closure.
