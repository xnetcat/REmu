# Wave64 lane-doubling — Phase 1 lowering design (2026-08-13)

Strategy selected by `~/Documents/REmu-research/silent-hill/wave64-split-audit-20260812.md`
(§3b, §4): run one guest wave64 on **one** Metal SIMD32 group, each Metal thread carrying two
guest lanes (`t` and `t+32`). Guest wave scope becomes Metal simdgroup scope exactly; every
half-local hazard class in that audit's §1.2 table disappears structurally. The live
`[REMu][WAVE64-CENSUS]` (docs/notes/sh-3d-20260813-evidence/EVIDENCE.md) fixes the validation
targets: the shadow-mask kernels `0x800838`/`0x80085d` (26 B64 mask ops each; feeds the
all-zero deferred lighting) and the FFT family `0x800959/0x80095b/0x800973` (SCC/EXEC branches
+ 80–160 barriers under 16 waves; the striped scanout).

## Scope of Phase 1

Compute stage only, `wave_lane_count == 64`, behind a new `MetalCompileOptions` switch
(`wave64_lane_doubled: bool`) so the presenter can select it per shape:

1. First enable for `split_wave64_is_whole_compute_threadgroup()` (1..=64 threads — one guest
   wave): dispatch shrinks to `threads/2` Metal threads; no `S_BARRIER` interaction.
2. Then multi-wave compute groups (the FFT: 1024 guest threads → 512 Metal threads,
   16 simdgroups): `S_BARRIER` remains a plain `threadgroup_barrier` (now legal — every wave
   is one simdgroup and all Metal threads reach it), and the audit's phase-2 deletion list
   applies (pair-state spin, readlane table, readfirstlane records — §4 phase 2).

Pixel/vertex stay on reinterpret-as-wave32; mesh follows last (needs the per-slot
`set_vertex`/`set_index` doubling).

## Register model

- VGPRs: `float remu_v_lo[N]; float remu_v_hi[N];` per Metal thread (current model keeps one
  `v[]`). Emission goes through the centralized accessors (`compiler/operand.rs:17
  raw_source`, `:105 raw_source64`, `compiler/vector/mod.rs vector_operand` /
  `destination_vector`), so the doubling point is the accessor layer, not the ~3,000 lines of
  per-opcode emission: each VALU emission site expands to two statements via a
  `for_each_guest_half` emitter that rewrites `v[` → `v_lo[`/`v_hi[` and `exec` →
  `exec_lo`/`exec_hi`.
- Scalars stay single (`s[]` is wave state, one copy per thread is redundant but correct;
  scalar ops emit once, outside the half expansion).
- EXEC/VCC and every mask SGPR pair hold the exact architectural 64-bit value:
  `ulong mask = (ulong)(uint)simd_ballot(cond_lo) | ((ulong)(uint)simd_ballot(cond_hi) << 32)`.
  `ballot_words` (compiler/mod.rs:344-354) loses its forced-zero partner word.
- Guest lane identity: lo lane = `remu_wave_index*64 + remu_lane`, hi lane = `+32`. Everything
  seeded from `remu_thread_index` today (compute system registers, DS addtid, meshlet reads)
  must seed per half.

## Per-construct lowering (from the audit §3b table, on REmu sites)

| Construct | Site today | Lane-doubled |
| --- | --- | --- |
| `V_CMP_*` → SGPR pair | `mod.rs:344` ballot_words | exact 64-bit ballot, both words real |
| B64 mask ops + SCC | `scalar.rs:535-550`, `:637-639` | plain ulong; SCC exact and uniform |
| `S_CBRANCH_EXECZ/VCCZ/SCC` | `control_flow.rs:130-139` | ordinary `if` on uniform bool; DELETE the driver.rs:779-855 atomic exchange |
| `V_READLANE/READFIRSTLANE` | `vector/alu.rs:57-137` | `sel<32 ? simd_shuffle(v_lo,sel) : simd_shuffle(v_hi,sel-32)`; DELETE table+barriers |
| `V_MBCNT_*` | `vector/alu.rs:225-241` | popcount over the exact mask |
| DPP16/row masks | `vector/lane.rs` | rows 0/1 from lo, 2/3 from hi |
| LDS (`DS_*`) | `memory/data_share.rs` | two accesses per thread (lo/hi addresses); wave-internal ordering = `simdgroup_barrier(mem_threadgroup)`; DELETE scalar.rs:568-623 spin |
| `S_BARRIER` | `driver.rs:596-618` | `threadgroup_barrier` (uniform by construction) |
| Image/buffer loads | vector memory paths | two per thread with per-half exec predication |

## Interfaces that must change with it

- `execute/compute.rs`: dispatch size halves for lane-doubled pipelines; the pipeline key must
  include the mode (a lane-doubled and a split build of the same CS differ in MSL).
- Threadgroup memory: guest LDS size unchanged (bytes are per group), but the compiler-state
  arrays (pair states, readlane tables) disappear — reclaims ~4.5 KiB against the 32 KiB cap.
- `validate_compute_launch_for_metal` (`shader/program.rs:142`): execution-width check gains
  the lane-doubled arm (`wave 64 on SIMD32 with threads/2`).
- Register pressure: `presenter/shader.rs:576-578` already halves wave64 VGPR budgets per the
  SDK rule; the doubled per-thread footprint lands back on the architectural count. Occupancy
  must be measured, not assumed (audit §3b costs).

## Validation

- Unit: a compare→mask→branch program asserting the exact 64-bit ballot in the generated MSL,
  a readlane(sel≥32) program, and an LDS exchange program, each compiled in lane-doubled mode.
- On-device: extend the M4 probe pattern (audit addendum) to execute a lane-doubled kernel
  with divergent halves and assert ballot/readlane results — the probe harness already exists
  in scratch form.
- Live: the chain-filtered scanout-writer dumps
  (`REMU_DIAGNOSTIC_SCANOUT_WRITERS=auto1080:<ps-list>`, `REMU_DUMP_TARGET_MIN_DRAW≈25000`)
  against `800a69` (lighting must turn nonzero once the shadow-mask kernels are exact) and the
  `800da3` scanout (stripes must break up once the FFT family is exact). Do not judge from the
  presented frame; presentation cadence is a separate open problem (docs/status.md
  2026-08-13).
