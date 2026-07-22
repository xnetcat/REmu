# Metal shader control-flow and register lowering

`remu-shader-metal` has two correctness-equivalent control-flow paths. The compiler selects the
path from the decoded shader; no title or shader hash is involved.

## Direct path

A shader is emitted as straight-line MSL when its reachable control flow is one acyclic path. The
path may contain `S_BRANCH`: its signed, PC-relative target is validated against instruction
boundaries and followed at compile time. Instructions skipped by that branch are not emitted.
`S_ENDPGM` terminates the path. This removes `pc`, `active`, `steps`, the loop, the switch, and one
host branch per guest instruction.

AMD's RDNA branching model defines `S_BRANCH` as unconditional and `S_CBRANCH_*` conditions in
terms of SCC, VCCZ/VCCNZ, and EXECZ/EXECNZ. Consequently, a reachable conditional branch is not
guessed or folded from one evaluation snapshot. It selects the dispatcher path described below.

The design was checked against AMD's [RDNA 2 ISA documentation][amd-rdna2], SharpEmu's
`BuildBasicBlocks`/Metal dispatcher, and Kyty's SPIR-V branch reconstruction. SharpEmu confirms the
same signed SIMM16 target calculation but retains a dispatcher in Metal. Kyty directly emits
structured branches for selected shapes, while its source explicitly marks general CFG analysis as
unfinished. REmu therefore uses only the provable single-path subset in this cohort and does not
copy Kyty's merge-shape heuristic.

## Register state

Lowering discovers every statically indexed `s[N]` and `v[N]` value used by the emitted program.
Each becomes an independent MSL local (`remu_s_N` or `remu_v_N`); unused architectural registers
are not declared or initialized. Stage inputs and compute system registers are initialized only
when the translated body reads their guest register. EXEC and VCC mask SGPRs are materialized only
when an instruction observes them.

This is intentionally not an array with compacted indices. Independent values give Apple's Metal
compiler explicit SSA/liveness information and prevent addressable `uint s[128]` and `uint v[256]`
objects from forcing all guest state to remain resident or spill together. `MetalRegisterFootprint`
reports the exact scalar/vector value counts for diagnostics and regression tests.

## Measured first cohort

The regression probe translates 64 `V_MOV_B32` instructions using the same two VGPRs followed by
`S_ENDPGM`, at compute stage. Measurements were taken on the same Rust toolchain from baseline
commit `df16577` and this cohort:

| Metric | Baseline | Direct/scalarized | Change |
| --- | ---: | ---: | ---: |
| Generated MSL bytes | 8,540 | 3,957 | -53.7% |
| Explicit guest register state | 384 array elements | 2 independent values | -99.5% |
| PC dispatch cases on the reachable path | 65 | 0 | -100% |

The generated memory, LDS/DPP, packed-half, vertex, pixel, and compute programs are compiled in the
test suite with Apple's offline Metal compiler, in addition to source-level semantic tests.

## Deliberate remaining dispatcher cases

- Any reachable `S_CBRANCH_SCC*`, `S_CBRANCH_VCC*`, or `S_CBRANCH_EXEC*`. These require structured
  CFG reconstruction before they can lose the dispatcher without changing SCC/VCC/EXEC behavior.
- Any reachable cycle, including an unconditional backedge. Loop reconstruction must preserve the
  existing 16,384-step safety bound or replace it with a separately validated policy.
- The fallback currently dispatches per instruction rather than per basic block. Forming blocks is
  the next safe reduction even before complete structured-CFG recovery.
- Indirect PC-changing operations are not treated as static branches. Opcodes without complete
  lowering semantics remain compile errors.

All branch targets, including targets in unreachable decoded regions, are validated. A malformed
target is a compile error rather than an implicit shader exit.

[amd-rdna2]: https://gpuopen.com/news/rdna2-isa-available/
