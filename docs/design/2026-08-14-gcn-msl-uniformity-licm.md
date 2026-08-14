# GCN-to-MSL uniformity, loop hoisting, and descriptor reuse

Status: design only, 2026-08-14. Implementation is intentionally deferred until the
Phase 2 integration and evidence gate is complete and a new base revision is approved.

This design addresses shader audit findings #1-4 in
`docs/notes/architecture-audit-20260814/shader-pipeline.md`. Today the Metal backend emits
the decoded GCN instruction stream directly into MSL. SGPR values are represented as
thread-private locals, scalar memory loads are emitted as per-thread device loads, loop
bodies are reproduced without invariant motion, and each MUBUF use re-decodes its V# and
repeats a large bounds expression. The target is an explicit, conservative optimization
plan between `remu_shader::Program` and MSL emission. It must preserve GCN EXEC and memory
semantics before it attempts to reduce work or source size.

## Scope and non-goals

The first implementation will:

- classify values by their variation domain and scalarize only proven-uniform work;
- hoist proven loop-invariant pure work and stable read-only scalar loads;
- decode V# and runtime T# descriptors once per reaching definition and reuse the result;
- centralize MUBUF out-of-bounds selection in a bounded helper; and
- retain the current lowering as a per-site or per-region fallback.

It will not infer physical non-aliasing from different guest virtual addresses, change the
runtime image-table ABI, invent cross-simdgroup synchronization for split wave64, or turn
an unsupported program into a compile failure in normal rollout mode.

## Optimization IR and dataflow

Add a typed analysis/lowering layer whose values have stable `ValueId`s and whose operations
name all explicit and implicit inputs and outputs. It can initially live in
`remu-shader-metal`, but its input is `remu_shader::Program` plus `ShaderEvaluation` and its
control-flow facts come from the existing `Cfg`, dominator, natural-loop, and `StructuredCfg`
analyses. It is not a textual MSL rewrite.

Every reachable instruction receives:

- SGPR, VGPR, SCC, VCC, and EXEC use/definition records;
- a side-effect summary;
- reaching `ValueId`s at block entry and exit; and
- the variation class of each produced value.

Uniformity uses this finite lattice, ordered from least to most varying:

```
Unreachable < KernelUniform < WorkgroupUniform < GuestWaveUniform < Divergent
```

Join chooses the more-varying class. `Unreachable` is used only by the solver; unknown or
unsupported definitions are `Divergent`, never optimistically uniform.

Seeds are:

- literals, encoded constants, user-data SGPRs, and `remu_initial_s`: `KernelUniform`;
- compute workgroup identifiers: `WorkgroupUniform`;
- SGPR/SALU results and correctly lowered collective results: at most
  `GuestWaveUniform`; and
- VGPRs, lane/thread/vertex identifiers, MUBUF/GLOBAL per-lane loads, and unknown device
  reads: `Divergent`.

An operation's output is the join of its inputs, additionally limited by opcode semantics.
An SMEM result is `GuestWaveUniform` only when its address is uniform and its binding is
proven stable for the operation's memory epoch. Ballot, read-first-lane, and read-lane are
marked collective operations: their results can be wave-uniform, but the operation itself
cannot be freely speculated or moved.

A forward fixed-point solver computes block-entry and block-exit state. At a CFG join,
uniformity is joined and a new phi-like `ValueId` is made when reaching definitions differ.
The opcode use/definition table must include implicit SCC, VCC, and EXEC effects. An unknown
opcode or unmodelled control operation creates an optimization barrier and kills the
affected state. Debug validation requires every reachable instruction to have an exhaustive
record before an optimized lowering may be emitted.

## EXEC masks and divergent control

GCN SALU and SMEM execute once for the guest wave and are not predicated by per-lane EXEC.
Scalarized MSL operations must therefore remain outside `if (exec)` guards. VALU, vector
memory, exports, and lane-visible side effects retain the current per-lane EXEC behavior.

Branches on SCC, or on correctly aggregated EXEC/VCC state, are guest-wave-uniform even
when the lane predicates that produced the aggregate were divergent. Conversely, being in
a structurally uniform block does not make a VGPR value uniform.

The initial implementation may use `simd_broadcast_first` only at a proven converged point
and only when one Metal simdgroup represents the required semantic scope. It must fall back
for dispatcher lowering, an unproven divergent collective, unknown graphics occupancy, and
split-wave64 cases that would require communication between two SIMD32 groups. No barrier or
threadgroup-wide exchange is introduced as an optimization shortcut. This deliberately
trades some missed scalarization for correct wave64 behavior.

## Loop-invariant code motion

LICM runs only for reducible natural loops that have a corresponding structured-loop
emission site. The optimizer records the natural-loop header, latches, body, exits, and
dominators, and gives the MSL emitter an explicit preheader plan immediately before the
current `while (true)` form.

An operation is invariant only when:

1. every operand is defined outside the loop or is already invariant;
2. it is pure, or it is a permitted stable read described below;
3. its definition dominates every use to be served by the hoisted value;
4. moving it does not make a path-conditional trap, invalid access, or side effect execute;
   and
5. it does not cross an EXEC/VCC collective, barrier, atomic, or unknown-effect boundary.

Conditional operations are not speculated merely because their operands are invariant.
Multi-dword SMEM loads and the runtime descriptor-shadow update they feed move as one unit,
or neither moves. Irreducible and dispatcher loops remain on legacy emission.

### Memory effects and invalidation

Each lowered operation has one of these effects, with a binding and exact/proven range when
available: `Pure`, `Collective`, `Read`, `Write`, `Atomic`, `Lds`, `Barrier`, or `Unknown`.
Writes, atomics, barriers/wait boundaries, and unknown effects end the reusable memory epoch
for possibly aliasing reads.

A scalar load can be reused or hoisted only if:

- its address is uniform and loop invariant;
- the evaluated binding is read-only for the full kernel lifetime;
- alignment and range behavior are preserved; and
- no operation in the loop can alias or invalidate it.

`GlobalMemoryBinding` currently describes guest addresses, not physical-byte identity.
Different guest addresses are therefore not proof of non-aliasing. Conservatively, the
first implementation will not hoist an SMEM read when a possibly aliasing writable global
binding exists, unless a later evaluator/host contract supplies explicit immutable and
physical no-alias evidence. `const device` is a C++ type qualifier, not that evidence.
Likewise, Metal `constant` address space or `restrict` is emitted only from such an explicit
host contract; it is never inferred from source syntax.

For aligned, statically ranged dwords, the optimized emitter may use a typed dword/vector
load. Dynamic or unaligned reads retain `remu_load_u32` and its exact bounds behavior.

## Descriptor decode once

Descriptor reuse is keyed by meaning, not a register number. A cache key contains descriptor
kind, SGPR base, the tuple of reaching `ValueId`s (four for V#, eight for T#), and all fields
that affect interpretation. A write to any member of the tuple invalidates the key. At a CFG
join, differing tuples decode separately or use legacy lowering.

For V#, decode once at the nearest safe dominating point after all tuple definitions:

- base and address bias;
- stride, swizzle, add-thread-id, and index-stride state;
- record count and null state; and
- format fields and the compact out-of-bounds selector.

Per-access vector indices and offsets stay at the MUBUF operation. Repeated component-level
out-of-bounds ternaries are replaced by one fixed helper operating on the decoded descriptor
and access terms. If the V# tuple is loop invariant, its decode is placed in the loop
preheader under the LICM rules above. Evaluator-proven constants may specialize fields only
when every source value and the pipeline key cover the proof.

For runtime heap T#, the existing canonical-slot and descriptor-shadow contract remains in
force. Heap/slot decode is materialized immediately after the complete eight-dword SMEM load
and reused at dominated image operations. It is hoisted only with that load and only under
the same stable-read proof. Overlapping SGPR writes, an incomplete descriptor, ambiguous
heap routing, or a binding-base/array-phase mismatch invalidates the shadow and refuses the
optimized route loudly. Static snapshot images require no runtime T# decode.

## Loud fallback and failure policy

Normal mode is conservative and local: an unsupported opcode, ambiguous reaching definition,
alias uncertainty, invalid dominance placement, or unsupported wave shape emits the current
legacy code for that site or structured region and records an exact reason and instruction
PC. It never silently substitutes an approximate optimization.

Malformed IR, an internally inconsistent analysis result, or an optimizer assertion is a
compile error. A force mode also turns any missed *eligible* transformation into a compile
error, making coverage gaps visible in tests and evidence runs. Fallback source is annotated
with debug comments in diagnostic builds so a captured MSL file explains the refusal.

## Options, cache keys, and rollback

Resolve environment policy once in the host and carry it in `MetalCompileOptions`; compiler
code must not read the environment. The proposed switches are:

- `REMU_SHADER_OPT=0|analyze|on|force`: legacy, analysis-only, safe selective lowering, or
  refusal on missed eligible work;
- `REMU_SHADER_UNIFORMITY=0`, `REMU_SHADER_LICM=0`, and
  `REMU_SHADER_DESCRIPTOR_CACHE=0`: independent emergency rollbacks; and
- `REMU_SHADER_CONSTANT_BINDINGS=0`: disables the later immutable/no-alias address-space
  specialization without disabling the other optimizations.

The resolved master mode and feature bits are part of graphics and compute pipeline/function
cache keys because they change generated MSL. All non-default switches are announced once at
startup. Rollout begins in `analyze`, advances feature by feature to `on`, and keeps `0` as
the release rollback until the evidence gate is retired deliberately.

## Telemetry and acceptance evidence

Add a `MetalOptimizationReport` to the compiled shader containing analyzed instructions,
uniform values, scalarized SALU/SMEM operations, hoisted loads, decoded/reused descriptors,
optimized/fallback loops, fallback reasons, analysis time, and estimated or actual MSL bytes
before and after. Aggregate these in a per-window `[REMu][PERF-SHADER-OPT]` line and attach
stage/address plus the report summary to existing shader compile diagnostics. Existing
lowering, Metal function, pipeline, and MSL-byte timings remain the A/B timing source.

Acceptance requires both semantic and workload evidence: no differential mismatches, no
new release-suite failures, a material fall in emitted bytes and scalar/device operations for
the captured tiny mesh shaders, and a corresponding reduction in their 150-290 ms compile or
execution path. A faster compile with a fallback spike is not acceptance.

## Tests

Focused release, single-thread suites will cover:

- `remu-shader`: lattice joins, fixed-point convergence, implicit SCC/VCC/EXEC uses and defs,
  dominator/natural-loop placement, and adversarial CFG joins;
- `remu-shader-metal`: one emitted SALU/SMEM value per safe Metal semantic scope, aligned
  vector loads, exact preheader placement, no hoist across a write/atomic/barrier/unknown
  effect, V#/T# reuse and invalidation, compact OOB helper emission, partial EXEC, collective
  control, wave32/split-wave64 refusal, and dispatcher fallback; and
- `remu-gpu-metal`: graphics and compute key separation for every option mode, host immutable
  contract propagation, real Metal compilation for all stages, and telemetry aggregation.

Add a small differential harness that executes the same supported IR through legacy and
optimized paths and compares registers, memory writes, and exports across randomized branch,
loop, EXEC-mask, and descriptor-redefinition cases. Keep captured problematic shader programs
as a corpus with code-size and transformation-count expectations, not brittle full-source
goldens.

Implementation validation commands are:

```
cargo test -p remu-shader --release -- --test-threads=1
cargo test -p remu-shader-metal --release -- --test-threads=1
cargo test -p remu-gpu-metal --release -- --test-threads=1
```

## Staged implementation commits

After Phase 2 evidence is complete and an updated base is provided, implement as separately
revertible commits:

1. typed def/use/effect IR and analyze-only uniformity solver with unit tests;
2. options, graphics/compute cache-key bits, reports, telemetry, and rollback plumbing;
3. uniform SALU/SMEM lowering behind its own switch, including wave64 refusals;
4. structured-loop preheaders and conservative read-only LICM;
5. V#/T# reaching-definition cache and the compact MUBUF OOB helper;
6. explicit host immutable/physical-no-alias contract and optional constant/restrict loads;
7. differential/corpus evidence, followed by a distinct commit enabling proven features by
   default.

Each stage must pass all three release single-thread suites before the next one begins. The
game harness and default-on decision belong to the later integration/evidence gate, not this
design commit.
