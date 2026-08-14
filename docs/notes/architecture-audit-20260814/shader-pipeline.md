# Shader translation pipeline audit (branch main)

Headline: the GCN→MSL backend has **no uniformity analysis, no loop-invariant hoisting, and no scalarization** — every guest SALU/SMEM operation re-executes per lane, and every guest memory access carries an inline bounds+alignment branch. That is the root cause of the 100–500 KB MSL outputs and the 150–290 ms tiny-draw GPU costs. The heap-table descriptor path has four live correctness bugs; the graphics pipeline key omits the field that selects between heap and snapshot lowerings.

## Tier 1 — structural: why the MSL is huge and slow

**1. Scalar register file is thread-private ⇒ all SALU work runs 32× redundantly.** `compiler/module.rs:1120-1128` declares each SGPR as a thread-local. **Fix:** uniformity lattice over decoded IR (uniform seeds: user data, `remu_initial_s`, constants, ballot results; divergent: VGPRs, lane, device loads); uniform values emitted once; `simd_broadcast_first` as escape hatch.

**2. SMEM loads become per-lane device loads with bounds check + unaligned fallback.** `compiler/memory/global.rs:60-70` → `remu_load_u32` (`module.rs:77-86`). One `S_LOAD_DWORDX8` = 8 device loads × 32 lanes, each with two branches. Bindings are `const device uchar*`, never `constant` (`module.rs:185-191`). **Fix:** `constant` address space for read-only uniform bindings; dword-typed loads for statically aligned offsets; vectorize N-dword loads.

**3. No loop-invariant code motion.** Loops emitted verbatim (`compiler/driver.rs:788-792`); no hoist/uniform/scalarize pass exists anywhere in `remu-shader{,-metal}`. Metal cannot hoist device loads past device stores. **Fix:** LICM over `remu_shader::StructuredCfg` for invariant SMEM addresses and descriptor extractions; `const`/`restrict` annotations.

**4. Per-instruction lowering emits enormous straight-line trees; one MUBUF ≈ 2–4 KB of MSL.** `compiler/memory/buffer.rs:84-166` (~21 temps per access); the ~400-char OOB ternary (`buffer.rs:702-728`) re-emitted **per component** (`buffer.rs:177-199`) — a DWORDX4 carries five copies. Direct explanation of the 516 KB output. **Fix:** decode each V# once per (SGPR quartet, program point); helper-fn OOB; specialize on evaluator-proven constant fields.

**5. Every VALU write is an exec-predicated store** (`compiler/vector.rs:113-117`) even inside regions already narrowed by `S_AND_SAVEEXEC`. **Fix:** region-level guards from the structurer.

**6. Split-wave64 emits two `simd_ballot` per lane-mask write.** `compiler/mod.rs:348-358` interpolates the ballot string into two ternaries; every `V_CMP` (`compare.rs:71-80` → `scalar.rs:21-44`) and `store_exec_mask` (`scalar.rs:46-51`) pays double. **Fix (one-liner):** hoist the ballot into a temp.

## Tier 2 — heap-table path correctness (all gated on REMU_HEAP_TABLES=1; none tested)

**7. Pipeline key omits `descriptor_index_is_runtime`.** `decode/pipeline.rs:1391-1445` (compute: `shader/program.rs:96-118`). The flag is forced from `decode_image_descriptor(...).is_err()` (`presenter/shader.rs:525-544`) — a predicate over fields absent from the key. Heap and snapshot variants of one shader collide silently; the refusal that would catch it is suppressed when tables are on (`texture/upload.rs:535-538`). Design doc flags it required (`docs/design/2026-08-13-runtime-image-descriptor-fetch.md:85-86`).

**8. Table map keyed by evaluation index, looked up by canonical slot.** Insert at `driver.rs:310, 344-345`; lookup converts through `image_slots.texture_slots[...]` (`image.rs:105-113`) into `heap_tables.get(&binding_index)` (`image.rs:85-97`). Domains agree only when the slot map is the identity — which it exists to not be (`ir/bindings.rs:385-431`).

**9. Discovery uses `destinations.any()`; shadow emission uses `destinations.first()`.** `driver.rs:323-327` vs `memory/global.rs:42-45`. An `S_LOAD_DWORDX8 s[8:15]` feeding a T# at s12 registers a table but never writes the shadow ⇒ silently samples heap slot 0 (initializers `module.rs:373-374`, fallback `image.rs:88-93`). No unset encoding fails loudly.

**10. Heap textures typed `texture2d<float>` unconditionally** (`module.rs:137-143`; stale 512 comment) while `texture_expression` feeds stores (`image.rs:244`), loads (`image.rs:341`), resinfo (`image.rs:177`), depth compares (`image.rs:501`). Uint/3D/array/writable bindings miscompile or reinterpret. 32-byte record stride hardcoded both sides (`global.rs:50`, `execute.rs:183`) against measured 16/24/32-byte records; the design doc's required assert (`:108-109`) does not exist.

## Tier 3 — cache-key instability, per-draw overhead

**11. Pipeline key rebuilt per draw incl. cache hits, with O(n²) slot-map recomputation.** `submission.rs:713` → `image_binding_slots()` (`pipeline.rs:1395`; O(n²) whole-descriptor compares, `bindings.rs:391-421`, `:164-179`); recomputed at `driver.rs:222`, `sampler.rs:103`, `sampler.rs:124`, `shader/program.rs:73`. Compute key built twice per dispatch (`submission.rs:652`, `execute/compute.rs:180`). **Fix:** memoize slot map in `ShaderEvaluation`; cache the key.

**12. Program identity is an Arc pointer** (`presenter.rs:983-995`): re-registration of identical bytes ⇒ new key ⇒ full re-lowering; no content-hash dedup (fingerprint exists, diagnostics-only: `agc_registers.rs:604-616`). Dedup recovers only at MSL-text level (`shader/archive.rs:481-497`, `shader/compile.rs:305-311`). **Fix:** content-hash identity.

**13. Key over-triggers:** slot maps derived from whole-descriptor equality (`pipeline.rs:1424-1425`) split/merge on address-only changes; live SH user data 0x00c..0x00f sampled as float bits flips `blend.enabled` on ±0.0 (`pipeline.rs:1297-1309`, `:1490-1521`). (Correctly excludes buffer base addresses/extents, `pipeline.rs:1398-1405`.) **Fix:** canonicalize on fields reaching the MSL signature.

**14. Compute has no in-memory MSL function cache** (`presenter/shader.rs:339`, signature `:296-301`); graphics does (`archive.rs:469-576`).

**15. Fresh-archive path serializes all four compile workers behind one write lock incl. a ~100 MiB checkpoint write** (`archive.rs:194-228` render, `:304-331` compute; loaded path correctly read-locks, `:131-144`).

**16. A pipeline miss blocks the guest queue** (`submission.rs:716-741`, `:654-671`); prefetch mitigates; the 516 KB shader's miss stalls the queue its full compile. Mesh pipelines bypass MTLBinaryArchive entirely (`shader.rs:776-818`) — full compile every process start.

## Tier 4 — evaluator soundness / correctness-by-heuristic

**17. VALU-tainted descriptors flagged but still used when heap tables are off (default).** `evaluator/resources.rs:769-787` sets `descriptor_index_is_runtime` + logs; snapshot recorded and bound anyway; without tables the flag has no consumer. Taint propagation itself (`evaluator.rs:684-703`) is sound.

**18. Unreadable bindings degrade silently to zero/black.** `resources.rs:916-942` (read fault ⇒ base 0/4-byte binding ⇒ loads return 0); `texture/upload.rs:499-612` manufactures 1×1 at address 0, log capped at 64 and suppressed under tables. Contrast: dynamic-descriptor ambiguity fails loudly (`resources.rs:820-825`). "Could not read" must be as loud as "read two".

**19. Path exploration truncation is invisible to the draw.** `MAX_PATH_STEPS = 1<<20` (`evaluator.rs:47`), forking `:575-594`; truncation recorded only in a static registry (`:186-229`); the draw receives a partial binding list indistinguishable from complete (`:181-185` acknowledges). Brent's detector (`control_flow.rs:69-99`) is good; abandonment needs a per-draw signal.

**20. Irreducible control flow silently degrades to a step-capped PC dispatcher.** `compiler/control_flow.rs:211-216` → `module.rs:786-793` (`steps >= limit → active = false` at `mod.rs:178-181`): cap exceeded ⇒ default clear color, silently. Dispatcher ballot degradation to single-lane EXEC/VCC (`compiler/scalar.rs:59-71`) is wrong for lane-count readbacks.

**21. `contains_register_reference` rescans the whole MSL body ~18×** (`module.rs:1093-1099`, `:1137-1155`; call sites through `emit_module`) — ~9 MB of scanning per 500 KB compile. Compute once.

## Order of attack

6 and 21 are one-line wins. 7 is the highest-value correctness fix (one key field). 1–4 are the real performance work with a shared prerequisite: an optimizing IR between `remu_shader::Program` and the MSL emitter carrying a uniformity lattice and a decoded-descriptor cache. Tier 3 is independently addressable.
