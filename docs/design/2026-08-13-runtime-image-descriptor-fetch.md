# Runtime image-descriptor fetch (heap texture tables)

Status: design from code, 2026-08-13. Prereq reading:
docs/notes/sh-3d-20260813-evidence/EVIDENCE.md (RETRACTIONS onward).

## The asymmetry this closes

Translated MSL already executes every SMEM load at runtime with the correct dynamic offset:
`compiler/memory/global.rs::emit_scalar_memory` lowers `S_LOAD_DWORDX8 s[32:39], s[b], sOff`
to `remu_load_u32(bN, s[off] + imm, len)` over the bound heap span — including offsets from
`V_READFIRSTLANE`, which the Metal shader executes properly. So at GPU time the T# *bytes*
in `s[32..39]` are correct. Only the **texture object** bound for the sample comes from the
CPU-side evaluation snapshot, whose readlane-fed offset is stale (evaluator skips VALU:
`STALE-VALU-OFFSET/DESCRIPTOR`, 43+43 hits in run 163916Z). Result: SH:TSM BasePass
materials sample wrong/black textures → scene targets 99-100% black (run 163003Z) → no
menu scene.

## Design

Per image binding whose descriptor chain is flagged stale (the evaluator now computes
`valu_scalar_destinations`; plumb a `descriptor_index_is_runtime: bool` onto ImageBinding):

1. **Compiler (remu-shader-metal):**
   - At each SMEM load, alongside the existing register stores, maintain a shadow per-SGPR
     u32 `remu_srd_slot[reg]` = `(bN_base_offset + dynamic + imm) / 32` for the FIRST
     destination register (and `0xffffffffu` on any other scalar write to that register).
     Cost: one extra assignment per scalar load; only emitted for programs containing
     flagged image ops.
   - At a flagged image op, instead of the fixed `texture2d<float> tN` parameter, emit
     `remu_heap_textures_bN[remu_srd_slot[srsrc]]` — an argument-buffer array bound per
     heap binding. Sampler unchanged (samplers are uniform in practice; extend later if a
     flagged S# appears).

2. **Encoder (remu-gpu-metal):**
   - For each heap binding referenced by flagged image ops: walk the bound guest span in
     32-byte steps, decode each candidate T# (`decode_image_descriptor`), resolve to the
     resident/sampled texture via the existing upload path (dedup by descriptor), and write
     the texture handles into a Metal argument buffer (`argument_buffers` tier2 on Apple
     silicon; array length = span/32, cap and log at e.g. 1024 entries).
   - Slots that do not decode get the fallback texture; slots the CPU rewrites are covered
     by the existing residency/oracle machinery because resolution goes through the normal
     upload path per slot.

3. **Evaluator (remu-shader):** keep the snapshot path for unflagged ops (uniform chains —
   the common case, zero cost change); flagged ops stop producing IMAGE-REFUSE because the
   binding no longer depends on the snapshot.

## Acceptance

- STALE-VALU-OFFSET/DESCRIPTOR sites route through heap tables (counter pairs).
- Discriminator rerun: BasePass black% at the menu falls from 99-100%; lighting chain
  carries real content; rooftop scene visible behind NEW GAME vs reference capture.
- No regression in PPSA02929 (the working title) — its materials are uniform-indexed and
  must keep the snapshot path.

## Notes

- VCC branch forking (second evaluator hole) is separable; land after the fetch.
- W3 lock moves (task #9) first or in parallel: verification runs must stop wedging.
