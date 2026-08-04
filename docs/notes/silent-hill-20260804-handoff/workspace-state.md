# Workspace state at committed stop

## Git

```text
repository: /Users/xnetcat/Documents/REmu
branch: main
checkpoint base: 30bd087 (gpu: accelerate sampled resident lookup)
prior FFmpeg/Rosetta commit: 1dc8744 (runtime: stabilize Rosetta execution and bundle FFmpeg)
checkpoint commits:
  6a082b1 gpu: extend NGG execution and texture pipelines
  8d899ba runtime: harden guest services and synchronization
  ed6e32a tools: improve runtime stall inspection
  2345cb6 docs: record Silent Hill bring-up evidence
  <commit containing this file> docs: preserve Silent Hill investigation handoff
```

These commits intentionally preserve the large stopped worktree without squashing unrelated
subsystems into one commit. The handoff commit contains the remaining copied evidence. The series
is a resumable checkpoint, not a compatibility-complete release.

## Last code edit

`crates/remu-gpu-metal/src/presenter/shader/program.rs`

- removed the `ngg_export_uses_general_lds` early return from `ngg_mesh_stage_for_draw`;
- supported fused NGG topologies now select mesh even when LDS-free;
- indirect NGG support was already part of the dirty tree.

`crates/remu-gpu-metal/src/presenter/tests/shader.rs`

- changed LDS-free buffer-fetching quad/strip expectations to mesh;
- updated memoryless direct-classifier and unsupported-topology assertions.

`ngg_export_uses_general_lds` is test-only, so the release/workspace checks no longer emit that
warning.

## Last passing focused checks

```text
cargo test -p remu-gpu-metal --lib presenter::tests::shader --no-default-features
34 passed, 0 failed

cargo test -p remu-gpu-metal --lib diagnostics --no-default-features
19 passed, 0 failed

cargo test -p remu-gpu-metal --lib \
  residency_trigger_keeps_the_debt_until_a_guest_read_asks_for_the_pixels \
  --no-default-features
1 passed, 0 failed

git diff --check
passed at the stage-selection checkpoint

cargo fmt --all -- --check
passed

cargo check --workspace --all-targets
passed, zero warnings

cargo check --workspace --all-targets --target x86_64-apple-darwin
passed, zero warnings

python3 -m unittest scripts/tests/test_lldb_taskgraph_stall_snapshot.py
15 passed, 0 failed

sh -n scripts/probe-taskgraph-stall.sh scripts/run-game.sh
passed
```

Both native and Rosetta-target compile checks were performed. No clippy gate or post-change
all-game matrix was performed after the last major edit. Broader serial unit-test attempts found
known failures and a runtime-suite hang; see `known-test-failures.md`.

## Key checkpoint areas to preserve/review

- `crates/remu-gpu-metal/src/presenter/**`: submission ordering, readback, residency, tiling/mips,
  formats, indirect NGG, mesh execution, shader diagnostics, output/present fixes.
- `crates/remu-shader-metal/src/**`: NGG registers, LDS operations, global/buffer/image lowering,
  control flow, export/interface work, scratch, scalar/vector ops.
- `crates/remu-shader/src/**`: decoder/IR/evaluator extensions.
- `crates/remu-runtime/src/**`: FFmpeg/AvPlayer/audio, AJM/Oodle, dialogs, synchronization,
  thread-affinity/runtime scheduling, TLS/linker/filesystem and AGC state.
- `crates/remu-memory/src/guest_ptr.rs`: guest-pointer work.
- `scripts/run-game.sh`, `scripts/probe-taskgraph-stall.sh`, and LLDB helper/tests.
- `docs/progress/silent-hill-20260804/**`: older investigation evidence; its observation note now
  includes the owner's correction that the hallway is not the title's 3D scene.
- `crates/remu-cli/examples/` and `crates/remu-loader/examples/relocations.rs`: committed diagnostic
  examples used during the investigation.

## Master-plan truth

The historical `NOTES.md` checkpoint marks many items partial-with-seam. In particular, the
definition of done is not satisfied while shader interfaces produce zero-filled fallbacks and the
compatibility gates fail. Other named tails include C1 direct NID routing, remaining C2 lock
duration, C4 getter shape, P3 Kraken integration, production S2 tiler/readback/absolute-address
wiring, pointer migration, codec coverage, and compatibility validation.

The checkpoint contains opt-in diagnostic flags and high-volume traces added during the
investigation. Audit their long-term suitability before treating the series as release-ready.
