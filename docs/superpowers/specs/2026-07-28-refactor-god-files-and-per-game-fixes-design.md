# Remove per-game fixes and split god files

Date: 2026-07-28
Baseline commit: `e57a6f1`
Branch: `refactor/split-god-files`

## Problem

Two distinct problems share one codebase.

**Per-title patches.** `CONTRIBUTING.md` states the project rule plainly: "Build a general
emulation layer, not per-title patches" and "Never add a stub, silent no-op, guessed success
result, dummy resource, fake shader output, fallback frame, or title/product-ID special case."
Several pieces of code violate this. The largest rewrites a specific Silent Hill build's x86
machine code at five hardcoded guest addresses, matched by exact byte signatures.

**God files.** Three files hold 64,207 lines between them:

| File | Lines | Production | Tests |
| --- | ---: | ---: | ---: |
| `crates/remu-runtime/src/lib.rs` | 30,651 | ~23,700 | ~6,900 |
| `crates/remu-gpu-metal/src/presenter.rs` | 21,453 | ~16,400 | ~5,000 |
| `crates/remu-shader-metal/src/lib.rs` | 12,103 | ~7,265 | ~4,840 |

`remu-runtime/src/lib.rs` alone declares 1,176 top-level items, 762 of them functions, plus 35
file-private statics. It contains the dynamic linker, the import dispatch table, every HLE
service, the AGC command processor, and the guest memory manager.

## Goals

1. Remove per-title and per-middleware special cases, or make them general.
2. Split the three god files into topic modules without changing behavior.
3. Keep the codebase green and bisectable throughout.

## Non-goals

- Decomposing `MetalPresenter` or `Compiler` into composed sub-structs. They stay single types
  whose impl blocks are distributed across topic modules. Extracting owned sub-objects is a
  genuine behavior risk and is deliberately deferred.
- Investigating why guest PSO storage is zeroed. That is real work, recorded as an open gap,
  but it is a debugging project rather than a refactor.
- Reordering, rewriting, or correcting logic. Anything that looks wrong gets noted, not fixed.
- Fixing the 256 pre-existing clippy warnings.

## Phase A: remove per-game fixes

Behavior-changing. One commit per item. Runs first, because roughly 490 lines of the code Phase B
would move are deleted here, and because separating the behavior change from the code motion keeps
`git bisect` meaningful.

### A1. Silent Hill graphics-PSO guard

`crates/remu-runtime/src/lib.rs:1280-1666`, called from `lib.rs:532`.

`install_known_incomplete_graphics_pso_guard` allocates an executable staging page, writes
hand-assembled x86 trampolines into it, and patches five sites in the guest executable:

| Constant | Guest address |
| --- | --- |
| `ENTRY` | `0x0000_0008_0113_e3e0` |
| `RENDER_TASK_SITE` | `0x0000_0008_01b7_ccde` |
| `NULL_CONTAINER_SITE` | `0x0000_0008_03d1_1d29` |
| `NULL_FIELD_HANDLER_SITE` | `0x0000_0008_0153_a917` |
| `STALE_POST_UPDATE_SITE` | `0x0000_0008_0180_e8bc` |

Each site is gated on an exact byte signature (17 to 41 bytes) from one build of one title.

Delete: the function, its `#[cfg(not(target_arch = "x86_64"))]` stub at `lib.rs:1660`, the five
`report_*` reporter functions (`report_stale_post_update_object`,
`report_null_guest_field_handler`, `report_null_guest_container_entry`,
`report_stale_render_task`, `report_incomplete_graphics_pso_rejection`), and the call site.

Also delete `relative_x86_branch` (`lib.rs:1667`). Verified: all twelve of its call sites are
inside the guard, so it becomes dead.

The `REMU_DISABLE_KNOWN_GUEST_PATCHES` environment variable becomes dead. Verified: its only two
references are the guard's doc comment (`lib.rs:1292`) and its own check (`lib.rs:1338`). It is not
documented in `README.md`, so no documentation update is needed.

### A2. FMOD trace hooks

`crates/remu-runtime/src/lib.rs`. Delete the statics `FMOD_CORE_SYSTEM_INITIALIZE_TARGET` and
`FMOD_STUDIO_SYSTEM_INITIALIZE_TARGET` (declared near `lib.rs:4395`), their resets at `lib.rs:416`
and `lib.rs:5537`, the `extern "C"` interceptors `trace_fmod_core_system_initialize` and
`trace_fmod_studio_system_initialize` (`lib.rs:8967`, `lib.rs:8990`), and the `REMU_TRACE_FMOD`
NID interception at `lib.rs:2405` and `lib.rs:9049`.

These bind host function pointers to FMOD's exact C ABI and redirect a resolved guest export
through them. That is middleware-specific interception, not an emulation service.

The NID-to-name mapping `"pKwslsMUmSk" => c"fmod"` at `lib.rs:2035` is **kept** — it is a symbol
name recovery entry, not a behavior special case.

### A3. Guessed mode-27 mip hierarchies

`crates/remu-gpu-metal/src/presenter.rs:3965` and `presenter.rs:12078`.

The branch fabricates a complete mip hierarchy for tile-mode-27 textures when the guest descriptor
declares only one level. Its own comment states the justification: "Silent Hill currently needs at
least one inferred hierarchy to advance past its warning screen, while indiscriminately inferring
all of them causes expensive false-positive write-watch reloads."

Inferring resources the guest did not declare is fabrication. Collapse the branch at
`presenter.rs:3965` to the non-speculative path and delete `guessed_mode27_hierarchies_enabled`
(`presenter.rs:12078`) and `guessed_mode27_hierarchy_maximum_bytes` (`presenter.rs:12097`).
Verified: the latter's only caller is inside the branch being removed.

This cascades further than it first appears, and the cascade is verified rather than assumed:

- `Mode27WorkOrigin::AutomaticPromotion` (`presenter.rs:363`) is constructed at exactly one site,
  `presenter.rs:3994`, inside the branch being deleted. The variant becomes unreachable and is
  removed.
- `ResidentGuestTexture::inferred_hierarchy` (`presenter.rs:1594`) is only ever set true by
  `origin == Mode27WorkOrigin::AutomaticPromotion` at `presenter.rs:11157`. With that variant gone
  the field is permanently `false`, so the field is removed.
- Its readers then simplify: `presenter.rs:4867`, `5053`, `5113`, `5753`, and the guard at
  `11099-11100`. Each `if resident.inferred_hierarchy { A } else { B }` collapses to `B`, and
  `presenter.rs:4090`, `4185`, `11049` drop the `inferred_hierarchy: false` initializer.

Collapsing a branch whose condition is provably constant is behavior-preserving, but it is the one
place in Phase A that changes code shape rather than only deleting. It gets its own commit,
separate from the deletion, so the two are independently reviewable.

### A4. Unreal path fixup

`crates/remu-runtime/src/lib.rs:20407`, called at `lib.rs:20451`.

`av_player_remove_unreal_leading_dot_segments` strips leading `./` segments in a form specific to
Unreal's path conventions. Delete the function and its call.

If removal breaks generic path handling that a correct implementation needs, the correct action is
to implement proper POSIX path normalization rather than restore the engine-specific version. Note
it and stop rather than reintroducing the fixup.

### A5. Debug script

Delete `scripts/lldb_silent_hill_pso_watch.py` (currently untracked).

### A6. Document the regression

Removing A1 will very likely return Silent Hill: The Short Message to crashing at startup. This is
accepted.

Add a named open gap to `docs/status.md` under Runtime / ABI recording:

- The symptom: guest graphics-PSO object storage is observed zeroed after the engine stores and
  references the candidate, and the title's own constructor guarantees the mandatory `+0x18` child
  is non-null, so this is not an object needing more construction time.
- The four other failure shapes the guard covered: stale RenderThread task with a zero vptr, null
  container entry, null field handler, and a parent object destroyed by a nested update.
- The conclusion: guest object storage going stale mid-run points at a REmu memory-lifetime,
  aliasing, or HLE-return defect, not a game defect. That is the bug to find.

### Kept deliberately

These are general rules whose comments cite a title as evidence. Comments are provenance and stay.

| Item | Location | Why it is general |
| --- | --- | --- |
| `skip_incomplete_rectangle_draw` | `presenter.rs:13665` | Rejects a draw whose pixel interpolation has no matching vertex PARAM export. An undefined interface is undefined for any title. |
| `initial_depth_clear_for_target` | `presenter.rs:14598` | Picks the comparison-neutral far value when a depth surface is first seen read-only. Correct for any stale clear register. |
| Mode-27 tiling equations | `presenter.rs:16174` | The exact GFX10 64KB_R_X address equation. Hardware behavior. |
| `"pKwslsMUmSk" => c"fmod"` | `lib.rs:2035` | NID name recovery, not behavior. |
| Unreal/Wwise sizing comments | `kernel_memory.rs:20,726`, `lib.rs:4627` etc. | Explain why a limit or ABI layout is what it is. Documentation. |

## Phase B: split the god files

Behavior-preserving. Code moves; logic does not change.

The seams below are the ones already implicit in each file's ordering. No new boundaries are
invented.

### B1. `remu-runtime/src/lib.rs`

```
crates/remu-runtime/src/
  lib.rs          re-exports, LinkedImage public API, LinkError
  globals.rs      the 35 shared statics, pub(crate)
  link/
    mod.rs        LinkedImage construction
    relocation.rs apply_relocations_at, resolve_symbol_at, R_X86_64_*
    tls.rs        install_tls_support, seed_tls_thread_control, helpers
    unwind.rs     eh_frame parsing, FDE validation, registration
    imports.rs    stub creation, native/synthetic import resolution
    exports.rs    process export collection, symbol keys
  dispatch/
    mod.rs        ProbeDispatchState
    trampoline.rs import_dispatch_trampoline, ImportCallFrame
    table.rs      import_dispatch, dispatch_probe_import, dispatch_stateless_import
  hle/
    audio.rs      audio_out2.rs  pad.rs        video_out.rs
    np.rs         playgo.rs      dialog.rs     savedata.rs
    pthread.rs    sync.rs        mspace.rs     libc.rs
    json.rs       avplayer.rs    ampr.rs       sysmodule.rs
  agc/
    mod.rs        shader.rs   packet.rs   draw.rs   submission.rs   event.rs
  memory/
    mod.rs        direct.rs   flexible.rs   protect.rs   query.rs
```

Existing sibling modules (`agc_acb`, `agc_command_abi`, `agc_dcb`, `agc_driver`, `agc_frontend`,
`agc_register_defaults`, `agc_state`, `ampr`, `audio`, `av`, `kernel_api`, `kernel_memory`,
`nid_names`, `np`) are left where they are. Folding them into the new tree is a follow-up, not
part of this work.

### B2. `remu-gpu-metal/src/presenter.rs`

`MetalPresenter` remains one type in `presenter/mod.rs`. Its 66 methods, spanning
`presenter.rs:2850-9101`, distribute by topic:

```
crates/remu-gpu-metal/src/presenter/
  mod.rs          MetalPresenter struct + construction
  texture.rs      residency, write watch, sampled cache, retirement
  pipeline.rs     graphics/compute pipeline keys, archive, caching
  shader.rs       compile queue, function cache, MSL library cache
  submission.rs   ordered event receiver, guest command batches, waits
  present.rs      flips, drawable, presentation resources, statistics
  readback.rs     texture readback, capture, guest writeback
  decode.rs       render target/depth/viewport/scissor decoding
  format.rs       the free-standing color conversion helpers (9102-9180)
```

### B3. `remu-shader-metal/src/lib.rs`

`Compiler` remains one type. The ~5,400-line impl (`lib.rs:602-6028`) splits by instruction class;
the ~4,840 lines of tests (`lib.rs:7265+`) move out.

```
crates/remu-shader-metal/src/
  lib.rs          public API: compile_msl, MetalShader, MetalCompileOptions, errors
  compiler/
    mod.rs        Compiler struct, entry point, interface validation
    control_flow.rs  analyze_control_flow, dispatcher blocks, branches
    scalar.rs     scalar ALU and register state
    vector.rs     vector ALU, VOP3P, comparisons
    memory.rs     buffer/global loads and stores, LDS
    image.rs      sampled and storage image operations
    prelude.rs    the emit_*_prelude source generators
  tests/          moved test modules
```

## Verification

### Per-commit gate

```sh
cargo fmt --all -- --check
cargo test --workspace --no-fail-fast
cargo build --workspace --target x86_64-apple-darwin
```

`cargo test --workspace --no-fail-fast` must report **725 passed, 3 failed**. The three failures
are environmental, not defects:

```
remu-shader-metal: generated_ds_msl_compiles_with_apple_metal_in_every_valid_stage
remu-shader-metal: generated_memory_msl_compiles_with_apple_metal
remu-shader-metal: generated_vop3p_conversion_and_lane_msl_compiles_with_apple_metal
```

All three fail with `cannot execute tool 'metal' due to missing Metal Toolchain`. Running
`xcodebuild -downloadComponent MetalToolchain` makes them pass and is recommended, since they are
the only tests that prove generated MSL actually compiles. If installed, the expected result
becomes 728 passed, 0 failed.

Phase A changes the expected count. Tests asserting removed behavior are deleted with the code and
the new expected count is recorded in that commit's message.

### Clippy

`cargo clippy --workspace --all-targets -- -D warnings` **fails on the baseline** and cannot be
used as a pass/fail gate. Baseline per-crate warning counts at `e57a6f1`:

| Crate | lib | lib test (new) |
| --- | ---: | ---: |
| `remu-runtime` | 135 | 26 |
| `remu-gpu-metal` | 100 | 45 |
| `remu-shader-metal` | 10 | 14 |
| `remu-shader` | 7 | 6 |
| `remu-gpu` | 4 | 0 |
| `remu-memory` | 0 | 2 |

The gate is: **no crate's count increases.** Moving code into modules can legitimately resolve some
warnings (notably the 42 `items_after_statements`); a decrease is fine and gets noted.

### Move verification, Phase B only

`git diff -M --stat` must show renames and moves rather than rewrites. A "mechanical move" commit
showing substantial content-line churn means something changed that should not have. Stop and
investigate before continuing.

Additionally, for each split commit, the set of item names before and after must be identical
except for added visibility modifiers. A dropped or renamed function is a defect.

### Runtime regression check

The test suite cannot catch a runtime behavior break. After Phase A completes, and after each
crate in Phase B, run:

```sh
scripts/run-game.sh /Volumes/Untitled/sharpemu/games/test-game/PPSA02929-app0/eboot.bin 120
```

PPSA02929 is Dreaming Sarah, the title that currently works. Expected: boots, renders, roughly 60
FPS, roughly 1,770 draws/s, no audio underruns. Any deviation blocks the commit.

Silent Hill (PPSA10112) is **not** a regression gate after Phase A, since A1 is expected to break
it.

## Risks

| Risk | Mitigation |
| --- | --- |
| Visibility churn across 1,176 items produces an unreviewable diff | One module extracted per commit, each independently green. Never split two modules in one commit. |
| A moved function silently changes meaning via different `use` resolution | Item-name set comparison per commit; `git diff -M` inspection. |
| The 35 shared statics create circular module dependencies | They land in one `globals.rs` as `pub(crate)` rather than being distributed. Not elegant, but it is a move, and distributing them is a design change. |
| Phase B conflicts with any parallel work on these files | The branch is `refactor/split-god-files`. Parallel work on `runtime/lib.rs` or `presenter.rs` will conflict badly and should be paused or landed first. |
| A4 removal breaks generic path handling | Note and stop rather than reintroducing the engine-specific fixup. |
| Phase A tests fail in ways that indicate the guard masked a broader problem | That is the point. Record findings in the `docs/status.md` gap. |

## Sequencing

Phase A fully lands and is verified before Phase B starts. Within Phase B, B1, B2 and B3 are
independent and may land in any order; each crate's split lands completely before the next starts.
