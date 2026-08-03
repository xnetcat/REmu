
## Batch A1 merge — 2026-08-03

- **main HEAD:** `d8c5143c433f7da1a5268f3cc325676c7c185104` (`d8c5143 runtime: VfsMount table + journaled savedata (G2/G3)`)
- **prior main:** `2106331` (A5 K3)
- **branch:** `worktree-batch-a1-fs-vfs` (was `ccc6391`, rebased → `d8c5143`)
- **worktree:** `/Users/xnetcat/Documents/REmu/.worktrees/batch-a1-fs-vfs`
- **rebase:** `git rebase main` in A1 worktree — **clean, no conflicts** (1/1 onto `2106331`)
- **merge path:** fast-forward `main` ← `worktree-batch-a1-fs-vfs` (`2106331..d8c5143`)
- **lib.rs conflict resolutions:** none (rebase applied cleanly). A1 kept:
  - `mod savedata_journal;` / `mod vfs;`
  - `vfs: vfs::VfsState` on probe/runtime state
  - `vfs.clear()` + `rebuild_static_mounts(...)` after save-root setup
  - temporary `/temp0` mount via `register_temp0` / `unregister("/temp0")`
- **checks (pre-merge, A1 worktree):**
  - `cargo check --workspace --all-targets` (host arm64): **0 errors**
  - `cargo check --workspace --all-targets --target x86_64-apple-darwin`: **0 errors**
- **checks (post-merge, main):** same dual-arch — **0 errors**
- **A2/A3 note:** did not reset/delete/modify `worktree-batch-a2-ajm-codecs` or `worktree-batch-a3-pad-hid` (both still at `b9d55a5` base). A3 worktree has in-progress `lib.rs` / pad→hid edits untouched.
- **lib.rs tips for A2/A3 rebases:** main now includes A1 vfs/savedata_journal module regs + `VfsState` field + mount rebuild / temp0 hooks. Keep those when resolving `lib.rs`; layer ajm / pad-hid changes alongside without dropping A1 mounts or journal mods.


## Batch A4 merge — 2026-08-03

- **main HEAD after A4:** `655d3d8b624c416de8507807bbdf55311f618ab6` (`655d3d8 shader-metal: delete six glyph peepholes behind Q5 NggRegisters proof (J3)`)
- **prior main:** `d8c5143` (A1 G2/G3)
- **branch:** `worktree-batch-a4-shader-metal` (was `a727f9b` / `6ae780a` on `4f852c1`, rebased → `655d3d8` / `6f32225`)
- **rebased SHAs:** `6f32225` (StructuredCfg MSL / Q3), `655d3d8` (J3 glyph peephole removal)
- **worktree:** `/Users/xnetcat/Documents/REmu/.claude/worktrees/batch-a4-shader-metal`
- **rebase:** `git rebase main` in A4 worktree — **clean, no conflicts** (2/2 onto `d8c5143`)
- **merge path:** fast-forward `main` ← `worktree-batch-a4-shader-metal` (`d8c5143..655d3d8`)
- **conflict resolutions:** none (disjoint remu-shader-metal only)
- **checks (pre-merge, A4 worktree):**
  - `cargo check --workspace --all-targets` (host arm64): **0 errors**; remu-runtime lib **42** / lib-test **21** warnings (same crate totals as A1 post-merge)
  - `cargo check --workspace --all-targets --target x86_64-apple-darwin`: **0 errors**; remu-runtime lib **4** / lib-test **3**
- **warnings note:** raw `^warning:` line count +1 vs A1 log due to remu-memory `IllegalInstruction` formatting (crate still reports 1 warning); not a real rise from A4

## Batch A3 merge — 2026-08-03

- **main HEAD after A3 (final):** `154cd7cc21fe6fe53e199e1c0c989fcd26e3e686` (`154cd7c runtime: I5 pad/hid lifetime, timestamps, deadzones, and motion`)
- **prior main:** `655d3d8` (A4)
- **branch:** `worktree-batch-a3-pad-hid` (was `2bcce21` on `b9d55a5`, rebased → `154cd7c`)
- **rebased SHA:** `154cd7c`
- **worktree:** `/Users/xnetcat/Documents/REmu/.worktrees/batch-a3-pad-hid`
- **rebase:** `git rebase main` in A3 worktree **after A4 on main** — **clean, no conflicts** (1/1 onto `655d3d8`)
- **merge path:** fast-forward `main` ← `worktree-batch-a3-pad-hid` (`655d3d8..154cd7c`)
- **lib.rs conflict resolutions:** none required. Post-rebase `lib.rs` retains A1 hooks and layers A3 pad/hid beside them:
  - `mod savedata_journal;` / `mod vfs;` + `vfs: vfs::VfsState`
  - `vfs.clear()` + `rebuild_static_mounts(...)` + `register_temp0` / `unregister("/temp0")`
  - A3 adds `mod hid;` + pad module split (`pad.rs` → `pad/mod.rs`), `reset_hid_runtime`, probe NIDs routed via `dispatch_pad_import`
- **checks (pre-merge, A3 worktree):**
  - `cargo check --workspace --all-targets` (host arm64): **0 errors**; remu-runtime lib **42** / lib-test **21**
  - `cargo check --workspace --all-targets --target x86_64-apple-darwin`: **0 errors**; remu-runtime lib **4** / lib-test **3**
- **A2 note:** did not touch `worktree-batch-a2-ajm-codecs` (left at pre-existing SHA).


## Batch A2 merge — 2026-08-03

- **main HEAD after A2 (final):** `a1a4c17afd305332a423dd79d16b66e35f5c3b73` (`a1a4c17 runtime: AJM AAC + MP3 codecs and ooz Kraken decode stage (P2/P3/P6)`)
- **prior main:** `154cd7c` (A3 I5)
- **branch:** `worktree-batch-a2-ajm-codecs` (was `9d9eaf5` on `b9d55a5`, rebased → `a1a4c17`)
- **rebased SHA:** `a1a4c17`
- **worktree:** `/Users/xnetcat/Documents/REmu/.claude/worktrees/worktree-batch-a2-ajm-codecs`
- **rebase:** `git rebase main` in A2 worktree — **clean, no conflicts** (1/1 onto `154cd7c`)
- **merge path:** fast-forward `main` ← `worktree-batch-a2-ajm-codecs` (`154cd7c..a1a4c17`)
- **conflict resolutions:** none. Cargo.toml / remu-runtime Cargo.toml / build.rs applied cleanly beside A1 vfs/savedata and A3 pad/hid. `ampr_api.rs` untouched (P3 seam only).
- **scope landed:** `ajm_engine*` (AAC/MP3/oodle), `vendored/{minimp3,ooz,simde}`, remu-runtime `build.rs` + Cargo.toml codec deps, workspace Cargo.lock
- **checks (pre-merge, A2 worktree):**
  - `cargo check --workspace --all-targets` (host arm64): **0 errors**; remu-runtime lib **42** / lib-test **21**
  - `cargo check --workspace --all-targets --target x86_64-apple-darwin`: **0 errors**; remu-runtime lib **4** / lib-test **3**
- **checks (post-merge, main):** same dual-arch — **0 errors**; warning totals unchanged vs A3 baseline
- **P3 seam still open:** production `ampr_submit_command_buffer_ordered` still passes `ampr_read_file_to_guest` (`ampr_api.rs:1120`); wire to `ajm_engine::oodle::ampr_kraken_decode_stage` via two-phase helper (`~1123`) / `ampr_execute_recorded_read` (`~1237`). A2 only shipped the matching closure.


## Batch A complete — 2026-08-03

All Batch A items are on `main` at HEAD `a1a4c17`:

| Batch | Topic | Commit(s) on main |
|-------|--------|-------------------|
| A6 | H2 build-lock + E3 replay/criterion scaffolding | `2f08ebb` |
| A5 | K3 gpu-metal PERF → telemetry.jsonl | `2106331` |
| A1 | G2/G3 VfsMount + journaled savedata | `d8c5143` |
| A4 | Q3 StructuredCfg MSL + J3 glyph peephole removal | `6f32225`, `655d3d8` |
| A3 | I5 pad/hid lifetime, timestamps, deadzones, motion | `154cd7c` |
| A2 | P2/P3/P6 AJM AAC+MP3 + ooz Kraken decode stage | `a1a4c17` |

- **Integration order on main:** A6 → A5 → A1 → A4 → A3 → A2 (last)
- **Open follow-up:** P3 AMPR Kraken wire — `ampr_api.rs` ~1120/1123/1237 still uses plain file read; swap in `ajm_engine::oodle::ampr_kraken_decode_stage` when AMPR owner lands decode-stage selection.

## Batch B6 — remu-memory GuestPtr/GuestSlice (M1) + unsafe review rule (M4) — 2026-08-03

- **branch:** `worktree-batch-b6-memory`
- **base:** `a1a4c17` (main tip at Batch A complete)
- **worktree:** `/Users/xnetcat/Documents/REmu/.worktrees/batch-b6-memory`
- **scope:** `remu-memory` + `docs/review-unsafe.md` (+ CONTRIBUTING pointer). **No remu-runtime body edits.**

### M1 — `GuestPtr<T>` / `GuestSlice` validated layer — COMPLETE (API + tests)

- New `crates/remu-memory/src/guest_ptr.rs`, exported on macOS with the shared-mapping registry:
  - `GuestAccessPin` — lease-backed pin (`GuestRangeLease::acquire`); null/overflow/unmapped refuse
  - `GuestPtr` / `GuestPtrOwned` — typed single-object read/write via shared backing
  - `GuestSlice` / `GuestSliceOwned` — typed/byte slices; `as_bytes`/`as_bytes_mut` lifetime-bound to pin;
    prefer `read_bytes`/`to_vec` when the dispatch lock is not held across the borrow
  - Watch-backed sites: acquire the pin inside `CpuWriteWatch::with_live_mapping` (documented)
- Tests (TDD, written; not run this batch per owner gate): null/unmapped refuse, backing R/W round-trip,
  identity-mapped slice bound to pin, alignment + read-only write refuse, too-short typed slice,
  empty slice, identity view revoked after unmap.

### M4 — review-rule doc — COMPLETE

- `docs/review-unsafe.md`: any new `unsafe` outside `remu-memory` needs a SAFETY comment naming
  **invariant** + **enforcer**; prefer `GuestPtr`/`GuestSlice` / existing wrappers over one-off
  `from_raw_parts`.
- `CONTRIBUTING.md` Validation section links the rule.

### Runtime raw `from_raw_parts` migration inventory (NOTES only — do not edit runtime yet)

Production sites to migrate onto `GuestAccessPin` / `GuestSlice` / `GuestPtr` (or refuse with the
correct SCE error). Six subsystems from the 2026-07-28 audit defect #9, plus adjacent samples:

| Subsystem | File:line | Pattern | Migration note |
| --- | --- | --- | --- |
| Audio | `audio.rs:563` | `from_raw_parts` audio buffer | `GuestSlice::<u8>::try_acquire_mut` / read pin; B1 may land AudioOut validation in parallel — rebase carefully |
| Filesystem | `guest_files.rs:947` | `from_raw_parts_mut` sceKernelRead dest | writable `GuestSlice` |
| Filesystem | `guest_files.rs:970` | `from_raw_parts` `_write` source | readable `GuestSlice` |
| Filesystem | `kernel_api/filesystem.rs:307` | `from_raw_parts` positional write | readable `GuestSlice` |
| AvPlayer | `av_player.rs:502` | InitData fixed-size copy | `GuestSlice::try_acquire(...).to_vec()` |
| AvPlayer | `av_player.rs:610` | InitEx size-prefixed copy | same |
| AvPlayer | `av_player.rs:994` | URI bytes | readable `GuestSlice` then UTF-8 |
| NP / services | `np_api.rs:24` | init structure copy | `GuestSlice::to_vec` |
| NP / services | `lib.rs:6001` | NpSessionSignaling param copy | same |
| SaveData | `save_data.rs:961` | fixed ASCII read | readable `GuestSlice` |
| SaveData | `save_data.rs:971` | fixed ASCII write | writable `GuestSlice` |
| Guest libc / journal | `guest_libc.rs:312` | post-memcpy journal sample | pin over dest sample window |
| Guest libc / journal | `lib.rs:805` | mspace copy journal sample | same |
| SHA-NI / image | `sha_ni.rs:172`, `:370` | RIP instruction bytes | **not shared-mapping** — GuestImage/code path; needs a separate image pin or `ProcessGuestMemory` read (follow-up, not lease-only) |

Test-only (lower priority): `kernel_api/tests/filesystem.rs:309`, `agc_command_abi/tests.rs:924`,
`platform_independent_tests/memory.rs:232`.

### Checks

- `cargo check --workspace --all-targets` (host arm64): **0 errors**; remu-runtime lib **42** / lib-test **21**; remu-memory **1** warning (pre-existing `IllegalInstruction`)
- `cargo check --workspace --all-targets --target x86_64-apple-darwin`: **0 errors**; remu-runtime lib **4** / lib-test **3**; remu-memory **1** warning
- No `cargo test` / game gates (owner-run)


## Batch B5 — attachment/MSAA/occlusion + J1 — 2026-08-03

- **branch:** `worktree-batch-b5-attachment` @ base `7ee8650`
- **worktree:** `/Users/xnetcat/Documents/REmu/.worktrees/worktree-batch-b5-attachment`
- **J1:** synthetic pixel-pipe counter deleted; `GpuEvent::OcclusionQuery` + MTLVisibilityResultBuffer
- **B4-TBDR:** store-action policy + memoryless MSAA gate (loud refuse below 1.2 draws/pass)
- **B1:** attachment zero-copy eligibility + loud refuse (bind deferred)
- Full write-up: `~/Documents/REmu-research/implementation-notes/NOTES.md` (Batch B5 entry)
- Checks: arm64 + x86_64 `cargo check --workspace --all-targets` → 0 errors; no test/gates
