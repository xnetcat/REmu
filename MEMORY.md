# REmu standing memory

Verify file:line before editing code that these facts describe. Prefer loud refusal over silent
substitution. Never return success for work not done. Never zero-fill guest-visible state as a
"temporary" measure. No title-keyed behavior; no consolation writebacks; no new heuristics — decode
registers/packets that state intent.

## Evidence rules

- No dirty-tree run counts as evidence (`run-game.sh` refuses unless `REMU_ALLOW_DIRTY_EVIDENCE=1`,
  which tags `evidence_status=dirty-tree-not-evidence`).
- Check `sysctl -n vm.loadavg` (1-minute < 5) before FPS/timing claims.
- Crash claims need n≥5 at one revision.
- Dreaming Sarah 45s interactive @ 60 FPS is the regression sentinel — run a `bf50225` worktree
  control the same hour first; if control isn't 60 FPS, the machine is not the change.
- Shipping build is x86_64 under Rosetta. Gate game runs on the built binary.
- `cargo check --workspace --all-targets` on arm64 and `x86_64-apple-darwin`.
- `remu-runtime` tests: `--test-threads=1`.

## Phase 1E — shader core (CLOSED 2026-08-06; on `main` @ `5bb169b`)

- Exact `image_is_comparison` lattice; VALU BFE/CLAMP/SCC; `V_DIV_SCALE/FMAS_F32`; sub-dword LDS;
  graphics scratch + GS wave; no raw VOP3/DS/MUBUF/MIMG operand fabrication; `0x351`=`V_MIN3_F16`.
- NGG: compacted EXP staging under proven `NggRegisters`; **`INTERFACE-FILL` deleted**;
  `SPI_PS_INPUT_CNTL` reconciled to EXP PARAM producers (`5853fa7`) — unmatched / no general LDS →
  SDK `UseDefault0`; general LDS + missing EXP loud-refuse (no UV invent).
- SH live: missing-PARAM pair fails 31→0 at `5853fa7`; still not 3D-scene closure.
- Barriers: structured OK; Dispatcher refuse. Graphics Dispatcher ballots = single-lane.
- Handoff: `docs/notes/phase1e-20260806-handoff/`.
- Phase 1F–H not started.

## Phase 1D — register model (CLOSED 2026-08-05)

- Seed Cx/Sh/Uc from `agc_register_defaults` on `AgcGraphicsState::default` and selective
  `apply_reset_queue_op` (`lib.rs`); never empty maps.
- Wrong fallbacks deleted: CW `PA_SU_SC_MODE_CNTL`, identity `SPI_PS_INPUT_CNTL`, full-target
  viewport, COL_FORMAT invent (`presenter/decode*.rs`).
- Decoded: stencil, depth bias (`set_depth_bias`), `CB_COLOR_INFO` blend bypass, Uc `0x24a`
  index offset, `CxVgtShaderStagesEn` + `UcGeCntl` NGG.
- `6a082b1` kept: absolute viewport height, no winding re-flip.
- Handoff: `docs/notes/phase1d-20260805-handoff/`. Head ~`1273aab`.

## Phase 1C — ordered publication (CLOSED 2026-08-05)

- Labels: aligned `publish_guest_shared_u64_if_identity` / `store_guest_visible_u64` release stores
  — not `write_guest_shared_backing` memcpy of 8 bytes (`submission/memory.rs` shared Release path).
- Payload→label: `guest_publish_release_fence` before the next label-shaped publish in
  `apply_pending_guest_memory_writes`.
- Word baseline guard: `guarded_backing_write` CAS for 4/8-byte runs (`shared.rs`).
- HLE: aligned `write_guest_u32/u64` / `read_guest_u32/u64` → same accessors (`lib.rs`).
- `docs/review-unsafe.md` SAFETY checklist includes ordering.
- Handoff: `docs/notes/phase1c-20260805-handoff/`. Head ~`1aa4d5b`.
- **Not deleted here:** bulk texture writeback baseline compare (Phase 1G).
- Next DeS blocker still ACM (`sceAcmBatchStartBuffers`), not publication.

## Phase 1B — TLS / `fs:` (CLOSED 2026-08-05)

- Load helpers: red-zone skip inside thunk + XSAVE; undo with `add rsp,128` / plain `ret` (not
  `ret 128`). Store sites keep site-side `lea` + `ret 128`. RSP dest slot is `ud2`.
- Discovery: seeded prefix-aware decode (`link/fs_decode.rs`) — not byte-scan. AOT shared zero-disp
  loads; per-site lea (first) / store-imm / load-disp; overflow + Other → SIGSEGV backstop
  (`tls_fault.rs`, before write-watch / null short-circuit).
- Store-imm AOT keeps Imm32 bits (`immediate as i32`); iced zero-extends Immediate32 into i64.
- Handoff: `docs/notes/phase1b-20260805-handoff/`. Head ~`4e0357f`.
- **Refuted:** sizing a huge per-site cushion to fix `AddressOverflow` — that was Imm32 `try_from`.
- Next DeS blocker still ACM (`sceAcmBatchStartBuffers`), not TLS.

## Phase 1A — sync-object identity (CLOSED 2026-08-05)

- `pthread_mutex_t` / `cond_t` / `rwlock_t` are pointer typedefs (`sys/_pthreadtypes.h`). Registry
  keys and resolve by the control-block handle written into the guest slot (`guest_sync/identity.rs`,
  `mutex.rs`, `condition.rs`, `rwlock.rs`).
- `SceKernelSema` create writes `write_guest_u64` (pointer-sized), not a 4-byte tear.
- Lazily fabricated mutex path + counter **deleted** (obsoleted by handle identity).
- Static `NULL` / adaptive mutex `1` still materialize on first use.
- Init always overwrites the slot with a fresh control block (Scream stack CreateMutex reuse).
  Destroy blanks `storage_address` only when it still holds this object's handle.
- Optional `REMU_TRACE_MUTEX_ABI=1` (forwarded by `run-game.sh`).
- Handoff: `docs/notes/phase1a-20260805-handoff/`.
- **Scream mutex path CLOSED** on `74aa06d` (no `snd_osfuncs` assert / rejected unlock).
- **Next DeS blocker (not 1A):** `sceAcmBatchStartBuffers` unimplemented (1774 calls on evidence run).

## Phase 0 counters (landed 2026-08-05)

- `remu_gpu::note_inert(nid)` → FABRICATED_SUCCESS histogram; printed from `write_abi_gap_summary`.
- Lazily fabricated mutex counter: **removed in Phase 1A** (was `guest_sync/mutex.rs` create branch).
- `REMU_TRACE_SHADER_OPCODES=1` histogram prints on the live ABI reporter (~15s) and clean exit
  (watchdog kill used to drop the end-of-run print).
- Shader frequency backlog: `docs/notes/phase0-20260805-handoff/SHADER-REFUSAL-BACKLOG.md`.

## Known stale status claims

- Void Terrarium "full diagnostic timeout" sustain (older `docs/status.md`): refuted 2026-08-05 on
  `663bc58` with ~1s SIGSEGV. Re-prove before repeating.
- Demon's Souls "Scream mutex / snd_osfuncs assert" is closed as of `74aa06d`; do not repeat that
  anti-claim. Active audio gap is ACM (`sceAcmBatchStartBuffers`), proven with n≥1 clean evidence.
