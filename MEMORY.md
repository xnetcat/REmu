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

## Phase 0 counters (landed 2026-08-05)

- `remu_gpu::note_inert(nid)` → FABRICATED_SUCCESS histogram; printed from `write_abi_gap_summary`.
- Lazily fabricated mutex count in `guest_sync/mutex.rs` create branch; same report.
- `REMU_TRACE_SHADER_OPCODES=1` histogram prints on the live ABI reporter (~15s) and clean exit
  (watchdog kill used to drop the end-of-run print).
- Shader frequency backlog: `docs/notes/phase0-20260805-handoff/SHADER-REFUSAL-BACKLOG.md`.

## Known stale status claims

- Void Terrarium "full diagnostic timeout" sustain (older `docs/status.md`): refuted 2026-08-05 on
  `663bc58` with ~1s SIGSEGV. Re-prove before repeating.
