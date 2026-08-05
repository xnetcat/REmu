# Phase 0 handoff — 2026-08-05

Instrumentation + hygiene only. No Phase 1 A–H behavior changes.

## Done

1. **FABRICATED_SUCCESS / `note_inert`** — counter + helpers; wired across Group-8 inert paths; printed from `write_abi_gap_summary` as `[REMu][REPORT] fabricated success …`.
2. **Lazily fabricated mutex** — counted on mutex create-on-first-use; reported in the same ABI summary.
3. **SIGSEGV/SIGBUS reporter** — installed at startup with `SA_ONSTACK`; `sigaltstack` on guest threads. Live: Demon's Souls / Dreaming Sarah / Void Terrarium produced `[REMu][WATCH] unhandled SIGSEGV …`.
4. **`/devlog/app` writable** — `VfsMountKind::DevLog` on static mount rebuild.
5. **Evidence hygiene** — `run.json` records `vm_loadavg`, `display_mode`, `dirty_tree`, `evidence_status`; `run-game.sh` refuses dirty trees unless `REMU_ALLOW_DIRTY_EVIDENCE=1`.
6. **Decoder hygiene** — VOPC `0xEC` test expects `VCmpNleF16`; SCRATCH uses `scalar_address_field == 0x7f`.
7. **Shader opcode matrix** — see `SHADER-REFUSAL-BACKLOG.md`.

## Shader matrix (item 7)

| Title | Revision | Outcome | Opcode extract |
| --- | --- | --- | ---: |
| Silent Hill | `663bc58` clean | 60s headless; `PERF-PIPELINE failed=0` | 199 lines (`shaders=65 opcodes=197 raw=0`) |
| Demon's Souls | `663bc58` | SIGSEGV ~12s before decode (`addr=0x0`) | 0 |
| Dreaming Sarah | `663bc58` | SIGSEGV ~1s; missing `/app0/shaders/*.vert` | 0 |
| Void Terrarium | `663bc58` | SIGSEGV ~1s null deref | 0 |
| Oregon Trail | `663bc58` | decrypted eboot (root eboot duplicates `libc.prx`) | 34 (`shaders=6 opcodes=32`) |
| Poppy (last) | `663bc58` | 45s headless | 18 (`shaders=2 opcodes=16`) |

Artifact symlinks: `runs/<title>` → `$TMPDIR/remu-runs-phase0/…`.

**Loud refusals**

- No Metal `Gen5 … failed:` lines in these short headless windows; backlog is **decoded-opcode frequency** (refusal-surrogate before touching the shader core), not a compile-failure ranking.
- Demon's Souls / Dreaming Sarah / Void Terrarium produced **no** opcode histogram — crashes before AGC decode. Do not invent frequencies for them.
- Dreaming Sarah 45s interactive / 60 FPS sentinel was **not** claimed; title died in ~1s on missing shader files. No `bf50225` control run (no FPS claim).
- `display_mode` in `run.json` is often `unknown` on headless runs that write the file before CVDisplayLink starts.
- Some non-Silent-Hill runs had 1-minute loadavg ≥5; those are not timing evidence.

## How to reproduce opcode traces

```sh
export REMU_SKIP_BUILD=1 REMU_TRACE_SHADER_OPCODES=1 REMU_HEADLESS_METAL=1 REMU_AUTO_DISMISS_DIALOGS=1
export REMU_RUN_ROOT="${TMPDIR}/remu-runs-phase0"
# tree must be clean, or set REMU_ALLOW_DIRTY_EVIDENCE=1 (tagged non-evidence)
./scripts/run-game.sh '<eboot>' 60
```

Histogram prints on the live ABI reporter path (~15s) and on clean exit. External watchdog kill used to drop the end-of-run print; that is fixed in `663bc58`.

## Commits (Phase 0)

See git log from `7635423` through docs commit; one concern per commit.
