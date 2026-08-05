# Phase 1A handoff — sync-object identity (2026-08-05)

Architectural data flow only. Phase 1B–H not started.

## Design (SDK ABI)

From `sdk-1000/sdk/target/include/sys/_pthreadtypes.h` and `_pthread.h`:

| Type | Typedef | Static initializer | Init write |
| --- | --- | --- | --- |
| mutex | `struct pthread_mutex *` | `NULL` / adaptive `(pthread_mutex_t)1` | control-block pointer into `*mutex` |
| cond | `struct pthread_cond *` | `NULL` | control-block pointer into `*cond` |
| rwlock | `struct pthread_rwlock *` | `NULL` | control-block pointer into `*rwlock` |
| `SceKernelSema` | `struct _SceKernelSema *` | n/a (create out-param) | full pointer-sized handle (`write_guest_u64`) |

Identity is the published control-block handle (or zero-extended kernel sema id), never the
address of the slot the guest passed. Resolve order: bare handle → `*slot` handle → static
materialization for `NULL`/`1` (mutex only for `1`). Unknown non-initializer values refuse.

`pthread_once` was already correct and is unchanged.

## Safeguards deleted

- Create-on-first-use mutex fabrication (`guest_mutex_entry(..., create=true)` for unknown
  addresses) and `LAZILY_FABRICATED_MUTEXES` / ABI-summary line.
- Condvar/rwlock auto-insert-by-slot-address without publishing a handle.

Static-initializer materialization is **not** a safeguard — it is the SDK contract.

## Live evidence (clean tree, revision `d3fc2f9`)

| Title | Outcome | Mutex note |
| --- | --- | --- |
| Silent Hill 60s headless | completed; `runs/silent-hill/` (`run.json` clean, rev `d3fc2f9`) | No `lazily fabricated mutex` line (counter removed). Phase 0 had `count=3`. Fabricated-success histogram still prints (unrelated Group-8 paths). |
| Demon's Souls 45s | guest-failure / exit 139; `runs/demons-souls/` | Assert moved `snd_osfuncs.cpp:397` → `:410`. Unlock of slot with `*slot==0` refused (`rejected unlock ... slot=Some(0)`). No HLE mutex init/lock observed for that slot before unlock; `sceAcmBatchStartBuffers` unimplemented on a worker. **Loud refusal: Scream path not closed.** `run.json` missing after crash — see `runs/demons-souls/EVIDENCE.md` + wrapper log. |

Loadavg: Silent Hill launch 1-minute ~4.85; Demon's Souls launch ~3.49. No FPS claim.

## Checks

- `cargo check --workspace --all-targets` — arm64 and `x86_64-apple-darwin` green.
- `cargo test -p remu-runtime -- --test-threads=1 synchronization::` — 27/27 both arches.
- Full `remu-runtime` suite: pre-existing hangers in `event_state::tests::*` (self-deadlock on
  nested write lock) and unrelated failures (`nid_u64` catalog sample, filesystem temp mount,
  ffmpeg golden when skipped). Sync module is green; full suite is not claimed green.

## Commits

- `513e8d4` runtime: resolve sync objects by SDK control-block handles
- `4e0e653` runtime: test sync-object identity via stack handle copies
- `d3fc2f9` runtime: optional REMU_TRACE_MUTEX_ABI for sync-object bring-up
- `636e1ba` tools: forward REMU_TRACE_MUTEX_ABI through run-game launch
- (this commit) docs: Phase 1A handoff with status and MEMORY updates

## Diagnostics

`REMU_TRACE_MUTEX_ABI=1` — first 64 init/lock/unlock lines with `arg` / `slot` / `object`.
Forwarded through `scripts/run-game.sh` (survives the launch `env -u` wrapper).

## Phase 1B–H

Not started.
