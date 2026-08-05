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
materialization for `NULL`/`1` (mutex only for `1`). Unknown non-initializer values refuse on
lock/unlock.

Init always overwrites the slot with a fresh control block (except null / bare control-block
arguments). Under the pointer-typedef ABI, Scream creates mutexes in stack locals, returns the
handle by value, then reuses the same stack offset; leftover bits may still name a live object
whose recorded `storage_address` points at that recycled frame. Refusing that as "already
initialized" is wrong. Destroy blanks publishing storage only when it still holds this object's
handle.

`pthread_once` was already correct and is unchanged.

## Safeguards deleted

- Create-on-first-use mutex fabrication (`guest_mutex_entry(..., create=true)` for unknown
  addresses) and `LAZILY_FABRICATED_MUTEXES` / ABI-summary line.
- Condvar/rwlock auto-insert-by-slot-address without publishing a handle.

Static-initializer materialization is **not** a safeguard — it is the SDK contract.

## Acceptance follow-up (2026-08-05, revision `74aa06d`)

Root cause of the remaining Scream unlock failure after identity landing: init refused a stack
slot that still held a leftover live handle from a prior CreateMutex at the same stack offset
(`refused re-init of canonical storage … slot=0x… handle=0x…`), so CreateMutex failed with
`EINVAL`, the title kept a NULL handle, and unlock hit `slot=Some(0)` / `snd_osfuncs.cpp:410`.

| Title | Outcome | Mutex note |
| --- | --- | --- |
| Demon's Souls 45s | guest-failure / exit 139 after ~26s; `runs/demons-souls/` | **Scream mutex path PASSED** — no rejected unlock, no `snd_osfuncs` assert, `ncaPumpThread` + AudioOut2 advance/push run. Next loud blocker: `sceAcmBatchStartBuffers` unimplemented (1774 calls), then SIGSEGV `addr=0x303030303030303`. |
| Silent Hill 45s headless | completed (watchdog); `runs/silent-hill/` (`run.json` clean, rev `74aa06d`) | No lazily-fabricated-mutex line (counter still deleted). No mutex unlock refusals. |

Loadavg: Demon's Souls launch 1-minute ~3.71; Silent Hill launch ~5.92 (no FPS claim).

## Checks

- `cargo check --workspace --all-targets` — arm64 and `x86_64-apple-darwin` green.
- `cargo test -p remu-runtime -- --test-threads=1 synchronization::` — 29/29 both arches.

## Commits

- `513e8d4` runtime: resolve sync objects by SDK control-block handles
- `4e0e653` runtime: test sync-object identity via stack handle copies
- `d3fc2f9` runtime: optional REMU_TRACE_MUTEX_ABI for sync-object bring-up
- `636e1ba` tools: forward REMU_TRACE_MUTEX_ABI through run-game launch
- `620d730` docs: Phase 1A sync-object identity handoff
- `7a97f45` / `ad774be` interim init overwrite of non-canonical handle copies (+ tests)
- `3016846` runtime: always overwrite pthread slots on init
- `74aa06d` runtime: test recycled stack mutex publishing storage overwrite
- (this commit) docs: Phase 1A acceptance closed for Scream mutex; ACM next

## Diagnostics

`REMU_TRACE_MUTEX_ABI=1` — first 64 init/lock/unlock lines with `arg` / `slot` / `object`.
Forwarded through `scripts/run-game.sh` (survives the launch `env -u` wrapper).

## Phase 1B–H

Phase 1B (TLS) closed — see `docs/notes/phase1b-20260805-handoff/`. Phase 1C
(ordered publication) closed — see `docs/notes/phase1c-20260805-handoff/`. Phase 1D closed — see `docs/notes/phase1d-20260805-handoff/`. Phase 1E–H not started.

## Loud refusal (post-1A)

`sceAcmBatchStartBuffers` (`8fe55ktlNVo`) is unimplemented and is the active Demon's Souls
blocker after the Scream mutex path. Not Phase 1A scope.
