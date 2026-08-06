# Phase 1F handoff — EXECUTION SUBSTRATE (2026-08-06)

Merged onto `main` at `cf0de3b` (substrate `483e943` + docs). Phase 1G–H **not** started.

Architectural data flow for guest signals, fault delivery, game-process core mask, real sleep,
and honest host priorities.

## Design

### Signal / exception delivery

Orbis titles use `sceKernelInstallExceptionHandler` / `sceKernelRaiseException` (not raw POSIX
`sigaction` for the GC path). Unity Boehm GC installs exception type **30** (SIGUSR1) and raises
it against workers; the handler must see a full Orbis `ucontext_t` (**0x500** bytes, mcontext at
**`+0x40`**, `rsp` at absolute **`0xf8`** — SharpEmu / Unity GC walk shape).

| Path | Behavior |
| --- | --- |
| Cooperative | Pending exception still drained at HLE import return (self-raise / no host pthread). |
| Interruptive | `sceKernelRaiseException` → host `pthread_kill(SIGUSR1)` after registering each guest thread's host `pthread_t`. Host SIGUSR1 handler translates Darwin x86_64 mcontext → Orbis ucontext and invokes the guest handler. |
| Fault → guest | After write-watch recovery declines, `GuestFaultDeliveryHandler` may deliver SIGSEGV/SIGBUS to an installed guest handler (`remu-memory` → `guest_signal::try_deliver_guest_fault`). |
| `_sigprocmask` | Real per-process mask + **oldset writeback** (was inert fabricated success). |

aarch64-native interruptive delivery: loud refuse (Rosetta x86_64 is the shipping path).

### Core mask

| Mask | Value | Role |
| --- | --- | --- |
| `DEFAULT_THREAD_AFFINITY_MASK` | `0x7f` (`SCE_KERNEL_CPUMASK_7CPU_ALL`) | Game-process attr default (~7 cores). |
| `SYSTEM_THREAD_AFFINITY_MASK` | `0x1fff` | Still placeable when a title sets SYSTEM/NATIVE affinity. |

**Live note:** Demon's Souls still spawns 13 `BPE JobWorkerThread`s on this tip — title-side sizing /
explicit affinity, not the default mask alone. Default is no longer the former `0x1fff` fiction.

### Sleep

- `usleep(0)` / zero-duration nanosleep → `thread::yield_now()` (was a no-op via zero `Duration`).
- Wired: `sceKernelUsleep` / `usleep` / `sceKernelNanosleep` / `_nanosleep` / `nanosleep` /
  `sceKernelSleep` / `sched_yield`.

### Priorities

Console FIFO bands map to Darwin QoS via `pthread_set_qos_class_self_np` on the calling thread
(`scePthreadSetprio`, `scePthreadSetschedparam`, `pthread_setschedparam`). First QoS failure logs a
loud refusal. Cross-thread priority is recorded for Getprio only (no portable cross-thread QoS).

## Safeguards deleted / lies removed

- `_sigprocmask` → `inert_ok` (oldset left garbage)
- Default affinity `0x1fff` presented as the game-process default
- `usleep(0)` no-op that left backoff loops spinning
- Undispatched `nanosleep` / `sleep` / `sched_yield` inert acceptance
- `pthread_setschedparam` fabricated success (`note_inert` + return 0 without host effect)

## Commits (branch `phase1f-execution-substrate`)

| Hash | Message |
| --- | --- |
| `a97a2dc` | memory: offer unrecovered faults to installed guest handlers |
| `170daf0` | runtime: implement the guest execution substrate |
| `483e943` | runtime: enforce QoS from pthread_setschedparam |
| (docs) | this handoff + status/MEMORY |

## Checks

- Focused `remu-runtime` tests (`guest_signal`, affinity, pending exception): pass
- `remu-memory --lib`: 92 pass
- `cargo check --workspace --all-targets` — arm64 and `x86_64-apple-darwin`: green
- Full `remu-runtime --test-threads=1`: pre-existing hangs (`event_state_*`) and AGC/AJM fails also
  hang/fail on main tip — not introduced by 1F

## Live matrix (`483e943` unless noted; clean tree; headless)

| Title | Outcome | Notes |
| --- | --- | --- |
| Silent Hill (primary) | watchdog timeout | `_sigprocmask` + `pthread_setschedparam` live; **setschedparam gone from fabricated list**; `captured_frames=0` (not 3D closure). loadavg ~4 at launch. |
| Dreaming Sarah | SIGSEGV `addr=0x0` ~0s (`170daf0`) | **No** `InstallExceptionHandler` before crash — Unity GC STW path not reached. Same early null class as Phase 1D/1E. |
| Void Terrarium | SIGSEGV `addr=0x0` ~2s | Same early null class; fault forwarded (no guest handler installed). |
| Demon's Souls | guest-failure after ACM | Scream mutex still clear; **`sceAcmBatchStartBuffers` unimplemented** (3166 calls) then SIGBUS — not 1F. Still 13 JobWorkers. |
| Oregon Trail | watchdog timeout | Use `…/decrypted/eboot.bin` (duplicate `libc.prx` under app0). |
| Poppy | watchdog timeout | loadavg ~7.7 at launch — **no FPS claim**. |

### Unity GC acceptance

Architectural delivery + Orbis ucontext layout are implemented and unit-tested
(`orbis_ucontext_import_layout_places_rsp_at_0xf8`). Live Dreaming Sarah dies **before** installing
exception type 30, so interruptive RaiseException / GC root walk was **not** observed on a title
run this session. Do not claim Unity GC stop-the-world closed from these runs.

## Loud refusals (unchanged / out of 1F)

- General LDS + missing EXP PARAM attribute ring
- `S_BARRIER` under Dispatcher
- `S_GETREG`
- F64 div
- ACM `sceAcmBatchStartBuffers` (Demon's Souls)
- aarch64-native interruptive guest-signal delivery
- Cross-thread host QoS (recorded only)

## Phase 1G–H

**Not started.**
