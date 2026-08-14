# Threading & async architecture audit (branch main)

No `crossbeam`/`flume`/`parking_lot`/`rayon` in the workspace — every channel is `std::sync::mpsc`, every lock `std::sync::Mutex/RwLock`. `available_parallelism` is never called.

**1. Global publication turnstile: process-wide serialization + thundering herd** — `crates/remu-gpu/src/channel.rs:240-255`: publishers send only at `turns.front()` of one global deque under one Mutex+Condvar; `channel.rs:296` `notify_all()` per 32-event chunk. A 3000-event submission = ~94 storms waking every parked publisher. **Fix:** per-queue condvars (coordinator already keys by `queue_id`, `channel.rs:231`); keep the global turnstile only for non-submission lifecycle events (`channel.rs:216-219`).

**2. Presenter main thread is the sole consumer doing all latency-critical work inline** — `presenter.rs:2905` → `present/window.rs:1716-1717`. Per tick on AppKit: `receive_shader_compilations` (twice: `submission.rs:88`, `:119`), `prefetch_pipeline_compilations` (`:106`, `:205`), encoding (`execute.rs:1944`), a synchronous ~8.3 MB `replace_region` memcpy per 1080p flip (`present/scanout.rs:1013`), retires, host UI, render; budget 2048 events/tick. **Fix:** dedicated upload thread with its own MTLCommandQueue + staging pool; prefetch/key work to a worker; flip producer writes into a persistently-mapped shared MTLBuffer.

**3. Prefetch rebuilds 512 deep pipeline keys per tick and discards them** — `shader.rs:55-95`; `GraphicsPipelineKey` (`presenter.rs:679-733`) has nested Vec allocations; called twice per tick. **Fix:** memoize on the snapshot (`OnceLock<GraphicsPipelineKey>` or 64-bit content hash).

**4. Compile requests deep-cloned three times, then dropped when the queue is full** — `shader.rs:126-129`, `:133`, `presenter.rs:2687/2701`; `Full` arm discards everything (`shader.rs:135-137`; compute `:169-171`). **Fix:** `Arc` keys end-to-end; park on Full instead of rebuilding.

**5. Compiler workers hold one mutex across a blocking `recv`** — `presenter.rs:2639`, `2674-2678`: only one of four workers can park in `recv`. Count fixed at 4 (`presenter.rs:321`), never scaled. **Fix:** real MPMC or Mutex<VecDeque>+Condvar; default from `available_parallelism`.

**6. One global mutex for every condvar/semaphore/event-flag operation** — `guest_sync/state.rs:299`; `guest_condition_wait` takes it twice (`condition.rs:212`, `:418`); notify takes global then per-object (`condition.rs:510`, `:516`). Mutexes/rwlocks are 64-way sharded with TLS cache and wake elision (`state.rs:311-316`, `mutex.rs:68-101`, `:526-530`); condvars got none of it. **Fix:** shard with the same scheme; move waiters and ticket counter per-object; add the TLS cache.

**7. `pthread_cond_signal` implemented as `notify_all`** — `condition.rs:544` after `release_one()` (`:525`): K waiters wake, K-1 re-park (`:265`). **Fix:** private park slot per waiter.

**8. Import dispatch is a linear string cascade; its u64 replacement is built and wired to nothing** — `lib.rs:3647` walks 14 string-match sites (`3684…4004`); `is_guest_sync_import` (`guest_sync.rs:1176`) matches ~60 more, then `dispatch_guest_sync_import_at_import` (`guest_sync.rs:1247`) re-matches. `ImportTelemetry.nids_u64` (`lib.rs:1812`) is dead code; `ImportRouteTable` written at `lib.rs:3953/3960/3967`, `routes.load` has zero call sites. **Fix:** `decode_nid` is const (`nid_u64.rs:14`) — jump-table match on u64; consult `routes.load(index)` first.

**9. Two-to-three globally shared atomic counters per import call** (~150k/s): `lib.rs:3687-3688`, `dispatch_telemetry.rs:360`. Striping exists but is cfg(test)-only (`dispatch_telemetry.rs:365-367`). **Fix:** promote striping or TLS counters.

**10. `PROBE_DISPATCH` write lock is the default fallthrough; re-entry is diagnosed, not prevented** — `lib.rs:4015-4016`; guard costs two `Instant::now()` + three atomic stores + TLS ops (`lib.rs:1730-1747`, drop `:1706-1711`). Re-entry prints "this will not complete" then deadlocks (`lib.rs:1723-1729`), reachable via AGC grow callbacks running guest code under the guard (`lib.rs:1669-1673`); `assert_wait_is_off_the_dispatch_lock` is cfg(test)-only (`lib.rs:1756-1767`). **Fix:** per-service locks (`lib.rs:4565` gestures at it); re-entry returns an error.

**11. Priority inversion set up and unmitigated** — `guest_signal.rs:645-668` applies real QoS classes; a UTILITY worker at `turns.front()` blocked on the full 8-slot channel holds the FIFO against USER_INTERACTIVE threads. Fixing #1 removes most exposure.

**12. `async_debug_log` writer silently discards ~half its records** — `async_debug_log.rs:121`: the flush probe `receiver.try_recv().is_err()` consumes a message whenever the queue is non-empty; not counted in DROPPED (`:124-128`). **Fix:** flush on counter or `recv_timeout`.

**13. `async_stderr` pump dies on any write error → process-wide eprintln deadlock** — `async_stderr.rs:121-123`: no EINTR retry on write (read has one, `:106-109`); signals installed without SA_RESTART (`write_watch/signal.rs:202`). After death, every `eprintln!` (942 sites) blocks on the full pipe holding stderr's lock. Reachable with `remu run … 2>&1 | head`. **Fix:** retry EINTR; on failure `dup2` REAL_STDERR back onto fd 2.

**14. `libc::_exit` bypasses every flush** — `lib.rs:10228` in `terminate_probe` (12+ reach sites); only flush is `remu-cli/src/main.rs:75`. End-of-run reports stranded; defeats `lib.rs:10198-10201` and the `[REMu][REPORT]` grep workflow. **Fix:** flush before `_exit`.

**15. AJM decodes under the engine mutex** — `ajm_engine.rs:429` runs the decode holding `AjmEngine.state`; both workers (`AJM_WORKER_COUNT = 2`, `:378`) serialize. **Fix:** take batch out under lock, decode outside, reinsert.

**16. Unbounded reclaimer queues silently degrade to inline drops** — `texture/retire.rs:80-95`, `shader_cache.rs:158-173`: on spawn failure the Sender is returned while the Receiver drops; every later send fails; callers drop inline on the hot thread (documented at `retire.rs:100-104`). Both unbounded.

**17. AvPlayer allocates ~3 MB per video frame, no pool** — `av_player.rs:1093`, `1071-1078`; consumer drops buffers (`:1143`); producer spins 1 ms when full (`:276-296`). **Fix:** return-channel buffer recycling.

## Preserve

The guest mutex path (64-way sharding + TLS cache + wake elision) is the model; `debug_log!` gating (`async_debug_log.rs:155-162`) is correct.
