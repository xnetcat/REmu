# REmu architecture audit — what we should be doing differently

2026-08-14, branch main @ 1058fdd5. Four parallel deep audits (CPU↔GPU sync/enqueue,
memory/residency, threading/async, shader pipeline — full reports in this directory) plus the
lessons of the 2026-08-12/14 Silent Hill campaign. 72 findings consolidated; file:line
references are in the per-area reports.

## The five structural changes that matter most

1. **Stop re-resolving physical identities the write-sets already carry** (memory #1-2).
   Every queued write records its resolved `GuestSharedMappingIdentity` at capture time; the
   read path (`queued_global_read_source`) discards them and re-resolves through the global
   registry RwLock per candidate, per batch, per binding, per draw — ~90k lock acquisitions
   per 5s window, the measured cause of the ~2.8ms/draw crawl. Resolve the read identity once
   per lookup and compare stored identities. Pure refactor, no semantic change.

2. **Take translation off the AppKit main thread** (sync #1, async #2). One thread does event
   decode, Metal encoding, texture upload, an 8.3MB memcpy per flip, pipeline-key
   construction, retirement, publication, host UI, and rendering, with a 2048-event tick
   budget. Everything else in this audit is a constant factor against that wall. A dedicated
   translation/encode thread owning the guest-batch state reduces AppKit to drawable
   acquisition and compositing.

3. **An optimizing IR between GCN decode and MSL text** (shader #1-5). The backend has no
   uniformity analysis, no loop-invariant motion, no scalarization: all scalar work runs
   32x redundantly per lane, SMEM loads become per-lane device loads with bounds branches,
   one MUBUF emits 2-4KB of MSL, and descriptors are re-decoded per access. This is why MSL
   outputs reach 516KB and why one 90-index UI draw costs 150-290ms of GPU. Uniformity
   lattice + LICM + descriptor-decode-once addresses performance and code size together.

4. **A sound residency model instead of a patched one** (memory #5-11). Format/pitch in the
   residency key makes every view of a surface a separate object, reconciled by a
   six-predicate disjunction grown one title-bug at a time (the BGRA/RGBA black-screen class).
   Rekey by physical bytes with format views inside the entry; epoch-stamp spans against the
   mapping generation; make writeback transactional (validate the lease BEFORE subtracting
   resident authority); route every resident removal through one retire() chokepoint that
   refuses to destroy unpublished GPU-authored bytes; never coarsen gpu_authored ranges.

5. **Fix the four heap-table correctness bugs before the path ships by default** (shader
   #7-10): `descriptor_index_is_runtime` missing from the pipeline key (heap/snapshot
   variants collide silently); table map keyed by evaluation index but looked up by canonical
   slot; discovery uses any-destination while shadow emission uses first-destination (silent
   slot-0 sampling); heap slots hardcoded `texture2d<float>` for all types; 32-byte record
   stride assumed against measured 16/24/32-byte records with no assert.

## Concurrency and dispatch

- Per-queue publication wakeups instead of the global turnstile whose `notify_all` fires per
  32-event chunk and wakes every publisher process-wide (async #1, sync #10).
- Kill the backpressure busy-spin: `has_runnable_events` must answer "runnable given
  backpressure" so the loop parks on the existing completion waker instead of spinning the
  main thread at 100% through every GPU-bound stall (sync #2). Replace
  `wait_until_settled_within`'s 250µs sleep-poll with a condvar signalled by
  `addCompletedHandler` (sync #5).
- Per-equeue wait lists: today one completion wakes every `sceKernelWaitEqueue` waiter and
  each one takes the process-wide PROBE_DISPATCH **write** lock to poll (sync #3). Split
  PROBE_DISPATCH per service; make guard re-entry an error instead of a printed deadlock;
  promote the wait-under-lock assert out of cfg(test) (async #10).
- Shard condvars/semaphores/event-flags with the 64-way scheme mutexes already use; give
  waiters private park slots so `pthread_cond_signal` wakes one thread, not all (async #6-7).
- Wire the already-built u64 NID jump table; today import dispatch is a linear 11-byte
  string-compare cascade and the route table is written but never read (async #8).
- Real MPMC for compiler workers (today one mutex held across blocking `recv` serializes the
  pool); scale worker counts from `available_parallelism` (async #5).
- Per-queue in-flight batch budgets instead of the global cap of 4 committed batches across
  all hardware queues (sync #4).

## Hot-path amortization

- Interval-index queued writes so lookups are O(log n), not a linear rescan of every write in
  every live batch per binding per draw (sync #6).
- Intern pipeline keys (content-hash fingerprint → Arc); compute once when the draw snapshot
  is built; stop rebuilding 512 deep keys per tick in prefetch and throwing them away
  (sync #9, async #3-4, memory #17).
- Key program identity by content hash of the instruction bytes, not `Arc` pointer identity —
  re-registration of identical bytes currently forces full re-lowering (shader #12). Remove
  float-bit sampling and whole-descriptor equality from the key (shader #13).
- Stop deep-cloning `ShaderEvaluation` per stage per draw (1.2KB + 4 Vecs; defeats the Arc
  cache) and `IndexedDraw` on cold paths; index caches must not clone a 120KB index array to
  build a probe key (memory #18-20). ~105 heap allocations/draw today.
- Route resident-texture scans through the existing interval index; skip the O(residents)
  `compatible` scan when the exact key already matched (memory #4, #12). Index full spans,
  not level-0 only, or mip/array aliasing escapes authority subtraction (memory #11).
- Write tracking: page-indexed dirty bitmap instead of O(all watched pages) per host write
  under a global mutex also held across multi-MB copies; surface `is_demoted()` so demoted
  resources stop paying 4x full-resource snapshots forever (memory #13-15).

## Straight bugs found by the audit (fix independently)

- `async_debug_log` flush probe uses a consuming `try_recv`, silently discarding roughly half
  of all records without counting them dropped (async #12).
- `async_stderr` pump exits on the first non-EINTR write error; afterwards every `eprintln!`
  in the process blocks forever on the full pipe while holding stderr's lock (async #13).
  `libc::_exit` in `terminate_probe` bypasses every flush, stranding end-of-run reports
  (async #14).
- Pressure eviction always refuses its own writeback then removes the resident anyway —
  guaranteed loss of GPU-authored bytes if a byte budget is ever set (memory #6).
- AJM audio decodes under the engine mutex, so its two workers cannot overlap (async #15).
- Reclaimer queues silently degrade to inline drops forever if their thread failed to spawn;
  both are unbounded (async #16). AvPlayer allocates ~3MB per video frame with no pool
  (async #17).

## Policy: loud failure and attribution as architecture

The campaign's costliest bugs were silent: placeholder-black textures multiplying through
tonemap; unreadable bindings degrading to zeros; truncated evaluator paths indistinguishable
from complete ones; dispatcher step-caps producing clear color; carried equeue completions
silently evicted; stale-VALU descriptors logged but still bound. Adopt as invariants:
- Every fallback is loud and counted; "could not read it" fails as loudly as "read two".
- Every instrument announces itself at startup (silent decline is indistinguishable from a
  healthy run).
- Per-stage time attribution stays always-on and cheap (the PERF-EVENT-LOOP /
  TRANSLATION-TERMS / SLOW-CB family landed this week is the model; host profilers cannot see
  into the Rosetta process).

## Process: how the campaign should have run

- **Command-stream capture/replay** as the primary dev loop: 160+ live 5-minute runs fighting
  boot timing, press schedules, and watchdogs reproduced states an offline replay would give
  in seconds, deterministically, with golden-image diffs per milestone screen.
- Evidence discipline that held: evidence-clean runs, n>=5 before crash claims, loadavg
  recorded per run (and consulted — the load-blind 157-run fps survey cost a night),
  A/B controls before blaming commits, no edits while a run's build is in flight.
- The suite must be order-independent; today it requires --test-threads=1 and its baseline
  drifted unnoticed (668/9 vs the actual 665/12).

## What is already right (preserve it)

64-way sharded guest mutexes with TLS lookup cache and wake elision; the `debug_log!` gating
pattern; Brent's cycle detection in the evaluator; the pipeline key correctly excluding
buffer base addresses; the seqlock'd shared-mapping epoch; bounded event channel with
backpressure as policy; the completion-waker plumbing (it exists — the spin paths just
don't use it).
