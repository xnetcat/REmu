# Memory-model / layout / residency audit (branch main @ 1058fdd5)

## Headline

**Every entry in the presenter's queued write-sets already carries a resolved physical identity, and the read path throws all of them away and re-resolves from the guest VA through the process-global registry lock — once per candidate, per batch, per binding, per draw.** That is the ~90k `guest_shared_backing_identity` calls per 5s window and the structural cause of the 0.35 ms `upload_global_buffers`.

`SharedEventGlobalBufferVersion.mapping_identity` (`presenter.rs:2040`), `PendingGuestWriteRange.physical_identity` (`presenter.rs:2114`), `PendingGuestMemoryWrite.mapping_identity` (`presenter.rs:2141`), `PendingGlobalWriteback.lease` (`presenter.rs:1812`) are all resolved at capture time. Then `contained_global_version_offset` (`cache.rs:1659`), `contained_guest_write_offset` (`cache.rs:1728`), `contained_global_writeback_offset` (`cache.rs:1823`) each call `guest_shared_backing_identity(address, byte_count)` with identical arguments every iteration (loops at `cache.rs:1428, 1463, 1486`).

## Tier 1 — the measured crawl

**1. Loop-invariant read identity re-resolved per candidate.** Per resolution: global RwLock read (`shared.rs:340`), BTreeMap descent, discarded Arc clone (`shared.rs:266`, `343-348`), two `Instant::now()` + two atomic RMWs (`shared.rs:339, 350-354`). Fan-out: bindings × (open + up to 4 committed batches) × candidates. **Fix:** resolve once in `queued_global_read_source` (`cache.rs:937`), pass `Option<GuestSharedMappingIdentity>` down, compare stored identities — zero lock acquisitions, pure refactor.

**2. The negative case is the expensive case** (`cache.rs:1396-1417`): when nothing overlaps virtually, `guest_shared_backing_any_range_overlaps(physical_candidates(), …)` resolves every candidate physically under one lock. Candidates already know their identity; same fix as #1.

**3. Per-call wall-clock instrumentation on the hottest path** (`shared.rs:339, 350-354`) — keep until #1 lands (it is the verification metric), then gate.

**Verify before fixing:** 0.35 ms/call ÷ ~25 resolutions/call ≈ 14 µs/resolution — 70-100× the uncontended BTreeMap cost: the signature of a **lock convoy**. If confirmed, removing acquisitions is worth far more than making resolution faster.

**4. Sample selection is an unindexed O(N) scan calling the same global lock** (`texture/upload.rs:838-911`): the `compatible` scan runs unconditionally when `texture_type == 9`, even after an exact-key hit; ~8 `share_*` predicates per candidate, several calling `guest_shared_mapping_identity` twice (`tiling/descriptors.rs:347-353`). ~100 residents × ~16 bindings/draw ⇒ up to ~10⁴ lock acquisitions per draw. `synchronize_sampled_aliases` (`sampled.rs:83-161`) repeats the shape. **Fix:** skip `compatible` on exact hit; drive from `resident_guest_textures.overlapping(span)`; memoise identity per (address, byte_count) per draw.

## Tier 2 — silent data loss

**5. Writeback hands authority to guest RAM before it can still refuse** (`readback.rs:191` vs `210-224, 246`): `subtract_resident_authority(…, StaleSource::GuestMemory, …)` runs first; the lease check and encode can then fail with RAM never written — every overlapping resident believes RAM is current; next read rebuilds from RAM = zeros. **Fix:** validate lease and encode first; publish bookkeeping only on success.

**6. Pressure eviction is guaranteed data loss** (`texture/residency.rs:1248-1259` + `readback.rs:1242`): `PressureEviction` is hard-wired to `RemuBookkeeping`, so its writeback always refuses; the resident is removed anyway, destroying the only copy of gpu_authored bytes. Masked only because the byte budget defaults to `None` (`presenter.rs:368-377`). **Fix:** allow eviction publish, or only evict clean residents.

**7. The `ensure` path drops GPU-only pixels with no writeback and no debt transfer** (`residency.rs:1474-1499`, admitted at `2771-2784`); 14 direct `resident_guest_textures.remove(...)` sites, no invariant. **Fix:** single `retire(descriptor, disposition)` chokepoint asserting `gpu_authored.is_empty() || disposition ∈ {Published, TransferredTo}`.

**8. RangeSet coarsening lets the GPU publish bytes it never wrote** (`texture/ranges.rs:59-70`, cap `ranges.rs:11`): past 256 runs, insert collapses to the bounding run — safe for `owned`, wrong for `gpu_authored` (written by the same call, `presenter.rs:1606-1620`; published by `readback.rs:206, 246`). This is the mechanism `readback.rs:1164-1166` blames for the SH SIGSEGV. **Fix:** coarsen `owned` only.

## Tier 3 — aliasing model

**9. Format and pitch are in the residency key**, so every view of one surface is a separate object (`resident_map.rs:11-14`, `presenter.rs:541-550`), reconciled per bind by a six-predicate disjunction (`tiling/descriptors.rs:147-375`) whose base case short-circuits on address equality alone (`descriptors.rs:338-340`); `gpu_content_executed` clauses (`upload.rs:868-872, 891-896`) are an explicit escape hatch for the SH black-menu bug layered on `resident_sample_authority_ok` (`sampled.rs:495-505`) — patched, not sound. **Fix:** key by physical bytes `(backing_id, backing_offset, byte_length, tiling, bpp)`; format views inside the entry via `newTextureViewWithPixelFormat` (machinery at `sampled.rs:322-338`).

**10. No epoch stamp ties a resident to the shared-mapping generation**: `guest_shared_mapping_epoch` is consulted in exactly one place in the GPU backend (`texture/watch.rs:334-353`). `Replaced` detection covers only the exact key (`upload.rs:758-779`) — precisely not the BGRA/RGBA case. **Fix:** `mapping_epoch` in `ResourceSpan` at admission; mismatch ⇒ `Replaced`.

**11. The interval index covers only level 0** (`resident_map.rs:132` vs `presenter.rs:1310-1312`): mip-tail/array-face aliasing escapes `overlapping()` and therefore `subtract_resident_authority` (`residency.rs:2642-2645`). **Fix:** index `resident.span`.

**12. Six more per-draw O(N) scans** the index makes unnecessary: `residency.rs:1500-1510`, `1535-1560`, `2484-2499`, `2501-2507`, `2723-2738`, `2209-2219`; `2464-2466`/`1603-1605` re-resolve identity per multisample entry per render-target mark; `resident_map.rs:212-218` linear dedupe; `IntervalList::insert` O(N) for `virtual_intervals` (`resident_map.rs:299-306`).

## Tier 4 — write tracking

**13. `notify_guest_memory_written_excluding` is O(all watched pages) per host write under the global mutex** (`write_watch.rs:100-130`), first taking a second global lock in `guest_write_alias_segments` (`shared.rs:503-554`) with two Vec allocations and a sort. Callers per packet: `submission/memory.rs:685`, `memory.rs:1700`, `texture/watch.rs:1305`, `av_player.rs:1421`. ~0.3-1 ms/call at ~200 watches. **Fix:** global page-indexed dirty bitmap + the sorted interval array `snapshot.rs:42-54` already builds.

**14. Demoted ranges re-run full teardown per rearm and always burn 4 snapshots** (`write_watch/state.rs:219-231`, `texture/watch.rs:1078-1093`, `MAX_STABLE_SNAPSHOT_ATTEMPTS = 4` at `watch.rs:53`); `CpuWriteWatch::is_demoted` (`write_watch.rs:358`) has no production caller. An 8 MB texture pays 4×8 MB memcpy per validation forever. **Fix:** surface `is_demoted()`; make demoted rearm one relaxed store.

**15. Global `Mutex<WatchState>` held across syscalls, allocations, multi-MB copies, and a handler-quiescence spin** (`state.rs:86`; `with_live_mapping` `state.rs:462` across whole-texture copies at `batch.rs:2494`, `memory.rs:1666`, `watch.rs:513,536`; `publish_snapshot` spins on `ACTIVE_HANDLERS` with mutex held, `snapshot.rs:120-133`). FS/TLS backstop decode runs first on every fault (`signal.rs:266-272`). `state.ranges` keyed by id ⇒ every guest mmap/mprotect linearly scans all watches (`state.rs:76`, `109-116`). Read-watch materializer (`signal.rs:446-498`, `state.rs:241-263`) is registered nowhere — dead weight on the fault path.

**16. Write journal: fixed 201 MiB ring, zero coalescing, O(4M) scan per query from signal context** (`guest_write_journal.rs:27`, 48 B/slot `:174-182`; scans `:405`, `:443`; `covering` uncapped `:409-411`; `uniform_byte` memcmps 8 MiB `:325-331`). **Fix:** page index, coalescing, cap `covering`.

## Tier 5 — per-draw allocation

Measured: `IndexedDraw` = 376 B; `ShaderEvaluation` = 1184 B; register payload 705×8 B = 5,640 B/draw; ≈105 heap + 2 Metal allocations ≈ 35 KB per draw.

**17. `prefetch_pipeline_work` builds a full key for every queued draw, every tick, before the cache check** (`events.rs:703-711`, `shader.rs:76-83`); key rebuilt on execute (`submission.rs:713`), hashed ≥5×, cloned per Blocked iteration. **Fix:** intern keys (`HashMap<u64, Arc<GraphicsPipelineKey>>`).

**18. `snapshot_small_readonly_global_bindings` deep-clones the whole `ShaderEvaluation` per stage per draw** (`shader_eval.rs:288-328`, `(*evaluation).clone()` at `:309`; `capture_global_binding_data: false` at `:345`; applied at `agc_draw.rs:690, 725`). **Fix:** `Arc<[u8]>` data or a side table keyed `(base_address, byte_count, generation)`.

**19. Index caches key on the whole index array by value** (`cache.rs:1282, 1301, 1311`): `indices.to_vec()` per call, hit or miss — 120 KB copy per 30k-index mesh draw. **Fix:** key on `(index_buffer_address, offset, count, size, base_vertex, primitive_type, restart_index)`.

**20. Assorted:** `IndexedDraw` deep-cloned twice on the cold path (`events.rs:1091`, `shader.rs:128`) — carry `Arc<IndexedDraw>`; index snapshots built in three passes up to 4 MiB (`shader_eval.rs:249-277`); `GuestCommandBatch::work` unreserved growth; dead fields `index_resource_address` (`resource.rs:112`), `draw_flags` (`resource.rs:127`); `scanout_target` 80 B inline.

## Recommended order

1. #1 + #2 (hoist read identity — the crawl fix; verify with existing counters).
2. #5 (three-line reorder; silent-corruption class).
3. #4 + #12 (route scans through the interval index).
4. #17 + #18 (intern keys; stop cloning evaluations).
5. #9 + #10 together (rekey residency physically + epoch stamps) — subsumes #7 and the write-attribution misses by construction.
