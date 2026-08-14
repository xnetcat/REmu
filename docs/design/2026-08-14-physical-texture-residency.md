# Physical-byte texture residency

Status: proposed Phase 3 design; no implementation in this change.

Audit scope: memory/residency findings 7-10, building on the transactional writeback and
full-span interval-index work. The current implementation files residents by
`GuestTextureDescriptor`, stores format and pitch in that key, records an admission-time
`ResourceSpan` without a mapping stamp, uses one coarsening `RangeSet` for both authority and
`gpu_authored`, and has direct `resident_guest_textures.remove(...)` calls across residency,
sampling, scanout, memory submission, mip, upload, and watch paths. Those properties make aliases
look like independent allocations and let a removal bypass publication-debt checks.

## Goal and non-goals

One physical byte interval has one ownership/debt ledger. Pixel format, guest VA, pitch, and the
selected mip/layer range are bindings or views of that ledger, not resident identity. A mapping
change must invalidate a binding without silently destroying the old physical allocation. A
resident with unpublished GPU-authored bytes may leave residency only after those exact bytes were
published or transferred to another live resident.

This change does not redesign the guest VM, write-watch, or write-journal implementations; merge
unrelated physical backings; pretend incompatible tiling or element layouts are views; add CPU
format conversion as a silent fallback; or cover buffer residency, AppKit threading, and shader
optimization. Metal-incompatible views may use an explicit GPU materialization, but inability to
make one is an error and a counter, never a zero texture or guest-RAM rebuild while debt exists.

## Data model

The types below are semantic sketches. All byte intervals are checked, half-open, and use backing
offsets, never guest VAs.

```rust
struct PhysicalByteRange {
    backing_id: u64,
    start: u64,
    len: u64,
}

struct PhysicalLayoutKey {
    tile_mode: u32,
    element: ElementLayout, // block width/height and bytes, not numeric format
    samples: SampleLayout,  // only fields which change physical addressing
    plane: PlaneLayout,
}

struct PhysicalAllocationKey {
    bytes: PhysicalByteRange,
    layout: PhysicalLayoutKey,
}

struct ResidentId {
    slot: u32,
    generation: u32,
}
```

`ResidentArena` owns generational slots. `by_key: HashMap<PhysicalAllocationKey, ResidentId>`
answers exact allocation hits; a backing-offset interval index answers overlap and containment.
Removing a slot increments its generation, so queued write attribution and stale-source links can
detect use-after-retire instead of accidentally naming a new descriptor at the same address. The
arena retains a backing-object pin when the memory model supplies one. That prevents backing-ID
reuse, but grants no publication licence.

Pitch, dimensions, numeric pixel format, texture type, and selected levels/layers describe a view:

```rust
struct SubresourceLayout {
    level: u32,
    layer: u32,
    bytes: ExactByteSet, // relative to PhysicalAllocationKey.bytes.start
    width: u32,
    height: u32,
    row_elements: u32,
    metal_level: u32,
    metal_slice: u32,
    metal_z: u32,
    tail_group: Option<u32>,
}

struct ViewKey {
    pixel_format: MTLPixelFormat,
    texture_type: MTLTextureType,
    base_level: u32,
    level_count: u32,
    base_layer: u32,
    layer_count: u32,
    swizzle: TextureSwizzle,
    shape: ViewShape, // dimensions and pitch interpretation
}
```

The complete subresource table is computed once from the decoded descriptor and checked against
the allocation byte range. Packed mip tails deliberately share the same byte runs. A request is
served by an existing allocation when it is an exact key hit or its requested byte runs match
known subresources of a containing allocation. `newTextureViewWithPixelFormat` is used only when
the formats share the existing Metal view class, the subresource selection is representable, and
the base texture has `PixelFormatView` usage. Differing numeric formats therefore share a ledger.

An incompatible pitch/shape is not a second authority owner. It is a `Materialization` inside the
same allocation, with a content generation and exact covered subresources. A GPU blit/retile must
complete before that materialization becomes current. The existing descriptor compatibility
disjunction remains only during migration and is deleted at cutover.

`ResidentAllocation` contains the physical key, subresource table, cached views/materializations,
one content generation, a write watch/binding set, authority, and publication debt:

```rust
struct ResidentAllocation {
    id: ResidentId,
    key: PhysicalAllocationKey,
    subresources: Arc<[SubresourceLayout]>,
    materializations: HashMap<MaterializationKey, Materialization>,
    views: HashMap<ViewKey, Texture>,
    authority: AuthoritySet,
    gpu_debt: GpuDebtSet,
    content_generation: u64,
    bindings: SmallVec<[GuestBinding; 2]>,
    state: ResidencyState,
}
```

`AuthoritySet` and `GpuDebtSet` are distinct types even if both initially wrap the same interval
implementation. Authority may later use a proven conservative representation. `GpuDebtSet` is an
exact sorted union: it coalesces only overlapping or adjacent runs and never replaces separated
runs with their bounding interval. There is no run-count truncation. A fragmentation high-water
mark requests early publication or pins the allocation; it never widens debt. Debt records the
exact union of admitted write footprints. When shader access cannot refine a write, the admitted
footprint is the complete bound subresource and is counted as such; later fragmentation handling
still may not widen it.

## Mapping bindings and ABA policy

Physical allocation identity and guest-VA validity are separate:

```rust
struct GuestBinding {
    va_start: u64,
    len: u64,
    physical: PhysicalByteRange,
    mapping_epoch: u64,
    mapping_token: MappingRangeToken,
    resident: ResidentId,
    view: ViewKey,
}
```

`remu-memory` gains one range-resolution operation which, under the shared-mapping seqlock,
returns the full physical range, a retained backing pin where possible, and a token covering every
mapping record that made up the range. Each registry insertion receives a monotonically increasing
record generation. The token includes those generations, so unmap/remap to the same numeric
`(backing_id, offset)` is observable for virtually keyed mappings.

Resolution reads an even epoch, resolves the complete range, then accepts only if the same even
epoch is still visible. An odd or changed epoch retries. Use has a cheap path when the global epoch
still equals `GuestBinding.mapping_epoch`. On drift it re-resolves once:

- For durable physically keyed backing, the binding may refresh when its complete original VA
  range still resolves to the same physical range. A different live alias keeps the allocation
  recoverable, but does not silently retarget this descriptor's binding.
- For virtually keyed backing, both the physical tuple and mapping-record token must match. An ABA
  remap is `Replaced`, even if the VA and offset numbers happen to be identical.
- A different or unresolved range invalidates only the `GuestBinding`. It does not erase the
  `ResidentAllocation`. Upload, readback, write attribution, and publication refuse the stale
  binding and locate/revalidate a live alias or leave the allocation pinned.

The epoch is an invalidation hint, not durable identity. The range token is the ABA proof. A
retained backing pin preserves object identity but is not permission to publish; publication still
needs a live, baseline-guarded `GuestRangeLease` for the exact runs.

## Authority and debt transitions

The invariants are:

1. Every `gpu_debt` run lies within the allocation's physical bytes and current authority.
2. No debt byte is lost, widened, or published from a different layout by bookkeeping.
3. A view creation changes no authority, debt, or content generation.
4. An ownership change becomes visible only after every fallible Metal operation needed to
   preserve content has encoded successfully; completion-dependent changes carry a fence/ticket.
5. A replaced mapping is not evidence that GPU debt is obsolete.

The state transitions are:

- **Admit from guest:** validate `GuestBinding`, create/upload the materialization, then claim its
  exact physical runs and subtract those runs from overlapping authority. Debt starts empty.
- **GPU write:** after the encoder accepts the operation, claim the exact `WrittenRegion` runs,
  subtract those runs from overlapping residents, add them to `gpu_debt`, and advance content
  generation. The write footprint is stored in physical-offset space.
- **Create compatible view:** cache a Metal view under `ViewKey`; no byte transition.
- **Materialize/retile:** encode a copy from a current materialization, then transfer the exact
  authority and debt runs with an ordered `TransferReceipt`. The source stays live until encode
  succeeds; an asynchronous copy does not become selectable before its fence.
- **CPU/guest write:** after the write watch or journal proves the exact guest-written runs, guest
  memory wins those runs. Subtract them from authority and debt before any clean retirement.
- **Publish:** acquire and validate the live binding/lease first, encode the exact debt runs, and
  retain a `PublishTicket`. Only successful completion may turn those runs guest-authoritative and
  remove them from debt. `GuestModified` baseline outcomes resolve only the reported runs in favor
  of the guest. Refusal, encoding failure, command-buffer error, or revoked lease leaves debt.
- **Mapping replacement:** invalidate the binding. Clean allocations can retire; dirty allocations
  find another live alias, transfer debt, or remain pinned with a loud refusal counter.

## The retirement chokepoint

`ResidentArena::retire(id, disposition)` is the only operation allowed to delete a resident slot.
The map and interval-index removal methods become private to it. Temporary remove/reinsert patterns
used to satisfy borrowing are replaced with scoped arena mutation; they are not retirements.

```rust
enum RetireDisposition {
    Clean { reason: RetireReason },
    Published { receipt: PublishReceipt },
    TransferredTo { target: ResidentId, receipt: TransferReceipt },
}
```

`Clean` requires empty debt. `Published` requires a completed receipt whose resolved physical runs
cover all debt present at retirement and whose mapping lease matches the source allocation.
`TransferredTo` requires a live, generation-matching target in the same physical offset space;
the target must already hold authority and debt for every transferred run, at an ordered content
generation. Receipts are unforgeable module-private values created only by the successful
transition functions.

Retirement first validates the disposition, then removes the exact-key and interval-index entries,
binding/view caches, watches, sampled/multisample references, and stale-source backreferences, and
finally increments the slot generation. Validation failure changes nothing and reports an error;
pressure eviction skips the resident, mapping replacement pins it, and shutdown reports unsettled
debt rather than providing a force-delete escape hatch. This makes the invariant executable:

```text
gpu_debt.is_empty() || disposition is completed Published or TransferredTo
```

## Migration and shadowing

The cutover is deliberately reversible and keeps one source of truth at a time:

1. Split `GpuDebtSet` from authority and add the >256-hole tests; retain descriptor residency.
2. Add stamped range resolution and mapping-record generations in `remu-memory`; carry a
   shadow `GuestBinding` beside current `ResourceSpan` and compare every revalidation result.
3. Introduce `retire()` while the descriptor map remains authoritative. Convert every direct
   resident removal and make raw removal private.
4. Add the generational arena, exact physical-key map, and full-span physical interval index as a
   shadow of the descriptor map. Assert insert/lookup/overlap/retire equivalence and emit mismatch
   telemetry, but route no rendering through it.
5. Carry `ResidentId` and binding stamps through decoded resources, selected residents, queued
   write footprints, shared-event attribution, stale-source links, and readback tickets.
6. Move compatible pixel-format views into `ResidentAllocation`; descriptor routing remains a
   checked fallback while exact physical hits are enabled behind `REMu_PHYSICAL_RESIDENCY=1`.
7. Add complete mip/layer/tail `SubresourceLayout` matching and internal GPU materializations.
8. Make the physical arena authoritative, leave the descriptor map as a shadow for one evidence
   cycle, then delete it and the six-predicate compatibility fallback after zero mismatches.

Each implementation stage is one independently revertible commit. The fallback flag selects the
old source of truth during stages 4-7; it never disables exact debt or the retirement invariant.
After cutover, rollback means reverting the physical-routing commits, not weakening `retire()` or
reintroducing debt coarsening.

## Telemetry and failure policy

A cheap five-second `[REMu][PERF-RESIDENCY]` line reports exact/containment/subresource hits,
format-view hits, materializations, epoch fast/slow/replaced/ABA results, current/peak debt bytes and
runs, publish/transfer completions and refusals, retire counts by disposition, dirty-retire
refusals, pinned dirty allocations, and shadow-index mismatches. Startup prints whether physical
routing and shadow comparison are enabled. Rare detailed diagnostics include `ResidentId`, both
physical ranges, mapping epochs/tokens, disposition, and debt runs; they are rate-limited but their
counters are not.

Hard acceptance invariants are zero shadow mismatches, zero stale-`ResidentId` uses, zero dirty
retirements without a receipt, zero widened-debt events (the counter exists only as an assertion
tripwire), and no direct map removals outside `retire()`.

## Tests and release gates

Unit/model tests cover checked physical-range arithmetic; exact debt with more than 256 separated
runs; split/subtract/intersection; mip, array, cube, and packed-tail subresource spans; compatible
and incompatible view classes; same bytes through BGRA/RGBA and UNORM/UINT views; pitch-changing
materialization; overlapping pooled tenants; and generational stale-ID rejection.

Mapping tests cover stable-epoch fast paths, unrelated epoch drift, direct-memory alias moves,
virtual-keyed unmap/remap ABA to the same numbers, partial range replacement, odd-epoch retry, and
publication refusal through a stale binding. Retirement model tests exercise every disposition,
pressure eviction, mapping replacement, command-buffer failure, partial publication, transfer
ordering, shutdown debt, and randomized transition sequences while checking the invariants after
every step. Shadow tests replay the same operations against descriptor and physical indexes and
compare exact, containment, overlap, and retirement results.

Every affected crate is tested in release mode, serially:

```sh
cargo test -p remu-memory --release -- --test-threads=1
cargo test -p remu-gpu --release -- --test-threads=1
cargo test -p remu-gpu-metal --release -- --test-threads=1
cargo test -p remu-runtime --release -- --test-threads=1
```

`remu-memory` and `remu-gpu-metal` must have zero failures (the audit floors were 93/0 and 504/0;
new tests increase, never reduce, the pass count). `remu-gpu` must have zero failures.
`remu-runtime` may have only its 12 named deterministic baseline failures and no new failure.
Before a phase merge, run the specified 280-second PPSA10112 evidence command from a clean tree,
require the physical-residency tripwires above to remain zero, inspect the captured frames and
translation/residency telemetry, and run the PPSA02929 regression gate. Do not fast-forward `main`
or push the phase until those gates pass.
