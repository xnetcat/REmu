# Unsafe review rule (plan M4)

REmu treats `remu-memory` as the designated unsafe boundary for guest-addressable storage.
New `unsafe` outside that crate is not free: it must name the invariant it relies on and who
enforces that invariant, so a later reader can tell proof from folklore.

## Rule

Any new `unsafe` block or `unsafe fn` **outside** `crates/remu-memory` must include a `// SAFETY:`
(or `/// SAFETY:`) comment that states, in plain language:

1. **Invariant** — what must be true for the operation to be sound (bounds, alignment, lifetime,
   exclusive access, ABI layout, FFI contract, …).
2. **Enforcer** — which code establishes and maintains that invariant (a validated
   `GuestAccessPin` / `GuestPtr` / `GuestSlice`, a Metal buffer length wrapper, a lock held across
   the borrow, a type that owns the allocation, an SDK-backed layout constant, …).
3. **Ordering** — when the operation publishes or observes guest-visible shared state (GPU labels,
   HLE sync words, sideband results, other cross-thread guest memory), name the memory-ordering
   contract: release publication of payload before label, acquire observation of a label, or an
   explicit `guest_publish_release_fence` between unordered payload stores and the label store.
   Prefer the shared `remu-memory` word accessors (`store_guest_visible_u64` /
   `publish_guest_shared_u64_if_identity` / `load_guest_visible_u64`) over ad-hoc `memcpy` or
   un-ordered volatile stores of fence labels.

If either the invariant or enforcer half cannot be named, the change does not land as raw `unsafe`.
Prefer a checked helper in `remu-memory` (guest pointers / ordered word publication) or an existing
length-checked wrapper (e.g. gpu-metal `MetalBytes`) over another one-off `from_raw_parts`.

## Applies to

- New production `unsafe` in any crate other than `remu-memory`.
- Expanding an existing `unsafe` block with a new assumption (update the comment; do not leave a
  stale proof).

## Does not replace

- `remu-memory` still documents its own sites (plan M3). This rule is the gate that stops new
  unprovenanced guest/FFI `unsafe` from accumulating in runtime, gpu-metal, platform, and friends.
- Clippy / CI may later automate parts of this; until then, reviewers enforce it on every PR that
  adds `unsafe` outside `remu-memory`.

## Migration note

Plan M1's `GuestPtr` / `GuestSlice` layer exists so runtime HLE sites can retire raw guest
`from_raw_parts` instead of re-justifying them. Prefer migrating a call site to that layer over
writing a new SAFETY comment that merely restates "the guest promised this buffer".
