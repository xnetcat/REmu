# Guest memory dump: descriptor-table cartography (2026-08-13)

Source: `20260813T171206Z-PPSA10112-app/frames/memdump` — 2973 mappings, 2.32 GiB, captured
t=100s. Same run's `guest.log` carries 9 `IMAGE-REFUSE` and 86 `STALE-VALU-DESCRIPTOR` records.
Scan scripts live in the session scratchpad (`heap/align.py`, `heap/final.py`); they read the
dump directly and need no repo state.

## Headline for stage 3

**There is no descriptor heap in this title.** The largest run of consecutive 32-byte-strided
T#s anywhere in 2.32 GiB is **15 descriptors (480 bytes)**. Not one array reaches 16. The
stage-3 design's two structural assumptions — a 512-slot heap, and a span walkable in uniform
32-byte steps — are both contradicted by the dump, and one decode formula in the brief is
wrong. Details below, then the concrete corrections.

## Method and its noise floor

T# validity predicate (SDK `agc/core/texture.h:230-278`): `TYPE` nibble (DWORD3 bits 31:28) in
8..15, `m_format` in 1..256, width/height in 1..16384, base address non-zero, 256-byte aligned
and inside a mapped range. Base address is reassembled per the SDK's `setDataAddress`:

```
address = (u64(word0) << 8) | (u64(word1 & 0xff) << 40)
```

Scanning every **dword-aligned** window (not just 32-byte-aligned) yields 74,589 candidates.
That predicate is loose enough to have a false-positive floor, which the data itself exposes:
isolated singletons distribute nearly uniformly across all eight 4-byte phases mod 32
(~900-3000 each), whereas descriptors inside real tables do not. **All population figures below
are restricted to tables of length >= 4**, where the floor is 2.4%.

## Q1 — Where the tables live

6,641 tables of length >= 4, holding ~40k descriptors, concentrated in ten VA buckets:

| VA bucket | tables |
| --- | --- |
| `0x804c……` | 2054 |
| `0x8053……` | 1026 |
| `0x804b……` | 884 |
| `0x8052……` | 817 |
| `0x8048……` | 772 |
| `0x8045……` | 543 |
| `0x8042……` | 198 |
| `0x8054……` | 135 |
| `0x804f……` | 115 |
| `0x8047……` | 90 |

These are not ten heaps. They are ~6.6k small per-draw resource blobs scattered across a bump
allocator, spaced 0x310-0x6c0 bytes apart within a bucket.

## Q2 — Length, and the 512-slot cap

| table length | count |
| --- | --- |
| 4-7 | 6003 |
| 8-15 | 638 |
| 16-63 | **0** |
| 64-511 | **0** |
| 512+ | **0** |

Max = 15. `REMU_HEAP_CAP = 512` over-provisions by **34x** against the observed maximum. The cap
is not wrong in the sense of clamping real data — nothing gets clamped — but every argument
buffer it sizes is 97% empty, and the encoder's per-draw walk cost scales with the cap, not the
content.

## Q3 — Stride and alignment: the assumption that breaks

**Stride within a table is genuinely 32 bytes** (44,194 of the inter-descriptor gaps are exactly
0x20, the dominant mode by 10x). That part of the design holds.

**Alignment does not.** Start-VA phase mod 32, for tables of length >= 4:

| phase | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tables | 1560 | 113 | 1417 | 16 | **1963** | 11 | 1541 | 20 |

The four 8-byte phases hold 6,481 of 6,641 (97.6%); the odd phases hold 160 (the FP floor).
**Only 23.5% of tables begin at a 32-byte-aligned VA**, and the most populous phase is +16, not
0. Verified by hand against a matched pair with byte-identical contents — `0x008053000e80`
(phase 0) and `0x008053489790` (phase 16) both decode to the same three descriptors
(type 9, fmt 56, 1920x1080, tile 27, bases `0x8038fc0000` / `0x801c400000` / `0x8034b70000`,
all backed).

**Tables are not the whole story either.** Dumping the neighbourhood of a refuse source shows
32-byte T#s interleaved with shorter records inside one region:

```
-0x78  T#  type9  base=0x…                 (32 bytes)
-0x58  T#  type9  base=0x…                 (32 bytes)   <- +0x20, on lattice
-0x38  [00000092 00fff000 05000000 00000000] + 64-bit pointer 0x80459062b8   (24 bytes)
-0x20  T#  type9 256x256 fmt169 base=0x0080567c0000     <- +0x38 from previous T#, OFF lattice
```

So a blind 32-byte walk from any base misparses this region after the first non-descriptor
record. There is no fixed T#+S# 48-byte pairing (gap=0x30 appears 101 times against 44,194 at
0x20) — the layout is a **packed per-shader resource blob** of mixed-width records, which is
what UE4's PS5 resource tables look like, not a bindless heap.

## Q4 — The refuse sources sit in a 16-byte record table

5 of the 9 distinct `source=` addresses have identical first 16 bytes:
`00000600 00fff000 06f00000 00000000`, repeated three times consecutively, and in all five the
nearest real T# is at exactly **-0x58**. That constant offset across five unrelated addresses is
structural, not a random index error.

The records immediately after the sentinels decode cleanly as V#s with stride 16:

| offset | base | stride | num_records |
| --- | --- | --- | --- |
| +0x30 | `0x008045906618` | 16 | 5 |
| +0x40 | `0x0074c0f52000` | 16 | 361 |
| +0x50 | `0x007120e0b940` | 16 | 11 |

All three backed. **This retires the "split-load GPU-assembled descriptor" conclusion**
(EVIDENCE.md:263-269). The eight SGPRs are not composed from two loads of a GPU-authored
descriptor. They are **one contiguous 32-byte read straddling two adjacent 16-byte records** in a
mixed-width table: `words[0..4]` is a sentinel record, `words[4..8]` is the next record, which
happens to be a V#. The "live V# whose base is X+0x340" is simply the next table entry.

Consistent with the readfirstlane theory in kind — the pointer lands somewhere it shouldn't —
but **the landing zone is a register-patch/resource table with 16-byte granularity, so the miss
distance is not a clean multiple of 0x20** and should not be modelled as one. The remaining 4 of
9 sources are unreliable anyway: two read as 32 bytes of zeros in the dump while the log's
`now=` showed content, and one has its nearest T# 0xb2c8 away. That is the known weakness of
`source=` as a single-dword value match (`upload.rs:605-611`), not new information.

## Q5 — Per-frame rewrite: caching should be content-keyed, not span-keyed

Of 6,641 tables of length >= 4, only **847 distinct byte-identical contents** — a **7.84x**
dedup ratio. 656 contents appear more than once, covering **97.1%** of all table instances. The
largest equivalence classes:

| copies | VA span | first deltas |
| --- | --- | --- |
| 276 | `0x804590c0f8` … `0x80524cd4f0` | 0x340, 0x340, 0x340 |
| 275 | `0x8045a0d9b0` … `0x8053504500` | 0x310, 0x310, 0x310 |
| 203 | `0x80458cd1d0` … `0x80544efe90` | 0x6c0, 0x360, 0x3cc0 |
| 104 | `0x804590ba38` … `0x805350a008` | 0x380, 0x1040, 0x3400 |
| 103 | `0x8045a0d080` … `0x8053504810` | 0x310, 0x310, 0xf50 |

The same content recurs at hundreds of *different* VAs spanning ~0xF00000 of address space —
a bump allocator re-emitting the same resource table per draw and per frame. **A span-keyed
content-version cache would miss almost every one of these**, because the span moves every time.
A content-hash-keyed cache collapses 6,641 encodes into 847.

## Corrections to the stage-3 plan

1. **Keep using `decode_image_descriptor`; do not hand-roll the base decode.** The formula
   circulated in the stage-3 brief — `word0 | (word1 & 0xffff) << 32` — understates the address
   by 256x and takes 8 bits too many. The correct decode is `(word0 << 8) | ((word1 & 0xff) << 40)`
   per SDK `setDataAddress`. Applied to a real descriptor, `word0=0x80862800` gives
   `0x8086280000` (a valid guest VA); the brief's formula gives `0x80862800`, which is not
   mapped. The landed code is already right —
   `crates/remu-gpu-metal/src/presenter/decode/image.rs:36` computes
   `((words[1] & 0xff) << 32 | words[0]) << 8`, algebraically identical, and
   `decode/targets.rs:607` and `execute.rs:344` agree. This is a warning about the brief, not a
   bug report: stage 3 must route through the existing decoder rather than reimplementing it.
2. **Drop `REMU_HEAP_CAP` from 512 to 16, or make it per-binding from the measured span.**
   Nothing in 2.32 GiB needs more than 15. Keep the clamp-and-log, but size the argument buffer
   to what the binding actually spans.
3. **Do not walk a span blindly in 32-byte steps.** Anchor the walk to the phase of the
   binding's base address and stop at the first slot that fails to decode, rather than striding
   through a fixed 512-slot window — the region past a short table contains 16- and 24-byte
   records that will decode as garbage T#s at a measurable rate. The design doc's runtime slot
   formula (`(base_offset + dynamic + imm) / 32`, line 25) is only sound when the array's phase
   equals the binding base's phase; that is worth asserting rather than assuming.
4. **Key the descriptor cache on content hash, not on guest span.** 7.84x hit rate available;
   span-keying gets close to zero because the allocator moves the table every draw.
5. **Tighten the T# predicate before trusting decoded slots.** The predicate used here has a
   ~2.4% false-positive rate against real tables and a much higher one against arbitrary memory.
   Requiring the base to be mapped and 256-byte aligned is necessary but not sufficient; a slot
   that decodes should also be cross-checked against a known resident or refused loudly.

## What this does not settle

The dump is one instant (t=100s), and the title is at a menu, not in gameplay. "No array
exceeds 15" is a strong statement about this title at this moment, not a general one about
UE4-on-PS5. It is, however, more than enough to say a 512-slot heap model is not what this
title needs.
