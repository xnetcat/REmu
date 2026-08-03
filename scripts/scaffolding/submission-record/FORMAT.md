# `submissions.bin` format draft (E3 / B23 — scaffolding)

Status: **draft + Python stub**. No runtime writer exists yet. Anything claiming a pass from this
file alone is fabricating success.

## Goals

- Capture enough AGC submit state that a headless `remu replay` can re-drive decode → translate →
  encode without the guest CPU.
- Stay small and legal: PM4 words + referenced guest ranges are emulator-observed state, not game
  assets. Store corpora outside the repo; reference by content hash.

## Proposed file layout (v0 draft)

Little-endian throughout.

```
magic:    b"REMUSub1"          # 8 bytes
version:  u32 = 1
flags:    u32                  # bit0 = has_referenced_memory_blob_index
count:    u64                  # number of SubmissionRecord entries
records:  SubmissionRecord × count
blobs:    optional concatenated referenced-memory payloads (if flags.bit0)
```

### `SubmissionRecord` (fixed header, then payloads)

```
seq:            u64
queue_id:       u32
pad:            u32            # reserved 0
timestamp_ns:   u64            # host capture time; replay ignores wall clock
dcb_len:        u32
ccb_len:        u32
range_count:    u32
pad2:           u32
dcb_bytes:      u8 × dcb_len   # display command buffer words as bytes
ccb_bytes:      u8 × ccb_len
ranges:         GuestRange × range_count
```

### `GuestRange`

```
guest_va:       u64
length:         u64
blob_offset:    u64            # into trailing blobs section; UINT64_MAX = not captured
```

Referenced memory is captured **lazily on first use** of a range (plan text). Uncaptured ranges
must fail loud at replay time — never invent zeros.

## Env (future, not wired)

- `REMU_RECORD_SUBMISSIONS=1` — enable recording into `$run_dir/submissions.bin`.
- Default off. Kill-switch cycle only after a measured-safe landing.

## Stub API

`record_stub.py` encodes/decodes the v0 header and refuses to claim replay readiness.
