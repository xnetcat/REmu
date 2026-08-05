# Phase 1B handoff — TLS / `fs:` architectural data flow (2026-08-05)

Architectural data flow only. Phase 1C closed — see `docs/notes/phase1c-20260805-handoff/`.
Phase 1D closed — see `docs/notes/phase1d-20260805-handoff/`. Phase 1E–H not started.

## Design

Rosetta has no writable FS base. Guest `fs:` memory operands must resolve against the guest
thread pointer (`current_guest_thread_pointer`), not host address 0.

### Load helper (red-zone + vector state)

Shared `mov r64, fs:[0]` helpers (`tls.rs`) match the store helper's preserve contract:

- Skip the 128-byte SysV red zone **inside** the helper (`lea rsp,[rsp-128]`), because nine-byte
  call sites cannot fit a site-side skip.
- Full flags + GPR + XSAVE/XRSTOR prologue (same body as the store thunk).
- Undo the helper-local skip with `add rsp,128` then a plain `ret`. **Not** `ret 128` — that
  contract belongs only to store sites that do `lea rsp,[rsp-128]; call` before the helper.
- Destination RSP slot is unused (`ud2`); the decoder never targets register 4.

### Prefix-aware seeded decode

`link/fs_decode.rs` walks from entry / `DT_INIT*` / reloc targets / `.eh_frame` FDE starts and
classifies authenticated `fs:` sites: `LoadGpr` (all widths + displacements), `StoreImm`,
`StoreGpr`, `LeaGpr`, and `Other` (non-mov / indexed). No byte-substring matcher.

AOT rewrite table:

| Shape | Action |
| --- | --- |
| `mov r64, fs:[0]` | Shared per-destination helper |
| `mov r64, fs:[disp]` / `lea rN, fs:[disp]` | Per-site helper (lea first — never faults) |
| `mov dword fs:[disp], imm32` | Per-site store helper (Imm32 bits kept; iced zero-extends) |
| Remaining capacity overflow / `StoreGpr` / `Other` | Deferred to SIGSEGV backstop |

Per-site reserve is the store-imm pattern ceiling (+ optional cushion). Lea that exceeds capacity
is a loud refusal; store-imm / displaced loads that do not fit defer to the fault path.

### Fault-driven backstop

`tls_fault.rs` registers with `remu_memory::set_fs_fault_handler` and runs **before** write-watch
and null-address short-circuit. It classifies the faulting instruction, emulates absolute
mov/lea/store against the guest TP, advances RIP, or prints a loud refusal (never silent death
at address 0 for a recognized FS shape).

## Safeguards deleted

- Byte-scan `mov r64, fs:[0]` substitute matcher.
- Boundary-mismatch rewrite gate that only logged and skipped mid-instruction false hits.
- STT_FUNC-only immediate-store discovery (replaced by seeded decode + pattern ceiling for sizing).

## Live evidence (`4e0357f`, clean tree)

| Title | Outcome | TLS note |
| --- | --- | --- |
| Silent Hill 20s | external-watchdog-timeout; `runs/silent-hill/` (`run.json` clean) | `decode-summary aot_loads=9 aot_stores=164 fault_backstop_sites=0` |
| Demon's Souls 45s | guest-failure ~25s; `runs/demons-souls/` (`run.json` clean) | `aot_loads=3779 aot_stores=161`; Scream/mutex path still advances; next blocker still `sceAcmBatchStartBuffers` then SIGSEGV `addr=0x303030303030303` |
| Dreaming Sarah 20s | guest-failure ~0–1s; `runs/dreaming-sarah/` | TLS links (`aot_loads=692`); null SIGSEGV after missing `/app0/shaders/*.vert` — not attributed to TLS |
| Void Terrarium 20s | guest-failure ~1s; `runs/void-terrarium/` | TLS links (`aot_loads=91`); null SIGSEGV — sustained-run claim still stale |

Loadavg at DeS launch 1-minute ~3.85 (no FPS claim). Silent Hill launch ~5.65 (no timing claim).

## Checks

- `cargo test -p remu-runtime --lib -- --test-threads=1` filters `fs_decode` / `tls::tests` / `tls_fault` — green arm64 and `x86_64-apple-darwin`.
- `cargo check --workspace --all-targets` — both arches green.

## Commits

- `bcf90f6` runtime: preserve red zone and vector state in TLS load helpers
- `32ec77c` runtime: replace FS byte-scan substitute with prefix-aware decode
- `0e1437f` runtime: emulate missed fs: accesses on the SIGSEGV backstop
- `cd5280c` runtime: size FS helper reserve without false-positive byte scans
- `2d43742` / `c22080f` cushion sizing experiments (superseded by Imm32 + packing on `d9a6675`)
- `a2700f5` runtime: leave ud2 in unused RSP TLS load-helper slot
- `d9a6675` runtime: encode FS store-imm immediates as Imm32 bits
- `4e0357f` runtime: undo TLS load-helper red-zone skip before ret
- (this commit) docs: Phase 1B TLS handoff

## Loud refusals / refutations

- `sceAcmBatchStartBuffers` remains unimplemented — Demon's Souls audio blocker after mutex; **not** Phase 1B.
- A 64Ki per-site helper cushion was chasing `AddressOverflow` from `i32::try_from` on zero-extended
  Immediate32; that is refuted — keep Imm32 bits and pack lea-first instead.
- Void Terrarium / Dreaming Sarah still null-fault early; do not claim TLS closed those titles' crashes.
- Phase 1C closed — see `docs/notes/phase1c-20260805-handoff/`. Phase 1D closed — see `docs/notes/phase1d-20260805-handoff/`. Phase 1E–H not started.
