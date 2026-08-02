# Contributing to REmu

REmu is a GPL-2.0-or-later PlayStation 5 userspace compatibility layer written in Rust. The
project values correct, general, source-backed emulation over quick title-specific results. This
document describes the engineering rules that keep the codebase in that state.

## Engineering principles

- Build a general emulation layer, not per-title patches. Every fix must follow PS5 ABI,
  AGC/RDNA2, kernel, library, or file-format semantics and be covered by a synthetic regression
  test that does not depend on game data.
- Never add a stub, silent no-op, guessed success result, dummy resource, fake shader output,
  fallback frame, or title/product-ID special case. If behavior is not yet implemented, fail
  explicitly with enough structured state to implement it correctly later.
- When an unimplemented ABI, NID, PM4 packet, shader instruction, format, or synchronization
  operation is encountered, implement its full reusable semantics rather than only the narrow
  values seen in one trace.
- Keep AGC, PM4, shader IR, memory, synchronization, and resource types backend-neutral. The
  Metal, D3D12, and Vulkan backends own API-specific concepts; guest-domain crates must not
  leak them.
- Rendering correctness and performance are co-equal. A frame is not accepted because it merely
  appears, and a correct path is not accepted if avoidable copies, synchronous compilation, or
  unbounded queues make it unusable.

## Research method

Do not rediscover behavior that is already established in public sources. Build understanding
from several independent directions before coding, then implement the shared hardware or ABI
invariant:

- AMD GFX10/RDNA2 ISA documentation, GPUOpen material, and public PM4 packet definitions.
- Mesa (RADV/AMDGPU) and Linux kernel AMDGPU driver code for public packet, register, format,
  and synchronization behavior. Distinguish hardware facts from driver policy.
- The LLVM AMDGPU backend for instruction encodings.
- Apple's Metal and Metal Shading Language documentation for backend work.
- Public ELF/System V AMD64 ABI documentation and FreeBSD/POSIX behavior for loader and kernel
  work; confirm PS5 deviations from real binaries rather than assuming PS4 behavior.
- Compatible open-source emulators, within the rules of [docs/source-policy.md](docs/source-policy.md).

Treat references as research, not unquestioned truth: triangulate between sources, write
synthetic tests, and record materially adapted sources in
[docs/provenance.md](docs/provenance.md).

## Legal constraints

- Never commit game files, console keys, firmware, proprietary SDK material, copyrighted assets,
  or decompiler output.
- Test only with content you are legally entitled to use. Game paths stay in your environment
  (see the `REMU_EBOOT_<NAME>` shortcuts in `scripts/run-game.sh`), never in the repository.
- Follow [docs/source-policy.md](docs/source-policy.md) before adapting any external code.

## Validation

Run the full check suite before submitting changes:

```sh
cargo fmt --all -- --check
cargo test --workspace -- --test-threads=1
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --target x86_64-apple-darwin -- --test-threads=1
cargo clippy --workspace --all-targets --target x86_64-apple-darwin -- -D warnings
python3 -m unittest discover -s scripts/tests
```

`--test-threads=1` is required until the order-dependent runtime suite recorded in
[docs/status.md](docs/status.md) is fixed. Without it the runtime tests fail intermittently for
reasons unrelated to the change under test, which makes the suite useless as a gate.

Three `remu-shader-metal` tests compile generated MSL with Apple's `metal` tool and fail without
it. Install it once with `xcodebuild -downloadComponent MetalToolchain`; they are the only tests
that prove generated MSL actually compiles.

The x86-64 target runs under Rosetta 2 on Apple Silicon and is the configuration that actually
executes guest code, so both native and x86-64 checks matter.

Prefer deterministic evidence over repeated live runs: frame dumps, PM4/command dumps, shader
dumps, structured log diffs, and import audits (`scripts/audit-import-coverage.py`). Use bounded
headless runs (`REMU_HEADLESS_METAL=1`) for GPU validation and interactive windows only when
window, input, or presentation behavior is itself under test.

## Commits

Keep commits small and focused, with lowercase imperative subjects (`resolve per-thread dynamic
TLS addresses`). When a change adapts external semantics, name the source and invariant in
`docs/provenance.md`.
