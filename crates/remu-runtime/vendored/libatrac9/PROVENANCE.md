# Vendored LibAtrac9

- Upstream: https://github.com/Thealexbarney/LibAtrac9
- Commit: efca2e3af35562a09a9bb6deed90e45b4b824dc4
- Path taken: the entire `C/src/` directory (decode-only ATRAC9 implementation).
- License: MIT (see `LICENSE`; Copyright (c) 2018 Alex Barney).

Why LibAtrac9 rather than libavcodec (plan P5's substrate): LibAtrac9 is a single,
self-contained ~2k-line C codebase with no external dependencies, buildable directly through
the `cc` crate. Standing up the whole ffmpeg/libavcodec is the large infrastructure task P5
owns; P1's own text permits "vendored LibAtrac9". It is the de-facto reference ATRAC9 decoder
already used by shadPS4, RPCSX and Kyty for exactly this AJM path.

The vendored source is compiled by `crates/remu-runtime/build.rs` via `cc`, with
`.warnings(false)` so the third-party code cannot perturb the workspace warning budget.
Only the public entry points in `src/libatrac9.h` are bound (see `ajm_engine::atrac9`):
`Atrac9GetHandle`, `Atrac9ReleaseHandle`, `Atrac9InitDecoder`, `Atrac9Decode`,
`Atrac9GetCodecInfo`.

No source files were modified from upstream.
