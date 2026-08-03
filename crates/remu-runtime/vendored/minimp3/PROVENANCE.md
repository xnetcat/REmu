# Vendored minimp3

- Upstream: https://github.com/lieff/minimp3
- Commit: ea99364f61c14656440e8d77e9c233ccf3124633
- Path taken: `minimp3.h` (single-header MP3 decoder).
- License: CC0-1.0 / public domain equivalent (see LICENSE).

Compiled by `crates/remu-runtime/build.rs` into a small C translation unit that
`#define MINIMP3_IMPLEMENTATION` before including the header. Used by the AJM
MP3 codec (plan P6).
