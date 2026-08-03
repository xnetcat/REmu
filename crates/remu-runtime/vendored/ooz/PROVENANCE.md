# Vendored ooz (Kraken / Mermaid / Selkie / Leviathan / LZNA / Bitknit)

- Upstream lineage: https://github.com/powzix/ooz (commit 05038060)
- Portable non-Windows `stdafx.h` pattern taken from https://github.com/rarten/ooz
  (SSE via SIMDe so arm64 and x86_64 both build).
- Path taken: decompress-only — `kraken.cpp` cut after `Kraken_Decompress`,
  plus `bitknit.cpp` and `lzna.cpp` (required by the Kraken dispatcher).
- The Windows CLI / DLL loader / compressor were deliberately not vendored.
- License: GPL-3.0-or-later (see LICENSE and the copyright header in kraken.cpp).

`stdafx.h` is a host shim (not an upstream file): it replaces Windows.h /
intrin.h with portable builtins and SIMDe SSE aliases. No algorithm changes.
