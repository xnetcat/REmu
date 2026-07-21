# Clean-room development policy

REmu is Apache-2.0. GPL-licensed emulator implementations are not source material for REmu code.

## Allowed inputs

- Public hardware and ABI documentation.
- Standard specifications such as ELF, DWARF, SPIR-V, Vulkan, Metal, and Direct3D.
- Black-box observations from legally obtained software and hardware.
- Logs, traces, test cases, and compatibility facts that describe external behavior.
- MIT, BSD, Apache-2.0, Zlib, or similarly permissive code after license review and attribution.

## Prohibited inputs

- Copying, mechanically translating, or closely paraphrasing GPL or unknown-license source code.
- Copying proprietary SDK headers, keys, firmware, game code, assets, or decompiler output into the
  repository.
- Committing test dumps or derived copyrighted assets.
- Implementing behavior whose only provenance is code that cannot legally be incorporated.

## Required practice

Compatibility work should record the observed input, output, platform, and source category without
including copyrighted payloads. Tests should use synthetic fixtures whenever possible. Tests that
require local game dumps must be ignored by default and reference paths through local configuration,
never hard-coded repository data.

