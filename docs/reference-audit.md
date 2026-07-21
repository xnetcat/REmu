# Reference audit

This audit records architectural observations and license constraints. It is not an implementation
specification. The snapshots are kept locally under `research/upstreams/`, which is excluded from
version control.

| Project | Snapshot | Relevant lesson | License consequence |
| --- | --- | --- | --- |
| Kyty | `4733b7e1` | Older PS4/PS5 HLE design with native x86-64 execution and Vulkan | MIT; permissive, but obsolete assumptions require independent validation |
| KytyPS5 | `8a4600c5` | Current PS5-oriented RDNA2 decoding, custom shader IR, and Vulkan path | GPL-2.0-only; reuse requires an explicit GPL-version compatibility decision |
| SharpEmu upstream / linked fork | `ada67a19` / `30fdd8d6` | Cross-platform native x86-64 HLE, host-fault recovery, and backend-neutral guest GPU boundary | GPL-2.0-or-later; may be adapted with provenance and notices |
| SharpEmu Metal branch | `2ee32b89` | A direct Metal backend is practical without a Vulkan translation layer | GPL-2.0-or-later; may be adapted with provenance and notices |
| Obliteration | `16e981ec` | Useful Rust organization and host virtualization experiments, but virtualization is not the shortest route to games | MIT OR Apache-2.0; evaluate individual dependencies before reuse |
| shadPS4 | `4c72de07` | Mature PS4 HLE, testing, and compatibility workflows | GPL-2.0-or-later; reusable with provenance, but PS4 semantics must not be assumed to be PS5 semantics |
| RPCSX | `e8ae1481` | Additional PS4/PS5 graphics and HLE experiments | GPL-2.0; reuse requires an explicit GPL-version compatibility decision |

The linked `galvesribeiro/sharpemu` Metal branch and the corresponding current direct-Metal source
areas were also compared. REmu can now adapt compatible GPL-2.0-or-later implementation work from
these trees as long as provenance and notices remain recorded.

## Decisions resulting from the audit

- Use a compatibility layer with native guest execution, not full-system virtualization.
- Target the x86-64 macOS process under Rosetta first; keep CPU execution behind a backend boundary
  so a future native AArch64 translator can replace it.
- Use direct Metal on macOS, Direct3D 12 on Windows, and Vulkan on Linux.
- Decode AGC/RDNA2 commands and shaders into guest-domain structures and a custom typed IR before
  lowering independently to MSL, HLSL/DXIL, or SPIR-V.
- Implement only the operating-system services needed by games. Do not implement the PS5 shell.
- Prefer a hybrid module strategy: HLE core services and load compatible user-dumped modules when
  that is the shorter, legally supportable path.

## Initial compatibility fixtures

- Dreaming Sarah (`PPSA02929`) is the first container/loading target because its dump is small and
  its dependency surface is narrow.
- The decrypted Superliminal (`PPSA06084`) executable is the first standard-ELF loader fixture.
- Larger titles remain later regression targets after the loader, memory, threading, and graphics
  foundations are measurable.

No game paths, payloads, keys, firmware, or derived assets belong in the repository.
