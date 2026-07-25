# Source and provenance policy

REmu is GPL-2.0-or-later. Existing GPL-2.0-or-later and compatible permissively licensed emulator
work may be adapted when doing so shortens the route to working games.

## Reusable inputs

- GPL-2.0-or-later code with its authorship and license notices preserved.
- MIT, BSD, Apache-2.0, Zlib, and similarly permissive code with required attribution.
- Public hardware, ABI, ELF, Vulkan, Metal, Direct3D, and shader-format documentation.
- Black-box observations, logs, traces, and compatibility facts from legally obtained software.

GPL-2.0-only code is not incorporated casually into the GPL-2.0-or-later tree. If it becomes the
fastest implementation source, the project-wide license compatibility and distribution terms must
be resolved explicitly first.

## Inputs that never belong in the repository

- Proprietary SDK headers or source code.
- Console keys, firmware, game executables, game assets, or copyrighted dump payloads.
- Decompiler output copied from proprietary binaries.
- Code with missing or unclear provenance.

## Required practice

- Record the upstream project and snapshot for materially adapted implementations.
- Preserve copyright and SPDX notices where source is copied or substantially adapted.
- Prefer synthetic repository tests. Local game tests are opt-in and path-configured.
- Keep gameplay traces bounded and exclude copyrighted payloads from logs and fixtures.

## Reference projects

Read-only snapshots of related emulators are kept locally under `research/upstreams/`, which is
excluded from version control. Architectural observations and license constraints from them:

| Project | Snapshot | Relevant lesson | License consequence |
| --- | --- | --- | --- |
| Kyty | `4733b7e1` | Older PS4/PS5 HLE design with native x86-64 execution and Vulkan | MIT; permissive, but obsolete assumptions require independent validation |
| KytyPS5 | `8a4600c5` | Current PS5-oriented RDNA2 decoding, custom shader IR, and Vulkan path | GPL-2.0-only; reuse requires an explicit GPL-version compatibility decision |
| SharpEmu | `ada67a19` | Cross-platform native x86-64 HLE, host-fault recovery, and backend-neutral guest GPU boundary | GPL-2.0-or-later; may be adapted with provenance and notices |
| SharpEmu Metal branch | `2ee32b89` | A direct Metal backend is practical without a Vulkan translation layer | GPL-2.0-or-later; may be adapted with provenance and notices |
| Obliteration | `16e981ec` | Useful Rust organization and host virtualization experiments, but virtualization is not the shortest route to games | MIT OR Apache-2.0; evaluate individual dependencies before reuse |
| shadPS4 | `4c72de07` | Mature PS4 HLE, testing, and compatibility workflows | GPL-2.0-or-later; reusable with provenance, but PS4 semantics must not be assumed to be PS5 semantics |
| RPCSX | `e8ae1481` | Additional PS4/PS5 graphics and HLE experiments | GPL-2.0; reuse requires an explicit GPL-version compatibility decision |

Adapted implementation details and their exact upstream files are recorded per subsystem in
[provenance.md](provenance.md).
