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
