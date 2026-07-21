# REmu

REmu is an experimental PlayStation 5 compatibility layer written in Rust. Its goal is to run
legally dumped games directly on desktop operating systems without emulating the PlayStation shell.

The project is at the early boot-probing stage. It maps and enters authorized game dumps under
Rosetta, but does not reach a game loop yet.

See the [architecture](docs/architecture.md), [implementation roadmap](docs/roadmap.md), and
[reference audit](docs/reference-audit.md). Adapted implementation sources are tracked in
[provenance](docs/provenance.md).

## Direction

- HLE operating-system services with native x86-64 guest execution.
- Direct Metal on macOS, Direct3D 12 on Windows, and Vulkan on Linux.
- x86-64 macOS builds under Rosetta 2 initially; a native AArch64 translator is a later project.
- A hybrid module model: HLE core services plus optional modules dumped by the user.
- CLI and debugger before a graphical game library.

## Current commands

```sh
cargo run -p remu-cli -- host-info
cargo run -p remu-cli -- inspect /path/to/eboot.bin
cargo run -p remu-cli -- imports /path/to/eboot.bin
cargo run -p remu-cli -- map /path/to/eboot.bin
cargo run -p remu-cli -- link /path/to/eboot.bin
cargo run -p remu-cli -- probe /path/to/eboot.bin
cargo run -p remu-cli -- gpu-info
```

On Apple Silicon, use the crash-capturing probe wrapper during game bring-up. It builds the
x86-64 release binary, launches it under Rosetta and LLDB, and keeps the guest logs, backtraces,
registers, revision, and launch settings together under `.remu/runs/`:

```sh
scripts/probe-game.sh /path/to/eboot.bin 2000000 120
```

The final argument is the diagnostic timeout in seconds; pass `0` for an interactive run with no
timer.

Build the initial Rosetta target on Apple Silicon with:

```sh
cargo build --workspace --target x86_64-apple-darwin
```

## Legal and source policy

REmu contains no games, console keys, firmware, or proprietary system software. Contributors must
test only material they are legally entitled to use. Compatible GPL and permissively licensed
emulator code may be adapted when its provenance, notices, and license obligations are preserved.
See [docs/source-policy.md](docs/source-policy.md).

REmu is not affiliated with or endorsed by Sony Interactive Entertainment. PlayStation is a
trademark of Sony Interactive Entertainment Inc.

## License

GPL-2.0-or-later.
