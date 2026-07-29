# REmu

REmu is an experimental PlayStation 5 compatibility layer written in Rust. Its goal is to run
legally dumped games directly on desktop operating systems without emulating the PlayStation shell.

The project maps and runs authorized x86-64 game dumps under Rosetta. The first small test title
currently boots through resource loading, submits AGC draw streams, compiles observed RDNA2 shaders
to Metal, and flips its display buffers. Early splash frames render correctly through the translated
path; complete render state and later gameplay draws are still in progress, so playability is not
claimed.

See the [architecture](docs/architecture.md), [implementation roadmap](docs/roadmap.md), and
[subsystem status](docs/status.md); adapted implementation sources are tracked in
[provenance](docs/provenance.md). Contributions are welcome — start with
[CONTRIBUTING.md](CONTRIBUTING.md).

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
cargo run -p remu-cli -- run /path/to/eboot.bin
cargo run -p remu-cli -- gpu-info
```

For repeatable Apple Silicon game runs, use the runner. It builds the x86-64 release, records
unresolved NIDs, and keeps logs and optional captures under `$TMPDIR/remu-runs` (override with
`REMU_RUN_ROOT`). With no second argument it is interactive and has no timer; pass a positive number
of seconds only for a bounded diagnostic run. Interactive launches run from a generated `REmu.app`
bundle so AppKit registers the process as a regular Dock application:

```sh
scripts/run-game.sh /path/to/eboot.bin
scripts/run-game.sh /path/to/eboot.bin 120
```

A non-path argument names a shortcut resolved from the environment, so frequently used titles do
not need their paths repeated:

```sh
export REMU_EBOOT_MYGAME=/path/to/eboot.bin
scripts/run-game.sh MYGAME
```

Frame capture is disabled by default because reading and writing a 4K guest buffer is expensive;
enable it with `REMU_CAPTURE_FRAMES=1`. `REMU_PRESENTATION_SIZE=window` (the default) renders the
host drawable at logical window resolution, `native` uses Retina backing resolution, and a value
such as `1920x1080` selects a fixed drawable size. These options scale only presentation: the guest
keeps the display-buffer dimensions it requested.

For one self-contained rendering investigation, set `REMU_DEEP_TRACE=1`. The run directory then
retains periodic frames, decoded AGC draw/register/program state, shader opcode coverage,
translated MSL and graphics LDS IR, plus VM, VFS, synchronization, timing, AVPlayer gate, stale
guest-thread, and image-alias traces. This mode perturbs timing and should not be used for FPS
measurements.

VideoOut normally limits translation look-ahead to 64 pending GPU work items and three accepted,
not-yet-presented flips. `REMU_FLIP_GPU_EVENT_BACKLOG` and `REMU_FLIP_PRESENT_BACKLOG` override
those limits; setting either to `0` disables that limit. Wall-clock AVPlayer frame skipping is
off by default and can be restored for diagnostics with `REMU_AVPLAYER_WALL_CLOCK_CATCH_UP=1`.

Enable Apple's in-window Metal Performance HUD with `REMU_METAL_HUD=1`; add
`REMU_METAL_HUD_LOG=1` to record its metrics. Automated input is disabled by default; opt into a
comma-separated Cross-button schedule with `REMU_AUTO_CROSS`. Detailed first-seen shader register dumps are opt-in through
`REMU_TRACE_AGC_SHADERS=1`.

For unattended Metal translation and frame capture without creating an AppKit window, set both
`REMU_HEADLESS_METAL=1` and `REMU_CAPTURE_FRAMES=1`. The normal interactive presenter remains the
default.

On Apple Silicon, use the crash-capturing probe wrapper during game bring-up. It builds the
x86-64 release binary, launches it under Rosetta and LLDB, and keeps the guest logs, backtraces,
registers, revision, and launch settings together under `$TMPDIR/remu-runs`:

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
