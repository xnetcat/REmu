#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 {SHORTCUT|/path/to/eboot.bin} [timeout-seconds]" >&2
    echo "  A non-path SHORTCUT resolves through the REMU_EBOOT_<SHORTCUT> environment" >&2
    echo "  variable, e.g. REMU_EBOOT_MYGAME=/path/to/eboot.bin $0 MYGAME" >&2
    exit 64
fi

if [ "$(uname -s)" != "Darwin" ]; then
    echo "error: the Rosetta game runner currently requires macOS" >&2
    exit 69
fi

game_arg=$1
if [ -f "$game_arg" ]; then
    game_path=$game_arg
    game_name=$(basename -- "$(dirname -- "$game_path")")
else
    # A non-path argument names a launch shortcut configured entirely by the caller's
    # environment, keeping game locations out of the repository.
    shortcut=$(printf '%s' "$game_arg" | tr '[:lower:]' '[:upper:]' | tr '-' '_' | tr -cd 'A-Z0-9_')
    game_path=$(printenv "REMU_EBOOT_$shortcut" || true)
    if [ -z "$game_path" ]; then
        echo "error: no file at '$game_arg' and REMU_EBOOT_$shortcut is not set" >&2
        exit 64
    fi
    game_name=$(printf '%s' "$shortcut" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
fi
# Interactive launches stay open until the guest exits or the user closes the window. Pass a
# positive second argument only for a deliberately bounded diagnostic run.
timeout_seconds=${2:-0}
if [ ! -f "$game_path" ]; then
    echo "error: game executable not found: $game_path" >&2
    exit 66
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(dirname -- "$script_dir")
bundled_metal_compiler="$repo_dir/.remu/toolchains/Metal.xctoolchain/usr/bin/metal"
# Offline libraries can be retained across launches, avoiding tens of seconds of repeated cold
# runtime compilation for Unity's large shader set. Set REMU_USE_BUNDLED_METAL_COMPILER=0 to
# diagnose Apple's in-process source compiler explicitly.
if [ "${REMU_USE_BUNDLED_METAL_COMPILER:-1}" != 0 ] && \
    [ -z "${REMU_METAL_COMPILER:-}" ] && [ -x "$bundled_metal_compiler" ]; then
    REMU_METAL_COMPILER=$bundled_metal_compiler
    export REMU_METAL_COMPILER
fi
run_id=$(date -u +%Y%m%dT%H%M%SZ)
# Keeping the live app's redirected stdout under Documents makes macOS attribute inherited log
# writes to REmu and present a modal Documents-folder consent sheet. Runtime artifacts are
# diagnostics rather than project sources, so default them to the user's temporary directory.
run_root=${REMU_RUN_ROOT:-"${TMPDIR:-/tmp}/remu-runs"}
run_dir="$run_root/$run_id-$game_name"
frame_dir="$run_dir/frames"
raw_binary="$repo_dir/target/x86_64-apple-darwin/release/remu"
app_bundle=${REMU_APP_BUNDLE:-"$repo_dir/target/x86_64-apple-darwin/release/REMu.app"}
launch_working_directory=${REMU_LAUNCH_WORKING_DIRECTORY:-"$repo_dir"}
mkdir -p "$frame_dir"

if [ ! -d "$launch_working_directory" ]; then
    echo "error: launch working directory not found: $launch_working_directory" >&2
    exit 66
fi

if [ "${REMU_SKIP_BUILD:-0}" != 1 ]; then
    echo "[REMu] building the Rosetta release runner"
    cargo build --manifest-path "$repo_dir/Cargo.toml" \
        --release --target x86_64-apple-darwin -p remu-cli
fi

if [ ! -x "$raw_binary" ]; then
    echo "error: Rosetta runner not found: $raw_binary" >&2
    exit 69
fi

if [ "${REMU_HEADLESS_METAL:-0}" = 1 ]; then
    binary=$raw_binary
else
    "$script_dir/build-macos-app.sh" "$raw_binary" "$app_bundle" >/dev/null
    binary="$app_bundle/Contents/MacOS/remu"
fi

# Interactive runs never inject controller input unless the caller explicitly supplies a schedule.
auto_cross=${REMU_AUTO_CROSS:-}
# Capturing a 4K PPM writes roughly 32 MiB per image and fingerprints the full frame on the
# presenter thread. Keep it explicit so ordinary game runs measure the emulator rather than the
# diagnostic I/O path.
capture_frames=${REMU_CAPTURE_FRAMES:-0}

# The emulator owns its live title-bar telemetry. Keep Apple's Metal HUD disabled by default so
# it cannot obscure guest output or report presentation cadence as if it were guest frame rate.
if [ "${REMU_METAL_HUD:-0}" = 1 ]; then
    MTL_HUD_ENABLED=${MTL_HUD_ENABLED:-1}
    export MTL_HUD_ENABLED
else
    unset MTL_HUD_ENABLED MTL_HUD_LOG_ENABLED
fi
if [ "${REMU_METAL_HUD_LOG:-0}" = 1 ]; then
    MTL_HUD_ENABLED=${MTL_HUD_ENABLED:-1}
    MTL_HUD_LOG_ENABLED=${MTL_HUD_LOG_ENABLED:-1}
    export MTL_HUD_ENABLED MTL_HUD_LOG_ENABLED
fi
revision=$(git -C "$repo_dir" rev-parse --short HEAD 2>/dev/null || echo unknown)
started_at=$(date +%s)
child_pid=

# Invoked indirectly by the traps below.
# shellcheck disable=SC2329
cleanup() {
    if [ -n "$child_pid" ] && kill -0 "$child_pid" 2>/dev/null; then
        kill "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT HUP INT TERM

echo "[REMu] launching $game_name; artifacts: $run_dir"
if [ "${REMU_HEADLESS_METAL:-0}" != 1 ]; then
    echo "[REMu] launching the interactive REmu.app bundle"
fi
if [ -n "${REMU_METAL_COMPILER:-}" ]; then
    echo "[REMu] using offline Metal compiler: $REMU_METAL_COMPILER"
fi
if [ "${MTL_HUD_ENABLED:-0}" = 1 ]; then
    echo "[REMu] Metal Performance HUD enabled"
fi
if [ "${MTL_HUD_LOG_ENABLED:-0}" = 1 ]; then
    echo "[REMu] Metal Performance HUD logging enabled"
fi
if [ "${REMU_HEADLESS_METAL:-0}" = 1 ]; then
    echo "[REMu] headless Metal presenter enabled"
fi
set +e
if [ "$capture_frames" = 1 ]; then
    (
        cd "$launch_working_directory"
        env REMU_AUTO_CROSS="$auto_cross" REMU_CAPTURE_DIR="$frame_dir" \
            arch -x86_64 "$binary" run --timeout-seconds "$timeout_seconds" "$game_path"
    ) > "$run_dir/guest.log" 2>&1 &
else
    (
        cd "$launch_working_directory"
        env -u REMU_CAPTURE_DIR REMU_AUTO_CROSS="$auto_cross" \
            arch -x86_64 "$binary" run --timeout-seconds "$timeout_seconds" "$game_path"
    ) > "$run_dir/guest.log" 2>&1 &
fi
child_pid=$!
wait "$child_pid"
status=$?
child_pid=
set -e

case "$status" in
    0)
        outcome=guest-exit
        runner_status=0
        ;;
    142)
        # The runtime uses SIGALRM to end intentionally bounded sessions on macOS.
        outcome=timeout
        runner_status=0
        ;;
    *)
        outcome=guest-failure
        runner_status=$status
        ;;
esac

finished_at=$(date +%s)
elapsed_seconds=$((finished_at - started_at))
frame_count=$(find "$frame_dir" -type f -name '*.ppm' | wc -l | tr -d ' ')
unimplemented_count=$(grep -c '\[REMu\]\[IMPORT\] unimplemented' "$run_dir/guest.log" || true)
{
    echo "revision=$revision"
    echo "game=$game_path"
    echo "timeout_seconds=$timeout_seconds"
    echo "elapsed_seconds=$elapsed_seconds"
    echo "outcome=$outcome"
    echo "exit_status=$status"
    echo "captured_frames=$frame_count"
    echo "unimplemented_nids=$unimplemented_count"
} > "$run_dir/summary.txt"

tail -n 30 "$run_dir/guest.log" || true
cat "$run_dir/summary.txt"
echo "[REMu] complete; full artifacts: $run_dir"
exit "$runner_status"
