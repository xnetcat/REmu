#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 {DREAMING|SUPERLIMINAL|/path/to/eboot.bin} [timeout-seconds]" >&2
    exit 64
fi

if [ "$(uname -s)" != "Darwin" ]; then
    echo "error: the Rosetta game runner currently requires macOS" >&2
    exit 69
fi

game_name=$1
case "$game_name" in
    DREAMING|dreaming|dreaming-sarah)
        game_name=dreaming-sarah
        game_path=${REMU_DREAMING_SARAH_EBOOT:-/Volumes/Untitled/sharpemu/games/test-game/PPSA02929-app0/eboot.bin}
        ;;
    SUPERLIMINAL|superliminal)
        game_name=superliminal
        game_path=${REMU_SUPERLIMINAL_EBOOT:-/Volumes/Untitled/sharpemu/games/PPSA06084-app/PPSA06084-app0/decrypted/eboot.bin}
        ;;
    *)
        game_path=$game_name
        game_name=$(basename -- "$(dirname -- "$game_path")")
        ;;
esac
timeout_seconds=${2:-120}
if [ ! -f "$game_path" ]; then
    echo "error: game executable not found: $game_path" >&2
    exit 66
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(dirname -- "$script_dir")
bundled_metal_compiler="$repo_dir/.remu/toolchains/Metal.xctoolchain/usr/bin/metal"
if [ -z "${REMU_METAL_COMPILER:-}" ] && [ -x "$bundled_metal_compiler" ]; then
    REMU_METAL_COMPILER=$bundled_metal_compiler
    export REMU_METAL_COMPILER
fi
run_id=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$repo_dir/.remu/runs/$run_id-$game_name"
frame_dir="$run_dir/frames"
binary="$repo_dir/target/x86_64-apple-darwin/release/remu"
mkdir -p "$frame_dir"

if [ "${REMU_SKIP_BUILD:-0}" != 1 ]; then
    echo "[REMu] building the Rosetta release runner"
    cargo build --manifest-path "$repo_dir/Cargo.toml" \
        --release --target x86_64-apple-darwin -p remu-cli
fi

if [ ! -x "$binary" ]; then
    echo "error: Rosetta runner not found: $binary" >&2
    exit 69
fi

auto_cross=${REMU_AUTO_CROSS:-8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118}
capture_frames=${REMU_CAPTURE_FRAMES:-1}
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
if [ -n "${REMU_METAL_COMPILER:-}" ]; then
    echo "[REMu] using offline Metal compiler: $REMU_METAL_COMPILER"
fi
set +e
if [ "$capture_frames" = 1 ]; then
    env REMU_AUTO_CROSS="$auto_cross" REMU_CAPTURE_DIR="$frame_dir" \
        arch -x86_64 "$binary" run --timeout-seconds "$timeout_seconds" "$game_path" \
        > "$run_dir/guest.log" 2>&1 &
else
    env -u REMU_CAPTURE_DIR REMU_AUTO_CROSS="$auto_cross" \
        arch -x86_64 "$binary" run --timeout-seconds "$timeout_seconds" "$game_path" \
        > "$run_dir/guest.log" 2>&1 &
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
