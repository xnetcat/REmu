#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "usage: $0 /path/to/eboot.bin [import-limit] [timeout-seconds]" >&2
    exit 64
fi

if [ "$(uname -s)" != "Darwin" ]; then
    echo "error: the native Rosetta probe currently requires macOS" >&2
    exit 69
fi

game_path=$1
import_limit=${2:-2000000}
timeout_seconds=${3:-120}
if [ ! -f "$game_path" ]; then
    echo "error: game executable not found: $game_path" >&2
    exit 66
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(dirname -- "$script_dir")
run_id=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$repo_dir/.remu/runs/$run_id"
binary="$repo_dir/target/x86_64-apple-darwin/release/remu"
mkdir -p "$run_dir"

echo "[REMu] building the Rosetta probe"
cargo build --manifest-path "$repo_dir/Cargo.toml" \
    --release --target x86_64-apple-darwin -p remu-cli

echo "[REMu] launching under LLDB; artifacts: $run_dir"
set +e
env TERM=dumb arch -x86_64 /usr/bin/lldb --no-lldbinit -b \
    -o 'target stop-hook add -I false -o "thread backtrace all -c 24" -o "register read"' \
    -o "process launch -o '$run_dir/guest.stdout' -e '$run_dir/guest.stderr'" \
    -- "$binary" probe "$game_path" \
    --import-limit "$import_limit" --timeout-seconds "$timeout_seconds" \
    > "$run_dir/debugger.log" 2>&1
debugger_status=$?
set -e

revision=$(git -C "$repo_dir" rev-parse --short HEAD 2>/dev/null || echo unknown)
{
    echo "revision=$revision"
    echo "game=$game_path"
    echo "import_limit=$import_limit"
    echo "timeout_seconds=$timeout_seconds"
    echo "debugger_status=$debugger_status"
} > "$run_dir/summary.txt"

tail -n 40 "$run_dir/debugger.log" || true
echo "[REMu] complete; full artifacts: $run_dir"
exit "$debugger_status"
