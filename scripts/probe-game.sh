#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "usage: $0 {SHORTCUT|/path/to/eboot.bin} [import-limit] [timeout-seconds]" >&2
    echo "  A non-path SHORTCUT resolves through the REMU_EBOOT_<SHORTCUT> environment" >&2
    echo "  variable, e.g. REMU_EBOOT_MYGAME=/path/to/eboot.bin $0 MYGAME" >&2
    exit 64
fi

if [ "$(uname -s)" != "Darwin" ]; then
    echo "error: the native Rosetta probe currently requires macOS" >&2
    exit 69
fi

game_arg=$1
if [ -f "$game_arg" ]; then
    game_path=$game_arg
else
    # A non-path argument names a launch shortcut configured entirely by the caller's
    # environment, keeping game locations out of the repository.
    shortcut=$(printf '%s' "$game_arg" | tr '[:lower:]' '[:upper:]' | tr '-' '_' | tr -cd 'A-Z0-9_')
    game_path=$(printenv "REMU_EBOOT_$shortcut" || true)
    if [ -z "$game_path" ]; then
        echo "error: no file at '$game_arg' and REMU_EBOOT_$shortcut is not set" >&2
        exit 64
    fi
fi
import_limit=${2:-2000000}
timeout_seconds=${3:-120}
if [ ! -f "$game_path" ]; then
    echo "error: game executable not found: $game_path" >&2
    exit 66
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(dirname -- "$script_dir")
run_id=$(date -u +%Y%m%dT%H%M%SZ)
# Runtime artifacts are diagnostics rather than project sources; keep them in the user's
# temporary directory by default, matching run-game.sh.
run_root=${REMU_RUN_ROOT:-"${TMPDIR:-/tmp}/remu-runs"}
run_dir="$run_root/$run_id-probe"
binary="$repo_dir/target/x86_64-apple-darwin/release/remu"
mkdir -p "$run_dir"

echo "[REMu] building the Rosetta probe"
cargo build --manifest-path "$repo_dir/Cargo.toml" \
    --release --target x86_64-apple-darwin -p remu-cli

echo "[REMu] launching under LLDB; artifacts: $run_dir"
set +e
env TERM=dumb arch -x86_64 /usr/bin/lldb --no-lldbinit -b \
    -o 'process handle SIGALRM -n false -p true -s false' \
    -o 'target stop-hook add -I false -o "thread backtrace all -c 24" -o "register read"' \
    -o "process launch -o '$run_dir/guest.stdout' -e '$run_dir/guest.stderr'" \
    -o 'quit' \
    -- "$binary" probe "$game_path" \
    --import-limit "$import_limit" --timeout-seconds "$timeout_seconds" \
    > "$run_dir/debugger.log" 2>&1
debugger_status=$?
set -e

revision=$(git -C "$repo_dir" rev-parse --short HEAD 2>/dev/null || echo unknown)
# A commit name for a tree with uncommitted changes describes a build that was never committed.
if [ "$revision" != unknown ] && ! git -C "$repo_dir" diff-index --quiet HEAD -- 2>/dev/null; then
    revision="$revision-dirty"
fi
process_result=$(grep -E 'Process [0-9]+ (exited|stopped)' "$run_dir/debugger.log" | tail -n 1 || true)
if printf '%s\n' "$process_result" | grep -q 'SIGALRM'; then
    outcome=timeout
elif printf '%s\n' "$process_result" | grep -q 'exited with status = 142'; then
    # The probe's own timeout terminates the guest with _exit(142); it no longer raises SIGALRM.
    outcome=timeout
elif printf '%s\n' "$process_result" | grep -q 'exited with status'; then
    outcome=guest-exit
elif [ -n "$process_result" ]; then
    outcome=debugger-stop
else
    outcome=launcher-failure
fi
{
    echo "revision=$revision"
    echo "game=$game_path"
    echo "import_limit=$import_limit"
    echo "timeout_seconds=$timeout_seconds"
    echo "debugger_status=$debugger_status"
    echo "outcome=$outcome"
    echo "process_result=$process_result"
} > "$run_dir/summary.txt"

tail -n 40 "$run_dir/debugger.log" || true
cat "$run_dir/summary.txt"
echo "[REMu] complete; full artifacts: $run_dir"
exit "$debugger_status"
