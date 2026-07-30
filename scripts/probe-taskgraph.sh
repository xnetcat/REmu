#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
    echo "usage: $0 {SHORTCUT|/path/to/eboot.bin} [timeout-seconds] [cycles] [hit-limit]" >&2
    echo "  Traces one RenderThread fence task at a time with at most four LLDB" >&2
    echo "  hardware code breakpoints. Guest executable bytes are never modified." >&2
    echo "  Defaults: timeout=120, cycles=16, hit-limit=4096." >&2
    echo "  Set REMU_TASKGRAPH_TRACE_PREPARE_ONLY=1 for one persistent breakpoint." >&2
    echo "  Set REMU_TASKGRAPH_TRACE_TARGET_FOLLOW=1 for caller-filtered stages." >&2
    echo "  Add REMU_TASKGRAPH_TRACE_QUEUE_HANDOFF=1 for named-queue handoff." >&2
    exit 64
fi

if [ "$(uname -s)" != "Darwin" ]; then
    echo "error: the Rosetta LLDB diagnostic requires macOS" >&2
    exit 69
fi

game_arg=$1
if [ -f "$game_arg" ]; then
    game_path=$game_arg
    game_name=$(basename -- "$(dirname -- "$game_path")")
else
    shortcut=$(printf '%s' "$game_arg" | tr '[:lower:]' '[:upper:]' | tr '-' '_' | tr -cd 'A-Z0-9_')
    game_path=$(printenv "REMU_EBOOT_$shortcut" || true)
    if [ -z "$game_path" ]; then
        echo "error: no file at '$game_arg' and REMU_EBOOT_$shortcut is not set" >&2
        exit 64
    fi
    game_name=$(printf '%s' "$shortcut" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
fi
if [ ! -f "$game_path" ]; then
    echo "error: game executable not found: $game_path" >&2
    exit 66
fi

timeout_seconds=${2:-120}
cycle_limit=${3:-16}
hit_limit=${4:-4096}
for value in "$timeout_seconds" "$cycle_limit" "$hit_limit"; do
    case "$value" in
        '' | *[!0-9]*)
            echo "error: timeout, cycle limit, and hit limit must be positive integers" >&2
            exit 64
            ;;
    esac
done
if [ "$timeout_seconds" -lt 1 ] || [ "$cycle_limit" -lt 1 ] || \
    [ "$cycle_limit" -gt 64 ] || [ "$hit_limit" -lt 1 ] || \
    [ "$hit_limit" -gt 100000 ]; then
    echo "error: require timeout>=1, cycles=1..64, hit-limit=1..100000" >&2
    exit 64
fi

# Do not add debugger pressure to an already active emulator. This diagnostic
# never kills an existing game.
if [ "${REMU_ALLOW_CONCURRENT_GAMES:-0}" != 1 ] && pgrep -x remu >/dev/null 2>&1; then
    echo "error: refusing to trace while another REmu process exists" >&2
    exit 75
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(dirname -- "$script_dir")
trace_script="$script_dir/lldb_taskgraph_trace.py"
run_id=$(date -u +%Y%m%dT%H%M%SZ)
run_root=${REMU_RUN_ROOT:-"${TMPDIR:-/tmp}/remu-runs"}
run_dir="$run_root/$run_id-$game_name-taskgraph"
raw_binary=${REMU_RUNNER_BINARY:-"$repo_dir/target/x86_64-apple-darwin/release/remu"}
app_bundle=${REMU_APP_BUNDLE:-"$repo_dir/target/x86_64-apple-darwin/release/REMu.app"}
launch_working_directory=${REMU_LAUNCH_WORKING_DIRECTORY:-"$repo_dir"}
watchdog_slack_seconds=${REMU_WATCHDOG_SLACK_SECONDS:-15}
watchdog_grace_seconds=${REMU_WATCHDOG_TERMINATION_GRACE_SECONDS:-2}
for value in "$watchdog_slack_seconds" "$watchdog_grace_seconds"; do
    case "$value" in
        '' | *[!0-9]*)
            echo "error: watchdog slack and grace must be non-negative integers" >&2
            exit 64
            ;;
    esac
done
hard_timeout_seconds=$((timeout_seconds + watchdog_slack_seconds))
mkdir -p "$run_dir"

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

auto_cross=${REMU_AUTO_CROSS:-}
target_follow=${REMU_TASKGRAPH_TRACE_TARGET_FOLLOW:-0}
case "$target_follow" in
    0 | 1) ;;
    *)
        echo "error: REMU_TASKGRAPH_TRACE_TARGET_FOLLOW must be 0 or 1" >&2
        exit 64
        ;;
esac
queue_handoff=${REMU_TASKGRAPH_TRACE_QUEUE_HANDOFF:-0}
case "$queue_handoff" in
    0 | 1) ;;
    *)
        echo "error: REMU_TASKGRAPH_TRACE_QUEUE_HANDOFF must be 0 or 1" >&2
        exit 64
        ;;
esac
if [ "$queue_handoff" = 1 ] && [ "$target_follow" != 1 ]; then
    echo "error: queue handoff requires target follow" >&2
    exit 64
fi
prepare_only=${REMU_TASKGRAPH_TRACE_PREPARE_ONLY:-0}
case "$prepare_only" in
    0 | 1) ;;
    *)
        echo "error: REMU_TASKGRAPH_TRACE_PREPARE_ONLY must be 0 or 1" >&2
        exit 64
        ;;
esac
if [ "$target_follow" = 1 ]; then
    prepare_only=1
fi
target_caller=${REMU_TASKGRAPH_TRACE_TARGET_CALLER:-0x801be0507}
target_parent=${REMU_TASKGRAPH_TRACE_TARGET_PARENT:-0x80101af94}
disable_cpu_write_watch=${REMU_DISABLE_CPU_WRITE_WATCH:-0}
if [ "$prepare_only" = 1 ]; then
    # macOS debugserver intercepts deliberate CPU write-watch faults before
    # REmu's handler.  PREPARE-only traces are long-lived, so use the existing
    # debugger mode that never arms those page protections.  Guest VMA
    # protections remain intact, and the LLDB script leaves every remaining
    # EXC_BAD_ACCESS stopped as a real fault.
    disable_cpu_write_watch=1
fi
case "$disable_cpu_write_watch" in
    0 | 1) ;;
    *)
        echo "error: REMU_DISABLE_CPU_WRITE_WATCH must be 0 or 1" >&2
        exit 64
        ;;
esac
revision=$(git -C "$repo_dir" rev-parse --short HEAD 2>/dev/null || echo unknown)
started_at=$(date +%s)

echo "[REMu] launching bounded TaskGraph LLDB trace; artifacts: $run_dir"
echo "[REMu] timeout=${timeout_seconds}s cycles=$cycle_limit hit-limit=$hit_limit"
echo "[REMu] prepare-only=$prepare_only cpu-write-watch-disabled=$disable_cpu_write_watch"
echo "[REMu] target-follow=$target_follow caller=$target_caller parent=$target_parent"
echo "[REMu] queue-handoff=$queue_handoff"
set +e
(
    cd "$launch_working_directory"
    "$script_dir/run-with-watchdog.sh" \
        "$hard_timeout_seconds" "$watchdog_grace_seconds" -- \
        env -u LaunchInstanceID -u XPC_FLAGS -u XPC_SERVICE_NAME \
            TERM=dumb \
            REMU_AUTO_CROSS="$auto_cross" \
            REMU_TASKGRAPH_TRACE_CYCLES="$cycle_limit" \
            REMU_TASKGRAPH_TRACE_HITS="$hit_limit" \
            REMU_TASKGRAPH_TRACE_PREPARE_ONLY="$prepare_only" \
            REMU_TASKGRAPH_TRACE_TARGET_FOLLOW="$target_follow" \
            REMU_TASKGRAPH_TRACE_QUEUE_HANDOFF="$queue_handoff" \
            REMU_TASKGRAPH_TRACE_TARGET_CALLER="$target_caller" \
            REMU_TASKGRAPH_TRACE_TARGET_PARENT="$target_parent" \
            REMU_DISABLE_CPU_WRITE_WATCH="$disable_cpu_write_watch" \
            arch -x86_64 /usr/bin/lldb --no-lldbinit -b \
            -o 'process handle SIGALRM -n false -p true -s false' \
            -o 'process handle SIGSEGV -n false -p true -s false' \
            -o 'process handle SIGBUS -n false -p true -s false' \
            -o "command script import '$trace_script'" \
            -o 'remu-taskgraph-trace-start' \
            -o "process launch -o '$run_dir/guest.stdout' -e '$run_dir/guest.stderr'" \
            -o 'remu-taskgraph-resume-diagnostic-stops' \
            -o 'thread backtrace all -c 24' \
            -o 'register read' \
            -o 'quit' \
            -- "$binary" run --timeout-seconds "$timeout_seconds" "$game_path"
) > "$run_dir/debugger.log" 2>&1
status=$?
set -e

process_result=$(grep -E 'Process [0-9]+ (exited|stopped)' "$run_dir/debugger.log" | tail -n 1 || true)
debugger_fault=$(grep -hE \
    'debugserver requesting translation removal|gTaskSuspended == true|rosetta error: received an un(ex|pex)cted watchpoint exception' \
    "$run_dir/debugger.log" "$run_dir/guest.stderr" | tail -n 1 || true)
trace_lines=$(grep -c '\[REMu\]\[LLDB\]\[TASKGRAPH\]' "$run_dir/debugger.log" || true)
trace_terminal=$(grep '\[REMu\]\[LLDB\]\[TASKGRAPH\].*done reason=' \
    "$run_dir/debugger.log" | tail -n 1 || true)
trace_last=$(grep '\[REMu\]\[LLDB\]\[TASKGRAPH\]' \
    "$run_dir/debugger.log" | tail -n 1 || true)
trace_phase=$(grep -E '\[REMu\]\[LLDB\]\[TASKGRAPH\] (armed phase=|persistent breakpoints=)' \
    "$run_dir/debugger.log" | tail -n 1 || true)
if [ -n "$debugger_fault" ]; then
    outcome=debugger-rosetta-fault
elif [ "$status" -eq 124 ]; then
    outcome=external-watchdog-timeout
elif printf '%s\n' "$process_result" | grep -q 'exited with status = 142'; then
    outcome=diagnostic-timeout
elif printf '%s\n' "$process_result" | grep -q 'exited with status'; then
    outcome=guest-exit
elif [ -n "$process_result" ]; then
    outcome=debugger-stop
else
    outcome=launcher-failure
fi
finished_at=$(date +%s)
{
    echo "revision=$revision"
    echo "game=$game_path"
    echo "timeout_seconds=$timeout_seconds"
    echo "cycle_limit=$cycle_limit"
    echo "hit_limit=$hit_limit"
    echo "prepare_only=$prepare_only"
    echo "target_follow=$target_follow"
    echo "queue_handoff=$queue_handoff"
    echo "target_caller=$target_caller"
    echo "target_parent=$target_parent"
    echo "cpu_write_watch_disabled=$disable_cpu_write_watch"
    echo "elapsed_seconds=$((finished_at - started_at))"
    echo "exit_status=$status"
    echo "outcome=$outcome"
    echo "process_result=$process_result"
    echo "debugger_fault=$debugger_fault"
    echo "trace_lines=$trace_lines"
    echo "trace_terminal=$trace_terminal"
    echo "trace_phase=$trace_phase"
    echo "trace_last=$trace_last"
} > "$run_dir/summary.txt"

tail -n 40 "$run_dir/debugger.log" || true
cat "$run_dir/summary.txt"
echo "[REMu] complete; full artifacts: $run_dir"
exit "$status"
