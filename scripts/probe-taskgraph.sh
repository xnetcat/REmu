#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
set -eu

usage() {
    echo "usage: $0 {SHORTCUT|/path/to/eboot.bin} [timeout-seconds] [cycles] [hit-limit]" >&2
    echo "       $0 --attach PID {SHORTCUT|/path/to/eboot.bin} [timeout-seconds] [cycles] [hit-limit]" >&2
    echo "  Traces one RenderThread fence task at a time with at most four LLDB" >&2
    echo "  hardware code breakpoints. Guest executable bytes are never modified." >&2
    echo "  Defaults: timeout=120, cycles=16, hit-limit=4096." >&2
    echo "  --attach traces one already-running, bounded REmu process without" >&2
    echo "  launching or terminating it. Set REMU_TASKGRAPH_ATTACH_DELAY_SECONDS" >&2
    echo "  to wait before attaching (default 0, maximum 3600)." >&2
    echo "  REMU_TASKGRAPH_ATTACH_TRACE_SECONDS controls clean-detach duration;" >&2
    echo "  longer probes reserve 30 seconds before timeout-seconds." >&2
    echo "  Set REMU_TASKGRAPH_TRACE_PREPARE_ONLY=1 for one persistent breakpoint." >&2
    echo "  Set REMU_TASKGRAPH_TRACE_TARGET_FOLLOW=1 for caller-filtered stages." >&2
    echo "  Add REMU_TASKGRAPH_TRACE_QUEUE_HANDOFF=1 for named-queue handoff." >&2
}

attach_mode=0
attach_pid=
if [ "${1:-}" = "--attach" ]; then
    if [ "$#" -lt 3 ] || [ "$#" -gt 6 ]; then
        usage
        exit 64
    fi
    attach_mode=1
    attach_pid=$2
    case "$attach_pid" in
        '' | *[!0-9]* | 0 | 1)
            echo "error: attach PID must be an integer greater than 1" >&2
            exit 64
            ;;
    esac
    shift 2
elif [ "$#" -lt 1 ] || [ "$#" -gt 4 ]; then
    usage
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

# Launch mode must not add debugger pressure to an already active emulator.
# Attach mode never launches a game and validates its one explicit target below.
if [ "$attach_mode" -eq 0 ] && \
    [ "${REMU_ALLOW_CONCURRENT_GAMES:-0}" != 1 ] && \
    pgrep -x remu >/dev/null 2>&1; then
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
attach_delay_seconds=${REMU_TASKGRAPH_ATTACH_DELAY_SECONDS:-0}
attach_trace_seconds=${REMU_TASKGRAPH_ATTACH_TRACE_SECONDS:-}
for value in "$watchdog_slack_seconds" "$watchdog_grace_seconds"; do
    case "$value" in
        '' | *[!0-9]*)
            echo "error: watchdog slack and grace must be non-negative integers" >&2
            exit 64
            ;;
    esac
done
case "$attach_delay_seconds" in
    '' | *[!0-9]*)
        echo "error: REMU_TASKGRAPH_ATTACH_DELAY_SECONDS must be a non-negative integer" >&2
        exit 64
        ;;
esac
if [ "$attach_delay_seconds" -gt 3600 ]; then
    echo "error: REMU_TASKGRAPH_ATTACH_DELAY_SECONDS must not exceed 3600" >&2
    exit 64
fi
if [ "$attach_mode" -eq 0 ] && [ "$attach_delay_seconds" -ne 0 ]; then
    echo "error: REMU_TASKGRAPH_ATTACH_DELAY_SECONDS requires --attach PID" >&2
    exit 64
fi
if [ "$attach_mode" -eq 1 ]; then
    if [ -z "$attach_trace_seconds" ]; then
        if [ "$timeout_seconds" -gt 30 ]; then
            attach_trace_seconds=$((timeout_seconds - 30))
        else
            attach_trace_seconds=$((timeout_seconds / 2))
            if [ "$attach_trace_seconds" -lt 1 ]; then
                attach_trace_seconds=1
            fi
        fi
    fi
    case "$attach_trace_seconds" in
        '' | *[!0-9]* | 0)
            echo "error: REMU_TASKGRAPH_ATTACH_TRACE_SECONDS must be a positive integer" >&2
            exit 64
            ;;
    esac
    if [ "$attach_trace_seconds" -gt "$timeout_seconds" ]; then
        echo "error: REMU_TASKGRAPH_ATTACH_TRACE_SECONDS must not exceed timeout-seconds" >&2
        exit 64
    fi
elif [ -n "$attach_trace_seconds" ]; then
    echo "error: REMU_TASKGRAPH_ATTACH_TRACE_SECONDS requires --attach PID" >&2
    exit 64
else
    attach_trace_seconds=0
fi
hard_timeout_seconds=$((timeout_seconds + watchdog_slack_seconds))
mkdir -p "$run_dir"

if [ ! -d "$launch_working_directory" ]; then
    echo "error: launch working directory not found: $launch_working_directory" >&2
    exit 66
fi

if [ "$attach_mode" -eq 0 ]; then
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

attach_identity=
validated_attach_identity() {
    pid=$1
    expected_game=$2
    require_cpu_write_watch_disabled=$3
    process_uid=$(/bin/ps -p "$pid" -o uid= 2>/dev/null | tr -d '[:space:]')
    process_state=$(/bin/ps -p "$pid" -o state= 2>/dev/null | tr -d '[:space:]')
    process_start=$(/bin/ps -p "$pid" -o lstart= 2>/dev/null | sed 's/^[[:space:]]*//')
    process_comm=$(/bin/ps -p "$pid" -o comm= 2>/dev/null | sed 's/^[[:space:]]*//')
    process_command=$(/bin/ps -ww -p "$pid" -o command= 2>/dev/null | sed 's/^[[:space:]]*//')
    process_environment=$(/bin/ps eww -p "$pid" -o command= 2>/dev/null | sed 's/^[[:space:]]*//')
    if [ -z "$process_uid" ] || [ -z "$process_state" ] || \
        [ -z "$process_start" ] || [ -z "$process_comm" ] || \
        [ -z "$process_command" ] || [ -z "$process_environment" ]; then
        echo "error: attach PID $pid does not identify a live process" >&2
        return 1
    fi
    if [ "$process_uid" != "$(id -u)" ]; then
        echo "error: attach PID $pid is owned by a different user" >&2
        return 1
    fi
    case "$process_state" in
        Z*)
            echo "error: attach PID $pid is a zombie" >&2
            return 1
            ;;
    esac
    case "$process_comm" in
        remu | */remu) ;;
        *)
            echo "error: attach PID $pid is not the REmu executable: $process_comm" >&2
            return 1
            ;;
    esac
    case "$process_command" in
        *" run --timeout-seconds "*)
            bounded_tail=${process_command#*" run --timeout-seconds "}
            bounded_timeout=${bounded_tail%% *}
            case "$bounded_timeout" in
                '' | *[!0-9]* | 0)
                    echo "error: attach PID $pid has an invalid run timeout" >&2
                    return 1
                    ;;
            esac
            ;;
        *)
            echo "error: attach PID $pid is not a bounded REmu run" >&2
            return 1
            ;;
    esac
    case "$process_command" in
        *"$expected_game"*) ;;
        *)
            echo "error: attach PID $pid is not running the requested game" >&2
            return 1
            ;;
    esac
    target_write_watch_disabled=0
    case "$process_environment" in
        *" REMU_DISABLE_CPU_WRITE_WATCH=1 "* | \
        *" REMU_DISABLE_CPU_WRITE_WATCH=1")
            target_write_watch_disabled=1
            ;;
    esac
    if [ "$require_cpu_write_watch_disabled" = 1 ] && \
        [ "$target_write_watch_disabled" != 1 ]; then
        echo "error: attach PID $pid requires target environment REMU_DISABLE_CPU_WRITE_WATCH=1 for this persistent trace" >&2
        return 1
    fi
    # Return an identity token, not the environment itself. This both avoids
    # leaking target environment values and makes PID-reuse checks deterministic.
    printf '%s\n%s\n%s\n%s\n%s\n' \
        "$process_uid" "$process_start" "$process_comm" "$process_command" \
        "$target_write_watch_disabled"
}

target_cpu_write_watch_disabled=$disable_cpu_write_watch
if [ "$attach_mode" -eq 1 ]; then
    attach_identity=$(
        validated_attach_identity \
            "$attach_pid" "$game_path" "$disable_cpu_write_watch"
    ) || exit 75
    target_cpu_write_watch_disabled=$(
        printf '%s\n' "$attach_identity" | tail -n 1
    )
    if [ "$attach_delay_seconds" -gt 0 ]; then
        echo "[REMu] waiting ${attach_delay_seconds}s before attaching to validated pid=$attach_pid"
        sleep "$attach_delay_seconds"
        revalidated_identity=$(
            validated_attach_identity \
                "$attach_pid" "$game_path" "$disable_cpu_write_watch"
        ) || exit 75
        if [ "$revalidated_identity" != "$attach_identity" ]; then
            echo "error: attach PID $attach_pid changed identity during the delay" >&2
            exit 75
        fi
    fi
    echo "[REMu] attaching bounded TaskGraph LLDB trace to pid=$attach_pid; artifacts: $run_dir"
else
    echo "[REMu] launching bounded TaskGraph LLDB trace; artifacts: $run_dir"
fi
echo "[REMu] timeout=${timeout_seconds}s cycles=$cycle_limit hit-limit=$hit_limit"
echo "[REMu] prepare-only=$prepare_only debugger-cpu-write-watch-disabled=$disable_cpu_write_watch"
echo "[REMu] target-cpu-write-watch-disabled=$target_cpu_write_watch_disabled"
echo "[REMu] target-follow=$target_follow caller=$target_caller parent=$target_parent"
echo "[REMu] queue-handoff=$queue_handoff"
echo "[REMu] attach-mode=$attach_mode attach-delay=${attach_delay_seconds}s attach-trace=${attach_trace_seconds}s"
set +e
if [ "$attach_mode" -eq 1 ]; then
    (
        cd "$launch_working_directory"
        # This watchdog owns only LLDB. The validated REmu target was launched
        # separately with its own timeout and is never signalled by this script.
        "$script_dir/run-with-watchdog.sh" \
            "$hard_timeout_seconds" "$watchdog_grace_seconds" -- \
            env -u LaunchInstanceID -u XPC_FLAGS -u XPC_SERVICE_NAME \
                TERM=dumb \
                REMU_TASKGRAPH_TRACE_ATTACH=1 \
                REMU_TASKGRAPH_TRACE_ATTACH_DURATION_SECONDS="$attach_trace_seconds" \
                REMU_TASKGRAPH_TRACE_CYCLES="$cycle_limit" \
                REMU_TASKGRAPH_TRACE_HITS="$hit_limit" \
                REMU_TASKGRAPH_TRACE_PREPARE_ONLY="$prepare_only" \
                REMU_TASKGRAPH_TRACE_TARGET_FOLLOW="$target_follow" \
                REMU_TASKGRAPH_TRACE_QUEUE_HANDOFF="$queue_handoff" \
                REMU_TASKGRAPH_TRACE_TARGET_CALLER="$target_caller" \
                REMU_TASKGRAPH_TRACE_TARGET_PARENT="$target_parent" \
                REMU_DISABLE_CPU_WRITE_WATCH="$disable_cpu_write_watch" \
                arch -x86_64 /usr/bin/lldb --no-lldbinit -b \
                -o 'settings set interpreter.stop-command-source-on-error false' \
                -o "command script import '$trace_script'" \
                -o "process attach --pid $attach_pid" \
                -o 'process handle SIGALRM -n false -p true -s false' \
                -o 'process handle SIGSEGV -n false -p true -s false' \
                -o 'process handle SIGBUS -n false -p true -s false' \
                -o 'remu-taskgraph-trace-start' \
                -o 'process continue' \
                -o 'remu-taskgraph-resume-diagnostic-stops' \
                -o 'remu-taskgraph-trace-finish' \
                -o 'quit'
    ) > "$run_dir/debugger.log" 2>&1
else
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
fi
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
trace_detached=$(grep '\[REMu\]\[LLDB\]\[TASKGRAPH\] detached without terminating target' \
    "$run_dir/debugger.log" | tail -n 1 || true)
if [ -n "$debugger_fault" ]; then
    outcome=debugger-rosetta-fault
elif [ -n "$trace_terminal" ] && [ -n "$trace_detached" ]; then
    outcome=trace-terminal-detached
elif [ "$attach_mode" -eq 1 ] && [ -n "$trace_detached" ]; then
    outcome=diagnostic-detached
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
    echo "attach_mode=$attach_mode"
    echo "attach_pid=$attach_pid"
    echo "attach_delay_seconds=$attach_delay_seconds"
    echo "attach_trace_seconds=$attach_trace_seconds"
    echo "target_caller=$target_caller"
    echo "target_parent=$target_parent"
    echo "cpu_write_watch_disabled=$target_cpu_write_watch_disabled"
    echo "debugger_cpu_write_watch_disabled=$disable_cpu_write_watch"
    echo "target_cpu_write_watch_disabled=$target_cpu_write_watch_disabled"
    echo "elapsed_seconds=$((finished_at - started_at))"
    echo "exit_status=$status"
    echo "outcome=$outcome"
    echo "process_result=$process_result"
    echo "debugger_fault=$debugger_fault"
    echo "trace_lines=$trace_lines"
    echo "trace_terminal=$trace_terminal"
    echo "trace_phase=$trace_phase"
    echo "trace_last=$trace_last"
    echo "trace_detached=$trace_detached"
} > "$run_dir/summary.txt"

tail -n 40 "$run_dir/debugger.log" || true
cat "$run_dir/summary.txt"
echo "[REMu] complete; full artifacts: $run_dir"
exit "$status"
