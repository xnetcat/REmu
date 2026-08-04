#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
set -eu

usage() {
    echo "usage: $0 --attach PID {SHORTCUT|/path/to/eboot.bin} [timeout-seconds]" >&2
    echo "  Takes one read-only post-stall TaskGraph snapshot from an existing" >&2
    echo "  bounded same-user REmu run, then detaches without terminating it." >&2
    echo "  The target must have REMU_DISABLE_CPU_WRITE_WATCH=1. Default timeout: 30." >&2
}

if [ "${1:-}" != "--attach" ] || [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    usage
    exit 64
fi

attach_pid=$2
case "$attach_pid" in
    '' | *[!0-9]* | 0 | 1)
        echo "error: attach PID must be an integer greater than 1" >&2
        exit 64
        ;;
esac
shift 2

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

timeout_seconds=${2:-30}
case "$timeout_seconds" in
    '' | *[!0-9]* | 0)
        echo "error: timeout-seconds must be a positive integer" >&2
        exit 64
        ;;
esac
if [ "$timeout_seconds" -gt 120 ]; then
    echo "error: timeout-seconds must not exceed 120" >&2
    exit 64
fi

validated_attach_identity() {
    pid=$1
    expected_game=$2
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
    case "$process_environment" in
        *" REMU_DISABLE_CPU_WRITE_WATCH=1 "* | \
        *" REMU_DISABLE_CPU_WRITE_WATCH=1") ;;
        *)
            echo "error: attach PID $pid requires target environment REMU_DISABLE_CPU_WRITE_WATCH=1" >&2
            return 1
            ;;
    esac
    printf '%s\n%s\n%s\n%s\n' \
        "$process_uid" "$process_start" "$process_comm" "$process_command"
}

attach_identity=$(
    validated_attach_identity "$attach_pid" "$game_path"
) || exit 75

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(dirname -- "$script_dir")
snapshot_script="$script_dir/lldb_taskgraph_stall_snapshot.py"
run_id=$(date -u +%Y%m%dT%H%M%SZ)
run_root=${REMU_RUN_ROOT:-"${TMPDIR:-/tmp}/remu-runs"}
run_dir="$run_root/$run_id-$game_name-taskgraph-stall-$attach_pid"
watchdog_grace_seconds=${REMU_WATCHDOG_TERMINATION_GRACE_SECONDS:-2}
case "$watchdog_grace_seconds" in
    '' | *[!0-9]*)
        echo "error: watchdog grace must be a non-negative integer" >&2
        exit 64
        ;;
esac
mkdir -p "$run_dir"

# Close the PID-reuse window immediately before invoking LLDB.
revalidated_identity=$(
    validated_attach_identity "$attach_pid" "$game_path"
) || exit 75
if [ "$revalidated_identity" != "$attach_identity" ]; then
    echo "error: attach PID $attach_pid changed identity before attach" >&2
    exit 75
fi

revision=$(git -C "$repo_dir" rev-parse --short HEAD 2>/dev/null || echo unknown)
started_at=$(date +%s)
echo "[REMu] read-only TaskGraph stall snapshot pid=$attach_pid"
echo "[REMu] artifacts: $run_dir"

set +e
(
    cd "$repo_dir"
    # This watchdog owns only LLDB. It never signals the separately launched
    # REmu target, whose own --timeout-seconds remains authoritative.
    # Native LLDB can inspect the Rosetta x86_64 target through debugserver.
    # Running LLDB itself as x86_64 is not portable because current arm64
    # Command Line Tools installations may provide only an arm64 libxcrun.
    "$script_dir/run-with-watchdog.sh" \
        "$timeout_seconds" "$watchdog_grace_seconds" -- \
        env -u LaunchInstanceID -u XPC_FLAGS -u XPC_SERVICE_NAME \
            TERM=dumb \
            REMU_DISABLE_CPU_WRITE_WATCH=1 \
            /usr/bin/lldb --no-lldbinit -b \
            -o 'settings set interpreter.stop-command-source-on-error false' \
            -o "command script import '$snapshot_script'" \
            -o "process attach --pid $attach_pid" \
            -o 'remu-taskgraph-stall-snapshot' \
            -o 'remu-taskgraph-stall-detach' \
            -o 'quit'
) > "$run_dir/debugger.log" 2>&1
status=$?
set -e

finished_at=$(date +%s)
snapshot_done=$(grep '\[REMu\]\[LLDB\]\[TASKGRAPH-STALL\] done reason=' \
    "$run_dir/debugger.log" | tail -n 1 || true)
detached=$(grep '\[REMu\]\[LLDB\]\[TASKGRAPH-STALL\] detached without terminating target' \
    "$run_dir/debugger.log" | tail -n 1 || true)
if [ -n "$snapshot_done" ] && [ -n "$detached" ]; then
    outcome=snapshot-detached
elif [ "$status" -eq 124 ]; then
    outcome=debugger-watchdog-timeout
elif [ -n "$detached" ]; then
    outcome=diagnostic-detached
else
    outcome=diagnostic-failure
fi
{
    echo "revision=$revision"
    echo "game=$game_path"
    echo "attach_pid=$attach_pid"
    echo "timeout_seconds=$timeout_seconds"
    echo "elapsed_seconds=$((finished_at - started_at))"
    echo "exit_status=$status"
    echo "outcome=$outcome"
    echo "snapshot_done=$snapshot_done"
    echo "detached=$detached"
} > "$run_dir/summary.txt"

tail -n 120 "$run_dir/debugger.log"
echo "[REMu] outcome=$outcome"
echo "[REMu] summary: $run_dir/summary.txt"

if [ "$outcome" = snapshot-detached ] || [ "$outcome" = diagnostic-detached ]; then
    exit 0
fi
if [ "$status" -ne 0 ]; then
    exit "$status"
fi
exit 1
