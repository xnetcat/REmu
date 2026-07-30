#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
set -u

if [ "$#" -lt 4 ] || [ "$3" != "--" ]; then
    echo "usage: $0 HARD-TIMEOUT-SECONDS TERMINATION-GRACE-SECONDS -- COMMAND [ARG ...]" >&2
    exit 64
fi

hard_timeout_seconds=$1
termination_grace_seconds=$2
shift 3

for value in "$hard_timeout_seconds" "$termination_grace_seconds"; do
    case "$value" in
        '' | *[!0-9]*)
            echo "error: watchdog durations must be non-negative integer seconds" >&2
            exit 64
            ;;
    esac
done

child_pid=
timer_pid=
timed_out=0

# These helpers form the trap call graph; ShellCheck does not treat a function named in `trap` as
# a root and otherwise reports every helper reachable only from that root as unused.
# shellcheck disable=SC2329
process_is_alive() {
    kill -0 "$1" 2>/dev/null
}

# `kill -0` continues to succeed for an unreaped zombie on some shells. Treat a zombie as exited
# so the caller can collect it with `wait` instead of escalating a signal that can no longer act.
# shellcheck disable=SC2329
process_has_exited() {
    pid=$1
    if ! process_is_alive "$pid"; then
        return 0
    fi
    state=$(ps -p "$pid" -o state= 2>/dev/null | tr -d '[:space:]')
    case "$state" in
        '' | Z*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# shellcheck disable=SC2329
wait_for_child_exit() {
    pid=$1
    checks=$((termination_grace_seconds * 10))
    while ! process_has_exited "$pid"; do
        if [ "$checks" -le 0 ]; then
            return 1
        fi
        sleep 0.1
        checks=$((checks - 1))
    done
    return 0
}

# shellcheck disable=SC2329
cancel_timer() {
    if [ -z "$timer_pid" ]; then
        return
    fi
    kill "$timer_pid" 2>/dev/null || true
    wait "$timer_pid" 2>/dev/null || true
    timer_pid=
}

# shellcheck disable=SC2329
terminate_child() {
    reason=$1
    pid=$child_pid
    if [ -z "$pid" ]; then
        return 0
    fi
    if process_has_exited "$pid"; then
        wait "$pid" 2>/dev/null || true
        child_pid=
        return 0
    fi

    echo "[REMu] watchdog stopping pid=$pid after $reason" >&2
    kill -TERM "$pid" 2>/dev/null || true
    if wait_for_child_exit "$pid"; then
        wait "$pid" 2>/dev/null || true
        child_pid=
        return 0
    fi

    echo "[REMu] watchdog escalating pid=$pid to SIGKILL" >&2
    kill -KILL "$pid" 2>/dev/null || true
    if wait_for_child_exit "$pid"; then
        wait "$pid" 2>/dev/null || true
        child_pid=
        return 0
    fi

    # A Darwin process in uninterruptible U state cannot be reclaimed from user space, even by
    # SIGKILL. Do not make the wrapper wait forever too: leave the exact PID in the diagnostic and
    # return failure so launch orchestration can refuse to stack another game on the wedged driver.
    state=$(ps -p "$pid" -o state= 2>/dev/null | tr -d '[:space:]')
    echo "[REMu] watchdog could not reclaim pid=$pid state=${state:-unknown}; process is still present after SIGKILL" >&2
    child_pid=
    return 1
}

# shellcheck disable=SC2329
handle_timeout() {
    if [ -z "$child_pid" ] || process_has_exited "$child_pid"; then
        return
    fi
    timed_out=1
    echo "[REMu] external launch watchdog reached ${hard_timeout_seconds}s" >&2
    terminate_child "external timeout" || true
}

# shellcheck disable=SC2329
handle_signal() {
    status=$1
    cancel_timer
    terminate_child "watchdog wrapper signal" || true
    trap - EXIT HUP INT TERM USR1
    exit "$status"
}

# shellcheck disable=SC2329
handle_exit() {
    cancel_timer
    terminate_child "watchdog wrapper exit" || true
}

trap handle_exit EXIT
trap 'handle_signal 129' HUP
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM
trap handle_timeout USR1

"$@" &
child_pid=$!

if [ "$hard_timeout_seconds" -gt 0 ]; then
    watchdog_pid=$$
    (
        # Parent-side cancellation is the normal way this timer ends. Exit without signaling the
        # wrapper when that happens, so a completed child cannot receive a stale timeout.
        timer_sleep_pid=
        trap 'kill "$timer_sleep_pid" 2>/dev/null || true; exit 0' HUP INT TERM
        sleep "$hard_timeout_seconds" &
        timer_sleep_pid=$!
        wait "$timer_sleep_pid" 2>/dev/null || exit 0
        timer_sleep_pid=
        kill -USR1 "$watchdog_pid" 2>/dev/null || true
    ) &
    timer_pid=$!
fi

wait "$child_pid"
status=$?
if [ "$timed_out" -ne 0 ]; then
    status=124
fi
child_pid=
cancel_timer
trap - EXIT HUP INT TERM USR1
exit "$status"
