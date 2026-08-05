#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Phase 0 item 7: one REMU_TRACE_SHADER_OPCODES=1 run per matrix title.
set -eu
cd "$(dirname "$0")/../../.."
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:$PATH"
export REMU_SKIP_BUILD=1
export REMU_TRACE_SHADER_OPCODES=1
export REMU_HEADLESS_METAL=1
export REMU_AUTO_DISMISS_DIALOGS=1
export REMU_RUN_ROOT="${TMPDIR:-/tmp}/remu-runs-phase0"
HANDOFF=docs/notes/phase0-20260805-handoff
mkdir -p "$HANDOFF/runs"

wait_for_load() {
    # Opcode traces are not FPS claims, but standing rules still want a cool machine.
    i=0
    while [ "$i" -lt 30 ]; do
        load=$(sysctl -n vm.loadavg)
        # First of three loadavg numbers (1-minute).
        one=$(printf '%s' "$load" | awk '{ gsub(/[{}]/,""); print $1 }')
        case "$one" in
            ''|*[!0-9.]*) echo "loadavg unreadable: $load"; return 0 ;;
        esac
        # Do not let `set -e` abort the matrix when the comparison is false.
        ready=0
        awk "BEGIN { exit !($one < 5) }" || ready=$?
        if [ "$ready" -eq 0 ]; then
            echo "loadavg ready: $load"
            return 0
        fi
        echo "waiting for loadavg <5 (now $load)"
        sleep 10
        i=$((i + 1))
    done
    echo "LOUD NOTE: proceeding with loadavg=$(sysctl -n vm.loadavg) after wait budget"
}

run_title() {
    name=$1
    path=$2
    timeout=$3
    wait_for_load
    echo "===== $name path=$path timeout=${timeout}s loadavg=$(sysctl -n vm.loadavg) ====="
    ./scripts/run-game.sh "$path" "$timeout" 2>&1 | tee "$HANDOFF/${name}-run.log"
    art=$(grep -E 'artifacts: ' "$HANDOFF/${name}-run.log" | tail -1 | sed 's/.*artifacts: //')
    if [ -n "$art" ] && [ -d "$art" ]; then
        ln -sfn "$art" "$HANDOFF/runs/$name"
        {
            grep -E '\[REMu\]\[SHADER-OPCODES\]' "$art/guest.log" || true
            grep -E 'Gen5 .* failed:|combined_lowering_error|unsupported opcode|refusing' "$art/guest.log" || true
        } > "$HANDOFF/${name}-shader-extract.txt"
        wc -l "$HANDOFF/${name}-shader-extract.txt"
        echo "artifact=$art"
        grep -E 'SHADER-OPCODES shaders=|PERF-PIPELINE|EVIDENCE|dirty-tree|unhandled SIG' "$art/guest.log" | head -20 || true
        python3 -c "import json;print(json.dumps(json.load(open('$art/run.json')),indent=2))" 2>/dev/null | head -30 || true
    else
        echo "LOUD REFUSAL: no artifact directory for $name"
    fi
}

run_title silent-hill '/Volumes/Untitled/sharpemu/games/SILENT HILL The Short Message/PPSA10112-app/eboot.bin' 60
run_title demons-souls '/Volumes/Untitled/sharpemu/games/Demon Souls/PPSA01342-app0/eboot.bin' 45
run_title dreaming-sarah '/Volumes/Untitled/sharpemu/games/Dreaming Sarah/PPSA02929-app0/eboot.bin' 45
run_title void-terrarium '/Volumes/Untitled/sharpemu/games/void tRrLM Void Terrarium/eboot.bin' 45
# Root eboot collides with decrypted/sce_module/libc.prx — use decrypted eboot only.
run_title oregon-trail '/Volumes/Untitled/sharpemu/games/The Oregon Trail/PPSA19244-app0/decrypted/eboot.bin' 45
run_title poppy '/Volumes/Untitled/sharpemu/games/Poppy Playtime Chapter 1/eboot.bin' 45
echo DONE
sysctl -n vm.loadavg
