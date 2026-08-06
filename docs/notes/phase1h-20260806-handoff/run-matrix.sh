#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
set -eu
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:${PATH:-}"
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
OUT="$ROOT/docs/notes/phase1h-20260806-handoff/runs"
cd "$ROOT"

wait_loadavg() {
  i=0
  while [ "$i" -lt 40 ]; do
    avg="$(sysctl -n vm.loadavg)"
    one="$(printf '%s' "$avg" | tr -d '{}' | awk '{print $1}')"
    printf 'loadavg=%s one=%s\n' "$avg" "$one"
    python3 -c "import sys; sys.exit(0 if float('$one') < 5.0 else 1)" && return 0
    sleep 5
    i=$((i + 1))
  done
  return 0
}

run_one() {
  name="$1"
  path="$2"
  secs="${3:-60}"
  wait_loadavg
  printf '===== START %s =====\n' "$name"
  mkdir -p "$OUT/$name"
  sysctl -n vm.loadavg >"$OUT/$name/loadavg-pre.txt"
  ./scripts/run-game.sh "$path" "$secs" 2>&1 | tee "$OUT/$name.log"
  art="$(rg -o 'full artifacts: .*' "$OUT/$name.log" | tail -1 | sed 's/full artifacts: //')"
  printf 'ART=%s\n' "$art"
  if [ -n "$art" ] && [ -d "$art" ]; then
    [ -f "$art/summary.txt" ] && cp "$art/summary.txt" "$OUT/$name/summary.txt"
    [ -f "$art/run.json" ] && cp "$art/run.json" "$OUT/$name/run.json"
    rm -rf "$OUT/${name}-artifacts"
    cp -R "$art" "$OUT/${name}-artifacts"
    if [ -f "$OUT/$name/summary.txt" ]; then
      head -10 "$OUT/$name/summary.txt"
    fi
  fi
}

run_one demon-souls '/Volumes/Untitled/sharpemu/games/Demon Souls/PPSA01342-app0/eboot.bin' 60
run_one dreaming-sarah '/Volumes/Untitled/sharpemu/games/Dreaming Sarah/PPSA02929-app0/eboot.bin' 60
run_one void-terrarium '/Volumes/Untitled/sharpemu/games/void tRrLM Void Terrarium/eboot.bin' 60
run_one oregon-trail '/Volumes/Untitled/sharpemu/games/The Oregon Trail/PPSA19244-app0/decrypted/eboot.bin' 60
printf '===== pre-poppy load wait =====\n'
wait_loadavg
run_one poppy '/Volumes/Untitled/sharpemu/games/Poppy Playtime Chapter 1/eboot.bin' 60
printf '===== MATRIX DONE =====\n'
