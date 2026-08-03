#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Thin wrapper so `scripts/remu-replay.sh <dir>` matches the plan's `remu replay` spelling.
set -eu
script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
exec python3 "$script_dir/remu-replay.py" "$@"
