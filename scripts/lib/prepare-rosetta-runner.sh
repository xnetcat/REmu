#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Prepare the Rosetta remu binary (and optional app bundle) under the H2 build lock.
# Invoked by run-game.sh / probe scripts via with_build_lock.py; prints the launch path.
set -eu

usage() {
    echo "usage: $0 --repo-dir DIR --raw-binary PATH --app-bundle PATH [--skip-build] [--headless]" >&2
    exit 64
}

repo_dir=
raw_binary=
app_bundle=
skip_build=0
headless=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --repo-dir)
            repo_dir=${2:-}
            shift 2
            ;;
        --raw-binary)
            raw_binary=${2:-}
            shift 2
            ;;
        --app-bundle)
            app_bundle=${2:-}
            shift 2
            ;;
        --skip-build)
            skip_build=1
            shift
            ;;
        --headless)
            headless=1
            shift
            ;;
        *)
            usage
            ;;
    esac
done

if [ -z "$repo_dir" ] || [ -z "$raw_binary" ] || [ -z "$app_bundle" ]; then
    usage
fi

if [ "$skip_build" -eq 0 ]; then
    echo "[REMu] building the Rosetta release runner" >&2
    cargo build --manifest-path "$repo_dir/Cargo.toml" \
        --release --target x86_64-apple-darwin -p remu-cli
fi

if [ ! -x "$raw_binary" ]; then
    echo "error: Rosetta runner not found: $raw_binary" >&2
    exit 69
fi

if [ "$headless" -eq 1 ]; then
    printf '%s\n' "$raw_binary"
else
    "$repo_dir/scripts/build-macos-app.sh" "$raw_binary" "$app_bundle" >/dev/null
    printf '%s\n' "$app_bundle/Contents/MacOS/remu"
fi
