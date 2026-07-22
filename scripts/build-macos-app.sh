#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
set -eu

if [ "$#" -gt 2 ]; then
    echo "usage: $0 [x86-64-remu-binary] [output-app]" >&2
    exit 64
fi

if [ "$(uname -s)" != "Darwin" ]; then
    echo "error: the REmu app bundle can only be assembled on macOS" >&2
    exit 69
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(dirname -- "$script_dir")
source_binary=${1:-"$repo_dir/target/x86_64-apple-darwin/release/remu"}
app_bundle=${2:-"$repo_dir/target/x86_64-apple-darwin/release/REmu.app"}
plist_source="$repo_dir/resources/macos/Info.plist"

if [ ! -x "$source_binary" ]; then
    echo "error: Rosetta runner not found: $source_binary" >&2
    exit 66
fi

contents_dir="$app_bundle/Contents"
macos_dir="$contents_dir/MacOS"
resources_dir="$contents_dir/Resources"
mkdir -p "$macos_dir" "$resources_dir"
install -m 0755 "$source_binary" "$macos_dir/remu"
install -m 0644 "$plist_source" "$contents_dir/Info.plist"
/usr/bin/plutil -lint "$contents_dir/Info.plist" >/dev/null

echo "$app_bundle"
