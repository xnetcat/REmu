# SPDX-License-Identifier: GPL-2.0-or-later
# shellcheck shell=sh
#
# Plan H2: build-lock helpers for parallel run-game / probe sessions.
# Source after script_dir and repo_dir are set:
#   . "$script_dir/lib/remu-build-lock.sh"
#
# Procedure for concurrent sessions sharing a checkout:
#   1. One session builds under the exclusive lock (default run-game.sh path).
#   2. Sibling sessions set REMU_SKIP_BUILD=1 after verifying the binary contains
#      their change (strings | grep), optionally pin REMU_APP_BUNDLE to a private
#      path so app-bundle install does not rewrite the shared default bundle.
# Shared-lock launches still wait for any in-progress exclusive build to finish.

remu_build_lock_py() {
    printf '%s\n' "$script_dir/lib/with_build_lock.py"
}

# Run a command under an exclusive flock (cargo build / shared bundle install).
remu_with_exclusive_build_lock() {
    python3 "$(remu_build_lock_py)" --exclusive "$repo_dir" -- "$@"
}

# Run a command under a shared flock (SKIP_BUILD launches wait for writers).
remu_with_shared_build_lock() {
    python3 "$(remu_build_lock_py)" --shared "$repo_dir" -- "$@"
}

# True when the app bundle path is the shared default under target/.
remu_uses_shared_default_app_bundle() {
    default_bundle="$repo_dir/target/x86_64-apple-darwin/release/REMu.app"
    [ "${1:-}" = "$default_bundle" ]
}
