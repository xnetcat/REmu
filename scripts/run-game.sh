#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 {SHORTCUT|/path/to/eboot.bin} [timeout-seconds]" >&2
    echo "  A non-path SHORTCUT resolves through the REMU_EBOOT_<SHORTCUT> environment" >&2
    echo "  variable, e.g. REMU_EBOOT_MYGAME=/path/to/eboot.bin $0 MYGAME" >&2
    echo "  Runs stop after 60 seconds by default; pass 0 or set REMU_GAME_TIMEOUT_SECONDS=0" >&2
    echo "  to disable the watchdog." >&2
    exit 64
fi

if [ "$(uname -s)" != "Darwin" ]; then
    echo "error: the Rosetta game runner currently requires macOS" >&2
    exit 69
fi

game_arg=$1
if [ -f "$game_arg" ]; then
    game_path=$game_arg
    game_name=$(basename -- "$(dirname -- "$game_path")")
else
    # A non-path argument names a launch shortcut configured entirely by the caller's
    # environment, keeping game locations out of the repository.
    shortcut=$(printf '%s' "$game_arg" | tr '[:lower:]' '[:upper:]' | tr '-' '_' | tr -cd 'A-Z0-9_')
    game_path=$(printenv "REMU_EBOOT_$shortcut" || true)
    if [ -z "$game_path" ]; then
        echo "error: no file at '$game_arg' and REMU_EBOOT_$shortcut is not set" >&2
        exit 64
    fi
    game_name=$(printf '%s' "$shortcut" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
fi
# Protect the shared Apple GPU and unified memory from an unattended wedged game. An explicit
# second argument wins, then the environment override, and zero deliberately disables the
# watchdog for an unbounded interactive session.
timeout_seconds=${2:-${REMU_GAME_TIMEOUT_SECONDS:-60}}
watchdog_slack_seconds=${REMU_WATCHDOG_SLACK_SECONDS:-5}
watchdog_termination_grace_seconds=${REMU_WATCHDOG_TERMINATION_GRACE_SECONDS:-2}
for value in "$timeout_seconds" "$watchdog_slack_seconds" "$watchdog_termination_grace_seconds"; do
    case "$value" in
        '' | *[!0-9]*)
            echo "error: timeout and watchdog durations must be non-negative integer seconds" >&2
            exit 64
            ;;
    esac
done
if [ "$timeout_seconds" -gt 0 ]; then
    # The in-process deadline produces the normal end-of-run reports. The independent shell
    # deadline is deliberately later: it exists for dyld/Rosetta hangs before Rust `main` and for
    # kernel teardown that never returns, neither of which can reach the runtime watchdog.
    watchdog_timeout_seconds=$((timeout_seconds + watchdog_slack_seconds))
else
    watchdog_timeout_seconds=0
fi
if [ ! -f "$game_path" ]; then
    echo "error: game executable not found: $game_path" >&2
    exit 66
fi
# Do not stack another translated game on an existing emulator. In particular, Darwin can retain
# a Rosetta process forever in uninterruptible U state while entering or leaving dyld; no signal
# can reclaim that process, and another launch puts additional pressure on the same kernel/driver
# path. This guard never kills an existing game. An expert can opt into deliberate concurrency.
if [ "${REMU_ALLOW_CONCURRENT_GAMES:-0}" != 1 ]; then
    active_game_processes=
    for pid in $(pgrep -x remu 2>/dev/null || true); do
        command=$(ps -p "$pid" -o command= 2>/dev/null || true)
        case "$command" in
            *" run --timeout-seconds "*)
                state=$(ps -p "$pid" -o state= 2>/dev/null | tr -d '[:space:]')
                active_game_processes="${active_game_processes}${active_game_processes:+
}pid=$pid state=${state:-unknown} command=$command"
                ;;
        esac
    done
    if [ -n "$active_game_processes" ]; then
        echo "error: refusing to launch while another REmu game process exists:" >&2
        printf '%s\n' "$active_game_processes" >&2
        echo "wait for it to exit; if it remains in U state, restart macOS before another safe launch" >&2
        echo "set REMU_ALLOW_CONCURRENT_GAMES=1 only for an intentional concurrent run" >&2
        exit 75
    fi
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(dirname -- "$script_dir")
bundled_metal_compiler="$repo_dir/.remu/toolchains/Metal.xctoolchain/usr/bin/metal"
# Xcode's downloadable Metal component is normally unpacked under a versioned directory (for
# example Metal-27A5209h/Metal.xctoolchain). Accept both that layout and the stable developer
# override above so an installed offline compiler is not silently ignored.
if [ ! -x "$bundled_metal_compiler" ]; then
    for candidate in "$repo_dir"/.remu/toolchains/Metal-*/Metal.xctoolchain/usr/bin/metal; do
        if [ -x "$candidate" ] && [ -x "$(dirname -- "$candidate")/metallib" ]; then
            bundled_metal_compiler=$candidate
            break
        fi
    done
fi
# Offline libraries can be retained across launches, avoiding tens of seconds of repeated cold
# runtime compilation for Unity's large shader set. Set REMU_USE_BUNDLED_METAL_COMPILER=0 to
# diagnose Apple's in-process source compiler explicitly.
if [ "${REMU_USE_BUNDLED_METAL_COMPILER:-1}" != 0 ] && \
    [ -z "${REMU_METAL_COMPILER:-}" ] && [ -x "$bundled_metal_compiler" ]; then
    REMU_METAL_COMPILER=$bundled_metal_compiler
    export REMU_METAL_COMPILER
fi
run_id=$(date -u +%Y%m%dT%H%M%SZ)
# Keeping the live app's redirected stdout under Documents makes macOS attribute inherited log
# writes to REmu and present a modal Documents-folder consent sheet. Runtime artifacts are
# diagnostics rather than project sources, so default them to the user's temporary directory.
run_root=${REMU_RUN_ROOT:-"${TMPDIR:-/tmp}/remu-runs"}
run_dir="$run_root/$run_id-$game_name"
frame_dir="$run_dir/frames"
guest_code_dir="$run_dir/guest-code"
msl_dir="$run_dir/msl"
graphics_lds_ir_dir="$run_dir/graphics-lds-ir"
scanout_msl_dir="$run_dir/scanout-msl"
generated_overlay_dir="$run_dir/generated-overlay"
raw_binary="$repo_dir/target/x86_64-apple-darwin/release/remu"
app_bundle=${REMU_APP_BUNDLE:-"$repo_dir/target/x86_64-apple-darwin/release/REMu.app"}
launch_working_directory=${REMU_LAUNCH_WORKING_DIRECTORY:-"$repo_dir"}
mkdir -p "$frame_dir"

# Ask the guest engine to build its render graph at the selected resolution. This is deliberately
# separate from REMU_PRESENTATION_SIZE: changing only the CAMetalLayer drawable leaves all guest
# render targets, post-processing and synchronization traffic at the title's original resolution.
# `native` preserves the title's request. Unreal titles read /app0/ue4commandline.txt before RHI
# initialization, so an isolated per-run overlay can select their real back-buffer and scene size
# without modifying the game dump. Other engines keep their native size until they receive an
# equivalent guest-facing mode implementation.
render_size=${REMU_RENDER_SIZE:-1920x1080}
render_policy=native
case "$(printf '%s' "$render_size" | tr '[:upper:]' '[:lower:]')" in
    native)
        ;;
    *x*)
        render_width=${render_size%%x*}
        render_height=${render_size#*x}
        case "$render_width" in
            '' | *[!0-9]*)
                echo "error: REMU_RENDER_SIZE must be native or WIDTHxHEIGHT, got '$render_size'" >&2
                exit 64
                ;;
        esac
        case "$render_height" in
            '' | *[!0-9]*)
                echo "error: REMU_RENDER_SIZE must be native or WIDTHxHEIGHT, got '$render_size'" >&2
                exit 64
                ;;
        esac
        if [ "$render_width" -eq 0 ] || [ "$render_height" -eq 0 ]; then
            echo "error: REMU_RENDER_SIZE dimensions must be positive" >&2
            exit 64
        fi
        if [ "$render_width" -gt 16384 ] || [ "$render_height" -gt 16384 ]; then
            echo "error: REMU_RENDER_SIZE dimensions must each be in 1..=16384" >&2
            exit 64
        fi
        render_size="${render_width}x${render_height}"
        if [ -z "${REMU_PRESENTATION_SIZE:-}" ]; then
            REMU_PRESENTATION_SIZE=$render_size
            export REMU_PRESENTATION_SIZE
        fi
        # A user overlay may carry more than a command line, so never hide it behind the generated
        # one. Users who supply REMU_OVERLAY_DIR retain complete control over UE configuration.
        if [ -z "${REMU_OVERLAY_DIR:-}" ]; then
            # Packaged UE executables do not necessarily retain their synthesized .uproject token.
            # The command-line file path itself is the reliable capability marker. Resolve the
            # project from the unique top-level Content/Paks directory, while allowing an explicit
            # token for unusual or ambiguous package layouts.
            ue_project=${REMU_UE_PROJECT:-}
            ue_commandline_marker=$(strings -a "$game_path" 2>/dev/null | awk '
                $0 == "/app0/ue4commandline.txt" {
                    print "yes"
                    exit
                }')
            if [ -z "$ue_project" ] && [ "$ue_commandline_marker" = yes ]; then
                game_root=$(dirname -- "$game_path")
                ue_project_count=0
                for candidate in "$game_root"/*; do
                    [ -d "$candidate" ] || continue
                    candidate_has_paks=0
                    for content_dir in "$candidate/content" "$candidate/Content"; do
                        if [ -d "$content_dir/paks" ] || [ -d "$content_dir/Paks" ]; then
                            candidate_has_paks=1
                            break
                        fi
                    done
                    [ "$candidate_has_paks" -eq 1 ] || continue
                    project_name=$(basename -- "$candidate")
                    ue_project="../../../$project_name/$project_name.uproject"
                    ue_project_count=$((ue_project_count + 1))
                done
                if [ "$ue_project_count" -ne 1 ]; then
                    ue_project=
                    render_policy=ue4-project-ambiguous
                fi
            fi
            if [ -n "$ue_project" ]; then
                ue_project_file=${ue_project##*/}
                ue_project_name=${ue_project_file%.uproject}
                if [ -z "$ue_project_name" ] || [ "$ue_project_name" = "$ue_project_file" ]; then
                    echo "error: REMU_UE_PROJECT must end in ProjectName.uproject, got '$ue_project'" >&2
                    exit 64
                fi
                ue_platform_config_dir="$generated_overlay_dir/app0/$ue_project_name/platforms/ps5/config"
                mkdir -p "$generated_overlay_dir/app0" "$ue_platform_config_dir"
                printf '%s -ResX=%s -ResY=%s -ForceRes -ExecCmds="r.SetRes %sx%sf"\n' \
                    "$ue_project" "$render_width" "$render_height" \
                    "$render_width" "$render_height" \
                    > "$generated_overlay_dir/app0/ue4commandline.txt"
                # UE4's PS5 display object consumes platform-specific back-buffer CVars before
                # the generic resolution command can resize it. The project/platform hierarchy is
                # the first loose layer loaded after the packaged Engine/Platforms/PS5 defaults,
                # so this changes the guest's real VideoOut buffer and render graph rather than
                # scaling a completed 4K frame in the presenter.
                printf '[SystemSettings]\nPS5.DefaultBackBufferWidth=%s\nPS5.DefaultBackBufferHeight=%s\n' \
                    "$render_width" "$render_height" \
                    > "$ue_platform_config_dir/ps5engine.ini"
                REMU_OVERLAY_DIR=$generated_overlay_dir
                export REMU_OVERLAY_DIR
                render_policy=ue4-ps5-backbuffer
            elif [ "$ue_commandline_marker" = yes ] && [ "$render_policy" != ue4-project-ambiguous ]; then
                render_policy=ue4-project-missing
            else
                render_policy=presentation-only
            fi
        else
            render_policy=user-overlay
        fi
        ;;
    *)
        echo "error: REMU_RENDER_SIZE must be native or WIDTHxHEIGHT, got '$render_size'" >&2
        exit 64
        ;;
esac
if [ "${REMU_DEEP_TRACE:-0}" = 1 ]; then
    # Keep one reproducible, self-contained view of the path from guest command construction
    # through shader translation and host presentation. These traces are intentionally opt-in:
    # register/draw logging and periodic frame readback perturb timing too much for ordinary
    # performance measurements.
    mkdir -p "$msl_dir" "$graphics_lds_ir_dir" "$scanout_msl_dir"
    REMU_CAPTURE_FRAMES=1
    REMU_CAPTURE_INTERVAL_SECONDS=${REMU_CAPTURE_INTERVAL_SECONDS:-2}
    REMU_TRACE_AGC_SHADERS=1
    REMU_TRACE_AGC_DRAWS=1
    REMU_TRACE_AGC_REGISTERS=1
    REMU_TRACE_AGC_PROGRAMS=1
    REMU_TRACE_IMAGES=1
    REMU_TRACE_TARGET_SAMPLING=1
    REMU_TRACE_VIDEO_TARGETS=1
    REMU_TRACE_SHADER_OPCODES=1
    REMU_TRACE_VFS=1
    REMU_TRACE_VM=1
    REMU_TRACE_SYNC=1
    REMU_TRACE_TIME=1
    REMU_TRACE_AVPLAYER=1
    REMU_TRACE_AVPLAYER_CONTENT=1
    REMU_TRACE_GUEST_THREADS=1
    REMU_TRACE_DISPATCH_STALLS=1
    REMU_GPU_TERMINAL_PHASES=1
    REMU_DUMP_MSL_DIR=$msl_dir
    REMU_DUMP_GRAPHICS_LDS_IR_DIR=$graphics_lds_ir_dir
    REMU_DUMP_SCANOUT_MSL_DIR=$scanout_msl_dir
    export REMU_CAPTURE_FRAMES REMU_CAPTURE_INTERVAL_SECONDS
    export REMU_TRACE_AGC_SHADERS REMU_TRACE_AGC_DRAWS
    export REMU_TRACE_AGC_REGISTERS REMU_TRACE_AGC_PROGRAMS
    export REMU_TRACE_IMAGES REMU_TRACE_TARGET_SAMPLING REMU_TRACE_VIDEO_TARGETS
    export REMU_TRACE_SHADER_OPCODES
    export REMU_TRACE_VFS REMU_TRACE_VM REMU_TRACE_SYNC REMU_TRACE_TIME
    export REMU_TRACE_AVPLAYER REMU_TRACE_AVPLAYER_CONTENT REMU_TRACE_GUEST_THREADS
    export REMU_TRACE_DISPATCH_STALLS REMU_GPU_TERMINAL_PHASES
    export REMU_DUMP_MSL_DIR
    export REMU_DUMP_GRAPHICS_LDS_IR_DIR REMU_DUMP_SCANOUT_MSL_DIR
fi
if [ -n "${REMU_TRACE_GUEST_CODE:-}" ] && [ -z "${REMU_DUMP_GUEST_CODE_DIR:-}" ]; then
    REMU_DUMP_GUEST_CODE_DIR=$guest_code_dir
    export REMU_DUMP_GUEST_CODE_DIR
fi

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

# Interactive runs never inject controller input unless the caller explicitly supplies a schedule.
# After its final scheduled pulse, the automation alternates Cross and Square once per second.
# Separate button edges keep title UIs from rejecting the two-button chord. Set
# REMU_AUTO_CROSS_REPEAT_MS=0 to keep an explicit schedule one-shot.
auto_cross=${REMU_AUTO_CROSS:-}
# Some first-boot and system-style title screens bind their explicit confirmation action to the
# controller's Options button. Keep this as a separate one-shot schedule so it never forms a
# chord with the existing Cross/Square repeat automation.
auto_options=${REMU_AUTO_OPTIONS:-}
# Scripted runs must never gate on a guest dialog waiting for a human. The runtime
# auto-completes dialogs when this is set; pass an explicit 0 to restore interactive dialogs.
REMU_AUTO_DISMISS_DIALOGS=${REMU_AUTO_DISMISS_DIALOGS:-1}
export REMU_AUTO_DISMISS_DIALOGS
# Capturing a 4K PPM writes roughly 32 MiB per image and fingerprints the full frame on the
# presenter thread. Keep it explicit so ordinary game runs measure the emulator rather than the
# diagnostic I/O path.
capture_frames=${REMU_CAPTURE_FRAMES:-0}

# The emulator owns its live title-bar telemetry. Keep Apple's Metal HUD disabled by default so
# it cannot obscure guest output or report presentation cadence as if it were guest frame rate.
if [ "${REMU_METAL_HUD:-0}" = 1 ]; then
    MTL_HUD_ENABLED=${MTL_HUD_ENABLED:-1}
    export MTL_HUD_ENABLED
else
    unset MTL_HUD_ENABLED MTL_HUD_LOG_ENABLED
fi
if [ "${REMU_METAL_HUD_LOG:-0}" = 1 ]; then
    MTL_HUD_ENABLED=${MTL_HUD_ENABLED:-1}
    MTL_HUD_LOG_ENABLED=${MTL_HUD_LOG_ENABLED:-1}
    export MTL_HUD_ENABLED MTL_HUD_LOG_ENABLED
fi
revision=$(git -C "$repo_dir" rev-parse --short HEAD 2>/dev/null || echo unknown)
# A commit name for a tree with uncommitted changes describes a build that was never committed.
if [ "$revision" != unknown ] && ! git -C "$repo_dir" diff-index --quiet HEAD -- 2>/dev/null; then
    revision="$revision-dirty"
fi
started_at=$(date +%s)
child_pid=

# Invoked indirectly by the traps below.
# shellcheck disable=SC2329
cleanup() {
    if [ -n "$child_pid" ] && kill -0 "$child_pid" 2>/dev/null; then
        # `child_pid` is the external watchdog wrapper. It owns bounded TERM/KILL escalation for
        # the emulator and remains alive if this script itself is forcefully terminated.
        kill "$child_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT HUP INT TERM

echo "[REMu] launching $game_name; artifacts: $run_dir"
echo "[REMu] render resolution: $render_size ($render_policy)"
if [ "$render_policy" = presentation-only ]; then
    echo "[REMu] warning: no guest-native resolution hook was found; only presentation is ${render_size}" >&2
fi
if [ "${REMU_DEEP_TRACE:-0}" = 1 ]; then
    echo "[REMu] unified deep trace enabled"
fi
if [ "${REMU_HEADLESS_METAL:-0}" != 1 ]; then
    echo "[REMu] launching the interactive REmu.app bundle"
fi
if [ "$timeout_seconds" -gt 0 ] 2>/dev/null; then
    echo "[REMu] automatic shutdown after ${timeout_seconds}s (pass 0 to disable)"
    echo "[REMu] external launch watchdog after ${watchdog_timeout_seconds}s"
fi
if [ -n "${REMU_METAL_COMPILER:-}" ]; then
    echo "[REMu] using offline Metal compiler: $REMU_METAL_COMPILER"
fi
if [ "${MTL_HUD_ENABLED:-0}" = 1 ]; then
    echo "[REMu] Metal Performance HUD enabled"
fi
if [ "${MTL_HUD_LOG_ENABLED:-0}" = 1 ]; then
    echo "[REMu] Metal Performance HUD logging enabled"
fi
if [ "${REMU_HEADLESS_METAL:-0}" = 1 ]; then
    echo "[REMu] headless Metal presenter enabled"
fi
set +e
# Ordered scanout-writer diagnostics also emit post-draw images through the presenter's capture
# root. They do not enable periodic frame capture, but they still need the run-local directory;
# otherwise the provenance trace succeeds while every requested pixel artifact is silently
# skipped. Keep ordinary runs isolated from an inherited capture directory as before.
if [ "$capture_frames" = 1 ] || [ -n "${REMU_DIAGNOSTIC_SCANOUT_WRITERS:-}" ]; then
    (
        cd "$launch_working_directory"
        "$script_dir/run-with-watchdog.sh" \
            "$watchdog_timeout_seconds" "$watchdog_termination_grace_seconds" -- \
            env -u LaunchInstanceID -u XPC_FLAGS -u XPC_SERVICE_NAME \
                REMU_AUTO_CROSS="$auto_cross" REMU_AUTO_OPTIONS="$auto_options" \
                REMU_CAPTURE_DIR="$frame_dir" \
                arch -x86_64 "$binary" run --timeout-seconds "$timeout_seconds" "$game_path"
    ) > "$run_dir/guest.log" 2>&1 &
else
    (
        cd "$launch_working_directory"
        "$script_dir/run-with-watchdog.sh" \
            "$watchdog_timeout_seconds" "$watchdog_termination_grace_seconds" -- \
            env -u LaunchInstanceID -u XPC_FLAGS -u XPC_SERVICE_NAME -u REMU_CAPTURE_DIR \
                REMU_AUTO_CROSS="$auto_cross" REMU_AUTO_OPTIONS="$auto_options" \
                arch -x86_64 "$binary" run --timeout-seconds "$timeout_seconds" "$game_path"
    ) > "$run_dir/guest.log" 2>&1 &
fi
child_pid=$!
wait "$child_pid"
status=$?
child_pid=
set -e

case "$status" in
    0)
        outcome=guest-exit
        runner_status=0
        ;;
    142)
        # The in-process diagnostic watchdog preserves the historical SIGALRM-style status.
        outcome=timeout
        runner_status=0
        ;;
    124)
        # The independent wrapper reached its later deadline. This means the in-process watchdog
        # never started or process teardown did not complete, so keep the run visibly failed.
        outcome=external-watchdog-timeout
        runner_status=$status
        ;;
    137)
        # SIGKILL from an external actor (concurrent launch cleanup, OOM killer, manual kill -9).
        outcome=host-sigkill
        runner_status=$status
        ;;
    143)
        # SIGTERM from orchestration (wrapper cleanup, concurrent run termination, shell trap).
        outcome=host-sigterm
        runner_status=$status
        ;;
    *)
        outcome=guest-failure
        runner_status=$status
        ;;
esac

finished_at=$(date +%s)
elapsed_seconds=$((finished_at - started_at))
frame_count=$(find "$frame_dir" -type f -name '*.ppm' | wc -l | tr -d ' ')
unimplemented_count=$(grep -c '\[REMu\]\[IMPORT\] unimplemented' "$run_dir/guest.log" || true)
{
    echo "revision=$revision"
    echo "game=$game_path"
    echo "timeout_seconds=$timeout_seconds"
    echo "elapsed_seconds=$elapsed_seconds"
    echo "outcome=$outcome"
    echo "exit_status=$status"
    echo "captured_frames=$frame_count"
    echo "unimplemented_nids=$unimplemented_count"
    echo "render_size=$render_size"
    echo "render_policy=$render_policy"
} > "$run_dir/summary.txt"
# The runtime prints exact ABI-gap totals ([REMu][REPORT] lines) periodically and on every exit
# path, so a run killed by a forwarded fault still leaves one. Keep only the final block: the
# earlier snapshots are prefixes of it, and copying them all would bury the totals that matter.
awk '/\[REMu\]\[REPORT\] --- ABI gap summary/ { block = "" }
     /\[REMu\]\[REPORT\]/ { block = block $0 "\n" }
     END { printf "%s", block }' "$run_dir/guest.log" >> "$run_dir/summary.txt" || true

tail -n 30 "$run_dir/guest.log" || true
cat "$run_dir/summary.txt"
echo "[REMu] complete; full artifacts: $run_dir"
exit "$runner_status"
