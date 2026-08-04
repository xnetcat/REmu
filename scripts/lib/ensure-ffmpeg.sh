#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Build the pinned static FFmpeg subset used by REmu's in-process media bridge.
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 {x86_64-apple-darwin|aarch64-apple-darwin}" >&2
    exit 64
fi

target=$1
case "$target" in
    x86_64-apple-darwin)
        ffmpeg_arch=x86_64
        arch_cflags="-arch x86_64"
        x86_config=--disable-x86asm
        ;;
    aarch64-apple-darwin)
        ffmpeg_arch=aarch64
        arch_cflags="-arch arm64"
        x86_config=
        ;;
    *)
        echo "error: bundled FFmpeg is not configured for target '$target'" >&2
        exit 69
        ;;
esac

if [ "$(uname -s)" != Darwin ]; then
    echo "error: bundled FFmpeg for Apple targets must be built on macOS" >&2
    exit 69
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(dirname -- "$(dirname -- "$script_dir")")
ffmpeg_version=8.1.2
ffmpeg_sha256=464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c
# Bump when configure flags or the integration contract changes. Keeping it in the cache path
# prevents a previous partial/minimal build from masquerading as the current dependency.
build_revision=remu4
deps_root=${REMU_DEPS_DIR:-"$repo_dir/.remu/deps"}
download_dir="$deps_root/downloads"
install_parent="$deps_root/ffmpeg/$ffmpeg_version-$build_revision"
prefix="$install_parent/$target"
complete_marker="$prefix/.remu-ffmpeg-complete"
archive="$download_dir/ffmpeg-$ffmpeg_version.tar.xz"
archive_url="https://ffmpeg.org/releases/ffmpeg-$ffmpeg_version.tar.xz"

required_install_is_complete() {
    [ -f "$complete_marker" ] || return 1
    [ -f "$prefix/include/libavcodec/avcodec.h" ] || return 1
    for library in avformat avcodec avutil swscale swresample; do
        [ -f "$prefix/lib/lib$library.a" ] || return 1
        [ -f "$prefix/lib/pkgconfig/lib$library.pc" ] || return 1
        grep -Fqx "prefix=$prefix" "$prefix/lib/pkgconfig/lib$library.pc" || return 1
    done
}

if required_install_is_complete; then
    printf '%s\n' "$prefix"
    exit 0
fi

mkdir -p "$download_dir" "$install_parent"
lock_dir="$install_parent/.$target.lock"
while ! mkdir "$lock_dir" 2>/dev/null; do
    if [ -f "$lock_dir/pid" ]; then
        lock_pid=$(sed -n '1p' "$lock_dir/pid" 2>/dev/null || true)
        case "$lock_pid" in
            '' | *[!0-9]*) lock_pid=0 ;;
        esac
        if [ "$lock_pid" -eq 0 ] || ! kill -0 "$lock_pid" 2>/dev/null; then
            rm -f "$lock_dir/pid"
            rmdir "$lock_dir" 2>/dev/null || true
            continue
        fi
    fi
    sleep 1
done
printf '%s\n' "$$" > "$lock_dir/pid"

build_dir=
cleanup() {
    if [ -n "$build_dir" ] && [ -d "$build_dir" ]; then
        rm -rf "$build_dir"
    fi
    rm -f "$lock_dir/pid"
    rmdir "$lock_dir" 2>/dev/null || true
}
trap cleanup EXIT HUP INT TERM

# Another Cargo process may have completed the dependency while this process waited for the lock.
if required_install_is_complete; then
    printf '%s\n' "$prefix"
    exit 0
fi

if [ -f "$archive" ]; then
    actual_sha256=$(shasum -a 256 "$archive" | awk '{print $1}')
    if [ "$actual_sha256" != "$ffmpeg_sha256" ]; then
        echo "[REMu][FFMPEG] cached source checksum mismatch; downloading a clean copy" >&2
        rm -f "$archive"
    fi
fi
if [ ! -f "$archive" ]; then
    archive_partial="$archive.partial.$$"
    echo "[REMu][FFMPEG] downloading FFmpeg $ffmpeg_version from $archive_url" >&2
    curl --fail --location --proto '=https' --tlsv1.2 \
        --output "$archive_partial" "$archive_url"
    actual_sha256=$(shasum -a 256 "$archive_partial" | awk '{print $1}')
    if [ "$actual_sha256" != "$ffmpeg_sha256" ]; then
        rm -f "$archive_partial"
        echo "error: FFmpeg $ffmpeg_version checksum mismatch" >&2
        echo "expected: $ffmpeg_sha256" >&2
        echo "actual:   $actual_sha256" >&2
        exit 65
    fi
    mv "$archive_partial" "$archive"
fi

build_dir=$(mktemp -d "$install_parent/.build-$target.XXXXXX")
source_dir="$build_dir/source"
object_dir="$build_dir/objects"
stage_prefix="$build_dir/install"
mkdir -p "$source_dir" "$object_dir" "$stage_prefix"
tar -xJf "$archive" -C "$source_dir" --strip-components=1

deployment_target=${MACOSX_DEPLOYMENT_TARGET:-13.0}
common_cflags="$arch_cflags -mmacosx-version-min=$deployment_target"
echo "[REMu][FFMPEG] configuring FFmpeg $ffmpeg_version for $target" >&2
(
    cd "$object_dir"
    "$source_dir/configure" \
        --prefix="$stage_prefix" \
        --arch="$ffmpeg_arch" \
        --target-os=darwin \
        --cc=clang \
        --extra-cflags="$common_cflags" \
        --extra-ldflags="$common_cflags" \
        --enable-static \
        --disable-shared \
        --enable-pic \
        --enable-pthreads \
        --disable-programs \
        --disable-doc \
        --disable-debug \
        --disable-network \
        --disable-autodetect \
        --disable-avdevice \
        --disable-avfilter \
        --disable-iconv \
        --disable-zlib \
        --disable-bzlib \
        --disable-lzma \
        --disable-videotoolbox \
        --disable-audiotoolbox \
        --disable-everything \
        --enable-avcodec \
        --enable-avformat \
        --enable-avutil \
        --enable-swresample \
        --enable-swscale \
        --enable-protocol=file \
        --enable-demuxer=mov \
        --enable-decoder=aac \
        --enable-decoder=h264 \
        --enable-decoder=hevc \
        --enable-parser=aac \
        --enable-parser=h264 \
        --enable-parser=hevc \
        $x86_config
) >&2

build_jobs=$(sysctl -n hw.logicalcpu 2>/dev/null || printf '4')
case "$build_jobs" in
    '' | *[!0-9]*) build_jobs=4 ;;
esac
echo "[REMu][FFMPEG] compiling FFmpeg $ffmpeg_version for $target ($build_jobs jobs)" >&2
make -C "$object_dir" -j"$build_jobs" >&2
make -C "$object_dir" install >&2

# FFmpeg writes its configure-time prefix into pkg-config files. The install is built in a staging
# directory for atomic publication, so rewrite that one path before moving the completed tree.
for pc_file in "$stage_prefix"/lib/pkgconfig/*.pc; do
    rewritten_pc="$pc_file.remu"
    sed \
        -e "s|$stage_prefix|$prefix|g" \
        -e 's/ -framework CoreFoundation//g' \
        -e 's/ -framework CoreMedia//g' \
        -e 's/ -framework CoreVideo//g' \
        "$pc_file" > "$rewritten_pc"
    mv "$rewritten_pc" "$pc_file"
done

for library in avformat avcodec avutil swscale swresample; do
    test -f "$stage_prefix/lib/lib$library.a"
    test -f "$stage_prefix/lib/pkgconfig/lib$library.pc"
    library_arch=$(lipo -archs "$stage_prefix/lib/lib$library.a")
    case " $library_arch " in
        *" ${arch_cflags#-arch } "*) ;;
        *)
            echo "error: lib$library.a has architecture '$library_arch', expected '${arch_cflags#-arch }'" >&2
            exit 65
            ;;
    esac
done

{
    echo "version=$ffmpeg_version"
    echo "source=$archive_url"
    echo "sha256=$ffmpeg_sha256"
    echo "target=$target"
    echo "deployment_target=$deployment_target"
    echo "build_revision=$build_revision"
} > "$stage_prefix/.remu-ffmpeg-complete"

if [ -d "$prefix" ]; then
    rm -rf "$prefix"
fi
mv "$stage_prefix" "$prefix"
echo "[REMu][FFMPEG] installed static FFmpeg at $prefix" >&2
printf '%s\n' "$prefix"
