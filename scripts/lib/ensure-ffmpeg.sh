#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later
# Build the pinned static FFmpeg subset used by REmu's in-process media bridge.
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 {x86_64-apple-darwin|aarch64-apple-darwin|x86_64-pc-windows-gnu}" >&2
    exit 64
fi

target=$1
host_os=$(uname -s)
case "$target" in
    x86_64-apple-darwin)
        ffmpeg_arch=x86_64
        ffmpeg_target_os=darwin
        ffmpeg_cc=clang
        arch_cflags="-arch x86_64"
        thread_config="--enable-pthreads"
        x86_config=--disable-x86asm
        case "$host_os" in
            Darwin) ;;
            *)
                echo "error: bundled FFmpeg for Apple targets must be built on macOS" >&2
                exit 69
                ;;
        esac
        ;;
    aarch64-apple-darwin)
        ffmpeg_arch=aarch64
        ffmpeg_target_os=darwin
        ffmpeg_cc=clang
        arch_cflags="-arch arm64"
        thread_config="--enable-pthreads"
        x86_config=
        case "$host_os" in
            Darwin) ;;
            *)
                echo "error: bundled FFmpeg for Apple targets must be built on macOS" >&2
                exit 69
                ;;
        esac
        ;;
    x86_64-pc-windows-gnu)
        ffmpeg_arch=x86_64
        ffmpeg_target_os=mingw32
        ffmpeg_cc=x86_64-w64-mingw32-gcc
        arch_cflags="-D_WIN32_WINNT=0x0A00"
        thread_config="--disable-pthreads --enable-w32threads"
        # Keep the bootstrap independent of a separately installed NASM. This only affects
        # FFmpeg's host-side codec implementation, never the guest GPU shader feature set.
        x86_config=--disable-x86asm
        case "$host_os" in
            MINGW* | MSYS* | CYGWIN*) ;;
            *)
                echo "error: bundled FFmpeg for Windows targets must be built in a MinGW/MSYS environment" >&2
                exit 69
                ;;
        esac
        ;;
    *)
        echo "error: bundled FFmpeg is not configured for target '$target'" >&2
        exit 69
        ;;
esac

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo "error: verifying bundled FFmpeg requires sha256sum or shasum" >&2
        return 69
    fi
}

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(dirname -- "$(dirname -- "$script_dir")")
ffmpeg_version=8.1.2
ffmpeg_sha256=464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c
# Bump when configure flags or the integration contract changes. Keeping it in the cache path
# prevents a previous partial/minimal build from masquerading as the current dependency.
build_revision=remu6
deps_root=${REMU_DEPS_DIR:-"$repo_dir/.remu/deps"}
download_dir="$deps_root/downloads"
install_parent="$deps_root/ffmpeg/$ffmpeg_version-$build_revision"
prefix="$install_parent/$target"
published_prefix=$prefix
if [ "$ffmpeg_target_os" = mingw32 ]; then
    published_prefix=$(cygpath -m "$prefix")
fi
complete_marker="$prefix/.remu-ffmpeg-complete"
archive="$download_dir/ffmpeg-$ffmpeg_version.tar.xz"
archive_url="https://ffmpeg.org/releases/ffmpeg-$ffmpeg_version.tar.xz"

required_install_is_complete() {
    [ -f "$complete_marker" ] || return 1
    [ -f "$prefix/include/libavcodec/avcodec.h" ] || return 1
    [ -f "$prefix/include/libavutil/samplefmt.h" ] || return 1
    [ -f "$prefix/include/libavutil/pixfmt.h" ] || return 1
    [ -f "$prefix/include/libavutil/frame.h" ] || return 1
    for library in avformat avcodec avutil swscale swresample; do
        [ -f "$prefix/lib/lib$library.a" ] || return 1
        [ -f "$prefix/lib/pkgconfig/lib$library.pc" ] || return 1
        grep -Fqx "prefix=$published_prefix" "$prefix/lib/pkgconfig/lib$library.pc" || return 1
    done
}

if required_install_is_complete; then
    printf '%s\n' "$published_prefix"
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
    printf '%s\n' "$published_prefix"
    exit 0
fi

if [ -f "$archive" ]; then
    actual_sha256=$(sha256_file "$archive")
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
    actual_sha256=$(sha256_file "$archive_partial")
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
object_dir="$build_dir/objects"
source_dir="$build_dir/source"
if [ "$ffmpeg_target_os" = mingw32 ]; then
    # Keeping the source under the object tree makes FFmpeg record `src` instead of a long
    # absolute path. Besides native Make compatibility, this keeps libavutil's large public-header
    # install command below Windows command-line limits.
    source_dir="$object_dir/src"
fi
stage_prefix="$build_dir/install"
mkdir -p "$source_dir" "$object_dir" "$stage_prefix"
tar -xJf "$archive" -C "$source_dir" --strip-components=1

configure_path="$source_dir/configure"
configure_prefix=$stage_prefix
native_source_dir=$source_dir
if [ "$ffmpeg_target_os" = mingw32 ]; then
    # Native mingw32-make cannot resolve MSYS /c/... prerequisites. Invoke configure through a
    # drive-letter path so its generated Makefiles use paths understood by both Make and sh.
    configure_path=$(cygpath -m "$configure_path")
    configure_prefix=$(cygpath -m "$stage_prefix")
    native_source_dir=$(cygpath -m "$source_dir")
fi

deployment_target=
common_cflags=$arch_cflags
if [ "$ffmpeg_target_os" = darwin ]; then
    deployment_target=${MACOSX_DEPLOYMENT_TARGET:-13.0}
    common_cflags="$arch_cflags -mmacosx-version-min=$deployment_target"
fi
echo "[REMu][FFMPEG] configuring FFmpeg $ffmpeg_version for $target" >&2
(
    cd "$object_dir"
    "$configure_path" \
        --prefix="$configure_prefix" \
        --arch="$ffmpeg_arch" \
        --target-os="$ffmpeg_target_os" \
        --cc="$ffmpeg_cc" \
        --extra-cflags="$common_cflags" \
        --extra-ldflags="$common_cflags" \
        --enable-static \
        --disable-shared \
        --enable-pic \
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
        $thread_config \
        $x86_config
) >&2

if [ "$ffmpeg_target_os" = mingw32 ]; then
    # Older/non-relative layouts can still make FFmpeg canonicalize its source through POSIX
    # `pwd`. Only these two generated Make metadata files consume that path; never scan the source
    # tree (which is intentionally nested below the object directory on Windows).
    for generated_file in "$object_dir/Makefile" "$object_dir/ffbuild/config.mak"; do
        if [ -f "$generated_file" ] && grep -Fq "$source_dir" "$generated_file" 2>/dev/null; then
            rewritten_file="$generated_file.remu"
            sed "s|$source_dir|$native_source_dir|g" "$generated_file" > "$rewritten_file"
            mv "$rewritten_file" "$generated_file"
        fi
    done
fi

build_jobs=${NUMBER_OF_PROCESSORS:-}
if [ -z "$build_jobs" ]; then
    build_jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || printf '4')
fi
case "$build_jobs" in
    '' | *[!0-9]*) build_jobs=4 ;;
esac
if command -v make >/dev/null 2>&1; then
    make_command=make
elif command -v mingw32-make >/dev/null 2>&1; then
    make_command=mingw32-make
else
    echo "error: building bundled FFmpeg requires GNU make" >&2
    exit 69
fi
echo "[REMu][FFMPEG] compiling FFmpeg $ffmpeg_version for $target ($build_jobs jobs)" >&2
"$make_command" -C "$object_dir" -j"$build_jobs" >&2
"$make_command" -C "$object_dir" install >&2

# FFmpeg writes its configure-time prefix into pkg-config files. The install is built in a staging
# directory for atomic publication, so rewrite that one path before moving the completed tree.
for pc_file in "$stage_prefix"/lib/pkgconfig/*.pc; do
    rewritten_pc="$pc_file.remu"
    sed \
        -e "s|$stage_prefix|$prefix|g" \
        -e "s|$configure_prefix|$published_prefix|g" \
        -e 's/ -framework CoreFoundation//g' \
        -e 's/ -framework CoreMedia//g' \
        -e 's/ -framework CoreVideo//g' \
        "$pc_file" > "$rewritten_pc"
    mv "$rewritten_pc" "$pc_file"
done

for library in avformat avcodec avutil swscale swresample; do
    test -f "$stage_prefix/lib/lib$library.a"
    test -f "$stage_prefix/lib/pkgconfig/lib$library.pc"
    if [ "$ffmpeg_target_os" = darwin ]; then
        library_arch=$(lipo -archs "$stage_prefix/lib/lib$library.a")
        case " $library_arch " in
            *" ${arch_cflags#-arch } "*) ;;
            *)
                echo "error: lib$library.a has architecture '$library_arch', expected '${arch_cflags#-arch }'" >&2
                exit 65
                ;;
        esac
    else
        # GNU ar rejects malformed/non-archive files. The target compiler identity guards against
        # accidentally publishing an archive built for a different Windows architecture.
        "$ffmpeg_cc" -dumpmachine | grep -Eq '^x86_64-w64-mingw32$'
        ffmpeg_ar=$("$ffmpeg_cc" -print-prog-name=ar)
        "$ffmpeg_ar" t "$stage_prefix/lib/lib$library.a" >/dev/null
    fi
done

{
    echo "version=$ffmpeg_version"
    echo "source=$archive_url"
    echo "sha256=$ffmpeg_sha256"
    echo "target=$target"
    if [ -n "$deployment_target" ]; then
        echo "deployment_target=$deployment_target"
    fi
    echo "build_revision=$build_revision"
} > "$stage_prefix/.remu-ffmpeg-complete"

if [ -d "$prefix" ]; then
    rm -rf "$prefix"
fi
mv "$stage_prefix" "$prefix"
echo "[REMu][FFMPEG] installed static FFmpeg at $published_prefix" >&2
printf '%s\n' "$published_prefix"
