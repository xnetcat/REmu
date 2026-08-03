// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! MSL compute kernels for mode-27 then mode-5 de/re-tile (MASTER-PLAN B2).
//!
//! Equations match `presenter/tiling` (CPU differential oracle). The specialised loops in
//! SDK `generated_27.cpp` / `generated_5.cpp` implement the same within-block XOR tables;
//! we port the tables, not the SIMD specialisations.

use metal::{CompileOptions, Device, Library};
use std::sync::OnceLock;

use super::equations::GpuTilerMode;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // Named by compile_tiler_library consumers once residency migrates.
pub(crate) enum TilerKernelKind {
    Detile,
    Retile,
}

/// Mode-27 (64KB_R_X) tiler — first B2 land target (HZB + most colour targets).
pub(crate) const MODE27_TILER_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Parameter layout matches the rematerialisation ABI so residency can migrate without
// rewriting encoders. Fields are ulong for an unambiguous host ABI.
enum RemuTiler27Param {
    kWidth = 0, kHeight = 1, kLinearRowBytes = 2, kLinearOffset = 3,
    kResourceBase = 4, kTargetBase = 5, kLevelOffset = 6, kPitch = 7,
    kBytesPerElement = 8, kBlockWidth = 9, kBlockHeight = 10,
    kTailX = 11, kTailY = 12, kSurfaceZXor = 13, kInTail = 14,
    kTargetLength = 15, kRunCount = 16
};

// Port of presenter/tiling/mode27.rs:mode27_within_block_byte_offset_for_format —
// same bit table specialised by generated_27.cpp's XOR chains (e.g. :31-34 for 2 BPE).
static uint remu_mode27_within_block(uint x, uint y, uint log2_bytes) {
    switch (log2_bytes) {
        case 0:
            return ((y << 2) & 0x0008u) ^ ((y << 4) & 0x0010u)
                ^ ((y << 3) & 0x00a0u) ^ ((y << 5) & 0x0f00u)
                ^ ((y << 6) & 0x1000u) ^ ((y << 7) & 0x4000u)
                ^ (x & 0x0007u) ^ ((x << 3) & 0x0040u)
                ^ ((x << 5) & 0x0300u) ^ ((x << 4) & 0x0400u)
                ^ ((x << 6) & 0x0800u) ^ ((x << 7) & 0x2000u)
                ^ ((x << 8) & 0x8000u);
        case 1:
            return ((y << 4) & 0x0070u) ^ ((y << 5) & 0x0f00u)
                ^ ((y << 8) & 0x5000u) ^ ((x << 1) & 0x000eu)
                ^ ((x << 4) & 0x0480u) ^ ((x << 5) & 0x0300u)
                ^ ((x << 6) & 0x0800u) ^ ((x << 7) & 0x2000u)
                ^ ((x << 8) & 0x8000u);
        case 2:
            return ((y << 4) & 0x0070u) ^ ((y << 5) & 0x0f00u)
                ^ ((y << 9) & 0x1000u) ^ ((y << 8) & 0x4000u)
                ^ ((x << 2) & 0x000cu) ^ ((x << 5) & 0x0380u)
                ^ ((x << 4) & 0x0400u) ^ ((x << 6) & 0x0800u)
                ^ ((x << 9) & 0xa000u);
        case 3:
            return ((y << 4) & 0x0010u) ^ ((y << 6) & 0x0080u)
                ^ ((y << 5) & 0x0f00u) ^ ((y << 10) & 0x5000u)
                ^ ((x << 3) & 0x0008u) ^ ((x << 4) & 0x0460u)
                ^ ((x << 5) & 0x0300u) ^ ((x << 6) & 0x0800u)
                ^ ((x << 10) & 0x2000u) ^ ((x << 9) & 0x8000u);
        default:
            return ((x << 4) & 0x0410u) ^ ((x << 5) & 0x0340u)
                ^ ((x << 6) & 0x0800u) ^ ((x << 11) & 0xa000u)
                ^ ((y << 5) & 0x0f20u) ^ ((y << 6) & 0x0080u)
                ^ ((y << 10) & 0x1000u) ^ ((y << 11) & 0x4000u);
    }
}

static ulong remu_mode27_physical(constant ulong* p, uint2 gid) {
    const uint block_width = uint(p[kBlockWidth]);
    const uint block_height = uint(p[kBlockHeight]);
    const bool tail = p[kInTail] != 0;
    const uint x = tail ? uint(p[kTailX]) + gid.x : gid.x;
    const uint y = tail ? uint(p[kTailY]) + gid.y : gid.y;
    const uint block_x = tail ? 0u : gid.x / block_width;
    const uint block_y = tail ? 0u : gid.y / block_height;
    const ulong blocks_per_row = (p[kPitch] + block_width - 1u) / block_width;
    const ulong block = tail ? 0ul : ulong(block_y) * blocks_per_row + block_x;
    const uint log2_bytes = ctz(uint(p[kBytesPerElement]));
    const uint within = remu_mode27_within_block(x, y, log2_bytes) ^ uint(p[kSurfaceZXor]);
    return p[kResourceBase] + p[kLevelOffset] + block * 65536ul + within;
}

static bool remu_in_debt(device const ulong2* runs, ulong count, ulong address) {
    for (ulong index = 0; index < count; ++index) {
        if (runs[index].x <= address && address < runs[index].y) return true;
    }
    return false;
}

kernel void remu_tiler27_detile(
    device const uchar* tiled [[buffer(0)]], device uchar* linear [[buffer(1)]],
    constant ulong* p [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p[kWidth] || gid.y >= p[kHeight]) return;
    const ulong physical = remu_mode27_physical(p, gid);
    const ulong destination = p[kLinearOffset] + ulong(gid.y) * p[kLinearRowBytes]
        + ulong(gid.x) * p[kBytesPerElement];
    if (physical < p[kTargetBase]
        || physical - p[kTargetBase] + p[kBytesPerElement] > p[kTargetLength]) return;
    for (ulong byte = 0; byte < p[kBytesPerElement]; ++byte) {
        linear[destination + byte] = tiled[physical - p[kTargetBase] + byte];
    }
}

kernel void remu_tiler27_retile(
    device const uchar* linear [[buffer(0)]], device uchar* tiled [[buffer(1)]],
    device const ulong2* debt [[buffer(2)]], constant ulong* p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p[kWidth] || gid.y >= p[kHeight]) return;
    const ulong physical = remu_mode27_physical(p, gid);
    const ulong source = p[kLinearOffset] + ulong(gid.y) * p[kLinearRowBytes]
        + ulong(gid.x) * p[kBytesPerElement];
    for (ulong byte = 0; byte < p[kBytesPerElement]; ++byte) {
        const ulong address = physical + byte;
        if (address >= p[kTargetBase]
            && address - p[kTargetBase] < p[kTargetLength]
            && remu_in_debt(debt, p[kRunCount], address)) {
            tiled[address - p[kTargetBase]] = linear[source + byte];
        }
    }
}
"#;

/// Mode-5 (4KB_S) tiler — second B2 land target after mode-27.
pub(crate) const MODE5_TILER_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

enum RemuTiler5Param {
    kWidth = 0, kHeight = 1, kLinearRowBytes = 2, kLinearOffset = 3,
    kResourceBase = 4, kTargetBase = 5, kLevelOffset = 6, kPitch = 7,
    kBytesPerElement = 8, kBlockWidth = 9, kBlockHeight = 10,
    kTailX = 11, kTailY = 12, kInTail = 13,
    kTargetLength = 14, kRunCount = 15
};

// Port of presenter/tiling/blocks.rs:mode5_within_block_byte_offset —
// bit-interleave table specialised by generated_5.cpp (e.g. :31-33 for common BPE).
static uint remu_mode5_within_block(uint x, uint y, uint log2_bytes) {
    switch (log2_bytes) {
        case 0:
            return (x & 0x000fu) | ((y & 0x000fu) << 4) | ((y & 0x0010u) << 4)
                | ((x & 0x0010u) << 5) | ((y & 0x0020u) << 5) | ((x & 0x0020u) << 6);
        case 1:
            return ((x & 0x0007u) << 1) | ((y & 0x0007u) << 4) | ((x & 0x0008u) << 4)
                | ((y & 0x0008u) << 5) | ((x & 0x0010u) << 5) | ((y & 0x0010u) << 6)
                | ((x & 0x0020u) << 6);
        case 2:
            return ((x & 0x0003u) << 2) | ((y & 0x0007u) << 4) | ((x & 0x0004u) << 5)
                | ((y & 0x0008u) << 5) | ((x & 0x0008u) << 6) | ((y & 0x0010u) << 6)
                | ((x & 0x0010u) << 7);
        case 3:
            return ((x & 0x0001u) << 3) | ((y & 0x0003u) << 4) | ((x & 0x0002u) << 5)
                | ((x & 0x0004u) << 5) | ((y & 0x0004u) << 6) | ((x & 0x0008u) << 6)
                | ((y & 0x0008u) << 7) | ((x & 0x0010u) << 7);
        default:
            return ((y & 0x0003u) << 4) | ((x & 0x0003u) << 6) | ((y & 0x0004u) << 6)
                | ((x & 0x0004u) << 7) | ((y & 0x0008u) << 7) | ((x & 0x0008u) << 8);
    }
}

static ulong remu_mode5_physical(constant ulong* p, uint2 gid) {
    const uint block_width = uint(p[kBlockWidth]);
    const uint block_height = uint(p[kBlockHeight]);
    const bool tail = p[kInTail] != 0;
    const uint x = tail ? uint(p[kTailX]) + gid.x : gid.x;
    const uint y = tail ? uint(p[kTailY]) + gid.y : gid.y;
    const uint block_x = tail ? 0u : gid.x / block_width;
    const uint block_y = tail ? 0u : gid.y / block_height;
    const ulong blocks_per_row = (p[kPitch] + block_width - 1u) / block_width;
    const ulong block = tail ? 0ul : ulong(block_y) * blocks_per_row + block_x;
    const uint log2_bytes = ctz(uint(p[kBytesPerElement]));
    const uint within = remu_mode5_within_block(x, y, log2_bytes);
    return p[kResourceBase] + p[kLevelOffset] + block * 4096ul + within;
}

static bool remu_in_debt5(device const ulong2* runs, ulong count, ulong address) {
    for (ulong index = 0; index < count; ++index) {
        if (runs[index].x <= address && address < runs[index].y) return true;
    }
    return false;
}

kernel void remu_tiler5_detile(
    device const uchar* tiled [[buffer(0)]], device uchar* linear [[buffer(1)]],
    constant ulong* p [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p[kWidth] || gid.y >= p[kHeight]) return;
    const ulong physical = remu_mode5_physical(p, gid);
    const ulong destination = p[kLinearOffset] + ulong(gid.y) * p[kLinearRowBytes]
        + ulong(gid.x) * p[kBytesPerElement];
    if (physical < p[kTargetBase]
        || physical - p[kTargetBase] + p[kBytesPerElement] > p[kTargetLength]) return;
    for (ulong byte = 0; byte < p[kBytesPerElement]; ++byte) {
        linear[destination + byte] = tiled[physical - p[kTargetBase] + byte];
    }
}

kernel void remu_tiler5_retile(
    device const uchar* linear [[buffer(0)]], device uchar* tiled [[buffer(1)]],
    device const ulong2* debt [[buffer(2)]], constant ulong* p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p[kWidth] || gid.y >= p[kHeight]) return;
    const ulong physical = remu_mode5_physical(p, gid);
    const ulong source = p[kLinearOffset] + ulong(gid.y) * p[kLinearRowBytes]
        + ulong(gid.x) * p[kBytesPerElement];
    for (ulong byte = 0; byte < p[kBytesPerElement]; ++byte) {
        const ulong address = physical + byte;
        if (address >= p[kTargetBase]
            && address - p[kTargetBase] < p[kTargetLength]
            && remu_in_debt5(debt, p[kRunCount], address)) {
            tiled[address - p[kTargetBase]] = linear[source + byte];
        }
    }
}
"#;

pub(crate) fn tiler_kernel_source(mode: GpuTilerMode) -> &'static str {
    match mode {
        GpuTilerMode::Mode5 => MODE5_TILER_MSL,
        GpuTilerMode::Mode27 => MODE27_TILER_MSL,
    }
}

#[allow(dead_code)] // Kernel entry lookup once residency migrates to this module's libraries.
pub(crate) fn tiler_kernel_name(mode: GpuTilerMode, kind: TilerKernelKind) -> &'static str {
    match (mode, kind) {
        (GpuTilerMode::Mode27, TilerKernelKind::Detile) => "remu_tiler27_detile",
        (GpuTilerMode::Mode27, TilerKernelKind::Retile) => "remu_tiler27_retile",
        (GpuTilerMode::Mode5, TilerKernelKind::Detile) => "remu_tiler5_detile",
        (GpuTilerMode::Mode5, TilerKernelKind::Retile) => "remu_tiler5_retile",
    }
}

/// Compiles the tiler library for `mode`. Failures stay loud — never fall back to a
/// silently-wrong kernel.
pub(crate) fn compile_tiler_library(device: &Device, mode: GpuTilerMode) -> Result<Library, String> {
    let options = CompileOptions::new();
    device
        .new_library_with_source(tiler_kernel_source(mode), &options)
        .map_err(|error| format!("GPU tiler MSL compile failed for {:?}: {error}", mode))
}

/// Cached per-process libraries so residency / deferred paths share one compile.
#[allow(dead_code)] // Used once residency migrates off the inline rematerialisation source.
pub(crate) fn cached_tiler_library(device: &Device, mode: GpuTilerMode) -> Result<&'static Library, String> {
    match mode {
        GpuTilerMode::Mode27 => {
            static LIB: OnceLock<Result<Library, String>> = OnceLock::new();
            match LIB.get_or_init(|| compile_tiler_library(device, GpuTilerMode::Mode27)) {
                Ok(library) => Ok(library),
                Err(error) => Err(error.clone()),
            }
        }
        GpuTilerMode::Mode5 => {
            static LIB: OnceLock<Result<Library, String>> = OnceLock::new();
            match LIB.get_or_init(|| compile_tiler_library(device, GpuTilerMode::Mode5)) {
                Ok(library) => Ok(library),
                Err(error) => Err(error.clone()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiler_sources_name_detile_and_retile_kernels() {
        assert!(MODE27_TILER_MSL.contains("kernel void remu_tiler27_detile"));
        assert!(MODE27_TILER_MSL.contains("kernel void remu_tiler27_retile"));
        assert!(MODE5_TILER_MSL.contains("kernel void remu_tiler5_detile"));
        assert!(MODE5_TILER_MSL.contains("kernel void remu_tiler5_retile"));
        assert_eq!(
            tiler_kernel_name(GpuTilerMode::Mode27, TilerKernelKind::Detile),
            "remu_tiler27_detile"
        );
    }

    #[test]
    fn mode27_tiler_library_compiles_on_metal() {
        let Some(device) = Device::system_default() else {
            eprintln!("[REMu][TILER] no Metal device; skipping compile proof");
            return;
        };
        let library = compile_tiler_library(&device, GpuTilerMode::Mode27)
            .expect("mode-27 tiler MSL must compile");
        assert!(
            library
                .get_function(tiler_kernel_name(GpuTilerMode::Mode27, TilerKernelKind::Detile), None)
                .is_ok()
        );
        assert!(
            library
                .get_function(tiler_kernel_name(GpuTilerMode::Mode27, TilerKernelKind::Retile), None)
                .is_ok()
        );
    }

    #[test]
    fn mode5_tiler_library_compiles_on_metal() {
        let Some(device) = Device::system_default() else {
            eprintln!("[REMu][TILER] no Metal device; skipping compile proof");
            return;
        };
        let library =
            compile_tiler_library(&device, GpuTilerMode::Mode5).expect("mode-5 tiler MSL must compile");
        assert!(
            library
                .get_function(tiler_kernel_name(GpuTilerMode::Mode5, TilerKernelKind::Detile), None)
                .is_ok()
        );
    }

    #[test]
    fn mode5_msl_within_block_matches_cpu_oracle_for_sampled_coords() {
        // Host oracle for the bit table embedded in MODE5_TILER_MSL. Keep this in lockstep
        // with remu_mode5_within_block above — a mismatch here is a silently-wrong pixel.
        fn msl_table(x: u32, y: u32, log2_bytes: u32) -> u32 {
            match log2_bytes {
                0 => {
                    (x & 0x000f)
                        | ((y & 0x000f) << 4)
                        | ((y & 0x0010) << 4)
                        | ((x & 0x0010) << 5)
                        | ((y & 0x0020) << 5)
                        | ((x & 0x0020) << 6)
                }
                1 => {
                    ((x & 0x0007) << 1)
                        | ((y & 0x0007) << 4)
                        | ((x & 0x0008) << 4)
                        | ((y & 0x0008) << 5)
                        | ((x & 0x0010) << 5)
                        | ((y & 0x0010) << 6)
                        | ((x & 0x0020) << 6)
                }
                2 => {
                    ((x & 0x0003) << 2)
                        | ((y & 0x0007) << 4)
                        | ((x & 0x0004) << 5)
                        | ((y & 0x0008) << 5)
                        | ((x & 0x0008) << 6)
                        | ((y & 0x0010) << 6)
                        | ((x & 0x0010) << 7)
                }
                3 => {
                    ((x & 0x0001) << 3)
                        | ((y & 0x0003) << 4)
                        | ((x & 0x0002) << 5)
                        | ((x & 0x0004) << 5)
                        | ((y & 0x0004) << 6)
                        | ((x & 0x0008) << 6)
                        | ((y & 0x0008) << 7)
                        | ((x & 0x0010) << 7)
                }
                _ => {
                    ((y & 0x0003) << 4)
                        | ((x & 0x0003) << 6)
                        | ((y & 0x0004) << 6)
                        | ((x & 0x0004) << 7)
                        | ((y & 0x0008) << 7)
                        | ((x & 0x0008) << 8)
                }
            }
        }
        use crate::presenter::tiling::mode5_within_block_byte_offset;
        for log2 in 0..5_u32 {
            let bytes = 1_usize << log2;
            for y in 0..32_u32 {
                for x in 0..32_u32 {
                    assert_eq!(
                        usize::try_from(msl_table(x, y, log2)).unwrap(),
                        mode5_within_block_byte_offset(
                            usize::try_from(x).unwrap(),
                            usize::try_from(y).unwrap(),
                            bytes
                        ),
                        "mode-5 MSL/CPU mismatch at ({x},{y}) bpe={bytes}"
                    );
                }
            }
        }
    }
}
