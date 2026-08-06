// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! Within-block address equations shared by the MSL tiler and the differential oracle.
//!
//! Ground truth for the specialised SIMD detile loops:
//! - mode 27 (SW_64KB_R_X): `sdk/.../agc_gpu_address/generated_27.cpp` (e.g. :27-35 XOR chain)
//! - mode 5  (SW_4KB_S):    `sdk/.../agc_gpu_address/generated_5.cpp`  (e.g. :27-34 XOR chain)
//! - mode 9  (SW_64KB_S) / mode 1 (SW_256_S): `presenter/tiling/blocks.rs` mode9/mode1 tables
//! - NPOT mip dims:         `sdk/.../agc_gpu_address/agc_gpu_address.cpp:62-64,529-575,735-737`
//!
//! The CPU tiler (`presenter/tiling/{mode5,mode27,blocks}.rs`) stays the byte oracle; these
//! helpers exist so the MSL port and the host-side differential harness share one expression.

#![allow(dead_code)] // Shared with MSL migration / differential harness.

use super::super::tiling::{
    mode5_within_block_byte_offset, mode9_within_block_byte_offset,
    mode27_within_block_byte_offset_for_format, tiled_block_bytes, tiled_block_dimensions,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuTilerMode {
    Mode1,
    Mode5,
    Mode9,
    Mode27,
}

impl GpuTilerMode {
    /// Selects the MSL library family. Mode-24 depth shares the 64 KB mode-27 library
    /// (`kTileMode` distinguishes the within-block table at dispatch time). Mode-1
    /// (SW_256_S) shares the mode-9 (SW_64KB_S) library the same way.
    pub(crate) fn from_tile_mode(tile_mode: u32) -> Option<Self> {
        match tile_mode {
            1 => Some(Self::Mode1),
            5 => Some(Self::Mode5),
            9 => Some(Self::Mode9),
            24 | 27 => Some(Self::Mode27),
            _ => None,
        }
    }

    pub(crate) fn tile_mode(self) -> u32 {
        match self {
            Self::Mode1 => 1,
            Self::Mode5 => 5,
            Self::Mode9 => 9,
            Self::Mode27 => 27,
        }
    }

    pub(crate) fn block_bytes(self) -> usize {
        match self {
            Self::Mode1 => 256,
            Self::Mode5 => 4_096,
            Self::Mode9 | Self::Mode27 => 65_536,
        }
    }
}

/// SDK `shiftCeil` (`agc_gpu_address.cpp:62-64`).
pub(crate) fn shift_ceil_u32(value: u32, shift: u32) -> Option<u32> {
    if shift >= 32 {
        return None;
    }
    let divisor = 1_u32 << shift;
    value.checked_add(divisor - 1).map(|value| value >> shift)
}

/// Host-side oracle for the MSL mode-27 within-block equation (must match CPU tiler).
pub(crate) fn mode27_within_block_byte_offset_msl_oracle(
    x: usize,
    y: usize,
    bytes_per_element: usize,
) -> usize {
    mode27_within_block_byte_offset_for_format(x, y, bytes_per_element)
}

/// Host-side oracle for the MSL mode-5 within-block equation (must match CPU tiler).
pub(crate) fn mode5_within_block_byte_offset_msl_oracle(
    x: usize,
    y: usize,
    bytes_per_element: usize,
) -> usize {
    mode5_within_block_byte_offset(x, y, bytes_per_element)
}

/// Host-side oracle for the MSL mode-9 within-block equation (must match CPU tiler).
/// Mode-1 (SW_256_S) uses the low eight bits of the same equation.
pub(crate) fn mode9_within_block_byte_offset_msl_oracle(
    x: usize,
    y: usize,
    bytes_per_element: usize,
) -> usize {
    mode9_within_block_byte_offset(x, y, bytes_per_element)
}

/// Physical byte offset of texel `(x, y)` inside a single 2D surface / mip level.
///
/// Mirrors the MSL `remu_tiled_physical` helper so differential tests can assert the host
/// expression without compiling Metal.
pub(crate) fn tiled_physical_byte_offset(
    mode: GpuTilerMode,
    x: usize,
    y: usize,
    pitch: usize,
    bytes_per_element: usize,
    level_offset: usize,
    tail_x: usize,
    tail_y: usize,
    in_tail: bool,
    surface_z_xor: usize,
) -> Result<usize, String> {
    let block_bytes = tiled_block_bytes(mode.tile_mode())
        .ok_or_else(|| format!("unsupported tiler mode {}", mode.tile_mode()))?;
    let (block_width, block_height) = tiled_block_dimensions(block_bytes, bytes_per_element)
        .ok_or("tiler block dimensions unavailable")?;
    let sample_x = if in_tail { tail_x + x } else { x };
    let sample_y = if in_tail { tail_y + y } else { y };
    let within = match mode {
        GpuTilerMode::Mode1 => {
            mode9_within_block_byte_offset_msl_oracle(sample_x, sample_y, bytes_per_element) & 0xff
        }
        GpuTilerMode::Mode5 => {
            mode5_within_block_byte_offset_msl_oracle(sample_x, sample_y, bytes_per_element)
        }
        GpuTilerMode::Mode9 => {
            mode9_within_block_byte_offset_msl_oracle(sample_x, sample_y, bytes_per_element)
        }
        GpuTilerMode::Mode27 => {
            mode27_within_block_byte_offset_msl_oracle(sample_x, sample_y, bytes_per_element)
                ^ surface_z_xor
        }
    };
    if in_tail {
        return level_offset
            .checked_add(within)
            .ok_or_else(|| "tiled physical offset overflows".to_owned());
    }
    let block_x = x / block_width;
    let block_y = y / block_height;
    let blocks_per_row = pitch.div_ceil(block_width);
    let block_index = block_y
        .checked_mul(blocks_per_row)
        .and_then(|value| value.checked_add(block_x))
        .ok_or("tiled block index overflows")?;
    let block_base = block_index
        .checked_mul(block_bytes)
        .ok_or("tiled block base overflows")?;
    level_offset
        .checked_add(block_base)
        .and_then(|value| value.checked_add(within))
        .ok_or_else(|| "tiled physical offset overflows".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presenter::tiling::{
        mode5_within_block_byte_offset, mode9_within_block_byte_offset,
        mode27_within_block_byte_offset_for_format,
    };

    #[test]
    fn sdk_shift_ceil_matches_agc_gpu_address() {
        assert_eq!(shift_ceil_u32(513, 1), Some(257));
        assert_eq!(shift_ceil_u32(512, 1), Some(256));
        assert_eq!(shift_ceil_u32(1, 0), Some(1));
    }

    #[test]
    fn msl_oracles_agree_with_cpu_tiler_across_formats() {
        for bytes in [1_usize, 2, 4, 8, 16] {
            for y in 0..16 {
                for x in 0..16 {
                    assert_eq!(
                        mode27_within_block_byte_offset_msl_oracle(x, y, bytes),
                        mode27_within_block_byte_offset_for_format(x, y, bytes),
                    );
                    assert_eq!(
                        mode9_within_block_byte_offset_msl_oracle(x, y, bytes),
                        mode9_within_block_byte_offset(x, y, bytes),
                    );
                    if bytes <= 16 {
                        assert_eq!(
                            mode5_within_block_byte_offset_msl_oracle(x, y, bytes.min(16)),
                            mode5_within_block_byte_offset(x, y, bytes.min(16)),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn physical_offset_places_mode27_4bpp_texels_inside_the_64kib_block() {
        let offset =
            tiled_physical_byte_offset(GpuTilerMode::Mode27, 3, 5, 128, 4, 0, 0, 0, false, 0)
                .expect("physical offset");
        assert!(offset < 65_536);
        assert_eq!(offset % 4, 0);
    }
}
