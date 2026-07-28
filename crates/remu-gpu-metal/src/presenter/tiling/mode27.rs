// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    GuestMipChainLayout, GuestMipLevelLayout, GuestTextureDescriptor, MODE27_DECODE_BYTES,
    MODE27_DECODE_COUNT, MODE27_DECODE_NANOSECONDS, record_texture_work, tiled_block_dimensions,
};
use std::sync::OnceLock;
use std::time::Instant;

pub(crate) const MODE27_MIP_TAIL_LOCATIONS: [[(usize, usize); 12]; 5] = [
    [
        (128, 0),
        (0, 128),
        (64, 0),
        (0, 64),
        (32, 0),
        (16, 32),
        (0, 48),
        (0, 32),
        (16, 16),
        (16, 0),
        (0, 16),
        (0, 0),
    ],
    [
        (128, 0),
        (0, 64),
        (64, 0),
        (0, 32),
        (32, 0),
        (16, 16),
        (0, 24),
        (0, 16),
        (16, 8),
        (16, 0),
        (0, 8),
        (0, 0),
    ],
    [
        (64, 0),
        (0, 64),
        (32, 0),
        (0, 32),
        (16, 0),
        (8, 16),
        (0, 24),
        (0, 16),
        (8, 8),
        (8, 0),
        (0, 8),
        (0, 0),
    ],
    [
        (64, 0),
        (0, 32),
        (32, 0),
        (0, 16),
        (16, 0),
        (8, 8),
        (0, 12),
        (0, 8),
        (8, 4),
        (8, 0),
        (0, 4),
        (0, 0),
    ],
    [
        (32, 0),
        (0, 32),
        (16, 0),
        (0, 16),
        (8, 0),
        (4, 8),
        (0, 12),
        (0, 8),
        (4, 4),
        (4, 0),
        (0, 4),
        (0, 0),
    ],
];

pub(crate) fn shift_ceil(value: usize, shift: u32) -> Option<usize> {
    let divisor = 1_usize.checked_shl(shift)?;
    value.checked_add(divisor - 1).map(|value| value / divisor)
}

pub(crate) fn mode27_mip_chain_layout(
    descriptor: GuestTextureDescriptor,
    mip_level_count: u32,
) -> Result<GuestMipChainLayout, String> {
    if descriptor.tile_mode != 27 || mip_level_count == 0 || mip_level_count > 16 {
        return Err(format!(
            "mode-27 mip layout requires 1..=16 levels, got tile={} levels={mip_level_count}",
            descriptor.tile_mode
        ));
    }
    let bytes_per_element = usize::try_from(descriptor.bytes_per_pixel)
        .map_err(|_| "mip element byte count does not fit usize")?;
    let bytes_log2 = usize::try_from(bytes_per_element.trailing_zeros())
        .map_err(|_| "mip element byte logarithm does not fit usize")?;
    if !bytes_per_element.is_power_of_two() || bytes_log2 >= MODE27_MIP_TAIL_LOCATIONS.len() {
        return Err(format!(
            "mode-27 mip layout has unsupported {}-byte elements",
            descriptor.bytes_per_pixel
        ));
    }
    let (block_width, block_height) = tiled_block_dimensions(65_536, bytes_per_element)
        .ok_or("mode-27 mip layout requires a 1, 2, 4, 8, or 16-byte element format")?;
    let width = usize::try_from(descriptor.width).map_err(|_| "mip width does not fit usize")?;
    let height = usize::try_from(descriptor.height).map_err(|_| "mip height does not fit usize")?;
    let level_count =
        usize::try_from(mip_level_count).map_err(|_| "mip count does not fit usize")?;
    let tail_width = block_width / 2;
    let mut first_tail = level_count;
    if level_count > 1 {
        for level in 0..level_count {
            let shift = u32::try_from(level).expect("level count is bounded");
            if shift_ceil(width, shift).is_some_and(|width| width <= tail_width)
                && shift_ceil(height, shift).is_some_and(|height| height <= block_height)
                && level_count - level <= 12
            {
                first_tail = level;
                break;
            }
        }
    }
    let empty = GuestMipLevelLayout {
        width: 0,
        height: 0,
        padded_width: 0,
        offset: 0,
        tail_x: 0,
        tail_y: 0,
        in_tail: false,
    };
    let mut levels = vec![empty; level_count];
    let mut offset = if first_tail < level_count {
        65_536_usize
    } else {
        0
    };
    for level in (0..first_tail).rev() {
        let shift = u32::try_from(level).expect("level count is bounded");
        let mip_width = shift_ceil(width, shift)
            .ok_or("mip width overflows")?
            .max(1);
        let mip_height = shift_ceil(height, shift)
            .ok_or("mip height overflows")?
            .max(1);
        let padded_width = mip_width
            .div_ceil(block_width)
            .checked_mul(block_width)
            .ok_or("mip padded width overflows")?;
        let padded_height = mip_height
            .div_ceil(block_height)
            .checked_mul(block_height)
            .ok_or("mip padded height overflows")?;
        let size = padded_width
            .checked_mul(padded_height)
            .and_then(|elements| elements.checked_mul(bytes_per_element))
            .ok_or("mip storage size overflows")?;
        levels[level] = GuestMipLevelLayout {
            width: (width >> shift).max(1),
            height: (height >> shift).max(1),
            padded_width,
            offset,
            ..empty
        };
        offset = offset.checked_add(size).ok_or("mip chain size overflows")?;
    }
    for level in first_tail..level_count {
        let shift = u32::try_from(level).expect("level count is bounded");
        let (tail_x, tail_y) = MODE27_MIP_TAIL_LOCATIONS[bytes_log2][level - first_tail];
        let mip_width = (width >> shift).max(1);
        let mip_height = (height >> shift).max(1);
        if tail_x + mip_width > block_width || tail_y + mip_height > block_height {
            return Err(format!(
                "mode-27 mip {level} tail placement ({tail_x},{tail_y}) {mip_width}x{mip_height} exceeds {block_width}x{block_height}"
            ));
        }
        levels[level] = GuestMipLevelLayout {
            width: mip_width,
            height: mip_height,
            padded_width: block_width,
            offset: 0,
            tail_x,
            tail_y,
            in_tail: true,
        };
    }
    Ok(GuestMipChainLayout {
        levels,
        byte_count: offset,
        block_width,
        block_height,
        bytes_per_element,
    })
}

pub(crate) fn copy_mode27_mip_level_to_linear(
    layout: &GuestMipChainLayout,
    level: GuestMipLevelLayout,
    raw: &[u8],
    linear: &mut [u8],
) -> Result<(), String> {
    let expected = level
        .width
        .checked_mul(level.height)
        .and_then(|elements| elements.checked_mul(layout.bytes_per_element))
        .ok_or("mip linear size overflows")?;
    if linear.len() != expected {
        return Err(format!(
            "mode-27 mip destination has {} bytes, expected {expected}",
            linear.len()
        ));
    }
    if level.in_tail {
        let offsets = mode27_block_byte_offsets(
            0,
            0,
            layout.bytes_per_element,
            layout.block_width,
            layout.block_height,
        );
        for y in 0..level.height {
            let destination_row = y * level.width * layout.bytes_per_element;
            let offset_row = (level.tail_y + y) * layout.block_width + level.tail_x;
            for x in 0..level.width {
                let source = level
                    .offset
                    .checked_add(usize::from(offsets[offset_row + x]))
                    .ok_or("mip-tail source offset overflows")?;
                let destination = destination_row + x * layout.bytes_per_element;
                linear[destination..destination + layout.bytes_per_element].copy_from_slice(
                    raw.get(source..source + layout.bytes_per_element)
                        .ok_or("mode-27 mip-tail texel exceeds backing")?,
                );
            }
        }
    } else {
        let blocks_per_row = level.padded_width / layout.block_width;
        let visible_blocks_per_row = level.width.div_ceil(layout.block_width);
        for block_y in 0..level.height.div_ceil(layout.block_height) {
            let rows = (level.height - block_y * layout.block_height).min(layout.block_height);
            for block_x in 0..visible_blocks_per_row {
                let columns = (level.width - block_x * layout.block_width).min(layout.block_width);
                let block_base = (block_y * blocks_per_row + block_x)
                    .checked_mul(65_536)
                    .and_then(|offset| level.offset.checked_add(offset))
                    .ok_or("mip block offset overflows")?;
                let offsets = mode27_block_byte_offsets(
                    block_x,
                    block_y,
                    layout.bytes_per_element,
                    layout.block_width,
                    layout.block_height,
                );
                for local_y in 0..rows {
                    let destination_row = ((block_y * layout.block_height + local_y) * level.width
                        + block_x * layout.block_width)
                        * layout.bytes_per_element;
                    let offset_row = local_y * layout.block_width;
                    for local_x in 0..columns {
                        let source = block_base + usize::from(offsets[offset_row + local_x]);
                        let destination = destination_row + local_x * layout.bytes_per_element;
                        linear[destination..destination + layout.bytes_per_element]
                            .copy_from_slice(
                                raw.get(source..source + layout.bytes_per_element)
                                    .ok_or("mode-27 mip texel exceeds backing")?,
                            );
                    }
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn copy_linear_to_mode27_mip_level(
    layout: &GuestMipChainLayout,
    level: GuestMipLevelLayout,
    linear: &[u8],
    raw: &mut [u8],
) -> Result<(), String> {
    let expected = level
        .width
        .checked_mul(level.height)
        .and_then(|elements| elements.checked_mul(layout.bytes_per_element))
        .ok_or("mip linear size overflows")?;
    if linear.len() != expected {
        return Err(format!(
            "mode-27 mip source has {} bytes, expected {expected}",
            linear.len()
        ));
    }
    if level.in_tail {
        let offsets = mode27_block_byte_offsets(
            0,
            0,
            layout.bytes_per_element,
            layout.block_width,
            layout.block_height,
        );
        for y in 0..level.height {
            let source_row = y * level.width * layout.bytes_per_element;
            let offset_row = (level.tail_y + y) * layout.block_width + level.tail_x;
            for x in 0..level.width {
                let source = source_row + x * layout.bytes_per_element;
                let destination = level
                    .offset
                    .checked_add(usize::from(offsets[offset_row + x]))
                    .ok_or("mip-tail destination offset overflows")?;
                raw.get_mut(destination..destination + layout.bytes_per_element)
                    .ok_or("mode-27 mip-tail texel exceeds backing")?
                    .copy_from_slice(&linear[source..source + layout.bytes_per_element]);
            }
        }
    } else {
        let blocks_per_row = level.padded_width / layout.block_width;
        let visible_blocks_per_row = level.width.div_ceil(layout.block_width);
        for block_y in 0..level.height.div_ceil(layout.block_height) {
            let rows = (level.height - block_y * layout.block_height).min(layout.block_height);
            for block_x in 0..visible_blocks_per_row {
                let columns = (level.width - block_x * layout.block_width).min(layout.block_width);
                let block_base = (block_y * blocks_per_row + block_x)
                    .checked_mul(65_536)
                    .and_then(|offset| level.offset.checked_add(offset))
                    .ok_or("mip block offset overflows")?;
                let offsets = mode27_block_byte_offsets(
                    block_x,
                    block_y,
                    layout.bytes_per_element,
                    layout.block_width,
                    layout.block_height,
                );
                for local_y in 0..rows {
                    let source_row = ((block_y * layout.block_height + local_y) * level.width
                        + block_x * layout.block_width)
                        * layout.bytes_per_element;
                    let offset_row = local_y * layout.block_width;
                    for local_x in 0..columns {
                        let source = source_row + local_x * layout.bytes_per_element;
                        let destination = block_base + usize::from(offsets[offset_row + local_x]);
                        raw.get_mut(destination..destination + layout.bytes_per_element)
                            .ok_or("mode-27 mip texel exceeds backing")?
                            .copy_from_slice(&linear[source..source + layout.bytes_per_element]);
                    }
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn decode_mode27_mip_chain_from_raw(
    descriptor: GuestTextureDescriptor,
    mip_level_count: u32,
    raw: &[u8],
) -> Result<(GuestMipChainLayout, Vec<Vec<u8>>), String> {
    let layout = mode27_mip_chain_layout(descriptor, mip_level_count)?;
    if raw.len() != layout.byte_count {
        return Err(format!(
            "mode-27 mip source has {} bytes, expected {}",
            raw.len(),
            layout.byte_count
        ));
    }
    let started = Instant::now();
    let mut decoded = Vec::with_capacity(layout.levels.len());
    for &level in &layout.levels {
        let mut linear = vec![
            0_u8;
            level
                .width
                .checked_mul(level.height)
                .and_then(|elements| elements.checked_mul(layout.bytes_per_element))
                .ok_or("mip linear size overflows")?
        ];
        copy_mode27_mip_level_to_linear(&layout, level, raw, &mut linear)?;
        decoded.push(linear);
    }
    record_texture_work(
        &MODE27_DECODE_COUNT,
        &MODE27_DECODE_BYTES,
        &MODE27_DECODE_NANOSECONDS,
        raw.len(),
        started.elapsed(),
    );
    Ok((layout, decoded))
}

pub(crate) fn encode_mode27_mip_chain_into_raw(
    layout: &GuestMipChainLayout,
    linear_levels: &[Vec<u8>],
    raw: &mut [u8],
) -> Result<(), String> {
    if raw.len() != layout.byte_count || linear_levels.len() != layout.levels.len() {
        return Err("mode-27 mip writeback layout mismatch".to_owned());
    }
    for (&level, linear) in layout.levels.iter().zip(linear_levels) {
        copy_linear_to_mode27_mip_level(layout, level, linear, raw)?;
    }
    Ok(())
}

pub(crate) fn mode27_block_byte_offsets(
    block_x: usize,
    block_y: usize,
    bytes_per_pixel: usize,
    block_width: usize,
    block_height: usize,
) -> &'static [u16] {
    static OFFSETS: OnceLock<[Box<[u16]>; 5]> = OnceLock::new();
    let offsets = OFFSETS.get_or_init(|| {
        std::array::from_fn(|log2_bytes| {
            let bytes_per_pixel = 1 << log2_bytes;
            let (block_width, block_height) = mode27_block_dimensions(bytes_per_pixel)
                .expect("mode 27 table bytes per pixel is supported");
            let elements_per_block = block_width * block_height;
            let mut offsets = vec![0_u16; 4 * elements_per_block];
            for block_y_parity in 0..2 {
                for block_x_parity in 0..2 {
                    let table = (block_y_parity * 2 + block_x_parity) * elements_per_block;
                    for local_y in 0..block_height {
                        for local_x in 0..block_width {
                            let global_x = local_x | (block_x_parity * block_width);
                            let global_y = local_y | (block_y_parity * block_height);
                            offsets[table + local_y * block_width + local_x] =
                                u16::try_from(mode27_within_block_byte_offset_for_format(
                                    global_x,
                                    global_y,
                                    bytes_per_pixel,
                                ))
                                .expect("mode 27 byte offset fits u16");
                        }
                    }
                }
            }
            offsets.into_boxed_slice()
        })
    });
    let log2_bytes =
        usize::try_from(bytes_per_pixel.trailing_zeros()).expect("mode 27 log2 bytes fits usize");
    let elements_per_block = block_width * block_height;
    let table = ((block_y & 1) * 2 + (block_x & 1)) * elements_per_block;
    &offsets[log2_bytes][table..table + elements_per_block]
}

pub(crate) fn mode27_block_dimensions(bytes_per_pixel: usize) -> Option<(usize, usize)> {
    tiled_block_dimensions(65_536, bytes_per_pixel)
}

pub(crate) const fn mode27_surface_z_xor(surface_z: usize) -> usize {
    // GFX10 RenderTarget64KB folds the low four surface-Z bits into address bits 11..8.
    // This applies to each depth/array slice independently even though the block itself is thin.
    ((surface_z & 8) << 5)
        ^ ((surface_z & 4) << 7)
        ^ ((surface_z & 2) << 9)
        ^ ((surface_z & 1) << 11)
}

pub(crate) fn mode27_within_block_byte_offset_for_format(
    x: usize,
    y: usize,
    bytes_per_pixel: usize,
) -> usize {
    let log2_bytes =
        usize::try_from(bytes_per_pixel.trailing_zeros()).expect("mode 27 log2 bytes fits usize");
    // Exact GFX10 64KB_R_X equation used by current KytyPS5. The previous
    // PATINFO-derived bit table was a different pipe configuration and produced
    // the striped/barcode render-target uploads visible in Silent Hill.
    match log2_bytes {
        0 => {
            ((y << 2) & 0x0008)
                ^ ((y << 4) & 0x0010)
                ^ ((y << 3) & 0x00a0)
                ^ ((y << 5) & 0x0f00)
                ^ ((y << 6) & 0x1000)
                ^ ((y << 7) & 0x4000)
                ^ (x & 0x0007)
                ^ ((x << 3) & 0x0040)
                ^ ((x << 5) & 0x0300)
                ^ ((x << 4) & 0x0400)
                ^ ((x << 6) & 0x0800)
                ^ ((x << 7) & 0x2000)
                ^ ((x << 8) & 0x8000)
        }
        1 => {
            ((y << 4) & 0x0070)
                ^ ((y << 5) & 0x0f00)
                ^ ((y << 8) & 0x5000)
                ^ ((x << 1) & 0x000e)
                ^ ((x << 4) & 0x0480)
                ^ ((x << 5) & 0x0300)
                ^ ((x << 6) & 0x0800)
                ^ ((x << 7) & 0x2000)
                ^ ((x << 8) & 0x8000)
        }
        2 => {
            ((y << 4) & 0x0070)
                ^ ((y << 5) & 0x0f00)
                ^ ((y << 9) & 0x1000)
                ^ ((y << 8) & 0x4000)
                ^ ((x << 2) & 0x000c)
                ^ ((x << 5) & 0x0380)
                ^ ((x << 4) & 0x0400)
                ^ ((x << 6) & 0x0800)
                ^ ((x << 9) & 0xa000)
        }
        3 => {
            ((y << 4) & 0x0010)
                ^ ((y << 6) & 0x0080)
                ^ ((y << 5) & 0x0f00)
                ^ ((y << 10) & 0x5000)
                ^ ((x << 3) & 0x0008)
                ^ ((x << 4) & 0x0460)
                ^ ((x << 5) & 0x0300)
                ^ ((x << 6) & 0x0800)
                ^ ((x << 10) & 0x2000)
                ^ ((x << 9) & 0x8000)
        }
        4 => {
            ((x << 4) & 0x0410)
                ^ ((x << 5) & 0x0340)
                ^ ((x << 6) & 0x0800)
                ^ ((x << 11) & 0xa000)
                ^ ((y << 5) & 0x0f20)
                ^ ((y << 6) & 0x0080)
                ^ ((y << 10) & 0x1000)
                ^ ((y << 11) & 0x4000)
        }
        _ => unreachable!("mode 27 bytes per pixel was validated"),
    }
}

#[cfg(test)]
pub(crate) fn mode27_source_byte_offset(x: usize, y: usize, width: usize) -> usize {
    let blocks_per_row = width.div_ceil(128);
    let block_x = x / 128;
    let block_y = y / 128;
    let block_index = block_y * blocks_per_row + block_x;
    let offsets = mode27_block_byte_offsets(block_x, block_y, 4, 128, 128);
    block_index * 65_536 + usize::from(offsets[(y & 127) * 128 + (x & 127)])
}

#[cfg(test)]
pub(crate) fn mode27_within_block_byte_offset(x: usize, y: usize) -> usize {
    mode27_within_block_byte_offset_for_format(x, y, 4)
}
