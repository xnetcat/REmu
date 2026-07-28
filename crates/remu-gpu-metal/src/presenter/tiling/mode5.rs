// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::blocks::mode5_block_byte_offsets;
use super::{
    GuestMipChainLayout, GuestMipLevelLayout, GuestTextureDescriptor, texture_element_layout,
    tiled_block_dimensions,
};

fn shift_ceil(value: usize, shift: u32) -> Option<usize> {
    let divisor = 1_usize.checked_shl(shift)?;
    value.checked_add(divisor - 1).map(|value| value / divisor)
}

const MODE5_MAX_MIPS_IN_TAIL: usize = 8;
const BLOCK_256_DIMENSIONS: [(usize, usize); 5] = [(16, 16), (16, 8), (8, 8), (8, 4), (4, 4)];

fn mode5_mip_tail_location(tail_level: usize, bytes_log2: usize) -> Result<(usize, usize), String> {
    if tail_level >= MODE5_MAX_MIPS_IN_TAIL || bytes_log2 >= 5 {
        return Err(format!(
            "invalid mode-5 mip-tail level/element logarithm {tail_level}/{bytes_log2}"
        ));
    }

    // This is AMD AddrLib's GFX10 thin-resource mip-tail placement, expanded from
    // Gfx10Lib::ComputeSurfaceInfoMacroTiled. SW_4KB_S has a 12-bit block size,
    // so its tail holds eight levels and does not take the odd-block X/Y swap.
    let m = MODE5_MAX_MIPS_IN_TAIL - 1 - tail_level;
    let mip_offset = if m > 6 { 16 << m } else { m << 8 };
    let mip_x = ((mip_offset >> 9) & 1)
        | ((mip_offset >> 10) & 2)
        | ((mip_offset >> 11) & 4)
        | ((mip_offset >> 12) & 8)
        | ((mip_offset >> 13) & 16)
        | ((mip_offset >> 14) & 32);
    let mip_y = ((mip_offset >> 8) & 1)
        | ((mip_offset >> 9) & 2)
        | ((mip_offset >> 10) & 4)
        | ((mip_offset >> 11) & 8)
        | ((mip_offset >> 12) & 16)
        | ((mip_offset >> 13) & 32);
    let (micro_width, micro_height) = BLOCK_256_DIMENSIONS[bytes_log2];
    Ok((mip_x * micro_width, mip_y * micro_height))
}

#[allow(clippy::too_many_lines)]
pub(crate) fn mode5_mip_chain_layout(
    descriptor: GuestTextureDescriptor,
    mip_level_count: u32,
) -> Result<GuestMipChainLayout, String> {
    if descriptor.tile_mode != 5 || mip_level_count == 0 || mip_level_count > 16 {
        return Err(format!(
            "mode-5 mip layout requires 1..=16 levels, got tile={} levels={mip_level_count}",
            descriptor.tile_mode
        ));
    }
    let (width, height, _, bytes_per_element) = texture_element_layout(descriptor)?;
    let bytes_log2 = usize::try_from(bytes_per_element.trailing_zeros())
        .map_err(|_| "mip element byte logarithm does not fit usize")?;
    if !bytes_per_element.is_power_of_two() || bytes_log2 >= 5 {
        return Err(format!(
            "mode-5 mip layout has unsupported {}-byte elements",
            descriptor.bytes_per_pixel
        ));
    }
    let (block_width, block_height) = tiled_block_dimensions(4_096, bytes_per_element)
        .ok_or("mode-5 mip layout requires a 1, 2, 4, 8, or 16-byte element format")?;
    let level_count =
        usize::try_from(mip_level_count).map_err(|_| "mip count does not fit usize")?;
    let tail_width = block_width / 2;
    let mut first_tail = level_count;
    if level_count > 1 {
        for level in 0..level_count {
            let shift = u32::try_from(level).expect("level count is bounded");
            if shift_ceil(width, shift).is_some_and(|width| width <= tail_width)
                && shift_ceil(height, shift).is_some_and(|height| height <= block_height)
                && level_count - level <= MODE5_MAX_MIPS_IN_TAIL
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
        4_096_usize
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
            width: mip_width,
            height: mip_height,
            padded_width,
            offset,
            ..empty
        };
        offset = offset.checked_add(size).ok_or("mip chain size overflows")?;
    }
    for (level, output) in levels.iter_mut().enumerate().skip(first_tail) {
        let shift = u32::try_from(level).expect("level count is bounded");
        let (tail_x, tail_y) = mode5_mip_tail_location(level - first_tail, bytes_log2)?;
        let mip_width = shift_ceil(width, shift)
            .ok_or("mip width overflows")?
            .max(1);
        let mip_height = shift_ceil(height, shift)
            .ok_or("mip height overflows")?
            .max(1);
        if tail_x + mip_width > block_width || tail_y + mip_height > block_height {
            return Err(format!(
                "mode-5 mip {level} tail placement ({tail_x},{tail_y}) {mip_width}x{mip_height} exceeds {block_width}x{block_height}"
            ));
        }
        *output = GuestMipLevelLayout {
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

fn copy_mode5_mip_level_to_linear(
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
            "mode-5 mip destination has {} bytes, expected {expected}",
            linear.len()
        ));
    }
    if level.in_tail {
        let offsets = mode5_block_byte_offsets(
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
                        .ok_or("mode-5 mip-tail texel exceeds backing")?,
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
                    .checked_mul(4_096)
                    .and_then(|offset| level.offset.checked_add(offset))
                    .ok_or("mip block offset overflows")?;
                let offsets = mode5_block_byte_offsets(
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
                                    .ok_or("mode-5 mip texel exceeds backing")?,
                            );
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn copy_linear_to_mode5_mip_level(
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
            "mode-5 mip source has {} bytes, expected {expected}",
            linear.len()
        ));
    }
    if level.in_tail {
        let offsets = mode5_block_byte_offsets(
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
                    .ok_or("mode-5 mip-tail texel exceeds backing")?
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
                    .checked_mul(4_096)
                    .and_then(|offset| level.offset.checked_add(offset))
                    .ok_or("mip block offset overflows")?;
                let offsets = mode5_block_byte_offsets(
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
                            .ok_or("mode-5 mip texel exceeds backing")?
                            .copy_from_slice(&linear[source..source + layout.bytes_per_element]);
                    }
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn decode_mode5_mip_chain_from_raw(
    descriptor: GuestTextureDescriptor,
    mip_level_count: u32,
    raw: &[u8],
) -> Result<(GuestMipChainLayout, Vec<Vec<u8>>), String> {
    let layout = mode5_mip_chain_layout(descriptor, mip_level_count)?;
    if raw.len() != layout.byte_count {
        return Err(format!(
            "mode-5 mip source has {} bytes, expected {}",
            raw.len(),
            layout.byte_count
        ));
    }
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
        copy_mode5_mip_level_to_linear(&layout, level, raw, &mut linear)?;
        decoded.push(linear);
    }
    Ok((layout, decoded))
}

#[cfg(test)]
pub(crate) fn encode_mode5_mip_chain_into_raw(
    layout: &GuestMipChainLayout,
    linear_levels: &[Vec<u8>],
    raw: &mut [u8],
) -> Result<(), String> {
    if raw.len() != layout.byte_count || linear_levels.len() != layout.levels.len() {
        return Err("mode-5 mip writeback layout mismatch".to_owned());
    }
    for (&level, linear) in layout.levels.iter().zip(linear_levels) {
        copy_linear_to_mode5_mip_level(layout, level, linear, raw)?;
    }
    Ok(())
}
