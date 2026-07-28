// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::mode27_block_byte_offsets;
use std::sync::OnceLock;

pub(crate) const fn tiled_block_bytes(tile_mode: u32) -> Option<usize> {
    match tile_mode {
        1 => Some(256),
        5 => Some(4_096),
        9 | 24 | 27 => Some(65_536),
        _ => None,
    }
}

pub(crate) fn tiled_block_dimensions(
    block_bytes: usize,
    bytes_per_element: usize,
) -> Option<(usize, usize)> {
    if !bytes_per_element.is_power_of_two()
        || !(1..=16).contains(&bytes_per_element)
        || !block_bytes.is_power_of_two()
        || block_bytes < bytes_per_element
    {
        return None;
    }
    let element_bits = usize::try_from((block_bytes / bytes_per_element).trailing_zeros()).ok()?;
    let width_bits = element_bits.div_ceil(2);
    Some((1 << width_bits, 1 << (element_bits - width_bits)))
}

pub(crate) fn tiled_block_byte_offsets(
    tile_mode: u32,
    block_x: usize,
    block_y: usize,
    bytes_per_element: usize,
    block_width: usize,
    block_height: usize,
) -> Result<&'static [u16], String> {
    match tile_mode {
        1 => Ok(mode1_block_byte_offsets(
            bytes_per_element,
            block_width,
            block_height,
        )),
        5 => Ok(mode5_block_byte_offsets(
            bytes_per_element,
            block_width,
            block_height,
        )),
        9 => Ok(mode9_block_byte_offsets(
            bytes_per_element,
            block_width,
            block_height,
        )),
        24 if bytes_per_element <= 8 => Ok(mode24_block_byte_offsets(
            block_x,
            block_y,
            bytes_per_element,
            block_width,
            block_height,
        )),
        24 => Err(format!(
            "unsupported mode 24 depth element size {bytes_per_element}"
        )),
        27 => Ok(mode27_block_byte_offsets(
            block_x,
            block_y,
            bytes_per_element,
            block_width,
            block_height,
        )),
        _ => Err(format!("unsupported texture tile mode {tile_mode}")),
    }
}

pub(crate) fn mode1_block_byte_offsets(
    bytes_per_element: usize,
    block_width: usize,
    block_height: usize,
) -> &'static [u16] {
    static OFFSETS: OnceLock<[Box<[u16]>; 5]> = OnceLock::new();
    let offsets = OFFSETS.get_or_init(|| {
        std::array::from_fn(|log2_bytes| {
            let bytes_per_element = 1 << log2_bytes;
            let (block_width, block_height) = tiled_block_dimensions(256, bytes_per_element)
                .expect("mode 1 table element size is supported");
            let mut offsets = vec![0_u16; block_width * block_height];
            for local_y in 0..block_height {
                for local_x in 0..block_width {
                    // GFX10 SW_256_S uses the same low eight address equations as
                    // SW_64K_S; its 256-byte block ends before the latter's higher bits.
                    offsets[local_y * block_width + local_x] = u16::try_from(
                        mode9_within_block_byte_offset(local_x, local_y, bytes_per_element) & 0xff,
                    )
                    .expect("mode 1 byte offset fits u16");
                }
            }
            offsets.into_boxed_slice()
        })
    });
    let log2_bytes =
        usize::try_from(bytes_per_element.trailing_zeros()).expect("mode 1 log2 bytes fits usize");
    let table = &offsets[log2_bytes];
    debug_assert_eq!(table.len(), block_width * block_height);
    table
}

pub(crate) fn mode9_block_byte_offsets(
    bytes_per_element: usize,
    block_width: usize,
    block_height: usize,
) -> &'static [u16] {
    static OFFSETS: OnceLock<[Box<[u16]>; 5]> = OnceLock::new();
    let offsets = OFFSETS.get_or_init(|| {
        std::array::from_fn(|log2_bytes| {
            let bytes_per_element = 1 << log2_bytes;
            let (block_width, block_height) = tiled_block_dimensions(65_536, bytes_per_element)
                .expect("mode 9 table element size is supported");
            let mut offsets = vec![0_u16; block_width * block_height];
            for local_y in 0..block_height {
                for local_x in 0..block_width {
                    offsets[local_y * block_width + local_x] = u16::try_from(
                        mode9_within_block_byte_offset(local_x, local_y, bytes_per_element),
                    )
                    .expect("mode 9 byte offset fits u16");
                }
            }
            offsets.into_boxed_slice()
        })
    });
    let log2_bytes =
        usize::try_from(bytes_per_element.trailing_zeros()).expect("mode 9 log2 bytes fits usize");
    let table = &offsets[log2_bytes];
    debug_assert_eq!(table.len(), block_width * block_height);
    table
}

pub(crate) fn mode9_within_block_byte_offset(
    x: usize,
    y: usize,
    bytes_per_element: usize,
) -> usize {
    let bit = |value: usize, index: usize| (value >> index) & 1;
    let log2_bytes =
        usize::try_from(bytes_per_element.trailing_zeros()).expect("mode 9 log2 bytes fits usize");
    // ADDR_SW_64KB_S is the standard, non-XOR GFX10 64 KiB layout. These are the
    // exact address bits assembled by GFX10_SW_64K_S_PATINFO for one through
    // sixteen byte elements. Low constant bits select a byte within the element;
    // the remaining bits interleave X and Y according to the hardware equation.
    let address_bits = match log2_bytes {
        0 => [
            bit(x, 0),
            bit(x, 1),
            bit(x, 2),
            bit(x, 3),
            bit(y, 0),
            bit(y, 1),
            bit(y, 2),
            bit(y, 3),
            bit(y, 4),
            bit(x, 4),
            bit(y, 5),
            bit(x, 5),
            bit(y, 6),
            bit(x, 6),
            bit(y, 7),
            bit(x, 7),
        ],
        1 => [
            0,
            bit(x, 0),
            bit(x, 1),
            bit(x, 2),
            bit(y, 0),
            bit(y, 1),
            bit(y, 2),
            bit(x, 3),
            bit(y, 3),
            bit(x, 4),
            bit(y, 4),
            bit(x, 5),
            bit(y, 5),
            bit(x, 6),
            bit(y, 6),
            bit(x, 7),
        ],
        2 => [
            0,
            0,
            bit(x, 0),
            bit(x, 1),
            bit(y, 0),
            bit(y, 1),
            bit(y, 2),
            bit(x, 2),
            bit(y, 3),
            bit(x, 3),
            bit(y, 4),
            bit(x, 4),
            bit(y, 5),
            bit(x, 5),
            bit(y, 6),
            bit(x, 6),
        ],
        3 => [
            0,
            0,
            0,
            bit(x, 0),
            bit(y, 0),
            bit(y, 1),
            bit(x, 1),
            bit(x, 2),
            bit(y, 2),
            bit(x, 3),
            bit(y, 3),
            bit(x, 4),
            bit(y, 4),
            bit(x, 5),
            bit(y, 5),
            bit(x, 6),
        ],
        4 => [
            0,
            0,
            0,
            0,
            bit(y, 0),
            bit(y, 1),
            bit(x, 0),
            bit(x, 1),
            bit(y, 2),
            bit(x, 2),
            bit(y, 3),
            bit(x, 3),
            bit(y, 4),
            bit(x, 4),
            bit(y, 5),
            bit(x, 5),
        ],
        _ => unreachable!("mode 9 element size was validated"),
    };
    address_bits
        .iter()
        .enumerate()
        .fold(0, |offset, (index, &value)| offset | (value << index))
}

pub(crate) fn mode24_block_byte_offsets(
    block_x: usize,
    block_y: usize,
    bytes_per_element: usize,
    block_width: usize,
    block_height: usize,
) -> &'static [u16] {
    static OFFSETS: OnceLock<[Box<[u16]>; 4]> = OnceLock::new();
    let offsets = OFFSETS.get_or_init(|| {
        std::array::from_fn(|log2_bytes| {
            let bytes_per_element = 1 << log2_bytes;
            let (block_width, block_height) = tiled_block_dimensions(65_536, bytes_per_element)
                .expect("mode 24 table element size is supported");
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
                                u16::try_from(mode24_within_block_byte_offset_for_format(
                                    global_x,
                                    global_y,
                                    bytes_per_element,
                                ))
                                .expect("mode 24 byte offset fits u16");
                        }
                    }
                }
            }
            offsets.into_boxed_slice()
        })
    });
    let log2_bytes =
        usize::try_from(bytes_per_element.trailing_zeros()).expect("mode 24 log2 bytes fits usize");
    let elements_per_block = block_width * block_height;
    let table = ((block_y & 1) * 2 + (block_x & 1)) * elements_per_block;
    &offsets[log2_bytes][table..table + elements_per_block]
}

pub(crate) fn mode24_within_block_byte_offset_for_format(
    x: usize,
    y: usize,
    bytes_per_element: usize,
) -> usize {
    let log2_bytes =
        usize::try_from(bytes_per_element.trailing_zeros()).expect("mode 24 log2 bytes fits usize");
    // GFX10 64KB_Z_X has a distinct complete address equation; only replacing the
    // low byte of the render-target equation corrupts the upper nibble ordering.
    // Keep this in sync with KytyPS5's current Depth64KB*OffsetBytes reference.
    match log2_bytes {
        0 => {
            (x & 0x0001)
                ^ ((x << 1) & 0x0004)
                ^ ((x << 2) & 0x0010)
                ^ ((x << 3) & 0x0040)
                ^ ((x << 5) & 0x0300)
                ^ ((x << 4) & 0x0400)
                ^ ((x << 6) & 0x0800)
                ^ ((x << 7) & 0x2000)
                ^ ((x << 8) & 0x8000)
                ^ ((y << 1) & 0x0002)
                ^ ((y << 2) & 0x0008)
                ^ ((y << 3) & 0x00a0)
                ^ ((y << 5) & 0x0f00)
                ^ ((y << 6) & 0x1000)
                ^ ((y << 7) & 0x4000)
        }
        1 => {
            ((x << 1) & 0x0002)
                ^ ((x << 2) & 0x0008)
                ^ ((x << 3) & 0x0020)
                ^ ((x << 4) & 0x0480)
                ^ ((x << 5) & 0x0300)
                ^ ((x << 6) & 0x0800)
                ^ ((x << 7) & 0x2000)
                ^ ((x << 8) & 0x8000)
                ^ ((y << 2) & 0x0004)
                ^ ((y << 3) & 0x0010)
                ^ ((y << 4) & 0x0040)
                ^ ((y << 5) & 0x0f00)
                ^ ((y << 8) & 0x5000)
        }
        2 => {
            ((x << 2) & 0x0004)
                ^ ((x << 3) & 0x0010)
                ^ ((x << 4) & 0x0440)
                ^ ((x << 5) & 0x0300)
                ^ ((x << 6) & 0x0800)
                ^ ((x << 9) & 0xa000)
                ^ ((y << 3) & 0x0008)
                ^ ((y << 4) & 0x0020)
                ^ ((y << 5) & 0x0f80)
                ^ ((y << 9) & 0x1000)
                ^ ((y << 8) & 0x4000)
        }
        3 => {
            ((x << 3) & 0x0008)
                ^ ((x << 4) & 0x0420)
                ^ ((x << 5) & 0x0380)
                ^ ((x << 6) & 0x0800)
                ^ ((x << 10) & 0x2000)
                ^ ((x << 9) & 0x8000)
                ^ ((y << 4) & 0x0010)
                ^ ((y << 5) & 0x0f40)
                ^ ((y << 10) & 0x5000)
        }
        // KytyPS5 and the GFX10 depth layout support at most 64-bit elements.
        _ => unreachable!("mode 24 element size was validated"),
    }
}

pub(crate) fn mode5_block_byte_offsets(
    bytes_per_element: usize,
    block_width: usize,
    block_height: usize,
) -> &'static [u16] {
    static OFFSETS: OnceLock<[Box<[u16]>; 5]> = OnceLock::new();
    let offsets = OFFSETS.get_or_init(|| {
        std::array::from_fn(|log2_bytes| {
            let bytes_per_element = 1 << log2_bytes;
            let (block_width, block_height) = tiled_block_dimensions(4_096, bytes_per_element)
                .expect("mode 5 table element size is supported");
            let mut offsets = vec![0_u16; block_width * block_height];
            for local_y in 0..block_height {
                for local_x in 0..block_width {
                    offsets[local_y * block_width + local_x] = u16::try_from(
                        mode5_within_block_byte_offset(local_x, local_y, bytes_per_element),
                    )
                    .expect("mode 5 byte offset fits u16");
                }
            }
            offsets.into_boxed_slice()
        })
    });
    let log2_bytes =
        usize::try_from(bytes_per_element.trailing_zeros()).expect("mode 5 log2 bytes fits usize");
    let table = &offsets[log2_bytes];
    debug_assert_eq!(table.len(), block_width * block_height);
    table
}

pub(crate) fn mode5_within_block_byte_offset(
    x: usize,
    y: usize,
    bytes_per_element: usize,
) -> usize {
    let bit = |value: usize, index: usize| (value >> index) & 1;
    let log2_bytes =
        usize::try_from(bytes_per_element.trailing_zeros()).expect("mode 5 log2 bytes fits usize");
    let address_bits = match log2_bytes {
        0 => [
            bit(x, 0),
            bit(x, 1),
            bit(x, 2),
            bit(x, 3),
            bit(y, 0),
            bit(y, 1),
            bit(y, 2),
            bit(y, 3),
            bit(y, 4),
            bit(x, 4),
            bit(y, 5),
            bit(x, 5),
        ],
        1 => [
            0,
            bit(x, 0),
            bit(x, 1),
            bit(x, 2),
            bit(y, 0),
            bit(y, 1),
            bit(y, 2),
            bit(x, 3),
            bit(y, 3),
            bit(x, 4),
            bit(y, 4),
            bit(x, 5),
        ],
        2 => [
            0,
            0,
            bit(x, 0),
            bit(x, 1),
            bit(y, 0),
            bit(y, 1),
            bit(y, 2),
            bit(x, 2),
            bit(y, 3),
            bit(x, 3),
            bit(y, 4),
            bit(x, 4),
        ],
        3 => [
            0,
            0,
            0,
            bit(x, 0),
            bit(y, 0),
            bit(y, 1),
            bit(x, 1),
            bit(x, 2),
            bit(y, 2),
            bit(x, 3),
            bit(y, 3),
            bit(x, 4),
        ],
        4 => [
            0,
            0,
            0,
            0,
            bit(y, 0),
            bit(y, 1),
            bit(x, 0),
            bit(x, 1),
            bit(y, 2),
            bit(x, 2),
            bit(y, 3),
            bit(x, 3),
        ],
        _ => unreachable!("mode 5 element size was validated"),
    };
    address_bits
        .iter()
        .enumerate()
        .fold(0, |offset, (index, &value)| offset | (value << index))
}
