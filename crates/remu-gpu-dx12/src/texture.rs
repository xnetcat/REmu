// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! GFX10 image descriptor decoding and host-linear texture conversion.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use remu_shader::ImageBinding;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC1_UNORM_SRGB, DXGI_FORMAT_BC2_UNORM,
    DXGI_FORMAT_BC2_UNORM_SRGB, DXGI_FORMAT_BC3_UNORM, DXGI_FORMAT_BC3_UNORM_SRGB,
    DXGI_FORMAT_BC4_SNORM, DXGI_FORMAT_BC4_UNORM, DXGI_FORMAT_BC5_SNORM, DXGI_FORMAT_BC5_UNORM,
    DXGI_FORMAT_BC6H_SF16, DXGI_FORMAT_BC6H_UF16, DXGI_FORMAT_BC7_UNORM,
    DXGI_FORMAT_BC7_UNORM_SRGB, DXGI_FORMAT_R8_SINT, DXGI_FORMAT_R8_SNORM, DXGI_FORMAT_R8_UINT,
    DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_SINT, DXGI_FORMAT_R8G8_SNORM, DXGI_FORMAT_R8G8_UINT,
    DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R8G8B8A8_SINT, DXGI_FORMAT_R8G8B8A8_SNORM,
    DXGI_FORMAT_R8G8B8A8_UINT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
    DXGI_FORMAT_R9G9B9E5_SHAREDEXP, DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R11G11B10_FLOAT,
    DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16_SINT, DXGI_FORMAT_R16_SNORM, DXGI_FORMAT_R16_UINT,
    DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16_SINT,
    DXGI_FORMAT_R16G16_SNORM, DXGI_FORMAT_R16G16_UINT, DXGI_FORMAT_R16G16_UNORM,
    DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_SINT, DXGI_FORMAT_R16G16B16A16_SNORM,
    DXGI_FORMAT_R16G16B16A16_UINT, DXGI_FORMAT_R16G16B16A16_UNORM, DXGI_FORMAT_R32_FLOAT,
    DXGI_FORMAT_R32_SINT, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32G32_FLOAT, DXGI_FORMAT_R32G32_SINT,
    DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32A32_SINT,
    DXGI_FORMAT_R32G32B32A32_UINT,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DecodedTexture {
    pub(super) address: u64,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) pitch: u32,
    pub(super) tile_mode: u32,
    pub(super) format: DXGI_FORMAT,
    /// Bytes per texel for ordinary formats or per 4x4 block for BC formats.
    pub(super) bytes_per_element: u32,
    pub(super) block_extent: [u32; 2],
    pub(super) texture_type: u32,
    pub(super) layer_count: u32,
    pub(super) mip_level_count: u32,
    pub(super) writable: bool,
}

impl std::hash::Hash for DecodedTexture {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.address, state);
        std::hash::Hash::hash(&self.width, state);
        std::hash::Hash::hash(&self.height, state);
        std::hash::Hash::hash(&self.pitch, state);
        std::hash::Hash::hash(&self.tile_mode, state);
        std::hash::Hash::hash(&self.format.0, state);
        std::hash::Hash::hash(&self.bytes_per_element, state);
        std::hash::Hash::hash(&self.block_extent, state);
        std::hash::Hash::hash(&self.texture_type, state);
        std::hash::Hash::hash(&self.layer_count, state);
        std::hash::Hash::hash(&self.mip_level_count, state);
        std::hash::Hash::hash(&self.writable, state);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GuestMipLevelLayout {
    width: usize,
    height: usize,
    padded_width: usize,
    offset: usize,
    tail_x: usize,
    tail_y: usize,
    in_tail: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GuestMipChainLayout {
    levels: Vec<GuestMipLevelLayout>,
    byte_count: usize,
    block_width: usize,
    block_height: usize,
    bytes_per_element: usize,
    block_bytes: usize,
    tile_mode: u32,
}

const MODE27_MIP_TAIL_LOCATIONS: [[(usize, usize); 12]; 5] = [
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

const MODE5_MAX_MIPS_IN_TAIL: usize = 8;
const BLOCK_256_DIMENSIONS: [(usize, usize); 5] = [(16, 16), (16, 8), (8, 8), (8, 4), (4, 4)];

impl DecodedTexture {
    pub(super) fn element_extent(self) -> [usize; 2] {
        [
            usize::try_from(self.width.div_ceil(self.block_extent[0])).expect("texture width fits"),
            usize::try_from(self.height.div_ceil(self.block_extent[1]))
                .expect("texture height fits"),
        ]
    }

    fn pitch_elements(self) -> usize {
        usize::try_from(self.pitch.div_ceil(self.block_extent[0])).expect("texture pitch fits")
    }

    pub(super) fn linear_byte_count(self) -> Result<usize, String> {
        let [width, height] = self.element_extent();
        width
            .checked_mul(height)
            .and_then(|count| count.checked_mul(self.bytes_per_element as usize))
            .and_then(|count| count.checked_mul(self.layer_count as usize))
            .ok_or_else(|| "linear texture size overflows".to_owned())
    }

    pub(super) fn raw_layer_byte_count(self) -> Result<usize, String> {
        let height = self.element_extent()[1];
        let bytes = self.bytes_per_element as usize;
        match tiled_block_bytes(self.tile_mode) {
            Some(block_bytes) => {
                let (block_width, block_height) = tiled_block_dimensions(block_bytes, bytes)
                    .ok_or("unsupported tiled texture element size")?;
                self.pitch_elements()
                    .div_ceil(block_width)
                    .checked_mul(height.div_ceil(block_height))
                    .and_then(|blocks| blocks.checked_mul(block_bytes))
                    .ok_or_else(|| "tiled texture size overflows".to_owned())
            }
            None if self.tile_mode == 0 => self
                .pitch_elements()
                .checked_mul(height)
                .and_then(|count| count.checked_mul(bytes))
                .ok_or_else(|| "linear pitched texture size overflows".to_owned()),
            None => Err(format!("unsupported texture tile mode {}", self.tile_mode)),
        }
    }

    pub(super) fn raw_byte_count(self) -> Result<usize, String> {
        self.raw_layer_byte_count()?
            .checked_mul(self.layer_count as usize)
            .ok_or_else(|| "layered texture size overflows".to_owned())
    }
}

pub(super) fn decode_texture(binding: &ImageBinding) -> Result<DecodedTexture, String> {
    let words = &binding.resource_descriptor;
    if words.len() < 4 {
        return Err(format!(
            "image descriptor at PC {:#x} has fewer than four dwords",
            binding.pc
        ));
    }
    if words.iter().all(|word| *word == 0) {
        let writable =
            binding.opcode.starts_with("ImageStore") || binding.opcode.starts_with("ImageAtomic");
        return Ok(DecodedTexture {
            address: 0,
            width: 1,
            height: 1,
            pitch: 1,
            tile_mode: 0,
            format: DXGI_FORMAT_R8G8B8A8_UNORM,
            bytes_per_element: 4,
            block_extent: [1, 1],
            texture_type: match binding.control.dimension {
                2 => 10,
                3 | 5 => 13,
                _ => 9,
            },
            layer_count: 1,
            mip_level_count: 1,
            writable,
        });
    }
    let address = ((u64::from(words[1] & 0xff) << 32) | u64::from(words[0])) << 8;
    let width = ((words[1] >> 30) | ((words[2] & 0x3fff) << 2))
        .checked_add(1)
        .ok_or("image width overflows")?;
    let height = ((words[2] >> 14) & 0xffff)
        .checked_add(1)
        .ok_or("image height overflows")?;
    let unified = (words[1] >> 20) & 0x1ff;
    let (data_format, number_format) = remu_shader::decode_unified_format(unified)
        .ok_or_else(|| format!("unknown GFX10 unified image format {unified}"))?;
    let (format, bytes_per_element, block_extent) = decode_dxgi_format(data_format, number_format)?;
    let texture_type = (words[3] >> 28) & 0xf;
    let tile_mode = (words[3] >> 20) & 0x1f;
    let descriptor_pitch_word = words.get(4).copied().unwrap_or_default();
    let explicit_pitch = matches!(texture_type, 8 | 9 | 14) && descriptor_pitch_word != 0;
    let pitch = if explicit_pitch {
        (descriptor_pitch_word & 0x3fff)
            .checked_add(1)
            .ok_or("image pitch overflows")?
    } else if tile_mode == 0 {
        let alignment = (256 / bytes_per_element.max(1)).max(1);
        width
            .checked_add(alignment - 1)
            .map(|value| value & !(alignment - 1))
            .ok_or("linear image pitch overflows")?
    } else {
        width
    };
    let layers = words.get(4).copied().unwrap_or_default();
    let layer_count = match texture_type {
        10 => (layers & 0x1fff)
            .checked_add(1)
            .ok_or("3D depth overflows")?,
        11 | 13 => {
            let first = (layers >> 16) & 0x1fff;
            let last = layers & 0x1fff;
            last.checked_sub(first)
                .and_then(|value| value.checked_add(1))
                .ok_or("invalid image array slice range")?
        }
        _ => 1,
    };
    let base_mip = (words[3] >> 12) & 0xf;
    let last_mip = (words[3] >> 16) & 0xf;
    let mip_level_count = last_mip
        .checked_sub(base_mip)
        .and_then(|value| value.checked_add(1))
        .ok_or("invalid image mip range")?;
    let writable =
        binding.opcode.starts_with("ImageStore") || binding.opcode.starts_with("ImageAtomic");
    if address < 0x10_000
        || width == 0
        || height == 0
        || width > 16_384
        || height > 16_384
        || pitch < width
        || pitch > 16_384
        || layer_count == 0
        || layer_count > 2_048
    {
        return Err(format!(
            "invalid image address/extent address={address:#x} {width}x{height} pitch={pitch} layers={layer_count}"
        ));
    }
    if !matches!(texture_type, 8..=11 | 13 | 14) {
        return Err(format!("unsupported GFX10 texture type {texture_type}"));
    }
    if !matches!(tile_mode, 0 | 1 | 5 | 9 | 24 | 27) {
        return Err(format!("unsupported color texture tile mode {tile_mode}"));
    }
    Ok(DecodedTexture {
        address,
        width,
        height,
        pitch,
        tile_mode,
        format,
        bytes_per_element,
        block_extent,
        texture_type,
        layer_count,
        mip_level_count,
        writable,
    })
}

pub(super) fn read_texture_linear(descriptor: DecodedTexture) -> Result<Vec<u8>, String> {
    let mut linear = vec![0_u8; descriptor.linear_byte_count()?];
    if descriptor.address == 0 {
        return Ok(linear);
    }
    let layer_raw_bytes = descriptor.raw_layer_byte_count()?;
    let mut raw = vec![0_u8; descriptor.raw_byte_count()?];
    // SAFETY: evaluated image descriptors name mapped guest allocations retained through event
    // retirement. Descriptor validation above bounds the host allocation.
    unsafe {
        std::ptr::copy_nonoverlapping(descriptor.address as *const u8, raw.as_mut_ptr(), raw.len());
    }
    let layer_linear_bytes = descriptor.linear_byte_count()? / descriptor.layer_count as usize;
    for layer in 0..descriptor.layer_count as usize {
        detile_layer(
            descriptor,
            layer,
            &raw[layer * layer_raw_bytes..(layer + 1) * layer_raw_bytes],
            &mut linear[layer * layer_linear_bytes..(layer + 1) * layer_linear_bytes],
        )?;
    }
    Ok(linear)
}

pub(super) fn read_texture_mip_chain(descriptor: DecodedTexture) -> Result<Vec<Vec<u8>>, String> {
    if descriptor.mip_level_count == 1 {
        return read_texture_linear(descriptor).map(|level| vec![level]);
    }
    if descriptor.texture_type != 9 || descriptor.layer_count != 1 {
        return Err(format!(
            "mipmapped D3D12 textures currently require one 2D layer, got type={} layers={}",
            descriptor.texture_type, descriptor.layer_count
        ));
    }
    let layout = mip_chain_layout(descriptor)?;
    let mut raw = vec![0_u8; layout.byte_count];
    // SAFETY: descriptor evaluation retains the complete guest mip allocation until submission
    // retirement, and the exact GFX10 layout calculation bounds this copy.
    unsafe {
        std::ptr::copy_nonoverlapping(descriptor.address as *const u8, raw.as_mut_ptr(), raw.len());
    }
    layout
        .levels
        .iter()
        .copied()
        .map(|level| {
            let mut linear = vec![
                0_u8;
                level
                    .width
                    .checked_mul(level.height)
                    .and_then(|count| count.checked_mul(layout.bytes_per_element))
                    .ok_or("mip linear size overflows")?
            ];
            copy_mip_level_to_linear(&layout, level, &raw, &mut linear)?;
            Ok(linear)
        })
        .collect()
}

#[cfg(test)]
pub(super) fn write_texture_mip_chain(
    descriptor: DecodedTexture,
    levels: &[Vec<u8>],
) -> Result<(), String> {
    if descriptor.mip_level_count == 1 {
        return levels
            .first()
            .ok_or_else(|| "texture mip chain is empty".to_owned())
            .and_then(|level| write_texture_linear(descriptor, level));
    }
    let layout = mip_chain_layout(descriptor)?;
    if levels.len() != layout.levels.len() {
        return Err(format!(
            "texture mip chain has {} levels, expected {}",
            levels.len(),
            layout.levels.len()
        ));
    }
    let mut raw = vec![0_u8; layout.byte_count];
    for ((&level, linear), level_index) in layout.levels.iter().zip(levels).zip(0_u32..) {
        copy_linear_to_mip_level(&layout, level, linear, &mut raw)
            .map_err(|error| format!("could not encode guest mip level {level_index}: {error}"))?;
    }
    // SAFETY: the evaluated writable descriptor retains the exact mip allocation through
    // submission retirement and the layout calculation bounds the complete copy.
    unsafe {
        std::ptr::copy_nonoverlapping(raw.as_ptr(), descriptor.address as *mut u8, raw.len());
    }
    remu_memory::record_guest_write(
        remu_memory::GuestWriteSite::TextureWritebackRange,
        descriptor.address,
        raw.len() as u64,
        &raw,
        remu_gpu::gpu_completed_submission_sequence(),
    );
    Ok(())
}

fn shift_ceil(value: usize, shift: u32) -> Option<usize> {
    let divisor = 1_usize.checked_shl(shift)?;
    value.checked_add(divisor - 1).map(|value| value / divisor)
}

fn mode5_mip_tail_location(tail_level: usize, bytes_log2: usize) -> Result<(usize, usize), String> {
    if tail_level >= MODE5_MAX_MIPS_IN_TAIL || bytes_log2 >= 5 {
        return Err(format!(
            "invalid mode-5 mip-tail level/element logarithm {tail_level}/{bytes_log2}"
        ));
    }
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

fn mip_chain_layout(descriptor: DecodedTexture) -> Result<GuestMipChainLayout, String> {
    if !matches!(descriptor.tile_mode, 5 | 27)
        || descriptor.mip_level_count == 0
        || descriptor.mip_level_count > 16
    {
        return Err(format!(
            "mip layout requires mode 5 or 27 and 1..=16 levels, got mode={} levels={}",
            descriptor.tile_mode, descriptor.mip_level_count
        ));
    }
    let [width, height] = descriptor.element_extent();
    let bytes = descriptor.bytes_per_element as usize;
    let bytes_log2 = bytes.trailing_zeros() as usize;
    if !bytes.is_power_of_two() || bytes_log2 >= 5 {
        return Err(format!("unsupported {bytes}-byte mip element"));
    }
    let block_bytes = if descriptor.tile_mode == 5 {
        4_096
    } else {
        65_536
    };
    let (block_width, block_height) =
        tiled_block_dimensions(block_bytes, bytes).ok_or("unsupported mip element size")?;
    let level_count = descriptor.mip_level_count as usize;
    let maximum_tail_levels = if descriptor.tile_mode == 5 { 8 } else { 12 };
    let tail_width = block_width / 2;
    let mut first_tail = level_count;
    if level_count > 1 {
        for level in 0..level_count {
            let shift = level as u32;
            if shift_ceil(width, shift).is_some_and(|value| value <= tail_width)
                && shift_ceil(height, shift).is_some_and(|value| value <= block_height)
                && level_count - level <= maximum_tail_levels
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
        block_bytes
    } else {
        0
    };
    for level in (0..first_tail).rev() {
        let shift = level as u32;
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
            .and_then(|count| count.checked_mul(bytes))
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
        let tail_index = level - first_tail;
        let (tail_x, tail_y) = if descriptor.tile_mode == 5 {
            mode5_mip_tail_location(tail_index, bytes_log2)?
        } else {
            *MODE27_MIP_TAIL_LOCATIONS[bytes_log2]
                .get(tail_index)
                .ok_or("mode-27 mip tail has too many levels")?
        };
        let mip_width = shift_ceil(width, level as u32)
            .ok_or("mip width overflows")?
            .max(1);
        let mip_height = shift_ceil(height, level as u32)
            .ok_or("mip height overflows")?
            .max(1);
        if tail_x + mip_width > block_width || tail_y + mip_height > block_height {
            return Err(format!(
                "mip {level} tail placement ({tail_x},{tail_y}) {mip_width}x{mip_height} exceeds {block_width}x{block_height}"
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
        bytes_per_element: bytes,
        block_bytes,
        tile_mode: descriptor.tile_mode,
    })
}

fn copy_mip_level_to_linear(
    layout: &GuestMipChainLayout,
    level: GuestMipLevelLayout,
    raw: &[u8],
    linear: &mut [u8],
) -> Result<(), String> {
    let expected = level
        .width
        .checked_mul(level.height)
        .and_then(|count| count.checked_mul(layout.bytes_per_element))
        .ok_or("mip linear size overflows")?;
    if linear.len() != expected {
        return Err("mip destination size mismatch".to_owned());
    }
    if level.in_tail {
        let offsets = block_offsets(
            layout.tile_mode,
            0,
            0,
            layout.bytes_per_element,
            layout.block_width,
            layout.block_height,
        )?;
        for y in 0..level.height {
            for x in 0..level.width {
                let source = level.offset
                    + usize::from(
                        offsets[(level.tail_y + y) * layout.block_width + level.tail_x + x],
                    );
                let destination = (y * level.width + x) * layout.bytes_per_element;
                linear[destination..destination + layout.bytes_per_element].copy_from_slice(
                    raw.get(source..source + layout.bytes_per_element)
                        .ok_or("mip-tail texel exceeds backing")?,
                );
            }
        }
        return Ok(());
    }
    let blocks_per_row = level.padded_width / layout.block_width;
    for block_y in 0..level.height.div_ceil(layout.block_height) {
        for block_x in 0..level.width.div_ceil(layout.block_width) {
            let block_base =
                level.offset + (block_y * blocks_per_row + block_x) * layout.block_bytes;
            let offsets = block_offsets(
                layout.tile_mode,
                block_x,
                block_y,
                layout.bytes_per_element,
                layout.block_width,
                layout.block_height,
            )?;
            let rows = (level.height - block_y * layout.block_height).min(layout.block_height);
            let columns = (level.width - block_x * layout.block_width).min(layout.block_width);
            for local_y in 0..rows {
                for local_x in 0..columns {
                    let source =
                        block_base + usize::from(offsets[local_y * layout.block_width + local_x]);
                    let destination = ((block_y * layout.block_height + local_y) * level.width
                        + block_x * layout.block_width
                        + local_x)
                        * layout.bytes_per_element;
                    linear[destination..destination + layout.bytes_per_element].copy_from_slice(
                        raw.get(source..source + layout.bytes_per_element)
                            .ok_or("mip texel exceeds backing")?,
                    );
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn copy_linear_to_mip_level(
    layout: &GuestMipChainLayout,
    level: GuestMipLevelLayout,
    linear: &[u8],
    raw: &mut [u8],
) -> Result<(), String> {
    let mut round_trip = vec![0_u8; linear.len()];
    if level.in_tail {
        let offsets = block_offsets(
            layout.tile_mode,
            0,
            0,
            layout.bytes_per_element,
            layout.block_width,
            layout.block_height,
        )?;
        for y in 0..level.height {
            for x in 0..level.width {
                let source = (y * level.width + x) * layout.bytes_per_element;
                let destination = level.offset
                    + usize::from(
                        offsets[(level.tail_y + y) * layout.block_width + level.tail_x + x],
                    );
                raw[destination..destination + layout.bytes_per_element]
                    .copy_from_slice(&linear[source..source + layout.bytes_per_element]);
            }
        }
    } else {
        let blocks_per_row = level.padded_width / layout.block_width;
        for block_y in 0..level.height.div_ceil(layout.block_height) {
            for block_x in 0..level.width.div_ceil(layout.block_width) {
                let block_base =
                    level.offset + (block_y * blocks_per_row + block_x) * layout.block_bytes;
                let offsets = block_offsets(
                    layout.tile_mode,
                    block_x,
                    block_y,
                    layout.bytes_per_element,
                    layout.block_width,
                    layout.block_height,
                )?;
                let rows = (level.height - block_y * layout.block_height).min(layout.block_height);
                let columns = (level.width - block_x * layout.block_width).min(layout.block_width);
                for local_y in 0..rows {
                    for local_x in 0..columns {
                        let source = ((block_y * layout.block_height + local_y) * level.width
                            + block_x * layout.block_width
                            + local_x)
                            * layout.bytes_per_element;
                        let destination = block_base
                            + usize::from(offsets[local_y * layout.block_width + local_x]);
                        raw[destination..destination + layout.bytes_per_element]
                            .copy_from_slice(&linear[source..source + layout.bytes_per_element]);
                    }
                }
            }
        }
    }
    copy_mip_level_to_linear(layout, level, raw, &mut round_trip)?;
    if round_trip != linear {
        return Err("mip level did not round trip".to_owned());
    }
    Ok(())
}

pub(super) fn write_texture_linear(
    descriptor: DecodedTexture,
    linear: &[u8],
) -> Result<(), String> {
    if descriptor.address == 0 {
        return Ok(());
    }
    if linear.len() != descriptor.linear_byte_count()? {
        return Err("texture writeback linear byte count mismatch".to_owned());
    }
    let layer_raw_bytes = descriptor.raw_layer_byte_count()?;
    let mut raw = vec![0_u8; descriptor.raw_byte_count()?];
    // A guest image owns only the texels named by width/height. Pitch padding and unused tiled
    // block slots can belong to another tenant in a pooled allocation (including allocator
    // metadata). Preserve those bytes instead of zero-filling and publishing the full footprint.
    // SAFETY: the evaluated descriptor retains the complete validated raw mapping through this
    // synchronous publication.
    unsafe {
        std::ptr::copy_nonoverlapping(descriptor.address as *const u8, raw.as_mut_ptr(), raw.len());
    }
    let layer_linear_bytes = linear.len() / descriptor.layer_count as usize;
    for layer in 0..descriptor.layer_count as usize {
        retile_layer(
            descriptor,
            layer,
            &linear[layer * layer_linear_bytes..(layer + 1) * layer_linear_bytes],
            &mut raw[layer * layer_raw_bytes..(layer + 1) * layer_raw_bytes],
        )?;
    }
    // SAFETY: the writable evaluated descriptor owns the complete validated raw range.
    unsafe {
        std::ptr::copy_nonoverlapping(raw.as_ptr(), descriptor.address as *mut u8, raw.len());
    }
    Ok(())
}

/// Converts one single-layer, four-byte mode-27 surface to tightly packed host rows.
///
/// Scanout and ordinary BGRA render targets use this layout. Keeping the hot loop here lets both
/// presentation and draw execution share the cached parity tables and avoids bounds-checked
/// four-byte slice copies for every pixel of a 4K frame.
pub(super) fn detile_mode27_bgra(
    raw: &[u8],
    width: usize,
    height: usize,
    pitch: usize,
) -> Result<Vec<u8>, String> {
    let mut linear = vec![
        0_u8;
        width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or("mode-27 output size overflows")?
    ];
    copy_mode27_bgra_to_linear(raw, width, height, pitch, &mut linear)?;
    Ok(linear)
}

fn copy_mode27_bgra_to_linear(
    raw: &[u8],
    width: usize,
    height: usize,
    pitch: usize,
    linear: &mut [u8],
) -> Result<(), String> {
    let blocks_per_row = pitch.div_ceil(128);
    let expected = blocks_per_row
        .checked_mul(height.div_ceil(128))
        .and_then(|blocks| blocks.checked_mul(65_536))
        .ok_or("mode-27 backing size overflows")?;
    let linear_bytes = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or("mode-27 linear size overflows")?;
    if raw.len() != expected || linear.len() != linear_bytes {
        return Err(format!(
            "mode-27 copy has raw/linear sizes {}/{}, expected {expected}/{linear_bytes}",
            raw.len(),
            linear.len()
        ));
    }
    for block_y in 0..height.div_ceil(128) {
        let rows = (height - block_y * 128).min(128);
        for block_x in 0..width.div_ceil(128) {
            let columns = (width - block_x * 128).min(128);
            let offsets = block_offsets(27, block_x, block_y, 4, 128, 128)?;
            let block = (block_y * blocks_per_row + block_x) * 65_536;
            for local_y in 0..rows {
                let destination_row = (block_y * 128 + local_y) * width + block_x * 128;
                for local_x in 0..columns {
                    let source = block + usize::from(offsets[local_y * 128 + local_x]);
                    let destination = (destination_row + local_x) * 4;
                    // SAFETY: the validated backing sizes and cached in-block offsets bound both
                    // four-byte accesses; unaligned operations match byte-oriented guest memory.
                    unsafe {
                        let pixel = raw.as_ptr().add(source).cast::<u32>().read_unaligned();
                        linear
                            .as_mut_ptr()
                            .add(destination)
                            .cast::<u32>()
                            .write_unaligned(pixel);
                    }
                }
            }
        }
    }
    Ok(())
}

fn copy_linear_to_mode27_bgra(
    linear: &[u8],
    width: usize,
    height: usize,
    pitch: usize,
    raw: &mut [u8],
) -> Result<(), String> {
    let blocks_per_row = pitch.div_ceil(128);
    let expected = blocks_per_row
        .checked_mul(height.div_ceil(128))
        .and_then(|blocks| blocks.checked_mul(65_536))
        .ok_or("mode-27 backing size overflows")?;
    let linear_bytes = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or("mode-27 linear size overflows")?;
    if raw.len() != expected || linear.len() != linear_bytes {
        return Err(format!(
            "mode-27 writeback has raw/linear sizes {}/{}, expected {expected}/{linear_bytes}",
            raw.len(),
            linear.len()
        ));
    }
    for block_y in 0..height.div_ceil(128) {
        let rows = (height - block_y * 128).min(128);
        for block_x in 0..width.div_ceil(128) {
            let columns = (width - block_x * 128).min(128);
            let offsets = block_offsets(27, block_x, block_y, 4, 128, 128)?;
            let block = (block_y * blocks_per_row + block_x) * 65_536;
            for local_y in 0..rows {
                let source_row = (block_y * 128 + local_y) * width + block_x * 128;
                for local_x in 0..columns {
                    let source = (source_row + local_x) * 4;
                    let destination = block + usize::from(offsets[local_y * 128 + local_x]);
                    // SAFETY: the validated backing sizes and cached in-block offsets bound both
                    // four-byte accesses; unaligned operations match byte-oriented guest memory.
                    unsafe {
                        let pixel = linear.as_ptr().add(source).cast::<u32>().read_unaligned();
                        raw.as_mut_ptr()
                            .add(destination)
                            .cast::<u32>()
                            .write_unaligned(pixel);
                    }
                }
            }
        }
    }
    Ok(())
}

fn detile_layer(
    descriptor: DecodedTexture,
    layer: usize,
    raw: &[u8],
    linear: &mut [u8],
) -> Result<(), String> {
    let [width, height] = descriptor.element_extent();
    let pitch = descriptor.pitch_elements();
    let bytes = descriptor.bytes_per_element as usize;
    if descriptor.tile_mode == 0 {
        for y in 0..height {
            linear[y * width * bytes..(y + 1) * width * bytes]
                .copy_from_slice(&raw[y * pitch * bytes..y * pitch * bytes + width * bytes]);
        }
        return Ok(());
    }
    if descriptor.tile_mode == 27 && bytes == 4 && layer == 0 {
        return copy_mode27_bgra_to_linear(raw, width, height, pitch, linear);
    }
    let block_bytes = tiled_block_bytes(descriptor.tile_mode).ok_or("unsupported tile mode")?;
    let (block_width, block_height) =
        tiled_block_dimensions(block_bytes, bytes).ok_or("unsupported tiled element size")?;
    let blocks_per_row = pitch.div_ceil(block_width);
    for block_y in 0..height.div_ceil(block_height) {
        for block_x in 0..width.div_ceil(block_width) {
            let offsets = block_offsets(
                descriptor.tile_mode,
                block_x,
                block_y,
                bytes,
                block_width,
                block_height,
            )?;
            let block = (block_y * blocks_per_row + block_x) * block_bytes;
            for local_y in 0..(height - block_y * block_height).min(block_height) {
                for local_x in 0..(width - block_x * block_width).min(block_width) {
                    let source = block
                        + (usize::from(offsets[local_y * block_width + local_x])
                            ^ if descriptor.tile_mode == 27 {
                                surface_z_xor(layer)
                            } else {
                                0
                            });
                    let destination = ((block_y * block_height + local_y) * width
                        + block_x * block_width
                        + local_x)
                        * bytes;
                    linear[destination..destination + bytes].copy_from_slice(
                        raw.get(source..source + bytes)
                            .ok_or("tiled texel exceeds backing")?,
                    );
                }
            }
        }
    }
    Ok(())
}

fn retile_layer(
    descriptor: DecodedTexture,
    layer: usize,
    linear: &[u8],
    raw: &mut [u8],
) -> Result<(), String> {
    let [width, height] = descriptor.element_extent();
    let pitch = descriptor.pitch_elements();
    let bytes = descriptor.bytes_per_element as usize;
    if descriptor.tile_mode == 0 {
        for y in 0..height {
            raw[y * pitch * bytes..y * pitch * bytes + width * bytes]
                .copy_from_slice(&linear[y * width * bytes..(y + 1) * width * bytes]);
        }
        return Ok(());
    }
    if descriptor.tile_mode == 27 && bytes == 4 && layer == 0 {
        return copy_linear_to_mode27_bgra(linear, width, height, pitch, raw);
    }
    let block_bytes = tiled_block_bytes(descriptor.tile_mode).ok_or("unsupported tile mode")?;
    let (block_width, block_height) =
        tiled_block_dimensions(block_bytes, bytes).ok_or("unsupported tiled element size")?;
    let blocks_per_row = pitch.div_ceil(block_width);
    for block_y in 0..height.div_ceil(block_height) {
        for block_x in 0..width.div_ceil(block_width) {
            let offsets = block_offsets(
                descriptor.tile_mode,
                block_x,
                block_y,
                bytes,
                block_width,
                block_height,
            )?;
            let block = (block_y * blocks_per_row + block_x) * block_bytes;
            for local_y in 0..(height - block_y * block_height).min(block_height) {
                for local_x in 0..(width - block_x * block_width).min(block_width) {
                    let source = ((block_y * block_height + local_y) * width
                        + block_x * block_width
                        + local_x)
                        * bytes;
                    let destination = block
                        + (usize::from(offsets[local_y * block_width + local_x])
                            ^ if descriptor.tile_mode == 27 {
                                surface_z_xor(layer)
                            } else {
                                0
                            });
                    raw.get_mut(destination..destination + bytes)
                        .ok_or("tiled texel exceeds backing")?
                        .copy_from_slice(&linear[source..source + bytes]);
                }
            }
        }
    }
    Ok(())
}

const fn tiled_block_bytes(mode: u32) -> Option<usize> {
    match mode {
        1 => Some(256),
        5 => Some(4_096),
        9 | 24 | 27 => Some(65_536),
        _ => None,
    }
}

fn tiled_block_dimensions(block_bytes: usize, bytes: usize) -> Option<(usize, usize)> {
    if !bytes.is_power_of_two() || !(1..=16).contains(&bytes) {
        return None;
    }
    let bits = (block_bytes / bytes).trailing_zeros() as usize;
    let width_bits = bits.div_ceil(2);
    Some((1 << width_bits, 1 << (bits - width_bits)))
}

fn block_offsets(
    mode: u32,
    block_x: usize,
    block_y: usize,
    bytes: usize,
    width: usize,
    height: usize,
) -> Result<Arc<[u16]>, String> {
    type CacheKey = (u32, usize, usize, usize, usize, usize);
    static CACHE: OnceLock<Mutex<HashMap<CacheKey, Arc<[u16]>>>> = OnceLock::new();
    let parity_mode = matches!(mode, 24 | 27);
    let parity_x = if parity_mode { block_x & 1 } else { 0 };
    let parity_y = if parity_mode { block_y & 1 } else { 0 };
    let key = (mode, bytes, width, height, parity_x, parity_y);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(offsets) = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&key)
        .cloned()
    {
        return Ok(offsets);
    }
    let mut offsets = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let global_x = if parity_mode {
                x | (parity_x * width)
            } else {
                x
            };
            let global_y = if parity_mode {
                y | (parity_y * height)
            } else {
                y
            };
            let offset = match mode {
                1 => standard_offset(global_x, global_y, bytes) & 0xff,
                5 => standard_offset_sized(global_x, global_y, bytes, 12),
                9 => standard_offset(global_x, global_y, bytes),
                24 => mode24_offset(global_x, global_y, bytes),
                27 => mode27_offset(global_x, global_y, bytes),
                _ => return Err(format!("unsupported tile mode {mode}")),
            };
            offsets.push(u16::try_from(offset).map_err(|_| "tile offset exceeds 16 bits")?);
        }
    }
    let offsets: Arc<[u16]> = offsets.into();
    let mut cache = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Ok(cache.entry(key).or_insert_with(|| offsets.clone()).clone())
}

fn standard_offset(x: usize, y: usize, bytes: usize) -> usize {
    standard_offset_sized(x, y, bytes, 16)
}

fn standard_offset_sized(x: usize, y: usize, bytes: usize, bits: usize) -> usize {
    let bit = |value: usize, index: usize| (value >> index) & 1;
    let log = bytes.trailing_zeros() as usize;
    let order: Vec<usize> = match log {
        0 => vec![
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
        1 => vec![
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
        2 => vec![
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
        3 => vec![
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
        4 => vec![
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
        _ => unreachable!(),
    };
    order
        .into_iter()
        .take(bits)
        .enumerate()
        .fold(0, |result, (index, value)| result | (value << index))
}

fn mode27_offset(x: usize, y: usize, bytes: usize) -> usize {
    match bytes.trailing_zeros() {
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
        _ => unreachable!(),
    }
}

fn mode24_offset(x: usize, y: usize, bytes: usize) -> usize {
    match bytes.trailing_zeros() {
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
        _ => unreachable!("mode 24 supports one- through eight-byte elements"),
    }
}

const fn surface_z_xor(z: usize) -> usize {
    ((z & 8) << 5) ^ ((z & 4) << 7) ^ ((z & 2) << 9) ^ ((z & 1) << 11)
}

pub(super) fn decode_dxgi_format(
    data: u32,
    number: u32,
) -> Result<(DXGI_FORMAT, u32, [u32; 2]), String> {
    let ordinary = |format, bytes| Ok((format, bytes, [1, 1]));
    match (data, number) {
        (1, 0) => ordinary(DXGI_FORMAT_R8_UNORM, 1),
        (1, 1) => ordinary(DXGI_FORMAT_R8_SNORM, 1),
        (1, 4) => ordinary(DXGI_FORMAT_R8_UINT, 1),
        (1, 5) => ordinary(DXGI_FORMAT_R8_SINT, 1),
        (2, 0) => ordinary(DXGI_FORMAT_R16_UNORM, 2),
        (2, 1) => ordinary(DXGI_FORMAT_R16_SNORM, 2),
        (2, 4) => ordinary(DXGI_FORMAT_R16_UINT, 2),
        (2, 5) => ordinary(DXGI_FORMAT_R16_SINT, 2),
        (2, 7) => ordinary(DXGI_FORMAT_R16_FLOAT, 2),
        (3, 0) => ordinary(DXGI_FORMAT_R8G8_UNORM, 2),
        (3, 1) => ordinary(DXGI_FORMAT_R8G8_SNORM, 2),
        (3, 4) => ordinary(DXGI_FORMAT_R8G8_UINT, 2),
        (3, 5) => ordinary(DXGI_FORMAT_R8G8_SINT, 2),
        (4, 4) => ordinary(DXGI_FORMAT_R32_UINT, 4),
        (4, 5) => ordinary(DXGI_FORMAT_R32_SINT, 4),
        (4, 7) => ordinary(DXGI_FORMAT_R32_FLOAT, 4),
        (5, 0) => ordinary(DXGI_FORMAT_R16G16_UNORM, 4),
        (5, 1) => ordinary(DXGI_FORMAT_R16G16_SNORM, 4),
        (5, 4) => ordinary(DXGI_FORMAT_R16G16_UINT, 4),
        (5, 5) => ordinary(DXGI_FORMAT_R16G16_SINT, 4),
        (5, 7) => ordinary(DXGI_FORMAT_R16G16_FLOAT, 4),
        (6, 7) => ordinary(DXGI_FORMAT_R11G11B10_FLOAT, 4),
        (9, 0) => ordinary(DXGI_FORMAT_R10G10B10A2_UNORM, 4),
        (10, 0) => ordinary(DXGI_FORMAT_R8G8B8A8_UNORM, 4),
        (10, 1) => ordinary(DXGI_FORMAT_R8G8B8A8_SNORM, 4),
        (10, 4) => ordinary(DXGI_FORMAT_R8G8B8A8_UINT, 4),
        (10, 5) => ordinary(DXGI_FORMAT_R8G8B8A8_SINT, 4),
        (10, 6 | 9) => ordinary(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 4),
        (11, 4) => ordinary(DXGI_FORMAT_R32G32_UINT, 8),
        (11, 5) => ordinary(DXGI_FORMAT_R32G32_SINT, 8),
        (11, 7) => ordinary(DXGI_FORMAT_R32G32_FLOAT, 8),
        (12, 0) => ordinary(DXGI_FORMAT_R16G16B16A16_UNORM, 8),
        (12, 1) => ordinary(DXGI_FORMAT_R16G16B16A16_SNORM, 8),
        (12, 4) => ordinary(DXGI_FORMAT_R16G16B16A16_UINT, 8),
        (12, 5) => ordinary(DXGI_FORMAT_R16G16B16A16_SINT, 8),
        (12, 7) => ordinary(DXGI_FORMAT_R16G16B16A16_FLOAT, 8),
        (14, 4) => ordinary(DXGI_FORMAT_R32G32B32A32_UINT, 16),
        (14, 5) => ordinary(DXGI_FORMAT_R32G32B32A32_SINT, 16),
        (14, 7) => ordinary(DXGI_FORMAT_R32G32B32A32_FLOAT, 16),
        (24, 7) => ordinary(DXGI_FORMAT_R9G9B9E5_SHAREDEXP, 4),
        (169, _) => Ok((DXGI_FORMAT_BC1_UNORM, 8, [4, 4])),
        (170, _) => Ok((DXGI_FORMAT_BC1_UNORM_SRGB, 8, [4, 4])),
        (171, _) => Ok((DXGI_FORMAT_BC2_UNORM, 16, [4, 4])),
        (172, _) => Ok((DXGI_FORMAT_BC2_UNORM_SRGB, 16, [4, 4])),
        (173, _) => Ok((DXGI_FORMAT_BC3_UNORM, 16, [4, 4])),
        (174, _) => Ok((DXGI_FORMAT_BC3_UNORM_SRGB, 16, [4, 4])),
        (175, _) => Ok((DXGI_FORMAT_BC4_UNORM, 8, [4, 4])),
        (176, _) => Ok((DXGI_FORMAT_BC4_SNORM, 8, [4, 4])),
        (177, _) => Ok((DXGI_FORMAT_BC5_UNORM, 16, [4, 4])),
        (178, _) => Ok((DXGI_FORMAT_BC5_SNORM, 16, [4, 4])),
        (179, _) => Ok((DXGI_FORMAT_BC6H_UF16, 16, [4, 4])),
        (180, _) => Ok((DXGI_FORMAT_BC6H_SF16, 16, [4, 4])),
        (181, _) => Ok((DXGI_FORMAT_BC7_UNORM, 16, [4, 4])),
        (182, _) => Ok((DXGI_FORMAT_BC7_UNORM_SRGB, 16, [4, 4])),
        _ => Err(format!("unsupported GFX10 image format {data}/{number}")),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DecodedTexture, copy_linear_to_mip_level, copy_mip_level_to_linear, detile_layer,
        mip_chain_layout, mode24_offset, retile_layer,
    };
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32_FLOAT,
    };

    #[test]
    fn mode27_round_trips_a_pitched_color_surface() {
        let descriptor = DecodedTexture {
            address: 0,
            width: 137,
            height: 131,
            pitch: 256,
            tile_mode: 27,
            format: DXGI_FORMAT_R8G8B8A8_UNORM,
            bytes_per_element: 4,
            block_extent: [1, 1],
            texture_type: 9,
            layer_count: 1,
            mip_level_count: 1,
            writable: false,
        };
        let linear = (0..descriptor.linear_byte_count().unwrap())
            .map(|index| (index.wrapping_mul(29) & 0xff) as u8)
            .collect::<Vec<_>>();
        let mut tiled = vec![0_u8; descriptor.raw_layer_byte_count().unwrap()];
        retile_layer(descriptor, 0, &linear, &mut tiled).unwrap();
        let mut actual = vec![0_u8; linear.len()];
        detile_layer(descriptor, 0, &tiled, &mut actual).unwrap();
        assert_eq!(actual, linear);
    }

    #[test]
    fn mode24_round_trips_depth_and_matches_known_offsets() {
        assert_eq!(mode24_offset(0, 0, 4), 0);
        assert_eq!(mode24_offset(1, 0, 4), 4);
        assert_eq!(mode24_offset(0, 1, 4), 8);
        assert_eq!(mode24_offset(2, 0, 4), 16);
        assert_eq!(mode24_offset(0, 2, 4), 32);
        let descriptor = DecodedTexture {
            address: 0,
            width: 257,
            height: 257,
            pitch: 384,
            tile_mode: 24,
            format: DXGI_FORMAT_R32_FLOAT,
            bytes_per_element: 4,
            block_extent: [1, 1],
            texture_type: 9,
            layer_count: 1,
            mip_level_count: 1,
            writable: true,
        };
        let linear = (0..descriptor.linear_byte_count().unwrap())
            .map(|index| (index.wrapping_mul(31) & 0xff) as u8)
            .collect::<Vec<_>>();
        let mut tiled = vec![0_u8; descriptor.raw_layer_byte_count().unwrap()];
        retile_layer(descriptor, 0, &linear, &mut tiled).unwrap();
        let mut actual = vec![0_u8; linear.len()];
        detile_layer(descriptor, 0, &tiled, &mut actual).unwrap();
        assert_eq!(actual, linear);
    }

    #[test]
    fn mode5_and_mode27_mip_chains_round_trip_nontail_and_tail_levels() {
        for (mode, width, height, expected_bytes, expected_base_offset) in [
            (5, 512, 512, 1_400_832, 352_256),
            (27, 512, 256, 786_432, 262_144),
        ] {
            let descriptor = DecodedTexture {
                address: 0,
                width,
                height,
                pitch: width,
                tile_mode: mode,
                format: DXGI_FORMAT_R8G8B8A8_UNORM,
                bytes_per_element: 4,
                block_extent: [1, 1],
                texture_type: 9,
                layer_count: 1,
                mip_level_count: 10,
                writable: false,
            };
            let layout = mip_chain_layout(descriptor).expect("complete GFX10 mip layout");
            assert_eq!(layout.byte_count, expected_bytes);
            assert_eq!(layout.levels[0].offset, expected_base_offset);
            assert!(layout.levels.iter().any(|level| level.in_tail));
            assert!(layout.levels.iter().any(|level| !level.in_tail));
            let levels = layout
                .levels
                .iter()
                .enumerate()
                .map(|(level_index, level)| {
                    (0..level.width * level.height * layout.bytes_per_element)
                        .map(|byte| ((level_index * 41 + byte * 17 + byte / 251) & 0xff) as u8)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut raw = vec![0x5a; layout.byte_count];
            for (&level, linear) in layout.levels.iter().zip(&levels) {
                copy_linear_to_mip_level(&layout, level, linear, &mut raw)
                    .expect("encode GFX10 mip level");
            }
            for (&level, expected) in layout.levels.iter().zip(&levels) {
                let mut actual = vec![0_u8; expected.len()];
                copy_mip_level_to_linear(&layout, level, &raw, &mut actual)
                    .expect("decode GFX10 mip level");
                assert_eq!(&actual, expected);
            }
        }
    }
}
