// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use metal::MTLPixelFormat;

use super::{
    GuestTextureDescriptor, decode_mode5_mip_chain_from_raw, decode_mode27_mip_chain_from_raw,
    encode_mode5_mip_chain_into_raw, encode_mode27_mip_chain_into_raw, guest_texture_byte_count,
    mode5_mip_chain_layout, mode5_within_block_byte_offset, mode9_within_block_byte_offset,
    mode24_within_block_byte_offset_for_format, mode27_block_dimensions, mode27_mip_chain_layout,
    mode27_source_byte_offset, mode27_within_block_byte_offset,
    mode27_within_block_byte_offset_for_format, snapshot_guest_texture_linear,
    tiled_block_dimensions, write_guest_texture_linear,
};

#[test]
fn sdk_npot_mip_rounding_uses_floor_for_visible_and_shift_ceil_for_padding() {
    // B2 precondition settled from SDK agc_gpu_address.cpp:
    // - computeWidthInElementsForMip (:529-532) floors with `>>`
    // - mipWidthCeil / first-tail entry (:565-566, :735-737) use shiftCeil (:62-64)
    // - mode-27 already matched; mode-5 must match the same split
    let mode5 = GuestTextureDescriptor {
        address: 0x70_2754_1000,
        width: 513,
        height: 257,
        pitch: 513,
        tile_mode: 5,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let mode5_layout = mode5_mip_chain_layout(mode5, 4).expect("mode-5 NPOT layout");
    assert_eq!(mode5_layout.levels[0].width, 513);
    assert_eq!(mode5_layout.levels[0].height, 257);
    assert_eq!(mode5_layout.levels[1].width, 256); // floor(513>>1), not ceil 257
    assert_eq!(mode5_layout.levels[1].height, 128); // floor(257>>1), not ceil 129
    assert_eq!(mode5_layout.levels[1].padded_width % mode5_layout.block_width, 0);
    // Padding is derived from shiftCeil, so it is at least the ceil extent.
    assert!(mode5_layout.levels[1].padded_width >= 257);

    let mode27 = GuestTextureDescriptor {
        tile_mode: 27,
        pitch: 513,
        ..mode5
    };
    let mode27_layout = mode27_mip_chain_layout(mode27, 4).expect("mode-27 NPOT layout");
    assert_eq!(mode27_layout.levels[1].width, 256);
    assert_eq!(mode27_layout.levels[1].height, 128);
    assert_eq!(
        mode5_layout.levels[1].width, mode27_layout.levels[1].width,
        "mode-5 and mode-27 must agree on visible NPOT dims"
    );
}

#[test]
fn superliminal_mode5_mip_chains_place_the_base_image_after_smaller_levels() {
    let rgba8 = GuestTextureDescriptor {
        address: 0x70_2754_1000,
        width: 512,
        height: 512,
        pitch: 512,
        tile_mode: 5,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let rgba8_layout = mode5_mip_chain_layout(rgba8, 10).expect("RGBA8 mode-5 mip layout");
    assert_eq!(rgba8_layout.byte_count, 1_400_832);
    assert_eq!(rgba8_layout.levels[0].offset, 352_256);
    assert_eq!(rgba8_layout.levels[4].offset, 4_096);
    assert!(rgba8_layout.levels[5..].iter().all(|level| level.in_tail));
    assert_eq!(
        rgba8_layout.levels[5..]
            .iter()
            .map(|level| (level.tail_x, level.tail_y))
            .collect::<Vec<_>>(),
        [(16, 0), (8, 16), (0, 24), (0, 16), (8, 8)]
    );

    let rgba16_float = GuestTextureDescriptor {
        address: 0x70_2968_1000,
        pixel_format: MTLPixelFormat::RGBA16Float,
        bytes_per_pixel: 8,
        ..rgba8
    };
    let rgba16_layout =
        mode5_mip_chain_layout(rgba16_float, 10).expect("RGBA16F mode-5 mip layout");
    assert_eq!(rgba16_layout.byte_count, 2_797_568);
    assert_eq!(rgba16_layout.levels[0].offset, 700_416);
    assert_eq!(rgba16_layout.levels[4].offset, 4_096);
    assert!(rgba16_layout.levels[5..].iter().all(|level| level.in_tail));
}

#[test]
fn mode5_mip_chain_round_trips_nontail_levels_and_the_tail() {
    let descriptor = GuestTextureDescriptor {
        address: 0x70_2754_1000,
        width: 512,
        height: 512,
        pitch: 512,
        tile_mode: 5,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let layout = mode5_mip_chain_layout(descriptor, 10).expect("complete mode-5 mip chain");
    let levels = layout
        .levels
        .iter()
        .enumerate()
        .map(|(level_index, level)| {
            (0..level.width * level.height * layout.bytes_per_element)
                .map(|byte| {
                    u8::try_from((level_index * 43 + byte * 19 + byte / 257) & 0xff)
                        .expect("masked byte fits")
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut raw = vec![0x5a; layout.byte_count];
    encode_mode5_mip_chain_into_raw(&layout, &levels, &mut raw)
        .expect("encode complete mode-5 mip chain");
    let (decoded_layout, decoded) = decode_mode5_mip_chain_from_raw(descriptor, 10, &raw)
        .expect("decode complete mode-5 mip chain");
    assert_eq!(decoded_layout, layout);
    assert_eq!(decoded, levels);
}

#[test]
fn silent_hill_mode27_mip_tail_round_trips_every_level() {
    let descriptor = GuestTextureDescriptor {
        address: 0x80b9_2700_00,
        width: 32,
        height: 32,
        pitch: 32,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RG16Uint,
        bytes_per_pixel: 4,
    };
    let layout = mode27_mip_chain_layout(descriptor, 6).expect("complete mip-tail layout");
    assert_eq!(layout.byte_count, 65_536);
    assert!(layout.levels.iter().all(|level| level.in_tail));
    assert_eq!(
        layout
            .levels
            .iter()
            .map(|level| (level.width, level.height, level.tail_x, level.tail_y))
            .collect::<Vec<_>>(),
        vec![
            (32, 32, 64, 0),
            (16, 16, 0, 64),
            (8, 8, 32, 0),
            (4, 4, 0, 32),
            (2, 2, 16, 0),
            (1, 1, 8, 16),
        ]
    );

    let levels = layout
        .levels
        .iter()
        .enumerate()
        .map(|(index, level)| {
            let byte_count = level.width * level.height * layout.bytes_per_element;
            (0..byte_count)
                .map(|byte| {
                    u8::try_from((index * 37 + byte) & 0xff).expect("masked test byte fits")
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut raw = vec![0xa5; layout.byte_count];
    encode_mode27_mip_chain_into_raw(&layout, &levels, &mut raw).expect("encode exact mip tail");
    let (decoded_layout, decoded) =
        decode_mode27_mip_chain_from_raw(descriptor, 6, &raw).expect("decode exact mip tail");
    assert_eq!(decoded_layout, layout);
    assert_eq!(decoded, levels);
}

#[test]
fn mode27_mip_chain_round_trips_multiple_nontail_blocks_and_tail() {
    let descriptor = GuestTextureDescriptor {
        address: 0x80b9_2800_00,
        width: 512,
        height: 256,
        pitch: 512,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let layout = mode27_mip_chain_layout(descriptor, 10).expect("complete mip-chain layout");
    assert!(layout.levels.iter().any(|level| !level.in_tail));
    assert!(layout.levels.iter().any(|level| level.in_tail));
    let levels = layout
        .levels
        .iter()
        .enumerate()
        .map(|(level_index, level)| {
            (0..level.width * level.height * layout.bytes_per_element)
                .map(|byte| {
                    u8::try_from((level_index * 41 + byte * 17 + byte / 251) & 0xff)
                        .expect("masked test byte fits")
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut raw = vec![0x5a; layout.byte_count];
    encode_mode27_mip_chain_into_raw(&layout, &levels, &mut raw)
        .expect("encode complete mip chain");
    let (decoded_layout, decoded) =
        decode_mode27_mip_chain_from_raw(descriptor, 10, &raw).expect("decode complete mip chain");
    assert_eq!(decoded_layout, layout);
    assert_eq!(decoded, levels);
}

#[test]
fn mode27_lookup_matches_the_scalar_address_equation_across_blocks() {
    let width = 385_usize;
    for y in 0..257 {
        for x in 0..width {
            let expected = (y / 128 * width.div_ceil(128) + x / 128) * 65_536
                + mode27_within_block_byte_offset(x, y);
            assert_eq!(mode27_source_byte_offset(x, y, width), expected);
        }
    }
}

#[test]
fn mode24_uses_the_exact_gfx10_z_order_low_address_bits() {
    assert_eq!(mode24_within_block_byte_offset_for_format(0, 0, 4), 0);
    assert_eq!(mode24_within_block_byte_offset_for_format(1, 0, 4), 4);
    assert_eq!(mode24_within_block_byte_offset_for_format(0, 1, 4), 8);
    assert_eq!(mode24_within_block_byte_offset_for_format(2, 0, 4), 16);
    assert_eq!(mode24_within_block_byte_offset_for_format(0, 2, 4), 32);
    assert_eq!(mode24_within_block_byte_offset_for_format(3, 5, 1), 0x27);
    assert_eq!(mode24_within_block_byte_offset_for_format(3, 5, 2), 0x4e);
    assert_eq!(mode24_within_block_byte_offset_for_format(3, 5, 4), 0x9c);
}

#[test]
fn mode27_uses_the_current_gfx10_render_target_address_equation() {
    assert_eq!(mode27_within_block_byte_offset_for_format(0, 0, 4), 0);
    assert_eq!(mode27_within_block_byte_offset_for_format(1, 0, 4), 4);
    assert_eq!(mode27_within_block_byte_offset_for_format(2, 0, 4), 8);
    assert_eq!(mode27_within_block_byte_offset_for_format(0, 1, 4), 16);
    assert_eq!(mode27_within_block_byte_offset_for_format(0, 2, 4), 32);
    assert_eq!(mode27_within_block_byte_offset_for_format(4, 0, 4), 128);
    assert_eq!(super::mode27_surface_z_xor(0), 0);
    assert_eq!(super::mode27_surface_z_xor(1), 0x800);
    assert_eq!(super::mode27_surface_z_xor(2), 0x400);
    assert_eq!(super::mode27_surface_z_xor(4), 0x200);
    assert_eq!(super::mode27_surface_z_xor(8), 0x100);
    assert_eq!(super::mode27_surface_z_xor(15), 0xf00);
}

#[test]
fn mode27_volume_detile_applies_each_surface_z_xor() {
    let layer_count = 32_u32;
    let descriptor = GuestTextureDescriptor {
        address: 0x1_0000,
        width: 32,
        height: 32,
        pitch: 32,
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGB10A2Unorm,
        bytes_per_pixel: 4,
    };
    let face_bytes = usize::try_from(guest_texture_byte_count(descriptor).expect("face size fits"))
        .expect("face size fits usize");
    assert_eq!(face_bytes, 65_536);
    let tight_face_bytes = 32 * 32 * 4;
    let mut expected = Vec::with_capacity(tight_face_bytes * layer_count as usize);
    let mut raw = vec![0_u8; face_bytes * layer_count as usize];
    for z in 0..layer_count as usize {
        let mut linear = vec![0_u8; tight_face_bytes];
        for (index, texel) in linear.chunks_exact_mut(4).enumerate() {
            texel.copy_from_slice(
                &(u32::try_from(z).expect("z fits") << 24
                    | u32::try_from(index).expect("texel index fits"))
                .to_ne_bytes(),
            );
        }
        expected.extend_from_slice(&linear);
        let raw_face = &mut raw[z * face_bytes..(z + 1) * face_bytes];
        super::copy_linear_to_tiled_at_z(&linear, descriptor, raw_face, z)
            .expect("volume slice tiles with its surface-Z parity");
    }
    let decoded = super::decode_sampled_texture_linear_from_raw(
        super::DecodedImageDescriptor {
            guest: descriptor,
            span: super::ResourceSpan::for_texture(descriptor),
            destination_select: 0xfac,
            base_mip_level: 0,
            mip_level_count: 1,
            texture_type: 10,
            layer_count,
        },
        &raw,
    )
    .expect("volume detiles");
    assert_eq!(decoded, expected);
}

#[test]
fn mode24_guest_copy_round_trips_partial_and_parity_blocks() {
    let width = 257_usize;
    let height = 257_usize;
    let mut descriptor = GuestTextureDescriptor {
        address: 0x1_0000,
        width: u32::try_from(width).expect("width fits u32"),
        height: u32::try_from(height).expect("height fits u32"),
        pitch: u32::try_from(width).expect("pitch fits u32"),
        tile_mode: 24,
        pixel_format: MTLPixelFormat::R32Float,
        bytes_per_pixel: 4,
    };
    let tiled_length =
        usize::try_from(guest_texture_byte_count(descriptor).expect("tiled byte count fits u64"))
            .expect("tiled byte count fits usize");
    let mut tiled = vec![0_u8; tiled_length];
    descriptor.address =
        u64::try_from(tiled.as_mut_ptr() as usize).expect("test allocation address fits u64");
    let linear = (0..width * height * 4)
        .map(|index| u8::try_from(index & 0xff).expect("masked byte fits u8"))
        .collect::<Vec<_>>();

    write_guest_texture_linear(descriptor, &linear).expect("mode 24 write succeeds");
    assert_eq!(
        snapshot_guest_texture_linear(descriptor).expect("mode 24 snapshot succeeds"),
        linear
    );
}

#[test]
fn mode9_uses_the_exact_standard_64k_address_bits() {
    assert_eq!(mode9_within_block_byte_offset(0, 0, 4), 0);
    assert_eq!(mode9_within_block_byte_offset(1, 0, 4), 4);
    assert_eq!(mode9_within_block_byte_offset(2, 0, 4), 8);
    assert_eq!(mode9_within_block_byte_offset(0, 1, 4), 16);
    assert_eq!(mode9_within_block_byte_offset(0, 2, 4), 32);
    assert_eq!(mode9_within_block_byte_offset(4, 0, 4), 128);
    assert_eq!(mode9_within_block_byte_offset(127, 127, 4), 65_532);
}

#[test]
fn mode9_round_trips_every_supported_element_size() {
    for (bytes_per_element, pixel_format) in [
        (1, MTLPixelFormat::R8Unorm),
        (2, MTLPixelFormat::RG8Unorm),
        (4, MTLPixelFormat::RGBA8Unorm),
        (8, MTLPixelFormat::RGBA16Float),
        (16, MTLPixelFormat::RGBA32Float),
    ] {
        let (block_width, block_height) = tiled_block_dimensions(65_536, bytes_per_element)
            .expect("element size has a mode 9 equation");
        let width = block_width + 1;
        let height = block_height + 1;
        let mut descriptor = GuestTextureDescriptor {
            address: 0x1_0000,
            width: u32::try_from(width).expect("test width fits u32"),
            height: u32::try_from(height).expect("test height fits u32"),
            pitch: u32::try_from(width).expect("test pitch fits u32"),
            tile_mode: 9,
            pixel_format,
            bytes_per_pixel: u32::try_from(bytes_per_element)
                .expect("test bytes per element fits u32"),
        };
        let tiled_length = usize::try_from(
            guest_texture_byte_count(descriptor).expect("tiled byte count fits u64"),
        )
        .expect("tiled byte count fits usize");
        let mut tiled = vec![0_u8; tiled_length];
        descriptor.address =
            u64::try_from(tiled.as_mut_ptr() as usize).expect("test allocation address fits u64");
        let linear = (0..width * height * bytes_per_element)
            .map(|index| u8::try_from(index & 0xff).expect("masked byte fits u8"))
            .collect::<Vec<_>>();
        write_guest_texture_linear(descriptor, &linear).expect("mode 9 write succeeds");
        assert_eq!(
            snapshot_guest_texture_linear(descriptor).expect("mode 9 snapshot succeeds"),
            linear,
            "bytes per element {bytes_per_element}",
        );
    }
}

#[test]
fn bulk_mode27_guest_copy_round_trips_partial_and_parity_blocks() {
    let width = 385_usize;
    let height = 257_usize;
    let mut tiled = vec![0_u8; width.div_ceil(128) * height.div_ceil(128) * 65_536];
    let descriptor = GuestTextureDescriptor {
        address: u64::try_from(tiled.as_mut_ptr() as usize).expect("test address fits u64"),
        width: u32::try_from(width).expect("width fits u32"),
        height: u32::try_from(height).expect("height fits u32"),
        pitch: u32::try_from(width).expect("pitch fits u32"),
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let mut linear = vec![0_u8; width * height * 4];
    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 4;
            linear[offset..offset + 4].copy_from_slice(&[
                u8::try_from(x & 0xff).expect("masked X fits u8"),
                u8::try_from(y & 0xff).expect("masked Y fits u8"),
                u8::try_from((x ^ y) & 0xff).expect("masked xor fits u8"),
                0xff,
            ]);
        }
    }

    write_guest_texture_linear(descriptor, &linear).expect("mode 27 write succeeds");
    assert_eq!(
        snapshot_guest_texture_linear(descriptor).expect("mode 27 snapshot succeeds"),
        linear
    );
}

#[test]
fn tiled_guest_copy_uses_physical_pitch_between_block_rows() {
    let width = 128_usize;
    let height = 129_usize;
    let pitch = 256_usize;
    let mut descriptor = GuestTextureDescriptor {
        address: 0x1_0000,
        width: u32::try_from(width).expect("width fits u32"),
        height: u32::try_from(height).expect("height fits u32"),
        pitch: u32::try_from(pitch).expect("pitch fits u32"),
        tile_mode: 27,
        pixel_format: MTLPixelFormat::RGBA8Unorm,
        bytes_per_pixel: 4,
    };
    let tiled_length =
        usize::try_from(guest_texture_byte_count(descriptor).expect("tiled byte count fits u64"))
            .expect("tiled byte count fits usize");
    assert_eq!(tiled_length, 4 * 65_536);
    let mut tiled = vec![0_u8; tiled_length];
    descriptor.address = u64::try_from(tiled.as_mut_ptr() as usize).expect("test address fits u64");
    let linear = (0..width * height * 4)
        .map(|index| u8::try_from(index & 0xff).expect("masked byte fits u8"))
        .collect::<Vec<_>>();

    write_guest_texture_linear(descriptor, &linear).expect("padded tile write succeeds");
    assert!(tiled[65_536..2 * 65_536].iter().all(|&byte| byte == 0));
    assert_eq!(
        snapshot_guest_texture_linear(descriptor).expect("padded tile snapshot succeeds"),
        linear
    );
}

#[test]
fn mode27_round_trips_every_supported_render_target_element_size() {
    for (bytes_per_pixel, pixel_format) in [
        (1, MTLPixelFormat::R8Unorm),
        (2, MTLPixelFormat::RG8Unorm),
        (4, MTLPixelFormat::RGBA8Unorm),
        (8, MTLPixelFormat::RGBA16Float),
        (16, MTLPixelFormat::RGBA32Float),
    ] {
        let (block_width, block_height) = mode27_block_dimensions(bytes_per_pixel)
            .expect("render-target element size has a mode 27 equation");
        let width = block_width + 1;
        let height = block_height + 1;
        let mut descriptor = GuestTextureDescriptor {
            address: 0x1_0000,
            width: u32::try_from(width).expect("test width fits u32"),
            height: u32::try_from(height).expect("test height fits u32"),
            pitch: u32::try_from(width).expect("test pitch fits u32"),
            tile_mode: 27,
            pixel_format,
            bytes_per_pixel: u32::try_from(bytes_per_pixel).expect("test bytes per pixel fits u32"),
        };
        let tiled_length = usize::try_from(
            guest_texture_byte_count(descriptor).expect("tiled byte count fits u64"),
        )
        .expect("tiled byte count fits usize");
        let mut tiled = vec![0_u8; tiled_length];
        descriptor.address =
            u64::try_from(tiled.as_mut_ptr() as usize).expect("test allocation address fits u64");
        let linear_length = width * height * bytes_per_pixel;
        let linear = (0..linear_length)
            .map(|index| u8::try_from(index & 0xff).expect("masked byte fits u8"))
            .collect::<Vec<_>>();
        write_guest_texture_linear(descriptor, &linear).expect("mode 27 write succeeds");
        assert_eq!(
            snapshot_guest_texture_linear(descriptor).expect("mode 27 snapshot succeeds"),
            linear,
            "bytes per pixel {bytes_per_pixel}",
        );
    }
}

#[test]
fn mode5_uses_the_exact_standard_4k_address_bits() {
    assert_eq!(mode5_within_block_byte_offset(0, 0, 4), 0);
    assert_eq!(mode5_within_block_byte_offset(1, 0, 4), 4);
    assert_eq!(mode5_within_block_byte_offset(2, 0, 4), 8);
    assert_eq!(mode5_within_block_byte_offset(0, 1, 4), 16);
    assert_eq!(mode5_within_block_byte_offset(0, 4, 4), 64);
    assert_eq!(mode5_within_block_byte_offset(4, 0, 4), 128);
}

#[test]
fn mode5_matches_the_official_gfx10_standard_4k_equation_for_every_block_texel() {
    fn reference(x: usize, y: usize, bytes_per_element: usize) -> usize {
        // Scalar expansion of AMD AddrLib's GFX10_SW_4K_S pattern. Keeping this
        // structurally separate catches swapped X/Y channels and shifted bits.
        match bytes_per_element {
            1 => {
                ((y << 4) & 0x1f0)
                    ^ ((y << 5) & 0x400)
                    ^ (x & 0x00f)
                    ^ ((x << 5) & 0x200)
                    ^ ((x << 6) & 0x800)
            }
            2 => {
                ((y << 4) & 0x070)
                    ^ ((y << 5) & 0x100)
                    ^ ((y << 6) & 0x400)
                    ^ ((x << 1) & 0x00e)
                    ^ ((x << 4) & 0x080)
                    ^ ((x << 5) & 0x200)
                    ^ ((x << 6) & 0x800)
            }
            4 => {
                ((y << 4) & 0x070)
                    ^ ((y << 5) & 0x100)
                    ^ ((y << 6) & 0x400)
                    ^ ((x << 2) & 0x00c)
                    ^ ((x << 5) & 0x080)
                    ^ ((x << 6) & 0x200)
                    ^ ((x << 7) & 0x800)
            }
            8 => {
                ((y << 4) & 0x030)
                    ^ ((y << 6) & 0x100)
                    ^ ((y << 7) & 0x400)
                    ^ ((x << 3) & 0x008)
                    ^ ((x << 5) & 0x0c0)
                    ^ ((x << 6) & 0x200)
                    ^ ((x << 7) & 0x800)
            }
            16 => {
                ((y << 4) & 0x030)
                    ^ ((y << 6) & 0x100)
                    ^ ((y << 7) & 0x400)
                    ^ ((x << 6) & 0x0c0)
                    ^ ((x << 7) & 0x200)
                    ^ ((x << 8) & 0x800)
            }
            _ => unreachable!("test covers supported element sizes"),
        }
    }

    for bytes_per_element in [1, 2, 4, 8, 16] {
        let (width, height) = tiled_block_dimensions(4_096, bytes_per_element)
            .expect("supported mode-5 element size");
        for y in 0..height {
            for x in 0..width {
                assert_eq!(
                    mode5_within_block_byte_offset(x, y, bytes_per_element),
                    reference(x, y, bytes_per_element),
                    "element size {bytes_per_element} at ({x},{y})",
                );
            }
        }
    }
}

#[test]
fn mode1_round_trips_every_supported_standard_element_size() {
    for (bytes_per_element, pixel_format) in [
        (1, MTLPixelFormat::R8Unorm),
        (2, MTLPixelFormat::RG8Unorm),
        (4, MTLPixelFormat::RGBA8Unorm),
        (8, MTLPixelFormat::RGBA16Float),
        (16, MTLPixelFormat::RGBA32Float),
    ] {
        let (block_width, block_height) = tiled_block_dimensions(256, bytes_per_element)
            .expect("element size has a mode 1 equation");
        let width = block_width + 1;
        let height = block_height + 1;
        let mut descriptor = GuestTextureDescriptor {
            address: 0x1_0000,
            width: u32::try_from(width).expect("test width fits u32"),
            height: u32::try_from(height).expect("test height fits u32"),
            pitch: u32::try_from(width).expect("test pitch fits u32"),
            tile_mode: 1,
            pixel_format,
            bytes_per_pixel: u32::try_from(bytes_per_element)
                .expect("test bytes per element fits u32"),
        };
        let tiled_length = usize::try_from(
            guest_texture_byte_count(descriptor).expect("tiled byte count fits u64"),
        )
        .expect("tiled byte count fits usize");
        let mut tiled = vec![0_u8; tiled_length];
        descriptor.address =
            u64::try_from(tiled.as_mut_ptr() as usize).expect("test allocation address fits u64");
        let linear = (0..width * height * bytes_per_element)
            .map(|index| u8::try_from(index & 0xff).expect("masked byte fits u8"))
            .collect::<Vec<_>>();

        write_guest_texture_linear(descriptor, &linear).expect("mode 1 write succeeds");
        assert_eq!(
            snapshot_guest_texture_linear(descriptor).expect("mode 1 snapshot succeeds"),
            linear,
            "bytes per element {bytes_per_element}",
        );
    }
}

#[test]
fn mode5_round_trips_uncompressed_and_bc_element_grids() {
    for (bytes_per_element, pixel_format, block_compressed) in [
        (1, MTLPixelFormat::R8Unorm, false),
        (2, MTLPixelFormat::RG8Unorm, false),
        (4, MTLPixelFormat::RGBA8Unorm, false),
        (8, MTLPixelFormat::RGBA16Float, false),
        (16, MTLPixelFormat::RGBA32Float, false),
        (16, MTLPixelFormat::BC7_RGBAUnorm_sRGB, true),
    ] {
        let (block_width, block_height) = tiled_block_dimensions(4_096, bytes_per_element)
            .expect("element size has a mode 5 equation");
        let elements_wide = block_width + 1;
        let elements_high = block_height + 1;
        let width = if block_compressed {
            elements_wide * 4
        } else {
            elements_wide
        };
        let height = if block_compressed {
            elements_high * 4
        } else {
            elements_high
        };
        let mut descriptor = GuestTextureDescriptor {
            address: 0x1_0000,
            width: u32::try_from(width).expect("test width fits u32"),
            height: u32::try_from(height).expect("test height fits u32"),
            pitch: u32::try_from(width).expect("test pitch fits u32"),
            tile_mode: 5,
            pixel_format,
            bytes_per_pixel: u32::try_from(bytes_per_element).expect("test element size fits u32"),
        };
        let tiled_length = usize::try_from(
            guest_texture_byte_count(descriptor).expect("tiled byte count fits u64"),
        )
        .expect("tiled byte count fits usize");
        let mut tiled = vec![0_u8; tiled_length];
        descriptor.address =
            u64::try_from(tiled.as_mut_ptr() as usize).expect("test allocation address fits u64");
        let linear_length = elements_wide * elements_high * bytes_per_element;
        let linear = (0..linear_length)
            .map(|index| u8::try_from(index & 0xff).expect("masked byte fits u8"))
            .collect::<Vec<_>>();

        write_guest_texture_linear(descriptor, &linear).expect("mode 5 write succeeds");
        assert_eq!(
            snapshot_guest_texture_linear(descriptor).expect("mode 5 snapshot succeeds"),
            linear,
            "element size {bytes_per_element}, compressed={block_compressed}",
        );
    }
}
