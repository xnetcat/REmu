// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    DecodedImageDescriptor, GuestTextureDescriptor, MIP_UPLOAD_BYTES, MIP_UPLOAD_COUNT,
    MIP_UPLOAD_NANOSECONDS, Mode27WorkOrigin, ResidentGuestTexture, WriteWatchRefresh,
    decode_mode27_mip_chain_from_raw, mode27_mip_chain_layout, record_mode27_work_detail,
    record_texture_work, refresh_resident_texture_write_watch, resident_texture_for_sample,
    snapshot_watched_guest_range, watch_resident_texture,
};
use metal::{
    Device, MTLRegion, MTLStorageMode, MTLTextureType, MTLTextureUsage, NSRange, Texture,
    TextureDescriptor,
};
use remu_shader::ImageBinding;
use std::collections::HashMap;
use std::ffi::c_void;
use std::time::Instant;

pub(crate) fn ensure_mipmapped_resident_texture_in_map(
    device: &Device,
    descriptor: DecodedImageDescriptor,
    resident_guest_textures: &mut HashMap<GuestTextureDescriptor, ResidentGuestTexture>,
    origin: Mode27WorkOrigin,
) -> Result<Texture, String> {
    if descriptor.texture_type != 9
        || descriptor.layer_count != 1
        || descriptor.guest.tile_mode != 27
        || descriptor.mip_level_count == 0
    {
        return Err(format!(
            "mipmapped storage image requires a nonempty 2D single-layer mode-27 resource, got type={} layers={} tile={} base={} count={}",
            descriptor.texture_type,
            descriptor.layer_count,
            descriptor.guest.tile_mode,
            descriptor.base_mip_level,
            descriptor.mip_level_count
        ));
    }
    let underlying_mip_count = descriptor
        .base_mip_level
        .checked_add(descriptor.mip_level_count)
        .ok_or("storage-image mip range overflows")?;
    if let Some(resident) = resident_guest_textures
        .get_mut(&descriptor.guest)
        .filter(|resident| {
            resident.texture_type == 9
                && resident.layer_count == 1
                && resident.mip_level_count >= underlying_mip_count
        })
    {
        match refresh_resident_texture_write_watch(resident)? {
            WriteWatchRefresh::Active
            | WriteWatchRefresh::Reprotecting
            | WriteWatchRefresh::Renewed { .. } => {
                return resident_texture_for_sample(resident, descriptor).ok_or_else(|| {
                    "resident mip hierarchy cannot represent storage view".to_owned()
                });
            }
            WriteWatchRefresh::Replaced => {}
        }
    }
    if resident_guest_textures
        .get(&descriptor.guest)
        .is_some_and(|resident| resident.gpu_written)
    {
        return Err(format!(
            "cannot reinterpret GPU-written image {:#x} as a {underlying_mip_count}-level hierarchy",
            descriptor.guest.address
        ));
    }
    resident_guest_textures.remove(&descriptor.guest);
    let started = Instant::now();
    let layout = mode27_mip_chain_layout(descriptor.guest, underlying_mip_count)?;
    let write_watch = watch_resident_texture(
        descriptor.guest,
        descriptor.texture_type,
        descriptor.layer_count,
        underlying_mip_count,
    )?;
    let raw =
        snapshot_watched_guest_range(descriptor.guest.address, layout.byte_count, &write_watch)?;
    let (_, linear_levels) =
        decode_mode27_mip_chain_from_raw(descriptor.guest, underlying_mip_count, &raw)?;
    let texture = create_mipmapped_guest_texture(
        device,
        descriptor.guest,
        underlying_mip_count,
        &linear_levels,
    )?;
    record_mode27_work_detail(
        origin,
        descriptor.guest,
        underlying_mip_count,
        layout.byte_count,
        started.elapsed(),
    );
    texture.set_label(&format!(
        "REMu {}-level storage image {:#x}",
        underlying_mip_count, descriptor.guest.address
    ));
    resident_guest_textures.insert(
        descriptor.guest,
        ResidentGuestTexture {
            texture,
            component_reorder_view: None,
            descriptor: descriptor.guest,
            texture_type: descriptor.texture_type,
            layer_count: descriptor.layer_count,
            mip_level_count: underlying_mip_count,
            write_watch,
            gpu_written: false,
            content_version: 0,
        },
    );
    resident_texture_for_sample(
        resident_guest_textures
            .get_mut(&descriptor.guest)
            .expect("inserted mipmapped resident exists"),
        descriptor,
    )
    .ok_or_else(|| "created mip hierarchy cannot represent storage view".to_owned())
}

pub(crate) fn create_mipmapped_guest_texture(
    device: &Device,
    guest: GuestTextureDescriptor,
    mip_level_count: u32,
    linear_levels: &[Vec<u8>],
) -> Result<Texture, String> {
    let started = Instant::now();
    let layout = mode27_mip_chain_layout(guest, mip_level_count)?;
    if linear_levels.len() != layout.levels.len() {
        return Err(format!(
            "Metal mip upload has {} levels, expected {}",
            linear_levels.len(),
            layout.levels.len()
        ));
    }
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(guest.pixel_format);
    descriptor.set_width(u64::from(guest.width));
    descriptor.set_height(u64::from(guest.height));
    descriptor.set_mipmap_level_count(u64::from(mip_level_count));
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(
        MTLTextureUsage::RenderTarget
            | MTLTextureUsage::ShaderRead
            | MTLTextureUsage::ShaderWrite
            | MTLTextureUsage::PixelFormatView,
    );
    let texture = device.new_texture(&descriptor);
    for (level_index, (&level, linear)) in layout.levels.iter().zip(linear_levels).enumerate() {
        let expected = level
            .width
            .checked_mul(level.height)
            .and_then(|elements| elements.checked_mul(layout.bytes_per_element))
            .ok_or("Metal mip upload size overflows")?;
        if linear.len() != expected {
            return Err(format!(
                "Metal mip {level_index} upload has {} bytes, expected {expected}",
                linear.len()
            ));
        }
        texture.replace_region(
            MTLRegion::new_2d(
                0,
                0,
                u64::try_from(level.width).expect("validated mip width fits u64"),
                u64::try_from(level.height).expect("validated mip height fits u64"),
            ),
            u64::try_from(level_index).expect("validated mip index fits u64"),
            linear.as_ptr().cast::<c_void>(),
            u64::try_from(level.width * layout.bytes_per_element)
                .expect("validated mip row size fits u64"),
        );
    }
    record_texture_work(
        &MIP_UPLOAD_COUNT,
        &MIP_UPLOAD_BYTES,
        &MIP_UPLOAD_NANOSECONDS,
        linear_levels.iter().map(Vec::len).sum(),
        started.elapsed(),
    );
    Ok(texture)
}

pub(crate) fn create_exact_mode27_sampled_mip_view(
    device: &Device,
    descriptor: DecodedImageDescriptor,
    raw: &[u8],
    comparison: bool,
) -> Result<Texture, String> {
    if comparison {
        return Err(format!(
            "exact comparison-sampled mip view base={} count={} is not implemented",
            descriptor.base_mip_level, descriptor.mip_level_count
        ));
    }
    if descriptor.texture_type != 9
        || descriptor.layer_count != 1
        || descriptor.guest.tile_mode != 27
        || descriptor.mip_level_count == 0
    {
        return Err(format!(
            "exact sampled mip view requires a nonempty 2D single-layer mode-27 resource, got type={} layers={} tile={} base={} count={}",
            descriptor.texture_type,
            descriptor.layer_count,
            descriptor.guest.tile_mode,
            descriptor.base_mip_level,
            descriptor.mip_level_count
        ));
    }
    let underlying_mip_count = descriptor
        .base_mip_level
        .checked_add(descriptor.mip_level_count)
        .ok_or("sampled mip range overflows")?;
    let started = Instant::now();
    let (_, linear_levels) =
        decode_mode27_mip_chain_from_raw(descriptor.guest, underlying_mip_count, raw)?;
    let texture = create_mipmapped_guest_texture(
        device,
        descriptor.guest,
        underlying_mip_count,
        &linear_levels,
    )?;
    record_mode27_work_detail(
        Mode27WorkOrigin::SampledCacheCreate,
        descriptor.guest,
        underlying_mip_count,
        raw.len(),
        started.elapsed(),
    );
    if descriptor.base_mip_level == 0 && descriptor.mip_level_count == underlying_mip_count {
        return Ok(texture);
    }
    Ok(texture.new_texture_view_from_slice(
        descriptor.guest.pixel_format,
        MTLTextureType::D2,
        NSRange::new(
            u64::from(descriptor.base_mip_level),
            u64::from(descriptor.mip_level_count),
        ),
        NSRange::new(0, 1),
    ))
}

pub(crate) fn storage_image_requires_mip_hierarchy(
    binding: &ImageBinding,
    descriptor: DecodedImageDescriptor,
) -> bool {
    // IMAGE_LOAD/STORE address the resource view's BASE_LEVEL. LAST_LEVEL only bounds the view
    // and does not make an ordinary base-zero access touch every exposed mip. Only the _MIP
    // variants consume a per-invocation mip coordinate. Avoid promoting a normal full-view UAV
    // to a complete CPU-decoded hierarchy; besides doing needless work, that incorrectly rejects
    // valid SW_64KB_S (mode 9) base-level resources.
    descriptor.base_mip_level != 0
        || binding.opcode.is("ImageLoadMip")
        || binding.opcode.is("ImageStoreMip")
}
