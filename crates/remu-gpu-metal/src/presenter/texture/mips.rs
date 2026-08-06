// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::GuestMipChainLayout;
use super::{
    DecodedImageDescriptor, GuestTextureDescriptor, MIP_UPLOAD_BYTES, MIP_UPLOAD_COUNT,
    MIP_UPLOAD_NANOSECONDS, Mode27WorkOrigin, RangeSet, ResidencyState, ResidentGuestTexture,
    ResidentTextureMap, WriteWatchRefresh, comparison_depth32_upload,
    compressed_texture_block_extent, copy_mode27_mip_level_to_linear,
    decode_mode5_mip_chain_from_raw, decode_mode27_mip_chain_from_raw, mode27_mip_chain_layout,
    record_mode27_work_detail, record_texture_work, refresh_resident_texture_write_watch,
    resident_texture_for_sample, sampled_texture_usage, snapshot_watched_guest_range,
    trace_resident_texture_removal, watch_resident_texture,
};
use metal::{
    CommandBufferRef, Device, MTLOrigin, MTLRegion, MTLSize, MTLStorageMode, MTLTextureType,
    MTLTextureUsage, NSRange, Texture, TextureDescriptor,
};
use remu_shader::ImageBinding;
use std::ffi::c_void;
use std::time::Instant;

pub(crate) fn ensure_mipmapped_resident_texture_in_map(
    device: &Device,
    queue: Option<&metal::CommandQueue>,
    promotion_command_buffer: Option<&CommandBufferRef>,
    descriptor: DecodedImageDescriptor,
    resident_guest_textures: &mut ResidentTextureMap,
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
                // A storage image is read as well as written, so a superseded hierarchy rebuilds.
                if resident.reclaim_for_bind(false) {
                    return resident_texture_for_sample(resident, descriptor).ok_or_else(|| {
                        "resident mip hierarchy cannot represent storage view".to_owned()
                    });
                }
            }
            WriteWatchRefresh::Replaced => {}
        }
    }
    if resident_guest_textures
        .get(&descriptor.guest)
        .is_some_and(|resident| resident.gpu_written())
    {
        return Err(format!(
            "cannot reinterpret GPU-written image {:#x} as a {underlying_mip_count}-level hierarchy",
            descriptor.guest.address
        ));
    }
    let clean_promotion = resident_guest_textures
        .get(&descriptor.guest)
        .is_some_and(|resident| {
            resident.is_live()
                && resident.texture_type == 9
                && resident.layer_count == 1
                && resident.mip_level_count < underlying_mip_count
        });
    if clean_promotion {
        let command_buffer = promotion_command_buffer.ok_or_else(|| {
            format!(
                "mip hierarchy growth for image {:#x} requires an ordered guest command buffer",
                descriptor.guest.address
            )
        })?;
        let resident = resident_guest_textures
            .remove(&descriptor.guest)
            .expect("selected clean mip resident exists");
        let promoted =
            promote_resident_mip_hierarchy(device, command_buffer, descriptor, resident, origin)?;
        resident_guest_textures.insert(descriptor.guest, promoted);
        return resident_texture_for_sample(
            resident_guest_textures
                .get_mut(&descriptor.guest)
                .expect("promoted clean mip resident exists"),
            descriptor,
        )
        .ok_or_else(|| "promoted clean mip hierarchy cannot represent storage view".to_owned());
    }
    if let Some(resident) = resident_guest_textures.remove(&descriptor.guest) {
        trace_resident_texture_removal("mip-hierarchy-replacement", &resident);
    }
    let started = Instant::now();
    let layout = mode27_mip_chain_layout(descriptor.guest, underlying_mip_count)?;
    let write_watch = watch_resident_texture(
        descriptor.guest,
        descriptor.texture_type,
        descriptor.layer_count,
        underlying_mip_count,
    )?;
    if !crate::presenter::tiler::gpu_tiler_owns_descriptor(descriptor.guest) {
        return Err(format!(
            "Phase 1G: mipmapped storage refuses tile={} format={:?} — no GPU tiler ownership",
            descriptor.guest.tile_mode, descriptor.guest.pixel_format
        ));
    }
    let queue = queue.ok_or_else(|| {
        format!(
            "Phase 1G: mipmapped storage at {:#x} needs a Metal queue for GPU upload",
            descriptor.guest.address
        )
    })?;
    let usage = MTLTextureUsage::RenderTarget
        | MTLTextureUsage::ShaderRead
        | MTLTextureUsage::ShaderWrite
        | MTLTextureUsage::PixelFormatView;
    let metal_descriptor = TextureDescriptor::new();
    metal_descriptor.set_texture_type(MTLTextureType::D2);
    metal_descriptor.set_pixel_format(descriptor.guest.pixel_format);
    metal_descriptor.set_width(u64::from(descriptor.guest.width));
    metal_descriptor.set_height(u64::from(descriptor.guest.height));
    metal_descriptor.set_mipmap_level_count(u64::from(underlying_mip_count));
    metal_descriptor.set_storage_mode(MTLStorageMode::Shared);
    metal_descriptor.set_usage(usage);
    let texture = device.new_texture(&metal_descriptor);
    let upload_buffer = queue.new_command_buffer().to_owned();
    upload_buffer.set_label("REMu Phase 1G mipmapped storage GPU upload");
    crate::presenter::tiler::encode_gpu_upload_from_guest_view(
        device,
        &upload_buffer,
        descriptor.guest,
        &texture,
        crate::presenter::tiler::GpuTilerView {
            texture_type: 9,
            layer_count: 1,
            mip_level_count: underlying_mip_count,
        },
    )?;
    upload_buffer.commit();
    upload_buffer.wait_until_completed();
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
    let (span, owned) = ResidentGuestTexture::declared_authority(&write_watch);
    resident_guest_textures.insert(
        descriptor.guest,
        ResidentGuestTexture {
            texture: texture.clone(),
            component_reorder_view: None,
            descriptor: descriptor.guest,
            texture_type: descriptor.texture_type,
            layer_count: descriptor.layer_count,
            mip_level_count: underlying_mip_count,
            write_watch,
            span,
            owned,
            gpu_authored: RangeSet::default(),
            state: ResidencyState::Live,
            content_version: 0,
            gpu_write_watch_generations: Vec::new(),
            last_use: super::super::ResidentTextureActivityStamp::default(),
            activity: super::super::ResidentTextureActivity::enabled(),
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
    let layout = mode27_mip_chain_layout(guest, mip_level_count)?;
    create_mipmapped_guest_texture_from_layout(
        device,
        guest,
        &layout,
        0,
        linear_levels,
        MTLTextureUsage::RenderTarget
            | MTLTextureUsage::ShaderRead
            | MTLTextureUsage::ShaderWrite
            | MTLTextureUsage::PixelFormatView,
    )
}

pub(crate) fn promote_resident_mip_hierarchy(
    device: &Device,
    command_buffer: &CommandBufferRef,
    descriptor: DecodedImageDescriptor,
    resident: ResidentGuestTexture,
    origin: Mode27WorkOrigin,
) -> Result<ResidentGuestTexture, String> {
    let underlying_mip_count = descriptor
        .base_mip_level
        .checked_add(descriptor.mip_level_count)
        .ok_or("storage-image mip range overflows")?;
    if descriptor.texture_type != 9
        || descriptor.layer_count != 1
        || descriptor.guest.tile_mode != 27
        || underlying_mip_count <= resident.mip_level_count
        || resident.texture_type != 9
        || resident.layer_count != 1
    {
        return Err(format!(
            "mip promotion requires a larger 2D mode-27 hierarchy, got old={}/{} new={underlying_mip_count} type={}/{} layers={}/{} tile={}",
            resident.mip_level_count,
            resident.texture.mipmap_level_count(),
            resident.texture_type,
            descriptor.texture_type,
            resident.layer_count,
            descriptor.layer_count,
            descriptor.guest.tile_mode
        ));
    }

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
    let first_new_level = resident.mip_level_count;
    let mut linear_levels = Vec::with_capacity(
        usize::try_from(underlying_mip_count - first_new_level)
            .expect("validated mip delta fits usize"),
    );
    for level_index in first_new_level..underlying_mip_count {
        let level = layout.levels
            [usize::try_from(level_index).map_err(|_| "promoted mip index does not fit usize")?];
        let byte_count = level
            .width
            .checked_mul(level.height)
            .and_then(|elements| elements.checked_mul(layout.bytes_per_element))
            .ok_or("promoted mip linear size overflows")?;
        let mut linear = vec![0; byte_count];
        copy_mode27_mip_level_to_linear(&layout, level, &raw, &mut linear)?;
        linear_levels.push(linear);
    }
    let texture = create_mipmapped_guest_texture_from_layout(
        device,
        descriptor.guest,
        &layout,
        usize::try_from(first_new_level).map_err(|_| "first promoted mip does not fit usize")?,
        &linear_levels,
        MTLTextureUsage::RenderTarget
            | MTLTextureUsage::ShaderRead
            | MTLTextureUsage::ShaderWrite
            | MTLTextureUsage::PixelFormatView,
    )?;

    // Seed only newly declared levels from guest RAM, then preserve every existing logical level
    // with a GPU-to-GPU copy. This is required for GPU-written residents because RAM is stale, and
    // avoids repeatedly detiling and Apple-compressing the same clean levels while Unreal grows a
    // hierarchy one mip at a time.
    let blit = command_buffer.new_blit_command_encoder();
    blit.set_label("REMu preserve GPU-current mip levels");
    let preserved_mip_count = resident.mip_level_count.min(underlying_mip_count);
    for level in 0..preserved_mip_count {
        let width = u64::from(
            descriptor
                .guest
                .width
                .checked_shr(level)
                .unwrap_or(0)
                .max(1),
        );
        let height = u64::from(
            descriptor
                .guest
                .height
                .checked_shr(level)
                .unwrap_or(0)
                .max(1),
        );
        blit.copy_from_texture(
            &resident.texture,
            0,
            u64::from(level),
            MTLOrigin { x: 0, y: 0, z: 0 },
            MTLSize::new(width, height, 1),
            &texture,
            0,
            u64::from(level),
            MTLOrigin { x: 0, y: 0, z: 0 },
        );
    }
    blit.end_encoding();

    record_mode27_work_detail(
        origin,
        descriptor.guest,
        underlying_mip_count,
        layout.byte_count,
        started.elapsed(),
    );
    texture.set_label(&format!(
        "REMu promoted {}-level image {:#x}",
        underlying_mip_count, descriptor.guest.address
    ));
    let gpu_write_watch_generations = if resident.gpu_written() {
        write_watch.generations()
    } else {
        Vec::new()
    };
    // Adding a level moves every other level's offset inside the chain, so the previous runs name
    // the wrong bytes. The promotion preserves every level's pixels, so a live allocation stays the
    // authority for its whole grown range and owes exactly what it owed before, widened to that
    // range — which is the range the pre-B15 code published anyway. A demoted one answers for
    // nothing, and growing its shape does not give it any bytes back.
    let (span, declared) = ResidentGuestTexture::declared_authority(&write_watch);
    let (owned, gpu_authored) = match (resident.state, resident.gpu_written()) {
        (ResidencyState::Stale(_), _) => (RangeSet::default(), RangeSet::default()),
        (ResidencyState::Live, true) => (declared.clone(), declared),
        (ResidencyState::Live, false) => (declared, RangeSet::default()),
    };
    eprintln!(
        "[REMu][METAL] encoded ordered mip promotion image={:#x} {}x{} {:?} levels={}->{} gpu_current={}",
        descriptor.guest.address,
        descriptor.guest.width,
        descriptor.guest.height,
        descriptor.guest.pixel_format,
        resident.mip_level_count,
        underlying_mip_count,
        resident.gpu_written(),
    );
    Ok(ResidentGuestTexture {
        texture,
        component_reorder_view: None,
        descriptor: descriptor.guest,
        texture_type: descriptor.texture_type,
        layer_count: descriptor.layer_count,
        mip_level_count: underlying_mip_count,
        write_watch,
        span,
        owned,
        gpu_authored,
        state: resident.state,
        content_version: resident.content_version,
        gpu_write_watch_generations,
        last_use: resident.last_use,
        activity: resident.activity,
    })
}

fn create_mipmapped_guest_texture_from_layout(
    device: &Device,
    guest: GuestTextureDescriptor,
    layout: &GuestMipChainLayout,
    first_level: usize,
    linear_levels: &[Vec<u8>],
    usage: MTLTextureUsage,
) -> Result<Texture, String> {
    let started = Instant::now();
    let uploaded_level_end = first_level
        .checked_add(linear_levels.len())
        .ok_or("Metal mip upload range overflows")?;
    if uploaded_level_end > layout.levels.len() {
        return Err(format!(
            "Metal mip upload range {first_level}..{uploaded_level_end} exceeds {} levels",
            layout.levels.len()
        ));
    }
    let descriptor = TextureDescriptor::new();
    descriptor.set_texture_type(MTLTextureType::D2);
    descriptor.set_pixel_format(guest.pixel_format);
    descriptor.set_width(u64::from(guest.width));
    descriptor.set_height(u64::from(guest.height));
    descriptor.set_mipmap_level_count(
        u64::try_from(layout.levels.len()).expect("validated mip count fits u64"),
    );
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(usage);
    let texture = device.new_texture(&descriptor);
    for (level_index, linear) in (first_level..uploaded_level_end).zip(linear_levels) {
        let level = layout.levels[level_index];
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
                u64::from(
                    guest
                        .width
                        .checked_shr(
                            u32::try_from(level_index).expect("validated mip index fits u32"),
                        )
                        .unwrap_or(0)
                        .max(1),
                ),
                u64::from(
                    guest
                        .height
                        .checked_shr(
                            u32::try_from(level_index).expect("validated mip index fits u32"),
                        )
                        .unwrap_or(0)
                        .max(1),
                ),
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

fn comparison_pixel_format(format: metal::MTLPixelFormat) -> metal::MTLPixelFormat {
    match format {
        metal::MTLPixelFormat::R16Unorm => metal::MTLPixelFormat::Depth16Unorm,
        _ => metal::MTLPixelFormat::Depth32Float,
    }
}

fn create_exact_sampled_mip_texture(
    device: &Device,
    descriptor: DecodedImageDescriptor,
    layout: &GuestMipChainLayout,
    linear_layers: &[Vec<Vec<u8>>],
    comparison: bool,
    writable: bool,
) -> Result<Texture, String> {
    let expected_layers = usize::try_from(descriptor.layer_count)
        .map_err(|_| "sampled layer count does not fit usize")?;
    if linear_layers.len() != expected_layers
        || linear_layers
            .iter()
            .any(|levels| levels.len() != layout.levels.len())
    {
        return Err(format!(
            "exact sampled hierarchy has {} layers and {:?} mip counts, expected {expected_layers} layers and {} mips",
            linear_layers.len(),
            linear_layers.iter().map(Vec::len).collect::<Vec<_>>(),
            layout.levels.len()
        ));
    }
    let texture_type = if descriptor.texture_type == 9 {
        MTLTextureType::D2
    } else {
        MTLTextureType::D2Array
    };
    let pixel_format = if comparison {
        comparison_pixel_format(descriptor.guest.pixel_format)
    } else {
        descriptor.guest.pixel_format
    };
    let mut usage = sampled_texture_usage(comparison, writable)?;
    if !comparison && compressed_texture_block_extent(descriptor.guest.pixel_format).is_none() {
        usage |= MTLTextureUsage::PixelFormatView;
    }
    let metal_descriptor = TextureDescriptor::new();
    metal_descriptor.set_texture_type(texture_type);
    metal_descriptor.set_pixel_format(pixel_format);
    metal_descriptor.set_width(u64::from(descriptor.guest.width));
    metal_descriptor.set_height(u64::from(descriptor.guest.height));
    metal_descriptor.set_mipmap_level_count(
        u64::try_from(layout.levels.len()).expect("validated mip count fits u64"),
    );
    if texture_type == MTLTextureType::D2Array {
        metal_descriptor.set_array_length(u64::from(descriptor.layer_count));
    }
    metal_descriptor.set_storage_mode(MTLStorageMode::Shared);
    metal_descriptor.set_usage(usage);
    let texture = device.new_texture(&metal_descriptor);
    let started = Instant::now();
    let mut uploaded_bytes = 0_usize;
    for (layer_index, linear_levels) in linear_layers.iter().enumerate() {
        for (level_index, (&level, linear)) in layout.levels.iter().zip(linear_levels).enumerate() {
            let expected = level
                .width
                .checked_mul(level.height)
                .and_then(|elements| elements.checked_mul(layout.bytes_per_element))
                .ok_or("Metal exact mip upload size overflows")?;
            if linear.len() != expected {
                return Err(format!(
                    "Metal exact layer {layer_index} mip {level_index} upload has {} bytes, expected {expected}",
                    linear.len()
                ));
            }
            let shift = u32::try_from(level_index).expect("validated mip index fits u32");
            let width = descriptor
                .guest
                .width
                .checked_shr(shift)
                .unwrap_or(0)
                .max(1);
            let height = descriptor
                .guest
                .height
                .checked_shr(shift)
                .unwrap_or(0)
                .max(1);
            let mip_guest = GuestTextureDescriptor {
                width,
                height,
                pitch: width,
                tile_mode: 0,
                ..descriptor.guest
            };
            let converted = comparison
                .then(|| comparison_depth32_upload(mip_guest, linear))
                .transpose()?
                .flatten();
            let upload = converted.as_deref().unwrap_or(linear);
            let bytes_per_element = if converted.is_some() {
                std::mem::size_of::<f32>()
            } else {
                layout.bytes_per_element
            };
            let elements_wide = if converted.is_some()
                && compressed_texture_block_extent(descriptor.guest.pixel_format).is_some()
            {
                usize::try_from(width).expect("validated mip width fits usize")
            } else {
                level.width
            };
            let bytes_per_row = elements_wide
                .checked_mul(bytes_per_element)
                .ok_or("Metal exact mip row size overflows")?;
            let expected_upload = if comparison {
                usize::try_from(width)
                    .expect("validated mip width fits usize")
                    .checked_mul(usize::try_from(height).expect("validated mip height fits usize"))
                    .and_then(|pixels| {
                        pixels.checked_mul(if pixel_format == metal::MTLPixelFormat::Depth16Unorm {
                            2
                        } else {
                            4
                        })
                    })
                    .ok_or("Metal exact comparison mip destination size overflows")?
            } else {
                expected
            };
            if upload.len() != expected_upload {
                return Err(format!(
                    "Metal exact layer {layer_index} mip {level_index} converted upload has {} bytes, expected {expected_upload}",
                    upload.len()
                ));
            }
            let region = MTLRegion::new_2d(0, 0, u64::from(width), u64::from(height));
            let mip = u64::try_from(level_index).expect("validated mip index fits u64");
            if texture_type == MTLTextureType::D2Array {
                texture.replace_region_in_slice(
                    region,
                    mip,
                    u64::try_from(layer_index).expect("validated layer index fits u64"),
                    upload.as_ptr().cast::<c_void>(),
                    u64::try_from(bytes_per_row).expect("validated mip row size fits u64"),
                    u64::try_from(upload.len()).expect("validated mip size fits u64"),
                );
            } else {
                texture.replace_region(
                    region,
                    mip,
                    upload.as_ptr().cast::<c_void>(),
                    u64::try_from(bytes_per_row).expect("validated mip row size fits u64"),
                );
            }
            uploaded_bytes = uploaded_bytes.saturating_add(upload.len());
        }
    }
    record_texture_work(
        &MIP_UPLOAD_COUNT,
        &MIP_UPLOAD_BYTES,
        &MIP_UPLOAD_NANOSECONDS,
        uploaded_bytes,
        started.elapsed(),
    );
    Ok(texture)
}

#[allow(clippy::too_many_lines)]
pub(crate) fn create_exact_mode5_sampled_mip_view(
    device: &Device,
    descriptor: DecodedImageDescriptor,
    raw: &[u8],
    comparison: bool,
    writable: bool,
) -> Result<Texture, String> {
    if !matches!(descriptor.texture_type, 9 | 11 | 13)
        || descriptor.layer_count == 0
        || descriptor.guest.tile_mode != 5
        || descriptor.mip_level_count == 0
    {
        return Err(format!(
            "exact mode-5 sampled mip view requires a nonempty 2D or array resource, got type={} layers={} tile={} base={} count={}",
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
    let layout = super::mode5_mip_chain_layout(descriptor.guest, underlying_mip_count)?;
    let layer_count = usize::try_from(descriptor.layer_count)
        .map_err(|_| "sampled layer count does not fit usize")?;
    let expected = layout
        .byte_count
        .checked_mul(layer_count)
        .ok_or("mode-5 layered mip-chain byte count overflows")?;
    if raw.len() != expected {
        return Err(format!(
            "mode-5 layered mip source has {} bytes, expected {expected}",
            raw.len()
        ));
    }
    let linear_layers = raw
        .chunks_exact(layout.byte_count)
        .map(|layer| {
            decode_mode5_mip_chain_from_raw(descriptor.guest, underlying_mip_count, layer)
                .map(|(_, levels)| levels)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let texture_type = if descriptor.texture_type == 9 {
        MTLTextureType::D2
    } else {
        MTLTextureType::D2Array
    };
    let texture = create_exact_sampled_mip_texture(
        device,
        descriptor,
        &layout,
        &linear_layers,
        comparison,
        writable,
    )?;
    let pixel_format = if comparison {
        comparison_pixel_format(descriptor.guest.pixel_format)
    } else {
        descriptor.guest.pixel_format
    };
    if descriptor.base_mip_level == 0 && descriptor.mip_level_count == underlying_mip_count {
        return Ok(texture);
    }
    Ok(texture.new_texture_view_from_slice(
        pixel_format,
        texture_type,
        NSRange::new(
            u64::from(descriptor.base_mip_level),
            u64::from(descriptor.mip_level_count),
        ),
        NSRange::new(0, u64::from(descriptor.layer_count)),
    ))
}

pub(crate) fn create_exact_mode27_sampled_mip_view(
    device: &Device,
    descriptor: DecodedImageDescriptor,
    raw: &[u8],
    comparison: bool,
) -> Result<Texture, String> {
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
    let layout = mode27_mip_chain_layout(descriptor.guest, underlying_mip_count)?;
    let texture = create_exact_sampled_mip_texture(
        device,
        descriptor,
        &layout,
        &[linear_levels],
        comparison,
        false,
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
        if comparison {
            comparison_pixel_format(descriptor.guest.pixel_format)
        } else {
            descriptor.guest.pixel_format
        },
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
