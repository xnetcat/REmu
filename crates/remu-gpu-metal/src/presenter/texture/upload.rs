// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::SharedEventDrawTextureOverride;
use super::super::capture_target_sample_textures;
use super::super::sampled_texture_cache_byte_capacity;
use super::{
    CachedSampledTexture, DepthArrayKey, DepthArrayResidentTexture, GuestTextureDescriptor,
    GuestTextureWriteWatch, Mode27WorkOrigin, MultisampleResidentTexture, MultisampleTextureKey,
    RangeSet, ResidencyState, ResidentGuestTexture, ResidentTextureActivityStamp,
    ResidentTextureMap, SAMPLED_TEXTURE_CACHE_CAPACITY, SAMPLED_TEXTURE_CACHE_CAPACITY_EVICTIONS,
    SAMPLED_TEXTURE_CACHE_DIRTY_RELOADS, SAMPLED_TEXTURE_CACHE_HITS, SAMPLED_TEXTURE_CACHE_MISSES,
    SampledTextureKey, SpanQueryOrder, UploadedImageTexture, WriteWatchRefresh,
    canonical_texture_bindings, create_sampled_guest_texture,
    create_sampled_guest_texture_from_guest_gpu, decode_image_descriptor,
    ensure_mipmapped_resident_texture_in_map, ensure_resident_texture_in_map,
    fallback_image_descriptor, guest_texture_byte_count,
    guest_texture_descriptors_have_same_layout, guest_texture_descriptors_overlap,
    guest_texture_descriptors_share_depth_sample_storage,
    guest_texture_descriptors_share_gpu_sample_storage, guest_texture_descriptors_share_storage,
    image_resource_descriptor_is_null, is_storage_image_opcode, is_writable_image_opcode,
    persistent_sampled_texture_write_watch, refresh_resident_texture_write_watch,
    refresh_sampled_texture_write_watch, render_resolve_candidate_matches,
    resident_color_view_sample_is_eligible, resident_depth_sample_is_eligible,
    resident_same_image_sample_is_eligible, resident_sample_authority_ok,
    resident_texture_for_sample, retire_sampled_texture, sampled_texture_cache_needs_reload,
    sampled_texture_key, sampled_texture_supports_access, sampled_texture_view_for_binding,
    sampler_uses_normalized_coordinates, storage_image_requires_mip_hierarchy,
    texture_element_layout, trace_alias_correlated_image_bindings, trace_glyph_atlas,
    trace_resident_texture_removal, trace_target_sampling, watch_sampled_texture,
};
use metal::{Device, MTLPixelFormat, Texture};
use remu_memory::guest_shared_mapping_identity;
use remu_shader::{ShaderEvaluation, image_is_comparison};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

pub(crate) fn sampled_texture_cache_accepts(
    resident_bytes: u64,
    byte_capacity: Option<u64>,
) -> bool {
    byte_capacity.is_none_or(|capacity| resident_bytes <= capacity)
}

pub(crate) fn sampled_texture_cache_requires_eviction(
    entry_count: usize,
    resident_bytes: u64,
    incoming_bytes: u64,
    byte_capacity: Option<u64>,
) -> bool {
    entry_count >= SAMPLED_TEXTURE_CACHE_CAPACITY
        || byte_capacity
            .is_some_and(|capacity| resident_bytes.saturating_add(incoming_bytes) > capacity)
}

fn diagnostic_r32_float_sample_override() -> Option<f32> {
    static VALUE: OnceLock<Option<f32>> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("REMU_DIAGNOSTIC_R32F_SAMPLE")
            .ok()?
            .parse::<f32>()
            .ok()
            .filter(|value| value.is_finite())
    })
}

fn diagnostic_zero_r16_row_samples() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DIAGNOSTIC_ZERO_R16_ROW").is_some_and(|value| value == "1")
    })
}

fn diagnostic_zero_rgba8_1x1_white() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DIAGNOSTIC_ZERO_RGBA8_1X1_WHITE").is_some_and(|value| value == "1")
    })
}

fn diagnostic_sampled_dcc_first_use_clear() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("REMU_DIAGNOSTIC_SAMPLED_DCC_FIRST_USE_CLEAR")
            .is_some_and(|value| value == "1")
    })
}

fn image_descriptor_dcc_enabled(words: &[u32]) -> bool {
    words.get(6).is_some_and(|word| word & (1 << 21) != 0)
}

#[allow(clippy::too_many_arguments)]
fn synthetic_sampled_texture(
    device: &Device,
    queue: &metal::CommandQueue,
    key: SampledTextureKey,
    descriptor: super::DecodedImageDescriptor,
    linear: &[u8],
    comparison: bool,
    writable: bool,
    label: &str,
    cache: &mut HashMap<SampledTextureKey, Texture>,
) -> Result<Texture, String> {
    if let Some(texture) = cache
        .get(&key)
        .filter(|texture| sampled_texture_supports_access(texture, writable))
    {
        return Ok(texture.clone());
    }
    cache.remove(&key);
    let texture = create_sampled_guest_texture(
        device,
        queue,
        descriptor.guest,
        linear,
        descriptor.base_mip_level,
        descriptor.mip_level_count,
        descriptor.texture_type,
        descriptor.layer_count,
        comparison,
        writable,
    )?;
    texture.set_label(label);
    cache.insert(key, texture.clone());
    Ok(texture)
}

fn rgba16_float_transparent_black_with_opaque_alpha(
    descriptor: super::DecodedImageDescriptor,
) -> Result<Vec<u8>, String> {
    if descriptor.guest.pixel_format != MTLPixelFormat::RGBA16Float {
        return Err(format!(
            "diagnostic sampled DCC clear does not support {:?}",
            descriptor.guest.pixel_format
        ));
    }
    let texel_count = usize::try_from(descriptor.guest.width)
        .ok()
        .and_then(|width| {
            usize::try_from(descriptor.guest.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|texels| {
            usize::try_from(descriptor.layer_count)
                .ok()
                .and_then(|layers| texels.checked_mul(layers))
        })
        .ok_or("diagnostic sampled DCC clear texel count overflows")?;
    let mut linear = vec![
        0;
        texel_count
            .checked_mul(8)
            .ok_or("diagnostic sampled DCC clear byte count overflows",)?
    ];
    // IEEE-754 binary16 1.0 is 0x3c00. The host is little-endian, as is the guest payload.
    for texel in linear.chunks_exact_mut(8) {
        texel[6..8].copy_from_slice(&0x3c00_u16.to_le_bytes());
    }
    Ok(linear)
}

pub(crate) fn comparison_depth32_upload(
    guest: GuestTextureDescriptor,
    linear: &[u8],
) -> Result<Option<Vec<u8>>, String> {
    let format = guest.pixel_format;
    if matches!(
        format,
        MTLPixelFormat::BC1_RGBA | MTLPixelFormat::BC1_RGBA_sRGB
    ) {
        let width =
            usize::try_from(guest.width).map_err(|_| "comparison BC1 width does not fit usize")?;
        let height = usize::try_from(guest.height)
            .map_err(|_| "comparison BC1 height does not fit usize")?;
        let blocks_wide = width.div_ceil(4);
        let blocks_high = height.div_ceil(4);
        let face_bytes = blocks_wide
            .checked_mul(blocks_high)
            .and_then(|blocks| blocks.checked_mul(8))
            .ok_or("comparison BC1 face size overflows")?;
        if face_bytes == 0 || !linear.len().is_multiple_of(face_bytes) {
            return Err(format!(
                "comparison BC1 source has {} bytes, not whole {face_bytes}-byte faces",
                linear.len()
            ));
        }
        let srgb = format == MTLPixelFormat::BC1_RGBA_sRGB;
        let mut depth = Vec::with_capacity(
            width
                .checked_mul(height)
                .and_then(|pixels| pixels.checked_mul(linear.len() / face_bytes))
                .and_then(|pixels| pixels.checked_mul(std::mem::size_of::<f32>()))
                .ok_or("comparison BC1 output size overflows")?,
        );
        for face in linear.chunks_exact(face_bytes) {
            for y in 0..height {
                for x in 0..width {
                    let block = (y / 4) * blocks_wide + x / 4;
                    let encoded = &face[block * 8..block * 8 + 8];
                    let color0 = u16::from_le_bytes([encoded[0], encoded[1]]);
                    let color1 = u16::from_le_bytes([encoded[2], encoded[3]]);
                    let red0 = f32::from((color0 >> 11) & 0x1f) / 31.0;
                    let red1 = f32::from((color1 >> 11) & 0x1f) / 31.0;
                    let palette = if color0 > color1 {
                        [
                            red0,
                            red1,
                            (2.0 * red0 + red1) / 3.0,
                            (red0 + 2.0 * red1) / 3.0,
                        ]
                    } else {
                        [red0, red1, (red0 + red1) / 2.0, 0.0]
                    };
                    let selectors =
                        u32::from_le_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]);
                    let texel = (y % 4) * 4 + x % 4;
                    let selector = usize::try_from((selectors >> (texel * 2)) & 0x3)
                        .expect("BC1 selector fits usize");
                    let encoded_red = palette[selector];
                    let value = if srgb {
                        if encoded_red <= 0.040_45 {
                            encoded_red / 12.92
                        } else {
                            ((encoded_red + 0.055) / 1.055).powf(2.4)
                        }
                    } else {
                        encoded_red
                    };
                    depth.extend_from_slice(&value.to_ne_bytes());
                }
            }
        }
        return Ok(Some(depth));
    }
    let (stride, red_offset, srgb) = match format {
        MTLPixelFormat::R16Unorm | MTLPixelFormat::R32Float => return Ok(None),
        MTLPixelFormat::R8Unorm => (1, 0, false),
        MTLPixelFormat::R8Unorm_sRGB => (1, 0, true),
        MTLPixelFormat::RG8Unorm => (2, 0, false),
        MTLPixelFormat::RG8Unorm_sRGB => (2, 0, true),
        MTLPixelFormat::RGBA8Unorm => (4, 0, false),
        MTLPixelFormat::RGBA8Unorm_sRGB => (4, 0, true),
        MTLPixelFormat::BGRA8Unorm => (4, 2, false),
        MTLPixelFormat::BGRA8Unorm_sRGB => (4, 2, true),
        _ => {
            return Err(format!(
                "comparison sampling cannot convert Metal image format {format:?} to depth"
            ));
        }
    };
    if !linear.len().is_multiple_of(stride) {
        return Err(format!(
            "comparison image byte count {} is not aligned to its {stride}-byte texel",
            linear.len()
        ));
    }
    let mut depth = Vec::with_capacity(linear.len() / stride * std::mem::size_of::<f32>());
    for texel in linear.chunks_exact(stride) {
        let encoded = f32::from(texel[red_offset]) / 255.0;
        let value = if srgb {
            if encoded <= 0.040_45 {
                encoded / 12.92
            } else {
                ((encoded + 0.055) / 1.055).powf(2.4)
            }
        } else {
            encoded
        };
        depth.extend_from_slice(&value.to_ne_bytes());
    }
    Ok(Some(depth))
}

/// Settles a dedicated Phase 1G guest→texture upload on `queue`.
fn settle_gpu_sampled_upload(
    device: &Device,
    queue: &metal::CommandQueue,
    descriptor: super::DecodedImageDescriptor,
    comparison: bool,
    writable: bool,
) -> Result<Texture, String> {
    let upload_buffer = queue.new_command_buffer().to_owned();
    upload_buffer.set_label("REMu Phase 1G sampled GPU upload");
    let texture = create_sampled_guest_texture_from_guest_gpu(
        device,
        &upload_buffer,
        descriptor,
        comparison,
        writable,
    )?;
    upload_buffer.commit();
    upload_buffer.wait_until_completed();
    Ok(texture)
}

pub(crate) fn exact_mode27_array_storage_supported(
    descriptor: super::DecodedImageDescriptor,
) -> bool {
    let underlying_mip_count = descriptor
        .base_mip_level
        .checked_add(descriptor.mip_level_count);
    descriptor.texture_type == 13
        && descriptor.guest.tile_mode == 27
        && descriptor.mip_level_count > 0
        && underlying_mip_count.is_some()
        && descriptor.layer_count > 0
        && (underlying_mip_count == Some(1) || descriptor.guest.pitch == descriptor.guest.width)
        && super::super::metal_uncompressed_bytes_per_pixel(descriptor.guest.pixel_format)
            == Some(descriptor.guest.bytes_per_pixel)
}

fn refuse_gpu_authored_array_aliases(
    descriptor: super::DecodedImageDescriptor,
    residents: &mut ResidentTextureMap,
    allow_exact_same_view: bool,
) -> Result<(), String> {
    let selected_span = descriptor.span.ok_or("array selected span overflows")?;
    let selected = RangeSet::for_span(&selected_span);
    let aliases = residents.overlapping(&selected_span, SpanQueryOrder::ResidentFirst);
    for alias in aliases {
        let exact_same_view = alias == descriptor.guest
            && residents.get(&alias).is_some_and(|resident| {
                let underlying_mip_count = descriptor
                    .base_mip_level
                    .checked_add(descriptor.mip_level_count);
                resident.texture_type == descriptor.texture_type
                    && resident.base_array_slice == descriptor.base_array_slice
                    && resident.layer_count == descriptor.layer_count
                    && underlying_mip_count == Some(resident.mip_level_count)
                    && resident.span == Some(selected_span)
            });
        if allow_exact_same_view && exact_same_view {
            continue;
        }
        let refresh = refresh_resident_texture_write_watch(
            residents
                .get_mut(&alias)
                .expect("indexed array alias exists"),
        )?;
        if refresh == WriteWatchRefresh::Replaced {
            if let Some(retired) = residents.remove(&alias) {
                trace_resident_texture_removal("array-alias-mapping-replaced", &retired);
            }
            continue;
        }
        let alias_has_selected_gpu_debt = residents.get(&alias).is_some_and(|resident| {
            resident.is_live() && resident.gpu_authored.intersects(&selected)
        });
        if alias_has_selected_gpu_debt {
            return Err(format!(
                "refusing array view at {:#x}: overlapping GPU-authored resident at {:#x} owns selected bytes",
                descriptor.guest.address, alias.address,
            ));
        }
    }
    Ok(())
}

fn ensure_exact_mode27_array_storage_resident(
    device: &Device,
    queue: &metal::CommandQueue,
    descriptor: super::DecodedImageDescriptor,
    residents: &mut ResidentTextureMap,
) -> Result<Texture, String> {
    if !exact_mode27_array_storage_supported(descriptor) {
        return Err(format!(
            "writable array residency requires an exact uncompressed mode-27 type-13 view, got type={} tile={} base_mip={} mips={} layers={} format={:?} bpe={}",
            descriptor.texture_type,
            descriptor.guest.tile_mode,
            descriptor.base_mip_level,
            descriptor.mip_level_count,
            descriptor.layer_count,
            descriptor.guest.pixel_format,
            descriptor.guest.bytes_per_pixel,
        ));
    }
    let selected_span = descriptor
        .span
        .ok_or("writable array selected span overflows")?;
    let underlying_mip_count = descriptor
        .base_mip_level
        .checked_add(descriptor.mip_level_count)
        .ok_or("writable array underlying mip count overflows")?;
    refuse_gpu_authored_array_aliases(descriptor, residents, true)?;
    let exact_existing = residents.get(&descriptor.guest).is_some_and(|resident| {
        resident.texture_type == descriptor.texture_type
            && resident.base_array_slice == descriptor.base_array_slice
            && resident.layer_count == descriptor.layer_count
            && resident.mip_level_count == underlying_mip_count
            && resident.span == descriptor.span
            && resident.owned.covers(&selected_span)
    });
    if exact_existing {
        let (refresh, cpu_changed, gpu_written) = {
            let resident = residents
                .get_mut(&descriptor.guest)
                .expect("exact array resident still exists");
            let refresh = refresh_resident_texture_write_watch(resident)?;
            if refresh == WriteWatchRefresh::Reprotecting {
                return Err(format!(
                    "array resident at {:#x} is being reprotected",
                    descriptor.guest.address
                ));
            }
            let cpu_may_have_changed =
                matches!(refresh, WriteWatchRefresh::Renewed { was_dirty: true })
                    || resident.write_watch.content_validation_pending()
                    || resident.write_watch.is_dirty();
            let cpu_changed = cpu_may_have_changed && resident.write_watch.content_has_changed()?;
            (refresh, cpu_changed, resident.gpu_written())
        };
        if refresh != WriteWatchRefresh::Replaced && !cpu_changed {
            return resident_texture_for_sample(
                residents
                    .get_mut(&descriptor.guest)
                    .expect("exact array resident still exists"),
                descriptor,
            )
            .ok_or_else(|| {
                "exact array resident cannot represent the requested mip view".to_owned()
            });
        }
        if cpu_changed && gpu_written {
            return Err(format!(
                "refusing to merge CPU-dirty guest bytes with dirty array resident at {:#x}",
                descriptor.guest.address
            ));
        }
    }
    if let Some(existing) = residents.get(&descriptor.guest)
        && existing.gpu_written()
    {
        return Err(format!(
            "refusing to replace dirty array resident at {:#x}: existing base={} layers={} mips={} requested base={} layers={} mips={}",
            descriptor.guest.address,
            existing.base_array_slice,
            existing.layer_count,
            existing.mip_level_count,
            descriptor.base_array_slice,
            descriptor.layer_count,
            descriptor.mip_level_count,
        ));
    }
    if let Some(retired) = residents.remove(&descriptor.guest) {
        trace_resident_texture_removal("array-storage-view-replaced", &retired);
    }
    let write_watch = watch_sampled_texture(descriptor)?;
    let full_descriptor = super::DecodedImageDescriptor {
        base_mip_level: 0,
        mip_level_count: underlying_mip_count,
        ..descriptor
    };
    let texture = settle_gpu_sampled_upload(device, queue, full_descriptor, false, true)?;
    let (span, owned) = ResidentGuestTexture::declared_authority(&write_watch);
    if span != Some(selected_span) {
        return Err("array storage resident watch does not match decoded selected span".to_owned());
    }
    residents.insert(
        descriptor.guest,
        ResidentGuestTexture {
            texture: texture.clone(),
            component_reorder_view: None,
            descriptor: descriptor.guest,
            texture_type: descriptor.texture_type,
            base_array_slice: descriptor.base_array_slice,
            layer_count: descriptor.layer_count,
            mip_level_count: underlying_mip_count,
            write_watch,
            span,
            owned,
            gpu_authored: RangeSet::default(),
            state: ResidencyState::Live,
            acquire_invalidated: false,
            content_version: 0,
            gpu_write_watch_generations: Vec::new(),
            last_use: ResidentTextureActivityStamp::default(),
            activity: super::super::ResidentTextureActivity::enabled(),
        },
    );
    resident_texture_for_sample(
        residents
            .get_mut(&descriptor.guest)
            .expect("new exact array resident exists"),
        descriptor,
    )
    .ok_or_else(|| "new exact array resident cannot represent the requested mip view".to_owned())
}

pub(crate) fn upload_image_textures(
    trace_stage: &'static str,
    device: &Device,
    queue: &metal::CommandQueue,
    evaluation: &ShaderEvaluation,
    resident_guest_textures: &mut ResidentTextureMap,
    depth_array_textures: &HashMap<DepthArrayKey, DepthArrayResidentTexture>,
    multisample_textures: &HashMap<MultisampleTextureKey, MultisampleResidentTexture>,
    sampled_texture_cache: &mut HashMap<SampledTextureKey, CachedSampledTexture>,
    sampled_texture_write_watches: &mut HashMap<SampledTextureKey, Arc<GuestTextureWriteWatch>>,
    fallback_texture_cache: &mut HashMap<SampledTextureKey, Texture>,
    next_sampled_texture_use: &mut u64,
    activity_stamp: ResidentTextureActivityStamp,
    shared_event_texture_overrides: &[SharedEventDrawTextureOverride],
) -> Result<Vec<UploadedImageTexture>, String> {
    trace_alias_correlated_image_bindings(trace_stage, evaluation, activity_stamp);
    let canonical_bindings = canonical_texture_bindings(evaluation);
    // Access mode is part of the MSL binding identity, so a shader can expose one guest image as
    // separate read-only and read-write texture arguments. Metal must still receive the same
    // native allocation for both arguments. Determine the capability superset before creating the
    // first cache entry; otherwise a load preceding a store would retain a read-only allocation
    // while the store silently promoted only its later binding.
    let shader_write_keys = canonical_bindings
        .iter()
        .copied()
        .filter(|binding| is_writable_image_opcode(binding.opcode))
        .filter_map(|binding| {
            decode_image_descriptor(&binding.resource_descriptor)
                .or_else(|_| fallback_image_descriptor(binding))
                .ok()
                .map(|descriptor| {
                    sampled_texture_key(descriptor, image_is_comparison(binding.opcode))
                })
        })
        .collect::<HashSet<_>>();
    canonical_bindings
        .into_iter()
        .map(|binding| {
            let writable = is_writable_image_opcode(binding.opcode);
            let storage = is_storage_image_opcode(binding.opcode);
            let comparison = image_is_comparison(binding.opcode);
            let decoded = decode_image_descriptor(&binding.resource_descriptor);
            let fallback = decoded.is_err();
            if let Err(error) = &decoded {
                let is_null =
                    image_resource_descriptor_is_null(&binding.resource_descriptor);
                let texture_type = binding
                    .resource_descriptor
                    .get(3)
                    .map_or(0, |word| (word >> 28) & 0xf);
                // GFX10 buffer V# / TYPE nibble 0 is not an image T#. Falling back to opaque
                // black 1x1 hides the bad bind (title compositor BG samples as solid black).
                // Null SRDs still take the optional-bind fallback below; non-null type-0 must
                // loud-refuse so the draw is attributable instead of silently modulated.
                if !is_null && texture_type == 0 {
                    static LOGGED: AtomicUsize = AtomicUsize::new(0);
                    if LOGGED.fetch_add(1, Ordering::Relaxed) < 64 {
                        eprintln!(
                            "[REMu][METAL][IMAGE-REFUSE] pc={:#x} opcode={} words={:08x?} error={error} reason=non-null-type-0-buffer-srd",
                            binding.pc, binding.opcode, binding.resource_descriptor,
                        );
                    }
                    return Err(format!(
                        "image at PC {:#x}: refusing opaque-black fallback for non-null type-0 SRD ({error})",
                        binding.pc
                    ));
                }
                // A fallback image samples as opaque black, so any draw that modulates by its
                // alpha renders solid covers (Poppy menu glyphs). Name the failing descriptor
                // once so the responsible guest SRD is attributable from the run log.
                // K1: lock-free bounded sampler (was a mutex-guarded HashSet dedup).
                static LOGGED: AtomicUsize = AtomicUsize::new(0);
                if LOGGED.fetch_add(1, Ordering::Relaxed) < 64 {
                    eprintln!(
                        "[REMu][METAL][IMAGE-FALLBACK] pc={:#x} opcode={} words={:08x?} error={error}",
                        binding.pc, binding.opcode, binding.resource_descriptor,
                    );
                }
            }
            let descriptor = decoded
                .or_else(|_| fallback_image_descriptor(binding))
                .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
            let key = sampled_texture_key(descriptor, comparison);
            let requires_shader_write = shader_write_keys.contains(&key);
            if !writable
                && !requires_shader_write
                && let Some(retained) = shared_event_texture_overrides
                .iter()
                .find(|retained| retained.matches(descriptor))
            {
                let texture = sampled_texture_view_for_binding(&retained.texture, binding)?;
                if comparison
                    && !matches!(
                        texture.pixel_format(),
                        MTLPixelFormat::Depth16Unorm | MTLPixelFormat::Depth32Float
                    )
                {
                    return Err(format!(
                        "image at PC {:#x}: comparison retained array override requires native Metal depth, got {:?}",
                        binding.pc,
                        texture.pixel_format(),
                    ));
                }
                return Ok(UploadedImageTexture {
                    texture,
                    descriptor,
                    writable: false,
                    selected_resident: None,
                });
            }
            if !storage && !comparison && descriptor.texture_type == 13 {
                // A per-face render can own bytes inside this array while guest RAM still holds
                // the pre-render contents. Until a GPU-to-array reconstruction path exists, a
                // read-only array upload must fail rather than silently sample that stale RAM.
                refuse_gpu_authored_array_aliases(descriptor, resident_guest_textures, true)?;
            }
            // Slate/UMG font atlases are single-channel sampled images whose whole glyph
            // rendering depends on the T# destination select reaching the shader. Log the raw
            // descriptor words and the decoded swizzle once per atlas so a tofu report can be
            // split into "wrong atlas bytes" versus "lost channel routing".
            if trace_glyph_atlas()
                && !writable
                && matches!(
                    descriptor.guest.pixel_format,
                    MTLPixelFormat::R8Unorm | MTLPixelFormat::A8Unorm
                )
            {
                // K1: lock-free bounded sampler (was a mutex-guarded HashSet dedup).
                static LOGGED: AtomicUsize = AtomicUsize::new(0);
                if LOGGED.fetch_add(1, Ordering::Relaxed) < 64 {
                    eprintln!(
                        "[REMu][METAL][GLYPH-ATLAS] stage={trace_stage} pc={:#x} fallback={fallback} words={:08x?} address={:#x} extent={}x{} pitch={} format={:?} tile={} dst_sel={:#x} mips={}..={} type={}",
                        binding.pc,
                        binding.resource_descriptor,
                        descriptor.guest.address,
                        descriptor.guest.width,
                        descriptor.guest.height,
                        descriptor.guest.pitch,
                        descriptor.guest.pixel_format,
                        descriptor.guest.tile_mode,
                        descriptor.destination_select,
                        descriptor.base_mip_level,
                        descriptor
                            .base_mip_level
                            .saturating_add(descriptor.mip_level_count.saturating_sub(1)),
                        descriptor.texture_type,
                    );
                }
            }
            if trace_target_sampling()
                && descriptor.guest.width == 1_920
                && descriptor.guest.height == 1_080
            {
                // K1: lock-free bounded sampler (was a mutex-guarded HashSet dedup).
                // Attribution runs that also set REMU_CAPTURE_TARGET_SAMPLES need late title
                // intermediates (e.g. 0x8044590000); the first-128 cap exhausted during splash.
                static LOGGED: AtomicUsize = AtomicUsize::new(0);
                let sample_cap = if capture_target_sample_textures() {
                    512
                } else {
                    128
                };
                let normalized_coordinates =
                    sampler_uses_normalized_coordinates(&binding.sampler_descriptor);
                if LOGGED.fetch_add(1, Ordering::Relaxed) < sample_cap {
                    let byte_count = guest_texture_byte_count(descriptor.guest);
                    let mapping = byte_count.and_then(|byte_count| {
                        guest_shared_mapping_identity(descriptor.guest.address, byte_count)
                    });
                    let mapping_start =
                        guest_shared_mapping_identity(descriptor.guest.address, 1);
                    let mapping_end = byte_count
                        .and_then(|byte_count| descriptor.guest.address.checked_add(byte_count - 1))
                        .and_then(|address| guest_shared_mapping_identity(address, 1));
                    let exact = resident_guest_textures
                        .get(&descriptor.guest)
                        .map(|resident| (resident.gpu_written(), resident.content_version));
                    let aliases = descriptor
                        .span
                        .iter()
                        .flat_map(|span| {
                            resident_guest_textures
                                .overlapping(span, SpanQueryOrder::ResidentFirst)
                        })
                        .filter_map(|alias| {
                            let resident = resident_guest_textures.get(&alias)?;
                            Some((
                                resident.descriptor.address,
                                resident.descriptor.width,
                                resident.descriptor.height,
                                resident.descriptor.pixel_format,
                                resident.descriptor.tile_mode,
                                resident.gpu_written(),
                                resident.content_version,
                            ))
                        })
                        .collect::<Vec<_>>();
                    let multisample_candidates = multisample_textures
                        .iter()
                        .filter(|(key, _)| {
                            guest_texture_descriptors_have_same_layout(
                                descriptor.guest,
                                key.descriptor,
                            )
                        })
                        .map(|(key, multisample)| {
                            (
                                key.descriptor.address,
                                key.sample_count,
                                multisample.initialized,
                                guest_texture_descriptors_overlap(
                                    descriptor.guest,
                                    key.descriptor,
                                ),
                                resident_guest_textures
                                    .get(&key.descriptor)
                                    .map(|resident| {
                                        (resident.gpu_written(), resident.content_version)
                                    }),
                            )
                        })
                        .collect::<Vec<_>>();
                    eprintln!(
                        "[REMu][METAL][SAMPLE] pc={:#x} op={} guest={:#x} 1920x1080 format={:?} tile={} select={:#x} mips={} type={} comparison={} normalized={} sampler={:08x?} bytes={byte_count:x?} mapping={mapping:?} endpoints={mapping_start:?}/{mapping_end:?} exact={exact:?} aliases={aliases:?} msaa={multisample_candidates:?}",
                        binding.pc,
                        binding.opcode,
                        descriptor.guest.address,
                        descriptor.guest.pixel_format,
                        descriptor.guest.tile_mode,
                        descriptor.destination_select,
                        descriptor.mip_level_count,
                        descriptor.texture_type,
                        comparison,
                        normalized_coordinates,
                        binding.sampler_descriptor,
                    );
                }
            }
            let underlying_mip_count = descriptor
                .base_mip_level
                .checked_add(descriptor.mip_level_count.max(1))
                .ok_or("sampled resident mip range overflows")?;
            if matches!(descriptor.texture_type, 9 | 10 | 13) {
                let matching_resident = resident_guest_textures
                    .get(&descriptor.guest)
                    .is_some_and(|resident| {
                        resident.texture_type == descriptor.texture_type
                            && resident.base_array_slice == descriptor.base_array_slice
                            && resident.layer_count == descriptor.layer_count
                            && if descriptor.texture_type == 13 {
                                resident.mip_level_count == underlying_mip_count
                                    && resident.span == descriptor.span
                            } else {
                                resident.mip_level_count >= underlying_mip_count
                            }
                    });
                let mut remove_resident = false;
                if matching_resident {
                    let (refresh, content_changed, gpu_written, complete_authority) = {
                        let resident = resident_guest_textures
                            .get_mut(&descriptor.guest)
                            .expect("matching image resident exists");
                        let refresh = refresh_resident_texture_write_watch(resident)?;
                        let validate_array_content = !comparison
                            && descriptor.texture_type == 13
                            && !matches!(
                                refresh,
                                WriteWatchRefresh::Replaced
                                    | WriteWatchRefresh::Reprotecting
                            );
                        let content_changed = validate_array_content
                            && (matches!(
                                refresh,
                                WriteWatchRefresh::Renewed { was_dirty: true }
                            ) || resident.write_watch.content_validation_pending()
                                || resident.write_watch.is_dirty())
                            && resident.write_watch.content_has_changed()?;
                        (
                            refresh,
                            content_changed,
                            resident.gpu_written(),
                            resident.serves_reads(),
                        )
                    };
                    if !comparison && descriptor.texture_type == 13 {
                        match refresh {
                            WriteWatchRefresh::Reprotecting => {
                                return Err(format!(
                                    "array resident at {:#x} is being reprotected",
                                    descriptor.guest.address
                                ));
                            }
                            WriteWatchRefresh::Replaced => remove_resident = true,
                            WriteWatchRefresh::Active | WriteWatchRefresh::Renewed { .. } => {
                                if (content_changed || !complete_authority) && gpu_written {
                                    return Err(format!(
                                        "refusing array bind at {:#x}: CPU-dirty or partially superseded GPU resident cannot serve the complete selected span",
                                        descriptor.guest.address
                                    ));
                                }
                                remove_resident = content_changed || !complete_authority;
                            }
                        }
                    } else {
                        remove_resident = refresh == WriteWatchRefresh::Replaced;
                    }
                }
                if remove_resident
                    && let Some(resident) = resident_guest_textures.remove(&descriptor.guest) {
                        trace_resident_texture_removal(
                            "sample-upload-array-stale-or-replaced",
                            &resident,
                        );
                    }
            }
            let multisample_resolve = (!comparison && descriptor.texture_type == 9)
                .then(|| {
                    multisample_textures
                        .iter()
                        .filter(|(_, multisample)| multisample.initialized)
                        .filter(|(key, _)| {
                            render_resolve_candidate_matches(
                                descriptor.guest,
                                key.descriptor,
                                key.sample_count,
                            )
                        })
                        .filter_map(|(key, _)| {
                            let resident = resident_guest_textures.get(&key.descriptor)?;
                            (resident.gpu_written() && resident.serves_reads()).then_some((
                                resident.content_version,
                                key.descriptor,
                                resident.texture.clone(),
                            ))
                        })
                        .max_by_key(|(version, _, _)| *version)
                })
                .flatten();
            let resident_descriptor = matches!(descriptor.texture_type, 9 | 10 | 13)
                .then(|| {
                    let exact = (!comparison)
                        .then(|| {
                            resident_guest_textures
                                .get(&descriptor.guest)
                                .filter(|resident| {
                                    // Same-extent GPU images keep answering after neighbour
                                    // ResourceSpan over-report clears serves_reads() (SH EyeAdapt).
                                    (resident.serves_reads()
                                        || (descriptor.texture_type == 9
                                            && resident.gpu_written()
                                            && resident_sample_authority_ok(
                                                resident,
                                                descriptor.guest,
                                            )))
                                        && resident.texture_type == descriptor.texture_type
                                        && resident.base_array_slice
                                            == descriptor.base_array_slice
                                        && resident.layer_count == descriptor.layer_count
                                        && if descriptor.texture_type == 13 {
                                            resident.mip_level_count == underlying_mip_count
                                                && resident.span == descriptor.span
                                        } else {
                                            resident.mip_level_count >= underlying_mip_count
                                        }
                                })
                                .map(|resident| {
                                    (resident.descriptor, resident.gpu_written(), resident.content_version)
                                })
                        })
                        .flatten();
                    let compatible = (descriptor.texture_type == 9).then(|| {
                        resident_guest_textures
                            .values()
                            .filter(|resident| {
                                resident.is_live()
                                    && resident.texture_type == 9
                                    && resident.layer_count == 1
                                    && resident.mip_level_count >= underlying_mip_count
                                    && ((!comparison
                                        && (guest_texture_descriptors_share_storage(
                                            resident.descriptor,
                                            descriptor.guest,
                                        ) || guest_texture_descriptors_share_gpu_sample_storage(
                                            resident.descriptor,
                                            descriptor.guest,
                                        ) || resident_color_view_sample_is_eligible(
                                            resident,
                                            descriptor,
                                            comparison,
                                        ) || resident_same_image_sample_is_eligible(
                                            resident,
                                            descriptor,
                                            comparison,
                                        )))
                                        || resident_depth_sample_is_eligible(resident, descriptor))
                                    // Exact same-address images may remain gpu-authored after a
                                    // neighbour partially supersedes owned-byte accounting; bind
                                    // through same-image / colour-view eligibility instead of
                                    // uploading unpublished guest RAM.
                                    && (resident.serves_reads()
                                        || resident_color_view_sample_is_eligible(
                                            resident,
                                            descriptor,
                                            comparison,
                                        )
                                        || resident_same_image_sample_is_eligible(
                                            resident,
                                            descriptor,
                                            comparison,
                                        ))
                            })
                            .max_by_key(|resident| {
                                // Prefer a Depth* DB surface over a same-address R32/R16 sample
                                // shadow even when the shadow has a newer content_version.
                                (
                                    resident_depth_sample_is_eligible(resident, descriptor),
                                    resident.gpu_written(),
                                    resident.content_version,
                                )
                            })
                            .map(|resident| {
                                (resident.descriptor, resident.gpu_written(), resident.content_version)
                            })
                    })
                    .flatten();
                    // An earlier R32Float ImageLoad/sample can leave a clean exact resident on the
                    // same address before the DB surface is created as Depth32Float. Prefer the
                    // GPU-authored native depth view over that empty shadow so depth_native hits.
                    // Also prefer Depth* when a later empty R32 "writer" bumped content_version
                    // without being allowed to demote the DB (Silent Hill title 801ac2).
                    match (exact, compatible) {
                        (
                            Some((exact_descriptor, _, _)),
                            Some((compatible_descriptor, true, _)),
                        ) if exact_descriptor != compatible_descriptor
                            && guest_texture_descriptors_share_depth_sample_storage(
                                compatible_descriptor,
                                exact_descriptor,
                                descriptor.destination_select,
                            ) =>
                        {
                            Some(compatible_descriptor)
                        }
                        (
                            Some((exact_descriptor, exact_gpu_written, exact_version)),
                            Some((compatible_descriptor, true, compatible_version)),
                        ) if exact_descriptor != compatible_descriptor
                            && (!exact_gpu_written || compatible_version >= exact_version) =>
                        {
                            Some(compatible_descriptor)
                        }
                        (Some((exact_descriptor, _, _)), _) => Some(exact_descriptor),
                        (None, Some((compatible_descriptor, _, _))) => Some(compatible_descriptor),
                        (None, None) => None,
                    }
                })
                .flatten();
            let resident_resolve = resident_descriptor.and_then(|resident_descriptor| {
                let resident = resident_guest_textures.get_mut(&resident_descriptor)?;
                let version = resident.content_version;
                resident_texture_for_sample(resident, descriptor)
                    .map(|texture| (resident_descriptor, version, texture))
            });
            // A guest descriptor can retain an exact single-sample resident from an earlier
            // render phase while the current frame is produced through a distinct MSAA target
            // with the same layout. Address equality alone is therefore not sufficient: choose
            // whichever compatible GPU image was written most recently. This also naturally
            // retains an exact post-process result when it is newer than the MSAA resolve.
            let select_multisample = match (&resident_resolve, &multisample_resolve) {
                (Some((_, resident_version, _)), Some((multisample_version, _, _))) => {
                    multisample_version > resident_version
                }
                (None, Some(_)) => true,
                _ => false,
            };
            if !select_multisample
                && let Some((resident_descriptor, ..)) = resident_resolve.as_ref()
                && let Some(resident) = resident_guest_textures.get_mut(resident_descriptor)
            {
                resident.note_read(activity_stamp);
            }
            if trace_target_sampling()
                && descriptor.guest.width == 1_920
                && descriptor.guest.height == 1_080
                && select_multisample
                && let Some((version, source, _)) = multisample_resolve.as_ref()
            {
                // K1: lock-free bounded sampler (was a mutex-guarded HashSet dedup).
                static LOGGED_RESOLVES: AtomicUsize = AtomicUsize::new(0);
                if LOGGED_RESOLVES.fetch_add(1, Ordering::Relaxed) < 256 {
                    eprintln!(
                        "[REMu][METAL][MSAA-SAMPLE] guest={:#x} source={:#x} version={} format={:?} tile={}",
                        descriptor.guest.address,
                        source.address,
                        version,
                        source.pixel_format,
                        source.tile_mode,
                    );
                }
            }
            let render_resolve = if select_multisample {
                multisample_resolve
                    .as_ref()
                    .map(|(_, _, texture)| texture.clone())
            } else {
                resident_resolve
                    .as_ref()
                    .map(|(_, _, texture)| texture.clone())
            };
            let render_resolve_descriptor = if select_multisample {
                multisample_resolve
                    .as_ref()
                    .map(|(_, descriptor, _)| *descriptor)
            } else {
                resident_resolve
                    .as_ref()
                    .map(|(descriptor, _, _)| *descriptor)
            };
            let mut selected_resident = None;
            let descriptor_dcc_enabled =
                image_descriptor_dcc_enabled(&binding.resource_descriptor);
            // A layered depth-sample SRD assembled from resident depth faces is served directly:
            // the assembly ran before this upload pass, inside the guest command batch, so the
            // array already orders after the face render passes.
            let depth_array_resolve = (!fallback && !storage)
                .then(|| {
                    depth_array_textures
                        .get(&DepthArrayKey {
                            descriptor: descriptor.guest,
                            base_array_slice: descriptor.base_array_slice,
                            layer_count: descriptor.layer_count,
                        })
                        .map(|entry| entry.texture.clone())
                })
                .flatten();
            if !storage
                && comparison
                && descriptor.texture_type == 13
                && depth_array_resolve.is_none()
                && render_resolve.is_none()
            {
                // Comparison sampling cannot bind a color-array resident directly. If no
                // complete native depth array or retained render resolve exists, even an exact
                // same-key color resident is an alias whose unpublished Metal bytes make a
                // guest-memory depth upload stale.
                refuse_gpu_authored_array_aliases(
                    descriptor,
                    resident_guest_textures,
                    false,
                )?;
            }
            let texture = if !storage
                && descriptor.guest.height == 1
                && descriptor.guest.pixel_format == MTLPixelFormat::R16Unorm
                && diagnostic_zero_r16_row_samples()
            {
                static LOGGED: OnceLock<()> = OnceLock::new();
                LOGGED.get_or_init(|| {
                    eprintln!(
                        "[REMu][METAL][DIAGNOSTIC] replacing sampled R16Unorm rows with zero"
                    );
                });
                let byte_count = usize::try_from(descriptor.guest.width)
                    .ok()
                    .and_then(|width| width.checked_mul(2))
                    .ok_or("diagnostic R16Unorm row byte count overflows")?;
                let texture = create_sampled_guest_texture(
                    device,
                    queue,
                    descriptor.guest,
                    &vec![0; byte_count],
                    0,
                    1,
                    descriptor.texture_type,
                    descriptor.layer_count,
                    false,
                    requires_shader_write,
                )?;
                texture.set_label("REMu diagnostic zero R16Unorm row");
                texture
            } else if !storage
                && descriptor.guest.width == 1
                && descriptor.guest.height == 1
                && descriptor.guest.pixel_format == MTLPixelFormat::R32Float
                && let Some(value) = diagnostic_r32_float_sample_override()
            {
                static LOGGED: OnceLock<()> = OnceLock::new();
                LOGGED.get_or_init(|| {
                    eprintln!(
                        "[REMu][METAL][DIAGNOSTIC] overriding sampled 1x1 R32Float images with {value}"
                    );
                });
                let texture = create_sampled_guest_texture(
                    device,
                    queue,
                    descriptor.guest,
                    &value.to_le_bytes(),
                    0,
                    1,
                    descriptor.texture_type,
                    descriptor.layer_count,
                    false,
                    requires_shader_write,
                )?;
                texture.set_label("REMu diagnostic R32Float sample");
                texture
            } else if !storage
                && descriptor.guest.width == 1
                && descriptor.guest.height == 1
                && descriptor.guest.pixel_format == MTLPixelFormat::RGBA8Unorm
                && diagnostic_zero_rgba8_1x1_white()
            {
                static LOGGED: OnceLock<()> = OnceLock::new();
                LOGGED.get_or_init(|| {
                    eprintln!(
                        "[REMu][METAL][DIAGNOSTIC] replacing sampled 1x1 RGBA8Unorm images with opaque white"
                    );
                });
                synthetic_sampled_texture(
                    device,
                    queue,
                    key,
                    descriptor,
                    &[u8::MAX; 4],
                    comparison,
                    requires_shader_write,
                    "REMu diagnostic white 1x1 RGBA8 sample",
                    fallback_texture_cache,
                )?
            } else if fallback {
                if let Some(texture) = fallback_texture_cache
                    .get(&key)
                    .filter(|texture| {
                        sampled_texture_supports_access(texture, requires_shader_write)
                    })
                {
                    texture.clone()
                } else {
                    fallback_texture_cache.remove(&key);
                    let (_, _, _, bytes_per_element) =
                        texture_element_layout(descriptor.guest)?;
                    let byte_count = bytes_per_element
                        .checked_mul(
                            usize::try_from(descriptor.layer_count)
                                .map_err(|_| "fallback image layer count does not fit usize")?,
                        )
                        .ok_or("fallback image byte count overflows")?;
                    let mut linear = vec![0; byte_count];
                    // Sampled unbound images use opaque black because guest shaders commonly
                    // modulate UI color by their alpha. Storage loads follow image-load semantics
                    // and return zero in every component.
                    if !storage && matches!(
                        descriptor.guest.pixel_format,
                        MTLPixelFormat::RGBA8Unorm
                            | MTLPixelFormat::RGBA8Unorm_sRGB
                            | MTLPixelFormat::RGBA8Uint
                            | MTLPixelFormat::RGBA8Sint
                    ) {
                        for texel in linear.chunks_exact_mut(4) {
                            texel[3] = u8::MAX;
                        }
                    }
                    let texture = create_sampled_guest_texture(
                        device,
                        queue,
                        descriptor.guest,
                        &linear,
                        0,
                        1,
                        descriptor.texture_type,
                        descriptor.layer_count,
                        comparison,
                        requires_shader_write,
                    )?;
                    texture.set_label("REMu fallback guest image");
                    fallback_texture_cache.insert(key, texture.clone());
                    texture
                }
            } else if storage && descriptor.texture_type == 9 {
                // `ensure_resident_texture` may have selected a GPU-current resident through a
                // coherent guest alias rather than the descriptor's exact virtual address. Do
                // not discard that selection and reconstruct an exact-key texture from guest
                // RAM: render-target and earlier storage writes intentionally remain resident
                // until an alias really requires a CPU-visible writeback. Metal encoder order
                // supplies the attachment/storage dependency for this same texture.
                if let Some(texture) = render_resolve {
                    selected_resident = render_resolve_descriptor;
                    texture
                } else if storage_image_requires_mip_hierarchy(binding, descriptor) {
                    selected_resident = Some(descriptor.guest);
                    ensure_mipmapped_resident_texture_in_map(
                        device,
                        Some(queue),
                        None,
                        descriptor,
                        resident_guest_textures,
                        Mode27WorkOrigin::StorageCreate,
                    )?
                } else {
                    selected_resident = Some(descriptor.guest);
                    ensure_resident_texture_in_map(
                        device,
                        Some(queue),
                        descriptor.guest,
                        resident_guest_textures,
                        true,
                    )?
                }
            } else if storage && descriptor.texture_type == 13 {
                if !exact_mode27_array_storage_supported(descriptor) {
                    return Err(format!(
                        "image at PC {:#x}: writable array view is outside exact mode-27 residency support",
                        binding.pc
                    ));
                }
                // The generic render-resolve probe only refreshes mapping replacement. Exact
                // writable arrays additionally need to reject a CPU/GPU dirty merge and rebuild
                // clean Metal content after a watched guest write, so always pass through the
                // array residency state machine.
                selected_resident = Some(descriptor.guest);
                ensure_exact_mode27_array_storage_resident(
                    device,
                    queue,
                    descriptor,
                    resident_guest_textures,
                )?
            } else if let Some(texture) = depth_array_resolve {
                texture
            } else if let Some(texture) = render_resolve {
                // Preserve the actual GPU-side source for opt-in scanout provenance. Read-only
                // bindings do not use this field for write attribution, so recording it cannot
                // change residency semantics.
                selected_resident = render_resolve_descriptor;
                texture
            } else if !storage
                && descriptor_dcc_enabled
                && descriptor.guest.pixel_format == MTLPixelFormat::RGBA16Float
                && diagnostic_sampled_dcc_first_use_clear()
            {
                static LOGGED: AtomicUsize = AtomicUsize::new(0);
                let sequence = LOGGED.fetch_add(1, Ordering::Relaxed) + 1;
                if sequence <= 16 || sequence.is_power_of_two() {
                    eprintln!(
                        "[REMu][METAL][DIAGNOSTIC] materializing sampled DCC first use at {:#x} as transparent black with alpha one sequence={sequence}",
                        descriptor.guest.address,
                    );
                }
                let linear = rgba16_float_transparent_black_with_opaque_alpha(descriptor)?;
                synthetic_sampled_texture(
                    device,
                    queue,
                    key,
                    descriptor,
                    &linear,
                    comparison,
                    requires_shader_write,
                    "REMu diagnostic sampled DCC first-use clear",
                    fallback_texture_cache,
                )?
            } else {
                if sampled_texture_cache.get(&key).is_some_and(|cached| {
                    !sampled_texture_supports_access(
                        &cached.texture,
                        requires_shader_write,
                    )
                }) && let Some(retired) = sampled_texture_cache.remove(&key)
                {
                    // Capabilities cannot be added to an existing Metal allocation. The key
                    // intentionally remains access-agnostic so the replacement becomes the one
                    // coherent identity reused by later samples, loads, and stores.
                    retire_sampled_texture(retired);
                }
                let reload = if let Some(cached) = sampled_texture_cache.get_mut(&key) {
                    let refresh = refresh_sampled_texture_write_watch(
                        key,
                        cached,
                        sampled_texture_write_watches,
                    )?;
                    let content_changed = match refresh {
                        WriteWatchRefresh::Active | WriteWatchRefresh::Renewed { .. } => {
                            cached.write_watch.content_has_changed()?
                        }
                        WriteWatchRefresh::Reprotecting | WriteWatchRefresh::Replaced => false,
                    };
                    sampled_texture_cache_needs_reload(refresh, content_changed)
                } else {
                    false
                };
                if reload {
                    // Coarse write-watch dirty means guest RAM changed; rebuild from guest via
                    // the GPU tiler. No fingerprint compare and no CPU snapshot/detile.
                    if let Some(retired) = sampled_texture_cache.remove(&key) {
                        SAMPLED_TEXTURE_CACHE_DIRTY_RELOADS.fetch_add(1, Ordering::Relaxed);
                        retire_sampled_texture(retired);
                    }
                }
                if let Some(cached) = sampled_texture_cache.get_mut(&key) {
                    SAMPLED_TEXTURE_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                    *next_sampled_texture_use = next_sampled_texture_use.saturating_add(1);
                    cached.last_use = *next_sampled_texture_use;
                    cached.texture.clone()
                } else {
                    SAMPLED_TEXTURE_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
                    let write_watch = persistent_sampled_texture_write_watch(
                        key,
                        descriptor,
                        sampled_texture_write_watches,
                    )
                        .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
                    // Rearm after observing dirty so the next guest store faults again.
                    let _ = write_watch.content_has_changed();
                    let texture = settle_gpu_sampled_upload(
                        device,
                        queue,
                        descriptor,
                        comparison,
                        requires_shader_write,
                    )
                    .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
                    texture.set_label(&format!(
                        "REMu sampled guest image {:#x}",
                        descriptor.guest.address
                    ));
                    // `allocatedSize` reflects the actual host texture retained by this cache
                    // entry. The guest range can be block-compressed and excludes synthesized mip
                    // levels, so charging it allowed a nominal 1.5 GiB cache to consume several
                    // GiB of unified memory. Keep the guest span as a conservative floor for
                    // drivers that report zero for texture views.
                    let incoming_bytes = texture.allocated_size()
                        .max(write_watch.range().1);
                    let byte_capacity = sampled_texture_cache_byte_capacity();
                    let mut cached_bytes = sampled_texture_cache.values().fold(
                        0_u64,
                        |total, cached| total.saturating_add(cached.resident_bytes),
                    );
                    let cacheable = sampled_texture_cache_accepts(incoming_bytes, byte_capacity);
                    while cacheable
                        && !sampled_texture_cache.is_empty()
                        && sampled_texture_cache_requires_eviction(
                            sampled_texture_cache.len(),
                            cached_bytes,
                            incoming_bytes,
                            byte_capacity,
                        )
                    {
                        let old_key = sampled_texture_cache
                            .iter()
                            .min_by_key(|(_, cached)| cached.last_use)
                            .map(|(key, _)| *key)
                            .expect("non-empty sampled texture cache has a least-recently-used key");
                        if let Some(retired) = sampled_texture_cache.remove(&old_key) {
                            cached_bytes = cached_bytes.saturating_sub(retired.resident_bytes);
                            SAMPLED_TEXTURE_CACHE_CAPACITY_EVICTIONS
                                .fetch_add(1, Ordering::Relaxed);
                            retire_sampled_texture(retired);
                        }
                    }
                    if cacheable {
                        *next_sampled_texture_use = next_sampled_texture_use.saturating_add(1);
                        sampled_texture_cache.insert(
                            key,
                            CachedSampledTexture {
                                texture: texture.clone(),
                                write_watch,
                                overlap: descriptor.span,
                                resident_bytes: incoming_bytes,
                                last_use: *next_sampled_texture_use,
                            },
                        );
                    } else {
                        // An allocation larger than the complete cache budget remains alive through
                        // the returned draw binding and its command batch, but must not make the
                        // persistent cache unbounded. Its watch is no longer useful after this
                        // immutable upload either.
                        sampled_texture_write_watches.remove(&key);
                    }
                    texture
                }
            };
            let texture = sampled_texture_view_for_binding(&texture, binding)?;
            Ok(UploadedImageTexture {
                texture,
                descriptor,
                writable: !fallback
                    && writable
                    && (descriptor.texture_type == 9
                        || exact_mode27_array_storage_supported(descriptor)),
                selected_resident,
            })
        })
        .collect()
}

#[cfg(test)]
mod exact_mode27_array_mip_tests {
    use super::{
        ensure_exact_mode27_array_storage_resident, exact_mode27_array_storage_supported,
        resident_texture_for_sample, settle_gpu_sampled_upload,
    };
    use crate::presenter::texture::rematerialisation_subresource_summary;
    use crate::presenter::{
        DecodedImageDescriptor, GuestTextureDescriptor, MetalPresenter,
        ResidentTextureActivityStamp, ResourceSpan, WritebackTrigger, WrittenRegion,
        copy_linear_to_mode64k_2d_mip_level, copy_linear_to_tiled_at_z,
        copy_mode64k_2d_mip_level_to_linear, decode_image_descriptor, mode27_surface_z_xor,
        mode64k_2d_mip_chain_layout,
    };
    use metal::foreign_types::ForeignType;
    use metal::{MTLPixelFormat, MTLRegion, MTLTextureType};
    use remu_shader::{ImageBinding, ImageControl, Opcode, ShaderEvaluation};

    #[test]
    fn writable_mode27_arrays_accept_full_and_nonzero_mip_views() {
        let guest = GuestTextureDescriptor {
            address: 0x70_2713_0000,
            width: 512,
            height: 256,
            pitch: 512,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        for (base_mip_level, mip_level_count) in [(0, 5), (4, 1)] {
            let descriptor = DecodedImageDescriptor {
                guest,
                span: ResourceSpan::for_image_view(
                    guest,
                    base_mip_level,
                    mip_level_count,
                    13,
                    3,
                    2,
                )
                .expect("exact array mip span"),
                destination_select: 0xfac,
                base_mip_level,
                mip_level_count,
                texture_type: 13,
                base_array_slice: 3,
                layer_count: 2,
            };
            assert!(exact_mode27_array_storage_supported(descriptor));
        }
    }

    #[test]
    fn nonzero_base_array_mip_storage_uploads_and_publishes_bound_levels() {
        use remu_memory::{
            GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
            GuestProtection, read_guest_shared_backing,
        };

        let template = GuestTextureDescriptor {
            address: 0,
            width: 128,
            height: 128,
            pitch: 128,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        let layout = mode64k_2d_mip_chain_layout(template, 2).expect("array chain layout");
        assert_eq!(layout.byte_count, 131_072);
        let allocation_size = u64::try_from(layout.byte_count * 5).expect("allocation size");
        let mut memory = GuestAddressSpace::new(allocation_size * 2).expect("address space");
        let backing = GuestPhysicalBacking::new(allocation_size).expect("array backing");
        let mapping = memory
            .map_shared(
                GuestMapPlacement::Anywhere,
                allocation_size,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .expect("array mapping");
        let guest = GuestTextureDescriptor {
            address: mapping.get(),
            ..template
        };
        let descriptor = DecodedImageDescriptor {
            guest,
            span: ResourceSpan::for_image_view(guest, 1, 1, 13, 3, 2).expect("selected span"),
            destination_select: 0xfac,
            base_mip_level: 1,
            mip_level_count: 1,
            texture_type: 13,
            base_array_slice: 3,
            layer_count: 2,
        };

        let mut guest_raw = vec![0_u8; usize::try_from(allocation_size).unwrap()];
        let mut expected = Vec::new();
        for absolute_slice in 3_usize..5 {
            let mut ordinary = vec![0_u8; layout.byte_count];
            let mut slice_levels = Vec::new();
            for (level_index, level) in layout.levels.iter().copied().enumerate() {
                let linear = (0..level.width * level.height * layout.bytes_per_element)
                    .map(|index| {
                        u8::try_from(
                            (absolute_slice * 41 + level_index * 17 + index * 13 + index / 97)
                                & 0xff,
                        )
                        .expect("pattern byte")
                    })
                    .collect::<Vec<_>>();
                copy_linear_to_mode64k_2d_mip_level(27, &layout, level, &linear, &mut ordinary)
                    .expect("retile ordinary mip chain");
                slice_levels.push(linear);
            }
            let xor = mode27_surface_z_xor(absolute_slice);
            let mut swizzled = vec![0_u8; layout.byte_count];
            for block_base in (0..layout.byte_count).step_by(65_536) {
                for byte in 0..65_536 {
                    swizzled[block_base + (byte ^ xor)] = ordinary[block_base + byte];
                }
            }
            let slice_base = absolute_slice * layout.byte_count;
            guest_raw[slice_base..slice_base + layout.byte_count].copy_from_slice(&swizzled);
            expected.push(slice_levels);
        }
        memory
            .write(mapping, &guest_raw)
            .expect("seed array mip chains");

        let (_sender, receiver) = std::sync::mpsc::channel();
        let mut presenter = MetalPresenter::new(receiver).expect("presenter");
        presenter.per_queue_guest_batches = false;
        let device = presenter.device.as_ref().expect("Metal device").clone();
        let queue = presenter
            .command_queue
            .as_ref()
            .expect("Metal queue")
            .clone();
        let sampled = settle_gpu_sampled_upload(&device, &queue, descriptor, false, false)
            .expect("nonzero-base sampled mip upload");
        assert_eq!(sampled.texture_type(), MTLTextureType::D2Array);
        assert_eq!(sampled.array_length(), 2);
        assert_eq!(sampled.mipmap_level_count(), 1);
        for (local_slice, levels) in expected.iter().enumerate() {
            let expected_level = &levels[1];
            let mut actual = vec![0_u8; expected_level.len()];
            sampled.get_bytes_in_slice(
                actual.as_mut_ptr().cast(),
                64 * 4,
                u64::try_from(actual.len()).unwrap(),
                MTLRegion::new_2d(0, 0, 64, 64),
                0,
                u64::try_from(local_slice).unwrap(),
            );
            assert_eq!(&actual, expected_level, "sampled local slice {local_slice}");
        }
        let selected = ensure_exact_mode27_array_storage_resident(
            &device,
            &queue,
            descriptor,
            &mut presenter.resident_guest_textures,
        )
        .expect("nonzero-base storage mip upload");
        assert_eq!(selected.texture_type(), MTLTextureType::D2Array);
        assert_eq!(selected.array_length(), 2);
        assert_eq!(selected.mipmap_level_count(), 1);
        assert_eq!((selected.width(), selected.height()), (64, 64));
        let compatible_full_hierarchy = DecodedImageDescriptor {
            base_mip_level: 0,
            mip_level_count: 2,
            ..descriptor
        };
        assert_eq!(descriptor.span, compatible_full_hierarchy.span);
        assert!(
            resident_texture_for_sample(
                presenter
                    .resident_guest_textures
                    .get_mut(&guest)
                    .expect("array mip resident"),
                compatible_full_hierarchy,
            )
            .is_some(),
            "views with the same underlying mip count and selected span may reuse storage"
        );
        let incompatible_single_level = DecodedImageDescriptor {
            span: ResourceSpan::for_image_view(guest, 0, 1, 13, 3, 2)
                .expect("single-level selected span"),
            base_mip_level: 0,
            mip_level_count: 1,
            ..descriptor
        };
        assert_ne!(descriptor.span, incompatible_single_level.span);
        assert!(
            resident_texture_for_sample(
                presenter
                    .resident_guest_textures
                    .get_mut(&guest)
                    .expect("array mip resident"),
                incompatible_single_level,
            )
            .is_none(),
            "a same-key view with a different full-chain stride must not reuse Metal storage"
        );
        let resident = &presenter.resident_guest_textures[&guest];
        assert_eq!(resident.mip_level_count, 2);
        assert_eq!(resident.base_array_slice, 3);
        assert_eq!(
            rematerialisation_subresource_summary(resident).expect("array remat plan"),
            vec![
                (0, 0, 65_536, 0xc00),
                (0, 1, 0, 0xc00),
                (1, 0, 196_608, 0x200),
                (1, 1, 131_072, 0x200),
            ]
        );

        for (local_slice, levels) in expected.iter().enumerate() {
            for (level_index, expected_level) in levels.iter().enumerate() {
                let level = layout.levels[level_index];
                let mut actual = vec![0_u8; expected_level.len()];
                let row_bytes = u64::try_from(level.width * layout.bytes_per_element).unwrap();
                resident.texture.get_bytes_in_slice(
                    actual.as_mut_ptr().cast(),
                    row_bytes,
                    u64::try_from(actual.len()).unwrap(),
                    MTLRegion::new_2d(
                        0,
                        0,
                        u64::try_from(level.width).unwrap(),
                        u64::try_from(level.height).unwrap(),
                    ),
                    u64::try_from(level_index).unwrap(),
                    u64::try_from(local_slice).unwrap(),
                );
                assert_eq!(
                    &actual, expected_level,
                    "slice {local_slice} level {level_index}"
                );
            }
        }

        let mut rendered = Vec::new();
        let bound = layout.levels[1];
        for local_slice in 0_usize..2 {
            let linear = (0..bound.width * bound.height * layout.bytes_per_element)
                .map(|index| u8::try_from((0xb7 + local_slice * 23 + index * 29) & 0xff).unwrap())
                .collect::<Vec<_>>();
            selected.replace_region_in_slice(
                MTLRegion::new_2d(
                    0,
                    0,
                    u64::try_from(bound.width).unwrap(),
                    u64::try_from(bound.height).unwrap(),
                ),
                0,
                u64::try_from(local_slice).unwrap(),
                linear.as_ptr().cast(),
                u64::try_from(bound.width * layout.bytes_per_element).unwrap(),
                u64::try_from(linear.len()).unwrap(),
            );
            rendered.push(linear);
        }
        presenter.mark_resident_texture_written_at(
            guest,
            descriptor.span,
            Some(guest),
            WrittenRegion::storage_image(descriptor),
            ResidentTextureActivityStamp { work: 7, flip: 2 },
        );
        assert!(
            presenter
                .settle_resident_texture_writeback(guest, WritebackTrigger::FlipTarget)
                .expect("publish bound array mip")
        );
        if let Some(batch) = presenter.open_guest_batch.take() {
            batch.command_buffer.commit();
            batch.command_buffer.wait_until_completed();
        }

        let mut published = vec![0_u8; usize::try_from(allocation_size).unwrap()];
        assert!(read_guest_shared_backing(guest.address, &mut published));
        let selected_base = 3 * layout.byte_count;
        assert_eq!(
            &published[..selected_base],
            &guest_raw[..selected_base],
            "publishing selected BASE_ARRAY slices must not touch allocation neighbours"
        );
        for (local_slice, expected_level) in rendered.iter().enumerate() {
            let absolute_slice = 3 + local_slice;
            let xor = mode27_surface_z_xor(absolute_slice);
            let slice_base = absolute_slice * layout.byte_count;
            let actual_chain = &published[slice_base..slice_base + layout.byte_count];
            let mut ordinary = vec![0_u8; layout.byte_count];
            for block_base in (0..layout.byte_count).step_by(65_536) {
                for byte in 0..65_536 {
                    ordinary[block_base + byte] = actual_chain[block_base + (byte ^ xor)];
                }
            }
            let mut preserved_base = vec![0_u8; expected[local_slice][0].len()];
            copy_mode64k_2d_mip_level_to_linear(
                27,
                &layout,
                layout.levels[0],
                &ordinary,
                &mut preserved_base,
            )
            .expect("detile preserved base mip");
            assert_eq!(
                preserved_base, expected[local_slice][0],
                "publishing mip1 must preserve slice {local_slice} mip0"
            );
            let mut decoded = vec![0_u8; expected_level.len()];
            copy_mode64k_2d_mip_level_to_linear(27, &layout, bound, &ordinary, &mut decoded)
                .expect("detile published array mip");
            assert_eq!(
                &decoded, expected_level,
                "published local slice {local_slice}"
            );
        }

        let guest_before_rematerialisation = published;
        let selected_address = descriptor.span.expect("selected array span").address();
        let tenant = GuestTextureDescriptor {
            address: selected_address,
            width: 256,
            height: 256,
            pitch: 256,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        presenter
            .ensure_resident_texture(&device, tenant)
            .expect("full selected-span tenant resident");
        let tenant_linear = (0..256 * 256 * 4)
            .map(|index| u8::try_from((index * 43 + index / 31 + 0x5d) & 0xff).unwrap())
            .collect::<Vec<_>>();
        presenter.resident_guest_textures[&tenant]
            .texture
            .replace_region(
                MTLRegion::new_2d(0, 0, 256, 256),
                0,
                tenant_linear.as_ptr().cast(),
                256 * 4,
            );
        presenter.mark_resident_texture_written(tenant);
        assert!(
            presenter
                .rematerialise_stale_resident_for_read(guest)
                .expect("array hierarchy rematerialisation")
        );
        let remat_batch = presenter
            .open_guest_batch
            .take()
            .expect("array rematerialisation stays ordered in the guest batch");
        remat_batch.command_buffer.commit();
        remat_batch.command_buffer.wait_until_completed();
        let target = &presenter.resident_guest_textures[&guest];
        assert!(target.gpu_written());
        let mut actual_rematerialised = vec![0_u8; 64 * 64 * 4];
        target.texture.get_bytes_in_slice(
            actual_rematerialised.as_mut_ptr().cast(),
            64 * 4,
            u64::try_from(actual_rematerialised.len()).unwrap(),
            MTLRegion::new_2d(0, 0, 64, 64),
            1,
            0,
        );
        let mut tenant_raw = vec![0_u8; 4 * 65_536];
        copy_linear_to_tiled_at_z(&tenant_linear, tenant, &mut tenant_raw, 0)
            .expect("retile tenant oracle");
        let xor = mode27_surface_z_xor(3);
        let mut target_ordinary = vec![0_u8; layout.byte_count];
        for byte in 0..65_536 {
            target_ordinary[byte] = tenant_raw[byte ^ xor];
        }
        let mut expected_rematerialised = vec![0_u8; 64 * 64 * 4];
        copy_mode64k_2d_mip_level_to_linear(
            27,
            &layout,
            layout.levels[1],
            &target_ordinary,
            &mut expected_rematerialised,
        )
        .expect("detile absolute-slice rematerialisation oracle");
        assert_eq!(actual_rematerialised, expected_rematerialised);
        let mut guest_after_rematerialisation =
            vec![0_u8; usize::try_from(allocation_size).unwrap()];
        assert!(read_guest_shared_backing(
            guest.address,
            &mut guest_after_rematerialisation
        ));
        assert_eq!(
            guest_after_rematerialisation, guest_before_rematerialisation,
            "GPU rematerialisation must not publish the tenant into guest RAM"
        );
    }

    fn exact_array_gpu_tenant_production_route(opcode: &'static str) {
        use remu_memory::{
            GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
            GuestProtection, read_guest_shared_backing,
        };

        let template = GuestTextureDescriptor {
            address: 0,
            width: 128,
            height: 128,
            pitch: 128,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        let layout = mode64k_2d_mip_chain_layout(template, 2).expect("two-level SDK layout");
        assert_eq!(layout.byte_count, 131_072);
        let allocation_size = u64::try_from(5 * layout.byte_count).unwrap();
        let mut memory = GuestAddressSpace::new(allocation_size * 2).expect("address space");
        let backing = GuestPhysicalBacking::new(allocation_size).expect("array backing");
        let mapping = memory
            .map_shared(
                GuestMapPlacement::Anywhere,
                allocation_size,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .expect("array mapping");
        let address = mapping.get();
        let guest_raw = (0..usize::try_from(allocation_size).unwrap())
            .map(|index| u8::try_from((index * 7 + index / 29 + 0x31) & 0xff).unwrap())
            .collect::<Vec<_>>();
        memory.write(mapping, &guest_raw).expect("seed guest array");

        let encoded_width = template.width - 1;
        let mut words = vec![0_u32; 8];
        words[0] = u32::try_from(address >> 8).expect("low image address fits");
        words[1] = u32::try_from(address >> 40).expect("high image address fits")
            | (130 << 20)
            | ((encoded_width & 0x3) << 30);
        words[2] = (encoded_width >> 2) | ((template.height - 1) << 14);
        words[3] = 0xfac | (1 << 16) | (27 << 20) | (13 << 28);
        words[4] = (3 << 16) | 4;
        let descriptor = decode_image_descriptor(&words).expect("exact array mip descriptor");
        let evaluation = ShaderEvaluation {
            initial_scalar_registers: Vec::new().into(),
            scalar_registers: Vec::new().into(),
            scalar_memory_reads: Vec::new(),
            image_bindings: vec![ImageBinding {
                pc: 0x40,
                opcode: Opcode::Known(opcode),
                control: ImageControl {
                    dmask: 0xf,
                    vector_address: 0,
                    address_registers: vec![0, 1, 2],
                    vector_data: 4,
                    scalar_resource: 0,
                    scalar_sampler: 0,
                    dimension: 5,
                    is_array: true,
                    glc: false,
                    slc: false,
                    a16: false,
                    d16: false,
                    unorm: false,
                },
                resource_descriptor: words.into(),
                sampler_descriptor: if opcode == "ImageSample" {
                    vec![0_u32; 4].into()
                } else {
                    Vec::new().into()
                },
                mip_level: None,
            }],
            global_memory_bindings: Vec::new(),
            compute_system_registers: None,
            vertex_inputs: Vec::new(),
        };

        let (_sender, receiver) = std::sync::mpsc::channel();
        let mut presenter = MetalPresenter::new(receiver).expect("presenter");
        presenter.per_queue_guest_batches = false;
        let device = presenter.device.as_ref().expect("Metal device").clone();
        let queue = presenter
            .command_queue
            .as_ref()
            .expect("Metal queue")
            .clone();
        if opcode == "ImageStore" {
            let short_descriptor = DecodedImageDescriptor {
                span: ResourceSpan::for_image_view(
                    descriptor.guest,
                    0,
                    1,
                    13,
                    descriptor.base_array_slice,
                    descriptor.layer_count,
                )
                .expect("short array hierarchy span"),
                mip_level_count: 1,
                ..descriptor
            };
            ensure_exact_mode27_array_storage_resident(
                &device,
                &queue,
                short_descriptor,
                &mut presenter.resident_guest_textures,
            )
            .expect("clean short hierarchy");
            assert_eq!(
                presenter.resident_guest_textures[&descriptor.guest].mip_level_count,
                1
            );
        }
        let tenant = GuestTextureDescriptor {
            address: descriptor.span.expect("selected span").address(),
            width: 256,
            height: 256,
            pitch: 256,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        presenter
            .ensure_resident_texture(&device, tenant)
            .expect("selected-span tenant resident");
        let tenant_linear = (0..256 * 256 * 4)
            .map(|index| u8::try_from((index * 43 + index / 31 + 0x5d) & 0xff).unwrap())
            .collect::<Vec<_>>();
        presenter.resident_guest_textures[&tenant]
            .texture
            .replace_region(
                MTLRegion::new_2d(0, 0, 256, 256),
                0,
                tenant_linear.as_ptr().cast(),
                256 * 4,
            );
        presenter.mark_resident_texture_written(tenant);

        if opcode == "ImageStore" {
            presenter
                .ensure_mipmapped_storage_texture(&device, descriptor)
                .expect("production storage pre-acquire rematerialises the array hierarchy");
        } else {
            presenter
                .synchronize_sampled_aliases(&[&evaluation], &[])
                .expect("production sampled pre-sync rematerialises the array hierarchy");
        }
        let uploaded = super::upload_image_textures(
            "test",
            &device,
            &queue,
            &evaluation,
            &mut presenter.resident_guest_textures,
            &presenter.depth_array_textures,
            &presenter.multisample_textures,
            &mut presenter.sampled_texture_cache,
            &mut presenter.sampled_texture_write_watches,
            &mut presenter.fallback_texture_cache,
            &mut presenter.next_sampled_texture_use,
            ResidentTextureActivityStamp::default(),
            &[],
        )
        .expect("production image upload binds the rematerialised hierarchy");
        assert_eq!(uploaded.len(), 1);
        assert_eq!(uploaded[0].texture.texture_type(), MTLTextureType::D2Array);
        assert_eq!(uploaded[0].texture.array_length(), 2);
        assert_eq!(uploaded[0].texture.mipmap_level_count(), 2);
        assert_eq!(uploaded[0].selected_resident, Some(descriptor.guest));
        assert_eq!(uploaded[0].writable, opcode == "ImageStore");

        let batch = presenter
            .open_guest_batch
            .take()
            .expect("rematerialisation remains in the ordered guest batch");
        batch.command_buffer.commit();
        batch.command_buffer.wait_until_completed();
        let target = &presenter.resident_guest_textures[&descriptor.guest];
        assert_eq!(target.mip_level_count, 2);
        assert!(target.gpu_written());
        assert!(
            target
                .owned
                .covers(&descriptor.span.expect("selected span"))
        );
        assert!(!presenter.resident_guest_textures[&tenant].is_live());

        let mut actual = vec![0_u8; 64 * 64 * 4];
        target.texture.get_bytes_in_slice(
            actual.as_mut_ptr().cast(),
            64 * 4,
            u64::try_from(actual.len()).unwrap(),
            MTLRegion::new_2d(0, 0, 64, 64),
            1,
            0,
        );
        let mut tenant_raw = vec![0_u8; 4 * 65_536];
        copy_linear_to_tiled_at_z(&tenant_linear, tenant, &mut tenant_raw, 0)
            .expect("retile tenant oracle");
        let mut target_ordinary = vec![0_u8; layout.byte_count];
        let xor = mode27_surface_z_xor(3);
        for byte in 0..65_536 {
            target_ordinary[byte] = tenant_raw[byte ^ xor];
        }
        let mut expected = vec![0_u8; actual.len()];
        copy_mode64k_2d_mip_level_to_linear(
            27,
            &layout,
            layout.levels[1],
            &target_ordinary,
            &mut expected,
        )
        .expect("detile rematerialisation oracle");
        assert_eq!(actual, expected);

        let mut guest_after = vec![0_u8; guest_raw.len()];
        assert!(read_guest_shared_backing(address, &mut guest_after));
        assert_eq!(
            guest_after, guest_raw,
            "GPU reconstruction must not publish RAM"
        );
    }

    #[test]
    fn sampled_array_mip_first_create_rematerialises_external_gpu_tenant() {
        exact_array_gpu_tenant_production_route("ImageSample");
    }

    #[test]
    fn storage_array_mip_first_create_rematerialises_external_gpu_tenant() {
        exact_array_gpu_tenant_production_route("ImageStore");
    }

    fn exact_array_presenter_refreshes_current_guest(remap: bool) {
        use remu_memory::{
            GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
            GuestProtection,
        };

        let allocation_size = 2 * 131_072_u64;
        let mut memory = GuestAddressSpace::new(4 * allocation_size).expect("address space");
        let backing = GuestPhysicalBacking::new(allocation_size).expect("array backing");
        let mapping = memory
            .map_shared(
                GuestMapPlacement::Anywhere,
                allocation_size,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .expect("array mapping");
        memory
            .write(
                mapping,
                &vec![0x11; usize::try_from(allocation_size).unwrap()],
            )
            .expect("seed original array");
        let guest = GuestTextureDescriptor {
            address: mapping.get(),
            width: 128,
            height: 128,
            pitch: 128,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        let descriptor = DecodedImageDescriptor {
            guest,
            span: ResourceSpan::for_image_view(guest, 0, 2, 13, 0, 2).expect("exact array span"),
            destination_select: 0xfac,
            base_mip_level: 0,
            mip_level_count: 2,
            texture_type: 13,
            base_array_slice: 0,
            layer_count: 2,
        };
        let (_sender, receiver) = std::sync::mpsc::channel();
        let mut presenter = MetalPresenter::new(receiver).expect("presenter");
        presenter.per_queue_guest_batches = false;
        let device = presenter.device.as_ref().expect("Metal device").clone();
        presenter
            .ensure_mipmapped_storage_texture(&device, descriptor)
            .expect("initial ordered array hierarchy");
        let batch = presenter
            .open_guest_batch
            .take()
            .expect("initial upload batch");
        batch.command_buffer.commit();
        batch.command_buffer.wait_until_completed();
        let original = presenter.resident_guest_textures[&guest].texture.clone();

        let expected_byte = if remap { 0x6c } else { 0xa5 };
        let current_descriptor = if remap {
            presenter
                .resident_guest_textures
                .get_mut(&guest)
                .expect("original array resident")
                .note_gpu_authored_whole_span();
            memory
                .unmap(mapping, allocation_size)
                .expect("replace array mapping");
            let replacement = GuestPhysicalBacking::new(allocation_size).expect("replacement");
            memory
                .map_shared(
                    GuestMapPlacement::Fixed(mapping),
                    allocation_size,
                    GUEST_PAGE_SIZE,
                    GuestProtection::READ_WRITE,
                    &replacement,
                    0,
                )
                .expect("replacement mapping");
            memory
                .write(
                    mapping,
                    &vec![expected_byte; usize::try_from(allocation_size).unwrap()],
                )
                .expect("seed replacement array");
            DecodedImageDescriptor {
                span: ResourceSpan::for_image_view(guest, 0, 2, 13, 0, 2)
                    .expect("replacement array span"),
                ..descriptor
            }
        } else {
            memory
                .write(
                    mapping,
                    &vec![expected_byte; usize::try_from(allocation_size).unwrap()],
                )
                .expect("dirty current array");
            descriptor
        };
        let selected = presenter
            .ensure_mipmapped_storage_texture(&device, current_descriptor)
            .expect("ordered array refresh uses current guest mapping");
        let batch = presenter
            .open_guest_batch
            .take()
            .expect("refresh upload batch");
        batch.command_buffer.commit();
        batch.command_buffer.wait_until_completed();
        assert_ne!(selected.as_ptr(), original.as_ptr());
        let mut pixel = [0_u8; 4];
        selected.get_bytes_in_slice(
            pixel.as_mut_ptr().cast(),
            4,
            4,
            MTLRegion::new_2d(0, 0, 1, 1),
            0,
            0,
        );
        assert_eq!(pixel, [expected_byte; 4]);
        assert!(!presenter.resident_guest_textures[&guest].gpu_written());
    }

    #[test]
    fn ordered_array_mip_refreshes_cpu_dirty_clean_resident() {
        exact_array_presenter_refreshes_current_guest(false);
    }

    #[test]
    fn ordered_array_mip_retires_gpu_debt_from_replaced_mapping() {
        exact_array_presenter_refreshes_current_guest(true);
    }

    fn ordered_array_mip_refuses_cpu_dirty_gpu_tenant(existing_target: bool) {
        use remu_memory::{
            GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
            GuestProtection,
        };

        let allocation_size = 2 * 131_072_u64;
        let mut memory = GuestAddressSpace::new(4 * allocation_size).expect("address space");
        let backing = GuestPhysicalBacking::new(allocation_size).expect("array backing");
        let mapping = memory
            .map_shared(
                GuestMapPlacement::Anywhere,
                allocation_size,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .expect("array mapping");
        let guest = GuestTextureDescriptor {
            address: mapping.get(),
            width: 128,
            height: 128,
            pitch: 128,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        let descriptor = DecodedImageDescriptor {
            guest,
            span: ResourceSpan::for_image_view(guest, 0, 2, 13, 0, 2).expect("exact array span"),
            destination_select: 0xfac,
            base_mip_level: 0,
            mip_level_count: 2,
            texture_type: 13,
            base_array_slice: 0,
            layer_count: 2,
        };
        let (_sender, receiver) = std::sync::mpsc::channel();
        let mut presenter = MetalPresenter::new(receiver).expect("presenter");
        presenter.per_queue_guest_batches = false;
        let device = presenter.device.as_ref().expect("Metal device").clone();
        if existing_target {
            presenter
                .ensure_mipmapped_storage_texture(&device, descriptor)
                .expect("initial exact array target");
            let batch = presenter
                .open_guest_batch
                .take()
                .expect("initial target batch");
            batch.command_buffer.commit();
            batch.command_buffer.wait_until_completed();
        }
        let tenant = GuestTextureDescriptor {
            address: guest.address,
            width: 256,
            height: 256,
            pitch: 256,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        presenter
            .ensure_resident_texture(&device, tenant)
            .expect("full-span tenant");
        presenter.resident_guest_textures[&tenant]
            .texture
            .replace_region(
                MTLRegion::new_2d(0, 0, 256, 256),
                0,
                vec![0x3c_u8; 256 * 256 * 4].as_ptr().cast(),
                256 * 4,
            );
        presenter.mark_resident_texture_written(tenant);
        memory
            .write(
                mapping,
                &vec![0xa5; usize::try_from(allocation_size).unwrap()],
            )
            .expect("CPU supersedes tenant bytes");

        let error = presenter
            .ensure_mipmapped_storage_texture(&device, descriptor)
            .expect_err("CPU-dirty GPU tenant must never overwrite newer guest RAM");
        assert!(
            error.contains("CPU-modified") && error.contains("GPU-current"),
            "unexpected dirty-tenant refusal: {error}"
        );
        assert!(presenter.open_guest_batch.is_none());
    }

    #[test]
    fn ordered_array_mip_first_create_refuses_cpu_dirty_gpu_tenant() {
        ordered_array_mip_refuses_cpu_dirty_gpu_tenant(false);
    }

    #[test]
    fn ordered_array_mip_partial_existing_refuses_cpu_dirty_gpu_tenant() {
        ordered_array_mip_refuses_cpu_dirty_gpu_tenant(true);
    }

    #[test]
    fn array_resident_identity_requires_the_exact_underlying_mip_hierarchy() {
        use remu_memory::{
            GUEST_PAGE_SIZE, GuestAddressSpace, GuestMapPlacement, GuestPhysicalBacking,
            GuestProtection,
        };

        let template = GuestTextureDescriptor {
            address: 0,
            width: 512,
            height: 256,
            pitch: 512,
            tile_mode: 27,
            pixel_format: MTLPixelFormat::RGBA8Unorm,
            bytes_per_pixel: 4,
        };
        let full_stride = mode64k_2d_mip_chain_layout(template, 5)
            .expect("five-level SDK layout")
            .byte_count;
        assert_eq!(full_stride, 786_432);
        let allocation_size = u64::try_from(full_stride * 2).expect("array allocation size");
        let mut memory = GuestAddressSpace::new(allocation_size * 2).expect("address space");
        let backing = GuestPhysicalBacking::new(allocation_size).expect("array backing");
        let mapping = memory
            .map_shared(
                GuestMapPlacement::Anywhere,
                allocation_size,
                GUEST_PAGE_SIZE,
                GuestProtection::READ_WRITE,
                &backing,
                0,
            )
            .expect("array mapping");
        let guest = GuestTextureDescriptor {
            address: mapping.get(),
            ..template
        };
        let image = |base_mip_level, mip_level_count| DecodedImageDescriptor {
            guest,
            span: ResourceSpan::for_image_view(guest, base_mip_level, mip_level_count, 13, 0, 2)
                .expect("selected hierarchy span"),
            destination_select: 0xfac,
            base_mip_level,
            mip_level_count,
            texture_type: 13,
            base_array_slice: 0,
            layer_count: 2,
        };

        let (_sender, receiver) = std::sync::mpsc::channel();
        let mut presenter = MetalPresenter::new(receiver).expect("presenter");
        let device = presenter.device.as_ref().expect("Metal device").clone();
        let queue = presenter
            .command_queue
            .as_ref()
            .expect("Metal queue")
            .clone();

        let full = image(0, 5);
        ensure_exact_mode27_array_storage_resident(
            &device,
            &queue,
            full,
            &mut presenter.resident_guest_textures,
        )
        .expect("five-level array resident");
        let retained_full = presenter.resident_guest_textures[&guest].texture.clone();
        assert_eq!(presenter.resident_guest_textures[&guest].mip_level_count, 5);

        let short = image(0, 2);
        assert_ne!(full.span, short.span);
        ensure_exact_mode27_array_storage_resident(
            &device,
            &queue,
            short,
            &mut presenter.resident_guest_textures,
        )
        .expect("two-level hierarchy rebuild");
        let short_resident = &presenter.resident_guest_textures[&guest];
        assert_eq!(short_resident.mip_level_count, 2);
        assert_ne!(short_resident.texture.as_ptr(), retained_full.as_ptr());

        let tail_view = image(4, 1);
        assert_eq!(full.span, tail_view.span);
        let selected = ensure_exact_mode27_array_storage_resident(
            &device,
            &queue,
            tail_view,
            &mut presenter.resident_guest_textures,
        )
        .expect("base-four view rebuilds its complete five-level hierarchy");
        assert_eq!(presenter.resident_guest_textures[&guest].mip_level_count, 5);
        assert_eq!(selected.mipmap_level_count(), 1);
        assert_eq!((selected.width(), selected.height()), (32, 16));
        let full_pointer = presenter.resident_guest_textures[&guest].texture.as_ptr();
        ensure_exact_mode27_array_storage_resident(
            &device,
            &queue,
            full,
            &mut presenter.resident_guest_textures,
        )
        .expect("same-underlying full view reuse");
        assert_eq!(
            presenter.resident_guest_textures[&guest].texture.as_ptr(),
            full_pointer
        );
    }
}
