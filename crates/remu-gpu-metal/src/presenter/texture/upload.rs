// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CachedSampledTexture, GuestTextureDescriptor, GuestTextureWriteWatch, Mode27WorkOrigin,
    MultisampleResidentTexture, MultisampleTextureKey, ResidentGuestTexture,
    SAMPLED_TEXTURE_CACHE_CAPACITY, SAMPLED_TEXTURE_CACHE_CAPACITY_EVICTIONS,
    SAMPLED_TEXTURE_CACHE_DIRTY_RELOADS, SAMPLED_TEXTURE_CACHE_HITS, SAMPLED_TEXTURE_CACHE_MISSES,
    SampledTextureKey, UploadedImageTexture, WriteWatchRefresh, canonical_texture_bindings,
    create_exact_mode27_sampled_mip_view, create_sampled_guest_texture, decode_image_descriptor,
    decode_sampled_texture_linear_from_raw, ensure_mipmapped_resident_texture_in_map,
    ensure_resident_texture_in_map, fallback_image_descriptor, guest_texture_byte_count,
    guest_texture_descriptors_have_same_layout, guest_texture_descriptors_overlap,
    guest_texture_descriptors_share_storage, is_storage_image_opcode, is_writable_image_opcode,
    persistent_sampled_texture_write_watch, refresh_resident_texture_write_watch,
    refresh_sampled_texture_write_watch, render_resolve_candidate_matches,
    resident_component_reordered_sample_is_eligible, resident_depth_sample_is_eligible,
    resident_texture_for_sample, retire_sampled_texture, sampled_texture_cache_needs_reload,
    sampled_texture_key, sampled_texture_uses_exact_mode27_mips,
    sampler_uses_normalized_coordinates, snapshot_watched_sampled_texture_raw,
    storage_image_requires_mip_hierarchy, texture_element_layout, trace_target_sampling,
};
use metal::{Device, MTLPixelFormat, Texture};
use remu_memory::guest_shared_mapping_identity;
use remu_shader::{ShaderEvaluation, image_is_comparison};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) fn comparison_depth32_upload(
    format: MTLPixelFormat,
    linear: &[u8],
) -> Result<Option<Vec<u8>>, String> {
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

pub(crate) fn upload_image_textures(
    device: &Device,
    queue: &metal::CommandQueue,
    evaluation: &ShaderEvaluation,
    resident_guest_textures: &mut HashMap<GuestTextureDescriptor, ResidentGuestTexture>,
    multisample_textures: &HashMap<MultisampleTextureKey, MultisampleResidentTexture>,
    sampled_texture_cache: &mut HashMap<SampledTextureKey, CachedSampledTexture>,
    sampled_texture_write_watches: &mut HashMap<SampledTextureKey, Arc<GuestTextureWriteWatch>>,
    fallback_texture_cache: &mut HashMap<SampledTextureKey, Texture>,
    sampled_texture_admission_order: &mut VecDeque<(SampledTextureKey, u64)>,
    next_sampled_texture_admission_id: &mut u64,
) -> Result<Vec<UploadedImageTexture>, String> {
    canonical_texture_bindings(evaluation)
        .into_iter()
        .map(|binding| {
            let writable = is_writable_image_opcode(binding.opcode);
            let storage = is_storage_image_opcode(binding.opcode);
            let comparison = image_is_comparison(binding.opcode);
            let decoded = decode_image_descriptor(&binding.resource_descriptor);
            let fallback = decoded.is_err();
            let descriptor = decoded
                .or_else(|_| fallback_image_descriptor(binding))
                .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
            if trace_target_sampling()
                && descriptor.guest.width == 1_920
                && descriptor.guest.height == 1_080
            {
                static LOGGED: OnceLock<Mutex<HashSet<(u64, u32, u32, u32, bool, bool)>>> =
                    OnceLock::new();
                let normalized_coordinates =
                    sampler_uses_normalized_coordinates(&binding.sampler_descriptor);
                let key = (
                    descriptor.guest.address,
                    descriptor.guest.tile_mode,
                    descriptor.destination_select,
                    descriptor.mip_level_count,
                    comparison,
                    normalized_coordinates,
                );
                let mut logged = LOGGED
                    .get_or_init(|| Mutex::new(HashSet::new()))
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if logged.len() < 128 && logged.insert(key) {
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
                        .map(|resident| (resident.gpu_written, resident.content_version));
                    let aliases = resident_guest_textures
                        .values()
                        .filter(|resident| {
                            guest_texture_descriptors_overlap(
                                resident.descriptor,
                                descriptor.guest,
                            )
                        })
                        .map(|resident| {
                            (
                                resident.descriptor.address,
                                resident.descriptor.width,
                                resident.descriptor.height,
                                resident.descriptor.pixel_format,
                                resident.descriptor.tile_mode,
                                resident.gpu_written,
                                resident.content_version,
                            )
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
                                        (resident.gpu_written, resident.content_version)
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
            if matches!(descriptor.texture_type, 9 | 10) {
                let replaced = if let Some(resident) = resident_guest_textures
                .get_mut(&descriptor.guest)
                .filter(|resident| {
                    resident.texture_type == descriptor.texture_type
                        && resident.layer_count == descriptor.layer_count
                        && resident.mip_level_count >= underlying_mip_count
                })
                {
                    refresh_resident_texture_write_watch(resident)?
                        == WriteWatchRefresh::Replaced
                } else {
                    false
                };
                if replaced {
                    resident_guest_textures.remove(&descriptor.guest);
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
                            resident.gpu_written.then_some((
                                resident.content_version,
                                key.descriptor,
                                resident.texture.clone(),
                            ))
                        })
                        .max_by_key(|(version, _, _)| *version)
                })
                .flatten();
            let resident_descriptor = matches!(descriptor.texture_type, 9 | 10)
                .then(|| {
                    (!comparison)
                        .then(|| {
                            resident_guest_textures
                                .get(&descriptor.guest)
                                .filter(|resident| {
                                    resident.texture_type == descriptor.texture_type
                                        && resident.layer_count == descriptor.layer_count
                                        && resident.mip_level_count >= underlying_mip_count
                                })
                                .map(|resident| resident.descriptor)
                        })
                        .flatten()
                        .or_else(|| {
                            (descriptor.texture_type == 9).then(|| {
                                resident_guest_textures
                                    .values()
                                    .filter(|resident| {
                                        resident.texture_type == 9
                                            && resident.layer_count == 1
                                            && resident.mip_level_count >= underlying_mip_count
                                            && ((!comparison
                                                && (guest_texture_descriptors_share_storage(
                                                    resident.descriptor,
                                                    descriptor.guest,
                                                ) || resident_component_reordered_sample_is_eligible(
                                                    resident,
                                                    descriptor,
                                                    comparison,
                                                )))
                                                || resident_depth_sample_is_eligible(
                                                    resident, descriptor,
                                                ))
                                    })
                                    .max_by_key(|resident| resident.content_version)
                                    .map(|resident| resident.descriptor)
                            })?
                        })
                })
                .flatten();
            let resident_resolve = resident_descriptor.and_then(|resident_descriptor| {
                let resident = resident_guest_textures.get_mut(&resident_descriptor)?;
                let version = resident.content_version;
                resident_texture_for_sample(resident, descriptor)
                    .map(|texture| (version, texture))
            });
            // A guest descriptor can retain an exact single-sample resident from an earlier
            // render phase while the current frame is produced through a distinct MSAA target
            // with the same layout. Address equality alone is therefore not sufficient: choose
            // whichever compatible GPU image was written most recently. This also naturally
            // retains an exact post-process result when it is newer than the MSAA resolve.
            let select_multisample = match (&resident_resolve, &multisample_resolve) {
                (Some((resident_version, _)), Some((multisample_version, _, _))) => {
                    multisample_version > resident_version
                }
                (None, Some(_)) => true,
                _ => false,
            };
            if trace_target_sampling()
                && descriptor.guest.width == 1_920
                && descriptor.guest.height == 1_080
                && select_multisample
                && let Some((version, source, _)) = multisample_resolve.as_ref()
            {
                static LOGGED_RESOLVES: OnceLock<Mutex<HashSet<(u64, u64)>>> = OnceLock::new();
                let key = (descriptor.guest.address, source.address);
                let mut logged = LOGGED_RESOLVES
                    .get_or_init(|| Mutex::new(HashSet::new()))
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if logged.len() < 256 && logged.insert(key) {
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
                    .map(|(_, texture)| texture.clone())
            };
            let texture = if fallback {
                let key = sampled_texture_key(descriptor, comparison);
                if let Some(texture) = fallback_texture_cache.get(&key) {
                    texture.clone()
                } else {
                    let (_, _, _, bytes_per_element) =
                        texture_element_layout(descriptor.guest)?;
                    let byte_count = bytes_per_element
                        .checked_mul(
                            usize::try_from(descriptor.layer_count)
                                .map_err(|_| "fallback image layer count does not fit usize")?,
                        )
                        .ok_or("fallback image byte count overflows")?;
                    let mut linear = vec![0; byte_count];
                    if matches!(
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
                    )?;
                    texture.set_label("REMu fallback guest image");
                    fallback_texture_cache.insert(key, texture.clone());
                    texture
                }
            } else if storage && descriptor.texture_type == 9 {
                if storage_image_requires_mip_hierarchy(binding, descriptor) {
                    ensure_mipmapped_resident_texture_in_map(
                        device,
                        descriptor,
                        resident_guest_textures,
                        Mode27WorkOrigin::StorageCreate,
                    )?
                } else {
                    ensure_resident_texture_in_map(
                        device,
                        descriptor.guest,
                        resident_guest_textures,
                        true,
                    )?
                }
            } else if let Some(texture) = render_resolve {
                texture
            } else {
                let key = sampled_texture_key(descriptor, comparison);
                let reload = if let Some(cached) = sampled_texture_cache.get_mut(&key) {
                    let refresh = refresh_sampled_texture_write_watch(
                        key,
                        cached,
                        sampled_texture_write_watches,
                    )?;
                    sampled_texture_cache_needs_reload(refresh, cached.write_watch.is_dirty())
                } else {
                    false
                };
                if reload {
                    // CPU-produced images such as AvPlayer's rotating NV12 frames can be updated
                    // immediately before a draw without an intervening ACQUIRE_MEM packet. The
                    // write watch has already observed that store, so retaining the Metal cache
                    // here would sample the zero-filled/predecessor frame forever. Rebuild from a
                    // stable watched snapshot; the persistent watch is shared with the new entry
                    // and is rearmed by snapshot_watched_sampled_texture_raw.
                    if let Some(retired) = sampled_texture_cache.remove(&key) {
                        SAMPLED_TEXTURE_CACHE_DIRTY_RELOADS.fetch_add(1, Ordering::Relaxed);
                        retire_sampled_texture(retired);
                    }
                }
                if let Some(cached) = sampled_texture_cache.get(&key) {
                    SAMPLED_TEXTURE_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                    cached.texture.clone()
                } else {
                    SAMPLED_TEXTURE_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
                    let write_watch = persistent_sampled_texture_write_watch(
                        key,
                        descriptor,
                        sampled_texture_write_watches,
                    )
                        .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
                    let raw = snapshot_watched_sampled_texture_raw(descriptor, &write_watch)
                        .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
                    let texture =
                        if sampled_texture_uses_exact_mode27_mips(descriptor) {
                            create_exact_mode27_sampled_mip_view(
                                device,
                                descriptor,
                                &raw,
                                comparison,
                            )
                            .map_err(|error| {
                                format!("image at PC {:#x}: {error}", binding.pc)
                            })?
                        } else {
                            let linear =
                                decode_sampled_texture_linear_from_raw(descriptor, &raw).map_err(
                                    |error| format!("image at PC {:#x}: {error}", binding.pc),
                                )?;
                            create_sampled_guest_texture(
                                device,
                                queue,
                                descriptor.guest,
                                &linear,
                                descriptor.base_mip_level,
                                descriptor.mip_level_count,
                                descriptor.texture_type,
                                descriptor.layer_count,
                                comparison,
                            )?
                        };
                    texture.set_label(&format!(
                        "REMu sampled guest image {:#x}",
                        descriptor.guest.address
                    ));
                    while sampled_texture_cache.len() >= SAMPLED_TEXTURE_CACHE_CAPACITY {
                        let Some((old_key, admission_id)) =
                            sampled_texture_admission_order.pop_front()
                        else {
                            break;
                        };
                        let is_current = sampled_texture_cache
                            .get(&old_key)
                            .is_some_and(|cached| cached.admission_id == admission_id);
                        if is_current {
                            if let Some(retired) = sampled_texture_cache.remove(&old_key) {
                                SAMPLED_TEXTURE_CACHE_CAPACITY_EVICTIONS
                                    .fetch_add(1, Ordering::Relaxed);
                                retire_sampled_texture(retired);
                            }
                        }
                    }
                    let admission_id = *next_sampled_texture_admission_id;
                    *next_sampled_texture_admission_id =
                        next_sampled_texture_admission_id.wrapping_add(1);
                    sampled_texture_cache.insert(
                        key,
                        CachedSampledTexture {
                            texture: texture.clone(),
                            write_watch,
                            admission_id,
                        },
                    );
                    sampled_texture_admission_order.push_back((key, admission_id));
                    texture
                }
            };
            Ok(UploadedImageTexture {
                texture,
                descriptor,
                // The resident 2D path has exact guest writeback today. A writable 3D image stays
                // resident in the shared sampled-image cache so later GPU reads see its contents;
                // it must not invalidate that cache through the 2D writeback path.
                writable: !fallback && writable && descriptor.texture_type == 9,
            })
        })
        .collect()
}
