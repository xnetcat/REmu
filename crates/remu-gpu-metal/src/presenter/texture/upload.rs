// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::SharedEventDrawTextureOverride;
use super::super::sampled_texture_cache_byte_capacity;
use super::{
    CachedSampledTexture, DepthArrayKey, DepthArrayResidentTexture, GuestTextureDescriptor,
    GuestTextureWriteWatch, Mode27WorkOrigin, MultisampleResidentTexture, MultisampleTextureKey,
    ResidentTextureActivityStamp, ResidentTextureMap, SAMPLED_TEXTURE_CACHE_CAPACITY,
    SAMPLED_TEXTURE_CACHE_CAPACITY_EVICTIONS, SAMPLED_TEXTURE_CACHE_DIRTY_RELOADS,
    SAMPLED_TEXTURE_CACHE_HITS, SAMPLED_TEXTURE_CACHE_MISSES, SampledTextureKey, SpanQueryOrder,
    UploadedImageTexture, WriteWatchRefresh, canonical_texture_bindings,
    create_sampled_guest_texture, create_sampled_guest_texture_from_guest_gpu,
    decode_image_descriptor, ensure_mipmapped_resident_texture_in_map,
    ensure_resident_texture_in_map, fallback_image_descriptor, guest_texture_byte_count,
    guest_texture_descriptors_have_same_layout, guest_texture_descriptors_overlap,
    guest_texture_descriptors_share_depth_sample_storage,
    guest_texture_descriptors_share_gpu_sample_storage, guest_texture_descriptors_share_storage,
    image_resource_descriptor_is_null, is_storage_image_opcode, is_writable_image_opcode,
    persistent_sampled_texture_write_watch, refresh_resident_texture_write_watch,
    refresh_sampled_texture_write_watch, render_resolve_candidate_matches,
    resident_color_view_sample_is_eligible, resident_depth_sample_is_eligible,
    resident_texture_for_sample, retire_sampled_texture, sampled_texture_cache_needs_reload,
    sampled_texture_key, sampled_texture_supports_access, sampler_uses_normalized_coordinates,
    storage_image_requires_mip_hierarchy, texture_element_layout,
    trace_alias_correlated_image_bindings, trace_glyph_atlas, trace_resident_texture_removal,
    trace_target_sampling,
};
use super::super::capture_target_sample_textures;
use metal::{Device, MTLPixelFormat, Texture};
use remu_memory::guest_shared_mapping_identity;
use remu_shader::{ShaderEvaluation, image_is_comparison};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

fn trace_glyph_atlas_linear(
    descriptor: GuestTextureDescriptor,
    destination_select: u32,
    raw: &[u8],
    linear: &[u8],
) {
    if !trace_glyph_atlas()
        || !matches!(
            descriptor.pixel_format,
            MTLPixelFormat::R8Unorm | MTLPixelFormat::A8Unorm
        )
    {
        return;
    }
    // K1: lock-free bounded sampler (was a mutex-guarded HashSet dedup).
    static LOGGED: AtomicUsize = AtomicUsize::new(0);
    if LOGGED.fetch_add(1, Ordering::Relaxed) >= 64 {
        return;
    }

    let mut histogram = [0_usize; 256];
    for &value in linear {
        histogram[usize::from(value)] += 1;
    }
    let occupied_values = histogram.iter().filter(|&&count| count != 0).count();
    let nonzero = linear.len().saturating_sub(histogram[0]);
    let fractional = nonzero.saturating_sub(histogram[255]);
    eprintln!(
        "[REMu][METAL][GLYPH-CONTENT] address={:#x} extent={}x{} pitch={} tile={} bytes={} zero={} full={} fractional={} values={occupied_values}",
        descriptor.address,
        descriptor.width,
        descriptor.height,
        descriptor.pitch,
        descriptor.tile_mode,
        linear.len(),
        histogram[0],
        histogram[255],
        fractional,
    );

    let expected = usize::try_from(descriptor.width).ok().and_then(|width| {
        usize::try_from(descriptor.height)
            .ok()
            .and_then(|height| width.checked_mul(height))
    });
    let Some(output_dir) = std::env::var_os("REMU_DUMP_MSL_DIR").map(PathBuf::from) else {
        return;
    };
    if expected != Some(linear.len()) {
        eprintln!(
            "[REMu][METAL][GLYPH-CONTENT] dump skipped: {} bytes do not match {}x{}",
            linear.len(),
            descriptor.width,
            descriptor.height,
        );
        return;
    }
    if expected == Some(raw.len()) {
        let mut raw_pgm =
            format!("P5\n{} {}\n255\n", descriptor.width, descriptor.height).into_bytes();
        raw_pgm.extend_from_slice(raw);
        let raw_path = output_dir.join(format!(
            "glyph-atlas-raw-as-linear-{:#x}-dst-{destination_select:03x}.pgm",
            descriptor.address
        ));
        if let Err(error) = std::fs::write(&raw_path, raw_pgm) {
            eprintln!(
                "[REMu][METAL][GLYPH-CONTENT] failed to write raw-as-linear {}: {error}",
                raw_path.display(),
            );
        } else {
            eprintln!(
                "[REMu][METAL][GLYPH-CONTENT] dumped raw-as-linear {}",
                raw_path.display(),
            );
        }
    }
    let mut pgm = format!("P5\n{} {}\n255\n", descriptor.width, descriptor.height).into_bytes();
    pgm.extend_from_slice(linear);
    let path = output_dir.join(format!(
        "glyph-atlas-{:#x}-dst-{destination_select:03x}.pgm",
        descriptor.address
    ));
    if let Err(error) = std::fs::write(&path, pgm) {
        eprintln!(
            "[REMu][METAL][GLYPH-CONTENT] failed to write {}: {error}",
            path.display(),
        );
    } else {
        eprintln!("[REMu][METAL][GLYPH-CONTENT] dumped {}", path.display(),);
    }
}

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
                return Ok(UploadedImageTexture {
                    texture: retained.texture.clone(),
                    descriptor,
                    writable: false,
                    selected_resident: None,
                });
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
                if replaced
                    && let Some(resident) = resident_guest_textures.remove(&descriptor.guest) {
                        trace_resident_texture_removal(
                            "sample-upload-mapping-replaced",
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
            let resident_descriptor = matches!(descriptor.texture_type, 9 | 10)
                .then(|| {
                    let exact = (!comparison)
                        .then(|| {
                            resident_guest_textures
                                .get(&descriptor.guest)
                                .filter(|resident| {
                                    resident.serves_reads()
                                        && resident.texture_type == descriptor.texture_type
                                        && resident.layer_count == descriptor.layer_count
                                        && resident.mip_level_count >= underlying_mip_count
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
                                        )))
                                        || resident_depth_sample_is_eligible(resident, descriptor))
                                    // Exact same-address colour views may remain gpu-authored after
                                    // a neighbour partially supersedes owned-byte accounting; those
                                    // still bind through resident_color_view_sample_is_eligible.
                                    // Every other compatible path still needs full owned coverage.
                                    && (resident.serves_reads()
                                        || resident_color_view_sample_is_eligible(
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
            // A layered depth-sample SRD assembled from resident depth faces is served directly:
            // the assembly ran before this upload pass, inside the guest command batch, so the
            // array already orders after the face render passes.
            let depth_array_resolve = (!fallback && !storage)
                .then(|| {
                    depth_array_textures
                        .get(&DepthArrayKey {
                            descriptor: descriptor.guest,
                            layer_count: descriptor.layer_count,
                        })
                        .map(|entry| entry.texture.clone())
                })
                .flatten();
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
            } else if let Some(texture) = depth_array_resolve {
                texture
            } else if let Some(texture) = render_resolve {
                // Preserve the actual GPU-side source for opt-in scanout provenance. Read-only
                // bindings do not use this field for write attribution, so recording it cannot
                // change residency semantics.
                selected_resident = render_resolve_descriptor;
                texture
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
            Ok(UploadedImageTexture {
                texture,
                descriptor,
                // The resident 2D path has exact guest writeback today. A writable 3D image stays
                // resident in the shared sampled-image cache so later GPU reads see its contents;
                // it must not invalidate that cache through the 2D writeback path.
                writable: !fallback && writable && descriptor.texture_type == 9,
                selected_resident,
            })
        })
        .collect()
}
