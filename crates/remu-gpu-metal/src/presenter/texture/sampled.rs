// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    DecodedImageDescriptor, GuestTextureDescriptor, MetalPresenter, MetalTextureSwizzleChannels,
    RESIDENT_DEPTH_SAMPLE_HITS, ResidentGuestTexture, ResidentTextureMap, ResourceSpan,
    SAMPLED_ALIAS_WRITEBACK_BYTES,
    SAMPLED_ALIAS_WRITEBACK_COUNT, SAMPLED_ALIAS_WRITEBACK_NANOSECONDS, SampledTextureKey,
    WriteWatchRefresh, comparison_depth32_upload, compressed_texture_block_extent,
    decode_guest_texture_linear_from_raw_at_z, guest_frame_fingerprint, guest_texture_byte_count,
    guest_texture_descriptors_overlap, guest_texture_descriptors_share_color_view_storage,
    guest_texture_descriptors_share_depth_sample_storage,
    guest_texture_descriptors_share_gpu_sample_storage, guest_texture_descriptors_share_storage,
    image_resource_descriptor_is_null, is_storage_image_opcode,
    refresh_resident_texture_write_watch, render_resolve_candidate_matches,
    resident_is_depth_array_face,
    sampled_alias_descriptor, texture_element_layout, trace_image_descriptors,
    trace_resident_texture_removal, trace_target_sampling, trace_video_target_consumers,
};
use super::super::{FlushPolicy, GuestBatchFlushReason, WritebackTrigger};
use metal::foreign_types::ForeignType;
use metal::{
    Device, MTLPixelFormat, MTLRegion, MTLStorageMode, MTLTextureType, MTLTextureUsage, NSRange,
    Texture, TextureDescriptor,
};
use objc::runtime::Object;
use objc::{msg_send, sel, sel_impl};
use remu_shader::{ShaderEvaluation, image_is_comparison};
use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

impl MetalPresenter {
    pub(crate) fn synchronize_sampled_aliases(
        &mut self,
        evaluations: &[&ShaderEvaluation],
    ) -> Result<(), String> {
        let mut aliases = Vec::new();
        for &evaluation in evaluations {
            for binding in evaluation
                .image_bindings
                .iter()
                .filter(|binding| !is_storage_image_opcode(binding.opcode))
                .filter(|binding| !image_resource_descriptor_is_null(&binding.resource_descriptor))
            {
                let Some(sampled) = sampled_alias_descriptor(binding) else {
                    // The upload path binds a dimension-compatible zero texture for malformed,
                    // sentinel, or otherwise unreadable sampled descriptors. Alias
                    // synchronization has no guest range to publish in that case and must not
                    // reject the entire draw before the fallback can be installed.
                    continue;
                };
                let comparison = image_is_comparison(binding.opcode);
                let sampled_bytes = guest_texture_byte_count(sampled.guest).ok_or_else(|| {
                    format!(
                        "image at PC {:#x} has an invalid guest byte extent",
                        binding.pc
                    )
                })?;
                // A Gen5 MSAA allocation can expose its shader-readable resolved image through
                // an overlapping SRD whose base follows the allocation's mask metadata. Keep the
                // Metal resolve GPU-current so `upload_image_textures` can bind it directly. A
                // CPU writeback here both discards that provenance and decodes from the shifted
                // SRD base, producing an empty post-process input.
                //
                // Each resident write watch already records every coherent CPU alias of the
                // underlying physical range. Consult those immutable ranges directly instead of
                // resolving both descriptors through the global mapping table for every
                // resident/sample pair. Silent Hill has roughly one hundred residents and
                // performs this query for nearly every draw; the old nested mapping-lock lookup
                // consumed about one quarter of presenter CPU time in a sampled title workload.
                aliases.extend(
                    self.resident_guest_textures
                        .values()
                        .filter(|resident| resident.gpu_written())
                        .filter(|resident| {
                            let overlaps_resident = resident
                                .write_watch
                                .overlaps_alias_range(sampled.guest.address, sampled_bytes);
                            let aliases_sample = overlaps_resident
                                && resident.descriptor != sampled.guest
                                && !guest_texture_descriptors_share_storage(
                                    resident.descriptor,
                                    sampled.guest,
                                )
                                && !guest_texture_descriptors_share_gpu_sample_storage(
                                    resident.descriptor,
                                    sampled.guest,
                                )
                                && !resident_color_view_sample_is_eligible(
                                    resident, sampled, comparison,
                                )
                                && !resident_depth_sample_is_eligible(resident, sampled)
                                && !(resident_is_depth_array_face(resident, sampled)
                                    && self.depth_array_is_fully_resident(sampled))
                                && !self.multisample_textures.iter().any(|(key, multisample)| {
                                    multisample.initialized
                                        && key.descriptor == resident.descriptor
                                        && render_resolve_candidate_matches(
                                            sampled.guest,
                                            key.descriptor,
                                            key.sample_count,
                                        )
                                });
                            if aliases_sample && trace_target_sampling() {
                                static LOGGED_ALIAS_FLUSHES: OnceLock<
                                    Mutex<HashSet<(u64, u64, u32, u32)>>,
                                > = OnceLock::new();
                                let key = (
                                    resident.descriptor.address,
                                    sampled.guest.address,
                                    sampled.base_mip_level,
                                    sampled.mip_level_count,
                                );
                                let mut logged = LOGGED_ALIAS_FLUSHES
                                    .get_or_init(|| Mutex::new(HashSet::new()))
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                                if logged.len() < 256 && logged.insert(key) {
                                    eprintln!(
                                        "[REMu][METAL][ALIAS-FLUSH] resident={:#x}/{}x{}/p{}/{:?}/tile{}/mips{}/type{}/layers{} sampled={:#x}/{}x{}/p{}/{:?}/tile{}/base{}/mips{}/type{}/layers{} select={:#x} comparison={comparison} pc={:#x} declared_overlap={}",
                                        resident.descriptor.address,
                                        resident.descriptor.width,
                                        resident.descriptor.height,
                                        resident.descriptor.pitch,
                                        resident.descriptor.pixel_format,
                                        resident.descriptor.tile_mode,
                                        resident.mip_level_count,
                                        resident.texture_type,
                                        resident.layer_count,
                                        sampled.guest.address,
                                        sampled.guest.width,
                                        sampled.guest.height,
                                        sampled.guest.pitch,
                                        sampled.guest.pixel_format,
                                        sampled.guest.tile_mode,
                                        sampled.base_mip_level,
                                        sampled.mip_level_count,
                                        sampled.texture_type,
                                        sampled.layer_count,
                                        sampled.destination_select,
                                        binding.pc,
                                        guest_texture_descriptors_overlap(
                                            resident.descriptor,
                                            sampled.guest,
                                        ),
                                    );
                                }
                            }
                            aliases_sample
                        })
                        .map(|resident| resident.descriptor),
                );
            }
        }
        aliases.sort_unstable_by_key(|descriptor| descriptor.address);
        aliases.dedup();
        aliases.sort_by_key(|descriptor| {
            (
                self.resident_guest_textures
                    .get(descriptor)
                    .map_or(0, |resident| resident.content_version),
                descriptor.address,
            )
        });
        if aliases.is_empty() {
            return Ok(());
        }
        let alias_started = Instant::now();
        let alias_bytes = aliases.iter().fold(0_u64, |total, &descriptor| {
            total.saturating_add(guest_texture_byte_count(descriptor).unwrap_or_default())
        });
        let alias_count = aliases.len();
        // End the open render pass and commit the current batch, but do not drain every in-flight
        // guest batch on the CPU. The following texture readbacks are encoded on the same Metal
        // queue and `PendingTextureReadback::finish` waits only for that blit, which implicitly
        // orders after the producer work already queued ahead of it. That ordering holds only if
        // the producer really was committed, so the flush may not defer. A synchronization that
        // publishes nothing has no readback to order and skips the drain.
        if WritebackTrigger::SampledAlias.publishes_to_guest() {
            self.flush_guest_command_batch(
                FlushPolicy::Required,
                GuestBatchFlushReason::CpuSynchronization,
            )?;
        }
        for descriptor in aliases {
            let refresh = refresh_resident_texture_write_watch(
                self.resident_guest_textures
                    .get_mut(&descriptor)
                    .expect("collected sampled alias exists"),
            )?;
            if refresh == WriteWatchRefresh::Replaced {
                if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
                    trace_resident_texture_removal("sampled-alias-mapping-replaced", &resident);
                }
                continue;
            }
            if refresh == WriteWatchRefresh::Reprotecting {
                return Err(format!(
                    "sampled alias mapping {:#x} is being reprotected",
                    descriptor.address
                ));
            }
            if self
                .resident_guest_textures
                .get_mut(&descriptor)
                .expect("refreshed sampled alias exists")
                .cpu_write_obsoletes_gpu_writeback(refresh)?
            {
                if let Some(resident) = self.resident_guest_textures.remove(&descriptor) {
                    // The CPU has repurposed this physical heap range since the render that
                    // produced `resident`. Keep the newer bytes authoritative so the following
                    // upload observes the guest's command/resource contents.
                    trace_resident_texture_removal("cpu-reused-sampled-alias", &resident);
                }
                continue;
            }
            // Reinterpreting a rendered range through a sampled SRD is a read REmu itself serves.
            // It can bind the resident's own texels, so the guest never needs these bytes staged
            // in RAM, and staging them is what can land pixels in a recycled heap block.
            self.settle_resident_texture_writeback(descriptor, WritebackTrigger::SampledAlias)?;
        }
        SAMPLED_ALIAS_WRITEBACK_COUNT.fetch_add(
            u64::try_from(alias_count).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        SAMPLED_ALIAS_WRITEBACK_BYTES.fetch_add(alias_bytes, Ordering::Relaxed);
        SAMPLED_ALIAS_WRITEBACK_NANOSECONDS.fetch_add(
            u64::try_from(alias_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        Ok(())
    }
}

pub(crate) fn decoded_image_for_sampled_key(key: SampledTextureKey) -> DecodedImageDescriptor {
    DecodedImageDescriptor {
        guest: key.descriptor,
        span: ResourceSpan::for_texture(key.descriptor),
        destination_select: 0,
        base_mip_level: key.base_mip_level,
        mip_level_count: key.mip_level_count,
        texture_type: key.texture_type,
        layer_count: key.layer_count,
    }
}

/// Returns the GPU-authored resident that can still serve a cached sampled texture after ACQUIRE.
pub(crate) fn gpu_written_resident_serving_sampled_key(
    residents: &ResidentTextureMap,
    key: SampledTextureKey,
) -> Option<GuestTextureDescriptor> {
    let sampled = decoded_image_for_sampled_key(key);
    residents
        .iter()
        .filter(|(_, resident)| resident.gpu_written() && resident.serves_reads())
        .filter(|(descriptor, resident)| {
            let descriptor = **descriptor;
            guest_texture_descriptors_overlap(descriptor, key.descriptor)
                || guest_texture_descriptors_share_gpu_sample_storage(descriptor, key.descriptor)
                || resident_depth_sample_is_eligible(resident, sampled)
                || resident_color_view_sample_is_eligible(resident, sampled, key.comparison)
        })
        .map(|(descriptor, resident)| (*descriptor, resident.content_version))
        .max_by_key(|(descriptor, version)| (*version, descriptor.address))
        .map(|(descriptor, _)| descriptor)
}

pub(crate) fn sampled_texture_key(
    descriptor: DecodedImageDescriptor,
    comparison: bool,
) -> SampledTextureKey {
    // GFX10 destination selection is applied by generated MSL at sample/load time. It neither
    // changes the decoded backing nor the Metal texture allocation, so including it here creates
    // duplicate 4–64 MiB uploads for views that differ only by channel selection.
    SampledTextureKey {
        descriptor: descriptor.guest,
        base_mip_level: descriptor.base_mip_level,
        mip_level_count: descriptor.mip_level_count,
        texture_type: descriptor.texture_type,
        layer_count: descriptor.layer_count,
        comparison,
    }
}

#[cfg(test)]
pub(crate) fn new_red_blue_swizzled_texture_view(
    texture: &Texture,
    base_mip_level: u32,
    mip_level_count: u32,
) -> Option<Texture> {
    new_texture_view_with_swizzle(
        texture,
        texture.pixel_format(),
        base_mip_level,
        mip_level_count,
        true,
    )
}

pub(crate) fn new_color_alias_texture_view(
    texture: &Texture,
    pixel_format: MTLPixelFormat,
    base_mip_level: u32,
    mip_level_count: u32,
) -> Option<Texture> {
    // Selecting the sampled pixel format reinterprets both the component byte order and the
    // linear/sRGB transfer function. An additional red/blue swizzle would double-swap BGRA/RGBA
    // aliases, so this view intentionally uses the identity component mapping.
    new_texture_view_with_swizzle(
        texture,
        pixel_format,
        base_mip_level,
        mip_level_count,
        false,
    )
}

fn new_texture_view_with_swizzle(
    texture: &Texture,
    pixel_format: MTLPixelFormat,
    base_mip_level: u32,
    mip_level_count: u32,
    swap_red_blue: bool,
) -> Option<Texture> {
    let underlying_mip_count = base_mip_level.checked_add(mip_level_count)?;
    if mip_level_count == 0 || underlying_mip_count > texture.mipmap_level_count() as u32 {
        return None;
    }
    const RED: u8 = 2;
    const GREEN: u8 = 3;
    const BLUE: u8 = 4;
    const ALPHA: u8 = 5;
    let swizzle = MetalTextureSwizzleChannels {
        red: if swap_red_blue { BLUE } else { RED },
        green: GREEN,
        blue: if swap_red_blue { RED } else { BLUE },
        alpha: ALPHA,
    };
    // SAFETY: `texture` is a live MTLTexture. The selector is available on every supported macOS
    // version (10.15+), the enum arguments use Metal's NSUInteger ABI, NSRange is repr(C), and the
    // swizzle structure above exactly matches MTLTextureSwizzleChannels. A `new...` selector
    // returns an owned (+1) object, which `Texture::from_ptr` assumes.
    let view: *mut Object = unsafe {
        msg_send![
            texture.as_ref(),
            newTextureViewWithPixelFormat: pixel_format
            textureType: texture.texture_type()
            levels: NSRange::new(u64::from(base_mip_level), u64::from(mip_level_count))
            slices: NSRange::new(0, texture.array_length())
            swizzle: swizzle
        ]
    };
    if view.is_null() {
        None
    } else {
        // SAFETY: The non-null Objective-C result conforms to MTLTexture by method contract and
        // carries the retained ownership documented above.
        Some(unsafe { Texture::from_ptr(view.cast()) })
    }
}

pub(crate) fn resident_texture_for_sample(
    resident: &mut ResidentGuestTexture,
    sampled: DecodedImageDescriptor,
) -> Option<Texture> {
    let underlying_mip_count = sampled
        .base_mip_level
        .checked_add(sampled.mip_level_count.max(1))?;
    if resident.mip_level_count < underlying_mip_count {
        return None;
    }
    if guest_texture_descriptors_share_depth_sample_storage(
        resident.descriptor,
        sampled.guest,
        sampled.destination_select,
    ) {
        RESIDENT_DEPTH_SAMPLE_HITS.fetch_add(1, Ordering::Relaxed);
        if sampled.base_mip_level == 0 && sampled.mip_level_count.max(1) == resident.mip_level_count
        {
            return Some(resident.texture.clone());
        }
        return Some(resident.texture.new_texture_view_from_slice(
            resident.texture.pixel_format(),
            MTLTextureType::D2,
            NSRange::new(
                u64::from(sampled.base_mip_level),
                u64::from(sampled.mip_level_count.max(1)),
            ),
            NSRange::new(0, 1),
        ));
    }
    if guest_texture_descriptors_share_storage(resident.descriptor, sampled.guest)
        || guest_texture_descriptors_share_gpu_sample_storage(resident.descriptor, sampled.guest)
    {
        if sampled.base_mip_level == 0 && sampled.mip_level_count.max(1) == resident.mip_level_count
        {
            return Some(resident.texture.clone());
        }
        return Some(resident.texture.new_texture_view_from_slice(
            resident.texture.pixel_format(),
            MTLTextureType::D2,
            NSRange::new(
                u64::from(sampled.base_mip_level),
                u64::from(sampled.mip_level_count.max(1)),
            ),
            NSRange::new(0, 1),
        ));
    }
    if !guest_texture_descriptors_share_color_view_storage(resident.descriptor, sampled.guest) {
        return None;
    }
    let sampled_mip_level_count = sampled.mip_level_count.max(1);
    let uses_complete_hierarchy =
        sampled.base_mip_level == 0 && sampled_mip_level_count == resident.mip_level_count;
    let cached_view_matches = resident
        .component_reorder_view
        .as_ref()
        .is_some_and(|view| view.pixel_format() == sampled.guest.pixel_format);
    if uses_complete_hierarchy && !cached_view_matches {
        let view = new_color_alias_texture_view(
            &resident.texture,
            sampled.guest.pixel_format,
            sampled.base_mip_level,
            sampled_mip_level_count,
        )?;
        view.set_label(&format!(
            "REMu color-aliased guest image {:#x}",
            resident.descriptor.address
        ));
        resident.component_reorder_view = Some(view);
        if trace_target_sampling() {
            static CREATED_COMPONENT_REORDER_VIEWS: AtomicUsize = AtomicUsize::new(0);
            let sequence = CREATED_COMPONENT_REORDER_VIEWS.fetch_add(1, Ordering::Relaxed) + 1;
            if sequence <= 16 || sequence.is_power_of_two() {
                eprintln!(
                    "[REMu][METAL][ALIAS-VIEW] sequence={sequence} address={:#x} rendered={:?} sampled={:?} extent={}x{} tile={}",
                    resident.descriptor.address,
                    resident.descriptor.pixel_format,
                    sampled.guest.pixel_format,
                    sampled.guest.width,
                    sampled.guest.height,
                    sampled.guest.tile_mode,
                );
            }
        }
    }
    if uses_complete_hierarchy {
        resident.component_reorder_view.clone()
    } else {
        let view = new_color_alias_texture_view(
            &resident.texture,
            sampled.guest.pixel_format,
            sampled.base_mip_level,
            sampled_mip_level_count,
        )?;
        view.set_label(&format!(
            "REMu color-aliased guest image {:#x} mips {}..{}",
            resident.descriptor.address,
            sampled.base_mip_level,
            sampled.base_mip_level + sampled_mip_level_count,
        ));
        Some(view)
    }
}

pub(crate) fn resident_color_view_sample_is_eligible(
    resident: &ResidentGuestTexture,
    sampled: DecodedImageDescriptor,
    comparison: bool,
) -> bool {
    let underlying_mip_count = sampled
        .base_mip_level
        .checked_add(sampled.mip_level_count.max(1));
    !comparison
        && resident.gpu_written()
        && resident.serves_reads()
        && resident.texture_type == 9
        && resident.layer_count == 1
        && sampled.texture_type == 9
        && sampled.layer_count == 1
        && underlying_mip_count.is_some_and(|count| resident.mip_level_count >= count)
        && guest_texture_descriptors_share_color_view_storage(resident.descriptor, sampled.guest)
}

pub(crate) fn resident_depth_sample_is_eligible(
    resident: &ResidentGuestTexture,
    sampled: DecodedImageDescriptor,
) -> bool {
    let underlying_mip_count = sampled
        .base_mip_level
        .checked_add(sampled.mip_level_count.max(1));
    resident.gpu_written()
        && resident.serves_reads()
        && resident.texture_type == 9
        && resident.layer_count == 1
        && sampled.texture_type == 9
        && sampled.layer_count == 1
        && underlying_mip_count.is_some_and(|count| resident.mip_level_count >= count)
        && guest_texture_descriptors_share_depth_sample_storage(
            resident.descriptor,
            sampled.guest,
            sampled.destination_select,
        )
}

pub(crate) fn sampled_texture_cache_needs_reload(
    refresh: WriteWatchRefresh,
    content_changed: bool,
) -> bool {
    match refresh {
        WriteWatchRefresh::Active | WriteWatchRefresh::Renewed { .. } => content_changed,
        WriteWatchRefresh::Replaced => true,
        WriteWatchRefresh::Reprotecting => false,
    }
}

pub(crate) fn decode_sampled_texture_linear_from_raw(
    descriptor: DecodedImageDescriptor,
    raw: &[u8],
) -> Result<Vec<u8>, String> {
    let face_bytes = usize::try_from(
        guest_texture_byte_count(descriptor.guest)
            .ok_or("sampled texture face byte count overflows")?,
    )
    .map_err(|_| "sampled texture face byte count does not fit usize")?;
    let expected = face_bytes
        .checked_mul(
            usize::try_from(descriptor.layer_count)
                .map_err(|_| "sampled layer count does not fit usize")?,
        )
        .ok_or("sampled texture byte count overflows")?;
    if raw.len() != expected {
        return Err(format!(
            "sampled texture source has {} bytes, expected {expected}",
            raw.len()
        ));
    }
    let mut linear = Vec::new();
    for (surface_z, face) in raw.chunks_exact(face_bytes).enumerate() {
        linear.extend(decode_guest_texture_linear_from_raw_at_z(
            descriptor.guest,
            face,
            surface_z,
        )?);
    }
    Ok(linear)
}

pub(crate) fn create_sampled_guest_texture(
    device: &Device,
    queue: &metal::CommandQueue,
    guest: GuestTextureDescriptor,
    linear: &[u8],
    base_mip_level: u32,
    requested_mip_level_count: u32,
    texture_type: u32,
    layer_count: u32,
    comparison: bool,
) -> Result<Texture, String> {
    let (elements_wide, elements_high, _, bytes_per_element) = texture_element_layout(guest)?;
    let source_face_bytes = elements_wide
        .checked_mul(elements_high)
        .and_then(|elements| elements.checked_mul(bytes_per_element))
        .ok_or("Metal sampled texture dimensions overflow")?;
    let required = source_face_bytes
        .checked_mul(
            usize::try_from(layer_count).map_err(|_| "sampled layer count does not fit usize")?,
        )
        .ok_or("Metal sampled texture layer dimensions overflow")?;
    if guest.width == 0 || guest.height == 0 || linear.len() != required {
        return Err(format!(
            "invalid sampled guest texture upload {}x{} format {:?} with {} bytes (expected {required})",
            guest.width,
            guest.height,
            guest.pixel_format,
            linear.len()
        ));
    }
    if (trace_image_descriptors() && guest.width >= 700 && guest.height >= 500
        || trace_video_target_consumers())
        && guest.tile_mode == 0
        && matches!(
            guest.pixel_format,
            MTLPixelFormat::R8Unorm | MTLPixelFormat::RG8Unorm
        )
    {
        static LOGGED_VIDEO_UPLOADS: OnceLock<Mutex<HashSet<(u64, u64)>>> = OnceLock::new();
        let fingerprint = guest_frame_fingerprint(linear);
        let mut logged = LOGGED_VIDEO_UPLOADS
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if logged.len() < 32 && logged.insert((guest.address, fingerprint)) {
            let (minimum, maximum, total) = linear.iter().fold(
                (u8::MAX, u8::MIN, 0_u64),
                |(minimum, maximum, total), &value| {
                    (
                        minimum.min(value),
                        maximum.max(value),
                        total + u64::from(value),
                    )
                },
            );
            let average = total as f64 / linear.len().max(1) as f64;
            eprintln!(
                "[REMu][METAL][VIDEO-PLANE] uploaded address={:#x} extent={}x{} pitch={} format={:?} bytes={} hash={fingerprint:016x} range={minimum}..={maximum} average={average:.1}",
                guest.address,
                guest.width,
                guest.height,
                guest.pitch,
                guest.pixel_format,
                linear.len(),
            );
        }
    }
    let comparison_linear = comparison
        .then(|| comparison_depth32_upload(guest.pixel_format, linear))
        .transpose()?
        .flatten();
    let upload = comparison_linear.as_deref().unwrap_or(linear);
    let upload_bytes_per_element = if comparison_linear.is_some() {
        std::mem::size_of::<f32>()
    } else {
        bytes_per_element
    };
    let face_bytes = elements_wide
        .checked_mul(elements_high)
        .and_then(|elements| elements.checked_mul(upload_bytes_per_element))
        .ok_or("Metal sampled texture upload dimensions overflow")?;
    // GFX10 packs small mip levels into a tiled mip tail. Until that guest layout is decoded,
    // generate the same filtered hierarchy from the exact base image on Metal. Compressed BC
    // formats cannot be filtered by Metal's blit mip generator, so they remain base-level views.
    let underlying_mip_level_count = base_mip_level
        .checked_add(requested_mip_level_count.max(1))
        .ok_or("sampled Metal mip range overflows")?;
    let mip_level_count = if compressed_texture_block_extent(guest.pixel_format).is_some() {
        if base_mip_level != 0 {
            return Err(format!(
                "compressed sampled Metal texture cannot synthesize base mip {base_mip_level}"
            ));
        }
        1
    } else {
        underlying_mip_level_count
    };
    let descriptor = TextureDescriptor::new();
    // Cube resources (type 11) are created as 2D arrays: translated shaders address them with
    // the GCN face-atlas convention (face layer plus in-face coordinates), which indexes array
    // layers directly. This also matches the layered depth-sample assembly, which serves cube
    // shadow SRDs as D2Array textures.
    descriptor.set_texture_type(match texture_type {
        9 => MTLTextureType::D2,
        10 => MTLTextureType::D3,
        11 | 13 => MTLTextureType::D2Array,
        _ => {
            return Err(format!(
                "unsupported sampled Metal texture type {texture_type}"
            ));
        }
    });
    let pixel_format = if comparison {
        match guest.pixel_format {
            MTLPixelFormat::R16Unorm => MTLPixelFormat::Depth16Unorm,
            MTLPixelFormat::R32Float => MTLPixelFormat::Depth32Float,
            _ => MTLPixelFormat::Depth32Float,
        }
    } else {
        guest.pixel_format
    };
    descriptor.set_pixel_format(pixel_format);
    descriptor.set_width(u64::from(guest.width));
    descriptor.set_height(u64::from(guest.height));
    if texture_type == 10 {
        descriptor.set_depth(u64::from(layer_count));
    }
    descriptor.set_mipmap_level_count(u64::from(mip_level_count));
    if matches!(texture_type, 11 | 13) {
        descriptor.set_array_length(u64::from(layer_count));
    }
    descriptor.set_storage_mode(MTLStorageMode::Shared);
    descriptor.set_usage(if texture_type == 10 && !comparison {
        MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite
    } else {
        MTLTextureUsage::ShaderRead
    });
    let texture = device.new_texture(&descriptor);
    let bytes_per_row = u64::try_from(elements_wide)
        .expect("validated texture element width fits u64")
        * u64::try_from(upload_bytes_per_element)
            .expect("validated texture element byte count fits u64");
    if texture_type == 10 {
        texture.replace_region_in_slice(
            MTLRegion::new_3d(
                0,
                0,
                0,
                u64::from(guest.width),
                u64::from(guest.height),
                u64::from(layer_count),
            ),
            0,
            0,
            upload.as_ptr().cast::<c_void>(),
            bytes_per_row,
            u64::try_from(face_bytes).expect("validated image byte count fits u64"),
        );
    } else if matches!(texture_type, 11 | 13) {
        for layer in 0..layer_count {
            let offset = usize::try_from(layer)
                .expect("validated layer fits usize")
                .checked_mul(face_bytes)
                .expect("validated layer offset fits usize");
            texture.replace_region_in_slice(
                MTLRegion::new_2d(0, 0, u64::from(guest.width), u64::from(guest.height)),
                0,
                u64::from(layer),
                upload[offset..].as_ptr().cast::<c_void>(),
                bytes_per_row,
                u64::try_from(face_bytes).expect("validated face byte count fits u64"),
            );
        }
    } else {
        texture.replace_region(
            MTLRegion::new_2d(0, 0, u64::from(guest.width), u64::from(guest.height)),
            0,
            upload.as_ptr().cast::<c_void>(),
            bytes_per_row,
        );
    }
    if mip_level_count > 1 {
        let command_buffer = queue.new_command_buffer().to_owned();
        command_buffer.set_label("REMu sampled image mip generation");
        let encoder = command_buffer.new_blit_command_encoder();
        encoder.set_label("REMu sampled image generated mip hierarchy");
        encoder.generate_mipmaps(&texture);
        encoder.end_encoding();
        command_buffer.commit();
    }
    if base_mip_level == 0 {
        return Ok(texture);
    }
    if texture_type != 9 {
        return Err(format!(
            "nonzero sampled base mip {base_mip_level} is not implemented for texture type {texture_type}"
        ));
    }
    Ok(texture.new_texture_view_from_slice(
        pixel_format,
        MTLTextureType::D2,
        NSRange::new(
            u64::from(base_mip_level),
            u64::from(requested_mip_level_count.max(1)),
        ),
        NSRange::new(0, 1),
    ))
}

#[cfg(test)]
pub(crate) fn apply_image_destination_select(
    rgba: &mut [u8],
    destination_select: u32,
) -> Result<(), String> {
    if destination_select == 0 || destination_select == 0xfac {
        return Ok(());
    }
    let selectors =
        std::array::from_fn::<_, 4, _>(|channel| (destination_select >> (channel * 3)) & 0x7);
    if selectors
        .iter()
        .any(|selector| !matches!(selector, 0 | 1 | 4..=7))
    {
        return Err(format!(
            "invalid image destination select {destination_select:#x}"
        ));
    }
    for pixel in rgba.chunks_exact_mut(4) {
        let source: [u8; 4] = pixel.try_into().expect("RGBA pixel has four channels");
        for (channel, selector) in selectors.iter().copied().enumerate() {
            pixel[channel] = match selector {
                0 => 0,
                1 => 255,
                4..=7 => source[usize::try_from(selector - 4).expect("channel fits usize")],
                _ => unreachable!("validated destination selector"),
            };
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn prepare_sampled_texture_upload(
    descriptor: GuestTextureDescriptor,
    mut linear: Vec<u8>,
    destination_select: u32,
) -> Result<(GuestTextureDescriptor, Vec<u8>), String> {
    if image_destination_select_is_native(descriptor.pixel_format, destination_select) {
        return Ok((descriptor, linear));
    }
    if descriptor.bytes_per_pixel == 4
        && matches!(
            descriptor.pixel_format,
            MTLPixelFormat::RGBA8Unorm
                | MTLPixelFormat::RGBA8Unorm_sRGB
                | MTLPixelFormat::RGBA8Snorm
                | MTLPixelFormat::RGBA8Uint
                | MTLPixelFormat::RGBA8Sint
                | MTLPixelFormat::BGRA8Unorm
                | MTLPixelFormat::BGRA8Unorm_sRGB
        )
    {
        apply_image_destination_select(&mut linear, destination_select)?;
        return Ok((descriptor, linear));
    }
    if matches!(
        descriptor.pixel_format,
        MTLPixelFormat::R8Unorm | MTLPixelFormat::R8Unorm_sRGB
    ) {
        let selectors =
            std::array::from_fn::<_, 4, _>(|channel| (destination_select >> (channel * 3)) & 0x7);
        if selectors
            .iter()
            .any(|selector| !matches!(selector, 0 | 1 | 4..=7))
        {
            return Err(format!(
                "invalid image destination select {destination_select:#x}"
            ));
        }
        let mut rgba = Vec::with_capacity(linear.len().saturating_mul(4));
        for red in linear {
            for &selector in &selectors {
                rgba.push(match selector {
                    0 | 5 | 6 => 0,
                    1 | 7 => u8::MAX,
                    4 => red,
                    _ => unreachable!("validated destination selector"),
                });
            }
        }
        return Ok((
            GuestTextureDescriptor {
                pixel_format: if descriptor.pixel_format == MTLPixelFormat::R8Unorm_sRGB {
                    MTLPixelFormat::RGBA8Unorm_sRGB
                } else {
                    MTLPixelFormat::RGBA8Unorm
                },
                bytes_per_pixel: 4,
                ..descriptor
            },
            rgba,
        ));
    }
    Err(format!(
        "destination select {destination_select:#x} is not representable for {:?}",
        descriptor.pixel_format
    ))
}

#[cfg(test)]
pub(crate) fn image_destination_select_is_native(
    pixel_format: MTLPixelFormat,
    destination_select: u32,
) -> bool {
    if destination_select == 0 {
        return true;
    }
    let native_select = match pixel_format {
        MTLPixelFormat::R8Unorm
        | MTLPixelFormat::R8Snorm
        | MTLPixelFormat::R8Uint
        | MTLPixelFormat::R8Sint
        | MTLPixelFormat::R8Unorm_sRGB
        | MTLPixelFormat::R16Unorm
        | MTLPixelFormat::R16Snorm
        | MTLPixelFormat::R16Uint
        | MTLPixelFormat::R16Sint
        | MTLPixelFormat::R16Float
        | MTLPixelFormat::R32Uint
        | MTLPixelFormat::R32Sint
        | MTLPixelFormat::R32Float => 0x204,
        MTLPixelFormat::RG8Unorm
        | MTLPixelFormat::RG8Snorm
        | MTLPixelFormat::RG8Uint
        | MTLPixelFormat::RG8Sint
        | MTLPixelFormat::RG8Unorm_sRGB
        | MTLPixelFormat::RG16Unorm
        | MTLPixelFormat::RG16Snorm
        | MTLPixelFormat::RG16Uint
        | MTLPixelFormat::RG16Sint
        | MTLPixelFormat::RG16Float
        | MTLPixelFormat::RG32Uint
        | MTLPixelFormat::RG32Sint
        | MTLPixelFormat::RG32Float => 0x22c,
        MTLPixelFormat::RG11B10Float | MTLPixelFormat::RGB9E5Float => 0x3ac,
        _ => 0xfac,
    };
    destination_select == native_select
}
