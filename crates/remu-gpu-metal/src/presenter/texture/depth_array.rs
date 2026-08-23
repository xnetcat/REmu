// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

//! GPU-side assembly of layered depth-sample views from per-face resident depth targets.
//!
//! Gen5 titles render shadow cubemaps and cascade atlases as individual 2D depth targets and
//! then bind one layered SRD whose layers tile the same guest allocation face after face.
//! Serving that view from guest memory forces a GPU-to-CPU writeback of every face and a
//! CPU re-upload of the whole array each frame. Instead, the faces are blit-copied into one
//! cached depth-format array texture inside the guest command batch, so ordering against the
//! face render passes is preserved and comparison samplers consume a real depth format.

use metal::{
    Device, MTLOrigin, MTLPixelFormat, MTLSize, MTLStorageMode, MTLTextureType, MTLTextureUsage,
    TextureDescriptor,
};
use remu_shader::ShaderEvaluation;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::super::SharedEventWriteKind;
use super::{
    DecodedImageDescriptor, DepthArrayKey, DepthArrayResidentTexture, GuestTextureDescriptor,
    MetalPresenter, ResidentGuestTexture, decode_image_descriptor, guest_texture_byte_count,
    image_resource_descriptor_is_null, is_storage_image_opcode, trace_target_sampling,
};

/// The layered texture types a depth-face array can serve: 2D arrays and cubes.
const fn is_layered_sample_type(texture_type: u32) -> bool {
    matches!(texture_type, 11 | 13)
}

const fn depth_render_format_for_sample(sampled: MTLPixelFormat) -> Option<MTLPixelFormat> {
    match sampled {
        MTLPixelFormat::R16Unorm => Some(MTLPixelFormat::Depth16Unorm),
        MTLPixelFormat::R32Float => Some(MTLPixelFormat::Depth32Float),
        _ => None,
    }
}

/// Resolves one layer of a layered depth-sample SRD to the per-face depth-target descriptor
/// that must be resident for the array to be served on the GPU.
///
/// The channel-selection gate matches the flat depth-sample path: the native Z plane exposed
/// through R, optionally replicated or padded with zero/one. Layered SRDs with mip pyramids
/// stay on the guest-memory path until a title exercises them.
pub(crate) fn depth_array_face_descriptor(
    sampled: DecodedImageDescriptor,
    layer: u32,
) -> Option<GuestTextureDescriptor> {
    if !is_layered_sample_type(sampled.texture_type)
        || sampled.layer_count < 2
        || layer >= sampled.layer_count
        || sampled.base_mip_level != 0
        || sampled.mip_level_count.max(1) != 1
    {
        return None;
    }
    if !matches!(sampled.destination_select, 0 | 0x004 | 0x204 | 0x924) {
        return None;
    }
    let pixel_format = depth_render_format_for_sample(sampled.guest.pixel_format)?;
    let face_bytes = guest_texture_byte_count(sampled.guest)?;
    let address = sampled.guest.address.checked_add(
        face_bytes.checked_mul(u64::from(sampled.base_array_slice.checked_add(layer)?))?,
    )?;
    Some(GuestTextureDescriptor {
        address,
        pixel_format,
        ..sampled.guest
    })
}

/// Returns whether a GPU-written resident is one face of a layered depth-sample SRD, in which
/// case it must not be written back to guest memory for that sample: the array assembly serves
/// it directly.
pub(crate) fn resident_is_depth_array_face(
    resident: &ResidentGuestTexture,
    sampled: DecodedImageDescriptor,
) -> bool {
    if resident.texture_type != 9 || resident.layer_count != 1 {
        return false;
    }
    let Some(face_bytes) = guest_texture_byte_count(sampled.guest) else {
        return false;
    };
    let Some(selected_offset) = face_bytes.checked_mul(u64::from(sampled.base_array_slice)) else {
        return false;
    };
    let Some(selected_address) = sampled.guest.address.checked_add(selected_offset) else {
        return false;
    };
    if face_bytes == 0 || resident.descriptor.address < selected_address {
        return false;
    }
    let offset = resident.descriptor.address - selected_address;
    if !offset.is_multiple_of(face_bytes) {
        return false;
    }
    let Ok(layer) = u32::try_from(offset / face_bytes) else {
        return false;
    };
    depth_array_face_descriptor(sampled, layer) == Some(resident.descriptor)
}

/// Returns whether every face required by a layered depth sample has a usable GPU resident.
///
/// Callers use this to decide whether a face can remain GPU-only. A partial set cannot be
/// assembled without losing the available face contents, so those residents must take the
/// ordinary alias-writeback path before the complete guest allocation is uploaded.
pub(crate) fn depth_array_face_set_is_complete(
    sampled: DecodedImageDescriptor,
    mut face_is_resident: impl FnMut(GuestTextureDescriptor) -> bool,
) -> bool {
    sampled.layer_count >= 2
        && (0..sampled.layer_count).all(|layer| {
            depth_array_face_descriptor(sampled, layer).is_some_and(&mut face_is_resident)
        })
}

impl MetalPresenter {
    pub(crate) fn depth_array_is_fully_resident(&self, sampled: DecodedImageDescriptor) -> bool {
        depth_array_face_set_is_complete(sampled, |face| {
            self.resident_guest_textures
                .get(&face)
                .is_some_and(|resident| {
                    resident.gpu_written()
                        && resident.serves_reads()
                        && resident.texture_type == 9
                        && resident.layer_count == 1
                })
        })
    }

    /// Assembles the depth-format array textures behind this draw's layered depth-sample SRDs.
    ///
    /// Every layer must resolve to a GPU-written resident depth face; partially resident arrays
    /// fall back to the guest-memory path, and their cache entry is dropped so a stale assembly
    /// is never served. Slice copies are encoded into the open guest command batch, which orders
    /// them after the face render passes already encoded there.
    pub(crate) fn prepare_depth_array_textures(
        &mut self,
        device: &Device,
        evaluations: &[&ShaderEvaluation],
    ) -> Result<(), String> {
        for &evaluation in evaluations {
            for binding in evaluation
                .image_bindings
                .iter()
                .filter(|binding| !is_storage_image_opcode(binding.opcode))
                .filter(|binding| !image_resource_descriptor_is_null(&binding.resource_descriptor))
            {
                let Ok(descriptor) = decode_image_descriptor(&binding.resource_descriptor) else {
                    continue;
                };
                if !is_layered_sample_type(descriptor.texture_type)
                    || depth_render_format_for_sample(descriptor.guest.pixel_format).is_none()
                {
                    continue;
                }
                let key = DepthArrayKey {
                    descriptor: descriptor.guest,
                    base_array_slice: descriptor.base_array_slice,
                    layer_count: descriptor.layer_count,
                };
                let faces: Option<Vec<(metal::Texture, u64)>> = (0..descriptor.layer_count)
                    .map(|layer| {
                        let face = depth_array_face_descriptor(descriptor, layer)?;
                        let resident = self.resident_guest_textures.get(&face)?;
                        (resident.gpu_written()
                            && resident.serves_reads()
                            && resident.texture_type == 9
                            && resident.layer_count == 1)
                            .then(|| (resident.texture.clone(), resident.content_version))
                    })
                    .collect();
                let Some(faces) = faces else {
                    static UNAVAILABLE_LOGS: AtomicUsize = AtomicUsize::new(0);
                    if trace_target_sampling()
                        && UNAVAILABLE_LOGS.fetch_add(1, Ordering::Relaxed) < 64
                    {
                        eprintln!(
                            "[REMu][METAL][DEPTH-ARRAY] unavailable address={:#x} layers={} faces={:?}",
                            descriptor.guest.address,
                            descriptor.layer_count,
                            (0..descriptor.layer_count)
                                .filter_map(|layer| {
                                    let face = depth_array_face_descriptor(descriptor, layer)?;
                                    Some((
                                        layer,
                                        face.address,
                                        self.resident_guest_textures.get(&face).map(|resident| {
                                            (
                                                resident.gpu_written(),
                                                resident.content_version,
                                                resident.texture_type,
                                                resident.layer_count,
                                            )
                                        }),
                                    ))
                                })
                                .collect::<Vec<_>>(),
                        );
                    }
                    // Serving a previously assembled array after its faces stop being resident
                    // would replay stale contents over whatever the guest wrote since.
                    self.depth_array_textures.remove(&key);
                    continue;
                };
                self.depth_array_textures.entry(key).or_insert_with(|| {
                    let metal_descriptor = TextureDescriptor::new();
                    metal_descriptor.set_texture_type(MTLTextureType::D2Array);
                    metal_descriptor.set_pixel_format(faces[0].0.pixel_format());
                    metal_descriptor.set_width(u64::from(descriptor.guest.width));
                    metal_descriptor.set_height(u64::from(descriptor.guest.height));
                    metal_descriptor.set_array_length(u64::from(descriptor.layer_count));
                    metal_descriptor.set_storage_mode(MTLStorageMode::Private);
                    metal_descriptor.set_usage(MTLTextureUsage::ShaderRead);
                    let texture = device.new_texture(&metal_descriptor);
                    texture.set_label(&format!(
                        "REMu layered depth-sample assembly {:#x} x{}",
                        descriptor.guest.address, descriptor.layer_count
                    ));
                    DepthArrayResidentTexture {
                        texture,
                        slice_versions: vec![u64::MAX; descriptor.layer_count as usize],
                    }
                });
                let entry = self
                    .depth_array_textures
                    .get_mut(&key)
                    .expect("depth array entry was inserted above");
                let stale: Vec<usize> = (0..faces.len())
                    .filter(|&index| entry.slice_versions[index] != faces[index].1)
                    .collect();
                static ASSEMBLY_LOGS: AtomicUsize = AtomicUsize::new(0);
                if trace_target_sampling()
                    && !stale.is_empty()
                    && ASSEMBLY_LOGS.fetch_add(1, Ordering::Relaxed) < 64
                {
                    eprintln!(
                        "[REMu][METAL][DEPTH-ARRAY] assemble address={:#x} layers={} versions={:?} stale={stale:?}",
                        descriptor.guest.address,
                        descriptor.layer_count,
                        faces
                            .iter()
                            .map(|(_, version)| *version)
                            .collect::<Vec<_>>(),
                    );
                }
                if stale.is_empty() {
                    continue;
                }
                let destination = entry.texture.clone();
                for &index in &stale {
                    entry.slice_versions[index] = faces[index].1;
                }
                let batch = self
                    .active_open_guest_batch_mut()
                    .ok_or("layered depth-sample assembly requires an open guest batch")?;
                batch.close_active_pass_encoders();
                let blit = batch.command_buffer.new_blit_command_encoder();
                blit.set_label("REMu layered depth-sample assembly");
                for index in stale {
                    blit.copy_from_texture(
                        &faces[index].0,
                        0,
                        0,
                        MTLOrigin { x: 0, y: 0, z: 0 },
                        MTLSize::new(
                            u64::from(descriptor.guest.width),
                            u64::from(descriptor.guest.height),
                            1,
                        ),
                        &destination,
                        u64::try_from(index).expect("layer index fits u64"),
                        0,
                        MTLOrigin { x: 0, y: 0, z: 0 },
                    );
                }
                blit.end_encoding();
                batch.record_resident_write_footprint(
                    SharedEventWriteKind::InternalResidentDestination,
                    descriptor.span,
                );
            }
        }
        Ok(())
    }
}
