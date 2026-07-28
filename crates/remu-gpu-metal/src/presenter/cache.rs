// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CachedGlobalBuffer, CachedSamplerArguments, DecodedDepthTarget, GlobalBufferCacheKey,
    MetalPresenter, SAMPLER_ARGUMENT_CACHE_CAPACITY, SamplerArgumentCacheKey, UploadedGlobalBuffer,
    canonical_sampler_bindings, create_sampler_argument_buffer, guest_global_buffer_view,
    metal_buffer_with_bytes,
};
use metal::foreign_types::ForeignType;
use metal::{
    Buffer, DepthStencilDescriptor, DepthStencilState, Device, Function, MTLCompareFunction,
    MTLResourceOptions, SamplerState,
};
use remu_shader::{GlobalMemoryBinding, ShaderEvaluation};
use std::ffi::c_void;

impl MetalPresenter {
    pub(super) fn cached_sampler_argument_buffer(
        &mut self,
        device: &Device,
        function: &Function,
        buffer_index: Option<u32>,
        evaluation: &ShaderEvaluation,
    ) -> Result<(Option<Buffer>, Vec<SamplerState>), String> {
        let sampled_descriptors = canonical_sampler_bindings(evaluation)
            .into_iter()
            .map(|binding| binding.sampler_descriptor.clone())
            .collect::<Vec<_>>();
        let key = (!sampled_descriptors.is_empty()).then(|| SamplerArgumentCacheKey {
            function: buffer_index
                .is_some()
                .then_some(function.as_ptr() as usize)
                .unwrap_or_default(),
            buffer_index,
            descriptors: sampled_descriptors
                .into_iter()
                .map(|descriptor| descriptor.to_vec())
                .collect(),
        });
        if let Some(cached) = key
            .as_ref()
            .and_then(|key| self.sampler_argument_cache.get(key))
        {
            return Ok((cached.buffer.clone(), cached.samplers.clone()));
        }
        let created = create_sampler_argument_buffer(device, function, buffer_index, evaluation)?;
        if self.sampler_argument_cache.len() < SAMPLER_ARGUMENT_CACHE_CAPACITY
            && let Some(key) = key
        {
            self.sampler_argument_cache.insert(
                key,
                CachedSamplerArguments {
                    buffer: created.0.clone(),
                    samplers: created.1.clone(),
                },
            );
        }
        Ok(created)
    }

    pub(super) fn cached_depth_stencil_state(
        &mut self,
        device: &Device,
        depth: DecodedDepthTarget,
    ) -> DepthStencilState {
        let compare = if depth.test_enabled && !depth.clear_enabled {
            depth.compare
        } else {
            MTLCompareFunction::Always
        };
        let write_enabled = depth.write_enabled && !depth.clear_enabled;
        let key = (compare as u64, write_enabled);
        if let Some(state) = self.depth_stencil_cache.get(&key) {
            return state.clone();
        }
        let descriptor = DepthStencilDescriptor::new();
        descriptor.set_depth_compare_function(compare);
        descriptor.set_depth_write_enabled(write_enabled);
        let state = device.new_depth_stencil_state(&descriptor);
        self.depth_stencil_cache.insert(key, state.clone());
        state
    }

    pub(super) fn upload_global_buffers(
        &mut self,
        device: &Device,
        evaluation: &ShaderEvaluation,
    ) -> Result<Vec<UploadedGlobalBuffer>, String> {
        evaluation
            .global_memory_bindings
            .iter()
            .map(|binding| self.upload_global_buffer(device, binding))
            .collect()
    }

    pub(super) fn upload_global_buffer(
        &mut self,
        device: &Device,
        binding: &GlobalMemoryBinding,
    ) -> Result<UploadedGlobalBuffer, String> {
        // Packet-time snapshots must win over the identity-mapped view. The guest can recycle
        // transient constants and small vertex buffers while this presenter is still compiling
        // shaders or draining older events. Zero-copy remains correct for writable or
        // deliberately uncaptured large bindings.
        if (binding.data.len() != binding.byte_count || binding.writable)
            && let Some((page_address, page_byte_count, offset)) =
                guest_global_buffer_view(binding)?
        {
            let key = (page_address, page_byte_count);
            let buffer = if let Some(buffer) = self.guest_global_buffer_cache.get(&key) {
                buffer.clone()
            } else {
                if self.guest_global_buffer_cache.len() >= GUEST_GLOBAL_BUFFER_CACHE_CAPACITY {
                    self.guest_global_buffer_cache.clear();
                }
                let length = u64::try_from(page_byte_count)
                    .map_err(|_| "page-aligned guest global size does not fit Metal")?;
                let buffer = device.new_buffer_with_bytes_no_copy(
                    usize::try_from(page_address)
                        .map_err(|_| "guest global address does not fit the host pointer")?
                        as *const c_void,
                    length,
                    MTLResourceOptions::StorageModeShared,
                    None,
                );
                buffer.set_label(&format!(
                    "REMu zero-copy guest global {page_address:#x}/{page_byte_count}"
                ));
                self.guest_global_buffer_cache.insert(key, buffer.clone());
                buffer
            };
            return Ok(UploadedGlobalBuffer {
                buffer,
                offset,
                guest_memory: true,
            });
        }
        if binding.data.len() != binding.byte_count {
            return Err(format!(
                "global binding {:#x}/{} is neither identity-mapped nor captured",
                binding.base_address, binding.byte_count
            ));
        }
        let key = GlobalBufferCacheKey {
            base_address: binding.base_address,
            byte_count: binding.data.len(),
        };
        if binding.writable || binding.data.len() > GLOBAL_BUFFER_CACHE_BYTE_CAPACITY {
            if let Some(cached) = self.transient_global_buffer_cache.get(&key)
                && cached.bytes == binding.data
            {
                return Ok(UploadedGlobalBuffer {
                    buffer: cached.buffer.clone(),
                    offset: cached.offset,
                    guest_memory: false,
                });
            }
            let (buffer, offset) = self
                .open_guest_batch
                .as_mut()
                .ok_or("captured global upload has no open guest batch")?
                .upload_global_bytes(device, &binding.data)?;
            self.transient_global_buffer_cache.insert(
                key,
                CachedGlobalBuffer {
                    bytes: binding.data.clone(),
                    buffer: buffer.clone(),
                    offset,
                },
            );
            return Ok(UploadedGlobalBuffer {
                buffer,
                offset,
                guest_memory: false,
            });
        }
        if let Some(cached) = self.global_buffer_cache.get(&key)
            && cached.bytes == binding.data
        {
            return Ok(UploadedGlobalBuffer {
                buffer: cached.buffer.clone(),
                offset: cached.offset,
                guest_memory: false,
            });
        }
        // A stable guest address whose packet-time bytes keep changing is a transient constant
        // stream. Keep its first immutable cache entry as a possible exact hit, but place later
        // variants in the current command batch's arena instead of allocating and destroying an
        // MTLBuffer for every draw.
        if self.global_buffer_cache.contains_key(&key) {
            let (buffer, offset) = self
                .open_guest_batch
                .as_mut()
                .ok_or("captured global upload has no open guest batch")?
                .upload_global_bytes(device, &binding.data)?;
            return Ok(UploadedGlobalBuffer {
                buffer,
                offset,
                guest_memory: false,
            });
        }
        if self.global_buffer_cache.len() >= GLOBAL_BUFFER_CACHE_CAPACITY
            || self
                .global_buffer_cache_bytes
                .saturating_add(binding.data.len())
                > GLOBAL_BUFFER_CACHE_BYTE_CAPACITY
        {
            let (buffer, offset) = self
                .open_guest_batch
                .as_mut()
                .ok_or("captured global upload has no open guest batch")?
                .upload_global_bytes(device, &binding.data)?;
            return Ok(UploadedGlobalBuffer {
                buffer,
                offset,
                guest_memory: false,
            });
        }
        let buffer = metal_buffer_with_bytes(device, &binding.data)?;
        let offset = 0;
        if let Some(replaced) = self.global_buffer_cache.insert(
            key,
            CachedGlobalBuffer {
                bytes: binding.data.clone(),
                buffer: buffer.clone(),
                offset,
            },
        ) {
            self.global_buffer_cache_bytes = self
                .global_buffer_cache_bytes
                .saturating_sub(replaced.bytes.len());
        }
        self.global_buffer_cache_bytes = self
            .global_buffer_cache_bytes
            .saturating_add(binding.data.len());
        Ok(UploadedGlobalBuffer {
            buffer,
            offset,
            guest_memory: false,
        })
    }

    pub(super) fn cached_index_buffer(
        &mut self,
        device: &Device,
        indices: &[u32],
    ) -> Result<Buffer, String> {
        if let Some(cached) = self.index_buffer_cache.get(indices) {
            return Ok(cached.clone());
        }
        let byte_count = indices
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or("index byte count overflow")?;
        let bytes = if byte_count == 0 {
            &[][..]
        } else {
            // SAFETY: u32 has no invalid bit patterns and this view has the exact byte extent of
            // the live index slice. Metal copies it before this method returns.
            unsafe { std::slice::from_raw_parts(indices.as_ptr().cast::<u8>(), byte_count) }
        };
        let buffer = metal_buffer_with_bytes(device, bytes)?;
        if self.index_buffer_cache.len() >= INDEX_BUFFER_CACHE_CAPACITY {
            self.index_buffer_cache.clear();
        }
        self.index_buffer_cache
            .insert(indices.to_vec(), buffer.clone());
        Ok(buffer)
    }
}
pub(super) const GLOBAL_BUFFER_CACHE_CAPACITY: usize = 4_096;

pub(super) const GLOBAL_BUFFER_CACHE_BYTE_CAPACITY: usize = 64 * 1024 * 1024;

pub(super) const GLOBAL_UPLOAD_CHUNK_BYTE_COUNT: usize = 2 * 1024 * 1024;

pub(super) const GLOBAL_UPLOAD_ALIGNMENT: usize = 256;

pub(super) const GUEST_GLOBAL_BUFFER_CACHE_CAPACITY: usize = 4_096;

pub(super) const INDEX_BUFFER_CACHE_CAPACITY: usize = 256;
