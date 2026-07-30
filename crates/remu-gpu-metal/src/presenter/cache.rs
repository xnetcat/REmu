// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CachedGlobalBuffer, CachedNggMeshlets, CachedSamplerArguments, DecodedDepthTarget,
    GlobalBufferCacheKey, GuestCommandBatch, MetalPresenter, PendingGlobalWriteback,
    SAMPLER_ARGUMENT_CACHE_CAPACITY, SamplerArgumentCacheKey, UploadedGlobalBuffer,
    canonical_sampler_bindings, create_sampler_argument_buffer, guest_batch_writes_overlap,
    guest_global_buffer_view, guest_physical_ranges_overlap, metal_buffer_with_bytes,
    prepare_ngg_meshlets,
};
use metal::foreign_types::ForeignType;
use metal::{
    Buffer, DepthStencilDescriptor, DepthStencilState, Device, Function, MTLCompareFunction,
    MTLResourceOptions, SamplerState,
};
use remu_memory::guest_shared_mapping_identity;
use remu_shader::{GlobalMemoryBinding, ShaderEvaluation};
use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

enum QueuedGlobalReadSource {
    None,
    Buffer { buffer: Buffer, offset: u64 },
    GuestMemory,
    Synchronize,
}

impl MetalPresenter {
    /// Resolves a guest range consumed by a native indirect GPU command.
    ///
    /// Indirect arguments are commonly produced by an earlier GPU pass. Reading them from guest
    /// memory while parsing the PM4 stream observes stale CPU backing; keep the read on Metal
    /// instead, forwarding the exact pending writeback buffer when one owns this range. The
    /// ordinary global-coherence fallback drains only ambiguous writers before taking a fresh
    /// copy, matching shader global reads.
    pub(super) fn upload_ordered_indirect_arguments(
        &mut self,
        device: &Device,
        address: u64,
        byte_count: usize,
    ) -> Result<UploadedGlobalBuffer, String> {
        if byte_count == 0 {
            return Err("indirect argument buffer is empty".to_owned());
        }
        let binding = GlobalMemoryBinding {
            scalar_address: 0,
            base_address: address,
            instruction_pcs: Default::default(),
            byte_count,
            data: Vec::new(),
            writable: false,
            write_back_to_guest: false,
        };
        let source = self.queued_global_read_source(&binding)?;
        static INDIRECT_UPLOAD_COUNT: AtomicUsize = AtomicUsize::new(0);
        static TRACE_ALL_INDIRECT_UPLOADS: OnceLock<bool> = OnceLock::new();
        let sequence = INDIRECT_UPLOAD_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        let trace_all = *TRACE_ALL_INDIRECT_UPLOADS
            .get_or_init(|| std::env::var_os("REMU_TRACE_INDIRECT_BUFFERS").is_some());
        if trace_all || sequence <= 16 || sequence.is_power_of_two() {
            let source_detail = match &source {
                QueuedGlobalReadSource::None => "guest".to_owned(),
                QueuedGlobalReadSource::Buffer { buffer, offset } => format!(
                    "gpu-writeback(buffer={:p},offset={offset:#x})",
                    buffer.as_ptr()
                ),
                QueuedGlobalReadSource::GuestMemory => "ordered-guest".to_owned(),
                QueuedGlobalReadSource::Synchronize => "synchronize".to_owned(),
            };
            eprintln!(
                "[REMu][METAL][INDIRECT-BUFFER] sequence={sequence} address={address:#x} bytes={byte_count} source={source_detail}"
            );
        }
        if matches!(source, QueuedGlobalReadSource::Synchronize) {
            let queue = self
                .command_queue
                .as_ref()
                .ok_or("Metal command queue is unavailable")?
                .clone();
            self.wait_for_guest_command_batches()?;
            self.ensure_open_guest_batch(&queue);
            return self.upload_fresh_global_buffer(device, &binding);
        }
        self.upload_global_buffer_from_source(device, &binding, source)
    }

    pub(super) fn upload_indirect_dispatch_arguments(
        &mut self,
        device: &Device,
        address: u64,
    ) -> Result<UploadedGlobalBuffer, String> {
        self.upload_ordered_indirect_arguments(device, address, 3 * std::mem::size_of::<u32>())
    }

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
        let sources = evaluation
            .global_memory_bindings
            .iter()
            .map(|binding| self.queued_global_read_source(binding))
            .collect::<Result<Vec<_>, _>>()?;
        if sources
            .iter()
            .any(|source| matches!(source, QueuedGlobalReadSource::Synchronize))
        {
            let queue = self
                .command_queue
                .as_ref()
                .ok_or("Metal command queue is unavailable")?
                .clone();
            self.wait_for_guest_command_batches()?;
            self.ensure_open_guest_batch(&queue);
            return evaluation
                .global_memory_bindings
                .iter()
                .zip(sources)
                .map(|(binding, source)| {
                    if matches!(source, QueuedGlobalReadSource::None) {
                        self.upload_global_buffer_from_packet(device, binding)
                    } else {
                        trace_global_coherence(binding, "synchronized");
                        self.upload_fresh_global_buffer(device, binding)
                    }
                })
                .collect();
        }
        evaluation
            .global_memory_bindings
            .iter()
            .zip(sources)
            .map(|(binding, source)| self.upload_global_buffer_from_source(device, binding, source))
            .collect()
    }

    #[cfg(test)]
    pub(super) fn upload_global_buffer(
        &mut self,
        device: &Device,
        binding: &GlobalMemoryBinding,
    ) -> Result<UploadedGlobalBuffer, String> {
        let source = self.queued_global_read_source(binding)?;
        if matches!(source, QueuedGlobalReadSource::Synchronize) {
            let queue = self
                .command_queue
                .as_ref()
                .ok_or("Metal command queue is unavailable")?
                .clone();
            self.wait_for_guest_command_batches()?;
            self.ensure_open_guest_batch(&queue);
            trace_global_coherence(binding, "synchronized");
            return self.upload_fresh_global_buffer(device, binding);
        }
        self.upload_global_buffer_from_source(device, binding, source)
    }

    fn upload_global_buffer_from_source(
        &mut self,
        device: &Device,
        binding: &GlobalMemoryBinding,
        source: QueuedGlobalReadSource,
    ) -> Result<UploadedGlobalBuffer, String> {
        match source {
            QueuedGlobalReadSource::Buffer { buffer, offset } => {
                trace_global_coherence(binding, "forwarded-buffer");
                Ok(UploadedGlobalBuffer {
                    buffer,
                    offset,
                    guest_memory: false,
                })
            }
            QueuedGlobalReadSource::GuestMemory => {
                trace_global_coherence(binding, "forwarded-guest-memory");
                if let Some(uploaded) = self.upload_identity_global_buffer(device, binding)? {
                    return Ok(uploaded);
                }
                // A queued producer may outlive the exact virtual mapping used when its work was
                // recorded. Metal's bytes-no-copy API does not reject a partially unmapped range
                // synchronously on every macOS version; the command buffer can instead remain
                // scheduled forever after IOGPUResource creation fails. Drain the producer and
                // take a validated backing copy when a complete identity view is unavailable.
                trace_global_coherence(binding, "forwarded-copy-fallback");
                let queue = self
                    .command_queue
                    .as_ref()
                    .ok_or("Metal command queue is unavailable")?
                    .clone();
                self.wait_for_guest_command_batches()?;
                self.ensure_open_guest_batch(&queue);
                self.upload_fresh_global_buffer(device, binding)
            }
            QueuedGlobalReadSource::Synchronize => {
                unreachable!("synchronizing global sources are handled before upload")
            }
            QueuedGlobalReadSource::None => self.upload_global_buffer_from_packet(device, binding),
        }
    }

    fn upload_global_buffer_from_packet(
        &mut self,
        device: &Device,
        binding: &GlobalMemoryBinding,
    ) -> Result<UploadedGlobalBuffer, String> {
        // Packet-time snapshots must win over the identity-mapped view. The guest can recycle
        // transient constants and small vertex buffers while this presenter is still compiling
        // shaders or draining older events. A queued GPU writer is resolved before this method,
        // because in that case the packet snapshot necessarily predates the producer.
        if (binding.data.len() != binding.byte_count || binding.writable)
            && let Some(uploaded) = self.upload_identity_global_buffer(device, binding)?
        {
            return Ok(uploaded);
        }
        if binding.data.len() != binding.byte_count {
            // Large bindings are deliberately not always captured in the packet. If page
            // alignment crosses an unmapped edge, use the physical-backing registry to copy the
            // exact requested range rather than asking Metal to pin the invalid aligned span.
            trace_global_coherence(binding, "packet-copy-fallback");
            return self.upload_fresh_global_buffer(device, binding);
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

    fn upload_identity_global_buffer(
        &mut self,
        device: &Device,
        binding: &GlobalMemoryBinding,
    ) -> Result<Option<UploadedGlobalBuffer>, String> {
        let Some((page_address, page_byte_count, offset)) = guest_global_buffer_view(binding)?
        else {
            return Ok(None);
        };
        let page_byte_count_u64 = u64::try_from(page_byte_count)
            .map_err(|_| "page-aligned guest global size does not fit u64")?;
        if page_byte_count_u64 > GUEST_GLOBAL_BUFFER_MAX_ZERO_COPY_BYTES {
            return Ok(None);
        }
        // `new_buffer_with_bytes_no_copy` needs every page in the outward-aligned host range.
        // Validating only the shader's interior binding left the first/last page able to cross a
        // retired mapping, which macOS reports as IOGPUSysMemory::memoryDescriptorWithClientBuffer
        // failure and can leave waitUntilCompleted blocked indefinitely.
        let Some(mapping_identity) =
            guest_shared_mapping_identity(page_address, page_byte_count_u64)
        else {
            return Ok(None);
        };
        let key = (page_address, page_byte_count, mapping_identity);
        let buffer = if let Some(buffer) = self.guest_global_buffer_cache.get(&key) {
            buffer.clone()
        } else {
            // Drop a cached view when the same VA is rebound to a different physical allocation.
            // In-flight command batches retain their own Buffer clones until completion.
            let mut retired_bytes = 0_u64;
            self.guest_global_buffer_cache
                .retain(|(cached_address, cached_bytes, _), buffer| {
                    let retain =
                        *cached_address != page_address || *cached_bytes != page_byte_count;
                    if !retain {
                        retired_bytes = retired_bytes.saturating_add(buffer.length());
                    }
                    retain
                });
            self.guest_global_buffer_cache_bytes = self
                .guest_global_buffer_cache_bytes
                .saturating_sub(retired_bytes);
            if self.guest_global_buffer_cache.len() >= GUEST_GLOBAL_BUFFER_CACHE_CAPACITY
                || self
                    .guest_global_buffer_cache_bytes
                    .saturating_add(page_byte_count_u64)
                    > GUEST_GLOBAL_BUFFER_CACHE_BYTE_CAPACITY
            {
                self.guest_global_buffer_cache.clear();
                self.guest_global_buffer_cache_bytes = 0;
            }
            let buffer = device.new_buffer_with_bytes_no_copy(
                usize::try_from(page_address)
                    .map_err(|_| "guest global address does not fit the host pointer")?
                    as *const c_void,
                page_byte_count_u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            buffer.set_label(&format!(
                "REMu zero-copy guest global {page_address:#x}/{page_byte_count}"
            ));
            self.guest_global_buffer_cache.insert(key, buffer.clone());
            self.guest_global_buffer_cache_bytes = self
                .guest_global_buffer_cache_bytes
                .saturating_add(page_byte_count_u64);
            buffer
        };
        Ok(Some(UploadedGlobalBuffer {
            buffer,
            offset,
            guest_memory: true,
        }))
    }

    fn upload_fresh_global_buffer(
        &mut self,
        device: &Device,
        binding: &GlobalMemoryBinding,
    ) -> Result<UploadedGlobalBuffer, String> {
        let (buffer, offset) = self
            .open_guest_batch
            .as_mut()
            .ok_or("synchronized global upload has no open guest batch")?
            .upload_guest_global_bytes(device, binding.base_address, binding.byte_count)?;
        Ok(UploadedGlobalBuffer {
            buffer,
            offset,
            guest_memory: false,
        })
    }

    fn queued_global_read_source(
        &self,
        binding: &GlobalMemoryBinding,
    ) -> Result<QueuedGlobalReadSource, String> {
        let byte_count = u64::try_from(binding.byte_count)
            .map_err(|_| "global binding byte count does not fit u64")?;
        if let Some(batch) = self.open_guest_batch.as_ref() {
            let source =
                queued_global_read_source_in_batch(batch, binding.base_address, byte_count);
            if !matches!(source, QueuedGlobalReadSource::None) {
                return Ok(source);
            }
        }
        for batch in self.committed_guest_batches.iter().rev() {
            let source =
                queued_global_read_source_in_batch(batch, binding.base_address, byte_count);
            if !matches!(source, QueuedGlobalReadSource::None) {
                return Ok(source);
            }
        }
        Ok(QueuedGlobalReadSource::None)
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

    pub(super) fn cached_ngg_meshlets(
        &mut self,
        device: &Device,
        indices: &[u32],
    ) -> Result<Option<CachedNggMeshlets>, String> {
        if let Some(cached) = self.ngg_meshlet_cache.get(indices) {
            return Ok(Some(cached.clone()));
        }
        let prepared = prepare_ngg_meshlets(indices);
        if prepared.group_count == 0 {
            return Ok(None);
        }
        let byte_count = prepared
            .words
            .len()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or("NGG meshlet byte count overflow")?;
        // SAFETY: u32 has no invalid bit patterns and Metal copies this exact live slice before
        // returning from metal_buffer_with_bytes.
        let bytes =
            unsafe { std::slice::from_raw_parts(prepared.words.as_ptr().cast::<u8>(), byte_count) };
        let cached = CachedNggMeshlets {
            buffer: metal_buffer_with_bytes(device, bytes)?,
            group_count: prepared.group_count,
            vertex_references: prepared.vertex_references,
            primitive_count: prepared.primitive_count,
        };
        if self.ngg_meshlet_cache.len() >= NGG_MESHLET_CACHE_CAPACITY {
            self.ngg_meshlet_cache.clear();
        }
        self.ngg_meshlet_cache
            .insert(indices.to_vec(), cached.clone());
        Ok(Some(cached))
    }
}

fn queued_global_read_source_in_batch(
    batch: &GuestCommandBatch,
    address: u64,
    byte_count: u64,
) -> QueuedGlobalReadSource {
    if !guest_batch_writes_overlap(batch, address, byte_count) {
        return QueuedGlobalReadSource::None;
    }
    if batch.memory_writes.iter().any(|write| {
        let (write_address, write_byte_count) = write.range();
        guest_physical_ranges_overlap(write_address, write_byte_count, address, byte_count)
    }) {
        return QueuedGlobalReadSource::Synchronize;
    }
    let mut overlapping_writebacks = batch.global_writebacks.iter().filter(|writeback| {
        u64::try_from(writeback.byte_count).is_ok_and(|writeback_byte_count| {
            guest_physical_ranges_overlap(
                writeback.guest_address,
                writeback_byte_count,
                address,
                byte_count,
            )
        })
    });
    let Some(first) = overlapping_writebacks.next() else {
        return QueuedGlobalReadSource::GuestMemory;
    };
    let Some(offset) = contained_global_writeback_offset(first, address, byte_count) else {
        return QueuedGlobalReadSource::Synchronize;
    };
    if overlapping_writebacks.any(|writeback| {
        contained_global_writeback_offset(writeback, address, byte_count).is_none_or(
            |candidate_offset| {
                writeback.buffer.as_ptr() != first.buffer.as_ptr() || candidate_offset != offset
            },
        )
    }) {
        return QueuedGlobalReadSource::Synchronize;
    }
    QueuedGlobalReadSource::Buffer {
        buffer: first.buffer.clone(),
        offset,
    }
}

fn contained_global_writeback_offset(
    writeback: &PendingGlobalWriteback,
    address: u64,
    byte_count: u64,
) -> Option<u64> {
    let writeback_byte_count = u64::try_from(writeback.byte_count).ok()?;
    let writeback_end = writeback.guest_address.checked_add(writeback_byte_count)?;
    let read_end = address.checked_add(byte_count)?;
    if address < writeback.guest_address || read_end > writeback_end {
        return None;
    }
    writeback
        .offset
        .checked_add(address - writeback.guest_address)
}

fn trace_global_coherence(binding: &GlobalMemoryBinding, source: &str) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var_os("REMU_TRACE_GLOBAL_COHERENCE").is_some()) {
        return;
    }
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    if COUNT.fetch_add(1, Ordering::Relaxed) >= 128 {
        return;
    }
    eprintln!(
        "[REMu][METAL][GLOBAL_COHERENCE] address={:#x} bytes={} source={source}",
        binding.base_address, binding.byte_count
    );
}
pub(super) const GLOBAL_BUFFER_CACHE_CAPACITY: usize = 4_096;

pub(super) const GLOBAL_BUFFER_CACHE_BYTE_CAPACITY: usize = 64 * 1024 * 1024;

pub(super) const GLOBAL_UPLOAD_CHUNK_BYTE_COUNT: usize = 2 * 1024 * 1024;

pub(super) const GLOBAL_UPLOAD_ALIGNMENT: usize = 256;

pub(super) const GUEST_GLOBAL_BUFFER_CACHE_CAPACITY: usize = 1_024;

pub(super) const GUEST_GLOBAL_BUFFER_CACHE_BYTE_CAPACITY: u64 = 512 * 1024 * 1024;

pub(super) const GUEST_GLOBAL_BUFFER_MAX_ZERO_COPY_BYTES: u64 = 256 * 1024 * 1024;

pub(super) const INDEX_BUFFER_CACHE_CAPACITY: usize = 256;

pub(super) const NGG_MESHLET_CACHE_CAPACITY: usize = 64;
