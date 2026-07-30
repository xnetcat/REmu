// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CachedGlobalBuffer, CachedNggMeshlets, CachedSamplerArguments, DecodedDepthTarget,
    GlobalBufferCacheKey, GuestCommandBatch, MetalPresenter, NggSubgroupLimits,
    PendingGlobalWriteback, SAMPLER_ARGUMENT_CACHE_CAPACITY, SamplerArgumentCacheKey,
    UploadedGlobalBuffer, canonical_sampler_bindings, create_sampler_argument_buffer,
    describe_guest_batch, guest_batch_writes_overlap, guest_physical_ranges_overlap,
    metal_buffer_with_bytes, prepare_ngg_meshlets,
};
use block::ConcreteBlock;
use metal::foreign_types::ForeignType;
use metal::{
    Buffer, DepthStencilDescriptor, DepthStencilState, Device, Function, MTLCompareFunction,
    MTLResourceOptions, SamplerState,
};
use remu_memory::{
    GuestSharedBackingRange, GuestSharedMappingIdentity, guest_shared_backing_identity,
    guest_shared_backing_range, read_guest_shared_backing,
};
use remu_shader::{GlobalMemoryBinding, ShaderEvaluation};
use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

enum QueuedGlobalReadSource {
    None,
    Buffer {
        buffer: Buffer,
        offset: u64,
        guest_memory: bool,
        physical_identity: Option<remu_memory::GuestSharedMappingIdentity>,
    },
    Synchronize,
}

fn read_indirect_guest_dwords(address: u64, byte_count: usize) -> Option<Vec<u32>> {
    let mut bytes = vec![0_u8; byte_count];
    if !read_guest_shared_backing(address, &mut bytes) {
        return None;
    }
    Some(decode_indirect_dwords(&bytes))
}

fn decode_indirect_dwords(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|word| u32::from_le_bytes(word.try_into().expect("dword chunk has four bytes")))
        .collect()
}

fn indirect_producer_overlap_summary(
    producer: &GuestCommandBatch,
    address: u64,
    byte_count: u64,
) -> String {
    let mut overlaps = Vec::new();
    overlaps.extend(
        producer
            .guest_write_ranges
            .iter()
            .filter(|range| {
                guest_physical_ranges_overlap(range.address, range.byte_count, address, byte_count)
            })
            .map(|range| format!("guest-write@{:#x}/{:#x}", range.address, range.byte_count)),
    );
    overlaps.extend(producer.global_writebacks.iter().filter_map(|writeback| {
        let writeback_byte_count = u64::try_from(writeback.byte_count).ok()?;
        guest_physical_ranges_overlap(
            writeback.guest_address,
            writeback_byte_count,
            address,
            byte_count,
        )
        .then(|| {
            format!(
                "global-writeback@{:#x}/{:#x}+buffer-offset-{:#x}",
                writeback.guest_address, writeback_byte_count, writeback.offset,
            )
        })
    }));
    overlaps.extend(producer.memory_writes.iter().filter_map(|write| {
        let (write_address, write_byte_count) = write.range();
        guest_physical_ranges_overlap(write_address, write_byte_count, address, byte_count)
            .then(|| format!("memory-write@{write_address:#x}/{write_byte_count:#x}"))
    }));
    overlaps.join(",")
}

pub(super) fn metal_buffer_with_guest_backing_range(
    device: &Device,
    backing_range: &GuestSharedBackingRange,
) -> Option<Buffer> {
    let backing_view = backing_range.map_view()?;
    let backing_pointer = backing_view.as_ptr();
    let view_byte_count = u64::try_from(backing_view.byte_count()).ok()?;
    let deallocator = ConcreteBlock::new(move |_bytes: *const c_void, _length: u64| {
        // Metal invokes this callback only when the final Buffer reference is released.
        // Capturing the mmap owner keeps the no-copy storage alive for cached and in-flight
        // command-buffer references alike.
        let _ = &backing_view;
    })
    .copy();
    Some(device.new_buffer_with_bytes_no_copy(
        backing_pointer,
        view_byte_count,
        MTLResourceOptions::StorageModeShared,
        Some(&deallocator),
    ))
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
        self.trace_indirect_producer_selection(address, byte_count, &source);
        static INDIRECT_UPLOAD_COUNT: AtomicUsize = AtomicUsize::new(0);
        static TRACE_ALL_INDIRECT_UPLOADS: OnceLock<bool> = OnceLock::new();
        let sequence = INDIRECT_UPLOAD_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        let trace_all = *TRACE_ALL_INDIRECT_UPLOADS
            .get_or_init(|| std::env::var_os("REMU_TRACE_INDIRECT_BUFFERS").is_some());
        if trace_all || sequence <= 16 || sequence.is_power_of_two() {
            let source_detail = match &source {
                QueuedGlobalReadSource::None => "guest".to_owned(),
                QueuedGlobalReadSource::Buffer {
                    buffer,
                    offset,
                    guest_memory,
                    physical_identity,
                } => format!(
                    "gpu-buffer(buffer={:p},offset={offset:#x},guest_memory={guest_memory},physical={physical_identity:?})",
                    buffer.as_ptr(),
                ),
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

    /// Identifies the queued producer selected for a native indirect argument read.
    ///
    /// The ordinary indirect-buffer trace deliberately reports only the source category. This
    /// opt-in diagnostic reports the producer's complete ordered Metal batch and the CPU-visible
    /// dwords at selection time, without synchronizing or otherwise perturbing the queue.
    fn trace_indirect_producer_selection(
        &self,
        address: u64,
        byte_count: usize,
        source: &QueuedGlobalReadSource,
    ) {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        if !*ENABLED.get_or_init(|| {
            std::env::var_os("REMU_TRACE_INDIRECT_PRODUCERS").is_some_and(|value| value != "0")
        }) {
            return;
        }
        let byte_count_u64 = u64::try_from(byte_count).unwrap_or(u64::MAX);
        let selected = self
            .open_guest_batch
            .as_ref()
            .filter(|batch| guest_batch_writes_overlap(batch, address, byte_count_u64))
            .map(|batch| ("open", 0, batch))
            .or_else(|| {
                self.committed_guest_batches
                    .iter()
                    .rev()
                    .enumerate()
                    .find(|(_, batch)| guest_batch_writes_overlap(batch, address, byte_count_u64))
                    .map(|(reverse_index, batch)| ("committed", reverse_index, batch))
            });
        let source = match source {
            QueuedGlobalReadSource::None => "guest",
            QueuedGlobalReadSource::Buffer {
                guest_memory: true, ..
            } => "gpu-guest-buffer",
            QueuedGlobalReadSource::Buffer {
                guest_memory: false,
                ..
            } => "gpu-writeback",
            QueuedGlobalReadSource::Synchronize => "synchronize",
        };
        let dwords = read_indirect_guest_dwords(address, byte_count);
        match selected {
            Some((location, reverse_index, producer)) => {
                let overlap = indirect_producer_overlap_summary(producer, address, byte_count_u64);
                eprintln!(
                    "[REMu][METAL][INDIRECT-PRODUCER][SELECT] address={address:#x} bytes={byte_count} \
                     source={source} producer_location={location} producer_reverse_index={reverse_index} \
                     producer_status={:?} overlap=[{overlap}] \
                     cpu_visible_dwords={dwords:?} producer=[{}]",
                    producer.command_buffer.status(),
                    describe_guest_batch(&producer.work),
                );
            }
            None => eprintln!(
                "[REMu][METAL][INDIRECT-PRODUCER][SELECT] address={address:#x} bytes={byte_count} \
                 source={source} producer=none cpu_visible_dwords={dwords:?}"
            ),
        }
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
            QueuedGlobalReadSource::Buffer {
                buffer,
                offset,
                guest_memory,
                physical_identity,
            } => {
                trace_global_coherence(binding, "forwarded-buffer");
                Ok(UploadedGlobalBuffer {
                    buffer,
                    offset,
                    guest_memory,
                    physical_identity,
                })
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
                    physical_identity: None,
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
                physical_identity: None,
            });
        }
        if let Some(cached) = self.global_buffer_cache.get(&key)
            && cached.bytes == binding.data
        {
            return Ok(UploadedGlobalBuffer {
                buffer: cached.buffer.clone(),
                offset: cached.offset,
                guest_memory: false,
                physical_identity: None,
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
                physical_identity: None,
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
                physical_identity: None,
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
            physical_identity: None,
        })
    }

    fn upload_identity_global_buffer(
        &mut self,
        device: &Device,
        binding: &GlobalMemoryBinding,
    ) -> Result<Option<UploadedGlobalBuffer>, String> {
        if binding.byte_count == 0
            || binding.base_address < 0x10_000
            || !binding.base_address.is_multiple_of(4)
        {
            return Ok(None);
        }
        let byte_count = u64::try_from(binding.byte_count)
            .map_err(|_| "guest global byte count does not fit u64")?;
        let Some(backing_range) = guest_shared_backing_range(binding.base_address, byte_count)
        else {
            return Ok(None);
        };
        let view_byte_count_u64 = u64::try_from(backing_range.view_byte_count())
            .map_err(|_| "page-aligned guest global size does not fit u64")?;
        if view_byte_count_u64 > GUEST_GLOBAL_BUFFER_MAX_ZERO_COPY_BYTES {
            return Ok(None);
        }
        let view_identity = backing_range.view_identity();
        let key = (view_identity, backing_range.view_byte_count());
        let buffer = if let Some(buffer) = self.guest_global_buffer_cache.get(&key) {
            buffer.clone()
        } else {
            if self.guest_global_buffer_cache.len() >= GUEST_GLOBAL_BUFFER_CACHE_CAPACITY
                || self
                    .guest_global_buffer_cache_bytes
                    .saturating_add(view_byte_count_u64)
                    > GUEST_GLOBAL_BUFFER_CACHE_BYTE_CAPACITY
            {
                self.guest_global_buffer_cache.clear();
                self.guest_global_buffer_cache_bytes = 0;
            }
            let Some(buffer) = metal_buffer_with_guest_backing_range(device, &backing_range) else {
                return Ok(None);
            };
            buffer.set_label(&format!(
                "REMu zero-copy guest global backing={:#x}+{:#x}/{view_byte_count_u64}",
                view_identity.backing_id, view_identity.backing_offset
            ));
            self.guest_global_buffer_cache.insert(key, buffer.clone());
            self.guest_global_buffer_cache_bytes = self
                .guest_global_buffer_cache_bytes
                .saturating_add(view_byte_count_u64);
            buffer
        };
        let physical_identity = GuestSharedMappingIdentity {
            backing_id: view_identity.backing_id,
            backing_offset: view_identity
                .backing_offset
                .checked_add(backing_range.resource_offset())
                .ok_or("guest global physical identity overflows")?,
        };
        Ok(Some(UploadedGlobalBuffer {
            buffer,
            offset: backing_range.resource_offset(),
            guest_memory: true,
            physical_identity: Some(physical_identity),
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
            physical_identity: None,
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
        limits: NggSubgroupLimits,
    ) -> Result<Option<CachedNggMeshlets>, String> {
        let cache_key = (limits, indices.to_vec());
        if let Some(cached) = self.ngg_meshlet_cache.get(&cache_key) {
            return Ok(Some(cached.clone()));
        }
        let prepared = prepare_ngg_meshlets(indices, limits);
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
        self.ngg_meshlet_cache.insert(cache_key, cached.clone());
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
    let mut candidate: Option<(Buffer, u64, bool, Option<GuestSharedMappingIdentity>)> = None;
    for writeback in &batch.global_writebacks {
        match contained_global_writeback_offset(writeback, address, byte_count) {
            QueuedWriteReadMatch::None => {}
            QueuedWriteReadMatch::Synchronize => {
                return QueuedGlobalReadSource::Synchronize;
            }
            QueuedWriteReadMatch::Buffer {
                offset,
                physical_identity,
            } => {
                if !merge_global_read_candidate(
                    &mut candidate,
                    &writeback.buffer,
                    offset,
                    false,
                    physical_identity,
                    byte_count,
                ) {
                    return QueuedGlobalReadSource::Synchronize;
                }
            }
        }
    }
    for range in &batch.guest_write_ranges {
        match contained_guest_write_offset(range, address, byte_count) {
            QueuedWriteReadMatch::None => {}
            QueuedWriteReadMatch::Synchronize => {
                return QueuedGlobalReadSource::Synchronize;
            }
            QueuedWriteReadMatch::Buffer {
                offset,
                physical_identity,
            } => {
                if !merge_global_read_candidate(
                    &mut candidate,
                    &range.buffer,
                    offset,
                    true,
                    physical_identity,
                    byte_count,
                ) {
                    return QueuedGlobalReadSource::Synchronize;
                }
            }
        }
    }
    match candidate {
        Some((buffer, offset, guest_memory, physical_identity)) => QueuedGlobalReadSource::Buffer {
            buffer,
            offset,
            guest_memory,
            physical_identity,
        },
        None => QueuedGlobalReadSource::None,
    }
}

enum QueuedWriteReadMatch {
    None,
    Buffer {
        offset: u64,
        physical_identity: Option<GuestSharedMappingIdentity>,
    },
    Synchronize,
}

fn contained_guest_write_offset(
    range: &super::PendingGuestWriteRange,
    address: u64,
    byte_count: u64,
) -> QueuedWriteReadMatch {
    let virtual_overlap =
        super::guest_ranges_overlap(range.address, range.byte_count, address, byte_count);
    let Some(read_identity) = guest_shared_backing_identity(address, byte_count) else {
        return if virtual_overlap {
            QueuedWriteReadMatch::Synchronize
        } else {
            QueuedWriteReadMatch::None
        };
    };
    if read_identity.backing_id != range.physical_identity.backing_id {
        return if virtual_overlap {
            QueuedWriteReadMatch::Synchronize
        } else {
            QueuedWriteReadMatch::None
        };
    }
    let producer_start = range.physical_identity.backing_offset;
    let Some(producer_end) = producer_start.checked_add(range.byte_count) else {
        return QueuedWriteReadMatch::Synchronize;
    };
    let read_start = read_identity.backing_offset;
    let Some(read_end) = read_start.checked_add(byte_count) else {
        return QueuedWriteReadMatch::Synchronize;
    };
    if producer_start >= read_end || read_start >= producer_end {
        return if virtual_overlap {
            QueuedWriteReadMatch::Synchronize
        } else {
            QueuedWriteReadMatch::None
        };
    }
    if read_start < producer_start || read_end > producer_end {
        return QueuedWriteReadMatch::Synchronize;
    }
    let Some(offset) = range.offset.checked_add(read_start - producer_start) else {
        return QueuedWriteReadMatch::Synchronize;
    };
    let Some(buffer_end) = offset.checked_add(byte_count) else {
        return QueuedWriteReadMatch::Synchronize;
    };
    if buffer_end > range.buffer.length() {
        return QueuedWriteReadMatch::Synchronize;
    }
    QueuedWriteReadMatch::Buffer {
        offset,
        physical_identity: Some(read_identity),
    }
}

fn merge_global_read_candidate(
    candidate: &mut Option<(Buffer, u64, bool, Option<GuestSharedMappingIdentity>)>,
    buffer: &Buffer,
    offset: u64,
    guest_memory: bool,
    physical_identity: Option<GuestSharedMappingIdentity>,
    byte_count: u64,
) -> bool {
    let Some(end) = offset.checked_add(byte_count) else {
        return false;
    };
    if end > buffer.length() {
        return false;
    }
    match candidate {
        Some((
            existing_buffer,
            existing_offset,
            existing_guest_memory,
            existing_physical_identity,
        )) => {
            existing_buffer.as_ptr() == buffer.as_ptr()
                && *existing_offset == offset
                && *existing_guest_memory == guest_memory
                && *existing_physical_identity == physical_identity
        }
        None => {
            *candidate = Some((buffer.clone(), offset, guest_memory, physical_identity));
            true
        }
    }
}

fn contained_global_writeback_offset(
    writeback: &PendingGlobalWriteback,
    address: u64,
    byte_count: u64,
) -> QueuedWriteReadMatch {
    let Ok(writeback_byte_count) = u64::try_from(writeback.byte_count) else {
        return QueuedWriteReadMatch::Synchronize;
    };
    let virtual_overlap = super::guest_ranges_overlap(
        writeback.guest_address,
        writeback_byte_count,
        address,
        byte_count,
    );
    if let Some(producer_identity) = writeback.mapping_identity {
        let Some(read_identity) = guest_shared_backing_identity(address, byte_count) else {
            return if virtual_overlap {
                QueuedWriteReadMatch::Synchronize
            } else {
                QueuedWriteReadMatch::None
            };
        };
        if read_identity.backing_id != producer_identity.backing_id {
            return if virtual_overlap {
                QueuedWriteReadMatch::Synchronize
            } else {
                QueuedWriteReadMatch::None
            };
        }
        let producer_start = producer_identity.backing_offset;
        let Some(producer_end) = producer_start.checked_add(writeback_byte_count) else {
            return QueuedWriteReadMatch::Synchronize;
        };
        let read_start = read_identity.backing_offset;
        let Some(read_end) = read_start.checked_add(byte_count) else {
            return QueuedWriteReadMatch::Synchronize;
        };
        if producer_start >= read_end || read_start >= producer_end {
            return if virtual_overlap {
                QueuedWriteReadMatch::Synchronize
            } else {
                QueuedWriteReadMatch::None
            };
        }
        if read_start < producer_start || read_end > producer_end {
            return QueuedWriteReadMatch::Synchronize;
        }
        let Some(offset) = writeback.offset.checked_add(read_start - producer_start) else {
            return QueuedWriteReadMatch::Synchronize;
        };
        return QueuedWriteReadMatch::Buffer {
            offset,
            physical_identity: Some(read_identity),
        };
    }
    if !guest_physical_ranges_overlap(
        writeback.guest_address,
        writeback_byte_count,
        address,
        byte_count,
    ) {
        return QueuedWriteReadMatch::None;
    }
    let Some(writeback_end) = writeback.guest_address.checked_add(writeback_byte_count) else {
        return QueuedWriteReadMatch::Synchronize;
    };
    let Some(read_end) = address.checked_add(byte_count) else {
        return QueuedWriteReadMatch::Synchronize;
    };
    if address < writeback.guest_address || read_end > writeback_end {
        return QueuedWriteReadMatch::Synchronize;
    }
    let Some(offset) = writeback
        .offset
        .checked_add(address - writeback.guest_address)
    else {
        return QueuedWriteReadMatch::Synchronize;
    };
    QueuedWriteReadMatch::Buffer {
        offset,
        physical_identity: None,
    }
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

#[cfg(test)]
mod tests {
    use super::decode_indirect_dwords;

    #[test]
    fn indirect_dword_diagnostics_decode_complete_little_endian_words() {
        assert_eq!(
            decode_indirect_dwords(&[0x01, 0x02, 0x03, 0x04, 0x78, 0x56, 0x34, 0x12, 0xaa, 0xbb,]),
            [0x0403_0201, 0x1234_5678]
        );
    }
}
