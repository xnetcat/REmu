// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CachedGlobalBuffer, CachedNggMeshlets, CachedSamplerArguments, DecodedDepthTarget,
    GlobalBufferCacheKey, GlobalBufferOriginalVersion, GuestCommandBatch, MetalPresenter,
    NggSubgroupLimits, PendingGlobalWriteback, PhysicalFootprintRange,
    SAMPLER_ARGUMENT_CACHE_CAPACITY, SamplerArgumentCacheKey, SharedEventComputeGlobalBindingPlan,
    SharedEventDirectComputePlan, SharedEventGlobalBufferVersion, SharedEventSignalCheckpoint,
    UploadedGlobalBuffer, canonical_sampler_bindings, create_sampler_argument_buffer,
    describe_guest_batch, effective_global_memory_binding, guest_batch_writes_overlap,
    guest_physical_ranges_overlap, guest_ranges_overlap, metal_buffer_with_bytes,
    prepare_ngg_meshlets,
};
use block::ConcreteBlock;
use metal::foreign_types::ForeignType;
use metal::{
    Buffer, DepthStencilDescriptor, DepthStencilState, Device, Function, MTLCompareFunction,
    MTLResourceOptions, SamplerState, StencilDescriptor,
};
use remu_memory::{
    GuestSharedBackingRange, GuestSharedMappingIdentity, guest_shared_backing_identity,
    guest_shared_backing_any_range_overlaps, guest_shared_backing_range,
    read_guest_shared_backing,
};
use remu_shader::{GlobalMemoryBinding, ShaderEvaluation};
use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

pub(super) static UPLOAD_GLOBALS_CALLS: AtomicU64 = AtomicU64::new(0);
pub(super) static UPLOAD_GLOBALS_NANOS: AtomicU64 = AtomicU64::new(0);

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

fn trace_shared_compute_binding_reject(binding: &GlobalMemoryBinding, reason: &'static str) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| {
        std::env::var_os("REMU_TRACE_SHARED_EVENT_COMPUTE").is_some_and(|value| value == "1")
    }) {
        return;
    }
    static OCCURRENCES: AtomicUsize = AtomicUsize::new(0);
    let occurrence = OCCURRENCES.fetch_add(1, Ordering::Relaxed) + 1;
    if occurrence <= 64 || occurrence.is_power_of_two() {
        eprintln!(
            "[REMu][METAL][SHARED-COMPUTE-BINDING] occurrence={occurrence} address={:#x} bytes={} packet-bytes={} writable={} reason={reason}",
            binding.base_address,
            binding.byte_count,
            binding.data.len(),
            binding.writable,
        );
    }
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
            self.wait_for_guest_command_batches()?;
            self.ensure_active_open_guest_batch()?;
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
            .active_open_guest_batch()
            .filter(|batch| guest_batch_writes_overlap(batch, address, byte_count_u64))
            .map(|batch| ("open", 0, batch))
            .or_else(|| {
                self.active_committed_guest_batches()
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
            function: if buffer_index.is_some() {
                function.as_ptr() as usize
            } else {
                0
            },
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
        let stencil = depth.stencil;
        let key = super::DepthStencilCacheKey {
            compare: compare as u64,
            write_enabled,
            stencil_enabled: stencil.enabled,
            separate_back: stencil.separate_back,
            front_compare: stencil.front.compare as u64,
            front_fail: stencil.front.stencil_fail as u64,
            front_depth_fail: stencil.front.depth_fail as u64,
            front_depth_pass: stencil.front.depth_pass as u64,
            front_read_mask: stencil.front.read_mask,
            front_write_mask: stencil.front.write_mask,
            back_compare: stencil.back.compare as u64,
            back_fail: stencil.back.stencil_fail as u64,
            back_depth_fail: stencil.back.depth_fail as u64,
            back_depth_pass: stencil.back.depth_pass as u64,
            back_read_mask: stencil.back.read_mask,
            back_write_mask: stencil.back.write_mask,
        };
        if let Some(state) = self.depth_stencil_cache.get(&key) {
            return state.clone();
        }
        let descriptor = DepthStencilDescriptor::new();
        descriptor.set_depth_compare_function(compare);
        descriptor.set_depth_write_enabled(write_enabled);
        if stencil.enabled {
            let front = StencilDescriptor::new();
            front.set_stencil_compare_function(stencil.front.compare);
            front.set_stencil_failure_operation(stencil.front.stencil_fail);
            front.set_depth_failure_operation(stencil.front.depth_fail);
            front.set_depth_stencil_pass_operation(stencil.front.depth_pass);
            front.set_read_mask(stencil.front.read_mask);
            front.set_write_mask(stencil.front.write_mask);
            descriptor.set_front_face_stencil(Some(&front));
            let back = StencilDescriptor::new();
            back.set_stencil_compare_function(stencil.back.compare);
            back.set_stencil_failure_operation(stencil.back.stencil_fail);
            back.set_depth_failure_operation(stencil.back.depth_fail);
            back.set_depth_stencil_pass_operation(stencil.back.depth_pass);
            back.set_read_mask(stencil.back.read_mask);
            back.set_write_mask(stencil.back.write_mask);
            descriptor.set_back_face_stencil(Some(&back));
        }
        let state = device.new_depth_stencil_state(&descriptor);
        self.depth_stencil_cache.insert(key, state.clone());
        state
    }

    pub(super) fn upload_global_buffers(
        &mut self,
        device: &Device,
        evaluation: &ShaderEvaluation,
    ) -> Result<Vec<UploadedGlobalBuffer>, String> {
        let started = Instant::now();
        let uploaded = self.upload_global_buffers_untimed(device, evaluation);
        UPLOAD_GLOBALS_CALLS.fetch_add(1, Ordering::Relaxed);
        UPLOAD_GLOBALS_NANOS.fetch_add(
            u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        uploaded
    }

    fn upload_global_buffers_untimed(
        &mut self,
        device: &Device,
        evaluation: &ShaderEvaluation,
    ) -> Result<Vec<UploadedGlobalBuffer>, String> {
        // Alias null stubs to a live sibling on the same SGPR slot so draw-time Metal buffers and
        // remu_initial_s lengths stay coherent even if evaluation left both entries (glyph tofu).
        let effective_bindings = evaluation
            .global_memory_bindings
            .iter()
            .map(|binding| effective_global_memory_binding(evaluation, binding))
            .collect::<Vec<_>>();
        let sources = effective_bindings
            .iter()
            .map(|binding| self.queued_global_read_source(binding))
            .collect::<Result<Vec<_>, _>>()?;
        if sources
            .iter()
            .any(|source| matches!(source, QueuedGlobalReadSource::Synchronize))
        {
            self.wait_for_guest_command_batches()?;
            self.ensure_active_open_guest_batch()?;
            return effective_bindings
                .into_iter()
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
        effective_bindings
            .into_iter()
            .zip(sources)
            .map(|(binding, source)| self.upload_global_buffer_from_source(device, binding, source))
            .collect()
    }

    pub(super) fn upload_compute_global_buffers(
        &mut self,
        device: &Device,
        evaluation: &ShaderEvaluation,
        plan: Option<&SharedEventDirectComputePlan>,
    ) -> Result<Vec<UploadedGlobalBuffer>, String> {
        let Some(plan) = plan else {
            return self.upload_global_buffers(device, evaluation);
        };
        if plan.global_bindings.len() != evaluation.global_memory_bindings.len() {
            return Err("shared compute plan/global binding count mismatch".to_owned());
        }
        let effective_bindings = evaluation
            .global_memory_bindings
            .iter()
            .map(|binding| effective_global_memory_binding(evaluation, binding))
            .collect::<Vec<_>>();
        let mut group_destinations: Vec<Option<UploadedGlobalBuffer>> =
            vec![None; effective_bindings.len()];
        let mut uploaded = Vec::with_capacity(effective_bindings.len());
        for (index, binding_plan) in plan.global_bindings.iter().enumerate() {
            if let Some(group) = binding_plan.writable_group {
                if group >= effective_bindings.len() {
                    return Err("shared compute writable group is out of range".to_owned());
                }
                let destination = if let Some(destination) = &group_destinations[group] {
                    destination.clone()
                } else {
                    let owner = &plan.global_bindings[group];
                    let destination = if let Some(source) = &owner.source {
                        self.upload_writable_global_buffer_from_gpu_version(
                            device,
                            effective_bindings[group],
                            source.clone(),
                            owner.source_offset,
                            Some(owner.physical_identity),
                        )?
                    } else if let Some(backing) = &owner.guest_backing {
                        self.upload_writable_global_buffer_from_guest_backing(
                            device,
                            effective_bindings[group],
                            backing,
                            owner.physical_identity,
                        )?
                    } else {
                        let binding = effective_bindings[group];
                        if binding.data.len() != binding.byte_count {
                            return Err("shared compute packet snapshot is incomplete".to_owned());
                        }
                        self.new_exact_writable_global_buffer(device, binding, &binding.data)?
                    };
                    group_destinations[group] = Some(destination.clone());
                    destination
                };
                uploaded.push(destination);
            } else if let Some(source) = &binding_plan.source {
                uploaded.push(UploadedGlobalBuffer {
                    buffer: source.clone(),
                    offset: binding_plan.source_offset,
                    guest_memory: false,
                    physical_identity: Some(binding_plan.physical_identity),
                    original_version: None,
                });
            } else if let Some(backing) = &binding_plan.guest_backing {
                uploaded.push(self.upload_shared_event_guest_global_buffer(
                    device,
                    effective_bindings[index],
                    backing,
                    binding_plan.physical_identity,
                )?);
            } else {
                uploaded.push(
                    self.upload_global_buffer_from_packet(device, effective_bindings[index])?,
                );
            }
        }
        Ok(uploaded)
    }

    #[cfg(test)]
    pub(super) fn upload_global_buffer(
        &mut self,
        device: &Device,
        binding: &GlobalMemoryBinding,
    ) -> Result<UploadedGlobalBuffer, String> {
        let source = self.queued_global_read_source(binding)?;
        if matches!(source, QueuedGlobalReadSource::Synchronize) {
            self.wait_for_guest_command_batches()?;
            self.ensure_active_open_guest_batch()?;
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
        if binding.writable {
            return match source {
                QueuedGlobalReadSource::Buffer {
                    buffer,
                    offset,
                    physical_identity,
                    ..
                } => self.upload_writable_global_buffer_from_gpu_version(
                    device,
                    binding,
                    buffer,
                    offset,
                    physical_identity,
                ),
                QueuedGlobalReadSource::Synchronize => {
                    unreachable!("synchronizing global sources are handled before upload")
                }
                QueuedGlobalReadSource::None => {
                    self.upload_writable_global_buffer_from_packet(device, binding)
                }
            };
        }
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
                    original_version: None,
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
        if binding.writable {
            return self.upload_writable_global_buffer_from_packet(device, binding);
        }
        if binding.data.len() != binding.byte_count
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
        // Full packet snapshot must win over live guest identity (producer may have recycled the
        // VA). Phase 1G deletes the host byte-compare cache: upload the packet snapshot fresh
        // into the open batch every time instead of retaining VA-keyed host copies.
        let (buffer, offset) = self
            .active_open_guest_batch_mut()
            .ok_or("captured global upload has no open guest batch")?
            .upload_global_bytes(device, &binding.data)?;
        Ok(UploadedGlobalBuffer {
            buffer,
            offset,
            guest_memory: false,
            physical_identity: None,
            original_version: None,
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
            original_version: None,
        }))
    }

    fn upload_writable_global_buffer_from_packet(
        &mut self,
        device: &Device,
        binding: &GlobalMemoryBinding,
    ) -> Result<UploadedGlobalBuffer, String> {
        if binding.byte_count == 0 {
            return Err("writable guest global upload is empty".to_owned());
        }
        let mut bytes = vec![0; binding.byte_count];
        if !read_guest_shared_backing(binding.base_address, &mut bytes) {
            if binding.data.len() != binding.byte_count {
                return Err(format!(
                    "writable global binding {:#x}/{} has neither readable guest backing nor a complete packet snapshot",
                    binding.base_address, binding.byte_count
                ));
            }
            bytes.copy_from_slice(&binding.data);
        }
        self.new_exact_writable_global_buffer(device, binding, &bytes)
    }

    /// Forks an immutable queued producer version into writable consumer storage in GPU order.
    ///
    /// The active command buffer already contains any registered SharedEvent wait selected by
    /// the scheduler. Closing its active encoder and appending this blit therefore establishes
    /// `wait -> baseline copy -> consumer compute` without observing producer bytes on the CPU.
    /// The source remains immutable and is retained as the completion-time diff baseline.
    fn upload_writable_global_buffer_from_gpu_version(
        &mut self,
        device: &Device,
        binding: &GlobalMemoryBinding,
        source: Buffer,
        source_offset: u64,
        physical_identity: Option<GuestSharedMappingIdentity>,
    ) -> Result<UploadedGlobalBuffer, String> {
        if binding.byte_count == 0 {
            return Err("queued writable global source is empty".to_owned());
        }
        let byte_count = u64::try_from(binding.byte_count)
            .map_err(|_| "queued writable global byte count does not fit Metal")?;
        if source_offset
            .checked_add(byte_count)
            .is_none_or(|end| end > source.length())
        {
            return Err(format!(
                "queued writable global source exceeds Metal buffer offset={source_offset} bytes={byte_count} length={}",
                source.length(),
            ));
        }
        let destination = device.new_buffer(byte_count, MTLResourceOptions::StorageModeShared);
        destination.set_label(&format!(
            "REMu COW writable global {:#x}/{byte_count:#x}",
            binding.base_address,
        ));
        self.ensure_active_open_guest_batch()?;
        let batch = self
            .active_open_guest_batch_mut()
            .expect("queued writable global COW opened a guest batch");
        batch.close_active_pass_encoders();
        let blit = batch.command_buffer.new_blit_command_encoder();
        blit.set_label("REMu copy retained global baseline");
        blit.copy_from_buffer(&source, source_offset, &destination, 0, byte_count);
        blit.end_encoding();

        static COW_COUNT: AtomicUsize = AtomicUsize::new(0);
        let sequence = COW_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if sequence <= 16 || sequence.is_power_of_two() {
            eprintln!(
                "[REMu][METAL][GLOBAL-COW] sequence={sequence} address={:#x} bytes={byte_count:#x} source={:p}+{source_offset:#x} destination={:p} physical={physical_identity:?}",
                binding.base_address,
                source.as_ptr(),
                destination.as_ptr(),
            );
        }
        Ok(UploadedGlobalBuffer {
            buffer: destination,
            offset: 0,
            guest_memory: false,
            physical_identity,
            original_version: Some(GlobalBufferOriginalVersion {
                buffer: source,
                offset: source_offset,
                _retained_source: None,
            }),
        })
    }

    /// Captures a retained physical guest range after the active SharedEvent wait entirely on
    /// the GPU, then forks that immutable baseline into writable compute storage.
    ///
    /// The first blit observes the source in guest queue order. The second copies only from the
    /// immutable baseline, so completion can diff the compute result even if guest aliases later
    /// change. Retaining the no-copy source explicitly keeps its independent mmap alive until the
    /// command buffer reaches a terminal state.
    fn upload_writable_global_buffer_from_guest_backing(
        &mut self,
        device: &Device,
        binding: &GlobalMemoryBinding,
        backing: &GuestSharedBackingRange,
        physical_identity: GuestSharedMappingIdentity,
    ) -> Result<UploadedGlobalBuffer, String> {
        if binding.byte_count == 0 {
            return Err("retained guest writable global source is empty".to_owned());
        }
        let source = self.upload_shared_event_guest_global_buffer(
            device,
            binding,
            backing,
            physical_identity,
        )?;
        let byte_count = u64::try_from(binding.byte_count)
            .map_err(|_| "retained guest writable global byte count does not fit Metal")?;
        let baseline = device.new_buffer(byte_count, MTLResourceOptions::StorageModeShared);
        baseline.set_label(&format!(
            "REMu retained guest global baseline {:#x}/{byte_count:#x}",
            binding.base_address,
        ));
        let destination = device.new_buffer(byte_count, MTLResourceOptions::StorageModeShared);
        destination.set_label(&format!(
            "REMu retained guest writable global {:#x}/{byte_count:#x}",
            binding.base_address,
        ));

        self.ensure_active_open_guest_batch()?;
        let batch = self
            .active_open_guest_batch_mut()
            .expect("retained guest global COW opened a guest batch");
        batch.close_active_pass_encoders();
        let blit = batch.command_buffer.new_blit_command_encoder();
        blit.set_label("REMu snapshot retained guest global baseline");
        blit.copy_from_buffer(&source.buffer, source.offset, &baseline, 0, byte_count);
        blit.copy_from_buffer(&baseline, 0, &destination, 0, byte_count);
        blit.end_encoding();

        static GUEST_COW_COUNT: AtomicUsize = AtomicUsize::new(0);
        let sequence = GUEST_COW_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
        if sequence <= 16 || sequence.is_power_of_two() {
            eprintln!(
                "[REMu][METAL][GLOBAL-GUEST-COW] sequence={sequence} address={:#x} bytes={byte_count:#x} source={:p}+{:#x} baseline={:p} destination={:p} physical={physical_identity:?}",
                binding.base_address,
                source.buffer.as_ptr(),
                source.offset,
                baseline.as_ptr(),
                destination.as_ptr(),
            );
        }
        Ok(UploadedGlobalBuffer {
            buffer: destination,
            offset: 0,
            guest_memory: false,
            physical_identity: Some(physical_identity),
            original_version: Some(GlobalBufferOriginalVersion {
                buffer: baseline,
                offset: 0,
                _retained_source: Some(source.buffer),
            }),
        })
    }

    fn upload_shared_event_guest_global_buffer(
        &self,
        device: &Device,
        binding: &GlobalMemoryBinding,
        backing: &GuestSharedBackingRange,
        physical_identity: GuestSharedMappingIdentity,
    ) -> Result<UploadedGlobalBuffer, String> {
        let byte_count = u64::try_from(binding.byte_count)
            .map_err(|_| "retained guest global byte count does not fit Metal")?;
        let resolved_identity = GuestSharedMappingIdentity {
            backing_id: backing.view_identity().backing_id,
            backing_offset: backing
                .view_identity()
                .backing_offset
                .checked_add(backing.resource_offset())
                .ok_or("retained guest global physical identity overflows")?,
        };
        if resolved_identity != physical_identity {
            return Err(format!(
                "retained guest global identity changed expected={physical_identity:?} resolved={resolved_identity:?}"
            ));
        }
        let buffer = metal_buffer_with_guest_backing_range(device, backing)
            .ok_or("could not map retained guest global backing for Metal")?;
        let end = backing
            .resource_offset()
            .checked_add(byte_count)
            .ok_or("retained guest global range overflows")?;
        if end > buffer.length() {
            return Err(format!(
                "retained guest global exceeds Metal buffer offset={} bytes={byte_count} length={}",
                backing.resource_offset(),
                buffer.length(),
            ));
        }
        buffer.set_label(&format!(
            "REMu retained guest global backing={:#x}+{:#x}/{byte_count:#x}",
            physical_identity.backing_id, physical_identity.backing_offset,
        ));
        Ok(UploadedGlobalBuffer {
            buffer,
            offset: backing.resource_offset(),
            guest_memory: true,
            physical_identity: Some(physical_identity),
            original_version: None,
        })
    }

    fn new_exact_writable_global_buffer(
        &self,
        device: &Device,
        binding: &GlobalMemoryBinding,
        bytes: &[u8],
    ) -> Result<UploadedGlobalBuffer, String> {
        if bytes.len() != binding.byte_count {
            return Err(format!(
                "writable global snapshot length mismatch bytes={} snapshot={}",
                binding.byte_count,
                bytes.len()
            ));
        }
        let byte_count = u64::try_from(binding.byte_count)
            .map_err(|_| "writable global byte count does not fit Metal")?;
        let buffer = metal_buffer_with_bytes(device, bytes)?;
        buffer.set_label(&format!(
            "REMu exact writable global {:#x}/{:#x}",
            binding.base_address, byte_count
        ));
        Ok(UploadedGlobalBuffer {
            buffer,
            offset: 0,
            guest_memory: false,
            physical_identity: guest_shared_backing_identity(binding.base_address, byte_count),
            original_version: None,
        })
    }

    fn upload_fresh_global_buffer(
        &mut self,
        device: &Device,
        binding: &GlobalMemoryBinding,
    ) -> Result<UploadedGlobalBuffer, String> {
        if binding.writable {
            return self.upload_writable_global_buffer_from_packet(device, binding);
        }
        let (buffer, offset) = self
            .active_open_guest_batch_mut()
            .ok_or("synchronized global upload has no open guest batch")?
            .upload_guest_global_bytes(device, binding.base_address, binding.byte_count)?;
        Ok(UploadedGlobalBuffer {
            buffer,
            offset,
            guest_memory: false,
            physical_identity: None,
            original_version: None,
        })
    }

    fn queued_global_read_source(
        &self,
        binding: &GlobalMemoryBinding,
    ) -> Result<QueuedGlobalReadSource, String> {
        let byte_count = u64::try_from(binding.byte_count)
            .map_err(|_| "global binding byte count does not fit u64")?;
        if let Some(batch) = self.active_open_guest_batch() {
            let source =
                queued_global_read_source_in_batch(batch, binding.base_address, byte_count);
            if !matches!(source, QueuedGlobalReadSource::None) {
                return Ok(stageable_writable_global_source(binding, source));
            }
        }
        for batch in self.active_committed_guest_batches().iter().rev() {
            let source =
                queued_global_read_source_in_batch(batch, binding.base_address, byte_count);
            if !matches!(source, QueuedGlobalReadSource::None) {
                return Ok(stageable_writable_global_source(binding, source));
            }
        }
        if !self.per_queue_guest_batches {
            return Ok(QueuedGlobalReadSource::None);
        }
        let Some(consumer_queue_id) = self.active_guest_queue else {
            return Ok(QueuedGlobalReadSource::None);
        };
        let selected = self.shared_event_dependency_global_read_source(
            consumer_queue_id,
            binding.base_address,
            byte_count,
        );
        Ok(stageable_writable_global_source(binding, selected))
    }

    fn shared_event_dependency_global_read_source(
        &self,
        consumer_queue_id: u64,
        address: u64,
        byte_count: u64,
    ) -> QueuedGlobalReadSource {
        let Some(consumer) = self.guest_queue_batches.get(&consumer_queue_id) else {
            return QueuedGlobalReadSource::None;
        };
        let mut selected = QueuedGlobalReadSource::None;
        for (&producer, checkpoint) in &consumer.resource_dependencies {
            if producer.queue_id() == consumer_queue_id {
                continue;
            }
            let source = queued_global_read_source_in_checkpoint(checkpoint, address, byte_count);
            if !merge_queued_global_read_source(&mut selected, source, byte_count) {
                return QueuedGlobalReadSource::Synchronize;
            }
        }
        selected
    }

    /// True when a draw can consume this range without observing guest RAM or forcing a host
    /// retirement. The producer checkpoint owns the exact copied Metal buffer version at its
    /// signal; the consumer command buffer's already-encoded SharedEvent wait orders the read.
    /// Any local queued writer or ambiguous producer version keeps the draw on the host gate.
    pub(super) fn shared_event_dependencies_supply_draw_global_range(
        &self,
        consumer_queue_id: u64,
        address: u64,
        byte_count: u64,
    ) -> bool {
        let Some(consumer) = self.guest_queue_batches.get(&consumer_queue_id) else {
            return false;
        };
        if consumer.open_batch.as_ref().is_some_and(|batch| {
            !matches!(
                queued_global_read_source_in_batch(batch, address, byte_count),
                QueuedGlobalReadSource::None
            )
        }) || consumer.committed_batches.iter().any(|batch| {
            !matches!(
                queued_global_read_source_in_batch(batch, address, byte_count),
                QueuedGlobalReadSource::None
            )
        }) {
            return false;
        }
        matches!(
            self.shared_event_dependency_global_read_source(
                consumer_queue_id,
                address,
                byte_count,
            ),
            QueuedGlobalReadSource::Buffer {
                guest_memory: false,
                physical_identity: Some(_),
                ..
            }
        )
    }

    /// A direct compute dispatch can cross a GPU wait only when every global input is an exact
    /// immutable version retained by that wait and no earlier local writer competes with it.
    #[cfg(test)]
    pub(super) fn shared_event_dependencies_supply_compute_globals(
        &self,
        evaluation: &ShaderEvaluation,
    ) -> bool {
        let Some(consumer_queue_id) = self.active_guest_queue else {
            return false;
        };
        evaluation.global_memory_bindings.iter().all(|binding| {
            let binding = effective_global_memory_binding(evaluation, binding);
            if binding.writable {
                return false;
            }
            let Ok(byte_count) = u64::try_from(binding.byte_count) else {
                return false;
            };
            if self.active_open_guest_batch().is_some_and(|batch| {
                !matches!(
                    queued_global_read_source_in_batch(batch, binding.base_address, byte_count),
                    QueuedGlobalReadSource::None
                )
            }) || self.active_committed_guest_batches().iter().any(|batch| {
                !matches!(
                    queued_global_read_source_in_batch(batch, binding.base_address, byte_count),
                    QueuedGlobalReadSource::None
                )
            }) {
                return false;
            }
            matches!(
                self.shared_event_dependency_global_read_source(
                    consumer_queue_id,
                    binding.base_address,
                    byte_count,
                ),
                QueuedGlobalReadSource::Buffer {
                    guest_memory: false,
                    ..
                }
            )
        })
    }

    pub(super) fn shared_event_compute_global_binding_plan(
        &self,
        evaluation: &ShaderEvaluation,
        dependency_write_footprint: &[PhysicalFootprintRange],
    ) -> Option<Vec<SharedEventComputeGlobalBindingPlan>> {
        let effective_bindings = evaluation
            .global_memory_bindings
            .iter()
            .map(|binding| effective_global_memory_binding(evaluation, binding))
            .collect::<Vec<_>>();
        let mut plans = Vec::with_capacity(effective_bindings.len());
        for binding in &effective_bindings {
            if binding.base_address == 0 || binding.byte_count == 0 {
                trace_shared_compute_binding_reject(binding, "invalid-range");
                return None;
            }
            let Some(byte_count) = u64::try_from(binding.byte_count).ok() else {
                trace_shared_compute_binding_reject(binding, "invalid-byte-count");
                return None;
            };
            let Some(binding_identity) =
                guest_shared_backing_identity(binding.base_address, byte_count)
            else {
                trace_shared_compute_binding_reject(binding, "missing-identity");
                return None;
            };
            let source = match self.queued_global_read_source(binding) {
                Ok(source) => source,
                Err(_) => {
                    trace_shared_compute_binding_reject(binding, "source-error");
                    return None;
                }
            };
            let (source, offset, guest_backing, physical_identity) = match source {
                QueuedGlobalReadSource::Buffer {
                    buffer,
                    offset,
                    guest_memory: false,
                    physical_identity: Some(physical_identity),
                } => {
                    if offset
                        .checked_add(byte_count)
                        .is_none_or(|end| end > buffer.length())
                    {
                        trace_shared_compute_binding_reject(
                            binding,
                            "retained-source-out-of-bounds",
                        );
                        return None;
                    }
                    (Some(buffer), offset, None, physical_identity)
                }
                QueuedGlobalReadSource::None => {
                    if dependency_write_footprint.iter().any(|producer| {
                        producer.mapping.backing_id == binding_identity.backing_id
                            && producer.mapping.backing_offset
                                < binding_identity.backing_offset.saturating_add(byte_count)
                            && binding_identity.backing_offset
                                < producer
                                    .mapping
                                    .backing_offset
                                    .saturating_add(producer.byte_count)
                    }) {
                        trace_shared_compute_binding_reject(
                            binding,
                            "dependency-overlap-without-version",
                        );
                        return None;
                    }
                    if binding.data.len() == binding.byte_count {
                        (None, 0, None, binding_identity)
                    } else {
                        let Some(backing) =
                            guest_shared_backing_range(binding.base_address, byte_count)
                        else {
                            trace_shared_compute_binding_reject(
                                binding,
                                "incomplete-packet-snapshot",
                            );
                            return None;
                        };
                        if backing.is_virtually_keyed() {
                            trace_shared_compute_binding_reject(
                                binding,
                                "virtually-keyed-guest-source",
                            );
                            return None;
                        }
                        let resolved_identity = GuestSharedMappingIdentity {
                            backing_id: backing.view_identity().backing_id,
                            backing_offset: backing
                                .view_identity()
                                .backing_offset
                                .checked_add(backing.resource_offset())?,
                        };
                        if resolved_identity != binding_identity {
                            trace_shared_compute_binding_reject(
                                binding,
                                "guest-source-identity-mismatch",
                            );
                            return None;
                        }
                        (None, 0, Some(backing), binding_identity)
                    }
                }
                QueuedGlobalReadSource::Buffer {
                    guest_memory: true, ..
                } => {
                    trace_shared_compute_binding_reject(binding, "zero-copy-source");
                    return None;
                }
                QueuedGlobalReadSource::Buffer {
                    physical_identity: None,
                    ..
                } => {
                    trace_shared_compute_binding_reject(
                        binding,
                        "retained-source-missing-identity",
                    );
                    return None;
                }
                QueuedGlobalReadSource::Synchronize => {
                    trace_shared_compute_binding_reject(binding, "source-requires-synchronization");
                    return None;
                }
            };
            plans.push(SharedEventComputeGlobalBindingPlan {
                source,
                source_offset: offset,
                guest_backing,
                physical_identity,
                byte_count,
                writable_group: None,
            });
        }

        for left in 0..plans.len() {
            for right in (left + 1)..plans.len() {
                if plans[left].physical_identity.backing_id
                    != plans[right].physical_identity.backing_id
                {
                    continue;
                }
                let left_start = plans[left].physical_identity.backing_offset;
                let right_start = plans[right].physical_identity.backing_offset;
                let left_end = left_start.checked_add(plans[left].byte_count)?;
                let right_end = right_start.checked_add(plans[right].byte_count)?;
                if left_start >= right_end || right_start >= left_end {
                    continue;
                }
                let exact = left_start == right_start
                    && plans[left].byte_count == plans[right].byte_count
                    && match (&plans[left].source, &plans[right].source) {
                        (Some(left_buffer), Some(right_buffer)) => {
                            left_buffer.as_ptr() == right_buffer.as_ptr()
                                && plans[left].guest_backing.is_none()
                                && plans[right].guest_backing.is_none()
                        }
                        (None, None) => {
                            match (&plans[left].guest_backing, &plans[right].guest_backing) {
                                (None, None) => true,
                                (Some(left), Some(right)) => {
                                    left.view_identity() == right.view_identity()
                                        && left.view_byte_count() == right.view_byte_count()
                                        && left.resource_offset() == right.resource_offset()
                                }
                                _ => false,
                            }
                        }
                        _ => false,
                    }
                    && plans[left].source_offset == plans[right].source_offset;
                if !exact {
                    trace_shared_compute_binding_reject(
                        effective_bindings[right],
                        "partial-or-mixed-alias",
                    );
                    return None;
                }
            }
        }
        for index in 0..plans.len() {
            let start = plans[index].physical_identity.backing_offset;
            let byte_count = plans[index].byte_count;
            plans[index].writable_group = plans
                .iter()
                .enumerate()
                .find(|(candidate, plan)| {
                    effective_bindings[*candidate].writable
                        && plan.physical_identity.backing_id
                            == plans[index].physical_identity.backing_id
                        && plan.physical_identity.backing_offset == start
                        && plan.byte_count == byte_count
                })
                .map(|(candidate, _)| candidate);
        }
        Some(plans)
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

fn stageable_writable_global_source(
    binding: &GlobalMemoryBinding,
    source: QueuedGlobalReadSource,
) -> QueuedGlobalReadSource {
    if binding.writable
        && matches!(
            source,
            QueuedGlobalReadSource::Buffer {
                physical_identity: None,
                ..
            }
        )
    {
        QueuedGlobalReadSource::Synchronize
    } else {
        source
    }
}

fn queued_global_read_source_in_batch(
    batch: &GuestCommandBatch,
    address: u64,
    byte_count: u64,
) -> QueuedGlobalReadSource {
    queued_global_read_source_in_batch_prefix(
        batch,
        batch.global_writebacks.len(),
        batch.guest_write_ranges.len(),
        batch.memory_writes.len(),
        address,
        byte_count,
    )
}

fn queued_global_read_source_in_batch_prefix(
    batch: &GuestCommandBatch,
    global_writeback_count: usize,
    guest_write_range_count: usize,
    memory_write_count: usize,
    address: u64,
    byte_count: u64,
) -> QueuedGlobalReadSource {
    let global_versions = &batch.global_buffer_versions;
    let global_writebacks = &batch.global_writebacks[..global_writeback_count];
    let guest_write_ranges = &batch.guest_write_ranges[..guest_write_range_count];
    let memory_writes = &batch.memory_writes[..memory_write_count];
    // Every write in the batch used to be tested with `guest_physical_ranges_overlap`, which takes
    // the shared-mapping registry lock and resolves both sides per candidate. That is one lock and
    // two resolutions for each of the four collections' entries, per binding, per queued batch —
    // the cost the presenter livelocks in once the submission backlog grows.
    //
    // `∃c. virtual(c) ∨ physical(c)` is `(∃c. virtual(c)) ∨ (∃c. physical(c))`, so the cheap virtual
    // pass runs first over everything and the registry is touched only when nothing matched
    // virtually — then once, for the whole scan, rather than once per candidate.
    let physical_candidates = || {
        global_versions
            .iter()
            .map(|version| (version.guest_address, version.byte_count))
            .chain(global_writebacks.iter().filter_map(|writeback| {
                u64::try_from(writeback.byte_count)
                    .ok()
                    .map(|length| (writeback.guest_address, length))
            }))
            .chain(
                guest_write_ranges
                    .iter()
                    .map(|range| (range.address, range.byte_count)),
            )
    };
    let overlaps = memory_writes
        .iter()
        .any(|write| write.overlaps(address, byte_count))
        || physical_candidates().any(|(candidate, candidate_byte_count)| {
            guest_ranges_overlap(candidate, candidate_byte_count, address, byte_count)
        })
        || guest_shared_backing_any_range_overlaps(physical_candidates(), address, byte_count);
    if !overlaps {
        return QueuedGlobalReadSource::None;
    }
    if memory_writes.iter().any(|write| {
        let (write_address, write_byte_count) = write.range();
        guest_physical_ranges_overlap(write_address, write_byte_count, address, byte_count)
    }) {
        return QueuedGlobalReadSource::Synchronize;
    }
    let mut candidate: Option<(Buffer, u64, bool, Option<GuestSharedMappingIdentity>)> = None;
    for version in global_versions {
        match contained_global_version_offset(version, address, byte_count) {
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
                    &version.buffer,
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
    if let Some((buffer, offset, guest_memory, physical_identity)) = candidate.take() {
        // The first-class version journal is appended only after the producing native command.
        // Its full-range COW baseline therefore subsumes older writeback/zero-copy producers for
        // any contained read. Memory writes were checked above because they are terminal host
        // effects and cannot be represented by this immutable Metal version.
        return QueuedGlobalReadSource::Buffer {
            buffer,
            offset,
            guest_memory,
            physical_identity,
        };
    }
    for writeback in global_writebacks {
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
    for range in guest_write_ranges {
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

fn queued_global_read_source_in_checkpoint(
    checkpoint: &SharedEventSignalCheckpoint,
    address: u64,
    byte_count: u64,
) -> QueuedGlobalReadSource {
    if checkpoint
        .unresolved_write_ranges
        .iter()
        .any(|write| unresolved_checkpoint_range_overlaps(write, address, byte_count))
    {
        return QueuedGlobalReadSource::Synchronize;
    }
    let mut candidate = None;
    for version in &checkpoint.global_versions {
        match contained_global_version_offset(version, address, byte_count) {
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
                    &version.buffer,
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

fn unresolved_checkpoint_range_overlaps(
    write: &super::SharedEventUnresolvedWriteRange,
    address: u64,
    byte_count: u64,
) -> bool {
    if super::guest_ranges_overlap(write.guest_address, write.byte_count, address, byte_count) {
        return true;
    }
    let Some(write_identity) = write.mapping_identity else {
        return guest_physical_ranges_overlap(
            write.guest_address,
            write.byte_count,
            address,
            byte_count,
        );
    };
    let Some(read_identity) = guest_shared_backing_identity(address, byte_count) else {
        return false;
    };
    write_identity.backing_id == read_identity.backing_id
        && super::guest_ranges_overlap(
            write_identity.backing_offset,
            write.byte_count,
            read_identity.backing_offset,
            byte_count,
        )
}

fn merge_queued_global_read_source(
    selected: &mut QueuedGlobalReadSource,
    source: QueuedGlobalReadSource,
    byte_count: u64,
) -> bool {
    match source {
        QueuedGlobalReadSource::None => true,
        QueuedGlobalReadSource::Synchronize => false,
        QueuedGlobalReadSource::Buffer {
            buffer,
            offset,
            guest_memory,
            physical_identity,
        } => match selected {
            QueuedGlobalReadSource::None => {
                *selected = QueuedGlobalReadSource::Buffer {
                    buffer,
                    offset,
                    guest_memory,
                    physical_identity,
                };
                true
            }
            QueuedGlobalReadSource::Buffer {
                buffer: existing,
                offset: existing_offset,
                guest_memory: existing_guest_memory,
                physical_identity: existing_identity,
            } => {
                let Some(end) = offset.checked_add(byte_count) else {
                    return false;
                };
                end <= buffer.length()
                    && existing.as_ptr() == buffer.as_ptr()
                    && *existing_offset == offset
                    && *existing_guest_memory == guest_memory
                    && *existing_identity == physical_identity
            }
            QueuedGlobalReadSource::Synchronize => false,
        },
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

fn contained_global_version_offset(
    version: &SharedEventGlobalBufferVersion,
    address: u64,
    byte_count: u64,
) -> QueuedWriteReadMatch {
    let virtual_overlap = super::guest_ranges_overlap(
        version.guest_address,
        version.byte_count,
        address,
        byte_count,
    );
    if let Some(producer_identity) = version.mapping_identity {
        if let Some(read_identity) = guest_shared_backing_identity(address, byte_count)
            && read_identity.backing_id == producer_identity.backing_id
        {
            let producer_start = producer_identity.backing_offset;
            if let (Some(producer_end), Some(read_end)) = (
                producer_start.checked_add(version.byte_count),
                read_identity.backing_offset.checked_add(byte_count),
            ) {
                let read_start = read_identity.backing_offset;
                if producer_start < read_end && read_start < producer_end {
                    if read_start >= producer_start
                        && read_end <= producer_end
                        && let Some(offset) =
                            version.offset.checked_add(read_start - producer_start)
                        && offset
                            .checked_add(byte_count)
                            .is_some_and(|end| end <= version.buffer.length())
                    {
                        return QueuedWriteReadMatch::Buffer {
                            offset,
                            physical_identity: Some(read_identity),
                        };
                    }
                    return QueuedWriteReadMatch::Synchronize;
                }
            }
        }
        if !virtual_overlap {
            return QueuedWriteReadMatch::None;
        }
    } else if !guest_physical_ranges_overlap(
        version.guest_address,
        version.byte_count,
        address,
        byte_count,
    ) {
        return QueuedWriteReadMatch::None;
    }
    let Some(version_end) = version.guest_address.checked_add(version.byte_count) else {
        return QueuedWriteReadMatch::Synchronize;
    };
    let Some(read_end) = address.checked_add(byte_count) else {
        return QueuedWriteReadMatch::Synchronize;
    };
    if address < version.guest_address || read_end > version_end {
        return QueuedWriteReadMatch::Synchronize;
    }
    let Some(offset) = version.offset.checked_add(address - version.guest_address) else {
        return QueuedWriteReadMatch::Synchronize;
    };
    if offset
        .checked_add(byte_count)
        .is_none_or(|end| end > version.buffer.length())
    {
        return QueuedWriteReadMatch::Synchronize;
    }
    QueuedWriteReadMatch::Buffer {
        offset,
        physical_identity: version.mapping_identity,
    }
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
    if let Some(producer_identity) = writeback.mapping_identity() {
        if let Some(read_identity) = guest_shared_backing_identity(address, byte_count)
            && read_identity.backing_id == producer_identity.backing_id
        {
            let producer_start = producer_identity.backing_offset;
            if let (Some(producer_end), Some(read_end)) = (
                producer_start.checked_add(writeback_byte_count),
                read_identity.backing_offset.checked_add(byte_count),
            ) {
                let read_start = read_identity.backing_offset;
                if producer_start < read_end && read_start < producer_end {
                    if read_start >= producer_start
                        && read_end <= producer_end
                        && let Some(offset) =
                            writeback.offset.checked_add(read_start - producer_start)
                    {
                        return QueuedWriteReadMatch::Buffer {
                            offset,
                            physical_identity: Some(read_identity),
                        };
                    }
                    return QueuedWriteReadMatch::Synchronize;
                }
            }
        }
        // A recycled VA names a new allocation once its physical identity changes. Forwarding the
        // producer buffer by virtual offset would hand stale bytes to that new owner. Retire the
        // old producer, then snapshot the replacement mapping.
        return if virtual_overlap {
            QueuedWriteReadMatch::Synchronize
        } else {
            QueuedWriteReadMatch::None
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
    contained_global_writeback_virtual_offset(writeback, address, byte_count, writeback_byte_count)
}

fn contained_global_writeback_virtual_offset(
    writeback: &PendingGlobalWriteback,
    address: u64,
    byte_count: u64,
    writeback_byte_count: u64,
) -> QueuedWriteReadMatch {
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
        physical_identity: writeback.mapping_identity(),
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
