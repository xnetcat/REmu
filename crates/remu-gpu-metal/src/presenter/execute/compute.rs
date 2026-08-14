// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::super::{ComputePipelineKey, SharedEventDirectComputePlan};
use super::WrittenRegion;
use super::{
    FlushPolicy, GUEST_BATCH_WORK_CAPACITY, GuestBatchFlushReason, GuestBatchWork, MetalPresenter,
    ResidencyState, ResidentTextureActivityStamp, SpanQueryOrder, StaleSource,
    append_unique_global_writebacks, append_unique_guest_write_ranges,
    collect_pending_global_writebacks, collect_pending_guest_write_ranges,
    collect_shared_event_global_versions, compute_program_can_split_wave64,
    create_global_argument_buffer, decode_image_descriptor,
    diagnostic_compute_texture_dispatch_is_eligible,
    diagnostic_dump_compute_texture_minimum_dispatch, diagnostic_overlapping_texture_writers,
    diagnostic_target_after_mrt_ready, diagnostic_texture_numeric_range, diagnostic_texture_writer,
    diagnostic_texture_writer_enabled, dump_msl_compute_filter_matches,
    image_resource_descriptor_is_null, is_storage_image_opcode, is_writable_image_opcode,
    metal_uncompressed_bytes_per_pixel, read_texture_linear,
    rearm_color_metadata_clear,
    rearm_depth_metadata_clear, record_diagnostic_global_writers, record_diagnostic_texture_writer,
    resident_color_view_sample_is_eligible, resident_depth_sample_is_eligible,
    resident_offset_alias_sample_is_eligible, resident_same_image_sample_is_eligible,
    resident_volume_sample_is_eligible,
    shader_scalar_metadata_for_registers,
    storage_image_requires_mip_hierarchy, trace_target_sampling, upload_image_textures,
    log_wave64_hazard_census, validate_compute_launch_for_metal, wave64_split_blockers,
};
use metal::{MTLPixelFormat, MTLResourceOptions, MTLResourceUsage, MTLSize};
use remu_gpu::{ComputeDispatch, ComputeGrid};
use std::ffi::c_void;
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const DIAGNOSTIC_COMPUTE_TRACE_CHECKPOINTS: usize = 8;
const DIAGNOSTIC_COMPUTE_TRACE_THREADS: usize = 64;
const DIAGNOSTIC_COMPUTE_TRACE_RECORD_DWORDS: usize = 128;
const DIAGNOSTIC_COMPUTE_TRACE_BYTES: usize = DIAGNOSTIC_COMPUTE_TRACE_CHECKPOINTS
    * DIAGNOSTIC_COMPUTE_TRACE_THREADS
    * DIAGNOSTIC_COMPUTE_TRACE_RECORD_DWORDS
    * size_of::<u32>();

fn diagnostic_compute_state_trace_matches(shader_address: u64) -> bool {
    std::env::var("REMU_TRACE_COMPUTE_STATE_CS")
        .ok()
        .and_then(|value| u64::from_str_radix(value.trim().trim_start_matches("0x"), 16).ok())
        == Some(shader_address)
}

fn diagnostic_compute_state_trace_has_scene_input(
    evaluation: &remu_shader::ShaderEvaluation,
) -> bool {
    let (width, height) = diagnostic_compute_texture_extent().unwrap_or((1_920, 1_080));
    evaluation.image_bindings.iter().any(|binding| {
        binding.pc == 0x220
            && decode_image_descriptor(&binding.resource_descriptor).is_ok_and(|descriptor| {
                descriptor.guest.width == width && descriptor.guest.height == height
            })
    })
}

fn storage_image_requires_exact_preacquire(
    opcode: remu_shader::Opcode,
    has_read_compatible_resident: bool,
) -> bool {
    is_writable_image_opcode(opcode) || !has_read_compatible_resident
}

fn diagnostic_compute_readback_descriptor(
    mut descriptor: super::super::GuestTextureDescriptor,
    native_format: MTLPixelFormat,
) -> Result<super::super::GuestTextureDescriptor, String> {
    if descriptor.pixel_format == native_format {
        return Ok(descriptor);
    }
    let native_bytes_per_pixel =
        metal_uncompressed_bytes_per_pixel(native_format).ok_or_else(|| {
            format!("native format {native_format:?} has no exact uncompressed readback layout")
        })?;
    if native_bytes_per_pixel != descriptor.bytes_per_pixel {
        return Err(format!(
            "native format {native_format:?} uses {native_bytes_per_pixel} bytes per pixel, guest view {:?} uses {}",
            descriptor.pixel_format, descriptor.bytes_per_pixel,
        ));
    }
    descriptor.pixel_format = native_format;
    Ok(descriptor)
}

fn parse_diagnostic_compute_texture_extent(value: &str) -> Option<(u32, u32)> {
    let (width, height) = value.split_once('x')?;
    let width = width.parse().ok()?;
    let height = height.parse().ok()?;
    (width != 0 && height != 0).then_some((width, height))
}

fn diagnostic_compute_texture_extent() -> Option<(u32, u32)> {
    static EXTENT: OnceLock<Option<(u32, u32)>> = OnceLock::new();
    *EXTENT.get_or_init(|| {
        std::env::var("REMU_DUMP_COMPUTE_TEXTURE_EXTENT")
            .ok()
            .and_then(|value| parse_diagnostic_compute_texture_extent(&value))
    })
}

fn diagnostic_compute_texture_dump_limit() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("REMU_DUMP_COMPUTE_TEXTURE_LIMIT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1)
            .clamp(1, 16)
    })
}

impl MetalPresenter {
    #[allow(clippy::too_many_lines)]
    pub(crate) fn execute_compute_dispatch(
        &mut self,
        dispatch: &ComputeDispatch,
        pipeline_key: Option<&ComputePipelineKey>,
        shared_event_plan: Option<&SharedEventDirectComputePlan>,
    ) -> Result<(), String> {
        self.retire_completed_guest_batches();
        let ack_phase_started = self
            .shared_event_ack_phase_timing_enabled()
            .then(Instant::now);
        if dispatch.launch.base_group != [0; 3] {
            return Err(format!(
                "Metal compute does not yet support nonzero base groups {:?}",
                dispatch.launch.base_group
            ));
        }
        let snapshot = dispatch
            .shader
            .as_ref()
            .ok_or("dispatch has no decoded compute shader")?;
        let evaluation = dispatch
            .shader_evaluation
            .as_ref()
            .ok_or("dispatch has no evaluated compute resources")?;
        {
            let store_ops = snapshot
                .program
                .instructions
                .iter()
                .filter(|instruction| is_writable_image_opcode(instruction.opcode))
                .count();
            let three_d = matches!(dispatch.launch.threads_per_group, [_, _, z] if z > 1)
                || matches!(dispatch.launch.group_count, [_, _, z] if z > 1);
            if store_ops > 0 || three_d {
                static LOGGED: AtomicUsize = AtomicUsize::new(0);
                if LOGGED.fetch_add(1, Ordering::Relaxed) < 12 {
                    let bindings = evaluation
                        .image_bindings
                        .iter()
                        .map(|binding| {
                            let decoded = decode_image_descriptor(&binding.resource_descriptor)
                                .map(|descriptor| {
                                    format!(
                                        "type={} {}x{}x{} {:#x} {:?}",
                                        descriptor.texture_type,
                                        descriptor.guest.width,
                                        descriptor.guest.height,
                                        descriptor.layer_count,
                                        descriptor.guest.address,
                                        descriptor.guest.pixel_format
                                    )
                                })
                                .unwrap_or_else(|error| error);
                            format!(
                                "pc={:#x} op={} null={} {decoded}",
                                binding.pc,
                                binding.opcode,
                                image_resource_descriptor_is_null(&binding.resource_descriptor),
                            )
                        })
                        .collect::<Vec<_>>();
                    let globals = evaluation
                        .global_memory_bindings
                        .iter()
                        .enumerate()
                        .map(|(index, binding)| {
                            let words = binding
                                .data
                                .chunks_exact(4)
                                .take(8)
                                .map(|chunk| {
                                    u32::from_le_bytes(chunk.try_into().expect("dword"))
                                })
                                .collect::<Vec<_>>();
                            format!(
                                "b{index} {:#x}/{}B wr={} data={words:08x?}",
                                binding.base_address,
                                binding.byte_count,
                                binding.writable,
                            )
                        })
                        .collect::<Vec<_>>();
                    eprintln!(
                        "[REMu][METAL][COMPUTE-IMAGE] seq={} cs={:#x} grid={:?} local={:?} store_ops={store_ops} insns={} bindings={bindings:?} globals={globals:?}",
                        dispatch.sequence,
                        dispatch.shader_address,
                        dispatch.launch.group_count,
                        dispatch.launch.threads_per_group,
                        snapshot.program.instructions.len(),
                    );
                }
            }
        }
        let device = self
            .device
            .as_ref()
            .ok_or("Metal device is unavailable")?
            .clone();
        let queue = self.active_guest_command_queue()?;

        if remu_memory::debug_log_enabled() {
            let mut writes = String::new();
            for binding in &evaluation.global_memory_bindings {
                if binding.writable {
                    use std::fmt::Write as _;
                    let _ = write!(
                        writes,
                        " gw={:#x}+{:#x}",
                        binding.base_address, binding.byte_count
                    );
                }
            }
            for binding in &evaluation.image_bindings {
                if binding.opcode.starts_with("ImageStore")
                    || binding.opcode.starts_with("ImageAtomic")
                {
                    use std::fmt::Write as _;
                    let address = u64::from(binding.resource_descriptor.first().copied().
                        unwrap_or(0))
                        << 8;
                    let _ = write!(writes, " iw={address:#x}");
                }
            }
            remu_memory::debug_log!(
                "compute",
                "seq={} program={:#x} groups={:?} wave{}{}",
                dispatch.sequence,
                dispatch.shader_address,
                dispatch.launch.group_count,
                dispatch.launch.wave_lane_count,
                writes
            );
        }
        let pipeline_key = pipeline_key.ok_or("dispatch has no immutable pipeline-key snapshot")?;
        if self.failed_compute_pipelines.contains(pipeline_key) {
            return Err("compute pipeline previously failed to compile".to_owned());
        }
        let compiled = self
            .compiled_compute_pipelines
            .get(pipeline_key)
            .ok_or("compute pipeline is not ready")?;
        let pipeline = compiled.pipeline.clone();
        let function = compiled.function.clone();
        let bindings = compiled.shader.bindings.clone();
        let msl_bytes = compiled.shader.source.len();
        let dispatcher = compiled.shader.source.contains("while (active)");
        let execution_width = u32::try_from(pipeline.thread_execution_width())
            .map_err(|_| "Metal compute execution width does not fit u32")?;
        let dynamic_lds = bindings.threadgroup_memory_bytes;
        let wave64_split_safe = compute_program_can_split_wave64(&snapshot.program);
        if dispatch.launch.wave_lane_count == 64 {
            log_wave64_hazard_census(&snapshot.program, "compute");
        }
        validate_compute_launch_for_metal(
            dispatch.launch,
            execution_width,
            wave64_split_safe,
            pipeline.max_total_threads_per_threadgroup(),
            device.max_threadgroup_memory_length(),
            pipeline.static_threadgroup_memory_length(),
            dynamic_lds,
        )
        .map_err(|error| {
            if dispatch.launch.wave_lane_count == 64 && execution_width == 32 && !wave64_split_safe
            {
                format!(
                    "{error}; wave-sensitive instructions [{}]",
                    wave64_split_blockers(&snapshot.program).join(",")
                )
            } else {
                error
            }
        })?;

        // Read-only sampled descriptors also participate in render-graph aliasing. Resolve an
        // exact stale tenant before sampled-cache upload can decode obsolete guest RAM.
        for binding in evaluation
            .image_bindings
            .iter()
            .filter(|binding| !image_resource_descriptor_is_null(&binding.resource_descriptor))
        {
            if let Ok(descriptor) = decode_image_descriptor(&binding.resource_descriptor)
                && self
                    .resident_guest_textures
                    .get(&descriptor.guest)
                    .is_some_and(|resident| {
                        matches!(
                            resident.state,
                            ResidencyState::Stale(StaleSource::Allocation { .. })
                        )
                    })
            {
                self.rematerialise_stale_resident_for_read(descriptor.guest)?;
            }
        }

        for binding in evaluation
            .image_bindings
            .iter()
            .filter(|binding| is_storage_image_opcode(binding.opcode))
            .filter(|binding| !image_resource_descriptor_is_null(&binding.resource_descriptor))
        {
            let descriptor = decode_image_descriptor(&binding.resource_descriptor)
                .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
            // ImageLoad is read-only and Metal can read a GPU-current RGBA/BGRA resident through
            // a component-swizzled view or a native depth surface through its R32/R16 Z view.
            // Forcing an exact descriptor here would first write that resident back to guest
            // memory, retire it, and reconstruct an exact-key texture before upload selection gets
            // a chance to preserve it. Writable storage images still require the conservative
            // exact resident.
            let has_read_compatible_resident = !is_writable_image_opcode(binding.opcode)
                && self.resident_guest_textures.values().any(|resident| {
                    resident_color_view_sample_is_eligible(resident, descriptor, false)
                        || resident_same_image_sample_is_eligible(resident, descriptor, false)
                        || resident_offset_alias_sample_is_eligible(resident, descriptor, false)
                        || resident_volume_sample_is_eligible(resident, descriptor, false)
                        || resident_depth_sample_is_eligible(resident, descriptor)
                });
            if !is_writable_image_opcode(binding.opcode)
                && !has_read_compatible_resident
                && trace_target_sampling()
            {
                let overlapping = descriptor
                    .span
                    .iter()
                    .flat_map(|span| {
                        self.resident_guest_textures
                            .overlapping(span, SpanQueryOrder::ResidentFirst)
                    })
                    .filter_map(|alias| self.resident_guest_textures.get(&alias))
                    .filter(|resident| resident.gpu_written())
                    .map(|resident| {
                        (
                            resident.descriptor.address,
                            resident.descriptor.width,
                            resident.descriptor.height,
                            resident.descriptor.pitch,
                            resident.descriptor.pixel_format,
                            resident.descriptor.tile_mode,
                            resident.mip_level_count,
                            resident.content_version,
                        )
                    })
                    .collect::<Vec<_>>();
                if !overlapping.is_empty() {
                    static LOGGED_PREACQUIRE_MISMATCHES: AtomicUsize = AtomicUsize::new(0);
                    let sequence = LOGGED_PREACQUIRE_MISMATCHES.fetch_add(1, Ordering::Relaxed) + 1;
                    if sequence <= 32 {
                        eprintln!(
                            "[REMu][METAL][STORAGE-PREACQUIRE] sequence={sequence} pc={:#x} sampled={:#x}/{}x{}/p{}/{:?}/tile{}/mips{}/type{}/layers{} overlapping_gpu={overlapping:?}",
                            binding.pc,
                            descriptor.guest.address,
                            descriptor.guest.width,
                            descriptor.guest.height,
                            descriptor.guest.pitch,
                            descriptor.guest.pixel_format,
                            descriptor.guest.tile_mode,
                            descriptor.mip_level_count,
                            descriptor.texture_type,
                            descriptor.layer_count,
                        );
                    }
                }
            }
            if descriptor.texture_type == 9 {
                if storage_image_requires_mip_hierarchy(binding, descriptor) {
                    self.ensure_mipmapped_storage_texture(&device, descriptor)?;
                } else if storage_image_requires_exact_preacquire(
                    binding.opcode,
                    has_read_compatible_resident,
                ) {
                    self.ensure_resident_texture_with_initial_contents(
                        &device,
                        descriptor.guest,
                        descriptor.span,
                        true,
                    )?;
                }
            } else if descriptor.texture_type == 10
                && descriptor.layer_count > 1
                && storage_image_requires_exact_preacquire(
                    binding.opcode,
                    has_read_compatible_resident,
                )
            {
                self.ensure_layered_render_target_texture(
                    &device,
                    super::storage_volume_as_render_target(descriptor),
                    false,
                )?;
            }
        }
        self.synchronize_sampled_aliases(&[evaluation], &[])?;

        self.ensure_active_open_guest_batch()?;
        self.prepare_depth_array_textures(&device, &[evaluation])?;
        let buffers = self.upload_compute_global_buffers(&device, evaluation, shared_event_plan)?;
        // DISPATCH_INDIRECT arguments are read by the GPU at this point in queue order. The
        // runtime's packet-time dimensions are useful diagnostics, but can be stale when an
        // earlier shader generated the argument dwords. Forward the pending Metal writeback
        // buffer and let Metal consume the live values instead.
        let indirect_arguments = match (dispatch.source, dispatch.launch.grid) {
            (
                remu_gpu::ComputeDispatchSource::Indirect { dimensions_address },
                ComputeGrid::Threadgroups(_),
            ) => Some(self.upload_indirect_dispatch_arguments(&device, dimensions_address)?),
            _ => None,
        };
        let global_argument_buffer = create_global_argument_buffer(
            &device,
            &function,
            bindings.global_argument_buffer_index,
            &buffers,
        )?;
        let activity_stamp = ResidentTextureActivityStamp {
            work: dispatch.sequence,
            flip: self.next_flip_sequence,
        };
        let textures = upload_image_textures(
            "compute",
            &device,
            &queue,
            evaluation,
            &mut self.resident_guest_textures,
            &self.depth_array_textures,
            &self.multisample_textures,
            &mut self.sampled_texture_cache,
            &mut self.sampled_texture_write_watches,
            &mut self.fallback_texture_cache,
            &mut self.next_sampled_texture_use,
            activity_stamp,
            &[],
        )?;
        self.begin_compositor_pass_samples();
        for texture in textures.iter().filter(|texture| !texture.writable) {
            self.note_compositor_scanout_sample(texture.descriptor.guest);
            self.note_compositor_pass_sampled_hdr(texture.descriptor.guest);
            self.note_compositor_pass_sampled_volume(
                texture.descriptor.texture_type,
                texture.descriptor.guest.width,
                texture.descriptor.guest.height,
                texture.descriptor.layer_count,
            );
        }
        // Some compute passes bind the same guest image as both a read source and a write
        // destination. Snapshot the producer before this dispatch replaces the writer-map entry;
        // the post-dispatch texture dump otherwise attributes an input to the pass consuming it.
        let diagnostic_dump_input_writers =
            std::env::var_os("REMU_DUMP_COMPUTE_TEXTURE_DIR").map(|_| {
                textures
                    .iter()
                    .map(|texture| {
                        (!texture.writable).then(|| {
                            (
                                diagnostic_texture_writer(texture.descriptor.guest),
                                diagnostic_overlapping_texture_writers(texture.descriptor.guest),
                            )
                        })
                    })
                    .collect::<Vec<_>>()
            });
        // Scene-color CS (e.g. SH 800c7e) may ImageLoad GBuffers that never appear in the
        // sampled-only writer lineage. Dump every image SRD once per filtered CS so missing
        // vs dummy-1x1 vs real half-res color is attributable without fabricating binds.
        // Early occurrences alone miss title-menu remaps: also log a few dispatches at/after
        // REMU_DUMP_COMPUTE_TEXTURE_MIN_DISPATCH (Silent Hill BasePass starts ~seq thousands).
        if dump_msl_compute_filter_matches(dispatch.shader_address) {
            static LOGGED_COMPUTE_IMAGES: AtomicUsize = AtomicUsize::new(0);
            static LOGGED_LATE_COMPUTE_IMAGES: AtomicUsize = AtomicUsize::new(0);
            let early_slot = LOGGED_COMPUTE_IMAGES.fetch_add(1, Ordering::Relaxed);
            let min_dispatch = diagnostic_dump_compute_texture_minimum_dispatch();
            let late_eligible =
                diagnostic_compute_texture_dispatch_is_eligible(dispatch.sequence, min_dispatch)
                    && min_dispatch > 0;
            let late_slot = if late_eligible {
                Some(LOGGED_LATE_COMPUTE_IMAGES.fetch_add(1, Ordering::Relaxed))
            } else {
                None
            };
            let log_early = early_slot < 4;
            let log_late = late_slot.is_some_and(|slot| slot < 2);
            if log_early || log_late {
                let images = evaluation
                    .image_bindings
                    .iter()
                    .map(|binding| {
                        let decoded = decode_image_descriptor(&binding.resource_descriptor).ok();
                        let uploaded = decoded.as_ref().and_then(|descriptor| {
                            textures.iter().find_map(|texture| {
                                (texture.descriptor.guest.address == descriptor.guest.address
                                    && texture.descriptor.guest.width == descriptor.guest.width
                                    && texture.descriptor.guest.height == descriptor.guest.height)
                                    .then_some((
                                        texture.writable,
                                        texture.descriptor.guest.address,
                                        texture.descriptor.guest.width,
                                        texture.descriptor.guest.height,
                                        texture.descriptor.guest.pixel_format,
                                    ))
                            })
                        });
                        (
                            binding.pc,
                            binding.opcode,
                            binding.resource_descriptor.as_slice(),
                            decoded.map(|descriptor| {
                                (
                                    descriptor.guest.address,
                                    descriptor.guest.width,
                                    descriptor.guest.height,
                                    descriptor.guest.pitch,
                                    descriptor.guest.tile_mode,
                                    descriptor.guest.pixel_format,
                                    descriptor.texture_type,
                                )
                            }),
                            uploaded,
                        )
                    })
                    .collect::<Vec<_>>();
                let phase = if log_late && !log_early {
                    "late"
                } else if log_late {
                    "early+late"
                } else {
                    "early"
                };
                eprintln!(
                    "[REMu][METAL][COMPUTE-IMAGES] phase={phase} occurrence={early_slot} late={} seq={} cs={:#x} grid={:?} local={:?} image_count={} uploaded={} images={images:08x?}",
                    late_slot.unwrap_or(usize::MAX),
                    dispatch.sequence,
                    dispatch.shader_address,
                    dispatch.launch.grid,
                    dispatch.launch.threads_per_group,
                    evaluation.image_bindings.len(),
                    textures.len(),
                );
            }
        }
        let (sampler_buffer, samplers) = self.cached_sampler_argument_buffer(
            &device,
            &function,
            bindings.sampler_argument_buffer_index,
            evaluation,
        )?;
        let buffer_indices = (0..buffers.len())
            .map(|index| u64::try_from(index).map_err(|_| "compute buffer index overflow"))
            .collect::<Result<Vec<_>, _>>()?;
        let texture_indices = (0..textures.len())
            .map(|index| u64::try_from(index).map_err(|_| "compute texture index overflow"))
            .collect::<Result<Vec<_>, _>>()?;
        let scalar_metadata =
            shader_scalar_metadata_for_registers(evaluation, &dispatch.shader_registers, 0x240, 0)?;
        let scalar_byte_count = scalar_metadata
            .len()
            .checked_mul(4)
            .and_then(|count| u64::try_from(count).ok())
            .ok_or("compute scalar byte count overflow")?;
        let diagnostic_compute_trace =
            diagnostic_compute_state_trace_matches(dispatch.shader_address).then(|| {
                device.new_buffer(
                    u64::try_from(DIAGNOSTIC_COMPUTE_TRACE_BYTES)
                        .expect("diagnostic compute trace byte count fits u64"),
                    MTLResourceOptions::StorageModeShared,
                )
            });
        if let Some(buffer) = diagnostic_compute_trace.as_ref() {
            // A skipped conditional checkpoint must remain distinguishable from a valid record
            // whose register payload happens to be zero.
            unsafe {
                std::ptr::write_bytes(
                    buffer.contents().cast::<u8>(),
                    0,
                    DIAGNOSTIC_COMPUTE_TRACE_BYTES,
                );
            }
        }
        let mut pending_writebacks = Vec::new();
        collect_pending_global_writebacks(&buffers, evaluation, &mut pending_writebacks)?;
        let mut pending_guest_write_ranges = Vec::new();
        collect_pending_guest_write_ranges(&buffers, evaluation, &mut pending_guest_write_ranges)?;
        let mut produced_global_versions = Vec::new();
        collect_shared_event_global_versions(&buffers, evaluation, &mut produced_global_versions)?;
        let ack_metal_encoding_started = ack_phase_started.map(|_| Instant::now());
        {
            let (batch, statistics) = self.active_open_guest_batch_and_statistics();
            let batch = batch.expect("guest command batch was opened above");
            if batch.end_render_pass() {
                statistics.render_pass_ends_for_compute =
                    statistics.render_pass_ends_for_compute.saturating_add(1);
            }
            batch.work.push(GuestBatchWork::Compute {
                sequence: dispatch.sequence,
                shader: dispatch.shader_address,
                source: dispatch.source,
                grid: dispatch.launch.grid,
                threads_per_group: dispatch.launch.threads_per_group,
                msl_bytes,
                dispatcher,
            });
            append_unique_global_writebacks(&mut batch.global_writebacks, pending_writebacks);
            append_unique_guest_write_ranges(
                &mut batch.guest_write_ranges,
                pending_guest_write_ranges,
            );
            let encoder = batch.ensure_compute_encoder();
            encoder.set_label(&format!("REMu Gen5 compute {:#x}", dispatch.shader_address));
            encoder.set_compute_pipeline_state(&pipeline);
            if let Some(buffer) = diagnostic_compute_trace.as_ref() {
                encoder.set_buffer(30, Some(buffer), 0);
            }
            if let (Some(index), Some(argument_buffer)) = (
                bindings.global_argument_buffer_index,
                global_argument_buffer.as_ref(),
            ) {
                encoder.set_buffer(u64::from(index), Some(argument_buffer), 0);
                for (buffer, binding) in buffers.iter().zip(&evaluation.global_memory_bindings) {
                    let usage = if binding.writable {
                        MTLResourceUsage::Read | MTLResourceUsage::Write
                    } else {
                        MTLResourceUsage::Read
                    };
                    encoder.use_resource(&buffer.buffer, usage);
                }
            } else {
                for (index, buffer) in buffer_indices.iter().zip(&buffers) {
                    encoder.set_buffer(*index, Some(&buffer.buffer), buffer.offset);
                }
            }
            encoder.set_bytes(
                u64::from(bindings.initial_scalar_buffer_index),
                scalar_byte_count,
                scalar_metadata.as_ptr().cast::<c_void>(),
            );
            for (index, texture) in texture_indices.iter().zip(&textures) {
                encoder.set_texture(*index, Some(&texture.texture));
            }
            if let (Some(index), Some(buffer)) = (
                bindings.sampler_argument_buffer_index,
                sampler_buffer.as_ref(),
            ) {
                encoder.set_buffer(u64::from(index), Some(buffer), 0);
            } else {
                for (index, sampler) in samplers.iter().enumerate() {
                    encoder.set_sampler_state(
                        u64::try_from(index).map_err(|_| "compute sampler index overflow")?,
                        Some(sampler),
                    );
                }
            }
            if dynamic_lds != 0 {
                encoder.set_threadgroup_memory_length(0, u64::from(dynamic_lds));
            }
            let local = MTLSize::new(
                u64::from(dispatch.launch.threads_per_group[0]),
                u64::from(dispatch.launch.threads_per_group[1]),
                u64::from(dispatch.launch.threads_per_group[2]),
            );
            match (dispatch.launch.grid, indirect_arguments.as_ref()) {
                (ComputeGrid::Threadgroups(_), Some(arguments)) => {
                    encoder.dispatch_thread_groups_indirect(
                        &arguments.buffer,
                        arguments.offset,
                        local,
                    );
                }
                (ComputeGrid::Threadgroups(groups), None) => encoder.dispatch_thread_groups(
                    MTLSize::new(
                        u64::from(groups[0]),
                        u64::from(groups[1]),
                        u64::from(groups[2]),
                    ),
                    local,
                ),
                (ComputeGrid::Threads(threads), _) => encoder
                    .dispatch_threads(MTLSize::new(threads[0], threads[1], threads[2]), local),
            }
            for version in produced_global_versions {
                if let Some(existing) = batch.global_buffer_versions.iter_mut().find(|existing| {
                    existing.byte_count == version.byte_count
                        && match (existing.mapping_identity, version.mapping_identity) {
                            (Some(existing), Some(version)) => existing == version,
                            (None, None) => existing.guest_address == version.guest_address,
                            _ => false,
                        }
                }) {
                    *existing = version;
                } else {
                    batch.global_buffer_versions.push(version);
                }
            }
        }
        let ack_metal_encoding_finished = ack_phase_started.map(|_| Instant::now());
        // GNM also clears HTILE through small compute fills. Follow shadPS4's conservative
        // metadata heuristic: a writable binding at a registered metadata base rearms the clear
        // unless XOR semantics indicate that this is an update rather than a fill.
        if !snapshot
            .program
            .instructions
            .iter()
            .any(|instruction| instruction.opcode.contains("Xor"))
        {
            for address in evaluation
                .global_memory_bindings
                .iter()
                .filter(|binding| binding.writable)
                .map(|binding| binding.base_address)
            {
                rearm_color_metadata_clear(&mut self.color_metadata_clears, address);
                rearm_depth_metadata_clear(&mut self.depth_metadata_clears, address);
            }
        }
        // Include ImageLoad storage inputs alongside ImageSample*; sampled-only lineage hid
        // GBuffer-sized color that lighting/fog CS threads actually read.
        let diagnostic_inputs = diagnostic_texture_writer_enabled().then(|| {
            evaluation
                .image_bindings
                .iter()
                .filter(|binding| !is_writable_image_opcode(binding.opcode))
                .filter_map(|binding| decode_image_descriptor(&binding.resource_descriptor).ok())
                .map(|descriptor| descriptor.guest)
                .collect::<Vec<_>>()
        });
        record_diagnostic_global_writers(
            "compute",
            dispatch.sequence,
            dispatch.shader_address,
            evaluation,
        );
        for texture in &textures {
            if texture.writable {
                if let Some(sampled) = &diagnostic_inputs {
                    record_diagnostic_texture_writer(
                        texture.descriptor.guest,
                        "compute",
                        dispatch.sequence,
                        dispatch.shader_address,
                        sampled.clone(),
                    );
                }
                self.mark_resident_texture_written_at(
                    texture.descriptor.guest,
                    texture.descriptor.span,
                    texture.selected_resident,
                    WrittenRegion::storage_image(texture.descriptor),
                    activity_stamp,
                );
                // Gen5 presentation targets are ordinary guest images and may be completed by a
                // storage-image compute pass (upscaling, tone mapping, or format conversion)
                // rather than a color attachment. Treat that write exactly like a render-target
                // write so the pending flip is finalized only after the producing submission.
                self.note_pending_flip_target_write(texture.descriptor.guest);
            }
        }
        if let Some(buffer) = diagnostic_compute_trace.as_ref()
            && diagnostic_target_after_mrt_ready()
            && diagnostic_compute_state_trace_has_scene_input(evaluation)
            && diagnostic_compute_texture_dispatch_is_eligible(
                dispatch.sequence,
                diagnostic_dump_compute_texture_minimum_dispatch(),
            )
            && diagnostic_compute_texture_extent().map_or(true, |(width, height)| {
                textures.iter().any(|texture| {
                    texture.descriptor.guest.width == width
                        && texture.descriptor.guest.height == height
                })
            })
            && {
                static COMPUTE_STATE_TRACES: AtomicUsize = AtomicUsize::new(0);
                COMPUTE_STATE_TRACES.fetch_add(1, Ordering::Relaxed) == 0
            }
        {
            self.wait_for_guest_command_batches()?;
            let directory = std::env::var_os("REMU_TRACE_COMPUTE_STATE_DIR")
                .map_or_else(|| std::path::PathBuf::from("/private/tmp"), Into::into);
            std::fs::create_dir_all(&directory)
                .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
            let path = directory.join(format!(
                "compute-state-{}-cs-{:x}.bin",
                dispatch.sequence, dispatch.shader_address
            ));
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    buffer.contents().cast::<u8>(),
                    DIAGNOSTIC_COMPUTE_TRACE_BYTES,
                )
            };
            std::fs::write(&path, bytes)
                .map_err(|error| format!("could not write {}: {error}", path.display()))?;
            let words = unsafe {
                std::slice::from_raw_parts(
                    buffer.contents().cast::<u32>(),
                    DIAGNOSTIC_COMPUTE_TRACE_BYTES / size_of::<u32>(),
                )
            };
            let checkpoints = (0..DIAGNOSTIC_COMPUTE_TRACE_CHECKPOINTS)
                .map(|checkpoint| {
                    let records = &words[checkpoint
                        * DIAGNOSTIC_COMPUTE_TRACE_THREADS
                        * DIAGNOSTIC_COMPUTE_TRACE_RECORD_DWORDS..];
                    let present = (0..DIAGNOSTIC_COMPUTE_TRACE_THREADS)
                        .filter(|&thread| {
                            records[thread * DIAGNOSTIC_COMPUTE_TRACE_RECORD_DWORDS] != 0
                        })
                        .count();
                    let active = (0..DIAGNOSTIC_COMPUTE_TRACE_THREADS)
                        .filter(|&thread| {
                            records[thread * DIAGNOSTIC_COMPUTE_TRACE_RECORD_DWORDS + 1] != 0
                        })
                        .count();
                    let pc = records.first().copied().unwrap_or_default();
                    (pc, present, active)
                })
                .collect::<Vec<_>>();
            eprintln!(
                "[REMu][METAL][COMPUTE-STATE-TRACE] seq={} cs={:#x} checkpoints={checkpoints:?} path={}",
                dispatch.sequence,
                dispatch.shader_address,
                path.display(),
            );
        }
        if let Some(directory) = std::env::var_os("REMU_DUMP_COMPUTE_TEXTURE_DIR")
            && dump_msl_compute_filter_matches(dispatch.shader_address)
            && diagnostic_target_after_mrt_ready()
            && diagnostic_compute_state_trace_has_scene_input(evaluation)
            && diagnostic_compute_texture_dispatch_is_eligible(
                dispatch.sequence,
                diagnostic_dump_compute_texture_minimum_dispatch(),
            )
            && diagnostic_compute_texture_extent().map_or_else(
                || {
                    textures.iter().any(|texture| {
                        texture.descriptor.guest.width >= 1_000
                            && texture.descriptor.guest.height >= 500
                    })
                },
                |(width, height)| {
                    textures.iter().any(|texture| {
                        texture.descriptor.guest.width == width
                            && texture.descriptor.guest.height == height
                    })
                },
            )
            && {
                static MATCHING_COMPUTE_DUMPS: AtomicUsize = AtomicUsize::new(0);
                MATCHING_COMPUTE_DUMPS.fetch_add(1, Ordering::Relaxed)
                    < diagnostic_compute_texture_dump_limit()
            }
        {
            self.wait_for_guest_command_batches()?;
            let directory = Path::new(&directory);
            std::fs::create_dir_all(directory)
                .map_err(|error| format!("could not create {}: {error}", directory.display()))?;
            for (index, texture) in textures.iter().enumerate() {
                let descriptor = texture.descriptor.guest;
                let native_descriptor = diagnostic_compute_readback_descriptor(
                    descriptor,
                    texture.texture.pixel_format(),
                );
                let bytes = native_descriptor.and_then(|native_descriptor| {
                    read_texture_linear(&device, &queue, &texture.texture, native_descriptor)
                        .map(|bytes| (native_descriptor, bytes))
                });
                let (native_descriptor, bytes) = match bytes {
                    Ok(result) => result,
                    Err(error) => {
                        eprintln!(
                            "[REMu][METAL][COMPUTE-TEXTURE-DUMP] seq={} cs={:#x} index={index} address={:#x} guest_format={:?} native_format={:?} error={error}",
                            dispatch.sequence,
                            dispatch.shader_address,
                            descriptor.address,
                            descriptor.pixel_format,
                            texture.texture.pixel_format(),
                        );
                        continue;
                    }
                };
                let access = if texture.writable { "write" } else { "read" };
                let path = directory.join(format!(
                    "compute-{}-cs-{:x}-{access}-{index}-{:x}-{}x{}-{:?}-native-{:?}.bin",
                    dispatch.sequence,
                    dispatch.shader_address,
                    descriptor.address,
                    descriptor.width,
                    descriptor.height,
                    descriptor.pixel_format,
                    native_descriptor.pixel_format,
                ));
                let nonzero_bytes = bytes.iter().filter(|&&byte| byte != 0).count();
                let numeric = diagnostic_texture_numeric_range(&bytes, native_descriptor);
                let (writer, overlapping_writers) = diagnostic_dump_input_writers
                    .as_ref()
                    .and_then(|writers| writers.get(index))
                    .and_then(Option::as_ref)
                    .cloned()
                    .unwrap_or_default();
                eprintln!(
                    "[REMu][METAL][COMPUTE-TEXTURE-DUMP] seq={} cs={:#x} access={access} index={index} address={:#x} extent={}x{} guest_format={:?} native_format={:?} bytes={} nonzero={} numeric={numeric:?} writer={writer:?} overlapping_writers={overlapping_writers:?} path={}",
                    dispatch.sequence,
                    dispatch.shader_address,
                    descriptor.address,
                    descriptor.width,
                    descriptor.height,
                    descriptor.pixel_format,
                    native_descriptor.pixel_format,
                    bytes.len(),
                    nonzero_bytes,
                    path.display(),
                );
                if let Err(error) = std::fs::write(&path, bytes) {
                    eprintln!(
                        "[REMu][METAL][COMPUTE-TEXTURE-DUMP] could not write {}: {error}",
                        path.display()
                    );
                }
            }
        }
        if let (Some(started), Some(encoding_started), Some(encoding_finished)) = (
            ack_phase_started,
            ack_metal_encoding_started,
            ack_metal_encoding_finished,
        ) {
            let finished = Instant::now();
            self.record_shared_event_ack_batch_cpu_phases(
                encoding_started
                    .saturating_duration_since(started)
                    .saturating_add(finished.saturating_duration_since(encoding_finished)),
                encoding_finished.saturating_duration_since(encoding_started),
                true,
            );
        }
        if self
            .active_open_guest_batch()
            .is_some_and(|batch| batch.work.len() >= GUEST_BATCH_WORK_CAPACITY)
        {
            self.flush_guest_command_batch(
                FlushPolicy::Opportunistic,
                GuestBatchFlushReason::WorkCapacity,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        diagnostic_compute_readback_descriptor, parse_diagnostic_compute_texture_extent,
        storage_image_requires_exact_preacquire,
    };
    use crate::presenter::GuestTextureDescriptor;
    use metal::MTLPixelFormat;
    use remu_shader::Opcode;

    #[test]
    fn read_only_storage_load_can_skip_exact_preacquire_for_compatible_gpu_resident() {
        assert!(!storage_image_requires_exact_preacquire(
            Opcode::Known("ImageLoad"),
            true,
        ));
    }

    #[test]
    fn writable_or_unresolved_storage_image_still_requires_exact_preacquire() {
        assert!(storage_image_requires_exact_preacquire(
            Opcode::Known("ImageLoad"),
            false,
        ));
        assert!(storage_image_requires_exact_preacquire(
            Opcode::Known("ImageStore"),
            true,
        ));
        assert!(storage_image_requires_exact_preacquire(
            Opcode::Known("ImageAtomicAdd"),
            true,
        ));
    }

    #[test]
    fn compute_dump_reads_compatible_native_depth_storage_without_dropping_dispatch() {
        let guest = GuestTextureDescriptor {
            address: 0x1234,
            width: 1920,
            height: 1080,
            pitch: 1920,
            tile_mode: 24,
            pixel_format: MTLPixelFormat::R32Float,
            bytes_per_pixel: 4,
        };
        let native = diagnostic_compute_readback_descriptor(guest, MTLPixelFormat::Depth32Float)
            .expect("R32 and Depth32 have the same exact readback layout");
        assert_eq!(native.pixel_format, MTLPixelFormat::Depth32Float);
        assert_eq!(native.bytes_per_pixel, guest.bytes_per_pixel);
    }

    #[test]
    fn compute_dump_extent_filter_requires_two_positive_dimensions() {
        assert_eq!(
            parse_diagnostic_compute_texture_extent("1920x1080"),
            Some((1920, 1080))
        );
        for invalid in ["", "1920", "1920X1080", "0x1080", "1920x0", "ax1080"] {
            assert_eq!(parse_diagnostic_compute_texture_extent(invalid), None);
        }
    }
}
