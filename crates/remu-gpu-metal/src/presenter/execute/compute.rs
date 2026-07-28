// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use super::{
    CompiledComputePipeline, GUEST_BATCH_WORK_CAPACITY, GuestBatchFlushReason, GuestBatchWork,
    MetalPresenter, append_unique_global_writebacks, append_unique_guest_write_ranges,
    collect_pending_global_writebacks, collect_pending_guest_write_ranges,
    compile_metal_compute_library, compute_pipeline_key, compute_program_can_split_wave64,
    create_global_argument_buffer, decode_image_descriptor, diagnostic_texture_writer_enabled,
    dump_compute_shader_msl, dump_msl_compute_filter_matches, image_resource_descriptor_is_null,
    is_storage_image_opcode, record_diagnostic_global_writers, record_diagnostic_texture_writer,
    sampled_image_bindings, shader_scalar_metadata_for_registers, upload_image_textures,
    validate_compute_launch_for_metal, wave64_split_blockers,
};
use metal::{MTLResourceUsage, MTLSize};
use remu_gpu::{ComputeDispatch, ComputeGrid};
use std::ffi::c_void;
use std::path::Path;
use std::time::Instant;

impl MetalPresenter {
    #[allow(clippy::too_many_lines)]
    pub(crate) fn execute_compute_dispatch(
        &mut self,
        dispatch: &ComputeDispatch,
    ) -> Result<(), String> {
        self.retire_completed_guest_batches();
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
        let device = self
            .device
            .as_ref()
            .ok_or("Metal device is unavailable")?
            .clone();
        let queue = self
            .command_queue
            .as_ref()
            .ok_or("Metal command queue is unavailable")?
            .clone();

        let pipeline_key = compute_pipeline_key(dispatch, evaluation, &snapshot.program);
        if self.failed_compute_pipelines.contains(&pipeline_key) {
            return Err("compute pipeline previously failed to compile".to_owned());
        }
        if !self.compiled_compute_pipelines.contains_key(&pipeline_key) {
            let compiled = (|| {
                let options = remu_shader_metal::MetalCompileOptions::compute_wave(
                    dispatch.launch.lds_size_bytes,
                    dispatch.launch.wave_lane_count,
                );
                let lowering_started = Instant::now();
                let lowered =
                    remu_shader_metal::compile_msl(&snapshot.program, evaluation, &options)
                        .map_err(|error| format!("compute lowering: {error}"))?;
                let lowering_duration = lowering_started.elapsed();
                if lowered.stage != remu_shader_metal::MetalShaderStage::Compute {
                    return Err("compute lowering returned a non-compute Metal stage".to_owned());
                }
                let function_started = Instant::now();
                let library = compile_metal_compute_library(&device, &lowered.source)?;
                let function = library
                    .get_function(&lowered.entry_point, None)
                    .map_err(|error| format!("Metal compute function: {error}"))?;
                let function_duration = function_started.elapsed();
                let pipeline_started = Instant::now();
                let pipeline = device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(|error| format!("Metal compute pipeline: {error}"))?;
                let pipeline_duration = pipeline_started.elapsed();
                Ok::<_, String>(CompiledComputePipeline {
                    pipeline,
                    function,
                    shader: lowered,
                    lowering_duration,
                    function_duration,
                    pipeline_duration,
                })
            })();
            let compiled = match compiled {
                Ok(compiled) => compiled,
                Err(error) => {
                    self.failed_compute_pipelines.insert(pipeline_key.clone());
                    return Err(error);
                }
            };
            eprintln!(
                "[REMu][METAL] compiled Gen5 compute cs={:#x} MSL={} bytes time={:.1}/{:.1}/{:.1}ms lower/function/pipeline local={:?} LDS={} bytes",
                dispatch.shader_address,
                compiled.shader.source.len(),
                compiled.lowering_duration.as_secs_f64() * 1_000.0,
                compiled.function_duration.as_secs_f64() * 1_000.0,
                compiled.pipeline_duration.as_secs_f64() * 1_000.0,
                dispatch.launch.threads_per_group,
                compiled.shader.bindings.threadgroup_memory_bytes,
            );
            if let Some(directory) = std::env::var_os("REMU_DUMP_MSL_DIR")
                && dump_msl_compute_filter_matches(dispatch.shader_address)
                && let Err(error) = dump_compute_shader_msl(
                    Path::new(&directory),
                    dispatch.shader_address,
                    &snapshot.program.instructions,
                    &compiled.shader.source,
                )
            {
                eprintln!(
                    "[REMu][METAL] could not dump compute shader cs={:#x}: {error}",
                    dispatch.shader_address
                );
            }
            self.compiled_compute_pipelines
                .insert(pipeline_key.clone(), compiled);
        }
        let compiled = self
            .compiled_compute_pipelines
            .get(&pipeline_key)
            .expect("compute pipeline was cached above");
        let pipeline = compiled.pipeline.clone();
        let function = compiled.function.clone();
        let bindings = compiled.shader.bindings.clone();
        let execution_width = u32::try_from(pipeline.thread_execution_width())
            .map_err(|_| "Metal compute execution width does not fit u32")?;
        let dynamic_lds = bindings.threadgroup_memory_bytes;
        let wave64_split_safe = compute_program_can_split_wave64(&snapshot.program);
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

        for binding in evaluation
            .image_bindings
            .iter()
            .filter(|binding| is_storage_image_opcode(binding.opcode))
            .filter(|binding| !image_resource_descriptor_is_null(&binding.resource_descriptor))
        {
            let descriptor = decode_image_descriptor(&binding.resource_descriptor)
                .map_err(|error| format!("image at PC {:#x}: {error}", binding.pc))?;
            if descriptor.texture_type == 9 {
                self.ensure_resident_texture(&device, descriptor.guest)?;
            }
        }
        self.synchronize_sampled_aliases(&[evaluation])?;

        self.ensure_open_guest_batch(&queue);
        let buffers = self.upload_global_buffers(&device, evaluation)?;
        let global_argument_buffer = create_global_argument_buffer(
            &device,
            &function,
            bindings.global_argument_buffer_index,
            &buffers,
        )?;
        let textures = upload_image_textures(
            &device,
            &queue,
            evaluation,
            &mut self.resident_guest_textures,
            &self.multisample_textures,
            &mut self.sampled_texture_cache,
            &mut self.sampled_texture_write_watches,
            &mut self.fallback_texture_cache,
            &mut self.sampled_texture_admission_order,
            &mut self.next_sampled_texture_admission_id,
        )?;
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
        let mut pending_writebacks = Vec::new();
        collect_pending_global_writebacks(&buffers, evaluation, &mut pending_writebacks)?;
        let mut pending_guest_write_ranges = Vec::new();
        collect_pending_guest_write_ranges(&buffers, evaluation, &mut pending_guest_write_ranges)?;
        {
            let batch = self
                .open_guest_batch
                .as_mut()
                .expect("guest command batch was opened above");
            if batch.end_render_pass() {
                self.statistics.render_pass_ends_for_compute = self
                    .statistics
                    .render_pass_ends_for_compute
                    .saturating_add(1);
            }
            batch.work.push(GuestBatchWork::Compute {
                shader: dispatch.shader_address,
            });
            append_unique_global_writebacks(&mut batch.global_writebacks, pending_writebacks);
            append_unique_guest_write_ranges(
                &mut batch.guest_write_ranges,
                pending_guest_write_ranges,
            );
            let encoder = batch.command_buffer.new_compute_command_encoder();
            encoder.set_label(&format!("REMu Gen5 compute {:#x}", dispatch.shader_address));
            encoder.set_compute_pipeline_state(&pipeline);
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
            match dispatch.launch.grid {
                ComputeGrid::Threadgroups(groups) => encoder.dispatch_thread_groups(
                    MTLSize::new(
                        u64::from(groups[0]),
                        u64::from(groups[1]),
                        u64::from(groups[2]),
                    ),
                    local,
                ),
                ComputeGrid::Threads(threads) => encoder
                    .dispatch_threads(MTLSize::new(threads[0], threads[1], threads[2]), local),
            }
            encoder.end_encoding();
        }
        let diagnostic_sampled = diagnostic_texture_writer_enabled().then(|| {
            sampled_image_bindings(evaluation)
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
                if let Some(sampled) = &diagnostic_sampled {
                    record_diagnostic_texture_writer(
                        texture.descriptor.guest,
                        "compute",
                        dispatch.sequence,
                        dispatch.shader_address,
                        sampled.clone(),
                    );
                }
                self.mark_resident_texture_written(texture.descriptor.guest);
                // Gen5 presentation targets are ordinary guest images and may be completed by a
                // storage-image compute pass (upscaling, tone mapping, or format conversion)
                // rather than a color attachment. Treat that write exactly like a render-target
                // write so the pending flip is finalized only after the producing submission.
                self.note_pending_flip_target_write(texture.descriptor.guest);
            }
        }
        if self
            .open_guest_batch
            .as_ref()
            .is_some_and(|batch| batch.work.len() >= GUEST_BATCH_WORK_CAPACITY)
        {
            self.flush_guest_command_batch(false, GuestBatchFlushReason::WorkCapacity)?;
        }
        Ok(())
    }
}
