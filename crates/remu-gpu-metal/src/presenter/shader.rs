// Copyright (C) 2026 REmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

use metal::{Device, MeshRenderPipelineDescriptor, RenderPipelineDescriptor};
use remu_gpu::{ComputeDispatch, GpuEvent, IndexedDraw};
use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{TryRecvError, TrySendError};
use std::time::Instant;

use super::{
    CompiledComputePipeline, CompiledShaderPair, ComputePipelineKey, GraphicsPipelineKey,
    MetalPresenter, NGG_MESH_LDS_BYTES, NggOutputPrimitiveType, PIPELINE_COMPILE_COMPLETED,
    PIPELINE_COMPILE_ENQUEUED, PIPELINE_COMPILE_FAILED, PIPELINE_COMPILE_QUEUE_FULL,
    PersistentRenderPipelineArchive, PipelineCompileCandidate, ProgramIdentity,
    RenderPipelineArchiveState, ShaderCompileRequest, ShaderCompileResult, ShaderFunctionCache,
    ShaderFunctionCacheEntry, ShaderFunctionStageCache, diagnostic_disable_blending,
    dump_msl_compute_filter_matches, dump_msl_instruction_filter_matches,
    dump_msl_pixel_filter_matches, evaluation_samples_large_linear_video, image_component_class,
    metal_color_write_mask, register_value, speculative_pipeline_compile_in_flight_capacity,
    video_target_binding_summary,
};
mod archive;
mod compile;
#[cfg(test)]
pub(crate) use self::compile::{MIN_VALID_METALLIB_BYTES, metallib_bytes_are_usable};
pub(crate) use self::compile::{
    compile_metal_compute_library, compile_metal_library_with_tools, compile_metal_shader_library,
    dump_compute_shader_msl, dump_shader_pair_ir, dump_shader_pair_msl,
    invalidate_metal_library_cache, metal_cache_root, metal_compile_failure_is_transient,
    offline_metal_compiler,
};
mod program;
pub(crate) use self::program::{
    compute_pipeline_key, compute_program_can_split_wave64, effective_global_memory_binding,
    fragmentless_pixel_evaluation, fragmentless_pixel_program,
    graphics_allocated_vector_register_count, image_resource_descriptor_is_null,
    is_storage_image_opcode, is_writable_image_opcode, missing_vertex_parameter_exports,
    ngg_export_uses_wave64_exec, ngg_memoryless_cover_export_stage, ngg_mesh_export_stage,
    ngg_mesh_stage_for_draw, pixel_interpolation_attribute_mask, pixel_scalars,
    procedural_rectangle_stage_for_draw, program_embeds_global_base, program_uses_lds,
    rectangle_coordinate_parameter_mask, shader_scalar_metadata_for_registers,
    validate_compute_launch_for_metal, vertex_parameter_export_mask, wave64_split_blockers,
};
#[cfg(test)]
pub(crate) use self::program::{
    lift_expanded_vertex_buffer_num_records, ngg_export_uses_general_lds, ngg_mesh_draw_supported,
    procedural_rectangle_export_stage, rectangle_draw_has_incomplete_interface,
    shader_scalar_metadata, truncated_memoryless_ngg_rectangle_cover,
};

impl MetalPresenter {
    pub(super) fn prefetch_pipeline_compilations(&mut self) {
        let speculative_available = speculative_pipeline_compile_in_flight_capacity()
            .saturating_sub(
                self.compiling_shader_pairs
                    .len()
                    .saturating_add(self.compiling_compute_pipelines.len()),
            );
        if speculative_available == 0 {
            return;
        }
        let compiled_shader_pairs = &self.compiled_shader_pairs;
        let compiling_shader_pairs = &self.compiling_shader_pairs;
        let failed_shader_pairs = &self.failed_shader_pairs;
        let compiled_compute_pipelines = &self.compiled_compute_pipelines;
        let compiling_compute_pipelines = &self.compiling_compute_pipelines;
        let failed_compute_pipelines = &self.failed_compute_pipelines;
        let mut seen_graphics = HashSet::new();
        let mut seen_compute = HashSet::new();
        let candidates =
            self.receiver
                .prefetch_pipeline_work(512, speculative_available, |event| match event {
                    GpuEvent::IndexedDraw(draw) => {
                        super::graphics_pipeline_key(draw).is_ok_and(|key| {
                            !compiled_shader_pairs.contains_key(&key)
                                && !compiling_shader_pairs.contains(&key)
                                && !failed_shader_pairs.contains(&key)
                                && seen_graphics.insert(key)
                        })
                    }
                    GpuEvent::ComputeDispatch(dispatch) => {
                        let (Some(snapshot), Some(evaluation)) = (
                            dispatch.shader.as_ref(),
                            dispatch.shader_evaluation.as_ref(),
                        ) else {
                            return false;
                        };
                        let key = compute_pipeline_key(dispatch, evaluation, &snapshot.program);
                        !compiled_compute_pipelines.contains_key(&key)
                            && !compiling_compute_pipelines.contains(&key)
                            && !failed_compute_pipelines.contains(&key)
                            && seen_compute.insert(key)
                    }
                    _ => false,
                });
        for candidate in candidates {
            match candidate {
                PipelineCompileCandidate::Graphics(draw) => {
                    if let Ok(key) = super::graphics_pipeline_key(&draw) {
                        self.compile_draw_shaders(&draw, &key);
                    }
                }
                PipelineCompileCandidate::Compute(dispatch) => {
                    if let (Some(snapshot), Some(evaluation)) = (
                        dispatch.shader.as_ref(),
                        dispatch.shader_evaluation.as_ref(),
                    ) {
                        let key = compute_pipeline_key(&dispatch, evaluation, &snapshot.program);
                        self.compile_compute_shader(&dispatch, &key);
                    }
                }
            }
        }
    }

    pub(super) fn compile_draw_shaders(&mut self, draw: &IndexedDraw, key: &GraphicsPipelineKey) {
        if self.compiled_shader_pairs.contains_key(key)
            || self.compiling_shader_pairs.contains(key)
            || self.failed_shader_pairs.contains(key)
        {
            return;
        }
        let request = ShaderCompileRequest::Graphics {
            key: key.clone(),
            draw: draw.clone(),
        };
        match self.shader_compile_sender.try_send(request) {
            Ok(()) => {
                PIPELINE_COMPILE_ENQUEUED.fetch_add(1, Ordering::Relaxed);
                self.compiling_shader_pairs.insert(key.clone());
            }
            Err(TrySendError::Full(_)) => {
                PIPELINE_COMPILE_QUEUE_FULL.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Disconnected(_)) => {
                PIPELINE_COMPILE_FAILED.fetch_add(1, Ordering::Relaxed);
                self.failed_shader_pairs.insert(key.clone());
                eprintln!(
                    "[REMu][METAL] Gen5 shader pair es={:#x} ps={:#x} failed: compiler worker stopped",
                    key.export_shader_address, key.pixel_shader_address
                );
            }
        }
    }

    pub(super) fn compile_compute_shader(
        &mut self,
        dispatch: &ComputeDispatch,
        key: &ComputePipelineKey,
    ) {
        if self.compiled_compute_pipelines.contains_key(key)
            || self.compiling_compute_pipelines.contains(key)
            || self.failed_compute_pipelines.contains(key)
        {
            return;
        }
        let request = ShaderCompileRequest::Compute {
            key: key.clone(),
            dispatch: dispatch.clone(),
        };
        match self.shader_compile_sender.try_send(request) {
            Ok(()) => {
                PIPELINE_COMPILE_ENQUEUED.fetch_add(1, Ordering::Relaxed);
                self.compiling_compute_pipelines.insert(key.clone());
            }
            Err(TrySendError::Full(_)) => {
                PIPELINE_COMPILE_QUEUE_FULL.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Disconnected(_)) => {
                PIPELINE_COMPILE_FAILED.fetch_add(1, Ordering::Relaxed);
                self.failed_compute_pipelines.insert(key.clone());
                eprintln!(
                    "[REMu][METAL] Gen5 compute cs={:#x} failed: compiler worker stopped",
                    dispatch.shader_address
                );
            }
        }
    }

    pub(super) fn receive_shader_compilations(&mut self) {
        loop {
            let result = match self.shader_compile_receiver.try_recv() {
                Ok(result) => result,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    for key in self.compiling_shader_pairs.drain() {
                        PIPELINE_COMPILE_FAILED.fetch_add(1, Ordering::Relaxed);
                        eprintln!(
                            "[REMu][METAL] Gen5 shader pair es={:#x} ps={:#x} failed: compiler worker stopped",
                            key.export_shader_address, key.pixel_shader_address
                        );
                        self.failed_shader_pairs.insert(key);
                    }
                    for key in self.compiling_compute_pipelines.drain() {
                        PIPELINE_COMPILE_FAILED.fetch_add(1, Ordering::Relaxed);
                        eprintln!(
                            "[REMu][METAL] Gen5 compute program={:#x} failed: compiler worker stopped",
                            key.program_identity.0.address
                        );
                        self.failed_compute_pipelines.insert(key);
                    }
                    break;
                }
            };
            match result {
                ShaderCompileResult::Graphics { key, compiled } => {
                    self.compiling_shader_pairs.remove(&key);
                    match compiled {
                        Ok(compiled) => {
                            PIPELINE_COMPILE_COMPLETED.fetch_add(1, Ordering::Relaxed);
                            eprintln!(
                                "[REMu][METAL] compiled Gen5 shader pair es={:#x} ps={:#x} stage={} MSL={vertex_bytes}/{pixel_bytes} bytes time={:.1}/{:.1}/{:.1}ms lower/function/pipeline reuse={vertex_reuse}/{pixel_reuse} diagnostic={vertex_pattern}/{pixel_color}",
                                key.export_shader_address,
                                key.pixel_shader_address,
                                if compiled.uses_mesh_stage {
                                    "mesh"
                                } else {
                                    "vertex"
                                },
                                compiled.lowering_duration.as_secs_f64() * 1_000.0,
                                compiled.function_duration.as_secs_f64() * 1_000.0,
                                compiled.pipeline_duration.as_secs_f64() * 1_000.0,
                                vertex_bytes = compiled.vertex.source.len(),
                                pixel_bytes = compiled.pixel.source.len(),
                                vertex_reuse = compiled.reused_vertex_function,
                                pixel_reuse = compiled.reused_pixel_function,
                                vertex_pattern =
                                    compiled.vertex.source.contains("remu_diagnostic_vertex"),
                                pixel_color = compiled
                                    .pixel
                                    .source
                                    .contains("float4(1.0f, 0.0f, 1.0f, 1.0f)"),
                            );
                            self.compiled_shader_pairs.insert(key, compiled);
                        }
                        Err(error) => {
                            if metal_compile_failure_is_transient(&error) {
                                eprintln!(
                                    "[REMu][METAL] Gen5 shader pair es={:#x} ps={:#x} compile deferred (transient): {error}",
                                    key.export_shader_address, key.pixel_shader_address
                                );
                            } else {
                                PIPELINE_COMPILE_FAILED.fetch_add(1, Ordering::Relaxed);
                                self.failed_shader_pairs.insert(key.clone());
                                eprintln!(
                                    "[REMu][METAL] Gen5 shader pair es={:#x} ps={:#x} failed: {error}",
                                    key.export_shader_address, key.pixel_shader_address
                                );
                            }
                        }
                    }
                }
                ShaderCompileResult::Compute {
                    key,
                    shader_address,
                    threads_per_group,
                    compiled,
                } => {
                    self.compiling_compute_pipelines.remove(&key);
                    match compiled {
                        Ok(compiled) => {
                            PIPELINE_COMPILE_COMPLETED.fetch_add(1, Ordering::Relaxed);
                            eprintln!(
                                "[REMu][METAL] compiled Gen5 compute cs={shader_address:#x} MSL={} bytes time={:.1}/{:.1}/{:.1}ms lower/function/pipeline local={:?} LDS={} bytes",
                                compiled.shader.source.len(),
                                compiled.lowering_duration.as_secs_f64() * 1_000.0,
                                compiled.function_duration.as_secs_f64() * 1_000.0,
                                compiled.pipeline_duration.as_secs_f64() * 1_000.0,
                                threads_per_group,
                                compiled.shader.bindings.threadgroup_memory_bytes,
                            );
                            self.compiled_compute_pipelines.insert(key, compiled);
                        }
                        Err(error) => {
                            if metal_compile_failure_is_transient(&error) {
                                eprintln!(
                                    "[REMu][METAL] Gen5 compute cs={shader_address:#x} compile deferred (transient): {error}"
                                );
                            } else {
                                PIPELINE_COMPILE_FAILED.fetch_add(1, Ordering::Relaxed);
                                self.failed_compute_pipelines.insert(key);
                                eprintln!(
                                    "[REMu][METAL] Gen5 compute cs={shader_address:#x} failed: {error}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    pub(super) fn compile_compute_pipeline(
        device: &Device,
        dispatch: &ComputeDispatch,
        key: &ComputePipelineKey,
        pipeline_archive: Option<&PersistentRenderPipelineArchive>,
    ) -> Result<CompiledComputePipeline, String> {
        let snapshot = dispatch
            .shader
            .as_ref()
            .ok_or("dispatch has no decoded compute shader")?;
        let evaluation = dispatch
            .shader_evaluation
            .as_ref()
            .ok_or("dispatch has no evaluated compute resources")?;
        let mut options = remu_shader_metal::MetalCompileOptions::compute_wave(
            dispatch.launch.lds_size_bytes,
            dispatch.launch.wave_lane_count,
        );
        options.scratch_size_bytes = dispatch.launch.scratch_size_bytes;
        options.allocated_vector_register_count = key.allocated_vector_register_count;
        let lowering_started = Instant::now();
        let lowered = remu_shader_metal::compile_msl(&snapshot.program, evaluation, &options)
            .map_err(|error| format!("compute lowering: {error}"))?;
        let lowering_duration = lowering_started.elapsed();
        if lowered.stage != remu_shader_metal::MetalShaderStage::Compute {
            return Err("compute lowering returned a non-compute Metal stage".to_owned());
        }
        if let Some(directory) = std::env::var_os("REMU_DUMP_MSL_DIR")
            && dump_msl_compute_filter_matches(dispatch.shader_address)
            && let Err(error) = dump_compute_shader_msl(
                Path::new(&directory),
                dispatch.shader_address,
                &snapshot.program.instructions,
                &lowered.source,
            )
        {
            eprintln!(
                "[REMu][METAL] could not dump compute shader cs={:#x}: {error}",
                dispatch.shader_address
            );
        }
        let function_started = Instant::now();
        let library = compile_metal_compute_library(device, &lowered.source)?;
        let function = library
            .get_function(&lowered.entry_point, None)
            .map_err(|error| format!("Metal compute function: {error}"))?;
        let function_duration = function_started.elapsed();
        let pipeline_started = Instant::now();
        let pipeline = if let Some(archive) = pipeline_archive {
            archive.new_compute_pipeline_state(device, &function)?
        } else {
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| format!("Metal compute pipeline: {error}"))?
        };
        let pipeline_duration = pipeline_started.elapsed();
        Ok(CompiledComputePipeline {
            pipeline,
            function,
            shader: lowered,
            lowering_duration,
            function_duration,
            pipeline_duration,
        })
    }

    pub(super) fn compile_draw_shader_pair(
        device: &Device,
        draw: &IndexedDraw,
        key: &GraphicsPipelineKey,
        function_cache: &ShaderFunctionCache,
        render_pipeline_archive: Option<&PersistentRenderPipelineArchive>,
    ) -> Result<CompiledShaderPair, String> {
        let lowering_started = Instant::now();
        let export = draw
            .export_shader
            .as_ref()
            .ok_or("draw has no decoded export shader")?;
        let export_evaluation = draw
            .export_shader_evaluation
            .as_ref()
            .ok_or("draw has no evaluated export shader resources")?;
        let (pixel_program, pixel_evaluation) = match (
            draw.pixel_shader.as_ref(),
            draw.pixel_shader_evaluation.as_ref(),
        ) {
            (Some(pixel), Some(evaluation)) => (pixel.program.as_ref(), evaluation.as_ref()),
            (None, None) if key.targets.is_empty() && key.depth.is_some() => (
                fragmentless_pixel_program().as_ref(),
                fragmentless_pixel_evaluation(),
            ),
            (None, None) => {
                return Err(
                    "fragmentless draw requires an active depth target and no color targets"
                        .to_owned(),
                );
            }
            (Some(_), None) => {
                return Err("draw has no evaluated pixel shader resources".to_owned());
            }
            (None, Some(_)) => {
                return Err("draw has pixel resources but no decoded pixel shader".to_owned());
            }
        };
        // Emit decoded IR before either stage is lowered. Keeping this ahead of `compile_msl`
        // makes the one-shot dump useful for precisely the shaders that fail translation instead
        // of only recording pairs that already compiled successfully.
        if let Some(directory) = std::env::var_os("REMU_DUMP_MSL_DIR")
            && dump_msl_pixel_filter_matches(draw.pixel_shader_address)
            && dump_msl_instruction_filter_matches(pixel_program.instructions.len())
            && let Err(error) = dump_shader_pair_ir(
                Path::new(&directory),
                draw.export_shader_address,
                draw.pixel_shader_address,
                &export.program.instructions,
                &pixel_program.instructions,
            )
        {
            eprintln!(
                "[REMu][METAL] could not dump shader IR for es={:#x} ps={:#x}: {error}",
                draw.export_shader_address, draw.pixel_shader_address
            );
        }
        if let Some(directory) = std::env::var_os("REMU_DUMP_GRAPHICS_LDS_IR_DIR")
            && (program_uses_lds(&export.program) || program_uses_lds(pixel_program))
            && let Err(error) = dump_shader_pair_ir(
                Path::new(&directory),
                draw.export_shader_address,
                draw.pixel_shader_address,
                &export.program.instructions,
                &pixel_program.instructions,
            )
        {
            eprintln!(
                "[REMu][METAL] could not dump graphics LDS IR for es={:#x} ps={:#x}: {error}",
                draw.export_shader_address, draw.pixel_shader_address
            );
        }
        let pixel_outputs = key
            .targets
            .iter()
            .enumerate()
            .map(|(host_location, target)| remu_shader::PixelOutputBinding {
                guest_slot: target.guest_slot,
                host_location: u32::try_from(host_location)
                    .expect("at most eight color targets fit u32"),
                kind: target.output_kind,
                export_format: target.export_format,
            })
            .collect();
        let mut pixel_options = remu_shader_metal::MetalCompileOptions::pixel(
            pixel_outputs,
            register_value(&draw.context_registers, 0x1b3).unwrap_or_default(),
            register_value(&draw.context_registers, 0x1b4).unwrap_or_default(),
        );
        pixel_options.allocated_vector_register_count = key.pixel_allocated_vector_register_count;
        pixel_options.scratch_size_bytes = key.wave_state.pixel_scratch_bytes_per_lane();
        pixel_options.render_target_array_length = key.render_target_array_length;
        pixel_options.pixel_depth_export = key.pixel_depth_export;
        pixel_options.pixel_input_controls = key
            .pixel_input_controls
            .map(remu_shader_metal::PixelInputControl::from_register);
        pixel_options.diagnostic_force_pixel_color = key.diagnostics.force_pixel_color;
        pixel_options.diagnostic_capture_pixel_output = key.diagnostics.capture_pixel_output;
        let pixel_msl =
            remu_shader_metal::compile_msl(pixel_program, pixel_evaluation, &pixel_options)
                .map_err(|error| format!("pixel lowering: {error}"))?;
        let uses_procedural_rectangle_stage = key.uses_procedural_rectangle_stage;
        if key.render_target_array_length == 1 {
            let exported_parameters = vertex_parameter_export_mask(&export.program);
            let interpolated_attributes = pixel_interpolation_attribute_mask(pixel_program);
            let missing = missing_vertex_parameter_exports(
                exported_parameters,
                interpolated_attributes,
                &key.pixel_input_controls,
            );
            if !missing.is_empty() {
                // Procedural rectangle covers synthesize their UV/parameter interface in the
                // host rectangle vertex stage. Every other missing SPI_PS_INPUT_CNTL → EXP PARAM
                // link has no architectural producer in the export program (compacted-record
                // staging only relocates payloads that EXP already wrote): refuse loudly instead
                // of declaring RemuVsOut slots and leaving them zero.
                let rectangle_covers_missing = uses_procedural_rectangle_stage
                    && rectangle_coordinate_parameter_mask(
                        pixel_program,
                        &key.pixel_input_controls,
                    )
                    .is_some_and(|mask| {
                        missing
                            .iter()
                            .all(|&(_, physical, _)| mask & (1 << physical) != 0)
                    });
                if !rectangle_covers_missing && !uses_procedural_rectangle_stage {
                    return Err(format!(
                        "pixel shader {:#x} interpolates SPI_PS_INPUT_CNTL slots without a matching \
                         EXP PARAM / proven NGG attribute staging from export {:#x} \
                         (missing logical/physical/control={missing:08x?}; exports={exported_parameters:#010x})",
                        draw.pixel_shader_address, draw.export_shader_address,
                    ));
                }
                if !rectangle_covers_missing && uses_procedural_rectangle_stage {
                    // Procedural stage with incomplete ImageSample provenance still synthesizes
                    // every live parameter as UV (see compile path below); that is an explicit
                    // rectangle ABI, not a silent NGG zero-fill.
                }
            }
        }
        let uses_mesh_stage = key.uses_mesh_stage;
        let mut vertex_options = if uses_mesh_stage {
            let limits = key
                .ngg_subgroup_limits
                .ok_or("NGG mesh shader is missing subgroup sizing registers")?;
            if limits
                .output_primitive_type
                .is_some_and(|primitive| primitive != NggOutputPrimitiveType::TriangleStrip)
            {
                return Err(format!(
                    "NGG output primitive {:?} cannot use Metal triangle mesh topology",
                    limits.output_primitive_type
                ));
            }
            let registers = key
                .ngg_registers
                .ok_or("NGG mesh shader is missing its raw SDK sizing registers")?;
            // Q5: derive the Metal interface and proof flag from the live SDK register dwords.
            // Meshlet preparation still uses the more detailed decoded limits above, including
            // the independent input-primitive bound.
            let mut mesh_options = remu_shader_metal::MetalCompileOptions::mesh_from_ngg_registers(
                pixel_msl.interpolated_attribute_count,
                NGG_MESH_LDS_BYTES,
                limits.threadgroup_threads,
                registers,
            );
            // Wave64 NGG exports (Poppy Playtime UI/LUT quads) gate their exports on
            // 64-bit EXEC masks and compute meshlet thread indices as
            // `s3[27:24] * 64 + v_mbcnt`. Splitting the guest wave across two SIMD32
            // halves keeps those indices equal to the Metal thread index; compiling
            // them as wave32 skipped every thread past the first 32 and left their
            // vertices unexported (letter-count white boxes instead of glyphs).
            //
            // `CxVgtShaderStagesEn.GsWaveSize` states the merged stage's architectural width
            // outright. A draw that never programmed the register still falls back to the export
            // program's own 64-bit EXEC use, which is how the width was inferred before the
            // register was decoded.
            if key.wave_state.export_lane_count == Some(64)
                || (key.wave_state.export_lane_count.is_none()
                    && ngg_export_uses_wave64_exec(&export.program))
            {
                let blockers = wave64_split_blockers(&export.program);
                if !blockers.is_empty() {
                    return Err(format!(
                        "wave64 NGG mesh export {:#x} cannot split across Metal SIMD32 groups: {blockers:?}",
                        export.program.address
                    ));
                }
                mesh_options.wave_lane_count = 64;
                // SDK PGM_RSRC1 reports VGPRs in wave32 units and explicitly requires
                // dividing by two for wave64 execution.
                mesh_options.allocated_vector_register_count =
                    key.export_allocated_vector_register_count;
            }
            mesh_options
        } else {
            // A Metal vertex function runs one guest lane per invocation and the split-wave64
            // lowering exists only for threadgroup stages, so a wave64 export can only be lowered
            // here when nothing in it observes the wave width. Anything that does would execute
            // against a 32-lane mask and silently drop the guest wave's upper half.
            if key.wave_state.export_lane_count == Some(64) {
                let blockers = wave64_split_blockers(&export.program);
                if !blockers.is_empty() {
                    return Err(format!(
                        "CxVgtShaderStagesEn.GsWaveSize selects wave64 for export shader {:#x}, whose wave-width-dependent operations a Metal vertex stage cannot represent: {blockers:?}",
                        export.program.address
                    ));
                }
            }
            remu_shader_metal::MetalCompileOptions::vertex(pixel_msl.interpolated_attribute_count)
        };
        if !uses_mesh_stage {
            vertex_options.allocated_vector_register_count =
                key.export_allocated_vector_register_count;
        }
        vertex_options.scratch_size_bytes = key
            .wave_state
            .scratch_bytes_per_lane(vertex_options.wave_lane_count);
        vertex_options.render_target_array_length = key.render_target_array_length;
        vertex_options.vertex_output_control = key.vertex_output_control;
        vertex_options.diagnostic_force_visible_clip_depth =
            key.diagnostics.force_visible_clip_depth;
        vertex_options.diagnostic_force_vertex_pattern = key.diagnostics.force_vertex_pattern;
        vertex_options.diagnostic_force_vertex_fetch_id = key.diagnostics.force_vertex_fetch_id;
        vertex_options.diagnostic_capture_vertex_position = key.diagnostics.capture_vertex_position;
        let vertex_msl = if uses_mesh_stage {
            remu_shader_metal::compile_msl(&export.program, export_evaluation, &vertex_options)
                .map_err(|error| format!("mesh lowering: {error}"))?
        } else if key.render_target_array_length > 1 {
            remu_shader_metal::compile_layered_rectangle_vertex(
                pixel_msl.interpolated_attribute_count,
                key.render_target_array_length,
            )
        } else if uses_procedural_rectangle_stage {
            let coordinate_parameter_mask =
                rectangle_coordinate_parameter_mask(pixel_program, &key.pixel_input_controls)
                    .unwrap_or_else(|| {
                        // UE fullscreen posts interpolate UV on attribute 0 (often with extra
                        // tint/aux attrs). The old multi-param fallback put UV on param1+ and
                        // unit tint on param0, so 3-attr Poppy/UE post covers sampled garbage
                        // across the letterboxed mid-band. Default every live parameter to UV;
                        // analysis still overrides when ImageSample provenance is complete.
                        let count = pixel_msl.interpolated_attribute_count.min(32);
                        if count == 0 { 0 } else { (1u32 << count) - 1 }
                    });
            remu_shader_metal::compile_procedural_rectangle_vertex(
                pixel_msl.interpolated_attribute_count,
                coordinate_parameter_mask,
            )
        } else {
            remu_shader_metal::compile_msl(&export.program, export_evaluation, &vertex_options)
                .map_err(|error| format!("vertex lowering: {error}"))?
        };
        let lowering_duration = lowering_started.elapsed();
        if let Some(directory) = std::env::var_os("REMU_DUMP_MSL_DIR")
            && dump_msl_pixel_filter_matches(draw.pixel_shader_address)
            && dump_msl_instruction_filter_matches(pixel_program.instructions.len())
        {
            let directory = Path::new(&directory);
            if let Err(error) = dump_shader_pair_msl(
                directory,
                draw.export_shader_address,
                draw.pixel_shader_address,
                &vertex_msl.source,
                &pixel_msl.source,
            ) {
                eprintln!(
                    "[REMu][METAL] could not dump MSL for es={:#x} ps={:#x}: {error}",
                    draw.export_shader_address, draw.pixel_shader_address
                );
            }
        }
        if let Some(directory) = std::env::var_os("REMU_DUMP_VIDEO_MSL_DIR")
            && evaluation_samples_large_linear_video(pixel_evaluation)
        {
            let directory = Path::new(&directory);
            if let Err(error) = dump_shader_pair_msl(
                directory,
                draw.export_shader_address,
                draw.pixel_shader_address,
                &vertex_msl.source,
                &pixel_msl.source,
            ) {
                eprintln!(
                    "[REMu][METAL] could not dump video MSL for es={:#x} ps={:#x}: {error}",
                    draw.export_shader_address, draw.pixel_shader_address
                );
            }
        }
        if let Some(directory) = std::env::var_os("REMU_DUMP_VIDEO_TARGET_MSL_DIR")
            && video_target_binding_summary(pixel_evaluation).is_some()
        {
            let directory = Path::new(&directory);
            if let Err(error) = dump_shader_pair_msl(
                directory,
                draw.export_shader_address,
                draw.pixel_shader_address,
                &vertex_msl.source,
                &pixel_msl.source,
            ) {
                eprintln!(
                    "[REMu][METAL] could not dump video-target MSL for es={:#x} ps={:#x}: {error}",
                    draw.export_shader_address, draw.pixel_shader_address
                );
            }
        }
        // Render-target formats, blending, depth state, and sample count produce distinct Metal
        // pipelines but do not alter the generated stage functions. Unity frequently emits many
        // such variants during scene loading, so compiling the same 70-100 KiB MSL function for
        // each variant stalls the guest for tens of seconds. Retain functions independently from
        // pipeline states and compile each exact MSL source only once on this worker.
        let function_started = Instant::now();
        let (vertex_function, reused_vertex_function) = if uses_mesh_stage {
            function_cache.mesh(device, &vertex_msl)?
        } else {
            function_cache.vertex(device, &vertex_msl)?
        };
        let (pixel_function, reused_pixel_function) = function_cache.pixel(device, &pixel_msl)?;
        let function_duration = function_started.elapsed();
        let make_descriptor = || {
            let descriptor = RenderPipelineDescriptor::new();
            descriptor.set_label(&format!(
                "REMu Gen5 ES {:#x} PS {:#x}",
                draw.export_shader_address, draw.pixel_shader_address
            ));
            descriptor.set_vertex_function(Some(&vertex_function));
            descriptor.set_fragment_function(Some(&pixel_function));
            descriptor.set_raster_sample_count(u64::from(key.sample_count));
            descriptor.set_input_primitive_topology(key.primitive_topology_class);
            for (location, target) in key.targets.iter().enumerate() {
                let attachment = descriptor
                    .color_attachments()
                    .object_at(u64::try_from(location).expect("color location fits u64"))
                    .ok_or_else(|| format!("Metal has no color attachment {location}"))?;
                attachment.set_pixel_format(target.pixel_format);
                attachment.set_write_mask(metal_color_write_mask(target.write_mask));
                let blending_enabled = target.blend.enabled
                    && !pixel_options.diagnostic_force_pixel_color
                    && !diagnostic_disable_blending();
                attachment.set_blending_enabled(blending_enabled);
                if blending_enabled {
                    attachment.set_source_rgb_blend_factor(target.blend.source_rgb);
                    attachment.set_destination_rgb_blend_factor(target.blend.destination_rgb);
                    attachment.set_rgb_blend_operation(target.blend.rgb_operation);
                    attachment.set_source_alpha_blend_factor(target.blend.source_alpha);
                    attachment.set_destination_alpha_blend_factor(target.blend.destination_alpha);
                    attachment.set_alpha_blend_operation(target.blend.alpha_operation);
                }
            }
            if let Some(depth) = key.depth {
                descriptor.set_depth_attachment_pixel_format(depth.pixel_format);
            }
            Ok(descriptor)
        };
        let pipeline_started = Instant::now();
        let pipeline = if uses_mesh_stage {
            let descriptor = MeshRenderPipelineDescriptor::new();
            descriptor.set_label(&format!(
                "REMu Gen5 NGG ES {:#x} PS {:#x}",
                draw.export_shader_address, draw.pixel_shader_address
            ));
            descriptor.set_mesh_function(Some(&vertex_function));
            descriptor.set_fragment_function(Some(&pixel_function));
            descriptor.set_raster_sample_count(u64::from(key.sample_count));
            let limits = key
                .ngg_subgroup_limits
                .expect("mesh pipeline subgroup limits were validated above");
            descriptor
                .set_max_total_threads_per_mesh_threadgroup(u64::from(limits.threadgroup_threads));
            descriptor.set_mesh_threadgroup_size_is_multiple_of_thread_execution_width(
                limits.threadgroup_threads.is_multiple_of(32),
            );
            for (location, target) in key.targets.iter().enumerate() {
                let attachment = descriptor
                    .color_attachments()
                    .object_at(u64::try_from(location).expect("color location fits u64"))
                    .ok_or_else(|| format!("Metal has no color attachment {location}"))?;
                attachment.set_pixel_format(target.pixel_format);
                attachment.set_write_mask(metal_color_write_mask(target.write_mask));
                let blending_enabled = target.blend.enabled
                    && !pixel_options.diagnostic_force_pixel_color
                    && !diagnostic_disable_blending();
                attachment.set_blending_enabled(blending_enabled);
                if blending_enabled {
                    attachment.set_source_rgb_blend_factor(target.blend.source_rgb);
                    attachment.set_destination_rgb_blend_factor(target.blend.destination_rgb);
                    attachment.set_rgb_blend_operation(target.blend.rgb_operation);
                    attachment.set_source_alpha_blend_factor(target.blend.source_alpha);
                    attachment.set_destination_alpha_blend_factor(target.blend.destination_alpha);
                    attachment.set_alpha_blend_operation(target.blend.alpha_operation);
                }
            }
            if let Some(depth) = key.depth {
                descriptor.set_depth_attachment_pixel_format(depth.pixel_format);
            }
            device
                .new_mesh_render_pipeline_state(&descriptor)
                .map_err(|error| format!("Metal mesh render pipeline: {error}"))?
        } else if let Some(archive) = render_pipeline_archive {
            archive.new_render_pipeline_state(device, make_descriptor)?
        } else {
            let descriptor = make_descriptor()?;
            device
                .new_render_pipeline_state(&descriptor)
                .map_err(|error| format!("Metal render pipeline: {error}"))?
        };
        let pipeline_duration = pipeline_started.elapsed();
        Ok(CompiledShaderPair {
            pipeline,
            vertex_function,
            pixel_function,
            vertex: vertex_msl,
            pixel: pixel_msl,
            reused_vertex_function,
            reused_pixel_function,
            lowering_duration,
            function_duration,
            pipeline_duration,
            uses_mesh_stage,
            uses_procedural_rectangle_stage,
        })
    }
}
pub(super) const SHADER_COMPILE_QUEUE_CAPACITY: usize = 64;
